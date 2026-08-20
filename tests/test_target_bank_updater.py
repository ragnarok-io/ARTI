from __future__ import annotations

import copy

import pytest
import torch

import arti
from arti.alpha import TargetBankUpdater, WriteRefinePolicy
from arti.recall_refine import RefineBudget, RefineStop


def _inputs(*, requires_grad: bool = False):
    torch.manual_seed(3107)
    trace = torch.randn(2, 5, 4, requires_grad=requires_grad)
    bank = torch.randn(2, 3, 4, requires_grad=requires_grad)
    trace_mask = torch.tensor(
        [[True, True, False, True, False], [True, False, True, True, True]]
    )
    return trace, bank, trace_mask


def _updater(
    *,
    steps: int = 3,
    private_slots: int = 2,
    target_coupling: str = "optional",
) -> TargetBankUpdater:
    torch.manual_seed(9173)
    updater = TargetBankUpdater(
        4,
        3,
        workspace_dim=8,
        private_slots=private_slots,
        policy=WriteRefinePolicy.fixed(steps),
        query_seed=41,
        target_coupling=target_coupling,
    )
    assert updater.gain_head is not None and updater.shift_head is not None
    with torch.no_grad():
        updater.gain_head.weight.normal_(std=0.04)
        updater.shift_head.weight.normal_(std=0.08)
    return updater


def test_component_is_versioned_and_alpha_only() -> None:
    updater = _updater()

    assert arti.component_ref(updater) == "arti/target-bank-updater@1"
    assert updater.transition_semantics == "target_conditioned_locally_affine"
    assert not hasattr(arti, "TargetBankUpdater")
    assert not hasattr(arti.nn, "TargetBankUpdater")


def test_required_target_coupling_is_version_two() -> None:
    updater = _updater(
        private_slots=0,
        target_coupling="required_after_bootstrap",
    )

    assert arti.component_ref(updater) == "arti/target-bank-updater@2"
    assert (
        updater.transition_semantics
        == "target_coupled_locally_affine_after_bootstrap"
    )
    resolved = arti.resolve_component(
        "arti/target-bank-updater@2",
        hidden_dim=4,
        slots=3,
        workspace_dim=8,
        private_slots=0,
    )
    assert arti.component_ref(resolved) == "arti/target-bank-updater@2"


def test_version_two_requires_target_read_after_bootstrap() -> None:
    updater = _updater(
        steps=4,
        private_slots=0,
        target_coupling="required_after_bootstrap",
    )
    trace, bank, trace_mask = _inputs()

    one = updater(
        trace,
        bank,
        trace_mask=trace_mask,
        policy=WriteRefinePolicy.fixed(1),
        _addressable_target=torch.zeros_like(bank),
    )
    blind = updater(
        trace,
        bank,
        trace_mask=trace_mask,
        _addressable_target=torch.zeros_like(bank),
    )

    expected = bank + (one - bank) / 4
    torch.testing.assert_close(blind, expected, rtol=1e-5, atol=1e-6)


def test_version_two_uses_updated_target_during_later_writes() -> None:
    updater = _updater(
        steps=4,
        private_slots=0,
        target_coupling="required_after_bootstrap",
    )
    trace, bank, trace_mask = _inputs()

    dynamic = updater(trace, bank, trace_mask=trace_mask)
    blind = updater(
        trace,
        bank,
        trace_mask=trace_mask,
        _addressable_target=torch.zeros_like(bank),
    )

    assert not torch.allclose(dynamic, blind)


def test_policy_reuses_budget_and_stop_components() -> None:
    policy = WriteRefinePolicy.adaptive(max_steps=8, min_steps=2)

    assert policy.budget == RefineBudget(max_steps=8, min_steps=2)
    assert isinstance(policy.stop, RefineStop)
    assert arti.component_ref(policy) == "arti/write-refine-policy@1"


def test_target_is_an_addressable_partition_not_only_conditioning() -> None:
    updater = _updater(steps=1)
    trace, bank, trace_mask = _inputs()
    shuffled = bank.flip(1)

    baseline, baseline_info = updater(
        trace, bank, trace_mask=trace_mask, return_info=True
    )
    intervention, intervention_info = updater(
        trace,
        bank,
        trace_mask=trace_mask,
        return_info=True,
        _addressable_target=shuffled,
    )

    assert not torch.allclose(
        baseline_info["target_route_weights"],
        intervention_info["target_route_weights"],
    )
    assert not torch.allclose(
        baseline_info["target_read_context"],
        intervention_info["target_read_context"],
    )
    assert not torch.allclose(baseline, intervention)


def test_reusing_first_target_route_is_an_internal_refine_ablation() -> None:
    updater = _updater(steps=4, private_slots=0)
    trace, bank, trace_mask = _inputs()

    _, dynamic = updater(
        trace, bank, trace_mask=trace_mask, return_info=True
    )
    _, frozen = updater(
        trace,
        bank,
        trace_mask=trace_mask,
        return_info=True,
        _reuse_first_target_route=True,
    )

    frozen_routes = frozen["target_route_weights"]
    assert torch.equal(
        frozen_routes[:, 1:],
        frozen_routes[:, :1].expand_as(frozen_routes[:, 1:]),
    )
    dynamic_routes = dynamic["target_route_weights"]
    assert not torch.equal(
        dynamic_routes[:, 1:],
        dynamic_routes[:, :1].expand_as(dynamic_routes[:, 1:]),
    )


def test_each_write_step_readdresses_the_updated_target() -> None:
    updater = _updater(steps=3, private_slots=0)
    trace, bank, trace_mask = _inputs()

    result, info = updater(trace, bank, trace_mask=trace_mask, return_info=True)

    assert result.shape == bank.shape
    assert info["target_route_weights"].shape == (2, 3, 3, 3)
    assert not torch.allclose(
        info["target_route_weights"][:, 0],
        info["target_route_weights"][:, 1],
    )
    assert not torch.allclose(
        info["target_read_context"][:, 0],
        info["target_read_context"][:, 1],
    )


def test_private_and_target_routes_normalize_independently() -> None:
    updater = _updater(steps=2, private_slots=4)
    trace, bank, trace_mask = _inputs()

    _, info = updater(trace, bank, trace_mask=trace_mask, return_info=True)

    torch.testing.assert_close(
        info["private_route_weights"].sum(dim=-1),
        torch.ones_like(info["private_route_weights"][..., 0]),
    )
    torch.testing.assert_close(
        info["target_route_weights"].sum(dim=-1),
        torch.ones_like(info["target_route_weights"][..., 0]),
    )
    torch.testing.assert_close(
        info["partition_gate"].sum(dim=-1),
        torch.ones_like(info["partition_gate"][..., 0]),
    )


def test_reported_target_read_reconstructs_from_real_routes() -> None:
    updater = _updater(steps=1, private_slots=2)
    trace, bank, trace_mask = _inputs()

    _, info = updater(trace, bank, trace_mask=trace_mask, return_info=True)
    values = updater.target_projection(bank)
    reconstructed = torch.einsum(
        "bst,btw->bsw", info["target_route_weights"][:, 0], values
    )

    torch.testing.assert_close(
        reconstructed,
        info["target_read_context"][:, 0],
        rtol=0,
        atol=0,
    )


def test_private_bank_is_not_mutated_by_forward() -> None:
    updater = _updater(steps=4, private_slots=3)
    assert updater.private_bank is not None
    before = updater.private_bank.detach().clone()
    trace, bank, trace_mask = _inputs()

    updater(trace, bank, trace_mask=trace_mask)

    torch.testing.assert_close(updater.private_bank, before, rtol=0, atol=0)


def test_target_only_mode_is_supported() -> None:
    updater = _updater(steps=2, private_slots=0)
    trace, bank, trace_mask = _inputs()

    result, info = updater(trace, bank, trace_mask=trace_mask, return_info=True)

    assert result.shape == bank.shape
    assert info["private_route_weights"].shape[-1] == 0
    assert torch.count_nonzero(info["partition_gate"][..., 0]) == 0
    assert torch.all(info["partition_gate"][..., 1] == 1)


def test_zero_exposure_is_an_exact_no_op() -> None:
    updater = _updater(steps=4)
    trace, bank, trace_mask = _inputs()

    result, info = updater(
        trace,
        bank,
        trace_mask=trace_mask,
        exposure=0.0,
        return_info=True,
    )

    torch.testing.assert_close(result, bank, rtol=0, atol=0)
    assert torch.count_nonzero(info["executed_write_steps"]) == 0
    assert info["target_change_norm"].shape == (2, 0)


def test_steps_partition_exposure_instead_of_repeating_full_change() -> None:
    trace, bank, trace_mask = _inputs()
    one = _updater(steps=1)
    four = _updater(steps=4)
    four.load_state_dict(copy.deepcopy(one.state_dict()))

    once = one(trace, bank, trace_mask=trace_mask)
    refined = four(trace, bank, trace_mask=trace_mask)
    repeated = bank + 4.0 * (once - bank)

    assert not torch.allclose(refined, repeated, rtol=1e-4, atol=1e-5)


def test_masks_exclude_trace_and_target_positions() -> None:
    updater = _updater(steps=3)
    trace, bank, trace_mask = _inputs()
    changed = trace.clone()
    changed[~trace_mask] = 1_000_000.0
    target_mask = torch.tensor([[True, False, True], [False, True, True]])

    expected, info = updater(
        trace,
        bank,
        trace_mask=trace_mask,
        target_mask=target_mask,
        return_info=True,
    )
    actual = updater(
        changed,
        bank,
        trace_mask=trace_mask,
        target_mask=target_mask,
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(actual[~target_mask], bank[~target_mask], rtol=0, atol=0)
    masked_weights = info["target_route_weights"] * (~target_mask)[:, None, None]
    assert torch.count_nonzero(masked_weights) == 0


def test_write_mask_is_independent_from_readable_target_mask() -> None:
    updater = _updater(
        steps=4,
        private_slots=0,
        target_coupling="required_after_bootstrap",
    )
    trace, bank, trace_mask = _inputs()
    write_mask = torch.tensor([[False, False, True], [False, False, True]])

    result = updater(
        trace,
        bank,
        trace_mask=trace_mask,
        write_mask=write_mask,
    )
    changed_bank = bank.clone()
    changed_bank[:, 0] = changed_bank[:, 0] + 3.0
    changed = updater(
        trace,
        changed_bank,
        trace_mask=trace_mask,
        write_mask=write_mask,
    )

    torch.testing.assert_close(result[~write_mask], bank[~write_mask], rtol=0, atol=0)
    torch.testing.assert_close(
        changed[~write_mask], changed_bank[~write_mask], rtol=0, atol=0
    )
    assert not torch.allclose(result[write_mask], changed[write_mask])


def test_gradients_reach_trace_target_and_private_memory() -> None:
    updater = _updater(steps=3, private_slots=2)
    trace, bank, trace_mask = _inputs(requires_grad=True)

    updater(trace, bank, trace_mask=trace_mask).square().mean().backward()

    assert trace.grad is not None and torch.isfinite(trace.grad).all()
    assert bank.grad is not None and torch.isfinite(bank.grad).all()
    assert updater.private_bank is not None
    assert updater.private_bank.grad is not None
    assert torch.isfinite(updater.private_bank.grad).all()


def test_batch_sessions_match_independent_updates() -> None:
    updater = _updater(
        steps=4,
        private_slots=0,
        target_coupling="required_after_bootstrap",
    ).eval()
    trace, bank, trace_mask = _inputs()

    batched = updater(trace, bank, trace_mask=trace_mask)
    independent = torch.stack(
        [
            updater(trace[row], bank[row], trace_mask=trace_mask[row])
            for row in range(trace.shape[0])
        ]
    )

    torch.testing.assert_close(batched, independent, rtol=1e-5, atol=1e-6)


def test_state_dict_round_trip_preserves_output() -> None:
    updater = _updater(steps=3).eval()
    clone = _updater(steps=3).eval()
    clone.load_state_dict(copy.deepcopy(updater.state_dict()), strict=True)
    trace, bank, trace_mask = _inputs()

    expected = updater(trace, bank, trace_mask=trace_mask)
    actual = clone(trace, bank, trace_mask=trace_mask)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_adaptive_stop_is_bounded() -> None:
    policy = WriteRefinePolicy.adaptive(
        max_steps=8,
        min_steps=2,
        absolute_tolerance=1e6,
        relative_tolerance=0.0,
    )
    updater = _updater(steps=1)
    trace, bank, trace_mask = _inputs()

    _, info = updater(
        trace,
        bank,
        trace_mask=trace_mask,
        policy=policy,
        return_info=True,
    )

    torch.testing.assert_close(
        info["executed_write_steps"], torch.full((2,), 2, dtype=torch.long)
    )


def test_stopped_sample_reports_no_unapplied_change() -> None:
    updater = _updater(steps=1)
    trace, bank, trace_mask = _inputs()

    _, info = updater(
        trace,
        bank,
        trace_mask=trace_mask,
        exposure=torch.tensor([0.0, 1.0]),
        policy=WriteRefinePolicy.fixed(4),
        return_info=True,
    )

    assert torch.count_nonzero(info["target_change_norm"][0]) == 0
    assert torch.count_nonzero(info["target_change_norm"][1]) > 0


@pytest.mark.parametrize("steps", [0, -1])
def test_fixed_policy_rejects_non_positive_steps(steps: int) -> None:
    with pytest.raises(ValueError):
        WriteRefinePolicy.fixed(steps)
