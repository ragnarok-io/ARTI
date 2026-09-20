from __future__ import annotations

import copy

import pytest
import torch

import arti
from arti import mechanisms
from benchmarks.run_reversible_topology_selective_compute import (
    ExternalScorePolicy,
    FrozenTensorBlock,
    SelectiveComputePipeline,
    _apply_to_indices,
    _balanced_output_loss,
    _calibrate_priority,
    _counterfactual_final_output_estimator,
    _target_indices,
)


def _fixed_pipeline(
    *, dim: int = 4, active_count: int = 2, use_half: bool = False
) -> SelectiveComputePipeline:
    return SelectiveComputePipeline(
        policy=mechanisms.FixedTopologyPolicy(order=[2, 0, 3, 1]),
        block=FrozenTensorBlock(dim, seed=41),
        active_count=active_count,
        use_half=use_half,
    )


def test_pipeline_calls_block_once_with_only_k_instances_and_restores_order() -> None:
    pipeline = _fixed_pipeline()
    x = torch.randn(3, 4, 4)
    mask = torch.ones(3, 4, dtype=torch.bool)
    pipeline.block.clear_observation()

    output, state = pipeline(x, mask, return_state=True)

    assert pipeline.block.calls == 1
    assert pipeline.block.last_input_shape == (3, 2, 4)
    assert torch.equal(state.record.active_index, torch.tensor([[2, 0]]).expand(3, -1))
    folded_index = state.record.folded_index.unsqueeze(-1).expand(3, 2, 4)
    assert torch.equal(
        torch.gather(output, -2, folded_index),
        torch.gather(x, -2, folded_index),
    )
    assert torch.equal(pipeline.unfold(state).value, x)


def test_frozen_block_has_equal_nonzero_intervention_energy_per_instance() -> None:
    block = FrozenTensorBlock(5, seed=42)
    x = torch.randn(3, 7, 5)

    delta_norm = (block(x) - x).norm(dim=-1)

    assert torch.all(delta_norm > 0)
    torch.testing.assert_close(
        delta_norm,
        torch.full_like(delta_norm, delta_norm[0, 0]),
        rtol=1e-6,
        atol=1e-7,
    )


def test_invalid_active_slots_are_identity_bypassed_when_valid_count_is_below_k() -> None:
    pipeline = SelectiveComputePipeline(
        policy=mechanisms.FixedTopologyPolicy(),
        block=FrozenTensorBlock(3, seed=43),
        active_count=3,
    )
    x = torch.randn(2, 4, 3)
    mask = torch.tensor([[True, False, False, False], [True, True, False, False]])

    output = pipeline(x, mask)

    assert isinstance(output, torch.Tensor)
    assert torch.equal(output[~mask], x[~mask])
    assert not torch.equal(output[mask], x[mask])


def test_real_block_pipeline_input_gradient_matches_direct_hard_lineage() -> None:
    torch.manual_seed(47)
    pipeline = _fixed_pipeline()
    direct_block = copy.deepcopy(pipeline.block)
    x = torch.randn(2, 4, 4, requires_grad=True)
    direct_x = x.detach().clone().requires_grad_()
    mask = torch.ones(2, 4, dtype=torch.bool)
    probe = torch.randn_like(x)

    output = pipeline(x, mask)
    assert isinstance(output, torch.Tensor)
    (output * probe).sum().backward()
    indices = torch.tensor([[2, 0], [2, 0]])
    direct_output = _apply_to_indices(direct_x, indices, direct_block, None)
    (direct_output * probe).sum().backward()

    torch.testing.assert_close(x.grad, direct_x.grad, rtol=1e-6, atol=1e-7)


def test_bank_receives_surrogate_gradient_from_final_tensor_loss() -> None:
    torch.manual_seed(53)
    bank = mechanisms.TopologyOperandBank(12, 4, factor_dim=2, bank_id="phase-c-grad")
    policy = mechanisms.BankFormulaTopologyPolicy(
        dim=6,
        key_dim=4,
        banks=[bank],
        formula=mechanisms.TopologyPriorityFormula(2, mode="gated_priority"),
    )
    pipeline = SelectiveComputePipeline(
        policy=policy,
        block=FrozenTensorBlock(6, seed=59),
        active_count=2,
    )
    target_block = copy.deepcopy(pipeline.block)
    x = torch.randn(4, 6, 6)
    mask = torch.ones(4, 6, dtype=torch.bool)
    target_indices = _target_indices(x[..., 0], mask, 2)
    target = _apply_to_indices(x, target_indices, target_block, None)

    prediction = pipeline(x, mask)
    assert isinstance(prediction, torch.Tensor)
    loss = _balanced_output_loss(prediction, target, x, mask)
    loss.backward()

    assert bank.values.grad is not None
    assert torch.isfinite(bank.values.grad).all()
    assert float(bank.values.grad.abs().sum()) > 0
    assert all(parameter.grad is None for parameter in pipeline.block.parameters())


def test_balanced_output_loss_excludes_invalid_padding_positions() -> None:
    prediction = torch.zeros(1, 4, 2)
    target = prediction.clone()
    mask = torch.tensor([[True, True, False, False]])
    target[:, 1] = 1.0

    source = prediction.clone()
    baseline = _balanced_output_loss(prediction, target, source, mask)
    prediction[:, 2:] = 10_000.0
    target[:, 2:] = -10_000.0
    with_padding_noise = _balanced_output_loss(
        prediction, target, source, mask
    )

    torch.testing.assert_close(with_padding_noise, baseline)


def test_counterfactual_estimator_preserves_hard_loss_value_and_routes_gradient() -> None:
    torch.manual_seed(54)
    policy = ExternalScorePolicy(torch.randn(4, 6))
    block = FrozenTensorBlock(3, seed=55)
    pipeline = SelectiveComputePipeline(
        policy=policy, block=copy.deepcopy(block), active_count=2
    )
    x = torch.randn(4, 6, 3, requires_grad=True)
    mask = torch.ones(4, 6, dtype=torch.bool)
    mask[:, -1] = False
    target_indices = _target_indices(x[..., 0], mask, 2)
    with torch.no_grad():
        target = _apply_to_indices(x, target_indices, block, None)
    prediction = pipeline(x, mask)
    assert isinstance(prediction, torch.Tensor)
    hard_loss = _balanced_output_loss(prediction, target, x, mask)
    estimator = _counterfactual_final_output_estimator(
        policy=policy,
        x=x,
        mask=mask,
        target=target,
        block=block,
        active_count=2,
    )
    training_loss = hard_loss.detach() + estimator - estimator.detach()

    torch.testing.assert_close(training_loss, hard_loss)
    estimator_gradient = torch.autograd.grad(
        estimator, policy.scores, retain_graph=True
    )[0]
    training_gradient, input_gradient = torch.autograd.grad(
        training_loss,
        (policy.scores, x),
        allow_unused=True,
    )
    torch.testing.assert_close(training_gradient, estimator_gradient)
    assert input_gradient is None
    assert torch.isfinite(training_gradient).all()
    assert float(training_gradient.abs().sum()) > 0
    assert torch.equal(
        training_gradient[~mask], torch.zeros_like(training_gradient[~mask])
    )


def test_concat_calibration_ignores_per_sample_priority_offsets() -> None:
    torch.manual_seed(56)
    x = torch.randn(5, 7, 3)
    mask = torch.ones(5, 7, dtype=torch.bool)
    utility = torch.randn(5, 7)
    scores = 2.0 * utility + torch.randn(5, 1)
    shifted_scores = scores + torch.tensor([[100.0], [-50.0], [3.0], [0.0], [81.0]])

    baseline = _calibrate_priority(
        ExternalScorePolicy(scores), x, mask, utility
    )
    shifted = _calibrate_priority(
        ExternalScorePolicy(shifted_scores), x, mask, utility
    )

    assert shifted["correlation"] == pytest.approx(baseline["correlation"], abs=1e-6)
    assert shifted["weight"] == pytest.approx(baseline["weight"], abs=1e-6)


def test_wrong_block_shape_is_rejected_by_folded_state_replace() -> None:
    class WrongShapeBlock(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x[..., :-1]

    pipeline = _fixed_pipeline()
    pipeline.block = WrongShapeBlock()

    with pytest.raises(ValueError, match="replacement active payload must preserve shape"):
        pipeline(torch.randn(1, 4, 4), torch.ones(1, 4, dtype=torch.bool))


def test_optional_half_is_explicit_in_component_graph_and_absent_when_disabled() -> None:
    disabled = _fixed_pipeline(use_half=False)
    enabled = _fixed_pipeline(use_half=True)

    disabled_refs = {
        node["ref"] for node in arti.component_provenance(disabled)["components"]
    }
    enabled_refs = {
        node["ref"] for node in arti.component_provenance(enabled)["components"]
    }

    assert "arti/half@1" not in disabled_refs
    assert "arti/half@1" in enabled_refs
    assert not any("half" in name for name in disabled.state_dict())

    graph = arti.component_graph(enabled)
    nodes_by_path = {
        mount["path"]: next(
            node for node in graph["nodes"] if node["id"] == mount["node"]
        )
        for mount in graph["mounts"]
    }
    assert nodes_by_path["$"]["variant"] == "opaque"
    assert nodes_by_path["$.block"]["variant"] == "opaque"
    assert nodes_by_path["$.block.layers"]["api"].endswith(".Linear")
    assert nodes_by_path["$.fold.topology"]["ref"] == "arti/reversible-topology@1"
    assert (
        nodes_by_path["$.unfold.inverse_contract"]["ref"]
        == "arti/inverse-topology-contract@1"
    )
    assert not any(path.startswith("$.unfold.") and "policy" in path for path in nodes_by_path)


def test_deterministic_pipeline_arti_st_round_trip(tmp_path) -> None:
    source = _fixed_pipeline(use_half=True).eval()
    target = _fixed_pipeline(use_half=True).eval()
    x = torch.randn(2, 4, 4)
    mask = torch.ones(2, 4, dtype=torch.bool)
    expected = source(x, mask)
    saved = arti.save(source, tmp_path / "selective-compute.arti.st")
    with torch.no_grad():
        target.block.layers.weight.fill_(7.0)
        target.block.layers.bias.fill_(-7.0)

    loaded = arti.load(saved.weights_path, model=target)
    actual = loaded.model(x, mask)

    assert isinstance(expected, torch.Tensor)
    assert isinstance(actual, torch.Tensor)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(target.block.layers.weight, source.block.layers.weight)
    torch.testing.assert_close(target.block.layers.bias, source.block.layers.bias)


def test_phase_c_pipeline_stays_private_while_pulse_stage_has_alpha_identity() -> None:
    references = {entry["ref"] for entry in arti.component_catalog()}

    assert "arti/selective-compute@1" in references
    assert not any("selective-compute-pipeline" in reference for reference in references)
    with pytest.raises(arti.UnknownComponentError):
        arti.component_ref(_fixed_pipeline())
    with pytest.raises(arti.UnknownComponentError):
        arti.component_ref(FrozenTensorBlock(4, seed=1))
    assert not hasattr(arti, "SelectiveComputePipeline")
    assert not hasattr(arti.nn, "SelectiveComputePipeline")
    assert not hasattr(mechanisms, "SelectiveComputePipeline")
    assert not hasattr(arti, "FrozenTensorBlock")
    assert not hasattr(arti.nn, "FrozenTensorBlock")
    assert not hasattr(mechanisms, "FrozenTensorBlock")
