from __future__ import annotations

import pytest
import torch

from arti.nn import LearnedPulse, Pulse
from arti.pulse_workspace import FHUPulse


def test_fhu_pulse_keeps_fixed_final_workspace_size() -> None:
    pulse = FHUPulse(k=7, dim=5, exposed=3)
    output, info = pulse(torch.randn(2, 11, 5), return_info=True)

    assert output.shape == (2, 7, 5)
    assert info["compact"].shape == (2, 4, 5)
    assert info["exposed_mask"].shape == (2, 7)
    assert info["exposed_mask"].sum(dim=-1).tolist() == [3, 3]


def test_fhu_pulse_backpropagates_through_fold_half_and_unfold() -> None:
    torch.manual_seed(4)
    pulse = FHUPulse(k=6, dim=4, exposed=2)
    x = torch.randn(3, 9, 4, requires_grad=True)

    pulse(x).square().mean().backward()

    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert pulse.fold.assignment is not None
    assert pulse.fold.assignment.weight.grad is not None
    assert pulse.unfold.exposed_queries.grad is not None
    assert pulse.unfold.layout_score[-1].weight.grad is not None


def test_fhu_pulse_masked_fragments_cannot_change_output() -> None:
    torch.manual_seed(9)
    pulse = FHUPulse(k=6, dim=4).eval()
    x = torch.randn(2, 8, 4)
    mask = torch.tensor(
        [[True, True, True, False, False, False, False, False], [True] * 8]
    )
    changed = x.clone()
    changed[0, 3:] = 1e4

    torch.manual_seed(91)
    first = pulse(x, mask=mask)
    torch.manual_seed(91)
    second = pulse(changed, mask=mask)

    torch.testing.assert_close(first, second)


def test_fhu_pulse_all_invalid_sample_emits_zero_workspace() -> None:
    pulse = FHUPulse(k=5, dim=3)
    x = torch.randn(2, 7, 3)
    mask = torch.tensor([[False] * 7, [True] * 7])
    output, info = pulse(x, mask=mask, return_info=True)

    assert torch.equal(output[0], torch.zeros_like(output[0]))
    assert not info["pulse_mask"][0].any()
    assert info["pulse_mask"][1].all()


def test_fhu_pulse_accepts_external_q_and_independent_ablations() -> None:
    x = torch.randn(2, 8, 4)
    q = torch.rand(2, 8)
    full = FHUPulse(k=6, dim=4)
    no_half = FHUPulse(k=6, dim=4, use_half=False)
    no_condition = FHUPulse(k=6, dim=4, condition_unfold=False)

    assert full(x, q=q).shape == (2, 6, 4)
    assert no_half(x, q=q).shape == (2, 6, 4)
    assert no_condition(x, q=q).shape == (2, 6, 4)


def test_fhu_pulse_calibration_changes_reference_not_feature_scale() -> None:
    torch.manual_seed(12)
    pulse = FHUPulse(k=6, dim=4, calibrate_half=True)
    pulse.half_act.stochastic = False
    sample = torch.randn(2, 8, 4)
    torch.manual_seed(12)
    _, info = pulse(sample, return_info=True)

    reference_scale = info["half_reference_scale"]
    compact = info["compact"]
    half_input = info["half_input"]
    survived = info["survived"]

    torch.testing.assert_close(half_input * reference_scale, compact)
    torch.manual_seed(12)
    torch.testing.assert_close(
        survived,
        pulse.half_act(half_input) * reference_scale,
    )
    assert torch.all(reference_scale > 0)


def test_fhu_pulse_accepts_half_activation_parameters() -> None:
    torch.manual_seed(14)
    default = FHUPulse(k=6, dim=4, calibrate_half=True)
    sharp = FHUPulse(
        k=6,
        dim=4,
        calibrate_half=True,
        half_threshold=0.75,
        half_base=0.25,
        half_scale=0.5,
    )
    sharp.load_state_dict(default.state_dict())
    x = torch.randn(2, 8, 4)

    assert not torch.equal(default(x), sharp(x))
    assert sharp.half_act.threshold == 0.75
    assert sharp.half_act.base == 0.25
    assert sharp.half_act.scale == 0.5


def test_fhu_pulse_transports_guide_with_fold_assignment() -> None:
    torch.manual_seed(15)
    pulse = FHUPulse(
        k=6,
        dim=4,
        exposed=2,
        guide_dim=1,
        unfold_layout_mode="canonical",
    ).eval()
    x = torch.randn(2, 7, 4)
    guide = torch.rand(2, 7, 1)
    output, info = pulse(x, guide=guide, return_info=True)

    assert output.shape == (2, 6, 4)
    assert info["compact_guide"].shape == (2, 4, 1)
    assert torch.isfinite(info["compact_guide"]).all()


def test_fhu_pulse_guide_transport_is_jointly_permutation_invariant() -> None:
    torch.manual_seed(16)
    pulse = FHUPulse(
        k=6,
        dim=4,
        exposed=2,
        guide_dim=1,
        unfold_layout_mode="canonical",
    ).eval()
    x = torch.randn(2, 8, 4)
    guide = torch.rand(2, 8, 1)
    mask = torch.tensor(
        [[True, True, True, True, True, False, False, False], [True] * 8]
    )
    permutation = torch.stack((torch.randperm(8), torch.randperm(8)))
    permuted_x = x.gather(1, permutation.unsqueeze(-1).expand_as(x))
    permuted_guide = guide.gather(
        1, permutation.unsqueeze(-1).expand_as(guide)
    )
    permuted_mask = mask.gather(1, permutation)

    torch.manual_seed(161)
    first, first_info = pulse(x, mask=mask, guide=guide, return_info=True)
    torch.manual_seed(161)
    second, second_info = pulse(
        permuted_x,
        mask=permuted_mask,
        guide=permuted_guide,
        return_info=True,
    )

    torch.testing.assert_close(first_info["compact"], second_info["compact"])
    torch.testing.assert_close(
        first_info["compact_guide"], second_info["compact_guide"]
    )
    torch.testing.assert_close(first, second)


@pytest.mark.parametrize("transport", ["mean", "hard"])
def test_fhu_pulse_guide_transport_modes_preserve_joint_permutation(
    transport: str,
) -> None:
    torch.manual_seed(17)
    pulse = FHUPulse(
        k=6,
        dim=4,
        exposed=2,
        guide_dim=1,
        guide_transport=transport,
        unfold_layout_mode="canonical",
    ).eval()
    x = torch.randn(1, 7, 4)
    guide = torch.rand(1, 7, 1)
    permutation = torch.randperm(7).reshape(1, -1)

    torch.manual_seed(171)
    first, first_info = pulse(x, guide=guide, return_info=True)
    torch.manual_seed(171)
    second, second_info = pulse(
        x.gather(1, permutation.unsqueeze(-1).expand_as(x)),
        guide=guide.gather(1, permutation.unsqueeze(-1).expand_as(guide)),
        return_info=True,
    )

    torch.testing.assert_close(
        first_info["compact_guide"], second_info["compact_guide"]
    )
    torch.testing.assert_close(first, second)


def test_fhu_pulse_hard_guide_transport_does_not_leak_invalid_values() -> None:
    pulse = FHUPulse(
        k=6,
        dim=4,
        exposed=2,
        guide_dim=1,
        guide_transport="hard",
        unfold_layout_mode="canonical",
    )
    x = torch.randn(2, 7, 4)
    mask = torch.tensor([[False] * 7, [True] * 7])
    guide = torch.randn(2, 7, 1)
    _, info = pulse(x, mask=mask, guide=guide, return_info=True)

    assert torch.equal(info["compact_guide"][0], torch.zeros_like(info["compact_guide"][0]))


def test_fhu_pulse_rejects_invalid_guide_contracts() -> None:
    x = torch.randn(2, 7, 4)
    with pytest.raises(ValueError, match="guide_dim is disabled"):
        FHUPulse(k=6, dim=4)(x, guide=torch.randn(2, 7, 1))
    with pytest.raises(ValueError, match="requires guide"):
        FHUPulse(
            k=6,
            dim=4,
            guide_dim=1,
            unfold_layout_mode="canonical",
        )(x)
    with pytest.raises(ValueError, match="guide must have shape"):
        FHUPulse(k=6, dim=4, guide_dim=2)(x, guide=torch.randn(2, 7, 1))
    with pytest.raises(ValueError, match="guide_transport"):
        FHUPulse(k=6, dim=4, guide_transport="unknown")


def test_fhu_pulse_state_dict_round_trip_is_deterministic() -> None:
    torch.manual_seed(13)
    source = FHUPulse(k=6, dim=4).eval()
    target = FHUPulse(k=6, dim=4).eval()
    target.load_state_dict(source.state_dict())
    x = torch.randn(2, 9, 4)

    torch.manual_seed(131)
    source_y = source(x)
    torch.manual_seed(131)
    torch.testing.assert_close(source_y, target(x))
    assert Pulse is LearnedPulse


def test_fhu_pulse_guide_transport_state_dict_round_trip() -> None:
    torch.manual_seed(19)
    source = FHUPulse(
        k=6,
        dim=4,
        exposed=2,
        guide_dim=1,
        guide_transport="hard",
        unfold_layout_mode="canonical",
    ).eval()
    target = FHUPulse(
        k=6,
        dim=4,
        exposed=2,
        guide_dim=1,
        guide_transport="hard",
        unfold_layout_mode="canonical",
    ).eval()
    target.load_state_dict(source.state_dict())
    x = torch.randn(2, 7, 4)
    guide = torch.rand(2, 7, 1)

    torch.manual_seed(191)
    source_y = source(x, guide=guide)
    torch.manual_seed(191)
    torch.testing.assert_close(source_y, target(x, guide=guide))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"k": 1, "dim": 4},
        {"k": 4, "dim": 0},
        {"k": 4, "dim": 4, "exposed": 0},
        {"k": 4, "dim": 4, "exposed": 4},
    ],
)
def test_fhu_pulse_rejects_invalid_shapes(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        FHUPulse(**kwargs)
