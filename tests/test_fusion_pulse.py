from __future__ import annotations

import pytest
import torch

import arti
import arti.torch as arti_torch
from arti.nn import FusionPulse


def test_fusion_pulse_stacked_and_concat_paths_are_equivalent() -> None:
    torch.manual_seed(31)
    fusion = FusionPulse(k=5, dim=6).eval()
    sources = tuple(torch.randn(2, 4, 6) for _ in range(3))
    masks = tuple(torch.rand(2, 4) > 0.2 for _ in range(3))

    stacked = fusion(torch.stack(sources, dim=1), mask=torch.stack(masks, dim=1))
    concatenated = fusion.concat(*sources, masks=masks)

    assert stacked.shape == (2, 5, 6)
    torch.testing.assert_close(stacked, concatenated)


def test_fusion_pulse_accepts_variable_lengths_and_reports_layout() -> None:
    fusion = FusionPulse(k=4, dim=5)
    sources = (
        torch.randn(2, 2, 5),
        torch.randn(2, 5, 5),
        torch.randn(2, 3, 5),
    )
    masks = (
        torch.tensor([[True, True], [True, False]]),
        torch.ones(2, 5, dtype=torch.bool),
        torch.tensor([[True, True, False], [True, False, False]]),
    )

    output, info = fusion.concat(*sources, masks=masks, return_info=True)

    assert output.shape == (2, 4, 5)
    assert info["survival"].shape == (2, 10, 5)
    assert info["input_mask"].shape == (2, 10)
    assert info["pulse_mask"].shape == (2, 4)
    assert info["source_index"].tolist() == [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
    invalid_survival = info["survival"][~info["input_mask"]]
    assert torch.equal(invalid_survival, torch.zeros_like(invalid_survival))


def test_fusion_pulse_structural_loss_backpropagates() -> None:
    torch.manual_seed(32)
    fusion = FusionPulse(k=4, dim=6)
    repeated = torch.randn(3, 1, 6)
    sources = tuple(
        torch.cat((repeated + 0.01 * torch.randn_like(repeated), torch.randn(3, 2, 6)), dim=1)
        for _ in range(3)
    )
    inputs = tuple(source.requires_grad_() for source in sources)

    output, info = fusion.concat(*inputs, return_info=True)
    loss = output.square().mean() + info["structural_loss"]
    loss.backward()

    assert torch.isfinite(info["structural_loss"])
    assert all(source.grad is not None and torch.isfinite(source.grad).all() for source in inputs)
    assert fusion.salience[-1].weight.grad is not None
    assert torch.isfinite(fusion.salience[-1].weight.grad).all()
    assert fusion.context.in_proj_weight.grad is not None
    assert fusion.unfold.exposed_queries.grad is not None


def test_fusion_pulse_structural_loss_prefers_one_representative() -> None:
    fusion = FusionPulse(k=3, dim=3)
    pulses = torch.tensor(
        [
            [
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ]
    )
    mask = torch.ones(1, 4, dtype=torch.bool)
    all_survive = torch.ones_like(pulses)
    all_fade = torch.full_like(pulses, 1.0 / 16.0)
    representative = torch.ones_like(pulses)
    representative[:, 1] = 1.0 / 16.0

    def structural_loss(survival: torch.Tensor) -> torch.Tensor:
        redundancy, support, retained = fusion._structural_losses(
            pulses,
            survival,
            mask,
        )
        return (
            fusion.redundancy_weight * redundancy
            + fusion.support_weight * support
            + fusion.representative_weight * retained
        )

    balanced = structural_loss(representative)
    assert balanced < structural_loss(all_survive)
    assert balanced < structural_loss(all_fade)


def test_fusion_pulse_source_count_is_dynamic() -> None:
    fusion = FusionPulse(k=3, dim=4).eval()

    two_sources = fusion.concat(*(torch.randn(2, 3, 4) for _ in range(2)))
    five_sources = fusion.concat(*(torch.randn(2, 3, 4) for _ in range(5)))

    assert two_sources.shape == (2, 3, 4)
    assert five_sources.shape == (2, 3, 4)


def test_fusion_pulse_all_invalid_sample_is_zero() -> None:
    fusion = FusionPulse(k=3, dim=4)
    sources = torch.randn(2, 2, 3, 4)
    mask = torch.tensor(
        [
            [[False, False, False], [False, False, False]],
            [[True, True, True], [True, False, False]],
        ]
    )

    output, info = fusion(sources, mask=mask, return_info=True)

    assert torch.equal(output[0], torch.zeros_like(output[0]))
    assert not info["pulse_mask"][0].any()
    assert info["pulse_mask"][1].all()
    assert torch.isfinite(info["structural_loss"])


def test_fusion_pulse_state_dict_round_trip() -> None:
    torch.manual_seed(33)
    source = FusionPulse(k=3, dim=4).eval()
    target = FusionPulse(k=3, dim=4).eval()
    target.load_state_dict(source.state_dict())
    pulses = torch.randn(2, 4, 3, 4)

    torch.testing.assert_close(source(pulses), target(pulses))


def test_fusion_pulse_device_and_dtype_smoke() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    fusion = FusionPulse(k=3, dim=8).to(device=device, dtype=dtype)
    pulses = torch.randn(2, 3, 4, 8, device=device, dtype=dtype, requires_grad=True)

    output = fusion(pulses)
    output.float().square().mean().backward()

    assert output.shape == (2, 3, 8)
    assert output.device.type == device.type
    assert output.dtype == dtype
    assert pulses.grad is not None and torch.isfinite(pulses.grad).all()


def test_fusion_pulse_public_exports() -> None:
    assert arti.FusionPulse is FusionPulse
    assert arti_torch.FusionPulse is FusionPulse


def test_fusion_pulse_rejects_invalid_contracts() -> None:
    with pytest.raises(ValueError, match="divide dim"):
        FusionPulse(k=3, dim=5, salience_heads=2)
    with pytest.raises(ValueError, match="greater than half_threshold"):
        FusionPulse(k=3, dim=4, half_threshold=4.0, salience_scale=4.0)

    fusion = FusionPulse(k=3, dim=4)
    with pytest.raises(ValueError, match="at least one"):
        fusion.concat()
    with pytest.raises(ValueError, match="feature dimension"):
        fusion(torch.randn(2, 3, 5))
    with pytest.raises(ValueError, match="one tensor per"):
        fusion.concat(
            torch.randn(2, 3, 4),
            torch.randn(2, 4, 4),
            masks=(torch.ones(2, 3),),
        )
