from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F

import arti
import arti.nn as arti_nn
import arti.torch as arti_torch


def config(**overrides) -> arti.VisualScanConfig:
    values = {
        "low_size": (8, 8),
        "scale": 2,
        "blur_sigma": 0.0,
        "patch_size": (4, 4),
        "pulse_count": 3,
        "pulse_hidden_dim": 24,
    }
    values.update(overrides)
    return arti.VisualScanConfig(**values)


def test_pixel_shift_observation_matches_dht_model_at_zero_shift() -> None:
    cfg = config(shifts=((0.0, 0.0),))
    high = torch.arange(256, dtype=torch.float32).reshape(1, 1, 16, 16) / 255.0

    observed = arti.pixel_shift_observe(high, cfg)
    expected = F.interpolate(high, size=(8, 8), mode="area")

    assert observed.frames.shape == (1, 1, 1, 8, 8)
    assert torch.allclose(observed.frames[:, 0], expected)
    assert torch.equal(observed.shifts, torch.zeros(1, 1, 2))


def test_complementary_pixel_shifts_produce_distinct_observations() -> None:
    cfg = config(shifts=((0.0, 0.0), (0.0, 0.5)))
    high = torch.zeros(1, 1, 16, 16)
    high[0, 0, 8, 8] = 1.0

    observed = arti.pixel_shift_observe(high, cfg)

    assert not torch.allclose(observed.frames[:, 0], observed.frames[:, 1])


def test_visual_scan_is_batch_mask_and_gradient_compatible() -> None:
    cfg = config()
    scan = arti_nn.VisualScan(cfg)
    frames = torch.randn(2, 4, 1, 8, 8, requires_grad=True)
    shifts = torch.tensor(cfg.shifts)
    mask = torch.tensor([[True, True, True, True], [True, True, False, False]])

    output = scan(frames, shifts, frame_mask=mask, return_info=True)
    output.pulses.square().mean().backward()

    assert output.pulses.shape == (2, 3, cfg.fragment_dim)
    assert output.fragments.shape == (2, 80, cfg.fragment_dim)
    assert output.mask.sum(dim=1).tolist() == [80, 48]
    assert frames.grad is not None
    assert torch.isfinite(frames.grad).all()


def test_visual_scan_is_invariant_to_paired_frame_order_without_persistence() -> None:
    cfg = config(persistence_steps=0)
    scan = arti.VisualScan(cfg).eval()
    frames = torch.randn(2, 4, 1, 8, 8)
    shifts = torch.tensor(cfg.shifts)
    order = torch.tensor([2, 0, 3, 1])

    direct = scan(frames, shifts)
    shuffled = scan(frames[:, order], shifts[order])

    assert torch.allclose(direct, shuffled, atol=1e-6, rtol=1e-5)


def test_visual_scan_applies_continuous_inverse_registration_before_fragmentation() -> None:
    cfg = config(shifts=((0.25, 0.75),), include_shift=False, register_shifts=True, include_registered_fusion=False)
    scan = arti.VisualScan(cfg).eval()
    frames = torch.randn(1, 1, 1, 8, 8)
    shifts = torch.tensor(cfg.shifts)

    output = scan(frames, shifts, return_info=True)
    lifted = F.interpolate(frames[:, 0], size=cfg.high_size, mode="bicubic", align_corners=False)
    expected = scan.field(arti.visual_scan.translate_image(lifted, -shifts * cfg.scale)).fragments

    assert torch.allclose(output.fragments, expected, atol=1e-6, rtol=1e-5)
    assert output.fragments.shape[-1] == cfg.fragment_dim


def test_registered_workspace_residual_preserves_the_physical_carrier() -> None:
    cfg = config(pulse_count=16, registered_workspace_residual=True)
    scan = arti.VisualScan(cfg).eval()
    frames = torch.randn(2, 4, 1, 8, 8)
    shifts = torch.tensor(cfg.shifts)

    with torch.no_grad():
        for parameter in scan.pulse.parameters():
            parameter.zero_()
    output = scan(frames, shifts, return_info=True)
    fused_fragments = output.fragments.reshape(2, 5, 16, -1)[:, -1]

    assert torch.allclose(output.pulses, fused_fragments, atol=1e-6, rtol=1e-5)


def test_visual_scan_carrier_only_path_allocates_no_pulse_parameters() -> None:
    cfg = config(pulse_count=16, use_pulse=False, registered_workspace_residual=True)
    scan = arti.VisualScan(cfg).eval()
    output = scan(torch.randn(2, 4, 1, 8, 8), torch.tensor(cfg.shifts), return_info=True)
    fused_fragments = output.fragments.reshape(2, 5, 16, -1)[:, -1]

    assert scan.pulse is None
    assert torch.allclose(output.pulses, fused_fragments)


def test_visual_scan_pulse_only_path_disables_carrier_residual() -> None:
    cfg = config(registered_workspace_residual=False, use_pulse=True, pulse_use_half=False)
    scan = arti.VisualScan(cfg)
    output = scan(torch.randn(2, 4, 1, 8, 8), torch.tensor(cfg.shifts))

    assert scan.pulse is not None
    assert isinstance(scan.pulse.half_act, torch.nn.Identity)
    assert output.shape == (2, cfg.pulse_count, cfg.fragment_dim)


def test_scan_persistence_has_finite_support_and_decay() -> None:
    cfg = config(persistence_steps=2, persistence_decay=0.5, include_registered_fusion=False)
    scan = arti.VisualScan(cfg)
    output = scan(torch.ones(1, 4, 1, 8, 8), torch.tensor(cfg.shifts), return_info=True)
    per_frame = output.survival.reshape(1, 4, -1)[:, :, 0]

    assert torch.equal(per_frame, torch.tensor([[0.0, 0.0, 0.5, 1.0]]))


def test_visual_scan_config_and_arti_st_roundtrip(tmp_path) -> None:
    cfg = config(persistence_steps=3, noise_std=0.01)
    assert arti.VisualScanConfig.from_dict(json.loads(json.dumps(cfg.to_dict()))) == cfg
    model = arti.VisualScan(cfg).eval()
    frames = torch.randn(2, 4, 1, 8, 8)
    shifts = torch.tensor(cfg.shifts)
    expected = model(frames, shifts)
    path = tmp_path / "visual-scan.arti.st"

    arti.save(model, path, config={"visual_scan": cfg.to_dict()})
    restored = arti.VisualScan.from_config(cfg.to_dict()).eval()
    arti.load(path, model=restored)

    assert torch.allclose(restored(frames, shifts), expected)


def test_shift_and_add_returns_registered_high_resolution_tensor() -> None:
    cfg = config()
    high = torch.randn(2, 1, 16, 16)
    observed = arti.pixel_shift_observe(high, cfg)
    reconstructed = arti.shift_and_add(observed.frames, observed.shifts, scale=2)
    assert reconstructed.shape == high.shape
    assert torch.isfinite(reconstructed).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_visual_scan_cuda_mixed_precision() -> None:
    cfg = config()
    scan = arti.VisualScan(cfg).cuda()
    frames = torch.randn(4, 4, 1, 8, 8, device="cuda")
    shifts = torch.tensor(cfg.shifts, device="cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = scan(frames, shifts)
    assert output.is_cuda
    assert output.dtype == torch.bfloat16


def test_visual_scan_public_namespaces() -> None:
    assert arti.VisualScan is arti_nn.VisualScan
    assert arti_torch.VisualScan is arti_nn.VisualScan
    assert arti.pixel_shift_observe is arti_torch.pixel_shift_observe
    assert arti.shift_and_add is arti_torch.shift_and_add
