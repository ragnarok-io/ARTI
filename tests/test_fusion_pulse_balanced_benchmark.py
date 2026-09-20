from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
MODULE_PATH = BENCHMARKS / "fusion_pulse_balanced.py"
sys.path.insert(0, str(BENCHMARKS))
SPEC = importlib.util.spec_from_file_location("fusion_pulse_balanced", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_balanced_half_is_differentiable_and_bounded() -> None:
    thinner = MODULE.BalancedThinner(
        mode="balanced_half",
        dim=6,
        pulse_count=4,
        slots_per_pulse=3,
    )
    x = torch.randn(3, 12, 6, requires_grad=True)

    y, survival = thinner(x)
    (y.square().mean() + survival.mean()).backward()

    assert y.shape == x.shape
    assert survival.shape == x.shape
    assert torch.all((survival >= 1.0 / 16.0) & (survival <= 1.0))
    assert x.grad is not None and torch.isfinite(x.grad).all()
    gradients = [
        parameter.grad for parameter in thinner.scorer.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum() for gradient in gradients) > 0


def test_structural_losses_prefer_one_representative_per_neighborhood() -> None:
    repeated = torch.tensor([1.0, 0.0, 0.0])
    unique_a = torch.tensor([0.0, 1.0, 0.0])
    unique_b = torch.tensor([0.0, 0.0, 1.0])
    pulses = torch.stack(
        (
            torch.stack((repeated, unique_a)),
            torch.stack((repeated, unique_b)),
        )
    ).unsqueeze(0)
    diffuse = torch.full_like(pulses, 0.25)
    representative = torch.full_like(pulses, 1.0 / 16.0)
    representative[:, 0] = 1.0
    representative[:, 1, 1] = 1.0

    diffuse_loss = MODULE.support_deficit(pulses, diffuse)
    diffuse_loss += MODULE.representative_deficit(pulses, diffuse)
    representative_loss = MODULE.support_deficit(pulses, representative)
    representative_loss += MODULE.representative_deficit(pulses, representative)

    assert representative_loss < diffuse_loss
    assert MODULE.redundancy_pressure(pulses, representative) < (
        MODULE.redundancy_pressure(pulses, torch.ones_like(pulses))
    )


def test_balanced_fusion_pulse_uses_one_shared_dynamic_unfold() -> None:
    model = MODULE.BalancedModel(
        mode="balanced_half",
        pulse_count=4,
        common_count=2,
        dim=6,
        value_operators=2,
    )
    pulses = torch.randn(3, 4, 3, 6)

    prediction, survival = model(pulses)

    assert prediction.shape == (3, 6, 6)
    assert survival.shape == pulses.shape
    assert model.unfold.exposed == 6
