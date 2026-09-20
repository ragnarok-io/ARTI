from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "benchmarks" / "fhu_concat_half_dedup.py"
SPEC = importlib.util.spec_from_file_location("fhu_concat_half_dedup", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_learned_salience_receives_finite_gradients_through_half() -> None:
    thinner = MODULE.PostConcatThinner(mode="learned_half", dim=6)
    x = torch.randn(3, 8, 6, requires_grad=True)

    y, survival = thinner(x)
    (y.square().mean() + survival.mean()).backward()

    assert y.shape == x.shape
    assert survival.shape == x.shape
    assert torch.all((survival >= 0.25) & (survival <= 1.0))
    assert x.grad is not None and torch.isfinite(x.grad).all()
    gradients = [
        parameter.grad for parameter in thinner.scorer.parameters() if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(gradient.abs().sum() for gradient in gradients) > 0


def test_shared_unfold_set_model_uses_dynamic_target_length() -> None:
    model = MODULE.SharedUnFoldSetModel(
        mode="learned_half",
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
