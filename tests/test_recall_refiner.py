from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

import arti
import arti.nn as arti_nn
import arti.torch as arti_torch


class ResidualRecall(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim))

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


class ConstantRecall(nn.Module):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return torch.full_like(h, self.value)


class KwargRecall(nn.Module):
    def forward(self, h: torch.Tensor, *, gain: torch.Tensor) -> torch.Tensor:
        return h * gain


@dataclass
class OutputLike:
    y: torch.Tensor


class OutputRecall(nn.Module):
    def forward(self, h: torch.Tensor) -> OutputLike:
        return OutputLike(y=h * 0.25)


def test_recall_refiner_compacts_api_shape_and_info() -> None:
    refiner = arti_nn.RecallRefiner(ResidualRecall(dim=5), steps=3, step_scale=0.5)
    h = torch.randn(2, 5)

    y, info = refiner(h, return_info=True, record_history=True)

    assert y.shape == h.shape
    assert info["steps"].item() == 3
    assert info["delta_norm"].shape == (3,)
    assert info["raw_delta_norm"].shape == (3,)
    assert info["update_norm"].shape == (3,)
    assert info["survival_mean"].shape == (3,)
    assert info["state_history"].shape == (4, 2, 5)


def test_recall_refiner_is_differentiable() -> None:
    refiner = arti_nn.RecallRefiner(ResidualRecall(dim=6), steps=2, step_scale=0.5, learnable_step_scale=True)
    h = torch.randn(3, 6, requires_grad=True)

    loss = refiner(h).square().mean()
    loss.backward()

    assert h.grad is not None
    assert torch.isfinite(h.grad).all()
    assert refiner.step_scale.grad is not None
    grads = [param.grad for param in refiner.parameters() if param.requires_grad and param.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_recall_refiner_runtime_steps_override_and_reuses_last_scale() -> None:
    refiner = arti_nn.RecallRefiner(ConstantRecall(1.25), steps=2, step_scale=[0.5])
    h = torch.zeros(1, 3)

    y, info = refiner(h, steps=4, return_info=True)

    assert info["steps"].item() == 4
    assert torch.allclose(info["step_scale"], torch.full((4,), 0.5))
    assert torch.all(y > 0)


def test_recall_refiner_early_stop() -> None:
    refiner = arti_nn.RecallRefiner(ConstantRecall(0.01), steps=5, step_scale=0.1, use_half=False)
    h = torch.zeros(1, 2)

    _, info = refiner(h, tolerance=0.01, return_info=True)

    assert info["stopped_early"].item()
    assert info["steps"].item() == 1


def test_recall_refiner_half_reduces_weak_delta_accumulation() -> None:
    h = torch.zeros(1, 8)
    no_half = arti_nn.RecallRefiner(ConstantRecall(0.05), steps=6, step_scale=1.0, use_half=False)
    with_half = arti_nn.RecallRefiner(ConstantRecall(0.05), steps=6, step_scale=1.0)

    y_no_half = no_half(h)
    y_with_half = with_half(h)

    assert y_with_half.norm() < y_no_half.norm()


def test_recall_refiner_passes_kwargs_to_recall_layer() -> None:
    refiner = arti_nn.RecallRefiner(KwargRecall(), steps=2, step_scale=1.0, use_half=False)
    h = torch.ones(1, 4)
    gain = torch.tensor(0.25)

    y = refiner(h, gain=gain)

    assert torch.allclose(y, torch.full_like(h, 1.5625))


def test_recall_refiner_accepts_output_object_with_y() -> None:
    refiner = arti_nn.RecallRefiner(OutputRecall(), steps=2, step_scale=1.0, use_half=False)
    h = torch.ones(1, 4)

    y = refiner(h)

    assert torch.allclose(y, torch.full_like(h, 1.5625))


def test_recall_refiner_public_namespaces() -> None:
    assert arti.RecallRefiner is arti_nn.RecallRefiner
    assert arti_torch.RecallRefiner is arti_nn.RecallRefiner
