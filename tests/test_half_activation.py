from __future__ import annotations

import torch
import pytest

import arti
import arti.functional as F
import arti.nn as arti_nn
import arti.torch as arti_torch


def test_half_function_matches_salience_formula() -> None:
    x = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    y = F.half(x)
    deficit = torch.relu(1.0 - x.abs())
    expected = torch.pow(torch.tensor(0.5), deficit) * x
    assert torch.allclose(y, expected)
    assert torch.equal(y[x.abs() >= 1.0], x[x.abs() >= 1.0])
    assert y[3].item() == 0.0


def test_half_module_is_activation_like_and_stateless() -> None:
    layer = arti_nn.Half()
    assert list(layer.parameters()) == []
    x = torch.randn(3, 4, requires_grad=True)
    y = layer(x)
    assert y.shape == x.shape
    y.sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"base": float("nan")}, "base"),
        ({"base": float("inf")}, "base"),
        ({"scale": float("nan")}, "scale"),
        ({"scale": float("inf")}, "scale"),
        ({"threshold": float("nan")}, "threshold"),
        ({"threshold": float("inf")}, "threshold"),
    ],
)
def test_half_rejects_nonfinite_scalar_parameters(
    kwargs: dict[str, float], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        arti_nn.Half(**kwargs)(torch.ones(2, 3))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"base": torch.tensor(float("nan"))}, "base"),
        ({"scale": torch.tensor(float("inf"))}, "scale"),
        ({"threshold": torch.tensor(float("nan"))}, "threshold"),
    ],
)
def test_half_rejects_nonfinite_tensor_parameters(
    kwargs: dict[str, torch.Tensor], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        arti_nn.Half(**kwargs)(torch.ones(2, 3))


def test_half_stochastic_only_applies_in_training_mode() -> None:
    x = torch.full((4096,), 0.0)
    layer = arti_nn.Half(stochastic=True)
    layer.train()
    train_y = layer(x + 1.0)
    assert torch.equal(train_y, x + 1.0)

    weak = torch.full((4096,), 0.0)
    train_weak = layer(weak + 0.25)
    survived = (train_weak != 0).float().mean().item()
    assert 0.45 < survived < 0.75

    layer.eval()
    eval_y = layer(weak + 0.25)
    expected = F.half(weak + 0.25)
    assert torch.allclose(eval_y, expected)


def test_half_public_namespaces() -> None:
    assert arti.Half is arti_nn.Half
    assert arti_torch.Half is arti_nn.Half
    assert callable(F.half)
    assert arti_torch.half is F.half
