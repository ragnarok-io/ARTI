from __future__ import annotations

import torch
import pytest

import arti
import arti.functional as F
import arti.nn as arti_nn
import arti.torch as arti_torch


def test_half_function_matches_salience_formula() -> None:
    x = torch.tensor([-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    y = F.half(x, stochastic=False)
    deficit = torch.relu(1.0 - x.abs())
    expected = torch.pow(torch.tensor(0.5), deficit) * x
    assert torch.allclose(y, expected)
    assert torch.equal(y[x.abs() >= 1.0], x[x.abs() >= 1.0])
    assert y[3].item() == 0.0


def test_half_module_is_activation_like_and_stateless() -> None:
    layer = arti_nn.Half(stochastic=False)
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


def test_half_stochastic_mode_is_independent_of_module_mode() -> None:
    weak = torch.full((4096,), 0.25)
    layer = arti_nn.Half(stochastic=True)

    torch.manual_seed(19)
    layer.train()
    train_y = layer(weak)
    torch.manual_seed(19)
    layer.eval()
    eval_y = layer(weak)

    assert torch.equal(train_y, eval_y)
    survived = (train_y != 0).float().mean().item()
    assert 0.45 < survived < 0.75


def test_half_accepts_explicit_uniform_without_consuming_global_rng() -> None:
    x = torch.full((2, 4), 0.25, requires_grad=True)
    uniform = torch.tensor(
        [[0.0, 0.2, 0.6, 0.9], [0.1, 0.3, 0.7, 0.8]],
        dtype=x.dtype,
    )
    layer = arti_nn.Half(stochastic=True, learnable=True)
    before = torch.random.get_rng_state().clone()
    first = layer(x, uniform=uniform)
    second = layer(x, uniform=uniform)

    assert torch.equal(first, second)
    assert torch.equal(torch.random.get_rng_state(), before)
    first.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in layer.parameters()
    )


def test_half_learnable_survival_curve_has_parameters_and_gradients() -> None:
    layer = arti_nn.Half(stochastic=False, learnable=True)
    assert {name for name, _ in layer.named_parameters()} == {
        "_threshold",
        "_base_logit",
        "_scale_raw",
    }
    x = torch.full((8, 4), 0.25, requires_grad=True)
    q = layer.survival(x)
    assert torch.isfinite(q).all()
    q.mean().backward()
    assert all(parameter.grad is not None for parameter in layer.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in layer.parameters())


def test_half_learnable_stochastic_mode_uses_surrogate_gradient() -> None:
    layer = arti_nn.Half(stochastic=True, learnable=True)
    x = torch.full((8, 4), 0.25, requires_grad=True)
    torch.manual_seed(23)
    layer(x).sum().backward()
    assert all(parameter.grad is not None for parameter in layer.parameters())


def test_half_learnable_parameters_round_trip() -> None:
    layer = arti_nn.Half(threshold=0.75, base=0.25, scale=0.5, stochastic=False, learnable=True)
    restored = arti_nn.Half(threshold=0.75, base=0.25, scale=0.5, stochastic=False, learnable=True)
    restored.load_state_dict(layer.state_dict())
    x = torch.randn(3, 4)
    assert torch.equal(restored.survival(x), layer.survival(x))


def test_contextual_half_is_same_shape_but_not_pointwise() -> None:
    layer = arti_nn.Half(stochastic=False, context_mode="contextual", context_gain=0.5)
    x = torch.tensor([[0.25, 0.25]], requires_grad=True)
    q = layer.survival(x)
    y = layer(x)

    assert q.shape == x.shape
    assert y.shape == x.shape
    assert torch.allclose(y, q * x)

    changed_context = torch.tensor([[0.25, 2.0]])
    changed_q = layer.survival(changed_context)
    assert not torch.equal(q[0, 0], changed_q[0, 0])

    y[0, 0].backward()
    assert x.grad is not None
    assert x.grad[0, 1].abs() > 0


def test_contextual_half_zero_gain_reduces_to_elementwise() -> None:
    x = torch.randn(2, 3, 4)
    pointwise = arti_nn.Half(stochastic=False)
    contextual = arti_nn.Half(
        stochastic=False,
        context_mode="contextual",
        context_gain=0.0,
        context_axes=(-2, -1),
    )
    assert torch.equal(contextual.survival(x), pointwise.survival(x))


def test_contextual_half_rejects_invalid_context_axes() -> None:
    with pytest.raises(ValueError, match="context axis"):
        arti_nn.Half(stochastic=False, context_mode="contextual", context_axes=4).survival(
            torch.ones(2, 3)
        )


def test_half_public_namespaces() -> None:
    assert arti.Half is arti_nn.Half
    assert arti_torch.Half is arti_nn.Half
    assert callable(F.half)
    assert arti_torch.half is F.half
