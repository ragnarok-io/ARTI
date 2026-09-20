from __future__ import annotations

import pytest
import torch
from torch import nn

import arti
import arti.functional as F
from arti.component_registry import ComponentCompatibilityError, component_provenance


class _LearnedSurvival(nn.Module):
    def __init__(self, declaration: str | None = None) -> None:
        super().__init__()
        self.logit = nn.Parameter(torch.tensor(0.0))
        self.survival_contract = (
            None
            if declaration is None
            else arti.SurvivalContract(identity=arti.SurvivalRef.parse(declaration))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.logit).expand_as(x)


def test_builtin_survival_is_versioned_and_same_shape() -> None:
    operator = arti.ExponentialSurvival(learnable=True)
    x = torch.randn(2, 3, 4, requires_grad=True)

    q = operator(x)
    assert q.shape == x.shape
    assert torch.isfinite(q).all()
    assert torch.all((q >= 0) & (q <= 1))
    q.mean().backward()
    assert all(parameter.grad is not None for parameter in operator.parameters())
    assert arti.describe_survival("arti/survival@1").portable


def test_half_can_resolve_builtin_survival_by_reference() -> None:
    layer = arti.Half(
        stochastic=False,
        survival="arti/survival@1",
        survival_config={"threshold": 0.5, "base": 0.25, "scale": 2.0},
    )
    x = torch.randn(2, 3)

    reference = arti.describe_survival("arti/survival@1").reference
    assert layer.survival_reference == reference
    assert torch.equal(layer(x), layer.survival(x) * x)
    root = next(item for item in component_provenance(layer)["components"] if item["path"] == "$")
    assert root["dependencies"] == [reference]
    assert root["config"]["survival"]["portable"] is True


def test_custom_module_survival_is_differentiable_but_runtime_only() -> None:
    layer = arti.Half(stochastic=False, survival=_LearnedSurvival())
    x = torch.randn(2, 3, requires_grad=True)

    y = layer(x)
    y.sum().backward()
    assert x.grad is not None
    assert layer.survival_runtime_only is True
    assert next(layer.survival_operator.parameters()).grad is not None
    with pytest.raises(ComponentCompatibilityError, match="runtime-only"):
        component_provenance(layer)


def test_functional_half_accepts_a_custom_survival_function() -> None:
    x = torch.tensor([[-1.0, 0.5]])
    y = F.half(x, stochastic=False, survival_fn=lambda value: torch.full_like(value, 0.25))
    assert torch.equal(y, x * 0.25)


def test_custom_survival_rejects_shape_range_and_device_contract_breaks() -> None:
    class BadShape(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.ones(x.shape[:-1])

    class BadRange(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.full_like(x, 2.0)

    x = torch.ones(2, 3)
    with pytest.raises(ValueError, match="same shape"):
        arti.Half(stochastic=False, survival=BadShape()).survival(x)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        arti.Half(stochastic=False, survival=BadRange()).survival(x)


def test_registered_application_survival_can_be_used_locally() -> None:
    reference = "test/constant-survival@1"

    arti.register_survival(
        reference,
        factory=lambda config: _LearnedSurvival(reference),
        description="test-only survival",
    )
    registration = arti.resolve_survival(reference)
    assert registration.portable is False
    layer = arti.Half(stochastic=False, survival=reference)
    assert layer.survival_runtime_only is True
    assert torch.allclose(layer.survival(torch.ones(2, 3)), torch.full((2, 3), 0.5))
    assert layer.survival_reference == arti.resolve_survival(reference).reference


def test_source_declarations_cannot_be_serialized() -> None:
    with pytest.raises(arti.InvalidSurvivalRefError, match="cannot be serialized"):
        arti.SurvivalRef.parse("arti/survival@1").to_dict()
