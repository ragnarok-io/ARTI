"""Black-box contract tests for :mod:`arti.recall_formula`."""

from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

from arti.recall_formula import (
    BUILTIN_RECALL_FORMULAS,
    FactorSpec,
    MAX_RECALL_FORMULA_FACTORS,
    RecallFormulaContract,
    validate_formula,
)


class AdditiveFormula(nn.Module):
    factor_names = ("content",)

    def forward(self, state: Tensor, factors: Tensor) -> Tensor:
        return state + factors[..., 0, :]


class TrainableFormula(nn.Module):
    factor_names = ("content", "gate")

    def __init__(self) -> None:
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.75))

    def forward(self, state: Tensor, factors: Tensor) -> Tensor:
        content, gate = factors.unbind(dim=-2)
        return state + self.gain * torch.tanh(gate) * content


@pytest.mark.parametrize(
    ("reference", "factor_count"),
    [("arti/delta@1", 1), ("arti/affine@1", 2), ("arti/state@1", 17)],
)
def test_builtin_ids_are_canonical(reference: str, factor_count: int) -> None:
    description = BUILTIN_RECALL_FORMULAS[reference]

    assert description.contract.identity is not None
    assert description.contract.identity.reference == reference
    assert description.contract.factor_count == factor_count
    assert len(set(description.contract.factor_names)) == factor_count


def test_custom_module_contract_means_complete_next_state() -> None:
    state = torch.randn(2, 3, 5)
    factors = torch.randn(2, 3, 1, 5)
    module = AdditiveFormula()

    contract = validate_formula(module, state)
    next_state = module(state, factors)

    assert contract.output_semantics == "next_state"
    assert contract.factor_names == ("content",)
    assert torch.equal(next_state, state + factors[..., 0, :])


def test_factor_names_define_count_and_are_not_reordered() -> None:
    contract = validate_formula(TrainableFormula(), torch.randn(2, 3, 5))

    assert contract.factor_names == ("content", "gate")
    assert contract.factor_count == 2


@pytest.mark.parametrize(
    "factor_names",
    [(), ("content", "content"), ("",), ("content", 3)],
)
def test_invalid_factor_names_are_rejected(factor_names: tuple[object, ...]) -> None:
    module = AdditiveFormula()
    module.factor_names = factor_names

    with pytest.raises((TypeError, ValueError), match="factor"):
        validate_formula(module, torch.randn(2, 3, 5))


def test_explicit_factor_count_and_names_must_agree() -> None:
    with pytest.raises(ValueError, match="factor_count"):
        validate_formula(
            AdditiveFormula(),
            torch.randn(2, 3, 5),
            factor_count=2,
            factor_names=("content",),
        )


class WrongShapeFormula(AdditiveFormula):
    def forward(self, state: Tensor, factors: Tensor) -> Tensor:
        return state[..., :-1]


class WrongDtypeFormula(AdditiveFormula):
    def forward(self, state: Tensor, factors: Tensor) -> Tensor:
        return state.to(torch.float64)


class WrongDeviceFormula(AdditiveFormula):
    def forward(self, state: Tensor, factors: Tensor) -> Tensor:
        return torch.empty_like(state, device="meta")


class NonFiniteFormula(AdditiveFormula):
    def forward(self, state: Tensor, factors: Tensor) -> Tensor:
        return torch.full_like(state, float("nan"))


@pytest.mark.parametrize(
    ("module", "message"),
    [
        (WrongShapeFormula(), "shape"),
        (WrongDtypeFormula(), "dtype"),
        (WrongDeviceFormula(), "device"),
        (NonFiniteFormula(), "finite"),
    ],
)
def test_output_shape_device_dtype_and_finiteness_are_checked(
    module: nn.Module,
    message: str,
) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        validate_formula(module, torch.randn(2, 3, 5))


def test_gradients_reach_state_factors_and_formula_parameters() -> None:
    module = TrainableFormula()
    validate_formula(module, torch.randn(2, 3, 5))
    state = torch.randn(2, 3, 5, requires_grad=True)
    factors = torch.randn(2, 3, 2, 5, requires_grad=True)

    module(state, factors).square().mean().backward()

    for gradient in (state.grad, factors.grad, module.gain.grad):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_identity_probe_accepts_zero_factor_identity() -> None:
    state = torch.randn(2, 3, 5)
    contract = validate_formula(AdditiveFormula(), state, test_identity=True)

    assert contract.identity_preserving


def test_identity_probe_rejects_non_identity_formula() -> None:
    class ConstantShift(nn.Module):
        factor_names = ("content",)

        def forward(self, state: Tensor, factors: Tensor) -> Tensor:
            return state + 1

    with pytest.raises(ValueError, match="identity"):
        validate_formula(
            ConstantShift(),
            torch.randn(2, 3, 5),
            test_identity=True,
        )


def test_plain_callable_is_not_an_extension_point() -> None:
    def function_formula(state: Tensor, factors: Tensor) -> Tensor:
        return state + factors[..., 0, :]

    with pytest.raises(TypeError, match="nn.Module"):
        validate_formula(function_formula, torch.randn(2, 3, 5))


def test_formula_cannot_publish_optimizer_control_hooks() -> None:
    class OptimizerOwningFormula(AdditiveFormula):
        def configure_optimizer(self, parameters: list[nn.Parameter]) -> torch.optim.Optimizer:
            return torch.optim.SGD(parameters, lr=1.0)

    with pytest.raises((TypeError, ValueError), match="optimizer"):
        validate_formula(OptimizerOwningFormula(), torch.randn(2, 3, 5))


def test_contract_probe_does_not_allow_input_or_module_state_mutation() -> None:
    class MutatingFormula(nn.Module):
        factor_names = ("content",)

        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("calls", torch.zeros((), dtype=torch.long))

        def forward(self, state: Tensor, factors: Tensor) -> Tensor:
            self.calls.add_(1)
            state.add_(factors[..., 0, :])
            return state

    module = MutatingFormula()
    state = torch.randn(2, 3, 5)
    original_state = state.clone()
    original_module_state = {name: value.clone() for name, value in module.state_dict().items()}

    with pytest.raises((TypeError, ValueError), match="mutat|state"):
        validate_formula(module, state)

    assert torch.equal(state, original_state)
    assert all(
        torch.equal(module.state_dict()[name], value)
        for name, value in original_module_state.items()
    )


def test_contract_probe_restores_state_mode_and_rng_when_formula_raises() -> None:
    class RaisingFormula(nn.Module):
        factor_names = ("content",)

        def __init__(self) -> None:
            super().__init__()
            self.register_buffer("calls", torch.zeros((), dtype=torch.long))

        def forward(self, state: Tensor, factors: Tensor) -> Tensor:
            self.calls.add_(1)
            self.eval()
            torch.rand(())
            raise RuntimeError("probe failed")

    module = RaisingFormula().train()
    probe = torch.randn(4)
    rng_state = torch.random.get_rng_state().clone()

    with pytest.raises(RuntimeError, match="probe failed"):
        validate_formula(module, probe)

    assert module.training
    assert module.calls.item() == 0
    assert torch.equal(torch.random.get_rng_state(), rng_state)


def test_contract_probe_rejects_unbounded_factor_count_before_allocation() -> None:
    module = AdditiveFormula()
    module.factor_names = tuple(
        f"factor_{index}" for index in range(MAX_RECALL_FORMULA_FACTORS + 1)
    )

    with pytest.raises(ValueError, match="factor count exceeds"):
        validate_formula(module, torch.randn(4))


def test_explicit_contract_factor_metadata_is_preserved() -> None:
    class ContractFormula(AdditiveFormula):
        recall_formula_contract = RecallFormulaContract(
            factors=(
                FactorSpec(
                    "content",
                    route="shared",
                    identity=0.0,
                    init="normal",
                    init_scale=0.01,
                ),
            ),
            identity_preserving=True,
        )

    contract = validate_formula(
        ContractFormula(),
        torch.randn(2, 3, 5),
        test_identity=True,
    )

    assert contract is ContractFormula.recall_formula_contract
    assert contract.factors[0].route == "shared"
    assert contract.factors[0].init == "normal"
    assert contract.factors[0].init_scale == pytest.approx(0.01)


@pytest.mark.skipif(
    not hasattr(torch, "compile"),
    reason="torch.compile is not supported by this environment",
)
def test_custom_formula_supports_fullgraph_compile_when_available() -> None:
    module = TrainableFormula()
    state = torch.randn(2, 3, 5)
    factors = torch.randn(2, 3, 2, 5)
    expected = module(state, factors)

    compiled = torch.compile(module, backend="eager", fullgraph=True)
    actual = compiled(state, factors)

    torch.testing.assert_close(actual, expected)
