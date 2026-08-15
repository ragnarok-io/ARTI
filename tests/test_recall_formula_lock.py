from __future__ import annotations

import json

import pytest
import torch
from torch import nn

import arti
from arti.recall_formula import (
    FactorSpec,
    RecallFormulaContract,
    RecallFormulaLock,
)


def _contract() -> RecallFormulaContract:
    return RecallFormulaContract(
        identity=arti.RecallFormulaId.parse("tests/locked@1"),
        factors=(
            FactorSpec("gain", route="shared", init="zero"),
            FactorSpec("shift", route="shared", init="normal", init_scale=0.02),
        ),
        identity_preserving=True,
        composition="custom",
        capabilities=("torch.compile", "torch.eager"),
    )


def test_contract_and_lock_round_trip_without_executable_state() -> None:
    contract = _contract()
    restored_contract = RecallFormulaContract.from_dict(
        json.loads(json.dumps(contract.to_dict()))
    )
    lock = RecallFormulaLock.bind(contract, hidden_dim=16, slots=8)
    restored_lock = RecallFormulaLock.from_dict(
        json.loads(json.dumps(lock.to_dict()))
    )

    assert restored_contract == contract
    assert contract.api_version == 2
    assert restored_contract.fingerprint == contract.fingerprint
    assert restored_lock == lock
    assert restored_lock.contract_fingerprint == contract.fingerprint
    assert "state_dict" not in json.dumps(lock.to_dict())


def test_lock_binds_shape_and_factor_count() -> None:
    contract = _contract()
    lock = RecallFormulaLock.bind(contract, hidden_dim=16, slots=8)

    lock.validate(contract, hidden_dim=16, slots=8)

    with pytest.raises(ValueError, match="shape"):
        lock.validate(contract, hidden_dim=32, slots=8)
    with pytest.raises(ValueError, match="fingerprint"):
        lock.validate(
            RecallFormulaContract(
                factors=(FactorSpec("other"), FactorSpec("shift")),
                capabilities=("torch.eager",),
            ),
            hidden_dim=16,
            slots=8,
        )
    with pytest.raises(ValueError, match="divisible"):
        RecallFormulaLock.bind(contract, hidden_dim=16, slots=7)


def test_contract_rejects_noncanonical_capabilities() -> None:
    with pytest.raises(ValueError, match="sorted"):
        RecallFormulaContract(
            factors=(FactorSpec("content"),),
            capabilities=("torch.eager", "torch.compile"),
        )


def test_recall_exposes_the_same_lock_used_by_its_manifest() -> None:
    recall = arti.Recall(dim=4, slots=8, formula="arti/affine@1")

    assert recall.formula_lock.hidden_dim == 4
    assert recall.formula_lock.slots == 8
    assert recall.formula_lock.factor_names == ("scale", "shift")
    assert recall.formula_lock.contract.composition == "product"
    assert recall.formula_manifest().capabilities == recall.formula_lock.capabilities


def test_batched_execution_contract_uses_flat_formula_abi() -> None:
    class BatchedFormula(nn.Module):
        recall_formula_contract = RecallFormulaContract(
            factors=(FactorSpec("content"),),
            execution=arti.RecallFormulaExecutionSpec(vectorization="batched"),
        )

        def forward(self, state: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
            if state.ndim != 2 or factors.ndim != 3:
                raise RuntimeError("batched ABI was not used")
            return state + factors[:, 0, :]

    recall = arti.Recall(dim=4, slots=8, formula=BatchedFormula())
    output = recall(torch.randn(2, 3, 4))

    assert output.shape == (2, 3, 4)
    assert recall.formula_lock.contract.execution.vectorization == "batched"


def test_batched_execution_rejects_cross_row_reduction() -> None:
    class ReducingFormula(nn.Module):
        recall_formula_contract = RecallFormulaContract(
            factors=(FactorSpec("content"),),
            execution=arti.RecallFormulaExecutionSpec(vectorization="batched"),
        )

        def forward(self, state: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
            return state + state.mean() + factors[:, 0, :]

    with pytest.raises(ValueError, match="row-independent"):
        arti.Recall(dim=4, slots=8, formula=ReducingFormula())


def test_stochastic_scalar_formula_uses_explicit_vmap_randomness() -> None:
    class StochasticFormula(nn.Module):
        recall_formula_contract = RecallFormulaContract(
            factors=(FactorSpec("content"),),
            execution=arti.RecallFormulaExecutionSpec(deterministic=False),
        )

        def forward(self, state: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
            return state + factors[0] + torch.rand_like(state) * 0.01

    recall = arti.Recall(dim=4, slots=8, formula=StochasticFormula(), activation="none")
    output = recall(torch.randn(2, 3, 4))

    assert output.shape == (2, 3, 4)


def test_formula_dtype_contract_is_checked_at_runtime() -> None:
    class Float32Formula(nn.Module):
        recall_formula_contract = RecallFormulaContract(
            factors=(FactorSpec("content"),),
            execution=arti.RecallFormulaExecutionSpec(supported_dtypes=("float32",)),
        )

        def forward(self, state: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
            return state + factors[0]

    recall = arti.Recall(
        dim=4,
        slots=8,
        formula=Float32Formula(),
        activation="none",
    ).half()

    with pytest.raises(TypeError, match="dtype"):
        recall(torch.randn(2, 3, 4, dtype=torch.float16))
