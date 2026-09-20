from __future__ import annotations

import pytest
import torch
from torch import nn

import arti
from arti.recall_formula import FactorSpec, RecallFormulaContract


class _MutatingFactors(nn.Module):
    recall_formula_contract = RecallFormulaContract(
        factors=(FactorSpec("content"),),
    )

    def forward(self, state: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
        factors.add_(1)
        return state


class _MutatingNonPersistentBuffer(nn.Module):
    recall_formula_contract = RecallFormulaContract(
        factors=(FactorSpec("content"),),
    )

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer("counter", torch.zeros(()), persistent=False)

    def forward(self, state: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
        self.counter.add_(1)
        return state


@pytest.mark.parametrize("formula", [_MutatingFactors(), _MutatingNonPersistentBuffer()])
def test_formula_validation_rejects_and_restores_nonpersistent_side_effects(
    formula: nn.Module,
) -> None:
    with pytest.raises(ValueError, match="factors, buffers"):
        arti.validate_formula(formula, torch.zeros(2, 4))

    if isinstance(formula, _MutatingNonPersistentBuffer):
        assert formula.counter.item() == 0
