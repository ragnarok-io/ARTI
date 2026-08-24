from __future__ import annotations

import pytest
import torch

import arti
from arti.alpha import (
    OperandKind,
    TypedBankFormulaTopologyPolicy,
    TypedTopologyOperandBank,
    TypedTopologyPriorityFormula,
)
from arti.component_registry import component_spec


def test_typed_topology_matches_v1_values_and_gradients() -> None:
    old_bank = arti.alpha.TopologyOperandBank(
        slots=5,
        key_dim=3,
        factor_dim=2,
        seed=7,
        value_seed=11,
    )
    typed_bank = TypedTopologyOperandBank(
        slots=5,
        key_dim=3,
        factor_dim=2,
        seed=7,
        value_seed=11,
    )
    old = arti.alpha.BankFormulaTopologyPolicy(dim=4, banks=[old_bank], key_dim=3)
    typed = TypedBankFormulaTopologyPolicy(dim=4, banks=[typed_bank], key_dim=3)
    x = torch.randn(2, 6, 4)
    mask = torch.tensor(
        [[True, True, False, True, True, False], [True, True, True, True, True, True]]
    )

    old_priority = old(x, mask).action.priority
    typed_priority = typed(x, mask).action.priority

    torch.testing.assert_close(typed_priority, old_priority, rtol=0, atol=0)
    old_priority.square().sum().backward()
    typed_priority.square().sum().backward()
    torch.testing.assert_close(typed_bank.values.grad, old_bank.values.grad, rtol=0, atol=0)


def test_bank_emits_formula_bound_typed_operands() -> None:
    bank = TypedTopologyOperandBank(slots=4, key_dim=3, factor_dim=2, seed=3)
    formula = TypedTopologyPriorityFormula(factor_dim=2)
    query = torch.randn(2, 5, 3)
    mask = torch.ones(2, 5, dtype=torch.bool)

    operands, route = bank.read_operands(query, mask, consumer=formula)
    output = formula.evaluate_operands(operands, source=bank)

    assert operands.contract.kind is OperandKind.TOPOLOGY
    assert operands.contract.source_ref == "arti/topology-operand-bank@2"
    assert operands.contract.consumer_ref == "arti/topology-priority-formula@2"
    assert operands.contract.source_asset_fingerprint == bank.asset_fingerprint
    assert route.shape == (2, 5, 4)
    assert output.priority.shape == (2, 5)


def test_typed_formula_rejects_wrong_source_asset_and_old_formula() -> None:
    source = TypedTopologyOperandBank(
        slots=4, key_dim=3, factor_dim=2, seed=3, bank_id="shared-bank"
    )
    other = TypedTopologyOperandBank(
        slots=4, key_dim=3, factor_dim=2, seed=9, bank_id="shared-bank"
    )
    formula = TypedTopologyPriorityFormula(factor_dim=2)
    query = torch.randn(1, 2, 3)
    mask = torch.ones(1, 2, dtype=torch.bool)
    operands, _ = source.read_operands(query, mask, consumer=formula)

    with pytest.raises(ValueError, match="source asset"):
        formula.evaluate_operands(operands, source=other)
    with pytest.raises(TypeError, match="TopologyPriorityFormula@2"):
        TypedBankFormulaTopologyPolicy(
            dim=4,
            banks=[source],
            key_dim=3,
            formula=arti.alpha.TopologyPriorityFormula(factor_dim=2),
        )


def test_typed_topology_versions_and_dependency_graph_are_explicit() -> None:
    bank = TypedTopologyOperandBank(slots=4, key_dim=3, factor_dim=2)
    formula = TypedTopologyPriorityFormula(factor_dim=2)
    policy = TypedBankFormulaTopologyPolicy(
        dim=4,
        banks=[bank],
        key_dim=3,
        formula=formula,
    )

    assert arti.component_ref(bank) == "arti/topology-operand-bank@2"
    assert arti.component_ref(formula) == "arti/topology-priority-formula@2"
    assert arti.component_ref(policy) == "arti/bank-formula-topology-policy@2"
    assert set(component_spec(policy).dependencies) == {
        "arti/fixed-topology-query@1",
        "arti/topology-operand-bank@2",
        "arti/topology-priority-formula@2",
    }


def test_typed_policy_drives_real_reversible_topology_and_bank_gradient() -> None:
    bank = TypedTopologyOperandBank(slots=6, key_dim=3, factor_dim=2)
    policy = TypedBankFormulaTopologyPolicy(dim=4, banks=[bank], key_dim=3)
    topology = arti.alpha.ReversibleTopology(
        active_count=2,
        policy=policy,
        surrogate=arti.alpha.SoftTopKTopologySurrogate(temperature=0.5),
    )
    x = torch.randn(2, 5, 4, requires_grad=True)
    mask = torch.tensor([[True, True, True, False, True], [True, True, True, True, True]])

    state = topology.fold(x, mask)
    restored = topology.unfold(state)

    torch.testing.assert_close(restored.value, x, rtol=0, atol=0)
    assert torch.equal(restored.mask, mask)
    state.active.square().sum().backward()
    assert bank.values.grad is not None
    assert torch.isfinite(bank.values.grad).all()
    assert float(bank.values.grad.abs().sum()) > 0
