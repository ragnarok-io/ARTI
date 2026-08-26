"""Typed Bank and Formula topology components for the ARTI vNext pipeline."""

from __future__ import annotations

import hashlib
import json
from typing import ClassVar, Sequence

import torch
from torch import Tensor

from .topology import (
    BankFormulaTopologyPolicy,
    TopologyFormulaOutput,
    TopologyOperandBank,
    TopologyPriorityFormula,
)
from .vnext_contracts import (
    OperandContract,
    OperandKind,
    OperandOwnership,
    SupportDomain,
    TypedOperands,
)


def _fingerprint(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class TypedTopologyOperandBank(TopologyOperandBank):
    """Topology Bank that emits consumer-bound typed operands."""

    _component_reference: ClassVar[str] = "arti/topology-operand-bank@2"

    def __init__(self, *args: object, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self._asset_fingerprint = _fingerprint(self.structure_contract)

    @property
    def asset_fingerprint(self) -> str:
        return self._asset_fingerprint

    def read_operands(
        self,
        query: Tensor,
        mask: Tensor,
        *,
        consumer: object,
        transition_id: str = "topology-read",
        ownership: OperandOwnership = OperandOwnership.BORROWED_TRAINING,
    ) -> tuple[TypedOperands, Tensor]:
        if query.ndim < 2 or query.shape[-1] != self.key_dim:
            raise ValueError(f"query must have shape [..., N, {self.key_dim}]")
        if mask.dtype != torch.bool or mask.shape != query.shape[:-1]:
            raise ValueError("mask must be boolean with shape query.shape[:-1]")
        values, route = self.read(query)
        domain = SupportDomain.for_tensor(
            mask,
            domain_id=f"{self.bank_id}-operands",
            owner_ref=self._component_reference,
            partition_id=self.bank_id,
            transition_id=transition_id,
            layout="dense",
        )
        from .component_registry import component_ref

        contract = OperandContract(
            kind=OperandKind.TOPOLOGY,
            source_ref=self._component_reference,
            partition_id=self.bank_id,
            consumer_ref=component_ref(consumer),
            factor_dim=self.factor_dim,
            layout="dense",
            domain=domain,
            source_asset_fingerprint=self.asset_fingerprint,
        )
        return TypedOperands(contract, values, mask, ownership=ownership), route


class TypedTopologyPriorityFormula(TopologyPriorityFormula):
    """Topology Formula that accepts only authorized typed operands."""

    _component_reference: ClassVar[str] = "arti/topology-priority-formula@2"

    def evaluate_operands(
        self,
        operands: TypedOperands,
        *,
        source: TypedTopologyOperandBank,
    ) -> TopologyFormulaOutput:
        if not isinstance(operands, TypedOperands):
            raise TypeError("TopologyPriorityFormula@2 requires TypedOperands")
        if not isinstance(source, TypedTopologyOperandBank):
            raise TypeError("TopologyPriorityFormula@2 requires TopologyOperandBank@2")
        contract = operands.contract
        values = operands.consume(
            consumer=self,
            kind=OperandKind.TOPOLOGY,
            source=source,
            partition_id=source.bank_id,
            domain=contract.domain,
            factor_dim=self.contract.factor_dim,
            layout="dense",
            source_asset_fingerprint=source.asset_fingerprint,
        )
        return super().evaluate(values)


class TypedBankFormulaTopologyPolicy(BankFormulaTopologyPolicy):
    """Fixed-Query topology policy whose Bank/Formula edge is typed."""

    _component_reference: ClassVar[str] = "arti/bank-formula-topology-policy@2"

    def __init__(
        self,
        dim: int,
        banks: Sequence[TypedTopologyOperandBank],
        **kwargs: object,
    ) -> None:
        if not banks or any(not isinstance(bank, TypedTopologyOperandBank) for bank in banks):
            raise TypeError("BankFormulaTopologyPolicy@2 requires TopologyOperandBank@2")
        formula = kwargs.get("formula")
        if formula is None:
            kwargs["formula"] = TypedTopologyPriorityFormula(banks[0].factor_dim)
        elif not isinstance(formula, TypedTopologyPriorityFormula):
            raise TypeError("BankFormulaTopologyPolicy@2 requires TopologyPriorityFormula@2")
        super().__init__(dim, banks, **kwargs)

    def bank_outputs(
        self,
        x: Tensor,
        mask: Tensor,
    ) -> tuple[tuple[TopologyFormulaOutput, ...], tuple[Tensor, ...]]:
        if x.ndim < 2 or x.shape[-1] != self.dim:
            raise ValueError(f"x must have shape [..., N, {self.dim}]")
        if mask.dtype != torch.bool or mask.shape != x.shape[:-1]:
            raise ValueError("mask must be boolean with shape x.shape[:-1]")
        source = torch.where(mask.unsqueeze(-1), x.detach(), torch.zeros_like(x))
        query = self.query(source)
        outputs = []
        routes = []
        for bank in self.banks:
            typed, route = bank.read_operands(query, mask, consumer=self.formula)
            outputs.append(self.formula.evaluate_operands(typed, source=bank))
            routes.append(route)
        return tuple(outputs), tuple(routes)

    def execution_outputs(
        self,
        x: Tensor,
        mask: Tensor,
    ) -> tuple[tuple[TopologyFormulaOutput, ...], tuple[Tensor, ...]]:
        """Compile-safe execution path for this construction-bound typed policy.

        The constructor fixes the typed Bank and Formula identities. Runtime
        tensor contracts are checked directly here so no registry lookup or
        Python TypedOperands envelope enters the compiled graph.
        """

        if x.ndim < 2 or x.shape[-1] != self.dim:
            raise ValueError(f"x must have shape [..., N, {self.dim}]")
        if mask.dtype != torch.bool or mask.shape != x.shape[:-1]:
            raise ValueError("mask must be boolean with shape x.shape[:-1]")
        source = torch.where(mask.unsqueeze(-1), x.detach(), torch.zeros_like(x))
        query = self.query(source)
        outputs = []
        routes = []
        for bank in self.banks:
            operands, route = bank.read(query)
            outputs.append(self.formula.evaluate(operands))
            routes.append(route)
        return tuple(outputs), tuple(routes)


__all__ = [
    "TypedBankFormulaTopologyPolicy",
    "TypedTopologyOperandBank",
    "TypedTopologyPriorityFormula",
]
