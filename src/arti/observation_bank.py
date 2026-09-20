"""Typed Bank operands for input-conditioned observation trajectories."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import ClassVar, Sequence

import torch
from torch import Tensor, nn

from .observation import ObservationPlan
from .runtime_contracts import (
    OperandContract,
    OperandKind,
    OperandOwnership,
    SupportDomain,
    TypedOperands,
)


def _fingerprint(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _registered_component_ref(value: object) -> str:
    from .component_registry import component_ref

    return component_ref(value)


def _canonical_contract_ref(reference: str) -> str:
    from .component_registry import canonical_contract_reference

    return canonical_contract_reference(reference)


class FixedObservationQuery(nn.Module):
    """Deterministic input and trajectory query for observation operand Banks."""

    _component_reference: ClassVar[str] = "arti/fixed-observation-query@1"

    def __init__(
        self,
        input_dim: int,
        key_dim: int,
        max_observations: int,
        *,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or key_dim <= 0 or max_observations <= 0:
            raise ValueError("query dimensions must be positive")
        self.input_dim = int(input_dim)
        self.key_dim = int(key_dim)
        self.max_observations = int(max_observations)
        self.seed = int(seed)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        input_basis = torch.randn(key_dim, input_dim, generator=generator) * input_dim**-0.5
        step_basis = torch.randn(max_observations, key_dim, generator=generator) * key_dim**-0.5
        self.register_buffer("input_basis", input_basis, persistent=True)
        self.register_buffer("step_basis", step_basis, persistent=True)

    def forward(self, substrate: Tensor, mask: Tensor) -> Tensor:
        if substrate.ndim != 3 or substrate.shape[-1] != self.input_dim:
            raise ValueError(
                f"substrate must have shape [B, N, {self.input_dim}]"
            )
        if mask.shape != substrate.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("mask must be boolean with shape [B, N]")
        source = torch.where(mask.unsqueeze(-1), substrate.detach(), torch.zeros_like(substrate))
        weights = mask.unsqueeze(-1).to(source.dtype)
        pooled = source.sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        input_query = torch.einsum("bd,kd->bk", pooled, self.input_basis.to(source))
        return input_query.unsqueeze(1) + self.step_basis.to(source).unsqueeze(0)

    def observation_query_contract(self) -> dict[str, object]:
        return {
            "ref": _canonical_contract_ref(self._component_reference),
            "input_dim": self.input_dim,
            "key_dim": self.key_dim,
            "max_observations": self.max_observations,
            "seed": self.seed,
            "fixed": True,
            "deterministic": True,
            "stateful": False,
        }


class ObservationOperandBank(nn.Module):
    """Fixed-address, trainable-value Bank for observation operands."""

    _component_reference: ClassVar[str] = "arti/observation-operand-bank@1"

    def __init__(
        self,
        slots: int,
        key_dim: int,
        factor_dim: int,
        *,
        seed: int = 0,
        value_seed: int | None = None,
        init_scale: float = 0.02,
        bank_id: str | None = None,
    ) -> None:
        super().__init__()
        if slots <= 0 or key_dim <= 0 or factor_dim <= 0:
            raise ValueError("Bank dimensions must be positive")
        if init_scale <= 0.0:
            raise ValueError("init_scale must be positive")
        self.slots = int(slots)
        self.key_dim = int(key_dim)
        self.factor_dim = int(factor_dim)
        self.seed = int(seed)
        self.value_seed = self.seed if value_seed is None else int(value_seed)
        self.bank_id = (
            f"observation-bank-{self.seed}-{self.value_seed}"
            if bank_id is None
            else str(bank_id)
        )
        if not self.bank_id:
            raise ValueError("bank_id must be non-empty")
        key_generator = torch.Generator(device="cpu").manual_seed(self.seed)
        value_generator = torch.Generator(device="cpu").manual_seed(self.value_seed)
        keys = torch.randn(slots, key_dim, generator=key_generator) * key_dim**-0.5
        values = torch.randn(slots, factor_dim, generator=value_generator) * init_scale
        self.register_buffer("keys", keys, persistent=True)
        self.values = nn.Parameter(values)
        self._asset_fingerprint = _fingerprint(self.structure_contract)

    @property
    def structure_contract(self) -> dict[str, object]:
        return {
            "ref": _canonical_contract_ref(self._component_reference),
            "bank_id": self.bank_id,
            "slots": self.slots,
            "key_dim": self.key_dim,
            "factor_dim": self.factor_dim,
            "seed": self.seed,
            "value_seed": self.value_seed,
        }

    @property
    def asset_fingerprint(self) -> str:
        return self._asset_fingerprint

    def _read_values(self, query: Tensor) -> tuple[Tensor, Tensor]:
        if query.ndim != 3 or query.shape[-1] != self.key_dim:
            raise ValueError(f"query must have shape [B, T, {self.key_dim}]")
        logits = torch.einsum("btk,sk->bts", query, self.keys.to(query))
        route = torch.softmax(logits * self.key_dim**-0.5, dim=-1)
        return torch.einsum("bts,sf->btf", route, self.values.to(query)), route

    def read_operands(
        self,
        query: Tensor,
        mask: Tensor,
        *,
        consumer: object,
        transition_id: str = "observation-read",
        ownership: OperandOwnership = OperandOwnership.BORROWED_TRAINING,
    ) -> tuple[TypedOperands, Tensor]:
        if mask.shape != query.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("mask must be boolean with shape [B, T]")
        values, route = self._read_values(query)
        domain = SupportDomain.for_tensor(
            mask,
            domain_id=f"{self.bank_id}-operands",
            owner_ref=_canonical_contract_ref(self._component_reference),
            partition_id=self.bank_id,
            transition_id=transition_id,
            layout="dense",
        )
        contract = OperandContract(
            kind=OperandKind.OBSERVATION,
            source_ref=_canonical_contract_ref(self._component_reference),
            partition_id=self.bank_id,
            consumer_ref=(
                getattr(consumer, "_component_reference", "")
                if torch.compiler.is_compiling()
                else _registered_component_ref(consumer)
            ),
            factor_dim=self.factor_dim,
            layout="dense",
            domain=domain,
            source_asset_fingerprint=self.asset_fingerprint,
        )
        return TypedOperands(contract, values, mask, ownership=ownership), route


@dataclass(frozen=True)
class ObservationFormulaOutput:
    states: Tensor
    continuation_logits: Tensor


class ObservationTrajectoryFormula(nn.Module):
    """Fixed interpretation of typed Bank factors as state and continuation."""

    _component_reference: ClassVar[str] = "arti/observation-trajectory-formula@1"

    def __init__(
        self,
        state_dim: int,
        *,
        state_scale: float = 1.0,
        continuation_scale: float = 1.0,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or state_scale <= 0.0 or continuation_scale <= 0.0:
            raise ValueError("Formula dimensions and scales must be positive")
        self.state_dim = int(state_dim)
        self.factor_dim = self.state_dim + 1
        self.state_scale = float(state_scale)
        self.continuation_scale = float(continuation_scale)

    def evaluate_operands(
        self,
        operands: TypedOperands,
        *,
        source: ObservationOperandBank,
    ) -> ObservationFormulaOutput:
        if not isinstance(operands, TypedOperands):
            raise TypeError("ObservationTrajectoryFormula requires TypedOperands")
        if not isinstance(source, ObservationOperandBank):
            raise TypeError("ObservationTrajectoryFormula requires ObservationOperandBank")
        contract = operands.contract
        values = operands.consume(
            consumer=self,
            kind=OperandKind.OBSERVATION,
            source=source,
            partition_id=source.bank_id,
            domain=contract.domain,
            factor_dim=self.factor_dim,
            layout="dense",
            source_asset_fingerprint=source.asset_fingerprint,
        )
        return self._evaluate_values(values)

    def _evaluate_values(self, values: Tensor) -> ObservationFormulaOutput:
        if values.ndim != 3 or values.shape[-1] != self.factor_dim:
            raise ValueError(
                f"observation factors must have shape [B, T, {self.factor_dim}]"
            )
        return ObservationFormulaOutput(
            torch.tanh(values[..., : self.state_dim]) * self.state_scale,
            values[..., self.state_dim] * self.continuation_scale,
        )


class BankConditionedObservationPolicy(nn.Module):
    """Bounded observation policy driven by fixed Query and typed Bank factors."""

    _component_reference: ClassVar[str] = "arti/bank-observation-policy@1"

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        max_observations: int,
        banks: Sequence[ObservationOperandBank],
        *,
        key_dim: int = 16,
        query_seed: int = 0,
        query: nn.Module | None = None,
        formula: ObservationTrajectoryFormula | None = None,
        bank_weights: Sequence[float] | None = None,
        min_observations: int = 1,
        stop_threshold: float = 0.5,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or state_dim <= 0 or max_observations <= 0 or key_dim <= 0:
            raise ValueError("policy dimensions must be positive")
        if not banks or any(not isinstance(bank, ObservationOperandBank) for bank in banks):
            raise TypeError("BankConditionedObservationPolicy requires observation Banks")
        if not 1 <= min_observations <= max_observations:
            raise ValueError("min_observations must be in [1, max_observations]")
        if not 0.0 < stop_threshold < 1.0 or temperature <= 0.0:
            raise ValueError("stop_threshold and temperature are invalid")
        expected_factor_dim = state_dim + 1
        if any(
            bank.key_dim != key_dim or bank.factor_dim != expected_factor_dim
            for bank in banks
        ):
            raise ValueError("all observation Banks must match key_dim and state_dim + 1")
        ids = [bank.bank_id for bank in banks]
        if len(ids) != len(set(ids)):
            raise ValueError("observation Bank IDs must be unique")
        self.input_dim = int(input_dim)
        self.state_dim = int(state_dim)
        self.max_observations = int(max_observations)
        self.key_dim = int(key_dim)
        self.query_seed = int(query_seed)
        self.min_observations = int(min_observations)
        self.stop_threshold = float(stop_threshold)
        self.temperature = float(temperature)
        self.banks = nn.ModuleList(banks)
        self.query = (
            FixedObservationQuery(input_dim, key_dim, max_observations, seed=query_seed)
            if query is None
            else query
        )
        if not isinstance(self.query, nn.Module):
            raise TypeError("observation Query must be an nn.Module")
        contract = getattr(self.query, "observation_query_contract", None)
        if not callable(contract):
            raise TypeError("observation Query must expose observation_query_contract()")
        declared = contract()
        if (
            declared.get("fixed") is not True
            or declared.get("deterministic") is not True
            or declared.get("stateful") is not False
            or any(parameter.requires_grad for parameter in self.query.parameters())
        ):
            raise ValueError("Bank observation Query must be fixed and deterministic")
        self.formula = formula or ObservationTrajectoryFormula(state_dim)
        if self.formula.state_dim != state_dim:
            raise ValueError("observation Formula state_dim does not match policy")
        weights = (
            torch.ones(len(banks), dtype=torch.float32)
            if bank_weights is None
            else torch.as_tensor(bank_weights, dtype=torch.float32)
        )
        if weights.shape != (len(banks),) or not bool(torch.isfinite(weights).all()):
            raise ValueError("bank_weights must be finite and match the Bank count")
        if float(weights.abs().sum()) == 0.0:
            raise ValueError("at least one bank weight must be non-zero")
        self.register_buffer("bank_weights", weights, persistent=True)

    def forward(self, substrate: Tensor, mask: Tensor) -> ObservationPlan:
        query = self.query(substrate, mask)
        route_mask = torch.ones(query.shape[:2], dtype=torch.bool, device=query.device)
        outputs = []
        for bank in self.banks:
            if torch.compiler.is_compiling():
                values, _ = bank._read_values(query)
                outputs.append(self.formula._evaluate_values(values))
            else:
                operands, _ = bank.read_operands(
                    query, route_mask, consumer=self.formula
                )
                outputs.append(self.formula.evaluate_operands(operands, source=bank))
        weights = self.bank_weights.to(query)
        denominator = weights.abs().sum().clamp_min(1e-8)
        states = sum(
            weight * output.states for weight, output in zip(weights, outputs, strict=True)
        ) / denominator
        continuation = sum(
            weight * output.continuation_logits
            for weight, output in zip(weights, outputs, strict=True)
        ) / denominator
        probability = torch.sigmoid(continuation / self.temperature)
        batch = substrate.shape[0]
        alive_hard = torch.ones(batch, dtype=torch.bool, device=substrate.device)
        alive_soft = torch.ones(batch, dtype=substrate.dtype, device=substrate.device)
        masks = []
        activity = []
        for index in range(self.max_observations):
            masks.append(alive_hard)
            hard = alive_hard.to(substrate.dtype)
            activity.append(hard + alive_soft - alive_soft.detach())
            if index + 1 >= self.max_observations:
                continue
            alive_soft = alive_soft * probability[:, index]
            if index + 1 >= self.min_observations:
                alive_hard = alive_hard & (
                    probability[:, index] >= self.stop_threshold
                )
        return ObservationPlan(
            states,
            torch.stack(masks, dim=1),
            torch.stack(activity, dim=1),
        )


__all__ = [
    "BankConditionedObservationPolicy",
    "FixedObservationQuery",
    "ObservationFormulaOutput",
    "ObservationOperandBank",
    "ObservationTrajectoryFormula",
]
