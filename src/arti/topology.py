"""Learnable operand sources for reversible computational topology.

Topology modules may decide which tensor instances are exposed, but they never
mix, summarize, or reconstruct tensor values.  A policy produces continuous
priority operands; a fixed operator turns them into a complete permutation.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import ClassVar, Sequence

import torch
from torch import Tensor, nn


def _tensor_hash(value: Tensor) -> str:
    payload = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class TopologyAction:
    """Continuous operands proposed for one topology operation."""

    priority: Tensor

    _component_reference: ClassVar[str] = "arti/topology-action@1"

    def __post_init__(self) -> None:
        if not isinstance(self.priority, Tensor) or not self.priority.is_floating_point():
            raise TypeError("TopologyAction priority must be a floating-point Tensor")
        if self.priority.ndim < 1:
            raise ValueError("TopologyAction priority must contain an instance axis")


@dataclass(frozen=True)
class TopologyProposal:
    """A continuous proposal awaiting validation by a topology operator."""

    action: TopologyAction


class StablePriorityPartition(nn.Module):
    """Convert priorities into a valid-first, stable, complete permutation."""

    _component_reference: ClassVar[str] = "arti/stable-priority-partition@1"

    def forward(self, action: TopologyAction, mask: Tensor) -> Tensor:
        priority = action.priority
        if mask.dtype != torch.bool or mask.shape != priority.shape:
            raise ValueError("mask must be boolean and match topology priority")
        if mask.device != priority.device:
            raise ValueError("mask and topology priority must share a device")
        compiler = getattr(torch, "compiler", None)
        is_compiling = bool(compiler is not None and compiler.is_compiling())
        finite = torch.isfinite(priority).all()
        if is_compiling:
            torch._assert_async(
                finite, "topology priority must contain only finite values"
            )
        elif not finite:
            raise ValueError("topology priority must contain only finite values")
        ranked = priority.masked_fill(~mask, -torch.inf)
        return torch.argsort(ranked, dim=-1, descending=True, stable=True)


class SoftTopKTopologySurrogate(nn.Module):
    """Backward-only active-set estimator for the first K hard positions.

    The returned assignment is never a value-path topology.  ReversibleTopology
    uses it only in a zero-valued autograd correction computed from detached
    tensor values.  A cardinality-constrained inclusion mass supplies the
    score gradient, while detached rank anchors keep the K surrogate rows
    position-sensitive without differentiating through a second sorter.
    """

    _component_reference: ClassVar[str] = "arti/topology-surrogate@1"

    def __init__(self, temperature: float = 0.25) -> None:
        super().__init__()
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        self.temperature = float(temperature)

    def forward(self, action: TopologyAction, mask: Tensor, active_count: int) -> Tensor:
        scores = action.priority
        length = scores.shape[-1]
        if not 1 <= active_count <= length:
            raise ValueError("active_count must be in [1, instance_count]")
        if mask.dtype != torch.bool or mask.shape != scores.shape:
            raise ValueError("mask must be boolean and match topology priority")

        compute_dtype = (
            torch.float32
            if scores.dtype in {torch.float16, torch.bfloat16}
            else scores.dtype
        )
        working_scores = scores.to(compute_dtype)
        validity = mask.to(compute_dtype)
        detached = working_scores.detach()
        valid_count = mask.sum(dim=-1, keepdim=True)
        budget = valid_count.clamp_max(active_count).to(compute_dtype)
        any_valid = mask.any(dim=-1, keepdim=True)

        positive_infinity = torch.full_like(detached, torch.inf)
        negative_infinity = torch.full_like(detached, -torch.inf)
        minimum = torch.where(mask, detached, positive_infinity).amin(
            dim=-1, keepdim=True
        )
        maximum = torch.where(mask, detached, negative_infinity).amax(
            dim=-1, keepdim=True
        )
        minimum = torch.where(any_valid, minimum, torch.zeros_like(minimum))
        maximum = torch.where(any_valid, maximum, torch.zeros_like(maximum))
        lower = minimum - 20.0 * self.temperature
        upper = maximum + 20.0 * self.temperature
        for _ in range(32):
            midpoint = (lower + upper) * 0.5
            count = (
                torch.sigmoid((detached - midpoint) / self.temperature) * validity
            ).sum(dim=-1, keepdim=True)
            lower = torch.where(count > budget, midpoint, lower)
            upper = torch.where(count > budget, upper, midpoint)
        threshold = (lower + upper) * 0.5
        inclusion = (
            torch.sigmoid((working_scores - threshold) / self.temperature) * validity
        )

        ranked_scores = detached.masked_fill(~mask, -torch.inf)
        rank_anchor = torch.topk(
            ranked_scores, active_count, dim=-1, sorted=True
        ).values
        rank_index = torch.arange(
            active_count, device=scores.device
        ).reshape((1,) * (scores.ndim - 1) + (active_count, 1))
        valid_rank = rank_index < valid_count.unsqueeze(-2)
        rank_anchor = torch.where(
            valid_rank.squeeze(-1), rank_anchor, torch.zeros_like(rank_anchor)
        )
        routing_temperature = max(self.temperature, 0.5)
        routing_logits = -(
            detached.unsqueeze(-2) - rank_anchor.unsqueeze(-1)
        ).abs() / routing_temperature
        routing_logits = routing_logits.masked_fill(
            ~mask.unsqueeze(-2), torch.finfo(compute_dtype).min
        )
        routing = torch.softmax(routing_logits, dim=-1)
        assignment = routing * (
            inclusion.unsqueeze(-2)
            / routing.sum(dim=-2, keepdim=True).clamp_min(1e-8)
        )
        assignment = assignment / assignment.sum(dim=-1, keepdim=True).clamp_min(
            1e-8
        )
        assignment = torch.where(
            any_valid.unsqueeze(-2) & valid_rank,
            assignment,
            torch.zeros_like(assignment),
        )
        return assignment.to(scores.dtype)

    def extra_repr(self) -> str:
        return f"temperature={self.temperature}"


class LearnedTopologyPolicy(nn.Module):
    """Permutation-equivariant learned priority policy.

    Tensor values are detached before scoring by default.  This keeps value
    gradients on the executed hard lineage while topology parameters learn
    through the explicitly declared surrogate.
    """

    _component_reference: ClassVar[str] = "arti/learned-topology-policy@1"

    def __init__(
        self,
        dim: int,
        *,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        width = max(8, dim) if hidden_dim is None else int(hidden_dim)
        if width <= 0:
            raise ValueError("hidden_dim must be positive")
        self.dim = int(dim)
        self.hidden_dim = width
        self.scorer = nn.Sequential(
            nn.Linear(dim * 2, width),
            nn.GELU(),
            nn.Linear(width, 1),
        )

    def forward(self, x: Tensor, mask: Tensor) -> TopologyProposal:
        if x.ndim < 2 or x.shape[-1] != self.dim:
            raise ValueError(f"x must have shape [..., N, {self.dim}]")
        if mask.dtype != torch.bool or mask.shape != x.shape[:-1]:
            raise ValueError("mask must be boolean with shape x.shape[:-1]")
        source = torch.where(mask.unsqueeze(-1), x.detach(), torch.zeros_like(x))
        weights = mask.unsqueeze(-1).to(source.dtype)
        context = (source * weights).sum(dim=-2, keepdim=True) / weights.sum(
            dim=-2, keepdim=True
        ).clamp_min(1)
        context = context.expand_as(source)
        action = TopologyAction(self.scorer(torch.cat((source, context), dim=-1)).squeeze(-1))
        return TopologyProposal(action)

    def topology_contract(self) -> dict[str, object]:
        return {
            "ref": self._component_reference,
            "dim": self.dim,
            "hidden_dim": self.hidden_dim,
            "value_input": "detached",
        }


@dataclass(frozen=True)
class TopologyFormulaContract:
    """Static contract for Formula modules that emit topology priority."""

    factor_dim: int
    mode: str = "linear"
    output_semantics: str = "topology_priority"
    api_version: int = 1

    def __post_init__(self) -> None:
        if self.factor_dim <= 0:
            raise ValueError("factor_dim must be positive")
        if self.mode not in {"linear", "gated_priority"}:
            raise ValueError("Topology Formula mode must be 'linear' or 'gated_priority'")
        if self.mode == "gated_priority" and self.factor_dim != 2:
            raise ValueError("gated_priority requires factors [priority, confidence]")
        if self.output_semantics != "topology_priority":
            raise ValueError("Topology Formula must emit topology_priority")
        if self.api_version != 1:
            raise ValueError("unsupported TopologyFormulaContract api_version")

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            {
                "factor_dim": self.factor_dim,
                "mode": self.mode,
                "output_semantics": self.output_semantics,
                "api_version": self.api_version,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class TopologyFormulaLock:
    """Code-free binding of one topology Formula implementation and state."""

    formula_ref: str
    contract_fingerprint: str
    factor_dim: int
    weight_hash: str
    lock_version: int = 1


@dataclass(frozen=True)
class TopologyFormulaOutput:
    """Typed Formula result before multiple topology Banks are composed."""

    priority: Tensor
    confidence: Tensor


class FixedTopologyQuery(nn.Module):
    """Deterministic, non-trainable query used by topology Banks."""

    _component_reference: ClassVar[str] = "arti/fixed-topology-query@1"

    def __init__(self, dim: int, key_dim: int, *, seed: int = 0) -> None:
        super().__init__()
        if dim <= 0 or key_dim <= 0:
            raise ValueError("dim and key_dim must be positive")
        self.dim = int(dim)
        self.key_dim = int(key_dim)
        self.seed = int(seed)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        basis = torch.randn(key_dim, dim, generator=generator) * dim**-0.5
        self.register_buffer("basis", basis, persistent=True)

    def forward(self, x: Tensor) -> Tensor:
        if x.shape[-1] != self.dim:
            raise ValueError(f"x must end with dim={self.dim}")
        return torch.einsum("...nd,kd->...nk", x.detach(), self.basis.to(x))

    def topology_contract(self) -> dict[str, object]:
        return {
            "ref": self._component_reference,
            "dim": self.dim,
            "key_dim": self.key_dim,
            "seed": self.seed,
            "basis_hash": _tensor_hash(self.basis),
            "fixed": True,
            "deterministic": True,
            "stateful": False,
        }


class TopologyPriorityFormula(nn.Module):
    """Affine interpretation of Bank operands as topology priority."""

    _component_reference: ClassVar[str] = "arti/topology-priority-formula@1"

    def __init__(
        self,
        factor_dim: int = 1,
        *,
        weight: Sequence[float] | Tensor | None = None,
        mode: str = "linear",
        trainable: bool = False,
    ) -> None:
        super().__init__()
        if factor_dim <= 0:
            raise ValueError("factor_dim must be positive")
        if mode == "gated_priority" and trainable:
            raise ValueError("gated_priority has fixed semantics and is not trainable")
        if weight is None:
            value = torch.ones(factor_dim) / factor_dim
        else:
            value = torch.as_tensor(weight, dtype=torch.float32)
            if value.shape != (factor_dim,):
                raise ValueError("weight must have shape [factor_dim]")
        self.contract = TopologyFormulaContract(factor_dim, mode=mode)
        if trainable:
            self.weight = nn.Parameter(value.clone())
        else:
            self.register_buffer("weight", value.clone(), persistent=True)
        self.trainable = bool(trainable)

    def evaluate(self, operands: Tensor) -> TopologyFormulaOutput:
        if operands.shape[-1] != self.contract.factor_dim:
            raise ValueError("operand factor dimension does not match Formula contract")
        if self.contract.mode == "gated_priority":
            priority = operands[..., 0]
            confidence = torch.sigmoid(operands[..., 1].mean(dim=-1, keepdim=True))
        else:
            priority = torch.einsum("...nf,f->...n", operands, self.weight.to(operands))
            confidence = torch.ones(
                *priority.shape[:-1], 1, device=priority.device, dtype=priority.dtype
            )
        return TopologyFormulaOutput(priority, confidence)

    def forward(self, operands: Tensor) -> TopologyAction:
        output = self.evaluate(operands)
        return TopologyAction(output.priority * output.confidence)

    @property
    def formula_lock(self) -> TopologyFormulaLock:
        return TopologyFormulaLock(
            formula_ref=self._component_reference,
            contract_fingerprint=self.contract.fingerprint,
            factor_dim=self.contract.factor_dim,
            weight_hash=_tensor_hash(self.weight),
        )


class TopologyOperandBank(nn.Module):
    """Fixed-address, trainable-value Bank of topology operands."""

    _component_reference: ClassVar[str] = "arti/topology-operand-bank@1"

    def __init__(
        self,
        slots: int,
        key_dim: int,
        factor_dim: int = 1,
        *,
        seed: int = 0,
        value_seed: int | None = None,
        init_scale: float = 0.02,
        bank_id: str | None = None,
    ) -> None:
        super().__init__()
        if slots <= 0 or key_dim <= 0 or factor_dim <= 0:
            raise ValueError("slots, key_dim, and factor_dim must be positive")
        self.slots = int(slots)
        self.key_dim = int(key_dim)
        self.factor_dim = int(factor_dim)
        self.seed = int(seed)
        self.value_seed = self.seed if value_seed is None else int(value_seed)
        self.bank_id = (
            f"topology-bank-{self.seed}-{self.value_seed}-{slots}x{key_dim}x{factor_dim}"
            if bank_id is None
            else str(bank_id)
        )
        if not self.bank_id:
            raise ValueError("bank_id must be non-empty")
        key_generator = torch.Generator(device="cpu").manual_seed(self.seed)
        value_generator = torch.Generator(device="cpu").manual_seed(self.value_seed)
        keys = torch.randn(slots, key_dim, generator=key_generator) * key_dim**-0.5
        values = (
            torch.randn(slots, factor_dim, generator=value_generator) * init_scale
        )
        self.register_buffer("keys", keys, persistent=True)
        self.values = nn.Parameter(values)

    def read(self, query: Tensor) -> tuple[Tensor, Tensor]:
        logits = torch.einsum("...nk,sk->...ns", query, self.keys.to(query))
        weights = torch.softmax(logits * self.key_dim**-0.5, dim=-1)
        return torch.einsum("...ns,sf->...nf", weights, self.values.to(query)), weights

    @property
    def asset_contract(self) -> dict[str, object]:
        return {
            **self.structure_contract,
            "values_hash": _tensor_hash(self.values),
        }

    @property
    def structure_contract(self) -> dict[str, object]:
        return {
            "ref": self._component_reference,
            "bank_id": self.bank_id,
            "slots": self.slots,
            "key_dim": self.key_dim,
            "factor_dim": self.factor_dim,
            "seed": self.seed,
            "value_seed": self.value_seed,
            "keys_hash": _tensor_hash(self.keys),
        }


class BankFormulaTopologyPolicy(nn.Module):
    """Use fixed Query, topology operand Banks, and Formula to propose priority."""

    _component_reference: ClassVar[str] = "arti/bank-formula-topology-policy@1"

    def __init__(
        self,
        dim: int,
        banks: Sequence[TopologyOperandBank],
        *,
        key_dim: int = 16,
        query_seed: int = 0,
        query: nn.Module | None = None,
        formula: nn.Module | None = None,
        bank_weights: Sequence[float] | None = None,
        diagnostics: str = "none",
        diagnostic_slot_limit: int = 4096,
    ) -> None:
        super().__init__()
        if dim <= 0 or key_dim <= 0 or not banks:
            raise ValueError("dim, key_dim, and at least one Bank are required")
        if any(bank.key_dim != key_dim for bank in banks):
            raise ValueError("all topology Banks must match key_dim")
        factor_dims = {bank.factor_dim for bank in banks}
        if len(factor_dims) != 1:
            raise ValueError("all topology Banks must match factor_dim")
        factor_dim = next(iter(factor_dims))
        bank_ids = [bank.bank_id for bank in banks]
        if len(set(bank_ids)) != len(bank_ids):
            raise ValueError("topology Bank IDs must be unique")
        if bank_weights is None:
            weights = torch.ones(len(banks))
        else:
            weights = torch.as_tensor(bank_weights, dtype=torch.float32)
            if weights.shape != (len(banks),):
                raise ValueError("bank_weights must match the Bank count")
        self.dim = int(dim)
        self.key_dim = int(key_dim)
        self.query_seed = int(query_seed)
        self.banks = nn.ModuleList(banks)
        self.query = (
            FixedTopologyQuery(dim, key_dim, seed=query_seed)
            if query is None
            else query
        )
        if not isinstance(self.query, nn.Module):
            raise TypeError("topology Query must be an nn.Module")
        query_contract = getattr(self.query, "topology_contract", None)
        if not callable(query_contract):
            raise TypeError("topology Query must expose topology_contract()")
        declared_query = query_contract()
        if declared_query.get("ref") != getattr(
            self.query, "_component_reference", None
        ):
            raise ValueError("topology Query contract identity does not match its component")
        if (
            declared_query.get("fixed") is not True
            or declared_query.get("deterministic") is not True
            or declared_query.get("stateful") is not False
        ):
            raise ValueError(
                "BankFormulaTopologyPolicy@1 requires a fixed deterministic Query contract"
            )
        if getattr(self.query, "dim", None) != dim or getattr(
            self.query, "key_dim", None
        ) != key_dim:
            raise ValueError("topology Query dimensions do not match the policy")
        if any(parameter.requires_grad for parameter in self.query.parameters()):
            raise ValueError("BankFormulaTopologyPolicy@1 requires a fixed Query")
        self.register_buffer("bank_weights", weights, persistent=True)
        self.formula = (
            TopologyPriorityFormula(factor_dim) if formula is None else formula
        )
        if not isinstance(self.formula, nn.Module):
            raise TypeError("topology Formula must be an nn.Module")
        formula_contract = getattr(self.formula, "contract", None)
        if not isinstance(formula_contract, TopologyFormulaContract):
            raise TypeError("topology Formula must expose TopologyFormulaContract@1")
        if formula_contract.factor_dim != factor_dim:
            raise ValueError("Formula factor_dim must match the topology Banks")
        if not callable(getattr(self.formula, "evaluate", None)):
            raise TypeError("topology Formula must expose evaluate(operands)")
        formula_lock = getattr(self.formula, "formula_lock", None)
        if not isinstance(formula_lock, TopologyFormulaLock):
            raise TypeError("topology Formula must expose TopologyFormulaLock@1")
        if formula_lock.formula_ref != getattr(
            self.formula, "_component_reference", None
        ):
            raise ValueError("topology Formula lock identity does not match its component")
        if (
            formula_lock.contract_fingerprint != formula_contract.fingerprint
            or formula_lock.factor_dim != formula_contract.factor_dim
            or formula_lock.lock_version != 1
            or not formula_lock.weight_hash
        ):
            raise ValueError("topology Formula lock does not match its contract")
        if diagnostics not in {"none", "summary"}:
            raise ValueError("diagnostics must be 'none' or 'summary'")
        if diagnostic_slot_limit <= 0:
            raise ValueError("diagnostic_slot_limit must be positive")
        if diagnostics == "summary" and sum(bank.slots for bank in banks) > diagnostic_slot_limit:
            raise ValueError(
                "topology diagnostic summaries exceed diagnostic_slot_limit"
            )
        self.diagnostics = diagnostics
        self.diagnostic_slot_limit = int(diagnostic_slot_limit)
        self._last_route_summary: tuple[Tensor, ...] = ()
        self._last_confidence_summary: tuple[Tensor, ...] = ()

    def bank_outputs(
        self, x: Tensor, mask: Tensor
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
            operands, route = bank.read(query)
            outputs.append(self.formula.evaluate(operands))
            routes.append(route)
        return tuple(outputs), tuple(routes)

    def forward(self, x: Tensor, mask: Tensor) -> TopologyProposal:
        if x.ndim < 2 or x.shape[-1] != self.dim:
            raise ValueError(f"x must have shape [..., N, {self.dim}]")
        if mask.dtype != torch.bool or mask.shape != x.shape[:-1]:
            raise ValueError("mask must be boolean with shape x.shape[:-1]")
        priority = torch.zeros(*x.shape[:-1], device=x.device, dtype=x.dtype)
        bank_outputs, route_weights = self.bank_outputs(x, mask)
        for weight, output in zip(self.bank_weights, bank_outputs, strict=True):
            priority = priority + weight.to(x) * output.confidence * output.priority
        compiler = getattr(torch, "compiler", None)
        if self.diagnostics == "summary" and not (
            compiler is not None and compiler.is_compiling()
        ):
            self._last_route_summary = tuple(
                route.detach()
                .mean(dim=tuple(range(route.ndim - 1)))
                .to(device="cpu")
                for route in route_weights
            )
            self._last_confidence_summary = tuple(
                output.confidence.detach().mean().reshape(1).to(device="cpu")
                for output in bank_outputs
            )
        action = TopologyAction(priority)
        return TopologyProposal(action)

    @property
    def last_route_summary(self) -> tuple[Tensor, ...]:
        return self._last_route_summary

    @property
    def last_confidence_summary(self) -> tuple[Tensor, ...]:
        return self._last_confidence_summary

    def clear_diagnostics(self) -> None:
        """Release opt-in route snapshots retained by the latest eager call."""

        self._last_route_summary = ()
        self._last_confidence_summary = ()

    def topology_contract(self) -> dict[str, object]:
        query_contract = self.query.topology_contract()
        formula_lock = self.formula.formula_lock
        formula_is_trainable = any(
            parameter.requires_grad for parameter in self.formula.parameters()
        )
        return {
            "ref": self._component_reference,
            "dim": self.dim,
            "key_dim": self.key_dim,
            "query_seed": self.query_seed,
            "bank_count": len(self.banks),
            "banks": [
                bank.structure_contract
                for bank in self.banks
            ],
            "bank_weights": self.bank_weights.detach().cpu().tolist(),
            "query": query_contract,
            "formula": {
                "ref": formula_lock.formula_ref,
                "contract_fingerprint": self.formula.contract.fingerprint,
                "mode": self.formula.contract.mode,
                "lock_version": formula_lock.lock_version,
                "weight_hash": (
                    None if formula_is_trainable else formula_lock.weight_hash
                ),
            },
        }


__all__ = [
    "BankFormulaTopologyPolicy",
    "FixedTopologyQuery",
    "LearnedTopologyPolicy",
    "SoftTopKTopologySurrogate",
    "StablePriorityPartition",
    "TopologyAction",
    "TopologyProposal",
    "TopologyFormulaContract",
    "TopologyFormulaLock",
    "TopologyFormulaOutput",
    "TopologyOperandBank",
    "TopologyPriorityFormula",
]
