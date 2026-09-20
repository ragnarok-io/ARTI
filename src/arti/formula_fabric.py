"""Bounded, typed Formula execution for controlled alpha experiments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Sequence

import torch
from torch import Tensor, nn

from .runtime_contracts import ContractLimits, DEFAULT_CONTRACT_LIMITS


FORMULA_FABRIC_PROGRAM_SCHEMA_VERSION = 1
FORMULA_FABRIC_TRACE_SCHEMA_VERSION = 1


def _canonical_contract_ref(reference: str) -> str:
    """Resolve a source declaration before it enters a receipt or fingerprint."""

    # Compiled route receipts are ephemeral graph values. Resolving through the
    # registry here would capture its Python lock; eager artifact construction
    # remains the canonical persistence boundary.
    if torch.compiler.is_compiling():
        return reference
    from .component_registry import canonical_contract_reference

    return canonical_contract_reference(reference)


class FormulaPrimitive(str, Enum):
    """Closed set of pure, shape-preserving Formula primitives."""

    IDENTITY = "identity"
    ADD = "add"
    SUBTRACT = "subtract"
    MULTIPLY = "multiply"
    SIN = "sin"
    COS = "cos"

    @property
    def arity(self) -> int:
        return 1 if self in {self.IDENTITY, self.SIN, self.COS} else 2


@dataclass(frozen=True)
class FormulaInvocation:
    """One statically declared Formula cell and its destination slot."""

    primitive: FormulaPrimitive
    output_slot: int

    def __post_init__(self) -> None:
        if not isinstance(self.primitive, FormulaPrimitive):
            raise TypeError("primitive must be a FormulaPrimitive")
        if (
            isinstance(self.output_slot, bool)
            or not isinstance(self.output_slot, int)
            or self.output_slot < 0
        ):
            raise ValueError("output_slot must be a non-negative integer")

    def to_dict(self) -> dict[str, object]:
        return {"primitive": self.primitive.value, "output_slot": self.output_slot}


@dataclass(frozen=True)
class FormulaFabricProgram:
    """Immutable, bounded Formula schedule with synchronous SSA steps."""

    arena_capacity: int
    feature_dim: int
    steps: tuple[tuple[FormulaInvocation, ...], ...]
    domain: str = "anonymous"
    schema_version: int = FORMULA_FABRIC_PROGRAM_SCHEMA_VERSION

    def __post_init__(self) -> None:
        try:
            normalized_steps = tuple(tuple(step) for step in self.steps)
        except TypeError as exc:
            raise TypeError("steps must be a sequence of Formula steps") from exc
        object.__setattr__(self, "steps", normalized_steps)
        for value, name in (
            (self.arena_capacity, "arena_capacity"),
            (self.feature_dim, "feature_dim"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.schema_version != FORMULA_FABRIC_PROGRAM_SCHEMA_VERSION:
            raise ValueError("unsupported FormulaFabricProgram schema_version")
        if not isinstance(self.domain, str) or not self.domain:
            raise ValueError("domain must be a non-empty string")
        if not self.steps:
            raise ValueError("FormulaFabricProgram requires at least one step")
        for step in self.steps:
            if not step:
                raise ValueError("FormulaFabricProgram steps must not be empty")
            if any(not isinstance(cell, FormulaInvocation) for cell in step):
                raise TypeError("steps must contain FormulaInvocation values")
            outputs = [cell.output_slot for cell in step]
            if len(outputs) != len(set(outputs)):
                raise ValueError("one synchronous step cannot write a slot twice")
            if any(slot >= self.arena_capacity for slot in outputs):
                raise ValueError("Formula output_slot exceeds arena_capacity")

    @property
    def max_cells(self) -> int:
        return max(len(step) for step in self.steps)

    @property
    def max_arity(self) -> int:
        return max(cell.primitive.arity for step in self.steps for cell in step)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "arena_capacity": self.arena_capacity,
            "feature_dim": self.feature_dim,
            "domain": self.domain,
            "steps": [
                [cell.to_dict() for cell in step]
                for step in self.steps
            ],
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class FormulaArenaState:
    """Fixed-capacity value arena and its logical SSA versions."""

    value: Tensor
    mask: Tensor
    version: Tensor
    domain: str = "anonymous"

    def __post_init__(self) -> None:
        if not isinstance(self.value, Tensor) or self.value.ndim != 3:
            raise ValueError("FormulaArenaState value must have shape [B, S, D]")
        if not self.value.is_floating_point():
            raise TypeError("FormulaArenaState value must be floating point")
        if (
            not isinstance(self.mask, Tensor)
            or self.mask.dtype != torch.bool
            or self.mask.shape != self.value.shape[:-1]
            or self.mask.device != self.value.device
        ):
            raise ValueError("arena mask must be boolean [B, S] on the value device")
        if (
            not isinstance(self.version, Tensor)
            or self.version.dtype != torch.int64
            or self.version.shape != self.value.shape[:-1]
            or self.version.device != self.value.device
        ):
            raise ValueError("arena version must be int64 [B, S] on the value device")
        if not isinstance(self.domain, str) or not self.domain:
            raise ValueError("arena domain must be a non-empty string")

    @classmethod
    def from_tensor(
        cls,
        value: Tensor,
        mask: Tensor,
        *,
        capacity: int,
        domain: str = "anonymous",
    ) -> FormulaArenaState:
        if value.ndim != 3:
            raise ValueError("source value must have shape [B, N, D]")
        if capacity < value.shape[1]:
            raise ValueError("capacity cannot be smaller than source length")
        if mask.dtype != torch.bool or mask.shape != value.shape[:-1]:
            raise ValueError("source mask must be boolean [B, N]")
        padding = capacity - value.shape[1]
        padded_value = torch.nn.functional.pad(value, (0, 0, 0, padding))
        padded_mask = torch.nn.functional.pad(mask, (0, padding), value=False)
        version = torch.zeros(
            value.shape[0], capacity, dtype=torch.int64, device=value.device
        )
        return cls(padded_value, padded_mask, version, domain)


@dataclass(frozen=True)
class FormulaRoutePlan:
    """Hard forward routes with explicit valid, fire, and commit masks."""

    weights: Tensor
    valid_mask: Tensor
    fire_mask: Tensor
    commit_mask: Tensor
    estimator: str = "hard"

    def __post_init__(self) -> None:
        if not isinstance(self.weights, Tensor) or self.weights.ndim != 5:
            raise ValueError("route weights must have shape [B, R, C, A, S]")
        if not self.weights.is_floating_point():
            raise TypeError("route weights must be floating point")
        expected = self.weights.shape[:3]
        for value, name in (
            (self.valid_mask, "valid_mask"),
            (self.fire_mask, "fire_mask"),
            (self.commit_mask, "commit_mask"),
        ):
            if (
                not isinstance(value, Tensor)
                or value.dtype != torch.bool
                or value.shape != expected
                or value.device != self.weights.device
            ):
                raise ValueError(f"{name} must be boolean [B, R, C] on route device")
        owned_tensors = (
            (self.weights, "route weights"),
            (self.valid_mask, "route valid mask"),
            (self.fire_mask, "route fire mask"),
            (self.commit_mask, "route commit mask"),
        )
        for value, name in owned_tensors:
            DEFAULT_CONTRACT_LIMITS.admit_tensor(value, name=name)
        ownership_bytes = sum(
            value.numel() * value.element_size() for value, _name in owned_tensors
        )
        if ownership_bytes > DEFAULT_CONTRACT_LIMITS.max_operation_bytes:
            raise ValueError("FormulaRoutePlan ownership exceeds max_operation_bytes")
        object.__setattr__(self, "weights", self.weights.clone())
        object.__setattr__(self, "valid_mask", self.valid_mask.detach().clone())
        object.__setattr__(self, "fire_mask", self.fire_mask.detach().clone())
        object.__setattr__(self, "commit_mask", self.commit_mask.detach().clone())
        checks = (
            (
                ~(self.fire_mask & ~self.valid_mask).any(),
                "fire_mask must be a subset of valid_mask",
            ),
            (
                ~(self.commit_mask & ~self.fire_mask).any(),
                "commit_mask must be a subset of fire_mask",
            ),
        )
        for condition, message in checks:
            if self.weights.device.type == "cpu":
                if not bool(condition):
                    raise ValueError(message)
            else:
                torch._assert_async(condition, message)
        if self.estimator not in {"hard", "straight-through"}:
            raise ValueError("estimator must be 'hard' or 'straight-through'")


@dataclass(frozen=True)
class FormulaFabricTrace:
    """Fixed-shape provenance for one Formula Fabric execution."""

    formula_id: Tensor
    input_slot: Tensor
    input_version: Tensor
    output_slot: Tensor
    output_version: Tensor
    valid_mask: Tensor
    fire_mask: Tensor
    commit_mask: Tensor
    program_fingerprint: str
    estimator: str
    schema_version: int = FORMULA_FABRIC_TRACE_SCHEMA_VERSION


@dataclass(frozen=True)
class FormulaFabricResult:
    state: FormulaArenaState
    trace: FormulaFabricTrace


@dataclass(frozen=True)
class FormulaCommitBlendTrace:
    """Formula provenance plus the bounded continuous commit operand."""

    formula: FormulaFabricTrace
    weights: Tensor
    _component_reference: ClassVar[str] = "arti/formula-commit-blend-trace@1"


@dataclass(frozen=True)
class FormulaCommitBlendResult:
    state: FormulaArenaState
    trace: FormulaCommitBlendTrace


def straight_through_route(logits: Tensor) -> Tensor:
    """Return one-hot forward routes with softmax surrogate gradients."""

    if not isinstance(logits, Tensor) or not logits.is_floating_point():
        raise TypeError("route logits must be a floating Tensor")
    if logits.ndim != 5:
        raise ValueError("route logits must have shape [B, R, C, A, S]")
    return _HardRoute.apply(logits)


class _HardRoute(torch.autograd.Function):
    """Exact hard forward with the softmax Jacobian as explicit surrogate."""

    @staticmethod
    def forward(logits: Tensor) -> Tensor:
        soft = torch.softmax(logits, dim=-1)
        index = soft.argmax(dim=-1)
        return torch.nn.functional.one_hot(index, logits.shape[-1]).to(logits.dtype)

    @staticmethod
    def setup_context(ctx: object, inputs: tuple[Tensor], output: Tensor) -> None:
        del output
        (logits,) = inputs
        ctx.save_for_backward(torch.softmax(logits, dim=-1))

    @staticmethod
    def backward(ctx: object, grad_output: Tensor) -> tuple[Tensor]:
        (soft,) = ctx.saved_tensors
        centered = grad_output - (grad_output * soft).sum(dim=-1, keepdim=True)
        return (soft * centered,)


def _apply_primitive(primitive: FormulaPrimitive, operands: Tensor) -> Tensor:
    first = operands[:, 0]
    if primitive is FormulaPrimitive.IDENTITY:
        return first
    if primitive is FormulaPrimitive.ADD:
        return first + operands[:, 1]
    if primitive is FormulaPrimitive.SUBTRACT:
        return first - operands[:, 1]
    if primitive is FormulaPrimitive.MULTIPLY:
        return first * operands[:, 1]
    if primitive is FormulaPrimitive.SIN:
        return first.sin()
    if primitive is FormulaPrimitive.COS:
        return first.cos()
    raise AssertionError(f"unhandled Formula primitive {primitive}")


class FormulaFabric(nn.Module):
    """Execute a closed, fixed-capacity Formula program over a tensor arena."""

    _component_reference: ClassVar[str] = "arti/formula-fabric@1"

    def __init__(
        self,
        program: FormulaFabricProgram,
        *,
        limits: ContractLimits = DEFAULT_CONTRACT_LIMITS,
    ) -> None:
        super().__init__()
        if not isinstance(program, FormulaFabricProgram):
            raise TypeError("FormulaFabric requires FormulaFabricProgram")
        if not isinstance(limits, ContractLimits):
            raise TypeError("limits must be ContractLimits")
        for name, hard_value in DEFAULT_CONTRACT_LIMITS.__dict__.items():
            if getattr(limits, name) > hard_value:
                raise ValueError(f"FormulaFabric limits cannot relax {name}")
        if program.arena_capacity > limits.max_dimension:
            raise ValueError("Formula arena_capacity exceeds max_dimension")
        if program.feature_dim > limits.max_dimension:
            raise ValueError("Formula feature_dim exceeds max_dimension")
        if len(program.steps) > limits.max_stages:
            raise ValueError("Formula program exceeds max_stages")
        if program.arena_capacity * program.feature_dim > limits.max_elements:
            raise ValueError("Formula arena exceeds max_elements")
        static_table_elements = len(program.steps) * program.max_cells
        if static_table_elements > limits.max_elements:
            raise ValueError("Formula static table exceeds max_elements")
        if static_table_elements * 8 > limits.max_tensor_bytes:
            raise ValueError("Formula static table exceeds max_tensor_bytes")
        if static_table_elements * 8 * 3 > limits.max_operation_bytes:
            raise ValueError("Formula static tables exceed max_operation_bytes")
        self.program = program
        self._limits = limits
        self._program_fingerprint = program.fingerprint

        formula_ids = torch.full(
            (len(program.steps), program.max_cells), -1, dtype=torch.int64
        )
        output_slots = torch.full_like(formula_ids, -1)
        step_output_slots = torch.full_like(formula_ids, 0)
        for step_index, step in enumerate(program.steps):
            for cell_index, cell in enumerate(step):
                formula_ids[step_index, cell_index] = list(FormulaPrimitive).index(
                    cell.primitive
                )
                output_slots[step_index, cell_index] = cell.output_slot
                step_output_slots[step_index, cell_index] = cell.output_slot

        self.register_buffer("_formula_ids", formula_ids, persistent=False)
        self.register_buffer("_output_slots", output_slots, persistent=False)
        self.register_buffer("_step_output_slots", step_output_slots, persistent=False)

    @property
    def limits(self) -> ContractLimits:
        return self._limits

    @staticmethod
    def _require_tensor(condition: Tensor, message: str) -> None:
        if torch.compiler.is_compiling() or condition.device.type != "cpu":
            torch._assert_async(condition, message)
        elif not bool(condition):
            raise ValueError(message)

    def _admit_runtime(
        self,
        state: FormulaArenaState,
        route: FormulaRoutePlan,
        commit_weights: Tensor | None = None,
    ) -> None:
        for value, name in (
            (state.value, "Formula arena value"),
            (state.mask, "Formula arena mask"),
            (state.version, "Formula arena version"),
            (route.weights, "Formula route weights"),
            (route.valid_mask, "Formula route valid mask"),
            (route.fire_mask, "Formula route fire mask"),
            (route.commit_mask, "Formula route commit mask"),
        ):
            self.limits.admit_tensor(value, name=name)
        commit_weight_bytes = 0
        if commit_weights is not None:
            self.limits.admit_tensor(
                commit_weights,
                name="Formula continuous commit weights",
            )
            commit_weight_bytes = commit_weights.numel() * commit_weights.element_size()
        trace_elements = (
            route.valid_mask.numel() * 5
            + route.valid_mask.shape[0]
            * route.valid_mask.shape[1]
            * route.valid_mask.shape[2]
            * self.program.max_arity
            * 2
        )
        operation_bytes = (
            state.value.numel()
            * state.value.element_size()
            * (4 + 3 * len(self.program.steps))
            + state.mask.numel() * state.mask.element_size() * 3
            + state.version.numel() * state.version.element_size() * 4
            + route.weights.numel() * route.weights.element_size() * 2
            + commit_weight_bytes * 2
            + trace_elements * 8
        )
        if operation_bytes > self.limits.max_operation_bytes:
            raise ValueError("FormulaFabric exceeds max_operation_bytes")

    def _validate(
        self,
        state: FormulaArenaState,
        route: FormulaRoutePlan,
        commit_weights: Tensor | None = None,
    ) -> None:
        if not isinstance(state, FormulaArenaState):
            raise TypeError("state must be FormulaArenaState")
        if not isinstance(route, FormulaRoutePlan):
            raise TypeError("route must be FormulaRoutePlan")
        if state.value.shape[1:] != (
            self.program.arena_capacity,
            self.program.feature_dim,
        ):
            raise ValueError("arena shape does not match FormulaFabricProgram")
        if state.domain != self.program.domain:
            raise ValueError("arena domain does not match FormulaFabricProgram")
        expected = (
            state.value.shape[0],
            len(self.program.steps),
            self.program.max_cells,
            self.program.max_arity,
            self.program.arena_capacity,
        )
        if route.weights.shape != expected:
            raise ValueError(f"route weights must have shape {expected}")
        if route.weights.device != state.value.device or route.weights.dtype != state.value.dtype:
            raise ValueError("route weights must share arena device and dtype")
        self._admit_runtime(state, route, commit_weights)
        detached = route.weights.detach()
        self._require_tensor(
            ~(route.fire_mask & ~route.valid_mask).any(),
            "fire_mask must be a subset of valid_mask",
        )
        self._require_tensor(
            ~(route.commit_mask & ~route.fire_mask).any(),
            "commit_mask must be a subset of fire_mask",
        )
        self._require_tensor(torch.isfinite(detached).all(), "route weights must be finite")
        safe_value = torch.where(
            state.mask.unsqueeze(-1), state.value.detach(), torch.zeros_like(state.value)
        )
        self._require_tensor(
            torch.isfinite(safe_value).all(),
            "valid Formula arena values must be finite",
        )
        one_hot = ((detached == 0) | (detached == 1)).all() & (
            detached.sum(dim=-1) == 1
        ).all()
        self._require_tensor(one_hot, "route forward values must be exactly one-hot")
        for step_index, step in enumerate(self.program.steps):
            if len(step) == self.program.max_cells:
                continue
            padded = route.valid_mask[:, step_index, len(step) :]
            self._require_tensor(
                ~padded.any(), "padded Formula cells must be invalid"
            )

    def forward(
        self,
        state: FormulaArenaState,
        route: FormulaRoutePlan,
    ) -> FormulaFabricResult:
        return self._execute(state, route, commit_weights=None)

    def _execute(
        self,
        state: FormulaArenaState,
        route: FormulaRoutePlan,
        *,
        commit_weights: Tensor | None,
    ) -> FormulaFabricResult:
        if commit_weights is not None:
            if not isinstance(route, FormulaRoutePlan):
                raise TypeError("route must be FormulaRoutePlan")
            expected = route.commit_mask.shape
            if (
                not isinstance(commit_weights, Tensor)
                or not commit_weights.is_floating_point()
                or commit_weights.shape != expected
                or commit_weights.device != state.value.device
                or commit_weights.dtype != state.value.dtype
            ):
                raise ValueError(
                    "commit_weights must be floating "
                    f"{expected} on the arena device with dtype {state.value.dtype}; "
                    f"got shape={getattr(commit_weights, 'shape', None)}, "
                    f"device={getattr(commit_weights, 'device', None)}, "
                    f"dtype={getattr(commit_weights, 'dtype', None)}"
                )
        self._validate(state, route, commit_weights)
        if commit_weights is not None:
            detached_commit = commit_weights.detach()
            self._require_tensor(
                torch.isfinite(detached_commit).all()
                & (detached_commit >= 0).all()
                & (detached_commit <= 1).all(),
                "commit_weights must be finite and within [0, 1]",
            )
        value, mask, version = state.value, state.mask, state.version
        batch = value.shape[0]
        formula_ids = self._formula_ids.clone()
        output_slots = self._output_slots.clone()
        output_versions = formula_ids.new_full(
            (batch, len(self.program.steps), self.program.max_cells), -1
        )
        input_slots = formula_ids.new_full(
            (
                batch,
                len(self.program.steps),
                self.program.max_cells,
                self.program.max_arity,
            ),
            -1,
        )
        input_versions = input_slots.clone()

        for step_index, step in enumerate(self.program.steps):
            base_value, base_mask, base_version = value, mask, version
            routed_value = torch.where(
                base_mask.unsqueeze(-1), base_value, torch.zeros_like(base_value)
            )
            committed_values: list[Tensor] = []
            committed_masks: list[Tensor] = []
            for cell_index, cell in enumerate(step):
                weights = route.weights[
                    :, step_index, cell_index, : cell.primitive.arity
                ]
                selected_slot = weights.detach().argmax(dim=-1)
                input_slots[
                    :, step_index, cell_index, : cell.primitive.arity
                ] = selected_slot
                input_versions[
                    :, step_index, cell_index, : cell.primitive.arity
                ] = torch.gather(base_version, 1, selected_slot)
                operands = torch.einsum("bas,bsd->bad", weights, routed_value)
                selected_valid = torch.gather(base_mask, 1, selected_slot).all(dim=-1)
                fire = route.fire_mask[:, step_index, cell_index]
                self._require_tensor(
                    ~(fire & ~selected_valid).any(),
                    "fired Formula cell selected an invalid arena value",
                )
                candidate = _apply_primitive(cell.primitive, operands)
                self._require_tensor(
                    ~(fire.unsqueeze(-1) & ~torch.isfinite(candidate)).any(),
                    "fired Formula cell produced a non-finite value",
                )
                commit = route.commit_mask[:, step_index, cell_index]
                old = base_value[:, cell.output_slot]
                safe_old = torch.where(
                    base_mask[:, cell.output_slot].unsqueeze(-1),
                    old,
                    torch.zeros_like(old),
                )
                if commit_weights is None:
                    committed = candidate
                else:
                    alpha = commit_weights[:, step_index, cell_index].unsqueeze(-1)
                    blended = (1 - alpha) * safe_old + alpha * candidate
                    endpoint_exact = torch.where(
                        alpha == 0,
                        safe_old,
                        torch.where(alpha == 1, candidate, blended),
                    )
                    committed = blended + (endpoint_exact - blended).detach()
                committed_values.append(
                    torch.where(commit.unsqueeze(-1), committed, old)
                )
                committed_masks.append(commit | base_mask[:, cell.output_slot])
            output_index = self._step_output_slots[step_index, : len(step)]
            value = torch.index_copy(value, 1, output_index, torch.stack(committed_values, dim=1))
            mask = torch.index_copy(mask, 1, output_index, torch.stack(committed_masks, dim=1))
            committed = route.commit_mask[:, step_index, : len(step)].to(torch.int64)
            next_versions = version[:, output_index] + committed
            version = torch.index_copy(version, 1, output_index, next_versions)
            output_versions[:, step_index, : len(step)] = next_versions

        trace = FormulaFabricTrace(
            formula_ids,
            input_slots,
            input_versions,
            output_slots,
            output_versions,
            route.valid_mask.detach().clone(),
            route.fire_mask.detach().clone(),
            route.commit_mask.detach().clone(),
            self._program_fingerprint,
            route.estimator,
        )
        return FormulaFabricResult(
            FormulaArenaState(value, mask, version, state.domain),
            trace,
        )

    @torch.no_grad()
    def reference(
        self,
        state: FormulaArenaState,
        route: FormulaRoutePlan,
    ) -> FormulaArenaState:
        """Slow independent interpreter used only for parity validation."""

        self._validate(state, route)
        value = state.value.detach().clone()
        mask = state.mask.detach().clone()
        version = state.version.detach().clone()
        indices = route.weights.detach().argmax(dim=-1)
        for step_index, step in enumerate(self.program.steps):
            previous_value = value.clone()
            previous_mask = mask.clone()
            for batch_index in range(value.shape[0]):
                for cell_index, cell in enumerate(step):
                    selected = indices[
                        batch_index,
                        step_index,
                        cell_index,
                        : cell.primitive.arity,
                    ]
                    if bool(route.fire_mask[batch_index, step_index, cell_index]):
                        if not bool(previous_mask[batch_index, selected].all()):
                            raise ValueError(
                                "fired Formula cell selected an invalid arena value"
                            )
                        operands = previous_value[batch_index, selected].unsqueeze(0)
                        candidate = _apply_primitive(cell.primitive, operands)[0]
                        if not bool(torch.isfinite(candidate).all()):
                            raise ValueError("fired Formula cell produced a non-finite value")
                        if bool(route.commit_mask[batch_index, step_index, cell_index]):
                            value[batch_index, cell.output_slot] = candidate
                            mask[batch_index, cell.output_slot] = True
            for cell_index, cell in enumerate(step):
                committed = route.commit_mask[:, step_index, cell_index]
                version[:, cell.output_slot] += committed.to(torch.int64)
        return FormulaArenaState(value, mask, version, state.domain)


class FormulaCommitBlend(nn.Module):
    """Apply bounded continuous strength to real Formula commits.

    Routing, fire and commit authority remain hard. A committed cell writes
    ``old + alpha * (candidate - old)`` with one alpha in ``[0, 1]``.
    """

    _component_reference: ClassVar[str] = "arti/formula-commit-blend@1"

    def __init__(self, fabric: FormulaFabric) -> None:
        super().__init__()
        if not isinstance(fabric, FormulaFabric):
            raise TypeError("fabric must be FormulaFabric@1")
        self.fabric = fabric
        self.program = fabric.program

    @property
    def limits(self) -> ContractLimits:
        return self.fabric.limits

    @property
    def _output_slots(self) -> Tensor:
        return self.fabric._output_slots

    def forward(
        self,
        state: FormulaArenaState,
        route: FormulaRoutePlan,
        commit_weights: Tensor,
    ) -> FormulaCommitBlendResult:
        result = self.fabric._execute(
            state,
            route,
            commit_weights=commit_weights,
        )
        return FormulaCommitBlendResult(
            result.state,
            FormulaCommitBlendTrace(
                result.trace,
                commit_weights.detach().clone(),
            ),
        )


class FormulaFabricCompute(nn.Module):
    """Apply a caller-owned Formula route to a folded Pulse workspace."""

    _component_reference: ClassVar[str] = "arti/formula-fabric-compute@1"

    def __init__(
        self,
        fabric: FormulaFabric | FormulaCommitBlend,
        *,
        active_count: int,
    ) -> None:
        super().__init__()
        if not isinstance(fabric, (FormulaFabric, FormulaCommitBlend)):
            raise TypeError("fabric must be FormulaFabric or FormulaCommitBlend")
        if isinstance(active_count, bool) or not isinstance(active_count, int) or active_count <= 0:
            raise ValueError("active_count must be a positive integer")
        if active_count > fabric.program.arena_capacity:
            raise ValueError("active_count cannot exceed Formula arena_capacity")
        self.fabric = fabric
        self.active_count = active_count
        self.commit_mode = (
            "weighted" if isinstance(fabric, FormulaCommitBlend) else "hard"
        )
        self.factor_contract = (
            "required" if isinstance(fabric, FormulaCommitBlend) else "forbidden"
        )
        self.route_contract = "external"
        self.arena_layout = "active-prefix-then-scratch"
        self.visibility_contract = "unsupported"
        from .component_registry import component_spec

        execution_config = {
            "ref": _canonical_contract_ref(self._component_reference),
            "fabric_ref": _canonical_contract_ref(fabric._component_reference),
            "fabric_config_fingerprint": component_spec(fabric).config_fingerprint,
            "program_fingerprint": fabric.program.fingerprint,
            "limits": dict(fabric.limits.__dict__),
            "active_count": active_count,
            "commit_mode": self.commit_mode,
            "factor_contract": self.factor_contract,
            "route_contract": self.route_contract,
            "arena_layout": self.arena_layout,
            "visibility": self.visibility_contract,
        }
        encoded = json.dumps(
            execution_config,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.execution_config_fingerprint = hashlib.sha256(
            encoded.encode("utf-8")
        ).hexdigest()

    def operation_bytes_upper_bound(
        self,
        workspace: object,
        factors: Tensor | None = None,
    ) -> int:
        """Return a conservative byte-work bound without executing Formula routes."""

        from .formula_attention import ActiveWorkspace

        if not isinstance(workspace, ActiveWorkspace):
            raise TypeError("FormulaFabricCompute requires ActiveWorkspace")
        if workspace.value.shape[-2:] != (
            self.active_count,
            self.fabric.program.feature_dim,
        ):
            raise ValueError("active workspace does not match FormulaFabricCompute")
        flat_batch = workspace.value.numel() // (
            self.active_count * workspace.value.shape[-1]
        )
        program = self.fabric.program
        capacity = program.arena_capacity
        arena_elements = flat_batch * capacity * program.feature_dim
        support_elements = flat_batch * capacity
        value_bytes = arena_elements * workspace.value.element_size()
        mask_bytes = support_elements
        version_bytes = support_elements * 8
        route_elements = (
            flat_batch
            * len(program.steps)
            * program.max_cells
            * program.max_arity
            * capacity
        )
        factor_bytes = (
            0 if factors is None else factors.numel() * factors.element_size()
        )
        valid_elements = flat_batch * len(program.steps) * program.max_cells
        trace_elements = valid_elements * 5 + valid_elements * program.max_arity * 2
        return int(
            value_bytes * (4 + 3 * len(program.steps))
            + mask_bytes * 3
            + version_bytes * 4
            + route_elements * workspace.value.element_size() * 2
            + factor_bytes * 2
            + trace_elements * 8
        )

    @staticmethod
    def _require_tensor(condition: Tensor, message: str) -> None:
        if torch.compiler.is_compiling() or condition.device.type != "cpu":
            torch._assert_async(condition, message)
        elif not bool(condition):
            raise ValueError(message)

    def forward(
        self,
        workspace: object,
        factors: Tensor | None = None,
        *,
        visibility: Tensor | None = None,
        formula_route: FormulaRoutePlan | None = None,
        return_info: bool = False,
    ) -> object:
        from .formula_attention import ActiveWorkspace

        if not isinstance(workspace, ActiveWorkspace):
            raise TypeError("FormulaFabricCompute requires ActiveWorkspace")
        if isinstance(self.fabric, FormulaCommitBlend):
            if factors is None:
                raise ValueError("FormulaCommitBlend requires compute_factors")
        elif factors is not None:
            raise ValueError("FormulaFabric does not consume compute_factors")
        if visibility is not None:
            raise ValueError("FormulaFabricCompute does not consume visibility")
        if not isinstance(formula_route, FormulaRoutePlan):
            raise ValueError("FormulaFabricCompute requires formula_route")
        if workspace.value.shape[-2:] != (
            self.active_count,
            self.fabric.program.feature_dim,
        ):
            raise ValueError("active workspace does not match FormulaFabricCompute")
        self.fabric.limits.admit_tensor(
            workspace.value, name="FormulaFabricCompute active workspace"
        )
        prefix = workspace.value.shape[:-2]
        flat_batch = workspace.value.numel() // (
            self.active_count * workspace.value.shape[-1]
        )
        active = workspace.value.reshape(
            flat_batch, self.active_count, workspace.value.shape[-1]
        )
        exposed = workspace.exposed.reshape(flat_batch, self.active_count)
        intervened = workspace.intervened.reshape(flat_batch, self.active_count)
        limits = self.fabric.limits
        capacity = self.fabric.program.arena_capacity
        feature_dim = self.fabric.program.feature_dim
        arena_elements = flat_batch * capacity * feature_dim
        support_elements = flat_batch * capacity
        if flat_batch > limits.max_dimension or arena_elements > limits.max_elements:
            raise ValueError("FormulaFabricCompute arena exceeds allocation limits")
        value_bytes = arena_elements * workspace.value.element_size()
        mask_bytes = support_elements
        version_bytes = support_elements * 8
        if max(value_bytes, mask_bytes, version_bytes) > limits.max_tensor_bytes:
            raise ValueError("FormulaFabricCompute arena exceeds tensor byte limits")
        for tensor, name in (
            (formula_route.weights, "Formula route weights"),
            (formula_route.valid_mask, "Formula route valid mask"),
            (formula_route.fire_mask, "Formula route fire mask"),
            (formula_route.commit_mask, "Formula route commit mask"),
        ):
            limits.admit_tensor(tensor, name=name)
        factor_bytes = 0
        if factors is not None:
            limits.admit_tensor(factors, name="Formula continuous commit weights")
            factor_bytes = factors.numel() * factors.element_size()
        trace_elements = (
            formula_route.valid_mask.numel() * 5
            + formula_route.valid_mask.numel() * self.fabric.program.max_arity * 2
        )
        operation_bytes = (
            value_bytes * (4 + 3 * len(self.fabric.program.steps))
            + mask_bytes * 3
            + version_bytes * 4
            + formula_route.weights.numel()
            * formula_route.weights.element_size()
            * 2
            + factor_bytes * 2
            + trace_elements * 8
        )
        if operation_bytes > limits.max_operation_bytes:
            raise ValueError("FormulaFabricCompute exceeds operation byte limits")
        state = FormulaArenaState.from_tensor(
            active,
            exposed,
            capacity=self.fabric.program.arena_capacity,
            domain=self.fabric.program.domain,
        )
        expected_route_shape = (
            flat_batch,
            len(self.fabric.program.steps),
            self.fabric.program.max_cells,
        )
        if formula_route.commit_mask.shape != expected_route_shape:
            raise ValueError(
                f"formula_route masks must have shape {expected_route_shape}"
            )

        output_slots = self.fabric._output_slots.clamp_min(0)
        output_authority = torch.gather(
            intervened,
            1,
            output_slots.reshape(1, -1).expand(flat_batch, -1).clamp_max(
                self.active_count - 1
            ),
        ).reshape(flat_batch, *output_slots.shape)
        writes_active = self.fabric._output_slots.unsqueeze(0) < self.active_count
        unauthorized = (
            formula_route.commit_mask
            & formula_route.valid_mask
            & writes_active
            & ~output_authority
        )
        self._require_tensor(
            ~unauthorized.any(),
            "Formula commit targets a slot without intervention authority",
        )

        if isinstance(self.fabric, FormulaCommitBlend):
            assert factors is not None
            result = self.fabric(state, formula_route, factors)
        else:
            result = self.fabric(state, formula_route)
        restored_mask = result.state.mask[:, : self.active_count]
        self._require_tensor(
            torch.eq(restored_mask, exposed).all(),
            "Formula Fabric changed active exposure support",
        )
        value = result.state.value[:, : self.active_count].reshape(
            *prefix, self.active_count, workspace.value.shape[-1]
        )
        updated = workspace.replace(value=value)
        if not return_info:
            return updated
        return updated, result.trace


@dataclass(frozen=True)
class BankFormulaRouteInfo:
    """Per-call diagnostics produced by a Bank/Formula route source."""

    selected_input_slots: Tensor
    valid_mask: Tensor
    fire_mask: Tensor
    commit_mask: Tensor
    availability: Tensor
    route_source_ref: str
    route_source_config_fingerprint: str


@dataclass(frozen=True)
class RoutedFormulaFabricComputeInfo:
    """Diagnostics for one routed Formula Fabric execution."""

    trace: FormulaFabricTrace | FormulaCommitBlendTrace
    route_origin: str
    route_source_ref: str | None
    route_source_config_fingerprint: str | None
    route: BankFormulaRouteInfo | None
    executor_ref: str
    executor_config_fingerprint: str
    adapter_ref: str
    adapter_config_fingerprint: str
    commit_mode: str
    factor_contract: str
    route_contract: str
    arena_layout: str
    visibility_contract: str


@dataclass(frozen=True)
class IterativeRoutedFormulaFabricComputeInfo:
    """Ordered route and execution records for state-conditioned iterations."""

    iterations: tuple[RoutedFormulaFabricComputeInfo, ...]
    configured_steps: int
    executed_steps: int
    adapter_ref: str
    adapter_config_fingerprint: str
    route_semantics: str


class BankFormulaRouteSource(nn.Module):
    """Generate a bounded FormulaRoutePlan from fixed-Query Bank/Formula policies.

    Policies are ordered by the program's live operand pins: step, cell, then
    operand. The source plans one complete Fabric invocation before execution;
    it does not re-query from intermediate Formula values.
    """

    _component_reference: ClassVar[str] = "arti/bank-formula-route-source@1"

    def __init__(
        self,
        program: FormulaFabricProgram,
        policies: Sequence[nn.Module],
        *,
        active_count: int,
        estimator: str = "straight-through",
        candidate_mask: Tensor | None = None,
        limits: ContractLimits = DEFAULT_CONTRACT_LIMITS,
    ) -> None:
        super().__init__()
        from .typed_topology import TypedBankFormulaTopologyPolicy

        if not isinstance(program, FormulaFabricProgram):
            raise TypeError("program must be FormulaFabricProgram")
        if not isinstance(limits, ContractLimits):
            raise TypeError("limits must be ContractLimits")
        for name, hard_value in DEFAULT_CONTRACT_LIMITS.__dict__.items():
            if getattr(limits, name) > hard_value:
                raise ValueError(f"route source limits cannot relax {name}")
        if (
            isinstance(active_count, bool)
            or not isinstance(active_count, int)
            or active_count <= 0
            or active_count > program.arena_capacity
        ):
            raise ValueError("active_count must be within the Formula arena")
        if estimator not in {"hard", "straight-through"}:
            raise ValueError("estimator must be 'hard' or 'straight-through'")
        pin_count = sum(
            cell.primitive.arity for step in program.steps for cell in step
        )
        if len(policies) != pin_count:
            raise ValueError(f"policies must contain exactly {pin_count} operand pins")
        if any(not isinstance(policy, TypedBankFormulaTopologyPolicy) for policy in policies):
            raise TypeError("route source requires BankFormulaTopologyPolicy@2 policies")
        if any(policy.dim != program.feature_dim for policy in policies):
            raise ValueError("route policy dimensions must match program feature_dim")
        if any(policy.diagnostics != "none" for policy in policies):
            raise ValueError("route policies must use diagnostics='none'")

        expected_mask = (
            len(program.steps),
            program.max_cells,
            program.max_arity,
            program.arena_capacity,
        )
        if candidate_mask is None:
            candidate_mask = torch.ones(expected_mask, dtype=torch.bool)
        elif (
            not isinstance(candidate_mask, Tensor)
            or candidate_mask.dtype != torch.bool
            or tuple(candidate_mask.shape) != expected_mask
            or candidate_mask.device.type != "cpu"
        ):
            raise ValueError(
                "candidate_mask must be a CPU boolean [R, C, A, S] Tensor"
            )
        candidate_mask = candidate_mask.clone()
        valid = torch.zeros(
            len(program.steps), program.max_cells, dtype=torch.bool
        )
        output_slots = torch.zeros_like(valid, dtype=torch.int64)
        for step_index, step in enumerate(program.steps):
            for cell_index, cell in enumerate(step):
                valid[step_index, cell_index] = True
                output_slots[step_index, cell_index] = cell.output_slot
                candidate_mask[
                    step_index, cell_index, cell.primitive.arity :
                ] = False
            candidate_mask[step_index, len(step) :] = False

        self.program = program
        self.active_count = active_count
        self.estimator = estimator
        self._limits = limits
        self.policies = nn.ModuleList(policies)
        from .component_registry import component_spec

        self._policy_config_fingerprints = tuple(
            component_spec(policy).config_fingerprint for policy in policies
        )
        policy_work_elements = 0
        max_policy_tensor_elements = 0
        for policy in policies:
            policy_work_elements += program.arena_capacity * (
                policy.key_dim + 1
            )
            for bank in policy.banks:
                bank_elements = program.arena_capacity * (
                    bank.slots + bank.factor_dim
                )
                policy_work_elements += bank_elements
                max_policy_tensor_elements = max(
                    max_policy_tensor_elements,
                    program.arena_capacity * bank.slots,
                )
        self._policy_work_elements_per_batch = policy_work_elements
        self._max_policy_tensor_elements_per_batch = max_policy_tensor_elements
        self.register_buffer("_candidate_mask", candidate_mask, persistent=False)
        self.register_buffer("_valid_cells", valid, persistent=False)
        self.register_buffer("_output_slots", output_slots, persistent=False)
        config = {
            "program_fingerprint": program.fingerprint,
            "active_count": active_count,
            "estimator": estimator,
            "policy_refs": [
                _canonical_contract_ref(policy._component_reference)
                for policy in policies
            ],
            "policy_config_fingerprints": list(self._policy_config_fingerprints),
            "candidate_mask_hash": hashlib.sha256(
                candidate_mask.to(torch.uint8).numpy().tobytes()
            ).hexdigest(),
            "limits": dict(limits.__dict__),
        }
        encoded = json.dumps(config, sort_keys=True, separators=(",", ":"))
        self.config_fingerprint = hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @property
    def limits(self) -> ContractLimits:
        return self._limits

    @property
    def candidate_mask(self) -> Tensor:
        """Return an inspection copy of the immutable candidate allowlist."""

        return self._candidate_mask.detach().clone()

    def _admit(self, workspace: object) -> tuple[Tensor, Tensor, Tensor]:
        from .formula_attention import ActiveWorkspace

        if not isinstance(workspace, ActiveWorkspace):
            raise TypeError("BankFormulaRouteSource requires ActiveWorkspace")
        if workspace.value.shape[-2:] != (
            self.active_count,
            self.program.feature_dim,
        ):
            raise ValueError("active workspace does not match route source")
        flat_batch = workspace.value.numel() // (
            self.active_count * self.program.feature_dim
        )
        value = workspace.value.reshape(
            flat_batch, self.active_count, self.program.feature_dim
        )
        exposed = workspace.exposed.reshape(flat_batch, self.active_count)
        intervened = workspace.intervened.reshape(flat_batch, self.active_count)
        route_elements = (
            flat_batch
            * len(self.program.steps)
            * self.program.max_cells
            * self.program.max_arity
            * self.program.arena_capacity
        )
        for tensor, name in (
            (workspace.value, "Formula route workspace value"),
            (workspace.validity, "Formula route workspace validity"),
            (workspace.exposed, "Formula route workspace exposure"),
            (workspace.intervened, "Formula route workspace intervention"),
        ):
            self.limits.admit_tensor(tensor, name=name)
        if route_elements > self.limits.max_elements:
            raise ValueError("Formula route source exceeds allocation limits")
        route_bytes = route_elements * workspace.value.element_size()
        arena_elements = (
            flat_batch * self.program.arena_capacity * self.program.feature_dim
        )
        arena_bytes = arena_elements * workspace.value.element_size()
        policy_elements = flat_batch * self._policy_work_elements_per_batch
        policy_bytes = policy_elements * workspace.value.element_size()
        largest_policy_tensor = (
            flat_batch * self._max_policy_tensor_elements_per_batch
        )
        if largest_policy_tensor > self.limits.max_elements:
            raise ValueError("Formula route Bank query exceeds allocation limits")
        if (
            largest_policy_tensor * workspace.value.element_size()
            > self.limits.max_tensor_bytes
        ):
            raise ValueError("Formula route Bank query exceeds tensor byte limits")
        if route_bytes > self.limits.max_tensor_bytes:
            raise ValueError("Formula route source exceeds tensor byte limits")
        cell_elements = (
            flat_batch * len(self.program.steps) * self.program.max_cells
        )
        diagnostic_bytes = (
            route_elements * 8
            + cell_elements * 3
            + flat_batch
            * (len(self.program.steps) + 1)
            * self.program.arena_capacity
        )
        operation_bytes = (
            arena_bytes * 3
            + route_bytes * 4
            + policy_bytes * 3
            + diagnostic_bytes
        )
        if operation_bytes > self.limits.max_operation_bytes:
            raise ValueError("Formula route source exceeds operation byte limits")
        return value, exposed, intervened

    def operation_bytes_upper_bound(self, workspace: object) -> int:
        """Return a conservative byte-work bound without querying any Bank."""

        from .formula_attention import ActiveWorkspace

        if not isinstance(workspace, ActiveWorkspace):
            raise TypeError("BankFormulaRouteSource requires ActiveWorkspace")
        if workspace.value.shape[-2:] != (
            self.active_count,
            self.program.feature_dim,
        ):
            raise ValueError("active workspace does not match route source")
        flat_batch = workspace.value.numel() // (
            self.active_count * self.program.feature_dim
        )
        route_elements = (
            flat_batch
            * len(self.program.steps)
            * self.program.max_cells
            * self.program.max_arity
            * self.program.arena_capacity
        )
        route_bytes = route_elements * workspace.value.element_size()
        arena_bytes = (
            flat_batch
            * self.program.arena_capacity
            * self.program.feature_dim
            * workspace.value.element_size()
        )
        policy_bytes = (
            flat_batch
            * self._policy_work_elements_per_batch
            * workspace.value.element_size()
        )
        cell_elements = flat_batch * len(self.program.steps) * self.program.max_cells
        diagnostic_bytes = (
            route_elements * 8
            + cell_elements * 3
            + flat_batch
            * (len(self.program.steps) + 1)
            * self.program.arena_capacity
        )
        return int(
            arena_bytes * 3
            + route_bytes * 4
            + policy_bytes * 3
            + diagnostic_bytes
        )

    @staticmethod
    def _require_tensor(condition: Tensor, message: str) -> None:
        if torch.compiler.is_compiling() or condition.device.type != "cpu":
            torch._assert_async(condition, message)
        elif not bool(condition):
            raise ValueError(message)

    def forward(
        self, workspace: object
    ) -> tuple[FormulaRoutePlan, BankFormulaRouteInfo]:
        value, exposed, intervened = self._admit(workspace)
        batch = value.shape[0]
        capacity = self.program.arena_capacity
        scratch = value.new_zeros(
            batch, capacity - self.active_count, self.program.feature_dim
        )
        arena = torch.cat((value, scratch), dim=1)
        availability = torch.cat(
            (
                exposed,
                torch.zeros(
                    batch,
                    capacity - self.active_count,
                    dtype=torch.bool,
                    device=value.device,
                ),
            ),
            dim=1,
        )
        logits = value.new_zeros(
            batch,
            len(self.program.steps),
            self.program.max_cells,
            self.program.max_arity,
            capacity,
        )
        valid = self._valid_cells.to(device=value.device).unsqueeze(0).expand(
            batch, -1, -1
        )
        fire_steps: list[Tensor] = []
        commit_steps: list[Tensor] = []
        availability_steps: list[Tensor] = [availability]
        minimum = torch.finfo(value.dtype).min
        policy_offset = 0

        for step_index, step in enumerate(self.program.steps):
            cell_fire: list[Tensor] = []
            cell_commit: list[Tensor] = []
            for cell_index, cell in enumerate(step):
                operand_available: list[Tensor] = []
                for operand_index in range(cell.primitive.arity):
                    policy = self.policies[policy_offset]
                    policy_offset += 1
                    priority = torch.zeros(
                        *arena.shape[:-1], device=value.device, dtype=value.dtype
                    )
                    bank_outputs, _bank_routes = policy.execution_outputs(
                        arena, availability
                    )
                    for bank_weight, output in zip(
                        policy.bank_weights, bank_outputs, strict=True
                    ):
                        priority = (
                            priority
                            + bank_weight.to(value)
                            * output.confidence
                            * output.priority
                        )
                    allowed = (
                        availability
                        & self._candidate_mask[
                            step_index, cell_index, operand_index
                        ].to(device=value.device)
                    )
                    self._require_tensor(
                        (~allowed | torch.isfinite(priority)).all(),
                        "Formula route priority must be finite on candidate slots",
                    )
                    operand_available.append(allowed.any(dim=-1))
                    logits[:, step_index, cell_index, operand_index] = torch.where(
                        allowed, priority, torch.full_like(priority, minimum)
                    )
                can_fire = torch.stack(operand_available, dim=-1).all(dim=-1)
                output_slot = cell.output_slot
                if output_slot < self.active_count:
                    authorized = intervened[:, output_slot]
                else:
                    authorized = torch.ones_like(can_fire)
                cell_fire.append(can_fire)
                cell_commit.append(can_fire & authorized)
            step_fire = torch.stack(cell_fire, dim=-1)
            step_commit = torch.stack(cell_commit, dim=-1)
            if len(step) < self.program.max_cells:
                pad = self.program.max_cells - len(step)
                step_fire = torch.nn.functional.pad(step_fire, (0, pad), value=False)
                step_commit = torch.nn.functional.pad(
                    step_commit, (0, pad), value=False
                )
            fire_steps.append(step_fire)
            commit_steps.append(step_commit)
            committed_slots = self._output_slots[step_index, : len(step)].to(
                device=value.device
            )
            availability = availability.scatter(
                1,
                committed_slots.reshape(1, -1).expand(batch, -1),
                torch.gather(availability, 1, committed_slots.reshape(1, -1).expand(batch, -1))
                | step_commit[:, : len(step)],
            )
            availability_steps.append(availability)

        if self.estimator == "straight-through":
            weights = straight_through_route(logits)
        else:
            index = logits.argmax(dim=-1)
            weights = torch.nn.functional.one_hot(index, capacity).to(value.dtype)
        fire = torch.stack(fire_steps, dim=1) & valid
        commit = torch.stack(commit_steps, dim=1) & fire
        plan = FormulaRoutePlan(weights, valid, fire, commit, self.estimator)
        info = BankFormulaRouteInfo(
            weights.detach().argmax(dim=-1),
            valid.detach().clone(),
            fire.detach().clone(),
            commit.detach().clone(),
            torch.stack(availability_steps, dim=1).detach().clone(),
            _canonical_contract_ref(self._component_reference),
            self.config_fingerprint,
        )
        return plan, info


class RoutedFormulaFabricCompute(nn.Module):
    """Thin adapter that supplies Bank/Formula routes to FormulaFabricCompute@1."""

    _component_reference: ClassVar[str] = "arti/routed-formula-fabric-compute@1"

    def __init__(
        self,
        compute: FormulaFabricCompute,
        route_source: BankFormulaRouteSource,
    ) -> None:
        super().__init__()
        if not isinstance(compute, FormulaFabricCompute):
            raise TypeError("compute must be FormulaFabricCompute@1")
        if not isinstance(route_source, BankFormulaRouteSource):
            raise TypeError("route_source must be BankFormulaRouteSource@1")
        if compute.fabric.program.fingerprint != route_source.program.fingerprint:
            raise ValueError("compute and route source program fingerprints differ")
        if compute.active_count != route_source.active_count:
            raise ValueError("compute and route source active_count differ")
        self.compute = compute
        self.route_source = route_source
        self.active_count = compute.active_count
        adapter_config = {
            "ref": _canonical_contract_ref(self._component_reference),
            "compute": compute.execution_config_fingerprint,
            "route_source": route_source.config_fingerprint,
            "active_count": self.active_count,
            "route_contract": "bound-source-or-explicit-override",
        }
        encoded = json.dumps(
            adapter_config,
            sort_keys=True,
            separators=(",", ":"),
        )
        self.config_fingerprint = hashlib.sha256(
            encoded.encode("utf-8")
        ).hexdigest()

    def forward(
        self,
        workspace: object,
        factors: Tensor | None = None,
        *,
        visibility: Tensor | None = None,
        formula_route: FormulaRoutePlan | None = None,
        return_info: bool = False,
    ) -> object:
        route_info: BankFormulaRouteInfo | None = None
        if formula_route is None:
            route, route_info = self.route_source(workspace)
            origin = "bank-formula"
        else:
            route = formula_route
            origin = "explicit"
        updated, trace = self.compute(
            workspace,
            factors,
            visibility=visibility,
            formula_route=route,
            return_info=True,
        )
        if not return_info:
            return updated
        return updated, RoutedFormulaFabricComputeInfo(
            trace=trace,
            route_origin=origin,
            route_source_ref=(
                None if route_info is None else route_info.route_source_ref
            ),
            route_source_config_fingerprint=(
                None
                if route_info is None
                else route_info.route_source_config_fingerprint
            ),
            route=route_info,
            executor_ref=_canonical_contract_ref(self.compute._component_reference),
            executor_config_fingerprint=self.compute.execution_config_fingerprint,
            adapter_ref=_canonical_contract_ref(self._component_reference),
            adapter_config_fingerprint=self.config_fingerprint,
            commit_mode=self.compute.commit_mode,
            factor_contract=self.compute.factor_contract,
            route_contract="bound-source-or-explicit-override",
            arena_layout=self.compute.arena_layout,
            visibility_contract=self.compute.visibility_contract,
        )


class IterativeRoutedFormulaFabricCompute(nn.Module):
    """Re-route after every complete Formula program execution.

    This is a composition adapter, not a second Formula executor. Each
    iteration invokes the same :class:`RoutedFormulaFabricCompute`; the updated
    workspace is then used to produce the next route.
    """

    _component_reference: ClassVar[str] = (
        "arti/iterative-routed-formula-fabric-compute@1"
    )

    def __init__(
        self,
        routed: RoutedFormulaFabricCompute,
        *,
        steps: int,
    ) -> None:
        super().__init__()
        if not isinstance(routed, RoutedFormulaFabricCompute):
            raise TypeError("routed must be RoutedFormulaFabricCompute@1")
        if (
            isinstance(steps, bool)
            or not isinstance(steps, int)
            or steps <= 0
            or steps > DEFAULT_CONTRACT_LIMITS.max_stages
        ):
            raise ValueError("steps must be within the bounded stage limit")
        self.routed = routed
        self.steps = steps
        self.active_count = routed.active_count
        config = {
            "ref": _canonical_contract_ref(self._component_reference),
            "routed_config_fingerprint": routed.config_fingerprint,
            "steps": steps,
            "route_semantics": "requery-after-program",
            "executor_reuse": True,
        }
        encoded = json.dumps(config, sort_keys=True, separators=(",", ":"))
        self.config_fingerprint = hashlib.sha256(
            encoded.encode("utf-8")
        ).hexdigest()

    def forward(
        self,
        workspace: object,
        factors: Tensor | None = None,
        *,
        visibility: Tensor | None = None,
        formula_route: FormulaRoutePlan | None = None,
        return_info: bool = False,
    ) -> object:
        if formula_route is not None:
            raise ValueError(
                "iterative routed compute does not accept one frozen formula_route"
            )
        route_bytes = self.routed.route_source.operation_bytes_upper_bound(workspace)
        compute_bytes = self.routed.compute.operation_bytes_upper_bound(
            workspace,
            factors,
        )
        total_bytes = self.steps * (route_bytes + compute_bytes)
        operation_limit = min(
            self.routed.route_source.limits.max_operation_bytes,
            self.routed.compute.fabric.limits.max_operation_bytes,
        )
        if total_bytes > operation_limit:
            raise ValueError(
                "iterative routed compute exceeds cumulative operation byte limits"
            )
        current = workspace
        records: list[RoutedFormulaFabricComputeInfo] = []
        for _step in range(self.steps):
            if return_info:
                current, info = self.routed(
                    current,
                    factors,
                    visibility=visibility,
                    return_info=True,
                )
                records.append(info)
            else:
                current = self.routed(
                    current,
                    factors,
                    visibility=visibility,
                    return_info=False,
                )
        if not return_info:
            return current
        return current, IterativeRoutedFormulaFabricComputeInfo(
            iterations=tuple(records),
            configured_steps=self.steps,
            executed_steps=self.steps,
            adapter_ref=_canonical_contract_ref(self._component_reference),
            adapter_config_fingerprint=self.config_fingerprint,
            route_semantics="requery-after-program",
        )


__all__ = [
    "FORMULA_FABRIC_PROGRAM_SCHEMA_VERSION",
    "FORMULA_FABRIC_TRACE_SCHEMA_VERSION",
    "FormulaArenaState",
    "FormulaFabric",
    "FormulaFabricCompute",
    "FormulaCommitBlend",
    "FormulaCommitBlendResult",
    "FormulaCommitBlendTrace",
    "FormulaFabricProgram",
    "FormulaFabricResult",
    "FormulaFabricTrace",
    "BankFormulaRouteInfo",
    "BankFormulaRouteSource",
    "FormulaInvocation",
    "FormulaPrimitive",
    "FormulaRoutePlan",
    "RoutedFormulaFabricCompute",
    "RoutedFormulaFabricComputeInfo",
    "IterativeRoutedFormulaFabricCompute",
    "IterativeRoutedFormulaFabricComputeInfo",
    "straight_through_route",
]
