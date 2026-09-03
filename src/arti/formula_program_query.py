"""Bounded Formula atom selection over an immutable typed SSA arena."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import math
import re
from typing import ClassVar

import torch
from torch import Tensor, nn

from .formula_v2 import (
    BankBinding,
    FormulaBankOperand,
    FormulaFabricV2,
    FormulaProgram,
    FormulaV2Error,
    InputBinding,
)


_NAME = re.compile(r"^[a-z][a-z0-9]*(?:[-_.][a-z0-9]+)*$")
_SUMMARY_WIDTH = 8


def _require_name(value: str, *, field: str) -> None:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise ValueError(f"{field} must be a canonical lowercase name")


@dataclass(frozen=True)
class FormulaProgramArena:
    """One immutable runtime snapshot of fixed named SSA slots."""

    slot_ids: tuple[str, ...]
    values: tuple[Tensor | None, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "slot_ids", tuple(self.slot_ids))
        object.__setattr__(self, "values", tuple(self.values))
        if not self.slot_ids or len(self.slot_ids) != len(self.values):
            raise ValueError("slot_ids and values must be non-empty and equally sized")
        if len(set(self.slot_ids)) != len(self.slot_ids):
            raise ValueError("slot_ids must be unique")
        for slot_id in self.slot_ids:
            _require_name(slot_id, field="slot_id")
        occupied = [value for value in self.values if value is not None]
        if not occupied:
            raise ValueError("an arena must contain at least one occupied slot")
        if any(not isinstance(value, Tensor) or value.ndim < 1 for value in occupied):
            raise TypeError("occupied slots must be tensors with a batch dimension")
        batch = occupied[0].shape[0]
        device = occupied[0].device
        if batch <= 0:
            raise ValueError("arena batch size must be positive")
        if any(value.shape[0] != batch or value.device != device for value in occupied):
            raise ValueError("all occupied slots must share batch size and device")

    @classmethod
    def from_mapping(
        cls,
        slot_ids: Sequence[str],
        values: Mapping[str, Tensor],
    ) -> FormulaProgramArena:
        normalized_slots = tuple(slot_ids)
        unknown = set(values).difference(normalized_slots)
        if unknown:
            raise ValueError(f"initial values contain unknown slots: {sorted(unknown)}")
        return cls(normalized_slots, tuple(values.get(slot) for slot in normalized_slots))

    @property
    def batch_size(self) -> int:
        return next(value for value in self.values if value is not None).shape[0]

    @property
    def device(self) -> torch.device:
        return next(value for value in self.values if value is not None).device

    def get(self, slot_id: str) -> Tensor | None:
        try:
            index = self.slot_ids.index(slot_id)
        except ValueError as exc:
            raise KeyError(slot_id) from exc
        return self.values[index]

    def write(self, slot_id: str, value: Tensor) -> FormulaProgramArena:
        if not isinstance(value, Tensor):
            raise TypeError("SSA slot values must be tensors")
        try:
            index = self.slot_ids.index(slot_id)
        except ValueError as exc:
            raise KeyError(slot_id) from exc
        if self.values[index] is not None:
            raise ValueError(f"SSA output slot {slot_id!r} is already occupied")
        values = list(self.values)
        values[index] = value
        return FormulaProgramArena(self.slot_ids, tuple(values))

    def occupancy(self) -> tuple[bool, ...]:
        return tuple(value is not None for value in self.values)


class _ProgramOperandStore(nn.Module):
    def __init__(
        self,
        values: Mapping[str, Tensor],
        *,
        trainable: Sequence[str],
    ) -> None:
        super().__init__()
        self.names = tuple(sorted(values))
        trainable_names = frozenset(trainable)
        if not trainable_names.issubset(self.names):
            raise ValueError("trainable_operands must name declared Bank operands")
        self.trainable_names = trainable_names
        self._attributes: dict[str, str] = {}
        self._external_buffers: dict[str, tuple[nn.Module, str]] = {}
        for index, name in enumerate(self.names):
            value = values[name]
            if not isinstance(value, Tensor):
                raise TypeError("Formula candidate operands must be tensors")
            attribute = f"operand_{index:04d}"
            self._attributes[name] = attribute
            cloned = value.detach().clone()
            if name in trainable_names:
                if not (cloned.is_floating_point() or cloned.is_complex()):
                    raise TypeError("trainable Formula operands must be floating or complex")
                self.register_parameter(attribute, nn.Parameter(cloned))
            else:
                self.register_buffer(attribute, cloned, persistent=True)

    def tensors(self) -> dict[str, Tensor]:
        return {name: self.tensor(name) for name in self.names}

    def tensor(self, name: str) -> Tensor:
        if name in self._external_buffers:
            owner, attribute = self._external_buffers[name]
            return getattr(owner, attribute)
        try:
            attribute = self._attributes[name]
        except KeyError as exc:
            raise KeyError(name) from exc
        return getattr(self, attribute)

    def _bind_external_buffer(self, name: str, owner: nn.Module, attribute: str) -> None:
        """Follow a transferred buffer without registering its owner a second time."""
        if name in self.trainable_names:
            raise ValueError("trainable operands cannot be transferred to a Bank owner")
        local_attribute = self._attributes[name]
        self._buffers.pop(local_attribute, None)
        object.__setattr__(self, local_attribute, None)
        # Resolve the owner's current buffer after .to() replaces its Tensor.
        self._external_buffers[name] = (owner, attribute)

    def install_(self, name: str, value: Tensor) -> None:
        current = self.tensor(name)
        if (
            not isinstance(value, Tensor)
            or value.shape != current.shape
            or value.dtype != current.dtype
            or value.device != current.device
        ):
            raise ValueError("installed Formula operand must exactly match its Bank slot")
        with torch.no_grad():
            current.copy_(value.detach())

    def contract_config(self) -> dict[str, object]:
        return {
            name: {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "trainable": name in self.trainable_names,
            }
            for name, value in self.tensors().items()
        }


class FormulaProgramCandidate(nn.Module):
    """One pre-admitted atom application with explicit SSA input and output wiring."""

    _component_reference: ClassVar[str] = "arti/formula-program-candidate@1"

    def __init__(
        self,
        candidate_id: str,
        program: FormulaProgram,
        *,
        input_slots: Mapping[str, str],
        output_slot: str,
        requires_empty_slots: Sequence[str] = (),
        operands: Mapping[str, Tensor] | None = None,
        trainable_operands: Sequence[str] = (),
        batch_broadcast_operands: Sequence[str] = (),
    ) -> None:
        super().__init__()
        _require_name(candidate_id, field="candidate_id")
        _require_name(output_slot, field="output_slot")
        if not isinstance(program, FormulaProgram):
            raise TypeError("program must be FormulaProgram")
        if len(program.instructions) != 1 or len(program.outputs) != 1:
            raise ValueError("FormulaProgramCandidate requires exactly one atom instruction")
        input_bindings = tuple(
            binding for binding in program.bindings if isinstance(binding, InputBinding)
        )
        normalized_inputs = dict(input_slots)
        if set(normalized_inputs) != {binding.name for binding in input_bindings}:
            raise ValueError("input_slots must bind every Formula InputBinding exactly once")
        for slot_id in normalized_inputs.values():
            _require_name(slot_id, field="input slot")
        normalized_empty_slots = tuple(requires_empty_slots)
        if len(set(normalized_empty_slots)) != len(normalized_empty_slots):
            raise ValueError("requires_empty_slots must not contain duplicates")
        for slot_id in normalized_empty_slots:
            _require_name(slot_id, field="required empty slot")
        bank_bindings = {
            binding.name: binding
            for binding in program.bindings
            if isinstance(binding, BankBinding)
        }
        normalized_operands = {} if operands is None else dict(operands)
        if set(normalized_operands) != set(bank_bindings):
            raise ValueError("operands must bind every Formula BankBinding exactly once")
        batch_broadcast = frozenset(batch_broadcast_operands)
        if not batch_broadcast.issubset(bank_bindings):
            raise ValueError("batch_broadcast_operands must name Formula BankBindings")

        self.candidate_id = candidate_id
        self.input_slots = normalized_inputs
        self.output_slot = output_slot
        self.requires_empty_slots = normalized_empty_slots
        self.fabric = FormulaFabricV2(program)
        self._bank_bindings = bank_bindings
        self.operand_store = _ProgramOperandStore(
            normalized_operands,
            trainable=tuple(trainable_operands),
        )
        self.batch_broadcast_operands = batch_broadcast

    @property
    def program(self) -> FormulaProgram:
        return self.fabric.program

    @property
    def atom_ref(self) -> str:
        return self.program.instructions[0].atom_ref

    def contract_config(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "program": self.program.to_dict(),
            "program_fingerprint": self.program.fingerprint,
            "atom_ref": self.atom_ref,
            "input_slots": dict(sorted(self.input_slots.items())),
            "output_slot": self.output_slot,
            "requires_empty_slots": list(self.requires_empty_slots),
            "operands": self.operand_store.contract_config(),
            "batch_broadcast_operands": sorted(self.batch_broadcast_operands),
        }

    def _bindings(
        self,
        arena: FormulaProgramArena,
    ) -> tuple[dict[str, Tensor], dict[str, FormulaBankOperand]]:
        inputs: dict[str, Tensor] = {}
        for input_name, slot_id in self.input_slots.items():
            value = arena.get(slot_id)
            if value is None:
                raise ValueError(f"candidate input slot {slot_id!r} is empty")
            inputs[input_name] = value
        tensors = self.operand_store.tensors()
        for name in self.batch_broadcast_operands:
            value = tensors[name]
            if value.ndim < 1 or value.shape[0] != 1:
                raise ValueError(
                    f"batch-broadcast operand {name!r} must have a leading singleton axis"
                )
            tensors[name] = value.expand(arena.batch_size, *value.shape[1:])
        banks = {
            name: binding.bind(tensors[name])
            for name, binding in self._bank_bindings.items()
        }
        return inputs, banks

    def accepts(self, arena: FormulaProgramArena) -> bool:
        if arena.get(self.output_slot) is not None:
            return False
        try:
            if any(arena.get(slot_id) is not None for slot_id in self.requires_empty_slots):
                return False
        except KeyError:
            return False
        try:
            inputs, banks = self._bindings(arena)
            self.fabric.bind_tensors(inputs=inputs, banks=banks)
        except (FormulaV2Error, KeyError, TypeError, ValueError):
            return False
        return True

    def forward(self, arena: FormulaProgramArena) -> FormulaProgramArena:
        if arena.get(self.output_slot) is not None:
            raise ValueError(f"SSA output slot {self.output_slot!r} is already occupied")
        inputs, banks = self._bindings(arena)
        value = self.fabric(inputs=inputs, banks=banks).values[0]
        return arena.write(self.output_slot, value)


@dataclass(frozen=True)
class FormulaProgramQueryResult:
    logits: Tensor
    masked_logits: Tensor
    eligible: Tensor


@dataclass(frozen=True)
class FormulaProgramQueryTraceStep:
    step: int
    candidate_id: str
    atom_ref: str | None
    input_slots: tuple[str, ...]
    output_slot: str | None


@dataclass(frozen=True)
class FormulaProgramQueryTrace:
    steps: tuple[FormulaProgramQueryTraceStep, ...]
    stopped: bool


class FormulaProgramQuery(nn.Module):
    """Learn a hard, bounded sequence of shape-valid atom applications."""

    _component_reference: ClassVar[str] = "arti/formula-program-query@1"

    def __init__(
        self,
        *,
        slot_ids: Sequence[str],
        candidates: Sequence[FormulaProgramCandidate],
        terminal_slot: str,
        min_steps: int = 1,
        max_steps: int = 8,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        normalized_slots = tuple(slot_ids)
        if not normalized_slots or len(set(normalized_slots)) != len(normalized_slots):
            raise ValueError("slot_ids must be a non-empty unique sequence")
        for slot_id in normalized_slots:
            _require_name(slot_id, field="slot_id")
        _require_name(terminal_slot, field="terminal_slot")
        if terminal_slot not in normalized_slots:
            raise ValueError("terminal_slot must name one declared slot")
        normalized_candidates = tuple(candidates)
        if not normalized_candidates or any(
            not isinstance(candidate, FormulaProgramCandidate)
            for candidate in normalized_candidates
        ):
            raise TypeError("candidates must contain FormulaProgramCandidate values")
        candidate_ids = tuple(candidate.candidate_id for candidate in normalized_candidates)
        if len(set(candidate_ids)) != len(candidate_ids) or "stop" in candidate_ids:
            raise ValueError("candidate ids must be unique and must not use 'stop'")
        for candidate in normalized_candidates:
            referenced = {
                *candidate.input_slots.values(),
                candidate.output_slot,
                *candidate.requires_empty_slots,
            }
            if not referenced.issubset(normalized_slots):
                raise ValueError("candidate wiring must reference declared SSA slots")
        if (
            isinstance(min_steps, bool)
            or not isinstance(min_steps, int)
            or min_steps < 0
            or isinstance(max_steps, bool)
            or not isinstance(max_steps, int)
            or max_steps < max(1, min_steps)
        ):
            raise ValueError("ProgramQuery step bounds are invalid")
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")

        self.slot_ids = normalized_slots
        self.candidates = nn.ModuleList(normalized_candidates)
        self.terminal_slot = terminal_slot
        self.min_steps = int(min_steps)
        self.max_steps = int(max_steps)
        self.hidden_dim = int(hidden_dim)
        self.network = nn.Sequential(
            nn.Linear(len(normalized_slots) * _SUMMARY_WIDTH, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, len(normalized_candidates) + 1),
        )
        action_ids = candidate_ids + ("stop",)
        lexical_rank = {item: rank for rank, item in enumerate(sorted(action_ids))}
        self.register_buffer(
            "_action_priority",
            torch.tensor([lexical_rank[item] for item in action_ids], dtype=torch.int64),
            persistent=False,
        )

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(candidate.candidate_id for candidate in self.candidates)

    @property
    def action_ids(self) -> tuple[str, ...]:
        return self.candidate_ids + ("stop",)

    def contract_config(self) -> dict[str, object]:
        return {
            "slot_ids": list(self.slot_ids),
            "candidate_ids": list(self.candidate_ids),
            "terminal_slot": self.terminal_slot,
            "min_steps": self.min_steps,
            "max_steps": self.max_steps,
            "hidden_dim": self.hidden_dim,
            "selection": "hard-one-shape-valid",
            "state": "immutable-ssa-arena",
        }

    def arena(self, values: Mapping[str, Tensor]) -> FormulaProgramArena:
        return FormulaProgramArena.from_mapping(self.slot_ids, values)

    def eligible(self, arena: FormulaProgramArena, *, steps: int) -> Tensor:
        if arena.slot_ids != self.slot_ids:
            raise ValueError("arena slot layout does not match ProgramQuery")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        candidate_eligible = [
            steps < self.max_steps and candidate.accepts(arena)
            for candidate in self.candidates
        ]
        stop_eligible = steps >= self.min_steps and arena.get(self.terminal_slot) is not None
        return torch.tensor(
            (*candidate_eligible, stop_eligible),
            dtype=torch.bool,
            device=arena.device,
        )

    def _summarize(self, arena: FormulaProgramArena) -> Tensor:
        parameter = next(self.network.parameters())
        if arena.device != parameter.device:
            raise ValueError("arena and FormulaProgramQuery must share device")
        rows: list[Tensor] = []
        for value in arena.values:
            if value is None:
                rows.append(
                    torch.zeros(
                        arena.batch_size,
                        _SUMMARY_WIDTH,
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                )
                continue
            numeric = value.to(dtype=parameter.dtype).reshape(arena.batch_size, -1)
            if not bool(torch.isfinite(numeric).all()):
                raise ValueError("ProgramQuery arena values must be finite")
            element_count = numeric.shape[-1]
            shape_feature = numeric.new_full(
                (arena.batch_size, 1), math.log1p(element_count)
            )
            rows.append(
                torch.cat(
                    (
                        numeric.new_ones((arena.batch_size, 1)),
                        numeric.mean(dim=-1, keepdim=True),
                        numeric.std(dim=-1, unbiased=False, keepdim=True),
                        numeric.abs().mean(dim=-1, keepdim=True),
                        numeric.amax(dim=-1, keepdim=True),
                        numeric.amin(dim=-1, keepdim=True),
                        numeric.square().mean(dim=-1, keepdim=True).sqrt(),
                        shape_feature,
                    ),
                    dim=-1,
                )
            )
        return torch.cat(rows, dim=-1)

    def query(
        self,
        arena: FormulaProgramArena,
        *,
        steps: int,
    ) -> FormulaProgramQueryResult:
        eligible = self.eligible(arena, steps=steps)
        if not bool(eligible.any()):
            raise RuntimeError("ProgramQuery has no shape-valid candidate or valid stop")
        logits = self.network(self._summarize(arena))
        masked = logits.masked_fill(~eligible.unsqueeze(0), -torch.inf)
        return FormulaProgramQueryResult(logits, masked, eligible)

    def _hard_index(self, masked_logits: Tensor) -> int:
        if masked_logits.shape[0] != 1:
            raise ValueError("hard ProgramQuery execution currently requires batch size one")
        maxima = masked_logits[0] == masked_logits[0].max()
        sentinel = torch.full_like(self._action_priority, len(self.action_ids))
        priority = torch.where(maxima, self._action_priority, sentinel)
        return int(priority.argmin().item())

    def forward(
        self,
        values: Mapping[str, Tensor] | FormulaProgramArena,
        *,
        return_trace: bool = False,
    ) -> Tensor | tuple[Tensor, FormulaProgramQueryTrace]:
        arena = values if isinstance(values, FormulaProgramArena) else self.arena(values)
        if arena.batch_size != 1:
            raise ValueError("hard ProgramQuery execution currently requires batch size one")
        trace: list[FormulaProgramQueryTraceStep] = []
        steps = 0
        while True:
            result = self.query(arena, steps=steps)
            selected = self._hard_index(result.masked_logits)
            if selected == len(self.candidates):
                value = arena.get(self.terminal_slot)
                assert value is not None
                trace.append(
                    FormulaProgramQueryTraceStep(steps, "stop", None, (), None)
                )
                record = FormulaProgramQueryTrace(tuple(trace), True)
                return (value, record) if return_trace else value
            candidate = self.candidates[selected]
            arena = candidate(arena)
            trace.append(
                FormulaProgramQueryTraceStep(
                    steps,
                    candidate.candidate_id,
                    candidate.atom_ref,
                    tuple(candidate.input_slots.values()),
                    candidate.output_slot,
                )
            )
            steps += 1


FormulaProgramTaskLoss = Callable[[Tensor, object], Tensor]


@dataclass(frozen=True)
class FormulaProgramQueryTrainingLoss:
    total: Tensor
    task: Tensor
    invalid: Tensor
    success_probability: Tensor
    visited_states: int
    per_row_total: Tensor


class ExactFormulaProgramQueryTraining:
    """Train ProgramQuery from exact expected final task loss over a bounded graph."""

    _component_reference: ClassVar[str] = "arti/exact-formula-program-query-training@1"

    def __init__(self, *, invalid_weight: float = 2.0, max_states: int = 512) -> None:
        if (
            isinstance(invalid_weight, bool)
            or not isinstance(invalid_weight, (int, float))
            or not math.isfinite(float(invalid_weight))
            or invalid_weight < 0
        ):
            raise ValueError("invalid_weight must be finite and non-negative")
        if isinstance(max_states, bool) or not isinstance(max_states, int) or max_states <= 0:
            raise ValueError("max_states must be a positive integer")
        self.invalid_weight = float(invalid_weight)
        self.max_states = int(max_states)

    def contract_config(self) -> dict[str, object]:
        return {
            "invalid_weight": self.invalid_weight,
            "max_states": self.max_states,
            "supervision": "final-task-loss-only",
            "route_teacher": False,
            "transition_teacher": False,
            "estimator": "exact-expected-program-policy",
            "state_merging": "ssa-producer-equivalence",
        }

    def loss(
        self,
        program_query: FormulaProgramQuery,
        *,
        initial: Mapping[str, Tensor] | FormulaProgramArena,
        target: object,
        task_loss: FormulaProgramTaskLoss,
    ) -> FormulaProgramQueryTrainingLoss:
        if not isinstance(program_query, FormulaProgramQuery):
            raise TypeError("program_query must be FormulaProgramQuery")
        arena = (
            initial
            if isinstance(initial, FormulaProgramArena)
            else program_query.arena(initial)
        )
        batch = arena.batch_size
        reference = next(value for value in arena.values if value is not None)
        expected_task = reference.new_zeros((batch,), dtype=torch.float32)
        success = reference.new_zeros((batch,), dtype=torch.float32)
        visited_states = 0
        slot_index = {slot_id: index for index, slot_id in enumerate(arena.slot_ids)}
        initial_key = tuple(-2 if value is not None else -1 for value in arena.values)
        frontier: dict[
            tuple[int, ...], tuple[FormulaProgramArena, Tensor]
        ] = {initial_key: (arena, expected_task.new_ones((batch,)))}
        steps = 0
        while frontier:
            next_frontier: dict[
                tuple[int, ...], tuple[FormulaProgramArena, Tensor]
            ] = {}
            for state_key, (state, probability) in frontier.items():
                visited_states += 1
                if visited_states > self.max_states:
                    raise RuntimeError("ProgramQuery path graph exceeded max_states")
                try:
                    result = program_query.query(state, steps=steps)
                except RuntimeError:
                    continue
                action_probability = result.masked_logits.softmax(dim=-1)
                stop_index = len(program_query.candidates)
                if bool(result.eligible[stop_index]):
                    output = state.get(program_query.terminal_slot)
                    assert output is not None
                    row_loss = task_loss(output, target)
                    if (
                        not isinstance(row_loss, Tensor)
                        or not row_loss.is_floating_point()
                        or row_loss.shape != (batch,)
                        or row_loss.device != arena.device
                        or not bool(torch.isfinite(row_loss).all())
                    ):
                        raise TypeError("task_loss must return finite floating [B] values")
                    stop_probability = probability * action_probability[:, stop_index]
                    expected_task = expected_task + stop_probability * row_loss.float()
                    success = success + stop_probability
                for candidate_index, candidate in enumerate(program_query.candidates):
                    if not bool(result.eligible[candidate_index]):
                        continue
                    next_probability = probability * action_probability[:, candidate_index]
                    output_index = slot_index[candidate.output_slot]
                    next_key_values = list(state_key)
                    next_key_values[output_index] = candidate_index
                    next_key = tuple(next_key_values)
                    previous = next_frontier.get(next_key)
                    if previous is None:
                        next_frontier[next_key] = (
                            candidate(state),
                            next_probability,
                        )
                    else:
                        next_frontier[next_key] = (
                            previous[0],
                            previous[1] + next_probability,
                        )
            frontier = next_frontier
            steps += 1
        invalid = (1.0 - success).clamp_min(0.0)
        task = expected_task.mean()
        invalid_loss = invalid.mean()
        per_row = expected_task + self.invalid_weight * invalid
        return FormulaProgramQueryTrainingLoss(
            total=task + self.invalid_weight * invalid_loss,
            task=task,
            invalid=invalid_loss,
            success_probability=success.mean(),
            visited_states=visited_states,
            per_row_total=per_row,
        )


__all__ = [
    "ExactFormulaProgramQueryTraining",
    "FormulaProgramArena",
    "FormulaProgramCandidate",
    "FormulaProgramQuery",
    "FormulaProgramQueryResult",
    "FormulaProgramQueryTrace",
    "FormulaProgramQueryTraceStep",
    "FormulaProgramQueryTrainingLoss",
    "FormulaProgramTaskLoss",
]
