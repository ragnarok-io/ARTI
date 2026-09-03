"""Final-loss search over ordinary Formula and self-network effect topology."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import re
from typing import ClassVar

import torch
from torch import Tensor, nn

from .formula_program_query import (
    FormulaProgramArena,
    FormulaProgramCandidate,
    FormulaProgramQueryResult,
    FormulaProgramTaskLoss,
    _ProgramOperandStore,
)
from .formula_v2 import (
    BankBinding,
    FormulaBankOperand,
    FormulaV2Error,
    InputBinding,
    _validate_tensor_against_type,
)
from .formula_v3 import (
    FormulaEffectProgramV2,
    FormulaFabricV4,
    apply_neural_plasticity_effect,
)


_NAME = re.compile(r"^[a-z][a-z0-9]*(?:[-_.][a-z0-9]+)*$")
_SUMMARY_WIDTH = 8


def _require_name(value: str, *, field: str) -> None:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise ValueError(f"{field} must be a canonical lowercase name")


@dataclass(frozen=True)
class FormulaProgramStateArena:
    """Immutable SSA values plus implicit execution-site network states.

    ProgramQuery observes only ``values``. States are available solely to an
    effect application or to an explicitly declared downstream Formula state
    binding, so a self-effect does not receive its own state as an operand.
    """

    values: FormulaProgramArena
    state_ids: tuple[str, ...]
    states: tuple[Tensor, ...]
    revisions: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "state_ids", tuple(self.state_ids))
        object.__setattr__(self, "states", tuple(self.states))
        object.__setattr__(self, "revisions", tuple(self.revisions))
        if not isinstance(self.values, FormulaProgramArena):
            raise TypeError("values must be FormulaProgramArena")
        if (
            not self.state_ids
            or len(set(self.state_ids)) != len(self.state_ids)
            or len(self.state_ids) != len(self.states)
            or len(self.state_ids) != len(self.revisions)
        ):
            raise ValueError("state ids, tensors, and revisions must be non-empty and aligned")
        for state_id in self.state_ids:
            _require_name(state_id, field="state_id")
        for state, revision in zip(self.states, self.revisions, strict=True):
            if not isinstance(state, Tensor) or not state.is_floating_point():
                raise TypeError("program network states must be floating Tensors")
            if state.device != self.values.device:
                raise ValueError("program values and network states must share device")
            if type(revision) is not int or revision < 0:
                raise ValueError("state revisions must be non-negative integers")

    @classmethod
    def from_mapping(
        cls,
        slot_ids: Sequence[str],
        values: Mapping[str, Tensor],
        state_ids: Sequence[str],
        states: Mapping[str, Tensor],
    ) -> FormulaProgramStateArena:
        normalized_state_ids = tuple(state_ids)
        unknown = set(states).difference(normalized_state_ids)
        missing = set(normalized_state_ids).difference(states)
        if unknown or missing:
            raise ValueError(
                f"network states do not match state_ids; missing={sorted(missing)}, "
                f"unknown={sorted(unknown)}"
            )
        return cls(
            FormulaProgramArena.from_mapping(slot_ids, values),
            normalized_state_ids,
            tuple(states[state_id] for state_id in normalized_state_ids),
            (0,) * len(normalized_state_ids),
        )

    @property
    def batch_size(self) -> int:
        return self.values.batch_size

    @property
    def device(self) -> torch.device:
        return self.values.device

    def state(self, state_id: str) -> Tensor:
        try:
            return self.states[self.state_ids.index(state_id)]
        except ValueError as exc:
            raise KeyError(state_id) from exc

    def revision(self, state_id: str) -> int:
        try:
            return self.revisions[self.state_ids.index(state_id)]
        except ValueError as exc:
            raise KeyError(state_id) from exc

    def with_values(self, values: FormulaProgramArena) -> FormulaProgramStateArena:
        return FormulaProgramStateArena(values, self.state_ids, self.states, self.revisions)

    def update_state(self, state_id: str, state: Tensor) -> FormulaProgramStateArena:
        try:
            index = self.state_ids.index(state_id)
        except ValueError as exc:
            raise KeyError(state_id) from exc
        states = list(self.states)
        revisions = list(self.revisions)
        states[index] = state
        revisions[index] += 1
        return FormulaProgramStateArena(
            self.values,
            self.state_ids,
            tuple(states),
            tuple(revisions),
        )


class FormulaProgramTensorCandidate(nn.Module):
    """Expose one ordinary Formula atom to state-aware topology search."""

    _component_reference: ClassVar[str] = "arti/formula-program-tensor-candidate@1"

    def __init__(
        self,
        candidate: FormulaProgramCandidate,
        *,
        state_operands: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(candidate, FormulaProgramCandidate):
            raise TypeError("candidate must be FormulaProgramCandidate")
        normalized = {} if state_operands is None else dict(state_operands)
        if not set(normalized).issubset(candidate._bank_bindings):
            raise ValueError("state_operands must name Formula Bank bindings")
        for state_id in normalized.values():
            _require_name(state_id, field="state operand")
        self.candidate = candidate
        self.state_operands = normalized

    @property
    def candidate_id(self) -> str:
        return self.candidate.candidate_id

    @property
    def atom_ref(self) -> str:
        return self.candidate.atom_ref

    @property
    def input_slots(self) -> Mapping[str, str]:
        return self.candidate.input_slots

    @property
    def output_slot(self) -> str:
        return self.candidate.output_slot

    @property
    def requires_empty_slots(self) -> tuple[str, ...]:
        return self.candidate.requires_empty_slots

    @property
    def effect_site_id(self) -> None:
        return None

    def contract_config(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.contract_config(),
            "state_operands": dict(sorted(self.state_operands.items())),
            "effect": False,
        }

    def _bindings(
        self,
        arena: FormulaProgramStateArena,
    ) -> tuple[dict[str, Tensor], dict[str, FormulaBankOperand]]:
        inputs, banks = self.candidate._bindings(arena.values)
        for name, state_id in self.state_operands.items():
            banks[name] = self.candidate._bank_bindings[name].bind(arena.state(state_id))
        return inputs, banks

    def accepts(self, arena: FormulaProgramStateArena) -> bool:
        if arena.values.get(self.output_slot) is not None:
            return False
        try:
            if any(arena.values.get(slot_id) is not None for slot_id in self.requires_empty_slots):
                return False
            inputs, banks = self._bindings(arena)
            self.candidate.fabric.bind_tensors(inputs=inputs, banks=banks)
        except (FormulaV2Error, KeyError, TypeError, ValueError):
            return False
        return True

    def forward(self, arena: FormulaProgramStateArena) -> FormulaProgramStateArena:
        inputs, banks = self._bindings(arena)
        value = self.candidate.fabric(inputs=inputs, banks=banks).values[0]
        return arena.with_values(arena.values.write(self.output_slot, value))


class FormulaProgramEffectCandidate(nn.Module):
    """One searchable NeuralPlasticity node with implicit site state."""

    _component_reference: ClassVar[str] = "arti/formula-program-effect-candidate@1"

    def __init__(
        self,
        candidate_id: str,
        effect_program: FormulaEffectProgramV2,
        *,
        input_slot: str,
        output_slot: str,
        state_id: str,
        requires_empty_slots: Sequence[str] = (),
        operands: Mapping[str, Tensor],
        trainable_operands: Sequence[str] = (),
        batch_broadcast_operands: Sequence[str] = (),
    ) -> None:
        super().__init__()
        for value, field in (
            (candidate_id, "candidate_id"),
            (input_slot, "input_slot"),
            (output_slot, "output_slot"),
            (state_id, "state_id"),
        ):
            _require_name(value, field=field)
        if not isinstance(effect_program, FormulaEffectProgramV2):
            raise TypeError("effect_program must be FormulaEffectProgramV2")
        input_bindings = tuple(
            binding
            for binding in effect_program.program.bindings
            if isinstance(binding, InputBinding)
        )
        if len(input_bindings) != 1 or input_bindings[0].name != effect_program.data_input_name:
            raise ValueError("effect search candidates require one current-data input")
        bank_bindings = {
            binding.name: binding
            for binding in effect_program.program.bindings
            if isinstance(binding, BankBinding)
        }
        normalized_empty = tuple(requires_empty_slots)
        if len(set(normalized_empty)) != len(normalized_empty):
            raise ValueError("requires_empty_slots must not contain duplicates")
        for slot_id in normalized_empty:
            _require_name(slot_id, field="required empty slot")
        broadcast = frozenset(batch_broadcast_operands)
        if not broadcast.issubset(bank_bindings):
            raise ValueError("batch_broadcast_operands must name Formula Bank bindings")

        self.candidate_id = candidate_id
        self.effect_program = effect_program
        self.input_slot = input_slot
        self.output_slot = output_slot
        self.state_id = state_id
        self.requires_empty_slots = normalized_empty
        self.fabric = FormulaFabricV4(effect_program)
        self._bank_bindings = bank_bindings
        self.operand_store = _ProgramOperandStore(
            dict(operands),
            trainable=tuple(trainable_operands),
        )
        if set(self.operand_store.names) != set(bank_bindings):
            raise ValueError("operands must bind every Formula BankBinding exactly once")
        self.batch_broadcast_operands = broadcast

    @property
    def atom_ref(self) -> str:
        return self.effect_program.effect_instruction.atom_ref

    @property
    def input_slots(self) -> Mapping[str, str]:
        return {self.effect_program.data_input_name: self.input_slot}

    @property
    def effect_site_id(self) -> str:
        return self.state_id

    def contract_config(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "effect_program": self.effect_program.to_dict(),
            "input_slot": self.input_slot,
            "output_slot": self.output_slot,
            "state_id": self.state_id,
            "requires_empty_slots": list(self.requires_empty_slots),
            "operands": self.operand_store.contract_config(),
            "batch_broadcast_operands": sorted(self.batch_broadcast_operands),
            "placement": "program-query-selected",
        }

    def _bindings(
        self,
        arena: FormulaProgramStateArena,
    ) -> tuple[dict[str, Tensor], dict[str, FormulaBankOperand]]:
        value = arena.values.get(self.input_slot)
        if value is None:
            raise ValueError(f"candidate input slot {self.input_slot!r} is empty")
        tensors = self.operand_store.tensors()
        for name in self.batch_broadcast_operands:
            operand = tensors[name]
            if operand.ndim < 1 or operand.shape[0] != 1:
                raise ValueError(
                    f"batch-broadcast operand {name!r} must have a leading singleton axis"
                )
            tensors[name] = operand.expand(arena.batch_size, *operand.shape[1:])
        banks = {name: binding.bind(tensors[name]) for name, binding in self._bank_bindings.items()}
        return {self.effect_program.data_input_name: value}, banks

    def accepts(self, arena: FormulaProgramStateArena) -> bool:
        if arena.values.get(self.output_slot) is not None:
            return False
        try:
            if any(arena.values.get(slot_id) is not None for slot_id in self.requires_empty_slots):
                return False
            inputs, banks = self._bindings(arena)
            self.fabric._executor.bind_tensors(inputs=inputs, banks=banks)
            state = arena.state(self.state_id)
            _validate_tensor_against_type(
                state,
                self.effect_program.state_type,
                name=f"{self.candidate_id}.state",
            )
            data_binding = next(
                binding
                for binding in self.effect_program.program.bindings
                if isinstance(binding, InputBinding)
            )
            data = inputs[self.effect_program.data_input_name]
            data_axes = {
                axis: data.shape[index]
                for index, axis in enumerate(data_binding.value_type.axis_names)
            }
            for index, axis in enumerate(self.effect_program.state_type.axis_names):
                if axis in data_axes and state.shape[index] != data_axes[axis]:
                    raise ValueError("effect state and current data axis extents disagree")
        except (FormulaV2Error, KeyError, TypeError, ValueError):
            return False
        return True

    def forward(self, arena: FormulaProgramStateArena) -> FormulaProgramStateArena:
        inputs, banks = self._bindings(arena)
        result = self.fabric._execute_owned(inputs=inputs, banks=banks)
        successor = apply_neural_plasticity_effect(
            result.effect,
            arena.state(self.state_id),
            state_type=self.effect_program.state_type,
        )
        values = arena.values.write(self.output_slot, result.value)
        return arena.with_values(values).update_state(self.state_id, successor)


FormulaProgramSearchCandidate = FormulaProgramTensorCandidate | FormulaProgramEffectCandidate


@dataclass(frozen=True)
class FormulaProgramQueryTraceStepV2:
    step: int
    candidate_id: str
    atom_ref: str | None
    input_slots: tuple[str, ...]
    output_slot: str | None
    effect_site_id: str | None
    state_revision: int | None


@dataclass(frozen=True)
class FormulaProgramQueryTraceV2:
    steps: tuple[FormulaProgramQueryTraceStepV2, ...]
    stopped: bool


@dataclass(frozen=True)
class FormulaProgramQueryExecutionV2:
    value: Tensor
    state_ids: tuple[str, ...]
    states: tuple[Tensor, ...]
    revisions: tuple[int, ...]
    trace: FormulaProgramQueryTraceV2

    def state(self, state_id: str) -> Tensor:
        try:
            return self.states[self.state_ids.index(state_id)]
        except ValueError as exc:
            raise KeyError(state_id) from exc


class FormulaProgramQueryV2(nn.Module):
    """Search ordinary and self-network nodes in one bounded SSA topology."""

    _component_reference: ClassVar[str] = "arti/formula-program-query@2"

    def __init__(
        self,
        *,
        slot_ids: Sequence[str],
        state_ids: Sequence[str],
        candidates: Sequence[FormulaProgramSearchCandidate],
        terminal_slot: str,
        min_steps: int = 1,
        max_steps: int = 8,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        normalized_slots = tuple(slot_ids)
        normalized_states = tuple(state_ids)
        if not normalized_slots or len(set(normalized_slots)) != len(normalized_slots):
            raise ValueError("slot_ids must be a non-empty unique sequence")
        if not normalized_states or len(set(normalized_states)) != len(normalized_states):
            raise ValueError("state_ids must be a non-empty unique sequence")
        for slot_id in normalized_slots:
            _require_name(slot_id, field="slot_id")
        for state_id in normalized_states:
            _require_name(state_id, field="state_id")
        _require_name(terminal_slot, field="terminal_slot")
        if terminal_slot not in normalized_slots:
            raise ValueError("terminal_slot must name one declared slot")
        normalized_candidates = tuple(candidates)
        if not normalized_candidates or any(
            not isinstance(
                candidate,
                (FormulaProgramTensorCandidate, FormulaProgramEffectCandidate),
            )
            for candidate in normalized_candidates
        ):
            raise TypeError("candidates must contain searchable Formula program candidates")
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
                isinstance(candidate, FormulaProgramEffectCandidate)
                and candidate.state_id not in normalized_states
            ):
                raise ValueError("effect candidates must target a declared state_id")
            if isinstance(candidate, FormulaProgramTensorCandidate) and not set(
                candidate.state_operands.values()
            ).issubset(normalized_states):
                raise ValueError("Formula state operands must target declared state_ids")
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
        self.state_ids = normalized_states
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
            "state_ids": list(self.state_ids),
            "candidate_ids": list(self.candidate_ids),
            "terminal_slot": self.terminal_slot,
            "min_steps": self.min_steps,
            "max_steps": self.max_steps,
            "hidden_dim": self.hidden_dim,
            "selection": "hard-one-shape-valid",
            "topology": "query-selected-ordinary-and-self-effect-nodes",
            "query_state_access": False,
            "effect_step_accounting": "program-step-not-refine",
        }

    def arena(
        self,
        values: Mapping[str, Tensor],
        *,
        states: Mapping[str, Tensor],
    ) -> FormulaProgramStateArena:
        return FormulaProgramStateArena.from_mapping(
            self.slot_ids,
            values,
            self.state_ids,
            states,
        )

    def eligible(self, arena: FormulaProgramStateArena, *, steps: int) -> Tensor:
        if arena.values.slot_ids != self.slot_ids or arena.state_ids != self.state_ids:
            raise ValueError("arena layout does not match ProgramQuery@2")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        candidate_eligible = [
            steps < self.max_steps and candidate.accepts(arena) for candidate in self.candidates
        ]
        stop_eligible = steps >= self.min_steps and arena.values.get(self.terminal_slot) is not None
        return torch.tensor(
            (*candidate_eligible, stop_eligible),
            dtype=torch.bool,
            device=arena.device,
        )

    def _summarize(self, arena: FormulaProgramStateArena) -> Tensor:
        parameter = next(self.network.parameters())
        if arena.device != parameter.device:
            raise ValueError("arena and FormulaProgramQuery must share device")
        rows: list[Tensor] = []
        for value in arena.values.values:
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
                        numeric.new_full(
                            (arena.batch_size, 1),
                            math.log1p(numeric.shape[-1]),
                        ),
                    ),
                    dim=-1,
                )
            )
        return torch.cat(rows, dim=-1)

    def query(
        self,
        arena: FormulaProgramStateArena,
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
        values: Mapping[str, Tensor] | FormulaProgramStateArena,
        *,
        states: Mapping[str, Tensor] | None = None,
    ) -> FormulaProgramQueryExecutionV2:
        if isinstance(values, FormulaProgramStateArena):
            if states is not None:
                raise ValueError("states must be omitted when values is already an arena")
            arena = values
        else:
            if states is None:
                raise ValueError("states are required for ProgramQuery@2")
            arena = self.arena(values, states=states)
        if arena.batch_size != 1:
            raise ValueError("hard ProgramQuery execution currently requires batch size one")
        trace: list[FormulaProgramQueryTraceStepV2] = []
        steps = 0
        while True:
            result = self.query(arena, steps=steps)
            selected = self._hard_index(result.masked_logits)
            if selected == len(self.candidates):
                value = arena.values.get(self.terminal_slot)
                assert value is not None
                trace.append(
                    FormulaProgramQueryTraceStepV2(steps, "stop", None, (), None, None, None)
                )
                return FormulaProgramQueryExecutionV2(
                    value,
                    arena.state_ids,
                    arena.states,
                    arena.revisions,
                    FormulaProgramQueryTraceV2(tuple(trace), True),
                )
            candidate = self.candidates[selected]
            arena = candidate(arena)
            revision = (
                None
                if candidate.effect_site_id is None
                else arena.revision(candidate.effect_site_id)
            )
            trace.append(
                FormulaProgramQueryTraceStepV2(
                    steps,
                    candidate.candidate_id,
                    candidate.atom_ref,
                    tuple(candidate.input_slots.values()),
                    candidate.output_slot,
                    candidate.effect_site_id,
                    revision,
                )
            )
            steps += 1


@dataclass(frozen=True)
class FormulaProgramQueryTrainingLossV2:
    total: Tensor
    task: Tensor
    invalid: Tensor
    success_probability: Tensor
    visited_states: int
    per_row_total: Tensor


class ExactFormulaProgramQueryTrainingV2:
    """Train effect topology from exact expected final task loss."""

    _component_reference: ClassVar[str] = "arti/exact-formula-program-query-training@2"

    def __init__(
        self,
        *,
        invalid_weight: float = 2.0,
        max_states: int = 512,
        exploration_probability: float = 0.0,
    ) -> None:
        if (
            isinstance(invalid_weight, bool)
            or not isinstance(invalid_weight, (int, float))
            or not math.isfinite(float(invalid_weight))
            or invalid_weight < 0
        ):
            raise ValueError("invalid_weight must be finite and non-negative")
        if isinstance(max_states, bool) or not isinstance(max_states, int) or max_states <= 0:
            raise ValueError("max_states must be a positive integer")
        if (
            isinstance(exploration_probability, bool)
            or not isinstance(exploration_probability, (int, float))
            or not math.isfinite(float(exploration_probability))
            or not 0.0 <= float(exploration_probability) < 1.0
        ):
            raise ValueError("exploration_probability must be finite in [0, 1)")
        self.invalid_weight = float(invalid_weight)
        self.max_states = int(max_states)
        self.exploration_probability = float(exploration_probability)

    def contract_config(self) -> dict[str, object]:
        return {
            "invalid_weight": self.invalid_weight,
            "max_states": self.max_states,
            "exploration_probability": self.exploration_probability,
            "supervision": "final-task-loss-only",
            "route_teacher": False,
            "effect_topology_teacher": False,
            "estimator": "exact-expected-stateful-program-policy",
            "state_merging": "none-path-state-is-order-sensitive",
        }

    def loss(
        self,
        program_query: FormulaProgramQueryV2,
        *,
        initial: Mapping[str, Tensor] | FormulaProgramStateArena,
        states: Mapping[str, Tensor] | None = None,
        target: object,
        task_loss: FormulaProgramTaskLoss,
    ) -> FormulaProgramQueryTrainingLossV2:
        if not isinstance(program_query, FormulaProgramQueryV2):
            raise TypeError("program_query must be FormulaProgramQueryV2")
        if isinstance(initial, FormulaProgramStateArena):
            if states is not None:
                raise ValueError("states must be omitted when initial is already an arena")
            arena = initial
        else:
            if states is None:
                raise ValueError("states are required for ProgramQuery@2 training")
            arena = program_query.arena(initial, states=states)
        batch = arena.batch_size
        reference = next(value for value in arena.values.values if value is not None)
        expected_task = reference.new_zeros((batch,), dtype=torch.float32)
        success = reference.new_zeros((batch,), dtype=torch.float32)
        visited_states = 0
        frontier: list[tuple[FormulaProgramStateArena, Tensor]] = [
            (arena, expected_task.new_ones((batch,)))
        ]
        steps = 0
        while frontier:
            next_frontier: list[tuple[FormulaProgramStateArena, Tensor]] = []
            for state, probability in frontier:
                visited_states += 1
                if visited_states > self.max_states:
                    raise RuntimeError("ProgramQuery path graph exceeded max_states")
                try:
                    result = program_query.query(state, steps=steps)
                except RuntimeError:
                    continue
                action_probability = result.masked_logits.softmax(dim=-1)
                if self.exploration_probability:
                    uniform = result.eligible.to(action_probability)
                    uniform = uniform / uniform.sum()
                    action_probability = (
                        1.0 - self.exploration_probability
                    ) * action_probability + self.exploration_probability * uniform.unsqueeze(0)
                stop_index = len(program_query.candidates)
                if bool(result.eligible[stop_index]):
                    output = state.values.get(program_query.terminal_slot)
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
                    next_frontier.append(
                        (
                            candidate(state),
                            probability * action_probability[:, candidate_index],
                        )
                    )
            frontier = next_frontier
            steps += 1
        invalid = (1.0 - success).clamp_min(0.0)
        conditional_task = expected_task / success.clamp_min(1e-8)
        task = conditional_task.mean()
        invalid_loss = invalid.mean()
        per_row = conditional_task + self.invalid_weight * invalid
        return FormulaProgramQueryTrainingLossV2(
            total=task + self.invalid_weight * invalid_loss,
            task=task,
            invalid=invalid_loss,
            success_probability=success.mean(),
            visited_states=visited_states,
            per_row_total=per_row,
        )


__all__ = [
    "ExactFormulaProgramQueryTrainingV2",
    "FormulaProgramEffectCandidate",
    "FormulaProgramQueryExecutionV2",
    "FormulaProgramQueryTraceStepV2",
    "FormulaProgramQueryTraceV2",
    "FormulaProgramQueryTrainingLossV2",
    "FormulaProgramQueryV2",
    "FormulaProgramSearchCandidate",
    "FormulaProgramStateArena",
    "FormulaProgramTensorCandidate",
]
