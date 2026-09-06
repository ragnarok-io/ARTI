"""Named Formula outputs with per-port lineage and all-heads Query termination."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import ClassVar

from torch import Tensor, nn

from .formula_program_call import FormulaProgramCallCandidateV1
from .formula_program_query import FormulaProgramArena, _require_name
from .formula_program_query_v3 import (
    FormulaProgramBankState,
    FormulaProgramCandidateV2,
    _program_output_dependencies,
)
from .formula_program_query_v4 import (
    FormulaProgramEffectCandidateV3,
    FormulaProgramQueryTensorEncoderV1,
    FormulaProgramQueryV4,
    FormulaProgramSearchCandidateV4,
    FormulaProgramTensorCandidateV3,
    _BankSlotEffectProposalV2,
    _FormulaProducerLineageV2,
    _FormulaProgramExecutionArenaV4,
)
from .formula_v2 import FormulaProgram, FormulaV2Error


def _named_slots(values: Mapping[str, str], *, field: str) -> dict[str, str]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError(f"{field} must be a non-empty named slot mapping")
    result = dict(values)
    for slot in result.values():
        _require_name(slot, field=f"{field} slot")
    return result


def _read_outputs(
    arena: _FormulaProgramExecutionArenaV4, slots: Mapping[str, str],
) -> dict[str, Tensor]:
    outputs = {}
    for name, slot in slots.items():
        value = arena.values.get(slot)
        assert value is not None
        outputs[name] = value
    return outputs


class FormulaProgramCandidateV3(FormulaProgramCandidateV2):
    """A Formula subprogram that publishes every named output without merging."""

    _component_reference: ClassVar[str] = "arti/formula-program-candidate@3"

    def __init__(
        self, candidate_id: str, program: FormulaProgram, *,
        input_slots: Mapping[str, str], output_slots: Mapping[str, str],
        requires_empty_slots: Sequence[str] = (), operands: Mapping[str, Tensor] | None = None,
        trainable_operands: Sequence[str] = (), batch_broadcast_operands: Sequence[str] = (),
    ) -> None:
        outputs = _named_slots(output_slots, field="output_slots")
        if not isinstance(program, FormulaProgram):
            raise TypeError("program must be FormulaProgram")
        if set(outputs) != set(program.outputs):
            raise ValueError("output_slots must bind every Formula output exactly once")
        if len(set(outputs.values())) != len(outputs):
            raise ValueError("Formula outputs must use distinct SSA slots")
        super().__init__(
            candidate_id, program, input_slots=input_slots, output_slot=outputs[program.outputs[0]],
            requires_empty_slots=requires_empty_slots, operands=operands,
            trainable_operands=trainable_operands, batch_broadcast_operands=batch_broadcast_operands,
        )
        self.output_slots = {name: outputs[name] for name in program.outputs}

    def _validate_program_outputs(self, program: FormulaProgram) -> None:
        if not program.outputs:
            raise ValueError("Formula program requires at least one public output")

    def contract_config(self) -> dict[str, object]:
        config = super().contract_config()
        config.pop("output_slot")
        config["output_slots"] = dict(self.output_slots)
        return config

    def accepts(self, arena: FormulaProgramArena) -> bool:
        if any(arena.get(slot) is not None for slot in self.output_slots.values()):
            return False
        return super().accepts(arena)

    def forward(self, arena: FormulaProgramArena) -> FormulaProgramArena:
        if any(arena.get(slot) is not None for slot in self.output_slots.values()):
            raise ValueError("SSA candidate outputs must be empty")
        inputs, banks = self._bindings(arena)
        result = self.fabric(inputs=inputs, banks=banks)
        return arena.write_many(dict(zip(self.output_slots.values(), result.values, strict=True)))


class FormulaProgramTensorCandidateV4(FormulaProgramTensorCandidateV3):
    """Multi-output producer with Bank lineage only on dependent output ports."""

    _component_reference: ClassVar[str] = "arti/formula-program-tensor-candidate@4"
    _allows_multiple_outputs: ClassVar[bool] = True

    def __init__(
        self, candidate: FormulaProgramCandidateV3, *,
        plastic_bank_slot: str | None = None, bank_owner_id: str | None = None,
    ) -> None:
        if not isinstance(candidate, FormulaProgramCandidateV3):
            raise TypeError("named-output producer requires FormulaProgramCandidate@3")
        super().__init__(candidate, plastic_bank_slot=plastic_bank_slot, bank_owner_id=bank_owner_id)
        self._head_dependencies = {
            name: _program_output_dependencies(candidate.program, name)
            for name in candidate.program.outputs
        }

    @property
    def output_slot_ids(self) -> tuple[str, ...]:
        return tuple(self.candidate.output_slots.values())

    def accepts(self, arena: _FormulaProgramExecutionArenaV4) -> bool:
        try:
            if any(arena.values.get(slot) is not None for slot in (
                *self.output_slot_ids, *self.requires_empty_slots,
            )):
                return False
            inputs, banks = self._bindings(arena)
            self.candidate.fabric.bind_tensors(inputs=inputs, banks=banks)
        except (FormulaV2Error, KeyError, TypeError, ValueError):
            return False
        return True

    def forward(self, arena: _FormulaProgramExecutionArenaV4, *,
                _input_values: Mapping[str, Tensor] | None = None) -> _FormulaProgramExecutionArenaV4:
        if any(arena.values.get(slot) is not None for slot in self.output_slot_ids):
            raise ValueError("SSA candidate outputs must be empty")
        inputs, banks = self._bindings(arena) if _input_values is None else self._bindings(arena, _input_values)
        result = self.candidate.fabric(inputs=inputs, banks=banks)
        return self._finish_outputs(arena, dict(zip(self.candidate.program.outputs, result.values, strict=True)))

    def _finish_outputs(
        self, arena: _FormulaProgramExecutionArenaV4, numeric: Mapping[str, Tensor],
    ) -> _FormulaProgramExecutionArenaV4:
        slot_ref = self.bank_slot_ref
        current, revision = (None, None) if slot_ref is None else arena.effect_state(slot_ref)
        producers = {}
        outputs = {}
        for name, slot in self.candidate.output_slots.items():
            owns_value = slot_ref is not None and self.plastic_bank_slot in self._head_dependencies[name]
            producers[slot] = _FormulaProducerLineageV2(
                arena.execution_id(self.candidate_id), self.bank_owner_id, slot,
                slot_ref if owns_value else None,
                revision if owns_value else None,
                current if owns_value else None,
            )
            outputs[slot] = numeric[name]
        return arena.write_many(outputs, producers=producers)

    def contract_config(self) -> dict[str, object]:
        config = super().contract_config()
        config["output_lineage"] = "per-port-declared-bank-dependency"
        return config

    def with_bindings(
        self, candidate_id: str, *, input_slots: Mapping[str, str],
        output_slots: Mapping[str, str], requires_empty_slots: Sequence[str] = (),
    ) -> FormulaProgramTensorCandidateV4:
        """Create a new SSA occurrence of the SAME Fabric and Bank parameters.

        Use different occurrences to offer finite input-source alternatives to
        ProgramQuery. Rebinding never initializes another trainable operand or
        changes the identity of a plastic Bank owner.
        """
        _require_name(candidate_id, field="candidate_id")
        inputs = _named_slots(input_slots, field="input_slots")
        outputs = _named_slots(output_slots, field="output_slots")
        if set(inputs) != set(self.input_slots) or set(outputs) != set(self.candidate.output_slots):
            raise ValueError("bindings must preserve the Formula's named input and output ports")
        if len(set(outputs.values())) != len(outputs):
            raise ValueError("Formula outputs must use distinct SSA slots")
        empty = tuple(requires_empty_slots)
        for slot in empty:
            _require_name(slot, field="requires_empty_slots")
        result = copy.copy(self)
        result._modules = dict(self._modules)
        occurrence = copy.copy(self.candidate)
        occurrence._modules = dict(self.candidate._modules)
        occurrence.candidate_id = candidate_id
        occurrence.input_slots = inputs
        occurrence.output_slots = {name: outputs[name] for name in occurrence.program.outputs}
        occurrence.output_slot = outputs[occurrence.program.outputs[0]]
        occurrence.requires_empty_slots = empty
        result.candidate = occurrence
        return result


@dataclass(frozen=True)
class FormulaProgramQueryTraceStepV5:
    step: int
    candidate_id: str
    atom_ref: str | None
    input_slots: tuple[str, ...]
    output_slots: tuple[str, ...]
    bank_owner_ids: tuple[str | None, ...]
    child_trace: FormulaProgramQueryTraceV5 | None = None
    input_bindings: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class FormulaProgramQueryTraceV5:
    steps: tuple[FormulaProgramQueryTraceStepV5, ...]
    stopped: bool
    invocation_path: tuple[str, ...] = ()

    @property
    def total_dispatches(self) -> int:
        """Include call dispatches and all nested work, but not STOP decisions."""
        return sum(
            int(step.candidate_id != "stop")
            + (0 if step.child_trace is None else step.child_trace.total_dispatches)
            for step in self.steps
        )


@dataclass(frozen=True)
class FormulaProgramQueryExecutionV5:
    outputs: Mapping[str, Tensor]
    output_producers: Mapping[str, _FormulaProducerLineageV2 | None]
    bank_state: FormulaProgramBankState
    proposals: tuple[_BankSlotEffectProposalV2, ...]
    trace: FormulaProgramQueryTraceV5
    _owner_token: object

    def __post_init__(self) -> None:
        object.__setattr__(self, "outputs", MappingProxyType(dict(self.outputs)))
        object.__setattr__(self, "output_producers", MappingProxyType(dict(self.output_producers)))


class FormulaProgramQueryV5(FormulaProgramQueryV4):
    """Query-selected SSA graph with named outputs; STOP waits for every head.

    Uses the same Fabric, immutable arena and predecessor Bank overlay as V4.
    Child calls run their own Query loop and return functional state and heads.
    """

    _component_reference: ClassVar[str] = "arti/formula-program-query@5"
    _allows_multiple_outputs: ClassVar[bool] = True
    _candidate_types: ClassVar[tuple[type[nn.Module], ...]] = (
        *FormulaProgramQueryV4._candidate_types, FormulaProgramCallCandidateV1,
    )

    def __init__(
        self, *, slot_ids: Sequence[str],
        candidates: Sequence[FormulaProgramSearchCandidateV4 | FormulaProgramCallCandidateV1],
        terminal_slots: Mapping[str, str], min_steps: int = 1, max_steps: int = 8,
        min_tensor_steps: int = 0, max_tensor_steps: int | None = None,
        max_effect_steps: int | None = None, hidden_dim: int = 64,
        tensor_encoder: FormulaProgramQueryTensorEncoderV1 | None = None,
    ) -> None:
        outputs = _named_slots(terminal_slots, field="terminal_slots")
        for name in outputs:
            _require_name(name, field="terminal output name")
        super().__init__(
            slot_ids=slot_ids, candidates=candidates, terminal_slot=next(iter(outputs.values())),
            min_steps=min_steps, max_steps=max_steps, min_tensor_steps=min_tensor_steps,
            max_tensor_steps=max_tensor_steps, max_effect_steps=max_effect_steps,
            hidden_dim=hidden_dim, tensor_encoder=tensor_encoder,
        )
        if not set(outputs.values()).issubset(self.slot_ids):
            raise ValueError("terminal_slots must reference declared SSA slots")
        self.terminal_slots = outputs

    @property
    def plastic_candidates(self) -> tuple[FormulaProgramTensorCandidateV3, ...]:
        producers = []
        for candidate in self.candidates:
            if isinstance(candidate, FormulaProgramCallCandidateV1):
                producers.extend(candidate.child.plastic_candidates)
            elif isinstance(candidate, FormulaProgramTensorCandidateV3) and candidate.bank_slot_ref is not None:
                producers.append(candidate)
        return tuple(producers)

    def _bind_shared_bank_owners(self) -> None:
        # A new parent can adopt a mounted owner, never replace it behind the
        # other parents that already share this child or ordinary occurrence.
        mounted = {}
        for candidate in self.plastic_candidates:
            owner = candidate.bank_owner
            if getattr(owner, "_is_query_owned", False):
                previous = mounted.setdefault(owner.slot_ref, owner)
                if previous is not owner:
                    raise ValueError(
                        "distinct mounted Bank owners share an identity; reuse the same owner "
                        "or give independent Banks distinct bank_owner_id values"
                    )
        for candidate in self.plastic_candidates:
            owner = mounted.get(candidate.bank_slot_ref)
            if owner is not None and candidate.bank_owner is not owner:
                current = candidate.bank_owner
                if (
                    owner.value.shape != current.value.shape or owner.value.dtype != current.value.dtype
                    or owner.value.device != current.value.device
                    or not owner.value.equal(current.value) or not owner.revision.equal(current.revision)
                ):
                    raise ValueError("shared Bank owner occurrences must start from identical state")
                candidate._bind_bank_owner(owner)
        super()._bind_shared_bank_owners()
        owners = {owner.slot_ref: owner for owner in self.owner_states}
        # Child owner registries must reference the same objects as their leaves.
        # Rebinding leaves alone would make child save/move/commit use stale owners.
        for module in tuple(self.modules()):
            if module is not self and isinstance(module, FormulaProgramQueryV5):
                module.owner_states = nn.ModuleList(owners[owner.slot_ref] for owner in module.owner_states)

    def contract_config(self) -> dict[str, object]:
        config = super().contract_config()
        config.pop("terminal_slot")
        config["terminal_slots"] = dict(self.terminal_slots)
        config["termination"] = "all-required-named-outputs"
        return config

    def _candidate_budget_eligible(
        self, candidate: FormulaProgramSearchCandidateV4, arena: _FormulaProgramExecutionArenaV4,
        *, steps: int,
    ) -> bool:
        if steps >= self.max_steps:
            return False
        if isinstance(candidate, FormulaProgramEffectCandidateV3):
            if self.max_effect_steps is not None and arena.effect_steps >= self.max_effect_steps:
                return False
        elif self.max_tensor_steps is not None and arena.tensor_steps >= self.max_tensor_steps:
            return False
        return True

    def _stop_eligible(self, arena: _FormulaProgramExecutionArenaV4, *, steps: int) -> bool:
        return (
            steps >= self.min_steps and arena.tensor_steps >= self.min_tensor_steps
            and all(arena.values.get(slot) is not None for slot in self.terminal_slots.values())
        )

    def commit_(self, execution: FormulaProgramQueryExecutionV5) -> None:
        if not isinstance(execution, FormulaProgramQueryExecutionV5):
            raise TypeError("execution must be a FormulaProgramQueryExecutionV5 winner")
        if execution._owner_token is not self._owner_token or not execution.trace.stopped:
            raise ValueError("only this ProgramQuery's stopped winner can be committed")
        for owner in self.owner_states:
            owner.install_(execution.bank_state)

    def reexecute(
        self, candidate_id: str, values: Mapping[str, Tensor], *, bank_state: FormulaProgramBankState,
    ) -> Mapping[str, Tensor]:
        matches = tuple(
            candidate for candidate in self.candidates
            if isinstance(candidate, FormulaProgramTensorCandidateV3)
            and candidate.bank_slot_ref is not None and candidate.candidate_id == candidate_id
        )
        if len(matches) != 1:
            raise ValueError("candidate_id must name one plastic ordinary occurrence")
        candidate = matches[0]
        arena = self._arena(values, bank_state=bank_state)
        if not candidate.accepts(arena):
            raise ValueError("values do not satisfy the selected ordinary occurrence")
        executed = candidate(arena)
        return MappingProxyType(_read_outputs(executed, {slot: slot for slot in candidate.output_slot_ids}))

    def forward(
        self, values: Mapping[str, Tensor], *, bank_state: FormulaProgramBankState | None = None,
        _call_entry: _FormulaProgramExecutionArenaV4 | None = None,
        _replay_trace: FormulaProgramQueryTraceV5 | None = None,
    ) -> FormulaProgramQueryExecutionV5:
        if _call_entry is None:
            entry = self._arena(values, bank_state=bank_state)
        else:
            if bank_state is not None:
                raise ValueError("child call already supplies its Bank overlay")
            # Keep nn.Module call hooks active, including input transformations.
            entry = replace(_call_entry, values=FormulaProgramArena.from_mapping(self.slot_ids, values))
        if _replay_trace is not None:
            return self._replay_arena(entry, _replay_trace)
        return self._execute_arena(entry)

    def _replay_arena(
        self, entry: _FormulaProgramExecutionArenaV4, saved: FormulaProgramQueryTraceV5,
    ) -> FormulaProgramQueryExecutionV5:
        """Replay recorded serial choices, recomputing values through real calls."""
        if not isinstance(saved, FormulaProgramQueryTraceV5) or not saved.stopped:
            raise ValueError("replay requires a completed named-output trace")
        if saved.invocation_path != entry.invocation_path:
            raise ValueError("replay invocation path differs")
        arena = entry
        trace = []
        for steps, recorded in enumerate(saved.steps):
            if recorded.step != steps or recorded.candidate_id not in self.action_ids:
                raise ValueError("replay action or step differs")
            index = self.action_ids.index(recorded.candidate_id)
            if not bool(self.eligible(arena, steps=steps)[index]):
                raise ValueError("replay action is not ready")
            if recorded.candidate_id == "stop":
                if steps != len(saved.steps) - 1:
                    raise ValueError("replay contains work after STOP")
                result = self._finish_execution(arena, trace, steps=steps)
                if result.trace.steps[-1] != recorded:
                    raise ValueError("replay STOP record differs")
                return result
            candidate = self.candidates[index]
            if (recorded.atom_ref != candidate.atom_ref
                    or recorded.input_slots != tuple(candidate.input_slots.values())
                    or recorded.input_bindings != tuple(candidate.input_slots.items())
                    or recorded.output_slots != candidate.output_slot_ids):
                raise ValueError("replay candidate binding differs")
            if isinstance(candidate, FormulaProgramCallCandidateV1):
                if recorded.child_trace is None:
                    raise ValueError("CALL replay requires a child decision trace")
                arena = candidate(arena, _replay_trace=recorded.child_trace)
            else:
                if recorded.child_trace is not None:
                    raise ValueError("ordinary replay cannot contain a child trace")
                arena = candidate(arena)
            actual = self._trace_step(candidate, arena, steps=steps)
            if actual != recorded:
                raise ValueError("replay producer or child graph differs")
            trace.append(actual)
        raise ValueError("replay ends before STOP")

    def _execute_arena(self, entry: _FormulaProgramExecutionArenaV4) -> FormulaProgramQueryExecutionV5:
        trace = []
        for steps, candidate, arena in self._walk_arena(entry):
            if candidate is None:
                return self._finish_execution(arena, trace, steps=steps)
            trace.append(self._trace_step(candidate, arena, steps=steps))
        raise RuntimeError("ProgramQuery execution ended without STOP")

    def _trace_step(
        self, candidate: FormulaProgramSearchCandidateV4 | FormulaProgramCallCandidateV1,
        arena: _FormulaProgramExecutionArenaV4, *, steps: int,
    ) -> FormulaProgramQueryTraceStepV5:
        lineage = tuple(arena.producer(slot) for slot in candidate.output_slot_ids)
        return FormulaProgramQueryTraceStepV5(
            steps, candidate.candidate_id, candidate.atom_ref, tuple(candidate.input_slots.values()),
            candidate.output_slot_ids,
            tuple(None if producer is None or producer.plastic_slot is None else producer.owner_id
                  for producer in lineage),
            arena.call_traces[-1] if isinstance(candidate, FormulaProgramCallCandidateV1) else None,
            tuple(candidate.input_slots.items()),
        )

    def _finish_execution(
        self, arena: _FormulaProgramExecutionArenaV4, trace: Sequence[FormulaProgramQueryTraceStepV5], *, steps: int,
    ) -> FormulaProgramQueryExecutionV5:
        if not self._stop_eligible(arena, steps=steps):
            raise ValueError("execution must satisfy every required head and local minimum before STOP")
        return FormulaProgramQueryExecutionV5(
            _read_outputs(arena, self.terminal_slots),
            {name: arena.producer(slot) for name, slot in self.terminal_slots.items()},
            arena.committed_state(), arena.proposals,
            FormulaProgramQueryTraceV5(
                (*trace, FormulaProgramQueryTraceStepV5(steps, "stop", None, (), (), ())),
                True, arena.invocation_path,
            ), self._owner_token,
        )


__all__ = [
    "FormulaProgramCandidateV3", "FormulaProgramTensorCandidateV4", "FormulaProgramQueryV5",
    "FormulaProgramQueryExecutionV5", "FormulaProgramQueryTraceV5", "FormulaProgramQueryTraceStepV5",
]
