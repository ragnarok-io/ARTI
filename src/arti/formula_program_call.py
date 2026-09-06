"""Returning calls to complete Query-selected Formula subgraphs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING, ClassVar

from torch import nn

from .formula_program_query import FormulaProgramArena, _require_name
from .formula_program_query_v4 import _FormulaProgramExecutionArenaV4
from .formula_v2 import FormulaV2Error

if TYPE_CHECKING:
    from .formula_program_query_v5 import FormulaProgramQueryExecutionV5, FormulaProgramQueryV5


class FormulaProgramCallCandidateV1(nn.Module):
    """Invoke a child Query and return all heads and functional Bank effects.

    Input keys are child entry slots; output keys are child terminal names.
    Values on both mappings are parent arena slots. Shared child instances and
    Bank owner identities remain shared; this call never installs Bank state.
    """

    _component_reference: ClassVar[str] = "arti/formula-program-call-candidate@1"
    atom_ref: ClassVar[None] = None

    def __init__(
        self, candidate_id: str, child: FormulaProgramQueryV5, *,
        input_slots: Mapping[str, str], output_slots: Mapping[str, str],
        requires_empty_slots: Sequence[str] = (),
    ) -> None:
        from .formula_program_query_v5 import FormulaProgramQueryV5, _named_slots

        super().__init__()
        _require_name(candidate_id, field="candidate_id")
        if not isinstance(child, FormulaProgramQueryV5):
            raise TypeError("child must be a named-output FormulaProgramQuery@5")
        inputs = _named_slots(input_slots, field="input_slots")
        outputs = _named_slots(output_slots, field="output_slots")
        produced = {slot for candidate in child.candidates for slot in candidate.output_slot_ids}
        required = {slot for candidate in child.candidates for slot in candidate.input_slots.values()}
        required.update(child.terminal_slots.values())
        entry_slots = required - produced
        if set(inputs) != entry_slots:
            raise ValueError("input_slots must bind exactly the child's external entry slots")
        if set(outputs) != set(child.terminal_slots):
            raise ValueError("output_slots must bind every child terminal name")
        if len(set(outputs.values())) != len(outputs):
            raise ValueError("child outputs must use distinct parent SSA slots")
        empty = tuple(requires_empty_slots)
        for slot in empty:
            _require_name(slot, field="requires_empty_slots")
        self.candidate_id = candidate_id
        self.child = child
        self.input_slots = inputs
        self.output_slots = outputs
        self.requires_empty_slots = empty

    @property
    def output_slot(self) -> str:
        return next(iter(self.output_slots.values()))

    @property
    def output_slot_ids(self) -> tuple[str, ...]:
        return tuple(self.output_slots.values())

    def contract_config(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "child_ref": self.child._component_reference,
            "input_slots": dict(self.input_slots),
            "output_slots": dict(self.output_slots),
            "requires_empty_slots": list(self.requires_empty_slots),
            "state": "functional-parent-overlay-no-install",
            "lineage": "original-producer-through-named-ports",
            "dispatch": "child-local-query-until-all-heads-stop",
        }

    def _entry(self, arena: _FormulaProgramExecutionArenaV4, *, _input_ports=None) -> _FormulaProgramExecutionArenaV4:
        values = {}
        lineage = {}
        if _input_ports is not None and set(_input_ports) != set(self.input_slots):
            raise ValueError("child input view must bind every named port")
        for port, slot in self.input_slots.items():
            value, producer = ((arena.values.get(slot), arena.producer(slot)) if _input_ports is None
                               else _input_ports[port])
            if value is None:
                raise ValueError(f"child input slot {slot!r} is empty")
            values[port] = value
            lineage[port] = None if producer is None else replace(producer, output_slot=port)
        # Keep the parent's current tensors by identity, including prior effects.
        # This also lets a child effect operate on an incoming real producer.
        return _FormulaProgramExecutionArenaV4(
            FormulaProgramArena.from_mapping(self.child.slot_ids, values),
            tuple(lineage.get(slot) for slot in self.child.slot_ids),
            arena.committed_state(),
            invocation_path=(*arena.invocation_path, self.candidate_id),
        )

    def accepts(self, arena: _FormulaProgramExecutionArenaV4) -> bool:
        try:
            if any(arena.values.get(slot) is not None for slot in (
                *self.output_slot_ids, *self.requires_empty_slots,
            )):
                return False
            entry = self._entry(arena)
            from .formula_program_query_v5 import FormulaProgramQueryV5, FormulaProgramTensorCandidateV4
            from .formula_program_query_v4 import FormulaProgramTensorCandidateV3, FormulaProgramEffectCandidateV3

            native = (FormulaProgramTensorCandidateV3, FormulaProgramTensorCandidateV4, FormulaProgramEffectCandidateV3)
            if (
                type(self.child) is FormulaProgramQueryV5
                and not any(name in self.child.__dict__ for name in ("eligible", "_candidate_eligible", "_stop_eligible"))
                and all(type(candidate) in native and not any(
                    name in candidate.__dict__ for name in ("accepts", "_bindings", "_target")
                ) for candidate in self.child.candidates)
            ):
                return self.child._has_eligible(entry, steps=0)
            return bool(self.child.eligible(entry, steps=0).any())
        except (FormulaV2Error, KeyError, TypeError, ValueError):
            return False

    def forward(
        self, arena: _FormulaProgramExecutionArenaV4, *, _replay_trace=None, _input_ports=None,
    ) -> _FormulaProgramExecutionArenaV4:
        if any(arena.values.get(slot) is not None for slot in (
            *self.output_slot_ids, *self.requires_empty_slots,
        )):
            raise ValueError("child call requires empty output and guard slots")
        entry = self._entry(arena) if _input_ports is None else self._entry(arena, _input_ports=_input_ports)
        options = {"_call_entry": entry}
        if _replay_trace is not None:
            options["_replay_trace"] = _replay_trace
        result = self.child({port: entry.values.get(port) for port in self.input_slots}, **options)
        return self._return_result(arena, result)

    def _return_result(
        self, arena: _FormulaProgramExecutionArenaV4, result: FormulaProgramQueryExecutionV5,
    ) -> _FormulaProgramExecutionArenaV4:
        updated = arena
        for proposal in result.proposals:
            updated = updated.append_proposal(proposal)
        outputs = {}
        producers = {}
        for name, slot in self.output_slots.items():
            outputs[slot] = result.outputs[name]
            producer = result.output_producers[name]
            producers[slot] = None if producer is None else replace(producer, output_slot=slot)
        updated = updated.write_many(outputs, producers=producers)
        return replace(updated, call_traces=(*updated.call_traces, result.trace))


__all__ = ["FormulaProgramCallCandidateV1"]
