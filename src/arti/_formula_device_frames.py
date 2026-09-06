"""Bounded tensor-only SSA/frame transitions for FormulaProgramQuery@5.

This module is deliberately an integration boundary, not another Formula
executor.  Host code prepares :class:`FormulaDeviceFrameSpec` from real
QueryV5/Call objects.  The runtime kernel receives a selected global action and
the handles produced by the native numeric executor, then performs only tensor
state transitions.  Value handles are opaque integers owned by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
from torch import Tensor, nn

from ._formula_device_sources import FormulaDeviceSources

from ._formula_device_response import is_response_query
from .formula_program_call import FormulaProgramCallCandidateV1
from .formula_program_query_v4 import (
    FormulaProgramEffectCandidateV3,
    FormulaProgramTensorCandidateV3,
)
from .formula_program_query_v5 import (
    FormulaProgramQueryV5,
    FormulaProgramTensorCandidateV4,
)


KIND_ORDINARY = 0
KIND_EFFECT = 1
KIND_CALL = 2
KIND_STOP = 3

EVENT_NONE = 0
EVENT_ORDINARY = 1
EVENT_EFFECT = 2
EVENT_CALL = 3
EVENT_RETURN = 4
EVENT_STOP = 5
EVENT_REJECT = 6


class FormulaDeviceFrameState(NamedTuple):
    """All mutable execution state, represented by tensors only.

    ``-1`` is the empty handle/index sentinel.  ``producer_candidate`` keeps
    the original ordinary producer identity; effects do not replace it.
    ``bank_value_handles`` is required for the native predecessor identity
    contract, while ``bank_revisions`` carries the logical version.
    ``response_candidate`` records the current invocation's response origin;
    CALL returns use the call identity without changing original Bank lineage.
    """

    active: Tensor
    completed: Tensor
    depth: Tensor
    frame_query: Tensor
    frame_steps: Tensor
    frame_tensor_steps: Tensor
    frame_effect_steps: Tensor
    frame_return_call: Tensor
    value_handles: Tensor
    producer_candidate: Tensor
    producer_bank: Tensor
    producer_revision: Tensor
    producer_bank_handle: Tensor
    bank_value_handles: Tensor
    bank_revisions: Tensor
    response_candidate: Tensor


class FormulaDeviceFrameEvent(NamedTuple):
    """Per-branch receipt returned by one state transition."""

    eligible: Tensor
    accepted: Tensor
    kind: Tensor
    target_bank: Tensor
    previous_revision: Tensor
    successor_revision: Tensor
    returned_handles: Tensor


@dataclass(frozen=True)
class FormulaDeviceFrameSpec:
    """Static host-prepared mapping for a recursive QueryV5 graph."""

    query_count: int
    max_slots: int
    max_outputs: int
    max_ports: int
    max_depth: int
    candidate_ids: tuple[str, ...]
    query_ids: tuple[int, ...]
    candidate_query: Tensor
    candidate_kind: Tensor
    candidate_required: Tensor
    candidate_input_slots: Tensor
    candidate_effect_port: Tensor
    candidate_empty: Tensor
    candidate_output_count: Tensor
    candidate_output_slots: Tensor
    candidate_output_bank: Tensor
    candidate_effect_input: Tensor
    candidate_child_query: Tensor
    candidate_call_parent_inputs: Tensor
    candidate_call_child_inputs: Tensor
    candidate_call_parent_outputs: Tensor
    candidate_call_child_outputs: Tensor
    query_terminal_count: Tensor
    query_terminal_slots: Tensor
    query_min_steps: Tensor
    query_max_steps: Tensor
    query_min_tensor_steps: Tensor
    query_max_tensor_steps: Tensor
    query_max_effect_steps: Tensor
    initial_bank_revisions: Tensor

    @classmethod
    def from_query(
        cls,
        query: FormulaProgramQueryV5,
        *,
        max_depth: int | None = None,
    ) -> "FormulaDeviceFrameSpec":
        """Prepare real QueryV5/Call wiring without reading runtime tensors."""
        if type(query) is not FormulaProgramQueryV5 and not is_response_query(query):
            raise TypeError("root query must be FormulaProgramQueryV5 or V6")

        queries: list[FormulaProgramQueryV5] = []
        visiting: set[int] = set()
        visited: set[int] = set()

        def visit(current: FormulaProgramQueryV5) -> None:
            identity = id(current)
            if identity in visiting:
                raise ValueError("recursive Call graph is not bounded")
            if identity in visited:
                return
            visiting.add(identity)
            queries.append(current)
            for candidate in current.candidates:
                if isinstance(candidate, FormulaProgramCallCandidateV1):
                    visit(candidate.child)
            visiting.remove(identity)
            visited.add(identity)

        visit(query)
        query_index = {id(item): index for index, item in enumerate(queries)}

        call_depth: dict[int, int] = {}

        def depth_of(current: FormulaProgramQueryV5) -> int:
            identity = id(current)
            if identity in call_depth:
                return call_depth[identity]
            child_depth = 0
            for candidate in current.candidates:
                if isinstance(candidate, FormulaProgramCallCandidateV1):
                    child_depth = max(child_depth, 1 + depth_of(candidate.child))
            call_depth[identity] = child_depth
            return child_depth

        required_depth = depth_of(query) + 1
        if max_depth is None:
            max_depth = required_depth
        if type(max_depth) is not int or max_depth < required_depth:
            raise ValueError("max_depth must cover every static Call depth")

        bank_refs = tuple(query.initial_bank_state().slot_refs)
        bank_index = {ref: index for index, ref in enumerate(bank_refs)}
        initial_revisions = torch.tensor(
            query.initial_bank_state().revisions, dtype=torch.int64,
        )

        max_slots = max(len(item.slot_ids) for item in queries)
        max_outputs = max(
            1,
            *(len(item.terminal_slots) for item in queries),
            *(len(candidate.output_slot_ids) for item in queries for candidate in item.candidates),
        )
        max_ports = max(
            1,
            *(len(candidate.input_slots) for item in queries for candidate in item.candidates),
        )

        candidate_ids: list[str] = []
        candidate_query: list[int] = []
        candidate_kind: list[int] = []
        candidate_required: list[list[bool]] = []
        candidate_input_slots: list[list[int]] = []
        candidate_effect_port: list[int] = []
        candidate_empty: list[list[bool]] = []
        candidate_output_count: list[int] = []
        candidate_output_slots: list[list[int]] = []
        candidate_output_bank: list[list[int]] = []
        candidate_effect_input: list[int] = []
        candidate_child_query: list[int] = []
        call_parent_inputs: list[list[int]] = []
        call_child_inputs: list[list[int]] = []
        call_parent_outputs: list[list[int]] = []
        call_child_outputs: list[list[int]] = []

        def empty_slots() -> list[bool]:
            return [False] * max_slots

        def padded(values: list[int], width: int) -> list[int]:
            return values + [-1] * (width - len(values))

        def output_banks(candidate: FormulaProgramTensorCandidateV3) -> list[int]:
            ref = candidate.bank_slot_ref
            if ref is None:
                return [-1] * len(candidate.output_slot_ids)
            if isinstance(candidate, FormulaProgramTensorCandidateV4):
                result = []
                for name in candidate.candidate.program.outputs:
                    owns = candidate.plastic_bank_slot in candidate._head_dependencies[name]
                    result.append(bank_index[ref] if owns else -1)
                return result
            return [bank_index[ref]] * len(candidate.output_slot_ids)

        for current in queries:
            qid = query_index[id(current)]
            slots = {name: index for index, name in enumerate(current.slot_ids)}
            for candidate in current.candidates:
                candidate_type = type(candidate)
                if candidate_type not in (
                    FormulaProgramTensorCandidateV3,
                    FormulaProgramTensorCandidateV4,
                    FormulaProgramEffectCandidateV3,
                    FormulaProgramCallCandidateV1,
                ):
                    raise TypeError(
                        f"candidate {candidate.candidate_id!r} has unsupported type "
                        f"{candidate_type.__name__}; use the native fallback"
                    )
                candidate_ids.append(candidate.candidate_id)
                candidate_query.append(qid)
                required = empty_slots()
                empty = empty_slots()
                for slot in candidate.input_slots.values():
                    required[slots[slot]] = True
                for slot in (*candidate.output_slot_ids, *candidate.requires_empty_slots):
                    empty[slots[slot]] = True
                candidate_required.append(required)
                candidate_input_slots.append(padded(
                    [slots[slot] for slot in candidate.input_slots.values()], max_ports,
                ))
                candidate_effect_port.append(
                    tuple(candidate.input_slots).index(candidate.effect_program.data_input_name)
                    if isinstance(candidate, FormulaProgramEffectCandidateV3) else -1
                )
                candidate_empty.append(empty)
                candidate_output_count.append(len(candidate.output_slot_ids))
                outputs = [slots[slot] for slot in candidate.output_slot_ids]
                candidate_output_slots.append(padded(outputs, max_outputs))
                candidate_output_bank.append(
                    padded(
                        output_banks(candidate)
                        if candidate_type in (
                            FormulaProgramTensorCandidateV3,
                            FormulaProgramTensorCandidateV4,
                        )
                        else [-1] * len(outputs),
                        max_outputs,
                    )
                )
                candidate_effect_input.append(
                    slots[candidate.input_slot]
                    if isinstance(candidate, FormulaProgramEffectCandidateV3)
                    else -1
                )
                if isinstance(candidate, FormulaProgramCallCandidateV1):
                    child = candidate.child
                    child_slots = {name: index for index, name in enumerate(child.slot_ids)}
                    candidate_child_query.append(query_index[id(child)])
                    call_parent_inputs.append(
                        padded([slots[slot] for slot in candidate.input_slots.values()], max_ports)
                    )
                    call_child_inputs.append(
                        padded([child_slots[port] for port in candidate.input_slots], max_ports)
                    )
                    call_parent_outputs.append(
                        padded([slots[slot] for slot in candidate.output_slots.values()], max_outputs)
                    )
                    call_child_outputs.append(
                        padded(
                            [child_slots[child.terminal_slots[name]] for name in candidate.output_slots],
                            max_outputs,
                        )
                    )
                else:
                    candidate_child_query.append(-1)
                    call_parent_inputs.append([-1] * max_ports)
                    call_child_inputs.append([-1] * max_ports)
                    call_parent_outputs.append([-1] * max_outputs)
                    call_child_outputs.append([-1] * max_outputs)
                candidate_kind.append(
                    KIND_CALL
                    if candidate_type is FormulaProgramCallCandidateV1
                    else KIND_EFFECT
                    if candidate_type is FormulaProgramEffectCandidateV3
                    else KIND_ORDINARY
                )
            candidate_ids.append("stop")
            candidate_query.append(qid)
            candidate_kind.append(KIND_STOP)
            candidate_required.append(empty_slots())
            candidate_input_slots.append([-1] * max_ports)
            candidate_effect_port.append(-1)
            candidate_empty.append(empty_slots())
            candidate_output_count.append(0)
            candidate_output_slots.append([-1] * max_outputs)
            candidate_output_bank.append([-1] * max_outputs)
            candidate_effect_input.append(-1)
            candidate_child_query.append(-1)
            call_parent_inputs.append([-1] * max_ports)
            call_child_inputs.append([-1] * max_ports)
            call_parent_outputs.append([-1] * max_outputs)
            call_child_outputs.append([-1] * max_outputs)

        terminal_slots = []
        terminal_count = []
        for current in queries:
            slots = {name: index for index, name in enumerate(current.slot_ids)}
            values = [slots[slot] for slot in current.terminal_slots.values()]
            terminal_count.append(len(values))
            terminal_slots.append(padded(values, max_outputs))

        def optional_limit(value: int | None) -> int:
            return -1 if value is None else int(value)

        return cls(
            query_count=len(queries),
            max_slots=max_slots,
            max_outputs=max_outputs,
            max_ports=max_ports,
            max_depth=max_depth,
            candidate_ids=tuple(candidate_ids),
            query_ids=tuple(range(len(queries))),
            candidate_query=torch.tensor(candidate_query, dtype=torch.int64),
            candidate_kind=torch.tensor(candidate_kind, dtype=torch.int64),
            candidate_required=torch.tensor(candidate_required, dtype=torch.bool),
            candidate_input_slots=torch.tensor(candidate_input_slots, dtype=torch.int64),
            candidate_effect_port=torch.tensor(candidate_effect_port, dtype=torch.int64),
            candidate_empty=torch.tensor(candidate_empty, dtype=torch.bool),
            candidate_output_count=torch.tensor(candidate_output_count, dtype=torch.int64),
            candidate_output_slots=torch.tensor(candidate_output_slots, dtype=torch.int64),
            candidate_output_bank=torch.tensor(candidate_output_bank, dtype=torch.int64),
            candidate_effect_input=torch.tensor(candidate_effect_input, dtype=torch.int64),
            candidate_child_query=torch.tensor(candidate_child_query, dtype=torch.int64),
            candidate_call_parent_inputs=torch.tensor(call_parent_inputs, dtype=torch.int64),
            candidate_call_child_inputs=torch.tensor(call_child_inputs, dtype=torch.int64),
            candidate_call_parent_outputs=torch.tensor(call_parent_outputs, dtype=torch.int64),
            candidate_call_child_outputs=torch.tensor(call_child_outputs, dtype=torch.int64),
            query_terminal_count=torch.tensor(terminal_count, dtype=torch.int64),
            query_terminal_slots=torch.tensor(terminal_slots, dtype=torch.int64),
            query_min_steps=torch.tensor([item.min_steps for item in queries], dtype=torch.int64),
            query_max_steps=torch.tensor([item.max_steps for item in queries], dtype=torch.int64),
            query_min_tensor_steps=torch.tensor(
                [item.min_tensor_steps for item in queries], dtype=torch.int64,
            ),
            query_max_tensor_steps=torch.tensor(
                [optional_limit(item.max_tensor_steps) for item in queries], dtype=torch.int64,
            ),
            query_max_effect_steps=torch.tensor(
                [optional_limit(item.max_effect_steps) for item in queries], dtype=torch.int64,
            ),
            initial_bank_revisions=initial_revisions,
        )


class FormulaDeviceFrameKernel(nn.Module):
    """Tensor-only one-action transition over a prepared QueryV5 graph."""

    def __init__(self, spec: FormulaDeviceFrameSpec) -> None:
        super().__init__()
        self.spec = spec
        for name in (
            "candidate_query", "candidate_kind", "candidate_required", "candidate_empty",
            "candidate_input_slots", "candidate_effect_port",
            "candidate_output_count", "candidate_output_slots", "candidate_output_bank",
            "candidate_effect_input", "candidate_child_query", "candidate_call_parent_inputs",
            "candidate_call_child_inputs", "candidate_call_parent_outputs",
            "candidate_call_child_outputs", "query_terminal_count", "query_terminal_slots",
            "query_min_steps", "query_max_steps", "query_min_tensor_steps",
            "query_max_tensor_steps", "query_max_effect_steps", "initial_bank_revisions",
        ):
            self.register_buffer(name, getattr(spec, name), persistent=False)

    @classmethod
    def from_query(
        cls,
        query: FormulaProgramQueryV5,
        *,
        max_depth: int | None = None,
    ) -> "FormulaDeviceFrameKernel":
        return cls(FormulaDeviceFrameSpec.from_query(query, max_depth=max_depth))

    @property
    def candidate_count(self) -> int:
        return len(self.spec.candidate_ids)

    @property
    def bank_count(self) -> int:
        return int(self.initial_bank_revisions.numel())

    def candidate_id(self, query_id: int, local_index: int) -> int:
        """Return a prepared global id for host-side integration/tests."""
        offset = 0
        for qid in range(query_id):
            offset += self.spec.candidate_ids[offset:].index("stop") + 1
        return offset + local_index

    def initial_state(
        self,
        branch_count: int,
        value_handles: Tensor,
        *,
        bank_value_handles: Tensor | None = None,
        bank_revisions: Tensor | None = None,
    ) -> FormulaDeviceFrameState:
        """Create root frames from opaque caller-owned handle indices."""
        if type(branch_count) is not int or branch_count <= 0:
            raise ValueError("branch_count must be positive")
        if value_handles.ndim != 2 or value_handles.shape[0] != branch_count:
            raise ValueError("value_handles must have shape [F, root_slots]")
        if value_handles.shape[1] > self.spec.max_slots:
            raise ValueError("value_handles exceeds prepared root slot capacity")
        device = value_handles.device
        dtype = value_handles.dtype
        if dtype is not torch.int64:
            raise TypeError("value_handles must be int64 handles")
        values = torch.full(
            (branch_count, self.spec.max_depth, self.spec.max_slots), -1,
            dtype=dtype, device=device,
        )
        values[:, 0, :value_handles.shape[1]] = value_handles
        shape_f = (branch_count, self.spec.max_depth)
        frame_query = torch.full(shape_f, -1, dtype=torch.int64, device=device)
        frame_query[:, 0] = 0
        zeros = torch.zeros(shape_f, dtype=torch.int64, device=device)
        frame_return = torch.full(shape_f, -1, dtype=torch.int64, device=device)
        shape_ssa = (branch_count, self.spec.max_depth, self.spec.max_slots)
        ssa_minus = torch.full(shape_ssa, -1, dtype=torch.int64, device=device)
        if bank_revisions is None:
            revisions = self.initial_bank_revisions.to(device=device).expand(branch_count, -1).clone()
        elif bank_revisions.ndim == 1:
            revisions = bank_revisions.to(device=device).expand(branch_count, -1).clone()
        else:
            revisions = bank_revisions.to(device=device).clone()
        if revisions.shape != (branch_count, self.bank_count):
            raise ValueError("bank_revisions must have shape [F, bank_count]")
        if bank_value_handles is None:
            bank_handles = torch.full_like(revisions, -1)
        elif bank_value_handles.ndim == 1:
            bank_handles = bank_value_handles.to(device=device).expand(branch_count, -1).clone()
        else:
            bank_handles = bank_value_handles.to(device=device).clone()
        if bank_handles.shape != (branch_count, self.bank_count):
            raise ValueError("bank_value_handles must have shape [F, bank_count]")
        return FormulaDeviceFrameState(
            torch.ones(branch_count, dtype=torch.bool, device=device),
            torch.zeros(branch_count, dtype=torch.bool, device=device),
            torch.zeros(branch_count, dtype=torch.int64, device=device),
            frame_query,
            zeros.clone(), zeros.clone(), zeros.clone(), frame_return,
            values, ssa_minus.clone(), ssa_minus.clone(), ssa_minus.clone(), ssa_minus.clone(),
            bank_handles, revisions, ssa_minus.clone(),
        )

    @staticmethod
    def _frame(tensor: Tensor, depth: Tensor) -> Tensor:
        depth_index = depth.view(-1, 1, 1).expand(-1, 1, tensor.shape[-1])
        return tensor.gather(1, depth_index).squeeze(1)

    @staticmethod
    def _frame_scalar(tensor: Tensor, depth: Tensor) -> Tensor:
        return tensor.gather(1, depth.view(-1, 1)).squeeze(1)

    @staticmethod
    def _write_depth(
        tensor: Tensor, depth: Tensor, value: Tensor, rows: Tensor,
    ) -> Tensor:
        axis = torch.arange(tensor.shape[1], device=tensor.device).view(1, -1, 1)
        mask = rows.view(-1, 1, 1) & axis.eq(depth.view(-1, 1, 1))
        return torch.where(mask, value[:, None, :], tensor)

    @staticmethod
    def _write_frame(
        tensor: Tensor, depth: Tensor, value: Tensor, rows: Tensor,
    ) -> Tensor:
        axis = torch.arange(tensor.shape[1], device=tensor.device).view(1, -1)
        mask = rows.view(-1, 1) & axis.eq(depth.view(-1, 1))
        return torch.where(mask, value[:, None], tensor)

    @staticmethod
    def _write_slot(
        frame: Tensor, slot: Tensor, value: Tensor, rows: Tensor,
    ) -> Tensor:
        axis = torch.arange(frame.shape[-1], device=frame.device).view(1, -1)
        return torch.where(rows[:, None] & axis.eq(slot[:, None]), value[:, None], frame)

    def forward(
        self,
        state: FormulaDeviceFrameState,
        selected_candidate: Tensor,
        numeric_output_handles: Tensor,
        numeric_valid: Tensor,
        numeric_bank_handle: Tensor,
        input_sources: FormulaDeviceSources | None = None,
    ) -> tuple[FormulaDeviceFrameState, FormulaDeviceFrameEvent]:
        """Apply one selected action; no tensor value is inspected on host.

        ``numeric_output_handles`` is ``[F,H]`` and ``numeric_bank_handle`` is
        used only by EFFECT.  CALL and STOP ignore both numeric inputs.  An
        ineligible or invalid numeric action leaves state unchanged and is
        reported as ``accepted=False`` for the caller's refill logic.
        """
        # Plain tuple inputs also support export without named-field guard aliases.
        state = FormulaDeviceFrameState(*state)
        active, completed, depth = state.active, state.completed, state.depth
        current_depth = depth.clamp(0, self.spec.max_depth - 1)
        current_query = self._frame_scalar(state.frame_query, current_depth)
        current_steps = self._frame_scalar(state.frame_steps, current_depth)
        current_tensor_steps = self._frame_scalar(state.frame_tensor_steps, current_depth)
        current_effect_steps = self._frame_scalar(state.frame_effect_steps, current_depth)
        current_values = self._frame(state.value_handles, current_depth)
        current_producer = self._frame(state.producer_candidate, current_depth)
        current_bank = self._frame(state.producer_bank, current_depth)
        current_revision = self._frame(state.producer_revision, current_depth)
        current_bank_handle = self._frame(state.producer_bank_handle, current_depth)
        occupied = current_values >= 0

        in_range = (selected_candidate >= 0) & (selected_candidate < self.candidate_count)
        safe_candidate = selected_candidate.clamp(0, self.candidate_count - 1)
        candidate_query = self.candidate_query.index_select(0, safe_candidate)
        candidate_kind = self.candidate_kind.index_select(0, safe_candidate)
        candidate_required = self.candidate_required.index_select(0, safe_candidate)
        candidate_empty = self.candidate_empty.index_select(0, safe_candidate)
        candidate_outputs = self.candidate_output_slots.index_select(0, safe_candidate)
        candidate_output_bank = self.candidate_output_bank.index_select(0, safe_candidate)
        output_count = self.candidate_output_count.index_select(0, safe_candidate)
        required_ok = ((~candidate_required) | occupied).all(-1)
        if input_sources is not None:
            input_sources = FormulaDeviceSources(*input_sources)
            required_ok = (input_sources.ready & input_sources.finite).all(-1)
        empty_ok = ((~candidate_empty) | ~occupied).all(-1)
        structure_ok = required_ok & empty_ok
        same_query = in_range & candidate_query.eq(current_query)

        max_tensor = self.query_max_tensor_steps.index_select(0, current_query.clamp_min(0))
        max_effect = self.query_max_effect_steps.index_select(0, current_query.clamp_min(0))
        budget_ok = current_steps < self.query_max_steps.index_select(0, current_query.clamp_min(0))
        budget_ok = budget_ok & ((candidate_kind != KIND_EFFECT) | (max_effect < 0) | (current_effect_steps < max_effect))
        budget_ok = budget_ok & ((candidate_kind == KIND_EFFECT) | (max_tensor < 0) | (current_tensor_steps < max_tensor))

        terminal_slots = self.query_terminal_slots.index_select(0, current_query.clamp_min(0))
        terminal_count = self.query_terminal_count.index_select(0, current_query.clamp_min(0))
        terminal_safe = terminal_slots.clamp_min(0)
        terminal_occupied = current_values.gather(1, terminal_safe) >= 0
        terminal_ok = terminal_occupied | (torch.arange(self.spec.max_outputs, device=current_values.device)[None, :] >= terminal_count[:, None])
        terminal_ok = terminal_ok.all(-1)
        is_stop = candidate_kind.eq(KIND_STOP)
        is_call = candidate_kind.eq(KIND_CALL)
        is_effect = candidate_kind.eq(KIND_EFFECT)
        is_ordinary = candidate_kind.eq(KIND_ORDINARY)
        stop_ok = (
            current_steps >= self.query_min_steps.index_select(0, current_query.clamp_min(0))
        ) & (
            current_tensor_steps >= self.query_min_tensor_steps.index_select(0, current_query.clamp_min(0))
        ) & terminal_ok
        child_query = self.candidate_child_query.index_select(0, safe_candidate)
        call_depth_ok = (~is_call) | (current_depth + 1 < self.spec.max_depth)
        eligible = active & same_query & structure_ok & call_depth_ok & torch.where(
            is_stop, stop_ok, budget_ok,
        )

        output_axis = torch.arange(self.spec.max_outputs, device=current_values.device)
        output_mask = output_axis[None, :] < output_count[:, None]
        output_handles_ok = ((~output_mask) | (numeric_output_handles >= 0)).all(-1)
        effect_input_slot = self.candidate_effect_input.index_select(0, safe_candidate).clamp_min(0)
        effect_input_handle = current_values.gather(1, effect_input_slot[:, None]).squeeze(1)
        effect_producer = current_producer.gather(1, effect_input_slot[:, None]).squeeze(1)
        effect_bank = current_bank.gather(1, effect_input_slot[:, None]).squeeze(1)
        effect_revision = current_revision.gather(1, effect_input_slot[:, None]).squeeze(1)
        effect_bank_handle = current_bank_handle.gather(1, effect_input_slot[:, None]).squeeze(1)
        if input_sources is not None:
            port = self.candidate_effect_port.index_select(0, safe_candidate).clamp_min(0)
            def effect_source(field):
                return field.gather(1, port[:, None]).squeeze(1)
            effect_input_handle = effect_source(input_sources.handles)
            effect_producer = effect_source(input_sources.producer)
            effect_bank = effect_source(input_sources.bank)
            effect_revision = effect_source(input_sources.revision)
            effect_bank_handle = effect_source(input_sources.bank_handle)
        safe_effect_bank = effect_bank.clamp(0, max(self.bank_count - 1, 0))
        if self.bank_count:
            live_bank_revision = state.bank_revisions.gather(1, safe_effect_bank[:, None]).squeeze(1)
            live_bank_handle = state.bank_value_handles.gather(1, safe_effect_bank[:, None]).squeeze(1)
        else:
            live_bank_revision = torch.full_like(effect_bank, -1)
            live_bank_handle = torch.full_like(effect_bank, -1)
        producer_in_range = (effect_producer >= 0) & (effect_producer < self.candidate_count)
        producer_is_ordinary = producer_in_range & self.candidate_kind.index_select(
            0, effect_producer.clamp(0, self.candidate_count - 1),
        ).eq(KIND_ORDINARY)
        target_ok = (
            is_effect & (effect_bank >= 0) & (effect_bank < self.bank_count) & producer_is_ordinary
            & effect_revision.eq(live_bank_revision)
            & effect_bank_handle.eq(live_bank_handle)
            & (effect_input_handle >= 0)
        )
        effect_result_ok = (
            numeric_valid & output_handles_ok & (numeric_output_handles[:, 0] == effect_input_handle)
            & (numeric_bank_handle >= 0)
        )
        numeric_ok = numeric_valid & output_handles_ok
        applied = eligible & torch.where(
            is_effect, target_ok & effect_result_ok,
            torch.where(is_call | is_stop, torch.ones_like(eligible), numeric_ok),
        )
        ordinary_applied = applied & is_ordinary
        effect_applied = applied & is_effect
        call_applied = applied & is_call
        child_stop = applied & is_stop & (current_depth > 0)
        root_stop = applied & is_stop & (current_depth == 0)

        next_bank_handles = state.bank_value_handles
        next_bank_revisions = state.bank_revisions
        bank_axis = torch.arange(self.bank_count, device=current_values.device).view(1, -1)
        bank_write = effect_applied[:, None] & bank_axis.eq(safe_effect_bank[:, None])
        next_bank_handles = torch.where(bank_write, numeric_bank_handle[:, None], next_bank_handles)
        next_bank_revisions = torch.where(
            bank_write, live_bank_revision[:, None] + 1, next_bank_revisions,
        )

        next_values = state.value_handles
        next_producer = state.producer_candidate
        next_response = state.response_candidate
        next_producer_bank = state.producer_bank
        next_producer_revision = state.producer_revision
        next_producer_bank_handle = state.producer_bank_handle
        numeric_write = ordinary_applied | effect_applied
        for output_index in range(self.spec.max_outputs):
            slot = candidate_outputs[:, output_index].clamp_min(0)
            handle = numeric_output_handles[:, output_index]
            static_bank = candidate_output_bank[:, output_index]
            safe_static_bank = static_bank.clamp_min(0)
            if self.bank_count:
                ordinary_revision = state.bank_revisions.gather(1, safe_static_bank[:, None]).squeeze(1)
                ordinary_bank_handle = state.bank_value_handles.gather(1, safe_static_bank[:, None]).squeeze(1)
            else:
                ordinary_revision = torch.full_like(static_bank, -1)
                ordinary_bank_handle = torch.full_like(static_bank, -1)
            write_bank = torch.where(is_effect, effect_bank, static_bank)
            write_revision = torch.where(is_effect, live_bank_revision + 1, ordinary_revision)
            write_bank_handle = torch.where(is_effect, numeric_bank_handle, ordinary_bank_handle)
            write_producer = torch.where(is_effect, effect_producer, safe_candidate)
            write_bank_valid = write_bank >= 0
            write_revision = torch.where(write_bank_valid, write_revision, torch.full_like(write_revision, -1))
            write_bank_handle = torch.where(write_bank_valid, write_bank_handle, torch.full_like(write_bank_handle, -1))
            row = numeric_write & (output_index < output_count)
            current_frame = self._frame(next_values, current_depth)
            current_frame = self._write_slot(current_frame, slot, handle, row)
            next_values = self._write_depth(next_values, current_depth, current_frame, row)
            current_frame = self._frame(next_producer, current_depth)
            current_frame = self._write_slot(current_frame, slot, write_producer, row)
            next_producer = self._write_depth(next_producer, current_depth, current_frame, row)
            response_frame = self._frame(next_response, current_depth)
            response_frame = self._write_slot(response_frame, slot, safe_candidate, row)
            next_response = self._write_depth(next_response, current_depth, response_frame, row)
            current_frame = self._frame(next_producer_bank, current_depth)
            current_frame = self._write_slot(current_frame, slot, write_bank, row)
            next_producer_bank = self._write_depth(next_producer_bank, current_depth, current_frame, row)
            current_frame = self._frame(next_producer_revision, current_depth)
            current_frame = self._write_slot(current_frame, slot, write_revision, row)
            next_producer_revision = self._write_depth(next_producer_revision, current_depth, current_frame, row)
            current_frame = self._frame(next_producer_bank_handle, current_depth)
            current_frame = self._write_slot(current_frame, slot, write_bank_handle, row)
            next_producer_bank_handle = self._write_depth(next_producer_bank_handle, current_depth, current_frame, row)

        next_frame_steps = state.frame_steps
        next_frame_tensor = state.frame_tensor_steps
        next_frame_effect = state.frame_effect_steps
        current_frame_steps = self._frame_scalar(next_frame_steps, current_depth)
        current_frame_steps = current_frame_steps + ordinary_applied.to(torch.int64) + effect_applied.to(torch.int64)
        next_frame_steps = self._write_frame(next_frame_steps, current_depth, current_frame_steps, ordinary_applied | effect_applied)
        current_frame_tensor = self._frame_scalar(next_frame_tensor, current_depth)
        current_frame_tensor = current_frame_tensor + ordinary_applied.to(torch.int64)
        next_frame_tensor = self._write_frame(next_frame_tensor, current_depth, current_frame_tensor, ordinary_applied)
        current_frame_effect = self._frame_scalar(next_frame_effect, current_depth)
        current_frame_effect = current_frame_effect + effect_applied.to(torch.int64)
        next_frame_effect = self._write_frame(next_frame_effect, current_depth, current_frame_effect, effect_applied)

        child_values = torch.full_like(current_values, -1)
        child_producer = torch.full_like(current_producer, -1)
        child_bank = torch.full_like(current_bank, -1)
        child_revision = torch.full_like(current_revision, -1)
        child_bank_handle = torch.full_like(current_bank_handle, -1)
        call_parent_inputs = self.candidate_call_parent_inputs.index_select(0, safe_candidate)
        call_child_inputs = self.candidate_call_child_inputs.index_select(0, safe_candidate)
        for port in range(self.spec.max_ports):
            port_valid = call_applied & (call_parent_inputs[:, port] >= 0)
            parent_slot = call_parent_inputs[:, port].clamp_min(0)
            child_slot = call_child_inputs[:, port].clamp_min(0)
            parent_values = current_values.gather(1, parent_slot[:, None]).squeeze(1)
            parent_producer = current_producer.gather(1, parent_slot[:, None]).squeeze(1)
            parent_bank = current_bank.gather(1, parent_slot[:, None]).squeeze(1)
            parent_revision = current_revision.gather(1, parent_slot[:, None]).squeeze(1)
            parent_bank_handle = current_bank_handle.gather(1, parent_slot[:, None]).squeeze(1)
            if input_sources is not None:
                parent_values = input_sources.handles[:, port]
                parent_producer = input_sources.producer[:, port]
                parent_bank = input_sources.bank[:, port]
                parent_revision = input_sources.revision[:, port]
                parent_bank_handle = input_sources.bank_handle[:, port]
            child_values = self._write_slot(child_values, child_slot, parent_values, port_valid)
            child_producer = self._write_slot(child_producer, child_slot, parent_producer, port_valid)
            child_bank = self._write_slot(child_bank, child_slot, parent_bank, port_valid)
            child_revision = self._write_slot(child_revision, child_slot, parent_revision, port_valid)
            child_bank_handle = self._write_slot(child_bank_handle, child_slot, parent_bank_handle, port_valid)
        child_depth = (current_depth + 1).clamp_max(self.spec.max_depth - 1)
        next_values = self._write_depth(next_values, child_depth, child_values, call_applied)
        next_producer = self._write_depth(next_producer, child_depth, child_producer, call_applied)
        # Imported data is not a response executed in this invocation.
        next_response = self._write_depth(next_response, child_depth, torch.full_like(child_producer, -1), call_applied)
        next_producer_bank = self._write_depth(next_producer_bank, child_depth, child_bank, call_applied)
        next_producer_revision = self._write_depth(next_producer_revision, child_depth, child_revision, call_applied)
        next_producer_bank_handle = self._write_depth(next_producer_bank_handle, child_depth, child_bank_handle, call_applied)
        next_frame_query = self._write_frame(
            state.frame_query, child_depth, child_query, call_applied,
        )
        next_frame_steps = self._write_frame(next_frame_steps, child_depth, torch.zeros_like(current_frame_steps), call_applied)
        next_frame_tensor = self._write_frame(next_frame_tensor, child_depth, torch.zeros_like(current_frame_tensor), call_applied)
        next_frame_effect = self._write_frame(next_frame_effect, child_depth, torch.zeros_like(current_frame_effect), call_applied)
        next_frame_return = self._write_frame(
            state.frame_return_call, child_depth, safe_candidate, call_applied,
        )

        parent_depth = (current_depth - 1).clamp_min(0)
        return_call = self._frame_scalar(state.frame_return_call, current_depth)
        safe_return_call = return_call.clamp(0, self.candidate_count - 1)
        return_parent_slots = self.candidate_call_parent_outputs.index_select(0, safe_return_call)
        return_child_slots = self.candidate_call_child_outputs.index_select(0, safe_return_call)
        returned_handles = current_values.gather(1, return_child_slots.clamp_min(0))
        returned_producer = current_producer.gather(1, return_child_slots.clamp_min(0))
        returned_bank = current_bank.gather(1, return_child_slots.clamp_min(0))
        returned_revision = current_revision.gather(1, return_child_slots.clamp_min(0))
        returned_bank_handle = current_bank_handle.gather(1, return_child_slots.clamp_min(0))
        for output_index in range(self.spec.max_outputs):
            slot = return_parent_slots[:, output_index].clamp_min(0)
            row = child_stop & (return_parent_slots[:, output_index] >= 0) & (return_child_slots[:, output_index] >= 0)
            parent_frame = self._frame(next_values, parent_depth)
            parent_frame = self._write_slot(parent_frame, slot, returned_handles[:, output_index], row)
            next_values = self._write_depth(next_values, parent_depth, parent_frame, row)
            parent_frame = self._frame(next_producer, parent_depth)
            parent_frame = self._write_slot(parent_frame, slot, returned_producer[:, output_index], row)
            next_producer = self._write_depth(next_producer, parent_depth, parent_frame, row)
            response_frame = self._frame(next_response, parent_depth)
            response_frame = self._write_slot(response_frame, slot, safe_return_call, row)
            next_response = self._write_depth(next_response, parent_depth, response_frame, row)
            parent_frame = self._frame(next_producer_bank, parent_depth)
            parent_frame = self._write_slot(parent_frame, slot, returned_bank[:, output_index], row)
            next_producer_bank = self._write_depth(next_producer_bank, parent_depth, parent_frame, row)
            parent_frame = self._frame(next_producer_revision, parent_depth)
            parent_frame = self._write_slot(parent_frame, slot, returned_revision[:, output_index], row)
            next_producer_revision = self._write_depth(next_producer_revision, parent_depth, parent_frame, row)
            parent_frame = self._frame(next_producer_bank_handle, parent_depth)
            parent_frame = self._write_slot(parent_frame, slot, returned_bank_handle[:, output_index], row)
            next_producer_bank_handle = self._write_depth(next_producer_bank_handle, parent_depth, parent_frame, row)
        parent_steps = self._frame_scalar(next_frame_steps, parent_depth) + child_stop.to(torch.int64)
        next_frame_steps = self._write_frame(next_frame_steps, parent_depth, parent_steps, child_stop)
        parent_tensor = self._frame_scalar(next_frame_tensor, parent_depth) + child_stop.to(torch.int64)
        next_frame_tensor = self._write_frame(next_frame_tensor, parent_depth, parent_tensor, child_stop)
        clear = torch.full_like(current_values, -1)
        next_values = self._write_depth(next_values, current_depth, clear, child_stop)
        next_producer = self._write_depth(next_producer, current_depth, clear, child_stop)
        next_response = self._write_depth(next_response, current_depth, clear, child_stop)
        next_producer_bank = self._write_depth(next_producer_bank, current_depth, clear, child_stop)
        next_producer_revision = self._write_depth(next_producer_revision, current_depth, clear, child_stop)
        next_producer_bank_handle = self._write_depth(next_producer_bank_handle, current_depth, clear, child_stop)
        next_frame_query = self._write_frame(next_frame_query, current_depth, torch.full_like(current_query, -1), child_stop)
        next_frame_steps = self._write_frame(next_frame_steps, current_depth, torch.zeros_like(current_steps), child_stop)
        next_frame_tensor = self._write_frame(next_frame_tensor, current_depth, torch.zeros_like(current_tensor_steps), child_stop)
        next_frame_effect = self._write_frame(next_frame_effect, current_depth, torch.zeros_like(current_effect_steps), child_stop)
        next_frame_return = self._write_frame(next_frame_return, current_depth, torch.full_like(return_call, -1), child_stop)

        next_depth = torch.where(child_stop, parent_depth, torch.where(call_applied, child_depth, current_depth))
        next_completed = completed | root_stop
        next_active = active & ~root_stop
        next_state = FormulaDeviceFrameState(
            next_active, next_completed, next_depth, next_frame_query, next_frame_steps,
            next_frame_tensor, next_frame_effect, next_frame_return, next_values,
            next_producer, next_producer_bank, next_producer_revision,
            next_producer_bank_handle, next_bank_handles, next_bank_revisions, next_response,
        )
        rejected = in_range & active & ~applied
        event_kind = torch.where(
            applied,
            torch.where(root_stop, torch.full_like(selected_candidate, EVENT_STOP),
                        torch.where(child_stop, torch.full_like(selected_candidate, EVENT_RETURN),
                                    torch.where(call_applied, torch.full_like(selected_candidate, EVENT_CALL),
                                                torch.where(is_effect, torch.full_like(selected_candidate, EVENT_EFFECT),
                                                            torch.full_like(selected_candidate, EVENT_ORDINARY))))),
            torch.where(rejected, torch.full_like(selected_candidate, EVENT_REJECT),
                        torch.full_like(selected_candidate, EVENT_NONE)),
        )
        event_target_bank = torch.where(effect_applied, effect_bank, torch.full_like(effect_bank, -1))
        event_previous_revision = torch.where(effect_applied, live_bank_revision, torch.full_like(live_bank_revision, -1))
        event_successor_revision = torch.where(effect_applied, live_bank_revision + 1, torch.full_like(live_bank_revision, -1))
        event_returned = torch.where(child_stop[:, None], returned_handles, torch.full_like(returned_handles, -1))
        return next_state, FormulaDeviceFrameEvent(
            eligible, applied, event_kind, event_target_bank,
            event_previous_revision, event_successor_revision, event_returned,
        )


__all__ = [
    "EVENT_CALL", "EVENT_EFFECT", "EVENT_NONE", "EVENT_ORDINARY", "EVENT_REJECT",
    "EVENT_RETURN", "EVENT_STOP", "FormulaDeviceFrameEvent", "FormulaDeviceFrameKernel",
    "FormulaDeviceFrameSpec", "FormulaDeviceFrameState", "KIND_CALL", "KIND_EFFECT",
    "KIND_ORDINARY", "KIND_STOP",
]
