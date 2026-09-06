"""Device-resident admission for a prepared Formula Query graph.

The native Query remains the semantic reference.  This module lowers its
runtime-varying admission state to tensors after shape, dtype, Formula, and
extension compatibility have been checked by the host preparation boundary.
It deliberately keeps the complete candidate universe, including STOP and
nested CALL candidates.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import Tensor, nn

from ._formula_device_frames import (
    KIND_CALL,
    KIND_EFFECT,
    KIND_ORDINARY,
    KIND_STOP,
    FormulaDeviceFrameKernel,
    FormulaDeviceFrameState,
)
from .formula_program_call import FormulaProgramCallCandidateV1
from .formula_program_query_v5 import FormulaProgramQueryV5
from ._formula_device_response import is_response_query
from ._formula_device_sources import FormulaDeviceSources


class FormulaDeviceAdmission(nn.Module):
    """Evaluate every prepared action without host reads or candidate loops.

    ``data_finite`` and ``bank_finite`` describe the caller-owned immutable
    value pools addressed by frame handles.  ``operand_finite`` is refreshed
    once per event and summarizes non-SSA, non-current-Bank operands for every
    global action.  Shape and Formula compatibility are preparation-time
    contracts; this kernel handles only dynamic execution admission.
    """

    def __init__(
        self,
        queries: Sequence[FormulaProgramQueryV5],
        frame_kernel: FormulaDeviceFrameKernel,
    ) -> None:
        super().__init__()
        normalized = tuple(queries)
        if not normalized or any(type(query) is not FormulaProgramQueryV5 and not is_response_query(query)
                                 for query in normalized):
            raise TypeError("queries must contain native FormulaProgramQueryV5 or V6 instances")
        if len(normalized) != frame_kernel.spec.query_count:
            raise ValueError("queries do not match the prepared frame graph")

        expected: list[FormulaProgramQueryV5] = []
        visited: set[int] = set()

        def visit(query: FormulaProgramQueryV5) -> None:
            if id(query) in visited:
                return
            visited.add(id(query))
            expected.append(query)
            for candidate in query.candidates:
                if isinstance(candidate, FormulaProgramCallCandidateV1):
                    visit(candidate.child)

        visit(normalized[0])
        if tuple(expected) != normalized:
            raise ValueError("queries must use the frame graph's root-first DFS order")
        self.queries = normalized
        self.frame_kernel = frame_kernel
        self.spec = frame_kernel.spec
        self.shape_admission = None

        offsets: list[int] = []
        cursor = 0
        for query_id, query in enumerate(normalized):
            offsets.append(cursor)
            expected_ids = tuple(candidate.candidate_id for candidate in query.candidates) + ("stop",)
            actual = self.spec.candidate_ids[cursor : cursor + len(expected_ids)]
            if actual != expected_ids or not torch.equal(
                self.spec.candidate_query[cursor : cursor + len(expected_ids)],
                torch.full((len(expected_ids),), query_id, dtype=torch.int64),
            ):
                raise ValueError("queries do not match frame candidate ordering")
            cursor += len(expected_ids)
        if cursor != frame_kernel.candidate_count:
            raise ValueError("frame candidate table has unexpected trailing actions")
        self._query_offsets = tuple(offsets)
        self._query_widths = tuple(len(query.candidates) + 1 for query in normalized)
        self._response_queries = tuple(is_response_query(query) for query in normalized)
        for index, (query, offset) in enumerate(zip(normalized, offsets, strict=True)):
            if not self._response_queries[index]:
                continue
            slots = {slot: i for i, slot in enumerate(query.slot_ids)}
            producers = {name: i for i, name in enumerate(query.candidate_ids)}
            edges = tuple((slots[slot], offset + producers[producer], column)
                          for producer, slot, column in query._response_columns)
            for column, name in enumerate(("slots", "producers", "destinations")):
                self.register_buffer(f"response_{name}_{index}", torch.tensor(
                    [edge[column] for edge in edges], dtype=torch.int64, device="cpu"), persistent=False)
            self.register_buffer(f"response_entries_{index}", torch.tensor(
                [action in query.entry_candidates for action in query.action_ids],
                dtype=torch.bool, device="cpu"), persistent=False)
        self._call_maps = {
            global_index: (
                int(self.spec.candidate_child_query[global_index]),
                tuple(
                    (int(parent), int(child))
                    for parent, child in zip(
                        self.spec.candidate_call_parent_inputs[global_index],
                        self.spec.candidate_call_child_inputs[global_index],
                        strict=True,
                    )
                    if int(parent) >= 0 and int(child) >= 0
                ),
            )
            for global_index, kind in enumerate(self.spec.candidate_kind)
            if int(kind) == KIND_CALL
        }
        self._terminal_counts = tuple(int(value) for value in self.spec.query_terminal_count)

        for name in (
            "candidate_query",
            "candidate_kind",
            "candidate_required",
            "candidate_empty",
            "candidate_output_bank",
            "candidate_effect_input",
            "candidate_child_query",
            "candidate_call_parent_inputs",
            "candidate_call_child_inputs",
            "query_terminal_count",
            "query_terminal_slots",
            "query_min_steps",
            "query_max_steps",
            "query_min_tensor_steps",
            "query_max_tensor_steps",
            "query_max_effect_steps",
        ):
            self.register_buffer(name, getattr(self.spec, name), persistent=False)

    @classmethod
    def from_query(
        cls,
        query: FormulaProgramQueryV5,
        frame_kernel: FormulaDeviceFrameKernel | None = None,
    ) -> "FormulaDeviceAdmission":
        kernel = FormulaDeviceFrameKernel.from_query(query) if frame_kernel is None else frame_kernel
        queries: list[FormulaProgramQueryV5] = []
        visited: set[int] = set()

        def visit(current: FormulaProgramQueryV5) -> None:
            if id(current) in visited:
                return
            visited.add(id(current))
            queries.append(current)
            for candidate in current.candidates:
                if isinstance(candidate, FormulaProgramCallCandidateV1):
                    visit(candidate.child)

        visit(query)
        return cls(queries, kernel)

    @staticmethod
    def _gather_frame(tensor: Tensor, depth: Tensor) -> Tensor:
        index = depth[:, None, None].expand(-1, 1, tensor.shape[-1])
        return tensor.gather(1, index).squeeze(1)

    @staticmethod
    def _gather_counter(tensor: Tensor, depth: Tensor) -> Tensor:
        return tensor.gather(1, depth[:, None]).squeeze(1)

    @staticmethod
    def _pool_flags(handles: Tensor, flags: Tensor) -> Tensor:
        """Gather pool flags while treating the -1 handle as non-finite."""
        if flags.ndim != 2 or flags.shape[0] != handles.shape[0]:
            raise ValueError("pool finite flags must have shape [rows, capacity]")
        sentinel = torch.zeros((flags.shape[0], 1), dtype=torch.bool, device=flags.device)
        padded = torch.cat((flags, sentinel), dim=1)
        safe = torch.where(handles >= 0, handles, handles.new_full((), flags.shape[1]))
        safe = safe.clamp(0, flags.shape[1])
        return padded.gather(1, safe)

    def _candidate_operand_flags(self, operand_finite: Tensor, rows: int) -> Tensor:
        if operand_finite.ndim == 1:
            if operand_finite.shape[0] != self.frame_kernel.candidate_count:
                raise ValueError("operand_finite must cover every global action")
            return operand_finite[None].expand(rows, -1)
        if operand_finite.shape != (rows, self.frame_kernel.candidate_count):
            raise ValueError("operand_finite must have shape [actions] or [rows, actions]")
        return operand_finite

    def _local_mask(
        self,
        query_id: int,
        values: Tensor,
        response_candidate: Tensor,
        producer_bank: Tensor,
        producer_revision: Tensor,
        producer_bank_handle: Tensor,
        steps: Tensor,
        tensor_steps: Tensor,
        effect_steps: Tensor,
        bank_handles: Tensor,
        bank_revisions: Tensor,
        data_finite: Tensor,
        bank_finite: Tensor,
        operands: Tensor,
        input_sources=None,
    ) -> Tensor:
        rows = values.shape[0]
        offset = self._query_offsets[query_id]
        width = self._query_widths[query_id]
        indices = torch.arange(offset, offset + width, device=values.device)
        kinds = self.candidate_kind.index_select(0, indices)
        required = self.candidate_required.index_select(0, indices)
        empty = self.candidate_empty.index_select(0, indices)
        occupied = values >= 0
        slot_finite = data_finite

        structural = (~required[None] | occupied[:, None, :]).all(dim=-1)
        structural &= ~(empty[None] & occupied[:, None, :]).any(dim=-1)
        finite_inputs = (~required[None] | slot_finite[:, None, :]).all(dim=-1)
        selected_sources = None
        if input_sources is not None:
            selected_sources = FormulaDeviceSources(*(field.index_select(1, indices) for field in input_sources))
            structural = selected_sources.ready.all(-1)
            structural &= ~(empty[None] & occupied[:, None, :]).any(-1)
            finite_inputs = selected_sources.finite.all(-1)
        allowed = structural & finite_inputs

        maximum = self.query_max_steps[query_id]
        allowed &= steps[:, None] < maximum
        is_effect = kinds == KIND_EFFECT
        is_stop = kinds == KIND_STOP
        maximum_tensor = self.query_max_tensor_steps[query_id]
        maximum_effect = self.query_max_effect_steps[query_id]
        tensor_budget = (maximum_tensor < 0) | (tensor_steps[:, None] < maximum_tensor)
        effect_budget = (maximum_effect < 0) | (effect_steps[:, None] < maximum_effect)
        allowed &= torch.where(is_effect[None], effect_budget, tensor_budget)
        allowed &= operands.index_select(1, indices)

        # An ordinary occurrence's mutable Bank is represented by the Bank id
        # attached to its owned output port. Other immutable operands are
        # already summarized by operand_finite.
        output_banks = self.candidate_output_bank.index_select(0, indices)
        ordinary_bank = output_banks.amax(dim=-1)
        has_bank = ordinary_bank >= 0
        if bank_handles.shape[1]:
            safe_bank = ordinary_bank.clamp(0, bank_handles.shape[1] - 1)
            current_finite = bank_finite.index_select(1, safe_bank)
            allowed &= ~(
                (kinds == KIND_ORDINARY) & has_bank
            )[None] | current_finite
        else:
            allowed &= ~((kinds == KIND_ORDINARY) & has_bank)[None]

        # Effects can target only the current revision/value handle of the
        # ordinary producer that supplied their data input.
        effect_slot = self.candidate_effect_input.index_select(0, indices)
        safe_slot = effect_slot.clamp(0, values.shape[1] - 1)
        lineage_bank = producer_bank.gather(1, safe_slot[None].expand(rows, -1))
        lineage_revision = producer_revision.gather(1, safe_slot[None].expand(rows, -1))
        lineage_handle = producer_bank_handle.gather(1, safe_slot[None].expand(rows, -1))
        if selected_sources is not None:
            ports = self.frame_kernel.candidate_effect_port.index_select(0, indices).clamp_min(0)
            port_index = ports[None, :, None].expand(rows, -1, 1)
            lineage_bank = selected_sources.bank.gather(2, port_index).squeeze(-1)
            lineage_revision = selected_sources.revision.gather(2, port_index).squeeze(-1)
            lineage_handle = selected_sources.bank_handle.gather(2, port_index).squeeze(-1)
        valid_lineage = (effect_slot >= 0)[None] & (lineage_bank >= 0) & (lineage_handle >= 0)
        if bank_handles.shape[1]:
            safe_lineage_bank = lineage_bank.clamp(0, bank_handles.shape[1] - 1)
            current_handle = bank_handles.gather(1, safe_lineage_bank)
            current_revision = bank_revisions.gather(1, safe_lineage_bank)
            valid_lineage &= (lineage_bank < bank_handles.shape[1])
            valid_lineage &= current_handle == lineage_handle
            valid_lineage &= current_revision == lineage_revision
            valid_lineage &= bank_finite.gather(1, safe_lineage_bank)
        else:
            valid_lineage &= False
        allowed &= ~is_effect[None] | valid_lineage
        if self.shape_admission is not None:
            if input_sources is None:
                shapes = self.shape_admission(values, producer_bank_handle, bank_handles)
            else:
                shapes = self.shape_admission(values, producer_bank_handle, bank_handles, input_sources)
            allowed &= shapes.index_select(1, indices)

        # A CALL is legal only when its mapped child entry has at least one
        # legal action (including child STOP). The static Python loop is traced
        # once; no runtime host decision depends on its Tensor result.
        for local_index, candidate in enumerate(self.queries[query_id].candidates):
            if not isinstance(candidate, FormulaProgramCallCandidateV1):
                continue
            global_index = offset + local_index
            child_id, port_map = self._call_maps[global_index]
            child_values = values.new_full((rows, self.spec.max_slots), -1)
            child_bank = producer_bank.new_full((rows, self.spec.max_slots), -1)
            child_revision = producer_revision.new_full((rows, self.spec.max_slots), -1)
            child_bank_handle = producer_bank_handle.new_full((rows, self.spec.max_slots), -1)
            child_finite = data_finite.new_zeros((rows, self.spec.max_slots))
            for port, (parent_slot, child_slot) in enumerate(port_map):
                child_values[:, child_slot] = (values[:, parent_slot] if selected_sources is None else
                                              selected_sources.handles[:, local_index, port])
                child_bank[:, child_slot] = (producer_bank[:, parent_slot] if selected_sources is None else
                                            selected_sources.bank[:, local_index, port])
                child_revision[:, child_slot] = (producer_revision[:, parent_slot] if selected_sources is None else
                                                selected_sources.revision[:, local_index, port])
                child_bank_handle[:, child_slot] = (producer_bank_handle[:, parent_slot] if selected_sources is None else
                                                   selected_sources.bank_handle[:, local_index, port])
                port_finite = (data_finite[:, parent_slot:parent_slot + 1] if selected_sources is None else
                               selected_sources.finite[:, local_index, port:port + 1])
                child_finite = child_finite.index_copy(
                    1, values.new_full((1,), child_slot), port_finite,
                )
            child_mask = self._local_mask(
                child_id,
                child_values,
                torch.full_like(child_values, -1),
                child_bank,
                child_revision,
                child_bank_handle,
                torch.zeros_like(steps),
                torch.zeros_like(tensor_steps),
                torch.zeros_like(effect_steps),
                bank_handles,
                bank_revisions,
                child_finite,
                bank_finite,
                operands,
            )
            allowed[:, local_index] &= child_mask.any(dim=1)

        terminal_count = self._terminal_counts[query_id]
        terminals = self.query_terminal_slots[query_id, :terminal_count]
        terminal_ready = occupied.index_select(1, terminals).all(dim=1)
        stop = (
            terminal_ready
            & (steps >= self.query_min_steps[query_id])
            & (tensor_steps >= self.query_min_tensor_steps[query_id])
        )
        allowed = torch.where(is_stop[None], stop[:, None], allowed)
        if self._response_queries[query_id]:
            slots = getattr(self, f"response_slots_{query_id}")
            producers = getattr(self, f"response_producers_{query_id}")
            destinations = getattr(self, f"response_destinations_{query_id}")
            here = (values.index_select(1, slots) >= 0) & (response_candidate.index_select(1, slots) == producers[None])
            # Integer OR-equivalent reduction, with no per-edge host dispatch.
            counts = torch.zeros_like(allowed, dtype=torch.int64).scatter_add(
                1, destinations[None].expand(values.shape[0], -1), here.to(torch.int64))
            entries = getattr(self, f"response_entries_{query_id}")
            available = (counts > 0) | ((steps == 0)[:, None] & entries[None])
            available[:, -1] = True
            allowed = allowed & available
        return allowed

    def forward(
        self,
        state: FormulaDeviceFrameState,
        data_finite: Tensor,
        bank_finite: Tensor,
        operand_finite: Tensor,
        *,
        reference_flags: bool = False,
        input_sources=None,
    ) -> Tensor:
        """Return the complete global ``[rows, actions]`` admission mask.

        ``reference_flags`` selects prepared per-frame-slot/per-Bank flags;
        the default pool-indexed interface remains available as a reference.
        This choice is static at export/capture time.
        """
        state = FormulaDeviceFrameState(*state)
        if input_sources is not None:
            input_sources = FormulaDeviceSources(*input_sources)
        rows = state.active.shape[0]
        operands = self._candidate_operand_flags(operand_finite, rows)
        depth = state.depth.clamp(0, self.spec.max_depth - 1)
        query = self._gather_counter(state.frame_query, depth)
        values = self._gather_frame(state.value_handles, depth)
        response_candidate = self._gather_frame(state.response_candidate, depth)
        producer_bank = self._gather_frame(state.producer_bank, depth)
        producer_revision = self._gather_frame(state.producer_revision, depth)
        producer_bank_handle = self._gather_frame(state.producer_bank_handle, depth)
        steps = self._gather_counter(state.frame_steps, depth)
        tensor_steps = self._gather_counter(state.frame_tensor_steps, depth)
        effect_steps = self._gather_counter(state.frame_effect_steps, depth)
        if not reference_flags:
            data_finite = self._pool_flags(values, data_finite)
            bank_finite = self._pool_flags(state.bank_value_handles, bank_finite)

        result = values.new_zeros(
            (rows, self.frame_kernel.candidate_count),
            dtype=torch.bool,
        )
        for query_id in range(len(self.queries)):
            offset = self._query_offsets[query_id]
            width = self._query_widths[query_id]
            local = self._local_mask(
                query_id,
                values,
                response_candidate,
                producer_bank,
                producer_revision,
                producer_bank_handle,
                steps,
                tensor_steps,
                effect_steps,
                state.bank_value_handles,
                state.bank_revisions,
                data_finite,
                bank_finite,
                operands,
                input_sources,
            )
            rows_here = state.active & ~state.completed & (query == query_id)
            result[:, offset : offset + width] = local & rows_here[:, None]
        return result


__all__ = ["FormulaDeviceAdmission"]
