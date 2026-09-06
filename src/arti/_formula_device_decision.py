"""A prepared device wave for admission, Query scoring, and local K selection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple

import torch
from torch import Tensor, nn

from ._formula_device_admission import FormulaDeviceAdmission
from ._formula_device_frames import FormulaDeviceFrameKernel, FormulaDeviceFrameState
from ._formula_device_query import FormulaDeviceQuery
from ._formula_device_response import FormulaDeviceResponse, is_response_query
from .formula_program_call import FormulaProgramCallCandidateV1
from .formula_program_query_v5 import FormulaProgramQueryV5


class FormulaDeviceDecisionResult(NamedTuple):
    """All device-resident outputs needed by the global frontier."""

    eligible: Tensor
    scores: Tensor
    selected_candidates: Tensor
    input_finite: Tensor
    score_finite: Tensor
    coverage_satisfied: Tensor
    masked_logits: Tensor
    score_is_double: Tensor
    logits_finite: Tensor


class FormulaDeviceDecisionWave(nn.Module):
    """Fuse dynamic admission, native Query scoring, and local K selection.

    Homogeneous values omit the native singleton branch batch dimension.
    Typed pools retain it and gather real-shaped values before native summaries;
    only those fixed-width summaries are combined, never padded raw tensors.
    Global beam pruning, numerical dispatch, and frame advancement are the
    following execution regions and remain outside this module.
    """

    def __init__(
        self,
        queries: Sequence[FormulaProgramQueryV5],
        frame_kernel: FormulaDeviceFrameKernel,
        *,
        candidate_family_ids: Tensor | Sequence[int],
        width: int,
    ) -> None:
        super().__init__()
        if type(width) is not int or width < 1:
            raise ValueError("width must be a positive integer")
        self.admission = FormulaDeviceAdmission(queries, frame_kernel)
        self.frame_kernel = frame_kernel
        self.spec = frame_kernel.spec
        self.width = width
        self.data_layout = None
        self.bank_layout = None

        families = torch.as_tensor(candidate_family_ids, dtype=torch.int64, device="cpu")
        if families.shape != (frame_kernel.candidate_count,):
            raise ValueError("candidate_family_ids must cover every global action")
        self.register_buffer("candidate_family_ids", families, persistent=False)

        offsets: list[int] = []
        cursor = 0
        waves = []
        for query in queries:
            offsets.append(cursor)
            action_count = len(query.candidates) + 1
            local_families = families[cursor : cursor + action_count]
            if is_response_query(query):
                slot_indices = {slot: index for index, slot in enumerate(query.slot_ids)}
                producer_indices = {producer: index for index, producer in enumerate(query.candidate_ids)}
                wave = FormulaDeviceResponse.from_query(
                    query, width=min(width, action_count), candidate_family_ids=local_families,
                    preserve_family_coverage=True,
                )
                wave.register_buffer("response_slots", torch.tensor(
                    [slot_indices[slot] for _, slot in wave.score_sources], dtype=torch.int64, device="cpu",
                ), persistent=False)
                wave.register_buffer("response_producers", torch.tensor(
                    [cursor + producer_indices[producer] for producer, _ in wave.score_sources],
                    dtype=torch.int64, device="cpu",
                ), persistent=False)
            else:
                wave = FormulaDeviceQuery(
                    query.network,
                    slot_count=len(query.slot_ids),
                    candidate_family_ids=local_families,
                    width=min(width, action_count),
                    tensor_encoder=query.tensor_encoder,
                    action_priority=query._action_priority,
                    preserve_family_coverage=True,
                )
            waves.append(wave)
            cursor += action_count
        self.query_waves = nn.ModuleList(waves)
        self._query_offsets = tuple(offsets)
        self._query_widths = tuple(len(query.candidates) + 1 for query in queries)
        self._query_slot_counts = tuple(len(query.slot_ids) for query in queries)

    @classmethod
    def from_query(
        cls,
        query: FormulaProgramQueryV5,
        *,
        candidate_family_ids: Tensor | Sequence[int],
        width: int,
        frame_kernel: FormulaDeviceFrameKernel | None = None,
    ) -> "FormulaDeviceDecisionWave":
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
        return cls(
            queries,
            kernel,
            candidate_family_ids=candidate_family_ids,
            width=width,
        )

    @staticmethod
    def _frame(tensor: Tensor, depth: Tensor) -> Tensor:
        index = depth[:, None, None].expand(-1, 1, tensor.shape[-1])
        return tensor.gather(1, index).squeeze(1)

    @staticmethod
    def _frame_scalar(tensor: Tensor, depth: Tensor) -> Tensor:
        return tensor.gather(1, depth[:, None]).squeeze(1)

    @staticmethod
    def _pool_finite(pool: Tensor) -> Tensor:
        return torch.isfinite(pool.reshape(pool.shape[0], -1)).all(dim=-1)

    @staticmethod
    def _gather_pool(pool: Tensor, handles: Tensor) -> Tensor:
        value = pool.index_select(0, handles.clamp(0, pool.shape[0] - 1))
        valid = (handles >= 0) & (handles < pool.shape[0])
        mask = valid.reshape((handles.shape[0],) + (1,) * (value.ndim - 1))
        return torch.where(mask, value, torch.zeros_like(value))

    def _reference_finite(self, pools, handles, layout):
        flat = handles.reshape(-1)
        if not flat.numel():
            return torch.zeros_like(handles, dtype=torch.bool)
        if layout is None:
            value = self._gather_pool(pools[0], flat)
            finite = self._pool_finite(value) & (flat >= 0) & (flat < pools[0].shape[0])
        else:
            finite = torch.zeros_like(flat, dtype=torch.bool)
            for bucket in range(len(pools)):
                value = layout.gather(pools, flat, bucket)
                finite = finite | (layout.contains(flat, bucket) & self._pool_finite(value))
        return finite.reshape(handles.shape)

    def prepare_typed_pools_(self, dispatch) -> None:
        if dispatch.frame_kernel is not self.frame_kernel or dispatch.data_layout is None:
            raise ValueError("typed decision requires the matching prepared numerical dispatcher")
        self.data_layout = dispatch.data_layout
        self.bank_layout = dispatch.bank_layout
        self.admission.shape_admission = dispatch.typed_admission

    def _typed_summary(self, wave, pools, handles, occupied):
        rows, slots = handles.shape
        flat_handles, flat_occupied = handles.reshape(-1), occupied.reshape(-1)
        summary = wave._reference_tensor().new_zeros((rows * slots, wave.summary_width))
        found = torch.zeros_like(flat_occupied)
        finite = torch.ones_like(flat_occupied)
        for bucket, shape in enumerate(self.data_layout.shapes):
            here = flat_occupied & self.data_layout.contains(flat_handles, bucket)
            # A Query's native encoder need not accept shapes used by other Banks.
            # Reject only live references, not the existence of such global pools.
            if wave.tensor_encoder is not None and (
                len(shape) < 2 or shape[-1] != wave.tensor_encoder.input_dim
                or not pools[bucket].is_floating_point()
            ):
                finite = finite & ~here
                continue
            value = self.data_layout.gather(pools, flat_handles, bucket)
            part, part_finite = wave.summarize_value(value, here)
            summary = summary + part
            finite = finite & part_finite
            found = found | here
        finite = finite & (~flat_occupied | found)
        return summary.reshape(rows, slots * wave.summary_width), finite.reshape(rows, slots).all(-1)

    @torch.no_grad()
    def _response_values(self, wave, pools, handles, sources, rows_here, parent_scores):
        """Gather scalar responses only, with invocation-local producer identity."""
        selected = handles.index_select(1, wave.response_slots)
        present = (selected >= 0) & (sources.index_select(1, wave.response_slots) == wave.response_producers[None])
        present = present & rows_here[:, None]
        values = parent_scores.new_zeros(selected.shape)
        found = torch.zeros_like(present)
        source_double = torch.zeros_like(present)
        reference_double = torch.zeros_like(rows_here)
        first_slot = (handles >= 0).to(torch.int64).argmax(-1)
        reference = handles.gather(1, first_slot[:, None]).squeeze(1)
        flat = selected.reshape(-1)
        if self.data_layout is None:
            if pools[0].shape[1:] == (1,) and pools[0].is_floating_point() and torch.promote_types(pools[0].dtype, parent_scores.dtype) == parent_scores.dtype:
                values = self._gather_pool(pools[0], flat).reshape(selected.shape).to(parent_scores.dtype)
                found = (selected >= 0) & (selected < pools[0].shape[0])
                if pools[0].dtype == torch.float64:
                    source_double = found
                    reference_double = reference >= 0
        else:
            for bucket, shape in enumerate(self.data_layout.shapes):
                if pools[bucket].dtype == torch.float64:
                    reference_double |= self.data_layout.contains(reference, bucket)
                if shape not in ((1,), (1, 1)) or not pools[bucket].is_floating_point() or torch.promote_types(pools[bucket].dtype, parent_scores.dtype) != parent_scores.dtype:
                    continue
                here = self.data_layout.contains(flat, bucket).reshape(selected.shape)
                part = self.data_layout.gather(pools, flat, bucket).reshape(selected.shape).to(parent_scores.dtype)
                values = values + torch.where(here, part, torch.zeros_like(part))
                found = found | here
                if pools[bucket].dtype == torch.float64:
                    source_double |= here
        return values, present, (~present | found).all(dim=-1), reference_double, source_double

    @torch.no_grad()
    def forward(
        self,
        state: FormulaDeviceFrameState,
        data_pool: Tensor | tuple[Tensor, ...],
        bank_pool: Tensor | tuple[Tensor, ...],
        operand_finite: Tensor,
        parent_scores: Tensor,
        selection_noise: Tensor | None = None,
        input_sources=None,
    ) -> FormulaDeviceDecisionResult:
        """Run one local decision wave without materializing a host packet."""
        state = FormulaDeviceFrameState(*state)
        rows = state.active.shape[0]
        typed = self.data_layout is not None
        data_pools = tuple(data_pool) if typed else (data_pool,)
        bank_pools = tuple(bank_pool) if typed else (bank_pool,)
        if any(pool.ndim < 2 or pool.shape[0] < 1 for pool in (*data_pools, *bank_pools)):
            raise ValueError("pools must have shape [capacity, ...]")
        if parent_scores.shape != (rows,):
            raise ValueError("parent_scores must have shape [rows]")
        if any(pool.device != parent_scores.device for pool in (*data_pools, *bank_pools)):
            raise ValueError("state pools and scores must share a device")

        depth = state.depth.clamp(0, self.spec.max_depth - 1)
        current_query = self._frame_scalar(state.frame_query, depth)
        handles = self._frame(state.value_handles, depth)
        response_sources = self._frame(state.response_candidate, depth)
        steps = self._frame_scalar(state.frame_steps, depth)
        data_finite = self._reference_finite(data_pools, handles, self.data_layout)
        bank_finite = self._reference_finite(bank_pools, state.bank_value_handles, self.bank_layout)
        eligible = self.admission(
            tuple(state), data_finite, bank_finite, operand_finite, reference_flags=True,
            input_sources=input_sources,
        )

        scores = parent_scores.new_full(
            (rows, self.frame_kernel.candidate_count), -float("inf")
        )
        masked_logits = torch.full_like(scores, -float("inf"))
        score_is_double = torch.zeros(rows, dtype=torch.bool, device=parent_scores.device)
        selected = parent_scores.new_full(
            (rows, self.width), -1, dtype=torch.int64,
        )
        input_finite = torch.ones(rows, dtype=torch.bool, device=parent_scores.device)
        score_finite = torch.ones_like(input_finite)
        logits_finite = torch.ones_like(input_finite)
        coverage = torch.ones_like(input_finite)

        for query_id, wave in enumerate(self.query_waves):
            offset = self._query_offsets[query_id]
            action_count = self._query_widths[query_id]
            slot_count = self._query_slot_counts[query_id]
            rows_here = state.active & ~state.completed & (current_query == query_id)
            local_eligible = eligible[:, offset : offset + action_count] & rows_here[:, None]
            occupied = (handles[:, :slot_count] >= 0) & rows_here[:, None]
            noise = None if selection_noise is None else selection_noise[:, offset : offset + action_count]
            if isinstance(wave, FormulaDeviceResponse):
                response_values, present, response_valid, reference_double, source_double = self._response_values(
                    wave, data_pools, handles, response_sources, rows_here, parent_scores,
                )
                result = wave(response_values, present, local_eligible, parent_scores, steps, noise,
                              reference_double, source_double)
                local_double = reference_double | (present & source_double).any(-1)
                response_valid = response_valid | ~result.eligible.any(dim=-1)
                result = result._replace(
                    input_finite=response_valid,
                    finite=result.finite & response_valid,
                    local_selected=result.local_selected & response_valid[:, None],
                )
            elif typed:
                summary, finite = self._typed_summary(wave, data_pools, handles[:, :slot_count], occupied)
                result = wave.score_summary(summary, finite, local_eligible, parent_scores, noise)
                local_double = torch.full_like(rows_here, result.masked_logits.dtype == torch.float64)
            else:
                values = tuple(self._gather_pool(data_pool, handles[:, slot]) for slot in range(slot_count))
                result = wave(values, occupied, local_eligible, parent_scores, noise)
                local_double = torch.full_like(rows_here, result.masked_logits.dtype == torch.float64)
            masked_logits[:, offset : offset + action_count] = torch.where(
                rows_here[:, None], result.masked_logits, masked_logits[:, offset : offset + action_count]
            )
            score_is_double = torch.where(rows_here, local_double, score_is_double)
            scores[:, offset : offset + action_count] = torch.where(
                rows_here[:, None], result.scores, scores[:, offset : offset + action_count]
            )
            eligible[:, offset : offset + action_count] = result.eligible
            local_width = min(self.width, action_count)
            local_order = result.selection_order[:, :local_width]
            local_valid = result.local_selected.gather(1, local_order) & rows_here[:, None]
            global_order = local_order + offset
            selected[:, :local_width] = torch.where(
                local_valid, global_order, selected[:, :local_width]
            )
            input_finite = torch.where(rows_here, result.input_finite, input_finite)
            score_finite = torch.where(rows_here, result.score_finite, score_finite)
            logits_finite = torch.where(rows_here, result.logits_finite, logits_finite)
            coverage = torch.where(rows_here, result.coverage_satisfied, coverage)

        return FormulaDeviceDecisionResult(
            eligible,
            scores,
            selected,
            input_finite,
            score_finite,
            coverage,
            masked_logits,
            score_is_double,
            logits_finite,
        )


__all__ = ["FormulaDeviceDecisionResult", "FormulaDeviceDecisionWave"]
