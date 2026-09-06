"""Same-snapshot cooperative frontiers over existing Formula device execution.

This is a prepared round, not the complete cooperative search/tape runtime.
Candidate bindings are finite prepared alternatives. W expansion heads and C
cooperating ordinary nodes are independent axes; neither is an episode axis.
"""

from typing import NamedTuple

import torch
from torch import Tensor, nn

from ._formula_device_decision import FormulaDeviceDecisionWave
from ._formula_device_dispatch import FormulaDeviceDispatchLayout, _query_graph, formula_device_dispatch_groups
from ._formula_device_frames import FormulaDeviceFrameState, KIND_ORDINARY
from ._formula_device_query import FormulaDeviceQuery
from ._formula_device_sources import FormulaDeviceSources, completed_sources, select_sources


class FormulaDeviceFrontiers(NamedTuple):
    candidates: Tensor
    valid: Tensor
    selection_log_score: Tensor
    remaining: Tensor


def cooperative_sequence_score(scores, candidates, remaining, score_is_double=None):
    """Score a frozen [row, head, sibling] sequence with its actual denominators.

    Shared by hard selection and differentiable replay. No argmax, admission or
    input rebinding happens here. Preserve the native per-row accumulation dtype.
    """
    logits = scores[:, None].expand(-1, candidates.shape[1], -1)
    total = torch.zeros_like(logits[:, :, 0])
    for column in range(candidates.shape[-1]):
        mask = remaining[:, :, column]
        available = logits.masked_fill(~mask, -torch.inf)
        safe = torch.where(mask.any(-1, keepdim=True), available, torch.zeros_like(available))
        normalizer = torch.logsumexp(safe, -1)
        if scores.dtype == torch.float64 and score_is_double is not None:
            # Unused FP32 reductions must not overflow and poison the FP64 VJP.
            low_input = torch.where(score_is_double[:, None, None], torch.zeros_like(safe), safe)
            normalizer = torch.where(score_is_double[:, None], normalizer,
                                     torch.logsumexp(low_input.float(), -1).double())
        choice = candidates[:, :, column]
        score = logits.gather(2, choice.clamp_min(0)[:, :, None]).squeeze(-1) - normalizer
        if scores.dtype == torch.float64 and score_is_double is not None:
            score = torch.where(score_is_double[:, None], score, score.float().double())
        total = total + torch.where(choice >= 0, score, 0)
        if scores.dtype == torch.float64 and score_is_double is not None:
            total = torch.where(score_is_double[:, None], total, total.float().double())
    return total


class FormulaDeviceCooperativeResult(NamedTuple):
    frames: FormulaDeviceFrameState
    frontiers: FormulaDeviceFrontiers
    accepted: Tensor
    products: FormulaDeviceSources
    next_occurrence: Tensor
    data_cursor: Tensor
    bank_cursor: Tensor
    requires_fallback: Tensor
    output_handles: Tensor
    score_is_double: Tensor
    eligible: Tensor


class FormulaDeviceCooperativeSelection(nn.Module):
    """Lower native V7 head expansion, sibling selection and set deduplication."""

    def __init__(self, query, kernel, *, head_width):
        super().__init__()
        if type(head_width) is not int or head_width < 1:
            raise ValueError("head_width must be a positive integer")
        owners = _query_graph(query)
        self.head_width = min(head_width, kernel.candidate_count)
        widths = [getattr(owner, "cooperation_width", 1) for owner in owners]
        self.cooperation_width = max(widths)
        spec = kernel.spec
        slots = spec.candidate_output_slots
        output = torch.zeros((kernel.candidate_count, spec.max_slots + 1), dtype=torch.bool, device=slots.device)
        output.scatter_(1, torch.where(slots >= 0, slots, spec.max_slots), True)
        touched = output[:, :spec.max_slots] | spec.candidate_empty
        conflict = torch.zeros((kernel.candidate_count, kernel.candidate_count), dtype=torch.bool, device=slots.device)
        # Inspect actual writes, never materialize [candidate, candidate, slot].
        for column in slots.unbind(1):
            conflict |= touched.index_select(1, column.clamp_min(0)).T & (column >= 0)[:, None]
        conflict = conflict | conflict.T
        conflict |= torch.eye(kernel.candidate_count, dtype=torch.bool, device=conflict.device)
        tables = {
            "conflict": conflict,
            "kinds": spec.candidate_kind,
            "query_widths": torch.tensor(widths, dtype=torch.int64, device="cpu"),
            "max_steps": spec.query_max_steps,
            "max_tensor_steps": spec.query_max_tensor_steps,
            "tie_order": torch.tensor(sorted(range(kernel.candidate_count),
                                             key=lambda i: spec.candidate_ids[i]),
                                      dtype=torch.int64, device="cpu"),
        }
        for name, value in tables.items():
            self.register_buffer(name, value, persistent=False)

    def forward(self, scores, eligible, query_ids, steps, tensor_steps, score_is_double=None,
                selection_bias=None):
        rows, actions = scores.shape
        width, columns = self.head_width, self.cooperation_width
        # Exploration changes support selection, never the model's recorded energy.
        ordering = scores if selection_bias is None else scores + selection_bias
        ranked = ordering.masked_fill(~eligible, -torch.inf).index_select(1, self.tie_order)
        ranks = torch.argsort(ranked, dim=-1, descending=True, stable=True)[:, :width]
        heads = self.tie_order[ranks]
        valid = eligible.gather(1, heads)
        remaining = eligible[:, None].expand(-1, width, -1).clone()
        logits = ordering[:, None].expand_as(remaining)
        running = valid.clone()
        selected, denominators = [], []
        native_width = self.query_widths[query_ids][:, None]
        maximum = self.max_steps[query_ids][:, None]
        maximum_tensor = self.max_tensor_steps[query_ids][:, None]
        for column in range(columns):
            available = logits.masked_fill(~remaining, -torch.inf)
            if column == 0:
                choice = heads
            else:
                order = available.index_select(-1, self.tie_order)
                choice = self.tie_order[order.argmax(-1)]
            kind = self.kinds[choice]
            take = running & remaining.any(-1)
            if column:
                take &= kind == KIND_ORDINARY
            selected.append(torch.where(take, choice, -1))
            denominators.append(remaining)
            remaining = remaining & ~self.conflict[choice]
            count = column + 1
            running = take & (kind == KIND_ORDINARY) & (count < native_width)
            running &= steps[:, None] + count < maximum
            running &= (maximum_tensor < 0) | (tensor_steps[:, None] + count < maximum_tensor)
        candidates = torch.stack(selected, dim=-1)
        # Preserve the first ranked head's order/score for an equivalent set.
        canonical = torch.sort(torch.where(candidates >= 0, candidates, actions), dim=-1).values
        equal = (canonical[:, :, None] == canonical[:, None, :]).all(-1)
        earlier = torch.arange(width, device=scores.device)[None, :] < torch.arange(width, device=scores.device)[:, None]
        duplicate = (equal & earlier[None] & valid[:, None, :]).any(-1)
        valid &= ~duplicate
        candidates = torch.where(valid[:, :, None], candidates, -1)
        remaining = torch.stack(denominators, dim=2)
        total_score = cooperative_sequence_score(scores, candidates, remaining, score_is_double)
        return FormulaDeviceFrontiers(
            candidates, valid, torch.where(valid, total_score, 0), remaining,
        )


class FormulaDeviceCooperativeWave(nn.Module):
    """Select, execute and atomically publish one frontier per retained head.

    Ordinary siblings are evaluated against the same frame and Bank snapshot.
    A failed sibling invalidates the whole frontier's publication. The pool may
    still contain attempted values; allocation work is not retroactively free.
    CALL enters a child frame and is not published as a completed child here.
    """

    def __init__(self, query, execution, *, head_width=1, native_child=False):
        super().__init__()
        self.execution = execution
        self.kernel = execution.frame_kernel
        self.selection = FormulaDeviceCooperativeSelection(query, self.kernel, head_width=head_width)
        self.decision = FormulaDeviceDecisionWave.from_query(
            query, frame_kernel=self.kernel, width=1,
            candidate_family_ids=[0] * self.kernel.candidate_count,
        )
        for local in self.decision.query_waves:
            local.preserve_family_coverage = False
        if execution.dispatch.data_layout is not None:
            self.decision.prepare_typed_pools_(execution.dispatch)
        self.layout = FormulaDeviceDispatchLayout(formula_device_dispatch_groups(query)[0])
        self.register_buffer("hard_child_queries", torch.tensor(
            [native_child and not hasattr(owner, "cooperation_width") for owner in _query_graph(query)],
            dtype=torch.bool, device="cpu",
        ), persistent=False)

    @torch.no_grad()
    def forward(self, state, data_pool, bank_pool, data_cursor, bank_cursor,
                operand_finite, occurrence_cursor, input_sources=None, selection_bias=None):
        state = FormulaDeviceFrameState(*state)
        rows = state.active.shape[0]
        width, columns = self.selection.head_width, self.selection.cooperation_width
        typed = self.execution.dispatch.data_layout is not None
        data = data_pool if typed else data_pool.squeeze(1)
        prototype = data_pool[0] if typed else data_pool
        pools = data_pool if typed else (data_pool,)
        # Storage must preserve legacy Query network precision independently of
        # payload dtype; response-query arithmetic retains its per-row dtype.
        double_query = any(isinstance(local, FormulaDeviceQuery) and
                           local._reference_tensor().dtype == torch.float64
                           for local in self.decision.query_waves)
        score_dtype = torch.float64 if double_query or any(pool.dtype == torch.float64 for pool in pools) else torch.float32
        decision = self.decision(state, data, bank_pool, operand_finite,
                                 prototype.new_zeros(rows, dtype=score_dtype), input_sources=input_sources)
        depth = state.depth.clamp(0, self.kernel.spec.max_depth - 1)
        current_query = self.kernel._frame_scalar(state.frame_query, depth)
        hard_child = self.hard_child_queries[current_query]
        frontiers = self.selection(
            decision.masked_logits, decision.eligible,
            current_query,
            self.kernel._frame_scalar(state.frame_steps, depth),
            self.kernel._frame_scalar(state.frame_tensor_steps, depth),
            decision.score_is_double,
            selection_bias,
        )
        # V5/V6 native CALL execution ranks finite raw logits. Its child
        # decisions do not contribute a normalized root beam score.
        frontiers = frontiers._replace(selection_log_score=torch.where(
            hard_child[:, None], 0, frontiers.selection_log_score,
        ))
        candidates = frontiers.candidates
        flat = candidates.reshape(rows, width * columns)
        packet = self.layout(flat)
        sources = None if input_sources is None else select_sources(
            input_sources, packet.source_rows, packet.candidate_ids.clamp_min(0),
        )
        result = self.execution(state, packet, data_pool, bank_pool, data_cursor, bank_cursor,
                                input_sources=sources)
        accepted_nodes = result.event.accepted.index_select(0, packet.inverse_order).reshape(rows, width, columns)
        accepted = frontiers.valid & ((candidates < 0) | accepted_nodes).all(-1)
        numerical_ok = decision.input_finite & torch.where(
            hard_child, decision.logits_finite, decision.score_finite,
        )
        accepted &= numerical_ok[:, None]
        count = rows * width
        parent = torch.arange(rows, device=prototype.device).repeat_interleave(width)
        base = self.execution._fork_state(state, parent)
        child_order = packet.inverse_order.reshape(count, columns)
        first_state = self.execution._fork_state(result.state, child_order[:, 0])
        ordinary = self.selection.kinds[candidates[:, :, 0].clamp_min(0)].eq(KIND_ORDINARY).reshape(-1)
        next_fields = [torch.where(ordinary.reshape((count,) + (1,) * (old.ndim - 1)), old, first)
                       for old, first in zip(base, first_state, strict=True)]
        merged = FormulaDeviceFrameState(*next_fields)
        base_depth = base.depth.clamp(0, self.kernel.spec.max_depth - 1)
        value_fields = ("value_handles", "producer_candidate", "producer_bank",
                        "producer_revision", "producer_bank_handle", "response_candidate")
        for column in range(columns):
            action = candidates.reshape(count, columns)[:, column]
            chosen = self.execution._fork_state(result.state, child_order[:, column])
            slots = self.kernel.candidate_output_slots[action.clamp_min(0)]
            owns = (slots[:, :, None] == torch.arange(self.kernel.spec.max_slots,
                                                     device=prototype.device)[None, None]).any(1)
            write = ordinary & (action >= 0)
            replacements = {}
            for name in value_fields:
                current = getattr(merged, name)
                view = self.kernel._frame(current, base_depth)
                selected = self.kernel._frame(getattr(chosen, name), base_depth)
                replacements[name] = self.kernel._write_depth(current, base_depth,
                                                               torch.where(owns, selected, view), write)
            merged = merged._replace(**replacements)
        increments = (candidates >= 0).sum(-1).reshape(-1)
        for name in ("frame_steps", "frame_tensor_steps"):
            current = getattr(merged, name)
            value = self.kernel._frame_scalar(current, base_depth) + increments
            merged = merged._replace(**{name: self.kernel._write_frame(current, base_depth, value, ordinary)})
        keep = accepted.reshape(-1)
        merged = FormulaDeviceFrameState(*(torch.where(keep.reshape((count,) + (1,) * (new.ndim - 1)), new, old)
                                          for old, new in zip(base, merged, strict=True)))
        merged = merged._replace(active=merged.active & keep, completed=merged.completed & keep)

        executed = (candidates >= 0) & accepted[:, :, None]
        ids = occurrence_cursor + executed.reshape(-1).to(torch.int64).cumsum(0) - 1
        ids = torch.where(executed.reshape(-1), ids, -1)
        packet_order = packet.source_rows * (width * columns) + packet.source_lanes
        products = completed_sources(self.kernel, result, packet, ids[packet_order])
        products = FormulaDeviceSources(*(field.reshape(-1, self.kernel.spec.max_outputs)
                                          .index_select(0, packet.inverse_order).reshape(-1)
                                          for field in products))
        publish = executed.reshape(-1, 1).expand(-1, self.kernel.spec.max_outputs).reshape(-1)
        products = products._replace(ready=products.ready & publish, finite=products.finite & publish)
        outputs = result.pool.output_handles.index_select(0, packet.inverse_order)
        outputs = torch.where(executed.reshape(-1, 1), outputs, -1)
        fallback = result.pool.overflow | ~numerical_ok.all()
        return FormulaDeviceCooperativeResult(
            merged, frontiers, accepted, products, occurrence_cursor + executed.sum(),
            result.pool.data_cursor, result.pool.bank_cursor, fallback,
            outputs.reshape(rows, width, columns, self.kernel.spec.max_outputs),
            decision.score_is_double, decision.eligible,
        )
