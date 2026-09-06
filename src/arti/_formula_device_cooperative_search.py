"""Completed-product sharing before pruning, over prepared cooperative rounds.

This connects V7 roots and complete child calls to the existing fixed beam.
Root expansion scores do not include the child's local greedy decisions.
"""

from typing import NamedTuple

import torch
from torch import Tensor, nn

from ._formula_device_cooperative import FormulaDeviceCooperativeWave
from ._formula_device_frames import FormulaDeviceFrameState, KIND_CALL, KIND_EFFECT, KIND_ORDINARY
from ._formula_device_frontier import TensorRouteFrontier
from ._formula_device_sources import FormulaDeviceSources, FormulaDeviceSourceBindings, append_completed_sources
from .formula_program_call import FormulaProgramCallCandidateV1
from .formula_program_query_v7 import FormulaProgramQueryV7


class FormulaDeviceCooperativeBeam(NamedTuple):
    frames: FormulaDeviceFrameState
    scores: Tensor
    routes: Tensor
    lengths: Tensor
    score_is_double: Tensor


class FormulaDeviceCallStep(NamedTuple):
    before: FormulaDeviceFrameState
    after: FormulaDeviceFrameState
    candidates: Tensor
    accepted: Tensor
    scores: Tensor


class FormulaDeviceCooperativeSearchResult(NamedTuple):
    state: FormulaDeviceCooperativeBeam
    directory: FormulaDeviceSources
    directory_cursor: Tensor
    occurrence_cursor: Tensor
    data_cursor: Tensor
    bank_cursor: Tensor
    requires_fallback: Tensor
    dropped_products: Tensor
    numeric_attempts: Tensor
    numeric_completed: Tensor
    parent_rows: Tensor
    frontier_rows: Tensor
    candidates: Tensor
    accepted: Tensor
    input_sources: FormulaDeviceSources
    first_occurrence: Tensor
    call_steps: tuple[FormulaDeviceCallStep, ...]
    call_dispatches: Tensor
    call_returns: Tensor
    before: FormulaDeviceFrameState
    eligible: Tensor
    remaining: Tensor
    score_is_double: Tensor


class FormulaDeviceCooperativeSearchWave(nn.Module):
    """Execute every retained expansion head, publish, then select K answers.

    Directory slots name finite port-binding alternatives, not local producers.
    A receiver retains its own Bank snapshot when consuming a donor's product.
    Already completed rows are carried without another STOP or execution.
    """

    def __init__(self, query, execution, *, width, head_width, product_slots=(), publish_slots=None,
                 product_bindings=None):
        super().__init__()
        if type(width) is not int or width < 1:
            raise ValueError("width must be a positive integer")
        if type(query) is not FormulaProgramQueryV7:
            raise TypeError("cooperative search requires a V7 query")
        slots = tuple(product_slots)
        if len(set(slots)) != len(slots) or not set(slots) <= set(query.slot_ids):
            raise ValueError("product slots must be distinct declared SSA slots")
        written = {s for c in query.candidates for s in c.output_slot_ids}
        if set(slots) & (written | set(query.terminal_slots.values())):
            raise ValueError("product slots must be reserved, not outputs or terminals")
        published = written if publish_slots is None else set(publish_slots)
        if not published <= written:
            raise ValueError("publish slots must be actual candidate outputs")
        self.width = width
        self.capacity = len(slots)
        self.wave = FormulaDeviceCooperativeWave(query, execution, head_width=head_width)
        self.kernel = execution.frame_kernel
        calls = tuple(c for c in query.candidates if isinstance(c, FormulaProgramCallCandidateV1))
        self.child_wave = FormulaDeviceCooperativeWave(query, execution, head_width=1, native_child=True) if calls else None
        costs = {}

        def call_cost(owner):
            if id(owner) not in costs:
                costs[id(owner)] = owner.max_steps + 1 + sum(
                    call_cost(c.child) for c in owner.candidates if isinstance(c, FormulaProgramCallCandidateV1))
            return costs[id(owner)]

        self.call_round_limit = max((call_cost(c.child) for c in calls), default=0)
        self.resolver = FormulaDeviceSourceBindings(self.kernel)
        self.prune = TensorRouteFrontier(width, False)
        positions = {name: i for i, name in enumerate(slots)}
        bindings = {} if product_bindings is None else dict(product_bindings)
        valid_ports = {(c.candidate_id, name) for c in query.candidates for name in c.input_slots}
        if not set(bindings) <= valid_ports or not set(bindings.values()) <= set(slots):
            raise ValueError("product bindings must name existing candidate ports and directory slots")
        port_count = self.kernel.candidate_input_slots.shape[1]
        refs = [[positions.get(bindings.get((c.candidate_id, name), s), -1)
                 for name, s in c.input_slots.items()] for c in query.candidates]
        refs.extend([] for _ in range(self.kernel.candidate_count - len(refs)))
        refs = [row + [-1] * (port_count - len(row)) for row in refs]
        # Output ordinals match FrameSpec and completed_sources, not Formula names.
        publication = [[s in published for s in c.output_slot_ids] for c in query.candidates]
        publication.extend([] for _ in range(self.kernel.candidate_count - len(publication)))
        publication = [row + [False] * (self.kernel.spec.max_outputs - len(row)) for row in publication]
        ranks = {name: i for i, name in enumerate(sorted(query.action_ids))}
        for name, value in (
            ("product_refs", torch.tensor(refs, dtype=torch.int64, device="cpu")),
            ("publication", torch.tensor(publication, dtype=torch.bool, device="cpu")),
            ("action_ranks", torch.tensor([ranks[a] for a in query.action_ids] +
                [0] * (self.kernel.candidate_count - len(query.action_ids)), dtype=torch.int64, device="cpu")),
        ):
            self.register_buffer(name, value, persistent=False)

    @staticmethod
    def _rows(frames, rows):
        return FormulaDeviceFrameState(*(value.index_select(0, rows) for value in frames))

    def empty_directory(self, *, device):
        negative = torch.full((self.capacity,), -1, dtype=torch.int64, device=device)
        flags = torch.zeros(self.capacity, dtype=torch.bool, device=device)
        return FormulaDeviceSources(negative.clone(), flags.clone(), flags.clone(),
                                     *(negative.clone() for _ in range(6)))

    @torch.no_grad()
    def forward_steps(self, search, directory, directory_cursor, occurrence_cursor,
                      data_pool, bank_pool, data_cursor, bank_cursor, operand_finite,
                      *, steps, batched=False, selection_bias=None):
        """Run a statically bounded search segment with its original round tape.

        Trace this method to ATen at preparation for whole-segment compilation,
        or capture its existing kernels. Stopped rows remain
        stopped; a live row at the horizon is not a completed answer. No host
        scalar read decides continuation. Pools are caller-owned append storage;
        consume results before reusing them for another search. The returned
        metadata tape grows with steps and beam capacity, not only winners.
        """
        if type(steps) is not int or steps < 1 or type(batched) is not bool:
            raise ValueError("search steps must be a positive static integer and batched a bool")
        if selection_bias is not None and selection_bias.shape[0] != steps:
            raise ValueError("selection bias needs one row per search step")
        run = self.forward_batch if batched else self.forward
        records = []
        for step in range(steps):
            result = run(search, directory, directory_cursor, occurrence_cursor,
                         data_pool, bank_pool, data_cursor, bank_cursor, operand_finite,
                         None if selection_bias is None else selection_bias[step])
            records.append(result)
            search, directory = result.state, result.directory
            directory_cursor, occurrence_cursor = result.directory_cursor, result.occurrence_cursor
            data_cursor, bank_cursor = result.data_cursor, result.bank_cursor
        return tuple(records)

    def _complete_calls(self, result, data, bank, finite, first_occurrence):
        zero = first_occurrence.new_zeros(())
        if self.child_wave is None:
            return result, (), zero, zero, zero, zero
        actions = result.frontiers.candidates
        rows, heads, columns = actions.shape
        count = rows * heads
        selected_call = ((actions[:, :, 0] >= 0) & result.accepted &
                         (self.kernel.candidate_kind[actions[:, :, 0].clamp_min(0)] == KIND_CALL)).reshape(-1)
        state = result.frames
        dc, bc, fallback = result.data_cursor, result.bank_cursor, result.requires_fallback
        attempts, completed, dispatches, returns = zero, zero, selected_call.sum(), zero
        traces = []
        for _ in range(self.call_round_limit):
            pending = selected_call & state.active & (state.depth > 0)
            before = state._replace(active=pending, completed=torch.zeros_like(state.completed))
            child = self.child_wave(before, data, bank, dc, bc, finite, zero)
            next_state = child.frames
            traces.append(FormulaDeviceCallStep(before, next_state, child.frontiers.candidates[:, 0],
                                                child.accepted[:, 0], child.frontiers.selection_log_score[:, 0]))
            kinds = self.kernel.candidate_kind[child.frontiers.candidates.clamp_min(0)]
            used = child.frontiers.candidates >= 0
            numeric = used & ((kinds == KIND_ORDINARY) | (kinds == KIND_EFFECT))
            attempts = attempts + numeric.sum()
            completed = completed + (numeric & child.accepted[:, :, None]).sum()
            dispatches = dispatches + (used & (kinds == KIND_CALL) & child.accepted[:, :, None]).sum()
            returns = returns + (pending & child.accepted[:, 0] & (next_state.depth < state.depth)).sum()
            state = FormulaDeviceFrameState(*(torch.where(pending.reshape((count,) + (1,) * (old.ndim - 1)), new, old)
                for old, new in zip(state, next_state, strict=True)))
            dc, bc = child.data_cursor, child.bank_cursor
            fallback = fallback | child.requires_fallback
        finished = ~selected_call | ((state.depth == 0) & state.active)
        accepted = result.accepted & finished.reshape(rows, heads)
        fallback = fallback | (selected_call & ~finished).any()
        state = state._replace(active=state.active & accepted.reshape(-1),
                               completed=state.completed & accepted.reshape(-1))
        # CALL publication is in original root frontier order, never finish-time order.
        executed = (actions >= 0) & accepted[:, :, None]
        ids = first_occurrence + executed.reshape(-1).to(torch.int64).cumsum(0) - 1
        ids = torch.where(executed.reshape(-1), ids, -1)
        expanded = self._rows(state, torch.arange(count, device=state.active.device).repeat_interleave(columns))
        slots = self.kernel.candidate_output_slots[actions.reshape(-1).clamp_min(0)]
        depth = torch.zeros(count * columns, dtype=torch.int64, device=state.active.device)

        def fields(name):
            return self.kernel._frame(getattr(expanded, name), depth).gather(1, slots.clamp_min(0)).reshape(-1)

        output_handles = fields("value_handles").reshape(*actions.shape, self.kernel.spec.max_outputs)
        valid = executed.reshape(-1, 1) & (slots >= 0) & (output_handles.reshape_as(slots) >= 0)
        products = FormulaDeviceSources(output_handles.reshape(-1), valid.reshape(-1), valid.reshape(-1),
            *(fields(name) for name in ("producer_candidate", "producer_bank", "producer_revision", "producer_bank_handle")),
            ids[:, None].expand_as(slots).reshape(-1),
            torch.arange(slots.shape[1], device=slots.device)[None].expand_as(slots).reshape(-1))
        result = result._replace(frames=state, accepted=accepted, products=products,
            next_occurrence=first_occurrence + executed.sum(), data_cursor=dc, bank_cursor=bc,
            requires_fallback=fallback, output_handles=torch.where(valid.reshape_as(output_handles), output_handles, -1))
        return result, tuple(traces), attempts, completed, dispatches, returns

    @torch.no_grad()
    def forward_batch(self, search, directory, directory_cursor, occurrence_cursor,
                      data_pool, bank_pool, data_cursor, bank_cursor, operand_finite, selection_bias=None):
        """Run independent events in parallel with shared prepared operands."""
        return torch.vmap(self.forward, in_dims=(0, 0, 0, 0, 0, 0, 0, 0, None,
                                                None if selection_bias is None else 0))(
            search, directory, directory_cursor, occurrence_cursor,
            data_pool, bank_pool, data_cursor, bank_cursor, operand_finite, selection_bias,
        )

    @torch.no_grad()
    def forward(self, search, directory, directory_cursor, occurrence_cursor,
                data_pool, bank_pool, data_cursor, bank_cursor, operand_finite, selection_bias=None):
        search = FormulaDeviceCooperativeBeam(*search)
        frames = FormulaDeviceFrameState(*search.frames)
        directory = FormulaDeviceSources(*directory)
        width = self.width
        if frames.active.shape != (2 * width,) or directory.handles.shape != (self.capacity,):
            raise ValueError("search frames or directory do not match prepared capacity")
        device = frames.active.device
        active_rows = torch.arange(width, device=device)
        active = self._rows(frames, active_rows)
        typed = self.wave.execution.dispatch.data_layout is not None
        pools = tuple(data_pool) if typed else (data_pool.squeeze(1),)
        depth = active.depth.clamp(0, self.kernel.spec.max_depth - 1)
        handles = self.kernel._frame(active.value_handles, depth)
        finite = self.wave.decision._reference_finite(pools, handles, self.wave.decision.data_layout)
        refs = self.product_refs[None].expand(width, -1, -1)
        indices = torch.where(refs >= 0, refs, self.capacity).reshape(-1)
        expected = [torch.cat((field, field.new_full((1,), -1)))[indices].reshape(refs.shape)
                    for field in (directory.occurrence, directory.port)]
        sources = self.resolver(active, finite, refs, *expected, directory)
        result = self.wave(active, data_pool, bank_pool, data_cursor, bank_cursor,
                           operand_finite, occurrence_cursor, sources, selection_bias)
        result, call_steps, child_attempts, child_completed, call_dispatches, call_returns = self._complete_calls(
            result, data_pool, bank_pool, operand_finite, occurrence_cursor)

        actions = result.frontiers.candidates
        columns = actions.shape[-1]
        heads = actions.shape[1]
        count = width * heads
        action_rows = actions.reshape(count, columns)
        success = result.accepted.reshape(-1)
        publish = self.publication[actions.clamp_min(0)].reshape(-1)
        pending = result.products._replace(ready=result.products.ready & publish,
                                            finite=result.products.finite & publish)
        directory, directory_cursor, dropped = append_completed_sources(directory, directory_cursor, pending)

        parent = active_rows.repeat_interleave(heads)
        local_scores = result.frontiers.selection_log_score.reshape(-1)
        score_double = search.score_is_double[parent] | result.score_is_double.repeat_interleave(heads)
        scores = search.scores[parent] + local_scores
        if scores.dtype == torch.float64:
            scores = torch.where(score_double, scores, (search.scores[parent].float() + local_scores.float()).double())
        routes, lengths = search.routes[parent], search.lengths[parent]
        route_full = torch.zeros(count, dtype=torch.bool, device=device)
        for column in range(columns):
            action = action_rows[:, column]
            take = success & (action >= 0)
            route_full = route_full | (take & (lengths >= routes.shape[1]))
            position = lengths.clamp_max(routes.shape[1] - 1)[:, None]
            value = self.action_ranks[action.clamp_min(0)]
            routes = routes.scatter(1, position, torch.where(take, value, routes.gather(1, position).squeeze(1))[:, None])
            lengths = lengths + take.to(torch.int64)

        # Carry old completed answers first to preserve stable reference ties.
        all_frames = FormulaDeviceFrameState(*(torch.cat((old[width:], new))
                                               for old, new in zip(frames, result.frames, strict=True)))
        all_scores = torch.cat((search.scores[width:], scores))
        all_routes = torch.cat((search.routes[width:], routes))
        all_lengths = torch.cat((search.lengths[width:], lengths))
        all_score_double = torch.cat((search.score_is_double[width:], score_double))
        identities = torch.arange(width + count, device=device)
        materialized = torch.ones_like(identities, dtype=torch.bool)
        membership = torch.zeros(width + count, 1, dtype=torch.bool, device=device)
        live = self.prune(all_scores, all_frames.active & ~all_frames.completed, materialized,
                          membership, all_routes, identities)[:, 0]
        done = self.prune(all_scores, all_frames.completed, materialized,
                          membership, all_routes, identities)[:, 0]
        selected = torch.cat((live, done))
        keep, safe = selected >= 0, selected.clamp_min(0)
        next_frames = self._rows(all_frames, safe)
        next_frames = next_frames._replace(active=next_frames.active & keep, completed=next_frames.completed & keep)
        state = FormulaDeviceCooperativeBeam(next_frames, all_scores[safe],
                                             all_routes[safe], all_lengths[safe], all_score_double[safe])
        all_parent = torch.cat((active_rows + width, parent))
        frontier_rows = torch.cat((torch.full_like(active_rows, -1), torch.arange(count, device=device)))
        kind = self.kernel.candidate_kind[actions.clamp_min(0)]
        numeric = (actions >= 0) & ((kind == KIND_ORDINARY) | (kind == KIND_EFFECT))
        return FormulaDeviceCooperativeSearchResult(
            state, directory, directory_cursor, result.next_occurrence,
            result.data_cursor, result.bank_cursor, result.requires_fallback | route_full.any(), dropped,
            numeric.sum() + child_attempts, (numeric & result.accepted[:, :, None]).sum() + child_completed,
            torch.where(keep, all_parent[safe], -1), torch.where(keep, frontier_rows[safe], -1), actions,
            result.accepted, sources, occurrence_cursor,
            call_steps, call_dispatches, call_returns,
            active, result.eligible, result.frontiers.remaining, result.score_is_double,
        )
