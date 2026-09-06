"""Prepared GPU beam waves joining native Query, frontier and Formula execution.

The first width rows are unfinished paths; the following width rows are
completed paths. Only the host boundary reconstructs Python route records.
Numerical rejection/refill is signalled for whole-search native fallback; the
device path must never publish an incompletely refilled beam.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn

from ._formula_device_decision import FormulaDeviceDecisionWave
from ._formula_device_dispatch import FormulaDeviceDispatchLayout, _query_graph, formula_device_dispatch_groups
from ._formula_device_execution import FormulaDeviceExecutionWave
from ._formula_device_frames import FormulaDeviceFrameState, KIND_EFFECT, KIND_ORDINARY, KIND_STOP
from ._formula_device_frontier import TensorRouteFrontier
from .formula_program_call import FormulaProgramCallCandidateV1
from ._formula_device_sources import FormulaDeviceSources, select_sources


class FormulaDeviceSearchState(NamedTuple):
    frames: FormulaDeviceFrameState
    scores: Tensor
    priorities: Tensor
    routes: Tensor
    lengths: Tensor
    membership: Tensor


class FormulaDeviceSearchResult(NamedTuple):
    state: FormulaDeviceSearchState
    data_cursor: Tensor
    bank_cursor: Tensor
    requires_fallback: Tensor
    selected_candidates: Tensor
    parent_rows: Tensor
    eligible: Tensor
    selected_expansions: Tensor
    numeric_attempts: Tensor


class FormulaDeviceSearchWave(nn.Module):
    """One entire search wave, with separate active and completed K-wide beams.

    Coverage columns describe route history, excluding the implicit no-effect
    column. They are supplied by the caller, not guessed from candidate names.
    Route tokens are ranks of full execution ids, including CALL namespaces.
    ``selection_noise`` is optional device-resident Gumbel exploration; model
    scores always remain unperturbed for training and final winner selection.
    """

    def __init__(self, query, execution: FormulaDeviceExecutionWave, *, width: int,
                 local_width: int, candidate_family_ids, candidate_membership,
                 preserve_coverage: bool = True):
        super().__init__()
        from .formula_program_query_v7 import FormulaProgramQueryV7
        if any(type(owner) is FormulaProgramQueryV7 for owner in _query_graph(query)):
            raise TypeError("V7 requires cooperative frontier execution, not a one-action beam wave")
        if type(width) is not int or width < 1:
            raise ValueError("width must be a positive integer")
        self.width = width
        self.local_width = local_width
        self.preserve_coverage = preserve_coverage
        self.execution = execution
        self.kernel = execution.frame_kernel
        self.decision = FormulaDeviceDecisionWave.from_query(
            query, candidate_family_ids=candidate_family_ids, width=local_width, frame_kernel=self.kernel,
        )
        if execution.dispatch.data_layout is not None:
            self.decision.prepare_typed_pools_(execution.dispatch)
        for local in self.decision.query_waves:
            local.preserve_family_coverage = preserve_coverage
        self.layout = FormulaDeviceDispatchLayout(formula_device_dispatch_groups(query)[0])
        self.frontier = TensorRouteFrontier(width, preserve_coverage)
        members = torch.as_tensor(candidate_membership, dtype=torch.bool, device="cpu")
        if members.ndim != 2 or members.shape[0] != self.kernel.candidate_count:
            raise ValueError("candidate_membership must have shape [actions, coverage families]")
        self.register_buffer("candidate_membership", members, persistent=False)

        actions = {}
        offset = 0
        for owner in _query_graph(query):
            actions[id(owner)] = tuple(enumerate((*owner.candidates, None), start=offset))
            offset += len(owner.candidates) + 1
        contexts = []

        def visit(owner, path, stack):
            if len(stack) >= self.kernel.spec.max_depth:
                return
            contexts.append((owner, path, stack))
            for global_id, candidate in actions[id(owner)]:
                if isinstance(candidate, FormulaProgramCallCandidateV1):
                    visit(candidate.child, (*path, candidate.candidate_id), (*stack, global_id))

        visit(query, (), ())
        names = sorted({"/".join((*path, "stop" if candidate is None else candidate.candidate_id))
                        for owner, path, _ in contexts for _, candidate in actions[id(owner)]})
        ranks = {name: index for index, name in enumerate(names)}
        self.execution_names = tuple(names)
        context_calls = torch.full((len(contexts), self.kernel.spec.max_depth), -1, dtype=torch.int64)
        execution_ranks = torch.zeros((len(contexts), self.kernel.candidate_count), dtype=torch.int64)
        for row, (owner, path, stack) in enumerate(contexts):
            context_calls[row, 1:1 + len(stack)] = torch.tensor(stack, dtype=torch.int64)
            for global_id, candidate in actions[id(owner)]:
                name = "/".join((*path, "stop" if candidate is None else candidate.candidate_id))
                execution_ranks[row, global_id] = ranks[name]
        self.register_buffer("context_calls", context_calls, persistent=False)
        self.register_buffer("execution_ranks", execution_ranks, persistent=False)

    @staticmethod
    def _rows(frames: FormulaDeviceFrameState, rows: Tensor) -> FormulaDeviceFrameState:
        return FormulaDeviceFrameState(*(value.index_select(0, rows) for value in frames))

    @torch.no_grad()
    def forward_batch(self, search: FormulaDeviceSearchState, data_pool: Tensor | tuple[Tensor, ...],
                      bank_pool: Tensor | tuple[Tensor, ...],
                      data_cursor: Tensor, bank_cursor: Tensor, operand_finite: Tensor,
                      selection_noise: Tensor | None = None, effect_tape=(), effect_tape_position=None,
                      input_sources=None) -> FormulaDeviceSearchResult:
        """Vectorize independent searches sharing one prepared Query/shape bucket.

        Every state/pool/cursor has a leading episode dimension; immutable
        operands are shared. K selection, allocation and failure remain local
        to each episode. Exploration noise and optional effect tapes/positions,
        when present, also have a leading episode dimension.
        """
        return torch.vmap(
            self.forward, in_dims=(0, 0, 0, 0, 0, None, None if selection_noise is None else 0,
                                   0 if effect_tape else None, None if effect_tape_position is None else 0,
                                   None if input_sources is None else 0),
        )(search, data_pool, bank_pool, data_cursor, bank_cursor, operand_finite,
          selection_noise, effect_tape, effect_tape_position, input_sources)

    @torch.no_grad()
    def forward(self, search: FormulaDeviceSearchState, data_pool: Tensor | tuple[Tensor, ...],
                bank_pool: Tensor | tuple[Tensor, ...],
                data_cursor: Tensor, bank_cursor: Tensor, operand_finite: Tensor,
                selection_noise: Tensor | None = None, effect_tape=(), effect_tape_position=None,
                input_sources=None) -> FormulaDeviceSearchResult:
        search = FormulaDeviceSearchState(*search)
        frames = FormulaDeviceFrameState(*search.frames)
        width = self.width
        if frames.active.shape != (2 * width,) or search.routes.shape[0] != 2 * width:
            raise ValueError("search storage requires width active and width completed rows")
        typed = self.execution.dispatch.data_layout is not None
        data_pools = tuple(data_pool) if typed else (data_pool,)
        if any(pool.ndim < 2 or pool.shape[1] != 1 for pool in data_pools):
            raise ValueError("each value pool requires one native branch batch row")
        device = frames.active.device
        active_rows = torch.arange(width, device=device)
        current = self._rows(frames, active_rows)
        current_sources = (None if input_sources is None else
                           FormulaDeviceSources(*(field[:width] for field in input_sources)))
        decision = self.decision(current, data_pool if typed else data_pool.squeeze(1), bank_pool,
                                 operand_finite, search.scores[:width], selection_noise, current_sources)
        selected = decision.selected_candidates
        count = width * self.local_width
        parent = active_rows[:, None].expand_as(selected).reshape(-1)
        candidates = selected.reshape(-1)
        safe_candidate = candidates.clamp_min(0)
        scores = decision.scores.gather(1, selected.clamp_min(0)).reshape(-1)
        priorities = scores if selection_noise is None else scores + selection_noise.gather(
            1, selected.clamp_min(0),
        ).reshape(-1)
        valid = (candidates >= 0) & current.active.index_select(0, parent)
        stop = self.kernel.candidate_kind.index_select(0, safe_candidate).eq(KIND_STOP)
        root_stop = stop & current.depth.index_select(0, parent).eq(0)

        context = (current.frame_return_call[:, None] == self.context_calls[None]).all(-1)
        context_ids = context.to(torch.int64).argmax(-1)
        ranks = self.execution_ranks.index_select(0, context_ids).gather(1, selected.clamp_min(0)).reshape(-1)
        routes = search.routes.index_select(0, parent)
        lengths = search.lengths.index_select(0, parent)
        route_full = valid & (lengths >= routes.shape[1])
        routes = routes.scatter(1, lengths.clamp_max(routes.shape[1] - 1)[:, None], ranks[:, None])
        membership = search.membership.index_select(0, parent) | self.candidate_membership.index_select(0, safe_candidate)

        carry_rows = active_rows + width
        parent = torch.cat((parent, carry_rows))
        candidates = torch.cat((candidates, torch.full_like(active_rows, -1)))
        scores = torch.cat((scores, search.scores[width:]))
        priorities = torch.cat((priorities, search.priorities[width:]))
        routes = torch.cat((routes, search.routes[width:]))
        lengths = torch.cat((lengths + 1, search.lengths[width:]))
        membership = torch.cat((membership, search.membership[width:]))
        coverage = torch.cat((membership, ~membership.any(1, keepdim=True)), dim=1)
        alive = torch.cat((valid, frames.completed[width:]))
        done = torch.cat((root_stop, torch.ones_like(active_rows, dtype=torch.bool)))
        ready = torch.arange(count + width, device=device) >= count
        identities = torch.arange(count + width, device=device)
        live_packet = self.frontier(priorities, alive & ~done, ready, coverage, routes, identities)
        done_packet = self.frontier(priorities, alive & done, ready, coverage, routes, identities)
        pool_rows = torch.cat((live_packet[:, 0], done_packet[:, 0]))
        keep = pool_rows >= 0
        pool_rows = pool_rows.clamp_min(0)
        sources = parent.index_select(0, pool_rows)
        actions = torch.where(keep, candidates.index_select(0, pool_rows), -1)
        forked = self._rows(frames, sources)
        forked = forked._replace(active=forked.active & keep, completed=forked.completed & keep)
        packet = self.layout(actions[:, None])
        packed_sources = None
        if input_sources is not None:
            # Packet rows refer to pruned/forked rows, not the old source view.
            packed_sources = select_sources(
                input_sources, sources.index_select(0, packet.source_rows),
                packet.candidate_ids.clamp_min(0),
            )
        # Each already-pruned search row executes one action. This width-one
        # packet makes a tape source row the stable result row of this wave.
        result = self.execution(forked, packet, data_pool, bank_pool, data_cursor, bank_cursor,
                                effect_tape, effect_tape_position, packed_sources)
        next_frames = self._rows(result.state, packet.inverse_order)
        accepted = result.event.accepted.index_select(0, packet.inverse_order)
        attempted = keep & (actions >= 0)
        requires_fallback = (
            result.pool.overflow | (attempted & ~accepted).any() | route_full.any()
            | (current.active.any() & ~alive.any())
            | (current.active & ~context.any(1)).any()
            | ~decision.input_finite.all() | ~decision.score_finite.all()
            | (self.preserve_coverage & ~decision.coverage_satisfied.all())
        )
        next_search = FormulaDeviceSearchState(
            FormulaDeviceFrameState(*(torch.where(requires_fallback, old, new)
                                      for old, new in zip(frames, next_frames, strict=True))),
            *(torch.where(requires_fallback, old, new.index_select(0, pool_rows))
              for old, new in zip(search[1:], (scores, priorities, routes, lengths, membership), strict=True)),
        )
        kinds = self.kernel.candidate_kind.index_select(0, actions.clamp_min(0))
        numeric = attempted & ((kinds == KIND_ORDINARY) | (kinds == KIND_EFFECT))
        return FormulaDeviceSearchResult(
            next_search, torch.where(requires_fallback, data_cursor, result.pool.data_cursor),
            torch.where(requires_fallback, bank_cursor, result.pool.bank_cursor), requires_fallback,
            actions, sources, decision.eligible, valid.sum(), numeric.sum(),
        )
