"""Prepared selection from scores emitted by ordinary Federation producers.

Only wiring is prepared. Forward consumes current score tensors and occurrence
presence, never executes a producer or owns a learned scoring network.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from ._formula_device_query import FormulaDeviceQueryResult


def is_response_query(query) -> bool:
    """Recognize the native execution-derived Query contract."""
    from .formula_program_query_v6 import FormulaProgramQueryV6
    from .formula_program_query_v7 import FormulaProgramQueryV7
    return type(query) in (FormulaProgramQueryV6, FormulaProgramQueryV7)


class FormulaDeviceResponse(nn.Module):
    """Sum emitted contributions and select against the native admission mask.

    ``score_values`` and ``present`` have shape [rows, source_count]. Each source
    is an exact (producer candidate, SSA output slot) pair in ``score_sources``.
    The caller establishes producer identity; occupancy alone is insufficient.
    All numerical decisions remain device-side. STOP is always structurally
    available but still requires the caller's ordinary stop admission.
    """

    def __init__(
        self, *, action_ids: Sequence[str], entry_candidates: Sequence[str],
        continuations: Mapping[str, Mapping[str, str]], width: int,
        candidate_family_ids: Tensor | Sequence[int] | None = None,
        action_priority: Tensor | Sequence[int] | None = None,
        preserve_family_coverage: bool = False,
    ) -> None:
        super().__init__()
        actions = tuple(action_ids)
        if not actions or len(set(actions)) != len(actions) or actions[-1] != "stop":
            raise ValueError("action_ids must be unique and end with stop")
        if type(width) is not int or width < 1:
            raise ValueError("width must be a positive integer")
        entries = tuple(entry_candidates)
        if len(set(entries)) != len(entries) or any(a not in actions[:-1] for a in entries):
            raise ValueError("entry_candidates must name unique non-stop actions")
        sources: list[tuple[str, str]] = []
        source_indices: dict[tuple[str, str], int] = {}
        action_indices = {action: index for index, action in enumerate(actions)}
        by_action: list[list[int]] = [[] for _ in actions]
        for producer, responses in sorted(continuations.items()):
            if producer not in actions[:-1]:
                raise ValueError("continuation producer must name a candidate")
            for action, slot in sorted(responses.items()):
                if action not in actions or not isinstance(slot, str) or not slot:
                    raise ValueError("continuation must name an action and score slot")
                source = (producer, slot)
                if source not in source_indices:
                    source_indices[source] = len(sources)
                    sources.append(source)
                by_action[action_indices[action]].append(source_indices[source])
        self.action_ids = actions
        self.score_sources = tuple(sources)
        self.action_sources = tuple(tuple(indices) for indices in by_action)
        depth = max(map(len, by_action), default=0)
        ordered = torch.full((depth, len(actions)), len(sources), dtype=torch.int64, device="cpu")
        for action, indices in enumerate(by_action):
            if indices:
                ordered[:len(indices), action] = torch.tensor(indices, dtype=torch.int64, device="cpu")
        self.register_buffer("ordered_sources", ordered, persistent=False)
        self.width = width
        self.preserve_family_coverage = preserve_family_coverage
        self.register_buffer("entry_mask", torch.tensor([a in entries for a in actions]), persistent=False)
        families = torch.as_tensor(
            [0] * len(actions) if candidate_family_ids is None else candidate_family_ids,
            dtype=torch.int64, device="cpu",
        )
        priority = torch.as_tensor(
            list(range(len(actions))) if action_priority is None else action_priority,
            dtype=torch.int64, device="cpu",
        )
        if families.shape != (len(actions),) or priority.shape != (len(actions),):
            raise ValueError("family IDs and action priority must match action_ids")
        self.register_buffer("action_priority", priority, persistent=False)
        self.register_buffer("candidate_family_ids", families, persistent=False)
        self.register_buffer("_tie_order", torch.argsort(priority, stable=True), persistent=False)
        family_values = tuple(dict.fromkeys(families.tolist()))
        self.register_buffer("_family_membership", families[:, None].eq(torch.tensor(family_values, device=families.device)[None]),
                             persistent=False)

    @classmethod
    def from_query(cls, query, *, width: int, candidate_family_ids=None, preserve_family_coverage=False):
        """Prepare the V6 public wiring contract without reading runtime values."""
        return cls(action_ids=query.action_ids, entry_candidates=query.entry_candidates,
                   continuations=query.continuations, width=width,
                   candidate_family_ids=candidate_family_ids, action_priority=query._action_priority,
                   preserve_family_coverage=preserve_family_coverage)

    def response_logits(self, score_values: Tensor, present: Tensor, parent_scores: Tensor,
                        reference_double=None, source_double=None, *, ordered_sources=None):
        """Return differentiable sums and structural continuation availability."""
        present = present.to(dtype=torch.bool)
        dtype = torch.float64 if score_values.dtype == torch.float64 else torch.float32
        score_values = score_values.to(dtype)
        zero = torch.zeros_like(parent_scores[:, None], dtype=dtype)
        padded_values = torch.cat((score_values, zero), dim=1)
        padded_present = torch.cat((present, torch.zeros_like(zero, dtype=torch.bool)), dim=1)
        value = zero.expand(-1, len(self.action_ids))
        available = torch.zeros_like(value, dtype=torch.bool)
        mixed = source_double is not None and dtype == torch.float64
        if mixed:
            double = reference_double[:, None].expand_as(available)
            padded_double = torch.cat((source_double, torch.zeros_like(zero, dtype=torch.bool)), dim=1)
        # Batch independent actions, preserving each action's sequential sum.
        order = self.ordered_sources if ordered_sources is None else ordered_sources
        for sources in order.unbind(0):
            here = padded_present.index_select(1, sources)
            part = torch.where(here, padded_values.index_select(1, sources), zero)
            if mixed:
                double = double | (here & padded_double.index_select(1, sources))
                # The unused FP32 branch must not overflow during backward.
                low_value = torch.where(double, zero, value).float()
                low_part = torch.where(double, zero, part).float()
                low = (low_value + low_part).to(dtype)
                summed = torch.where(double, value + part, low)
            else:
                summed = value + part
            value = torch.where(sources[None] < score_values.shape[1], summed, value)
            available = available | here
        return value, available

    def forward(
        self, score_values: Tensor, present: Tensor, eligible: Tensor,
        parent_scores: Tensor, steps: Tensor, selection_noise: Tensor | None = None,
        reference_double=None, source_double=None,
    ) -> FormulaDeviceQueryResult:
        """Consume [R,C] scores, [R,A] admission and scalar or [R] device steps."""
        logits, available = self.response_logits(score_values, present, parent_scores,
                                                  reference_double, source_double)
        initial = (steps == 0).expand_as(parent_scores)
        stop = torch.arange(len(self.action_ids), device=eligible.device) == len(self.action_ids) - 1
        frontier = available | (initial[:, None] & self.entry_mask[None]) | stop[None]
        eligible = eligible.to(dtype=torch.bool) & frontier
        rows, actions = eligible.shape
        has_legal = eligible.any(dim=-1)
        legal_logits = torch.where(eligible, logits, torch.zeros_like(logits))
        eligible_finite = torch.isfinite(legal_logits).all(dim=-1)
        legal_logits = torch.nan_to_num(legal_logits, nan=0.0, posinf=0.0, neginf=0.0)
        masked_logits = legal_logits.masked_fill(~eligible, -float("inf"))
        softmax_logits = torch.where(has_legal[:, None], masked_logits, torch.zeros_like(masked_logits))
        log_probs = torch.log_softmax(softmax_logits, dim=-1)
        if source_double is not None and logits.dtype == torch.float64:
            row_double = reference_double | (present & source_double).any(-1)
            log_probs = torch.where(row_double[:, None], log_probs,
                                    torch.log_softmax(softmax_logits.float(), dim=-1).double())
        logprob_finite = torch.isfinite(torch.where(eligible, log_probs, torch.zeros_like(log_probs))).all(-1)
        parent_finite = ~(torch.isnan(parent_scores) | torch.isposinf(parent_scores))
        parents = torch.where(parent_finite, parent_scores, torch.zeros_like(parent_scores))
        input_finite = torch.ones_like(has_legal)
        logits_finite = torch.where(has_legal, parent_finite & eligible_finite, input_finite)
        score_finite = torch.where(has_legal, parent_finite & eligible_finite & logprob_finite, input_finite)
        scores = parents[:, None] + log_probs
        scores = torch.where((score_finite & has_legal)[:, None], scores, torch.zeros_like(scores))

        # Keep local ranking independent of parent magnitude, including -inf.
        tie_order = self._tie_order.expand(rows, -1)
        positions = torch.argsort(log_probs.gather(1, tie_order), dim=1, descending=True, stable=True)
        order = tie_order.gather(1, positions)
        if selection_noise is not None:
            positions = torch.argsort((scores + selection_noise).gather(1, order), dim=1,
                                     descending=True, stable=True)
            order = order.gather(1, positions)
        ranks = torch.zeros_like(order).scatter(
            1, order, torch.arange(actions, device=order.device).expand(rows, -1),
        )
        valid = eligible & score_finite[:, None]
        members = valid[:, :, None] & self._family_membership[None]
        rank_values = ranks[:, :, None]
        sentinel = torch.full_like(rank_values.expand_as(members), actions)
        first_rank = torch.where(members, rank_values, sentinel).amin(dim=1)
        representative = (members & (rank_values == first_rank[:, None])).any(dim=-1)
        if self.preserve_family_coverage:
            category = torch.where(valid, torch.where(representative, torch.zeros_like(ranks),
                                   torch.ones_like(ranks)), torch.full_like(ranks, 2))
        else:
            category = torch.where(valid, torch.zeros_like(ranks), torch.ones_like(ranks))
        selection_order = torch.argsort(category * actions + ranks, dim=1, stable=True)
        selected = selection_order[:, :self.width]
        local_selected = torch.zeros_like(valid).scatter(1, selected, valid.gather(1, selected))
        return FormulaDeviceQueryResult(
            scores, masked_logits, eligible, input_finite, score_finite, score_finite,
            ranks, order, local_selected, representative, selection_order,
            representative.sum(dim=-1) <= self.width,
            logits_finite,
        )
