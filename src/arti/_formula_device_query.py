"""Device-side candidate scoring and stable local selection.

This module deliberately contains no runtime synchronization or host-side
selection. It is the tensor-only part of a prepared Formula Query path:
native slot summaries are built, the supplied network produces candidate
logits, and local selection follows the same family-first ordering as
``_take_candidates``.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import torch
from torch import Tensor, nn

from .formula_program_query_v4 import _SUMMARY_WIDTH, _numeric_summary


class FormulaDeviceQueryResult(NamedTuple):
    """Tensor-only result of a prepared device query."""

    scores: Tensor
    masked_logits: Tensor
    eligible: Tensor
    input_finite: Tensor
    score_finite: Tensor
    finite: Tensor
    ranks: Tensor
    order: Tensor
    local_selected: Tensor
    representative: Tensor
    selection_order: Tensor
    coverage_satisfied: Tensor
    logits_finite: Tensor


class FormulaDeviceQuery(nn.Module):
    """Run a prepared Formula query and local candidate selection on-device.

    ``network`` and ``tensor_encoder`` are the native Formula components. A
    caller with a ``FormulaProgramQueryV4`` can pass its ``network`` and
    ``tensor_encoder`` directly, together with candidate priority and family
    metadata. Eligibility is supplied by the caller so the hot path needs no
    Python-side candidate filtering.
    """

    def __init__(
        self,
        network: nn.Module,
        *,
        slot_count: int,
        candidate_family_ids: Tensor | Sequence[int],
        width: int,
        tensor_encoder: nn.Module | None = None,
        action_priority: Tensor | Sequence[int] | None = None,
        preserve_family_coverage: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(network, nn.Module):
            raise TypeError("network must be an nn.Module")
        if slot_count <= 0:
            raise ValueError("slot_count must be positive")
        if width <= 0:
            raise ValueError("width must be positive")

        # Metadata is prepared on CPU and becomes device-local only through
        # the normal Module.to() migration, including CUDA constructor input.
        family_ids = torch.as_tensor(
            candidate_family_ids, dtype=torch.long, device="cpu"
        )
        if family_ids.ndim != 1 or family_ids.numel() == 0:
            raise ValueError("candidate_family_ids must be a non-empty vector")
        if action_priority is None:
            priority = torch.arange(family_ids.numel(), dtype=torch.long)
        else:
            priority = torch.as_tensor(action_priority, dtype=torch.long, device="cpu")
            if priority.ndim != 1 or priority.numel() != family_ids.numel():
                raise ValueError("action_priority must match candidate_family_ids")

        self.network = network
        self.tensor_encoder = tensor_encoder
        self.slot_count = slot_count
        self.width = width
        self.preserve_family_coverage = preserve_family_coverage
        self.register_buffer("candidate_family_ids", family_ids, persistent=False)
        self.register_buffer("action_priority", priority, persistent=False)
        self.register_buffer(
            "_tie_order",
            torch.argsort(priority, stable=True),
            persistent=False,
        )

        # One column per distinct family keeps the runtime working set at
        # [rows,A,F], rather than the quadratic [rows,A,A] representation.
        family_values: list[int] = []
        for family_id in family_ids.tolist():
            if family_id not in family_values:
                family_values.append(family_id)
        family_values_tensor = torch.tensor(
            family_values, dtype=torch.long, device="cpu"
        )
        self.register_buffer("_family_values", family_values_tensor, persistent=False)
        self.register_buffer(
            "_family_membership",
            family_ids[:, None].eq(family_values_tensor[None, :]),
            persistent=False,
        )

    def _reference_tensor(self) -> Tensor:
        for parameter in self.network.parameters():
            return parameter
        for buffer in self.network.buffers():
            return buffer
        if self.tensor_encoder is not None:
            for parameter in self.tensor_encoder.parameters():
                return parameter
            for buffer in self.tensor_encoder.buffers():
                return buffer
        raise ValueError("network or tensor_encoder must have a device/dtype tensor")

    @torch.no_grad()
    def summarize_value(self, value: Tensor, occupied: Tensor) -> tuple[Tensor, Tensor]:
        """The native single-slot summary, shared by homogeneous and typed rows."""
        reference = self._reference_tensor()
        rows = occupied.shape[0]
        value = (value.to(dtype=reference.dtype) if self.tensor_encoder is None
                 else self.tensor_encoder._prepare_tokens(value))
        value_finite = torch.isfinite(value.reshape(rows, -1)).all(dim=-1)
        finite = ~occupied | value_finite
        value = torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        mask = occupied.reshape((rows,) + (1,) * (value.ndim - 1))
        value = torch.where(mask, value, torch.zeros_like(value))
        summary = (_numeric_summary(value.reshape(rows, -1)) if self.tensor_encoder is None
                   else self.tensor_encoder._encode_tokens(value))
        return torch.where(occupied[:, None], summary, torch.zeros_like(summary)), finite

    @property
    def summary_width(self) -> int:
        return _SUMMARY_WIDTH if self.tensor_encoder is None else self.tensor_encoder.output_width

    @torch.no_grad()
    def _summarize(
        self, ssa_values: tuple[Tensor, ...], occupied: Tensor
    ) -> tuple[Tensor, Tensor]:
        rows = occupied.shape[0]
        summaries: list[Tensor] = []
        input_finite = torch.ones(rows, dtype=torch.bool, device=occupied.device)

        for slot, value in enumerate(ssa_values):
            slot_summary, finite = self.summarize_value(value, occupied[:, slot])
            input_finite = input_finite & finite
            summaries.append(slot_summary)

        summary = torch.cat(summaries, dim=-1)
        return summary, input_finite

    @torch.no_grad()
    def forward(
        self,
        ssa_values: tuple[Tensor, ...],
        occupied: Tensor,
        eligible: Tensor,
        parent_scores: Tensor,
        selection_noise: Tensor | None = None,
    ) -> FormulaDeviceQueryResult:
        """Return device tensors for scores, ordering, and local selection."""

        occupied = occupied.to(dtype=torch.bool)
        summary, input_finite = self._summarize(tuple(ssa_values), occupied)
        return self.score_summary(summary, input_finite, eligible, parent_scores, selection_noise)

    @torch.no_grad()
    def score_summary(self, summary, input_finite, eligible, parent_scores, selection_noise=None):
        """Score already-native summaries without changing Query mathematics."""
        eligible = eligible.to(dtype=torch.bool)
        rows = summary.shape[0]
        actions = self.action_priority.numel()
        has_legal = eligible.any(dim=-1)
        summary_finite = torch.isfinite(summary).all(dim=-1)
        summary = torch.where(
            (input_finite & has_legal)[:, None], summary, torch.zeros_like(summary)
        )
        logits = self.network(summary)

        # Replace only values that can participate in normalization. An
        # ineligible NaN/Inf therefore cannot poison an otherwise legal row.
        legal_logits = torch.where(eligible, logits, torch.zeros_like(logits))
        eligible_finite = torch.isfinite(legal_logits).all(dim=-1)
        legal_logits = torch.nan_to_num(
            legal_logits, nan=0.0, posinf=0.0, neginf=0.0
        )
        masked_logits = legal_logits.masked_fill(~eligible, -float("inf"))
        softmax_logits = torch.where(
            has_legal[:, None], masked_logits, torch.zeros_like(masked_logits)
        )
        log_probs = torch.log_softmax(softmax_logits, dim=-1)
        legal_logprob_finite = torch.isfinite(
            torch.where(eligible, log_probs, torch.zeros_like(log_probs))
        ).all(dim=-1)

        # A parent -inf is a legal low score in the native search. Only NaN
        # and +inf invalidate the parent contribution.
        parent_finite = ~(torch.isnan(parent_scores) | torch.isposinf(parent_scores))
        parent_scores = torch.where(
            parent_finite, parent_scores, torch.zeros_like(parent_scores)
        )
        # Dead rows are already excluded by eligibility and must not become a
        # numerical rejection. Live rows retain separate input/score gates.
        input_finite = torch.where(has_legal, input_finite, torch.ones_like(input_finite))
        logits_finite = torch.where(
            has_legal, parent_finite & summary_finite & eligible_finite,
            torch.ones_like(parent_finite),
        )
        score_finite = torch.where(
            has_legal,
            parent_finite & summary_finite & eligible_finite & legal_logprob_finite,
            torch.ones_like(parent_finite),
        )
        finite = input_finite & score_finite
        scores = parent_scores[:, None] + log_probs
        scores = torch.where(
            (finite & has_legal)[:, None], scores, torch.zeros_like(scores)
        )

        # Stable score ordering with lower-priority-first ties. The
        # preliminary order is exactly the fixed action priority.
        tie_order = self._tie_order.expand(rows, -1)
        # Native local ranking is based on this Query's log probabilities.
        # Parent path scores are added only to the expansion score: using the
        # total here would erase local order for a legal -inf parent (and can
        # introduce rounding ties for large finite parent magnitudes).
        score_positions = torch.argsort(
            log_probs.gather(1, tie_order), dim=1, descending=True, stable=True
        )
        order = tie_order.gather(1, score_positions)
        if selection_noise is not None:
            # Exploration changes selection only, never the model probability.
            exploration_order = torch.argsort(
                (scores + selection_noise).gather(1, order), dim=1, descending=True, stable=True,
            )
            order = order.gather(1, exploration_order)
        positions = torch.arange(actions, device=order.device).expand(rows, -1)
        ranks = torch.zeros_like(order).scatter(1, order, positions)

        valid = eligible & finite[:, None]
        members = valid[:, :, None] & self._family_membership[None, :, :]
        rank_values = ranks[:, :, None]
        sentinel = torch.full_like(rank_values.expand_as(members), actions)
        first_rank = torch.where(members, rank_values, sentinel).amin(dim=1)
        representative = (
            members & (rank_values == first_rank[:, None, :])
        ).any(dim=-1)

        if self.preserve_family_coverage:
            category = torch.where(
                valid,
                torch.where(
                    representative, torch.zeros_like(ranks), torch.ones_like(ranks)
                ),
                torch.full_like(ranks, 2),
            )
        else:
            category = torch.where(valid, torch.zeros_like(ranks), torch.ones_like(ranks))

        admission_priority = category * actions + ranks
        selection_order = torch.argsort(
            admission_priority, dim=1, descending=False, stable=True
        )
        selected = selection_order[:, : self.width]
        selected_valid = valid.gather(1, selected)
        local_selected = torch.zeros_like(valid).scatter(1, selected, selected_valid)
        coverage_satisfied = representative.sum(dim=-1) <= self.width

        return FormulaDeviceQueryResult(
            scores=scores,
            masked_logits=masked_logits,
            eligible=eligible,
            input_finite=input_finite,
            score_finite=score_finite,
            finite=finite,
            ranks=ranks,
            order=order,
            local_selected=local_selected,
            representative=representative,
            selection_order=selection_order,
            coverage_satisfied=coverage_satisfied,
            logits_finite=logits_finite,
        )
