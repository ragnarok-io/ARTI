"""Federation selection expressed by ordinary, already executed Fabric outputs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import ClassVar

import torch
from torch import Tensor

from .formula_program_call import FormulaProgramCallCandidateV1
from .formula_program_query_v4 import FormulaProgramEffectCandidateV3, _FormulaProgramExecutionArenaV4
from .formula_program_query_v5 import FormulaProgramQueryV5


class FormulaProgramQueryV6(FormulaProgramQueryV5):
    """An execution-derived frontier, without a separate neural Query owner.

    ``continuations[producer_id][action_id]`` binds a scalar-per-example SSA
    output to a successor's logit. Contributions from executed ordinary members
    are summed. They remain ordinary outputs, retaining their task and Bank
    ancestry; scoring never executes them a second time. Entry candidates start
    with equal scores. STOP has a zero baseline and the existing terminal bounds.

    This version fixes the declared candidate graph, not its numerical behavior.
    A changed Bank affects a response when its producer executes again, never
    retroactively changes an earlier SSA value.
    """

    _component_reference: ClassVar[str] = "arti/formula-program-query@6"
    _uses_external_query: ClassVar[bool] = False

    def __init__(
        self, *, slot_ids: Sequence[str], candidates: Sequence,
        terminal_slots: Mapping[str, str], entry_candidates: Sequence[str],
        continuations: Mapping[str, Mapping[str, str]],
        min_steps: int = 1, max_steps: int = 8, min_tensor_steps: int = 0,
        max_tensor_steps: int | None = None, max_effect_steps: int | None = None,
    ) -> None:
        super().__init__(
            slot_ids=slot_ids, candidates=candidates, terminal_slots=terminal_slots,
            min_steps=min_steps, max_steps=max_steps, min_tensor_steps=min_tensor_steps,
            max_tensor_steps=max_tensor_steps, max_effect_steps=max_effect_steps,
        )
        entries = tuple(entry_candidates)
        members = {candidate.candidate_id: candidate for candidate in self.candidates}
        action_indices = {action: index for index, action in enumerate(self.action_ids)}
        if not entries or len(set(entries)) != len(entries) or not set(entries) <= members.keys():
            raise ValueError("entry_candidates must name unique ordinary members")
        if any(isinstance(members[name], FormulaProgramEffectCandidateV3) for name in entries):
            raise ValueError("entry_candidates cannot start with predecessor effects")
        if not isinstance(continuations, Mapping):
            raise TypeError("continuations must map producer ids to action/SSA bindings")
        bindings = {}
        for producer, edges in sorted(continuations.items()):
            if producer not in members or not isinstance(edges, Mapping) or not edges:
                raise ValueError("continuations require a known producer and non-empty bindings")
            if isinstance(members[producer], FormulaProgramEffectCandidateV3):
                raise ValueError("continuation scores must come from ordinary member outputs")
            outputs = members[producer].output_slot_ids
            normalized = dict(sorted(edges.items()))
            if any(action not in action_indices or slot not in outputs for action, slot in normalized.items()):
                raise ValueError("continuation must bind a known action to its producer output")
            bindings[producer] = normalized
        self.entry_candidates = entries
        self.continuations = bindings
        self._response_columns = tuple(
            (producer, slot, action_indices[action])
            for producer, edges in bindings.items() for action, slot in edges.items()
        )
        self._response_calls = frozenset(
            name for name in bindings if isinstance(members[name], FormulaProgramCallCandidateV1)
        )

    def _response_available(self, arena, producer, slot):
        if producer in self._response_calls:
            return arena.values.get(slot) is not None and any(
                trace.invocation_path == (*arena.invocation_path, producer) and trace.stopped
                for trace in arena.call_traces
            )
        lineage = arena.producer(slot)
        return (
            arena.values.get(slot) is not None and lineage is not None
            and lineage.execution_id == arena.execution_id(producer)
        )

    def routing_mask(self, arena: _FormulaProgramExecutionArenaV4, *, steps: int) -> Tensor:
        available = [False] * len(self.action_ids)
        if steps == 0:
            for name in self.entry_candidates:
                available[self.action_ids.index(name)] = True
        for producer, slot, index in self._response_columns:
            if self._response_available(arena, producer, slot):
                available[index] = True
        available[-1] = True  # Terminal readiness and min/max bounds remain admission's job.
        return torch.tensor(available, dtype=torch.bool, device=arena.device)

    def query_logits(self, arena: _FormulaProgramExecutionArenaV4) -> Tensor:
        contributions: list[list[Tensor]] = [[] for _ in self.action_ids]
        reference = next(value for value in arena.values.values if value is not None)
        dtype = torch.float64 if reference.dtype == torch.float64 else torch.float32
        zero = torch.zeros(arena.batch_size, device=arena.device, dtype=dtype)
        for producer, slot, index in self._response_columns:
            if not self._response_available(arena, producer, slot):
                continue
            value = arena.values.get(slot)
            if not value.is_floating_point() or value.shape not in (
                (arena.batch_size,), (arena.batch_size, 1),
            ):
                raise ValueError("continuation response must be a floating [B] or [B,1] tensor")
            accumulation_dtype = torch.float64 if value.dtype == torch.float64 else torch.float32
            contributions[index].append(value.reshape(arena.batch_size).to(accumulation_dtype))
        return torch.stack(tuple(sum(values, zero) for values in contributions), dim=-1)

    def query(self, arena, *, steps):
        result = super().query(arena, steps=steps)
        if not bool(torch.isfinite(result.logits[:, result.eligible]).all()):
            raise ValueError("non-finite aggregated continuation response")
        return result

    def _has_eligible(self, arena: _FormulaProgramExecutionArenaV4, *, steps: int) -> bool:
        return bool(self.eligible(arena, steps=steps).any())

    def contract_config(self) -> dict[str, object]:
        config = super().contract_config()
        config.pop("hidden_dim")
        config.pop("tensor_encoder_ref")
        config.update(
            entry_candidates=list(self.entry_candidates),
            continuations={producer: dict(edges) for producer, edges in self.continuations.items()},
            query_expression="executed-federation-responses",
            response_reduction="sum-local-logit-contributions",
            response_order="lexicographic-producer-action",
            response_accumulation="float32-minimum",
            response_visibility="immutable-executed-ssa",
        )
        return config


__all__ = ["FormulaProgramQueryV6"]
