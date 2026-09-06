"""Cooperative SSA frontiers driven by executed federation responses.

This is a native reference executor, not a device-captured beam search. Candidate
identities include their declared input-source tuple; multiple such candidates
can compete for the same output. Completed products remain available to every
later consumer in the bounded SSA arena.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import ClassVar

import torch
from torch import Tensor

from .formula_program_call import FormulaProgramCallCandidateV1
from .formula_program_query_v3 import FormulaProgramBankState
from .formula_program_query_v4 import FormulaProgramTensorCandidateV3
from .formula_program_query_v5 import FormulaProgramQueryExecutionV5, FormulaProgramQueryTraceV5
from .formula_program_query_v6 import FormulaProgramQueryV6


@dataclass(frozen=True)
class FormulaProgramGraphNodeV1:
    """One executed occurrence and its exact, potentially multi-parent inputs."""

    candidate_id: str
    inputs: tuple[tuple[str, str], ...]
    parents: tuple[str, ...]
    outputs: tuple[str, ...]
    child_trace: FormulaProgramQueryTraceV5 | None = None
    external_inputs: tuple[tuple[str, int, str], ...] = ()
    occurrence_id: int | None = None


@dataclass(frozen=True)
class FormulaProgramGraphFrontierV1:
    nodes: tuple[FormulaProgramGraphNodeV1, ...]
    # Sequential masked log scores. Greedy scheduling is not a sampled graph
    # distribution; these are surrogate-credit inputs, not graph probabilities.
    selection_log_score: Tensor = field(compare=False, repr=False)


@dataclass(frozen=True)
class _FormulaProgramFrontierStep:
    arena: object
    frontier: FormulaProgramGraphFrontierV1
    trace_steps: tuple
    stopped: bool


@dataclass(frozen=True)
class FormulaProgramGraphTraceV1(FormulaProgramQueryTraceV5):
    """Named-output trace with the actual cooperative frontier decisions."""

    frontiers: tuple[FormulaProgramGraphFrontierV1, ...] = ()


def _graph_selection_scores(trace):
    if isinstance(trace, FormulaProgramGraphTraceV1):
        for frontier in trace.frontiers:
            yield frontier.selection_log_score
            for node in frontier.nodes:
                if node.child_trace is not None:
                    yield from _graph_selection_scores(node.child_trace)
    else:
        # Serial children do not publish selection scores, but can themselves
        # call cooperative grandchildren. Never visit the same call via both
        # the flattened trace and the frontier records.
        for step in trace.steps:
            if step.child_trace is not None:
                yield from _graph_selection_scores(step.child_trace)


@dataclass(frozen=True)
class FormulaProgramGraphExecutionV1(FormulaProgramQueryExecutionV5):
    frontiers: tuple[FormulaProgramGraphFrontierV1, ...]
    products: Mapping[str, Tensor]

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "products", MappingProxyType(dict(self.products)))

    @property
    def decision_log_score(self) -> Tensor:
        """Local surrogate scores counted once, never imported ancestor sums."""
        return torch.stack(tuple(_graph_selection_scores(self.trace))).sum()


class FormulaProgramQueryV7(FormulaProgramQueryV6):
    """Execute a searched cooperative DAG using finite typed binding alternatives.

    ``cooperation_width`` counts actual independent ordinary operations, not
    alternative answer beams or Observation views. Each frontier reads one
    snapshot and publishes together. Effects and child calls use singleton
    frontiers, preserving their existing ordered Bank-state semantics.

    The native reference supports one episode at a time, just like V6's native
    hard executor. It does not claim per-input device-pool dispatch or capture.
    """

    _component_reference: ClassVar[str] = "arti/formula-program-query@7"

    def __init__(self, *, cooperation_width: int = 4, **kwargs) -> None:
        if (isinstance(cooperation_width, bool) or not isinstance(cooperation_width, int)
                or cooperation_width < 1):
            raise ValueError("cooperation_width must be a positive integer")
        super().__init__(**kwargs)
        self.cooperation_width = cooperation_width

    def contract_config(self) -> dict[str, object]:
        config = super().contract_config()
        config.update(
            cooperation_width=self.cooperation_width,
            topology="cooperative-multi-parent-ssa",
            selection="response-ranked-finite-input-binding-alternatives",
            source_capacity=len(self.slot_ids),
            publication="completed-frontier",
            effect_scheduling="singleton-ordered-frontiers",
            authoritative_commit="completed-cooperative-graph",
            executor="native-reference",
        )
        return config

    def _select_frontier(self, arena, *, steps, forced=None, first=None):
        if forced is not None and first is not None:
            raise ValueError("use recorded actions or an expansion head, not both")
        response = self.query(arena, steps=steps)
        if arena.batch_size != 1:
            raise ValueError("native cooperative execution requires batch size one")
        # One boundary transfer for reference scheduling, never one CUDA read per
        # candidate. The future pool executor owns device-only scheduling.
        scores = response.masked_logits[0]
        available = response.eligible.tolist()
        score_values = scores.detach().cpu().tolist()
        selected = []
        probabilities = []
        outputs = set()
        tensor_count = 0
        requested = None if forced is None else list(forced)
        while True:
            allowed = [i for i, ok in enumerate(available) if ok]
            if not allowed:
                if selected:
                    break
                raise ValueError("frontier has no admissible action")
            if first is not None and not selected:
                if first not in self.action_ids:
                    raise ValueError("unknown frontier expansion head")
                index = self.action_ids.index(first)
                if index not in allowed:
                    raise ValueError("frontier expansion head is not ready")
            elif requested is None:
                index = min(allowed, key=lambda i: (-score_values[i], self.action_ids[i]))
            else:
                if not requested:
                    break
                action = requested.pop(0)
                if action not in self.action_ids:
                    raise ValueError("replay references an unknown action")
                index = self.action_ids.index(action)
                if index not in allowed:
                    raise ValueError("replay action is not ready in its frontier snapshot")
            stop = index == len(self.candidates)
            if stop and selected:
                if requested is not None:
                    raise ValueError("STOP requires its own frontier")
                break
            candidate = None if stop else self.candidates[index]
            ordinary = isinstance(candidate, FormulaProgramTensorCandidateV3)
            if selected and not ordinary:
                if requested is not None:
                    raise ValueError("effects and child calls require singleton frontiers")
                break
            mask = torch.tensor(available, dtype=torch.bool, device=scores.device)
            probabilities.append(scores[index] - torch.logsumexp(scores.masked_fill(~mask, -torch.inf), 0))
            selected.append(index)
            if stop or not ordinary:
                break
            tensor_count += 1
            outputs.update(candidate.output_slot_ids)
            available[index] = False
            for i, other in enumerate(self.candidates):
                if outputs.intersection((*other.output_slot_ids, *other.requires_empty_slots)):
                    available[i] = False
                # Empty-slot guards of already selected siblings apply both ways.
                if any(set(other.output_slot_ids).intersection(self.candidates[j].requires_empty_slots)
                       for j in selected):
                    available[i] = False
            if (len(selected) >= self.cooperation_width or steps + len(selected) >= self.max_steps
                    or (self.max_tensor_steps is not None
                        and arena.tensor_steps + tensor_count >= self.max_tensor_steps)):
                break
        if requested:
            raise ValueError("replay frontier exceeds width or dispatch budget")
        if not selected:
            raise ValueError("replay must contain a nonempty frontier")
        return selected, torch.stack(probabilities).sum()

    def advance_frontier(self, arena, *, steps: int, saved=None, first=None, external_sources=()):
        """Execute one completed frontier, including before a root can STOP.

        Search may seed a legal first action; all remaining siblings are chosen
        by the same response-ranked scheduler as native execution. External
        sources record (local slot, occurrence, output port) without changing
        the original Bank lineage or installing the donor's proposal overlay.
        """
        if saved is not None and any(n.external_inputs for n in saved.nodes):
            raise NotImplementedError("external products require dependency-graph replay")
        forced = None if saved is None else tuple(n.candidate_id for n in saved.nodes)
        indices, probability = self._select_frontier(arena, steps=steps, forced=forced, first=first)
        if indices == [len(self.candidates)]:
            node = FormulaProgramGraphNodeV1("stop", (), (), ())
            if saved is not None:
                if tuple(replace(n, occurrence_id=None) for n in saved.nodes) != (node,):
                    raise ValueError("replay STOP binding differs")
                node = replace(node, occurrence_id=saved.nodes[0].occurrence_id)
            return _FormulaProgramFrontierStep(
                arena, FormulaProgramGraphFrontierV1((node,), probability), (), True,
            )
        sources = {slot: (occurrence, port) for slot, occurrence, port in external_sources}
        candidates = tuple(self.candidates[i] for i in indices)
        nodes = tuple(FormulaProgramGraphNodeV1(
            c.candidate_id, tuple(c.input_slots.items()),
            tuple(dict.fromkeys(p.execution_id for slot in c.input_slots.values()
                                if (p := arena.producer(slot)) is not None)),
            c.output_slot_ids,
            external_inputs=tuple((port, *sources[slot]) for port, slot in c.input_slots.items()
                                  if slot in sources),
        ) for c in candidates)
        if saved is not None and nodes != tuple(replace(n, child_trace=None, occurrence_id=None) for n in saved.nodes):
            raise ValueError("replay input binding or dependency differs")
        if saved is not None and isinstance(candidates[0], FormulaProgramCallCandidateV1):
            if saved.nodes[0].child_trace is None:
                raise ValueError("CALL replay requires a child decision trace")
            results = (candidates[0](arena, _replay_trace=saved.nodes[0].child_trace),)
        else:
            if saved is not None and any(n.child_trace is not None for n in saved.nodes):
                raise ValueError("ordinary replay cannot contain a child trace")
            results = self.execute_many(tuple((c, arena) for c in candidates))
        nodes = tuple(
            replace(node, child_trace=result.call_traces[-1])
            if isinstance(candidate, FormulaProgramCallCandidateV1) else node
            for node, candidate, result in zip(nodes, candidates, results, strict=True)
        )
        if saved is not None:
            nodes = tuple(replace(node, occurrence_id=old.occurrence_id)
                          for node, old in zip(nodes, saved.nodes, strict=True))
        trace_steps = tuple(self._trace_step(c, result, steps=steps + i)
                            for i, (c, result) in enumerate(zip(candidates, results, strict=True)))
        if len(candidates) == 1:
            arena = results[0]
        else:
            for candidate, result in zip(candidates, results, strict=True):
                arena = arena.write_many(
                    {s: result.values.get(s) for s in candidate.output_slot_ids},
                    producers={s: result.producer(s) for s in candidate.output_slot_ids},
                )
        return _FormulaProgramFrontierStep(
            arena, FormulaProgramGraphFrontierV1(nodes, probability), trace_steps, False,
        )

    def _finish_graph(self, arena, trace, *, steps, frontiers):
        result = self._finish_execution(arena, trace, steps=steps)
        frontiers = tuple(frontiers)
        graph_trace = FormulaProgramGraphTraceV1(
            result.trace.steps, result.trace.stopped, result.trace.invocation_path, frontiers,
        )
        return FormulaProgramGraphExecutionV1(
            result.outputs, result.output_producers, result.bank_state,
            result.proposals, graph_trace, result._owner_token, frontiers,
            {slot: value for slot, value in zip(arena.values.slot_ids, arena.values.values, strict=True)
             if value is not None},
        )

    def _run_graph(self, entry, *, replay=None):
        if replay is not None:
            replay = tuple(replay)
            if any(n.external_inputs for f in replay for n in f.nodes):
                raise NotImplementedError("external products require dependency-graph replay")
        arena = entry
        steps = 0
        trace = []
        frontiers = []
        replay_rows = None if replay is None else iter(replay)
        while True:
            saved = None if replay_rows is None else next(replay_rows, None)
            if replay_rows is not None and saved is None:
                raise ValueError("replay ends before STOP")
            advance = self.advance_frontier(arena, steps=steps, saved=saved)
            frontiers.append(advance.frontier)
            if advance.stopped:
                if replay_rows is not None and next(replay_rows, None) is not None:
                    raise ValueError("replay contains work after STOP")
                return self._finish_graph(arena, trace, steps=steps, frontiers=frontiers)
            arena = advance.arena
            trace.extend(advance.trace_steps)
            steps += len(advance.trace_steps)

    def _execute_arena(self, entry) -> FormulaProgramGraphExecutionV1:
        return self._run_graph(entry)

    def _replay_arena(self, entry, saved) -> FormulaProgramGraphExecutionV1:
        if not isinstance(saved, FormulaProgramGraphTraceV1) or not saved.stopped:
            raise ValueError("cooperative replay requires a completed frontier trace")
        if saved.invocation_path != entry.invocation_path:
            raise ValueError("replay invocation path differs")
        result = self._run_graph(entry, replay=saved.frontiers)
        if result.trace.steps != saved.steps:
            raise ValueError("replay child trace differs from recorded frontiers")
        return result

    def replay(
        self, values: Mapping[str, Tensor], frontiers: Sequence[FormulaProgramGraphFrontierV1],
        *, bank_state: FormulaProgramBankState | None = None,
    ) -> FormulaProgramGraphExecutionV1:
        """Rebuild each recorded occurrence once, sharing all real input tensors.

        Supply the same initial Bank snapshot for numerical reproduction. This
        retains autograd for values and current response scores; it neither
        imitates teacher intermediates nor replays a producer for each consumer.
        CALL nodes carry their own recursive decision tape. Replaying recomputes
        current values but never replaces recorded child choices by a new search.
        """
        frontiers = tuple(frontiers)
        return self._run_graph(self._arena(values, bank_state=bank_state), replay=frontiers)


__all__ = [
    "FormulaProgramQueryV7", "FormulaProgramGraphExecutionV1",
    "FormulaProgramGraphFrontierV1", "FormulaProgramGraphNodeV1", "FormulaProgramGraphTraceV1",
]
