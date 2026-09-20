from dataclasses import replace

import torch

from arti import mechanisms as m
from benchmarks._federated_v4_federation import build_autonomous_effect_federation
from benchmarks.train_federated_autonomous_federation import (
    _candidates_at_step,
    _replay_route,
    search_to_terminal,
)
from benchmarks.train_federated_branch_visible_federation import _SearchBranch, _ranked_candidates_many


def _fixture():
    federation = build_autonomous_effect_federation(
        hidden_dim=4, rank=4, seed=981, device=torch.device("cpu"),
        plastic_branches=4, min_operations=1, max_operations=2,
        max_effect_operations=3, effect_families=("outer", "blend"),
    )
    first = federation.transition_layers[0][0]
    terminal = federation.terminal_layers[0][0]
    with torch.no_grad():
        head = federation.query.network[-1]
        head.weight.zero_()
        head.bias.fill_(-100.0)
        for candidate in (first, terminal):
            head.bias[federation.query.candidate_ids.index(candidate.candidate_id)] = 100.0
        for layer in federation.tail_layers:
            head.bias[federation.query.candidate_ids.index(layer[0].candidate_id)] = 90.0
        head.bias[-1] = 0.0
    return federation, first, terminal


def test_native_lazy_eager_and_replay_search_real_post_abi_effects_and_stop():
    federation, _, _ = _fixture()
    query = federation.query
    x = torch.randn(1, 2, 4)
    root = query.initial_bank_state()
    native = query({"x": x})
    native_route = tuple(step.candidate_id for step in native.trace.steps)
    assert native_route[-1] == "stop"
    assert len(native.proposals) == 3
    assert all("tail" in item.effect_candidate_id for item in native.proposals)
    outputs = []
    for lazy in (False, True):
        result = search_to_terminal(
            federation, (_SearchBranch(query._arena({"x": x}), x.new_zeros(()), ()),),
            width=1, beam_width=1, preserve_effect_coverage=False, prune_before_execute=lazy,
        )
        branch = result.branches[0]
        route = tuple(row["candidate_id"] for row in branch.route)
        assert route == native_route
        assert branch.arena.tensor_steps == 2 and branch.arena.effect_steps == 3
        assert branch.arena.values.get("effect-tail-2") is branch.arena.values.get("terminal")
        assert result.scored_stop_expansions == 1
        assert result.executed_expansions == 5
        replay = _replay_route(federation, x, root, route)
        torch.testing.assert_close(replay.value, native.value)
        for actual, expected in zip(replay.bank_state.values, native.bank_state.values, strict=True):
            torch.testing.assert_close(actual, expected)
        outputs.append(branch.log_probability)
    torch.testing.assert_close(*outputs)
    assert root.revisions == query.initial_bank_state().revisions


def test_same_layout_different_legality_gets_independent_stop_normalization():
    federation, first, terminal = _fixture()
    query = federation.query
    x = torch.randn(1, 2, 4)
    ready = terminal(first(query._arena({"x": x})))
    spent = replace(ready, effect_steps=query.max_effect_steps)
    branches = tuple(_SearchBranch(arena, x.new_zeros(()), ()) for arena in (ready, spent))
    candidates = _candidates_at_step(federation, 2)
    rows = _ranked_candidates_many(
        federation, branches, candidates, steps=2, width=len(candidates) + 1, include_stop=True,
    )
    assert any(candidate is not None for candidate, _ in rows[0])
    assert len(rows[1]) == 1 and rows[1][0][0] is None
    torch.testing.assert_close(rows[1][0][1], x.new_zeros(()))
    for branch, row in zip(branches, rows, strict=True):
        direct = query.query(branch.arena, steps=2).masked_logits.log_softmax(-1)[0]
        for candidate, probability in row:
            index = len(query.candidates) if candidate is None else query.candidate_ids.index(candidate.candidate_id)
            torch.testing.assert_close(probability, direct[index])
    stop_probability = next(score for candidate, score in rows[0] if candidate is None)
    gradient = torch.autograd.grad(-stop_probability, query.network[-1].bias)[0]
    assert gradient[-1] < 0 and torch.isfinite(gradient).all()


def test_effect_paths_can_consume_budget_before_or_after_output_selection():
    federation, first, _ = _fixture()
    query = federation.query
    arena = first(query._arena({"x": torch.randn(1, 2, 4)}))
    for index in (1, 2, 3):
        effect = next(item for item in federation.transition_layers[index] if isinstance(item, m.FormulaProgramEffectCandidateV3))
        assert query._candidate_eligible(effect, arena, steps=index)
        arena = effect(arena)
    assert arena.tensor_steps == 1 and arena.effect_steps == 3
    terminal = federation.terminal_layers[3][0]
    assert query._candidate_eligible(terminal, arena, steps=4)
    arena = terminal(arena)
    assert query._stop_eligible(arena, steps=5)
    assert not any(query._candidate_eligible(item, arena, steps=5) for item in federation.tail_layers[0])


def test_terminal_cannot_close_tensor_path_before_minimum_after_effect():
    federation = build_autonomous_effect_federation(
        hidden_dim=4, rank=4, seed=983, device=torch.device("cpu"),
        plastic_branches=4, min_operations=2, max_operations=3,
        max_effect_operations=1, effect_families=("outer",),
    )
    query = federation.query
    first = federation.transition_layers[0][0]
    effect = next(item for item in federation.transition_layers[1] if isinstance(item, m.FormulaProgramEffectCandidateV3))
    premature = federation.terminal_layers[0][0]
    continuation = federation.transition_layers[2][0]
    terminal = federation.terminal_layers[1][0]
    x = torch.randn(1, 2, 4)
    arena = effect(first(query._arena({"x": x})))
    assert premature.accepts(arena)
    assert not query._candidate_eligible(premature, arena, steps=2)
    assert query._candidate_eligible(continuation, arena, steps=2)
    with torch.no_grad():
        head = query.network[-1]
        head.weight.zero_()
        head.bias.fill_(-100.0)
        for candidate in (first, effect, premature, continuation, terminal):
            head.bias[query.candidate_ids.index(candidate.candidate_id)] = 100.0
        head.bias[query.candidate_ids.index(terminal.candidate_id)] = 110.0
        head.bias[-1] = 200.0
    expected = tuple(item.candidate_id for item in (first, effect, continuation, terminal)) + ("stop",)
    assert tuple(row.candidate_id for row in query({"x": x}).trace.steps) == expected
    for lazy in (False, True):
        result = search_to_terminal(
            federation, (_SearchBranch(query._arena({"x": x}), x.new_zeros(()), ()),),
            width=1, beam_width=1, preserve_effect_coverage=False, prune_before_execute=lazy,
        )
        branch = result.branches[0]
        assert tuple(row["candidate_id"] for row in branch.route) == expected
        assert branch.arena.tensor_steps == 3 and branch.arena.effect_steps == 1
