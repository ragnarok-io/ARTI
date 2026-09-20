from __future__ import annotations

import pytest
import torch
from safetensors.torch import save_file

from arti import mechanisms
from benchmarks._federated_v4_federation import build_autonomous_effect_federation
from benchmarks.train_federated_autonomous_federation import (
    _SearchBranch,
    _candidate_action_family,
    _candidates_at_step,
    _coverage_prune,
    _family_balanced_exploration_weights,
    _load_query_weights,
    _ranked_candidates_with_coverage_many,
    _replay_route,
    evaluate_frozen_adaptation,
    expected_episode_loss,
)
from benchmarks.train_federated_self_modifying_federation import (
    make_association_episodes,
)
from benchmarks.train_federated_branch_visible_federation import (
    _prune,
    _ranked_candidates_many,
)


def _fixture() -> object:
    return build_autonomous_effect_federation(
        hidden_dim=4,
        rank=4,
        seed=811,
        device=torch.device("cpu"),
        plastic_branches=4,
        min_operations=1,
        max_operations=3,
        effect_families=("outer", "blend"),
    )


def test_autonomous_search_trains_slow_law_without_mutating_task_bank() -> None:
    federation = _fixture()
    episode = make_association_episodes(
        split="train",
        seed=821,
        count=1,
        hidden_dim=4,
        support_count=2,
        device=torch.device("cpu"),
    )[0]
    before = tuple(owner.value.clone() for owner in federation.query.owner_states)

    loss, diagnostics = expected_episode_loss(
        federation,
        episode,
        width=4,
        beam_width=4,
        hard_alignment_weight=0.1,
    )
    loss.backward()

    assert bool(torch.isfinite(loss))
    assert diagnostics["retained_paths"] >= 1.0
    assert diagnostics["maximum_frontier"] >= diagnostics["retained_paths"]
    assert 0.0 < diagnostics["effect_path_fraction"] <= 1.0
    assert diagnostics["training_mse"] >= 0.0
    assert any(parameter.grad is not None for parameter in federation.query.parameters())
    assert all(
        torch.equal(previous, owner.value)
        for previous, owner in zip(before, federation.query.owner_states, strict=True)
    )


def test_autonomous_step_slices_equal_full_query_eligibility() -> None:
    federation = _fixture()
    arena = federation.query._arena({"x": torch.randn(1, 2, 4)})
    first = next(
        candidate
        for candidate in federation.transition_layers[0]
        if isinstance(candidate, mechanisms.FormulaProgramTensorCandidateV3)
    )
    arenas = (arena, first(arena))

    for step, current in enumerate(arenas):
        eligible = federation.query.eligible(current, steps=step)
        eligible_ids = {
            candidate.candidate_id
            for candidate, allowed in zip(
                federation.query.candidates,
                eligible[:-1],
                strict=True,
            )
            if bool(allowed)
        }
        sliced_ids = {
            candidate.candidate_id
            for candidate in _candidates_at_step(federation, step)
            if candidate.accepts(current)
        }
        assert sliced_ids == eligible_ids


def test_batched_ranking_preserves_log_probabilities_and_gradients() -> None:
    federation = _fixture()
    inputs = (torch.randn(1, 2, 4), torch.randn(1, 2, 4))
    branches = tuple(
        _SearchBranch(federation.query._arena({"x": value}), value.new_zeros(()), ())
        for value in inputs
    )
    candidates = _candidates_at_step(federation, 0)
    available = tuple(candidate for candidate in candidates if candidate.accepts(branches[0].arena))
    encoder_batches = []
    handle = federation.query.tensor_encoder.register_forward_pre_hook(
        lambda _module, args: encoder_batches.append(args[0].shape[0])
    )
    try:
        rows = _ranked_candidates_many(federation, branches, candidates, steps=0, width=3)
    finally:
        handle.remove()
    assert encoder_batches == [len(branches)]
    all_ids = [candidate.candidate_id for candidate in federation.query.candidates]
    summaries = torch.cat(tuple(federation.query._summarize(branch.arena) for branch in branches))
    logits = federation.query.network(summaries)
    expected_rows = []
    for row_index, actual in enumerate(rows):
        probabilities = torch.stack(tuple(logits[row_index, all_ids.index(candidate.candidate_id)] for candidate in available)).log_softmax(0)
        order = sorted(range(len(available)), key=lambda i: (-float(probabilities[i].detach()), available[i].candidate_id))[:3]
        expected = probabilities[order]
        assert [candidate.candidate_id for candidate, _ in actual] == [available[i].candidate_id for i in order]
        assert torch.allclose(torch.stack(tuple(value for _, value in actual)), expected)
        expected_rows.append(expected)
    actual_loss = sum(value.square() for row in rows for _, value in row)
    expected_loss = sum(row.square().sum() for row in expected_rows)
    parameter = federation.query.network[-1].bias
    actual_gradient = torch.autograd.grad(actual_loss, parameter)[0]
    expected_gradient = torch.autograd.grad(expected_loss, parameter)[0]
    assert torch.allclose(actual_gradient, expected_gradient, atol=1e-7, rtol=1e-5)
    assert _prune((), beam_width=3) == ()


def test_effect_coverage_beam_retains_low_probability_effect_family() -> None:
    federation = _fixture()
    value = torch.randn(1, 2, 4)
    arena = federation.query._arena({"x": value})
    ordinary = _SearchBranch(
        arena,
        value.new_tensor(0.0),
        (
            {
                "step": 0,
                "kind": "ordinary",
                "atom_ref": "ordinary",
                "candidate_id": "ordinary",
            },
        ),
    )
    effect = _SearchBranch(
        arena,
        value.new_tensor(-100.0),
        (
            {
                "step": 0,
                "kind": "ordinary",
                "atom_ref": "ordinary",
                "candidate_id": "ordinary",
            },
            {
                "step": 1,
                "kind": "effect",
                "atom_ref": "effect-a",
                "candidate_id": "effect-a",
            },
            {
                "step": 0,
                "kind": "ordinary",
                "atom_ref": "ordinary-next-event",
                "candidate_id": "ordinary-next-event",
            },
        ),
    )

    retained = _coverage_prune((ordinary, effect), beam_width=2)

    assert ordinary in retained
    assert effect in retained


def test_underflowed_candidate_keeps_route_learning_gradient() -> None:
    from benchmarks.train_federated_branch_visible_federation import _expand_stage

    federation = _fixture()
    value = torch.randn(1, 2, 4)
    branch = _SearchBranch(federation.query._arena({"x": value}), value.new_zeros(()), ())
    candidates = tuple(item for item in _candidates_at_step(federation, 0) if item.accepts(branch.arena))
    all_ids = [item.candidate_id for item in federation.query.candidates]
    high = all_ids.index(candidates[0].candidate_id)
    low = all_ids.index(candidates[-1].candidate_id)
    head = federation.query.network[-1]
    with torch.no_grad():
        head.weight.zero_()
        head.bias.zero_()
        head.bias[high] = 100.0
        head.bias[low] = -100.0
    expanded = _expand_stage(federation, (branch,), candidates, steps=0, width=len(candidates), beam_width=len(candidates))
    target = next(item for item in expanded if item.route[-1]["candidate_id"] == candidates[-1].candidate_id)
    assert float(target.log_probability.detach()) == pytest.approx(-200.0)
    gradient = torch.autograd.grad(-target.log_probability, head.bias)[0]
    assert gradient[low] < -0.99
    assert gradient[high] > 0.99
    narrow = _ranked_candidates_many(federation, (branch,), candidates, steps=0, width=2)[0]
    wide = _ranked_candidates_many(federation, (branch,), candidates, steps=0, width=len(candidates))[0]
    wide_scores = {candidate.candidate_id: score for candidate, score in wide}
    for candidate, score in narrow:
        torch.testing.assert_close(score, wide_scores[candidate.candidate_id])


def test_candidate_coverage_happens_before_width_truncation() -> None:
    federation = _fixture()
    value = torch.randn(1, 2, 4)
    arena = federation.query._arena({"x": value})
    first = next(
        candidate
        for candidate in federation.transition_layers[0]
        if isinstance(candidate, mechanisms.FormulaProgramTensorCandidateV3)
    )
    branch = _SearchBranch(first(arena), value.new_zeros(()), ())

    rows = _ranked_candidates_with_coverage_many(
        federation,
        (branch,),
        _candidates_at_step(federation, 1),
        steps=1,
        width=4,
    )
    families = {
        _candidate_action_family(federation, candidate) for candidate, _ in rows[0]
    }

    assert families == {
        "ordinary",
        "terminal",
        "arti/formula-atom-neural-plasticity-blend@1",
        "arti/formula-atom-neural-plasticity-outer@2",
    }


def test_coverage_classifies_each_candidate_once_without_changing_routes(monkeypatch) -> None:
    from benchmarks import train_federated_autonomous_federation as experiment

    federation = _fixture()
    first = federation.transition_layers[0][0]
    branches = tuple(
        _SearchBranch(
            first(federation.query._arena({"x": torch.randn(1, 2, 4)})),
            torch.zeros(()),
            (),
        )
        for _ in range(3)
    )
    candidates = _candidates_at_step(federation, 1)
    full_rows = _ranked_candidates_many(
        federation, branches, candidates, steps=1, width=len(candidates)
    )
    expected = []
    for ranked in full_rows:
        representatives = {}
        for candidate, score in ranked:
            representatives.setdefault(_candidate_action_family(federation, candidate), (candidate, score))
        expected.append(tuple(representatives.values()))

    calls = []
    classify = experiment._candidate_action_family

    def counted(*args, **kwargs):
        calls.append(args[1].candidate_id)
        return classify(*args, **kwargs)

    monkeypatch.setattr(experiment, "_candidate_action_family", counted)
    actual = _ranked_candidates_with_coverage_many(
        federation, branches, candidates, steps=1, width=4
    )
    assert calls == [candidate.candidate_id for candidate in candidates]
    for actual_row, expected_row in zip(actual, expected, strict=True):
        assert [item[0].candidate_id for item in actual_row] == [item[0].candidate_id for item in expected_row]
        torch.testing.assert_close(
            torch.stack([item[1] for item in actual_row]),
            torch.stack([item[1] for item in expected_row]),
            rtol=0, atol=0,
        )
    parameters = tuple(federation.query.network.parameters())
    actual_gradients = torch.autograd.grad(sum(score for row in actual for _, score in row), parameters)
    expected_gradients = torch.autograd.grad(sum(score for row in expected for _, score in row), parameters)
    for actual_gradient, expected_gradient in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual_gradient, expected_gradient, rtol=0, atol=0)


def test_exploration_mass_is_balanced_by_effect_family() -> None:
    federation = _fixture()
    value = torch.randn(1, 2, 4)
    arena = federation.query._arena({"x": value})

    def branch(*atom_refs: str) -> _SearchBranch:
        route = tuple(
            {
                "step": index,
                "kind": "effect" if atom_ref != "ordinary" else "ordinary",
                "atom_ref": atom_ref,
                "candidate_id": f"{atom_ref}-{index}",
            }
            for index, atom_ref in enumerate(atom_refs)
        )
        return _SearchBranch(arena, value.new_zeros(()), route)

    branches = (
        branch("ordinary"),
        branch("ordinary", "effect-a"),
        branch("ordinary", "effect-a"),
        branch("ordinary", "effect-b"),
    )
    weights = _family_balanced_exploration_weights(
        branches,
        reference=value.new_zeros((len(branches),)),
    )

    assert torch.allclose(weights, value.new_tensor((1 / 3, 1 / 6, 1 / 6, 1 / 3)))
    assert torch.allclose(weights.sum(), value.new_tensor(1.0))


def test_autonomous_frozen_report_records_bounded_learned_paths() -> None:
    federation = _fixture()
    federation.query.requires_grad_(False)
    episodes = make_association_episodes(
        split="held-out",
        seed=823,
        count=2,
        hidden_dim=4,
        support_count=2,
        device=torch.device("cpu"),
    )

    report = evaluate_frozen_adaptation(federation, episodes)

    assert report["fresh_replay_exact_fraction"] == 1.0
    assert report["discarded_support_compute_exact_fraction"] == 1.0
    assert report["native_vs_replay_output_max_abs"] == 0.0
    assert report["slow_state_digest_unchanged"] is True
    assert report["write_operation_depth"]["minimum"] >= 2  # type: ignore[index]
    assert report["write_operation_depth"]["maximum"] <= 4  # type: ignore[index]
    assert report["read_operation_depth"]["minimum"] >= 2  # type: ignore[index]
    assert report["read_operation_depth"]["maximum"] <= 4  # type: ignore[index]
    assert set(report["mse"]) == {
        "correct",
        "replay_correct",
        "reset",
        "no_effect",
        "swap",
        "scrub",
    }


def test_native_hard_route_replays_without_query_selection() -> None:
    federation = _fixture()
    value = torch.randn(1, 2, 4)
    state = federation.query.initial_bank_state()

    native = federation.query({"x": value}, bank_state=state)
    route = tuple(step.candidate_id for step in native.trace.steps)
    replayed = _replay_route(federation, value, state, route)

    assert torch.equal(replayed.value, native.value)
    assert replayed.bank_state.revisions == native.bank_state.revisions
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            replayed.bank_state.values,
            native.bank_state.values,
            strict=True,
        )
    )
    assert replayed.proposal_count == len(native.proposals)


def test_no_effect_replay_executes_but_discards_bank_proposals() -> None:
    federation = _fixture()
    value = torch.randn(1, 2, 4)
    initial = federation.query.initial_bank_state()
    producer = next(
        candidate
        for candidate in federation.transition_layers[0]
        if isinstance(candidate, mechanisms.FormulaProgramTensorCandidateV3)
    )
    effect = next(
        candidate
        for candidate in federation.transition_layers[1]
        if isinstance(candidate, mechanisms.FormulaProgramEffectCandidateV3)
    )
    terminal = federation.terminal_layers[1][0]
    route = (producer.candidate_id, effect.candidate_id, terminal.candidate_id, "stop")

    committed = _replay_route(federation, value, initial, route)
    discarded = _replay_route(
        federation,
        value,
        initial,
        route,
        apply_effects=False,
    )

    assert committed.proposal_count == discarded.proposal_count == 1
    assert any(
        not torch.equal(left, right)
        for left, right in zip(committed.bank_state.values, initial.values, strict=True)
    )
    assert discarded.bank_state.revisions == initial.revisions
    assert all(
        torch.equal(left, right)
        for left, right in zip(discarded.bank_state.values, initial.values, strict=True)
    )


def test_old_checkpoint_migration_only_allows_new_execution_counts(tmp_path) -> None:
    source = _fixture()
    old_state = {
        name: value
        for name, value in source.query.state_dict().items()
        if not name.endswith(".execution_count")
    }
    checkpoint = tmp_path / "old.safetensors"
    save_file(old_state, checkpoint)

    target = _fixture()
    missing = _load_query_weights(
        target,
        checkpoint,
        device=torch.device("cpu"),
    )

    assert missing
    assert all(name.endswith(".execution_count") for name in missing)


def test_old_checkpoint_migration_rejects_other_missing_weights(tmp_path) -> None:
    source = _fixture()
    old_state = {
        name: value
        for name, value in source.query.state_dict().items()
        if not name.endswith(".execution_count")
    }
    removed = next(name for name in old_state if old_state[name].is_floating_point())
    del old_state[removed]
    checkpoint = tmp_path / "invalid.safetensors"
    save_file(old_state, checkpoint)

    target = _fixture()
    with pytest.raises(RuntimeError, match="checkpoint does not match"):
        _load_query_weights(
            target,
            checkpoint,
            device=torch.device("cpu"),
        )
