from types import SimpleNamespace
import json

import pytest
from safetensors.torch import load_file
import torch

from arti import mechanisms as m
from benchmarks._federated_captured_search import captured_search_execution
from benchmarks._federated_plasticity_participation import replay_participant_graphs
from benchmarks.train_federated_interacting_learners import (
    InteractionEpisode, Learner, interact, interact_many, learner_objective, save_round, train_minibatch,
)
from test_federated_plasticity_participation import _fork_query, _search


def _learners(device="cpu"):
    learners = []
    for index in range(2):
        source, *_ = _fork_query(device, unused_branches=False)
        query = m.FormulaProgramQueryV5(
            slot_ids=(*source.slot_ids, "message"), candidates=tuple(source.candidates),
            terminal_slots={"output": "out"}, min_steps=4, max_steps=4, hidden_dim=8,
        ).to(device)
        with torch.no_grad():
            query.network[-1].weight.zero_()
            query.network[-1].bias.zero_()
            query.network[-1].bias[2] = 1
        learners.append(Learner(f"learner-{index}", SimpleNamespace(query=query),
                                torch.optim.AdamW(query.parameters(), lr=1e-3)))
    return tuple(learners)


def _episode(device="cpu", factor=1.0):
    x = torch.tensor([[0.25, -0.4, 0.7]], device=device) * factor
    return InteractionEpisode(str(factor), (x, x * -0.3), (x * 0.8, x * -0.6), (x * 0.4, x * 0.2))


def test_probe_cannot_change_the_primal_even_when_a_local_difference_overflows():
    from benchmarks._federated_plasticity_participation import _OccurrenceSensitivity

    successor = torch.tensor([2e38, -0.0], requires_grad=True)
    previous = torch.tensor([-2e38, 0.0])
    probe = torch.zeros((), requires_grad=True)
    value = _OccurrenceSensitivity.apply(successor, previous, probe)
    assert torch.equal(value.detach().view(torch.int32), successor.detach().view(torch.int32))
    assert bool(torch.isfinite(value).all())


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_probe_preserves_primal_and_parameter_gradient_and_has_local_finite_difference(device):
    query, first, *_ = _fork_query(device)
    x = torch.tensor([[0.25, -0.4, 0.7]], device=device)
    initial = query.initial_bank_state()
    searched = _search(query, x, initial)
    _, reference = replay_participant_graphs(query, {"x": x}, searched.branches, bank_state=initial)
    _, measured = replay_participant_graphs(query, {"x": x}, searched.branches, bank_state=initial,
                                            participation_probe=True)
    torch.testing.assert_close(reference.state.values, measured.state.values, rtol=0, atol=0)
    assert measured.participation_probe.numel() == 3 and measured.shared_references == 1
    assert measured.prefix_scores.shape == (3,)
    params = tuple(query.parameters())
    loss = measured.state.value(first.bank_slot_ref).square().mean()
    actual = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    expected = torch.autograd.grad(reference.state.value(first.bank_slot_ref).square().mean(), params, allow_unused=True)
    for a, b in zip(actual, expected, strict=True):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b)
    marginal = torch.autograd.grad(loss, measured.participation_probe)[0]

    def perturbed(delta, occurrence):
        value = initial.value(first.bank_slot_ref)
        with torch.no_grad():
            for index, proposal in enumerate(reference.occurrences):
                successor = proposal.transition.apply(value)
                value = successor + (delta * (successor - value) if index == occurrence else 0)
        return value.square().mean()

    finite = torch.stack([(perturbed(1e-3, i) - perturbed(-1e-3, i)) / 2e-3 for i in range(3)])
    torch.testing.assert_close(marginal, finite, rtol=3e-3, atol=2e-4)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("mode", ("ordered-operators", "parallel-delta"))
def test_actual_carry_replay_and_saved_bank_are_all_participant_not_winner(device, mode, tmp_path):
    learners, episode = _learners(device), _episode(device)
    with captured_search_execution(horizon=8, record_effect_operands=True) as backend:
        result = interact(learners, episode, width=2, plasticity_participation="retained", plasticity_composition=mode)
        if device == "cuda":
            assert backend.completed == 8 and not backend.fallbacks
    for history in result.events:
        for index, event in enumerate(history):
            assert len(event.composition.occurrences) == 3
            assert sum(event.after.revisions) == 3 * (index + 1)
            assert sum(event.search.winner.execution.bank_state.revisions) == 3 * index + 2
            assert any(not torch.equal(a, b) for a, b in zip(event.after.values,
                                                             event.search.winner.execution.bank_state.values))
            if index + 1 < len(history):
                assert history[index + 1].before is event.after
    objective, row = learner_objective(learners, episode, result, 0, route_credit="retained-vjp",
                                       exploration_weight=0, external_credit=0)
    assert row["bank_replay_max_abs_error"] < 1e-5 and row["service_replay_max_abs_error"] < 1e-5
    assert row["participation_credit_norm"] > 0
    peer = tuple(learners[1].query.parameters())
    assert all(g is None for g in torch.autograd.grad(objective, peer, allow_unused=True, retain_graph=True))
    objective.backward()
    query = learners[0].query
    right = next(c for c in query.candidates if c.candidate_id == "right")
    assert all(step["candidate_id"] != "right" for event in result.events[0] for step in event.route)
    for parameter in (right.operand_store.tensor("writer"), right.execution_count_tensor()):
        assert parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
    assert any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in query.network.parameters())
    assert all(owner.value.grad is None and not owner.value.requires_grad for owner in query.owner_states)
    save_round(tmp_path / "round", learners, step=1, config={"route_credit": "retained-vjp"}, metrics={}, interaction=result)
    saved = load_file(str(tmp_path / "round" / "learner-0" / "episode-bank.safetensors"), device=device)
    for index, value in enumerate(result.events[0][-1].after.values):
        torch.testing.assert_close(saved[str(index)], value, rtol=0, atol=0)
    events = json.loads((tmp_path / "round" / "learner-0" / "events.json").read_text())
    assert events["plasticity_participation"] == "retained"
    assert events["plasticity_composition"] == mode
    assert row["plasticity_composition"] == mode
    assert all(e["participating_effects"] == 3 and len(e["participant_routes"]) == 2 for e in events["events"])


def test_legacy_credit_cannot_train_a_retained_forward_and_reverse():
    learners, episode = _learners(), _episode()
    retained = interact(learners, episode, width=2, plasticity_participation="retained")
    with pytest.raises(ValueError, match="retained carry requires"):
        learner_objective(learners, episode, retained, 0)
    winner = interact(learners, episode, width=2)
    with pytest.raises(ValueError, match="requires a retained-participant"):
        learner_objective(learners, episode, winner, 0, route_credit="retained-vjp")


@pytest.mark.parametrize("mode", ("ordered-operators", "parallel-delta"))
def test_retained_minibatch_keeps_optimizer_bank_and_peers_separate(mode):
    learners = _learners()
    episodes = (_episode(), _episode(factor=-0.7))
    results, metrics = train_minibatch(learners, episodes, width=2, route_credit="retained-vjp", external_credit=0,
                                     plasticity_composition=mode)
    assert metrics["plasticity_composition"] == mode
    assert len(results) == 2 and metrics["plasticity_participation"] == "retained"
    for row in metrics["episodes"]:
        assert all(r["bank_replay_max_abs_error"] < 1e-5 for r in row["learners"])
    for learner in learners:
        assert learner.optimizer.state
        optimized = {id(p) for group in learner.optimizer.param_groups for p in group["params"]}
        assert not optimized & {id(o.value) for o in learner.query.owner_states}
        assert all(o.value.grad is None for o in learner.query.owner_states)


def test_retained_episode_batch_and_repeated_capture_are_independent():
    learners = _learners("cuda")
    episodes = (_episode("cuda"), _episode("cuda", factor=-0.7))
    expected = tuple(interact(learners, e, width=2, plasticity_participation="retained") for e in episodes)
    with captured_search_execution(horizon=8, record_effect_operands=True) as backend:
        actual = interact_many(learners, episodes, width=2, plasticity_participation="retained")
        assert backend.batched_searches == 16 and not backend.fallbacks
        for a, b in zip(expected, actual, strict=True):
            for ah, bh in zip(a.events, b.events, strict=True):
                for ae, be in zip(ah, bh, strict=True):
                    torch.testing.assert_close(ae.after.values, be.after.values)
                    torch.testing.assert_close(ae.output, be.output)
        objective, row = learner_objective(learners, episodes[0], actual[0], 0, route_credit="retained-vjp", external_credit=0)
        objective.backward()
        assert row["bank_replay_max_abs_error"] < 1e-5


@pytest.mark.parametrize("mode", ("ordered-operators", "parallel-delta"))
def test_frozen_retained_evaluation_batches_true_reset_without_peer_state_carry(mode):
    from benchmarks.evaluate_federated_interacting_learners import evaluate_episode, evaluate_episodes

    learners = _learners("cuda")
    for learner in learners:
        learner.query.requires_grad_(False)
    snapshots = [{name: value.clone() for name, value in learner.query.state_dict().items()} for learner in learners]
    episodes = (_episode("cuda"), _episode("cuda", factor=-0.7))
    expected = [evaluate_episode(learners, e, width=2, reset_control=True, plasticity_participation="retained",
                                  plasticity_composition=mode)
                for e in episodes]
    with captured_search_execution(horizon=8, record_effect_operands=True) as backend:
        actual = evaluate_episodes(learners, episodes, width=2, reset_control=True, plasticity_participation="retained",
                                    plasticity_composition=mode)
        assert backend.batched_searches == 24 and not backend.fallbacks
    for a, b in zip(expected, actual, strict=True):
        for ar, br in zip(a["learners"], b["learners"], strict=True):
            for name in ("mse", "reset_mse", "reset_output_max_abs_difference",
                         "service_mse_for_peer", "service_output_max_abs", "final_output_max_abs"):
                assert ar[name] == pytest.approx(br[name], rel=1e-5, abs=1e-5)
            assert br["participating_support_effects"] == br["pre_service_revision_sum"] == 6
            assert br["reset_output_max_abs_difference"] > 0
    for before, learner in zip(snapshots, learners, strict=True):
        for name, value in learner.query.state_dict().items():
            torch.testing.assert_close(before[name], value, rtol=0, atol=0)
        assert all(parameter.grad is None for parameter in learner.query.parameters())


def test_service_error_is_attributed_to_producer_not_the_receiver_final_head():
    from benchmarks.evaluate_federated_interacting_learners import _episode_report

    def event(value):
        return SimpleNamespace(output=torch.tensor([float(value)]), before=SimpleNamespace(revisions=(0,)),
                               route=({"candidate_id": "return-external-message"},))

    episode = SimpleNamespace(episode_id="different-producer-and-receiver", targets=(torch.tensor([2.]), torch.tensor([5.])))
    result = SimpleNamespace(events=((event(5), event(100)), (event(100), event(5))), service_index=0)
    rows = _episode_report(episode, result, None)["learners"]
    assert rows[0]["mse"] == rows[1]["service_mse_for_peer"] == 98**2
    assert rows[1]["mse"] == rows[0]["service_mse_for_peer"] == 0
    assert [row["service_receiver"] for row in rows] == [1, 0]


def test_failed_frozen_arm_is_saved_and_cannot_report_a_successful_comparison(tmp_path, monkeypatch):
    from benchmarks import evaluate_federated_interacting_learners as evaluation

    learners = _learners()
    checkpoint = tmp_path / "round"
    config = {"seed": 1, "hidden_dim": 3, "branches": 2, "device": "cpu", "max_operations": 4,
              "max_effect_operations": 4, "route_credit": "retained-vjp", "width": 2}
    save_round(checkpoint, learners, step=0, config=config, metrics={})
    monkeypatch.setattr(evaluation, "build_learners", lambda **kwargs: _learners())

    def failed(*args, **kwargs):
        error = ValueError("non-finite fixture")
        error.add_note("episode=fixture, event=2")
        raise error

    monkeypatch.setattr(evaluation, "evaluate_episodes", failed)
    output = tmp_path / "evaluation.json"
    monkeypatch.setattr("sys.argv", ["evaluate", "--checkpoint", str(checkpoint), "--baseline", str(checkpoint),
                                     "--output", str(output), "--episodes", "1"])
    with pytest.raises(SystemExit) as error:
        evaluation.main()
    assert error.value.code == 1
    report = json.loads(output.read_text())
    assert not report["comparison_complete"] and report["parameters_frozen_and_unchanged"]
    assert all(arm["status"] == "failed" and arm["context"] == ["episode=fixture, event=2"]
               for arm in report["arms"].values())
