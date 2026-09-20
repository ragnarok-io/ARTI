from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from benchmarks import _federated_association_tasks as tasks
from benchmarks import train_federated_interacting_learners as training
from benchmarks import evaluate_federated_transfer as evaluation
from benchmarks._federated_captured_search import captured_search_execution
from benchmarks.evaluate_federated_lifetime import load_carry, save_carry
from benchmarks.train_federated_branch_visible_federation import _module_digest
from test_federated_interacting_learners import _tiny_learners
from test_federated_retained_training import _episode, _learners


@pytest.mark.parametrize("device", ("cpu", "cuda"))
@pytest.mark.parametrize("key", (0, 2))
def test_related_control_changes_only_prior_geometry(device, key):
    rng = torch.get_rng_state().clone()
    args = dict(seed=172, hidden_dim=8, device=device, presentations=2, changed_key=key)
    related = tasks.make_transfer_episode(**args)
    unrelated = tasks.make_transfer_episode(**args, history="unrelated")
    baseline = tasks.make_episode(seed=172, hidden_dim=8, device=device)
    assert torch.equal(rng, torch.get_rng_state())
    torch.testing.assert_close(related.queries, unrelated.queries, rtol=0, atol=0)
    torch.testing.assert_close(related.targets, unrelated.targets, rtol=0, atol=0)
    torch.testing.assert_close(related.supports[:3], unrelated.supports[:3], rtol=0, atol=0)
    torch.testing.assert_close(related.supports[5:], unrelated.supports[5:], rtol=0, atol=0)
    assert len(related.supports) == 8 and related.updated_key == key
    for episode in (related, unrelated):
        torch.testing.assert_close(episode.supports[5], baseline.supports[2 - key], rtol=0, atol=0)
        assert all(not torch.equal(episode.supports[-1], past) for past in episode.supports[:6])
        values = [value[:, 1:] for value in baseline.supports]
        values[key] = episode.supports[-1][:, 1:]
        for i, (left, right) in enumerate(((0, 1), (1, 2))):
            torch.testing.assert_close(episode.targets[i], episode.queries[i] + (values[left] + values[right]) / 2**0.5)
    change = related.supports[-1][:, 1:] - baseline.supports[key][:, 1:]
    unrelated_change = unrelated.supports[3][:, 1:] - baseline.supports[2 - key][:, 1:]
    assert abs(float((change * unrelated_change).sum())) < 1e-6
    assert not torch.equal(related.supports[3], unrelated.supports[3])
    for a, b in zip(related.supports[3:5], unrelated.supports[3:5], strict=True):
        torch.testing.assert_close(a[:, 1:].norm(), b[:, 1:].norm(), rtol=1e-5, atol=1e-6)
        assert abs(float(a[:, 1:].mean())) < 1e-6
        assert abs(float(b[:, 1:].mean())) < 1e-6
    for value in (*related.supports, *related.targets, *related.queries):
        assert value.device.type == device and not value.requires_grad


def test_presentation_budget_only_repeats_the_new_actual_observation():
    options = dict(seed=123, hidden_dim=8, device="cpu")
    zero = tasks.make_transfer_episode(**options, presentations=0)
    for n in (1, 2, 3):
        episode = tasks.make_transfer_episode(**options, presentations=n)
        assert len(episode.supports) == 6 + n
        torch.testing.assert_close(episode.supports[:6], zero.supports, rtol=0, atol=0)
        torch.testing.assert_close(episode.targets, zero.targets, rtol=0, atol=0)
        assert all(torch.equal(s, episode.supports[-1]) for s in episode.supports[6:])
    cold = tasks.make_transfer_episode(**options, presentations=1, history="none")
    assert len(cold.supports) == 4
    torch.testing.assert_close(cold.targets, zero.targets, rtol=0, atol=0)


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_new_value_stream_has_stable_prefixes_and_only_observed_latest_targets(device):
    options = dict(seed=628, hidden_dim=8, device=device, changed_key=0, current_mode="new-values")
    stages = [tasks.make_transfer_episode(**options, presentations=n) for n in range(5)]
    baseline = tasks.make_episode(seed=628, hidden_dim=8, device=device)
    repeated = tasks.make_transfer_episode(seed=628, hidden_dim=8, device=device, changed_key=0, presentations=1)
    torch.testing.assert_close(stages[1].supports, repeated.supports, rtol=0, atol=0)
    for previous, current in zip(stages, stages[1:]):
        torch.testing.assert_close(current.supports[:-1], previous.supports, rtol=0, atol=0)
        unrelated = tasks.make_transfer_episode(**options, history="unrelated", presentations=len(current.supports) - 6)
        torch.testing.assert_close(current.supports[5:], unrelated.supports[5:], rtol=0, atol=0)
        torch.testing.assert_close(current.targets, unrelated.targets, rtol=0, atol=0)
        expected = current.queries[0] + (current.supports[-1][:, 1:] + baseline.supports[1][:, 1:]) / 2**0.5
        torch.testing.assert_close(current.targets[0], expected, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(current.targets[1], baseline.targets[1], rtol=1e-6, atol=1e-7)
        if len(previous.supports) > 6:
            assert not torch.equal(current.targets[0], previous.targets[0])
            assert not torch.equal(current.supports[-1], previous.supports[-1])
    changes = torch.cat([s[:, 1:] - baseline.supports[0][:, 1:] for s in stages[-1].supports[6:]], dim=1).squeeze(0)
    assert torch.linalg.matrix_rank(changes.float(), atol=1e-6) == 2


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_staged_observation_stream_matches_full_prefix_and_uses_each_new_target(device):
    learners = _learners(device)
    for learner in learners:
        learner.query.eval().requires_grad_(False)
    source = _episode(device)
    prefix = source.supports * 3
    stages = tuple(tuple(replace(source, supports=prefix + tuple(source.supports[0] * (0.4 + .1*j) for j in range(n)),
                                targets=tuple(t * (1 + .1*n) for t in source.targets), updated_key=0)
                         for _ in range(2)) for n in range(3))
    scope = captured_search_execution(horizon=8, record_effect_operands=True) if device == "cuda" else nullcontext()
    with torch.no_grad(), scope:
        rows, _, outputs = evaluation.evaluate_transfer(
            learners, stages[-1], presentations=2, width=2, plasticity_composition="parallel-delta", stage_episodes=stages)
        prior_direct = None
        for n, batch in enumerate(stages):
            direct = training.interact_many(learners, batch, width=2, plasticity_participation="retained",
                                           plasticity_composition="parallel-delta")
            for index, (episode, result) in enumerate(zip(batch, direct, strict=True)):
                errors = [float((history[-1].output - target).square().mean())
                          for history, target in zip(result.events, episode.targets, strict=True)]
                assert rows[n][index]["learner_mse"] == pytest.approx(errors, rel=1e-5, abs=1e-6)
                if n == 2:
                    torch.testing.assert_close(tuple(h[-1].output for h in result.events), outputs[index])
                if n:
                    before = [float((history[-1].output - target).square().mean())
                              for history, target in zip(prior_direct[index].events, episode.targets, strict=True)]
                    assert rows[n][index]["without_new_observation_mse"] == pytest.approx(before[0])
                    assert rows[n][index]["observation_gain"] == pytest.approx(before[0] - errors[0], abs=1e-6)
                else:
                    assert "observation_gain" not in rows[n][index]
            prior_direct = direct


def test_early_absolute_loss_recipe_keeps_a_single_final_endpoint(tmp_path, monkeypatch):
    batches, receipts = [], []
    monkeypatch.setattr(training, "build_learners", lambda **kwargs: _tiny_learners())
    monkeypatch.setattr(training, "save_round", lambda *args, **kwargs: receipts.append(kwargs))

    def minibatch(learners, episodes, **kwargs):
        batches.append(episodes)
        return (), {}

    monkeypatch.setattr(training, "train_minibatch", minibatch)
    args = SimpleNamespace(seed=9, hidden_dim=8, branches=2, device="cpu", max_operations=1,
                           max_effect_operations=1, width=3, message_credit="trajectory-input-vjp",
                           route_credit="retained-vjp", candidate_exploration="gumbel", plasticity_composition="parallel-delta",
                           episode_batch_size=4, episode_execution="batched", rounds=4, resume=None,
                           output_dir=tmp_path / "run", numeric_backend="reference", query_backend="reference",
                           search_backend="reference", gradient_scale=None, episode_task="transfer-acquisition", max_replacements=4)
    training._train(args, 0)
    assert [len(batch[0].supports) - 6 for batch in batches] == [1, 2, 3, 1]
    for batch, receipt in zip(batches, receipts[1:], strict=True):
        assert len(batch) == 4 and len({len(e.supports) for e in batch}) == 1
        assert [e.updated_key for e in batch] == [0, 2, 0, 2]
        assert all(len(e.targets) == 2 for e in batch)
        meta = receipt["metrics"]["transfer"]
        assert meta["current_presentations"] >= 1
        assert meta["endpoint_cycle"] == [1, 1, 2, 3]
        assert "absolute" in meta["reward"]


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_transfer_evaluation_carries_only_support_and_reloads_queries(device, tmp_path, monkeypatch):
    learners = _learners(device)
    for learner in learners:
        learner.query.eval().requires_grad_(False)
    digests = tuple(_module_digest(learner.query) for learner in learners)
    source = _episode(device)
    first = replace(source, supports=(*source.supports * 3, source.supports[0] * 0.4), updated_key=0)
    second = replace(first, supports=(first.supports[0] * 0.9, *first.supports[1:]))
    calls = []

    def observe(*args, **kwargs):
        result = training.interact_many(*args, **kwargs)
        calls.append((kwargs["initial_bank_states"], result))
        return result

    monkeypatch.setattr(evaluation, "interact_many", observe)
    scope = captured_search_execution(horizon=8, record_effect_operands=True) if device == "cuda" else nullcontext()
    with scope as backend:
        rows, states, outputs = evaluation.evaluate_transfer(
            learners, (first, second), presentations=2, width=2, plasticity_composition="parallel-delta")
        if backend is not None:
            assert backend.completed == 56 and not backend.fallbacks
        for stage in (1, 2):
            for e in range(2):
                prior = calls[stage - 1][1][e]
                for i in range(2):
                    assert calls[stage][0][e][i] is prior.events[i][prior.service_index].before
                    assert calls[stage][0][e][i] is not prior.events[i][-1].after
        save_carry(tmp_path / "state", states)
        restored = load_carry(tmp_path / "state", states)
        probes = training.interact_many(learners, tuple(replace(e, supports=()) for e in (first, second)),
                                         initial_bank_states=restored, width=2, plasticity_participation="retained",
                                         plasticity_composition="parallel-delta")
        for result, values in zip(probes, outputs, strict=True):
            torch.testing.assert_close(tuple(h[-1].output for h in result.events), values, rtol=1e-6, atol=1e-7)
    assert [r["arm"] for r in rows[-1]] == ["related", "unrelated"]
    assert [r["presentations"] for r in rows[-1]] == [2, 2]
    assert digests == tuple(_module_digest(learner.query) for learner in learners)
    assert all(p.grad is None for learner in learners for p in learner.query.parameters())
