from contextlib import nullcontext
from dataclasses import replace
import json

import pytest
import torch

from benchmarks._federated_captured_search import captured_search_execution
from benchmarks._federated_episode_beam import search_episode_beam, search_episode_beams_many
from benchmarks.train_federated_interacting_learners import (
    interact, learner_objective, load_round, make_episode, save_round, train_minibatch, train_round,
)
from tests.test_federated_interacting_learners import _tiny_learners


def _state_close(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _state_close(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        for a, b in zip(left, right, strict=True):
            _state_close(a, b)
    elif isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-6)
    else:
        assert left == right


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("explore", [False, True])
def test_episode_beams_batch_preserves_ragged_history_and_independent_rng(device, explore):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    query = _tiny_learners()[0].query.to(device)
    episodes = [make_episode(seed=seed, hidden_dim=3, device=torch.device(device)) for seed in (13, 27, 44)]
    inputs = [tuple({"x": x} for x in episode.supports[:i + 1]) for i, episode in enumerate(episodes)]
    seeds = [17, 59, 77] if explore else [None] * 3
    with captured_search_execution(horizon=8):
        expected = [search_episode_beam(query, history, width=3, exploration_seed=seed)
                    for history, seed in zip(inputs, seeds, strict=True)]
    with captured_search_execution(horizon=8) as backend:
        actual = search_episode_beams_many(query, inputs, width=3, exploration_seeds=seeds)
        if device == "cuda":
            assert backend.batched_searches == 5
            assert not backend.fallbacks
    for left, right in zip(expected, actual, strict=True):
        assert left.event_searches == right.event_searches
        for a, b in zip(left.result.branches, right.result.branches, strict=True):
            assert a.route == b.route
            torch.testing.assert_close(a.execution.outputs, b.execution.outputs)
            torch.testing.assert_close(a.log_probability, b.log_probability)
            assert a.execution.bank_state.revisions == b.execution.bank_state.revisions


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_minibatch_serial_and_batched_gradients_adam_and_peer_ownership(device, tmp_path, monkeypatch):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    pairs = []
    for _ in range(2):
        torch.manual_seed(314)
        pairs.append(_tiny_learners())
    for learners in pairs:
        for learner in learners:
            learner.query.to(device)
    episodes = tuple(make_episode(seed=seed, hidden_dim=3, device=torch.device(device)) for seed in (3, 9))
    results = []
    for learners, batched in zip(pairs, (False, True), strict=True):
        steps = []
        for learner in learners:
            step = learner.optimizer.step

            def checked(*args, _step=step, **kwargs):
                assert all(any(p.grad is not None for p in item.query.parameters()) for item in learners)
                steps.append(1)
                return _step(*args, **kwargs)

            monkeypatch.setattr(learner.optimizer, "step", checked)
        with captured_search_execution(horizon=8) if device == "cuda" else nullcontext():
            results.append(train_minibatch(learners, episodes, width=3, batch_search=batched,
                                           message_credit="trajectory-input-vjp", exploration_seeds=(41, 91)))
        assert len(steps) == 2
    assert results[0][1]["total_executed_search_expansions"] == results[1][1]["total_executed_search_expansions"]
    for left, right in zip(*pairs, strict=True):
        _state_close(left.query.state_dict(), right.query.state_dict())
        _state_close(left.optimizer.state_dict(), right.optimizer.state_dict())
        for a, b in zip(left.query.parameters(), right.query.parameters(), strict=True):
            assert (a.grad is None) == (b.grad is None)
            if a.grad is not None:
                torch.testing.assert_close(a.grad, b.grad)
        assert all(owner.value.grad is None for owner in right.query.owner_states)
    save_round(tmp_path / "batch", pairs[1], step=1, config={"episode_batch_size": 2},
               metrics=results[1][1], interaction=results[1][0])
    assert len(list((tmp_path / "batch").glob("learner-*/episodes/*/events.json"))) == 4
    restored = _tiny_learners()
    for learner in restored:
        learner.query.to(device)
    saved = load_round(tmp_path / "batch", restored)
    assert saved["config"]["episode_batch_size"] == 2
    for left, right in zip(pairs[1], restored, strict=True):
        _state_close(left.optimizer.state_dict(), right.optimizer.state_dict())
        _state_close(left.query.state_dict(), right.query.state_dict())


def test_minibatch_size_one_keeps_original_optimizer_semantics():
    torch.manual_seed(271)
    left = _tiny_learners()
    torch.manual_seed(271)
    right = _tiny_learners()
    episode = make_episode(seed=67, hidden_dim=3, device=torch.device("cpu"))
    train_round(left, episode, width=3, message_credit="input-vjp", exploration_seed=97)
    train_minibatch(right, (episode,), width=3, message_credit="input-vjp", exploration_seeds=(97,))
    for a, b in zip(left, right, strict=True):
        _state_close(a.query.state_dict(), b.query.state_dict())
        _state_close(a.optimizer.state_dict(), b.optimizer.state_dict())


def test_batched_beam_targets_are_only_used_by_objective():
    learners = _tiny_learners()
    episode = make_episode(seed=53, hidden_dim=3, device=torch.device("cpu"))
    interaction = interact(learners, episode, width=3)
    inputs = tuple(e.inputs for e in interaction.events[0])
    beam = search_episode_beams_many(learners[0].query, (inputs,), width=3)[0]
    objective, _ = learner_objective(learners, episode, interaction, 0, width=3, beam=beam)
    other, _ = learner_objective(learners, replace(episode, targets=tuple(t + 1 for t in episode.targets)),
                                 interaction, 0, width=3, beam=beam)
    assert objective.item() != other.item()
    peer = tuple(p for p in learners[1].query.parameters() if p.requires_grad)
    assert all(g is None for g in torch.autograd.grad(objective, peer, allow_unused=True))


def test_batch_size_recipe_fork_preserves_assets_and_backend_is_not_recipe(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from benchmarks import train_federated_interacting_learners as harness

    source = tmp_path / "source"
    config = dict(seed=4, hidden_dim=3, branches=2, device="cpu", max_operations=1, max_effect_operations=1,
                  width=3, message_credit="input-vjp", route_credit="episode-beam", candidate_exploration="ranked")
    learners = _tiny_learners()
    save_round(source, learners, step=7, config=config, metrics={})
    monkeypatch.setattr(harness, "build_learners", lambda **kw: _tiny_learners())
    args = SimpleNamespace(**config, episode_batch_size=4, episode_execution="serial", rounds=0, resume=source,
                           output_dir=tmp_path / "fork", numeric_backend="reference", query_backend="reference",
                           search_backend="reference", gradient_scale=None, episode_task="association", max_replacements=4)
    harness._train(args, 0)
    assert args.gradient_scale == 1.0
    receipt = json.loads((args.output_dir / "credit-fork.json").read_text())
    assert receipt["from"]["episode_batch_size"] == 1 and receipt["to"]["episode_batch_size"] == 4
    assert receipt["parameters_and_adam_preserved"]
