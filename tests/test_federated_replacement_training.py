import json
from types import SimpleNamespace

import pytest
import torch

from benchmarks import _federated_association_tasks as tasks
from benchmarks import train_federated_interacting_learners as training
from benchmarks import evaluate_federated_lifetime as evaluation
from benchmarks.train_federated_self_modifying_federation import make_association_episodes
from test_federated_interacting_learners import _tiny_learners


def test_shared_data_entrypoints_are_the_same_objects_and_keep_original_tensors():
    assert training.InteractionEpisode is tasks.InteractionEpisode
    assert training.make_episode is tasks.make_episode
    assert evaluation.make_lifetime is tasks.make_lifetime
    episode = tasks.make_episode(seed=52, hidden_dim=8, device="cpu")
    original = make_association_episodes(split="train", seed=52, count=1, hidden_dim=8,
                                         support_count=3, device="cpu")[0]
    torch.testing.assert_close(episode.supports, original.supports, rtol=0, atol=0)
    torch.testing.assert_close(episode.queries[0], original.query, rtol=0, atol=0)
    torch.testing.assert_close(episode.targets[0], original.target, rtol=0, atol=0)


@pytest.mark.parametrize("replacements", (1, 2, 3, 4))
@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_training_stream_uses_observed_history_and_only_final_targets(device, replacements):
    episode = tasks.make_replacement_episode(seed=20360960, hidden_dim=8, device=device,
                                             replacements=replacements)
    stages = tasks.make_lifetime(seed=20360960, hidden_dim=8, device=device,
                                replacements=replacements, split="train")
    assert len(episode.supports) == 3 + replacements
    torch.testing.assert_close(episode.supports, tuple(s for stage in stages for s in stage.supports), rtol=0, atol=0)
    torch.testing.assert_close(episode.queries, stages[-1].queries, rtol=0, atol=0)
    torch.testing.assert_close(episode.targets, stages[-1].targets, rtol=0, atol=0)
    changed = (replacements - 1) % 2
    assert not torch.equal(episode.targets[changed], stages[-2].targets[changed])
    assert torch.equal(episode.targets[1 - changed], stages[-2].targets[1 - changed])
    assert all(not value.requires_grad for value in (*episode.supports, *episode.queries, *episode.targets))


@pytest.mark.parametrize("target_task", ("replacement", "mixed-replacement", "transfer-acquisition", "transfer-paired", "transfer-stream"))
def test_replacement_fork_restores_model_and_adam_without_changing_architecture(tmp_path, monkeypatch, target_task):
    source = _tiny_learners()
    for learner in source:
        learner.optimizer.zero_grad(set_to_none=True)
        sum(p.square().mean() for p in learner.query.parameters()).backward()
        learner.optimizer.step()
    config = dict(seed=4, hidden_dim=3, branches=2, device="cpu", max_operations=1, max_effect_operations=1,
                  width=3, message_credit="input-vjp", route_credit="episode-beam", candidate_exploration="ranked")
    source_task = "association" if target_task == "replacement" else "replacement"
    saved_config = config if source_task == "association" else {**config, "episode_task": source_task, "max_replacements": 4}
    training.save_round(tmp_path / "source", source, step=55, config=saved_config, metrics={})
    loaded = _tiny_learners()
    monkeypatch.setattr(training, "build_learners", lambda **kwargs: loaded)
    args = SimpleNamespace(**config, episode_batch_size=4, episode_execution="batched", rounds=0,
                           resume=tmp_path / "source", output_dir=tmp_path / "fork", numeric_backend="reference",
                           query_backend="reference", search_backend="reference", gradient_scale=None,
                           episode_task=target_task, max_replacements=4)
    training._train(args, 0)
    receipt = json.loads((args.output_dir / "credit-fork.json").read_text())
    assert receipt["source_round"] == 55 and receipt["parameters_and_adam_preserved"]
    assert receipt["from"]["episode_task"] == source_task
    assert receipt["to"]["episode_task"] == target_task
    for a, b in zip(source, loaded, strict=True):
        torch.testing.assert_close(a.query.state_dict(), b.query.state_dict(), rtol=0, atol=0)
        torch.testing.assert_close(a.optimizer.state_dict(), b.optimizer.state_dict(), rtol=0, atol=0)


def test_training_schedule_only_passes_final_task_targets_and_preserves_batch_shapes(tmp_path, monkeypatch):
    observed = []
    monkeypatch.setattr(training, "build_learners", lambda **kwargs: _tiny_learners())
    monkeypatch.setattr(training, "save_round", lambda *args, **kwargs: None)

    def minibatch(learners, episodes, **kwargs):
        observed.append(episodes)
        return (), {}

    monkeypatch.setattr(training, "train_minibatch", minibatch)
    args = SimpleNamespace(seed=9, hidden_dim=8, branches=2, device="cpu", max_operations=1,
                           max_effect_operations=1, width=3, message_credit="trajectory-input-vjp",
                           route_credit="retained-vjp", candidate_exploration="gumbel", plasticity_composition="parallel-delta",
                           episode_batch_size=4, episode_execution="batched", rounds=4, resume=None,
                           output_dir=tmp_path / "run", numeric_backend="reference", query_backend="reference",
                           search_backend="reference", gradient_scale=None, episode_task="replacement", max_replacements=4)
    training._train(args, 0)
    assert [len(batch[0].supports) for batch in observed] == [5, 6, 7, 4]
    for batch in observed:
        assert len(batch) == 4 and len({len(episode.supports) for episode in batch}) == 1
        for episode in batch:
            assert isinstance(episode, tasks.InteractionEpisode) and len(episode.targets) == 2
            assert all(value.shape == (1, 1, 8) for value in (*episode.queries, *episode.targets))


def test_random_order_is_replayable_complemented_and_does_not_change_global_rng():
    before = torch.get_rng_state().clone()
    schedules = []
    for seed in range(12):
        a = tasks.sample_replacement_keys(seed=seed, replacements=4)
        b = tasks.sample_replacement_keys(seed=seed, replacements=4, complement=True)
        assert a == tasks.sample_replacement_keys(seed=seed, replacements=4)
        assert all(x + y == 2 for x, y in zip(a, b, strict=True))
        schedules.append(a)
    assert len(set(schedules)) > 2
    assert any(a == b for keys in schedules for a, b in zip(keys, keys[1:]))
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("device", ("cpu", "cuda"))
def test_repeated_same_key_uses_latest_observed_value_and_preserves_other_target(device):
    keys = (2, 2, 0, 0)
    stages = tasks.make_lifetime(seed=204, hidden_dim=8, device=device, replacements=4, replacement_keys=keys)
    current = [value[:, 1:].clone() for value in stages[0].supports]
    for previous, stage, key in zip(stages, stages[1:], keys):
        assert stage.updated_key == key
        torch.testing.assert_close(stage.supports[0][:, :1], stages[0].supports[key][:, :1], rtol=0, atol=0)
        current[key] = stage.supports[0][:, 1:]
        for i, (left, right) in enumerate(((0, 1), (1, 2))):
            torch.testing.assert_close(stage.targets[i], stage.queries[i] + (current[left] + current[right]) / 2**0.5,
                                       rtol=0, atol=0)
        assert torch.equal(stage.targets[1 - key // 2], previous.targets[1 - key // 2])
    episode = tasks.make_replacement_episode(seed=204, hidden_dim=8, device=device, replacements=4, replacement_keys=keys)
    assert episode.updated_key == keys[-1]
    torch.testing.assert_close(episode.targets, stages[-1].targets, rtol=0, atol=0)
    torch.testing.assert_close(episode.supports, tuple(s for stage in stages for s in stage.supports), rtol=0, atol=0)


@pytest.mark.parametrize("keys", ((0,), (0, 1), (0, True)))
def test_invalid_explicit_schedule_is_not_silently_reinterpreted(keys):
    with pytest.raises(ValueError, match="replacement keys"):
        tasks.make_lifetime(seed=2, hidden_dim=8, device="cpu", replacements=2, replacement_keys=keys)


def test_mixed_training_includes_acquisition_and_balances_each_event_position(tmp_path, monkeypatch):
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
                           episode_batch_size=4, episode_execution="batched", rounds=10, resume=None,
                           output_dir=tmp_path / "run", numeric_backend="reference", query_backend="reference",
                           search_backend="reference", gradient_scale=None, episode_task="mixed-replacement", max_replacements=4)
    training._train(args, 0)
    assert [len(batch[0].supports) - 3 for batch in batches] == [1, 2, 3, 4, 0] * 2
    for batch, receipt in zip(batches, receipts[1:], strict=True):
        schedules = receipt["metrics"]["replacement_schedules"]
        assert len(batch) == 4 and len({len(episode.supports) for episode in batch}) == 1
        assert not torch.equal(batch[0].supports[0], batch[1].supports[0])
        if not schedules[0]:
            assert all(episode.updated_key is None for episode in batch)
            continue
        for a, b in ((0, 1), (2, 3)):
            assert all(x + y == 2 for x, y in zip(schedules[a], schedules[b], strict=True))
        for episode, keys in zip(batch, schedules, strict=True):
            assert episode.updated_key == keys[-1]
            for support, key in zip(episode.supports[3:], keys, strict=True):
                torch.testing.assert_close(support[:, :1], episode.supports[key][:, :1], rtol=0, atol=0)
