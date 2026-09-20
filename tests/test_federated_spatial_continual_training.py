from dataclasses import replace
import json

import pytest
import torch

from benchmarks import _federated_spatial_continual_training as continual
from benchmarks._federated_episode_risk import replay_cooperative_episode_panel
from benchmarks._federated_spatial_episode_search import collect_spatial_episode_panel
from benchmarks._federated_spatial_events import SpatialEventSpec
from benchmarks._federated_spatial_programs import SpatialProgramSpec
from benchmarks.train_federated_spatial_learning import SpatialTrainingConfig, SpatialExplorationPolicy, make_training_run


@pytest.fixture(autouse=True)
def cpu_accelerator():
    from accelerate.state import AcceleratorState
    AcceleratorState._reset_state(reset_partial_state=True)
    torch.set_num_threads(1)
    yield
    AcceleratorState._reset_state(reset_partial_state=True)


def config():
    return SpatialTrainingConfig(event=SpatialEventSpec(3, 4, 3, reuse_questions=3),
        program=SpatialProgramSpec(latent_positions=2, hidden_dim=8, rank=4, banks=2),
        events=2, width=2, heads=3, cooperation=2, latent_slots=2, views=1, steps=12, replay_group=2)


def test_derivation_and_fresh_resume_preserve_assets_but_reset_task_stream(tmp_path):
    settings, policy = config(), continual.ContinuousSpatialPolicy(length=2, panel_width=2)
    exploration = SpatialExplorationPolicy()
    _, source, old = make_training_run(settings, tmp_path / "source", device="cpu", exploration=exploration)
    # Populate a real optimizer checkpoint, without spending a task training run.
    for parameter in source.trainable.parameters():
        if parameter.requires_grad:
            parameter.grad = torch.full_like(parameter, .01)
    source.step_accumulated_gradients()
    source.record_step(examples=2, tokens=0)
    old.next_batch_index = 9
    old.pending_validation_step = 1
    old.validation_answers = [0.25]
    old.validation_next_batch = 1
    old.training_seconds, old.evaluation_seconds = 125., 30.
    checkpoint = source.checkpoint(reason="source-test")
    pointer = (tmp_path / "source" / "latest.json").read_bytes()
    expected_random = torch.rand(4)
    actual_config, graph, run, progress = continual.derive_continuous_run(checkpoint, tmp_path / "derived",
        policy=policy, device="cpu")
    assert actual_config == settings
    torch.testing.assert_close(torch.rand(4), expected_random, rtol=0, atol=0)
    torch.testing.assert_close(run.trainable.state_dict(), source.trainable.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(run.optimizer.state_dict(), source.optimizer.state_dict(), rtol=0, atol=0)
    assert run.progress.step == source.progress.step == 1
    assert run.progress.examples == source.progress.examples == 2
    assert progress.next_batch_index == 0 and progress.pending_validation_step == 1
    assert progress.validation_next_batch == 0 and progress.validation_answers == []
    assert progress.training_seconds == 125. and progress.evaluation_seconds == 30.
    lineage = json.loads((tmp_path / "derived" / "derivation.json").read_text())
    assert lineage["source_task_progress"]["next_batch_index"] == 9
    assert (tmp_path / "source" / "latest.json").read_bytes() == pointer
    _, restored, restored_progress = continual.make_continuous_run(settings, policy, tmp_path / "derived",
        origin=run.config.contract["origin"], device="cpu", exploration=exploration)
    restored.resume()
    assert restored_progress.state_dict() == progress.state_dict()
    torch.testing.assert_close(restored.trainable.state_dict(), run.trainable.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(restored.optimizer.state_dict(), run.optimizer.state_dict(), rtol=0, atol=0)
    fast_ids = {id(value) for value in graph.query.initial_bank_state().values}
    optimizer_ids = {id(value) for group in run.optimizer.param_groups for value in group["params"]}
    assert not fast_ids.intersection(optimizer_ids)


def test_training_matches_joint_final_risk_and_updates_once(tmp_path, monkeypatch):
    settings, policy = config(), continual.ContinuousSpatialPolicy(length=2, panel_width=2)
    graph, run, progress = continual.make_continuous_run(settings, policy, tmp_path,
        origin={"source_step": 0}, device="cpu")
    batch = continual.prepare_continuous_batch(settings, policy, graph, progress, capture=False)
    episode = continual.episode_batch(settings, policy, 0, device="cpu")
    panel = collect_spatial_episode_panel(batch, episode, panel_width=2)
    leaves = tuple(p for p in run.trainable.parameters() if p.requires_grad)
    initial = graph.query.initial_bank_state()
    risks = []
    final = episode.events[-1]
    for row, paths in enumerate(panel.paths):
        replay = replay_cooperative_episode_panel(graph.query,
            tuple(event.writing_inputs(row) for event in episode.events), tuple(path.steps for path in paths))
        losses = torch.stack([(item.outputs["answer"] - final.answer[row:row+1]).square().mean() +
            (run.trainable.reader(item.bank_state, final.readonly_questions[row]) - final.readonly_answers[row]).square().mean()
            for item in replay.terminals])
        risks.append((torch.softmax(replay.energies.double(), 0) * losses.double()).sum())
    objective = torch.stack(risks).mean()
    expected = torch.autograd.grad(objective, leaves, allow_unused=True)
    monkeypatch.setattr(continual, "collect_spatial_episode_panel", lambda *args, **kwargs: panel)
    original_step, calls = run.step_accumulated_gradients, []
    def step():
        for parameter, gradient in zip(leaves, expected, strict=True):
            if gradient is None:
                assert parameter.grad is None
            else:
                torch.testing.assert_close(parameter.grad, gradient, rtol=2e-5, atol=2e-6)
        calls.append(1)
        return original_step()
    monkeypatch.setattr(run, "step_accumulated_gradients", step)
    result = continual.continuous_training_batch(settings, policy, run, progress, batch)
    assert result["panel_risk"] == pytest.approx(float(objective.detach()), rel=2e-5, abs=2e-6)
    assert result["updated"] and len(calls) == 1 and run.progress.step == 1
    assert result["episodes"] == 2 and result["external_scenes"] == run.progress.examples == 4
    assert progress.next_batch_index == 1
    torch.testing.assert_close(graph.query.initial_bank_state().values, initial.values, rtol=0, atol=0)


def test_online_evaluation_matches_committed_history_without_backward(tmp_path):
    settings, policy = config(), continual.ContinuousSpatialPolicy(length=2, panel_width=2)
    graph, run, progress = continual.make_continuous_run(settings, policy, tmp_path,
        origin={"source_step": 0}, device="cpu")
    batch = continual.prepare_continuous_batch(settings, policy, graph, progress, capture=False)
    episode = continual.episode_batch(settings, policy, 0, device="cpu", split="validation")
    panel = collect_spatial_episode_panel(batch, episode, panel_width=1)
    before = {name: value.clone() for name, value in run.trainable.state_dict().items()}
    batch.exploration(scale=1., seed=37)
    result, states = continual.evaluate_continuous_episode(run, batch, episode)
    assert torch.count_nonzero(batch.selection_bias) == 0
    with torch.no_grad():
        for row, paths in enumerate(panel.paths):
            replay = replay_cooperative_episode_panel(graph.query,
                tuple(event.writing_inputs(row) for event in episode.events), tuple(path.steps for path in paths))
            actual = replay.terminals[0]
            torch.testing.assert_close(states[row].values, actual.bank_state.values)
            assert states[row].revisions == actual.bank_state.revisions
            expected = (actual.outputs["answer"] - episode.events[-1].answer[row:row+1]).square().mean()
            assert result["per_episode_answer"][row] == pytest.approx(float(expected), rel=2e-5, abs=2e-6)
    torch.testing.assert_close(run.trainable.state_dict(), before, rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in run.trainable.parameters())
    assert all(not v.requires_grad for state in states for v in state.values)
    assert progress.next_batch_index == run.progress.step == 0
    assert result["slow_parameter_updates"] == 0


def test_online_validation_resumes_partial_panel_at_same_step(tmp_path):
    settings = replace(config(), validation_events=4)
    policy = continual.ContinuousSpatialPolicy(length=2, panel_width=2)
    origin = {"source_step": 0}
    graph, run, progress = continual.make_continuous_run(settings, policy, tmp_path, origin=origin, device="cpu")
    batch = continual.prepare_continuous_batch(settings, policy, graph, progress, capture=False)
    first = continual.continuous_validation_batch(settings, policy, run, progress, batch)
    assert first["kind"] == "continuous_validation_progress" and first["completed_episodes"] == 2
    assert progress.pending_validation_step == 0 and progress.validation_next_batch == 1
    checkpoint = run.checkpoint(reason="partial-validation")
    expected = continual.continuous_validation_batch(settings, policy, run, progress, batch)
    next_graph, restored, saved = continual.make_continuous_run(settings, policy, tmp_path,
        origin=origin, device="cpu")
    restored.resume(checkpoint)
    assert saved.pending_validation_step == 0 and saved.validation_next_batch == 1
    resumed_batch = continual.prepare_continuous_batch(settings, policy, next_graph, saved, capture=False)
    actual = continual.continuous_validation_batch(settings, policy, restored, saved, resumed_batch)
    for key in ("per_episode_answer", "per_episode_reuse", "changed_bank_slots"):
        assert actual[key] == expected[key]
    assert actual["kind"] == "continuous_validation" and actual["episodes"] == 4
    assert actual["external_scenes"] == 8
    assert saved.pending_validation_step is None and saved.validation_next_batch == 0
    assert saved.validation_answers == [] and saved.last_validation_step == 0
    assert restored.progress.step == saved.next_batch_index == 0
