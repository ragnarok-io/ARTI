from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest
import torch

from benchmarks._federated_spatial_events import SpatialEventSpec
from benchmarks._federated_spatial_programs import SpatialProgramSpec
from benchmarks.probe_federated_spatial_replay_groups import compare_groups
from benchmarks._federated_spatial_diagnostics import diagnose_reuse
from benchmarks import train_federated_spatial_learning as spatial_training
from benchmarks.train_federated_spatial_learning import (
    SpatialExplorationPolicy, SpatialTrainingConfig, _search, checkpoint_exploration, derive_exploration_run,
    derive_replay_group_run, evaluate, event_batch,
    make_training_run, prepare_batch, training_batch, validation_batch,
)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    from accelerate.state import AcceleratorState
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    AcceleratorState._reset_state(reset_partial_state=True)
    torch.set_num_threads(1)
    yield request.param
    AcceleratorState._reset_state(reset_partial_state=True)


def config():
    return SpatialTrainingConfig(event=SpatialEventSpec(height=3, width=4, markers=3, reuse_questions=3),
        program=SpatialProgramSpec(latent_positions=2, hidden_dim=8, rank=4, banks=2),
        events=2, width=2, heads=3, cooperation=2, latent_slots=2, views=1, steps=12, replay_group=2)


@pytest.mark.parametrize("resume", [False, True])
def test_cli_restores_complete_nondefault_configuration(tmp_path, monkeypatch, resume):
    settings = replace(config(), seed=37, reader_seed=101, validation_events=14, evaluate_every=13)
    root = tmp_path / "source"
    checkpoint = root / "checkpoints" / "saved"
    checkpoint.mkdir(parents=True)
    (checkpoint / "manifest.json").write_text(json.dumps({"contract": {"config": asdict(settings)}}))
    (root / "latest.json").write_text(json.dumps({"checkpoint": "checkpoints/saved"}))
    expected = settings if resume else replace(settings, replay_group=8)
    def capture(config, *args, **kwargs):
        assert config == expected
        raise RuntimeError("launch captured")
    monkeypatch.setattr(spatial_training, "make_training_run", capture)
    monkeypatch.setattr(spatial_training, "derive_replay_group_run", capture)
    args = SimpleNamespace(target_steps=500, segment_seconds=300, seed=None, reuse_weight=1.,
        replay_group=None if resume else 8, validation_events=None, evaluate_every=None,
        exploration_scale=None, exploration_seed=None,
        resume=resume, derive_from=None if resume else checkpoint,
        output=root if resume else tmp_path / "derived", device="cpu")
    with pytest.raises(RuntimeError, match="launch captured"):
        spatial_training.main(args)


def test_replay_group_derivation_preserves_optimizer_rng_and_resumes(tmp_path, device):
    original = config()
    graph, source, progress = make_training_run(original, tmp_path / "source", device=device)
    batch = prepare_batch(original, graph, progress, capture=False)
    assert training_batch(original, source, progress, batch)["updated"]
    progress.pending_validation_step = None
    checkpoint = source.checkpoint(reason="source")
    pointer = (tmp_path / "source" / "latest.json").read_bytes()
    expected_random = torch.rand(4)
    changed = replace(original, replay_group=1)
    destination = tmp_path / "derived"
    derived_graph, derived, derived_progress = derive_replay_group_run(
        changed, destination, checkpoint, device=device)
    torch.testing.assert_close(torch.rand(4), expected_random, rtol=0, atol=0)
    torch.testing.assert_close(derived.trainable.state_dict(), source.trainable.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(derived.optimizer.state_dict(), source.optimizer.state_dict(), rtol=0, atol=0)
    assert derived_progress.state_dict() == progress.state_dict()
    assert derived.progress.step == source.progress.step == 1
    assert (tmp_path / "source" / "latest.json").read_bytes() == pointer
    lineage = json.loads((destination / "derivation.json").read_text())
    assert lineage["previous_replay_group"] == 2 and lineage["replay_group"] == 1
    derived_batch = prepare_batch(changed, derived_graph, derived_progress, capture=device == "cuda")
    expected = training_batch(changed, derived, derived_progress, derived_batch)
    next_model = {name: value.clone() for name, value in derived.trainable.state_dict().items()}
    restored_graph, restored, restored_progress = make_training_run(changed, destination, device=device)
    restored.resume()
    restored_batch = prepare_batch(changed, restored_graph, restored_progress, capture=device == "cuda")
    actual = training_batch(changed, restored, restored_progress, restored_batch)
    assert actual["updated"] and actual["step"] == expected["step"] == 2
    for key in ("panel_risk", "gradient_norm", "selected_answer_mse", "selected_reuse_mse"):
        assert actual[key] == pytest.approx(expected[key], rel=2e-5, abs=2e-6)
    torch.testing.assert_close(restored.trainable.state_dict(), next_model, rtol=2e-5, atol=2e-6)
    with pytest.raises(ValueError, match="other training settings"):
        derive_replay_group_run(replace(changed, learning_rate=.01), tmp_path / "invalid", checkpoint, device=device)


def test_exploration_derivation_resumes_and_validation_stays_deterministic(tmp_path, device):
    settings, policy = config(), SpatialExplorationPolicy(scale=1., seed=101)
    graph, source, progress = make_training_run(settings, tmp_path / "source", device=device)
    batch = prepare_batch(settings, graph, progress, capture=False)
    assert training_batch(settings, source, progress, batch)["updated"]
    progress.pending_validation_step = 1
    progress.validation_next_batch = 1
    progress.validation_answers = [.1, .2]
    checkpoint = source.checkpoint(reason="source")
    assert checkpoint_exploration(checkpoint) is None
    pointer = (tmp_path / "source" / "latest.json").read_bytes()
    random = torch.rand(4)
    derived_graph, run, restored = derive_exploration_run(settings, tmp_path / "derived", checkpoint,
                                                        exploration=policy, device=device)
    torch.testing.assert_close(torch.rand(4), random, rtol=0, atol=0)
    torch.testing.assert_close(run.trainable.state_dict(), source.trainable.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(run.optimizer.state_dict(), source.optimizer.state_dict(), rtol=0, atol=0)
    assert restored.state_dict() == progress.state_dict()
    assert run.progress.step == source.progress.step == 1
    assert (tmp_path / "source" / "latest.json").read_bytes() == pointer
    latest = tmp_path / "derived" / json.loads((tmp_path / "derived" / "latest.json").read_text())["checkpoint"]
    assert checkpoint_exploration(latest) == policy
    derived_batch = prepare_batch(settings, derived_graph, restored, capture=device == "cuda", exploration=policy)
    derived_batch.exploration(scale=policy.scale, seed=policy.batch_seed(1))
    deterministic = evaluate(settings, source, progress, batch, event_count=2)
    actual = evaluate(settings, run, restored, derived_batch, event_count=2)
    for key in ("per_event_answer", "per_event_reuse", "selected_write_occurrences"):
        assert actual[key] == pytest.approx(deterministic[key], rel=2e-5, abs=2e-6)
    assert torch.count_nonzero(derived_batch.selection_bias) == 0
    expected = training_batch(settings, run, restored, derived_batch)
    assert expected["exploration_seed"] == policy.batch_seed(1)
    assert torch.count_nonzero(derived_batch.selection_bias) > 0
    model = {name: value.clone() for name, value in run.trainable.state_dict().items()}
    new_graph, resumed, new_progress = make_training_run(settings, tmp_path / "derived", device=device,
                                                        exploration=checkpoint_exploration(latest))
    resumed.resume(latest)
    new_batch = prepare_batch(settings, new_graph, new_progress, capture=device == "cuda", exploration=policy)
    result = training_batch(settings, resumed, new_progress, new_batch)
    for key in ("exploration_seed", "panel_risk", "selected_answer_mse", "selected_reuse_mse", "gradient_norm"):
        assert result[key] == pytest.approx(expected[key], rel=2e-5, abs=2e-6)
    torch.testing.assert_close(resumed.trainable.state_dict(), model, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(resumed.optimizer.state_dict(), run.optimizer.state_dict(), rtol=2e-5, atol=2e-6)
    assert policy.batch_seed(1) != policy.batch_seed(2)
    with pytest.raises(ValueError, match="other training settings"):
        derive_exploration_run(replace(settings, learning_rate=.01), tmp_path / "invalid", checkpoint,
                               exploration=policy, device=device)


def test_replay_group_probe_preserves_trained_state_and_event_mean(tmp_path, device):
    settings = config()
    graph, run, progress = make_training_run(settings, tmp_path, device=device)
    batch = prepare_batch(settings, graph, progress, capture=device == "cuda")
    assert training_batch(settings, run, progress, batch)["updated"]
    model = {name: value.clone() for name, value in run.trainable.state_dict().items()}
    events = event_batch(settings, progress.next_batch_index, device=device)
    decoded, _ = _search(batch, events, decisions=True)
    report = compare_groups(settings, run, batch, events, decoded, groups=(1, 2))
    assert report["optimizer_steps"] == 0 and report["slow_parameters_unchanged"]
    assert report["gradient_leaf_count"] > 0 and report["events"] == settings.events
    assert run.progress.step == progress.next_batch_index == 1
    assert all(parameter.grad is None for parameter in run.trainable.parameters())
    torch.testing.assert_close(run.trainable.state_dict(), model, rtol=0, atol=0)


def test_reuse_diagnostic_preserves_parameters_and_grad_buffers(tmp_path, device):
    settings = config()
    graph, run, progress = make_training_run(settings, tmp_path, device=device)
    batch = prepare_batch(settings, graph, progress, capture=device == "cuda")
    events = event_batch(settings, 0, device=device)
    decoded, _ = _search(batch, events, decisions=True)
    model = {name: value.clone() for name, value in run.trainable.state_dict().items()}
    for value in run.trainable.parameters():
        value.grad = torch.ones_like(value)
    report = diagnose_reuse(settings, run, batch, events, decoded)
    assert report["optimizer_steps"] == 0 and run.progress.step == 0
    assert sum(row["probability"] for row in report["endpoints"]) == pytest.approx(1.)
    assert len(report["endpoints"]) == len(decoded[0][1])
    assert abs(sum(row["choice_coefficient"] for row in report["endpoints"])) < 1e-12
    assert all(torch.isfinite(torch.tensor(row["reuse_loss"])) for row in report["endpoints"])
    for value in run.trainable.parameters():
        torch.testing.assert_close(value.grad, torch.ones_like(value), rtol=0, atol=0)
    torch.testing.assert_close(run.trainable.state_dict(), model, rtol=0, atol=0)


def test_resume_preserves_next_event_optimizer_and_captured_refresh(tmp_path, device):
    settings = config()
    graph, run, progress = make_training_run(settings, tmp_path, device=device)
    batch = prepare_batch(settings, graph, progress, capture=device == "cuda")
    initial = graph.query.initial_bank_state()
    first = training_batch(settings, run, progress, batch)
    assert first["updated"] and progress.next_batch_index == 1
    checkpoint = run.checkpoint(reason="test")
    expected_events = event_batch(settings, progress.next_batch_index, device=device)
    expected = training_batch(settings, run, progress, batch)
    torch.testing.assert_close(initial.values, graph.query.initial_bank_state().values, rtol=0, atol=0)
    expected_model = {name: value.clone() for name, value in run.trainable.state_dict().items()}
    expected_optimizer = run.optimizer.state_dict()

    restored_graph, restored, restored_progress = make_training_run(settings, tmp_path, device=device)
    restored.resume(checkpoint)
    next_events = event_batch(settings, restored_progress.next_batch_index, device=device)
    torch.testing.assert_close(next_events.x0, expected_events.x0, rtol=0, atol=0)
    torch.testing.assert_close(next_events.question, expected_events.question, rtol=0, atol=0)
    restored_batch = prepare_batch(settings, restored_graph, restored_progress, capture=device == "cuda")
    actual = training_batch(settings, restored, restored_progress, restored_batch)
    assert actual["updated"] and restored.progress.step == 2
    assert restored_progress.next_batch_index == progress.next_batch_index == 2
    assert restored.progress.examples == run.progress.examples == 4
    for key in ("panel_risk", "selected_answer_mse", "selected_reuse_mse", "gradient_norm"):
        assert actual[key] == pytest.approx(expected[key], rel=2e-5, abs=2e-6)
    assert actual["selected_return"] == expected["selected_return"]
    torch.testing.assert_close(restored.trainable.state_dict(), expected_model, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(restored.optimizer.state_dict(), expected_optimizer, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(restored_graph.query.initial_bank_state().values, initial.values, rtol=0, atol=0)


def test_evaluation_reads_return_without_training_or_installing_it(tmp_path, device):
    settings = config()
    graph, run, progress = make_training_run(settings, tmp_path, device=device)
    batch = prepare_batch(settings, graph, progress, capture=device == "cuda")
    row = training_batch(settings, run, progress, batch)
    assert row["updated"]
    before = {name: value.clone() for name, value in run.trainable.state_dict().items()}
    result = evaluate(settings, run, progress, batch, event_count=2)
    assert result["slow_parameter_updates"] == 0 and result["events"] == 2
    assert result["step"] == run.progress.step == 1 and progress.next_batch_index == 1
    torch.testing.assert_close(run.trainable.state_dict(), before, rtol=0, atol=0)
    assert all(not value.requires_grad and torch.count_nonzero(value) == 0
               for value in graph.query.initial_bank_state().values)
    assert all(parameter.grad is None for parameter in run.trainable.parameters())


def test_arms_start_with_the_same_full_model_and_event_stream(tmp_path, device):
    settings = config()
    _, answer, pa = make_training_run(replace(settings, reuse_weight=0.), tmp_path / "answer", device=device)
    _, reuse, pr = make_training_run(settings, tmp_path / "reuse", device=device)
    torch.testing.assert_close(answer.trainable.state_dict(), reuse.trainable.state_dict(), rtol=0, atol=0)
    torch.testing.assert_close(event_batch(settings, pa.next_batch_index, device=device).x0,
                               event_batch(settings, pr.next_batch_index, device=device).x0, rtol=0, atol=0)
    parameter_ids = {id(p) for p in answer.trainable.parameters()}
    optimizer_ids = {id(p) for group in answer.optimizer.param_groups for p in group["params"]}
    assert parameter_ids == optimizer_ids
    assert {id(p) for p in answer.trainable.reader.parameters()} <= optimizer_ids
    fast_buffers = {id(value) for value in answer.trainable.query.buffers()}
    assert not fast_buffers.intersection(optimizer_ids)


def test_initial_evaluation_resumes_without_repeating_completed_batches(tmp_path, device):
    settings = replace(config(), validation_events=4)
    graph, run, progress = make_training_run(settings, tmp_path, device=device)
    batch = prepare_batch(settings, graph, progress, capture=False)
    first_checkpoint = run.checkpoint(reason="initial")
    first = validation_batch(settings, run, progress, batch)
    assert first["kind"] == "validation_progress" and progress.validation_next_batch == 1
    second_checkpoint = run.checkpoint(reason="partial-validation")
    assert first_checkpoint != second_checkpoint
    expected = validation_batch(settings, run, progress, batch)
    other_graph, restored, other_progress = make_training_run(settings, tmp_path, device=device)
    restored.resume(second_checkpoint)
    assert other_progress.pending_validation_step == 0 and other_progress.validation_next_batch == 1
    other_batch = prepare_batch(settings, other_graph, other_progress, capture=False)
    actual = validation_batch(settings, restored, other_progress, other_batch)
    assert actual["kind"] == "validation" and actual["events"] == 4
    assert actual["per_event_answer"] == pytest.approx(expected["per_event_answer"], rel=2e-5, abs=2e-6)
    assert actual["per_event_reuse"] == pytest.approx(expected["per_event_reuse"], rel=2e-5, abs=2e-6)
    assert other_progress.pending_validation_step is None and other_progress.last_validation_step == 0
    assert restored.progress.step == other_progress.next_batch_index == 0
