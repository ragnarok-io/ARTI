from __future__ import annotations

import copy
import json
import random
import time

import pytest
import torch
from torch import nn

import arti
from arti import mechanisms


class TaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.features.weight.copy_(torch.eye(4))
        self.register_buffer("observations", torch.zeros(()))

    def forward(self, inputs, labels=None):
        self.observations.add_(1)
        prediction = self.features(inputs)
        return {"prediction": prediction, "loss": (prediction - labels).square().mean()}


def graph_layer():
    pattern = mechanisms.TensorViewPattern(min_rank=2, max_rank=2, allowed_axis_roles=("batch", "feature"))
    resources = tuple(
        mechanisms.TensorResource(
            mechanisms.TensorResourceSpec(name, pattern),
            mechanisms.TensorView.from_tensor(torch.zeros(1, 4), axis_names=("batch", "feature"), axis_roles=("batch", "feature")),
        )
        for name in ("inputs", "outputs")
    )
    graph = mechanisms.ProgramGraph(
        resources,
        (mechanisms.Connection("adapt", mechanisms.ResourcePort("inputs"), mechanisms.ResourcePort("outputs"), transfer=mechanisms.LearnableAffineTransfer(gain=1.0)),),
        programs={"task": ("adapt",)},
    )
    return arti.ARTILayer(graph=graph, graph_program_id="task", graph_input_resource_id="inputs", graph_output_resource_id="outputs", axis_names=("batch", "feature"), axis_roles=("batch", "feature"))


def attached():
    return arti.ARTI.attach(TaskModel(), graph_layer(), layers="features")


def data():
    return [{"inputs": torch.full((3, 4), value), "labels": torch.full((3, 4), 2 * value)} for value in (1.0, 2.0, 3.0)]


def policy(path, **kwargs):
    return arti.ARTITrainingRunConfig(output_dir=path, **kwargs)


def test_task_loss_is_the_default_and_check_is_read_only():
    assert arti.torch.ARTITrainingRunConfig is arti.ARTITrainingRunConfig
    assert arti.torch.ARTIEvaluationResult is arti.ARTIEvaluationResult
    model = attached()
    session = model.arti.trainer()
    before = copy.deepcopy(model.state_dict())
    rng = torch.get_rng_state()
    result = session.check(data()[0])
    assert session.config.objective == "model_loss"
    assert result["ok"] and result["connected_parameters"] > 0
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])
    assert session.plan()["estimates_are_static"]


def test_periodic_checkpoints_metrics_and_readable_run(tmp_path):
    model = attached()
    session = model.arti.trainer(learning_rate=0.02)
    events = []
    result = session.fit(data(), steps=5, run=policy(tmp_path, checkpoint_steps=2, log_steps=100, metadata={"dataset": "task-v1", "role": "train"}), reporter=events.append)
    assert result.checkpoint_path.exists()
    assert result.run_directory == tmp_path
    assert result.steps == 5
    assert len(result.loss_history) == 5
    report = json.loads((tmp_path / "run.json").read_text())
    assert report["status"] == "completed"
    assert report["data_resume"] == "indexed_sequence"
    assert report["run"]["metadata"]["dataset"] == "task-v1"
    assert [event["step"] for event in events if event["type"] == "checkpoint"] == [1, 2, 4, 5]
    assert len([event for event in events if event["type"] == "metrics"]) < 5
    restored = arti.ARTI.load(TaskModel(), result.checkpoint_path, layer=graph_layer())
    torch.testing.assert_close(model(data()[0]["inputs"], data()[0]["labels"])["prediction"], restored(data()[0]["inputs"], data()[0]["labels"])["prediction"])


def test_sequence_rng_and_optimizer_resume_matches_continuous(tmp_path):
    def stochastic_objective(model, batch):
        target = batch["labels"] + torch.rand_like(batch["labels"]) * random.random()
        return model(batch["inputs"], target)["loss"]

    continuous_model = attached()
    continuous = continuous_model.arti.trainer(objective=stochastic_objective)
    torch.manual_seed(21)
    random.seed(21)
    continuous.fit(data(), steps=7, run=policy(tmp_path / "continuous"))

    split_model = attached()
    split = split_model.arti.trainer(objective=stochastic_objective)
    torch.manual_seed(21)
    random.seed(21)
    result = split.fit(data(), steps=2, run=policy(tmp_path / "split"))
    restored = arti.ARTI.load(TaskModel(), result.checkpoint_path, layer=graph_layer())
    resumed = restored.arti.trainer(objective=stochastic_objective, resume_from_checkpoint=result.checkpoint_path)
    resumed.fit(data(), steps=5, run=policy(tmp_path / "resumed"))
    assert resumed.micro_batches_seen == 7
    assert resumed.loss_history == continuous.loss_history
    for expected, actual in zip(continuous_model.arti.parameters(), restored.arti.parameters(), strict=True):
        assert torch.equal(expected, actual)


def test_weighted_evaluation_restores_buffers_rng_and_individual_modes():
    model = attached()
    model.train()
    model.features.eval()
    session = model.arti.trainer()
    before = copy.deepcopy(model.state_dict())
    modes = [module.training for module in model.modules()]
    rng = torch.get_rng_state()
    result = session.evaluate(data()[:2], weight=lambda batch: float(batch["inputs"][0, 0]))
    assert result.loss == pytest.approx(3.0)
    assert result.weight == 3.0 and result.batches == 2
    assert modes == [module.training for module in model.modules()]
    assert torch.equal(rng, torch.get_rng_state())
    for name, value in model.state_dict().items():
        assert torch.equal(value, before[name])


def test_evaluation_is_written_to_run_without_committing_extra_observations(tmp_path):
    model = attached()
    session = model.arti.trainer()
    session.fit(data(), steps=4, run=policy(tmp_path, evaluation_steps=2), eval_data=data())
    report = json.loads((tmp_path / "run.json").read_text())
    assert [item["step"] for item in report["evaluations"]] == [2, 4]
    assert model.observations == 4


def test_evaluation_restores_graph_resource_backing_and_epochs():
    model = attached()
    graph = model.arti.layers["features"].graph
    before = graph.state()

    def objective(host, batch):
        graph.execute_program("task")
        return host(**batch)["loss"]

    model.arti.trainer(objective=objective).evaluate(data())
    after = graph.state()
    for expected, actual in zip(before.resources, after.resources, strict=True):
        assert expected.epoch == actual.epoch
        assert expected.step_index == actual.step_index
        assert torch.equal(expected.default_view.value, actual.default_view.value)
        assert torch.equal(expected.active_view.value, actual.active_view.value)


def test_orderly_stop_saves_latest_step(tmp_path):
    model = attached()
    session = model.arti.trainer()
    result = session.fit(data(), steps=20, run=policy(tmp_path), stop_requested=lambda: session.global_step >= 3)
    assert result.steps == 3 and result.stop_reason == "requested"
    assert result.checkpoint_path.exists()
    assert json.loads((tmp_path / "run.json").read_text())["global_step"] == 3


def test_time_limit_is_checked_at_optimizer_boundaries(tmp_path):
    def slow_loss(model, batch):
        time.sleep(0.03)
        return model(**batch)["loss"]

    model = attached()
    result = model.arti.trainer(objective=slow_loss).fit(data(), steps=50, run=policy(tmp_path, max_seconds=0.02))
    assert result.steps <= 1 and result.stop_reason == "time_budget"
    assert result.checkpoint_path.exists()


def test_abrupt_exception_keeps_the_last_completed_checkpoint(tmp_path):
    calls = 0

    def objective(model, batch):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt()
        return model(**batch)["loss"]

    model = attached()
    with pytest.raises(KeyboardInterrupt):
        model.arti.trainer(objective=objective).fit(data(), steps=5, run=policy(tmp_path, checkpoint_steps=2))
    report = json.loads((tmp_path / "run.json").read_text())
    assert report["status"] == "interrupted"
    assert (tmp_path / report["latest_checkpoint"]).exists()
    restored = arti.ARTI.load(TaskModel(), tmp_path / report["latest_checkpoint"], layer=graph_layer())
    session = restored.arti.trainer(resume_from_checkpoint=tmp_path / report["latest_checkpoint"])
    assert session.global_step == 2


def test_failing_remote_reporter_does_not_destroy_local_training(tmp_path):
    def reporter(_event):
        raise ConnectionError("offline")

    model = attached()
    with pytest.warns(RuntimeWarning, match="reporter disabled"):
        result = model.arti.trainer().fit(data(), steps=3, run=policy(tmp_path), reporter=reporter)
    assert result.steps == 3 and result.checkpoint_path.exists()
    assert json.loads((tmp_path / "run.json").read_text())["reporter_error"] == "ConnectionError"


def test_invalid_or_missing_evaluation_is_rejected_before_training(tmp_path):
    model = attached()
    session = model.arti.trainer()
    with pytest.raises(ValueError, match="explicit eval_data"):
        session.fit(data(), steps=2, run=policy(tmp_path, evaluation_steps=1))
    with pytest.raises(ValueError, match="empty"):
        session.evaluate([])
    with pytest.raises(ValueError, match="weights"):
        session.evaluate(data(), weight=lambda _: 0)


@pytest.mark.parametrize("options", [{"log_steps": 0}, {"checkpoint_steps": 0}, {"max_seconds": float("nan")}, {"checkpoint_seconds": float("inf")}, {"evaluation_steps": 0}])
def test_run_policy_values_are_validated(options):
    with pytest.raises(ValueError):
        arti.ARTITrainingRunConfig(**options)


def test_general_iterable_does_not_claim_exact_data_resume(tmp_path):
    model = attached()
    model.arti.trainer().fit(iter(data()), steps=2, run=policy(tmp_path))
    assert json.loads((tmp_path / "run.json").read_text())["data_resume"] == "weights_optimizer_rng_only"


def test_hub_export_preserves_complete_training_state(tmp_path):
    model = attached()
    session = model.arti.trainer()
    session.fit(data(), steps=2, run=policy(tmp_path / "run"))
    result = model.arti.save_pretrained(tmp_path / "export", base_model="example/task-base", training_session=session)
    loaded = arti.load(result.artifact_path, load_resources=False)
    assert loaded.training_state["micro_batches_seen"] == 2
    assert "rng" in loaded.training_state


def test_native_graph_assets_are_part_of_the_checkpoint_lock(tmp_path):
    result = attached().arti.trainer().fit(data(), steps=2, run=policy(tmp_path))
    loaded = arti.load(result.checkpoint_path, load_resources=False)
    graphs = loaded.manifest["architecture"]["native_program_graphs"]
    assert len(graphs) == 1
    row = graphs[0]
    assert row["path"] == "layers.0.runtime.graph"
    graph_file = result.checkpoint_path.parent / row["tensors"]["file"]
    assert graph_file.exists()
    graph_file.write_bytes(b"changed graph backing")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        arti.load(result.checkpoint_path)


def test_stateful_loader_cursor_is_restored(tmp_path):
    class Loader:
        def __init__(self):
            self.cursor = 0

        def __iter__(self):
            while True:
                item = data()[self.cursor % 3]
                self.cursor += 1
                yield item

        def state_dict(self):
            return {"cursor": self.cursor}

        def load_state_dict(self, state):
            self.cursor = state["cursor"]

    model = attached()
    first = model.arti.trainer().fit(Loader(), steps=2, run=policy(tmp_path / "first"))
    restored = arti.ARTI.load(TaskModel(), first.checkpoint_path, layer=graph_layer())
    resumed = restored.arti.trainer(resume_from_checkpoint=first.checkpoint_path)
    loader = Loader()
    resumed.fit(loader, steps=1, run=policy(tmp_path / "resumed"))
    assert loader.cursor == 3
    assert json.loads((tmp_path / "resumed/run.json").read_text())["data_resume"] == "loader_state"


def test_existing_run_is_not_overwritten_by_a_new_training_attempt(tmp_path):
    first = attached().arti.trainer().fit(data(), steps=2, run=policy(tmp_path))
    report = (tmp_path / "run.json").read_bytes()
    weights = first.checkpoint_path.read_bytes()
    model = attached()
    session = model.arti.trainer()
    with pytest.raises(FileExistsError, match="already contains a run"):
        session.fit(data(), steps=3, run=policy(tmp_path))
    assert session.global_step == 0
    assert (tmp_path / "run.json").read_bytes() == report
    assert first.checkpoint_path.read_bytes() == weights
