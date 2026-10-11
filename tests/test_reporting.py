from __future__ import annotations

from dataclasses import dataclass
import json
import sys
import types

import pytest
import torch
from torch import nn

import arti
from arti.reporting import MLflowReporter, ReporterGroup, WandbReporter


class FakeRun:
    def __init__(self):
        self.summary = {}
        self.logs = []
        self.definitions = []
        self.artifacts = []

    def define_metric(self, *args, **kwargs):
        self.definitions.append((args, kwargs))

    def log(self, values, **kwargs):
        assert not kwargs
        self.logs.append(values)

    def log_artifact(self, value):
        self.artifacts.append(value)


class FakeClient:
    def __init__(self):
        self.tags = {}
        self.batches = []
        self.artifacts = []

    def log_batch(self, run_id, *, metrics, synchronous):
        assert run_id == "external-run" and synchronous is True
        self.batches.append(metrics)

    def set_tag(self, run_id, key, value):
        assert run_id == "external-run"
        self.tags[key] = value

    def log_artifacts(self, run_id, local_dir, artifact_path):
        self.artifacts.append((run_id, local_dir, artifact_path))


@dataclass
class FakeMetric:
    key: str
    value: float
    timestamp: int
    step: int


@pytest.fixture
def metric_sdk(monkeypatch):
    entities = types.ModuleType("mlflow.entities")
    entities.Metric = FakeMetric
    monkeypatch.setitem(sys.modules, "mlflow.entities", entities)


def event(kind, **values):
    return {"type": kind, "run_id": "local-run", "step": 7, "timestamp": 1.25, **values}


def test_reporters_are_public_without_importing_tracker_sdks():
    assert arti.WandbReporter is arti.torch.WandbReporter is WandbReporter
    assert arti.MLflowReporter is arti.torch.MLflowReporter is MLflowReporter
    assert arti.ReporterGroup is arti.torch.ReporterGroup is ReporterGroup


def test_group_keeps_other_sinks_and_isolates_event_mutation():
    calls = []
    received = []

    def failing(values):
        calls.append(values)
        values["metrics"]["loss"] = -10
        raise ConnectionError("offline")

    group = ReporterGroup(failing, received.append)
    original = event("evaluation", metrics={"loss": 2.0})
    with pytest.warns(RuntimeWarning, match="reporter 0 disabled"):
        group(original)
    group(original)
    assert len(calls) == 1 and len(received) == 2
    assert received[0]["metrics"]["loss"] == original["metrics"]["loss"] == 2.0
    assert group.errors == [{"index": 0, "reporter": "function", "error_type": "ConnectionError"}]


def test_wandb_uses_custom_axis_and_keeps_same_step_evaluations():
    run = FakeRun()
    reporter = WandbReporter(run)
    reporter(event("metrics", loss=2.0, mean_loss=3.0, optimizer_steps=2))
    reporter(event("evaluation", split="validation", metrics={"loss": 1.0, "accuracy": 0.75}))
    reporter(event("evaluation", split="test", loss=1.5, batches=1, weight=3))
    assert len(run.logs) == 3
    assert all(values["arti/optimizer_step"] == 7 for values in run.logs)
    assert run.definitions == [
        (("arti/optimizer_step",), {}),
        (("arti/*",), {"step_metric": "arti/optimizer_step"}),
    ]
    assert run.logs[1]["arti/validation/accuracy"] == 0.75
    assert run.logs[2]["arti/test/loss"] == 1.5


def test_continuation_clears_stale_workflow_summary():
    run = FakeRun()
    reporter = WandbReporter(run)
    reporter(event("workflow_finished", status="failed", stage="export", error_type="RuntimeError"))
    reporter(event("artifact", name="model", path="unused"))
    reporter(event("run_started", requested_steps=10, parent_run_id="previous"))
    assert run.summary["arti/workflow/status"] == "not_reported"
    assert run.summary["arti/workflow/error_type"] == ""
    assert run.summary["arti/model_exported"] is False


def test_tracking_does_not_send_paths_records_or_unselected_metadata(metric_sdk):
    run, client = FakeRun(), FakeClient()
    for reporter in (WandbReporter(run), MLflowReporter(client, "external-run")):
        reporter(
            event(
                "run_started",
                requested_steps=3,
                parent_run_id="parent",
                output_dir="private/local",
                metadata={"secret": "hidden"},
            )
        )
        reporter(
            event(
                "evaluation",
                metrics={
                    "loss": 2.0,
                    "path": "private/local",
                    "records": [{"answer": "hidden"}],
                    "invalid": float("nan"),
                },
            )
        )
        reporter(event("checkpoint", path="private/local/model.st"))
        reporter(event("artifact", path="private/local", name="model"))
    encoded = json.dumps(
        [
            run.logs,
            run.summary,
            client.tags,
            [metric.__dict__ for batch in client.batches for metric in batch],
        ]
    )
    assert "private/local" not in encoded and "hidden" not in encoded and "invalid" not in encoded
    assert run.artifacts == client.artifacts == []
    assert client.batches[0][0] == FakeMetric("arti/validation/loss", 2.0, 1250, 7)
    assert client.tags["arti/parent_run_id"] == "parent"


def test_inference_artifact_upload_requires_explicit_opt_in(tmp_path, monkeypatch):
    artifacts = []

    class Artifact:
        def __init__(self, name, *, type):
            self.name, self.type = name, type
            self.directories = []
            artifacts.append(self)

        def add_dir(self, path):
            self.directories.append(path)

    sdk = types.ModuleType("wandb")
    sdk.Artifact = Artifact
    monkeypatch.setitem(sys.modules, "wandb", sdk)
    run, client = FakeRun(), FakeClient()
    for reporter in (
        WandbReporter(run, upload_artifacts=True),
        MLflowReporter(client, "external-run", upload_artifacts=True),
    ):
        reporter(event("checkpoint", path=str(tmp_path / "checkpoints")))
        reporter(event("artifact", name="model", path=str(tmp_path)))
    assert len(run.artifacts) == 1 and artifacts[0].directories == [str(tmp_path)]
    assert client.artifacts == [("external-run", str(tmp_path), "arti/local-run/model")]


class TaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2, 1)

    def forward(self, inputs, labels):
        return {"loss": (self.linear(inputs) - labels).square().mean()}


def batches():
    return [{"inputs": torch.ones(2, 2), "labels": torch.ones(2, 1)}]


def test_training_lifecycle_resume_parent_and_local_sink_errors(tmp_path):
    events = []

    def failing(_event):
        raise ConnectionError("offline")

    group = ReporterGroup(failing, events.append)
    session = arti.trainer(TaskModel(), config=arti.ARTITrainingConfig(device="cpu"))
    with pytest.warns(RuntimeWarning, match="reporter 0 disabled"):
        first = session.fit(
            batches(),
            steps=2,
            run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "first"),
            reporter=group,
        )
    payload = json.loads((first.run_directory / "run.json").read_text())
    assert payload["reporter_errors"] == group.errors
    assert events[0]["type"] == "run_started"
    assert events[-1]["type"] == "run_finished" and events[-1]["status"] == "completed"
    restored = arti.trainer(
        TaskModel(), config=session.config, resume_from_checkpoint=first.checkpoint_path
    )
    continuation = []
    restored.fit(
        batches(),
        steps=1,
        run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "continued"),
        reporter=continuation.append,
    )
    assert continuation[0]["step"] == 2
    assert continuation[0]["parent_run_id"] == payload["run_id"]
    assert continuation[0]["run_id"] != payload["run_id"]
    assert continuation[-1]["step"] == 3


@pytest.mark.parametrize(
    "error,status", [(ValueError, "failed"), (KeyboardInterrupt, "interrupted")]
)
def test_failure_emits_terminal_status_and_keeps_completed_asset(tmp_path, error, status):
    calls = 0
    events = []

    def objective(model, batch):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise error()
        return model(**batch)["loss"]

    session = arti.trainer(
        TaskModel(), objective=objective, config=arti.ARTITrainingConfig(device="cpu")
    )
    with pytest.raises(error):
        session.fit(
            batches(),
            steps=3,
            run=arti.ARTITrainingRunConfig(output_dir=tmp_path),
            reporter=events.append,
        )
    payload = json.loads((tmp_path / "run.json").read_text())
    assert events[-1]["status"] == payload["status"] == status
    assert events[-1]["step"] == 1 and events[-1]["error_type"] == error.__name__
    assert (tmp_path / payload["latest_checkpoint"]).exists()


@pytest.mark.parametrize("value", [True, 1.5, -1])
def test_fit_rejects_fractional_or_boolean_steps_before_creating_run(tmp_path, value):
    session = arti.trainer(TaskModel(), config=arti.ARTITrainingConfig(device="cpu"))
    with pytest.raises(ValueError, match="nonnegative integer"):
        session.fit(
            batches(), steps=value, run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "run")
        )
    assert not (tmp_path / "run").exists()


@pytest.mark.parametrize("field", ["checkpoint_steps", "log_steps", "evaluation_steps"])
@pytest.mark.parametrize("value", [True, 1.5, -1])
def test_reporting_intervals_are_integer_optimizer_steps(field, value):
    with pytest.raises(ValueError, match="positive integer"):
        arti.ARTITrainingRunConfig(**{field: value})


@pytest.mark.parametrize("field", ["checkpoint_seconds", "max_seconds"])
@pytest.mark.parametrize("value", [True, "1", float("nan"), 0])
def test_time_intervals_are_numeric_not_truthy_flags(field, value):
    with pytest.raises(ValueError, match="finite and positive"):
        arti.ARTITrainingRunConfig(**{field: value})


def test_actual_mlflow_sqlite_client_metrics_and_artifacts(tmp_path):
    mlflow = pytest.importorskip("mlflow")
    pytest.importorskip("sqlalchemy")
    pytest.importorskip("alembic")
    client = mlflow.MlflowClient(tracking_uri=f"sqlite:///{(tmp_path / 'tracking.db').as_posix()}")
    experiment = client.create_experiment(
        "arti-local", artifact_location=(tmp_path / "artifacts").as_uri()
    )
    tracked = client.create_run(experiment)
    reporter = MLflowReporter(client, tracked.info.run_id, upload_artifacts=True)
    reporter(event("run_started", requested_steps=3))
    reporter(event("metrics", loss=2.5))
    reporter(event("evaluation", metrics={"loss": 1.5}))
    model = tmp_path / "model"
    model.mkdir()
    (model / "declaration.json").write_text('{"task":"classification"}')
    reporter(event("artifact", name="model", path=str(model)))
    reporter(event("workflow_finished", stage="completed", status="completed"))
    actual = client.get_run(tracked.info.run_id)
    assert actual.data.metrics["arti/train/loss"] == 2.5
    assert actual.data.tags["arti/workflow/status"] == "completed"
    assert actual.info.status == "RUNNING"
    history = client.get_metric_history(tracked.info.run_id, "arti/validation/loss")
    assert history[0].step == 7
    assert client.list_artifacts(tracked.info.run_id, "arti/local-run/model")[0].path.endswith(
        "declaration.json"
    )
    client.set_terminated(tracked.info.run_id)


def test_actual_wandb_offline_run(tmp_path, monkeypatch):
    wandb = pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    with wandb.init(
        project="arti-local",
        dir=str(tmp_path),
        settings=wandb.Settings(
            console="off", disable_git=True, save_code=False, x_disable_meta=True
        ),
    ) as run:
        reporter = WandbReporter(run, upload_artifacts=True)
        reporter(event("run_started", requested_steps=3))
        reporter(event("metrics", loss=2.5))
        reporter(event("evaluation", metrics={"loss": 1.5}))
        model = tmp_path / "model"
        model.mkdir()
        (model / "declaration.json").write_text('{"task":"classification"}')
        reporter(event("artifact", name="model", path=str(model)))
        reporter(event("workflow_finished", status="completed", stage="completed"))
        assert run.summary["arti/workflow/status"] == "completed"
        assert run.summary["arti/run_id"] == "local-run"
        run.log({"caller/final": 1.0})
