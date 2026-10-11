"""Optional experiment trackers for the shared training/workflow event stream."""

from __future__ import annotations

import copy
import math
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Mapping


RunReporter = Callable[[Mapping[str, Any]], None]


class ReporterGroup:
    """Send events to independent sinks; disable only the sink that fails."""

    def __init__(self, *reporters: RunReporter) -> None:
        self.reporters = tuple(reporters)
        self.errors: list[dict[str, Any]] = []
        self._disabled: set[int] = set()

    def __call__(self, event: Mapping[str, Any]) -> None:
        for index, reporter in enumerate(self.reporters):
            if index in self._disabled:
                continue
            try:
                reporter(copy.deepcopy(dict(event)))
            except Exception as error:
                self._disabled.add(index)
                self.errors.append(
                    {
                        "index": index,
                        "reporter": type(reporter).__name__,
                        "error_type": type(error).__name__,
                    }
                )
                warnings.warn(
                    f"training reporter {index} disabled after {type(error).__name__}",
                    RuntimeWarning,
                    stacklevel=2,
                )


def _prefix(value: str) -> str:
    value = value.strip("/")
    if not value or "*" in value:
        raise ValueError("reporting prefix must be nonempty and contain no wildcard")
    return value


def _numeric_values(values: Mapping[str, Any], prefix: str) -> dict[str, float]:
    result = {}
    for name, value in values.items():
        key = f"{prefix}/{name}"
        if isinstance(value, Mapping):
            result.update(_numeric_values(value, key))
        elif (
            isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
        ):
            result[key] = float(value)
    return result


def _metrics(event: Mapping[str, Any], prefix: str) -> dict[str, float]:
    kind = event["type"]
    if kind == "metrics":
        values = {
            name: event[name] for name in ("loss", "mean_loss", "optimizer_steps") if name in event
        }
        return _numeric_values(values, f"{prefix}/train")
    if kind == "evaluation":
        values = event.get("metrics")
        if values is None:
            values = {name: event[name] for name in ("loss", "weight", "batches") if name in event}
        return _numeric_values(values, f"{prefix}/{event.get('split', 'validation')}")
    if kind == "run_finished":
        return _numeric_values(
            {"elapsed_seconds": event.get("elapsed_seconds")}, f"{prefix}/training"
        )
    if kind == "selection":
        return _numeric_values(
            {
                name: event.get(name)
                for name in (
                    "best_value",
                    "best_step",
                    "selected_step",
                    "evaluations",
                    "bad_evaluations",
                )
            },
            f"{prefix}/selection",
        )
    return {}


def _summary(event: Mapping[str, Any], prefix: str) -> dict[str, Any]:
    kind = event["type"]
    if kind == "run_started":
        values = {
            "run_id": event["run_id"],
            "parent_run_id": event.get("parent_run_id") or "",
            "training/status": "running",
            "training/error_type": "",
            "training/stop_reason": "",
            "training/initial_step": event["step"],
            "training/requested_steps": event["requested_steps"],
            "workflow/status": "not_reported",
            "workflow/stage": "",
            "workflow/error_type": "",
            "model_exported": False,
            "selection/monitor": "",
            "selection/best_step": "",
            "selection/best_value": "",
            "selection/selected_step": "",
            "selection/selection": "",
            "selection/reason": "",
            "selection/evaluations": "",
            "selection/bad_evaluations": "",
        }
    elif kind == "run_finished":
        values = {"training/status": event["status"], "training/final_step": event["step"]}
        values.update(
            {
                f"training/{name}": event[name]
                for name in ("stop_reason", "error_type")
                if name in event
            }
        )
    elif kind == "workflow_stage":
        values = {"workflow/status": "running", "workflow/stage": event["stage"]}
    elif kind == "workflow_finished":
        values = {"workflow/status": event["status"], "workflow/stage": event["stage"]}
        if "error_type" in event:
            values["workflow/error_type"] = event["error_type"]
    elif kind == "checkpoint":
        values = {"checkpoint_step": event["step"]}
    elif kind == "artifact":
        values = {"model_exported": True}
    elif kind == "selection":
        values = {
            f"selection/{name}": event[name]
            for name in (
                "selection",
                "reason",
                "selected_step",
                "monitor",
                "mode",
                "best_step",
                "best_value",
                "evaluations",
                "bad_evaluations",
            )
            if name in event and event[name] is not None
        }
    else:
        values = {}
    return {f"{prefix}/{name}": value for name, value in values.items()}


class WandbReporter:
    """Report to a caller-owned W&B Run; never initialize or finish the Run.

    The optimizer step is a custom chart axis, not W&B's monotonic history step.
    Only numeric metrics and selected lifecycle fields are sent by default.
    ``upload_artifacts=True`` additionally sends completed inference bundles.
    """

    def __init__(self, run: Any, *, prefix: str = "arti", upload_artifacts: bool = False) -> None:
        self.run = run
        self.prefix = _prefix(prefix)
        self.upload_artifacts = upload_artifacts
        self._defined = False

    def __call__(self, event: Mapping[str, Any]) -> None:
        metrics = _metrics(event, self.prefix)
        if metrics:
            axis = f"{self.prefix}/optimizer_step"
            if not self._defined:
                self.run.define_metric(axis)
                self.run.define_metric(f"{self.prefix}/*", step_metric=axis)
                self._defined = True
            self.run.log({axis: event["step"], **metrics})
        summary = _summary(event, self.prefix)
        if summary:
            self.run.summary.update(summary)
        if event["type"] == "artifact" and self.upload_artifacts:
            import wandb

            artifact = wandb.Artifact(f"arti-{event['run_id']}-{event['name']}", type="model")
            artifact.add_dir(str(Path(event["path"])))
            self.run.log_artifact(artifact)


class MLflowReporter:
    """Report to an explicit MLflow client/run ID without global run mutation.

    Metric records use MLflow's batched API at reporting boundaries. The caller
    owns run creation/termination and the SDK's connection/timeout settings.
    """

    def __init__(
        self, client: Any, run_id: str, *, prefix: str = "arti", upload_artifacts: bool = False
    ) -> None:
        self.client = client
        self.run_id = run_id
        self.prefix = _prefix(prefix)
        self.upload_artifacts = upload_artifacts

    def __call__(self, event: Mapping[str, Any]) -> None:
        metrics = _metrics(event, self.prefix)
        if metrics:
            from mlflow.entities import Metric

            timestamp = int(event.get("timestamp", time.time()) * 1000)
            self.client.log_batch(
                self.run_id,
                metrics=[
                    Metric(key, value, timestamp, event["step"]) for key, value in metrics.items()
                ],
                synchronous=True,
            )
        for key, value in _summary(event, self.prefix).items():
            self.client.set_tag(self.run_id, key, str(value))
        if event["type"] == "artifact" and self.upload_artifacts:
            self.client.log_artifacts(
                self.run_id,
                str(Path(event["path"])),
                artifact_path=f"{self.prefix}/{event['run_id']}/{event['name']}",
            )


__all__ = ["MLflowReporter", "ReporterGroup", "RunReporter", "WandbReporter"]
