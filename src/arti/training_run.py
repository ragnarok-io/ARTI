"""Local training-run policy and reporting, outside neural execution graphs."""

from __future__ import annotations

import json
import math
import time
import uuid
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from .serialization import _atomic_json
from .reporting import ReporterGroup, RunReporter
from .training_validation import ARTIValidationConfig


@dataclass(frozen=True)
class ARTITrainingRunConfig:
    output_dir: str | Path | None = None
    checkpoint_steps: int = 50
    checkpoint_seconds: float = 60.0
    log_steps: int = 10
    max_seconds: float | None = None
    evaluation_steps: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    validation: ARTIValidationConfig | None = None

    def __post_init__(self) -> None:
        if isinstance(self.validation, Mapping):
            object.__setattr__(self, "validation", ARTIValidationConfig(**self.validation))
        if self.validation is not None and not isinstance(self.validation, ARTIValidationConfig):
            raise TypeError("run.validation must be an ARTIValidationConfig or mapping")
        for name in ("checkpoint_steps", "log_steps", "evaluation_steps"):
            value = getattr(self, name)
            if name == "evaluation_steps" and value is None:
                continue
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("checkpoint_seconds", "max_seconds"):
            value = getattr(self, name)
            if name == "max_seconds" and value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        json.dumps(dict(self.metadata), allow_nan=False)
        if self.validation is not None and self.evaluation_steps is None:
            raise ValueError("run.validation requires explicit evaluation_steps")

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["output_dir"] = None if self.output_dir is None else str(self.output_dir)
        return values


def _prepare_run_directory(config: ARTITrainingRunConfig) -> Path:
    directory = Path(config.output_dir or Path(".arti-runs") / uuid.uuid4().hex)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "run.json").exists():
        raise FileExistsError("output_dir already contains a run; resume the checkpoint into a new run directory")
    return directory


class _TrainingRun:
    def __init__(
        self,
        config: ARTITrainingRunConfig,
        plan: Mapping[str, Any],
        reporter: RunReporter | None,
        *,
        parent_run_id: str | None = None,
    ) -> None:
        self.config = config
        self.directory = _prepare_run_directory(config)
        self.started = time.monotonic()
        self.last_saved = self.started
        self.reporter = reporter
        self.latest: Path | None = None
        self.payload: dict[str, Any] = {
            "format": "arti.training-run",
            "format_version": 1,
            "run_id": uuid.uuid4().hex,
            "parent_run_id": parent_run_id,
            "status": "running",
            "plan": dict(plan),
            "run": config.to_dict(),
            "latest_checkpoint": None,
            "evaluations": [],
        }
        self.write()
        self.emit(
            "run_started",
            step=plan["global_step"],
            requested_steps=plan["training"]["steps"],
            parent_run_id=parent_run_id,
        )
        self.write()

    def write(self) -> None:
        self.payload["elapsed_seconds"] = time.monotonic() - self.started
        _atomic_json(self.directory / "run.json", self.payload)

    def emit(self, kind: str, **values: Any) -> None:
        event = {
            "type": kind,
            "run_id": self.payload["run_id"],
            "timestamp": time.time(),
            "elapsed_seconds": time.monotonic() - self.started,
            **values,
        }
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        if self.reporter is not None:
            try:
                self.reporter(event)
            except Exception as exc:
                self.reporter = None
                self.payload["reporter_error"] = type(exc).__name__
                warnings.warn(f"training reporter disabled after {type(exc).__name__}", RuntimeWarning, stacklevel=2)
            if isinstance(self.reporter, ReporterGroup) and self.reporter.errors:
                self.payload["reporter_errors"] = list(self.reporter.errors)

    def finish(self, status: str, step: int, **values: Any) -> None:
        self.payload.update(status=status, global_step=step, **values)
        self.write()
        self.emit("run_finished", status=status, step=step, **values)
        self.write()

    def expired(self) -> bool:
        limit = self.config.max_seconds
        return limit is not None and time.monotonic() - self.started >= limit

    def checkpoint_due(self, step: int, first_step: int) -> bool:
        return (
            step == first_step
            or step % self.config.checkpoint_steps == 0
            or time.monotonic() - self.last_saved >= self.config.checkpoint_seconds
        )

    def checkpoint_path(self, step: int) -> Path:
        return self.directory / "checkpoints" / f"step-{step:08d}" / "model.arti.st"

    def saved(self, path: Path, step: int) -> None:
        self.latest = path
        self.last_saved = time.monotonic()
        self.payload["latest_checkpoint"] = str(path.relative_to(self.directory))
        self.payload["global_step"] = step
        self.write()
        self.emit("checkpoint", step=step, path=self.payload["latest_checkpoint"])


__all__ = ["ARTITrainingRunConfig", "RunReporter"]
