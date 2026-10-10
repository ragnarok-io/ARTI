"""Local training-run policy and reporting, outside neural execution graphs."""

from __future__ import annotations

import json
import math
import time
import uuid
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from .serialization import _atomic_json


@dataclass(frozen=True)
class ARTITrainingRunConfig:
    output_dir: str | Path | None = None
    checkpoint_steps: int = 50
    checkpoint_seconds: float = 60.0
    log_steps: int = 10
    max_seconds: float | None = None
    evaluation_steps: int | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.checkpoint_steps <= 0 or self.log_steps <= 0:
            raise ValueError("checkpoint_steps and log_steps must be positive")
        for name in ("checkpoint_seconds", "max_seconds"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value <= 0):
                raise ValueError(f"{name} must be finite and positive")
        if self.evaluation_steps is not None and self.evaluation_steps <= 0:
            raise ValueError("evaluation_steps must be positive")
        json.dumps(dict(self.metadata), allow_nan=False)

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["output_dir"] = None if self.output_dir is None else str(self.output_dir)
        return values


RunReporter = Callable[[Mapping[str, Any]], None]


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
            "status": "running",
            "plan": dict(plan),
            "run": config.to_dict(),
            "latest_checkpoint": None,
            "evaluations": [],
        }
        self.write()

    def write(self) -> None:
        self.payload["elapsed_seconds"] = time.monotonic() - self.started
        _atomic_json(self.directory / "run.json", self.payload)

    def emit(self, kind: str, **values: Any) -> None:
        event = {"type": kind, **values}
        with (self.directory / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, sort_keys=True, allow_nan=False) + "\n")
        if self.reporter is not None:
            try:
                self.reporter(event)
            except Exception as exc:
                self.reporter = None
                self.payload["reporter_error"] = type(exc).__name__
                warnings.warn(f"training reporter disabled after {type(exc).__name__}", RuntimeWarning, stacklevel=2)

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
