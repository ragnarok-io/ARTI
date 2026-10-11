"""Validation-based selection and optional stopping, outside neural execution."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
import os
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class ARTIValidationConfig:
    monitor: str = "loss"
    mode: str = "min"
    selection: str = "best"
    min_delta: float = 0.0
    patience: int | None = None
    start_step: int = 0
    min_evaluations: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.monitor, str) or not self.monitor:
            raise ValueError("validation.monitor must be a nonempty metric name")
        if self.mode not in {"min", "max"} or self.selection not in {"best", "last"}:
            raise ValueError("validation requires mode=min|max and selection=best|last")
        if (
            isinstance(self.min_delta, bool)
            or not isinstance(self.min_delta, (int, float))
            or not math.isfinite(self.min_delta)
            or self.min_delta < 0
        ):
            raise ValueError("validation.min_delta must be finite and nonnegative")
        if self.patience is not None and (type(self.patience) is not int or self.patience <= 0):
            raise ValueError("validation.patience must be a positive integer or None")
        if type(self.start_step) is not int or self.start_step < 0:
            raise ValueError("validation.start_step must be a nonnegative integer")
        if type(self.min_evaluations) is not int or self.min_evaluations <= 0:
            raise ValueError("validation.min_evaluations must be a positive integer")


class _ValidationTracker:
    def __init__(self, config: ARTIValidationConfig) -> None:
        self.config = config
        self.best_value: float | None = None
        self.reference_value: float | None = None
        self.best_step: int | None = None
        self.best_checkpoint: Path | None = None
        self.evaluations = 0
        self.bad_evaluations = 0
        self.last_step: int | None = None
        self.cadence: int | None = None

    def _improves(self, value: float, baseline: float | None, delta: float = 0.0) -> bool:
        if baseline is None:
            return True
        return baseline - value > delta if self.config.mode == "min" else value - baseline > delta

    def observe(
        self, metrics: Mapping[str, Any], step: int, *, scheduled: bool = True
    ) -> tuple[bool, bool]:
        name = self.config.monitor
        value = metrics.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"validation monitor {name!r} must be a finite numeric metric")
        self.last_step = step
        if step < self.config.start_step:
            return False, False
        value = float(value)
        improved = self._improves(value, self.best_value)
        if improved:
            self.best_value, self.best_step = value, step
        # An off-cadence final check selects an asset without changing patience.
        if scheduled:
            self.evaluations += 1
            if self._improves(value, self.reference_value, self.config.min_delta):
                self.reference_value, self.bad_evaluations = value, 0
            else:
                self.bad_evaluations += 1
        stop = (
            scheduled
            and self.config.patience is not None
            and self.evaluations >= self.config.min_evaluations
            and self.bad_evaluations >= self.config.patience
        )
        return improved, stop

    def state_dict(self, directory: Path) -> dict[str, Any]:
        return {
            "config": asdict(self.config),
            "best_value": self.best_value,
            "reference_value": self.reference_value,
            "best_step": self.best_step,
            "best_checkpoint": None
            if self.best_checkpoint is None
            else os.path.relpath(self.best_checkpoint, directory),
            "evaluations": self.evaluations,
            "bad_evaluations": self.bad_evaluations,
            "last_step": self.last_step,
            "cadence": self.cadence,
        }

    @classmethod
    def from_state(cls, state: Mapping[str, Any], directory: Path) -> _ValidationTracker:
        tracker = cls(ARTIValidationConfig(**state["config"]))
        for name in (
            "best_value",
            "reference_value",
            "best_step",
            "evaluations",
            "bad_evaluations",
            "last_step",
        ):
            setattr(tracker, name, state[name])
        if state["best_checkpoint"] is not None:
            tracker.best_checkpoint = (directory / state["best_checkpoint"]).resolve()
        tracker.cadence = state.get("cadence")
        return tracker

    def receipt(self, step: int) -> dict[str, Any]:
        selected_best = self.config.selection == "best" and self.best_checkpoint is not None
        return {
            "selection": "best" if selected_best else "last",
            "reason": "validation_metric"
            if selected_best
            else "no_eligible_validation"
            if self.config.selection == "best"
            else "declared_last",
            "selected_step": self.best_step if selected_best else step,
            "monitor": self.config.monitor,
            "mode": self.config.mode,
            "best_value": self.best_value,
            "best_step": self.best_step,
            "evaluations": self.evaluations,
            "bad_evaluations": self.bad_evaluations,
        }


__all__ = ["ARTIValidationConfig"]
