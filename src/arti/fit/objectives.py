"""Declarative objective plans for ``arti.fit``."""

from __future__ import annotations

from collections.abc import Iterable


OBJECTIVE_ALIASES = {
    "calibrate": "preserve-output",
    "preserve": "preserve-output",
    "preserve-output": "preserve-output",
    "task": "task-fit",
    "fit": "task-fit",
    "task-fit": "task-fit",
    "train": "task-fit",
    "validate": "validate",
    "validation": "validate",
}


def resolve_objectives(objective: str | Iterable[str] | None) -> tuple[str, ...]:
    """Normalize a user objective declaration into executable task names."""
    if objective is None:
        return ()
    values = (objective,) if isinstance(objective, str) else tuple(objective)
    resolved = []
    for value in values:
        normalized = value.strip().lower()
        try:
            task = OBJECTIVE_ALIASES[normalized]
        except KeyError as exc:
            known = ", ".join(sorted(OBJECTIVE_ALIASES))
            raise ValueError(f"unknown ARTI fit objective {value!r}; expected one of: {known}") from exc
        if task not in resolved:
            resolved.append(task)
    return tuple(resolved)


def infer_objectives(
    *,
    objective: str | Iterable[str] | None = None,
    has_calibration: bool = False,
    has_training: bool = False,
    has_validation: bool = False,
) -> tuple[str, ...]:
    """Resolve explicit objectives, or infer a plan from supplied loaders."""
    explicit = resolve_objectives(objective)
    if explicit:
        return explicit
    inferred = []
    if has_calibration:
        inferred.append("preserve-output")
    if has_training:
        inferred.append("task-fit")
    if has_validation:
        inferred.append("validate")
    return tuple(inferred)
