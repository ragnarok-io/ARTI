"""Ordered composition of independently exported ARTI adapters."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .artifacts import ARTIFitResult


ADAPTER_STACK_FORMAT = "arti.fit.adapter-stack.v1"


def apply_adapter_stack(
    model: nn.Module,
    manifest: str | Path,
    *,
    sample_batch: Any | None = None,
    map_location: str | torch.device | None = None,
    trust_artifact_contract: bool = False,
) -> tuple[ARTIFitResult, ...]:
    """Apply independently loadable adapters in their declared order."""

    from .project import apply_adapter

    manifest_path = Path(manifest).resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or payload.get("format") != ADAPTER_STACK_FORMAT:
        raise ValueError(f"adapter stack format must be {ADAPTER_STACK_FORMAT!r}")
    load_order = payload.get("load_order")
    if not isinstance(load_order, list) or not load_order:
        raise ValueError("adapter stack load_order must be a non-empty list")
    results = []
    for index, entry in enumerate(load_order):
        if not isinstance(entry, Mapping):
            raise ValueError(f"adapter stack entry {index} must be a mapping")
        relative_path = entry.get("path")
        expected_sha256 = entry.get("sha256")
        if not isinstance(relative_path, str) or not relative_path:
            raise ValueError(f"adapter stack entry {index} path must be a non-empty string")
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError(f"adapter stack entry {index} sha256 must be a SHA-256 digest")
        artifact = (manifest_path.parent / relative_path).resolve()
        if not artifact.is_file():
            raise FileNotFoundError(f"adapter stack artifact does not exist: {artifact}")
        actual_sha256 = hashlib.sha256(artifact.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            raise ValueError(f"adapter stack artifact hash mismatch: {artifact}")
        results.append(
            apply_adapter(
                model,
                artifact,
                sample_batch=sample_batch,
                map_location=map_location,
                trust_artifact_contract=trust_artifact_contract,
            )
        )
    return tuple(results)


__all__ = ["ADAPTER_STACK_FORMAT", "apply_adapter_stack"]
