"""Private SafeTensors contract for composable Recall expert routing."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from ._recall_state import RecallStateExpertAssembly


FORMAT = "arti.recall-expert-router-overlay.v2"
_RESERVED_METADATA = {
    "format",
    "model",
    "layer_indices",
    "expert_names",
    "expert_files",
    "expert_sha256",
    "expert_count",
    "site_count",
}


def artifact_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of one expert artifact."""

    resolved = Path(path)
    digest = hashlib.sha256()
    with resolved.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_assembly(
    assemblies: Sequence[RecallStateExpertAssembly],
    expert_names: Sequence[str],
    expert_artifacts: Sequence[str | Path],
    layer_indices: Sequence[int],
) -> tuple[tuple[str, ...], tuple[Path, ...], tuple[int, ...]]:
    if not assemblies:
        raise ValueError("at least one Recall expert assembly is required")
    names = tuple(str(name) for name in expert_names)
    artifacts = tuple(Path(path) for path in expert_artifacts)
    layers = tuple(int(index) for index in layer_indices)
    if not names or len(names) != len(set(names)) or any(not name.strip() for name in names):
        raise ValueError("expert names must be non-empty and unique")
    if len(artifacts) != len(names):
        raise ValueError("expert names and artifacts must have the same length")
    if len(assemblies) != len(layers):
        raise ValueError("each layer index requires one Recall expert assembly")
    if tuple(sorted(set(layers))) != layers:
        raise ValueError("layer_indices must be sorted and unique")
    if any(assembly.expert_count != len(names) for assembly in assemblies):
        raise ValueError("every assembly must contain every named expert")
    missing = [str(path) for path in artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Recall expert artifacts do not exist: {missing}")
    return names, artifacts, layers


def save_recall_expert_router_overlay(
    path: str | Path,
    assemblies: Sequence[RecallStateExpertAssembly],
    *,
    expert_names: Sequence[str],
    expert_artifacts: Sequence[str | Path],
    layer_indices: Sequence[int],
    model_id: str,
    metadata: Mapping[str, str] | None = None,
) -> dict[str, object]:
    """Save only route calibration while binding it to ordered expert assets."""

    names, artifacts, layers = _validate_assembly(
        assemblies, expert_names, expert_artifacts, layer_indices
    )
    if not str(model_id).strip():
        raise ValueError("model_id must not be empty")
    extra = {} if metadata is None else {str(key): str(value) for key, value in metadata.items()}
    overlap = _RESERVED_METADATA.intersection(extra)
    if overlap:
        raise ValueError(f"metadata cannot replace reserved fields: {sorted(overlap)}")
    hashes = tuple(artifact_sha256(artifact) for artifact in artifacts)
    safetensors_metadata = {
        "format": FORMAT,
        "model": str(model_id),
        "layer_indices": json.dumps(layers),
        "expert_names": json.dumps(names),
        "expert_files": json.dumps(tuple(artifact.name for artifact in artifacts)),
        "expert_sha256": json.dumps(hashes),
        "expert_count": str(len(names)),
        "site_count": str(len(assemblies)),
        **extra,
    }
    tensors = {
        "route_weight": torch.stack(
            [assembly.route_weight.detach().cpu() for assembly in assemblies]
        ).contiguous(),
        "route_bias": torch.stack(
            [assembly.route_bias.detach().cpu() for assembly in assemblies]
        ).contiguous(),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    save_file(tensors, str(temporary), metadata=safetensors_metadata)
    temporary.replace(destination)
    return {
        "format": FORMAT,
        "path": str(destination),
        "model": str(model_id),
        "layer_indices": layers,
        "expert_names": names,
        "expert_sha256": hashes,
    }


def load_recall_expert_router_overlay(
    path: str | Path,
    assemblies: Sequence[RecallStateExpertAssembly],
    *,
    expert_names: Sequence[str],
    expert_artifacts: Sequence[str | Path],
    layer_indices: Sequence[int],
    model_id: str,
) -> dict[str, object]:
    """Strictly restore a route overlay for the exact ordered source assets."""

    names, artifacts, layers = _validate_assembly(
        assemblies, expert_names, expert_artifacts, layer_indices
    )
    source = Path(path)
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
    expected_hashes = tuple(artifact_sha256(artifact) for artifact in artifacts)
    expected_metadata = {
        "format": FORMAT,
        "model": str(model_id),
        "layer_indices": json.dumps(layers),
        "expert_names": json.dumps(names),
        "expert_sha256": json.dumps(expected_hashes),
        "expert_count": str(len(names)),
        "site_count": str(len(assemblies)),
    }
    mismatched = {
        key: (metadata.get(key), expected)
        for key, expected in expected_metadata.items()
        if metadata.get(key) != expected
    }
    if mismatched:
        raise ValueError(f"Recall expert router overlay contract mismatch: {mismatched}")
    tensors = load_file(str(source), device="cpu")
    expected_weight_shape = (len(assemblies), len(names), len(names))
    expected_bias_shape = (len(assemblies), len(names))
    if set(tensors) != {"route_weight", "route_bias"}:
        raise ValueError("Recall expert router overlay contains unknown tensors")
    if tuple(tensors["route_weight"].shape) != expected_weight_shape:
        raise ValueError(f"route_weight must have shape {expected_weight_shape}")
    if tuple(tensors["route_bias"].shape) != expected_bias_shape:
        raise ValueError(f"route_bias must have shape {expected_bias_shape}")
    with torch.no_grad():
        for site, assembly in enumerate(assemblies):
            assembly.route_weight.copy_(
                tensors["route_weight"][site].to(assembly.route_weight)
            )
            assembly.route_bias.copy_(tensors["route_bias"][site].to(assembly.route_bias))
    return {
        "format": FORMAT,
        "path": str(source),
        "model": str(model_id),
        "layer_indices": layers,
        "expert_names": names,
        "expert_sha256": expected_hashes,
    }
