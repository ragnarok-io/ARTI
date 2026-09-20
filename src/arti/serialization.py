"""SafeTensors-based ARTI weight and training-checkpoint protocol."""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict, dataclass, is_dataclass
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor

from ._version import __version__
from .component_graph import (
    component_graph as build_component_graph,
    validate_component_graph,
    verify_component_graph,
)
from .component_registry import (
    canonical_contract_reference,
    component_state_contract,
    component_provenance,
    get_component_registry,
    validate_component_state_contract,
    validate_component_provenance,
    verify_component_provenance,
)


ARTI_ST_FORMAT = "arti.st"
ARTI_ST_FORMAT_VERSION = 1
MAX_JSON_BYTES = 16 * 1024 * 1024
MAX_CHECKPOINT_TREE_DEPTH = 64
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ARTISaveResult:
    """Paths and hashes produced by :func:`save`."""

    weights_path: Path
    manifest_path: Path
    lock_path: Path
    glyphs_path: Path | None
    vocab_path: Path | None
    checkpoint_path: Path | None
    checkpoint_metadata_path: Path | None
    weights_sha256: str
    manifest_sha256: str


@dataclass(frozen=True)
class ARTILoadResult:
    """Validated contents returned by :func:`load`."""

    model: nn.Module | None
    state_dict: dict[str, Tensor]
    manifest: dict[str, Any]
    glyph_tensors: dict[str, Tensor] | None
    vocab_metadata: Any | None
    training_state: Any | None
    missing_keys: tuple[str, ...]
    unexpected_keys: tuple[str, ...]
    device: str


def save(
    model: nn.Module,
    path: str | Path = "arti.st",
    *,
    config: Mapping[str, Any] | None = None,
    state_dict: Mapping[str, Tensor] | None = None,
    glyph_tensors: Tensor | Mapping[str, Tensor] | None = None,
    vocab_metadata: Any | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    training_state: Any | None = None,
    scope: str = "all",
) -> ARTISaveResult:
    """Save an ARTI model as weights plus strictly separated sidecars.

    ``path`` must end in ``.st``. The SafeTensors file contains only model
    tensors. Architecture/configuration, rigid glyph tensors, external
    vocabulary metadata, and resumable training state are written separately
    and bound by ``<stem>.lock.json`` SHA-256 records.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    _assert_artifact_components_portable(model)
    _validate_sealed_bank_queries(model)
    target = _weight_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if scope not in {"all", "trainable"}:
        raise ValueError("scope must be 'all' or 'trainable'")
    state = _model_state(model, scope=scope) if state_dict is None else dict(state_dict)
    if not state:
        raise ValueError("state_dict selected no model tensors")
    architecture = _architecture_payload(model, config, state_dict=state, scope=scope)
    return _save_state(
        state,
        target,
        architecture=architecture,
        scope=scope,
        glyph_tensors=glyph_tensors,
        vocab_metadata=vocab_metadata,
        optimizer=optimizer,
        scheduler=scheduler,
        training_state=training_state,
    )


def _assert_artifact_components_portable(model: nn.Module) -> None:
    """Reject runtime-only or host-bound records at the artifact boundary."""

    registry = get_component_registry()
    provenance = component_provenance(model)
    blocked: list[str] = []
    for item in provenance["components"]:
        registration = registry.registration_for_reference(item["ref"])
        if registration.artifact_policy != "portable":
            blocked.append(
                f"{item['path']}:{item['ref']}({registration.artifact_policy})"
            )
    if blocked:
        joined = ", ".join(blocked)
        raise ValueError(
            "arti.st cannot persist non-portable components; save the owning "
            f"constructible module instead: {joined}"
        )


def _validate_sealed_bank_queries(model: nn.Module) -> None:
    """Fail before writing when a mounted Query no longer matches its asset."""

    from .bank_query import SealedBankQuery
    from .shape_query import SealedTensorViewBankQuery

    for module in model.modules():
        if isinstance(module, SealedBankQuery):
            module.validate_runtime_state()
            module.signature.validate_query(module.query)
        elif isinstance(module, SealedTensorViewBankQuery):
            module.validate_runtime_state()
            module.signature.validate_query(module.query)


def load(
    path: str | Path = "arti.st",
    *,
    model: nn.Module | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    map_location: str | torch.device = "cpu",
    strict: bool | None = None,
    verify_architecture: bool = True,
    allow_legacy: bool = False,
    load_resources: bool = True,
    load_checkpoint: bool = True,
) -> ARTILoadResult:
    """Validate and load an ``arti.st`` package without executing model code.

    When ``model`` is provided its state is restored and the model is moved to
    ``map_location``. Without a model, the validated state dict is returned for
    explicit caller-controlled construction.
    """

    target = _weight_path(path)
    paths = _sidecar_paths(target)
    manifest, lock = _read_and_validate_package(
        target,
        paths,
        allow_legacy=allow_legacy,
    )
    device = str(torch.device(map_location))
    weights = _load_safetensors(
        target,
        expected_kind="weights",
        device=device,
        expected_component_graph=manifest["architecture"]["component_provenance"]["fingerprint"],
        expected_state_schema=manifest["architecture"]["state_contract"]["state_schema"]["fingerprint"],
    )
    try:
        validate_component_state_contract(
            manifest["architecture"]["state_contract"],
            state_dict=weights,
            model=model,
            allow_legacy=allow_legacy,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"arti.st component provenance/state contract is invalid: {error}") from error
    expected_count = manifest.get("weights", {}).get("tensor_count")
    if expected_count != len(weights):
        raise ValueError("arti.st tensor count does not match manifest")
    missing: tuple[str, ...] = ()
    unexpected: tuple[str, ...] = ()
    if model is not None:
        if verify_architecture:
            _verify_model_architecture(
                model,
                manifest["architecture"],
                allow_legacy=allow_legacy,
            )
        model.to(torch.device(map_location))
        resolved_strict = manifest.get("weight_scope") == "all" if strict is None else strict
        incompatible = model.load_state_dict(weights, strict=resolved_strict)
        missing = tuple(incompatible.missing_keys)
        unexpected = tuple(incompatible.unexpected_keys)
        if unexpected:
            raise ValueError(f"arti.st contains keys absent from target model: {unexpected}")

    glyphs = None
    vocab = None
    resources = manifest.get("resources", {})
    if load_resources and resources.get("glyphs") is not None:
        glyph_path = _member_path(target.parent, resources["glyphs"]["file"])
        glyphs = _load_safetensors(glyph_path, expected_kind="glyphs", device=device)
    if load_resources and resources.get("vocab") is not None:
        vocab_path = _member_path(target.parent, resources["vocab"]["file"])
        vocab = _load_json(vocab_path)
        _reject_noncanonical_persistent_contract_refs(vocab, label="ARTI vocabulary metadata")

    restored_training_state = None
    checkpoint = manifest.get("checkpoint")
    if load_checkpoint and checkpoint is not None:
        tensor_path = _member_path(target.parent, checkpoint["tensors_file"])
        metadata_path = _member_path(target.parent, checkpoint["metadata_file"])
        checkpoint_tensors = _load_safetensors(tensor_path, expected_kind="checkpoint", device=device)
        checkpoint_tree = _load_json(metadata_path)
        if checkpoint_tree.get("format") != ARTI_ST_FORMAT or checkpoint_tree.get("format_version") != ARTI_ST_FORMAT_VERSION:
            raise ValueError("ARTI checkpoint metadata format or version is invalid")
        if "state" not in checkpoint_tree:
            raise ValueError("ARTI checkpoint metadata is missing state")
        _reject_noncanonical_persistent_contract_refs(
            checkpoint_tree["state"],
            label="ARTI checkpoint metadata",
        )
        restored = _decode_tree(checkpoint_tree["state"], checkpoint_tensors)
        if optimizer is not None and restored.get("optimizer") is not None:
            optimizer.load_state_dict(restored["optimizer"])
        if scheduler is not None and restored.get("scheduler") is not None:
            scheduler.load_state_dict(restored["scheduler"])
        restored_training_state = restored.get("training_state")

    _validate_lock_matches_manifest(lock, manifest)
    return ARTILoadResult(
        model=model,
        state_dict=weights,
        manifest=manifest,
        glyph_tensors=glyphs,
        vocab_metadata=vocab,
        training_state=restored_training_state,
        missing_keys=missing,
        unexpected_keys=unexpected,
        device=device,
    )


def _save_state(
    state: Mapping[str, Tensor],
    target: Path,
    *,
    architecture: dict[str, Any],
    scope: str,
    glyph_tensors: Tensor | Mapping[str, Tensor] | None,
    vocab_metadata: Any | None,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any | None,
    training_state: Any | None,
) -> ARTISaveResult:
    target.parent.mkdir(parents=True, exist_ok=True)
    paths = _sidecar_paths(target)
    prepared_state = _prepare_tensors(state, label="model state")
    _atomic_safetensors(
        prepared_state,
        target,
        metadata={
            "format": ARTI_ST_FORMAT,
            "format_version": str(ARTI_ST_FORMAT_VERSION),
            "kind": "weights",
            "scope": scope,
            "component_graph_sha256": architecture["component_provenance"]["fingerprint"],
            "state_schema_sha256": architecture["state_contract"]["state_schema"]["fingerprint"],
        },
    )
    weights_sha = _file_sha256(target)
    files: dict[str, dict[str, Any]] = {
        "weights": _file_record(target, weights_sha),
    }

    glyph_record = None
    if glyph_tensors is not None:
        glyph_state = {"glyphs": glyph_tensors} if isinstance(glyph_tensors, Tensor) else dict(glyph_tensors)
        prepared_glyphs = _prepare_tensors(glyph_state, label="glyph tensors")
        _atomic_safetensors(
            prepared_glyphs,
            paths["glyphs"],
            metadata={"format": ARTI_ST_FORMAT, "format_version": str(ARTI_ST_FORMAT_VERSION), "kind": "glyphs"},
        )
        glyph_record = {**_file_record(paths["glyphs"], _file_sha256(paths["glyphs"])), "tensor_count": len(prepared_glyphs)}
        files["glyphs"] = glyph_record
    else:
        _remove_stale(paths["glyphs"])

    vocab_record = None
    if vocab_metadata is not None:
        normalized_vocab = _normalize_persistent_payload(vocab_metadata)
        _atomic_json(paths["vocab"], normalized_vocab)
        vocab_record = _file_record(paths["vocab"], _file_sha256(paths["vocab"]))
        files["vocab"] = vocab_record
    else:
        _remove_stale(paths["vocab"])

    checkpoint_record = None
    if optimizer is not None or scheduler is not None or training_state is not None:
        raw_checkpoint = _canonicalize_persistent_contract_refs({
            "optimizer": None if optimizer is None else optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "training_state": training_state,
        })
        checkpoint_tensors: dict[str, Tensor] = {}
        encoded = _encode_tree(raw_checkpoint, checkpoint_tensors, path="root")
        _atomic_safetensors(
            _prepare_tensors(checkpoint_tensors, label="checkpoint tensors"),
            paths["checkpoint"],
            metadata={"format": ARTI_ST_FORMAT, "format_version": str(ARTI_ST_FORMAT_VERSION), "kind": "checkpoint"},
        )
        checkpoint_metadata = {"format": ARTI_ST_FORMAT, "format_version": ARTI_ST_FORMAT_VERSION, "state": encoded}
        _atomic_json(paths["checkpoint_metadata"], checkpoint_metadata)
        tensor_record = _file_record(paths["checkpoint"], _file_sha256(paths["checkpoint"]))
        metadata_record = _file_record(paths["checkpoint_metadata"], _file_sha256(paths["checkpoint_metadata"]))
        files["checkpoint_tensors"] = tensor_record
        files["checkpoint_metadata"] = metadata_record
        checkpoint_record = {
            "tensors_file": paths["checkpoint"].name,
            "tensors_sha256": tensor_record["sha256"],
            "metadata_file": paths["checkpoint_metadata"].name,
            "metadata_sha256": metadata_record["sha256"],
            "optimizer_class": None if optimizer is None else _class_path(optimizer),
            "scheduler_class": None if scheduler is None else _class_path(scheduler),
        }
    else:
        _remove_stale(paths["checkpoint"])
        _remove_stale(paths["checkpoint_metadata"])

    manifest = _canonicalize_persistent_contract_refs({
        "format": ARTI_ST_FORMAT,
        "format_version": ARTI_ST_FORMAT_VERSION,
        "package_name": "arti",
        "package_version": __version__,
        "backend": "torch",
        "weight_scope": scope,
        "architecture": architecture,
        "weights": {**files["weights"], "tensor_count": len(prepared_state)},
        "resources": {"glyphs": glyph_record, "vocab": vocab_record},
        "checkpoint": checkpoint_record,
    })
    _atomic_json(paths["manifest"], manifest)
    manifest_sha = _file_sha256(paths["manifest"])
    files["manifest"] = _file_record(paths["manifest"], manifest_sha)
    lock = {
        "format": ARTI_ST_FORMAT,
        "format_version": ARTI_ST_FORMAT_VERSION,
        "manifest_file": paths["manifest"].name,
        "manifest_sha256": manifest_sha,
        "files": files,
    }
    _atomic_json(paths["lock"], lock)
    return ARTISaveResult(
        weights_path=target,
        manifest_path=paths["manifest"],
        lock_path=paths["lock"],
        glyphs_path=paths["glyphs"] if glyph_record is not None else None,
        vocab_path=paths["vocab"] if vocab_record is not None else None,
        checkpoint_path=paths["checkpoint"] if checkpoint_record is not None else None,
        checkpoint_metadata_path=paths["checkpoint_metadata"] if checkpoint_record is not None else None,
        weights_sha256=weights_sha,
        manifest_sha256=manifest_sha,
    )


def _read_and_validate_package(
    target: Path,
    paths: dict[str, Path],
    *,
    allow_legacy: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if not target.exists():
        raise FileNotFoundError(target)
    if not paths["manifest"].exists() or not paths["lock"].exists():
        raise ValueError("arti.st requires matching .json and .lock.json sidecars")
    lock = _load_json(paths["lock"])
    if lock.get("format") != ARTI_ST_FORMAT or lock.get("format_version") != ARTI_ST_FORMAT_VERSION:
        raise ValueError("unsupported ARTI lock format or version")
    if lock.get("manifest_file") != paths["manifest"].name:
        raise ValueError("ARTI lock manifest path does not match package stem")
    expected_manifest_hash = lock.get("manifest_sha256")
    if not _valid_sha(expected_manifest_hash) or _file_sha256(paths["manifest"]) != expected_manifest_hash:
        raise ValueError("ARTI manifest SHA-256 mismatch")
    manifest = _load_json(paths["manifest"])
    _validate_manifest(manifest, allow_legacy=allow_legacy)
    for record in _manifest_file_records(manifest):
        member = _member_path(target.parent, record["file"])
        if not member.exists() or _file_sha256(member) != record["sha256"]:
            raise ValueError(f"ARTI package SHA-256 mismatch for {record['file']}")
    return manifest, lock


def _validate_manifest(manifest: dict[str, Any], *, allow_legacy: bool = False) -> None:
    _reject_noncanonical_persistent_contract_refs(manifest, label="ARTI manifest")
    if manifest.get("format") != ARTI_ST_FORMAT or manifest.get("format_version") != ARTI_ST_FORMAT_VERSION:
        raise ValueError("unsupported arti.st format or version")
    if manifest.get("package_name") != "arti" or manifest.get("backend") != "torch":
        raise ValueError("arti.st package identity or backend is invalid")
    package_version = manifest.get("package_version")
    if not isinstance(package_version, str) or not package_version:
        raise ValueError("arti.st package_version must be a non-empty string")
    _check_version_compatibility(package_version, __version__)
    if manifest.get("weight_scope") not in {"all", "trainable"}:
        raise ValueError("arti.st weight_scope is invalid")
    if not isinstance(manifest.get("architecture"), dict):
        raise ValueError("arti.st architecture must be a dictionary")
    try:
        validate_component_provenance(
            manifest["architecture"].get("component_provenance"),
            allow_legacy=allow_legacy,
            artifact_scope=True,
        )
        component_graph_value = manifest["architecture"].get("component_graph")
        if component_graph_value is not None:
            validate_component_graph(component_graph_value)
        validate_component_state_contract(
            manifest["architecture"].get("state_contract"),
            allow_legacy=allow_legacy,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"arti.st component provenance is invalid: {error}") from error
    weights = manifest.get("weights")
    if not isinstance(weights, dict) or not isinstance(weights.get("tensor_count"), int):
        raise ValueError("arti.st weights record is invalid")
    for record in _manifest_file_records(manifest):
        if Path(record.get("file", "")).name != record.get("file") or not _valid_sha(record.get("sha256")):
            raise ValueError("arti.st file record is invalid")


def _validate_lock_matches_manifest(lock: dict[str, Any], manifest: dict[str, Any]) -> None:
    locked = lock.get("files")
    if not isinstance(locked, dict):
        raise ValueError("ARTI lock files must be a dictionary")
    manifest_records = {record["file"]: record["sha256"] for record in _manifest_file_records(manifest)}
    lock_records = {
        record.get("file"): record.get("sha256")
        for record in locked.values()
        if isinstance(record, dict) and record.get("file") != lock.get("manifest_file")
    }
    if manifest_records != lock_records:
        raise ValueError("ARTI lock file records do not match manifest")


def _manifest_file_records(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    records = [manifest["weights"]]
    resources = manifest.get("resources") or {}
    for name in ("glyphs", "vocab"):
        if resources.get(name) is not None:
            records.append(resources[name])
    checkpoint = manifest.get("checkpoint")
    if checkpoint is not None:
        records.extend(
            [
                {"file": checkpoint["tensors_file"], "sha256": checkpoint["tensors_sha256"]},
                {"file": checkpoint["metadata_file"], "sha256": checkpoint["metadata_sha256"]},
            ]
        )
    return records


def _model_state(model: nn.Module, *, scope: str) -> dict[str, Tensor]:
    state = model.state_dict()
    if scope == "all":
        return dict(state)
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    selected = {name: value for name, value in state.items() if name in trainable_names}
    if not selected:
        raise ValueError("scope='trainable' selected no trainable model tensors")
    return selected


def _prepare_tensors(state: Mapping[str, Tensor], *, label: str) -> dict[str, Tensor]:
    prepared: dict[str, Tensor] = {}
    for key, value in state.items():
        if not isinstance(key, str) or not key:
            raise ValueError(f"{label} keys must be non-empty strings")
        if not isinstance(value, Tensor):
            raise ValueError(f"{label} value {key!r} is not a tensor")
        if value.layout != torch.strided:
            raise ValueError(f"{label} value {key!r} must use strided tensor layout")
        prepared[key] = value.detach().to("cpu").contiguous().clone()
    return prepared


def _architecture_payload(
    model: nn.Module,
    config: Mapping[str, Any] | None,
    *,
    state_dict: Mapping[str, Tensor],
    scope: str,
) -> dict[str, Any]:
    resolved: Any = config
    if resolved is None:
        serialization_config = getattr(model, "serialization_config", None)
        if callable(serialization_config):
            resolved = serialization_config()
        candidate = getattr(model, "config", None)
        if resolved is None and candidate is not None:
            if is_dataclass(candidate):
                resolved = asdict(candidate)
            elif callable(getattr(candidate, "to_dict", None)):
                resolved = candidate.to_dict()
    payload = {
        "module": model.__class__.__module__,
        "class_name": model.__class__.__qualname__,
        "config": _normalize_persistent_payload({} if resolved is None else resolved),
        "component_provenance": component_provenance(model),
        "component_graph": build_component_graph(model),
        "state_contract": component_state_contract(model, state_dict, scope=scope),
    }
    candidate = getattr(model, "config", None)
    contract = getattr(candidate, "context_contract", None)
    if callable(contract):
        payload["context_contract"] = _normalize_persistent_payload(contract())
    return payload


def _verify_model_architecture(
    model: nn.Module,
    architecture: Mapping[str, Any],
    *,
    allow_legacy: bool = False,
) -> None:
    expected_module = architecture.get("module")
    expected_class = architecture.get("class_name")
    if expected_module is None and expected_class is None:
        return
    actual_module = model.__class__.__module__
    actual_class = model.__class__.__qualname__
    if expected_module != actual_module or expected_class != actual_class:
        raise ValueError(
            "arti.st architecture does not match target model: "
            f"expected {expected_module}.{expected_class}, got {actual_module}.{actual_class}"
        )
    try:
        verify_component_provenance(
            model,
            architecture["component_provenance"],
            allow_legacy=allow_legacy,
        )
        component_graph_value = architecture.get("component_graph")
        if component_graph_value is not None:
            verify_component_graph(model, component_graph_value)
        validate_component_state_contract(
            architecture["state_contract"],
            model=model,
            allow_legacy=allow_legacy,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(str(error)) from error


def _encode_tree(value: Any, tensors: dict[str, Tensor], *, path: str) -> Any:
    if isinstance(value, Tensor):
        key = f"tensor_{len(tensors):08d}"
        tensors[key] = value
        return {"__arti_tensor__": key}
    if isinstance(value, Mapping):
        return {"__arti_dict__": [[_encode_tree(key, tensors, path=f"{path}.key"), _encode_tree(item, tensors, path=f"{path}.{key}")] for key, item in value.items()]}
    if isinstance(value, tuple):
        return {"__arti_tuple__": [_encode_tree(item, tensors, path=f"{path}[]") for item in value]}
    if isinstance(value, list):
        return {"__arti_list__": [_encode_tree(item, tensors, path=f"{path}[]") for item in value]}
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ValueError(f"checkpoint value at {path} is not tensor/JSON compatible: {type(value).__name__}")


def _decode_tree(value: Any, tensors: Mapping[str, Tensor], *, depth: int = 0) -> Any:
    if depth > MAX_CHECKPOINT_TREE_DEPTH:
        raise ValueError("checkpoint metadata tree exceeds the maximum depth")
    if isinstance(value, dict) and set(value) == {"__arti_tensor__"}:
        key = value["__arti_tensor__"]
        if key not in tensors:
            raise ValueError(f"checkpoint tensor reference {key!r} is missing")
        return tensors[key]
    if isinstance(value, dict) and set(value) == {"__arti_dict__"}:
        return {
            _decode_tree(key, tensors, depth=depth + 1): _decode_tree(item, tensors, depth=depth + 1)
            for key, item in value["__arti_dict__"]
        }
    if isinstance(value, dict) and set(value) == {"__arti_tuple__"}:
        return tuple(_decode_tree(item, tensors, depth=depth + 1) for item in value["__arti_tuple__"])
    if isinstance(value, dict) and set(value) == {"__arti_list__"}:
        return [_decode_tree(item, tensors, depth=depth + 1) for item in value["__arti_list__"]]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ValueError("checkpoint metadata tree is invalid")


def _load_json(path: Path) -> Any:
    size = path.stat().st_size
    if size > MAX_JSON_BYTES:
        raise ValueError(f"ARTI JSON sidecar {path.name!r} exceeds the {MAX_JSON_BYTES:,}-byte limit")
    return json.loads(path.read_text(encoding="utf-8"))


def _json_normalize(value: Any) -> Any:
    if is_dataclass(value):
        return _json_normalize(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_normalize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_normalize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise ValueError(f"configuration value is not JSON compatible: {type(value).__name__}")


def _normalize_persistent_payload(value: Any) -> Any:
    """JSON-normalize a payload and resolve declarations before persistence."""

    return _canonicalize_persistent_contract_refs(_json_normalize(value))


def _canonicalize_persistent_contract_refs(value: Any) -> Any:
    """Resolve exact component declarations without rewriting contract semantic keys."""

    if isinstance(value, Mapping):
        is_component_contract = value.get("kind") == "arti.component-contract"
        return {
            key: (
                item
                if is_component_contract and key == "semantic_key"
                else _canonicalize_persistent_contract_refs(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize_persistent_contract_refs(item) for item in value]
    if isinstance(value, str):
        try:
            return canonical_contract_reference(value)
        except (TypeError, ValueError):
            return value
    return value


def _reject_noncanonical_persistent_contract_refs(value: Any, *, label: str) -> None:
    if _canonicalize_persistent_contract_refs(value) != value:
        raise ValueError(f"{label} contains a non-canonical component reference")


def _load_safetensors(
    path: Path,
    *,
    expected_kind: str,
    device: str,
    expected_component_graph: str | None = None,
    expected_state_schema: str | None = None,
) -> dict[str, Tensor]:
    with safe_open(path, framework="pt", device=device) as handle:
        metadata = handle.metadata() or {}
    if metadata.get("format") != ARTI_ST_FORMAT or metadata.get("format_version") != str(ARTI_ST_FORMAT_VERSION):
        raise ValueError(f"{path.name} is not an ARTI SafeTensors file")
    if metadata.get("kind") != expected_kind:
        raise ValueError(f"{path.name} kind does not match expected {expected_kind!r}")
    if (
        expected_component_graph is not None
        and metadata.get("component_graph_sha256") != expected_component_graph
    ):
        raise ValueError("ARTI SafeTensors component graph fingerprint does not match manifest")
    if (
        expected_state_schema is not None
        and metadata.get("state_schema_sha256") != expected_state_schema
    ):
        raise ValueError("ARTI SafeTensors state schema fingerprint does not match manifest")
    return load_file(path, device=device)


def _check_version_compatibility(saved: str, current: str) -> None:
    saved_version = _version_tuple(saved)
    current_version = _version_tuple(current)
    if saved_version[0] == 0 and current_version[0] == 1:
        if saved_version[1] > 2:
            raise ValueError(f"arti.st alpha version {saved} is newer than ARTI {current}")
        return
    if saved_version[0] in {0, 1} and current_version[0] == 2:
        maximum_minor = 2 if saved_version[0] == 0 else 9
        if saved_version[1] > maximum_minor:
            raise ValueError(f"arti.st legacy version {saved} is newer than ARTI {current}")
        return
    if saved_version[0] != current_version[0]:
        raise ValueError(f"arti.st package major version {saved} is incompatible with ARTI {current}")
    if saved_version[1] > current_version[1]:
        raise ValueError(f"arti.st package version {saved} is newer than ARTI {current}")
    if saved_version[0] == 0 and saved_version[1] > current_version[1]:
        raise ValueError(f"arti.st alpha version {saved} is newer than ARTI {current}")


def _version_tuple(value: str) -> tuple[int, int, int]:
    match = re.match(r"^(\d+)\.(\d+)\.(\d+)", value)
    if match is None:
        raise ValueError(f"invalid ARTI package version {value!r}")
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def _weight_path(path: str | Path) -> Path:
    target = Path(path)
    if target.suffix.lower() != ".st":
        raise ValueError("ARTI SafeTensors weight path must end in .st (for example arti.st)")
    return target


def _sidecar_paths(target: Path) -> dict[str, Path]:
    stem = target.with_suffix("")
    return {
        "manifest": stem.with_suffix(".json"),
        "lock": stem.with_suffix(".lock.json"),
        "glyphs": stem.with_suffix(".glyphs.st"),
        "vocab": stem.with_suffix(".vocab.json"),
        "checkpoint": stem.with_suffix(".checkpoint.st"),
        "checkpoint_metadata": stem.with_suffix(".checkpoint.json"),
    }


def _member_path(root: Path, name: str) -> Path:
    if Path(name).name != name:
        raise ValueError("ARTI package member paths must be sibling file names")
    return root / name


def _atomic_safetensors(tensors: Mapping[str, Tensor], path: Path, *, metadata: Mapping[str, str]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        save_file(dict(tensors), temporary, metadata=dict(metadata))
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _remove_stale(path: Path) -> None:
    if path.exists():
        path.unlink()


def _file_record(path: Path, digest: str) -> dict[str, Any]:
    return {"file": path.name, "sha256": digest, "bytes": path.stat().st_size}


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_sha(value: Any) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


def _class_path(value: Any) -> str:
    return f"{value.__class__.__module__}.{value.__class__.__qualname__}"


__all__ = [
    "ARTI_ST_FORMAT",
    "ARTI_ST_FORMAT_VERSION",
    "ARTILoadResult",
    "ARTISaveResult",
    "load",
    "save",
]
