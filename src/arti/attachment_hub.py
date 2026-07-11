"""Hugging Face-compatible directory lifecycle for ARTI attachments."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, TYPE_CHECKING

import torch
import torch.nn as nn

from ._version import __version__
from .attachment_config import write_attach_config

if TYPE_CHECKING:
    from .attachment import ARTIAttachment


HUB_MANIFEST = "arti-hub.json"
HUB_FORMAT = "arti.hf.bundle"
HUB_VERSION = 1
DEFAULT_ARTIFACT = "model.recall.arti.st"
DEFAULT_CONFIG = "arti-attach.toml"
DEFAULT_LOCK = "arti.attach.lock.json"
BASE_WEIGHT_NAMES = {
    "pytorch_model.bin",
    "model.safetensors",
    "pytorch_model.bin.index.json",
    "model.safetensors.index.json",
}


@dataclass(frozen=True)
class ARTIHubSaveResult:
    directory: Path
    manifest_path: Path
    artifact_path: Path
    config_path: Path
    lock_path: Path
    base_model: str
    revision: str | None


@dataclass(frozen=True)
class ARTIDoctorReport:
    ok: bool
    model_class: str
    layers: tuple[str, ...]
    trainable_parameters: int
    devices: tuple[str, ...]
    dtypes: tuple[str, ...]
    backbone_frozen: bool
    gradient_checkpointing: bool
    quantized: bool
    distributed_wrapper: str | None
    issues: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def save_attachment_pretrained(
    attachment: "ARTIAttachment",
    directory: str | Path,
    *,
    base_model: str | Path | None = None,
    revision: str | None = None,
    training_session: Any | None = None,
) -> ARTIHubSaveResult:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    _reject_base_weights(target)
    source = str(base_model) if base_model is not None else _infer_base_source(attachment._model)
    if not source:
        raise ValueError("base_model is required when model.config._name_or_path is unavailable")
    config_revision = getattr(getattr(attachment._model, "config", None), "_commit_hash", None)
    resolved_revision = revision or config_revision
    artifact_path = target / DEFAULT_ARTIFACT
    if training_session is None:
        saved = attachment.save(artifact_path)
    else:
        if training_session.attachment is not attachment:
            raise ValueError("training_session belongs to a different ARTI attachment")
        saved = attachment.save(
            artifact_path,
            optimizer=training_session.optimizer,
            scheduler=training_session.scheduler,
            training_state={
                "engine": training_session.config.engine,
                "global_step": training_session.global_step,
                "loss_history": list(training_session.loss_history),
            },
        )
    config_path = target / DEFAULT_CONFIG
    if attachment.declaration is not None:
        write_attach_config(config_path, attachment.declaration)
    else:
        from .attachment_config import ARTIAttachConfig

        write_attach_config(config_path, ARTIAttachConfig(recall=_portable_recall(attachment.config)))
    lock_path = attachment.write_lock(target / DEFAULT_LOCK)
    manifest = {
        "format": HUB_FORMAT,
        "format_version": HUB_VERSION,
        "arti_version": __version__,
        "base_model": {"source": source, "revision": resolved_revision},
        "artifact": {"file": artifact_path.name, "sha256": saved.weights_sha256},
        "config": {"file": config_path.name, "sha256": _sha256(config_path)},
        "lock": {"file": lock_path.name, "sha256": _sha256(lock_path)},
    }
    manifest["fingerprint"] = _json_sha256(manifest)
    manifest_path = target / HUB_MANIFEST
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _reject_base_weights(target)
    return ARTIHubSaveResult(target, manifest_path, artifact_path, config_path, lock_path, source, resolved_revision)


def load_attachment_pretrained(
    directory: str | Path,
    *,
    model: nn.Module | None = None,
    map_location: str | torch.device | None = None,
    model_kwargs: Mapping[str, Any] | None = None,
) -> nn.Module:
    target = Path(directory)
    _reject_base_weights(target)
    manifest = _read_manifest(target)
    base = manifest["base_model"]
    if model is None:
        try:
            from transformers import AutoModelForCausalLM
        except ImportError as exc:
            raise RuntimeError("ARTI.from_pretrained requires `uv sync --extra qwen`") from exc
        source = str(base["source"])
        candidate = target / source
        if not Path(source).is_absolute() and candidate.exists():
            source = str(candidate)
        kwargs = dict(model_kwargs or {})
        if base.get("revision") is not None and not Path(source).exists():
            kwargs.setdefault("revision", base["revision"])
        model = AutoModelForCausalLM.from_pretrained(source, **kwargs)
    from .attachment import ARTI

    artifact = target / manifest["artifact"]["file"]
    loaded = ARTI.load(model, artifact, map_location=map_location)
    loaded.arti._pretrained_directory = target.resolve()
    loaded.arti._resume_artifact = artifact.resolve()
    return loaded


def attachment_doctor(attachment: "ARTIAttachment") -> ARTIDoctorReport:
    issues: list[str] = []
    wrappers = attachment._layered.wrappers
    if tuple(wrappers) != attachment.paths:
        issues.append("resolved layer paths no longer match attached wrappers")
    devices = tuple(sorted({str(parameter.device) for parameter in attachment.parameters()}))
    dtypes = tuple(sorted({str(parameter.dtype) for parameter in attachment.parameters()}))
    if len(devices) > 1:
        issues.append("Recall parameters span multiple devices")
    if len(dtypes) > 1:
        issues.append("Recall parameters use mixed dtypes")
    recall_ids = {id(parameter) for parameter in attachment.parameters()}
    backbone = [parameter for parameter in attachment._model.parameters() if id(parameter) not in recall_ids]
    frozen = all(not parameter.requires_grad for parameter in backbone)
    if attachment.config.freeze_backbone and not frozen:
        issues.append("configuration freezes the backbone but trainable backbone parameters were found")
    for path, wrapper in wrappers.items():
        reference = next((parameter for parameter in wrapper.base.parameters() if parameter.is_floating_point()), None)
        recall = next(wrapper.recall.parameters(), None)
        if reference is not None and recall is not None and reference.device != recall.device:
            issues.append(f"layer {path} base and Recall devices differ")
    model = attachment._model
    distributed = _distributed_wrapper(model)
    gradient_checkpointing = bool(getattr(model, "is_gradient_checkpointing", False))
    quantized = bool(getattr(model, "is_quantized", False) or getattr(model, "is_loaded_in_4bit", False) or getattr(model, "is_loaded_in_8bit", False))
    return ARTIDoctorReport(
        ok=not issues,
        model_class=f"{type(model).__module__}.{type(model).__qualname__}",
        layers=attachment.paths,
        trainable_parameters=sum(parameter.numel() for parameter in attachment.parameters()),
        devices=devices,
        dtypes=dtypes,
        backbone_frozen=frozen,
        gradient_checkpointing=gradient_checkpointing,
        quantized=quantized,
        distributed_wrapper=distributed,
        issues=tuple(issues),
    )


def _read_manifest(directory: Path) -> dict[str, Any]:
    manifest_path = directory / HUB_MANIFEST
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    fingerprint = payload.pop("fingerprint", None)
    if payload.get("format") != HUB_FORMAT or payload.get("format_version") != HUB_VERSION:
        raise ValueError("unsupported ARTI Hub bundle")
    if fingerprint != _json_sha256(payload):
        raise ValueError("ARTI Hub manifest fingerprint is invalid")
    for key in ("artifact", "config", "lock"):
        record = payload.get(key)
        if not isinstance(record, Mapping):
            raise ValueError(f"ARTI Hub manifest is missing {key}")
        member = directory / str(record["file"])
        if member.parent.resolve() != directory.resolve() or _sha256(member) != record["sha256"]:
            raise ValueError(f"ARTI Hub {key} integrity check failed")
    payload["fingerprint"] = fingerprint
    return payload


def _infer_base_source(model: nn.Module) -> str:
    config = getattr(model, "config", None)
    source = getattr(config, "_name_or_path", None)
    return "" if source in {None, ""} else str(source)


def _portable_recall(config: Any) -> dict[str, Any]:
    return {
        "layers": list(config.paths),
        "rank": {layer.path: layer.rank for layer in config.layers},
        "slots": {layer.path: layer.slots for layer in config.layers},
        "half": {layer.path: layer.use_half for layer in config.layers},
        "recognition": {layer.path: layer.recognition_mode for layer in config.layers},
        "freeze_backbone": config.freeze_backbone,
    }


def _reject_base_weights(directory: Path) -> None:
    forbidden = sorted(path.name for path in directory.iterdir() if path.name in BASE_WEIGHT_NAMES or path.name.startswith("model-") and path.suffix == ".safetensors")
    if forbidden:
        raise ValueError(f"ARTI Hub bundle must not contain base model weights: {forbidden}")


def _distributed_wrapper(model: nn.Module) -> str | None:
    name = type(model).__name__
    return name if name in {"DistributedDataParallel", "FullyShardedDataParallel", "DataParallel"} else None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ARTIDoctorReport",
    "ARTIHubSaveResult",
    "BASE_WEIGHT_NAMES",
    "DEFAULT_ARTIFACT",
    "HUB_MANIFEST",
    "attachment_doctor",
    "load_attachment_pretrained",
    "save_attachment_pretrained",
]
