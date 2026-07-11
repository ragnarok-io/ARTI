"""Declarative configuration and reproducible locks for ``ARTI.attach``."""

from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from ._version import __version__


@dataclass(frozen=True)
class ARTIAttachTrainingConfig:
    engine: str = "torch"
    objective: str = "recall_alignment"
    learning_rate: float = 1e-3
    steps: int = 1
    gradient_accumulation_steps: int = 1
    mixed_precision: str = "no"
    corruption_probability: float = 0.15

    def __post_init__(self) -> None:
        if self.engine not in {"torch", "transformers", "accelerate"}:
            raise ValueError("training.engine must be 'torch', 'transformers', or 'accelerate'")
        if self.objective not in {"recall_alignment", "model_loss"}:
            raise ValueError("training.objective must be 'recall_alignment' or 'model_loss'")
        if self.learning_rate <= 0 or self.steps <= 0 or self.gradient_accumulation_steps <= 0:
            raise ValueError("training learning_rate, steps, and gradient_accumulation_steps must be positive")
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError("training.mixed_precision must be 'no', 'fp16', or 'bf16'")
        if not 0 <= self.corruption_probability < 1:
            raise ValueError("training.corruption_probability must be in [0, 1)")


@dataclass(frozen=True)
class ARTIAttachConfig:
    recall: Mapping[str, Any]
    training: ARTIAttachTrainingConfig = ARTIAttachTrainingConfig()
    format_version: int = 1
    source_path: Path | None = None
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.format_version != 1:
            raise ValueError("unsupported ARTI attachment config format_version")
        if not self.recall:
            raise ValueError("attachment recall configuration must not be empty")

    def to_dict(self, *, include_source: bool = False) -> dict[str, Any]:
        payload = {
            "format_version": self.format_version,
            "recall": _json_value(dict(self.recall)),
            "training": asdict(self.training),
        }
        if include_source:
            payload["source_path"] = None if self.source_path is None else str(self.source_path)
            payload["source_sha256"] = self.source_sha256
        return payload


def load_attach_config(path: str | Path) -> ARTIAttachConfig:
    target = Path(path)
    if target.suffix.lower() != ".toml":
        raise ValueError("ARTI attachment config must end in .toml")
    raw = target.read_bytes()
    payload = tomllib.loads(raw.decode("utf-8"))
    root = payload.get("arti", {})
    format_version = int(root.get("format_version", 1))
    recall = payload.get("recall")
    if not isinstance(recall, Mapping):
        raise ValueError("attachment TOML requires a [recall] table")
    normalized = dict(recall)
    if "layers" in normalized and isinstance(normalized["layers"], list):
        normalized["layers"] = tuple(str(value) for value in normalized["layers"])
    training_payload = payload.get("training", {})
    if not isinstance(training_payload, Mapping):
        raise ValueError("[training] must be a TOML table")
    return ARTIAttachConfig(
        recall=normalized,
        training=ARTIAttachTrainingConfig(**training_payload),
        format_version=format_version,
        source_path=target.resolve(),
        source_sha256=hashlib.sha256(raw).hexdigest(),
    )


def attach_config_from_dict(payload: Mapping[str, Any]) -> ARTIAttachConfig:
    training = payload.get("training", {})
    if not isinstance(training, Mapping):
        raise ValueError("attachment config training value must be a mapping")
    recall = payload.get("recall")
    if not isinstance(recall, Mapping):
        raise ValueError("attachment config recall value must be a mapping")
    source = payload.get("source_path")
    return ARTIAttachConfig(
        recall=dict(recall),
        training=ARTIAttachTrainingConfig(**training),
        format_version=int(payload.get("format_version", 1)),
        source_path=None if source is None else Path(str(source)),
        source_sha256=payload.get("source_sha256"),
    )


def write_attach_config(path: str | Path, config: ARTIAttachConfig | None = None) -> Path:
    target = Path(path)
    if target.suffix.lower() != ".toml":
        raise ValueError("ARTI attachment config must end in .toml")
    value = config or ARTIAttachConfig(recall={"layers": ["model.layers.*"], "rank": 16, "slots": 8})
    recall = dict(value.recall)
    lines = ["[arti]", f"format_version = {value.format_version}", "", "[recall]"]
    for key, item in recall.items():
        lines.append(f"{key} = {_toml_value(item)}")
    lines.extend(["", "[training]"])
    for key, item in asdict(value.training).items():
        lines.append(f"{key} = {_toml_value(item)}")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return target


def write_attach_lock(
    path: str | Path,
    *,
    config: ARTIAttachConfig,
    resolved_recall: Mapping[str, Any],
    host_structure: str,
) -> Path:
    target = Path(path)
    payload = {
        "format": "arti.attach.lock",
        "format_version": 1,
        "arti_version": __version__,
        "config_sha256": config.source_sha256 or _sha256_json(config.to_dict()),
        "resolved_recall": _json_value(dict(resolved_recall)),
        "host_structure": host_structure,
    }
    payload["fingerprint"] = _sha256_json(payload)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def validate_attach_lock(
    path: str | Path,
    *,
    config: ARTIAttachConfig,
    resolved_recall: Mapping[str, Any],
    host_structure: str,
) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    fingerprint = payload.pop("fingerprint", None)
    if fingerprint != _sha256_json(payload):
        raise ValueError("ARTI attachment lock fingerprint is invalid")
    expected = {
        "config_sha256": config.source_sha256 or _sha256_json(config.to_dict()),
        "resolved_recall": _json_value(dict(resolved_recall)),
        "host_structure": host_structure,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"ARTI attachment lock {key} does not match")
    payload["fingerprint"] = fingerprint
    return payload


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if isinstance(value, Mapping):
        return "{ " + ", ".join(f"{json.dumps(str(key))} = {_toml_value(item)}" for key, item in value.items()) + " }"
    return json.dumps(str(value), ensure_ascii=False)


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"attachment config value is not serializable: {type(value).__name__}")


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "ARTIAttachConfig",
    "ARTIAttachTrainingConfig",
    "attach_config_from_dict",
    "load_attach_config",
    "validate_attach_lock",
    "write_attach_config",
    "write_attach_lock",
]
