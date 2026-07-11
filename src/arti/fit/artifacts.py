"""Fit reports and adapter artifacts."""

from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .._version import __version__
from .config import load_fit_config, validate_fit_config
from .insertion import AdapterInsertionPlan
from .insertion import InsertedAdapter
from .insertion import InsertionSpec
from .profiles import AdapterProfile
from .scales import AdapterScale
from .scanner import ScanReport


@dataclass(frozen=True)
class FitTaskRecord:
    """Auditable record for one adaptation build task."""

    name: str
    status: str
    steps: int = 0
    metric_name: str | None = None
    metric_value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "steps": self.steps,
            "metric_name": self.metric_name,
            "metric_value": self.metric_value,
        }


@dataclass(frozen=True)
class BuildTaskSpec:
    """Declarative task in an ARTI adaptation build plan."""

    name: str
    kind: str
    depends_on: tuple[str, ...] = ()
    enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "depends_on": list(self.depends_on),
            "enabled": self.enabled,
        }


@dataclass(frozen=True)
class MechanismSummary:
    """Resolved ARTI mechanism settings for an adaptation build."""

    profile: str
    scale: str
    observer_phase: bool
    coord_dim: int
    coord_frame_mode: str
    virtual_recall: bool
    operator_count: int
    interface_slots: int
    recall_slots: int
    recall_steps: int
    recall_activation: str
    hidden_multiplier: float

    @classmethod
    def from_config(cls, profile: AdapterProfile, scale: AdapterScale, *, scale_name: str) -> "MechanismSummary":
        return cls(
            profile=profile.name,
            scale=scale_name,
            observer_phase=profile.observer_phase,
            coord_dim=profile.coord_dim if profile.observer_phase else 0,
            coord_frame_mode=profile.coord_frame_mode if profile.observer_phase else "none",
            virtual_recall=profile.virtual_recall or scale.recall_slots > 0 or scale.recall_steps > 0,
            operator_count=scale.operator_count,
            interface_slots=scale.interface_slots,
            recall_slots=scale.recall_slots,
            recall_steps=scale.recall_steps,
            recall_activation=scale.recall_activation,
            hidden_multiplier=scale.hidden_multiplier,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "scale": self.scale,
            "observer_phase": self.observer_phase,
            "coord_dim": self.coord_dim,
            "coord_frame_mode": self.coord_frame_mode,
            "virtual_recall": self.virtual_recall,
            "operator_count": self.operator_count,
            "interface_slots": self.interface_slots,
            "recall_slots": self.recall_slots,
            "recall_steps": self.recall_steps,
            "recall_activation": self.recall_activation,
            "hidden_multiplier": self.hidden_multiplier,
        }


@dataclass(frozen=True)
class AdapterArtifactManifest:
    """Stable metadata for an exported ARTI adapter artifact."""

    format_version: int
    package_name: str
    package_version: str
    backend: str
    include_base: bool
    adapter_key_count: int
    adapter_parameters: int
    profile: str
    scale: str
    config_fingerprint: str | None = None
    adapter_state_sha256: str | None = None
    report_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "format_version": self.format_version,
            "package_name": self.package_name,
            "package_version": self.package_version,
            "backend": self.backend,
            "include_base": self.include_base,
            "adapter_key_count": self.adapter_key_count,
            "adapter_parameters": self.adapter_parameters,
            "profile": self.profile,
            "scale": self.scale,
            "config_fingerprint": self.config_fingerprint,
            "adapter_state_sha256": self.adapter_state_sha256,
            "report_sha256": self.report_sha256,
        }


@dataclass(frozen=True)
class FitReportSummary:
    """Compact CI-friendly summary derived from an ARTI fit report."""

    candidate_count: int
    inserted_count: int
    adapter_parameters: int
    total_parameters: int
    adapter_parameter_ratio: float
    frozen_base: bool
    budget_limit: int | None
    budget_used: int
    budget_exhausted: bool
    last_loss: float | None
    last_calibration_loss: float | None
    last_validation_metric: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_count": self.candidate_count,
            "inserted_count": self.inserted_count,
            "adapter_parameters": self.adapter_parameters,
            "total_parameters": self.total_parameters,
            "adapter_parameter_ratio": self.adapter_parameter_ratio,
            "frozen_base": self.frozen_base,
            "budget_limit": self.budget_limit,
            "budget_used": self.budget_used,
            "budget_exhausted": self.budget_exhausted,
            "last_loss": self.last_loss,
            "last_calibration_loss": self.last_calibration_loss,
            "last_validation_metric": self.last_validation_metric,
        }


@dataclass(frozen=True)
class ParameterSummary:
    """Parameter audit for a patched ARTI model."""

    total_parameters: int
    trainable_parameters: int
    adapter_parameters: int
    trainable_adapter_parameters: int
    base_parameters: int
    trainable_base_parameters: int
    frozen_base: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "adapter_parameters": self.adapter_parameters,
            "trainable_adapter_parameters": self.trainable_adapter_parameters,
            "base_parameters": self.base_parameters,
            "trainable_base_parameters": self.trainable_base_parameters,
            "frozen_base": self.frozen_base,
        }


@dataclass(frozen=True)
class ForwardProfile:
    """Forward-pass runtime profile for an adapted model."""

    repeats: int
    warmup: int
    mean_ms: float
    min_ms: float
    max_ms: float
    output_shape: tuple[int, ...]
    output_dtype: str
    output_device: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "repeats": self.repeats,
            "warmup": self.warmup,
            "mean_ms": self.mean_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "output_shape": list(self.output_shape),
            "output_dtype": self.output_dtype,
            "output_device": self.output_device,
        }


@dataclass(frozen=True)
class ARTIFitReport:
    profile: str
    scale: str
    plugins: tuple[str, ...]
    plugin_details: tuple[dict[str, Any], ...]
    scanned: ScanReport
    inserted: tuple[InsertedAdapter, ...]
    frozen_base: bool
    mechanism: MechanismSummary | None = None
    insertion: InsertionSpec | None = None
    runtime_causal: bool = False
    steps: int = 0
    objective_plan: tuple[str, ...] = ()
    calibration_objective: str | None = None
    loss_history: tuple[float, ...] = ()
    calibration_history: tuple[float, ...] = ()
    validation_history: tuple[dict[str, float], ...] = ()
    task_history: tuple[FitTaskRecord, ...] = ()
    build_plan: tuple[BuildTaskSpec, ...] = ()
    parameters: ParameterSummary | None = None
    forward_profiles: tuple[ForwardProfile, ...] = ()
    insertion_plan: AdapterInsertionPlan | None = None
    fit_config: dict[str, Any] | None = None
    config_fingerprint: str | None = None
    applied_artifact: dict[str, Any] | None = None

    @property
    def adapter_parameters(self) -> int:
        return sum(adapter.parameters for adapter in self.inserted)

    @property
    def summary(self) -> FitReportSummary:
        total_parameters = self.scanned.total_parameters
        budget_limit = None if self.insertion is None else self.insertion.max_extra_params
        budget_used = self.adapter_parameters
        return FitReportSummary(
            candidate_count=len(self.scanned.candidates),
            inserted_count=len(self.inserted),
            adapter_parameters=budget_used,
            total_parameters=total_parameters,
            adapter_parameter_ratio=0.0 if total_parameters <= 0 else budget_used / total_parameters,
            frozen_base=self.frozen_base,
            budget_limit=budget_limit,
            budget_used=budget_used,
            budget_exhausted=budget_limit is not None and budget_used >= budget_limit,
            last_loss=self.loss_history[-1] if self.loss_history else None,
            last_calibration_loss=self.calibration_history[-1] if self.calibration_history else None,
            last_validation_metric=self.validation_history[-1].get("mean_metric") if self.validation_history else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "profile": self.profile,
            "scale": self.scale,
            "plugins": list(self.plugins),
            "plugin_details": list(self.plugin_details),
            "scanned": self.scanned.to_dict(),
            "inserted": [adapter.__dict__ for adapter in self.inserted],
            "frozen_base": self.frozen_base,
            "mechanism": None if self.mechanism is None else self.mechanism.to_dict(),
            "runtime_causal": self.runtime_causal,
            "insertion": None
            if self.insertion is None
            else {
                "where": list(self.insertion.where),
                "every": self.insertion.every,
                "max_adapters": self.insertion.max_adapters,
                "max_extra_params": self.insertion.max_extra_params,
            },
            "steps": self.steps,
            "objective_plan": list(self.objective_plan),
            "calibration_objective": self.calibration_objective,
            "loss_history": list(self.loss_history),
            "calibration_history": list(self.calibration_history),
            "validation_history": list(self.validation_history),
            "task_history": [task.to_dict() for task in self.task_history],
            "build_plan": [task.to_dict() for task in self.build_plan],
            "adapter_parameters": self.adapter_parameters,
            "summary": self.summary.to_dict(),
            "parameters": None if self.parameters is None else self.parameters.to_dict(),
            "forward_profiles": [profile.to_dict() for profile in self.forward_profiles],
            "insertion_plan": None if self.insertion_plan is None else self.insertion_plan.to_dict(),
            "fit_config": self.fit_config,
            "config_fingerprint": self.config_fingerprint,
            "applied_artifact": self.applied_artifact,
        }

    def to_markdown(self) -> str:
        lines = [
            "# ARTI Fit Report",
            "",
            f"Profile: `{self.profile}`",
            f"Scale: `{self.scale}`",
            f"Plugins: `{list(self.plugins)}`",
            f"Frozen base: `{self.frozen_base}`",
            f"Runtime causal: `{self.runtime_causal}`",
            f"Mechanism: `{self.mechanism.to_dict() if self.mechanism is not None else None}`",
            f"Where: `{list(self.insertion.where) if self.insertion is not None else []}`",
            f"Every: `{self.insertion.every if self.insertion is not None else 1}`",
            f"Max adapters: `{self.insertion.max_adapters if self.insertion is not None else None}`",
            f"Max extra params: `{self.insertion.max_extra_params if self.insertion is not None else None}`",
            f"Fit steps: `{self.steps}`",
            f"Objective plan: `{list(self.objective_plan)}`",
            f"Config fingerprint: `{self.config_fingerprint}`",
            f"Calibration objective: `{self.calibration_objective}`",
            f"Adapter parameters: `{self.adapter_parameters}`",
            f"Last loss: `{self.loss_history[-1] if self.loss_history else None}`",
            f"Last calibration loss: `{self.calibration_history[-1] if self.calibration_history else None}`",
            "",
            "## Summary",
            "",
            "| Metric | Value |",
            "| --- | ---: |",
        ]
        for key, value in self.summary.to_dict().items():
            lines.append(f"| `{key}` | `{value}` |")
        if self.parameters is not None:
            lines.extend(["", "## Parameters", "", "| Metric | Value |", "| --- | ---: |"])
            for key, value in self.parameters.to_dict().items():
                lines.append(f"| `{key}` | `{value}` |")
        if self.forward_profiles:
            lines.extend(["", "## Forward Profiles", "", "| Run | Mean ms | Min ms | Max ms | Output Shape | Device | DType |", "| ---: | ---: | ---: | ---: | --- | --- | --- |"])
            for index, profile in enumerate(self.forward_profiles, start=1):
                lines.append(
                    f"| {index} | {profile.mean_ms:.6f} | {profile.min_ms:.6f} | {profile.max_ms:.6f} | "
                    f"`{list(profile.output_shape)}` | `{profile.output_device}` | `{profile.output_dtype}` |"
                )
        if self.insertion_plan is not None:
            lines.extend(["", "## Insertion Plan", "", "| Module | Dim | Planned Params | Profile | Scale |", "| --- | ---: | ---: | --- | --- |"])
            for adapter in self.insertion_plan.selected:
                lines.append(f"| `{adapter.name}` | {adapter.dim} | {adapter.parameters} | `{adapter.profile}` | `{adapter.scale}` |")
            if not self.insertion_plan.selected:
                lines.append("| _none_ |  |  |  |  |")
            if self.insertion_plan.skipped_budget:
                lines.extend(["", "Budget-skipped modules:", ""])
                for adapter in self.insertion_plan.skipped_budget:
                    lines.append(f"- `{adapter.name}` ({adapter.parameters} params)")
        lines.extend(
            [
                "",
            "## Build Plan",
            "",
            "| Task | Kind | Depends On | Enabled |",
            "| --- | --- | --- | --- |",
            ]
        )
        for task in self.build_plan:
            lines.append(f"| `{task.name}` | `{task.kind}` | `{list(task.depends_on)}` | `{task.enabled}` |")
        if not self.build_plan:
            lines.append("| _none_ |  |  |  |")
        lines.extend(
            [
                "",
            "## Task History",
            "",
            "| Task | Status | Steps | Metric | Value |",
            "| --- | --- | ---: | --- | ---: |",
            ]
        )
        for task in self.task_history:
            lines.append(
                f"| `{task.name}` | `{task.status}` | {task.steps} | `{task.metric_name or ''}` | "
                f"{task.metric_value if task.metric_value is not None else ''} |"
            )
        if not self.task_history:
            lines.append("| _none_ |  |  |  |  |")
        lines.extend(
            [
                "",
                "## Scan",
                "",
                f"Candidate count: `{len(self.scanned.candidates)}`",
                f"Scanned modules: `{self.scanned.scanned_modules}`",
                f"Candidate events: `{self.scanned.candidate_events}`",
                f"Duplicate events: `{self.scanned.duplicate_events}`",
                f"Total parameters: `{self.scanned.total_parameters}`",
                f"Trainable parameters before insert: `{self.scanned.trainable_parameters}`",
                f"Device: `{self.scanned.device}`",
                f"DType: `{self.scanned.dtype}`",
                "",
                "## Inserted Adapters",
                "",
                "| Module | Dim | Adapter Params | Profile | Scale |",
                "| --- | ---: | ---: | --- | --- |",
            ]
        )
        for adapter in self.inserted:
            lines.append(f"| `{adapter.name}` | {adapter.dim} | {adapter.parameters} | `{adapter.profile}` | `{adapter.scale}` |")
        if not self.inserted:
            lines.append("| _none_ |  |  |  |  |")
        if self.validation_history:
            lines.extend(["", "## Validation", "", "| Run | Mean Metric | Batches |", "| ---: | ---: | ---: |"])
            for index, row in enumerate(self.validation_history, start=1):
                lines.append(f"| {index} | {row.get('mean_metric', 0.0):.6f} | {row.get('batches', 0.0):.0f} |")
        if self.applied_artifact is not None:
            lines.extend(["", "## Applied Artifact", "", "| Field | Value |", "| --- | --- |"])
            for key, value in self.applied_artifact.items():
                lines.append(f"| `{key}` | `{value}` |")
        return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class ARTIFitResult:
    model: nn.Module
    report: ARTIFitReport

    @property
    def adapter_count(self) -> int:
        return len(self.report.inserted)

    def export(
        self,
        path: str | Path,
        *,
        include_base: bool = False,
        build_metadata: dict[str, Any] | None = None,
    ) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        adapter_items = [
            (key, value)
            for key, value in self.model.state_dict().items()
            if ".adapter." in key or key.startswith("adapter.") or key.endswith(".output_gate") or key == "output_gate"
        ]
        adapter_items.sort(key=lambda item: (torch.is_tensor(item[1]) and item[1].numel() == 0, item[0]))
        adapter_state = dict(adapter_items)
        adapter_state_sha256 = hash_tensor_state_dict(adapter_state)
        report_dict = self.report.to_dict()
        report_sha256 = stable_json_sha256(report_dict)
        payload = {
            "manifest": AdapterArtifactManifest(
                format_version=1,
                package_name="arti",
                package_version=__version__,
                backend="torch",
                include_base=include_base,
                adapter_key_count=len(adapter_state),
                adapter_parameters=self.report.adapter_parameters,
                profile=self.report.profile,
                scale=self.report.scale,
                config_fingerprint=self.report.config_fingerprint,
                adapter_state_sha256=adapter_state_sha256,
                report_sha256=report_sha256,
            ).to_dict(),
            "report": report_dict,
            "adapter_state_dict": adapter_state,
        }
        if build_metadata is not None:
            payload["build"] = dict(build_metadata)
        if include_base:
            payload["state_dict"] = self.model.state_dict()
        torch.save(payload, target)
        return target

    def write_report(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix.lower() == ".json":
            target.write_text(json.dumps(self.report.to_dict(), indent=2), encoding="utf-8")
        else:
            target.write_text(self.report.to_markdown(), encoding="utf-8")
        return target

    def export_st(
        self,
        path: str | Path = "arti.st",
        *,
        include_base: bool = False,
        glyph_tensors: torch.Tensor | dict[str, torch.Tensor] | None = None,
        vocab_metadata: Any | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        training_state: Any | None = None,
    ):
        """Export this fitted model with the SafeTensors ``arti.st`` protocol."""

        from ..serialization import save

        return save(
            self.model,
            path,
            glyph_tensors=glyph_tensors,
            vocab_metadata=vocab_metadata,
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=training_state,
            scope="all" if include_base else "trainable",
        )


def load_artifact(path: str | Path, *, map_location: str | torch.device | None = None) -> dict[str, Any]:
    return torch.load(Path(path), map_location=map_location, weights_only=True)


def hash_tensor_state_dict(state_dict: dict[str, Any]) -> str:
    """Return a stable SHA256 fingerprint for tensor values in a state dict."""
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        value = state_dict[key]
        if not torch.is_tensor(value):
            raise ValueError(f"ARTI artifact state key {key!r} is not a tensor")
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tuple(int(dim) for dim in tensor.shape)).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(b"\0")
        digest.update(tensor.numpy().tobytes())
        digest.update(b"\0")
    return digest.hexdigest()


def stable_json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def plan_provenance_fingerprint(provenance: dict[str, Any]) -> str:
    return stable_json_sha256(provenance)


def validate_artifact_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate an exported ARTI adapter artifact payload."""
    if not isinstance(payload, dict):
        raise ValueError("ARTI artifact must be a dictionary payload")
    for key in ("manifest", "report", "adapter_state_dict"):
        if key not in payload:
            raise ValueError(f"ARTI artifact is missing required key {key!r}")
    if "build" in payload and not isinstance(payload["build"], dict):
        raise ValueError("ARTI artifact build metadata must be a dictionary")
    manifest = payload["manifest"]
    if not isinstance(manifest, dict):
        raise ValueError("ARTI artifact manifest must be a dictionary")
    if manifest.get("format_version") != 1:
        raise ValueError(f"unsupported ARTI artifact format_version={manifest.get('format_version')!r}")
    if manifest.get("package_name") != "arti":
        raise ValueError("ARTI artifact manifest package_name must be 'arti'")
    if not isinstance(manifest.get("package_version"), str) or not manifest["package_version"]:
        raise ValueError("ARTI artifact manifest package_version must be a non-empty string")
    if manifest.get("backend") != "torch":
        raise ValueError("ARTI artifact manifest backend must be 'torch'")
    if not isinstance(manifest.get("include_base"), bool):
        raise ValueError("ARTI artifact manifest include_base must be a boolean")
    for key in ("adapter_key_count", "adapter_parameters"):
        if not isinstance(manifest.get(key), int) or manifest[key] < 0:
            raise ValueError(f"ARTI artifact manifest {key} must be a non-negative integer")
    for key in ("profile", "scale"):
        if not isinstance(manifest.get(key), str) or not manifest[key]:
            raise ValueError(f"ARTI artifact manifest {key} must be a non-empty string")
    for key in ("adapter_state_sha256", "report_sha256"):
        value = manifest.get(key)
        if value is not None and (not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value)):
            raise ValueError(f"ARTI artifact manifest {key} must be a 64-character lowercase sha256 hex digest")
    config_fingerprint = manifest.get("config_fingerprint")
    if config_fingerprint is not None and (not isinstance(config_fingerprint, str) or not config_fingerprint):
        raise ValueError("ARTI artifact manifest config_fingerprint must be a non-empty string or null")
    adapter_state = payload["adapter_state_dict"]
    if not isinstance(adapter_state, dict):
        raise ValueError("ARTI artifact adapter_state_dict must be a dictionary")
    if manifest.get("adapter_key_count") != len(adapter_state):
        raise ValueError("ARTI artifact adapter_key_count does not match adapter_state_dict")
    expected_hash = manifest.get("adapter_state_sha256")
    if expected_hash is not None and expected_hash != hash_tensor_state_dict(adapter_state):
        raise ValueError("ARTI artifact adapter_state_sha256 does not match adapter_state_dict")
    report = payload["report"]
    if not isinstance(report, dict):
        raise ValueError("ARTI artifact report must be a dictionary")
    inserted = report.get("inserted")
    if not isinstance(inserted, list):
        raise ValueError("ARTI artifact report.inserted must be a list")
    summary = report.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("ARTI artifact report.summary must be a dictionary")
    if not isinstance(summary.get("inserted_count"), int) or summary["inserted_count"] < 0:
        raise ValueError("ARTI artifact report.summary.inserted_count must be a non-negative integer")
    if summary["inserted_count"] != len(inserted):
        raise ValueError("ARTI artifact report.summary.inserted_count does not match report.inserted")
    if not isinstance(report.get("adapter_parameters"), int) or report["adapter_parameters"] < 0:
        raise ValueError("ARTI artifact report.adapter_parameters must be a non-negative integer")
    if summary.get("adapter_parameters") != report.get("adapter_parameters"):
        raise ValueError("ARTI artifact report.summary.adapter_parameters does not match report.adapter_parameters")
    expected_report_hash = manifest.get("report_sha256")
    if expected_report_hash is not None and expected_report_hash != stable_json_sha256(report):
        raise ValueError("ARTI artifact report_sha256 does not match report")
    if manifest.get("profile") != report.get("profile"):
        raise ValueError("ARTI artifact manifest profile does not match report")
    if manifest.get("scale") != report.get("scale"):
        raise ValueError("ARTI artifact manifest scale does not match report")
    if manifest.get("adapter_parameters") != report.get("adapter_parameters"):
        raise ValueError("ARTI artifact manifest adapter_parameters does not match report")
    if manifest.get("config_fingerprint") != report.get("config_fingerprint"):
        raise ValueError("ARTI artifact manifest config_fingerprint does not match report")
    if manifest.get("include_base") and "state_dict" not in payload:
        raise ValueError("ARTI artifact manifest declares include_base=True but state_dict is missing")
    return payload


def validate_artifact(path: str | Path, *, map_location: str | torch.device | None = None) -> dict[str, Any]:
    """Load and validate an ARTI adapter artifact."""
    return validate_artifact_payload(load_artifact(path, map_location=map_location))


def validate_plan_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate an exported ARTI fit-plan JSON payload."""
    if not isinstance(payload, dict):
        raise ValueError("ARTI fit plan must be a dictionary payload")
    if payload.get("format_version") != 1:
        raise ValueError(f"unsupported ARTI fit plan format_version={payload.get('format_version')!r}")
    if payload.get("package_name") != "arti":
        raise ValueError("ARTI fit plan package_name must be 'arti'")
    if payload.get("kind") != "fit-plan":
        raise ValueError("ARTI fit plan kind must be 'fit-plan'")
    provenance = payload.get("provenance")
    if provenance is not None:
        if not isinstance(provenance, dict):
            raise ValueError("ARTI fit plan provenance must be a dictionary")
        if "model" in provenance and not isinstance(provenance["model"], str):
            raise ValueError("ARTI fit plan provenance.model must be a string")
        if "sample_shape" in provenance and provenance["sample_shape"] is not None:
            if not isinstance(provenance["sample_shape"], list) or not all(isinstance(dim, int) and dim > 0 for dim in provenance["sample_shape"]):
                raise ValueError("ARTI fit plan provenance.sample_shape must be a list of positive integers")
        expected_provenance_fingerprint = payload.get("provenance_fingerprint")
        if expected_provenance_fingerprint is not None and expected_provenance_fingerprint != plan_provenance_fingerprint(provenance):
            raise ValueError("ARTI fit plan provenance_fingerprint does not match provenance")
    report = payload.get("report")
    if not isinstance(report, dict):
        raise ValueError("ARTI fit plan report must be a dictionary")
    if not isinstance(report.get("scanned"), dict):
        raise ValueError("ARTI fit plan report.scanned must be a dictionary")
    if not isinstance(report.get("build_plan"), list):
        raise ValueError("ARTI fit plan report.build_plan must be a list")
    if not isinstance(report.get("objective_plan"), list):
        raise ValueError("ARTI fit plan report.objective_plan must be a list")
    insertion_plan = report.get("insertion_plan")
    if not isinstance(insertion_plan, dict):
        raise ValueError("ARTI fit plan report.insertion_plan must be a dictionary")
    for key in ("selected", "skipped_budget"):
        if not isinstance(insertion_plan.get(key), list):
            raise ValueError(f"ARTI fit plan insertion_plan.{key} must be a list")
    if not isinstance(insertion_plan.get("spec"), dict):
        raise ValueError("ARTI fit plan insertion_plan.spec must be a dictionary")
    if insertion_plan.get("adapter_parameters") != sum(int(row.get("parameters", 0)) for row in insertion_plan["selected"]):
        raise ValueError("ARTI fit plan adapter_parameters does not match selected adapters")
    spec = insertion_plan["spec"]
    max_adapters = spec.get("max_adapters")
    if max_adapters is not None and len(insertion_plan["selected"]) > int(max_adapters):
        raise ValueError("ARTI fit plan selected adapters exceed max_adapters")
    max_extra_params = spec.get("max_extra_params")
    if max_extra_params is not None and int(insertion_plan["adapter_parameters"]) > int(max_extra_params):
        raise ValueError("ARTI fit plan adapter_parameters exceed max_extra_params")
    if report.get("fit_config") is not None:
        expected = validate_fit_config(report["fit_config"]).fingerprint
        if report.get("config_fingerprint") != expected:
            raise ValueError("ARTI fit plan config_fingerprint does not match fit_config")
    return payload


def validate_plan(path: str | Path) -> dict[str, Any]:
    """Load and validate an ARTI fit-plan JSON artifact."""
    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    return validate_plan_payload(payload)


def create_task_graph_payload(*, command_kind: str, task_graph: dict[str, Any]) -> dict[str, Any]:
    """Create a machine-readable ARTI CLI task graph payload."""

    return validate_task_graph_payload(
        {
            "format_version": 1,
            "package_name": "arti",
            "kind": "task-graph",
            "command_kind": command_kind,
            "task_graph": task_graph,
        }
    )


def write_task_graph_artifact(path: str | Path, *, command_kind: str, task_graph: dict[str, Any]) -> Path:
    """Write a validated ARTI task graph JSON artifact."""

    target = Path(path)
    payload = create_task_graph_payload(command_kind=command_kind, task_graph=task_graph)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return target


def validate_task_graph_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Validate an ARTI CLI task graph JSON payload."""

    if not isinstance(payload, dict):
        raise ValueError("ARTI task graph must be a dictionary payload")
    if payload.get("format_version") != 1:
        raise ValueError(f"unsupported ARTI task graph format_version={payload.get('format_version')!r}")
    if payload.get("package_name") != "arti":
        raise ValueError("ARTI task graph package_name must be 'arti'")
    if payload.get("kind") != "task-graph":
        raise ValueError("ARTI task graph kind must be 'task-graph'")
    if payload.get("command_kind") not in {"build", "apply"}:
        raise ValueError("ARTI task graph command_kind must be 'build' or 'apply'")
    graph = payload.get("task_graph")
    if not isinstance(graph, dict):
        raise ValueError("ARTI task graph task_graph must be a dictionary")
    tasks = graph.get("tasks")
    artifacts = graph.get("artifacts")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("ARTI task graph tasks must be a non-empty list")
    if not isinstance(artifacts, dict):
        raise ValueError("ARTI task graph artifacts must be a dictionary")
    for key, value in artifacts.items():
        if not isinstance(key, str) or not key:
            raise ValueError("ARTI task graph artifact keys must be non-empty strings")
        if value is not None and not isinstance(value, str):
            raise ValueError("ARTI task graph artifact values must be strings or null")
    names: set[str] = set()
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError("ARTI task graph task entries must be dictionaries")
        name = task.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("ARTI task graph task.name must be a non-empty string")
        if name in names:
            raise ValueError("ARTI task graph task names must be unique")
        names.add(name)
        depends_on = task.get("depends_on")
        if not isinstance(depends_on, list):
            raise ValueError("ARTI task graph task.depends_on must be a list")
        for dependency in depends_on:
            if dependency not in names:
                raise ValueError("ARTI task graph task dependency must reference an earlier task")
    return payload


def validate_task_graph(path: str | Path) -> dict[str, Any]:
    """Load and validate an ARTI CLI task graph JSON artifact."""

    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    return validate_task_graph_payload(payload)


def create_build_lock(
    path: str | Path,
    *,
    artifact: str | Path,
    plan: str | Path | None = None,
    config: str | Path | None = None,
    map_location: str | torch.device | None = None,
) -> Path:
    """Write a lockfile that binds reviewed ARTI build artifacts together."""

    lock_path = Path(path)
    artifact_path = resolve_input_path(artifact)
    plan_path = None if plan is None else resolve_input_path(plan)
    config_path = None if config is None else resolve_input_path(config)
    artifact_payload = validate_artifact(artifact_path, map_location=map_location)
    artifact_manifest = artifact_payload["manifest"]
    artifact_report = artifact_payload["report"]
    plan_payload = None if plan_path is None else validate_plan(plan_path)
    config_fingerprint = None
    if config_path is not None:
        config_fingerprint = load_fit_config(config_path).fingerprint
    payload = {
        "format_version": 1,
        "package_name": "arti",
        "kind": "build-lock",
        "artifact": {
            "path": lock_relative_path(lock_path, artifact_path),
            "adapter_state_sha256": artifact_manifest.get("adapter_state_sha256"),
            "report_sha256": artifact_manifest.get("report_sha256"),
            "config_fingerprint": artifact_manifest.get("config_fingerprint"),
            "adapter_parameters": artifact_manifest.get("adapter_parameters"),
            "inserted_count": artifact_report.get("summary", {}).get("inserted_count"),
            "adapter_key_count": artifact_manifest.get("adapter_key_count"),
            "profile": artifact_manifest.get("profile"),
            "scale": artifact_manifest.get("scale"),
            "mechanism": artifact_report.get("mechanism"),
            "runtime": (artifact_report.get("fit_config") or {}).get("runtime"),
        },
        "plan": None
        if plan_payload is None
        else {
            "path": lock_relative_path(lock_path, plan_path),
            "provenance_fingerprint": plan_payload.get("provenance_fingerprint"),
            "config_fingerprint": plan_payload.get("report", {}).get("config_fingerprint"),
            "planned_count": len(plan_payload.get("report", {}).get("insertion_plan", {}).get("selected", [])),
            "adapter_parameters": plan_payload.get("report", {}).get("insertion_plan", {}).get("adapter_parameters"),
        },
        "config": None if config_path is None else {"path": lock_relative_path(lock_path, config_path), "config_fingerprint": config_fingerprint},
    }
    if artifact_payload.get("build") is not None:
        payload["artifact"]["build"] = artifact_payload["build"]
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return lock_path


def validate_build_lock(path: str | Path, *, map_location: str | torch.device | None = None) -> dict[str, Any]:
    """Validate that a build lock still matches its referenced ARTI artifacts."""

    lock_path = Path(path)
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ARTI build lock must be a dictionary payload")
    if payload.get("format_version") != 1:
        raise ValueError(f"unsupported ARTI build lock format_version={payload.get('format_version')!r}")
    if payload.get("package_name") != "arti":
        raise ValueError("ARTI build lock package_name must be 'arti'")
    if payload.get("kind") != "build-lock":
        raise ValueError("ARTI build lock kind must be 'build-lock'")
    artifact_lock = payload.get("artifact")
    if not isinstance(artifact_lock, dict) or not artifact_lock.get("path"):
        raise ValueError("ARTI build lock artifact must include a path")
    artifact_path = resolve_locked_path(lock_path, artifact_lock["path"])
    artifact_payload = validate_artifact(artifact_path, map_location=map_location)
    manifest = artifact_payload["manifest"]
    for key in ("adapter_state_sha256", "report_sha256", "config_fingerprint", "adapter_parameters", "adapter_key_count", "profile", "scale"):
        if artifact_lock.get(key) != manifest.get(key):
            raise ValueError(f"ARTI build lock artifact.{key} does not match artifact")
    expected_inserted = artifact_payload.get("report", {}).get("summary", {}).get("inserted_count")
    if artifact_lock.get("inserted_count") != expected_inserted:
        raise ValueError("ARTI build lock artifact.inserted_count does not match artifact")
    if artifact_lock.get("mechanism") != artifact_payload.get("report", {}).get("mechanism"):
        raise ValueError("ARTI build lock artifact.mechanism does not match artifact")
    if artifact_lock.get("runtime") != (artifact_payload.get("report", {}).get("fit_config") or {}).get("runtime"):
        raise ValueError("ARTI build lock artifact.runtime does not match artifact")
    if "build" in artifact_lock and artifact_lock.get("build") != artifact_payload.get("build"):
        raise ValueError("ARTI build lock artifact.build does not match artifact")
    plan_lock = payload.get("plan")
    if plan_lock is not None:
        if not isinstance(plan_lock, dict) or not plan_lock.get("path"):
            raise ValueError("ARTI build lock plan must include a path")
        plan_payload = validate_plan(resolve_locked_path(lock_path, plan_lock["path"]))
        insertion_plan = plan_payload.get("report", {}).get("insertion_plan", {})
        expected_plan = {
            "provenance_fingerprint": plan_payload.get("provenance_fingerprint"),
            "config_fingerprint": plan_payload.get("report", {}).get("config_fingerprint"),
            "planned_count": len(insertion_plan.get("selected", [])),
            "adapter_parameters": insertion_plan.get("adapter_parameters"),
        }
        for key, value in expected_plan.items():
            if plan_lock.get(key) != value:
                raise ValueError(f"ARTI build lock plan.{key} does not match plan")
    config_lock = payload.get("config")
    if config_lock is not None:
        if not isinstance(config_lock, dict) or not config_lock.get("path"):
            raise ValueError("ARTI build lock config must include a path")
        config_path = resolve_locked_path(lock_path, config_lock["path"])
        expected_fingerprint = load_fit_config(config_path).fingerprint
        if config_lock.get("config_fingerprint") != expected_fingerprint:
            raise ValueError("ARTI build lock config_fingerprint does not match config")
        if manifest.get("config_fingerprint") != expected_fingerprint:
            raise ValueError("ARTI build lock config_fingerprint does not match artifact")
    return payload


def create_deployment_manifest(
    path: str | Path,
    *,
    lock: str | Path,
    artifact: str | Path,
    applied_report: str | Path,
    state_dict: str | Path,
    map_location: str | torch.device | None = None,
) -> Path:
    """Write a deployment manifest for a locked ARTI adapter application."""

    manifest_path = Path(path)
    lock_path = resolve_input_path(lock)
    artifact_path = resolve_input_path(artifact)
    applied_report_path = resolve_input_path(applied_report)
    state_dict_path = resolve_input_path(state_dict)
    lock_payload = validate_build_lock(lock_path, map_location=map_location)
    artifact_payload = validate_artifact(artifact_path, map_location=map_location)
    artifact_report = artifact_payload["report"]
    applied_report_payload = json.loads(applied_report_path.read_text(encoding="utf-8"))
    state_dict_payload = torch.load(state_dict_path, map_location=map_location, weights_only=True)
    if not isinstance(state_dict_payload, dict):
        raise ValueError("ARTI deployment manifest state_dict must be a dictionary")
    payload = {
        "format_version": 1,
        "package_name": "arti",
        "kind": "deployment-manifest",
        "lock": {
            "path": lock_relative_path(manifest_path, lock_path),
            "artifact_report_sha256": lock_payload["artifact"].get("report_sha256"),
        },
        "artifact": {
            "path": lock_relative_path(manifest_path, artifact_path),
            "adapter_state_sha256": artifact_payload["manifest"].get("adapter_state_sha256"),
            "report_sha256": artifact_payload["manifest"].get("report_sha256"),
            "config_fingerprint": artifact_payload["manifest"].get("config_fingerprint"),
            "adapter_parameters": artifact_payload["manifest"].get("adapter_parameters"),
            "inserted_count": artifact_report.get("summary", {}).get("inserted_count"),
            "adapter_key_count": artifact_payload["manifest"].get("adapter_key_count"),
            "profile": artifact_payload["manifest"].get("profile"),
            "scale": artifact_payload["manifest"].get("scale"),
            "mechanism": artifact_report.get("mechanism"),
            "runtime": (artifact_report.get("fit_config") or {}).get("runtime"),
        },
        "applied_report": {
            "path": lock_relative_path(manifest_path, applied_report_path),
            "report_sha256": stable_json_sha256(applied_report_payload),
        },
        "state_dict": {
            "path": lock_relative_path(manifest_path, state_dict_path),
            "state_dict_sha256": hash_tensor_state_dict(state_dict_payload),
            "state_key_count": len(state_dict_payload),
        },
    }
    if artifact_payload.get("build") is not None:
        payload["artifact"]["build"] = artifact_payload["build"]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return manifest_path


def validate_deployment_manifest(path: str | Path, *, map_location: str | torch.device | None = None) -> dict[str, Any]:
    """Validate an ARTI deployment manifest and its referenced files."""

    manifest_path = Path(path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("ARTI deployment manifest must be a dictionary payload")
    if payload.get("format_version") != 1:
        raise ValueError(f"unsupported ARTI deployment manifest format_version={payload.get('format_version')!r}")
    if payload.get("package_name") != "arti":
        raise ValueError("ARTI deployment manifest package_name must be 'arti'")
    if payload.get("kind") != "deployment-manifest":
        raise ValueError("ARTI deployment manifest kind must be 'deployment-manifest'")
    lock_entry = require_manifest_entry(payload, "lock")
    artifact_entry = require_manifest_entry(payload, "artifact")
    applied_report_entry = require_manifest_entry(payload, "applied_report")
    state_dict_entry = require_manifest_entry(payload, "state_dict")
    lock_payload = validate_build_lock(resolve_locked_path(manifest_path, lock_entry["path"]), map_location=map_location)
    if lock_entry.get("artifact_report_sha256") != lock_payload["artifact"].get("report_sha256"):
        raise ValueError("ARTI deployment manifest lock artifact_report_sha256 does not match lock")
    artifact_path = resolve_locked_path(manifest_path, artifact_entry["path"])
    artifact_payload = validate_artifact(artifact_path, map_location=map_location)
    for key in ("adapter_state_sha256", "report_sha256", "config_fingerprint", "adapter_parameters", "adapter_key_count", "profile", "scale"):
        if artifact_entry.get(key) != artifact_payload["manifest"].get(key):
            raise ValueError(f"ARTI deployment manifest artifact.{key} does not match artifact")
        if artifact_entry.get(key) != lock_payload["artifact"].get(key):
            raise ValueError(f"ARTI deployment manifest artifact.{key} does not match lock")
    if artifact_entry.get("mechanism") != artifact_payload.get("report", {}).get("mechanism"):
        raise ValueError("ARTI deployment manifest artifact.mechanism does not match artifact")
    if artifact_entry.get("mechanism") != lock_payload["artifact"].get("mechanism"):
        raise ValueError("ARTI deployment manifest artifact.mechanism does not match lock")
    if artifact_entry.get("runtime") != (artifact_payload.get("report", {}).get("fit_config") or {}).get("runtime"):
        raise ValueError("ARTI deployment manifest artifact.runtime does not match artifact")
    if artifact_entry.get("runtime") != lock_payload["artifact"].get("runtime"):
        raise ValueError("ARTI deployment manifest artifact.runtime does not match lock")
    if "build" in artifact_entry:
        if artifact_entry.get("build") != artifact_payload.get("build"):
            raise ValueError("ARTI deployment manifest artifact.build does not match artifact")
        if artifact_entry.get("build") != lock_payload["artifact"].get("build"):
            raise ValueError("ARTI deployment manifest artifact.build does not match lock")
    expected_inserted = artifact_payload.get("report", {}).get("summary", {}).get("inserted_count")
    if artifact_entry.get("inserted_count") != expected_inserted:
        raise ValueError("ARTI deployment manifest artifact.inserted_count does not match artifact")
    if artifact_entry.get("inserted_count") != lock_payload["artifact"].get("inserted_count"):
        raise ValueError("ARTI deployment manifest artifact.inserted_count does not match lock")
    applied_report_path = resolve_locked_path(manifest_path, applied_report_entry["path"])
    applied_report_payload = json.loads(applied_report_path.read_text(encoding="utf-8"))
    if applied_report_entry.get("report_sha256") != stable_json_sha256(applied_report_payload):
        raise ValueError("ARTI deployment manifest applied_report.report_sha256 does not match applied report")
    applied_artifact = applied_report_payload.get("applied_artifact")
    if isinstance(applied_artifact, dict):
        applied_adapter_sha256 = applied_artifact.get("adapter_state_sha256")
        if applied_adapter_sha256 is not None and applied_adapter_sha256 != artifact_entry.get("adapter_state_sha256"):
            raise ValueError("ARTI deployment manifest applied_report adapter_state_sha256 does not match artifact")
        applied_artifact_path = applied_artifact.get("path")
        if applied_artifact_path is not None and Path(applied_artifact_path).resolve() != artifact_path.resolve():
            raise ValueError("ARTI deployment manifest applied_report artifact path does not match artifact")
    state_dict_payload = torch.load(resolve_locked_path(manifest_path, state_dict_entry["path"]), map_location=map_location, weights_only=True)
    if not isinstance(state_dict_payload, dict):
        raise ValueError("ARTI deployment manifest state_dict must be a dictionary")
    if state_dict_entry.get("state_dict_sha256") != hash_tensor_state_dict(state_dict_payload):
        raise ValueError("ARTI deployment manifest state_dict_sha256 does not match state_dict")
    if state_dict_entry.get("state_key_count") != len(state_dict_payload):
        raise ValueError("ARTI deployment manifest state_key_count does not match state_dict")
    return payload


def require_manifest_entry(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict) or not value.get("path"):
        raise ValueError(f"ARTI deployment manifest {key} must include a path")
    return value


def resolve_locked_path(lock_path: Path, value: str) -> Path:
    target = Path(value)
    if target.is_absolute():
        return target
    return lock_path.parent / target


def resolve_input_path(value: str | Path) -> Path:
    target = Path(value)
    if target.is_absolute():
        return target
    return Path.cwd() / target


def lock_relative_path(lock_path: Path, target: Path) -> str:
    lock_dir = lock_path.parent if lock_path.parent != Path("") else Path(".")
    lock_dir_abs = lock_dir.resolve()
    target_abs = target.resolve()
    try:
        return str(target_abs.relative_to(lock_dir_abs))
    except ValueError:
        try:
            return str(Path("..") / target_abs.relative_to(lock_dir_abs.parent))
        except ValueError:
            return str(target_abs)
