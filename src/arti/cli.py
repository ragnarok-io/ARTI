"""Command line interface for ARTI build artifacts."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .fit import MechanismOverrides, MechanismSummary, apply_adapter, capabilities, check_fit_config_schema, check_generated_docs, check_task_graph_schema, create_build_lock, create_deployment_manifest, doctor_report, fit, generate_capabilities_markdown, generate_fit_config_schema_json, generate_task_graph_schema_json, list_plugins, list_profiles, list_scales, load_fit_config, plan_provenance_fingerprint, resolve_fit_config_mechanism, validate_artifact, validate_build_lock, validate_deployment_manifest, validate_plan, validate_task_graph, validate_task_graph_payload, write_doctor_report, write_fit_config_schema, write_fit_config_template, write_generated_docs, write_task_graph_artifact, write_task_graph_schema
from .fit.artifacts import hash_tensor_state_dict
from .pretrained_cli import pretrained_cli_report


MECHANISM_FIELDS = {
    "profile",
    "scale",
    "observer_phase",
    "coord_dim",
    "coord_frame_mode",
    "virtual_recall",
    "operator_count",
    "interface_slots",
    "recall_slots",
    "recall_steps",
    "recall_activation",
    "hidden_multiplier",
}

RUNTIME_FIELD_NAMES = {
    "mask_key",
    "coord_key",
    "observer_coord_key",
    "frame_operators_key",
}

MAX_SAMPLE_ELEMENTS = 16_777_216


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="arti", description="ARTI adaptation build utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="Print ARTI fit capabilities as JSON.")
    inspect_parser.add_argument(
        "section",
        nargs="?",
        choices=("all", "profiles", "scales", "plugins"),
        default="all",
        help="Capability section to print.",
    )
    inspect_parser.add_argument("--phases", type=int, default=None, help="Observer-phase coord dimension preview.")

    doctor_parser = subparsers.add_parser("doctor", help="Inspect local ARTI backend, CUDA, and optional JAX readiness.")
    doctor_parser.add_argument(
        "--allow-cpu-torch",
        action="store_true",
        help="Report visible NVIDIA hardware with a CPU-only PyTorch runtime as a warning instead of a failure.",
    )
    doctor_parser.add_argument("--output", type=Path, default=None, help="Write the doctor report to .json or .md.")
    doctor_parser.add_argument("--require-cuda-smoke", action="store_true", help="Fail unless the CUDA smoke check passed.")
    doctor_parser.add_argument("--require-jax-smoke", action="store_true", help="Fail unless the JAX smoke check passed.")

    docs_parser = subparsers.add_parser("docs", help="Generate or check source-backed ARTI docs.")
    docs_subparsers = docs_parser.add_subparsers(dest="kind", required=True)
    docs_generate_parser = docs_subparsers.add_parser("generate", help="Generate source-backed reference docs.")
    docs_generate_parser.add_argument("--output", type=Path, default=Path("docs/reference/capabilities.md"), help="Generated capabilities reference path.")
    docs_generate_parser.add_argument("--phases", type=int, default=16, help="Observer-phase preview coordinate dimension.")
    docs_check_parser = docs_subparsers.add_parser("check", help="Check generated source-backed reference docs.")
    docs_check_parser.add_argument("--output", type=Path, default=Path("docs/reference/capabilities.md"), help="Generated capabilities reference path.")
    docs_check_parser.add_argument("--phases", type=int, default=16, help="Observer-phase preview coordinate dimension.")

    schema_parser = subparsers.add_parser("schema", help="Generate or check machine-readable ARTI schemas.")
    schema_subparsers = schema_parser.add_subparsers(dest="kind", required=True)
    schema_fit_parser = schema_subparsers.add_parser("fit-config", help="Generate or check the ARTI fit config JSON Schema.")
    schema_fit_parser.add_argument("action", choices=("generate", "check"), help="Generate or check the schema file.")
    schema_fit_parser.add_argument("--output", type=Path, default=Path("docs/reference/fit-config.schema.json"), help="Generated fit config schema path.")
    schema_task_parser = schema_subparsers.add_parser("task-graph", help="Generate or check the ARTI task graph JSON Schema.")
    schema_task_parser.add_argument("action", choices=("generate", "check"), help="Generate or check the schema file.")
    schema_task_parser.add_argument("--output", type=Path, default=Path("docs/reference/task-graph.schema.json"), help="Generated task graph schema path.")

    pretrained_parser = subparsers.add_parser("pretrained", help="Run declarative pretrained-model workflows.")
    pretrained_subparsers = pretrained_parser.add_subparsers(dest="pretrained_action", required=True)
    pretrained_doctor = pretrained_subparsers.add_parser("doctor", help="Report pretrained provider availability.")
    pretrained_doctor.add_argument("--provider", choices=("torch", "transformers", "peft", "diffusers"), default=None)
    pretrained_validate = pretrained_subparsers.add_parser("validate-lock", help="Validate a pretrained reproducibility lock.")
    pretrained_validate.add_argument("--lock", type=Path, required=True)
    for action in ("scan", "plan", "apply", "fit", "export"):
        action_parser = pretrained_subparsers.add_parser(action, help=f"Run pretrained {action} from JSON/TOML config.")
        action_parser.add_argument("config", type=Path)
        action_parser.add_argument("--plan", type=Path, default=None, help="Use an existing reviewed ARTIPlan.")
        action_parser.add_argument("--output", type=Path, default=None, help="Scan report or plan output path.")
        action_parser.add_argument("--weights", type=Path, default=None, help="Destination arti.st path.")
        action_parser.add_argument("--include-base", action="store_true", help="Include frozen base weights in arti.st.")

    plan_build_parser = subparsers.add_parser("plan", help="Create a dry-run ARTI fit plan from an importable model factory.")
    plan_build_parser.add_argument("model", help="Import path for an nn.Module or factory, such as package.module:make_model.")
    plan_build_parser.add_argument("output", type=Path, help="Destination .json or .md plan report.")
    plan_build_parser.add_argument("--model-kwargs-json", type=Path, default=None, help="JSON keyword arguments passed to the model factory.")
    sample_group = plan_build_parser.add_mutually_exclusive_group(required=True)
    sample_group.add_argument("--sample-shape", help="Comma-separated sample tensor shape, for example 2,4 or 2,16,768.")
    sample_group.add_argument("--sample-json", type=Path, help="JSON tensor-batch schema for dict samples.")
    plan_build_parser.add_argument("--config", type=Path, default=None, help="Optional ARTI fit config JSON/TOML.")
    plan_build_parser.add_argument("--target-modules", default=None, help="Comma-separated module names or glob patterns.")
    plan_build_parser.add_argument("--profile", default="latent-adapt", help="Profile preset or alias.")
    plan_build_parser.add_argument("--phases", type=int, default=None, help="Observer-phase coordinate dimension.")
    plan_build_parser.add_argument("--scale", default="small", help="Scale preset.")
    plan_build_parser.add_argument("--mechanism-coord-dim", type=int, default=None, help="Override resolved mechanism coord_dim.")
    plan_build_parser.add_argument("--mechanism-coord-frame-mode", choices=("none", "paired_rotation", "operator_bank"), default=None, help="Override resolved mechanism coordinate frame mode.")
    plan_build_parser.add_argument("--mechanism-observer-phase", action="store_true", help="Force observer phase on for this plan.")
    plan_build_parser.add_argument("--mechanism-virtual-recall", action="store_true", help="Force virtual recall on for this plan.")
    plan_build_parser.add_argument("--mechanism-operator-count", type=int, default=None, help="Override dynamic operator count.")
    plan_build_parser.add_argument("--mechanism-interface-slots", type=int, default=None, help="Override virtual interface slot count.")
    plan_build_parser.add_argument("--mechanism-recall-slots", type=int, default=None, help="Override private recall slot count.")
    plan_build_parser.add_argument("--mechanism-recall-steps", type=int, default=None, help="Override private recall update steps.")
    plan_build_parser.add_argument("--mechanism-recall-activation", choices=("half", "none"), default=None, help="Override Recall trace-survival activation.")
    plan_build_parser.add_argument("--mechanism-hidden-multiplier", type=float, default=None, help="Override adapter hidden width multiplier.")
    plan_build_parser.add_argument("--max-adapters", type=int, default=None)
    plan_build_parser.add_argument("--max-extra-params", default=None)
    plan_build_parser.add_argument("--causal", action="store_true")
    plan_build_parser.add_argument("--mask-key", default=None, help="Batch field name used as ARTI mask/visibility input.")
    plan_build_parser.add_argument("--coord-key", default=None, help="Batch field name used as ARTI coordinate input.")
    plan_build_parser.add_argument("--observer-coord-key", default=None, help="Batch field name used as ARTI observer coordinate input.")
    plan_build_parser.add_argument("--frame-operators-key", default=None, help="Batch field name used as ARTI inverse frame operator bank input.")
    plan_build_parser.add_argument("--no-freeze-base", action="store_true")

    artifact_build_parser = subparsers.add_parser("build", help="Build and export an ARTI adapter artifact from an importable model factory.")
    artifact_build_parser.add_argument("model", help="Import path for an nn.Module or factory, such as package.module:make_model.")
    artifact_build_parser.add_argument("artifact", type=Path, help="Destination adapter artifact .pt file.")
    artifact_build_parser.add_argument("--model-kwargs-json", type=Path, default=None, help="JSON keyword arguments passed to the model factory.")
    artifact_sample_group = artifact_build_parser.add_mutually_exclusive_group(required=True)
    artifact_sample_group.add_argument("--sample-shape", help="Comma-separated sample tensor shape, for example 2,4 or 2,16,768.")
    artifact_sample_group.add_argument("--sample-json", type=Path, help="JSON tensor-batch schema for dict samples.")
    artifact_build_parser.add_argument("--config", type=Path, default=None, help="Optional ARTI fit config JSON/TOML.")
    artifact_build_parser.add_argument("--target-modules", default=None, help="Comma-separated module names or glob patterns.")
    artifact_build_parser.add_argument("--profile", default="latent-adapt", help="Profile preset or alias.")
    artifact_build_parser.add_argument("--phases", type=int, default=None, help="Observer-phase coordinate dimension.")
    artifact_build_parser.add_argument("--scale", default="small", help="Scale preset.")
    artifact_build_parser.add_argument("--mechanism-coord-dim", type=int, default=None, help="Override resolved mechanism coord_dim.")
    artifact_build_parser.add_argument("--mechanism-coord-frame-mode", choices=("none", "paired_rotation", "operator_bank"), default=None, help="Override resolved mechanism coordinate frame mode.")
    artifact_build_parser.add_argument("--mechanism-observer-phase", action="store_true", help="Force observer phase on for this artifact.")
    artifact_build_parser.add_argument("--mechanism-virtual-recall", action="store_true", help="Force virtual recall on for this artifact.")
    artifact_build_parser.add_argument("--mechanism-operator-count", type=int, default=None, help="Override dynamic operator count.")
    artifact_build_parser.add_argument("--mechanism-interface-slots", type=int, default=None, help="Override virtual interface slot count.")
    artifact_build_parser.add_argument("--mechanism-recall-slots", type=int, default=None, help="Override private recall slot count.")
    artifact_build_parser.add_argument("--mechanism-recall-steps", type=int, default=None, help="Override private recall update steps.")
    artifact_build_parser.add_argument("--mechanism-recall-activation", choices=("half", "none"), default=None, help="Override Recall trace-survival activation.")
    artifact_build_parser.add_argument("--mechanism-hidden-multiplier", type=float, default=None, help="Override adapter hidden width multiplier.")
    artifact_build_parser.add_argument("--max-adapters", type=int, default=None)
    artifact_build_parser.add_argument("--max-extra-params", default=None)
    artifact_build_parser.add_argument("--causal", action="store_true")
    artifact_build_parser.add_argument("--mask-key", default=None, help="Batch field name used as ARTI mask/visibility input.")
    artifact_build_parser.add_argument("--coord-key", default=None, help="Batch field name used as ARTI coordinate input.")
    artifact_build_parser.add_argument("--observer-coord-key", default=None, help="Batch field name used as ARTI observer coordinate input.")
    artifact_build_parser.add_argument("--frame-operators-key", default=None, help="Batch field name used as ARTI inverse frame operator bank input.")
    artifact_build_parser.add_argument("--include-base", action="store_true", help="Include the patched full model state_dict in the artifact.")
    artifact_build_parser.add_argument("--report", type=Path, default=None, help="Optional JSON or Markdown fit report output path.")
    artifact_build_parser.add_argument("--lock-output", type=Path, default=None, help="Optional build lockfile written after the artifact passes build checks.")
    artifact_build_parser.add_argument("--task-graph-output", type=Path, default=None, help="Optional JSON task graph artifact for this build command.")
    artifact_build_parser.add_argument("--expect-plan", type=Path, default=None, help="Require the built artifact to match an approved dry-run fit plan.")
    artifact_build_parser.add_argument("--no-freeze-base", action="store_true")

    apply_parser = subparsers.add_parser("apply", help="Apply an ARTI adapter artifact to an importable model and write an application report.")
    apply_parser.add_argument("model", help="Import path for an nn.Module or factory, such as package.module:make_model.")
    apply_parser.add_argument("artifact", type=Path, help="Adapter artifact produced by ARTIFitResult.export().")
    apply_parser.add_argument("output", type=Path, help="Destination .json or .md application report.")
    apply_parser.add_argument("--model-kwargs-json", type=Path, default=None, help="JSON keyword arguments passed to the model factory.")
    apply_sample_group = apply_parser.add_mutually_exclusive_group()
    apply_sample_group.add_argument("--sample-shape", help="Comma-separated sample tensor shape, for example 2,4 or 2,16,768.")
    apply_sample_group.add_argument("--sample-json", type=Path, help="JSON tensor-batch schema for dict samples.")
    apply_parser.add_argument("--map-location", default=None, help="Optional torch.load map_location.")
    apply_parser.add_argument("--expect-config", type=Path, default=None, help="Require the applied artifact fingerprint to match this config.")
    apply_parser.add_argument("--expect-adapter-state-sha256", default=None, help="Require adapter tensor state SHA256 to match this value.")
    apply_parser.add_argument("--lock", type=Path, default=None, help="Require the artifact to match an ARTI build lockfile before applying.")
    apply_parser.add_argument("--max-adapters", type=int, default=None, help="Fail if applied adapter count exceeds this value.")
    apply_parser.add_argument("--save-state-dict", type=Path, default=None, help="Write the patched model state_dict after all apply gates pass.")
    apply_parser.add_argument("--deployment-output", type=Path, default=None, help="Optional deployment manifest written after apply and state_dict export pass.")
    apply_parser.add_argument("--task-graph-output", type=Path, default=None, help="Optional JSON task graph artifact for this apply command.")
    apply_parser.add_argument("--no-freeze-base", action="store_true")

    lock_parser = subparsers.add_parser("lock", help="Create an auditable ARTI build lockfile.")
    lock_parser.add_argument("output", type=Path, help="Destination build lock JSON file.")
    lock_parser.add_argument("--artifact", type=Path, required=True, help="Adapter artifact to lock.")
    lock_parser.add_argument("--plan", type=Path, default=None, help="Optional fit plan to lock with the artifact.")
    lock_parser.add_argument("--config", type=Path, default=None, help="Optional fit config to lock with the artifact.")
    lock_parser.add_argument("--map-location", default=None, help="Optional torch.load map_location for the artifact.")

    deployment_parser = subparsers.add_parser("deployment-manifest", help="Create an auditable ARTI deployment manifest.")
    deployment_parser.add_argument("output", type=Path, help="Destination deployment manifest JSON file.")
    deployment_parser.add_argument("--lock", type=Path, required=True, help="Build lock used for deployment.")
    deployment_parser.add_argument("--artifact", type=Path, required=True, help="Adapter artifact applied to the model.")
    deployment_parser.add_argument("--applied-report", type=Path, required=True, help="Application report produced by arti apply.")
    deployment_parser.add_argument("--state-dict", type=Path, required=True, help="Patched model state_dict produced by arti apply.")
    deployment_parser.add_argument("--map-location", default=None, help="Optional torch.load map_location.")

    init_parser = subparsers.add_parser("init-config", help="Write a starter ARTI fit config.")
    init_parser.add_argument("path", nargs="?", type=Path, default=Path("arti.json"))
    init_parser.add_argument("--profile", default="latent-adapt", help="Starter profile preset.")
    init_parser.add_argument("--scale", default="small", help="Starter scale preset.")
    init_parser.add_argument("--force", action="store_true", help="Overwrite an existing config file.")

    validate_parser = subparsers.add_parser("validate", help="Validate ARTI build artifacts.")
    validate_subparsers = validate_parser.add_subparsers(dest="kind", required=True)

    plan_parser = validate_subparsers.add_parser("plan", help="Validate an ARTI fit-plan JSON file.")
    plan_parser.add_argument("path", type=Path)
    plan_parser.add_argument("--expect-config", type=Path, default=None, help="Require the plan fingerprint to match this config.")
    plan_parser.add_argument("--expect-provenance-fingerprint", default=None, help="Require the plan provenance fingerprint to match this value.")
    plan_parser.add_argument("--expect-profile", default=None, help="Require the plan profile to match this value.")
    plan_parser.add_argument("--expect-scale", default=None, help="Require the plan scale to match this value.")
    plan_parser.add_argument("--expect-mechanism", action="append", default=[], metavar="KEY=VALUE", help="Require a resolved report.mechanism field to match; may be repeated.")
    plan_parser.add_argument("--expect-runtime-field", action="append", default=[], metavar="KEY=VALUE", help="Require a normalized fit_config.runtime field to match; may be repeated.")
    plan_parser.add_argument("--max-adapters", type=int, default=None, help="Fail if the plan selects more adapters.")
    plan_parser.add_argument("--max-extra-params", type=int, default=None, help="Fail if planned adapter parameters exceed this value.")

    artifact_parser = validate_subparsers.add_parser("artifact", help="Validate an ARTI adapter artifact file.")
    artifact_parser.add_argument("path", type=Path)
    artifact_parser.add_argument("--map-location", default=None, help="Optional torch.load map_location.")
    artifact_parser.add_argument("--expect-config", type=Path, default=None, help="Require the artifact fingerprint to match this config.")
    artifact_parser.add_argument("--expect-plan", type=Path, default=None, help="Require the artifact report and build metadata to match this dry-run fit plan.")
    artifact_parser.add_argument("--expect-adapter-state-sha256", default=None, help="Require adapter tensor state SHA256 to match this value.")
    artifact_parser.add_argument("--expect-profile", default=None, help="Require the artifact profile to match this value.")
    artifact_parser.add_argument("--expect-scale", default=None, help="Require the artifact scale to match this value.")
    artifact_parser.add_argument("--expect-mechanism", action="append", default=[], metavar="KEY=VALUE", help="Require a resolved report.mechanism field to match; may be repeated.")
    artifact_parser.add_argument("--expect-runtime-field", action="append", default=[], metavar="KEY=VALUE", help="Require a normalized artifact fit_config.runtime field to match; may be repeated.")
    artifact_parser.add_argument("--max-adapters", type=int, default=None, help="Fail if the artifact inserted more adapters.")
    artifact_parser.add_argument("--max-extra-params", type=int, default=None, help="Fail if artifact adapter parameters exceed this value.")

    config_parser = validate_subparsers.add_parser("config", help="Validate an ARTI fit project config file.")
    config_parser.add_argument("path", type=Path)
    config_parser.add_argument("--expect-profile", default=None, help="Require the config profile to match this value.")
    config_parser.add_argument("--expect-scale", default=None, help="Require the config scale to match this value.")
    config_parser.add_argument("--expect-mechanism", action="append", default=[], metavar="KEY=VALUE", help="Require a resolved config mechanism field to match; may be repeated.")
    config_parser.add_argument("--expect-runtime-field", action="append", default=[], metavar="KEY=VALUE", help="Require a normalized config runtime field to match; may be repeated.")

    lock_validate_parser = validate_subparsers.add_parser("lock", help="Validate an ARTI build lockfile.")
    lock_validate_parser.add_argument("path", type=Path)
    lock_validate_parser.add_argument("--map-location", default=None, help="Optional torch.load map_location for the artifact.")
    lock_validate_parser.add_argument("--expect-config", type=Path, default=None, help="Require the locked config fingerprint to match this config.")
    lock_validate_parser.add_argument("--expect-plan", type=Path, default=None, help="Require the locked artifact build metadata to match this dry-run fit plan.")
    lock_validate_parser.add_argument("--expect-provenance-fingerprint", default=None, help="Require the locked plan provenance fingerprint to match this value.")
    lock_validate_parser.add_argument("--expect-adapter-state-sha256", default=None, help="Require the locked adapter tensor SHA256 to match this value.")
    lock_validate_parser.add_argument("--expect-report-sha256", default=None, help="Require the locked artifact report SHA256 to match this value.")
    lock_validate_parser.add_argument("--expect-profile", default=None, help="Require the locked artifact profile to match this value.")
    lock_validate_parser.add_argument("--expect-scale", default=None, help="Require the locked artifact scale to match this value.")
    lock_validate_parser.add_argument("--expect-mechanism", action="append", default=[], metavar="KEY=VALUE", help="Require a locked artifact.mechanism field to match; may be repeated.")
    lock_validate_parser.add_argument("--expect-runtime-field", action="append", default=[], metavar="KEY=VALUE", help="Require a locked artifact.runtime field to match; may be repeated.")
    lock_validate_parser.add_argument("--max-adapters", type=int, default=None, help="Fail if the locked artifact inserted more adapters.")
    lock_validate_parser.add_argument("--max-extra-params", type=int, default=None, help="Fail if locked artifact adapter parameters exceed this value.")

    state_dict_parser = validate_subparsers.add_parser("state-dict", help="Validate a saved PyTorch state_dict fingerprint.")
    state_dict_parser.add_argument("path", type=Path)
    state_dict_parser.add_argument("--map-location", default=None, help="Optional torch.load map_location.")
    state_dict_parser.add_argument("--expect-state-dict-sha256", default=None, help="Require the saved state_dict SHA256 to match this value.")

    task_graph_parser = validate_subparsers.add_parser("task-graph", help="Validate an ARTI CLI task graph JSON file.")
    task_graph_parser.add_argument("path", type=Path)
    task_graph_parser.add_argument("--expect-kind", choices=("build", "apply"), default=None, help="Require a task graph command kind.")
    task_graph_parser.add_argument("--expect-artifact", action="append", default=[], metavar="KEY=VALUE", help="Require task_graph.artifacts[KEY] to match VALUE; may be repeated.")
    task_graph_parser.add_argument("--require-existing-artifacts", action="store_true", help="Require every non-null task graph artifact path to exist.")

    deployment_validate_parser = validate_subparsers.add_parser("deployment", help="Validate an ARTI deployment manifest.")
    deployment_validate_parser.add_argument("path", type=Path)
    deployment_validate_parser.add_argument("--map-location", default=None, help="Optional torch.load map_location.")
    deployment_validate_parser.add_argument("--expect-plan", type=Path, default=None, help="Require deployment artifact build metadata to match this dry-run fit plan.")
    deployment_validate_parser.add_argument("--expect-adapter-state-sha256", default=None, help="Require deployment adapter SHA256 to match this value.")
    deployment_validate_parser.add_argument("--expect-state-dict-sha256", default=None, help="Require deployment state_dict SHA256 to match this value.")
    deployment_validate_parser.add_argument("--expect-profile", default=None, help="Require deployment artifact profile to match this value.")
    deployment_validate_parser.add_argument("--expect-scale", default=None, help="Require deployment artifact scale to match this value.")
    deployment_validate_parser.add_argument("--expect-mechanism", action="append", default=[], metavar="KEY=VALUE", help="Require a deployment artifact.mechanism field to match; may be repeated.")
    deployment_validate_parser.add_argument("--expect-runtime-field", action="append", default=[], metavar="KEY=VALUE", help="Require a deployment artifact.runtime field to match; may be repeated.")
    deployment_validate_parser.add_argument("--max-adapters", type=int, default=None, help="Fail if deployment artifact inserted more adapters.")
    deployment_validate_parser.add_argument("--max-extra-params", type=int, default=None, help="Fail if deployment adapter parameters exceed this value.")

    return parser


def summarize_plan(
    payload: dict[str, Any],
    *,
    expect_config: Path | None = None,
    expect_provenance_fingerprint: str | None = None,
    expect_profile: str | None = None,
    expect_scale: str | None = None,
    expect_mechanism: dict[str, Any] | None = None,
    expect_runtime_fields: dict[str, Any] | None = None,
    max_adapters: int | None = None,
    max_extra_params: int | None = None,
) -> dict[str, Any]:
    report = payload["report"]
    insertion_plan = report["insertion_plan"]
    spec = insertion_plan.get("spec", {})
    budget_limit = spec.get("max_extra_params")
    budget_used = insertion_plan.get("adapter_parameters", 0)
    expected_fingerprint = None if expect_config is None else load_fit_config(expect_config).fingerprint
    if expected_fingerprint is not None and report.get("config_fingerprint") != expected_fingerprint:
        raise ValueError("ARTI fit plan config_fingerprint does not match expected config")
    provenance_fingerprint = payload.get("provenance_fingerprint")
    if expect_provenance_fingerprint is not None and provenance_fingerprint != expect_provenance_fingerprint:
        raise ValueError("ARTI fit plan provenance_fingerprint does not match expected value")
    profile = report.get("profile")
    scale = report.get("scale")
    if expect_profile is not None and profile != expect_profile:
        raise ValueError("ARTI fit plan profile does not match expected value")
    if expect_scale is not None and scale != expect_scale:
        raise ValueError("ARTI fit plan scale does not match expected value")
    mechanism = report.get("mechanism") or {}
    check_expected_mechanism(mechanism, expect_mechanism, label="ARTI fit plan")
    runtime = (report.get("fit_config") or {}).get("runtime") or {}
    check_expected_runtime_fields(runtime, expect_runtime_fields, label="ARTI fit plan")
    planned_count = len(insertion_plan.get("selected", []))
    if max_adapters is not None and planned_count > max_adapters:
        raise ValueError("ARTI fit plan exceeds CLI max_adapters")
    if max_extra_params is not None and int(budget_used) > max_extra_params:
        raise ValueError("ARTI fit plan exceeds CLI max_extra_params")
    return {
        "ok": True,
        "kind": "fit-plan",
        "path_kind": payload["kind"],
        "profile": profile,
        "scale": scale,
        "expected_profile": expect_profile,
        "expected_scale": expect_scale,
        "mechanism": mechanism,
        "expected_mechanism": expect_mechanism or {},
        "runtime": runtime,
        "expected_runtime_fields": expect_runtime_fields or {},
        "candidate_count": len(report.get("scanned", {}).get("candidates", [])),
        "planned_count": planned_count,
        "skipped_budget_count": len(insertion_plan.get("skipped_budget", [])),
        "adapter_parameters": budget_used,
        "budget_limit": budget_limit,
        "budget_exhausted": budget_limit is not None and int(budget_used) >= int(budget_limit),
        "config_fingerprint": report.get("config_fingerprint"),
        "expected_config_fingerprint": expected_fingerprint,
        "cli_max_adapters": max_adapters,
        "cli_max_extra_params": max_extra_params,
        "objective_plan": report.get("objective_plan", []),
        "provenance": payload.get("provenance"),
        "provenance_fingerprint": provenance_fingerprint,
        "expected_provenance_fingerprint": expect_provenance_fingerprint,
    }


def summarize_artifact(
    payload: dict[str, Any],
    *,
    expect_config: Path | None = None,
    expect_plan: Path | None = None,
    expect_adapter_state_sha256: str | None = None,
    expect_profile: str | None = None,
    expect_scale: str | None = None,
    expect_mechanism: dict[str, Any] | None = None,
    expect_runtime_fields: dict[str, Any] | None = None,
    max_adapters: int | None = None,
    max_extra_params: int | None = None,
) -> dict[str, Any]:
    manifest = payload["manifest"]
    report = payload["report"]
    expected_fingerprint = None if expect_config is None else load_fit_config(expect_config).fingerprint
    if expected_fingerprint is not None and manifest.get("config_fingerprint") != expected_fingerprint:
        raise ValueError("ARTI adapter artifact config_fingerprint does not match expected config")
    plan_payload = None
    expected_build_metadata = None
    if expect_plan is not None:
        plan_payload = validate_plan(expect_plan)
        check_artifact_matches_plan(payload, plan_payload)
        expected_build_metadata = build_metadata_from_plan(expect_plan, plan_payload)
        if payload.get("build") != expected_build_metadata:
            raise ValueError("ARTI adapter artifact build metadata does not match expected plan")
    adapter_state_sha256 = manifest.get("adapter_state_sha256")
    if expect_adapter_state_sha256 is not None and adapter_state_sha256 != expect_adapter_state_sha256:
        raise ValueError("ARTI adapter artifact adapter_state_sha256 does not match expected value")
    profile = manifest.get("profile")
    scale = manifest.get("scale")
    if expect_profile is not None and profile != expect_profile:
        raise ValueError("ARTI adapter artifact profile does not match expected value")
    if expect_scale is not None and scale != expect_scale:
        raise ValueError("ARTI adapter artifact scale does not match expected value")
    mechanism = report.get("mechanism") or {}
    check_expected_mechanism(mechanism, expect_mechanism, label="ARTI adapter artifact")
    runtime = (report.get("fit_config") or {}).get("runtime") or {}
    check_expected_runtime_fields(runtime, expect_runtime_fields, label="ARTI adapter artifact")
    inserted_count = report.get("summary", {}).get("inserted_count")
    adapter_parameters = manifest.get("adapter_parameters")
    if max_adapters is not None and int(inserted_count or 0) > max_adapters:
        raise ValueError("ARTI adapter artifact exceeds CLI max_adapters")
    if max_extra_params is not None and int(adapter_parameters or 0) > max_extra_params:
        raise ValueError("ARTI adapter artifact exceeds CLI max_extra_params")
    return {
        "ok": True,
        "kind": "adapter-artifact",
        "backend": manifest.get("backend"),
        "profile": profile,
        "scale": scale,
        "expected_profile": expect_profile,
        "expected_scale": expect_scale,
        "mechanism": mechanism,
        "expected_mechanism": expect_mechanism or {},
        "runtime": runtime,
        "expected_runtime_fields": expect_runtime_fields or {},
        "adapter_key_count": manifest.get("adapter_key_count"),
        "adapter_parameters": adapter_parameters,
        "inserted_count": inserted_count,
        "include_base": manifest.get("include_base"),
        "config_fingerprint": manifest.get("config_fingerprint"),
        "expected_config_fingerprint": expected_fingerprint,
        "expected_plan": None if expect_plan is None else str(expect_plan),
        "expected_plan_provenance_fingerprint": None if plan_payload is None else plan_payload.get("provenance_fingerprint"),
        "build": payload.get("build"),
        "expected_build": expected_build_metadata,
        "adapter_state_sha256": adapter_state_sha256,
        "report_sha256": manifest.get("report_sha256"),
        "expected_adapter_state_sha256": expect_adapter_state_sha256,
        "cli_max_adapters": max_adapters,
        "cli_max_extra_params": max_extra_params,
    }


def summarize_config(
    path: Path,
    *,
    expect_profile: str | None = None,
    expect_scale: str | None = None,
    expect_mechanism: dict[str, Any] | None = None,
    expect_runtime_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    config = load_fit_config(path)
    profile, scale = resolve_fit_config_mechanism(config)
    mechanism = MechanismSummary.from_config(profile, scale, scale_name=config.scale).to_dict()
    if expect_profile is not None and config.profile != expect_profile:
        raise ValueError("ARTI fit config profile does not match expected value")
    if expect_scale is not None and config.scale != expect_scale:
        raise ValueError("ARTI fit config scale does not match expected value")
    check_expected_mechanism(mechanism, expect_mechanism, label="ARTI fit config")
    runtime = config.to_dict()["runtime"]
    check_expected_runtime_fields(runtime, expect_runtime_fields, label="ARTI fit config")
    return {
        "ok": True,
        "kind": "fit-config",
        "config_fingerprint": config.fingerprint,
        "config": config.to_dict(),
        "profile": config.profile,
        "scale": config.scale,
        "expected_profile": expect_profile,
        "expected_scale": expect_scale,
        "mechanism": mechanism,
        "expected_mechanism": expect_mechanism or {},
        "runtime": runtime,
        "expected_runtime_fields": expect_runtime_fields or {},
    }


def initialize_config(path: Path, *, profile: str, scale: str, force: bool) -> dict[str, Any]:
    target = write_fit_config_template(path, profile=profile, scale=scale, overwrite=force)
    config = load_fit_config(target)
    return {
        "ok": True,
        "kind": "fit-config-template",
        "path": str(target),
        "config_fingerprint": config.fingerprint,
        "config": config.to_dict(),
    }


def generate_docs_report(args: argparse.Namespace) -> dict[str, Any]:
    target = write_generated_docs(args.output, phases=args.phases)
    content = generate_capabilities_markdown(phases=args.phases)
    return {
        "ok": True,
        "kind": "generated-docs",
        "action": "generate",
        "path": str(target),
        "phases": args.phases,
        "bytes": len(content.encode("utf-8")),
    }


def check_docs_report(args: argparse.Namespace) -> dict[str, Any]:
    check_generated_docs(args.output, phases=args.phases)
    return {
        "ok": True,
        "kind": "generated-docs",
        "action": "check",
        "path": str(args.output),
        "phases": args.phases,
    }


def schema_report(args: argparse.Namespace) -> dict[str, Any]:
    if args.kind == "fit-config":
        generator = generate_fit_config_schema_json
        writer = write_fit_config_schema
        checker = check_fit_config_schema
        kind = "fit-config-schema"
        stale_name = "fit config"
    else:
        generator = generate_task_graph_schema_json
        writer = write_task_graph_schema
        checker = check_task_graph_schema
        kind = "task-graph-schema"
        stale_name = "task graph"
    if args.action == "generate":
        target = writer(args.output)
    else:
        checker(args.output)
        target = args.output
    content = generator()
    return {
        "ok": True,
        "kind": kind,
        "schema": stale_name,
        "action": args.action,
        "path": str(target),
        "bytes": len(content.encode("utf-8")),
    }


def maybe_write_task_graph_artifact(path: Path | None, *, command_kind: str, task_graph: dict[str, Any]) -> str | None:
    if path is None:
        return None
    return str(write_task_graph_artifact(path, command_kind=command_kind, task_graph=task_graph))


def parse_expected_mechanism(values: tuple[str, ...] | list[str]) -> dict[str, Any]:
    expected: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise ValueError("ARTI mechanism expectation must use KEY=VALUE")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("ARTI mechanism expectation key must be non-empty")
        if key not in MECHANISM_FIELDS:
            raise ValueError(f"unknown ARTI mechanism field {key!r}; expected one of {sorted(MECHANISM_FIELDS)}")
        expected[key] = parse_scalar_value(raw_value.strip())
    return expected


def parse_expected_runtime_fields(values: tuple[str, ...] | list[str]) -> dict[str, Any]:
    expected: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise ValueError("ARTI runtime field expectation must use KEY=VALUE")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("ARTI runtime field expectation key must be non-empty")
        if key not in RUNTIME_FIELD_NAMES:
            raise ValueError(f"unknown ARTI runtime field {key!r}; expected one of {sorted(RUNTIME_FIELD_NAMES)}")
        expected[key] = parse_scalar_value(raw_value.strip())
    return expected


def parse_expected_artifacts(values: tuple[str, ...] | list[str]) -> dict[str, Any]:
    expected: dict[str, Any] = {}
    for item in values:
        if "=" not in item:
            raise ValueError("ARTI task graph artifact expectation must use KEY=VALUE")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("ARTI task graph artifact expectation key must be non-empty")
        expected[key] = parse_scalar_value(raw_value.strip())
    return expected


def parse_scalar_value(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(value.replace("_", ""))
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def check_expected_mechanism(mechanism: dict[str, Any], expected: dict[str, Any] | None, *, label: str) -> None:
    for key, value in (expected or {}).items():
        if key not in mechanism:
            raise ValueError(f"{label} mechanism.{key} is missing")
        actual = mechanism[key]
        if isinstance(value, float) and isinstance(actual, (float, int)):
            if abs(float(actual) - value) > 1e-9:
                raise ValueError(f"{label} mechanism.{key} does not match expected value")
        elif actual != value:
            raise ValueError(f"{label} mechanism.{key} does not match expected value")


def check_expected_runtime_fields(runtime: dict[str, Any], expected: dict[str, Any] | None, *, label: str) -> None:
    for key, value in (expected or {}).items():
        if key not in runtime:
            raise ValueError(f"{label} runtime.{key} is missing")
        if runtime[key] != value:
            raise ValueError(f"{label} runtime.{key} does not match expected value")


def summarize_lock(
    payload: dict[str, Any],
    *,
    path: Path | None = None,
    expect_config: Path | None = None,
    expect_plan: Path | None = None,
    expect_provenance_fingerprint: str | None = None,
    expect_adapter_state_sha256: str | None = None,
    expect_report_sha256: str | None = None,
    expect_profile: str | None = None,
    expect_scale: str | None = None,
    expect_mechanism: dict[str, Any] | None = None,
    expect_runtime_fields: dict[str, Any] | None = None,
    max_adapters: int | None = None,
    max_extra_params: int | None = None,
) -> dict[str, Any]:
    artifact = payload["artifact"]
    plan = payload.get("plan")
    config = payload.get("config")
    profile = artifact.get("profile")
    scale = artifact.get("scale")
    adapter_state_sha256 = artifact.get("adapter_state_sha256")
    report_sha256 = artifact.get("report_sha256")
    inserted_count = artifact.get("inserted_count")
    adapter_parameters = artifact.get("adapter_parameters")
    config_fingerprint = artifact.get("config_fingerprint")
    provenance_fingerprint = None if plan is None else plan.get("provenance_fingerprint")
    expected_config_fingerprint = None if expect_config is None else load_fit_config(expect_config).fingerprint
    if expected_config_fingerprint is not None and config_fingerprint != expected_config_fingerprint:
        raise ValueError("ARTI build lock config_fingerprint does not match expected config")
    expected_build_metadata = expected_plan_build_metadata(expect_plan)
    if expected_build_metadata is not None and artifact.get("build") != expected_build_metadata:
        raise ValueError("ARTI build lock artifact.build does not match expected plan")
    if expect_provenance_fingerprint is not None and provenance_fingerprint != expect_provenance_fingerprint:
        raise ValueError("ARTI build lock provenance_fingerprint does not match expected value")
    if expect_adapter_state_sha256 is not None and adapter_state_sha256 != expect_adapter_state_sha256:
        raise ValueError("ARTI build lock adapter_state_sha256 does not match expected value")
    if expect_report_sha256 is not None and report_sha256 != expect_report_sha256:
        raise ValueError("ARTI build lock report_sha256 does not match expected value")
    if expect_profile is not None and profile != expect_profile:
        raise ValueError("ARTI build lock profile does not match expected value")
    if expect_scale is not None and scale != expect_scale:
        raise ValueError("ARTI build lock scale does not match expected value")
    mechanism = artifact.get("mechanism") or {}
    check_expected_mechanism(mechanism, expect_mechanism, label="ARTI build lock")
    runtime = artifact.get("runtime") or {}
    check_expected_runtime_fields(runtime, expect_runtime_fields, label="ARTI build lock")
    if max_adapters is not None and int(inserted_count or 0) > max_adapters:
        raise ValueError("ARTI build lock exceeds CLI max_adapters")
    if max_extra_params is not None and int(adapter_parameters or 0) > max_extra_params:
        raise ValueError("ARTI build lock exceeds CLI max_extra_params")
    return {
        "ok": True,
        "kind": "build-lock",
        "path": None if path is None else str(path),
        "artifact": artifact.get("path"),
        "plan": None if plan is None else plan.get("path"),
        "config": None if config is None else config.get("path"),
        "profile": profile,
        "scale": scale,
        "adapter_state_sha256": adapter_state_sha256,
        "expected_adapter_state_sha256": expect_adapter_state_sha256,
        "report_sha256": report_sha256,
        "expected_report_sha256": expect_report_sha256,
        "expected_profile": expect_profile,
        "expected_scale": expect_scale,
        "mechanism": mechanism,
        "expected_mechanism": expect_mechanism or {},
        "runtime": runtime,
        "expected_runtime_fields": expect_runtime_fields or {},
        "config_fingerprint": config_fingerprint,
        "expected_config_fingerprint": expected_config_fingerprint,
        "build": artifact.get("build"),
        "expected_plan": None if expect_plan is None else str(expect_plan),
        "expected_build": expected_build_metadata,
        "provenance_fingerprint": provenance_fingerprint,
        "expected_provenance_fingerprint": expect_provenance_fingerprint,
        "adapter_parameters": adapter_parameters,
        "inserted_count": inserted_count,
        "cli_max_adapters": max_adapters,
        "cli_max_extra_params": max_extra_params,
        "adapter_key_count": artifact.get("adapter_key_count"),
    }


def summarize_state_dict(
    path: Path,
    *,
    map_location: str | torch.device | None = None,
    expect_state_dict_sha256: str | None = None,
) -> dict[str, Any]:
    state_dict = torch.load(path, map_location=map_location, weights_only=True)
    if not isinstance(state_dict, dict):
        raise ValueError("ARTI state-dict validation expects a dictionary payload")
    state_dict_sha256 = hash_tensor_state_dict(state_dict)
    if expect_state_dict_sha256 is not None and state_dict_sha256 != expect_state_dict_sha256:
        raise ValueError("ARTI saved state_dict_sha256 does not match expected value")
    return {
        "ok": True,
        "kind": "state-dict",
        "path": str(path),
        "state_key_count": len(state_dict),
        "state_dict_sha256": state_dict_sha256,
        "expected_state_dict_sha256": expect_state_dict_sha256,
    }


def summarize_task_graph(
    payload: dict[str, Any],
    *,
    path: Path | None = None,
    expect_kind: str | None = None,
    expect_artifacts: dict[str, Any] | None = None,
    require_existing_artifacts: bool = False,
) -> dict[str, Any]:
    command_kind = payload.get("command_kind")
    if expect_kind is not None and command_kind != expect_kind:
        raise ValueError("ARTI task graph command_kind does not match expected value")
    graph = payload["task_graph"]
    artifacts = graph["artifacts"]
    for key, value in (expect_artifacts or {}).items():
        if artifacts.get(key) != value:
            raise ValueError(f"ARTI task graph artifacts.{key} does not match expected value")
    missing_artifacts = missing_task_graph_artifacts(artifacts, base_path=path)
    if require_existing_artifacts and missing_artifacts:
        raise ValueError(f"ARTI task graph artifacts are missing: {', '.join(missing_artifacts)}")
    return {
        "ok": True,
        "kind": "task-graph",
        "path": None if path is None else str(path),
        "command_kind": command_kind,
        "task_count": len(graph["tasks"]),
        "tasks": [task["name"] for task in graph["tasks"]],
        "artifacts": artifacts,
        "missing_artifacts": missing_artifacts,
        "require_existing_artifacts": require_existing_artifacts,
        "expected_kind": expect_kind,
        "expected_artifacts": expect_artifacts or {},
    }


def missing_task_graph_artifacts(artifacts: dict[str, Any], *, base_path: Path | None = None) -> list[str]:
    base_dir = None if base_path is None else base_path.parent
    missing = []
    for key, value in artifacts.items():
        if value is None:
            continue
        target = Path(str(value))
        if not target.is_absolute() and base_dir is not None:
            target = base_dir / target
        if not target.exists():
            missing.append(key)
    return missing


def summarize_deployment(
    payload: dict[str, Any],
    *,
    path: Path | None = None,
    expect_plan: Path | None = None,
    expect_adapter_state_sha256: str | None = None,
    expect_state_dict_sha256: str | None = None,
    expect_profile: str | None = None,
    expect_scale: str | None = None,
    expect_mechanism: dict[str, Any] | None = None,
    expect_runtime_fields: dict[str, Any] | None = None,
    max_adapters: int | None = None,
    max_extra_params: int | None = None,
) -> dict[str, Any]:
    artifact = payload["artifact"]
    state_dict = payload["state_dict"]
    applied_report = payload["applied_report"]
    adapter_state_sha256 = artifact.get("adapter_state_sha256")
    state_dict_sha256 = state_dict.get("state_dict_sha256")
    profile = artifact.get("profile")
    scale = artifact.get("scale")
    inserted_count = artifact.get("inserted_count")
    adapter_parameters = artifact.get("adapter_parameters")
    expected_build_metadata = expected_plan_build_metadata(expect_plan)
    if expected_build_metadata is not None and artifact.get("build") != expected_build_metadata:
        raise ValueError("ARTI deployment manifest artifact.build does not match expected plan")
    if expect_adapter_state_sha256 is not None and adapter_state_sha256 != expect_adapter_state_sha256:
        raise ValueError("ARTI deployment manifest adapter_state_sha256 does not match expected value")
    if expect_state_dict_sha256 is not None and state_dict_sha256 != expect_state_dict_sha256:
        raise ValueError("ARTI deployment manifest state_dict_sha256 does not match expected value")
    if expect_profile is not None and profile != expect_profile:
        raise ValueError("ARTI deployment manifest profile does not match expected value")
    if expect_scale is not None and scale != expect_scale:
        raise ValueError("ARTI deployment manifest scale does not match expected value")
    mechanism = artifact.get("mechanism") or {}
    check_expected_mechanism(mechanism, expect_mechanism, label="ARTI deployment manifest")
    runtime = artifact.get("runtime") or {}
    check_expected_runtime_fields(runtime, expect_runtime_fields, label="ARTI deployment manifest")
    if max_adapters is not None and int(inserted_count or 0) > max_adapters:
        raise ValueError("ARTI deployment manifest exceeds CLI max_adapters")
    if max_extra_params is not None and int(adapter_parameters or 0) > max_extra_params:
        raise ValueError("ARTI deployment manifest exceeds CLI max_extra_params")
    return {
        "ok": True,
        "kind": "deployment-manifest",
        "path": None if path is None else str(path),
        "artifact": artifact.get("path"),
        "lock": payload["lock"].get("path"),
        "applied_report": applied_report.get("path"),
        "state_dict": state_dict.get("path"),
        "profile": profile,
        "scale": scale,
        "expected_profile": expect_profile,
        "expected_scale": expect_scale,
        "mechanism": mechanism,
        "expected_mechanism": expect_mechanism or {},
        "runtime": runtime,
        "expected_runtime_fields": expect_runtime_fields or {},
        "build": artifact.get("build"),
        "expected_plan": None if expect_plan is None else str(expect_plan),
        "expected_build": expected_build_metadata,
        "adapter_parameters": adapter_parameters,
        "inserted_count": inserted_count,
        "cli_max_adapters": max_adapters,
        "cli_max_extra_params": max_extra_params,
        "adapter_state_sha256": adapter_state_sha256,
        "expected_adapter_state_sha256": expect_adapter_state_sha256,
        "artifact_report_sha256": artifact.get("report_sha256"),
        "applied_report_sha256": applied_report.get("report_sha256"),
        "state_dict_sha256": state_dict_sha256,
        "expected_state_dict_sha256": expect_state_dict_sha256,
        "state_key_count": state_dict.get("state_key_count"),
    }


def expected_plan_build_metadata(expect_plan: Path | None) -> dict[str, Any] | None:
    if expect_plan is None:
        return None
    return build_metadata_from_plan(expect_plan, validate_plan(expect_plan))


def create_deployment_report(args: argparse.Namespace) -> dict[str, Any]:
    manifest_path = create_deployment_manifest(
        args.output,
        lock=args.lock,
        artifact=args.artifact,
        applied_report=args.applied_report,
        state_dict=args.state_dict,
        map_location=args.map_location,
    )
    return summarize_deployment(validate_deployment_manifest(manifest_path, map_location=args.map_location), path=manifest_path)


def create_lock_report(args: argparse.Namespace) -> dict[str, Any]:
    lock_path = create_build_lock(
        args.output,
        artifact=args.artifact,
        plan=args.plan,
        config=args.config,
        map_location=args.map_location,
    )
    return summarize_lock(validate_build_lock(lock_path, map_location=args.map_location), path=lock_path)


def load_model_kwargs(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("model kwargs JSON must be an object")
    return payload


def load_model_reference(reference: str, *, kwargs: dict[str, Any] | None = None) -> nn.Module:
    if ":" not in reference:
        raise ValueError("model reference must use module:attribute syntax")
    kwargs = {} if kwargs is None else kwargs
    module_name, attribute = reference.split(":", 1)
    module = importlib.import_module(module_name)
    obj = module
    for part in attribute.split("."):
        obj = getattr(obj, part)
    model = obj(**kwargs) if callable(obj) and not isinstance(obj, nn.Module) else obj
    if isinstance(model, type) and issubclass(model, nn.Module):
        model = model(**kwargs)
    if not isinstance(model, nn.Module):
        raise ValueError("model reference must resolve to an nn.Module or a factory returning nn.Module")
    return model


def parse_sample_shape(value: str) -> tuple[int, ...]:
    try:
        shape = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError("sample shape must be comma-separated integers") from exc
    if not shape or any(dim <= 0 for dim in shape):
        raise ValueError("sample shape dimensions must be positive")
    return validate_sample_shape(shape)


def validate_sample_shape(shape: tuple[int, ...]) -> tuple[int, ...]:
    elements = 1
    for dim in shape:
        elements *= dim
        if elements > MAX_SAMPLE_ELEMENTS:
            raise ValueError(f"sample tensor cannot exceed {MAX_SAMPLE_ELEMENTS:,} elements")
    return shape


def torch_dtype(name: str) -> torch.dtype:
    aliases = {
        "bool": torch.bool,
        "float": torch.float32,
        "float32": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
        "int": torch.int64,
        "int64": torch.int64,
        "long": torch.long,
        "int32": torch.int32,
    }
    try:
        return aliases[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported sample dtype {name!r}") from exc


def sample_tensor_from_spec(spec: dict[str, Any]) -> torch.Tensor:
    if "shape" not in spec:
        raise ValueError("sample tensor spec must include shape")
    shape = tuple(int(dim) for dim in spec["shape"])
    if not shape or any(dim <= 0 for dim in shape):
        raise ValueError("sample tensor shape dimensions must be positive")
    validate_sample_shape(shape)
    dtype = torch_dtype(str(spec.get("dtype", "float32")))
    kind = str(spec.get("kind", "randn")).lower()
    if kind == "zeros":
        return torch.zeros(*shape, dtype=dtype)
    if kind == "ones":
        return torch.ones(*shape, dtype=dtype)
    if kind == "randint":
        low = int(spec.get("low", 0))
        high = int(spec.get("high", 2))
        if high <= low:
            raise ValueError("randint sample spec requires high > low")
        return torch.randint(low, high, shape, dtype=dtype)
    if kind == "randn":
        if not dtype.is_floating_point:
            raise ValueError("randn sample spec requires a floating dtype")
        return torch.randn(*shape, dtype=dtype)
    raise ValueError(f"unsupported sample tensor kind {kind!r}")


def sample_from_json(path: Path) -> Any:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "fields" in payload:
        fields = payload["fields"]
        if not isinstance(fields, dict):
            raise ValueError("sample JSON fields must be a dictionary")
        return {key: sample_tensor_from_spec(value) for key, value in fields.items()}
    if isinstance(payload, dict) and "shape" in payload:
        return sample_tensor_from_spec(payload)
    if isinstance(payload, dict):
        return {key: sample_tensor_from_spec(value) for key, value in payload.items()}
    raise ValueError("sample JSON must be a tensor spec or a dictionary of tensor specs")


def split_patterns(value: str | None) -> list[str] | None:
    if value is None:
        return None
    patterns = [part.strip() for part in value.split(",") if part.strip()]
    return patterns or None


def cli_mechanism_overrides(args: argparse.Namespace) -> MechanismOverrides | None:
    payload = {
        "coord_dim": args.mechanism_coord_dim,
        "coord_frame_mode": args.mechanism_coord_frame_mode,
        "observer_phase": True if args.mechanism_observer_phase else None,
        "virtual_recall": True if args.mechanism_virtual_recall else None,
        "operator_count": args.mechanism_operator_count,
        "interface_slots": args.mechanism_interface_slots,
        "recall_slots": args.mechanism_recall_slots,
        "recall_steps": args.mechanism_recall_steps,
        "recall_activation": args.mechanism_recall_activation,
        "hidden_multiplier": args.mechanism_hidden_multiplier,
    }
    overrides = MechanismOverrides.from_mapping({key: value for key, value in payload.items() if value is not None}).validate()
    return overrides if overrides.has_values() else None


def plan_provenance(args: argparse.Namespace) -> dict[str, Any]:
    mechanism = cli_mechanism_overrides(args)
    return {
        "model": args.model,
        "model_kwargs_json": None if args.model_kwargs_json is None else str(args.model_kwargs_json),
        "sample_shape": None if args.sample_shape is None else list(parse_sample_shape(args.sample_shape)),
        "sample_json": None if args.sample_json is None else str(args.sample_json),
        "config": None if args.config is None else str(args.config),
        "target_modules": split_patterns(args.target_modules),
        "profile": args.profile,
        "phases": args.phases,
        "scale": args.scale,
        "mechanism": None if mechanism is None else mechanism.to_dict(),
        "max_adapters": args.max_adapters,
        "max_extra_params": args.max_extra_params,
        "causal": args.causal,
        "runtime_fields": {
            "mask_key": args.mask_key,
            "coord_key": args.coord_key,
            "observer_coord_key": args.observer_coord_key,
            "frame_operators_key": args.frame_operators_key,
        },
        "freeze_base": not args.no_freeze_base,
    }


def append_plan_provenance_markdown(markdown: str, provenance: dict[str, Any], provenance_fingerprint: str) -> str:
    lines = [
        markdown.rstrip(),
        "",
        "## Plan Provenance",
        "",
        f"Provenance fingerprint: `{provenance_fingerprint}`",
        "",
        "| Field | Value |",
        "| --- | --- |",
    ]
    for key, value in provenance.items():
        lines.append(f"| `{key}` | `{value}` |")
    return "\n".join(lines) + "\n"


def create_fit_plan(args: argparse.Namespace) -> dict[str, Any]:
    model = load_model_reference(args.model, kwargs=load_model_kwargs(args.model_kwargs_json))
    sample = sample_from_json(args.sample_json) if args.sample_json is not None else torch.randn(*parse_sample_shape(args.sample_shape))
    mechanism = cli_mechanism_overrides(args)
    provenance = plan_provenance(args)
    provenance_fingerprint = plan_provenance_fingerprint(provenance)
    result = fit(
        model,
        config=args.config,
        sample_batch=sample,
        target_modules=split_patterns(args.target_modules),
        profile=args.profile,
        phases=args.phases,
        scale=args.scale,
        mechanism=mechanism,
        freeze_base=not args.no_freeze_base,
        max_adapters=args.max_adapters,
        max_extra_params=args.max_extra_params,
        causal=args.causal,
        mask_key=args.mask_key,
        coord_key=args.coord_key,
        observer_coord_key=args.observer_coord_key,
        frame_operators_key=args.frame_operators_key,
        dry_run=True,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() in {".md", ".markdown"}:
        args.output.write_text(append_plan_provenance_markdown(result.report.to_markdown(), provenance, provenance_fingerprint), encoding="utf-8")
        payload = {
            "ok": True,
            "kind": "fit-plan",
            "output": str(args.output),
            "profile": result.report.profile,
            "scale": result.report.scale,
            "candidate_count": len(result.report.scanned.candidates),
            "planned_count": 0 if result.report.insertion_plan is None else len(result.report.insertion_plan.selected),
            "adapter_parameters": result.report.adapter_parameters,
            "config_fingerprint": result.report.config_fingerprint,
            "provenance": provenance,
            "provenance_fingerprint": provenance_fingerprint,
        }
    else:
        payload = {
            "format_version": 1,
            "package_name": "arti",
            "kind": "fit-plan",
            "provenance": provenance,
            "provenance_fingerprint": provenance_fingerprint,
            "report": result.report.to_dict(),
        }
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        payload = summarize_plan(payload)
        payload["output"] = str(args.output)
    return payload


def create_artifact_report(args: argparse.Namespace) -> dict[str, Any]:
    plan_payload = None if args.expect_plan is None else validate_plan(args.expect_plan)
    build_metadata = None if plan_payload is None else build_metadata_from_plan(args.expect_plan, plan_payload)
    model = load_model_reference(args.model, kwargs=load_model_kwargs(args.model_kwargs_json))
    sample = sample_from_json(args.sample_json) if args.sample_json is not None else torch.randn(*parse_sample_shape(args.sample_shape))
    mechanism = cli_mechanism_overrides(args)
    result = fit(
        model,
        config=args.config,
        sample_batch=sample,
        target_modules=split_patterns(args.target_modules),
        profile=args.profile,
        phases=args.phases,
        scale=args.scale,
        mechanism=mechanism,
        freeze_base=not args.no_freeze_base,
        max_adapters=args.max_adapters,
        max_extra_params=args.max_extra_params,
        causal=args.causal,
        mask_key=args.mask_key,
        coord_key=args.coord_key,
        observer_coord_key=args.observer_coord_key,
        frame_operators_key=args.frame_operators_key,
    )
    artifact_path = result.export(args.artifact, include_base=args.include_base, build_metadata=build_metadata)
    payload = validate_artifact(artifact_path)
    if plan_payload is not None:
        check_artifact_matches_plan(payload, plan_payload)
    if args.report is not None:
        result.write_report(args.report)
    lock_summary = None
    if args.lock_output is not None:
        lock_path = create_build_lock(
            args.lock_output,
            artifact=artifact_path,
            plan=args.expect_plan,
            config=args.config,
        )
        lock_summary = summarize_lock(validate_build_lock(lock_path), path=lock_path, expect_plan=args.expect_plan)
    task_graph = cli_build_task_graph(result.report.to_dict().get("build_plan", []), artifact_path=artifact_path, report_path=args.report, lock_path=args.lock_output)
    task_graph_output = maybe_write_task_graph_artifact(args.task_graph_output, command_kind="build", task_graph=task_graph)
    manifest = payload["manifest"]
    return {
        "ok": True,
        "kind": "adapter-artifact",
        "task_graph": task_graph,
        "task_graph_output": task_graph_output,
        "artifact": str(artifact_path),
        "report": None if args.report is None else str(args.report),
        "lock": None if lock_summary is None else lock_summary["path"],
        "lock_summary": lock_summary,
        "model": args.model,
        "adapter_count": result.adapter_count,
        "adapter_parameters": manifest.get("adapter_parameters"),
        "adapter_state_sha256": manifest.get("adapter_state_sha256"),
        "report_sha256": manifest.get("report_sha256"),
        "config_fingerprint": manifest.get("config_fingerprint"),
        "profile": manifest.get("profile"),
        "scale": manifest.get("scale"),
        "include_base": manifest.get("include_base"),
        "build": payload.get("build"),
        "expected_plan": None if args.expect_plan is None else str(args.expect_plan),
        "expected_plan_provenance_fingerprint": None if plan_payload is None else plan_payload.get("provenance_fingerprint"),
    }


def cli_build_task_graph(
    report_tasks: list[dict[str, Any]],
    *,
    artifact_path: Path,
    report_path: Path | None = None,
    lock_path: Path | None = None,
) -> dict[str, Any]:
    tasks = [dict(task) for task in report_tasks]
    last_name = tasks[-1]["name"] if tasks else "insert"
    tasks.append({"name": "export-artifact", "kind": "artifact", "depends_on": [last_name], "enabled": True, "status": "success"})
    if report_path is not None:
        tasks.append({"name": "write-report", "kind": "report", "depends_on": ["export-artifact"], "enabled": True, "status": "success"})
    if lock_path is not None:
        tasks.append({"name": "write-lock", "kind": "lock", "depends_on": ["export-artifact"], "enabled": True, "status": "success"})
    return {
        "tasks": tasks,
        "artifacts": {
            "adapter": str(artifact_path),
            "report": None if report_path is None else str(report_path),
            "lock": None if lock_path is None else str(lock_path),
        },
    }


def build_metadata_from_plan(plan_path: Path, plan_payload: dict[str, Any]) -> dict[str, Any]:
    plan_report = plan_payload.get("report", {})
    insertion_plan = plan_report.get("insertion_plan") or {}
    return {
        "expected_plan": str(plan_path),
        "expected_plan_provenance_fingerprint": plan_payload.get("provenance_fingerprint"),
        "expected_plan_config_fingerprint": plan_report.get("config_fingerprint"),
        "expected_plan_profile": plan_report.get("profile"),
        "expected_plan_scale": plan_report.get("scale"),
        "expected_plan_adapter_parameters": insertion_plan.get("adapter_parameters"),
        "expected_plan_selected": [row.get("name") for row in insertion_plan.get("selected", [])],
    }


def check_artifact_matches_plan(artifact_payload: dict[str, Any], plan_payload: dict[str, Any]) -> None:
    report = artifact_payload["report"]
    plan_report = plan_payload["report"]
    plan_selected = plan_report.get("insertion_plan", {}).get("selected", [])
    inserted = report.get("inserted", [])
    planned_names = [row.get("name") for row in plan_selected]
    inserted_names = [row.get("name") for row in inserted]
    if inserted_names != planned_names:
        raise ValueError("ARTI built artifact inserted adapters do not match expected plan")
    plan_parameters = plan_report.get("insertion_plan", {}).get("adapter_parameters")
    if report.get("adapter_parameters") != plan_parameters:
        raise ValueError("ARTI built artifact adapter_parameters do not match expected plan")
    for key in ("profile", "scale", "config_fingerprint"):
        if report.get(key) != plan_report.get(key):
            raise ValueError(f"ARTI built artifact {key} does not match expected plan")


def sample_from_args(args: argparse.Namespace) -> Any | None:
    if getattr(args, "sample_json", None) is not None:
        return sample_from_json(args.sample_json)
    if getattr(args, "sample_shape", None) is not None:
        return torch.randn(*parse_sample_shape(args.sample_shape))
    return None


def resolve_cli_locked_path(lock_path: Path, value: str) -> Path:
    target = Path(value)
    if target.is_absolute():
        return target
    return lock_path.parent / target


def create_apply_report(args: argparse.Namespace) -> dict[str, Any]:
    lock_payload = None
    if args.lock is not None:
        lock_payload = validate_build_lock(args.lock, map_location=args.map_location)
        locked_artifact = resolve_cli_locked_path(args.lock, lock_payload["artifact"]["path"]).resolve()
        requested_artifact = args.artifact.resolve()
        if locked_artifact != requested_artifact:
            raise ValueError("ARTI build lock artifact path does not match requested apply artifact")
    model = load_model_reference(args.model, kwargs=load_model_kwargs(args.model_kwargs_json))
    result = apply_adapter(
        model,
        args.artifact,
        sample_batch=sample_from_args(args),
        freeze_base=not args.no_freeze_base,
        map_location=args.map_location,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() in {".md", ".markdown"}:
        args.output.write_text(result.report.to_markdown(), encoding="utf-8")
    else:
        args.output.write_text(json.dumps(result.report.to_dict(), indent=2, sort_keys=True), encoding="utf-8")
    applied = result.report.applied_artifact or {}
    expected_config_fingerprint = None if args.expect_config is None else load_fit_config(args.expect_config).fingerprint
    if expected_config_fingerprint is not None and applied.get("config_fingerprint") != expected_config_fingerprint:
        raise ValueError("ARTI applied adapter report config_fingerprint does not match expected config")
    if args.expect_adapter_state_sha256 is not None and applied.get("adapter_state_sha256") != args.expect_adapter_state_sha256:
        raise ValueError("ARTI applied adapter report adapter_state_sha256 does not match expected value")
    if args.max_adapters is not None and result.adapter_count > args.max_adapters:
        raise ValueError("ARTI applied adapter report exceeds CLI max_adapters")
    saved_state_dict_sha256 = None
    if args.save_state_dict is not None:
        args.save_state_dict.parent.mkdir(parents=True, exist_ok=True)
        saved_state_dict = result.model.state_dict()
        torch.save(saved_state_dict, args.save_state_dict)
        saved_state_dict_sha256 = hash_tensor_state_dict(saved_state_dict)
    deployment_summary = None
    if args.deployment_output is not None:
        if args.lock is None:
            raise ValueError("ARTI apply --deployment-output requires --lock")
        if args.save_state_dict is None:
            raise ValueError("ARTI apply --deployment-output requires --save-state-dict")
        deployment_path = create_deployment_manifest(
            args.deployment_output,
            lock=args.lock,
            artifact=args.artifact,
            applied_report=args.output,
            state_dict=args.save_state_dict,
            map_location=args.map_location,
        )
        deployment_summary = summarize_deployment(validate_deployment_manifest(deployment_path, map_location=args.map_location), path=deployment_path)
    task_graph = cli_apply_task_graph(
        report_path=args.output,
        state_dict_path=args.save_state_dict,
        deployment_path=args.deployment_output,
    )
    task_graph_output = maybe_write_task_graph_artifact(args.task_graph_output, command_kind="apply", task_graph=task_graph)
    return {
        "ok": True,
        "kind": "applied-adapter-report",
        "task_graph": task_graph,
        "task_graph_output": task_graph_output,
        "output": str(args.output),
        "saved_state_dict": None if args.save_state_dict is None else str(args.save_state_dict),
        "saved_state_dict_sha256": saved_state_dict_sha256,
        "deployment": None if deployment_summary is None else deployment_summary["path"],
        "deployment_summary": deployment_summary,
        "model": args.model,
        "artifact": str(args.artifact),
        "adapter_count": result.adapter_count,
        "adapter_state_sha256": applied.get("adapter_state_sha256"),
        "expected_adapter_state_sha256": args.expect_adapter_state_sha256,
        "lock": None if args.lock is None else str(args.lock),
        "lock_report_sha256": None if lock_payload is None else lock_payload["artifact"].get("report_sha256"),
        "cli_max_adapters": args.max_adapters,
        "config_fingerprint": applied.get("config_fingerprint"),
        "expected_config_fingerprint": expected_config_fingerprint,
    }


def cli_apply_task_graph(
    *,
    report_path: Path,
    state_dict_path: Path | None = None,
    deployment_path: Path | None = None,
) -> dict[str, Any]:
    tasks = [
        {"name": "apply-adapter", "kind": "apply", "depends_on": [], "enabled": True, "status": "success"},
        {"name": "write-apply-report", "kind": "report", "depends_on": ["apply-adapter"], "enabled": True, "status": "success"},
    ]
    if state_dict_path is not None:
        tasks.append({"name": "write-state-dict", "kind": "state-dict", "depends_on": ["apply-adapter"], "enabled": True, "status": "success"})
    if deployment_path is not None:
        tasks.append(
            {
                "name": "write-deployment-manifest",
                "kind": "deployment",
                "depends_on": ["write-apply-report", "write-state-dict"],
                "enabled": True,
                "status": "success",
            }
        )
    return {
        "tasks": tasks,
        "artifacts": {
            "apply_report": str(report_path),
            "state_dict": None if state_dict_path is None else str(state_dict_path),
            "deployment": None if deployment_path is None else str(deployment_path),
        },
    }


def enforce_doctor_requirements(summary: dict[str, Any], *, require_cuda_smoke: bool = False, require_jax_smoke: bool = False) -> None:
    """Apply CLI-only doctor requirements to an existing doctor summary."""

    failures = summary.setdefault("failures", [])
    capabilities = summary.get("capabilities", {})
    if require_cuda_smoke and capabilities.get("torch_cuda_smoke_status") != "passed":
        failures.append("CUDA smoke is required but torch_cuda_smoke_status is not passed")
    if require_jax_smoke and capabilities.get("jax_smoke_status") != "passed":
        failures.append("JAX smoke is required but jax_smoke_status is not passed")
    summary["ok"] = not failures


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "inspect":
            if args.section == "profiles":
                summary = {"ok": True, "kind": "profiles", "profiles": list_profiles(phases=args.phases)}
            elif args.section == "scales":
                summary = {"ok": True, "kind": "scales", "scales": list_scales()}
            elif args.section == "plugins":
                summary = {"ok": True, "kind": "plugins", "plugins": list_plugins()}
            else:
                summary = {"ok": True, **capabilities(phases=args.phases)}
        elif args.command == "doctor":
            summary = doctor_report(allow_cpu_torch=args.allow_cpu_torch)
            enforce_doctor_requirements(
                summary,
                require_cuda_smoke=args.require_cuda_smoke,
                require_jax_smoke=args.require_jax_smoke,
            )
            if args.output is not None:
                write_doctor_report(summary, args.output)
                summary["output"] = str(args.output)
            if not summary["ok"]:
                print(json.dumps(summary, indent=2, sort_keys=True))
                return 1
        elif args.command == "docs" and args.kind == "generate":
            summary = generate_docs_report(args)
        elif args.command == "docs" and args.kind == "check":
            summary = check_docs_report(args)
        elif args.command == "schema":
            summary = schema_report(args)
        elif args.command == "pretrained":
            summary = pretrained_cli_report(
                args.pretrained_action,
                getattr(args, "config", None),
                provider=getattr(args, "provider", None),
                lock=getattr(args, "lock", None),
                plan=getattr(args, "plan", None),
                output=getattr(args, "output", None),
                weights=getattr(args, "weights", None),
                include_base=getattr(args, "include_base", False),
            )
            if not summary.get("ok", False):
                print(json.dumps(summary, indent=2, sort_keys=True))
                return 1
        elif args.command == "plan":
            summary = create_fit_plan(args)
        elif args.command == "build":
            summary = create_artifact_report(args)
        elif args.command == "apply":
            summary = create_apply_report(args)
        elif args.command == "lock":
            summary = create_lock_report(args)
        elif args.command == "deployment-manifest":
            summary = create_deployment_report(args)
        elif args.command == "init-config":
            summary = initialize_config(args.path, profile=args.profile, scale=args.scale, force=args.force)
        elif args.command == "validate" and args.kind == "plan":
            summary = summarize_plan(
                validate_plan(args.path),
                expect_config=args.expect_config,
                expect_provenance_fingerprint=args.expect_provenance_fingerprint,
                expect_profile=args.expect_profile,
                expect_scale=args.expect_scale,
                expect_mechanism=parse_expected_mechanism(args.expect_mechanism),
                expect_runtime_fields=parse_expected_runtime_fields(args.expect_runtime_field),
                max_adapters=args.max_adapters,
                max_extra_params=args.max_extra_params,
            )
        elif args.command == "validate" and args.kind == "artifact":
            summary = summarize_artifact(
                validate_artifact(args.path, map_location=args.map_location),
                expect_config=args.expect_config,
                expect_plan=args.expect_plan,
                expect_adapter_state_sha256=args.expect_adapter_state_sha256,
                expect_profile=args.expect_profile,
                expect_scale=args.expect_scale,
                expect_mechanism=parse_expected_mechanism(args.expect_mechanism),
                expect_runtime_fields=parse_expected_runtime_fields(args.expect_runtime_field),
                max_adapters=args.max_adapters,
                max_extra_params=args.max_extra_params,
            )
        elif args.command == "validate" and args.kind == "config":
            summary = summarize_config(
                args.path,
                expect_profile=args.expect_profile,
                expect_scale=args.expect_scale,
                expect_mechanism=parse_expected_mechanism(args.expect_mechanism),
                expect_runtime_fields=parse_expected_runtime_fields(args.expect_runtime_field),
            )
        elif args.command == "validate" and args.kind == "lock":
            summary = summarize_lock(
                validate_build_lock(args.path, map_location=args.map_location),
                path=args.path,
                expect_config=args.expect_config,
                expect_plan=args.expect_plan,
                expect_provenance_fingerprint=args.expect_provenance_fingerprint,
                expect_adapter_state_sha256=args.expect_adapter_state_sha256,
                expect_report_sha256=args.expect_report_sha256,
                expect_profile=args.expect_profile,
                expect_scale=args.expect_scale,
                expect_mechanism=parse_expected_mechanism(args.expect_mechanism),
                expect_runtime_fields=parse_expected_runtime_fields(args.expect_runtime_field),
                max_adapters=args.max_adapters,
                max_extra_params=args.max_extra_params,
            )
        elif args.command == "validate" and args.kind == "state-dict":
            summary = summarize_state_dict(
                args.path,
                map_location=args.map_location,
                expect_state_dict_sha256=args.expect_state_dict_sha256,
            )
        elif args.command == "validate" and args.kind == "task-graph":
            summary = summarize_task_graph(
                validate_task_graph(args.path),
                path=args.path,
                expect_kind=args.expect_kind,
                expect_artifacts=parse_expected_artifacts(args.expect_artifact),
                require_existing_artifacts=args.require_existing_artifacts,
            )
        elif args.command == "validate" and args.kind == "deployment":
            summary = summarize_deployment(
                validate_deployment_manifest(args.path, map_location=args.map_location),
                path=args.path,
                expect_plan=args.expect_plan,
                expect_adapter_state_sha256=args.expect_adapter_state_sha256,
                expect_state_dict_sha256=args.expect_state_dict_sha256,
                expect_profile=args.expect_profile,
                expect_scale=args.expect_scale,
                expect_mechanism=parse_expected_mechanism(args.expect_mechanism),
                expect_runtime_fields=parse_expected_runtime_fields(args.expect_runtime_field),
                max_adapters=args.max_adapters,
                max_extra_params=args.max_extra_params,
            )
        else:
            parser.error("unsupported command")
    except Exception as exc:
        print(f"ARTI validation failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
