"""Read-only task project status from existing lifecycle artifacts.

This module never constructs a task, imports a caller callback, probes a host,
or resumes work. Storage integrity, recorded outcomes and runtime qualification
remain separate facts.
"""

from __future__ import annotations

import hashlib
from importlib.resources import files
import json
from pathlib import Path
import shlex

from .recipe_assets import _entry, inspect_dataset_snapshot
from .recipe_data import json_fingerprint
from .recipe_deployment import _recipe_declaration, inspect_recipe_deployment
from .recipe_export import inspect_result_export
from .recipe_prediction import inspect_prediction
from .recipe_recovery import _checkpoint_metadata, _output_path, plan_recipe_resume
from .recipe_suite import evaluation_source_identity
from .task_project import check_task_project, read_task_project


def _json(path):
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("stored receipt must be a JSON object")
    return value


def _receipt(path, formats):
    value = _json(path)
    legacy_runtime = (
        "arti.task-runtime-execution" in formats
        and value.get("format") is None
        and value.get("kind") == "task-runtime-execution"
        and value.get("inspection", {}).get("format") == "arti.task-runtime-inspection"
    )
    if not legacy_runtime and (value.get("format") not in formats or value.get("format_version") != 1):
        raise ValueError("unsupported stored receipt format/version")
    sealed = value.get("sha256")
    if sealed is not None and sealed != json_fingerprint(
        {key: item for key, item in value.items() if key != "sha256"}
    ):
        raise ValueError("stored receipt digest differs")
    return value, "self_digest_verified" if sealed is not None else "unsealed_stored_receipt"


def _training(path):
    record = _json(path / "run.json")
    if record.get("format") != "arti.training-run" or record.get("format_version") != 1:
        raise ValueError("unsupported training run format/version")
    report_path = path / "task_result.json"
    report = _json(report_path) if report_path.is_file() else {}
    result = {
        "status": report.get("status", record.get("status", "unrecorded")),
        "stage": report.get("stage", "training"),
        "run_id": record.get("run_id"),
        "last_checkpoint": None,
        "best_checkpoint": None,
        "selected_checkpoint": None,
        "selection": report.get("selection"),
        "checkpoint_validation": {},
        "evidence_validation": "recorded_metadata_only",
    }
    if record.get("latest_checkpoint") is None:
        result.update(action="no_checkpoint", remaining_optimizer_steps=None)
        return result
    plan = plan_recipe_resume(path)
    result.update(
        resume_plan=plan,
        action=plan["action"],
        checkpoint_step=plan["checkpoint_step"],
        target_optimizer_step=plan["target_optimizer_step"],
        remaining_optimizer_steps=plan["remaining_optimizer_steps"],
        task_recipe_sha256=plan["task_recipe_sha256"],
        last_checkpoint=plan["checkpoint"],
        evidence_validation="last_checkpoint_metadata_and_file_digests_verified",
    )
    result["checkpoint_validation"]["last"] = "verified"
    for name in ("best", "selected"):
        value = report.get(f"{name}_checkpoint")
        if value is not None:
            checkpoint = _output_path(value, path, report).resolve()
            _manifest, state = _checkpoint_metadata(checkpoint)
            if state["task_recipe_sha256"] != plan["task_recipe_sha256"]:
                raise ValueError(f"{name} checkpoint belongs to a different task")
            if name == "selected" and (report.get("selection") or {}).get("selected_step") != state["global_step"]:
                raise ValueError("selected checkpoint step differs from the recorded selection")
            result[f"{name}_checkpoint"] = str(checkpoint)
            result[f"{name}_checkpoint_step"] = state["global_step"]
            result["checkpoint_validation"][name] = "verified"
    preparation = path / "data_preparation.json"
    if preparation.is_file():
        result["preparation"] = _json(preparation)
        result["preparation_attribution"] = "stored_training_receipt; counters not remeasured"
    return result


def _model(path):
    artifact, manifest, declaration = _recipe_declaration(path)
    if declaration["format"] == "arti.causal-lm-recipe":
        return {
            "status": "present",
            "artifact": str(artifact),
            "task_recipe_sha256": declaration["sha256"],
            "model_manifest_sha256": json_fingerprint(manifest),
            "selection": manifest["architecture"]["config"].get("selection"),
            "evidence_validation": "serialized_package_metadata_and_file_digests_verified",
            "dependency_validation": "not_checked; tokenizer/host import excluded",
            "runtime_reload_verified": False,
        }
    result = inspect_recipe_deployment(path)
    return {
        **result,
        "status": "present" if result["ok"] else "missing_dependencies",
        "evidence_validation": "serialized_package_and_resource_file_digests_verified",
    }


def _prediction(path):
    result = inspect_prediction(path)
    if result["committed_records"]:
        source = evaluation_source_identity(path, allow_partial=True)
        result.update({key: source[key] for key in (
            "execution_sha256", "input_sha256", "task_recipe_sha256", "files"
        ) if key in source})
    return {
        **result,
        "status": "completed" if result["action"] == "already_completed"
        else "blocked" if result["action"] == "recovery_not_supported"
        else "awaiting_publication" if result["remaining_records"] == 0
        else "partial",
        "evidence_validation": "pinned_plan_and_committed_part_closures_verified",
    }


def _dataset(path):
    result = inspect_dataset_snapshot(path)
    return {
        "status": "completed",
        "sha256": result["sha256"],
        "records": result["records"]["count"],
        "records_sha256": result["records"]["sha256"],
        "inventory": result["inventory"],
        "fields": result["fields"],
        "splits": result["splits"],
        "evidence_validation": "record_and_asset_file_digests_verified",
    }


def _cache(path):
    entries = path / "entries"
    keys = sorted(member.name for member in entries.iterdir() if member.is_dir()) \
        if entries.is_dir() else []
    for key in keys:
        _entry(path, key)
    return {
        "status": "present",
        "verified_entries": len(keys),
        "reused": None,
        "produced": None,
        "counter_attribution": "not_recorded; cache presence is not a reuse counter",
        "evidence_validation": "existing_entry_identities_and_file_closures_verified",
    }


def _preparation(path):
    value, validation = _receipt(path, {"arti.asset-preparation-report"})
    return {
        "status": "completed" if value["complete"] else "partial",
        "dataset_sha256": value["dataset_sha256"],
        "transform": value["transform"],
        "produced": value["produced"],
        "reused": value["reused"],
        "unique_sources": value["unique_sources"],
        "assets": value["assets"],
        "evidence_validation": validation,
        "counter_attribution": "stored_preparation_receipt; counters not remeasured",
    }


def _evaluation(path):
    value, validation = _receipt(path, {"arti.evaluation-suite", "arti.prediction-scores",
                                      "arti.deployment-validation"})
    result = {
        "status": value.get("status", "evaluated"),
        "recorded_ok": value.get("ok"),
        "criteria_passed": value.get("criteria_passed"),
        "complete_evidence": value.get("complete_evidence"),
        "source": value.get("source"),
        "metrics": value.get("metrics"),
        "checks": value.get("checks"),
        "samples": value.get("samples"),
        "selection": value.get("selection"),
        "coverage": value.get("coverage"),
        "original_population": (value.get("source") or {}).get("total_records"),
        "references": value.get("references", value.get("targets")),
        "evidence_validation": validation,
        "scorer_execution": "not_repeated",
        "views": {},
    }
    for name, view in value.get("views", {}).items():
        result["views"][name] = {
            key: view.get(key) for key in ("status", "origin", "coverage", "metrics", "checks",
                                         "aggregate_available", "availability", "requires")
        }
        phases = {}
        for outcome in view.get("outcomes", []):
            phase = (outcome.get("execution") or {}).get("phase", "unrecorded")
            phases[phase] = phases.get(phase, 0) + 1
        result["views"][name]["execution_phases"] = phases
    if value["format"] == "arti.deployment-validation":
        result.update(
            runtime=value.get("runtime"),
            prediction=value.get("prediction"),
            evaluation=value.get("evaluation"),
            deployment_sha256=value.get("deployment_sha256"),
        )
    return result


def _runtime(path):
    value, validation = _receipt(path, {"arti.task-runtime-execution",
                                      "arti.task-runtime-inspection"})
    inspection = value.get("inspection", value)
    inspection_only = value.get("format") == "arti.task-runtime-inspection"
    return {
        "status": "not_run" if inspection_only else value.get("status", "not_run"),
        "compatibility_status": inspection.get("status"),
        "evidence_kind": "inspection_only" if inspection_only else "execution_receipt",
        "recorded_ok": value.get("ok"),
        "application_executed": value.get("application_executed", False),
        "deployment_sha256": value.get("deployment_sha256", inspection.get("deployment_sha256")),
        "profile_sha256": value.get("profile_sha256", inspection.get("profile_sha256")),
        "origin": value.get("origin", value.get("evidence_origin", "local; stored receipt")),
        "locally_verified_execution": value.get("locally_verified_execution"),
        "declared": inspection.get("profile", inspection.get("declared")),
        "observed": value.get("observed", inspection.get("observed")),
        "enforced": value.get("enforced", {}),
        "unenforced": value.get("unenforced", inspection.get("unenforced", {})),
        "unsupported": inspection.get("unsupported_enforcement"),
        "missing_dependencies": inspection.get("missing_dependencies"),
        "incompatible_dependencies": inspection.get("incompatible_dependencies"),
        "evidence_validation": validation,
        "runtime_verification": "stored_receipt_only; no current host probe or execution",
    }


def _export(path):
    result = inspect_result_export(path)
    return {**result, "status": "completed", "evidence_validation": "exact_archive_closure_verified"}


_INSPECTORS = {
    "training_run": _training,
    "model": _model,
    "dataset": _dataset,
    "cache": _cache,
    "preparation_report": _preparation,
    "prediction_run": _prediction,
    "evaluation": _evaluation,
    "deployment": _model,
    "result_export": _export,
    "runtime_receipt": _runtime,
}


def _links(artifacts):
    results = []

    def compare(name, a, akey, b, bkey):
        left = artifacts.get(a, {}).get("details", {})
        right = artifacts.get(b, {}).get("details", {})
        for key in akey.split("."):
            left = left.get(key) if isinstance(left, dict) else None
        for key in bkey.split("."):
            right = right.get(key) if isinstance(right, dict) else None
        available = left is not None and right is not None
        results.append({
            "name": name,
            "status": "matched" if available and left == right else "mismatch" if available
            else "not_checked",
            "left": a,
            "right": b,
            "left_identity": left,
            "right_identity": right,
        })

    compare("training_model_task", "training_run", "task_recipe_sha256", "model", "task_recipe_sha256")
    compare("training_model_selected_step", "training_run", "selected_checkpoint_step", "model", "selection.selected_step")
    compare("prediction_model", "prediction_run", "model_manifest_sha256", "model", "model_manifest_sha256")
    compare("prediction_deployment", "prediction_run", "deployment_sha256", "deployment", "deployment_sha256")
    compare("evaluation_prediction_plan", "evaluation", "source.plan_sha256", "prediction_run", "plan_sha256")
    compare("evaluation_prediction_bytes", "evaluation", "source.predictions_sha256", "prediction_run", "result_sha256")
    compare("evaluation_prediction_model", "evaluation", "source.model_manifest_sha256", "prediction_run", "model_manifest_sha256")
    compare("evaluation_prediction_input", "evaluation", "source.input_sha256", "prediction_run", "input_sha256")
    compare("evaluation_prediction_execution", "evaluation", "source.execution_sha256", "prediction_run", "execution_sha256")
    compare("evaluation_prediction_population", "evaluation", "source.total_records", "prediction_run", "total_records")
    compare("evaluation_deployment", "evaluation", "source.deployment_sha256", "deployment", "deployment_sha256")
    compare("export_prediction_plan", "result_export", "source.plan_sha256", "prediction_run", "plan_sha256")
    compare("export_prediction_bytes", "result_export", "source.predictions_sha256", "prediction_run", "result_sha256")
    compare("export_prediction_execution", "result_export", "source.execution_sha256", "prediction_run", "execution_sha256")
    compare("export_prediction_input", "result_export", "source.input_sha256", "prediction_run", "input_sha256")
    compare("export_deployment", "result_export", "source.deployment_sha256", "deployment", "deployment_sha256")
    compare("runtime_deployment", "runtime_receipt", "deployment_sha256", "deployment", "deployment_sha256")
    compare("preparation_dataset", "preparation_report", "dataset_sha256", "dataset", "sha256")
    return results


def _scaffold_resume_argv(project, factory):
    """Recognize the packaged application driver without executing caller code."""
    driver = project.project_root / "workflow.py"
    if factory.get("reference") != "task_app:build_task" or not driver.is_file() \
        or not (project.project_root / "code/task_app.py").is_file():
        return None
    try:
        original = files("arti").joinpath("templates", "task", "workflow.py").read_text(encoding="utf-8")
        if driver.read_text(encoding="utf-8") != original or project.resolved_recipe().factory.to_dict() != factory:
            return None
    except (OSError, ValueError, TypeError, KeyError):
        return None
    return ["python", str(driver), "predict", "--resume"]


def _next_actions(project, artifacts):
    actions = []
    training = artifacts.get("training_run", {})
    prediction = artifacts.get("prediction_run", {})
    initial = "prepare" if artifacts.get("preparation_report", {}).get("status") == "not_created" \
        else "train" if training.get("status") == "not_created" else None
    if initial is not None:
        try:
            factory = project.resolved_recipe().factory.to_dict()
            driver_argv = _scaffold_resume_argv(project, factory)
        except (OSError, ValueError, TypeError, KeyError):
            driver_argv = None
        if driver_argv is not None:
            argv = [*driver_argv[:2], initial]
            actions.append({
                "artifact": "preparation_report" if initial == "prepare" else "training_run",
                "action": "prepare_inputs" if initial == "prepare" else "start_training",
                "reason": "Use the generated application driver to prepare declared inputs."
                if initial == "prepare" else "Use the generated application driver to start the declared native training run.",
                "command": shlex.join(argv), "argv": argv,
            })
    for name, entry in artifacts.items():
        if entry["status"] == "invalid":
            actions.append({"artifact": name, "action": "repair_or_rebind_artifact",
                            "reason": entry["error"]["message"], "command": None})
    if (training.get("details") or {}).get("action") in {"resume_training", "finalize"}:
        actions.append({"artifact": "training_run", "action": "inspect_resume_plan",
                        "reason": "Preserve the original optimizer-step horizon.",
                        "command": f"arti recipe resume-run {shlex.quote(training['path'])} --plan-only"})
    detail = prediction.get("details") or {}
    if detail.get("action") == "recovery_not_supported":
        actions.append({"artifact": "prediction_run", "action": "caller_reconciliation_required",
                        "reason": "Ambiguous external effects must be reconciled by the caller; no retry is suggested.",
                        "command": None, "blockers": detail.get("episode_recovery", {}).get("blockers")})
    elif detail.get("action") in {"resume_prediction", "finalize", "publish_episode_part"}:
        argv = ["arti", "recipe", "predict-run"]
        reason = "Use the existing journal; completed outputs are retained."
        source_name = "deployment" if detail.get("deployment_sha256") is not None else "model"
        source = artifacts.get(source_name, {})
        source_details = source.get("details") or {}
        key = "deployment_sha256" if source_name == "deployment" else "model_manifest_sha256"
        matched = source.get("status") in {"present", "completed"} and (
            source_details.get(key) is not None and source_details[key] == detail.get(key)
        )
        command = None
        if detail["action"] == "resume_prediction" and not matched:
            reason = f"Uncommitted work requires the pinned {source_name} identity; restore or rebind that artifact before resuming."
            argv = None
        else:
            if detail["action"] == "resume_prediction":
                argv.append(source["path"])
            argv.extend(["--run-dir", prediction["path"], "--resume"])
            factory = source_details.get("task_factory") or {}
            if detail["action"] == "resume_prediction" and source_details.get("requires_task_factory"):
                reference = factory.get("reference")
                if reference is None:
                    argv = None
                    reason = "The pinned generic task needs its explicit caller factory; restore its declaration before resuming."
                else:
                    argv.extend(["--task-factory", reference])
                    reason += " The saved caller factory must be installed/importable in the command environment; saved factory kwargs are reused."
            if argv is not None:
                command = shlex.join(argv)
                # This optional application-owned driver is a package template,
                # not another runner. Check its template text without importing it.
                driver_argv = _scaffold_resume_argv(project, factory) \
                    if detail["action"] == "resume_prediction" and matched else None
                if driver_argv is not None:
                    argv = driver_argv
                    command = shlex.join(argv)
                    reason = "Use the generated application driver, which binds its caller code path and the pinned deployment; completed outputs are retained."
        actions.append({"artifact": "prediction_run", "action": detail["action"],
                        "reason": reason, "command": command, "argv": argv,
                        "source_artifact": source_name})
    for name, entry in artifacts.items():
        if entry["status"] == "not_created":
            actions.append({"artifact": name, "action": "not_created", "command": None,
                            "reason": "Declared future output has no artifact yet; creation remains explicit."})
    return actions


def task_project_status(path, *, explain=False):
    """Consolidate stored lifecycle evidence without executing user code or writing files.

    ``ok`` describes integrity/configuration checks. ``lifecycle_complete`` is
    limited to declared artifacts and never implies task quality or current host
    qualification. Evaluation and runtime results have independent summaries.
    """
    project = read_task_project(path)
    checked = check_task_project(path)
    artifacts = {}
    for name, artifact in project.resolved_paths("artifacts").items():
        entry = {"path": str(artifact), "status": "not_created", "details": None}
        if artifact.exists():
            try:
                inspector = _INSPECTORS.get(name)
                if inspector is None:
                    entry.update(status="present", details={"evidence_validation": "not_checked"})
                else:
                    details = inspector(artifact)
                    entry.update(status=details["status"], details=details)
            except Exception as error:
                entry.update(status="invalid", error={"type": type(error).__name__, "message": str(error)})
        artifacts[name] = entry
    links = _links(artifacts)
    artifact_integrity = all(entry["status"] != "invalid" for entry in artifacts.values()) \
        and all(link["status"] != "mismatch" for link in links)
    integrity = checked["ok"] and artifact_integrity
    unfinished = {"not_created", "invalid", "partial", "blocked", "awaiting_publication",
                  "running", "interrupted", "failed", "missing_dependencies", "not_run"}
    evaluation = artifacts.get("evaluation", {}).get("details") or {}
    runtime = artifacts.get("runtime_receipt", {}).get("details") or {}
    result = {
        "ok": integrity,
        "format": "arti.task-project-status",
        "format_version": 1,
        "name": project.name,
        "project": str(project.project_file),
        "project_sha256": hashlib.sha256(Path(project.project_file).read_bytes()).hexdigest(),
        "configuration": checked,
        "artifacts": artifacts,
        "links": links,
        "summary": {
            "lifecycle_complete": bool(artifacts) and artifact_integrity and all(
                entry["status"] not in unfinished for entry in artifacts.values()
            ),
            "completion_scope": "declared artifact lifecycle only",
            "configuration_valid": checked["ok"],
            "artifact_integrity": artifact_integrity,
            "evaluation_status": evaluation.get("status", "not_run"),
            "evaluation_complete_evidence": evaluation.get("complete_evidence"),
            "criteria_passed": evaluation.get("criteria_passed"),
            "runtime_status": runtime.get("status", "not_run"),
            "current_runtime_qualified": False,
            "invalid_artifacts": [name for name, entry in artifacts.items() if entry["status"] == "invalid"],
            "missing_artifacts": [name for name, entry in artifacts.items() if entry["status"] == "not_created"],
        },
        "loads_model_weights": False,
        "imports_caller_code": False,
        "probes_environment": False,
        "executes_actions": False,
        "writes_files": False,
    }
    if explain:
        result["next_actions"] = _next_actions(project, artifacts)
        result["next_actions"].extend({
            "artifact": link["left"], "action": "reconcile_identity_mismatch", "command": None,
            "reason": f"{link['name']}: the declared artifacts describe different lifecycle evidence.",
        } for link in links if link["status"] == "mismatch")
        result["interpretation"] = {
            "integrity": "Digests verify stored bytes and declared links; they do not authenticate an external producer.",
            "evaluation": "Recorded coverage, failed/unavailable views and caller criteria remain separate from lifecycle completion.",
            "runtime": "Only saved receipts are shown. Current environment compatibility and enforcement are not tested.",
        }
    return result


def explain_task_project(path):
    """Return status plus explicit, non-executing next-step explanations."""
    return task_project_status(path, explain=True)


def render_task_project_status(report):
    """Render a compact human-readable view of the same structured evidence."""
    summary = report["summary"]
    lines = [f"Task project: {report['name']}",
             f"Integrity: {'ok' if report['ok'] else 'issues'}",
             f"Declared lifecycle: {'complete' if summary['lifecycle_complete'] else 'incomplete'}"]
    for issue in report["configuration"].get("issues", []):
        lines.append(f"Config {issue['field']}: {issue['message']}")
    for name, entry in report["artifacts"].items():
        lines.append(f"{name}: {entry['status']}")
        details = entry.get("details") or {}
        if name == "training_run" and "checkpoint_step" in details:
            lines.append(f"  optimizer steps {details['checkpoint_step']}/{details['target_optimizer_step']}; remaining {details['remaining_optimizer_steps']}")
        if name == "prediction_run":
            lines.append(f"  committed {details.get('committed_records', '?')}/{details.get('total_records', '?')}; remaining {details.get('remaining_records', '?')}")
        if name == "cache":
            lines.append(f"  verified entries {details.get('verified_entries', '?')}; reuse counters not recorded here")
        if name == "preparation_report":
            lines.append(f"  produced {details.get('produced', '?')}; reused {details.get('reused', '?')} (stored receipt)")
        if name == "evaluation":
            for view, evidence in details.get("views", {}).items():
                coverage = evidence.get("coverage") or {}
                lines.append(f"  {view}: {evidence.get('status', 'unrecorded')}; scored {coverage.get('scored', '?')}/{coverage.get('total', '?')}; origin {evidence.get('origin', 'unrecorded')}")
        if entry.get("error"):
            lines.append(f"  {entry['error']['type']}: {entry['error']['message']}")
    for link in report["links"]:
        if link["status"] == "mismatch":
            lines.append(f"Link {link['name']}: identity mismatch")
    lines.extend([f"Evaluation: {summary['evaluation_status']}; criteria {summary['criteria_passed']}",
                  f"Runtime: {summary['runtime_status']}; current host untested"])
    for action in report.get("next_actions", []):
        lines.append(f"Next ({action['artifact']}): {action['reason']}")
        if action.get("command"):
            lines.append(f"  {action['command']}")
    return "\n".join(lines)


__all__ = ["task_project_status", "explain_task_project", "render_task_project_status"]
