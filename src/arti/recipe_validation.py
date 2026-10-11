"""Model-independent deployment checks using existing prediction and scoring."""

from __future__ import annotations

import json
import math
from pathlib import Path
from collections.abc import Mapping

from .recipe_data import json_fingerprint
from .recipe_deployment import inspect_recipe_deployment
from .recipe_prediction import PredictionRunConfig, CausalLMPredictionRunConfig, run_prediction
from .recipe_scoring import PredictionScoringConfig, score_prediction_run
from .serialization import _atomic_json, _load_json, _manifest_file_records, _sidecar_paths


def _criteria(value):
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError("criteria must map metric names to minimum/maximum bounds")
    result = {}
    for name, bounds in value.items():
        if not isinstance(name, str) or not name or not isinstance(bounds, Mapping):
            raise ValueError("criteria require nonempty metric names and bound objects")
        if not bounds or bounds.keys() - {"minimum", "maximum"}:
            raise ValueError("criterion bounds must contain minimum and/or maximum")
        for bound in bounds.values():
            if type(bound) not in {int, float} or not math.isfinite(bound):
                raise ValueError("criterion bounds must be finite numeric values")
        if "minimum" in bounds and "maximum" in bounds and bounds["minimum"] > bounds["maximum"]:
            raise ValueError("criterion minimum exceeds maximum")
        result[name] = dict(bounds)
    return json.loads(json.dumps(result, allow_nan=False))


def _check(metrics, criteria):
    checks = []
    for name, bounds in criteria.items():
        value = metrics.get(name)
        available = type(value) in {int, float} and math.isfinite(value)
        passed = available and (
            ("minimum" not in bounds or value >= bounds["minimum"])
            and ("maximum" not in bounds or value <= bounds["maximum"])
        )
        checks.append(
            {
                "metric": name,
                "bounds": bounds,
                "value": value,
                "passed": passed,
                "status": "passed" if passed else "rejected" if available else "not_evaluable",
            }
        )
    return checks


def _publish(path, report):
    payload = {key: value for key, value in report.items() if key != "sha256"}
    payload["sha256"] = json_fingerprint(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    archive = path.parent / f"{path.stem}.reports" / f"{payload['sha256']}.json"
    archive.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(archive, payload)
    _atomic_json(path, payload)
    return payload


def _prior_runtime(path, plan_sha256):
    if not path.is_file():
        return None
    try:
        value = _load_json(path)
        valid = value.get("sha256") == json_fingerprint(
            {key: item for key, item in value.items() if key != "sha256"}
        )
    except (OSError, ValueError):
        return None
    if (
        valid
        and value.get("prediction", {}).get("plan_sha256") == plan_sha256
        and value.get("runtime_reload_verified") is True
    ):
        return value["sha256"]
    return None


def validate_recipe_deployment(
    path,
    data_path,
    run,
    scoring,
    *,
    criteria=None,
    targets=None,
    resume=False,
    output=None,
    device="cpu",
    task_factory=None,
    layer_factory=None,
    base_model=None,
    stop_requested=None,
    on_batch=None,
    external=None,
):
    """Predict, score and apply caller criteria without adding a training service."""
    if not isinstance(run, (PredictionRunConfig, CausalLMPredictionRunConfig)):
        raise ValueError("run must be an existing prediction run configuration")
    from .recipe_suite import EvaluationSuiteConfig, evaluate_prediction_suite
    from .recipe_assets import DatasetSnapshot

    suite = isinstance(scoring, EvaluationSuiteConfig)
    if not suite and not isinstance(scoring, PredictionScoringConfig):
        raise ValueError("scoring must be PredictionScoringConfig or EvaluationSuiteConfig")
    if not suite and external is not None:
        raise ValueError("external evaluation evidence requires EvaluationSuiteConfig")
    criteria = _criteria(criteria)
    directory = run.output_dir
    report_path = (
        Path(output).resolve()
        if output is not None
        else directory.with_name(f"{directory.name}.validation.json")
    )
    score_directory = report_path.parent / f"{report_path.stem}.scores"
    protected = {
        Path(value.directory if isinstance(value, DatasetSnapshot) else value).resolve()
        for value in (data_path, targets, path)
        if value is not None
    }
    if suite:
        protected.update(Path(value).resolve() for value in (external or {}).values())
        reference_root = (
            Path(targets.directory if isinstance(targets, DatasetSnapshot) else targets).resolve()
            if targets is not None
            else None
        )
        if (
            reference_root is not None
            and reference_root.is_dir()
            and (
                report_path.is_relative_to(reference_root)
                or score_directory.is_relative_to(reference_root)
            )
        ):
            raise ValueError("validation output must be outside reference snapshots")
    if (
        report_path.is_relative_to(directory)
        or report_path in protected
        or score_directory in protected
    ):
        raise ValueError("validation output must be outside prediction/source inputs")
    if path is not None:
        source = Path(path).resolve()
        package_directory = source if source.is_dir() else source.parent
        if (source.is_dir() or (package_directory / "deployment.json").is_file()) and (
            report_path.is_relative_to(package_directory)
            or score_directory.is_relative_to(package_directory)
        ):
            raise ValueError("validation output must be outside the model asset directory")
    report = {
        "ok": False,
        "kind": "deployment-validation",
        "format": "arti.deployment-validation",
        "format_version": 1,
        "status": "running",
        "stage": "inspection",
        "prediction_directory": str(directory),
        "scoring": scoring.to_dict(),
        "criteria": criteria,
        "runtime_reload_verified": False,
        "runtime_evidence": {"origin": "not_run"},
        "inference_completed": False,
        "scoring_completed": False,
        "criteria_passed": None,
        "output": str(report_path),
    }

    def batch(info, receipt):
        report["prediction"] = info
        report["runtime_reload_verified"] = True
        report["runtime_evidence"] = {
            "origin": "current_attempt",
            "plan_sha256": info["plan_sha256"],
        }
        if on_batch is not None:
            on_batch(info, receipt)

    try:
        if path is not None:
            report["inspection"] = inspect_recipe_deployment(path, base_model=base_model)
            if not report["inspection"]["files_available"]:
                raise ValueError("deployment dependency files are unavailable")
            artifact = Path(report["inspection"]["artifact"])
            sidecars = _sidecar_paths(artifact)
            manifest = _load_json(sidecars["manifest"])
            members = {
                artifact.parent / member["file"] for member in _manifest_file_records(manifest)
            }
            members.update(sidecars.values())
            members.add(artifact.parent / "deployment.json")
            if report_path in members or score_directory in members:
                raise ValueError("validation output must not overwrite model dependencies")
        report["stage"] = "prediction"
        predicted = run_prediction(
            path,
            data_path,
            run,
            resume=resume,
            device=device,
            task_factory=task_factory,
            layer_factory=layer_factory,
            base_model=base_model,
            stop_requested=stop_requested,
            on_batch=batch,
        )
        report["prediction"] = predicted
        if not report["runtime_reload_verified"]:
            previous = _prior_runtime(report_path, predicted["plan_sha256"])
            if previous is not None:
                report["runtime_reload_verified"] = True
                report["runtime_evidence"] = {
                    "origin": "prior_validation",
                    "report_sha256": previous,
                    "plan_sha256": predicted["plan_sha256"],
                }
        if predicted["status"] != "completed":
            report.update(ok=True, status="stopped", stage="prediction")
            return _publish(report_path, report)
        report["inference_completed"] = True
        report["stage"] = "scoring"
        scored = (
            evaluate_prediction_suite(directory, scoring, references=targets, external=external)
            if suite
            else score_prediction_run(directory, scoring, targets=targets)
        )
        report["stage"] = "score_publication"
        score_path = score_directory / f"{scored['sha256']}.json"
        score_path.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(score_path, scored)
        report.update(
            scoring_completed=True,
            metrics=scored["metrics"],
            samples=scored["samples"],
            coverage=scored["samples"] / scored["source"]["total_records"]
            if suite
            else scored["coverage"],
            source=scored["source"],
            targets=scored["references"] if suite else scored["targets"],
            scores={"file": str(score_path), "sha256": scored["sha256"]},
            stage="criteria",
        )
        checks = _check(scored["metrics"], criteria)
        if suite:
            checks += [{**check, "scope": "suite"} for check in scored["checks"]]
            checks += [
                {**check, "scope": view}
                for view, result in scored["views"].items()
                for check in result["checks"]
            ]
            report.update(
                evaluation_status=scored["status"],
                complete_evidence=scored["complete_evidence"],
                views={
                    name: {
                        key: value
                        for key, value in result.items()
                        if key not in {"scores", "outcomes"}
                    }
                    for name, result in scored["views"].items()
                },
            )
        accepted = all(check["passed"] for check in checks) if checks else None
        evaluation_failed = suite and scored["status"] == "failed"
        status = (
            "failed"
            if evaluation_failed
            else "evaluated"
            if accepted is None
            else "passed"
            if accepted
            else "rejected"
        )
        if suite and not scored["complete_evidence"] and status in {"evaluated", "passed"}:
            status += "_with_gaps"
        report.update(
            checks=checks,
            criteria_passed=accepted,
            status=status,
            ok=not evaluation_failed and accepted is not False,
            scoring_completed=not evaluation_failed,
            stage="publication",
        )
        published = _publish(report_path, {**report, "stage": "completed"})
        return published
    except BaseException as error:
        failed = {
            **report,
            "ok": False,
            "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            "error_type": type(error).__name__,
        }
        try:
            _publish(report_path.with_suffix(".failure.json"), failed)
        except OSError as publication_error:
            raise error from publication_error
        raise


__all__ = ["validate_recipe_deployment"]
