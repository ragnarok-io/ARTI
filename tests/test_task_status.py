from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

import arti
from arti.recipe_data import json_fingerprint
from arti.recipe_workflow import write_json
from arti.task_status import explain_task_project, render_task_project_status, task_project_status
from test_recipe_assets import forbidden
from test_recipe_suite import fixture, suite


def project(tmp_path, artifacts):
    path = tmp_path / "project.json"
    write_json(path, {
        "format": "arti.task-project", "format_version": 1, "name": "generated-task",
        "configs": {"recipe": "training/resolved_recipe.json"},
        "artifacts": {name: str(Path(value).relative_to(tmp_path)) for name, value in artifacts.items()},
    })
    return path


def inventory(path):
    return {
        str(file.relative_to(path)): (file.stat().st_mtime_ns, hashlib.sha256(file.read_bytes()).hexdigest())
        for file in path.rglob("*") if file.is_file()
    }


def test_complete_native_file_status_preserves_closures_without_caller_import_or_writes(tmp_path, monkeypatch):
    package, _requests, _files, directory, targets, _rows = fixture(tmp_path)
    evaluated = tmp_path / "evaluation.json"
    arti.evaluate_prediction_suite(directory, suite(), references=targets, output=evaluated)
    from test_recipe_export import config
    archive = tmp_path / "export.zip"
    arti.export_prediction_results(directory, config(archive="zip"), archive, deployment=package,
                                   evidence={"evaluation": evaluated})
    path = project(tmp_path, {
        "training_run": tmp_path / "training", "model": package,
        "dataset": tmp_path / "snapshot", "cache": tmp_path / "cache",
        "prediction_run": directory, "evaluation": evaluated, "deployment": package,
        "result_export": archive,
    })
    monkeypatch.setattr(arti.PythonFactoryConfig, "resolve", forbidden)
    import arti.recipe_runtime as runtime
    monkeypatch.setattr(runtime, "_python_probe", forbidden)
    before = inventory(tmp_path)
    report = explain_task_project(path)
    assert report["ok"] and report["summary"]["lifecycle_complete"]
    assert not report["summary"]["current_runtime_qualified"]
    assert report["summary"]["evaluation_complete_evidence"] is True
    training = report["artifacts"]["training_run"]["details"]
    assert training["checkpoint_step"] == 2 and training["remaining_optimizer_steps"] == 0
    assert training["selected_checkpoint"] and training["checkpoint_validation"]["selected"] == "verified"
    prediction = report["artifacts"]["prediction_run"]["details"]
    assert prediction["committed_records"] == 7 and prediction["remaining_records"] == 0
    assert prediction["output_files"] == 21
    assert report["artifacts"]["dataset"]["details"]["records"] == 16
    assert report["artifacts"]["cache"]["details"]["reused"] is None
    assert report["artifacts"]["evaluation"]["details"]["views"]["labels"]["coverage"]["total"] == 7
    assert all(link["status"] != "mismatch" for link in report["links"])
    assert inventory(tmp_path) == before
    assert "optimizer steps 2/2" in render_task_project_status(report)
    assert all(report[flag] is False for flag in ("loads_model_weights", "imports_caller_code",
                                                "probes_environment", "executes_actions", "writes_files"))


def test_partial_prediction_and_failed_or_unavailable_views_remain_distinct(tmp_path, monkeypatch):
    package, _requests, _files, directory, targets, rows = fixture(tmp_path, interrupt=True)
    evidence = tmp_path / "evaluation.json"
    from arti.recipe_workflow import write_jsonl
    write_jsonl(targets, [{**row, "available": False} for row in rows])
    evaluated = arti.evaluate_prediction_suite(directory, suite(external=True), references=targets,
                                               allow_partial=True, output=evidence)
    path = project(tmp_path, {"prediction_run": directory, "model": package, "evaluation": evidence,
                              "result_export": tmp_path / "future.zip"})
    monkeypatch.setattr(arti.PythonFactoryConfig, "resolve", forbidden)
    report = explain_task_project(path)
    assert report["ok"] and not report["summary"]["lifecycle_complete"]
    assert report["artifacts"]["prediction_run"]["status"] == "partial"
    assert report["summary"]["evaluation_status"] == evaluated["status"]
    assert report["summary"]["evaluation_complete_evidence"] is False
    views = report["artifacts"]["evaluation"]["details"]["views"]
    assert views["labels"]["coverage"]["unavailable_ids"]
    assert views["outcome"]["origin"] == "external"
    assert views["format"]["coverage"]["total"] == 7
    assert report["artifacts"]["result_export"]["status"] == "not_created"
    assert any(action["action"] == "resume_prediction" for action in report["next_actions"])
    assert "not_created" in render_task_project_status(report)


@pytest.mark.parametrize("changed", ["prediction", "dataset", "evaluation", "model"])
def test_each_changed_artifact_is_reported_without_hiding_other_progress(tmp_path, changed):
    package, _requests, _files, directory, targets, _rows = fixture(tmp_path)
    evidence = tmp_path / "evaluation.json"
    arti.evaluate_prediction_suite(directory, suite(), references=targets, output=evidence)
    path = project(tmp_path, {"prediction_run": directory, "dataset": tmp_path / "snapshot",
                              "evaluation": evidence, "model": package})
    if changed == "prediction":
        target = next((directory / "parts").glob("*/files/**/*.*"))
        role = "prediction_run"
    elif changed == "dataset":
        target, role = tmp_path / "snapshot" / "records.jsonl", "dataset"
    elif changed == "model":
        target, role = package / "model.st", "model"
    else:
        target, role = evidence, "evaluation"
    target.write_bytes(target.read_bytes() + b"changed")
    before = inventory(tmp_path)
    report = explain_task_project(path)
    assert not report["ok"] and not report["summary"]["lifecycle_complete"]
    assert report["artifacts"][role]["status"] == "invalid"
    assert set(report["summary"]["invalid_artifacts"]) == {role}
    assert inventory(tmp_path) == before


def test_receipt_identity_mismatch_is_not_a_completed_project(tmp_path):
    package, _requests, _files, directory, targets, _rows = fixture(tmp_path)
    evidence = tmp_path / "evaluation.json"
    value = arti.evaluate_prediction_suite(directory, suite(), references=targets, output=evidence)
    value["source"]["plan_sha256"] = "different-plan"
    value.pop("sha256")
    write_json(evidence, {**value, "sha256": json_fingerprint(value)})
    path = project(tmp_path, {"prediction_run": directory, "model": package, "evaluation": evidence})
    report = task_project_status(path)
    assert not report["ok"]
    assert next(link for link in report["links"] if link["name"] == "evaluation_prediction_plan")["status"] == "mismatch"
    assert not report["summary"]["lifecycle_complete"]
    assert "next_actions" not in report


def test_runtime_not_run_limits_and_external_origin_are_stored_evidence_only(tmp_path, monkeypatch):
    package, _requests, _files, directory, _targets, _rows = fixture(tmp_path)
    deployment = json.loads((package / "deployment.json").read_text())
    runtime = tmp_path / "runtime.json"
    receipt = {
        "format": "arti.task-runtime-execution", "format_version": 1,
        "status": "not_run", "ok": False, "application_executed": False,
        "deployment_sha256": deployment["sha256"], "origin": "backend-reported",
        "locally_verified_execution": False,
        "inspection": {"declared": {"limits": {"application-limit": 1}},
                       "unsupported_enforcement": ["application-limit"]},
        "enforced": {}, "unenforced": {"application-limit": 1},
    }
    write_json(runtime, {**receipt, "sha256": json_fingerprint(receipt)})
    path = project(tmp_path, {"deployment": package, "runtime_receipt": runtime,
                              "prediction_run": directory})
    monkeypatch.setattr(arti.PythonFactoryConfig, "resolve", forbidden)
    report = explain_task_project(path)
    assert report["ok"] and not report["summary"]["lifecycle_complete"]
    assert report["summary"]["runtime_status"] == "not_run"
    saved = report["artifacts"]["runtime_receipt"]["details"]
    assert saved["origin"] == "backend-reported" and saved["enforced"] == {}
    assert saved["unsupported"] == ["application-limit"]
    assert not saved["locally_verified_execution"] and not report["summary"]["current_runtime_qualified"]


def test_training_before_checkpoint_and_future_artifacts_do_not_error(tmp_path):
    from test_custom_task_workflow import setup
    cfg, _rows = setup(tmp_path / "data", steps=2)
    training = tmp_path / "training"
    training.mkdir()
    write_json(training / "resolved_recipe.json", cfg.to_dict())
    write_json(training / "run.json", {"format": "arti.training-run", "format_version": 1,
                                      "status": "running", "latest_checkpoint": None})
    path = project(tmp_path, {"training_run": training, "model": tmp_path / "future-model"})
    report = explain_task_project(path)
    assert report["ok"] and not report["summary"]["lifecycle_complete"]
    assert report["artifacts"]["training_run"]["details"]["action"] == "no_checkpoint"
    assert report["artifacts"]["model"]["status"] == "not_created"


def test_failed_view_does_not_turn_into_quality_pass(tmp_path):
    package, _requests, _files, directory, targets, _rows = fixture(tmp_path)
    evidence = tmp_path / "evaluation.json"
    config = suite()
    broken = replace(config, views=(replace(config.views[0], scoring=arti.PredictionScoringConfig(
        "caller_missing_evaluator:score", "caller-revision-v1")), config.views[1]))
    evaluated = arti.evaluate_prediction_suite(directory, broken, references=targets, output=evidence)
    path = project(tmp_path, {"model": package, "prediction_run": directory, "evaluation": evidence})
    report = task_project_status(path)
    assert report["ok"] and evaluated["status"] == "failed"
    assert report["summary"]["evaluation_status"] == "failed"
    assert report["artifacts"]["evaluation"]["details"]["views"]["labels"]["status"] == "failed"
    assert not report["summary"]["lifecycle_complete"]


def test_ambiguous_episode_intent_explain_never_suggests_automatic_retry(tmp_path, monkeypatch):
    package, _requests, _files, _directory, _targets, _rows = fixture(tmp_path)
    from test_recipe_episode import episodes, records
    from arti.recipe_workflow import write_jsonl
    source = tmp_path / "episode-input.jsonl"
    write_jsonl(source, records(horizons=(1,)))
    destination = tmp_path / "episode-run"
    config = arti.PredictionRunConfig(destination, batch_size=1,
                                      episodes=episodes(effects="external", recovery="none"))
    # Only an intent is saved over a local fixture. No environment action occurs.
    with pytest.raises(KeyboardInterrupt):
        arti.run_prediction(package, source, config,
                            task_factory="examples.asset_task_recipe:build_task", on_episode_event=lambda event: (
            (_ for _ in ()).throw(KeyboardInterrupt()) if event["kind"] == "intent" else None
        ))
    path = project(tmp_path, {"prediction_run": destination, "model": package})
    monkeypatch.setattr(arti.PythonFactoryConfig, "resolve", forbidden)
    before = inventory(tmp_path)
    report = explain_task_project(path)
    assert report["artifacts"]["prediction_run"]["status"] == "blocked"
    action = next(item for item in report["next_actions"] if item["action"] == "caller_reconciliation_required")
    assert action["command"] is None and action["blockers"]
    assert not any(item["action"] == "resume_prediction" for item in report["next_actions"])
    assert inventory(tmp_path) == before


@pytest.mark.parametrize("sealed", [False, True])
def test_legacy_not_run_runtime_receipts_and_compatible_inspections_do_not_claim_execution(tmp_path, sealed):
    from test_custom_task_workflow import setup
    cfg, _rows = setup(tmp_path / "data", steps=2)
    training = tmp_path / "training"
    training.mkdir()
    write_json(training / "resolved_recipe.json", cfg.to_dict())
    receipt = {
        "format": "arti.task-runtime-inspection", "format_version": 1,
        "status": "compatible", "ok": True, "declared": {}, "unsupported_enforcement": [],
    }
    if not sealed:
        receipt = {"kind": "task-runtime-execution", "status": "not_run", "ok": False,
                   "application_executed": False, "inspection": receipt}
    else:
        receipt = {**receipt, "sha256": json_fingerprint(receipt)}
    path = tmp_path / "runtime.json"
    write_json(path, receipt)
    report = task_project_status(project(tmp_path, {"runtime_receipt": path}))
    assert report["ok"] and report["summary"]["runtime_status"] == "not_run"
    assert not report["summary"]["lifecycle_complete"]
    assert report["artifacts"]["runtime_receipt"]["details"]["compatibility_status"] == "compatible"


def test_valid_changed_deployment_keeps_same_model_but_exposes_stale_result_bindings(tmp_path, monkeypatch):
    package, _requests, _files, directory, targets, _rows = fixture(tmp_path)
    evidence = tmp_path / "evaluation.json"
    arti.evaluate_prediction_suite(directory, suite(), references=targets, output=evidence)
    from test_recipe_export import config
    archive = tmp_path / "export.zip"
    arti.export_prediction_results(directory, config(archive="zip"), archive,
                                   deployment=package, evidence={"evaluation": evidence})
    caller = tmp_path / "changed_task.py"
    caller.write_text("raise AssertionError('status must not import changed caller code')\n", encoding="utf-8")
    changed = tmp_path / "changed-deployment"
    packaged = arti.package_recipe_deployment(
        package, changed,
        runtime=arti.TaskRuntimeProfile("changed_task:run", "changed-caller-profile-v1"),
        code={"application": caller},
    )
    original = arti.inspect_recipe_deployment(package)
    assert packaged["model_manifest_sha256"] == original["model_manifest_sha256"]
    assert packaged["deployment_sha256"] != original["deployment_sha256"]
    path = project(tmp_path, {"model": package, "deployment": changed,
                              "prediction_run": directory, "evaluation": evidence,
                              "result_export": archive})
    monkeypatch.setattr(arti.PythonFactoryConfig, "resolve", forbidden)
    before = inventory(tmp_path)
    report = explain_task_project(path)
    assert not report["ok"] and not report["summary"]["lifecycle_complete"]
    assert all(entry["status"] != "invalid" for entry in report["artifacts"].values())
    links = {link["name"]: link["status"] for link in report["links"]}
    assert links["prediction_model"] == "matched"
    assert links["prediction_deployment"] == links["evaluation_deployment"] == links["export_deployment"] == "mismatch"
    assert inventory(tmp_path) == before


def test_model_only_historical_predictions_do_not_gain_a_deployment_binding(tmp_path, monkeypatch):
    from test_custom_task_workflow import setup
    from arti.recipe_workflow import write_jsonl
    cfg, rows = setup(tmp_path / "data", steps=1)
    trained = arti.TaskRecipe.from_config(cfg).fit(
        run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "training", checkpoint_steps=1)
    )
    requests = tmp_path / "requests.jsonl"
    write_jsonl(requests, rows[-3:])
    directory = tmp_path / "prediction"
    arti.run_prediction(trained["model"], requests, arti.PredictionRunConfig(directory, batch_size=2),
                        task_factory=cfg.factory)
    package = tmp_path / "package"
    arti.package_recipe_deployment(trained["model"], package, resources={})
    path = project(tmp_path, {"model": Path(trained["model"]), "deployment": package,
                              "prediction_run": directory})
    monkeypatch.setattr(arti.PythonFactoryConfig, "resolve", forbidden)
    report = task_project_status(path)
    assert report["ok"]
    links = {link["name"]: link for link in report["links"]}
    assert links["prediction_deployment"]["status"] == "not_checked"
    assert links["prediction_deployment"]["left_identity"] is None
    assert "deployment_sha256" not in report["artifacts"]["prediction_run"]["details"]


def test_partial_package_prediction_explain_selects_pinned_deployment_and_executable_factory(tmp_path, monkeypatch):
    package, _requests, _files, directory, _targets, _rows = fixture(tmp_path, interrupt=True)
    raw_model = tmp_path / "training/model/model.st"
    path = project(tmp_path, {"model": raw_model, "deployment": package,
                              "prediction_run": directory})
    with monkeypatch.context() as patch:
        patch.setattr(arti.PythonFactoryConfig, "resolve", forbidden)
        before = inventory(tmp_path)
        report = explain_task_project(path)
        assert inventory(tmp_path) == before
    action = next(item for item in report["next_actions"] if item["action"] == "resume_prediction")
    assert action["source_artifact"] == "deployment"
    assert str(package) in action["argv"] and str(raw_model) not in action["argv"]
    assert action["argv"][-2:] == ["--task-factory", "examples.asset_task_recipe:build_task"]
    assert "installed/importable" in action["reason"]
    from arti.cli import main
    before = inventory(directory / "parts")
    assert main(action["argv"][1:]) == 0
    assert arti.inspect_prediction(directory)["remaining_records"] == 0
    after = inventory(directory / "parts")
    assert all(after[name] == value for name, value in before.items())


def test_partial_prediction_with_different_valid_deployment_has_no_retry_command(tmp_path, monkeypatch):
    package, _requests, _files, directory, _targets, _rows = fixture(tmp_path, interrupt=True)
    different = tmp_path / "different-deployment"
    arti.package_recipe_deployment(package, different, requirements=["torch>=2.2"])
    path = project(tmp_path, {"model": tmp_path / "training/model/model.st",
                              "deployment": different, "prediction_run": directory})
    monkeypatch.setattr(arti.PythonFactoryConfig, "resolve", forbidden)
    before = inventory(tmp_path)
    report = explain_task_project(path)
    assert not report["ok"]
    action = next(item for item in report["next_actions"] if item["action"] == "resume_prediction")
    assert action["command"] is None and action["argv"] is None
    assert action["source_artifact"] == "deployment" and "pinned deployment identity" in action["reason"]
    assert inventory(tmp_path) == before


def test_generated_project_explain_uses_application_driver_for_local_code_path(tmp_path, monkeypatch):
    from arti.task_scaffold import create_task_project
    from test_task_scaffold import _driver
    root = tmp_path / "generated"
    create_task_project(root)
    with monkeypatch.context() as patch:
        patch.setattr(arti.PythonFactoryConfig, "resolve", forbidden)
        before = inventory(root)
        fresh = explain_task_project(root / "project.json")
        assert inventory(root) == before
    prepare = next(item for item in fresh["next_actions"] if item["action"] == "prepare_inputs")
    assert prepare["argv"] == ["python", str(root / "workflow.py"), "prepare"]
    _driver(root, *prepare["argv"][2:])
    with monkeypatch.context() as patch:
        patch.setattr(arti.PythonFactoryConfig, "resolve", forbidden)
        prepared = explain_task_project(root / "project.json")
    train = next(item for item in prepared["next_actions"] if item["action"] == "start_training")
    assert train["argv"] == ["python", str(root / "workflow.py"), "train"]
    _driver(root, *train["argv"][2:])
    _driver(root, "package")
    _driver(root, "predict", "--stop-after", "1")
    with monkeypatch.context() as patch:
        patch.setattr(arti.PythonFactoryConfig, "resolve", forbidden)
        before = inventory(root)
        report = explain_task_project(root / "project.json")
        assert inventory(root) == before
    action = next(item for item in report["next_actions"] if item["action"] == "resume_prediction")
    assert action["argv"] == ["python", str(root / "workflow.py"), "predict", "--resume"]
    assert "caller code path" in action["reason"]
    completed = _driver(root, *action["argv"][2:])
    assert completed["newly_predicted_records"] == 2 and completed["reused_records"] == 2
