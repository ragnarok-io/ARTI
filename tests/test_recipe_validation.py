from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

import arti
from arti.cli import main
from arti.recipe_workflow import write_json, write_jsonl
from test_custom_task_workflow import forbid
from test_task_prediction_run import trained
from test_task_deployment import RESOURCE_FACTORY, trained as resource_trained


def scoring(kind="regression"):
    return arti.PredictionScoringConfig(f"examples.prediction_scorers:{kind}", f"{kind}-v1")


def setup(tmp_path, kind="custom"):
    _current, asset, rows = trained(tmp_path, kind)
    bundle = arti.package_recipe_deployment(asset, tmp_path / "bundle")
    data = tmp_path / "requests.jsonl"
    write_jsonl(data, rows)
    run = arti.PredictionRunConfig(tmp_path / "job", batch_size=2)
    return bundle, data, run


@pytest.mark.parametrize("kind", ["custom", "classification"])
def test_actual_validation_without_implicit_quality_standard_and_no_host_recovery(
    tmp_path, monkeypatch, kind
):
    bundle, data, run = setup(tmp_path, kind)
    current = arti.validate_recipe_deployment(
        bundle["directory"],
        data,
        run,
        scoring("classification" if kind == "classification" else "regression"),
        task_factory="examples.custom_task_recipe:build_task" if kind == "custom" else None,
    )
    assert current["status"] == "evaluated" and current["criteria_passed"] is None
    assert (
        current["inference_completed"]
        and current["scoring_completed"]
        and current["runtime_reload_verified"]
    )
    assert current["runtime_evidence"]["origin"] == "current_attempt"
    assert current["source"]["deployment_sha256"] == bundle["deployment_sha256"]
    rows = list((run.output_dir / "parts").glob("*/predictions.jsonl"))
    original = [path.read_bytes() for path in rows]
    shutil.rmtree(bundle["directory"])
    data.unlink()
    import arti.recipes as recipes

    monkeypatch.setattr(recipes, "load_recipe_predictor", forbid)
    repeated = arti.validate_recipe_deployment(
        None,
        None,
        run,
        scoring("classification" if kind == "classification" else "regression"),
        resume=True,
    )
    assert repeated["prediction"]["newly_predicted_records"] == 0
    assert (
        repeated["runtime_reload_verified"]
        and repeated["runtime_evidence"]["origin"] == "prior_validation"
    )
    assert repeated["metrics"] == current["metrics"]
    original_receipt = (
        Path(current["output"]).parent
        / f"{Path(current['output']).stem}.reports"
        / f"{current['sha256']}.json"
    )
    assert json.loads(original_receipt.read_text()) == current
    assert [path.read_bytes() for path in rows] == original
    assert arti.torch.validate_recipe_deployment is arti.validate_recipe_deployment


def test_changed_criteria_reuses_same_predictions_and_reports_unevaluable_metrics(
    tmp_path, monkeypatch
):
    bundle, data, run = setup(tmp_path)
    first = arti.validate_recipe_deployment(
        bundle["directory"],
        data,
        run,
        scoring(),
        task_factory="examples.custom_task_recipe:build_task",
    )
    import arti.recipes as recipes

    monkeypatch.setattr(recipes, "load_recipe_predictor", forbid)
    actual = first["metrics"]["mse"]
    accepted = arti.validate_recipe_deployment(
        None,
        None,
        run,
        scoring(),
        resume=True,
        criteria={"mse": {"minimum": actual, "maximum": actual}},
    )
    assert accepted["criteria_passed"] and accepted["status"] == "passed"
    rejected = arti.validate_recipe_deployment(
        None, None, run, scoring(), resume=True, criteria={"mse": {"maximum": actual / 2}}
    )
    assert not rejected["ok"] and rejected["status"] == "rejected"
    missing = arti.validate_recipe_deployment(
        None, None, run, scoring(), resume=True, criteria={"unknown": {"minimum": 0}}
    )
    assert missing["checks"][0]["status"] == "not_evaluable" and missing["criteria_passed"] is False
    assert accepted["sha256"] != rejected["sha256"]
    assert accepted["scores"] == rejected["scores"]
    assert (
        accepted["prediction"]["newly_predicted_records"]
        == rejected["prediction"]["newly_predicted_records"]
        == 0
    )


@pytest.mark.parametrize(
    "criteria",
    [
        {"mse": {}},
        {"mse": {"maximum": True}},
        {"mse": {"maximum": float("inf")}},
        {"mse": {"minimum": 2, "maximum": 1}},
        {"mse": {"mode": "min"}},
    ],
)
def test_invalid_criteria_fail_before_model_or_creating_job(tmp_path, monkeypatch, criteria):
    import arti.recipe_validation as module

    monkeypatch.setattr(module, "inspect_recipe_deployment", forbid)
    run = arti.PredictionRunConfig(tmp_path / "job")
    with pytest.raises(ValueError, match="criterion"):
        arti.validate_recipe_deployment(
            tmp_path / "missing.st", None, run, scoring(), criteria=criteria
        )
    assert not run.output_dir.exists()


def test_budget_stop_reports_no_quality_result_and_then_resumes_remaining_only(tmp_path):
    bundle, data, run = setup(tmp_path)
    counter = []
    stopped = arti.validate_recipe_deployment(
        bundle["directory"],
        data,
        run,
        scoring(),
        task_factory="examples.custom_task_recipe:build_task",
        stop_requested=lambda: bool(counter),
        on_batch=lambda *_args: counter.append(1),
    )
    assert stopped["status"] == "stopped" and not stopped["inference_completed"]
    assert stopped["criteria_passed"] is None and not stopped["scoring_completed"]
    assert stopped["prediction"]["committed_records"] == 2
    finished = arti.validate_recipe_deployment(
        bundle["directory"],
        None,
        run,
        scoring(),
        task_factory="examples.custom_task_recipe:build_task",
        resume=True,
    )
    assert finished["prediction"]["newly_predicted_records"] == 5
    assert finished["prediction"]["reused_records"] == 2


def test_scorer_failure_preserves_predictions_and_earlier_accepted_evidence(tmp_path):
    bundle, data, run = setup(tmp_path)
    first = arti.validate_recipe_deployment(
        bundle["directory"],
        data,
        run,
        scoring(),
        task_factory="examples.custom_task_recipe:build_task",
    )
    accepted_report = Path(first["output"]).read_bytes()
    score_bytes = Path(first["scores"]["file"]).read_bytes()
    broken = arti.PredictionScoringConfig("test_prediction_scoring:invalid_scorer", "broken-rule")
    with pytest.raises(TypeError):
        arti.validate_recipe_deployment(None, None, run, broken, resume=True)
    failure = json.loads(Path(first["output"]).with_suffix(".failure.json").read_text())
    assert failure["stage"] == "scoring" and failure["status"] == "failed"
    assert Path(first["output"]).read_bytes() == accepted_report
    assert Path(first["scores"]["file"]).read_bytes() == score_bytes
    assert arti.inspect_prediction(run.output_dir)["committed_records"] == 7


def test_primary_publication_failure_preserves_previous_report_and_records_stage(
    tmp_path, monkeypatch
):
    bundle, data, run = setup(tmp_path)
    first = arti.validate_recipe_deployment(
        bundle["directory"],
        data,
        run,
        scoring(),
        task_factory="examples.custom_task_recipe:build_task",
    )
    output = Path(first["output"])
    before = output.read_bytes()
    import arti.recipe_validation as module

    original = module._atomic_json

    def write(path, payload):
        if path == output:
            raise OSError("publication failed")
        return original(path, payload)

    monkeypatch.setattr(module, "_atomic_json", write)
    with pytest.raises(OSError):
        arti.validate_recipe_deployment(None, None, run, scoring(), resume=True)
    failure = json.loads(output.with_suffix(".failure.json").read_text())
    assert failure["stage"] == "publication" and failure["scoring_completed"]
    assert output.read_bytes() == before


def test_initial_inspection_failure_has_durable_diagnostic_without_poisoning_job(tmp_path):
    run = arti.PredictionRunConfig(tmp_path / "job")
    with pytest.raises(FileNotFoundError):
        arti.validate_recipe_deployment(tmp_path / "absent" / "model.st", None, run, scoring())
    failure = json.loads((tmp_path / "job.validation.failure.json").read_text())
    assert failure["stage"] == "inspection" and not run.output_dir.exists()


def test_completed_external_job_does_not_claim_a_new_reload_or_previous_validation(tmp_path):
    bundle, data, run = setup(tmp_path)
    arti.run_prediction(
        bundle["directory"], data, run, task_factory="examples.custom_task_recipe:build_task"
    )
    observed = arti.validate_recipe_deployment(None, None, run, scoring(), resume=True)
    assert observed["status"] == "evaluated" and not observed["runtime_reload_verified"]
    assert observed["runtime_evidence"]["origin"] == "not_run"
    assert observed["prediction"]["newly_predicted_records"] == 0


def test_resource_task_package_cli_and_metric_criteria_exit_codes(tmp_path, capsys):
    _current, result, rows, resources = resource_trained(tmp_path / "source", resources=True)
    bundle = arti.package_recipe_deployment(
        result["model"], tmp_path / "bundle", resources=resources
    )
    data = tmp_path / "requests.jsonl"
    write_jsonl(data, rows)
    job = tmp_path / "job"
    common = [
        "--run-dir",
        str(job),
        "--scorer",
        "examples.prediction_scorers:regression",
        "--scorer-id",
        "mse-v1",
    ]
    assert (
        main(
            [
                "recipe",
                "validate-deployment",
                bundle["directory"],
                "--data",
                str(data),
                "--task-factory",
                RESOURCE_FACTORY,
                *common,
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["status"] == "evaluated"
    criteria = tmp_path / "criteria.json"
    write_json(criteria, {"mse": {"maximum": -1}})
    assert (
        main(
            ["recipe", "validate-deployment", "--resume", *common, "--criteria-json", str(criteria)]
        )
        == 1
    )
    rejected = json.loads(capsys.readouterr().out)
    assert (
        rejected["status"] == "rejected"
        and rejected["source"]["deployment_sha256"] == bundle["deployment_sha256"]
    )


def test_legacy_language_prediction_profile_uses_task_owned_scoring(tmp_path):
    from test_recipe_deployment import trained as language_trained
    from test_prediction_run import requests

    _current, asset, base, _result = language_trained(tmp_path / "source")
    data = tmp_path / "requests.jsonl"
    rows = requests(3)
    write_jsonl(data, rows)
    targets = tmp_path / "targets.jsonl"
    write_jsonl(targets, [{"id": row["id"], "target": "red"} for row in rows])
    run = arti.CausalLMPredictionRunConfig(tmp_path / "job", batch_size=2, max_new_tokens=4)
    scores = arti.PredictionScoringConfig("test_prediction_scoring:reply_scorer", "whole-reply-v1")
    result = arti.validate_recipe_deployment(
        asset, data, run, scores, targets=targets, base_model=base
    )
    assert result["samples"] == 3 and "generation_score" in result["metrics"]
    assert result["status"] == "evaluated" and result["runtime_reload_verified"]


def test_standalone_asset_file_allows_sibling_validation_without_overwriting_dependencies(tmp_path):
    current, _asset, rows = trained(tmp_path / "source")
    saved = arti.save(
        current.binding.state_module(),
        tmp_path / "standalone.st",
        config={"task_recipe": current.task_contract},
    ).weights_path
    data = tmp_path / "requests.jsonl"
    write_jsonl(data, rows)
    run = arti.PredictionRunConfig(tmp_path / "job")
    report = arti.validate_recipe_deployment(
        saved, data, run, scoring(), task_factory="examples.custom_task_recipe:build_task"
    )
    assert report["status"] == "evaluated" and report["samples"] == 7
    original = (tmp_path / "standalone.json").read_bytes()
    with pytest.raises(ValueError, match="overwrite model dependencies"):
        arti.validate_recipe_deployment(
            saved, None, run, scoring(), resume=True, output=tmp_path / "standalone.json"
        )
    assert (tmp_path / "standalone.json").read_bytes() == original
