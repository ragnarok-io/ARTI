from __future__ import annotations

from dataclasses import replace
import json
import os
import shutil
import struct
import subprocess
import sys

import pytest

import arti
from arti.cli import main
from arti.recipe_data import read_jsonl
from arti.recipe_workflow import write_json, write_jsonl
from test_prediction_files import setup, run
from test_recipe_assets import forbidden


def suite(*, trials=False, external=False, aggregate=False):
    views = [
        arti.EvaluationViewConfig(
            "labels",
            arti.PredictionScoringConfig(
                "examples.prediction_scorers:regression", "original-units-v1"
            ),
            reference_field="target",
            availability_field="available",
        ),
        arti.EvaluationViewConfig(
            "format",
            arti.PredictionScoringConfig(
                "examples.evaluation_suite_scorers:file_validity", "array-format-v1"
            ),
            requires="output",
            files=True,
        ),
    ]
    if external:
        views.append(
            arti.EvaluationViewConfig(
                "outcome", requires="external", external_protocol="application-protocol-v1"
            )
        )
    return arti.EvaluationSuiteConfig(
        tuple(views),
        aggregator=arti.PythonFactoryConfig(
            "examples.evaluation_suite_scorers:aggregate_views",
            {"coefficients": [["labels", "mse", -1], ["format", "valid_fraction", 2]]},
        )
        if aggregate
        else None,
        aggregation_id="application-composite-v1" if aggregate else None,
        trials=arti.EvaluationTrialFields("initial", "seed", "trial")
        if trials
        else arti.EvaluationTrialFields(),
    )


def fixture(tmp_path, *, trials=False, interrupt=False):
    package, requests, files, rows = setup(tmp_path)
    if trials:
        values = list(read_jsonl(requests))
        for index, row in enumerate(values):
            row.update(initial={"revision": f"state-{index // 2}"}, seed=index % 2, trial=index)
        write_jsonl(requests, values)
    directory = tmp_path / "prediction"
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            run(
                package,
                requests,
                files,
                directory,
                on_batch=lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()),
            )
    else:
        run(package, requests, files, directory)
    targets = tmp_path / "targets.jsonl"
    write_jsonl(targets, [{**row, "available": True} for row in rows])
    return package, requests, files, directory, targets, rows


def external_evidence(directory, config, ids, *, value=1, allow_partial=False):
    return {
        "source": arti.evaluation_source_identity(directory, allow_partial=allow_partial),
        "protocol": config.views[-1].external_protocol,
        "producer": {"name": "application-evaluator", "revision": "local-fixture-v1"},
        "metrics": {"outcome_rate": value},
        "scores": [
            {"id": identifier, "outcome": value, "unit": "task-defined"} for identifier in ids
        ],
    }


def test_missing_references_are_not_input_labels_or_zero_scores_and_original_coverage_is_visible(
    tmp_path, monkeypatch
):
    package, _requests, _files, directory, targets, rows = fixture(tmp_path)
    references = [{**row, "available": index != 2} for index, row in enumerate(rows[:-1])]
    references[1].pop("target")
    write_jsonl(targets, references)
    import arti.recipes as recipes

    monkeypatch.setattr(recipes, "load_recipe_predictor", forbidden)
    shutil.rmtree(package)
    result = arti.evaluate_prediction_suite(directory, suite(), references=targets)
    assert result["status"] == "evaluated_with_gaps" and result["criteria_passed"] is None
    view = result["views"]["labels"]
    assert view["coverage"]["scored"] == 4 and view["coverage"]["total"] == 7
    assert view["coverage"]["total_fraction"] == 4 / 7
    assert [row["reason"] for row in view["outcomes"] if row["status"] == "unavailable"] == [
        "missing_reference_field",
        "reference_unavailable",
        "missing_reference",
    ]
    assert all(row["score"] is None for row in view["outcomes"] if row["status"] == "unavailable")
    assert result["views"]["format"]["coverage"]["scored"] == 7
    absent = arti.evaluate_prediction_suite(directory, suite())
    assert absent["views"]["labels"]["status"] == "unavailable"
    assert absent["views"]["labels"]["metrics"] == {} and absent["views"]["labels"]["scores"] == []
    assert absent["views"]["format"]["metrics"]["valid_fraction"] == 1


def test_task_aggregation_explicit_criteria_and_measurement_failure_archive(tmp_path):
    _package, _requests, _files, directory, targets, _rows = fixture(tmp_path)
    config = suite(aggregate=True)
    output = tmp_path / "evaluation.json"
    first = arti.evaluate_prediction_suite(directory, config, references=targets, output=output)
    assert first["complete_evidence"] and first["status"] == "evaluated"
    assert first["metrics"]["composite"] == -first["views"]["labels"]["metrics"]["mse"] + 2
    passed = arti.evaluate_prediction_suite(
        directory,
        replace(config, criteria={"minimum_coverage": {"minimum": 1}}),
        references=targets,
    )
    assert passed["status"] == "passed" and passed["criteria_passed"] is True
    rejected = arti.evaluate_prediction_suite(
        directory, replace(config, criteria={"absent": {"maximum": 0}}), references=targets
    )
    assert rejected["status"] == "rejected" and rejected["checks"][0]["status"] == "not_evaluable"
    bad = replace(
        config,
        views=(
            replace(
                config.views[0],
                scoring=arti.PredictionScoringConfig(
                    "examples.evaluation_suite_scorers:fail", "error-v1"
                ),
            ),
            config.views[1],
        ),
    )
    failed = arti.evaluate_prediction_suite(directory, bad, references=targets, output=output)
    assert failed["status"] == "failed" and not failed["ok"]
    assert len(failed["views"]["labels"]["coverage"]["failed_ids"]) == 7
    assert failed["views"]["format"]["status"] == "scored"
    archived = json.loads((tmp_path / "evaluation.reports" / f"{first['sha256']}.json").read_text())
    assert archived == first


def test_binary_reference_snapshot_availability_move_and_content_identity(tmp_path):
    _package, _requests, _files, directory, _targets, rows = fixture(tmp_path)
    records = tmp_path / "reference-records.jsonl"
    assets, bindings = {}, {}
    for index, row in enumerate(rows):
        name = f"reference-{index}"
        path = tmp_path / f"{name}.bin"
        path.write_bytes(struct.pack("<d", row["target"]))
        assets[name] = path if index != 2 else None
        bindings[row["id"]] = [arti.AssetReference(name, "reference", required=False)]
    write_jsonl(records, rows)
    snapshot = arti.create_dataset_snapshot(
        arti.DatasetSnapshotConfig(records, assets, bindings), tmp_path / "references"
    )
    config = arti.EvaluationSuiteConfig(
        (
            arti.EvaluationViewConfig(
                "binary",
                arti.PredictionScoringConfig(
                    "examples.evaluation_suite_scorers:binary_error", "binary-original-units-v1"
                ),
                files=True,
                reference_role="reference",
            ),
        )
    )
    first = arti.evaluate_prediction_suite(directory, config, references=snapshot)
    assert first["views"]["binary"]["coverage"]["scored"] == 6
    assert first["views"]["binary"]["outcomes"][2]["reason"] == "reference_assets_unavailable"
    moved = tmp_path / "moved"
    shutil.move(snapshot.directory, moved)
    for source in assets.values():
        if source is not None:
            source.unlink()
    second = arti.evaluate_prediction_suite(directory, config, references=moved)
    assert second["references"]["sha256"] == first["references"]["sha256"]
    assert second["views"]["binary"]["metrics"] == first["views"]["binary"]["metrics"]
    reopened = arti.DatasetSnapshot(moved)
    reopened.asset("reference-0").write_bytes(b"changed")
    with pytest.raises(ValueError, match="asset content differs"):
        arti.evaluate_prediction_suite(directory, config, references=reopened)


def test_recovery_phase_declared_initial_state_seed_and_paired_same_trials(tmp_path):
    package, requests, files, directory, targets, _rows = fixture(
        tmp_path, trials=True, interrupt=True
    )
    config = suite(trials=True)
    with pytest.raises(ValueError, match="unfinished"):
        arti.evaluate_prediction_suite(directory, config, references=targets)
    partial = arti.evaluate_prediction_suite(
        directory, config, references=targets, allow_partial=True
    )
    assert partial["views"]["format"]["coverage"]["total_fraction"] == 3 / 7
    assert partial["views"]["format"]["coverage"]["uncommitted"] == 4
    run(package, None, files, directory, resume=True)
    report = arti.evaluate_prediction_suite(directory, config, references=targets)
    phases = [row["execution"]["phase"] for row in report["views"]["format"]["outcomes"]]
    assert phases == ["initial"] * 3 + ["recovered"] * 4
    assert report["views"]["format"]["outcomes"][0]["trial"] == {
        "initial_state": {"revision": "state-0"},
        "seed": 0,
        "trial": 0,
    }
    baseline = tmp_path / "baseline"
    run(package, requests, files, baseline)
    comparison = arti.compare_prediction_suites(directory, baseline, config, references=targets)
    assert comparison["paired_samples"] == 7 and comparison["views"]["labels"]["comparable"]
    assert comparison["views"]["labels"]["delta"]["mse"] == 0
    assert (
        comparison["arms"]["candidate"]["source"]["execution_sha256"]
        != comparison["arms"]["baseline"]["source"]["execution_sha256"]
    )
    assert (
        comparison["arms"]["baseline"]["views"]["format"]["outcomes"][-1]["execution"]["phase"]
        == "initial"
    )
    changed = list(read_jsonl(requests))
    changed[0]["seed"] += 1
    write_jsonl(requests, changed)
    other = tmp_path / "other"
    run(package, requests, files, other)
    with pytest.raises(ValueError, match="trial/input differs"):
        arti.compare_prediction_suites(directory, other, config, references=targets)


def test_external_evidence_binds_source_protocol_and_stays_distinct_from_local_scores(tmp_path):
    _package, _requests, _files, directory, targets, rows = fixture(tmp_path)
    config = suite(external=True)
    missing = arti.evaluate_prediction_suite(directory, config, references=targets)
    assert missing["views"]["outcome"]["status"] == "unavailable"
    evidence = external_evidence(directory, config, [row["id"] for row in rows[:-2]])
    path = tmp_path / "external.json"
    imported = arti.import_evaluation_evidence(directory, config, "outcome", evidence, output=path)
    assert imported["origin"].startswith("external")
    measured = arti.evaluate_prediction_suite(
        directory, config, references=targets, external={"outcome": str(path)}
    )
    view = measured["views"]["outcome"]
    assert view["origin"] == "external" and view["coverage"]["scored"] == 5
    assert view["metrics"]["outcome_rate"] == 1
    assert view["scores"][0]["score"]["unit"] == "task-defined"
    wrong = {**evidence, "source": {**evidence["source"], "input_sha256": "changed"}}
    with pytest.raises(ValueError, match="matching source"):
        arti.import_evaluation_evidence(directory, config, "outcome", wrong)
    with pytest.raises(ValueError, match="matching source"):
        arti.import_evaluation_evidence(
            directory, config, "outcome", {**evidence, "protocol": "other"}
        )
    with pytest.raises(ValueError, match="unknown IDs"):
        arti.import_evaluation_evidence(
            directory, config, "outcome", {**evidence, "scores": [{"id": "unknown", "outcome": 1}]}
        )
    changed_view = replace(config.views[-1], external_protocol="other")
    changed = arti.evaluate_prediction_suite(
        directory,
        replace(config, views=(*config.views[:-1], changed_view)),
        external={"outcome": path},
    )
    assert changed["views"]["outcome"]["status"] == "failed"


def test_external_populations_are_not_comparable_when_coverage_differs(tmp_path):
    package, requests, files, directory, targets, rows = fixture(tmp_path)
    baseline = tmp_path / "baseline"
    run(package, requests, files, baseline)
    config = suite(external=True)
    paths = {}
    for name, root, count in (("candidate", directory, 5), ("baseline", baseline, 4)):
        path = tmp_path / f"{name}.json"
        arti.import_evaluation_evidence(
            root,
            config,
            "outcome",
            external_evidence(root, config, [row["id"] for row in rows[:count]]),
            output=path,
        )
        paths[name] = {"outcome": path}
    result = arti.compare_prediction_suites(
        directory, baseline, config, references=targets, external=paths
    )
    assert result["views"]["labels"]["comparable"]
    assert (
        not result["views"]["outcome"]["comparable"] and result["views"]["outcome"]["delta"] == {}
    )
    assert not result["composite_comparable"] and result["delta"] == {}


def test_mutable_scorer_receives_copies_and_report_cannot_overwrite_inputs(tmp_path):
    _package, _requests, _files, directory, targets, _rows = fixture(tmp_path)
    config = suite()
    config = replace(
        config,
        views=(
            replace(
                config.views[0],
                scoring=arti.PredictionScoringConfig(
                    "examples.evaluation_suite_scorers:mutate", "mutation-v1"
                ),
            ),
        ),
    )
    original = targets.read_bytes()
    result = arti.evaluate_prediction_suite(directory, config, references=targets)
    assert result["views"]["labels"]["scores"][0]["input"]["id"] != "modified"
    assert isinstance(result["views"]["labels"]["scores"][0]["target"]["target"], float)
    assert targets.read_bytes() == original
    with pytest.raises(ValueError, match="overwrite"):
        arti.evaluate_prediction_suite(directory, config, references=targets, output=targets)
    with pytest.raises(ValueError, match="overwrite"):
        arti.evaluate_prediction_suite(
            directory, config, references=targets, output=directory / "input.jsonl"
        )


def test_cli_suite_config_relative_external_manifest_and_no_model_replay(
    tmp_path, monkeypatch, capsys
):
    package, _requests, _files, directory, targets, rows = fixture(tmp_path)
    config = suite(external=True, aggregate=True)
    config_file = tmp_path / "suite.json"
    write_json(config_file, config.to_dict())
    assert arti.EvaluationSuiteConfig.read(config_file) == config
    source = tmp_path / "raw.json"
    write_json(source, external_evidence(directory, config, [row["id"] for row in rows]))
    imported = tmp_path / "external.json"
    assert (
        main(
            [
                "recipe",
                "import-evaluation",
                str(directory),
                "--suite",
                str(config_file),
                "--view",
                "outcome",
                "--evidence",
                str(source),
                "--output",
                str(imported),
            ]
        )
        == 0
    )
    capsys.readouterr()
    mapping = tmp_path / "external-paths.json"
    write_json(mapping, {"outcome": "external.json"})
    import arti.recipes as recipes

    monkeypatch.setattr(recipes, "load_recipe_predictor", forbidden)
    shutil.rmtree(package)
    output = tmp_path / "suite-report.json"
    assert (
        main(
            [
                "recipe",
                "score-suite",
                str(directory),
                "--suite",
                str(config_file),
                "--references",
                str(targets),
                "--external-json",
                str(mapping),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    reported = json.loads(capsys.readouterr().out)
    assert reported["complete_evidence"] and reported["criteria_passed"] is None
    assert "scores" not in reported["views"]["labels"]
    assert len(json.loads(output.read_text())["views"]["labels"]["scores"]) == 7
    assert arti.torch.EvaluationSuiteConfig is arti.EvaluationSuiteConfig
    arm_mapping = tmp_path / "arms.json"
    write_json(
        arm_mapping, {arm: {"outcome": "external.json"} for arm in ("candidate", "baseline")}
    )
    assert (
        main(
            [
                "recipe",
                "compare-suites",
                str(directory),
                "--baseline",
                str(directory),
                "--suite",
                str(config_file),
                "--references",
                str(targets),
                "--external-json",
                str(arm_mapping),
                "--output",
                str(tmp_path / "comparison.json"),
            ]
        )
        == 0
    )
    compared = json.loads(capsys.readouterr().out)
    assert compared["paired_samples"] == 7 and compared["delta"]["composite"] == 0
    code = (
        "import sys,json;from pathlib import Path;import arti;"
        "assert Path(arti.__file__).resolve()==Path(sys.argv[4]).resolve();"
        "r=arti.evaluate_prediction_suite(sys.argv[1],sys.argv[2],references=sys.argv[3]);"
        "print(json.dumps({'metrics':r['views']['labels']['metrics'],'samples':r['samples']}))"
    )
    cold = subprocess.run(
        [sys.executable, "-c", code, str(directory), str(config_file), str(targets), arti.__file__],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        text=True,
        capture_output=True,
        check=True,
    )
    restored = json.loads(cold.stdout)
    assert (
        restored["samples"] == 7 and restored["metrics"] == reported["views"]["labels"]["metrics"]
    )


def test_intersection_keeps_original_denominators_and_does_not_reuse_external_aggregate(tmp_path):
    package, requests, files, directory, targets, rows = fixture(tmp_path)
    baseline = tmp_path / "partial"
    with pytest.raises(KeyboardInterrupt):
        run(
            package,
            requests,
            files,
            baseline,
            on_batch=lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
    config = suite(external=True)
    external = {}
    for arm, root, selected in (("candidate", directory, rows), ("baseline", baseline, rows[:3])):
        path = tmp_path / f"{arm}-external.json"
        arti.import_evaluation_evidence(
            root,
            config,
            "outcome",
            external_evidence(root, config, [row["id"] for row in selected], allow_partial=True),
            output=path,
            allow_partial=True,
        )
        external[arm] = {"outcome": path}
    with pytest.raises(ValueError, match="populations differ"):
        arti.compare_prediction_suites(
            directory, baseline, config, references=targets, allow_partial=True
        )
    report = arti.compare_prediction_suites(
        directory,
        baseline,
        config,
        references=targets,
        external=external,
        pairing="intersection",
        allow_partial=True,
    )
    assert report["paired_samples"] == 3
    current = report["arms"]["candidate"]
    assert len(current["selection"]["excluded_ids"]) == 4
    assert current["views"]["labels"]["coverage"]["total_fraction"] == 3 / 7
    assert current["views"]["outcome"]["metrics"] == {}
    assert not current["views"]["outcome"]["aggregate_available"]
    assert len(current["views"]["outcome"]["unselected_external_ids"]) == 4
    assert not report["views"]["outcome"]["comparable"]


def test_old_receipts_do_not_fabricate_initial_phase_and_extra_references_are_visible(tmp_path):
    _package, _requests, _files, directory, targets, rows = fixture(tmp_path)
    for path in directory.glob("parts/*/receipt.json"):
        value = json.loads(path.read_text())
        value.pop("execution")
        write_json(path, value)
    write_jsonl(
        targets,
        [{**row, "available": True} for row in rows]
        + [{"id": "not-selected", "available": True, "target": 1}],
    )
    report = arti.evaluate_prediction_suite(directory, suite(), references=targets)
    assert all(
        row["execution"]["phase"] == "unrecorded" for row in report["views"]["format"]["outcomes"]
    )
    assert report["views"]["labels"]["reference_population"]["unused_ids"] == ["not-selected"]


def test_invalid_availability_and_declared_trial_fields_are_not_silently_accepted(tmp_path):
    _package, _requests, _files, directory, targets, rows = fixture(tmp_path)
    write_jsonl(targets, [{**row, "available": "yes"} for row in rows])
    result = arti.evaluate_prediction_suite(directory, suite(), references=targets)
    assert result["views"]["labels"]["status"] == "failed"
    assert result["views"]["format"]["status"] == "scored"
    with pytest.raises(ValueError, match="trial initial_state is missing"):
        arti.evaluate_prediction_suite(directory, suite(trials=True))


def test_suite_deployment_validation_reuses_outputs_and_distinguishes_gaps_from_policy(
    tmp_path, monkeypatch
):
    package, requests, files, _directory, targets, _rows = fixture(tmp_path)
    config, job = suite(aggregate=True), tmp_path / "validated"
    run_config = arti.PredictionRunConfig(job, batch_size=3, files=files)
    report = arti.validate_recipe_deployment(
        package,
        requests,
        run_config,
        config,
        targets=targets,
        task_factory="examples.asset_task_recipe:build_task",
    )
    assert report["status"] == "evaluated" and report["complete_evidence"]
    assert report["runtime_evidence"]["origin"] == "current_attempt"
    assert (
        report["criteria_passed"] is None and report["views"]["labels"]["coverage"]["scored"] == 7
    )
    import arti.recipes as recipes

    monkeypatch.setattr(recipes, "load_recipe_predictor", forbidden)
    repeated = arti.validate_recipe_deployment(
        None,
        None,
        arti.PredictionRunConfig(job),
        config,
        targets=targets,
        resume=True,
        criteria={"minimum_coverage": {"minimum": 1}},
    )
    assert repeated["status"] == "passed" and repeated["prediction"]["newly_predicted_records"] == 0
    assert repeated["runtime_evidence"]["origin"] == "prior_validation"
    without = arti.validate_recipe_deployment(
        None, None, arti.PredictionRunConfig(job), config, resume=True
    )
    assert without["status"] == "evaluated_with_gaps" and not without["complete_evidence"]
    assert without["views"]["labels"]["coverage"]["scored"] == 0


def test_cli_validation_suite_shares_native_file_workflow(tmp_path, capsys):
    package, requests, files, _directory, targets, _rows = fixture(tmp_path)
    config = tmp_path / "suite.json"
    write_json(config, suite(aggregate=True).to_dict())
    output = tmp_path / "validation.json"
    assert (
        main(
            [
                "recipe",
                "validate-deployment",
                str(package),
                "--data",
                str(requests),
                "--run-dir",
                str(tmp_path / "job"),
                "--evaluation-suite",
                str(config),
                "--references",
                str(targets),
                "--task-factory",
                "examples.asset_task_recipe:build_task",
                "--output-files-field",
                files.field,
                "--output",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["complete_evidence"] and result["status"] == "evaluated"
    assert result["views"]["format"]["metrics"]["valid_fraction"] == 1
