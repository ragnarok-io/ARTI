from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import zipfile

import pytest

import arti
from arti.cli import main
from arti.recipe_workflow import write_json
from arti.recipe_data import json_fingerprint
from test_recipe_suite import fixture, suite
from test_recipe_assets import forbidden


WRITER = "examples.result_writers:write_results"
VALIDATOR = "examples.result_writers:validate_results"


def config(*, archive="directory"):
    return arti.ResultExportConfig(
        WRITER,
        "layout-v1",
        validator=VALIDATOR,
        validation_id="layout-validator-v1",
        archive=archive,
        instructions="Use the declared application entrypoint in your selected environment.",
        unresolved_checks=("external environment execution remains unverified",),
    )


@pytest.mark.parametrize("archive", ["directory", "zip"])
def test_full_native_closure_layout_move_and_source_deletion_without_inference(
    tmp_path, monkeypatch, archive
):
    package, _requests, _files, directory, targets, rows = fixture(tmp_path)
    evidence = tmp_path / "evaluation.json"
    evaluated = arti.evaluate_prediction_suite(
        directory, suite(), references=targets, output=evidence
    )
    manifest = package / "deployment.json"
    selection = json.loads(manifest.read_text())["model_manifest_sha256"]
    note = tmp_path / "application-note.txt"
    note.write_text("caller-owned handoff note", encoding="utf-8")
    import arti.recipes as recipes

    monkeypatch.setattr(recipes, "load_recipe_predictor", forbidden)
    destination = tmp_path / ("result.zip" if archive == "zip" else "result")
    result = arti.export_prediction_results(
        directory,
        replace(config(archive=archive), examples=(rows[0]["id"],)),
        destination,
        expected_ids=[row["id"] for row in rows],
        evidence={"evaluation": evidence},
        attachments={"note": note},
        deployment=package,
    )
    assert result["records"] == 7 and not result["regenerates_predictions"]
    assert not result["submits_externally"]
    inspected = arti.inspect_result_export(destination)
    assert inspected["records"] == 7 and inspected["source"]["model_manifest_sha256"] == selection
    extracted = destination
    if archive == "zip":
        extracted = tmp_path / "extracted"
        with zipfile.ZipFile(destination) as package_zip:
            package_zip.extractall(extracted)
    export_manifest = json.loads((extracted / ".arti-export/manifest.json").read_text())
    assert export_manifest["evidence"]["evaluation"]["sha256"] == evaluated["sha256"]
    assert (
        export_manifest["deployment"]["deployment_sha256"]
        == json.loads((extracted / ".arti-export/deployment/deployment.json").read_text())["sha256"]
    )
    assert len(list((extracted / "records").glob("*.json"))) == 7
    assert len(list((extracted / ".arti-export/native/parts").glob("*/files/*/*"))) == 21
    example_file = export_manifest["examples"][0]["prediction"]["files"]["binary"]["file"]
    assert (extracted / example_file).is_file()
    shutil.rmtree(directory)
    shutil.rmtree(package)
    evidence.unlink()
    targets.unlink()
    note.unlink()
    moved = tmp_path / "moved"
    shutil.move(extracted, moved)
    monkeypatch.setattr(arti.PythonFactoryConfig, "resolve", forbidden)
    assert arti.inspect_result_export(moved)["sha256"] == inspected["sha256"]


def test_format_changes_reuse_same_native_results_and_shared_table_aliases(tmp_path, monkeypatch):
    _package, _requests, _files, directory, _targets, _rows = fixture(tmp_path)
    one, two = tmp_path / "individual", tmp_path / "table"
    first = arti.export_prediction_results(directory, config(), one)
    table = replace(
        config(),
        writer=arti.PythonFactoryConfig("examples.result_writers:write_table"),
        revision="table-v1",
    )
    second = arti.export_prediction_results(directory, table, two)
    assert first["sha256"] != second["sha256"]
    a = json.loads((one / ".arti-export/manifest.json").read_text())
    b = json.loads((two / ".arti-export/manifest.json").read_text())
    assert a["source"] == b["source"]
    assert (two / "results.csv").is_file()
    assert len(list(two.glob("*.csv"))) == 1
    assert a["unresolved_checks"] and b["unresolved_checks"]


@pytest.mark.parametrize(
    "failure",
    ["missing-id", "duplicate-id", "missing-file", "writer", "validator", "copy", "publish"],
)
def test_export_failure_never_publishes_partial_result_or_changes_source(
    tmp_path, monkeypatch, failure
):
    _package, _requests, _files, directory, _targets, _rows = fixture(tmp_path)
    before = arti.inspect_prediction(directory)["result_sha256"]
    import examples.result_writers as application

    original = application.write_results
    import arti.recipe_export as module

    if failure in {"missing-id", "duplicate-id", "missing-file"}:

        def bad(context, items):
            rows = original(context, items)
            if failure == "missing-id":
                rows.pop()
            elif failure == "duplicate-id":
                rows[-1]["id"] = rows[0]["id"]
            else:
                rows[-1]["files"]["record"] = "absent.json"
            return rows

        monkeypatch.setattr(application, "write_results", bad)
    elif failure == "writer":
        monkeypatch.setattr(
            application,
            "write_results",
            lambda *args: (_ for _ in ()).throw(RuntimeError("writer failed")),
        )
    elif failure == "validator":
        monkeypatch.setattr(application, "validate_results", lambda *args, **kwargs: {"ok": False})
    elif failure == "copy":
        monkeypatch.setattr(
            module, "_copy_file", lambda *args: (_ for _ in ()).throw(OSError("copy failed"))
        )
    else:
        monkeypatch.setattr(
            module,
            "_replace_atomic_file",
            lambda *args: (_ for _ in ()).throw(OSError("publish failed")),
        )
    destination = tmp_path / "result"
    with pytest.raises((ValueError, RuntimeError, OSError)):
        arti.export_prediction_results(directory, config(), destination)
    assert not destination.exists() and not list(tmp_path.glob(".result-*"))
    assert arti.inspect_prediction(directory)["result_sha256"] == before


def test_altered_native_files_or_mismatched_evidence_are_not_accepted_as_complete(tmp_path):
    _package, _requests, _files, directory, targets, _rows = fixture(tmp_path)
    evidence = tmp_path / "evaluation.json"
    value = arti.evaluate_prediction_suite(directory, suite(), references=targets, output=evidence)
    value["source"]["predictions_sha256"] = "wrong"
    from arti.recipe_data import json_fingerprint

    value["sha256"] = json_fingerprint(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    write_json(evidence, value)
    with pytest.raises(ValueError, match="different result content"):
        arti.export_prediction_results(
            directory, config(), tmp_path / "wrong", evidence={"evaluation": evidence}
        )
    native = next(directory.glob("parts/*/files/*/value.bin"))
    native.unlink()
    with pytest.raises(ValueError, match="asset content differs"):
        arti.export_prediction_results(directory, config(), tmp_path / "missing")


@pytest.mark.parametrize("archive", ["directory", "zip"])
def test_export_inspection_verifies_every_file_after_publication(tmp_path, archive):
    _package, _requests, _files, directory, _targets, _rows = fixture(tmp_path)
    destination = tmp_path / ("result.zip" if archive == "zip" else "result")
    arti.export_prediction_results(directory, config(archive=archive), destination)
    if archive == "directory":
        (destination / "records/0.json").write_text("changed", encoding="utf-8")
    else:
        with zipfile.ZipFile(destination, "a") as zipped:
            zipped.writestr("extra.txt", "undeclared")
    with pytest.raises(ValueError, match="file content differs|file inventory differs"):
        arti.inspect_result_export(destination)


def test_expected_population_is_not_silently_shrunk_and_cli_export_keeps_writer_separate(
    tmp_path, monkeypatch, capsys
):
    _package, _requests, _files, directory, _targets, rows = fixture(tmp_path)
    with pytest.raises(ValueError, match="complete original"):
        arti.export_prediction_results(
            directory, config(), tmp_path / "partial", expected_ids=[row["id"] for row in rows[:-1]]
        )
    config_file = tmp_path / "export.json"
    write_json(config_file, config().to_dict())
    assert arti.ResultExportConfig.read(config_file) == config()
    output = tmp_path / "result"
    assert (
        main(
            [
                "recipe",
                "export-results",
                str(directory),
                "--config",
                str(config_file),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["records"] == 7
    monkeypatch.setattr(arti.PythonFactoryConfig, "resolve", forbidden)
    assert main(["recipe", "inspect-results", str(output)]) == 0
    assert not json.loads(capsys.readouterr().out)["imports_writer"]
    assert arti.torch.ResultExportConfig is arti.ResultExportConfig


@pytest.mark.parametrize("identity", ["execution_sha256", "deployment_sha256", "total_records"])
def test_export_binds_execution_package_and_original_evaluation_population(tmp_path, identity):
    _package, _requests, _files, directory, targets, _rows = fixture(tmp_path)
    evidence = tmp_path / "evaluation.json"
    value = arti.evaluate_prediction_suite(directory, suite(), references=targets, output=evidence)
    value["source"][identity] = 8 if identity == "total_records" else "0" * 64
    value["sha256"] = json_fingerprint(
        {key: item for key, item in value.items() if key != "sha256"}
    )
    write_json(evidence, value)
    destination = tmp_path / "result"
    with pytest.raises(ValueError, match="different result content"):
        arti.export_prediction_results(
            directory, config(), destination, evidence={"evaluation": evidence}
        )
    assert not destination.exists()


def test_evidence_selected_before_writer_keeps_its_exact_original_bytes(tmp_path, monkeypatch):
    _package, _requests, _files, directory, targets, _rows = fixture(tmp_path)
    evidence = tmp_path / "evaluation.json"
    arti.evaluate_prediction_suite(directory, suite(), references=targets, output=evidence)
    import examples.result_writers as application

    original = application.write_results

    def writer(context, items):
        rows = original(context, items)
        # The semantic report is unchanged, but these are no longer the selected bytes.
        evidence.write_bytes(evidence.read_bytes() + b"\n")
        return rows

    monkeypatch.setattr(application, "write_results", writer)
    destination = tmp_path / "result"
    with pytest.raises(ValueError, match="asset content differs"):
        arti.export_prediction_results(
            directory, config(), destination, evidence={"evaluation": evidence}
        )
    assert not destination.exists()


def test_validator_does_not_implicitly_select_an_extra_formatted_output(tmp_path, monkeypatch):
    _package, _requests, _files, directory, _targets, _rows = fixture(tmp_path)
    import examples.result_writers as application

    original = application.validate_results

    def validator(rows, *, directory):
        (directory / "unselected.bin").write_bytes(b"application scratch")
        return original(rows, directory=directory)

    monkeypatch.setattr(application, "validate_results", validator)
    destination = tmp_path / "result"
    with pytest.raises(ValueError, match="declared file closure differs"):
        arti.export_prediction_results(directory, config(), destination)
    assert not destination.exists()


def test_inspection_checks_declared_output_closure_and_empty_attachment_directories(tmp_path):
    _package, _requests, _files, directory, _targets, _rows = fixture(tmp_path)
    attachment = tmp_path / "application"
    (attachment / "empty").mkdir(parents=True)
    destination = tmp_path / "result"
    arti.export_prediction_results(
        directory, config(), destination, attachments={"application": attachment}
    )
    empty = destination / ".arti-export/attachments/application/empty"
    empty.rmdir()
    with pytest.raises(ValueError, match="directory inventory differs"):
        arti.inspect_result_export(destination)
    empty.mkdir()
    formatted = destination / ".arti-export/formatted-results.jsonl"
    rows = [json.loads(line) for line in formatted.read_text().splitlines()]
    rows[0]["files"]["record"]["file"] = "absent.json"
    formatted.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest_path = destination / ".arti-export/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    import hashlib

    payload = formatted.read_bytes()
    manifest["files"][".arti-export/formatted-results.jsonl"] = {
        "sha256": hashlib.sha256(payload).hexdigest(),
        "bytes": len(payload),
    }
    manifest["sha256"] = json_fingerprint(
        {key: item for key, item in manifest.items() if key != "sha256"}
    )
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="declared file closure differs"):
        arti.inspect_result_export(destination)


def test_cold_runtime_evaluation_export_and_recovered_handoff_reuse_committed_results(tmp_path):
    from test_recipe_runtime import fixture as runtime_fixture, profile
    from arti.recipe_workflow import write_jsonl

    runtime = profile(
        "validate",
        inputs={"requests": "file", "suite": "file", "references": "file"},
        outputs={"results": "directory", "evaluation": "file"},
        options={"task_factory": "examples.asset_task_recipe:build_task", "batch_size": 3},
    )
    source, portable, requests, rows, _runtime, packaged = runtime_fixture(
        tmp_path, runtime=runtime
    )
    references, suite_file = tmp_path / "references.jsonl", tmp_path / "suite.json"
    write_jsonl(references, [{**row, "available": True} for row in rows])
    write_json(suite_file, suite(aggregate=True).to_dict())
    job, evaluation = tmp_path / "job", tmp_path / "evaluation.json"
    runtime_evidence = tmp_path / "runtime.json"
    initial = arti.run_task_runtime(
        portable,
        inputs={"requests": requests, "suite": suite_file, "references": references},
        outputs={"results": job, "evaluation": evaluation},
        output=runtime_evidence,
    )
    assert initial["status"] == "completed" and initial["observed"]["pid"] != os.getpid()
    validation = json.loads(evaluation.read_text())
    assert validation["inference_completed"] and validation["scoring_completed"]
    score = Path(validation["scores"]["file"])
    raw = arti.inspect_prediction(job)["result_sha256"]
    destination = tmp_path / "export.zip"
    arti.export_prediction_results(
        job,
        config(archive="zip"),
        destination,
        deployment=portable,
        evidence={"runtime": runtime_evidence, "validation": evaluation, "suite": score},
        attachments={
            "journal": job,
            "requests": requests,
            "references": references,
            "suite": suite_file,
        },
    )
    handoff = tmp_path / "handoff"
    with zipfile.ZipFile(destination) as archive:
        archive.extractall(handoff)
    exported = json.loads((handoff / ".arti-export/manifest.json").read_text())
    assert exported["deployment"]["deployment_sha256"] == packaged["deployment_sha256"]
    assert exported["evidence"]["runtime"]["binding_scope"] == "deployment-only"
    assert exported["evidence"]["suite"]["binding_scope"] == "result-content"
    for directory in (source, portable, job, score.parent):
        shutil.rmtree(directory)
    for file in (requests, references, suite_file, evaluation, runtime_evidence, destination):
        file.unlink()
    attachment_root = handoff / ".arti-export/attachments"
    recovered = tmp_path / "consumer-job"
    shutil.copytree(attachment_root / "journal", recovered)
    before = {
        path.relative_to(recovered): path.read_bytes()
        for path in (recovered / "parts").rglob("*")
        if path.is_file()
    }
    recovered_evaluation = tmp_path / "consumer-evaluation.json"
    final = arti.run_task_runtime(
        handoff / ".arti-export/deployment",
        inputs={
            "requests": attachment_root / "requests" / requests.name,
            "suite": attachment_root / "suite" / suite_file.name,
            "references": attachment_root / "references" / references.name,
        },
        outputs={"results": recovered, "evaluation": recovered_evaluation},
        options={"resume": True},
    )
    assert final["status"] == "completed" and final["observed"]["pid"] != os.getpid()
    assert final["worker"]["result"]["prediction"]["newly_predicted_records"] == 0
    assert arti.inspect_prediction(recovered)["result_sha256"] == raw
    assert before == {
        path.relative_to(recovered): path.read_bytes()
        for path in (recovered / "parts").rglob("*")
        if path.is_file()
    }
    assert json.loads(recovered_evaluation.read_text())["metrics"] == validation["metrics"]
    assert arti.inspect_result_export(handoff)["sha256"] == exported["sha256"]


def test_json_only_native_results_and_original_score_receipts_export_without_model(
    tmp_path, monkeypatch
):
    from test_prediction_scoring import config as scoring, job
    from arti.recipe_data import read_jsonl

    _current, asset, rows, journal = job(tmp_path)
    evidence = tmp_path / "scores.json"
    scores = arti.score_prediction_run(journal, scoring(), output=evidence)
    original = list(read_jsonl(journal / "predictions.jsonl"))
    assert all("files" not in row for row in original)
    assert "execution_sha256" not in scores["source"]
    Path(asset).unlink()
    (journal / "predictions.jsonl").unlink()
    import arti.recipes as recipes

    monkeypatch.setattr(recipes, "load_recipe_predictor", forbidden)
    destination = tmp_path / "json-only"
    result = arti.export_prediction_results(
        journal, config(), destination, evidence={"original-scores": evidence}
    )
    assert result["records"] == len(rows) and not result["regenerates_predictions"]
    assert list(read_jsonl(destination / ".arti-export/native-results.jsonl")) == original
    assert not (destination / ".arti-export/native").exists()
    assert arti.inspect_result_export(destination)["records"] == len(rows)


def test_legacy_causal_json_results_keep_original_receipt_and_selected_bundled_base(
    tmp_path, monkeypatch
):
    from test_recipe_deployment import trained
    from test_prediction_run import requests, policy
    from arti.recipe_data import read_jsonl
    from arti.recipe_workflow import write_jsonl

    _current, asset, base, _result = trained(tmp_path / "training")
    portable = tmp_path / "portable"
    arti.package_causal_lm_deployment(asset, portable, include_base=True, base_model=base)
    rows = requests(3)
    data = tmp_path / "requests.jsonl"
    write_jsonl(data, rows)
    journal = tmp_path / "language"
    arti.run_causal_lm_prediction(portable, data, policy(journal))
    expected = list(read_jsonl(journal / "predictions.jsonl"))
    targets = tmp_path / "targets.jsonl"
    write_jsonl(targets, [{"id": row["id"], "target": "red"} for row in rows])
    evidence = tmp_path / "scores.json"
    scored = arti.score_prediction_run(
        journal,
        arti.PredictionScoringConfig("test_prediction_scoring:reply_scorer", "whole-reply-v1"),
        targets=targets,
        output=evidence,
    )
    assert "execution_sha256" not in scored["source"]
    import arti.recipes as recipes

    monkeypatch.setattr(recipes, "load_recipe_predictor", forbidden)
    destination = tmp_path / "legacy-export"
    result = arti.export_prediction_results(
        journal,
        replace(config(), options={"value_field": "text"}),
        destination,
        deployment=portable,
        evidence={"original-scores": evidence},
    )
    assert not result["regenerates_predictions"]
    assert list(read_jsonl(destination / ".arti-export/native-results.jsonl")) == expected
    shutil.rmtree(tmp_path / "training")
    shutil.rmtree(portable)
    shutil.rmtree(journal)
    assert (
        arti.inspect_recipe_deployment(destination / ".arti-export/deployment")["base_source"]
        == "bundled"
    )
    assert arti.inspect_result_export(destination)["records"] == len(rows)


@pytest.mark.parametrize("archive", ["directory", "zip"])
def test_export_preserves_exact_noncanonical_input_snapshot_bytes(tmp_path, monkeypatch, archive):
    from test_task_prediction_run import trained, loaded

    _current, asset, rows = trained(tmp_path)
    payload = (
        b"\r\n"
        + b"\r\n\r\n".join(json.dumps(row, separators=(",", ":")).encode("utf-8") for row in rows)
        + b"\r\n"
    )
    data = tmp_path / "requests.jsonl"
    data.write_bytes(payload)
    journal = tmp_path / "job"
    arti.run_prediction(
        asset, data, arti.PredictionRunConfig(journal, batch_size=3), predictor=loaded(asset)
    )
    plan = json.loads((journal / "plan.json").read_text())
    assert plan["input"]["file"] == "input.jsonl"
    assert (journal / "input.jsonl").read_bytes() == payload
    assert plan["input"]["sha256"] == hashlib.sha256(payload).hexdigest()
    data.unlink()
    Path(asset).unlink()
    import arti.recipes as recipes

    monkeypatch.setattr(recipes, "load_recipe_predictor", forbidden)
    destination = tmp_path / ("result.zip" if archive == "zip" else "result")
    result = arti.export_prediction_results(journal, config(archive=archive), destination)
    shutil.rmtree(journal)
    moved = tmp_path / ("moved.zip" if archive == "zip" else "moved")
    shutil.move(destination, moved)
    assert arti.inspect_result_export(moved)["sha256"] == result["sha256"]
    root = moved
    if archive == "zip":
        root = tmp_path / "extracted"
        with zipfile.ZipFile(moved) as package:
            assert package.read(".arti-export/inputs.jsonl") == payload
            package.extractall(root)
    assert (root / ".arti-export/inputs.jsonl").read_bytes() == payload
    assert arti.inspect_result_export(root)["sha256"] == result["sha256"]
    # Reformatting the JSON changes the pinned bytes even when every record is identical.
    snapshot = root / ".arti-export/inputs.jsonl"
    canonical = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
    snapshot.write_bytes(canonical)
    manifest_path = root / ".arti-export/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][manifest["inputs"]] = {
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "bytes": len(canonical),
    }
    manifest["sha256"] = json_fingerprint(
        {key: item for key, item in manifest.items() if key != "sha256"}
    )
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="pinned input snapshot differs"):
        arti.inspect_result_export(root)


def test_export_inspection_pairs_result_ids_with_the_original_input_population(tmp_path):
    _package, _requests, _files, directory, _targets, _rows = fixture(tmp_path)
    destination = tmp_path / "result"
    arti.export_prediction_results(directory, config(), destination)
    manifest_path = destination / ".arti-export/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    original = manifest["expected_ids"][0]
    manifest["expected_ids"][0] = "changed-logical-id"
    for name in ("formatted_results", "native_results"):
        path = destination / manifest[name]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            if row["id"] == original:
                row["id"] = "changed-logical-id"
        payload = "".join(json.dumps(row) + "\n" for row in rows).encode("utf-8")
        path.write_bytes(payload)
        manifest["files"][manifest[name]] = {
            "sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload),
        }
    manifest["sha256"] = json_fingerprint(
        {key: item for key, item in manifest.items() if key != "sha256"}
    )
    write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="original input ID population differs"):
        arti.inspect_result_export(destination)
