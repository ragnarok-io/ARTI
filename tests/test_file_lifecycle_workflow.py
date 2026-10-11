from __future__ import annotations

import json
from pathlib import Path
import shutil
import zipfile

import pytest

import arti
from examples.file_lifecycle_workflow import _cold_runtime, _files, run_workflow


@pytest.fixture(scope="module")
def lifecycle(tmp_path_factory):
    root = tmp_path_factory.mktemp("file-lifecycle") / "workflow"
    return root, run_workflow(root)


def test_native_file_lifecycle_recovery_cache_original_population_and_export(lifecycle):
    root, report = lifecycle
    assert report["ok"] and report["native_state_equal"] and report["cache_files_unchanged"]
    assert report["objective"] == "scaled-mse" and report["optimizer_updates_recovered"] == 2
    assert report["selection"]["selection"] == "best" and report["selection"]["reason"] == "validation_metric"
    assert report["preparation"]["produced"] == 8 and report["cache_reuse"]["reused"] == 8
    assert report["snapshot"]["inventory"]["unique_contents"] == 8
    assert report["snapshot"]["inventory"]["references"] == 15
    assert report["snapshot"]["inventory"]["unavailable_references"] == 1
    assert report["prediction"] == {
        "committed_records": 4, "reused_records": 3, "newly_predicted_records": 1, "output_files": 8,
    }
    assert report["generated_on_recovery"] == ["sample-13"]
    assert report["execution_phases"] == ["initial", "initial", "initial", "recovered"]
    for view in report["suite"]["views"].values():
        assert view["coverage"]["total"] == view["coverage"]["scored"] == 4
    assert report["cold_runtime"]["status"] == "completed"
    assert report["cold_runtime"]["newly_predicted_records"] == 0
    assert not report["cold_runtime"]["network_enforced"] and report["cold_runtime"]["enforced"] == {}
    assert not report["cold_runtime"]["installs_dependencies"]
    assert report["originals_deleted"] and not (root / "original").exists()
    assert Path(report["framework_file"]).resolve() == Path(arti.__file__).resolve()
    assert not report["export"]["regenerates_predictions"] and not report["export"]["submits_externally"]
    assert report["inspection"]["unresolved_checks"]


def test_archived_selected_code_and_journal_execute_in_fresh_offline_consumer_after_deletion(
    lifecycle, tmp_path, monkeypatch,
):
    root, report = lifecycle
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(report["archive"]) as archive:
        archive.extractall(extracted)
    manifest = json.loads((extracted / ".arti-export/manifest.json").read_text())
    assert manifest["expected_ids"] == [f"sample-{index}" for index in range(10, 14)]
    deployment = arti.inspect_recipe_deployment(extracted / ".arti-export/deployment")
    assert deployment["selection"]["selected_step"] == report["selection"]["selected_step"]
    assert len(list((extracted / "records").glob("*.json"))) == 4
    bindings = manifest["attachments"]
    inputs = extracted / bindings["inputs"]["path"]
    journal = tmp_path / "prediction"
    shutil.copytree(extracted / bindings["journal"]["path"], journal)
    before = _files(journal / "parts")
    result = _cold_runtime(
        extracted / ".arti-export/deployment", inputs, journal,
        tmp_path / "validation.json", tmp_path / "runtime.json", tmp_path / "consumer",
    )
    assert result["status"] == "completed" and _files(journal / "parts") == before
    assert not (root / "local").exists() and not (root / "original").exists()
    assert not list((extracted / ".arti-export/deployment").rglob("__pycache__"))

    def forbidden(*args, **kwargs):
        raise AssertionError("archive inspection must not import writer or model code")

    monkeypatch.setattr(arti.PythonFactoryConfig, "resolve", forbidden)
    assert arti.inspect_result_export(report["archive"])["sha256"] == manifest["sha256"]
    assert arti.inspect_result_export(extracted)["sha256"] == manifest["sha256"]
