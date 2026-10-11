from __future__ import annotations

from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import zipfile

import pytest
import torch

import arti
from arti.recipe_workflow import read_recipe_config
from arti.task_project import read_task_project
from arti.task_scaffold import create_task_project


@pytest.fixture(autouse=True)
def local_process_temporary_directory(tmp_path, monkeypatch):
    directory = tmp_path / "process-temp"
    directory.mkdir()
    for name in ("TEMP", "TMP", "TMPDIR"):
        monkeypatch.setenv(name, str(directory))
    monkeypatch.setattr(tempfile, "tempdir", str(directory))


def _files(directory):
    return {
        path.relative_to(directory).as_posix(): (path.stat().st_mtime_ns, path.read_bytes())
        for path in directory.rglob("*")
        if path.is_file()
    }


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _equal(a, b)
    else:
        assert left == right


def _driver(root, *args):
    framework = Path(arti.__file__).resolve().parent.parent
    result = subprocess.run(
        [sys.executable, str(root / "workflow.py"), *args],
        cwd=root,
        env={
            **os.environ,
            "PYTHONPATH": str(framework),
            "PYTHONDONTWRITEBYTECODE": "1",
            "CUDA_VISIBLE_DEVICES": "",
        },
        capture_output=True,
        text=True,
        timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return json.loads(result.stdout)


def test_scaffold_is_fresh_atomic_and_application_owned(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("scaffold must not import caller code")

    monkeypatch.setattr(arti.PythonFactoryConfig, "resolve", forbidden)
    root = tmp_path / "task"
    receipt = create_task_project(root, name="neutral application")
    assert (
        not receipt["application_executed"]
        and not receipt["trained"]
        and not receipt["installs_dependencies"]
    )
    assert set(receipt["files"]) == {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    source = (root / "code/task_app.py").read_text(encoding="utf-8")
    assert "from examples" not in source and "import examples" not in source
    snapshot = arti.DatasetSnapshot(root / "data/snapshot")
    assert snapshot.manifest()["inventory"]["unique_contents"] == 6
    assert snapshot.manifest()["inventory"]["references"] == 12
    assert read_task_project(root / "project.json").name == "neutral application"
    tutorial = (root / "TUTORIAL.md").read_bytes()
    from importlib.resources import files

    assert tutorial == files("arti").joinpath("templates", "task", "TUTORIAL.md").read_bytes()
    source_guide = Path(__file__).resolve().parents[1] / "docs/guides/installed-task-workflow.md"
    if source_guide.exists():
        assert tutorial == source_guide.read_bytes()
    before = _files(root)
    with pytest.raises(FileExistsError):
        create_task_project(root)
    assert _files(root) == before
    for name, template in (("", "files"), ("task", "missing")):
        with pytest.raises(ValueError):
            create_task_project(tmp_path / "invalid", name=name, template=template)
    assert not (tmp_path / "invalid").exists()


def test_failed_generation_does_not_publish_partial_project(tmp_path, monkeypatch):
    import arti.task_scaffold as module

    def fail(*args, **kwargs):
        raise RuntimeError("snapshot generation stopped")

    monkeypatch.setattr(module, "create_dataset_snapshot", fail)
    root = tmp_path / "task"
    with pytest.raises(RuntimeError, match="snapshot generation stopped"):
        create_task_project(root)
    assert not root.exists() and not list(tmp_path.glob(".task-*"))


def test_generated_native_objective_arti_parameters_and_exact_recovery(tmp_path, monkeypatch):
    root = tmp_path / "task"
    create_task_project(root)
    monkeypatch.syspath_prepend(str(root / "code"))
    monkeypatch.delitem(sys.modules, "task_app", raising=False)
    app = importlib.import_module("task_app")
    config = read_task_project(root / "project.json").resolved_recipe()
    full, partial = arti.TaskRecipe.from_config(config), arti.TaskRecipe.from_config(config)
    initial = {
        name: parameter.detach().clone() for name, parameter in full.model.named_parameters()
    }
    buffers = {
        name: value.clone()
        for name, value in full.model.named_buffers()
        if name.startswith("feature_")
    }
    cache = _files(root / "data/cache")
    batch = full.train_data[0]
    with full.session.evaluation_context(), torch.no_grad():
        full.model.eval()
        assert torch.equal(
            app.objective(full.model, batch),
            torch.nn.functional.mse_loss(full.model(batch["features"]), batch["target"]),
        )
    rng = torch.get_rng_state().clone()
    full_report = full.fit(run=replace(config.run, output_dir=root / "runs/full"))
    torch.set_rng_state(rng)

    def stop(session, _checkpoint):
        if session.global_step == 2:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        partial.fit(run=replace(config.run, output_dir=root / "runs/partial"), on_checkpoint=stop)
    moved = tmp_path / "moved"
    shutil.move(root, moved)
    # Only the accepted immutable snapshot/cache is needed after relocation.
    original_source = moved / "data/source"
    assert original_source.resolve().is_relative_to(moved.resolve())
    shutil.rmtree(original_source)
    monkeypatch.syspath_prepend(str(moved / "code"))
    monkeypatch.delitem(sys.modules, "task_app", raising=False)
    recovered = arti.TaskRecipe.from_config(
        read_task_project(moved / "project.json").resolved_recipe()
    )
    report = recovered.resume_run(moved / "runs/partial", output_dir=moved / "runs/recovered")
    assert report["optimizer_updates_this_attempt"] == 2
    assert full_report["selection"] == report["selection"]
    assert (
        full.task_fingerprint == recovered.task_fingerprint
        and recovered.task_contract["objective"] == "native-mse"
    )
    _equal(full.model.state_dict(), recovered.model.state_dict())
    _equal(full.session.optimizer.state_dict(), recovered.session.optimizer.state_dict())
    _equal(full.session.loss_history, recovered.session.loss_history)
    _equal(
        buffers,
        {
            name: value
            for name, value in recovered.model.named_buffers()
            if name.startswith("feature_")
        },
    )
    assert _files(moved / "data/cache") == cache
    changed = [
        name
        for name, parameter in recovered.model.named_parameters()
        if not torch.equal(initial[name], parameter)
    ]
    assert any(name.startswith("program.") for name in changed)
    assert any(name.startswith("output.") for name in changed)


def test_generated_driver_file_recovery_cold_handoff_and_portable_export(tmp_path):
    root = tmp_path / "task"
    create_task_project(root)
    prepare = _driver(root, "prepare")
    cache = _files(root / "data/cache")
    reuse = _driver(root, "prepare")
    assert prepare["produced"] == 6 and reuse["produced"] == 0 and reuse["reused"] == 6
    assert _files(root / "data/cache") == cache
    assert _driver(root, "train", "--stop-after", "2")["status"] == "interrupted"
    recovered = _driver(root, "recover")
    assert recovered["optimizer_updates_this_attempt"] == 2
    assert read_task_project(root / "project.json").artifacts["training_run"] == "runs/recovered"
    _driver(root, "package")
    assert _driver(root, "predict", "--stop-after", "1")["status"] == "interrupted"
    committed = _files(root / "artifacts/prediction/parts")
    predicted = _driver(root, "predict", "--resume")
    assert predicted["reused_records"] == predicted["newly_predicted_records"] == 2
    after = _files(root / "artifacts/prediction/parts")
    assert all(after[name] == value for name, value in committed.items())
    completed = _driver(root, "predict", "--resume")
    assert (
        completed["newly_predicted_records"] == 0
        and _files(root / "artifacts/prediction/parts") == after
    )
    suite = _driver(root, "evaluate")
    assert suite["complete_evidence"]
    assert all(
        view["coverage"]["total"] == view["coverage"]["scored"] == 4
        for view in suite["views"].values()
    )
    cold = _driver(root, "runtime")
    assert (
        cold["status"] == "completed"
        and not cold["network_enforced"]
        and not cold["installs_dependencies"]
    )
    cold_before = _files(root / "artifacts/cold-prediction/parts")
    cold_reuse = _driver(root, "runtime", "--resume")
    assert cold_reuse["worker"]["result"]["prediction"]["newly_predicted_records"] == 0
    assert _files(root / "artifacts/cold-prediction/parts") == cold_before
    exported = _driver(root, "export")
    assert not exported["regenerates_predictions"] and not exported["submits_externally"]
    archive = tmp_path / "results.zip"
    shutil.copyfile(root / "artifacts/results.zip", archive)
    assert root.resolve().is_relative_to(tmp_path.resolve())
    shutil.rmtree(root)
    extracted = tmp_path / "consumer"
    with zipfile.ZipFile(archive) as stream:
        stream.extractall(extracted)
    manifest = read_recipe_config(extracted / ".arti-export/manifest.json")
    assert manifest["expected_ids"] == [f"record-{index}" for index in range(8, 12)]
    assert arti.inspect_result_export(archive)["sha256"] == exported["sha256"]
    attachments = manifest["attachments"]
    # Existing completed journal remains readable with original task sources gone.
    journal = extracted / attachments["journal"]["path"]
    assert arti.inspect_prediction(journal)["remaining_records"] == 0
    runtime = arti.run_task_runtime(
        extracted / ".arti-export/deployment",
        inputs={
            "requests": extracted / attachments["inputs"]["path"],
            "references": extracted / attachments["references"]["path"],
            "suite": extracted / attachments["suite"]["path"],
        },
        outputs={
            "results": tmp_path / "fresh-prediction",
            "evaluation": tmp_path / "fresh-evaluation.json",
        },
        output=tmp_path / "fresh-runtime.json",
    )
    assert runtime["status"] == "completed"
    assert runtime["runtime_identity"]["framework_root"] == str(
        Path(arti.__file__).resolve().parent.parent
    )
    assert runtime["worker"]["result"]["prediction"]["newly_predicted_records"] == 4
    original_rows = (journal / "predictions.jsonl").read_text(encoding="utf-8")
    fresh_rows = (tmp_path / "fresh-prediction/predictions.jsonl").read_text(encoding="utf-8")
    assert [json.loads(line)["prediction"] for line in fresh_rows.splitlines()] == [
        json.loads(line)["prediction"] for line in original_rows.splitlines()
    ]
    cold_journal = extracted / attachments["cold_journal"]["path"]
    cold_parts = _files(cold_journal / "parts")
    reused = arti.run_task_runtime(
        extracted / ".arti-export/deployment",
        inputs={
            "requests": extracted / attachments["inputs"]["path"],
            "references": extracted / attachments["references"]["path"],
            "suite": extracted / attachments["suite"]["path"],
        },
        outputs={"results": cold_journal, "evaluation": tmp_path / "reused-evaluation.json"},
        options={"resume": True},
        output=tmp_path / "reused-runtime.json",
    )
    assert (
        reused["status"] == "completed"
        and reused["worker"]["result"]["prediction"]["newly_predicted_records"] == 0
    )
    assert _files(cold_journal / "parts") == cold_parts
