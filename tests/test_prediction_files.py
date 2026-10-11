from __future__ import annotations

from dataclasses import replace
import json
import shutil

import pytest

import arti
from arti.cli import main
from arti.recipe_data import read_jsonl
from arti.recipe_workflow import write_json, write_jsonl
from examples.asset_task_recipe import FACTORY
from test_recipe_assets import fixture, forbidden


VALIDATOR = "examples.asset_task_recipe:validate_files"


def setup(tmp_path, *, field="files", formats=None):
    cfg, rows = fixture(tmp_path)
    snapshot = arti.create_dataset_snapshot(cfg, tmp_path / "snapshot")
    resources = {"snapshot": str(snapshot.directory), "cache": str(tmp_path / "cache")}
    task = arti.TaskRecipe.from_config(
        arti.TaskRecipeConfig(
            "array-regression",
            FACTORY,
            options={
                "input_shape": [4],
                "batch_size": 4,
                "file_field": field,
                **({} if formats is None else {"file_formats": formats}),
            },
            resources=resources,
            training=arti.ARTITrainingConfig(device="cpu", steps=2, learning_rate=0.02),
        )
    )
    result = task.fit(
        run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "training", checkpoint_steps=1)
    )
    package = tmp_path / "package"
    arti.package_recipe_deployment(result["model"], package, resources=resources)
    requests = tmp_path / "requests.jsonl"
    write_jsonl(requests, [{"id": row["id"]} for row in rows[-7:]])
    files = arti.PredictionFilesConfig(
        field, arti.PythonFactoryConfig(VALIDATOR, {"field": field}), "file-contract-v1"
    )
    return package, requests, files, rows[-7:]


def run(package, requests, files, directory, **kwargs):
    return arti.run_prediction(
        package,
        requests,
        arti.PredictionRunConfig(directory, batch_size=3, files=files),
        task_factory=FACTORY,
        **kwargs,
    )


@pytest.mark.parametrize("field", ["files", "artifacts"])
def test_named_native_files_atomic_parts_source_deletion_relocation_and_zero_replay(
    tmp_path, monkeypatch, field
):
    package, requests, files, _rows = setup(tmp_path, field=field)
    directory = tmp_path / "job"
    result = run(package, requests, files, directory)
    assert result["status"] == "completed" and result["committed_batches"] == 3
    assert result["output_files"] == 21 and result["output_bytes"] > 0
    assert result["file_preparation_seconds"] >= 0
    assert not list(directory.glob(".producer-*")) and not list(directory.glob(".batch-*"))
    rows = list(read_jsonl(directory / "predictions.jsonl"))
    assert len(rows) == 7
    for row in rows:
        assert set(row[field]) == {"array", "binary", "image"}
        for file in row[field].values():
            assert (directory / file["file"]).is_file()
            assert file["file"].startswith("parts/")
    receipt = json.loads((directory / "parts/00000002/receipt.json").read_text())
    assert receipt["records"] == 1 and receipt["file_validation"]["ok"]
    identity = arti.inspect_prediction(directory)["result_sha256"]
    before = {
        path.relative_to(directory): path.read_bytes()
        for path in directory.glob("parts/*/files/**/*")
        if path.is_file()
    }
    shutil.rmtree(package)
    shutil.rmtree(tmp_path / "snapshot")
    shutil.rmtree(tmp_path / "cache")
    requests.unlink()
    moved = tmp_path / "moved"
    shutil.move(directory, moved)
    monkeypatch.setattr(arti.PythonFactoryConfig, "resolve", forbidden)
    repeated = arti.run_prediction(None, None, arti.PredictionRunConfig(moved), resume=True)
    assert repeated["newly_predicted_records"] == 0 and repeated["output_files"] == 21
    assert arti.inspect_prediction(moved)["result_sha256"] == identity
    for path, value in before.items():
        assert (moved / path).read_bytes() == value
    (moved / "predictions.jsonl").unlink()
    finalized = arti.run_prediction(None, None, arti.PredictionRunConfig(moved), resume=True)
    assert finalized["newly_predicted_records"] == 0
    assert list(read_jsonl(moved / "predictions.jsonl")) == rows


def test_interrupted_file_job_only_produces_missing_rows_and_keeps_closure(tmp_path, monkeypatch):
    package, requests, files, _rows = setup(tmp_path)
    directory = tmp_path / "job"

    def interrupt(report, receipt):
        assert report["output_files"] == 9
        raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run(package, requests, files, directory, on_batch=interrupt)
    original = {
        path: (path.stat().st_mtime_ns, path.read_bytes())
        for path in (directory / "parts/00000000").rglob("*")
        if path.is_file()
    }
    predictor = arti.TaskPredictor.from_pretrained(package, task_factory=FACTORY)
    callback, seen = predictor._predict_fn, []

    def tracked(model, records, **kwargs):
        seen.extend(row["id"] for row in records)
        return callback(model, records, **kwargs)

    monkeypatch.setattr(predictor, "_predict_fn", tracked)
    result = run(package, None, files, directory, resume=True, predictor=predictor)
    assert result["reused_records"] == 3 and result["newly_predicted_records"] == 4
    assert seen == [row["id"] for row in list(read_jsonl(directory / "input.jsonl"))[3:]]
    for path, (timestamp, value) in original.items():
        assert path.stat().st_mtime_ns == timestamp and path.read_bytes() == value


@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_file_closure_inspection_detects_missing_or_changed_output(tmp_path, damage):
    package, requests, files, _rows = setup(tmp_path)
    directory = tmp_path / "job"
    run(package, requests, files, directory)
    row = next(read_jsonl(directory / "predictions.jsonl"))
    path = directory / row["files"]["binary"]["file"]
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(b"changed")
    with pytest.raises(ValueError, match="asset content differs"):
        arti.inspect_prediction(directory)
    with pytest.raises(ValueError, match="asset content differs"):
        arti.run_prediction(None, None, arti.PredictionRunConfig(directory), resume=True)


@pytest.mark.parametrize("failure", ["producer", "missing-file", "copy", "publish", "validator"])
def test_failure_never_publishes_partial_json_or_files_and_preserves_old_parts(
    tmp_path, monkeypatch, failure
):
    package, requests, files, _rows = setup(tmp_path)
    directory = tmp_path / "job"
    with pytest.raises(KeyboardInterrupt):
        run(
            package,
            requests,
            files,
            directory,
            on_batch=lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
    original = (directory / "parts/00000000/receipt.json").read_bytes()
    predictor = arti.TaskPredictor.from_pretrained(package, task_factory=FACTORY)
    if failure == "producer":

        def failed(model, records, *, output_dir, **kwargs):
            (output_dir / "partial.bin").write_bytes(b"partial")
            raise RuntimeError("producer failed")

        monkeypatch.setattr(predictor, "_predict_fn", failed)
    elif failure == "missing-file":
        callback = predictor._predict_fn

        def failed(model, records, **kwargs):
            outputs = callback(model, records, **kwargs)
            outputs[0]["files"]["binary"]["file"] = "absent.bin"
            return outputs

        monkeypatch.setattr(predictor, "_predict_fn", failed)
    elif failure == "copy":
        import arti.recipe_files as module

        monkeypatch.setattr(
            module.shutil, "copyfile", lambda *args: (_ for _ in ()).throw(OSError("copy failed"))
        )
    elif failure == "publish":
        import arti.recipe_prediction as module

        original_replace = module._replace_atomic_file

        def failed(source, target):
            if target.parent.name == "parts":
                raise OSError("publish failed")
            return original_replace(source, target)

        monkeypatch.setattr(module, "_replace_atomic_file", failed)
    else:
        import examples.asset_task_recipe as application

        monkeypatch.setattr(application, "validate_files", lambda *args, **kwargs: {"ok": False})
    with pytest.raises((OSError, RuntimeError, ValueError)):
        run(package, None, files, directory, resume=True, predictor=predictor)
    assert [part.name for part in (directory / "parts").iterdir()] == ["00000000"]
    assert (directory / "parts/00000000/receipt.json").read_bytes() == original
    assert not list(directory.glob(".producer-*")) and not list(directory.glob(".batch-*"))


def test_validator_identity_and_file_contract_pinned_across_recovery(tmp_path):
    package, requests, files, _rows = setup(tmp_path)
    directory = tmp_path / "job"
    run(package, requests, files, directory)
    with pytest.raises(ValueError, match="file producer/validator differs"):
        run(None, None, replace(files, revision="different"), directory, resume=True)
    with pytest.raises(ValueError, match="supplied by the prediction run"):
        arti.PredictionRunConfig(
            tmp_path / "other", files=files, predict_kwargs={"output_dir": "external"}
        )
    assert arti.torch.PredictionFilesConfig is arti.PredictionFilesConfig


def test_empty_file_results_keep_structured_outputs_and_a_real_validator_directory(tmp_path):
    package, requests, files, _rows = setup(tmp_path, formats=[])
    result = run(package, requests, files, tmp_path / "job")
    assert result["output_files"] == result["output_bytes"] == 0
    rows = list(read_jsonl(tmp_path / "job/predictions.jsonl"))
    assert len(rows) == 7 and all(row["files"] == {} for row in rows)
    assert (tmp_path / "job/parts/00000000/files").is_dir()


def test_named_aliases_count_physical_files_and_validator_cannot_rewrite_native_evidence(
    tmp_path, monkeypatch
):
    package, requests, files, _rows = setup(tmp_path)
    predictor = arti.TaskPredictor.from_pretrained(package, task_factory=FACTORY)
    callback = predictor._predict_fn

    def aliases(model, records, **kwargs):
        rows = callback(model, records, **kwargs)
        for row in rows:
            row["files"]["alias"] = dict(row["files"]["binary"])
        return rows

    monkeypatch.setattr(predictor, "_predict_fn", aliases)
    import examples.asset_task_recipe as application

    validate = application.validate_files

    def mutate(inputs, predictions, **kwargs):
        receipt = validate(inputs, predictions, **kwargs)
        assert kwargs["directory"].is_dir()
        inputs[0]["id"] = "changed"
        predictions[0]["prediction"] = "changed"
        return receipt

    monkeypatch.setattr(application, "validate_files", mutate)
    result = run(package, requests, files, tmp_path / "job", predictor=predictor)
    assert result["output_files"] == 21
    rows = list(read_jsonl(tmp_path / "job/predictions.jsonl"))
    assert [row["id"] for row in rows] == [row["id"] for row in read_jsonl(requests)]
    assert isinstance(rows[0]["prediction"], float)
    assert rows[0]["files"]["alias"] == rows[0]["files"]["binary"]


def test_file_outputs_connect_existing_validation_and_cli_without_model_replay(
    tmp_path, monkeypatch, capsys
):
    package, requests, files, rows = setup(tmp_path)
    targets = tmp_path / "targets.jsonl"
    write_jsonl(targets, rows)
    job, report = tmp_path / "job", tmp_path / "validation.json"
    args = [
        "recipe",
        "validate-deployment",
        str(package),
        "--data",
        str(requests),
        "--run-dir",
        str(job),
        "--task-factory",
        FACTORY,
        "--batch-size",
        "3",
        "--output-files-field",
        "files",
        "--file-validator",
        VALIDATOR,
        "--file-validator-id",
        "file-contract-v1",
        "--scorer",
        "examples.prediction_scorers:regression",
        "--scorer-id",
        "original-unit-v1",
        "--targets",
        str(targets),
        "--output",
        str(report),
    ]
    # The explicit factory kwargs match this example's file-field declaration.
    validator_kwargs = tmp_path / "validator.json"
    write_json(validator_kwargs, {"field": "files"})
    args.extend(["--file-validator-kwargs-json", str(validator_kwargs)])
    assert main(args) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["status"] == "evaluated" and first["criteria_passed"] is None
    import arti.recipes as recipes

    monkeypatch.setattr(arti.PythonFactoryConfig, "resolve", forbidden)
    monkeypatch.setattr(recipes, "load_recipe_predictor", forbidden)
    assert main(["recipe", "predict-run", "--run-dir", str(job), "--resume"]) == 0
    assert json.loads(capsys.readouterr().out)["newly_predicted_records"] == 0
