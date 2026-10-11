from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

import arti
from arti.cli import main
from arti.recipe_workflow import write_json, write_jsonl
from arti.recipes import load_recipe_predictor
from examples.custom_task_recipe import FACTORY, build_task
from test_custom_task_workflow import forbid, setup
from test_recipe_recovery import equal, policy, recipe


RESOURCE_FACTORY = "test_task_deployment:resource_task"


def resource_task(config, *, mode="train", contract=None):
    result = build_task(config, mode=mode, contract=contract)
    original = result._predict_fn
    unit = json.loads(Path(config.resources["unit"]).read_text())["name"]
    family = json.loads((Path(config.resources["processor"]) / "metadata.json").read_text())[
        "family"
    ]

    def predict(model, records, **kwargs):
        return [
            {**row, "unit": unit, "family": family} for row in original(model, records, **kwargs)
        ]

    result._predict_fn = predict
    return result


def trained(tmp_path, *, kind="custom", resources=False):
    if kind == "classification":
        from test_task_recipes import _records

        current = recipe("classification", tmp_path / "data", steps=2)
        rows = _records()[-3:]
    else:
        config, rows = setup(tmp_path / "data", steps=2)
        if resources:
            unit = tmp_path / "inputs" / "unit.json"
            processor = tmp_path / "inputs" / "processor"
            write_json(unit, {"name": "original"})
            write_json(processor / "metadata.json", {"family": "numeric"})
            (processor / "empty").mkdir()
            config = replace(
                config,
                factory=RESOURCE_FACTORY,
                resources={**config.resources, "unit": str(unit), "processor": str(processor)},
            )
        current = arti.TaskRecipe.from_config(config)
        rows = rows[-3:]
    result = current.fit(run=policy(tmp_path / "run"))
    bindings = (
        {key: value for key, value in current.config.resources.items() if key != "data"}
        if resources
        else {}
    )
    return current, result, rows, bindings


@pytest.mark.parametrize("kind", ["custom", "classification"])
def test_package_copies_only_inference_closure_and_preserves_actual_output(
    tmp_path, monkeypatch, kind
):
    current, result, rows, _resources = trained(tmp_path / "source", kind=kind)
    artifact = Path(result["model"])
    source = load_recipe_predictor(artifact, task_factory=FACTORY if kind == "custom" else None)
    expected = source.predict_records(rows)
    (artifact.parent / "training.jsonl").write_text("private training residue")
    (artifact.parent / "unrelated.json").write_text("do not copy")
    report = arti.package_recipe_deployment(
        artifact, tmp_path / "package", requirements=["torch>=2"]
    )
    files = {path.name for path in Path(report["directory"]).iterdir()}
    assert "training.jsonl" not in files and "unrelated.json" not in files
    if kind == "custom":
        assert "model.graph-0.json" in files and "model.graph-0.safetensors" in files
    with monkeypatch.context() as patch:
        patch.setattr(arti.PythonFactoryConfig, "resolve", forbid)
        patch.setattr(torch.nn.Linear, "__init__", forbid)
        inspected = arti.inspect_recipe_deployment(report["directory"])
    assert inspected["files_available"] and not inspected["runtime_reload_verified"]
    assert not inspected["loads_weight_tensors"] and not inspected["code_dependencies_verified"]
    assert inspected["task_recipe_sha256"] == source.contract["sha256"]
    assert inspected["python_requirements"] == ["torch>=2"]
    Path(result["checkpoint"]).unlink()
    shutil.rmtree(artifact.parent)
    loaded = load_recipe_predictor(
        report["directory"], task_factory=FACTORY if kind == "custom" else None
    )
    assert loaded.predict_records(rows) == expected
    assert loaded.model_source_sha256 == source.model_source_sha256
    assert loaded.deployment_sha256 == report["deployment_sha256"]
    equal(loaded.model.state_dict(), source.model.state_dict())
    assert arti.torch.package_recipe_deployment is arti.package_recipe_deployment


def test_resources_relocate_repackage_and_feed_predict_mode_factory(tmp_path, monkeypatch):
    current, result, rows, resources = trained(tmp_path / "source", resources=True)
    expected = current.predictor().predict_records(rows)
    package = arti.package_recipe_deployment(
        result["model"],
        tmp_path / "package",
        resources=resources,
        requirements=["example-task==1.0"],
    )
    moved = tmp_path / "moved"
    shutil.move(str(package["directory"]), str(moved))
    shutil.rmtree(tmp_path / "source")
    inspection = arti.inspect_recipe_deployment(moved)
    assert all(Path(value).is_relative_to(moved) for value in inspection["resources"].values())
    assert (Path(inspection["resources"]["processor"]) / "empty").is_dir()
    monkeypatch.setattr(torch.optim, "AdamW", forbid)
    loaded = arti.TaskPredictor.from_pretrained(moved, task_factory=RESOURCE_FACTORY)
    assert loaded.predict_records(rows) == expected
    copied = arti.package_recipe_deployment(moved, tmp_path / "repacked")
    assert copied["deployment_sha256"] == package["deployment_sha256"]
    assert (
        arti.TaskPredictor.from_pretrained(
            copied["directory"], task_factory=RESOURCE_FACTORY
        ).predict_records(rows)
        == expected
    )


@pytest.mark.parametrize("change", ["missing", "modified", "added"])
def test_resource_failure_is_reported_before_factory_or_model(tmp_path, monkeypatch, change):
    _current, result, _rows, resources = trained(tmp_path / "source", resources=True)
    package = arti.package_recipe_deployment(
        result["model"], tmp_path / "package", resources=resources
    )
    root = Path(package["directory"]) / "resources" / "processor"
    if change == "missing":
        (root / "metadata.json").unlink()
    elif change == "modified":
        (root / "metadata.json").write_text('{"family":"changed"}')
    else:
        (root / "unexpected.json").write_text("{}")
    report = arti.inspect_recipe_deployment(package["directory"])
    assert not report["files_available"] and report["missing_dependencies"]
    monkeypatch.setattr(arti.PythonFactoryConfig, "resolve", forbid)
    with pytest.raises(ValueError, match="dependencies unavailable"):
        arti.TaskPredictor.from_pretrained(package["directory"], task_factory=RESOURCE_FACTORY)
    with pytest.raises(ValueError, match="dependencies unavailable"):
        arti.package_recipe_deployment(package["directory"], tmp_path / "bad-copy")
    assert not (tmp_path / "bad-copy").exists()


@pytest.mark.parametrize("stage", ["copy", "publish"])
def test_failed_copy_or_publish_leaves_no_partial_bundle(tmp_path, monkeypatch, stage):
    _current, result, _rows, resources = trained(tmp_path / "source", resources=True)
    import arti.recipe_deployment as module

    name = "_copy_file" if stage == "copy" else "_replace_atomic_file"
    monkeypatch.setattr(
        module, name, lambda *_args: (_ for _ in ()).throw(OSError("delivery failed"))
    )
    with pytest.raises(OSError, match="delivery failed"):
        arti.package_recipe_deployment(result["model"], tmp_path / "bundle", resources=resources)
    assert not (tmp_path / "bundle").exists()
    assert not list(tmp_path.glob(".arti-task-deployment-*"))
    assert arti.inspect_recipe_deployment(result["model"])["files_available"]


def test_checkpoint_existing_destination_and_missing_resource(tmp_path):
    _current, result, _rows, _resources = trained(tmp_path / "source")
    with pytest.raises(ValueError, match="inference asset"):
        arti.package_recipe_deployment(result["checkpoint"], tmp_path / "checkpoint")
    with pytest.raises(FileNotFoundError):
        arti.package_recipe_deployment(
            result["model"], tmp_path / "missing", resources={"feature": tmp_path / "absent"}
        )
    assert not (tmp_path / "missing").exists()
    target = tmp_path / "existing"
    target.mkdir()
    (target / "keep").write_text("keep")
    with pytest.raises(FileExistsError):
        arti.package_recipe_deployment(result["model"], target)
    assert (target / "keep").read_text() == "keep"


def test_nondefault_artifact_name_is_resolved_through_package_metadata(tmp_path):
    current, _result, rows, _resources = trained(tmp_path / "source")
    asset = arti.save(
        current.binding.state_module(),
        tmp_path / "other" / "task-asset.st",
        config={"task_recipe": current.task_contract},
    ).weights_path
    expected = arti.TaskPredictor.from_pretrained(asset, task_factory=FACTORY).predict_records(rows)
    package = arti.package_recipe_deployment(asset, tmp_path / "bundle")
    assert Path(package["artifact"]).name == "task-asset.st"
    assert (
        arti.TaskPredictor.from_pretrained(
            package["directory"], task_factory=FACTORY
        ).predict_records(rows)
        == expected
    )


def test_cold_subprocess_reload_without_training_tree_or_source_factory_state(tmp_path):
    current, result, rows, resources = trained(tmp_path / "source", resources=True)
    expected = current.predictor().predict_records(rows)
    package = arti.package_recipe_deployment(
        result["model"], tmp_path / "package", resources=resources
    )
    shutil.rmtree(tmp_path / "source")
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join(
            (
                str(Path(arti.__file__).resolve().parent.parent),
                str(Path(__file__).parent),
                str(Path(__file__).parents[1]),
            )
        ),
        CUDA_VISIBLE_DEVICES="-1",
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    code = "import json,sys;import arti;from arti.recipes import load_recipe_predictor;p=load_recipe_predictor(sys.argv[1],task_factory=sys.argv[2]);print(json.dumps({'predictions':p.predict_records(json.loads(sys.argv[3])),'deployment':p.deployment_sha256}))"
    process = subprocess.run(
        [sys.executable, "-c", code, package["directory"], RESOURCE_FACTORY, json.dumps(rows)],
        cwd=package["directory"],
        env=env,
        capture_output=True,
        text=True,
        timeout=90,
        check=True,
    )
    observed = json.loads(process.stdout)
    assert (
        observed["predictions"] == expected
        and observed["deployment"] == package["deployment_sha256"]
    )


def test_cli_batch_prediction_scoring_and_actual_asgi_on_packaged_task(tmp_path, capsys):
    from fastapi.testclient import TestClient
    from arti.recipe_serving import create_recipe_app

    current, result, rows, resources = trained(tmp_path / "source", resources=True)
    expected = current.predictor().predict_records(rows)
    declared = tmp_path / "resource-bindings.json"
    requirements = tmp_path / "requirements.json"
    write_json(declared, resources)
    write_json(requirements, ["example-task==1.0"])
    assert (
        main(
            [
                "recipe",
                "package-deployment",
                result["model"],
                "--resources-json",
                str(declared),
                "--requirements-json",
                str(requirements),
                "--output",
                str(tmp_path / "bundle"),
            ]
        )
        == 0
    )
    package = json.loads(capsys.readouterr().out)
    assert main(["recipe", "inspect-deployment", package["directory"]]) == 0
    assert json.loads(capsys.readouterr().out)["files_available"]
    data = tmp_path / "requests.jsonl"
    write_jsonl(data, rows)
    run = arti.PredictionRunConfig(tmp_path / "prediction", batch_size=2)
    predicted = arti.run_prediction(package["directory"], data, run, task_factory=RESOURCE_FACTORY)
    scores = arti.score_prediction_run(
        run.output_dir,
        arti.PredictionScoringConfig("examples.prediction_scorers:regression", "mse-v1"),
    )
    assert (
        predicted["status"] == "completed"
        and scores["source"]["deployment_sha256"] == package["deployment_sha256"]
    )
    loaded = load_recipe_predictor(package["directory"], task_factory=RESOURCE_FACTORY)
    with TestClient(create_recipe_app(loaded)) as client:
        assert client.get("/health").json()["deployment_sha256"] == package["deployment_sha256"]
        reply = client.post("/predict", json={"records": rows})
        assert reply.status_code == 200 and reply.json()["predictions"] == expected


def test_recovery_pins_resource_package_in_addition_to_model_manifest(tmp_path, monkeypatch):
    _current, result, rows, resources = trained(tmp_path / "source", resources=True)
    first = arti.package_recipe_deployment(result["model"], tmp_path / "first", resources=resources)
    requests = tmp_path / "requests.jsonl"
    write_jsonl(requests, rows)
    run = arti.PredictionRunConfig(tmp_path / "run", batch_size=1)
    from test_task_prediction_run import interrupt

    with pytest.raises(KeyboardInterrupt):
        arti.run_prediction(
            first["directory"], requests, run, task_factory=RESOURCE_FACTORY, on_batch=interrupt
        )
    write_json(Path(resources["unit"]), {"name": "changed"})
    second = arti.package_recipe_deployment(
        result["model"], tmp_path / "second", resources=resources
    )
    assert second["model_manifest_sha256"] == first["model_manifest_sha256"]
    assert second["deployment_sha256"] != first["deployment_sha256"]
    monkeypatch.setattr(arti.PythonFactoryConfig, "resolve", forbid)
    with pytest.raises(ValueError, match="deployment differs"):
        arti.run_prediction(
            second["directory"], None, run, resume=True, task_factory=RESOURCE_FACTORY
        )


def test_nested_directory_link_is_not_silently_copied_as_empty(tmp_path, monkeypatch):
    _current, result, _rows, resources = trained(tmp_path / "source", resources=True)
    link = Path(resources["processor"]) / "empty"
    original = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda path: path == link or original(path))
    with pytest.raises(ValueError, match="materialize nested directory symlinks"):
        arti.package_recipe_deployment(result["model"], tmp_path / "invalid", resources=resources)
    assert not (tmp_path / "invalid").exists()


def test_generic_named_classification_still_uses_its_own_factory(tmp_path):
    cfg, rows = setup(tmp_path / "data", steps=2)
    cfg = replace(cfg, task="classification")
    current = arti.TaskRecipe.from_config(cfg)
    result = current.fit(run=policy(tmp_path / "run"))
    package = arti.package_recipe_deployment(result["model"], tmp_path / "bundle")
    assert arti.inspect_recipe_deployment(package["directory"])["requires_task_factory"]
    assert isinstance(
        load_recipe_predictor(package["directory"], task_factory=FACTORY), arti.TaskPredictor
    )
    assert (
        len(
            load_recipe_predictor(package["directory"], task_factory=FACTORY).predict_records(
                rows[-3:]
            )
        )
        == 3
    )
