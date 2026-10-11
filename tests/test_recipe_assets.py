from __future__ import annotations

import copy
from dataclasses import replace
import json
import os
import shutil
import struct
import subprocess
import sys

import pytest
import torch

import arti
from arti.cli import main
from arti.recipe_workflow import write_json, write_jsonl
from examples.asset_task_recipe import FACTORY, TRANSFORM
from test_recipe_recovery import equal, interrupt_after, policy


def fixture(tmp_path, *, shape=(4,)):
    source = tmp_path / "source"
    source.mkdir(parents=True)
    values = torch.randn(16, *shape, generator=torch.Generator().manual_seed(3))
    records, assets, bindings, metadata = [], {}, {}, {}
    for index, value in enumerate(values):
        name = f"array-{index}"
        path = source / f"{index}.bin"
        path.write_bytes(struct.pack(f"<{value.numel()}f", *value.flatten().tolist()))
        assets[name], metadata[name] = path, {"shape": list(shape)}
        records.append(
            {
                "id": str(index),
                "target": float(value.sum() * 2),
                "split": "train" if index < 9 else "validation" if index < 13 else "test",
                "group": f"group-{index}",
                "time": index,
            }
        )
        bindings[str(index)] = [arti.AssetReference(name)]
    path = source / "records.jsonl"
    write_jsonl(path, records)
    config = arti.DatasetSnapshotConfig(
        path, assets, bindings, metadata, group_field="group", source_revision="local-revision-a"
    )
    return config, records


def transform(**options):
    return arti.AssetTransformConfig(TRANSFORM, "float-array-v1", options)


def forbidden(*args, **kwargs):
    raise AssertionError("a complete cache must not run application transforms")


def test_deduplicated_snapshot_native_records_optional_reference_and_relocation(
    tmp_path, monkeypatch
):
    cfg, rows = fixture(tmp_path)
    alias = tmp_path / "alias.bin"
    shutil.copyfile(cfg.assets["array-0"], alias)
    cfg = replace(
        cfg,
        assets={**cfg.assets, "alias": alias, "missing": None},
        asset_metadata={**cfg.asset_metadata, "alias": cfg.asset_metadata["array-0"]},
        bindings={
            **cfg.bindings,
            "0": [
                arti.AssetReference("array-0"),
                arti.AssetReference("alias"),
                arti.AssetReference("missing", "reference", False),
            ],
        },
    )
    snapshot = arti.create_dataset_snapshot(cfg, tmp_path / "snapshot")
    manifest = snapshot.manifest()
    assert list(snapshot.records()) == rows
    assert manifest["inventory"] == {
        "references": 18,
        "logical_assets": 18,
        "unique_contents": 16,
        "unique_bytes": 256,
        "referenced_bytes": 272,
        "unavailable_references": 1,
    }
    assert snapshot.asset("alias") == snapshot.asset("array-0")
    assert snapshot.asset("missing") is None
    assert snapshot.references("0", role="reference")[0]["available"] is False
    assert list(snapshot.records("test")) == rows[-3:]
    prepared = arti.prepare_dataset_assets(snapshot, tmp_path / "cache", transform(), batch_size=5)
    assert prepared.report["produced"] == 16 and prepared.report["batches"] == 4
    assert prepared.files("alias") == prepared.files("array-0")
    before = prepared.manifest()
    file = prepared.files("array-0")["array.json"]
    timestamp, bytes = file.stat().st_mtime_ns, file.read_bytes()
    shutil.rmtree(cfg.records.parent)
    alias.unlink()
    moved = tmp_path / "moved"
    shutil.move(snapshot.directory, moved)
    moved_cache = tmp_path / "moved-cache"
    shutil.move(prepared.cache, moved_cache)
    monkeypatch.setattr(arti.PythonFactoryConfig, "resolve", forbidden)
    relocated = arti.DatasetSnapshot(moved)
    assert relocated.manifest() == manifest
    cached = arti.prepare_dataset_assets(relocated, moved_cache, transform(), batch_size=2)
    assert cached.report["reused"] == 16 and cached.report["produced"] == 0
    assert cached.manifest() == before
    after = cached.files("array-0")["array.json"]
    assert after.stat().st_mtime_ns == timestamp and after.read_bytes() == bytes


@pytest.mark.parametrize("change", ["bytes", "transform", "metadata", "revision"])
def test_source_or_transform_identity_invalidates_only_affected_work(tmp_path, change):
    cfg, _rows = fixture(tmp_path)
    old = arti.create_dataset_snapshot(cfg, tmp_path / "old")
    original = arti.prepare_dataset_assets(old, tmp_path / "cache", transform())
    new_transform = transform()
    if change == "bytes":
        cfg.assets["array-0"].write_bytes(struct.pack("<4f", 1, 2, 3, 4))
    elif change == "transform":
        new_transform = transform(scale=2)
    elif change == "metadata":
        cfg = replace(cfg, asset_metadata={**cfg.asset_metadata, "array-0": {"shape": [2, 2]}})
    else:
        new_transform = replace(new_transform, revision="float-array-v2")
    new = arti.create_dataset_snapshot(cfg, tmp_path / "new")
    result = arti.prepare_dataset_assets(new, tmp_path / "cache", new_transform)
    assert result.report["produced"] == (1 if change in {"bytes", "metadata"} else 16)
    assert result.report["reused"] == (15 if change in {"bytes", "metadata"} else 0)
    assert result.manifest()["sha256"] != original.manifest()["sha256"]
    if change == "bytes":
        assert old.asset("array-0").read_bytes() != cfg.assets["array-0"].read_bytes()


def test_preparation_interruption_preserves_real_committed_files(tmp_path, monkeypatch):
    cfg, _rows = fixture(tmp_path)
    snapshot = arti.create_dataset_snapshot(cfg, tmp_path / "snapshot")

    def interrupt(report):
        assert report["produced"] == 4
        raise KeyboardInterrupt

    cache = tmp_path / "cache"
    with pytest.raises(KeyboardInterrupt):
        arti.prepare_dataset_assets(snapshot, cache, transform(), batch_size=4, on_batch=interrupt)
    existing = {
        path: (path.stat().st_mtime_ns, path.read_bytes())
        for path in cache.glob("entries/*/array.json")
    }
    assert len(existing) == 4 and not list(cache.glob(".prepare-*"))
    import examples.asset_task_recipe as application

    original, calls = application.decode_arrays, []

    def tracked(requests, **kwargs):
        calls.append(len(requests))
        return original(requests, **kwargs)

    monkeypatch.setattr(application, "decode_arrays", tracked)
    resumed = arti.prepare_dataset_assets(snapshot, cache, transform(), batch_size=5)
    assert calls == [5, 5, 2]
    assert resumed.report["produced"] == 12 and resumed.report["reused"] == 4
    for path, (timestamp, bytes) in existing.items():
        assert path.stat().st_mtime_ns == timestamp and path.read_bytes() == bytes


def test_host_stop_incomplete_source_or_cache_corruption_and_no_partial_publication(
    tmp_path, monkeypatch
):
    cfg, _rows = fixture(tmp_path)
    snapshot = arti.create_dataset_snapshot(cfg, tmp_path / "snapshot")
    stopped = arti.prepare_dataset_assets(
        snapshot, tmp_path / "cache", transform(), stop_requested=lambda: True
    )
    assert not stopped.report["complete"] and stopped.report["produced"] == 0
    with pytest.raises(ValueError, match="incomplete"):
        stopped.manifest()
    ready = arti.prepare_dataset_assets(snapshot, tmp_path / "cache", transform())
    ready.files("array-0")["array.json"].write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="asset content differs"):
        arti.prepare_dataset_assets(snapshot, tmp_path / "cache", transform())
    snapshot.asset("array-1").unlink()
    with pytest.raises(ValueError, match="asset content differs"):
        arti.DatasetSnapshot(snapshot.directory)
    import arti.recipe_assets as module

    monkeypatch.setattr(
        module, "_replace_atomic_file", lambda *args: (_ for _ in ()).throw(OSError("publish"))
    )
    with pytest.raises(OSError, match="publish"):
        arti.create_dataset_snapshot(cfg, tmp_path / "failed")
    assert not (tmp_path / "failed").exists() and not list(tmp_path.glob(".failed-*"))


@pytest.mark.parametrize("failure", ["group", "required", "binding", "duplicate"])
def test_source_contract_rejects_ambiguous_dataset(tmp_path, failure):
    cfg, rows = fixture(tmp_path)
    if failure == "group":
        rows[-1]["group"] = rows[0]["group"]
    elif failure == "required":
        cfg = replace(cfg, assets={**cfg.assets, "array-0": None})
    elif failure == "binding":
        cfg = replace(cfg, bindings={**cfg.bindings, "0": [arti.AssetReference("unknown")]})
    else:
        rows[-1]["id"] = rows[0]["id"]
    write_jsonl(cfg.records, rows)
    with pytest.raises(ValueError):
        arti.create_dataset_snapshot(cfg, tmp_path / "snapshot")
    assert not (tmp_path / "snapshot").exists()


def test_roles_do_not_expose_reference_assets_to_input_preparation(tmp_path):
    cfg, _rows = fixture(tmp_path)
    cfg = replace(
        cfg, bindings={**cfg.bindings, "15": [arti.AssetReference("array-15", "reference")]}
    )
    snapshot = arti.create_dataset_snapshot(cfg, tmp_path / "snapshot")
    result = arti.prepare_dataset_assets(snapshot, tmp_path / "cache", transform(), roles=["input"])
    assert result.report["unique_sources"] == 15 and "array-15" not in result.report["assets"]
    assert "array-15" in snapshot.manifest()["assets"]


def test_unlabeled_unsplit_records_and_callback_failure_do_not_invent_evidence(
    tmp_path, monkeypatch
):
    cfg, rows = fixture(tmp_path)
    write_jsonl(cfg.records, [{"key": row["id"], "time": row["time"]} for row in rows])
    snapshot = arti.create_dataset_snapshot(
        replace(cfg, id_field="key", split_field=None, group_field=None), tmp_path / "snapshot"
    )
    assert "target" not in next(snapshot.records())
    with pytest.raises(ValueError, match="no declared split"):
        list(snapshot.records("train"))
    import examples.asset_task_recipe as application

    def fail(requests, **kwargs):
        (requests[0].directory / "partial.bin").write_bytes(b"partial")
        raise RuntimeError("decode failed")

    monkeypatch.setattr(application, "decode_arrays", fail)
    with pytest.raises(RuntimeError, match="decode failed"):
        arti.prepare_dataset_assets(snapshot, tmp_path / "cache", transform())
    assert not list((tmp_path / "cache/entries").iterdir())
    assert not list((tmp_path / "cache").glob(".prepare-*"))


def test_publication_interruption_retains_completed_entries(tmp_path, monkeypatch):
    cfg, _rows = fixture(tmp_path)
    snapshot = arti.create_dataset_snapshot(cfg, tmp_path / "snapshot")
    import arti.recipe_assets as module

    original, published = module._replace_atomic_file, []

    def publish(source, target):
        if target.parent.name == "entries":
            if published:
                raise OSError("commit interrupted")
            published.append(target)
        return original(source, target)

    monkeypatch.setattr(module, "_replace_atomic_file", publish)
    with pytest.raises(OSError, match="commit interrupted"):
        arti.prepare_dataset_assets(snapshot, tmp_path / "cache", transform(), batch_size=4)
    assert len(list((tmp_path / "cache/entries").iterdir())) == 1
    completed = (published[0] / "array.json").read_bytes()
    monkeypatch.setattr(module, "_replace_atomic_file", original)
    result = arti.prepare_dataset_assets(snapshot, tmp_path / "cache", transform(), batch_size=4)
    assert result.report["produced"] == 15 and result.report["reused"] == 1
    assert (published[0] / "array.json").read_bytes() == completed


@pytest.mark.parametrize("shape", [(4,), (2, 2)])
def test_native_asset_task_training_recovery_prediction_and_portable_delivery(tmp_path, shape):
    cfg, rows = fixture(tmp_path, shape=shape)
    snapshot = arti.create_dataset_snapshot(cfg, tmp_path / "snapshot")
    config = arti.TaskRecipeConfig(
        "array-regression",
        FACTORY,
        options={"input_shape": list(shape), "batch_size": 4, "dropout": 0.2},
        resources={"snapshot": str(snapshot.directory), "cache": str(tmp_path / "cache")},
        training=arti.ARTITrainingConfig(
            device="cpu", steps=4, learning_rate=0.02, gradient_accumulation_steps=2
        ),
    )
    full, split = arti.TaskRecipe.from_config(config), arti.TaskRecipe.from_config(config)
    initial_rng = torch.get_rng_state().clone()
    full.fit(run=policy(tmp_path / "full"))
    torch.set_rng_state(initial_rng)
    with pytest.raises(KeyboardInterrupt):
        split.fit(run=policy(tmp_path / "partial"), on_checkpoint=interrupt_after(2))
    moved, moved_cache = tmp_path / "moved", tmp_path / "moved-cache"
    shutil.move(snapshot.directory, moved)
    shutil.move(tmp_path / "cache", moved_cache)
    shutil.rmtree(cfg.records.parent)
    resources = {"snapshot": str(moved), "cache": str(moved_cache)}
    restored = arti.TaskRecipe.from_config(replace(config, resources=resources))
    recovered = restored.resume_run(tmp_path / "partial", output_dir=tmp_path / "recovered")
    assert recovered["optimizer_updates_this_attempt"] == 2
    assert restored.task_fingerprint == full.task_fingerprint
    equal(full.model.state_dict(), restored.model.state_dict())
    equal(full.session.optimizer.state_dict(), restored.session.optimizer.state_dict())
    assert full.session.loss_history == restored.session.loss_history
    assert restored.task_contract["objective"] == "scaled-mse"
    assert tuple(restored.train_data[0]["features"].shape[1:]) == shape
    assert restored.data_preparation["produced"] == 0
    assert json.loads((tmp_path / "recovered/data_preparation.json").read_text())["reused"] == 16
    package = tmp_path / "delivery"
    arti.package_recipe_deployment(recovered["model"], package, resources=resources)
    requests = [{"id": row["id"]} for row in rows[-3:]]
    before = restored.predictor().predict_records(requests)
    shutil.rmtree(moved)
    shutil.rmtree(moved_cache)
    delivered = arti.TaskPredictor.from_pretrained(package, task_factory=FACTORY)
    state = copy.deepcopy(delivered.model.state_dict())
    assert delivered.predict_records(requests) == before
    equal(state, delivered.model.state_dict())
    assert delivered.contract["sha256"] == restored.task_fingerprint
    request_file = tmp_path / "requests.jsonl"
    write_jsonl(request_file, requests)
    result = arti.run_prediction(
        package,
        request_file,
        arti.PredictionRunConfig(tmp_path / "prediction", batch_size=2),
        task_factory=FACTORY,
    )
    assert result["status"] == "completed"
    targets = tmp_path / "targets.jsonl"
    write_jsonl(targets, rows[-3:])
    scoring = arti.score_prediction_run(
        tmp_path / "prediction",
        arti.PredictionScoringConfig("examples.prediction_scorers:regression", "original-unit-v1"),
        targets=targets,
    )
    assert len(scoring["scores"]) == 3
    assert scoring["metrics"]["mse"] >= 0
    code = (
        "import json,sys;from pathlib import Path;import arti;"
        "assert Path(arti.__file__).resolve()==Path(sys.argv[3]).resolve();"
        "p=arti.TaskPredictor.from_pretrained(sys.argv[1],task_factory='examples.asset_task_recipe:build_task');"
        "print(json.dumps(p.predict_records(json.loads(sys.argv[2]))))"
    )
    process = subprocess.run(
        [sys.executable, "-c", code, str(package), json.dumps(requests), arti.__file__],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(sys.path)},
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(process.stdout) == before


def test_cli_relative_config_inspection_and_cache_without_factory_on_reuse(
    tmp_path, monkeypatch, capsys
):
    cfg, _rows = fixture(tmp_path)
    config = tmp_path / "snapshot.json"
    write_json(
        config,
        {
            "records": "source/records.jsonl",
            "assets": {name: f"source/{path.name}" for name, path in cfg.assets.items()},
            "bindings": {
                name: [item.to_dict() for item in refs] for name, refs in cfg.bindings.items()
            },
            "asset_metadata": cfg.asset_metadata,
            "group_field": "group",
        },
    )
    snapshot = tmp_path / "snapshot"
    assert main(["recipe", "snapshot-data", str(config), "--output", str(snapshot)]) == 0
    sha = json.loads(capsys.readouterr().out)["sha256"]
    assert main(["recipe", "inspect-data", str(snapshot)]) == 0
    assert json.loads(capsys.readouterr().out)["sha256"] == sha
    transform_file = tmp_path / "transform.json"
    write_json(transform_file, transform().to_dict())
    args = [
        "recipe",
        "prepare-assets",
        str(snapshot),
        "--cache-dir",
        str(tmp_path / "cache"),
        "--transform",
        str(transform_file),
        "--batch-size",
        "3",
        "--role",
        "input",
    ]
    assert main(args) == 0
    first = json.loads(capsys.readouterr().out)
    assert first["produced"] == 16 and first["batches"] == 6
    monkeypatch.setattr(arti.PythonFactoryConfig, "resolve", forbidden)
    assert main(args) == 0
    second = json.loads(capsys.readouterr().out)
    assert second["reused"] == 16 and second["dataset"] == first["dataset"]
    assert arti.torch.DatasetSnapshot is arti.DatasetSnapshot
