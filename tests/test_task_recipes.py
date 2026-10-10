from __future__ import annotations

from dataclasses import replace
import json
import shutil
import threading
from urllib.request import ProxyHandler, Request, build_opener

import pytest
import torch
from torch.nn import functional as F

import arti
from arti.cli import main
from arti.recipe_data import ClassificationBatches, ClassificationDataset, ClassificationFields
from arti.recipes import ClassificationPredictor, ClassificationRecipe, ClassificationRecipeConfig


@pytest.fixture(autouse=True)
def small_cpu_threads(monkeypatch):
    monkeypatch.setenv("PYTORCH_NVML_BASED_CUDA_CHECK", "0")
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _records():
    return [
        {
            "id": f"{split}-{index}",
            "features": [value, value * 0.5, 3.0],
            "label": "negative" if value < 0 else "positive",
            "split": split,
        }
        for split, values in (
            ("train", [-3, 2, -2, 1, -1, 3, -4]),
            ("validation", [-5, 5, 4]),
            ("test", [-6, 6, 2]),
        )
        for index, value in enumerate(values)
    ]


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def _config(tmp_path, **kwargs):
    data = _write(tmp_path / "records.jsonl", _records())
    return ClassificationRecipeConfig(
        data,
        hidden_dim=8,
        batch_size=3,
        training=arti.ARTITrainingConfig(device="cpu", steps=10, learning_rate=0.03),
        **kwargs,
    )


def test_preprocessing_and_label_mapping_use_only_train(tmp_path):
    rows = _records()
    for row in rows:
        if row["split"] != "train":
            row["features"] = [10000, 20000, 3.0]
    data = ClassificationDataset(_write(tmp_path / "data.jsonl", rows))
    mean, scale = data.preprocessing("standardize")
    torch.testing.assert_close(mean, data.splits["train"].features.mean(0))
    assert float(mean[0]) < 1
    assert float(scale[-1]) == 1
    assert data.labels == ("negative", "positive")
    assert data.manifest()["splits"]["test"]["samples"] == 3


def test_explicit_label_order_and_custom_fields(tmp_path):
    fields = ClassificationFields(
        id="sample_id", features="pixels", label="class_name", split="role"
    )
    rows = [
        {
            "sample_id": row["id"],
            "pixels": row["features"],
            "class_name": row["label"],
            "role": row["split"],
        }
        for row in _records()
    ]
    data = ClassificationDataset(
        _write(tmp_path / "data.jsonl", rows), fields=fields, labels=("positive", "negative")
    )
    assert int(data.splits["train"].labels[0]) == 1
    assert data.fields.features == "pixels"


@pytest.mark.parametrize(
    "change,message",
    [
        (lambda rows: rows[-1].update(id=rows[0]["id"]), "duplicate"),
        (lambda rows: rows[-1].update(label="unknown"), "absent from train"),
        (lambda rows: rows[-1].update(features=[1]), "expected 3 features"),
        (lambda rows: rows[-1].update(features=[True, 2, 3]), "booleans"),
        (lambda rows: rows[-1].update(features=[float("nan"), 2, 3]), "finite"),
        (lambda rows: rows[-1].update(features=[1e100, 2, 3]), "float32"),
        (lambda rows: rows[-1].update(split="holdout"), "split must"),
    ],
)
def test_data_errors_report_the_task_contract(tmp_path, change, message):
    rows = _records()
    change(rows)
    with pytest.raises(ValueError, match=message):
        ClassificationDataset(_write(tmp_path / "data.jsonl", rows))


def test_batches_resume_across_an_epoch_boundary(tmp_path):
    data = ClassificationDataset(_config(tmp_path).data_path)
    batches = ClassificationBatches(data.splits["train"], batch_size=3, seed=12)
    iterator = iter(batches)
    consumed = [next(iterator) for _ in range(3)]
    assert sum(len(batch["labels"]) for batch in consumed) == 7
    snapshot = batches.state_dict()
    assert snapshot["cursor"] == 7
    with pytest.raises(StopIteration):
        next(iterator)
    expected = next(iter(batches))
    fresh = ClassificationBatches(data.splits["train"], batch_size=3, seed=12)
    fresh.load_state_dict(snapshot)
    actual = next(iter(fresh))
    torch.testing.assert_close(actual["features"], expected["features"], rtol=0, atol=0)
    assert fresh.epoch == 1


def test_native_graph_recipe_has_gradients_and_real_classification_loss(tmp_path):
    recipe = ClassificationRecipe(_config(tmp_path))
    before = {key: value.clone() for key, value in recipe.model.state_dict().items()}
    initial = recipe.evaluate()["loss"]
    assert recipe.check()["ok"]
    for key, value in recipe.model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    plan = recipe.plan()
    assert plan["task_recipe"]["objective"] == "cross_entropy"
    assert plan["deployment"]["labels_included"]
    report = recipe.fit(
        run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "run", checkpoint_steps=3)
    )
    assert report["validation"]["loss"] < initial
    assert recipe.model.graph.nodes["encode"].fabric.program.instructions
    assert report["optimizer_steps"] == 10
    assert (tmp_path / "run" / "resolved_recipe.json").is_file()
    assert (tmp_path / "run" / "split_manifest.json").is_file()
    assert any(
        not torch.equal(value, before[key]) for key, value in recipe.model.state_dict().items()
    )
    assert not (tmp_path / "run" / "test.json").exists()


def test_evaluation_is_sample_weighted_and_keeps_raw_ids(tmp_path):
    config = replace(_config(tmp_path), batch_size=2)
    recipe = ClassificationRecipe(config)
    data = recipe.data.splits["validation"]
    logits = recipe.model(data.features)["logits"]
    expected = F.cross_entropy(logits, data.labels)
    report = recipe.evaluate(output=tmp_path / "validation.json")
    assert report["loss"] == pytest.approx(float(expected.detach()), rel=1e-6)
    predictions = [
        json.loads(line)
        for line in (tmp_path / "validation.predictions.jsonl").read_text().splitlines()
    ]
    assert [row["id"] for row in predictions] == list(data.ids)
    assert report["samples"] == 3
    assert sum(sum(row) for row in report["confusion_matrix"]) == 3
    assert recipe.model.training


def test_exact_resume_preserves_epoch_order_optimizer_and_weights(tmp_path):
    config = _config(tmp_path)
    config = replace(config, training=replace(config.training, gradient_accumulation_steps=2))
    complete = ClassificationRecipe(config)
    complete.fit(steps=7, run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "complete"))
    partial = ClassificationRecipe(config)
    first = partial.fit(steps=3, run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "partial"))
    restored = ClassificationRecipe(config)
    second = restored.fit(
        steps=4,
        resume_from_checkpoint=first["checkpoint"],
        run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "continued"),
    )
    assert second["optimizer_steps"] == 7 and second["microbatches"] == 14
    for key, value in complete.model.state_dict().items():
        torch.testing.assert_close(restored.model.state_dict()[key], value, rtol=0, atol=0)
    assert complete.train_data.state_dict() == restored.train_data.state_dict()
    assert complete.session.loss_history == restored.session.loss_history


@pytest.mark.parametrize(
    "change,message",
    [
        (lambda config: replace(config, batch_size=2), "data/order"),
        (lambda config: replace(config, seed=100), "data/order"),
        (
            lambda config: replace(
                config, training=replace(config.training, gradient_accumulation_steps=2)
            ),
            "gradient_accumulation",
        ),
        (lambda config: replace(config, labels=("positive", "negative")), "labels"),
    ],
)
def test_resume_does_not_silently_change_task_or_batching(tmp_path, change, message):
    config = _config(tmp_path)
    recipe = ClassificationRecipe(config)
    report = recipe.fit(steps=1, run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "run"))
    changed = ClassificationRecipe(change(config))
    with pytest.raises(ValueError, match=message):
        changed.session.load_checkpoint(report["checkpoint"])


def test_portable_inference_asset_needs_no_training_data_or_sklearn(tmp_path):
    config = _config(tmp_path)
    recipe = ClassificationRecipe(config)
    recipe.fit(run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "run"))
    asset = recipe.export(tmp_path / "deploy")
    predictor = ClassificationPredictor.from_pretrained(asset)
    raw = recipe.data.splits["test"].features
    expected = predictor.predict(raw, batch_size=2)
    config.data_path.unlink()
    shutil.copytree(asset.parent, tmp_path / "moved")
    portable = ClassificationPredictor.from_pretrained(tmp_path / "moved")
    assert portable.predict(raw, batch_size=2) == expected
    torch.testing.assert_close(
        portable.model.feature_mean, recipe.model.feature_mean, rtol=0, atol=0
    )
    manifest = json.loads(asset.with_suffix(".json").read_text())
    assert str(tmp_path) not in json.dumps(manifest["architecture"]["config"])
    assert portable.predict([]) == []
    assert (
        portable.predict_records([{"id": "unlabeled", "features": raw[0].tolist()}])[0]["id"]
        == "unlabeled"
    )
    with pytest.raises(ValueError, match="expected 3 features"):
        portable.predict([[1, 2]])


def test_reading_relative_config_does_not_depend_on_working_directory(tmp_path):
    config = _config(tmp_path)
    payload = config.to_dict()
    payload["data"]["path"] = "records.jsonl"
    payload["run"]["output_dir"] = "runs"
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(payload))
    loaded = ClassificationRecipeConfig.read(path)
    assert loaded.data_path == tmp_path / "records.jsonl"
    assert loaded.run.output_dir == str(tmp_path / "runs")


def test_deployed_asset_evaluates_new_labels_without_refitting(tmp_path, capsys):
    recipe = ClassificationRecipe(_config(tmp_path))
    model = recipe.export(tmp_path / "model")
    predictor = ClassificationPredictor.from_pretrained(model)
    rows = [row for row in _records() if row["split"] == "test"]
    expected = recipe.evaluate("test")
    recipe.config.data_path.unlink()
    mean = predictor.model.feature_mean.clone()
    actual = predictor.evaluate_records(rows, batch_size=2, output=tmp_path / "external.json")
    assert actual["loss"] == pytest.approx(expected["loss"], rel=1e-6)
    assert actual["accuracy"] == expected["accuracy"]
    torch.testing.assert_close(predictor.model.feature_mean, mean, rtol=0, atol=0)
    data = _write(tmp_path / "new-labels.jsonl", rows)
    assert (
        main(
            [
                "recipe",
                "evaluate",
                str(model),
                "--data",
                str(data),
                "--output",
                str(tmp_path / "evaluated.json"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["samples"] == 3


def test_resolved_config_retains_actual_run_overrides_and_paths(tmp_path):
    recipe = ClassificationRecipe(_config(tmp_path))
    recipe.fit(steps=2, run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "overridden"))
    resolved = ClassificationRecipeConfig.read(tmp_path / "overridden" / "resolved_recipe.json")
    assert resolved.training.steps == 2
    assert resolved.data_path == recipe.config.data_path
    assert resolved.run.output_dir == str(tmp_path / "overridden")


def test_interrupted_recipe_retains_config_split_ids_and_first_checkpoint(tmp_path, monkeypatch):
    recipe = ClassificationRecipe(_config(tmp_path))
    forward = recipe.model.forward
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated process work failure")
        return forward(*args, **kwargs)

    monkeypatch.setattr(recipe.model, "forward", interrupted)
    with pytest.raises(RuntimeError, match="simulated"):
        recipe.fit(steps=3, run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "failed"))
    assert (tmp_path / "failed" / "checkpoints" / "step-00000001" / "model.arti.st").exists()
    resolved = ClassificationRecipeConfig.read(tmp_path / "failed" / "resolved_recipe.json")
    assert resolved.training.steps == 3
    manifest = json.loads((tmp_path / "failed" / "split_manifest.json").read_text())
    assert manifest["splits"]["train"]["ids"] == list(recipe.data.splits["train"].ids)


def test_cli_covers_plan_train_eval_predict_and_export(tmp_path, capsys):
    config = _config(tmp_path)
    config_path = tmp_path / "recipe.json"
    config_path.write_text(json.dumps(config.to_dict()))
    assert main(["recipe", "plan", str(config_path)]) == 0
    assert json.loads(capsys.readouterr().out)["task_recipe"]["objective"] == "cross_entropy"
    assert main(["recipe", "check", str(config_path)]) == 0
    capsys.readouterr()
    assert (
        main(
            [
                "recipe",
                "train",
                str(config_path),
                "--run-dir",
                str(tmp_path / "run"),
                "--steps",
                "3",
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    model = tmp_path / "run" / "model"
    assert (
        main(
            [
                "recipe",
                "evaluate",
                str(config_path),
                "--model",
                str(model),
                "--split",
                "test",
                "--output",
                str(tmp_path / "test.json"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["split"] == "test"
    inputs = _write(tmp_path / "predict.jsonl", [{"id": "new", "features": [1, 0.5, 3]}])
    assert (
        main(
            [
                "recipe",
                "predict",
                str(model),
                "--data",
                str(inputs),
                "--output",
                str(tmp_path / "prediction.jsonl"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["samples"] == 1
    assert (
        main(
            [
                "recipe",
                "export",
                str(config_path),
                "--checkpoint",
                result["checkpoint"],
                "--output",
                str(tmp_path / "exported"),
            ]
        )
        == 0
    )
    capsys.readouterr()
    assert ClassificationPredictor.from_pretrained(tmp_path / "exported").predict([[1, 0.5, 3]])


def test_recipe_requires_the_actual_task_loss(tmp_path):
    config = _config(tmp_path)
    with pytest.raises(ValueError, match="cross-entropy"):
        replace(config, training=replace(config.training, objective="tensor_alignment"))


def test_native_recipe_hot_path_is_a_full_tensor_graph(tmp_path):
    recipe = ClassificationRecipe(_config(tmp_path))
    features = recipe.data.splits["validation"].features
    compiled = torch.compile(recipe.model, backend="eager", fullgraph=True)
    expected = recipe.model(features)["logits"]
    torch.testing.assert_close(compiled(features)["logits"], expected)
    loss = compiled(features, recipe.data.splits["validation"].labels)["loss"]
    gradients = torch.autograd.grad(loss, tuple(recipe.model.parameters()))
    assert all(
        gradient is not None and bool(torch.isfinite(gradient).all()) for gradient in gradients
    )


def test_output_lowering_keeps_external_formula_operand_identity(tmp_path):
    recipe = ClassificationRecipe(_config(tmp_path))
    assert {id(value) for value in recipe.model._plan.parameters()} == {
        id(value) for value in recipe.model.parameters()
    }


def test_recipe_torch_exports_match_root():
    import arti.torch as backend

    assert backend.ClassificationRecipe is arti.ClassificationRecipe
    assert backend.ClassificationPredictor is arti.ClassificationPredictor


def test_local_http_worker_loads_once_and_accepts_request_batches(tmp_path):
    from http.server import HTTPServer
    from examples.serve_classification import handler_for

    recipe = ClassificationRecipe(_config(tmp_path))
    recipe.fit(steps=4, run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "run"))
    predictor = ClassificationPredictor.from_pretrained(tmp_path / "run" / "model")
    server = HTTPServer(("127.0.0.1", 0), handler_for(predictor))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        root = f"http://127.0.0.1:{server.server_port}"
        opener = build_opener(ProxyHandler({}))
        with opener.open(root + "/health", timeout=5) as response:
            assert json.load(response)["ready"]
        records = [
            {"id": "first", "features": [1, 0.5, 3]},
            {"id": "second", "features": [-2, -1, 3]},
        ]
        expected = predictor.predict_records(records)
        for _ in range(2):
            request = Request(
                root + "/predict",
                data=json.dumps({"records": records}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with opener.open(request, timeout=5) as response:
                result = json.load(response)
            assert result["predictions"] == expected
            assert result["labels"] == list(predictor.labels)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
    assert not thread.is_alive()
