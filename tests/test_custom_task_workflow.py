from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest
import torch

import arti
from arti.cli import main
from arti.recipe_workflow import write_json, write_jsonl
from examples.custom_task_recipe import FACTORY
from test_recipe_recovery import equal, interrupt_after, policy


def setup(tmp_path, *, kind="regression", engine="torch", shape=(4,), steps=6):
    tmp_path.mkdir(parents=True, exist_ok=True)
    x = torch.randn(16, *shape, generator=torch.Generator().manual_seed(3))
    y = x.flatten(1).sum(-1) * 2 if kind == "regression" else (x.flatten(1).sum(-1) > 0).long()
    records = [
        {
            "id": str(index),
            "features": features.tolist(),
            "target": target.item(),
            "split": "train" if index < 9 else "validation" if index < 13 else "test",
        }
        for index, (features, target) in enumerate(zip(x, y, strict=True))
    ]
    data = tmp_path / "records.jsonl"
    write_jsonl(data, records)
    config = arti.TaskRecipeConfig(
        task=f"numeric-{kind}",
        factory=FACTORY,
        options={
            "kind": kind,
            "input_shape": list(shape),
            "classes": 2,
            "batch_size": 4,
            "dropout": 0.2,
        },
        resources={"data": str(data)},
        training=arti.ARTITrainingConfig(
            device="cpu",
            steps=steps,
            learning_rate=0.02,
            engine=engine,
            gradient_accumulation_steps=2,
        ),
    )
    return config, records


def forbid(*_args, **_kwargs):
    raise AssertionError("inference/metadata planning must not create a trainer or optimizer")


@pytest.mark.parametrize("kind,shape", [("regression", (4,)), ("classification", (2, 2))])
@pytest.mark.parametrize("engine", ["torch", "accelerate"])
def test_two_nonlanguage_tasks_exact_resumption_and_runtime_resource_relocation(
    tmp_path, monkeypatch, kind, shape, engine
):
    if engine == "accelerate":
        pytest.importorskip("accelerate")
    cfg, _rows = setup(tmp_path / "data", kind=kind, shape=shape, engine=engine)
    full, split = arti.TaskRecipe.from_config(cfg), arti.TaskRecipe.from_config(cfg)
    initial = copy.deepcopy(full.model.state_dict())
    initial_rng = torch.get_rng_state().clone()
    full.fit(run=policy(tmp_path / "full"))
    final_rng = torch.get_rng_state().clone()
    torch.set_rng_state(initial_rng)
    with pytest.raises(KeyboardInterrupt):
        split.fit(run=policy(tmp_path / "partial"), on_checkpoint=interrupt_after(2))
    moved = tmp_path / "relocated.jsonl"
    shutil.move(cfg.resources["data"], moved)
    captured, resume = [], arti.TaskRecipe.resume_run

    def capture(self, *args, **kwargs):
        captured.append(self)
        return resume(self, *args, **kwargs)

    monkeypatch.setattr(arti.TaskRecipe, "resume_run", capture)
    report = arti.resume_recipe_run(
        tmp_path / "partial", output_dir=tmp_path / "recovered", resources={"data": str(moved)}
    )
    restored = captured[0]
    assert restored.task_fingerprint == full.task_fingerprint
    assert report["optimizer_steps"] == 6 and report["optimizer_updates_this_attempt"] == 4
    equal(full.model.state_dict(), restored.model.state_dict())
    equal(full.session.optimizer.state_dict(), restored.session.optimizer.state_dict())
    equal(full.session.scheduler.state_dict(), restored.session.scheduler.state_dict())
    assert full.session.loss_history == restored.session.loss_history
    assert full.session.micro_batches_seen == restored.session.micro_batches_seen == 12
    assert torch.equal(torch.get_rng_state(), final_rng)
    assert any(
        not torch.equal(initial[key], value) for key, value in full.model.state_dict().items()
    )
    assert report["recovery"]["resources"] == {"data": str(moved)}
    assert full.task_contract["objective"] == (
        "scaled-mse" if kind == "regression" else "cross-entropy"
    )
    assert "resources" not in full.task_contract and not hasattr(full, "tokenizer")


def test_deployment_is_data_independent_optimizer_free_and_read_only(tmp_path, monkeypatch):
    cfg, rows = setup(tmp_path / "data")
    recipe = arti.TaskRecipe.from_config(cfg)
    report = recipe.fit(run=policy(tmp_path / "run"))
    expected = recipe.predictor().predict_records(rows[-3:], batch_size=2)
    Path(cfg.resources["data"]).unlink()
    import arti.task_recipe as module

    monkeypatch.setattr(module, "trainer", forbid)
    monkeypatch.setattr(torch.optim, "AdamW", forbid)
    predictor = arti.TaskPredictor.from_pretrained(report["model"], task_factory=FACTORY)
    state = copy.deepcopy(predictor.model.state_dict())
    rng = torch.get_rng_state().clone()
    assert predictor.predict_records(rows[-3:], batch_size=2) == expected
    evaluation = predictor.evaluate_records(rows[-3:], output=tmp_path / "heldout.json")
    assert evaluation["samples"] == 3 and evaluation["mae"] >= 0
    assert evaluation["model_source_sha256"] == predictor.model_source_sha256
    assert evaluation["dataset_sha256"] != recipe.dataset_manifest["sha256"]
    equal(predictor.model.state_dict(), state)
    assert torch.equal(rng, torch.get_rng_state())
    assert not hasattr(predictor, "session")


def test_config_roundtrip_and_factory_before_construction(tmp_path, monkeypatch):
    cfg, _rows = setup(tmp_path)
    path = tmp_path / "task.json"
    write_json(path, cfg.to_dict())
    assert arti.TaskRecipeConfig.read(path) == cfg
    assert arti.torch.TaskRecipe is arti.TaskRecipe
    assert arti.torch.TaskPredictor is arti.TaskPredictor
    assert arti.torch.PythonFactoryConfig is arti.PythonFactoryConfig
    monkeypatch.setattr(arti.PythonFactoryConfig, "resolve", forbid)
    with pytest.raises(ValueError, match="reference/kwargs differ"):
        arti.TaskRecipe.from_config(cfg, task_factory="missing.package:factory")


def test_complete_generic_cli_train_evaluate_export_predict_and_metadata_plan(
    tmp_path, monkeypatch, capsys
):
    cfg, rows = setup(tmp_path / "data", steps=3)
    path = tmp_path / "recipe.json"
    write_json(path, cfg.to_dict())
    assert main(["recipe", "plan", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["task_recipe"]["task"] == "numeric-regression"
    assert main(["recipe", "check", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"]
    run = tmp_path / "run"
    assert main(["recipe", "train", str(path), "--run-dir", str(run)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["optimizer_steps"] == 3
    assert (
        main(
            [
                "recipe",
                "export",
                str(path),
                "--checkpoint",
                result["checkpoint"],
                "--output",
                str(tmp_path / "export"),
            ]
        )
        == 0
    )
    exported = json.loads(capsys.readouterr().out)["model"]
    independent = tmp_path / "test.jsonl"
    write_jsonl(independent, rows[-3:])
    import arti.task_recipe as module

    monkeypatch.setattr(module, "trainer", forbid)
    monkeypatch.setattr(torch.optim, "AdamW", forbid)
    assert (
        main(
            ["recipe", "evaluate", exported, "--data", str(independent), "--task-factory", FACTORY]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["samples"] == 3
    output = tmp_path / "prediction.jsonl"
    assert (
        main(
            [
                "recipe",
                "predict",
                exported,
                "--data",
                str(independent),
                "--task-factory",
                FACTORY,
                "--batch-size",
                "2",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["samples"] == 3
    assert [json.loads(line)["id"] for line in output.read_text().splitlines()] == [
        row["id"] for row in rows[-3:]
    ]
    monkeypatch.setattr(arti.PythonFactoryConfig, "resolve", forbid)
    assert main(["recipe", "resume-run", str(run), "--plan-only"]) == 0
    assert json.loads(capsys.readouterr().out)["loads_checkpoint_tensors"] is False
    assert main(["recipe", "resume-run", str(run)]) == 0
    assert json.loads(capsys.readouterr().out)["optimizer_updates_this_attempt"] == 0


def test_task_mismatch_and_explicit_inference_factory(tmp_path):
    cfg, _rows = setup(tmp_path / "data")
    recipe = arti.TaskRecipe.from_config(cfg)
    result = recipe.fit(run=policy(tmp_path / "run"))
    changed = arti.TaskRecipe.from_config(replace(cfg, options={**cfg.options, "dropout": 0.4}))
    with pytest.raises(ValueError, match="task/data/objective/factory"):
        changed.session.load_checkpoint(result["checkpoint"])
    with pytest.raises(ValueError, match="explicit task_factory required"):
        arti.TaskPredictor.from_pretrained(result["model"])
    with pytest.raises(ValueError, match="reference/kwargs differ"):
        arti.TaskPredictor.from_pretrained(result["model"], task_factory="another.task:factory")


def test_metadata_plan_and_cli_resume_accept_named_resource_overrides(tmp_path, capsys):
    cfg, _rows = setup(tmp_path / "data")
    recipe = arti.TaskRecipe.from_config(cfg)
    with pytest.raises(KeyboardInterrupt):
        recipe.fit(run=policy(tmp_path / "source"), on_checkpoint=interrupt_after(2))
    moved = tmp_path / "moved.jsonl"
    shutil.move(cfg.resources["data"], moved)
    resources = tmp_path / "resources.json"
    write_json(resources, {"data": str(moved)})
    assert (
        main(
            [
                "recipe",
                "resume-run",
                str(tmp_path / "source"),
                "--plan-only",
                "--resources-json",
                str(resources),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["resources"]["data"] == str(moved)
    assert (
        main(
            [
                "recipe",
                "resume-run",
                str(tmp_path / "source"),
                "--resources-json",
                str(resources),
                "--run-dir",
                str(tmp_path / "recovered"),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["optimizer_steps"] == 6 and result["optimizer_updates_this_attempt"] == 4


def test_early_stop_finalize_keeps_last_optimizer_and_requires_original_evaluator(
    tmp_path, monkeypatch
):
    cfg, _rows = setup(tmp_path / "data")
    recipe = arti.TaskRecipe.from_config(cfg)
    monkeypatch.setattr(
        recipe, "export", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("export"))
    )

    def validation(_session):
        return {"loss": 1.0}

    with pytest.raises(OSError):
        recipe.fit(
            run=policy(
                tmp_path / "source",
                evaluation_steps=1,
                validation=arti.ARTIValidationConfig(patience=1),
            ),
            validation_fn=validation,
        )
    plan = arti.plan_recipe_resume(tmp_path / "source")
    assert plan["action"] == "finalize" and plan["training_terminal_reason"] == "early_stopping"
    with pytest.raises(ValueError, match="requires validation_fn"):
        arti.resume_recipe_run(tmp_path / "source", output_dir=tmp_path / "incorrect")
    result = arti.resume_recipe_run(
        tmp_path / "source", output_dir=tmp_path / "finalized", validation_fn=validation
    )
    assert (
        result["optimizer_updates_this_attempt"] == 0 and result["selection"]["selected_step"] == 1
    )


def test_native_asgi_and_cli_server_factory(tmp_path, monkeypatch, capsys):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    import uvicorn
    import arti.task_recipe as module

    cfg, rows = setup(tmp_path / "data", steps=2)
    recipe = arti.TaskRecipe.from_config(cfg)
    result = recipe.fit(run=policy(tmp_path / "run"))
    monkeypatch.setattr(module, "trainer", forbid)
    monkeypatch.setattr(torch.optim, "AdamW", forbid)
    predictor = arti.TaskPredictor.from_pretrained(result["model"], task_factory=FACTORY)
    expected = predictor.predict_records(rows[-3:], batch_size=2)
    with TestClient(arti.create_recipe_app(predictor)) as client:
        assert client.get("/health").json()["ready"]
        response = client.post("/predict", json={"records": rows[-3:], "batch_size": 2})
        assert response.status_code == 200 and response.json()["predictions"] == expected
    captured = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **kwargs: captured.append(app.state.predictor))
    assert main(["recipe", "serve", result["model"], "--task-factory", FACTORY]) == 0
    capsys.readouterr()
    assert len(captured) == 1 and isinstance(captured[0], arti.TaskPredictor)


@pytest.mark.parametrize("task", ["classification", "causal-lm"])
def test_application_name_is_not_a_builtin_adapter_selector(tmp_path, capsys, task):
    cfg, rows = setup(tmp_path / "data", steps=2)
    cfg = replace(cfg, task=task)
    path = tmp_path / "recipe.json"
    write_json(path, cfg.to_dict())
    assert main(["recipe", "plan", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["task_recipe"]["format"] == "arti.task-recipe"
    recipe = arti.TaskRecipe.from_config(cfg)
    result = recipe.fit(run=policy(tmp_path / "run"))
    assert arti.plan_recipe_resume(tmp_path / "run")["resource_scope"] == "task-factory"
    source = tmp_path / "test.jsonl"
    write_jsonl(source, rows[-3:])
    assert (
        main(
            [
                "recipe",
                "predict",
                result["model"],
                "--data",
                str(source),
                "--output",
                str(tmp_path / "pred.jsonl"),
                "--task-factory",
                FACTORY,
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["samples"] == 3
    assert (
        main(
            [
                "recipe",
                "evaluate",
                result["model"],
                "--data",
                str(source),
                "--task-factory",
                FACTORY,
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["samples"] == 3


@pytest.mark.parametrize("engine", ["torch", "accelerate"])
def test_binding_owns_only_selected_trainable_parameters(tmp_path, engine):
    if engine == "accelerate":
        pytest.importorskip("accelerate")
    cfg, rows = setup(tmp_path / "data", steps=3, engine=engine)
    cfg = replace(cfg, options={**cfg.options, "use_binding": True})
    recipe = arti.TaskRecipe.from_config(cfg)
    frozen = copy.deepcopy(recipe.model.output.state_dict())
    selected = {id(parameter) for parameter in recipe.model.program.parameters()}
    assert selected == {id(parameter) for parameter in recipe.binding.parameters()}
    assert all(name.startswith("program.") for name in recipe.task_contract["trainable_parameters"])
    result = recipe.fit(run=policy(tmp_path / "run"))
    equal(frozen, recipe.model.output.state_dict())
    assert len(recipe.session.optimizer.state) == len(selected)
    loaded = arti.TaskPredictor.from_pretrained(result["model"], task_factory=FACTORY)
    assert loaded.predict_records(rows[-3:]) == recipe.predictor().predict_records(rows[-3:])


def test_check_and_prediction_restore_resource_state_buffers_and_rng(tmp_path):
    import random
    from arti.attachment_training import _rng_state
    from torch.utils.data import DataLoader

    cfg, rows = setup(tmp_path / "data")
    recipe = arti.TaskRecipe.from_config(cfg)
    recipe.train_data = DataLoader(recipe.train_data, batch_size=None, num_workers=0)
    before, rng = copy.deepcopy(recipe.model.state_dict()), _rng_state()
    assert recipe.check()["ok"]
    equal(before, recipe.model.state_dict())
    equal(rng, _rng_state())
    predictor = recipe.predictor()
    predictor.model.register_buffer("observed_calls", torch.zeros(()))
    graph = predictor.model.program.graph
    graph_state = copy.deepcopy(graph.state())
    before, rng = copy.deepcopy(predictor.model.state_dict()), _rng_state()

    def observe(model, _args, _output):
        model.observed_calls.add_(1)
        torch.rand(1)
        random.random()

    handle = predictor.model.register_forward_hook(observe)
    try:
        predictor.predict_records(rows[-3:], batch_size=2)
    finally:
        handle.remove()
    equal(before, predictor.model.state_dict())
    equal(rng, _rng_state())
    restored = graph.state()
    assert graph_state.join_cursors == restored.join_cursors
    for before, after in zip(graph_state.resources, restored.resources, strict=True):
        equal(before.default_view.value, after.default_view.value)
        assert (before.epoch, before.step_index) == (after.epoch, after.step_index)
    recipe.train_data = iter(recipe.train_data)
    with pytest.raises(ValueError, match="explicit batch"):
        recipe.check()


def test_prediction_id_alignment_and_failed_evaluation_restore_state(tmp_path):
    cfg, rows = setup(tmp_path / "data")
    recipe = arti.TaskRecipe.from_config(cfg)
    predictor = recipe.predictor()
    predictor._predict_fn = lambda _model, records, **_kwargs: [{"id": "wrong"} for _row in records]
    with pytest.raises(ValueError, match="preserve input record ids"):
        predictor.predict_records(rows[-3:])
    state, rng = copy.deepcopy(predictor.model.state_dict()), torch.get_rng_state().clone()

    def fail(model, _rows):
        model.feature_mean.add_(1)
        torch.rand(1)
        raise RuntimeError("evaluation")

    predictor._evaluate_fn = fail
    with pytest.raises(RuntimeError, match="evaluation"):
        predictor.evaluate_records(rows[-3:])
    equal(state, predictor.model.state_dict())
    assert torch.equal(rng, torch.get_rng_state())


def test_native_tensor_prediction_and_explicit_dataset_identity_do_not_coerce_inputs(tmp_path):
    cfg, _rows = setup(tmp_path)
    recipe = arti.TaskRecipe.from_config(cfg)
    seen = []

    def predict(model, records, *, batch_size=7):
        seen.append((records, batch_size))
        return model(records)

    def evaluate(model, records):
        return {"norm": float(model(records).square().mean())}

    predictor = arti.TaskPredictor(
        recipe.model, recipe.task_contract, predict_fn=predict, evaluate_fn=evaluate
    )
    inputs = torch.ones(3, 4)
    outputs = predictor.predict_records(inputs)
    assert seen[0][0] is inputs and seen[0][1] == 7
    assert len(outputs) == 3 and all(isinstance(value, torch.Tensor) for value in outputs)
    from arti.recipe_data import json_fingerprint
    identity = json_fingerprint({"value": 1, "shape": [3, 4], "dtype": "float32"})
    report = predictor.evaluate_records(inputs, dataset_manifest={"sha256": identity})
    assert report["dataset_sha256"] == identity and report["norm"] >= 0
