from __future__ import annotations

import asyncio
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch

import arti
from arti.cli import main


REFERENCE = "examples.qwen_program_factory:build_layer"


def factory(width=4):
    return arti.LayerFactoryConfig(REFERENCE, {"width": width})


def current(directory, *, steps=4, descriptor=None):
    pytest.importorskip("transformers")
    from test_language_recipe import base_model, config, tokenizer

    directory.mkdir(parents=True, exist_ok=True)
    cfg = config(directory, layer_factory=descriptor or factory())
    cfg = replace(cfg, training=replace(cfg.training, steps=steps))
    return arti.CausalLMRecipe(cfg, model=base_model(), tokenizer=tokenizer())


def requests():
    return [{"id": "query", "messages": [{"role": "user", "content": "color red"}]}]


def test_descriptor_resolves_actual_resource_program_and_public_aliases():
    descriptor = factory()
    layer = descriptor.resolve()(SimpleNamespace(dim=16))
    x = torch.randn(2, 3, 16)
    assert torch.equal(layer(x), x)
    assert descriptor.to_dict() == {"reference": REFERENCE, "kwargs": {"width": 4}}
    assert arti.torch.LayerFactoryConfig is arti.LayerFactoryConfig
    assert len(layer.graph.nodes) == 2


@pytest.mark.parametrize("reference", ["module", "module:", ":factory", "path/file.py:factory"])
def test_descriptor_requires_a_module_reference(reference):
    with pytest.raises(ValueError, match="module:callable"):
        arti.LayerFactoryConfig(reference)


def test_missing_dependency_is_reported_before_model_or_tokenizer_allocation(tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    import arti.language_recipe as module
    from test_language_recipe import config

    def fail(*_args, **_kwargs):
        raise AssertionError("must not allocate host/tokenizer for missing factory")

    monkeypatch.setattr(module, "_load_model", fail)
    monkeypatch.setattr(module, "_load_tokenizer", fail)
    cfg = config(tmp_path, layer_factory=arti.LayerFactoryConfig("missing_recipe_package:build"))
    with pytest.raises(ValueError, match="cannot import layer factory"):
        arti.CausalLMRecipe(cfg)


def test_configuration_roundtrip_identity_and_default_scope(tmp_path):
    recipe = current(tmp_path)
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe.config.to_dict()))
    cfg = arti.CausalLMRecipeConfig.read(path)
    assert cfg == recipe.config
    assert recipe.task_contract["layer_factory"] == factory().to_dict()
    assert recipe.task_contract["adapter_factory"] == "caller-supplied"
    assert recipe.plan()["deployment"]["state_scope"] == "caller-managed"
    assert current(tmp_path, descriptor=factory(5)).task_fingerprint != recipe.task_fingerprint


def test_meta_plan_does_not_execute_factory_or_guess_parameter_count(tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    from transformers import AutoConfig
    from test_language_recipe import base_model, tokenizer
    import arti.language_recipe as module

    recipe = current(tmp_path)
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe.config.to_dict()))
    native = base_model().config
    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *_args, **_kwargs: native)
    monkeypatch.setattr(module, "_load_tokenizer", lambda *_args, **_kwargs: tokenizer())

    def fail(*_args, **_kwargs):
        raise AssertionError("meta plan must not execute installed factory")

    monkeypatch.setattr(arti.LayerFactoryConfig, "resolve", fail)
    report = module.plan_causal_lm_recipe(path)
    assert report["weights_loaded"] is False and report["model_allocation"] == "meta"
    assert report["model"]["trainable_parameters"] is None
    assert report["task_recipe"]["layer_factory"] == factory().to_dict()


def test_custom_factory_real_training_and_exact_continuation(tmp_path):
    from test_recipe_recovery import equal, interrupt_after, policy
    from test_language_recipe import base_model, tokenizer

    full = current(tmp_path / "data")
    split = current(tmp_path / "data")
    before = [p.detach().clone() for p in full.model.arti.parameters()]
    frozen = {
        name: parameter.detach().clone()
        for name, parameter in full.model.named_parameters()
        if not parameter.requires_grad
    }
    full.fit(run=policy(tmp_path / "full"))
    assert any(
        not torch.equal(a, b) for a, b in zip(before, full.model.arti.parameters(), strict=True)
    )
    assert all(
        torch.equal(value, dict(full.model.named_parameters())[name])
        for name, value in frozen.items()
    )
    with pytest.raises(KeyboardInterrupt):
        split.fit(run=policy(tmp_path / "split"), on_checkpoint=interrupt_after(2))
    cfg = arti.CausalLMRecipeConfig.read(tmp_path / "split" / "resolved_recipe.json")
    resumed = arti.CausalLMRecipe(cfg, model=base_model(), tokenizer=tokenizer())
    result = resumed.resume_run(tmp_path / "split", run=policy(tmp_path / "continued"))
    assert result["optimizer_updates_this_attempt"] == 2
    equal(full.model.arti._bundle().state_dict(), resumed.model.arti._bundle().state_dict())
    equal(full.session.optimizer.state_dict(), resumed.session.optimizer.state_dict())
    assert full.session.loss_history == resumed.session.loss_history
    assert full.train_data.state_dict() == resumed.train_data.state_dict()


def packaged(tmp_path):
    from test_language_recipe import base_model

    native = tmp_path / "native"
    base_model().save_pretrained(native)
    recipe = current(tmp_path / "data", steps=2)
    result = recipe.fit(run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "run"))
    package = arti.package_causal_lm_deployment(
        result["model"], tmp_path / "package", base_model=native, include_base=True
    )
    return recipe, package


def test_package_describes_factory_without_executing_it_and_reloads_exactly(tmp_path, monkeypatch):
    recipe, package = packaged(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(
            arti.LayerFactoryConfig,
            "resolve",
            lambda *_args: (_ for _ in ()).throw(AssertionError("inspection does not import")),
        )
        report = arti.inspect_causal_lm_deployment(package["directory"])
    assert report["requires_layer_factory"] and report["layer_factory"] == factory().to_dict()
    with pytest.raises(ValueError, match="required: examples.qwen_program_factory"):
        arti.CausalLMPredictor.from_pretrained(package["directory"])
    moved = tmp_path / "moved"
    shutil.copytree(package["directory"], moved)
    loaded = arti.CausalLMPredictor.from_pretrained(
        moved, layer_factory=REFERENCE, local_files_only=True
    )
    assert loaded.serving_state_scope == "caller-managed"
    batch = recipe.data.splits["test"].batches(2)[0]
    with torch.no_grad():
        torch.testing.assert_close(
            loaded.model(**batch).logits, recipe.model(**batch).logits, rtol=0, atol=0
        )
    assert loaded.contract == recipe.task_contract
    with pytest.raises(ValueError, match="reference/kwargs differ"):
        arti.CausalLMPredictor.from_pretrained(package["directory"], layer_factory=factory(6))


def test_cli_native_train_evaluate_predict_and_resume_use_configured_factory(
    tmp_path, monkeypatch, capsys
):
    import arti.language_recipe as module
    from test_language_recipe import base_model, tokenizer

    recipe = current(tmp_path / "data", steps=2)
    cfg = tmp_path / "recipe.json"
    cfg.write_text(json.dumps(recipe.config.to_dict()))
    monkeypatch.setattr(module, "_load_model", lambda *_args, **_kwargs: base_model())
    monkeypatch.setattr(module, "_load_tokenizer", lambda *_args, **_kwargs: tokenizer())
    run = tmp_path / "cli-run"
    assert main(["recipe", "train", str(cfg), "--run-dir", str(run)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["optimizer_steps"] == 2
    assert (
        json.loads((run / "resolved_recipe.json").read_text())["layer_factory"]
        == factory().to_dict()
    )
    assert main(["recipe", "resume-run", str(run)]) == 0
    assert json.loads(capsys.readouterr().out)["optimizer_updates_this_attempt"] == 0
    assert (
        main(
            [
                "recipe",
                "evaluate",
                result["model"],
                "--data",
                str(recipe.config.data_path),
                "--layer-factory",
                REFERENCE,
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["supervised_tokens"] > 0
    inputs = tmp_path / "requests.jsonl"
    inputs.write_text(json.dumps(requests()[0]) + "\n")
    assert (
        main(
            [
                "recipe",
                "predict",
                str(Path(result["model"]).parent),
                "--layer-factory",
                REFERENCE,
                "--data",
                str(inputs),
                "--output",
                str(tmp_path / "replies.jsonl"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["samples"] == 1


def test_serve_loader_and_asgi_keep_custom_state_scope(tmp_path, monkeypatch):
    httpx = pytest.importorskip("httpx")
    pytest.importorskip("fastapi")
    import uvicorn
    from arti.recipe_serving import serve_recipe

    _recipe, package = packaged(tmp_path)
    apps = []
    monkeypatch.setattr(uvicorn, "run", lambda app, **_kwargs: apps.append(app))
    serve_recipe(package["directory"], layer_factory=REFERENCE)
    app = apps[0]

    async def exercise():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post("/predict", json={"records": requests()})
                assert response.status_code == 200
                assert len(response.json()["predictions"]) == 1

    asyncio.run(exercise())


def test_factory_dependency_is_explicit_in_another_process(tmp_path):
    _recipe, package = packaged(tmp_path)
    root = Path(__file__).resolve().parents[1]
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join((str(Path(arti.__file__).resolve().parent.parent), str(root))),
        HF_HOME=str(tmp_path / "empty-cache"),
        HF_HUB_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="-1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    code = "import arti,json,sys; model=arti.CausalLMPredictor.from_pretrained(sys.argv[1],layer_factory=sys.argv[2],local_files_only=True); print(json.dumps({'factory':model.contract['layer_factory'],'reply':model.predict_records(json.loads(sys.argv[3]))}))"
    child = subprocess.run(
        [sys.executable, "-c", code, package["directory"], REFERENCE, json.dumps(requests())],
        env=env,
        cwd=package["directory"],
        capture_output=True,
        text=True,
        timeout=90,
        check=True,
    )
    result = json.loads(child.stdout)
    assert result["factory"] == factory().to_dict() and len(result["reply"]) == 1


def test_custom_factory_selected_export_matches_best_without_rewinding_session(tmp_path):
    recipe = current(tmp_path / "data", steps=3)
    run = arti.ARTITrainingRunConfig(
        output_dir=tmp_path / "selected",
        evaluation_steps=1,
        validation=arti.ARTIValidationConfig(monitor="fixture_score"),
    )
    result = recipe.fit(
        run=run, validation_fn=lambda session: {"fixture_score": float(session.global_step)}
    )
    assert result["selection"]["selected_step"] == 1
    saved = {
        key: value.detach().clone()
        for key, value in recipe.model.arti._bundle().state_dict().items()
    }
    from test_language_recipe import base_model

    loaded = arti.CausalLMPredictor.from_pretrained(
        result["model"], model=base_model(), layer_factory=REFERENCE
    )
    batch = recipe.data.splits["validation"].batches(1)[0]
    with recipe.session.checkpoint_context(result["selected_checkpoint"]), torch.no_grad():
        expected = recipe.model(**batch, use_cache=False).logits
    with torch.no_grad():
        torch.testing.assert_close(
            loaded.model(**batch, use_cache=False).logits, expected, rtol=0, atol=0
        )
    for key, value in recipe.model.arti._bundle().state_dict().items():
        torch.testing.assert_close(value, saved[key], rtol=0, atol=0)
    assert recipe.session.global_step == 3


def test_cli_explicit_factory_kwargs_and_serve_are_forwarded(tmp_path, monkeypatch, capsys):
    from arti import recipe_serving

    recipe, package = packaged(tmp_path)
    kwargs = tmp_path / "kwargs.json"
    kwargs.write_text(json.dumps({"width": 4}))
    data = tmp_path / "requests.jsonl"
    data.write_text(json.dumps(requests()[0]) + "\n")
    assert (
        main(
            [
                "recipe",
                "predict",
                package["directory"],
                "--layer-factory",
                REFERENCE,
                "--factory-kwargs-json",
                str(kwargs),
                "--data",
                str(data),
                "--output",
                str(tmp_path / "replies.jsonl"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["samples"] == 1
    calls = []
    monkeypatch.setattr(
        recipe_serving, "serve_recipe", lambda *args, **values: calls.append(values)
    )
    assert (
        main(
            [
                "recipe",
                "serve",
                package["directory"],
                "--layer-factory",
                REFERENCE,
                "--factory-kwargs-json",
                str(kwargs),
            ]
        )
        == 0
    )
    assert calls[0]["layer_factory"] == factory()
    assert recipe.task_contract["layer_factory"] == factory().to_dict()


def test_cli_interrupted_run_restores_factory_and_absolute_horizon(tmp_path, monkeypatch, capsys):
    from test_recipe_recovery import interrupt_after, policy
    from test_language_recipe import base_model, tokenizer
    import arti.language_recipe as module

    recipe = current(tmp_path / "data", steps=4)
    with pytest.raises(KeyboardInterrupt):
        recipe.fit(run=policy(tmp_path / "source"), on_checkpoint=interrupt_after(2))
    monkeypatch.setattr(module, "_load_model", lambda *_args, **_kwargs: base_model())
    monkeypatch.setattr(module, "_load_tokenizer", lambda *_args, **_kwargs: tokenizer())
    assert (
        main(
            [
                "recipe",
                "resume-run",
                str(tmp_path / "source"),
                "--run-dir",
                str(tmp_path / "restored"),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["optimizer_steps"] == 4 and result["optimizer_updates_this_attempt"] == 2
    assert (
        json.loads((tmp_path / "restored" / "resolved_recipe.json").read_text())["layer_factory"]
        == factory().to_dict()
    )


def test_explicit_python_factory_does_not_inherit_builtin_auto_batching(tmp_path):
    pytest.importorskip("transformers")
    from test_language_recipe import base_model, config, tokenizer

    cfg = config(tmp_path)
    recipe = arti.CausalLMRecipe(cfg, model=base_model(), tokenizer=tokenizer())
    asset = recipe.export(tmp_path / "asset")
    loaded = arti.CausalLMPredictor.from_pretrained(
        asset, model=base_model(), layer_factory=arti.formula_adapter_factory(cfg.adapter)
    )
    assert loaded.contract["adapter_factory"] == "builtin-formula"
    assert loaded.serving_state_scope == "caller-managed"
