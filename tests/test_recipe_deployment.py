from __future__ import annotations

import json
import asyncio
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest
import torch

import arti
from arti.cli import main

pytest.importorskip("transformers")
from test_language_recipe import base_model, config, tokenizer


def trained(tmp_path, *, sharded=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    base = tmp_path / "native-base"
    base_model().save_pretrained(base, max_shard_size="5KB" if sharded else "5GB")
    cfg = config(tmp_path)
    current = arti.CausalLMRecipe(cfg, model=base_model(), tokenizer=tokenizer())
    before = [parameter.detach().clone() for parameter in current.model.arti.parameters()]
    result = current.fit(steps=2, run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "run"))
    assert any(
        not torch.equal(a, b) for a, b in zip(before, current.model.arti.parameters(), strict=True)
    )
    return current, Path(result["model"]).parent, base, result


def requests():
    return [{"id": "query", "messages": [{"role": "user", "content": "color red"}]}]


@pytest.mark.parametrize("sharded", [False, True])
def test_native_package_preserves_logits_replies_and_relative_sources(tmp_path, sharded):
    current, asset, base, _result = trained(tmp_path / "work", sharded=sharded)
    predictor = arti.CausalLMPredictor.from_pretrained(
        asset, base_model=base, local_files_only=True
    )
    expected = predictor.predict_records(requests())
    state = [p.detach().clone() for p in current.model.arti.parameters()]
    result = arti.package_causal_lm_deployment(
        asset, tmp_path / "package", include_base=True, base_model=base
    )
    moved = tmp_path / "moved"
    shutil.copytree(result["directory"], moved)
    restored = arti.CausalLMPredictor.from_pretrained(moved, local_files_only=True)
    assert restored.predict_records(requests()) == expected
    batch = current.data.splits["test"].batches(2)[0]
    with torch.no_grad():
        torch.testing.assert_close(
            restored.model(**batch).logits, current.model(**batch).logits, rtol=0, atol=0
        )
    assert restored.contract == current.task_contract
    assert all(
        torch.equal(a, b) for a, b in zip(state, current.model.arti.parameters(), strict=True)
    )
    deployment = json.loads((moved / "deployment.json").read_text())
    assert (
        deployment["base_model_directory"] == "base_model" and deployment["base_weights_included"]
    )
    assert str(tmp_path) not in json.dumps(deployment)
    inspected = arti.inspect_causal_lm_deployment(moved)
    assert inspected["files_available"] and inspected["base_source"] == "bundled"
    assert inspected["runtime_reload_verified"] is False


def test_inspection_checks_files_without_host_or_tensor_loading(tmp_path, monkeypatch):
    import arti.language_recipe as language
    import arti.serialization as codec

    _current, asset, base, _result = trained(tmp_path / "work")

    def fail(*_args, **_kwargs):
        raise AssertionError("no model/optimizer/tensor payload construction")

    monkeypatch.setattr(language, "_load_model", fail)
    monkeypatch.setattr(torch.optim, "AdamW", fail)
    monkeypatch.setattr(codec, "_load_safetensors", fail)
    report = arti.inspect_causal_lm_deployment(asset, base_model=base)
    assert report["files_available"] and not report["loads_base_weight_tensors"]
    assert report["base_bytes"] > 0
    assert arti.torch.inspect_causal_lm_deployment is arti.inspect_causal_lm_deployment


def test_package_copies_declared_closure_not_training_residue(tmp_path):
    _current, asset, base, _result = trained(tmp_path / "work")
    (asset / "training.jsonl").write_text("private training data")
    (asset / "model.arti.checkpoint.json").write_text("unused checkpoint residue")
    (asset / "tokenizer" / "unrelated.json").write_text("private unrelated metadata")
    (base / "unrelated.txt").write_text("not needed for native loading")
    result = arti.package_causal_lm_deployment(
        asset, tmp_path / "package", include_base=True, base_model=base
    )
    directory = Path(result["directory"])
    assert not (directory / "training.jsonl").exists()
    assert not (directory / "model.arti.checkpoint.json").exists()
    assert not (directory / "tokenizer" / "unrelated.json").exists()
    assert not (directory / "base_model" / "unrelated.txt").exists()


def test_missing_native_dependency_has_no_partial_published_package(tmp_path):
    _current, asset, base, _result = trained(tmp_path / "work")
    (base / "model.safetensors").unlink()
    report = arti.inspect_causal_lm_deployment(asset, base_model=base)
    assert (
        not report["files_available"]
        and "base_model/model.safetensors" in report["missing_dependencies"]
    )
    with pytest.raises(ValueError, match="complete native SafeTensors"):
        arti.package_causal_lm_deployment(
            asset, tmp_path / "package", include_base=True, base_model=base
        )
    assert not (tmp_path / "package").exists()


def test_checkpoint_and_existing_destination_are_not_overwritten(tmp_path):
    _current, asset, base, result = trained(tmp_path / "work")
    with pytest.raises(ValueError, match="inference asset"):
        arti.package_causal_lm_deployment(
            result["checkpoint"], tmp_path / "invalid", include_base=True, base_model=base
        )
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "keep").write_text("keep")
    with pytest.raises(FileExistsError):
        arti.package_causal_lm_deployment(asset, destination, include_base=True, base_model=base)
    assert (destination / "keep").read_text() == "keep"


def test_moved_package_cold_process_needs_no_training_directory_or_cache(tmp_path):
    current, asset, base, _result = trained(tmp_path / "work", sharded=True)
    expected = arti.CausalLMPredictor.from_pretrained(
        asset, base_model=base, local_files_only=True
    ).predict_records(requests())
    arti.package_causal_lm_deployment(
        asset, tmp_path / "work" / "package", include_base=True, base_model=base
    )
    moved = tmp_path / "deployment"
    shutil.copytree(tmp_path / "work" / "package", moved)
    root = tmp_path.resolve()
    assert (tmp_path / "work").resolve().is_relative_to(root) and (
        tmp_path / "retired"
    ).resolve().is_relative_to(root)
    (tmp_path / "work").rename(tmp_path / "retired")
    env = dict(
        os.environ,
        PYTHONPATH=str(Path(arti.__file__).resolve().parent.parent),
        HF_HOME=str(tmp_path / "empty-cache"),
        HF_HUB_OFFLINE="1",
        HF_DATASETS_OFFLINE="1",
        CUDA_VISIBLE_DEVICES="-1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    code = "import json,sys; import arti; predictor=arti.CausalLMPredictor.from_pretrained(sys.argv[1],local_files_only=True); rows=json.loads(sys.argv[2]); print(json.dumps({'predictions':predictor.predict_records(rows),'task':predictor.contract['sha256'],'base':str(predictor.model.config._name_or_path)}))"
    result = subprocess.run(
        [sys.executable, "-c", code, str(moved), json.dumps(requests())],
        env=env,
        cwd=moved,
        text=True,
        capture_output=True,
        timeout=90,
        check=True,
    )
    cold = json.loads(result.stdout)
    assert cold["predictions"] == expected and cold["task"] == current.task_fingerprint
    assert Path(cold["base"]).resolve() == moved / "base_model"


def test_cli_inspection_package_prediction_and_evaluation(tmp_path, capsys):
    current, asset, base, _result = trained(tmp_path / "work")
    assert main(["recipe", "inspect-deployment", str(asset), "--base-model", str(base)]) == 0
    assert json.loads(capsys.readouterr().out)["files_available"]
    assert (
        main(
            [
                "recipe",
                "package-deployment",
                str(asset),
                "--base-model",
                str(base),
                "--include-base",
                "--output",
                str(tmp_path / "package"),
            ]
        )
        == 0
    )
    package = json.loads(capsys.readouterr().out)
    inputs = tmp_path / "requests.jsonl"
    inputs.write_text("\n".join(json.dumps(row) for row in requests()) + "\n")
    assert (
        main(
            [
                "recipe",
                "predict",
                package["directory"],
                "--data",
                str(inputs),
                "--output",
                str(tmp_path / "replies.jsonl"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["samples"] == 1
    assert (
        main(["recipe", "evaluate", package["directory"], "--data", str(current.config.data_path)])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["supervised_tokens"] > 0
    cfg = tmp_path / "recipe.json"
    cfg.write_text(json.dumps(current.config.to_dict()))
    assert (
        main(["recipe", "evaluate", str(cfg), "--model", package["artifact"], "--split", "test"])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["samples"] == 2


def test_slim_package_and_explicit_base_override_stay_supported(tmp_path):
    _current, asset, base, _result = trained(tmp_path / "work")
    result = arti.package_causal_lm_deployment(asset, tmp_path / "slim", base_model=base)
    assert not result["base_weights_included"] and not (tmp_path / "slim" / "base_model").exists()
    before = arti.CausalLMPredictor.from_pretrained(asset, base_model=base, local_files_only=True)
    after = arti.CausalLMPredictor.from_pretrained(
        result["directory"], base_model=base, local_files_only=True
    )
    assert after.predict_records(requests()) == before.predict_records(requests())


def test_pinned_existing_cache_resolution_does_not_download(tmp_path, monkeypatch):
    import huggingface_hub

    current, asset, base, _result = trained(tmp_path / "work")

    def cached(repo, filename, *, revision):
        assert repo == current.config.model.id and filename == "config.json"
        assert revision == current.config.model.revision
        return str(base / filename)

    monkeypatch.setattr(huggingface_hub, "try_to_load_from_cache", cached)
    report = arti.inspect_causal_lm_deployment(asset)
    assert report["base_source"] == "hub-cache" and report["files_available"]
    result = arti.package_causal_lm_deployment(asset, tmp_path / "package", include_base=True)
    assert (
        arti.CausalLMPredictor.from_pretrained(result["directory"], local_files_only=True).contract
        == current.task_contract
    )


def test_custom_factory_requirement_survives_packaging(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    base = tmp_path / "base"
    base_model().save_pretrained(base)
    factory = arti.formula_adapter_factory(arti.FormulaAdapterConfig(rank=3, activation="gelu"))
    current = arti.CausalLMRecipe(
        config(tmp_path), model=base_model(), tokenizer=tokenizer(), layer_factory=factory
    )
    asset = current.export(tmp_path / "asset")
    result = arti.package_causal_lm_deployment(
        asset, tmp_path / "package", include_base=True, base_model=base
    )
    assert arti.inspect_causal_lm_deployment(result["directory"])["requires_layer_factory"]
    with pytest.raises(ValueError, match="caller-supplied layer_factory"):
        arti.CausalLMPredictor.from_pretrained(result["directory"], local_files_only=True)
    loaded = arti.CausalLMPredictor.from_pretrained(
        result["directory"], local_files_only=True, layer_factory=factory
    )
    assert loaded.contract == current.task_contract


def test_loaded_offline_package_serves_native_asgi_requests(tmp_path):
    httpx = pytest.importorskip("httpx")
    pytest.importorskip("fastapi")
    _current, asset, base, _result = trained(tmp_path / "work")
    result = arti.package_causal_lm_deployment(
        asset, tmp_path / "package", include_base=True, base_model=base
    )
    predictor = arti.load_recipe_predictor(result["directory"])
    expected = predictor.predict_records(requests())
    app = arti.create_recipe_app(predictor)

    async def exercise():
        async with app.router.lifespan_context(app):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post("/predict", json={"records": requests()})
                assert response.status_code == 200 and response.json()["predictions"] == expected
                health = await client.get("/health")
                assert health.status_code == 200 and health.json()["ready"]

    asyncio.run(exercise())
