from __future__ import annotations

from dataclasses import replace
import asyncio
import json

import pytest
import torch

import arti
from test_language_recipe import base_model, config, tokenizer


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_real_causal_predictor_serves_batched_chat_from_one_loaded_asset(tmp_path):
    from httpx import ASGITransport, AsyncClient

    current = arti.CausalLMRecipe(config(tmp_path), model=base_model(), tokenizer=tokenizer())
    result = current.fit(run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "run"))
    predictor = arti.CausalLMPredictor.from_pretrained(result["model"], model=base_model())
    records = [
        {"id": "a", "messages": [{"role": "user", "content": "color red"}]},
        {"id": "b", "messages": [{"role": "user", "content": "color blue"}]},
    ]
    expected = predictor.predict_records(records)
    app = arti.create_recipe_app(predictor)
    assert app.state.predictor is predictor

    async def exercise():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            health = await client.get("/health")
            assert health.json()["task"] == "causal-lm"
            first = await client.post("/predict", json={"records": records})
            assert first.status_code == 200 and first.json()["predictions"] == expected
            second = await client.post("/predict", json={"records": records, "batch_size": 2})
            assert second.json()["predictions"] == expected
            chat = await client.post("/chat", json={"messages": records[0]["messages"]})
            assert chat.status_code == 200
            assert chat.json()["text"] == expected[0]["text"]
            invalid = await client.post("/chat", json={"messages": [], "max_new_tokens": 0})
            assert invalid.status_code == 422
            schema = (await client.get("/openapi.json")).json()
            assert {"/predict", "/chat", "/health"} <= schema["paths"].keys()
            assert str(tmp_path) not in json.dumps(health.json())

    asyncio.run(exercise())


def test_classification_and_language_share_the_same_asgi_service(tmp_path):
    from httpx import ASGITransport, AsyncClient
    from test_task_recipes import _config

    classifier = arti.ClassificationRecipe(_config(tmp_path))
    predictor = arti.ClassificationPredictor.from_pretrained(classifier.export(tmp_path / "model"))
    app = arti.create_recipe_app(predictor)

    async def exercise():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post("/predict", json={"records": [{"features": [1, 0.5, 3]}]})
            assert response.status_code == 200
            assert response.json()["labels"] == list(predictor.labels)
            chat = await client.post(
                "/chat", json={"messages": [{"role": "user", "content": "hello"}]}
            )
            assert chat.status_code == 422

    asyncio.run(exercise())


def test_workflow_export_failure_does_not_report_deployment_success(tmp_path, monkeypatch):
    current = arti.CausalLMRecipe(config(tmp_path), model=base_model(), tokenizer=tokenizer())

    def failed_export(directory, **kwargs):
        raise RuntimeError("export stopped")

    monkeypatch.setattr(current, "export", failed_export)
    with pytest.raises(RuntimeError, match="export stopped"):
        current.fit(steps=1, run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "failed"))
    result = json.loads((tmp_path / "failed" / "task_result.json").read_text())
    assert not result["ok"] and result["status"] == "failed" and result["stage"] == "export"
    assert (tmp_path / "failed" / "checkpoints" / "step-00000001" / "model.arti.st").is_file()


def test_sft_recipe_does_not_require_serving_dependencies_for_training(tmp_path):
    cfg = config(tmp_path)
    resolved = replace(cfg, run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "normal"))
    assert arti.CausalLMRecipe(resolved, model=base_model(), tokenizer=tokenizer()).check()["ok"]


def test_concurrent_qwen_requests_share_one_real_generation_batch(tmp_path, monkeypatch):
    from httpx import ASGITransport, AsyncClient

    current = arti.CausalLMRecipe(config(tmp_path), model=base_model(), tokenizer=tokenizer())
    result = current.fit(run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "run"))
    predictor = arti.CausalLMPredictor.from_pretrained(result["model"], model=base_model())
    records = [{"id":"a", "messages":[{"role":"user","content":"color red"}]}, {"id":"b", "messages":[{"role":"user","content":"color blue"}]}]
    expected = predictor.predict_records(records)
    calls = []
    original = predictor.model.generate

    def generate(*args,**kwargs):
        calls.append(tuple(kwargs["input_ids"].shape))
        return original(*args,**kwargs)

    monkeypatch.setattr(predictor.model,"generate",generate)
    app = arti.create_recipe_app(predictor,config=arti.RecipeServingConfig(batch_wait_ms=20))

    async def exercise():
        async with app.router.lifespan_context(app):
            async with AsyncClient(transport=ASGITransport(app=app),base_url="http://test") as client:
                responses = await asyncio.gather(*(client.post("/predict",json={"records":[record]}) for record in records))
                assert [response.json()["predictions"][0] for response in responses] == expected
                assert len(calls) == 1 and calls[0][0] == 2

    asyncio.run(exercise())


def test_concurrent_classifier_requests_share_one_tensor_forward(tmp_path):
    from httpx import ASGITransport,AsyncClient
    from test_task_recipes import _config

    recipe = arti.ClassificationRecipe(_config(tmp_path))
    predictor = arti.ClassificationPredictor.from_pretrained(recipe.export(tmp_path / "model"))
    records = [{"id":str(value),"features":[value,value*0.5,3]} for value in (1,-1,2)]
    expected = predictor.predict_records(records)
    shapes = []
    hook = predictor.model.register_forward_pre_hook(lambda _module,args,kwargs: shapes.append(tuple((args[0] if args else kwargs["features"]).shape)),with_kwargs=True)
    app = arti.create_recipe_app(predictor,config=arti.RecipeServingConfig(batch_wait_ms=20))

    async def exercise():
        async with app.router.lifespan_context(app):
            async with AsyncClient(transport=ASGITransport(app=app),base_url="http://test") as client:
                responses = await asyncio.gather(*(client.post("/predict",json={"records":[row]}) for row in records))
                assert all(response.status_code == 200 for response in responses), [response.text for response in responses]
                actual = [response.json()["predictions"][0] for response in responses]
                assert [row["id"] for row in actual] == [row["id"] for row in expected]
                assert [row["label"] for row in actual] == [row["label"] for row in expected]
                torch.testing.assert_close(torch.tensor([row["probabilities"] for row in actual]),torch.tensor([row["probabilities"] for row in expected]),rtol=1e-6,atol=1e-7)
                assert shapes == [(3,3)]

    try:
        asyncio.run(exercise())
    finally:
        hook.remove()
