from __future__ import annotations

import asyncio
from dataclasses import asdict
import threading

import pytest

import arti
from arti.recipe_batching import RecipeServingConfig, _BatchDispatcher, _ServingUnavailable


class Predictor:
    contract = {"task": "classification", "sha256": "fixture"}
    labels = ("answer",)
    serving_state_scope = "read-only"

    def __init__(self):
        self.calls = []

    def validate_records(self, records):
        for row in records:
            if "value" not in row:
                raise ValueError("value is required")

    def predict_records(self, records, **kwargs):
        self.calls.append(([dict(row) for row in records], dict(kwargs)))
        return [{"id": row.get("id"), "value": row["value"] * 2} for row in records]


def test_config_is_shared_and_serializable():
    assert arti.RecipeServingConfig is arti.torch.RecipeServingConfig
    assert RecipeServingConfig(**asdict(RecipeServingConfig())) == RecipeServingConfig()


@pytest.mark.parametrize(
    "options",
    [
        {"batching": "maybe"},
        {"max_batch_size": 0},
        {"max_batch_size": True},
        {"max_pending_records": 1.5},
        {"batch_wait_ms": -1},
        {"batch_wait_ms": float("nan")},
    ],
)
def test_invalid_config(options):
    with pytest.raises(ValueError):
        RecipeServingConfig(**options)


def test_multiple_requests_share_a_real_predictor_call_and_keep_order():
    predictor = Predictor()

    async def exercise():
        worker = _BatchDispatcher(
            predictor.predict_records,
            RecipeServingConfig(max_batch_size=8, batch_wait_ms=20),
            dynamic=True,
        )
        try:
            requests = [
                [{"id": "same", "value": value}, {"id": str(value), "value": value + 1}]
                for value in (1, 3, 5)
            ]
            actual = await asyncio.gather(*(worker.submit(rows, batch_size=8) for rows in requests))
            assert actual[0] == [{"id": "same", "value": 2}, {"id": "1", "value": 4}]
            assert actual[2] == [{"id": "same", "value": 10}, {"id": "5", "value": 12}]
            assert len(predictor.calls) == 1 and len(predictor.calls[0][0]) == 6
            assert worker.snapshot()["pending_records"] == 0
        finally:
            await worker.close()
        assert worker._task.done()

    asyncio.run(exercise())


def test_large_request_is_split_by_physical_dispatch_limit():
    predictor = Predictor()

    async def exercise():
        worker = _BatchDispatcher(
            predictor.predict_records,
            RecipeServingConfig(max_batch_size=3, batch_wait_ms=0),
            dynamic=True,
        )
        try:
            actual = await worker.submit([{"value": value} for value in range(8)], batch_size=10)
            assert [row["value"] for row in actual] == [2 * value for value in range(8)]
            assert [len(rows) for rows, _ in predictor.calls] == [3, 3, 2]
        finally:
            await worker.close()

    asyncio.run(exercise())


def test_incompatible_generation_limits_remain_separate():
    predictor = Predictor()

    async def exercise():
        worker = _BatchDispatcher(
            predictor.predict_records, RecipeServingConfig(batch_wait_ms=10), dynamic=True
        )
        try:
            await asyncio.gather(
                worker.submit([{"value": 1}], batch_size=8, max_new_tokens=4),
                worker.submit([{"value": 2}], batch_size=8, max_new_tokens=8),
            )
            assert len(predictor.calls) == 2
            assert {options["max_new_tokens"] for _, options in predictor.calls} == {4, 8}
        finally:
            await worker.close()

    asyncio.run(exercise())


def test_cancellation_before_dispatch_does_not_poison_other_requests():
    predictor = Predictor()

    async def exercise():
        worker = _BatchDispatcher(
            predictor.predict_records, RecipeServingConfig(batch_wait_ms=25), dynamic=True
        )
        task = asyncio.create_task(worker.submit([{"value": 1}]))
        await asyncio.sleep(0)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        try:
            assert await worker.submit([{"value": 2}]) == [{"id": None, "value": 4}]
            assert len(predictor.calls) == 1 and predictor.calls[0][0] == [{"value": 2}]
            assert worker.pending_records == 0
        finally:
            await worker.close()

    asyncio.run(exercise())


def test_queue_capacity_and_close_reject_pending_without_model_work():
    predictor = Predictor()

    async def exercise():
        worker = _BatchDispatcher(
            predictor.predict_records,
            RecipeServingConfig(max_pending_records=1, batch_wait_ms=100),
            dynamic=True,
        )
        first = asyncio.create_task(worker.submit([{"value": 1}]))
        await asyncio.sleep(0)
        with pytest.raises(_ServingUnavailable, match="full"):
            await worker.submit([{"value": 2}])
        await worker.close()
        with pytest.raises(_ServingUnavailable, match="closing"):
            await first
        assert not predictor.calls and worker.pending_records == 0
        with pytest.raises(_ServingUnavailable, match="closing"):
            await worker.submit([{"value": 3}])

    asyncio.run(exercise())


def test_close_waits_for_inflight_model_and_preserves_result():
    entered, release = threading.Event(), threading.Event()

    def predict(records, **kwargs):
        entered.set()
        assert release.wait(2)
        return records

    async def exercise():
        worker = _BatchDispatcher(predict, RecipeServingConfig(batch_wait_ms=0), dynamic=True)
        first = asyncio.create_task(worker.submit([{"value": 1}]))
        while not entered.is_set():
            await asyncio.sleep(0.001)
        closing = asyncio.create_task(worker.close())
        await asyncio.sleep(0)
        assert not closing.done()
        release.set()
        assert await first == [{"value": 1}]
        await closing
        assert worker.pending_records == 0

    asyncio.run(exercise())


def test_model_failure_does_not_stop_the_dispatcher():
    def predict(records, **kwargs):
        if any(row["value"] < 0 for row in records):
            raise RuntimeError("model failure")
        return records

    async def exercise():
        worker = _BatchDispatcher(predict, RecipeServingConfig(batch_wait_ms=0), dynamic=True)
        try:
            with pytest.raises(RuntimeError, match="model failure"):
                await worker.submit([{"value": -1}])
            assert await worker.submit([{"value": 1}]) == [{"value": 1}]
            assert worker.stats["failed_requests"] == 1
        finally:
            await worker.close()

    asyncio.run(exercise())


def test_http_merges_good_requests_and_isolates_invalid_input():
    from httpx import ASGITransport, AsyncClient

    predictor = Predictor()
    app = arti.create_recipe_app(predictor, config=RecipeServingConfig(batch_wait_ms=20))

    async def exercise():
        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                responses = await asyncio.gather(
                    *(
                        client.post("/predict", json={"records": rows})
                        for rows in ([{"value": 1}], [{"bad": 2}], [{"value": 3}])
                    )
                )
                assert [row.status_code for row in responses] == [200, 422, 200]
                assert len(predictor.calls) == 1 and len(predictor.calls[0][0]) == 2
                metrics = (await client.get("/metrics")).json()
                assert metrics["predictor_calls"] == 1
                assert metrics["largest_batch"] == 2
                assert "records" not in metrics and "prompt" not in metrics

    asyncio.run(exercise())


def test_caller_managed_predictor_defaults_to_per_request_execution():
    from httpx import ASGITransport, AsyncClient

    predictor = Predictor()
    predictor.serving_state_scope = "caller-managed"
    app = arti.create_recipe_app(predictor)

    async def exercise():
        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                await asyncio.gather(
                    client.post("/predict", json={"records": [{"value": 1}]}),
                    client.post("/predict", json={"records": [{"value": 2}]}),
                )
                assert len(predictor.calls) == 2
                assert not (await client.get("/health")).json()["dynamic_batching"]

    asyncio.run(exercise())


def test_cli_forwards_serving_configuration_without_loading_model(monkeypatch):
    from arti.cli import main
    from arti import recipe_serving

    calls = []
    monkeypatch.setattr(
        recipe_serving, "serve_recipe", lambda *args, **kwargs: calls.append((args, kwargs))
    )
    assert (
        main(
            [
                "recipe",
                "serve",
                "model",
                "--batching",
                "off",
                "--max-batch-size",
                "8",
                "--batch-wait-ms",
                "2",
                "--max-pending-records",
                "32",
            ]
        )
        == 0
    )
    assert calls[0][1]["config"] == RecipeServingConfig("off", 8, 2, 32)


def test_asgi_disconnect_cancels_queued_request_before_model_execution():
    import json

    predictor = Predictor()
    app = arti.create_recipe_app(predictor, config=RecipeServingConfig(batch_wait_ms=100))

    async def exercise():
        incoming = asyncio.Queue()
        await incoming.put(
            {
                "type": "http.request",
                "body": json.dumps({"records": [{"value": 1}]}).encode(),
                "more_body": False,
            }
        )
        sent = []

        async def send(message):
            sent.append(message)

        scope = {
            "type": "http",
            "asgi": {"version": "3.0"},
            "http_version": "1.1",
            "method": "POST",
            "scheme": "http",
            "path": "/predict",
            "raw_path": b"/predict",
            "query_string": b"",
            "headers": [(b"content-type", b"application/json")],
            "client": ("127.0.0.1", 1234),
            "server": ("test", 80),
            "root_path": "",
        }
        async with app.router.lifespan_context(app):
            request = asyncio.create_task(app(scope, incoming.get, send))
            while app.state.batcher.pending_records == 0:
                await asyncio.sleep(0)
            await incoming.put({"type": "http.disconnect"})
            await request
            assert not predictor.calls
            assert app.state.batcher.pending_records == 0
            assert sent[0]["status"] == 499

    asyncio.run(exercise())


def test_http_queue_pressure_is_503_and_empty_batch_does_not_run_model():
    from httpx import ASGITransport, AsyncClient

    predictor = Predictor()
    app = arti.create_recipe_app(predictor, config=RecipeServingConfig(max_pending_records=1))

    async def exercise():
        async with app.router.lifespan_context(app):
            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                empty = await client.post("/predict", json={"records": []})
                assert empty.json()["predictions"] == [] and not predictor.calls
                response = await client.post(
                    "/predict", json={"records": [{"value": 1}, {"value": 2}]}
                )
                assert response.status_code == 503 and response.headers["Retry-After"] == "1"

    asyncio.run(exercise())
