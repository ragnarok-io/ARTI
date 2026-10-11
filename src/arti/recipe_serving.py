"""Optional ASGI deployment for load-once, batched task predictors."""

import asyncio
from contextlib import asynccontextmanager
from typing import Any

from ._version import __version__
from .recipe_batching import RecipeServingConfig, _BatchDispatcher, _ServingUnavailable


def create_recipe_app(predictor, *, config: RecipeServingConfig | None = None):
    """Create an ASGI app without making serving dependencies mandatory."""
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse
        from pydantic import BaseModel, Field
    except ImportError as error:
        raise RuntimeError("recipe serving requires `pip install 'arti-fit[serve]'`") from error

    class BatchRequest(BaseModel):
        records: list[dict[str, Any]]
        batch_size: int | None = Field(default=None, gt=0)
        max_new_tokens: int | None = Field(default=None, gt=0)

    class ChatRequest(BaseModel):
        messages: list[dict[str, str]]
        max_new_tokens: int | None = Field(default=None, gt=0)

    task = predictor.contract["task"]
    language_task = task == "causal-lm" and predictor.contract.get("format") != "arti.task-recipe"
    config = config or RecipeServingConfig()
    dynamic = config.batching == "on" or (
        config.batching == "auto"
        and getattr(predictor, "serving_state_scope", "caller-managed") == "read-only"
    )
    dispatcher = _BatchDispatcher(predictor.predict_records, config, dynamic=dynamic)

    @asynccontextmanager
    async def lifespan(_app):
        dispatcher.start()
        try:
            yield
        finally:
            await dispatcher.close()

    app = FastAPI(title="ARTI Task Inference", version=__version__, lifespan=lifespan)
    app.state.predictor = predictor
    app.state.batcher = dispatcher
    app.state.serving_config = config

    async def predict(records, batch_size=None, max_new_tokens=None):
        kwargs = {}
        if batch_size is not None:
            kwargs["batch_size"] = batch_size
        if language_task:
            kwargs["max_new_tokens"] = max_new_tokens
        elif max_new_tokens is not None:
            raise HTTPException(
                status_code=422, detail="max_new_tokens applies only to a causal-LM task"
            )
        try:
            validate = getattr(predictor, "validate_records", None)
            if validate is not None:
                validate(records)
            if dynamic:
                default_batch = (
                    predictor.generation.batch_size
                    if language_task and hasattr(predictor, "generation")
                    else config.max_batch_size
                )
                kwargs["batch_size"] = min(config.max_batch_size, batch_size or default_batch)
                if (
                    language_task
                    and max_new_tokens is None
                    and hasattr(predictor, "generation")
                ):
                    kwargs["max_new_tokens"] = predictor.generation.max_new_tokens
            return await dispatcher.submit(records, **kwargs)
        except _ServingUnavailable as error:
            raise HTTPException(
                status_code=503, detail=str(error), headers={"Retry-After": "1"}
            ) from error
        except (ValueError, TypeError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except Exception as error:
            raise HTTPException(
                status_code=500, detail=f"model batch execution failed: {type(error).__name__}"
            ) from error

    async def connected_prediction(connection, operation):
        async def disconnected():
            while (await connection.receive())["type"] != "http.disconnect":
                pass

        prediction = asyncio.create_task(operation)
        disconnect = asyncio.create_task(disconnected())
        try:
            done, _ = await asyncio.wait(
                (prediction, disconnect), return_when=asyncio.FIRST_COMPLETED
            )
            if prediction in done:
                return await prediction
            raise HTTPException(status_code=499, detail="client disconnected")
        finally:
            for task_handle in (prediction, disconnect):
                if not task_handle.done():
                    task_handle.cancel()
            await asyncio.gather(prediction, disconnect, return_exceptions=True)

    @app.get("/health")
    def health():
        result = {
            "ready": dispatcher.accepting,
            "task": task,
            "task_recipe_sha256": predictor.contract["sha256"],
            "dynamic_batching": dynamic,
        }
        for name in ("model_source_sha256", "deployment_sha256"):
            value = getattr(predictor, name, None)
            if value is not None:
                result[name] = value
        if task == "classification" and hasattr(predictor, "labels"):
            result["labels"] = list(predictor.labels)
        return JSONResponse(result, status_code=200 if dispatcher.accepting else 503)

    @app.get("/metrics")
    def metrics():
        return dispatcher.snapshot()

    @app.post("/predict")
    async def batch_predict(request: BatchRequest, connection: Request):
        result = {
            "task": task,
            "predictions": await connected_prediction(
                connection, predict(request.records, request.batch_size, request.max_new_tokens)
            ),
        }
        if task == "classification" and hasattr(predictor, "labels"):
            result["labels"] = list(predictor.labels)
        return result

    @app.post("/chat")
    async def chat(request: ChatRequest, connection: Request):
        if not language_task:
            raise HTTPException(status_code=422, detail="this asset is not a causal-LM task")
        records = [{predictor.fields.messages: request.messages}]
        return (
            await connected_prediction(
                connection, predict(records, max_new_tokens=request.max_new_tokens)
            )
        )[0]

    return app


def serve_recipe(
    path, *, device="cpu", host="127.0.0.1", port=8080, config: RecipeServingConfig | None = None, base_model=None, layer_factory=None, task_factory=None
):
    try:
        import uvicorn
    except ImportError as error:
        raise RuntimeError("recipe serving requires `pip install 'arti-fit[serve]'`") from error
    from .recipes import load_recipe_predictor

    predictor = load_recipe_predictor(path, device=device, layer_factory=layer_factory, task_factory=task_factory, **({"base_model": base_model} if base_model is not None else {}))
    uvicorn.run(create_recipe_app(predictor, config=config), host=host, port=port)


__all__ = ["RecipeServingConfig", "create_recipe_app", "serve_recipe"]
