"""Optional ASGI deployment for load-once, batched task predictors."""

from threading import Lock
from typing import Any

from ._version import __version__


def create_recipe_app(predictor):
    """Create an ASGI app without making serving dependencies mandatory."""
    try:
        from fastapi import FastAPI, HTTPException
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
    app = FastAPI(title="ARTI Task Inference", version=__version__)
    app.state.predictor = predictor
    # Each request is batched. A worker serializes model use so native caches
    # and model mode/state are not mutated by concurrent generation calls.
    model_lock = Lock()

    def predict(records, batch_size=None, max_new_tokens=None):
        kwargs = {}
        if batch_size is not None:
            kwargs["batch_size"] = batch_size
        if task == "causal-lm":
            kwargs["max_new_tokens"] = max_new_tokens
        elif max_new_tokens is not None:
            raise HTTPException(
                status_code=422, detail="max_new_tokens applies only to a causal-LM task"
            )
        try:
            with model_lock:
                return predictor.predict_records(records, **kwargs)
        except (ValueError, TypeError) as error:
            raise HTTPException(status_code=422, detail=str(error)) from error

    @app.get("/health")
    def health():
        result = {"ready": True, "task": task, "task_recipe_sha256": predictor.contract["sha256"]}
        if task == "classification":
            result["labels"] = list(predictor.labels)
        return result

    @app.post("/predict")
    def batch_predict(request: BatchRequest):
        result = {
            "task": task,
            "predictions": predict(request.records, request.batch_size, request.max_new_tokens),
        }
        if task == "classification":
            result["labels"] = list(predictor.labels)
        return result

    @app.post("/chat")
    def chat(request: ChatRequest):
        if task != "causal-lm":
            raise HTTPException(status_code=422, detail="this asset is not a causal-LM task")
        records = [{predictor.fields.messages: request.messages}]
        return predict(records, max_new_tokens=request.max_new_tokens)[0]

    return app


def serve_recipe(path, *, device="cpu", host="127.0.0.1", port=8080):
    try:
        import uvicorn
    except ImportError as error:
        raise RuntimeError("recipe serving requires `pip install 'arti-fit[serve]'`") from error
    from .recipes import load_recipe_predictor

    predictor = load_recipe_predictor(path, device=device)
    uvicorn.run(create_recipe_app(predictor), host=host, port=port)


__all__ = ["create_recipe_app", "serve_recipe"]
