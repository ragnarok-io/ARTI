"""Exercise a real local ASGI worker, then close it deterministically."""

from __future__ import annotations

import argparse
import json
import socket
import threading
import time
from urllib.request import ProxyHandler, Request, build_opener

import torch
import uvicorn

import arti
from arti.recipe_data import read_jsonl
from arti.recipe_workflow import write_json


def check(model, data, *, device, output, limit=2):
    predictor = arti.load_recipe_predictor(model, device=device)
    records = list(read_jsonl(data))[:limit]
    expected = predictor.predict_records(records)
    server = uvicorn.Server(uvicorn.Config(arti.create_recipe_app(predictor), log_level="warning"))
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    sock.listen()
    port = sock.getsockname()[1]
    worker = threading.Thread(target=server.run, kwargs={"sockets": [sock]}, daemon=True)
    worker.start()
    opener = build_opener(ProxyHandler({}))
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            if not worker.is_alive() or time.monotonic() >= deadline:
                raise RuntimeError("ASGI worker did not become ready")
            time.sleep(0.05)
        with opener.open(f"http://127.0.0.1:{port}/health", timeout=10) as response:
            health = json.load(response)
        request = Request(
            f"http://127.0.0.1:{port}/predict",
            data=json.dumps({"records": records}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with opener.open(request, timeout=60) as response:
            actual = json.load(response)["predictions"]
        assert actual == expected
        result = {
            "ok": True,
            "samples": len(records),
            "task": predictor.contract["task"],
            "task_recipe_sha256": predictor.contract["sha256"],
            "http_prediction_parity": True,
            "load_once": True,
            "health": health,
            "predictions": actual,
        }
        write_json(output, result)
        return result
    finally:
        server.should_exit = True
        worker.join(timeout=10)
        sock.close()
        if worker.is_alive():
            raise RuntimeError("ASGI worker did not shut down")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("data")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(
        json.dumps(check(args.model, args.data, device=args.device, output=args.output), indent=2)
    )
