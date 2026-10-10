"""Load-once local HTTP inference for an exported classification recipe.

This standard-library example is a local integration, not a production gateway.
Run: python examples/serve_classification.py .tmp/digits-workflow/resume/model
POST /predict: {"records": [{"id": "sample", "features": [..]}]}
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json

import arti


def handler_for(predictor: arti.ClassificationPredictor):
    class Handler(BaseHTTPRequestHandler):
        def reply(self, status, value):
            body = json.dumps(value, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self.reply(200, {"ready": True, "labels": list(predictor.labels)})
            else:
                self.reply(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/predict":
                self.reply(404, {"error": "not found"})
                return
            try:
                payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
                if not isinstance(payload, dict) or not isinstance(payload.get("records"), list):
                    raise ValueError("body requires a records array")
                if any(not isinstance(row, dict) for row in payload["records"]):
                    raise ValueError("records must be objects")
                self.reply(
                    200,
                    {
                        "labels": list(predictor.labels),
                        "predictions": predictor.predict_records(payload["records"]),
                    },
                )
            except (ValueError, TypeError) as error:
                self.reply(400, {"error": str(error)})

    return Handler


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    predictor = arti.ClassificationPredictor.from_pretrained(args.model, device=args.device)
    server = HTTPServer((args.host, args.port), handler_for(predictor))
    print(f"Listening at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
