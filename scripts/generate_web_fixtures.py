"""Generate deterministic ARTI Web parity artifacts for TypeScript tests."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from arti.nn import Fold, Half, LearnedPulse
from arti.web import export


def _tensor_payload(tensor: torch.Tensor) -> dict[str, object]:
    value = tensor.detach().cpu().to(torch.float32).contiguous()
    return {"dims": list(value.shape), "data": value.flatten().tolist()}


def _run(module, inputs):
    with torch.inference_mode():
        if isinstance(module, Half):
            return module(inputs["x"])
        return module(inputs["x"], q=inputs.get("q"), mask=inputs.get("mask"))


def _case(module, inputs):
    return {
        "inputs": {name: _tensor_payload(value) for name, value in inputs.items()},
        "expected": _tensor_payload(_run(module, inputs)),
    }


def _write_fixture(root: Path, name: str, module, first, second, tolerance):
    target = root / name
    export(module.eval(), target, example_inputs=first)
    payload = {"name": name, "tolerance": tolerance, "cases": [_case(module, first), _case(module, second)]}
    (target / "case.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def generate(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(240713)
    x5 = torch.randn(2, 5, 4)
    x7 = torch.randn(3, 7, 4)
    mask5 = torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 0, 0]], dtype=torch.float32)
    mask7 = torch.ones(3, 7, dtype=torch.float32)
    mask7[:, -2:] = 0
    q5 = torch.sigmoid(torch.randn(2, 5, 1))
    q7 = torch.sigmoid(torch.randn(3, 7, 1))

    _write_fixture(root, "half", Half().eval(), {"x": x5}, {"x": x7}, {"atol": 1e-5, "rtol": 1e-5})
    _write_fixture(
        root,
        "fold-salience",
        Fold(k=3, dim=4).eval(),
        {"x": x5, "mask": mask5},
        {"x": x7, "mask": mask7},
        {"atol": 1e-4, "rtol": 1e-3},
    )
    _write_fixture(
        root,
        "fold-q",
        Fold(k=3, dim=4).eval(),
        {"x": x5, "q": q5, "mask": mask5},
        {"x": x7, "q": q7, "mask": mask7},
        {"atol": 1e-4, "rtol": 1e-3},
    )
    _write_fixture(
        root,
        "learned-pulse",
        LearnedPulse(k=3, dim=4, hidden_dim=6).eval(),
        {"x": x5, "q": q5, "mask": mask5},
        {"x": x7, "q": q7, "mask": mask7},
        {"atol": 1e-4, "rtol": 1e-3},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
