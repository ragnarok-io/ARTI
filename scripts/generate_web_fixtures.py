"""Generate deterministic Python-first ARTI Web parity artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from arti.nn import Fold, Half, LearnedPulse
from arti.web import export


class GenericAffine(nn.Module):
    """Test-only module proving the Web runtime has no ARTI class registry."""

    def forward(self, signal: torch.Tensor, gate: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"result": signal * (1.0 + gate), "salience": gate.expand_as(signal)}


def _tensor_payload(tensor: torch.Tensor) -> dict[str, object]:
    value = tensor.detach().cpu().to(torch.float32).contiguous()
    return {"dims": list(value.shape), "data": value.flatten().tolist()}


def _run(module, inputs):
    with torch.inference_mode():
        value = module(**inputs)
    if isinstance(value, torch.Tensor):
        return {"y": value}
    if isinstance(value, dict):
        return value
    return {f"output_{index}": tensor for index, tensor in enumerate(value)}


def _case(module, inputs):
    return {
        "inputs": {name: _tensor_payload(value) for name, value in inputs.items()},
        "outputs": {name: _tensor_payload(value) for name, value in _run(module, inputs).items()},
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
    _write_fixture(root, "fold-salience", Fold(k=3, dim=4).eval(), {"x": x5, "mask": mask5}, {"x": x7, "mask": mask7}, {"atol": 1e-4, "rtol": 1e-3})
    _write_fixture(root, "fold-q", Fold(k=3, dim=4).eval(), {"x": x5, "q": q5, "mask": mask5}, {"x": x7, "q": q7, "mask": mask7}, {"atol": 1e-4, "rtol": 1e-3})
    _write_fixture(root, "learned-pulse", LearnedPulse(k=3, dim=4, hidden_dim=6).eval(), {"x": x5, "q": q5, "mask": mask5}, {"x": x7, "q": q7, "mask": mask7}, {"atol": 1e-4, "rtol": 1e-3})
    _write_fixture(root, "generic-affine", GenericAffine().eval(), {"signal": x5, "gate": q5}, {"signal": x7, "gate": q7}, {"atol": 1e-5, "rtol": 1e-5})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
