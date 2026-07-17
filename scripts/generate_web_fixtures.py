"""Generate deterministic Python-first ARTI Web parity artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn

from arti.nn import Fold, FusionPulse, Half, LearnedPulse, StatefulRecall
from arti.web import ARTIWebTensorMetadata, export, export_stateful_recall


class GenericAffine(nn.Module):
    """Test-only module proving the Web runtime has no ARTI class registry."""

    def forward(self, signal: torch.Tensor, gate: torch.Tensor) -> dict[str, torch.Tensor]:
        return {"result": signal * (1.0 + gate), "salience": gate.expand_as(signal)}


def _tensor_payload(tensor: torch.Tensor) -> dict[str, object]:
    value = tensor.detach().cpu().contiguous()
    dtype = {
        torch.float32: "float32",
        torch.bool: "bool",
        torch.int64: "int64",
    }.get(value.dtype)
    if dtype is None:
        raise ValueError(f"unsupported fixture dtype {value.dtype}")
    return {"dtype": dtype, "dims": list(value.shape), "data": value.flatten().tolist()}


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
    _write_fusion_pulse_fixture(root)
    _write_stateful_fixture(root)


def _write_fusion_pulse_fixture(root: Path) -> None:
    torch.manual_seed(240717)
    module = FusionPulse(k=3, dim=4, hidden_dim=8).eval()
    pulses = torch.randn(2, 4, 5, 4)
    slot_mask = torch.ones(2, 4, 5, dtype=torch.bool)
    slot_mask[0, 2, -1] = False
    source_mask = torch.tensor(
        [[True, True, True, False], [True, False, True, True]],
        dtype=torch.bool,
    )
    changed = pulses.clone()
    changed[:, 1] = changed[:, 1].flip(1) * 4.0
    first = {"pulses": pulses, "mask": slot_mask, "source_mask": source_mask}
    second = {"pulses": changed, "mask": slot_mask, "source_mask": source_mask}
    with torch.inference_mode():
        _, info = module(**first, return_info=True)
    output_names = ["fused", *info]
    inspect_outputs = (
        "fused",
        "survival",
        "source_index",
        "input_mask",
        "pulse_mask",
        "workspace",
        "workspace_mask",
        "unfold_source_index",
    )
    batch_outputs = {
        "pulses",
        "mask",
        "source_mask",
        "fused",
        "survival",
        "input_mask",
        "pulse_mask",
        "workspace",
        "workspace_mask",
        "unfold_source_index",
    }
    dynamic_axes = {name: {0: "batch"} for name in batch_outputs}
    output_metadata = {
        name: ARTIWebTensorMetadata(
            role=("primary" if name == "fused" else "workspace" if name == "workspace" else "diagnostic"),
            atol=(1e-4 if name not in {"source_index", "input_mask", "pulse_mask", "workspace_mask", "unfold_source_index"} else 0.0),
            rtol=(1e-3 if name not in {"source_index", "input_mask", "pulse_mask", "workspace_mask", "unfold_source_index"} else 0.0),
            max_bytes=8 * 1024 * 1024,
        )
        for name in output_names
    }
    target = root / "fusion-pulse-inspect"
    result = export(
        module,
        target,
        example_inputs=first,
        forward_kwargs={"return_info": True},
        output_names=output_names,
        include_outputs=inspect_outputs,
        input_metadata={
            "pulses": ARTIWebTensorMetadata(max_bytes=8 * 1024 * 1024),
            "mask": ARTIWebTensorMetadata(logical_type="mask", max_bytes=1024 * 1024),
            "source_mask": ARTIWebTensorMetadata(logical_type="mask", max_bytes=1024 * 1024),
        },
        output_metadata=output_metadata,
        dynamic_axes=dynamic_axes,
        dynamic_batch=False,
        dynamic_tokens=False,
    )
    cases = [
        _case_with_info(module, first, inspect_outputs),
        _case_with_info(module, second, inspect_outputs),
    ]
    _verify_python_ort(result.model_path, cases)
    first_outputs = cases[0]["outputs"]
    second_outputs = cases[1]["outputs"]
    if first_outputs["survival"]["data"] == second_outputs["survival"]["data"]:
        raise RuntimeError("FusionPulse fixture does not change real survival")
    if first_outputs["unfold_source_index"]["data"] == second_outputs["unfold_source_index"]["data"]:
        raise RuntimeError("FusionPulse fixture does not change the real UnFold source mapping")
    payload = {"name": "fusion-pulse-inspect", "cases": cases}
    (target / "case.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _case_with_info(
    module: FusionPulse,
    inputs: dict[str, torch.Tensor],
    include_outputs: tuple[str, ...],
) -> dict[str, object]:
    with torch.inference_mode():
        fused, info = module(**inputs, return_info=True)
    all_outputs = {"fused": fused, **info}
    return {
        "inputs": {name: _tensor_payload(value) for name, value in inputs.items()},
        "outputs": {
            name: _tensor_payload(all_outputs[name]) for name in include_outputs
        },
    }


def _verify_python_ort(model_path: Path, cases: list[dict[str, object]]) -> None:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    for case in cases:
        inputs = case["inputs"]
        expected = case["outputs"]
        feeds = {
            name: _payload_numpy(payload)
            for name, payload in inputs.items()
        }
        names = list(expected)
        actual = session.run(names, feeds)
        for name, value in zip(names, actual, strict=True):
            payload = expected[name]
            reference = _payload_numpy(payload)
            if payload["dtype"] == "float32":
                np.testing.assert_allclose(value, reference, atol=1e-4, rtol=1e-3)
            else:
                np.testing.assert_array_equal(value, reference)


def _payload_numpy(payload: dict[str, object]) -> np.ndarray:
    dtype = {"float32": np.float32, "bool": np.bool_, "int64": np.int64}[payload["dtype"]]
    return np.asarray(payload["data"], dtype=dtype).reshape(payload["dims"])


def _write_stateful_fixture(root: Path) -> None:
    module = StatefulRecall(4, slots=4, recognition_threshold=0.97, recognition_temperature=0.01, write_rate=1.0, decay=1.0).eval()
    with torch.no_grad():
        for projection in (module.query, module.key, module.value, module.emit):
            projection.weight.copy_(torch.eye(4))
        module.slot_anchors.copy_(torch.eye(4))
        module.write_quality.weight.zero_()
        module.write_quality.bias.fill_(6.0)
    full = torch.tensor([[[2.0, 0.0, 0.0, 0.0], [1.8, 0.0, 0.0, 0.0]]])
    corrupt = full + torch.tensor([[[0.0, 0.03, 0.0, 0.0], [0.0, -0.02, 0.0, 0.0]]])
    unseen = torch.tensor([[[2.0, 0.8, 0.0, 0.0], [1.8, 0.72, 0.0, 0.0]]])
    mask = torch.ones(1, 2)
    target = root / "stateful-recall"
    export_stateful_recall(module, target, example_x=full, example_mask=mask)
    state = module.initial_state(1)
    first = module.read(full, **state, mask=mask)
    for _ in range(4):
        update = module.update(first["trace_key"], full, **state, mask=mask)
        state = {name: update[name] for name in module.state_names}
    seen = module.read(corrupt, **state, mask=mask)
    novel = module.read(unseen, **state, mask=mask)
    payload = {
        "inputs": {"full": _tensor_payload(full), "corrupt": _tensor_payload(corrupt), "unseen": _tensor_payload(unseen), "mask": _tensor_payload(mask)},
        "expected": {
            "initial_recognition": _tensor_payload(first["recognition"]),
            "seen_recognition": _tensor_payload(seen["recognition"]),
            "seen_delta": _tensor_payload(seen["delta"]),
            "unseen_recognition": _tensor_payload(novel["recognition"]),
            "strengths": _tensor_payload(state["strengths"]),
        },
        "tolerance": {"atol": 1e-4, "rtol": 1e-3},
    }
    (target / "case.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
