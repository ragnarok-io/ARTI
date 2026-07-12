"""Export explicit-state Recall read/update graphs for browser inference."""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from .._version import __version__
from ..stateful_recall import StatefulRecall
from .contract import ARTI_WEB_FORMAT, ARTI_WEB_LOCK, ARTI_WEB_MANIFEST

STATEFUL_FORMAT_VERSION = 3
READ_MODEL = "read.onnx"
UPDATE_MODEL = "update.onnx"


@dataclass(frozen=True)
class ARTIStatefulWebExportResult:
    directory: Path
    manifest_path: Path
    read_model_path: Path
    update_model_path: Path
    lock_path: Path


class _ReadGraph(nn.Module):
    def __init__(self, module: StatefulRecall, with_mask: bool) -> None:
        super().__init__()
        self.module = module
        self.with_mask = with_mask

    def forward(self, x, keys, values, strengths, mask=None):
        output = self.module.read(x, keys, values, strengths, mask if self.with_mask else None)
        return output["y"], output["delta"], output["recognition"], output["trace_key"], output["trace_value"]


class _UpdateGraph(nn.Module):
    def __init__(self, module: StatefulRecall, with_mask: bool) -> None:
        super().__init__()
        self.module = module
        self.with_mask = with_mask

    def forward(self, trace_key, observed, keys, values, strengths, mask=None):
        output = self.module.update(trace_key, observed, keys, values, strengths, mask if self.with_mask else None)
        return output["keys"], output["values"], output["strengths"], output["write_assignment"], output["write_quality"]


def export_stateful_recall(
    module: StatefulRecall,
    path: str | Path,
    *,
    example_x: Tensor,
    example_mask: Tensor | None = None,
    opset_version: int = 18,
) -> ARTIStatefulWebExportResult:
    """Export paired read/update graphs and a fixed-size empty Recall state."""

    if not isinstance(module, StatefulRecall):
        raise TypeError("module must be StatefulRecall")
    if module.training:
        raise ValueError("stateful Web export requires module.eval()")
    if example_x.dtype != torch.float32 or example_x.device.type != "cpu":
        raise ValueError("stateful Web export requires CPU float32 example_x")
    if example_x.ndim != 3 or example_x.shape[-1] != module.dim:
        raise ValueError(f"example_x must have shape [B, N, {module.dim}]")
    if example_mask is not None and (example_mask.shape != example_x.shape[:2] or example_mask.dtype != torch.float32):
        raise ValueError("example_mask must be float32 with shape [B, N]")
    if opset_version < 18:
        raise ValueError("opset_version must be at least 18")

    state = module.initial_state(example_x.shape[0], device="cpu", dtype=torch.float32)
    with_mask = example_mask is not None
    read_inputs = (example_x, state["keys"], state["values"], state["strengths"])
    read_names = ["x", "keys", "values", "strengths"]
    if with_mask:
        read_inputs += (example_mask,)
        read_names.append("mask")
    with torch.inference_mode():
        read_sample = _ReadGraph(module, with_mask)(*read_inputs)
    update_inputs = (read_sample[3], example_x, state["keys"], state["values"], state["strengths"])
    update_names = ["trace_key", "observed", "keys", "values", "strengths"]
    if with_mask:
        update_inputs += (example_mask,)
        update_names.append("mask")
    with torch.inference_mode():
        update_sample = _UpdateGraph(module, with_mask)(*update_inputs)

    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    read_path, update_path = target / READ_MODEL, target / UPDATE_MODEL
    read_outputs = ["y", "delta", "recognition", "trace_key", "trace_value"]
    update_outputs = ["next_keys", "next_values", "next_strengths", "write_assignment", "write_quality"]
    _onnx_export(_ReadGraph(module, with_mask), read_inputs, read_path, read_names, read_outputs, opset_version)
    _onnx_export(_UpdateGraph(module, with_mask), update_inputs, update_path, update_names, update_outputs, opset_version)

    read_axes = {name: {0: "batch"} for name in [*read_names, *read_outputs]}
    for name in ("x", "mask", "y", "delta", "recognition"):
        if name in read_axes:
            read_axes[name][1] = "tokens"
    update_axes = {name: {0: "batch"} for name in [*update_names, *update_outputs]}
    for name in ("observed", "mask"):
        if name in update_axes:
            update_axes[name][1] = "tokens"
    files = {name: {"sha256": _sha256(file), "size": file.stat().st_size} for name, file in ((READ_MODEL, read_path), (UPDATE_MODEL, update_path))}
    manifest = {
        "format": ARTI_WEB_FORMAT,
        "format_version": STATEFUL_FORMAT_VERSION,
        "artifact_kind": "stateful",
        "package_version": __version__,
        "producer": {"backend": "torch", "graph_format": "onnx"},
        "module": {"type": f"{type(module).__module__}.{type(module).__qualname__}", "config": {
            "dim": module.dim, "slots": module.slots, "key_dim": module.key_dim,
            "use_half": module.use_half, "write_rate": module.write_rate, "decay": module.decay, "learnable_dynamics": module.learnable_dynamics,
        }},
        "runtime": {"dtype": "float32", "opset_version": opset_version, "execution_providers": ["webgpu", "wasm"]},
        "state": [
            _contract("keys", state["keys"], {0: "batch"}, initializer="zeros"),
            _contract("values", state["values"], {0: "batch"}, initializer="zeros"),
            _contract("strengths", state["strengths"], {0: "batch"}, initializer="zeros"),
        ],
        "entrypoints": {
            "read": {"file": READ_MODEL, "inputs": [_contract(n, t, read_axes[n]) for n, t in zip(read_names, read_inputs, strict=True)], "outputs": [_contract(n, t, read_axes[n]) for n, t in zip(read_outputs, read_sample, strict=True)]},
            "update": {"file": UPDATE_MODEL, "inputs": [_contract(n, t, update_axes[n]) for n, t in zip(update_names, update_inputs, strict=True)], "outputs": [_contract(n, t, update_axes[n]) for n, t in zip(update_outputs, update_sample, strict=True)], "state_outputs": {"keys": "next_keys", "values": "next_values", "strengths": "next_strengths"}},
        },
        "files": files,
        "limits": {"max_state_bytes_per_batch": (module.slots * module.key_dim + module.slots * module.dim + module.slots) * 4},
        "persistence": "explicit",
    }
    manifest_path, lock_path = target / ARTI_WEB_MANIFEST, target / ARTI_WEB_LOCK
    _write_json(manifest_path, manifest)
    lock = {"format": ARTI_WEB_FORMAT, "format_version": STATEFUL_FORMAT_VERSION, "manifest": {"file": ARTI_WEB_MANIFEST, "sha256": _sha256(manifest_path)}, "files": files}
    _write_json(lock_path, lock)
    return ARTIStatefulWebExportResult(target, manifest_path, read_path, update_path, lock_path)


def _onnx_export(module, inputs, path, input_names, output_names, opset):
    dynamic_axes = {name: {0: "batch"} for name in [*input_names, *output_names]}
    for name in ("x", "observed", "mask", "y", "delta", "recognition"):
        if name in dynamic_axes:
            dynamic_axes[name][1] = "tokens"
    options = {"input_names": input_names, "output_names": output_names, "dynamic_axes": dynamic_axes, "opset_version": opset, "do_constant_folding": True}
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        options["dynamo"] = False
    with torch.inference_mode():
        torch.onnx.export(module.eval(), inputs, path, **options)


def _contract(name: str, tensor: Tensor, axes: dict[int, str], initializer: str | None = None):
    shape: list[int | str] = list(tensor.shape)
    for axis, symbol in axes.items():
        shape[axis] = symbol
    result = {"name": name, "dtype": "float32", "shape": shape}
    if initializer is not None:
        result["initializer"] = initializer
    return result


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
