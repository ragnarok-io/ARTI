"""Export Python-defined ARTI modules for the generic Web runtime."""

from __future__ import annotations

import hashlib
import inspect
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch import Tensor

from .._version import __version__
from ..nn import Fold, Half, LearnedPulse
from .contract import (
    ARTI_WEB_FORMAT,
    ARTI_WEB_FORMAT_VERSION,
    ARTI_WEB_LOCK,
    ARTI_WEB_MANIFEST,
    ARTI_WEB_MODEL,
    ARTI_WEB_TYPESCRIPT,
    write_artifact_typescript,
)


@dataclass(frozen=True)
class ARTIWebExportResult:
    """Files and hashes produced by :func:`export`."""

    directory: Path
    manifest_path: Path
    model_path: Path
    lock_path: Path
    manifest_sha256: str
    model_sha256: str
    typescript_path: Path
    typescript_sha256: str


class _ExportWrapper(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        input_names: tuple[str, ...],
        output_kind: str,
        output_keys: tuple[str, ...],
    ) -> None:
        super().__init__()
        self.module = module
        self.input_names = input_names
        self.output_kind = output_kind
        self.output_keys = output_keys

    def forward(self, *values: Tensor):
        result = self.module(**dict(zip(self.input_names, values, strict=True)))
        if self.output_kind == "tensor":
            return result
        if self.output_kind == "mapping":
            return tuple(result[key] for key in self.output_keys)
        return tuple(result)


def export(
    module: nn.Module,
    path: str | Path,
    *,
    example_inputs: Mapping[str, Tensor],
    output_names: Sequence[str] | None = None,
    dynamic_batch: bool = True,
    dynamic_tokens: bool = True,
    opset_version: int = 18,
) -> ARTIWebExportResult:
    """Compile a Python module into a generic ARTI Web artifact.

    The Python module is the source of truth. The exporter calls its real
    ``forward(**example_inputs)`` method, records the resulting tensor
    contract, and exports that same call path to ONNX. JavaScript does not
    inspect the module type or reproduce mechanism-specific behavior.
    """

    if not isinstance(module, nn.Module):
        raise TypeError("module must be a torch.nn.Module")
    if module.training:
        raise ValueError("Web export requires module.eval()")
    if opset_version < 18:
        raise ValueError("opset_version must be at least 18")
    _validate_supported_mode(module)
    input_names, tensors = _validate_inputs(example_inputs)

    with torch.inference_mode():
        sample = module(**dict(zip(input_names, tensors, strict=True)))
    resolved_names, output_tensors, output_kind, output_keys = _normalize_outputs(sample, output_names)

    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    model_path = target / ARTI_WEB_MODEL
    manifest_path = target / ARTI_WEB_MANIFEST
    lock_path = target / ARTI_WEB_LOCK

    wrapper = _ExportWrapper(module, input_names, output_kind, output_keys).eval()
    dynamic_axes = _dynamic_axes(
        input_names,
        tensors,
        resolved_names,
        output_tensors,
        dynamic_batch,
        dynamic_tokens,
    )
    export_options = {
        "input_names": list(input_names),
        "output_names": list(resolved_names),
        "dynamic_axes": dynamic_axes or None,
        "opset_version": opset_version,
        "do_constant_folding": True,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_options["dynamo"] = False
    with torch.inference_mode():
        torch.onnx.export(wrapper, tensors, model_path, **export_options)

    model_sha = _sha256(model_path)
    manifest = {
        "format": ARTI_WEB_FORMAT,
        "format_version": ARTI_WEB_FORMAT_VERSION,
        "package_version": __version__,
        "producer": {"backend": "torch", "graph_format": "onnx"},
        "module": {"type": _module_type(module), "config": _module_config(module)},
        "runtime": {
            "dtype": "float32",
            "opset_version": opset_version,
            "execution_providers": ["webgpu", "wasm"],
        },
        "inputs": [
            _tensor_contract(name, tensor, dynamic_axes.get(name, {}))
            for name, tensor in zip(input_names, tensors, strict=True)
        ],
        "outputs": [
            _tensor_contract(name, tensor, dynamic_axes.get(name, {}))
            for name, tensor in zip(resolved_names, output_tensors, strict=True)
        ],
        "files": {ARTI_WEB_MODEL: {"sha256": model_sha, "size": model_path.stat().st_size}},
    }
    typescript_path = target / ARTI_WEB_TYPESCRIPT
    write_artifact_typescript(manifest, typescript_path)
    typescript_sha = _sha256(typescript_path)
    typescript_record = {"sha256": typescript_sha, "size": typescript_path.stat().st_size}
    manifest["files"][ARTI_WEB_TYPESCRIPT] = typescript_record
    _write_json(manifest_path, manifest)
    manifest_sha = _sha256(manifest_path)
    lock = {
        "format": ARTI_WEB_FORMAT,
        "format_version": ARTI_WEB_FORMAT_VERSION,
        "manifest": {"file": ARTI_WEB_MANIFEST, "sha256": manifest_sha},
        "files": {
            ARTI_WEB_MODEL: {"sha256": model_sha, "size": model_path.stat().st_size},
            ARTI_WEB_TYPESCRIPT: typescript_record,
        },
    }
    _write_json(lock_path, lock)
    return ARTIWebExportResult(target, manifest_path, model_path, lock_path, manifest_sha, model_sha, typescript_path, typescript_sha)


def _validate_supported_mode(module: nn.Module) -> None:
    if isinstance(module, Half):
        if module.stochastic:
            raise ValueError("stochastic Half is not supported by Web export")
        return
    if isinstance(module, Fold):
        if module.dim is None:
            raise ValueError("Web export requires Fold to have an explicit dim")
        if module.mode != "soft":
            raise ValueError("Web export currently supports Fold(mode='soft') only")
        if module.topk is not None:
            raise ValueError("Web export does not support Fold topk")
        if module.dropout.p != 0:
            raise ValueError("Web export requires Fold dropout=0")
        return
    if isinstance(module, LearnedPulse):
        if module.dim is None:
            raise ValueError("Web export requires LearnedPulse to have an explicit dim")
        if module.fold_mode != "soft":
            raise ValueError("Web export currently supports LearnedPulse(fold_mode='soft') only")
        if module.fold_topk is not None or module.q_topk is not None:
            raise ValueError("Web export does not support LearnedPulse topk modes")
        if module.dropout.p != 0:
            raise ValueError("Web export requires LearnedPulse dropout=0")


def _validate_inputs(values: Mapping[str, Tensor]) -> tuple[tuple[str, ...], tuple[Tensor, ...]]:
    if not values:
        raise ValueError("example_inputs must contain at least one tensor")
    names = tuple(values)
    if len(set(names)) != len(names) or any(not name or not isinstance(name, str) for name in names):
        raise ValueError("example input names must be unique non-empty strings")
    tensors = tuple(values[name] for name in names)
    for name, tensor in zip(names, tensors, strict=True):
        if not isinstance(tensor, Tensor):
            raise ValueError(f"{name} must be a Tensor")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must use float32 for Web export")
        if tensor.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU Tensor")
        if tensor.ndim == 0:
            raise ValueError(f"{name} must have at least one dimension")
    return names, tensors


def _normalize_outputs(value, names: Sequence[str] | None):
    if isinstance(value, Tensor):
        tensors = (value,)
        kind = "tensor"
        keys: tuple[str, ...] = ()
        defaults = ("y",)
    elif isinstance(value, Mapping):
        keys = tuple(value)
        tensors = tuple(value[key] for key in keys)
        kind = "mapping"
        defaults = keys
    elif isinstance(value, (tuple, list)):
        tensors = tuple(value)
        kind = "sequence"
        keys = ()
        defaults = tuple(f"output_{index}" for index in range(len(tensors)))
    else:
        raise TypeError("Web export requires Tensor, tensor mapping, or tensor sequence outputs")
    if not tensors or any(not isinstance(tensor, Tensor) for tensor in tensors):
        raise TypeError("all Web export outputs must be tensors")
    for tensor in tensors:
        if tensor.dtype != torch.float32 or tensor.device.type != "cpu" or tensor.ndim == 0:
            raise ValueError("Web export outputs must be non-scalar CPU float32 tensors")
    resolved = tuple(names) if names is not None else defaults
    if len(resolved) != len(tensors) or len(set(resolved)) != len(resolved) or any(not name for name in resolved):
        raise ValueError("output_names must uniquely name every output tensor")
    return resolved, tensors, kind, keys


def _dynamic_axes(input_names, inputs, output_names, outputs, dynamic_batch, dynamic_tokens):
    axes: dict[str, dict[int, str]] = {}
    reference = inputs[0]
    for name, tensor in [*zip(input_names, inputs, strict=True), *zip(output_names, outputs, strict=True)]:
        entry: dict[int, str] = {}
        if dynamic_batch and tensor.ndim >= 1 and tensor.shape[0] == reference.shape[0]:
            entry[0] = "batch"
        if dynamic_tokens and tensor.ndim >= 2 and reference.ndim >= 2 and tensor.shape[1] == reference.shape[1]:
            entry[1] = "tokens"
        if entry:
            axes[name] = entry
    return axes


def _tensor_contract(name: str, tensor: Tensor, axes: Mapping[int, str]):
    shape: list[int | str] = list(tensor.shape)
    for axis, symbol in axes.items():
        shape[axis] = symbol
    return {"name": name, "dtype": "float32", "shape": shape}


def _module_type(module: nn.Module) -> str:
    cls = type(module)
    return f"{cls.__module__}.{cls.__qualname__}"


def _module_config(module: nn.Module):
    if isinstance(module, Half):
        return {"threshold": module.threshold, "base": module.base, "scale": module.scale, "stochastic": False}
    if isinstance(module, Fold):
        return {"k": module.k, "dim": module.dim, "hidden_dim": module.hidden_dim, "temperature": module.temperature, "mode": module.mode}
    if isinstance(module, LearnedPulse):
        return {
            "k": module.k,
            "dim": module.dim,
            "hidden_dim": module.hidden_dim,
            "refine": module.refine_enabled,
            "refine_mode": module.refine_mode,
            "fold_mode": module.fold_mode,
            "use_half": module.use_half,
        }
    return {}


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
