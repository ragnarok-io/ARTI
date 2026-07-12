"""Export deterministic ARTI modules for the TypeScript Web runtime."""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
from torch import Tensor

from .._version import __version__
from ..nn import Fold, Half, LearnedPulse


ARTI_WEB_FORMAT = "arti.web"
ARTI_WEB_FORMAT_VERSION = 1
ARTI_WEB_MANIFEST = "arti-web.json"
ARTI_WEB_MODEL = "model.onnx"
ARTI_WEB_LOCK = "arti-web.lock.json"


@dataclass(frozen=True)
class ARTIWebExportResult:
    """Files and hashes produced by :func:`export`."""

    directory: Path
    manifest_path: Path
    model_path: Path
    lock_path: Path
    manifest_sha256: str
    model_sha256: str


class _ExportWrapper(nn.Module):
    def __init__(self, module: nn.Module, input_names: tuple[str, ...]) -> None:
        super().__init__()
        self.module = module
        self.input_names = input_names

    def forward(self, *values: Tensor) -> Tensor:
        inputs = dict(zip(self.input_names, values, strict=True))
        x = inputs["x"]
        if isinstance(self.module, Half):
            return self.module(x)
        return self.module(x, q=inputs.get("q"), mask=inputs.get("mask"))


def export(
    module: nn.Module,
    path: str | Path,
    *,
    example_inputs: Mapping[str, Tensor],
    dynamic_batch: bool = True,
    dynamic_tokens: bool = True,
    opset_version: int = 18,
) -> ARTIWebExportResult:
    """Export a deterministic ARTI module as a static Web artifact directory.

    The alpha exporter supports deterministic :class:`Half`, soft
    :class:`Fold`, and soft-fold :class:`LearnedPulse` modules with float32
    ``[B, N, D]`` inputs. Optional ``q`` and ``mask`` inputs become part of the
    artifact contract only when present in ``example_inputs``.
    """

    if not isinstance(module, (Half, Fold, LearnedPulse)):
        raise TypeError("Web export supports Half, Fold, and LearnedPulse modules")
    if module.training:
        raise ValueError("Web export requires module.eval()")
    if opset_version < 18:
        raise ValueError("opset_version must be at least 18")
    _validate_module(module)
    names, tensors = _validate_inputs(module, example_inputs)

    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    model_path = target / ARTI_WEB_MODEL
    manifest_path = target / ARTI_WEB_MANIFEST
    lock_path = target / ARTI_WEB_LOCK

    wrapper = _ExportWrapper(module, names).eval()
    dynamic_axes = _dynamic_axes(names, module, tensors, dynamic_batch, dynamic_tokens)
    export_options = {
        "input_names": list(names),
        "output_names": ["y"],
        "dynamic_axes": dynamic_axes or None,
        "opset_version": opset_version,
        "do_constant_folding": True,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        export_options["dynamo"] = False
    with torch.inference_mode():
        torch.onnx.export(wrapper, tensors, model_path, **export_options)

    model_sha = _sha256(model_path)
    output_shape = _output_shape(module, tensors[0])
    manifest = {
        "format": ARTI_WEB_FORMAT,
        "format_version": ARTI_WEB_FORMAT_VERSION,
        "package_version": __version__,
        "module": {"type": type(module).__name__, "config": _module_config(module)},
        "runtime": {
            "dtype": "float32",
            "opset_version": opset_version,
            "execution_providers": ["webgpu", "wasm"],
        },
        "inputs": [_tensor_contract(name, tensor, dynamic_batch, dynamic_tokens) for name, tensor in zip(names, tensors, strict=True)],
        "output": _shape_contract("y", output_shape, dynamic_batch, dynamic_tokens and isinstance(module, Half)),
        "files": {ARTI_WEB_MODEL: {"sha256": model_sha, "size": model_path.stat().st_size}},
    }
    _write_json(manifest_path, manifest)
    manifest_sha = _sha256(manifest_path)
    lock = {
        "format": ARTI_WEB_FORMAT,
        "format_version": ARTI_WEB_FORMAT_VERSION,
        "manifest": {"file": ARTI_WEB_MANIFEST, "sha256": manifest_sha},
        "files": {ARTI_WEB_MODEL: {"sha256": model_sha, "size": model_path.stat().st_size}},
    }
    _write_json(lock_path, lock)
    return ARTIWebExportResult(target, manifest_path, model_path, lock_path, manifest_sha, model_sha)


def _validate_module(module: nn.Module) -> None:
    if isinstance(module, Half):
        if module.stochastic:
            raise ValueError("stochastic Half is not supported by Web export")
        return
    if module.dim is None:
        raise ValueError("Web export requires an explicit dim")
    if isinstance(module, Fold):
        if module.mode != "soft":
            raise ValueError("Web export currently supports Fold(mode='soft') only")
        if module.topk is not None:
            raise ValueError("Web export does not support Fold topk")
        if module.dropout.p != 0:
            raise ValueError("Web export requires Fold dropout=0")
        return
    if module.fold_mode != "soft":
        raise ValueError("Web export currently supports LearnedPulse(fold_mode='soft') only")
    if module.fold_topk is not None or module.q_topk is not None:
        raise ValueError("Web export does not support LearnedPulse topk modes")
    if module.dropout.p != 0:
        raise ValueError("Web export requires LearnedPulse dropout=0")


def _validate_inputs(module: nn.Module, values: Mapping[str, Tensor]) -> tuple[tuple[str, ...], tuple[Tensor, ...]]:
    allowed = {"x"} if isinstance(module, Half) else {"x", "q", "mask"}
    unknown = set(values) - allowed
    if unknown:
        raise ValueError(f"unsupported Web export inputs: {sorted(unknown)}")
    if "x" not in values:
        raise ValueError("example_inputs must contain x")
    names = tuple(name for name in ("x", "q", "mask") if name in values)
    tensors = tuple(values[name] for name in names)
    x = tensors[0]
    if not isinstance(x, Tensor) or x.ndim != 3:
        raise ValueError("x must be a Tensor with shape [B, N, D]")
    if x.dtype != torch.float32:
        raise ValueError("Web alpha export supports float32 only")
    if x.device.type != "cpu":
        raise ValueError("example inputs must be on CPU")
    dim = getattr(module, "dim", None)
    if dim is not None and x.shape[-1] != dim:
        raise ValueError(f"expected x feature dim {dim}, got {x.shape[-1]}")
    for name, tensor in zip(names[1:], tensors[1:], strict=True):
        if not isinstance(tensor, Tensor) or tensor.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU Tensor")
        if tensor.dtype != torch.float32:
            raise ValueError(f"{name} must use float32 for Web export")
        if tensor.ndim not in {2, 3} or tensor.shape[:2] != x.shape[:2] or (tensor.ndim == 3 and tensor.shape[-1] != 1):
            raise ValueError(f"{name} must have shape [B, N] or [B, N, 1]")
    return names, tensors


def _dynamic_axes(names, module, tensors, dynamic_batch, dynamic_tokens):
    axes: dict[str, dict[int, str]] = {}
    for name, tensor in zip(names, tensors, strict=True):
        entry: dict[int, str] = {}
        if dynamic_batch:
            entry[0] = "batch"
        if dynamic_tokens and tensor.ndim >= 2:
            entry[1] = "tokens"
        if entry:
            axes[name] = entry
    output: dict[int, str] = {}
    if dynamic_batch:
        output[0] = "batch"
    if dynamic_tokens and isinstance(module, Half):
        output[1] = "tokens"
    if output:
        axes["y"] = output
    return axes


def _tensor_contract(name: str, tensor: Tensor, dynamic_batch: bool, dynamic_tokens: bool):
    return _shape_contract(name, list(tensor.shape), dynamic_batch, dynamic_tokens)


def _shape_contract(name: str, shape: list[int], dynamic_batch: bool, dynamic_tokens: bool):
    resolved: list[int | str] = list(shape)
    if dynamic_batch:
        resolved[0] = "batch"
    if dynamic_tokens and len(resolved) >= 2:
        resolved[1] = "tokens"
    return {"name": name, "dtype": "float32", "shape": resolved}


def _output_shape(module: nn.Module, x: Tensor) -> list[int]:
    if isinstance(module, Half):
        return list(x.shape)
    return [x.shape[0], module.k, x.shape[-1]]


def _module_config(module: nn.Module):
    if isinstance(module, Half):
        return {"threshold": module.threshold, "base": module.base, "scale": module.scale, "stochastic": False}
    if isinstance(module, Fold):
        return {"k": module.k, "dim": module.dim, "hidden_dim": module.hidden_dim, "temperature": module.temperature, "mode": module.mode}
    return {
        "k": module.k,
        "dim": module.dim,
        "hidden_dim": module.hidden_dim,
        "refine": module.refine_enabled,
        "refine_mode": module.refine_mode,
        "fold_mode": module.fold_mode,
        "use_half": module.use_half,
    }


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
