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
from ..nn import Fold, FusionPulse, Half, LearnedPulse
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


@dataclass(frozen=True)
class ARTIWebTensorMetadata:
    """Python-owned deployment metadata for one artifact tensor."""

    logical_type: str | None = None
    role: str | None = None
    max_bytes: int = 256 * 1024 * 1024
    atol: float | None = None
    rtol: float | None = None


_SUPPORTED_DTYPES = {
    torch.float32: "float32",
    torch.bool: "bool",
    torch.int64: "int64",
}
_LOGICAL_TYPES = {"tensor", "mask", "index"}
_OUTPUT_ROLES = {"primary", "workspace", "diagnostic"}
_PathPart = str | int
_OutputPath = tuple[_PathPart, ...]


class _ExportWrapper(nn.Module):
    def __init__(
        self,
        module: nn.Module,
        input_names: tuple[str, ...],
        output_paths: tuple[_OutputPath, ...],
        forward_kwargs: Mapping[str, object],
    ) -> None:
        super().__init__()
        self.module = module
        self.input_names = input_names
        self.output_paths = output_paths
        self.forward_kwargs = dict(forward_kwargs)

    def forward(self, *values: Tensor):
        kwargs = dict(zip(self.input_names, values, strict=True))
        result = self.module(**kwargs, **self.forward_kwargs)
        outputs = tuple(_resolve_output_path(result, path) for path in self.output_paths)
        return outputs[0] if len(outputs) == 1 else outputs


def export(
    module: nn.Module,
    path: str | Path,
    *,
    example_inputs: Mapping[str, Tensor],
    output_names: Sequence[str] | None = None,
    include_outputs: Sequence[str] | None = None,
    forward_kwargs: Mapping[str, object] | None = None,
    input_metadata: Mapping[str, ARTIWebTensorMetadata | Mapping[str, object]] | None = None,
    output_metadata: Mapping[str, ARTIWebTensorMetadata | Mapping[str, object]] | None = None,
    dynamic_axes: Mapping[str, Mapping[int, str]] | None = None,
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
    static_kwargs = _validate_forward_kwargs(forward_kwargs)

    with torch.inference_mode():
        sample = module(
            **dict(zip(input_names, tensors, strict=True)),
            **static_kwargs,
        )
    resolved_names, output_tensors, output_paths = _normalize_outputs(sample, output_names)
    resolved_names, output_tensors, output_paths = _select_outputs(
        resolved_names,
        output_tensors,
        output_paths,
        include_outputs,
    )

    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    model_path = target / ARTI_WEB_MODEL
    manifest_path = target / ARTI_WEB_MANIFEST
    lock_path = target / ARTI_WEB_LOCK

    wrapper = _ExportWrapper(module, input_names, output_paths, static_kwargs).eval()
    resolved_dynamic_axes = _dynamic_axes(
        input_names,
        tensors,
        resolved_names,
        output_tensors,
        dynamic_batch,
        dynamic_tokens,
        dynamic_axes,
    )
    export_options = {
        "input_names": list(input_names),
        "output_names": list(resolved_names),
        "dynamic_axes": resolved_dynamic_axes or None,
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
        "module": {
            "type": _module_type(module),
            "config": _module_config(module),
            "forward_kwargs": static_kwargs,
        },
        "runtime": {
            "dtype": "float32",
            "opset_version": opset_version,
            "execution_providers": ["webgpu", "wasm"],
        },
        "inputs": [
            _tensor_contract(
                name,
                tensor,
                resolved_dynamic_axes.get(name, {}),
                _metadata_for(name, input_metadata),
                output=False,
                primary=False,
            )
            for name, tensor in zip(input_names, tensors, strict=True)
        ],
        "outputs": [
            _tensor_contract(
                name,
                tensor,
                resolved_dynamic_axes.get(name, {}),
                _metadata_for(name, output_metadata),
                output=True,
                primary=index == 0,
            )
            for index, (name, tensor) in enumerate(
                zip(resolved_names, output_tensors, strict=True)
            )
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
        return
    if isinstance(module, FusionPulse):
        if module.unfold.hard_backend != "sort":
            raise ValueError("Web export currently supports FusionPulse with UnFold sort only")


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
        if tensor.dtype not in _SUPPORTED_DTYPES:
            supported = ", ".join(_SUPPORTED_DTYPES.values())
            raise ValueError(f"{name} must use one of the Web dtypes: {supported}")
        if tensor.device.type != "cpu":
            raise ValueError(f"{name} must be a CPU Tensor")
        if tensor.ndim == 0:
            raise ValueError(f"{name} must have at least one dimension")
    return names, tensors


def _validate_forward_kwargs(values: Mapping[str, object] | None) -> dict[str, object]:
    if values is None:
        return {}
    if not isinstance(values, Mapping):
        raise TypeError("forward_kwargs must be a mapping")
    if any(not isinstance(name, str) or not name for name in values):
        raise ValueError("forward_kwargs names must be non-empty strings")
    if any(isinstance(value, Tensor) for value in values.values()):
        raise TypeError("forward_kwargs cannot contain tensors; declare tensor inputs in example_inputs")
    try:
        return json.loads(json.dumps(dict(values), allow_nan=False))
    except (TypeError, ValueError) as error:
        raise TypeError("forward_kwargs must contain JSON-compatible static values") from error


def _normalize_outputs(
    value: object,
    names: Sequence[str] | None,
) -> tuple[tuple[str, ...], tuple[Tensor, ...], tuple[_OutputPath, ...]]:
    flattened = _flatten_outputs(value)
    if not flattened:
        raise TypeError("Web export requires at least one nested Tensor output")
    paths = tuple(path for path, _ in flattened)
    tensors = tuple(tensor for _, tensor in flattened)
    for tensor in tensors:
        if tensor.dtype not in _SUPPORTED_DTYPES or tensor.device.type != "cpu":
            supported = ", ".join(_SUPPORTED_DTYPES.values())
            raise ValueError(f"Web export outputs must be CPU tensors using: {supported}")
    defaults = tuple(_default_output_name(path) for path in paths)
    resolved = tuple(names) if names is not None else defaults
    if len(resolved) != len(tensors) or len(set(resolved)) != len(resolved) or any(not name for name in resolved):
        raise ValueError("output_names must uniquely name every output tensor")
    return resolved, tensors, paths


def _flatten_outputs(value: object, path: _OutputPath = ()) -> list[tuple[_OutputPath, Tensor]]:
    if isinstance(value, Tensor):
        return [(path, value)]
    if isinstance(value, Mapping):
        flattened: list[tuple[_OutputPath, Tensor]] = []
        for key, nested in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError("nested Web output mapping keys must be non-empty strings")
            flattened.extend(_flatten_outputs(nested, (*path, key)))
        return flattened
    if isinstance(value, (tuple, list)):
        flattened = []
        for index, nested in enumerate(value):
            flattened.extend(_flatten_outputs(nested, (*path, index)))
        return flattened
    raise TypeError("nested Web outputs may contain only tensors, mappings, tuples, and lists")


def _select_outputs(
    names: tuple[str, ...],
    tensors: tuple[Tensor, ...],
    paths: tuple[_OutputPath, ...],
    requested: Sequence[str] | None,
) -> tuple[tuple[str, ...], tuple[Tensor, ...], tuple[_OutputPath, ...]]:
    if requested is None:
        return names, tensors, paths
    selected = tuple(requested)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("include_outputs must contain unique output names")
    positions = {name: index for index, name in enumerate(names)}
    unknown = [name for name in selected if name not in positions]
    if unknown:
        raise ValueError(f"include_outputs contains unknown names: {unknown}")
    indices = tuple(positions[name] for name in selected)
    return (
        selected,
        tuple(tensors[index] for index in indices),
        tuple(paths[index] for index in indices),
    )


def _resolve_output_path(value: object, path: _OutputPath) -> Tensor:
    current = value
    for part in path:
        current = current[part]  # type: ignore[index]
    if not isinstance(current, Tensor):
        raise RuntimeError("exported output path no longer resolves to a Tensor")
    return current


def _default_output_name(path: _OutputPath) -> str:
    if not path:
        return "y"
    if path == (0,):
        return "y"
    visible = path[1:] if len(path) > 1 and path[0] == 1 else path
    parts = [str(part) if isinstance(part, str) else f"output_{part}" for part in visible]
    return "__".join(parts)


def _dynamic_axes(
    input_names,
    inputs,
    output_names,
    outputs,
    dynamic_batch,
    dynamic_tokens,
    overrides,
):
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
    if overrides is not None:
        tensors = dict(
            [*zip(input_names, inputs, strict=True), *zip(output_names, outputs, strict=True)]
        )
        for name, requested in overrides.items():
            tensor = tensors.get(name)
            if tensor is None:
                raise ValueError(f"dynamic_axes references unknown tensor {name!r}")
            entry: dict[int, str] = {}
            for axis, symbol in requested.items():
                if not isinstance(axis, int) or axis < 0 or axis >= tensor.ndim:
                    raise ValueError(f"dynamic axis {axis!r} is invalid for {name}")
                if not isinstance(symbol, str) or not symbol:
                    raise ValueError("dynamic axis symbols must be non-empty strings")
                entry[axis] = symbol
            if entry:
                axes[name] = entry
            else:
                axes.pop(name, None)
    return axes


def _metadata_for(
    name: str,
    values: Mapping[str, ARTIWebTensorMetadata | Mapping[str, object]] | None,
) -> ARTIWebTensorMetadata:
    if values is None or name not in values:
        return ARTIWebTensorMetadata()
    value = values[name]
    if isinstance(value, ARTIWebTensorMetadata):
        return value
    if not isinstance(value, Mapping):
        raise TypeError(f"metadata for {name} must be ARTIWebTensorMetadata or a mapping")
    try:
        return ARTIWebTensorMetadata(**value)
    except TypeError as error:
        raise TypeError(f"invalid metadata for {name}") from error


def _tensor_contract(
    name: str,
    tensor: Tensor,
    axes: Mapping[int, str],
    metadata: ARTIWebTensorMetadata,
    *,
    output: bool,
    primary: bool,
):
    shape: list[int | str] = list(tensor.shape)
    for axis, symbol in axes.items():
        shape[axis] = symbol
    dtype = _SUPPORTED_DTYPES[tensor.dtype]
    logical_type = metadata.logical_type or (
        "mask" if tensor.dtype == torch.bool else "index" if tensor.dtype == torch.int64 else "tensor"
    )
    if logical_type not in _LOGICAL_TYPES:
        raise ValueError(f"logical_type for {name} must be one of {sorted(_LOGICAL_TYPES)}")
    if not isinstance(metadata.max_bytes, int) or metadata.max_bytes <= 0 or metadata.max_bytes > 2**53 - 1:
        raise ValueError(f"max_bytes for {name} must be a positive safe integer")
    sample_bytes = tensor.numel() * tensor.element_size()
    if sample_bytes > metadata.max_bytes:
        raise ValueError(f"sample tensor {name} requires {sample_bytes} bytes, exceeding max_bytes")
    default_atol = 1e-4 if output and tensor.is_floating_point() else 0.0
    default_rtol = 1e-3 if output and tensor.is_floating_point() else 0.0
    atol = default_atol if metadata.atol is None else float(metadata.atol)
    rtol = default_rtol if metadata.rtol is None else float(metadata.rtol)
    if not torch.isfinite(torch.tensor([atol, rtol])).all() or atol < 0 or rtol < 0:
        raise ValueError(f"tolerance for {name} must be finite and non-negative")
    contract = {
        "name": name,
        "dtype": dtype,
        "logical_type": logical_type,
        "shape": shape,
        "dynamic_axes": {str(axis): symbol for axis, symbol in sorted(axes.items())},
        "max_bytes": metadata.max_bytes,
        "tolerance": {"atol": atol, "rtol": rtol},
    }
    if output:
        role = metadata.role or ("primary" if primary else "diagnostic")
        if role not in _OUTPUT_ROLES:
            raise ValueError(f"role for {name} must be one of {sorted(_OUTPUT_ROLES)}")
        contract["role"] = role
    elif metadata.role is not None:
        raise ValueError(f"input metadata for {name} cannot declare an output role")
    return contract


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
    if isinstance(module, FusionPulse):
        return {
            "k": module.k,
            "dim": module.dim,
            "hidden_dim": module.hidden_dim,
            "salience_heads": module.salience_heads,
            "half_threshold": module.half_threshold,
            "salience_scale": module.salience_scale,
            "unfold_backend": module.unfold.hard_backend,
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
