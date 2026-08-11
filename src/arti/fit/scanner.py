"""PyTorch model scanning for ARTI insertion points."""

from __future__ import annotations

import fnmatch
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from ..tensor_boundary import PRIMARY_TENSOR_KEYS, TensorLayout, find_primary_tensor
from .batch_schema import BatchSchema, infer_batch_schema
from .runtime import RuntimeFieldConfig, adapter_context, runtime_keys, runtime_kwargs_from_batch


@dataclass(frozen=True)
class InsertionCandidate:
    name: str
    module_path: str
    position: str
    module_type: str
    output_shape: tuple[int, ...]
    dim: int
    parameters: int
    source: str = "forward"
    tensor_rank: int | None = None
    path_depth: int = 0
    output_path: tuple[str | int, ...] = ()
    tensor_path: tuple[str | int, ...] = ()
    batch_axis: int | None = None
    feature_axis: int | None = None
    device: str | None = None
    dtype: str | None = None
    is_leaf: bool = True


@dataclass(frozen=True)
class ScanReport:
    candidates: tuple[InsertionCandidate, ...]
    total_parameters: int
    trainable_parameters: int
    device: str
    dtype: str
    batch_schema: BatchSchema | None = None
    scanned_modules: int = 0
    candidate_events: int = 0
    duplicate_events: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidates": [candidate.__dict__ for candidate in self.candidates],
            "candidate_count": len(self.candidates),
            "scanned_modules": self.scanned_modules,
            "candidate_events": self.candidate_events,
            "duplicate_events": self.duplicate_events,
            "total_parameters": self.total_parameters,
            "trainable_parameters": self.trainable_parameters,
            "device": self.device,
            "dtype": self.dtype,
            "batch_schema": None if self.batch_schema is None else self.batch_schema.to_dict(),
        }


def run_model(model: nn.Module, sample_batch: Any, *, causal: bool = False, runtime_fields: RuntimeFieldConfig | None = None) -> Any:
    if isinstance(sample_batch, dict):
        schema = infer_batch_schema(sample_batch)
        ignored = runtime_keys(runtime_fields)
        if schema is not None and schema.label_key is not None:
            ignored.add(schema.label_key)
        if schema is not None and schema.mask_key is not None:
            ignored.discard(schema.mask_key)
        context_kwargs = runtime_kwargs_from_batch(sample_batch, runtime_fields)
        with adapter_context(**context_kwargs, causal=causal):
            return model(**{key: value for key, value in sample_batch.items() if key not in ignored})
    if isinstance(sample_batch, tuple):
        return model(*sample_batch)
    return model(sample_batch)


STRUCTURED_TENSOR_KEYS = PRIMARY_TENSOR_KEYS


def tensor_from_module_output(output: Any) -> Tensor | None:
    found = find_primary_tensor(output)
    return None if found is None else found[0]


SCANNABLE_MODULE_TYPES = (
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.GroupNorm,
    nn.Linear,
    nn.Embedding,
    nn.LayerNorm,
    nn.MultiheadAttention,
    nn.RNN,
    nn.LSTM,
    nn.GRU,
    nn.TransformerEncoderLayer,
    nn.TransformerDecoderLayer,
)


def scan_model(
    model: nn.Module,
    sample_batch: Any | None = None,
    *,
    causal: bool = False,
    runtime_fields: RuntimeFieldConfig | None = None,
    batch_axis: int | Mapping[str, int] | None = None,
    feature_axis: int | Mapping[str, int] | None = None,
    positions: str | tuple[str, ...] = ("output",),
) -> ScanReport:
    resolved_positions = _normalize_positions(positions)
    if sample_batch is None and "input" in resolved_positions:
        raise ValueError("input boundary discovery requires sample_batch")
    candidates: list[InsertionCandidate] = []
    seen_names: set[str] = set()
    hooks = []
    candidate_events = 0
    duplicate_events = 0

    def append_candidate(candidate: InsertionCandidate) -> None:
        nonlocal candidate_events, duplicate_events
        candidate_events += 1
        if candidate.name in seen_names:
            duplicate_events += 1
            return
        seen_names.add(candidate.name)
        candidates.append(candidate)

    def output_hook(name: str, module: nn.Module):
        def _capture(_: nn.Module, __: tuple[Any, ...], output: Any) -> None:
            found = find_primary_tensor(output)
            if found is not None:
                tensor, output_path = found
                inferred = TensorLayout.infer(tensor, module)
                layout = TensorLayout(
                    tuple(int(value) for value in tensor.shape),
                    _axis_for_path(batch_axis, name, inferred.batch_axis),
                    _axis_for_path(feature_axis, name, inferred.feature_axis),
                )
                append_candidate(
                    InsertionCandidate(
                        name=name,
                        module_path=name,
                        position="output",
                        module_type=module.__class__.__name__,
                        output_shape=tuple(int(dim) for dim in tensor.shape),
                        dim=layout.feature_dim,
                        parameters=sum(param.numel() for param in module.parameters()),
                        source="forward",
                        tensor_rank=int(tensor.ndim),
                        path_depth=name.count(".") + 1,
                        output_path=output_path,
                        tensor_path=output_path,
                        batch_axis=layout.batch_axis,
                        feature_axis=layout.feature_axis,
                        device=str(tensor.device),
                        dtype=str(tensor.dtype),
                        is_leaf=not any(module.children()),
                    )
                )

        return _capture

    def input_hook(name: str, module: nn.Module):
        def _capture(_: nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
            tree = {"args": args, "kwargs": kwargs}
            found = find_primary_tensor(tree)
            if found is None:
                return
            tensor, tensor_path = found
            inferred = TensorLayout.infer(tensor, module)
            candidate_name = f"{name}::input"
            layout = TensorLayout(
                tuple(int(value) for value in tensor.shape),
                _axis_for_path(batch_axis, candidate_name, inferred.batch_axis),
                _axis_for_path(feature_axis, candidate_name, inferred.feature_axis),
            )
            append_candidate(
                InsertionCandidate(
                    name=candidate_name,
                    module_path=name,
                    position="input",
                    module_type=module.__class__.__name__,
                    output_shape=tuple(int(dim) for dim in tensor.shape),
                    dim=layout.feature_dim,
                    parameters=sum(param.numel() for param in module.parameters()),
                    source="forward-input",
                    tensor_rank=int(tensor.ndim),
                    path_depth=name.count(".") + 1,
                    tensor_path=tensor_path,
                    batch_axis=layout.batch_axis,
                    feature_axis=layout.feature_axis,
                    device=str(tensor.device),
                    dtype=str(tensor.dtype),
                    is_leaf=not any(module.children()),
                )
            )

        return _capture

    named_modules = tuple((name, module) for name, module in model.named_modules() if name)
    scannable_modules = (
        named_modules
        if sample_batch is not None
        else tuple((name, module) for name, module in named_modules if is_scannable_module(module))
    )
    try:
        for name, module in scannable_modules:
            if "input" in resolved_positions:
                hooks.append(module.register_forward_pre_hook(input_hook(name, module), with_kwargs=True))
            if "output" in resolved_positions:
                hooks.append(module.register_forward_hook(output_hook(name, module)))
        if sample_batch is not None:
            was_training = model.training
            model.eval()
            try:
                with torch.no_grad():
                    run_model(
                        model,
                        sample_batch,
                        causal=causal,
                        runtime_fields=runtime_fields,
                    )
            finally:
                model.train(was_training)
        else:
            for name, module in scannable_modules:
                dim = static_module_dim(module)
                if dim is None:
                    continue
                append_candidate(
                    InsertionCandidate(
                        name=name,
                        module_path=name,
                        position="output",
                        module_type=module.__class__.__name__,
                        output_shape=(),
                        dim=dim,
                        parameters=sum(param.numel() for param in module.parameters()),
                        source="static",
                        tensor_rank=None,
                        path_depth=name.count(".") + 1,
                    )
                )
    finally:
        for handle in hooks:
            handle.remove()

    params = list(model.parameters())
    first = next((param for param in params), None)
    return ScanReport(
        candidates=tuple(candidates),
        total_parameters=sum(param.numel() for param in params),
        trainable_parameters=sum(param.numel() for param in params if param.requires_grad),
        device=str(first.device) if first is not None else "cpu",
        dtype=str(first.dtype) if first is not None else "unknown",
        batch_schema=infer_batch_schema(sample_batch),
        scanned_modules=len(scannable_modules),
        candidate_events=candidate_events,
        duplicate_events=duplicate_events,
    )


def is_scannable_module(module: nn.Module) -> bool:
    return isinstance(module, SCANNABLE_MODULE_TYPES)


def static_module_dim(module: nn.Module) -> int | None:
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        return int(module.num_features)
    if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
        return int(module.out_channels)
    if isinstance(module, nn.GroupNorm):
        return int(module.num_channels)
    if isinstance(module, nn.Linear):
        return int(module.out_features)
    if isinstance(module, nn.Embedding):
        return int(module.embedding_dim)
    if isinstance(module, nn.LayerNorm):
        shape = module.normalized_shape
        return int(shape[-1]) if isinstance(shape, tuple) else int(shape)
    if isinstance(module, nn.MultiheadAttention):
        return int(module.embed_dim)
    if isinstance(module, (nn.RNN, nn.LSTM, nn.GRU)):
        directions = 2 if bool(getattr(module, "bidirectional", False)) else 1
        return int(module.hidden_size) * directions
    return None


def _axis_for_path(
    value: int | Mapping[str, int] | None,
    path: str,
    default: int,
) -> int:
    if value is None:
        return default
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, Mapping):
        matches = [axis for pattern, axis in value.items() if fnmatch.fnmatchcase(path, pattern)]
        if not matches:
            return default
        if len(matches) > 1 and len(set(matches)) > 1:
            raise ValueError(f"conflicting tensor-axis overrides for module {path!r}")
        axis = matches[-1]
        if isinstance(axis, int) and not isinstance(axis, bool):
            return axis
    raise TypeError("tensor-axis override must be an integer or mapping of glob patterns to integers")


def _normalize_positions(value: str | tuple[str, ...]) -> tuple[str, ...]:
    positions = (value,) if isinstance(value, str) else tuple(value)
    if not positions:
        raise ValueError("positions must contain 'input' and/or 'output'")
    invalid = set(positions) - {"input", "output"}
    if invalid:
        raise ValueError(f"unknown tensor boundary positions: {sorted(invalid)}")
    return tuple(dict.fromkeys(positions))
