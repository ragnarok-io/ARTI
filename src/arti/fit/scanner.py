"""PyTorch model scanning for ARTI insertion points."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor

from .batch_schema import BatchSchema, infer_batch_schema
from .runtime import RuntimeFieldConfig, adapter_context, runtime_keys, runtime_kwargs_from_batch


@dataclass(frozen=True)
class InsertionCandidate:
    name: str
    module_type: str
    output_shape: tuple[int, ...]
    dim: int
    parameters: int
    source: str = "forward"
    tensor_rank: int | None = None
    path_depth: int = 0


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


STRUCTURED_TENSOR_KEYS = ("last_hidden_state", "hidden_state", "logits", "output")


def tensor_from_module_output(output: Any) -> Tensor | None:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    if isinstance(output, Mapping):
        for key in STRUCTURED_TENSOR_KEYS:
            value = output.get(key)
            if torch.is_tensor(value):
                return value
    return None


SCANNABLE_MODULE_TYPES = (
    nn.BatchNorm2d,
    nn.Conv2d,
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


def scan_model(model: nn.Module, sample_batch: Any | None = None, *, causal: bool = False, runtime_fields: RuntimeFieldConfig | None = None) -> ScanReport:
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

    def hook(name: str, module: nn.Module):
        def _capture(_: nn.Module, __: tuple[Any, ...], output: Any) -> None:
            tensor = tensor_from_module_output(output)
            if torch.is_tensor(tensor) and tensor.is_floating_point() and tensor.ndim in {2, 3, 4}:
                dim = int(tensor.shape[1]) if tensor.ndim == 4 else int(tensor.shape[-1])
                append_candidate(
                    InsertionCandidate(
                        name=name,
                        module_type=module.__class__.__name__,
                        output_shape=tuple(int(dim) for dim in tensor.shape),
                        dim=dim,
                        parameters=sum(param.numel() for param in module.parameters()),
                        source="forward",
                        tensor_rank=int(tensor.ndim),
                        path_depth=name.count(".") + 1,
                    )
                )

        return _capture

    scannable_modules = tuple((name, module) for name, module in model.named_modules() if name and is_scannable_module(module))
    for name, module in scannable_modules:
        hooks.append(module.register_forward_hook(hook(name, module)))
    if sample_batch is not None:
        was_training = model.training
        model.eval()
        with torch.no_grad():
            run_model(model, sample_batch, causal=causal, runtime_fields=runtime_fields)
        model.train(was_training)
    else:
        for name, module in scannable_modules:
            dim = static_module_dim(module)
            if dim is None:
                continue
            append_candidate(
                InsertionCandidate(
                    name=name,
                    module_type=module.__class__.__name__,
                    output_shape=(),
                    dim=dim,
                    parameters=sum(param.numel() for param in module.parameters()),
                    source="static",
                    tensor_rank=None,
                    path_depth=name.count(".") + 1,
                )
            )
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
    if isinstance(module, nn.BatchNorm2d):
        return int(module.num_features)
    if isinstance(module, nn.Conv2d):
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
