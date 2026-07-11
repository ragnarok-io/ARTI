"""Runtime inspection for ARTI modules and ordinary PyTorch compositions."""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
from torch import Tensor


@dataclass(frozen=True)
class InspectionReport:
    """Serializable shape, mechanism, parameter, and resource report."""

    module: str
    training: bool
    devices: tuple[str, ...]
    dtypes: tuple[str, ...]
    total_parameters: int
    trainable_parameters: int
    parameter_groups: dict[str, int]
    input_shapes: Any
    output_shapes: Any
    input_bytes: int
    output_bytes: int
    latency_seconds: float | None
    peak_cuda_memory_bytes: int | None
    mechanisms: dict[str, bool]
    required_inputs: dict[str, bool]
    accepted_inputs: dict[str, bool]
    workspace: dict[str, int]
    synthetic_context: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_markdown(self) -> str:
        enabled = ", ".join(name for name, value in self.mechanisms.items() if value) or "none"
        required = ", ".join(name for name, value in self.required_inputs.items() if value) or "x"
        lines = [
            f"# ARTI inspection: {self.module}",
            "",
            f"- Enabled mechanisms: {enabled}",
            f"- Required inputs: {required}",
            f"- Parameters: {self.total_parameters:,} total / {self.trainable_parameters:,} trainable",
            f"- Devices: {', '.join(self.devices) or 'parameterless'}",
            f"- Dtypes: {', '.join(self.dtypes) or 'parameterless'}",
            f"- Input shapes: `{self.input_shapes}`",
            f"- Output shapes: `{self.output_shapes}`",
        ]
        if self.latency_seconds is not None:
            lines.append(f"- Forward latency: {self.latency_seconds * 1000.0:.3f} ms")
        if self.peak_cuda_memory_bytes is not None:
            lines.append(f"- CUDA peak memory: {self.peak_cuda_memory_bytes / (1024**2):.2f} MiB")
        return "\n".join(lines) + "\n"


def inspect(module: nn.Module, example: Any | None = None, /, **forward_kwargs: Any) -> InspectionReport:
    """Inspect a module, optionally executing one non-mutating example forward."""

    if not isinstance(module, nn.Module):
        raise TypeError("module must be a torch.nn.Module")
    parameters = tuple(module.named_parameters())
    devices = tuple(sorted({str(parameter.device) for _, parameter in parameters}))
    dtypes = tuple(sorted({str(parameter.dtype).removeprefix("torch.") for _, parameter in parameters}))
    parameter_groups: dict[str, int] = {}
    for name, parameter in parameters:
        group = _parameter_group(name)
        parameter_groups[group] = parameter_groups.get(group, 0) + parameter.numel()

    explanation = _explanation(module)
    output = None
    elapsed = None
    peak_memory = None
    training = module.training
    if example is not None:
        cuda_device = _first_cuda_device(example, forward_kwargs)
        if cuda_device is not None:
            torch.cuda.synchronize(cuda_device)
            torch.cuda.reset_peak_memory_stats(cuda_device)
        started = time.perf_counter()
        try:
            module.eval()
            with torch.no_grad():
                output = _run(module, example, forward_kwargs)
            if cuda_device is not None:
                torch.cuda.synchronize(cuda_device)
                peak_memory = int(torch.cuda.max_memory_allocated(cuda_device))
            elapsed = time.perf_counter() - started
        finally:
            module.train(training)

    return InspectionReport(
        module=f"{type(module).__module__}.{type(module).__qualname__}",
        training=training,
        devices=devices,
        dtypes=dtypes,
        total_parameters=sum(parameter.numel() for _, parameter in parameters),
        trainable_parameters=sum(parameter.numel() for _, parameter in parameters if parameter.requires_grad),
        parameter_groups=dict(sorted(parameter_groups.items())),
        input_shapes=_shape_tree(example),
        output_shapes=_shape_tree(output),
        input_bytes=_tensor_bytes(example) + _tensor_bytes(forward_kwargs),
        output_bytes=_tensor_bytes(output),
        latency_seconds=elapsed,
        peak_cuda_memory_bytes=peak_memory,
        mechanisms=dict(explanation.get("mechanisms", {})),
        required_inputs=dict(explanation.get("required_inputs", {"x": True})),
        accepted_inputs=dict(explanation.get("accepted_inputs", {"x": True})),
        workspace={key: int(value) for key, value in explanation.get("capacities", {}).items() if key.endswith("slots") or key.endswith("steps")},
        synthetic_context=bool(explanation.get("synthetic_context", False)),
    )


def _run(module: nn.Module, example: Any, kwargs: Mapping[str, Any]) -> Any:
    if isinstance(example, Mapping):
        if kwargs:
            raise ValueError("forward kwargs cannot be combined with a mapping example")
        return module(**dict(example))
    if isinstance(example, tuple):
        return module(*example, **dict(kwargs))
    return module(example, **dict(kwargs))


def _explanation(module: nn.Module) -> dict[str, Any]:
    config = getattr(module, "config", None)
    if config is not None and callable(getattr(config, "explain", None)):
        return config.explain()
    return {
        "mechanisms": {},
        "required_inputs": {"x": True},
        "accepted_inputs": {"x": True},
        "capacities": {},
        "synthetic_context": False,
    }


def _parameter_group(name: str) -> str:
    if "state.phase" in name:
        return "phase"
    if "state.interface" in name:
        return "virtual_interface"
    if "state.recall" in name:
        return "recall"
    if "virtual_recall" in name:
        return "virtual_recall"
    if ".pulse" in name or name.startswith("pulse"):
        return "pulse"
    return "core"


def _shape_tree(value: Any) -> Any:
    if isinstance(value, Tensor):
        return list(value.shape)
    if isinstance(value, Mapping):
        return {str(key): _shape_tree(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_shape_tree(item) for item in value]
    if is_dataclass(value) and not isinstance(value, type):
        return {field.name: _shape_tree(getattr(value, field.name)) for field in fields(value)}
    if value is None:
        return None
    return type(value).__name__


def _tensor_bytes(value: Any) -> int:
    if isinstance(value, Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, Mapping):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(item) for item in value)
    if is_dataclass(value) and not isinstance(value, type):
        return sum(_tensor_bytes(getattr(value, field.name)) for field in fields(value))
    return 0


def _first_cuda_device(*values: Any) -> torch.device | None:
    def visit(value: Any) -> torch.device | None:
        if isinstance(value, Tensor) and value.is_cuda:
            return value.device
        if isinstance(value, Mapping):
            for item in value.values():
                found = visit(item)
                if found is not None:
                    return found
        if isinstance(value, (tuple, list)):
            for item in value:
                found = visit(item)
                if found is not None:
                    return found
        return None

    for value in values:
        found = visit(value)
        if found is not None:
            return found
    return None


__all__ = ["InspectionReport", "inspect"]
