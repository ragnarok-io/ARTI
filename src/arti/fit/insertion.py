"""Adapter insertion utilities."""

from __future__ import annotations

import fnmatch
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Iterator

import torch
import torch.nn as nn
from torch import Tensor

from ..blocks import ARTIResidualBlock
from .profiles import AdapterProfile
from .runtime import current_context
from .scales import AdapterScale
from .scanner import STRUCTURED_TENSOR_KEYS, InsertionCandidate


@dataclass(frozen=True)
class InsertionSpec:
    where: tuple[str, ...] = ("*",)
    every: int = 1
    freeze_base: bool = True
    max_adapters: int | None = None
    max_extra_params: int | None = None
    identity_gate: bool = False
    require_runtime_context: bool = False


@dataclass(frozen=True)
class InsertedAdapter:
    name: str
    dim: int
    parameters: int
    profile: str
    scale: str


@dataclass(frozen=True)
class AdapterInsertionPlan:
    """Dry-run plan for adapter insertion without mutating the model."""

    selected: tuple[InsertedAdapter, ...]
    skipped_budget: tuple[InsertedAdapter, ...]
    spec: InsertionSpec

    @property
    def adapter_parameters(self) -> int:
        return sum(adapter.parameters for adapter in self.selected)

    def to_dict(self) -> dict[str, object]:
        return {
            "selected": [adapter.__dict__ for adapter in self.selected],
            "skipped_budget": [adapter.__dict__ for adapter in self.skipped_budget],
            "adapter_parameters": self.adapter_parameters,
            "spec": {
                "where": list(self.spec.where),
                "every": self.spec.every,
                "freeze_base": self.spec.freeze_base,
                "max_adapters": self.spec.max_adapters,
                "max_extra_params": self.spec.max_extra_params,
                "identity_gate": self.spec.identity_gate,
                "require_runtime_context": self.spec.require_runtime_context,
            },
        }


class ARTIAdapterWrapper(nn.Module):
    """Wrap an existing module and adapt tensor outputs with ARTI."""

    def __init__(
        self,
        base: nn.Module,
        adapter: ARTIResidualBlock,
        *,
        freeze_base: bool = True,
        identity_gate: bool = False,
        require_runtime_context: bool = False,
    ) -> None:
        super().__init__()
        self.base = base
        self.adapter = adapter
        self.enabled = True
        self.output_gate = nn.Parameter(torch.zeros(())) if identity_gate else None
        self.require_runtime_context = require_runtime_context
        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False

    def forward(self, *args, **kwargs):
        output = self.base(*args, **kwargs)
        if not self.enabled:
            return output
        if isinstance(output, tuple):
            first = output[0]
            if isinstance(first, Tensor) and first.is_floating_point() and first.ndim in {2, 3, 4}:
                return (self._adapt(first), *output[1:])
            return output
        if isinstance(output, Mapping):
            for key in STRUCTURED_TENSOR_KEYS:
                value = output.get(key)
                if isinstance(value, Tensor) and value.is_floating_point() and value.ndim in {2, 3, 4}:
                    return replace_mapping_value(output, key, self._adapt(value))
            return output
        if isinstance(output, Tensor) and output.is_floating_point() and output.ndim in {2, 3, 4}:
            return self._adapt(output)
        return output

    def _adapt(self, tensor: Tensor) -> Tensor:
        if tensor.ndim == 4:
            batch, channels, height, width = tensor.shape
            sequence = tensor.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
            adapted = self._adapt_sequence(sequence)
            return adapted.reshape(batch, height, width, channels).permute(0, 3, 1, 2).contiguous()
        if tensor.ndim == 3 and isinstance(self.base, (nn.RNN, nn.LSTM, nn.GRU)) and not bool(getattr(self.base, "batch_first", False)):
            adapted = self._adapt_sequence(tensor.transpose(0, 1))
            return adapted.transpose(0, 1).contiguous()
        return self._adapt_sequence(tensor)

    def _adapt_sequence(self, tensor: Tensor) -> Tensor:
        context = current_context()
        if self.require_runtime_context and context is None:
            raise ValueError("ARTI pretrained phase adapter requires workflow.context(...) or arti_context= at runtime")
        if self.output_gate is not None and not self.training and bool(torch.count_nonzero(self.output_gate.detach()) == 0):
            return tensor
        if context is None or tensor.ndim != 3:
            return self._blend(tensor, self.adapter(tensor))
        kwargs = {}
        if context.mask is not None and context.mask.shape == tensor.shape[:2]:
            kwargs["mask"] = context.mask.to(device=tensor.device, dtype=tensor.dtype)
        if context.visibility is not None and context.visibility.shape == (*tensor.shape[:2], tensor.shape[1]):
            kwargs["visibility"] = context.visibility.to(device=tensor.device)
        layer = getattr(self.adapter, "layer", None)
        config = getattr(layer, "config", None)
        coord_dim = int(getattr(config, "coord_dim", 0))
        if coord_dim > 0:
            if context.coord is None or context.coord.shape != (*tensor.shape[:2], coord_dim):
                raise ValueError(f"observer-phase ARTI adapter requires coord with shape {(*tensor.shape[:2], coord_dim)}")
            kwargs["coord"] = context.coord.to(device=tensor.device, dtype=tensor.dtype)
            if context.observer_coord is not None:
                kwargs["observer_coord"] = context.observer_coord.to(device=tensor.device, dtype=tensor.dtype)
            if getattr(config, "coord_frame_mode", "none") == "operator_bank":
                if context.frame_operators is None:
                    raise ValueError("operator_bank ARTI adapter requires frame_operators in the runtime context")
                kwargs["frame_operators"] = context.frame_operators.to(device=tensor.device, dtype=tensor.dtype)
        return self._blend(tensor, self.adapter(tensor, **kwargs))

    def _blend(self, original: Tensor, adapted: Tensor) -> Tensor:
        if self.output_gate is None:
            return adapted
        return original + torch.tanh(self.output_gate).to(dtype=original.dtype) * (adapted - original)


def iter_adapter_wrappers(model: nn.Module) -> Iterator[ARTIAdapterWrapper]:
    for module in model.modules():
        if isinstance(module, ARTIAdapterWrapper):
            yield module


def replace_mapping_value(output: Mapping, key: str, value: Tensor):
    if type(output) is dict:
        replaced = dict(output)
        replaced[key] = value
        return replaced
    try:
        replaced = output.copy()
        replaced[key] = value
        return replaced
    except Exception:
        replaced = dict(output)
        replaced[key] = value
        return replaced


@contextmanager
def adapters_enabled(model: nn.Module, enabled: bool) -> Iterator[None]:
    wrappers = list(iter_adapter_wrappers(model))
    previous = [wrapper.enabled for wrapper in wrappers]
    for wrapper in wrappers:
        wrapper.enabled = enabled
    try:
        yield
    finally:
        for wrapper, value in zip(wrappers, previous):
            wrapper.enabled = value


def select_candidates(candidates: tuple[InsertionCandidate, ...], spec: InsertionSpec) -> tuple[InsertionCandidate, ...]:
    matched = []
    seen = set()
    for pattern in spec.where:
        for candidate in candidates:
            if candidate.name not in seen and fnmatch.fnmatch(candidate.name, pattern):
                matched.append(candidate)
                seen.add(candidate.name)
    selected = matched if spec.every <= 1 else [candidate for index, candidate in enumerate(matched) if index % spec.every == 0]
    if spec.max_adapters is not None:
        selected = selected[: spec.max_adapters]
    return tuple(selected)


def get_parent_module(model: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parent = model
    parts = qualified_name.split(".")
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)) else getattr(parent, part)
    return parent, parts[-1]


def set_child_module(parent: nn.Module, child_name: str, module: nn.Module) -> None:
    if child_name.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(child_name)] = module
    else:
        setattr(parent, child_name, module)


def make_adapter(candidate: InsertionCandidate, profile: AdapterProfile, scale: AdapterScale) -> ARTIResidualBlock:
    coord_dim = profile.coord_dim if profile.observer_phase else 0
    hidden_dim = max(1, int(round(candidate.dim * scale.hidden_multiplier)))
    return ARTIResidualBlock(
        dim=candidate.dim,
        coord_dim=coord_dim,
        hidden_dim=hidden_dim,
        operator_count=scale.operator_count,
        interface_slots=scale.interface_slots,
        recall_slots=scale.recall_slots,
        recall_steps=scale.recall_steps,
        recall_activation=scale.recall_activation,
        coord_frame_mode=profile.coord_frame_mode if profile.observer_phase else "none",
    )


def plan_adapters(
    candidates: tuple[InsertionCandidate, ...],
    spec: InsertionSpec,
    profile: AdapterProfile,
    scale: AdapterScale,
    *,
    scale_name: str,
) -> AdapterInsertionPlan:
    selected = []
    skipped_budget = []
    used_params = 0
    for candidate in select_candidates(candidates, spec):
        adapter = make_adapter(candidate, profile, scale)
        adapter_params = sum(param.numel() for param in adapter.parameters()) + (1 if spec.identity_gate else 0)
        planned = InsertedAdapter(
            name=candidate.name,
            dim=candidate.dim,
            parameters=adapter_params,
            profile=profile.name,
            scale=scale_name,
        )
        if spec.max_extra_params is not None and used_params + adapter_params > spec.max_extra_params:
            skipped_budget.append(planned)
            continue
        selected.append(planned)
        used_params += adapter_params
    return AdapterInsertionPlan(selected=tuple(selected), skipped_budget=tuple(skipped_budget), spec=spec)


def insert_adapters(
    model: nn.Module,
    candidates: tuple[InsertionCandidate, ...],
    spec: InsertionSpec,
    profile: AdapterProfile,
    scale: AdapterScale,
    *,
    scale_name: str,
) -> tuple[InsertedAdapter, ...]:
    inserted = []
    planned = plan_adapters(candidates, spec, profile, scale, scale_name=scale_name)
    planned_names = {adapter.name: adapter for adapter in planned.selected}
    for candidate in select_candidates(candidates, spec):
        if candidate.name not in planned_names:
            continue
        parent, child_name = get_parent_module(model, candidate.name)
        base = parent[int(child_name)] if child_name.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)) else getattr(parent, child_name)
        adapter = make_adapter(candidate, profile, scale)
        wrapper = ARTIAdapterWrapper(
            base,
            adapter,
            freeze_base=spec.freeze_base,
            identity_gate=spec.identity_gate,
            require_runtime_context=spec.require_runtime_context,
        )
        reference = next((parameter for parameter in base.parameters() if parameter.is_floating_point()), None)
        if reference is not None:
            wrapper.to(device=reference.device, dtype=reference.dtype)
        set_child_module(parent, child_name, wrapper)
        inserted.append(planned_names[candidate.name])
    return tuple(inserted)
