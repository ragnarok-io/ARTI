"""Federated-program-backed ARTI layers attached to arbitrary PyTorch paths."""

from __future__ import annotations

import copy
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any, Callable, Iterable, Mapping

import torch
import torch.nn as nn
from torch import Tensor

from .arti_layer import ARTILayer
from .federal_layer import ProgramLayerResult
from .fit.insertion import get_parent_module, set_child_module
from .fit.scanner import run_model
from .tensor_boundary import TensorLayout, find_primary_tensor, replace_tensor_at_path


LayerFactory = Callable[["AttachedARTILayerSpec"], ARTILayer]


@dataclass(frozen=True)
class AttachedARTILayerSpec:
    """Serializable host-boundary declaration for one ARTILayer@3 instance."""

    path: str
    dim: int | None = None
    batch_axis: int | None = None
    feature_axis: int | None = None
    program_contract_fingerprint: str | None = None

    def __post_init__(self) -> None:
        if not self.path:
            raise ValueError("path must not be empty")
        if self.dim is not None and self.dim <= 0:
            raise ValueError("dim must be positive when provided")
        for name in ("batch_axis", "feature_axis"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise TypeError(f"{name} must be an integer or None")


@dataclass(frozen=True)
class AttachedARTILayerConfig:
    """Ordered ARTI layer attachment topology."""

    layers: tuple[AttachedARTILayerSpec, ...]
    freeze_backbone: bool = True

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("layers must not be empty")
        paths = tuple(layer.path for layer in self.layers)
        if len(set(paths)) != len(paths):
            raise ValueError("layer paths must be unique")

    @property
    def paths(self) -> tuple[str, ...]:
        return tuple(layer.path for layer in self.layers)


class AttachedARTILayer(nn.Module):
    """Preserve a host layer's output tree while applying ARTILayer@3."""

    output_semantics = "layer_output"

    def __init__(self, base: nn.Module, layer: ARTILayer, spec: AttachedARTILayerSpec) -> None:
        super().__init__()
        self.base = base
        self.layer = layer
        self.spec = spec
        self.layer_path = spec.path
        self.batch_axis = spec.batch_axis
        self.feature_axis = spec.feature_axis
        self.enabled = True
        self.capture = False
        self._delta_gate = None
        self.last_pre: Tensor | None = None
        self.last_delta: Tensor | None = None
        self.last_post: Tensor | None = None
        self.last_result: ProgramLayerResult | None = None

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        output = self.base(*args, **kwargs)
        found = find_primary_tensor(output)
        if found is None:
            return output
        tensor, output_path = found
        inferred = TensorLayout.infer(tensor, self.base)
        layout = TensorLayout(
            tuple(int(value) for value in tensor.shape),
            inferred.batch_axis if self.batch_axis is None else self.batch_axis,
            inferred.feature_axis if self.feature_axis is None else self.feature_axis,
        )
        sequence = layout.pack(tensor)
        result = None
        if self.enabled:
            result = self.layer.run(sequence, mask=_compatible_mask(kwargs, sequence))
            post_sequence = result.value
            if self._delta_gate is not None:
                gate = _normalize_delta_gate(self._delta_gate(sequence), sequence)
                post_sequence = sequence + (post_sequence - sequence) * gate.unsqueeze(-1).to(sequence)
        else:
            post_sequence = sequence
        delta_sequence = post_sequence - sequence
        post = layout.restore(post_sequence)
        if self.capture:
            self.last_pre = tensor
            self.last_delta = layout.restore(delta_sequence)
            self.last_post = post
            self.last_result = result
        return replace_tensor_at_path(output, output_path, post)

    @property
    def has_delta_gate(self) -> bool:
        return self._delta_gate is not None

    def set_delta_gate(self, gate) -> None:
        if gate is not None and not callable(gate):
            raise TypeError("delta gate must be callable or None")
        self._delta_gate = gate

    def clear_trace(self) -> None:
        self.last_pre = None
        self.last_delta = None
        self.last_post = None
        self.last_result = None


class AttachedARTIModel(nn.Module):
    """Path-indexed collection of federated-program-backed ARTILayer instances."""

    def __init__(self, model: nn.Module, config: AttachedARTILayerConfig) -> None:
        super().__init__()
        self.model = model
        self.config = config
        for path in config.paths:
            if not isinstance(self._module_at(path), AttachedARTILayer):
                raise ValueError(f"layer path {path!r} is not an AttachedARTILayer")

    @classmethod
    def from_config(
        cls,
        model: nn.Module,
        config: AttachedARTILayerConfig,
        *,
        layer: ARTILayer | LayerFactory | None = None,
        sample_batch: Any | None = None,
    ) -> "AttachedARTIModel":
        specs = list(config.layers)
        unresolved = tuple(spec.path for spec in specs if spec.dim is None)
        runtime_dims = (
            runtime_path_dims(
                model,
                unresolved,
                sample_batch,
                batch_axis={spec.path: spec.batch_axis for spec in specs},
                feature_axis={spec.path: spec.feature_axis for spec in specs},
            )
            if unresolved and sample_batch is not None
            else {}
        )
        if config.freeze_backbone:
            for parameter in model.parameters():
                parameter.requires_grad_(False)
        for index, declared in enumerate(specs):
            parent, leaf = get_parent_module(model, declared.path)
            base = _child(parent, leaf)
            dim = declared.dim or runtime_dims.get(declared.path) or infer_module_dim(base)
            if dim is None:
                raise ValueError(
                    f"cannot infer hidden dimension for {declared.path!r}; pass sample_batch or dim"
                )
            instance = _make_layer(layer, replace(declared, dim=int(dim)))
            fingerprint = instance.program_contract_fingerprint
            if (
                declared.program_contract_fingerprint is not None
                and declared.program_contract_fingerprint != fingerprint
            ):
                raise ValueError(
                    f"ARTILayer program contract does not match layer {declared.path!r}"
                )
            spec = replace(
                declared,
                dim=int(dim),
                program_contract_fingerprint=fingerprint,
            )
            specs[index] = spec
            reference = next(
                (parameter for parameter in base.parameters() if parameter.is_floating_point()),
                None,
            )
            if reference is not None:
                instance.to(device=reference.device, dtype=reference.dtype)
            set_child_module(parent, leaf, AttachedARTILayer(base, instance, spec))
        return cls(
            model,
            AttachedARTILayerConfig(tuple(specs), freeze_backbone=config.freeze_backbone),
        )

    @property
    def wrappers(self) -> dict[str, AttachedARTILayer]:
        return {path: self._wrapper_at(path) for path in self.config.paths}

    def set_enabled(self, enabled: bool, *, paths: Iterable[str] | None = None) -> None:
        selected = self._validate_paths(self.config.paths if paths is None else paths)
        for path in selected:
            self._wrapper_at(path).enabled = bool(enabled)

    @contextmanager
    def enabled_layers(self, paths: Iterable[str]):
        selected = set(self._validate_paths(paths))
        previous = {path: wrapper.enabled for path, wrapper in self.wrappers.items()}
        try:
            for path, wrapper in self.wrappers.items():
                wrapper.enabled = path in selected
            yield self
        finally:
            for path, enabled in previous.items():
                self._wrapper_at(path).enabled = enabled

    @contextmanager
    def disabled(self):
        with self.enabled_layers(()) as value:
            yield value

    def diagnostics(self) -> dict[str, ProgramLayerResult]:
        return {
            path: wrapper.last_result
            for path, wrapper in self.wrappers.items()
            if wrapper.last_result is not None
        }

    def clear_traces(self) -> None:
        for wrapper in self.wrappers.values():
            wrapper.clear_trace()

    def arti_parameters(self, *, trainable_only: bool = True) -> Iterable[nn.Parameter]:
        for wrapper in self.wrappers.values():
            for parameter in wrapper.layer.parameters():
                if not trainable_only or parameter.requires_grad:
                    yield parameter

    def _module_at(self, path: str) -> nn.Module:
        module = self.model
        for part in path.split("."):
            module = _child(module, part)
        return module

    def _wrapper_at(self, path: str) -> AttachedARTILayer:
        module = self._module_at(path)
        if not isinstance(module, AttachedARTILayer):
            raise ValueError(f"layer path {path!r} is not an AttachedARTILayer")
        return module

    def _validate_paths(self, paths: Iterable[str]) -> tuple[str, ...]:
        selected = tuple(paths)
        unknown = tuple(path for path in selected if path not in self.config.paths)
        if unknown:
            raise ValueError(f"unknown ARTI layer paths: {unknown}")
        return selected


def infer_module_dim(module: nn.Module) -> int | None:
    for attribute in (
        "out_features",
        "out_channels",
        "num_channels",
        "num_features",
        "normalized_shape",
        "hidden_size",
        "embed_dim",
    ):
        value = getattr(module, attribute, None)
        if isinstance(value, int):
            return value
        if isinstance(value, (tuple, list)) and len(value) == 1:
            return int(value[0])
    return None


def runtime_path_dims(
    model: nn.Module,
    paths: Iterable[str],
    sample_batch: Any,
    *,
    batch_axis: int | Mapping[str, int] | None = None,
    feature_axis: int | Mapping[str, int] | None = None,
) -> dict[str, int]:
    observed: dict[str, int] = {}
    handles = []

    def capture(path: str):
        def hook(module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            found = find_primary_tensor(output)
            if found is None:
                return
            tensor = found[0]
            inferred = TensorLayout.infer(tensor, module)
            selected_batch = _optional_path_value(batch_axis, path)
            selected_feature = _optional_path_value(feature_axis, path)
            layout = TensorLayout(
                tuple(int(value) for value in tensor.shape),
                inferred.batch_axis if selected_batch is None else int(selected_batch),
                inferred.feature_axis if selected_feature is None else int(selected_feature),
            )
            observed[path] = layout.feature_dim

        return hook

    for path in paths:
        module = model
        for part in path.split("."):
            module = _child(module, part)
        handles.append(module.register_forward_hook(capture(path)))
    training = model.training
    try:
        model.eval()
        with torch.no_grad():
            run_model(model, sample_batch)
    finally:
        model.train(training)
        for handle in handles:
            handle.remove()
    return observed


def _make_layer(
    source: ARTILayer | LayerFactory | None,
    spec: AttachedARTILayerSpec,
) -> ARTILayer:
    if source is None:
        return ARTILayer()
    if isinstance(source, ARTILayer):
        return copy.deepcopy(source)
    if callable(source):
        result = source(spec)
        if not isinstance(result, ARTILayer):
            raise TypeError("layer factory must return ARTILayer")
        return result
    raise TypeError("layer must be ARTILayer, a layer factory, or None")


def _compatible_mask(kwargs: Mapping[str, Any], sequence: Tensor) -> Tensor | None:
    for name in ("attention_mask", "mask"):
        value = kwargs.get(name)
        if isinstance(value, Tensor) and tuple(value.shape) == tuple(sequence.shape[:2]):
            return value.to(device=sequence.device, dtype=torch.bool)
    return None


def _normalize_delta_gate(gate: Tensor, hidden: Tensor) -> Tensor:
    if not isinstance(gate, Tensor):
        raise TypeError("delta gate must return a Tensor")
    expected = hidden.shape[:-1]
    if hidden.ndim == 3 and tuple(gate.shape) == (hidden.shape[0],):
        gate = gate.unsqueeze(-1).expand(expected)
    if tuple(gate.shape) != tuple(expected):
        raise ValueError(f"delta gate returned shape {tuple(gate.shape)}; expected {tuple(expected)}")
    if not gate.is_floating_point():
        raise TypeError("delta gate must return a floating-point Tensor")
    return gate


def _optional_path_value(value: Any | Mapping[str, Any] | None, path: str) -> Any:
    if value is None:
        return None
    return value.get(path) if isinstance(value, Mapping) else value


def _child(parent: nn.Module, leaf: str) -> nn.Module:
    if leaf.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        return parent[int(leaf)]
    return getattr(parent, leaf)


__all__ = [
    "AttachedARTILayer",
    "AttachedARTILayerConfig",
    "AttachedARTILayerSpec",
    "AttachedARTIModel",
    "LayerFactory",
    "infer_module_dim",
    "runtime_path_dims",
]
