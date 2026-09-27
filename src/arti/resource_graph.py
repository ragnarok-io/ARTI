"""Explicit tensor resources and learnable program connections.

This module deliberately separates where a tensor lives from how a Formula or
ordinary program operates on it.  A resource carries no mandatory Query or
edit loop; a connection can be direct or use a caller-supplied local condition.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, replace
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
from typing import ClassVar, Literal

import torch
from torch import Tensor, nn
from safetensors.torch import load_file, save_file

from .credit_boundary import CreditBoundary, CreditBoundaryMode, CreditStructureChoice
from .formula_v2 import (
    FormulaProgram,
    FormulaBankOperand,
    FormulaExecutionPlanV2,
    FormulaFabricV2,
    InputBinding,
    PreparedFormulaBindings,
)
from .tensor_view import AxisDescriptor, TensorIndexMap, TensorView, TensorViewPattern


_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class ResourceGraphError(ValueError):
    """Raised when a resource, connection, or graph contract is invalid."""


class ResourceGraphCompileError(ResourceGraphError):
    """Raised when a graph relation cannot be lowered as a fixed tensor path."""


@dataclass(frozen=True)
class LocalVJPResult:
    """One node-local reverse-mode result.

    A local VJP deliberately receives and returns tensors only.  It is the
    contract used by the fixed graph credit lowering below; the graph remains
    responsible for composing port relations and for accumulating shared
    parameter contributions.
    """

    input_cotangents: tuple[Tensor | None, ...]
    parameter_cotangents: tuple[Tensor | None, ...]


LocalVJP = Callable[
    [nn.Module, tuple[Tensor, ...], tuple[Tensor, ...], tuple[Tensor, ...], tuple[nn.Parameter, ...], bool],
    LocalVJPResult,
]


def autograd_local_vjp(
    module: nn.Module,
    inputs: tuple[Tensor, ...],
    outputs: tuple[Tensor, ...],
    output_cotangents: tuple[Tensor, ...],
    parameters: tuple[nn.Parameter, ...],
    create_graph: bool,
) -> LocalVJPResult:
    """Derive one declared node-local VJP with ordinary PyTorch autograd."""

    if len(outputs) != len(output_cotangents):
        raise ResourceGraphCompileError("local VJP output cotangents must match node outputs")
    active = tuple(
        (output, cotangent)
        for output, cotangent in zip(outputs, output_cotangents, strict=True)
        if output.requires_grad
    )
    if not active:
        return LocalVJPResult(
            (None,) * len(inputs),
            (None,) * len(parameters),
        )
    active_input_indices = tuple(index for index, value in enumerate(inputs) if value.requires_grad)
    active_parameter_indices = tuple(
        index for index, value in enumerate(parameters) if value.requires_grad
    )
    active_inputs = tuple(inputs[index] for index in active_input_indices)
    active_parameters = tuple(parameters[index] for index in active_parameter_indices)
    if not active_inputs and not active_parameters:
        return LocalVJPResult((None,) * len(inputs), (None,) * len(parameters))
    gradients = torch.autograd.grad(
        tuple(item[0] for item in active),
        (*active_inputs, *active_parameters),
        grad_outputs=tuple(item[1] for item in active),
        retain_graph=True,
        create_graph=create_graph,
        allow_unused=True,
    )
    input_cotangents: list[Tensor | None] = [None] * len(inputs)
    parameter_cotangents: list[Tensor | None] = [None] * len(parameters)
    for index, gradient in zip(active_input_indices, gradients[:len(active_inputs)], strict=True):
        input_cotangents[index] = gradient
    for index, gradient in zip(active_parameter_indices, gradients[len(active_inputs):], strict=True):
        parameter_cotangents[index] = gradient
    return LocalVJPResult(tuple(input_cotangents), tuple(parameter_cotangents))


class ResourceLifetime(str, Enum):
    """Storage lifetime; it does not imply trainability or gradient policy."""

    CALL = "call"
    PERSISTENT = "persistent"
    PARAMETER = "parameter"
    STATE = "state"


def _require_identifier(value: str, *, field: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ResourceGraphError(f"{field} must be an identifier")
    return value


class _CreditMaskSamples(dict[str, Tensor]):
    """Per-program-call samples, shared by nested uses and reset between loop steps."""

    def __init__(self, values: Mapping[str, Tensor] | None = None) -> None:
        super().__init__({} if values is None else values)
        self._provided = frozenset(self)

    def reset_execution(self) -> None:
        provided = {name: self[name] for name in self._provided}
        self.clear()
        self.update(provided)


def _credit_mask_map(credit_masks: Mapping[str, Tensor] | None) -> _CreditMaskSamples:
    """Validate explicit samples and preserve the current execution cache."""

    if isinstance(credit_masks, _CreditMaskSamples):
        return credit_masks
    if credit_masks is None:
        return _CreditMaskSamples()
    if not isinstance(credit_masks, Mapping):
        raise TypeError("credit_masks must be a mapping or None")
    result = dict(credit_masks)
    for connection_id, mask in result.items():
        _require_identifier(connection_id, field="credit mask connection id")
        if not isinstance(mask, Tensor) or mask.dtype is not torch.bool:
            raise TypeError("credit_masks must map connection ids to boolean Tensors")
    return _CreditMaskSamples(result)


def _execution_credit_mask(
    connection: "Connection", supplied: Tensor | None, source: Tensor,
    *, samples: _CreditMaskSamples,
) -> Tensor | None:
    boundary = connection.credit_boundary
    if supplied is not None or boundary is None or boundary.mode is not CreditBoundaryMode.BERNOULLI:
        return supplied
    cached = samples.get(connection.connection_id)
    if cached is not None:
        return cached
    if connection.connection_id in samples:
        return None
    mask = torch.rand((), device=source.device) < boundary.permeability.detach()
    samples[connection.connection_id] = mask
    return mask


def _clone_view(view: TensorView) -> TensorView:
    """Clone payload ownership without imposing a detach boundary."""

    index_map = view.index_map
    if index_map is not None and index_map.coordinates is not None:
        index_map = TensorIndexMap(
            index_map.source_axes,
            index_map.source_shape,
            index_map.target_shape,
            index_map.coordinates.clone(),
            index_map.schema_version,
        )
    return TensorView(
        view.value.clone(),
        tuple(
            AxisDescriptor(axis.name, axis.role, axis.extent, axis.origin, axis.scale)
            for axis in view.axes
        ),
        index_map=index_map,
        mask=None if view.mask is None else view.mask.clone(),
    )


def _detach_clone_view(view: TensorView) -> TensorView:
    """Clone a view for a new graph generation without retaining old autograd history."""

    value = view.value.detach().clone()
    value.requires_grad_(view.value.requires_grad)
    index_map = view.index_map
    if index_map is not None and index_map.coordinates is not None:
        index_map = TensorIndexMap(
            index_map.source_axes,
            index_map.source_shape,
            index_map.target_shape,
            index_map.coordinates.detach().clone(),
            index_map.schema_version,
        )
    return TensorView(
        value,
        tuple(
            AxisDescriptor(axis.name, axis.role, axis.extent, axis.origin, axis.scale)
            for axis in view.axes
        ),
        index_map=index_map,
        mask=None if view.mask is None else view.mask.detach().clone(),
    )


@dataclass(frozen=True)
class TensorResourceSpec:
    """Logical resource contract independent of its current backing tensor."""

    resource_id: str
    view_pattern: TensorViewPattern
    lifetime: ResourceLifetime = ResourceLifetime.CALL
    axis_capacity: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(self.resource_id, field="resource_id")
        if not isinstance(self.view_pattern, TensorViewPattern):
            raise TypeError("view_pattern must be TensorViewPattern")
        if not isinstance(self.lifetime, ResourceLifetime):
            raise TypeError("lifetime must be ResourceLifetime")
        capacity = tuple((str(axis), limit) for axis, limit in self.axis_capacity)
        if len({axis for axis, _ in capacity}) != len(capacity):
            raise ResourceGraphError("axis_capacity names must be unique")
        for axis, limit in capacity:
            _require_identifier(axis, field="axis_capacity name")
            if type(limit) is not int or limit < 0:
                raise ResourceGraphError("axis_capacity limits must be non-negative integers")
        object.__setattr__(self, "axis_capacity", tuple(sorted(capacity)))

    def validate(self, view: TensorView, *, name: str | None = None) -> None:
        self.view_pattern.validate(view, name=name or self.resource_id)
        extents = {axis.name: axis.extent for axis in view.axes}
        for axis, limit in self.axis_capacity:
            if axis not in extents:
                raise ResourceGraphError(f"{self.resource_id} does not contain capacity axis {axis!r}")
            if extents[axis] > limit:
                raise ResourceGraphError(
                    f"{self.resource_id} axis {axis!r} exceeds capacity {limit}"
                )

    def contract_config(self) -> dict[str, object]:
        return {
            "resource_id": self.resource_id,
            "view_pattern": self.view_pattern.to_dict(),
            "lifetime": self.lifetime.value,
            "axis_capacity": [[axis, limit] for axis, limit in self.axis_capacity],
        }


@dataclass(frozen=True)
class ResourceBinding:
    """Identity of the backing selected for a resource at one observation point."""

    resource_id: str
    source: Literal["default", "external"]
    epoch: int
    step_index: int

    def __post_init__(self) -> None:
        _require_identifier(self.resource_id, field="resource_id")
        if self.source not in ("default", "external"):
            raise ResourceGraphError("binding source must be 'default' or 'external'")
        if type(self.epoch) is not int or self.epoch < 0:
            raise ResourceGraphError("binding epoch must be a non-negative integer")
        if type(self.step_index) is not int or self.step_index < 0:
            raise ResourceGraphError("binding step_index must be a non-negative integer")


@dataclass(frozen=True)
class ResourceSnapshot:
    """One immutable observation of a resource backing and its logical binding."""

    view: TensorView
    binding: ResourceBinding


@dataclass(frozen=True)
class TensorResourceState:
    """In-memory save/restore payload for one logical resource.

    The state carries tensors directly so callers can choose their own artifact
    format and detached-versus-differentiable persistence policy.
    """

    spec: TensorResourceSpec
    default_view: TensorView
    active_view: TensorView
    active_source: Literal["default", "external"]
    epoch: int
    step_index: int


class TensorResource:
    """A default-backed logical tensor resource with optional hot mounting."""

    _component_reference: ClassVar[str] = "arti/tensor-resource@1"

    def __init__(self, spec: TensorResourceSpec, default_view: TensorView) -> None:
        if not isinstance(spec, TensorResourceSpec):
            raise TypeError("spec must be TensorResourceSpec")
        spec.validate(default_view, name="default_view")
        self.spec = spec
        self._default_view = _clone_view(default_view)
        self._external_view: TensorView | None = None
        self._epoch = 0
        self._step_index = 0

    @property
    def mounted(self) -> bool:
        return self._external_view is not None

    @property
    def epoch(self) -> int:
        return self._epoch

    @property
    def step_index(self) -> int:
        return self._step_index

    def resolve(self) -> ResourceSnapshot:
        source: Literal["default", "external"] = "external" if self.mounted else "default"
        view = self._external_view if self._external_view is not None else self._default_view
        return ResourceSnapshot(view, ResourceBinding(self.spec.resource_id, source, self._epoch, self._step_index))

    def mount(self, view: TensorView) -> None:
        if self.mounted:
            raise RuntimeError("an external backing is already mounted; use replace")
        self.spec.validate(view, name="external_view")
        self._external_view = view
        self._epoch += 1

    def replace(self, view: TensorView) -> None:
        if not self.mounted:
            raise RuntimeError("no external backing is mounted; use mount")
        self.spec.validate(view, name="external_view")
        self._external_view = view
        self._epoch += 1

    def detach_to_default(self) -> None:
        if self.mounted:
            self._external_view = None
            self._epoch += 1

    def advance(self, view: TensorView, *, expected_epoch: int | None = None) -> ResourceSnapshot:
        """Publish the next resource state without introducing a detach boundary."""

        if expected_epoch is not None and expected_epoch != self._epoch:
            raise ResourceGraphError("resource epoch changed before advance")
        self.spec.validate(view, name="next_view")
        if self.mounted:
            self._external_view = view
        else:
            self._default_view = view
        self._epoch += 1
        self._step_index += 1
        return self.resolve()

    def state(self) -> TensorResourceState:
        snapshot = self.resolve()
        return TensorResourceState(
            self.spec,
            _clone_view(self._default_view),
            _clone_view(snapshot.view),
            snapshot.binding.source,
            self._epoch,
            self._step_index,
        )

    @classmethod
    def restore(
        cls,
        state: TensorResourceState,
        *,
        fresh_history: bool = False,
    ) -> TensorResource:
        """Restore resource state, optionally beginning a fresh autograd generation."""

        if not isinstance(state, TensorResourceState):
            raise TypeError("state must be TensorResourceState")
        if type(fresh_history) is not bool:
            raise TypeError("fresh_history must be a bool")
        state.spec.validate(state.default_view, name="state.default_view")
        state.spec.validate(state.active_view, name="state.active_view")
        resource = cls(state.spec, state.default_view)
        clone_view = _detach_clone_view if fresh_history else _clone_view
        resource._default_view = clone_view(state.default_view)
        if state.active_source == "external":
            resource._external_view = clone_view(state.active_view)
        elif state.active_source != "default":
            raise ResourceGraphError("state active_source is invalid")
        elif tuple(state.active_view.value.shape) != tuple(resource._default_view.value.shape) or not torch.equal(
            state.active_view.value, resource._default_view.value
        ):
            resource._default_view = clone_view(state.active_view)
        resource._epoch = state.epoch
        resource._step_index = state.step_index
        return resource

    def fork(self) -> TensorResource:
        """Create an independent resource state without detaching tensor gradients."""

        return self.restore(self.state())

    def contract_config(self) -> dict[str, object]:
        """Describe logical storage without serializing its mutable payload."""

        return self.spec.contract_config()


@dataclass(frozen=True)
class AxisRange:
    """A serializable half-open range over one non-batch logical axis."""

    axis: str
    start: int = 0
    stop: int | None = None
    step: int = 1

    def __post_init__(self) -> None:
        _require_identifier(self.axis, field="axis")
        if type(self.start) is not int:
            raise ResourceGraphError("range start must be an integer")
        if self.stop is not None and type(self.stop) is not int:
            raise ResourceGraphError("range stop must be an integer or None")
        if type(self.step) is not int or self.step <= 0:
            raise ResourceGraphError("range step must be a positive integer")

    def resolve(self, extent: int) -> slice:
        return slice(self.start, self.stop, self.step)

    def contract_config(self) -> dict[str, object]:
        return {"axis": self.axis, "start": self.start, "stop": self.stop, "step": self.step}


@dataclass(frozen=True)
class ResourceView:
    """A named region of a resource, separate from both storage and operations."""

    resource_id: str
    ranges: tuple[AxisRange, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(self.resource_id, field="resource_id")
        ranges = tuple(self.ranges)
        if any(not isinstance(item, AxisRange) for item in ranges):
            raise TypeError("ranges must contain AxisRange values")
        if len({item.axis for item in ranges}) != len(ranges):
            raise ResourceGraphError("ResourceView ranges must target unique axes")
        object.__setattr__(self, "ranges", ranges)

    def _slices(self, snapshot: ResourceSnapshot) -> tuple[TensorView, tuple[slice, ...]]:
        if not isinstance(snapshot, ResourceSnapshot):
            raise TypeError("snapshot must be ResourceSnapshot")
        if snapshot.binding.resource_id != self.resource_id:
            raise ResourceGraphError("ResourceView does not match the snapshot resource")
        source = snapshot.view
        ranges = {item.axis: item for item in self.ranges}
        axes_by_name = {axis.name: (index, axis) for index, axis in enumerate(source.axes)}
        missing = ranges.keys() - axes_by_name.keys()
        if missing:
            raise ResourceGraphError(f"ResourceView names unknown axes: {sorted(missing)!r}")
        if any(axes_by_name[name][1].role == "batch" for name in ranges):
            raise ResourceGraphError("ResourceView does not slice the batch axis")
        return source, tuple(
            ranges[axis.name].resolve(axis.extent) if axis.name in ranges else slice(None)
            for axis in source.axes
        )

    def resolve(self, snapshot: ResourceSnapshot) -> TensorView:
        source, slices = self._slices(snapshot)
        value = source.value[slices]
        mask = None if source.mask is None else source.mask[slices]
        axes = []
        for index, axis in enumerate(source.axes):
            start, _stop, step = slices[index].indices(axis.extent)
            axes.append(
                AxisDescriptor(
                    axis.name,
                    axis.role,
                    int(value.shape[index]),
                    axis.origin + start * axis.scale,
                    axis.scale * step,
                )
        )
        return TensorView(value, tuple(axes), index_map=_slice_index_map(source, slices), mask=mask)

    def write(self, snapshot: ResourceSnapshot, payload: TensorView) -> TensorView:
        """Functionally replace this region while preserving the resource layout.

        Region writes are an operation performed through a connection, not a
        property of the resource itself.  The output must describe exactly the
        selected target region; a shape change therefore remains explicit at a
        full-resource connection boundary instead of being silently reshaped.
        """

        if not isinstance(payload, TensorView):
            raise TypeError("ResourceView payload must be TensorView")
        source, slices = self._slices(snapshot)
        target = self.resolve(snapshot)
        if tuple(payload.value.shape) != tuple(target.value.shape):
            raise ResourceGraphError(
                "region write payload shape must equal the declared target ResourceView"
            )
        if tuple(axis.name for axis in payload.axes) != tuple(axis.name for axis in target.axes):
            raise ResourceGraphError("region write payload axes must match the target ResourceView")
        if payload.value.device != source.value.device or payload.value.dtype != source.value.dtype:
            raise ResourceGraphError("region write payload must preserve target resource device and dtype")
        value = source.value.clone()
        value[slices] = payload.value
        if source.mask is None:
            if payload.mask is None:
                mask = None
            else:
                mask = torch.zeros_like(source.value, dtype=torch.bool)
                mask[slices] = payload.mask.to(dtype=torch.bool)
        elif payload.mask is None:
            mask = source.mask
        else:
            mask = source.mask.clone()
            mask[slices] = payload.mask.to(dtype=torch.bool)
        return TensorView(value, source.axes, index_map=source.index_map, mask=mask)

    def contract_config(self) -> dict[str, object]:
        return {
            "resource_id": self.resource_id,
            "ranges": [item.contract_config() for item in self.ranges],
        }


def _slice_index_map(source: TensorView, slices: tuple[slice, ...]) -> TensorIndexMap:
    """Keep a resource-region view tied to the source logical coordinates."""

    batch_axis = source.batch_axis
    non_batch_axes = tuple(axis for index, axis in enumerate(source.axes) if index != batch_axis)
    current_shape = tuple(axis.extent for axis in non_batch_axes)
    index_map = source.index_map
    if index_map is None:
        source_axes = tuple(axis.name for axis in non_batch_axes)
        source_shape = current_shape
        grids = torch.meshgrid(
            *(torch.arange(size, device=source.value.device, dtype=torch.int64) for size in current_shape),
            indexing="ij",
        )
        coordinates = torch.stack(grids, dim=-1)
    else:
        source_axes = index_map.source_axes
        source_shape = index_map.source_shape
        if index_map.coordinates is None:
            grids = torch.meshgrid(
                *(torch.arange(size, device=source.value.device, dtype=torch.int64) for size in current_shape),
                indexing="ij",
            )
            coordinates = torch.stack(grids, dim=-1)
        else:
            coordinates = index_map.coordinates
    coordinate_slices = tuple(
        selector for index, selector in enumerate(slices) if index != batch_axis
    )
    if coordinates.ndim == len(current_shape) + 2:
        coordinates = coordinates[(slice(None), *coordinate_slices, slice(None))]
    else:
        coordinates = coordinates[(*coordinate_slices, slice(None))]
    target_shape = tuple(
        int(size) for index, size in enumerate(source.value[slices].shape) if index != batch_axis
    )
    return TensorIndexMap(source_axes, source_shape, target_shape, coordinates)


@dataclass(frozen=True)
class ResourcePort:
    """A named graph port owned by a logical tensor resource."""

    resource_id: str
    port: str = "value"

    def __post_init__(self) -> None:
        _require_identifier(self.resource_id, field="resource_id")
        _require_identifier(self.port, field="port")

    def contract_config(self) -> dict[str, object]:
        return {"resource_id": self.resource_id, "port": self.port}


class LearnableAffineTransfer(nn.Module):
    """An input-independent, trainable direct Connection transfer.

    This is deliberately not a Query or route: one declared Connection keeps
    its endpoints and only learns the continuous transmission law
    ``gain * source + bias``.  It is useful when architecture search has
    already fixed a relation but training should still tune that relation.
    """

    def __init__(
        self,
        *,
        gain: float = 1.0,
        bias: float = 0.0,
        learnable: bool = True,
    ) -> None:
        super().__init__()
        if not isinstance(gain, (int, float)) or not isinstance(bias, (int, float)):
            raise TypeError("gain and bias must be real scalars")
        if type(learnable) is not bool:
            raise TypeError("learnable must be boolean")
        gain_value = torch.tensor(float(gain), dtype=torch.float32)
        bias_value = torch.tensor(float(bias), dtype=torch.float32)
        if learnable:
            self.gain = nn.Parameter(gain_value)
            self.bias = nn.Parameter(bias_value)
        else:
            self.register_buffer("gain", gain_value)
            self.register_buffer("bias", bias_value)
        self.learnable = learnable

    def forward(self, source: TensorView) -> TensorView:
        if not isinstance(source, TensorView):
            raise TypeError("LearnableAffineTransfer source must be TensorView")
        if not source.value.is_floating_point():
            raise ResourceGraphError("LearnableAffineTransfer requires a floating source")
        if self.gain.device != source.value.device or self.bias.device != source.value.device:
            raise ResourceGraphError(
                "LearnableAffineTransfer and its source must share a device; move the "
                "Connection or compiled plan before execution"
            )
        value = source.value * self.gain.to(dtype=source.value.dtype) + self.bias.to(
            dtype=source.value.dtype
        )
        return TensorView(value, source.axes, index_map=source.index_map, mask=source.mask)

    def contract_config(self) -> dict[str, object]:
        return {
            "gain": float(self.gain.detach().cpu()),
            "bias": float(self.bias.detach().cpu()),
            "learnable": self.learnable,
        }


class Connection(nn.Module):
    """One direct or locally conditional relation between two resource ports.

    ``transfer`` receives and returns a TensorView.  ``activation`` is optional
    and receives the caller-provided local context tensor; it returns a gate
    broadcastable to the transferred value.  No global candidate scan happens
    unless a caller deliberately supplies one in the activation module.
    """

    _component_reference: ClassVar[str] = "arti/connection@1"

    def __init__(
        self,
        connection_id: str,
        source: ResourcePort,
        destination: ResourcePort,
        *,
        source_view: ResourceView | None = None,
        destination_view: ResourceView | None = None,
        operand_views: Mapping[str, ResourceView] | None = None,
        depends_on: Sequence[str] = (),
        transfer: Callable[[TensorView], TensorView] | nn.Module | None = None,
        activation: Callable[[Tensor], Tensor] | nn.Module | None = None,
        credit_boundary: CreditBoundary | None = None,
    ) -> None:
        super().__init__()
        declared_id = _require_identifier(connection_id, field="connection_id")
        if hasattr(nn.Module, declared_id):
            raise ResourceGraphError(
                f"connection_id {declared_id!r} collides with a reserved nn.Module attribute"
            )
        if not isinstance(source, ResourcePort) or not isinstance(destination, ResourcePort):
            raise TypeError("source and destination must be ResourcePort")
        if source_view is not None and not isinstance(source_view, ResourceView):
            raise TypeError("source_view must be ResourceView or None")
        if destination_view is not None and not isinstance(destination_view, ResourceView):
            raise TypeError("destination_view must be ResourceView or None")
        if source_view is not None and source_view.resource_id != source.resource_id:
            raise ResourceGraphError("source_view must address the source resource")
        if destination_view is not None and destination_view.resource_id != destination.resource_id:
            raise ResourceGraphError("destination_view must address the destination resource")
        operands = {} if operand_views is None else dict(operand_views)
        if any(not isinstance(name, str) for name in operands):
            raise TypeError("operand view names must be strings")
        for name, operand in operands.items():
            _require_identifier(name, field="operand view name")
            if not isinstance(operand, ResourceView):
                raise TypeError("operand_views must map names to ResourceView values")
        dependencies = tuple(depends_on)
        if len(set(dependencies)) != len(dependencies) or any(
            not isinstance(connection_id, str) for connection_id in dependencies
        ):
            raise ResourceGraphError("depends_on must contain unique connection ids")
        if declared_id in dependencies:
            raise ResourceGraphError("a connection cannot depend on itself")
        for dependency in dependencies:
            _require_identifier(dependency, field="depends_on connection id")
        if transfer is not None and not callable(transfer):
            raise TypeError("transfer must be callable or None")
        if activation is not None and not callable(activation):
            raise TypeError("activation must be callable or None")
        if credit_boundary is not None and not isinstance(credit_boundary, CreditBoundary):
            raise TypeError("credit_boundary must be CreditBoundary or None")
        self.connection_id = declared_id
        self.source = source
        self.destination = destination
        self.source_view = source_view
        self.destination_view = destination_view
        self.operand_views = operands
        self.depends_on = dependencies
        self.transfer = transfer
        self.activation = activation
        self.credit_boundary = credit_boundary

    @property
    def is_conditional(self) -> bool:
        return self.activation is not None

    def forward(
        self,
        source: TensorView,
        *,
        operands: Mapping[str, TensorView] | None = None,
        context: Tensor | None = None,
        credit_mask: Tensor | None = None,
    ) -> TensorView:
        result = self.data_forward(source, operands=operands, context=context)
        if self.credit_boundary is None:
            if credit_mask is not None:
                raise ResourceGraphError("credit_mask requires a credit boundary")
            return result
        return TensorView(
            self.credit_boundary(result.value, credit_mask=credit_mask),
            result.axes,
            index_map=result.index_map,
            mask=result.mask,
        )

    def data_forward(
        self,
        source: TensorView,
        *,
        operands: Mapping[str, TensorView] | None = None,
        context: Tensor | None = None,
    ) -> TensorView:
        """Apply only the data relation, excluding its port credit rule."""

        if not isinstance(source, TensorView):
            raise TypeError("connection source must be TensorView")
        operand_values = {} if operands is None else dict(operands)
        if any(not isinstance(value, TensorView) for value in operand_values.values()):
            raise TypeError("connection operands must be TensorView values")
        if self.transfer is None:
            if operand_values:
                raise ResourceGraphError("identity connections cannot consume named operands")
            output = source
        elif isinstance(self.transfer, (FormulaTensorViewTransfer, StaticFormulaTensorViewTransfer)):
            output = self.transfer(source, operands=operand_values)
        else:
            if operand_values:
                raise ResourceGraphError(
                    "named operands require a FormulaTensorViewTransfer; use a Formula program "
                    "rather than hiding resource bindings in a custom callable"
                )
            output = self.transfer(source)
        if not isinstance(output, TensorView):
            raise TypeError("connection transfer must return TensorView")
        if self.activation is None:
            result = output
        else:
            if context is None or not isinstance(context, Tensor):
                raise ResourceGraphError("conditional connections require a local Tensor context")
            gate = self.activation(context)
            if not isinstance(gate, Tensor) or not gate.is_floating_point():
                raise TypeError("connection activation must return a floating Tensor")
            if gate.device != output.value.device:
                raise ResourceGraphError("connection activation must share the output device")
            try:
                value = output.value * gate.to(dtype=output.value.dtype)
            except RuntimeError as error:
                raise ResourceGraphError("connection activation is not broadcastable to the output") from error
            result = TensorView(value, output.axes, index_map=output.index_map, mask=output.mask)
        return result

    def apply(
        self,
        source_snapshot: ResourceSnapshot,
        destination_snapshot: ResourceSnapshot,
        *,
        resources: Mapping[str, ResourceSnapshot] | None = None,
        context: Tensor | None = None,
        credit_mask: Tensor | None = None,
    ) -> TensorView:
        """Apply this relation between explicit resource snapshots."""

        if source_snapshot.binding.resource_id != self.source.resource_id:
            raise ResourceGraphError("connection source snapshot does not match its source port")
        if destination_snapshot.binding.resource_id != self.destination.resource_id:
            raise ResourceGraphError("connection destination snapshot does not match its destination port")
        source = (
            source_snapshot.view
            if self.source_view is None
            else self.source_view.resolve(source_snapshot)
        )
        if self.operand_views:
            if resources is None:
                raise ResourceGraphError("connection operands require resource snapshots")
            operands = {}
            for name, resource_view in self.operand_views.items():
                try:
                    operand_snapshot = resources[resource_view.resource_id]
                except KeyError as error:
                    raise ResourceGraphError(
                        f"connection operand {name!r} is missing resource {resource_view.resource_id!r}"
                    ) from error
                operands[name] = resource_view.resolve(operand_snapshot)
        else:
            operands = {}
        output = self(source, operands=operands, context=context, credit_mask=credit_mask)
        return (
            output
            if self.destination_view is None
            else self.destination_view.write(destination_snapshot, output)
        )

    def contract_config(self) -> dict[str, object]:
        """Describe the declared relation without claiming generic module reload."""

        def callable_kind(value: Callable[..., object] | nn.Module | None) -> str | None:
            if value is None:
                return None
            if isinstance(value, FormulaTensorViewTransfer):
                return "formula-transfer"
            if isinstance(value, nn.Module):
                return f"module:{type(value).__module__}.{type(value).__qualname__}"
            return f"callable:{getattr(value, '__module__', '')}.{getattr(value, '__qualname__', type(value).__qualname__)}"

        transfer_config: dict[str, object] | None = None
        if isinstance(self.transfer, FormulaTensorViewTransfer):
            transfer_config = {
                "program_fingerprint": self.transfer.fabric.program.fingerprint,
                "source_input": self.transfer.source_input,
                "output_name": self.transfer.output_name,
            }
        elif isinstance(self.transfer, LearnableAffineTransfer):
            transfer_config = self.transfer.contract_config()
        return {
            "connection_id": self.connection_id,
            "source": self.source.contract_config(),
            "destination": self.destination.contract_config(),
            "source_view": None if self.source_view is None else self.source_view.contract_config(),
            "destination_view": (
                None if self.destination_view is None else self.destination_view.contract_config()
            ),
            "operand_views": {
                name: self.operand_views[name].contract_config()
                for name in sorted(self.operand_views)
            },
            "depends_on": list(self.depends_on),
            "transfer_kind": callable_kind(self.transfer),
            "transfer": transfer_config,
            "activation_kind": callable_kind(self.activation),
            "credit_boundary": (
                None if self.credit_boundary is None else self.credit_boundary.contract_config()
            ),
        }


def _validate_connection_dependencies(connections: Sequence[Connection]) -> None:
    """Reject dependency cycles while keeping scheduling ownership in the graph."""

    by_id = {connection.connection_id: connection for connection in connections}
    marks: dict[str, int] = {}

    def visit(connection_id: str) -> None:
        mark = marks.get(connection_id, 0)
        if mark == 1:
            raise ResourceGraphError("connection dependencies must not contain a cycle")
        if mark == 2:
            return
        marks[connection_id] = 1
        for dependency in by_id[connection_id].depends_on:
            visit(dependency)
        marks[connection_id] = 2

    for connection_id in by_id:
        visit(connection_id)


class FormulaTensorViewTransfer(nn.Module):
    """Use an existing typed Formula program as a TensorView transfer expression.

    The named source is supplied by the connection.  Other Formula inputs are
    either explicit static bindings or named dynamic operands supplied by the
    connection's resource views.  This adapter neither invents a Query nor
    duplicates Formula execution.
    """

    def __init__(
        self,
        fabric: FormulaFabricV2,
        *,
        source_input: str,
        output_name: str | None = None,
        static_inputs: Mapping[str, Tensor] | None = None,
        dynamic_inputs: Sequence[str] = (),
        banks: Mapping[str, FormulaBankOperand] | None = None,
        axis_roles: Mapping[str, str] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(fabric, FormulaFabricV2):
            raise TypeError("fabric must be FormulaFabricV2")
        _require_identifier(source_input, field="source_input")
        input_names = {binding.name for binding in fabric.program.bindings if isinstance(binding, InputBinding)}
        if source_input not in input_names:
            raise ResourceGraphError("source_input must name an InputBinding in the Formula program")
        output_name = fabric.program.outputs[0] if output_name is None else output_name
        if output_name not in fabric.program.outputs:
            raise ResourceGraphError("output_name must name a Formula program output")
        static = dict(static_inputs or {})
        if source_input in static:
            raise ResourceGraphError("static_inputs cannot replace source_input")
        if set(static) - (input_names - {source_input}):
            raise ResourceGraphError("static_inputs contain unknown Formula inputs")
        dynamic = tuple(dynamic_inputs)
        if len(set(dynamic)) != len(dynamic) or any(not isinstance(name, str) for name in dynamic):
            raise ResourceGraphError("dynamic_inputs must be unique Formula input names")
        if source_input in dynamic or set(dynamic) - (input_names - {source_input}):
            raise ResourceGraphError("dynamic_inputs contain unknown or source Formula inputs")
        if set(static).intersection(dynamic):
            raise ResourceGraphError("Formula inputs cannot be both static and dynamic")
        if input_names - {source_input} != set(static).union(dynamic):
            raise ResourceGraphError(
                "static_inputs and dynamic_inputs must bind every non-source Formula input"
            )
        roles = dict(axis_roles or {})
        output_axes = fabric.program.slot_types[output_name].axis_names
        if set(roles) - set(output_axes):
            raise ResourceGraphError("axis_roles contain an axis absent from Formula output")
        if sum(roles.get(axis, "batch" if axis.casefold() in {"b", "batch"} else "generic") == "batch" for axis in output_axes) != 1:
            raise ResourceGraphError("Formula output needs exactly one declared or inferred batch axis")
        self.fabric = fabric
        self.source_input = source_input
        self.output_name = output_name
        self.static_inputs = static
        self.dynamic_inputs = dynamic
        self.banks = dict(banks or {})
        self.axis_roles = roles

    def forward(
        self,
        source: TensorView,
        *,
        operands: Mapping[str, TensorView] | None = None,
    ) -> TensorView:
        if not isinstance(source, TensorView):
            raise TypeError("FormulaTensorViewTransfer source must be TensorView")
        operand_values = {} if operands is None else dict(operands)
        if set(operand_values) != set(self.dynamic_inputs):
            raise ResourceGraphError(
                "Formula connection operands must exactly bind the declared dynamic_inputs"
            )
        if any(not isinstance(value, TensorView) for value in operand_values.values()):
            raise TypeError("Formula connection operands must be TensorView values")
        inputs = {
            **self.static_inputs,
            **{name: operand_values[name].value for name in self.dynamic_inputs},
            self.source_input: source.value,
        }
        output_index = self.fabric.program.outputs.index(self.output_name)
        value = self.fabric(inputs=inputs, banks=self.banks).values[output_index]
        return self._output_view(source, value, operands=operand_values)

    def _output_view(
        self,
        source: TensorView,
        value: Tensor,
        *,
        operands: Mapping[str, TensorView],
    ) -> TensorView:
        output_type = self.fabric.program.slot_types[self.output_name]
        if not value.is_floating_point():
            raise ResourceGraphError("Formula transfers must produce a floating TensorView")
        source_axes = {axis.name: axis for axis in source.axes}
        for operand in operands.values():
            for axis in operand.axes:
                source_axes.setdefault(axis.name, axis)
        axes = tuple(
            AxisDescriptor(
                axis_name,
                self.axis_roles.get(
                    axis_name,
                    source_axes[axis_name].role
                    if axis_name in source_axes
                    else "batch" if axis_name.casefold() in {"b", "batch"} else "generic",
                ),
                int(extent),
                source_axes[axis_name].origin if axis_name in source_axes else 0.0,
                source_axes[axis_name].scale if axis_name in source_axes else 1.0,
            )
            for axis_name, extent in zip(output_type.axis_names, value.shape, strict=True)
        )
        same_layout = tuple(axis.name for axis in source.axes) == output_type.axis_names and tuple(
            source.value.shape
        ) == tuple(value.shape)
        return TensorView(
            value,
            axes,
            index_map=source.index_map if same_layout else None,
            mask=source.mask if same_layout else None,
        )

    def lower_static(self) -> StaticFormulaTensorViewTransfer:
        """Lower host-admitted fixed bindings to Formula's tensor-only plan."""

        return StaticFormulaTensorViewTransfer(self)


class StaticFormulaTensorViewTransfer(nn.Module):
    """Tensor-only lowering of one FormulaTensorViewTransfer for a static path."""

    def __init__(self, transfer: FormulaTensorViewTransfer) -> None:
        super().__init__()
        if not isinstance(transfer, FormulaTensorViewTransfer):
            raise TypeError("transfer must be FormulaTensorViewTransfer")
        self.execution_plan: FormulaExecutionPlanV2 = transfer.fabric.execution_plan()
        self.program_fingerprint = transfer.fabric.program.fingerprint
        self.binding_names = tuple(binding.name for binding in transfer.fabric.program.bindings)
        self.source_input = transfer.source_input
        self.output_name = transfer.output_name
        self.output_index = transfer.fabric.program.outputs.index(transfer.output_name)
        self.output_type = transfer.fabric.program.slot_types[transfer.output_name]
        self.axis_roles = dict(transfer.axis_roles)
        self.dynamic_inputs = transfer.dynamic_inputs
        names: list[str | None] = []
        sources: list[str] = []
        for index, binding in enumerate(transfer.fabric.program.bindings):
            if isinstance(binding, InputBinding) and binding.name == transfer.source_input:
                names.append(None)
                sources.append("source")
            elif isinstance(binding, InputBinding) and binding.name in transfer.dynamic_inputs:
                names.append(None)
                sources.append(f"dynamic:{binding.name}")
            else:
                value = (
                    transfer.static_inputs[binding.name]
                    if isinstance(binding, InputBinding)
                    else transfer.banks[binding.name].consume(binding)
                )
                name = f"_static_binding_{index}"
                if isinstance(value, nn.Parameter):
                    self.register_parameter(name, value)
                else:
                    self.register_buffer(name, value)
                names.append(name)
                sources.append("static")
        self._static_binding_names = tuple(names)
        self._binding_sources = tuple(sources)

    def forward(
        self,
        source: TensorView,
        *,
        operands: Mapping[str, TensorView] | None = None,
    ) -> TensorView:
        operand_values = {} if operands is None else dict(operands)
        if set(operand_values) != set(self.dynamic_inputs):
            raise ResourceGraphError(
                "Formula connection operands must exactly bind the declared dynamic_inputs"
            )
        if any(not isinstance(value, TensorView) for value in operand_values.values()):
            raise TypeError("Formula connection operands must be TensorView values")
        values: list[Tensor] = []
        for source_kind, name in zip(
            self._binding_sources, self._static_binding_names, strict=True
        ):
            if source_kind == "source":
                values.append(source.value)
            elif source_kind.startswith("dynamic:"):
                values.append(operand_values[source_kind.removeprefix("dynamic:")].value)
            else:
                assert name is not None
                values.append(getattr(self, name))
        prepared = PreparedFormulaBindings(
            self.program_fingerprint,
            self.binding_names,
            tuple(values),
        )
        value = self.execution_plan(prepared)[self.output_index]
        source_axes = {axis.name: axis for axis in source.axes}
        for operand in operand_values.values():
            for axis in operand.axes:
                source_axes.setdefault(axis.name, axis)
        axes = tuple(
            AxisDescriptor(
                axis_name,
                self.axis_roles.get(
                    axis_name,
                    source_axes[axis_name].role
                    if axis_name in source_axes
                    else "batch" if axis_name.casefold() in {"b", "batch"} else "generic",
                ),
                int(extent),
                source_axes[axis_name].origin if axis_name in source_axes else 0.0,
                source_axes[axis_name].scale if axis_name in source_axes else 1.0,
            )
            for axis_name, extent in zip(self.output_type.axis_names, value.shape, strict=True)
        )
        same_layout = tuple(axis.name for axis in source.axes) == self.output_type.axis_names and tuple(
            source.value.shape
        ) == tuple(value.shape)
        return TensorView(
            value,
            axes,
            index_map=source.index_map if same_layout else None,
            mask=source.mask if same_layout else None,
        )


@dataclass(frozen=True)
class ConnectionExecution:
    """One connection application with its read and published resource versions."""

    connection_id: str
    source: ResourceBinding
    destination: ResourceBinding
    destination_before: ResourceBinding
    credit_mask: Tensor | None = None
    operands: tuple[tuple[str, ResourceBinding], ...] = ()
    context: Tensor | None = None


@dataclass(frozen=True)
class ProgramNodeExecution:
    """One program-node call committed into a functional graph state."""

    node_id: str
    source: ResourceBinding
    destination: ResourceBinding
    receipt: object | None = None


@dataclass(frozen=True)
class MultiPortProgramNodeExecution:
    """One atomic multi-port node publication in a functional graph state.

    All inputs name snapshots from the same graph instant.  Outputs are only
    published after the node has returned every declared port, so a branch
    cannot observe a sibling output halfway through the same node call.
    """

    node_id: str
    inputs: tuple[tuple[str, ResourceBinding], ...]
    outputs: tuple[tuple[str, ResourceBinding], ...]
    receipt: object | None = None


@dataclass(frozen=True)
class ProgramNodeInvocation:
    """A typed program-node result before the graph publishes its output."""

    output: TensorView
    receipt: object | None = None


@dataclass(frozen=True)
class ProgramLoop:
    """A graph loop whose continuation is a batch mask resource.

    A finite maximum supports fixed-horizon compiled training. ``None`` keeps
    the horizon out of the graph contract; eager execution then uses a runtime
    host limit and records whether that limit or the program stopped the loop.
    """

    loop_id: str
    program_id: str
    continue_resource_id: str
    max_iterations: int | None
    min_iterations: int = 1

    def __post_init__(self) -> None:
        _require_identifier(self.loop_id, field="loop_id")
        _require_identifier(self.program_id, field="program_id")
        _require_identifier(self.continue_resource_id, field="continue_resource_id")
        if self.max_iterations is not None and (
            type(self.max_iterations) is not int or self.max_iterations <= 0
        ):
            raise ResourceGraphError("max_iterations must be a positive integer or None")
        if (
            type(self.min_iterations) is not int
            or self.min_iterations <= 0
            or (self.max_iterations is not None and self.min_iterations > self.max_iterations)
        ):
            raise ResourceGraphError("min_iterations must be positive and within max_iterations")

    def contract_config(self) -> dict[str, object]:
        return {
            "loop_id": self.loop_id,
            "program_id": self.program_id,
            "continue_resource_id": self.continue_resource_id,
            "max_iterations": self.max_iterations,
            "min_iterations": self.min_iterations,
        }


@dataclass(frozen=True)
class ProgramStage:
    """A declared frontier of graph steps that observes one shared snapshot."""

    step_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        step_ids = tuple(self.step_ids)
        if not step_ids or len(set(step_ids)) != len(step_ids):
            raise ResourceGraphError("ProgramStage requires unique, non-empty step ids")
        for step_id in step_ids:
            _require_identifier(step_id, field="ProgramStage step id")
        object.__setattr__(self, "step_ids", step_ids)

    def contract_config(self) -> dict[str, object]:
        return {"parallel": list(self.step_ids)}


@dataclass(frozen=True)
class ProgramRoute:
    """Choose one committed graph result from scores published by the program.

    ``all_candidates`` evaluates every pure-data path before hard selection;
    it trades extra candidate compute for fewer dynamic branch launches.
    """

    route_id: str
    score_resource_id: str
    candidates: tuple[str, ...]
    selection_scope: Literal["batch", "sample"] = "batch"
    execution_mode: Literal["sparse", "all_candidates"] = "sparse"

    def __post_init__(self) -> None:
        _require_identifier(self.route_id, field="route_id")
        _require_identifier(self.score_resource_id, field="score_resource_id")
        candidates = tuple(self.candidates)
        if len(candidates) < 2 or len(set(candidates)) != len(candidates):
            raise ResourceGraphError("ProgramRoute requires at least two distinct candidate steps")
        for candidate in candidates:
            _require_identifier(candidate, field="ProgramRoute candidate")
        if self.selection_scope not in {"batch", "sample"}:
            raise ResourceGraphError("ProgramRoute selection_scope must be 'batch' or 'sample'")
        if self.execution_mode not in {"sparse", "all_candidates"}:
            raise ResourceGraphError("ProgramRoute execution_mode must be 'sparse' or 'all_candidates'")
        object.__setattr__(self, "candidates", candidates)

    def contract_config(self) -> dict[str, object]:
        route = {
            "route_id": self.route_id,
            "score_resource_id": self.score_resource_id,
            "candidates": list(self.candidates),
        }
        if self.selection_scope != "batch":
            route["selection_scope"] = self.selection_scope
        if self.execution_mode != "sparse":
            route["execution_mode"] = self.execution_mode
        return {"route": route}


@dataclass(frozen=True)
class ProgramJoin:
    """A named-port barrier which fires after every input has a fresh arrival.

    A join is deliberately separate from :class:`ProgramStage`: a stage is a
    synchronous frontier chosen by the program declaration, while a join
    consumes independently published resource versions.  The graph state
    records the last epoch consumed at each named input port, so a producer may
    publish now and another producer may publish in a later graph invocation.
    """

    join_id: str
    node_id: str

    def __post_init__(self) -> None:
        _require_identifier(self.join_id, field="join_id")
        _require_identifier(self.node_id, field="join node_id")

    def contract_config(self) -> dict[str, object]:
        return {"join": {"join_id": self.join_id, "node_id": self.node_id, "firing": "all_new"}}


@dataclass(frozen=True)
class ProgramLoopExecution:
    """Functional result, actual dispatches, and loop termination provenance."""

    state: "ProgramGraphState"
    active_masks: tuple[Tensor, ...]
    connections: tuple[ConnectionExecution, ...]
    nodes: tuple[ProgramNodeExecution | MultiPortProgramNodeExecution, ...]
    joins: tuple[ProgramJoinExecution, ...] = ()
    dispatches: tuple[ProgramDispatch, ...] = ()
    routes: tuple[ProgramRouteExecution, ...] = ()
    termination_reason: Literal["fixed_horizon", "endogenous", "host_limit"] = "fixed_horizon"
    fate_selections: tuple[tuple[str, str], ...] = ()

    @property
    def actual_iterations(self) -> int:
        return len(self.active_masks)

    def structure_objective(
        self, final_loss: Tensor, *, baseline: Tensor | float = 0.0,
        fate_sample: ProgramFateSample | None = None,
    ) -> Tensor:
        """Score-function credit for actual route dispatches and executed node fates."""

        route_terms = tuple(
            route.structure_objective(final_loss, baseline=baseline)
            * self.active_masks[route.iteration].any().to(route.log_probability.dtype)
            for route in self.routes if route.sampled
        )
        if fate_sample is None:
            if not self.routes or len(route_terms) != len(self.routes):
                raise ResourceGraphError("loop structure credit requires sampled route dispatches")
            return torch.stack(route_terms).sum()
        if not isinstance(fate_sample, ProgramFateSample):
            raise TypeError("fate_sample must be ProgramFateSample")
        if dict(fate_sample.selections) != dict(self.fate_selections):
            raise ResourceGraphError("fate sample does not match the executed candidate selections")
        if not isinstance(final_loss, Tensor) or final_loss.numel() != 1:
            raise ResourceGraphError("joint loop structure loss must be scalar")
        baseline_value = torch.as_tensor(baseline, device=final_loss.device, dtype=final_loss.dtype)
        if baseline_value.numel() != 1:
            raise ResourceGraphError("joint loop structure baseline must be scalar")
        selected = dict(self.fate_selections)
        reached: set[str] = set()
        for node in self.nodes:
            if node.node_id not in selected:
                continue
            if not isinstance(node.receipt, Mapping) or node.receipt.get("candidate_id") != selected[node.node_id]:
                raise ResourceGraphError("executed fate receipt differs from the sampled candidate")
            reached.add(node.node_id)
        if len(fate_sample.fate_log_probabilities) != len(fate_sample.selections):
            raise ResourceGraphError("fate sample terms do not match its selections")
        advantage = final_loss.detach() - baseline_value.detach()
        fate_terms = tuple(
            advantage * log_probability
            for (node_id, _), log_probability in zip(
                fate_sample.selections, fate_sample.fate_log_probabilities, strict=True,
            ) if node_id in reached
        )
        if not route_terms and not fate_terms:
            raise ResourceGraphError("loop executed no sampled structure decision")
        return torch.stack((*route_terms, *fate_terms)).sum()


@dataclass(frozen=True)
class MultiPortProgramNodeInvocation:
    """A complete named output set produced by a multi-port program node."""

    outputs: Mapping[str, TensorView]
    receipt: object | None = None

    def __post_init__(self) -> None:
        outputs = dict(self.outputs)
        if not outputs:
            raise ResourceGraphError("multi-port node results require at least one output")
        for name, view in outputs.items():
            _require_identifier(name, field="multi-port output name")
            if not isinstance(view, TensorView):
                raise TypeError("multi-port node outputs must be TensorView values")
        object.__setattr__(self, "outputs", outputs)


class ProgramNode(nn.Module):
    """A typed executable region mounted between two graph resources.

    Nodes own local numerical behavior.  ``ProgramGraph`` owns resource
    lifetime, functional state, and publication of the returned view.  This
    keeps a direct graph edge direct while making a routed region an explicit
    graph step rather than a second execution framework.
    """

    _component_reference: ClassVar[str] = "arti/program-node@1"

    def __init__(
        self,
        node_id: str,
        *,
        input_resource_id: str,
        output_resource_id: str,
    ) -> None:
        super().__init__()
        self.node_id = _require_identifier(node_id, field="node_id")
        self.input_resource_id = _require_identifier(
            input_resource_id, field="input_resource_id"
        )
        self.output_resource_id = _require_identifier(
            output_resource_id, field="output_resource_id"
        )

    def contract_config(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "node_ref": self._component_reference,
            "input_resource_id": self.input_resource_id,
            "output_resource_id": self.output_resource_id,
        }

    def invoke(self, view: TensorView) -> ProgramNodeInvocation:
        """Evaluate one local region without publishing graph state.

        Subclasses must return a fully described ``TensorView``.  The graph
        validates it against the declared output resource before advancing the
        branch-local backing.
        """

        raise NotImplementedError("ProgramNode subclasses must implement invoke")


class MultiPortProgramNode(nn.Module):
    """A typed region with named resource inputs and outputs.

    Unlike :class:`ProgramNode`, this node is not a single tensor adapter.
    Its ports are first-class graph endpoints: one invocation reads every
    named input from one snapshot and returns every named output before the
    graph advances any destination resource.  Fan-out is therefore expressed
    by multiple output ports, while convergence remains an explicit downstream
    Formula/Fabric node rather than a hidden runtime merge.
    """

    _component_reference: ClassVar[str] = "arti/multi-port-program-node@1"

    def __init__(
        self,
        node_id: str,
        *,
        input_ports: Mapping[str, ResourcePort],
        output_ports: Mapping[str, ResourcePort],
    ) -> None:
        super().__init__()
        self.node_id = _require_identifier(node_id, field="node_id")
        inputs = dict(input_ports)
        outputs = dict(output_ports)
        if not inputs or not outputs:
            raise ResourceGraphError("multi-port nodes require at least one input and output port")
        for name, port in (*inputs.items(), *outputs.items()):
            _require_identifier(name, field="multi-port name")
            if not isinstance(port, ResourcePort):
                raise TypeError("multi-port node ports must be ResourcePort values")
        destinations = tuple((port.resource_id, port.port) for port in outputs.values())
        if len(set(destinations)) != len(destinations):
            raise ResourceGraphError(
                "multi-port node outputs must target distinct resource ports; "
                "use an explicit Formula Fabric join for convergence"
            )
        self.input_ports = inputs
        self.output_ports = outputs

    def contract_config(self) -> dict[str, object]:
        return {
            "node_id": self.node_id,
            "node_ref": self._component_reference,
            "input_ports": {
                name: self.input_ports[name].contract_config() for name in sorted(self.input_ports)
            },
            "output_ports": {
                name: self.output_ports[name].contract_config() for name in sorted(self.output_ports)
            },
        }

    def invoke_ports(
        self, inputs: Mapping[str, TensorView]
    ) -> MultiPortProgramNodeInvocation:
        """Evaluate the region without publishing any graph resource state."""

        raise NotImplementedError("MultiPortProgramNode subclasses must implement invoke_ports")


class FormulaProgramNode(MultiPortProgramNode):
    """Mount one typed Formula Fabric program as a named multi-port region.

    Formula input bindings are node input ports and Formula outputs are node
    output ports.  This is the canonical convergence primitive: several
    predecessor resources are consumed as independent operands, then the
    Formula program explicitly declares how they combine.  The graph runtime
    never infers a merge law from a shared destination.
    """

    _component_reference: ClassVar[str] = "arti/formula-program-node@1"

    def __init__(
        self,
        node_id: str,
        fabric: FormulaFabricV2,
        *,
        input_ports: Mapping[str, ResourcePort],
        output_ports: Mapping[str, ResourcePort],
        output_slots: Mapping[str, str] | None = None,
        banks: Mapping[str, FormulaBankOperand] | None = None,
        axis_roles: Mapping[str, Mapping[str, str]] | None = None,
    ) -> None:
        if not isinstance(fabric, FormulaFabricV2):
            raise TypeError("fabric must be FormulaFabricV2")
        input_names = tuple(
            binding.name for binding in fabric.program.bindings if isinstance(binding, InputBinding)
        )
        if set(input_ports) != set(input_names):
            raise ResourceGraphError("FormulaProgramNode ports must bind every Formula input exactly once")
        slots = (
            {name: name for name in output_ports}
            if output_slots is None
            else dict(output_slots)
        )
        if set(slots) != set(output_ports) or set(slots.values()) != set(fabric.program.outputs):
            raise ResourceGraphError(
                "FormulaProgramNode output_slots must bind every Formula output exactly once"
            )
        if any(not isinstance(slot, str) for slot in slots.values()):
            raise TypeError("FormulaProgramNode output_slots values must be strings")
        roles = {name: dict(mapping) for name, mapping in (axis_roles or {}).items()}
        if set(roles) - set(output_ports):
            raise ResourceGraphError("axis_roles may only name Formula output ports")
        for output_name, mapping in roles.items():
            output_axes = set(fabric.program.slot_types[slots[output_name]].axis_names)
            if set(mapping) - output_axes:
                raise ResourceGraphError("axis_roles name an axis absent from the Formula output")
        super().__init__(node_id, input_ports=input_ports, output_ports=output_ports)
        self.fabric = fabric
        self.output_slots = slots
        self.banks = dict(banks or {})
        self.axis_roles = roles

    def contract_config(self) -> dict[str, object]:
        return {
            **super().contract_config(),
            "program_fingerprint": self.fabric.program.fingerprint,
            "output_slots": {name: self.output_slots[name] for name in sorted(self.output_slots)},
            "axis_roles": {
                output: dict(self.axis_roles[output]) for output in sorted(self.axis_roles)
            },
        }

    def invoke_ports(
        self, inputs: Mapping[str, TensorView]
    ) -> MultiPortProgramNodeInvocation:
        if set(inputs) != set(self.input_ports):
            raise ResourceGraphError("FormulaProgramNode received an incomplete input port set")
        if any(not isinstance(view, TensorView) for view in inputs.values()):
            raise TypeError("FormulaProgramNode inputs must be TensorView values")
        result = self.fabric(
            inputs={name: inputs[name].value for name in self.input_ports}, banks=self.banks
        )
        source_axes = {
            axis.name: axis for view in inputs.values() for axis in view.axes
        }
        outputs: dict[str, TensorView] = {}
        formula_values = dict(zip(self.fabric.program.outputs, result.values, strict=True))
        for output_name, slot_name in self.output_slots.items():
            value = formula_values[slot_name]
            output_type = self.fabric.program.slot_types[slot_name]
            roles = self.axis_roles.get(output_name, {})
            axes = tuple(
                AxisDescriptor(
                    axis_name,
                    roles.get(
                        axis_name,
                        source_axes[axis_name].role
                        if axis_name in source_axes
                        else "batch" if axis_name.casefold() in {"b", "batch"} else "generic",
                    ),
                    int(extent),
                    source_axes[axis_name].origin if axis_name in source_axes else 0.0,
                    source_axes[axis_name].scale if axis_name in source_axes else 1.0,
                )
                for axis_name, extent in zip(output_type.axis_names, value.shape, strict=True)
            )
            outputs[output_name] = TensorView(value, axes)
        return MultiPortProgramNodeInvocation(outputs)


@dataclass(frozen=True)
class FabricNodeSpecialization:
    """The frozen single-program result selected from a differentiable node."""

    candidate_id: str
    node: MultiPortProgramNode


@dataclass(frozen=True)
class DifferentiableFateLosses:
    """One completed hard-candidate evaluation at a differentiable node.

    ``losses`` are scalar downstream task losses produced after each candidate
    has executed as an ordinary single path.  They are deliberately distinct
    from a loss evaluated after numerically blending candidate outputs.
    """

    candidate_ids: tuple[str, ...]
    losses: tuple[Tensor, ...]

    def __post_init__(self) -> None:
        if not self.candidate_ids or len(self.candidate_ids) != len(self.losses):
            raise ResourceGraphError("differentiable fate losses must cover each candidate exactly once")
        devices: set[torch.device] = set()
        for candidate_id, loss in zip(self.candidate_ids, self.losses, strict=True):
            _require_identifier(candidate_id, field="candidate_id")
            if not isinstance(loss, Tensor) or loss.numel() != 1:
                raise ResourceGraphError(
                    f"differentiable fate loss for {candidate_id!r} must be a scalar Tensor"
                )
            devices.add(loss.device)
        if len(devices) != 1:
            raise ResourceGraphError("differentiable fate losses must share a device")

    def as_dict(self) -> dict[str, Tensor]:
        return dict(zip(self.candidate_ids, self.losses, strict=True))


class DifferentiableFabricNode(MultiPortProgramNode):
    """A shared-port node whose compatible pure-data regions compete by gradient.

    This is a node-level architecture variable rather than a Formula operand:
    every candidate is a complete Fabric region with the same public ports.
    While it is trainable, pure data candidates execute into their own values
    and are blended by a learned structural distribution.  ``specialize``
    removes that distribution and returns the selected ordinary node.  Formula
    candidates remain eligible for static program lowering; a registered
    custom module may provide its own lowering separately.

    Effectful execution classes (arrival joins, loops, resource writes, and
    neural-plasticity effects) intentionally do not enter this numerical blend;
    they retain their own explicit program-graph semantics.
    """

    _component_reference: ClassVar[str] = "arti/differentiable-fabric-node@1"

    def __init__(
        self,
        node_id: str,
        candidates: Mapping[str, MultiPortProgramNode],
        *,
        temperature: float = 1.0,
    ) -> None:
        candidate_map = dict(candidates)
        if len(candidate_map) < 2:
            raise ResourceGraphError("DifferentiableFabricNode requires at least two candidates")
        if not isinstance(temperature, (int, float)) or float(temperature) <= 0.0:
            raise ValueError("temperature must be a positive real scalar")
        for candidate_id, candidate in candidate_map.items():
            _require_identifier(candidate_id, field="candidate_id")
            if not isinstance(candidate, MultiPortProgramNode):
                raise TypeError(
                    "DifferentiableFabricNode candidates must be MultiPortProgramNode values"
                )
        reference = next(iter(candidate_map.values()))
        for candidate in candidate_map.values():
            if candidate.input_ports != reference.input_ports or candidate.output_ports != reference.output_ports:
                raise ResourceGraphError(
                    "DifferentiableFabricNode candidates must share identical named port bindings"
                )
        super().__init__(
            node_id,
            input_ports=reference.input_ports,
            output_ports=reference.output_ports,
        )
        self.candidate_ids = tuple(candidate_map)
        self._candidate_names = tuple(f"candidate_{index}" for index in range(len(self.candidate_ids)))
        self._candidate_modules = nn.ModuleDict(
            {
                module_name: candidate_map[candidate_id]
                for candidate_id, module_name in zip(
                    self.candidate_ids, self._candidate_names, strict=True
                )
            }
        )
        self.logits = nn.Parameter(torch.zeros(len(self.candidate_ids), dtype=torch.float32))
        self.temperature = float(temperature)

    def candidate(self, candidate_id: str) -> MultiPortProgramNode:
        try:
            index = self.candidate_ids.index(candidate_id)
        except ValueError as error:
            raise ResourceGraphError(f"unknown DifferentiableFabricNode candidate {candidate_id!r}") from error
        return self._candidate_modules[self._candidate_names[index]]

    def contract_config(self) -> dict[str, object]:
        return {
            **super().contract_config(),
            "candidate_ids": list(self.candidate_ids),
            "candidates": {
                candidate_id: self.candidate(candidate_id).contract_config()
                for candidate_id in self.candidate_ids
            },
            "temperature": self.temperature,
        }

    def probabilities(self) -> Tensor:
        """Return the trainable, input-independent node-kind distribution."""

        return torch.softmax(self.logits / self.temperature, dim=0)

    def invoke_candidate_ports(
        self, candidate_id: str, inputs: Mapping[str, TensorView]
    ) -> MultiPortProgramNodeInvocation:
        """Run one fate as its actual single-path node execution."""

        if set(inputs) != set(self.input_ports):
            raise ResourceGraphError("DifferentiableFabricNode received an incomplete input port set")
        invocation = self.candidate(candidate_id).invoke_ports(inputs)
        if set(invocation.outputs) != set(self.output_ports):
            raise ResourceGraphError("DifferentiableFabricNode candidate outputs do not match its ports")
        return MultiPortProgramNodeInvocation(
            invocation.outputs,
            {
                "candidate_id": candidate_id,
                "candidate_receipt": invocation.receipt,
            },
        )

    def fate_losses(self, losses: Mapping[str, Tensor]) -> DifferentiableFateLosses:
        """Validate completed downstream losses in the node's canonical order."""

        if set(losses) != set(self.candidate_ids):
            raise ResourceGraphError("differentiable fate losses must name every candidate exactly once")
        return DifferentiableFateLosses(
            self.candidate_ids,
            tuple(losses[candidate_id] for candidate_id in self.candidate_ids),
        )

    def discrete_structure_objective(self, losses: Mapping[str, Tensor]) -> Tensor:
        """Learn fate logits from hard-candidate downstream losses.

        Candidate losses are detached because this objective estimates the
        effect of selecting an implementation.  Candidate weights are trained
        through a separately declared exposure objective.
        """

        evaluated = self.fate_losses(losses)
        probabilities = self.probabilities().to(
            device=evaluated.losses[0].device,
            dtype=evaluated.losses[0].dtype,
        )
        stacked = torch.stack(tuple(loss.detach() for loss in evaluated.losses))
        return (probabilities * stacked).sum()

    def candidate_training_objective(
        self,
        losses: Mapping[str, Tensor],
        *,
        exposure: Mapping[str, float | Tensor] | None = None,
    ) -> Tensor:
        """Train candidate parameters under an explicit exposure distribution.

        The default gives every evaluated fate equal training opportunity.  A
        caller may instead provide a normalized exposure distribution, for
        example one derived from detached structural probabilities plus an
        exploration floor.  Exposure never supplies gradients to logits.
        """

        evaluated = self.fate_losses(losses)
        reference = evaluated.losses[0]
        if exposure is None:
            weights = torch.full(
                (len(self.candidate_ids),),
                1.0 / len(self.candidate_ids),
                device=reference.device,
                dtype=reference.dtype,
            )
        else:
            if set(exposure) != set(self.candidate_ids):
                raise ResourceGraphError("differentiable fate exposure must name every candidate exactly once")
            weights = torch.stack(
                tuple(
                    torch.as_tensor(exposure[candidate_id], device=reference.device, dtype=reference.dtype)
                    for candidate_id in self.candidate_ids
                )
            )
            if any(weight.numel() != 1 for weight in weights):
                raise ResourceGraphError("differentiable fate exposure values must be scalar")
            if not torch.isfinite(weights).all() or bool((weights < 0).any()):
                raise ResourceGraphError("differentiable fate exposure must be finite and non-negative")
            total = weights.sum()
            if not bool(total > 0):
                raise ResourceGraphError("differentiable fate exposure must have positive total mass")
            weights = weights / total
        return (weights.detach() * torch.stack(evaluated.losses)).sum()

    def paired_credit_structure_objective(
        self,
        losses: Mapping[str, Tensor],
        *,
        choice: CreditStructureChoice,
        direct_candidate_id: str,
        boundary_candidate_id: str,
    ) -> Tensor:
        """Apply a credit-boundary choice to two completed hard-fate outcomes.

        Direct and membrane candidates share their forward value but differ in
        their functional update and subsequent query loss.  This adapter keeps
        that delayed consequence in the same candidate-loss contract used by
        ordinary Formula fates.
        """

        if not isinstance(choice, CreditStructureChoice):
            raise TypeError("choice must be CreditStructureChoice")
        evaluated = self.fate_losses(losses).as_dict()
        if direct_candidate_id == boundary_candidate_id:
            raise ResourceGraphError("direct and boundary candidates must differ")
        if direct_candidate_id not in evaluated or boundary_candidate_id not in evaluated:
            raise ResourceGraphError("paired credit candidates must name node candidates")
        return choice.expected_query_loss(
            evaluated[direct_candidate_id], evaluated[boundary_candidate_id]
        )

    def invoke_ports(
        self, inputs: Mapping[str, TensorView]
    ) -> MultiPortProgramNodeInvocation:
        if set(inputs) != set(self.input_ports):
            raise ResourceGraphError("DifferentiableFabricNode received an incomplete input port set")
        probabilities = self.probabilities()
        invocations = tuple(self.invoke_candidate_ports(candidate_id, inputs) for candidate_id in self.candidate_ids)
        if any(set(invocation.outputs) != set(self.output_ports) for invocation in invocations):
            raise ResourceGraphError("DifferentiableFabricNode candidate outputs do not match its ports")
        outputs: dict[str, TensorView] = {}
        for output_name in self.output_ports:
            candidates = tuple(invocation.outputs[output_name] for invocation in invocations)
            reference = candidates[0]
            if any(
                candidate.value.shape != reference.value.shape or candidate.axes != reference.axes
                for candidate in candidates[1:]
            ):
                raise ResourceGraphError(
                    "DifferentiableFabricNode candidates must produce one compatible Tensor ABI per output port"
                )
            stacked = torch.stack(tuple(candidate.value for candidate in candidates), dim=0)
            weights = probabilities.to(dtype=stacked.dtype).reshape(
                (len(candidates),) + (1,) * reference.value.ndim
            )
            outputs[output_name] = TensorView(
                (stacked * weights).sum(dim=0), reference.axes
            )
        return MultiPortProgramNodeInvocation(
            outputs,
            {
                "candidate_ids": self.candidate_ids,
                "probabilities": probabilities,
            },
        )

    def specialize(self, candidate_id: str | None = None) -> FabricNodeSpecialization:
        """Freeze an explicit fate, or the highest-logit fate when omitted."""

        if candidate_id is None:
            index = int(torch.argmax(self.logits.detach()).item())
            candidate_id = self.candidate_ids[index]
        elif candidate_id not in self.candidate_ids:
            raise ResourceGraphError(f"unknown DifferentiableFabricNode candidate {candidate_id!r}")
        candidate = self.candidate(candidate_id)
        node = deepcopy(candidate)
        node.node_id = self.node_id
        return FabricNodeSpecialization(candidate_id, node)


class StaticFormulaProgramNode(nn.Module):
    """Tensor-only lowering of one frozen :class:`FormulaProgramNode`."""

    def __init__(self, node: FormulaProgramNode) -> None:
        super().__init__()
        if not isinstance(node, FormulaProgramNode):
            raise TypeError("node must be FormulaProgramNode")
        self.execution_plan: FormulaExecutionPlanV2 = node.fabric.execution_plan()
        self.program_fingerprint = node.fabric.program.fingerprint
        self.binding_names = tuple(binding.name for binding in node.fabric.program.bindings)
        self.input_names = tuple(node.input_ports)
        self.output_names = tuple(node.output_ports)
        self._output_indices = tuple(
            node.fabric.program.outputs.index(node.output_slots[name])
            for name in self.output_names
        )
        names: list[str | None] = []
        sources: list[str] = []
        for index, binding in enumerate(node.fabric.program.bindings):
            if isinstance(binding, InputBinding):
                names.append(None)
                sources.append(f"input:{binding.name}")
                continue
            value = node.banks[binding.name].consume(binding)
            name = f"_static_binding_{index}"
            if isinstance(value, nn.Parameter):
                self.register_parameter(name, value)
            else:
                self.register_buffer(name, value)
            names.append(name)
            sources.append("static")
        self._static_binding_names = tuple(names)
        self._binding_sources = tuple(sources)

    def forward(self, *inputs: Tensor) -> tuple[Tensor, ...]:
        if len(inputs) != len(self.input_names):
            raise ResourceGraphCompileError("static Formula node received an incorrect input count")
        input_values = dict(zip(self.input_names, inputs, strict=True))
        values: list[Tensor] = []
        for source_kind, name in zip(
            self._binding_sources, self._static_binding_names, strict=True
        ):
            if source_kind.startswith("input:"):
                values.append(input_values[source_kind.removeprefix("input:")])
            else:
                assert name is not None
                values.append(getattr(self, name))
        result = self.execution_plan(
            PreparedFormulaBindings(
                self.program_fingerprint,
                self.binding_names,
                tuple(values),
            )
        )
        return tuple(result[index] for index in self._output_indices)

    def local_vjp(
        self,
        inputs: tuple[Tensor, ...],
        outputs: tuple[Tensor, ...],
        output_cotangents: tuple[Tensor, ...],
        parameters: tuple[nn.Parameter, ...],
        create_graph: bool,
    ) -> LocalVJPResult:
        """Formula programs use their ordinary tensor implementation as a VJP."""

        return autograd_local_vjp(
            self,
            inputs,
            outputs,
            output_cotangents,
            parameters,
            create_graph,
        )


@dataclass(frozen=True)
class ProgramGraphState:
    """A graph-state receipt that keeps mutable backings separate from modules."""

    contract_fingerprint: str
    resources: tuple[TensorResourceState, ...]
    # (join id, ((input port name, last consumed resource epoch), ...)).  This
    # is plain metadata, not a Tensor payload or a second scheduler state.
    join_cursors: tuple[tuple[str, tuple[tuple[str, int], ...]], ...] = ()


@dataclass(frozen=True)
class ProgramGraphExecution:
    """One functional graph transition and the exact resulting resource state."""

    state: ProgramGraphState
    connections: tuple[ConnectionExecution, ...]
    nodes: tuple[ProgramNodeExecution | MultiPortProgramNodeExecution, ...] = ()
    joins: tuple["ProgramJoinExecution", ...] = ()
    dispatches: tuple["ProgramDispatch", ...] = ()
    routes: tuple["ProgramRouteExecution", ...] = ()


@dataclass(frozen=True)
class ProgramGraphSpecialization:
    """A graph-wide hard-fate substitution and its retained candidate ids."""

    graph: "ProgramGraph"
    selections: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ProgramFateSample:
    """One joint hard-fate draw for a declared program execution window."""

    selections: tuple[tuple[str, str], ...]
    log_probability: Tensor
    fate_log_probabilities: tuple[Tensor, ...] = ()
    fate_route_paths: tuple[tuple[tuple[tuple[str, int], ...], ...], ...] = ()
    route_selection_scopes: tuple[tuple[str, Literal["batch", "sample"]], ...] = ()

    def structure_objective(
        self, final_loss: Tensor, *, baseline: Tensor | float = 0.0,
        route_selections: Mapping[str, Tensor] | None = None,
    ) -> Tensor:
        """Credit each sampled fate from the final loss of rows that executed it."""

        if not isinstance(final_loss, Tensor) or final_loss.ndim > 1 or not final_loss.numel():
            raise ResourceGraphError("joint structure final_loss must be scalar or per-row")
        baseline_value = torch.as_tensor(baseline, device=final_loss.device, dtype=final_loss.dtype)
        if baseline_value.numel() != 1 and baseline_value.shape != final_loss.shape:
            raise ResourceGraphError("joint structure baseline must be scalar or match per-row loss")
        if baseline_value.numel() == 1:
            baseline_value = baseline_value.reshape(())
        advantage = final_loss.detach() - baseline_value.detach()
        if len(self.fate_log_probabilities) != len(self.fate_route_paths):
            raise ResourceGraphError("fate credit terms and route paths must match")
        scopes = dict(self.route_selection_scopes)
        if final_loss.ndim == 0 and any(
            scopes.get(route_id) == "sample"
            for paths in self.fate_route_paths for path in paths for route_id, _ in path
        ):
            raise ResourceGraphError("sample-scoped fate credit requires per-row final losses")
        terms: list[Tensor] = []
        for log_probability, paths in zip(
            self.fate_log_probabilities, self.fate_route_paths, strict=True
        ):
            selected = torch.zeros_like(final_loss, dtype=torch.bool)
            for path in paths:
                on_path = torch.ones_like(selected)
                for route_id, candidate_index in path:
                    if route_selections is None or route_id not in route_selections:
                        raise ResourceGraphError(f"fate credit requires actual route selections for {route_id!r}")
                    choice = route_selections[route_id]
                    if scopes.get(route_id) == "batch":
                        if choice.ndim != 0:
                            raise ResourceGraphError("batch route selection must be scalar")
                    elif choice.shape != final_loss.shape:
                        raise ResourceGraphError("route selection must match per-row final loss")
                    on_path = on_path & (choice == candidate_index)
                selected = selected | on_path
            contribution = torch.where(selected, advantage, torch.zeros_like(advantage))
            terms.append(log_probability * contribution.mean())
        return torch.stack(terms).sum()

    def trajectory_structure_objective(
        self, final_loss: Tensor, *, route_ids: Sequence[str],
        route_selections: Sequence[Sequence[Tensor]], iteration_active: Sequence[Tensor],
        baseline: Tensor | float = 0.0,
    ) -> Tensor:
        """Credit a sampled fate only if an actual hard loop path reached it."""

        if not isinstance(final_loss, Tensor) or final_loss.ndim > 1 or not final_loss.numel():
            raise ResourceGraphError("trajectory final_loss must be scalar or per-row")
        baseline_value = torch.as_tensor(baseline, device=final_loss.device, dtype=final_loss.dtype)
        if baseline_value.numel() != 1 and baseline_value.shape != final_loss.shape:
            raise ResourceGraphError("trajectory baseline must be scalar or match per-row loss")
        if len(route_selections) != len(iteration_active):
            raise ResourceGraphError("route selections and activity must have the same horizon")
        if len(self.fate_log_probabilities) != len(self.fate_route_paths):
            raise ResourceGraphError("fate credit terms and route paths must match")
        routes: dict[str, list[int]] = {}
        for index, route_id in enumerate(route_ids):
            routes.setdefault(route_id, []).append(index)
        advantage = final_loss.detach() - baseline_value.detach()
        objective = final_loss.new_zeros(())
        for log_probability, paths in zip(
            self.fate_log_probabilities, self.fate_route_paths, strict=True,
        ):
            reached = torch.zeros_like(final_loss, dtype=torch.bool)
            for active, choices in zip(iteration_active, route_selections, strict=True):
                for path in paths:
                    selected = active if final_loss.ndim else active.any()
                    if final_loss.ndim and selected.shape != final_loss.shape:
                        raise ResourceGraphError("trajectory activity must match per-row loss")
                    for route_id, candidate_index in path:
                        if route_id not in routes:
                            raise ResourceGraphError(f"unknown fate route {route_id!r}")
                        reached_route = torch.stack(tuple(
                            choices[index] == candidate_index for index in routes[route_id]
                        )).any(dim=0)
                        if not final_loss.ndim:
                            reached_route = reached_route.any()
                        elif reached_route.ndim and reached_route.shape != final_loss.shape:
                            raise ResourceGraphError("trajectory route choices must match per-row loss")
                        selected = selected & reached_route
                    reached = reached | selected
            contribution = torch.where(reached, advantage, torch.zeros_like(advantage))
            objective = objective + log_probability * contribution.mean()
        return objective


@dataclass(frozen=True)
class ProgramJoinExecution:
    """Receipt for one readiness check of a declared :class:`ProgramJoin`."""

    join_id: str
    node_id: str
    fired: bool
    inputs: tuple[tuple[str, ResourceBinding], ...]


@dataclass(frozen=True)
class ProgramRouteExecution:
    """Actual local dispatch; sampled routes admit score-function credit."""

    route_id: str
    candidate_id: str
    scores: ResourceBinding
    log_probability: Tensor
    sampled: bool
    iteration: int = 0

    def structure_objective(self, final_loss: Tensor, *, baseline: Tensor | float = 0.0) -> Tensor:
        if not self.sampled:
            raise ResourceGraphError("route structure credit requires a sampled dispatch")
        if not isinstance(final_loss, Tensor) or final_loss.numel() != 1:
            raise ResourceGraphError("route final_loss must be a scalar Tensor")
        baseline_value = torch.as_tensor(baseline, device=final_loss.device, dtype=final_loss.dtype)
        if baseline_value.numel() != 1:
            raise ResourceGraphError("route baseline must be scalar")
        return (final_loss.detach() - baseline_value.detach()) * self.log_probability


@dataclass(frozen=True)
class ProgramDispatch:
    """One ordered program frontier, whose members read a shared snapshot."""

    iteration: int
    frontier: int
    members: tuple[ConnectionExecution | ProgramNodeExecution | MultiPortProgramNodeExecution, ...]
    join: ProgramJoinExecution | None = None
    frontier_path: tuple[int, ...] = ()
    route: ProgramRouteExecution | None = None


@dataclass(frozen=True)
class ProgramGraphInvocation:
    """A branch-local subprogram call with an explicitly declared view result."""

    output: TensorView
    state: ProgramGraphState
    connections: tuple[ConnectionExecution, ...]
    nodes: tuple[ProgramNodeExecution | MultiPortProgramNodeExecution, ...] = ()


def _loop_continue_mask(view: TensorView) -> Tensor:
    """Validate the compact batch mask ABI used by :class:`ProgramLoop`."""

    if not view.value.is_floating_point() or view.value.ndim != 1:
        raise ResourceGraphError(
            "ProgramLoop continuation resource must be a rank-1 floating Tensor mask"
        )
    if view.batch_axis != 0:
        raise ResourceGraphError("ProgramLoop continuation resource must use its only axis as batch")
    return view.value > 0.0


def _mask_loop_publication(
    previous: TensorView, candidate: TensorView, active: Tensor
) -> TensorView:
    """Keep inactive batch rows at their previous resource value."""

    if previous.value.shape != candidate.value.shape or previous.axes != candidate.axes:
        raise ResourceGraphError(
            "conditional ProgramLoop updates must preserve each resource Tensor ABI"
        )
    if active.ndim != 1 or active.shape[0] != candidate.value.shape[candidate.batch_axis]:
        raise ResourceGraphError("ProgramLoop active mask does not match a resource batch axis")
    mask_shape = [1] * candidate.value.ndim
    mask_shape[candidate.batch_axis] = active.shape[0]
    value = torch.where(active.reshape(mask_shape), candidate.value, previous.value)
    if previous.mask is None and candidate.mask is None:
        mask = None
    elif previous.mask is None or candidate.mask is None:
        mask = candidate.mask if previous.mask is None else previous.mask
    else:
        mask = torch.where(active.reshape(mask_shape), candidate.mask, previous.mask)
    return TensorView(value, candidate.axes, index_map=candidate.index_map, mask=mask)


class ProgramGraph(nn.Module):
    """Small explicit resource graph with direct and conditional connections.

    It is an execution substrate, not a replacement for Formula programs.  A
    caller may use its connections as edges inside a larger program, while the
    existing Formula Fabric continues to define numerical operations.
    """

    _component_reference: ClassVar[str] = "arti/program-graph@1"

    def __init__(
        self,
        resources: Sequence[TensorResource],
        connections: Sequence[Connection],
        *,
        programs: Mapping[str, Sequence[str | ProgramStage | ProgramJoin | ProgramRoute]] | None = None,
        nodes: Sequence[ProgramNode | MultiPortProgramNode] = (),
        loops: Sequence[ProgramLoop] = (),
    ) -> None:
        super().__init__()
        resources = tuple(resources)
        connections = tuple(connections)
        nodes = tuple(nodes)
        loops = tuple(loops)
        if not resources:
            raise ResourceGraphError("ProgramGraph requires at least one resource")
        ids = [resource.spec.resource_id for resource in resources]
        if len(set(ids)) != len(ids):
            raise ResourceGraphError("resource ids must be unique")
        if any(not isinstance(resource, TensorResource) for resource in resources):
            raise TypeError("resources must contain TensorResource values")
        connection_ids = [connection.connection_id for connection in connections]
        if len(set(connection_ids)) != len(connection_ids):
            raise ResourceGraphError("connection ids must be unique")
        if any(not isinstance(connection, Connection) for connection in connections):
            raise TypeError("connections must contain Connection values")
        node_ids = [node.node_id for node in nodes]
        nodes_by_id = {node.node_id: node for node in nodes}
        if len(set(node_ids)) != len(node_ids):
            raise ResourceGraphError("program-node ids must be unique")
        if any(hasattr(nn.Module, node_id) for node_id in node_ids):
            raise ResourceGraphError("program-node ids must not collide with nn.Module attributes")
        if set(connection_ids).intersection(node_ids):
            raise ResourceGraphError("program-node ids must not collide with connection ids")
        if any(not isinstance(node, (ProgramNode, MultiPortProgramNode)) for node in nodes):
            raise TypeError("nodes must contain ProgramNode or MultiPortProgramNode values")
        known = set(ids)
        for connection in connections:
            if connection.source.resource_id not in known or connection.destination.resource_id not in known:
                raise ResourceGraphError("connection ports must reference graph resources")
            if any(view.resource_id not in known for view in connection.operand_views.values()):
                raise ResourceGraphError("connection operand views must reference graph resources")
            if any(dependency not in connection_ids for dependency in connection.depends_on):
                raise ResourceGraphError("connection dependencies must reference graph connections")
        for node in nodes:
            if isinstance(node, ProgramNode):
                node_ports = (
                    ResourcePort(node.input_resource_id),
                    ResourcePort(node.output_resource_id),
                )
            else:
                node_ports = (*node.input_ports.values(), *node.output_ports.values())
            if any(port.resource_id not in known for port in node_ports):
                raise ResourceGraphError("program-node ports must reference graph resources")
        _validate_connection_dependencies(connections)
        program_mapping = {} if programs is None else dict(programs)
        if any(not isinstance(program_id, str) for program_id in program_mapping):
            raise TypeError("program ids must be strings")
        known_steps = set(connection_ids).union(node_ids)
        known_programs = set(program_mapping)
        normalized_programs: dict[str, tuple[str | ProgramStage | ProgramJoin | ProgramRoute, ...]] = {}
        for program_id, sequence in program_mapping.items():
            _require_identifier(program_id, field="program_id")
            step_sequence = tuple(sequence)
            if any(not isinstance(step, (str, ProgramStage, ProgramJoin, ProgramRoute)) for step in step_sequence):
                raise ResourceGraphError("programs must contain declared connection or node ids")
            flat_step_ids = tuple(
                step_id
                for step in step_sequence
                for step_id in (
                    step.step_ids
                    if isinstance(step, ProgramStage)
                    else tuple(candidate for candidate in step.candidates if candidate not in known_programs)
                    if isinstance(step, ProgramRoute)
                    else (step.node_id,) if isinstance(step, ProgramJoin) else (step,)
                )
            )
            if not step_sequence or any(step_id not in known_steps for step_id in flat_step_ids):
                raise ResourceGraphError("programs must contain declared connection or node ids")
            for step in step_sequence:
                if isinstance(step, ProgramJoin) and not isinstance(nodes_by_id[step.node_id], MultiPortProgramNode):
                    raise ResourceGraphError("ProgramJoin must reference a MultiPortProgramNode")
                if isinstance(step, ProgramRoute):
                    if step.score_resource_id not in known:
                        raise ResourceGraphError("ProgramRoute scores must reference a graph resource")
                    for candidate in step.candidates:
                        if candidate in known_programs:
                            if candidate in known_steps:
                                raise ResourceGraphError("route candidate program and graph step ids must be distinct")
                            candidate_entries = tuple(program_mapping[candidate])
                            if not candidate_entries or any(
                                not isinstance(member, (str, ProgramStage, ProgramJoin, ProgramRoute))
                                for member in candidate_entries
                            ):
                                raise ResourceGraphError(
                                    "route candidate programs must contain steps, parallel stages, joins or routes"
                                )
                            if any(
                                member_id not in known_steps
                                for member in candidate_entries
                                for member_id in (
                                    member.step_ids if isinstance(member, ProgramStage)
                                    else (member.node_id,) if isinstance(member, ProgramJoin)
                                    else () if isinstance(member, ProgramRoute)
                                    else (member,)
                                )
                            ):
                                raise ResourceGraphError("route candidate program names an unknown graph step")
                    if any(
                        connection.connection_id in step.candidates and connection.depends_on
                        for connection in connections
                    ):
                        raise ResourceGraphError("ProgramRoute candidates cannot have undeclared edge dependencies")
            join_ids = [step.join_id for step in step_sequence if isinstance(step, ProgramJoin)]
            if len(set(join_ids)) != len(join_ids):
                raise ResourceGraphError("programs must not declare a ProgramJoin id twice")
            normalized_programs[program_id] = step_sequence
        visiting: set[str] = set()
        visited: set[str] = set()

        def check_route_cycles(program_id: str) -> None:
            if program_id in visiting:
                raise ResourceGraphError("route candidate programs must not form a cycle")
            if program_id in visited:
                return
            visiting.add(program_id)
            for entry in normalized_programs[program_id]:
                if isinstance(entry, ProgramRoute):
                    for candidate in entry.candidates:
                        if candidate in normalized_programs:
                            check_route_cycles(candidate)
            visiting.remove(program_id)
            visited.add(program_id)

        for program_id in normalized_programs:
            check_route_cycles(program_id)
        if any(not isinstance(loop, ProgramLoop) for loop in loops):
            raise TypeError("loops must contain ProgramLoop values")
        loop_ids = [loop.loop_id for loop in loops]
        if len(set(loop_ids)) != len(loop_ids):
            raise ResourceGraphError("program-loop ids must be unique")
        for loop in loops:
            if loop.program_id not in normalized_programs:
                raise ResourceGraphError("ProgramLoop must reference a declared program")
            if loop.continue_resource_id not in known:
                raise ResourceGraphError("ProgramLoop continuation must reference a graph resource")
        self._resources = {resource.spec.resource_id: resource for resource in resources}
        self.connections = nn.ModuleDict({connection.connection_id: connection for connection in connections})
        self.nodes = nn.ModuleDict({node.node_id: node for node in nodes})
        self._programs = normalized_programs
        self._loops = {loop.loop_id: loop for loop in loops}
        self._joins = {
            step.join_id: step
            for steps in normalized_programs.values()
            for step in steps
            if isinstance(step, ProgramJoin)
        }
        if len(self._joins) != sum(
            isinstance(step, ProgramJoin) for steps in normalized_programs.values() for step in steps
        ):
            raise ResourceGraphError("ProgramJoin ids must be globally unique")
        route_ids = [
            step.route_id for steps in normalized_programs.values()
            for step in steps if isinstance(step, ProgramRoute)
        ]
        if len(set(route_ids)) != len(route_ids):
            raise ResourceGraphError("ProgramRoute ids must be globally unique")
        self._join_cursors = self._initial_join_cursors(self._resources)

    @property
    def resources(self) -> Mapping[str, TensorResource]:
        return self._resources.copy()

    def contract_config(self) -> dict[str, object]:
        """Return the resource and relation declaration, excluding tensor payloads."""

        return {
            "resources": [
                self._resources[resource_id].spec.contract_config()
                for resource_id in sorted(self._resources)
            ],
            "connections": [
                self.connections[connection_id].contract_config()
                for connection_id in sorted(self.connections)
            ],
            "nodes": [self.nodes[node_id].contract_config() for node_id in sorted(self.nodes)],
            "programs": {
                program_id: [
                    step if isinstance(step, str) else step.contract_config()
                    for step in self._programs[program_id]
                ]
                for program_id in sorted(self._programs)
            },
            "loops": [self._loops[loop_id].contract_config() for loop_id in sorted(self._loops)],
        }

    @property
    def contract_fingerprint(self) -> str:
        payload = json.dumps(
            self.contract_config(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def state(self) -> ProgramGraphState:
        """Capture mutable resource backing without serializing operation modules."""

        return self._state_from_resources(self._resources, join_cursors=self._join_cursors)

    def static_program_inputs(
        self,
        plan: StaticDataflowProgramExecutionPlan,
        *,
        state: ProgramGraphState | None = None,
        contexts: Mapping[str, Tensor] | None = None,
        credit_masks: Mapping[str, Tensor] | None = None,
    ) -> tuple[Tensor, ...]:
        """Carry a functional snapshot into a compiled mixed-program plan.

        Functional Join cursors are graph-wide epochs; the compiled plan uses
        per-row readiness. Each pending arrival is therefore broadcast across
        the snapshot's batch when execution switches to the compiled plan.
        """

        if not isinstance(plan, StaticDataflowProgramExecutionPlan):
            raise TypeError("plan must be a StaticDataflowProgramExecutionPlan")
        if tuple(plan.resource_ids) != tuple(self._resources):
            raise ResourceGraphError("compiled plan resources do not match this graph")
        snapshot = self.state() if state is None else state
        resources = self._restore_resources(snapshot)
        cursors = self._join_cursors_from_state(snapshot, resources)
        values = tuple(resources[resource_id].resolve().view.value for resource_id in plan.resource_ids)
        supplied_contexts = {} if contexts is None else dict(contexts)
        supplied_masks = _credit_mask_map(credit_masks)
        if set(supplied_contexts) != set(plan.context_connection_ids):
            raise ResourceGraphError("compiled plan contexts do not match its declared connections")
        if set(supplied_masks) != set(plan.credit_mask_connection_ids):
            raise ResourceGraphError("compiled plan credit masks do not match its declared connections")
        batch_size = values[0].shape[plan.templates[0].batch_axis]
        pending = torch.tensor(
            [
                resources[resource_id].resolve().binding.epoch > cursors[join_id][port_name]
                for join_id, port_name, resource_id in plan.arrival_ports
            ],
            dtype=torch.bool,
            device=values[0].device,
        )
        arrivals = pending.unsqueeze(0).expand(batch_size, -1)
        return (
            *values,
            *(supplied_contexts[name] for name in plan.context_connection_ids),
            *(supplied_masks[name] for name in plan.credit_mask_connection_ids),
            arrivals,
        )

    def _copy_for_specialization(
        self,
        *,
        programs: Mapping[str, Sequence[str | ProgramStage | ProgramJoin | ProgramRoute]] | None = None,
        excluded_steps: frozenset[str] = frozenset(),
        share_module_state: bool = False,
        state: ProgramGraphState | None = None,
    ) -> ProgramGraph:
        # A live resource may have non-leaf autograd history; only modules share the deepcopy memo.
        source_resources = self._resources if state is None else self._restore_resources(state)
        source_cursors = (
            self._join_cursors if state is None
            else self._join_cursors_from_state(state, source_resources)
        )
        resource_states: list[TensorResourceState] = []
        for resource_id in source_resources:
            resource_state = source_resources[resource_id].state()
            resource_states.append(
                TensorResourceState(
                    resource_state.spec,
                    _detach_clone_view(resource_state.default_view),
                    _detach_clone_view(resource_state.active_view),
                    resource_state.active_source,
                    resource_state.epoch,
                    resource_state.step_index,
                )
            )
        memo: dict[int, object] = {}
        if share_module_state:
            memo.update((id(parameter), parameter) for parameter in self.parameters())
            memo.update((id(buffer), buffer) for buffer in self.buffers())
        specialized = ProgramGraph(
            tuple(TensorResource.restore(state, fresh_history=True) for state in resource_states),
            tuple(
                deepcopy(connection, memo)
                for connection_id, connection in self.connections.items()
                if connection_id not in excluded_steps
            ),
            nodes=tuple(
                deepcopy(node, memo)
                for node_id, node in self.nodes.items()
                if node_id not in excluded_steps
            ),
            programs=deepcopy(self._programs if programs is None else programs, memo),
            loops=tuple(deepcopy(loop, memo) for loop in self._loops.values()),
        )
        specialized.restore_state(
            ProgramGraphState(
                specialized.contract_fingerprint,
                tuple(resource.state() for resource in specialized._resources.values()),
                self._freeze_join_cursors({
                    join_id: cursors for join_id, cursors in source_cursors.items()
                    if join_id in specialized._joins
                }),
            ),
            fresh_history=True,
        )
        return specialized

    def specialize_differentiable_nodes(
        self,
        selections: Mapping[str, str] | None = None,
        *,
        share_module_state: bool = False,
        state: ProgramGraphState | None = None,
    ) -> ProgramGraphSpecialization:
        """Produce a hard-fate graph while preserving graph-wide module aliases.

        Node-local specialization is intentionally a portable copy. A graph
        must instead be copied as one object and substituted in that copy, so
        candidates that share parameters or stateful modules with other graph
        nodes retain their actual identity. ``share_module_state`` preserves
        live parameters and buffers for finite hard-plan training caches;
        the default remains a separate deployment copy. ``state`` optionally
        starts the specialized graph from a functional execution snapshot.
        """

        requested = {} if selections is None else dict(selections)
        unknown = set(requested) - set(self.nodes)
        if unknown:
            raise ResourceGraphError(
                f"differentiable graph specialization names unknown nodes: {sorted(unknown)!r}"
            )
        for node_id in requested:
            if not isinstance(self.nodes[node_id], DifferentiableFabricNode):
                raise ResourceGraphError(
                    f"graph specialization node {node_id!r} is not a DifferentiableFabricNode"
                )
        specialized = self._copy_for_specialization(
            share_module_state=share_module_state, state=state,
        )
        chosen: list[tuple[str, str]] = []
        for node_id, node in tuple(specialized.nodes.items()):
            if not isinstance(node, DifferentiableFabricNode):
                continue
            candidate_id = requested.get(node_id)
            if candidate_id is None:
                candidate_id = node.candidate_ids[int(torch.argmax(node.logits.detach()).item())]
            if candidate_id not in node.candidate_ids:
                raise ResourceGraphError(
                    f"unknown DifferentiableFabricNode candidate {candidate_id!r} for node {node_id!r}"
                )
            candidate = node.candidate(candidate_id)
            candidate.node_id = node_id
            specialized.nodes[node_id] = candidate
            chosen.append((node_id, candidate_id))
        return ProgramGraphSpecialization(specialized, tuple(chosen))

    def specialize_program_routes(
        self, selections: Mapping[str, str], *,
        share_module_state: bool = False,
        state: ProgramGraphState | None = None,
    ) -> ProgramGraphSpecialization:
        """Freeze routes, optionally carrying a functional execution snapshot forward."""

        requested = dict(selections)
        routes = {
            entry.route_id: entry
            for entries in self._programs.values()
            for entry in entries
            if isinstance(entry, ProgramRoute)
        }
        unknown = set(requested) - set(routes)
        if unknown:
            raise ResourceGraphError(f"route specialization names unknown routes: {sorted(unknown)!r}")
        for route_id, candidate_id in requested.items():
            if candidate_id not in routes[route_id].candidates:
                raise ResourceGraphError(
                    f"unknown ProgramRoute candidate {candidate_id!r} for route {route_id!r}"
                )
        def expand_entries(
            entries: Sequence[str | ProgramStage | ProgramJoin | ProgramRoute], *,
            inlined: bool = False,
        ) -> tuple[str | ProgramStage | ProgramJoin | ProgramRoute, ...]:
            expanded: list[str | ProgramStage | ProgramJoin | ProgramRoute] = []
            for entry in entries:
                if not isinstance(entry, ProgramRoute):
                    expanded.append(entry)
                elif entry.route_id in requested:
                    candidate = requested[entry.route_id]
                    expanded.extend(expand_entries(
                        self._route_candidate_entries(candidate),
                        inlined=candidate in self._programs,
                    ))
                elif inlined:
                    raise ResourceGraphError(
                        "inlining a routed candidate requires selections for its nested routes"
                    )
                else:
                    expanded.append(entry)
            return tuple(expanded)

        programs = {
            program_id: expand_entries(entries)
            for program_id, entries in self._programs.items()
        }
        inlined_programs = {
            candidate for candidate in requested.values() if candidate in self._programs
        }
        unselected_programs = {
            candidate
            for route_id, route in routes.items() if route_id in requested
            for candidate in route.candidates
            if candidate != requested[route_id] and candidate in self._programs
        }
        still_referenced_programs = {
            candidate
            for entries in programs.values() for entry in entries
            if isinstance(entry, ProgramRoute) for candidate in entry.candidates
        }
        for program_id in (unselected_programs | inlined_programs) - still_referenced_programs - {
            loop.program_id for loop in self._loops.values()
        }:
            programs.pop(program_id)
        def leaf_steps(
            entries: Sequence[str | ProgramStage | ProgramJoin | ProgramRoute],
            program_mapping: Mapping[str, Sequence[str | ProgramStage | ProgramJoin | ProgramRoute]],
        ) -> set[str]:
            leaves: set[str] = set()
            for entry in entries:
                if isinstance(entry, ProgramRoute):
                    for candidate in entry.candidates:
                        leaves.update(leaf_steps(
                            program_mapping[candidate] if candidate in program_mapping else (candidate,),
                            program_mapping,
                        ))
                elif isinstance(entry, ProgramStage):
                    leaves.update(entry.step_ids)
                elif isinstance(entry, ProgramJoin):
                    leaves.add(entry.node_id)
                else:
                    leaves.add(entry)
            return leaves

        retained_steps = set().union(*(
            leaf_steps(entries, programs) for entries in programs.values()
        ))
        unselected = set().union(*(
            leaf_steps(self._route_candidate_entries(candidate), self._programs)
            for route_id, route in routes.items() if route_id in requested
            for candidate in route.candidates if candidate != requested[route_id]
        ))
        specialized = self._copy_for_specialization(
            programs=programs,
            excluded_steps=frozenset(unselected - retained_steps),
            share_module_state=share_module_state,
            state=state,
        )
        return ProgramGraphSpecialization(
            specialized, tuple(sorted(requested.items())),
        )

    def _initial_join_cursors(
        self, resources: Mapping[str, TensorResource]
    ) -> dict[str, dict[str, int]]:
        """Treat the current graph snapshot as already observed by each join."""

        return {
            join_id: {
                name: resources[port.resource_id].resolve().binding.epoch
                for name, port in self._join_node(join).input_ports.items()
            }
            for join_id, join in self._joins.items()
        }

    def _join_node(self, join: ProgramJoin) -> MultiPortProgramNode:
        node = self.node(join.node_id)
        if not isinstance(node, MultiPortProgramNode):  # constructor invariant, kept for restored state
            raise ResourceGraphError("ProgramJoin must reference a MultiPortProgramNode")
        return node

    @staticmethod
    def _freeze_join_cursors(
        cursors: Mapping[str, Mapping[str, int]]
    ) -> tuple[tuple[str, tuple[tuple[str, int], ...]], ...]:
        return tuple(
            (join_id, tuple(sorted((name, int(epoch)) for name, epoch in ports.items())))
            for join_id, ports in sorted(cursors.items())
        )

    def _join_cursors_from_state(
        self,
        state: ProgramGraphState,
        resources: Mapping[str, TensorResource],
    ) -> dict[str, dict[str, int]]:
        """Validate persisted join receipts and fill absent joins from this snapshot."""

        supplied = {join_id: dict(ports) for join_id, ports in state.join_cursors}
        if len(supplied) != len(state.join_cursors) or set(supplied) - set(self._joins):
            raise ResourceGraphError("ProgramGraphState join cursors do not match this graph")
        baseline = self._initial_join_cursors(resources)
        for join_id, cursors in supplied.items():
            expected = set(baseline[join_id])
            if set(cursors) != expected or any(type(epoch) is not int or epoch < 0 for epoch in cursors.values()):
                raise ResourceGraphError("ProgramGraphState join cursor ports are invalid")
            baseline[join_id] = cursors
        return baseline

    def _state_from_resources(
        self,
        resources: Mapping[str, TensorResource],
        *,
        join_cursors: Mapping[str, Mapping[str, int]] | None = None,
    ) -> ProgramGraphState:
        cursors = self._initial_join_cursors(resources) if join_cursors is None else join_cursors
        return ProgramGraphState(
            self.contract_fingerprint,
            tuple(resources[resource_id].state() for resource_id in sorted(resources)),
            self._freeze_join_cursors(cursors),
        )

    def _restore_resources(
        self,
        state: ProgramGraphState,
        *,
        fresh_history: bool = False,
    ) -> dict[str, TensorResource]:
        if not isinstance(state, ProgramGraphState):
            raise TypeError("state must be ProgramGraphState")
        if state.contract_fingerprint != self.contract_fingerprint:
            raise ResourceGraphError("resource graph state does not match this graph contract")
        restored = {
            item.spec.resource_id: TensorResource.restore(item, fresh_history=fresh_history)
            for item in state.resources
        }
        if set(restored) != set(self._resources):
            raise ResourceGraphError("resource graph state does not declare this graph's resources")
        return restored

    def restore_state(self, state: ProgramGraphState, *, fresh_history: bool = False) -> None:
        """Restore compatible resource state into this graph's declared relations.

        ``fresh_history`` starts a new autograd generation while retaining the
        same resource values, which is required by deployment specialization.
        """

        self._resources = self._restore_resources(state, fresh_history=fresh_history)
        self._join_cursors = self._join_cursors_from_state(state, self._resources)

    def resource(self, resource_id: str) -> TensorResource:
        try:
            return self._resources[resource_id]
        except KeyError as error:
            raise ResourceGraphError(f"unknown resource {resource_id!r}") from error

    def connection(self, connection_id: str) -> Connection:
        try:
            return self.connections[connection_id]
        except KeyError as error:
            raise ResourceGraphError(f"unknown connection {connection_id!r}") from error

    def node(self, node_id: str) -> ProgramNode | MultiPortProgramNode:
        try:
            return self.nodes[node_id]
        except KeyError as error:
            raise ResourceGraphError(f"unknown program node {node_id!r}") from error

    def _connection_sequence(
        self, connection_ids: Sequence[str], *, completed_connections: Sequence[str] = ()
    ) -> tuple[Connection, ...]:
        """Resolve a serial edge sequence while enforcing declared dependencies."""

        connections = tuple(self.connection(connection_id) for connection_id in connection_ids)
        completed: set[str] = set(completed_connections)
        for connection in connections:
            missing = set(connection.depends_on) - completed
            if missing:
                raise ResourceGraphError(
                    f"connection {connection.connection_id!r} requires prior dependencies "
                    f"{sorted(missing)!r}"
                )
            completed.add(connection.connection_id)
        return connections

    def _program_steps(
        self, step_ids: Sequence[str], *, completed_connections: set[str] | None = None
    ) -> tuple[Connection | ProgramNode | MultiPortProgramNode, ...]:
        """Resolve a declared mixed sequence and enforce edge dependencies."""

        steps: list[Connection | ProgramNode | MultiPortProgramNode] = []
        completed = set() if completed_connections is None else completed_connections
        for step_id in step_ids:
            if step_id in self.connections:
                step: Connection | ProgramNode | MultiPortProgramNode = self.connection(step_id)
                missing = set(step.depends_on) - completed
                if missing:
                    raise ResourceGraphError(
                        f"connection {step.connection_id!r} requires prior dependencies "
                        f"{sorted(missing)!r}"
                    )
                completed.add(step.connection_id)
            else:
                step = self.node(step_id)
            steps.append(step)
        return tuple(steps)

    def program(self, program_id: str) -> tuple[str | ProgramStage | ProgramJoin | ProgramRoute, ...]:
        try:
            return self._programs[program_id]
        except KeyError as error:
            raise ResourceGraphError(f"unknown resource program {program_id!r}") from error

    def _route_candidate_entries(
        self, candidate_id: str,
    ) -> tuple[str | ProgramStage | ProgramJoin | ProgramRoute, ...]:
        if candidate_id in self._programs:
            return self._programs[candidate_id]
        return (candidate_id,)

    def _route_candidate_stages(self, candidate_id: str) -> tuple[tuple[str, ...], ...]:
        if any(
            isinstance(entry, ProgramRoute)
            for entry in self._route_candidate_entries(candidate_id)
        ):
            raise ResourceGraphError("nested ProgramRoute requires recursive route lowering")
        return tuple(
            entry.step_ids if isinstance(entry, ProgramStage)
            else (entry.node_id,) if isinstance(entry, ProgramJoin)
            else (entry,)
            for entry in self._route_candidate_entries(candidate_id)
        )

    def sample_program_fates(
        self, program_id: str, *, generator: torch.Generator | None = None,
        route_selections: Mapping[str, str] | None = None,
    ) -> ProgramFateSample:
        """Draw hard node fates once for a program window, including parallel stages.

        Sampling is a host-side structure-window operation, not a GPU dispatch
        inside each program step. The returned choices can be reused for many
        hard forwards before the next structure window.
        """

        selections: list[tuple[str, str]] = []
        log_probabilities: list[Tensor] = []
        node_ids: list[str] = []
        seen_nodes: set[str] = set()
        route_paths: dict[str, list[tuple[tuple[str, int], ...]]] = {}
        route_scopes: dict[str, Literal["batch", "sample"]] = {}
        def visit(
            entries: Sequence[str | ProgramStage | ProgramJoin | ProgramRoute],
            path: tuple[tuple[str, int], ...],
        ) -> None:
            for entry in entries:
                if isinstance(entry, ProgramRoute):
                    route_scopes[entry.route_id] = entry.selection_scope
                    selected_batch = (
                        entry.selection_scope == "batch"
                        and route_selections is not None and entry.route_id in route_selections
                    )
                    if selected_batch:
                        candidates = (route_selections[entry.route_id],)
                        if candidates[0] not in entry.candidates:
                            raise ResourceGraphError("route fate selection is not a declared candidate")
                    else:
                        candidates = entry.candidates
                    for candidate in candidates:
                        candidate_path = (
                            path if selected_batch else
                            (*path, (entry.route_id, entry.candidates.index(candidate)))
                        )
                        visit(self._route_candidate_entries(candidate), candidate_path)
                    continue
                members = entry.step_ids if isinstance(entry, ProgramStage) else (
                    (entry.node_id,) if isinstance(entry, ProgramJoin) else (entry,)
                )
                for node_id in members:
                    if node_id not in seen_nodes:
                        seen_nodes.add(node_id)
                        node_ids.append(node_id)
                    paths = route_paths.setdefault(node_id, [])
                    if path not in paths:
                        paths.append(path)

        visit(self.program(program_id), ())
        fate_paths: list[tuple[tuple[tuple[str, int], ...], ...]] = []
        for node_id in node_ids:
            if node_id not in self.nodes:
                continue
            node = self.nodes[node_id]
            if not isinstance(node, DifferentiableFabricNode):
                continue
            probabilities = node.probabilities()
            index = torch.multinomial(probabilities.detach(), 1, generator=generator)
            selections.append((node_id, node.candidate_ids[int(index.item())]))
            log_probabilities.append(torch.log_softmax(node.logits / node.temperature, dim=0)[index[0]])
            fate_paths.append(tuple(route_paths[node_id]))
        if not log_probabilities:
            raise ResourceGraphError("program contains no differentiable node fates")
        return ProgramFateSample(
            tuple(selections), torch.stack(log_probabilities).sum(),
            tuple(log_probabilities), tuple(fate_paths), tuple(route_scopes.items()),
        )

    def loop(self, loop_id: str) -> ProgramLoop:
        try:
            return self._loops[loop_id]
        except KeyError as error:
            raise ResourceGraphError(f"unknown resource loop {loop_id!r}") from error

    def _execute_stage_functional(
        self,
        stage: ProgramStage,
        *,
        state: ProgramGraphState,
        input_views: Mapping[str, TensorView] | None,
        contexts: Mapping[str, Tensor] | None,
        credit_masks: Mapping[str, Tensor] | None,
        candidate_selections: Mapping[str, str] | None = None,
        completed_connections: set[str] | None = None,
    ) -> ProgramGraphExecution:
        """Evaluate one declared frontier against exactly one resource snapshot."""

        credit_masks = _credit_mask_map(credit_masks)
        resources = self._restore_resources(state)
        join_cursors = self._join_cursors_from_state(state, resources)
        for resource_id, input_view in (input_views or {}).items():
            resource = resources[resource_id]
            if resource.mounted:
                resource.replace(input_view)
            else:
                resource.mount(input_view)
        snapshots = {resource_id: resource.resolve() for resource_id, resource in resources.items()}
        pending: list[tuple[str, TensorView]] = []
        node_receipts: list[
            tuple[ProgramNode | MultiPortProgramNode, object | None, Mapping[str, ResourceBinding]]
        ] = []
        stage_steps = self._program_steps(
            stage.step_ids, completed_connections=completed_connections
        )
        writers: dict[str, str] = {}
        inputs_by_step: dict[str, set[str]] = {}
        for step in stage_steps:
            step_id = step.connection_id if isinstance(step, Connection) else step.node_id
            if isinstance(step, Connection):
                outputs = {step.destination.resource_id}
                inputs = {step.source.resource_id, *(view.resource_id for view in step.operand_views.values())}
            elif isinstance(step, ProgramNode):
                outputs = {step.output_resource_id}
                inputs = {step.input_resource_id}
            else:
                outputs = {port.resource_id for port in step.output_ports.values()}
                inputs = {port.resource_id for port in step.input_ports.values()}
            for resource_id in outputs:
                if resource_id in writers:
                    raise ResourceGraphError("ProgramStage writes require distinct resources")
                writers[resource_id] = step_id
            inputs_by_step[step_id] = inputs
        for step_id, inputs in inputs_by_step.items():
            conflicting = {resource_id: writer for resource_id, writer in writers.items() if resource_id in inputs and writer != step_id}
            if conflicting:
                raise ResourceGraphError(
                    "ProgramStage peers cannot read another peer's publication; use a later ProgramJoin"
                )
        applied_credit_masks: dict[str, Tensor | None] = {}
        for step in stage_steps:
            if isinstance(step, Connection):
                if set(step.depends_on).intersection(stage.step_ids):
                    raise ResourceGraphError("ProgramStage connections cannot depend on stage peers")
                credit_mask = _execution_credit_mask(
                    step, credit_masks.get(step.connection_id),
                    snapshots[step.source.resource_id].view.value,
                    samples=credit_masks,
                )
                applied_credit_masks[step.connection_id] = credit_mask
                output = step.apply(
                    snapshots[step.source.resource_id],
                    snapshots[step.destination.resource_id],
                    resources=snapshots,
                    context=(contexts or {}).get(step.connection_id),
                    credit_mask=credit_mask,
                )
                pending.append((step.destination.resource_id, output))
                continue
            if isinstance(step, ProgramNode):
                result = step.invoke(snapshots[step.input_resource_id].view)
                if not isinstance(result, ProgramNodeInvocation):
                    raise TypeError("ProgramNode.invoke must return ProgramNodeInvocation")
                pending.append((step.output_resource_id, result.output))
                node_receipts.append((step, result.receipt, {"value": snapshots[step.input_resource_id].binding}))
                continue
            inputs = {name: snapshots[port.resource_id].view for name, port in step.input_ports.items()}
            candidate_id = (candidate_selections or {}).get(step.node_id)
            if candidate_id is not None:
                if not isinstance(step, DifferentiableFabricNode):
                    raise ResourceGraphError(
                        f"candidate selection names non-differentiable node {step.node_id!r}"
                    )
                result = step.invoke_candidate_ports(candidate_id, inputs)
            else:
                result = step.invoke_ports(inputs)
            if not isinstance(result, MultiPortProgramNodeInvocation):
                raise TypeError("MultiPortProgramNode.invoke_ports must return MultiPortProgramNodeInvocation")
            if set(result.outputs) != set(step.output_ports):
                raise ResourceGraphError("ProgramStage node outputs do not match its declared ports")
            pending.extend((step.output_ports[name].resource_id, result.outputs[name]) for name in step.output_ports)
            node_receipts.append((
                step,
                result.receipt,
                {name: snapshots[port.resource_id].binding for name, port in step.input_ports.items()},
            ))
        destinations = [resource_id for resource_id, _ in pending]
        if len(destinations) != len(set(destinations)):
            raise ResourceGraphError("ProgramStage writes require distinct resources")
        for resource_id, output in pending:
            resources[resource_id].spec.validate(output, name=f"ProgramStage {resource_id}")
        published = {resource_id: resources[resource_id].advance(output) for resource_id, output in pending}
        connections = tuple(
            ConnectionExecution(
                step.connection_id,
                snapshots[step.source.resource_id].binding,
                published[step.destination.resource_id].binding,
                snapshots[step.destination.resource_id].binding,
                credit_mask=applied_credit_masks[step.connection_id],
                operands=tuple(
                    (name, snapshots[view.resource_id].binding)
                    for name, view in step.operand_views.items()
                ),
                context=(contexts or {}).get(step.connection_id),
            )
            for step in stage_steps
            if isinstance(step, Connection)
        )
        nodes: list[ProgramNodeExecution | MultiPortProgramNodeExecution] = []
        for step, receipt, inputs in node_receipts:
            if isinstance(step, ProgramNode):
                nodes.append(
                    ProgramNodeExecution(
                        step.node_id, inputs["value"], published[step.output_resource_id].binding, receipt
                    )
                )
            else:
                nodes.append(
                    MultiPortProgramNodeExecution(
                        step.node_id,
                        tuple(inputs.items()),
                        tuple(
                            (name, published[port.resource_id].binding)
                            for name, port in step.output_ports.items()
                        ),
                        receipt,
                    )
                )
        return ProgramGraphExecution(
            self._state_from_resources(resources, join_cursors=join_cursors), connections, tuple(nodes)
        )

    def _execute_join_functional(
        self,
        join: ProgramJoin,
        *,
        state: ProgramGraphState,
        candidate_selections: Mapping[str, str] | None = None,
    ) -> ProgramGraphExecution:
        """Fire one explicit join only when every named input advanced since consumption."""

        resources = self._restore_resources(state)
        cursors = self._join_cursors_from_state(state, resources)
        node = self._join_node(join)
        snapshots = {resource_id: resource.resolve() for resource_id, resource in resources.items()}
        inputs = tuple((name, snapshots[port.resource_id].binding) for name, port in node.input_ports.items())
        prior = cursors[join.join_id]
        ready = all(binding.epoch > prior[name] for name, binding in inputs)
        receipt = ProgramJoinExecution(join.join_id, node.node_id, ready, inputs)
        if not ready:
            return ProgramGraphExecution(
                self._state_from_resources(resources, join_cursors=cursors), (), (), (receipt,)
            )
        input_views = {name: snapshots[port.resource_id].view for name, port in node.input_ports.items()}
        candidate_id = (candidate_selections or {}).get(node.node_id)
        if candidate_id is not None:
            if not isinstance(node, DifferentiableFabricNode):
                raise ResourceGraphError(
                    f"candidate selection names non-differentiable node {node.node_id!r}"
                )
            result = node.invoke_candidate_ports(candidate_id, input_views)
        else:
            result = node.invoke_ports(input_views)
        if not isinstance(result, MultiPortProgramNodeInvocation):
            raise TypeError("MultiPortProgramNode.invoke_ports must return MultiPortProgramNodeInvocation")
        if set(result.outputs) != set(node.output_ports):
            raise ResourceGraphError("ProgramJoin node outputs do not match its declared ports")
        destinations = [port.resource_id for port in node.output_ports.values()]
        if len(destinations) != len(set(destinations)):
            raise ResourceGraphError("ProgramJoin outputs require distinct resources; use an explicit Formula join")
        for name, port in node.output_ports.items():
            resources[port.resource_id].spec.validate(result.outputs[name], name=f"join {join.join_id}.{name}")
        published = tuple(
            (name, resources[port.resource_id].advance(result.outputs[name]).binding)
            for name, port in node.output_ports.items()
        )
        cursors[join.join_id] = {name: binding.epoch for name, binding in inputs}
        execution = MultiPortProgramNodeExecution(node.node_id, inputs, published, result.receipt)
        return ProgramGraphExecution(
            self._state_from_resources(resources, join_cursors=cursors), (), (execution,), (receipt,)
        )

    def execute_program(
        self,
        program_id: str,
        *,
        contexts: Mapping[str, Tensor] | None = None,
        credit_masks: Mapping[str, Tensor] | None = None,
    ) -> tuple[ConnectionExecution, ...]:
        """Call a direct-only declared subprogram; ordered edges observe writes.

        Existing imperative callers retain their compact connection receipt.
        Programs containing mounted nodes use :meth:`execute_program_functional`
        so their state transition remains explicit.
        """

        step_ids = self.program(program_id)
        if any(
            isinstance(step_id, (ProgramStage, ProgramJoin, ProgramRoute))
            or (isinstance(step_id, str) and step_id in self.nodes)
            for step_id in step_ids
        ):
            raise ResourceGraphError(
                "imperative execute_program does not publish ProgramNode state; "
                "use execute_program_functional"
            )
        return self.execute(tuple(step_ids), contexts=contexts, credit_masks=credit_masks)

    def execute_program_functional(
        self,
        program_id: str,
        *,
        state: ProgramGraphState | None = None,
        input_views: Mapping[str, TensorView] | None = None,
        contexts: Mapping[str, Tensor] | None = None,
        credit_masks: Mapping[str, Tensor] | None = None,
        candidate_selections: Mapping[str, str] | None = None,
        iterations: int = 1,
        sample_routes: bool = False,
        active_rows: Tensor | None = None,
        _share_credit_mask_samples: bool = False,
    ) -> ProgramGraphExecution:
        """Run declared edges against one state value.

        Each invocation iteration is one automatic Bernoulli-mask scope;
        nested route reuse shares a connection's sample within that scope.
        """

        if type(iterations) is not int or iterations <= 0:
            raise ResourceGraphError("iterations must be a positive integer")
        if active_rows is not None and (active_rows.ndim != 1 or active_rows.dtype != torch.bool):
            raise ResourceGraphError("active_rows must be a rank-one boolean Tensor")
        credit_masks = _credit_mask_map(credit_masks)
        execution: ProgramGraphExecution | None = None
        connections: list[ConnectionExecution] = []
        nodes: list[ProgramNodeExecution | MultiPortProgramNodeExecution] = []
        joins: list[ProgramJoinExecution] = []
        dispatches: list[ProgramDispatch] = []
        routes: list[ProgramRouteExecution] = []
        for iteration in range(iterations):
            if not _share_credit_mask_samples:
                credit_masks.reset_execution()
            current = (self.state() if state is None else state) if iteration == 0 else execution.state
            first_inputs = input_views if iteration == 0 else None
            completed_connections: set[str] = set()
            for frontier, entry in enumerate(self.program(program_id)):
                if isinstance(entry, ProgramStage):
                    execution = self._execute_stage_functional(
                        entry,
                        state=current,
                        input_views=first_inputs,
                        contexts=contexts,
                        credit_masks=credit_masks,
                        candidate_selections=candidate_selections,
                        completed_connections=completed_connections,
                    )
                elif isinstance(entry, ProgramJoin):
                    if first_inputs:
                        raise ResourceGraphError("ProgramJoin cannot mount inputs; publish them through a producer first")
                    execution = self._execute_join_functional(
                        entry, state=current, candidate_selections=candidate_selections,
                    )
                elif isinstance(entry, ProgramRoute):
                    if entry.selection_scope == "sample":
                        raise ResourceGraphError(
                            "sample-scoped ProgramRoute requires a compiled fixed-shape program"
                        )
                    if first_inputs:
                        mounted = self._execute_steps_functional(
                            (), state=current, input_views=first_inputs,
                            contexts=contexts, credit_masks=credit_masks,
                        )
                        current = mounted.state
                        first_inputs = None
                    scores = self._restore_resources(current)[entry.score_resource_id].resolve()
                    value = scores.view.value
                    if value.ndim != 2 or value.shape[1] != len(entry.candidates):
                        raise ResourceGraphError("ProgramRoute scores must have shape [batch, candidates]")
                    if active_rows is not None:
                        if active_rows.shape[0] != value.shape[0]:
                            raise ResourceGraphError("active_rows batch size does not match route scores")
                        weights = active_rows.to(dtype=value.dtype, device=value.device).unsqueeze(1)
                        logits = (value * weights).sum(dim=0) / weights.sum().clamp_min(1)
                    else:
                        logits = value.mean(dim=0)
                    log_probabilities = torch.log_softmax(logits, dim=0)
                    if sample_routes:
                        selected = int(torch.multinomial(log_probabilities.detach().exp(), 1).item())
                    else:
                        selected = int(torch.argmax(logits).item())
                    candidate = entry.candidates[selected]
                    route_receipt = ProgramRouteExecution(
                        entry.route_id, candidate, scores.binding,
                        log_probabilities[selected], sample_routes, iteration,
                    )
                    routes.append(route_receipt)
                    execution = (
                        self.execute_program_functional(
                            candidate, state=current, contexts=contexts,
                            credit_masks=credit_masks,
                            candidate_selections=candidate_selections,
                            sample_routes=sample_routes, active_rows=active_rows,
                            _share_credit_mask_samples=True,
                        )
                        if candidate in self._programs
                        else self._execute_steps_functional(
                            (candidate,), state=current, input_views=None,
                            contexts=contexts, credit_masks=credit_masks,
                            candidate_selections=candidate_selections,
                            completed_connections=completed_connections,
                        )
                    )
                else:
                    execution = self._execute_steps_functional(
                        (entry,),
                        state=current,
                        input_views=first_inputs,
                        contexts=contexts,
                        credit_masks=credit_masks,
                        candidate_selections=candidate_selections,
                        completed_connections=completed_connections,
                    )
                current = execution.state
                first_inputs = None
                connections.extend(execution.connections)
                nodes.extend(execution.nodes)
                joins.extend(execution.joins)
                routes.extend(replace(route, iteration=iteration) for route in execution.routes)
                if isinstance(entry, ProgramRoute) and route_receipt.candidate_id in self._programs:
                    dispatches.append(ProgramDispatch(
                        iteration, frontier, (), frontier_path=(frontier,), route=route_receipt,
                    ))
                    dispatches.extend(
                        replace(
                            dispatch,
                            iteration=iteration,
                            frontier_path=(frontier, *dispatch.frontier_path),
                            route=(
                                None if dispatch.route is None
                                else replace(dispatch.route, iteration=iteration)
                            ),
                        )
                        for dispatch in execution.dispatches
                    )
                    continue
                if isinstance(entry, ProgramJoin):
                    members = execution.nodes
                    join_receipt = execution.joins[0]
                else:
                    step_ids = entry.step_ids if isinstance(entry, ProgramStage) else (
                        (route_receipt.candidate_id,) if isinstance(entry, ProgramRoute) else (entry,)
                    )
                    receipts = {
                        **{item.connection_id: item for item in execution.connections},
                        **{item.node_id: item for item in execution.nodes},
                    }
                    members = tuple(receipts[step_id] for step_id in step_ids)
                    join_receipt = None
                dispatches.append(ProgramDispatch(
                    iteration, frontier, members, join_receipt,
                    frontier_path=(frontier,),
                    route=route_receipt if isinstance(entry, ProgramRoute) else None,
                ))
        assert execution is not None
        return ProgramGraphExecution(
            execution.state, tuple(connections), tuple(nodes), tuple(joins), tuple(dispatches), tuple(routes)
        )

    def execute_repeated(
        self,
        program_id: str,
        *,
        iterations: int,
        contexts: Mapping[str, Tensor] | None = None,
        credit_masks: Mapping[str, Tensor] | None = None,
    ) -> tuple[tuple[ConnectionExecution, ...], ...]:
        """Execute a fixed number of calls without redefining event semantics.

        Data-dependent exit remains a Formula/program concern.  This helper
        only makes a statically declared local loop explicit for reference
        execution and later fixed-horizon lowering.
        """

        if type(iterations) is not int or iterations <= 0:
            raise ResourceGraphError("iterations must be a positive integer")
        return tuple(
            self.execute_program(program_id, contexts=contexts, credit_masks=credit_masks)
            for _ in range(iterations)
        )

    def execute_loop_functional(
        self,
        loop_id: str,
        *,
        state: ProgramGraphState | None = None,
        input_views: Mapping[str, TensorView] | None = None,
        contexts: Mapping[str, Tensor] | None = None,
        credit_masks: Mapping[str, Tensor] | None = None,
        candidate_selections: Mapping[str, str] | None = None,
        sample_routes: bool = False,
        host_step_limit: int | None = None,
    ) -> ProgramLoopExecution:
        """Run a tensor-controlled loop without mutating this graph.

        Finite loops retain their fixed-horizon training semantics. An open
        horizon executes only actual iterations; its host limit is supplied
        per invocation rather than baked into the program or compilation.
        """

        credit_masks = _credit_mask_map(credit_masks)
        loop = self.loop(loop_id)
        if loop.max_iterations is None:
            if type(host_step_limit) is not int or host_step_limit < loop.min_iterations:
                raise ResourceGraphError(
                    "open-horizon ProgramLoop requires host_step_limit >= min_iterations"
                )
        elif host_step_limit is not None:
            raise ResourceGraphError("host_step_limit applies only to open-horizon ProgramLoop")
        horizon = loop.max_iterations if loop.max_iterations is not None else host_step_limit
        assert horizon is not None
        current = self.state() if state is None else state
        previous_resources = self._restore_resources(current)
        active = _loop_continue_mask(
            previous_resources[loop.continue_resource_id].resolve().view
        ).new_ones(
            previous_resources[loop.continue_resource_id].resolve().view.value.shape,
            dtype=torch.bool,
        )
        masks: list[Tensor] = []
        connections: list[ConnectionExecution] = []
        nodes: list[ProgramNodeExecution | MultiPortProgramNodeExecution] = []
        joins: list[ProgramJoinExecution] = []
        dispatches: list[ProgramDispatch] = []
        routes: list[ProgramRouteExecution] = []
        termination_reason: Literal["fixed_horizon", "endogenous", "host_limit"] = "fixed_horizon"
        for iteration in range(horizon):
            masks.append(active)
            candidate = self.execute_program_functional(
                loop.program_id,
                state=current,
                input_views=input_views if iteration == 0 else None,
                contexts=contexts,
                credit_masks=credit_masks,
                candidate_selections=candidate_selections,
                sample_routes=sample_routes,
                active_rows=active,
            )
            candidate_resources = self._restore_resources(candidate.state)
            next_resources: dict[str, TensorResource] = {}
            for resource_id, prior in previous_resources.items():
                candidate_resource = candidate_resources[resource_id]
                next_resource = TensorResource.restore(prior.state())
                if candidate_resource.resolve().binding.epoch > prior.resolve().binding.epoch:
                    view = _mask_loop_publication(
                        prior.resolve().view, candidate_resource.resolve().view, active
                    )
                    next_resource.advance(view)
                next_resources[resource_id] = next_resource
            candidate_cursors = self._join_cursors_from_state(candidate.state, candidate_resources)
            next_cursors: dict[str, dict[str, int]] = {}
            for join_id, cursors in candidate_cursors.items():
                join = self._joins[join_id]
                node = self._join_node(join)
                next_cursors[join_id] = {}
                for name, port in node.input_ports.items():
                    resource_id = port.resource_id
                    candidate_epoch = candidate_resources[resource_id].resolve().binding.epoch
                    next_epoch = next_resources[resource_id].resolve().binding.epoch
                    next_cursors[join_id][name] = (
                        next_epoch - 1 if candidate_epoch > cursors[name] else next_epoch
                    )
            current = self._state_from_resources(next_resources, join_cursors=next_cursors)
            previous_resources = next_resources
            continuation = _loop_continue_mask(
                previous_resources[loop.continue_resource_id].resolve().view
            )
            if iteration + 1 >= loop.min_iterations:
                active = active & continuation
            connections.extend(candidate.connections)
            nodes.extend(candidate.nodes)
            joins.extend(candidate.joins)
            routes.extend(replace(route, iteration=iteration) for route in candidate.routes)
            dispatches.extend(
                replace(
                    dispatch, iteration=iteration,
                    route=(
                        None if dispatch.route is None
                        else replace(dispatch.route, iteration=iteration)
                    ),
                )
                for dispatch in candidate.dispatches
            )
            if loop.max_iterations is None and not bool(active.any().item()):
                termination_reason = "endogenous"
                break
        if loop.max_iterations is None and termination_reason != "endogenous":
            termination_reason = "host_limit"
        return ProgramLoopExecution(
            current, tuple(masks), tuple(connections), tuple(nodes),
            tuple(joins), tuple(dispatches), tuple(routes), termination_reason,
            tuple((candidate_selections or {}).items()),
        )

    def execute(
        self,
        connection_ids: Sequence[str],
        *,
        contexts: Mapping[str, Tensor] | None = None,
        credit_masks: Mapping[str, Tensor] | None = None,
    ) -> tuple[ConnectionExecution, ...]:
        """Run a declared serial relation sequence with explicit state visibility."""

        contexts = {} if contexts is None else contexts
        credit_masks = _credit_mask_map(credit_masks)
        if not isinstance(contexts, Mapping):
            raise TypeError("contexts must be a mapping or None")
        result: list[ConnectionExecution] = []
        for connection in self._connection_sequence(connection_ids):
            source_snapshot = self.resource(connection.source.resource_id).resolve()
            destination = self.resource(connection.destination.resource_id)
            destination_entry = destination.resolve()
            resources = {resource_id: resource.resolve() for resource_id, resource in self._resources.items()}
            credit_mask = _execution_credit_mask(
                connection, credit_masks.get(connection.connection_id), source_snapshot.view.value,
                samples=credit_masks,
            )
            output = connection.apply(
                source_snapshot,
                destination_entry,
                resources=resources,
                context=contexts.get(connection.connection_id),
                credit_mask=credit_mask,
            )
            destination_snapshot = destination.advance(output)
            result.append(
                ConnectionExecution(
                    connection.connection_id,
                    source_snapshot.binding,
                    destination_snapshot.binding,
                    destination_entry.binding,
                    credit_mask=credit_mask,
                    operands=tuple(
                        (name, resources[view.resource_id].binding)
                        for name, view in connection.operand_views.items()
                    ),
                    context=contexts.get(connection.connection_id),
                )
            )
        return tuple(result)

    def execute_functional(
        self,
        connection_ids: Sequence[str],
        *,
        state: ProgramGraphState | None = None,
        input_views: Mapping[str, TensorView] | None = None,
        contexts: Mapping[str, Tensor] | None = None,
        credit_masks: Mapping[str, Tensor] | None = None,
    ) -> ProgramGraphExecution:
        """Evaluate a serial relation sequence without mutating this graph.

        This is the branch-safe counterpart to :meth:`execute`.  It preserves
        the same direct, conditional, region, and dependency semantics while
        making the resulting resource state an explicit value for a caller's
        K-wide search or later commit policy. ``input_views`` are functionally
        hot-mounted for this one branch; live resource bindings remain intact.
        """

        return self._execute_steps_functional(
            connection_ids,
            state=state,
            input_views=input_views,
            contexts=contexts,
            credit_masks=credit_masks,
            direct_only=True,
        )

    def _execute_steps_functional(
        self,
        step_ids: Sequence[str],
        *,
        state: ProgramGraphState | None,
        input_views: Mapping[str, TensorView] | None,
        contexts: Mapping[str, Tensor] | None,
        credit_masks: Mapping[str, Tensor] | None,
        candidate_selections: Mapping[str, str] | None = None,
        direct_only: bool = False,
        completed_connections: set[str] | None = None,
    ) -> ProgramGraphExecution:
        """Evaluate graph steps without mutating live resources.

        ``direct_only`` protects the historical connection-only API.  Declared
        programs call the same executor without that restriction, so direct
        edges and local routed regions observe exactly one evolving snapshot.
        """

        contexts = {} if contexts is None else contexts
        credit_masks = _credit_mask_map(credit_masks)
        input_views = {} if input_views is None else input_views
        if not isinstance(contexts, Mapping) or not isinstance(input_views, Mapping):
            raise TypeError("contexts and input_views must be mappings or None")
        entry = self.state() if state is None else state
        resources = self._restore_resources(entry)
        join_cursors = self._join_cursors_from_state(entry, resources)
        for resource_id, input_view in input_views.items():
            if not isinstance(resource_id, str) or not isinstance(input_view, TensorView):
                raise TypeError("input_views must map resource ids to TensorView values")
            try:
                resource = resources[resource_id]
            except KeyError as error:
                raise ResourceGraphError(
                    f"functional input names unknown resource {resource_id!r}"
                ) from error
            if resource.mounted:
                resource.replace(input_view)
            else:
                resource.mount(input_view)
        if direct_only:
            steps: tuple[Connection | ProgramNode | MultiPortProgramNode, ...] = (
                self._connection_sequence(step_ids)
            )
        else:
            steps = self._program_steps(
                step_ids, completed_connections=completed_connections
            )
        connections: list[ConnectionExecution] = []
        nodes: list[ProgramNodeExecution | MultiPortProgramNodeExecution] = []
        for step in steps:
            if isinstance(step, Connection):
                source_snapshot = resources[step.source.resource_id].resolve()
                destination = resources[step.destination.resource_id]
                destination_entry = destination.resolve()
                snapshots = {
                    resource_id: resource.resolve() for resource_id, resource in resources.items()
                }
                credit_mask = _execution_credit_mask(
                    step, credit_masks.get(step.connection_id), source_snapshot.view.value,
                    samples=credit_masks,
                )
                output = step.apply(
                    source_snapshot,
                    destination_entry,
                    resources=snapshots,
                    context=contexts.get(step.connection_id),
                    credit_mask=credit_mask,
                )
                destination_snapshot = destination.advance(output)
                connections.append(
                    ConnectionExecution(
                        step.connection_id,
                        source_snapshot.binding,
                        destination_snapshot.binding,
                        destination_entry.binding,
                        credit_mask=credit_mask,
                        operands=tuple(
                            (name, snapshots[view.resource_id].binding)
                            for name, view in step.operand_views.items()
                        ),
                        context=contexts.get(step.connection_id),
                    )
                )
                continue

            if isinstance(step, ProgramNode):
                source_snapshot = resources[step.input_resource_id].resolve()
                destination = resources[step.output_resource_id]
                result = step.invoke(source_snapshot.view)
                if not isinstance(result, ProgramNodeInvocation):
                    raise TypeError("ProgramNode.invoke must return ProgramNodeInvocation")
                destination_snapshot = destination.advance(result.output)
                nodes.append(
                    ProgramNodeExecution(
                        step.node_id,
                        source_snapshot.binding,
                        destination_snapshot.binding,
                        result.receipt,
                    )
                )
                continue

            snapshots = {
                resource_id: resource.resolve() for resource_id, resource in resources.items()
            }
            inputs = {
                name: snapshots[port.resource_id].view
                for name, port in step.input_ports.items()
            }
            candidate_id = (candidate_selections or {}).get(step.node_id)
            if candidate_id is not None:
                if not isinstance(step, DifferentiableFabricNode):
                    raise ResourceGraphError(
                        f"candidate selection names non-differentiable node {step.node_id!r}"
                    )
                result = step.invoke_candidate_ports(candidate_id, inputs)
            else:
                result = step.invoke_ports(inputs)
            if not isinstance(result, MultiPortProgramNodeInvocation):
                raise TypeError(
                    "MultiPortProgramNode.invoke_ports must return "
                    "MultiPortProgramNodeInvocation"
                )
            if set(result.outputs) != set(step.output_ports):
                raise ResourceGraphError(
                    f"multi-port node {step.node_id!r} must return exactly its declared outputs"
                )
            for name, port in step.output_ports.items():
                output = result.outputs[name]
                destination = resources[port.resource_id]
                destination.spec.validate(output, name=f"node {step.node_id}.{name}")
            published: list[tuple[str, ResourceBinding]] = []
            for name, port in step.output_ports.items():
                output = result.outputs[name]
                destination = resources[port.resource_id]
                destination_snapshot = destination.advance(output)
                published.append((name, destination_snapshot.binding))
            nodes.append(
                MultiPortProgramNodeExecution(
                    step.node_id,
                    tuple(
                        (name, snapshots[port.resource_id].binding)
                        for name, port in step.input_ports.items()
                    ),
                    tuple(published),
                    result.receipt,
                )
            )
        return ProgramGraphExecution(
            self._state_from_resources(resources, join_cursors=join_cursors), tuple(connections), tuple(nodes)
        )

    def invoke_functional(
        self,
        connection_ids: Sequence[str],
        *,
        input_resource_id: str,
        input_view: TensorView,
        output_resource_id: str,
        state: ProgramGraphState | None = None,
        contexts: Mapping[str, Tensor] | None = None,
        iterations: int = 1,
    ) -> ProgramGraphInvocation:
        """Call a declared graph fragment against one branch-local input view.

        This is a convenience boundary for callers such as a larger program or
        federation.  It declares which resource receives the incoming view and
        which resource supplies the outgoing view; the graph retains ownership
        of all intermediate resource bindings and connections.
        """

        if not isinstance(input_resource_id, str) or not isinstance(output_resource_id, str):
            raise TypeError("input_resource_id and output_resource_id must be strings")
        if not isinstance(input_view, TensorView):
            raise TypeError("input_view must be TensorView")
        if input_resource_id not in self._resources or output_resource_id not in self._resources:
            raise ResourceGraphError("subprogram input and output must name graph resources")
        if type(iterations) is not int or iterations <= 0:
            raise ResourceGraphError("iterations must be a positive integer")
        execution: ProgramGraphExecution | None = None
        for iteration in range(iterations):
            execution = self.execute_functional(
                connection_ids,
                state=state if iteration == 0 else execution.state,
                input_views={input_resource_id: input_view} if iteration == 0 else None,
                contexts=contexts,
            )
        assert execution is not None
        for resource_state in execution.state.resources:
            if resource_state.spec.resource_id == output_resource_id:
                return ProgramGraphInvocation(
                    resource_state.active_view,
                    execution.state,
                    execution.connections,
                    execution.nodes,
                )
        raise AssertionError("validated output resource was missing from invocation state")

    def execute_parallel(
        self,
        connection_ids: Sequence[str],
        *,
        contexts: Mapping[str, Tensor] | None = None,
        credit_masks: Mapping[str, Tensor] | None = None,
        merge: Mapping[str, Callable[[TensorView, tuple[TensorView, ...]], TensorView]] | None = None,
    ) -> tuple[ConnectionExecution, ...]:
        """Run connections from a common resource snapshot, then publish outputs.

        Multiple writes to one destination require an explicit merge operation.
        This preserves the distinction between cooperative parallel branches and
        an accidental sequential write ordering.
        """

        contexts = {} if contexts is None else contexts
        credit_masks = _credit_mask_map(credit_masks)
        merge = {} if merge is None else merge
        if not isinstance(contexts, Mapping) or not isinstance(merge, Mapping):
            raise TypeError("contexts and merge must be mappings or None")
        connections = tuple(self.connection(connection_id) for connection_id in connection_ids)
        group_ids = {connection.connection_id for connection in connections}
        conflicting = {
            connection.connection_id: sorted(group_ids.intersection(connection.depends_on))
            for connection in connections
            if group_ids.intersection(connection.depends_on)
        }
        if conflicting:
            raise ResourceGraphError(
                "parallel connections cannot depend on another member of the same frontier: "
                f"{conflicting!r}"
            )
        snapshots = {resource_id: resource.resolve() for resource_id, resource in self._resources.items()}
        outputs: dict[str, list[tuple[Connection, ResourceSnapshot, TensorView]]] = {}
        applied_credit_masks: dict[str, Tensor | None] = {}
        for connection in connections:
            source_snapshot = snapshots[connection.source.resource_id]
            destination_snapshot = snapshots[connection.destination.resource_id]
            credit_mask = _execution_credit_mask(
                connection, credit_masks.get(connection.connection_id), source_snapshot.view.value,
                samples=credit_masks,
            )
            applied_credit_masks[connection.connection_id] = credit_mask
            output = connection.apply(
                source_snapshot,
                destination_snapshot,
                resources=snapshots,
                context=contexts.get(connection.connection_id),
                credit_mask=credit_mask,
            )
            outputs.setdefault(connection.destination.resource_id, []).append(
                (connection, source_snapshot, output)
            )
        published: dict[str, ResourceSnapshot] = {}
        for destination_id, writes in outputs.items():
            destination = self.resource(destination_id)
            previous = snapshots[destination_id].view
            values = tuple(output for _, _, output in writes)
            if len(values) == 1:
                next_view = values[0]
            else:
                try:
                    next_view = merge[destination_id](previous, values)
                except KeyError as error:
                    raise ResourceGraphError(
                        f"parallel writes to {destination_id!r} require an explicit merge"
                    ) from error
                if not isinstance(next_view, TensorView):
                    raise TypeError("parallel merge must return TensorView")
            published[destination_id] = destination.advance(
                next_view, expected_epoch=snapshots[destination_id].binding.epoch
            )
        return tuple(
            ConnectionExecution(
                connection.connection_id,
                snapshots[connection.source.resource_id].binding,
                published[connection.destination.resource_id].binding,
                snapshots[connection.destination.resource_id].binding,
                credit_mask=applied_credit_masks[connection.connection_id],
                operands=tuple(
                    (name, snapshots[view.resource_id].binding)
                    for name, view in connection.operand_views.items()
                ),
                context=contexts.get(connection.connection_id),
            )
            for connection in connections
        )

    def execute_parallel_functional(
        self,
        connection_ids: Sequence[str],
        *,
        state: ProgramGraphState | None = None,
        input_views: Mapping[str, TensorView] | None = None,
        contexts: Mapping[str, Tensor] | None = None,
        credit_masks: Mapping[str, Tensor] | None = None,
        merge: Mapping[str, Callable[[TensorView, tuple[TensorView, ...]], TensorView]] | None = None,
    ) -> ProgramGraphExecution:
        """Evaluate one common-snapshot parallel frontier without live mutation.

        The returned state is a branch proposal.  As with :meth:`execute_parallel`,
        concurrent writes require a declared merge instead of inheriting an
        accidental Python or kernel execution order.
        """

        contexts = {} if contexts is None else contexts
        credit_masks = _credit_mask_map(credit_masks)
        input_views = {} if input_views is None else input_views
        merge = {} if merge is None else merge
        if (
            not isinstance(contexts, Mapping)
            or not isinstance(input_views, Mapping)
            or not isinstance(merge, Mapping)
        ):
            raise TypeError("contexts, input_views, and merge must be mappings or None")
        entry = self.state() if state is None else state
        resources = self._restore_resources(entry)
        for resource_id, input_view in input_views.items():
            if not isinstance(resource_id, str) or not isinstance(input_view, TensorView):
                raise TypeError("input_views must map resource ids to TensorView values")
            try:
                resource = resources[resource_id]
            except KeyError as error:
                raise ResourceGraphError(
                    f"functional input names unknown resource {resource_id!r}"
                ) from error
            if resource.mounted:
                resource.replace(input_view)
            else:
                resource.mount(input_view)
        connections = tuple(self.connection(connection_id) for connection_id in connection_ids)
        group_ids = {connection.connection_id for connection in connections}
        conflicting = {
            connection.connection_id: sorted(group_ids.intersection(connection.depends_on))
            for connection in connections
            if group_ids.intersection(connection.depends_on)
        }
        if conflicting:
            raise ResourceGraphError(
                "parallel connections cannot depend on another member of the same frontier: "
                f"{conflicting!r}"
            )
        snapshots = {resource_id: resource.resolve() for resource_id, resource in resources.items()}
        outputs: dict[str, list[tuple[Connection, TensorView]]] = {}
        applied_credit_masks: dict[str, Tensor | None] = {}
        for connection in connections:
            credit_mask = _execution_credit_mask(
                connection, credit_masks.get(connection.connection_id),
                snapshots[connection.source.resource_id].view.value,
                samples=credit_masks,
            )
            applied_credit_masks[connection.connection_id] = credit_mask
            output = connection.apply(
                snapshots[connection.source.resource_id],
                snapshots[connection.destination.resource_id],
                resources=snapshots,
                context=contexts.get(connection.connection_id),
                credit_mask=credit_mask,
            )
            outputs.setdefault(connection.destination.resource_id, []).append((connection, output))
        published: dict[str, ResourceSnapshot] = {}
        for destination_id, writes in outputs.items():
            previous = snapshots[destination_id].view
            values = tuple(output for _connection, output in writes)
            if len(values) == 1:
                next_view = values[0]
            else:
                try:
                    next_view = merge[destination_id](previous, values)
                except KeyError as error:
                    raise ResourceGraphError(
                        f"parallel writes to {destination_id!r} require an explicit merge"
                    ) from error
                if not isinstance(next_view, TensorView):
                    raise TypeError("parallel merge must return TensorView")
            published[destination_id] = resources[destination_id].advance(
                next_view,
                expected_epoch=snapshots[destination_id].binding.epoch,
            )
        executions = tuple(
            ConnectionExecution(
                connection.connection_id,
                snapshots[connection.source.resource_id].binding,
                published[connection.destination.resource_id].binding,
                snapshots[connection.destination.resource_id].binding,
                credit_mask=applied_credit_masks[connection.connection_id],
                operands=tuple(
                    (name, snapshots[view.resource_id].binding)
                    for name, view in connection.operand_views.items()
                ),
                context=contexts.get(connection.connection_id),
            )
            for connection in connections
        )
        return ProgramGraphExecution(self._state_from_resources(resources), executions)


class ResourceGraphExecutionPlan(nn.Module):
    """A functional, fixed-connection lowering of a ProgramGraph fragment.

    Resource values are graph inputs and outputs.  The plan therefore preserves
    normal autograd and can be passed to ``torch.compile`` without turning an
    in-memory TensorResource mutation into a hidden Python side effect.
    """

    def __init__(
        self,
        *,
        resource_ids: Sequence[str],
        templates: Sequence[TensorView],
        connections: Sequence[Connection],
    ) -> None:
        super().__init__()
        self.resource_ids = tuple(resource_ids)
        self.templates = tuple(_clone_view(template) for template in templates)
        self.connections = nn.ModuleList(tuple(connections))
        if not self.resource_ids or len(self.resource_ids) != len(self.templates):
            raise ResourceGraphCompileError("compiled plans require one template for each resource")
        if len(set(self.resource_ids)) != len(self.resource_ids):
            raise ResourceGraphCompileError("compiled resource ids must be unique")
        index = {resource_id: position for position, resource_id in enumerate(self.resource_ids)}
        self._source_indices = tuple(index[connection.source.resource_id] for connection in self.connections)
        self._destination_indices = tuple(index[connection.destination.resource_id] for connection in self.connections)
        source_templates: list[TensorView] = []
        source_slices: list[tuple[slice, ...] | None] = []
        destination_slices: list[tuple[slice, ...] | None] = []
        operand_specs: list[tuple[tuple[str, int, tuple[slice, ...] | None, TensorView], ...]] = []
        for connection, source_index, destination_index in zip(
            self.connections,
            self._source_indices,
            self._destination_indices,
            strict=True,
        ):
            source_template = self.templates[source_index]
            source_snapshot = ResourceSnapshot(
                source_template,
                ResourceBinding(connection.source.resource_id, "default", 0, 0),
            )
            if connection.source_view is None:
                source_templates.append(source_template)
                source_slices.append(None)
            else:
                source_templates.append(connection.source_view.resolve(source_snapshot))
                _source, source_slice = connection.source_view._slices(source_snapshot)
                source_slices.append(source_slice)
            if connection.destination_view is None:
                destination_slices.append(None)
            else:
                destination_snapshot = ResourceSnapshot(
                    self.templates[destination_index],
                    ResourceBinding(connection.destination.resource_id, "default", 0, 0),
                )
                _destination, destination_slice = connection.destination_view._slices(
                    destination_snapshot
                )
                destination_slices.append(destination_slice)
            connection_operands: list[tuple[str, int, tuple[slice, ...] | None, TensorView]] = []
            for name, resource_view in sorted(connection.operand_views.items()):
                resource_index = index[resource_view.resource_id]
                operand_snapshot = ResourceSnapshot(
                    self.templates[resource_index],
                    ResourceBinding(resource_view.resource_id, "default", 0, 0),
                )
                operand_template = resource_view.resolve(operand_snapshot)
                _operand, operand_slice = resource_view._slices(operand_snapshot)
                connection_operands.append((name, resource_index, operand_slice, operand_template))
            operand_specs.append(tuple(connection_operands))
        self._source_templates = tuple(source_templates)
        self._source_slices = tuple(source_slices)
        self._destination_slices = tuple(destination_slices)
        self._operand_specs = tuple(operand_specs)
        self._conditional_indices = tuple(
            index for index, connection in enumerate(self.connections) if connection.is_conditional
        )
        conditional_positions = {
            connection_index: position
            for position, connection_index in enumerate(self._conditional_indices)
        }
        self._context_positions = tuple(
            conditional_positions.get(connection_index, -1)
            for connection_index in range(len(self.connections))
        )
        self._credit_mask_indices = tuple(
            index
            for index, connection in enumerate(self.connections)
            if connection.credit_boundary is not None
            and connection.credit_boundary.mode is CreditBoundaryMode.BERNOULLI
        )
        credit_mask_positions = {
            connection_index: position
            for position, connection_index in enumerate(self._credit_mask_indices)
        }
        self._credit_mask_positions = tuple(
            credit_mask_positions.get(connection_index, -1)
            for connection_index in range(len(self.connections))
        )

    @property
    def context_connection_ids(self) -> tuple[str, ...]:
        return tuple(self.connections[index].connection_id for index in self._conditional_indices)

    @property
    def credit_mask_connection_ids(self) -> tuple[str, ...]:
        """Bernoulli boundary ids whose replay masks are tensor-plan inputs."""

        return tuple(self.connections[index].connection_id for index in self._credit_mask_indices)

    def capture(self, *inputs: Tensor) -> "CapturedResourceGraphExecutionPlan":
        """Capture one fixed CUDA resource bucket for inference replay.

        The resource graph remains a functional tensor program: capture only
        fixes physical addresses for a declared input/context bucket.  It does
        not make dynamic resource bindings or conditional graph relations
        compile-compatible.
        """

        expected = len(self.resource_ids) + len(self._conditional_indices) + len(self._credit_mask_indices)
        if len(inputs) != expected:
            raise ResourceGraphCompileError("CUDA Graph capture received an incorrect input count")
        if any(not isinstance(value, Tensor) for value in inputs):
            raise TypeError("CUDA Graph capture inputs must be tensors")
        first = inputs[0]
        if not first.is_cuda:
            raise ResourceGraphCompileError("CUDA Graph capture requires CUDA inputs")
        if any(
            not value.is_cuda
            or value.device != first.device
            or value.requires_grad
            for value in inputs
        ):
            raise ResourceGraphCompileError(
                "CUDA Graph capture requires no-grad inputs on one CUDA device"
            )

        static_inputs = tuple(value.detach().clone() for value in inputs)
        device = first.device
        warmup_stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(warmup_stream), torch.no_grad():
            self(*static_inputs)
        torch.cuda.current_stream(device).wait_stream(warmup_stream)
        torch.cuda.synchronize(device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph), torch.no_grad():
            outputs = self(*static_inputs)
        return CapturedResourceGraphExecutionPlan(self, graph, static_inputs, outputs)

    def forward(self, *inputs: Tensor) -> tuple[Tensor, ...]:
        expected = len(self.resource_ids) + len(self._conditional_indices) + len(self._credit_mask_indices)
        if len(inputs) != expected:
            raise ResourceGraphCompileError("compiled plan received an incorrect resource/context value count")
        values = inputs[:len(self.resource_ids)]
        context_end = len(self.resource_ids) + len(self._conditional_indices)
        contexts = inputs[len(self.resource_ids):context_end]
        credit_masks = inputs[context_end:]
        views = [
            TensorView(
                value,
                template.axes,
                index_map=template.index_map,
                mask=template.mask,
            )
            for value, template in zip(values, self.templates, strict=True)
        ]
        for (
            connection,
            source_index,
            destination_index,
            source_template,
            source_slice,
            destination_slice,
            operand_spec,
            context_position,
            credit_mask_position,
        ) in zip(
            self.connections,
            self._source_indices,
            self._destination_indices,
            self._source_templates,
            self._source_slices,
            self._destination_slices,
            self._operand_specs,
            self._context_positions,
            self._credit_mask_positions,
            strict=True,
        ):
            context = None if context_position < 0 else contexts[context_position]
            credit_mask = None if credit_mask_position < 0 else credit_masks[credit_mask_position]
            source_value = (
                views[source_index].value
                if source_slice is None
                else views[source_index].value[source_slice]
            )
            source_view = TensorView(
                source_value,
                source_template.axes,
                index_map=source_template.index_map,
                mask=source_template.mask,
            )
            operands = {
                name: TensorView(
                    views[resource_index].value
                    if operand_slice is None
                    else views[resource_index].value[operand_slice],
                    operand_template.axes,
                    index_map=operand_template.index_map,
                    mask=operand_template.mask,
                )
                for name, resource_index, operand_slice, operand_template in operand_spec
            }
            output = connection(
                source_view,
                operands=operands,
                context=context,
                credit_mask=credit_mask,
            )
            if destination_slice is None:
                views[destination_index] = output
                continue
            destination = views[destination_index]
            value = destination.value.clone()
            value[destination_slice] = output.value
            if destination.mask is None:
                if output.mask is None:
                    mask = None
                else:
                    mask = torch.zeros_like(destination.value, dtype=torch.bool)
                    mask[destination_slice] = output.mask.to(dtype=torch.bool)
            elif output.mask is None:
                mask = destination.mask
            else:
                mask = destination.mask.clone()
                mask[destination_slice] = output.mask.to(dtype=torch.bool)
            views[destination_index] = TensorView(
                value,
                destination.axes,
                index_map=destination.index_map,
                mask=mask,
            )
        return tuple(view.value for view in views)

    def credit_gradient(
        self,
        *inputs: Tensor,
        terminal_cotangents: Mapping[str, Tensor],
        create_graph: bool = True,
    ) -> "StaticProgramGraphCreditResult":
        """Compose direct connection VJPs with declared port credit rules.

        This is the connection counterpart to
        :meth:`StaticProgramGraphExecutionPlan.credit_gradient`.  It keeps
        connection data relations local, applies a ``CreditBoundary`` exactly
        at the producer-to-consumer port in reverse, then accumulates source,
        operand, destination-slice, and shared-parameter cotangents by SSA
        version. Conditional routes treat their declared context as a local
        relation input and return an explicit context cotangent receipt.
        """

        expected = len(self.resource_ids) + len(self._conditional_indices) + len(self._credit_mask_indices)
        if len(inputs) != expected:
            raise ResourceGraphCompileError(
                "connection credit lowering received an incorrect resource/context value count"
            )
        if not isinstance(terminal_cotangents, Mapping) or not terminal_cotangents:
            raise TypeError("terminal_cotangents must be a non-empty resource-id mapping")
        resource_index = {resource_id: index for index, resource_id in enumerate(self.resource_ids)}
        unknown = set(terminal_cotangents).difference(resource_index)
        if unknown:
            raise ResourceGraphCompileError(
                f"terminal cotangents name unknown resources: {sorted(unknown)!r}"
            )

        values = list(inputs[: len(self.resource_ids)])
        context_end = len(self.resource_ids) + len(self._conditional_indices)
        contexts = inputs[len(self.resource_ids) : context_end]
        credit_masks = inputs[context_end:]
        versions = list(range(len(values)))
        next_version = len(values)
        tapes: list[
            tuple[
                Connection,
                int,
                tuple[int, ...],
                int | None,
                int,
                tuple[slice, ...] | None,
                Tensor,
                tuple[Tensor, ...],
                str | None,
                Tensor | None,
                tuple[nn.Parameter, ...],
                Tensor | None,
            ]
        ] = []
        parameter_names: dict[int, str] = {}
        parameter_values: dict[int, nn.Parameter] = {}

        for (
            connection,
            source_index,
            destination_index,
            source_template,
            source_slice,
            destination_slice,
            operand_spec,
            context_position,
            credit_mask_position,
        ) in zip(
            self.connections,
            self._source_indices,
            self._destination_indices,
            self._source_templates,
            self._source_slices,
            self._destination_slices,
            self._operand_specs,
            self._context_positions,
            self._credit_mask_positions,
            strict=True,
        ):
            source_version = versions[source_index]
            source_value = values[source_index] if source_slice is None else values[source_index][source_slice]
            if not (source_value.is_floating_point() or source_value.is_complex()):
                raise ResourceGraphCompileError("connection credit lowering requires floating or complex source values")
            local_source = source_value.detach().requires_grad_(True)
            local_operands: list[Tensor] = []
            operand_versions: list[int] = []
            operand_views: dict[str, TensorView] = {}
            for name, resource_position, operand_slice, operand_template in operand_spec:
                operand_value = (
                    values[resource_position]
                    if operand_slice is None
                    else values[resource_position][operand_slice]
                )
                if not (operand_value.is_floating_point() or operand_value.is_complex()):
                    raise ResourceGraphCompileError(
                        "connection credit lowering requires floating or complex operand values"
                    )
                local_operand = operand_value.detach().requires_grad_(True)
                local_operands.append(local_operand)
                operand_versions.append(versions[resource_position])
                operand_views[name] = TensorView(
                    local_operand,
                    operand_template.axes,
                    index_map=operand_template.index_map,
                    mask=operand_template.mask,
                )
            context_id: str | None = None
            local_context: Tensor | None = None
            if context_position >= 0:
                context_value = contexts[context_position]
                if not (context_value.is_floating_point() or context_value.is_complex()):
                    raise ResourceGraphCompileError(
                        "connection credit lowering requires floating or complex conditional contexts"
                    )
                context_id = connection.connection_id
                local_context = context_value.detach().requires_grad_(True)
            output = connection.data_forward(
                TensorView(
                    local_source,
                    source_template.axes,
                    index_map=source_template.index_map,
                    mask=source_template.mask,
                ),
                operands=operand_views,
                context=local_context,
            )
            if not isinstance(output, TensorView):
                raise ResourceGraphCompileError("connection data relation must return TensorView")
            previous_destination = values[destination_index]
            previous_version = versions[destination_index]
            if destination_slice is None:
                values[destination_index] = output.value
                destination_input_version: int | None = None
                local_destination: Tensor | None = None
            else:
                local_destination = previous_destination.detach()
                next_value = local_destination.clone()
                next_value[destination_slice] = output.value
                values[destination_index] = next_value
                destination_input_version = previous_version
            output_version = next_version
            next_version += 1
            versions[destination_index] = output_version
            parameter_items = tuple(
                (name, parameter)
                for name, parameter in connection.named_parameters()
                if not name.startswith("credit_boundary.")
            )
            parameters = tuple(parameter for _name, parameter in parameter_items)
            for name, parameter in parameter_items:
                identity = id(parameter)
                parameter_names.setdefault(identity, f"{connection.connection_id}.{name}")
                parameter_values[identity] = parameter
            credit_mask = None if credit_mask_position < 0 else credit_masks[credit_mask_position]
            tapes.append(
                (
                    connection,
                    source_version,
                    tuple(operand_versions),
                    destination_input_version,
                    output_version,
                    destination_slice,
                    output.value,
                    (local_source, *local_operands),
                    context_id,
                    local_context,
                    parameters,
                    credit_mask,
                )
            )

        cotangents: dict[int, Tensor] = {}
        for resource_id, cotangent in terminal_cotangents.items():
            if not isinstance(cotangent, Tensor):
                raise TypeError("terminal cotangents must be Tensors")
            index = resource_index[resource_id]
            expected_value = values[index]
            if cotangent.shape != expected_value.shape:
                raise ResourceGraphCompileError(
                    f"terminal cotangent for {resource_id!r} must match its final resource shape"
                )
            cotangents[versions[index]] = cotangent.to(
                device=expected_value.device,
                dtype=expected_value.dtype,
            )

        parameter_cotangents: dict[int, Tensor] = {}
        context_cotangents: dict[str, Tensor] = {}
        for (
            connection,
            source_version,
            operand_versions,
            destination_input_version,
            output_version,
            destination_slice,
            output_value,
            local_inputs,
            context_id,
            local_context,
            parameters,
            credit_mask,
        ) in reversed(tapes):
            output_cotangent = cotangents.get(output_version, torch.zeros_like(output_value))
            if destination_input_version is not None:
                assert destination_slice is not None
                untouched = output_cotangent.clone()
                untouched[destination_slice] = 0
                cotangents[destination_input_version] = (
                    untouched
                    if destination_input_version not in cotangents
                    else cotangents[destination_input_version] + untouched
                )
                output_cotangent = output_cotangent[destination_slice]
            if connection.credit_boundary is not None:
                scale = connection.credit_boundary.resolve_credit_scale(
                    output_value,
                    credit_mask=credit_mask,
                )
                output_cotangent = torch.where(
                    scale == 0,
                    torch.zeros_like(output_cotangent),
                    output_cotangent * scale,
                )
            gradient_inputs = (
                *local_inputs,
                *((local_context,) if local_context is not None else ()),
                *parameters,
            )
            gradients = torch.autograd.grad(
                output_value,
                gradient_inputs,
                grad_outputs=output_cotangent,
                retain_graph=True,
                create_graph=create_graph,
                allow_unused=True,
            )
            for version, input_value, cotangent in zip(
                (source_version, *operand_versions), local_inputs, gradients[: len(local_inputs)], strict=True
            ):
                if cotangent is None:
                    continue
                if cotangent.shape != input_value.shape:
                    raise ResourceGraphCompileError("connection credit input cotangent shape mismatch")
                cotangents[version] = cotangent if version not in cotangents else cotangents[version] + cotangent
            parameter_gradient_start = len(local_inputs)
            if local_context is not None:
                assert context_id is not None
                context_cotangent = gradients[parameter_gradient_start]
                parameter_gradient_start += 1
                if context_cotangent is not None:
                    if context_cotangent.shape != local_context.shape:
                        raise ResourceGraphCompileError("connection credit context cotangent shape mismatch")
                    context_cotangents[context_id] = (
                        context_cotangent
                        if context_id not in context_cotangents
                        else context_cotangents[context_id] + context_cotangent
                    )
            for parameter, cotangent in zip(
                parameters, gradients[parameter_gradient_start:], strict=True
            ):
                if cotangent is None:
                    continue
                if cotangent.shape != parameter.shape:
                    raise ResourceGraphCompileError("connection credit parameter cotangent shape mismatch")
                identity = id(parameter)
                parameter_cotangents[identity] = (
                    cotangent
                    if identity not in parameter_cotangents
                    else parameter_cotangents[identity] + cotangent
                )

        return StaticProgramGraphCreditResult(
            resource_values=tuple(values),
            resource_cotangents={
                resource_id: cotangents.get(index)
                for index, resource_id in enumerate(self.resource_ids)
            },
            parameter_cotangents={
                parameter_names[identity]: parameter_cotangents.get(identity)
                for identity in parameter_names
            },
            parameters={
                parameter_names[identity]: parameter_values[identity]
                for identity in parameter_names
            },
            context_cotangents={
                connection_id: context_cotangents.get(connection_id)
                for connection_id in self.context_connection_ids
            },
        )


class CapturedResourceGraphExecutionPlan:
    """Fixed-address CUDA replay handle for one static tensor-plan bucket."""

    _component_reference: ClassVar[str] = "arti/captured-resource-graph-execution-plan@1"

    def __init__(
        self,
        plan: nn.Module,
        graph: torch.cuda.CUDAGraph,
        static_inputs: tuple[Tensor, ...],
        outputs: tuple[Tensor, ...],
    ) -> None:
        self.plan = plan
        self.graph = graph
        self._static_inputs = static_inputs
        self._outputs = outputs

    def replay(self, *inputs: Tensor, copy_output: bool = True) -> tuple[Tensor, ...]:
        """Replay later tensors with the captured resource/context ABI."""

        if len(inputs) != len(self._static_inputs):
            raise ResourceGraphCompileError("CUDA Graph replay received an incorrect input count")
        for value, expected in zip(inputs, self._static_inputs, strict=True):
            if (
                not isinstance(value, Tensor)
                or value.requires_grad
                or value.device != expected.device
                or value.dtype != expected.dtype
                or value.shape != expected.shape
            ):
                raise ResourceGraphCompileError(
                    "CUDA Graph replay inputs must match the captured resource bucket"
                )
        with torch.no_grad():
            for destination, source in zip(self._static_inputs, inputs, strict=True):
                destination.copy_(source)
            self.graph.replay()
        if copy_output:
            return tuple(value.clone() for value in self._outputs)
        return self._outputs


def _capture_static_tensor_plan(
    plan: nn.Module, *inputs: Tensor, capture_forward: Callable[..., tuple[Tensor, ...]] | None = None,
) -> CapturedResourceGraphExecutionPlan:
    """Capture one no-grad CUDA bucket shared by every static plan variant."""

    if not inputs:
        raise ResourceGraphCompileError("CUDA Graph capture requires at least one tensor input")
    if any(not isinstance(value, Tensor) for value in inputs):
        raise TypeError("CUDA Graph capture inputs must be tensors")
    first = inputs[0]
    if not first.is_cuda:
        raise ResourceGraphCompileError("CUDA Graph capture requires CUDA inputs")
    if any(
        not value.is_cuda
        or value.device != first.device
        or value.requires_grad
        for value in inputs
    ):
        raise ResourceGraphCompileError(
            "CUDA Graph capture requires no-grad inputs on one CUDA device"
        )
    static_inputs = tuple(value.detach().clone() for value in inputs)
    forward = plan if capture_forward is None else capture_forward
    device = first.device
    warmup_stream = torch.cuda.Stream(device=device)
    with torch.cuda.stream(warmup_stream), torch.no_grad():
        forward(*static_inputs)
    torch.cuda.current_stream(device).wait_stream(warmup_stream)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph), torch.no_grad():
        outputs = forward(*static_inputs)
    return CapturedResourceGraphExecutionPlan(plan, graph, static_inputs, outputs)


def _lower_static_program_node(node: MultiPortProgramNode) -> nn.Module:
    """Lower one pure data graph region without creating a parallel executor."""

    if isinstance(node, FormulaProgramNode):
        return StaticFormulaProgramNode(node)
    # Keep the optional custom-module bridge outside this core module so an
    # ordinary resource graph has no decorator-specific import at load time.
    from .fabric_decorators import FabricModuleNode

    if isinstance(node, FabricModuleNode):
        return node.lower_static()
    raise ResourceGraphCompileError(
        "static program lowering supports FormulaProgramNode or registered FabricModuleNode regions only"
    )


@dataclass(frozen=True)
class StaticProgramGraphCreditResult:
    """Explicit credit result for one fixed, locally differentiated graph call."""

    resource_values: tuple[Tensor, ...]
    resource_cotangents: Mapping[str, Tensor | None]
    parameter_cotangents: Mapping[str, Tensor | None]
    parameters: Mapping[str, nn.Parameter]
    context_cotangents: Mapping[str, Tensor | None]


@dataclass(frozen=True)
class StaticProgramJoinCreditResult:
    """Credit result and fired-row receipt for one lowered :class:`ProgramJoin`.

    ``ready`` is control-plane evidence, not a differentiable input. The
    cotangent of a row that did not fire is routed to the pre-join destination
    value, while a fired row is lowered through the join node's declared local
    VJP.
    """

    resource_values: tuple[Tensor, ...]
    resource_cotangents: Mapping[str, Tensor | None]
    parameter_cotangents: Mapping[str, Tensor | None]
    parameters: Mapping[str, nn.Parameter]
    ready: Tensor
    remaining_arrivals: Tensor
    publications: Tensor


@dataclass(frozen=True)
class StaticProgramLoopCreditResult:
    """Credit result and per-iteration activity receipt for a bounded loop."""

    resource_values: tuple[Tensor, ...]
    resource_cotangents: Mapping[str, Tensor | None]
    parameter_cotangents: Mapping[str, Tensor | None]
    parameters: Mapping[str, nn.Parameter]
    iteration_active: tuple[Tensor, ...]


@dataclass(frozen=True)
class StaticDataflowProgramCreditResult:
    """Credit result and operation receipts for a static dataflow program."""

    resource_values: tuple[Tensor, ...]
    resource_cotangents: Mapping[str, Tensor | None]
    parameter_cotangents: Mapping[str, Tensor | None]
    parameters: Mapping[str, nn.Parameter]
    context_cotangents: Mapping[str, Tensor | None]
    arrivals: Tensor
    join_ready: tuple[Tensor | None, ...]
    route_selections: tuple[Tensor, ...] = ()


@dataclass(frozen=True)
class StaticDataflowLoopCreditResult:
    """Credit and activity receipts for a bounded dataflow loop."""

    resource_values: tuple[Tensor, ...]
    resource_cotangents: Mapping[str, Tensor | None]
    parameter_cotangents: Mapping[str, Tensor | None]
    parameters: Mapping[str, nn.Parameter]
    context_cotangents: Mapping[str, Tensor | None]
    arrivals: Tensor
    iteration_active: tuple[Tensor, ...]
    join_ready: tuple[tuple[Tensor | None, ...], ...]
    route_selections: tuple[tuple[Tensor, ...], ...] = ()
    route_log_probability: Tensor | None = None
    route_log_terms: tuple[Tensor, ...] = ()
    route_ids: tuple[str, ...] = ()

    def structure_objective(
        self, final_loss: Tensor, *, baseline: Tensor | float = 0.0,
        fate_sample: ProgramFateSample | None = None,
    ) -> Tensor:
        if self.route_log_probability is None:
            raise ResourceGraphCompileError("route structure credit requires sampled route receipts")
        if not isinstance(final_loss, Tensor) or final_loss.ndim > 1 or not final_loss.numel():
            raise ResourceGraphCompileError("route final_loss must be scalar or per-row")
        baseline_value = torch.as_tensor(baseline, device=final_loss.device, dtype=final_loss.dtype)
        if baseline_value.numel() != 1 and baseline_value.shape != final_loss.shape:
            raise ResourceGraphCompileError("route baseline must be scalar or match per-row loss")
        advantage = final_loss.detach() - baseline_value.detach()
        if final_loss.ndim == 0:
            objective = advantage * self.route_log_probability
        else:
            if len(self.route_log_terms) != len(self.iteration_active):
                raise ResourceGraphCompileError("per-row route credit requires per-iteration terms")
            objective = final_loss.new_zeros(())
            for term, active in zip(self.route_log_terms, self.iteration_active, strict=True):
                if active.shape != final_loss.shape or (
                    term.ndim and term.shape != final_loss.shape
                ):
                    raise ResourceGraphCompileError("route credit rows must match final loss")
                contribution = torch.where(active, advantage, torch.zeros_like(advantage))
                if term.ndim:
                    objective = objective + (contribution * term).sum()
                else:
                    objective = objective + term * contribution.sum() / active.sum().clamp_min(1)
        if fate_sample is None:
            return objective
        return objective + fate_sample.trajectory_structure_objective(
            final_loss, baseline=baseline, route_ids=self.route_ids,
            route_selections=self.route_selections, iteration_active=self.iteration_active,
        )


class _StaticConnectionProgramStep(nn.Module):
    """Use the existing connection lowering as one mixed-program step."""

    def __init__(self, plan: ResourceGraphExecutionPlan) -> None:
        super().__init__()
        if len(plan.connections) != 1:
            raise ResourceGraphCompileError("a mixed-program connection step must contain one relation")
        self.plan = plan
        self.connection_id = plan.connections[0].connection_id
        self.destination_id = plan.connections[0].destination.resource_id
        self._destination_index = plan.resource_ids.index(self.destination_id)

    def forward(self, *inputs: Tensor) -> tuple[Tensor, ...]:
        return (self.plan(*inputs)[self._destination_index],)

    def local_vjp(
        self,
        inputs: tuple[Tensor, ...],
        outputs: tuple[Tensor, ...],
        output_cotangents: tuple[Tensor, ...],
        parameters: tuple[nn.Parameter, ...],
        create_graph: bool,
    ) -> LocalVJPResult:
        if len(outputs) != 1 or len(output_cotangents) != 1:
            raise ResourceGraphCompileError("mixed-program connection step has one output")
        result = self.plan.credit_gradient(
            *inputs,
            terminal_cotangents={self.destination_id: output_cotangents[0]},
            create_graph=create_graph,
        )
        parameter_cotangents = {
            id(parameter): result.parameter_cotangents[name]
            for name, parameter in result.parameters.items()
        }
        input_cotangents = (
            *(result.resource_cotangents[resource_id] for resource_id in self.plan.resource_ids),
            *(result.context_cotangents[connection_id] for connection_id in self.plan.context_connection_ids),
            *(None for _ in self.plan.credit_mask_connection_ids),
        )
        return LocalVJPResult(
            tuple(input_cotangents),
            tuple(parameter_cotangents.get(id(parameter)) for parameter in parameters),
        )


class StaticProgramGraphExecutionPlan(nn.Module):
    """SSA-style tensor lowering for a frozen connection and Formula subgraph.

    The fixed program sequence is visible to ``torch.export`` and
    ``torch.compile``; it contains no runtime resource lookup or routed-node
    callback.
    """

    _component_reference: ClassVar[str] = "arti/static-program-graph-execution-plan@1"

    def __init__(
        self,
        *,
        resource_ids: Sequence[str],
        stages: Sequence[Sequence[MultiPortProgramNode | Connection]],
        connection_plans: Mapping[str, ResourceGraphExecutionPlan] | None = None,
    ) -> None:
        super().__init__()
        self.resource_ids = tuple(resource_ids)
        if len(set(self.resource_ids)) != len(self.resource_ids):
            raise ResourceGraphCompileError("static program resources must be unique")
        if not stages or any(not stage for stage in stages):
            raise ResourceGraphCompileError("static program requires non-empty program stages")
        plans = {} if connection_plans is None else dict(connection_plans)
        connection_steps: dict[str, _StaticConnectionProgramStep] = {}
        for stage in stages:
            for step in stage:
                if isinstance(step, Connection) and step.connection_id not in connection_steps:
                    connection_steps[step.connection_id] = _StaticConnectionProgramStep(
                        plans[step.connection_id]
                    )
        self.context_connection_ids = tuple(
            connection_id for connection_id, step in connection_steps.items()
            if step.plan.context_connection_ids
        )
        self.credit_mask_connection_ids = tuple(
            connection_id for connection_id, step in connection_steps.items()
            if step.plan.credit_mask_connection_ids
        )
        self.stages = nn.ModuleList(
            nn.ModuleList(
                connection_steps[step.connection_id]
                if isinstance(step, Connection) else _lower_static_program_node(step)
                for step in stage
            )
            for stage in stages
        )
        self._node_ids = tuple(
            tuple(step.connection_id if isinstance(step, Connection) else step.node_id for step in stage)
            for stage in stages
        )
        index = {resource_id: position for position, resource_id in enumerate(self.resource_ids)}
        context_index = {
            connection_id: len(index) + position
            for position, connection_id in enumerate(self.context_connection_ids)
        }
        mask_index = {
            connection_id: len(index) + len(context_index) + position
            for position, connection_id in enumerate(self.credit_mask_connection_ids)
        }
        self._input_indices = tuple(
            tuple(
                (
                    (*range(len(index)),
                     *((context_index[step.connection_id],) if step.connection_id in context_index else ()),
                     *((mask_index[step.connection_id],) if step.connection_id in mask_index else ()))
                    if isinstance(step, Connection)
                    else tuple(index[port.resource_id] for port in step.input_ports.values())
                )
                for step in stage
            )
            for stage in stages
        )
        self._output_indices = tuple(
            tuple(
                (index[step.destination.resource_id],) if isinstance(step, Connection)
                else tuple(index[port.resource_id] for port in step.output_ports.values())
                for step in stage
            )
            for stage in stages
        )

    def forward(self, *inputs: Tensor) -> tuple[Tensor, ...]:
        expected = len(self.resource_ids) + len(self.context_connection_ids) + len(self.credit_mask_connection_ids)
        if len(inputs) != expected:
            raise ResourceGraphCompileError(
                "static program graph received an incorrect resource/context value count"
            )
        values = list(inputs)
        for stage, stage_inputs, stage_outputs in zip(
            self.stages, self._input_indices, self._output_indices, strict=True
        ):
            snapshot = tuple(values)
            publications: list[tuple[int, Tensor]] = []
            for node, input_indices, output_indices in zip(
                stage, stage_inputs, stage_outputs, strict=True
            ):
                outputs = node(*(snapshot[index] for index in input_indices))
                publications.extend(zip(output_indices, outputs, strict=True))
            for index, value in publications:
                values[index] = value
        return tuple(values[:len(self.resource_ids)])

    def credit_gradient(
        self,
        *inputs: Tensor,
        terminal_cotangents: Mapping[str, Tensor],
        create_graph: bool = True,
    ) -> "StaticProgramGraphCreditResult":
        """Lower fixed program-node credit through declared local VJPs.

        The forward values are deliberately detached at every node boundary.
        This makes each module pullback local rather than accidentally letting
        one ``autograd.grad`` traverse the whole graph.  The plan then composes
        the declared VJPs over the same static SSA publication order used by
        :meth:`forward`.

        Join readiness and dynamic route credit use their own receipts.
        """

        expected_count = len(self.resource_ids) + len(self.context_connection_ids) + len(self.credit_mask_connection_ids)
        if len(inputs) != expected_count:
            raise ResourceGraphCompileError(
                "static program graph credit lowering received an incorrect resource/context value count"
            )
        if not isinstance(terminal_cotangents, Mapping) or not terminal_cotangents:
            raise TypeError("terminal_cotangents must be a non-empty resource-id mapping")
        resource_index = {resource_id: index for index, resource_id in enumerate(self.resource_ids)}
        unknown = set(terminal_cotangents).difference(resource_index)
        if unknown:
            raise ResourceGraphCompileError(
                f"terminal cotangents name unknown resources: {sorted(unknown)!r}"
            )

        values = list(inputs)
        versions = list(range(len(values)))
        next_version = len(values)
        tapes: list[
            tuple[
                str,
                nn.Module,
                tuple[int, ...],
                tuple[int, ...],
                tuple[Tensor, ...],
                tuple[Tensor, ...],
                tuple[nn.Parameter, ...],
            ]
        ] = []
        parameter_names: dict[int, str] = {}
        parameter_values: dict[int, nn.Parameter] = {}
        for stage, stage_ids, stage_inputs, stage_outputs in zip(
            self.stages, self._node_ids, self._input_indices, self._output_indices, strict=True
        ):
            snapshot_values = tuple(values)
            snapshot_versions = tuple(versions)
            publications: list[tuple[int, int, Tensor]] = []
            for node, node_id, input_indices, output_indices in zip(
                stage, stage_ids, stage_inputs, stage_outputs, strict=True
            ):
                local_inputs = tuple(
                    value.detach().requires_grad_(value.is_floating_point() or value.is_complex())
                    for value in (snapshot_values[index] for index in input_indices)
                )
                if not isinstance(node, _StaticConnectionProgramStep) and any(
                    not value.requires_grad for value in local_inputs
                ):
                    raise ResourceGraphCompileError(
                        "static graph credit lowering requires floating or complex node inputs"
                    )
                outputs = tuple(node(*local_inputs))
                if len(outputs) != len(output_indices):
                    raise ResourceGraphCompileError("static graph node returned an incorrect output count")
                parameters = tuple(node.parameters())
                for name, parameter in node.named_parameters():
                    parameter_id = id(parameter)
                    parameter_names.setdefault(
                        parameter_id,
                        f"{node_id}.{name.removeprefix('plan.connections.0.').removeprefix('module.')}",
                    )
                    parameter_values[parameter_id] = parameter
                output_versions = tuple(range(next_version, next_version + len(outputs)))
                next_version += len(outputs)
                tapes.append(
                    (
                        node_id,
                        node,
                        tuple(snapshot_versions[index] for index in input_indices),
                        output_versions,
                        local_inputs,
                        outputs,
                        parameters,
                    )
                )
                for resource_id, version, output in zip(output_indices, output_versions, outputs, strict=True):
                    publications.append((resource_id, version, output))
            for resource_id, version, output in publications:
                values[resource_id] = output
                versions[resource_id] = version

        cotangents: dict[int, Tensor] = {}
        for resource_id, cotangent in terminal_cotangents.items():
            if not isinstance(cotangent, Tensor):
                raise TypeError("terminal cotangents must be Tensors")
            index = resource_index[resource_id]
            expected = values[index]
            if cotangent.shape != expected.shape:
                raise ResourceGraphCompileError(
                    f"terminal cotangent for {resource_id!r} must match its final resource shape"
                )
            cotangents[versions[index]] = cotangent.to(device=expected.device, dtype=expected.dtype)

        parameter_cotangents: dict[int, Tensor] = {}
        for _node_id, node, input_versions, output_versions, local_inputs, outputs, parameters in reversed(tapes):
            output_cotangents = tuple(
                cotangents.get(version, torch.zeros_like(output))
                for version, output in zip(output_versions, outputs, strict=True)
            )
            local_vjp = getattr(node, "local_vjp", None)
            if not callable(local_vjp):
                raise ResourceGraphCompileError(
                    "static graph credit lowering requires every lowered node to declare local_vjp"
                )
            result = local_vjp(
                local_inputs,
                outputs,
                output_cotangents,
                parameters,
                create_graph,
            )
            if not isinstance(result, LocalVJPResult):
                raise ResourceGraphCompileError("node local_vjp must return LocalVJPResult")
            if len(result.input_cotangents) != len(input_versions):
                raise ResourceGraphCompileError("node local_vjp returned an incorrect input cotangent count")
            if len(result.parameter_cotangents) != len(parameters):
                raise ResourceGraphCompileError("node local_vjp returned an incorrect parameter cotangent count")
            for version, input_value, cotangent in zip(
                input_versions, local_inputs, result.input_cotangents, strict=True
            ):
                if cotangent is None:
                    continue
                if cotangent.shape != input_value.shape:
                    raise ResourceGraphCompileError("node local_vjp input cotangent shape mismatch")
                cotangents[version] = cotangent if version not in cotangents else cotangents[version] + cotangent
            for parameter, cotangent in zip(parameters, result.parameter_cotangents, strict=True):
                if cotangent is None:
                    continue
                if cotangent.shape != parameter.shape:
                    raise ResourceGraphCompileError("node local_vjp parameter cotangent shape mismatch")
                parameter_id = id(parameter)
                parameter_cotangents[parameter_id] = (
                    cotangent
                    if parameter_id not in parameter_cotangents
                    else parameter_cotangents[parameter_id] + cotangent
                )

        initial_cotangents = {
            resource_id: cotangents.get(index)
            for index, resource_id in enumerate(self.resource_ids)
        }
        return StaticProgramGraphCreditResult(
            resource_values=tuple(values[:len(self.resource_ids)]),
            resource_cotangents=initial_cotangents,
            parameter_cotangents={
                parameter_names[parameter_id]: parameter_cotangents.get(parameter_id)
                for parameter_id in parameter_names
            },
            parameters={
                parameter_names[parameter_id]: parameter_values[parameter_id]
                for parameter_id in parameter_names
            },
            context_cotangents={
                connection_id: cotangents.get(len(self.resource_ids) + position)
                for position, connection_id in enumerate(self.context_connection_ids)
            },
        )

    def capture(self, *inputs: Tensor) -> CapturedResourceGraphExecutionPlan:
        """Capture this frozen program graph for a fixed CUDA input bucket."""

        return _capture_static_tensor_plan(self, *inputs)


class _StaticRouteBranch(nn.Module):
    def __init__(
        self,
        plan: StaticProgramGraphExecutionPlan | StaticDataflowProgramExecutionPlan,
        indices: tuple[int, ...],
        *,
        input_positions: tuple[int, ...],
        include_arrivals: bool = False,
        receipt_width: int = 0,
        receipt_offset: int = 0,
        receipt_scopes: tuple[Literal["batch", "sample"], ...] = (),
    ) -> None:
        super().__init__()
        self.plan = plan
        self.indices = indices
        self.input_positions = input_positions
        self.include_arrivals = include_arrivals
        self.receipt_width = receipt_width
        self.receipt_offset = receipt_offset
        self.receipt_scopes = receipt_scopes

    def forward(self, *values: Tensor) -> tuple[Tensor, ...]:
        result = self.plan(*(values[index] for index in self.input_positions))
        selected = tuple(result[index].clone() for index in self.indices)
        if self.include_arrivals:
            return (*selected, result[-1].to(dtype=values[0].dtype).clone())
        return selected

    def forward_with_nested_credit(
        self, *values: Tensor, selections: tuple[Tensor, ...] | None = None,
        active_rows: Tensor | None = None,
        sample: bool = False,
        cohort_receipt: bool = False,
    ) -> tuple[Tensor, ...]:
        operands = tuple(values[index] for index in self.input_positions)
        credit_anchor = sum(
            (value.reshape(-1)[0] * 0 for value in values
             if value.is_floating_point() and value.numel()),
            values[0].new_zeros(()),
        )
        if isinstance(self.plan, StaticDataflowProgramExecutionPlan):
            child_selections = None if selections is None else tuple(
                choice[0] if cohort_receipt and scope == "batch" else choice
                for choice, scope in zip(
                    selections[self.receipt_offset:self.receipt_offset + self.plan.route_receipt_width],
                    self.plan.route_receipt_scopes, strict=True,
                )
            )
            result, choices, log_probability = self.plan._forward_with_routes(
                *operands, sample_routes=sample,
                route_selections=child_selections,
                active_rows=active_rows,
            )
        else:
            result = self.plan(*operands)
            choices = ()
            log_probability = values[0].new_zeros(())
        log_probability = log_probability + credit_anchor
        row_count = active_rows.shape[0] if active_rows is not None else values[0].shape[0]
        if log_probability.ndim == 0 and (cohort_receipt or "sample" in self.receipt_scopes):
            live = active_rows.sum().clamp_min(1) if active_rows is not None else row_count
            log_probability = log_probability.expand(row_count) / live
        selected = tuple(result[index].clone() for index in self.indices)
        if self.include_arrivals:
            selected = (*selected, result[-1].to(dtype=values[0].dtype).clone())
        padded = tuple(
            (
                choices[index - self.receipt_offset].expand(row_count)
                if cohort_receipt and choices[index - self.receipt_offset].ndim == 0
                else choices[index - self.receipt_offset]
            ).to(dtype=values[0].dtype)
            if self.receipt_offset <= index < self.receipt_offset + len(choices)
            else torch.full(
                (row_count,) if cohort_receipt or scope == "sample" else (),
                -1, dtype=values[0].dtype, device=values[0].device,
            )
            for index, scope in enumerate(self.receipt_scopes)
        )
        return (*selected, log_probability, *padded)


class StaticProgramRouteStep(nn.Module):
    """Dispatch one declared relation without executing its unchosen peer."""

    def __init__(
        self,
        resource_ids: Sequence[str],
        route: ProgramRoute,
        candidates: Sequence[StaticProgramGraphExecutionPlan | StaticDataflowProgramExecutionPlan],
        batch_axes: Sequence[int],
        cohort_plans: Mapping[
            tuple[int, int],
            Sequence[StaticProgramGraphExecutionPlan | StaticDataflowProgramExecutionPlan],
        ] | None = None,
        share_cohort_plans: bool = False,
    ) -> None:
        super().__init__()
        if len(candidates) != len(route.candidates):
            raise ResourceGraphCompileError("route candidate plans do not match the route contract")
        self.resource_ids = tuple(resource_ids)
        self.route_id = route.route_id
        self.candidate_ids = route.candidates
        self.selection_scope = route.selection_scope
        self.execution_mode = route.execution_mode
        self.has_arrivals = isinstance(candidates[0], StaticDataflowProgramExecutionPlan)
        if any(self.has_arrivals != isinstance(plan, StaticDataflowProgramExecutionPlan)
               for plan in candidates[1:]):
            raise ResourceGraphCompileError("route candidates must agree on their arrival state ABI")
        self.context_connection_ids = tuple(dict.fromkeys(
            name for plan in candidates for name in plan.context_connection_ids
        ))
        self.credit_mask_connection_ids = tuple(dict.fromkeys(
            name for plan in candidates for name in plan.credit_mask_connection_ids
        ))
        self.batch_axes = (
            *batch_axes,
            *(0 for _ in self.context_connection_ids),
            *(0 for _ in self.credit_mask_connection_ids),
            *((0,) if self.has_arrivals else ()),
        )
        context_positions = {
            name: len(self.resource_ids) + index
            for index, name in enumerate(self.context_connection_ids)
        }
        mask_positions = {
            name: len(self.resource_ids) + len(context_positions) + index
            for index, name in enumerate(self.credit_mask_connection_ids)
        }

        def input_positions(
            plan: StaticProgramGraphExecutionPlan | StaticDataflowProgramExecutionPlan,
        ) -> tuple[int, ...]:
            return (
                *range(len(self.resource_ids)),
                *(context_positions[name] for name in plan.context_connection_ids),
                *(mask_positions[name] for name in plan.credit_mask_connection_ids),
                *((len(self.batch_axes) - 1,) if self.has_arrivals else ()),
            )
        self.score_index = self.resource_ids.index(route.score_resource_id)
        def published_indices(
            plan: StaticProgramGraphExecutionPlan | StaticDataflowProgramExecutionPlan,
        ) -> set[int]:
            if isinstance(plan, StaticDataflowProgramExecutionPlan):
                return {index for outputs in plan._output_indices for index in outputs}
            return {
                index for stage in plan._output_indices
                for outputs in stage for index in outputs
            }

        self._candidate_publications = tuple(published_indices(plan) for plan in candidates)
        self._written_indices = tuple(sorted(set.union(*self._candidate_publications)))
        self._branch_output_indices = (
            (*self._written_indices, len(self.batch_axes) - 1)
            if self.has_arrivals else self._written_indices
        )
        candidate_widths = tuple(
            plan.route_receipt_width if isinstance(plan, StaticDataflowProgramExecutionPlan) else 0
            for plan in candidates
        )
        candidate_offsets = tuple(sum(candidate_widths[:index]) for index in range(len(candidates)))
        self.nested_receipt_width = sum(candidate_widths)
        self.nested_receipt_ids = tuple(
            route_id for plan in candidates if isinstance(plan, StaticDataflowProgramExecutionPlan)
            for route_id in plan.route_receipt_ids
        )
        declared_nested_scopes = tuple(
            scope for plan in candidates if isinstance(plan, StaticDataflowProgramExecutionPlan)
            for scope in plan.route_receipt_scopes
        )
        self.nested_receipt_scopes = tuple(
            "sample" if self.selection_scope == "sample" else scope
            for scope in declared_nested_scopes
        )
        self.branches = nn.ModuleList(
            _StaticRouteBranch(
                plan, self._written_indices, input_positions=input_positions(plan),
                include_arrivals=self.has_arrivals,
                receipt_width=self.nested_receipt_width,
                receipt_offset=candidate_offsets[index],
                receipt_scopes=self.nested_receipt_scopes,
            ) for index, plan in enumerate(candidates)
        )
        self._cohort_branches = nn.ModuleDict()
        batch_size = max(
            (size for start, size in (cohort_plans or {}) if start == 0), default=0,
        )
        self._cohort_root_key = str(batch_size) if share_cohort_plans else f"0_{batch_size}"
        self._cohort_children: dict[str, tuple[int, int, str, str]] = {}
        for (start, size), cohort_candidates in (cohort_plans or {}).items():
            key = str(size) if share_cohort_plans else f"{start}_{size}"
            self._cohort_branches[key] = nn.ModuleList(
                _StaticRouteBranch(
                    plan, self._written_indices, input_positions=input_positions(plan),
                    include_arrivals=self.has_arrivals,
                    receipt_width=self.nested_receipt_width,
                    receipt_offset=candidate_offsets[index],
                    receipt_scopes=self.nested_receipt_scopes,
                ) for index, plan in enumerate(cohort_candidates)
            )
            if size > 1:
                middle = size // 2
                left_key = str(middle) if share_cohort_plans else f"{start}_{middle}"
                right_key = str(size - middle) if share_cohort_plans else f"{start + middle}_{size - middle}"
                self._cohort_children[key] = (middle, size - middle, left_key, right_key)

    def _dispatch_candidate(
        self, values: tuple[Tensor, ...], selection: Tensor,
        branches: nn.ModuleList, start: int = 0, end: int | None = None,
        nested_credit: bool = False,
        nested_selections: tuple[Tensor, ...] | None = None,
        active_rows: Tensor | None = None,
        nested_sample: bool = False,
        cohort_receipt: bool = False,
    ) -> tuple[Tensor, ...]:
        end = len(branches) if end is None else end
        if end - start == 1:
            return (branches[start].forward_with_nested_credit(
                *values, selections=nested_selections, active_rows=active_rows,
                sample=nested_sample, cohort_receipt=cohort_receipt,
            )
                    if nested_credit else branches[start](*values))
        middle = (start + end) // 2

        def first_half(*operands: Tensor) -> tuple[Tensor, ...]:
            return self._dispatch_candidate(
                operands, selection, branches, start, middle, nested_credit, nested_selections,
                active_rows,
                nested_sample,
                cohort_receipt,
            )

        def second_half(*operands: Tensor) -> tuple[Tensor, ...]:
            return self._dispatch_candidate(
                operands, selection, branches, middle, end, nested_credit, nested_selections,
                active_rows,
                nested_sample,
                cohort_receipt,
            )

        return torch.cond(selection < middle, first_half, second_half, values)

    def _dispatch_samples(
        self, values: tuple[Tensor, ...], choices: Tensor, key: str,
        nested_credit: bool = False,
        nested_selections: tuple[Tensor, ...] | None = None,
        active_rows: Tensor | None = None,
        nested_sample: bool = False,
    ) -> tuple[Tensor, ...]:
        branches = self._cohort_branches[key]
        if key not in self._cohort_children:
            return self._dispatch_candidate(
                values, choices[0], branches, nested_credit=nested_credit,
                nested_selections=nested_selections, active_rows=active_rows,
                nested_sample=nested_sample, cohort_receipt=nested_credit,
            )
        middle, right_size, left_key, right_key = self._cohort_children[key]
        cohort_size = middle + right_size

        def run_uniform(*operands: Tensor) -> tuple[Tensor, ...]:
            return self._dispatch_candidate(
                operands[:-1], operands[-1][0], branches,
                nested_credit=nested_credit, nested_selections=nested_selections,
                active_rows=active_rows, nested_sample=nested_sample,
                cohort_receipt=nested_credit,
            )

        def split(*operands: Tensor) -> tuple[Tensor, ...]:
            current = operands[:-1]
            selected = operands[-1]
            left_values = tuple(
                value.narrow(axis, 0, middle)
                if value.ndim > axis and value.shape[axis] == cohort_size else value
                for value, axis in zip(current, self.batch_axes, strict=True)
            )
            right_values = tuple(
                value.narrow(axis, middle, right_size)
                if value.ndim > axis and value.shape[axis] == cohort_size else value
                for value, axis in zip(current, self.batch_axes, strict=True)
            )
            left_nested = tuple(choice[:middle] for choice in nested_selections) if nested_selections else None
            right_nested = tuple(choice[middle:] for choice in nested_selections) if nested_selections else None
            left_active = active_rows[:middle] if active_rows is not None else None
            right_active = active_rows[middle:] if active_rows is not None else None
            left = self._dispatch_samples(
                left_values, selected[:middle], left_key, nested_credit,
                left_nested, left_active, nested_sample,
            )
            right = self._dispatch_samples(
                right_values, selected[middle:], right_key, nested_credit,
                right_nested, right_active, nested_sample,
            )
            output_axes = (
                *(self.batch_axes[index] for index in self._branch_output_indices),
                *((0,) * (self.nested_receipt_width + 1) if nested_credit else ()),
            )
            return tuple(
                torch.cat((left[slot], right[slot]), dim=axis)
                for slot, axis in enumerate(output_axes)
            )

        if key == self._cohort_root_key:
            values = tuple(value.clone() for value in values)
        operands = (*values, choices)
        return torch.cond((choices == choices[0]).all(), run_uniform, split, operands)

    def _dispatch_all(self, values: tuple[Tensor, ...], selection: Tensor) -> tuple[Tensor, ...]:
        results = tuple(branch(*values) for branch in self.branches)
        outputs = list(results[0])
        for candidate_index, result in enumerate(results[1:], start=1):
            for slot, index in enumerate(self._written_indices):
                selected = selection == candidate_index
                mask = selected.reshape(
                    tuple(selected.shape[0] if axis == self.batch_axes[index] else 1
                          for axis in range(result[slot].ndim))
                ) if selected.ndim else selected
                outputs[slot] = torch.where(mask, result[slot], outputs[slot])
        return tuple(outputs)

    def _forward_with_selection(
        self, *values: Tensor, sample: bool = False, selection: Tensor | None = None,
        active_rows: Tensor | None = None,
        nested_credit: bool = False,
        nested_selections: tuple[Tensor, ...] | None = None,
    ) -> tuple[tuple[Tensor, ...], Tensor, Tensor, Tensor, tuple[Tensor, ...]]:
        scores = values[self.score_index]
        if scores.ndim != 2 or scores.shape[1] != len(self.branches):
            raise ResourceGraphCompileError("ProgramRoute scores must have shape [batch, candidates]")
        if active_rows is not None and (
            active_rows.ndim != 1 or active_rows.dtype != torch.bool
            or active_rows.shape[0] != scores.shape[0]
        ):
            raise ResourceGraphCompileError("route active_rows must match the score batch")
        if nested_credit and active_rows is None:
            active_rows = torch.ones(scores.shape[0], dtype=torch.bool, device=scores.device)
        if active_rows is not None and self.selection_scope == "batch":
            weights = active_rows.to(dtype=scores.dtype, device=scores.device).unsqueeze(1)
            logits = (scores * weights).sum(dim=0) / weights.sum().clamp_min(1)
        else:
            logits = scores if self.selection_scope == "sample" else scores.mean(dim=0)
        log_probabilities = torch.log_softmax(logits, dim=-1)
        if selection is not None:
            selection = selection.to(device=logits.device, dtype=torch.long).reshape(logits.shape[:-1])
        elif sample:
            noise = -torch.log(-torch.log(torch.rand_like(logits).clamp(1e-7, 1 - 1e-7)))
            selection = (logits + noise).argmax(dim=-1)
        else:
            selection = logits.argmax(dim=-1)
        choose_first = selection == 0
        if nested_credit and self.execution_mode != "sparse":
            raise ResourceGraphCompileError("nested route credit requires sparse dispatch")
        if self.execution_mode == "all_candidates":
            if self.has_arrivals:
                raise ResourceGraphCompileError("arrival-bearing routes require sparse execution")
            outputs = self._dispatch_all(values, selection)
        elif self.selection_scope == "sample":
            outputs = self._dispatch_samples(
                values, selection, self._cohort_root_key,
                nested_credit=nested_credit, nested_selections=nested_selections,
                active_rows=active_rows, nested_sample=sample,
            )
        else:
            outputs = self._dispatch_candidate(
                values, selection, self.branches, nested_credit=nested_credit,
                nested_selections=nested_selections, active_rows=active_rows,
                nested_sample=sample,
            )
        if nested_credit:
            nested_term = outputs[-self.nested_receipt_width - 1]
            nested_choices = tuple(
                choice.to(dtype=torch.long) for choice in outputs[-self.nested_receipt_width:]
            ) if self.nested_receipt_width else ()
            outputs = outputs[:-(self.nested_receipt_width + 1)]
        else:
            nested_term = None
            nested_choices = ()
        result = list(values)
        branch_values = outputs[:-1] if self.has_arrivals else outputs
        for index, output in zip(self._written_indices, branch_values, strict=True):
            result[index] = output
        if self.has_arrivals:
            result[-1] = outputs[-1].to(dtype=torch.bool)
        selected_log_probability = log_probabilities.gather(-1, selection.unsqueeze(-1)).squeeze(-1)
        if nested_term is not None:
            if selected_log_probability.ndim == 0 and nested_term.ndim:
                assert active_rows is not None
                selected_log_probability = selected_log_probability / active_rows.sum().clamp_min(1)
            selected_log_probability = selected_log_probability + nested_term
        return (tuple(result), selected_log_probability, choose_first, selection, nested_choices)

    def forward_with_selection(
        self, *values: Tensor, sample: bool = False, selection: Tensor | None = None,
        active_rows: Tensor | None = None,
    ) -> tuple[tuple[Tensor, ...], Tensor, Tensor, Tensor]:
        result, term, first, choice, _ = self._forward_with_selection(
            *values, sample=sample, selection=selection, active_rows=active_rows,
        )
        return result, term, first, choice

    def forward_with_nested_credit(
        self, *values: Tensor, sample: bool = False,
        active_rows: Tensor | None = None,
        selection: Tensor | None = None,
        nested_selections: tuple[Tensor, ...] | None = None,
    ) -> tuple[tuple[Tensor, ...], Tensor, Tensor, tuple[Tensor, ...]]:
        result, term, _, choice, nested = self._forward_with_selection(
            *values, sample=sample, selection=selection, active_rows=active_rows,
            nested_credit=True, nested_selections=nested_selections,
        )
        return result, term, choice, nested

    def forward_with_receipt(
        self, *values: Tensor, sample: bool = False,
    ) -> tuple[tuple[Tensor, ...], Tensor]:
        result, log_probability, _, _ = self.forward_with_selection(*values, sample=sample)
        return result, log_probability

    def forward(self, *values: Tensor) -> tuple[Tensor, ...]:
        return self.forward_with_receipt(*values)[0]


class StaticRoutedProgramExecutionPlan(nn.Module):
    """Fixed-shape program with tensor-selected relation steps."""

    _component_reference: ClassVar[str] = "arti/static-routed-program-execution-plan@1"

    def __init__(self, resource_ids: Sequence[str], blocks: Sequence[nn.Module]) -> None:
        super().__init__()
        self.resource_ids = tuple(resource_ids)
        self.blocks = nn.ModuleList(blocks)
        self.context_connection_ids = tuple(dict.fromkeys(
            name for block in blocks for name in block.context_connection_ids
        ))
        self.credit_mask_connection_ids = tuple(dict.fromkeys(
            name for block in blocks for name in block.credit_mask_connection_ids
        ))
        resource_count = len(self.resource_ids)
        context_positions = {
            name: resource_count + index
            for index, name in enumerate(self.context_connection_ids)
        }
        mask_positions = {
            name: resource_count + len(context_positions) + index
            for index, name in enumerate(self.credit_mask_connection_ids)
        }
        self._block_input_indices = tuple(
            (
                *range(resource_count),
                *(context_positions[name] for name in block.context_connection_ids),
                *(mask_positions[name] for name in block.credit_mask_connection_ids),
            )
            for block in blocks
        )

    def forward(self, *values: Tensor) -> tuple[Tensor, ...]:
        expected = len(self.resource_ids) + len(self.context_connection_ids) + len(self.credit_mask_connection_ids)
        if len(values) != expected:
            raise ResourceGraphCompileError("routed program received an incorrect resource/context count")
        current = list(values)
        for block, indices in zip(self.blocks, self._block_input_indices, strict=True):
            result = block(*(current[index] for index in indices))
            current[:len(self.resource_ids)] = result[:len(self.resource_ids)]
        return tuple(current[:len(self.resource_ids)])

    def forward_with_route_credit(
        self, *values: Tensor,
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...], Tensor]:
        """Sample hard graph paths and expose choices and score-function terms."""

        expected = len(self.resource_ids) + len(self.context_connection_ids) + len(self.credit_mask_connection_ids)
        if len(values) != expected:
            raise ResourceGraphCompileError("routed program received an incorrect resource/context count")
        current = list(values)
        terms: list[Tensor] = []
        choices: list[Tensor] = []
        for block, indices in zip(self.blocks, self._block_input_indices, strict=True):
            block_inputs = tuple(current[index] for index in indices)
            if isinstance(block, StaticProgramRouteStep):
                result, log_probability, _, selection = block.forward_with_selection(
                    *block_inputs, sample=True,
                )
                terms.append(log_probability)
                choices.append(selection)
            else:
                result = block(*block_inputs)
            current[:len(self.resource_ids)] = result[:len(self.resource_ids)]
        if not terms:
            return tuple(current[:len(self.resource_ids)]), (), current[0].new_zeros(())
        joint_log_probability = terms[0]
        for term in terms[1:]:
            joint_log_probability = joint_log_probability + term
        return tuple(current[:len(self.resource_ids)]), tuple(choices), joint_log_probability


def _selected_route_mask(mask: Tensor, selection: Tensor) -> Tensor:
    return mask.index_select(0, selection.reshape(-1)).reshape(
        *selection.shape, mask.shape[1],
    )


class _StaticDataflowRouteStep(nn.Module):
    """Present one routed relation to the existing mixed dataflow tape."""

    def __init__(self, route: ProgramRoute, step: StaticProgramRouteStep) -> None:
        super().__init__()
        self.route_id = route.route_id
        self.candidates = route.candidates
        self.step = step
        self.output_indices = step._written_indices
        self.register_buffer(
            "publications",
            torch.tensor(
                [[index in candidate for index in self.output_indices]
                 for candidate in step._candidate_publications],
                dtype=torch.bool,
            ),
        )

    def forward(self, *values: Tensor) -> tuple[Tensor, ...]:
        result = self.step(*values)
        return tuple(result[index] for index in self.output_indices)

    def forward_with_selection(
        self, *values: Tensor, sample: bool = False, selection: Tensor | None = None,
        active_rows: Tensor | None = None,
    ) -> tuple[tuple[Tensor, ...], Tensor, Tensor, Tensor | None]:
        result, log_probability, _, selected = self.step.forward_with_selection(
            *values, sample=sample, selection=selection, active_rows=active_rows,
        )
        return (
            tuple(result[index] for index in self.output_indices),
            log_probability, selected,
            result[-1] if self.step.has_arrivals else None,
        )

    def forward_with_nested_credit(
        self, *values: Tensor, active_rows: Tensor | None = None,
        selection: Tensor | None = None,
        nested_selections: tuple[Tensor, ...] | None = None,
        sample: bool = False,
    ) -> tuple[tuple[Tensor, ...], Tensor, Tensor, Tensor | None, tuple[Tensor, ...]]:
        result, term, selected, nested = self.step.forward_with_nested_credit(
            *values, sample=sample,
            active_rows=active_rows,
            selection=selection, nested_selections=nested_selections,
        )
        return (
            tuple(result[index] for index in self.output_indices),
            term, selected, result[-1] if self.step.has_arrivals else None, nested,
        )

    def local_vjp(
        self,
        inputs: tuple[Tensor, ...],
        outputs: tuple[Tensor, ...],
        output_cotangents: tuple[Tensor, ...],
        parameters: tuple[nn.Parameter, ...],
        create_graph: bool,
    ) -> LocalVJPResult:
        if self.step.has_arrivals:
            result = autograd_local_vjp(
                self, inputs[:-1], outputs, output_cotangents, parameters, create_graph,
            )
            return LocalVJPResult((*result.input_cotangents, None), result.parameter_cotangents)
        return autograd_local_vjp(self, inputs, outputs, output_cotangents, parameters, create_graph)


class StaticProgramJoinExecutionPlan(nn.Module):
    """Tensor-only lowering of one Formula-backed all-new :class:`ProgramJoin`.

    The final input is a ``[batch, input_port]`` boolean arrival mask.  A row
    runs only when every named input port has arrived since its previous
    consumption.  The result appends the remaining arrival mask and a
    ``[batch, output_port]`` publication mask, so a surrounding static schedule
    can wire further joins without returning to Python epoch bookkeeping.
    """

    _component_reference: ClassVar[str] = "arti/static-program-join-execution-plan@1"

    def __init__(
        self,
        *,
        resource_ids: Sequence[str],
        templates: Sequence[TensorView],
        join: ProgramJoin,
        node: MultiPortProgramNode,
    ) -> None:
        super().__init__()
        self.resource_ids = tuple(resource_ids)
        self.templates = tuple(templates)
        if len(self.resource_ids) != len(self.templates) or len(set(self.resource_ids)) != len(self.resource_ids):
            raise ResourceGraphCompileError("static join resources must be unique and match templates")
        self.join_id = join.join_id
        self.node_id = node.node_id
        self.node = _lower_static_program_node(node)
        index = {resource_id: position for position, resource_id in enumerate(self.resource_ids)}
        self._input_indices = tuple(index[port.resource_id] for port in node.input_ports.values())
        self._output_indices = tuple(index[port.resource_id] for port in node.output_ports.values())
        self._input_batch_axes = tuple(self.templates[item].batch_axis for item in self._input_indices)
        self._output_batch_axes = tuple(self.templates[item].batch_axis for item in self._output_indices)

    def forward(self, *inputs: Tensor) -> tuple[Tensor, ...]:
        expected = len(self.resource_ids) + 1
        if len(inputs) != expected:
            raise ResourceGraphCompileError("static ProgramJoin received an incorrect resource/readiness count")
        values = list(inputs[:-1])
        arrivals = inputs[-1]
        if arrivals.ndim != 2 or arrivals.shape[1] != len(self._input_indices):
            raise ResourceGraphCompileError(
                "static ProgramJoin arrivals must have shape [batch, input_port]"
            )
        ready = arrivals.to(dtype=torch.bool).all(dim=1)
        for resource_index, batch_axis in zip(self._input_indices, self._input_batch_axes, strict=True):
            if values[resource_index].shape[batch_axis] != ready.shape[0]:
                raise ResourceGraphCompileError("static ProgramJoin batch dimensions do not match arrivals")
        outputs = self.node(*(values[index] for index in self._input_indices))
        for resource_index, batch_axis, candidate in zip(
            self._output_indices, self._output_batch_axes, outputs, strict=True
        ):
            previous = values[resource_index]
            if previous.shape != candidate.shape or previous.shape[batch_axis] != ready.shape[0]:
                raise ResourceGraphCompileError("static ProgramJoin output Tensor ABI changed")
            mask_shape = [1] * candidate.ndim
            mask_shape[batch_axis] = ready.shape[0]
            values[resource_index] = torch.where(ready.reshape(mask_shape), candidate, previous)
        remaining = torch.where(
            ready.unsqueeze(1), torch.zeros_like(arrivals, dtype=torch.bool), arrivals.to(dtype=torch.bool)
        )
        publications = ready.unsqueeze(1).expand(-1, len(self._output_indices))
        return (*values, remaining, publications)

    def credit_gradient(
        self,
        *inputs: Tensor,
        terminal_cotangents: Mapping[str, Tensor],
        create_graph: bool = True,
    ) -> StaticProgramJoinCreditResult:
        """Lower one all-new join with its actual fired rows as the receipt.

        Arrival is discrete scheduling evidence. It therefore has no
        cotangent: a non-fired row is an identity carry of the prior output
        resource, while a fired row contributes only through ``node.local_vjp``.
        """

        expected = len(self.resource_ids) + 1
        if len(inputs) != expected:
            raise ResourceGraphCompileError(
                "static ProgramJoin credit lowering received an incorrect resource/readiness count"
            )
        if not isinstance(terminal_cotangents, Mapping) or not terminal_cotangents:
            raise TypeError("terminal_cotangents must be a non-empty resource-id mapping")
        resource_index = {resource_id: index for index, resource_id in enumerate(self.resource_ids)}
        unknown = set(terminal_cotangents).difference(resource_index)
        if unknown:
            raise ResourceGraphCompileError(
                f"terminal cotangents name unknown resources: {sorted(unknown)!r}"
            )

        values = list(inputs[:-1])
        arrivals = inputs[-1]
        if arrivals.ndim != 2 or arrivals.shape[1] != len(self._input_indices):
            raise ResourceGraphCompileError(
                "static ProgramJoin arrivals must have shape [batch, input_port]"
            )
        ready = arrivals.to(dtype=torch.bool).all(dim=1)
        for resource_index_value, batch_axis in zip(
            self._input_indices, self._input_batch_axes, strict=True
        ):
            if values[resource_index_value].shape[batch_axis] != ready.shape[0]:
                raise ResourceGraphCompileError("static ProgramJoin batch dimensions do not match arrivals")

        versions = list(range(len(values)))
        next_version = len(values)
        snapshot_values = tuple(values)
        snapshot_versions = tuple(versions)
        local_inputs = tuple(
            value.detach().requires_grad_(value.is_floating_point() or value.is_complex())
            for value in (snapshot_values[index] for index in self._input_indices)
        )
        if any(not value.requires_grad for value in local_inputs):
            raise ResourceGraphCompileError(
                "static ProgramJoin credit lowering requires floating or complex node inputs"
            )
        outputs = tuple(self.node(*local_inputs))
        if len(outputs) != len(self._output_indices):
            raise ResourceGraphCompileError("static ProgramJoin returned an incorrect output count")

        output_versions: list[int] = []
        output_input_versions: list[int] = []
        for resource_index_value, batch_axis, candidate in zip(
            self._output_indices, self._output_batch_axes, outputs, strict=True
        ):
            previous = values[resource_index_value]
            if previous.shape != candidate.shape or previous.shape[batch_axis] != ready.shape[0]:
                raise ResourceGraphCompileError("static ProgramJoin output Tensor ABI changed")
            mask_shape = [1] * candidate.ndim
            mask_shape[batch_axis] = ready.shape[0]
            values[resource_index_value] = torch.where(ready.reshape(mask_shape), candidate, previous)
            output_input_versions.append(versions[resource_index_value])
            output_versions.append(next_version)
            versions[resource_index_value] = next_version
            next_version += 1

        remaining = torch.where(
            ready.unsqueeze(1), torch.zeros_like(arrivals, dtype=torch.bool), arrivals.to(dtype=torch.bool)
        )
        publications = ready.unsqueeze(1).expand(-1, len(self._output_indices))
        cotangents: dict[int, Tensor] = {}
        for resource_id, cotangent in terminal_cotangents.items():
            if not isinstance(cotangent, Tensor):
                raise TypeError("terminal cotangents must be Tensors")
            index = resource_index[resource_id]
            expected_value = values[index]
            if cotangent.shape != expected_value.shape:
                raise ResourceGraphCompileError(
                    f"terminal cotangent for {resource_id!r} must match its final resource shape"
                )
            cotangents[versions[index]] = cotangent.to(
                device=expected_value.device,
                dtype=expected_value.dtype,
            )

        node_output_cotangents: list[Tensor] = []
        for batch_axis, candidate, prior_version, output_version in zip(
            self._output_batch_axes,
            outputs,
            output_input_versions,
            output_versions,
            strict=True,
        ):
            output_cotangent = cotangents.get(output_version, torch.zeros_like(candidate))
            mask_shape = [1] * candidate.ndim
            mask_shape[batch_axis] = ready.shape[0]
            fired_mask = ready.reshape(mask_shape)
            node_output_cotangents.append(
                torch.where(fired_mask, output_cotangent, torch.zeros_like(output_cotangent))
            )
            carried = torch.where(fired_mask, torch.zeros_like(output_cotangent), output_cotangent)
            cotangents[prior_version] = (
                carried if prior_version not in cotangents else cotangents[prior_version] + carried
            )

        parameters = tuple(self.node.parameters())
        local_vjp = getattr(self.node, "local_vjp", None)
        if not callable(local_vjp):
            raise ResourceGraphCompileError(
                "static ProgramJoin credit lowering requires the lowered node to declare local_vjp"
            )
        result = local_vjp(
            local_inputs,
            outputs,
            tuple(node_output_cotangents),
            parameters,
            create_graph,
        )
        if not isinstance(result, LocalVJPResult):
            raise ResourceGraphCompileError("node local_vjp must return LocalVJPResult")
        if len(result.input_cotangents) != len(self._input_indices):
            raise ResourceGraphCompileError("node local_vjp returned an incorrect input cotangent count")
        if len(result.parameter_cotangents) != len(parameters):
            raise ResourceGraphCompileError("node local_vjp returned an incorrect parameter cotangent count")
        for resource_index_value, input_value, cotangent in zip(
            self._input_indices, local_inputs, result.input_cotangents, strict=True
        ):
            if cotangent is None:
                continue
            if cotangent.shape != input_value.shape:
                raise ResourceGraphCompileError("node local_vjp input cotangent shape mismatch")
            version = snapshot_versions[resource_index_value]
            cotangents[version] = cotangent if version not in cotangents else cotangents[version] + cotangent

        named_parameters = tuple(self.node.named_parameters())
        if tuple(parameter for _name, parameter in named_parameters) != parameters:
            raise ResourceGraphCompileError("static ProgramJoin parameter enumeration changed during lowering")
        parameter_names: dict[int, str] = {}
        parameter_values: dict[int, nn.Parameter] = {}
        parameter_cotangents: dict[int, Tensor] = {}
        for (name, parameter), cotangent in zip(
            named_parameters, result.parameter_cotangents, strict=True
        ):
            identity = id(parameter)
            parameter_names[identity] = f"{self.node_id}.{name.removeprefix('module.')}"
            parameter_values[identity] = parameter
            if cotangent is None:
                continue
            if cotangent.shape != parameter.shape:
                raise ResourceGraphCompileError("node local_vjp parameter cotangent shape mismatch")
            parameter_cotangents[identity] = cotangent

        return StaticProgramJoinCreditResult(
            resource_values=tuple(values),
            resource_cotangents={
                resource_id: cotangents.get(index)
                for index, resource_id in enumerate(self.resource_ids)
            },
            parameter_cotangents={
                parameter_names[identity]: parameter_cotangents.get(identity)
                for identity in parameter_names
            },
            parameters={
                parameter_names[identity]: parameter_values[identity]
                for identity in parameter_names
            },
            ready=ready,
            remaining_arrivals=remaining,
            publications=publications,
        )

    def capture(self, *inputs: Tensor) -> CapturedResourceGraphExecutionPlan:
        """Capture this frozen join plan for a fixed CUDA input bucket."""

        return _capture_static_tensor_plan(self, *inputs)


class StaticDataflowProgramExecutionPlan(nn.Module):
    """Lower a frozen Formula program containing ordinary nodes and all-new joins.

    ``arrivals`` is one ``[batch, port]`` Tensor spanning every Join input in
    the declared program.  Ordinary Formula nodes publish the ports fed by
    their resource outputs; a Join consumes only its own ready ports and then
    publishes its outputs.  The execution order is fixed, but readiness stays
    per-sample and tensor-resident.
    """

    _component_reference: ClassVar[str] = "arti/static-dataflow-program-execution-plan@1"

    def __init__(
        self,
        *,
        resource_ids: Sequence[str],
        templates: Sequence[TensorView],
        operations: Sequence[
            tuple[
                Literal["node", "join", "connection", "route"],
                MultiPortProgramNode | Connection | _StaticDataflowRouteStep,
                ProgramJoin | None,
            ]
        ],
        connection_plans: Mapping[str, ResourceGraphExecutionPlan] | None = None,
        join_scope: Sequence[tuple[ProgramJoin, MultiPortProgramNode]] | None = None,
    ) -> None:
        super().__init__()
        self.resource_ids = tuple(resource_ids)
        self.templates = tuple(templates)
        if len(self.resource_ids) != len(self.templates) or len(set(self.resource_ids)) != len(self.resource_ids):
            raise ResourceGraphCompileError("static dataflow resources must be unique and match templates")
        if not operations:
            raise ResourceGraphCompileError("static dataflow lowering requires at least one operation")
        index = {resource_id: position for position, resource_id in enumerate(self.resource_ids)}
        normalized = tuple(operations)
        if any(
            kind not in {"node", "join", "connection", "route"}
            or ((kind == "connection") != isinstance(node, Connection))
            or (kind == "route" and not isinstance(node, _StaticDataflowRouteStep))
            or (kind in {"node", "join"} and not isinstance(node, MultiPortProgramNode))
            for kind, node, _ in normalized
        ):
            raise TypeError("static dataflow operations must contain declared graph relations or multi-port nodes")
        if any((kind == "join") != isinstance(join, ProgramJoin) for kind, _node, join in normalized):
            raise ResourceGraphCompileError("static dataflow join declarations are invalid")
        plans = {} if connection_plans is None else dict(connection_plans)
        connection_steps = {
            node.connection_id: _StaticConnectionProgramStep(plans[node.connection_id])
            for kind, node, _join in normalized if kind == "connection"
        }
        self.context_connection_ids = tuple(dict.fromkeys(
            connection_id
            for kind, node, _join in normalized
            for connection_id in (
                node.step.context_connection_ids if kind == "route"
                else (node.connection_id,) if kind == "connection"
                and connection_steps[node.connection_id].plan.context_connection_ids else ()
            )
        ))
        self.credit_mask_connection_ids = tuple(dict.fromkeys(
            connection_id
            for kind, node, _join in normalized
            for connection_id in (
                node.step.credit_mask_connection_ids if kind == "route"
                else (node.connection_id,) if kind == "connection"
                and connection_steps[node.connection_id].plan.credit_mask_connection_ids else ()
            )
        ))
        context_index = {
            connection_id: len(index) + position
            for position, connection_id in enumerate(self.context_connection_ids)
        }
        mask_index = {
            connection_id: len(index) + len(context_index) + position
            for position, connection_id in enumerate(self.credit_mask_connection_ids)
        }
        declared_joins = (
            tuple((join, node) for kind, node, join in normalized if kind == "join" and join is not None)
            if join_scope is None else tuple(join_scope)
        )
        join_columns: dict[tuple[str, str], int] = {}
        join_resources: dict[tuple[str, str], str] = {}
        for join, node in declared_joins:
            for input_name, port in node.input_ports.items():
                key = (join.join_id, input_name)
                join_columns.setdefault(key, len(join_columns))
                join_resources[key] = port.resource_id
        self.arrival_width = len(join_columns)
        self.arrival_ports = tuple(
            (join_id, input_name, join_resources[(join_id, input_name)])
            for join_id, input_name in join_columns
        )
        self.nodes = nn.ModuleList(
            connection_steps[node.connection_id]
            if isinstance(node, Connection) else node if isinstance(node, _StaticDataflowRouteStep)
            else _lower_static_program_node(node)
            for _kind, node, _join in normalized
        )
        self._kinds = tuple(kind for kind, _node, _join in normalized)
        self.has_nested_routes = any(
            kind == "route" and any(
                isinstance(branch.plan, StaticDataflowProgramExecutionPlan)
                and "route" in branch.plan._kinds
                for branch in node.step.branches
            )
            for kind, node, _join in normalized
        )
        self.route_receipt_ids = tuple(
            route_id for kind, node, _join in normalized if kind == "route"
            for route_id in (node.route_id, *node.step.nested_receipt_ids)
        )
        self.route_receipt_scopes = tuple(
            scope for kind, node, _join in normalized if kind == "route"
            for scope in (node.step.selection_scope, *node.step.nested_receipt_scopes)
        )
        self.route_receipt_width = len(self.route_receipt_ids)
        self._node_ids = tuple(
            node.connection_id if isinstance(node, Connection)
            else node.route_id if isinstance(node, _StaticDataflowRouteStep) else node.node_id
            for _kind, node, _join in normalized
        )
        self._input_indices = tuple(
            (
                (*range(len(index)),
                 *((context_index[node.connection_id],) if node.connection_id in context_index else ()),
                 *((mask_index[node.connection_id],) if node.connection_id in mask_index else ()))
                if isinstance(node, Connection)
                else (
                    *range(len(index)),
                    *(context_index[name] for name in node.step.context_connection_ids),
                    *(mask_index[name] for name in node.step.credit_mask_connection_ids),
                ) if isinstance(node, _StaticDataflowRouteStep)
                else tuple(index[port.resource_id] for port in node.input_ports.values())
            )
            for _kind, node, _join in normalized
        )
        self._output_indices = tuple(
            (index[node.destination.resource_id],) if isinstance(node, Connection)
            else node.output_indices if isinstance(node, _StaticDataflowRouteStep)
            else tuple(index[port.resource_id] for port in node.output_ports.values())
            for _kind, node, _join in normalized
        )
        self._output_batch_axes = tuple(
            tuple(self.templates[resource_index].batch_axis for resource_index in output_indices)
            for output_indices in self._output_indices
        )
        self._input_batch_axes = tuple(
            tuple(self.templates[resource_index].batch_axis for resource_index in input_indices)
            if kind == "join" else ()
            for kind, input_indices in zip(self._kinds, self._input_indices, strict=True)
        )
        resource_arrival_columns: dict[int, set[int]] = {position: set() for position in range(len(self.resource_ids))}
        for join, node in declared_joins:
            for input_name, port in node.input_ports.items():
                resource_arrival_columns[index[port.resource_id]].add(join_columns[(join.join_id, input_name)])
        output_mask_names: list[str] = []
        consume_mask_names: list[str] = []
        join_index_names: list[str] = []
        route_mask_names: list[str | None] = []
        for operation_index, (kind, node, _join) in enumerate(normalized):
            output_mask = torch.zeros(self.arrival_width, dtype=torch.bool)
            for output_index in self._output_indices[operation_index]:
                for column in resource_arrival_columns[output_index]:
                    output_mask[column] = True
            output_name = f"_arrival_outputs_{operation_index}"
            self.register_buffer(output_name, output_mask)
            output_mask_names.append(output_name)
            consume = torch.zeros(self.arrival_width, dtype=torch.bool)
            columns = ()
            if kind == "join":
                assert _join is not None
                columns = tuple(join_columns[(_join.join_id, name)] for name in node.input_ports)
                for column in columns:
                    consume[column] = True
            consume_name = f"_arrival_consumes_{operation_index}"
            self.register_buffer(consume_name, consume)
            consume_mask_names.append(consume_name)
            index_name = f"_arrival_indices_{operation_index}"
            self.register_buffer(index_name, torch.tensor(columns, dtype=torch.long))
            join_index_names.append(index_name)
            if kind == "route":
                assert isinstance(node, _StaticDataflowRouteStep)
                branch_masks = []
                for publication in node.publications:
                    branch_mask = torch.zeros(self.arrival_width, dtype=torch.bool)
                    for output_slot, output_index in enumerate(self._output_indices[operation_index]):
                        if publication[output_slot]:
                            for column in resource_arrival_columns[output_index]:
                                branch_mask[column] = True
                    branch_masks.append(branch_mask)
                name = f"_route_arrivals_{operation_index}"
                self.register_buffer(name, torch.stack(branch_masks))
                route_mask_names.append(name)
            else:
                route_mask_names.append(None)
        self._output_mask_names = tuple(output_mask_names)
        self._consume_mask_names = tuple(consume_mask_names)
        self._join_index_names = tuple(join_index_names)
        self._route_mask_names = tuple(route_mask_names)

    def forward(self, *inputs: Tensor) -> tuple[Tensor, ...]:
        return self._forward_with_routes(*inputs)[0]

    def forward_with_active_rows(
        self, *inputs: Tensor, active_rows: Tensor,
    ) -> tuple[Tensor, ...]:
        return self._forward_with_routes(*inputs, active_rows=active_rows)[0]

    def forward_with_route_credit(
        self, *inputs: Tensor, active_rows: Tensor | None = None,
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...], Tensor]:
        """Sample hard routes on device and return the exact choices and joint log probability."""

        return self._forward_with_routes(*inputs, sample_routes=True, active_rows=active_rows)

    def forward_with_route_selections(
        self, *inputs: Tensor, route_selections: tuple[Tensor, ...],
    ) -> tuple[Tensor, ...]:
        """Execute explicitly selected hard paths for paired candidate evaluation."""

        return self._forward_with_routes(*inputs, route_selections=route_selections)[0]

    def _forward_with_routes(
        self, *inputs: Tensor, sample_routes: bool = False,
        route_selections: tuple[Tensor, ...] | None = None,
        active_rows: Tensor | None = None,
        capture_routes: bool = False,
    ) -> tuple[tuple[Tensor, ...], tuple[Tensor, ...], Tensor]:
        if route_selections is not None and len(route_selections) != self.route_receipt_width:
            raise ResourceGraphCompileError("route selections must match the executed route count")
        expected = len(self.resource_ids) + len(self.context_connection_ids) + len(self.credit_mask_connection_ids) + 1
        if len(inputs) != expected:
            raise ResourceGraphCompileError("static dataflow program received an incorrect resource/readiness count")
        values = list(inputs[:-1])
        arrivals = inputs[-1]
        if arrivals.ndim != 2 or arrivals.shape[1] != self.arrival_width:
            raise ResourceGraphCompileError(
                "static dataflow arrivals must have shape [batch, declared_join_input_ports]"
            )
        arrivals = arrivals.to(dtype=torch.bool)
        if active_rows is not None and (
            active_rows.ndim != 1 or active_rows.dtype != torch.bool
            or active_rows.shape[0] != arrivals.shape[0]
        ):
            raise ResourceGraphCompileError("dataflow active_rows must match the arrival batch")
        route_choices: list[Tensor] = []
        route_log_probabilities: list[Tensor] = []
        for operation_index, (node, kind, input_indices, output_indices, output_axes, index_name) in enumerate(
            zip(
                self.nodes,
                self._kinds,
                self._input_indices,
                self._output_indices,
                self._output_batch_axes,
                self._join_index_names,
                strict=True,
            )
        ):
            if kind == "join":
                ready = arrivals.index_select(1, getattr(self, index_name)).all(dim=1)
            else:
                ready = torch.ones(arrivals.shape[0], dtype=torch.bool, device=arrivals.device)
            if kind == "route":
                route_inputs = tuple(values[resource_index] for resource_index in input_indices)
                if node.step.has_arrivals:
                    route_inputs = (*route_inputs, arrivals)
                if (sample_routes or capture_routes or route_selections is not None) and node.step.nested_receipt_width:
                    offset = len(route_choices)
                    outputs, log_probability, selection, route_arrivals, nested_choices = (
                        node.forward_with_nested_credit(
                            *route_inputs, active_rows=active_rows,
                            sample=sample_routes,
                            selection=(route_selections[offset] if route_selections is not None else None),
                            nested_selections=(
                                route_selections[offset + 1:offset + 1 + node.step.nested_receipt_width]
                                if route_selections is not None else None
                            ),
                        )
                    )
                else:
                    outputs, log_probability, selection, route_arrivals = node.forward_with_selection(
                        *route_inputs,
                        sample=sample_routes,
                        active_rows=active_rows,
                        selection=(
                            route_selections[len(route_choices)] if route_selections is not None else None
                        ),
                    )
                    nested_choices = ()
                route_choices.append(selection)
                route_choices.extend(nested_choices)
                route_log_probabilities.append(log_probability)
            else:
                outputs = node(*(values[resource_index] for resource_index in input_indices))
                selection = None
                route_arrivals = None
            route_publications = (
                _selected_route_mask(node.publications, selection)
                if selection is not None else None
            )
            for output_slot, (resource_index, batch_axis, candidate) in enumerate(zip(
                output_indices, output_axes, outputs, strict=True
            )):
                previous = values[resource_index]
                if previous.shape != candidate.shape or previous.shape[batch_axis] != ready.shape[0]:
                    raise ResourceGraphCompileError("static dataflow node changed a declared output Tensor ABI")
                if kind in {"join", "route"}:
                    mask_shape = [1] * candidate.ndim
                    mask_shape[batch_axis] = ready.shape[0]
                    if route_publications is None:
                        published = ready
                    elif route_publications.ndim == 2:
                        published = route_publications[:, output_slot]
                    else:
                        published = route_publications[output_slot].expand_as(ready)
                    values[resource_index] = torch.where(
                        published.reshape(mask_shape), candidate, previous,
                    )
                else:
                    values[resource_index] = candidate
            output_mask = getattr(self, self._output_mask_names[operation_index])
            route_masks = self._route_mask_names[operation_index]
            if route_masks is not None:
                output_mask = _selected_route_mask(getattr(self, route_masks), selection)
            if route_arrivals is not None:
                arrivals = route_arrivals
                continue
            if kind == "join":
                consume_mask = getattr(self, self._consume_mask_names[operation_index])
                arrivals = torch.where(
                    ready.unsqueeze(1) & consume_mask.unsqueeze(0),
                    torch.zeros_like(arrivals),
                    arrivals,
                )
            arrivals = arrivals | (
                ready.unsqueeze(1) & (output_mask.unsqueeze(0) if output_mask.ndim == 1 else output_mask)
            )
        joint_log_probability = values[0].new_zeros(())
        has_sample_terms = any(term.ndim for term in route_log_probabilities)
        live_rows = (
            active_rows.sum().clamp_min(1) if active_rows is not None else arrivals.shape[0]
        )
        for term in route_log_probabilities:
            joint_log_probability = joint_log_probability + (
                term / live_rows if has_sample_terms and term.ndim == 0 else term
            )
        return (*values[:len(self.resource_ids)], arrivals), tuple(route_choices), joint_log_probability

    def credit_gradient(
        self,
        *inputs: Tensor,
        terminal_cotangents: Mapping[str, Tensor],
        create_graph: bool = True,
        route_selections: tuple[Tensor, ...] | None = None,
        active_rows: Tensor | None = None,
    ) -> StaticDataflowProgramCreditResult:
        """Compose local VJPs over node and all-new join receipts.

        ``arrivals`` stays a tensor-resident control receipt. A join contributes
        node credit only for rows that were ready at that operation; every
        non-fired row instead carries its cotangent to the prior resource
        version. Ordinary nodes use an all-true receipt.
        """

        expected = len(self.resource_ids) + len(self.context_connection_ids) + len(self.credit_mask_connection_ids) + 1
        if len(inputs) != expected:
            raise ResourceGraphCompileError(
                "static dataflow credit lowering received an incorrect resource/readiness count"
            )
        if not isinstance(terminal_cotangents, Mapping) or not terminal_cotangents:
            raise TypeError("terminal_cotangents must be a non-empty resource-id mapping")
        resource_index = {resource_id: index for index, resource_id in enumerate(self.resource_ids)}
        unknown = set(terminal_cotangents).difference(resource_index)
        if unknown:
            raise ResourceGraphCompileError(
                f"terminal cotangents name unknown resources: {sorted(unknown)!r}"
            )
        if route_selections is not None and len(route_selections) != self.route_receipt_width:
            raise ResourceGraphCompileError("route selections must match the executed route count")
        values = list(inputs[:-1])
        arrivals = inputs[-1]
        if arrivals.ndim != 2 or arrivals.shape[1] != self.arrival_width:
            raise ResourceGraphCompileError(
                "static dataflow arrivals must have shape [batch, declared_join_input_ports]"
            )
        arrivals = arrivals.to(dtype=torch.bool)
        batch_size = arrivals.shape[0]
        if active_rows is not None and (
            active_rows.ndim != 1 or active_rows.dtype != torch.bool
            or active_rows.shape[0] != batch_size
        ):
            raise ResourceGraphCompileError("dataflow active_rows must match the arrival batch")
        versions = list(range(len(values)))
        next_version = len(values)
        tapes: list[
            tuple[
                str,
                nn.Module,
                tuple[int, ...],
                tuple[Tensor, ...],
                tuple[Tensor, ...],
                tuple[nn.Parameter, ...],
            ]
        ] = []
        receipts: list[
            tuple[
                int,
                tuple[tuple[int, int, int, int, Tensor, Tensor], ...],
                Tensor | None,
            ]
        ] = []
        join_ready: list[Tensor | None] = []
        actual_route_selections: list[Tensor] = []
        parameter_names: dict[int, str] = {}
        parameter_values: dict[int, nn.Parameter] = {}

        for operation_index, (
            node,
            node_id,
            kind,
            input_indices,
            output_indices,
            output_axes,
            input_axes,
            index_name,
        ) in enumerate(
            zip(
                self.nodes,
                self._node_ids,
                self._kinds,
                self._input_indices,
                self._output_indices,
                self._output_batch_axes,
                self._input_batch_axes,
                self._join_index_names,
                strict=True,
            )
        ):
            if kind == "join":
                ready = arrivals.index_select(1, getattr(self, index_name)).all(dim=1)
                for resource_index_value, batch_axis in zip(input_indices, input_axes, strict=True):
                    if values[resource_index_value].shape[batch_axis] != batch_size:
                        raise ResourceGraphCompileError(
                            "static dataflow join batch dimensions do not match arrivals"
                        )
                join_ready.append(ready)
            else:
                ready = torch.ones(batch_size, dtype=torch.bool, device=arrivals.device)
                join_ready.append(None)
            snapshot_values = tuple(values)
            snapshot_versions = tuple(versions)
            if kind == "route":
                route_scores = snapshot_values[node.step.score_index]
                if active_rows is not None and node.step.selection_scope == "batch":
                    weights = active_rows.to(dtype=route_scores.dtype, device=route_scores.device).unsqueeze(1)
                    route_logits = (route_scores * weights).sum(dim=0) / weights.sum().clamp_min(1)
                else:
                    route_logits = route_scores if node.step.selection_scope == "sample" else route_scores.mean(dim=0)
                route_offset = len(actual_route_selections)
                selection = (
                    route_selections[route_offset]
                    if route_selections is not None
                    else route_logits.argmax(dim=-1)
                )
                selection = selection.to(
                    device=snapshot_values[node.step.score_index].device, dtype=torch.long,
                ).reshape((batch_size,) if node.step.selection_scope == "sample" else ())
                actual_route_selections.append(selection)
            else:
                selection = None
            route_publications = (
                _selected_route_mask(node.publications, selection)
                if selection is not None else None
            )
            local_inputs = tuple(
                value.detach().requires_grad_(value.is_floating_point() or value.is_complex())
                for value in (snapshot_values[index] for index in input_indices)
            )
            route_arrival_input = kind == "route" and node.step.has_arrivals
            if route_arrival_input:
                local_inputs = (*local_inputs, arrivals.detach())
            differentiable_inputs = local_inputs[:-1] if route_arrival_input else local_inputs
            if kind not in {"connection", "route"} and any(
                not value.requires_grad for value in differentiable_inputs
            ):
                raise ResourceGraphCompileError(
                    "static dataflow credit lowering requires floating or complex node inputs"
                )
            if kind == "route":
                if node.step.nested_receipt_width:
                    nested_selections = (
                        route_selections[
                            route_offset + 1:route_offset + 1 + node.step.nested_receipt_width
                        ] if route_selections is not None else None
                    )
                    outputs, _, _, route_arrivals, nested_choices = node.forward_with_nested_credit(
                        *local_inputs, active_rows=active_rows, selection=selection,
                        nested_selections=nested_selections,
                    )
                    actual_route_selections.extend(nested_choices)
                else:
                    outputs, _, _, route_arrivals = node.forward_with_selection(
                        *local_inputs, selection=selection,
                    )
            else:
                outputs = tuple(node(*local_inputs))
                route_arrivals = None
            if len(outputs) != len(output_indices):
                raise ResourceGraphCompileError(
                    "static dataflow node returned an incorrect output count"
                )
            parameters = tuple(node.parameters())
            for name, parameter in node.named_parameters():
                identity = id(parameter)
                parameter_names.setdefault(
                    identity,
                    f"{node_id}.{name.removeprefix('plan.connections.0.').removeprefix('module.')}",
                )
                parameter_values[identity] = parameter
            tape_index = len(tapes)
            tapes.append(
                (
                    node_id,
                    node,
                    (*tuple(snapshot_versions[index] for index in input_indices),
                     *((-1,) if route_arrival_input else ())),
                    local_inputs,
                    outputs,
                    parameters,
                )
            )
            commits: list[tuple[int, int, int, int, Tensor, Tensor]] = []
            for output_slot, (resource_index_value, batch_axis, candidate) in enumerate(
                zip(output_indices, output_axes, outputs, strict=True)
            ):
                previous = values[resource_index_value]
                if previous.shape != candidate.shape or previous.shape[batch_axis] != batch_size:
                    raise ResourceGraphCompileError("static dataflow node changed a declared output Tensor ABI")
                mask_shape = [1] * candidate.ndim
                mask_shape[batch_axis] = batch_size
                prior_version = versions[resource_index_value]
                output_version = next_version
                next_version += 1
                if route_publications is None:
                    published = ready
                elif route_publications.ndim == 2:
                    published = route_publications[:, output_slot]
                else:
                    published = route_publications[output_slot].expand_as(ready)
                values[resource_index_value] = torch.where(
                    published.reshape(mask_shape), candidate, previous,
                )
                versions[resource_index_value] = output_version
                commits.append(
                    (output_slot, prior_version, output_version, batch_axis, candidate, published)
                )
            output_mask = getattr(self, self._output_mask_names[operation_index])
            route_masks = self._route_mask_names[operation_index]
            if route_masks is not None:
                output_mask = _selected_route_mask(getattr(self, route_masks), selection)
            if route_arrivals is not None:
                arrivals = route_arrivals
                receipts.append((tape_index, tuple(commits), join_ready[-1]))
                continue
            if kind == "join":
                consume_mask = getattr(self, self._consume_mask_names[operation_index])
                arrivals = torch.where(
                    ready.unsqueeze(1) & consume_mask.unsqueeze(0),
                    torch.zeros_like(arrivals),
                    arrivals,
                )
            arrivals = arrivals | (
                ready.unsqueeze(1) & (output_mask.unsqueeze(0) if output_mask.ndim == 1 else output_mask)
            )
            receipts.append((tape_index, tuple(commits), join_ready[-1]))

        cotangents: dict[int, Tensor] = {}
        for resource_id, cotangent in terminal_cotangents.items():
            if not isinstance(cotangent, Tensor):
                raise TypeError("terminal cotangents must be Tensors")
            index = resource_index[resource_id]
            expected_value = values[index]
            if cotangent.shape != expected_value.shape:
                raise ResourceGraphCompileError(
                    f"terminal cotangent for {resource_id!r} must match its final resource shape"
                )
            cotangents[versions[index]] = cotangent.to(
                device=expected_value.device,
                dtype=expected_value.dtype,
            )

        node_output_cotangents: dict[int, list[Tensor]] = {
            tape_index: [torch.zeros_like(output) for output in tape[4]]
            for tape_index, tape in enumerate(tapes)
        }
        parameter_cotangents: dict[int, Tensor] = {}
        for tape_index, commits, _join_receipt in reversed(receipts):
            for output_slot, prior_version, output_version, batch_axis, candidate, ready in reversed(commits):
                output_cotangent = cotangents.get(output_version, torch.zeros_like(candidate))
                mask_shape = [1] * candidate.ndim
                mask_shape[batch_axis] = batch_size
                fired_mask = ready.reshape(mask_shape)
                node_output_cotangents[tape_index][output_slot] = torch.where(
                    fired_mask, output_cotangent, torch.zeros_like(output_cotangent)
                )
                carried = torch.where(fired_mask, torch.zeros_like(output_cotangent), output_cotangent)
                cotangents[prior_version] = (
                    carried if prior_version not in cotangents else cotangents[prior_version] + carried
                )
            _node_id, node, input_versions, local_inputs, outputs, parameters = tapes[tape_index]
            local_vjp = getattr(node, "local_vjp", None)
            if not callable(local_vjp):
                raise ResourceGraphCompileError(
                    "static dataflow credit lowering requires every lowered node to declare local_vjp"
                )
            result = local_vjp(
                local_inputs,
                outputs,
                tuple(node_output_cotangents[tape_index]),
                parameters,
                create_graph,
            )
            if not isinstance(result, LocalVJPResult):
                raise ResourceGraphCompileError("node local_vjp must return LocalVJPResult")
            if len(result.input_cotangents) != len(input_versions):
                raise ResourceGraphCompileError("node local_vjp returned an incorrect input cotangent count")
            if len(result.parameter_cotangents) != len(parameters):
                raise ResourceGraphCompileError("node local_vjp returned an incorrect parameter cotangent count")
            for version, input_value, cotangent in zip(
                input_versions, local_inputs, result.input_cotangents, strict=True
            ):
                if cotangent is None:
                    continue
                if cotangent.shape != input_value.shape:
                    raise ResourceGraphCompileError("node local_vjp input cotangent shape mismatch")
                cotangents[version] = cotangent if version not in cotangents else cotangents[version] + cotangent
            for parameter, cotangent in zip(parameters, result.parameter_cotangents, strict=True):
                if cotangent is None:
                    continue
                if cotangent.shape != parameter.shape:
                    raise ResourceGraphCompileError("node local_vjp parameter cotangent shape mismatch")
                identity = id(parameter)
                parameter_cotangents[identity] = (
                    cotangent
                    if identity not in parameter_cotangents
                    else parameter_cotangents[identity] + cotangent
                )

        return StaticDataflowProgramCreditResult(
            resource_values=tuple(values[:len(self.resource_ids)]),
            resource_cotangents={
                resource_id: cotangents.get(index)
                for index, resource_id in enumerate(self.resource_ids)
            },
            parameter_cotangents={
                parameter_names[identity]: parameter_cotangents.get(identity)
                for identity in parameter_names
            },
            parameters={
                parameter_names[identity]: parameter_values[identity]
                for identity in parameter_names
            },
            context_cotangents={
                connection_id: cotangents.get(len(self.resource_ids) + position)
                for position, connection_id in enumerate(self.context_connection_ids)
            },
            arrivals=arrivals,
            join_ready=tuple(join_ready),
            route_selections=tuple(actual_route_selections),
        )

    def capture(self, *inputs: Tensor) -> CapturedResourceGraphExecutionPlan:
        """Capture this frozen dataflow plan for a fixed CUDA input bucket."""

        return _capture_static_tensor_plan(self, *inputs)


class StaticDataflowLoopExecutionPlan(nn.Module):
    """Bounded per-row loop over one compiled connection/Formula/Join program."""

    _component_reference: ClassVar[str] = "arti/static-dataflow-loop-execution-plan@1"

    def __init__(self, *, body: StaticDataflowProgramExecutionPlan, loop: ProgramLoop) -> None:
        super().__init__()
        self.body = body
        self.resource_ids = body.resource_ids
        self.context_connection_ids = body.context_connection_ids
        self.credit_mask_connection_ids = body.credit_mask_connection_ids
        self.max_iterations = loop.max_iterations
        self.min_iterations = loop.min_iterations
        try:
            self._continue_index = self.resource_ids.index(loop.continue_resource_id)
        except ValueError as error:
            raise ResourceGraphCompileError("static dataflow loop continuation resource is missing") from error
        _loop_continue_mask(body.templates[self._continue_index])
        self._batch_axes = tuple(template.batch_axis for template in body.templates)

    def _split_inputs(self, inputs: tuple[Tensor, ...]) -> tuple[list[Tensor], tuple[Tensor, ...], Tensor]:
        resource_count = len(self.resource_ids)
        expected = resource_count + len(self.context_connection_ids) + len(self.credit_mask_connection_ids) + 1
        if len(inputs) != expected:
            raise ResourceGraphCompileError("static dataflow loop received an incorrect input count")
        return list(inputs[:resource_count]), inputs[resource_count:-1], inputs[-1]

    def _commit(
        self, values: list[Tensor], arrivals: Tensor, candidates: tuple[Tensor, ...], active: Tensor
    ) -> tuple[list[Tensor], Tensor]:
        next_values: list[Tensor] = []
        for previous, candidate, batch_axis in zip(values, candidates[:-1], self._batch_axes, strict=True):
            if previous.shape != candidate.shape or previous.shape[batch_axis] != active.shape[0]:
                raise ResourceGraphCompileError("static dataflow loop output Tensor ABI changed")
            mask_shape = [1] * candidate.ndim
            mask_shape[batch_axis] = active.shape[0]
            next_values.append(torch.where(active.reshape(mask_shape), candidate, previous))
        next_arrivals = torch.where(active.unsqueeze(1), candidates[-1], arrivals)
        return next_values, next_arrivals

    def forward(self, *inputs: Tensor) -> tuple[Tensor, ...]:
        if self.max_iterations is None:
            raise ResourceGraphCompileError(
                "open-horizon dataflow loop requires forward_until_done with a runtime host_step_limit"
            )
        if not torch.is_grad_enabled():
            return self.forward_until_done(*inputs)[:-2]
        return self.forward_bounded(*inputs)

    def forward_bounded(self, *inputs: Tensor) -> tuple[Tensor, ...]:
        """Fixed-horizon execution for training and CUDA Graph capture."""

        if self.max_iterations is None:
            raise ResourceGraphCompileError("open-horizon loop has no fixed training horizon")
        values, extras, arrivals = self._split_inputs(inputs)
        continuation = values[self._continue_index]
        if continuation.ndim != 1:
            raise ResourceGraphCompileError("static dataflow loop continuation must remain rank one")
        active = torch.ones_like(continuation, dtype=torch.bool)
        for iteration in range(self.max_iterations):
            candidates = self.body.forward_with_active_rows(
                *values, *extras, arrivals, active_rows=active,
            )
            values, arrivals = self._commit(values, arrivals, candidates, active)
            if iteration + 1 >= self.min_iterations:
                active = active & (values[self._continue_index] > 0.0)
        return (*values, arrivals)

    def forward_with_route_credit(
        self, *inputs: Tensor,
    ) -> tuple[tuple[Tensor, ...], tuple[tuple[Tensor, ...], ...], Tensor, tuple[Tensor, ...]]:
        """Sample hard routes and retain each active iteration's choice and likelihood."""

        if self.max_iterations is None:
            raise ResourceGraphCompileError("open-horizon route credit requires the eager execution receipts")
        values, extras, arrivals = self._split_inputs(inputs)
        continuation = values[self._continue_index]
        if continuation.ndim != 1:
            raise ResourceGraphCompileError("static dataflow loop continuation must remain rank one")
        active = torch.ones_like(continuation, dtype=torch.bool)
        route_selections: list[tuple[Tensor, ...]] = []
        iteration_active: list[Tensor] = []
        log_probabilities: list[Tensor] = []
        for iteration in range(self.max_iterations):
            iteration_active.append(active)
            candidates, choices, log_probability = self.body.forward_with_route_credit(
                *values, *extras, arrivals, active_rows=active,
            )
            route_selections.append(choices)
            receipt_active = active if log_probability.ndim else active.any()
            log_probabilities.append(log_probability * receipt_active.to(log_probability.dtype))
            values, arrivals = self._commit(values, arrivals, candidates, active)
            if iteration + 1 >= self.min_iterations:
                active = active & (values[self._continue_index] > 0.0)
        return (
            (*values, arrivals), tuple(route_selections),
            torch.stack(log_probabilities).sum(dim=0), tuple(iteration_active),
        )

    def forward_with_route_selections(
        self, *inputs: Tensor, route_selections: tuple[tuple[Tensor, ...], ...],
    ) -> tuple[Tensor, ...]:
        """Replay one fixed-horizon trajectory using explicit hard route choices."""

        if self.max_iterations is None:
            raise ResourceGraphCompileError("open-horizon route replay requires the eager execution receipts")
        if len(route_selections) != self.max_iterations:
            raise ResourceGraphCompileError("route selections must match the loop horizon")
        values, extras, arrivals = self._split_inputs(inputs)
        continuation = values[self._continue_index]
        if continuation.ndim != 1:
            raise ResourceGraphCompileError("static dataflow loop continuation must remain rank one")
        active = torch.ones_like(continuation, dtype=torch.bool)
        for iteration, choices in enumerate(route_selections):
            candidates = self.body.forward_with_route_selections(
                *values, *extras, arrivals, route_selections=choices,
            )
            values, arrivals = self._commit(values, arrivals, candidates, active)
            if iteration + 1 >= self.min_iterations:
                active = active & (values[self._continue_index] > 0.0)
        return (*values, arrivals)

    @torch.no_grad()
    def forward_until_done(
        self, *inputs: Tensor, host_step_limit: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        return self.forward_until_done_with_status(
            *inputs, host_step_limit=host_step_limit,
        )[:-1]

    @torch.no_grad()
    def forward_until_done_with_status(
        self, *inputs: Tensor, host_step_limit: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        """Execute only active iterations during inference.

        The final three tensors are physical calls, per-row logical steps,
        and remaining active rows. Remaining rows distinguish a runtime budget
        exit from endogenous completion, even when completion occurs at the
        last budgeted step. Training uses the bounded reverse tape instead.
        """

        values, extras, arrivals = self._split_inputs(inputs)
        continuation = values[self._continue_index]
        if continuation.ndim != 1:
            raise ResourceGraphCompileError("static dataflow loop continuation must remain rank one")
        if self.max_iterations is None:
            if host_step_limit is None or host_step_limit.ndim != 0 or host_step_limit.dtype != torch.int64:
                raise ResourceGraphCompileError(
                    "open-horizon dataflow loop requires a scalar int64 host_step_limit Tensor"
                )
            limit = host_step_limit
        elif host_step_limit is not None:
            raise ResourceGraphCompileError("host_step_limit applies only to open-horizon loops")
        else:
            limit = self.max_iterations
        step = torch.zeros((), dtype=torch.int64, device=continuation.device)
        active = torch.ones_like(continuation, dtype=torch.bool)
        row_steps = torch.zeros_like(continuation, dtype=torch.int64)

        def condition(current_step: Tensor, current_active: Tensor, current_steps: Tensor,
                      *current_values: Tensor) -> Tensor:
            del current_steps, current_values
            return (current_step < limit) & current_active.any()

        def body(current_step: Tensor, current_active: Tensor, current_steps: Tensor,
                 *current_values: Tensor) -> tuple[Tensor, ...]:
            candidates = self.body.forward_with_active_rows(
                *current_values[:-1], *extras, current_values[-1], active_rows=current_active,
            )
            next_values, next_arrivals = self._commit(
                list(current_values[:-1]), current_values[-1], candidates, current_active,
            )
            next_step = current_step + 1
            next_active = current_active & torch.where(
                next_step >= self.min_iterations,
                next_values[self._continue_index] > 0.0,
                torch.ones_like(current_active),
            )
            return (
                next_step, next_active, current_steps + current_active.to(torch.int64),
                *next_values, next_arrivals,
            )

        physical_steps, remaining_active, row_steps, *final_values = torch.while_loop(
            condition,
            body,
            (step, active, row_steps, *values, arrivals),
        )
        return (*final_values, physical_steps, row_steps, remaining_active)

    @torch.no_grad()
    def forward_until_done_with_route_receipt(
        self, *inputs: Tensor, host_step_limit: Tensor, receipt_capacity: int,
        sample_routes: bool = False,
        credit_mask_histories: tuple[Tensor, ...] | None = None,
        initial_active: Tensor | None = None,
        completed_steps: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        """Execute an open loop and record actual route choices on device.

        The trailing tensors are physical steps, row steps, remaining rows,
        per-step activity, whether routes were sampled, one choice history
        per route, and any supplied per-step Bernoulli masks. A capacity exit
        leaves remaining rows active just like a host-budget exit. Passing the
        returned state and remaining rows to the next call continues the same
        logical loop; ``completed_steps`` keeps its minimum depth global.
        """

        if self.body.has_nested_routes and "sample" in self.body.route_receipt_scopes:
            raise ResourceGraphCompileError(
                "sample-scoped nested routes require cohort route receipt lowering"
            )

        if self.max_iterations is not None:
            raise ResourceGraphCompileError("route receipt is for open-horizon loops")
        if host_step_limit.ndim != 0 or host_step_limit.dtype != torch.int64:
            raise ResourceGraphCompileError("host_step_limit must be a scalar int64 Tensor")
        if not isinstance(receipt_capacity, int) or receipt_capacity < 1:
            raise ResourceGraphCompileError("receipt_capacity must be a positive integer")
        values, extras, arrivals = self._split_inputs(inputs)
        continuation = values[self._continue_index]
        if continuation.ndim != 1:
            raise ResourceGraphCompileError("static dataflow loop continuation must remain rank one")
        if credit_mask_histories is not None:
            if len(credit_mask_histories) != len(self.credit_mask_connection_ids) or any(
                mask.dtype is not torch.bool or mask.shape[0] != receipt_capacity
                for mask in credit_mask_histories
            ):
                raise ResourceGraphCompileError("per-step credit masks must match the receipt capacity")
        batch_size = continuation.shape[0]
        if initial_active is not None and (
            initial_active.ndim != 1 or initial_active.dtype is not torch.bool
            or initial_active.shape != continuation.shape or initial_active.device != continuation.device
        ):
            raise ResourceGraphCompileError("initial_active must match the continuation batch")
        if completed_steps is not None and (
            completed_steps.ndim != 0 or completed_steps.dtype != torch.int64
            or completed_steps.device != continuation.device
        ):
            raise ResourceGraphCompileError("completed_steps must be a scalar int64 Tensor on the loop device")
        prior_steps = (
            torch.zeros((), dtype=torch.int64, device=continuation.device)
            if completed_steps is None else completed_steps
        )
        histories = tuple(
            torch.full(
                (receipt_capacity, batch_size) if scope == "sample"
                else (receipt_capacity,),
                -1, dtype=torch.long, device=continuation.device,
            )
            for scope in self.body.route_receipt_scopes
        )
        activity = torch.zeros(
            (receipt_capacity, batch_size), dtype=torch.bool, device=continuation.device,
        )
        step = torch.zeros((), dtype=torch.int64, device=continuation.device)
        active = (
            torch.ones_like(continuation, dtype=torch.bool)
            if initial_active is None else initial_active
        )
        row_steps = torch.zeros_like(continuation, dtype=torch.int64)
        limit = torch.minimum(host_step_limit, torch.as_tensor(receipt_capacity, device=step.device))
        resource_count = len(values)
        context_count = len(self.context_connection_ids)

        def condition(current_step: Tensor, current_active: Tensor, current_steps: Tensor,
                      current_activity: Tensor, *carried: Tensor) -> Tensor:
            del current_steps, current_activity, carried
            return (current_step < limit) & current_active.any()

        def body(current_step: Tensor, current_active: Tensor, current_steps: Tensor,
                 current_activity: Tensor, *carried: Tensor) -> tuple[Tensor, ...]:
            current_values = carried[:resource_count]
            current_arrivals = carried[resource_count]
            current_histories = carried[resource_count + 1:]
            step_extras = extras
            if credit_mask_histories is not None:
                step_extras = (
                    *extras[:context_count],
                    *(
                        mask.index_select(0, current_step.reshape(1)).squeeze(0)
                        for mask in credit_mask_histories
                    ),
                )
            candidates, choices, _ = self.body._forward_with_routes(
                *current_values, *step_extras, current_arrivals,
                sample_routes=sample_routes, active_rows=current_active, capture_routes=True,
            )
            next_values, next_arrivals = self._commit(
                list(current_values), current_arrivals, candidates, current_active,
            )
            index = current_step.reshape(1)
            next_activity = current_activity.index_copy(0, index, current_active.unsqueeze(0))
            next_histories = tuple(
                history.index_copy(0, index, choice.unsqueeze(0))
                for history, choice in zip(current_histories, choices, strict=True)
            )
            next_step = current_step + 1
            next_active = current_active & torch.where(
                prior_steps + next_step >= self.min_iterations,
                next_values[self._continue_index] > 0.0,
                torch.ones_like(current_active),
            )
            return (
                next_step, next_active, current_steps + current_active.to(torch.int64),
                next_activity, *next_values, next_arrivals, *next_histories,
            )

        physical_steps, remaining_active, row_steps, activity, *carried = torch.while_loop(
            condition, body, (step, active, row_steps, activity, *values, arrivals, *histories),
        )
        return (
            *carried[:resource_count + 1], physical_steps, row_steps, remaining_active,
            activity, torch.as_tensor(sample_routes, device=step.device),
            *carried[resource_count + 1:],
            *((credit_mask_histories or ())),
        )

    def replay_tensor_receipt(
        self, *inputs: Tensor, tensor_receipt: tuple[Tensor, ...],
        row_advantage: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        """Differentiate the recorded hard path with a fixed-shape graph.

        The original forward controls which steps and branches happened. This
        replay uses that receipt for parameter gradients and route likelihood;
        it does not choose a second path or change the committed state. Eager
        replay stops at the recorded horizon. Compiled replay keeps the fixed
        receipt capacity so its graph shape does not depend on a tensor value.
        """

        if self.max_iterations is not None:
            raise ResourceGraphCompileError("tensor receipt replay requires an open-horizon loop")
        values, extras, arrivals = self._split_inputs(inputs)
        route_count = self.body.route_receipt_width
        if len(tensor_receipt) not in (
            route_count + 3, route_count + len(self.credit_mask_connection_ids) + 3,
        ):
            raise ResourceGraphCompileError("tensor receipt has an incorrect route or mask count")
        _, activity, _, *histories = tensor_receipt
        if row_advantage is not None and row_advantage.shape != activity.shape[1:]:
            raise ResourceGraphCompileError("route advantage must match receipt rows")
        route_histories = histories[:route_count]
        mask_histories = histories[route_count:]
        context_count = len(self.context_connection_ids)
        log_probability = values[0].new_zeros(())
        horizon = activity.shape[0] if torch.compiler.is_compiling() else int(tensor_receipt[0].item())
        if horizon < 0 or horizon > activity.shape[0]:
            raise ResourceGraphCompileError("tensor receipt horizon exceeds its capacity")
        for iteration in range(horizon):
            active = activity[iteration]
            choices = tuple(
                torch.where(active.any(), history[iteration], torch.zeros_like(history[iteration]))
                for history in route_histories
            )
            step_extras = extras
            if mask_histories:
                step_extras = (
                    *extras[:context_count],
                    *(mask[iteration] for mask in mask_histories),
                )
            candidates, _, step_log_probability = self.body._forward_with_routes(
                *values, *step_extras, arrivals,
                route_selections=choices,
                active_rows=active,
            )
            receipt_active = active if step_log_probability.ndim else active.any()
            if row_advantage is None:
                log_probability = log_probability + (
                    step_log_probability * receipt_active.to(step_log_probability.dtype)
                ).sum()
            elif step_log_probability.ndim:
                log_probability = log_probability + (
                    step_log_probability * active.to(step_log_probability.dtype) * row_advantage
                ).sum()
            else:
                log_probability = log_probability + (
                    step_log_probability * (row_advantage * active).sum()
                    / active.sum().clamp_min(1)
                )
            values, arrivals = self._commit(values, arrivals, candidates, active)
        return (*values, arrivals, log_probability)

    def credit_gradient(
        self,
        *inputs: Tensor,
        terminal_cotangents: Mapping[str, Tensor],
        create_graph: bool = True,
        route_selections: tuple[tuple[Tensor, ...], ...] | None = None,
        execution: ProgramLoopExecution | None = None,
        tensor_receipt: tuple[Tensor, ...] | None = None,
    ) -> StaticDataflowLoopCreditResult:
        """Reverse the executed dataflow, including a variable-length route receipt.

        An open-horizon loop replays only the recorded hard choices and active
        rows through the existing mixed Connection/Program/Join local VJP tape.
        Route-controller credit remains the separate score-function objective
        on ``ProgramLoopExecution``; this method does not differentiate argmax.
        """
        if self.max_iterations is None:
            if (execution is None) == (tensor_receipt is None):
                raise ResourceGraphCompileError(
                    "open-horizon reverse tape requires one actual execution receipt"
                )
            if route_selections is not None:
                raise ResourceGraphCompileError("open-horizon route selections come from the execution receipt")
            recorded_selections: list[tuple[Tensor, ...]] = []
            if execution is not None:
                def route_identity(route: ProgramRouteExecution) -> tuple[object, ...]:
                    return (
                        route.route_id, route.candidate_id, route.scores,
                        route.sampled, route.iteration,
                    )

                horizon = execution.actual_iterations
                for iteration in range(horizon):
                    reached = tuple(route for route in execution.routes if route.iteration == iteration)
                    dispatched = tuple(
                        dispatch.route for dispatch in execution.dispatches
                        if dispatch.iteration == iteration and dispatch.route is not None
                    )
                    if tuple(map(route_identity, dispatched)) != tuple(map(route_identity, reached)):
                        raise ResourceGraphCompileError("execution receipt is missing a dispatched route")
                    cursor = 0
                    unvisited = torch.tensor(-1, dtype=torch.long, device=inputs[0].device)

                    def follow_dispatches(plan: StaticDataflowProgramExecutionPlan) -> tuple[Tensor, ...]:
                        nonlocal cursor
                        choices: list[Tensor] = []
                        for kind, node in zip(plan._kinds, plan.nodes, strict=True):
                            if kind != "route":
                                continue
                            if cursor >= len(dispatched) or dispatched[cursor].route_id != node.route_id:
                                raise ResourceGraphCompileError(
                                    "execution receipt is missing a dispatched route"
                                )
                            route = dispatched[cursor]
                            cursor += 1
                            try:
                                selected = node.step.candidate_ids.index(route.candidate_id)
                            except ValueError as error:
                                raise ResourceGraphCompileError(
                                    "execution receipt names a route candidate outside the compiled graph"
                                ) from error
                            choices.append(torch.tensor(selected, dtype=torch.long, device=inputs[0].device))
                            for candidate_index, branch in enumerate(node.step.branches):
                                if not isinstance(branch.plan, StaticDataflowProgramExecutionPlan):
                                    continue
                                if candidate_index == selected:
                                    choices.extend(follow_dispatches(branch.plan))
                                else:
                                    choices.extend((unvisited,) * branch.plan.route_receipt_width)
                        return tuple(choices)

                    choices = follow_dispatches(self.body)
                    if cursor != len(dispatched) or len(choices) != self.body.route_receipt_width:
                        raise ResourceGraphCompileError("execution receipt route path does not match compiled graph")
                    recorded_selections.append(choices)
            else:
                if tensor_receipt is None or len(tensor_receipt) not in (
                    self.body.route_receipt_width + 3,
                    self.body.route_receipt_width + len(self.credit_mask_connection_ids) + 3,
                ):
                    raise ResourceGraphCompileError("tensor receipt has an incorrect route count")
                physical_steps, activity_history, sampled_flag, *histories = tensor_receipt
                route_histories = histories[:self.body.route_receipt_width]
                mask_histories = histories[self.body.route_receipt_width:]
                if sampled_flag.ndim != 0 or sampled_flag.dtype is not torch.bool:
                    raise ResourceGraphCompileError("tensor receipt sampled flag must be scalar boolean")
                horizon = int(physical_steps.item())
                if activity_history.ndim != 2 or activity_history.shape[1] != inputs[self._continue_index].shape[0]:
                    raise ResourceGraphCompileError("tensor receipt activity has an incorrect batch shape")
                if horizon < 0 or horizon > activity_history.shape[0]:
                    raise ResourceGraphCompileError("tensor receipt horizon exceeds its capacity")
                if any(history.shape[0] != activity_history.shape[0] for history in route_histories):
                    raise ResourceGraphCompileError("tensor receipt route capacity does not match activity")
                if any(mask.dtype is not torch.bool or mask.shape[0] != activity_history.shape[0]
                       for mask in mask_histories):
                    raise ResourceGraphCompileError("tensor receipt credit mask capacity does not match activity")
                if any(
                    history.shape[1:] != ((activity_history.shape[1],) if scope == "sample" else ())
                    for history, scope in zip(
                        route_histories, self.body.route_receipt_scopes, strict=True,
                    )
                ):
                    raise ResourceGraphCompileError("tensor receipt route scope does not match its declared route")
                recorded_selections = [
                    tuple(history[iteration] for history in route_histories)
                    for iteration in range(horizon)
                ]
            route_selections = tuple(recorded_selections)
        else:
            if execution is not None or tensor_receipt is not None:
                raise ResourceGraphCompileError("bounded reverse tape uses its declared horizon")
            horizon = self.max_iterations
        if not isinstance(terminal_cotangents, Mapping) or not terminal_cotangents:
            raise TypeError("terminal_cotangents must be a non-empty resource-id mapping")
        unknown = set(terminal_cotangents).difference(self.resource_ids)
        if unknown:
            raise ResourceGraphCompileError(f"unknown terminal resources: {sorted(unknown)!r}")
        if route_selections is not None and len(route_selections) != horizon:
            raise ResourceGraphCompileError("route selections must match the loop horizon")
        recorded_masks: list[dict[str, Tensor]] | None = None
        if execution is not None and self.credit_mask_connection_ids:
            required = set(self.credit_mask_connection_ids)
            recorded_masks = [{} for _ in range(horizon)]
            executed_masks = [set() for _ in range(horizon)]
            dispatch_count = 0
            for dispatch in execution.dispatches:
                if dispatch.iteration < 0 or dispatch.iteration >= horizon:
                    raise ResourceGraphCompileError("execution has a dispatch outside its recorded horizon")
                for member in dispatch.members:
                    if not isinstance(member, ConnectionExecution) or member.connection_id not in required:
                        continue
                    dispatch_count += 1
                    masks = recorded_masks[dispatch.iteration]
                    if member.credit_mask is None:
                        raise ResourceGraphCompileError(
                            "open-horizon Bernoulli credit requires a recorded mask per executed connection"
                        )
                    previous = masks.get(member.connection_id)
                    if previous is not None and not torch.equal(previous, member.credit_mask):
                        raise ResourceGraphCompileError(
                            "reused Bernoulli connections must share one mask within a loop iteration"
                        )
                    masks[member.connection_id] = member.credit_mask
                    executed_masks[dispatch.iteration].add(member.connection_id)
            execution_count = sum(
                member.connection_id in required for member in execution.connections
            )
            if dispatch_count != execution_count or any(
                set(masks) != executed
                for masks, executed in zip(recorded_masks, executed_masks, strict=True)
            ):
                raise ResourceGraphCompileError("execution is missing Bernoulli connection dispatch receipts")
        elif tensor_receipt is not None and mask_histories:
            recorded_masks = [
                {
                    connection_id: mask_history[iteration]
                    for connection_id, mask_history in zip(
                        self.credit_mask_connection_ids, mask_histories, strict=True
                    )
                }
                for iteration in range(horizon)
            ]
        values, extras, arrivals = self._split_inputs(inputs)
        continuation = values[self._continue_index]
        if continuation.ndim != 1:
            raise ResourceGraphCompileError("static dataflow loop continuation must remain rank one")
        active = torch.ones_like(continuation, dtype=torch.bool)
        snapshots: list[
            tuple[tuple[Tensor, ...], Tensor, Tensor, tuple[Tensor, ...] | None, tuple[Tensor, ...]]
        ] = []
        route_log_terms: list[Tensor] = []
        sampled_routes = bool(tensor_receipt[2].item()) if tensor_receipt is not None else False
        for iteration in range(horizon):
            if execution is not None:
                active = execution.active_masks[iteration].to(device=continuation.device)
            elif tensor_receipt is not None:
                active = tensor_receipt[1][iteration].to(device=continuation.device)
            choices = route_selections[iteration] if route_selections is not None else None
            step_extras = extras
            if recorded_masks is not None:
                step_extras_list = list(extras)
                for index, connection_id in enumerate(self.credit_mask_connection_ids):
                    if connection_id in recorded_masks[iteration]:
                        step_extras_list[len(self.context_connection_ids) + index] = recorded_masks[iteration][connection_id]
                step_extras = tuple(step_extras_list)
            snapshots.append((tuple(values), arrivals, active, choices, step_extras))
            if choices is not None:
                candidates, _, log_probability = self.body._forward_with_routes(
                    *values, *step_extras, arrivals,
                    route_selections=choices, active_rows=active,
                )
                if sampled_routes:
                    receipt_active = active if log_probability.ndim else active.any()
                    route_log_terms.append(log_probability * receipt_active.to(log_probability.dtype))
            else:
                candidates = self.body.forward_with_active_rows(
                    *values, *step_extras, arrivals, active_rows=active,
                )
            values, arrivals = self._commit(values, arrivals, candidates, active)
            if iteration + 1 >= self.min_iterations:
                active = active & (values[self._continue_index] > 0.0)

        cotangents = [
            terminal_cotangents.get(resource_id, torch.zeros_like(value))
            for resource_id, value in zip(self.resource_ids, values, strict=True)
        ]
        parameter_cotangents: dict[str, Tensor | None] = {}
        parameters: dict[str, nn.Parameter] = {}
        context_cotangents: dict[str, Tensor | None] = {}
        join_ready: list[tuple[Tensor | None, ...]] = []
        actual_route_selections: list[tuple[Tensor, ...]] = []
        for snapshot_values, snapshot_arrivals, activity, choices, step_extras in reversed(snapshots):
            candidate_cotangents: dict[str, Tensor] = {}
            carried: list[Tensor] = []
            for resource_id, cotangent, batch_axis in zip(
                self.resource_ids, cotangents, self._batch_axes, strict=True
            ):
                mask_shape = [1] * cotangent.ndim
                mask_shape[batch_axis] = activity.shape[0]
                mask = activity.reshape(mask_shape)
                candidate_cotangents[resource_id] = torch.where(mask, cotangent, torch.zeros_like(cotangent))
                carried.append(torch.where(mask, torch.zeros_like(cotangent), cotangent))
            result = self.body.credit_gradient(
                *snapshot_values, *step_extras, snapshot_arrivals,
                terminal_cotangents=candidate_cotangents,
                create_graph=create_graph,
                route_selections=choices,
                active_rows=activity,
            )
            actual_route_selections.append(result.route_selections)
            cotangents = [
                carry if result.resource_cotangents[resource_id] is None
                else carry + result.resource_cotangents[resource_id]
                for resource_id, carry in zip(self.resource_ids, carried, strict=True)
            ]
            parameters.update(result.parameters)
            for name, cotangent in result.parameter_cotangents.items():
                previous = parameter_cotangents.get(name)
                parameter_cotangents[name] = (
                    cotangent if previous is None else previous if cotangent is None else previous + cotangent
                )
            for name, cotangent in result.context_cotangents.items():
                previous = context_cotangents.get(name)
                context_cotangents[name] = (
                    cotangent if previous is None else previous if cotangent is None else previous + cotangent
                )
            join_ready.append(result.join_ready)
        return StaticDataflowLoopCreditResult(
            resource_values=tuple(values),
            resource_cotangents=dict(zip(self.resource_ids, cotangents, strict=True)),
            parameter_cotangents=parameter_cotangents,
            parameters=parameters,
            context_cotangents=context_cotangents,
            arrivals=arrivals,
            iteration_active=tuple(snapshot[2] for snapshot in snapshots),
            join_ready=tuple(reversed(join_ready)),
            route_selections=tuple(reversed(actual_route_selections)),
            route_log_probability=(
                values[0].new_zeros(()) + sum(term.sum() for term in route_log_terms)
                if sampled_routes else None
            ),
            route_log_terms=tuple(route_log_terms),
            route_ids=self.body.route_receipt_ids,
        )

    def capture(self, *inputs: Tensor) -> CapturedResourceGraphExecutionPlan:
        return _capture_static_tensor_plan(self, *inputs, capture_forward=self.forward_bounded)


class StaticProgramLoopExecutionPlan(nn.Module):
    """Tensor-only lowering for a bounded program loop with static frontiers.

    Each item in ``stages`` observes the current resource snapshot once.  A
    stage may contain several independent nodes; their output publications are
    committed only after every peer has run.  This preserves
    :class:`ProgramStage` semantics inside a bounded conditional loop instead
    of silently serialising a declared parallel frontier.
    """

    _component_reference: ClassVar[str] = "arti/static-program-loop-execution-plan@1"

    def __init__(
        self,
        *,
        resource_ids: Sequence[str],
        templates: Sequence[TensorView],
        stages: Sequence[Sequence[MultiPortProgramNode]],
        loop: ProgramLoop,
    ) -> None:
        super().__init__()
        self.resource_ids = tuple(resource_ids)
        self.templates = tuple(templates)
        self.max_iterations = loop.max_iterations
        self.min_iterations = loop.min_iterations
        if len(self.resource_ids) != len(self.templates):
            raise ResourceGraphCompileError("static loop templates must match resources")
        index = {resource_id: position for position, resource_id in enumerate(self.resource_ids)}
        try:
            self._continue_index = index[loop.continue_resource_id]
        except KeyError as error:
            raise ResourceGraphCompileError("static loop continuation resource is missing") from error
        continue_template = self.templates[self._continue_index]
        _loop_continue_mask(continue_template)
        if not stages or any(not stage for stage in stages):
            raise ResourceGraphCompileError("static loop requires non-empty program stages")
        self.stages = nn.ModuleList(
            nn.ModuleList(_lower_static_program_node(node) for node in stage)
            for stage in stages
        )
        self._node_ids = tuple(tuple(node.node_id for node in stage) for stage in stages)
        self._input_indices = tuple(
            tuple(
                tuple(index[port.resource_id] for port in node.input_ports.values())
                for node in stage
            )
            for stage in stages
        )
        self._output_indices = tuple(
            tuple(
                tuple(index[port.resource_id] for port in node.output_ports.values())
                for node in stage
            )
            for stage in stages
        )
        written_indices = {
            index for stage in self._output_indices for node in stage for index in node
        }
        self._unwritten_indices = tuple(
            index for index in range(len(self.resource_ids)) if index not in written_indices
        )
        self._batch_axes = tuple(template.batch_axis for template in self.templates)

    def forward(self, *inputs: Tensor) -> tuple[Tensor, ...]:
        if len(inputs) != len(self.resource_ids):
            raise ResourceGraphCompileError(
                "static ProgramLoop received an incorrect resource value count"
            )
        if not torch.is_grad_enabled():
            return self.forward_until_done(*inputs)[:-2]
        return self.forward_bounded(*inputs)

    def forward_bounded(self, *inputs: Tensor) -> tuple[Tensor, ...]:
        """Fixed-horizon execution for training and CUDA Graph capture."""

        if len(inputs) != len(self.resource_ids):
            raise ResourceGraphCompileError(
                "static ProgramLoop received an incorrect resource value count"
            )
        values = list(inputs)
        continuation = values[self._continue_index]
        if continuation.ndim != 1:
            raise ResourceGraphCompileError("static ProgramLoop continuation must remain rank one")
        active = torch.ones_like(continuation, dtype=torch.bool)
        for iteration in range(self.max_iterations):
            values = list(self._iteration(tuple(values), active))
            if iteration + 1 >= self.min_iterations:
                active = active & (values[self._continue_index] > 0.0)
        return tuple(values)

    def _iteration(self, current: tuple[Tensor, ...], active: Tensor) -> tuple[Tensor, ...]:
        values = list(current)
        for stage, stage_inputs, stage_outputs in zip(
            self.stages, self._input_indices, self._output_indices, strict=True
        ):
            snapshot = tuple(values)
            publications: list[tuple[int, Tensor]] = []
            for node, input_indices, output_indices in zip(
                stage, stage_inputs, stage_outputs, strict=True
            ):
                outputs = node(*(snapshot[index] for index in input_indices))
                publications.extend(zip(output_indices, outputs, strict=True))
            for index, candidate in publications:
                previous = values[index]
                mask_shape = [1] * candidate.ndim
                mask_shape[self._batch_axes[index]] = active.shape[0]
                values[index] = torch.where(active.reshape(mask_shape), candidate, previous)
        return tuple(values)

    @torch.no_grad()
    def forward_until_done(self, *inputs: Tensor) -> tuple[Tensor, ...]:
        """Run an inference loop until every row stops or the host budget ends.

        The final two outputs report physical body calls and per-row active
        steps. The trainable ``forward`` and ``credit_gradient`` retain the
        fixed-horizon reverse tape.
        """

        if len(inputs) != len(self.resource_ids):
            raise ResourceGraphCompileError(
                "static ProgramLoop received an incorrect resource value count"
            )
        continuation = inputs[self._continue_index]
        if continuation.ndim != 1:
            raise ResourceGraphCompileError("static ProgramLoop continuation must remain rank one")
        step = torch.zeros((), dtype=torch.int64, device=continuation.device)
        active = torch.ones_like(continuation, dtype=torch.bool)
        row_steps = torch.zeros_like(continuation, dtype=torch.int64)

        def condition(current_step: Tensor, current_active: Tensor, current_steps: Tensor,
                      *current_values: Tensor) -> Tensor:
            del current_steps, current_values
            return (current_step < self.max_iterations) & current_active.any()

        def body(current_step: Tensor, current_active: Tensor, current_steps: Tensor,
                 *current_values: Tensor) -> tuple[Tensor, ...]:
            next_values = list(self._iteration(current_values, current_active))
            for index in self._unwritten_indices:
                next_values[index] = next_values[index].clone()
            next_step = current_step + 1
            next_active = current_active & torch.where(
                next_step >= self.min_iterations,
                next_values[self._continue_index] > 0.0,
                torch.ones_like(current_active),
            )
            return (
                next_step, next_active, current_steps + current_active.to(torch.int64),
                *next_values,
            )

        physical_steps, _, row_steps, *final_values = torch.while_loop(
            condition,
            body,
            (step, active, row_steps, *inputs),
        )
        return (*final_values, physical_steps, row_steps)

    def credit_gradient(
        self,
        *inputs: Tensor,
        terminal_cotangents: Mapping[str, Tensor],
        create_graph: bool = True,
    ) -> StaticProgramLoopCreditResult:
        """Compose local VJPs through the bounded horizon actually executed.

        The continuation comparison is a control receipt. Each iteration's
        activity mask gates the candidate-versus-carry split in reverse, but
        no derivative is assigned to the discrete continuation predicate.
        """

        if len(inputs) != len(self.resource_ids):
            raise ResourceGraphCompileError(
                "static ProgramLoop credit lowering received an incorrect resource value count"
            )
        if not isinstance(terminal_cotangents, Mapping) or not terminal_cotangents:
            raise TypeError("terminal_cotangents must be a non-empty resource-id mapping")
        resource_index = {resource_id: index for index, resource_id in enumerate(self.resource_ids)}
        unknown = set(terminal_cotangents).difference(resource_index)
        if unknown:
            raise ResourceGraphCompileError(
                f"terminal cotangents name unknown resources: {sorted(unknown)!r}"
            )

        values = list(inputs)
        continuation = values[self._continue_index]
        if continuation.ndim != 1:
            raise ResourceGraphCompileError("static ProgramLoop continuation must remain rank one")
        versions = list(range(len(values)))
        next_version = len(values)
        active = torch.ones_like(continuation, dtype=torch.bool)
        iteration_active: list[Tensor] = []
        tapes: list[
            tuple[
                str,
                nn.Module,
                tuple[int, ...],
                tuple[Tensor, ...],
                tuple[Tensor, ...],
                tuple[nn.Parameter, ...],
            ]
        ] = []
        stage_receipts: list[
            tuple[
                tuple[int, ...],
                tuple[tuple[int, int, int, int, int, Tensor, Tensor], ...],
            ]
        ] = []
        parameter_names: dict[int, str] = {}
        parameter_values: dict[int, nn.Parameter] = {}

        for iteration in range(self.max_iterations):
            iteration_active.append(active)
            for stage, stage_ids, stage_inputs, stage_outputs in zip(
                self.stages,
                self._node_ids,
                self._input_indices,
                self._output_indices,
                strict=True,
            ):
                snapshot_values = tuple(values)
                snapshot_versions = tuple(versions)
                publications: list[tuple[int, int, Tensor]] = []
                stage_tape_indices: list[int] = []
                for node, node_id, input_indices, output_indices in zip(
                    stage, stage_ids, stage_inputs, stage_outputs, strict=True
                ):
                    local_inputs = tuple(
                        value.detach().requires_grad_(value.is_floating_point() or value.is_complex())
                        for value in (snapshot_values[index] for index in input_indices)
                    )
                    if any(not value.requires_grad for value in local_inputs):
                        raise ResourceGraphCompileError(
                            "static ProgramLoop credit lowering requires floating or complex node inputs"
                        )
                    outputs = tuple(node(*local_inputs))
                    if len(outputs) != len(output_indices):
                        raise ResourceGraphCompileError(
                            "static ProgramLoop node returned an incorrect output count"
                        )
                    parameters = tuple(node.parameters())
                    for name, parameter in node.named_parameters():
                        identity = id(parameter)
                        parameter_names.setdefault(
                            identity,
                            f"{node_id}.{name.removeprefix('module.')}",
                        )
                        parameter_values[identity] = parameter
                    tape_index = len(tapes)
                    stage_tape_indices.append(tape_index)
                    tapes.append(
                        (
                            node_id,
                            node,
                            tuple(snapshot_versions[index] for index in input_indices),
                            local_inputs,
                            outputs,
                            parameters,
                        )
                    )
                    publications.extend(
                        (resource_index_value, tape_index, candidate)
                        for resource_index_value, candidate in zip(output_indices, outputs, strict=True)
                    )
                stage_commits: list[tuple[int, int, int, int, int, Tensor, Tensor]] = []
                for resource_index_value, tape_index, candidate in publications:
                    previous = values[resource_index_value]
                    if previous.shape != candidate.shape:
                        raise ResourceGraphCompileError("static ProgramLoop output Tensor ABI changed")
                    batch_axis = self._batch_axes[resource_index_value]
                    if previous.shape[batch_axis] != active.shape[0]:
                        raise ResourceGraphCompileError("static ProgramLoop batch dimensions do not match continuation")
                    mask_shape = [1] * candidate.ndim
                    mask_shape[batch_axis] = active.shape[0]
                    prior_version = versions[resource_index_value]
                    output_version = next_version
                    next_version += 1
                    values[resource_index_value] = torch.where(active.reshape(mask_shape), candidate, previous)
                    versions[resource_index_value] = output_version
                    output_slot = len([item for item in stage_commits if item[0] == tape_index])
                    stage_commits.append(
                        (
                            tape_index,
                            output_slot,
                            prior_version,
                            output_version,
                            batch_axis,
                            candidate,
                            active,
                        )
                    )
                stage_receipts.append((tuple(stage_tape_indices), tuple(stage_commits)))
            if iteration + 1 >= self.min_iterations:
                active = active & (values[self._continue_index] > 0.0)

        cotangents: dict[int, Tensor] = {}
        for resource_id, cotangent in terminal_cotangents.items():
            if not isinstance(cotangent, Tensor):
                raise TypeError("terminal cotangents must be Tensors")
            index = resource_index[resource_id]
            expected_value = values[index]
            if cotangent.shape != expected_value.shape:
                raise ResourceGraphCompileError(
                    f"terminal cotangent for {resource_id!r} must match its final resource shape"
                )
            cotangents[versions[index]] = cotangent.to(
                device=expected_value.device,
                dtype=expected_value.dtype,
            )

        node_output_cotangents: dict[int, list[Tensor]] = {
            tape_index: [torch.zeros_like(output) for output in tape[4]]
            for tape_index, tape in enumerate(tapes)
        }
        parameter_cotangents: dict[int, Tensor] = {}
        for stage_tape_indices, stage_commits in reversed(stage_receipts):
            for (
                tape_index,
                output_slot,
                prior_version,
                output_version,
                batch_axis,
                candidate,
                iteration_mask,
            ) in reversed(stage_commits):
                output_cotangent = cotangents.get(output_version, torch.zeros_like(candidate))
                mask_shape = [1] * candidate.ndim
                mask_shape[batch_axis] = iteration_mask.shape[0]
                active_mask = iteration_mask.reshape(mask_shape)
                node_output_cotangents[tape_index][output_slot] = torch.where(
                    active_mask, output_cotangent, torch.zeros_like(output_cotangent)
                )
                carried = torch.where(active_mask, torch.zeros_like(output_cotangent), output_cotangent)
                cotangents[prior_version] = (
                    carried if prior_version not in cotangents else cotangents[prior_version] + carried
                )
            for tape_index in reversed(stage_tape_indices):
                _node_id, node, input_versions, local_inputs, outputs, parameters = tapes[tape_index]
                local_vjp = getattr(node, "local_vjp", None)
                if not callable(local_vjp):
                    raise ResourceGraphCompileError(
                        "static ProgramLoop credit lowering requires every lowered node to declare local_vjp"
                    )
                result = local_vjp(
                    local_inputs,
                    outputs,
                    tuple(node_output_cotangents[tape_index]),
                    parameters,
                    create_graph,
                )
                if not isinstance(result, LocalVJPResult):
                    raise ResourceGraphCompileError("node local_vjp must return LocalVJPResult")
                if len(result.input_cotangents) != len(input_versions):
                    raise ResourceGraphCompileError("node local_vjp returned an incorrect input cotangent count")
                if len(result.parameter_cotangents) != len(parameters):
                    raise ResourceGraphCompileError("node local_vjp returned an incorrect parameter cotangent count")
                for version, input_value, cotangent in zip(
                    input_versions, local_inputs, result.input_cotangents, strict=True
                ):
                    if cotangent is None:
                        continue
                    if cotangent.shape != input_value.shape:
                        raise ResourceGraphCompileError("node local_vjp input cotangent shape mismatch")
                    cotangents[version] = (
                        cotangent if version not in cotangents else cotangents[version] + cotangent
                    )
                for parameter, cotangent in zip(parameters, result.parameter_cotangents, strict=True):
                    if cotangent is None:
                        continue
                    if cotangent.shape != parameter.shape:
                        raise ResourceGraphCompileError("node local_vjp parameter cotangent shape mismatch")
                    identity = id(parameter)
                    parameter_cotangents[identity] = (
                        cotangent
                        if identity not in parameter_cotangents
                        else parameter_cotangents[identity] + cotangent
                    )

        return StaticProgramLoopCreditResult(
            resource_values=tuple(values),
            resource_cotangents={
                resource_id: cotangents.get(index)
                for index, resource_id in enumerate(self.resource_ids)
            },
            parameter_cotangents={
                parameter_names[identity]: parameter_cotangents.get(identity)
                for identity in parameter_names
            },
            parameters={
                parameter_names[identity]: parameter_values[identity]
                for identity in parameter_names
            },
            iteration_active=tuple(iteration_active),
        )

    def capture(self, *inputs: Tensor) -> CapturedResourceGraphExecutionPlan:
        """Capture this frozen bounded-loop plan for a fixed CUDA input bucket."""

        return _capture_static_tensor_plan(self, *inputs, capture_forward=self.forward_bounded)


class StaticProgramOutputExecutionPlan(nn.Module):
    """A compiled program whose public result is an explicit resource subset."""

    _component_reference: ClassVar[str] = "arti/static-program-output-execution-plan@1"

    def __init__(
        self,
        body: ResourceGraphExecutionPlan | StaticProgramGraphExecutionPlan | StaticDataflowProgramExecutionPlan | StaticRoutedProgramExecutionPlan | StaticProgramLoopExecutionPlan | StaticDataflowLoopExecutionPlan,
        output_resource_ids: Sequence[str],
        executed_step_ids: Sequence[str],
    ) -> None:
        super().__init__()
        self.body = body
        self.resource_ids = body.resource_ids
        self.output_resource_ids = tuple(output_resource_ids)
        self.executed_step_ids = tuple(executed_step_ids)
        self._output_indices = tuple(self.resource_ids.index(name) for name in self.output_resource_ids)
        self.context_connection_ids = getattr(body, "context_connection_ids", ())
        self.credit_mask_connection_ids = getattr(body, "credit_mask_connection_ids", ())
        self.arrival_width = getattr(body, "arrival_width", getattr(getattr(body, "body", None), "arrival_width", 0))

    def forward(self, *inputs: Tensor) -> tuple[Tensor, ...]:
        values = self.body(*inputs)
        return tuple(values[index] for index in self._output_indices)

    @torch.no_grad()
    def forward_until_done(
        self, *inputs: Tensor, host_step_limit: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        if isinstance(self.body, StaticDataflowLoopExecutionPlan):
            values = self.body.forward_until_done(*inputs, host_step_limit=host_step_limit)
        elif isinstance(self.body, StaticProgramLoopExecutionPlan):
            if host_step_limit is not None:
                raise ResourceGraphCompileError("host_step_limit requires an open dataflow loop")
            values = self.body.forward_until_done(*inputs)
        else:
            raise ResourceGraphCompileError("forward_until_done requires a loop output plan")
        suffix_start = len(self.resource_ids) + isinstance(self.body, StaticDataflowLoopExecutionPlan)
        return (*self._select_outputs(values), *values[suffix_start:])

    @torch.no_grad()
    def forward_until_done_with_status(
        self, *inputs: Tensor, host_step_limit: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        if not isinstance(self.body, StaticDataflowLoopExecutionPlan):
            raise ResourceGraphCompileError("status output requires a dataflow loop")
        values = self.body.forward_until_done_with_status(*inputs, host_step_limit=host_step_limit)
        return (*self._select_outputs(values), *values[len(self.resource_ids) + 1:])

    @torch.no_grad()
    def forward_until_done_with_route_receipt(
        self, *inputs: Tensor, host_step_limit: Tensor, receipt_capacity: int,
        sample_routes: bool = False,
        credit_mask_histories: tuple[Tensor, ...] | None = None,
        initial_active: Tensor | None = None,
        completed_steps: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        if not isinstance(self.body, StaticDataflowLoopExecutionPlan):
            raise ResourceGraphCompileError("route receipt requires an open dataflow loop")
        values = self.body.forward_until_done_with_route_receipt(
            *inputs, host_step_limit=host_step_limit, receipt_capacity=receipt_capacity,
            sample_routes=sample_routes, credit_mask_histories=credit_mask_histories,
            initial_active=initial_active, completed_steps=completed_steps,
        )
        return (*self._select_outputs(values), *values[len(self.resource_ids) + 1:])

    def _select_outputs(self, values: tuple[Tensor, ...]) -> tuple[Tensor, ...]:
        return tuple(values[index] for index in self._output_indices)

    def replay_tensor_receipt(
        self, *inputs: Tensor, tensor_receipt: tuple[Tensor, ...],
        row_advantage: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        """Replay a recorded hard loop and expose only declared outputs and route credit."""

        if not isinstance(self.body, StaticDataflowLoopExecutionPlan):
            raise ResourceGraphCompileError("tensor receipt replay requires an open dataflow loop")
        values = self.body.replay_tensor_receipt(
            *inputs, tensor_receipt=tensor_receipt, row_advantage=row_advantage,
        )
        return (*self._select_outputs(values), values[-1])

    def credit_gradient(
        self,
        *inputs: Tensor,
        terminal_cotangents: Mapping[str, Tensor],
        create_graph: bool = True,
        **credit_options: object,
    ) -> StaticProgramGraphCreditResult | StaticProgramLoopCreditResult | StaticDataflowProgramCreditResult | StaticDataflowLoopCreditResult:
        """Reverse declared outputs through the retained graph and its actual receipt."""

        unknown = set(terminal_cotangents).difference(self.output_resource_ids)
        if unknown:
            raise ResourceGraphCompileError(
                f"terminal cotangents name unexposed outputs: {sorted(unknown)!r}"
            )
        if not hasattr(self.body, "credit_gradient"):
            raise ResourceGraphCompileError(
                "this routed output plan has no reverse tape; compile with credit=True"
            )
        return self.body.credit_gradient(
            *inputs,
            terminal_cotangents=terminal_cotangents,
            create_graph=create_graph,
            **credit_options,
        )


class ResourceGraphCompiler:
    """Lower static direct connections while leaving dynamic relation semantics explicit."""

    @classmethod
    def _project_program_graph(
        cls,
        graph: ProgramGraph,
        program_id: str,
        output_resource_ids: Sequence[str],
        *,
        discardable_step_ids: Iterable[str],
        loop_continue_resource_id: str | None = None,
    ) -> tuple[ProgramGraph, tuple[str, ...]]:
        """Prune declared pure steps, closing liveness over loop-carried state."""

        entries = graph.program(program_id)
        outputs = tuple(output_resource_ids)
        if not outputs or len(outputs) != len(set(outputs)) or set(outputs) - set(graph.resources):
            raise ResourceGraphCompileError("output resources must be distinct graph resources")
        if loop_continue_resource_id is not None and loop_continue_resource_id not in graph.resources:
            raise ResourceGraphCompileError("loop continuation must be a declared resource")

        def step_ids(entry: str | ProgramStage | ProgramJoin | ProgramRoute) -> tuple[str | ProgramJoin | ProgramRoute, ...]:
            if isinstance(entry, ProgramStage):
                return entry.step_ids
            if isinstance(entry, ProgramJoin):
                return (entry,)
            if isinstance(entry, ProgramRoute):
                return (entry,)
            return (entry,)

        declared_steps = {
            step_id for entry in entries for step_id in step_ids(entry)
            if isinstance(step_id, str)
        }
        discardable = set(discardable_step_ids)
        if discardable - declared_steps:
            raise ResourceGraphCompileError("discardable steps must occur in the projected program")

        def resources_for(step_id: str | ProgramJoin | ProgramRoute) -> tuple[set[str], set[str], tuple[str, ...]]:
            if isinstance(step_id, ProgramRoute):
                route = step_id
                reads = {route.score_resource_id}
                guaranteed_writes: set[str] | None = None
                dependencies: set[str] = set()
                for candidate in route.candidates:
                    candidate_writes: set[str] = set()
                    for entry in graph._route_candidate_entries(candidate):
                        for member in step_ids(entry):
                            member_reads, member_writes, member_dependencies = resources_for(member)
                            reads.update(member_reads)
                            candidate_writes.update(member_writes)
                            dependencies.update(member_dependencies)
                    guaranteed_writes = (
                        candidate_writes if guaranteed_writes is None
                        else guaranteed_writes & candidate_writes
                    )
                return reads, guaranteed_writes or set(), tuple(sorted(dependencies))
            if isinstance(step_id, ProgramJoin):
                node = graph.nodes[step_id.node_id]
                if not isinstance(node, MultiPortProgramNode):
                    raise ResourceGraphCompileError("ProgramJoin requires a multi-port node")
                return {port.resource_id for port in node.input_ports.values()}, set(), ()
            if step_id in graph.connections:
                connection = graph.connections[step_id]
                reads = {connection.source.resource_id}
                reads.update(view.resource_id for view in connection.operand_views.values())
                if connection.destination_view is not None:
                    reads.add(connection.destination.resource_id)
                return reads, {connection.destination.resource_id}, connection.depends_on
            node = graph.nodes[step_id]
            if isinstance(node, ProgramNode):
                return {node.input_resource_id}, {node.output_resource_id}, ()
            return (
                {port.resource_id for port in node.input_ports.values()},
                {port.resource_id for port in node.output_ports.values()},
                (),
            )

        seed = set(outputs)
        if loop_continue_resource_id is not None:
            seed.add(loop_continue_resource_id)
        while True:
            live = set(seed)
            required_connections: set[str] = set()
            retained: list[str | ProgramStage | ProgramJoin | ProgramRoute] = []
            for entry in reversed(entries):
                members = step_ids(entry)
                chosen = tuple(
                    step_id for step_id in members
                    if isinstance(step_id, (ProgramRoute, ProgramJoin))
                    or step_id not in discardable
                    or bool(resources_for(step_id)[1] & live)
                    or step_id in required_connections
                )
                if not chosen:
                    continue
                reads: set[str] = set()
                writes: set[str] = set()
                for step_id in chosen:
                    step_reads, step_writes, dependencies = resources_for(step_id)
                    reads.update(step_reads)
                    writes.update(step_writes)
                    required_connections.update(dependencies)
                live.difference_update(writes)
                live.update(reads)
                if isinstance(entry, ProgramStage):
                    retained.append(ProgramStage(chosen))
                else:
                    retained.append(entry)
            if loop_continue_resource_id is None or live.issubset(seed):
                break
            seed.update(live)
        retained.reverse()
        if not retained:
            raise ResourceGraphCompileError("projected program has no executable steps")
        programs = {**graph._programs, program_id: tuple(retained)}
        projected = graph._copy_for_specialization(programs=programs, share_module_state=True)
        executed_step_ids = tuple(
            step_id.route_id if isinstance(step_id, ProgramRoute)
            else step_id.node_id if isinstance(step_id, ProgramJoin) else step_id
            for entry in retained for step_id in step_ids(entry)
        )
        return projected, executed_step_ids

    @classmethod
    def compile_program_outputs(
        cls,
        graph: ProgramGraph,
        program_id: str,
        output_resource_ids: Sequence[str],
        *,
        discardable_step_ids: Iterable[str] = (),
        credit: bool = False,
    ) -> StaticProgramOutputExecutionPlan:
        """Compile a program for declared outputs, pruning dead pure steps.

        ``discardable_step_ids`` is an explicit purity assertion. Other steps
        remain executable even when their outputs are not observed. The result
        exposes only ``output_resource_ids``; incidental resource state is not
        part of this plan's output contract. Dynamic routes stay live and keep
        their scoring dependencies; candidate execution remains hard-routed.
        ``credit=True`` lowers a routed program through the existing mixed
        reverse tape; its input also includes the declared arrivals tensor.
        """

        if any(loop.program_id == program_id for loop in graph._loops.values()):
            raise ResourceGraphCompileError("loop body output projection requires compile_loop_outputs")
        projected, executed_step_ids = cls._project_program_graph(
            graph, program_id, output_resource_ids,
            discardable_step_ids=discardable_step_ids,
        )
        body = cls.compile_program(projected, program_id, _force_dataflow=credit)
        if not isinstance(body, (ResourceGraphExecutionPlan, StaticProgramGraphExecutionPlan,
                                 StaticDataflowProgramExecutionPlan, StaticRoutedProgramExecutionPlan)):
            raise ResourceGraphCompileError("projected program did not lower to an execution plan")
        return StaticProgramOutputExecutionPlan(body, output_resource_ids, executed_step_ids)

    @classmethod
    def compile_loop_outputs(
        cls,
        graph: ProgramGraph,
        loop_id: str,
        output_resource_ids: Sequence[str],
        *,
        discardable_step_ids: Iterable[str] = (),
        example_contexts: Mapping[str, Tensor] | None = None,
        example_credit_masks: Mapping[str, Tensor] | None = None,
    ) -> StaticProgramOutputExecutionPlan:
        """Project loop outputs without dropping state needed by later iterations."""

        loop = graph.loop(loop_id)
        projected, executed_step_ids = cls._project_program_graph(
            graph, loop.program_id, output_resource_ids,
            discardable_step_ids=discardable_step_ids,
            loop_continue_resource_id=loop.continue_resource_id,
        )
        body = cls.compile_loop(
            projected, loop_id,
            example_contexts=example_contexts,
            example_credit_masks=example_credit_masks,
        )
        return StaticProgramOutputExecutionPlan(body, output_resource_ids, executed_step_ids)

    @classmethod
    def compile_program(
        cls,
        graph: ProgramGraph,
        program_id: str,
        *,
        iterations: int = 1,
        example_contexts: Mapping[str, Tensor] | None = None,
        example_credit_masks: Mapping[str, Tensor] | None = None,
        _force_dataflow: bool = False,
    ) -> ResourceGraphExecutionPlan | StaticProgramGraphExecutionPlan | StaticDataflowProgramExecutionPlan | StaticRoutedProgramExecutionPlan:
        """Lower a declared connection program with a fixed local iteration count.

        ``iterations`` is a compile-time horizon, not an external event count.
        The resulting plan carries each iteration's resource state directly in
        its tensor values, so it has no Python loop during invocation.
        """

        if not isinstance(graph, ProgramGraph):
            raise TypeError("graph must be ProgramGraph")
        if type(iterations) is not int or iterations <= 0:
            raise ResourceGraphCompileError("iterations must be a positive integer")
        entries = graph.program(program_id)
        if any(isinstance(entry, ProgramRoute) for entry in entries):
            resource_ids = tuple(graph.resources)
            batch_axes = tuple(resource.resolve().view.batch_axis for resource in graph.resources.values())
            contexts = {} if example_contexts is None else dict(example_contexts)
            masks = _credit_mask_map(example_credit_masks)
            referenced_joins: set[str] = set()
            def visit_joins(sequence: Sequence[str | ProgramStage | ProgramJoin | ProgramRoute]) -> None:
                for entry in sequence:
                    if isinstance(entry, ProgramJoin):
                        referenced_joins.add(entry.join_id)
                    elif isinstance(entry, ProgramRoute):
                        for candidate in entry.candidates:
                            visit_joins(graph._route_candidate_entries(candidate))

            visit_joins(entries)
            join_scope = tuple(
                (join, graph._join_node(join))
                for join_id, join in graph._joins.items() if join_id in referenced_joins
            )

            def lower_stages(
                stages: Sequence[tuple[str, ...]],
                *, batch_region: tuple[int, int] | None = None,
            ) -> StaticProgramGraphExecutionPlan:
                completed_connections: set[str] = set()
                resolved = tuple(
                    graph._program_steps(stage, completed_connections=completed_connections)
                    for stage in stages
                )
                if any(
                    isinstance(step, Connection)
                    and set(step.depends_on).intersection(stage_ids)
                    for stage_ids, stage in zip(stages, resolved, strict=True)
                    for step in stage
                ):
                    raise ResourceGraphCompileError(
                        "parallel stage connections cannot depend on stage peers"
                    )
                if any(
                    not isinstance(step, (Connection, MultiPortProgramNode))
                    for stage in resolved for step in stage
                ):
                    raise ResourceGraphCompileError(
                        "routed program lowering requires lowered multi-port nodes or connections"
                    )
                connection_plans = {}
                for stage in resolved:
                    for step in stage:
                        if isinstance(step, Connection):
                            compiled = cls.compile(
                                graph, (step.connection_id,),
                                example_contexts={step.connection_id: contexts[step.connection_id]}
                                if step.is_conditional and step.connection_id in contexts else {},
                                example_credit_masks={step.connection_id: masks[step.connection_id]}
                                if step.connection_id in masks else {},
                                _completed_connections=step.depends_on,
                            )
                            connection_plans[step.connection_id] = (
                                ResourceGraphExecutionPlan(
                                    resource_ids=compiled.resource_ids,
                                    templates=tuple(
                                        template.slice_batch_range(*batch_region)
                                        for template in compiled.templates
                                    ),
                                    connections=tuple(compiled.connections),
                                )
                                if batch_region is not None else compiled
                            )
                return StaticProgramGraphExecutionPlan(
                    resource_ids=resource_ids, stages=resolved,
                    connection_plans=connection_plans,
                )

            def lower_dataflow_candidate(
                candidate: str, *, batch_region: tuple[int, int] | None = None,
            ) -> StaticDataflowProgramExecutionPlan:
                candidate_entries = graph._route_candidate_entries(candidate)
                operations: list[
                    tuple[str, MultiPortProgramNode | Connection | _StaticDataflowRouteStep, ProgramJoin | None]
                ] = []
                connections: dict[str, ResourceGraphExecutionPlan] = {}
                completed_connections: set[str] = set()
                for member in candidate_entries:
                    if isinstance(member, ProgramRoute):
                        operations.append((
                            "route", _StaticDataflowRouteStep(
                                member, lower_route(member, batch_region=batch_region),
                            ), None,
                        ))
                        continue
                    if isinstance(member, ProgramJoin):
                        node = graph._join_node(member)
                        _lower_static_program_node(node)
                        operations.append(("join", node, member))
                        continue
                    step_ids = member.step_ids if isinstance(member, ProgramStage) else (member,)
                    if isinstance(member, ProgramStage) and any(
                        set(graph.connections[step_id].depends_on).intersection(step_ids)
                        for step_id in step_ids if step_id in graph.connections
                    ):
                        raise ResourceGraphCompileError(
                            "parallel stage connections cannot depend on stage peers"
                        )
                    for step in graph._program_steps(
                        step_ids, completed_connections=completed_connections,
                    ):
                        if isinstance(step, Connection):
                            if step.connection_id not in connections:
                                compiled = cls.compile(
                                    graph, (step.connection_id,),
                                    example_contexts={step.connection_id: contexts[step.connection_id]}
                                    if step.is_conditional and step.connection_id in contexts else {},
                                    example_credit_masks={step.connection_id: masks[step.connection_id]}
                                    if step.connection_id in masks else {},
                                    _completed_connections=step.depends_on,
                                )
                                connections[step.connection_id] = (
                                    ResourceGraphExecutionPlan(
                                        resource_ids=compiled.resource_ids,
                                        templates=tuple(
                                            template.slice_batch_range(*batch_region)
                                            for template in compiled.templates
                                        ),
                                        connections=tuple(compiled.connections),
                                    ) if batch_region is not None else compiled
                                )
                            operations.append(("connection", step, None))
                        else:
                            _lower_static_program_node(step)
                            operations.append(("node", step, None))
                return StaticDataflowProgramExecutionPlan(
                    resource_ids=resource_ids,
                    templates=tuple(
                        resource.resolve().view.slice_batch_range(*batch_region)
                        if batch_region is not None else resource.resolve().view
                        for resource in graph.resources.values()
                    ),
                    operations=operations,
                    connection_plans=connections,
                    join_scope=join_scope,
                )

            def lower_route(
                entry: ProgramRoute, *, batch_region: tuple[int, int] | None = None,
            ) -> StaticProgramRouteStep:
                def candidate_has_join(candidate: str) -> bool:
                    return any(
                        isinstance(member, ProgramJoin)
                        or isinstance(member, ProgramRoute) and any(
                            candidate_has_join(child) for child in member.candidates
                        )
                        for member in graph._route_candidate_entries(candidate)
                    )

                has_join = any(candidate_has_join(candidate) for candidate in entry.candidates)
                has_nested_route = any(
                    isinstance(member, ProgramRoute)
                    for candidate in entry.candidates
                    for member in graph._route_candidate_entries(candidate)
                )
                if (has_join or has_nested_route) and entry.execution_mode == "all_candidates":
                    raise ResourceGraphCompileError(
                        "joining or nested candidates require sparse routed execution"
                    )
                lower_candidate = lower_dataflow_candidate if has_join or has_nested_route else None
                if entry.selection_scope == "batch" or entry.execution_mode == "all_candidates":
                    return StaticProgramRouteStep(
                        resource_ids, entry,
                        tuple(
                            lower_candidate(candidate, batch_region=batch_region) if lower_candidate else
                            lower_stages(graph._route_candidate_stages(candidate), batch_region=batch_region)
                            for candidate in entry.candidates
                        ),
                        batch_axes,
                    )
                base = 0 if batch_region is None else batch_region[0]
                batch_size = (
                    graph.resource(entry.score_resource_id).resolve().view.value.shape[0]
                    if batch_region is None else batch_region[1] - batch_region[0]
                )
                views = tuple(resource.resolve().view for resource in graph.resources.values())
                share_cohort_plans = all(
                    view.mask is None
                    and (
                        view.index_map is None
                        or view.index_map.coordinates is None
                        or view.index_map.coordinates.ndim != len(view.index_map.target_shape) + 2
                    )
                    for view in views
                )
                pending = [(0, batch_size)]
                regions: set[tuple[int, int]] = set()
                while pending:
                    start, size = pending.pop()
                    if (start, size) in regions:
                        continue
                    regions.add((start, size))
                    if size > 1:
                        middle = size // 2
                        pending.extend(((start, middle), (start + middle, size - middle)))
                plan_regions = (
                    {(0, size) for _start, size in regions}
                    if share_cohort_plans else regions
                )
                plans = {
                    (start, size): tuple(
                        lower_candidate(
                            candidate,
                            batch_region=(base + start, base + start + size)
                            if size != batch_size else batch_region,
                        ) if lower_candidate else lower_stages(
                            graph._route_candidate_stages(candidate),
                            batch_region=(base + start, base + start + size)
                            if size != batch_size else batch_region,
                        )
                        for candidate in entry.candidates
                    )
                    for start, size in sorted(plan_regions)
                }
                return StaticProgramRouteStep(
                    resource_ids, entry, plans[(0, batch_size)], batch_axes,
                    cohort_plans=plans, share_cohort_plans=share_cohort_plans,
                )

            has_nested_routes = any(
                isinstance(entry, ProgramRoute)
                and any(
                    isinstance(member, ProgramRoute)
                    for candidate in entry.candidates
                    for member in graph._route_candidate_entries(candidate)
                )
                for entry in entries
            )
            if join_scope or _force_dataflow or has_nested_routes:
                operations: list[tuple[str, MultiPortProgramNode | Connection | _StaticDataflowRouteStep, ProgramJoin | None]] = []
                connections: dict[str, ResourceGraphExecutionPlan] = {}
                for entry in entries * iterations:
                    if isinstance(entry, ProgramRoute):
                        routed = lower_route(entry)
                        operations.append(("route", _StaticDataflowRouteStep(entry, routed), None))
                    elif isinstance(entry, ProgramJoin):
                        node = graph._join_node(entry)
                        _lower_static_program_node(node)
                        operations.append(("join", node, entry))
                    else:
                        step_ids = entry.step_ids if isinstance(entry, ProgramStage) else (entry,)
                        for step in graph._program_steps(step_ids):
                            if isinstance(step, Connection):
                                if step.connection_id not in connections:
                                    connections[step.connection_id] = cls.compile(
                                        graph, (step.connection_id,),
                                        example_contexts={step.connection_id: contexts[step.connection_id]}
                                        if step.is_conditional and step.connection_id in contexts else {},
                                        example_credit_masks={step.connection_id: masks[step.connection_id]}
                                        if step.connection_id in masks else {},
                                    )
                                operations.append(("connection", step, None))
                            else:
                                _lower_static_program_node(step)
                                operations.append(("node", step, None))
                resources = tuple(graph.resources.values())
                plan = StaticDataflowProgramExecutionPlan(
                    resource_ids=resource_ids,
                    templates=tuple(resource.resolve().view for resource in resources),
                    operations=tuple(operations),
                    connection_plans=connections,
                    join_scope=join_scope,
                )
                if set(contexts) != set(plan.context_connection_ids) or set(masks) != set(
                    plan.credit_mask_connection_ids
                ):
                    raise ResourceGraphCompileError(
                        "routed dataflow contexts and credit masks must match declared connections"
                    )
                return plan

            blocks: list[nn.Module] = []
            pending: list[tuple[str, ...]] = []
            for entry in entries * iterations:
                if isinstance(entry, ProgramRoute):
                    if pending:
                        blocks.append(lower_stages(pending))
                        pending.clear()
                    blocks.append(lower_route(entry))
                else:
                    pending.append(entry.step_ids if isinstance(entry, ProgramStage) else (entry,))
            if pending:
                blocks.append(lower_stages(pending))
            plan = StaticRoutedProgramExecutionPlan(resource_ids, blocks)
            if set(contexts) != set(plan.context_connection_ids) or set(masks) != set(
                plan.credit_mask_connection_ids
            ):
                raise ResourceGraphCompileError(
                    "routed program contexts and credit masks must match declared connections"
                )
            return plan
        if any(isinstance(entry, ProgramJoin) for entry in entries):
            example_contexts = {} if example_contexts is None else dict(example_contexts)
            example_credit_masks = _credit_mask_map(example_credit_masks)
            operations: list[
                tuple[Literal["node", "join", "connection"], MultiPortProgramNode | Connection, ProgramJoin | None]
            ] = []
            connections: dict[str, ResourceGraphExecutionPlan] = {}
            for _iteration in range(iterations):
                for entry in entries:
                    if isinstance(entry, ProgramJoin):
                        node = graph._join_node(entry)
                        _lower_static_program_node(node)
                        operations.append(("join", node, entry))
                        continue
                    node_ids = entry.step_ids if isinstance(entry, ProgramStage) else (entry,)
                    steps = graph._program_steps(node_ids)
                    for step in steps:
                        if isinstance(step, Connection):
                            if step.connection_id not in connections:
                                connections[step.connection_id] = cls.compile(
                                    graph,
                                    (step.connection_id,),
                                    example_contexts={
                                        step.connection_id: example_contexts[step.connection_id]
                                    } if step.is_conditional and step.connection_id in example_contexts else {},
                                    example_credit_masks={
                                        step.connection_id: example_credit_masks[step.connection_id]
                                    } if step.connection_id in example_credit_masks else {},
                                )
                            operations.append(("connection", step, None))
                        else:
                            _lower_static_program_node(step)
                            operations.append(("node", step, None))
            if set(example_contexts) != {
                connection_id for connection_id, plan in connections.items() if plan.context_connection_ids
            }:
                raise ResourceGraphCompileError(
                    "example_contexts must bind every conditional connection in the program"
                )
            if set(example_credit_masks) != {
                connection_id for connection_id, plan in connections.items() if plan.credit_mask_connection_ids
            }:
                raise ResourceGraphCompileError(
                    "example_credit_masks must bind every Bernoulli connection in the program"
                )
            resources = tuple(graph.resources.values())
            with torch.no_grad():
                graph.execute_program_functional(
                    program_id,
                    iterations=iterations,
                    contexts=example_contexts,
                    credit_masks=example_credit_masks,
                )
            return StaticDataflowProgramExecutionPlan(
                resource_ids=tuple(resource.spec.resource_id for resource in resources),
                templates=tuple(resource.resolve().view for resource in resources),
                operations=tuple(operations),
                connection_plans=connections,
            )
        step_ids = tuple(
            step_id
            for entry in entries
            for step_id in (entry.step_ids if isinstance(entry, ProgramStage) else (entry,))
        )
        if any(step_id in graph.nodes for step_id in step_ids):
            stages: list[tuple[MultiPortProgramNode | Connection, ...]] = []
            connections: dict[str, ResourceGraphExecutionPlan] = {}
            example_contexts = {} if example_contexts is None else dict(example_contexts)
            example_credit_masks = _credit_mask_map(example_credit_masks)
            for _iteration in range(iterations):
                for entry in entries:
                    node_ids = entry.step_ids if isinstance(entry, ProgramStage) else (entry,)
                    steps = graph._program_steps(node_ids)
                    if any(not isinstance(step, (MultiPortProgramNode, Connection)) for step in steps):
                        raise ResourceGraphCompileError(
                            "static program lowering supports Connection and lowered multi-port nodes, not ProgramNode"
                        )
                    for step in steps:
                        if isinstance(step, Connection):
                            if step.connection_id not in connections:
                                connections[step.connection_id] = cls.compile(
                                    graph,
                                    (step.connection_id,),
                                    example_contexts={
                                        step.connection_id: example_contexts[step.connection_id]
                                    } if step.is_conditional and step.connection_id in example_contexts else {},
                                    example_credit_masks={
                                        step.connection_id: example_credit_masks[step.connection_id]
                                    } if step.connection_id in example_credit_masks else {},
                                )
                        else:
                            _lower_static_program_node(step)
                    stages.append(tuple(steps))
            required_contexts = {
                connection_id for connection_id, plan in connections.items()
                if plan.context_connection_ids
            }
            if set(example_contexts) != required_contexts:
                raise ResourceGraphCompileError(
                    "example_contexts must bind every conditional connection in the program"
                )
            required_masks = {
                connection_id for connection_id, plan in connections.items()
                if plan.credit_mask_connection_ids
            }
            if set(example_credit_masks) != required_masks:
                raise ResourceGraphCompileError(
                    "example_credit_masks must bind every Bernoulli connection in the program"
                )
            resources = tuple(graph.resources.values())
            with torch.no_grad():
                execution = graph.execute_program_functional(
                    program_id,
                    iterations=iterations,
                    contexts=example_contexts,
                    credit_masks=example_credit_masks,
                )
            final_views = {
                item.spec.resource_id: item.active_view for item in execution.state.resources
            }
            for resource in resources:
                resource.spec.validate(
                    final_views[resource.spec.resource_id],
                    name=f"compiled {resource.spec.resource_id}",
                )
            return StaticProgramGraphExecutionPlan(
                resource_ids=tuple(resource.spec.resource_id for resource in resources),
                stages=tuple(stages),
                connection_plans=connections,
            )
        return cls.compile(
            graph,
            step_ids * iterations,
            example_contexts=example_contexts,
            example_credit_masks=example_credit_masks,
        )

    @classmethod
    def compile_join(cls, graph: ProgramGraph, join_id: str) -> StaticProgramJoinExecutionPlan:
        """Lower one Formula-backed all-new join to resource tensors plus readiness masks."""

        if not isinstance(graph, ProgramGraph):
            raise TypeError("graph must be ProgramGraph")
        try:
            join = graph._joins[join_id]
        except KeyError as error:
            raise ResourceGraphCompileError(f"unknown ProgramJoin {join_id!r}") from error
        node = graph._join_node(join)
        _lower_static_program_node(node)
        resources = tuple(graph.resources.values())
        return StaticProgramJoinExecutionPlan(
            resource_ids=tuple(resource.spec.resource_id for resource in resources),
            templates=tuple(resource.resolve().view for resource in resources),
            join=join,
            node=node,
        )

    @classmethod
    def compile_loop(
        cls,
        graph: ProgramGraph,
        loop_id: str,
        *,
        example_contexts: Mapping[str, Tensor] | None = None,
        example_credit_masks: Mapping[str, Tensor] | None = None,
    ) -> StaticProgramLoopExecutionPlan | StaticDataflowLoopExecutionPlan:
        """Lower a bounded program loop, including declared static frontiers."""

        if not isinstance(graph, ProgramGraph):
            raise TypeError("graph must be ProgramGraph")
        loop = graph.loop(loop_id)
        entries = graph.program(loop.program_id)
        if any(isinstance(entry, (ProgramJoin, ProgramRoute)) for entry in entries):
            try:
                body = cls.compile_program(
                    graph,
                    loop.program_id,
                    example_contexts=example_contexts,
                    example_credit_masks=example_credit_masks,
                    _force_dataflow=True,
                )
            except ResourceGraphCompileError as error:
                if loop.max_iterations is None:
                    raise ResourceGraphCompileError(
                        f"open-horizon ProgramLoop body cannot be lowered: {error}"
                    ) from error
                raise
            if not isinstance(body, StaticDataflowProgramExecutionPlan):
                raise ResourceGraphCompileError("mixed loop body must lower to a dataflow program")
            plan = StaticDataflowLoopExecutionPlan(body=body, loop=loop)
            if loop.max_iterations is not None and not any(
                isinstance(entry, ProgramRoute) and entry.selection_scope == "sample"
                for entry in entries
            ):
                with torch.no_grad():
                    graph.execute_loop_functional(
                        loop_id,
                        contexts=example_contexts,
                        credit_masks=example_credit_masks,
                    )
            return plan
        if loop.max_iterations is None:
            raise ResourceGraphCompileError(
                "open-horizon ProgramLoop lowering requires a mixed dataflow body"
            )
        stages: list[tuple[MultiPortProgramNode, ...]] = []
        for entry in entries:
            node_ids = entry.step_ids if isinstance(entry, ProgramStage) else (entry,)
            steps = graph._program_steps(node_ids)
            if any(not isinstance(step, MultiPortProgramNode) for step in steps):
                raise ResourceGraphCompileError(
                    "static ProgramLoop lowering supports FormulaProgramNode or registered FabricModuleNode regions only"
                )
            for step in steps:
                _lower_static_program_node(step)
            stages.append(tuple(steps))
        resources = tuple(graph.resources.values())
        with torch.no_grad():
            execution = graph.execute_loop_functional(loop_id)
        final_views = {item.spec.resource_id: item.active_view for item in execution.state.resources}
        for resource in resources:
            resource.spec.validate(
                final_views[resource.spec.resource_id], name=f"compiled {resource.spec.resource_id}"
            )
        return StaticProgramLoopExecutionPlan(
            resource_ids=tuple(resource.spec.resource_id for resource in resources),
            templates=tuple(resource.resolve().view for resource in resources),
            stages=tuple(stages),
            loop=loop,
        )

    @classmethod
    def compile(
        cls,
        graph: ProgramGraph,
        connection_ids: Sequence[str],
        *,
        example_contexts: Mapping[str, Tensor] | None = None,
        example_credit_masks: Mapping[str, Tensor] | None = None,
        _completed_connections: Sequence[str] = (),
    ) -> ResourceGraphExecutionPlan:
        if not isinstance(graph, ProgramGraph):
            raise TypeError("graph must be ProgramGraph")
        connection_ids = tuple(connection_ids)
        if not connection_ids:
            raise ResourceGraphCompileError("compiled graph fragments require at least one connection")
        connections = graph._connection_sequence(
            connection_ids, completed_connections=_completed_connections
        )
        example_contexts = {} if example_contexts is None else dict(example_contexts)
        example_credit_masks = _credit_mask_map(example_credit_masks)
        conditional_ids = {connection.connection_id for connection in connections if connection.is_conditional}
        if set(example_contexts) != conditional_ids:
            raise ResourceGraphCompileError(
                "example_contexts must bind every conditional connection and no direct connection"
            )
        bernoulli_ids = {
            connection.connection_id
            for connection in connections
            if connection.credit_boundary is not None
            and connection.credit_boundary.mode is CreditBoundaryMode.BERNOULLI
        }
        if set(example_credit_masks) != bernoulli_ids:
            raise ResourceGraphCompileError(
                "example_credit_masks must bind every Bernoulli boundary and no other connection"
            )
        for connection in connections:
            if connection.is_conditional:
                if not isinstance(connection.activation, nn.Module):
                    raise ResourceGraphCompileError(
                        f"conditional connection {connection.connection_id!r} has a non-module activation"
                    )
                if not isinstance(example_contexts[connection.connection_id], Tensor):
                    raise ResourceGraphCompileError(
                        f"conditional connection {connection.connection_id!r} needs a Tensor example context"
                    )
            if connection.transfer is not None and not isinstance(connection.transfer, nn.Module):
                raise ResourceGraphCompileError(
                    f"connection {connection.connection_id!r} has a non-module transfer"
                )
            if getattr(connection, "_static_compile_compatible", True) is not True:
                raise ResourceGraphCompileError(
                    f"connection {connection.connection_id!r} is not static-compile compatible"
                )
        resources = tuple(graph.resources.values())
        prototype_views = {
            resource.spec.resource_id: resource.resolve().view for resource in resources
        }
        # Admission retains the ordinary Formula path's full metadata, Bank, and
        # finite-value checks before a fixed tensor-only execution plan is made.
        with torch.no_grad():
            for connection in connections:
                source_snapshot = ResourceSnapshot(
                    prototype_views[connection.source.resource_id],
                    ResourceBinding(connection.source.resource_id, "default", 0, 0),
                )
                destination_snapshot = ResourceSnapshot(
                    prototype_views[connection.destination.resource_id],
                    ResourceBinding(connection.destination.resource_id, "default", 0, 0),
                )
                resource_snapshots = {
                    resource_id: ResourceSnapshot(
                        view,
                        ResourceBinding(resource_id, "default", 0, 0),
                    )
                    for resource_id, view in prototype_views.items()
                }
                output = connection.apply(
                    source_snapshot,
                    destination_snapshot,
                    resources=resource_snapshots,
                    context=example_contexts.get(connection.connection_id),
                    credit_mask=example_credit_masks.get(connection.connection_id),
                )
                destination = graph.resource(connection.destination.resource_id)
                destination.spec.validate(
                    output, name=f"compiled {destination.spec.resource_id}"
                )
                prototype_views[connection.destination.resource_id] = output
        # Keep the lowered structure separate while retaining the graph's trainable
        # parameter identities for shared weights and optimizer state.
        parameter_memo = {id(parameter): parameter for parameter in graph.parameters()}
        lowered_connections = tuple(deepcopy(connection, parameter_memo) for connection in connections)
        for connection in lowered_connections:
            if isinstance(connection.transfer, FormulaTensorViewTransfer):
                connection.transfer = connection.transfer.lower_static()
        plan = ResourceGraphExecutionPlan(
            resource_ids=tuple(resource.spec.resource_id for resource in resources),
            templates=tuple(resource.resolve().view for resource in resources),
            connections=lowered_connections,
        )
        with torch.no_grad():
            outputs = plan(
                *(resource.resolve().view.value for resource in resources),
                *(example_contexts[connection_id] for connection_id in plan.context_connection_ids),
                *(example_credit_masks[connection_id] for connection_id in plan.credit_mask_connection_ids),
            )
        for resource, value in zip(resources, outputs, strict=True):
            template = resource.resolve().view
            resource.spec.validate(
                TensorView(value, template.axes, index_map=template.index_map, mask=template.mask),
                name=f"compiled {resource.spec.resource_id}",
            )
        return plan


PROGRAM_GRAPH_ARTIFACT_FORMAT = "arti.program-graph"
PROGRAM_GRAPH_ARTIFACT_VERSION = 4


@dataclass(frozen=True)
class ProgramGraphSaveResult:
    """Paths and integrity data produced by :func:`save_program_graph`."""

    tensors_path: Path
    manifest_path: Path
    contract_fingerprint: str
    manifest_sha256: str


def _artifact_paths(path: str | Path) -> tuple[Path, Path]:
    target = Path(path)
    if target.suffix != ".safetensors":
        target = target.with_suffix(".safetensors")
    return target, target.with_suffix(".json")


def _artifact_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _artifact_fingerprint(value: object) -> str:
    return hashlib.sha256(_artifact_json(value).encode("utf-8")).hexdigest()


def _store_artifact_tensor(tensors: dict[str, Tensor], value: Tensor, *, name: str) -> str:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a Tensor")
    key = f"tensor_{len(tensors):08d}"
    tensors[key] = value.detach().contiguous().cpu().clone()
    return key


def _load_artifact_tensor(tensors: Mapping[str, Tensor], value: object, *, name: str) -> Tensor:
    if not isinstance(value, str) or value not in tensors:
        raise ResourceGraphError(f"program graph artifact is missing tensor {name!r}")
    return tensors[value]


def _node_to_artifact(
    node: ProgramNode | MultiPortProgramNode,
    tensors: dict[str, Tensor],
    *,
    name: str,
) -> dict[str, object]:
    """Serialize a registered node's declaration and tensor state.

    Node code remains caller-owned.  The counterpart loader requires a typed
    node instance and verifies this declaration before loading its state.
    """

    try:
        from .component_registry import component_ref

        reference = component_ref(node)
    except Exception as error:
        raise ResourceGraphError(
            "program graph artifacts support only registered ProgramNode modules"
        ) from error
    state = {
        key: _store_artifact_tensor(tensors, value, name=f"{name}.state.{key}")
        for key, value in sorted(node.state_dict().items())
    }
    return {
        "node_id": node.node_id,
        "ref": reference,
        "contract": node.contract_config(),
        "state": state,
    }


def _load_node_state(
    node: ProgramNode | MultiPortProgramNode,
    payload: object,
    tensors: Mapping[str, Tensor],
    *,
    name: str,
    device: torch.device,
) -> None:
    if not isinstance(payload, Mapping):
        raise ResourceGraphError(f"program graph artifact has invalid node {name!r}")
    required = {"node_id", "ref", "contract", "state"}
    if set(payload) != required:
        raise ResourceGraphError(f"program graph artifact node {name!r} has invalid fields")
    if payload["node_id"] != node.node_id or payload["contract"] != node.contract_config():
        raise ResourceGraphError(f"program graph artifact node {name!r} does not match supplied node")
    try:
        from .component_registry import component_ref

        if payload["ref"] != component_ref(node):
            raise ResourceGraphError(f"program graph artifact node {name!r} has a different component ref")
    except ResourceGraphError:
        raise
    except Exception as error:
        raise ResourceGraphError(
            f"program graph artifact node {name!r} is not a registered ProgramNode"
        ) from error
    raw_state = payload["state"]
    if not isinstance(raw_state, Mapping):
        raise ResourceGraphError(f"program graph artifact node {name!r} has invalid state")
    node.to(device)
    state = {
        str(key): _load_artifact_tensor(tensors, value, name=f"{name}.state.{key}").to(device)
        for key, value in raw_state.items()
    }
    try:
        node.load_state_dict(state, strict=True)
    except RuntimeError as error:
        raise ResourceGraphError(f"program graph artifact node {name!r} state does not match") from error


def _view_to_artifact(
    view: TensorView, tensors: dict[str, Tensor], *, name: str
) -> dict[str, object]:
    index_map = view.index_map
    return {
        "value": _store_artifact_tensor(tensors, view.value, name=f"{name}.value"),
        "axes": [axis.to_dict() for axis in view.axes],
        "mask": None
        if view.mask is None
        else _store_artifact_tensor(tensors, view.mask, name=f"{name}.mask"),
        "index_map": None
        if index_map is None
        else {
            "source_axes": list(index_map.source_axes),
            "source_shape": list(index_map.source_shape),
            "target_shape": list(index_map.target_shape),
            "schema_version": index_map.schema_version,
            "coordinates": None
            if index_map.coordinates is None
            else _store_artifact_tensor(
                tensors, index_map.coordinates, name=f"{name}.index_map.coordinates"
            ),
        },
    }


def _view_from_artifact(
    value: object, tensors: Mapping[str, Tensor], *, name: str, device: torch.device
) -> TensorView:
    if not isinstance(value, Mapping) or set(value) != {"value", "axes", "mask", "index_map"}:
        raise ResourceGraphError(f"program graph artifact has invalid TensorView {name!r}")
    raw_axes = value["axes"]
    if not isinstance(raw_axes, list):
        raise ResourceGraphError(f"program graph artifact has invalid axes for {name!r}")
    index_payload = value["index_map"]
    if index_payload is None:
        index_map = None
    else:
        required = {"source_axes", "source_shape", "target_shape", "schema_version", "coordinates"}
        if not isinstance(index_payload, Mapping) or set(index_payload) != required:
            raise ResourceGraphError(f"program graph artifact has invalid index map for {name!r}")
        raw_coordinates = index_payload["coordinates"]
        coordinates = (
            None
            if raw_coordinates is None
            else _load_artifact_tensor(
                tensors, raw_coordinates, name=f"{name}.index_map.coordinates"
            ).to(device)
        )
        index_map = TensorIndexMap(
            tuple(index_payload["source_axes"]),
            tuple(index_payload["source_shape"]),
            tuple(index_payload["target_shape"]),
            coordinates,
            index_payload["schema_version"],
        )
    raw_mask = value["mask"]
    mask = None if raw_mask is None else _load_artifact_tensor(tensors, raw_mask, name=f"{name}.mask").to(device)
    return TensorView(
        _load_artifact_tensor(tensors, value["value"], name=f"{name}.value").to(device),
        tuple(AxisDescriptor.from_dict(item) for item in raw_axes),
        index_map=index_map,
        mask=mask,
    )


def _resource_view_to_artifact(value: ResourceView | None) -> dict[str, object] | None:
    return None if value is None else value.contract_config()


def _resource_view_from_artifact(value: object) -> ResourceView | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"resource_id", "ranges"}:
        raise ResourceGraphError("program graph artifact has invalid resource view")
    ranges = value["ranges"]
    if not isinstance(ranges, list):
        raise ResourceGraphError("program graph artifact resource-view ranges must be a list")
    return ResourceView(
        value["resource_id"],
        tuple(AxisRange(item["axis"], item["start"], item["stop"], item["step"]) for item in ranges),
    )


def _transfer_to_artifact(
    transfer: Callable[[TensorView], TensorView] | nn.Module | None,
    tensors: dict[str, Tensor],
    *,
    name: str,
) -> dict[str, object]:
    if transfer is None:
        return {"kind": "identity"}
    if isinstance(transfer, LearnableAffineTransfer):
        return {
            "kind": "learnable-affine",
            "learnable": transfer.learnable,
            "gain": _store_artifact_tensor(tensors, transfer.gain, name=f"{name}.gain"),
            "bias": _store_artifact_tensor(tensors, transfer.bias, name=f"{name}.bias"),
        }
    if isinstance(transfer, FormulaTensorViewTransfer):
        banks: dict[str, object] = {}
        for bank_name, operand in sorted(transfer.banks.items()):
            banks[bank_name] = {
                "value": _store_artifact_tensor(tensors, operand.value, name=f"{name}.banks.{bank_name}"),
                "source_ref": operand.source_ref,
                "partition_id": operand.partition_id,
                "asset_fingerprint": operand.asset_fingerprint,
                "route_ref": operand.route_ref,
                "bundle_id": operand.bundle_id,
                "member_ids": list(operand.member_ids),
            }
        return {
            "kind": "formula-v2",
            "program": transfer.fabric.program.to_dict(),
            "source_input": transfer.source_input,
            "output_name": transfer.output_name,
            "static_inputs": {
                key: _store_artifact_tensor(tensors, item, name=f"{name}.static_inputs.{key}")
                for key, item in sorted(transfer.static_inputs.items())
            },
            "dynamic_inputs": list(transfer.dynamic_inputs),
            "banks": banks,
            "axis_roles": dict(sorted(transfer.axis_roles.items())),
        }
    raise ResourceGraphError(
        "program graph artifact only supports identity, LearnableAffineTransfer, and "
        "FormulaTensorViewTransfer connections; arbitrary Python callables remain runtime-only"
    )


def _transfer_from_artifact(
    value: object, tensors: Mapping[str, Tensor], *, name: str, device: torch.device
) -> Callable[[TensorView], TensorView] | nn.Module | None:
    if not isinstance(value, Mapping) or not isinstance(value.get("kind"), str):
        raise ResourceGraphError(f"program graph artifact has invalid transfer {name!r}")
    kind = value["kind"]
    if kind == "identity":
        if set(value) != {"kind"}:
            raise ResourceGraphError("identity transfer has unexpected fields")
        return None
    if kind == "learnable-affine":
        if set(value) != {"kind", "learnable", "gain", "bias"}:
            raise ResourceGraphError("learnable-affine transfer has unexpected fields")
        transfer = LearnableAffineTransfer(learnable=value["learnable"])
        transfer.gain.data.copy_(_load_artifact_tensor(tensors, value["gain"], name=f"{name}.gain").to(device))
        transfer.bias.data.copy_(_load_artifact_tensor(tensors, value["bias"], name=f"{name}.bias").to(device))
        return transfer.to(device)
    if kind != "formula-v2":
        raise ResourceGraphError(f"program graph artifact does not support transfer kind {kind!r}")
    required = {
        "kind", "program", "source_input", "output_name", "static_inputs", "dynamic_inputs", "banks", "axis_roles"
    }
    if set(value) != required:
        raise ResourceGraphError("formula transfer has missing or unknown fields")
    static_payload = value["static_inputs"]
    bank_payload = value["banks"]
    if not isinstance(static_payload, Mapping) or not isinstance(bank_payload, Mapping):
        raise ResourceGraphError("formula transfer bindings must be mappings")
    static = {
        str(key): _load_artifact_tensor(tensors, item, name=f"{name}.static_inputs.{key}").to(device)
        for key, item in static_payload.items()
    }
    banks: dict[str, FormulaBankOperand] = {}
    for bank_name, raw in bank_payload.items():
        required_bank = {
            "value", "source_ref", "partition_id", "asset_fingerprint", "route_ref", "bundle_id", "member_ids"
        }
        if not isinstance(raw, Mapping) or set(raw) != required_bank:
            raise ResourceGraphError(f"formula transfer Bank {bank_name!r} has invalid fields")
        banks[str(bank_name)] = FormulaBankOperand(
            _load_artifact_tensor(tensors, raw["value"], name=f"{name}.banks.{bank_name}").to(device),
            raw["source_ref"], raw["partition_id"], raw["asset_fingerprint"], raw["route_ref"],
            raw["bundle_id"], tuple(raw["member_ids"]),
        )
    fabric = FormulaFabricV2(FormulaProgram.from_dict(value["program"]))
    return FormulaTensorViewTransfer(
        fabric,
        source_input=value["source_input"],
        output_name=value["output_name"],
        static_inputs=static,
        dynamic_inputs=tuple(value["dynamic_inputs"]),
        banks=banks,
        axis_roles=value["axis_roles"],
    )


def save_program_graph(graph: ProgramGraph, path: str | Path) -> ProgramGraphSaveResult:
    """Save one portable ProgramGraph declaration and its current resource state.

    The graph artifact is deliberately separate from :func:`arti.save`: it stores
    resource backings and only accepts native, reconstructible connection laws.
    Runtime callables and local activation functions are rejected rather than
    being serialized as import-path guesses.
    """

    if not isinstance(graph, ProgramGraph):
        raise TypeError("graph must be ProgramGraph")
    tensors: dict[str, Tensor] = {}
    resources = []
    for resource_id, resource in graph.resources.items():
        state = resource.state()
        resources.append(
            {
                "spec": state.spec.contract_config(),
                "default_view": _view_to_artifact(state.default_view, tensors, name=f"resources.{resource_id}.default"),
                "active_view": _view_to_artifact(state.active_view, tensors, name=f"resources.{resource_id}.active"),
                "active_source": state.active_source,
                "epoch": state.epoch,
                "step_index": state.step_index,
            }
        )
    connections = []
    for connection_id, connection in sorted(graph.connections.items()):
        if connection.activation is not None:
            raise ResourceGraphError(
                "program graph artifacts cannot persist conditional activation callables; "
                "use a native Formula transfer or restore the runtime graph explicitly"
            )
        connections.append(
            {
                "connection_id": connection_id,
                "source": connection.source.contract_config(),
                "destination": connection.destination.contract_config(),
                "source_view": _resource_view_to_artifact(connection.source_view),
                "destination_view": _resource_view_to_artifact(connection.destination_view),
                "operand_views": {
                    key: _resource_view_to_artifact(item)
                    for key, item in sorted(connection.operand_views.items())
                },
                "depends_on": list(connection.depends_on),
                "transfer": _transfer_to_artifact(connection.transfer, tensors, name=f"connections.{connection_id}.transfer"),
            }
        )
    nodes = [
        _node_to_artifact(graph.nodes[node_id], tensors, name=f"nodes.{node_id}")
        for node_id in sorted(graph.nodes)
    ]
    state = graph.state()
    payload = {
        "format": PROGRAM_GRAPH_ARTIFACT_FORMAT,
        "version": PROGRAM_GRAPH_ARTIFACT_VERSION,
        "contract_fingerprint": graph.contract_fingerprint,
        "resources": resources,
        "connections": connections,
        "nodes": nodes,
        "programs": {
            program_id: [
                step
                if isinstance(step, str)
                else {"stage": list(step.step_ids)}
                if isinstance(step, ProgramStage)
                else step.contract_config()
                for step in steps
            ]
            for program_id, steps in sorted(graph._programs.items())
        },
        "loops": [graph._loops[loop_id].contract_config() for loop_id in sorted(graph._loops)],
        "join_cursors": [
            {"join_id": join_id, "ports": {name: epoch for name, epoch in ports}}
            for join_id, ports in state.join_cursors
        ],
    }
    manifest = {**payload, "manifest_sha256": _artifact_fingerprint(payload)}
    tensors_path, manifest_path = _artifact_paths(path)
    tensors_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(tensors_path), metadata={"format": PROGRAM_GRAPH_ARTIFACT_FORMAT, "version": str(PROGRAM_GRAPH_ARTIFACT_VERSION)})
    manifest_path.write_text(_artifact_json(manifest), encoding="utf-8")
    return ProgramGraphSaveResult(tensors_path, manifest_path, graph.contract_fingerprint, manifest["manifest_sha256"])


def load_program_graph(
    path: str | Path,
    *,
    nodes: Sequence[ProgramNode | MultiPortProgramNode] = (),
    map_location: str | torch.device = "cpu",
) -> ProgramGraph:
    """Load a graph artifact into fresh resources and supplied typed nodes.

    Direct-only graphs require no ``nodes``.  A graph containing executable
    regions requires matching ``ProgramNode`` modules from the caller; the
    artifact verifies their component contract before restoring tensor state.
    """

    tensors_path, manifest_path = _artifact_paths(path)
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ResourceGraphError(f"cannot read program graph manifest {manifest_path}") from error
    if not isinstance(manifest, Mapping):
        raise ResourceGraphError("program graph manifest must be an object")
    required = {
        "format",
        "version",
        "contract_fingerprint",
        "resources",
        "connections",
        "nodes",
        "programs",
        "loops",
        "join_cursors",
        "manifest_sha256",
    }
    if set(manifest) != required or manifest["format"] != PROGRAM_GRAPH_ARTIFACT_FORMAT or manifest["version"] != PROGRAM_GRAPH_ARTIFACT_VERSION:
        raise ResourceGraphError("unsupported program graph artifact")
    payload = {key: manifest[key] for key in required - {"manifest_sha256"}}
    if manifest["manifest_sha256"] != _artifact_fingerprint(payload):
        raise ResourceGraphError("program graph manifest fingerprint is invalid")
    try:
        tensors = load_file(str(tensors_path), device=str(torch.device(map_location)))
    except OSError as error:
        raise ResourceGraphError(f"cannot read program graph tensors {tensors_path}") from error
    device = torch.device(map_location)
    raw_resources = manifest["resources"]
    raw_connections = manifest["connections"]
    raw_nodes = manifest["nodes"]
    if (
        not isinstance(raw_resources, list)
        or not isinstance(raw_connections, list)
        or not isinstance(raw_nodes, list)
        or not isinstance(manifest["programs"], Mapping)
        or not isinstance(manifest["loops"], list)
        or not isinstance(manifest["join_cursors"], list)
    ):
        raise ResourceGraphError("program graph manifest has invalid collections")
    supplied_nodes = tuple(nodes)
    if any(not isinstance(node, (ProgramNode, MultiPortProgramNode)) for node in supplied_nodes):
        raise TypeError("nodes must contain ProgramNode or MultiPortProgramNode values")
    supplied_by_id = {node.node_id: node for node in supplied_nodes}
    if len(supplied_by_id) != len(supplied_nodes):
        raise ResourceGraphError("supplied ProgramNode ids must be unique")
    artifact_node_ids = tuple(
        item.get("node_id") if isinstance(item, Mapping) else None for item in raw_nodes
    )
    if set(artifact_node_ids) != set(supplied_by_id) or len(artifact_node_ids) != len(supplied_by_id):
        raise ResourceGraphError("program graph artifact nodes do not match supplied nodes")
    for index, raw in enumerate(raw_nodes):
        assert isinstance(raw, Mapping)
        node_id = raw["node_id"]
        assert isinstance(node_id, str)
        _load_node_state(
            supplied_by_id[node_id],
            raw,
            tensors,
            name=f"nodes[{index}]",
            device=device,
        )
    resources: list[TensorResource] = []
    for index, raw in enumerate(raw_resources):
        required_resource = {"spec", "default_view", "active_view", "active_source", "epoch", "step_index"}
        if not isinstance(raw, Mapping) or set(raw) != required_resource:
            raise ResourceGraphError(f"resource entry {index} has invalid fields")
        spec_payload = raw["spec"]
        if not isinstance(spec_payload, Mapping):
            raise ResourceGraphError(f"resource entry {index} has invalid specification")
        spec = TensorResourceSpec(
            spec_payload["resource_id"],
            TensorViewPattern.from_dict(spec_payload["view_pattern"]),
            ResourceLifetime(spec_payload["lifetime"]),
            tuple((str(axis), int(limit)) for axis, limit in spec_payload["axis_capacity"]),
        )
        state = TensorResourceState(
            spec,
            _view_from_artifact(raw["default_view"], tensors, name=f"resources[{index}].default", device=device),
            _view_from_artifact(raw["active_view"], tensors, name=f"resources[{index}].active", device=device),
            raw["active_source"], raw["epoch"], raw["step_index"],
        )
        resources.append(TensorResource.restore(state))
    connections: list[Connection] = []
    for index, raw in enumerate(raw_connections):
        required_connection = {"connection_id", "source", "destination", "source_view", "destination_view", "operand_views", "depends_on", "transfer"}
        if not isinstance(raw, Mapping) or set(raw) != required_connection:
            raise ResourceGraphError(f"connection entry {index} has invalid fields")
        source = raw["source"]
        destination = raw["destination"]
        operand_views = raw["operand_views"]
        if not isinstance(source, Mapping) or not isinstance(destination, Mapping) or not isinstance(operand_views, Mapping):
            raise ResourceGraphError(f"connection entry {index} has invalid ports or operands")
        connections.append(
            Connection(
                raw["connection_id"],
                ResourcePort(source["resource_id"], source["port"]),
                ResourcePort(destination["resource_id"], destination["port"]),
                source_view=_resource_view_from_artifact(raw["source_view"]),
                destination_view=_resource_view_from_artifact(raw["destination_view"]),
                operand_views={key: _resource_view_from_artifact(item) for key, item in operand_views.items()},
                depends_on=tuple(raw["depends_on"]),
                transfer=_transfer_from_artifact(raw["transfer"], tensors, name=f"connections[{index}].transfer", device=device),
            )
        )
    loops: list[ProgramLoop] = []
    for index, raw in enumerate(manifest["loops"]):
        required_loop = {
            "loop_id",
            "program_id",
            "continue_resource_id",
            "max_iterations",
            "min_iterations",
        }
        if not isinstance(raw, Mapping) or set(raw) != required_loop:
            raise ResourceGraphError(f"loop entry {index} has invalid fields")
        loops.append(
            ProgramLoop(
                raw["loop_id"],
                raw["program_id"],
                raw["continue_resource_id"],
                raw["max_iterations"],
                raw["min_iterations"],
            )
        )
    programs: dict[str, tuple[str | ProgramStage | ProgramJoin | ProgramRoute, ...]] = {}
    for program_id, raw_steps in manifest["programs"].items():
        if not isinstance(program_id, str) or not isinstance(raw_steps, list):
            raise ResourceGraphError("program graph artifact programs are invalid")
        steps: list[str | ProgramStage | ProgramJoin | ProgramRoute] = []
        for raw_step in raw_steps:
            if isinstance(raw_step, str):
                steps.append(raw_step)
            elif isinstance(raw_step, Mapping) and set(raw_step) == {"stage"} and isinstance(raw_step["stage"], list):
                steps.append(ProgramStage(tuple(raw_step["stage"])))
            elif isinstance(raw_step, Mapping) and set(raw_step) == {"join"}:
                raw_join = raw_step["join"]
                if (
                    not isinstance(raw_join, Mapping)
                    or set(raw_join) != {"join_id", "node_id", "firing"}
                    or raw_join["firing"] != "all_new"
                ):
                    raise ResourceGraphError("program graph artifact join is invalid")
                steps.append(ProgramJoin(raw_join["join_id"], raw_join["node_id"]))
            elif isinstance(raw_step, Mapping) and set(raw_step) == {"route"}:
                raw_route = raw_step["route"]
                if (
                    not isinstance(raw_route, Mapping)
                    or not {"route_id", "score_resource_id", "candidates"} <= set(raw_route)
                    <= {"route_id", "score_resource_id", "candidates", "selection_scope", "execution_mode"}
                    or not isinstance(raw_route["candidates"], list)
                ):
                    raise ResourceGraphError("program graph artifact route is invalid")
                steps.append(ProgramRoute(
                    raw_route["route_id"], raw_route["score_resource_id"],
                    tuple(raw_route["candidates"]),
                    raw_route.get("selection_scope", "batch"),
                    raw_route.get("execution_mode", "sparse"),
                ))
            else:
                raise ResourceGraphError("program graph artifact step is invalid")
        programs[program_id] = tuple(steps)
    graph = ProgramGraph(resources, connections, nodes=supplied_nodes, programs=programs, loops=loops)
    cursor_entries: list[tuple[str, tuple[tuple[str, int], ...]]] = []
    for raw_cursor in manifest["join_cursors"]:
        if (
            not isinstance(raw_cursor, Mapping)
            or set(raw_cursor) != {"join_id", "ports"}
            or not isinstance(raw_cursor["join_id"], str)
            or not isinstance(raw_cursor["ports"], Mapping)
            or any(not isinstance(name, str) or type(epoch) is not int for name, epoch in raw_cursor["ports"].items())
        ):
            raise ResourceGraphError("program graph artifact join cursor is invalid")
        cursor_entries.append((raw_cursor["join_id"], tuple(sorted(raw_cursor["ports"].items()))))
    if len({join_id for join_id, _ in cursor_entries}) != len(cursor_entries):
        raise ResourceGraphError("program graph artifact join cursor ids must be unique")
    graph.restore_state(
        ProgramGraphState(
            graph.contract_fingerprint,
            tuple(resource.state() for resource in resources),
            tuple(sorted(cursor_entries)),
        )
    )
    if graph.contract_fingerprint != manifest["contract_fingerprint"]:
        raise ResourceGraphError("program graph artifact contract does not match reconstructed graph")
    return graph


__all__ = [
    "PROGRAM_GRAPH_ARTIFACT_FORMAT",
    "PROGRAM_GRAPH_ARTIFACT_VERSION",
    "Connection",
    "ConnectionExecution",
    "FormulaTensorViewTransfer",
    "FormulaProgramNode",
    "DifferentiableFabricNode",
    "DifferentiableFateLosses",
    "FabricNodeSpecialization",
    "StaticFormulaProgramNode",
    "StaticProgramGraphCreditResult",
    "StaticProgramJoinCreditResult",
    "StaticProgramLoopCreditResult",
    "StaticDataflowProgramCreditResult",
    "StaticDataflowLoopCreditResult",
    "StaticFormulaTensorViewTransfer",
    "ResourceGraphCompileError",
    "LocalVJP",
    "LocalVJPResult",
    "autograd_local_vjp",
    "ResourceGraphCompiler",
    "ResourceGraphExecutionPlan",
    "StaticProgramGraphExecutionPlan",
    "StaticRoutedProgramExecutionPlan",
    "StaticProgramRouteStep",
    "StaticDataflowProgramExecutionPlan",
    "StaticProgramOutputExecutionPlan",
    "StaticDataflowLoopExecutionPlan",
    "StaticProgramJoinExecutionPlan",
    "StaticProgramLoopExecutionPlan",
    "CapturedResourceGraphExecutionPlan",
    "AxisRange",
    "ProgramGraph",
    "ProgramGraphExecution",
    "ProgramDispatch",
    "ProgramGraphSpecialization",
    "ProgramFateSample",
    "ProgramGraphInvocation",
    "ProgramStage",
    "ProgramRoute",
    "ProgramRouteExecution",
    "ProgramJoin",
    "ProgramJoinExecution",
    "ProgramLoop",
    "ProgramLoopExecution",
    "ProgramGraphSaveResult",
    "ProgramGraphState",
    "ProgramNode",
    "ProgramNodeExecution",
    "ProgramNodeInvocation",
    "MultiPortProgramNode",
    "MultiPortProgramNodeExecution",
    "MultiPortProgramNodeInvocation",
    "ResourceBinding",
    "ResourceGraphError",
    "ResourceLifetime",
    "ResourcePort",
    "ResourceSnapshot",
    "ResourceView",
    "TensorResource",
    "TensorResourceSpec",
    "TensorResourceState",
    "load_program_graph",
    "save_program_graph",
]
