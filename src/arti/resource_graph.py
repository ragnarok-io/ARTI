"""Explicit tensor resources and learnable program connections.

This module deliberately separates where a tensor lives from how a Formula or
ordinary program operates on it.  A resource carries no mandatory Query or
edit loop; a connection can be direct or use a caller-supplied local condition.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
import re
from typing import ClassVar, Literal

import torch
from torch import Tensor, nn
from safetensors.torch import load_file, save_file

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
    def restore(cls, state: TensorResourceState) -> TensorResource:
        if not isinstance(state, TensorResourceState):
            raise TypeError("state must be TensorResourceState")
        state.spec.validate(state.default_view, name="state.default_view")
        state.spec.validate(state.active_view, name="state.active_view")
        resource = cls(state.spec, state.default_view)
        if state.active_source == "external":
            resource._external_view = _clone_view(state.active_view)
        elif state.active_source != "default":
            raise ResourceGraphError("state active_source is invalid")
        elif tuple(state.active_view.value.shape) != tuple(resource._default_view.value.shape) or not torch.equal(
            state.active_view.value, resource._default_view.value
        ):
            resource._default_view = _clone_view(state.active_view)
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
        self.connection_id = declared_id
        self.source = source
        self.destination = destination
        self.source_view = source_view
        self.destination_view = destination_view
        self.operand_views = operands
        self.depends_on = dependencies
        self.transfer = transfer
        self.activation = activation

    @property
    def is_conditional(self) -> bool:
        return self.activation is not None

    def forward(
        self,
        source: TensorView,
        *,
        operands: Mapping[str, TensorView] | None = None,
        context: Tensor | None = None,
    ) -> TensorView:
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
            return output
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
        return TensorView(value, output.axes, index_map=output.index_map, mask=output.mask)

    def apply(
        self,
        source_snapshot: ResourceSnapshot,
        destination_snapshot: ResourceSnapshot,
        *,
        resources: Mapping[str, ResourceSnapshot] | None = None,
        context: Tensor | None = None,
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
        output = self(source, operands=operands, context=context)
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
    """One completed connection application and its destination snapshot."""

    connection_id: str
    source: ResourceBinding
    destination: ResourceBinding


@dataclass(frozen=True)
class ProgramNodeExecution:
    """One program-node call committed into a functional graph state."""

    node_id: str
    source: ResourceBinding
    destination: ResourceBinding
    receipt: object | None = None


@dataclass(frozen=True)
class ProgramNodeInvocation:
    """A typed program-node result before the graph publishes its output."""

    output: TensorView
    receipt: object | None = None


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


@dataclass(frozen=True)
class ProgramGraphState:
    """A graph-state receipt that keeps mutable backings separate from modules."""

    contract_fingerprint: str
    resources: tuple[TensorResourceState, ...]


@dataclass(frozen=True)
class ProgramGraphExecution:
    """One functional graph transition and the exact resulting resource state."""

    state: ProgramGraphState
    connections: tuple[ConnectionExecution, ...]
    nodes: tuple[ProgramNodeExecution, ...] = ()


@dataclass(frozen=True)
class ProgramGraphInvocation:
    """A branch-local subprogram call with an explicitly declared view result."""

    output: TensorView
    state: ProgramGraphState
    connections: tuple[ConnectionExecution, ...]
    nodes: tuple[ProgramNodeExecution, ...] = ()


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
        programs: Mapping[str, Sequence[str]] | None = None,
        nodes: Sequence[ProgramNode] = (),
    ) -> None:
        super().__init__()
        resources = tuple(resources)
        connections = tuple(connections)
        nodes = tuple(nodes)
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
        if len(set(node_ids)) != len(node_ids):
            raise ResourceGraphError("program-node ids must be unique")
        if set(connection_ids).intersection(node_ids):
            raise ResourceGraphError("program-node ids must not collide with connection ids")
        if any(not isinstance(node, ProgramNode) for node in nodes):
            raise TypeError("nodes must contain ProgramNode values")
        known = set(ids)
        for connection in connections:
            if connection.source.resource_id not in known or connection.destination.resource_id not in known:
                raise ResourceGraphError("connection ports must reference graph resources")
            if any(view.resource_id not in known for view in connection.operand_views.values()):
                raise ResourceGraphError("connection operand views must reference graph resources")
            if any(dependency not in connection_ids for dependency in connection.depends_on):
                raise ResourceGraphError("connection dependencies must reference graph connections")
        for node in nodes:
            if node.input_resource_id not in known or node.output_resource_id not in known:
                raise ResourceGraphError("program-node ports must reference graph resources")
        _validate_connection_dependencies(connections)
        program_mapping = {} if programs is None else dict(programs)
        if any(not isinstance(program_id, str) for program_id in program_mapping):
            raise TypeError("program ids must be strings")
        normalized_programs: dict[str, tuple[str, ...]] = {}
        for program_id, sequence in program_mapping.items():
            _require_identifier(program_id, field="program_id")
            step_sequence = tuple(sequence)
            known_steps = set(connection_ids).union(node_ids)
            if not step_sequence or any(step_id not in known_steps for step_id in step_sequence):
                raise ResourceGraphError("programs must contain declared connection or node ids")
            normalized_programs[program_id] = step_sequence
        self._resources = {resource.spec.resource_id: resource for resource in resources}
        self.connections = nn.ModuleDict({connection.connection_id: connection for connection in connections})
        self.nodes = nn.ModuleDict({node.node_id: node for node in nodes})
        self._programs = normalized_programs

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
                program_id: list(self._programs[program_id]) for program_id in sorted(self._programs)
            },
        }

    @property
    def contract_fingerprint(self) -> str:
        payload = json.dumps(
            self.contract_config(), sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def state(self) -> ProgramGraphState:
        """Capture mutable resource backing without serializing operation modules."""

        return self._state_from_resources(self._resources)

    def _state_from_resources(
        self, resources: Mapping[str, TensorResource]
    ) -> ProgramGraphState:
        return ProgramGraphState(
            self.contract_fingerprint,
            tuple(resources[resource_id].state() for resource_id in sorted(resources)),
        )

    def _restore_resources(
        self, state: ProgramGraphState
    ) -> dict[str, TensorResource]:
        if not isinstance(state, ProgramGraphState):
            raise TypeError("state must be ProgramGraphState")
        if state.contract_fingerprint != self.contract_fingerprint:
            raise ResourceGraphError("resource graph state does not match this graph contract")
        restored = {item.spec.resource_id: TensorResource.restore(item) for item in state.resources}
        if set(restored) != set(self._resources):
            raise ResourceGraphError("resource graph state does not declare this graph's resources")
        return restored

    def restore_state(self, state: ProgramGraphState) -> None:
        """Restore compatible resource state into this graph's declared relations."""

        self._resources = self._restore_resources(state)

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

    def node(self, node_id: str) -> ProgramNode:
        try:
            return self.nodes[node_id]
        except KeyError as error:
            raise ResourceGraphError(f"unknown program node {node_id!r}") from error

    def _connection_sequence(self, connection_ids: Sequence[str]) -> tuple[Connection, ...]:
        """Resolve a serial edge sequence while enforcing declared dependencies."""

        connections = tuple(self.connection(connection_id) for connection_id in connection_ids)
        completed: set[str] = set()
        for connection in connections:
            missing = set(connection.depends_on) - completed
            if missing:
                raise ResourceGraphError(
                    f"connection {connection.connection_id!r} requires prior dependencies "
                    f"{sorted(missing)!r}"
                )
            completed.add(connection.connection_id)
        return connections

    def _program_steps(self, step_ids: Sequence[str]) -> tuple[Connection | ProgramNode, ...]:
        """Resolve a declared mixed sequence and enforce edge dependencies."""

        steps: list[Connection | ProgramNode] = []
        completed: set[str] = set()
        for step_id in step_ids:
            if step_id in self.connections:
                step: Connection | ProgramNode = self.connection(step_id)
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

    def program(self, program_id: str) -> tuple[str, ...]:
        try:
            return self._programs[program_id]
        except KeyError as error:
            raise ResourceGraphError(f"unknown resource program {program_id!r}") from error

    def execute_program(
        self,
        program_id: str,
        *,
        contexts: Mapping[str, Tensor] | None = None,
    ) -> tuple[ConnectionExecution, ...]:
        """Call a direct-only declared subprogram; ordered edges observe writes.

        Existing imperative callers retain their compact connection receipt.
        Programs containing mounted nodes use :meth:`execute_program_functional`
        so their state transition remains explicit.
        """

        step_ids = self.program(program_id)
        if any(step_id in self.nodes for step_id in step_ids):
            raise ResourceGraphError(
                "imperative execute_program does not publish ProgramNode state; "
                "use execute_program_functional"
            )
        return self.execute(step_ids, contexts=contexts)

    def execute_program_functional(
        self,
        program_id: str,
        *,
        state: ProgramGraphState | None = None,
        input_views: Mapping[str, TensorView] | None = None,
        contexts: Mapping[str, Tensor] | None = None,
    ) -> ProgramGraphExecution:
        """Run declared direct edges and mounted regions against one state value."""

        return self._execute_steps_functional(
            self.program(program_id),
            state=state,
            input_views=input_views,
            contexts=contexts,
        )

    def execute_repeated(
        self,
        program_id: str,
        *,
        iterations: int,
        contexts: Mapping[str, Tensor] | None = None,
    ) -> tuple[tuple[ConnectionExecution, ...], ...]:
        """Execute a fixed number of calls without redefining event semantics.

        Data-dependent exit remains a Formula/program concern.  This helper
        only makes a statically declared local loop explicit for reference
        execution and later fixed-horizon lowering.
        """

        if type(iterations) is not int or iterations <= 0:
            raise ResourceGraphError("iterations must be a positive integer")
        return tuple(
            self.execute_program(program_id, contexts=contexts) for _ in range(iterations)
        )

    def execute(
        self,
        connection_ids: Sequence[str],
        *,
        contexts: Mapping[str, Tensor] | None = None,
    ) -> tuple[ConnectionExecution, ...]:
        """Run a declared serial relation sequence with explicit state visibility."""

        contexts = {} if contexts is None else contexts
        if not isinstance(contexts, Mapping):
            raise TypeError("contexts must be a mapping or None")
        result: list[ConnectionExecution] = []
        for connection in self._connection_sequence(connection_ids):
            source_snapshot = self.resource(connection.source.resource_id).resolve()
            destination = self.resource(connection.destination.resource_id)
            destination_entry = destination.resolve()
            resources = {resource_id: resource.resolve() for resource_id, resource in self._resources.items()}
            output = connection.apply(
                source_snapshot,
                destination_entry,
                resources=resources,
                context=contexts.get(connection.connection_id),
            )
            destination_snapshot = destination.advance(output)
            result.append(
                ConnectionExecution(
                    connection.connection_id,
                    source_snapshot.binding,
                    destination_snapshot.binding,
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
            direct_only=True,
        )

    def _execute_steps_functional(
        self,
        step_ids: Sequence[str],
        *,
        state: ProgramGraphState | None,
        input_views: Mapping[str, TensorView] | None,
        contexts: Mapping[str, Tensor] | None,
        direct_only: bool = False,
    ) -> ProgramGraphExecution:
        """Evaluate graph steps without mutating live resources.

        ``direct_only`` protects the historical connection-only API.  Declared
        programs call the same executor without that restriction, so direct
        edges and local routed regions observe exactly one evolving snapshot.
        """

        contexts = {} if contexts is None else contexts
        input_views = {} if input_views is None else input_views
        if not isinstance(contexts, Mapping) or not isinstance(input_views, Mapping):
            raise TypeError("contexts and input_views must be mappings or None")
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
        if direct_only:
            steps: tuple[Connection | ProgramNode, ...] = self._connection_sequence(step_ids)
        else:
            steps = self._program_steps(step_ids)
        connections: list[ConnectionExecution] = []
        nodes: list[ProgramNodeExecution] = []
        for step in steps:
            if isinstance(step, Connection):
                source_snapshot = resources[step.source.resource_id].resolve()
                destination = resources[step.destination.resource_id]
                destination_entry = destination.resolve()
                snapshots = {
                    resource_id: resource.resolve() for resource_id, resource in resources.items()
                }
                output = step.apply(
                    source_snapshot,
                    destination_entry,
                    resources=snapshots,
                    context=contexts.get(step.connection_id),
                )
                destination_snapshot = destination.advance(output)
                connections.append(
                    ConnectionExecution(
                        step.connection_id,
                        source_snapshot.binding,
                        destination_snapshot.binding,
                    )
                )
                continue

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
        return ProgramGraphExecution(
            self._state_from_resources(resources), tuple(connections), tuple(nodes)
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
        merge: Mapping[str, Callable[[TensorView, tuple[TensorView, ...]], TensorView]] | None = None,
    ) -> tuple[ConnectionExecution, ...]:
        """Run connections from a common resource snapshot, then publish outputs.

        Multiple writes to one destination require an explicit merge operation.
        This preserves the distinction between cooperative parallel branches and
        an accidental sequential write ordering.
        """

        contexts = {} if contexts is None else contexts
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
        for connection in connections:
            source_snapshot = snapshots[connection.source.resource_id]
            destination_snapshot = snapshots[connection.destination.resource_id]
            output = connection.apply(
                source_snapshot,
                destination_snapshot,
                resources=snapshots,
                context=contexts.get(connection.connection_id),
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
        merge: Mapping[str, Callable[[TensorView, tuple[TensorView, ...]], TensorView]] | None = None,
    ) -> ProgramGraphExecution:
        """Evaluate one common-snapshot parallel frontier without live mutation.

        The returned state is a branch proposal.  As with :meth:`execute_parallel`,
        concurrent writes require a declared merge instead of inheriting an
        accidental Python or kernel execution order.
        """

        contexts = {} if contexts is None else contexts
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
        for connection in connections:
            output = connection.apply(
                snapshots[connection.source.resource_id],
                snapshots[connection.destination.resource_id],
                resources=snapshots,
                context=contexts.get(connection.connection_id),
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
        self.connections = nn.ModuleList(deepcopy(tuple(connections)))
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

    @property
    def context_connection_ids(self) -> tuple[str, ...]:
        return tuple(self.connections[index].connection_id for index in self._conditional_indices)

    def forward(self, *inputs: Tensor) -> tuple[Tensor, ...]:
        expected = len(self.resource_ids) + len(self._conditional_indices)
        if len(inputs) != expected:
            raise ResourceGraphCompileError("compiled plan received an incorrect resource/context value count")
        values = inputs[:len(self.resource_ids)]
        contexts = inputs[len(self.resource_ids):]
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
        ) in zip(
            self.connections,
            self._source_indices,
            self._destination_indices,
            self._source_templates,
            self._source_slices,
            self._destination_slices,
            self._operand_specs,
            self._context_positions,
            strict=True,
        ):
            context = None if context_position < 0 else contexts[context_position]
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
            output = connection(source_view, operands=operands, context=context)
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


class ResourceGraphCompiler:
    """Lower static direct connections while leaving dynamic relation semantics explicit."""

    @classmethod
    def compile_program(
        cls,
        graph: ProgramGraph,
        program_id: str,
        *,
        iterations: int = 1,
        example_contexts: Mapping[str, Tensor] | None = None,
    ) -> ResourceGraphExecutionPlan:
        """Lower a declared connection program with a fixed local iteration count.

        ``iterations`` is a compile-time horizon, not an external event count.
        The resulting plan carries each iteration's resource state directly in
        its tensor values, so it has no Python loop during invocation.
        """

        if not isinstance(graph, ProgramGraph):
            raise TypeError("graph must be ProgramGraph")
        if type(iterations) is not int or iterations <= 0:
            raise ResourceGraphCompileError("iterations must be a positive integer")
        connections = graph.program(program_id)
        if any(step_id in graph.nodes for step_id in connections):
            raise ResourceGraphCompileError(
                "fixed connection lowering does not yet lower mounted ProgramNode regions"
            )
        return cls.compile(
            graph,
            connections * iterations,
            example_contexts=example_contexts,
        )

    @classmethod
    def compile(
        cls,
        graph: ProgramGraph,
        connection_ids: Sequence[str],
        *,
        example_contexts: Mapping[str, Tensor] | None = None,
    ) -> ResourceGraphExecutionPlan:
        if not isinstance(graph, ProgramGraph):
            raise TypeError("graph must be ProgramGraph")
        connection_ids = tuple(connection_ids)
        if not connection_ids:
            raise ResourceGraphCompileError("compiled graph fragments require at least one connection")
        connections = graph._connection_sequence(connection_ids)
        example_contexts = {} if example_contexts is None else dict(example_contexts)
        conditional_ids = {connection.connection_id for connection in connections if connection.is_conditional}
        if set(example_contexts) != conditional_ids:
            raise ResourceGraphCompileError(
                "example_contexts must bind every conditional connection and no direct connection"
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
                )
                destination = graph.resource(connection.destination.resource_id)
                destination.spec.validate(
                    output, name=f"compiled {destination.spec.resource_id}"
                )
                prototype_views[connection.destination.resource_id] = output
        lowered_connections = tuple(deepcopy(connection) for connection in connections)
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
            )
        for resource, value in zip(resources, outputs, strict=True):
            template = resource.resolve().view
            resource.spec.validate(
                TensorView(value, template.axes, index_map=template.index_map, mask=template.mask),
                name=f"compiled {resource.spec.resource_id}",
            )
        return plan


PROGRAM_GRAPH_ARTIFACT_FORMAT = "arti.program-graph"
PROGRAM_GRAPH_ARTIFACT_VERSION = 2


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
    tensors[key] = value.detach().contiguous().cpu()
    return key


def _load_artifact_tensor(tensors: Mapping[str, Tensor], value: object, *, name: str) -> Tensor:
    if not isinstance(value, str) or value not in tensors:
        raise ResourceGraphError(f"program graph artifact is missing tensor {name!r}")
    return tensors[value]


def _node_to_artifact(
    node: ProgramNode,
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
    node: ProgramNode,
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
    for resource_id, resource in sorted(graph.resources.items()):
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
    payload = {
        "format": PROGRAM_GRAPH_ARTIFACT_FORMAT,
        "version": PROGRAM_GRAPH_ARTIFACT_VERSION,
        "contract_fingerprint": graph.contract_fingerprint,
        "resources": resources,
        "connections": connections,
        "nodes": nodes,
        "programs": {
            program_id: list(connection_ids)
            for program_id, connection_ids in sorted(graph._programs.items())
        },
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
    nodes: Sequence[ProgramNode] = (),
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
    ):
        raise ResourceGraphError("program graph manifest has invalid collections")
    supplied_nodes = tuple(nodes)
    if any(not isinstance(node, ProgramNode) for node in supplied_nodes):
        raise TypeError("nodes must contain ProgramNode values")
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
    graph = ProgramGraph(resources, connections, nodes=supplied_nodes, programs=manifest["programs"])
    if graph.contract_fingerprint != manifest["contract_fingerprint"]:
        raise ResourceGraphError("program graph artifact contract does not match reconstructed graph")
    return graph


__all__ = [
    "PROGRAM_GRAPH_ARTIFACT_FORMAT",
    "PROGRAM_GRAPH_ARTIFACT_VERSION",
    "Connection",
    "ConnectionExecution",
    "FormulaTensorViewTransfer",
    "StaticFormulaTensorViewTransfer",
    "ResourceGraphCompileError",
    "ResourceGraphCompiler",
    "ResourceGraphExecutionPlan",
    "AxisRange",
    "ProgramGraph",
    "ProgramGraphExecution",
    "ProgramGraphSaveResult",
    "ProgramGraphState",
    "ProgramNode",
    "ProgramNodeExecution",
    "ProgramNodeInvocation",
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
