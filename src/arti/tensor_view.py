"""Logical tensor views for shape-polymorphic ARTI programs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import ClassVar, Mapping

import torch
from torch import Tensor

from .component_registry import canonical_contract_reference


TENSOR_VIEW_PATTERN_VERSION = 1
AXIS_DESCRIPTOR_VERSION = 1
TENSOR_INDEX_MAP_VERSION = 1

_IDENTIFIER = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_TOKEN = re.compile(r"^[a-z][a-z0-9]*(?:[-_.][a-z0-9]+)*$")


class TensorViewError(ValueError):
    """Raised when a logical tensor view is malformed or inadmissible."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _dtype_matches(value: Tensor, contract: str) -> bool:
    if contract == "any":
        return True
    if contract == "floating":
        return value.is_floating_point()
    if contract == "complex":
        return value.is_complex()
    if contract == "integral":
        return (
            not value.is_floating_point()
            and not value.is_complex()
            and value.dtype != torch.bool
        )
    if contract == "boolean":
        return value.dtype == torch.bool
    return str(value.dtype).removeprefix("torch.") == contract


@dataclass(frozen=True)
class AxisDescriptor:
    """One named logical axis in a runtime TensorView."""

    name: str
    role: str
    extent: int
    origin: float = 0.0
    scale: float = 1.0
    schema_version: int = AXIS_DESCRIPTOR_VERSION

    _component_reference: ClassVar[str] = "arti/axis-descriptor@1"

    def __post_init__(self) -> None:
        if self.schema_version != AXIS_DESCRIPTOR_VERSION:
            raise TensorViewError("unsupported AxisDescriptor version")
        if not isinstance(self.name, str) or _IDENTIFIER.fullmatch(self.name) is None:
            raise TensorViewError("axis name must be an identifier")
        if not isinstance(self.role, str) or _TOKEN.fullmatch(self.role) is None:
            raise TensorViewError("axis role must be a canonical token")
        if isinstance(self.extent, bool) or not isinstance(self.extent, int) or self.extent < 0:
            raise TensorViewError("axis extent must be a non-negative integer")
        if not math.isfinite(float(self.origin)):
            raise TensorViewError("axis origin must be finite")
        if not math.isfinite(float(self.scale)) or float(self.scale) <= 0:
            raise TensorViewError("axis scale must be finite and positive")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "ref": canonical_contract_reference(self._component_reference),
            "name": self.name,
            "role": self.role,
            "extent": self.extent,
            "origin": float(self.origin),
            "scale": float(self.scale),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> AxisDescriptor:
        required = {
            "schema_version",
            "ref",
            "name",
            "role",
            "extent",
            "origin",
            "scale",
            "fingerprint",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise TensorViewError("AxisDescriptor payload contains missing or unknown fields")
        if value["ref"] != canonical_contract_reference(cls._component_reference):
            raise TensorViewError("AxisDescriptor reference is invalid")
        result = cls(
            name=value["name"],
            role=value["role"],
            extent=value["extent"],
            origin=value["origin"],
            scale=value["scale"],
            schema_version=value["schema_version"],
        )
        if value["fingerprint"] != result.fingerprint:
            raise TensorViewError("AxisDescriptor fingerprint is invalid")
        return result


@dataclass(frozen=True)
class TensorIndexMap:
    """Optional source coordinates for a physically transformed tensor payload.

    ``coordinates`` has shape ``[*target_shape, source_rank]`` or
    ``[B, *target_shape, source_rank]``. A missing coordinate tensor denotes the
    identity map in the view's current logical axes.
    """

    source_axes: tuple[str, ...]
    source_shape: tuple[int, ...]
    target_shape: tuple[int, ...]
    coordinates: Tensor | None = None
    schema_version: int = TENSOR_INDEX_MAP_VERSION

    _runtime_contract_ref: ClassVar[str] = "arti/tensor-index-map@1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_axes", tuple(self.source_axes))
        object.__setattr__(self, "source_shape", tuple(self.source_shape))
        object.__setattr__(self, "target_shape", tuple(self.target_shape))
        if self.schema_version != TENSOR_INDEX_MAP_VERSION:
            raise TensorViewError("unsupported TensorIndexMap version")
        if len(self.source_axes) != len(self.source_shape):
            raise TensorViewError("source_axes and source_shape must have equal length")
        if len(set(self.source_axes)) != len(self.source_axes):
            raise TensorViewError("source_axes must be unique")
        if any(_IDENTIFIER.fullmatch(axis) is None for axis in self.source_axes):
            raise TensorViewError("source_axes must contain identifiers")
        if any(type(size) is not int or size < 0 for size in self.source_shape):
            raise TensorViewError("source_shape must contain non-negative integers")
        if any(type(size) is not int or size < 0 for size in self.target_shape):
            raise TensorViewError("target_shape must contain non-negative integers")
        coordinates = self.coordinates
        if coordinates is None:
            if self.source_shape != self.target_shape:
                raise TensorViewError("identity index maps require equal source and target shapes")
            return
        if not isinstance(coordinates, Tensor) or coordinates.dtype != torch.int64:
            raise TypeError("TensorIndexMap.coordinates must be an int64 Tensor")
        unbatched = (*self.target_shape, len(self.source_shape))
        if tuple(coordinates.shape) != unbatched and (
            coordinates.ndim != len(unbatched) + 1
            or tuple(coordinates.shape[1:]) != unbatched
        ):
            raise TensorViewError("TensorIndexMap coordinates do not match target/source ranks")
        for source_axis, extent in enumerate(self.source_shape):
            coordinate = coordinates[..., source_axis]
            if coordinate.numel() and (
                bool((coordinate < 0).any())
                or bool((coordinate >= extent).any())
            ):
                raise TensorViewError("TensorIndexMap coordinate is out of bounds")

    @classmethod
    def identity(cls, axes: tuple[str, ...], shape: tuple[int, ...]) -> TensorIndexMap:
        return cls(axes, shape, shape)

    @property
    def is_identity(self) -> bool:
        return self.coordinates is None

    def validate_target(self, value: Tensor, *, batch_axis: int) -> None:
        target = tuple(
            int(size) for index, size in enumerate(value.shape) if index != batch_axis
        )
        if target != self.target_shape:
            raise TensorViewError("TensorIndexMap target_shape does not match TensorView")
        if self.coordinates is not None and self.coordinates.ndim == len(self.target_shape) + 2:
            if self.coordinates.shape[0] != value.shape[batch_axis]:
                raise TensorViewError("batched TensorIndexMap does not match TensorView batch")
        if self.coordinates is not None and self.coordinates.device != value.device:
            raise TensorViewError("TensorIndexMap coordinates must share the Tensor device")


@dataclass(frozen=True)
class TensorView:
    """A tensor payload together with its current logical axes and source map."""

    value: Tensor
    axes: tuple[AxisDescriptor, ...]
    index_map: TensorIndexMap | None = None
    mask: Tensor | None = None

    _runtime_contract_ref: ClassVar[str] = "arti/tensor-view@1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "axes", tuple(self.axes))
        if not isinstance(self.value, Tensor) or not self.value.is_floating_point():
            raise TypeError("TensorView.value must be a floating Tensor")
        if self.value.ndim == 0:
            raise TensorViewError("TensorView must include a batch axis")
        if len(self.axes) != self.value.ndim:
            raise TensorViewError("TensorView axes must describe every tensor dimension")
        if any(not isinstance(axis, AxisDescriptor) for axis in self.axes):
            raise TypeError("TensorView axes must contain AxisDescriptor values")
        if len({axis.name for axis in self.axes}) != len(self.axes):
            raise TensorViewError("TensorView axis names must be unique")
        for axis, extent in zip(self.axes, self.value.shape, strict=True):
            if axis.extent != int(extent):
                raise TensorViewError("TensorView axis extent does not match its tensor")
        batch_axes = [index for index, axis in enumerate(self.axes) if axis.role == "batch"]
        if len(batch_axes) != 1:
            raise TensorViewError("TensorView must declare exactly one batch axis")
        if self.index_map is not None:
            if not isinstance(self.index_map, TensorIndexMap):
                raise TypeError("index_map must be TensorIndexMap or None")
            self.index_map.validate_target(self.value, batch_axis=batch_axes[0])
        if self.mask is not None:
            if (
                not isinstance(self.mask, Tensor)
                or self.mask.dtype != torch.bool
                or self.mask.shape != self.value.shape
                or self.mask.device != self.value.device
            ):
                raise TensorViewError("TensorView mask must be bool, same-shape, and same-device")

    @classmethod
    def from_tensor(
        cls,
        value: Tensor,
        *,
        axis_names: tuple[str, ...] | None = None,
        axis_roles: tuple[str, ...] | None = None,
        index_map: TensorIndexMap | None = None,
        mask: Tensor | None = None,
    ) -> TensorView:
        if not isinstance(value, Tensor):
            raise TypeError("value must be a Tensor")
        names = axis_names or tuple(f"axis{index}" for index in range(value.ndim))
        roles = axis_roles or tuple(
            "batch" if index == 0 else "generic" for index in range(value.ndim)
        )
        if len(names) != value.ndim or len(roles) != value.ndim:
            raise TensorViewError("axis names and roles must match tensor rank")
        axes = tuple(
            AxisDescriptor(name, role, int(extent))
            for name, role, extent in zip(names, roles, value.shape, strict=True)
        )
        return cls(value, axes, index_map=index_map, mask=mask)

    @property
    def batch_axis(self) -> int:
        return next(index for index, axis in enumerate(self.axes) if axis.role == "batch")

    @property
    def logical_shape(self) -> tuple[int, ...]:
        return tuple(axis.extent for axis in self.axes)

    @property
    def descriptor_fingerprint(self) -> str:
        return _fingerprint(
            {
                "ref": canonical_contract_reference(self._runtime_contract_ref),
                "axes": [axis.to_dict() for axis in self.axes],
                "index_source_axes": (
                    None if self.index_map is None else list(self.index_map.source_axes)
                ),
                "index_source_shape": (
                    None if self.index_map is None else list(self.index_map.source_shape)
                ),
                "index_kind": (
                    "implicit_identity"
                    if self.index_map is None
                    else "identity" if self.index_map.is_identity else "explicit"
                ),
            }
        )

    def slice_batch(self, index: int) -> TensorView:
        """Return one batch row while preserving logical and source metadata."""

        if type(index) is not int or not 0 <= index < self.value.shape[self.batch_axis]:
            raise IndexError("TensorView batch index is out of range")
        slices = [slice(None)] * self.value.ndim
        slices[self.batch_axis] = slice(index, index + 1)
        value = self.value[tuple(slices)]
        mask = None if self.mask is None else self.mask[tuple(slices)]
        axes = tuple(
            AxisDescriptor(
                axis.name,
                axis.role,
                1 if axis_index == self.batch_axis else axis.extent,
                axis.origin,
                axis.scale,
            )
            for axis_index, axis in enumerate(self.axes)
        )
        index_map = self.index_map
        if index_map is not None and index_map.coordinates is not None:
            coordinates = index_map.coordinates
            if coordinates.ndim == len(index_map.target_shape) + 2:
                index_map = TensorIndexMap(
                    index_map.source_axes,
                    index_map.source_shape,
                    index_map.target_shape,
                    coordinates[index : index + 1],
                )
        return TensorView(value, axes, index_map=index_map, mask=mask)


@dataclass(frozen=True)
class TensorViewPattern:
    """A bounded admission contract that does not prescribe one tensor rank."""

    dtype: str = "floating"
    device_class: str = "any"
    min_rank: int = 1
    max_rank: int = 8
    batch_role: str = "batch"
    allowed_axis_roles: tuple[str, ...] = ()
    require_index_map: bool = False
    schema_version: int = TENSOR_VIEW_PATTERN_VERSION

    _component_reference: ClassVar[str] = "arti/tensor-view-pattern@1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "allowed_axis_roles", tuple(self.allowed_axis_roles))
        if self.schema_version != TENSOR_VIEW_PATTERN_VERSION:
            raise TensorViewError("unsupported TensorViewPattern version")
        if _TOKEN.fullmatch(self.dtype) is None or _TOKEN.fullmatch(self.device_class) is None:
            raise TensorViewError("dtype and device_class must be canonical tokens")
        if type(self.min_rank) is not int or self.min_rank < 1:
            raise TensorViewError("min_rank must be a positive integer")
        if type(self.max_rank) is not int or self.max_rank < self.min_rank:
            raise TensorViewError("max_rank must not be smaller than min_rank")
        if _TOKEN.fullmatch(self.batch_role) is None:
            raise TensorViewError("batch_role must be a canonical token")
        if len(set(self.allowed_axis_roles)) != len(self.allowed_axis_roles):
            raise TensorViewError("allowed_axis_roles must be unique")
        if any(_TOKEN.fullmatch(role) is None for role in self.allowed_axis_roles):
            raise TensorViewError("allowed_axis_roles must be canonical tokens")
        if type(self.require_index_map) is not bool:
            raise TypeError("require_index_map must be bool")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "ref": canonical_contract_reference(self._component_reference),
            "dtype": self.dtype,
            "device_class": self.device_class,
            "min_rank": self.min_rank,
            "max_rank": self.max_rank,
            "batch_role": self.batch_role,
            "allowed_axis_roles": list(self.allowed_axis_roles),
            "require_index_map": self.require_index_map,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TensorViewPattern:
        required = {
            "schema_version",
            "ref",
            "dtype",
            "device_class",
            "min_rank",
            "max_rank",
            "batch_role",
            "allowed_axis_roles",
            "require_index_map",
            "fingerprint",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise TensorViewError("TensorViewPattern payload contains missing or unknown fields")
        if value["ref"] != canonical_contract_reference(cls._component_reference):
            raise TensorViewError("TensorViewPattern reference is invalid")
        roles = value["allowed_axis_roles"]
        if not isinstance(roles, (list, tuple)):
            raise TensorViewError("allowed_axis_roles must be a sequence")
        result = cls(
            dtype=value["dtype"],
            device_class=value["device_class"],
            min_rank=value["min_rank"],
            max_rank=value["max_rank"],
            batch_role=value["batch_role"],
            allowed_axis_roles=tuple(roles),
            require_index_map=value["require_index_map"],
            schema_version=value["schema_version"],
        )
        if value["fingerprint"] != result.fingerprint:
            raise TensorViewError("TensorViewPattern fingerprint is invalid")
        return result

    def validate(self, view: TensorView, *, name: str = "view") -> None:
        if not isinstance(view, TensorView):
            raise TypeError(f"{name} must be TensorView")
        if not self.min_rank <= view.value.ndim <= self.max_rank:
            raise TensorViewError(f"{name} rank is outside the TensorViewPattern bounds")
        if not _dtype_matches(view.value, self.dtype):
            raise TensorViewError(f"{name} dtype does not match TensorViewPattern")
        if self.device_class != "any" and view.value.device.type != self.device_class:
            raise TensorViewError(f"{name} device does not match TensorViewPattern")
        if sum(axis.role == self.batch_role for axis in view.axes) != 1:
            raise TensorViewError(f"{name} does not contain the declared batch role")
        if self.allowed_axis_roles and any(
            axis.role not in self.allowed_axis_roles for axis in view.axes
        ):
            raise TensorViewError(f"{name} contains an unsupported logical axis role")
        if self.require_index_map and view.index_map is None:
            raise TensorViewError(f"{name} requires an explicit TensorIndexMap")


__all__ = [
    "AXIS_DESCRIPTOR_VERSION",
    "TENSOR_INDEX_MAP_VERSION",
    "TENSOR_VIEW_PATTERN_VERSION",
    "AxisDescriptor",
    "TensorIndexMap",
    "TensorView",
    "TensorViewError",
    "TensorViewPattern",
]
