"""Versioned tensor, shape, and gradient contracts for shape-autonomous ARTI programs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import re
from typing import ClassVar, Literal, Mapping, Sequence

import torch
from torch import Tensor

from .component_registry import ComponentRef, canonical_contract_reference


TENSOR_SCHEMA_VERSION = 1
SHAPE_RELATION_VERSION = 1
GRADIENT_CONTRACT_VERSION = 1

_SYMBOL = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_TOKEN = re.compile(r"^[a-z][a-z0-9]*(?:[-_.][a-z0-9]+)*$")

ShapeDimension = int | str
ShapeRelationKind = Literal[
    "preserves_shape",
    "maps_shape",
    "arbitrary_to_terminal",
]
GradientMode = Literal["autograd", "custom_vjp", "straight_through", "detached"]


class TensorSchemaError(ValueError):
    """Raised when a tensor contract or tensor admission is invalid."""


def is_shape_dimension(value: object, *, allow_zero: bool = True) -> bool:
    """Return whether a value uses ARTI's canonical dimension representation."""

    if isinstance(value, bool) or not isinstance(value, (int, str)):
        return False
    if isinstance(value, int):
        return value >= 0 if allow_zero else value > 0
    return _SYMBOL.fullmatch(value) is not None


def normalize_shape_dimensions(
    dimensions: Sequence[ShapeDimension],
    *,
    allow_zero: bool = True,
) -> tuple[ShapeDimension, ...]:
    """Validate and freeze concrete or symbolic dimensions."""

    if isinstance(dimensions, (str, bytes)):
        raise TensorSchemaError("dimensions must be a sequence, not text")
    result = tuple(dimensions)
    if not all(is_shape_dimension(item, allow_zero=allow_zero) for item in result):
        qualifier = "non-negative" if allow_zero else "positive"
        raise TensorSchemaError(
            f"dimensions must be {qualifier} integers or valid symbols"
        )
    return result


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _validate_token(value: str, *, name: str) -> None:
    if not isinstance(value, str) or _TOKEN.fullmatch(value) is None:
        raise TensorSchemaError(f"{name} must be a canonical lowercase token")


def _dtype_matches(value: Tensor, contract: str) -> bool:
    if contract == "any":
        return True
    if contract == "floating":
        return value.is_floating_point()
    if contract == "complex":
        return value.is_complex()
    if contract == "integral":
        return not value.is_floating_point() and not value.is_complex() and value.dtype != torch.bool
    if contract == "boolean":
        return value.dtype == torch.bool
    return str(value.dtype).removeprefix("torch.") == contract


@dataclass(frozen=True)
class TensorSchema:
    """A logical tensor schema with concrete or named symbolic dimensions."""

    dtype: str
    device_class: str
    dimensions: tuple[ShapeDimension, ...]
    semantic_axes: tuple[str, ...]
    layout: str = "strided"
    mask_semantics: str | None = None
    schema_version: int = TENSOR_SCHEMA_VERSION

    _component_reference: ClassVar[str] = "arti/tensor-schema@1"

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "dimensions",
            normalize_shape_dimensions(self.dimensions),
        )
        object.__setattr__(self, "semantic_axes", tuple(self.semantic_axes))
        if self.schema_version != TENSOR_SCHEMA_VERSION:
            raise TensorSchemaError("unsupported TensorSchema version")
        _validate_token(self.dtype, name="dtype")
        _validate_token(self.device_class, name="device_class")
        _validate_token(self.layout, name="layout")
        if len(self.dimensions) != len(self.semantic_axes):
            raise TensorSchemaError("dimensions and semantic_axes must have equal length")
        if len(set(self.semantic_axes)) != len(self.semantic_axes):
            raise TensorSchemaError("semantic_axes must be unique")
        for axis in self.semantic_axes:
            if not isinstance(axis, str) or _SYMBOL.fullmatch(axis) is None:
                raise TensorSchemaError("semantic axes must be valid identifiers")
        if self.mask_semantics is not None:
            _validate_token(self.mask_semantics, name="mask_semantics")

    @property
    def rank(self) -> int:
        return len(self.dimensions)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "ref": canonical_contract_reference(self._component_reference),
            "dtype": self.dtype,
            "device_class": self.device_class,
            "rank": self.rank,
            "dimensions": list(self.dimensions),
            "semantic_axes": list(self.semantic_axes),
            "layout": self.layout,
            "mask_semantics": self.mask_semantics,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TensorSchema:
        required = {
            "schema_version",
            "ref",
            "dtype",
            "device_class",
            "rank",
            "dimensions",
            "semantic_axes",
            "layout",
            "mask_semantics",
            "fingerprint",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise TensorSchemaError("TensorSchema payload contains missing or unknown fields")
        if value["ref"] != canonical_contract_reference(cls._component_reference):
            raise TensorSchemaError("TensorSchema reference is invalid")
        dimensions = value["dimensions"]
        axes = value["semantic_axes"]
        if not isinstance(dimensions, (list, tuple)) or not isinstance(axes, (list, tuple)):
            raise TensorSchemaError("TensorSchema dimensions and axes must be sequences")
        result = cls(
            dtype=value["dtype"],
            device_class=value["device_class"],
            dimensions=tuple(dimensions),
            semantic_axes=tuple(axes),
            layout=value["layout"],
            mask_semantics=value["mask_semantics"],
            schema_version=value["schema_version"],
        )
        if value["rank"] != result.rank or value["fingerprint"] != result.fingerprint:
            raise TensorSchemaError("TensorSchema rank or fingerprint is invalid")
        return result

    @classmethod
    def from_tensor(
        cls,
        value: Tensor,
        *,
        semantic_axes: tuple[str, ...] | None = None,
        mask_semantics: str | None = None,
    ) -> TensorSchema:
        if not isinstance(value, Tensor):
            raise TypeError("value must be a Tensor")
        axes = semantic_axes or tuple(f"axis{index}" for index in range(value.ndim))
        return cls(
            dtype=str(value.dtype).removeprefix("torch."),
            device_class=value.device.type,
            dimensions=tuple(value.shape),
            semantic_axes=axes,
            layout=str(value.layout).removeprefix("torch."),
            mask_semantics=mask_semantics,
        )

    def validate_tensor(
        self,
        value: Tensor,
        *,
        symbols: Mapping[str, int] | None = None,
        name: str = "tensor",
    ) -> dict[str, int]:
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a Tensor")
        if value.ndim != self.rank:
            raise TensorSchemaError(f"{name} rank does not match TensorSchema")
        if not _dtype_matches(value, self.dtype):
            raise TensorSchemaError(f"{name} dtype does not match TensorSchema")
        if self.device_class != "any" and value.device.type != self.device_class:
            raise TensorSchemaError(f"{name} device does not match TensorSchema")
        actual_layout = str(value.layout).removeprefix("torch.")
        if self.layout != "any" and actual_layout != self.layout:
            raise TensorSchemaError(f"{name} layout does not match TensorSchema")
        resolved = dict(symbols or {})
        for expected, actual in zip(self.dimensions, value.shape, strict=True):
            if isinstance(expected, int):
                if expected != actual:
                    raise TensorSchemaError(f"{name} shape does not match TensorSchema")
                continue
            previous = resolved.setdefault(expected, actual)
            if previous != actual:
                raise TensorSchemaError(
                    f"{name} symbolic dimension {expected!r} is inconsistent"
                )
        return resolved


@dataclass(frozen=True)
class ShapeRelation:
    """A declared relation between one component's logical input and output shapes."""

    kind: ShapeRelationKind
    expression: str
    schema_version: int = SHAPE_RELATION_VERSION

    _component_reference: ClassVar[str] = "arti/shape-relation@1"

    def __post_init__(self) -> None:
        if self.schema_version != SHAPE_RELATION_VERSION:
            raise TensorSchemaError("unsupported ShapeRelation version")
        if self.kind not in {"preserves_shape", "maps_shape", "arbitrary_to_terminal"}:
            raise TensorSchemaError("unsupported shape relation kind")
        if not isinstance(self.expression, str) or not self.expression.strip():
            raise TensorSchemaError("shape relation expression must be non-empty")

    @classmethod
    def preserves_shape(cls) -> ShapeRelation:
        return cls("preserves_shape", "output.shape == input.shape")

    @classmethod
    def maps_shape(cls, expression: str) -> ShapeRelation:
        return cls("maps_shape", expression)

    @classmethod
    def arbitrary_to_terminal(cls, expression: str) -> ShapeRelation:
        return cls("arbitrary_to_terminal", expression)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "ref": canonical_contract_reference(self._component_reference),
            "kind": self.kind,
            "expression": self.expression,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> ShapeRelation:
        required = {"schema_version", "ref", "kind", "expression", "fingerprint"}
        if not isinstance(value, Mapping) or set(value) != required:
            raise TensorSchemaError("ShapeRelation payload contains missing or unknown fields")
        if value["ref"] != canonical_contract_reference(cls._component_reference):
            raise TensorSchemaError("ShapeRelation reference is invalid")
        result = cls(value["kind"], value["expression"], value["schema_version"])
        if value["fingerprint"] != result.fingerprint:
            raise TensorSchemaError("ShapeRelation fingerprint is invalid")
        return result

    def validate_schemas(self, source: TensorSchema, target: TensorSchema) -> None:
        if not isinstance(source, TensorSchema) or not isinstance(target, TensorSchema):
            raise TypeError("shape relation requires TensorSchema values")
        if self.kind == "preserves_shape" and (
            source.dimensions != target.dimensions
            or source.semantic_axes != target.semantic_axes
        ):
            raise TensorSchemaError("preserves_shape requires identical logical shapes")


@dataclass(frozen=True)
class GradientContract:
    """A versioned declaration of how gradients cross a tensor-program boundary."""

    mode: GradientMode
    contract_ref: str
    schema_version: int = GRADIENT_CONTRACT_VERSION

    _component_reference: ClassVar[str] = "arti/gradient-contract@1"

    def __post_init__(self) -> None:
        if self.schema_version != GRADIENT_CONTRACT_VERSION:
            raise TensorSchemaError("unsupported GradientContract version")
        if self.mode not in {"autograd", "custom_vjp", "straight_through", "detached"}:
            raise TensorSchemaError("unsupported gradient contract mode")
        reference = canonical_contract_reference(self.contract_ref)
        ComponentRef.parse(reference)
        object.__setattr__(self, "contract_ref", reference)

    @classmethod
    def autograd(cls) -> GradientContract:
        return cls("autograd", "arti/gradient-autograd@1")

    @classmethod
    def straight_through(cls) -> GradientContract:
        return cls("straight_through", "arti/gradient-straight-through@1")

    @classmethod
    def detached(cls) -> GradientContract:
        return cls("detached", "arti/gradient-detached@1")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "ref": canonical_contract_reference(self._component_reference),
            "mode": self.mode,
            "contract_ref": self.contract_ref,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> GradientContract:
        required = {"schema_version", "ref", "mode", "contract_ref", "fingerprint"}
        if not isinstance(value, Mapping) or set(value) != required:
            raise TensorSchemaError("GradientContract payload contains missing or unknown fields")
        if value["ref"] != canonical_contract_reference(cls._component_reference):
            raise TensorSchemaError("GradientContract reference is invalid")
        result = cls(value["mode"], value["contract_ref"], value["schema_version"])
        if value["fingerprint"] != result.fingerprint:
            raise TensorSchemaError("GradientContract fingerprint is invalid")
        return result


__all__ = [
    "GRADIENT_CONTRACT_VERSION",
    "SHAPE_RELATION_VERSION",
    "TENSOR_SCHEMA_VERSION",
    "GradientContract",
    "GradientMode",
    "ShapeDimension",
    "ShapeRelation",
    "ShapeRelationKind",
    "TensorSchema",
    "TensorSchemaError",
]
