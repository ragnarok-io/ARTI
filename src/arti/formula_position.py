"""Typed positional operands for Formula Fabric.

These atoms make coordinates and their use explicit Formula values. They do
not own an embedding table or a host-model position policy: learned tables are
ordinary Bank operands, while a host can bind its native position ids as an
ordinary input. This keeps position semantics composable with the rest of the
Fabric graph.
"""

from __future__ import annotations

import math
from typing import ClassVar, Mapping

import torch
from torch import Tensor, nn

from .formula_v2 import (
    FormulaBindingError,
    FormulaOperand,
    FormulaTypeError,
    TensorType,
    _FLOATING_DTYPES,
    _FormulaExpr,
    _POSITION_ATOM_SIGNATURES,
    _align_tensor,
    _as_expr,
    _require_axis_extent_equal,
    _require_compatible_domains,
    _require_compatible_dtypes,
    _validate_atom_operands,
    _validate_tensor_against_type,
)


def _error(message: str) -> None:
    raise FormulaTypeError("FF2_POSITION_TYPE", message)


def _coordinate_type(value: TensorType) -> None:
    if value.dtype not in _FLOATING_DTYPES | {"int64"}:
        _error("Position coordinates must be int64 or floating")


def _positive_int(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        _error(f"{name} must be a positive integer")
    return value


def _base(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _error("base must be a finite number greater than one")
    result = float(value)
    if not math.isfinite(result) or result <= 1.0:
        _error("base must be a finite number greater than one")
    return result


def position_output_type(
    reference: str, types: tuple[TensorType, ...], attrs: Mapping[str, object]
) -> TensorType:
    arity, fields = _POSITION_ATOM_SIGNATURES[reference]
    if len(types) != arity or set(attrs) != fields:
        _error("invalid positional operand/attribute signature")
    if reference.endswith("position-sinusoidal@1"):
        coordinates = types[0]
        _coordinate_type(coordinates)
        feature_axis = attrs["feature_axis"]
        if not isinstance(feature_axis, str) or feature_axis in coordinates.axis_names:
            _error("feature_axis must be a new named axis")
        feature_size = _positive_int(attrs["feature_size"], name="feature_size")
        dtype = attrs["dtype"]
        if dtype not in _FLOATING_DTYPES | {"floating"}:
            _error("Sinusoidal position dtype must be floating")
        _base(attrs["base"])
        return TensorType(
            (*coordinates.axis_names, feature_axis),
            (*coordinates.sizes, feature_size),
            dtype=dtype,
            domain=coordinates.domain,
        )
    if reference.endswith("position-relative@1"):
        left, right = types
        _coordinate_type(left)
        _coordinate_type(right)
        _require_compatible_domains(left, right)
        _require_compatible_dtypes(left, right)
        for axis in left.axis_names:
            if axis in right.axis_names:
                _require_axis_extent_equal(left, axis, right, axis)
        axes = (*left.axis_names, *(axis for axis in right.axis_names if axis not in left.axis_names))
        sizes = (*left.sizes, *(right.size_for(axis) for axis in right.axis_names if axis not in left.axis_names))
        return TensorType(axes, sizes, dtype=left.dtype, domain=left.domain)
    value, coordinates = types
    if value.dtype not in _FLOATING_DTYPES | {"floating"}:
        _error("Rotary position values must be floating")
    _coordinate_type(coordinates)
    _require_compatible_domains(value, coordinates)
    feature_axis = attrs["feature_axis"]
    if not isinstance(feature_axis, str) or feature_axis not in value.axis_names:
        _error("feature_axis must be present in the rotary value")
    if feature_axis in coordinates.axis_names:
        _error("Rotary coordinates must not vary over the feature axis")
    feature_size = value.size_for(feature_axis)
    if not isinstance(feature_size, int) or feature_size <= 0 or feature_size % 2:
        _error("Rotary feature axis must have a positive, static, even extent")
    for axis in coordinates.axis_names:
        if axis not in value.axis_names:
            _error("Rotary coordinate axes must be a subset of value axes")
        _require_axis_extent_equal(value, axis, coordinates, axis)
    _base(attrs["base"])
    return value


def _position(
    reference: str, operands: tuple[FormulaOperand, ...], **attributes: object
) -> _FormulaExpr:
    expressions = tuple(_as_expr(operand) for operand in operands)
    output = position_output_type(reference, tuple(item.value_type for item in expressions), attributes)
    return _FormulaExpr(output, reference, expressions, tuple(attributes.items()))


def sinusoidal_position(
    coordinates: FormulaOperand,
    *,
    feature_axis: str,
    feature_size: int,
    dtype: str = "floating",
    base: float = 10000.0,
) -> _FormulaExpr:
    """Encode explicit coordinates as a standard sin/cos feature field."""
    return _position(
        "arti/formula-atom-position-sinusoidal@1",
        (coordinates,),
        feature_axis=feature_axis,
        feature_size=feature_size,
        dtype=dtype,
        base=base,
    )


def relative_position(left: FormulaOperand, right: FormulaOperand) -> _FormulaExpr:
    """Create a named-axis field of ``left - right`` relative coordinates."""
    return _position("arti/formula-atom-position-relative@1", (left, right))


def rotary_position(
    value: FormulaOperand,
    coordinates: FormulaOperand,
    *,
    feature_axis: str,
    base: float = 10000.0,
) -> _FormulaExpr:
    """Apply a RoPE-compatible pair rotation using explicit coordinates."""
    return _position(
        "arti/formula-atom-position-rotary@1",
        (value, coordinates),
        feature_axis=feature_axis,
        base=base,
    )


def validate_position_dtypes(reference: str, dtypes: tuple[torch.dtype, ...]) -> None:
    if reference.endswith("position-relative@1") and dtypes[0] != dtypes[1]:
        raise FormulaBindingError(
            "FF2_RUNTIME_DTYPE_MISMATCH", "Relative position operands must share a dtype"
        )


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    return torch.float32 if dtype in {torch.float16, torch.bfloat16} else dtype


def execute_position(
    reference: str,
    operands: tuple[Tensor, ...],
    types: tuple[TensorType, ...],
    attrs: Mapping[str, object],
) -> Tensor:
    if reference.endswith("position-sinusoidal@1"):
        coordinates = operands[0]
        feature_size = int(attrs["feature_size"])
        dtype = (
            coordinates.dtype
            if attrs["dtype"] == "floating" and coordinates.is_floating_point()
            else torch.float32
            if attrs["dtype"] == "floating"
            else getattr(torch, attrs["dtype"])
        )
        compute_dtype = _compute_dtype(dtype)
        even = torch.arange(0, feature_size, 2, device=coordinates.device, dtype=compute_dtype)
        inverse_frequency = torch.exp(-math.log(float(attrs["base"])) * even / feature_size)
        phase = coordinates.to(compute_dtype).unsqueeze(-1) * inverse_frequency
        pairs = torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)
        return pairs[..., :feature_size].to(dtype)
    if reference.endswith("position-relative@1"):
        output_type = position_output_type(reference, types, attrs)
        return _align_tensor(operands[0], types[0].axis_names, output_type.axis_names) - _align_tensor(
            operands[1], types[1].axis_names, output_type.axis_names
        )
    value, coordinates = operands
    value_type, coordinate_type = types
    feature_axis = attrs["feature_axis"]
    feature_dimension = value_type.axis_names.index(feature_axis)
    permutation = tuple(index for index in range(value.ndim) if index != feature_dimension) + (feature_dimension,)
    inverse_permutation = tuple(permutation.index(index) for index in range(value.ndim))
    ordered = value.permute(permutation)
    aligned = _align_tensor(coordinates, coordinate_type.axis_names, value_type.axis_names).permute(permutation)
    compute_dtype = _compute_dtype(value.dtype)
    width = ordered.shape[-1]
    frequencies = torch.arange(0, width, 2, device=value.device, dtype=compute_dtype)
    inverse_frequency = torch.exp(-math.log(float(attrs["base"])) * frequencies / width)
    phase = aligned.to(compute_dtype) * inverse_frequency
    first, second = ordered[..., 0::2].to(compute_dtype), ordered[..., 1::2].to(compute_dtype)
    rotated = torch.stack(
        (first * phase.cos() - second * phase.sin(), first * phase.sin() + second * phase.cos()),
        dim=-1,
    ).flatten(-2)
    return rotated.to(value.dtype).permute(inverse_permutation)


def position_scratch_bytes(
    reference: str, output_shape: tuple[int, ...], output_dtype: torch.dtype
) -> int:
    """Visible phase/trigonometric temporaries; excludes backend/autograd workspace."""
    elements = math.prod(output_shape)
    compute_size = torch.empty((), dtype=_compute_dtype(output_dtype)).element_size()
    if reference.endswith("position-relative@1"):
        return elements * torch.empty((), dtype=output_dtype).element_size()
    return elements * compute_size * 3


class _PositionAtom(nn.Module):
    def __init__(self, operand_types: tuple[TensorType, ...], **attributes: object) -> None:
        super().__init__()
        self.operand_types = tuple(operand_types)
        self.attributes = dict(attributes)
        self.output_type = position_output_type(
            self._component_reference, self.operand_types, self.attributes
        )

    def forward(self, *operands: Tensor) -> Tensor:
        _validate_atom_operands(operands, self.operand_types, require_same_dtype=False)
        validate_position_dtypes(self._component_reference, tuple(value.dtype for value in operands))
        result = execute_position(self._component_reference, operands, self.operand_types, self.attributes)
        _validate_tensor_against_type(result, self.output_type, name="output")
        return result

    def component_config(self) -> dict[str, object]:
        return {
            "operand_types": [value.to_dict() for value in self.operand_types],
            "output_type": self.output_type.to_dict(),
            "attributes": dict(self.attributes),
        }


class SinusoidalPositionAtom(_PositionAtom):
    _component_reference: ClassVar[str] = "arti/formula-atom-position-sinusoidal@1"

    def __init__(self, operand_types, *, feature_axis, feature_size, dtype="floating", base=10000.0):
        super().__init__(operand_types, feature_axis=feature_axis, feature_size=feature_size, dtype=dtype, base=base)


class RelativePositionAtom(_PositionAtom):
    _component_reference: ClassVar[str] = "arti/formula-atom-position-relative@1"


class RotaryPositionAtom(_PositionAtom):
    _component_reference: ClassVar[str] = "arti/formula-atom-position-rotary@1"

    def __init__(self, operand_types, *, feature_axis, base=10000.0):
        super().__init__(operand_types, feature_axis=feature_axis, base=base)


POSITION_ATOM_CLASSES = {
    cls._component_reference: cls
    for cls in (SinusoidalPositionAtom, RelativePositionAtom, RotaryPositionAtom)
}


__all__ = [
    "RelativePositionAtom",
    "RotaryPositionAtom",
    "SinusoidalPositionAtom",
    "relative_position",
    "rotary_position",
    "sinusoidal_position",
]
