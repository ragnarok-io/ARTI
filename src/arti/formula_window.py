"""Named-axis local windows, composed with ordinary Contract and Bank operands."""

from __future__ import annotations

import math
from typing import ClassVar, Mapping

import torch.nn.functional as F
from torch import Tensor, nn

from .formula_v2 import (
    FormulaBindingError, FormulaOperand, FormulaTypeError, TensorType,
    _FormulaExpr, _as_expr, _bind_axis_extents, _validate_atom_operands,
)


WINDOW_ATOM_REF = "arti/formula-atom-window@1"
_FIELDS = {"axis", "output_axis", "window_axis", "kernel_size", "stride", "dilation", "padding", "output_size"}


def _positions(length, attrs):
    left, right = attrs["padding"]
    width = attrs["dilation"] * (attrs["kernel_size"] - 1) + 1
    return (length + left + right - width) // attrs["stride"] + 1


def window_output_type(value_type: TensorType, attrs: Mapping[str, object]) -> TensorType:
    if set(attrs) != _FIELDS:
        raise FormulaTypeError("FF2_WINDOW_TYPE", "Window attribute signature is invalid")
    axis, output_axis, window_axis = (attrs[name] for name in ("axis", "output_axis", "window_axis"))
    if axis not in value_type.axis_names:
        raise FormulaTypeError("FF2_WINDOW_TYPE", "Window source axis must exist")
    if output_axis == window_axis or output_axis in value_type.axis_names or window_axis in value_type.axis_names:
        raise FormulaTypeError("FF2_WINDOW_TYPE", "Window output/window axes must be distinct new names")
    if any(type(attrs[name]) is not int or attrs[name] <= 0 for name in ("kernel_size", "stride", "dilation")):
        raise FormulaTypeError("FF2_WINDOW_TYPE", "Window kernel_size, stride and dilation must be positive integers")
    padding = attrs["padding"]
    if not isinstance(padding, (tuple, list)) or len(padding) != 2 or any(type(p) is not int or p < 0 for p in padding):
        raise FormulaTypeError("FF2_WINDOW_TYPE", "Window padding must be a pair of nonnegative integers")
    length, declared = value_type.size_for(axis), attrs["output_size"]
    if isinstance(length, int):
        count = _positions(length, attrs)
        if count <= 0:
            raise FormulaTypeError("FF2_WINDOW_TYPE", "Window must produce at least one position")
        if declared is None:
            declared = count
        elif isinstance(declared, int) and declared != count:
            raise FormulaTypeError("FF2_WINDOW_TYPE", "Window output_size differs from its geometry")
    elif declared is None:
        raise FormulaTypeError("FF2_WINDOW_TYPE", "Dynamic Window requires an explicit output_size integer or symbol")
    dim = value_type.axis_names.index(axis)
    return TensorType(
        value_type.axis_names[:dim] + (output_axis, window_axis) + value_type.axis_names[dim + 1:],
        value_type.sizes[:dim] + (declared, attrs["kernel_size"]) + value_type.sizes[dim + 1:],
        dtype=value_type.dtype, domain=value_type.domain,
    )


def window(
    value: FormulaOperand, *, axis: str, output_axis: str, window_axis: str,
    kernel_size: int, stride: int = 1, dilation: int = 1, padding=(0, 0), output_size=None,
) -> _FormulaExpr:
    """Replace one axis by (position, tap); out-of-bounds values are zero.

    Tap k at position p reads p*stride - padding[0] + k*dilation.
    Dynamic positions use an ordinary TensorSchema symbol, not a shape expression.
    """
    value = _as_expr(value)
    attrs = dict(axis=axis, output_axis=output_axis, window_axis=window_axis,
                 kernel_size=kernel_size, stride=stride, dilation=dilation,
                 padding=padding, output_size=output_size)
    result_type = window_output_type(value.value_type, attrs)
    attrs["padding"] = tuple(padding)
    attrs["output_size"] = result_type.size_for(output_axis)
    return _FormulaExpr(result_type, WINDOW_ATOM_REF, (value,), tuple(attrs.items()))


def bind_window_extents(shape, value_type, output_type, attrs, extents):
    """Bind derived dimensions before allocation using host shape metadata only."""
    dim = value_type.axis_names.index(attrs["axis"])
    count = _positions(shape[dim], attrs)
    if count <= 0:
        raise FormulaBindingError("FF2_WINDOW_SHAPE", "Window must produce at least one position")
    output_shape = tuple(shape[:dim]) + (count, attrs["kernel_size"]) + tuple(shape[dim + 1:])
    for declared, actual in zip(output_type.sizes, output_shape, strict=True):
        expected = declared if isinstance(declared, int) else extents.setdefault(declared, int(actual))
        if expected != actual:
            raise FormulaBindingError("FF2_WINDOW_SHAPE", f"Window dimension {declared!r} conflicts with derived extent {actual}")
    return output_shape


def window_padding_shape(shape, value_type, attrs):
    padded = list(shape)
    padded[value_type.axis_names.index(attrs["axis"])] += sum(attrs["padding"])
    return tuple(padded)


def window_scratch_bytes(shape, value_type, attrs, element_size):
    return math.prod(window_padding_shape(shape, value_type, attrs)) * element_size if any(attrs["padding"]) else 0


def execute_window(value, value_type, attrs):
    dim = value_type.axis_names.index(attrs["axis"])
    if any(attrs["padding"]):
        padding = (0, 0) * (value.ndim - dim - 1) + tuple(attrs["padding"])
        value = F.pad(value, padding)
    width = attrs["dilation"] * (attrs["kernel_size"] - 1) + 1
    # Native unfold appends a tap axis. Move it beside position without copying.
    return value.unfold(dim, width, attrs["stride"])[..., ::attrs["dilation"]].movedim(-1, dim + 1)


class WindowAtom(nn.Module):
    _component_reference: ClassVar[str] = WINDOW_ATOM_REF

    def __init__(self, value_type: TensorType, **attributes):
        super().__init__()
        from .formula_v2 import InputBinding

        expression = window(InputBinding("value", value_type), **attributes)
        self.value_type = value_type
        self.output_type = expression.value_type
        self.attributes = dict(expression.attributes)

    def forward(self, value: Tensor) -> Tensor:
        _validate_atom_operands((value,), (self.value_type,))
        extents = {}
        _bind_axis_extents(extents, value, self.value_type, name="value")
        bind_window_extents(value.shape, self.value_type, self.output_type, self.attributes, extents)
        return execute_window(value, self.value_type, self.attributes)

    def component_config(self):
        return {"value_type": self.value_type.to_dict(), "output_type": self.output_type.to_dict(),
                "attributes": dict(self.attributes)}


__all__ = ["WindowAtom", "window"]
