"""Typed index, mask and segmented computation in the existing Fabric plan."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import ClassVar, Mapping

import torch
from torch import Tensor, nn

from .formula_v2 import (
    FormulaBindingError, FormulaOperand, FormulaTypeError, TensorType,
    _ACCUMULATION_DTYPES, _INDEX_ATOM_SIGNATURES, _FormulaExpr, _accumulation_dtype,
    _align_tensor, _as_expr, _require_exact_type, _require_runtime_condition,
    _validate_atom_operands, _validate_index_contract, _validate_mask_contract,
    _validate_tensor_against_type,
)


def _error(message: str) -> None:
    raise FormulaTypeError("FF2_INDEXING_TYPE", message)


def _gather_type(value: TensorType, indices: TensorType, axis: str, index_axis: str) -> TensorType:
    _validate_index_contract(value, indices, axis=axis, index_axis=index_axis, allow_integer_values=True)
    return TensorType(
        tuple(index_axis if a == axis else a for a in value.axis_names),
        tuple(indices.size_for(index_axis) if a == axis else value.size_for(a) for a in value.axis_names),
        dtype=value.dtype, domain=value.domain,
    )


def indexing_output_type(reference: str, types: tuple[TensorType, ...], attrs: Mapping[str, object]) -> TensorType:
    arity, fields = _INDEX_ATOM_SIGNATURES[reference]
    if len(types) != arity or set(attrs) != fields:
        _error("invalid indexing operand/attribute signature")
    value = types[0]
    if reference.endswith("axis-index@1"):
        axis = attrs["axis"]
        if axis not in value.axis_names:
            _error("AxisIndex axis must be present in the reference")
        return TensorType((axis,), (value.size_for(axis),), dtype="int64", domain=value.domain)
    if reference.endswith("compare@1"):
        _require_exact_type(value, types[1])
        if attrs["mode"] not in {"eq", "ne", "lt", "le", "gt", "ge"}:
            _error("unsupported comparison mode")
        if value.dtype == "boolean" and attrs["mode"] not in {"eq", "ne"}:
            _error("boolean comparison supports eq/ne only")
        return replace(value, dtype="boolean")
    if reference.endswith(("boolean-binary@1", "boolean-not@1")):
        if value.dtype != "boolean":
            _error("Boolean atoms require boolean operands")
        if len(types) == 2:
            _require_exact_type(value, types[1])
            if attrs["mode"] not in {"and", "or", "xor"}:
                _error("unsupported boolean mode")
        return value
    if reference.endswith("gather@2"):
        return _gather_type(value, types[1], attrs["axis"], attrs["index_axis"])
    if attrs["accumulation_dtype"] not in _ACCUMULATION_DTYPES:
        _error("unsupported accumulation dtype")
    if value.dtype in {"int64", "boolean"}:
        _error("Scatter add and Segment values must be floating")
    if reference.endswith("scatter@2"):
        if attrs["mode"] != "add":
            _error("Scatter@2 supports add only; use Scatter@1 for unique replace")
        expected = _gather_type(value, types[1], attrs["axis"], attrs["index_axis"])
        _require_exact_type(expected, types[2])
        return value
    axis, group = attrs["axis"], attrs["segment_axis"]
    if axis not in value.axis_names or group in value.axis_names:
        _error("Segment must replace a source axis with a new segment axis")
    if type(attrs["num_segments"]) is not int or attrs["num_segments"] <= 0:
        _error("num_segments must be a positive static integer")
    if attrs["mode"] not in {"sum", "mean", "amax", "softmax"}:
        _error("unsupported Segment mode")
    ids, mask = types[1:]
    if ids.dtype != "int64" or axis not in ids.axis_names:
        _error("Segment ids must be int64 and contain the source axis")
    _validate_mask_contract(replace(ids, dtype="boolean"), value)
    if ids.size_for(axis) != value.size_for(axis):
        _error("Segment ids must match the source axis extent")
    _validate_mask_contract(mask, value)
    output = TensorType(
        tuple(group if a == axis else a for a in value.axis_names),
        tuple(attrs["num_segments"] if a == axis else value.size_for(a) for a in value.axis_names),
        dtype=value.dtype, domain=value.domain,
    )
    return value if attrs["mode"] == "softmax" else output


def _expression(reference: str, operands: tuple[FormulaOperand, ...], **attrs: object) -> _FormulaExpr:
    values = tuple(_as_expr(v) for v in operands)
    result = indexing_output_type(reference, tuple(v.value_type for v in values), attrs)
    return _FormulaExpr(result, reference, values, tuple(attrs.items()))


def axis_index(reference: FormulaOperand, *, axis: str) -> _FormulaExpr:
    """Return int64 coordinates for one runtime axis; reference values are not read."""
    return _expression("arti/formula-atom-axis-index@1", (reference,), axis=axis)


def compare(left: FormulaOperand, right: FormulaOperand, *, mode: str) -> _FormulaExpr:
    """Compare explicitly same-shaped operands; no surrogate gradient is implied."""
    return _expression("arti/formula-atom-compare@1", (left, right), mode=mode)


def boolean_binary(left: FormulaOperand, right: FormulaOperand, *, mode: str) -> _FormulaExpr:
    return _expression("arti/formula-atom-boolean-binary@1", (left, right), mode=mode)


def boolean_not(value: FormulaOperand) -> _FormulaExpr:
    return _expression("arti/formula-atom-boolean-not@1", (value,))


def gather_v2(value: FormulaOperand, indices: FormulaOperand, *, axis: str, index_axis: str) -> _FormulaExpr:
    """Gather floating, boolean or int64 payload; repeated indices are allowed."""
    return _expression("arti/formula-atom-gather@2", (value, indices), axis=axis, index_axis=index_axis)


def scatter_add(
    base: FormulaOperand, indices: FormulaOperand, updates: FormulaOperand,
    *, axis: str, index_axis: str, accumulation_dtype: str = "float32",
) -> _FormulaExpr:
    """Add every indexed update to a base without mutating it."""
    return _expression("arti/formula-atom-scatter@2", (base, indices, updates),
                       axis=axis, index_axis=index_axis, mode="add", accumulation_dtype=accumulation_dtype)


def segment(
    value: FormulaOperand, ids: FormulaOperand, mask: FormulaOperand, *, axis: str,
    segment_axis: str, num_segments: int, mode: str = "sum", accumulation_dtype: str = "float32",
) -> _FormulaExpr:
    """Masked grouped sum/mean/amax or source-shaped stable grouped softmax."""
    return _expression("arti/formula-atom-segment@1", (value, ids, mask), axis=axis,
                       segment_axis=segment_axis, num_segments=num_segments, mode=mode,
                       accumulation_dtype=accumulation_dtype)


def validate_indexing_dtypes(reference: str, dtypes: tuple[torch.dtype, ...]) -> None:
    if reference.endswith(("compare@1", "boolean-binary@1")) and dtypes[0] != dtypes[1]:
        raise FormulaBindingError("FF2_RUNTIME_DTYPE_MISMATCH", "comparison/boolean dtypes must match")
    if reference.endswith(("gather@2", "scatter@2", "segment@1")) and dtypes[1] != torch.int64:
        raise FormulaBindingError("FF2_INDEX_DTYPE", "indices must use int64")
    if reference.endswith("scatter@2") and dtypes[0] != dtypes[2]:
        raise FormulaBindingError("FF2_RUNTIME_DTYPE_MISMATCH", "Scatter base/updates must share dtype")
    if reference.endswith("segment@1") and dtypes[2] != torch.bool:
        raise FormulaBindingError("FF2_MASK_DTYPE", "Segment mask must be boolean")


def _workset_indices(value, indices, value_type, index_type, axis, index_axis):
    dim = value_type.axis_names.index(axis)
    axes = tuple(index_axis if a == axis else a for a in value_type.axis_names)
    shape = list(value.shape)
    shape[dim] = indices.shape[index_type.axis_names.index(index_axis)]
    aligned = _align_tensor(indices, index_type.axis_names, axes).expand(shape)
    valid = ((indices >= 0) & (indices < value.shape[dim])).all()
    return aligned.clamp(0, value.shape[dim] - 1), dim, valid


def execute_indexing(reference, operands, types, attrs):
    """Return numeric output and device validity; invalid indices never become success."""
    value = operands[0]
    valid = torch.ones((), dtype=torch.bool, device=value.device)
    if reference.endswith("axis-index@1"):
        return torch.arange(value.shape[types[0].axis_names.index(attrs["axis"])], dtype=torch.int64, device=value.device), valid
    if reference.endswith("compare@1"):
        return getattr(torch, attrs["mode"])(*operands), valid
    if reference.endswith("boolean-binary@1"):
        return getattr(torch, "logical_" + attrs["mode"])(*operands), valid
    if reference.endswith("boolean-not@1"):
        return torch.logical_not(value), valid
    if reference.endswith(("gather@2", "scatter@2")):
        indices, dim, valid = _workset_indices(value, operands[1], types[0], types[1], attrs["axis"], attrs["index_axis"])
        if reference.endswith("gather@2"):
            return value.gather(dim, indices), valid
        dtype = _accumulation_dtype(value.dtype, attrs["accumulation_dtype"])
        return value.to(dtype).scatter_add(dim, indices, operands[2].to(dtype)).to(value.dtype), valid
    dim = types[0].axis_names.index(attrs["axis"])
    size = attrs["num_segments"]
    shape = list(value.shape)
    shape[dim] = size
    ids = _align_tensor(operands[1], types[1].axis_names, types[0].axis_names).expand_as(value)
    visible = _align_tensor(operands[2], types[2].axis_names, types[0].axis_names).expand_as(value)
    valid = ((~visible) | ((ids >= 0) & (ids < size))).all()
    safe_ids = torch.where(visible, ids, 0).clamp(0, size - 1)
    compute = value.to(_accumulation_dtype(value.dtype, attrs["accumulation_dtype"]))
    mode = attrs["mode"]
    if mode in {"mean", "amax"}:
        counts = torch.zeros(shape, dtype=torch.int64, device=value.device).scatter_add(dim, safe_ids, visible.to(torch.int64))
    if mode in {"sum", "mean"}:
        source = torch.where(visible, compute, 0)
        result = compute.new_zeros(shape).scatter_add(dim, safe_ids, source)
        if mode == "mean":
            result = result / counts.clamp_min(1).to(compute.dtype)
    else:
        source = torch.where(visible, compute, -torch.inf)
        maximum = compute.new_full(shape, -torch.inf).scatter_reduce(dim, safe_ids, source, reduce="amax", include_self=True)
        if mode == "amax":
            result = torch.where(counts > 0, maximum, 0)
        else:
            # Sanitize the shift before exp, including masked entries in empty groups.
            shift = torch.where(visible, compute - maximum.gather(dim, safe_ids), 0)
            exponent = torch.where(visible, torch.exp(shift), 0)
            total = compute.new_zeros(shape).scatter_add(dim, safe_ids, exponent)
            divisor = total.gather(dim, safe_ids)
            result = exponent / torch.where(divisor > 0, divisor, 1)
    return result.to(value.dtype), valid


def indexing_scratch_bytes(reference, shapes, types, compute_size, attrs):
    if reference.endswith("gather@2"):
        dim = types[0].axis_names.index(attrs["axis"])
        elements = math.prod(shapes[0]) // shapes[0][dim] * shapes[1][types[1].axis_names.index(attrs["index_axis"])]
        return 8 * elements + 3 * math.prod(shapes[1])
    if reference.endswith("scatter@2"):
        return 8 * math.prod(shapes[2]) + 3 * math.prod(shapes[1])
    if reference.endswith("segment@1"):
        elements = math.prod(shapes[0])
        dim = types[0].axis_names.index(attrs["axis"])
        groups = elements // shapes[0][dim] * attrs["num_segments"]
        # Safe index/mask/count buffers plus source/max/sum/normalization tensors.
        # No N*S one-hot storage; excludes backend allocator/autograd saved tensors.
        return elements * (16 + 3 + 7 * compute_size) + groups * (16 + 2 + 5 * compute_size)
    return 0


class _IndexAtom(nn.Module):
    def __init__(self, operand_types: tuple[TensorType, ...], **attributes: object) -> None:
        super().__init__()
        self.operand_types = tuple(operand_types)
        self.attributes = dict(attributes)
        self.output_type = indexing_output_type(self._component_reference, self.operand_types, self.attributes)

    def forward(self, *operands: Tensor) -> Tensor:
        _validate_atom_operands(operands, self.operand_types, require_same_dtype=False)
        validate_indexing_dtypes(self._component_reference, tuple(v.dtype for v in operands))
        result, valid = execute_indexing(self._component_reference, operands, self.operand_types, self.attributes)
        _require_runtime_condition(valid, code="FF2_INDEX_RANGE", message="active indices are outside the indexed axis")
        _validate_tensor_against_type(result, self.output_type, name="output")
        return result

    def component_config(self) -> dict[str, object]:
        return {"operand_types": [t.to_dict() for t in self.operand_types],
                "output_type": self.output_type.to_dict(), "attributes": dict(self.attributes)}


class AxisIndexAtom(_IndexAtom):
    _component_reference: ClassVar[str] = "arti/formula-atom-axis-index@1"


class CompareAtom(_IndexAtom):
    _component_reference: ClassVar[str] = "arti/formula-atom-compare@1"


class BooleanBinaryAtom(_IndexAtom):
    _component_reference: ClassVar[str] = "arti/formula-atom-boolean-binary@1"


class BooleanNotAtom(_IndexAtom):
    _component_reference: ClassVar[str] = "arti/formula-atom-boolean-not@1"


class GatherAtomV2(_IndexAtom):
    _component_reference: ClassVar[str] = "arti/formula-atom-gather@2"


class ScatterAtomV2(_IndexAtom):
    _component_reference: ClassVar[str] = "arti/formula-atom-scatter@2"

    def __init__(self, operand_types, *, axis, index_axis, mode="add", accumulation_dtype="float32"):
        super().__init__(operand_types, axis=axis, index_axis=index_axis, mode=mode, accumulation_dtype=accumulation_dtype)


class SegmentAtom(_IndexAtom):
    _component_reference: ClassVar[str] = "arti/formula-atom-segment@1"

    def __init__(self, operand_types, *, axis, segment_axis, num_segments, mode="sum", accumulation_dtype="float32"):
        super().__init__(operand_types, axis=axis, segment_axis=segment_axis, num_segments=num_segments,
                         mode=mode, accumulation_dtype=accumulation_dtype)


INDEX_ATOM_CLASSES = {cls._component_reference: cls for cls in (
    AxisIndexAtom, CompareAtom, BooleanBinaryAtom, BooleanNotAtom, GatherAtomV2, ScatterAtomV2, SegmentAtom,
)}

__all__ = ["AxisIndexAtom", "CompareAtom", "BooleanBinaryAtom", "BooleanNotAtom", "GatherAtomV2",
           "ScatterAtomV2", "SegmentAtom", "axis_index", "compare", "boolean_binary", "boolean_not",
           "gather_v2", "scatter_add", "segment"]
