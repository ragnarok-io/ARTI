"""Native JAX lowering of the shared, typed Formula program."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import math
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import torch

from ..formula_v2 import (
    BankBinding,
    FormulaBindingError,
    FormulaProgram,
    InputBinding,
    TensorType,
    _bind_axis_extents,
    _bind_instruction_derived_extents,
    _validate_instruction_dtype_contract,
    formula_program_effect_refs,
)
from .contracts import register_binding


class FormulaResult(NamedTuple):
    outputs: dict[str, Any]
    valid: Any


def _array(value: Any) -> Any:
    if hasattr(value, "detach"):
        value = value.detach().cpu()
        if str(value.dtype) == "torch.bfloat16":
            return jnp.asarray(value.float().numpy(), dtype=jnp.bfloat16)
        value = value.numpy()
    return jnp.asarray(value)


def _validate(value: Any, value_type: TensorType, name: str) -> None:
    if value.ndim != len(value_type.axis_names):
        raise FormulaBindingError("FF2_RUNTIME_SHAPE", f"{name!r} has the wrong rank")
    for actual, expected in zip(value.shape, value_type.sizes, strict=True):
        if isinstance(expected, int) and actual != expected:
            raise FormulaBindingError("FF2_RUNTIME_SHAPE", f"{name!r} has the wrong shape")
    expected_dtype = value_type.dtype
    if expected_dtype == "floating":
        matches = jnp.issubdtype(value.dtype, jnp.floating)
    elif expected_dtype == "boolean":
        matches = value.dtype == jnp.bool_
    else:
        if expected_dtype in {"int64", "float64"} and not jax.config.x64_enabled:
            raise FormulaBindingError(
                "FF2_RUNTIME_DTYPE", "the declared 64-bit Formula type requires JAX X64"
            )
        matches = value.dtype == jnp.dtype(expected_dtype)
    if not matches:
        raise FormulaBindingError(
            "FF2_RUNTIME_DTYPE", f"{name!r} dtype does not match its Formula type"
        )


def _align(value: Any, source: tuple[str, ...], target: tuple[str, ...]) -> Any:
    positions = [target.index(axis) for axis in source]
    permutation = tuple(sorted(range(len(source)), key=positions.__getitem__))
    value = jnp.transpose(value, permutation) if source else value
    shape = [1] * len(target)
    for index in range(len(source)):
        shape[target.index(source[permutation[index]])] = value.shape[index]
    return value.reshape(shape)


def _compute(value: Any, policy: str = "float32") -> Any:
    dtype = value.dtype if policy == "activation" or value.dtype == jnp.float64 else jnp.float32
    return value.astype(dtype)


@jax.custom_jvp
def _abs(value: Any) -> Any:
    return jnp.abs(value)


@_abs.defjvp
def _abs_jvp(primals, tangents):
    (value,) = primals
    (tangent,) = tangents
    return jnp.abs(value), jnp.sign(value) * tangent


@jax.custom_jvp
def _vector_norm(value):
    return jnp.sqrt(jnp.sum(jnp.square(value), axis=-1))


@_vector_norm.defjvp
def _vector_norm_jvp(primals, tangents):
    (value,) = primals
    (tangent,) = tangents
    norm = _vector_norm(value)
    denominator = jnp.where(norm > 0, norm, 1)
    derivative = jnp.sum(value * tangent, axis=-1) / denominator
    return norm, jnp.where(norm > 0, derivative, 0)


def _scalar_map(value: Any, mode: str) -> Any:
    functions = {
        "gelu": lambda x: jax.nn.gelu(x, approximate=False),
        "relu": jax.nn.relu,
        "rsqrt": jax.lax.rsqrt,
        "sigmoid": jax.nn.sigmoid,
        "silu": jax.nn.silu,
        "tanh": jnp.tanh,
        "abs": _abs,
        "exp": jnp.exp,
        "expm1": jnp.expm1,
        "log": jnp.log,
        "log1p": jnp.log1p,
        "softplus": lambda x: jnp.where(x > 20, x, jax.nn.softplus(x)),
        "reciprocal": jnp.reciprocal,
        "sin": jnp.sin,
        "cos": jnp.cos,
    }
    return functions[mode](value)


def _softmax(value: Any, mask: Any, axis: int) -> Any:
    masked = jnp.where(mask, value, jnp.finfo(value.dtype).min)
    maximum = jnp.max(masked, axis=axis, keepdims=True)
    exponent = jnp.where(mask, jnp.exp(masked - maximum), 0)
    total = jnp.sum(exponent, axis=axis, keepdims=True)
    return exponent / jnp.where(total > 0, total, 1)


def _indices(value, indices, types, axis, index_axis):
    dimension = types[0].axis_names.index(axis)
    axes = tuple(index_axis if name == axis else name for name in types[0].axis_names)
    shape = list(value.shape)
    shape[dimension] = indices.shape[types[1].axis_names.index(index_axis)]
    expanded = jnp.broadcast_to(_align(indices, types[1].axis_names, axes), shape)
    valid = jnp.all((indices >= 0) & (indices < value.shape[dimension]))
    return jnp.clip(expanded, 0, value.shape[dimension] - 1), dimension, valid


def _scatter(base, indices, values, dimension, mode):
    coordinates = list(jnp.indices(indices.shape, sparse=True))
    coordinates[dimension] = indices
    indexed = base.at[tuple(coordinates)]
    if mode == "add":
        return indexed.add(values)
    if mode == "max":
        return indexed.max(values)
    return indexed.set(values)


def _segment(operands, types, attrs):
    value, ids, mask = operands
    dimension = types[0].axis_names.index(attrs["axis"])
    size = attrs["num_segments"]
    shape = list(value.shape)
    shape[dimension] = size
    ids = jnp.broadcast_to(_align(ids, types[1].axis_names, types[0].axis_names), value.shape)
    visible = jnp.broadcast_to(_align(mask, types[2].axis_names, types[0].axis_names), value.shape)
    valid = jnp.all(~visible | ((ids >= 0) & (ids < size)))
    safe = jnp.clip(jnp.where(visible, ids, 0), 0, size - 1)
    compute = _compute(value, attrs["accumulation_dtype"])
    counts = _scatter(
        jnp.zeros(shape, dtype=jnp.int32), safe, visible.astype(jnp.int32), dimension, "add"
    )
    mode = attrs["mode"]
    if mode in {"sum", "mean"}:
        result = _scatter(
            jnp.zeros(shape, dtype=compute.dtype),
            safe,
            jnp.where(visible, compute, 0),
            dimension,
            "add",
        )
        if mode == "mean":
            result = result / jnp.maximum(counts, 1)
    else:
        maximum = _scatter(
            jnp.full(shape, -jnp.inf, dtype=compute.dtype),
            safe,
            jnp.where(visible, compute, -jnp.inf),
            dimension,
            "max",
        )
        if mode == "amax":
            result = jnp.where(counts > 0, maximum, 0)
        else:
            selected = jnp.take_along_axis(maximum, safe, axis=dimension)
            shift = jnp.where(visible, compute - selected, 0)
            exponent = jnp.where(visible, jnp.exp(shift), 0)
            total = _scatter(
                jnp.zeros(shape, dtype=compute.dtype), safe, exponent, dimension, "add"
            )
            divisor = jnp.take_along_axis(total, safe, axis=dimension)
            result = exponent / jnp.where(divisor > 0, divisor, 1)
    return result.astype(value.dtype), valid


def _observation(reference, operands, attrs):
    value, states, mask, activity = operands[:4]
    if reference.endswith("observe-identity@1"):
        observed = jnp.broadcast_to(
            value[:, None], (value.shape[0], states.shape[1], *value.shape[1:])
        )
    elif reference.endswith("observe-affine@1"):
        projected = states @ operands[4].T + operands[5]
        raw_scale, raw_shift = jnp.split(projected, 2, axis=-1)
        feature_scale = 1 + attrs["scale"] * jnp.tanh(raw_scale)
        feature_shift = attrs["scale"] * jnp.tanh(raw_shift)
        observed = value[:, None] * feature_scale[:, :, None] + feature_shift[:, :, None]
    else:
        height, width = attrs["spatial_shape"]
        image = _compute(value).reshape(value.shape[0], height, width, value.shape[-1])
        state = _compute(states)
        if attrs["state_mode"] == "cartesian":
            dx, dy = state[..., 0], state[..., 1]
        else:
            radius, direction = state[..., 0], state[..., 1:3]
            norm = _vector_norm(direction)
            unit = direction / jnp.maximum(norm[..., None], attrs["direction_epsilon"])
            unit = jnp.where((norm > attrs["direction_epsilon"])[..., None], unit, 0)
            dx, dy = radius * unit[..., 0], radius * unit[..., 1]
        fy = jnp.fft.fftfreq(height).astype(image.dtype)
        fx = jnp.fft.fftfreq(width).astype(image.dtype)
        angle = (
            dy[..., None, None] * fy[None, None, :, None]
            + dx[..., None, None] * fx[None, None, None, :]
        )
        phase = jnp.exp(-2j * jnp.pi * angle)
        spectrum = jnp.fft.fftn(image, axes=(1, 2))
        observed = jnp.fft.ifftn(spectrum[:, None] * phase[..., None], axes=(2, 3)).real
        observed = observed.reshape(
            value.shape[0], states.shape[1], height * width, value.shape[-1]
        ).astype(value.dtype)
    weighted = observed * activity.astype(observed.dtype)[..., None, None]
    return jnp.where(mask[:, None, :, None], weighted, 0)


def _position(reference, operands, types, attrs, output_type):
    value = operands[0]
    if reference.endswith("position-sinusoidal@1"):
        dtype = (
            (value.dtype if jnp.issubdtype(value.dtype, jnp.floating) else jnp.float32)
            if attrs["dtype"] == "floating"
            else jnp.dtype(attrs["dtype"])
        )
        coordinates = _compute(value.astype(dtype))
        width = attrs["feature_size"]
        frequency = jnp.exp(
            -math.log(attrs["base"]) * jnp.arange(0, width, 2, dtype=coordinates.dtype) / width
        )
        phase = coordinates[..., None] * frequency
        return (
            jnp.stack((jnp.sin(phase), jnp.cos(phase)), axis=-1)
            .reshape(*coordinates.shape, -1)[..., :width]
            .astype(dtype)
        )
    if reference.endswith("position-relative@1"):
        return _align(value, types[0].axis_names, output_type.axis_names) - _align(
            operands[1], types[1].axis_names, output_type.axis_names
        )
    axis = types[0].axis_names.index(attrs["feature_axis"])
    ordered = jnp.moveaxis(value, axis, -1)
    coordinates = jnp.moveaxis(
        _align(operands[1], types[1].axis_names, types[0].axis_names), axis, -1
    )
    compute = _compute(ordered)
    frequency = jnp.exp(
        -math.log(attrs["base"])
        * jnp.arange(0, compute.shape[-1], 2, dtype=compute.dtype)
        / compute.shape[-1]
    )
    phase = coordinates.astype(compute.dtype) * frequency
    even, odd = compute[..., 0::2], compute[..., 1::2]
    result = jnp.stack(
        (
            even * jnp.cos(phase) - odd * jnp.sin(phase),
            even * jnp.sin(phase) + odd * jnp.cos(phase),
        ),
        axis=-1,
    ).reshape(ordered.shape)
    return jnp.moveaxis(result.astype(value.dtype), -1, axis)


def _execute(instruction, operands, types, output_type, extents):
    attrs = dict(instruction.attributes)
    reference = instruction.atom_ref
    value = operands[0]
    valid = jnp.asarray(True)
    if "formula-atom-observe-" in reference:
        return _observation(reference, operands, attrs), valid
    if "formula-atom-position-" in reference:
        return _position(reference, operands, types, attrs, output_type), valid
    if reference.endswith("formula-atom-window@1"):
        axis = types[0].axis_names.index(attrs["axis"])
        padding = [(0, 0)] * value.ndim
        padding[axis] = attrs["padding"]
        value = jnp.pad(value, padding)
        count = (value.shape[axis] - attrs["dilation"] * (attrs["kernel_size"] - 1) - 1) // attrs[
            "stride"
        ] + 1
        indices = (
            jnp.arange(count)[:, None] * attrs["stride"]
            + jnp.arange(attrs["kernel_size"])[None] * attrs["dilation"]
        )
        return jnp.take(value, indices, axis=axis), valid
    if reference.endswith("formula-atom-contract@1"):
        labels = dict.fromkeys((*types[0].axis_names, *types[1].axis_names, *attrs["output_axes"]))
        labels = {name: index for index, name in enumerate(labels)}
        right = dict(labels)
        for left_axis, right_axis in attrs["reduce_axes"]:
            right[right_axis] = labels[left_axis]
        result = jnp.einsum(
            _compute(value, attrs["accumulation_dtype"]),
            tuple(labels[n] for n in types[0].axis_names),
            _compute(operands[1], attrs["accumulation_dtype"]),
            tuple(right[n] for n in types[1].axis_names),
            tuple(labels[n] for n in attrs["output_axes"]),
            precision=jax.lax.Precision.HIGHEST,
        )
        return result.astype(value.dtype), valid
    if reference.endswith(("formula-atom-add@1", "formula-atom-scale@1")):
        right = operands[1]
        if reference.endswith("scale@1"):
            right = _align(right, types[1].axis_names, types[0].axis_names)
        left, right = (
            _compute(value, attrs["accumulation_dtype"]),
            _compute(right, attrs["accumulation_dtype"]),
        )
        result = left + right if reference.endswith("add@1") else left * right
        return result.astype(value.dtype), valid
    if reference.endswith(("formula-atom-reduce@1", "formula-atom-reduce@2")):
        axis = types[0].axis_names.index(attrs["axis"])
        compute = _compute(value, attrs["accumulation_dtype"])
        if compute.shape[axis] == 0:
            raise FormulaBindingError("FF2_EMPTY_REDUCTION", "Reduce does not accept an empty axis")
        if reference.endswith("reduce@1"):
            terms = jnp.moveaxis(compute, axis, 0)
            result, _ = jax.lax.scan(lambda carry, term: (carry + term, None), terms[0], terms[1:])
        else:
            functions = {
                "sum": jnp.sum,
                "mean": jnp.mean,
                "amax": jnp.max,
                "logsumexp": jax.scipy.special.logsumexp,
            }
            result = functions[attrs["mode"]](compute, axis=axis)
        return result.astype(value.dtype), valid
    if reference.endswith("formula-atom-reshape@1"):
        shape = tuple(
            size if isinstance(size, int) else extents.get(size, -1)
            for size in attrs["output_sizes"]
        )
        return value.reshape(shape), valid
    if reference.endswith("formula-atom-permute@1"):
        return jnp.transpose(
            value, tuple(types[0].axis_names.index(n) for n in attrs["output_axes"])
        ), valid
    if reference.endswith("formula-atom-axis-index@1"):
        return jnp.arange(
            value.shape[types[0].axis_names.index(attrs["axis"])], dtype=jnp.int64
        ), valid
    if reference.endswith("formula-atom-compare@1"):
        names = {
            "eq": "equal",
            "ne": "not_equal",
            "lt": "less",
            "le": "less_equal",
            "gt": "greater",
            "ge": "greater_equal",
        }
        return getattr(jnp, names[attrs["mode"]])(*operands), valid
    if reference.endswith("formula-atom-boolean-binary@1"):
        return getattr(jnp, "logical_" + attrs["mode"])(*operands), valid
    if reference.endswith("formula-atom-boolean-not@1"):
        return jnp.logical_not(value), valid
    if reference.endswith("formula-atom-segment@1"):
        return _segment(operands, types, attrs)
    if reference.endswith(
        (
            "formula-atom-gather@1",
            "formula-atom-gather@2",
            "fold@2",
            "formula-atom-scatter@1",
            "formula-atom-scatter@2",
            "unfold@2",
        )
    ):
        indices, axis, valid = _indices(
            value, operands[1], types, attrs["axis"], attrs["index_axis"]
        )
        if reference.endswith(("gather@1", "gather@2", "fold@2")) and not reference.endswith(
            "unfold@2"
        ):
            return jnp.take_along_axis(value, indices, axis=axis), valid
        mode = attrs.get("mode", "replace")
        if mode == "replace":
            ordered = jnp.sort(operands[1], axis=types[1].axis_names.index(attrs["index_axis"]))
            ordered = jnp.moveaxis(ordered, types[1].axis_names.index(attrs["index_axis"]), -1)
            valid = valid & jnp.all(ordered[..., 1:] != ordered[..., :-1])
        base = _compute(value, attrs.get("accumulation_dtype", "activation"))
        result = _scatter(
            base,
            indices,
            operands[2].astype(base.dtype),
            axis,
            "add" if mode == "add" else "replace",
        )
        return result.astype(value.dtype), valid
    if reference.endswith(("formula-atom-scalar-map@1", "formula-atom-scalar-map@2")):
        result = _scalar_map(
            _compute(value, attrs.get("accumulation_dtype", "activation")), attrs["mode"]
        )
        return result.astype(value.dtype), valid
    if reference.endswith("formula-atom-cast@1"):
        return value.astype(jnp.dtype(attrs["dtype"])), valid
    if reference.endswith("formula-atom-broadcast@1"):
        shape = tuple(
            size if isinstance(size, int) else extents[size] for size in attrs["output_sizes"]
        )
        return jnp.broadcast_to(
            _align(value, types[0].axis_names, output_type.axis_names), shape
        ), valid
    if reference.endswith("formula-atom-select@1"):
        return jnp.where(
            _align(value, types[0].axis_names, types[1].axis_names), operands[1], operands[2]
        ), valid
    if reference.endswith("formula-atom-lookup@1"):
        axis = types[0].axis_names.index(attrs["table_axis"])
        indices = operands[1]
        valid = jnp.all((indices >= 0) & (indices < value.shape[axis]))
        result = jnp.take(value, jnp.clip(indices, 0, value.shape[axis] - 1), axis=axis)
        axes = (*types[0].axis_names[:axis], *types[1].axis_names, *types[0].axis_names[axis + 1 :])
        return jnp.transpose(result, tuple(axes.index(n) for n in attrs["output_axes"])), valid
    if reference.endswith("formula-atom-slice@1"):
        indices = [slice(None)] * value.ndim
        indices[types[0].axis_names.index(attrs["axis"])] = slice(
            attrs["start"], attrs["stop"], attrs["step"]
        )
        return value[tuple(indices)], valid
    if reference.endswith("formula-atom-concat@1"):
        return jnp.concatenate(operands, axis=types[0].axis_names.index(attrs["axis"])), valid
    if reference.endswith("formula-atom-masked-softmax@1"):
        mask = _align(operands[1], types[1].axis_names, types[0].axis_names)
        result = _softmax(
            _compute(value, attrs["accumulation_dtype"]),
            mask,
            types[0].axis_names.index(attrs["axis"]),
        )
        return result.astype(value.dtype), valid
    raise NotImplementedError(f"no native JAX lowering for {reference}")


class BoundFormula:
    """One shared Formula IR with native JAX parameter and input bindings."""

    _component_reference = "arti/jax-formula-binding@1"

    def __init__(self, program: FormulaProgram):
        if not isinstance(program, FormulaProgram):
            raise TypeError("bind_formula requires FormulaProgram")
        if formula_program_effect_refs(program):
            raise ValueError(
                "effect-bearing Formula programs need a stateful native module binding"
            )
        self.program = program
        self.fingerprint = program.fingerprint
        self.input_names = tuple(b.name for b in program.bindings if isinstance(b, InputBinding))
        self.parameter_names = tuple(b.name for b in program.bindings if isinstance(b, BankBinding))
        self.output_names = program.outputs
        self._types = program.slot_types

    def init(
        self, key: Any, *, bank_initializers: Mapping[str, Any | Callable] | None = None
    ) -> dict[str, Any]:
        initializers = {} if bank_initializers is None else dict(bank_initializers)
        if set(initializers) != set(self.parameter_names):
            raise ValueError(f"bank_initializers must bind {self.parameter_names}")
        keys = jax.random.split(key, max(len(self.parameter_names), 1))
        parameters = {}
        for index, name in enumerate(self.parameter_names):
            initializer = initializers[name]
            value = (
                initializer(keys[index], self._types[name])
                if callable(initializer)
                else initializer
            )
            parameters[name] = _array(value)
            _validate(parameters[name], self._types[name], name)
        return parameters

    def apply_checked(self, params: Mapping[str, Any], inputs: Mapping[str, Any]) -> FormulaResult:
        if set(params) != set(self.parameter_names) or set(inputs) != set(self.input_names):
            raise ValueError("Formula parameters and inputs must match the declared bindings")
        values = {**params, **inputs}
        extents: dict[str, int] = {}
        valid = jnp.asarray(True)
        for binding in self.program.bindings:
            value = jnp.asarray(values[binding.name])
            _validate(value, binding.value_type, binding.name)
            _bind_axis_extents(extents, value, binding.value_type, name=binding.name)
            values[binding.name] = value
            valid = valid & jnp.all(jnp.isfinite(value))
        for instruction in self.program.instructions:
            operands = tuple(values[n] for n in instruction.input_slots)
            types = tuple(self._types[n] for n in instruction.input_slots)
            output_type = self._types[instruction.output_slot]
            _validate_instruction_dtype_contract(
                instruction, tuple(getattr(torch, str(value.dtype)) for value in operands), types
            )
            _bind_instruction_derived_extents(
                instruction, tuple(v.shape for v in operands), types, output_type, extents
            )
            output, condition = _execute(instruction, operands, types, output_type, extents)
            _validate(output, output_type, instruction.output_slot)
            _bind_axis_extents(extents, output, output_type, name=instruction.output_slot)
            values[instruction.output_slot] = output
            valid = valid & condition & jnp.all(jnp.isfinite(output))
        return FormulaResult({name: values[name] for name in self.output_names}, valid)

    def apply(self, params: Mapping[str, Any], inputs: Mapping[str, Any]) -> dict[str, Any]:
        return self.apply_checked(params, inputs).outputs


class BoundFormulaScan:
    """The shared FormulaScan contract lowered to a native lax.scan."""

    _component_reference = "arti/jax-formula-scan-binding@1"

    def __init__(self, scan):
        from ..formula_scan import FormulaScan

        if not isinstance(scan, FormulaScan):
            raise TypeError("bind_formula_scan requires FormulaScan")
        self.scan = scan
        self.body = BoundFormula(scan.body)

    def init(self, key, *, bank_initializers=None):
        return self.body.init(key, bank_initializers=bank_initializers)

    def apply(self, params, inputs):
        sequences = {}
        length = None
        for name, value_type in self.scan.sequence_types:
            value = jnp.asarray(inputs[name])
            _validate(value, value_type, name)
            axis = value_type.axis_names.index(self.scan.axis)
            sequences[name] = jnp.moveaxis(value, axis, 0)
            current = value.shape[axis]
            if length is not None and length != current:
                raise ValueError("FormulaScan sequence lengths must match")
            length = current
        if not 1 <= length <= self.scan.max_length:
            raise ValueError("FormulaScan length exceeds its declared horizon")
        carry = {name: inputs[name] for name, _ in self.scan.carry_outputs}
        constants = {
            name: value
            for name, value in inputs.items()
            if name not in carry and name not in sequences
        }

        def step(state, frame):
            result = self.body.apply(params, {**constants, **state, **frame})
            next_state = {name: result[output] for name, output in self.scan.carry_outputs}
            emitted = {name: result[output] for name, output in self.scan.emissions}
            return next_state, emitted

        carry, emitted = jax.lax.scan(step, carry, sequences)
        return {
            **{"carry." + name: value for name, value in carry.items()},
            **{"emit." + name: value for name, value in emitted.items()},
        }


def bind_formula(program: FormulaProgram) -> BoundFormula:
    return BoundFormula(program)


def bind_formula_scan(scan) -> BoundFormulaScan:
    return BoundFormulaScan(scan)


register_binding(
    BoundFormula,
    BoundFormula._component_reference,
    config_builder=lambda value: {"backend": "jax", "program_fingerprint": value.fingerprint},
)
register_binding(
    BoundFormulaScan,
    BoundFormulaScan._component_reference,
    config_builder=lambda value: {"backend": "jax", "scan": value.scan.to_dict()},
)


__all__ = ["BoundFormula", "BoundFormulaScan", "FormulaResult", "bind_formula", "bind_formula_scan"]
