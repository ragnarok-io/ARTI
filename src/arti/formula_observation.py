"""Versioned Observation atoms backed by the existing observation operators."""

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
    _as_expr,
    _FormulaExpr,
    _OBSERVATION_ATOM_SIGNATURES as OBSERVATION_ATOM_SIGNATURES,
    _validate_atom_operands,
    broadcast,
    select,
)
from .observation import (
    apply_observation_activity,
    fourier_observation,
    identity_observation,
    state_affine_observation,
)


def _error(message: str) -> None:
    raise FormulaTypeError("FF2_OBSERVATION_TYPE", message)


def observation_output_type(
    reference: str, types: tuple[TensorType, ...], attributes: Mapping[str, object]
) -> TensorType:
    arity, fields = OBSERVATION_ATOM_SIGNATURES[reference]
    if len(types) != arity or set(attributes) != fields:
        _error("Observation operand or attribute signature is invalid")
    value, states, mask, activity = types[:4]
    if len(value.axis_names) != 3 or len(states.axis_names) != 3:
        _error("Observation expects substrate [B,N,D] and states [B,T,S]")
    batch, position, feature = value.axis_names
    if states.axis_names[0] != batch or states.sizes[0] != value.sizes[0]:
        _error("Observation states and substrate must share the batch axis and extent")
    observation_axis = states.axis_names[1]
    if observation_axis in value.axis_names:
        _error("Observation must introduce a distinct trajectory axis")
    if (
        mask.axis_names != (batch, position)
        or mask.sizes != value.sizes[:2]
        or mask.dtype != "boolean"
    ):
        _error("Observation substrate mask must be boolean [B,N]")
    if activity.axis_names != states.axis_names[:2] or activity.sizes != states.sizes[:2]:
        _error("Observation activity must match states [B,T]")
    if (
        value.dtype in {"boolean", "int64"}
        or states.dtype in {"boolean", "int64"}
        or activity.dtype == "int64"
    ):
        _error("Observation requires floating substrate/states and boolean or floating activity")
    if reference.endswith("observe-affine@1"):
        weight, bias = types[4:]
        dim, state_dim = value.sizes[-1], states.sizes[-1]
        if not isinstance(dim, int) or not isinstance(state_dim, int):
            _error("Affine observation feature and state dimensions must be static")
        if weight.sizes != (2 * dim, state_dim) or bias.sizes != (2 * dim,):
            _error("Affine observation requires weight [2D,S] and bias [2D]")
        if (
            weight.axis_names[-1] != states.axis_names[-1]
            or bias.axis_names != weight.axis_names[:1]
        ):
            _error("Affine observation projection axes must match state and bias axes")
        if any(t.dtype in {"boolean", "int64"} for t in (weight, bias)):
            _error("Affine observation projection operands must be floating")
        scale = attributes["scale"]
        if (
            isinstance(scale, bool)
            or not isinstance(scale, (int, float))
            or not math.isfinite(scale)
            or scale <= 0
        ):
            _error("Affine observation scale must be finite and positive")
    if reference.endswith("observe-fourier@1"):
        shape = attributes["spatial_shape"]
        if (
            not isinstance(shape, (tuple, list))
            or len(shape) != 2
            or any(type(n) is not int or n <= 0 for n in shape)
        ):
            _error("Fourier observation requires a positive integer spatial_shape [H,W]")
        if value.sizes[1] != math.prod(shape):
            _error("Fourier observation substrate length must equal H*W")
        mode = attributes["state_mode"]
        if mode not in {"cartesian", "polar"} or states.sizes[-1] != (
            2 if mode == "cartesian" else 3
        ):
            _error("Fourier states must end with 2 (cartesian) or 3 (polar)")
        epsilon = attributes["direction_epsilon"]
        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, (int, float))
            or not math.isfinite(epsilon)
            or epsilon <= 0
        ):
            _error("Fourier direction_epsilon must be finite and positive")
        if attributes["compile_policy"] not in {"safe_training", "fullgraph_forward"}:
            _error("Fourier compile_policy must be safe_training or fullgraph_forward")
    return TensorType(
        (batch, observation_axis, position, feature),
        (value.sizes[0], states.sizes[1], *value.sizes[1:]),
        dtype=value.dtype,
        domain=value.domain,
    )


def _observe(
    reference: str, operands: tuple[FormulaOperand, ...], **attributes: object
) -> _FormulaExpr:
    expressions = tuple(_as_expr(operand) for operand in operands)
    output = observation_output_type(
        reference, tuple(e.value_type for e in expressions), attributes
    )
    return _FormulaExpr(output, reference, expressions, tuple(attributes.items()))


def observe_identity(
    value: FormulaOperand,
    states: FormulaOperand,
    mask: FormulaOperand,
    activity: FormulaOperand,
) -> _FormulaExpr:
    """Observe the unchanged original substrate at each trajectory state."""
    return _observe("arti/formula-atom-observe-identity@1", (value, states, mask, activity))


def observe_affine(
    value: FormulaOperand,
    states: FormulaOperand,
    mask: FormulaOperand,
    activity: FormulaOperand,
    weight: FormulaOperand,
    bias: FormulaOperand,
    *,
    scale: float = 0.1,
) -> _FormulaExpr:
    """StateAffineObservation with explicit, optionally Bank-owned parameters."""
    return _observe(
        "arti/formula-atom-observe-affine@1",
        (value, states, mask, activity, weight, bias),
        scale=scale,
    )


def observe_fourier(
    value: FormulaOperand,
    states: FormulaOperand,
    mask: FormulaOperand,
    activity: FormulaOperand,
    *,
    spatial_shape: tuple[int, int],
    state_mode: str = "cartesian",
    direction_epsilon: float = 1e-6,
    compile_policy: str = "safe_training",
) -> _FormulaExpr:
    """Circular Cartesian/polar observation using the shared torch.fft backend."""
    return _observe(
        "arti/formula-atom-observe-fourier@1",
        (value, states, mask, activity),
        spatial_shape=tuple(spatial_shape),
        state_mode=state_mode,
        direction_epsilon=direction_epsilon,
        compile_policy=compile_policy,
    )


def observation_mask(
    substrate_mask: FormulaOperand, trajectory_mask: FormulaOperand
) -> _FormulaExpr:
    """Compose [B,N] and [B,T] masks using ordinary Fabric instructions."""
    source, trajectory = _as_expr(substrate_mask), _as_expr(trajectory_mask)
    left, right = source.value_type, trajectory.value_type
    if (
        left.dtype != "boolean"
        or right.dtype != "boolean"
        or len(left.sizes) != 2
        or len(right.sizes) != 2
    ):
        _error("Observation masks must be boolean [B,N] and [B,T]")
    if left.axis_names[0] != right.axis_names[0] or left.sizes[0] != right.sizes[0]:
        _error("Observation masks must share their batch axis and extent")
    axes = (left.axis_names[0], right.axis_names[1], left.axis_names[1])
    sizes = (left.sizes[0], right.sizes[1], left.sizes[1])
    source = broadcast(source, output_axes=axes, output_sizes=sizes)
    trajectory = broadcast(trajectory, output_axes=axes, output_sizes=sizes)
    return select(trajectory, source, trajectory)


def validate_observation_dtypes(reference: str, dtypes: tuple[torch.dtype, ...]) -> None:
    if reference.endswith("observe-affine@1") and len({dtypes[i] for i in (0, 1, 4, 5)}) != 1:
        raise FormulaBindingError(
            "FF2_RUNTIME_DTYPE_MISMATCH",
            "Affine observation substrate, states and projection dtypes must match",
        )


def execute_observation(
    reference: str, operands: tuple[Tensor, ...], attributes: Mapping[str, object]
) -> Tensor:
    value, states, mask, activity = operands[:4]
    if reference.endswith("observe-identity@1"):
        observed = identity_observation(value, states)
    elif reference.endswith("observe-affine@1"):
        observed = state_affine_observation(
            value, states, operands[4], operands[5], scale=attributes["scale"]
        )
    else:
        observed = fourier_observation(value, states, **attributes)
    return apply_observation_activity(observed, mask, activity)


def observation_scratch_bytes(
    reference: str, shapes: tuple[tuple[int, ...], ...], dtype: torch.dtype
) -> int:
    """Explicit tensor temporary estimate; excludes backend FFT plans/autograd saves."""
    batch, count, dim = shapes[0]
    steps = shapes[1][1]
    elements = batch * steps * count * dim
    size = torch.empty((), dtype=dtype).element_size()
    if reference.endswith("observe-fourier@1"):
        real = 8 if dtype == torch.float64 else 4
        # Cast image/state, phase construction, spectrum, product, inverse result,
        # and activity/where buffers. torch.fft owns any additional plan scratch.
        return (
            real
            * (
                batch * count * dim * 3
                + math.prod(shapes[1])
                + batch * steps * count * 8
                + elements * 4
            )
            + elements * size * 3
        )
    if reference.endswith("observe-affine@1"):
        return size * (batch * steps * dim * 8 + elements * 4)
    return size * elements * 2


class _ObservationAtom(nn.Module):
    def __init__(self, operand_types: tuple[TensorType, ...], **attributes: object) -> None:
        super().__init__()
        self.operand_types = tuple(operand_types)
        self.attributes = attributes
        self.output_type = observation_output_type(
            self._component_reference, self.operand_types, attributes
        )

    def forward(self, *operands: Tensor) -> Tensor:
        _validate_atom_operands(operands, self.operand_types, require_same_dtype=False)
        validate_observation_dtypes(self._component_reference, tuple(t.dtype for t in operands))
        return execute_observation(self._component_reference, operands, self.attributes)

    def component_config(self) -> dict[str, object]:
        return {
            "operand_types": [t.to_dict() for t in self.operand_types],
            "output_type": self.output_type.to_dict(),
            "attributes": dict(self.attributes),
        }


class IdentityObservationAtom(_ObservationAtom):
    _component_reference: ClassVar[str] = "arti/formula-atom-observe-identity@1"


class StateAffineObservationAtom(_ObservationAtom):
    _component_reference: ClassVar[str] = "arti/formula-atom-observe-affine@1"

    def __init__(self, operand_types: tuple[TensorType, ...], *, scale: float = 0.1) -> None:
        super().__init__(operand_types, scale=scale)


class FourierObservationAtom(_ObservationAtom):
    _component_reference: ClassVar[str] = "arti/formula-atom-observe-fourier@1"

    def __init__(
        self,
        operand_types: tuple[TensorType, ...],
        *,
        spatial_shape: tuple[int, int],
        state_mode: str = "cartesian",
        direction_epsilon: float = 1e-6,
        compile_policy: str = "safe_training",
    ) -> None:
        super().__init__(
            operand_types,
            spatial_shape=tuple(spatial_shape),
            state_mode=state_mode,
            direction_epsilon=direction_epsilon,
            compile_policy=compile_policy,
        )


OBSERVATION_ATOM_CLASSES = {
    cls._component_reference: cls
    for cls in (IdentityObservationAtom, StateAffineObservationAtom, FourierObservationAtom)
}


__all__ = [
    "IdentityObservationAtom",
    "StateAffineObservationAtom",
    "FourierObservationAtom",
    "observe_identity",
    "observe_affine",
    "observe_fourier",
    "observation_mask",
]
