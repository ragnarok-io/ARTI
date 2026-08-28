"""Explicit tensor context contracts for ARTI layers.

The legacy layer arguments remain supported for compatibility.  New code can
pass :class:`TensorContext` to make validity, visibility, and observation-frame
semantics explicit and strictly validated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor


FrameMode = Literal["none", "paired_rotation", "operator_bank"]
_FRAME_MODES = frozenset({"none", "paired_rotation", "operator_bank"})


def _require_tensor(name: str, value: object) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    return value


def _require_floating_tensor(name: str, value: object, *, device: torch.device, dtype: torch.dtype) -> Tensor:
    tensor = _require_tensor(name, value)
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype")
    if tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")
    if not bool(torch.isfinite(tensor).all()):
        raise ValueError(f"{name} must contain only finite values")
    return tensor


def validate_valid_mask(
    valid_mask: Tensor | None,
    *,
    batch: int,
    tokens: int,
    device: torch.device,
) -> Tensor:
    """Validate the explicit validity mask without coercing its dtype/device."""

    if valid_mask is None:
        return torch.ones(batch, tokens, dtype=torch.bool, device=device)
    mask = _require_tensor("valid_mask", valid_mask)
    if mask.shape != (batch, tokens):
        raise ValueError(
            f"valid_mask must have shape {(batch, tokens)}, got {tuple(mask.shape)}"
        )
    if mask.dtype is not torch.bool:
        raise TypeError(f"valid_mask must have dtype torch.bool, got {mask.dtype}")
    if mask.device != device:
        raise ValueError(f"valid_mask must be on {device}, got {mask.device}")
    return mask


def validate_visibility(
    visibility: Tensor | None,
    *,
    valid_mask: Tensor,
) -> Tensor | None:
    """Validate query-to-source visibility and return its effective mask.

    ``visibility`` is kept conceptually separate from ``valid_mask``.  The
    returned tensor is the computational intersection used by attention-like
    readers; callers should retain the original visibility tensor when they
    need to inspect authorization separately.
    """

    if visibility is None:
        return None
    visible = _require_tensor("visibility", visibility)
    batch, tokens = valid_mask.shape
    expected = (batch, tokens, tokens)
    if visible.shape != expected:
        raise ValueError(f"visibility must have shape {expected}, got {tuple(visible.shape)}")
    if visible.dtype is not torch.bool:
        raise TypeError(f"visibility must have dtype torch.bool, got {visible.dtype}")
    if visible.device != valid_mask.device:
        raise ValueError(
            f"visibility must be on {valid_mask.device}, got {visible.device}"
        )
    return visible & valid_mask.unsqueeze(1) & valid_mask.unsqueeze(2)


@dataclass(frozen=True)
class FrameContext:
    """Observation-frame tensors supplied to one ARTI invocation.

    ``coord`` identifies the frame in which each token was observed.
    ``observer_coord`` optionally identifies the frame of the current observer
    (for example, the token being generated).  ``frame_operators`` contains
    the already-defined operators used by ``operator_bank`` mode; ARTI does
    not infer or learn their inverse at runtime.
    """

    coord: Tensor | None = None
    observer_coord: Tensor | None = None
    frame_operators: Tensor | None = None
    mode: FrameMode | None = None
    rotation_tolerance: float = 1e-4

    def validate(
        self,
        *,
        batch: int,
        tokens: int,
        hidden_dim: int,
        coord_dim: int,
        configured_mode: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        if configured_mode not in _FRAME_MODES:
            raise ValueError(f"configured frame mode is unsupported: {configured_mode!r}")
        if self.mode is not None:
            if self.mode not in _FRAME_MODES:
                raise ValueError(f"frame.mode is unsupported: {self.mode!r}")
            if self.mode != configured_mode:
                raise ValueError(
                    "frame.mode must match the layer's coord_frame_mode: "
                    f"{self.mode!r} != {configured_mode!r}"
                )
        if isinstance(self.rotation_tolerance, bool) or not isinstance(
            self.rotation_tolerance, (int, float)
        ):
            raise TypeError("frame.rotation_tolerance must be a non-negative number")
        if self.rotation_tolerance < 0:
            raise ValueError("frame.rotation_tolerance must be non-negative")

        if self.coord is not None:
            coord = _require_floating_tensor(
                "frame.coord",
                self.coord,
                device=device,
                dtype=dtype,
            )
            expected = (batch, tokens, coord_dim)
            if coord.shape != expected:
                raise ValueError(
                    f"frame.coord must have shape {expected}, got {tuple(coord.shape)}"
                )
        elif coord_dim > 0 and configured_mode != "none":
            raise ValueError(
                "frame.coord is required when a coordinate-frame inverse is enabled"
            )

        if self.observer_coord is not None:
            if configured_mode == "none":
                raise ValueError(
                    "frame.observer_coord requires an enabled coordinate-frame inverse"
                )
            observer = _require_floating_tensor(
                "frame.observer_coord",
                self.observer_coord,
                device=device,
                dtype=dtype,
            )
            valid_shapes = {(batch, coord_dim), (batch, 1, coord_dim), (batch, tokens, coord_dim)}
            if tuple(observer.shape) not in valid_shapes:
                raise ValueError(
                    "frame.observer_coord must have shape "
                    f"[B, C], [B, 1, C], or [B, N, C] with C={coord_dim}; "
                    f"got {tuple(observer.shape)}"
                )
            if self.coord is None:
                raise ValueError("frame.observer_coord requires frame.coord")

        if self.frame_operators is not None:
            operators = _require_floating_tensor(
                "frame.frame_operators",
                self.frame_operators,
                device=device,
                dtype=dtype,
            )
            expected = (coord_dim, hidden_dim, hidden_dim)
            if operators.shape != expected:
                raise ValueError(
                    "frame.frame_operators must have shape "
                    f"{expected}, got {tuple(operators.shape)}"
                )
        if configured_mode == "operator_bank" and self.frame_operators is None:
            raise ValueError("operator_bank mode requires frame.frame_operators")
        if configured_mode != "operator_bank" and self.frame_operators is not None:
            raise ValueError(
                "frame.frame_operators is only valid when coord_frame_mode='operator_bank'"
            )

        if configured_mode == "paired_rotation":
            if coord_dim < 2:
                raise ValueError("paired_rotation requires coord_dim >= 2")
            if hidden_dim % 2:
                raise ValueError("paired_rotation requires an even hidden dimension")
            self._validate_rotation(self.coord, "frame.coord")
            self._validate_rotation(self.observer_coord, "frame.observer_coord")

    def _validate_rotation(self, coord: Tensor | None, name: str) -> None:
        if coord is None:
            return
        norm_sq = coord[..., 0].square() + coord[..., 1].square()
        if not bool(
            torch.allclose(
                norm_sq,
                torch.ones_like(norm_sq),
                atol=float(self.rotation_tolerance),
                rtol=float(self.rotation_tolerance),
            )
        ):
            raise ValueError(
                f"{name}[..., :2] must be a unit rotation [sin(theta), cos(theta)]"
            )


@dataclass(frozen=True)
class TensorContext:
    """Strict tensor-side context for one ARTI layer call.

    ``valid_mask`` answers whether a token exists. ``visibility`` answers
    which valid source tokens may influence each valid query token.  Neither
    field is inferred from token names or business objects.
    """

    valid_mask: Tensor | None = None
    visibility: Tensor | None = None
    frame: FrameContext | None = None

    def validate(
        self,
        *,
        batch: int,
        tokens: int,
        hidden_dim: int,
        coord_dim: int,
        configured_mode: str,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[Tensor, Tensor | None]:
        mask = validate_valid_mask(
            self.valid_mask,
            batch=batch,
            tokens=tokens,
            device=device,
        )
        effective_visibility = validate_visibility(
            self.visibility,
            valid_mask=mask,
        )
        frame = self.frame if self.frame is not None else FrameContext()
        frame.validate(
            batch=batch,
            tokens=tokens,
            hidden_dim=hidden_dim,
            coord_dim=coord_dim,
            configured_mode=configured_mode,
            device=device,
            dtype=dtype,
        )
        return mask, effective_visibility


__all__ = [
    "FrameContext",
    "FrameMode",
    "TensorContext",
    "validate_valid_mask",
    "validate_visibility",
]
