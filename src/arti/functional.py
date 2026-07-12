"""Pure tensor helpers for masks, visibility, and pooling."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def half(
    x: Tensor,
    *,
    threshold: float | Tensor = 1.0,
    base: float | Tensor = 0.5,
    scale: float | Tensor = 1.0,
    stochastic: bool = False,
    training: bool = False,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Salience-conditioned Half activation.

    ``half`` computes elementwise salience as ``abs(x)``, converts insufficient
    salience into ``D = relu((threshold - salience) / scale)``, then applies
    ``q = base ** D``. Deterministic mode returns ``q * x``. In stochastic
    training mode, ``q`` is used as the survival probability and dropped
    features are set to zero without inverted-dropout rescaling.
    """

    if isinstance(base, Tensor):
        if torch.any(~torch.isfinite(base) | (base <= 0) | (base > 1)):
            raise ValueError("base must be in the interval (0, 1]")
    elif not math.isfinite(base) or not 0 < base <= 1:
        raise ValueError("base must be in the interval (0, 1]")
    if isinstance(scale, Tensor):
        if torch.any(~torch.isfinite(scale) | (scale <= 0)):
            raise ValueError("scale must be positive")
    elif not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be positive")

    if isinstance(threshold, Tensor):
        if torch.any(~torch.isfinite(threshold)):
            raise ValueError("threshold must be finite")
    elif not math.isfinite(threshold):
        raise ValueError("threshold must be finite")

    threshold_t = torch.as_tensor(threshold, device=x.device, dtype=x.dtype)
    base_t = torch.as_tensor(base, device=x.device, dtype=x.dtype)
    scale_t = torch.as_tensor(scale, device=x.device, dtype=x.dtype)
    deficit = torch.relu((threshold_t - x.abs()) / scale_t)
    if not isinstance(base, Tensor) and base == 1.0:
        survival = torch.ones_like(deficit)
    elif not isinstance(base, Tensor):
        survival = torch.exp(deficit * math.log(base))
    else:
        survival = torch.pow(base_t, deficit)
    if stochastic and training:
        try:
            mask = torch.bernoulli(survival, generator=generator)
        except TypeError:
            if generator is not None:
                raise
            mask = torch.bernoulli(survival)
        return mask * x
    return survival * x


def as_sequence(x: Tensor) -> tuple[Tensor, bool]:
    """Return a rank-3 tensor and whether the input was originally rank-2."""

    if x.ndim == 2:
        return x.unsqueeze(1), True
    if x.ndim == 3:
        return x, False
    raise ValueError("x must have shape [B, D] or [B, N, D]")


def restore_input_rank(x: Tensor, was_vector: bool) -> Tensor:
    """Remove the singleton token dimension when the input was [B, D]."""

    return x.squeeze(1) if was_vector else x


def ensure_mask(mask: Tensor | None, batch: int, tokens: int, device: torch.device) -> Tensor:
    """Return a boolean token mask with shape [B, N]."""

    if mask is None:
        return torch.ones(batch, tokens, dtype=torch.bool, device=device)
    if mask.shape != (batch, tokens):
        raise ValueError(f"mask must have shape {(batch, tokens)}, got {tuple(mask.shape)}")
    return mask.to(device=device, dtype=torch.bool)


def ensure_coord(coord: Tensor | None, batch: int, tokens: int, coord_dim: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Return a coordinate tensor with shape [B, N, C]."""

    if coord_dim == 0:
        return torch.empty(batch, tokens, 0, device=device, dtype=dtype)
    if coord is None:
        return torch.zeros(batch, tokens, coord_dim, device=device, dtype=dtype)
    if coord.shape != (batch, tokens, coord_dim):
        raise ValueError(f"coord must have shape {(batch, tokens, coord_dim)}, got {tuple(coord.shape)}")
    return coord.to(device=device, dtype=dtype)


def ensure_visibility(visibility: Tensor | None, mask: Tensor) -> Tensor:
    """Return token-to-token visibility with shape [B, N, N]."""

    batch, tokens = mask.shape
    if visibility is None:
        return mask.unsqueeze(1) & mask.unsqueeze(2)
    if visibility.shape != (batch, tokens, tokens):
        raise ValueError(f"visibility must have shape {(batch, tokens, tokens)}, got {tuple(visibility.shape)}")
    return visibility.to(device=mask.device, dtype=torch.bool) & mask.unsqueeze(1) & mask.unsqueeze(2)


def masked_softmax(logits: Tensor, mask: Tensor | None, dim: int = -1) -> Tensor:
    """Softmax with invalid positions assigned zero probability."""

    if mask is None:
        return torch.softmax(logits, dim=dim)
    mask = mask.to(device=logits.device, dtype=torch.bool)
    masked_logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
    weights = torch.softmax(masked_logits, dim=dim)
    return weights.masked_fill(~mask, 0.0)


def masked_mean(x: Tensor, mask: Tensor | None, dim: int = 1, keepdim: bool = False) -> Tensor:
    """Mean over valid elements only."""

    if mask is None:
        return x.mean(dim=dim, keepdim=keepdim)
    mask = mask.to(device=x.device, dtype=x.dtype)
    while mask.ndim < x.ndim:
        mask = mask.unsqueeze(-1)
    total = (x * mask).sum(dim=dim, keepdim=keepdim)
    count = mask.sum(dim=dim, keepdim=keepdim).clamp_min(1.0)
    return total / count


def apply_coord_frame_inverse(
    x: Tensor,
    coord: Tensor,
    mode: str = "none",
    frame_operators: Tensor | None = None,
    observer_coord: Tensor | None = None,
) -> Tensor:
    """Apply a deterministic coordinate-frame inverse to latent channels.

    When ``observer_coord`` is provided, it defines the active observer frame for
    every token in the context. This is the autoregressive use case: the next
    token's phase is the reference frame, so same-frame context is canonical and
    other-frame context keeps its relative phase difference.
    """

    if mode == "none":
        return x
    active_coord = coord if observer_coord is None else _expand_observer_coord(observer_coord, coord)
    if mode == "operator_bank":
        if frame_operators is None:
            raise ValueError("operator_bank mode requires frame_operators with shape [K, D, D]")
        if frame_operators.ndim != 3 or frame_operators.shape[1:] != (x.shape[-1], x.shape[-1]):
            raise ValueError(f"frame_operators must have shape [K, {x.shape[-1]}, {x.shape[-1]}]")
        if active_coord.shape[-1] != frame_operators.shape[0]:
            raise ValueError(f"operator_bank coord last dim must equal {frame_operators.shape[0]}")
        weights = active_coord.to(device=x.device, dtype=x.dtype)
        operators = frame_operators.to(device=x.device, dtype=x.dtype)
        return torch.einsum("bnk,kde,bne->bnd", weights, operators, x)
    if mode != "paired_rotation":
        raise ValueError("mode must be 'none', 'paired_rotation', or 'operator_bank'")
    if active_coord.shape[-1] < 2:
        raise ValueError("paired_rotation requires coord[..., :2] = [sin(theta), cos(theta)]")
    if x.shape[-1] % 2 != 0:
        raise ValueError("paired_rotation requires an even latent dimension")
    sin_t = active_coord[..., 0]
    cos_t = active_coord[..., 1]
    even = x[..., 0::2]
    odd = x[..., 1::2]
    canonical_even = cos_t.unsqueeze(-1) * even + sin_t.unsqueeze(-1) * odd
    canonical_odd = -sin_t.unsqueeze(-1) * even + cos_t.unsqueeze(-1) * odd
    canonical = torch.empty_like(x)
    canonical[..., 0::2] = canonical_even
    canonical[..., 1::2] = canonical_odd
    return canonical


def _expand_observer_coord(observer_coord: Tensor, coord: Tensor) -> Tensor:
    if observer_coord.ndim == 2:
        observer_coord = observer_coord.unsqueeze(1)
    if observer_coord.ndim != 3:
        raise ValueError("observer_coord must have shape [B, C] or [B, 1, C] or [B, N, C]")
    if observer_coord.shape[0] != coord.shape[0] or observer_coord.shape[-1] != coord.shape[-1]:
        raise ValueError(f"observer_coord must match coord batch and coord dims, got {tuple(observer_coord.shape)} and {tuple(coord.shape)}")
    if observer_coord.shape[1] == 1:
        return observer_coord.expand(-1, coord.shape[1], -1)
    if observer_coord.shape[1] != coord.shape[1]:
        raise ValueError(f"observer_coord token dim must be 1 or {coord.shape[1]}, got {observer_coord.shape[1]}")
    return observer_coord


def mask_coverage(mask: Tensor) -> Tensor:
    """Fraction of valid tokens per batch item."""

    return mask.float().mean(dim=1)
