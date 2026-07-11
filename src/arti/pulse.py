"""Legacy explicit pulse compression for tokenization-stable latent streams.

This module keeps deterministic, externally indexed pulse compression for
compatibility. The default neural pulse layer is ``arti.nn.Pulse``; this legacy
path maps an external token stream onto pulse ids supplied by the caller.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .distinctness import LatentDistinctnessReport, assert_latent_distinct, latent_distinctness_report


@dataclass
class PulseOutput:
    """Output of pulse compression."""

    pulse: Tensor
    mask: Tensor
    coverage: Tensor
    start: Tensor
    end: Tensor


class PulseCompressor(torch.nn.Module):
    """Legacy explicit compressor for fixed pulse slots by weighted averaging.

    Parameters are intentionally absent in the alpha layer. It is a tensor
    contract layer: token features and token weights define how the stream is
    resampled into pulse coordinates.
    """

    def forward(
        self,
        x: Tensor,
        pulse_ids: Tensor,
        *,
        mask: Tensor | None = None,
        token_weight: Tensor | None = None,
        pulse_count: int | None = None,
    ) -> PulseOutput:
        return pulse_compress(x, pulse_ids, mask=mask, token_weight=token_weight, pulse_count=pulse_count)


def pulse_compress(
    x: Tensor,
    pulse_ids: Tensor,
    *,
    mask: Tensor | None = None,
    token_weight: Tensor | None = None,
    pulse_count: int | None = None,
) -> PulseOutput:
    """Compress ``x`` with shape ``[B, T, D]`` into ``[B, P, D]``.

    ``pulse_ids`` has shape ``[B, T]``. Negative ids are ignored. ``token_weight``
    can encode how much raw external span each token covers, which is important
    when different runtime vocabularies segment the same stream differently.
    """

    if x.ndim != 3:
        raise ValueError("x must have shape [B, T, D]")
    if pulse_ids.shape != x.shape[:2]:
        raise ValueError("pulse_ids must have shape [B, T]")
    batch, tokens, dim = x.shape
    device = x.device
    ids = pulse_ids.to(device=device, dtype=torch.long)
    valid = ids >= 0
    if mask is not None:
        if mask.shape != x.shape[:2]:
            raise ValueError("mask must have shape [B, T]")
        valid = valid & mask.to(device=device, dtype=torch.bool)
    if pulse_count is None:
        pulse_count = int(ids[valid].max().item()) + 1 if bool(valid.any()) else 0
    if pulse_count < 0:
        raise ValueError("pulse_count must be non-negative")
    if pulse_count > tokens:
        raise ValueError("pulse_count cannot exceed the input token count")
    if bool(valid.any()) and int(ids[valid].max().item()) >= pulse_count:
        raise ValueError("valid pulse_ids must be smaller than pulse_count")

    weights = torch.ones(batch, tokens, device=device, dtype=x.dtype) if token_weight is None else token_weight.to(device=device, dtype=x.dtype)
    if weights.shape != x.shape[:2]:
        raise ValueError("token_weight must have shape [B, T]")
    weights = weights * valid.to(dtype=x.dtype)
    safe_ids = ids.clamp_min(0)

    pulse = x.new_zeros(batch, pulse_count, dim)
    coverage = x.new_zeros(batch, pulse_count)
    pulse.scatter_add_(1, safe_ids.unsqueeze(-1).expand(-1, -1, dim), x * weights.unsqueeze(-1))
    coverage.scatter_add_(1, safe_ids, weights)
    pulse = pulse / coverage.clamp_min(torch.finfo(x.dtype).eps).unsqueeze(-1)
    pulse_mask = coverage > 0

    positions = torch.arange(tokens, device=device).expand(batch, tokens)
    large = torch.full((batch, pulse_count), tokens, device=device, dtype=torch.long)
    start = large.scatter_reduce(1, safe_ids, torch.where(valid, positions, tokens), reduce="amin", include_self=True)
    end = torch.zeros(batch, pulse_count, device=device, dtype=torch.long)
    end.scatter_reduce_(1, safe_ids, torch.where(valid, positions + 1, 0), reduce="amax", include_self=True)
    start = torch.where(pulse_mask, start, -1)
    end = torch.where(pulse_mask, end, -1)
    return PulseOutput(pulse=pulse, mask=pulse_mask, coverage=coverage, start=start, end=end)


def fixed_width_pulse_ids(lengths: Tensor, *, pulse_width: int) -> Tensor:
    """Build pulse ids for padded streams from per-row valid lengths."""

    if lengths.ndim != 1:
        raise ValueError("lengths must have shape [B]")
    if pulse_width <= 0:
        raise ValueError("pulse_width must be positive")
    max_len = int(lengths.max().item()) if lengths.numel() else 0
    positions = torch.arange(max_len, device=lengths.device).expand(lengths.numel(), max_len)
    ids = positions // pulse_width
    return torch.where(positions < lengths[:, None], ids, torch.full_like(ids, -1))


def pulse_distinctness_report(raw: Tensor, pulse: Tensor, *, pulse_atol: float = 0.0, entropy_bins: int = 256) -> LatentDistinctnessReport:
    """Report whether pulse compression preserves candidate distinctions."""

    return latent_distinctness_report(raw, pulse, latent_atol=pulse_atol, entropy_bins=entropy_bins)


def assert_pulse_distinct(
    raw: Tensor,
    pulse: Tensor,
    *,
    min_pulse_distance: float = 0.0,
    min_distance_retention: float = 0.0,
    max_collision_atol: float = 0.0,
    min_entropy_bits: float = 0.0,
) -> LatentDistinctnessReport:
    """Raise if pulse compression collapses distinguishable candidates."""

    return assert_latent_distinct(
        raw,
        pulse,
        min_latent_distance=min_pulse_distance,
        min_distance_retention=min_distance_retention,
        max_collision_atol=max_collision_atol,
        min_entropy_bits=min_entropy_bits,
    )
