"""Latent distinctness reports for compression and pulse validation."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class LatentDistinctnessReport:
    """Pairwise distinctness report for raw and transformed tensors."""

    count: int
    raw_min_distance: float
    latent_min_distance: float
    distance_retention: float
    raw_collision_count: int
    latent_collision_count: int
    latent_entropy_bits: float
    rank_correlation: float
    nearest_neighbor_consistency: float
    latent_collisions: tuple[tuple[int, int], ...]
    weakest_pair: tuple[int, int] | None


def latent_distinctness_report(
    raw: Tensor,
    latent: Tensor,
    *,
    raw_atol: float = 0.0,
    latent_atol: float = 0.0,
    entropy_bins: int = 256,
) -> LatentDistinctnessReport:
    """Compare whether ``latent`` preserves distinctions present in ``raw``.

    Both inputs use the first dimension as the candidate axis and flatten all
    remaining dimensions. This is intentionally domain-free: it can validate
    pulse outputs, pooled representations, image embeddings, or any other
    tensor compression.
    """

    raw_flat = _flatten_candidates(raw)
    latent_flat = _flatten_candidates(latent)
    if raw_flat.shape[0] != latent_flat.shape[0]:
        raise ValueError("raw and latent must have the same candidate count")
    count = raw_flat.shape[0]
    if count == 0:
        return LatentDistinctnessReport(0, 0.0, 0.0, 0.0, 0, 0, 0.0, 0.0, 0.0, (), None)

    raw_dist = torch.cdist(raw_flat, raw_flat, p=2)
    latent_dist = torch.cdist(latent_flat, latent_flat, p=2)
    raw_collisions = _collision_pairs(raw_dist, raw_atol)
    latent_collisions = _collision_pairs(latent_dist, latent_atol)
    raw_min, raw_weakest = _min_pair(raw_dist)
    latent_min, latent_weakest = _min_pair(latent_dist)
    retention = 0.0 if raw_min <= 0.0 else latent_min / raw_min
    return LatentDistinctnessReport(
        count=count,
        raw_min_distance=raw_min,
        latent_min_distance=latent_min,
        distance_retention=retention,
        raw_collision_count=len(raw_collisions),
        latent_collision_count=len(latent_collisions),
        latent_entropy_bits=_quantized_entropy_bits(latent_flat, bins=entropy_bins),
        rank_correlation=_distance_rank_correlation(raw_dist, latent_dist),
        nearest_neighbor_consistency=_nearest_neighbor_consistency(raw_dist, latent_dist),
        latent_collisions=tuple(latent_collisions),
        weakest_pair=latent_weakest,
    )


def assert_latent_distinct(
    raw: Tensor,
    latent: Tensor,
    *,
    min_latent_distance: float = 0.0,
    min_distance_retention: float = 0.0,
    max_collision_atol: float = 0.0,
    min_entropy_bits: float = 0.0,
) -> LatentDistinctnessReport:
    """Raise if latent compression loses required candidate distinctions."""

    report = latent_distinctness_report(raw, latent, latent_atol=max_collision_atol)
    if report.latent_collision_count:
        raise ValueError(f"latent has {report.latent_collision_count} collisions: {report.latent_collisions[:5]}")
    if report.count > 1 and report.latent_min_distance <= min_latent_distance:
        raise ValueError(f"latent min distance {report.latent_min_distance:.8f} <= {min_latent_distance:.8f}")
    if report.distance_retention < min_distance_retention:
        raise ValueError(f"distance retention {report.distance_retention:.6f} < {min_distance_retention:.6f}")
    if report.latent_entropy_bits < min_entropy_bits:
        raise ValueError(f"latent entropy {report.latent_entropy_bits:.4f} < {min_entropy_bits:.4f}")
    return report


def _flatten_candidates(tensor: Tensor) -> Tensor:
    if tensor.ndim < 2:
        raise ValueError("tensor must have shape [K, ...]")
    return tensor.detach().to(dtype=torch.float32).flatten(start_dim=1)


def _collision_pairs(distances: Tensor, atol: float) -> list[tuple[int, int]]:
    count = distances.shape[0]
    return [(i, j) for i in range(count) for j in range(i + 1, count) if float(distances[i, j].item()) <= atol]


def _min_pair(distances: Tensor) -> tuple[float, tuple[int, int] | None]:
    count = distances.shape[0]
    if count < 2:
        return float("inf"), None
    upper = torch.triu(torch.ones(count, count, device=distances.device, dtype=torch.bool), diagonal=1)
    values = distances[upper]
    index = int(values.argmin().item())
    pairs = upper.nonzero(as_tuple=False)
    pair = pairs[index]
    return float(values[index].item()), (int(pair[0].item()), int(pair[1].item()))


def _distance_rank_correlation(raw_dist: Tensor, latent_dist: Tensor) -> float:
    count = raw_dist.shape[0]
    if count < 3:
        return 1.0
    upper = torch.triu(torch.ones(count, count, device=raw_dist.device, dtype=torch.bool), diagonal=1)
    raw = raw_dist[upper]
    latent = latent_dist[upper]
    raw_centered = raw - raw.mean()
    latent_centered = latent - latent.mean()
    denom = raw_centered.norm() * latent_centered.norm()
    if float(denom.item()) == 0.0:
        return 0.0
    return float((raw_centered * latent_centered).sum().div(denom).item())


def _nearest_neighbor_consistency(raw_dist: Tensor, latent_dist: Tensor) -> float:
    count = raw_dist.shape[0]
    if count < 2:
        return 1.0
    raw_masked = raw_dist.clone()
    latent_masked = latent_dist.clone()
    raw_masked.fill_diagonal_(float("inf"))
    latent_masked.fill_diagonal_(float("inf"))
    return float((raw_masked.argmin(dim=1) == latent_masked.argmin(dim=1)).to(torch.float32).mean().item())


def _quantized_entropy_bits(flat: Tensor, *, bins: int) -> float:
    if bins <= 1:
        raise ValueError("bins must be greater than 1")
    values = flat
    min_value = values.min()
    max_value = values.max()
    span = (max_value - min_value).clamp_min(torch.finfo(values.dtype).eps)
    scaled = (values - min_value) / span
    quantized = torch.clamp((scaled * (bins - 1)).round().to(torch.long), 0, bins - 1)
    counts = torch.bincount(quantized.reshape(-1), minlength=bins).to(torch.float32)
    probs = counts[counts > 0] / counts.sum().clamp_min(1.0)
    return float((-(probs * torch.log2(probs)).sum()).item())
