"""Prepared heterogeneous storage; frame handles stay opaque int64 values."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class FormulaDevicePoolLayout:
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[torch.dtype, ...]
    capacities: tuple[int, ...]
    offsets: tuple[int, ...]

    @classmethod
    def from_samples(cls, samples, capacities):
        samples = tuple(samples)
        capacities = (
            (capacities,) * len(samples) if isinstance(capacities, int) else tuple(capacities)
        )
        if (
            not samples
            or len(samples) != len(capacities)
            or any(type(c) is not int or c < 1 for c in capacities)
        ):
            raise ValueError("typed pools need one positive capacity per sample")
        keys = tuple((tuple(value.shape), value.dtype) for value in samples)
        if len(set(keys)) != len(keys):
            raise ValueError("typed pool shapes/dtypes must be unique")
        offsets, offset = [], 0
        for capacity in capacities:
            offsets.append(offset)
            offset += capacity + 1
        return cls(
            tuple(key[0] for key in keys), tuple(key[1] for key in keys), capacities, tuple(offsets)
        )

    def index(self, value: Tensor) -> int:
        return tuple(zip(self.shapes, self.dtypes, strict=True)).index(
            (tuple(value.shape), value.dtype)
        )

    def allocate(self, device):
        return tuple(
            torch.zeros((capacity + 1, *shape), dtype=dtype, device=device)
            for shape, dtype, capacity in zip(
                self.shapes, self.dtypes, self.capacities, strict=True
            )
        )

    def contains(self, handles: Tensor, bucket: int) -> Tensor:
        return (handles >= self.offsets[bucket]) & (
            handles < self.offsets[bucket] + self.capacities[bucket]
        )

    def gather(self, pools, handles: Tensor, bucket: int) -> Tensor:
        local = handles - self.offsets[bucket]
        safe = torch.where(
            self.contains(handles, bucket), local, torch.full_like(local, self.capacities[bucket])
        )
        return pools[bucket].index_select(0, safe)
