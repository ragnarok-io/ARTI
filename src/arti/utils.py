"""Validation helpers for ARTI modules."""

from __future__ import annotations

import torch
from torch import Tensor


def assert_floating_tensor(name: str, value: Tensor) -> None:
    if not torch.is_floating_point(value):
        raise TypeError(f"{name} must be a floating point tensor")


def detach_diagnostics(diagnostics: dict[str, Tensor]) -> dict[str, Tensor]:
    """Detach diagnostics so callers can inspect them without extending graphs."""

    return {key: value.detach() for key, value in diagnostics.items()}
