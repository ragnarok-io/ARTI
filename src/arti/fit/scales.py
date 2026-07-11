"""Scale presets for ARTI adapters."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdapterScale:
    hidden_multiplier: float = 1.0
    interface_slots: int = 8
    recall_slots: int = 4
    recall_steps: int = 0
    recall_activation: str = "half"
    operator_count: int = 4


SCALE_PRESETS = {
    "tiny": AdapterScale(hidden_multiplier=1.0, interface_slots=4, recall_slots=1, recall_steps=0, operator_count=2),
    "small": AdapterScale(hidden_multiplier=1.0, interface_slots=8, recall_slots=4, recall_steps=0, operator_count=4),
    "base": AdapterScale(hidden_multiplier=2.0, interface_slots=16, recall_slots=8, recall_steps=1, operator_count=4),
    "large": AdapterScale(hidden_multiplier=4.0, interface_slots=32, recall_slots=16, recall_steps=1, operator_count=8),
}


def resolve_scale(scale: str | AdapterScale | None) -> AdapterScale:
    if scale is None:
        return SCALE_PRESETS["small"]
    if isinstance(scale, AdapterScale):
        return scale
    if scale in SCALE_PRESETS:
        return SCALE_PRESETS[scale]
    raise ValueError(f"unknown ARTI scale {scale!r}; expected one of {sorted(SCALE_PRESETS)}")
