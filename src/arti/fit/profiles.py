"""Named adaptation profiles."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AdapterProfile:
    name: str = "latent-adapt"
    coord_frame_mode: str = "none"
    coord_dim: int = 0
    virtual_recall: bool = False
    observer_phase: bool = False


PROFILE_PRESETS = {
    "latent-adapt": AdapterProfile(name="latent-adapt"),
    "virtual-recall": AdapterProfile(name="virtual-recall", virtual_recall=True),
    "observer-phase": AdapterProfile(name="observer-phase", coord_frame_mode="operator_bank", observer_phase=True),
    "autoregressive-observer": AdapterProfile(name="autoregressive-observer", coord_frame_mode="operator_bank", observer_phase=True),
}


def resolve_profile(profile: str | AdapterProfile | None, *, phases: int | None = None) -> AdapterProfile:
    if profile is None:
        base = PROFILE_PRESETS["latent-adapt"]
    elif isinstance(profile, AdapterProfile):
        base = profile
    elif profile in PROFILE_PRESETS:
        base = PROFILE_PRESETS[profile]
    else:
        raise ValueError(f"unknown ARTI profile {profile!r}; expected one of {sorted(PROFILE_PRESETS)}")
    if base.observer_phase:
        return AdapterProfile(
            name=base.name,
            coord_frame_mode=base.coord_frame_mode,
            coord_dim=int(phases or base.coord_dim or 16),
            virtual_recall=base.virtual_recall,
            observer_phase=base.observer_phase,
        )
    return base
