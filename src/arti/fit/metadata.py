"""Machine-readable fit capability metadata."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .plugins import PLUGIN_REGISTRY
from .profiles import PROFILE_PRESETS, resolve_profile
from .scales import SCALE_PRESETS


def list_profiles(*, phases: int | None = None) -> dict[str, dict[str, Any]]:
    """Return available ARTI fit profiles as JSON-friendly metadata."""

    return {name: asdict(resolve_profile(name, phases=phases)) for name in sorted(PROFILE_PRESETS)}


def list_scales() -> dict[str, dict[str, Any]]:
    """Return available ARTI adapter scale presets."""

    return {name: asdict(SCALE_PRESETS[name]) for name in sorted(SCALE_PRESETS)}


def list_plugins() -> dict[str, dict[str, object]]:
    """Return registered ARTI fit plugins and optional dependency status."""

    return {name: PLUGIN_REGISTRY[name].to_dict() for name in sorted(PLUGIN_REGISTRY)}


def capabilities(*, phases: int | None = None) -> dict[str, object]:
    """Return a complete machine-readable ARTI fit capability report."""

    return {
        "format_version": 1,
        "package_name": "arti",
        "kind": "capabilities",
        "profiles": list_profiles(phases=phases),
        "scales": list_scales(),
        "plugins": list_plugins(),
    }
