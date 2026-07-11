"""TOML compatibility boundary for supported Python versions."""

from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - exercised on Python 3.10.
    import tomli as tomllib

loads = tomllib.loads

__all__ = ["loads"]
