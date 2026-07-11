"""Backend discovery helpers."""

from __future__ import annotations

from importlib.util import find_spec


def available_backends() -> tuple[str, ...]:
    """Return backend namespaces that can be imported today."""

    backends = ["torch"]
    if find_spec("jax") is not None:
        backends.append("jax")
    return tuple(backends)


def planned_backends() -> tuple[str, ...]:
    """Return backend namespaces reserved by the package roadmap."""

    return () if find_spec("jax") is not None else ("jax",)
