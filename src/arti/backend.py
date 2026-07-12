"""Backend discovery helpers."""

from __future__ import annotations

from functools import lru_cache
from importlib.util import find_spec


@lru_cache(maxsize=1)
def jax_backend_status() -> str:
    """Return unavailable, broken, or available for the optional JAX runtime."""

    try:
        if find_spec("jax") is None:
            return "unavailable"
        import jax
        import jaxlib  # noqa: F401

        jax.devices()
    except Exception:
        return "broken"
    return "available"


def available_backends() -> tuple[str, ...]:
    """Return backend namespaces that can be imported today."""

    backends = ["torch"]
    if jax_backend_status() == "available":
        backends.append("jax")
    return tuple(backends)


def planned_backends() -> tuple[str, ...]:
    """Return backend namespaces reserved by the package roadmap."""

    return () if jax_backend_status() == "available" else ("jax",)
