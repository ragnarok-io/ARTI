"""Optional JAX backend for ARTI.

This backend is intentionally small in 1.0: it provides pure functional
latent tensor transforms that preserve ARTI's tensor contracts. The full
Gradle-like adaptation builder remains PyTorch-first.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from typing import Any


class ARTIJAXBackendNotImplementedError(NotImplementedError):
    """Backward-compatible alias for callers expecting the old planned error."""


class ARTIJAXBackendUnavailableError(ImportError):
    """Raised when JAX APIs are requested without JAX installed."""


@dataclass(frozen=True)
class JAXARTIConfig:
    input_dim: int
    hidden_dim: int
    coord_dim: int = 0


def backend_status() -> str:
    """Return the current status of the JAX backend."""

    return "available" if find_spec("jax") is not None else "unavailable"


def require_jax_backend() -> None:
    """Raise a clear error when optional JAX dependencies are missing."""

    if backend_status() != "available":
        raise ARTIJAXBackendUnavailableError("arti.jax requires optional dependencies; install with `uv sync --extra jax`.")


def smoke_report() -> dict[str, Any]:
    """Run a tiny JAX functional smoke check and return a structured report."""

    if backend_status() != "available":
        return {
            "smoke_status": "skipped",
            "backend_status": "unavailable",
            "init_ok": False,
            "forward_ok": False,
            "jit_ok": False,
            "grad_ok": False,
            "reason": "JAX optional dependencies are not installed.",
        }
    try:
        jax, jnp = _jax_modules()
        key = jax.random.PRNGKey(0)
        params = init_layer(key, input_dim=2, hidden_dim=3, coord_dim=1)
        x = jnp.ones((1, 2, 2))
        coord = jnp.ones((1, 2, 1))
        mask = jnp.array([[True, False]])
        out = apply_layer(params, x, coord=coord, mask=mask)
        forward_ok = tuple(out["y"].shape) == (1, 2, 3) and tuple(out["pooled"].shape) == (1, 3)
        jitted = jax.jit(lambda values: apply_layer(params, values, coord=coord, mask=mask)["pooled"])(x)
        jit_ok = tuple(jitted.shape) == (1, 3)

        def objective(kernel: Any) -> Any:
            updated = dict(params)
            updated["input_kernel"] = kernel
            return jnp.sum(apply_layer(updated, x, coord=coord, mask=mask)["pooled"])

        grad = jax.grad(objective)(params["input_kernel"])
        grad_ok = tuple(grad.shape) == tuple(params["input_kernel"].shape)
    except Exception as exc:  # pragma: no cover - exercised only on broken local JAX installs.
        return {
            "smoke_status": "failed",
            "backend_status": "available",
            "init_ok": False,
            "forward_ok": False,
            "jit_ok": False,
            "grad_ok": False,
            "reason": str(exc),
        }
    return {
        "smoke_status": "passed" if forward_ok and jit_ok and grad_ok else "failed",
        "backend_status": "available",
        "init_ok": True,
        "forward_ok": bool(forward_ok),
        "jit_ok": bool(jit_ok),
        "grad_ok": bool(grad_ok),
        "y_shape": list(out["y"].shape),
        "pooled_shape": list(out["pooled"].shape),
    }


def _jax_modules():
    require_jax_backend()
    import jax
    import jax.numpy as jnp

    return jax, jnp


def init_layer(key: Any, *, input_dim: int, hidden_dim: int, coord_dim: int = 0, scale: float = 0.02) -> dict[str, Any]:
    """Initialize a minimal JAX ARTI latent transform parameter PyTree."""

    jax, jnp = _jax_modules()
    input_key, coord_key = jax.random.split(key)
    params: dict[str, Any] = {
        "config": {"input_dim": int(input_dim), "hidden_dim": int(hidden_dim), "coord_dim": int(coord_dim)},
        "input_kernel": scale * jax.random.normal(input_key, (input_dim, hidden_dim)),
        "bias": jnp.zeros((hidden_dim,)),
    }
    if coord_dim > 0:
        params["coord_kernel"] = scale * jax.random.normal(coord_key, (coord_dim, hidden_dim))
    return params


def masked_softmax(logits: Any, mask: Any | None, axis: int = -1) -> Any:
    """JAX softmax with invalid positions assigned zero probability."""

    _, jnp = _jax_modules()
    values = jnp.asarray(logits)
    if mask is None:
        return jax_nn_softmax(values, axis=axis)
    mask_array = jnp.asarray(mask).astype(bool)
    masked_logits = jnp.where(mask_array, values, jnp.finfo(values.dtype).min)
    weights = jax_nn_softmax(masked_logits, axis=axis)
    return jnp.where(mask_array, weights, 0.0)


def masked_mean(x: Any, mask: Any | None, axis: int = 1, keepdims: bool = False) -> Any:
    """JAX mean over valid mask positions only."""

    _, jnp = _jax_modules()
    values = jnp.asarray(x)
    if mask is None:
        return jnp.mean(values, axis=axis, keepdims=keepdims)
    mask_array = jnp.asarray(mask).astype(values.dtype)
    while mask_array.ndim < values.ndim:
        mask_array = jnp.expand_dims(mask_array, axis=-1)
    total = jnp.sum(values * mask_array, axis=axis, keepdims=keepdims)
    count = jnp.maximum(jnp.sum(mask_array, axis=axis, keepdims=keepdims), 1.0)
    return total / count


def mask_coverage(mask: Any) -> Any:
    """Return the fraction of valid tokens per batch item."""

    _, jnp = _jax_modules()
    return jnp.mean(jnp.asarray(mask).astype(jnp.float32), axis=1)


def ensure_visibility(visibility: Any | None, mask: Any) -> Any:
    """Return token-to-token visibility with shape ``[B, N, N]``.

    Invalid masked tokens are always removed from both source and target sides,
    matching the PyTorch backend contract.
    """

    _, jnp = _jax_modules()
    mask_array = jnp.asarray(mask).astype(bool)
    base = jnp.logical_and(jnp.expand_dims(mask_array, 1), jnp.expand_dims(mask_array, 2))
    if visibility is None:
        return base
    visibility_array = jnp.asarray(visibility).astype(bool)
    return jnp.logical_and(visibility_array, base)


def attention_mask_to_visibility(attention_mask: Any, *, causal: bool = False) -> Any:
    """Convert a JAX attention mask ``[B, N]`` into visibility ``[B, N, N]``."""

    _, jnp = _jax_modules()
    mask_array = jnp.asarray(attention_mask).astype(bool)
    visibility = ensure_visibility(None, mask_array)
    if causal:
        tokens = mask_array.shape[1]
        causal_mask = jnp.tril(jnp.ones((tokens, tokens), dtype=bool))
        visibility = jnp.logical_and(visibility, jnp.expand_dims(causal_mask, 0))
    return visibility


def apply_coord_frame_inverse(
    x: Any,
    coord: Any,
    mode: str = "none",
    frame_operators: Any | None = None,
    observer_coord: Any | None = None,
) -> Any:
    """Apply a JAX coordinate-frame inverse to latent channels.

    ``paired_rotation`` and ``operator_bank`` mirror the PyTorch backend's
    functional contract. When ``observer_coord`` is provided, it defines the
    active reference frame for the whole context.
    """

    _, jnp = _jax_modules()
    values = jnp.asarray(x)
    if mode == "none":
        return values
    coord_array = jnp.asarray(coord)
    active_coord = coord_array if observer_coord is None else _expand_observer_coord(observer_coord, coord_array)
    if mode == "operator_bank":
        if frame_operators is None:
            raise ValueError("operator_bank mode requires frame_operators with shape [K, D, D]")
        operators = jnp.asarray(frame_operators).astype(values.dtype)
        weights = active_coord.astype(values.dtype)
        return jnp.einsum("bnk,kde,bne->bnd", weights, operators, values)
    if mode != "paired_rotation":
        raise ValueError("mode must be 'none', 'paired_rotation', or 'operator_bank'")
    sin_t = active_coord[..., 0]
    cos_t = active_coord[..., 1]
    even = values[..., 0::2]
    odd = values[..., 1::2]
    canonical_even = jnp.expand_dims(cos_t, -1) * even + jnp.expand_dims(sin_t, -1) * odd
    canonical_odd = -jnp.expand_dims(sin_t, -1) * even + jnp.expand_dims(cos_t, -1) * odd
    return jnp.stack((canonical_even, canonical_odd), axis=-1).reshape(values.shape)


def _expand_observer_coord(observer_coord: Any, coord: Any) -> Any:
    _, jnp = _jax_modules()
    observer = jnp.asarray(observer_coord)
    coord_array = jnp.asarray(coord)
    if observer.ndim == 2:
        observer = jnp.expand_dims(observer, 1)
    if observer.shape[1] == 1:
        return jnp.broadcast_to(observer, coord_array.shape)
    return observer


def jax_nn_softmax(values: Any, *, axis: int) -> Any:
    jax, _ = _jax_modules()
    return jax.nn.softmax(values, axis=axis)


def apply_layer(params: dict[str, Any], x: Any, *, coord: Any | None = None, mask: Any | None = None) -> dict[str, Any]:
    """Apply a minimal JAX ARTI latent transform.

    Supports ``x`` as ``[B, D]`` or ``[B, N, D]``. ``mask`` may be ``[B, N]``.
    Returns a dictionary with ``y``, ``pooled``, and ``diagnostics``.
    """

    _, jnp = _jax_modules()
    config = params["config"]
    x_array = jnp.asarray(x)
    was_vector = x_array.ndim == 2
    z = x_array[:, None, :] if was_vector else x_array
    y = jnp.einsum("bnd,dh->bnh", z, params["input_kernel"]) + params["bias"]
    if coord is not None and int(config.get("coord_dim", 0)) > 0:
        coord_array = jnp.asarray(coord)
        coord_seq = coord_array[:, None, :] if coord_array.ndim == 2 else coord_array
        y = y + jnp.einsum("bnc,ch->bnh", coord_seq, params["coord_kernel"])
    y = jnp.tanh(y)
    if mask is None:
        pooled = masked_mean(y, None, axis=1)
        mask_coverage = jnp.ones((z.shape[0],))
    else:
        mask_seq = jnp.asarray(mask)
        mask_seq = mask_seq[:, None] if mask_seq.ndim == 1 else mask_seq
        pooled = masked_mean(y, mask_seq, axis=1)
        mask_coverage = globals()["mask_coverage"](mask_seq)
    return {
        "y": y[:, 0, :] if was_vector else y,
        "pooled": pooled,
        "diagnostics": {"mask_coverage": mask_coverage},
    }


__all__ = [
    "ARTIJAXBackendNotImplementedError",
    "ARTIJAXBackendUnavailableError",
    "JAXARTIConfig",
    "apply_layer",
    "apply_coord_frame_inverse",
    "attention_mask_to_visibility",
    "backend_status",
    "ensure_visibility",
    "init_layer",
    "mask_coverage",
    "masked_mean",
    "masked_softmax",
    "require_jax_backend",
    "smoke_report",
]
