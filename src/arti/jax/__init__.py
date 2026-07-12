"""Optional functional JAX subset for ARTI.

The JAX namespace implements a deliberately small tensor contract. It is not a
second implementation of ARTI modules, attachment, fitting, or Recall.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..backend import jax_backend_status


class ARTIJAXBackendNotImplementedError(NotImplementedError):
    """Backward-compatible alias for callers expecting the old planned error."""


class ARTIJAXBackendUnavailableError(ImportError):
    """Raised when JAX APIs are requested without a working JAX runtime."""


@dataclass(frozen=True)
class JAXARTIConfig:
    """Static shape configuration for the minimal functional JAX layer."""

    input_dim: int
    hidden_dim: int
    coord_dim: int = 0

    def __post_init__(self) -> None:
        if self.input_dim <= 0 or self.hidden_dim <= 0:
            raise ValueError("input_dim and hidden_dim must be positive")
        if self.coord_dim < 0:
            raise ValueError("coord_dim must be non-negative")


def backend_status() -> str:
    """Return ``unavailable``, ``broken``, or ``available`` for JAX."""

    return jax_backend_status()


def require_jax_backend() -> None:
    """Raise a clear error unless JAX imports and initializes successfully."""

    status = backend_status()
    if status == "unavailable":
        raise ARTIJAXBackendUnavailableError("arti.jax requires optional dependencies; install with `uv sync --extra jax`.")
    if status == "broken":
        raise ARTIJAXBackendUnavailableError("arti.jax is installed but its runtime could not be initialized; check jaxlib and platform compatibility.")


def _jax_modules():
    require_jax_backend()
    import jax
    import jax.numpy as jnp

    return jax, jnp


def smoke_report() -> dict[str, Any]:
    """Verify forward, JIT, whole-tree gradients, and VMAP semantics."""

    status = backend_status()
    if status == "unavailable":
        return {
            "smoke_status": "skipped",
            "backend_status": status,
            "init_ok": False,
            "forward_ok": False,
            "jit_ok": False,
            "grad_ok": False,
            "vmap_ok": False,
            "reason": "JAX optional dependencies are not installed.",
        }
    if status == "broken":
        return {
            "smoke_status": "failed",
            "backend_status": status,
            "init_ok": False,
            "forward_ok": False,
            "jit_ok": False,
            "grad_ok": False,
            "vmap_ok": False,
            "reason": "JAX is installed but its runtime could not be initialized.",
        }
    checks = {"init_ok": False, "forward_ok": False, "jit_ok": False, "grad_ok": False, "vmap_ok": False}
    try:
        jax, jnp = _jax_modules()
        params = init_layer(jax.random.PRNGKey(0), input_dim=2, hidden_dim=3, coord_dim=1)
        checks["init_ok"] = True
        x = jnp.ones((2, 2, 2), dtype=jnp.float32)
        coord = jnp.ones((2, 2, 1), dtype=jnp.float32)
        mask = jnp.array([[True, False], [True, True]])
        out = apply_layer(params, x, coord=coord, mask=mask)
        checks["forward_ok"] = out["y"].shape == (2, 2, 3) and out["pooled"].shape == (2, 3)
        jitted = jax.jit(lambda p, values, phase, valid: apply_layer(p, values, coord=phase, mask=valid)["pooled"])(params, x, coord, mask)
        checks["jit_ok"] = jitted.shape == (2, 3) and bool(jnp.all(jnp.isfinite(jitted)))

        def objective(p: dict[str, Any]) -> Any:
            return jnp.sum(apply_layer(p, x, coord=coord, mask=mask)["pooled"])

        _, gradients = jax.jit(jax.value_and_grad(objective))(params)
        checks["grad_ok"] = all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(gradients))
        mapped = jax.vmap(lambda values, phase, valid: apply_layer_single(params, values, coord=phase, mask=valid))(x, coord, mask)
        checks["vmap_ok"] = bool(jnp.allclose(mapped["y"], out["y"])) and bool(jnp.allclose(mapped["pooled"], out["pooled"]))
    except Exception as exc:  # pragma: no cover - exercised only on broken local JAX installs.
        return {
            "smoke_status": "failed",
            "backend_status": status,
            **checks,
            "reason": str(exc),
        }
    passed = all(checks.values())
    return {
        "smoke_status": "passed" if passed else "failed",
        "backend_status": status,
        **checks,
        "y_shape": list(out["y"].shape),
        "pooled_shape": list(out["pooled"].shape),
    }


def init_layer(key: Any, *, input_dim: int, hidden_dim: int, coord_dim: int = 0, scale: float = 0.02) -> dict[str, Any]:
    """Initialize an array-only parameter PyTree for the minimal JAX layer."""

    config = JAXARTIConfig(int(input_dim), int(hidden_dim), int(coord_dim))
    jax, jnp = _jax_modules()
    input_key, coord_key = jax.random.split(key)
    params: dict[str, Any] = {
        "input_kernel": scale * jax.random.normal(input_key, (config.input_dim, config.hidden_dim)),
        "bias": jnp.zeros((config.hidden_dim,)),
    }
    if config.coord_dim > 0:
        params["coord_kernel"] = scale * jax.random.normal(coord_key, (config.coord_dim, config.hidden_dim))
    return params


def masked_softmax(logits: Any, mask: Any | None, axis: int = -1) -> Any:
    """JAX softmax with invalid positions assigned zero probability."""

    _, jnp = _jax_modules()
    values = jnp.asarray(logits)
    if mask is None:
        return jax_nn_softmax(values, axis=axis)
    mask_array = jnp.asarray(mask).astype(bool)
    try:
        jnp.broadcast_shapes(values.shape, mask_array.shape)
    except ValueError as exc:
        raise ValueError(f"mask shape {mask_array.shape} is not broadcastable to logits shape {values.shape}") from exc
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
    try:
        jnp.broadcast_shapes(values.shape, mask_array.shape)
    except ValueError as exc:
        raise ValueError(f"mask shape {mask_array.shape} is not broadcastable to input shape {values.shape}") from exc
    total = jnp.sum(values * mask_array, axis=axis, keepdims=keepdims)
    count = jnp.maximum(jnp.sum(mask_array, axis=axis, keepdims=keepdims), 1.0)
    return total / count


def mask_coverage(mask: Any) -> Any:
    """Return the fraction of valid tokens per batch item."""

    _, jnp = _jax_modules()
    mask_array = jnp.asarray(mask)
    if mask_array.ndim != 2:
        raise ValueError("mask must have shape [B, N]")
    return jnp.mean(mask_array.astype(jnp.float32), axis=1)


def ensure_visibility(visibility: Any | None, mask: Any) -> Any:
    """Return token-to-token visibility with shape ``[B, N, N]``."""

    _, jnp = _jax_modules()
    mask_array = jnp.asarray(mask).astype(bool)
    if mask_array.ndim != 2:
        raise ValueError("mask must have shape [B, N]")
    batch, tokens = mask_array.shape
    base = jnp.logical_and(jnp.expand_dims(mask_array, 1), jnp.expand_dims(mask_array, 2))
    if visibility is None:
        return base
    visibility_array = jnp.asarray(visibility).astype(bool)
    if visibility_array.shape != (batch, tokens, tokens):
        raise ValueError(f"visibility must have shape {(batch, tokens, tokens)}, got {visibility_array.shape}")
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
    """Apply the shared ARTI coordinate-frame inverse contract in JAX."""

    _, jnp = _jax_modules()
    values = jnp.asarray(x)
    if values.ndim != 3:
        raise ValueError("x must have shape [B, N, D]")
    if mode == "none":
        return values
    coord_array = jnp.asarray(coord)
    if coord_array.ndim != 3 or coord_array.shape[:2] != values.shape[:2]:
        raise ValueError("coord must have shape [B, N, C] matching x")
    active_coord = coord_array if observer_coord is None else _expand_observer_coord(observer_coord, coord_array)
    if mode == "operator_bank":
        if frame_operators is None:
            raise ValueError("operator_bank mode requires frame_operators with shape [K, D, D]")
        operators = jnp.asarray(frame_operators)
        if operators.ndim != 3 or operators.shape[1:] != (values.shape[-1], values.shape[-1]):
            raise ValueError(f"frame_operators must have shape [K, {values.shape[-1]}, {values.shape[-1]}]")
        if active_coord.shape[-1] != operators.shape[0]:
            raise ValueError(f"operator_bank coord last dim must equal {operators.shape[0]}")
        return jnp.einsum("bnk,kde,bne->bnd", active_coord.astype(values.dtype), operators.astype(values.dtype), values)
    if mode != "paired_rotation":
        raise ValueError("mode must be 'none', 'paired_rotation', or 'operator_bank'")
    if active_coord.shape[-1] < 2:
        raise ValueError("paired_rotation requires coord[..., :2] = [sin(theta), cos(theta)]")
    if values.shape[-1] % 2 != 0:
        raise ValueError("paired_rotation requires an even latent dimension")
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
    if observer.ndim != 3:
        raise ValueError("observer_coord must have shape [B, C] or [B, 1, C] or [B, N, C]")
    if observer.shape[0] != coord_array.shape[0] or observer.shape[-1] != coord_array.shape[-1]:
        raise ValueError(f"observer_coord must match coord batch and coord dims, got {observer.shape} and {coord_array.shape}")
    if observer.shape[1] == 1:
        return jnp.broadcast_to(observer, coord_array.shape)
    if observer.shape[1] != coord_array.shape[1]:
        raise ValueError(f"observer_coord token dim must be 1 or {coord_array.shape[1]}, got {observer.shape[1]}")
    return observer


def jax_nn_softmax(values: Any, *, axis: int) -> Any:
    jax, _ = _jax_modules()
    return jax.nn.softmax(values, axis=axis)


def _validate_params(params: dict[str, Any]) -> tuple[int, int, int]:
    _, jnp = _jax_modules()
    if not isinstance(params, dict) or "input_kernel" not in params or "bias" not in params:
        raise ValueError("params must contain input_kernel and bias")
    unknown = set(params) - {"input_kernel", "bias", "coord_kernel"}
    if unknown:
        raise ValueError(f"params contains unsupported entries: {sorted(unknown)}")
    kernel = jnp.asarray(params["input_kernel"])
    bias = jnp.asarray(params["bias"])
    if not jnp.issubdtype(kernel.dtype, jnp.inexact) or not jnp.issubdtype(bias.dtype, jnp.inexact):
        raise ValueError("input_kernel and bias must be floating-point or complex arrays")
    if kernel.ndim != 2 or bias.shape != (kernel.shape[1],):
        raise ValueError("input_kernel must have shape [D, H] and bias must have shape [H]")
    coord_dim = 0
    if "coord_kernel" in params:
        coord_kernel = jnp.asarray(params["coord_kernel"])
        if not jnp.issubdtype(coord_kernel.dtype, jnp.inexact):
            raise ValueError("coord_kernel must be a floating-point or complex array")
        if coord_kernel.ndim != 2 or coord_kernel.shape[1] != kernel.shape[1]:
            raise ValueError("coord_kernel must have shape [C, H]")
        coord_dim = int(coord_kernel.shape[0])
    return int(kernel.shape[0]), int(kernel.shape[1]), coord_dim


def apply_layer_single(params: dict[str, Any], x: Any, *, coord: Any | None = None, mask: Any | None = None) -> dict[str, Any]:
    """Apply the minimal layer to one vector ``[D]`` or sequence ``[N, D]``."""

    _, jnp = _jax_modules()
    input_dim, _, coord_dim = _validate_params(params)
    values = jnp.asarray(x)
    if values.ndim not in {1, 2} or values.shape[-1] != input_dim:
        raise ValueError(f"x must have shape [{input_dim}] or [N, {input_dim}]")
    was_vector = values.ndim == 1
    sequence = values[None, :] if was_vector else values
    tokens = sequence.shape[0]
    if coord_dim:
        if coord is None:
            coord_sequence = jnp.zeros((tokens, coord_dim), dtype=sequence.dtype)
        else:
            coord_array = jnp.asarray(coord)
            coord_sequence = coord_array[None, :] if was_vector and coord_array.ndim == 1 else coord_array
            if coord_sequence.shape != (tokens, coord_dim):
                raise ValueError(f"coord must have shape {(tokens, coord_dim)}, got {coord_sequence.shape}")
    elif coord is not None:
        raise ValueError("coord was provided but params do not contain coord_kernel")
    valid = jnp.ones((tokens,), dtype=bool) if mask is None else jnp.asarray(mask).astype(bool)
    if was_vector and valid.ndim == 0:
        valid = valid[None]
    if valid.shape != (tokens,):
        raise ValueError(f"mask must have shape {(tokens,)}, got {valid.shape}")
    y = jnp.einsum("nd,dh->nh", sequence, params["input_kernel"]) + params["bias"]
    if coord_dim:
        y = y + jnp.einsum("nc,ch->nh", coord_sequence, params["coord_kernel"])
    y = jnp.tanh(y) * valid[:, None].astype(y.dtype)
    pooled = masked_mean(y, valid, axis=0)
    coverage = jnp.mean(valid.astype(jnp.float32))
    return {"y": y[0] if was_vector else y, "pooled": pooled, "diagnostics": {"mask_coverage": coverage}}


def apply_layer(params: dict[str, Any], x: Any, *, coord: Any | None = None, mask: Any | None = None) -> dict[str, Any]:
    """Apply the minimal layer to ``[B, D]`` or ``[B, N, D]`` inputs."""

    jax, jnp = _jax_modules()
    input_dim, _, coord_dim = _validate_params(params)
    values = jnp.asarray(x)
    if values.ndim not in {2, 3} or values.shape[-1] != input_dim:
        raise ValueError(f"x must have shape [B, {input_dim}] or [B, N, {input_dim}]")
    batch = values.shape[0]
    was_vector = values.ndim == 2
    tokens = 1 if was_vector else values.shape[1]
    if coord_dim:
        if coord is None:
            coord_batch = jnp.zeros((batch, coord_dim), dtype=values.dtype) if was_vector else jnp.zeros((batch, tokens, coord_dim), dtype=values.dtype)
        else:
            coord_batch = jnp.asarray(coord)
            expected = (batch, coord_dim) if was_vector else (batch, tokens, coord_dim)
            if coord_batch.shape != expected:
                raise ValueError(f"coord must have shape {expected}, got {coord_batch.shape}")
    elif coord is not None:
        raise ValueError("coord was provided but params do not contain coord_kernel")
    if mask is None:
        mask_batch = jnp.ones((batch,), dtype=bool) if was_vector else jnp.ones((batch, tokens), dtype=bool)
    else:
        mask_batch = jnp.asarray(mask).astype(bool)
        expected_mask = (batch,) if was_vector else (batch, tokens)
        if mask_batch.shape != expected_mask:
            raise ValueError(f"mask must have shape {expected_mask}, got {mask_batch.shape}")

    if coord_dim:
        return jax.vmap(lambda values_i, coord_i, mask_i: apply_layer_single(params, values_i, coord=coord_i, mask=mask_i))(values, coord_batch, mask_batch)
    return jax.vmap(lambda values_i, mask_i: apply_layer_single(params, values_i, mask=mask_i))(values, mask_batch)


__all__ = [
    "ARTIJAXBackendNotImplementedError",
    "ARTIJAXBackendUnavailableError",
    "JAXARTIConfig",
    "apply_layer",
    "apply_layer_single",
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
