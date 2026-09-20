# JAX Backend

ARTI is PyTorch-first and exposes a small optional JAX functional subset for
shared tensor contracts. It is native JAX code, but it is not a second
implementation of ARTI modules, fitting, attachment, serialization, or Recall.

The package reports JAX availability from the local environment:

```python
import arti.jax as arti_jax

assert arti_jax.backend_status() in {"available", "broken", "unavailable"}
print(arti_jax.smoke_report()["smoke_status"])
```

`smoke_report()` checks forward execution, JIT, whole-parameter-tree gradients,
and VMAP semantics. It returns `skipped` when JAX is absent and `failed` when a
discovered JAX installation is broken. Passing proves only the documented
functional subset, not full parity with PyTorch modules or ARTI mechanisms.

CI jobs that install the JAX extra can require this smoke check explicitly:

```bash
uv sync --locked --extra dev --extra jax
uv run python scripts/quality_gate.py jax --fail-fast
```

Current boundary:

```text
arti.core      -> backend-independent configs and contracts
arti.torch     -> nn.Module implementation
arti.jax       -> pure functional helpers and PyTree parameter dictionaries
```

Current compatibility layer:

```python
from arti import ARTILayer
from arti.torch import ARTILayer as TorchARTILayer

assert TorchARTILayer is ARTILayer
```

The current JAX layer preserves ARTI's basic tensor contracts:

- `x: [B, N, D]` or `[B, D]`
- `coord: [B, N, C]`
- `mask: [B, N]`
- `y`, `pooled`, `diagnostics`

It also exposes JIT-compatible mask helpers:

```python
pooled = arti_jax.masked_mean(x, mask, axis=1)
weights = arti_jax.masked_softmax(logits, visibility_mask, axis=-1)
coverage = arti_jax.mask_coverage(mask)
visibility = arti_jax.ensure_visibility(None, mask)
causal_visibility = arti_jax.attention_mask_to_visibility(mask, causal=True)
```

Coordinate-frame inverse helpers are also available for early parity work:

```python
canonical = arti_jax.apply_coord_frame_inverse(
    x,
    coord,
    mode="operator_bank",
    frame_operators=inverse_operator_bank,
    observer_coord=next_token_phase,
)
```

The test suite checks these JAX helpers against the PyTorch functional backend for masked mean, masked softmax, visibility construction, and operator-bank frame inverse.

Minimal usage:

```python
import jax
import jax.numpy as jnp
import arti.jax as arti_jax

key = jax.random.PRNGKey(0)
params = arti_jax.init_layer(key, input_dim=32, hidden_dim=64, coord_dim=8)

x = jnp.ones((4, 16, 32))
coord = jnp.ones((4, 16, 8))
mask = jnp.ones((4, 16), dtype=bool)

out = arti_jax.apply_layer(params, x, coord=coord, mask=mask)
```

`params` is an array-only PyTree containing `input_kernel`, `bias`, and an
optional `coord_kernel`; static dimensions are not stored as gradient leaves.
For explicit `vmap` composition, use `apply_layer_single()` with `[D]` or
`[N, D]` samples.

Unsupported in this subset: `arti.fit()`, attachment, Recall, Stateful Recall,
Membrane, Half/Fold/Pulse, checkpointing, multi-device execution, and complete
`ARTILayer` parity.
