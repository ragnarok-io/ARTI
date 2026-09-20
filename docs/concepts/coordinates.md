# Coordinates And Phase

In ARTI, coordinates are not ordinary extra features. They can identify the observation frame in which a latent tensor is currently expressed.

Coordinates are optional. Data without a meaningful phase should use
`coord_dim=0`, `coord_frame_mode="none"`, and `fallback_context="none"`.
Set `use_phase_mixer=False` when dynamic phase/operator mixing is also unwanted.
In that configuration ARTI neither invents a coordinate nor allocates phase
mixer parameters.

The `operator_bank` coordinate mode treats `coord` as a frame selector and applies a deterministic inverse tensor operator before projection:

```text
observed tensor + coord + inverse operator bank -> canonical latent tensor
```

This is useful when the raw tensor has already been observed through different phase-like transformations. ARTI does not need to rotate the input again; it needs to invert the current observation frame.

## Autoregressive Observer Frame

For autoregressive models, the next token can define the active observer frame. In that case, pass `observer_coord` in addition to the per-token `coord`:

```python
out = layer(
    context_x,
    coord=context_coord,
    observer_coord=next_token_coord,
    mask=context_mask,
    frame_operators=inverse_bank,
)
```

`coord` still describes the frame in which each context token was observed. `observer_coord` describes the reference frame of the token currently being predicted. ARTI applies the observer frame inverse to the whole visible context:

```text
observer inverse(next token phase) @ observed context
```

Same-frame context becomes canonical in the active observer frame. Other-frame context keeps its relative phase difference. This is the intended self/world separation mechanism for autoregressive ARTI: identity is represented by the tensor reference frame, not by an extra feature channel.

## Placement

When coordinates represent externally assigned token identity, source authority,
participant identity, or sensor identity, place the ARTI layer before those
tokens have been mixed by attention, pooling, or other cross-token operators.
The coordinate is attached to the original observation frame; once a hidden
state contains multiple sources, a single original-token coordinate may no
longer be well-defined.

Use early ARTI layers for externally grounded phase:

```text
raw token/source tensor -> coord-aware ARTI -> downstream mixing layers
```

Use later ARTI layers only when the coordinate describes the current mixed
latent state itself, or when the architecture preserves source-aligned token
states.

## Random Fallback Coordinates

When no external coordinate, mask, or visibility signal is available, ARTI can
optionally generate a stable random fallback context:

```python
from arti.legacy import ARTILayer

layer = ARTILayer(
    input_dim=64,
    coord_dim=8,
    fallback_context="random_coord",
)
```

`random_coord` assigns a deterministic learned-layer buffer coordinate by token
position. `random_context` additionally supplies the default all-valid
token-to-token visibility derived from the mask.

This fallback is useful for inserting ARTI into arbitrary middle layers where no
external source phase exists. It should not be confused with externally grounded
participant, authority, sensor, or token-source phase. External phase carries
identity semantics and should be applied before source mixing; random fallback
phase is a generic latent scaffold for routing and regularization.

## Modes

- `none`: no coordinate inverse is applied.
- `paired_rotation`: simple paired-channel rotation inverse, useful as a toy mechanism.
- `operator_bank`: general tensor operator inverse selected by `coord`.

## Contract

```python
out = layer(
    x,
    coord=coord,
    mask=mask,
    frame_operators=inverse_bank,
)
```

For `operator_bank`, `coord.shape[-1]` must match `frame_operators.shape[0]`.

`observer_coord` may have shape `[B, C]`, `[B, 1, C]`, or `[B, N, C]`. `[B, C]` and `[B, 1, C]` are broadcast across all context tokens.
