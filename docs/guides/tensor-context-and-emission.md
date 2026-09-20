# Tensor Context And Emission

ARTI accepts two context styles:

1. The original keyword arguments (`coord`, `mask`, `visibility`,
   `frame_operators`, and `observer_coord`) remain supported.
2. New integrations can pass one explicit `TensorContext` object. This is the
   preferred boundary for adapters, serialization, and model pipelines.

```python
import torch
from arti import FrameContext, TensorContext
from arti.legacy import ARTILayer

layer = ARTILayer(
    input_dim=32,
    hidden_dim=64,
    coord_dim=2,
    coord_frame_mode="paired_rotation",
)
x = torch.randn(4, 16, 32)
coord = torch.zeros(4, 16, 2)
coord[..., 1] = 1.0
valid = torch.ones(4, 16, dtype=torch.bool)
visible = valid[:, :, None] & valid[:, None, :]

context = TensorContext(
    valid_mask=valid,
    visibility=visible,
    frame=FrameContext(coord=coord),
)
out = layer(x, context=context)
```

`valid_mask` has shape `[B, N]` and dtype `torch.bool`. It identifies slots
that exist. `visibility` has shape `[B, N, N]` and dtype `torch.bool`; it
controls which source slots may influence each query slot. The strict path
does not silently cast either tensor's dtype or device, and it intersects
visibility with the valid mask before a reader uses it.

`FrameContext` keeps observation-frame tensors together. `coord` is
`[B, N, C]`, `observer_coord` is `[B, C]`, `[B, 1, C]`, or `[B, N, C]`, and
`frame_operators` is `[C, D, D]` for `operator_bank`. In
`paired_rotation` mode, the first two coordinate values must be a unit
`[sin(theta), cos(theta)]` pair. A disabled frame (`coord_frame_mode="none"`)
does not apply an inverse; `TensorContext()` is therefore a valid no-op
context.

Do not mix the two styles in one call. This is rejected so a stale mask cannot
silently override a serialized context object:

```python
layer(x, context=context)                 # strict contract
layer(x, coord=coord, mask=valid)         # compatibility contract
```

## Emission Routing

`EmissionRouter` is the generic replacement boundary for application-specific
half-transparent output policies. It knows only numbered streams, not
assistant, user, public, inner, or any other business role.

```python
from arti import EmissionRouter, EmissionRouterConfig

router = EmissionRouter(
    EmissionRouterConfig(
        hidden_dim=64,
        stream_count=3,
        emit_streams=(0, 2),
    )
)
routed = router(hidden, valid_mask=valid)
stream_ids = routed.stream_ids
emit_mask = routed.emit_mask
```

Authorization is a separate tensor policy. Use `build_stream_visibility()`
with a boolean `stream_readable_by` matrix to construct a query-to-source
visibility tensor. Emission probability is never treated as authorization.
`MembraneVisibilityRouter` remains only as a compatibility adapter for the
old two-stream assistant API.

## Saved Contracts

`arti.save()` keeps runtime tensors out of `arti.st`, but records an
`architecture.context_contract` for ARTI layers. The same contract is included
in Python-owned Web artifact metadata when the layer is exported. It describes
shapes, dtype categories, required fields, and frame mode; it does not contain
sample data or authorization policy values.

```python
saved = arti.save(layer, "layer.arti.st")
loaded = arti.load(saved.weights_path)
print(loaded.manifest["architecture"]["context_contract"])
```
