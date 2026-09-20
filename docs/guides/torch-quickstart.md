# PyTorch Quickstart

Install:

```bash
uv sync --extra torch --extra dev
```

Use ARTI from another local project:

```bash
uv add --editable /path/to/ARTI
```

Use ARTI as a normal PyTorch module:

```python
import torch
from arti.legacy import ARTILayer

layer = ARTILayer(input_dim=32, coord_dim=4, hidden_dim=64)

x = torch.randn(8, 16, 32)
coord = torch.randn(8, 16, 4)
mask = torch.ones(8, 16, dtype=torch.bool)

out = layer(x, coord=coord, mask=mask)

loss = out.pooled.square().mean()
loss.backward()
```

This is the retired classic latent-tensor API. Modern Federal program hosts use
`arti.ARTILayer(program)` as described in the stable release surface.

Backend-explicit imports are also available for the modern Program host:

```python
from arti.torch import ARTILayer
from arti.torch.training import virtual_recall_alignment_loss
```

For `[B, D]` inputs, ARTI treats the tensor as a single-token sequence.

Reference classifier:

```python
import torch
from arti import ARTIClassifier

model = ARTIClassifier(input_dim=16, hidden_dim=32, output_dim=3)
logits = model(torch.randn(8, 16))
```

Runnable example:

```bash
uv run --extra torch python examples/pytorch_dependency_quickstart.py
uv run --extra torch python examples/coord_mask_visibility_recall.py
```

The second example shows how to pass `coord`, `mask`, token-to-token
`visibility`, and an auxiliary `recall` tensor into the retired classic tensor
layer, then train with a small recall-alignment loss.

Fallback context for middle-layer insertion:

```python
from arti import ARTIResidualBlock

block = ARTIResidualBlock(
    dim=64,
    coord_dim=8,
    fallback_context="random_coord",
)

y = block(hidden)  # no external coord/mask/visibility required
```

Optional mechanisms can be enabled independently:

```python
layer = ARTILayer(
    input_dim=64,
    coord_dim=0,
    use_phase_mixer=False,
    use_virtual_interface=False,
    use_pairwise_context=False,
    use_recall=False,
    use_virtual_recall=False,
    fallback_context="none",
)
```

Disabled mechanisms allocate no mechanism-specific parameters and execute no
mechanism-specific branch. ARTI does not synthesize phase semantics when
`coord_dim=0`, `coord_frame_mode="none"`, and `fallback_context="none"`.

Common switches:

| Mechanism | Switch |
| --- | --- |
| Phase/operator mixing | `use_phase_mixer=False` |
| Virtual interface | `use_virtual_interface=False` |
| Pairwise visibility context | `use_pairwise_context=False` |
| Recall update loop | `use_recall=False` (or `recall_steps=0`) |
| Virtual recall output | `use_virtual_recall=False` |
| External coordinate input | `coord_dim=0` or omit `coord` |
| Coordinate frame inverse | `coord_frame_mode="none"` |
| Random fallback context | `fallback_context="none"` |
