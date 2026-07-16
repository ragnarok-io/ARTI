# FusionPulse

`FusionPulse` combines multiple compact Pulse workspaces into one fixed-size
workspace. It is an alpha PyTorch layer and does not replace `Pulse`.

```python
import torch
from arti.nn import FusionPulse, Pulse

left = Pulse(k=8, dim=64)(torch.randn(4, 48, 64))
right = Pulse(k=8, dim=64)(torch.randn(4, 96, 64))

fusion = FusionPulse(k=8, dim=64)
z = fusion.concat(left, right)

assert z.shape == (4, 8, 64)
```

The operation is:

```text
Pulse workspaces -> concat -> learned salience -> Half -> shared UnFold
```

The source count is dynamic. Pulse workspaces may also have different slot
counts when the `concat()` convenience method is used. Equal-size workspaces
can use a stacked tensor directly:

```python
stacked = torch.stack((left, right), dim=1)  # [B, sources, slots, dim]
z = fusion(stacked)
```

## Masks

Pass one mask per source when slot counts differ:

```python
z = fusion.concat(
    left,
    right,
    masks=(left_mask, right_mask),
)
```

For stacked inputs, use a stacked `[B, sources, slots]` mask. Invalid slots do
not contribute to salience, structural losses, or the shared UnFold queries.

## Balanced training

The task loss trains the fused workspace normally. When repeated observations
should consolidate without erasing source-specific observations, request the
label-free structural term:

```python
z, info = fusion.concat(left, right, return_info=True)
loss = task_loss(z, target) + info["structural_loss"]
```

The term combines three pressures derived only from input tensors and masks:

- discourage simultaneous survival of very similar candidates;
- retain support in every similarity neighborhood;
- retain at least one strong representative in every neighborhood.

The diagnostics also include feature-wise `survival`, source indices, masks,
and the three unweighted structural-loss components. `Half`, `Fold`, and
`UnFold` remain independent public layers.

## Scope

`FusionPulse` uses joint attention over the concatenated compact workspaces.
It is intended for already compact Pulse tensors, not unbounded raw sequences.
Its structural-loss defaults are alpha and should be validated for the target
task before deployment.
