# Activation And Workspace APIs

This guide shows the small neural-network-native APIs that sit below the larger
ARTI blocks. They can be used independently in ordinary PyTorch modules.

## Half

`Half` is a stateless activation:

```python
import torch
import arti
import arti.nn as ann

act = ann.Half()
x = torch.randn(8, 32)
y = act(x)
```

It converts feature strength into survival. Strong features pass close to
unchanged; weak or ambiguous features fade. In Recall branches, the intended
pattern is:

```python
delta = recall_layer(h)
delta = ann.Half()(delta)
h = h + delta
```

`stochastic` and `learnable` are independent options. `stochastic=True` keeps
Bernoulli survival in both `train()` and `eval()`; it never switches to the
expected value automatically. `learnable=True` makes the threshold, survival
base, and scale trainable:

```python
sampled = ann.Half(stochastic=True)
learned = ann.Half(stochastic=False, learnable=True)
q = learned.survival(x)  # differentiable q(x), without sampling
```

Use `stochastic=False` when a deterministic artifact or reproducible tensor
path is required. `Half` has no per-call runtime state; fixed and learnable
survival curves are both ordinary module configurations.

Contextual survival is an explicit versioned variant. It still returns a tensor
with the same shape and still only applies `y = q * x`, but `q` can depend on
other values along declared aggregation axes:

```python
contextual = ann.Half(
    stochastic=False,
    context_mode="contextual",
    context_axes=(-1,),
)
q = contextual.survival(x)  # q.shape == x.shape, no sampling
y = contextual(x)            # y.shape == x.shape
```

The pointwise contract is registered as `arti/half@1`; contextual instances
are registered as `arti/half@2`. This is a component mechanism reference, not
the Python class name or the package version. Contextual Half is not currently
accepted by the Web exporter until its artifact contract is validated.

### Custom Survival Rules

`Half` separates the survival executor from the salience rule. A custom rule
is a differentiable module that returns a same-shape probability tensor:

```python
class MySurvival(torch.nn.Module):
    survival_contract = arti.SurvivalContract(
        identity=arti.SurvivalRef.parse("myorg/survival@1"),
    )

    def forward(self, x):
        return torch.sigmoid(x.abs() - 0.75)

layer = ann.Half(stochastic=False, survival=MySurvival())
```

For reusable local experiments, register a factory with an explicit identity:

```python
arti.register_survival(
    "myorg/survival@1",
    factory=lambda config: MySurvival(),
)
layer = ann.Half(survival="myorg/survival@1")
```

The `namespace/name@integer` string is a source declaration accepted at this
registration and constructor boundary. Registration resolves it through the
module's static `SurvivalContract` to a full `@sha256:<digest>` identity;
provenance and artifacts persist only that immutable address. `Half` owns
deterministic multiplication or Bernoulli sampling; the survival rule owns
only `q`. Unregistered or third-party non-portable rules remain usable for
forward/training experiments but are rejected by provenance and artifact
export until an explicit portability contract exists.

## Fold

`Fold` compacts a variable or overcomplete latent sequence into a fixed-size
workspace:

```python
fold = ann.Fold(k=16, dim=64)
x = torch.randn(4, 128, 64)
z = fold(x)  # [4, 16, 64]
```

Use `mask` for padding or invalid slots:

```python
mask = torch.ones(4, 128, dtype=torch.bool)
z = fold(x, mask=mask)
```

Use `q` only for salience or survival guidance:

```python
survival = torch.rand(4, 128)
z = fold(x, q=survival)
```

`mask` answers "is this slot valid?" while `q` answers "how much should this
slot survive?" Keeping them separate matters when sparse or top-k modes are
enabled.

## Pulse

`Pulse` is the default learned pulse layer. It uses `Half` plus `Fold` to form a
compact latent workspace:

```python
pulse = ann.Pulse(k=8, dim=64, hidden_dim=128)
x = torch.randn(4, 256, 64)
z = pulse(x, mask=mask)  # [4, 8, 64]
```

`Half` is enabled inside Pulse by default and can be disabled independently for
an ablation or a task whose fragments already carry calibrated survival:

```python
pulse = ann.Pulse(k=8, dim=64, use_half=False)
```

Use it when an upstream system produces many fragments and the next layer should
consume a fixed number of latent slots. `LearnedPulse` is an explicit alias for
the same layer. `PulseCompressor` is the legacy explicit pulse-id path.

## FusionPulse

`FusionPulse` merges several already compact Pulse workspaces into one fixed-size
workspace. It automatically concatenates the inputs, learns feature-wise
salience in their joint context, applies `Half`, and lets one shared `UnFold`
query the fused output:

```python
left = ann.Pulse(k=8, dim=64)(left_fragments)
right = ann.Pulse(k=8, dim=64)(right_fragments)

fusion = ann.FusionPulse(k=8, dim=64)
z = fusion.concat(left, right)  # [B, 8, 64]
```

The Pulse workspaces may have different slot counts. For equal-size workspaces,
the stack form is convenient and easier to export:

```python
z = fusion(torch.stack((left, right), dim=1))  # [B, sources, slots, dim]
```

The source count is dynamic; `FusionPulse` does not allocate one embedding table
row per source. It derives a learned source representation from relative concat
position, so the same module can accept more sources without reinitialization.

Balanced consolidation needs an explicit training signal. Request diagnostics
and add the label-free structural term to the task loss:

```python
z, info = fusion.concat(left, right, return_info=True)
loss = task_loss(z, target) + info["structural_loss"]
```

The structural term penalizes simultaneous survival of very similar candidates
while requiring every similarity neighborhood to retain support and at least
one strong representative. It uses only the input tensors and masks, not task
labels or source semantics. The individual `Half`, `Fold`, and `UnFold` layers
remain unchanged and independently usable.

## Visual Field concat

`VisualField` converts rigid glyph bitmaps into Pulse fragments without resizing
or interpolation. Split fields can be concatenated before one shared Pulse:

```python
field = ann.VisualField(patch_size=(4, 4))

left = field(glyph, window=(0, 0, 16, 48))
right = field(glyph, window=(0, 48, 16, 48))
visual = ann.concat_visual_fields(left, right)

pulse = ann.Pulse(k=8, dim=visual.fragments.shape[-1])
z = pulse(visual.fragments, mask=visual.mask)
```

Concat expands the fragment axis: `[N1, D] + [N2, D] -> [N1 + N2, D]`.
Pixel patches remain rigid, while normalized absolute patch bounds are appended
to each model-facing fragment so Pulse can distinguish equal strokes at
different locations. The same Pulse parameters process every concatenated
field; this is neither image blending nor output ensembling.

## VisualScan

`VisualScan` consumes registered low-resolution observations from a pixel-shift
acquisition process:

\[
y_t = D H T_{\delta_t} x + \epsilon_t.
\]

`T` is a recorded subpixel translation, `H` is optical blur, and `D` is
downsampling. The high-resolution target is used for training or evaluation; it
is never passed to `VisualScan` during inference.

```python
config = arti.VisualScanConfig(
    low_size=(8, 8),
    scale=2,
    shifts=((0.0, 0.0), (0.0, 0.5), (0.5, 0.0), (0.5, 0.5)),
    patch_size=(2, 2),
    pulse_count=8,
)
scan = arti.nn.VisualScan(config)

observation = arti.pixel_shift_observe(high_resolution_training_target, config)
pulses = scan(observation.frames, shifts=observation.shifts, mask=observation.mask)
```

Each frame is first lifted and inversely registered by its continuous shift.
Registered frame patches and a deterministic shift-and-add carrier are then
concatenated as VisualField fragments and compacted by the learned `Pulse`
(`Half` + `Fold`) path to a fixed `[B, K, D]` workspace. The carrier is added as
a residual workspace, so Pulse learns correction rather than relearning the
coordinate transform from discrete shift labels.
Set `persistence_steps` and `persistence_decay` for finite scan persistence; no
hidden state survives beyond the supplied frame stack.

VisualScan also keeps its learned Pulse and deterministic carrier orthogonal:
`use_pulse=False` selects a carrier-only workspace,
`registered_workspace_residual=False` selects Pulse-only compaction, and
`pulse_use_half=False` disables Half only inside that Pulse. At least one of
Pulse or the registered carrier must remain enabled.

The layer cannot invent spatial frequencies absent from its observations.
Repeated phases and single-frame masks provide no complementary samples, wrong
shift metadata damages registration, and unrecorded scene motion still violates
the model. Use it only when acquisition geometry is known or estimated.

## RecallExecutor

`RecallExecutor` is a runtime-policy adapter around `arti.nn.Recall`. It does not
own a second loop or add another activation:

```python
policy = arti.ExecutionPolicy.fixed(3, trace_level="summary")
executor = ann.RecallExecutor(recall_layer)
h_next, info = executor(h, policy=policy, return_info=True)
```

Activation, Formula, routing, finite checks, checkpoints, and stopping remain
owned by the wrapped Recall and its explicit `ExecutionPolicy`.

For generated-text probes, do not judge RecallExecutor only by exact string
matching. Iterative execution can produce a different sentence. Track
coherence, repetition, degeneration, and task correctness separately. Also make
the training loss preserve the conditioning signal. A next-token loss by itself
can let iterative execution wash out prompt identity; the Qwen string-first
probe therefore adds a lightweight prompt-conditioning auxiliary loss.

## Minimal Composition

```python
class CompactRecallBlock(torch.nn.Module):
    def __init__(self, dim: int, k: int) -> None:
        super().__init__()
        self.pulse = ann.Pulse(k=k, dim=dim, hidden_dim=dim * 2)
        self.readout = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(k * dim, dim),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        z = self.pulse(x, mask=mask)
        return self.readout(z)
```

This is still tensor-in / tensor-out. Coordinates, visibility, full ARTI blocks,
runtime vocab, and Qwen adapters can be added only when the task needs them.
