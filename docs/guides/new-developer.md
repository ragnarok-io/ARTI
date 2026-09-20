# New Developer Guide

This page is the shortest path for a new developer to understand and use ARTI.

## What ARTI Is

ARTI is a domain-free neural-network dependency. Downstream projects convert
their data into tensors, coordinates, masks, and optional visibility tensors.
ARTI transforms those anonymous latent tensors and returns structured latent
representations.

The main runtime contract is:

```text
x          [B, N, D] or [B, D]
coord      [B, N, C] optional
mask       [B, N] optional
visibility [B, N, N] optional

out.y
out.pooled
out.diagnostics
```

## First Run

Set up the repository:

```bash
uv sync --extra torch --extra dev
```

Run the dependency quickstart:

```bash
uv run --extra torch python examples/pytorch_dependency_quickstart.py
```

Expected output includes:

```text
layer.y (4, 12, 64)
layer.pooled (4, 64)
classifier.logits (8, 3)
```

Then run the first quality gate:

```bash
uv run --extra dev python scripts/quality_gate.py quick
```

If these two commands pass, ARTI is usable as a local stable dependency. The
next page to keep open is the validation matrix:

```text
docs/reference/validation-matrix.md
```

It maps each mechanism to the command that protects it and the claim boundary
that should not be crossed.

## Use ARTI In Another Project

Add this checkout as an editable dependency:

```bash
uv add --editable /path/to/ARTI
```

Then import the stable API:

```python
from arti.nn import Layer
from arti import ARTIResidualBlock, ARTIClassifier, legacy
from arti import Half, Fold, Pulse, Recall, RecallExecutor
```

## Minimal Layer

```python
import torch
import arti

layer = arti.nn.Layer(dim=32)

x = torch.randn(8, 16, 32)
mask = torch.ones(8, 16, dtype=torch.bool)

out = layer(x, mask=mask)
loss = out.pooled.square().mean()
loss.backward()
```

This minimal path allocates no phase, interface, pairwise-context, Recall, or
virtual-Recall parameters. Use `profile="recall"`, `profile="multisource"`, or
an explicit `arti.features(...)` only when the data and task need them. See the
[Progressive API](progressive-api.md).

## Recall Contracts

Use `arti.nn.Recall` when the standalone Formula should return a complete
`next_state` tensor:

```python
recall = Recall(dim=64, slots=32, formula="arti/delta@1")
next_state = recall(hidden)
```

`RecallExecutor` accepts only a producer that declares
`output_semantics="next_state"`. It converts that next state into
`next_state - hidden` before applying Half and the step scale. Residual,
undeclared, or complete layer outputs are rejected instead of guessed.

The forward Bank updater is available separately as
`arti.mechanisms.RecallValueUpdater`; it is the stable state-transition
primitive, while retired TTT sessions remain experimental/internal.

## Where The Code Lives

- `src/arti/layers.py`: core tensor-in/tensor-out layers.
- `src/arti/blocks.py`: shape-stable blocks for insertion into PyTorch models.
- `src/arti/models.py`: small reference models.
- `src/arti/nn.py`: activation-style APIs: `Half`, `Fold`, `Pulse`, and `RecallExecutor`.
- `src/arti/training.py`: auxiliary recall/replay training losses.
- `src/arti/runtime_vocab.py`: alpha runtime vocab binding input and output heads.
- `src/arti/text_bitmap.py`: optional text-to-bitmap rigid tensor helpers for runtime vocab.
- `src/arti/pulse.py`: explicit pulse-id compression for tokenization-stable latent streams.
- `src/arti/source_integrity.py`: source carrier superposition and integrity diagnostics.
- `src/arti/fit/`: adapter planning, project metadata, and `.fit()` helpers.
- `benchmarks/`: controlled mechanism validations and generated evidence.
- `examples/`: copyable usage examples.

## Mechanism Map

| Concept | Code | Purpose |
| --- | --- | --- |
| Coordinates | `coord`, `coord_frame_mode` | Reference-frame context for latent tensors. |
| Mask | `mask` | Excludes padding or invalid tokens from pooling and updates. |
| Visibility | `visibility` | Controls token-to-token influence. |
| Virtual interface | `ARTIVirtualInterfaceMixer` | Fixed-size workspace for scalable token synchronization. |
| Recall | `arti.nn.Recall` | Formula layer returning a complete `next_state`. |
| Updater | `arti.mechanisms.RecallValueUpdater` | Forward trace-to-Bank state transition. |
| Full latent layer | `Layer` / `legacy.ARTILayer` | Returns `ARTIOutput`, including `y` and diagnostics. |
| Recall executor | `RecallExecutor` | Iterative next-state correction loop. |
| Runtime vocab | `RuntimeVocabInput`, `RuntimeVocabHead` | Binds input and output to the current rigid vocab tensor. |
| Text bitmap | `render_text_vocab`, `BitmapTextRenderer` | Renders words into rigid visual tensors, optionally from font files. |
| Pulse | `Pulse`, `LearnedPulse` | Learns compact latent pulse workspaces from overcomplete fragments. |
| Legacy explicit pulse | `PulseCompressor`, `pulse_compress` | Resamples variable token streams into externally supplied pulse slots. |
| Source integrity | `SourceIntegrityCarrier`, `superpose_sources`, `source_integrity_report` | Keeps synchronized multi-source fields separable without adding tokens; supports `off`, `pilot`, `summary`, and `full` diagnostic modes. |
| Diagnostics | `ARTIOutput.diagnostics` | Exposes mechanism signals for tests and debugging. |

Text bitmap defaults are glyph-first for visible characters. For experiments
with small punctuation, repeated letters, or narrow characters, use at least the
default `14x96` canvas, keep the renderer settings fixed in evidence metadata,
and run `benchmarks/run_micro_glyph_distinctness.py`. Control characters such as
newline, page break, and zero-width marks should use the text tensor auxiliary
control channels rather than pretending to be visible glyphs.

## Half, Fold, Pulse, RecallExecutor

These lower-level APIs are useful when you want ARTI mechanisms without a full
`ARTILayer`:

```python
import torch
import arti.nn as ann

x = torch.randn(4, 128, 64)
mask = torch.ones(4, 128, dtype=torch.bool)

pulse = ann.Pulse(k=8, dim=64, hidden_dim=128)
z = pulse(x, mask=mask)
```

Use `mask` for padding or invalid slots. Use `q` only for real salience or
survival guidance:

```python
survival = torch.rand(4, 128)
z = pulse(x, q=survival, mask=mask)
```

This separation is intentional. A padding mask says whether a slot exists; it
does not say which valid slot is more important. The longer cookbook is
`docs/guides/activation-workspace.md`.

## External Phase Vs Fallback Phase

External phase carries source identity. Apply it near the front of a model while
hidden states still align with original tokens or sensors.

Fallback phase is different. If no external `coord`, `mask`, or `visibility`
exists, a layer can generate stable random coordinates:

```python
layer = legacy.ARTILayer(
    input_dim=64,
    coord_dim=8,
    fallback_context="random_coord",
)
```

Use fallback phase for arbitrary middle-layer insertion, routing scaffolds, or
regularization. Do not treat it as proof of participant identity or authority.

## Optional Mechanisms

ARTI mechanisms are independently optional. Keep unused behavior off when you
want a small insertion layer or a controlled ablation:

```python
layer = legacy.ARTILayer(
    input_dim=64,
    coord_dim=0,
    recall_steps=0,
    use_phase_mixer=False,
    use_virtual_interface=False,
    use_pairwise_context=False,
    fallback_context="none",
)
```

This makes ARTI behave closer to a plain tensor transformation while preserving
the same output contract.

## Checks Before Changing Code

Use the quick gate while editing:

```bash
uv run --extra dev python scripts/quality_gate.py quick
```

Check packaging:

```bash
uv run --extra dev python scripts/check_package.py
```

Check docs:

```bash
uv run --extra docs python scripts/quality_gate.py docs
```

Check controlled mechanisms:

```bash
uv run --extra dev python scripts/quality_gate.py mechanism
```

Run the Qwen 0.6B-class smoke path:

```bash
uv run --extra dev python scripts/quality_gate.py qwen
```

Interpret Qwen results by stage:

- Simulated or Qwen-shaped runs do not forward a real Qwen model.
- Real-forward runs load a frozen Qwen model when available, extract hidden
  states, and train only ARTI-side adapters/routers/heads.
- Adapter training is not base-model fine-tuning.

The default Qwen gate records `Qwen/Qwen3-0.6B` as the selected Qwen-class
checkpoint, keeps `Qwen/Qwen2.5-0.5B-Instruct` as a documented fallback, and
verifies ARTI integration paths. For a live model load, install the optional
Hugging Face dependencies and allow network/model download:

```bash
uv sync --extra torch --extra qwen
uv run --extra torch --extra qwen python benchmarks/run_qwen_arti_smoke.py --run-model
uv run --extra torch --extra qwen python benchmarks/verify_qwen_arti_smoke.py
```

## Claim Boundaries

ARTI 3.0.11 is a stable dependency. It has package, docs, API, CUDA/JAX
boundary, Qwen smoke, and controlled mechanism checks. The evidence supports
local mechanism behavior and integration readiness, not broad downstream
benchmark superiority or trained Qwen quality gains.

When adding examples or docs, keep the package domain-free: do not put business
schemas, task labels, or domain rules into the core API.

## Windows uv Cache

If the default user cache is not writable or you want all state inside the
checkout, run commands with a workspace-local uv cache:

```powershell
$env:UV_CACHE_DIR=(Join-Path (Get-Location) '.uv-cache')
uv run --extra dev python scripts/quality_gate.py quick
```

uv can manage CUDA-capable PyTorch environments. Whether CUDA is actually
available is decided by the installed PyTorch build and verified by:

```bash
uv run --extra dev python benchmarks/verify_torch_cuda_runtime.py
```
