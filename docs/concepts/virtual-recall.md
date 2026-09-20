# Virtual Recall

Virtual recall is a training-time auxiliary output mechanism.

ARTI produces:

```text
y         -> main latent output for downstream layers
virtual_y -> auxiliary latent output for recall alignment
```

`virtual_y` is not passed directly to downstream layers by default. It is used to train corrupted or partial inputs to recover the latent representation produced by clean inputs.

## Alignment Objective

```text
clean_x   -> ARTI -> clean.y
corrupt_x -> ARTI -> corrupt.virtual_y

loss = mse(corrupt.virtual_y, stop_gradient(clean.y))
```

The helper `virtual_recall_alignment_loss` implements this pattern.

Epochs before `align_start_epoch` can train `virtual_y` toward zero. Later epochs align corrupted-input `virtual_y` to clean-input `y`.

## Recall State Writes

Runtime Recall writes use `Half` by default:

```python
write = recall_bank.query(h)
write = Half()(write)
h = apply_write(h, write)
```

`apply_write` denotes an update of the current state inside Recall. The
modified state is returned directly and becomes the query for the next step;
the host does not add the original tensor again. With the default
`recall_value_composition="single"`, Bank values are complete
host-dimensional writes. The optional `"product"` mode divides the value Bank
into two independently routed halves and computes:

```text
gain = 1 + tanh(scale)
candidate_h = gain * (h + shift)
write = Half()(candidate_h - h)
next_h = h + write
```

Set `recall_activation="none"` to apply `candidate_h` exactly. The bounded gain
keeps the raw multiplicative factor in `[0, 2]` while the shift can introduce
new values. Both halves share only the batch dimension, the total value-Bank
parameter count is unchanged, and an all-zero Bank is an exact identity.

The optional `"state"` mode divides the same value Bank into seventeen independently
routed factors:

```text
coarse_content    = query(coarse_content_bank)
fine_content      = stopgrad(max(abs(coarse_content), 1)) *
                    tanh(query(fine_content_bank))
content           = coarse_content + fine_content
modulation[i]     = query(modulation_bank[i]), i = 1..13
write_direction   = tanh(query(write_bank))
memory_opacity    = tanh(query(opacity_bank)) ** 2

polynomial = product(1 + tanh(modulation[i]) / 13)
recalled_content = content * polynomial
host_weight = 1 - memory_opacity
memory_weight = memory_opacity * write_direction
next_h = host_weight * h + memory_weight * recalled_content
```

This mode does not blindly replace the host tensor. Recall learns both a signed
memory contribution and per-feature host transparency. The coarse-content factor
is initialized with nonzero values and calibrated once to the host feature mean
and scale. The bounded fine-content factor learns an independently routed
relative fractional correction, separating large state changes from detail
detail correction without giving the scale term a second gradient path.
Legacy state Banks migrate exactly by retaining their content as the coarse limb
and initializing the fine limb to zero. Thirteen small modulation factors form a normalized polynomial
expansion: degree-\(d\) cross terms are attenuated by \(13^{-d}\). The opacity coupling
enforces `host_weight + abs(memory_weight) <= 1`, preventing host and Recall
from independently accumulating unbounded amplitude. The seventeen factors keep
the total value-Bank parameter count fixed.

This keeps Recall lightweight. Recall proposes candidate latent traces; `Half`
turns feature strength into survival, so strong trace features pass while weak
or ambiguous trace features fade. Recall has no separate trainable strength
controller. By default, `recall_recognition_mode="none"`, so positive trace
acquisition is not blocked by an untrained familiarity decision.

Recognition is an optional second-stage mechanism. Explicit mode computes
agreement between the query and retrieved trace and applies a configurable
threshold. Alignment mode learns a recognizer and should be trained with
appropriate positive and negative examples. Threshold and temperature only
apply to explicit mode. Set `recall_activation="none"` on
`arti.legacy.ARTILayer` or `ARTIConfig` only for ablations that need the raw
recall delta.

The retired monolithic layer can run the same Recall field as an adaptive
micro-cycle. The field parameters are shared across iterations; increasing the
maximum step count does
not allocate another Recall bank:

```python
from arti.legacy import ARTILayer as ClassicARTILayer

layer = ClassicARTILayer(
    input_dim=768,
    recall_steps=3,       # maximum steps
    recall_min_steps=1,
    recall_tolerance=1e-2,
)
```

The tolerance is applied to the change caused by the queried write relative
to the current latent state. Each batch sample can stop committing updates
independently after `recall_min_steps`. This version-1 path uses fixed masked
execution through the configured maximum. Diagnostics include
`recall_steps_attempted`, `recall_steps_committed`, and
`recall_step_update_ratio`. Leave `recall_tolerance=None` for a fixed number of
steps.

## Iterative Recall Execution

`RecallExecutor` is a stable policy adapter for the canonical Recall engine:

```python
executor = RecallExecutor(recall_layer)
h_next, info = executor(
    h,
    policy=ExecutionPolicy(max_steps=3, min_steps=1, tolerance=1e-3),
    return_info=True,
)
h_more = executor(h, policy=ExecutionPolicy.fixed(5))
```

The adapter owns no parameters, activation, or second execution loop. Formula,
Half configuration, dynamic Bank routing, and next-state composition remain in
the wrapped Recall; the policy controls only per-call execution.

For a high compute ceiling with token-resolved stopping, use the separately
versioned adaptive contract:

```python
policy = ExecutionPolicy.adaptive(
    max_steps=64,
    min_steps=2,
    scope="token",
    relative_tolerance=1e-3,
    route_tolerance=1e-2,
    patience=2,
    executor="early_break",
)
h_next, trace = executor(h, policy=policy, return_trace=True)
```

`max_steps` is a ceiling, not an expected depth. `RecallTraceV2` records token
steps, stop reasons, active fractions, logical token work, and executed loop
steps separately. `early_break` skips the remaining loop only after every token
has stopped. Tokens that stop earlier are frozen but remain visible to other
positions; the first version does not compact active tokens into smaller GPU
batches. For Formula implementations with cross-token coupling, token stopping
is therefore an explicit monotonic-freeze approximation.

Route and index histories record every attempted read, including a read that is
later rejected by finite-value or stopping checks. Mask trajectory analysis with
`recall_step_committed`; the top-level route, weights, and indices always describe
the last committed read.

`RetrievalRoutePlan@1` freezes a routing decision, not a recalled tensor. Replaying
the plan gathers the current Bank values and applies the current Formula to the
current latent state. `RetrievalRouteStack@1` is the runtime-only recursive container
used when a caller must compose exact plans across several generic execution
axes. Candidate-group restrictions are search constraints and are not exact
route replay.

The synthetic RecallExecutor benchmark compares a direct MLP denoiser,
single-step residual recall, multi-step recall without Half, and multi-step
RecallExecutor with Half:

```bash
uv run --extra dev python benchmarks/run_recall_refiner.py
uv run --extra dev python benchmarks/verify_recall_refiner.py
```

The benchmark measures final MSE to a clean hidden vector, intermediate MSE per
step, update norms, update-norm stability, and clean-start drift from weak noisy
corrections. Its claim boundary is narrow: iterative recall can be tested as
latent iterative execution on a controlled synthetic task, not as broad downstream
superiority.

## Folded Workspace

`Fold` can be used after Recall and `Half` when a model needs a compact fixed
workspace:

```python
delta = Half()(recall_layer(h))
traces = h + delta
workspace = Fold(k=16)(traces)
```

This is tensor defragmentation rather than manifold preservation. Recall
proposes candidate traces, `Half` applies survival pressure, and `Fold`
compacts surviving information into `[B, K, D]`. Downstream layers learn to
operate on the folded representation.

The synthetic Fold benchmark compares unguided mean pooling, q-guided mean
pooling, Fold without q, Fold with q, and Half+Fold on a role-bound pair task.
Two surviving traces carry different roles; pooling sees a bag of traces, while
Fold can organize them into a fixed workspace before the downstream head reads
the ordered pair:

```bash
uv run --extra dev python benchmarks/run_fold_compaction.py
uv run --extra dev python benchmarks/verify_fold_compaction.py
```

The benchmark is deliberately narrow. It checks that Fold compacts sparse
latent traces into a trainable fixed workspace without collapsing role binding;
it does not claim broad downstream superiority.

## Pulse

`Pulse` is the stable module for learned latent pulse formation:

```python
pulse = Pulse(k=8)
z = pulse(x)  # [B, N, D] -> [B, 8, D]
```

It starts from overcomplete latent fragments, applies `Half` survival pressure,
then uses `Fold` to compact the surviving information into a fixed-size pulse
workspace. `LearnedPulse` remains available as a descriptive alias. The older
`PulseCompressor` / `pulse_compress` path is legacy explicit compression, whose
pulse ids are externally specified.

The synthetic Pulse benchmark compares mean pooling, legacy explicit Pulse
compression, learned Pulse with two and four pulse slots, and an optional
residual correction block on the same role-bound latent signal task:

```bash
uv run --extra dev python benchmarks/run_learned_pulse_compaction.py
uv run --extra dev python benchmarks/verify_learned_pulse_compaction.py
```

The benchmark measures accuracy, signal-only retention, noise sensitivity, and
throughput on a controlled synthetic task. It does not claim tokenizer-free
modeling or general compression superiority.

## Pulse Efficiency

Pulse follows the same practical direction as Perceiver-style latent bottleneck
modules: move overcomplete fragments into a fixed smaller workspace before
standard layers read them. The expensive part is the soft assignment path:
fragment projection, `Half`, assignment logits, softmax over input slots, and
the weighted aggregation into `[B, K, D]`.

The current optimized path keeps the public API unchanged and applies two
conservative engineering changes:

- `Fold` uses explicit batched matrix multiplication for the final `[B, K, D]`
  aggregation instead of the previous equivalent `einsum` contraction, and it
  avoids forcing an extra contiguous copy before `bmm`.
- When external `q` guidance is supplied, `Pulse` uses that pre-normalized
  guidance for folding and skips the extra norm-based guide computation unless
  `return_info=True`.
- The explicit deterministic `Half(base=0.5, stochastic=False)` path uses `exp2(-deficit)` instead
  of the generic `pow` expression. Other bases still use the general path.
- When `dim` and `hidden_dim` are known, `Pulse` shares the first
  `Linear + GELU` trunk between fragment projection and fold assignment, then
  uses separate lightweight heads. This removes a duplicate tiny MLP trunk while
  keeping the same public `Pulse(k=..., dim=..., hidden_dim=...)` API.

The reference variants in the Pulse and Qwen benchmarks keep the previous
norm+max guidance and `einsum` aggregation. This lets the benchmark report
whether the optimized path keeps accuracy while improving throughput and memory.

Additional experimental knobs are available for resource-sensitive settings:

- `Fold(topk=...)` keeps only the strongest input slots per folded output slot.
  It can reduce assignment work, but the Qwen adapter probe shows that using it
  inside corrected Pulse can damage held-out tokenization accuracy.
- `Fold(mode="attention", heads=...)` uses a small learned query workspace and
  PyTorch scaled dot-product attention. This is useful to test Perceiver-style
  folding on larger fragment sets; it is not the small-fragment default.
- `Pulse(q_topk=...)` prunes by external survival guidance before projection.
  It is useful when `q` is already trusted and sparse. In the current probes it
  improves base-Pulse throughput, but `q_topk + iteration` gives worse held-out
  Qwen accuracy than the default corrected path. Use the separate `mask=...`
  argument for padding or validity masks; a mask is not a salience ranking.
- `Pulse(correction_mode="gated")` provides a lighter gated residual correction. It is
  stable, but the current Qwen probe favors the default shared-trunk MLP iteration.
- `torch.compile` can accelerate fixed-shape Pulse inference when Triton is
  available. On Windows this was tested with `triton-windows`; use workspace
  cache directories such as `TRITON_CACHE_DIR=.tmp/triton-cache` and
  `TORCHINDUCTOR_CACHE_DIR=.tmp/torchinductor-cache`. Compilation has warmup
  cost, so it is an inference deployment option rather than a training default.

References used for this pass:

- PyTorch `torch.bmm`: explicit batched matrix multiplication for 3-D tensor
  batches.
- PyTorch `torch.einsum`: general contraction notation, useful but less
  specific than `bmm` for this two-input batch matmul.
- Perceiver / Perceiver IO: latent bottleneck designs that distill large
  inputs into smaller latent workspaces.
