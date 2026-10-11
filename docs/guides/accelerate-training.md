# Train With Local Accelerate

Single-process Accelerate now uses the same ARTI run lifecycle as local Torch:
periodic SafeTensors checkpoints, budget/requested stops, weighted evaluation,
validation selection, early stopping, reporters and checkpoint callbacks. The
engine owns preparation, device placement, backward/AMP, gradient clipping and
optimizer/scheduler wrappers; ARTI retains program ownership and run state.
This is not a new trainer or a distributed checkpoint format.

## Existing Session Entrances

```python
import arti

session = model.arti.trainer(
    engine="accelerate", device="cpu", objective="model_loss",
    gradient_accumulation_steps=2,
)
result = session.fit(
    train_data, steps=100,
    run=arti.ARTITrainingRunConfig(
        output_dir="outputs/first", max_seconds=300,
        checkpoint_steps=20, evaluation_steps=20,
    ),
    eval_data=validation_data,
    objective_weight=lambda batch: batch["labels"].numel(),
    evaluation_weight=lambda batch: batch["labels"].numel(),
)
```

That sample weight applies to sample-level labels. Causal LM uses supervised
shifted, non-masked token counts instead; the Qwen recipe provides that weight
automatically. Standalone `arti.trainer(model, ...)` and pretrained
`workflow.trainer()`/`workflow.fit()` use the same session boundary.

Accelerator preparation occurs once per session. Repeated `fit(steps=N)` continues
N additional optimizer boundaries without rewrapping the model, optimizer or
scheduler. A new run can retain its own output directory and budget. A checkpoint
restores the existing selected parameter owners, optimizer/scheduler state,
supported RNG/scaler state and supported data cursor, not a copy of the frozen
host weights.

## Qwen And Classification Recipes

Use the normal recipe with:

```toml
[training]
engine = "accelerate"
device = "cpu"
mixed_precision = "no"
gradient_accumulation_steps = 2
```

```bash
arti recipe check recipe.toml
arti recipe train recipe.toml --run-dir outputs/first
arti recipe resume-run outputs/first --plan-only
arti recipe resume-run outputs/first --run-dir outputs/continued
```

Data preparation, native task objectives, custom layer factories, validation,
export, offline native-base packaging and serving keep their existing contracts.
An interrupted recipe recovers its original target; completed training can finish
evaluation/export with zero new updates. Best-model export does not rewind the
training session's last optimizer state. Engine identity is recorded: switching
a Torch checkpoint to Accelerate is not an implicit exact-resume operation.

## Correct Accumulation

For microbatch mean losses `L_j` and sample/token counts `w_j`, one boundary uses
`sum(w_j * L_j) / sum(w_j)`. It does not average unequal-token microbatch means.
The shared loop knows the complete accumulation group's denominator before
backward. Accelerate's automatic accumulation divisor is accounted for explicitly,
so it is not applied twice. The native plugin disables dataloader-end implicit
flushes; ARTI ends a group at its declared optimizer boundary.

Loss transfer/reporting remains batched at reporting/checkpoint boundaries.
Budget and requested stops are checked before starting the next group. The
current group's device work and required save can finish after a time allowance;
the allowance does not cover model construction, final evaluation or export.
This engine integration alone is not a CUDA throughput claim or removal of the
host training loop.

## Qualification Boundaries

The local profile requires one process. A multiprocess launch cannot masquerade
as a single-writer run. The existing `distributed=True` engine entrance remains
separate; this feature does not qualify DDP/FSDP/DeepSpeed collective resume,
sharded optimizers, TPUs or JAX run state.

Indexed sequences and supported stateful loaders restore their cursor. Generic
iterators remain `weights_optimizer_rng_only`, not exact data-order recovery.
Local CPU float32 trajectory parity and CPU BF16 training are tested; GPU FP16
overflow/skipped-update behavior and GPU performance need their own qualification.
Accelerate owns one process-wide precision configuration, so changing precision
requires a fresh process rather than resetting its global internals in training.

Install Accelerate through the existing optional integration dependency. Ordinary
Torch recipes still work without it. Local artifacts/reporters need no account,
tracking server, model download or cloud job.

References:
[Accelerate accumulation and variable-size token weighting](https://huggingface.co/docs/accelerate/usage_guides/gradient_accumulation),
[Accelerator API](https://huggingface.co/docs/accelerate/package_reference/accelerator).
