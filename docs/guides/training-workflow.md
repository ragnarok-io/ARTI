# Train, Resume, Evaluate And Export

Use one training session from an attachment, a pretrained provider, or a native
PyTorch model/program. Start from an explicitly configured `ARTILayer` for the
attachment path.
The task objective is the host's scalar `.loss` or `['loss']` by default. Use a
custom callable when the host does not calculate its loss:

```python
session = model.arti.trainer(
    objective=lambda host, batch: task_loss(host(batch["inputs"]), batch["labels"]),
    learning_rate=3e-4,
)
```

## Three Entrances

```python
# Attached host: selected ARTI parameter owners, no frozen host weight copies.
session = model.arti.trainer(objective="model_loss")

# Native program: no attachment or TITO requirement on the model itself.
session = arti.trainer(program_model, objective=task_objective)

# Provider model: same session, with its native task and asset bindings.
session = workflow.trainer()
result = workflow.fit(train_loader, run=run_config, eval_data=validation_loader)
```

All use `ARTITrainingSession` and `ARTITrainingConfig`. The historical
`ARTIAttachTrainingConfig` name refers to the same config class. `TrainingSpec`
is the serialized pretrained plan's projection of these settings and delegates
validation to that config, not a second execution engine.

Use `TrainingBinding` when execution, trainable parameter ownership and saved
state are different modules. Its checkpoint callbacks use the caller's existing
codec; it is not another trainer or neural layer. Native models normally save
their complete module state. Pretrained frozen-host workflows save all ARTI
adapter parameters and buffers, not just tensors marked trainable.

Pretrained `loss_fn(output, target)` and `step_fn(model, batch)` remain explicit.
When a model accepts labels and provides its native loss, the default keeps that
loss, including causal shifting/masking. Transformers' native loss handling stays
with Trainer. The explicit `loss_fn` is not silently replaced by it.

Optimizer updates and consumed microbatches have separate counters. Configured
accumulation and mixed precision apply to all three entrances. Accelerate owns
its accumulation contexts and distributed mechanics; local Torch checkpoint
rules do not masquerade as qualified FSDP/ZeRO recovery.

`workflow.fit()` reuses its session for continuation. It also accepts `steps=N`
for additional updates and `resume_from_checkpoint=path`. The default export
excludes a frozen host, but includes adapter buffers and the complete supported
training checkpoint state.
`workflow.trainer()` returns the existing session without resetting progress;
use `reset=True` only to explicitly start a fresh optimizer/session.

The pretrained CLI reads the same `[run]` settings. A data section may supply a
`validation_factory`/`validation_kwargs`, and continuation uses:

```bash
arti pretrained fit config.toml --weights continued.st --resume-from-checkpoint checkpoint/model.arti.st --steps 100
```

Use fresh output directories for each continuation. The stored model plan does
not change merely because a run uses a different output path or step budget.

The Tensor alignment objective is opt-in with `objective="tensor_alignment"`.
It is a corruption/reconstruction task, not a replacement for classification,
language modeling, ranking or a flow loss.

## Inspect Before Training

```python
plan = session.plan(steps=1000)
check = session.check(example_batch)
assert check["ok"]
```

The plan reports selected layers and trainable parameters. Its estimates are
static, not measured throughput. The check runs the actual objective and verifies
its gradient connection without an optimizer update. It preserves registered
module state, buffers and Torch/Python RNG. An inactive route may legitimately
leave some parameters unused; the result reports that rather than claiming every
branch trained.

## Run Locally

```python
from arti import ARTITrainingRunConfig

result = session.fit(
    train_loader,
    steps=1000,
    run=ARTITrainingRunConfig(
        output_dir="outputs/my-task",
        checkpoint_steps=50,
        checkpoint_seconds=60,
        log_steps=10,
        max_seconds=1800,
        evaluation_steps=100,
        metadata={"dataset": "my-task-v1", "split_manifest": "sha256:..."},
    ),
    eval_data=validation_loader,
)
```

The native `torch` engine records a local run by default. With no output path,
each invocation gets a separate directory under `.arti-runs/`. It saves the first
completed update, then at the configured step/time interval, and at orderly
completion. Checkpoints contain only ARTI-owned model tensors plus training state;
the frozen host is not copied into every checkpoint.

Use a fresh explicit output directory for each invocation, including continuation
runs. Existing run directories are not silently overwritten. Resume identifies
the input checkpoint independently of the new output directory.

```text
outputs/my-task/
  run.json
  events.jsonl
  checkpoints/step-00000001/model.arti.st
  checkpoints/step-00000050/model.arti.st
  ... existing SafeTensors/config/lock sidecars ...
```

`run.json` identifies the latest complete checkpoint, the task declaration,
dataset references, evaluation history and the stop reason. Model and optimizer
assets are saved before advancing the latest pointer. A crash in an unfinished
update does not replace the last completed checkpoint.

Time budgets stop at optimizer boundaries. One in-flight step, evaluation and
final save can exceed the requested wall time; this API is not a process killer.
Use `stop_requested=callable` for an orderly external stop at the same boundary.
`KeyboardInterrupt` or an exception records the interruption and retains previous
checkpoints; it does not claim that partial CUDA/optimizer work is recoverable.

Losses are detached and transferred together at reporting/checkpoint boundaries,
not converted to Python once per microbatch. The orchestration remains outside
the neural graph; this change alone is not a physical CUDA speed benchmark.

This local run policy currently applies to `engine="torch"`. Transformers and
Accelerate retain their engine-owned training interfaces. Their distributed,
sharded run lifecycle is not silently replaced by local checkpoint rules.

## Resume

```python
restored_model = arti.ARTI.load(fresh_base, result.checkpoint_path, layer=layer_factory)
resumed = restored_model.arti.trainer(
    objective=objective,
    resume_from_checkpoint=result.checkpoint_path,
)
resumed.fit(train_loader, steps=500, run=ARTITrainingRunConfig(output_dir="outputs/continued"))
```

`steps` means additional optimizer updates, and `result.steps` is the cumulative
step counter. Resume restores optimizer, scheduler, Torch/Python/CUDA RNG and AMP
scaler state when present.

An indexed sequence resumes at its saved microbatch position. A stateful loader
can supply `state_dict()` and `load_state_dict()`. A general iterator cannot
provide exact data-order recovery; the report explicitly labels that limitation.
NumPy RNG and arbitrary Python-side model state are not captured automatically.
The host weights and preprocessing must be restored from the same source.

## Evaluate Without Committing Adaptation

```python
metrics = session.evaluate(
    validation_loader,
    split="validation",
    weight=lambda batch: batch["labels"].numel(),
)
```

The default is a mean over batch losses. Explicit weighting can select sample or
valid-token units; account for padding in the callback. Evaluation preserves
individual module train/eval modes, registered buffers, ARTI state and RNG. It
does not commit incidental forward adaptation. Periodic validation must use a
reiterable data source. A genuine online-adaptation benchmark should use an
explicit state-committing execution loop instead.

Validation data is caller-supplied. ARTI does not synthesize labels, turn model
agreement into ground truth or convert a diagnostic split into a test split.

## Track And Export

`reporter=callable` receives structured metrics/checkpoint/evaluation events.
Connect it to an existing W&B or MLflow client without adding a tracker dependency
to the core. Remote failures disable that reporter with a warning; the local
asset and event log remain available. The SDK does not upload anything by default.

```python
asset = model.arti.save_pretrained(
    "outputs/model",
    base_model="org/base-model",
    revision="pinned-revision",
    training_session=session,
)
```

Native `ProgramGraph` attachments embed their existing graph artifacts in the
SafeTensors package, including default/active resource backing. A matching graph
or layer factory is still supplied on reload; arbitrary Python activation and
connection callables remain runtime-only. Graph, weight and optimizer state are
not separate, unbound files.

This is an ARTI asset, not an automatic LoRA conversion. vLLM, SGLang, NIM and
TensorRT-LLM compatibility requires a separately qualified inference profile.

## Executable Example

```bash
python examples/attachment_training_workflow.py --output .tmp/workflow-example
python examples/shared_training_workflow.py --output .tmp/shared-workflow-example
```

The CPU example uses actual tensor resources and a learnable Connection, trains
with task loss, resumes, evaluates, exports and checks reload parity. Its small
regression task establishes lifecycle correctness, not downstream model quality.
See [Task Recipes](task-recipes.md) and [Qwen SFT Workflow](qwen-sft-workflow.md)
for real labeled tasks, preprocessing and deployment boundaries.
