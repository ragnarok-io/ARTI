# Train, Resume, Evaluate And Export

For model-independent data/loss/evaluation/deployment integration, use
[Custom Task Workflows](custom-task-workflows.md). Its native Python task
factory reuses this same training session, without requiring a language model.

For interrupted classification or Qwen SFT jobs, use
[Recover a Task Run](recipe-recovery.md) to finish the original target and
remaining evaluation/export stages. The shared session's ordinary `fit(steps=N)`
continues to mean N additional optimizer updates.

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
its accumulation contexts and distributed mechanics; local single-process checkpoint
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

The local `torch` and single-process `accelerate` engines record a run by default. With no output path,
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
On Windows, brief file-sharing/indexing locks during atomic replacement receive
a bounded retry (at most 0.31 seconds of waits). A persistent failure still raises
and does not silently replace the previous complete JSON/tensor file.

Time budgets stop at optimizer boundaries. One in-flight step, evaluation and
final save can exceed the requested wall time; this API is not a process killer.
Use `stop_requested=callable` for an orderly external stop at the same boundary.
`KeyboardInterrupt` or an exception records the interruption and retains previous
checkpoints; it does not claim that partial CUDA/optimizer work is recoverable.

Losses are detached and transferred together at reporting/checkpoint boundaries,
not converted to Python once per microbatch. The orchestration remains outside
the neural graph; this change alone is not a physical CUDA speed benchmark.

This local run policy applies to Torch and single-process Accelerate. See
[Train With Local Accelerate](accelerate-training.md). Transformers and distributed
Accelerate retain their engine-owned lifecycle; sharded recovery is not silently
replaced by local checkpoint rules.

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

## Select And Stop On Validation

Selection and early stopping are optional. With no validation policy, runs keep
the final model and do not stop for short-term metric regressions. A policy uses
only an explicit validation source and cadence:

```python
from arti import ARTITrainingRunConfig, ARTIValidationConfig

run = ARTITrainingRunConfig(
    output_dir="outputs/selected",
    evaluation_steps=50,
    validation=ARTIValidationConfig(
        monitor="loss", mode="min", selection="best",
        patience=6, min_delta=1e-4, start_step=100, min_evaluations=4,
    ),
)
result = session.fit(train_loader, run=run, eval_data=validation_loader)
```

Patience counts eligible validation calls, not epochs/updates. Warmup checks before
`start_step` do not select/stop. `min_evaluations` guards early stopping; set
`patience=None` for selection without stopping. Strict metric improvement selects
the best snapshot; ties keep the earlier one. `min_delta` applies to patience
reset, so a small genuine improvement can still become the best model.

The run records the declared metric/direction, scores, bad-check count and chosen
step. An off-cadence final check selects an asset without changing patience.
If the time budget has expired, it skips this extra check. When no check is
eligible, selection explicitly falls back to last (`no_eligible_validation`).

For task-specific metrics, supply `validation_fn(session) -> mapping`. It runs
inside the read-only evaluation context. Recipe validation uses its real task
metrics; a custom callback must return the declared numeric monitor and use
validation data, not test. A missing/nonfinite monitor is an actionable error,
not a silently ignored selection condition.

The result distinguishes three paths:

- `checkpoint_path`: last complete training state, used for normal continuation.
- `best_checkpoint_path`: the best complete snapshot, if a check qualified.
- `selected_checkpoint_path`: the policy's best/last choice for inference.

The live session and optimizer remain at last. To inspect best temporarily:

```python
with session.checkpoint_context(result.selected_checkpoint_path):
    metrics = session.evaluate(validation_loader)
    # Export model-only inference weights through your model's export API.
```

The context restores model/graph resources, modes, buffers and RNG. It does not
rewind the optimizer or data loader. Training-state saves/resume/fit are therefore
not allowed inside this model-only context: best weights with a last-step
optimizer are not a consistent resumable checkpoint. Explicitly resume from the
best checkpoint outside the context when an actual rewind is intended.

Resume with the same validation policy and `evaluation_steps`. Scores, patience,
scheduled check count and last-check position are restored. The new run imports
its inherited best bundle once, outside the neural hot path. Preserve the whole
run tree when moving run checkpoints, including the referenced best bundle;
`save_checkpoint` or Hub/pretrained training export to a separate directory packs
a `validation-best` companion there. Model-only inference export needs neither
the optimizer nor that training companion.

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

`reporter=callable` receives structured lifecycle/metrics/checkpoint/evaluation
events. Tracker SDKs remain optional. Remote failures disable that reporter
with a warning; the local
asset and event log remain available. The SDK does not upload anything by default.

`ReporterGroup` fans out to independent sinks. A failed sink is disabled without
disabling the others; its error type is recorded in `run.json`. Events include a
local `run_id`, wall-clock timestamp and cumulative optimizer `step`. Continuation
checkpoints retain that run ID, and a subsequent run declares `parent_run_id`.
These IDs describe ARTI run lineage, not a remote service's run ID.

Training emits `run_started` and `run_finished`, including interrupted/failed
status. A task recipe additionally emits `workflow_stage`, final validation,
the completed inference `artifact` and `workflow_finished`. Training success
does not mean evaluation/export success. `run.json` describes training;
`task_result.json` describes the full recipe, and `events.jsonl` contains both.

### Optional Experiment Trackers

Install the desired optional SDK (`wandb` or `mlflow-skinny`); the source package
also provides separate `wandb`/`mlflow` extras. Importing ARTI or the reporters
does not import these SDKs, log in, create a run or select a server.

```python
import wandb
from arti import ReporterGroup, WandbReporter

with wandb.init(project="my-task", mode="offline") as tracked_run:
    reporter = ReporterGroup(WandbReporter(tracked_run), print)
    result = recipe.fit(reporter=reporter)
```

The W&B reporter uses `arti/optimizer_step` as a custom chart axis. It does not
overwrite W&B's history step, so checkpoint/validation events at the same
optimizer step and restored continuations do not silently drop records.

```python
import mlflow
from mlflow import MlflowClient
from arti import MLflowReporter

# Select your tracking URI before starting the run.
with mlflow.start_run() as tracked_run:
    reporter = MLflowReporter(MlflowClient(), tracked_run.info.run_id)
    result = recipe.fit(reporter=reporter)
```

Both adapters use caller-owned runs: the SDK context manager or caller controls
termination. By default they send only numeric metrics and selected lifecycle
fields, not raw records, paths, run metadata or checkpoint files. Explicit
`upload_artifacts=True` uploads the completed recipe's inference bundle, including
its declared task metadata; checkpoint/data directories are not uploaded.
Remote SDK timeout/flush behavior remains SDK-controlled. MLflow metrics are
batched synchronously at reporting boundaries; W&B uses its SDK event queue.
This integration is outside the neural graph, but a slow tracking service can
still add wall-clock time at those boundaries. Choose a suitable `log_steps`.

Interfaces follow [W&B's Run API](https://github.com/wandb/wandb/blob/main/wandb/sdk/wandb_run.py)
and [MLflow's client API](https://mlflow.org/docs/latest/api_reference/python_api/mlflow.client.html).

The `run(..., reporter=...)` functions in `examples/classification_workflow.py`
and `examples/qwen_sft_workflow.py` forward the same reporter through training
and restored continuation. Their data/objective/evaluation/deployment behavior
is otherwise unchanged. MLflow's skinny client supports an existing tracking
service; a local SQLite backend additionally needs `sqlalchemy` and `alembic`
(or the full `mlflow` installation). No server is started by the adapter.

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

For native Transformers/TRL training, see
[Transformers and TRL Training](transformers-training.md). This uses the same
session/binding while the selected Hugging Face class owns the loop and native
checkpoint directory. Portable ARTI continuation and exact Trainer continuation
are explicitly different operations.

For caller-owned Ray jobs, see [Ray Train Checkpoint Handoff](ray-training.md).
Use `on_checkpoint` for required asset delivery; optional experiment reporters
retain their independent failure-isolation semantics.

```bash
python examples/attachment_training_workflow.py --output .tmp/workflow-example
python examples/shared_training_workflow.py --output .tmp/shared-workflow-example
```

The CPU example uses actual tensor resources and a learnable Connection, trains
with task loss, resumes, evaluates, exports and checks reload parity. Its small
regression task establishes lifecycle correctness, not downstream model quality.
See [custom task workflows](custom-task-workflows.md) and
[runtime profiles](task-runtime.md) for application-owned lifecycle contracts
and the declared local support boundaries.
