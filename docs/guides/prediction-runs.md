# Recoverable Task Prediction Runs

Use the same durable batch workflow for classification, your own TaskPredictor,
or a language-model deployment. The task's original JSON predictions remain
unchanged: ARTI does not require text, labels or token metrics for every task.

## Start And Resume

```python
import arti

run = arti.PredictionRunConfig(
    "outputs/prediction",
    batch_size=64,
    predict_kwargs={"batch_size": 16},
)
result = arti.run_prediction(
    "outputs/model", "requests.jsonl", run,
    task_factory="my_package.tasks:build_task",
)
result = arti.run_prediction(
    "outputs/model", None, run, resume=True,
    task_factory="my_package.tasks:build_task",
)
```

The outer `batch_size` bounds one committed output partition, defaulting to
128 records. `predict_kwargs` are arguments to the task's prediction callback;
its internal batch size can differ. Omit that option to keep the callback's
default. Classification uses its native defaults; the generic language path
resolves native generation defaults. The saved plan pins both policies.

The input is JSONL with unique nonempty string IDs. `id_field` defaults to the
built-in asset's field mapping or `id` for a custom task; `output_id_field`
defaults to `id`. A custom task may supply both explicitly. Its callback returns
one JSON-compatible record per input in the original order. It owns media
decoding and prediction semantics; no automatic label conversion is performed.

```bash
arti recipe predict-run outputs/model --data requests.jsonl --run-dir outputs/prediction --batch-size 64 --task-factory my_package.tasks:build_task --predict-kwargs-json prediction-options.json
arti recipe inspect-prediction outputs/prediction
arti recipe predict-run outputs/model --run-dir outputs/prediction --resume --task-factory my_package.tasks:build_task
```

For built-in classification, omit `--task-factory`. Its feature schema, label
interpretation and original probabilities are preserved. Custom task factory
kwargs use `--task-factory-kwargs-json`. Data/factory descriptions and predictions
stay separate from training optimizer state: this workflow never builds a
trainer or optimizer.

Existing `CausalLMPredictionRunConfig`, `run_causal_lm_prediction` and their saved
plan/receipt formats continue to work through the same executor. The normal
causal-LM CLI retains its native batch/generation profile; see the
[language prediction guide](qwen-prediction-runs.md). New generic plans do not
invent `generated_tokens` even when the task is non-generative.

## Durable Progress

The initial input snapshot, ordered IDs, source model manifest, task identity,
partition size and callback options are pinned in `plan.json`. A partition
publishes `predictions.jsonl` and its receipt together by an atomic directory
rename. That committed part, not mutable `run.json` progress, is authoritative.

Recovery adopts a published receipt if the progress write was interrupted.
Completed parts are not predicted again. A failed/unpublished batch can execute
again; this is durable result recovery, not physical exactly-once inference.
The original error and the uncommitted requested record count remain recorded.
No automatic retries or hidden remote scheduler are introduced.

When all parts exist, the final JSONL is streamed from them and atomically
published. A missing/changed final file can be rebuilt with zero model calls.
You can finish this stage without the model asset, task code or original data:

```python
result = arti.run_prediction(None, None, run, resume=True)
```

```bash
arti recipe predict-run --run-dir outputs/prediction --resume
```

Omitting the model is valid only when no prediction remains. A supplied model
must still match the pinned source, even for a completed run. An unfinished run
requires its original source asset at its current location. The input snapshot
inside the run makes original data deletion safe; a whole run/model directory
can move without changing source identity. Changed input, explicit policy or
model source are reported before loading a host model.

## State And Costs

The model loads once per attempt. Each prediction preserves declared owned
parameters/resources, registered buffers, modes and supported RNG. These are
independent read-only batches, not a continuously self-modifying session.
Callback input identity is captured before preprocessing, and nested options
are passed as a fresh value so local mutation cannot change the pinned policy.

Only one input/output partition is processed at a time, plus bounded callback
and model state. The unique-ID set remains O(record count); this is not a
constant-memory index or a byte-capacity promise. A callback's temporary memory,
large frozen-host state ownership and arbitrary external Python state remain
the developer's responsibility.

`max_seconds` stops new batch starts after model loading; it does not abort an
active callback or impose a deadline on input validation, loading or final
merge. `stop_requested` similarly acts at batch boundaries. Each manual attempt
has its recorded allowance, with actual prediction seconds, new/reused rows and
committed parts reported. No claims about GPU throughput follow from local CPU
lifecycle tests.

Use one local writer per run directory. Distributed writers, sharded/native-JAX
execution, remote retries, continuously mutable sessions, arbitrary unregistered
state and external factory/base byte-identity audits are not qualified here.
Task evaluation remains independent; use actual held-out labels, not invented
answers from its own predictions. [Offline Prediction Scoring](prediction-scoring.md)
scores committed parts with no model call. The exported predictor remains the
entry point for model-dependent metrics, such as losses requiring logits that
were not saved by prediction.
