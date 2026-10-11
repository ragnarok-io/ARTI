# Resume A Qwen Prediction Run

For classification or custom model-independent tasks, use
[Recoverable Task Prediction Runs](prediction-runs.md). Both profiles now
share the same durable executor; this guide retains the native language profile.

Generate a large local JSONL request set in committed batches. A completed batch
is saved immediately; restarting does not regenerate that batch. The model loads
once per attempt, and input/output text is processed one batch at a time.

## Start And Resume

Each record needs a unique nonempty string ID and a complete message history
ending in a user message, using the asset's saved field names:

```json
{"id":"request-001","messages":[{"role":"user","content":"A task input"}]}
```

```bash
arti recipe predict-run deployments/candidate --data requests.jsonl --run-dir outputs/prediction-v1 --batch-size 16 --max-new-tokens 64
arti recipe inspect-prediction outputs/prediction-v1
arti recipe predict-run deployments/candidate --run-dir outputs/prediction-v1 --resume
```

The run uses a fresh directory. `--resume` is explicit; an existing run is not
silently overwritten. Batch size and generation length default to the exported
asset's profile on creation, and to the pinned run policy on resume. Explicit
changes to either setting require a new run. The serialized model manifest,
including its declared ARTI weight identities and pinned base reference, is
also fixed for this run.

Loading defaults to CPU; request `--device cuda` explicitly for an authorized
GPU workload. `--base-model` supports the existing local native-base override.
The caller remains responsible for the correspondence of arbitrary supplied
base weights to the pinned model/revision; this workflow does not rehash a
multi-GB base on every retry. Custom programs still need `--layer-factory` and
optional `--factory-kwargs-json` or an explicit Python factory. Saved metadata
does not execute third-party Python code by itself.

## Durable Results

```text
prediction-v1/
  plan.json
  input.jsonl
  run.json
  parts/
    00000000/
      predictions.jsonl
      receipt.json
    00000001/
      predictions.jsonl
      receipt.json
  predictions.jsonl
```

Creation validates IDs/messages and copies the input snapshot in one read,
recording its content digest. The original request file can then move or be
deleted. The run itself and a native bundled model package can also move; pass
the model's new path and the same copied run to resume.

A batch publishes its complete output and receipt together in a fresh directory.
The receipt binds ordered input, output digest/count and pinned plan. These
receipts, not the mutable progress counter in `run.json`, define committed work.
If interruption happens after batch publication but before progress logging,
inspection/recovery adopts that batch. Missing or invalid committed outputs
produce a consistency error rather than a silently reduced completed count.

Final consolidation streams committed files into one atomic `predictions.jsonl`.
If consolidation fails, all batch outputs remain; resume can rebuild the final
file without loading or calling the model. A completed run is likewise a no-model
operation on retry. No reply is rescored, rewritten or extracted from prose.
IDs, text, token IDs/counts and finish reasons use the existing predictor format.

Output comparison/scoring can use explicit real labels joined by ID through
the existing `score_causal_lm_predictions` interface. Prediction does not invent
labels or application-specific quality thresholds.

## Inspect Progress Without A Model

`inspect-prediction` validates the input snapshot and committed batch receipts
and returns committed/remaining records, generated tokens, prediction time,
attempt count, and one action:

- `resume_prediction`: generate remaining batches.
- `finalize`: only merge already committed outputs.
- `already_completed`: final output is ready.

It reads text/metadata but does not instantiate a model, decode model tensors or
create an optimizer. It scans relevant files; it is not a constant-time telemetry
endpoint. Batch output digest, ordered IDs/counts and concatenated digest are
checked in one pass over each committed output.

Run reports separate reused records from newly predicted records. Attempts keep
failure/interruption status and uncommitted requested rows. A batch computed but
not published can be recomputed: committed-result reuse is not a guarantee of
exactly-once physical inference. No automatic retry of unchanged model failures
is performed.

## Budget And State Semantics

```bash
arti recipe predict-run deployments/candidate --run-dir outputs/prediction-v1 --resume --max-seconds 300
```

This is an allowance for starting new batches after model loading, per attempt.
It is not a wall-clock deadline for input validation, model loading, final merge
or an already running generation. The current batch completes and commits before
a stop; no CUDA kernel or generation thread is force-killed. Zero allows no new
batch and does not load the model. Callback-requested stop works at the same
boundary. A fully committed run can still finalize with zero new-generation work.

This workflow is **independent batch prediction**. Each batch snapshots/restores
owned ARTI parameters/resources, registered host buffers, modes and supported
Torch/Python/CUDA RNG through the existing evaluation binding. Internal forward
operations may run, but their persistent state is not carried to the next batch.
Use an explicit session runtime for continuously self-modifying conversations;
this is not that runtime. Arbitrary unregistered Python state, external random
generators and caller mutation of frozen host parameters are outside this
qualified restore scope. Model loading preserves the supported existing RNG.

Only one local writer is supported per run directory. This is not a concurrent
distributed scheduler or remote object-store transaction. Storage needs the input
snapshot plus committed outputs and final merged output. Text/results are not
accumulated in RAM; global unique-ID validation still needs an O(record-count)
ID set. This does not promise constant memory for arbitrarily large ID catalogs.

## Python API

```python
import arti

run = arti.CausalLMPredictionRunConfig(
    "outputs/prediction-v1", batch_size=16, max_new_tokens=64,
    max_seconds=300,
)
result = arti.run_causal_lm_prediction(
    "deployments/candidate", "requests.jsonl", run,
    local_files_only=True,
)
progress = arti.inspect_causal_lm_prediction(run.output_dir)
result = arti.run_causal_lm_prediction(
    "deployments/candidate", None, run, resume=True,
    local_files_only=True,
)
```

Pass an already loaded `predictor=` to reuse it. `stop_requested()` returns a
boolean before the next batch. `on_batch(report, receipt)` runs only after a
durable batch commit; its exception/interruption does not erase that result.
The old one-shot `recipe predict` entrance is retained for small files.

The supported profile is local, single-writer greedy Qwen text inference. Actual
CPU tiny-Qwen, native package and installed-wheel checks qualify the lifecycle,
not new task quality, GPU throughput or distributed recovery. Reference patterns:
[Ray batch iteration](https://docs.ray.io/en/latest/data/iterating-over-data.html)
and [partitioned JSON output](https://docs.ray.io/en/latest/data/api/doc/ray.data.Dataset.write_json.html).
