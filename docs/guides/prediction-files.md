# Durable Task File Outputs

Opt into files with the existing recoverable prediction runner. The application
owns output formats, media dimensions, schema metadata and optional validation.
ARTI owns the committed file closure and resume boundary, not a media codec.

```python
run = arti.PredictionRunConfig(
    "outputs/prediction", batch_size=16,
    predict_kwargs={"batch_size": 8},
    files=arti.PredictionFilesConfig(
        field="files",
        validator="my_package.validation:check_outputs",
        revision="my-file-contract-revision",
    ),
)
result = arti.run_prediction(
    "outputs/model", "requests.jsonl", run,
    task_factory="my_package.tasks:build_task",
)
```

`files=None` is the original JSON-only behavior. File output requires an
explicit generic TaskRecipe producer; built-in classification/causal predictors
are not silently converted to file generators. The file field cannot replace
the output ID field. The output configuration and validator identity are pinned
in the existing prediction plan.

## Task Callback

The existing `predict_fn(model, records, ...)` receives `output_dir` only when
file output is enabled. It writes files under that temporary directory and
returns one native JSON result per input, in original ID order:

```python
def predict(model, records, *, output_dir, batch_size=8):
    outputs = infer_native_batch(model, records, batch_size=batch_size)
    results = []
    for index, (record, value) in enumerate(zip(records, outputs, strict=True)):
        relative = f"sample-{index}/result.bin"
        destination = output_dir / relative
        destination.parent.mkdir(parents=True)
        write_my_format(destination, value)
        results.append({
            "id": record["id"],
            "prediction": summarize(value),
            "files": {
                "result": {"file": relative, "metadata": {"encoding": "my-format"}},
            },
        })
    return results
```

A string relative path is shorthand for `{file: path, metadata: {}}`.
Output names are application labels. The callback does not supply authoritative
checksums/byte counts: the runner computes them over the actual copied files.
Only declared files are included; scratch residue is not delivered. Different
names may refer to one physical file. Empty file mappings are allowed when the
task explicitly returns a structured outcome without an artifact; an optional
validator can require files for that application's protocol.

Files are copied/hashed in streams, not base64-encoded in JSON or loaded as
large host byte arrays by the core. The producer's own codec/memory behavior is
application code. Both staged source and committed closure are immutable during
validation/publication; concurrent file mutation is outside this local contract.

## Validate And Commit

An optional installed callback checks application formats before publication:

```python
def check_outputs(inputs, predictions, *, directory):
    for row in predictions:
        path = directory / row["files"]["result"]["file"]
        inspect_my_format(path, row["files"]["result"]["metadata"])
    return {"ok": True, "samples": len(predictions)}
```

It receives original input copies, native result copies, staged relative file
paths, content digests/byte counts and the file directory. Return finite JSON
with `ok=True`, or reject/raise. It is format evidence, not automatically a task
quality score. `PythonFactoryConfig` supplies explicit validator kwargs. The
revision must change when the validator semantics/dependencies change; it is
caller-declared, not automatic code-byte identification.

ARTI rechecks the file closure after validation, then atomically publishes
`parts/<index>/` containing JSON results, files and receipt. The committed JSON
contains run-relative file paths, checksums, byte counts and metadata. Validator
output is content-identified in the part receipt. Producer scratch is deleted.
Missing files, failed producers/copy/validators/publication never leave partial
results visible as a committed part. Previously committed parts remain valid.

`inspect_prediction` verifies all declared files without a model/task/validator
import. A path by itself is not a completed result. Move the whole prediction
directory and open paths relative to its new root. The merged JSON is a derived
view; deleting it only requires re-merging existing parts, not re-generating
files. No promise of filesystem power-loss durability, distributed exactly-once
generation or multi-writer arbitration is made.

## Resume And Accounting

```python
continued = arti.run_prediction(
    "outputs/model", None, run, resume=True,
    task_factory="my_package.tasks:build_task",
)
completed = arti.run_prediction(
    None, None, arti.PredictionRunConfig("outputs/prediction"), resume=True,
)
```

An incomplete run needs its pinned model/deployment. Completed retries and
merge-only recovery require no model, input source or validator. Specified
file/validator configuration must match the plan; omitted configuration reuses
the pinned behavior. Complete portions are never replayed. Uncommitted callback
work may run again after an interruption; callbacks must not perform ambiguous
external actions. Interactive actions have a separate planned lifecycle.

Reports expose logical committed/new/reused records, physical output file/byte
counts and file preparation seconds. Prediction seconds include producer work;
file preparation measures copying, checksums and format validation, excluding
the final atomic publication. Attempt elapsed time also includes model loading
and publication. These are observed counters, not a throughput guarantee or a
host-enforced deadline. `max_seconds` still controls new batch starts after
loading; it cannot preempt a codec/model call already running.

## CLI And Validation

```bash
arti recipe predict-run outputs/model --data requests.jsonl --run-dir outputs/prediction --task-factory my_package.tasks:build_task --output-files-field files --file-validator my_package.validation:check_outputs --file-validator-id my-file-contract-revision
arti recipe inspect-prediction outputs/prediction
arti recipe predict-run --run-dir outputs/prediction --resume
```

The same file options work with `recipe validate-deployment` and
`validate_recipe_deployment(..., run, ...)`. Reuse the existing task-owned scorer
and explicit criteria; execution/file completeness and quality remain distinct.
`--file-validator-kwargs-json` supplies its declared JSON kwargs. JSON-only
direct prediction/ASGI endpoints do not acquire file-delivery behavior merely
because this option exists.

`examples/asset_task_recipe.py` includes caller-owned array, binary scalar and
tiny bitmap output/validation fixtures over native predictions. They demonstrate
the general file lifecycle, not image-generation quality or a built-in codec.
No external task data, names, protocols or fixed media constants enter the core.

Reference checked 2026-10-11:
[Ray directory checkpoints](https://docs.ray.io/en/latest/train/user-guides/checkpoints.html)
treat required auxiliary files as part of the saved directory, not an orphaned
path. Transfer that closure ownership principle; ARTI keeps its existing atomic
prediction journal rather than adding a Ray result engine or dependency.
