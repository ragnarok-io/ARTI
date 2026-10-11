# Task Recipes

One recipe connects records, preprocessing, task loss, checkpoints, evaluation
and the deployable model. The first supported recipe is supervised dense-feature
classification with the local Torch engine. It uses the existing shared training
session, not an additional trainer.

[Recover a Task Run](recipe-recovery.md) explains how to finish an interrupted
task or retry evaluation/export without duplicating training.

[Import Local Qwen SFT Data](qwen-data-import.md) accepts explicitly split chat
and prompt/completion exports without hand-writing a JSONL conversion script.

## Run A Real Task

The digit recognition example uses scikit-learn's offline `load_digits` dataset:
1,797 labeled 8x8 grayscale images. Preparation needs scikit-learn; training and
inference need only ARTI's core dependencies.

```bash
python examples/classification_workflow.py --output .tmp/digits-workflow
python examples/classification_workflow.py --select-best --output .tmp/digits-selected
```

This runs preparation, gradient preflight, 200 optimizer updates, checkpoint
reload, 40 additional updates, independent test evaluation, inference export and
reload parity. It runs on CPU and does not download a base model. This is a real
task workflow example, not an ARTI-versus-Transformer effectiveness benchmark.

For separate commands:

```bash
python examples/classification_workflow.py --prepare-only --output .tmp/digits-cli
arti recipe plan .tmp/digits-cli/recipe.json
arti recipe check .tmp/digits-cli/recipe.json
arti recipe train .tmp/digits-cli/recipe.json --run-dir .tmp/digits-cli/first
arti recipe train .tmp/digits-cli/recipe.json --resume-from-checkpoint .tmp/digits-cli/first/checkpoints/step-00000200/model.arti.st --steps 40 --run-dir .tmp/digits-cli/continued
arti recipe evaluate .tmp/digits-cli/recipe.json --model .tmp/digits-cli/continued/model --split test --output .tmp/digits-cli/test.json
arti recipe predict .tmp/digits-cli/continued/model --data .tmp/digits-cli/unlabeled.jsonl --output .tmp/digits-cli/predictions.jsonl
```

Use the `checkpoint` path returned by train, or `run.json`'s `latest_checkpoint`, rather than
guessing a step when a time budget or orderly stop ended the run early.

## Data Contract

Each JSONL record has explicit split membership and a stable, globally unique ID:

```json
{"id":"record-001","features":[0.2,1.0,0.7],"label":"accepted","split":"train"}
{"id":"record-002","features":[0.4,0.3,0.1],"label":"rejected","split":"validation"}
{"id":"record-003","features":[0.8,0.1,0.2],"label":"accepted","split":"test"}
```

Features are finite numeric vectors of one width. Labels are strings. Field
names are configurable; class order is explicit or inferred from **train only**.
Train and validation are required; test is optional until an explicit test
evaluation. Unknown validation/test classes are errors, not silently reassigned
label IDs. Real image/text/task encoders belong upstream of this dense-feature
adapter; arbitrary tensor shape does not imply correct semantic preprocessing.

```toml
task = "classification"
seed = 17

[data]
path = "records.jsonl"
labels = ["accepted", "rejected"]
batch_size = 64
preprocessing = "standardize"

[data.fields]
id = "id"
features = "features"
label = "label"
split = "split"

[model]
hidden_dim = 64

[training]
device = "cpu"
learning_rate = 0.003
steps = 200
gradient_accumulation_steps = 1

[run]
output_dir = "runs/first"
checkpoint_steps = 50
evaluation_steps = 50
```

Paths are relative to the recipe file, not the shell's working directory. The
resolved configuration and split IDs are recorded locally. Standardization
mean/scale are fitted only on training features and become model buffers; zero
variance features use scale 1. `preprocessing = "none"` is explicit.

This first data adapter loads dense features into memory. Large streaming,
multimodal and sharded recipes are not implied by this implementation.

## Training And Evaluation

The default model is a small fixed Formula program: Contract, Broadcast, Add,
GELU, then a classification projection. Operands are actual trainable tensors
bound to the existing Formula graph. Its resource graph is lowered once to a
tensor-only execution plan; each minibatch does not interpret the graph again.
It does not claim that architecture search chose this particular model.

The objective is cross-entropy over real labels, not tensor alignment. Training
uses epoch-shuffled batches with a saved seed, epoch and cursor. Resume restores
optimizer/RNG/precision state and data position. It verifies task identity and
batching/accumulation semantics. `--steps` always means additional updates.

```python
import arti

recipe = arti.ClassificationRecipe.from_config("recipe.toml")
print(recipe.plan())
assert recipe.check()["ok"]
result = recipe.fit()
metrics = recipe.evaluate("test", output="outputs/test.json")
```

Training reads validation, not test, and exports the final completed model.
This is the default. Optional validation-driven selection is configured as:

```toml
[run.validation]
monitor = "accuracy"
mode = "max"
selection = "best"
patience = 6
min_evaluations = 4
min_delta = 0.001
```

Keep `run.evaluation_steps` explicit. Omit `patience` to select without stopping.
The workflow returns `checkpoint` for last-state resume and `selected_checkpoint`
for the best/last inference source. It evaluates and exports the selected source
without rewinding the live recipe/optimizer. The exported model's `selection`
metadata records the actual choice; task declaration fields retain the default.
`recipe.evaluate("test", checkpoint=result["selected_checkpoint"])` evaluates the
same chosen model. Only validation selects it; test is a final independent check.

There is no hidden selection against test accuracy. Evaluation reports actual
sample-mean cross-entropy, accuracy, macro F1 and a labeled confusion matrix.
The companion `*.predictions.jsonl` retains record IDs, true labels, predictions
and probabilities for inspection. Unequal final batch sizes do not distort the
sample mean. Probabilities are not claimed to be calibrated confidence scores.

Periodic and final checkpoints use the existing `arti.st` protocol. A run also
contains `resolved_recipe.json`, `split_manifest.json`, `validation.json`, its
prediction records and `task_result.json`. See [Training Workflow](training-workflow.md)
for checkpoint timing and abrupt-process-stop limitations.

The resolved recipe and split manifest are written before the first update.
The resolved record includes this run's step/directory overrides and absolute
local data path; paths are excluded from the portable model declaration.

## Deploy The Same Asset

`<run>/model/model.st` and its bound sidecars contain all model tensors, native
graph declarations, preprocessing, field mapping and label order. No training
dataset, original configuration path, pickle or scikit-learn runtime is needed
for loading. Move the **whole model directory**, not just the `.st` file.

The exported asset can also evaluate new labeled records, without the training
recipe or original dataset and without refitting preprocessing:

```bash
arti recipe evaluate outputs/model --data new-labeled.jsonl --output outputs/new-evaluation.json
```

The API equivalent is `predictor.evaluate_records(records)`. Its metrics and
raw-ID prediction receipt use the same evaluation implementation as training.

```python
predictor = arti.ClassificationPredictor.from_pretrained("outputs/model")
results = predictor.predict_records([
    {"id": "request-1", "features": [0.2, 1.0, 0.7]},
    {"id": "request-2", "features": [0.4, 0.3, 0.1]},
])
# Load predictor once in a worker; call it for subsequent request batches.
```

Predict does not require labels or split fields. It applies the saved training
preprocessing and returns labels plus probability vectors in `predictor.labels`
order. Empty request batches return an empty list.

To export a training checkpoint separately:

```bash
arti recipe export recipe.toml --checkpoint outputs/first/checkpoints/step-00000200/model.arti.st --output outputs/model
```

The exported inference asset omits optimizer state. It is not a substitute for a
resumable training checkpoint.

The standard-library HTTP example loads one model once and accepts batches:

```bash
python examples/serve_classification.py outputs/model --port 8080
```

`POST /predict` accepts `{"records": [...]}`; `GET /health` reports readiness and
label order. This is a local serving integration, not an authenticated production
gateway or a qualified vLLM, NIM, TensorRT-LLM or cloud deployment profile.

## Cross-Request Serving

The optional ASGI worker serves classification and Qwen through the same path:

```python
from arti import RecipeServingConfig, create_recipe_app

app = create_recipe_app(predictor, config=RecipeServingConfig(
    batching="auto", max_batch_size=32, batch_wait_ms=4,
    max_pending_records=1024,
))
```

```bash
arti recipe serve outputs/model --batching auto --max-batch-size 32 --batch-wait-ms 4
python scripts/check_recipe_service.py outputs/model requests.jsonl --concurrency 16 --limit 16 --batch-wait-ms 10 --output outputs/service-check.json
```

Builtin read-only predictors combine compatible requests into one ordinary tensor
batch. Results preserve each request's record order/IDs, even when a large request
uses multiple dispatches. `on` declares independent custom predictor rows; `off`
keeps one request per dispatch. `auto` preserves caller-managed forward-state
semantics. Custom predictors can provide `validate_records` for model-free input
checks. Builtins do so before queueing, isolating bad input from valid batches.

One worker uses one loaded model sequentially, with model work in a thread and
HTTP awaiting asynchronously. FastAPI lifespan starts/closes the dispatcher.
HTTPX `ASGITransport` does not run lifespan automatically; tests/embedded callers
should use `async with app.router.lifespan_context(app)`. Lazy startup is supported,
but direct users must close `app.state.batcher` when finished.

Queued disconnect/cancellation skips model work. Closing rejects queued requests
and waits for the active model call; it does not abort an active Torch/CUDA kernel.
Pending limits include active records. Queue pressure returns 503/Retry-After;
ordinary model errors fail the affected batch but leave the worker usable.
`/health` and `/metrics` contain no prompt/path/record content. Metrics are aggregate
diagnostics, not a cloud SLA, GPU benchmark, token-level continuous batching or
authentication layer. Tune waiting versus latency for the workload. The standard
library HTTP example above remains per-request rather than this dispatcher.

Dataset and preprocessing references:
[load_digits](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html),
[train-only preprocessing and leakage](https://scikit-learn.org/stable/common_pitfalls.html).
