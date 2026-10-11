# Model-Independent Task Workflows

Use `TaskRecipe` for your own PyTorch model, ARTI program or explicit
`TrainingBinding`. The task supplies data, its actual loss, evaluation and
prediction. ARTI supplies the existing training session, run records,
checkpoint selection, recovery and tensor-state export. There is no required
language model, tokenizer, fixed input shape or replacement reconstruction loss.

Use the [installed task project tutorial](installed-task-workflow.md) to generate
a runnable application and inspect its declarations/evidence with `arti task
check`, `status` and `explain`. The project entrance calls the interfaces below.

## Define The Task Once

An installed factory receives `TaskRecipeConfig` and a mode. Training returns
`TaskRecipe`; deployment returns `TaskPredictor` without constructing an
optimizer, reading the training data or creating another training loop:

```python
def build_task(config, *, mode, contract=None):
    model = build_model(config.options)
    if mode == "predict":
        return arti.TaskPredictor(
            model, contract,
            predict_fn=predict,
            evaluate_fn=evaluate_labeled_records,
        )
    data = prepare_data(config.resources)
    return arti.TaskRecipe(
        config, model,
        train_data=data.train,
        validation_data=data.validation,
        dataset_manifest=data.manifest(),
        objective=task_loss,
        objective_id="my-original-objective-v1",
        evaluate_fn=lambda model, split: evaluate_split(model, data, split),
        predict_fn=predict,
    )
```

`objective(model, batch)` returns the actual differentiable scalar task loss.
The shared engine owns backward/accumulation/optimizer steps. Supply
`training_weight(batch)` when unequal batches need normalization by examples,
tokens, pixels or another declared task unit. `evaluation_weight` similarly
controls scheduled native-loss evaluation. `evaluate_fn` owns the units and
meaning of its metrics; ARTI does not fabricate labels or reinterpret answers.

The dataset manifest includes a nonempty `sha256` identifying the actual
snapshot/revision and may include record IDs, splits, processor revisions and
schemas. The task computes/verifies that identity. The runtime cannot infer
whether an arbitrary user callback truthfully describes its data.

For native binary inputs, [Native Asset Snapshots](asset-snapshots.md) provides
portable record/file identities and recoverable batched preparation. Task code
still owns decoding, collation/sampling and the objective. An optional
`data_preparation` JSON report is saved with the run; it does not replace the
semantic dataset manifest or install another trainer.

```python
config = arti.TaskRecipeConfig(
    task="my-task",
    factory="my_package.tasks:build_task",
    options={"input_shape": [32, 32, 3], "loss_profile": "original-v1"},
    resources={"data": "/current/location/dataset.jsonl"},
    training=arti.ARTITrainingConfig(device="cpu", steps=100),
)
recipe = arti.TaskRecipe.from_config(config)
result = recipe.fit(run=arti.ARTITrainingRunConfig(output_dir="outputs/run"))
```

Use `PythonFactoryConfig(reference, kwargs)` for explicit factory keyword
arguments. JSON/TOML files use `format = "arti.task-recipe-config"`, the same
`task`, `factory`, `options`, `resources`, `training` and `run` fields.
Application task names do not select built-in adapters: the declared recipe
format does. Runtime `resources` are excluded from task identity; semantic
`options`, objective, dataset, owned parameter names and application metadata
are included. Keep paths in `resources`, not semantic options.

## Train, Recover And Deploy

```bash
arti recipe plan recipe.json
arti recipe check recipe.json
arti recipe train recipe.json --run-dir outputs/run
arti recipe resume-run outputs/run --plan-only
arti recipe resume-run outputs/run --run-dir outputs/recovered --resources-json relocated-resources.json
arti recipe evaluate outputs/recovered/model --data new-test.jsonl --task-factory my_package.tasks:build_task
arti recipe predict outputs/recovered/model --data requests.jsonl --output predictions.jsonl --task-factory my_package.tasks:build_task
arti recipe serve outputs/recovered/model --task-factory my_package.tasks:build_task
```

`relocated-resources.json` supplies named resource replacements, for example
`{"data": "/new/location/dataset.jsonl"}`. The same original task/data/loss
contract must still match. Recovery preserves the original absolute update
target, engine and complete supported optimizer/scheduler/RNG/data/validation
state. Evaluation/export failures can finish with zero new optimizer updates.
Caller-supplied validation must be supplied again on recovery; it is not silently
replaced by the task's default metrics. See [Recover A Task Run](recipe-recovery.md).

`plan`/`check` execute the explicit training factory and can construct its model
and optimizer. They perform no optimizer updates. `resume-run --plan-only` and
a successful completed no-op instead inspect metadata without importing task
code or decoding model/checkpoint tensors. A one-pass iterator needs an explicit
batch for Python `recipe.check(batch)`; normal reiterable/stateful loaders are
checked without advancing their resumable state.

```python
loaded = arti.TaskPredictor.from_pretrained(
    result["model"], task_factory="my_package.tasks:build_task",
)
predictions = loaded.predict_records(requests, batch_size=16)
metrics = loaded.evaluate_records(new_labeled_records)
```

`predict_fn(model, records, *, batch_size=...)` returns one result per input in
order. JSON records with IDs retain those IDs. Without a batch override, the
callback's own default is used. Native Tensor/structured inputs also work in
Python; their evaluation supplies `dataset_manifest={"sha256": revision}`
instead of forcing tensors through JSON or copying them to CPU for hashing.
CLI/ASGI records use JSON and task-owned decoding. Independent evaluation
records the new dataset identity and, for loaded assets, the serialized model
source identity separately from the original training task.

Predictions and evaluations preserve declared owned parameters, registered
buffers, ProgramGraph resources, model modes and supported RNG. They are
read-only batches, not a continuously mutating session. The same predictor
works with `create_recipe_app` and existing bounded dynamic batching.
For long file-based jobs, [Recoverable Task Prediction Runs](prediction-runs.md)
commit each partition and recover unfinished work without regenerating results.

## State Ownership And Support

Use `TrainingBinding(model, parameters=..., state_module=...)` when execution,
selected trainable parameters and saved tensor state differ. The state module
must contain all values/buffers/resources needed to recover this task; the
factory reconstructs any excluded immutable host from its pinned definition.
Keep the owned state compact when a large frozen host does not need cloning.
Explicit optimizers and schedulers are accepted by `TaskRecipe` and remain in
the existing session's checkpoint protocol.

This recipe uses the standard `arti.st` tensor-state codec. A binding's external
provider-specific checkpoint callbacks are not adopted. Arbitrary external
codecs, one-pass streaming recovery, sharded/distributed recovery and native JAX
training recovery are not qualified by the local Torch/Accelerate tests.
Installed factory code, external immutable host identity and arbitrary
unregistered Python state remain caller-owned. Factory descriptions are not
serialized executable Python and are not automatically imported from an asset.

## Offline Non-Language Example

```bash
python examples/custom_task_recipe.py --output outputs/diabetes --steps 30
python examples/custom_task_recipe.py --output outputs/diabetes-accelerate --steps 30 --engine accelerate
```

This example uses the bundled scikit-learn diabetes dataset, a native ARTI
resource graph, original regression labels, train-only preprocessing stored in
model buffers and scale-normalized MSE. It writes the recipe/data/run,
reloadable model, independent held-out metrics and ordered predictions.
`scikit-learn` is only an example data dependency, not a core or serving
dependency. This demonstrates a real task lifecycle, not a medically qualified
predictor or an ARTI-versus-model-quality benchmark.
