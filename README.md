# ARTI

ARTI = AI x RT.

ARTI is a programmable neural runtime for PyTorch and JAX. Compose neural
programs with Formula Fabric, explicit tensor resources, learned connections,
local iteration and forward plasticity. Adapt an existing model or build a
standalone model, then train, resume, evaluate and deploy the same asset.

The PyPI distribution is `arti-fit`; the Python import is `arti`.

## Install

```bash
pip install "arti-fit==3.3.0"
pip install "arti-fit[qwen,serve]==3.3.0"  # language recipes and optional ASGI serving
pip install "arti-fit[jax,flax]==3.3.0"    # native JAX and optional Flax integration
```

Python 3.10+ is supported. PyTorch is a core dependency; model, JAX, Flax,
documentation and serving integrations are optional extras.

For an unpublished candidate, install its reviewed local wheel with
`python -m pip install --no-index --no-deps /path/to/arti_fit-3.3.0-py3-none-any.whl`
in an environment whose declared dependencies are already provisioned. A
candidate version string does not imply availability on PyPI.

## Start An Application Task

```bash
arti task init my-task --name "my task"
cd my-task
arti task check project.json
arti task explain project.json
python workflow.py prepare
python workflow.py train
python workflow.py package
python workflow.py predict
python workflow.py evaluate
python workflow.py runtime
python workflow.py export
```

The generated project owns its model, native objective, preprocessing, file
producer, scorers, runtime entrypoint and result writer. Project checks and
status/explain inspect existing declarations and evidence without running
application callbacks. [The complete installed-user tutorial](docs/guides/installed-task-workflow.md)
is also included as `TUTORIAL.md` in the project. It demonstrates interrupted
recovery, cache reuse, committed file outputs and an offline archive consumer.

## One Training Workflow

Native models, attached programs and pretrained workflows use one session:

```python
import arti

# model is a configured native model or a host with trained ARTI parameters.
session = arti.trainer(model, objective=task_objective)
print(session.plan(steps=1000))
assert session.check(example_batch)["ok"]
result = session.fit(
    train_loader,
    steps=1000,
    run=arti.ARTITrainingRunConfig(output_dir="outputs/my-task"),
)
metrics = session.evaluate(validation_loader)
```

The local Torch profile saves the first update, periodic checkpoints and a final
checkpoint. Continuation retains optimizer, scheduler, supported RNG/scaler state
and the declared data cursor. Evaluation preserves model and ARTI state. The
host's task loss is the default; tensor alignment is an explicit alternative.

Use the [training guide](docs/guides/training-workflow.md) for native, attachment
and pretrained entrances, data-resume guarantees and export.

## Real Task Recipes

The same lifecycle is available from JSON/TOML:

```bash
arti recipe plan recipe.toml --output plan.json
arti recipe check recipe.toml
arti recipe train recipe.toml --run-dir outputs/first
arti recipe train recipe.toml --resume-from-checkpoint outputs/first/checkpoints/step-00000200/model.arti.st --steps 40 --run-dir outputs/continued
arti recipe evaluate recipe.toml --model outputs/continued/model --split test --output outputs/test.json
arti recipe predict outputs/continued/model --data requests.jsonl --output outputs/predictions.jsonl
arti recipe serve outputs/continued/model --port 8080
```

- [Classification](docs/guides/task-recipes.md): explicit train/validation/test
  records, train-only preprocessing, cross-entropy, metrics and portable batched
  prediction without the original dataset.
- [Qwen SFT](docs/guides/qwen-sft-workflow.md): complete conversation input,
  final-assistant/EOS supervision, target-token-weighted accumulation, pinned
  base revision, checkpoint continuation and tokenizer-bound deployment.
- Optional FastAPI/Uvicorn serving loads the asset once per worker and exposes
  `/predict`, `/chat`, `/health` and OpenAPI. It is a local ASGI integration,
  not an authenticated gateway or a vLLM/TensorRT-LLM export.

Runnable repository examples:

```bash
uv sync --extra dev --extra qwen --extra serve
python examples/classification_workflow.py --output .tmp/digits-workflow
python examples/qwen_sft_workflow.py --prepare-only --output .tmp/qwen-sft
```

Qwen planning loads native configuration/tokenization and a meta-device model,
not base weights or CUDA allocations. `check` deliberately performs a real
gradient preflight. The recipe's fixed Formula adaptation profile is one usable
composition, not a claim that all ARTI features or architecture search are enabled.

## Compose Neural Programs

`ARTILayer` hosts a configured `FederatedProgram` or `ProgramGraph`. An empty
layer is an exact identity, not an untrained adapter that pretends to know a task.

```python
import torch
import arti

layer = arti.ARTILayer()
x = torch.randn(2, 8, 64)
assert torch.equal(layer(x), x)

# Configure the graph or federation before attaching a useful program.
layer = arti.ARTILayer(graph=graph, graph_program_id="forward",
                      graph_input_resource_id="input",
                      graph_output_resource_id="output")
model = arti.ARTI.attach(model, layer, layers="model.layers.*")
session = model.arti.trainer(objective="model_loss")
```

The [developer guide](docs/guides/new-developer.md) and
[Formula Fabric guide](docs/guides/formula-fabric.md) cover typed operations,
named ports, routes, parallel stages, joins, loops, tensor resources and
compiler boundaries. [Learning boundaries](docs/guides/learning-boundaries.md)
allow independently owned optimizer domains and region-specific objectives.

Reusable operations live in `arti.mechanisms`. Retired layered Recall APIs live
only in `arti.legacy`. Component contracts use content-addressed provenance;
package versions and component identities are independent.

## Native JAX

`arti.jax` binds the same Formula/ProgramGraph declarations to native array
execution, with explicit parameter and forward-state PyTrees. Use `jit`, `grad`,
`vmap` and fixed-horizon `scan`; integrate custom native modules or Flax without
running Torch operations in a compiled JAX invocation.

```bash
uv sync --extra dev --extra jax --extra flax
python examples/jax_native_training.py
```

Read the [native JAX guide](docs/guides/jax-backend.md) for supported atoms,
resource state, native overrides and limitations. This is not full parity with
every Torch dynamic search, custom VJP or optimizer-domain engine.

## Assets And Support

ARTI uses SafeTensors and validated component/graph metadata instead of Python
pickle for its model assets. Frozen host weights remain separate from adapter
exports. Custom Python layers still require their reconstruction code.

[Stability](STABILITY.md) distinguishes supported workflow contracts from
experimental component surfaces. [Release checks](docs/guides/release.md)
cover software correctness, installation, documentation and privacy. They do
not imply general task superiority or universal acceleration.

```bash
uv run --extra dev python scripts/quality_gate.py quick --fail-fast
uv run --extra dev --extra qwen --extra serve python scripts/quality_gate.py recipes --fail-fast
```

MIT licensed. Designed and maintained by [Thiocy](https://github.com/Thiocy).
