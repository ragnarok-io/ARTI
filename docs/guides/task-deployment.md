# Portable Task Delivery

The same deployment entry points support built-in classification, developer
[task recipes](custom-task-workflows.md) and native language assets. A native
task package contains its validated `arti.st` inference files, including graph
manifests/tensors, and only explicitly declared inference resources. It does
not copy the training directory, logs, checkpoints, environment or caller code.

## Inspect And Package

```python
import arti

report = arti.inspect_recipe_deployment("outputs/run/model")
package = arti.package_recipe_deployment(
    "outputs/run/model", "outputs/delivery",
    requirements=["my-task-package==1.2.0"],
)
```

```bash
arti recipe inspect-deployment outputs/run/model
arti recipe package-deployment outputs/run/model --output outputs/delivery
```

Inspection checks metadata, task/model identity and file checksums without
constructing a model, importing task code or decoding tensors. Requirements are
Python dependency declarations, not installations or proofs of compatibility.
`files_available` is deliberately separate from `runtime_reload_verified`,
`code_dependencies_verified` and `task_quality_evaluated`; inspection does not
set the latter three to true.

Only inference exports may be packaged. Use the selected model from `fit()` or
`export()`, not a resume checkpoint. Weight bytes, original task contract and
selection metadata are retained, not rewritten. The output directory must be
fresh. Copy and publication happen in a temporary directory, then one atomic
directory rename publishes the complete bundle. A failed copy/publication does
not expose a partial final package or overwrite an existing destination.

## Named Inference Resources

A custom task can need processor files, a vocabulary, calibration arrays or
other immutable inputs outside its saved tensor state. Bind them explicitly:

```python
package = arti.package_recipe_deployment(
    "outputs/run/model", "outputs/delivery",
    resources={
        "processor": "assets/processor",    # a directory
        "vocabulary": "assets/vocabulary.json",  # one file
    },
    requirements=["my-task-package==1.2.0"],
)
```

For CLI use, `resources.json` contains the same name-to-path mapping and
`requirements.json` contains an array of Python requirement strings:

```bash
arti recipe package-deployment outputs/run/model --resources-json resources.json --requirements-json requirements.json --output outputs/delivery
```

The package uses relative resource paths, preserves nested/empty directories,
and records per-file hashes plus directory inventory. Source absolute paths
are not saved as bindings. The resource names are preserved and are not
interpreted as data/model adapters. Only generic task factories consume named
resources; the built-in classifier needs no external processor resources.

The explicit factory receives resolved paths through the existing config:

```python
def build_task(config, *, mode="train", contract=None):
    if mode == "predict":
        processor_path = config.resources["processor"]
        vocabulary_path = config.resources["vocabulary"]
        # Construct your TaskPredictor with native preprocessing and outputs.
        return make_predictor(contract, processor_path, vocabulary_path)
    return make_training_recipe(config)
```

There is no new loader callback convention. Predict mode does not require
training records or an optimizer. Code remains an explicit installed
dependency; ARTI neither bundles executable code nor auto-imports a factory
named in an artifact. Install and pin your task package independently, then
pass the matching factory reference. Do not use raw training data, private
project roots, mutable caches or output directories as inference resources.
Directory resources are complete declared inputs: adding/removing/changing
their files changes the dependency inventory. Create a new package for changes.
The package output cannot be inside a directory resource being copied.
Nested directory symlinks must be materialized first; they are not silently
copied as empty directories. File links are copied as file contents. Other
filesystem link/mount semantics are outside this regular-file/directory scope.

## Move And Consume

```python
from arti.recipes import load_recipe_predictor

predictor = load_recipe_predictor(
    "delivery-moved", task_factory="my_task_package.tasks:build_task",
)
predictions = predictor.predict_records(records)
```

For the built-in classifier, omit `task_factory`. For generic tasks it is
required and must match the saved reference/kwargs. Resources resolve relative
to the moved bundle. Artifact filenames need not be `model.st`; the deployment
manifest identifies the original filename. Task names such as `classification`
are labels, not instructions to coerce a generic task to a built-in adapter.

```bash
arti recipe predict delivery-moved --task-factory my_task_package.tasks:build_task --data requests.jsonl --output results.jsonl
arti recipe serve delivery-moved --task-factory my_task_package.tasks:build_task
arti recipe predict-run delivery-moved --task-factory my_task_package.tasks:build_task --data requests.jsonl --run-dir outputs/prediction
```

Repackaging an existing bundle inherits its named resources and requirements
unless explicitly replaced. Equivalent bytes/bindings preserve its package
identity. Loading checks missing/changed resources before invoking the factory.
`/health` reports available model/package identities separately from task identity.

Recoverable prediction pins the package identity as well as the model manifest,
so unchanged weights cannot hide changed resources. Moving an unchanged bundle
is compatible with resume; substituting a different bundle is not. A raw asset
and a newly packaged bundle have different delivery declarations: start a new
prediction run when changing that declaration. Completed result recovery can
still omit the source entirely.

Saved prediction runs remain compatible with [offline scoring](prediction-scoring.md).
Score reports preserve the package identity from the prediction plan; no model
needs to be loaded again. Independent model-dependent evaluation records the
same identity when exposed by the custom predictor.

## Scope

The native task bundle is a local, single-writer handoff, using the existing
PyTorch `arti.st` state codec. It preserves only the model state actually owned
and declared by that codec plus caller-selected resources. It cannot infer
hidden frozen host dependencies, arbitrary Python state, external services,
custom provider codecs or unregistered assets. Code byte provenance, license
eligibility and dependency installation remain the consumer's responsibility.
Checksums provide content identity, not authenticity or a code sandbox.

Packaging performs no training, conversion, merge, quantization or quality
evaluation. Verify a real fresh consumer process, original task outputs,
preprocessing, state and metrics before relying on deployment readiness. CPU
handoff tests do not establish GPU/distributed throughput or hosted availability.

Native causal-LM assets continue through the existing tokenizer/base dependency
contract and `inspect_causal_lm_deployment`/`package_causal_lm_deployment` APIs.
The unified entry points delegate to them. Use `include_base`/`base_model` for
that contract, not generic resources/requirements; see the
[native language handoff guide](qwen-deployment.md).
