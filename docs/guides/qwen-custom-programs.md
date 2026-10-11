# Use A Custom Qwen Program

The builtin residual Formula profile is a convenience, not the limit of the task
workflow. A developer can supply an `ARTILayer` with a ProgramGraph, resources,
Fabric operations or a compatible federation through the existing layer factory
protocol. Data, native final-answer loss, optimizer/run state, validation,
continuation and export still use the same Qwen recipe.

## Declare An Installed Factory

The callable receives the discovered attachment specification and returns a
layer. It can accept explicit JSON-compatible keyword arguments:

```python
def build_layer(spec, *, width=8):
    return MyProgramLayer(dim=spec.dim, width=width)
```

Install the factory's module in the training and deployment environments, then
add this table to the usual Qwen TOML recipe:

```toml
[layer_factory]
reference = "my_project.programs:build_layer"

[layer_factory.kwargs]
width = 8
```

JSON uses the same `layer_factory.reference` and `layer_factory.kwargs` fields.
`adapter.layers`/`last_layers` still select native insertion points; the selected
factory defines their program. Builtin `adapter.rank`/`activation` do not replace
the custom factory's own options.

```bash
arti recipe plan recipe.toml
arti recipe check recipe.toml
arti recipe train recipe.toml --run-dir outputs/custom
arti recipe resume-run outputs/custom
```

The resolved recipe and task declaration preserve the reference and kwargs.
Continuation reconstructs the explicitly configured factory and restores the
existing parameter, optimizer, scheduler and data cursor state. Changes to the
declared factory/options change task identity; they are not an exact resume.

Meta planning does not execute the factory or pretend to know its parameter
count. It reports `trainable_parameters=null`. The instantiated recipe's plan
reports actual trainable ownership, and `check` tests the actual loss/gradients.
Missing import dependencies are reported before the recipe allocates a host.

## Python Entry Points

```python
import arti

factory = arti.LayerFactoryConfig(
    "my_project.programs:build_layer", {"width": 8},
)
recipe = arti.CausalLMRecipe.from_config("recipe.toml")

# Alternatively, choose a descriptor in Python for a config with no factory:
recipe = arti.CausalLMRecipe(config, layer_factory=factory)
```

The descriptor is copied into the recipe used for run recovery. Existing anonymous
`layer_factory=callable` still works, but its Python implementation is a
caller-supplied dependency, not a serialized function. Use one factory source:
the recipe declaration or an explicit callable. Do not label an unrelated
callable with a configured factory's identity.

## Export And Load Explicitly

ARTI exports operands and supported graphs with the same Hub/SafeTensors protocol.
The factory reference is dependency metadata, not executable source embedded in
weights. Inspection and deployment packaging display/preserve it without importing
the factory. The optional native base remains separate under `base_model/`.

```bash
arti recipe inspect-deployment outputs/custom/model
arti recipe package-deployment outputs/custom/model --include-base --output deployments/custom
arti recipe predict deployments/custom --layer-factory my_project.programs:build_layer --data requests.jsonl --output replies.jsonl
arti recipe evaluate deployments/custom --layer-factory my_project.programs:build_layer --data labeled.jsonl --output evaluation.json
arti recipe serve deployments/custom --layer-factory my_project.programs:build_layer --device cpu
```

An explicit matching reference reuses the saved kwargs. To provide them explicitly,
add `--factory-kwargs-json options.json` containing a JSON object. An explicit
descriptor must agree with a recorded descriptor. Anonymous historical assets
have no saved factory options, so supply their original callable or reference
plus options yourself.

```python
predictor = arti.CausalLMPredictor.from_pretrained(
    "deployments/custom",
    layer_factory="my_project.programs:build_layer",
    local_files_only=True,
)
```

The loader never imports a factory solely because an artifact names it. Without
the explicit factory it reports the dependency before loading the host. A caller
can still pass an existing Python callable. Ordinary structural/shape checks
remain the existing attachment protocol, not a new plugin registry.

Custom factories retain `caller-managed` serving state scope and per-request
dispatch by default. Do not assume row independence or session isolation for a
program with forward plasticity. The developer owns state binding and decides
whether cross-request batching is valid; this feature does not create shared
mutable session storage.

## Resource Program Example

`examples/qwen_program_factory.py` supplies `build_layer(spec, width=8)`: one
Fabric stage contracts into a named workspace and applies GELU; another consumes
that workspace and the original input to form the residual output. Operands are
trainable parameters, and the existing resource compiler supplies its tensor-only
path. It illustrates an installed factory contract, not a new primitive or a
model-quality claim. From this source checkout its reference is
`examples.qwen_program_factory:build_layer`; deployment elsewhere requires the
module to be installed in that environment.

Factories are ordinary trusted Python model code. Selecting a module can execute
its import code; this is not a sandbox or low-permission extension. ARTI does not
download, install or package that code automatically. Keep its version/environment
with the project, just as for other custom network modules.

Design reference: [Hydra's configuration-selected object construction](https://hydra.cc/docs/advanced/instantiate_objects/overview/).
Only the explicit reference/kwargs pattern is adopted; ARTI does not add a Hydra
dependency or its configuration/execution policy machinery.
