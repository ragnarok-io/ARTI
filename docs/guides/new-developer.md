# New Developer Guide

ARTI provides neural programs, tensor resources and a usable training lifecycle.
The public package is `arti-fit`; the import is `arti`.

## Choose Your Entry Point

| Goal | Entry point |
| --- | --- |
| Train a labeled task | `arti recipe`, `ClassificationRecipe`, `CausalLMRecipe` |
| Adapt an existing model | `ARTI.attach` with a configured layer or factory |
| Train a native Torch model | `arti.trainer(model, objective=...)` |
| Build a neural execution graph | `ProgramGraph`, `FormulaProgram`, `FederatedProgram` |
| Execute native JAX | `arti.jax.bind_formula`, `bind_resource_graph`, `ARTILayer` |
| Deploy a task asset | `load_recipe_predictor`, optional `create_recipe_app` |

Start with [Classification](task-recipes.md) or [Qwen SFT](qwen-sft-workflow.md)
for complete data-to-deployment tasks. For custom models, read the
[shared training workflow](training-workflow.md).

## Default Layer

```python
import arti
import torch

layer = arti.ARTILayer()
x = torch.randn(2, 8, 64)
assert torch.equal(layer(x), x)
```

An unconfigured layer is explicit identity. Install a real `ProgramGraph` or
`FederatedProgram` before expecting trainable behavior. The host returns a Tensor;
`return_info=True` also returns execution information. Core programs may change
shape internally. Only a host attachment boundary must satisfy its declared
input/output shape contract.

## Programs And Resources

`ProgramGraph` separates named tensor resources, numerical nodes and connections.
Use explicit ports and Formula operations for parallel branches and convergence;
declare loops and stop conditions rather than relying on a Python callback to
schedule each GPU operation.

- [Formula Fabric](formula-fabric.md)
- [Federal Contracts](federal-contracts.md)
- [Cooperative Formula Graphs](cooperative-formula-graphs.md)
- [Learning Boundaries](learning-boundaries.md)
- [Native JAX](jax-backend.md)

JAX bindings use native array execution and explicit parameter/state PyTrees.
Custom Torch callables require native overrides; they are not automatically
translated. Frozen graphs and dynamic graphs have different compiler boundaries.

## Assets And Checks

Train, continue and export through the same session. The result reports its run,
checkpoint and task metrics. Keep raw task predictions; lower internal loss does
not by itself establish correct answers.

SafeTensors assets retain supported parameters, forward state and component/graph
provenance. Frozen base model weights remain separate from adapter bundles.
Custom layer assets require reconstruction code at load time.

```bash
uv sync --extra dev --extra qwen --extra serve
uv run --no-sync python scripts/quality_gate.py quick --fail-fast
uv run --no-sync python scripts/quality_gate.py recipes --fail-fast
```

See [Release Surface](stable-release-surface.md), [Component Provenance](component-provenance.md)
and [Release Checklist](release.md). Retired monolithic layer APIs belong to
`arti.legacy`; they are not the default program host.
