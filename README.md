# ARTI

**A PyTorch runtime for composable tensor programs.**

ARTI lets a model work with tensors through explicit, versioned components:
programs, retrieval, Formula Fabric operations, resources, and bounded
execution. It is useful when a model needs more than a fixed sequence of
layers, while still keeping tensor contracts, component identity, and artifacts
inspectable.

ARTI does not prescribe a tokenizer, model family, task head, or training loop.
Applications define the meaning of their tensors; ARTI provides the program
runtime around them.

## Install

The PyPI distribution is `arti-fit`; the Python package is `arti`.

```bash
uv add "arti-fit==3.1.0a2"
```

ARTI requires Python 3.10+ and PyTorch 2.2+. Install the PyTorch build
appropriate for your CPU or CUDA environment first.

Optional integrations are opt-in:

```bash
uv add "arti-fit[qwen]==3.1.0a2"
uv add "arti-fit[sd]==3.1.0a2"
uv add "arti-fit[web]==3.1.0a2"
```

## Start With A Program Host

`ARTILayer` is the tensor-in/tensor-out host boundary. With no configured
program it is an exact identity, making it safe to attach before selecting a
program graph.

```python
import torch
import arti

x = torch.randn(2, 16, 64)
layer = arti.ARTILayer()

assert torch.equal(layer(x), x)
assert arti.component_ref(layer) == "arti/layer@3"
```

For configured execution, build a `FederatedProgram` or a `ProgramGraph` from
`arti.mechanisms`, then provide it to `ARTILayer`. The program owns its local
iteration, Formula Fabric operations, resource bindings, and traversal; the
host layer preserves the ordinary tensor boundary.

## Core Concepts

| Concept | Role |
| --- | --- |
| `ARTILayer` | Host boundary for a federated program or resource graph. |
| `FederatedProgram` | A composable program with sealed Query assets, local iteration, and a terminal tensor ABI. |
| `FormulaFabricV2` | Typed, bounded tensor operations and explicit operands. |
| `ProgramGraph` | Program nodes, resource nodes, connections, and executable graph state. |
| `TensorResource` | A logical tensor resource with an explicit lifecycle, binding, and views. |
| `Retrieve` / `RecallExecutor` | Retrieval and bounded execution helpers for standalone use. |
| `ExecutionPolicy` | Limits and traces execution without changing an artifact. |
| `Half`, `Fold`, `UnFold`, `Pulse` | Independently usable tensor layers and workspace operations. |

These components are optional. A program can use only the pieces its tensor
contract needs.

## Attach To A Host Model

ARTI can attach a configured layer at selected tensor boundaries without
rewriting the host model class.

```python
import arti

layer = arti.ARTILayer()
model = arti.ARTI.attach(model, layer, layers="model.layers.*")

model.arti.save("adapter.arti.st")
restored = arti.ARTI.load(fresh_model, "adapter.arti.st", layer=layer)
```

Attachment preserves the host model type and restores tensor-tree structure,
dtype, layout, and position at the boundary. See the
[Unified Attachment guide](docs/unified-attachment.md).

## Artifacts And Component Identity

`arti.save()` and `arti.load()` use SafeTensors alongside architecture metadata
and SHA-256 locks. Component identities, component contracts, and package
versions are separate. This lets a saved program state declare exactly which
components and tensor contracts it requires.

Named compatible Banks and program assets can be composed without silently
losing source identity. Persistent resources are explicit program state rather
than hidden Python side effects.

## Status

ARTI 3.1.0a2 is an alpha release. `ARTILayer`, component identity, artifact
contracts, Formula Fabric, program graphs, Federal execution, and resource
operations are available for structured experimentation and integration.

Public API is not yet frozen for the alpha program/runtime surfaces. Alpha does
not claim broad downstream superiority, a production service SLA, full JAX
parity, or universal acceleration. Dynamic execution only becomes faster when
the selected program, tensor shapes, and backend can be compiled efficiently.

The `arti.legacy` namespace contains retired compatibility readers for historical
artifacts. New code should use `arti`, `arti.nn`, and `arti.mechanisms`.

## Documentation

- [Stable Release Surface](docs/stable-release-surface.md)
- [Unified Attachment](docs/unified-attachment.md)
- [Formula Fabric](docs/formula-fabric.md)
- [Federal Contracts](docs/federal-contracts.md)
- [Program Roles And Resources](docs/reference/program-roles.md)
- [Operable Tensor Port](docs/operable-tensor-port.md)
- [Component Provenance](docs/component-provenance.md)
- [Public API Reference](docs/api/public.md)

## Development

```bash
uv sync --locked --extra dev
uv run --extra dev pytest
uv build
```

ARTI is released under the [MIT License](LICENSE). Citation metadata is in
[`CITATION.cff`](CITATION.cff).
