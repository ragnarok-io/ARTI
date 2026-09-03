# ARTI

**AI x RT: composable latent tensor dynamics for PyTorch.**

ARTI is a domain-independent neural-network library for transforming hidden
tensors at runtime. It provides versioned components for activation, compact
workspaces, Recall, Formula execution, reversible topology, persistent tensor
operations, and model attachment.

```text
tensor input -> ARTI layer or mechanism -> tensor output
```

ARTI does not prescribe a tokenizer, task head, dataset, model family, or
training loop. Applications decide what a tensor means; ARTI supplies reusable
ways to observe, route, transform, remember, and compose it.

## Install

The PyPI distribution is `arti-fit`; the Python package is `arti`:

```bash
uv add "arti-fit==3.0.13a1"
```

ARTI requires Python 3.10 or newer and PyTorch 2.2 or newer. The consuming
project chooses the appropriate CPU or CUDA build of PyTorch.

Optional integrations are installed only when needed:

```bash
uv add "arti-fit[jax]==3.0.13a1"
uv add "arti-fit[qwen]==3.0.13a1"
uv add "arti-fit[sd]==3.0.13a1"
uv add "arti-fit[web]==3.0.13a1"
```

The browser runtime remains a separate alpha package:

```bash
pnpm add @arti-fit/web@alpha
```

## Alpha 3.0.13a1

This prerelease adds a typed NeuralPlasticity effect lane to Formula Fabric.
An effect leaves its local tensor value unchanged while producing a transition
for the owning execution site's explicit network state. Ordinary Formula nodes
can consume the resulting state later in the same program.

`FormulaProgramQueryV2` makes ordinary Formula nodes and NeuralPlasticity nodes
part of one bounded search space. It selects their number, order, placement,
and wiring from final task loss; callers declare local typed candidates rather
than a complete prebuilt self-modifying chain. Program steps remain separate
from Bank-local Refine steps. These new component identities have lifecycle
`alpha` and do not expand the stable compatibility surface.

The stable 3.0.12 base retains K-wide Recall traversal, the typed Formula atom
basis, shape-polymorphic Federal Bank execution, and the default
AdaptivePulse-backed `ARTILayer` described below.

`arti.ARTILayer` is now `arti/layer@2`: a tensor-in/tensor-out host for one
composable `AdaptivePulse` graph. An empty graph is an exact identity, so a
layer can be inserted before an application chooses its mechanisms.

```python
import torch
import arti

x = torch.randn(2, 16, 64)
layer = arti.ARTILayer()
y = layer(x)

assert torch.equal(x, y)
assert arti.component_ref(layer) == "arti/layer@2"
```

Build a layer from ordinary ARTI modules:

```python
pulse = arti.mechanisms.AdaptivePulse(
    half=arti.Half(stochastic=False, learnable=True),
)
layer = arti.ARTILayer(pulse)
y, info = layer(x, return_info=True)
```

The public namespaces have explicit roles:

- `arti` and `arti.torch`: stable high-level PyTorch APIs.
- `arti.nn`: practical standalone tensor layers and explicit layer profiles.
- `arti.mechanisms`: stable, versioned composition/runtime mechanisms.
- `arti.legacy`: the retired monolithic `ARTILayer@1`, `LayerRecall`, and
  `StatefulRecall`, retained for historical artifact inspection.
- `arti.experimental`: integrations that are still experimental; currently the
  Python-first Web exporter.

`arti.alpha` remains a compatibility name for `arti.mechanisms`; new code should
use the stable namespace.

## Attach To A Model

Attach an AdaptivePulse-backed layer to selected tensor boundaries without
changing the host model class:

```python
import arti

pulse = arti.mechanisms.AdaptivePulse(
    half=arti.Half(stochastic=False, learnable=True),
)
layer = arti.ARTILayer(pulse)

model = arti.ARTI.attach(
    model,
    layer,
    layers="model.layers.*",
)

print(model.arti.summary())
model.arti.save("assistant.arti.st")
```

Reload into a fresh host or remove the attached layers:

```python
restored = arti.ARTI.load(fresh_model, "assistant.arti.st", layer=layer)
restored = restored.arti.detach()
```

Attachment preserves the host model type and arbitrary tensor trees. Boundary
values are packed for ARTI execution and restored to their original rank,
layout, dtype, and position. See [Unified Attachment](docs/unified-attachment.md).

## Mechanism Map

| Need | API |
| --- | --- |
| Same-shape survival pressure | `arti.Half` |
| Soft learned workspace compaction | `arti.nn.Fold` |
| Trainable learned expansion | `arti.nn.UnFold` |
| Reversible active/folded topology | `arti.mechanisms.Fold`, `arti.mechanisms.UnFold` |
| Compact learned pulse workspace | `arti.Pulse` |
| Composable bounded execution graph | `arti.mechanisms.AdaptivePulse` |
| K-wide Recall with hard winner | `arti.nn.Recall` |
| Iterative hidden-state refinement | `arti.RecallRefiner` |
| Typed Formula programs | `arti.mechanisms.FormulaFabricV2` |
| Bounded typed program selection (alpha) | `arti.mechanisms.FormulaProgramQuery` |
| Execution-site network-state effects (alpha) | `arti.mechanisms.FormulaFabricV3` / `V4` / `V5` |
| Self-effect topology search (alpha) | `arti.mechanisms.FormulaProgramQueryV2` |
| Addressable forward Bank updates | `arti.mechanisms.TargetBankUpdater` |
| Shape-autonomous Bank hierarchy | `arti.mechanisms.FederalRecall` |
| Persistent auxiliary tensor editing | `arti.mechanisms.TensorOperationLoop` |

All mechanisms remain independently usable. Coordinates, masks, visibility,
Observation, Recall, Formula, Fold, and tensor operations are not mandatory
parts of one monolithic architecture.

## Recall

New `arti.nn.Recall` modules use `arti/recall@4`. One query can preserve K
candidate routes, refine them independently, and forward exactly one hard
winner. Weighted merging is opt-in.

```python
recall = arti.nn.Recall(dim=768, slots=64)  # portable default K=8
next_state = recall(x)

wide = arti.nn.Recall(dim=768, slots=64, breadth=16, group_topk=16)
next_state, branches = wide(x, return_branches=True)
next_state = wide(x, active_k=4)
```

Eight is the portable default; 8-32 is a useful starting range when Bank
capacity and runtime budget allow. `breadth=1` executes one candidate.

Runtime refine depth is explicit and may be adjusted without rewriting an
artifact:

```python
arti.set_recall_refine_steps(model, 10, min_steps=2, tolerance=0.003)
arti.set_recall_refine_schedule(
    model,
    {
        "model.layers.0": 2,
        "model.layers.1": 6,
        "model.layers.2": 12,
    },
)
```

## Formula And Federal Banks

Formula Fabric executes bounded, typed tensor programs with declared operands,
shapes, route sources, and output contracts. Built-in atoms include ordinary
tensor transforms as well as Fabric-native Fold and UnFold operations.

The stable atom basis can compose Transformer-like subgraphs without hiding an
opaque Attention, MLP, or Transformer primitive. `FormulaProgramQuery@1` is an
alpha controller for selecting a bounded, shape-valid SSA path from final task
loss; it chooses one hard candidate per step rather than averaging outputs.

`FormulaProgramQuery@2` extends that search to execution-site network-state
effects. Its Query observes SSA tensor values, not the implicit state it may
modify. The selected program returns successor state and revisions explicitly;
the owning Bank runtime decides whether that successor becomes persistent.

Federal Banks can carry their own sealed Query, Formula program, local Refine
policy, and terminal ABI. A Bank may change its tensor shape internally and
re-query after every local step; it leaves the Bank only after producing a
value accepted by the shared terminal contract. This separates federation
depth from Bank-local computation depth.

See [Formula Fabric](docs/formula-fabric.md) and
[Federal Contracts](docs/federal-contracts.md).

## Operable Tensors

The tensor-operation API supplies a default-backed, hot-swappable auxiliary
tensor port. Reader Refine and tensor operations consume the same call-boundary
snapshot in parallel. Operations can address ranges or sparse index maps of an
arbitrary-rank logical tensor; proposals become visible after the caller
advances the port for a later invocation.

See [Operable Tensor Port](docs/operable-tensor-port.md).

## Artifacts

`arti.save()` and `arti.load()` store model state in SafeTensors with separate
architecture metadata and SHA-256 locks. Component identities and artifact
schemas are versioned independently from the package version.

Recall Bank artifacts record the host, reader, Formula, optional Updater, Bank
layout, dtype, shapes, provenance, and tensor hashes. Named compatible Banks
can be composed without silently changing their source identity.

See [Recall Artifacts](docs/recall-artifacts.md) and
[Component Provenance](docs/component-provenance.md).

## Documentation

- [Stable Release Surface](docs/stable-release-surface.md)
- [Unified Attachment](docs/unified-attachment.md)
- [Adaptive Observation](docs/adaptive-observation.md)
- [Reversible Topology](docs/reversible-topology.md)
- [Formula Fabric](docs/formula-fabric.md)
- [Federal Contracts](docs/federal-contracts.md)
- [Batched Refine](docs/batched-refine.md)
- [Flattened Refine Training](docs/flattened-refine-training.md)
- [Refine Exit](docs/refine-exit.md)
- [Operable Tensor Port](docs/operable-tensor-port.md)
- [WebGPU Alpha](docs/webgpu-alpha.md)

## Development

```bash
uv sync --locked --extra dev
uv run --extra dev pytest
uv build
uv run --extra dev python scripts/check_package.py
```

Public CI runs Python 3.10, 3.11, and 3.12, the optional JAX contract suite,
the Python-owned Web artifact generator, and the TypeScript browser runtime.

## Scope

Stable means the documented component identities, tensor contracts, artifact
formats, and composition semantics are reviewed and release-gated. It does not
claim universal downstream superiority, a production service SLA, full JAX
parity, or WebGPU training.

ARTI is licensed under the [MIT License](LICENSE). Citation metadata is in
[`CITATION.cff`](CITATION.cff).

## 中文简介

ARTI 是一个领域无关、PyTorch-first 的可组合张量动力学基础库。3.0.13a1
在 3.0.12 稳定表面之上加入 alpha 级 NeuralPlasticity Formula effect 与
可搜索的自修改程序拓扑：网络从局部候选中学习自修改节点的数量、顺序、位置
和连接，而不是由调用者预先拼好完整链。应用负责张量的业务语义，ARTI 负责
张量的观察、路由、变换、记忆与组合。
