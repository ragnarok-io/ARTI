# Stable Release Surface

The private development line makes FederatedProgram execution the default
``ARTILayer`` runtime. Public 3.0.13a5 remains an earlier released snapshot.

## Default Layer

`ARTILayer` is a host around a FederatedProgram graph:

```python
import torch
import arti

layer = arti.ARTILayer()
x = torch.randn(2, 8, 64)
y = layer(x)
```

The unconfigured Federal shell is an exact identity. Configure a Federation
when the layer should perform Program-local Execution and Formula Fabric work:

```python
# ``program`` is a configured FederatedProgram execution region.
layer = arti.ARTILayer(program)
y, info = layer(x, return_info=True)
```

The private source declarations are `arti/layer@3` and
`arti/federal-recall@3`; configured instances persist their resolved full
contract addresses. The same default layer is exported as
`arti.ARTILayer` and `arti.torch.ARTILayer`.
`arti.nn.Layer` remains the explicit profile-based constructor for the classic
tensor pipeline; it does not define the default `ARTILayer` identity.

## Stable Mechanisms

Versioned mechanism APIs are available from `arti.mechanisms`:

```python
from arti import mechanisms

pulse = mechanisms.AdaptivePulse(...)
runtime = mechanisms.RecallRuntime(...)
topology = mechanisms.ReversibleTopology(...)
fabric = mechanisms.FormulaFabric(...)
```

Half, Fold, UnFold, Recall, execution iteration, Formula Fabric, adaptive observation,
operable tensors, K-wide branch search, Target Bank updates, and their typed runtime
contracts report lifecycle `stable` in the component registry. `ARTILayer@3` and
FederatedProgram graph construction remain `alpha`: their provenance is strict, but
their public construction contracts are still evolving.

Stable means their versioned identity and declared contract are release APIs.
It does not turn a benchmark result into a universal quality or performance
claim.

## Legacy

The former monolithic `ARTILayer@1`, `LayerRecall`, and `StatefulRecall` are
available only through `arti.legacy`:

```python
from arti import legacy
```

They are retained for explicit historical inspection and do not define the
default layer or attachment path.

## Attach To A Model

```python
model = arti.ARTI.attach(
    model,
    layer,
    layers="model.layers.*",
)

model.arti.save("model.arti.st")
fresh = arti.ARTI.load(
    fresh_model,
    "model.arti.st",
    layer=layer,
)
```

See [Unified Model Attachment](unified-attachment.md) for configuration,
training, Hub bundles, and strict graph reload behavior.

## Validation

```bash
uv run --extra dev python scripts/quality_gate.py quick
uv run --extra dev python scripts/quality_gate.py package
uv run --extra docs mkdocs build --strict
```

Mechanism-specific evidence and hardware claim boundaries remain documented in
the validation and reference sections.
