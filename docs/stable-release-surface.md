# Stable Release Surface

ARTI 3.0.12 keeps the composable mechanism runtime as the stable default.
The 3.0.13a1 prerelease adds optional alpha NeuralPlasticity effects and
self-effect topology search without changing that stable surface.

## Default Layer

`ARTILayer` is a small host around `AdaptivePulse`:

```python
import torch
import arti

layer = arti.ARTILayer()
x = torch.randn(2, 8, 64)
y = layer(x)
```

The empty Pulse graph is an exact identity. Configure only the stages the
application needs:

```python
pulse = arti.mechanisms.AdaptivePulse(
    half=arti.Half(stochastic=False, learnable=True),
)
layer = arti.ARTILayer(pulse)
y, info = layer(x, return_info=True)
```

The canonical identities are `arti/layer@2` and `arti/pulse@2`. The same default
layer is exported as `arti.ARTILayer` and `arti.torch.ARTILayer`.
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

Half, Fold, UnFold, Recall, Refine, Formula Fabric, adaptive observation,
operable tensors, Federal Bank programs, Batched Refine, Target Bank updates,
and their typed runtime contracts now report lifecycle `stable` in the
component registry.

The new Formula atoms for scalar maps, explicit broadcast, selection, lookup,
slicing, concatenation, and masked softmax are stable typed primitives.
`FormulaProgramQuery@1`, `FormulaProgramQuery@2`, `FormulaFabric@3` through
`FormulaFabric@5`, NeuralPlasticity effects, `FederalRecall@3`, and the
TensorView query family are alpha components; inspect each component lifecycle
instead of inferring it from the containing Python namespace.

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
