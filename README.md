# ARTI

**AI x RT: composable latent tensor layers for PyTorch.**

ARTI is a domain-independent neural-network library for transforming hidden
tensors at runtime. Its layers work with ordinary tensors and can optionally
use coordinates, masks, visibility, latent recall, and compact workspaces.

```text
hidden tensor -> ARTI layer or block -> transformed latent tensor
```

ARTI does not define a tokenizer, task head, data schema, or business model.
Applications remain responsible for encoding their context into tensors.

Version 1.3.0 is a **Stable Candidate**. The supported 1.x surface is frozen
for final compatibility verification, but this release does not yet carry an
LTS commitment. See [Stability](STABILITY.md) and [Security](SECURITY.md).

## Install

Add ARTI to a project with [uv](https://docs.astral.sh/uv/):

```bash
uv add arti-fit
```

ARTI requires Python 3.10 or newer and PyTorch 2.2 or newer. The consuming
project chooses the appropriate CPU or CUDA build of PyTorch.

The PyPI distribution is named `arti-fit`; the Python import remains `arti`.

Optional integrations can be installed as needed:

```bash
uv sync --extra jax
uv sync --extra qwen
uv sync --extra peft
uv sync --extra sd
uv sync --extra web
```

The alpha browser runtime is published separately:

```bash
pnpm add @arti-fit/web@alpha
```

## Use ARTI As A Layer

The smallest API behaves like a normal PyTorch layer:

```python
import arti
import torch

layer = arti.nn.Layer(dim=32)
x = torch.randn(4, 16, 32)
mask = torch.ones(4, 16, dtype=torch.bool)

out = layer(x, mask=mask)

assert out.y.shape == (4, 16, 32)
assert out.pooled.shape == (4, 32)
print(out.diagnostics.keys())
```

For `[B, D]` inputs, ARTI treats each row as a single token and restores the
original rank on output.

Capabilities are opt-in. Enable only the structure carried by the data:

```python
recall_layer = arti.nn.Layer(dim=32, profile="recall")
multisource = arti.nn.Layer(dim=32, profile="multisource", coord_dim=4)
```

## Compose Recall, Half, And Fold

ARTI mechanisms are also available as standalone modules:

```python
import arti

recall = arti.ARTILatentRecallField(hidden_dim=64, slots=8)
half = arti.nn.Half()
fold = arti.nn.Fold(k=16, dim=64)
```

The common Recall branch pattern is deliberately small:

```python
delta = recall(h, mask, recall=None)[0]
h = h + half(delta)
workspace = fold(h, mask=mask)
```

`Recall` proposes latent traces, `Half` applies feature-strength survival, and
`Fold` compacts surviving information into a fixed-size workspace. Each module
can be used independently.

## Attach To An Existing Model

ARTI can discover and attach Recall branches without changing the model class:

```python
import arti

model = arti.ARTI.attach(
    model,
    recall={
        "layers": "model.layers.*",
        "rank": 16,
        "slots": 8,
    },
)

print(model.arti.summary())
model.arti.save("assistant.recall.arti.st")
```

Attachment configuration supports explicit layer paths, per-layer dimensions,
independent Recall lines, Half switches, resource previews, and reversible
removal. Transformers, PEFT, and Diffusers are optional integration boundaries;
the core package remains PyTorch-first.

## Save And Load Weights

ARTI uses SafeTensors with JSON integrity sidecars:

```python
saved = arti.save(layer, "layer.arti.st")
loaded = arti.load("layer.arti.st", model=fresh_layer)

print(saved.weights_sha256)
print(loaded.missing_keys, loaded.unexpected_keys)
```

ARTI 1.x reads compatible format-version 1 artifacts produced by the pre-public
0.x line. Legacy `.pt` migration uses PyTorch's restricted tensor-only loader:

```python
arti.migrate_pt("legacy-state.pt", "layer.arti.st")
```

Artifact hashes detect modification relative to their lock files; they are not
publisher signatures. Obtain models and weights from trusted sources.

## Public Modules

- `arti.nn`: `Layer`, `Half`, `Fold`, `Pulse`, `RecallRefiner`, and visual workspace modules.
- `arti`: complete ARTI layers, residual blocks, reference models, attachment, serialization, and diagnostics.
- `arti.torch`: backend-explicit aliases for PyTorch applications.
- `arti.jax`: optional functional JAX subset with array-only parameter trees,
  JIT, whole-tree gradients, and batch/VMAP-consistent single-sample APIs.
- `arti.functional`: mask, visibility, pooling, coordinate-frame, and activation helpers.

Experimental and legacy APIs are identified in their docstrings and are not
frozen at the same level as the supported core surface.

ARTI remains PyTorch-first. The JAX namespace does not provide attachment,
training helpers, Recall, serialization, or full `ARTILayer` parity.

## WebGPU Alpha

`arti.web.export(...)` calls the real Python module and compiles its named
tensor inputs and outputs into a hashed artifact v2 ONNX graph. The separate
`@arti-fit/web` package is a generic executor: it contains no Half, Fold,
Pulse, Recall, `q`, or `mask` rules. It uses WebGPU and falls back to
WebAssembly when `device: "auto"` is selected. See
[WebGPU Alpha](docs/webgpu-alpha.md).

Stateful Recall can be exported as paired read/update artifact v3 graphs and
loaded with `loadArtiStateful(...)`. Model parameters remain read-only;
mutable state is explicit, fixed-size, bounded by caller budgets, and
non-persistent unless the application requests a snapshot.

## Develop

```bash
git clone https://github.com/ragnarok-io/ARTI.git
cd ARTI
uv sync --extra dev
uv run --extra dev pytest
uv build
```

The test suite covers tensor shapes, masks, gradients, serialization, malformed
artifacts, public API imports, and optional backend boundaries. Contribution
guidance is in [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation And Authorship

ARTI was initiated and designed by [Thiocy](https://github.com/Thiocy).
Citation metadata is provided in [CITATION.cff](CITATION.cff). The project also
documents [authorship](AUTHORS.md) and [AI assistance](AI_ASSISTANCE.md).

## License

[MIT](LICENSE)
