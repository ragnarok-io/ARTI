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

Version 2.0.0 is a **Stable Candidate**. The supported 2.x surface is frozen
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

## What Is New In 2.0

ARTI 2.0 makes the current Recall architecture the public default. Recall now
uses a fixed query basis, host-dimensional Bank values, versioned Formula
contracts, explicit per-Bank composition, and bounded iterative refinement.
The package also exposes reusable Recall artifact, expert, policy, workspace,
and value-transition primitives without coupling them to a training loop.

The experimental 1.x `RecallTTTSession` API has been removed. It mixed
optimization policy with the tensor layer and is not part of the 2.x
replacement. Applications should compose `arti.nn.Recall`, Formula contracts,
and explicit artifact/state APIs instead.

The `arti.st` file format remains version 1. ARTI 2.0 can strictly load the
value Bank and query basis from compatible 1.x Recall states, while discarding
the obsolete learned routing key Bank and initializing new state metadata.
Because routing semantics changed, migrated Recall artifacts must be validated
on their intended workload; load compatibility does not claim identical 1.x
numerical behavior.

## What Was New In 1.9

ARTI 1.9 adds runtime control over the exact number of Recall refinement steps
without changing or rewriting adapter weights:

```python
arti.set_recall_refine_steps(model, 6)
arti.set_recall_refine_steps(model, 0)  # exact Recall bypass

arti.set_recall_refine_schedule(model, [1, 1, 3, 3, 6, 6])
arti.set_recall_refine_schedule(
    model,
    {
        "model.layers.0": 1,
        "model.layers.1": 3,
        "model.layers.2": 6,
    },
)
```

Sequence schedules follow `model.named_modules()` registration order. Named
schedules must cover every attached adapter exactly and are recommended for
persistent configuration. ARTI validates the complete schedule before changing
any layer, so an invalid depth cannot leave a partially updated model.

The controls only change runtime refinement depth. They do not change adapter
parameters, artifact format, optimizer state, or the separately versioned Web
runtime. A positive depth cannot enable an adapter that was initialized without
a Recall field.

## What Was New In 1.8

ARTI 1.8 introduces `arti.nn.Recall`, a standalone tensor-in/tensor-out layer
with an extensible Formula API:

```text
current state + routed Bank factors -> Formula -> next state
```

The Bank owns trainable tensors and routing. The Formula only defines how the
current state and a fixed, named set of factors produce the next state. This
separation lets applications change Recall mathematics without rebuilding
routing, serialization, masking, or iterative execution.

The release includes:

- versioned `delta-v1`, `affine-v1`, and `state-v1` formulas;
- local custom formulas implemented as ordinary `torch.nn.Module` objects;
- explicit process-local registration for trusted application formulas;
- stable factor ordering and passive manifest metadata;
- masked `[B, D]` and `[B, N, D]` execution, optional iterative steps, and
  diagnostics.

Recall formulas do not own optimizers, gradient policy, files, network access,
or training schedules. Third-party formula code is never imported from an
artifact. `Recall` and the Formula API are alpha surfaces in 1.8; the rest of
the supported 1.x surface retains its existing stability level.

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

## Use Recall As A Layer

`Recall` can be inserted anywhere a shape-preserving PyTorch layer is useful:

```python
import arti
import torch

recall = arti.nn.Recall(
    dim=64,
    slots=32,
    formula="affine-v1",
    steps=2,
)

h = torch.randn(2, 32, 64)
mask = torch.ones(2, 32, dtype=torch.bool)

h, info = recall(h, mask=mask, return_info=True)

assert h.shape == (2, 32, 64)
print(info["recall_steps_executed"])
```

`Recall` routes trainable Bank factors and applies a versioned formula to the
current state. Its deterministic default uses `Half` on each proposed update.
Built-in formulas are `delta-v1`, `affine-v1`, and `state-v1`; legacy names
`single`, `product`, and `state` remain accepted.

| Formula | Bank factors | Minimum slot multiple |
| --- | --- | --- |
| `delta-v1` | `content` | 1 |
| `affine-v1` | `scale`, `shift` | 2 |
| `state-v1` | coarse/fine content, modulation, direction, opacity | 17 |

`slots` is the total Bank slot count and must be divisible by the selected
formula's factor count. `steps`, `min_steps`, and `tolerance` control bounded
iterative execution. Set `activation="none"` when a Recall application should
not use the default `Half` survival activation.

### Define A Local Formula

Applications can pass a trusted local `torch.nn.Module`. A custom Formula
receives one state vector `[D]` and its ordered factors `[F, D]`, then returns
the complete next state `[D]`:

```python
import torch
import arti


class SignedGate(torch.nn.Module):
    factor_names = ("content", "gate")

    def forward(
        self,
        state: torch.Tensor,
        factors: torch.Tensor,
    ) -> torch.Tensor:
        content, gate = factors.unbind(dim=0)
        return state + torch.tanh(gate) * content


recall = arti.nn.Recall(
    dim=64,
    slots=32,
    formula=SignedGate(),
)
output = recall(torch.randn(2, 16, 64))
```

Custom formulas run independently for every latent vector, so Formula-side
reductions cannot couple batch items or tokens. Their parameters participate in
normal autograd and `state_dict()` handling. For reusable process-local names,
register a trusted factory explicitly with `arti.register_formula(...)`.

See [Custom Recall formulas](docs/custom-recall-formulas.md) for the complete
contract, initialization rules, validation checklist, registration model, and
portability boundaries.

`Half`, `Fold`, and `Recall` remain independently usable tensor layers.

## Expand And Rearrange With UnFold

`UnFold` exposes values queried from an input tensor and learns a hard,
sample-conditioned layout while preserving every original input instance:

```python
import arti
import torch

x = torch.randn(4, 16, 64)
unfold = arti.nn.UnFold(dim=64, exposed=8)
y, exposed_mask = unfold(x, return_exposed_mask=True)

assert y.shape == (4, 24, 64)
assert exposed_mask.shape == (4, 24)
```

Original values may move, but they are not averaged, interpolated, projected,
or discarded. The queried region and layout remain trainable and support masks,
optional guide tensors, autograd, CUDA, and `arti.st` serialization. `UnFold`
is unrelated to `torch.nn.Unfold`, which extracts image patches. See the
[UnFold guide](docs/unfold.md).

One UnFold capacity can serve different runtime workspace sizes by passing
`target_length`. Only the required prefix of exposed query parameters is active
for that call.

## Fuse Compact Workspaces With FusionPulse

`FusionPulse` is an alpha layer for combining several already compact Pulse
workspaces. It learns feature-wise salience in their joint context, applies
`Half`, and lets one shared `UnFold` query a fixed-size fused workspace:

```python
left = arti.nn.Pulse(k=8, dim=64)(left_fragments)
right = arti.nn.Pulse(k=8, dim=64)(right_fragments)

fusion = arti.nn.FusionPulse(k=8, dim=64)
z = fusion.concat(left, right)

assert z.shape == (left.shape[0], 8, 64)
```

Inputs may have different slot counts and the number of sources may change
between calls. For balanced consolidation during training, request diagnostics
and add `info["structural_loss"]` to the task loss. See the
[FusionPulse guide](docs/fusion-pulse.md).

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

ARTI 2.x reads compatible format-version 1 artifacts produced by the pre-public
0.x and public 1.x lines. Legacy `.pt` migration uses PyTorch's restricted
tensor-only loader:

```python
arti.migrate_pt("legacy-state.pt", "layer.arti.st")
```

Artifact hashes detect modification relative to their lock files; they are not
publisher signatures. Obtain models and weights from trusted sources.

## Public Modules

- `arti.nn`: `Layer`, `Half`, `Fold`, `UnFold`, `Pulse`, alpha `Recall`,
  alpha `FusionPulse`, `RecallRefiner`, and visual workspace modules.
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

The binding also provides a CPU-friendly `predict()` path, contract-aware
tensor factories, structured errors, cancellable loading, Python-generated
artifact-specific TypeScript clients, and a native module Worker example.
The low-level `run()` API remains available for GPU-resident and preallocated
tensor workflows.

Inspectable exports flatten tensors from the module's real
`forward(..., return_info=True)` result into Python-declared ONNX outputs.
`module.inspect(...)` selectively retains and downloads those outputs while
reporting device and timing metadata through an explicitly disposable result.
JavaScript treats workspace, diagnostic, mask, and index labels as contract
metadata; it does not implement their ARTI semantics.

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
