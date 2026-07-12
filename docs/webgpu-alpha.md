# WebGPU Alpha

ARTI includes an experimental Python-first browser deployment path. Python
calls the real module, defines the complete artifact contract, and exports its
ONNX graph. `@arti-fit/web` generically validates and executes the declared
inputs and outputs with ONNX Runtime Web; it contains no ARTI mechanism rules.

This is an inference boundary. It does not change `arti.st`, support browser
training, or claim full parity with every ARTI mechanism.

```bash
pnpm add @arti-fit/web@alpha
```

## Export From Python

Install the optional export dependencies:

```bash
uv sync --extra web
```

Export an initialized module in evaluation mode:

```python
import torch
from arti.nn import LearnedPulse
from arti.web import export

layer = LearnedPulse(k=8, dim=64).eval()
x = torch.randn(1, 32, 64)
mask = torch.ones(1, 32)

export(
    layer,
    "layer-web",
    example_inputs={"x": x, "mask": mask},
)
```

The directory contains `arti-web.json`, `model.onnx`, and
`arti-web.lock.json`. The example inputs define the deployment contract. If
`q` or `mask` is exported, the Web runtime requires it; if it is omitted, the
runtime rejects it. Batch and token axes are dynamic by default, while feature
dimension and folded workspace size remain fixed.

## Run In The Browser

```ts
import { loadArti, Tensor } from "@arti-fit/web";

const layer = await loadArti("/layer-web/", { device: "auto" });
const x = new Tensor("float32", values, [batch, tokens, 64]);
const mask = new Tensor("float32", maskValues, [batch, tokens]);
const { y } = await layer.run({ x, mask });

await y.getData(); // Explicitly downloads a WebGPU result to CPU.
y.dispose();
await layer.dispose();
```

`device: "auto"` tries WebGPU and then WASM. `device: "webgpu"` never silently
falls back. Outputs stay on the GPU when WebGPU is active. Pass a second named
output map to `run(inputs, outputs)` when the caller owns a GPU buffer pool.

Self-hosted and offline applications can pass `wasmPaths` or `wasmBinary` to
`loadArti`. Applications remain responsible for serving the matching ONNX
Runtime Web control binary.

## Python-Owned Contract

- Python accepts supported CPU float32 tensor inputs and calls
  `module.forward(**example_inputs)` directly.
- Tensor, named tensor mapping, and tensor sequence outputs become named ONNX
  outputs in artifact v2.
- Input/output names, shapes, dynamic axes, module metadata, and supported
  modes are decided by Python.
- TypeScript contracts and validators are generated from
  `arti.web.contract`; CI rejects generated-file drift.
- JavaScript retains only browser concerns: fetch, SHA-256, provider choice,
  GPU buffers, execution, and disposal.

Unsupported Python module modes fail before export. Artifact v1 is rejected
with a request to re-export it from the current Python package.

## Development Checks

```bash
uv run python scripts/generate_web_fixtures.py .tmp/web-fixtures
uv run python scripts/generate_web_contract.py packages/web/src/generated/contract.ts
pnpm build:web
pnpm test:web
pnpm --filter @arti-fit/web test:browser
pnpm docs:web
```

The browser test requires a local Chrome installation and hardware WebGPU. CI
runs artifact generation, TypeScript build, and WASM parity; it does not treat
a software WebGPU adapter as hardware evidence.

## Stateful Recall Artifact

Stateful Recall is an alpha inference-time adaptation path, not browser
training. Python exports the same initialized `StatefulRecall` as two graphs:

```python
import torch
from arti.nn import StatefulRecall
from arti.web import export_stateful_recall

recall = StatefulRecall(dim=64, slots=16).eval()
export_stateful_recall(
    recall,
    "recall-web",
    example_x=torch.randn(1, 32, 64),
    example_mask=torch.ones(1, 32),
)
```

`read.onnx` reads immutable model parameters and explicit state without
changing either. `update.onnx` applies Python-defined recognition, learned
write quality, slot assignment, delta correction, and decay, and returns a new
fixed-size state. The TypeScript runtime only follows manifest entrypoints and
state bindings.

```ts
const recall = await loadArtiStateful("/recall-web/");
const read = await recall.run("read", {x, mask});
const info = await recall.commit("update", {
  trace_key: read.trace_key,
  observed: x,
  mask,
});
```

`commit()` swaps state only after every declared next-state tensor is produced,
then disposes the previous buffers. State is memory-only and non-persistent;
`snapshot()`, `restore()`, `fork()`, and `reset()` are explicit lifecycle
operations. `snapshot()` and `fork()` may transfer state through CPU memory,
while the continuous WebGPU `run`/`commit` path keeps committed state on GPU.
Applications should set `maxStateBytes` according to their device budget and
`maxArtifactBytes` according to their download and session-creation budget.
The runtime derives state bytes from declared tensor shapes, requires the
manifest budget to match, rejects non-local artifact file names, bounds file
and entrypoint fan-out, and downloads model files sequentially. Recall masks
must match `[batch, tokens]` exactly; broadcastable higher-rank masks are
rejected before latent computation.
