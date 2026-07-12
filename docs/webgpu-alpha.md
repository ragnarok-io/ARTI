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
