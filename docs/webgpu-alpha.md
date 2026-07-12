# WebGPU Alpha

ARTI includes an experimental browser deployment path for deterministic
`Half`, soft `Fold`, and soft-fold `LearnedPulse` modules. Python exports an
ONNX artifact; `@arti-fit/web` validates and executes it with ONNX Runtime Web.

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
const y = await layer.forward(x, { mask });

await y.getData(); // Explicitly downloads a WebGPU result to CPU.
y.dispose();
await layer.dispose();
```

`device: "auto"` tries WebGPU and then WASM. `device: "webgpu"` never silently
falls back. Outputs stay on the GPU when WebGPU is active. Use `forwardInto`
with an ORT tensor backed by a preallocated GPU buffer when the caller owns a
buffer pool.

Self-hosted and offline applications can pass `wasmPaths` or `wasmBinary` to
`loadArti`. Applications remain responsible for serving the matching ONNX
Runtime Web control binary.

## Supported Surface

- float32 `[B, N, D]` input
- deterministic `Half`
- `Fold(mode="soft", topk=None, dropout=0)`
- soft-fold `LearnedPulse` without top-k or dropout
- optional exported `q` and `mask` contracts
- dynamic batch and token dimensions
- artifact size and SHA-256 verification

Stochastic Half, attention/top-k Fold, lazy feature dimensions, Recall, and
training fail during export with an explicit unsupported-mode error.

## Development Checks

```bash
uv run python scripts/generate_web_fixtures.py .tmp/web-fixtures
pnpm build:web
pnpm test:web
pnpm --filter @arti-fit/web test:browser
pnpm docs:web
```

The browser test requires a local Chrome installation and hardware WebGPU. CI
runs artifact generation, TypeScript build, and WASM parity; it does not treat
a software WebGPU adapter as hardware evidence.
