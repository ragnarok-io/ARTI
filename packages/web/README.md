# @arti-fit/web

Python-first, inference-only ARTI binding for WebGPU with a WebAssembly
fallback. Python defines and exports the graph contract; this package is a
generic verifier and executor with no Half, Fold, Pulse, `q`, or `mask` logic.

```bash
pnpm add @arti-fit/web@alpha
```

```ts
import { loadArti, Tensor } from "@arti-fit/web";

const layer = await loadArti("/layer-web/", { device: "auto" });
const x = new Tensor("float32", values, [batch, tokens, dim]);
const { y } = await layer.run({ x });
await layer.dispose();
```

For ordinary application code, `predict()` accepts CPU tensor values and owns
the temporary ONNX Runtime tensors it creates:

```ts
import {loadArti} from "@arti-fit/web";

const layer = await loadArti("/layer-web/", {
  device: "auto",
  signal: abortController.signal,
  onProgress: ({stage, loadedBytes}) => console.log(stage, loadedBytes),
});
const output = await layer.predict({
  x: {data: values, dims: [batch, tokens, dim]},
});
console.log(output.y.data);
await layer.dispose();
```

`run()` remains the advanced Tensor API for GPU-resident outputs and
caller-owned preallocated buffers. Public failures use `ArtiWebError` with a
stable code, stage, artifact URL, device, and tensor-contract context.

Each Python v2 export also writes `artifact.ts`. Its input and output property
names come directly from the hashed manifest, so single-input artifacts expose
a typed `forward()` client and named multi-input artifacts expose typed
`run()`. JavaScript does not infer module mechanisms or duplicate Python rules.

## Stateful Recall (alpha)

Python can export a paired, explicit-state Recall artifact. Model parameters
remain read-only in the browser: `read` proposes a recalled trace and `update`
returns the next fixed-size state.

```ts
import {loadArtiStateful, Tensor} from "@arti-fit/web";

const recall = await loadArtiStateful("/recall-web/", {
  device: "auto",
  maxStateBytes: 64 * 1024 * 1024,
  maxArtifactBytes: 128 * 1024 * 1024,
});
const first = await recall.run("read", {x, mask});
const diagnostics = await recall.commit("update", {
  trace_key: first.trace_key,
  observed: x,
  mask,
});

const snapshot = await recall.snapshot(); // explicit GPU-to-CPU transfer
await recall.reset();
await recall.restore(snapshot);
await recall.dispose();
```

State is non-persistent by default. Continuous `run`/`commit` calls retain
committed state in GPU buffers on WebGPU. `snapshot()` and `fork()` are explicit
portability operations and may download state. The runtime enforces the
artifact's fixed shapes and the caller's `maxStateBytes` limit. Artifact file
names remain inside the artifact directory, model fan-out and aggregate bytes
are bounded, and `maxArtifactBytes` lets applications apply a tighter download
budget than the built-in 512 MiB ceiling.

Artifact v2 accepts Python-supported float32 tensor-in/tensor-out modules.
Browser training is not part of this release. Artifact v1 must be re-exported
with the current Python package.

## Module Worker

A minimal native module Worker example starts at
[`examples/worker/main.ts`](./examples/worker/main.ts). The main thread sends named float32
inputs as `{data: ArrayBuffer, dims}` and includes each buffer in the
`postMessage` transfer list. The Worker constructs `Tensor` values, loads an
artifact with `loadArti`, calls `run`, downloads every result to CPU, and
transfers the result buffers back.

The protocol supports `load`, `run`, `dispose`, and `cancel`, with structured
error responses. Cancellation is cooperative: ONNX Runtime inference already
in progress may finish inside the Worker, but its late result is suppressed and
disposed. Only CPU `ArrayBuffer` values cross the thread boundary. A WebGPU
`Tensor` and its `GPUBuffer` remain Worker-owned and must never be posted to the
main thread.

Create the Worker through a bundler that supports `new URL(..., import.meta.url)`:

```ts
const worker = new Worker(new URL('./arti.worker.ts', import.meta.url), {
  type: 'module',
});
worker.postMessage(runRequest, inputTransferList);
```

Before creating a release tag, run the bounded local browser gate on a
WebGPU-capable machine:

```bash
pnpm gate:web
```

The gate regenerates Python-owned contracts and fixtures, builds the package,
runs unit and WASM parity tests, executes the real module Worker and stateful
Recall paths under Playwright WebGPU, and regenerates TypeDoc.
