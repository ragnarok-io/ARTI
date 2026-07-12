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
recall.reset();
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
