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

Artifact v2 accepts Python-supported float32 tensor-in/tensor-out modules.
Browser training is not part of this release. Artifact v1 must be re-exported
with the current Python package.
