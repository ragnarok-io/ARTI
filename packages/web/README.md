# @arti-fit/web

Experimental, inference-only ARTI runtime for WebGPU with a WebAssembly
fallback. Artifacts are exported from Python with `arti.web.export` and are
verified before ONNX Runtime creates an inference session.

```bash
pnpm add @arti-fit/web@alpha
```

```ts
import { loadArti, Tensor } from "@arti-fit/web";

const layer = await loadArti("/layer-web/", { device: "auto" });
const x = new Tensor("float32", values, [batch, tokens, dim]);
const y = await layer.forward(x);
await layer.dispose();
```

Alpha support is limited to deterministic `Half`, soft `Fold`, and soft-fold
`LearnedPulse` artifacts using float32. Browser training and Recall are not
part of this release.
