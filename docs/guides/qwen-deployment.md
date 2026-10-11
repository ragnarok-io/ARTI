# Hand Off a Qwen Deployment

Training, export and deployment are separate jobs. A slim Qwen recipe export
contains ARTI state, its task/evaluation/selection metadata and tokenizer. Its
base model remains a separate pinned dependency. This guide makes that handoff
explicit without a model registry, new trainer or mandatory quality threshold.

## Inspect Before Allocating Model Memory

```bash
arti recipe inspect-deployment outputs/model --output dependencies.json
arti recipe inspect-deployment outputs/model --base-model local-native-base
```

Python provides `arti.inspect_causal_lm_deployment(...)`. Inspection validates the
existing ARTI package/Hub references and saved tokenizer, locates a supplied local
base or an already cached pinned revision, and lists native config/SafeTensors
files and their sizes. It does not download, construct the host, create an
optimizer, decode base tensors or allocate CUDA.

`files_available` is a dependency-file result. `runtime_reload_verified` remains
false and `task_quality_evaluated` false: file availability is neither successful
inference nor answer quality. The report exposes missing dependencies and
`requires_layer_factory` separately. Inspection exits nonzero for missing files,
with a structured report rather than starting a failed GPU allocation.

Native single-file `model.safetensors` and sharded
`model.safetensors.index.json` snapshots are qualified. Legacy pickle base weights,
remote-code models and optimized non-Torch engines are outside this packaging
profile. Normal slim loading remains supported through its existing loader.

## Publish a Movable Local Package

```bash
arti recipe package-deployment outputs/model --include-base --output deployments/candidate
arti recipe package-deployment outputs/model --base-model local-native-base --include-base --output deployments/candidate
```

The first command uses a complete existing pinned Hub cache. The second uses an
explicit local native snapshot. Neither downloads. A caller-provided snapshot's
correspondence to the original model ID/revision remains caller-owned; packaging
does not prove the provenance of arbitrary weights supplied by the caller.

The fresh destination contains the same `model.arti.st`, bound graph/Hub files,
tokenizer and `deployment.json`, with original native files under `base_model/`.
The relative base reference and copied file sizes/hashes are added to the existing
deployment metadata. No new neural artifact format or component identity is
introduced. Base and ARTI weights stay separate; this is not a LoRA merge or
vLLM/SGLang/TensorRT-LLM conversion.

Copying reads each included file once while recording its digest. Routine reload
checks declared dependency availability/sizes, not a full multi-GB base rehash.
Original native model config/weights are copied; an already ARTI-hooked host is
never resaved as a supposedly clean native base.

The packager copies the referenced model closure and supported tokenizer files,
not the whole training directory. Checkpoints, optimizer state, data, preparation
caches and unrelated files do not enter the package. Publication uses a private
staging directory and a fresh final directory. Existing destinations are not
overwritten; a failed package leaves the source export unchanged.

Omit `--include-base` for another slim package. Packaging a candidate does not
declare that it passed an application-specific acceptance policy. Preserve the
evaluation receipts and make that decision in the caller's workflow.

## Load, Evaluate and Serve After Moving

[Independent Asset Evaluation](qwen-independent-evaluation.md) supports a
separate data/scoring recipe and test-only datasets without optimizer creation.
[Recoverable Prediction Runs](qwen-prediction-runs.md) retain committed batch
outputs when a long local inference job is interrupted.

```bash
arti recipe predict deployments/candidate --data requests.jsonl --output replies.jsonl
arti recipe evaluate deployments/candidate --data labeled-records.jsonl
arti recipe serve deployments/candidate --device cpu --port 8080
```

The Predictor and native CLI use the package's relative base automatically.
No old training directory, dataset or Hub cache is needed for loading. Prediction
still requires its input records; evaluation requires its labeled records.

Slim exports can use an explicitly mounted native base through Python or CLI:

```python
import arti

predictor = arti.CausalLMPredictor.from_pretrained(
    "outputs/model", base_model="mounted-native-base", local_files_only=True,
)
```

```bash
arti recipe predict outputs/model --base-model mounted-native-base --data requests.jsonl --output replies.jsonl
arti recipe evaluate outputs/model --base-model mounted-native-base --data labeled-records.jsonl
arti recipe serve outputs/model --base-model mounted-native-base --device cpu
```

An explicit base override wins over the package's bundled base. Developers remain
responsible for supplying the right base. Custom `layer_factory` assets still
require their installed Python factory; packaging operands does not serialize
arbitrary Python code. The CLI can now accept an explicit `--layer-factory
installed.module:callable` and optional `--factory-kwargs-json options.json`.
See [Use A Custom Qwen Program](qwen-custom-programs.md) for configuration,
continuation and deployment. Artifact metadata never automatically imports code.

The executable inspection/package wrapper is `examples/qwen_deployment_handoff.py`.
The qualified profile is local single-process Torch native causal-LM/Qwen with
SafeTensors, not cloud upload, distributed state, automatic model promotion or
runtime-quality certification.

Design references:
[Transformers local save/load](https://huggingface.co/docs/transformers/main_classes/model),
[Hub cached snapshots](https://huggingface.co/docs/huggingface_hub/guides/manage-cache).
