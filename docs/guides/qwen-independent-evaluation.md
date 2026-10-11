# Evaluate A Qwen Asset Independently

Evaluate an exported model on new labeled data without constructing a training
session, optimizer or scheduler. The model asset determines the architecture,
weights, tokenizer and chat contract; the evaluation recipe determines the
dataset, split, scoring, generation and batching policy.

## Use A Separate Evaluation Recipe

Copy the trained recipe's pinned `[model]` and `template_kwargs`. Change `[data]`
to the evaluation snapshot. The file may contain only `test` records:

```json
{"id":"heldout-001","split":"test","messages":[{"role":"user","content":"A labeled task input"},{"role":"assistant","content":"The real expected response"}]}
```

It may also contain multiple explicit splits. Only the selected split is encoded
or evaluated; a long training example does not consume evaluation tokenization
or fail the test split's length budget. The importer/annotation adapter supplies
real labels, not the evaluator. Full earlier turns remain in the prompt.

```toml
task = "causal-lm"

[model]
# Copy the exact values from the trained recipe, including revision/dtype/attention.
id = "Qwen/Qwen3-0.6B"
revision = "c1899de289a04d12100db370d81485cdf75e47ca"
dtype = "bfloat16"
attn_implementation = "sdpa"
local_files_only = true

[data]
path = "heldout.jsonl"
cache_dir = "prepared-evaluation"
batch_size = 16
max_tokens_per_batch = 4096
max_length = 1024
overlength = "error"
pad_to_multiple_of = 8

[data.template_kwargs]
enable_thinking = false

[evaluation]
generate_samples = 32
sampling = "seeded"
sample_seed = 17
comparison = "exact_text"

[generation]
max_new_tokens = 64
batch_size = 8
```

The illustrated model settings match the pinned example workflow; use your own
asset's settings. A mismatched model, tokenizer or template is rejected before
host allocation. `local_files_only` is a loading preference, not neural identity.
An explicit local native base remains caller-provided provenance.

```bash
arti recipe evaluate evaluation.toml --model deployments/candidate --split test --output test.json
arti recipe evaluate evaluation.toml --model outputs/model --base-model snapshots/native-qwen --split test --output test.json
```

`--device` now controls this standalone evaluation load. The default is CPU;
request CUDA explicitly when available and budgeted. The asset's native bundled
base is used after a package move. No original training data, run directory,
optimizer checkpoint or training engine is needed.

An existing full training recipe remains accepted. Its adapter, optimizer,
training and run sections do not rebuild or alter the exported model. Its
generation/evaluation sections apply to this evaluation, including the existing
`--generate-samples 0` override. Only training/resuming training enforces the
original task/data/optimizer continuation contract.

## Keep Preparation And Attribution Explicit

Selected data uses the same native chat template, final-assistant-through-EOS
supervision, target masks and prepared SafeTensors cache as training. Warm
evaluation does not render/tokenize again. A split-specific cache uses the
selected normalized rows; it does not claim to reuse a full mixed-split training
cache under a different identity. Training's default train/validation requirement
is unchanged. A requested empty or fully dropped split is an error.

`overlength="error"` is the default. Opt-in `drop` records excluded IDs and counts
in the report; answers are never silently truncated. The token budget applies
to teacher-forced dense batches, while generation retains its separate profile.

Optional `data.source_manifest` is verified against the complete input snapshot
before selection. The report then distinguishes:

- `task_recipe_sha256`: the exported model's original training contract.
- `model_source.manifest_sha256`: the serialized source manifest loaded, including
  its declared weight identities. This is not a hash of live mutable Python state.
- `dataset_sha256`: this split's actual encoded examples and labels.
- `dataset.sha256`: this evaluation's selected normalized source rows.
- `dataset.source_dataset_sha256`: the complete input snapshot before selection.
- `source_manifest_sha256`: the optional imported snapshot's provenance.

The data manifest includes selected roles, retained/excluded IDs and token counts.
The report keeps the effective data, generation, scoring and batching settings,
plus cache hit/work information. It records that no optimizer/training session
was created and no training state was loaded. A completed evaluation is not an
automatic model promotion or proof of application quality.

## Reuse An Already Loaded Model

```python
import arti

predictor = arti.CausalLMPredictor.from_pretrained(
    "deployments/candidate", local_files_only=True,
)
report = predictor.evaluate_recipe(
    "evaluation.toml", split="test", output="test.json",
)
```

Or load and evaluate in one call:

```python
report = arti.evaluate_causal_lm_deployment(
    "deployments/candidate", "evaluation.toml", split="test",
    output="test.json", local_files_only=True,
)
```

The helper prepares and validates data before constructing the host. Both
entrances reuse the existing inference/evaluation implementation, not another
trainer. Owned ARTI parameters, resource state, registered host buffers, modes
and supported Torch/Python/CUDA RNG are restored between teacher-forced and
generation branches. Model loading preserves the supported existing RNG stream.
There is no copy of all frozen host parameters. Arbitrary external generators,
unregistered Python state and manual host/parameter mutations remain caller-owned.

Custom program assets require their explicit installed factory through the
recipe's `layer_factory` or `--layer-factory`; saved metadata alone does not import
Python code. An already loaded predictor does not need another factory call.

The raw-record entrance is retained:

```bash
arti recipe evaluate deployments/candidate --data new-labeled.jsonl --output answers.json
```

This evaluates all supplied rows through the asset's saved profile. Use the
recipe entrance for explicit split selection, cache, length/drop and data-source
policy. Compare the resulting stored receipts without another model run:

```bash
arti recipe compare-evaluations test.json --baseline baseline-test.json --output comparison.json
```

Local CPU tiny-Qwen and installed-package checks qualify this workflow. CUDA
performance, remote serving, judges and model-quality improvements are separate
checks. Reference mechanisms:
[HF's independent model/data evaluator](https://huggingface.co/docs/evaluate/base_evaluator)
and [PEFT model loading](https://huggingface.co/docs/peft/package_reference/peft_model).
