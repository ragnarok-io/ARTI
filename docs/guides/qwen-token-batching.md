# Batch Qwen Data By A Token Budget

Use a row cap and a dense padded-token budget when conversations have different
lengths. Short examples can share a larger batch while long examples use fewer
rows. Each conversation remains an independent row with its own attention mask;
this is not sequence packing or concatenation of unrelated histories.

## Configure The Existing Recipe

```toml
[data]
path = "chat.jsonl"
batch_size = 64
max_tokens_per_batch = 4096
pad_to_multiple_of = 8
group_by_length = true
```

`batch_size` is still the maximum number of rows. `max_tokens_per_batch` is
optional; omit it to retain fixed-row batching. A budgeted batch satisfies:

```text
row_count * aligned_max_sequence_length <= max_tokens_per_batch
```

Alignment includes `pad_to_multiple_of`. An individual retained record that
cannot fit reports its ID and required padded tokens before the recipe loads
host weights. Increase the budget or deliberately change the data policy; this
feature does not silently truncate an answer, remove EOS or drop records.

This token count is a shape/work proxy, not a VRAM promise. Activation memory
depends on the backbone, attention implementation, precision and program. It
does not estimate quadratic attention cost or allocate/free device memory.

## Inspect Before Training

```bash
arti recipe prepare-data recipe.toml --output preparation.json
arti recipe plan recipe.toml --output plan.json
arti recipe check recipe.toml
arti recipe train recipe.toml --run-dir outputs/train
```

`batching_plan` describes one full deterministic epoch: batch count, min/max rows,
maximum/total padded tokens, useful input tokens and padding efficiency. Planning
does not advance the loader or allocate all padded batches. It is static token
accounting, not a measured GPU speed or memory result. `check` constructs only
the first batch instead of materializing the complete padded training split.

Prepared-token caches do not depend on this budget: tokenization, original IDs,
targets and supervision are unchanged. Switching the budget can reuse the same
derived tokens without another model load or encoding pass.

## Keep Training And Recovery Coherent

Torch and local Accelerate consume the same budgeted epoch order. Existing
seed/length grouping is retained, and the cursor counts examples, not a guessed
fixed-size batch number. Gradient accumulation still counts microbatches; one
optimizer boundary is normalized by its actual supervised tokens, not by an
average of unequal-row/unequal-token batch means.

```bash
arti recipe resume-run outputs/train --run-dir outputs/continued
```

The resolved recipe and loader state preserve the budget. Exact continuation
requires the same row cap, grouping, seed, budget and data. An old checkpoint
without a budget still resumes with the default `None`. Do not silently change
these settings under the label of exact resume.

Batching settings are runtime metadata, not a new neural component or prepared
data format. They do not change the task/data fingerprint or prevent model-only
loading into a deliberately new training run. Validation/checkpoint-selection,
budgeted stops and complete checkpoint saving use the existing workflow.

## Evaluation And Deployment

Teacher-forced validation and final evaluation use the same budget, and reports
include `batching_config` with the actual maximum padded input size observed.
The existing export/checkpoint metadata and `deployment.json` carry the optional
budget; inspection and offline packaging preserve it. Reloaded predictors apply
it to teacher-forced evaluation of new labeled records.

```bash
arti recipe evaluate outputs/train/model --data labeled.jsonl --output evaluation.json
```

Generated replies are different work: `generation.batch_size` and
`generation.max_new_tokens` remain their own settings. A training/evaluation
token budget is not a generation/KV-cache memory limit. Paired evaluation
comparison exposes batching metadata; matching per-token NLL remains subject to
the original encoded-data/denominator rules.

The qualified profile is local single-process Torch/Accelerate with the existing
text/Qwen recipe and explicit custom-program dependencies. GPU throughput,
distributed loaders, packing/block-diagonal attention and automatic VRAM tuning
require separate work and measurement.

Reference: [Transformers dynamic padding and alignment](https://huggingface.co/docs/transformers/main/pad_truncation).
