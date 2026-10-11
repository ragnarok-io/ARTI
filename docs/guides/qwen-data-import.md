# Import Local Qwen SFT Data

Import chat exports or prompt/completion files into one reusable recipe snapshot.
The importer does not load a tokenizer, model or optimizer. It preserves the
actual text and complete message history; the recipe still trains only the final
assistant answer through the model's native EOS.

## Declare Sources

Each source has a logical name and an explicit `train`, `validation` or `test`
role. There is no automatic random split. Train and validation are required;
test is optional. Declare the export's existing qualification/filtering upstream:
an `accepted` field is not automatically interpreted by the importer.

```toml
task = "causal-lm"

[[sources]]
name = "reviewed-train"
path = "exports/train.jsonl"
split = "train"
format = "messages"
[sources.fields]
id = "record_id"
messages = "conversation"

[[sources]]
name = "reviewed-validation"
path = "exports/validation.json"
split = "validation"
format = "prompt_completion"
file_format = "json"
[sources.fields]
prompt = "question"
completion = "answer"

[[sources]]
name = "reviewed-test"
path = "exports/test.jsonl"
split = "test"
format = "messages"
```

Paths resolve relative to the import config, not the current working directory.
`file_format="jsonl"` is the default; `json` means a top-level array of records.
Use as many independently declared sources per split as needed.

Supported data formats:

| Format | Source values | Conversion |
| --- | --- | --- |
| `messages` | Nonempty role/content message list ending in assistant | Preserve the full list |
| `prompt_completion` | Prompt text + completion text | One user + one assistant message |
| `prompt_completion` | Prompt message list ending in user + one assistant completion message | Preserve all prompt history, append completion |

Only `system`, `user`, `assistant` text messages are supported. Tool calling,
multimodal payloads, reasoning-channel fields and multi-message completions need
an explicit task-specific adapter; they are not silently flattened. Extra row
metadata is not used as supervision. No answer is synthesized, trimmed, scored,
or extracted from a rationale. If a raw completion includes a rationale, it is
preserved verbatim: supply the intended final-answer field explicitly.

## Publish A Snapshot

```bash
arti recipe import-data import.toml --output data/reviewed-v1
```

This creates `dataset.jsonl` and `manifest.json` together in a fresh directory.
Existing snapshots are not overwritten; use a new directory for a new revision.
Publication failure does not leave a partially published snapshot.

Every output ID is `source-name:source-id`. A configured `fields.id` accepts a
nonempty string or integer. Without it, IDs use the one-based ordinal among
nonblank source records. This fallback is stable for the frozen source ordering,
not across reordered exports. Use stable upstream IDs for long-lived datasets.

Duplicate output IDs and identical completed conversations across splits are
errors with source/record locations. Within-split repeated conversations remain
in the data and are counted, not deduplicated. This is exact-content checking,
not entity-aware or near-duplicate leakage detection. A native mixed-split ARTI
JSONL can already be used directly without import. An import source assigns all
of its records to its declared role; do not pass a mixed-split file as a
single-role export. Existing recipe split policies are not changed by import.

The manifest records logical source names, raw content hashes, field mappings,
formats, ID rules, per-split counts/fingerprints, repeated counts and the emitted
data hash. It does not embed source paths, raw records or tokenizer state.
Original exports can move or be deleted after successful import. Keep them
separately when audit/reprocessing requires the original records.

## Continue Through The Existing Workflow

Set these paths in a normal causal-LM recipe, relative to that recipe file:

```toml
[data]
path = "data/reviewed-v1/dataset.jsonl"
source_manifest = "data/reviewed-v1/manifest.json"
cache_dir = "data/prepared-chat"
batch_size = 16
max_tokens_per_batch = 4096
max_length = 1024
overlength = "error"
```

```bash
arti recipe prepare-data qwen.toml --output prepared.json
arti recipe check qwen.toml --output preflight.json
arti recipe train qwen.toml --run-dir runs/reviewed-v1
arti recipe resume-run runs/reviewed-v1 --plan-only
arti recipe resume-run runs/reviewed-v1 --run-dir runs/reviewed-v1-recovered
arti recipe evaluate qwen.toml --model runs/reviewed-v1-recovered/model --split test --output test.json
arti recipe package-deployment runs/reviewed-v1-recovered/model --include-base --base-model snapshots/qwen --output deploy/reviewed-v1
```

Use the `model` directory from the completed attempt, including the first run
when no recovery is needed. `check`/`train` load the configured model; `import-data`
does not. `prepare-data` needs only the tokenizer. Local deployment packaging
requires the native base snapshot; it never downloads missing base weights.

`source_manifest` is optional for existing native data. When supplied, preparation
checks its fingerprint and normalized dataset identity before host loading. This
allows equivalent JSON reformatting while rejecting different records, IDs,
message order or split membership. Token-cache identity remains based on the
existing encoded-data contract, not manifest location or irrelevant raw metadata.

The portable manifest enters split/task metadata, `data_source_manifest.json`
in each run, checkpoints and exported model metadata. Evaluation records its
fingerprint. Exact continuation keeps this declared source identity; relocate
both snapshot files without changing their contents to keep the same identity.
Serving/reloading the exported model needs neither data snapshot nor raw files.

## Python API

```python
from arti import (
    CausalLMDataImportConfig, CausalLMDataSource, CausalLMImportFields,
    import_causal_lm_data,
)

declaration = CausalLMDataImportConfig((
    CausalLMDataSource("train.jsonl", "train", "train"),
    CausalLMDataSource(
        "validation.jsonl", "heldout", "validation",
        format="prompt_completion",
        fields=CausalLMImportFields(prompt="question", completion="answer"),
    ),
))
snapshot = import_causal_lm_data(declaration, "data/reviewed-v1")
# Merge snapshot["data"] into the existing recipe's data section.
```

The same API is available from `arti.torch`. This local importer creates an
in-memory text snapshot. `declaration.to_dict()` can be written as a JSON import
config and reloaded with `CausalLMDataImportConfig.read(path)`. It is
not a streaming/remote dataset engine or annotation
review system. Qualification covers offline CPU tiny-Qwen workflow integration;
it does not establish GPU throughput or downstream model-quality gains.

References for the transferred format mechanisms:
[TRL dataset formats](https://huggingface.co/docs/trl/dataset_formats) and
[Hugging Face local data files](https://huggingface.co/docs/datasets/loading).
