# Prepare Qwen Training Data

[Import Local SFT Data](qwen-data-import.md) converts chat or prompt/completion
exports to the native JSONL format and supplies an optional source manifest.

Prepared tokens can be reused across [token-budget batch policies](qwen-token-batching.md).
The budget changes scheduling/padding, not tokenization or the cache identity.

Prepare and check text supervision before allocating host-model memory. The
preparation entrance uses only the recipe, original JSONL and native tokenizer;
it does not instantiate the host, load weights, create an optimizer or use CUDA.

## One Preparation for the Workflow

Declare a writable local cache in the same recipe used for training:

```toml
[data]
path = "conversations.jsonl"
cache_dir = "prepared-chat"
batch_size = 16
max_length = 1024
overlength = "error"
pad_to_multiple_of = 8
group_by_length = true

[data.template_kwargs]
enable_thinking = false
```

Paths are relative to the recipe file. Omit `cache_dir` (or use JSON `null`) to
retain in-memory preparation without disk writes. Shared/read-only input data
can use an explicitly writable cache directory elsewhere; ARTI does not require
write access beside the dataset.

```bash
arti recipe prepare-data recipe.toml --output prepared.json
arti recipe plan recipe.toml
arti recipe train recipe.toml --run-dir outputs/first
arti recipe resume-run outputs/first --run-dir outputs/recovered
```

The first preparation renders/tokenizes each original record once and stores its
unpadded `input_ids`, labels and prompt boundary. Later plan/train/recovery calls
reuse compatible prepared tensors across processes. Each call still reads and
validates the original JSONL and its identity; a cache is not a new supervision
source and does not make the original dataset optional.

`prepared.json` reports per-split retained/dropped IDs, input and supervised token
counts, maximum sequence length, tokenizer/data fingerprints and preparation
status. A run saves the same local cache provenance in `data_preparation.json`
and its task result. `cache_hit`, `encoded_samples` and `elapsed_seconds` make it
possible to distinguish actual reuse from merely having a cache directory.

The source of truth remains the complete conversation, with only the final
assistant answer through native EOS supervised. Earlier turns stay in the input;
prompt and padding labels remain `-100`. Real EOS remains supervised even when
PAD shares its ID. Oversized records error by default or appear under explicit
`dropped_ids`; neither preparation nor cache silently truncates answers.

## Cache Identity

The content key includes normalized records, tokenizer vocabulary/special tokens/
native template, field mapping, template kwargs, max length, overlength policy
and the preparation format. Changing those inputs creates a new entry.
Formatting the same JSONL differently or moving it does not change normalized
data identity. A cache is a derived local artifact, not a training checkpoint or
an inference model.
Cached preparation is a snapshot of the rendered dataset. Keep rendering inputs
deterministic or supply stable template kwargs when exact reuse is required.

Batch size, padding multiple, shuffle seed, length grouping and optimizer settings
do not alter stored unpadded tokenization. They can reuse the same entry while
retaining their own strict training/loader compatibility rules. Cache paths,
timing and hits are runtime provenance, not part of the task fingerprint or
portable deployment metadata. An old checkpoint can therefore be resumed with
an enabled cache without pretending that its task changed.

Each entry contains JSON metadata and SafeTensors with flattened sequences and
offsets. Files publish from a unique staging directory; the complete manifest is
published last. Missing/incomplete/invalid derived cache rebuilds from the source.
Source validation or encoding errors still fail preparation. No automatic cache
pruning, registry, remote upload, permissions service or new trainer is added.

## Python and Local Tokenizers

```python
import arti

report = arti.prepare_causal_lm_data("recipe.toml", output="prepared.json")
assert not report["weights_loaded"]
print(report["data_preparation"])
```

Advanced callers can supply their already loaded native tokenizer:

```python
config = arti.CausalLMRecipeConfig.read("recipe.toml")
report = arti.prepare_causal_lm_data(config, tokenizer=native_tokenizer)
```

Training must use the same tokenizer/template identity to hit that entry. A
different tokenizer is a distinct preparation, not an implicit compatibility
conversion. For an existing run's saved tokenizer, the executable example is:

```bash
python examples/qwen_prepare_data.py recipe.toml --tokenizer outputs/first/tokenizer --output prepared.json
```

Its optional `--cache-dir` overrides only that command; configure `data.cache_dir`
in the training recipe to reuse the same location throughout the workflow.

The qualified profile remains local in-memory text-only causal-LM/Qwen SFT. It
does not imply streaming/mmap datasets, multimodal preprocessing, distributed
preparation or GPU training acceleration. Measured cold/warm preparation time is
startup evidence, not model quality or steady-state training throughput.
