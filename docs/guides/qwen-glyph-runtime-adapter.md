# Qwen Glyph Runtime Adapter

`QwenGlyphRuntimeAdapter` is an alpha integration for experiments that need a
Qwen-class dialogue model to keep its normal dialogue path while a separate ARTI
path reads an external visible-word vocabulary.

Install the optional dependencies:

```bash
uv sync --extra torch --extra qwen --extra font
```

Minimal usage:

```python
from arti.integrations.qwen import QwenGlyphRuntimeAdapter

adapter = QwenGlyphRuntimeAdapter.from_pretrained("Qwen/Qwen3-0.6B")

dialogue = adapter.generate("User: Say hello briefly.\nAssistant:", max_new_tokens=16)

readout = adapter.read_glyph_vocab(
    "User: Read the external visible word.\nAssistant:",
    ["strawberry", "strawberrry", "banana", "phase"],
    query_text="strawberrry",
)

print(dialogue)
print(readout.local_index, readout.text)
```

The ordinary dialogue path calls the original frozen Qwen `generate` or
next-token logits. The glyph path renders the supplied vocabulary into rigid
bitmap tensors and scores local runtime slots with an ARTI readout. The output
index is the current local slot, not a permanent tokenizer row.

To verify that an operation does not mutate the dialogue path:

```python
drift = adapter.dialogue_drift(
    ["User: Hello, who are you?\nAssistant:"],
    operation=lambda: adapter.read_glyph_vocab(
        "User: Read the external visible word.\nAssistant:",
        ["alpha", "phase"],
        query_text="phase",
    ),
)

assert drift.max_abs_logit_delta <= 1e-5
assert drift.top1_preserved
```

## Boundaries

This interface does not replace Qwen's tokenizer by itself. It is a frozen-base
adapter surface for controlled runtime-vocab experiments. Full model fine-tuning
needs a separate training loop and must re-run dialogue-preservation checks.
