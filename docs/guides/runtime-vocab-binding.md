# Runtime Vocab Binding

Runtime vocab binding is a stable API for models whose output softmax should be
defined by the current external vocabulary view, not by a fixed learned head row.

The core object is a rigid vocabulary tensor:

```text
vocab_tensor[j] = fixed tensor for the j-th item in this runtime vocab view
```

The tensor can come from bitmap text rendering, an external physical symbol
encoder, or any other deterministic source. ARTI does not learn this tensor.
Learning happens after the tensor is read.

For text vocabularies, ARTI provides a small bitmap renderer:

```python
from arti import render_text_vocab

vocab_tensor = render_text_vocab(["strawbery", "strawberry"], height=16, width=96)
```

If Pillow is installed, pass a font file:

```python
vocab_tensor = render_text_vocab(words, font_path="fonts/Inter-Regular.ttf")
```

Different fonts can be used during training. The tensor stays rigid for each
rendered text/font pair, while the model can learn both stable text identity and
font-dependent visual differences.

After rendering, check that the bitmap vocabulary did not collapse distinct
symbols:

```python
from arti import assert_bitmap_vocab_distinct, bitmap_vocab_report

report = bitmap_vocab_report(vocab_tensor)
assert_bitmap_vocab_distinct(vocab_tensor, min_entropy_bits=0.1)
```

The report includes entropy, pairwise distance, and exact collision pairs. Even
tiny visual differences should remain present in the tensor; exact collisions
mean the rendered vocab has lost symbol information.

## Contract

```python
from arti import RuntimeVocabInput, RuntimeVocabHead

reader = RuntimeVocabInput(vocab_tensor_dim=64, hidden_dim=128)
head = RuntimeVocabHead(hidden_dim=128, vocab_tensor_dim=64)

hidden_in = reader(token_ids, vocab_tensor)  # [B, N, 128]
logits = head(hidden, vocab_tensor)          # [B, N, K]
```

Shapes:

```text
token_ids    [B, N]
vocab_tensor [K, ...]
hidden       [B, N, D] or [B, D]
logits       [B, N, K] or [B, K]
```

`RuntimeVocabInput` and `RuntimeVocabHead` keep the original shared-view API.
For independent input/output vocabularies and padded per-sample output views,
use the literal vocabulary API below.

## Independent Input And Output Vocabularies

The input and output ranges do not need to have the same size, ordering,
segmentation, or item tensor shape:

```python
from arti import LiteralVocabModel

model = LiteralVocabModel(
    input_vocab_tensor_dim=64,
    output_vocab_tensor_dim=96,
    hidden_dim=256,
)

logits = model(
    input_ids,
    input_vocab=input_literal_tensors,    # [K_in, ...]
    output_vocab=output_literal_tensors,  # [K_out, ...]
)
```

`logits.shape[-1]` is always `K_out`. Input ids are meaningful only under the
supplied input view; output indices are meaningful only under the independently
supplied output view.

The reference model has four explicit parts:

```text
LiteralInput(input_vocab)
    -> OutputLexiconContext(output_vocab)
    -> model body
    -> LiteralOutputHead(output_vocab)
```

`OutputLexiconContext` is a small cross-attention residual placed before the
body that should understand the current output range. `LiteralOutputHead` stays
thin: it projects the final hidden state and scores every literal item directly.
The head therefore does not own a permanent output-row codebook.

For an existing model, use the two pieces independently:

```python
from arti import LiteralOutputHead, OutputLexiconContext

lexicon = OutputLexiconContext(
    hidden_dim=model_dim,
    vocab_tensor_dim=literal_dim,
)
head = LiteralOutputHead(
    hidden_dim=model_dim,
    vocab_tensor_dim=literal_dim,
    encoder=lexicon.encoder,  # share one literal encoder
)

cache = lexicon.prepare(output_vocab, detach=True)
hidden = lexicon(input_hidden, cache)
hidden = model_body(hidden)
logits = head(hidden, cache)
```

The explicit `LiteralVocabCache` avoids re-encoding a large output vocabulary
for every autoregressive step. Use `detach=True` for inference. During training,
leave it false so gradients reach the literal encoder and rebuild the cache
after every optimizer update; reusing a training graph or stale encoded keys is
not supported. Batched padded output views are supported with `batched=True`
and a boolean `[B, K]` mask.

Run the standalone decoupled-vocabulary example with:

```bash
uv run --extra torch python examples/literal_vocab_decoupled.py
```

Compaction may be used to summarize a large vocabulary for body conditioning,
but final local-slot scoring should still read every surviving literal item.
Otherwise visually small but identity-bearing differences can disappear from
the output range.

The frozen-Qwen controlled comparison for this split interface is:

```bash
uv run --extra torch --extra qwen python benchmarks/run_qwen_literal_output_context.py
uv run --extra dev python benchmarks/verify_qwen_literal_output_context.py
```

Qwen supplies frozen task-query hidden states while the output range is a
separate shuffled glyph vocabulary. The held-out tasks replace candidate sets,
not merely their slot order, and include set-relative targets that an
independent per-item scorer cannot fully determine. This is evidence for the
placement of output-vocabulary conditioning at the front of the ARTI-side
adapter body. It is not yet evidence for unseen-glyph semantics or open-ended
generation.

## Different Input And Output Segmentation

`LiteralSequenceDecoder` handles the case where the upstream model and output
vocabulary do not share token boundaries. A language model can supply one
context vector while the decoder emits characters or other literal fragments:

```python
from arti import LiteralSequenceDecoder

decoder = LiteralSequenceDecoder(
    context_dim=1024,
    vocab_tensor_dim=225,
    hidden_dim=256,
)

# teacher_ids are local rows in this batch's independently supplied output vocab.
output = decoder(
    qwen_context,
    character_vocab_tensors,
    teacher_ids=target_local_character_ids,
    batched_vocab=True,
)

loss = cross_entropy(output.logits, target_local_character_ids)
```

For inference, prepare a vocabulary once and supply the current local EOS row:

```python
cache = decoder.prepare_output_vocab(character_vocab_tensors, batched=True, detach=True)
generated = decoder.generate(
    qwen_context,
    cache,
    eos_local_ids=eos_local_ids,
    max_steps=64,
)
```

The caller maps `generated.local_ids` back through its current vocabulary. ARTI
does not own the strings or a permanent row order. Visible characters should
remain glyph-only. EOS, padding, newline, and other non-visible controls may
use explicit auxiliary control channels.

The decoder continuously supplies the upstream context to its recurrent loop,
so a long shared sentence prefix cannot erase the answer condition. The output
vocabulary encoder is shared by `OutputLexiconContext`, recurrent literal
feedback, and the terminal `LiteralOutputHead`.

For a small adaptation job, the decoder exposes a tensor-native `.fit()`
recipe:

```python
result = decoder.fit(
    [
        {
            "context": qwen_context,
            "output_vocab": character_vocab_tensors,
            "teacher_ids": target_local_character_ids,
            "target_mask": target_mask,
            "loss_weights": answer_span_weights,
            "batched_vocab": True,
        }
    ],
    steps=500,
    lr=1e-3,
)
```

The recipe performs masked local-slot cross-entropy and optional gradient
clipping. It does not move tensors, construct labels, interpret strings, or own
distributed execution. Larger projects can use the same decoder directly with
Accelerate, DDP, or their existing trainer.

Run the compact API example with:

```bash
uv run --extra torch python examples/literal_sequence_fit.py
```

Run the frozen-Qwen segmentation comparison with:

```bash
uv run --extra torch --extra qwen python benchmarks/run_qwen_literal_segmentation_generation.py
uv run --extra dev python benchmarks/verify_qwen_literal_segmentation_generation.py
```

This benchmark keeps Qwen BPE input, independently shuffles a glyph-character
output vocabulary, and decodes complete answer sentences. It compares a normal
fixed character head, a parameter-matched terminal dynamic head, and front
output-vocabulary conditioning. It does not claim unrestricted tokenizer
replacement or open-ended language-model parity.

## Why The Head Reads The Vocab

The runtime vocab tensor must enter both sides:

```text
input id -> vocab_tensor[id] -> input reader
hidden   -> dynamic head(vocab_tensor) -> K-way softmax
```

A fixed `lm_head` would re-bind the model to permanent output rows and lose the
runtime vocab identity after deep processing. `RuntimeVocabHead` instead scores
the hidden state against keys generated from the current vocabulary tensor.

## Shuffle Equivariance

If the runtime vocabulary view is shuffled, input ids and output softmax order
should shuffle with it:

```python
from arti import permute_runtime_vocab, remap_token_ids

shuffled_vocab = permute_runtime_vocab(vocab_tensor, permutation)
shuffled_ids = remap_token_ids(token_ids, permutation)
```

The intended property is:

```text
model(shuffled_ids, shuffled_vocab) = permutation(model(token_ids, vocab))
```

This keeps token identity attached to the rigid vocab tensor rather than a fixed
row index.

## Replacement

The stronger alpha target is runtime vocabulary replacement. The current
softmax range is defined by the supplied `vocab_tensor`, so a deployment can
replace the candidate set instead of only shuffling a fixed global table:

```text
runtime vocab A -> logits over A
runtime vocab B -> logits over B
```

For this to work, the output head must directly read the same rigid symbol
source used by the input side. Otherwise the model can drift back toward a
private codebook and memorize output rows instead of learning the symbol.

The mechanism benchmark `benchmarks/run_runtime_vocab_replacement.py` tests this
with disjoint train and held-out symbol sets. The model must copy by rigid
symbol into the current runtime softmax range, not by permanent row id.

## Legacy Explicit Pulse Pairing

Runtime vocab and the legacy explicit pulse compressor can be used independently,
but they become more useful together. Different vocabularies can segment the
same external stream into different token sequences. Legacy explicit pulse
compression lets those token sequences map back to the same or nearby latent
pulse sequence:

```text
runtime vocab A token stream -> pulse sequence
runtime vocab B token stream -> pulse sequence
```

Use legacy `PulseCompressor` with `token_weight` when tokens cover different
amounts of the original stream. The mechanism benchmark
`benchmarks/run_pulse_vocab_invariance.py` checks this boundary with single
symbol tokens versus variable-length chunk tokens.

For compression safety, validate the pulse output itself:

```python
from arti import assert_pulse_distinct, pulse_distinctness_report

report = pulse_distinctness_report(raw_vocab_tensor, pulse_tensor)
assert_pulse_distinct(
    raw_vocab_tensor,
    pulse_tensor,
    min_pulse_distance=1e-6,
    min_distance_retention=0.05,
)
```

Legacy explicit pulse may reduce sequence length, but it should not collapse visible symbol
differences such as `r` versus `rr` into the same latent state.

## RuntimeVocabPulseAdapter

`RuntimeVocabPulseAdapter` is the reusable alpha layer used by the Qwen semantic
benchmark. It is intentionally small: an upstream model produces a context
tensor, the current runtime vocab supplies candidate tensors, and the adapter
returns logits over the current local slots.

```python
from arti import RuntimeVocabPulseAdapter

adapter = RuntimeVocabPulseAdapter(
    context_dim=1024,
    vocab_tensor_dim=64,
    hidden_dim=256,
)

logits = adapter(context, runtime_vocab)
```

Shapes:

```text
context       [B, C]
runtime_vocab [B, K, ...] or [K, ...]
logits        [B, K]
```

If candidates come from variable token streams, pulse-compress each candidate
before passing it to the adapter:

```text
external token stream -> pulse_compress(...) -> runtime_vocab[b, k]
```

The adapter does not own the vocabulary identity. It reads the supplied tensor
view at inference time, so `logits[:, j]` means "candidate `j` in this current
runtime vocab", not a fixed row in a global language-model head.

Run the minimal local example:

```bash
uv run --extra torch python examples/runtime_vocab_pulse_adapter.py
```

Run the Qwen adapter-level semantic gate:

```bash
uv run --extra torch --extra qwen python benchmarks/run_qwen_runtime_vocab_pulse_semantic.py
uv run --extra torch --extra qwen python benchmarks/verify_qwen_runtime_vocab_pulse_semantic.py
```

The Qwen benchmark freezes `Qwen/Qwen3-0.6B`, trains small adapters, and compares
fixed head, runtime vocab only, pulse only, and runtime vocab + pulse. The
answer metric is exact semantic match through the current local vocab index.
This remains adapter-level validation; it is not full tokenizer retraining.

## Boundary

This feature is alpha. The built-in fallback renderer is intentionally small and
ASCII-oriented. For serious text experiments, use explicit font files and keep
the font, canvas size, and normalization pipeline locked in the experiment
metadata.

For micro-glyph sensitive vocabularies, keep the default `14x96` canvas or
larger, avoid resizing after rendering, and validate the vocabulary with:

```bash
uv run --extra dev python benchmarks/run_micro_glyph_distinctness.py
uv run --extra dev python benchmarks/verify_micro_glyph_distinctness.py
```

Visible characters should remain glyph-first. Use codepoint auxiliary channels
only for control characters, invisible format marks, replacement glyphs, or
explicit confusable-character audits.
