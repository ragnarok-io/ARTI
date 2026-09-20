"""Independent input/output literal vocabulary quickstart."""

from __future__ import annotations

import torch

from arti import LiteralVocabModel, render_text_vocab


input_words = ["read", "this", "string"]
output_words = ["yes", "no", "maybe", "later"]

input_vocab = render_text_vocab(input_words, height=14, width=64)
output_vocab = render_text_vocab(output_words, height=14, width=96)

model = LiteralVocabModel(
    input_vocab_tensor_dim=14 * 64,
    output_vocab_tensor_dim=14 * 96,
    hidden_dim=64,
).eval()

# Input ids address only input_vocab. Output logits address only output_vocab.
input_ids = torch.tensor([[0, 1, 2]])
output_cache = model.prepare_output_vocab(output_vocab, detach=True)

with torch.no_grad():
    logits = model(input_ids, input_vocab, output_cache)

local_index = int(logits[0, -1].argmax())
print({
    "input_vocab_size": len(input_words),
    "output_vocab_size": len(output_words),
    "logits_shape": list(logits.shape),
    "selected_output_literal": output_words[local_index],
})
