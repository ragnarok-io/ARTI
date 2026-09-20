"""Quick-fit a character-level dynamic literal decoder."""

from __future__ import annotations

import torch

from arti import LiteralSequenceDecoder, render_text_vocab


visible_symbols = ["a", "b", "."]
visible = render_text_vocab(visible_symbols, height=14, width=16).flatten(start_dim=1)
visible = torch.cat([visible, torch.zeros(len(visible_symbols), 1)], dim=-1)
eos = torch.zeros(1, visible.shape[-1])
eos[0, -1] = 1.0
output_vocab = torch.cat([visible, eos], dim=0)

decoder = LiteralSequenceDecoder(
    context_dim=8,
    vocab_tensor_dim=output_vocab.shape[-1],
    hidden_dim=16,
    key_dim=8,
)

batch = {
    "context": torch.randn(2, 8),
    "output_vocab": output_vocab,
    "teacher_ids": torch.tensor([[0, 1, 2, 3], [1, 0, 2, 3]]),
    "target_mask": torch.ones(2, 4, dtype=torch.bool),
}

result = decoder.fit([batch], steps=30, lr=3e-3)
cache = decoder.prepare_output_vocab(output_vocab, detach=True)
generated = decoder.generate(
    batch["context"],
    cache,
    eos_local_ids=torch.full((2,), 3, dtype=torch.long),
    max_steps=8,
)

print({"final_loss": result.final_loss, "local_ids": generated.local_ids.tolist()})
