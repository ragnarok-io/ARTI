"""Save and resume an ARTI model with the arti.st protocol."""

from __future__ import annotations

from tempfile import TemporaryDirectory

import torch

import arti
from arti import LiteralSequenceDecoder


torch.manual_seed(7)
model = LiteralSequenceDecoder(context_dim=8, vocab_tensor_dim=5, hidden_dim=16, key_dim=8)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
context = torch.randn(2, 8)
runtime_vocab = torch.randn(2, 6, 5)
teacher_ids = torch.randint(0, 6, (2, 4))

loss = model(context, runtime_vocab, teacher_ids=teacher_ids, batched_vocab=True).logits.square().mean()
loss.backward()
optimizer.step()

with TemporaryDirectory() as directory:
    saved = arti.save(
        model,
        f"{directory}/arti.st",
        glyph_tensors=torch.randn(6, 1, 14, 16),
        vocab_metadata={"items": ["a", "b", "c", ".", " ", "<eos>"]},
        optimizer=optimizer,
        training_state={"step": 1},
    )

    restored = LiteralSequenceDecoder(context_dim=8, vocab_tensor_dim=5, hidden_dim=16, key_dim=8)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=1e-3)
    loaded = arti.load(
        saved.weights_path,
        model=restored,
        optimizer=restored_optimizer,
        map_location="cpu",
    )

    print(
        {
            "weights": saved.weights_path.name,
            "glyphs": saved.glyphs_path.name if saved.glyphs_path else None,
            "step": loaded.training_state["step"],
            "weight_sha256": saved.weights_sha256,
        }
    )
