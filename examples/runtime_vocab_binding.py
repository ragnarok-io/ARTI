"""Runtime vocabulary binding alpha example."""

from __future__ import annotations

import torch

from arti import RuntimeVocabModel, permute_runtime_vocab, remap_token_ids


if __name__ == "__main__":
    torch.manual_seed(17)
    model = RuntimeVocabModel(vocab_tensor_dim=16, hidden_dim=32)
    vocab_tensor = torch.randn(12, 4, 4)
    token_ids = torch.tensor([[1, 5, 9]])

    logits = model(token_ids, vocab_tensor)

    permutation = torch.randperm(vocab_tensor.shape[0])
    shuffled_vocab = permute_runtime_vocab(vocab_tensor, permutation)
    shuffled_ids = remap_token_ids(token_ids, permutation)
    shuffled_logits = model(shuffled_ids, shuffled_vocab)

    print("logits", tuple(logits.shape))
    print("shuffle_equivariant", torch.allclose(shuffled_logits, logits.index_select(-1, permutation), atol=1e-5))
