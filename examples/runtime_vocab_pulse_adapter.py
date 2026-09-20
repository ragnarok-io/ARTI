"""Train a tiny RuntimeVocabPulseAdapter on replaceable runtime vocab views.

This example keeps the upstream model abstract: ``context`` can be a frozen LLM
hidden state, a transformer block output, or any other tensor. The adapter reads
the current runtime vocab tensors and returns logits over the current local
slots.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from arti import RuntimeVocabPulseAdapter


def make_vocab(universe: int, dim: int) -> torch.Tensor:
    ids = torch.arange(universe, dtype=torch.float32)
    freqs = torch.arange(1, dim + 1, dtype=torch.float32)
    return torch.sin(ids[:, None] * freqs[None, :] * 0.13)


def sample_batch(vocab: torch.Tensor, *, batch: int, view_size: int, context_dim: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    contexts = []
    views = []
    targets = []
    for _ in range(batch):
        order = torch.randperm(vocab.shape[0])[:view_size]
        target = torch.randint(0, view_size, ())
        view = vocab.index_select(0, order)
        # A real model would provide this context. Here it is a noisy projection
        # of the correct candidate so the example trains quickly.
        context = F.pad(view[target], (0, context_dim - view.shape[-1])) + 0.01 * torch.randn(context_dim)
        contexts.append(context)
        views.append(view)
        targets.append(target)
    return torch.stack(contexts), torch.stack(views), torch.stack(targets)


def main() -> None:
    torch.manual_seed(23)
    context_dim = 16
    vocab_dim = 8
    view_size = 6
    adapter = RuntimeVocabPulseAdapter(context_dim=context_dim, vocab_tensor_dim=vocab_dim, hidden_dim=32)
    vocab = make_vocab(universe=32, dim=vocab_dim)
    opt = torch.optim.AdamW(adapter.parameters(), lr=3e-3)

    for _ in range(120):
        context, view, target = sample_batch(vocab, batch=64, view_size=view_size, context_dim=context_dim)
        logits = adapter(context, view)
        loss = F.cross_entropy(logits, target)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    context, view, target = sample_batch(vocab, batch=32, view_size=view_size, context_dim=context_dim)
    pred = adapter(context, view).argmax(dim=-1)
    accuracy = (pred == target).float().mean().item()
    print(f"runtime vocab local-index accuracy: {accuracy:.3f}")
    print(f"first target={int(target[0])} predicted={int(pred[0])} current_vocab_shape={tuple(view.shape)}")


if __name__ == "__main__":
    main()
