"""Use text tensor layouts as runtime vocab candidates."""

from __future__ import annotations

import torch

from arti import RuntimeVocabPulseAdapter, TextTensorConfig, render_text_layout


def candidate_tensor(text: str, config: TextTensorConfig) -> torch.Tensor:
    layout = render_text_layout(text, config=config)
    return layout.to_pulse_tensor(pulse_count=4).pulse.squeeze(0).flatten()


def main() -> None:
    torch.manual_seed(31)
    config = TextTensorConfig(glyph_height=7, glyph_width=5, normalization="raw", identity_mode="glyph_only")
    candidates = ["rr", "r\u200br", "r\nr", "r\fr"]
    vocab = torch.stack([candidate_tensor(text, config) for text in candidates])
    adapter = RuntimeVocabPulseAdapter(context_dim=vocab.shape[-1], vocab_tensor_dim=vocab.shape[-1], hidden_dim=32)

    context = vocab[1].unsqueeze(0)
    logits = adapter(context, vocab)
    print(f"runtime vocab tensor shape: {tuple(vocab.shape)}")
    print(f"target candidate: {candidates[1]!r}")
    print(f"logits shape: {tuple(logits.shape)}")


if __name__ == "__main__":
    main()
