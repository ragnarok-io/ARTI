from __future__ import annotations

import torch

import arti
from arti import LiteralSequenceDecoder


def test_literal_sequence_decoder_teacher_forcing_shape_and_gradient() -> None:
    torch.manual_seed(5)
    decoder = LiteralSequenceDecoder(context_dim=12, vocab_tensor_dim=8, hidden_dim=10, key_dim=6)
    context = torch.randn(3, 12)
    vocab = torch.randn(3, 7, 2, 4)
    teacher_ids = torch.randint(0, 7, (3, 5))

    output = decoder(context, vocab, teacher_ids=teacher_ids, batched_vocab=True)
    output.logits.square().mean().backward()

    assert output.logits.shape == (3, 5, 7)
    assert output.local_ids.shape == (3, 5)
    assert output.lengths is None
    assert any(parameter.grad is not None for parameter in decoder.parameters())


def test_literal_sequence_decoder_supports_terminal_head_ablation() -> None:
    decoder = LiteralSequenceDecoder(
        context_dim=9,
        vocab_tensor_dim=4,
        hidden_dim=8,
        key_dim=5,
        condition_on_vocab=False,
    )
    output = decoder(torch.randn(2, 9), torch.randn(6, 4), steps=4)

    assert decoder.lexicon_context is None
    assert output.logits.shape == (2, 4, 6)
    assert output.local_ids.max().item() < 6


def test_literal_sequence_decoder_reuses_detached_output_cache() -> None:
    torch.manual_seed(7)
    decoder = LiteralSequenceDecoder(context_dim=10, vocab_tensor_dim=6, hidden_dim=9, key_dim=5).eval()
    cache = decoder.prepare_output_vocab(torch.randn(8, 2, 3), detach=True)

    first = decoder(torch.randn(2, 10), cache, steps=3)
    second = decoder(torch.randn(2, 10), cache, steps=2)

    assert not cache.keys.requires_grad
    assert first.logits.shape == (2, 3, 8)
    assert second.logits.shape == (2, 2, 8)


def test_literal_sequence_generate_stops_on_per_sample_eos() -> None:
    decoder = LiteralSequenceDecoder(context_dim=6, vocab_tensor_dim=4, hidden_dim=7, key_dim=5).eval()
    for parameter in decoder.parameters():
        torch.nn.init.zeros_(parameter)

    output = decoder.generate(
        torch.randn(3, 6),
        torch.randn(5, 4),
        eos_local_ids=torch.zeros(3, dtype=torch.long),
        max_steps=6,
    )

    assert output.local_ids.shape == (3, 1)
    assert output.local_ids.eq(0).all()
    assert output.lengths is not None
    assert output.lengths.eq(1).all()


def test_literal_sequence_decoder_is_exported_from_torch_namespace() -> None:
    assert arti.torch.LiteralSequenceDecoder is LiteralSequenceDecoder
    assert arti.torch.LiteralSequenceOutput is arti.LiteralSequenceOutput


def test_literal_sequence_decoder_fit_recipe_updates_parameters() -> None:
    torch.manual_seed(11)
    decoder = LiteralSequenceDecoder(context_dim=6, vocab_tensor_dim=4, hidden_dim=8, key_dim=5)
    batch = {
        "context": torch.randn(3, 6),
        "output_vocab": torch.randn(3, 7, 4),
        "teacher_ids": torch.randint(0, 7, (3, 4)),
        "target_mask": torch.tensor([[True, True, True, False], [True, True, True, True], [True, True, False, False]]),
        "loss_weights": torch.tensor([[1.0, 1.0, 2.0, 0.0], [1.0, 1.0, 2.0, 2.0], [1.0, 2.0, 0.0, 0.0]]),
        "batched_vocab": True,
    }
    before = decoder.bos.detach().clone()

    result = decoder.fit([batch], steps=3, lr=1e-2)

    assert result.steps == 3
    assert result.examples == 9
    assert len(result.losses) == 3
    assert torch.isfinite(torch.tensor(result.final_loss))
    assert not torch.equal(decoder.bos.detach(), before)
    assert arti.fit_literal_sequence is arti.torch.fit_literal_sequence


def test_literal_sequence_fit_recipe_rejects_empty_batches() -> None:
    decoder = LiteralSequenceDecoder(context_dim=5, vocab_tensor_dim=3, hidden_dim=6, key_dim=4)
    try:
        decoder.fit([], steps=1)
    except ValueError as exc:
        assert "at least one batch" in str(exc)
    else:
        raise AssertionError("empty literal fit batches should fail")


def test_literal_sequence_decoder_state_dict_round_trip() -> None:
    torch.manual_seed(17)
    source = LiteralSequenceDecoder(context_dim=7, vocab_tensor_dim=5, hidden_dim=9, key_dim=6).eval()
    context = torch.randn(2, 7)
    vocab = torch.randn(8, 5)
    expected = source(context, vocab, steps=4).logits
    restored = LiteralSequenceDecoder(context_dim=7, vocab_tensor_dim=5, hidden_dim=9, key_dim=6).eval()
    restored.load_state_dict(source.state_dict())

    actual = restored(context, vocab, steps=4).logits

    assert torch.allclose(actual, expected)


def test_literal_sequence_decoder_masks_padded_output_rows() -> None:
    torch.manual_seed(19)
    decoder = LiteralSequenceDecoder(context_dim=6, vocab_tensor_dim=4, hidden_dim=8, key_dim=5).eval()
    context = torch.randn(2, 6)
    vocab = torch.randn(2, 6, 4)
    mask = torch.tensor([[True, True, True, False, False, False], [True, True, True, True, True, False]])

    output = decoder(context, vocab, steps=5, output_mask=mask, batched_vocab=True)

    assert output.local_ids[0].lt(3).all()
    assert output.local_ids[1].lt(5).all()
    assert (output.logits[0, :, 3:] == torch.finfo(output.logits.dtype).min).all()
    assert (output.logits[1, :, 5:] == torch.finfo(output.logits.dtype).min).all()
