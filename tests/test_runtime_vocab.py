from __future__ import annotations

import torch

import arti

from arti import (
    LiteralOutputHead,
    LiteralVocabModel,
    OutputLexiconContext,
    RuntimeVocabHead,
    RuntimeVocabInput,
    RuntimeVocabModel,
    RuntimeVocabPulseAdapter,
    attach_runtime_vocab_semantics,
    gather_runtime_vocab,
    permute_runtime_vocab,
    remap_token_ids,
)


def test_runtime_vocab_input_gathers_current_vocab_view() -> None:
    torch.manual_seed(5)
    reader = RuntimeVocabInput(vocab_tensor_dim=6, hidden_dim=4)
    vocab = torch.randn(7, 2, 3)
    token_ids = torch.tensor([[0, 3, 6]])

    keys = reader.encoder(vocab)
    x = reader(token_ids, vocab)

    assert x.shape == (1, 3, 4)
    assert torch.allclose(x[0, 1], keys[3])


def test_runtime_vocab_head_outputs_current_vocab_size() -> None:
    torch.manual_seed(7)
    head = RuntimeVocabHead(hidden_dim=5, vocab_tensor_dim=4)
    hidden = torch.randn(2, 3, 5)
    vocab = torch.randn(11, 4)

    logits = head(hidden, vocab)

    assert logits.shape == (2, 3, 11)


def test_runtime_vocab_permutation_equivariance_for_head() -> None:
    torch.manual_seed(11)
    head = RuntimeVocabHead(hidden_dim=6, vocab_tensor_dim=4)
    hidden = torch.randn(2, 6)
    vocab = torch.randn(9, 2, 2)
    permutation = torch.tensor([3, 1, 8, 0, 2, 5, 4, 7, 6])

    logits = head(hidden, vocab)
    shuffled_logits = head(hidden, permute_runtime_vocab(vocab, permutation))

    assert torch.allclose(shuffled_logits, logits.index_select(-1, permutation), atol=1e-6)


def test_runtime_vocab_model_remaps_token_ids_under_vocab_shuffle() -> None:
    torch.manual_seed(13)
    model = RuntimeVocabModel(vocab_tensor_dim=4, hidden_dim=8)
    vocab = torch.randn(10, 4)
    token_ids = torch.tensor([[1, 4, 7, 2]])
    permutation = torch.tensor([2, 9, 4, 1, 7, 0, 3, 8, 5, 6])
    shuffled_vocab = permute_runtime_vocab(vocab, permutation)
    shuffled_ids = remap_token_ids(token_ids, permutation)

    logits = model(token_ids, vocab)
    shuffled_logits = model(shuffled_ids, shuffled_vocab)

    assert torch.allclose(shuffled_logits, logits.index_select(-1, permutation), atol=1e-5)


def test_gather_runtime_vocab_accepts_vector_token_ids() -> None:
    keys = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    gathered = gather_runtime_vocab(keys, torch.tensor([2, 0]))

    assert gathered.tolist() == [[6.0, 7.0, 8.0], [0.0, 1.0, 2.0]]


def test_runtime_vocab_pulse_adapter_scores_batched_current_vocab() -> None:
    torch.manual_seed(17)
    adapter = RuntimeVocabPulseAdapter(context_dim=5, vocab_tensor_dim=8, hidden_dim=12)
    context = torch.randn(3, 5)
    runtime_vocab = torch.randn(3, 4, 2, 4)

    logits = adapter(context, runtime_vocab)
    loss = logits.square().mean()
    loss.backward()

    assert logits.shape == (3, 4)
    assert all(param.grad is not None for param in adapter.parameters())


def test_runtime_vocab_pulse_adapter_supports_shared_vocab_view() -> None:
    torch.manual_seed(19)
    adapter = RuntimeVocabPulseAdapter(context_dim=5, vocab_tensor_dim=8, hidden_dim=12)
    context = torch.randn(2, 5)
    runtime_vocab = torch.randn(6, 2, 4)

    logits = adapter(context, runtime_vocab)

    assert logits.shape == (2, 6)


def test_attach_runtime_vocab_semantics_appends_per_item_anchor() -> None:
    vocab = torch.arange(24, dtype=torch.float32).reshape(4, 2, 3)
    semantic = torch.ones(4, 5)

    combined = attach_runtime_vocab_semantics(vocab, semantic, vocab_scale=0.5, semantic_scale=2.0)

    assert combined.shape == (4, 11)
    assert torch.allclose(combined[0, :6], vocab[0].flatten() * 0.5)
    assert torch.allclose(combined[0, 6:], torch.full((5,), 2.0))
    assert arti.torch.attach_runtime_vocab_semantics is attach_runtime_vocab_semantics


def test_attach_runtime_vocab_semantics_supports_batched_vocab() -> None:
    vocab = torch.randn(2, 3, 4)
    semantic = torch.randn(2, 3, 5)

    combined = attach_runtime_vocab_semantics(vocab, semantic)

    assert combined.shape == (2, 3, 9)
    assert torch.allclose(combined[:, :, :4], vocab)
    assert torch.allclose(combined[:, :, 4:], semantic)


def test_attach_runtime_vocab_semantics_rejects_misaligned_axes() -> None:
    vocab = torch.randn(2, 3, 4)
    semantic = torch.randn(2, 4, 5)

    try:
        attach_runtime_vocab_semantics(vocab, semantic)
    except ValueError as exc:
        assert "leading axes" in str(exc)
    else:
        raise AssertionError("expected ValueError for misaligned semantic axes")


def test_literal_vocab_model_decouples_input_and_output_vocabularies() -> None:
    torch.manual_seed(23)
    model = LiteralVocabModel(input_vocab_tensor_dim=6, output_vocab_tensor_dim=10, hidden_dim=12)
    input_vocab = torch.randn(7, 2, 3)
    output_vocab = torch.randn(11, 2, 5)
    token_ids = torch.tensor([[1, 4, 6], [0, 2, 3]])

    logits = model(token_ids, input_vocab, output_vocab)
    loss = logits.square().mean()
    loss.backward()

    assert logits.shape == (2, 3, 11)
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_literal_vocab_model_is_equivariant_to_output_vocab_permutation() -> None:
    torch.manual_seed(29)
    model = LiteralVocabModel(input_vocab_tensor_dim=4, output_vocab_tensor_dim=6, hidden_dim=10).eval()
    input_vocab = torch.randn(5, 4)
    output_vocab = torch.randn(8, 2, 3)
    token_ids = torch.tensor([[0, 3, 1]])
    permutation = torch.tensor([5, 1, 7, 0, 4, 2, 6, 3])

    logits = model(token_ids, input_vocab, output_vocab)
    shuffled = model(token_ids, input_vocab, output_vocab.index_select(0, permutation))

    assert torch.allclose(shuffled, logits.index_select(-1, permutation), atol=1e-5)


def test_output_lexicon_cache_is_reusable_across_context_and_head() -> None:
    torch.manual_seed(31)
    context = OutputLexiconContext(hidden_dim=8, vocab_tensor_dim=6)
    head = LiteralOutputHead(hidden_dim=8, vocab_tensor_dim=6, encoder=context.encoder)
    vocab = torch.randn(9, 2, 3)
    hidden = torch.randn(2, 4, 8)
    cache = context.prepare(vocab)

    conditioned = context(hidden, cache)
    logits = head(conditioned, cache)

    assert cache.keys.shape == (9, 8)
    assert conditioned.shape == hidden.shape
    assert logits.shape == (2, 4, 9)


def test_literal_modules_support_batched_padded_output_vocabularies() -> None:
    torch.manual_seed(37)
    model = LiteralVocabModel(input_vocab_tensor_dim=4, output_vocab_tensor_dim=6, hidden_dim=8)
    input_vocab = torch.randn(6, 4)
    output_vocab = torch.randn(2, 5, 2, 3)
    output_mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, True]])
    token_ids = torch.tensor([[1, 2], [3, 4]])

    cache = model.prepare_output_vocab(output_vocab, mask=output_mask, batched=True)
    logits = model(token_ids, input_vocab, cache)

    assert logits.shape == (2, 2, 5)
    assert torch.isfinite(logits[0, :, :3]).all()
    assert (logits[0, :, 3:] == torch.finfo(logits.dtype).min).all()
    assert torch.isfinite(logits[1]).all()


def test_detached_literal_cache_supports_inference_reuse() -> None:
    torch.manual_seed(41)
    model = LiteralVocabModel(input_vocab_tensor_dim=3, output_vocab_tensor_dim=5, hidden_dim=7).eval()
    output_vocab = torch.randn(4, 5)
    cache = model.prepare_output_vocab(output_vocab, detach=True)

    first = model(torch.tensor([[0]]), torch.randn(3, 3), cache)
    second = model(torch.tensor([[1]]), torch.randn(3, 3), cache)

    assert not cache.keys.requires_grad
    assert first.shape == second.shape == (1, 1, 4)
