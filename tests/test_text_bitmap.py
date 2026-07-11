from __future__ import annotations

import pytest
import torch

from arti import BitmapTextConfig, BitmapTextRenderer, assert_bitmap_vocab_distinct, bitmap_vocab_report, render_text_bitmap, render_text_vocab


def test_render_text_bitmap_returns_fixed_tensor_shape() -> None:
    bitmap = render_text_bitmap("strawberry", height=14, width=80)

    assert bitmap.shape == (1, 14, 80)
    assert bitmap.dtype == torch.float32
    assert float(bitmap.max()) > 0.0


def test_text_vocab_renders_distinct_repeated_letter_words() -> None:
    vocab = render_text_vocab(["strawbery", "strawberry"], height=14, width=80)

    assert vocab.shape == (2, 1, 14, 80)
    assert not torch.allclose(vocab[0], vocab[1])


def test_renderer_vocab_uses_shared_config() -> None:
    renderer = BitmapTextRenderer(BitmapTextConfig(height=10, width=32))

    vocab = renderer.vocab(["r", "rr"])

    assert vocab.shape == (2, 1, 10, 32)
    assert not torch.allclose(vocab[0], vocab[1])
    assert float(vocab.sum()) > 0.0


def test_font_path_requires_existing_font_and_pillow() -> None:
    with pytest.raises((FileNotFoundError, RuntimeError)):
        BitmapTextRenderer(font_path="missing-font-file.ttf")


def test_bitmap_vocab_report_detects_entropy_and_distinctness() -> None:
    vocab = render_text_vocab(["strawbery", "strawberry", "strawberrry"], height=14, width=80)

    report = bitmap_vocab_report(vocab)

    assert report.count == 3
    assert report.unique_count == 3
    assert report.collision_count == 0
    assert report.min_pairwise_distance > 0.0
    assert report.entropy_bits > 0.0
    assert_bitmap_vocab_distinct(vocab, min_distance=0.0, min_entropy_bits=0.1)


def test_micro_punctuation_and_narrow_glyphs_are_distinct() -> None:
    vocab = render_text_vocab([".", ",", ":", ";", "i", "l", "1", "r", "rr"], height=14, width=96)

    report = bitmap_vocab_report(vocab)

    assert report.collision_count == 0
    assert report.min_pairwise_distance > 0.01
    assert not torch.allclose(vocab[0], vocab[1])


def test_bitmap_vocab_distinctness_rejects_collisions() -> None:
    vocab = render_text_vocab(["same", "same"], height=14, width=80)

    report = bitmap_vocab_report(vocab)

    assert report.unique_count == 1
    assert report.collision_count == 1
    with pytest.raises(ValueError, match="collisions"):
        assert_bitmap_vocab_distinct(vocab)
