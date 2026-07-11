from __future__ import annotations

import torch

from arti import TextControlKind, TextTensorConfig, TextTensorRenderer, render_text_layout, render_text_tensor


def channel(layout, kind: TextControlKind) -> torch.Tensor:
    return layout.control[:, int(kind)]


def test_text_tensor_keeps_zero_width_as_control_channel() -> None:
    plain = render_text_layout("ab")
    zero_width = render_text_layout("a\u200bb")

    assert plain.sequence.shape[0] == 2
    assert zero_width.sequence.shape[0] == 3
    assert int(channel(zero_width, TextControlKind.ZERO_WIDTH_SPACE).sum().item()) == 1
    assert zero_width.visible_mask.tolist() == [True, False, True]
    assert not torch.allclose(plain.pooled_tensor(), zero_width.pooled_tensor())


def test_space_newline_and_page_break_do_not_collapse() -> None:
    space = render_text_layout("a b")
    newline = render_text_layout("a\nb")
    page = render_text_layout("a\fb")

    assert int(channel(space, TextControlKind.SPACE).sum().item()) == 1
    assert int(channel(newline, TextControlKind.NEWLINE).sum().item()) == 1
    assert int(channel(page, TextControlKind.PAGE_BREAK).sum().item()) == 1
    assert newline.coord[2, 1].item() == 1.0
    assert newline.coord[2, 2].item() == 0.0
    assert page.coord[2, 0].item() == 1.0
    assert page.coord[2, 1].item() == 0.0
    assert not torch.allclose(space.pooled_tensor(), newline.pooled_tensor())
    assert not torch.allclose(newline.pooled_tensor(), page.pooled_tensor())


def test_repeated_r_variants_remain_distinct() -> None:
    variants = [render_text_layout(text).pooled_tensor() for text in ("r", "rr", "r\u200br", "r\nr")]

    for left in range(len(variants)):
        for right in range(left + 1, len(variants)):
            assert not torch.allclose(variants[left], variants[right])


def test_normalization_raw_nfc_nfd_controls_combining_identity() -> None:
    composed_raw = render_text_layout("\u00e9", normalization="raw")
    decomposed_raw = render_text_layout("e\u0301", normalization="raw")
    composed_nfc = render_text_layout("\u00e9", normalization="NFC")
    decomposed_nfc = render_text_layout("e\u0301", normalization="NFC")
    composed_nfd = render_text_layout("\u00e9", normalization="NFD")
    decomposed_nfd = render_text_layout("e\u0301", normalization="NFD")

    assert composed_raw.codepoints.tolist() != decomposed_raw.codepoints.tolist()
    assert composed_nfc.codepoints.tolist() == decomposed_nfc.codepoints.tolist()
    assert composed_nfd.codepoints.tolist() == decomposed_nfd.codepoints.tolist()
    assert int(channel(decomposed_raw, TextControlKind.COMBINING_MARK).sum().item()) == 1
    assert int(channel(composed_nfd, TextControlKind.COMBINING_MARK).sum().item()) == 1


def test_tab_zwj_zwnj_bom_are_visible_in_control_tensor() -> None:
    layout = render_text_layout("\ufeffa\t\u200d\u200cb")

    assert int(channel(layout, TextControlKind.BOM).sum().item()) == 1
    assert int(channel(layout, TextControlKind.TAB).sum().item()) == 1
    assert int(channel(layout, TextControlKind.ZERO_WIDTH_JOINER).sum().item()) == 1
    assert int(channel(layout, TextControlKind.ZERO_WIDTH_NON_JOINER).sum().item()) == 1
    assert layout.visible_mask.tolist() == [False, True, False, False, False, True]


def test_text_tensor_sequence_and_pulse_views_are_runtime_vocab_ready() -> None:
    config = TextTensorConfig(glyph_height=5, glyph_width=5)
    renderer = TextTensorRenderer(config)
    layout = renderer.layout("r\u200br")

    sequence = layout.to_sequence_tensor()
    pulse = layout.to_pulse_tensor(torch.tensor([0, 0, 1]), pulse_count=2)
    tensor = render_text_tensor("r\u200br", config=config)

    assert sequence.shape == tensor.shape
    assert pulse.pulse.shape == (1, 2, sequence.shape[-1])
    assert pulse.mask.tolist() == [[True, True]]


def test_visible_sequence_uses_glyph_appearance_not_codepoint_cipher() -> None:
    left = render_text_layout("\u00e9")
    right = render_text_layout("\u00f8")

    assert left.codepoints.tolist() != right.codepoints.tolist()
    assert torch.allclose(left.to_sequence_tensor(), right.to_sequence_tensor())
    assert left.metadata["sequence_fields"] == ("glyph_appearance", "control", "coord")
    assert left.metadata["identity_mode"] == "glyph_only"


def test_known_visible_glyphs_are_distinguished_by_shape() -> None:
    left = render_text_layout("a")
    right = render_text_layout("b")

    assert left.codepoints.tolist() != right.codepoints.tolist()
    assert not torch.allclose(left.to_sequence_tensor(), right.to_sequence_tensor())


def test_glyph_plus_codepoint_aux_can_preserve_confusable_or_fallback_differences() -> None:
    config = TextTensorConfig(identity_mode="glyph_plus_codepoint_aux")
    left = render_text_layout("\u00e9", config=config)
    right = render_text_layout("\u00f8", config=config)

    assert left.codepoints.tolist() != right.codepoints.tolist()
    assert not torch.allclose(left.to_sequence_tensor(), right.to_sequence_tensor())
    assert left.metadata["sequence_fields"] == ("glyph_appearance", "control", "coord", "codepoint_aux")


def test_control_codepoint_aux_preserves_invisible_and_fallback_identity_only() -> None:
    config = TextTensorConfig(identity_mode="control_codepoint_aux")
    known_left = render_text_layout("a", config=config)
    known_right = render_text_layout("b", config=config)
    fallback_left = render_text_layout("\u00e9", config=config)
    fallback_right = render_text_layout("\u00f8", config=config)
    zero_width_space = render_text_layout("a\u200bb", config=config)
    zero_width_joiner = render_text_layout("a\u200db", config=config)

    assert not torch.allclose(known_left.to_sequence_tensor(), known_right.to_sequence_tensor())
    assert not torch.allclose(fallback_left.to_sequence_tensor(), fallback_right.to_sequence_tensor())
    assert not torch.allclose(zero_width_space.to_sequence_tensor(), zero_width_joiner.to_sequence_tensor())
    assert fallback_left.metadata["sequence_fields"] == ("glyph_appearance", "control", "coord", "codepoint_aux_control_only")


def test_control_aux_keeps_visible_glyph_first_for_known_micro_chars() -> None:
    config = TextTensorConfig(identity_mode="control_codepoint_aux")
    period = render_text_layout(".", config=config)
    comma = render_text_layout(",", config=config)
    newline = render_text_layout("\n", config=config)
    zero_width = render_text_layout("\u200b", config=config)

    assert period.metadata["sequence_fields"] == ("glyph_appearance", "control", "coord", "codepoint_aux_control_only")
    assert not torch.allclose(period.to_sequence_tensor(), comma.to_sequence_tensor())
    assert not torch.allclose(newline.to_sequence_tensor(), zero_width.to_sequence_tensor())
    assert period.visible_mask.tolist() == [True]


def test_invalid_identity_mode_is_rejected() -> None:
    config = TextTensorConfig(identity_mode="codepoint_cipher")

    try:
        render_text_layout("a", config=config)
    except ValueError as exc:
        assert "identity_mode" in str(exc)
    else:
        raise AssertionError("invalid identity_mode should fail")
