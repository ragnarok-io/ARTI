"""Unicode-aware text tensor layout alpha API.

This module preserves text control information that a bitmap-only renderer can
hide. It is a small dependency-free fallback, not a HarfBuzz/Pango replacement.
Future shaping backends can fill the same contract with real glyph ids and
font-specific positions.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import unicodedata

import torch
from torch import Tensor

from .pulse import PulseOutput, pulse_compress
from .text_bitmap import _GLYPHS_5X7


class TextControlKind(IntEnum):
    """Stable control-channel indices for text tensor layouts."""

    VISIBLE = 0
    SPACE = 1
    TAB = 2
    NEWLINE = 3
    PAGE_BREAK = 4
    ZERO_WIDTH_SPACE = 5
    ZERO_WIDTH_JOINER = 6
    ZERO_WIDTH_NON_JOINER = 7
    WORD_JOINER = 8
    BOM = 9
    COMBINING_MARK = 10
    FORMAT = 11
    CONTROL = 12
    REPLACEMENT = 13


TEXT_CONTROL_CHANNELS: tuple[str, ...] = tuple(kind.name.lower() for kind in TextControlKind)
TEXT_IDENTITY_MODES: tuple[str, ...] = ("glyph_only", "glyph_plus_codepoint_aux", "control_codepoint_aux")


@dataclass(frozen=True)
class TextTensorConfig:
    """Configuration for text layout tensorization."""

    normalization: str = "raw"
    identity_mode: str = "glyph_only"
    glyph_height: int = 7
    glyph_width: int = 5
    codepoint_aux_dim: int = 8
    tab_width: int = 4
    line_height: float = 1.0
    visible_advance: float = 1.0
    space_advance: float = 1.0
    replacement_codepoint: int = 0xFFFD

    def normalized_text(self, text: str) -> str:
        if self.normalization == "raw":
            return text
        if self.normalization in {"NFC", "NFD"}:
            return unicodedata.normalize(self.normalization, text)
        raise ValueError("normalization must be 'raw', 'NFC', or 'NFD'")

    def validate_identity_mode(self) -> None:
        if self.identity_mode not in TEXT_IDENTITY_MODES:
            raise ValueError(f"identity_mode must be one of {TEXT_IDENTITY_MODES}")


@dataclass(frozen=True)
class TextTensorLayout:
    """Tensorized physical text layout.

    ``sequence`` is suitable as a runtime-vocab tensor source. It concatenates
    fallback glyph appearance features, control channels, and normalized layout
    coordinates for every input position. Codepoints are preserved as metadata
    but are not part of the default model-facing sequence.
    """

    text: str
    normalized_text: str
    codepoints: Tensor
    grapheme_ids: Tensor
    cluster_ids: Tensor
    control: Tensor
    visible_mask: Tensor
    coord: Tensor
    sequence: Tensor
    metadata: dict[str, object]

    def to_sequence_tensor(self) -> Tensor:
        return self.sequence

    def to_pulse_tensor(self, pulse_ids: Tensor | None = None, *, pulse_count: int | None = None, token_weight: Tensor | None = None) -> PulseOutput:
        if pulse_ids is None:
            pulse_ids = torch.arange(self.sequence.shape[0], device=self.sequence.device, dtype=torch.long)
        if pulse_ids.ndim != 1 or pulse_ids.shape[0] != self.sequence.shape[0]:
            raise ValueError("pulse_ids must have shape [L]")
        return pulse_compress(
            self.sequence.unsqueeze(0),
            pulse_ids.unsqueeze(0),
            mask=torch.ones(1, self.sequence.shape[0], device=self.sequence.device, dtype=torch.bool),
            token_weight=None if token_weight is None else token_weight.unsqueeze(0),
            pulse_count=pulse_count,
        )

    def pooled_tensor(self) -> Tensor:
        if self.sequence.numel() == 0:
            return self.sequence.new_zeros(self.sequence.shape[-1])
        return self.sequence.mean(dim=0)


class TextTensorRenderer:
    """Dependency-free text tensor renderer for ARTI runtime vocab experiments."""

    def __init__(self, config: TextTensorConfig | None = None) -> None:
        self.config = TextTensorConfig() if config is None else config

    def layout(self, text: str) -> TextTensorLayout:
        self.config.validate_identity_mode()
        normalized = self.config.normalized_text(text)
        chars = list(normalized)
        length = len(chars)
        codepoints = torch.tensor([ord(char) for char in chars], dtype=torch.long)
        grapheme_ids = torch.empty(length, dtype=torch.long)
        cluster_ids = torch.empty(length, dtype=torch.long)
        control = torch.zeros(length, len(TextControlKind), dtype=torch.float32)
        visible_mask = torch.zeros(length, dtype=torch.bool)
        coord = torch.zeros(length, 5, dtype=torch.float32)
        glyph = torch.zeros(length, self.config.glyph_height * self.config.glyph_width, dtype=torch.float32)
        codepoint_aux = torch.zeros(length, self.config.codepoint_aux_dim, dtype=torch.float32)

        page = 0
        line = 0
        x = 0.0
        grapheme_id = -1
        for index, char in enumerate(chars):
            kind = _classify_control(char)
            control[index, kind] = 1.0
            glyph[index] = _glyph_features(char, kind, height=self.config.glyph_height, width=self.config.glyph_width)
            codepoint_aux[index] = _codepoint_aux_features(ord(char), self.config.codepoint_aux_dim)
            is_combining = kind == TextControlKind.COMBINING_MARK
            is_visible = kind in {TextControlKind.VISIBLE, TextControlKind.SPACE}
            if char == "\t":
                advance = float(self.config.tab_width)
            elif kind == TextControlKind.SPACE:
                advance = float(self.config.space_advance)
            elif kind == TextControlKind.VISIBLE:
                advance = float(self.config.visible_advance)
            else:
                advance = 0.0

            if not is_combining:
                grapheme_id += 1
            grapheme_ids[index] = max(grapheme_id, 0)
            cluster_ids[index] = grapheme_ids[index]
            visible_mask[index] = is_visible
            coord[index] = torch.tensor([float(page), float(line), x, float(line) * self.config.line_height, advance])

            if kind == TextControlKind.NEWLINE:
                line += 1
                x = 0.0
            elif kind == TextControlKind.PAGE_BREAK:
                page += 1
                line = 0
                x = 0.0
            else:
                x += advance

        sequence_parts = [glyph, control, coord]
        sequence_fields: tuple[str, ...] = ("glyph_appearance", "control", "coord")
        if self.config.identity_mode == "glyph_plus_codepoint_aux":
            sequence_parts.append(codepoint_aux)
            sequence_fields = (*sequence_fields, "codepoint_aux")
        elif self.config.identity_mode == "control_codepoint_aux":
            aux_mask = _codepoint_aux_mask(control, glyph, height=self.config.glyph_height, width=self.config.glyph_width)
            sequence_parts.append(codepoint_aux * aux_mask.unsqueeze(-1))
            sequence_fields = (*sequence_fields, "codepoint_aux_control_only")
        sequence = torch.cat(sequence_parts, dim=-1)
        metadata: dict[str, object] = {
            "normalization": self.config.normalization,
            "identity_mode": self.config.identity_mode,
            "control_channels": TEXT_CONTROL_CHANNELS,
            "coord_fields": ("page", "line", "x", "y", "advance"),
            "sequence_fields": sequence_fields,
            "backend": "fallback-unicode-layout",
            "shaping_backend": None,
        }
        return TextTensorLayout(
            text=text,
            normalized_text=normalized,
            codepoints=codepoints,
            grapheme_ids=grapheme_ids,
            cluster_ids=cluster_ids,
            control=control,
            visible_mask=visible_mask,
            coord=coord,
            sequence=sequence,
            metadata=metadata,
        )

    def tensor(self, text: str) -> Tensor:
        return self.layout(text).to_sequence_tensor()


def render_text_layout(text: str, *, config: TextTensorConfig | None = None, normalization: str | None = None) -> TextTensorLayout:
    """Render text into a control-aware layout object."""

    resolved = config
    if normalization is not None:
        base = TextTensorConfig() if config is None else config
        resolved = TextTensorConfig(
            normalization=normalization,
            identity_mode=base.identity_mode,
            glyph_height=base.glyph_height,
            glyph_width=base.glyph_width,
            codepoint_aux_dim=base.codepoint_aux_dim,
            tab_width=base.tab_width,
            line_height=base.line_height,
            visible_advance=base.visible_advance,
            space_advance=base.space_advance,
            replacement_codepoint=base.replacement_codepoint,
        )
    return TextTensorRenderer(resolved).layout(text)


def render_text_tensor(text: str, *, config: TextTensorConfig | None = None, normalization: str | None = None) -> Tensor:
    """Render text into a ``[L, D]`` sequence tensor."""

    return render_text_layout(text, config=config, normalization=normalization).to_sequence_tensor()


def _classify_control(char: str) -> TextControlKind:
    if char == " ":
        return TextControlKind.SPACE
    if char == "\t":
        return TextControlKind.TAB
    if char in {"\n", "\r"}:
        return TextControlKind.NEWLINE
    if char == "\f":
        return TextControlKind.PAGE_BREAK
    if char == "\u200b":
        return TextControlKind.ZERO_WIDTH_SPACE
    if char == "\u200d":
        return TextControlKind.ZERO_WIDTH_JOINER
    if char == "\u200c":
        return TextControlKind.ZERO_WIDTH_NON_JOINER
    if char == "\u2060":
        return TextControlKind.WORD_JOINER
    if char == "\ufeff":
        return TextControlKind.BOM
    category = unicodedata.category(char)
    if category.startswith("M"):
        return TextControlKind.COMBINING_MARK
    if category == "Cf":
        return TextControlKind.FORMAT
    if category.startswith("C"):
        return TextControlKind.CONTROL
    if ord(char) == 0xFFFD:
        return TextControlKind.REPLACEMENT
    return TextControlKind.VISIBLE


def _glyph_features(char: str, kind: TextControlKind, *, height: int, width: int) -> Tensor:
    if height <= 0 or width <= 0:
        raise ValueError("glyph_height and glyph_width must be positive")
    if kind not in {TextControlKind.VISIBLE, TextControlKind.SPACE, TextControlKind.REPLACEMENT}:
        return torch.zeros(height * width, dtype=torch.float32)
    rows = _GLYPHS_5X7.get(char.lower(), _GLYPHS_5X7["?"])
    base = torch.tensor([[1.0 if value == "1" else 0.0 for value in row] for row in rows], dtype=torch.float32)
    if (height, width) != base.shape:
        base = torch.nn.functional.interpolate(base.unsqueeze(0).unsqueeze(0), size=(height, width), mode="nearest").squeeze(0).squeeze(0)
    return base.flatten()


def _codepoint_aux_features(codepoint: int, dim: int) -> Tensor:
    if dim <= 0:
        raise ValueError("codepoint_aux_dim must be positive")
    x = torch.tensor(float(codepoint) / 0x10FFFF, dtype=torch.float32)
    freqs = torch.arange(1, dim // 2 + 1, dtype=torch.float32)
    features = torch.cat([torch.sin(x * freqs * torch.pi), torch.cos(x * freqs * torch.pi)], dim=0)
    if features.numel() < dim:
        features = torch.cat([features, torch.tensor([1.0 if codepoint % 2 else -1.0], dtype=torch.float32)], dim=0)
    return features[:dim]


def _codepoint_aux_mask(control: Tensor, glyph: Tensor, *, height: int, width: int) -> Tensor:
    visible = control[:, int(TextControlKind.VISIBLE)] > 0
    space = control[:, int(TextControlKind.SPACE)] > 0
    replacement = control[:, int(TextControlKind.REPLACEMENT)] > 0
    fallback_visible = visible & _looks_like_fallback_question(glyph, height=height, width=width)
    invisible_or_control = ~(visible | space)
    return invisible_or_control | replacement | fallback_visible


def _looks_like_fallback_question(glyph: Tensor, *, height: int, width: int) -> Tensor:
    question = _glyph_features("?", TextControlKind.VISIBLE, height=height, width=width)
    if glyph.shape[-1] != question.numel():
        return torch.zeros(glyph.shape[0], dtype=torch.bool)
    return torch.isclose(glyph, question.unsqueeze(0)).all(dim=-1)
