"""Rigid text bitmap tensors for runtime vocab experiments.

The fallback renderer is deliberately small and dependency-free. If Pillow is
installed, ``BitmapTextRenderer(font_path=...)`` can load real font files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F
from torch import Tensor


_GLYPHS_5X7: dict[str, tuple[str, ...]] = {
    " ": ("00000", "00000", "00000", "00000", "00000", "00000", "00000"),
    ".": ("00000", "00000", "00000", "00000", "00000", "01100", "01100"),
    ",": ("00000", "00000", "00000", "00000", "00000", "01100", "00100"),
    ":": ("00000", "01100", "01100", "00000", "01100", "01100", "00000"),
    ";": ("00000", "01100", "01100", "00000", "01100", "00100", "01000"),
    "'": ("00100", "00100", "01000", "00000", "00000", "00000", "00000"),
    '"': ("01010", "01010", "01010", "00000", "00000", "00000", "00000"),
    "-": ("00000", "00000", "00000", "11110", "00000", "00000", "00000"),
    "_": ("00000", "00000", "00000", "00000", "00000", "00000", "11111"),
    "!": ("00100", "00100", "00100", "00100", "00100", "00000", "00100"),
    "?": ("01110", "10001", "00001", "00010", "00100", "00000", "00100"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "10001", "01110"),
    "6": ("00110", "01000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00010", "01100"),
    "a": ("00000", "01110", "00001", "01111", "10001", "10011", "01101"),
    "b": ("10000", "10000", "10110", "11001", "10001", "10001", "11110"),
    "c": ("00000", "01110", "10001", "10000", "10000", "10001", "01110"),
    "d": ("00001", "00001", "01101", "10011", "10001", "10001", "01111"),
    "e": ("00000", "01110", "10001", "11111", "10000", "10001", "01110"),
    "f": ("00110", "01001", "01000", "11100", "01000", "01000", "01000"),
    "g": ("00000", "01111", "10001", "10001", "01111", "00001", "01110"),
    "h": ("10000", "10000", "10110", "11001", "10001", "10001", "10001"),
    "i": ("00100", "00000", "01100", "00100", "00100", "00100", "01110"),
    "j": ("00010", "00000", "00110", "00010", "00010", "10010", "01100"),
    "k": ("10000", "10010", "10100", "11000", "10100", "10010", "10001"),
    "l": ("01100", "00100", "00100", "00100", "00100", "00100", "01110"),
    "m": ("00000", "11010", "10101", "10101", "10101", "10101", "10101"),
    "n": ("00000", "10110", "11001", "10001", "10001", "10001", "10001"),
    "o": ("00000", "01110", "10001", "10001", "10001", "10001", "01110"),
    "p": ("00000", "11110", "10001", "10001", "11110", "10000", "10000"),
    "q": ("00000", "01111", "10001", "10001", "01111", "00001", "00001"),
    "r": ("00000", "10110", "11001", "10000", "10000", "10000", "10000"),
    "s": ("00000", "01111", "10000", "01110", "00001", "10001", "01110"),
    "t": ("01000", "01000", "11100", "01000", "01000", "01001", "00110"),
    "u": ("00000", "10001", "10001", "10001", "10001", "10011", "01101"),
    "v": ("00000", "10001", "10001", "10001", "10001", "01010", "00100"),
    "w": ("00000", "10001", "10001", "10101", "10101", "10101", "01010"),
    "x": ("00000", "10001", "01010", "00100", "01010", "10001", "10001"),
    "y": ("00000", "10001", "10001", "10001", "01111", "00001", "01110"),
    "z": ("00000", "11111", "00010", "00100", "01000", "10000", "11111"),
}


@dataclass(frozen=True)
class BitmapTextConfig:
    """Configuration for fixed-size text bitmap rendering."""

    height: int = 16
    width: int = 96
    font_size: int = 16
    normalize: bool = True


@dataclass(frozen=True)
class BitmapVocabReport:
    """Distinctness and entropy report for a rendered bitmap vocabulary."""

    count: int
    unique_count: int
    collision_count: int
    min_pairwise_distance: float
    entropy_bits: float
    mean_active_fraction: float
    collisions: tuple[tuple[int, int], ...]


class BitmapTextRenderer:
    """Render text into rigid ``torch.Tensor`` bitmaps.

    If ``font_path`` is provided, Pillow is required and the font file is loaded
    with ``ImageFont.truetype``. Without ``font_path``, ARTI uses a small
    dependency-free bitmap font for ASCII development and tests.
    """

    def __init__(self, config: BitmapTextConfig | None = None, *, font_path: str | Path | None = None) -> None:
        self.config = BitmapTextConfig() if config is None else config
        self.font_path = None if font_path is None else Path(font_path)
        self._font = None
        if self.font_path is not None:
            self._font = _load_pillow_font(self.font_path, self.config.font_size)

    def render(self, text: str) -> Tensor:
        image = self._render_pillow(text) if self._font is not None else _render_fallback(text)
        image = image.to(dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(image, size=(self.config.height, self.config.width), mode="bilinear", align_corners=False)
        out = resized.squeeze(0)
        if self.config.normalize:
            out = out.clamp(0.0, 1.0)
        return out

    def vocab(self, texts: Iterable[str]) -> Tensor:
        return torch.stack([self.render(text) for text in texts], dim=0)

    def _render_pillow(self, text: str) -> Tensor:
        from PIL import Image, ImageDraw

        image = Image.new("L", (self.config.width, self.config.height), color=0)
        draw = ImageDraw.Draw(image)
        draw.text((0, 0), text, fill=255, font=self._font)
        data = torch.tensor(list(image.getdata()), dtype=torch.float32).reshape(self.config.height, self.config.width)
        return data / 255.0


def render_text_bitmap(text: str, *, height: int = 16, width: int = 96, font_path: str | Path | None = None) -> Tensor:
    """Convenience wrapper returning a ``[1, H, W]`` bitmap tensor."""

    return BitmapTextRenderer(BitmapTextConfig(height=height, width=width), font_path=font_path).render(text)


def render_text_vocab(texts: Iterable[str], *, height: int = 16, width: int = 96, font_path: str | Path | None = None) -> Tensor:
    """Render a runtime vocab as ``[K, 1, H, W]`` rigid tensors."""

    return BitmapTextRenderer(BitmapTextConfig(height=height, width=width), font_path=font_path).vocab(texts)


def bitmap_vocab_report(vocab_tensor: Tensor, *, atol: float = 0.0, bins: int = 256) -> BitmapVocabReport:
    """Report whether bitmap vocab items preserve distinguishable information.

    ``atol`` controls the collision tolerance. With the default ``0.0``, only
    exactly equal flattened tensors collide. Use a small positive value when the
    renderer introduces subpixel floating point noise and near-identical tensors
    should be treated as unsafe.
    """

    flat = _flatten_vocab(vocab_tensor)
    count = flat.shape[0]
    if count == 0:
        return BitmapVocabReport(0, 0, 0, 0.0, 0.0, 0.0, ())
    distances = torch.cdist(flat, flat, p=2)
    if count > 1:
        upper = torch.triu(torch.ones(count, count, device=flat.device, dtype=torch.bool), diagonal=1)
        pair_values = distances[upper]
        min_distance = float(pair_values.min().item())
    else:
        min_distance = float("inf")
    collision_pairs = []
    parent = list(range(count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for i in range(count):
        for j in range(i + 1, count):
            if float(distances[i, j].item()) <= atol:
                collision_pairs.append((i, j))
                union(i, j)
    entropy = _quantized_entropy_bits(flat, bins=bins)
    active = (flat > 0).to(torch.float32).mean(dim=1)
    unique_count = len({find(index) for index in range(count)})
    return BitmapVocabReport(
        count=count,
        unique_count=unique_count,
        collision_count=len(collision_pairs),
        min_pairwise_distance=min_distance,
        entropy_bits=entropy,
        mean_active_fraction=float(active.mean().item()),
        collisions=tuple(collision_pairs),
    )


def assert_bitmap_vocab_distinct(vocab_tensor: Tensor, *, min_distance: float = 0.0, min_entropy_bits: float = 0.0, atol: float = 0.0) -> BitmapVocabReport:
    """Validate that a bitmap vocab has no collisions and enough entropy."""

    report = bitmap_vocab_report(vocab_tensor, atol=atol)
    if report.collision_count:
        raise ValueError(f"bitmap vocab has {report.collision_count} collisions: {report.collisions[:5]}")
    if report.count > 1 and report.min_pairwise_distance <= min_distance:
        raise ValueError(f"bitmap vocab min distance {report.min_pairwise_distance:.8f} <= {min_distance:.8f}")
    if report.entropy_bits < min_entropy_bits:
        raise ValueError(f"bitmap vocab entropy {report.entropy_bits:.4f} < {min_entropy_bits:.4f}")
    return report


def _load_pillow_font(font_path: Path, font_size: int):
    try:
        from PIL import ImageFont
    except ModuleNotFoundError as exc:
        raise RuntimeError("font_path rendering requires Pillow. Install with `uv sync --extra font`.") from exc
    if not font_path.exists():
        raise FileNotFoundError(font_path)
    return ImageFont.truetype(str(font_path), font_size)


def _render_fallback(text: str) -> Tensor:
    glyph_height = 7
    glyph_width = 5
    spacing = 1
    canvas = torch.zeros(glyph_height, max(1, len(text) * (glyph_width + spacing) - spacing), dtype=torch.float32)
    for index, char in enumerate(text.lower()):
        glyph = _GLYPHS_5X7.get(char, _GLYPHS_5X7["?"])
        x0 = index * (glyph_width + spacing)
        for y, row in enumerate(glyph):
            for x, value in enumerate(row):
                canvas[y, x0 + x] = 1.0 if value == "1" else 0.0
    return canvas


def _flatten_vocab(vocab_tensor: Tensor) -> Tensor:
    if vocab_tensor.ndim < 2:
        raise ValueError("vocab_tensor must have shape [K, ...]")
    return vocab_tensor.detach().to(dtype=torch.float32).flatten(start_dim=1)


def _quantized_entropy_bits(flat: Tensor, *, bins: int) -> float:
    if bins <= 1:
        raise ValueError("bins must be greater than 1")
    values = flat.clamp(0.0, 1.0)
    quantized = torch.clamp((values * (bins - 1)).round().to(torch.long), 0, bins - 1)
    counts = torch.bincount(quantized.reshape(-1), minlength=bins).to(torch.float32)
    probs = counts[counts > 0] / counts.sum().clamp_min(1.0)
    return float((-(probs * torch.log2(probs)).sum()).item())
