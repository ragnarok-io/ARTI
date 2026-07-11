"""Lossless glyph visual fields for Pulse fragment formation."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass(frozen=True)
class VisualFieldOutput:
    """Patch fragments and structural metadata produced by :class:`VisualField`.

    ``pixels`` contains untouched bitmap samples. ``coord`` stores normalized
    absolute patch bounds plus the caller-supplied field id. ``fragments`` is
    the model-facing tensor and includes ``coord`` when positional structure is
    enabled. All tensors use the Pulse-compatible ``[B, N, D]`` convention.
    """

    fragments: Tensor
    pixels: Tensor
    coord: Tensor
    mask: Tensor
    source_size: tuple[int, int]
    windows: tuple[tuple[int, int, int, int], ...]
    include_position: bool


class VisualField(nn.Module):
    """Turn a rigid glyph bitmap window into Pulse-compatible fragments.

    The layer never resizes or interpolates glyph pixels. Edge windows are
    padded on the bottom/right only so every source pixel remains represented.
    ``stride`` may overlap patches but may not exceed ``patch_size`` because a
    larger stride would create blind gaps in the visual field.
    """

    def __init__(
        self,
        patch_size: int | tuple[int, int] = (4, 4),
        *,
        stride: int | tuple[int, int] | None = None,
        include_position: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = _pair(patch_size, "patch_size")
        self.stride = self.patch_size if stride is None else _pair(stride, "stride")
        if self.stride[0] > self.patch_size[0] or self.stride[1] > self.patch_size[1]:
            raise ValueError("stride may not exceed patch_size because VisualField must not leave pixel gaps")
        self.include_position = bool(include_position)

    def forward(
        self,
        glyph: Tensor,
        *,
        window: tuple[int, int, int, int] | None = None,
        field_id: float = 0.0,
    ) -> VisualFieldOutput:
        if glyph.ndim != 4:
            raise ValueError("glyph must have shape [B, C, H, W]")
        if not glyph.is_floating_point():
            raise TypeError("glyph must be a floating-point Tensor")
        batch, channels, source_height, source_width = glyph.shape
        if source_height <= 0 or source_width <= 0:
            raise ValueError("glyph height and width must be positive")
        resolved = _validate_window(window, source_height, source_width)
        top, left, height, width = resolved
        view = glyph[:, :, top : top + height, left : left + width]
        patch_height, patch_width = self.patch_size
        stride_height, stride_width = self.stride
        rows = 1 if height <= patch_height else math.ceil((height - patch_height) / stride_height) + 1
        columns = 1 if width <= patch_width else math.ceil((width - patch_width) / stride_width) + 1
        padded_height = (rows - 1) * stride_height + patch_height
        padded_width = (columns - 1) * stride_width + patch_width
        padded = F.pad(view, (0, padded_width - width, 0, padded_height - height))
        pixels = F.unfold(padded, kernel_size=self.patch_size, stride=self.stride).transpose(1, 2)

        row_origins = torch.arange(rows, device=glyph.device, dtype=glyph.dtype) * stride_height + top
        column_origins = torch.arange(columns, device=glyph.device, dtype=glyph.dtype) * stride_width + left
        grid_y, grid_x = torch.meshgrid(row_origins, column_origins, indexing="ij")
        bottom = (grid_y + patch_height).clamp(max=top + height)
        right = (grid_x + patch_width).clamp(max=left + width)
        coord = torch.stack(
            [
                grid_y / source_height,
                grid_x / source_width,
                bottom / source_height,
                right / source_width,
                torch.full_like(grid_y, float(field_id)),
            ],
            dim=-1,
        ).reshape(1, rows * columns, 5).expand(batch, -1, -1)
        fragments = torch.cat([pixels, coord], dim=-1) if self.include_position else pixels
        mask = torch.ones(batch, rows * columns, device=glyph.device, dtype=torch.bool)
        return VisualFieldOutput(
            fragments=fragments,
            pixels=pixels,
            coord=coord,
            mask=mask,
            source_size=(source_height, source_width),
            windows=(resolved,),
            include_position=self.include_position,
        )

    def extra_repr(self) -> str:
        return f"patch_size={self.patch_size}, stride={self.stride}, include_position={self.include_position}"


def concat_visual_fields(*fields: VisualFieldOutput) -> VisualFieldOutput:
    """Concatenate visual fields along the Pulse fragment axis.

    Pixel values are not averaged or resampled. Fields must describe the same
    batch/source geometry and use compatible patch dimensions.
    """

    if not fields:
        raise ValueError("concat_visual_fields requires at least one field")
    first = fields[0]
    for field in fields[1:]:
        if field.source_size != first.source_size:
            raise ValueError("visual fields must have matching source_size")
        if field.include_position != first.include_position:
            raise ValueError("visual fields must agree on include_position")
        if field.fragments.shape[0] != first.fragments.shape[0]:
            raise ValueError("visual fields must have matching batch size")
        if field.fragments.shape[-1] != first.fragments.shape[-1] or field.pixels.shape[-1] != first.pixels.shape[-1]:
            raise ValueError("visual fields must have matching fragment dimensions")
        if field.fragments.device != first.fragments.device or field.fragments.dtype != first.fragments.dtype:
            raise ValueError("visual fields must have matching device and dtype")
    return VisualFieldOutput(
        fragments=torch.cat([field.fragments for field in fields], dim=1),
        pixels=torch.cat([field.pixels for field in fields], dim=1),
        coord=torch.cat([field.coord for field in fields], dim=1),
        mask=torch.cat([field.mask for field in fields], dim=1),
        source_size=first.source_size,
        windows=tuple(window for field in fields for window in field.windows),
        include_position=first.include_position,
    )


def _pair(value: int | tuple[int, int], name: str) -> tuple[int, int]:
    pair = (value, value) if isinstance(value, int) else value
    if len(pair) != 2 or pair[0] <= 0 or pair[1] <= 0:
        raise ValueError(f"{name} must contain two positive integers")
    return int(pair[0]), int(pair[1])


def _validate_window(
    window: tuple[int, int, int, int] | None,
    source_height: int,
    source_width: int,
) -> tuple[int, int, int, int]:
    resolved = (0, 0, source_height, source_width) if window is None else window
    if len(resolved) != 4:
        raise ValueError("window must be (top, left, height, width)")
    top, left, height, width = (int(value) for value in resolved)
    if top < 0 or left < 0 or height <= 0 or width <= 0:
        raise ValueError("window coordinates must be non-negative with positive height and width")
    if top + height > source_height or left + width > source_width:
        raise ValueError("window must stay inside the glyph source")
    return top, left, height, width


__all__ = ["VisualField", "VisualFieldOutput", "concat_visual_fields"]
