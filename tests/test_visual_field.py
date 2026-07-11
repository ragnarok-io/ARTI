from __future__ import annotations

import pytest
import torch

import arti
import arti.nn as arti_nn
import arti.torch as arti_torch


def test_visual_field_patchifies_without_resizing_or_pixel_loss() -> None:
    glyph = torch.arange(1, 64, dtype=torch.float32).reshape(1, 1, 7, 9)
    field = arti_nn.VisualField(patch_size=(4, 4), include_position=True)

    output = field(glyph)

    assert output.pixels.shape == (1, 6, 16)
    assert output.coord.shape == (1, 6, 5)
    assert output.fragments.shape == (1, 6, 21)
    assert torch.equal(output.pixels.sum(), glyph.sum())
    assert output.mask.all()


def test_concat_visual_fields_expands_fragment_axis_and_preserves_rigid_pixels() -> None:
    glyph = torch.zeros(2, 1, 8, 8)
    glyph[0, 0, 1, 1] = 1.0
    glyph[1, 0, 6, 6] = 1.0
    field = arti_nn.VisualField(patch_size=(4, 4))
    left = field(glyph, window=(0, 0, 8, 4), field_id=0.0)
    right = field(glyph, window=(0, 4, 8, 4), field_id=1.0)

    joined = arti_nn.concat_visual_fields(left, right)

    assert joined.fragments.shape == (2, 4, 21)
    assert joined.windows == ((0, 0, 8, 4), (0, 4, 8, 4))
    assert torch.equal(joined.pixels.sum(dim=(1, 2)), glyph.sum(dim=(1, 2, 3)))
    assert not torch.equal(joined.fragments[0], joined.fragments[1])


def test_visual_field_concat_is_pulse_compatible_and_differentiable() -> None:
    glyph = torch.randn(3, 1, 8, 8, requires_grad=True)
    field = arti_nn.VisualField(patch_size=(4, 4))
    joined = arti_nn.concat_visual_fields(
        field(glyph, window=(0, 0, 8, 4), field_id=0.0),
        field(glyph, window=(0, 4, 8, 4), field_id=1.0),
    )
    pulse = arti_nn.Pulse(k=2, dim=joined.fragments.shape[-1], hidden_dim=24)

    output = pulse(joined.fragments, mask=joined.mask)
    output.square().mean().backward()

    assert output.shape == (3, 2, 21)
    assert glyph.grad is not None
    assert torch.isfinite(glyph.grad).all()


def test_visual_field_rejects_blind_stride_and_invalid_windows() -> None:
    with pytest.raises(ValueError, match="pixel gaps"):
        arti_nn.VisualField(patch_size=(2, 2), stride=(3, 2))
    field = arti_nn.VisualField(patch_size=2)
    with pytest.raises(ValueError, match="inside"):
        field(torch.zeros(1, 1, 4, 4), window=(0, 3, 4, 2))


def test_visual_field_public_namespaces() -> None:
    assert arti.VisualField is arti_nn.VisualField
    assert arti_torch.VisualField is arti_nn.VisualField
    assert arti.concat_visual_fields is arti_nn.concat_visual_fields
    assert arti_torch.concat_visual_fields is arti_nn.concat_visual_fields
