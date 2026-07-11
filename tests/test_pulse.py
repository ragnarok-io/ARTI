from __future__ import annotations

import pytest
import torch

from arti import PulseCompressor, assert_pulse_distinct, fixed_width_pulse_ids, pulse_compress, pulse_distinctness_report


def test_pulse_compress_weighted_mean_and_spans() -> None:
    x = torch.tensor([[[1.0], [3.0], [10.0], [20.0]]])
    pulse_ids = torch.tensor([[0, 0, 1, -1]])
    token_weight = torch.tensor([[1.0, 3.0, 2.0, 1.0]])

    out = pulse_compress(x, pulse_ids, token_weight=token_weight, pulse_count=3)

    assert out.pulse.shape == (1, 3, 1)
    assert torch.allclose(out.pulse[0, 0], torch.tensor([2.5]))
    assert torch.allclose(out.pulse[0, 1], torch.tensor([10.0]))
    assert out.mask.tolist() == [[True, True, False]]
    assert out.coverage.tolist() == [[4.0, 2.0, 0.0]]
    assert out.start.tolist() == [[0, 2, -1]]
    assert out.end.tolist() == [[2, 3, -1]]


def test_pulse_compressor_keeps_gradients() -> None:
    x = torch.randn(2, 5, 3, requires_grad=True)
    pulse_ids = torch.tensor([[0, 0, 1, 1, -1], [0, 1, 1, 2, 2]])
    layer = PulseCompressor()

    out = layer(x, pulse_ids, pulse_count=3)
    loss = out.pulse[out.mask].square().mean()
    loss.backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


def test_fixed_width_pulse_ids_marks_padding() -> None:
    ids = fixed_width_pulse_ids(torch.tensor([5, 2]), pulse_width=2)

    assert ids.tolist() == [[0, 0, 1, 1, 2], [0, 0, -1, -1, -1]]


def test_pulse_distinctness_reports_retention() -> None:
    raw = torch.tensor([[[1.0, 0.0]], [[1.0, 1.0]], [[1.0, 2.0]]])
    pulse = torch.tensor([[[0.5, 0.0]], [[0.5, 0.5]], [[0.5, 1.0]]])

    report = pulse_distinctness_report(raw, pulse)

    assert report.count == 3
    assert report.latent_collision_count == 0
    assert report.latent_min_distance > 0.0
    assert report.distance_retention > 0.0
    assert report.nearest_neighbor_consistency >= 0.0
    assert_pulse_distinct(raw, pulse, min_pulse_distance=0.0, min_distance_retention=0.1)


def test_pulse_distinctness_rejects_collapse() -> None:
    raw = torch.tensor([[[1.0, 0.0]], [[1.0, 1.0]]])
    collapsed = torch.zeros(2, 1, 2)

    report = pulse_distinctness_report(raw, collapsed)

    assert report.latent_collision_count == 1
    with pytest.raises(ValueError, match="collisions"):
        assert_pulse_distinct(raw, collapsed)


def test_pulse_rejects_huge_ids_before_output_allocation() -> None:
    x = torch.randn(1, 2, 4)
    pulse_ids = torch.tensor([[0, 2**31]])

    with pytest.raises(ValueError, match="cannot exceed"):
        pulse_compress(x, pulse_ids)


def test_pulse_rejects_ids_outside_explicit_workspace() -> None:
    x = torch.randn(1, 3, 4)
    pulse_ids = torch.tensor([[0, 1, 2]])

    with pytest.raises(ValueError, match="smaller than pulse_count"):
        pulse_compress(x, pulse_ids, pulse_count=2)
