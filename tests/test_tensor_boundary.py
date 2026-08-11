from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from arti.tensor_boundary import TensorLayout, find_primary_tensor, replace_tensor_at_path


@pytest.mark.parametrize(
    ("module", "shape", "feature_axis", "packed_shape"),
    [
        (None, (2, 7), 1, (2, 7)),
        (None, (2, 5, 7), 2, (2, 5, 7)),
        (nn.Conv2d(3, 7, 1), (2, 7, 4, 3), 1, (2, 12, 7)),
        (None, (2, 4, 3, 7), 3, (2, 12, 7)),
        (nn.Conv3d(3, 7, 1), (2, 7, 3, 4, 5), 1, (2, 60, 7)),
    ],
)
def test_tensor_layout_round_trips_without_value_loss(
    module, shape, feature_axis, packed_shape
) -> None:
    tensor = torch.randn(shape)
    layout = TensorLayout.infer(tensor, module)
    packed = layout.pack(tensor)

    assert layout.feature_axis == feature_axis
    assert packed.shape == packed_shape
    assert torch.equal(layout.restore(packed), tensor)


def test_time_first_recurrent_layout_preserves_batch_axis() -> None:
    tensor = torch.randn(5, 2, 7)
    layout = TensorLayout.infer(tensor, nn.GRU(3, 7, batch_first=False))

    assert layout.batch_axis == 1
    assert layout.pack(tensor).shape == (2, 5, 7)
    assert torch.equal(layout.restore(layout.pack(tensor)), tensor)


def test_nested_output_path_replacement_preserves_other_values() -> None:
    primary = torch.randn(2, 3, 4)
    side = torch.tensor([1, 2])
    output = {"metadata": side, "payload": [{"sample": primary}, "keep"]}

    found = find_primary_tensor(output)
    assert found is not None
    tensor, path = found
    replaced = replace_tensor_at_path(output, path, tensor + 1)

    assert path == ("payload", 0, "sample")
    assert torch.equal(replaced["payload"][0]["sample"], primary + 1)
    assert torch.equal(replaced["metadata"], side)
    assert replaced["payload"][1] == "keep"


def test_layout_rejects_shape_drift() -> None:
    layout = TensorLayout.infer(torch.randn(2, 3, 4))
    with pytest.raises(ValueError, match="does not match layout"):
        layout.pack(torch.randn(2, 4, 4))
