import torch
import torch.nn as nn

from arti import ARTIResidualBlock


def test_residual_block_works_in_sequential_for_vectors():
    model = nn.Sequential(
        nn.Linear(32, 64),
        ARTIResidualBlock(dim=64),
        nn.Linear(64, 10),
    )

    y = model(torch.randn(8, 32))

    assert y.shape == (8, 10)
