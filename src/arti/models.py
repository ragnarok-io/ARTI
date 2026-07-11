"""Small reference models showing how ARTI composes with PyTorch."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from .blocks import ARTIResidualBlock


class ARTIClassifier(nn.Module):
    """Minimal reference classifier, intentionally domain-free."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, coord_dim: int = 0) -> None:
        super().__init__()
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.arti = ARTIResidualBlock(hidden_dim, coord_dim=coord_dim)
        self.head = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.head(self.arti(self.in_proj(x)))
