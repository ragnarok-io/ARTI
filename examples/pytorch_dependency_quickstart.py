"""Minimal ARTI dependency quickstart.

Run from the repository root:

    uv run --extra torch python examples/pytorch_dependency_quickstart.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from arti import ARTIClassifier, ARTILayer


def layer_example() -> None:
    layer = ARTILayer(input_dim=32, coord_dim=4, hidden_dim=64)
    x = torch.randn(4, 12, 32)
    coord = torch.randn(4, 12, 4)
    mask = torch.ones(4, 12, dtype=torch.bool)

    out = layer(x, coord=coord, mask=mask)
    loss = out.pooled.square().mean()
    loss.backward()

    print("layer.y", tuple(out.y.shape))
    print("layer.pooled", tuple(out.pooled.shape))
    print("diagnostics", sorted(out.diagnostics)[:5])


def classifier_example() -> None:
    model = ARTIClassifier(input_dim=16, hidden_dim=32, output_dim=3)
    x = torch.randn(8, 16)
    target = torch.randint(0, 3, (8,))
    logits = model(x)
    loss = F.cross_entropy(logits, target)
    loss.backward()

    print("classifier.logits", tuple(logits.shape))


if __name__ == "__main__":
    layer_example()
    classifier_example()
