"""Minimal ARTI dependency quickstart.

Run from the repository root:

    uv run --extra torch python examples/pytorch_dependency_quickstart.py
"""

from __future__ import annotations

import torch
from arti import ARTILayer
from arti.nn import Layer


def layer_example() -> None:
    x = torch.randn(4, 12, 32)
    mask = torch.ones(4, 12, dtype=torch.bool)

    host = ARTILayer()
    value, info = host(x, mask=mask, return_info=True)
    assert torch.equal(value, x)
    assert torch.equal(info.value, x)
    print("program.value", tuple(value.shape))

    profile = Layer(dim=32)
    out = profile(x, mask=mask)
    loss = out.pooled.square().mean()
    loss.backward()
    print("profile.y", tuple(out.y.shape))
    print("profile.pooled", tuple(out.pooled.shape))


if __name__ == "__main__":
    layer_example()
