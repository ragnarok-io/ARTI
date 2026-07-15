"""Expand a tensor while preserving original values under a learned layout."""

from __future__ import annotations

import torch

from arti.nn import UnFold


def main() -> None:
    torch.manual_seed(7)
    layer = UnFold(dim=16, exposed=4, guide_dim=3, condition_dim=2)
    x = torch.randn(4, 6, 16)
    guide = torch.randn(4, 6, 3)
    condition = torch.randn(4, 2)
    y, exposed_mask, source_index = layer(
        x,
        guide=guide,
        condition=condition,
        return_exposed_mask=True,
        return_source_index=True,
    )

    print(f"input: {tuple(x.shape)}")
    print(f"output: {tuple(y.shape)}")
    print(f"exposed values per sample: {int(exposed_mask[0].sum())}")
    print(f"first layout source indices: {source_index[0].tolist()}")


if __name__ == "__main__":
    main()
