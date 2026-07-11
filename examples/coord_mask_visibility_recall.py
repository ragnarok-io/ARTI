"""ARTI runtime context example: coord, mask, visibility, and recall.

Run from the repository root:

    uv run --extra torch python examples/coord_mask_visibility_recall.py
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from arti import ARTILayer


def main() -> None:
    torch.manual_seed(7)

    batch = 2
    tokens = 6
    input_dim = 16
    coord_dim = 4
    hidden_dim = 24

    layer = ARTILayer(
        input_dim=input_dim,
        coord_dim=coord_dim,
        hidden_dim=hidden_dim,
        recall_steps=1,
        use_pairwise_context=True,
    )
    recall_target = torch.nn.Linear(input_dim, hidden_dim)

    x = torch.randn(batch, tokens, input_dim)
    coord = torch.randn(batch, tokens, coord_dim)

    mask = torch.ones(batch, tokens, dtype=torch.bool)
    mask[:, -1] = False

    visibility = torch.ones(batch, tokens, tokens, dtype=torch.bool)
    visibility[:, :, -1] = False
    visibility[:, -1, :] = False

    recall = torch.randn(batch, tokens, hidden_dim)

    out = layer(x, coord=coord, mask=mask, visibility=visibility, recall=recall)

    pooled_loss = out.pooled.square().mean()
    recall_loss = F.mse_loss(out.recall_prediction, recall_target(x).detach())
    loss = pooled_loss + 0.1 * recall_loss
    loss.backward()

    print("y", tuple(out.y.shape))
    print("pooled", tuple(out.pooled.shape))
    print("recall_prediction", tuple(out.recall_prediction.shape))
    print("loss", round(float(loss.detach()), 6))
    print("diagnostics", sorted(out.diagnostics)[:8])


if __name__ == "__main__":
    main()
