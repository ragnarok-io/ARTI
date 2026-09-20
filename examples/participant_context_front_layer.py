"""Recommended front-layer participant phase placement.

External participant/source phase should be applied while hidden states still
align one-to-one with the original tokens.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from arti import ARTILayer, build_participant_context


class FrontARTIEncoder(nn.Module):
    def __init__(self, vocab_size: int, dim: int, participants: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dim)
        self.participant_coord = nn.Parameter(torch.eye(participants, participants), requires_grad=False)
        self.arti = ARTILayer(
            input_dim=dim,
            coord_dim=participants,
            hidden_dim=dim,
            use_pairwise_context=True,
        )
        self.downstream = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(dim, nhead=4, batch_first=True, dropout=0.0),
            num_layers=1,
        )

    def forward(self, token_ids: torch.Tensor, participant_ids: torch.Tensor, readable_by: torch.Tensor) -> torch.Tensor:
        mask = token_ids.ne(0)
        ctx = build_participant_context(
            participant_ids,
            self.participant_coord,
            readable_by,
            mask=mask,
            assistant_id=0,
        )
        x = self.embedding(token_ids)
        out = self.arti(x, coord=ctx.coord, mask=ctx.mask, visibility=ctx.visibility, observer_coord=ctx.observer_coord)
        return self.downstream(out.y, src_key_padding_mask=~ctx.mask)


if __name__ == "__main__":
    model = FrontARTIEncoder(vocab_size=32, dim=16, participants=3)
    token_ids = torch.tensor([[4, 7, 9, 2, 0]])
    participant_ids = torch.tensor([[1, 0, 2, 0, 0]])
    readable_by = torch.ones(3, 3, dtype=torch.bool)
    hidden = model(token_ids, participant_ids, readable_by)
    print("hidden", tuple(hidden.shape))
