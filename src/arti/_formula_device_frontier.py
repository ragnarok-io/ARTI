"""Tensor-only route ordering and refill for bounded Formula execution waves.

Route entries are ranks of complete execution IDs in a prepared string table.
The -1 suffix represents the end of a route, preserving Python tuple ordering.
Numeric values, scores, live rows and selected paths are never cached here.
"""

import torch
from torch import nn


class TensorRankSelect(nn.Module):
    def __init__(self, width, coverage):
        super().__init__()
        self.width = width
        self.coverage = coverage

    def forward(self, scores, alive, materialized, membership, tie_order):
        order = tie_order.index_select(0, torch.argsort(scores.index_select(0, tie_order), descending=True, stable=True))
        rank = torch.zeros_like(order).scatter(0, order, torch.arange(order.shape[0], device=order.device))
        category = torch.where(alive, 0, 2)
        if self.coverage:
            members = membership & alive[:, None]
            first_rank = torch.where(members, rank[:, None], order.shape[0]).amin(0)
            representative = (members & (rank[:, None] == first_rank)).any(1)
            category = torch.where(alive, torch.where(representative, 0, 1), 2)
        selected = torch.argsort(category * order.shape[0] + rank)[:self.width]
        valid = alive[selected]
        pending = valid & ~materialized[selected]
        return torch.stack((torch.where(valid, selected, -1), pending.to(torch.int64)), dim=1)


class TensorRouteFrontier(nn.Module):
    """Select a GPU pool, including route ties and repeated object identities."""

    def __init__(self, width, coverage):
        super().__init__()
        self.selector = TensorRankSelect(width, coverage)
        self.coverage = coverage

    def forward(self, scores, alive, materialized, membership, routes, identities):
        positions = torch.arange(scores.shape[0], device=scores.device)
        tie_order = positions
        for column in range(routes.shape[1] - 1, -1, -1):
            tie_order = tie_order[torch.argsort(routes[tie_order, column], stable=True)]
        if self.coverage:
            # Identity deduplication matches the reference coverage selector.
            # Keep its first live occurrence; plain top-k intentionally keeps duplicates.
            identity_order = torch.argsort(identities, stable=True)
            ordered_ids = identities[identity_order]
            starts = torch.cat((torch.ones_like(ordered_ids[:1], dtype=torch.bool), ordered_ids[1:] != ordered_ids[:-1]))
            groups = starts.to(torch.int64).cumsum(0) - 1
            first = torch.full_like(positions, scores.shape[0]).scatter_reduce(
                0, groups, torch.where(alive[identity_order], identity_order, scores.shape[0]), reduce="amin",
            )
            unique = torch.zeros_like(alive).scatter(0, identity_order, identity_order == first[groups])
            alive = alive & unique
        return self.selector(scores, alive, materialized, membership, tie_order)


class TensorFrontierRecord(nn.Module):
    """Apply finite-result masks to a selected packet without reading it on CPU."""

    def forward(self, alive, materialized, identities, packet, accepted):
        selected, pending = packet[:, 0], packet[:, 1].bool()
        selected_identity = identities[selected.clamp_min(0)]
        attempted = pending & (selected >= 0)
        same = identities[:, None] == selected_identity[None]
        rejected = (same & (attempted & ~accepted)[None]).any(1)
        ready = (same & (attempted & accepted)[None]).any(1)
        return alive & ~rejected, materialized | ready
