"""Runtime-only compilation helpers for ARTI hot paths."""

from __future__ import annotations

import math
from functools import lru_cache
from typing import Callable

import torch
import torch.nn.functional as F
from torch import Tensor


CompiledProductTail = Callable[[Tensor, Tensor, Tensor, Tensor, Tensor], Tensor]


@lru_cache(maxsize=32)
def grouped_product_tail(
    *,
    group_size: int,
    factor_groups: int,
    key_dim: int,
    threshold: float,
    base: float,
    half_scale: float,
    mode: str,
    dynamic: bool,
    fullgraph: bool,
) -> CompiledProductTail:
    """Return one shared compiled tail for a static Recall configuration.

    Parameters remain ordinary function inputs. Consequently, layers with the
    same layout share the compiled graph without sharing weights.
    """

    route_scale = key_dim**-0.5
    log_base = math.log(base)

    def tail(
        state: Tensor,
        query: Tensor,
        selected_groups: Tensor,
        key_bank: Tensor,
        value_bank: Tensor,
    ) -> Tensor:
        grouped_keys = key_bank.reshape(
            factor_groups * 2,
            group_size,
            key_dim,
        )
        selected_keys = grouped_keys[selected_groups]
        slot_logits = (
            torch.einsum("bnd,bnftmd->bnftm", query, selected_keys)
            * route_scale
        )
        weights = torch.softmax(slot_logits, dim=-1)
        slot_offsets = torch.arange(group_size, device=state.device)
        indices = selected_groups.unsqueeze(-1) * group_size + slot_offsets
        factors = F.embedding_bag(
            indices.reshape(-1, group_size),
            value_bank,
            mode="sum",
            per_sample_weights=weights.reshape(-1, group_size).to(value_bank.dtype),
        ).reshape(*state.shape[:2], 2, state.shape[-1]).to(state.dtype)
        scale_factor = factors[..., 0, :]
        shift = factors[..., 1, :]
        raw_write = (1.0 + torch.tanh(scale_factor)) * (state + shift) - state
        deficit = torch.relu((threshold - raw_write.abs()) / half_scale)
        return torch.exp(log_base * deficit) * raw_write

    return torch.compile(
        tail,
        mode=mode,
        dynamic=dynamic,
        fullgraph=fullgraph,
    )
