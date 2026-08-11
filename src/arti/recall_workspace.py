"""Optional compact workspaces for Recall candidate traces."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from .nn import Fold, UnFold


class RecallWorkspace(nn.Module):
    """Alpha workspace that compacts, thins, expands, and reads Recall traces.

    The module deliberately does not produce a residual update. It receives
    candidate traces and token queries, then returns one read context per
    query. Recall remains responsible for recognition, emission, final trace
    survival, and the residual update.

    ``fold``, ``half``, and ``unfold`` are independent optional stages. This
    keeps the composition experimental without changing the public contracts
    of the three generic tensor layers.
    """

    def __init__(
        self,
        dim: int,
        *,
        fold: Fold | None,
        half: nn.Module | None,
        unfold: UnFold | None,
        calibrate_half: bool = True,
        read_temperature: float = 1.0,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if read_temperature <= 0 or not math.isfinite(read_temperature):
            raise ValueError("read_temperature must be finite and positive")
        if eps <= 0 or not math.isfinite(eps):
            raise ValueError("eps must be finite and positive")
        if fold is not None and fold.dim not in {None, dim}:
            raise ValueError("fold feature dimension must match dim")
        if unfold is not None and unfold.dim != dim:
            raise ValueError("unfold feature dimension must match dim")
        self.dim = int(dim)
        self.fold = fold
        self.workspace_half = half
        self.unfold = unfold
        self.calibrate_half = bool(calibrate_half)
        self.read_temperature = float(read_temperature)
        self.eps = float(eps)

        self.read_query = nn.Linear(dim, dim, bias=False)
        self.read_key = nn.Linear(dim, dim, bias=False)
        self.read_value = nn.Linear(dim, dim, bias=False)
        self.read_output = nn.Linear(dim, dim, bias=False)
        # Layout positions affect addressing but never enter emitted values.
        self.position_key = nn.Linear(1, dim, bias=False)
        self.expand_condition = (
            None
            if unfold is None or unfold.condition_dim is None
            else nn.Linear(dim, unfold.condition_dim, bias=False)
        )

    def forward(
        self,
        queries: Tensor,
        candidates: Tensor,
        *,
        query_mask: Tensor | None = None,
        candidate_mask: Tensor | None = None,
        return_info: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if queries.ndim != 3 or queries.shape[-1] != self.dim:
            raise ValueError(f"queries must have shape [B, N, {self.dim}]")
        if candidates.ndim != 3 or candidates.shape[-1] != self.dim:
            raise ValueError(f"candidates must have shape [B, M, {self.dim}]")
        if queries.shape[0] != candidates.shape[0]:
            raise ValueError("queries and candidates must use the same batch size")
        if queries.device != candidates.device or queries.dtype != candidates.dtype:
            raise ValueError("queries and candidates must share one device and dtype")
        if candidates.shape[1] == 0:
            raise ValueError("candidates must contain at least one trace")

        qmask = self._mask(query_mask, queries.shape[:2], queries.device, "query_mask")
        cmask = self._mask(candidate_mask, candidates.shape[:2], candidates.device, "candidate_mask")
        if qmask is None:
            qmask = torch.ones(queries.shape[:2], dtype=torch.bool, device=queries.device)
        if cmask is None:
            cmask = torch.ones(candidates.shape[:2], dtype=torch.bool, device=candidates.device)

        valid_sample = cmask.any(dim=-1, keepdim=True)
        if self.fold is None:
            folded = candidates * cmask.unsqueeze(-1).to(candidates.dtype)
            folded_mask = cmask
        else:
            # A constant q isolates Fold assignment from Half survival.
            fold_q = torch.ones(candidates.shape[:2], device=candidates.device, dtype=candidates.dtype)
            masked_candidates = candidates * cmask.unsqueeze(-1).to(candidates.dtype)
            folded = self.fold(masked_candidates, q=fold_q, mask=cmask)
            folded_mask = valid_sample.expand(-1, folded.shape[1])

        half_input = (
            self._calibrate(folded, folded_mask)
            if self.calibrate_half and self.workspace_half is not None
            else folded
        )
        survived = self.workspace_half(half_input) if self.workspace_half is not None else half_input
        survived = survived * folded_mask.unsqueeze(-1).to(survived.dtype)

        if self.unfold is None:
            expanded = survived
            workspace_mask = folded_mask
            exposed_mask = torch.zeros_like(workspace_mask)
            source_index = torch.arange(
                expanded.shape[1], device=expanded.device, dtype=torch.long
            ).unsqueeze(0).expand(expanded.shape[0], -1)
        else:
            condition = None
            if self.expand_condition is not None:
                query_weight = qmask.unsqueeze(-1).to(queries.dtype)
                query_summary = (queries * query_weight).sum(dim=1)
                query_summary = query_summary / query_weight.sum(dim=1).clamp_min(1)
                condition = self.expand_condition(query_summary)
            unfolded = self.unfold(
                survived,
                mask=folded_mask,
                condition=condition,
                return_exposed_mask=True,
                return_source_index=True,
            )
            expanded, workspace_mask, exposed_mask, source_index = unfolded

        context, read_weights = self._read(
            queries,
            expanded,
            query_mask=qmask,
            workspace_mask=workspace_mask,
        )
        if not return_info:
            return context

        input_strength = half_input.abs().mean(dim=(-2, -1))
        output_strength = survived.abs().mean(dim=(-2, -1))
        survival = output_strength / input_strength.clamp_min(self.eps)
        query_count = qmask.sum(dim=-1, keepdim=True).clamp_min(1).to(read_weights.dtype)
        slot_usage = read_weights.sum(dim=1) / query_count
        return context, {
            "folded": folded,
            "half_input": half_input,
            "survived": survived,
            "expanded": expanded,
            "workspace_mask": workspace_mask,
            "exposed_mask": exposed_mask,
            "source_index": source_index,
            "expand_condition": (
                queries.new_zeros((queries.shape[0], 0))
                if self.expand_condition is None
                else condition
            ),
            "read_weights": read_weights,
            "slot_usage": slot_usage,
            "survival": survival,
            "context_norm": context.norm(dim=-1),
        }

    def _calibrate(self, value: Tensor, mask: Tensor) -> Tensor:
        weights = mask.unsqueeze(-1).to(value.dtype)
        count = weights.sum(dim=(-2, -1), keepdim=True) * value.shape[-1]
        rms = (value.square() * weights).sum(dim=(-2, -1), keepdim=True)
        rms = (rms / count.clamp_min(1)).sqrt().clamp_min(self.eps)
        calibrated = value / rms
        return calibrated * weights

    def _read(
        self,
        queries: Tensor,
        workspace: Tensor,
        *,
        query_mask: Tensor,
        workspace_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        length = workspace.shape[1]
        positions = torch.linspace(
            -1.0,
            1.0,
            length,
            device=workspace.device,
            dtype=workspace.dtype,
        ).reshape(1, length, 1)
        query_key = self.read_query(queries)
        workspace_key = self.read_key(workspace) + self.position_key(positions)
        logits = torch.bmm(query_key, workspace_key.transpose(1, 2))
        logits = logits / (self.dim**0.5 * self.read_temperature)
        logits = logits.masked_fill(~workspace_mask[:, None], -1e4)
        weights = torch.softmax(logits, dim=-1)
        weights = weights * workspace_mask[:, None].to(weights.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(self.eps)
        weights = weights * query_mask.unsqueeze(-1).to(weights.dtype)
        context = torch.bmm(weights, self.read_value(workspace))
        context = self.read_output(context)
        context = context * query_mask.unsqueeze(-1).to(context.dtype)
        return context, weights

    @staticmethod
    def _mask(
        mask: Tensor | None,
        shape: torch.Size,
        device: torch.device,
        name: str,
    ) -> Tensor | None:
        if mask is None:
            return None
        if mask.shape != shape:
            raise ValueError(f"{name} must have shape {list(shape)}")
        if mask.dtype != torch.bool:
            raise ValueError(f"{name} must be a boolean tensor")
        if mask.device != device:
            raise ValueError(f"{name} must share the input device")
        return mask


__all__ = ["RecallWorkspace"]
