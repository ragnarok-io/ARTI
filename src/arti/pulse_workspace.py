"""Experimental fixed-size Pulse workspaces composed from generic ARTI layers."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch import Tensor

from .nn import Fold, Half, UnFold


class FHUPulse(nn.Module):
    """Alpha fixed-size Pulse formed by Fold, Half, and UnFold.

    ``Fold`` first compacts fragments into ``k - exposed`` semantic slots,
    ``Half`` applies survival pressure to that dense core, and ``UnFold``
    queries ``exposed`` additional slots. The final output always has ``k``
    slots. This module is an isolated prototype and does not replace the
    public ``Pulse`` alias.
    """

    def __init__(
        self,
        k: int,
        dim: int,
        *,
        exposed: int | None = None,
        fragment_hidden_dim: int | None = None,
        fold_hidden_dim: int | None = None,
        use_half: bool = True,
        calibrate_half: bool = False,
        half_threshold: float = 1.0,
        half_base: float = 0.5,
        half_scale: float = 1.0,
        guide_dim: int | None = None,
        guide_transport: str = "mean",
        unfold_layout_mode: str = "learned",
        condition_unfold: bool = True,
        value_operators: int = 4,
        value_rank: int | None = None,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if k <= 1:
            raise ValueError("k must be greater than one")
        if dim <= 0:
            raise ValueError("dim must be positive")
        exposed_count = max(1, k // 2) if exposed is None else int(exposed)
        if not 0 < exposed_count < k:
            raise ValueError("exposed must be in the interval [1, k)")
        if fragment_hidden_dim is not None and fragment_hidden_dim <= 0:
            raise ValueError("fragment_hidden_dim must be positive")
        if fold_hidden_dim is not None and fold_hidden_dim <= 0:
            raise ValueError("fold_hidden_dim must be positive")
        if guide_dim is not None and guide_dim <= 0:
            raise ValueError("guide_dim must be positive")
        if guide_transport not in {"mean", "hard"}:
            raise ValueError("guide_transport must be 'mean' or 'hard'")
        if eps <= 0 or not math.isfinite(eps):
            raise ValueError("eps must be finite and positive")

        self.k = int(k)
        self.dim = int(dim)
        self.exposed = exposed_count
        self.compact_k = self.k - self.exposed
        self.use_half = bool(use_half)
        self.calibrate_half = bool(calibrate_half)
        self.guide_dim = None if guide_dim is None else int(guide_dim)
        self.guide_transport = guide_transport
        self.condition_unfold = bool(condition_unfold)
        self.eps = float(eps)

        self.fragment_proj = (
            nn.Linear(dim, dim)
            if fragment_hidden_dim is None
            else nn.Sequential(
                nn.Linear(dim, fragment_hidden_dim),
                nn.GELU(),
                nn.Linear(fragment_hidden_dim, dim),
            )
        )
        self.fold = Fold(k=self.compact_k, dim=dim, hidden_dim=fold_hidden_dim)
        self.half_act = (
            Half(threshold=half_threshold, base=half_base, scale=half_scale)
            if use_half
            else nn.Identity()
        )
        self.unfold = UnFold(
            dim=dim,
            exposed=self.exposed,
            guide_dim=self.guide_dim,
            condition_dim=dim if condition_unfold else None,
            layout_mode=unfold_layout_mode,
            value_operators=value_operators,
            value_rank=value_rank,
            query_chunk_size=self.exposed,
        )

    def forward(
        self,
        x: Tensor,
        q: Tensor | None = None,
        *,
        mask: Tensor | None = None,
        guide: Tensor | None = None,
        return_info: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if x.ndim != 3 or x.shape[-1] != self.dim:
            raise ValueError(f"x must have shape [B, N, {self.dim}]")
        if x.shape[1] == 0:
            raise ValueError("x must contain at least one fragment")
        mask_bool = self._normalize_mask(mask, x)
        guide_value = self._normalize_guide(guide, x)
        mask_value = mask_bool.unsqueeze(-1).to(x.dtype)
        fragments = self.fragment_proj(x) * mask_value
        compact = self.fold(fragments, q=q, mask=mask_bool)
        compact_guide = (
            None
            if guide_value is None
            else self._transport_guide(
                fragments,
                guide_value,
                q=q,
                mask=mask_bool,
            )
        )
        valid_sample = mask_bool.any(dim=-1, keepdim=True)
        compact_mask = valid_sample.expand(-1, self.compact_k)
        calibration_scale = None
        if self.use_half and self.calibrate_half:
            half_input, calibration_scale = self._calibrate(
                compact, compact_mask
            )
        else:
            half_input = compact
        survived = self.half_act(half_input)
        if calibration_scale is not None:
            survived = survived * calibration_scale
        survived = survived * compact_mask.unsqueeze(-1).to(survived.dtype)

        condition = None
        if self.condition_unfold:
            count = mask_value.sum(dim=1).clamp_min(1)
            condition = (fragments * mask_value).sum(dim=1) / count
        unfolded = self.unfold(
            survived,
            mask=compact_mask,
            guide=compact_guide,
            condition=condition,
            return_exposed_mask=True,
            return_source_index=True,
        )
        pulses, pulse_mask, exposed_mask, source_index = unfolded
        pulses = pulses * pulse_mask.unsqueeze(-1).to(pulses.dtype)
        if pulses.shape[1] != self.k:
            raise RuntimeError("FHUPulse produced an invalid workspace size")
        if not return_info:
            return pulses

        input_strength = compact.abs().mean(dim=(-2, -1))
        output_strength = survived.abs().mean(dim=(-2, -1))
        survival = output_strength / input_strength.clamp_min(self.eps)
        info = {
            "compact": compact,
            "half_input": half_input,
            "survived": survived,
            "pulse_mask": pulse_mask,
            "exposed_mask": exposed_mask,
            "source_index": source_index,
            "survival": survival,
            "fragment_norm": fragments.norm(dim=-1).mean().detach(),
            "pulse_norm": pulses.norm(dim=-1).mean().detach(),
        }
        if calibration_scale is not None:
            info["half_reference_scale"] = calibration_scale
        if compact_guide is not None:
            info["compact_guide"] = compact_guide
        return pulses, info

    def _transport_guide(
        self,
        value: Tensor,
        guide: Tensor,
        *,
        q: Tensor | None,
        mask: Tensor,
    ) -> Tensor:
        if self.fold.mode != "soft" or self.fold.assignment is None:
            raise RuntimeError("guide transport requires the soft Fold path")
        if q is None:
            input_survival = torch.sigmoid(self.fold.salience(value))
        else:
            input_survival = q.unsqueeze(-1) if q.ndim == 2 else q
            input_survival = input_survival.to(
                device=value.device,
                dtype=value.dtype,
            ).clamp(0.0, 1.0)
        input_survival = input_survival * mask.unsqueeze(-1).to(value.dtype)
        logits = self.fold.assignment(value) / self.fold.temperature
        logits = logits + input_survival.clamp_min(self.fold.eps).log()
        assignment = torch.softmax(logits, dim=1)
        contribution = assignment * input_survival
        mean_transport = contribution / contribution.sum(dim=1, keepdim=True).clamp_min(
            self.fold.eps
        )
        if self.guide_transport == "mean":
            transport = mean_transport
        else:
            source = contribution.argmax(dim=1, keepdim=True)
            transport = torch.zeros_like(contribution).scatter_(1, source, 1.0)
            valid = mask.any(dim=1, keepdim=True).unsqueeze(-1)
            transport = transport * valid.to(transport.dtype)
        return torch.bmm(transport.transpose(1, 2), guide)

    def _calibrate(self, value: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        weight = mask.unsqueeze(-1).to(value.dtype)
        count = weight.sum(dim=(-2, -1), keepdim=True) * value.shape[-1]
        rms = (value.square() * weight).sum(dim=(-2, -1), keepdim=True)
        rms = (rms / count.clamp_min(1)).sqrt().clamp_min(self.eps)
        return value / rms * weight, rms

    @staticmethod
    def _normalize_mask(mask: Tensor | None, x: Tensor) -> Tensor:
        if mask is None:
            return torch.ones(x.shape[:2], dtype=torch.bool, device=x.device)
        if mask.ndim == 3 and mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        if mask.shape != x.shape[:2]:
            raise ValueError("mask must have shape [B, N] or [B, N, 1]")
        if mask.device != x.device:
            raise ValueError("mask and x must share one device")
        if mask.dtype == torch.bool:
            return mask
        if not mask.is_floating_point():
            raise ValueError("mask must be boolean or floating point")
        if not torch.isfinite(mask).all():
            raise ValueError("mask must contain only finite values")
        return mask > 0

    def _normalize_guide(self, guide: Tensor | None, x: Tensor) -> Tensor | None:
        if guide is None:
            if self.unfold.layout_mode == "canonical":
                raise ValueError("canonical UnFold layout requires guide")
            return None
        if self.guide_dim is None:
            raise ValueError("guide was provided but guide_dim is disabled")
        expected = (*x.shape[:2], self.guide_dim)
        if guide.shape != expected:
            raise ValueError(f"guide must have shape {list(expected)}")
        if guide.device != x.device:
            raise ValueError("guide and x must share one device")
        if not guide.is_floating_point() or not torch.isfinite(guide).all():
            raise ValueError("guide must be a finite floating-point tensor")
        return guide.to(dtype=x.dtype)

    def extra_repr(self) -> str:
        args = [f"k={self.k}", f"dim={self.dim}", f"exposed={self.exposed}"]
        if not self.use_half:
            args.append("use_half=False")
        if self.calibrate_half:
            args.append("calibrate_half=True")
        if self.use_half and self.half_act.threshold != 1.0:
            args.append(f"half_threshold={self.half_act.threshold:g}")
        if self.use_half and self.half_act.base != 0.5:
            args.append(f"half_base={self.half_act.base:g}")
        if self.use_half and self.half_act.scale != 1.0:
            args.append(f"half_scale={self.half_act.scale:g}")
        if self.guide_dim is not None:
            args.append(f"guide_dim={self.guide_dim}")
        if self.guide_transport != "mean":
            args.append(f"guide_transport={self.guide_transport!r}")
        if self.unfold.layout_mode != "learned":
            args.append(f"unfold_layout_mode={self.unfold.layout_mode!r}")
        if not self.condition_unfold:
            args.append("condition_unfold=False")
        return ", ".join(args)


__all__ = ["FHUPulse"]
