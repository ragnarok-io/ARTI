"""Activation-style and compact neural network modules."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .functional import half
from .visual_field import VisualField, VisualFieldOutput, concat_visual_fields


def _fold_weighted_sum(x_weighted: Tensor, logits: Tensor, dropout: nn.Dropout, topk: int | None = None) -> Tensor:
    if topk is not None and topk < logits.shape[1]:
        keep = min(topk, logits.shape[1])
        values, indices = logits.transpose(1, 2).topk(keep, dim=-1)
        weights = torch.softmax(values, dim=-1)
        weights = dropout(weights)
        selected = x_weighted.unsqueeze(1).expand(-1, logits.shape[2], -1, -1).gather(
            2, indices.unsqueeze(-1).expand(-1, -1, -1, x_weighted.shape[-1])
        )
        return (weights.unsqueeze(-1) * selected).sum(dim=2)
    weights = torch.softmax(logits, dim=1)
    weights = dropout(weights)
    return torch.bmm(weights.transpose(1, 2), x_weighted)


class _LazySameDimProjection(nn.Module):
    def __init__(self, hidden_dim: int | None = None) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.proj: nn.Module | None = None

    def forward(self, x: Tensor) -> Tensor:
        if self.proj is None:
            dim = x.shape[-1]
            if self.hidden_dim is None:
                proj: nn.Module = nn.Linear(dim, dim)
            else:
                proj = nn.Sequential(nn.Linear(dim, self.hidden_dim), nn.GELU(), nn.Linear(self.hidden_dim, dim))
            self.proj = proj.to(device=x.device, dtype=x.dtype)
        return self.proj(x)


class _GatedRefineDelta(nn.Module):
    def __init__(self, dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        mid = dim if hidden_dim is None else min(dim, hidden_dim)
        self.norm = nn.LayerNorm(dim)
        self.delta = nn.Sequential(nn.Linear(dim, mid), nn.GELU(), nn.Linear(mid, dim))
        self.gate = nn.Linear(dim, dim)

    def forward(self, x: Tensor) -> Tensor:
        z = self.norm(x)
        return torch.sigmoid(self.gate(z)) * self.delta(z)


class Half(nn.Module):
    """Salience-conditioned activation.

    Strong features pass with survival close to ``1``. Features below
    ``threshold`` fade by ``base ** D`` where ``D`` is the scaled salience
    deficit. With the default ``base=0.5``, each unit of insufficient salience
    halves feature survival.
    """

    def __init__(
        self,
        threshold: float = 1.0,
        base: float = 0.5,
        scale: float = 1.0,
        *,
        stochastic: bool = False,
    ) -> None:
        super().__init__()
        if not math.isfinite(threshold):
            raise ValueError("threshold must be finite")
        if not math.isfinite(base) or not 0 < base <= 1:
            raise ValueError("base must be in the interval (0, 1]")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be positive")
        self.threshold = float(threshold)
        self.base = float(base)
        self.scale = float(scale)
        self.stochastic = bool(stochastic)

    def forward(self, x: Tensor) -> Tensor:
        return half(
            x,
            threshold=self.threshold,
            base=self.base,
            scale=self.scale,
            stochastic=self.stochastic,
            training=self.training,
        )

    def extra_repr(self) -> str:
        args = [f"threshold={self.threshold:g}", f"base={self.base:g}", f"scale={self.scale:g}"]
        if self.stochastic:
            args.append("stochastic=True")
        return ", ".join(args)


class Fold(nn.Module):
    """Soft tensor compaction into a fixed-size latent workspace.

    ``Fold`` maps ``x: [B, N, D]`` into ``[B, K, D]`` with differentiable
    soft assignments. Optional ``q`` values guide survival/salience; optional
    ``mask`` values only mark valid input slots. When ``q`` is omitted,
    salience is estimated from ``x``. Keep ``q`` and ``mask`` separate when a
    padding mask is not meant to rank slot importance.
    """

    def __init__(
        self,
        k: int,
        *,
        dim: int | None = None,
        hidden_dim: int | None = None,
        temperature: float = 1.0,
        dropout: float = 0.0,
        mode: str = "soft",
        topk: int | None = None,
        heads: int = 1,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if k <= 0:
            raise ValueError("k must be positive")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in the interval [0, 1)")
        if mode not in {"soft", "attention"}:
            raise ValueError("mode must be 'soft' or 'attention'")
        if topk is not None and topk <= 0:
            raise ValueError("topk must be positive")
        if heads <= 0:
            raise ValueError("heads must be positive")
        if mode == "attention" and dim is None:
            raise ValueError("dim must be provided when mode='attention'")
        if mode == "attention" and dim is not None and dim % heads != 0:
            raise ValueError("dim must be divisible by heads when mode='attention'")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.k = int(k)
        self.dim = None if dim is None else int(dim)
        self.hidden_dim = hidden_dim
        self.temperature = float(temperature)
        self.mode = mode
        self.topk = None if topk is None else int(topk)
        self.heads = int(heads)
        self.eps = float(eps)
        self.dropout = nn.Dropout(dropout)
        if mode == "attention":
            assert dim is not None
            self.assignment: nn.Module | None = None
            self.query = nn.Parameter(torch.empty(self.k, dim))
            self.key = nn.Linear(dim, dim)
            self.value = nn.Linear(dim, dim)
            self.output = nn.Linear(dim, dim)
            nn.init.normal_(self.query, std=dim**-0.5)
            if hidden_dim is None:
                self.salience = nn.Linear(dim, 1)
            else:
                if hidden_dim <= 0:
                    raise ValueError("hidden_dim must be positive")
                self.salience = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        elif hidden_dim is None:
            self.assignment = nn.Linear(dim, self.k) if dim is not None else nn.LazyLinear(self.k)
            self.salience = nn.Linear(dim, 1) if dim is not None else nn.LazyLinear(1)
        else:
            if hidden_dim <= 0:
                raise ValueError("hidden_dim must be positive")
            if dim is None:
                self.assignment = nn.Sequential(nn.LazyLinear(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, self.k))
                self.salience = nn.Sequential(nn.LazyLinear(hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
            else:
                self.assignment = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, self.k))
                self.salience = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))

    def forward(self, x: Tensor, q: Tensor | None = None, *, mask: Tensor | None = None) -> Tensor:
        if x.ndim != 3:
            raise ValueError("x must have shape [B, N, D]")
        if x.shape[1] == 0:
            raise ValueError("x must contain at least one input slot")
        if self.dim is not None and x.shape[-1] != self.dim:
            raise ValueError(f"expected x.shape[-1] == {self.dim}, got {x.shape[-1]}")

        if q is None:
            survival = torch.sigmoid(self.salience(x))
        else:
            survival = self._normalize_q(q, x)
        if mask is not None:
            survival = survival * self._normalize_mask(mask, x)

        if self.mode == "attention":
            return self._attention_fold(x, survival)

        if self.assignment is None:
            raise RuntimeError("soft Fold path is not initialized")
        logits = self.assignment(x) / self.temperature
        logits = logits + survival.clamp_min(self.eps).log()
        return _fold_weighted_sum(x * survival, logits, self.dropout, self.topk)

    def _normalize_q(self, q: Tensor, x: Tensor) -> Tensor:
        if q.ndim == 2:
            q = q.unsqueeze(-1)
        if q.ndim != 3 or q.shape[:2] != x.shape[:2] or q.shape[-1] != 1:
            raise ValueError("q must have shape [B, N] or [B, N, 1]")
        return q.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)

    def _normalize_mask(self, mask: Tensor, x: Tensor) -> Tensor:
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)
        if mask.ndim != 3 or mask.shape[:2] != x.shape[:2] or mask.shape[-1] != 1:
            raise ValueError("mask must have shape [B, N] or [B, N, 1]")
        return mask.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)

    def _attention_fold(self, x: Tensor, survival: Tensor) -> Tensor:
        if self.dim is None:
            raise RuntimeError("attention Fold requires a static dim")
        batch, tokens, dim = x.shape
        head_dim = dim // self.heads
        query = self.query.to(device=x.device, dtype=x.dtype).unsqueeze(0).expand(batch, -1, -1)
        query = query.reshape(batch, self.k, self.heads, head_dim).transpose(1, 2)
        key = self.key(x).reshape(batch, tokens, self.heads, head_dim).transpose(1, 2)
        value = self.value(x * survival).reshape(batch, tokens, self.heads, head_dim).transpose(1, 2)
        bias = survival.clamp_min(self.eps).log().reshape(batch, 1, 1, tokens)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=bias,
            dropout_p=self.dropout.p if self.training else 0.0,
        )
        attended = attended.transpose(1, 2).reshape(batch, self.k, dim)
        return self.output(attended)

    def extra_repr(self) -> str:
        args = [f"k={self.k}"]
        if self.dim is not None:
            args.append(f"dim={self.dim}")
        if self.hidden_dim is not None:
            args.append(f"hidden_dim={self.hidden_dim}")
        if self.temperature != 1.0:
            args.append(f"temperature={self.temperature:g}")
        if self.dropout.p != 0.0:
            args.append(f"dropout={self.dropout.p:g}")
        if self.mode != "soft":
            args.append(f"mode={self.mode!r}")
        if self.topk is not None:
            args.append(f"topk={self.topk}")
        if self.heads != 1:
            args.append(f"heads={self.heads}")
        return ", ".join(args)


class LearnedPulse(nn.Module):
    """Alpha learned latent pulse formation.

    ``LearnedPulse`` forms a compact ``[B, K, D]`` pulse workspace from
    overcomplete latent fragments ``x: [B, N, D]``. It applies an optional
    trainable fragment projection, ``Half`` survival pressure, and ``Fold``
    compaction. Pass ``mask`` for padding/valid-slot masks and ``q`` only for
    salience or survival guidance. ``Pulse`` is the default public alias for
    this layer. The older externally indexed pulse compressor remains available
    as the legacy explicit path for token-bound pulse ids.
    """

    def __init__(
        self,
        k: int,
        *,
        dim: int | None = None,
        hidden_dim: int | None = None,
        refine: bool = False,
        refine_mode: str = "mlp",
        dropout: float = 0.0,
        fold_mode: str = "soft",
        fold_topk: int | None = None,
        fold_heads: int = 1,
        q_topk: int | None = None,
        use_half: bool = True,
    ) -> None:
        super().__init__()
        if dim is not None and dim <= 0:
            raise ValueError("dim must be positive")
        if hidden_dim is not None and hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if refine and dim is None:
            raise ValueError("dim must be provided when refine=True")
        if refine_mode not in {"mlp", "gated"}:
            raise ValueError("refine_mode must be 'mlp' or 'gated'")
        if fold_mode not in {"soft", "attention"}:
            raise ValueError("fold_mode must be 'soft' or 'attention'")
        if fold_mode == "attention" and dim is None:
            raise ValueError("dim must be provided when fold_mode='attention'")
        if q_topk is not None and q_topk <= 0:
            raise ValueError("q_topk must be positive")
        self.k = int(k)
        self.dim = None if dim is None else int(dim)
        self.hidden_dim = None if hidden_dim is None else int(hidden_dim)
        self.refine_enabled = bool(refine)
        self.refine_mode = refine_mode
        self.fold_mode = fold_mode
        self.fold_topk = None if fold_topk is None else int(fold_topk)
        self.q_topk = None if q_topk is None else int(q_topk)
        self.use_half = bool(use_half)
        self.temperature = 1.0
        self.eps = 1e-6
        self.dropout = nn.Dropout(dropout)

        use_shared_trunk = dim is not None and hidden_dim is not None and fold_mode == "soft"

        if dim is None:
            self.fragment_proj: nn.Module = _LazySameDimProjection(hidden_dim)
            self.shared_trunk: nn.Module | None = None
            self.fragment_head: nn.Module | None = None
            self.assignment_head: nn.Module | None = None
            self.fold: Fold | None = Fold(k=k, dim=dim, hidden_dim=hidden_dim, dropout=dropout, topk=fold_topk)
        elif use_shared_trunk:
            self.shared_trunk = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU())
            self.fragment_head = nn.Linear(hidden_dim, dim)
            self.assignment_head = nn.Linear(hidden_dim, k)
            self.fragment_proj = nn.Identity()
            self.fold = None
        elif hidden_dim is None:
            self.fragment_proj = nn.Linear(dim, dim)
            self.shared_trunk = None
            self.fragment_head = None
            self.assignment_head = None
            self.fold = Fold(k=k, dim=dim, hidden_dim=hidden_dim, dropout=dropout, mode=fold_mode, topk=fold_topk, heads=fold_heads)
        else:
            self.fragment_proj = nn.Sequential(nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim))
            self.shared_trunk = None
            self.fragment_head = None
            self.assignment_head = None
            self.fold = Fold(k=k, dim=dim, hidden_dim=hidden_dim, dropout=dropout, mode=fold_mode, topk=fold_topk, heads=fold_heads)

        self.half_act = Half() if self.use_half else nn.Identity()
        if refine and dim is not None and refine_mode == "mlp":
            self.refine = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        elif refine and dim is not None:
            self.refine = _GatedRefineDelta(dim, hidden_dim)
        else:
            self.refine = None

    def forward(
        self,
        x: Tensor,
        q: Tensor | None = None,
        *,
        mask: Tensor | None = None,
        return_info: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        if x.ndim != 3:
            raise ValueError("x must have shape [B, N, D]")
        if self.dim is not None and x.shape[-1] != self.dim:
            raise ValueError(f"expected x.shape[-1] == {self.dim}, got {x.shape[-1]}")
        mask_norm = None if mask is None else self._normalize_mask(mask, x).squeeze(-1)
        if q is not None and self.q_topk is not None:
            x, q, mask_norm = self._select_q_topk(x, q, self.q_topk, mask=mask_norm)

        assignment_logits = None
        if self.shared_trunk is not None and self.fragment_head is not None and self.assignment_head is not None:
            trunk = self.shared_trunk(x)
            fragments = self.fragment_head(trunk)
            assignment_logits = self.assignment_head(trunk) / self.temperature
        else:
            fragments = self.fragment_proj(x)
        survived = self.half_act(fragments)
        if q is not None:
            guide = self._normalize_q(q, fragments).squeeze(-1)
            survival = (
                survived.norm(dim=-1) / fragments.norm(dim=-1).clamp_min(1e-6)
                if return_info
                else guide
            )
        else:
            survival = survived.norm(dim=-1) / fragments.norm(dim=-1).clamp_min(1e-6)
            guide = survival
        if mask_norm is not None:
            guide = guide * mask_norm.to(device=fragments.device, dtype=fragments.dtype)
            survival = survival * mask_norm.to(device=fragments.device, dtype=fragments.dtype)
        if assignment_logits is None:
            if self.fold is None:
                raise RuntimeError("Fold path is not initialized")
            pulses = self.fold(survived, q=guide)
        else:
            pulses = self._fold_with_logits(survived, guide, assignment_logits)
        if self.refine is not None:
            pulses = pulses + self.refine(pulses)
        if not return_info:
            return pulses
        info = {
            "survival_mean": survival.mean().detach(),
            "survival_min": survival.min().detach(),
            "survival_max": survival.max().detach(),
            "fragment_norm": fragments.norm(dim=-1).mean().detach(),
            "pulse_norm": pulses.norm(dim=-1).mean().detach(),
        }
        return pulses, info

    def _normalize_q(self, q: Tensor, x: Tensor) -> Tensor:
        if q.ndim == 2:
            q = q.unsqueeze(-1)
        if q.ndim != 3 or q.shape[:2] != x.shape[:2] or q.shape[-1] != 1:
            raise ValueError("q must have shape [B, N] or [B, N, 1]")
        return q.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)

    def _normalize_mask(self, mask: Tensor, x: Tensor) -> Tensor:
        if mask.ndim == 2:
            mask = mask.unsqueeze(-1)
        if mask.ndim != 3 or mask.shape[:2] != x.shape[:2] or mask.shape[-1] != 1:
            raise ValueError("mask must have shape [B, N] or [B, N, 1]")
        return mask.to(device=x.device, dtype=x.dtype).clamp(0.0, 1.0)

    def _select_q_topk(self, x: Tensor, q: Tensor, topk: int, *, mask: Tensor | None = None) -> tuple[Tensor, Tensor, Tensor | None]:
        q_norm = self._normalize_q(q, x).squeeze(-1)
        if topk >= x.shape[1]:
            return x, q_norm, mask
        scores = q_norm if mask is None else q_norm.masked_fill(mask.to(device=x.device, dtype=torch.bool) == 0, -torch.inf)
        values, indices = scores.topk(topk, dim=1)
        values = torch.where(torch.isfinite(values), q_norm.gather(1, indices), torch.zeros_like(values))
        selected = x.gather(1, indices.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
        selected_mask = None if mask is None else mask.gather(1, indices)
        return selected, values, selected_mask

    def _fold_with_logits(self, x: Tensor, q: Tensor, assignment_logits: Tensor) -> Tensor:
        survival = self._normalize_q(q, x)
        logits = assignment_logits + survival.clamp_min(self.eps).log()
        return _fold_weighted_sum(x * survival, logits, self.dropout, self.fold_topk)

    def extra_repr(self) -> str:
        args = [f"k={self.k}"]
        if self.dim is not None:
            args.append(f"dim={self.dim}")
        if self.hidden_dim is not None:
            args.append(f"hidden_dim={self.hidden_dim}")
        if self.refine_enabled:
            args.append("refine=True")
        if self.refine_enabled and self.refine_mode != "mlp":
            args.append(f"refine_mode={self.refine_mode!r}")
        if self.fold_mode != "soft":
            args.append(f"fold_mode={self.fold_mode!r}")
        if self.fold_topk is not None:
            args.append(f"fold_topk={self.fold_topk}")
        if self.q_topk is not None:
            args.append(f"q_topk={self.q_topk}")
        if not self.use_half:
            args.append("use_half=False")
        return ", ".join(args)


class RecallRefiner(nn.Module):
    """Alpha iterative latent recall refinement.

    ``RecallRefiner`` repeatedly asks a recall layer for candidate corrections,
    optionally thins those corrections with ``Half``, and applies scaled
    residual updates to the hidden state.
    """

    def __init__(
        self,
        recall_layer: nn.Module,
        *,
        steps: int = 3,
        step_scale: float | list[float] | tuple[float, ...] | Tensor = 1.0,
        learnable_step_scale: bool = False,
        use_half: bool = True,
        activation: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if steps < 0:
            raise ValueError("steps must be non-negative")
        self.recall_layer = recall_layer
        self.steps = int(steps)
        self.learnable_step_scale = bool(learnable_step_scale)
        self.activation = Half() if activation is None and use_half else activation

        scale = torch.as_tensor(step_scale, dtype=torch.float32)
        if scale.ndim == 0:
            scale = scale.repeat(max(1, self.steps))
        elif scale.ndim != 1:
            raise ValueError("step_scale must be a scalar or one-dimensional sequence")
        if scale.numel() == 0:
            raise ValueError("step_scale must contain at least one value")
        if learnable_step_scale:
            self.step_scale = nn.Parameter(scale.clone())
        else:
            self.register_buffer("step_scale", scale.clone())

    def forward(
        self,
        h: Tensor,
        *args,
        steps: int | None = None,
        tolerance: float | None = None,
        return_info: bool = False,
        record_history: bool = False,
        **kwargs,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        run_steps = self.steps if steps is None else int(steps)
        if run_steps < 0:
            raise ValueError("steps must be non-negative")
        if tolerance is not None and tolerance < 0:
            raise ValueError("tolerance must be non-negative")

        state = h
        start_state = h
        delta_norms: list[Tensor] = []
        raw_delta_norms: list[Tensor] = []
        update_norms: list[Tensor] = []
        survival_means: list[Tensor] = []
        applied_scales: list[Tensor] = []
        history: list[Tensor] = [state.detach()] if record_history else []
        stopped_early = False

        for step_index in range(run_steps):
            raw_delta = self._call_recall(state, *args, **kwargs)
            delta = self.activation(raw_delta) if self.activation is not None else raw_delta
            scale = self._scale_for_step(step_index, state)
            update = scale * delta
            state = state + update

            raw_norm = raw_delta.norm().detach()
            delta_norm = delta.norm().detach()
            update_norm = update.norm().detach()
            raw_delta_norms.append(raw_norm)
            delta_norms.append(delta_norm)
            update_norms.append(update_norm)
            survival_means.append((delta.abs().mean() / raw_delta.abs().mean().clamp_min(1e-6)).detach())
            applied_scales.append(scale.detach().mean())
            if record_history:
                history.append(state.detach())
            if tolerance is not None and float(update_norm) <= tolerance:
                stopped_early = True
                break

        if not return_info:
            return state
        info = {
            "steps": torch.as_tensor(len(delta_norms), device=h.device),
            "stopped_early": torch.as_tensor(stopped_early, device=h.device),
            "delta_norm": self._stack_or_empty(delta_norms, h),
            "raw_delta_norm": self._stack_or_empty(raw_delta_norms, h),
            "update_norm": self._stack_or_empty(update_norms, h),
            "survival_mean": self._stack_or_empty(survival_means, h),
            "step_scale": self._stack_or_empty(applied_scales, h),
            "state_change_norm": (state - start_state).norm().detach(),
        }
        if record_history:
            info["state_history"] = torch.stack(history)
        return state, info

    def _call_recall(self, h: Tensor, *args, **kwargs) -> Tensor:
        output = self.recall_layer(h, *args, **kwargs)
        if isinstance(output, Tensor):
            return output
        if hasattr(output, "y") and isinstance(output.y, Tensor):
            return output.y
        if isinstance(output, tuple) and output and isinstance(output[0], Tensor):
            return output[0]
        raise TypeError("recall_layer must return a Tensor, a tuple whose first item is a Tensor, or an object with Tensor attribute 'y'")

    def _scale_for_step(self, step_index: int, h: Tensor) -> Tensor:
        scale = self.step_scale[min(step_index, self.step_scale.shape[0] - 1)]
        return scale.to(device=h.device, dtype=h.dtype)

    def _stack_or_empty(self, values: list[Tensor], like: Tensor) -> Tensor:
        if not values:
            return torch.empty(0, device=like.device, dtype=like.dtype)
        return torch.stack([value.to(device=like.device, dtype=like.dtype) for value in values])

    def extra_repr(self) -> str:
        args = [f"steps={self.steps}"]
        if self.learnable_step_scale:
            args.append("learnable_step_scale=True")
        if self.activation is None:
            args.append("use_half=False")
        return ", ".join(args)


Pulse = LearnedPulse

from .visual_scan import PixelShiftObservation, VisualScan, VisualScanConfig, VisualScanOutput
__all__ = ["Layer", "Half", "Fold", "Pulse", "LearnedPulse", "RecallRefiner", "VisualField", "VisualFieldOutput", "concat_visual_fields", "VisualScan", "VisualScanConfig", "VisualScanOutput", "PixelShiftObservation"]


def __getattr__(name: str):
    if name == "Layer":
        from .usage import Layer

        return Layer
    raise AttributeError(name)
