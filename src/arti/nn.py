"""Activation-style and compact neural network modules."""

from __future__ import annotations

from functools import lru_cache
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .functional import half
from .visual_field import VisualField, VisualFieldOutput, concat_visual_fields


@lru_cache(maxsize=1)
def _triton_palette_is_available() -> bool:
    try:
        from ._triton import is_available
    except (ImportError, OSError):
        return False
    return is_available()


@lru_cache(maxsize=None)
def _device_supports_triton_palette(device_index: int) -> bool:
    return torch.cuda.get_device_capability(device_index) >= (8, 0)


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


class _HardLayoutSoftGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, hard: Tensor, soft: Tensor) -> Tensor:
        return hard

    @staticmethod
    def backward(ctx: object, gradient: Tensor) -> tuple[Tensor, Tensor]:
        return gradient, gradient


class UnFold(nn.Module):
    """Expand and rearrange a tensor while preserving every input value.

    ``UnFold`` queries ``exposed`` new values from the input, combines them
    with the original values, and learns a sample-conditioned layout. The
    forward path uses a hard gather: every original input instance appears
    exactly once, but its position and adjacency may change. A soft layout is
    used only as a surrogate gradient during training.

    This layer is unrelated to :class:`torch.nn.Unfold`, which extracts image
    patches.
    """

    def __init__(
        self,
        dim: int,
        exposed: int = 1,
        *,
        guide_dim: int | None = None,
        condition_dim: int | None = None,
        hidden_dim: int | None = None,
        temperature: float = 0.25,
        sinkhorn_steps: int = 16,
        max_length: int = 128,
        hard_backend: str = "sort",
        layout_mode: str = "learned",
        value_operators: int = 8,
        value_rank: int | None = None,
        query_chunk_size: int | None = None,
        operator_chunk_size: int | None = None,
    ) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        if exposed <= 0:
            raise ValueError("exposed must be positive")
        if guide_dim is not None and guide_dim <= 0:
            raise ValueError("guide_dim must be positive")
        if condition_dim is not None and condition_dim <= 0:
            raise ValueError("condition_dim must be positive")
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if sinkhorn_steps <= 0:
            raise ValueError("sinkhorn_steps must be positive")
        if max_length <= 0:
            raise ValueError("max_length must be positive")
        if hard_backend not in {"sort", "greedy", "auction"}:
            raise ValueError("hard_backend must be 'sort', 'greedy', or 'auction'")
        if hard_backend != "sort" and max_length <= exposed:
            raise ValueError("max_length must be greater than exposed for dense backends")
        if layout_mode not in {"learned", "canonical"}:
            raise ValueError("layout_mode must be 'learned' or 'canonical'")
        if layout_mode != "learned" and guide_dim != 1:
            raise ValueError("canonical layout mode requires guide_dim=1")
        if layout_mode == "canonical" and hard_backend != "sort":
            raise ValueError("canonical layout mode requires hard_backend='sort'")
        if value_operators <= 0:
            raise ValueError("value_operators must be positive")
        if value_rank is not None and value_rank <= 0:
            raise ValueError("value_rank must be positive")
        if query_chunk_size is not None and query_chunk_size <= 0:
            raise ValueError("query_chunk_size must be positive")
        if operator_chunk_size is not None and operator_chunk_size <= 0:
            raise ValueError("operator_chunk_size must be positive")
        hidden = max(8, min(64, dim * 2)) if hidden_dim is None else hidden_dim
        if hidden <= 0:
            raise ValueError("hidden_dim must be positive")
        self.dim = int(dim)
        self.exposed = int(exposed)
        self.guide_dim = None if guide_dim is None else int(guide_dim)
        self.condition_dim = None if condition_dim is None else int(condition_dim)
        self.hidden_dim = int(hidden)
        self.temperature = float(temperature)
        self.sinkhorn_steps = int(sinkhorn_steps)
        self.max_length = int(max_length)
        self.hard_backend = hard_backend
        self.layout_mode = layout_mode
        self.value_operators = int(value_operators)
        self.value_rank = value_rank
        self.query_chunk_size = query_chunk_size
        self.operator_chunk_size = operator_chunk_size
        # Internal execution policy. Auto remains conservative and falls back
        # to the established differentiable slot path outside its measured gate.
        self._operator_schedule = "auto"

        self.exposed_queries = nn.Parameter(torch.empty(self.exposed, self.hidden_dim))
        self.exposed_value_operators = (
            nn.Parameter(torch.empty(self.value_operators, self.dim, self.dim))
            if value_rank is None
            else None
        )
        self.exposed_value_left = (
            None
            if value_rank is None
            else nn.Parameter(torch.empty(self.value_operators, self.dim, value_rank))
        )
        self.exposed_value_right = (
            None
            if value_rank is None
            else nn.Parameter(torch.empty(self.value_operators, value_rank, self.dim))
        )
        self.exposed_value_mix = nn.Parameter(
            torch.empty(self.exposed, self.value_operators)
        )
        self.exposed_scale = nn.Parameter(torch.empty(self.exposed, self.dim))
        self.exposed_bias = nn.Parameter(torch.empty(self.exposed, self.dim))
        self.content_key = nn.Linear(self.dim, self.hidden_dim, bias=False)
        self.guide_proj = None if guide_dim is None else nn.Linear(guide_dim, self.dim, bias=False)
        self.condition_proj = (
            None if condition_dim is None else nn.Linear(condition_dim, self.dim, bias=False)
        )
        self.exposed_guide = (
            None if guide_dim is None else nn.Parameter(torch.empty(self.exposed, guide_dim))
        )
        coordinate_input_dim = self.dim + self.hidden_dim
        if condition_dim is not None:
            coordinate_input_dim += condition_dim
        self.exposed_coordinate = (
            nn.Sequential(
                nn.Linear(coordinate_input_dim, self.hidden_dim),
                nn.GELU(),
                nn.Linear(self.hidden_dim, 1),
            )
            if layout_mode == "canonical"
            else None
        )
        # Candidate layout must not depend on the storage order of the input.
        # A candidate sees its own value and a symmetric masked summary.
        layout_input_dim = self.dim * 2
        if guide_dim is not None:
            layout_input_dim += self.dim
        if condition_dim is not None:
            layout_input_dim += self.dim
        self.layout_score = nn.Sequential(
            nn.Linear(layout_input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.layout_slots = nn.Sequential(
            nn.Linear(1, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.rank_score = nn.Linear(self.hidden_dim, 1)
        self.reset_parameters()

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    def reset_parameters(self) -> None:
        nn.init.normal_(self.exposed_queries, std=0.02)
        if self.exposed_value_operators is not None:
            nn.init.xavier_uniform_(self.exposed_value_operators)
        else:
            assert self.exposed_value_left is not None
            assert self.exposed_value_right is not None
            nn.init.xavier_uniform_(self.exposed_value_left)
            nn.init.xavier_uniform_(self.exposed_value_right)
        nn.init.normal_(
            self.exposed_value_mix,
            std=self.value_operators**-0.5,
        )
        nn.init.normal_(self.exposed_scale, mean=1.0, std=0.02)
        nn.init.normal_(self.exposed_bias, std=0.02)
        self.content_key.reset_parameters()
        if self.guide_proj is not None:
            self.guide_proj.reset_parameters()
            assert self.exposed_guide is not None
            nn.init.normal_(self.exposed_guide, std=0.02)
        if self.condition_proj is not None:
            self.condition_proj.reset_parameters()
        coordinate_modules = (
            () if self.exposed_coordinate is None else tuple(self.exposed_coordinate)
        )
        for module in (
            *self.layout_score,
            *self.layout_slots,
            self.rank_score,
            *coordinate_modules,
        ):
            if isinstance(module, nn.Linear):
                module.reset_parameters()

    def _query_weights(
        self,
        keys: Tensor,
        mask: Tensor,
        start: int,
        stop: int,
    ) -> Tensor:
        queries = self.exposed_queries[start:stop]
        logits = torch.einsum("eh,bnh->ben", queries, keys)
        logits = logits / self.hidden_dim**0.5
        weights = torch.softmax(logits.masked_fill(~mask[:, None], -1e4), dim=-1)
        weights = weights * mask[:, None].to(weights.dtype)
        return weights / weights.sum(-1, keepdim=True).clamp_min(
            torch.finfo(weights.dtype).eps
        )

    def _query_context(self, x: Tensor, mask: Tensor) -> Tensor:
        keys = self.content_key(x)
        chunk_size = (
            self.exposed if self.query_chunk_size is None else self.query_chunk_size
        )
        attended_chunks = []
        for start in range(0, self.exposed, chunk_size):
            stop = min(start + chunk_size, self.exposed)
            weights = self._query_weights(keys, mask, start, stop)
            attended_chunks.append(torch.bmm(weights, x))
        return torch.cat(attended_chunks, dim=1)

    def _form_exposed_values(self, attended: Tensor) -> Tensor:
        if self.value_rank is not None:
            assert self.exposed_value_left is not None
            assert self.exposed_value_right is not None
            low_rank = torch.einsum(
                "bed,rdp->berp",
                attended,
                self.exposed_value_left,
            )
            low_rank = low_rank * self.exposed_value_mix[None, :, :, None]
            values = torch.einsum(
                "berp,rph->beh",
                low_rank,
                self.exposed_value_right,
            )
            return values * self.exposed_scale + self.exposed_bias
        assert self.exposed_value_operators is not None
        chunk_size = (
            self.value_operators
            if self.operator_chunk_size is None
            else self.operator_chunk_size
        )
        values = torch.zeros_like(attended)
        for start in range(0, self.value_operators, chunk_size):
            stop = min(start + chunk_size, self.value_operators)
            transformed = torch.einsum(
                "bed,rdh->berh",
                attended,
                self.exposed_value_operators[start:stop],
            )
            values = values + (
                transformed
                * self.exposed_value_mix[None, :, start:stop, None]
            ).sum(dim=2)
        return values * self.exposed_scale + self.exposed_bias

    def _form_exposed_values_effective(self, attended: Tensor) -> Tensor:
        """Apply the full operator bank after exact per-slot materialization."""

        assert self.exposed_value_operators is not None
        # Bound the temporary [E_chunk, D, D] tensor to roughly 8 MiB. This is
        # an execution detail, not a persistent inference cache.
        bytes_per_value = attended.element_size()
        budget_elements = max(1, (8 * 1024 * 1024) // bytes_per_value)
        effective_chunk = max(1, budget_elements // (self.dim * self.dim))
        if self.query_chunk_size is not None:
            effective_chunk = min(effective_chunk, self.query_chunk_size)
        chunk_size = min(self.exposed, effective_chunk)
        chunks = []
        for start in range(0, self.exposed, chunk_size):
            stop = min(start + chunk_size, self.exposed)
            effective = torch.einsum(
                "er,rdh->edh",
                self.exposed_value_mix[start:stop],
                self.exposed_value_operators,
            )
            values = torch.einsum(
                "bed,edh->beh",
                attended[:, start:stop],
                effective,
            )
            chunks.append(
                values * self.exposed_scale[start:stop]
                + self.exposed_bias[start:stop]
            )
        return torch.cat(chunks, dim=1)

    def _query_exposed_palette(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        """Move the exact operator bank to the input workspace before emission."""

        assert self.exposed_value_operators is not None
        keys = self.content_key(x)
        transformed = torch.einsum(
            "bnd,rdh->bnrh",
            x,
            self.exposed_value_operators,
        ).permute(0, 2, 1, 3)
        chunk_size = (
            self.exposed if self.query_chunk_size is None else self.query_chunk_size
        )
        exposed_chunks = []
        attended_chunks = []
        for start in range(0, self.exposed, chunk_size):
            stop = min(start + chunk_size, self.exposed)
            weights = self._query_weights(keys, mask, start, stop)
            attended_chunks.append(torch.bmm(weights, x))
            operator_context = torch.matmul(weights[:, None], transformed)
            values = (
                operator_context
                * self.exposed_value_mix[start:stop].T[None, :, :, None]
            ).sum(dim=1)
            exposed_chunks.append(
                values * self.exposed_scale[start:stop]
                + self.exposed_bias[start:stop]
            )
        return torch.cat(exposed_chunks, dim=1), torch.cat(attended_chunks, dim=1)

    def _query_exposed_triton_palette(
        self,
        x: Tensor,
        mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        assert self.exposed_value_operators is not None
        from ._triton.unfold import _palette_values_unchecked

        keys = self.content_key(x)
        weights = self._query_weights(keys, mask, 0, self.exposed)
        attended = torch.bmm(weights, x)
        exposed = _palette_values_unchecked(
            x,
            weights,
            self.exposed_value_operators,
            self.exposed_value_mix,
            self.exposed_scale,
            self.exposed_bias,
        )
        return exposed, attended

    def _estimate_operator_schedule(self, batch: int, length: int) -> str:
        if self.value_rank is not None:
            return "slot"

        # Common query-key work is omitted because it is identical. The model
        # compares value aggregation and operator contractions only.
        b = batch
        n = length
        e = self.exposed
        d = self.dim
        r = self.value_operators
        slot_cost = b * e * n * d + b * e * r * d * d
        effective_cost = b * e * n * d + e * r * d * d + b * e * d * d
        palette_cost = (
            b * n * r * d * d
            + b * e * n * r * d
            + b * e * n * d
        )
        costs = {
            "slot": slot_cost,
            "palette": palette_cost,
            "effective": effective_cost,
        }
        candidate = min(costs, key=costs.__getitem__)
        # FLOP estimates do not capture grouped-GEMM utilization and launch
        # overhead. Require a material analytic margin before leaving the
        # established slot path.
        return candidate if costs[candidate] * 4 <= slot_cost * 3 else "slot"

    def _can_use_triton_palette(self, x: Tensor) -> bool:
        compiler = getattr(torch, "compiler", None)
        is_compiling = bool(compiler is not None and compiler.is_compiling())
        if (
            self.training
            or torch.is_grad_enabled()
            or is_compiling
            or not x.is_cuda
            or x.dtype not in {torch.float32, torch.bfloat16}
            or self.value_rank is not None
            or self.query_chunk_size is not None
            or self.dim < 128
            or self.exposed < 128
            or x.shape[1] > 128
            or self.value_operators > 16
        ):
            return False
        device_index = x.device.index
        if device_index is None:
            device_index = torch.cuda.current_device()
        if not _device_supports_triton_palette(device_index):
            return False
        return _triton_palette_is_available()

    def _select_operator_schedule(self, x: Tensor) -> str:
        if self._operator_schedule == "auto":
            return "palette_triton" if self._can_use_triton_palette(x) else "slot"
        if self._operator_schedule not in {
            "slot",
            "palette",
            "effective",
            "palette_triton",
        }:
            raise RuntimeError(
                f"invalid private operator schedule: {self._operator_schedule!r}"
            )
        if self._operator_schedule == "palette_triton" and not self._can_use_triton_palette(x):
            raise RuntimeError("private Triton palette schedule is unavailable for this input")
        return self._operator_schedule

    def _query_exposed(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        schedule = self._select_operator_schedule(x)
        if schedule == "palette_triton":
            return self._query_exposed_triton_palette(x, mask)
        if schedule == "palette":
            return self._query_exposed_palette(x, mask)
        attended = self._query_context(x, mask)
        if schedule == "effective":
            if self.value_rank is not None:
                raise RuntimeError("effective schedule requires a full operator bank")
            return self._form_exposed_values_effective(attended), attended
        return self._form_exposed_values(attended), attended

    def _soft_layout(self, logits: Tensor) -> Tensor:
        log_layout = (logits / self.temperature).float()
        for _ in range(self.sinkhorn_steps):
            log_layout = log_layout - torch.logsumexp(log_layout, dim=-1, keepdim=True)
            log_layout = log_layout - torch.logsumexp(log_layout, dim=-2, keepdim=True)
        return log_layout.exp().to(logits.dtype)

    @staticmethod
    def _normalize_guide(guide: Tensor, mask: Tensor) -> Tensor:
        stats_dtype = torch.float64 if guide.dtype == torch.float64 else torch.float32
        values = guide.to(stats_dtype)
        valid = mask[..., None]
        weights = valid.to(stats_dtype)
        count = weights.sum(dim=1, keepdim=True).clamp_min(1)
        masked_values = torch.where(valid, values, torch.zeros_like(values))
        mean = masked_values.sum(dim=1, keepdim=True) / count
        centered = torch.where(valid, values - mean, torch.zeros_like(values))
        variance = centered.square().sum(dim=1, keepdim=True) / count
        normalized = centered / variance.sqrt().clamp_min(torch.finfo(stats_dtype).eps**0.5)
        return normalized.to(guide.dtype)

    def _soft_rank_layout(self, scores: Tensor) -> Tensor:
        pairwise = torch.sigmoid((scores[:, :, None] - scores[:, None, :]) / self.temperature)
        soft_rank = pairwise.sum(dim=-1) - 0.5
        anchors = torch.arange(scores.shape[1], device=scores.device, dtype=scores.dtype)
        log_layout = -(anchors[None, :, None] - soft_rank[:, None, :]).square()
        log_layout = log_layout.float()
        for _ in range(self.sinkhorn_steps):
            log_layout = log_layout - torch.logsumexp(log_layout, dim=-1, keepdim=True)
            log_layout = log_layout - torch.logsumexp(log_layout, dim=-2, keepdim=True)
        return log_layout.exp().to(scores.dtype)

    @staticmethod
    def _greedy_matching(logits: Tensor) -> Tensor:
        """Global-edge greedy baseline retained for assignment diagnostics."""

        batch, total, _ = logits.shape
        available_rows = torch.ones(batch, total, dtype=torch.bool, device=logits.device)
        available_columns = torch.ones_like(available_rows)
        source = torch.full((batch, total), -1, dtype=torch.long, device=logits.device)
        scores = logits.detach()
        batch_index = torch.arange(batch, device=logits.device)
        for _ in range(total):
            available = available_rows[:, :, None] & available_columns[:, None, :]
            selected = scores.masked_fill(~available, -torch.inf).flatten(1).argmax(dim=1)
            row = selected // total
            column = selected % total
            source[batch_index, row] = column
            available_rows[batch_index, row] = False
            available_columns[batch_index, column] = False
        return source

    @staticmethod
    def _auction_matching(logits: Tensor) -> Tensor:
        """Batched auction matching used by the hard forward path."""

        batch, total, _ = logits.shape
        scores = logits.float().detach()
        prices = torch.zeros(batch, total, device=logits.device)
        owner = torch.full((batch, total), -1, dtype=torch.long, device=logits.device)
        source = torch.full_like(owner, -1)
        batch_index = torch.arange(batch, device=logits.device)
        epsilon = 1e-2
        for _ in range(100 * total * total):
            unmatched = source < 0
            if not unmatched.any():
                break
            row = unmatched.to(torch.int64).argmax(dim=1)
            active = unmatched.any(dim=1)
            utility = scores[batch_index, row] - prices
            best_two = utility.topk(2, dim=1)
            column = best_two.indices[:, 0]
            bid = best_two.values[:, 0] - best_two.values[:, 1] + epsilon
            previous = owner[batch_index, column]
            active_batch = batch_index[active]
            active_column = column[active]
            active_row = row[active]
            displaced = previous[active]
            displaced_mask = displaced >= 0
            if displaced_mask.any():
                source[active_batch[displaced_mask], displaced[displaced_mask]] = -1
            prices[active_batch, active_column] += bid[active]
            owner[active_batch, active_column] = active_row
            source[active_batch, active_row] = active_column
        if (source < 0).any():
            raise RuntimeError("UnFold auction matching did not converge")
        return source

    def _hard_matching(self, logits: Tensor) -> Tensor:
        if self.hard_backend == "auction":
            return self._auction_matching(logits)
        return self._greedy_matching(logits)

    def forward(
        self,
        x: Tensor,
        mask: Tensor | None = None,
        *,
        guide: Tensor | None = None,
        exposed_guide: Tensor | None = None,
        exposed_mask: Tensor | None = None,
        condition: Tensor | None = None,
        return_exposed_mask: bool = False,
        return_source_index: bool = False,
        return_soft_layout: bool = False,
    ) -> Tensor | tuple[Tensor, ...]:
        if x.ndim < 2:
            raise ValueError("x must have shape [..., N, D]")
        leading_shape = x.shape[:-2]
        length, dim = x.shape[-2:]
        if length == 0:
            raise ValueError("x sequence length must be positive")
        if dim != self.dim:
            raise ValueError(f"x feature dimension {dim} does not match dim={self.dim}")
        if x.device != self.exposed_bias.device or x.dtype != self.exposed_bias.dtype:
            raise ValueError("x and UnFold parameters must have the same device and dtype")
        if mask is not None:
            if mask.dtype != torch.bool:
                raise ValueError("mask must be a boolean tensor")
            if mask.shape != x.shape[:-1]:
                raise ValueError(f"mask must have shape {list(x.shape[:-1])}")
            if mask.device != x.device:
                raise ValueError("mask and x must be on the same device")
        if guide is not None:
            if self.guide_dim is None:
                raise ValueError("guide was provided but guide_dim is disabled")
            expected_guide_shape = (*x.shape[:-1], self.guide_dim)
            if guide.shape != expected_guide_shape:
                raise ValueError(f"guide must have shape {list(expected_guide_shape)}")
            if not guide.is_floating_point():
                raise ValueError("guide must be a floating-point tensor")
            if guide.device != x.device:
                raise ValueError("guide and x must be on the same device")
            if not torch.isfinite(guide).all():
                raise ValueError("guide must contain only finite values")
        if exposed_guide is not None:
            if self.layout_mode != "canonical":
                raise ValueError("exposed_guide is only supported by canonical layout")
            expected_exposed_guide_shape = (*leading_shape, self.exposed, 1)
            if exposed_guide.shape != expected_exposed_guide_shape:
                raise ValueError(
                    f"exposed_guide must have shape {list(expected_exposed_guide_shape)}"
                )
            if not exposed_guide.is_floating_point():
                raise ValueError("exposed_guide must be a floating-point tensor")
            if exposed_guide.device != x.device:
                raise ValueError("exposed_guide and x must be on the same device")
            if not torch.isfinite(exposed_guide).all():
                raise ValueError("exposed_guide must contain only finite values")
        if exposed_mask is not None:
            expected_exposed_mask_shape = (*leading_shape, self.exposed)
            if exposed_mask.shape != expected_exposed_mask_shape:
                raise ValueError(
                    f"exposed_mask must have shape {list(expected_exposed_mask_shape)}"
                )
            if exposed_mask.dtype != torch.bool:
                raise ValueError("exposed_mask must be a boolean tensor")
            if exposed_mask.device != x.device:
                raise ValueError("exposed_mask and x must be on the same device")
        if condition is not None:
            if self.condition_dim is None:
                raise ValueError("condition was provided but condition_dim is disabled")
            compact_shape = (*leading_shape, self.condition_dim)
            singleton_shape = (*leading_shape, 1, self.condition_dim)
            if condition.shape not in {compact_shape, singleton_shape}:
                raise ValueError(
                    f"condition must have shape {list(compact_shape)} or {list(singleton_shape)}"
                )
            if not condition.is_floating_point():
                raise ValueError("condition must be a floating-point tensor")
            if condition.device != x.device:
                raise ValueError("condition and x must be on the same device")
            if not torch.isfinite(condition).all():
                raise ValueError("condition must contain only finite values")

        flat_x = x.reshape(-1, length, dim)
        batch = flat_x.shape[0]
        flat_mask = (
            torch.ones(batch, length, dtype=torch.bool, device=x.device)
            if mask is None
            else mask.reshape(batch, length)
        )
        exposed, attended = self._query_exposed(flat_x, flat_mask)
        candidates = torch.cat((flat_x, exposed), dim=1)
        valid_sample = flat_mask.any(dim=1, keepdim=True)
        flat_exposed_mask = (
            torch.ones(batch, self.exposed, dtype=torch.bool, device=x.device)
            if exposed_mask is None
            else exposed_mask.reshape(batch, self.exposed)
        )
        flat_exposed_mask = flat_exposed_mask & valid_sample
        candidate_mask = torch.cat((flat_mask, flat_exposed_mask), dim=1)
        total = candidates.shape[1]
        if self.hard_backend != "sort" and total > self.max_length:
            raise ValueError(
                f"UnFold output length {total} exceeds max_length={self.max_length}; "
                "use a shorter tensor or explicitly raise the alpha backend limit"
            )

        positions = torch.linspace(-1, 1, total, device=x.device, dtype=x.dtype)
        flat_condition = None
        if self.condition_dim is not None:
            if condition is None:
                flat_condition = flat_x.new_zeros(batch, self.condition_dim)
            else:
                flat_condition = condition.reshape(batch, self.condition_dim).to(x.dtype)
        if self.layout_mode == "canonical":
            if guide is None:
                raise ValueError("layout_mode='canonical' requires guide")
            original_scores = self._normalize_guide(
                guide.reshape(batch, length, 1).to(x.dtype), flat_mask
            ).squeeze(-1)
            if exposed_guide is not None:
                exposed_scores = exposed_guide.reshape(batch, self.exposed).to(x.dtype)
            else:
                assert self.exposed_coordinate is not None
                query_identity = self.exposed_queries.to(x.dtype).unsqueeze(0).expand(
                    batch, -1, -1
                )
                coordinate_parts = [attended, query_identity]
                if flat_condition is not None:
                    coordinate_parts.append(
                        flat_condition[:, None].expand(-1, self.exposed, -1)
                    )
                exposed_scores = 3.0 * torch.tanh(
                    self.exposed_coordinate(
                        torch.cat(coordinate_parts, dim=-1)
                    ).squeeze(-1)
                )
            rank_scores = torch.cat((original_scores, exposed_scores), dim=1)
            invalid_rank = (
                rank_scores.detach().abs().amax(dim=1, keepdim=True)
                + 2.0
                + positions / max(total, 1)
            )
            rank_scores = torch.where(candidate_mask, rank_scores, invalid_rank)
            source = torch.argsort(rank_scores, dim=-1, stable=True)
            soft_layout = (
                self._soft_rank_layout(rank_scores)
                if self.training or return_soft_layout
                else None
            )
        else:
            valid_weight = flat_mask.to(flat_x.dtype)
            context = (flat_x * valid_weight[..., None]).sum(1)
            context = context / valid_weight.sum(1, keepdim=True).clamp_min(1)
            layout_parts = [
                candidates,
                context[:, None].expand(-1, total, -1),
            ]
            if self.guide_proj is not None:
                if guide is None:
                    candidate_guide = flat_x.new_zeros(batch, total, self.guide_dim)
                else:
                    flat_guide = guide.reshape(batch, length, self.guide_dim).to(x.dtype)
                    normalized_guide = self._normalize_guide(flat_guide, flat_mask)
                    assert self.exposed_guide is not None
                    learned_exposed_guide = (
                        self.exposed_guide.to(x.dtype)
                        .unsqueeze(0)
                        .expand(batch, -1, -1)
                    )
                    candidate_guide = torch.cat(
                        (normalized_guide, learned_exposed_guide), dim=1
                    )
                layout_parts.append(self.guide_proj(candidate_guide))
            if self.condition_proj is not None:
                assert flat_condition is not None
                projected_condition = self.condition_proj(flat_condition)
                layout_parts.append(
                    projected_condition[:, None].expand(-1, total, -1)
                )
            layout_input = torch.cat(layout_parts, dim=-1)
            candidate_keys = F.normalize(self.layout_score(layout_input), dim=-1)
            if self.hard_backend == "sort":
                rank_scores = torch.tanh(self.rank_score(candidate_keys).squeeze(-1))
                invalid_rank = (
                    rank_scores.detach().abs().amax(dim=1, keepdim=True)
                    + 2.0
                    + positions / max(total, 1)
                )
                rank_scores = torch.where(candidate_mask, rank_scores, invalid_rank)
                source = torch.argsort(rank_scores, dim=-1, stable=True)
                soft_layout = (
                    self._soft_rank_layout(rank_scores)
                    if self.training or return_soft_layout
                    else None
                )
            else:
                slot_inputs = positions.reshape(total, 1)
                slot_keys = F.normalize(self.layout_slots(slot_inputs), dim=-1)
                logits = torch.einsum("rh,bch->brc", slot_keys, candidate_keys)
                invalid_preference = positions.reshape(1, total, 1) * 4.0
                logits = torch.where(candidate_mask[:, None], logits, invalid_preference)
                source = self._hard_matching(logits)
                soft_layout = (
                    self._soft_layout(logits)
                    if self.training or return_soft_layout
                    else None
                )
        gather_index = source[..., None].expand(-1, -1, dim)
        hard = candidates.gather(1, gather_index)
        if self.training:
            assert soft_layout is not None
            soft = torch.bmm(soft_layout, candidates.detach())
            y = _HardLayoutSoftGradient.apply(hard, soft)
        else:
            y = hard

        output_shape = (*leading_shape, total)
        outputs: list[Tensor] = [y.reshape(*output_shape, dim)]
        if mask is not None:
            outputs.append(candidate_mask.gather(1, source).reshape(*output_shape))
        if return_exposed_mask:
            exposed_candidates = torch.arange(total, device=x.device) >= length
            outputs.append(exposed_candidates[source].reshape(*output_shape))
        if return_source_index:
            outputs.append(source.reshape(*output_shape))
        if return_soft_layout:
            assert soft_layout is not None
            outputs.append(soft_layout.reshape(*leading_shape, total, total))
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, exposed={self.exposed}, guide_dim={self.guide_dim}, "
            f"condition_dim={self.condition_dim}, hidden_dim={self.hidden_dim}, "
            f"temperature={self.temperature:g}, sinkhorn_steps={self.sinkhorn_steps}, "
            f"max_length={self.max_length}, hard_backend={self.hard_backend!r}, "
            f"layout_mode={self.layout_mode!r}, value_operators={self.value_operators}, "
            f"value_rank={self.value_rank}, "
            f"query_chunk_size={self.query_chunk_size}, "
            f"operator_chunk_size={self.operator_chunk_size}"
        )


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

from .stateful_recall import StatefulRecall
from .visual_scan import PixelShiftObservation, VisualScan, VisualScanConfig, VisualScanOutput
__all__ = ["Layer", "Half", "UnFold", "Fold", "Pulse", "LearnedPulse", "RecallRefiner", "StatefulRecall", "VisualField", "VisualFieldOutput", "concat_visual_fields", "VisualScan", "VisualScanConfig", "VisualScanOutput", "PixelShiftObservation"]


def __getattr__(name: str):
    if name == "Layer":
        from .usage import Layer

        return Layer
    raise AttributeError(name)
