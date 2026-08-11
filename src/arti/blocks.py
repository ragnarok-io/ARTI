"""Composable ARTI blocks."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .config import ARTIConfig
from .layers import ARTIRecallWriteLayer, ARTILayer
from .outputs import ARTIOutput


class ARTIHostBridge(nn.Module):
    """Project an ARTI representation into a host residual stream.

    ``radial`` separates every output row into a unit direction and an
    independently trainable, signed radius. Zero radii preserve the host
    function exactly while limiting the first adaptive-optimizer update to
    one scalar per output channel. The fixed residual budget is non-zero, so
    it changes optimization coordinates without bounding the learned map.

    ``dense`` retains the historical zero-initialized linear projection for
    loading artifacts that explicitly request its original semantics.
    """

    MODES = frozenset({"radial", "dense"})

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        mode: str = "radial",
        residual_budget: float = 1.0,
        rms_limit: float = 4.0,
        bias: bool = False,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("ARTI host bridge dimensions must be positive")
        if mode not in self.MODES:
            raise ValueError(f"ARTI host bridge mode must be one of {sorted(self.MODES)}")
        if not 0.0 < residual_budget <= 1.0:
            raise ValueError("ARTI host bridge residual_budget must be in (0, 1]")
        if rms_limit <= 0.0:
            raise ValueError("ARTI host bridge rms_limit must be positive")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.mode = mode
        self.residual_budget = float(residual_budget)
        self.rms_limit = float(rms_limit)
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)

        if mode == "radial":
            nn.init.normal_(self.linear.weight)
            self.radius = nn.Parameter(torch.zeros(output_dim, dtype=torch.float32))
            if self.linear.bias is not None:
                nn.init.zeros_(self.linear.bias)
        else:
            nn.init.zeros_(self.linear.weight)
            if self.linear.bias is not None:
                nn.init.zeros_(self.linear.bias)
            self.register_parameter("radius", None)

    def _apply(self, fn, recurse: bool = True):
        result = super()._apply(fn, recurse=recurse)
        if self.radius is not None and self.radius.is_floating_point() and self.radius.dtype != torch.float32:
            self.radius.data = self.radius.data.float()
            if self.radius.grad is not None:
                self.radius.grad.data = self.radius.grad.data.float()
        return result

    def forward(self, x: Tensor) -> Tensor:
        if self.mode == "dense":
            return self.linear(x)

        statistics = x.float() if x.dtype in {torch.float16, torch.bfloat16} else x
        rms = statistics.square().mean(dim=-1, keepdim=True).add(torch.finfo(statistics.dtype).eps).sqrt()
        clip_scale = torch.clamp(rms / self.rms_limit, min=1.0).to(dtype=x.dtype)
        clipped = x / clip_scale
        direction = F.normalize(self.linear.weight.float(), dim=-1).to(dtype=x.dtype)
        bias = None if self.linear.bias is None else self.linear.bias.to(dtype=x.dtype)
        projected = F.linear(clipped, direction, bias)
        projected_statistics = projected.float() if projected.dtype in {torch.float16, torch.bfloat16} else projected
        projected_rms = projected_statistics.square().mean(dim=-1, keepdim=True).add(
            torch.finfo(projected_statistics.dtype).eps
        ).sqrt()
        projected_scale = torch.clamp(projected_rms / self.rms_limit, min=1.0).to(dtype=x.dtype)
        projected = projected / projected_scale
        scale = self.radius.to(device=x.device, dtype=x.dtype)
        scale = scale * self.residual_budget
        return projected * scale

    def effective_weight(self) -> Tensor:
        """Return the folded dense weight represented by the radial bridge."""

        if self.mode == "dense":
            return self.linear.weight
        direction = F.normalize(self.linear.weight, dim=-1)
        scale = self.radius.to(dtype=direction.dtype)
        scale = scale * self.residual_budget
        return direction * scale.unsqueeze(-1)


class ARTIResidualBlock(nn.Module):
    """Shape-stable residual block for insertion into ordinary PyTorch models."""

    def __init__(
        self,
        dim: int,
        coord_dim: int = 0,
        hidden_dim: int | None = None,
        dropout: float = 0.0,
        *,
        operator_count: int = 4,
        interface_slots: int = 8,
        recall_slots: int = 4,
        recall_steps: int = 1,
        recall_min_steps: int = 1,
        recall_tolerance: float | None = None,
        recall_activation: str = "half",
        recall_recognition_mode: str = "none",
        recall_routing: str = "dense",
        recall_key_dim: int = 32,
        recall_query_mode: str = "fixed",
        recall_query_seed: int = 0,
        recall_group_size: int = 16,
        recall_group_topk: int = 2,
        recall_value_composition: str = "single",
        recall_formula: nn.Module | None = None,
        recall_route_exploration: float = 0.0,
        use_phase_mixer: bool = True,
        use_virtual_interface: bool = True,
        use_recall: bool = True,
        use_virtual_recall: bool = True,
        require_coord: bool = False,
        require_visibility: bool = False,
        coord_frame_mode: str = "none",
        fallback_context: str = "none",
        fallback_slots: int = 32,
        zero_init_output: bool = False,
        bridge_mode: str = "radial",
        residual_budget: float = 1.0,
        direct_recall: bool = False,
    ) -> None:
        super().__init__()
        resolved_hidden_dim = dim if hidden_dim is None else hidden_dim
        self.zero_init_output = bool(zero_init_output)
        self.direct_recall = bool(direct_recall)
        self._compiled_static_recall_steps = False
        self.bridge_mode = "recall-write" if self.direct_recall else bridge_mode
        self.residual_budget = float(residual_budget)
        if self.direct_recall:
            self.layer = ARTIRecallWriteLayer(
                ARTIConfig(
                    input_dim=dim,
                    hidden_dim=dim,
                    coord_dim=coord_dim,
                    dropout=dropout,
                    operator_count=operator_count,
                    interface_slots=interface_slots,
                    recall_slots=recall_slots,
                    recall_steps=recall_steps,
                    recall_min_steps=recall_min_steps,
                    recall_tolerance=recall_tolerance,
                    recall_activation=recall_activation,
                    recall_recognition_mode=recall_recognition_mode,
                    recall_routing=recall_routing,
                    recall_key_dim=min(recall_key_dim, dim),
                    recall_query_mode=recall_query_mode,
                    recall_query_seed=recall_query_seed,
                    recall_group_size=recall_group_size,
                    recall_group_topk=recall_group_topk,
                    recall_value_composition=recall_value_composition,
                    recall_route_exploration=recall_route_exploration,
                    use_phase_mixer=False,
                    use_virtual_interface=False,
                    use_pairwise_context=False,
                    use_recall=use_recall,
                    use_virtual_recall=use_virtual_recall,
                    require_coord=require_coord,
                    require_visibility=False,
                    coord_frame_mode=coord_frame_mode,
                    fallback_context=fallback_context,
                    fallback_slots=fallback_slots,
                ),
                identity_init_bank=zero_init_output,
                formula=recall_formula,
            )
            self.norm = nn.Identity()
        else:
            if recall_formula is not None:
                raise ValueError(
                    "custom Recall formulas currently require direct_recall=True"
                )
            self.layer = ARTILayer(
                input_dim=dim,
                hidden_dim=resolved_hidden_dim,
                coord_dim=coord_dim,
                dropout=dropout,
                operator_count=operator_count,
                interface_slots=interface_slots,
                recall_slots=recall_slots,
                recall_steps=recall_steps,
                recall_min_steps=recall_min_steps,
                recall_tolerance=recall_tolerance,
                recall_activation=recall_activation,
                recall_recognition_mode=recall_recognition_mode,
                recall_routing=recall_routing,
                recall_key_dim=recall_key_dim,
                recall_query_mode=recall_query_mode,
                recall_query_seed=recall_query_seed,
                recall_group_size=recall_group_size,
                recall_group_topk=recall_group_topk,
                recall_value_composition=recall_value_composition,
                recall_route_exploration=recall_route_exploration,
                use_phase_mixer=use_phase_mixer,
                use_virtual_interface=use_virtual_interface,
                use_recall=use_recall,
                use_virtual_recall=use_virtual_recall,
                require_coord=require_coord,
                require_visibility=require_visibility,
                coord_frame_mode=coord_frame_mode,
                fallback_context=fallback_context,
                fallback_slots=fallback_slots,
            )
            self.out = (
                ARTIHostBridge(
                    resolved_hidden_dim,
                    dim,
                    mode=bridge_mode,
                    residual_budget=residual_budget,
                    bias=bridge_mode == "dense",
                )
                if self.zero_init_output
                else nn.Linear(resolved_hidden_dim, dim)
                if resolved_hidden_dim != dim
                else nn.Identity()
            )
            self.norm = nn.Identity() if self.zero_init_output else nn.LayerNorm(dim)

    def forward(self, x: Tensor, **kwargs: Tensor) -> Tensor:
        if self.direct_recall:
            if self.layer._forward_hooks:
                recall = self.layer(x, **kwargs)
                return recall.y
            else:
                kwargs.setdefault(
                    "static_steps",
                    self.training or self._compiled_static_recall_steps,
                )
                return self.layer.forward_write(x, **kwargs)
        out = self.layer(x, **kwargs)
        updated = x + self.out(out.y)
        return updated if self.zero_init_output else self.norm(updated)


class ARTISequenceBlock(nn.Module):
    """Sequence block that returns the full ARTIOutput."""

    def __init__(self, dim: int, coord_dim: int = 0, hidden_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        self.layer = ARTILayer(input_dim=dim, hidden_dim=hidden_dim, coord_dim=coord_dim, dropout=dropout)

    def forward(self, x: Tensor, **kwargs: Tensor) -> ARTIOutput:
        return self.layer(x, **kwargs)


class ARTIPooledBlock(nn.Module):
    """Return only the pooled latent representation."""

    def __init__(self, dim: int, coord_dim: int = 0, hidden_dim: int | None = None, dropout: float = 0.0) -> None:
        super().__init__()
        self.layer = ARTILayer(input_dim=dim, hidden_dim=hidden_dim, coord_dim=coord_dim, dropout=dropout)

    def forward(self, x: Tensor, **kwargs: Tensor) -> Tensor:
        return self.layer(x, **kwargs).pooled
