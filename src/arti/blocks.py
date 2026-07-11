"""Composable ARTI blocks."""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from .layers import ARTILayer
from .outputs import ARTIOutput


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
        recall_activation: str = "half",
        recall_recognition_mode: str = "explicit",
        use_phase_mixer: bool = True,
        use_virtual_interface: bool = True,
        use_recall: bool = True,
        use_virtual_recall: bool = True,
        require_coord: bool = False,
        require_visibility: bool = False,
        coord_frame_mode: str = "none",
        fallback_context: str = "none",
        fallback_slots: int = 32,
    ) -> None:
        super().__init__()
        resolved_hidden_dim = dim if hidden_dim is None else hidden_dim
        self.layer = ARTILayer(
            input_dim=dim,
            hidden_dim=resolved_hidden_dim,
            coord_dim=coord_dim,
            dropout=dropout,
            operator_count=operator_count,
            interface_slots=interface_slots,
            recall_slots=recall_slots,
            recall_steps=recall_steps,
            recall_activation=recall_activation,
            recall_recognition_mode=recall_recognition_mode,
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
        self.out = nn.Identity() if resolved_hidden_dim == dim else nn.Linear(resolved_hidden_dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor, **kwargs: Tensor) -> Tensor:
        out = self.layer(x, **kwargs)
        return self.norm(x + self.out(out.y))


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
