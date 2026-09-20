"""Explicit lossy aggregation after reversible topology reunion."""

from __future__ import annotations

from typing import ClassVar

import torch
from torch import Tensor, nn

from .nn import Fold as SoftWorkspaceFold
from .runtime_contracts import EnvelopeRef, SupportDomain, TensorEnvelope


class SoftFoldAggregate(nn.Module):
    """Use the published soft Fold@1 mechanism as a final aggregation kernel."""

    _component_reference: ClassVar[str] = "arti/soft-fold-aggregate@1"

    def __init__(
        self,
        k: int,
        *,
        dim: int,
        hidden_dim: int | None = None,
        temperature: float = 1.0,
        dropout: float = 0.0,
        mode: str = "soft",
        topk: int | None = None,
        heads: int = 1,
    ) -> None:
        super().__init__()
        self.k = int(k)
        self.dim = int(dim)
        self.fold = SoftWorkspaceFold(
            k,
            dim=dim,
            hidden_dim=hidden_dim,
            temperature=temperature,
            dropout=dropout,
            mode=mode,
            topk=topk,
            heads=heads,
        )

    def forward(
        self,
        x: Tensor,
        *,
        mask: Tensor,
        q: Tensor | None = None,
    ) -> Tensor:
        return self.fold(x, q, mask=mask)


class ReunionAggregate(nn.Module):
    """Host a lossy kernel after every reversible topology has been closed."""

    _component_reference: ClassVar[str] = "arti/reunion-aggregate@1"

    def __init__(self, kernel: nn.Module) -> None:
        super().__init__()
        if not isinstance(kernel, nn.Module):
            raise TypeError("aggregate kernel must be an nn.Module")
        from .component_registry import get_component_registry

        registration = get_component_registry().registration_for(kernel)
        if registration is None or "pulse.aggregate.kernel" not in registration.capabilities:
            raise ValueError("kernel must declare pulse.aggregate.kernel capability")
        self.kernel = kernel

    def forward(
        self,
        reunited: TensorEnvelope,
        *,
        q: Tensor | None = None,
    ) -> TensorEnvelope:
        closed_refs = {EnvelopeRef.WORLD, EnvelopeRef.OBSERVATION, EnvelopeRef.REUNITED}
        if not isinstance(reunited, TensorEnvelope) or reunited.ref not in closed_refs:
            raise TypeError("ReunionAggregate requires a closed TensorEnvelope")
        value = reunited.value
        mask = reunited._mask_for_execution()
        if value.ndim < 3:
            raise ValueError("aggregate input must have batch, instance, and feature axes")
        batch, dim = value.shape[0], value.shape[-1]
        flat_value = value.reshape(batch, -1, dim)
        flat_mask = mask.reshape(batch, -1)
        flat_q = None
        if q is not None:
            if q.shape == mask.shape:
                flat_q = q.reshape(batch, -1)
            elif q.shape == (*mask.shape, 1):
                flat_q = q.reshape(batch, -1, 1)
            else:
                raise ValueError("q must match the aggregate input mask shape")
        result = self.kernel(
            flat_value,
            mask=flat_mask,
            q=flat_q,
        )
        if not isinstance(result, Tensor) or result.ndim != 3:
            raise TypeError("aggregate kernel must return a [B, M, D] Tensor")
        if (
            result.shape[0] != batch
            or result.shape[-1] != dim
        ):
            raise ValueError("aggregate kernel must preserve batch and feature dimensions")
        if result.device != value.device or result.dtype != value.dtype:
            raise ValueError("aggregate kernel must preserve device and dtype")
        output_mask = flat_mask.any(dim=-1, keepdim=True).expand(
            result.shape[:-1]
        )
        domain = SupportDomain.for_tensor(
            output_mask,
            domain_id="pulse-output",
            owner_ref=self._component_reference,
            partition_id="pulse",
            transition_id="aggregate",
        )
        return TensorEnvelope(
            EnvelopeRef.PULSE,
            torch.where(output_mask.unsqueeze(-1), result, torch.zeros_like(result)),
            output_mask,
            domain,
        )


__all__ = ["ReunionAggregate", "SoftFoldAggregate"]
