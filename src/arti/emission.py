"""Generic emission routing and stream visibility utilities.

The public API deals only in numbered streams.  Names such as ``public`` or
``inner`` belong in an application adapter, not in the tensor router itself.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor


def _require_integer(name: str, value: Tensor, *, device: torch.device) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}")
    if value.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise TypeError(f"{name} must have an integer dtype, got {value.dtype}")
    return value


def _validate_stream_ids(
    stream_ids: Tensor,
    *,
    shape: tuple[int, ...],
    stream_count: int,
    device: torch.device,
    name: str = "stream_ids",
) -> Tensor:
    ids = _require_integer(name, stream_ids, device=device)
    if tuple(ids.shape) != shape:
        raise ValueError(f"{name} must have shape {shape}, got {tuple(ids.shape)}")
    if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= stream_count):
        raise ValueError(f"{name} contains an id outside [0, {stream_count})")
    return ids.to(dtype=torch.long)


def _validate_valid_mask(
    valid_mask: Tensor | None,
    *,
    shape: tuple[int, ...],
    device: torch.device,
) -> Tensor:
    if valid_mask is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    if not isinstance(valid_mask, Tensor):
        raise TypeError("valid_mask must be a torch.Tensor")
    if tuple(valid_mask.shape) != shape:
        raise ValueError(f"valid_mask must have shape {shape}, got {tuple(valid_mask.shape)}")
    if valid_mask.dtype is not torch.bool:
        raise TypeError(f"valid_mask must have dtype torch.bool, got {valid_mask.dtype}")
    if valid_mask.device != device:
        raise ValueError(f"valid_mask must be on {device}, got {valid_mask.device}")
    return valid_mask


@dataclass(frozen=True)
class EmissionRouterConfig:
    """Static configuration for a numbered-stream emission router."""

    hidden_dim: int
    stream_count: int = 2
    emit_streams: tuple[int, ...] = (0,)
    temperature: float = 1.0

    def __post_init__(self) -> None:
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.stream_count <= 0:
            raise ValueError("stream_count must be positive")
        if not self.emit_streams:
            raise ValueError("emit_streams must contain at least one stream")
        if any(stream < 0 or stream >= self.stream_count for stream in self.emit_streams):
            raise ValueError("emit_streams contains an out-of-range stream")
        if not torch.isfinite(torch.tensor(float(self.temperature))) or self.temperature <= 0:
            raise ValueError("temperature must be positive and finite")


@dataclass(frozen=True)
class EmissionRouterOutput:
    """Tensor outputs of :class:`EmissionRouter`."""

    stream_logits: Tensor
    stream_probs: Tensor
    stream_ids: Tensor
    emit_mask: Tensor
    diagnostics: dict[str, Tensor]


class EmissionRouter(nn.Module):
    """Choose a numbered emission stream for each hidden position.

    The router does not know what a stream means and does not generate token
    ids.  Applications can map ``emit_mask`` to their own transport or output
    policy, while the tensor module remains reusable across model families.
    """

    def __init__(self, config: EmissionRouterConfig) -> None:
        super().__init__()
        self._emission_config = config
        self.config = config
        self.router = nn.Linear(config.hidden_dim, config.stream_count)

    def forward(
        self,
        hidden: Tensor,
        *,
        stream_ids: Tensor | None = None,
        valid_mask: Tensor | None = None,
    ) -> EmissionRouterOutput:
        if not isinstance(hidden, Tensor) or not hidden.is_floating_point():
            raise TypeError("hidden must be a floating-point torch.Tensor")
        if hidden.ndim not in {2, 3}:
            raise ValueError("hidden must have shape [B, D] or [B, N, D]")
        config = self._emission_config
        if hidden.shape[-1] != config.hidden_dim:
            raise ValueError(
                f"hidden last dim must be {config.hidden_dim}, got {hidden.shape[-1]}"
            )
        logits = self.router(hidden)
        probs = torch.softmax(logits / config.temperature, dim=-1)
        position_shape = tuple(logits.shape[:-1])
        valid_shape = position_shape
        mask = _validate_valid_mask(
            valid_mask,
            shape=valid_shape,
            device=hidden.device,
        )
        if stream_ids is None:
            routed = probs.argmax(dim=-1)
        else:
            routed = _validate_stream_ids(
                stream_ids,
                shape=position_shape,
                stream_count=config.stream_count,
                device=hidden.device,
            )
        emit = stream_emit_mask(routed, streams=config.emit_streams) & mask
        diagnostics = {
            "emission_entropy": (
                -(probs.clamp_min(torch.finfo(probs.dtype).eps).log() * probs).sum(dim=-1)
            ).detach(),
            "emission_valid_mask": mask.detach(),
        }
        return EmissionRouterOutput(
            stream_logits=logits,
            stream_probs=probs,
            stream_ids=routed,
            emit_mask=emit,
            diagnostics=diagnostics,
        )


def stream_emit_mask(stream_ids: Tensor, *, streams: tuple[int, ...] = (0,)) -> Tensor:
    """Return the application-defined emission mask for numbered streams."""

    if not isinstance(stream_ids, Tensor):
        raise TypeError("stream_ids must be a torch.Tensor")
    if not streams:
        raise ValueError("streams must contain at least one stream")
    stream = stream_ids.to(dtype=torch.long)
    emit = torch.zeros_like(stream, dtype=torch.bool)
    for stream_id in streams:
        emit |= stream == int(stream_id)
    return emit


def build_stream_visibility(
    stream_ids: Tensor,
    viewer_ids: Tensor,
    stream_readable_by: Tensor,
    *,
    stream_count: int | None = None,
    valid_mask: Tensor | None = None,
) -> Tensor:
    """Build query-to-source visibility from a numbered stream policy.

    ``stream_readable_by[viewer, stream]`` is an application-owned boolean
    policy.  This helper applies it as a tensor gate and never infers
    authorization from token values, names, or emission probabilities.
    """

    if not isinstance(stream_ids, Tensor) or stream_ids.ndim != 2:
        raise ValueError("stream_ids must have shape [B, N]")
    batch, tokens = stream_ids.shape
    if not isinstance(viewer_ids, Tensor) or viewer_ids.shape != (batch,):
        raise ValueError(f"viewer_ids must have shape {(batch,)}")
    if not isinstance(stream_readable_by, Tensor) or stream_readable_by.dtype is not torch.bool:
        raise TypeError("stream_readable_by must be a boolean torch.Tensor")
    if stream_readable_by.device != stream_ids.device:
        raise ValueError("stream_readable_by must be on the stream_ids device")
    if stream_readable_by.ndim == 2:
        readable = stream_readable_by.unsqueeze(0).expand(batch, -1, -1)
    elif stream_readable_by.ndim == 3 and stream_readable_by.shape[0] == batch:
        readable = stream_readable_by
    else:
        raise ValueError("stream_readable_by must have shape [P, S] or [B, P, S]")
    if stream_count is None:
        stream_count = readable.shape[-1]
    if readable.shape[-1] != stream_count:
        raise ValueError("stream_readable_by last dim must equal stream_count")
    ids = _validate_stream_ids(
        stream_ids,
        shape=(batch, tokens),
        stream_count=stream_count,
        device=stream_ids.device,
    )
    viewers = _validate_stream_ids(
        viewer_ids,
        shape=(batch,),
        stream_count=readable.shape[1],
        device=stream_ids.device,
        name="viewer_ids",
    )
    mask = _validate_valid_mask(
        valid_mask,
        shape=(batch, tokens),
        device=stream_ids.device,
    )
    rows = torch.arange(batch, device=stream_ids.device).unsqueeze(1)
    source_visible = readable[rows, viewers.unsqueeze(1), ids] & mask
    return source_visible.unsqueeze(1).expand(batch, tokens, tokens) & mask.unsqueeze(1)


__all__ = [
    "EmissionRouter",
    "EmissionRouterConfig",
    "EmissionRouterOutput",
    "build_stream_visibility",
    "stream_emit_mask",
]
