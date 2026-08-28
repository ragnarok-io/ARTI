"""Membrane visibility routing for autoregressive token streams.

Membrane routing keeps generated tokens as ordinary tokens while assigning them
to a visibility domain. Inner-speech tokens stay in the model-side context, but
are not emitted to the user and are not readable by non-assistant viewers.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .emission import (
    EmissionRouter,
    EmissionRouterConfig,
    build_stream_visibility,
    stream_emit_mask,
)


MEMBRANE_STREAM_ASSISTANT_PUBLIC = 0
MEMBRANE_STREAM_ASSISTANT_INNER = 1
MEMBRANE_STREAM_NAMES: tuple[str, ...] = ("assistant_public", "assistant_inner")


@dataclass(frozen=True)
class MembraneRoutingConfig:
    """Configuration for a two-domain assistant token router."""

    hidden_dim: int
    public_stream_id: int = MEMBRANE_STREAM_ASSISTANT_PUBLIC
    inner_stream_id: int = MEMBRANE_STREAM_ASSISTANT_INNER
    public_emit_streams: tuple[int, ...] = (MEMBRANE_STREAM_ASSISTANT_PUBLIC,)


@dataclass(frozen=True)
class MembraneRoutingOutput:
    """Output of ``MembraneVisibilityRouter``."""

    stream_logits: Tensor
    stream_probs: Tensor
    stream_ids: Tensor
    public_emit_mask: Tensor
    diagnostics: dict[str, Tensor]


@dataclass(frozen=True)
class MembraneContext:
    """Token stream context after membrane assignment."""

    token_ids: Tensor
    stream_ids: Tensor
    participant_ids: Tensor
    phase_ids: Tensor
    public_emit_mask: Tensor
    mask: Tensor
    visibility: Tensor


class MembraneVisibilityRouter(EmissionRouter):
    """Compatibility adapter for the legacy two-stream membrane API.

    New integrations should use :class:`arti.EmissionRouter` and provide their
    own numbered-stream policy.  This adapter preserves the old output field
    names and diagnostics for existing callers.
    """

    def __init__(self, config: MembraneRoutingConfig) -> None:
        super().__init__(
            EmissionRouterConfig(
                hidden_dim=config.hidden_dim,
                stream_count=len(MEMBRANE_STREAM_NAMES),
                emit_streams=config.public_emit_streams,
            )
        )
        self.config = config

    def forward(
        self,
        hidden: Tensor,
        *,
        stream_ids: Tensor | None = None,
    ) -> MembraneRoutingOutput:
        routed = super().forward(hidden, stream_ids=stream_ids)
        diagnostics = dict(routed.diagnostics)
        diagnostics.update(
            {
                "membrane_public_probability": routed.stream_probs[
                    ..., self.config.public_stream_id
                ].detach(),
                "membrane_inner_probability": routed.stream_probs[
                    ..., self.config.inner_stream_id
                ].detach(),
            }
        )
        return MembraneRoutingOutput(
            stream_logits=routed.stream_logits,
            stream_probs=routed.stream_probs,
            stream_ids=routed.stream_ids,
            public_emit_mask=routed.emit_mask,
            diagnostics=diagnostics,
        )


def membrane_public_emit_mask(stream_ids: Tensor, *, public_streams: tuple[int, ...] = (MEMBRANE_STREAM_ASSISTANT_PUBLIC,)) -> Tensor:
    """Return a bool mask of tokens that should be emitted to the user."""

    return stream_emit_mask(stream_ids, streams=public_streams)


def membrane_emit_tokens(token_ids: Tensor, stream_ids: Tensor, *, public_streams: tuple[int, ...] = (MEMBRANE_STREAM_ASSISTANT_PUBLIC,)) -> list[list[int]]:
    """Return public token ids for each batch row.

    Inner tokens remain in the model-side context but are not returned here.
    """

    if token_ids.shape != stream_ids.shape:
        raise ValueError("token_ids and stream_ids must have the same shape")
    emit = membrane_public_emit_mask(stream_ids, public_streams=public_streams)
    return [row[mask].to(dtype=torch.long).tolist() for row, mask in zip(token_ids, emit)]


def build_membrane_visibility(
    stream_ids: Tensor,
    viewer_ids: Tensor,
    stream_readable_by: Tensor,
    *,
    mask: Tensor | None = None,
) -> Tensor:
    """Build ``[B, N, N]`` visibility from stream readability.

    ``stream_readable_by[viewer, stream]`` means the viewer participant may read
    tokens in that stream. The returned visibility can be passed to ARTI layers.
    """

    if stream_ids.ndim != 2:
        raise ValueError("stream_ids must have shape [B, N]")
    batch, tokens = stream_ids.shape
    if viewer_ids.shape != (batch,):
        raise ValueError(f"viewer_ids must have shape {(batch,)}")
    readable = _expand_stream_readability(stream_readable_by, batch)
    readable = readable.to(device=stream_ids.device, dtype=torch.bool)
    legacy_mask = (
        None
        if mask is None
        else mask.to(device=stream_ids.device, dtype=torch.bool)
    )
    return build_stream_visibility(
        stream_ids.to(device=stream_ids.device),
        viewer_ids.to(device=stream_ids.device),
        readable,
        valid_mask=legacy_mask,
    )


def append_membrane_tokens(
    token_ids: Tensor,
    stream_ids: Tensor,
    new_token_ids: Tensor,
    new_stream_ids: Tensor,
    *,
    participant_ids: Tensor | None = None,
    new_participant_ids: Tensor | None = None,
    phase_ids: Tensor | None = None,
    new_phase_ids: Tensor | None = None,
    mask: Tensor | None = None,
) -> dict[str, Tensor]:
    """Append generated tokens and their stream metadata to a context."""

    if token_ids.shape != stream_ids.shape:
        raise ValueError("token_ids and stream_ids must have the same shape")
    if new_token_ids.shape != new_stream_ids.shape:
        raise ValueError("new_token_ids and new_stream_ids must have the same shape")
    if token_ids.shape[0] != new_token_ids.shape[0]:
        raise ValueError("new tokens must share batch size")
    out = {
        "token_ids": torch.cat([token_ids, new_token_ids.to(device=token_ids.device, dtype=token_ids.dtype)], dim=1),
        "stream_ids": torch.cat([stream_ids, new_stream_ids.to(device=stream_ids.device, dtype=stream_ids.dtype)], dim=1),
    }
    if mask is not None:
        out["mask"] = torch.cat([mask.to(device=token_ids.device, dtype=torch.bool), torch.ones_like(new_token_ids, dtype=torch.bool)], dim=1)
    if participant_ids is not None or new_participant_ids is not None:
        if participant_ids is None or new_participant_ids is None:
            raise ValueError("participant_ids and new_participant_ids must be provided together")
        out["participant_ids"] = torch.cat([participant_ids, new_participant_ids.to(device=participant_ids.device, dtype=participant_ids.dtype)], dim=1)
    if phase_ids is not None or new_phase_ids is not None:
        if phase_ids is None or new_phase_ids is None:
            raise ValueError("phase_ids and new_phase_ids must be provided together")
        out["phase_ids"] = torch.cat([phase_ids, new_phase_ids.to(device=phase_ids.device, dtype=phase_ids.dtype)], dim=1)
    return out


def _expand_stream_readability(readable_by: Tensor, batch: int) -> Tensor:
    if readable_by.ndim == 2:
        return readable_by.unsqueeze(0).expand(batch, -1, -1)
    if readable_by.ndim == 3 and readable_by.shape[0] == batch:
        return readable_by
    raise ValueError("stream_readable_by must have shape [P, S] or [B, P, S]")
