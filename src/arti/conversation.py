"""Participant-context helpers for conversation-like token tensors.

The helpers in this module are tensor adapters. They do not encode business
roles. Downstream systems decide which participant can read which context; ARTI
only turns those decisions into ``coord``, ``mask``, ``visibility``, and
``observer_coord`` tensors.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class ParticipantContextTensors:
    """Tensor bundle for participant-scoped ARTI calls."""

    coord: Tensor
    mask: Tensor
    visibility: Tensor
    observer_coord: Tensor
    active_participant: Tensor
    observer_participant: Tensor


def last_non_assistant_participant(participant_ids: Tensor, mask: Tensor | None = None, *, assistant_id: int) -> Tensor:
    """Return the most recent non-assistant participant id for each batch item.

    ``participant_ids`` has shape ``[B, N]``. Invalid tokens are ignored when
    ``mask`` is provided. If a row has no non-assistant token, ``assistant_id``
    is returned for that row.
    """

    if participant_ids.ndim != 2:
        raise ValueError("participant_ids must have shape [B, N]")
    valid = torch.ones_like(participant_ids, dtype=torch.bool) if mask is None else mask.to(device=participant_ids.device, dtype=torch.bool)
    if valid.shape != participant_ids.shape:
        raise ValueError(f"mask must have shape {tuple(participant_ids.shape)}, got {tuple(valid.shape)}")
    candidates = valid & (participant_ids != assistant_id)
    positions = torch.arange(participant_ids.shape[1], device=participant_ids.device).unsqueeze(0).expand_as(participant_ids)
    last_pos = positions.masked_fill(~candidates, -1).amax(dim=1)
    fallback = torch.full((participant_ids.shape[0],), assistant_id, device=participant_ids.device, dtype=participant_ids.dtype)
    gathered = participant_ids.gather(1, last_pos.clamp_min(0).unsqueeze(1)).squeeze(1)
    return torch.where(last_pos >= 0, gathered, fallback)


def build_participant_context(
    participant_ids: Tensor,
    participant_coord: Tensor,
    readable_by: Tensor,
    *,
    mask: Tensor | None = None,
    active_participant: Tensor | int | None = None,
    assistant_id: int | None = None,
    observer_participant: Tensor | int | None = None,
) -> ParticipantContextTensors:
    """Build ARTI coord, visibility, and observer tensors for participants.

    Args:
        participant_ids: Integer tensor ``[B, N]``. Each token's owner/speaker.
        participant_coord: Coordinate table ``[P, C]`` or ``[B, P, C]``.
        readable_by: Boolean authorization matrix ``[P, P]`` or ``[B, P, P]``.
            ``readable_by[viewer, owner]`` means ``viewer`` may read tokens owned
            by ``owner``.
        mask: Optional valid-token mask ``[B, N]``.
        active_participant: Optional viewer/recipient. If omitted, it is the
            most recent non-assistant participant when ``assistant_id`` is
            supplied.
        assistant_id: Optional assistant participant id. Used as the default
            autoregressive observer and for active-participant inference.
        observer_participant: Optional coordinate frame for generation/training.
            If omitted, ``assistant_id`` is used when present; otherwise the
            active participant is used.

    Returns:
        A ``ParticipantContextTensors`` bundle. Pass ``coord``, ``mask``,
        ``visibility``, and ``observer_coord`` into ``ARTILayer``.
    """

    if participant_ids.ndim != 2:
        raise ValueError("participant_ids must have shape [B, N]")
    batch, tokens = participant_ids.shape
    token_mask = torch.ones(batch, tokens, dtype=torch.bool, device=participant_ids.device) if mask is None else mask.to(device=participant_ids.device, dtype=torch.bool)
    if token_mask.shape != (batch, tokens):
        raise ValueError(f"mask must have shape {(batch, tokens)}, got {tuple(token_mask.shape)}")

    coord_table = _expand_table(participant_coord, batch, "participant_coord").to(device=participant_ids.device)
    readable_table = _expand_table(readable_by, batch, "readable_by").to(device=participant_ids.device, dtype=torch.bool)
    if readable_table.shape[1] != readable_table.shape[2]:
        raise ValueError("readable_by must have shape [P, P] or [B, P, P]")
    if coord_table.shape[1] != readable_table.shape[1]:
        raise ValueError("participant_coord and readable_by must use the same participant count")

    active = _resolve_active_participant(participant_ids, token_mask, active_participant, assistant_id)
    observer = _resolve_participant(observer_participant, batch, participant_ids.device, participant_ids.dtype)
    if observer is None:
        if assistant_id is not None:
            observer = torch.full((batch,), assistant_id, device=participant_ids.device, dtype=participant_ids.dtype)
        else:
            observer = active

    token_coord = _gather_participant_coord(coord_table, participant_ids)
    observer_coord = _gather_participant_coord(coord_table, observer.unsqueeze(1)).squeeze(1)
    visible_sources = _gather_readability(readable_table, active, participant_ids) & token_mask
    visibility = visible_sources.unsqueeze(1).expand(batch, tokens, tokens) & token_mask.unsqueeze(1)
    return ParticipantContextTensors(
        coord=token_coord,
        mask=token_mask & visible_sources,
        visibility=visibility,
        observer_coord=observer_coord,
        active_participant=active,
        observer_participant=observer,
    )


def _expand_table(table: Tensor, batch: int, name: str) -> Tensor:
    if table.ndim == 2:
        return table.unsqueeze(0).expand(batch, -1, -1)
    if table.ndim == 3 and table.shape[0] == batch:
        return table
    raise ValueError(f"{name} must have shape [P, C]/[P, P] or [B, P, C]/[B, P, P]")


def _resolve_active_participant(
    participant_ids: Tensor,
    mask: Tensor,
    active_participant: Tensor | int | None,
    assistant_id: int | None,
) -> Tensor:
    active = _resolve_participant(active_participant, participant_ids.shape[0], participant_ids.device, participant_ids.dtype)
    if active is not None:
        return active
    if assistant_id is None:
        raise ValueError("active_participant is required when assistant_id is not provided")
    return last_non_assistant_participant(participant_ids, mask, assistant_id=assistant_id)


def _resolve_participant(value: Tensor | int | None, batch: int, device: torch.device, dtype: torch.dtype) -> Tensor | None:
    if value is None:
        return None
    if isinstance(value, int):
        return torch.full((batch,), value, device=device, dtype=dtype)
    if value.shape != (batch,):
        raise ValueError(f"participant selector must have shape {(batch,)}, got {tuple(value.shape)}")
    return value.to(device=device, dtype=dtype)


def _gather_participant_coord(coord_table: Tensor, ids: Tensor) -> Tensor:
    if ids.min().item() < 0 or ids.max().item() >= coord_table.shape[1]:
        raise ValueError("participant id is out of range for participant_coord")
    index = ids.to(dtype=torch.long).unsqueeze(-1).expand(*ids.shape, coord_table.shape[-1])
    return coord_table.gather(1, index)


def _gather_readability(readable_table: Tensor, viewer: Tensor, owners: Tensor) -> Tensor:
    if viewer.min().item() < 0 or viewer.max().item() >= readable_table.shape[1]:
        raise ValueError("active participant id is out of range for readable_by")
    if owners.min().item() < 0 or owners.max().item() >= readable_table.shape[2]:
        raise ValueError("participant id is out of range for readable_by")
    batch_indices = torch.arange(owners.shape[0], device=owners.device).unsqueeze(1)
    viewer_indices = viewer.to(dtype=torch.long).unsqueeze(1).expand_as(owners)
    owner_indices = owners.to(dtype=torch.long)
    return readable_table[batch_indices, viewer_indices, owner_indices]
