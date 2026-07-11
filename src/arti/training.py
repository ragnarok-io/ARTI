"""Training helpers for ARTI auxiliary objectives."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from .outputs import ARTIOutput


def virtual_recall_alignment_loss(
    layer: Callable[..., ARTIOutput],
    clean_x: Tensor,
    corrupt_x: Tensor,
    *,
    coord: Tensor | None = None,
    corrupt_coord: Tensor | None = None,
    mask: Tensor | None = None,
    corrupt_mask: Tensor | None = None,
    visibility: Tensor | None = None,
    corrupt_visibility: Tensor | None = None,
    frame_operators: Tensor | None = None,
    observer_coord: Tensor | None = None,
    corrupt_observer_coord: Tensor | None = None,
    epoch: int = 1,
    align_start_epoch: int = 2,
    detach_clean_target: bool = True,
) -> tuple[Tensor, ARTIOutput, ARTIOutput]:
    """Align a corrupted-input virtual output to the clean-input latent output.

    Epochs before ``align_start_epoch`` train the virtual output toward zero.
    From ``align_start_epoch`` onward, the corrupted branch's ``virtual_y`` is
    aligned to the clean branch's ``y``. The clean target is detached by default
    so the auxiliary objective teaches the virtual recall path instead of moving
    both branches toward each other.
    """

    if align_start_epoch < 1:
        raise ValueError("align_start_epoch must be >= 1")
    if epoch < 1:
        raise ValueError("epoch must be >= 1")

    corrupt_out = layer(
        corrupt_x,
        coord=coord if corrupt_coord is None else corrupt_coord,
        mask=mask if corrupt_mask is None else corrupt_mask,
        visibility=visibility if corrupt_visibility is None else corrupt_visibility,
        frame_operators=frame_operators,
        observer_coord=observer_coord if corrupt_observer_coord is None else corrupt_observer_coord,
    )
    if corrupt_out.virtual_y is None:
        raise ValueError("layer output must include virtual_y")

    if epoch < align_start_epoch:
        clean_out = layer(clean_x, coord=coord, mask=mask, visibility=visibility, frame_operators=frame_operators, observer_coord=observer_coord)
        target = torch.zeros_like(corrupt_out.virtual_y)
    else:
        if detach_clean_target:
            with torch.no_grad():
                clean_out = layer(clean_x, coord=coord, mask=mask, visibility=visibility, frame_operators=frame_operators, observer_coord=observer_coord)
            target = clean_out.y.detach()
        else:
            clean_out = layer(clean_x, coord=coord, mask=mask, visibility=visibility, frame_operators=frame_operators, observer_coord=observer_coord)
            target = clean_out.y

    loss = F.mse_loss(corrupt_out.virtual_y, target)
    return loss, clean_out, corrupt_out


def experiential_recall_alignment_loss(
    layer: Callable[..., ARTIOutput],
    clean_x: Tensor,
    corrupt_x: Tensor,
    **kwargs,
) -> tuple[Tensor, ARTIOutput, ARTIOutput]:
    """Align corrupted-input recall traces to clean-input latent traces.

    This is the preferred name for ARTI's recall objective. It keeps
    ``virtual_recall_alignment_loss`` as a backward-compatible alias while
    emphasizing that recall is an internal trace of prior signal processing,
    not an external memory of task facts.
    """

    return virtual_recall_alignment_loss(layer, clean_x, corrupt_x, **kwargs)


def experiential_recall_selectivity_loss(
    layer: Callable[..., ARTIOutput],
    clean_x: Tensor,
    corrupt_x: Tensor,
    unseen_x: Tensor,
    *,
    unseen_weight: float = 1.0,
    unseen_coord: Tensor | None = None,
    unseen_mask: Tensor | None = None,
    unseen_visibility: Tensor | None = None,
    unseen_observer_coord: Tensor | None = None,
    **alignment_kwargs,
) -> tuple[Tensor, ARTIOutput, ARTIOutput, ARTIOutput]:
    """Train Recall alignment and unseen-trace suppression together.

    This objective is intended for ``recall_recognition_mode="alignment"``.
    It contains no fixed familiarity threshold: corrupted views of experienced
    signals are aligned to their complete processing trace, while Recall
    influence for unrelated unseen signals is trained toward zero.
    """

    if unseen_weight < 0:
        raise ValueError("unseen_weight must be non-negative")
    alignment_loss, clean_out, corrupt_out = experiential_recall_alignment_loss(
        layer,
        clean_x,
        corrupt_x,
        **alignment_kwargs,
    )
    unseen_out = layer(
        unseen_x,
        coord=unseen_coord,
        mask=unseen_mask,
        visibility=unseen_visibility,
        frame_operators=alignment_kwargs.get("frame_operators"),
        observer_coord=unseen_observer_coord,
    )
    if unseen_out.recall_influence is None:
        raise ValueError("layer output must include recall_influence")
    suppression_loss = unseen_out.recall_influence.square().mean()
    return alignment_loss + unseen_weight * suppression_loss, clean_out, corrupt_out, unseen_out
