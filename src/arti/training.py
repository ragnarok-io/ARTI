"""Training helpers for ARTI auxiliary objectives."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor

from .outputs import ARTIOutput


def recall_route_exterior_penalty(
    anchor: ARTIOutput,
    similar: ARTIOutput,
    *,
    tolerance: float = 0.01,
    weight: float = 1.0,
    mask: Tensor | None = None,
) -> Tensor:
    """Penalize only excessive drift between similar Recall read routes.

    This is an exterior regularizer, not a Recall target. The downstream task
    loss remains responsible for teaching the bank what to store. The anchor
    route is detached so the penalty cannot move both views toward collapse.
    """

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if weight < 0:
        raise ValueError("weight must be non-negative")
    if anchor.recall_route is None or similar.recall_route is None:
        raise ValueError("both outputs must include recall_route")
    if anchor.recall_route.shape != similar.recall_route.shape:
        raise ValueError("Recall routes must have identical shapes")
    if anchor.recall_route.ndim < 2:
        raise ValueError("Recall routes must include a route dimension")

    distance = (
        similar.recall_route.float() - anchor.recall_route.detach().float()
    ).square().mean(dim=-1)
    excess = torch.relu(distance - tolerance)
    if mask is not None:
        if mask.shape != distance.shape:
            raise ValueError(
                f"mask must have shape {tuple(distance.shape)}, got {tuple(mask.shape)}"
            )
        valid = mask.to(device=distance.device, dtype=distance.dtype)
        penalty = (excess.square() * valid).sum() / valid.sum().clamp_min(1.0)
    else:
        penalty = excess.square().mean()
    return penalty * weight


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
    """Align a corrupted-input Recall prediction to a clean latent target.

    ``recall_trace`` remains the self-generated private input.
    ``recall_prediction`` is the trainable output: with Recall enabled it is
    the cumulative queried state write after survival activation, projected
    into the same output basis as ``y``. Before
    ``align_start_epoch`` it is trained toward zero.
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
        observer_coord=(
            observer_coord
            if corrupt_observer_coord is None
            else corrupt_observer_coord
        ),
    )
    if corrupt_out.recall_prediction is None:
        raise ValueError("layer output must include recall_prediction")

    clean_kwargs = {
        "coord": coord,
        "mask": mask,
        "visibility": visibility,
        "frame_operators": frame_operators,
        "observer_coord": observer_coord,
    }
    if epoch < align_start_epoch:
        clean_out = layer(clean_x, **clean_kwargs)
        target = torch.zeros_like(corrupt_out.recall_prediction)
    elif detach_clean_target:
        with torch.no_grad():
            clean_out = layer(clean_x, **clean_kwargs)
        target = clean_out.y.detach()
    else:
        clean_out = layer(clean_x, **clean_kwargs)
        target = clean_out.y

    loss = F.mse_loss(corrupt_out.recall_prediction, target)
    return loss, clean_out, corrupt_out


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
