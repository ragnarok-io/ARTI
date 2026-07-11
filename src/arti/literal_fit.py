"""Small tensor-native fit recipe for literal sequence decoders."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch
import torch.nn.functional as F
from torch import Tensor

from .literal_decoder import LiteralSequenceDecoder


@dataclass(frozen=True)
class LiteralFitResult:
    """Summary returned by :func:`fit_literal_sequence`."""

    steps: int
    examples: int
    final_loss: float
    losses: tuple[float, ...]


def fit_literal_sequence(
    decoder: LiteralSequenceDecoder,
    batches: Iterable[Mapping[str, object]],
    *,
    steps: int,
    lr: float = 1e-3,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip_norm: float | None = 1.0,
) -> LiteralFitResult:
    """Fit a decoder with standard masked local-slot cross-entropy.

    Each batch must provide ``context``, ``output_vocab``, and ``teacher_ids``.
    Optional fields are ``target_mask``, ``loss_weights``, ``output_mask``, and
    ``batched_vocab``. Tensors stay on the caller-selected device; this helper
    deliberately does not own data loading, distributed execution, or task
    semantics.
    """

    if steps <= 0:
        raise ValueError("steps must be positive")
    if lr <= 0:
        raise ValueError("lr must be positive")
    if grad_clip_norm is not None and grad_clip_norm <= 0:
        raise ValueError("grad_clip_norm must be positive or None")
    trainable = [parameter for parameter in decoder.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("decoder has no trainable parameters")
    resolved_optimizer = optimizer or torch.optim.AdamW(trainable, lr=lr)
    iterator = iter(batches)
    losses: list[float] = []
    examples = 0
    decoder.train()
    for _step in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(batches)
            try:
                batch = next(iterator)
            except StopIteration as exc:
                raise ValueError("batches must yield at least one batch") from exc
        context = _tensor_field(batch, "context")
        output_vocab = batch.get("output_vocab")
        if output_vocab is None:
            raise ValueError("batch is missing output_vocab")
        teacher_ids = _tensor_field(batch, "teacher_ids").to(dtype=torch.long)
        output = decoder(
            context,
            output_vocab,  # type: ignore[arg-type]
            teacher_ids=teacher_ids,
            output_mask=_optional_tensor(batch, "output_mask"),
            batched_vocab=bool(batch.get("batched_vocab", False)),
        )
        token_losses = F.cross_entropy(
            output.logits.flatten(0, 1),
            teacher_ids.flatten(),
            reduction="none",
        ).reshape_as(teacher_ids)
        target_mask = _optional_tensor(batch, "target_mask")
        weights = torch.ones_like(token_losses)
        if target_mask is not None:
            if target_mask.shape != teacher_ids.shape:
                raise ValueError("target_mask must match teacher_ids shape")
            weights = weights * target_mask.to(device=weights.device, dtype=weights.dtype)
        loss_weights = _optional_tensor(batch, "loss_weights")
        if loss_weights is not None:
            if loss_weights.shape != teacher_ids.shape:
                raise ValueError("loss_weights must match teacher_ids shape")
            weights = weights * loss_weights.to(device=weights.device, dtype=weights.dtype)
        if not bool((weights > 0).any()):
            raise ValueError("each fit batch must contain at least one positive-loss token")
        loss = (token_losses * weights).sum() / weights.sum()
        resolved_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(trainable, grad_clip_norm)
        resolved_optimizer.step()
        losses.append(float(loss.detach().item()))
        examples += int(context.shape[0])
    return LiteralFitResult(steps=steps, examples=examples, final_loss=losses[-1], losses=tuple(losses))


def _tensor_field(batch: Mapping[str, object], name: str) -> Tensor:
    value = batch.get(name)
    if not isinstance(value, Tensor):
        raise ValueError(f"batch field {name} must be a Tensor")
    return value


def _optional_tensor(batch: Mapping[str, object], name: str) -> Tensor | None:
    value = batch.get(name)
    if value is None:
        return None
    if not isinstance(value, Tensor):
        raise ValueError(f"batch field {name} must be a Tensor when provided")
    return value


__all__ = ["LiteralFitResult", "fit_literal_sequence"]
