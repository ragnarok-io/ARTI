"""Typed neural control for stopping a Recall refine trajectory."""

from __future__ import annotations

from typing import ClassVar, Literal, NamedTuple

import torch
from torch import Tensor, nn


RefineExitScope = Literal["token", "branch"]
RefineExitInputKind = Literal["predicate", "logit"]


class RefineExitRequest(NamedTuple):
    """Tensor-only request emitted after one committed Refine transition."""

    requested: Tensor
    score: Tensor
    finite: Tensor


RefineExitRequest._component_reference = "arti/refine-exit-request@1"  # type: ignore[attr-defined]


class FormulaRefineExit(nn.Module):
    """Convert a typed predicate or halt logit into a hard exit request.

    The atom is stateless. A trainable module may produce ``signal`` upstream;
    the unmodified floating-point score remains available for surrogate losses.
    """

    _component_reference: ClassVar[str] = "arti/formula-atom-refine-exit@1"

    def __init__(
        self,
        *,
        input_kind: RefineExitInputKind = "logit",
        scope: RefineExitScope = "token",
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        if input_kind not in {"predicate", "logit"}:
            raise ValueError("input_kind must be 'predicate' or 'logit'")
        if scope not in {"token", "branch"}:
            raise ValueError("scope must be 'token' or 'branch'")
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise TypeError("threshold must be a finite number")
        if not torch.isfinite(torch.tensor(float(threshold))):
            raise ValueError("threshold must be finite")
        self.input_kind = input_kind
        self.scope = scope
        self.threshold = float(threshold)

    def _normalize_signal(self, signal: Tensor, mask: Tensor) -> Tensor:
        if self.scope == "token":
            if signal.ndim == 3 and signal.shape[-1] == 1:
                signal = signal.squeeze(-1)
            if signal.shape != mask.shape:
                raise ValueError("token exit signal must have shape [B, N] or [B, N, 1]")
            return signal

        if signal.ndim == 2 and signal.shape[1] == 1:
            signal = signal.squeeze(1)
        if signal.ndim != 1 or signal.shape[0] != mask.shape[0]:
            raise ValueError("branch exit signal must have shape [B] or [B, 1]")
        return signal.unsqueeze(1).expand_as(mask)

    def forward(self, signal: Tensor, *, mask: Tensor) -> RefineExitRequest:
        if not isinstance(signal, Tensor):
            raise TypeError("signal must be a Tensor")
        if not isinstance(mask, Tensor) or mask.ndim != 2 or mask.dtype != torch.bool:
            raise TypeError("mask must be a boolean Tensor with shape [B, N]")
        if signal.device != mask.device:
            raise ValueError("signal and mask must use the same device")
        signal = self._normalize_signal(signal, mask)
        if self.input_kind == "predicate":
            if signal.dtype != torch.bool:
                raise TypeError("predicate exit signal must be boolean")
            finite = torch.ones_like(mask)
            score = signal.to(dtype=torch.get_default_dtype())
            requested = signal
        else:
            if not signal.is_floating_point():
                raise TypeError("logit exit signal must be floating point")
            finite = torch.isfinite(signal)
            score = signal
            requested = finite & (signal >= self.threshold)
        finite = finite | ~mask
        requested = requested & mask
        score = torch.where(mask, score, torch.zeros_like(score))
        return RefineExitRequest(requested=requested, score=score, finite=finite)

    def extra_repr(self) -> str:
        return (
            f"input_kind={self.input_kind!r}, scope={self.scope!r}, "
            f"threshold={self.threshold}"
        )


class RefineExitControl(nn.Module):
    """Compose a caller-owned neural signal source with ``FormulaRefineExit``."""

    _component_reference: ClassVar[str] = "arti/refine-exit-control@1"

    def __init__(
        self,
        source: nn.Module,
        *,
        atom: FormulaRefineExit | None = None,
        input_kind: RefineExitInputKind = "logit",
        scope: RefineExitScope = "token",
        threshold: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(source, nn.Module):
            raise TypeError("source must be a torch.nn.Module")
        if atom is not None and not isinstance(atom, FormulaRefineExit):
            raise TypeError("atom must be FormulaRefineExit or None")
        if atom is not None and (
            input_kind != "logit" or scope != "token" or threshold != 0.0
        ):
            raise ValueError(
                "input_kind, scope, and threshold cannot be combined with an explicit atom"
            )
        self.source = source
        self.atom = atom or FormulaRefineExit(
            input_kind=input_kind,
            scope=scope,
            threshold=threshold,
        )

    def forward(self, state: Tensor, *, mask: Tensor) -> RefineExitRequest:
        if not isinstance(state, Tensor) or state.ndim != 3 or not state.is_floating_point():
            raise TypeError("state must be a floating-point Tensor with shape [B, N, D]")
        if not isinstance(mask, Tensor) or mask.shape != state.shape[:2] or mask.dtype != torch.bool:
            raise TypeError("mask must be a boolean Tensor with shape [B, N]")
        if mask.device != state.device:
            raise ValueError("state and mask must use the same device")

        batch, tokens, dim = state.shape
        if self.atom.scope == "token":
            signal = self.source(state.reshape(batch * tokens, dim))
            if signal.ndim == 2 and signal.shape == (batch * tokens, 1):
                signal = signal.squeeze(-1)
            if signal.shape != (batch * tokens,):
                raise ValueError("token exit source must map [B*N, D] to [B*N] or [B*N, 1]")
            signal = signal.reshape(batch, tokens)
        else:
            weight = mask.to(dtype=state.dtype).unsqueeze(-1)
            pooled = (state * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1)
            signal = self.source(pooled)
            if signal.ndim == 2 and signal.shape == (batch, 1):
                signal = signal.squeeze(-1)
            if signal.shape != (batch,):
                raise ValueError("branch exit source must map [B, D] to [B] or [B, 1]")
        return self.atom(signal, mask=mask)


__all__ = [
    "FormulaRefineExit",
    "RefineExitControl",
    "RefineExitInputKind",
    "RefineExitRequest",
    "RefineExitScope",
]
