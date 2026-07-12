"""Alpha online-state Recall with learned read and constrained write rules."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .nn import Half


class StatefulRecall(nn.Module):
    """Recall processed latent signals through an explicit fixed-size state.

    The module parameters learn how to read and write Recall. Inference-time
    adaptation changes only the ``keys``, ``values``, and ``strengths`` tensors
    returned by :meth:`update`; model parameters remain read-only.
    """

    state_names = ("keys", "values", "strengths")

    def __init__(
        self,
        dim: int,
        *,
        slots: int = 16,
        key_dim: int | None = None,
        recognition_threshold: float = 0.55,
        recognition_temperature: float = 0.1,
        write_rate: float = 0.75,
        decay: float = 0.999,
        use_half: bool = True,
        learnable_dynamics: bool = True,
    ) -> None:
        super().__init__()
        if dim <= 0 or slots <= 0:
            raise ValueError("dim and slots must be positive")
        resolved_key_dim = dim if key_dim is None else int(key_dim)
        if resolved_key_dim <= 0:
            raise ValueError("key_dim must be positive")
        if recognition_temperature <= 0:
            raise ValueError("recognition_temperature must be positive")
        if not 0 < write_rate <= 1 or not 0 <= decay <= 1:
            raise ValueError("write_rate must be in (0, 1] and decay in [0, 1]")
        self.dim = int(dim)
        self.slots = int(slots)
        self.key_dim = resolved_key_dim
        self.recognition_threshold = float(recognition_threshold)
        self.recognition_temperature = float(recognition_temperature)
        self.use_half = bool(use_half)
        self.learnable_dynamics = bool(learnable_dynamics)

        self.query = nn.Linear(dim, resolved_key_dim, bias=False)
        self.key = nn.Linear(dim, resolved_key_dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.emit = nn.Linear(dim, dim, bias=False)
        self.write_quality = nn.Linear(resolved_key_dim + dim, 1)
        self.slot_anchors = nn.Parameter(torch.randn(slots, resolved_key_dim) * resolved_key_dim**-0.5)
        self.write_rate_logit = nn.Parameter(_logit(write_rate), requires_grad=learnable_dynamics)
        self.decay_logit = nn.Parameter(_logit(decay), requires_grad=learnable_dynamics)
        self.survival = Half() if use_half else nn.Identity()

    @property
    def write_rate(self) -> float:
        return float(torch.sigmoid(self.write_rate_logit.detach()))

    @property
    def decay(self) -> float:
        return float(torch.sigmoid(self.decay_logit.detach()))

    def initial_state(self, batch_size: int, *, device=None, dtype=None) -> dict[str, Tensor]:
        """Create an empty fixed-capacity Recall state."""

        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        reference = self.slot_anchors
        options = {"device": reference.device if device is None else device, "dtype": reference.dtype if dtype is None else dtype}
        return {
            "keys": torch.zeros(batch_size, self.slots, self.key_dim, **options),
            "values": torch.zeros(batch_size, self.slots, self.dim, **options),
            "strengths": torch.zeros(batch_size, self.slots, **options),
        }

    def read(
        self,
        x: Tensor,
        keys: Tensor,
        values: Tensor,
        strengths: Tensor,
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Read recognized traces without mutating Recall state."""

        sequence, squeezed = self._sequence(x)
        self._validate_state(sequence, keys, values, strengths)
        valid = self._valid_mask(sequence, mask)
        query = F.normalize(self.query(sequence), dim=-1, eps=1e-6)
        normalized_keys = F.normalize(keys, dim=-1, eps=1e-6)
        similarity = torch.einsum("bnr,bkr->bnk", query, normalized_keys)
        occupancy = strengths.clamp(0, 1)
        logits = similarity / self.recognition_temperature + torch.log(occupancy.unsqueeze(1) + 1e-6)
        weights = torch.softmax(logits, dim=-1)
        context = torch.einsum("bnk,bkd->bnd", weights, values)
        best_similarity = similarity.max(dim=-1).values
        available = torch.einsum("bnk,bk->bn", weights, occupancy)
        recognition = torch.sigmoid(
            (best_similarity - self.recognition_threshold) / self.recognition_temperature
        ) * available * valid
        raw_delta = self.emit(context) * recognition.unsqueeze(-1)
        delta = self.survival(raw_delta) * valid.unsqueeze(-1)
        pooled = (sequence * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
        trace_key = F.normalize(self.key(pooled), dim=-1, eps=1e-6)
        trace_value = self.value(pooled)
        y = sequence + delta
        if squeezed:
            y, delta, raw_delta, recognition, weights = y[:, 0], delta[:, 0], raw_delta[:, 0], recognition[:, 0], weights[:, 0]
        return {
            "y": y,
            "delta": delta,
            "raw_delta": raw_delta,
            "recognition": recognition,
            "weights": weights,
            "trace_key": trace_key,
            "trace_value": trace_value,
        }

    def update(
        self,
        trace_key: Tensor,
        observed: Tensor,
        keys: Tensor,
        values: Tensor,
        strengths: Tensor,
        mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Return a corrected next state; no module parameter is modified."""

        sequence, _ = self._sequence(observed)
        self._validate_state(sequence, keys, values, strengths)
        if trace_key.shape != (sequence.shape[0], self.key_dim):
            raise ValueError(f"trace_key must have shape [B, {self.key_dim}]")
        valid = self._valid_mask(sequence, mask)
        pooled = (sequence * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(dim=1, keepdim=True).clamp_min(1)
        target_key = F.normalize(trace_key, dim=-1, eps=1e-6)
        target_value = self.value(pooled)
        occupied_similarity = torch.einsum("br,bkr->bk", target_key, F.normalize(keys, dim=-1, eps=1e-6))
        empty_similarity = torch.einsum("br,kr->bk", target_key, F.normalize(self.slot_anchors, dim=-1, eps=1e-6))
        occupancy = strengths.clamp(0, 1)
        assignment_logits = occupancy * occupied_similarity + (1 - occupancy) * empty_similarity
        assignment = torch.softmax(assignment_logits / self.recognition_temperature, dim=-1)
        quality = torch.sigmoid(self.write_quality(torch.cat([target_key, target_value], dim=-1))).squeeze(-1)
        rate = torch.sigmoid(self.write_rate_logit).to(sequence) * quality.unsqueeze(-1) * assignment
        decay = torch.sigmoid(self.decay_logit).to(sequence)
        next_keys = decay * keys + rate.unsqueeze(-1) * (target_key.unsqueeze(1) - keys)
        next_values = decay * values + rate.unsqueeze(-1) * (target_value.unsqueeze(1) - values)
        next_strengths = (decay * strengths + rate * (1 - strengths)).clamp(0, 1)
        return {
            "keys": next_keys,
            "values": next_values,
            "strengths": next_strengths,
            "write_assignment": assignment,
            "write_quality": quality,
        }

    def forward(self, x: Tensor, keys: Tensor, values: Tensor, strengths: Tensor, mask: Tensor | None = None) -> dict[str, Tensor]:
        return self.read(x, keys, values, strengths, mask)

    def _sequence(self, x: Tensor) -> tuple[Tensor, bool]:
        if x.ndim == 2 and x.shape[-1] == self.dim:
            return x.unsqueeze(1), True
        if x.ndim == 3 and x.shape[-1] == self.dim:
            return x, False
        raise ValueError(f"x must have shape [B, {self.dim}] or [B, N, {self.dim}]")

    def _validate_state(self, sequence: Tensor, keys: Tensor, values: Tensor, strengths: Tensor) -> None:
        batch = sequence.shape[0]
        if keys.shape != (batch, self.slots, self.key_dim):
            raise ValueError(f"keys must have shape [B, {self.slots}, {self.key_dim}]")
        if values.shape != (batch, self.slots, self.dim):
            raise ValueError(f"values must have shape [B, {self.slots}, {self.dim}]")
        if strengths.shape != (batch, self.slots):
            raise ValueError(f"strengths must have shape [B, {self.slots}]")

    @staticmethod
    def _valid_mask(sequence: Tensor, mask: Tensor | None) -> Tensor:
        expected = sequence.shape[:2]
        if mask is None:
            return torch.ones(expected, device=sequence.device, dtype=sequence.dtype)
        if mask.shape != expected:
            raise ValueError(f"mask must have shape {tuple(expected)}, got {tuple(mask.shape)}")
        return mask.to(device=sequence.device, dtype=sequence.dtype)


def _logit(value: float) -> Tensor:
    clipped = torch.tensor(float(value)).clamp(1e-6, 1 - 1e-6)
    return torch.logit(clipped)
