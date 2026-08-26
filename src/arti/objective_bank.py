"""Bounded Bank operands for predictive write-exposure experiments."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import ClassVar, Literal, overload

import torch
from torch import Tensor, nn


@dataclass(frozen=True)
class ObjectiveExposureOutput:
    """Exposure operand and its addressability diagnostics."""

    exposure: Tensor
    route_weights: Tensor
    operand: Tensor


class ObjectiveExposureBank(nn.Module):
    """Read a bounded write exposure from fixed keys and trainable values.

    This module does not compute a loss, inspect future targets, update another
    Bank, or authorize persistence. It only maps a current/past query to one
    scalar exposure operand per query row.
    """

    _component_reference: ClassVar[str] = "arti/objective-exposure-bank@1"

    def __init__(
        self,
        slots: int,
        query_dim: int,
        *,
        key_layout: Literal["random_unit", "hypercube", "circle"] = "random_unit",
        key_seed: int = 0,
        temperature: float = 1.0,
        min_exposure: float = 0.0,
        max_exposure: float = 1.0,
        init_scale: float = 0.0,
    ) -> None:
        super().__init__()
        if slots <= 0 or query_dim <= 0:
            raise ValueError("slots and query_dim must be positive")
        if key_layout not in {"random_unit", "hypercube", "circle"}:
            raise ValueError(
                "key_layout must be 'random_unit', 'hypercube', or 'circle'"
            )
        if isinstance(key_seed, bool) or not isinstance(key_seed, int):
            raise TypeError("key_seed must be an integer")
        if not 0 <= key_seed < 2**63:
            raise ValueError("key_seed must be in [0, 2**63)")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= min_exposure < max_exposure:
            raise ValueError("exposure bounds must satisfy 0 <= min < max")
        if init_scale < 0.0:
            raise ValueError("init_scale must be non-negative")
        if key_layout == "hypercube" and slots > 2**query_dim:
            raise ValueError("hypercube slots cannot exceed 2**query_dim")
        if key_layout == "circle" and query_dim != 2:
            raise ValueError("circle key layout requires query_dim=2")

        self.slots = int(slots)
        self.query_dim = int(query_dim)
        self.key_layout = key_layout
        self.key_seed = int(key_seed)
        self.temperature = float(temperature)
        self.min_exposure = float(min_exposure)
        self.max_exposure = float(max_exposure)
        self.init_scale = float(init_scale)

        if key_layout == "hypercube":
            corners = list(itertools.product((-1.0, 1.0), repeat=query_dim))
            keys = torch.tensor(corners[:slots], dtype=torch.float32)
        elif key_layout == "circle":
            angles = torch.arange(slots, dtype=torch.float32) * (2.0 * torch.pi / slots)
            keys = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1)
        else:
            generator = torch.Generator(device="cpu").manual_seed(key_seed)
            keys = torch.randn(slots, query_dim, generator=generator)
        keys = torch.nn.functional.normalize(keys, dim=-1)
        self.register_buffer("keys", keys, persistent=True)
        self.values = nn.Parameter(torch.empty(slots))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.init_scale == 0.0:
            nn.init.zeros_(self.values)
            return
        generator = torch.Generator(device="cpu").manual_seed(self.key_seed + 1)
        values = torch.randn(self.slots, generator=generator) * self.init_scale
        with torch.no_grad():
            self.values.copy_(values)

    @overload
    def forward(
        self, query: Tensor, *, return_info: Literal[False] = False
    ) -> Tensor: ...

    @overload
    def forward(
        self, query: Tensor, *, return_info: Literal[True]
    ) -> ObjectiveExposureOutput: ...

    def forward(
        self, query: Tensor, *, return_info: bool = False
    ) -> Tensor | ObjectiveExposureOutput:
        if not isinstance(query, Tensor) or not query.is_floating_point():
            raise TypeError("query must be a floating-point Tensor")
        if query.ndim < 2 or query.shape[-1] != self.query_dim:
            raise ValueError(
                f"query must have trailing shape [..., {self.query_dim}]"
            )
        finite = torch.isfinite(query).all()
        if torch.compiler.is_compiling() or finite.device.type != "cpu":
            torch._assert_async(finite, "query must be finite")
        elif not bool(finite):
            raise ValueError("query must be finite")
        fixed_query = query.detach()
        logits = torch.einsum("...d,sd->...s", fixed_query, self.keys.to(query))
        route = torch.softmax(logits * self.temperature, dim=-1)
        operand = torch.einsum("...s,s->...", route, self.values.to(query))
        unit = torch.sigmoid(operand)
        exposure = self.min_exposure + (
            self.max_exposure - self.min_exposure
        ) * unit
        if not return_info:
            return exposure
        return ObjectiveExposureOutput(exposure, route, operand)


__all__ = ["ObjectiveExposureBank", "ObjectiveExposureOutput"]
