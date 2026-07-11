"""Standard outputs returned by ARTI modules."""

from __future__ import annotations

from dataclasses import dataclass, field

from torch import Tensor


@dataclass
class ARTIOutput:
    """Tensor output, pooled representation, experiential recall outputs, and diagnostics."""

    y: Tensor
    pooled: Tensor
    virtual_y: Tensor | None = None
    recall_trace: Tensor | None = None
    recall_prediction: Tensor | None = None
    recall_influence: Tensor | None = None
    diagnostics: dict[str, Tensor] = field(default_factory=dict)
