"""Recall kernel for the vNext selective-compute workspace."""

from __future__ import annotations

from typing import ClassVar

from torch import Tensor, nn

from .nn import Recall
from .recall_refine import AdaptiveRefinePolicy, RefinePolicy


class SelectiveRecallKernel(nn.Module):
    """Apply canonical Recall only to packed intervention queries."""

    _component_reference: ClassVar[str] = "arti/selective-recall-kernel@1"

    def __init__(
        self,
        recall: Recall,
        *,
        refine_policy: RefinePolicy | AdaptiveRefinePolicy | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(recall, Recall):
            raise TypeError("SelectiveRecallKernel requires arti.nn.Recall")
        if refine_policy is None:
            refine_policy = RefinePolicy.fixed(1)
        if not isinstance(refine_policy, (RefinePolicy, AdaptiveRefinePolicy)):
            raise TypeError("refine_policy must be a versioned Recall refine policy")
        self.recall = recall
        self.refine_policy = refine_policy

    def forward(
        self,
        query: Tensor,
        _source: Tensor,
        _factors: Tensor,
        *,
        query_mask: Tensor,
        source_mask: Tensor,
        visibility: Tensor | None = None,
    ) -> Tensor:
        del source_mask, visibility
        return self.recall(
            query,
            mask=query_mask,
            refine_policy=self.refine_policy,
        )


__all__ = ["SelectiveRecallKernel"]
