"""Recall kernel for the runtime selective-compute workspace."""

from __future__ import annotations

from typing import ClassVar

from torch import Tensor, nn

from .nn import Recall
from .execution import AdaptiveExecutionPolicy, ExecutionPolicy


class SelectiveRecallKernel(nn.Module):
    """Apply canonical Recall only to packed intervention queries."""

    _component_reference: ClassVar[str] = "arti/selective-recall-kernel@1"

    def __init__(
        self,
        recall: Recall,
        *,
        execution_policy: ExecutionPolicy | AdaptiveExecutionPolicy | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(recall, Recall):
            raise TypeError("SelectiveRecallKernel requires arti.nn.Recall")
        if execution_policy is None:
            execution_policy = ExecutionPolicy.fixed(1)
        if not isinstance(execution_policy, (ExecutionPolicy, AdaptiveExecutionPolicy)):
            raise TypeError("execution_policy must be a versioned Recall execution policy")
        self.recall = recall
        self.execution_policy = execution_policy

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
            execution_policy=self.execution_policy,
        )


__all__ = ["SelectiveRecallKernel"]
