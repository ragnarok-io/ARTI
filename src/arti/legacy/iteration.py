"""Retired Refine-era names for historical inspection only.

The canonical runtime contracts are ``Execution*``, ``LocalIteration*`` and
``WriteIntegrationPolicy``.  These aliases deliberately keep old notebooks
and artifact-inspection code explicit about their legacy dependency.
"""

from ..federal_recall import LocalIterationPolicy, LocalIterationTraceStep
from ..target_bank import WriteIntegrationPolicy


BankLocalRefinePolicy = LocalIterationPolicy
BankLocalRefineTraceStep = LocalIterationTraceStep
WriteRefinePolicy = WriteIntegrationPolicy


__all__ = [
    "BankLocalRefinePolicy",
    "BankLocalRefineTraceStep",
    "WriteRefinePolicy",
]
