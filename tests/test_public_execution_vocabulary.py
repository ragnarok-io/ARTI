from __future__ import annotations

import arti
from arti import legacy, mechanisms


def test_root_and_mechanisms_publish_execution_vocabulary() -> None:
    assert arti.Retrieve is mechanisms.Retrieve
    assert arti.ExecutionPolicy is mechanisms.ExecutionPolicy
    assert arti.RetrieveRefiner is mechanisms.RetrieveRefiner
    assert not hasattr(arti, "Recall")
    assert not hasattr(arti, "RefinePolicy")
    assert legacy.Recall.__name__ == "Recall"
    assert legacy.RefinePolicy.__name__ == "RefinePolicy"


def test_retrieve_and_execution_policy_have_distinct_contracts() -> None:
    retrieve = arti.Retrieve(dim=4, slots=8, activation="none")
    fixed = arti.ExecutionPolicy.fixed(3, trace_level="summary")
    adaptive = arti.ExecutionPolicy.adaptive(max_steps=6, min_steps=2)

    assert arti.component_ref(retrieve).startswith("arti/retrieve@sha256:")
    assert arti.component_ref(fixed).startswith("arti/execution-policy@sha256:")
    assert arti.component_ref(adaptive).startswith("arti/execution-policy@sha256:")
    assert fixed.max_steps == fixed.min_steps == 3
    assert adaptive.budget.max_steps == 6
    assert adaptive.budget.min_steps == 2
