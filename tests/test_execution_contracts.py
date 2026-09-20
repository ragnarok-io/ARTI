from __future__ import annotations

import importlib
import inspect

import pytest
import torch

import arti
from arti import legacy
from arti.component_registry import component_spec


def test_execution_contracts_are_the_stable_runtime_surface() -> None:
    fixed = arti.ExecutionPolicy.fixed(3, trace_level="routes")
    adaptive = arti.ExecutionPolicy.adaptive(max_steps=8, min_steps=2)

    assert isinstance(fixed, arti.ExecutionPolicy)
    assert isinstance(adaptive, arti.AdaptiveExecutionPolicy)
    assert isinstance(adaptive.budget, arti.ExecutionBudget)
    assert isinstance(adaptive.stop, arti.ExecutionStop)
    assert component_spec(fixed).reference.startswith("arti/execution-policy@sha256:")
    assert component_spec(adaptive).reference.startswith("arti/execution-policy@sha256:")


def test_retired_refine_contracts_are_legacy_only() -> None:
    retired = legacy.RefinePolicy.fixed(2)

    assert not hasattr(arti, "RefinePolicy")
    assert component_spec(retired).reference.startswith("arti/refine-policy@sha256:")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("arti.recall_refine")


def test_retrieval_route_contracts_have_canonical_names() -> None:
    assert callable(arti.RetrievalRoutePlan)
    assert callable(arti.RetrievalRouteStack)
    assert not hasattr(arti, "RecallRoutePlan")


def test_k_wide_runtime_uses_branch_search_vocabulary() -> None:
    assert callable(arti.alpha.BranchSearchOperation)
    assert callable(arti.alpha.BranchSearchPlan)
    assert callable(arti.alpha.BranchSearchResult)
    assert callable(arti.alpha.BranchSearchPolicy)
    assert callable(arti.alpha.run_branch_search)
    assert not hasattr(arti.alpha, "BatchedRefinePlan")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("arti.batched_refine")
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("arti.branch_refine")


def test_federal_compiler_uses_execution_iteration_vocabulary() -> None:
    assert callable(arti.alpha.FederalIteration)
    assert callable(arti.alpha.FederalStatefulExecutionGraph)
    assert not hasattr(arti.alpha, "FederalRefine")


def test_reader_iteration_schedule_has_canonical_name() -> None:
    assert callable(arti.alpha.ReaderIterationSchedule)
    assert not hasattr(arti.alpha, "ReaderRefineSchedule")


def test_generic_runtime_entrypoints_take_execution_policy() -> None:
    retrieve_parameters = inspect.signature(arti.Retrieve.forward).parameters
    search_parameters = inspect.signature(arti.alpha.run_branch_search).parameters

    assert "execution_policy" in retrieve_parameters
    assert "execution_policy" in search_parameters
    assert "refine_policy" not in retrieve_parameters
    assert "refine_policy" not in search_parameters


def test_execution_adapters_use_executor_vocabulary() -> None:
    assert callable(arti.nn.RecallExecutor)
    assert callable(arti.RetrieveExecutor)
    assert not hasattr(arti.nn, "RecallRefiner")
    assert not hasattr(arti, "RetrieveRefiner")


def test_execution_rng_plan_uses_iteration_index_contract() -> None:
    plan = arti.alpha.ExecutionRNGPlan(
        seed=7,
        run_nonce="execution-vocabulary",
        stream_key="encoder.block-1.retrieve",
        sample_keys=("sample",),
    )
    source = plan.bind(torch.zeros((1, 1), dtype=torch.long))

    assert plan.contract_ref.startswith("arti/execution-rng-plan@sha256:")
    assert source.uniform("half-survival", 0, torch.zeros((1, 2))).shape == (1, 2)
    with pytest.raises(TypeError):
        source.uniform("half-survival", refine_step=0, reference=torch.zeros((1, 2)))


def test_selective_recall_provenance_uses_execution_policy() -> None:
    from arti.selective_recall import SelectiveRecallKernel

    kernel = SelectiveRecallKernel(
        arti.nn.Recall(dim=2, slots=2),
        execution_policy=arti.ExecutionPolicy.fixed(1),
    )
    spec = component_spec(kernel)

    assert spec.reference.startswith("arti/selective-recall-kernel@sha256:")
    assert "execution_policy" in spec.config
    assert "refine_policy" not in spec.config


def test_fit_helpers_use_retrieval_iteration_vocabulary() -> None:
    assert callable(arti.set_retrieval_iteration_steps)
    assert callable(arti.set_retrieval_iteration_schedule)
    assert not hasattr(arti, "set_recall_refine_steps")


def test_gpu_resident_contract_uses_iteration_steps() -> None:
    from arti.gpu_resident import FixedResidentBucket, FormulaResidentOperation

    bucket_parameters = inspect.signature(FixedResidentBucket).parameters
    operation_parameters = inspect.signature(FormulaResidentOperation).parameters

    assert "iteration_steps" in bucket_parameters
    assert "iteration_steps" in operation_parameters
    assert "refine_steps" not in bucket_parameters
    assert "refine_steps" not in operation_parameters
