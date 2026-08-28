"""Execution-derived Formula receipts for the K=2 branch reference harness."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import ClassVar, Sequence

import torch
from torch import Tensor

from .branch_refine import (
    BranchBatchSpec,
    BranchRunReceipt,
    BranchWorkReceipt,
    K2BranchHarness,
    OverlayProposal,
)
from .formula_attention import ActiveWorkspace
from .formula_fabric import (
    FormulaCommitBlendTrace,
    FormulaFabricCompute,
    FormulaFabricTrace,
    FormulaRoutePlan,
)
from .tensor_binding import ExternalTensorProposal
from .tensor_transaction import TensorTransactionContractError


_EXECUTION_FACTORY_TOKEN = object()
_SCORE_FACTORY_TOKEN = object()


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def tensor_content_fingerprint(value: Tensor) -> str:
    if value.device.type != "cpu":
        raise TensorTransactionContractError("S3 Formula receipts require CPU tensors")
    owned = value.detach().contiguous()
    raw = owned.view(torch.uint8).numpy().tobytes()
    return _fingerprint(
        {
            "dtype": str(owned.dtype),
            "shape": list(owned.shape),
            "bytes_sha256": hashlib.sha256(raw).hexdigest(),
        }
    )


def workspace_fingerprint(workspace: ActiveWorkspace) -> str:
    """Fingerprint one concrete branch workspace without preserving tensor data."""

    if not isinstance(workspace, ActiveWorkspace):
        raise TensorTransactionContractError("workspace must be ActiveWorkspace")
    return _fingerprint(
        {
            "value": tensor_content_fingerprint(workspace.value),
            "validity": tensor_content_fingerprint(workspace.validity),
            "exposed": tensor_content_fingerprint(workspace.exposed),
            "intervened": tensor_content_fingerprint(workspace.intervened),
        }
    )


def formula_route_fingerprint(route: FormulaRoutePlan) -> str:
    """Fingerprint an owned, fixed Formula route plan."""

    if not isinstance(route, FormulaRoutePlan):
        raise TensorTransactionContractError("route must be FormulaRoutePlan")
    return _fingerprint(
        {
            "weights": tensor_content_fingerprint(route.weights),
            "valid": tensor_content_fingerprint(route.valid_mask),
            "fire": tensor_content_fingerprint(route.fire_mask),
            "commit": tensor_content_fingerprint(route.commit_mask),
            "estimator": route.estimator,
        }
    )


def deterministic_formula_rng_fingerprint() -> str:
    """Canonical RNG receipt for the deterministic fixed-route CPU executor."""

    return _fingerprint(
        {
            "ref": "arti/formula-branch-execution@1",
            "rng": "none",
        }
    )


def _trace_formula(trace: FormulaFabricTrace | FormulaCommitBlendTrace) -> FormulaFabricTrace:
    return trace.formula if isinstance(trace, FormulaCommitBlendTrace) else trace


def _trace_fingerprint(trace: FormulaFabricTrace | FormulaCommitBlendTrace) -> str:
    formula = _trace_formula(trace)
    content: dict[str, object] = {
        "formula_id": tensor_content_fingerprint(formula.formula_id),
        "input_slot": tensor_content_fingerprint(formula.input_slot),
        "input_version": tensor_content_fingerprint(formula.input_version),
        "output_slot": tensor_content_fingerprint(formula.output_slot),
        "output_version": tensor_content_fingerprint(formula.output_version),
        "valid": tensor_content_fingerprint(formula.valid_mask),
        "fire": tensor_content_fingerprint(formula.fire_mask),
        "commit": tensor_content_fingerprint(formula.commit_mask),
        "program_fingerprint": formula.program_fingerprint,
        "estimator": formula.estimator,
    }
    if isinstance(trace, FormulaCommitBlendTrace):
        content["commit_weights"] = tensor_content_fingerprint(trace.weights)
    return _fingerprint(content)


@dataclass(frozen=True, init=False)
class FormulaBranchExecution:
    """A branch candidate whose counters derive from actual Formula traces."""

    branch_id: str
    spec_fingerprint: str
    workspace: ActiveWorkspace
    work: BranchWorkReceipt
    step_input_fingerprints: tuple[str, ...]
    step_output_fingerprints: tuple[str, ...]
    trace_fingerprints: tuple[str, ...]
    execution_fingerprint: str
    _runtime_contract_ref: ClassVar[str] = "arti/formula-branch-execution@1"

    def __init__(
        self,
        *,
        branch_id: str,
        spec_fingerprint: str,
        workspace: ActiveWorkspace,
        work: BranchWorkReceipt,
        step_input_fingerprints: tuple[str, ...],
        step_output_fingerprints: tuple[str, ...],
        trace_fingerprints: tuple[str, ...],
        execution_fingerprint: str,
        _factory_token: object,
    ) -> None:
        if _factory_token is not _EXECUTION_FACTORY_TOKEN:
            raise TensorTransactionContractError(
                "FormulaBranchExecution must come from execute_fixed_formula_branch"
            )
        owned = ActiveWorkspace(
            workspace.value.detach().clone(),
            workspace.validity.detach().clone(),
            workspace.exposed.detach().clone(),
            workspace.intervened.detach().clone(),
        )
        object.__setattr__(self, "branch_id", branch_id)
        object.__setattr__(self, "spec_fingerprint", spec_fingerprint)
        object.__setattr__(self, "workspace", owned)
        object.__setattr__(self, "work", work)
        object.__setattr__(self, "step_input_fingerprints", step_input_fingerprints)
        object.__setattr__(self, "step_output_fingerprints", step_output_fingerprints)
        object.__setattr__(self, "trace_fingerprints", trace_fingerprints)
        object.__setattr__(self, "execution_fingerprint", execution_fingerprint)

    def overlay(
        self,
        spec: BranchBatchSpec,
        proposals: Sequence[ExternalTensorProposal],
    ) -> OverlayProposal:
        """Bind complete next-state proposals to this verified execution."""

        if spec.fingerprint != self.spec_fingerprint:
            raise TensorTransactionContractError("execution does not bind this branch spec")
        normalized = tuple(proposals)
        if len(normalized) != 1:
            raise TensorTransactionContractError(
                "Formula branch v1 requires one complete workspace proposal"
            )
        candidate = normalized[0].value
        if (
            candidate.dtype != self.workspace.value.dtype
            or candidate.shape != self.workspace.value.shape
            or not torch.equal(candidate, self.workspace.value)
        ):
            raise TensorTransactionContractError(
                "staged Formula candidate must equal the final executed workspace"
            )
        return OverlayProposal(
            branch_id=self.branch_id,
            spec_fingerprint=self.spec_fingerprint,
            proposals=normalized,
            work=self.work,
            step_input_fingerprints=self.step_input_fingerprints,
            step_output_fingerprints=self.step_output_fingerprints,
            execution_fingerprint=self.execution_fingerprint,
        )


def execute_fixed_formula_branch(
    compute: FormulaFabricCompute,
    workspace: ActiveWorkspace,
    route: FormulaRoutePlan,
    spec: BranchBatchSpec,
    *,
    branch_id: str,
    steps: int,
    factors: Tensor | None = None,
) -> FormulaBranchExecution:
    """Execute one fixed-route CPU branch and derive receipts from real traces."""

    if not isinstance(compute, FormulaFabricCompute):
        raise TensorTransactionContractError("compute must be FormulaFabricCompute")
    if branch_id not in spec.branch_ids:
        raise TensorTransactionContractError("branch_id is not in BranchBatchSpec")
    branch_index = spec.branch_ids.index(branch_id)
    budget = spec.budgets[branch_index]
    if isinstance(steps, bool) or not isinstance(steps, int) or not budget.min_steps <= steps <= budget.max_steps:
        raise TensorTransactionContractError("steps violate the branch budget")
    if route.estimator != "hard":
        raise TensorTransactionContractError("S3 requires a fixed hard Formula route")
    if compute._component_reference != spec.executor_ref:
        raise TensorTransactionContractError("executor_ref does not match Formula compute")
    if compute.fabric.program.fingerprint != spec.program_fingerprint:
        raise TensorTransactionContractError("program fingerprint does not match Formula compute")
    if formula_route_fingerprint(route) != spec.route_fingerprint:
        raise TensorTransactionContractError("route fingerprint does not match fixed route")
    if spec.rng_fingerprint != deterministic_formula_rng_fingerprint():
        raise TensorTransactionContractError(
            "fixed Formula execution requires the canonical no-RNG receipt"
        )
    if workspace_fingerprint(workspace) != spec.input_fingerprint:
        raise TensorTransactionContractError("workspace does not match shared branch input")
    if workspace.value.device.type != "cpu":
        raise TensorTransactionContractError("S3 Formula branch execution is CPU-only")

    current = workspace
    inputs: list[str] = []
    outputs: list[str] = []
    traces: list[str] = []
    formula_cells = 0
    fire_count = 0
    commit_count = 0
    operation_count = 0
    with torch.no_grad():
        for _step in range(steps):
            inputs.append(workspace_fingerprint(current))
            current, trace = compute(
                current,
                factors,
                formula_route=route,
                return_info=True,
            )
            formula = _trace_formula(trace)
            formula_cells += int(formula.valid_mask.sum().item())
            step_fire = int(formula.fire_mask.sum().item())
            step_commit = int(formula.commit_mask.sum().item())
            fire_count += step_fire
            commit_count += step_commit
            operation_count += step_fire
            outputs.append(workspace_fingerprint(current))
            traces.append(_trace_fingerprint(trace))
    finite = bool(torch.isfinite(current.value).all())
    if not finite:
        raise TensorTransactionContractError("Formula branch produced non-finite state")
    work = BranchWorkReceipt(
        actual_steps=steps,
        formula_cells=formula_cells,
        route_applications=steps,
        fire_count=fire_count,
        commit_count=commit_count,
        operation_count=operation_count,
        stop_reason="budget",
        residual_fingerprint=tensor_content_fingerprint(current.value - workspace.value),
        finite=True,
    )
    factor_fingerprint = None if factors is None else tensor_content_fingerprint(factors)
    execution_fingerprint = _fingerprint(
        {
            "ref": FormulaBranchExecution._runtime_contract_ref,
            "spec_fingerprint": spec.fingerprint,
            "branch_id": branch_id,
            "compute_config": compute.execution_config_fingerprint,
            "factor_fingerprint": factor_fingerprint,
            "step_inputs": inputs,
            "step_outputs": outputs,
            "traces": traces,
            "work": {
                "steps": steps,
                "formula_cells": formula_cells,
                "route_applications": steps,
                "fire_count": fire_count,
                "commit_count": commit_count,
                "operation_count": operation_count,
            },
        }
    )
    return FormulaBranchExecution(
        branch_id=branch_id,
        spec_fingerprint=spec.fingerprint,
        workspace=current,
        work=work,
        step_input_fingerprints=tuple(inputs),
        step_output_fingerprints=tuple(outputs),
        trace_fingerprints=tuple(traces),
        execution_fingerprint=execution_fingerprint,
        _factory_token=_EXECUTION_FACTORY_TOKEN,
    )


@dataclass(frozen=True, init=False)
class FrozenMSEScoreReceipt:
    """Score-only receipt; it has no commit authority and no trainable state."""

    spec_fingerprint: str
    candidate_execution_fingerprints: tuple[str, str]
    candidate_output_fingerprints: tuple[str, str]
    future_fingerprint: str
    scores: tuple[float, float]
    scorer_ref: str = "arti/frozen-mse-scorer@1"

    def __init__(
        self,
        *,
        spec_fingerprint: str,
        candidate_execution_fingerprints: tuple[str, str],
        candidate_output_fingerprints: tuple[str, str],
        future_fingerprint: str,
        scores: tuple[float, float],
        _factory_token: object,
    ) -> None:
        if _factory_token is not _SCORE_FACTORY_TOKEN:
            raise TensorTransactionContractError(
                "FrozenMSEScoreReceipt must come from score_formula_branches"
            )
        object.__setattr__(self, "spec_fingerprint", spec_fingerprint)
        object.__setattr__(
            self,
            "candidate_execution_fingerprints",
            candidate_execution_fingerprints,
        )
        object.__setattr__(
            self,
            "candidate_output_fingerprints",
            candidate_output_fingerprints,
        )
        object.__setattr__(self, "future_fingerprint", future_fingerprint)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "scorer_ref", "arti/frozen-mse-scorer@1")


def score_formula_branches(
    candidates: tuple[FormulaBranchExecution, FormulaBranchExecution],
    future: Tensor,
    spec: BranchBatchSpec,
) -> FrozenMSEScoreReceipt:
    """Evaluate two completed candidates against one exogenous future tensor."""

    if len(candidates) != 2 or candidates[0].branch_id == candidates[1].branch_id:
        raise TensorTransactionContractError("scoring requires two unique branch candidates")
    if tuple(candidate.branch_id for candidate in candidates) != spec.branch_ids:
        raise TensorTransactionContractError("candidate order must match BranchBatchSpec")
    if any(candidate.spec_fingerprint != spec.fingerprint for candidate in candidates):
        raise TensorTransactionContractError("candidate does not bind the scoring spec")
    if future.device.type != "cpu" or future.requires_grad:
        raise TensorTransactionContractError("future must be a detached CPU tensor")
    if (future.is_floating_point() or future.is_complex()) and not bool(
        torch.isfinite(future).all()
    ):
        raise TensorTransactionContractError("future must contain only finite values")
    future_fingerprint = tensor_content_fingerprint(future)
    if future_fingerprint != spec.future_tape_fingerprint:
        raise TensorTransactionContractError("future tensor does not match branch provenance")
    scores: list[float] = []
    with torch.no_grad():
        for candidate in candidates:
            if candidate.workspace.value.shape != future.shape:
                raise TensorTransactionContractError("future shape must match branch output")
            score = float(torch.mean((candidate.workspace.value - future) ** 2).item())
            if not torch.isfinite(torch.tensor(score)):
                raise TensorTransactionContractError("frozen scorer produced a non-finite score")
            scores.append(score)
    return FrozenMSEScoreReceipt(
        spec_fingerprint=spec.fingerprint,
        candidate_execution_fingerprints=(
            candidates[0].execution_fingerprint,
            candidates[1].execution_fingerprint,
        ),
        candidate_output_fingerprints=(
            tensor_content_fingerprint(candidates[0].workspace.value),
            tensor_content_fingerprint(candidates[1].workspace.value),
        ),
        future_fingerprint=future_fingerprint,
        scores=(scores[0], scores[1]),
        _factory_token=_SCORE_FACTORY_TOKEN,
    )


def select_scored_formula_branch(
    harness: K2BranchHarness,
    score: FrozenMSEScoreReceipt,
    *,
    idempotency_key: str,
) -> BranchRunReceipt:
    """Explicitly authorize the lower frozen-score candidate for publication."""

    if not isinstance(harness, K2BranchHarness):
        raise TensorTransactionContractError("harness must be K2BranchHarness")
    if not isinstance(score, FrozenMSEScoreReceipt):
        raise TensorTransactionContractError("score must be FrozenMSEScoreReceipt")
    if score.spec_fingerprint != harness.spec.fingerprint:
        raise TensorTransactionContractError("score does not bind this branch run")
    if set(harness._proposals) != set(harness.spec.branch_ids):
        raise TensorTransactionContractError("both scored proposals must be staged")
    actual = tuple(
        harness._proposals[branch_id].execution_fingerprint
        for branch_id in harness.spec.branch_ids
    )
    if actual != score.candidate_execution_fingerprints:
        raise TensorTransactionContractError("score does not bind the staged executions")
    staged_outputs = tuple(
        tensor_content_fingerprint(harness._proposals[branch_id].proposals[0].value)
        if len(harness._proposals[branch_id].proposals) == 1
        else ""
        for branch_id in harness.spec.branch_ids
    )
    if staged_outputs != score.candidate_output_fingerprints:
        raise TensorTransactionContractError("score does not bind the staged candidate values")
    winner_index = 0 if score.scores[0] <= score.scores[1] else 1
    return harness.select(
        harness.spec.branch_ids[winner_index],
        idempotency_key=idempotency_key,
    )


__all__ = [
    "FormulaBranchExecution",
    "FrozenMSEScoreReceipt",
    "deterministic_formula_rng_fingerprint",
    "execute_fixed_formula_branch",
    "formula_route_fingerprint",
    "score_formula_branches",
    "select_scored_formula_branch",
    "tensor_content_fingerprint",
    "workspace_fingerprint",
]
