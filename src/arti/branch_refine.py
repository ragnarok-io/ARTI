"""K=2 branch-local coordination over volatile tensor transactions."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import re
from dataclasses import dataclass
from enum import Enum
from threading import RLock
from typing import ClassVar, Mapping, Sequence

import torch
from torch import Tensor, nn

from .tensor_binding import (
    ExternalTensorBinding,
    ExternalTensorProposal,
    stage_external_proposal,
)
from .tensor_transaction import (
    CommitReceipt,
    ConflictReceipt,
    RollbackReceipt,
    TensorSnapshot,
    TensorTransaction,
    TensorTransactionContractError,
    VolatileTensorRuntime,
)


_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_COMPONENT_REF = re.compile(r"^arti/[a-z0-9][a-z0-9-]*@[1-9][0-9]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


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


def _require_identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise TensorTransactionContractError(f"{name} must be a canonical identifier")
    return value


def _require_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TensorTransactionContractError(f"{name} must be lowercase SHA-256 hex")
    return value


def _require_component_ref(value: str, name: str) -> str:
    if not isinstance(value, str) or _COMPONENT_REF.fullmatch(value) is None:
        raise TensorTransactionContractError(f"{name} must be a canonical component reference")
    return value


@dataclass(frozen=True)
class BranchBudget:
    """Bounded refine work allowed for one branch."""

    min_steps: int
    max_steps: int
    _runtime_contract_ref: ClassVar[str] = "arti/branch-budget@1"

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (self.min_steps, self.max_steps)
        ):
            raise TensorTransactionContractError(
                "branch steps must be non-negative integers"
            )
        if self.min_steps > self.max_steps:
            raise TensorTransactionContractError("branch min_steps cannot exceed max_steps")


@dataclass(frozen=True)
class BranchBatchSpec:
    """Frozen provenance shared by exactly two branch-local runs."""

    run_id: str
    branch_ids: tuple[str, str]
    parent_store_instance_id: str
    parent_world_id: str
    parent_root_id: str
    parent_epoch: int
    parent_root_fingerprint: str
    executor_ref: str
    program_fingerprint: str
    route_fingerprint: str
    input_fingerprint: str
    rng_fingerprint: str
    future_tape_fingerprint: str
    budgets: tuple[BranchBudget, BranchBudget]
    require_matched_work: bool = True
    _runtime_contract_ref: ClassVar[str] = "arti/branch-batch-spec@1"

    def __post_init__(self) -> None:
        _require_identifier(self.run_id, "run_id")
        normalized = tuple(self.branch_ids)
        object.__setattr__(self, "branch_ids", normalized)
        if len(normalized) != 2 or len(set(normalized)) != 2:
            raise TensorTransactionContractError("BranchBatchSpec@1 requires exactly two unique branches")
        for branch_id in normalized:
            _require_identifier(branch_id, "branch_id")
        _require_identifier(self.parent_store_instance_id, "parent_store_instance_id")
        _require_identifier(self.parent_world_id, "parent_world_id")
        _require_sha256(self.parent_root_id, "parent_root_id")
        _require_sha256(self.parent_root_fingerprint, "parent_root_fingerprint")
        _require_component_ref(self.executor_ref, "executor_ref")
        for value, name in (
            (self.program_fingerprint, "program_fingerprint"),
            (self.route_fingerprint, "route_fingerprint"),
            (self.input_fingerprint, "input_fingerprint"),
            (self.rng_fingerprint, "rng_fingerprint"),
            (self.future_tape_fingerprint, "future_tape_fingerprint"),
        ):
            _require_sha256(value, name)
        if isinstance(self.parent_epoch, bool) or not isinstance(self.parent_epoch, int) or self.parent_epoch < 0:
            raise TensorTransactionContractError("parent_epoch must be non-negative")
        budgets = tuple(self.budgets)
        object.__setattr__(self, "budgets", budgets)
        if len(budgets) != 2 or any(not isinstance(item, BranchBudget) for item in budgets):
            raise TensorTransactionContractError("budgets must contain two BranchBudget values")
        if not isinstance(self.require_matched_work, bool):
            raise TensorTransactionContractError("require_matched_work must be boolean")

    @classmethod
    def from_snapshot(
        cls,
        snapshot: TensorSnapshot,
        *,
        run_id: str,
        branch_ids: tuple[str, str],
        executor_ref: str,
        program_fingerprint: str,
        route_fingerprint: str,
        input_fingerprint: str,
        rng_fingerprint: str,
        future_tape_fingerprint: str,
        budgets: tuple[BranchBudget, BranchBudget],
        require_matched_work: bool = True,
    ) -> "BranchBatchSpec":
        return cls(
            run_id=run_id,
            branch_ids=branch_ids,
            parent_store_instance_id=snapshot.store_instance_id,
            parent_world_id=snapshot.world_id,
            parent_root_id=snapshot.root_id,
            parent_epoch=snapshot.epoch,
            parent_root_fingerprint=snapshot.root_fingerprint,
            executor_ref=executor_ref,
            program_fingerprint=program_fingerprint,
            route_fingerprint=route_fingerprint,
            input_fingerprint=input_fingerprint,
            rng_fingerprint=rng_fingerprint,
            future_tape_fingerprint=future_tape_fingerprint,
            budgets=budgets,
            require_matched_work=require_matched_work,
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "ref": self._runtime_contract_ref,
                "run_id": self.run_id,
                "branch_ids": list(self.branch_ids),
                "parent_store_instance_id": self.parent_store_instance_id,
                "parent_world_id": self.parent_world_id,
                "parent_root_id": self.parent_root_id,
                "parent_epoch": self.parent_epoch,
                "parent_root_fingerprint": self.parent_root_fingerprint,
                "executor_ref": self.executor_ref,
                "program_fingerprint": self.program_fingerprint,
                "route_fingerprint": self.route_fingerprint,
                "input_fingerprint": self.input_fingerprint,
                "rng_fingerprint": self.rng_fingerprint,
                "future_tape_fingerprint": self.future_tape_fingerprint,
                "budgets": [item.__dict__ for item in self.budgets],
                "require_matched_work": self.require_matched_work,
            }
        )


@dataclass(frozen=True)
class BranchBatchSpecV2:
    """Frozen arbitrary-K authority contract for one branch decision."""

    run_id: str
    branch_ids: tuple[str, ...]
    parent_store_instance_id: str
    parent_world_id: str
    parent_root_id: str
    parent_epoch: int
    parent_root_fingerprint: str
    executor_ref: str
    candidate_manifest_fingerprint: str
    program_fingerprint: str
    route_fingerprint: str
    input_fingerprint: str
    rng_fingerprint: str
    future_tape_fingerprint: str
    scorer_ref: str
    scorer_config_fingerprint: str
    tie_policy: str
    allowed_write_keys: tuple[str, ...]
    budgets: tuple[BranchBudget, ...]
    require_matched_work: bool = False
    _runtime_contract_ref: ClassVar[str] = "arti/branch-batch-spec@2"

    def __post_init__(self) -> None:
        _require_identifier(self.run_id, "run_id")
        branches = tuple(self.branch_ids)
        object.__setattr__(self, "branch_ids", branches)
        if not branches or len(set(branches)) != len(branches):
            raise TensorTransactionContractError("BranchBatchSpec@2 requires unique K>=1 branches")
        for branch_id in branches:
            _require_identifier(branch_id, "branch_id")
        _require_identifier(self.parent_store_instance_id, "parent_store_instance_id")
        _require_identifier(self.parent_world_id, "parent_world_id")
        _require_sha256(self.parent_root_id, "parent_root_id")
        _require_sha256(self.parent_root_fingerprint, "parent_root_fingerprint")
        _require_component_ref(self.executor_ref, "executor_ref")
        _require_component_ref(self.scorer_ref, "scorer_ref")
        for value, name in (
            (self.candidate_manifest_fingerprint, "candidate_manifest_fingerprint"),
            (self.program_fingerprint, "program_fingerprint"),
            (self.route_fingerprint, "route_fingerprint"),
            (self.input_fingerprint, "input_fingerprint"),
            (self.rng_fingerprint, "rng_fingerprint"),
            (self.future_tape_fingerprint, "future_tape_fingerprint"),
            (self.scorer_config_fingerprint, "scorer_config_fingerprint"),
        ):
            _require_sha256(value, name)
        if isinstance(self.parent_epoch, bool) or not isinstance(self.parent_epoch, int) or self.parent_epoch < 0:
            raise TensorTransactionContractError("parent_epoch must be non-negative")
        _require_identifier(self.tie_policy, "tie_policy")
        if self.tie_policy != "lowest-score-then-branch-order":
            raise TensorTransactionContractError("unsupported BranchBatchSpec@2 tie policy")
        write_keys = tuple(self.allowed_write_keys)
        object.__setattr__(self, "allowed_write_keys", write_keys)
        if len(write_keys) != 1:
            raise TensorTransactionContractError(
                "BranchBatchSpec@2 currently requires one exact target page"
            )
        for key in write_keys:
            _require_identifier(key, "allowed_write_key")
        budgets = tuple(self.budgets)
        object.__setattr__(self, "budgets", budgets)
        if len(budgets) != len(branches) or any(not isinstance(item, BranchBudget) for item in budgets):
            raise TensorTransactionContractError("budgets must contain one BranchBudget per branch")
        if not isinstance(self.require_matched_work, bool):
            raise TensorTransactionContractError("require_matched_work must be boolean")

    @classmethod
    def from_snapshot(
        cls,
        snapshot: TensorSnapshot,
        *,
        run_id: str,
        branch_ids: Sequence[str],
        executor_ref: str,
        candidate_manifest_fingerprint: str,
        program_fingerprint: str,
        route_fingerprint: str,
        input_fingerprint: str,
        rng_fingerprint: str,
        future_tape_fingerprint: str,
        scorer_ref: str,
        scorer_config_fingerprint: str,
        allowed_write_keys: Sequence[str],
        budgets: Sequence[BranchBudget],
        tie_policy: str = "lowest-score-then-branch-order",
        require_matched_work: bool = False,
    ) -> "BranchBatchSpecV2":
        return cls(
            run_id=run_id,
            branch_ids=tuple(branch_ids),
            parent_store_instance_id=snapshot.store_instance_id,
            parent_world_id=snapshot.world_id,
            parent_root_id=snapshot.root_id,
            parent_epoch=snapshot.epoch,
            parent_root_fingerprint=snapshot.root_fingerprint,
            executor_ref=executor_ref,
            candidate_manifest_fingerprint=candidate_manifest_fingerprint,
            program_fingerprint=program_fingerprint,
            route_fingerprint=route_fingerprint,
            input_fingerprint=input_fingerprint,
            rng_fingerprint=rng_fingerprint,
            future_tape_fingerprint=future_tape_fingerprint,
            scorer_ref=scorer_ref,
            scorer_config_fingerprint=scorer_config_fingerprint,
            tie_policy=tie_policy,
            allowed_write_keys=tuple(allowed_write_keys),
            budgets=tuple(budgets),
            require_matched_work=require_matched_work,
        )

    @classmethod
    def from_batched_refine(
        cls,
        snapshot: TensorSnapshot,
        executor: "BatchedRefineExecutor",
        *,
        run_id: str,
        branch_ids: Sequence[str],
        input_fingerprint: str,
        future_tape_fingerprint: str,
        scorer_ref: str,
        scorer_config_fingerprint: str,
        allowed_write_keys: Sequence[str],
        budgets: Sequence[BranchBudget],
        tie_policy: str = "lowest-score-then-branch-order",
        require_matched_work: bool = False,
    ) -> "BranchBatchSpecV2":
        """Create authority provenance from one factory-owned executor closure."""

        if not isinstance(executor, BatchedRefineExecutor):
            raise TypeError("executor must be BatchedRefineExecutor")
        return cls.from_snapshot(
            snapshot,
            run_id=run_id,
            branch_ids=branch_ids,
            executor_ref=executor._component_reference,
            candidate_manifest_fingerprint=executor.candidate_manifest_fingerprint,
            program_fingerprint=executor.plan_config_fingerprint,
            route_fingerprint=executor.route_fingerprint,
            input_fingerprint=input_fingerprint,
            rng_fingerprint=executor.execution_context.fingerprint,
            future_tape_fingerprint=future_tape_fingerprint,
            scorer_ref=scorer_ref,
            scorer_config_fingerprint=scorer_config_fingerprint,
            allowed_write_keys=allowed_write_keys,
            budgets=budgets,
            tie_policy=tie_policy,
            require_matched_work=require_matched_work,
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "ref": self._runtime_contract_ref,
                "run_id": self.run_id,
                "branch_ids": list(self.branch_ids),
                "parent": {
                    "store": self.parent_store_instance_id,
                    "world": self.parent_world_id,
                    "root": self.parent_root_id,
                    "epoch": self.parent_epoch,
                    "fingerprint": self.parent_root_fingerprint,
                },
                "executor_ref": self.executor_ref,
                "candidate_manifest_fingerprint": self.candidate_manifest_fingerprint,
                "program_fingerprint": self.program_fingerprint,
                "route_fingerprint": self.route_fingerprint,
                "input_fingerprint": self.input_fingerprint,
                "rng_fingerprint": self.rng_fingerprint,
                "future_tape_fingerprint": self.future_tape_fingerprint,
                "scorer_ref": self.scorer_ref,
                "scorer_config_fingerprint": self.scorer_config_fingerprint,
                "tie_policy": self.tie_policy,
                "allowed_write_keys": list(self.allowed_write_keys),
                "budgets": [item.__dict__ for item in self.budgets],
                "require_matched_work": self.require_matched_work,
            }
        )


@dataclass(frozen=True)
class BranchWorkReceipt:
    """Logical work and explicitly labelled cost estimates for one branch."""

    actual_steps: int
    formula_cells: int
    route_applications: int
    fire_count: int
    commit_count: int
    operation_count: int
    stop_reason: str
    residual_fingerprint: str
    finite: bool
    logical_steps_attempted: int | None = None
    logical_steps_committed: int | None = None
    outer_kernel_steps: int | None = None
    resident_operation_steps: int = 0
    measurement_kind: str = "declared-estimate"
    logical_step_unit: str = "branch-step"
    _runtime_contract_ref: ClassVar[str] = "arti/branch-work-receipt@2"

    def __post_init__(self) -> None:
        for value, name in (
            (self.actual_steps, "actual_steps"),
            (self.formula_cells, "formula_cells"),
            (self.route_applications, "route_applications"),
            (self.fire_count, "fire_count"),
            (self.commit_count, "commit_count"),
            (self.operation_count, "operation_count"),
            (self.resident_operation_steps, "resident_operation_steps"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TensorTransactionContractError(f"{name} must be non-negative")
        for name in (
            "logical_steps_attempted",
            "logical_steps_committed",
            "outer_kernel_steps",
        ):
            value = getattr(self, name)
            if value is None:
                value = self.actual_steps
                object.__setattr__(self, name, value)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise TensorTransactionContractError(f"{name} must be non-negative")
        assert self.logical_steps_attempted is not None
        assert self.logical_steps_committed is not None
        if self.logical_steps_committed > self.logical_steps_attempted:
            raise TensorTransactionContractError(
                "logical committed work cannot exceed attempted work"
            )
        _require_identifier(self.stop_reason, "stop_reason")
        _require_identifier(self.measurement_kind, "measurement_kind")
        _require_identifier(self.logical_step_unit, "logical_step_unit")
        _require_sha256(self.residual_fingerprint, "residual_fingerprint")
        if not isinstance(self.finite, bool):
            raise TensorTransactionContractError("finite must be boolean")

    @property
    def matched_key(self) -> tuple[object, ...]:
        return (
            self.actual_steps,
            self.formula_cells,
            self.route_applications,
            self.fire_count,
            self.commit_count,
            self.operation_count,
            self.logical_steps_attempted,
            self.logical_steps_committed,
            self.outer_kernel_steps,
            self.resident_operation_steps,
            self.measurement_kind,
            self.logical_step_unit,
        )


@dataclass(frozen=True)
class OverlayProposal:
    """Immutable branch-local set of complete page proposals."""

    branch_id: str
    spec_fingerprint: str
    proposals: tuple[ExternalTensorProposal, ...]
    work: BranchWorkReceipt
    step_input_fingerprints: tuple[str, ...]
    step_output_fingerprints: tuple[str, ...]
    execution_fingerprint: str
    _runtime_contract_ref: ClassVar[str] = "arti/overlay-proposal@1"

    def __post_init__(self) -> None:
        _require_identifier(self.branch_id, "branch_id")
        _require_sha256(self.spec_fingerprint, "spec_fingerprint")
        normalized = tuple(self.proposals)
        object.__setattr__(self, "proposals", normalized)
        if not normalized or any(not isinstance(item, ExternalTensorProposal) for item in normalized):
            raise TensorTransactionContractError("proposals must contain ExternalTensorProposal values")
        keys = tuple(item.binding.tensor_ref.key for item in normalized)
        if len(set(keys)) != len(keys):
            raise TensorTransactionContractError("one branch cannot propose the same page twice")
        if not isinstance(self.work, BranchWorkReceipt):
            raise TensorTransactionContractError("work must be BranchWorkReceipt")
        step_inputs = tuple(self.step_input_fingerprints)
        object.__setattr__(self, "step_input_fingerprints", step_inputs)
        if len(step_inputs) != self.work.actual_steps:
            raise TensorTransactionContractError(
                "step_input_fingerprints must match actual_steps"
            )
        for value in step_inputs:
            _require_sha256(value, "step_input_fingerprint")
        step_outputs = tuple(self.step_output_fingerprints)
        object.__setattr__(self, "step_output_fingerprints", step_outputs)
        if len(step_outputs) != self.work.actual_steps:
            raise TensorTransactionContractError(
                "step_output_fingerprints must match actual_steps"
            )
        for value in step_outputs:
            _require_sha256(value, "step_output_fingerprint")
        if step_inputs[1:] != step_outputs[:-1]:
            raise TensorTransactionContractError(
                "each refine step must read the preceding step output"
            )
        _require_sha256(self.execution_fingerprint, "execution_fingerprint")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "ref": self._runtime_contract_ref,
                "branch_id": self.branch_id,
                "spec_fingerprint": self.spec_fingerprint,
                "proposal_provenance": [item.provenance_fingerprint for item in self.proposals],
                "work": {
                    "actual_steps": self.work.actual_steps,
                    "formula_cells": self.work.formula_cells,
                    "route_applications": self.work.route_applications,
                    "fire_count": self.work.fire_count,
                    "commit_count": self.work.commit_count,
                    "operation_count": self.work.operation_count,
                    "logical_steps_attempted": self.work.logical_steps_attempted,
                    "logical_steps_committed": self.work.logical_steps_committed,
                    "outer_kernel_steps": self.work.outer_kernel_steps,
                    "resident_operation_steps": self.work.resident_operation_steps,
                    "measurement_kind": self.work.measurement_kind,
                    "logical_step_unit": self.work.logical_step_unit,
                    "stop_reason": self.work.stop_reason,
                    "residual_fingerprint": self.work.residual_fingerprint,
                    "finite": self.work.finite,
                },
                "step_input_fingerprints": list(self.step_input_fingerprints),
                "step_output_fingerprints": list(self.step_output_fingerprints),
                "execution_fingerprint": self.execution_fingerprint,
            }
        )


_BATCHED_OVERLAY_FACTORY_TOKEN = object()
_BATCH_SCORE_FACTORY_TOKEN = object()


@dataclass(frozen=True, init=False)
class BatchedRefineOverlayProposal:
    """Factory-owned CPU proposal for one real Batched Refine trajectory."""

    branch_id: str
    spec_fingerprint: str
    candidate_fingerprint: str
    trajectory_fingerprint: str
    proposal: ExternalTensorProposal
    delta_fingerprint: str
    work: BranchWorkReceipt
    _delta: Tensor
    _runtime_contract_ref: ClassVar[str] = "arti/batched-refine-overlay@1"

    def __init__(
        self,
        *,
        branch_id: str,
        spec_fingerprint: str,
        candidate_fingerprint: str,
        trajectory_fingerprint: str,
        proposal: ExternalTensorProposal,
        delta: Tensor,
        work: BranchWorkReceipt,
        _factory_token: object,
    ) -> None:
        if _factory_token is not _BATCHED_OVERLAY_FACTORY_TOKEN:
            raise TensorTransactionContractError(
                "BatchedRefineOverlayProposal must come from stage_batched_refine_result"
            )
        _require_identifier(branch_id, "branch_id")
        _require_sha256(spec_fingerprint, "spec_fingerprint")
        _require_sha256(candidate_fingerprint, "candidate_fingerprint")
        _require_sha256(trajectory_fingerprint, "trajectory_fingerprint")
        if not isinstance(proposal, ExternalTensorProposal):
            raise TensorTransactionContractError("proposal must be ExternalTensorProposal")
        if (
            not isinstance(delta, Tensor)
            or delta.device.type != "cpu"
            or delta.requires_grad
            or not delta.is_contiguous()
            or tuple(delta.shape) != proposal.binding.tensor_ref.shape
            or str(delta.dtype) != proposal.binding.tensor_ref.dtype
        ):
            raise TensorTransactionContractError(
                "branch delta must be detached contiguous CPU storage matching the target page"
            )
        if not isinstance(work, BranchWorkReceipt):
            raise TensorTransactionContractError("work must be BranchWorkReceipt")
        owned_delta = delta.detach().clone(memory_format=torch.contiguous_format)
        delta_fingerprint = _cpu_tensor_fingerprint(owned_delta)
        object.__setattr__(self, "branch_id", branch_id)
        object.__setattr__(self, "spec_fingerprint", spec_fingerprint)
        object.__setattr__(self, "candidate_fingerprint", candidate_fingerprint)
        object.__setattr__(self, "trajectory_fingerprint", trajectory_fingerprint)
        object.__setattr__(self, "proposal", proposal)
        object.__setattr__(self, "_delta", owned_delta)
        object.__setattr__(self, "delta_fingerprint", delta_fingerprint)
        object.__setattr__(self, "work", work)

    @property
    def delta(self) -> Tensor:
        return self._delta.clone()

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "ref": self._runtime_contract_ref,
                "branch_id": self.branch_id,
                "spec_fingerprint": self.spec_fingerprint,
                "candidate_fingerprint": self.candidate_fingerprint,
                "trajectory_fingerprint": self.trajectory_fingerprint,
                "proposal_fingerprint": self.proposal.provenance_fingerprint,
                "delta_fingerprint": self.delta_fingerprint,
                "work": self.work.__dict__,
            }
        )


@dataclass(frozen=True, init=False)
class BranchScoreBatchReceipt:
    """Frozen arbitrary-K score evidence without publication authority."""

    spec_fingerprint: str
    branch_ids: tuple[str, ...]
    overlay_fingerprints: tuple[str, ...]
    candidate_fingerprints: tuple[str, ...]
    future_fingerprint: str
    scorer_ref: str
    scorer_config_fingerprint: str
    tie_policy: str
    scores: tuple[float, ...]
    receipt_fingerprint: str
    _runtime_contract_ref: ClassVar[str] = "arti/branch-score-batch@1"

    def __init__(
        self,
        *,
        spec_fingerprint: str,
        branch_ids: tuple[str, ...],
        overlay_fingerprints: tuple[str, ...],
        candidate_fingerprints: tuple[str, ...],
        future_fingerprint: str,
        scorer_ref: str,
        scorer_config_fingerprint: str,
        tie_policy: str,
        scores: tuple[float, ...],
        _factory_token: object,
    ) -> None:
        if _factory_token is not _BATCH_SCORE_FACTORY_TOKEN:
            raise TensorTransactionContractError(
                "BranchScoreBatchReceipt must come from score_batched_refine_proposals"
            )
        count = len(branch_ids)
        if count < 1 or any(len(values) != count for values in (
            overlay_fingerprints,
            candidate_fingerprints,
            scores,
        )):
            raise TensorTransactionContractError("score receipt fields must have one entry per branch")
        for value in (spec_fingerprint, future_fingerprint, scorer_config_fingerprint):
            _require_sha256(value, "score fingerprint")
        for value in (*overlay_fingerprints, *candidate_fingerprints):
            _require_sha256(value, "score candidate fingerprint")
        _require_component_ref(scorer_ref, "scorer_ref")
        _require_identifier(tie_policy, "tie_policy")
        if any(not isinstance(value, float) or not torch.isfinite(torch.tensor(value)) for value in scores):
            raise TensorTransactionContractError("scores must be finite floats")
        content = {
            "ref": self._runtime_contract_ref,
            "spec_fingerprint": spec_fingerprint,
            "branch_ids": list(branch_ids),
            "overlay_fingerprints": list(overlay_fingerprints),
            "candidate_fingerprints": list(candidate_fingerprints),
            "future_fingerprint": future_fingerprint,
            "scorer_ref": scorer_ref,
            "scorer_config_fingerprint": scorer_config_fingerprint,
            "tie_policy": tie_policy,
            "scores": list(scores),
        }
        for name, value in content.items():
            if name != "ref":
                object.__setattr__(self, name, tuple(value) if isinstance(value, list) else value)
        object.__setattr__(self, "receipt_fingerprint", _fingerprint(content))


class BranchRunStatus(str, Enum):
    COMMITTED = "committed"
    DISCARDED = "discarded"
    CONFLICTED = "conflicted"


_BRANCH_RUN_RECEIPT_FACTORY_TOKEN = object()


def _work_receipt_payload(value: BranchWorkReceipt) -> dict[str, object]:
    return {
        "ref": value._runtime_contract_ref,
        "actual_steps": value.actual_steps,
        "formula_cells": value.formula_cells,
        "route_applications": value.route_applications,
        "fire_count": value.fire_count,
        "commit_count": value.commit_count,
        "operation_count": value.operation_count,
        "logical_steps_attempted": value.logical_steps_attempted,
        "logical_steps_committed": value.logical_steps_committed,
        "outer_kernel_steps": value.outer_kernel_steps,
        "resident_operation_steps": value.resident_operation_steps,
        "measurement_kind": value.measurement_kind,
        "logical_step_unit": value.logical_step_unit,
        "stop_reason": value.stop_reason,
        "residual_fingerprint": value.residual_fingerprint,
        "finite": value.finite,
    }


def _commit_receipt_payload(
    value: CommitReceipt | ConflictReceipt | None,
) -> dict[str, object] | None:
    if value is None:
        return None
    common: dict[str, object] = {
        "ref": value._runtime_contract_ref,
        "store_instance_id": value.store_instance_id,
        "world_id": value.world_id,
        "transaction_id": value.transaction_id,
        "branch_id": value.branch_id,
        "idempotency_key": value.idempotency_key,
        "request_fingerprint": value.request_fingerprint,
        "base_root_id": value.base_root_id,
        "base_epoch": value.base_epoch,
    }
    if isinstance(value, CommitReceipt):
        return common | {
            "new_root_id": value.new_root_id,
            "new_epoch": value.new_epoch,
            "read_set_digest": value.read_set_digest,
            "write_set_digest": value.write_set_digest,
            "provenance_head": value.provenance_head,
            "receipt_fingerprint": value.receipt_fingerprint,
        }
    return common | {
        "reason": value.reason.value,
        "current_root_id": value.current_root_id,
        "current_epoch": value.current_epoch,
        "page_conflicts": [
            {
                "key": item.key,
                "expected_version": item.expected_version,
                "current_version": item.current_version,
                "expected_content_sha256": item.expected_content_sha256,
                "current_content_sha256": item.current_content_sha256,
            }
            for item in value.page_conflicts
        ],
    }


@dataclass(frozen=True, init=False)
class BranchRunReceipt:
    """Aggregate receipt for one explicit winner or all-discard decision."""

    spec_fingerprint: str
    status: BranchRunStatus
    winner_branch_id: str | None
    decision_idempotency_key: str
    decision_request_fingerprint: str
    proposal_fingerprints: tuple[tuple[str, str], ...]
    work_receipts: tuple[tuple[str, BranchWorkReceipt], ...]
    commit_receipt: CommitReceipt | ConflictReceipt | None
    rollback_receipts: tuple[RollbackReceipt, ...]
    receipt_fingerprint: str
    decision_kind: str = "winner"
    _runtime_contract_ref: ClassVar[str] = "arti/branch-run-receipt@2"

    def __init__(
        self,
        *,
        spec_fingerprint: str,
        status: BranchRunStatus,
        winner_branch_id: str | None,
        decision_idempotency_key: str,
        decision_request_fingerprint: str,
        proposal_fingerprints: Sequence[tuple[str, str]],
        work_receipts: Sequence[tuple[str, BranchWorkReceipt]],
        commit_receipt: CommitReceipt | ConflictReceipt | None,
        rollback_receipts: Sequence[RollbackReceipt],
        decision_kind: str,
        _factory_token: object,
    ) -> None:
        if _factory_token is not _BRANCH_RUN_RECEIPT_FACTORY_TOKEN:
            raise TensorTransactionContractError(
                "BranchRunReceipt values are authority-factory owned"
            )
        _require_sha256(spec_fingerprint, "spec_fingerprint")
        if not isinstance(status, BranchRunStatus):
            raise TensorTransactionContractError("status must be BranchRunStatus")
        _require_identifier(decision_idempotency_key, "decision_idempotency_key")
        _require_sha256(
            decision_request_fingerprint, "decision_request_fingerprint"
        )
        if decision_kind not in {"winner", "mixture", "discard"}:
            raise TensorTransactionContractError("unsupported branch decision_kind")
        proposals = tuple(proposal_fingerprints)
        works = tuple(work_receipts)
        rollbacks = tuple(rollback_receipts)
        proposal_names: set[str] = set()
        for branch_id, fingerprint in proposals:
            _require_identifier(branch_id, "proposal branch_id")
            _require_sha256(fingerprint, "proposal fingerprint")
            if branch_id in proposal_names:
                raise TensorTransactionContractError("proposal branch_ids must be unique")
            proposal_names.add(branch_id)
        work_names: set[str] = set()
        for branch_id, work in works:
            _require_identifier(branch_id, "work branch_id")
            if not isinstance(work, BranchWorkReceipt):
                raise TensorTransactionContractError("work receipt type is invalid")
            if branch_id in work_names:
                raise TensorTransactionContractError("work branch_ids must be unique")
            work_names.add(branch_id)
        if proposal_names != work_names:
            raise TensorTransactionContractError(
                "proposal and work receipt branch sets must match"
            )
        rollback_names: set[str] = set()
        for rollback in rollbacks:
            if not isinstance(rollback, RollbackReceipt):
                raise TensorTransactionContractError("rollback receipt type is invalid")
            if rollback.branch_id in rollback_names:
                raise TensorTransactionContractError("rollback branch_ids must be unique")
            rollback_names.add(rollback.branch_id)
        if status is BranchRunStatus.DISCARDED:
            if winner_branch_id is not None or decision_kind != "discard":
                raise TensorTransactionContractError(
                    "discarded branch receipt requires decision_kind='discard' and no winner"
                )
            if commit_receipt is not None:
                raise TensorTransactionContractError(
                    "discarded branch receipt cannot contain a commit receipt"
                )
        elif decision_kind == "discard":
            raise TensorTransactionContractError(
                "decision_kind='discard' requires discarded branch status"
            )
        if decision_kind == "winner" and winner_branch_id is None:
            raise TensorTransactionContractError(
                "winner branch receipt requires winner_branch_id"
            )
        if decision_kind == "mixture" and winner_branch_id is not None:
            raise TensorTransactionContractError(
                "mixture branch receipt cannot name a winner"
            )
        if status is BranchRunStatus.COMMITTED and not isinstance(
            commit_receipt, CommitReceipt
        ):
            raise TensorTransactionContractError(
                "committed branch receipt requires CommitReceipt"
            )
        if status is BranchRunStatus.CONFLICTED and not isinstance(
            commit_receipt, ConflictReceipt
        ):
            raise TensorTransactionContractError(
                "conflicted branch receipt requires ConflictReceipt"
            )
        if commit_receipt is not None and (
            commit_receipt.idempotency_key != decision_idempotency_key
        ):
            raise TensorTransactionContractError(
                "commit receipt does not bind the branch decision idempotency key"
            )
        content = {
            "ref": self._runtime_contract_ref,
            "spec_fingerprint": spec_fingerprint,
            "status": status.value,
            "winner_branch_id": winner_branch_id,
            "decision_idempotency_key": decision_idempotency_key,
            "decision_request_fingerprint": decision_request_fingerprint,
            "decision_kind": decision_kind,
            "proposal_fingerprints": [list(item) for item in proposals],
            "work_receipts": [
                [branch_id, _work_receipt_payload(work)]
                for branch_id, work in works
            ],
            "commit_receipt": _commit_receipt_payload(commit_receipt),
            "rollback_receipts": [
                {
                    "ref": item._runtime_contract_ref,
                    "transaction_id": item.transaction_id,
                    "branch_id": item.branch_id,
                    "base_root_id": item.base_root_id,
                    "base_epoch": item.base_epoch,
                }
                for item in rollbacks
            ],
        }
        object.__setattr__(self, "spec_fingerprint", spec_fingerprint)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "winner_branch_id", winner_branch_id)
        object.__setattr__(self, "decision_idempotency_key", decision_idempotency_key)
        object.__setattr__(
            self, "decision_request_fingerprint", decision_request_fingerprint
        )
        object.__setattr__(self, "proposal_fingerprints", proposals)
        object.__setattr__(self, "work_receipts", works)
        object.__setattr__(self, "commit_receipt", commit_receipt)
        object.__setattr__(self, "rollback_receipts", rollbacks)
        object.__setattr__(self, "decision_kind", decision_kind)
        object.__setattr__(self, "receipt_fingerprint", _fingerprint(content))


class K2BranchHarness:
    """Host-only K=2 coordinator; it never executes or selects neural work."""

    _runtime_contract_ref: ClassVar[str] = "arti/k2-branch-harness@1"

    def __init__(
        self,
        runtime: VolatileTensorRuntime,
        snapshot: TensorSnapshot,
        spec: BranchBatchSpec,
    ) -> None:
        if not isinstance(runtime, VolatileTensorRuntime):
            raise TensorTransactionContractError("runtime must be VolatileTensorRuntime")
        if not isinstance(spec, BranchBatchSpec):
            raise TensorTransactionContractError("spec must be BranchBatchSpec")
        runtime._resolve_snapshot(snapshot)
        if (
            spec.parent_store_instance_id != snapshot.store_instance_id
            or spec.parent_world_id != snapshot.world_id
            or spec.parent_root_id != snapshot.root_id
            or spec.parent_epoch != snapshot.epoch
            or spec.parent_root_fingerprint != snapshot.root_fingerprint
        ):
            raise TensorTransactionContractError("BranchBatchSpec parent does not match snapshot")
        self._runtime = runtime
        self._snapshot = snapshot
        self.spec = spec
        self._transactions: dict[str, TensorTransaction] = {}
        for branch_id in spec.branch_ids:
            transaction_digest = hashlib.sha256(
                f"{spec.run_id}:{branch_id}".encode("ascii")
            ).hexdigest()[:32]
            self._transactions[branch_id] = runtime.begin(
                snapshot,
                transaction_id=f"branch-{transaction_digest}",
                branch_id=branch_id,
            )
        self._proposals: dict[str, OverlayProposal] = {}
        self._receipt: BranchRunReceipt | None = None

    def propose(self, proposal: OverlayProposal) -> None:
        if self._receipt is not None:
            raise TensorTransactionContractError("branch run is already closed")
        if not isinstance(proposal, OverlayProposal) or proposal.branch_id not in self._transactions:
            raise TensorTransactionContractError("proposal branch is not part of this run")
        if proposal.spec_fingerprint != self.spec.fingerprint:
            raise TensorTransactionContractError("proposal does not bind this branch spec")
        if proposal.branch_id in self._proposals:
            raise TensorTransactionContractError("branch already has a proposal")
        branch_index = self.spec.branch_ids.index(proposal.branch_id)
        budget = self.spec.budgets[branch_index]
        if not budget.min_steps <= proposal.work.actual_steps <= budget.max_steps:
            raise TensorTransactionContractError("branch actual_steps violate its budget")
        if not proposal.work.finite:
            raise TensorTransactionContractError("branch work must be finite")
        if proposal.step_input_fingerprints[0] != self.spec.input_fingerprint:
            raise TensorTransactionContractError("first refine step does not bind the shared input")
        transaction = self._transactions[proposal.branch_id]
        try:
            for item in proposal.proposals:
                binding = item.binding
                if (
                    binding.root_id != self.spec.parent_root_id
                    or binding.root_epoch != self.spec.parent_epoch
                    or binding.root_fingerprint != self.spec.parent_root_fingerprint
                ):
                    raise TensorTransactionContractError(
                        "branch proposal is not rooted at the parent snapshot"
                    )
                stage_external_proposal(transaction, item)
        except Exception:
            self.abort(idempotency_key=f"abort-{self.spec.run_id}")
            raise
        self._proposals[proposal.branch_id] = proposal

    def _validate_ready(self) -> None:
        if set(self._proposals) != set(self.spec.branch_ids):
            raise TensorTransactionContractError("both branches require proposals before selection")
        if self.spec.require_matched_work:
            keys = {proposal.work.matched_key for proposal in self._proposals.values()}
            if len(keys) != 1:
                raise TensorTransactionContractError("branch work is not matched")

    def select(
        self,
        winner_branch_id: str | None,
        *,
        idempotency_key: str,
    ) -> BranchRunReceipt:
        decision_key = _require_identifier(idempotency_key, "idempotency_key")
        if self._receipt is not None:
            if (
                self._receipt.winner_branch_id == winner_branch_id
                and self._receipt.decision_idempotency_key == decision_key
            ):
                return self._receipt
            raise TensorTransactionContractError("branch run already closed with another decision")
        self._validate_ready()
        if winner_branch_id is not None and winner_branch_id not in self._transactions:
            raise TensorTransactionContractError("winner must be one branch or None")
        commit: CommitReceipt | ConflictReceipt | None = None
        rollbacks: list[RollbackReceipt] = []
        if winner_branch_id is None:
            for branch_id in self.spec.branch_ids:
                rollbacks.append(self._transactions[branch_id].rollback())
            status = BranchRunStatus.DISCARDED
        else:
            commit = self._transactions[winner_branch_id].commit(
                idempotency_key=decision_key
            )
            for branch_id in self.spec.branch_ids:
                if branch_id != winner_branch_id:
                    rollbacks.append(self._transactions[branch_id].rollback())
            status = (
                BranchRunStatus.COMMITTED
                if isinstance(commit, CommitReceipt)
                else BranchRunStatus.CONFLICTED
            )
        proposal_fingerprints = tuple(
            (branch_id, self._proposals[branch_id].fingerprint)
            for branch_id in self.spec.branch_ids
        )
        work_receipts = tuple(
            (branch_id, self._proposals[branch_id].work)
            for branch_id in self.spec.branch_ids
        )
        decision_request_fingerprint = _fingerprint(
            {
                "spec": self.spec.fingerprint,
                "winner": winner_branch_id,
                "decision_key": decision_key,
                "proposals": proposal_fingerprints,
                "kind": "discard" if winner_branch_id is None else "winner",
            }
        )
        self._receipt = BranchRunReceipt(
            spec_fingerprint=self.spec.fingerprint,
            status=status,
            winner_branch_id=winner_branch_id,
            decision_idempotency_key=decision_key,
            decision_request_fingerprint=decision_request_fingerprint,
            proposal_fingerprints=proposal_fingerprints,
            work_receipts=work_receipts,
            commit_receipt=commit,
            rollback_receipts=tuple(rollbacks),
            decision_kind="discard" if winner_branch_id is None else "winner",
            _factory_token=_BRANCH_RUN_RECEIPT_FACTORY_TOKEN,
        )
        return self._receipt


    def abort(self, *, idempotency_key: str) -> BranchRunReceipt:
        """Discard every still-open legacy K=2 branch."""

        decision_key = _require_identifier(idempotency_key, "idempotency_key")
        if self._receipt is not None:
            if (
                self._receipt.status is BranchRunStatus.DISCARDED
                and self._receipt.decision_idempotency_key == decision_key
            ):
                return self._receipt
            raise TensorTransactionContractError(
                "branch run already closed with another decision"
            )
        rollbacks = tuple(
            self._transactions[branch_id].rollback()
            for branch_id in self.spec.branch_ids
        )
        proposal_fingerprints = tuple(
            (branch_id, self._proposals[branch_id].fingerprint)
            for branch_id in self.spec.branch_ids
            if branch_id in self._proposals
        )
        work_receipts = tuple(
            (branch_id, self._proposals[branch_id].work)
            for branch_id in self.spec.branch_ids
            if branch_id in self._proposals
        )
        decision_request_fingerprint = _fingerprint(
            {
                "spec": self.spec.fingerprint,
                "decision_key": decision_key,
                "proposals": proposal_fingerprints,
                "kind": "abort",
            }
        )
        self._receipt = BranchRunReceipt(
            spec_fingerprint=self.spec.fingerprint,
            status=BranchRunStatus.DISCARDED,
            winner_branch_id=None,
            decision_idempotency_key=decision_key,
            decision_request_fingerprint=decision_request_fingerprint,
            proposal_fingerprints=proposal_fingerprints,
            work_receipts=work_receipts,
            commit_receipt=None,
            rollback_receipts=rollbacks,
            decision_kind="discard",
            _factory_token=_BRANCH_RUN_RECEIPT_FACTORY_TOKEN,
        )
        return self._receipt


def _cpu_tensor_fingerprint(value: Tensor) -> str:
    if value.device.type != "cpu" or not value.is_contiguous():
        raise TensorTransactionContractError("fingerprinted tensor must be contiguous CPU storage")
    normalized = value.detach().clone(memory_format=torch.contiguous_format).reshape(-1)
    raw = normalized.view(torch.uint8).numpy().tobytes()
    return _fingerprint(
        {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "bytes_sha256": hashlib.sha256(raw).hexdigest(),
        }
    )


_BATCHED_REFINE_EXECUTOR_FACTORY_TOKEN = object()
_EXECUTION_CONTEXT_RECEIPT_FACTORY_TOKEN = object()

FROZEN_MSE_SCORER_REF = "arti/frozen-mse-scorer@1"


@dataclass(frozen=True, init=False)
class ExecutionContextReceipt:
    """Factory-owned identity of the randomness used by one branch run."""

    schema_version: int
    mode: str
    algorithm: str
    candidate_manifest_fingerprint: str
    plan_config_fingerprint: str
    route_fingerprint: str
    branch_policy_fingerprint: str | None
    execution_rng_fingerprint: str | None
    execution_rng_stream_key: str | None
    consumed_domains: tuple[str, ...]
    _runtime_contract_ref: ClassVar[str] = "arti/execution-context-receipt@3"

    def __init__(
        self,
        *,
        schema_version: int,
        mode: str,
        algorithm: str,
        candidate_manifest_fingerprint: str,
        plan_config_fingerprint: str,
        route_fingerprint: str,
        branch_policy_fingerprint: str | None,
        execution_rng_fingerprint: str | None,
        execution_rng_stream_key: str | None,
        consumed_domains: tuple[str, ...],
        _factory_token: object,
    ) -> None:
        if _factory_token is not _EXECUTION_CONTEXT_RECEIPT_FACTORY_TOKEN:
            raise TensorTransactionContractError(
                "ExecutionContextReceipt must come from a completed Batched Refine run"
            )
        if schema_version != 3:
            raise TensorTransactionContractError(
                "unsupported ExecutionContextReceipt schema"
            )
        if mode not in {"deterministic", "keyed"}:
            raise TensorTransactionContractError("unsupported execution context mode")
        expected_algorithm = (
            "none" if mode == "deterministic" else "sha256-seeded-torch-generator@2"
        )
        if algorithm != expected_algorithm:
            raise TensorTransactionContractError(
                "execution context algorithm does not match its mode"
            )
        for value, name in (
            (candidate_manifest_fingerprint, "candidate_manifest_fingerprint"),
            (plan_config_fingerprint, "plan_config_fingerprint"),
            (route_fingerprint, "route_fingerprint"),
        ):
            _require_sha256(value, name)
        if branch_policy_fingerprint is not None:
            _require_sha256(branch_policy_fingerprint, "branch_policy_fingerprint")
        if execution_rng_fingerprint is not None:
            _require_sha256(execution_rng_fingerprint, "execution_rng_fingerprint")
        if (mode == "keyed") != (execution_rng_fingerprint is not None):
            raise TensorTransactionContractError(
                "keyed execution requires exactly one RNG plan fingerprint"
            )
        if (mode == "keyed") != (execution_rng_stream_key is not None):
            raise TensorTransactionContractError(
                "keyed execution requires exactly one RNG stream key"
            )
        if execution_rng_stream_key is not None:
            _require_identifier(execution_rng_stream_key, "execution_rng_stream_key")
        if (
            not isinstance(consumed_domains, tuple)
            or tuple(sorted(set(consumed_domains))) != consumed_domains
            or any(
                value
                not in {
                    "candidate-route",
                    "refine-route",
                    "half-survival",
                    "recall-dropout",
                }
                for value in consumed_domains
            )
            or bool(consumed_domains) != (mode == "keyed")
        ):
            raise TensorTransactionContractError("invalid consumed RNG domains")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "algorithm", algorithm)
        object.__setattr__(self, "candidate_manifest_fingerprint", candidate_manifest_fingerprint)
        object.__setattr__(self, "plan_config_fingerprint", plan_config_fingerprint)
        object.__setattr__(self, "route_fingerprint", route_fingerprint)
        object.__setattr__(self, "branch_policy_fingerprint", branch_policy_fingerprint)
        object.__setattr__(self, "execution_rng_fingerprint", execution_rng_fingerprint)
        object.__setattr__(self, "execution_rng_stream_key", execution_rng_stream_key)
        object.__setattr__(self, "consumed_domains", consumed_domains)

    @classmethod
    def deterministic(
        cls,
        *,
        candidate_manifest_fingerprint: str,
        plan_config_fingerprint: str,
        route_fingerprint: str,
        branch_policy_fingerprint: str | None,
    ) -> "ExecutionContextReceipt":
        return cls(
            schema_version=3,
            mode="deterministic",
            algorithm="none",
            candidate_manifest_fingerprint=candidate_manifest_fingerprint,
            plan_config_fingerprint=plan_config_fingerprint,
            route_fingerprint=route_fingerprint,
            branch_policy_fingerprint=branch_policy_fingerprint,
            execution_rng_fingerprint=None,
            execution_rng_stream_key=None,
            consumed_domains=(),
            _factory_token=_EXECUTION_CONTEXT_RECEIPT_FACTORY_TOKEN,
        )

    @classmethod
    def keyed(
        cls,
        *,
        candidate_manifest_fingerprint: str,
        plan_config_fingerprint: str,
        route_fingerprint: str,
        branch_policy_fingerprint: str | None,
        execution_rng_fingerprint: str,
        execution_rng_stream_key: str,
        consumed_domains: tuple[str, ...],
    ) -> "ExecutionContextReceipt":
        return cls(
            schema_version=3,
            mode="keyed",
            algorithm="sha256-seeded-torch-generator@2",
            candidate_manifest_fingerprint=candidate_manifest_fingerprint,
            plan_config_fingerprint=plan_config_fingerprint,
            route_fingerprint=route_fingerprint,
            branch_policy_fingerprint=branch_policy_fingerprint,
            execution_rng_fingerprint=execution_rng_fingerprint,
            execution_rng_stream_key=execution_rng_stream_key,
            consumed_domains=consumed_domains,
            _factory_token=_EXECUTION_CONTEXT_RECEIPT_FACTORY_TOKEN,
        )

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "ref": self._runtime_contract_ref,
                "schema_version": self.schema_version,
                "mode": self.mode,
                "algorithm": self.algorithm,
                "candidate_manifest_fingerprint": self.candidate_manifest_fingerprint,
                "plan_config_fingerprint": self.plan_config_fingerprint,
                "route_fingerprint": self.route_fingerprint,
                "branch_policy_fingerprint": self.branch_policy_fingerprint,
                "execution_rng_fingerprint": self.execution_rng_fingerprint,
                "execution_rng_stream_key": self.execution_rng_stream_key,
                "consumed_domains": list(self.consumed_domains),
            }
        )


def frozen_mse_scorer_config_fingerprint() -> str:
    """Return the canonical identity of the built-in detached MSE scorer."""

    return _fingerprint(
        {
            "ref": FROZEN_MSE_SCORER_REF,
            "metric": "mean-squared-error",
            "reduction": "mean-all-elements",
            "lower_is_better": True,
            "future": "detached-cpu-exogenous",
        }
    )


def _batched_refine_result_state_fingerprint(
    result: object,
    manifest_fingerprint: str,
) -> str:
    from .batched_refine import BatchedRefineResult

    if not isinstance(result, BatchedRefineResult):
        raise TypeError("result must be BatchedRefineResult")
    result.candidates.assert_unchanged()
    result.assert_unchanged()
    diagnostics = {
        **{
            f"branch:{name}": _cpu_tensor_fingerprint(
                tensor.detach().to("cpu").contiguous()
            )
            for name, tensor in sorted(result.branch_diagnostics.items())
        },
        **{
            f"global:{name}": _cpu_tensor_fingerprint(
                tensor.detach().to("cpu").contiguous()
            )
            for name, tensor in sorted(result.global_diagnostics.items())
        },
    }
    return _fingerprint(
        {
            "ref": "arti/batched-refine-executor-state@1",
            "candidate_manifest_fingerprint": manifest_fingerprint,
            "value": _cpu_tensor_fingerprint(
                result.value.detach().to("cpu").contiguous()
            ),
            "delta": _cpu_tensor_fingerprint(
                result.delta.detach().to("cpu").contiguous()
            ),
            "diagnostics": diagnostics,
        }
    )


def _runtime_tensor_identity(value: Tensor) -> dict[str, object]:
    return {
        "instance_token": id(value),
        "data_ptr": value.data_ptr(),
        "version": value._version,
        "shape": list(value.shape),
        "stride": list(value.stride()),
        "dtype": str(value.dtype),
        "device": str(value.device),
    }


def _resident_batched_refine_manifest_fingerprint(result: object) -> str:
    """Bind one live candidate object without copying CUDA payload to host."""

    from .batched_refine import BatchedRefineResult
    from .component_registry import component_ref

    if not isinstance(result, BatchedRefineResult):
        raise TypeError("result must be BatchedRefineResult")
    candidates = result.candidates
    candidates.assert_unchanged()
    result.assert_unchanged()
    return _fingerprint(
        {
            "ref": "arti/resident-batched-refine-manifest@1",
            "candidate_ref": component_ref(candidates),
            "candidate_instance_token": id(candidates),
            "candidate_tensor_tokens": list(candidates.candidate_tensor_tokens),
            "candidate_tensor_versions": list(candidates.candidate_tensor_versions),
            "source_ref": candidates.source_ref,
            "source_instance_token": candidates.source_instance_token,
            "source_config_fingerprint": candidates.source_config_fingerprint,
            "source_execution_tensor_lineage": [
                list(value) for value in candidates.source_execution_tensor_lineage
            ],
            "layout_fingerprint": candidates.layout_fingerprint,
            "execution_layout": result.execution_layout,
            "partition_layout_fingerprint": candidates.partition_layout_fingerprint,
            "formula_ref": candidates.formula_ref,
            "formula_config_fingerprint": candidates.formula_config_fingerprint,
            "topology_lineage": list(candidates.topology_lineage),
            "max_k": candidates.max_k,
            "schema_version": candidates.schema_version,
        }
    )


def _resident_batched_refine_state_fingerprint(
    result: object,
    manifest_fingerprint: str,
) -> str:
    """Fingerprint live CUDA identity and versions, never tensor contents."""

    from .batched_refine import BatchedRefineResult

    if not isinstance(result, BatchedRefineResult):
        raise TypeError("result must be BatchedRefineResult")
    result.candidates.assert_unchanged()
    result.assert_unchanged()
    return _fingerprint(
        {
            "ref": "arti/resident-batched-refine-executor-state@1",
            "candidate_manifest_fingerprint": manifest_fingerprint,
            "value": _runtime_tensor_identity(result.value),
            "delta": _runtime_tensor_identity(result.delta),
            "diagnostics": {
                **{
                    f"branch:{name}": _runtime_tensor_identity(tensor)
                    for name, tensor in sorted(result.branch_diagnostics.items())
                },
                **{
                    f"global:{name}": _runtime_tensor_identity(tensor)
                    for name, tensor in sorted(result.global_diagnostics.items())
                },
            },
        }
    )


@dataclass(frozen=True, init=False)
class BatchedRefineExecutor:
    """Factory-owned authority identity for one completed branch execution."""

    schema_version: int
    candidate_ref: str
    candidate_manifest_fingerprint: str
    plan_ref: str
    plan_config_fingerprint: str
    execution_layout: str
    operation_ref: str | None
    route_fingerprint: str
    topology_refs: tuple[str, ...]
    topology_contract_fingerprints: tuple[str, ...]
    branch_policy_fingerprint: str | None
    execution_context: ExecutionContextReceipt
    state_fingerprint: str
    identity_mode: str
    _component_reference: ClassVar[str] = "arti/batched-refine@1"

    def __init__(
        self,
        *,
        schema_version: int,
        candidate_ref: str,
        candidate_manifest_fingerprint: str,
        plan_ref: str,
        plan_config_fingerprint: str,
        execution_layout: str,
        operation_ref: str | None,
        route_fingerprint: str,
        topology_refs: tuple[str, ...],
        topology_contract_fingerprints: tuple[str, ...],
        branch_policy_fingerprint: str | None,
        execution_context: ExecutionContextReceipt,
        state_fingerprint: str,
        identity_mode: str,
        _factory_token: object,
    ) -> None:
        if _factory_token is not _BATCHED_REFINE_EXECUTOR_FACTORY_TOKEN:
            raise TensorTransactionContractError(
                "BatchedRefineExecutor must come from BatchedRefineExecutor.from_result"
            )
        if schema_version != 1:
            raise TensorTransactionContractError(
                "unsupported BatchedRefineExecutor schema"
            )
        for value, name in (
            (candidate_ref, "candidate_ref"),
            (plan_ref, "plan_ref"),
        ):
            _require_component_ref(value, name)
        if operation_ref is not None:
            _require_component_ref(operation_ref, "operation_ref")
        if execution_layout not in {"static_capacity", "packed_active"}:
            raise TensorTransactionContractError(
                "unsupported BatchedRefineExecutor execution layout"
            )
        for value, name in (
            (candidate_manifest_fingerprint, "candidate_manifest_fingerprint"),
            (plan_config_fingerprint, "plan_config_fingerprint"),
            (route_fingerprint, "route_fingerprint"),
            (state_fingerprint, "state_fingerprint"),
        ):
            _require_sha256(value, name)
        if branch_policy_fingerprint is not None:
            _require_sha256(branch_policy_fingerprint, "branch_policy_fingerprint")
        if identity_mode not in {"content", "resident_identity"}:
            raise TensorTransactionContractError(
                "unsupported BatchedRefineExecutor identity mode"
            )
        if not isinstance(execution_context, ExecutionContextReceipt):
            raise TypeError("execution_context must be ExecutionContextReceipt")
        topology = tuple(topology_refs)
        for reference in topology:
            _require_component_ref(reference, "topology_refs")
        topology_contracts = tuple(topology_contract_fingerprints)
        for fingerprint in topology_contracts:
            _require_sha256(fingerprint, "topology_contract_fingerprints")
        object.__setattr__(self, "schema_version", schema_version)
        object.__setattr__(self, "candidate_ref", candidate_ref)
        object.__setattr__(
            self,
            "candidate_manifest_fingerprint",
            candidate_manifest_fingerprint,
        )
        object.__setattr__(self, "plan_ref", plan_ref)
        object.__setattr__(self, "plan_config_fingerprint", plan_config_fingerprint)
        object.__setattr__(self, "execution_layout", execution_layout)
        object.__setattr__(self, "operation_ref", operation_ref)
        object.__setattr__(self, "route_fingerprint", route_fingerprint)
        object.__setattr__(self, "topology_refs", topology)
        object.__setattr__(
            self,
            "topology_contract_fingerprints",
            topology_contracts,
        )
        object.__setattr__(
            self, "branch_policy_fingerprint", branch_policy_fingerprint
        )
        object.__setattr__(self, "execution_context", execution_context)
        object.__setattr__(self, "state_fingerprint", state_fingerprint)
        object.__setattr__(self, "identity_mode", identity_mode)

    @classmethod
    def from_result(cls, result: object) -> "BatchedRefineExecutor":
        from .batched_refine import BatchedRefineResult
        from .component_registry import component_ref

        if not isinstance(result, BatchedRefineResult):
            raise TypeError("result must be BatchedRefineResult")
        manifest = batched_refine_manifest_fingerprint(result)
        route_fingerprint = result.formula_route_fingerprint or ("0" * 64)
        context_factory = (
            ExecutionContextReceipt.deterministic
            if result.execution_rng_fingerprint is None
            else ExecutionContextReceipt.keyed
        )
        context_kwargs = {
            "candidate_manifest_fingerprint": manifest,
            "plan_config_fingerprint": result.plan_config_fingerprint,
            "route_fingerprint": route_fingerprint,
            "branch_policy_fingerprint": result.branch_policy_fingerprint,
        }
        if result.execution_rng_fingerprint is not None:
            context_kwargs["execution_rng_fingerprint"] = result.execution_rng_fingerprint
            context_kwargs["execution_rng_stream_key"] = result.execution_rng_stream_key
            context_kwargs["consumed_domains"] = result.execution_rng_domains
        execution_context = context_factory(**context_kwargs)
        return cls(
            schema_version=1,
            candidate_ref=component_ref(result.candidates),
            candidate_manifest_fingerprint=manifest,
            plan_ref=result.plan_ref,
            plan_config_fingerprint=result.plan_config_fingerprint,
            execution_layout=result.execution_layout,
            operation_ref=result.operation_ref,
            route_fingerprint=route_fingerprint,
            topology_refs=result.topology_refs,
            topology_contract_fingerprints=(
                result.topology_contract_fingerprints
            ),
            branch_policy_fingerprint=result.branch_policy_fingerprint,
            execution_context=execution_context,
            state_fingerprint=_batched_refine_result_state_fingerprint(
                result,
                manifest,
            ),
            identity_mode="content",
            _factory_token=_BATCHED_REFINE_EXECUTOR_FACTORY_TOKEN,
        )

    @classmethod
    def from_resident_result(cls, result: object) -> "BatchedRefineExecutor":
        """Bind a live CUDA result without a full device-to-host fingerprint."""

        from .batched_refine import BatchedRefineResult
        from .component_registry import component_ref

        if not isinstance(result, BatchedRefineResult):
            raise TypeError("result must be BatchedRefineResult")
        if result.value.device.type != "cuda":
            raise TensorTransactionContractError(
                "resident executor requires a CUDA Batched Refine result"
            )
        manifest = _resident_batched_refine_manifest_fingerprint(result)
        route_fingerprint = result.formula_route_fingerprint or ("0" * 64)
        context_factory = (
            ExecutionContextReceipt.deterministic
            if result.execution_rng_fingerprint is None
            else ExecutionContextReceipt.keyed
        )
        context_kwargs = {
            "candidate_manifest_fingerprint": manifest,
            "plan_config_fingerprint": result.plan_config_fingerprint,
            "route_fingerprint": route_fingerprint,
            "branch_policy_fingerprint": result.branch_policy_fingerprint,
        }
        if result.execution_rng_fingerprint is not None:
            context_kwargs["execution_rng_fingerprint"] = (
                result.execution_rng_fingerprint
            )
            context_kwargs["execution_rng_stream_key"] = (
                result.execution_rng_stream_key
            )
            context_kwargs["consumed_domains"] = result.execution_rng_domains
        execution_context = context_factory(**context_kwargs)
        return cls(
            schema_version=1,
            candidate_ref=component_ref(result.candidates),
            candidate_manifest_fingerprint=manifest,
            plan_ref=result.plan_ref,
            plan_config_fingerprint=result.plan_config_fingerprint,
            execution_layout=result.execution_layout,
            operation_ref=result.operation_ref,
            route_fingerprint=route_fingerprint,
            topology_refs=result.topology_refs,
            topology_contract_fingerprints=(
                result.topology_contract_fingerprints
            ),
            branch_policy_fingerprint=result.branch_policy_fingerprint,
            execution_context=execution_context,
            state_fingerprint=_resident_batched_refine_state_fingerprint(
                result,
                manifest,
            ),
            identity_mode="resident_identity",
            _factory_token=_BATCHED_REFINE_EXECUTOR_FACTORY_TOKEN,
        )

    @property
    def config_fingerprint(self) -> str:
        from .component_registry import component_spec

        return component_spec(self).config_fingerprint

    def assert_matches(self, result: object) -> None:
        from .batched_refine import BatchedRefineResult
        from .component_registry import component_ref

        if not isinstance(result, BatchedRefineResult):
            raise TypeError("result must be BatchedRefineResult")
        manifest = (
            batched_refine_manifest_fingerprint(result)
            if self.identity_mode == "content"
            else _resident_batched_refine_manifest_fingerprint(result)
        )
        route_fingerprint = result.formula_route_fingerprint or ("0" * 64)
        if (
            component_ref(result.candidates) != self.candidate_ref
            or manifest != self.candidate_manifest_fingerprint
            or result.plan_ref != self.plan_ref
            or result.plan_config_fingerprint != self.plan_config_fingerprint
            or result.execution_layout != self.execution_layout
            or result.operation_ref != self.operation_ref
            or route_fingerprint != self.route_fingerprint
            or tuple(result.topology_refs) != self.topology_refs
            or tuple(result.topology_contract_fingerprints)
            != self.topology_contract_fingerprints
            or result.branch_policy_fingerprint != self.branch_policy_fingerprint
            or (
                ExecutionContextReceipt.deterministic(
                    candidate_manifest_fingerprint=manifest,
                    plan_config_fingerprint=result.plan_config_fingerprint,
                    route_fingerprint=route_fingerprint,
                    branch_policy_fingerprint=result.branch_policy_fingerprint,
                )
                if result.execution_rng_fingerprint is None
                else ExecutionContextReceipt.keyed(
                    candidate_manifest_fingerprint=manifest,
                    plan_config_fingerprint=result.plan_config_fingerprint,
                    route_fingerprint=route_fingerprint,
                    branch_policy_fingerprint=result.branch_policy_fingerprint,
                    execution_rng_fingerprint=result.execution_rng_fingerprint,
                    execution_rng_stream_key=result.execution_rng_stream_key,
                    consumed_domains=result.execution_rng_domains,
                )
            ).fingerprint
            != self.execution_context.fingerprint
            or (
                _batched_refine_result_state_fingerprint(result, manifest)
                if self.identity_mode == "content"
                else _resident_batched_refine_state_fingerprint(result, manifest)
            )
            != self.state_fingerprint
        ):
            raise TensorTransactionContractError(
                "Batched Refine result does not match its executor identity"
            )

    def bind(
        self,
        runtime: VolatileTensorRuntime,
        snapshot: TensorSnapshot,
        key: str,
        *,
        address_namespace: str,
        partition_id: str,
        logical_id: str,
        role: str,
        authority: object,
        state_schema_ref: str = "arti/batched-refine-result@1",
        provenance_fingerprint: str,
    ) -> ExternalTensorBinding:
        """Bind a target page to this exact completed executor closure."""

        from .tensor_binding import TensorAuthority, bind_external_tensor

        if not isinstance(authority, TensorAuthority):
            raise TypeError("authority must be TensorAuthority")
        return bind_external_tensor(
            runtime,
            snapshot,
            key,
            address_namespace=address_namespace,
            partition_id=partition_id,
            logical_id=logical_id,
            role=role,
            authority=authority,
            component_ref=self._component_reference,
            component_config_fingerprint=self.config_fingerprint,
            state_schema_ref=state_schema_ref,
            producer_state_fingerprint=self.state_fingerprint,
            provenance_fingerprint=provenance_fingerprint,
        ).binding


def batched_refine_manifest_fingerprint(result: object) -> str:
    """Fingerprint the real K-candidate bridge at an explicit CPU authority boundary."""

    from .batched_refine import BatchedRefineResult

    if not isinstance(result, BatchedRefineResult):
        raise TypeError("result must be BatchedRefineResult")
    result.candidates.assert_unchanged()
    result.assert_unchanged()
    candidates = result.candidates
    tensors = {
        "group": candidates.candidate_group_index,
        "partition": candidates.candidate_partition_index,
        "slot": candidates.candidate_slot_index,
        "slot_weight": candidates.candidate_slot_weight,
        "context": candidates.candidate_context,
        "route_mass": candidates.route_mass,
        "selection_weight": candidates.selection_weight,
        "candidate_log_score": candidates.candidate_log_score,
        "mask": candidates.candidate_mask,
        "branch_mask": candidates.branch_mask,
        "token_mask": candidates.token_mask,
        "branch_origin_index": candidates.branch_origin_index,
        "active_k": candidates.active_k,
        "requested_active_k": candidates.requested_active_k,
    }
    if candidates.factor_candidate_rank is not None:
        tensors["factor_candidate_rank"] = candidates.factor_candidate_rank
    if candidates.factor_route_index is not None:
        tensors["factor_route_index"] = candidates.factor_route_index
    fingerprints = {
        name: _cpu_tensor_fingerprint(tensor.detach().to("cpu").contiguous())
        for name, tensor in tensors.items()
    }
    return _fingerprint(
        {
            "ref": "arti/batched-refine-manifest@2",
            "source_ref": candidates.source_ref,
            "source_config_fingerprint": candidates.source_config_fingerprint,
            "layout_fingerprint": candidates.layout_fingerprint,
            "max_k": candidates.max_k,
            "formula_beam_width": candidates.formula_beam_width,
            "active_k": [int(value) for value in candidates.active_k.tolist()],
            "requested_active_k": [
                int(value) for value in candidates.requested_active_k.tolist()
            ],
            "active_partition_count": [
                int(value)
                for value in candidates.active_partition_count().tolist()
            ],
            "active_partition_mask": candidates.active_partition_mask().tolist(),
            "active_branch_count_by_partition": (
                candidates.active_branch_count_by_partition().tolist()
            ),
            "source_topk": candidates.source_topk,
            "group_count": candidates.group_count,
            "group_size": candidates.group_size,
            "routing_normalizer": candidates.routing_normalizer,
            "partition_names": list(candidates.partition_names),
            "partition_ranges": [list(value) for value in candidates.partition_ranges],
            "partition_member_fingerprints": list(
                candidates.partition_member_fingerprints
            ),
            "partition_layout_fingerprint": candidates.partition_layout_fingerprint,
            "value_composition": candidates.value_composition,
            "candidate_policy": candidates.candidate_policy,
            "partition_quota": list(candidates.partition_quota),
            "partition_coherence": candidates.partition_coherence,
            "factor_count": candidates.factor_count,
            "formula_ref": candidates.formula_ref,
            "formula_config_fingerprint": candidates.formula_config_fingerprint,
            "candidate_schema_version": candidates.schema_version,
            "plan_ref": result.plan_ref,
            "plan_config_fingerprint": result.plan_config_fingerprint,
            "execution_layout": result.execution_layout,
            "operation_ref": result.operation_ref,
            "formula_route_fingerprint": result.formula_route_fingerprint,
            "topology_refs": list(result.topology_refs),
            "topology_contract_fingerprints": list(
                result.topology_contract_fingerprints
            ),
            "branch_policy_fingerprint": result.branch_policy_fingerprint,
            "execution_rng_fingerprint": result.execution_rng_fingerprint,
            "execution_rng_stream_key": result.execution_rng_stream_key,
            "execution_rng_domains": list(result.execution_rng_domains),
            "candidate_tensors": fingerprints,
        }
    )


def stage_batched_refine_result(
    result: object,
    executor: BatchedRefineExecutor,
    spec: BranchBatchSpecV2,
    bindings: Sequence[ExternalTensorBinding],
) -> tuple[BatchedRefineOverlayProposal, ...]:
    """Stage K real trajectories as CPU COW proposals without rerunning Recall."""

    from .batched_refine import BatchedRefineResult

    if not isinstance(result, BatchedRefineResult):
        raise TypeError("result must be BatchedRefineResult")
    if not isinstance(executor, BatchedRefineExecutor):
        raise TypeError("executor must be BatchedRefineExecutor")
    if not isinstance(spec, BranchBatchSpecV2):
        raise TypeError("spec must be BranchBatchSpecV2")
    executor.assert_matches(result)
    result.candidates.assert_unchanged()
    result.assert_unchanged()
    branch_mask = result.candidates.branch_mask.detach().to("cpu")
    branch_origin = result.candidates.branch_origin_index.detach().to("cpu")
    physical_by_origin = torch.argsort(branch_origin, dim=1, stable=True)
    active_by_origin = branch_mask.gather(1, physical_by_origin)
    active_origins = torch.nonzero(
        active_by_origin.any(dim=0),
        as_tuple=False,
    ).flatten().tolist()
    if len(active_origins) != len(spec.branch_ids):
        raise TensorTransactionContractError(
            "active Batched Refine K does not match BranchBatchSpec@2"
        )
    if active_origins != list(range(len(spec.branch_ids))):
        raise TensorTransactionContractError(
            "active Batched Refine origins do not match canonical branch authority"
        )
    manifest = batched_refine_manifest_fingerprint(result)
    if manifest != spec.candidate_manifest_fingerprint:
        raise TensorTransactionContractError("Batched Refine candidates do not match branch spec")
    if (
        spec.executor_ref != executor._component_reference
        or spec.program_fingerprint != executor.plan_config_fingerprint
        or spec.route_fingerprint != executor.route_fingerprint
    ):
        raise TensorTransactionContractError(
            "BranchBatchSpec@2 does not match the Batched Refine executor"
        )
    normalized_bindings = tuple(bindings)
    if len(normalized_bindings) != len(spec.branch_ids):
        raise TensorTransactionContractError("bindings must contain one target binding per branch")
    token_steps_tensor = result.branch_diagnostics.get(
        "recall_token_steps_committed"
    )
    token_attempted_tensor = result.branch_diagnostics.get(
        "recall_token_steps_attempted"
    )
    steps_tensor = result.branch_diagnostics.get("recall_steps_committed")
    attempted_tensor = result.branch_diagnostics.get("recall_steps_attempted")
    stop_reason_tensor = result.branch_diagnostics.get("recall_token_stop_reason")
    if stop_reason_tensor is None:
        stop_reason_tensor = result.branch_diagnostics.get("recall_stop_reason")
    if (
        steps_tensor is None
        or attempted_tensor is None
        or stop_reason_tensor is None
    ):
        raise TensorTransactionContractError(
            "Batched Refine staging requires committed-step and stop diagnostics"
        )
    formula_cells_tensor = result.branch_diagnostics.get("batched_formula_cells")
    formula_route_tensor = result.branch_diagnostics.get(
        "batched_formula_route_applications"
    )
    formula_fire_tensor = result.branch_diagnostics.get(
        "batched_formula_fire_count"
    )
    formula_commit_tensor = result.branch_diagnostics.get(
        "batched_formula_commit_count"
    )
    outputs = result.value.detach().to("cpu").contiguous()
    deltas = result.delta.detach().to("cpu").contiguous()
    batch_index_cpu = torch.arange(outputs.shape[0], dtype=torch.long)

    def canonical_branch_slice(tensor: Tensor, origin: int) -> Tensor:
        physical = physical_by_origin[:, origin].to(device=tensor.device)
        batch_index = torch.arange(
            tensor.shape[0],
            device=tensor.device,
            dtype=torch.long,
        )
        return tensor[batch_index, physical]

    staged: list[BatchedRefineOverlayProposal] = []
    for origin in active_origins:
        branch_id = spec.branch_ids[origin]
        binding = normalized_bindings[origin]
        if not isinstance(binding, ExternalTensorBinding):
            raise TensorTransactionContractError("bindings must be ExternalTensorBinding values")
        if binding.component_ref != executor._component_reference:
            raise TensorTransactionContractError(
                "binding must match the branch executor contract"
            )
        if (
            binding.component_config_fingerprint != executor.config_fingerprint
            or binding.producer_state_fingerprint != executor.state_fingerprint
        ):
            raise TensorTransactionContractError(
                "binding does not match the Batched Refine executor identity"
            )
        if binding.tensor_ref.key != spec.allowed_write_keys[0]:
            raise TensorTransactionContractError(
                "binding is outside the coordinator write authority"
            )
        physical = physical_by_origin[:, origin]
        output = outputs[batch_index_cpu, physical].contiguous()
        delta = deltas[batch_index_cpu, physical].contiguous()
        finite = bool(torch.isfinite(output).all() and torch.isfinite(delta).all())
        if not finite:
            raise TensorTransactionContractError(
                "Batched Refine output and delta must be finite before staging"
            )
        proposal = ExternalTensorProposal(
            binding,
            output,
            producer_ref=executor._component_reference,
            producer_config_fingerprint=executor.config_fingerprint,
            producer_state_fingerprint=executor.state_fingerprint,
        )
        branch_steps = (
            canonical_branch_slice(token_steps_tensor, origin).detach().to("cpu")
            if token_steps_tensor is not None
            else canonical_branch_slice(steps_tensor, origin).detach().to("cpu")
        )
        branch_attempted = (
            canonical_branch_slice(token_attempted_tensor, origin).detach().to("cpu")
            if token_attempted_tensor is not None
            else canonical_branch_slice(attempted_tensor, origin).detach().to("cpu")
        )
        actual_steps = int(branch_steps.max().item())
        branch_stop_reason = canonical_branch_slice(
            stop_reason_tensor,
            origin,
        ).detach().to("cpu")
        valid_tokens = result.candidates.token_mask.detach().to("cpu") & active_by_origin[
            :, origin
        ].unsqueeze(1)
        if branch_stop_reason.ndim == 1:
            reason_values = branch_stop_reason[valid_tokens.any(dim=1)]
        elif branch_stop_reason.shape == valid_tokens.shape:
            reason_values = branch_stop_reason[valid_tokens]
        else:
            raise TensorTransactionContractError(
                "Batched Refine stop diagnostics have an invalid branch shape"
            )
        if reason_values.numel() == 0:
            stop_reason = "masked"
        else:
            unique_reasons = torch.unique(reason_values).tolist()
            reason_names = {
                0: "max-steps",
                1: "converged",
                2: "nonfinite",
                3: "cycle",
                4: "masked",
            }
            stop_reason = (
                reason_names.get(int(unique_reasons[0]), "unknown")
                if len(unique_reasons) == 1
                else "mixed"
            )
        formula_cells = (
            0
            if formula_cells_tensor is None
            else int(
                canonical_branch_slice(formula_cells_tensor, origin)
                .detach()
                .sum()
                .to("cpu")
                .item()
            )
        )
        formula_routes = (
            actual_steps
            if formula_route_tensor is None
            else int(
                canonical_branch_slice(formula_route_tensor, origin)
                .detach()
                .sum()
                .to("cpu")
                .item()
            )
        )
        formula_fire = (
            0
            if formula_fire_tensor is None
            else int(
                canonical_branch_slice(formula_fire_tensor, origin)
                .detach()
                .sum()
                .to("cpu")
                .item()
            )
        )
        formula_commit = (
            0
            if formula_commit_tensor is None
            else int(
                canonical_branch_slice(formula_commit_tensor, origin)
                .detach()
                .sum()
                .to("cpu")
                .item()
            )
        )
        candidate_fp = _fingerprint(
            {
                "manifest": manifest,
                "physical_index_by_sample": physical.tolist(),
                "branch_origin": origin,
                "branch_id": branch_id,
            }
        )
        output_fp = _cpu_tensor_fingerprint(output)
        delta_fp = _cpu_tensor_fingerprint(delta)
        trajectory_fp = _fingerprint(
            {
                "ref": "arti/batched-refine-trajectory@1",
                "candidate": candidate_fp,
                "output": output_fp,
                "delta": delta_fp,
                "actual_steps": actual_steps,
                "token_steps": _cpu_tensor_fingerprint(branch_steps.contiguous()),
                "stop_reason": _cpu_tensor_fingerprint(
                    branch_stop_reason.contiguous()
                ),
            }
        )
        work = BranchWorkReceipt(
            actual_steps=actual_steps,
            formula_cells=formula_cells,
            route_applications=formula_routes,
            fire_count=formula_fire,
            commit_count=formula_commit,
            operation_count=output.numel() * actual_steps + formula_cells,
            stop_reason=stop_reason,
            residual_fingerprint=delta_fp,
            finite=finite,
            logical_steps_attempted=int(branch_attempted.sum().item()),
            logical_steps_committed=int(branch_steps.sum().item()),
            outer_kernel_steps=int(
                result.global_diagnostics["recall_kernel_steps"]
                .detach()
                .to("cpu")
                .item()
            ),
            resident_operation_steps=int(
                result.global_diagnostics.get(
                    "batched_resident_operation_steps",
                    torch.zeros((), dtype=torch.int64),
                )
                .detach()
                .to("cpu")
                .item()
            ),
            measurement_kind="declared-estimate",
            logical_step_unit=(
                "token-step" if token_steps_tensor is not None else "sample-step"
            ),
        )
        staged.append(
            BatchedRefineOverlayProposal(
                branch_id=branch_id,
                spec_fingerprint=spec.fingerprint,
                candidate_fingerprint=candidate_fp,
                trajectory_fingerprint=trajectory_fp,
                proposal=proposal,
                delta=delta,
                work=work,
                _factory_token=_BATCHED_OVERLAY_FACTORY_TOKEN,
            )
        )
    staged.sort(key=lambda item: spec.branch_ids.index(item.branch_id))
    return tuple(staged)


def score_batched_refine_proposals(
    proposals: Sequence[BatchedRefineOverlayProposal],
    future: Tensor,
    spec: BranchBatchSpecV2,
) -> BranchScoreBatchReceipt:
    """Score K staged trajectories against one frozen, exogenous CPU target."""

    normalized = tuple(proposals)
    if (
        spec.scorer_ref != FROZEN_MSE_SCORER_REF
        or spec.scorer_config_fingerprint
        != frozen_mse_scorer_config_fingerprint()
    ):
        raise TensorTransactionContractError(
            "Batched Refine scoring requires the canonical frozen MSE scorer"
        )
    if tuple(item.branch_id for item in normalized) != spec.branch_ids:
        raise TensorTransactionContractError("proposal order must match BranchBatchSpec@2")
    if any(item.spec_fingerprint != spec.fingerprint for item in normalized):
        raise TensorTransactionContractError("proposal does not bind the scoring spec")
    if future.device.type != "cpu" or future.requires_grad:
        raise TensorTransactionContractError("future must be detached CPU storage")
    if (future.is_floating_point() or future.is_complex()) and not bool(torch.isfinite(future).all()):
        raise TensorTransactionContractError("future must contain only finite values")
    frozen_future = future.detach().clone(memory_format=torch.contiguous_format)
    future_fp = _cpu_tensor_fingerprint(frozen_future)
    if future_fp != spec.future_tape_fingerprint:
        raise TensorTransactionContractError("future tensor does not match branch provenance")
    scores: list[float] = []
    with torch.no_grad():
        for item in normalized:
            value = item.proposal.value
            if value.shape != frozen_future.shape:
                raise TensorTransactionContractError("future shape must match each branch output")
            score = float(torch.mean((value - frozen_future) ** 2).item())
            if not torch.isfinite(torch.tensor(score)):
                raise TensorTransactionContractError("frozen scorer produced a non-finite score")
            scores.append(score)
    return BranchScoreBatchReceipt(
        spec_fingerprint=spec.fingerprint,
        branch_ids=spec.branch_ids,
        overlay_fingerprints=tuple(item.fingerprint for item in normalized),
        candidate_fingerprints=tuple(item.candidate_fingerprint for item in normalized),
        future_fingerprint=future_fp,
        scorer_ref=spec.scorer_ref,
        scorer_config_fingerprint=spec.scorer_config_fingerprint,
        tie_policy=spec.tie_policy,
        scores=tuple(scores),
        _factory_token=_BATCH_SCORE_FACTORY_TOKEN,
    )


_RESIDENT_SCORE_FACTORY_TOKEN = object()
_RESIDENT_DECISION_FACTORY_TOKEN = object()
_RESIDENT_COMMIT_FACTORY_TOKEN = object()
_RESIDENT_RUN_COUNTER = itertools.count(1)
_RESIDENT_RUN_COUNTER_LOCK = RLock()


@dataclass(frozen=True, init=False)
class ResidentBranchScoreReceipt:
    """Host-visible K-score receipt for a GPU-resident branch result."""

    spec_fingerprint: str
    run_instance_token: int
    result_manifest_fingerprint: str
    branch_ids: tuple[str, ...]
    future_fingerprint: str
    scorer_ref: str
    scorer_config_fingerprint: str
    tie_policy: str
    scores: tuple[float, ...]
    receipt_fingerprint: str
    _runtime_contract_ref: ClassVar[str] = "arti/resident-branch-score@1"

    def __init__(
        self,
        *,
        spec_fingerprint: str,
        run_instance_token: int,
        result_manifest_fingerprint: str,
        branch_ids: tuple[str, ...],
        future_fingerprint: str,
        scorer_ref: str,
        scorer_config_fingerprint: str,
        tie_policy: str,
        scores: tuple[float, ...],
        _factory_token: object,
    ) -> None:
        if _factory_token is not _RESIDENT_SCORE_FACTORY_TOKEN:
            raise TensorTransactionContractError(
                "ResidentBranchScoreReceipt must come from ResidentBranchRun.score"
            )
        if not branch_ids or len(branch_ids) != len(scores):
            raise TensorTransactionContractError(
                "resident score must contain one value per branch"
            )
        if (
            isinstance(run_instance_token, bool)
            or not isinstance(run_instance_token, int)
            or run_instance_token <= 0
        ):
            raise TensorTransactionContractError(
                "resident score run_instance_token must be positive"
            )
        for value in (
            spec_fingerprint,
            result_manifest_fingerprint,
            future_fingerprint,
            scorer_config_fingerprint,
        ):
            _require_sha256(value, "resident score fingerprint")
        _require_component_ref(scorer_ref, "scorer_ref")
        _require_identifier(tie_policy, "tie_policy")
        if any(not math.isfinite(value) for value in scores):
            raise TensorTransactionContractError("resident scores must be finite")
        payload = {
            "ref": self._runtime_contract_ref,
            "spec_fingerprint": spec_fingerprint,
            "run_instance_token": run_instance_token,
            "result_manifest_fingerprint": result_manifest_fingerprint,
            "branch_ids": list(branch_ids),
            "future_fingerprint": future_fingerprint,
            "scorer_ref": scorer_ref,
            "scorer_config_fingerprint": scorer_config_fingerprint,
            "tie_policy": tie_policy,
            "scores": list(scores),
        }
        for name, value in payload.items():
            if name == "ref":
                continue
            object.__setattr__(self, name, tuple(value) if isinstance(value, list) else value)
        object.__setattr__(self, "receipt_fingerprint", _fingerprint(payload))


@dataclass(frozen=True, init=False)
class ResidentBranchDecision:
    """Factory-owned host authorization for one resident publication."""

    kind: str
    run_instance_token: int
    spec_fingerprint: str
    score_receipt_fingerprint: str | None
    winner_origin: int | None
    weights: tuple[float, ...]
    idempotency_key: str
    decision_fingerprint: str
    _runtime_contract_ref: ClassVar[str] = "arti/resident-branch-decision@1"

    def __init__(
        self,
        *,
        kind: str,
        run_instance_token: int,
        spec_fingerprint: str,
        score_receipt_fingerprint: str | None,
        winner_origin: int | None,
        weights: tuple[float, ...],
        idempotency_key: str,
        _factory_token: object,
    ) -> None:
        if _factory_token is not _RESIDENT_DECISION_FACTORY_TOKEN:
            raise TensorTransactionContractError(
                "ResidentBranchDecision must come from ResidentBranchRun"
            )
        if kind not in {"winner", "mixture", "discard"}:
            raise TensorTransactionContractError("invalid resident decision kind")
        if (
            isinstance(run_instance_token, bool)
            or not isinstance(run_instance_token, int)
            or run_instance_token <= 0
        ):
            raise TensorTransactionContractError(
                "resident decision run_instance_token must be positive"
            )
        _require_sha256(spec_fingerprint, "spec_fingerprint")
        if score_receipt_fingerprint is not None:
            _require_sha256(score_receipt_fingerprint, "score_receipt_fingerprint")
        _require_identifier(idempotency_key, "idempotency_key")
        payload = {
            "ref": self._runtime_contract_ref,
            "kind": kind,
            "run_instance_token": run_instance_token,
            "spec_fingerprint": spec_fingerprint,
            "score_receipt_fingerprint": score_receipt_fingerprint,
            "winner_origin": winner_origin,
            "weights": list(weights),
            "idempotency_key": idempotency_key,
        }
        for name, value in payload.items():
            if name == "ref":
                continue
            object.__setattr__(self, name, tuple(value) if isinstance(value, list) else value)
        object.__setattr__(self, "decision_fingerprint", _fingerprint(payload))


@dataclass(frozen=True, init=False)
class ResidentBranchCommitReceipt:
    """Receipt for one authority-mediated GPU resident decision."""

    status: str
    decision_fingerprint: str
    idempotency_key: str
    result_manifest_fingerprint: str
    pool_layout_fingerprint: str
    committed_linear_refs: tuple[int, ...]
    generations: tuple[int, ...]
    versions_before: tuple[int, ...]
    versions_after: tuple[int, ...]
    receipt_fingerprint: str
    _runtime_contract_ref: ClassVar[str] = "arti/resident-branch-commit@1"

    def __init__(
        self,
        *,
        status: str,
        decision_fingerprint: str,
        idempotency_key: str,
        result_manifest_fingerprint: str,
        pool_layout_fingerprint: str,
        committed_linear_refs: tuple[int, ...],
        generations: tuple[int, ...],
        versions_before: tuple[int, ...],
        versions_after: tuple[int, ...],
        _factory_token: object,
    ) -> None:
        if _factory_token is not _RESIDENT_COMMIT_FACTORY_TOKEN:
            raise TensorTransactionContractError(
                "ResidentBranchCommitReceipt must come from ResidentBranchRun.commit"
            )
        if status not in {"committed", "discarded"}:
            raise TensorTransactionContractError("invalid resident commit status")
        for value in (
            decision_fingerprint,
            result_manifest_fingerprint,
            pool_layout_fingerprint,
        ):
            _require_sha256(value, "resident commit fingerprint")
        _require_identifier(idempotency_key, "idempotency_key")
        width = len(committed_linear_refs)
        if any(
            len(values) != width
            for values in (generations, versions_before, versions_after)
        ):
            raise TensorTransactionContractError(
                "resident commit vectors must have one item per committed ref"
            )
        if status == "committed" and any(
            after != before + 1
            for before, after in zip(versions_before, versions_after, strict=True)
        ):
            raise TensorTransactionContractError(
                "resident commit must increment each target version exactly once"
            )
        if status == "discarded" and width:
            raise TensorTransactionContractError(
                "discarded resident decision must not contain committed refs"
            )
        payload = {
            "ref": self._runtime_contract_ref,
            "status": status,
            "decision_fingerprint": decision_fingerprint,
            "idempotency_key": idempotency_key,
            "result_manifest_fingerprint": result_manifest_fingerprint,
            "pool_layout_fingerprint": pool_layout_fingerprint,
            "committed_linear_refs": list(committed_linear_refs),
            "generations": list(generations),
            "versions_before": list(versions_before),
            "versions_after": list(versions_after),
        }
        for name, value in payload.items():
            if name == "ref":
                continue
            object.__setattr__(self, name, tuple(value) if isinstance(value, list) else value)
        object.__setattr__(self, "receipt_fingerprint", _fingerprint(payload))


class _ResidentSelectionOperation(nn.Module):
    def __init__(self, value: Tensor) -> None:
        super().__init__()
        self._value = value

    def forward(
        self,
        _input: Tensor,
        _read_mask: Tensor,
        _write_mask: Tensor,
        _intervened: Tensor,
    ) -> Tensor:
        return self._value


class ResidentBranchRun:
    """Thin authority bridge from one CUDA K-way result to HotPagePool."""

    _runtime_contract_ref: ClassVar[str] = "arti/resident-branch-run@1"

    def __init__(
        self,
        result: object,
        executor: BatchedRefineExecutor,
        spec: BranchBatchSpecV2,
        future: Tensor,
        pool: object,
    ) -> None:
        from .batched_refine import BatchedRefineResult
        from .gpu_resident import BoundHotPagePool

        if not isinstance(result, BatchedRefineResult):
            raise TypeError("result must be BatchedRefineResult")
        if not isinstance(executor, BatchedRefineExecutor):
            raise TypeError("executor must be BatchedRefineExecutor")
        if not isinstance(spec, BranchBatchSpecV2):
            raise TypeError("spec must be BranchBatchSpecV2")
        if not isinstance(pool, BoundHotPagePool):
            raise TypeError("pool must be BoundHotPagePool")
        if executor.identity_mode != "resident_identity":
            raise TensorTransactionContractError(
                "resident branch run requires BatchedRefineExecutor.from_resident_result"
            )
        executor.assert_matches(result)
        result.assert_unchanged()
        manifest = executor.candidate_manifest_fingerprint
        if manifest != spec.candidate_manifest_fingerprint:
            raise TensorTransactionContractError(
                "resident result does not match branch authority spec"
            )
        if (
            spec.scorer_ref != FROZEN_MSE_SCORER_REF
            or spec.scorer_config_fingerprint
            != frozen_mse_scorer_config_fingerprint()
        ):
            raise TensorTransactionContractError(
                "resident branch v1 requires the frozen MSE scorer"
            )
        if result.value.device.type != "cuda" or result.value.requires_grad:
            raise TensorTransactionContractError(
                "resident branch result must be detached CUDA storage"
            )
        expected_shape = (
            pool.bucket.batch_size,
            pool.bucket.workset_slots,
            pool.bucket.feature_dim,
        )
        if (
            not isinstance(future, Tensor)
            or future.requires_grad
            or future.device.type != "cpu"
            or not future.is_contiguous()
            or future.dtype != result.value.dtype
            or tuple(future.shape) != expected_shape
        ):
            raise TensorTransactionContractError(
                "resident future must be detached contiguous CPU authority storage "
                "matching the CUDA branch output page"
            )
        if (
            result.value.shape[0],
            result.value.shape[2],
            result.value.shape[3],
        ) != expected_shape:
            raise TensorTransactionContractError(
                "resident pool shape must match [B,N,D] branch output"
            )
        if not bool(torch.isfinite(future).all()):
            raise TensorTransactionContractError(
                "resident future must contain only finite values"
            )
        branch_mask = result.candidates.branch_mask
        if not torch.equal(branch_mask, branch_mask[:1].expand_as(branch_mask)):
            raise TensorTransactionContractError(
                "resident branch v1 requires one shared active branch set"
            )
        canonical_mask = branch_mask.gather(
            1,
            torch.argsort(result.candidates.branch_origin_index, dim=1, stable=True),
        )
        active_count = int(canonical_mask[0].sum().item())
        if active_count != len(spec.branch_ids) or not bool(
            canonical_mask[:, :active_count].all()
        ):
            raise TensorTransactionContractError(
                "resident active origins do not match branch authority spec"
            )
        if bool(canonical_mask[:, active_count:].any()):
            raise TensorTransactionContractError(
                "resident active origins must form the canonical prefix"
            )
        self.result = result
        self.executor = executor
        self.spec = spec
        self.pool = pool
        future_fingerprint = _cpu_tensor_fingerprint(future)
        if future_fingerprint != spec.future_tape_fingerprint:
            raise TensorTransactionContractError(
                "resident future does not match branch authority spec"
            )
        self._future = future.detach().to(
            device=result.value.device,
            memory_format=torch.contiguous_format,
        )
        self._future_version = self._future._version
        self._future_fingerprint = future_fingerprint
        self._manifest = manifest
        self._physical_by_origin = torch.argsort(
            result.candidates.branch_origin_index,
            dim=1,
            stable=True,
        )
        self._active_count = active_count
        with pool.authority_lock:
            self._pool_layout = pool.pointer_layout_receipt()
            self._pool_layout_fingerprint = _fingerprint(self._pool_layout.__dict__)
            self._generation = pool.pool.generation.detach().clone()
            self._version = pool.pool.version.detach().clone()
        with _RESIDENT_RUN_COUNTER_LOCK:
            self._run_instance_token = next(_RESIDENT_RUN_COUNTER)
        self._decision: ResidentBranchDecision | None = None
        self._receipt: ResidentBranchCommitReceipt | None = None
        self._lock = RLock()

    def _assert_unchanged(self) -> None:
        self.result.assert_unchanged()
        if self._future._version != self._future_version:
            raise TensorTransactionContractError("resident score target changed")
        if self.pool.pointer_layout_receipt() != self._pool_layout:
            raise TensorTransactionContractError("resident pool pointer layout changed")
        if not torch.equal(self.pool.pool.generation, self._generation):
            raise TensorTransactionContractError("resident pool generation changed")
        if not torch.equal(self.pool.pool.version, self._version):
            raise TensorTransactionContractError("resident pool version changed")

    def _canonical_value(self) -> Tensor:
        index = self._physical_by_origin[:, : self._active_count]
        index = index[:, :, None, None].expand(
            -1,
            -1,
            self.result.value.shape[2],
            self.result.value.shape[3],
        )
        return self.result.value.gather(1, index)

    def _winner_value(self, origin: int) -> Tensor:
        physical = self._physical_by_origin[:, origin]
        index = physical[:, None, None, None].expand(
            -1,
            1,
            self.result.value.shape[2],
            self.result.value.shape[3],
        )
        return self.result.value.gather(1, index).squeeze(1)

    def score(self) -> ResidentBranchScoreReceipt:
        with self._lock:
            if self._decision is not None:
                raise TensorTransactionContractError("resident run is already decided")
            self._assert_unchanged()
            with torch.no_grad():
                scores = (
                    (self._canonical_value() - self._future.unsqueeze(1))
                    .float()
                    .square()
                    .mean(dim=(0, 2, 3))
                )
            host_scores = tuple(float(value) for value in scores.to("cpu").tolist())
            return ResidentBranchScoreReceipt(
                spec_fingerprint=self.spec.fingerprint,
                run_instance_token=self._run_instance_token,
                result_manifest_fingerprint=self._manifest,
                branch_ids=self.spec.branch_ids,
                future_fingerprint=self._future_fingerprint,
                scorer_ref=self.spec.scorer_ref,
                scorer_config_fingerprint=self.spec.scorer_config_fingerprint,
                tie_policy=self.spec.tie_policy,
                scores=host_scores,
                _factory_token=_RESIDENT_SCORE_FACTORY_TOKEN,
            )

    def decide(
        self,
        score: ResidentBranchScoreReceipt,
        *,
        idempotency_key: str,
    ) -> ResidentBranchDecision:
        with self._lock:
            if not isinstance(score, ResidentBranchScoreReceipt):
                raise TensorTransactionContractError(
                    "score must be ResidentBranchScoreReceipt"
                )
            if (
                score.run_instance_token != self._run_instance_token
                or score.spec_fingerprint != self.spec.fingerprint
                or score.result_manifest_fingerprint != self._manifest
                or score.branch_ids != self.spec.branch_ids
                or score.future_fingerprint != self._future_fingerprint
                or score.scorer_ref != self.spec.scorer_ref
                or score.scorer_config_fingerprint
                != self.spec.scorer_config_fingerprint
                or score.tie_policy != self.spec.tie_policy
            ):
                raise TensorTransactionContractError(
                    "resident score does not bind this run"
                )
            winner = min(
                range(len(score.scores)),
                key=lambda index: (score.scores[index], index),
            )
            return self._bind_decision(
                kind="winner",
                score_receipt_fingerprint=score.receipt_fingerprint,
                winner_origin=winner,
                weights=(),
                idempotency_key=idempotency_key,
            )

    def mix(
        self,
        weights: Sequence[float],
        *,
        idempotency_key: str,
    ) -> ResidentBranchDecision:
        normalized = tuple(float(value) for value in weights)
        if (
            len(normalized) != self._active_count
            or any(not math.isfinite(value) or value < 0 for value in normalized)
            or not math.isclose(sum(normalized), 1.0, rel_tol=1e-6, abs_tol=1e-6)
        ):
            raise TensorTransactionContractError(
                "resident mix weights must be finite, non-negative, and sum to one"
            )
        with self._lock:
            return self._bind_decision(
                kind="mixture",
                score_receipt_fingerprint=None,
                winner_origin=None,
                weights=normalized,
                idempotency_key=idempotency_key,
            )

    def discard(self, *, idempotency_key: str) -> ResidentBranchDecision:
        with self._lock:
            return self._bind_decision(
                kind="discard",
                score_receipt_fingerprint=None,
                winner_origin=None,
                weights=(),
                idempotency_key=idempotency_key,
            )

    def _bind_decision(
        self,
        *,
        kind: str,
        score_receipt_fingerprint: str | None,
        winner_origin: int | None,
        weights: tuple[float, ...],
        idempotency_key: str,
    ) -> ResidentBranchDecision:
        decision = ResidentBranchDecision(
            kind=kind,
            run_instance_token=self._run_instance_token,
            spec_fingerprint=self.spec.fingerprint,
            score_receipt_fingerprint=score_receipt_fingerprint,
            winner_origin=winner_origin,
            weights=weights,
            idempotency_key=idempotency_key,
            _factory_token=_RESIDENT_DECISION_FACTORY_TOKEN,
        )
        if self._decision is not None:
            if self._decision.decision_fingerprint == decision.decision_fingerprint:
                return self._decision
            raise TensorTransactionContractError(
                "resident run already has another decision"
            )
        self._assert_unchanged()
        self._decision = decision
        return decision

    def commit(
        self,
        decision: ResidentBranchDecision,
    ) -> ResidentBranchCommitReceipt:
        with self._lock:
            if decision is not self._decision:
                raise TensorTransactionContractError(
                    "decision does not belong to this resident run"
                )
            if self._receipt is not None:
                return self._receipt
            with self.pool.authority_lock:
                self._assert_unchanged()
                if decision.kind == "discard":
                    receipt = ResidentBranchCommitReceipt(
                        status="discarded",
                        decision_fingerprint=decision.decision_fingerprint,
                        idempotency_key=decision.idempotency_key,
                        result_manifest_fingerprint=self._manifest,
                        pool_layout_fingerprint=self._pool_layout_fingerprint,
                        committed_linear_refs=(),
                        generations=(),
                        versions_before=(),
                        versions_after=(),
                        _factory_token=_RESIDENT_COMMIT_FACTORY_TOKEN,
                    )
                    self._receipt = receipt
                    return receipt
                if decision.kind == "winner":
                    assert decision.winner_origin is not None
                    selected = self._winner_value(decision.winner_origin)
                else:
                    canonical = self._canonical_value()
                    weights = torch.tensor(
                        decision.weights,
                        device=canonical.device,
                        dtype=torch.float32,
                    ).view(1, -1, 1, 1)
                    selected = (canonical.float() * weights).sum(dim=1).to(
                        canonical.dtype
                    )
                if not bool(torch.isfinite(selected).all()):
                    raise TensorTransactionContractError(
                        "resident authorized value must be finite"
                    )
                flat_version = self.pool.pool.version.reshape(-1)
                flat_generation = self.pool.pool.generation.reshape(-1)
                refs = self.pool.commit_linear_ref
                before = flat_version.index_select(0, refs).clone()
                generations = flat_generation.index_select(0, refs).clone()
                self.pool.eager_step(_ResidentSelectionOperation(selected), commit=True)
                after = flat_version.index_select(0, refs).clone()
                receipt = ResidentBranchCommitReceipt(
                    status="committed",
                    decision_fingerprint=decision.decision_fingerprint,
                    idempotency_key=decision.idempotency_key,
                    result_manifest_fingerprint=self._manifest,
                    pool_layout_fingerprint=self._pool_layout_fingerprint,
                    committed_linear_refs=tuple(
                        int(value) for value in refs.to("cpu").tolist()
                    ),
                    generations=tuple(
                        int(value) for value in generations.to("cpu").tolist()
                    ),
                    versions_before=tuple(
                        int(value) for value in before.to("cpu").tolist()
                    ),
                    versions_after=tuple(
                        int(value) for value in after.to("cpu").tolist()
                    ),
                    _factory_token=_RESIDENT_COMMIT_FACTORY_TOKEN,
                )
                self._receipt = receipt
                return receipt


def bind_resident_branch_run(
    result: object,
    executor: BatchedRefineExecutor,
    spec: BranchBatchSpecV2,
    future: Tensor,
    pool: object,
) -> ResidentBranchRun:
    """Bind one CUDA Batched Refine result to resident host authority."""

    return ResidentBranchRun(result, executor, spec, future, pool)


class BranchBatchHarness:
    """Host-only arbitrary-K COW authority for completed branch trajectories."""

    _runtime_contract_ref: ClassVar[str] = "arti/branch-batch-harness@2"

    def __init__(
        self,
        runtime: VolatileTensorRuntime,
        snapshot: TensorSnapshot,
        spec: BranchBatchSpecV2,
    ) -> None:
        if not isinstance(runtime, VolatileTensorRuntime):
            raise TensorTransactionContractError("runtime must be VolatileTensorRuntime")
        if not isinstance(spec, BranchBatchSpecV2):
            raise TensorTransactionContractError("spec must be BranchBatchSpecV2")
        runtime._resolve_snapshot(snapshot)
        if (
            spec.parent_store_instance_id != snapshot.store_instance_id
            or spec.parent_world_id != snapshot.world_id
            or spec.parent_root_id != snapshot.root_id
            or spec.parent_epoch != snapshot.epoch
            or spec.parent_root_fingerprint != snapshot.root_fingerprint
        ):
            raise TensorTransactionContractError("BranchBatchSpec@2 parent does not match snapshot")
        self._runtime = runtime
        self._snapshot = snapshot
        self.spec = spec
        self._transactions = {
            branch_id: runtime.begin(
                snapshot,
                transaction_id="branch-" + hashlib.sha256(
                    f"{spec.run_id}:{branch_id}".encode("ascii")
                ).hexdigest()[:32],
                branch_id=branch_id,
            )
            for branch_id in spec.branch_ids
        }
        self._proposals: dict[str, BatchedRefineOverlayProposal] = {}
        self._receipt: BranchRunReceipt | None = None
        self._decision_request_fingerprint: str | None = None

    def propose(self, proposal: BatchedRefineOverlayProposal) -> None:
        if self._receipt is not None:
            raise TensorTransactionContractError("branch run is already closed")
        if not isinstance(proposal, BatchedRefineOverlayProposal):
            raise TensorTransactionContractError("proposal must be a BatchedRefineOverlayProposal")
        if proposal.branch_id not in self._transactions:
            raise TensorTransactionContractError("proposal branch is not part of this run")
        if proposal.spec_fingerprint != self.spec.fingerprint:
            raise TensorTransactionContractError("proposal does not bind this branch spec")
        if proposal.branch_id in self._proposals:
            raise TensorTransactionContractError("branch already has a proposal")
        budget = self.spec.budgets[self.spec.branch_ids.index(proposal.branch_id)]
        if not budget.min_steps <= proposal.work.actual_steps <= budget.max_steps:
            raise TensorTransactionContractError("branch actual_steps violate its budget")
        key = proposal.proposal.binding.tensor_ref.key
        if key not in self.spec.allowed_write_keys:
            raise TensorTransactionContractError("proposal is outside the coordinator write authority")
        binding = proposal.proposal.binding
        if (
            binding.root_id != self.spec.parent_root_id
            or binding.root_epoch != self.spec.parent_epoch
            or binding.root_fingerprint != self.spec.parent_root_fingerprint
        ):
            raise TensorTransactionContractError("branch proposal is not rooted at the parent snapshot")
        try:
            stage_external_proposal(self._transactions[proposal.branch_id], proposal.proposal)
        except Exception:
            self.discard(idempotency_key=f"abort-{self.spec.run_id}")
            raise
        self._proposals[proposal.branch_id] = proposal

    def _validate_ready(self) -> None:
        if tuple(self._proposals) != self.spec.branch_ids:
            raise TensorTransactionContractError("all K branches require ordered proposals")
        if self.spec.require_matched_work:
            if len({item.work.matched_key for item in self._proposals.values()}) != 1:
                raise TensorTransactionContractError("branch work is not matched")
        written = {
            item.proposal.binding.tensor_ref.key for item in self._proposals.values()
        }
        if written != set(self.spec.allowed_write_keys):
            raise TensorTransactionContractError("branch proposals do not cover the allowed write set")

    def decide(
        self,
        score: BranchScoreBatchReceipt,
        *,
        idempotency_key: str,
    ) -> BranchRunReceipt:
        decision_key = _require_identifier(idempotency_key, "idempotency_key")
        if not isinstance(score, BranchScoreBatchReceipt):
            raise TensorTransactionContractError("score must be BranchScoreBatchReceipt")
        self._validate_ready()
        actual_overlays = tuple(self._proposals[name].fingerprint for name in self.spec.branch_ids)
        request_fp = _fingerprint(
            {
                "spec": self.spec.fingerprint,
                "overlays": actual_overlays,
                "score": score.receipt_fingerprint,
                "decision_key": decision_key,
            }
        )
        if self._receipt is not None:
            if self._decision_request_fingerprint == request_fp:
                return self._receipt
            raise TensorTransactionContractError("branch run already closed with another decision")
        if (
            score.spec_fingerprint != self.spec.fingerprint
            or score.branch_ids != self.spec.branch_ids
            or score.overlay_fingerprints != actual_overlays
            or score.future_fingerprint != self.spec.future_tape_fingerprint
            or score.scorer_ref != self.spec.scorer_ref
            or score.scorer_config_fingerprint != self.spec.scorer_config_fingerprint
            or score.tie_policy != self.spec.tie_policy
        ):
            raise TensorTransactionContractError("score receipt does not bind this K-branch run")
        winner_index = min(range(len(score.scores)), key=lambda index: (score.scores[index], index))
        winner = self.spec.branch_ids[winner_index]
        commit = self._transactions[winner].commit(idempotency_key=decision_key)
        rollbacks = tuple(
            self._transactions[name].rollback()
            for name in self.spec.branch_ids
            if name != winner
        )
        status = BranchRunStatus.COMMITTED if isinstance(commit, CommitReceipt) else BranchRunStatus.CONFLICTED
        proposal_fingerprints = tuple(
            (name, self._proposals[name].fingerprint) for name in self.spec.branch_ids
        )
        work_receipts = tuple(
            (name, self._proposals[name].work) for name in self.spec.branch_ids
        )
        self._receipt = BranchRunReceipt(
            spec_fingerprint=self.spec.fingerprint,
            status=status,
            winner_branch_id=winner,
            decision_idempotency_key=decision_key,
            decision_request_fingerprint=request_fp,
            proposal_fingerprints=proposal_fingerprints,
            work_receipts=work_receipts,
            commit_receipt=commit,
            rollback_receipts=rollbacks,
            decision_kind="winner",
            _factory_token=_BRANCH_RUN_RECEIPT_FACTORY_TOKEN,
        )
        self._decision_request_fingerprint = request_fp
        return self._receipt

    def mix(
        self,
        weights: Sequence[float],
        *,
        idempotency_key: str,
    ) -> BranchRunReceipt:
        """Publish one host-authorized convex mixture of all K private states."""

        decision_key = _require_identifier(idempotency_key, "idempotency_key")
        self._validate_ready()
        normalized = tuple(float(value) for value in weights)
        if (
            len(normalized) != len(self.spec.branch_ids)
            or any(not math.isfinite(value) or value < 0 for value in normalized)
            or not math.isclose(sum(normalized), 1.0, rel_tol=1e-6, abs_tol=1e-6)
        ):
            raise TensorTransactionContractError(
                "mix weights must be finite, non-negative, and sum to one"
            )
        proposal_fingerprints = tuple(
            (name, self._proposals[name].fingerprint) for name in self.spec.branch_ids
        )
        request_fp = _fingerprint(
            {
                "spec": self.spec.fingerprint,
                "proposals": proposal_fingerprints,
                "weights": normalized,
                "decision_key": decision_key,
                "kind": "mixture",
            }
        )
        if self._receipt is not None:
            if self._decision_request_fingerprint == request_fp:
                return self._receipt
            raise TensorTransactionContractError("branch run already closed with another decision")
        first = self._proposals[self.spec.branch_ids[0]].proposal
        mixed = torch.zeros_like(first.value)
        for branch_id, weight in zip(self.spec.branch_ids, normalized, strict=True):
            mixed.add_(self._proposals[branch_id].proposal.value, alpha=weight)
        mixture = ExternalTensorProposal(
            first.binding,
            mixed,
            producer_ref=first.producer_ref,
            producer_config_fingerprint=first.producer_config_fingerprint,
            producer_state_fingerprint=first.producer_state_fingerprint,
        )
        transaction_digest = hashlib.sha256(
            f"{self.spec.run_id}:mixture".encode("ascii")
        ).hexdigest()[:32]
        transaction = self._runtime.begin(
            self._snapshot,
            transaction_id=f"branch-{transaction_digest}",
            branch_id="mixture",
        )
        stage_external_proposal(transaction, mixture)
        commit = transaction.commit(idempotency_key=decision_key)
        rollbacks = tuple(
            self._transactions[name].rollback() for name in self.spec.branch_ids
        )
        status = (
            BranchRunStatus.COMMITTED
            if isinstance(commit, CommitReceipt)
            else BranchRunStatus.CONFLICTED
        )
        self._receipt = BranchRunReceipt(
            spec_fingerprint=self.spec.fingerprint,
            status=status,
            winner_branch_id=None,
            decision_idempotency_key=decision_key,
            decision_request_fingerprint=request_fp,
            proposal_fingerprints=proposal_fingerprints,
            work_receipts=tuple(
                (name, self._proposals[name].work) for name in self.spec.branch_ids
            ),
            commit_receipt=commit,
            rollback_receipts=rollbacks,
            decision_kind="mixture",
            _factory_token=_BRANCH_RUN_RECEIPT_FACTORY_TOKEN,
        )
        self._decision_request_fingerprint = request_fp
        return self._receipt

    def discard(self, *, idempotency_key: str) -> BranchRunReceipt:
        decision_key = _require_identifier(idempotency_key, "idempotency_key")
        if self._receipt is not None:
            if (
                self._receipt.status is BranchRunStatus.DISCARDED
                and self._receipt.decision_idempotency_key == decision_key
            ):
                return self._receipt
            raise TensorTransactionContractError("branch run already closed")
        rollbacks = tuple(self._transactions[name].rollback() for name in self.spec.branch_ids)
        proposal_fingerprints = tuple(
            (name, self._proposals[name].fingerprint)
            for name in self.spec.branch_ids
            if name in self._proposals
        )
        decision_request_fingerprint = _fingerprint(
            {
                "spec": self.spec.fingerprint,
                "decision_key": decision_key,
                "proposals": proposal_fingerprints,
                "kind": "discard",
            }
        )
        self._receipt = BranchRunReceipt(
            spec_fingerprint=self.spec.fingerprint,
            status=BranchRunStatus.DISCARDED,
            winner_branch_id=None,
            decision_idempotency_key=decision_key,
            decision_request_fingerprint=decision_request_fingerprint,
            proposal_fingerprints=proposal_fingerprints,
            work_receipts=tuple(
                (name, self._proposals[name].work)
                for name in self.spec.branch_ids
                if name in self._proposals
            ),
            commit_receipt=None,
            rollback_receipts=rollbacks,
            decision_kind="discard",
            _factory_token=_BRANCH_RUN_RECEIPT_FACTORY_TOKEN,
        )
        return self._receipt

    def abort(self, *, idempotency_key: str) -> BranchRunReceipt:
        """Discard every still-open branch without requiring complete or matched work."""

        decision_key = _require_identifier(idempotency_key, "idempotency_key")
        if self._receipt is not None:
            if (
                self._receipt.status is BranchRunStatus.DISCARDED
                and self._receipt.decision_idempotency_key == decision_key
            ):
                return self._receipt
            raise TensorTransactionContractError("branch run already closed with another decision")
        rollbacks = tuple(
            self._transactions[branch_id].rollback()
            for branch_id in self.spec.branch_ids
        )
        proposal_fingerprints = tuple(
            (branch_id, self._proposals[branch_id].fingerprint)
            for branch_id in self.spec.branch_ids
            if branch_id in self._proposals
        )
        work_receipts = tuple(
            (branch_id, self._proposals[branch_id].work)
            for branch_id in self.spec.branch_ids
            if branch_id in self._proposals
        )
        decision_request_fingerprint = _fingerprint(
            {
                "spec": self.spec.fingerprint,
                "decision_key": decision_key,
                "proposals": proposal_fingerprints,
                "kind": "abort",
            }
        )
        self._receipt = BranchRunReceipt(
            spec_fingerprint=self.spec.fingerprint,
            status=BranchRunStatus.DISCARDED,
            winner_branch_id=None,
            decision_idempotency_key=decision_key,
            decision_request_fingerprint=decision_request_fingerprint,
            proposal_fingerprints=proposal_fingerprints,
            work_receipts=work_receipts,
            commit_receipt=None,
            rollback_receipts=rollbacks,
            decision_kind="discard",
            _factory_token=_BRANCH_RUN_RECEIPT_FACTORY_TOKEN,
        )
        return self._receipt


def assert_matched_branch_work(
    left: BranchRunReceipt,
    right: BranchRunReceipt,
) -> None:
    """Fail closed unless two run receipts declare the same branch work."""

    left_work: Mapping[str, BranchWorkReceipt] = dict(left.work_receipts)
    right_work: Mapping[str, BranchWorkReceipt] = dict(right.work_receipts)
    if set(left_work) != set(right_work):
        raise TensorTransactionContractError("branch sets differ")
    for branch_id in left_work:
        if left_work[branch_id].matched_key != right_work[branch_id].matched_key:
            raise TensorTransactionContractError("branch work receipts are not matched")


__all__ = [
    "BatchedRefineOverlayProposal",
    "BranchBatchHarness",
    "BranchBatchSpec",
    "BranchBatchSpecV2",
    "BranchScoreBatchReceipt",
    "BranchBudget",
    "BranchRunReceipt",
    "BranchRunStatus",
    "BranchWorkReceipt",
    "ExecutionContextReceipt",
    "BatchedRefineExecutor",
    "K2BranchHarness",
    "OverlayProposal",
    "ResidentBranchCommitReceipt",
    "ResidentBranchDecision",
    "ResidentBranchRun",
    "ResidentBranchScoreReceipt",
    "assert_matched_branch_work",
    "batched_refine_manifest_fingerprint",
    "bind_resident_branch_run",
    "score_batched_refine_proposals",
    "stage_batched_refine_result",
]
