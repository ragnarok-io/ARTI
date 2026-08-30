"""Task-loss-curve training for neural Refine exit control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Literal

import torch
from torch import Tensor

from .refine_exit import RefineExitControl
from .refine_training import (
    RefineRollout,
    RefineStepTrainingResult,
    RefineTrainingContractError,
    _assert_fixed_query,
    _is_sha256,
)


RefineExitCurveScope = Literal["token", "branch"]


def _require_finite_non_negative(value: float, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a finite non-negative number")
    value = float(value)
    if not torch.isfinite(torch.tensor(value)) or value < 0:
        raise ValueError(f"{name} must be a finite non-negative number")
    return value


@dataclass(frozen=True)
class RefineExitCurve:
    """Detached full-depth states and real task loss for exit training."""

    _component_reference: ClassVar[str] = "arti/refine-exit-curve@1"

    post_state: Tensor
    task_loss: Tensor
    mask: Tensor
    sample_id: Tensor
    branch_id: Tensor
    sample_count: int
    breadth: int
    scope: RefineExitCurveScope
    source_ref: str
    source_structure_fingerprint: str
    source_behavior_fingerprint: str
    source_execution_fingerprint: str
    query_fingerprint: str
    bank_fingerprint: str
    formula_fingerprint: str
    snapshot_fingerprint: str
    snapshot_generation: int = 0
    sampling_policy: str = "fixed_depth_no_model_exit"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.post_state, Tensor)
            or self.post_state.ndim != 4
            or not self.post_state.is_floating_point()
        ):
            raise TypeError("post_state must be floating point [T, R, N, D]")
        trajectories, depth, tokens, _ = self.post_state.shape
        if trajectories <= 0 or depth <= 0 or tokens <= 0:
            raise ValueError("post_state dimensions T, R, and N must be positive")
        if self.post_state.requires_grad or self.post_state.grad_fn is not None:
            raise RefineTrainingContractError("exit-curve states must be detached")
        if (
            not isinstance(self.mask, Tensor)
            or self.mask.dtype != torch.bool
            or self.mask.shape != (trajectories, depth, tokens)
            or self.mask.device != self.post_state.device
        ):
            raise TypeError("mask must be bool [T, R, N] on the state device")
        if not torch.equal(self.mask, self.mask[:, :1].expand_as(self.mask)):
            raise RefineTrainingContractError(
                "full-depth exit curves require a depth-invariant token mask"
            )
        if not bool(self.mask[:, 0].any(dim=-1).all()):
            raise RefineTrainingContractError(
                "every exit-curve trajectory must contain a valid token"
            )
        if not bool(torch.isfinite(self.post_state[self.mask]).all()):
            raise RefineTrainingContractError(
                "exit-curve states must be finite at every live depth"
            )
        if self.scope not in {"token", "branch"}:
            raise ValueError("scope must be 'token' or 'branch'")
        expected_loss_shape = (
            (trajectories, depth, tokens)
            if self.scope == "token"
            else (trajectories, depth)
        )
        if (
            not isinstance(self.task_loss, Tensor)
            or not self.task_loss.is_floating_point()
            or self.task_loss.shape != expected_loss_shape
            or self.task_loss.device != self.post_state.device
        ):
            raise TypeError(
                f"{self.scope} task_loss must be floating point {expected_loss_shape}"
            )
        if self.task_loss.requires_grad or self.task_loss.grad_fn is not None:
            raise RefineTrainingContractError("exit-curve task loss must be detached")
        live_loss = self.task_loss[self.mask] if self.scope == "token" else self.task_loss
        if not bool(torch.isfinite(live_loss).all()):
            raise RefineTrainingContractError(
                "exit-curve task loss must be finite at every live depth"
            )
        for value, name in (
            (self.sample_id, "sample_id"),
            (self.branch_id, "branch_id"),
        ):
            if (
                not isinstance(value, Tensor)
                or value.dtype != torch.int64
                or value.shape != (trajectories,)
                or value.device != self.post_state.device
            ):
                raise TypeError(f"{name} must be int64 [T] on the state device")
        if isinstance(self.sample_count, bool) or self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if isinstance(self.breadth, bool) or self.breadth <= 0:
            raise ValueError("breadth must be positive")
        if trajectories != self.sample_count * self.breadth:
            raise RefineTrainingContractError(
                "trajectory count must equal sample_count * breadth"
            )
        trajectory = torch.arange(trajectories, device=self.post_state.device)
        if not torch.equal(self.sample_id, trajectory // self.breadth):
            raise RefineTrainingContractError("sample lineage is not canonical")
        if not torch.equal(self.branch_id, trajectory % self.breadth):
            raise RefineTrainingContractError("branch lineage is not canonical")
        for value, name in (
            (self.source_structure_fingerprint, "source_structure_fingerprint"),
            (self.source_behavior_fingerprint, "source_behavior_fingerprint"),
            (self.source_execution_fingerprint, "source_execution_fingerprint"),
            (self.query_fingerprint, "query_fingerprint"),
            (self.bank_fingerprint, "bank_fingerprint"),
            (self.formula_fingerprint, "formula_fingerprint"),
            (self.snapshot_fingerprint, "snapshot_fingerprint"),
        ):
            if not _is_sha256(value):
                raise ValueError(f"{name} must be a SHA-256 digest")
        if isinstance(self.snapshot_generation, bool) or self.snapshot_generation < 0:
            raise ValueError("snapshot_generation must be non-negative")
        if self.sampling_policy != "fixed_depth_no_model_exit":
            raise ValueError(
                "refine-exit-curve@1 requires fixed_depth_no_model_exit sampling"
            )

    @property
    def depth(self) -> int:
        return self.post_state.shape[1]

    @property
    def trajectory_count(self) -> int:
        return self.post_state.shape[0]


@dataclass(frozen=True)
class RefineExitTrainingLoss:
    """Quality-constrained hazard loss and curve diagnostics."""

    total: Tensor
    expected_task: Tensor
    full_depth_task: Tensor
    expected_logical_depth: Tensor
    quality_violation: Tensor
    stop_probability: Tensor
    terminal_probability: Tensor
    valid_trajectories: Tensor


@dataclass(frozen=True)
class RefineExitTraining:
    """Train a Refine exit controller from the model's own task-loss curve."""

    _component_reference: ClassVar[str] = "arti/refine-exit-training@1"

    temperature: float = 1.0
    compute_weight: float = 0.0
    quality_tolerance: float = 0.0
    quality_weight: float = 1.0

    def __post_init__(self) -> None:
        temperature = _require_finite_non_negative(
            self.temperature,
            name="temperature",
        )
        if temperature == 0:
            raise ValueError("temperature must be positive")
        _require_finite_non_negative(self.compute_weight, name="compute_weight")
        _require_finite_non_negative(
            self.quality_tolerance,
            name="quality_tolerance",
        )
        _require_finite_non_negative(self.quality_weight, name="quality_weight")

    @staticmethod
    def assert_optimizer_contract(
        recall: object,
        control: RefineExitControl,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        """Require controller-only optimization and an unchanged fixed Query."""

        _assert_fixed_query(recall)
        if not isinstance(control, RefineExitControl):
            raise TypeError("control must be a RefineExitControl")
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch.optim.Optimizer")
        trainable = {id(parameter) for parameter in control.parameters() if parameter.requires_grad}
        if not trainable:
            raise RefineTrainingContractError(
                "exit control must expose at least one trainable parameter"
            )
        owned = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        if owned != trainable:
            raise RefineTrainingContractError(
                "exit training optimizer must own exactly the trainable control parameters"
            )
        recall_parameters = {id(parameter) for parameter in recall.parameters()}
        if owned & recall_parameters:
            raise RefineTrainingContractError(
                "exit training optimizer must not own Recall parameters"
            )

    @staticmethod
    def build_curve(
        rollout: RefineRollout,
        result: RefineStepTrainingResult,
        task_loss: Tensor,
        *,
        scope: RefineExitCurveScope = "token",
    ) -> RefineExitCurve:
        """Freeze a complete no-exit full-depth replay and its real task loss."""

        if not isinstance(rollout, RefineRollout):
            raise TypeError("rollout must be a RefineRollout")
        if not isinstance(result, RefineStepTrainingResult):
            raise TypeError("result must be a RefineStepTrainingResult")
        if scope not in {"token", "branch"}:
            raise ValueError("scope must be 'token' or 'branch'")
        for result_value, rollout_value, name in (
            (result.trajectory_id, rollout.trajectory_id, "trajectory_id"),
            (result.sample_id, rollout.sample_id, "sample_id"),
            (result.branch_id, rollout.branch_id, "branch_id"),
            (result.step_index, rollout.step_index, "step_index"),
        ):
            if not torch.equal(result_value, rollout_value):
                raise RefineTrainingContractError(
                    f"exit-curve replay {name} differs from its rollout"
                )
        if result.trajectory_count != rollout.trajectory_count:
            raise RefineTrainingContractError(
                "exit-curve replay trajectory count differs from its rollout"
            )
        if result.sample_count * rollout.breadth != rollout.trajectory_count:
            raise RefineTrainingContractError("exit-curve replay sample count is invalid")
        if not torch.equal(result.valid_token_mask, rollout.mask):
            raise RefineTrainingContractError(
                "exit training requires every live token to complete full depth"
            )
        pairs, tokens, _ = result.value.shape
        expected_loss_shape = (pairs, tokens) if scope == "token" else (pairs,)
        if (
            not isinstance(task_loss, Tensor)
            or not task_loss.is_floating_point()
            or task_loss.shape != expected_loss_shape
            or task_loss.device != result.value.device
        ):
            raise TypeError(
                f"{scope} task_loss must be floating point {expected_loss_shape}"
            )
        depth = pairs // result.trajectory_count
        packed = result.trajectory_id * depth + result.step_index
        order = torch.argsort(packed, stable=True)
        expected = torch.arange(pairs, device=packed.device, dtype=torch.int64)
        if not torch.equal(packed.index_select(0, order), expected):
            raise RefineTrainingContractError(
                "exit-curve replay does not contain one complete fixed-depth grid"
            )

        def canonical(value: Tensor) -> Tensor:
            ordered = value.index_select(0, order)
            return ordered.reshape(result.trajectory_count, depth, *value.shape[1:])

        sample_grid = canonical(result.sample_id).reshape(result.trajectory_count, depth)
        branch_grid = canonical(result.branch_id).reshape(result.trajectory_count, depth)
        return RefineExitCurve(
            post_state=canonical(result.value.detach()).clone(),
            task_loss=canonical(task_loss.detach()).clone(),
            mask=canonical(rollout.mask).clone(),
            sample_id=sample_grid[:, 0].clone(),
            branch_id=branch_grid[:, 0].clone(),
            sample_count=result.sample_count,
            breadth=rollout.breadth,
            scope=scope,
            source_ref=rollout.source_ref,
            source_structure_fingerprint=rollout.source_structure_fingerprint,
            source_behavior_fingerprint=rollout.source_behavior_fingerprint,
            source_execution_fingerprint=rollout.source_execution_fingerprint,
            query_fingerprint=rollout.query_fingerprint,
            bank_fingerprint=rollout.bank_fingerprint,
            formula_fingerprint=rollout.formula_fingerprint,
            snapshot_fingerprint=rollout.snapshot_fingerprint,
            snapshot_generation=rollout.snapshot_generation,
        )

    @staticmethod
    def _balanced_mean(value: Tensor, curve: RefineExitCurve) -> Tensor:
        if curve.scope == "token":
            token_mask = curve.mask[:, 0]
            weight = token_mask.to(dtype=value.dtype)
            per_trajectory = (
                torch.where(token_mask, value, 0).sum(dim=-1)
                / weight.sum(dim=-1).clamp_min(1)
            )
        else:
            per_trajectory = value
        per_sample = per_trajectory.reshape(curve.sample_count, curve.breadth).mean(dim=1)
        return per_sample.mean()

    def loss(
        self,
        control: RefineExitControl,
        curve: RefineExitCurve,
        *,
        min_steps: int = 1,
    ) -> RefineExitTrainingLoss:
        """Evaluate a differentiable stop hazard over a frozen depth curve."""

        if not isinstance(control, RefineExitControl):
            raise TypeError("control must be a RefineExitControl")
        if control.atom.input_kind != "logit":
            raise RefineTrainingContractError(
                "exit training requires a logit RefineExitControl"
            )
        if not isinstance(curve, RefineExitCurve):
            raise TypeError("curve must be a RefineExitCurve")
        if control.atom.scope != curve.scope:
            raise RefineTrainingContractError(
                "exit control scope must match the task-loss curve scope"
            )
        if isinstance(min_steps, bool) or not isinstance(min_steps, int):
            raise TypeError("min_steps must be an integer")
        if min_steps < 1 or min_steps > curve.depth:
            raise ValueError("min_steps must be in [1, curve.depth]")

        trajectories, depth, tokens, dim = curve.post_state.shape
        state = curve.post_state.reshape(trajectories * depth, tokens, dim)
        mask = curve.mask.reshape(trajectories * depth, tokens)
        request = control(state, mask=mask)
        if not bool(request.finite[mask].all()):
            raise RefineTrainingContractError(
                "exit control produced a non-finite score on a live state"
            )
        score = request.score.reshape(trajectories, depth, tokens)
        if curve.scope == "branch":
            weight = curve.mask.to(dtype=score.dtype)
            score = (score * weight).sum(dim=-1) / weight.sum(dim=-1).clamp_min(1)

        raw_stop = torch.sigmoid(
            (score - float(control.atom.threshold)) / float(self.temperature)
        )
        depth_index = torch.arange(1, depth + 1, device=score.device)
        view_shape = (1, depth, *([1] * (raw_stop.ndim - 2)))
        depth_index = depth_index.reshape(view_shape)
        allowed = (depth_index >= min_steps) & (depth_index < depth)
        stop_probability = torch.where(allowed, raw_stop, torch.zeros_like(raw_stop))
        stop_probability = torch.where(
            depth_index == depth,
            torch.ones_like(stop_probability),
            stop_probability,
        )
        reach_probability = torch.cat(
            (
                torch.ones_like(stop_probability[:, :1]),
                torch.cumprod(1.0 - stop_probability[:, :-1], dim=1),
            ),
            dim=1,
        )
        terminal_probability = reach_probability * stop_probability
        detached_task = curve.task_loss.detach()
        expected_task_item = (terminal_probability * detached_task).sum(dim=1)
        full_depth_item = detached_task[:, -1]
        depth_value = depth_index.to(dtype=terminal_probability.dtype)
        expected_depth_item = (terminal_probability * depth_value).sum(dim=1)
        quality_violation_item = torch.relu(
            expected_task_item
            - full_depth_item
            - float(self.quality_tolerance)
        )

        expected_task = self._balanced_mean(expected_task_item, curve)
        full_depth_task = self._balanced_mean(full_depth_item, curve)
        expected_depth = self._balanced_mean(expected_depth_item, curve)
        quality_violation = self._balanced_mean(quality_violation_item, curve)
        total = (
            expected_task
            + float(self.compute_weight) * expected_depth
            + float(self.quality_weight) * quality_violation
        )
        return RefineExitTrainingLoss(
            total=total,
            expected_task=expected_task,
            full_depth_task=full_depth_task,
            expected_logical_depth=expected_depth,
            quality_violation=quality_violation,
            stop_probability=stop_probability,
            terminal_probability=terminal_probability,
            valid_trajectories=torch.tensor(
                trajectories,
                device=curve.post_state.device,
                dtype=torch.int64,
            ),
        )


__all__ = [
    "RefineExitCurve",
    "RefineExitCurveScope",
    "RefineExitTraining",
    "RefineExitTrainingLoss",
]
