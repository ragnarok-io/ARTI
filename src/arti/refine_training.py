"""Flattened task-loss training contracts for sequential Recall refinement.

The rollout is a detached view of the canonical Recall executor. Exact-generation
replay is the default; bounded stale replay is an explicit off-policy option for
single-branch training only.
Replay always performs a fresh one-step query from each recorded hidden state;
the rollout never stores a next-hidden teacher or a reusable route plan.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
from typing import ClassVar

import torch
from torch import Tensor

from .component_registry import canonical_contract_reference
from .recall_experts import (
    _canonical_formula_reference,
    canonical_tensor_state_sha256,
    module_behavior_fingerprint,
    module_structure_fingerprint,
    module_value_sha256,
)
from .execution import AdaptiveExecutionPolicy, ExecutionStopReason, ExecutionPolicy


class RefineTrainingContractError(ValueError):
    """Raised when a flattened Refine training contract is violated."""


def _is_sha256(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _require_non_negative_int(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise TypeError(f"{name} must be a non-negative integer")
    return value


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _recall_field(recall: object):
    from .nn import Recall

    if not isinstance(recall, Recall):
        raise TypeError("recall must be an arti.nn.Recall")
    return recall.state.recall


def _role_state(recall: object, roles: frozenset[str]) -> dict[str, Tensor]:
    field = _recall_field(recall)
    parameters = dict(field.named_parameters())
    tags = field.recall_parameter_tags()
    return {
        f"parameter:{tag.parameter_name}": parameters[tag.parameter_name]
        for tag in tags
        if tag.role in roles
    }


def _query_fingerprint(recall: object) -> str:
    state = _role_state(recall, frozenset({"fixed_query", "query"}))
    if not state:
        raise RefineTrainingContractError("Recall exposes no Query parameter")
    return canonical_tensor_state_sha256(state)


def _bank_fingerprint(recall: object) -> str:
    return canonical_tensor_state_sha256(
        _role_state(recall, frozenset({"value_bank", "routing"}))
    )


def _formula_execution_identity(recall: object) -> dict[str, str]:
    """Return a persisted identity without treating local formulas as registrations."""

    field = _recall_field(recall)
    try:
        return {"kind": "registered", "ref": _canonical_formula_reference(recall.formula_id)}
    except ValueError:
        return {
            "kind": "custom",
            "structure_fingerprint": module_structure_fingerprint(field.formula),
            "behavior_fingerprint": module_behavior_fingerprint(field.formula),
            "program_fingerprint": _formula_program_fingerprint(field.formula) or "",
        }


def _formula_fingerprint(recall: object) -> str:
    field = _recall_field(recall)
    formula_state = _role_state(recall, frozenset({"formula"}))
    activation = recall.state.recall_activation
    activation_state = {
        f"activation.parameter:{name}": value
        for name, value in activation.named_parameters()
    }
    activation_state.update(
        {
            f"activation.buffer:{name}": value
            for name, value in activation.named_buffers()
        }
    )
    return _sha256_json(
        {
            "formula": _formula_execution_identity(recall),
            "factor_names": list(field.factor_names),
            "program_fingerprint": _formula_program_fingerprint(field.formula),
            "tensor_sha256": canonical_tensor_state_sha256(
                {**formula_state, **activation_state}
            ),
        }
    )


def _formula_program_fingerprint(formula: object) -> str | None:
    direct = getattr(formula, "program_fingerprint", None)
    if isinstance(direct, str):
        return direct
    program = getattr(formula, "program", None)
    nested = getattr(program, "fingerprint", None)
    return nested if isinstance(nested, str) else None


def _execution_config_fingerprint(recall: object) -> str:
    field = _recall_field(recall)
    config = recall.state.config
    return _sha256_json(
        {
            "source_ref": canonical_contract_reference(recall._component_reference),
            "formula": _formula_execution_identity(recall),
            "formula_program_fingerprint": _formula_program_fingerprint(field.formula),
            "breadth": recall.breadth,
            "breadth_mode": recall.breadth_mode,
            "breadth_aggregation": recall.breadth_aggregation,
            "routing_normalizer": recall.routing_normalizer,
            "activation": config.recall_activation,
            "dropout": config.dropout,
            "recognition_mode": field.recognition_mode,
            "routing": field.routing,
            "route_exploration": field.route_exploration,
            "key_dim": field.key_dim,
            "group_size": field.group_size,
            "group_topk": field.group_topk,
            "value_composition": field.value_composition,
            "query_contract": field.query_contract,
            "factor_names": list(field.factor_names),
            "factor_route_names": list(field.factor_route_names),
            "training_modes": {
                name: module.training for name, module in recall.named_modules()
            },
        }
    )


def _assert_deterministic_execution(recall: object) -> None:
    field = _recall_field(recall)
    dropout_types = (
        torch.nn.Dropout,
        torch.nn.Dropout1d,
        torch.nn.Dropout2d,
        torch.nn.Dropout3d,
        torch.nn.AlphaDropout,
        torch.nn.FeatureAlphaDropout,
    )
    stochastic_modules = tuple(
        name or "<root>"
        for name, module in recall.named_modules()
        if bool(getattr(module, "stochastic", False))
    )
    active_dropout = tuple(
        name or "<root>"
        for name, module in recall.named_modules()
        if isinstance(module, dropout_types)
        and module.training
        and module.p > 0
    )
    route_exploration = (
        field.training
        and field._bank_gradient_enabled
        and field.route_exploration > 0
    )
    if stochastic_modules or active_dropout or route_exploration:
        raise RefineTrainingContractError(
            "refine-step-training@1 requires deterministic Recall execution; "
            "stochastic Half, active dropout, and route exploration require a later "
            "identity-keyed RNG contract"
        )


def _assert_fixed_query(recall: object) -> None:
    field = _recall_field(recall)
    query = field.query
    if (
        field.query_mode != "fixed"
        or not bool(getattr(query, "_arti_fixed_query", False))
        or any(parameter.requires_grad for parameter in query.parameters())
    ):
        raise RefineTrainingContractError(
            "flattened Refine training requires a fixed Query outside the optimizer"
        )


def _canonical_branches(value: Tensor, origin: Tensor) -> Tensor:
    if value.shape[:2] != origin.shape:
        raise RefineTrainingContractError("branch tensor does not match branch lineage")
    order = torch.argsort(origin, dim=1, stable=True)
    gather = order.reshape(*order.shape, *([1] * (value.ndim - 2))).expand_as(value)
    return value.gather(1, gather)


def _as_sequence(value: Tensor, mask: Tensor | None) -> tuple[Tensor, Tensor, bool]:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError("hidden_state must be a floating-point Tensor")
    if value.ndim not in {2, 3}:
        raise ValueError("hidden_state must have shape [B,D] or [B,N,D]")
    was_vector = value.ndim == 2
    sequence = value.unsqueeze(1) if was_vector else value
    if mask is None:
        token_mask = torch.ones(sequence.shape[:2], device=value.device, dtype=torch.bool)
    else:
        expected = value.shape[:1] if was_vector else value.shape[:2]
        if not isinstance(mask, Tensor) or mask.dtype != torch.bool or mask.shape != expected:
            raise ValueError(f"mask must be bool with shape {tuple(expected)}")
        if mask.device != value.device:
            raise ValueError("mask must be on the hidden-state device")
        token_mask = mask.unsqueeze(1) if was_vector else mask
    return sequence, token_mask, was_vector


def _validate_fixed_depth_lineage(
    trajectory_id: Tensor,
    sample_id: Tensor,
    branch_id: Tensor,
    step_index: Tensor,
    *,
    trajectory_count: int,
    breadth: int,
) -> None:
    pairs = trajectory_id.numel()
    if trajectory_count % breadth:
        raise RefineTrainingContractError("trajectory_count must be divisible by breadth")
    if pairs == 0 or pairs % trajectory_count:
        raise RefineTrainingContractError(
            "fixed-depth rollout must contain the same number of rows per trajectory"
        )
    depth = pairs // trajectory_count
    sample_count = trajectory_count // breadth
    if bool(torch.any(sample_id < 0)) or bool(torch.any(sample_id >= sample_count)):
        raise RefineTrainingContractError("sample_id is outside the source batch")
    if bool(torch.any(branch_id < 0)) or bool(torch.any(branch_id >= breadth)):
        raise RefineTrainingContractError("branch_id is outside breadth")
    expected_trajectory = sample_id * breadth + branch_id
    if not torch.equal(trajectory_id, expected_trajectory):
        raise RefineTrainingContractError(
            "trajectory_id must identify the canonical sample/branch pair"
        )
    packed = trajectory_id * depth + step_index
    expected = torch.arange(pairs, device=packed.device, dtype=torch.int64)
    if not torch.equal(torch.sort(packed).values, expected):
        raise RefineTrainingContractError(
            "each trajectory must contain every fixed-depth step exactly once"
        )


@dataclass(frozen=True)
class RefineRollout:
    """Detached adjacent-step states from one frozen Recall snapshot."""

    _component_reference: ClassVar[str] = "arti/refine-rollout@1"

    hidden_state: Tensor
    mask: Tensor
    attempted: Tensor
    committed: Tensor
    finite: Tensor
    stop_reason: Tensor
    trajectory_id: Tensor
    sample_id: Tensor
    branch_id: Tensor
    step_index: Tensor
    trajectory_count: int
    breadth: int
    source_ref: str
    source_structure_fingerprint: str
    source_behavior_fingerprint: str
    source_execution_fingerprint: str
    query_fingerprint: str
    bank_fingerprint: str
    formula_fingerprint: str
    snapshot_fingerprint: str
    snapshot_generation: int = 0
    sampling_policy: str = "fixed_depth"

    def __post_init__(self) -> None:
        if not isinstance(self.hidden_state, Tensor) or self.hidden_state.ndim != 3:
            raise TypeError("hidden_state must be a rank-3 Tensor [P,N,D]")
        pairs, tokens, _ = self.hidden_state.shape
        for value, name in (
            (self.mask, "mask"),
            (self.attempted, "attempted"),
            (self.committed, "committed"),
            (self.finite, "finite"),
        ):
            if (
                not isinstance(value, Tensor)
                or value.dtype != torch.bool
                or value.shape != (pairs, tokens)
                or value.device != self.hidden_state.device
            ):
                raise TypeError(f"{name} must be bool [P,N] on the rollout device")
        if (
            not isinstance(self.stop_reason, Tensor)
            or self.stop_reason.dtype != torch.int64
            or self.stop_reason.shape != (pairs, tokens)
            or self.stop_reason.device != self.hidden_state.device
        ):
            raise TypeError("stop_reason must be int64 [P,N] on the rollout device")
        for value, name in (
            (self.trajectory_id, "trajectory_id"),
            (self.sample_id, "sample_id"),
            (self.branch_id, "branch_id"),
            (self.step_index, "step_index"),
        ):
            if (
                not isinstance(value, Tensor)
                or value.dtype != torch.int64
                or value.shape != (pairs,)
                or value.device != self.hidden_state.device
            ):
                raise TypeError(f"{name} must be int64 [P] on the rollout device")
        if self.hidden_state.requires_grad or self.hidden_state.grad_fn is not None:
            raise RefineTrainingContractError("rollout hidden states must be detached")
        tensors = (
            self.hidden_state,
            self.mask,
            self.attempted,
            self.committed,
            self.finite,
            self.stop_reason,
            self.trajectory_id,
            self.sample_id,
            self.branch_id,
            self.step_index,
        )
        if any(value.grad_fn is not None for value in tensors):
            raise RefineTrainingContractError("rollout tensors must not retain an autograd graph")
        _require_non_negative_int(self.snapshot_generation, name="snapshot_generation")
        try:
            source_ref = canonical_contract_reference(self.source_ref)
        except ValueError as error:
            raise ValueError("source_ref must be a component contract reference") from error
        if source_ref != self.source_ref:
            raise ValueError("source_ref must be a canonical contract reference")
        if isinstance(self.trajectory_count, bool) or self.trajectory_count <= 0:
            raise ValueError("trajectory_count must be positive")
        if isinstance(self.breadth, bool) or self.breadth <= 0:
            raise ValueError("breadth must be positive")
        if pairs:
            if bool(torch.any(self.trajectory_id < 0)) or bool(
                torch.any(self.trajectory_id >= self.trajectory_count)
            ):
                raise ValueError("trajectory_id is outside trajectory_count")
            if bool(torch.any(self.branch_id < 0)) or bool(
                torch.any(self.branch_id >= self.breadth)
            ):
                raise ValueError("branch_id is outside breadth")
            if bool(torch.any(self.step_index < 0)):
                raise ValueError("step_index must be non-negative")
            _validate_fixed_depth_lineage(
                self.trajectory_id,
                self.sample_id,
                self.branch_id,
                self.step_index,
                trajectory_count=self.trajectory_count,
                breadth=self.breadth,
            )
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
        if self.sampling_policy != "fixed_depth":
            raise ValueError("refine-rollout@1 supports only fixed_depth sampling")

    @property
    def valid_token_mask(self) -> Tensor:
        """Tokens whose source transition was live, finite, and committed."""

        return self.mask & self.attempted & self.committed & self.finite

    @property
    def valid_item_mask(self) -> Tensor:
        """Flattened rows containing at least one trainable transition."""

        return self.valid_token_mask.any(dim=-1)

    def permute(self, order: Tensor) -> "RefineRollout":
        """Return an equivalent pair ordering without changing trajectory identity."""

        if (
            not isinstance(order, Tensor)
            or order.dtype != torch.int64
            or order.shape != (self.hidden_state.shape[0],)
            or order.device != self.hidden_state.device
        ):
            raise TypeError("order must be int64 [P] on the rollout device")
        expected = torch.arange(order.numel(), device=order.device)
        if not torch.equal(torch.sort(order).values, expected):
            raise ValueError("order must be a permutation of [0,P)")
        fields = {
            name: getattr(self, name).index_select(0, order)
            for name in (
                "hidden_state",
                "mask",
                "attempted",
                "committed",
                "finite",
                "stop_reason",
                "trajectory_id",
                "sample_id",
                "branch_id",
                "step_index",
            )
        }
        return replace(self, **fields)


@dataclass(frozen=True)
class RefineStepTrainingResult:
    """Fresh one-step outputs aligned with a :class:`RefineRollout`."""

    value: Tensor
    valid_token_mask: Tensor
    trajectory_id: Tensor
    sample_id: Tensor
    branch_id: Tensor
    step_index: Tensor
    trajectory_count: int
    sample_count: int
    route: Tensor
    indices: Tensor
    weights: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.value, Tensor) or self.value.ndim != 3:
            raise TypeError("value must be a rank-3 Tensor [P,N,D]")
        pairs, tokens, _ = self.value.shape
        if isinstance(self.trajectory_count, bool) or self.trajectory_count <= 0:
            raise ValueError("trajectory_count must be positive")
        if isinstance(self.sample_count, bool) or self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if (
            not isinstance(self.valid_token_mask, Tensor)
            or self.valid_token_mask.dtype != torch.bool
            or self.valid_token_mask.shape != (pairs, tokens)
            or self.valid_token_mask.device != self.value.device
        ):
            raise TypeError("valid_token_mask must be bool [P,N] on the value device")
        for tensor, name in (
            (self.trajectory_id, "trajectory_id"),
            (self.sample_id, "sample_id"),
            (self.branch_id, "branch_id"),
            (self.step_index, "step_index"),
        ):
            if (
                not isinstance(tensor, Tensor)
                or tensor.dtype != torch.int64
                or tensor.shape != (pairs,)
                or tensor.device != self.value.device
            ):
                raise TypeError(f"{name} must be int64 [P] on the value device")
        for tensor, name in (
            (self.route, "route"),
            (self.weights, "weights"),
        ):
            if (
                not isinstance(tensor, Tensor)
                or not tensor.is_floating_point()
                or tensor.shape[:2] != (pairs, tokens)
                or tensor.device != self.value.device
            ):
                raise TypeError(f"{name} must be floating point [P,N,...] on the value device")
        if (
            not isinstance(self.indices, Tensor)
            or self.indices.dtype != torch.int64
            or self.indices.shape[:2] != (pairs, tokens)
            or self.indices.device != self.value.device
        ):
            raise TypeError("indices must be int64 [P,N,...] on the value device")
        if self.trajectory_count % self.sample_count:
            raise RefineTrainingContractError(
                "trajectory_count must be divisible by sample_count"
            )
        _validate_fixed_depth_lineage(
            self.trajectory_id,
            self.sample_id,
            self.branch_id,
            self.step_index,
            trajectory_count=self.trajectory_count,
            breadth=self.trajectory_count // self.sample_count,
        )

    @property
    def valid_item_mask(self) -> Tensor:
        return self.valid_token_mask.any(dim=-1)


@dataclass(frozen=True)
class RefineTrainingLoss:
    """Trajectory-normalized downstream loss and optional local pressure."""

    total: Tensor
    task: Tensor
    improvement: Tensor
    valid_trajectories: Tensor


@dataclass(frozen=True)
class RefineStepTraining:
    """Replay detached Refine states through one fresh differentiable step."""

    _component_reference: ClassVar[str] = "arti/refine-step-training@1"

    max_snapshot_staleness: int = 0

    def __post_init__(self) -> None:
        _require_non_negative_int(
            self.max_snapshot_staleness,
            name="max_snapshot_staleness",
        )

    def assert_optimizer_contract(self, recall: object, optimizer: torch.optim.Optimizer) -> None:
        """Reject a Query that is trainable or present in optimizer parameter groups."""

        _assert_fixed_query(recall)
        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch.optim.Optimizer")
        query_ids = {id(parameter) for parameter in _recall_field(recall).query.parameters()}
        if any(
            id(parameter) in query_ids
            for group in optimizer.param_groups
            for parameter in group["params"]
        ):
            raise RefineTrainingContractError(
                "fixed Query parameters must not be owned by the optimizer"
            )

    def capture(
        self,
        recall: object,
        hidden_state: Tensor,
        *,
        policy: AdaptiveExecutionPolicy,
        mask: Tensor | None = None,
        breadth: int = 1,
        snapshot_generation: int = 0,
    ) -> RefineRollout:
        """Capture a detached rollout from the canonical Recall executor."""

        _assert_fixed_query(recall)
        _assert_deterministic_execution(recall)
        if not recall.state.state_bank_calibrated:
            raise RefineTrainingContractError(
                "Recall state Bank must be calibrated before rollout capture"
            )
        if not isinstance(policy, AdaptiveExecutionPolicy):
            raise TypeError("policy must be an AdaptiveExecutionPolicy")
        if policy.max_steps <= 0:
            raise ValueError("rollout policy must execute at least one step")
        if policy.stop.scope != "token":
            raise ValueError("flattened Refine training requires token-scoped stopping")
        if policy.min_steps != policy.max_steps:
            raise RefineTrainingContractError(
                "refine-rollout@1 requires fixed depth; adaptive stopping is a later contract"
            )
        if isinstance(breadth, bool) or not isinstance(breadth, int) or breadth <= 0:
            raise ValueError("breadth must be a positive integer")
        _require_non_negative_int(snapshot_generation, name="snapshot_generation")
        sequence, token_mask, _was_vector = _as_sequence(hidden_state, mask)
        if sequence.shape[-1] != recall.dim:
            raise ValueError(f"hidden_state last dim must be {recall.dim}")

        query_before = _query_fingerprint(recall)
        snapshot_before = module_value_sha256(recall)
        structure = module_structure_fingerprint(recall)
        behavior = module_behavior_fingerprint(recall)
        capture_policy = policy.replace(
            checkpoints=tuple(range(1, policy.max_steps + 1)),
            trace_level="routes",
            checkpoint_mode="detached",
            executor="static_masked",
        )

        with torch.no_grad():
            if breadth == 1:
                _value, _delta, diagnostics = recall.state(
                    sequence,
                    token_mask,
                    execution_policy=capture_policy,
                )
                branch_mask = token_mask.unsqueeze(1)
                branch_diagnostics = {
                    name: value.unsqueeze(1)
                    if value.ndim > 0 and value.shape[0] == sequence.shape[0]
                    else value
                    for name, value in diagnostics.items()
                }
            else:
                from .branch_search import run_branch_search
                result = run_branch_search(
                    recall,
                    sequence,
                    mask=token_mask,
                    max_k=breadth,
                    active_k=breadth,
                    execution_policy=capture_policy,
                )
                origin = result.candidates.branch_origin_index
                branch_mask = _canonical_branches(
                    result.candidates.candidate_mask.permute(0, 2, 1),
                    origin,
                )
                branch_diagnostics = {
                    name: _canonical_branches(value, origin)
                    if value.ndim >= 2 and value.shape[:2] == origin.shape
                    else value
                    for name, value in result.branch_diagnostics.items()
                }

        if module_value_sha256(recall) != snapshot_before:
            raise RefineTrainingContractError("rollout capture mutated the Recall snapshot")
        if _query_fingerprint(recall) != query_before:
            raise RefineTrainingContractError("rollout capture changed the fixed Query")

        attempted = branch_diagnostics.get("recall_token_step_attempted")
        committed = branch_diagnostics.get("recall_token_step_committed")
        stop_reason = branch_diagnostics.get("recall_token_stop_reason")
        steps_attempted = branch_diagnostics.get("recall_token_steps_attempted")
        expected_prefix = (sequence.shape[0], breadth)
        expected_steps = policy.max_steps
        if (
            not isinstance(attempted, Tensor)
            or attempted.shape != (*expected_prefix, expected_steps, sequence.shape[1])
            or not isinstance(committed, Tensor)
            or committed.shape != attempted.shape
            or not isinstance(stop_reason, Tensor)
            or stop_reason.shape != (*expected_prefix, sequence.shape[1])
            or not isinstance(steps_attempted, Tensor)
            or steps_attempted.shape != stop_reason.shape
        ):
            raise RefineTrainingContractError(
                "canonical Recall diagnostics do not expose token-resolved rollout state"
            )

        checkpoints = []
        for step in range(1, expected_steps + 1):
            checkpoint = branch_diagnostics.get(f"recall_checkpoint_{step}")
            expected_checkpoint_shape = (
                sequence.shape[0],
                breadth,
                sequence.shape[1],
                sequence.shape[2],
            )
            if (
                not isinstance(checkpoint, Tensor)
                or checkpoint.shape != expected_checkpoint_shape
            ):
                raise RefineTrainingContractError(
                    f"rollout is missing detached checkpoint {step}"
                )
            checkpoints.append(checkpoint.detach())
        states = [
            sequence.unsqueeze(1).expand(-1, breadth, -1, -1).detach(),
            *checkpoints[:-1],
        ]
        state_tensor = torch.stack(states, dim=2)
        post_state_tensor = torch.stack(checkpoints, dim=2)
        mask_tensor = branch_mask.unsqueeze(2).expand(-1, -1, expected_steps, -1)
        finite = torch.isfinite(post_state_tensor).all(dim=-1)
        depth = torch.arange(expected_steps, device=sequence.device).view(1, 1, -1, 1)
        is_terminal = depth + 1 >= steps_attempted.unsqueeze(2)
        per_step_reason = torch.where(
            is_terminal,
            stop_reason.unsqueeze(2),
            torch.full_like(stop_reason.unsqueeze(2), int(ExecutionStopReason.MAX_STEPS)),
        ).expand_as(attempted)

        batch = sequence.shape[0]
        trajectory_count = batch * breadth
        trajectory = torch.arange(
            trajectory_count, device=sequence.device, dtype=torch.int64
        ).reshape(batch, breadth, 1).expand(-1, -1, expected_steps)
        sample = torch.arange(batch, device=sequence.device, dtype=torch.int64).reshape(
            batch, 1, 1
        ).expand(-1, breadth, expected_steps)
        branch = torch.arange(breadth, device=sequence.device, dtype=torch.int64).reshape(
            1, breadth, 1
        ).expand(batch, -1, expected_steps)
        step = torch.arange(
            expected_steps, device=sequence.device, dtype=torch.int64
        ).reshape(1, 1, expected_steps).expand(batch, breadth, -1)

        def flatten(value: Tensor) -> Tensor:
            return value.reshape(trajectory_count * expected_steps, *value.shape[3:]).clone()

        return RefineRollout(
            hidden_state=flatten(state_tensor),
            mask=flatten(mask_tensor),
            attempted=flatten(attempted),
            committed=flatten(committed),
            finite=flatten(finite),
            stop_reason=flatten(per_step_reason),
            trajectory_id=trajectory.reshape(-1).clone(),
            sample_id=sample.reshape(-1).clone(),
            branch_id=branch.reshape(-1).clone(),
            step_index=step.reshape(-1).clone(),
            trajectory_count=trajectory_count,
            breadth=breadth,
            source_ref=canonical_contract_reference(recall._component_reference),
            source_structure_fingerprint=structure,
            source_behavior_fingerprint=behavior,
            source_execution_fingerprint=_execution_config_fingerprint(recall),
            query_fingerprint=query_before,
            bank_fingerprint=_bank_fingerprint(recall),
            formula_fingerprint=_formula_fingerprint(recall),
            snapshot_fingerprint=snapshot_before,
            snapshot_generation=snapshot_generation,
        )

    def _validate_replay_source(
        self,
        recall: object,
        rollout: RefineRollout,
        *,
        current_generation: int,
    ) -> None:
        _assert_fixed_query(recall)
        _assert_deterministic_execution(recall)
        _require_non_negative_int(current_generation, name="current_generation")
        if canonical_contract_reference(recall._component_reference) != rollout.source_ref:
            raise RefineTrainingContractError("rollout source component does not match Recall")
        if module_structure_fingerprint(recall) != rollout.source_structure_fingerprint:
            raise RefineTrainingContractError("rollout source structure changed")
        if module_behavior_fingerprint(recall) != rollout.source_behavior_fingerprint:
            raise RefineTrainingContractError("rollout source behavior changed")
        if _execution_config_fingerprint(recall) != rollout.source_execution_fingerprint:
            raise RefineTrainingContractError("rollout execution configuration changed")
        if _query_fingerprint(recall) != rollout.query_fingerprint:
            raise RefineTrainingContractError("fixed Query changed since rollout capture")
        staleness = current_generation - rollout.snapshot_generation
        if staleness < 0:
            raise RefineTrainingContractError("current generation predates the rollout")
        if staleness > self.max_snapshot_staleness:
            raise RefineTrainingContractError("rollout exceeded bounded snapshot staleness")
        if rollout.breadth > 1 and staleness:
            raise RefineTrainingContractError(
                "K-wide rollout replay requires the exact capture generation"
            )
        if staleness == 0 and module_value_sha256(recall) != rollout.snapshot_fingerprint:
            raise RefineTrainingContractError("same-generation Recall snapshot changed")

    @staticmethod
    def _one_step_policy() -> AdaptiveExecutionPolicy:
        return ExecutionPolicy.adaptive(
            max_steps=1,
            min_steps=1,
            scope="token",
            relative_tolerance=1e-12,
            trace_level="routes",
            executor="static_masked",
        )

    @staticmethod
    def _run_rows(
        recall: object,
        hidden: Tensor,
        mask: Tensor,
        *,
        selected_groups: Tensor | None = None,
    ) -> tuple[Tensor, dict[str, Tensor]]:
        value, _delta, diagnostics = recall.state(
            hidden,
            mask,
            execution_policy=RefineStepTraining._one_step_policy(),
            selected_groups=selected_groups,
            selected_groups_first_step_only=True,
        )
        return value, diagnostics

    def replay(
        self,
        recall: object,
        rollout: RefineRollout,
        *,
        current_generation: int = 0,
    ) -> RefineStepTrainingResult:
        """Run each valid pair through one fresh differentiable Recall query."""

        if not isinstance(rollout, RefineRollout):
            raise TypeError("rollout must be a RefineRollout")
        self._validate_replay_source(
            recall,
            rollout,
            current_generation=current_generation,
        )
        query_before = _query_fingerprint(recall)
        source_valid_tokens = rollout.valid_token_mask
        valid_tokens = torch.zeros_like(source_valid_tokens)
        valid_rows = torch.nonzero(
            source_valid_tokens.any(dim=-1), as_tuple=False
        ).reshape(-1)
        value = rollout.hidden_state.clone()
        route = rollout.hidden_state.new_empty(
            rollout.hidden_state.shape[0], rollout.hidden_state.shape[1], 0
        )
        indices = torch.empty(
            rollout.hidden_state.shape[0],
            rollout.hidden_state.shape[1],
            0,
            device=rollout.hidden_state.device,
            dtype=torch.int64,
        )
        weights = route.clone()
        if valid_rows.numel():
            first_mask = rollout.step_index.index_select(0, valid_rows) == 0
            groups: list[tuple[Tensor, Tensor, dict[str, Tensor], Tensor]] = []
            for select_first in (True, False):
                local = valid_rows[first_mask == select_first]
                if local.numel() == 0:
                    continue
                hidden = rollout.hidden_state.index_select(0, local)
                token_mask = rollout.mask.index_select(0, local)
                selected_groups = None
                if select_first and rollout.breadth > 1:
                    from .branch_search import query_recall_branches

                    local_sample = rollout.sample_id.index_select(0, local)
                    sample_order = torch.argsort(local_sample, stable=True)
                    ordered_sample = local_sample.index_select(0, sample_order)
                    representative = torch.ones_like(ordered_sample, dtype=torch.bool)
                    representative[1:] = ordered_sample[1:] != ordered_sample[:-1]
                    representative_row = sample_order[representative]
                    representative_sample = local_sample.index_select(
                        0, representative_row
                    )
                    candidates = query_recall_branches(
                        recall,
                        hidden.index_select(0, representative_row),
                        mask=token_mask.index_select(0, representative_row),
                        max_k=rollout.breadth,
                        active_k=rollout.breadth,
                    )
                    origin = candidates.branch_origin_index
                    flat_candidate_groups = candidates.flattened_execution_groups()
                    candidate_groups = flat_candidate_groups.reshape(
                        representative_row.numel(),
                        rollout.breadth,
                        *flat_candidate_groups.shape[1:],
                    )
                    candidate_groups = _canonical_branches(candidate_groups, origin)
                    sample_to_representative = torch.full(
                        (rollout.trajectory_count // rollout.breadth,),
                        -1,
                        device=local.device,
                        dtype=torch.int64,
                    )
                    sample_to_representative[representative_sample] = torch.arange(
                        representative_sample.numel(),
                        device=local.device,
                        dtype=torch.int64,
                    )
                    local_representative = sample_to_representative[local_sample]
                    local_branch = rollout.branch_id.index_select(0, local)
                    selected_groups = candidate_groups[
                        local_representative,
                        local_branch,
                    ]
                local_value, diagnostics = self._run_rows(
                    recall,
                    hidden,
                    token_mask,
                    selected_groups=selected_groups,
                )
                attempted = diagnostics.get("recall_token_step_attempted")
                committed = diagnostics.get("recall_token_step_committed")
                expected_status_shape = (local.numel(), 1, hidden.shape[1])
                if (
                    not isinstance(attempted, Tensor)
                    or attempted.shape != expected_status_shape
                    or not isinstance(committed, Tensor)
                    or committed.shape != expected_status_shape
                ):
                    raise RefineTrainingContractError(
                        "one-step replay did not expose token-resolved transition status"
                    )
                current_valid = (
                    source_valid_tokens.index_select(0, local)
                    & attempted[:, 0]
                    & committed[:, 0]
                    & torch.isfinite(local_value).all(dim=-1)
                )
                groups.append((local, local_value, diagnostics, current_valid))

            sample_diagnostics = groups[0][2]
            route_shape = sample_diagnostics["recall_route"].shape[1:]
            index_shape = sample_diagnostics["recall_bank_indices"].shape[1:]
            weight_shape = sample_diagnostics["recall_bank_weights"].shape[1:]
            route = rollout.hidden_state.new_zeros(
                (rollout.hidden_state.shape[0], *route_shape)
            )
            indices = torch.full(
                (rollout.hidden_state.shape[0], *index_shape),
                -1,
                device=rollout.hidden_state.device,
                dtype=torch.int64,
            )
            weights = rollout.hidden_state.new_zeros(
                (rollout.hidden_state.shape[0], *weight_shape)
            )
            for local, local_value, diagnostics, current_valid in groups:
                value = value.index_copy(0, local, local_value)
                valid_tokens = valid_tokens.index_copy(0, local, current_valid)
                route = route.index_copy(0, local, diagnostics["recall_route"])
                indices = indices.index_copy(0, local, diagnostics["recall_bank_indices"])
                weights = weights.index_copy(0, local, diagnostics["recall_bank_weights"])

        if _query_fingerprint(recall) != query_before:
            raise RefineTrainingContractError("one-step replay changed the fixed Query")
        return RefineStepTrainingResult(
            value=value,
            valid_token_mask=valid_tokens,
            trajectory_id=rollout.trajectory_id,
            sample_id=rollout.sample_id,
            branch_id=rollout.branch_id,
            step_index=rollout.step_index,
            trajectory_count=rollout.trajectory_count,
            sample_count=rollout.trajectory_count // rollout.breadth,
            route=route,
            indices=indices,
            weights=weights,
        )

    @staticmethod
    def _row_loss(loss: Tensor, result: RefineStepTrainingResult) -> tuple[Tensor, Tensor]:
        pairs, tokens = result.valid_token_mask.shape
        if not isinstance(loss, Tensor) or not loss.is_floating_point() or loss.shape[0] != pairs:
            raise TypeError("task loss must be a floating-point Tensor beginning with [P]")
        valid_item = result.valid_item_mask
        if loss.ndim == 1:
            if not bool(torch.isfinite(loss[valid_item]).all()):
                raise RefineTrainingContractError("task loss is non-finite on a live row")
            return torch.where(valid_item, loss, 0), valid_item
        if loss.shape[1] == tokens:
            reduced = loss
            if loss.ndim > 2:
                reduced = loss.mean(dim=tuple(range(2, loss.ndim)))
            if not bool(torch.isfinite(reduced[result.valid_token_mask]).all()):
                raise RefineTrainingContractError("task loss is non-finite on a live token")
            weights = result.valid_token_mask.to(dtype=reduced.dtype)
            safe_reduced = torch.where(result.valid_token_mask, reduced, 0)
            row = (safe_reduced * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)
            return row, valid_item
        row = loss.mean(dim=tuple(range(1, loss.ndim)))
        if not bool(torch.isfinite(row[valid_item]).all()):
            raise RefineTrainingContractError("task loss is non-finite on a live row")
        return torch.where(valid_item, row, 0), valid_item

    @staticmethod
    def _sample_balanced_mean(
        row_loss: Tensor,
        valid_item: Tensor,
        trajectory_id: Tensor,
        sample_id: Tensor,
        trajectory_count: int,
        sample_count: int,
    ) -> tuple[Tensor, Tensor]:
        _require_non_negative_int(trajectory_count, name="trajectory_count")
        if trajectory_count == 0:
            raise ValueError("trajectory_count must be positive")
        _require_non_negative_int(sample_count, name="sample_count")
        if sample_count == 0:
            raise ValueError("sample_count must be positive")
        if bool(torch.any(trajectory_id < 0)) or bool(
            torch.any(trajectory_id >= trajectory_count)
        ):
            raise RefineTrainingContractError("trajectory_id is outside trajectory_count")
        if bool(torch.any(sample_id < 0)) or bool(torch.any(sample_id >= sample_count)):
            raise RefineTrainingContractError("sample_id is outside sample_count")
        weights = valid_item.to(dtype=row_loss.dtype)
        all_rows = torch.ones_like(weights)
        total_counts = row_loss.new_zeros((trajectory_count,)).scatter_add(
            0, trajectory_id, all_rows
        )
        sums = row_loss.new_zeros((trajectory_count,)).scatter_add(
            0, trajectory_id, row_loss * weights
        )
        counts = row_loss.new_zeros((trajectory_count,)).scatter_add(
            0, trajectory_id, weights
        )
        incomplete_trajectory = (counts > 0) & (counts < total_counts)
        if bool(torch.any(incomplete_trajectory)):
            raise RefineTrainingContractError(
                "a trajectory contains only a partial set of valid refine depths"
            )
        valid_trajectory = counts == total_counts
        if not bool(torch.any(valid_trajectory)):
            raise RefineTrainingContractError(
                "task loss contains no live, finite, committed transition"
            )
        trajectory_mean = sums / counts.clamp_min(1)
        valid_trajectory_id = trajectory_id[valid_item]
        valid_sample_id = sample_id[valid_item]
        trajectory_sample_min = torch.full(
            (trajectory_count,),
            sample_count,
            device=sample_id.device,
            dtype=torch.int64,
        )
        trajectory_sample_max = torch.full(
            (trajectory_count,),
            -1,
            device=sample_id.device,
            dtype=torch.int64,
        )
        trajectory_sample_min.scatter_reduce_(
            0,
            valid_trajectory_id,
            valid_sample_id,
            reduce="amin",
            include_self=True,
        )
        trajectory_sample_max.scatter_reduce_(
            0,
            valid_trajectory_id,
            valid_sample_id,
            reduce="amax",
            include_self=True,
        )
        if bool(
            torch.any(
                trajectory_sample_min[valid_trajectory]
                != trajectory_sample_max[valid_trajectory]
            )
        ):
            raise RefineTrainingContractError("one trajectory contains multiple sample IDs")
        trajectory_sample = trajectory_sample_min
        sample_sums = row_loss.new_zeros((sample_count,)).scatter_add(
            0,
            trajectory_sample[valid_trajectory],
            trajectory_mean[valid_trajectory],
        )
        sample_counts = row_loss.new_zeros((sample_count,)).scatter_add(
            0,
            trajectory_sample[valid_trajectory],
            torch.ones_like(trajectory_mean[valid_trajectory]),
        )
        breadth = trajectory_count // sample_count
        partial_sample = (sample_counts > 0) & (sample_counts < breadth)
        if bool(torch.any(partial_sample)):
            raise RefineTrainingContractError(
                "a source sample contains only a partial set of valid branches"
            )
        valid_sample = sample_counts == breadth
        mean = (sample_sums / sample_counts.clamp_min(1))[valid_sample].mean()
        return mean, valid_trajectory.sum(dtype=torch.int64)

    def reduce_task_loss(
        self,
        task_loss: Tensor,
        result: RefineStepTrainingResult,
        *,
        baseline_loss: Tensor | None = None,
        improvement_weight: float = 0.0,
        margin: float = 0.0,
    ) -> RefineTrainingLoss:
        """Normalize task loss per depth, branch, and source sample."""

        if not isinstance(result, RefineStepTrainingResult):
            raise TypeError("result must be a RefineStepTrainingResult")
        if not isinstance(improvement_weight, (int, float)) or improvement_weight < 0:
            raise ValueError("improvement_weight must be non-negative")
        if not isinstance(margin, (int, float)) or margin < 0:
            raise ValueError("margin must be non-negative")
        row_task, valid_item = self._row_loss(task_loss, result)
        task, valid_trajectories = self._sample_balanced_mean(
            row_task,
            valid_item,
            result.trajectory_id,
            result.sample_id,
            result.trajectory_count,
            result.sample_count,
        )
        improvement = task.new_zeros(())
        if baseline_loss is not None:
            row_baseline, baseline_valid = self._row_loss(baseline_loss, result)
            if not torch.equal(baseline_valid, valid_item):
                raise RefineTrainingContractError("baseline validity differs from task loss")
            row_improvement = torch.relu(
                row_task - row_baseline.detach() + float(margin)
            )
            improvement, _ = self._sample_balanced_mean(
                row_improvement,
                valid_item,
                result.trajectory_id,
                result.sample_id,
                result.trajectory_count,
                result.sample_count,
            )
        elif improvement_weight:
            raise ValueError("improvement_weight requires baseline_loss")
        total = task + float(improvement_weight) * improvement
        return RefineTrainingLoss(
            total=total,
            task=task,
            improvement=improvement,
            valid_trajectories=valid_trajectories,
        )


__all__ = [
    "RefineRollout",
    "RefineStepTraining",
    "RefineStepTrainingResult",
    "RefineTrainingContractError",
    "RefineTrainingLoss",
]
