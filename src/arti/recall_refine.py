"""Versioned runtime contracts for iterative Recall reads."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import IntEnum
import math
from types import MappingProxyType
from typing import ClassVar, TypeAlias
from collections.abc import Mapping

import torch
from torch import Tensor


RECALL_TRACE_SCHEMA_VERSION = 1
RECALL_TRACE_V2_SCHEMA_VERSION = 2


_TRACE_LEVELS = frozenset({"none", "summary", "routes", "full"})
_CHECKPOINT_MODES = frozenset({"detached", "gradient"})
_NONFINITE_ACTIONS = frozenset({"stop", "raise"})
_REFINE_SCOPES = frozenset({"sample", "token"})
_REFINE_EXECUTORS = frozenset({"static_masked", "early_break"})


def _validate_non_negative_number(value: float | None, *, name: str) -> None:
    if value is None:
        return
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{name} must be a finite non-negative number")


def _validate_cycle_periods(values: tuple[int, ...]) -> None:
    if not isinstance(values, tuple):
        raise TypeError("cycle_periods must be a tuple")
    if any(isinstance(value, bool) or not isinstance(value, int) or value <= 1 for value in values):
        raise ValueError("cycle_periods must contain integers greater than one")
    if len(set(values)) != len(values):
        raise ValueError("cycle_periods must not contain duplicates")


def _validate_checkpoints(values: tuple[int, ...], *, max_steps: int) -> None:
    if not isinstance(values, tuple):
        raise TypeError("checkpoints must be a tuple")
    if any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0 or value > max_steps
        for value in values
    ):
        raise ValueError("checkpoints must be positive steps within max_steps")
    if tuple(sorted(set(values))) != values:
        raise ValueError("checkpoints must be sorted and unique")


class RecallStopReason(IntEnum):
    """Tensor-safe reason codes used by :class:`RecallTrace`."""

    MAX_STEPS = 0
    CONVERGED = 1
    NONFINITE = 2
    CYCLE = 3
    MASKED = 4


@dataclass(frozen=True)
class RefinePolicy:
    """Per-call Recall compute policy.

    A policy is runtime configuration, not model state. It is deliberately not
    mounted below ``Recall`` and therefore does not change Bank, component, or
    artifact fingerprints.
    """

    _component_reference: ClassVar[str] = "arti/refine-policy@1"

    max_steps: int = 1
    min_steps: int = 1
    tolerance: float | None = None
    checkpoints: tuple[int, ...] = ()
    trace_level: str = "none"
    cycle_tolerance: float | None = None
    cycle_periods: tuple[int, ...] = (2, 3, 4)
    check_finite: bool = True
    nonfinite_action: str = "stop"
    checkpoint_mode: str = "detached"

    def __post_init__(self) -> None:
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int):
            raise TypeError("max_steps must be an integer")
        if isinstance(self.min_steps, bool) or not isinstance(self.min_steps, int):
            raise TypeError("min_steps must be an integer")
        if self.max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        if self.min_steps < 0 or self.min_steps > self.max_steps:
            raise ValueError("min_steps must be between zero and max_steps")
        if self.tolerance is not None and (
            isinstance(self.tolerance, bool)
            or not isinstance(self.tolerance, (int, float))
            or not math.isfinite(self.tolerance)
            or self.tolerance < 0
        ):
            raise ValueError("tolerance must be a finite non-negative number")
        if self.trace_level not in {"none", "summary", "routes", "full"}:
            raise ValueError("trace_level must be 'none', 'summary', 'routes', or 'full'")
        if self.cycle_tolerance is not None and (
            isinstance(self.cycle_tolerance, bool)
            or not isinstance(self.cycle_tolerance, (int, float))
            or not math.isfinite(self.cycle_tolerance)
            or self.cycle_tolerance < 0
        ):
            raise ValueError("cycle_tolerance must be a finite non-negative number")
        if not isinstance(self.check_finite, bool):
            raise TypeError("check_finite must be a bool")
        if self.nonfinite_action not in {"stop", "raise"}:
            raise ValueError("nonfinite_action must be 'stop' or 'raise'")
        if self.checkpoint_mode not in {"detached", "gradient"}:
            raise ValueError("checkpoint_mode must be 'detached' or 'gradient'")
        if self.nonfinite_action == "raise" and not self.check_finite:
            raise ValueError("nonfinite_action='raise' requires check_finite=True")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 1
            for value in self.cycle_periods
        ):
            raise ValueError("cycle_periods must contain integers greater than one")
        if len(set(self.cycle_periods)) != len(self.cycle_periods):
            raise ValueError("cycle_periods must not contain duplicates")
        if any(
            isinstance(value, bool)
            or not isinstance(value, int)
            or value <= 0
            or value > self.max_steps
            for value in self.checkpoints
        ):
            raise ValueError("checkpoints must be positive steps within max_steps")
        if tuple(sorted(set(self.checkpoints))) != self.checkpoints:
            raise ValueError("checkpoints must be sorted and unique")

    @classmethod
    def fixed(cls, steps: int, **kwargs: object) -> "RefinePolicy":
        """Create a fixed-budget runtime policy."""

        return cls(max_steps=steps, min_steps=steps, **kwargs)

    @classmethod
    def adaptive(
        cls,
        *,
        max_steps: int,
        min_steps: int = 1,
        scope: str = "sample",
        absolute_tolerance: float = 0.0,
        relative_tolerance: float = 1e-3,
        route_tolerance: float | None = None,
        patience: int = 1,
        cycle_tolerance: float | None = None,
        cycle_periods: tuple[int, ...] = (2, 3),
        checkpoints: tuple[int, ...] = (),
        trace_level: str = "summary",
        check_finite: bool = True,
        nonfinite_action: str = "stop",
        checkpoint_mode: str = "detached",
        executor: str = "static_masked",
    ) -> "AdaptiveRefinePolicy":
        """Create the version-2 adaptive runtime contract.

        This factory does not reinterpret version 1. It constructs a separate
        component with explicit budget, stopping, tracing, and execution rules.
        """

        return AdaptiveRefinePolicy(
            budget=RefineBudget(max_steps=max_steps, min_steps=min_steps),
            stop=RefineStop(
                scope=scope,
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
                route_tolerance=route_tolerance,
                patience=patience,
                cycle_tolerance=cycle_tolerance,
                cycle_periods=cycle_periods,
            ),
            checkpoints=checkpoints,
            trace_level=trace_level,
            check_finite=check_finite,
            nonfinite_action=nonfinite_action,
            checkpoint_mode=checkpoint_mode,
            executor=executor,
        )

    def replace(self, **changes: object) -> "RefinePolicy":
        """Return a validated policy with selected runtime fields replaced."""

        return replace(self, **changes)


@dataclass(frozen=True)
class RefineBudget:
    """Versioned deterministic bound for one adaptive refine execution."""

    _component_reference: ClassVar[str] = "arti/refine-budget@1"

    max_steps: int
    min_steps: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.max_steps, bool) or not isinstance(self.max_steps, int):
            raise TypeError("max_steps must be an integer")
        if isinstance(self.min_steps, bool) or not isinstance(self.min_steps, int):
            raise TypeError("min_steps must be an integer")
        if self.max_steps < 0:
            raise ValueError("max_steps must be non-negative")
        if self.min_steps < 0 or self.min_steps > self.max_steps:
            raise ValueError("min_steps must be between zero and max_steps")


@dataclass(frozen=True)
class RefineStop:
    """Versioned convergence and cycle rules for adaptive refinement."""

    _component_reference: ClassVar[str] = "arti/refine-stop@1"

    scope: str = "sample"
    absolute_tolerance: float = 0.0
    relative_tolerance: float = 1e-3
    route_tolerance: float | None = None
    patience: int = 1
    cycle_tolerance: float | None = None
    cycle_periods: tuple[int, ...] = (2, 3)

    def __post_init__(self) -> None:
        if self.scope not in _REFINE_SCOPES:
            raise ValueError("scope must be 'sample' or 'token'")
        _validate_non_negative_number(self.absolute_tolerance, name="absolute_tolerance")
        _validate_non_negative_number(self.relative_tolerance, name="relative_tolerance")
        _validate_non_negative_number(self.route_tolerance, name="route_tolerance")
        _validate_non_negative_number(self.cycle_tolerance, name="cycle_tolerance")
        if self.absolute_tolerance == 0 and self.relative_tolerance == 0:
            raise ValueError("at least one state tolerance must be greater than zero")
        if isinstance(self.patience, bool) or not isinstance(self.patience, int):
            raise TypeError("patience must be an integer")
        if self.patience <= 0:
            raise ValueError("patience must be positive")
        _validate_cycle_periods(self.cycle_periods)


@dataclass(frozen=True)
class AdaptiveRefinePolicy:
    """Composable version-2 runtime policy for adaptive Recall refinement.

    The policy describes numerical semantics only. ``static_masked`` and
    ``early_break`` may differ in physical work, but must produce the same
    committed token states and trace semantics.
    """

    _component_reference: ClassVar[str] = "arti/refine-policy@2"

    budget: RefineBudget
    stop: RefineStop
    checkpoints: tuple[int, ...] = ()
    trace_level: str = "summary"
    check_finite: bool = True
    nonfinite_action: str = "stop"
    checkpoint_mode: str = "detached"
    executor: str = "static_masked"

    def __post_init__(self) -> None:
        if not isinstance(self.budget, RefineBudget):
            raise TypeError("budget must be a RefineBudget")
        if not isinstance(self.stop, RefineStop):
            raise TypeError("stop must be a RefineStop")
        _validate_checkpoints(self.checkpoints, max_steps=self.budget.max_steps)
        if self.trace_level not in _TRACE_LEVELS:
            raise ValueError("trace_level must be 'none', 'summary', 'routes', or 'full'")
        if not isinstance(self.check_finite, bool):
            raise TypeError("check_finite must be a bool")
        if self.nonfinite_action not in _NONFINITE_ACTIONS:
            raise ValueError("nonfinite_action must be 'stop' or 'raise'")
        if self.checkpoint_mode not in _CHECKPOINT_MODES:
            raise ValueError("checkpoint_mode must be 'detached' or 'gradient'")
        if self.executor not in _REFINE_EXECUTORS:
            raise ValueError("executor must be 'static_masked' or 'early_break'")
        if self.nonfinite_action == "raise" and not self.check_finite:
            raise ValueError("nonfinite_action='raise' requires check_finite=True")

    @property
    def max_steps(self) -> int:
        return self.budget.max_steps

    @property
    def min_steps(self) -> int:
        return self.budget.min_steps

    def replace(self, **changes: object) -> "AdaptiveRefinePolicy":
        """Return a validated version-2 policy with selected fields replaced."""

        return replace(self, **changes)


@dataclass(frozen=True)
class RecallRoutePlan:
    """Versioned frozen Recall routing decision.

    A route plan stores selected positions, their mixing weights, and the
    normalized per-factor route mass used by optional influence controls.
    Replaying it still gathers the current Bank values and recomputes
    recognition, influence, and Formula application from the current state.
    It is therefore suitable for isolating dynamic rerouting from repeated
    state evolution without turning the Bank into a cached tensor read.
    """

    _component_reference: ClassVar[str] = "arti/recall-route-plan@1"
    schema_version: int
    routing: str
    value_composition: str
    slots: int
    composition_factor: int
    group_size: int
    layout_fingerprint: str
    weights: Tensor
    indices: Tensor
    route: Tensor

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                f"unsupported Recall route-plan schema {self.schema_version}; expected 1"
            )
        if self.routing not in {"dense", "grouped"}:
            raise ValueError("route-plan routing must be 'dense' or 'grouped'")
        if self.value_composition not in {"single", "product", "state", "custom"}:
            raise ValueError("route-plan value_composition is invalid")
        for name, value in {
            "slots": self.slots,
            "composition_factor": self.composition_factor,
            "group_size": self.group_size,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"route-plan {name} must be a positive integer")
        if not isinstance(self.layout_fingerprint, str) or not self.layout_fingerprint:
            raise ValueError("route-plan layout_fingerprint must be a non-empty string")
        if not isinstance(self.weights, Tensor) or not self.weights.is_floating_point():
            raise TypeError("route-plan weights must be a floating-point Tensor")
        if not isinstance(self.indices, Tensor) or self.indices.dtype != torch.long:
            raise TypeError("route-plan indices must be a torch.long Tensor")
        if not isinstance(self.route, Tensor) or not self.route.is_floating_point():
            raise TypeError("route-plan route must be a floating-point Tensor")
        if self.weights.ndim != 4 or self.indices.ndim != 4:
            raise ValueError("route-plan weights and indices must have shape [B,N,F,K]")
        if self.route.ndim != 4:
            raise ValueError("route-plan route must have shape [B,N,F,G]")
        if self.weights.shape[:2] != self.route.shape[:2]:
            raise ValueError("route-plan weights and route must share [B, N]")
        if self.indices.shape != self.weights.shape:
            raise ValueError("route-plan indices must match weights")
        if self.weights.shape[-2] != self.composition_factor:
            raise ValueError("route-plan factor axis does not match composition_factor")
        if self.route.shape[-2] != self.composition_factor:
            raise ValueError("route-plan route factor axis does not match composition_factor")
        device = self.weights.device
        if self.indices.device != device or self.route.device != device:
            raise ValueError("route-plan tensors must use one device")

    def detach(self) -> "RecallRoutePlan":
        return replace(
            self,
            weights=self.weights.detach(),
            indices=self.indices.detach(),
            route=self.route.detach(),
        )

    def clone(self) -> "RecallRoutePlan":
        return replace(
            self,
            weights=self.weights.clone(),
            indices=self.indices.clone(),
            route=self.route.clone(),
        )


RecallRouteNode: TypeAlias = "RecallRoutePlan | RecallRouteStack"


@dataclass(frozen=True)
class RecallRouteStack:
    """Versioned recursive composition of runtime-only Recall route plans."""

    _component_reference: ClassVar[str] = "arti/recall-route-stack@1"
    axis: str
    items: tuple[RecallRouteNode, ...]
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(
                f"unsupported Recall route-stack schema {self.schema_version}; expected 1"
            )
        if not isinstance(self.axis, str) or not self.axis:
            raise ValueError("route-stack axis must be a non-empty string")
        if not self.items:
            raise ValueError("route-stack items must not be empty")
        if any(not isinstance(item, (RecallRoutePlan, RecallRouteStack)) for item in self.items):
            raise TypeError("route-stack items must be RecallRoutePlan or RecallRouteStack")

    def detach(self) -> "RecallRouteStack":
        return replace(self, items=tuple(item.detach() for item in self.items))

    def clone(self) -> "RecallRouteStack":
        return replace(self, items=tuple(item.clone() for item in self.items))

    def to(self, device: torch.device | str) -> "RecallRouteStack":
        def move(item: RecallRouteNode) -> RecallRouteNode:
            if isinstance(item, RecallRouteStack):
                return item.to(device)
            return replace(
                item,
                weights=item.weights.to(device),
                indices=item.indices.to(device),
                route=item.route.to(device),
            )

        return replace(self, items=tuple(move(item) for item in self.items))


@dataclass(frozen=True)
class RecallTrace:
    """Fixed-shape observations from one iterative Recall execution.

    ``route`` and ``indices`` record attempted reads so the trace keeps the
    evidence that led to convergence or rejection. Consumers analyzing state
    transitions must mask those histories with ``step_committed``; top-level
    Recall diagnostics separately expose the last committed read.
    """

    steps_attempted: Tensor
    steps_committed: Tensor
    step_attempted: Tensor
    step_committed: Tensor
    state_change_ratio: Tensor
    route_change: Tensor
    effective_read_change: Tensor
    stop_reason: Tensor
    route: Tensor
    indices: Tensor
    checkpoints: Mapping[int, Tensor]
    schema_version: int = RECALL_TRACE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RECALL_TRACE_SCHEMA_VERSION:
            raise ValueError("unsupported RecallTrace schema version")
        if (
            self.steps_attempted.ndim != 1
            or self.steps_committed.shape != self.steps_attempted.shape
        ):
            raise ValueError("step counts must have shape [B]")
        batch = self.steps_attempted.shape[0]
        if self.step_attempted.ndim != 2 or self.step_attempted.shape[0] != batch:
            raise ValueError("step_attempted must have shape [B, R]")
        if self.step_committed.shape != self.step_attempted.shape:
            raise ValueError("step_committed must match step_attempted")
        expected = self.step_attempted.shape
        for name, value in (
            ("state_change_ratio", self.state_change_ratio),
            ("route_change", self.route_change),
            ("effective_read_change", self.effective_read_change),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} must match step_attempted")
        if self.stop_reason.shape != (batch,):
            raise ValueError("stop_reason must have shape [B]")
        if self.step_attempted.dtype != torch.bool or self.step_committed.dtype != torch.bool:
            raise TypeError("step attempt/commit masks must be boolean")
        if (
            self.steps_attempted.dtype != torch.int64
            or self.steps_committed.dtype != torch.int64
            or self.stop_reason.dtype != torch.int64
        ):
            raise TypeError("step counts and stop_reason must be int64")
        if self.route.ndim < 3 or self.route.shape[:2] != (batch, expected[1]):
            raise ValueError("route must begin with [B, R]")
        if self.indices.ndim < 3 or self.indices.shape[:2] != (batch, expected[1]):
            raise ValueError("indices must begin with [B, R]")
        if any(step <= 0 or step > expected[1] for step in self.checkpoints):
            raise ValueError("checkpoint keys must be executed-depth indices")
        object.__setattr__(self, "checkpoints", MappingProxyType(dict(self.checkpoints)))

    def validate(self) -> "RecallTrace":
        """Run value-dependent integrity checks outside compiled execution."""

        if not torch.equal(self.steps_attempted, self.step_attempted.sum(dim=1)):
            raise ValueError("steps_attempted must equal the attempted-step count")
        if not torch.equal(self.steps_committed, self.step_committed.sum(dim=1)):
            raise ValueError("steps_committed must equal the committed-step count")
        if bool(torch.any(self.step_committed & ~self.step_attempted)):
            raise ValueError("a committed step must also be attempted")
        valid_reasons = torch.tensor(
            [int(value) for value in RecallStopReason],
            device=self.stop_reason.device,
            dtype=self.stop_reason.dtype,
        )
        if not bool(torch.isin(self.stop_reason, valid_reasons).all()):
            raise ValueError("stop_reason contains an unknown reason code")
        if self.step_attempted.shape[1] > 1:
            reactivated = self.step_attempted[:, 1:] & ~self.step_attempted[:, :-1]
            if bool(torch.any(reactivated)):
                raise ValueError("step_active must be monotonically non-increasing")
        return self

    @property
    def max_steps(self) -> int:
        return int(self.step_attempted.shape[1])

    def diagnostics(self) -> dict[str, Tensor]:
        """Return tensor-only diagnostics for existing ARTI output APIs."""

        result = {
            "recall_trace_schema": torch.tensor(
                self.schema_version,
                device=self.steps_attempted.device,
                dtype=torch.int64,
            ),
            "recall_steps_attempted": self.steps_attempted,
            "recall_steps_committed": self.steps_committed,
            "recall_step_attempted": self.step_attempted,
            "recall_step_committed": self.step_committed,
            "recall_step_update_ratio": self.state_change_ratio,
            "recall_step_route_change": self.route_change,
            "recall_step_effective_read_change": self.effective_read_change,
            "recall_stop_reason": self.stop_reason,
            "recall_route_history": self.route,
            "recall_index_history": self.indices,
        }
        for step, state in self.checkpoints.items():
            result[f"recall_checkpoint_{step}"] = state
        return result

    @classmethod
    def from_diagnostics(
        cls,
        values: dict[str, Tensor],
        *,
        validate_schema: bool = True,
        validate_values: bool = True,
    ) -> "RecallTrace":
        """Reconstruct the typed view from tensor-only ARTI diagnostics."""

        schema = values.get("recall_trace_schema")
        if validate_schema and (
            not isinstance(schema, Tensor)
            or schema.numel() != 1
            or int(schema.detach().cpu()) != RECALL_TRACE_SCHEMA_VERSION
        ):
            raise ValueError("Recall diagnostics use an unsupported trace schema")
        required = {
            "recall_steps_attempted",
            "recall_steps_committed",
            "recall_step_attempted",
            "recall_step_committed",
            "recall_step_update_ratio",
            "recall_step_route_change",
            "recall_step_effective_read_change",
            "recall_stop_reason",
            "recall_route_history",
            "recall_index_history",
        }
        missing = required - set(values)
        if missing:
            raise ValueError(f"Recall diagnostics are missing trace fields: {sorted(missing)}")
        prefix = "recall_checkpoint_"
        checkpoints = {
            int(name[len(prefix) :]): value
            for name, value in values.items()
            if name.startswith(prefix)
        }
        trace = cls(
            steps_attempted=values["recall_steps_attempted"],
            steps_committed=values["recall_steps_committed"],
            step_attempted=values["recall_step_attempted"],
            step_committed=values["recall_step_committed"],
            state_change_ratio=values["recall_step_update_ratio"],
            route_change=values["recall_step_route_change"],
            effective_read_change=values["recall_step_effective_read_change"],
            stop_reason=values["recall_stop_reason"],
            route=values["recall_route_history"],
            indices=values["recall_index_history"],
            checkpoints=checkpoints,
        )
        return trace.validate() if validate_values else trace


@dataclass(frozen=True)
class RecallTraceV2:
    """Token-resolved observations from an adaptive Recall execution.

    Token counts and masks describe logical refinement. ``kernel_steps``
    separately reports physical loop execution so masked execution cannot be
    mistaken for compute savings.
    """

    token_steps_attempted: Tensor
    token_steps_committed: Tensor
    token_stop_reason: Tensor
    token_step_attempted: Tensor
    token_step_committed: Tensor
    state_change_ratio: Tensor
    route_change: Tensor
    effective_read_change: Tensor
    active_fraction: Tensor
    kernel_steps: Tensor
    logical_token_steps: Tensor
    route: Tensor
    indices: Tensor
    checkpoints: Mapping[int, Tensor]
    schema_version: int = RECALL_TRACE_V2_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RECALL_TRACE_V2_SCHEMA_VERSION:
            raise ValueError("unsupported RecallTraceV2 schema version")
        if self.token_steps_attempted.ndim != 2:
            raise ValueError("token_steps_attempted must have shape [B, N]")
        if self.token_steps_committed.shape != self.token_steps_attempted.shape:
            raise ValueError("token_steps_committed must match token_steps_attempted")
        if self.token_stop_reason.shape != self.token_steps_attempted.shape:
            raise ValueError("token_stop_reason must have shape [B, N]")

        batch, tokens = self.token_steps_attempted.shape
        if self.token_step_attempted.ndim != 3:
            raise ValueError("token_step_attempted must have shape [B, R, N]")
        if (
            self.token_step_attempted.shape[0] != batch
            or self.token_step_attempted.shape[2] != tokens
        ):
            raise ValueError("token_step_attempted must share [B, N] with token counts")
        if self.token_step_committed.shape != self.token_step_attempted.shape:
            raise ValueError("token_step_committed must match token_step_attempted")

        expected = self.token_step_attempted.shape
        for name, value in (
            ("state_change_ratio", self.state_change_ratio),
            ("route_change", self.route_change),
            ("effective_read_change", self.effective_read_change),
        ):
            if value.shape != expected:
                raise ValueError(f"{name} must have shape [B, R, N]")

        refine_steps = expected[1]
        if self.active_fraction.shape != (refine_steps,):
            raise ValueError("active_fraction must have shape [R]")
        if self.kernel_steps.ndim != 0:
            raise ValueError("kernel_steps must be a scalar Tensor")
        if self.logical_token_steps.ndim != 0:
            raise ValueError("logical_token_steps must be a scalar Tensor")

        if self.token_step_attempted.dtype != torch.bool:
            raise TypeError("token_step_attempted must be boolean")
        if self.token_step_committed.dtype != torch.bool:
            raise TypeError("token_step_committed must be boolean")
        for name, value in (
            ("token_steps_attempted", self.token_steps_attempted),
            ("token_steps_committed", self.token_steps_committed),
            ("token_stop_reason", self.token_stop_reason),
            ("kernel_steps", self.kernel_steps),
            ("logical_token_steps", self.logical_token_steps),
        ):
            if value.dtype != torch.int64:
                raise TypeError(f"{name} must be int64")
        for name, value in (
            ("state_change_ratio", self.state_change_ratio),
            ("route_change", self.route_change),
            ("effective_read_change", self.effective_read_change),
            ("active_fraction", self.active_fraction),
        ):
            if not value.is_floating_point():
                raise TypeError(f"{name} must be floating point")

        reference_device = self.token_steps_attempted.device
        tensors = (
            self.token_steps_committed,
            self.token_stop_reason,
            self.token_step_attempted,
            self.token_step_committed,
            self.state_change_ratio,
            self.route_change,
            self.effective_read_change,
            self.active_fraction,
            self.kernel_steps,
            self.logical_token_steps,
            self.route,
            self.indices,
        )
        if any(value.device != reference_device for value in tensors):
            raise ValueError("RecallTraceV2 tensors must use one device")
        if self.route.ndim < 4 or self.route.shape[:3] != expected:
            raise ValueError("route must begin with [B, R, N]")
        if self.indices.ndim < 4 or self.indices.shape[:3] != expected:
            raise ValueError("indices must begin with [B, R, N]")
        if self.indices.dtype != torch.int64:
            raise TypeError("indices must be int64")
        if any(step <= 0 or step > refine_steps for step in self.checkpoints):
            raise ValueError("checkpoint keys must be executed-depth indices")
        if any(not isinstance(state, Tensor) for state in self.checkpoints.values()):
            raise TypeError("checkpoint values must be Tensors")
        object.__setattr__(self, "checkpoints", MappingProxyType(dict(self.checkpoints)))

    def validate(self) -> "RecallTraceV2":
        """Run value-dependent integrity checks outside compiled execution."""

        if not torch.equal(
            self.token_steps_attempted,
            self.token_step_attempted.sum(dim=1, dtype=torch.int64),
        ):
            raise ValueError("token_steps_attempted must equal attempted token-step counts")
        if not torch.equal(
            self.token_steps_committed,
            self.token_step_committed.sum(dim=1, dtype=torch.int64),
        ):
            raise ValueError("token_steps_committed must equal committed token-step counts")
        if bool(torch.any(self.token_step_committed & ~self.token_step_attempted)):
            raise ValueError("a committed token-step must also be attempted")
        if self.max_steps > 1:
            reactivated = self.token_step_attempted[:, 1:] & ~self.token_step_attempted[:, :-1]
            if bool(torch.any(reactivated)):
                raise ValueError("token activity must be monotonically non-increasing")

        valid_reasons = torch.tensor(
            [int(value) for value in RecallStopReason],
            device=self.token_stop_reason.device,
            dtype=self.token_stop_reason.dtype,
        )
        if not bool(torch.isin(self.token_stop_reason, valid_reasons).all()):
            raise ValueError("token_stop_reason contains an unknown reason code")
        if bool(torch.any(self.token_steps_committed > self.token_steps_attempted)):
            raise ValueError("committed token-step counts cannot exceed attempted counts")

        expected_logical = self.token_steps_committed.sum(dtype=torch.int64)
        if not torch.equal(self.logical_token_steps, expected_logical):
            raise ValueError("logical_token_steps must equal committed token-step count")
        expected_active = self.token_step_attempted.to(self.active_fraction.dtype).mean(dim=(0, 2))
        if not torch.allclose(self.active_fraction, expected_active, rtol=1e-6, atol=1e-7):
            raise ValueError("active_fraction must equal attempted-token fraction per step")
        if (
            int(self.kernel_steps.detach().cpu()) < 0
            or int(self.kernel_steps.detach().cpu()) > self.max_steps
        ):
            raise ValueError("kernel_steps must be between zero and max_steps")
        return self

    @property
    def max_steps(self) -> int:
        return int(self.token_step_attempted.shape[1])

    @property
    def batch_size(self) -> int:
        return int(self.token_step_attempted.shape[0])

    @property
    def token_count(self) -> int:
        return int(self.token_step_attempted.shape[2])

    @property
    def sample_steps_attempted(self) -> Tensor:
        """Number of loop depths reached by at least one token per sample."""

        return self.token_step_attempted.any(dim=2).sum(dim=1, dtype=torch.int64)

    @property
    def sample_steps_committed(self) -> Tensor:
        """Number of loop depths committing at least one token per sample."""

        return self.token_step_committed.any(dim=2).sum(dim=1, dtype=torch.int64)

    @property
    def sample_step_attempted(self) -> Tensor:
        return self.token_step_attempted.any(dim=2)

    @property
    def sample_step_committed(self) -> Tensor:
        return self.token_step_committed.any(dim=2)

    def diagnostics(self) -> dict[str, Tensor]:
        """Return tensor-only diagnostics without collapsing token evidence."""

        result = {
            "recall_trace_schema": torch.tensor(
                self.schema_version,
                device=self.token_steps_attempted.device,
                dtype=torch.int64,
            ),
            "recall_token_steps_attempted": self.token_steps_attempted,
            "recall_token_steps_committed": self.token_steps_committed,
            "recall_token_stop_reason": self.token_stop_reason,
            "recall_token_step_attempted": self.token_step_attempted,
            "recall_token_step_committed": self.token_step_committed,
            "recall_token_step_update_ratio": self.state_change_ratio,
            "recall_token_step_route_change": self.route_change,
            "recall_token_step_effective_read_change": self.effective_read_change,
            "recall_active_fraction": self.active_fraction,
            "recall_kernel_steps": self.kernel_steps,
            "recall_logical_token_steps": self.logical_token_steps,
            "recall_route_history": self.route,
            "recall_index_history": self.indices,
            "recall_steps_attempted": self.sample_steps_attempted,
            "recall_steps_committed": self.sample_steps_committed,
            "recall_step_attempted": self.sample_step_attempted,
            "recall_step_committed": self.sample_step_committed,
        }
        for step, state in self.checkpoints.items():
            result[f"recall_checkpoint_{step}"] = state
        return result

    @classmethod
    def from_diagnostics(
        cls,
        values: Mapping[str, Tensor],
        *,
        validate_schema: bool = True,
        validate_values: bool = True,
    ) -> "RecallTraceV2":
        """Reconstruct a token-resolved trace from tensor diagnostics."""

        schema = values.get("recall_trace_schema")
        if validate_schema and (
            not isinstance(schema, Tensor)
            or schema.numel() != 1
            or int(schema.detach().cpu()) != RECALL_TRACE_V2_SCHEMA_VERSION
        ):
            raise ValueError("Recall diagnostics use an unsupported trace schema")
        required = {
            "recall_token_steps_attempted",
            "recall_token_steps_committed",
            "recall_token_stop_reason",
            "recall_token_step_attempted",
            "recall_token_step_committed",
            "recall_token_step_update_ratio",
            "recall_token_step_route_change",
            "recall_token_step_effective_read_change",
            "recall_active_fraction",
            "recall_kernel_steps",
            "recall_logical_token_steps",
            "recall_route_history",
            "recall_index_history",
        }
        missing = required - set(values)
        if missing:
            raise ValueError(f"Recall diagnostics are missing trace fields: {sorted(missing)}")
        prefix = "recall_checkpoint_"
        checkpoints = {
            int(name[len(prefix) :]): value
            for name, value in values.items()
            if name.startswith(prefix)
        }
        trace = cls(
            token_steps_attempted=values["recall_token_steps_attempted"],
            token_steps_committed=values["recall_token_steps_committed"],
            token_stop_reason=values["recall_token_stop_reason"],
            token_step_attempted=values["recall_token_step_attempted"],
            token_step_committed=values["recall_token_step_committed"],
            state_change_ratio=values["recall_token_step_update_ratio"],
            route_change=values["recall_token_step_route_change"],
            effective_read_change=values["recall_token_step_effective_read_change"],
            active_fraction=values["recall_active_fraction"],
            kernel_steps=values["recall_kernel_steps"],
            logical_token_steps=values["recall_logical_token_steps"],
            route=values["recall_route_history"],
            indices=values["recall_index_history"],
            checkpoints=checkpoints,
        )
        return trace.validate() if validate_values else trace


__all__ = [
    "RECALL_TRACE_SCHEMA_VERSION",
    "RECALL_TRACE_V2_SCHEMA_VERSION",
    "AdaptiveRefinePolicy",
    "RecallRoutePlan",
    "RecallRouteStack",
    "RecallStopReason",
    "RecallTrace",
    "RecallTraceV2",
    "RefineBudget",
    "RefinePolicy",
    "RefineStop",
]
