"""Adaptive overcomplete observation over one immutable tensor substrate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import Tensor, nn

from .runtime_contracts import (
    ContractLimits,
    DEFAULT_CONTRACT_LIMITS,
    EnvelopeRef,
    SupportDomain,
    TensorEnvelope,
)


@dataclass(frozen=True)
class ObservationPlan:
    """Bounded observation states and their logical activity mask."""

    states: Tensor
    mask: Tensor
    weights: Tensor | None = None

    def __post_init__(self) -> None:
        if self.states.ndim != 3:
            raise ValueError("observation states must have shape [B, T_max, S]")
        if self.mask.shape != self.states.shape[:2] or self.mask.dtype != torch.bool:
            raise ValueError("observation mask must be boolean with shape [B, T_max]")
        if self.states.device != self.mask.device or not self.states.is_floating_point():
            raise ValueError("observation states and mask must share device and use floating states")
        if self.weights is not None:
            if self.weights.shape != self.mask.shape:
                raise ValueError("observation weights must have shape [B, T_max]")
            if (
                self.weights.device != self.states.device
                or not self.weights.is_floating_point()
            ):
                raise ValueError(
                    "observation weights must share device and use floating values"
                )

    def activity_weights(self) -> Tensor:
        """Return hard-forward activity with an optional differentiable backward path."""

        if self.weights is None:
            return self.mask.to(self.states.dtype)
        return self.weights


class FixedObservationPolicy(nn.Module):
    """Batch-shared bounded observation trajectory for reference and testing."""

    _component_reference: ClassVar[str] = "arti/fixed-observation-policy@1"

    def __init__(
        self,
        states: Tensor,
        *,
        active_count: int | None = None,
        learnable: bool = False,
    ) -> None:
        super().__init__()
        value = torch.as_tensor(states, dtype=torch.float32)
        if value.ndim != 2 or value.shape[0] <= 0 or value.shape[1] <= 0:
            raise ValueError("states must have shape [T_max, state_dim]")
        count = value.shape[0] if active_count is None else int(active_count)
        if count <= 0 or count > value.shape[0]:
            raise ValueError("active_count must be in [1, T_max]")
        self.active_count = count
        self.max_observations = int(value.shape[0])
        self.state_dim = int(value.shape[1])
        if learnable:
            self.states = nn.Parameter(value.clone())
        else:
            self.register_buffer("states", value.clone(), persistent=True)

    def forward(self, substrate: Tensor, _mask: Tensor) -> ObservationPlan:
        batch = substrate.shape[0]
        states = self.states.to(substrate).unsqueeze(0).expand(batch, -1, -1)
        indices = torch.arange(self.max_observations, device=substrate.device)
        active = (indices < self.active_count).unsqueeze(0).expand(batch, -1)
        return ObservationPlan(states, active)


class IdentityObservationOperator(nn.Module):
    """Reference operator that observes the unchanged substrate at every state."""

    _component_reference: ClassVar[str] = "arti/identity-observation-operator@1"

    def forward(self, substrate: Tensor, _state: Tensor) -> Tensor:
        return substrate

    def forward_trajectory(self, substrate: Tensor, states: Tensor) -> Tensor:
        return identity_observation(substrate, states)


def identity_observation(substrate: Tensor, states: Tensor) -> Tensor:
    """Batch the identity views without copying the substrate before masking."""
    return substrate.unsqueeze(1).expand(-1, states.shape[1], -1, -1)


def state_affine_observation(
    substrate: Tensor, states: Tensor, weight: Tensor, bias: Tensor, *, scale: float
) -> Tensor:
    """Observe the original substrate independently at every trajectory state."""
    raw_scale, raw_shift = torch.nn.functional.linear(states, weight, bias).chunk(2, dim=-1)
    feature_scale = 1.0 + scale * torch.tanh(raw_scale)
    feature_shift = scale * torch.tanh(raw_shift)
    return substrate.unsqueeze(1) * feature_scale.unsqueeze(2) + feature_shift.unsqueeze(2)


def apply_observation_activity(
    values: Tensor, substrate_mask: Tensor, activity: Tensor
) -> Tensor:
    """Keep activity's surrogate gradient separate from the boolean output mask."""
    weighted = values * activity.to(values).unsqueeze(-1).unsqueeze(-1)
    return torch.where(substrate_mask[:, None, :, None], weighted, torch.zeros_like(weighted))


class StateAffineObservationOperator(nn.Module):
    """Observe one substrate through a bounded state-conditioned feature frame."""

    _component_reference: ClassVar[str] = "arti/state-affine-observation-operator@1"

    def __init__(self, dim: int, state_dim: int, *, scale: float = 0.1) -> None:
        super().__init__()
        if dim <= 0 or state_dim <= 0:
            raise ValueError("dim and state_dim must be positive")
        if scale <= 0.0:
            raise ValueError("scale must be positive")
        self.dim = int(dim)
        self.state_dim = int(state_dim)
        self.scale = float(scale)
        self.projection = nn.Linear(self.state_dim, self.dim * 2)
        nn.init.normal_(self.projection.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.projection.bias)

    def forward(self, substrate: Tensor, state: Tensor) -> Tensor:
        if substrate.ndim != 3 or substrate.shape[-1] != self.dim:
            raise ValueError(f"substrate must have shape [B, N, {self.dim}]")
        if state.ndim != 2 or state.shape != (substrate.shape[0], self.state_dim):
            raise ValueError(f"state must have shape [B, {self.state_dim}]")
        return self.forward_trajectory(substrate, state.unsqueeze(1)).squeeze(1)

    def forward_trajectory(self, substrate: Tensor, states: Tensor) -> Tensor:
        if substrate.ndim != 3 or substrate.shape[-1] != self.dim:
            raise ValueError(f"substrate must have shape [B, N, {self.dim}]")
        if states.ndim != 3 or states.shape[0] != substrate.shape[0] or states.shape[-1] != self.state_dim:
            raise ValueError(f"states must have shape [B, T, {self.state_dim}]")
        return state_affine_observation(
            substrate, states, self.projection.weight, self.projection.bias, scale=self.scale
        )


def _native_fourier_shift(
    image: Tensor,
    dx: Tensor,
    dy: Tensor,
) -> Tensor:
    height, width = image.shape[1:3]
    compute_dtype = image.dtype
    fy = torch.fft.fftfreq(height, device=image.device, dtype=compute_dtype)
    fx = torch.fft.fftfreq(width, device=image.device, dtype=compute_dtype)
    phase_angle = (
        dy.unsqueeze(-1).unsqueeze(-1) * fy.view(1, 1, height, 1)
        + dx.unsqueeze(-1).unsqueeze(-1) * fx.view(1, 1, 1, width)
    )
    complex_dtype = (
        torch.complex128 if compute_dtype == torch.float64 else torch.complex64
    )
    phase = torch.exp((-2j * torch.pi * phase_angle).to(complex_dtype))
    spectrum = torch.fft.fftn(image, dim=(1, 2))
    return torch.fft.ifftn(
        spectrum.unsqueeze(1) * phase.unsqueeze(-1), dim=(2, 3)
    ).real


_safe_native_fourier_shift = torch.compiler.disable(_native_fourier_shift)


def fourier_observation(
    substrate: Tensor,
    states: Tensor,
    *,
    spatial_shape: tuple[int, int],
    state_mode: str,
    direction_epsilon: float,
    compile_policy: str,
) -> Tensor:
    """Shared torch.fft implementation for module and Fabric execution."""
    height, width = spatial_shape
    compute_dtype = torch.float64 if substrate.dtype == torch.float64 else torch.float32
    image = substrate.to(compute_dtype).reshape(
        substrate.shape[0], height, width, substrate.shape[-1]
    )
    state = states.to(compute_dtype)
    if state_mode == "cartesian":
        dx, dy = state[..., 0], state[..., 1]
    else:
        radius, direction = state[..., 0], state[..., 1:3]
        norm = torch.linalg.vector_norm(direction, dim=-1)
        unit = direction / norm.clamp_min(direction_epsilon).unsqueeze(-1)
        unit = torch.where((norm > direction_epsilon).unsqueeze(-1), unit, torch.zeros_like(unit))
        dx, dy = radius * unit[..., 0], radius * unit[..., 1]
    shift = _safe_native_fourier_shift if compile_policy == "safe_training" else _native_fourier_shift
    return shift(image, dx, dy).reshape(
        substrate.shape[0], states.shape[1], height * width, substrate.shape[-1]
    ).to(substrate.dtype)


class FourierShiftObservationOperator(nn.Module):
    """Circular subpixel translation from Cartesian or polar displacement state."""

    _component_reference: ClassVar[str] = "arti/fourier-observation-operator@1"

    def __init__(
        self,
        spatial_shape: tuple[int, int],
        *,
        state_mode: str = "cartesian",
        direction_epsilon: float = 1e-6,
        compile_policy: str = "safe_training",
    ) -> None:
        super().__init__()
        if (
            len(spatial_shape) != 2
            or spatial_shape[0] <= 0
            or spatial_shape[1] <= 0
        ):
            raise ValueError("spatial_shape must contain positive (height, width)")
        if state_mode not in {"cartesian", "polar"}:
            raise ValueError("state_mode must be 'cartesian' or 'polar'")
        if direction_epsilon <= 0.0:
            raise ValueError("direction_epsilon must be positive")
        if compile_policy not in {"safe_training", "fullgraph_forward"}:
            raise ValueError(
                "compile_policy must be 'safe_training' or 'fullgraph_forward'"
            )
        self.spatial_shape = (int(spatial_shape[0]), int(spatial_shape[1]))
        self.state_mode = state_mode
        self.direction_epsilon = float(direction_epsilon)
        self.compile_policy = compile_policy
        self.boundary = "circular"
        self.state_dim = 2 if state_mode == "cartesian" else 3

    def execution_contract(self) -> dict[str, object]:
        return {
            "backend": "torch.fft",
            "compile_policy": self.compile_policy,
            "compiled_outer_graph_training": self.compile_policy == "safe_training",
            "compiled_fullgraph_forward": self.compile_policy == "fullgraph_forward",
            "compiled_fullgraph_backward": False,
            "eager_backward": True,
        }

    def forward_trajectory(self, substrate: Tensor, states: Tensor) -> Tensor:
        height, width = self.spatial_shape
        if substrate.ndim != 3 or substrate.shape[1] != height * width:
            raise ValueError(
                f"substrate must have shape [B, {height * width}, D]"
            )
        if (
            states.ndim != 3
            or states.shape[0] != substrate.shape[0]
            or states.shape[-1] != self.state_dim
        ):
            raise ValueError(
                f"states must have shape [B, T, {self.state_dim}]"
            )
        return fourier_observation(
            substrate, states, spatial_shape=self.spatial_shape, state_mode=self.state_mode,
            direction_epsilon=self.direction_epsilon, compile_policy=self.compile_policy,
        )

    def forward(self, substrate: Tensor, state: Tensor) -> Tensor:
        if state.ndim != 2:
            raise ValueError("state must have shape [B, state_dim]")
        return self.forward_trajectory(substrate, state.unsqueeze(1)).squeeze(1)


class LearnedObservationPolicy(nn.Module):
    """Input-conditioned recurrent observation states with learned bounded halting.

    The boolean mask is the deterministic forward decision. ``weights`` has the
    same hard 0/1 forward value and carries a straight-through gradient to the
    continuation controller, so downstream losses can shape trajectory length.
    """

    _component_reference: ClassVar[str] = "arti/learned-observation-policy@1"

    def __init__(
        self,
        input_dim: int,
        state_dim: int,
        max_observations: int,
        *,
        hidden_dim: int | None = None,
        min_observations: int = 1,
        stop_threshold: float = 0.5,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or state_dim <= 0 or max_observations <= 0:
            raise ValueError("input_dim, state_dim, and max_observations must be positive")
        if not 1 <= min_observations <= max_observations:
            raise ValueError("min_observations must be in [1, max_observations]")
        if not 0.0 < stop_threshold < 1.0:
            raise ValueError("stop_threshold must be in (0, 1)")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        width = state_dim if hidden_dim is None else int(hidden_dim)
        if width <= 0:
            raise ValueError("hidden_dim must be positive")

        self.input_dim = int(input_dim)
        self.state_dim = int(state_dim)
        self.hidden_dim = width
        self.max_observations = int(max_observations)
        self.min_observations = int(min_observations)
        self.stop_threshold = float(stop_threshold)
        self.temperature = float(temperature)
        self.context = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.initial_state = nn.Linear(self.hidden_dim, self.state_dim)
        self.recurrent = nn.GRUCell(self.hidden_dim, self.state_dim)
        self.continuation = nn.Linear(self.state_dim, 1)

    def forward(self, substrate: Tensor, mask: Tensor) -> ObservationPlan:
        if substrate.ndim != 3 or substrate.shape[-1] != self.input_dim:
            raise ValueError(
                f"substrate must have shape [B, N, {self.input_dim}]"
            )
        if mask.shape != substrate.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("mask must be boolean with shape [B, N]")
        valid = mask.unsqueeze(-1).to(substrate.dtype)
        pooled = (substrate * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
        context = self.context(pooled)
        state = torch.tanh(self.initial_state(context))
        batch = substrate.shape[0]
        alive_hard = torch.ones(batch, dtype=torch.bool, device=substrate.device)
        alive_soft = torch.ones(batch, dtype=substrate.dtype, device=substrate.device)
        states: list[Tensor] = []
        masks: list[Tensor] = []
        weights: list[Tensor] = []

        for index in range(self.max_observations):
            if index > 0:
                state = self.recurrent(context, state)
            states.append(state)
            masks.append(alive_hard)
            hard_weight = alive_hard.to(substrate.dtype)
            weights.append(hard_weight + alive_soft - alive_soft.detach())
            if index + 1 >= self.max_observations:
                continue
            probability = torch.sigmoid(
                self.continuation(state).squeeze(-1) / self.temperature
            )
            alive_soft = alive_soft * probability
            if index + 1 >= self.min_observations:
                alive_hard = alive_hard & (probability >= self.stop_threshold)

        return ObservationPlan(
            torch.stack(states, dim=1),
            torch.stack(masks, dim=1),
            torch.stack(weights, dim=1),
        )


class AdaptiveObservation(nn.Module):
    """Produce a bounded logical observation trajectory from one fixed substrate."""

    _component_reference: ClassVar[str] = "arti/adaptive-observation@1"

    def __init__(
        self,
        policy: nn.Module,
        *,
        operator: nn.Module | None = None,
        executor: str = "static_masked",
        limits: ContractLimits = DEFAULT_CONTRACT_LIMITS,
    ) -> None:
        super().__init__()
        if not isinstance(policy, nn.Module):
            raise TypeError("observation policy must be an nn.Module")
        if operator is not None and not isinstance(operator, nn.Module):
            raise TypeError("observation operator must be an nn.Module")
        if executor not in {"static_masked", "early_break"}:
            raise ValueError("executor must be 'static_masked' or 'early_break'")
        if limits is not DEFAULT_CONTRACT_LIMITS:
            for name, hard_value in DEFAULT_CONTRACT_LIMITS.__dict__.items():
                if getattr(limits, name) > hard_value:
                    raise ValueError(f"observation limits cannot relax hard ceiling {name}")
        self.policy = policy
        self.operator = IdentityObservationOperator() if operator is None else operator
        self.executor = executor
        self.limits = limits

    def forward(
        self,
        world: TensorEnvelope,
        *,
        plan: ObservationPlan | None = None,
        return_info: bool = False,
    ) -> TensorEnvelope | tuple[TensorEnvelope, ObservationPlan]:
        if not isinstance(world, TensorEnvelope) or world.ref is not EnvelopeRef.WORLD:
            raise TypeError("AdaptiveObservation expects a WORLD TensorEnvelope")
        if world.value.ndim != 3:
            raise ValueError("AdaptiveObservation expects WORLD values with shape [B, N, D]")
        trajectory = self.policy(world.value, world.mask) if plan is None else plan
        if not isinstance(trajectory, ObservationPlan):
            raise TypeError("observation policy must return ObservationPlan")
        if trajectory.states.shape[0] != world.value.shape[0]:
            raise ValueError("observation plan batch does not match the substrate")
        if trajectory.states.device != world.value.device:
            raise ValueError("observation plan and substrate must share a device")
        if self.executor == "early_break" and not torch.compiler.is_compiling():
            inactive_seen = (~trajectory.mask).cumsum(dim=1) > 0
            if bool((trajectory.mask & inactive_seen).any()):
                raise ValueError("early_break requires a prefix-contiguous observation mask")
        max_observations = trajectory.states.shape[1]
        output_elements = world.value.numel() * max_observations
        if output_elements > self.limits.max_elements:
            raise ValueError("observation output exceeds max_elements")
        output_bytes = output_elements * world.value.element_size()
        if output_bytes > self.limits.max_operation_bytes:
            raise ValueError("observation output exceeds max_operation_bytes")

        trajectory_operator = getattr(self.operator, "forward_trajectory", None)
        if callable(trajectory_operator):
            executed_observations = max_observations
            if self.executor == "early_break" and not torch.compiler.is_compiling():
                executed_observations = int(
                    trajectory.mask.any(dim=0).sum().detach().cpu().item()
                )
            values = trajectory_operator(
                world.value, trajectory.states[:, :executed_observations]
            )
            if executed_observations < max_observations:
                padding = torch.zeros(
                    world.value.shape[0],
                    max_observations - executed_observations,
                    *world.value.shape[1:],
                    device=world.value.device,
                    dtype=world.value.dtype,
                )
                values = torch.cat((values, padding), dim=1)
            expected_shape = (
                world.value.shape[0],
                max_observations,
                *world.value.shape[1:],
            )
            if not isinstance(values, Tensor) or values.shape != expected_shape:
                raise ValueError(
                    "trajectory observation operator must preserve substrate shape"
                )
        else:
            observations = []
            stopped = False
            for index in range(max_observations):
                if (
                    self.executor == "early_break"
                    and not torch.compiler.is_compiling()
                    and not bool(trajectory.mask[:, index].any())
                ):
                    stopped = True
                if stopped:
                    observations.append(torch.zeros_like(world.value))
                    continue
                # Every step observes the original substrate, never observations[-1].
                observed = self.operator(world.value, trajectory.states[:, index])
                if not isinstance(observed, Tensor) or observed.shape != world.value.shape:
                    raise ValueError("observation operator must preserve substrate shape")
                observations.append(observed)
            values = torch.stack(observations, dim=1)
        mask = trajectory.mask.unsqueeze(-1) & world.mask.unsqueeze(1)
        values = apply_observation_activity(values, world.mask, trajectory.activity_weights())
        observed_domain = SupportDomain(
            domain_id=f"{world.domain.domain_id}-observed",
            owner_ref=self._component_reference,
            partition_id="observations",
            transition_id=world.domain.transition_id,
            layout="dense",
            shape=tuple(mask.shape),
            device=str(values.device),
            axis=-2,
        )
        result = TensorEnvelope(
            EnvelopeRef.OBSERVATION,
            values,
            mask,
            observed_domain,
            limits=self.limits,
        )
        return (result, trajectory) if return_info else result


__all__ = [
    "AdaptiveObservation",
    "FixedObservationPolicy",
    "FourierShiftObservationOperator",
    "IdentityObservationOperator",
    "LearnedObservationPolicy",
    "ObservationPlan",
    "StateAffineObservationOperator",
]
