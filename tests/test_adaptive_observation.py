from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import Tensor, nn

import arti
from arti.alpha import (
    AdaptiveObservation,
    DEFAULT_CONTRACT_LIMITS,
    EnvelopeRef,
    FixedObservationPolicy,
    LearnedObservationPolicy,
    ObservationPlan,
    PulseSupports,
    SupportDomain,
    SupportKind,
    SupportMask,
    StateAffineObservationOperator,
    TensorEnvelope,
    lift_observation_supports,
)
from arti.component_registry import component_spec


class AddObservationState(nn.Module):
    def forward(self, substrate: Tensor, state: Tensor) -> Tensor:
        return substrate + state[:, :1].unsqueeze(-1)


class CountingObservationOperator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, substrate: Tensor, _state: Tensor) -> Tensor:
        self.calls += 1
        return substrate


def world_envelope(value: Tensor, mask: Tensor) -> TensorEnvelope:
    domain = SupportDomain.for_tensor(
        mask,
        domain_id="observation-world",
        owner_ref="arti/adaptive-observation@1",
        partition_id="world",
        transition_id="observation-transition",
    )
    return TensorEnvelope(EnvelopeRef.WORLD, value, mask, domain)


def test_each_observation_reads_the_original_substrate() -> None:
    value = torch.tensor([[[2.0], [5.0]]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    observation = AdaptiveObservation(
        FixedObservationPolicy(torch.tensor([[0.0], [1.0], [2.0]])),
        operator=AddObservationState(),
    )

    result, plan = observation(world_envelope(value, mask), return_info=True)

    expected = torch.stack((value, value + 1.0, value + 2.0), dim=1)
    torch.testing.assert_close(result.value, expected, rtol=0, atol=0)
    assert plan.mask.all()
    assert result.ref is EnvelopeRef.OBSERVATION
    assert result.domain.shape == (1, 3, 2)


def test_inactive_and_invalid_observations_have_zero_value_and_gradient() -> None:
    value = torch.randn(1, 3, 2, requires_grad=True)
    validity = torch.tensor([[True, False, True]])
    observation = AdaptiveObservation(
        FixedObservationPolicy(
            torch.tensor([[0.0], [1.0], [2.0]]),
            active_count=2,
        ),
        operator=AddObservationState(),
    )

    result = observation(world_envelope(value, validity))
    assert not result.mask[:, 2].any()
    assert not result.value[:, 2].any()
    assert not result.value[:, :, 1].any()
    result.value.sum().backward()
    expected_grad = validity.unsqueeze(-1).expand_as(value).to(value) * 2.0
    torch.testing.assert_close(value.grad, expected_grad, rtol=0, atol=0)


def test_external_plan_is_bounded_and_batch_checked() -> None:
    value = torch.randn(2, 3, 4)
    mask = torch.ones(2, 3, dtype=torch.bool)
    observation = AdaptiveObservation(FixedObservationPolicy(torch.ones(1, 1)))
    wrong_batch = ObservationPlan(
        states=torch.ones(1, 2, 1),
        mask=torch.ones(1, 2, dtype=torch.bool),
    )
    with pytest.raises(ValueError, match="batch"):
        observation(world_envelope(value, mask), plan=wrong_batch)

    tiny = replace(DEFAULT_CONTRACT_LIMITS, max_elements=8)
    bounded = AdaptiveObservation(
        FixedObservationPolicy(torch.ones(2, 1)),
        limits=tiny,
    )
    with pytest.raises(ValueError, match="max_elements"):
        bounded(world_envelope(value, mask))


def test_operator_must_preserve_substrate_shape() -> None:
    class BadOperator(nn.Module):
        def forward(self, substrate: Tensor, _state: Tensor) -> Tensor:
            return substrate[..., :1]

    value = torch.randn(1, 2, 3)
    mask = torch.ones(1, 2, dtype=torch.bool)
    observation = AdaptiveObservation(
        FixedObservationPolicy(torch.ones(1, 1)),
        operator=BadOperator(),
    )
    with pytest.raises(ValueError, match="preserve substrate shape"):
        observation(world_envelope(value, mask))


def test_observation_rejects_non_sequence_world_layout() -> None:
    value = torch.randn(2, 4)
    mask = torch.ones(2, dtype=torch.bool)
    observation = AdaptiveObservation(FixedObservationPolicy(torch.ones(1, 1)))

    with pytest.raises(ValueError, match=r"\[B, N, D\]"):
        observation(world_envelope(value, mask))


def test_observation_component_config_binds_policy_and_operator() -> None:
    first = AdaptiveObservation(FixedObservationPolicy(torch.ones(2, 1)))
    second = AdaptiveObservation(FixedObservationPolicy(torch.ones(3, 2)))
    first_spec = component_spec(first)
    second_spec = component_spec(second)
    assert first_spec.reference == "arti/adaptive-observation@1"
    assert first_spec.config_fingerprint != second_spec.config_fingerprint
    assert set(first_spec.dependencies) == {
        "arti/fixed-observation-policy@1",
        "arti/identity-observation-operator@1",
    }
    assert arti.component_ref(first.policy) == "arti/fixed-observation-policy@1"


def test_support_lift_repeats_world_support_and_intersects_observation_mask() -> None:
    value = torch.randn(1, 4, 2)
    validity = torch.tensor([[True, True, False, True]])
    world = world_envelope(value, validity)
    domain = world.domain
    supports = PulseSupports(
        SupportMask(SupportKind.OBSERVED, validity, domain),
        SupportMask(
            SupportKind.EXPOSED,
            torch.tensor([[True, False, False, True]]),
            domain,
        ),
        SupportMask(
            SupportKind.INTERVENED,
            torch.tensor([[False, False, False, True]]),
            domain,
        ),
        validity=validity,
    )
    observation = AdaptiveObservation(
        FixedObservationPolicy(torch.ones(3, 1), active_count=2)
    )(world)

    lifted = lift_observation_supports(supports, observation)

    assert lifted.validity.shape == (1, 3, 4)
    assert torch.equal(lifted.validity, observation.mask)
    assert torch.equal(
        lifted.exposed.mask,
        supports.exposed.mask.unsqueeze(1) & observation.mask,
    )
    assert torch.equal(
        lifted.intervened.mask,
        supports.intervened.mask.unsqueeze(1) & observation.mask,
    )
    assert lifted.observed.domain == observation.domain


def test_support_lift_rejects_observation_with_changed_instance_layout() -> None:
    value = torch.randn(1, 4, 2)
    validity = torch.ones(1, 4, dtype=torch.bool)
    world = world_envelope(value, validity)
    supports = PulseSupports.identity(validity, world.domain)
    wrong_value = torch.randn(1, 2, 3, 2)
    wrong_mask = torch.ones(1, 2, 3, dtype=torch.bool)
    wrong_domain = SupportDomain.for_tensor(
        wrong_mask,
        domain_id="wrong-observation",
        owner_ref="arti/adaptive-observation@1",
        partition_id="observations",
        transition_id=world.domain.transition_id,
    )
    wrong = TensorEnvelope(
        EnvelopeRef.OBSERVATION,
        wrong_value,
        wrong_mask,
        wrong_domain,
    )

    with pytest.raises(ValueError, match="preserve world instance layout"):
        lift_observation_supports(supports, wrong)


def input_conditioned_policy() -> LearnedObservationPolicy:
    policy = LearnedObservationPolicy(
        input_dim=2,
        state_dim=1,
        hidden_dim=2,
        max_observations=4,
        stop_threshold=0.5,
    )
    policy.context = nn.Identity()
    with torch.no_grad():
        policy.initial_state.weight.copy_(torch.tensor([[1.0, 0.0]]))
        policy.initial_state.bias.zero_()
        policy.continuation.weight.fill_(4.0)
        policy.continuation.bias.zero_()
        for parameter in policy.recurrent.parameters():
            parameter.zero_()
    return policy


def test_learned_policy_uses_input_conditioned_per_sample_counts() -> None:
    policy = input_conditioned_policy().eval()
    substrate = torch.tensor([[[-1.0, 0.0]], [[1.0, 0.0]]])
    mask = torch.ones(2, 1, dtype=torch.bool)

    first = policy(substrate, mask)
    second = policy(substrate, mask)

    assert first.mask.sum(dim=1).tolist() == [1, 4]
    assert torch.equal(first.mask, second.mask)
    torch.testing.assert_close(first.states, second.states, rtol=0, atol=0)
    assert first.weights is not None
    assert torch.equal(first.weights, first.mask.to(first.states.dtype))


def test_state_affine_operator_is_identity_at_zero_and_state_conditioned() -> None:
    operator = StateAffineObservationOperator(3, 2, scale=0.2)
    substrate = torch.randn(2, 4, 3, requires_grad=True)
    zero = torch.zeros(2, 2)
    state = torch.randn(2, 2, requires_grad=True)

    identity = operator(substrate, zero)
    changed = operator(substrate, state)

    torch.testing.assert_close(identity, substrate, rtol=0, atol=0)
    assert not torch.equal(changed, substrate)
    changed.square().mean().backward()
    assert state.grad is not None and state.grad.abs().sum() > 0
    assert arti.component_ref(operator) == "arti/state-affine-observation-operator@1"


def test_downstream_loss_reaches_learned_continuation_controller() -> None:
    policy = LearnedObservationPolicy(2, 2, 3, stop_threshold=0.5)
    observation = AdaptiveObservation(policy)
    value = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], requires_grad=True)
    mask = torch.ones(1, 2, dtype=torch.bool)

    result = observation(world_envelope(value, mask))
    result.value.sum().backward()

    assert policy.continuation.weight.grad is not None
    assert policy.continuation.weight.grad.abs().sum() > 0
    assert value.grad is not None and torch.isfinite(value.grad).all()


def test_early_break_skips_inactive_operator_calls_but_keeps_static_shape() -> None:
    operator = CountingObservationOperator()
    observation = AdaptiveObservation(
        FixedObservationPolicy(torch.ones(5, 1), active_count=2),
        operator=operator,
        executor="early_break",
    )
    value = torch.randn(2, 3, 4)
    mask = torch.ones(2, 3, dtype=torch.bool)

    result = observation(world_envelope(value, mask))

    assert operator.calls == 2
    assert result.value.shape == (2, 5, 3, 4)
    assert not result.value[:, 2:].any()


def test_early_break_rejects_non_prefix_external_plan() -> None:
    observation = AdaptiveObservation(
        FixedObservationPolicy(torch.ones(3, 1)), executor="early_break"
    )
    value = torch.randn(1, 2, 3)
    mask = torch.ones(1, 2, dtype=torch.bool)
    plan = ObservationPlan(
        torch.ones(1, 3, 1),
        torch.tensor([[True, False, True]]),
    )

    with pytest.raises(ValueError, match="prefix-contiguous"):
        observation(world_envelope(value, mask), plan=plan)


def test_learned_policy_arti_st_round_trip_and_component_identity(
    tmp_path: Path,
) -> None:
    source = LearnedObservationPolicy(4, 3, 5, hidden_dim=7, min_observations=2)
    target = LearnedObservationPolicy(4, 3, 5, hidden_dim=7, min_observations=2)
    source_observation = AdaptiveObservation(source)
    target_observation = AdaptiveObservation(target)
    saved = arti.save(source_observation, tmp_path / "observation.arti.st")
    arti.load(saved.weights_path, model=target_observation)
    value = torch.randn(2, 6, 4)
    mask = torch.tensor(
        [[True, True, True, False, False, False], [True] * 6]
    )

    expected = source(value, mask)
    actual = target(value, mask)

    assert arti.component_ref(source) == "arti/learned-observation-policy@1"
    assert torch.equal(expected.mask, actual.mask)
    torch.testing.assert_close(expected.states, actual.states, rtol=0, atol=0)
    assert expected.weights is not None and actual.weights is not None
    torch.testing.assert_close(expected.weights, actual.weights, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_learned_policy_cuda_compile_matches_eager_and_backpropagates() -> None:
    eager = LearnedObservationPolicy(4, 3, 4).cuda().train()
    compiled = torch.compile(eager, fullgraph=True)
    value = torch.randn(3, 8, 4, device="cuda", requires_grad=True)
    mask = torch.ones(3, 8, dtype=torch.bool, device="cuda")

    expected = eager(value, mask)
    actual = compiled(value, mask)

    assert torch.equal(expected.mask, actual.mask)
    torch.testing.assert_close(expected.states, actual.states)
    assert expected.weights is not None and actual.weights is not None
    torch.testing.assert_close(expected.weights, actual.weights)
    (actual.states.sum() + actual.weights.sum()).backward()
    assert eager.continuation.weight.grad is not None
    assert torch.isfinite(eager.continuation.weight.grad).all()
