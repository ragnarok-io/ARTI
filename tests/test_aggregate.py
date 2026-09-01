from __future__ import annotations

import pytest
import torch

import arti
from arti.mechanisms import (
    EnvelopeRef,
    ReunionAggregate,
    SoftFoldAggregate,
    SupportDomain,
    TensorEnvelope,
)
from arti.component_registry import component_spec


def envelope(
    value: torch.Tensor,
    mask: torch.Tensor,
    ref: EnvelopeRef = EnvelopeRef.REUNITED,
) -> TensorEnvelope:
    domain = SupportDomain.for_tensor(
        mask,
        domain_id="aggregate-input",
        owner_ref="arti/reunion-aggregate@1",
        partition_id="reunited",
        transition_id="aggregate-test",
    )
    return TensorEnvelope(ref, value, mask, domain)


def test_reunion_aggregate_is_the_explicit_lossy_boundary() -> None:
    aggregate = ReunionAggregate(SoftFoldAggregate(k=3, dim=4))
    x = torch.randn(2, 7, 4)
    mask = torch.tensor(
        [[True, True, True, True, False, False, False], [True] * 7]
    )

    result = aggregate(envelope(x, mask))

    assert result.ref is EnvelopeRef.PULSE
    assert result.value.shape == (2, 3, 4)
    assert result.mask.shape == (2, 3)
    assert result.mask.all()
    assert result.domain.owner_ref == "arti/reunion-aggregate@1"


def test_aggregate_accepts_closed_world_or_observation_without_fold() -> None:
    aggregate = ReunionAggregate(SoftFoldAggregate(k=2, dim=3))
    x = torch.randn(1, 4, 3)
    mask = torch.ones(1, 4, dtype=torch.bool)

    world = aggregate(envelope(x, mask, EnvelopeRef.WORLD))
    observation = aggregate(envelope(x, mask, EnvelopeRef.OBSERVATION))

    assert world.value.shape == observation.value.shape == (1, 2, 3)


def test_aggregate_flattens_observation_and_instance_axes_after_reunion() -> None:
    torch.manual_seed(13)
    aggregate = ReunionAggregate(SoftFoldAggregate(k=3, dim=4)).eval()
    x = torch.randn(2, 3, 5, 4)
    mask = torch.tensor(
        [
            [[True, True, True, False, False]] * 3,
            [[True, True, True, True, True]] * 3,
        ]
    )

    actual = aggregate(envelope(x, mask, EnvelopeRef.OBSERVATION))
    expected = aggregate.kernel(
        x.reshape(2, 15, 4),
        mask=mask.reshape(2, 15),
    )

    torch.testing.assert_close(actual.value, expected, rtol=0, atol=0)
    assert actual.mask.shape == (2, 3)


def test_aggregate_rejects_non_envelope_input() -> None:
    aggregate = ReunionAggregate(SoftFoldAggregate(k=2, dim=3))

    with pytest.raises(TypeError, match="closed TensorEnvelope"):
        aggregate(torch.randn(1, 4, 3))


def test_no_intervention_fold_unfold_matches_direct_aggregate() -> None:
    torch.manual_seed(7)
    aggregate = ReunionAggregate(SoftFoldAggregate(k=2, dim=3)).eval()
    topology = arti.mechanisms.ReversibleTopology(
        active_count=2,
        policy=arti.mechanisms.FixedTopologyPolicy(order=[3, 0, 4, 1, 2]),
    )
    x = torch.randn(2, 5, 3)
    mask = torch.tensor([[True, True, True, True, False], [True] * 5])
    restored = topology.unfold(topology.fold(x, mask))

    direct = aggregate(envelope(x, mask))
    round_trip = aggregate(envelope(restored.value, restored.mask))

    torch.testing.assert_close(round_trip.value, direct.value, rtol=0, atol=0)
    assert torch.equal(round_trip.mask, direct.mask)


def test_active_intervention_is_aggregated_only_after_reunion() -> None:
    aggregate = ReunionAggregate(SoftFoldAggregate(k=2, dim=3)).eval()
    topology = arti.mechanisms.ReversibleTopology(
        active_count=2,
        policy=arti.mechanisms.FixedTopologyPolicy(order=[3, 0, 4, 1, 2]),
    )
    x = torch.randn(1, 5, 3)
    mask = torch.ones(1, 5, dtype=torch.bool)
    state = topology.fold(x, mask)
    changed = topology.unfold(state.replace(active=state.active + 2.0))

    actual = aggregate(envelope(changed.value, changed.mask))
    expected = aggregate.kernel(changed.value, mask=changed.mask)

    torch.testing.assert_close(actual.value, expected, rtol=0, atol=0)


def test_aggregate_preserves_gradient_to_valid_reunited_values() -> None:
    aggregate = ReunionAggregate(SoftFoldAggregate(k=2, dim=3))
    x = torch.randn(2, 6, 3, requires_grad=True)
    mask = torch.ones(2, 6, dtype=torch.bool)

    result = aggregate(envelope(x, mask))
    result.value.square().mean().backward()

    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    assert float(x.grad.abs().sum()) > 0


def test_empty_sample_remains_invalid_and_zero() -> None:
    aggregate = ReunionAggregate(SoftFoldAggregate(k=2, dim=3)).eval()
    x = torch.randn(2, 4, 3)
    mask = torch.tensor([[False, False, False, False], [True, True, False, False]])

    result = aggregate(envelope(x, mask))

    assert not result.mask[0].any()
    assert torch.equal(result.value[0], torch.zeros_like(result.value[0]))
    assert result.mask[1].all()


def test_aggregate_component_graph_keeps_old_fold_as_kernel_dependency() -> None:
    kernel = SoftFoldAggregate(k=2, dim=3)
    aggregate = ReunionAggregate(kernel)
    kernel_spec = component_spec(kernel)
    aggregate_spec = component_spec(aggregate)

    assert kernel_spec.reference == "arti/soft-fold-aggregate@1"
    assert kernel_spec.capabilities == ("pulse.aggregate.kernel",)
    assert kernel_spec.dependencies == ("arti/fold@1",)
    assert aggregate_spec.reference == "arti/reunion-aggregate@1"
    assert aggregate_spec.capabilities == ("pulse.stage.aggregate",)
    assert aggregate_spec.dependencies == ("arti/soft-fold-aggregate@1",)
