from __future__ import annotations

from pathlib import Path

import pytest
import torch

import arti
from arti.mechanisms import (
    AdaptiveObservation,
    BankConditionedObservationPolicy,
    EnvelopeRef,
    FourierShiftObservationOperator,
    ObservationOperandBank,
    ObservationTrajectoryFormula,
    OperandKind,
    StateAffineObservationOperator,
    SupportDomain,
    TensorEnvelope,
)
from arti.component_registry import component_spec


def world(value: torch.Tensor) -> TensorEnvelope:
    mask = torch.ones(value.shape[:2], dtype=torch.bool, device=value.device)
    domain = SupportDomain.for_tensor(
        mask,
        domain_id="optional-observation-world",
        owner_ref="arti/adaptive-observation@1",
        partition_id="world",
        transition_id="optional-observation",
    )
    return TensorEnvelope(EnvelopeRef.WORLD, value, mask, domain)


def test_fourier_zero_inverse_and_magnitude_contract() -> None:
    operator = FourierShiftObservationOperator((7, 7))
    value = torch.randn(2, 49, 3, dtype=torch.float64)
    zero = torch.zeros(2, 2, dtype=torch.float64)
    shift = torch.tensor([[0.25, -0.4], [-0.75, 0.125]], dtype=torch.float64)

    identity = operator(value, zero)
    shifted = operator(value, shift)
    restored = operator(shifted, -shift)

    torch.testing.assert_close(identity, value, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(restored, value, rtol=1e-11, atol=1e-11)
    torch.testing.assert_close(
        shifted.square().sum(dim=1), value.square().sum(dim=1), rtol=1e-11, atol=1e-11
    )
    assert not shifted.is_complex()
    assert operator.boundary == "circular"
    assert operator.execution_contract() == {
        "backend": "torch.fft",
        "compile_policy": "safe_training",
        "compiled_outer_graph_training": True,
        "compiled_fullgraph_forward": False,
        "compiled_fullgraph_backward": False,
        "eager_backward": True,
    }


def test_polar_direction_matches_cartesian_displacement() -> None:
    cartesian = FourierShiftObservationOperator((4, 4), state_mode="cartesian")
    polar = FourierShiftObservationOperator((4, 4), state_mode="polar")
    value = torch.randn(2, 16, 2)
    radius = torch.tensor([0.5, 1.25])
    direction = torch.tensor([[3.0, 4.0], [-4.0, 3.0]])
    unit = direction / direction.norm(dim=-1, keepdim=True)
    cartesian_state = radius.unsqueeze(-1) * unit
    polar_state = torch.cat((radius.unsqueeze(-1), direction), dim=-1)

    expected = cartesian(value, cartesian_state)
    actual = polar(value, polar_state)

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    zero_direction = polar(value, torch.tensor([[1.0, 0.0, 0.0]]).expand(2, -1))
    torch.testing.assert_close(zero_direction, value, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fourier_trajectory_uses_one_substrate_and_backpropagates(
    dtype: torch.dtype,
) -> None:
    operator = FourierShiftObservationOperator((4, 4))
    value = torch.randn(2, 16, 2, dtype=dtype, requires_grad=True)
    states = (torch.randn(2, 3, 2, dtype=dtype) * 0.2).requires_grad_()

    trajectory = operator.forward_trajectory(value, states)
    expected = torch.stack(
        [operator(value, states[:, index]) for index in range(states.shape[1])],
        dim=1,
    )

    torch.testing.assert_close(trajectory, expected)
    trajectory.square().mean().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert states.grad is not None and torch.isfinite(states.grad).all()


def observation_bank(
    *,
    seed: int,
    bank_id: str,
    continuation: float,
) -> ObservationOperandBank:
    bank = ObservationOperandBank(8, 4, 4, seed=seed, bank_id=bank_id)
    with torch.no_grad():
        bank.values[:, :3].normal_(mean=0.0, std=0.4)
        bank.values[:, 3].fill_(continuation)
    return bank


def bank_policy(
    banks: list[ObservationOperandBank],
    *,
    weights: list[float] | None = None,
) -> BankConditionedObservationPolicy:
    return BankConditionedObservationPolicy(
        input_dim=2,
        state_dim=3,
        max_observations=4,
        banks=banks,
        key_dim=4,
        query_seed=17,
        bank_weights=weights,
    )


def test_observation_bank_emits_consumer_bound_typed_operands() -> None:
    bank = observation_bank(seed=3, bank_id="observation-a", continuation=2.0)
    formula = ObservationTrajectoryFormula(3)
    query = torch.randn(2, 4, 4)
    mask = torch.ones(2, 4, dtype=torch.bool)

    operands, route = bank.read_operands(query, mask, consumer=formula)
    output = formula.evaluate_operands(operands, source=bank)

    assert operands.contract.kind is OperandKind.OBSERVATION
    assert operands.contract.consumer_ref.startswith(
        "arti/observation-trajectory-formula@sha256:"
    )
    assert output.states.shape == (2, 4, 3)
    assert output.continuation_logits.shape == (2, 4)
    torch.testing.assert_close(route.sum(dim=-1), torch.ones(2, 4))
    with pytest.raises(ValueError, match="consumer"):
        operands.consume(
            consumer=StateAffineObservationOperator(2, 3),
            kind=OperandKind.OBSERVATION,
            source=bank,
            partition_id=bank.bank_id,
            domain=operands.contract.domain,
            factor_dim=bank.factor_dim,
            layout="dense",
            source_asset_fingerprint=bank.asset_fingerprint,
        )


def test_bank_values_control_state_and_dynamic_observation_count() -> None:
    positive = observation_bank(seed=5, bank_id="positive-bank", continuation=4.0)
    negative = observation_bank(seed=5, bank_id="negative-bank", continuation=-4.0)
    value = torch.randn(3, 6, 2)
    mask = torch.ones(3, 6, dtype=torch.bool)

    continued = bank_policy([positive])(value, mask)
    stopped = bank_policy([negative])(value, mask)

    assert continued.mask.sum(dim=1).tolist() == [4, 4, 4]
    assert stopped.mask.sum(dim=1).tolist() == [1, 1, 1]
    assert not torch.equal(continued.states, stopped.states)


def test_bank_concat_uses_independent_routes_and_explicit_weights() -> None:
    first = observation_bank(seed=7, bank_id="first-bank", continuation=2.0)
    second = observation_bank(seed=11, bank_id="second-bank", continuation=-2.0)
    value = torch.randn(2, 5, 2)
    mask = torch.ones(2, 5, dtype=torch.bool)

    isolated = bank_policy([first])(value, mask)
    concatenated = bank_policy([first, second], weights=[1.0, 0.0])(value, mask)

    torch.testing.assert_close(concatenated.states, isolated.states, rtol=0, atol=0)
    assert torch.equal(concatenated.mask, isolated.mask)


def test_bank_conditioned_fourier_observation_trains_bank_values() -> None:
    bank = observation_bank(seed=13, bank_id="polar-bank", continuation=2.0)
    policy = bank_policy([bank])
    observation = AdaptiveObservation(
        policy,
        operator=FourierShiftObservationOperator((4, 4), state_mode="polar"),
    )
    value = torch.randn(2, 16, 2, requires_grad=True)

    result = observation(world(value))
    result.value.square().mean().backward()

    assert result.value.shape == (2, 4, 16, 2)
    assert bank.values.grad is not None and bank.values.grad.abs().sum() > 0
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert not any(parameter.requires_grad for parameter in policy.query.parameters())


def test_optional_observation_component_graph_and_arti_st_round_trip(
    tmp_path: Path,
) -> None:
    source = AdaptiveObservation(
        bank_policy([observation_bank(seed=19, bank_id="saved-bank", continuation=2.0)]),
        operator=FourierShiftObservationOperator((4, 4), state_mode="polar"),
    )
    target = AdaptiveObservation(
        bank_policy([observation_bank(seed=19, bank_id="saved-bank", continuation=2.0)]),
        operator=FourierShiftObservationOperator((4, 4), state_mode="polar"),
    )
    saved = arti.save(source, tmp_path / "optional-observation.arti.st")
    arti.load(saved.weights_path, model=target)
    value = torch.randn(2, 16, 2)

    expected = source(world(value))
    actual = target(world(value))

    torch.testing.assert_close(actual.value, expected.value, rtol=0, atol=0)
    spec = component_spec(source)
    assert spec.reference.startswith("arti/adaptive-observation@sha256:")
    assert any(
        ref.startswith("arti/bank-observation-policy@sha256:")
        for ref in spec.dependencies
    )
    assert any(
        ref.startswith("arti/fourier-observation-operator@sha256:")
        for ref in spec.dependencies
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_bank_fourier_observation_cuda_fullgraph() -> None:
    module = AdaptiveObservation(
        bank_policy([observation_bank(seed=23, bank_id="cuda-bank", continuation=2.0)]),
        operator=FourierShiftObservationOperator(
            (4, 4), state_mode="polar", compile_policy="fullgraph_forward"
        ),
    ).cuda()
    compiled = torch.compile(module, fullgraph=True)
    value = torch.randn(2, 16, 2, device="cuda", requires_grad=True)
    envelope = world(value)

    expected = module(envelope)
    actual = compiled(envelope)

    torch.testing.assert_close(actual.value, expected.value)
    expected.value.square().mean().backward()
    bank = module.policy.banks[0]
    assert bank.values.grad is not None and torch.isfinite(bank.values.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_bank_fourier_observation_safe_compiled_training() -> None:
    module = AdaptiveObservation(
        bank_policy([observation_bank(seed=29, bank_id="safe-bank", continuation=2.0)]),
        operator=FourierShiftObservationOperator((4, 4), state_mode="polar"),
    ).cuda()
    compiled = torch.compile(module)
    value = torch.randn(2, 16, 2, device="cuda", requires_grad=True)

    actual = compiled(world(value))
    actual.value.square().mean().backward()

    bank = module.policy.banks[0]
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert bank.values.grad is not None and torch.isfinite(bank.values.grad).all()
