from __future__ import annotations

import pytest
import torch

import arti
from arti import alpha


def _program(dim: int = 2) -> alpha.FormulaFabricProgram:
    return alpha.FormulaFabricProgram(
        arena_capacity=4,
        feature_dim=dim,
        steps=((alpha.FormulaInvocation(alpha.FormulaPrimitive.ADD, 2),),),
        domain="formula-commit-blend",
    )


def _route(value: torch.Tensor) -> alpha.FormulaRoutePlan:
    weights = value.new_zeros(value.shape[0], 1, 1, 2, 4)
    weights[:, 0, 0, 0, 0] = 1
    weights[:, 0, 0, 1, 1] = 1
    enabled = torch.ones(
        value.shape[0], 1, 1, dtype=torch.bool, device=value.device
    )
    return alpha.FormulaRoutePlan(weights, enabled, enabled, enabled)


def _state(value: torch.Tensor) -> alpha.FormulaArenaState:
    mask = torch.ones(value.shape[:2], dtype=torch.bool, device=value.device)
    return alpha.FormulaArenaState.from_tensor(
        value,
        mask,
        capacity=4,
        domain="formula-commit-blend",
    )


def _blend(dim: int = 2) -> alpha.FormulaCommitBlend:
    return alpha.FormulaCommitBlend(alpha.FormulaFabric(_program(dim)))


def test_zero_and_one_commit_weights_match_boundaries() -> None:
    value = torch.tensor([[[1.0, 2.0], [3.0, 5.0], [9.0, 11.0]]])
    state = _state(value)
    route = _route(value)
    blend = _blend()

    zero = blend(state, route, value.new_zeros(1, 1, 1))
    one = blend(state, route, value.new_ones(1, 1, 1))
    hard = blend.fabric(state, route)

    torch.testing.assert_close(zero.state.value[:, 2], value[:, 2])
    torch.testing.assert_close(one.state.value, hard.state.value)
    assert torch.equal(zero.state.version, hard.state.version)
    assert torch.equal(one.state.version, hard.state.version)
    assert torch.equal(zero.trace.weights, value.new_zeros(1, 1, 1))


def test_zero_and_one_commit_weights_are_exact_for_extreme_finite_values() -> None:
    maximum = torch.finfo(torch.float32).max
    value = torch.tensor(
        [[[maximum, maximum], [-maximum, -maximum], [maximum, -maximum]]]
    )
    state = _state(value)
    route = _route(value)
    blend = _blend()

    zero = blend(state, route, value.new_zeros(1, 1, 1))
    one = blend(state, route, value.new_ones(1, 1, 1))
    hard = blend.fabric(state, route)

    assert torch.equal(zero.state.value[:, 2], value[:, 2])
    assert torch.equal(one.state.value, hard.state.value)
    assert torch.isfinite(zero.state.value).all()
    assert torch.isfinite(one.state.value).all()


@pytest.mark.parametrize("boundary", [0.0, 1.0])
def test_commit_weight_boundaries_keep_linear_blend_gradient(boundary: float) -> None:
    value = torch.tensor([[[1.0, 2.0], [3.0, 5.0], [9.0, 11.0]]])
    weight = torch.tensor([[[boundary]]], requires_grad=True)
    result = _blend()(_state(value), _route(value), weight)

    result.state.value[:, 2].sum().backward()

    expected = (value[:, 0] + value[:, 1] - value[:, 2]).sum()
    torch.testing.assert_close(weight.grad.squeeze(), expected)


def test_commit_weights_are_per_sample_and_differentiable() -> None:
    value = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 5.0], [9.0, 11.0]],
            [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]],
        ],
        requires_grad=True,
    )
    weights = torch.tensor([[[0.25]], [[0.75]]], requires_grad=True)
    result = _blend()(_state(value), _route(value), weights)
    expected = value[:, 2] + weights[:, 0] * (
        value[:, 0] + value[:, 1] - value[:, 2]
    )

    torch.testing.assert_close(result.state.value[:, 2], expected)
    result.state.value[:, 2].square().mean().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert weights.grad is not None and torch.count_nonzero(weights.grad) > 0


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan")])
def test_invalid_commit_weights_fail_closed(bad: float) -> None:
    value = torch.randn(1, 3, 2)
    with pytest.raises(ValueError, match="within"):
        _blend()(_state(value), _route(value), value.new_full((1, 1, 1), bad))


def test_non_tensor_commit_weights_fail_with_contract_error() -> None:
    value = torch.randn(1, 3, 2)
    with pytest.raises(ValueError, match="commit_weights"):
        _blend()(_state(value), _route(value), 0.5)  # type: ignore[arg-type]


def test_formula_commit_blend_runs_inside_pulse() -> None:
    topology = alpha.ReversibleTopology(active_count=3)
    fold, unfold = topology.operations()
    compute = alpha.FormulaFabricCompute(_blend(), active_count=3)
    pulse = alpha.AdaptivePulse(fold=fold, selective_compute=compute, unfold=unfold)
    value = torch.tensor([[[1.0, 2.0], [3.0, 5.0], [9.0, 11.0]]])

    result = pulse.run_tensor(
        value,
        intervened=torch.tensor([[False, False, True]]),
        formula_route=_route(value),
        compute_factors=value.new_full((1, 1, 1), 0.5),
    )

    torch.testing.assert_close(
        result.value[:, 2],
        0.5 * value[:, 2] + 0.5 * (value[:, 0] + value[:, 1]),
    )
    assert isinstance(result.diagnostics.compute, alpha.FormulaCommitBlendTrace)


def test_formula_commit_blend_counts_factors_in_operation_budget() -> None:
    limits = alpha.ContractLimits(max_operation_bytes=504)
    plain = alpha.FormulaFabricCompute(
        alpha.FormulaFabric(_program(), limits=limits),
        active_count=3,
    )
    blended = alpha.FormulaFabricCompute(
        alpha.FormulaCommitBlend(alpha.FormulaFabric(_program(), limits=limits)),
        active_count=3,
    )
    value = torch.tensor([[[1.0, 2.0], [3.0, 5.0], [9.0, 11.0]]])
    support = torch.ones(1, 3, dtype=torch.bool)
    workspace = alpha.ActiveWorkspace(
        value,
        support,
        support,
        torch.tensor([[False, False, True]]),
    )

    plain(workspace, formula_route=_route(value))
    with pytest.raises(ValueError, match="operation byte"):
        blended(
            workspace,
            value.new_full((1, 1, 1), 0.5),
            formula_route=_route(value),
        )


def test_direct_formula_commit_blend_counts_factors_in_operation_budget() -> None:
    limits = alpha.ContractLimits(max_operation_bytes=504)
    fabric = alpha.FormulaFabric(_program(), limits=limits)
    value = torch.tensor([[[1.0, 2.0], [3.0, 5.0], [9.0, 11.0]]])
    state = _state(value)
    route = _route(value)

    fabric(state, route)
    with pytest.raises(ValueError, match="max_operation_bytes"):
        alpha.FormulaCommitBlend(fabric)(
            state,
            route,
            value.new_full((1, 1, 1), 0.5),
        )


def test_bank_routed_formula_commit_blend_uses_compute_factors() -> None:
    program = _program()
    policies = []
    for seed in (31, 32):
        bank = alpha.TypedTopologyOperandBank(
            slots=4,
            key_dim=4,
            factor_dim=1,
            seed=seed,
            value_seed=seed + 100,
            bank_id=f"commit-blend-{seed}",
        )
        policies.append(
            alpha.TypedBankFormulaTopologyPolicy(
                2,
                [bank],
                key_dim=4,
                query_seed=seed + 200,
                diagnostics="none",
            )
        )
    candidate = torch.zeros(1, 1, 2, 4, dtype=torch.bool)
    candidate[..., :2] = True
    route_source = alpha.BankFormulaRouteSource(
        program,
        policies,
        active_count=3,
        candidate_mask=candidate,
    )
    routed = alpha.RoutedFormulaFabricCompute(
        alpha.FormulaFabricCompute(_blend(), active_count=3),
        route_source,
    )
    value = torch.tensor([[[1.0, 2.0], [3.0, 5.0], [9.0, 11.0]]])
    support = torch.ones(1, 3, dtype=torch.bool)
    workspace = alpha.ActiveWorkspace(
        value,
        support,
        support,
        torch.tensor([[False, False, True]]),
    )

    updated, info = routed(
        workspace,
        value.new_full((1, 1, 1), 0.5),
        return_info=True,
    )

    assert updated.value.shape == value.shape
    assert info.route_origin == "bank-formula"
    assert isinstance(info.trace, alpha.FormulaCommitBlendTrace)


def test_formula_commit_blend_arti_st_round_trip(tmp_path) -> None:
    source_topology = alpha.ReversibleTopology(active_count=3)
    source_fold, source_unfold = source_topology.operations()
    source = alpha.AdaptivePulse(
        fold=source_fold,
        selective_compute=alpha.FormulaFabricCompute(_blend(), active_count=3),
        unfold=source_unfold,
        aggregate=alpha.ReunionAggregate(alpha.SoftFoldAggregate(k=3, dim=2)),
    ).eval()
    value = torch.randn(2, 3, 2)
    route = _route(value)
    weights = torch.rand(2, 1, 1)
    expected = source.run_tensor(
        value,
        intervened=torch.tensor([[False, False, True]]).expand(2, -1),
        formula_route=route,
        compute_factors=weights,
    ).value

    saved = arti.save(source, tmp_path / "formula-commit-blend.arti.st")
    target_topology = alpha.ReversibleTopology(active_count=3)
    target_fold, target_unfold = target_topology.operations()
    target = alpha.AdaptivePulse(
        fold=target_fold,
        selective_compute=alpha.FormulaFabricCompute(_blend(), active_count=3),
        unfold=target_unfold,
        aggregate=alpha.ReunionAggregate(alpha.SoftFoldAggregate(k=3, dim=2)),
    ).eval()
    loaded = arti.load(saved.weights_path, model=target)
    actual = target.run_tensor(
        value,
        intervened=torch.tensor([[False, False, True]]).expand(2, -1),
        formula_route=route,
        compute_factors=weights,
    ).value
    refs = {
        node["ref"]
        for node in loaded.manifest["architecture"]["component_graph"]["nodes"]
    }

    torch.testing.assert_close(actual, expected)
    assert refs >= {"arti/formula-commit-blend@1", "arti/formula-fabric@1"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_formula_commit_blend_pulse_cuda_fullgraph(dtype: torch.dtype) -> None:
    topology = alpha.ReversibleTopology(active_count=3)
    fold, unfold = topology.operations()
    pulse = alpha.AdaptivePulse(
        fold=fold,
        selective_compute=alpha.FormulaFabricCompute(_blend(), active_count=3),
        unfold=unfold,
    ).cuda().to(dtype).train()
    compiled = torch.compile(pulse, backend="inductor", fullgraph=True)
    value = torch.randn(2, 3, 2, device="cuda", dtype=dtype, requires_grad=True)
    route = _route(value)
    factors = torch.rand(2, 1, 1, device="cuda", dtype=dtype, requires_grad=True)
    mask = torch.ones(2, 3, dtype=torch.bool, device="cuda")
    intervened = torch.tensor(
        [[False, False, True], [False, False, True]], device="cuda"
    )
    domain = alpha.SupportDomain.for_tensor(
        mask,
        domain_id="formula-commit-blend",
        owner_ref="arti/pulse@2",
        partition_id="world",
        transition_id="compiled",
    )
    world = alpha.TensorEnvelope(alpha.EnvelopeRef.WORLD, value, mask, domain)
    supports = alpha.PulseSupports(
        alpha.SupportMask(alpha.SupportKind.OBSERVED, mask, domain),
        alpha.SupportMask(alpha.SupportKind.EXPOSED, mask, domain),
        alpha.SupportMask(alpha.SupportKind.INTERVENED, intervened, domain),
        validity=mask,
    )

    result = compiled(
        world,
        supports,
        formula_route=route,
        compute_factors=factors,
    )
    result.value.float().square().mean().backward()

    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert factors.grad is not None and torch.isfinite(factors.grad).all()
