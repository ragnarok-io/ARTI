from __future__ import annotations

import inspect
from unittest import mock

import pytest
import torch

import arti
from arti import alpha


def _program() -> alpha.FormulaFabricProgram:
    return alpha.FormulaFabricProgram(
        arena_capacity=4,
        feature_dim=2,
        steps=((alpha.FormulaInvocation(alpha.FormulaPrimitive.ADD, 2),),),
        domain="objective-formula-control",
    )


def _route(value: torch.Tensor) -> alpha.FormulaRoutePlan:
    weights = value.new_zeros(value.shape[0], 1, 1, 2, 4)
    weights[:, 0, 0, 0, 0] = 1
    weights[:, 0, 0, 1, 1] = 1
    enabled = torch.ones(
        value.shape[0], 1, 1, dtype=torch.bool, device=value.device
    )
    return alpha.FormulaRoutePlan(weights, enabled, enabled, enabled)


def _workspace(value: torch.Tensor) -> alpha.ActiveWorkspace:
    support = torch.ones(value.shape[:-1], dtype=torch.bool, device=value.device)
    intervened = torch.zeros_like(support)
    intervened[:, 2] = True
    return alpha.ActiveWorkspace(value, support, support, intervened)


def _objective() -> alpha.ObjectiveExposureBank:
    objective = alpha.ObjectiveExposureBank(
        slots=2,
        query_dim=2,
        key_layout="circle",
        min_exposure=0.0,
        max_exposure=1.0,
    )
    with torch.no_grad():
        objective.values.copy_(torch.tensor([-1.5, 1.5]))
    return objective


def _controlled() -> alpha.ObjectiveFormulaFabricCompute:
    compute = alpha.FormulaFabricCompute(
        alpha.FormulaCommitBlend(alpha.FormulaFabric(_program())),
        active_count=3,
    )
    return alpha.ObjectiveFormulaFabricCompute(compute, _objective())


def _routed_controlled() -> alpha.ObjectiveFormulaFabricCompute:
    program = _program()
    policies = []
    for seed in (31, 32):
        bank = alpha.TypedTopologyOperandBank(
            slots=4,
            key_dim=4,
            factor_dim=1,
            seed=seed,
            value_seed=seed + 100,
            bank_id=f"objective-formula-{seed}",
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
    source = alpha.BankFormulaRouteSource(
        program,
        policies,
        active_count=3,
        candidate_mask=candidate,
    )
    compute = alpha.FormulaFabricCompute(
        alpha.FormulaCommitBlend(alpha.FormulaFabric(program)),
        active_count=3,
    )
    return alpha.ObjectiveFormulaFabricCompute(
        alpha.RoutedFormulaFabricCompute(compute, source),
        _objective(),
    )


def _pulse() -> alpha.AdaptivePulse:
    topology = alpha.ReversibleTopology(active_count=3)
    fold, unfold = topology.operations()
    return alpha.AdaptivePulse(
        fold=fold,
        selective_compute=_controlled(),
        unfold=unfold,
    )


def test_control_matches_same_external_commit_weights() -> None:
    value = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 5.0], [9.0, 11.0]],
            [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]],
        ]
    )
    query = torch.tensor([[1.0, 0.0], [-1.0, 0.0]])
    workspace = _workspace(value)
    route = _route(value)
    controlled = _controlled()

    actual, info = controlled(
        workspace,
        objective_query=query,
        formula_route=route,
        return_info=True,
    )
    expected = controlled.compute(
        workspace,
        info.commit_weights,
        formula_route=route,
    )

    torch.testing.assert_close(actual.value, expected.value)
    assert info.commit_weights.shape == (2, 1, 1)
    assert torch.all((info.commit_weights >= 0) & (info.commit_weights <= 1))
    assert not torch.equal(info.commit_weights[0], info.commit_weights[1])


def test_control_is_differentiable_only_through_objective_values() -> None:
    value = torch.tensor(
        [[[1.0, 2.0], [3.0, 5.0], [9.0, 11.0]]], requires_grad=True
    )
    query = torch.tensor([[1.0, 0.0]], requires_grad=True)
    controlled = _controlled()

    result = controlled(
        _workspace(value),
        objective_query=query,
        formula_route=_route(value),
    )
    result.value.square().mean().backward()

    assert controlled.objective.values.grad is not None
    assert torch.count_nonzero(controlled.objective.values.grad) > 0
    assert query.grad is None
    assert value.grad is not None and torch.isfinite(value.grad).all()


def test_control_rejects_external_factors_missing_query_and_wrong_shape() -> None:
    value = torch.randn(2, 3, 2)
    controlled = _controlled()
    workspace = _workspace(value)
    route = _route(value)

    with pytest.raises(ValueError, match="does not accept external"):
        controlled(
            workspace,
            torch.ones(2, 1, 1),
            objective_query=torch.randn(2, 2),
            formula_route=route,
        )
    with pytest.raises(ValueError, match="requires objective_query"):
        controlled(workspace, formula_route=route)
    with pytest.raises(ValueError, match="objective_query must have shape"):
        controlled(
            workspace,
            objective_query=torch.randn(2, 1, 2),
            formula_route=route,
        )
    with pytest.raises(ValueError, match="share dtype"):
        controlled(
            workspace,
            objective_query=torch.randn(2, 2, dtype=torch.bfloat16),
            formula_route=route,
        )


def test_control_admits_total_cost_before_objective_execution() -> None:
    controlled = _controlled()
    value = torch.randn(2, 3, 2)
    workspace = _workspace(value)
    controlled.limits = alpha.ContractLimits(max_operation_bytes=1)

    with mock.patch.object(
        controlled.objective, "forward", side_effect=AssertionError("executed")
    ):
        with pytest.raises(ValueError, match="cumulative operation byte"):
            controlled(
                workspace,
                objective_query=torch.randn(2, 2),
                formula_route=_route(value),
                return_info=True,
            )


def test_control_has_no_future_input_and_pulse_requires_explicit_query() -> None:
    assert "future" not in inspect.signature(_controlled().forward).parameters
    pulse = _pulse()
    value = torch.randn(2, 3, 2)
    route = _route(value)

    with pytest.raises(ValueError, match="requires objective_query"):
        pulse.run_tensor(value, formula_route=route)
    with pytest.raises(ValueError, match="does not accept external"):
        pulse.run_tensor(
            value,
            formula_route=route,
            objective_query=torch.randn(2, 2),
            compute_factors=torch.ones(2, 1, 1),
        )
    with pytest.raises(ValueError, match="Objective-controlled compute"):
        alpha.AdaptivePulse().run_tensor(value, objective_query=torch.randn(2, 2))


def test_control_runs_as_an_optional_pulse_compute_stage() -> None:
    pulse = _pulse()
    value = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 5.0], [9.0, 11.0]],
            [[2.0, 4.0], [6.0, 8.0], [10.0, 12.0]],
        ]
    )
    result = pulse.run_tensor(
        value,
        intervened=torch.tensor([[False, False, True]]).expand(2, -1),
        formula_route=_route(value),
        objective_query=torch.tensor([[1.0, 0.0], [-1.0, 0.0]]),
    )

    assert isinstance(
        result.diagnostics.compute, alpha.ObjectiveFormulaFabricComputeInfo
    )
    assert result.value.shape == value.shape
    assert not torch.equal(result.value[0, 2], result.value[1, 2])


def test_control_reuses_bank_routed_formula_executor() -> None:
    value = torch.tensor([[[1.0, 2.0], [3.0, 5.0], [9.0, 11.0]]])
    controlled = _routed_controlled()

    updated, info = controlled(
        _workspace(value),
        objective_query=torch.tensor([[1.0, 0.0]]),
        return_info=True,
    )

    assert updated.value.shape == value.shape
    assert info.formula.route_origin == "bank-formula"
    assert info.formula.factor_contract == "required"
    assert controlled.route_contract == "bound-source-or-explicit-override"


def test_control_component_graph_and_arti_st_round_trip(tmp_path) -> None:
    source = _pulse().eval()
    value = torch.randn(2, 3, 2)
    query = torch.randn(2, 2)
    route = _route(value)
    intervened = torch.tensor([[False, False, True]]).expand(2, -1)
    expected = source.run_tensor(
        value,
        intervened=intervened,
        formula_route=route,
        objective_query=query,
    ).value

    saved = arti.save(source, tmp_path / "objective-formula.arti.st")
    target = _pulse().eval()
    loaded = arti.load(saved.weights_path, model=target)
    actual = target.run_tensor(
        value,
        intervened=intervened,
        formula_route=route,
        objective_query=query,
    ).value
    refs = {
        node["ref"]
        for node in loaded.manifest["architecture"]["component_graph"]["nodes"]
    }

    torch.testing.assert_close(actual, expected)
    assert refs >= {
        "arti/objective-formula-fabric-compute@1",
        "arti/objective-exposure-bank@1",
        "arti/formula-commit-blend@1",
        "arti/formula-fabric@1",
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_control_cuda_fullgraph(dtype: torch.dtype) -> None:
    pulse = _pulse().cuda().to(dtype).train()
    compiled = torch.compile(pulse, backend="inductor", fullgraph=True)
    value = torch.randn(8, 3, 2, device="cuda", dtype=dtype, requires_grad=True)
    query = torch.randn(8, 2, device="cuda", dtype=dtype)
    route = _route(value)
    mask = torch.ones(8, 3, dtype=torch.bool, device="cuda")
    intervened = torch.zeros_like(mask)
    intervened[:, 2] = True
    domain = alpha.SupportDomain.for_tensor(
        mask,
        domain_id="objective-fullgraph",
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
    formula_before = {
        name: tensor.detach().clone()
        for name, tensor in pulse.selective_compute.compute.state_dict().items()
    }

    result = compiled(
        world,
        supports,
        objective_query=query,
        formula_route=route,
    )
    result.value.float().square().mean().backward()

    assert isinstance(
        result.diagnostics.compute, alpha.ObjectiveFormulaFabricComputeInfo
    )
    assert pulse.selective_compute.objective.values.grad is not None
    assert torch.isfinite(pulse.selective_compute.objective.values.grad).all()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert all(
        torch.equal(tensor, pulse.selective_compute.compute.state_dict()[name])
        for name, tensor in formula_before.items()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_control_cuda_autocast_accepts_fp32_objective_with_bf16_workspace() -> None:
    controlled = _controlled().cuda().train()
    value = torch.randn(
        4, 3, 2, device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    query = torch.randn(4, 2, device="cuda", dtype=torch.bfloat16)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = controlled(
            _workspace(value),
            objective_query=query,
            formula_route=_route(value),
        )
        loss = result.value.float().square().mean()
    loss.backward()

    assert result.value.dtype == torch.bfloat16
    assert torch.isfinite(result.value).all()
    assert controlled.objective.values.dtype == torch.float32
    assert controlled.objective.values.grad is not None
    assert torch.isfinite(controlled.objective.values.grad).all()
    assert value.grad is not None and torch.isfinite(value.grad).all()
