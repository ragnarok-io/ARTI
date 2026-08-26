from __future__ import annotations

import inspect

import pytest
import torch

import arti
from arti import alpha


class _DynamicFoldSource(torch.nn.Module):
    _component_reference = "example/dynamic-formula-fold-source@1"

    def __init__(self, dim: int = 2) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(dim))

    def propose(
        self,
        keys: torch.Tensor,
        query: torch.Tensor,
        *,
        mask: torch.Tensor,
    ) -> alpha.TopologyProposal:
        priority = torch.einsum("bnd,bd->bn", keys @ self.weight, query)
        return alpha.TopologyProposal(alpha.TopologyAction(priority))

    def topology_contract(self) -> dict[str, object]:
        return {
            "ref": self._component_reference,
            "input_schema": ["keys[B,N,D]", "query[B,D]"],
            "input_instance_axes": [1, None],
            "output": "arti/topology-proposal@1",
        }


class _StateSwitchingRouteSource(alpha.BankFormulaRouteSource):
    """Deterministic fixture whose second operand follows committed slot zero."""

    def __init__(self, program: alpha.FormulaFabricProgram) -> None:
        torch.nn.Module.__init__(self)
        self.program = program
        self.active_count = 2
        self.config_fingerprint = "state-switching-route-source"
        self._limits = alpha.ContractLimits()

    def operation_bytes_upper_bound(self, workspace: object) -> int:
        return 1

    def forward(
        self, workspace: object
    ) -> tuple[alpha.FormulaRoutePlan, alpha.BankFormulaRouteInfo]:
        assert isinstance(workspace, alpha.ActiveWorkspace)
        value = workspace.value
        batch = value.shape[0]
        first = torch.zeros(batch, dtype=torch.long, device=value.device)
        second = torch.where(
            value[:, 0, 0] > 2,
            torch.zeros_like(first),
            torch.ones_like(first),
        )
        selected = torch.stack((first, second), dim=-1).reshape(batch, 1, 1, 2)
        weights = torch.nn.functional.one_hot(selected, 2).to(value.dtype)
        valid = torch.ones(batch, 1, 1, dtype=torch.bool, device=value.device)
        route = alpha.FormulaRoutePlan(weights, valid, valid, valid)
        availability = torch.ones(
            batch, 2, 2, dtype=torch.bool, device=value.device
        )
        return route, alpha.BankFormulaRouteInfo(
            selected,
            valid,
            valid,
            valid,
            availability,
            self._component_reference,
            self.config_fingerprint,
        )


def _policy(dim: int, seed: int) -> alpha.TypedBankFormulaTopologyPolicy:
    bank = alpha.TypedTopologyOperandBank(
        slots=4,
        key_dim=4,
        factor_dim=1,
        seed=seed,
        value_seed=seed + 100,
        bank_id=f"formula-route-{seed}",
    )
    return alpha.TypedBankFormulaTopologyPolicy(
        dim,
        [bank],
        key_dim=4,
        query_seed=seed + 200,
        diagnostics="none",
    )


def _program(dim: int = 2) -> alpha.FormulaFabricProgram:
    return alpha.FormulaFabricProgram(
        arena_capacity=4,
        feature_dim=dim,
        steps=((alpha.FormulaInvocation(alpha.FormulaPrimitive.ADD, 2),),),
        domain="bank-formula-route",
    )


def _stack(
    *,
    estimator: str = "straight-through",
    weighted: bool = False,
    limits: alpha.ContractLimits | None = None,
) -> tuple[
    alpha.RoutedFormulaFabricCompute,
    alpha.BankFormulaRouteSource,
    alpha.FormulaFabricCompute,
]:
    program = _program()
    limits = alpha.ContractLimits() if limits is None else limits
    candidate = torch.zeros(1, 1, 2, 4, dtype=torch.bool)
    candidate[..., :2] = True
    source = alpha.BankFormulaRouteSource(
        program,
        [_policy(2, 1), _policy(2, 2)],
        active_count=3,
        estimator=estimator,
        candidate_mask=candidate,
        limits=limits,
    )
    fabric: torch.nn.Module = alpha.FormulaFabric(program, limits=limits)
    if weighted:
        fabric = alpha.FormulaCommitBlend(fabric)
    compute = alpha.FormulaFabricCompute(fabric, active_count=3)
    return alpha.RoutedFormulaFabricCompute(compute, source), source, compute


def _workspace(*, requires_grad: bool = False) -> alpha.ActiveWorkspace:
    value = torch.tensor(
        [[[1.0, 2.0], [3.0, 5.0], [9.0, 11.0]]],
        requires_grad=requires_grad,
    )
    support = torch.ones(1, 3, dtype=torch.bool)
    intervened = torch.tensor([[False, False, True]])
    return alpha.ActiveWorkspace(value, support, support, intervened)


def _typed_inputs(
    value: torch.Tensor,
    *,
    intervened: torch.Tensor | None = None,
) -> tuple[alpha.TensorEnvelope, alpha.PulseSupports]:
    mask = torch.ones(value.shape[:-1], dtype=torch.bool, device=value.device)
    if intervened is None:
        intervened = mask
    domain = alpha.SupportDomain.for_tensor(
        mask,
        domain_id="dynamic-routed-formula",
        owner_ref="arti/pulse@2",
        partition_id="world",
        transition_id="joint-path",
    )
    world = alpha.TensorEnvelope(alpha.EnvelopeRef.WORLD, value, mask, domain)
    supports = alpha.PulseSupports(
        alpha.SupportMask(alpha.SupportKind.OBSERVED, mask, domain),
        alpha.SupportMask(alpha.SupportKind.EXPOSED, mask, domain),
        alpha.SupportMask(
            alpha.SupportKind.INTERVENED,
            intervened,
            domain,
        ),
        validity=mask,
    )
    return world, supports


def test_bank_formula_route_matches_explicit_plan() -> None:
    routed, source, compute = _stack()
    workspace = _workspace()
    plan, route_info = source(workspace)

    automatic, info = routed(workspace, return_info=True)
    explicit = compute(workspace, formula_route=plan)

    torch.testing.assert_close(automatic.value, explicit.value)
    assert info.route_origin == "bank-formula"
    assert info.route is not None
    assert info.route.route_source_ref == "arti/bank-formula-route-source@1"
    assert torch.equal(info.route.selected_input_slots, route_info.selected_input_slots)
    assert torch.equal(plan.weights.sum(dim=-1), torch.ones_like(plan.weights[..., 0]))
    assert info.executor_ref == "arti/formula-fabric-compute@1"
    assert info.adapter_ref == "arti/routed-formula-fabric-compute@1"
    assert info.commit_mode == "hard"
    assert info.factor_contract == "forbidden"
    assert info.route_contract == "bound-source-or-explicit-override"
    assert info.arena_layout == "active-prefix-then-scratch"
    assert info.visibility_contract == "unsupported"
    assert len(info.executor_config_fingerprint) == 64
    assert len(info.adapter_config_fingerprint) == 64


def test_compile_safe_policy_path_matches_typed_operands_path() -> None:
    policy = _policy(2, 77)
    workspace = _workspace()
    typed, typed_routes = policy.bank_outputs(workspace.value, workspace.exposed)
    direct, direct_routes = policy.execution_outputs(
        workspace.value, workspace.exposed
    )

    for actual, expected in zip(direct, typed, strict=True):
        torch.testing.assert_close(actual.priority, expected.priority)
        torch.testing.assert_close(actual.confidence, expected.confidence)
    for actual, expected in zip(direct_routes, typed_routes, strict=True):
        torch.testing.assert_close(actual, expected)


def test_routed_compute_supports_dynamic_batch() -> None:
    routed, _source, _compute = _stack()
    for batch in (1, 3, 5):
        base = _workspace()
        workspace = alpha.ActiveWorkspace(
            base.value.expand(batch, -1, -1).clone(),
            base.validity.expand(batch, -1),
            base.exposed.expand(batch, -1),
            base.intervened.expand(batch, -1),
        )
        assert routed(workspace).value.shape == (batch, 3, 2)


def test_candidate_mask_is_exposed_as_an_inspection_copy() -> None:
    _routed, source, _compute = _stack()
    snapshot = source.candidate_mask
    snapshot.zero_()

    assert source.candidate_mask.any()


def test_policy_config_and_pin_order_are_bound_to_provenance() -> None:
    _routed, source, _compute = _stack()
    original = arti.component_spec(source).config_fingerprint
    source.policies[0], source.policies[1] = source.policies[1], source.policies[0]
    reordered = arti.component_spec(source).config_fingerprint

    assert reordered != original


def test_pulse_hot_path_does_not_query_component_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    routed, _source, _compute = _stack()
    topology = alpha.ReversibleTopology(active_count=3)
    fold, unfold = topology.operations()
    pulse = alpha.AdaptivePulse(
        fold=fold,
        selective_compute=routed,
        unfold=unfold,
    )
    registry = arti.get_component_registry()

    def fail_if_queried(_value: object) -> object:
        raise AssertionError("component registry was queried in Pulse.forward")

    monkeypatch.setattr(registry, "registration_for", fail_if_queried)
    assert pulse.run_tensor(
        _workspace().value,
        intervened=torch.tensor([[False, False, True]]),
    ).value.shape == (1, 3, 2)


def test_explicit_manifest_validation_rejects_policy_drift() -> None:
    routed, source, _compute = _stack()
    topology = alpha.ReversibleTopology(active_count=3)
    fold, unfold = topology.operations()
    pulse = alpha.AdaptivePulse(
        fold=fold,
        selective_compute=routed,
        unfold=unfold,
    )
    source.policies[0], source.policies[1] = source.policies[1], source.policies[0]

    with pytest.raises(ValueError, match="drifted from its manifest"):
        pulse.validate_manifest()


def test_explicit_route_skips_route_source(monkeypatch: pytest.MonkeyPatch) -> None:
    routed, source, _compute = _stack()
    workspace = _workspace()
    plan, _ = source(workspace)

    def fail_if_called(_workspace: object) -> object:
        raise AssertionError("route source executed despite explicit override")

    monkeypatch.setattr(source, "forward", fail_if_called)
    _updated, info = routed(workspace, formula_route=plan, return_info=True)

    assert info.route_origin == "explicit"
    assert info.route is None
    assert info.route_source_ref is None


def test_straight_through_route_trains_bank_values() -> None:
    routed, source, _compute = _stack()
    workspace = _workspace(requires_grad=True)

    updated = routed(workspace)
    updated.value[:, 2].square().mean().backward()

    grads = [
        policy.banks[0].values.grad
        for policy in source.policies
    ]
    assert workspace.value.grad is not None
    assert all(grad is not None and torch.isfinite(grad).all() for grad in grads)
    assert any(torch.count_nonzero(grad).item() > 0 for grad in grads if grad is not None)


def test_hard_route_has_no_surrogate_bank_gradient() -> None:
    routed, source, _compute = _stack(estimator="hard")
    workspace = _workspace(requires_grad=True)

    routed(workspace).value.square().mean().backward()

    assert all(policy.banks[0].values.grad is None for policy in source.policies)


def test_route_source_tracks_scratch_availability() -> None:
    program = alpha.FormulaFabricProgram(
        arena_capacity=4,
        feature_dim=2,
        steps=(
            (alpha.FormulaInvocation(alpha.FormulaPrimitive.ADD, 3),),
            (alpha.FormulaInvocation(alpha.FormulaPrimitive.ADD, 2),),
        ),
        domain="scratch-route",
    )
    candidate = torch.zeros(2, 1, 2, 4, dtype=torch.bool)
    candidate[0, 0, :, :2] = True
    candidate[1, 0, 0, 3] = True
    candidate[1, 0, 1, 0] = True
    source = alpha.BankFormulaRouteSource(
        program,
        [_policy(2, seed) for seed in range(10, 14)],
        active_count=3,
        candidate_mask=candidate,
    )
    plan, info = source(_workspace())

    assert not bool(info.availability[:, 0, 3].any())
    assert bool(info.availability[:, 1, 3].all())
    assert plan.fire_mask.all()
    assert plan.commit_mask.all()
    assert torch.equal(info.selected_input_slots[:, 1, 0, 0], torch.tensor([3]))


def test_route_source_preserves_host_write_authority() -> None:
    _routed, source, _compute = _stack()
    workspace = _workspace()
    denied = workspace.replace(intervened=torch.zeros_like(workspace.intervened))

    plan, _info = source(denied)

    assert plan.fire_mask.all()
    assert not plan.commit_mask.any()


def test_route_source_rejects_joint_budget_before_bank_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    program = _program()
    policies = [_policy(2, 31), _policy(2, 32)]
    source = alpha.BankFormulaRouteSource(
        program,
        policies,
        active_count=3,
        limits=alpha.ContractLimits(max_operation_bytes=1),
    )
    called: list[bool] = []

    def fail_if_read(_query: torch.Tensor) -> object:
        called.append(True)
        raise AssertionError("Bank read occurred before route admission")

    monkeypatch.setattr(policies[0].banks[0], "read", fail_if_read)
    with pytest.raises(ValueError, match="operation byte limits"):
        source(_workspace())
    assert not called


def test_routed_compute_runs_as_pulse_stage() -> None:
    routed, _source, _compute = _stack()
    topology = alpha.ReversibleTopology(active_count=3)
    fold, unfold = topology.operations()
    pulse = alpha.AdaptivePulse(
        fold=fold,
        selective_compute=routed,
        unfold=unfold,
    )
    value = _workspace().value
    result = pulse.run_tensor(
        value,
        intervened=torch.tensor([[False, False, True]]),
    )

    assert result.value.shape == value.shape
    assert isinstance(result.diagnostics.compute, alpha.RoutedFormulaFabricComputeInfo)
    assert result.diagnostics.compute.route_origin == "bank-formula"


def test_dynamic_fold_weighted_routed_formula_unfold_joint_path() -> None:
    routed, _route_source, _compute = _stack(weighted=True)
    topology = alpha.ReversibleTopology(active_count=3)
    fold, unfold = topology.operations()
    pulse = alpha.AdaptivePulse(
        fold=fold,
        selective_compute=routed,
        unfold=unfold,
    )
    source = _DynamicFoldSource()
    value = torch.tensor(
        [[[50.0, 60.0], [1.0, 2.0], [3.0, 5.0], [9.0, 11.0]]],
        requires_grad=True,
    )
    keys = torch.tensor(
        [[[0.0, 0.0], [4.0, 0.0], [3.0, 0.0], [2.0, 0.0]]]
    )
    query = torch.tensor([[1.0, 0.0]])
    factors = value.new_full((1, 1, 1), 0.5, requires_grad=True)
    world, supports = _typed_inputs(
        value,
        intervened=torch.tensor([[False, True, True, True]]),
    )

    result = pulse(
        world,
        supports,
        compute_factors=factors,
        topology_source=source,
        topology_source_inputs=(keys, query),
    )

    record = result.diagnostics.topology_record
    info = result.diagnostics.compute
    assert record is not None
    assert torch.equal(record.active_index, torch.tensor([[1, 2, 3]]))
    torch.testing.assert_close(result.value[:, 0], value[:, 0])
    assert isinstance(info, alpha.RoutedFormulaFabricComputeInfo)
    assert isinstance(info.trace, alpha.FormulaCommitBlendTrace)
    assert info.route_origin == "bank-formula"
    assert info.commit_mode == "weighted"
    assert info.factor_contract == "required"
    result.value.square().mean().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert factors.grad is not None and torch.isfinite(factors.grad).all()


def test_dynamic_fold_routed_formula_pulse_component_graph_and_round_trip(
    tmp_path,
) -> None:
    routed, _route_source, _compute = _stack(weighted=True)
    topology = alpha.ReversibleTopology(active_count=3)
    fold, unfold = topology.operations()
    source = alpha.AdaptivePulse(
        fold=fold,
        selective_compute=routed,
        unfold=unfold,
    ).eval()
    topology_source = _DynamicFoldSource().eval()
    value = torch.tensor(
        [[[50.0, 60.0], [1.0, 2.0], [3.0, 5.0], [9.0, 11.0]]]
    )
    keys = torch.tensor(
        [[[0.0, 0.0], [4.0, 0.0], [3.0, 0.0], [2.0, 0.0]]]
    )
    query = torch.tensor([[1.0, 0.0]])
    factors = value.new_full((1, 1, 1), 0.5)
    world, supports = _typed_inputs(
        value,
        intervened=torch.tensor([[False, True, True, True]]),
    )
    expected = source(
        world,
        supports,
        compute_factors=factors,
        topology_source=topology_source,
        topology_source_inputs=(keys, query),
    )

    saved = arti.save(source, tmp_path / "dynamic-routed-formula.arti.st")
    target_routed, _target_route_source, _target_compute = _stack(weighted=True)
    target_topology = alpha.ReversibleTopology(active_count=3)
    target_fold, target_unfold = target_topology.operations()
    target = alpha.AdaptivePulse(
        fold=target_fold,
        selective_compute=target_routed,
        unfold=target_unfold,
    ).eval()
    loaded = arti.load(saved.weights_path, model=target)
    actual = target(
        world,
        supports,
        compute_factors=factors,
        topology_source=topology_source,
        topology_source_inputs=(keys, query),
    )
    refs = {
        node["ref"]
        for node in loaded.manifest["architecture"]["component_graph"]["nodes"]
    }

    torch.testing.assert_close(actual.value, expected.value)
    actual_info = actual.diagnostics.compute
    expected_info = expected.diagnostics.compute
    assert isinstance(actual_info, alpha.RoutedFormulaFabricComputeInfo)
    assert isinstance(expected_info, alpha.RoutedFormulaFabricComputeInfo)
    assert actual_info.executor_config_fingerprint == (
        expected_info.executor_config_fingerprint
    )
    assert actual_info.adapter_config_fingerprint == (
        expected_info.adapter_config_fingerprint
    )
    assert isinstance(actual_info.trace, alpha.FormulaCommitBlendTrace)
    assert isinstance(expected_info.trace, alpha.FormulaCommitBlendTrace)
    torch.testing.assert_close(actual_info.trace.weights, expected_info.trace.weights)
    assert actual_info.trace.formula.program_fingerprint == (
        expected_info.trace.formula.program_fingerprint
    )
    assert refs >= {
        "arti/pulse@2",
        "arti/fold@2",
        "arti/unfold@2",
        "arti/routed-formula-fabric-compute@1",
        "arti/bank-formula-route-source@1",
        "arti/formula-fabric-compute@1",
        "arti/formula-commit-blend@1",
        "arti/formula-fabric@1",
    }


def test_fold_source_contract_binding_is_manifest_owned_and_fail_closed() -> None:
    source = _DynamicFoldSource()
    fold = alpha.Fold(active_count=2).bind_source_contract(source)
    binding = arti.component_spec(fold).config["source_contract_binding"]

    assert binding["source_ref"] == source._component_reference
    assert len(binding["source_contract_fingerprint"]) == 64
    assert binding["source_input_count"] == 2
    assert binding["source_instance_axes"] == [1, None]

    class DriftedSource(_DynamicFoldSource):
        def topology_contract(self) -> dict[str, object]:
            contract = super().topology_contract()
            contract["input_schema"] = ["changed[B,N,D]", "query[B,D]"]
            return contract

    with pytest.raises(ValueError, match="bound source contract"):
        fold.from_source(
            torch.randn(1, 4, 2),
            source=DriftedSource(),
            source_inputs=(torch.randn(1, 4, 2), torch.randn(1, 2)),
        )


def test_fold_source_preflight_rejects_payload_aliases_and_binding_reuse() -> None:
    source = _DynamicFoldSource()
    fold = alpha.Fold(active_count=2).bind_source_contract(source)
    payload = torch.randn(1, 4, 2)
    query = torch.randn(1, 2)

    public_parameters = inspect.signature(
        alpha.ReversibleTopology.fold_from_source
    ).parameters
    assert "source_provenance_fingerprint" not in public_parameters
    assert "_source_binding" not in public_parameters
    with pytest.raises(ValueError, match="payload must not enter"):
        fold.prepare_source_inputs(payload, (payload.view_as(payload), query))
    with pytest.raises(ValueError, match="payload must not enter"):
        fold.prepare_source_inputs(payload, (payload[:, :, :], query))

    prepared = fold.prepare_source_inputs(
        payload,
        (torch.randn(1, 4, 2), query),
    )
    other_fold = alpha.Fold(active_count=2).bind_source_contract(source)
    with pytest.raises(ValueError, match="another binding"):
        other_fold.from_source(
            payload,
            source=source,
            source_inputs=prepared,
        )


def test_bound_fold_component_round_trip_requires_matching_rebind(tmp_path) -> None:
    source = _DynamicFoldSource()
    fold = alpha.Fold(active_count=2).bind_source_contract(source).eval()
    saved = arti.save(fold, tmp_path / "bound-fold.arti.st")

    target_source = _DynamicFoldSource()
    target = alpha.Fold(active_count=2).bind_source_contract(target_source).eval()
    arti.load(saved.weights_path, model=target)

    unbound = alpha.Fold(active_count=2).eval()
    with pytest.raises(ValueError, match="component (graph|state contract) does not match"):
        arti.load(saved.weights_path, model=unbound)


def test_iterative_routed_formula_requeries_committed_workspace() -> None:
    program = alpha.FormulaFabricProgram(
        arena_capacity=2,
        feature_dim=1,
        steps=((alpha.FormulaInvocation(alpha.FormulaPrimitive.ADD, 0),),),
        domain="iterative-route-test",
    )
    compute = alpha.FormulaFabricCompute(
        alpha.FormulaFabric(program),
        active_count=2,
    )
    source = _StateSwitchingRouteSource(program)
    routed = alpha.RoutedFormulaFabricCompute(compute, source)
    iterative = alpha.IterativeRoutedFormulaFabricCompute(routed, steps=2)
    value = torch.tensor([[[1.0], [2.0]]], requires_grad=True)
    support = torch.ones(1, 2, dtype=torch.bool)
    workspace = alpha.ActiveWorkspace(value, support, support, support)

    initial_route, _ = source(workspace)
    frozen_once = compute(workspace, formula_route=initial_route)
    frozen_twice = compute(frozen_once, formula_route=initial_route)
    actual, info = iterative(workspace, return_info=True)

    torch.testing.assert_close(frozen_twice.value[:, 0], torch.tensor([[5.0]]))
    torch.testing.assert_close(actual.value[:, 0], torch.tensor([[6.0]]))
    assert info.route_semantics == "requery-after-program"
    assert info.executed_steps == 2
    first_route = info.iterations[0].route
    second_route = info.iterations[1].route
    assert first_route is not None and second_route is not None
    assert first_route.selected_input_slots[0, 0, 0, 1] == 1
    assert second_route.selected_input_slots[0, 0, 0, 1] == 0
    actual.value.sum().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()


def test_iterative_routed_formula_component_round_trip(tmp_path) -> None:
    routed, _route_source, _compute = _stack()
    source = alpha.IterativeRoutedFormulaFabricCompute(routed, steps=3).eval()
    expected = source(_workspace()).value
    saved = arti.save(source, tmp_path / "iterative-route.arti.st")
    target_routed, _target_route_source, _target_compute = _stack()
    target = alpha.IterativeRoutedFormulaFabricCompute(
        target_routed,
        steps=3,
    ).eval()
    loaded = arti.load(saved.weights_path, model=target)
    actual = target(_workspace()).value
    refs = {
        node["ref"]
        for node in loaded.manifest["architecture"]["component_graph"]["nodes"]
    }

    torch.testing.assert_close(actual, expected)
    assert refs >= {
        "arti/iterative-routed-formula-fabric-compute@1",
        "arti/routed-formula-fabric-compute@1",
        "arti/formula-fabric-compute@1",
        "arti/formula-fabric@1",
    }


def test_iterative_routed_formula_admits_cumulative_cost_before_route() -> None:
    routed, source, compute = _stack()
    workspace = _workspace()
    per_iteration = source.operation_bytes_upper_bound(
        workspace
    ) + compute.operation_bytes_upper_bound(workspace)
    tight = alpha.ContractLimits(max_operation_bytes=per_iteration + 1)
    routed, source, compute = _stack(limits=tight)
    iterative = alpha.IterativeRoutedFormulaFabricCompute(routed, steps=2)

    with pytest.raises(ValueError, match="cumulative operation byte limits"):
        iterative(workspace)


def test_route_limits_change_transitive_runtime_fingerprints() -> None:
    default_routed, default_source, _ = _stack()
    tight_routed, tight_source, _ = _stack(
        limits=alpha.ContractLimits(max_operation_bytes=1_000_000)
    )
    default_iterative = alpha.IterativeRoutedFormulaFabricCompute(
        default_routed,
        steps=2,
    )
    tight_iterative = alpha.IterativeRoutedFormulaFabricCompute(
        tight_routed,
        steps=2,
    )

    assert default_source.config_fingerprint != tight_source.config_fingerprint
    assert default_routed.config_fingerprint != tight_routed.config_fingerprint
    assert default_iterative.config_fingerprint != tight_iterative.config_fingerprint
    with pytest.raises(AttributeError):
        default_source.limits = alpha.ContractLimits(max_operation_bytes=1_000_000)


def test_routed_compute_component_graph_and_round_trip(tmp_path) -> None:
    routed, _source, _compute = _stack()
    refs = {
        node["ref"] for node in arti.component_provenance(routed)["components"]
    }
    assert refs >= {
        "arti/routed-formula-fabric-compute@1",
        "arti/bank-formula-route-source@1",
        "arti/formula-fabric-compute@1",
        "arti/formula-fabric@1",
        "arti/bank-formula-topology-policy@2",
    }

    workspace = _workspace()
    expected = routed(workspace).value
    saved = arti.save(routed, tmp_path / "routed-formula.arti.st")
    target, _source, _compute = _stack()
    arti.load(saved.weights_path, model=target)
    actual = target(workspace).value
    torch.testing.assert_close(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_routed_compute_cuda_fullgraph(dtype: torch.dtype) -> None:
    routed, source, _compute = _stack()
    routed = routed.cuda().to(dtype).train()
    compiled = torch.compile(routed, backend="inductor", fullgraph=True)
    value = torch.tensor(
        [[[1.0, 2.0], [3.0, 5.0], [9.0, 11.0]]],
        device="cuda",
        dtype=dtype,
    ).expand(2, -1, -1).clone().requires_grad_(True)
    support = torch.ones(2, 3, dtype=torch.bool, device="cuda")
    intervened = torch.tensor(
        [[False, False, True]], device="cuda"
    ).expand(2, -1)
    workspace = alpha.ActiveWorkspace(value, support, support, intervened)

    actual = compiled(workspace).value
    actual.float().square().mean().backward()

    assert actual.is_cuda and actual.dtype is dtype
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert any(
        policy.banks[0].values.grad is not None
        for policy in source.policies
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_routed_formula_full_pulse_cuda_fullgraph() -> None:
    routed, _source, _compute = _stack()
    topology = alpha.ReversibleTopology(active_count=3)
    fold, unfold = topology.operations()
    pulse = alpha.AdaptivePulse(
        fold=fold,
        selective_compute=routed,
        unfold=unfold,
    ).cuda().train()
    compiled = torch.compile(pulse, backend="inductor", fullgraph=True)
    value = torch.tensor(
        [[[1.0, 2.0], [3.0, 5.0], [9.0, 11.0]]],
        device="cuda",
    ).expand(2, -1, -1).clone().requires_grad_(True)
    mask = torch.ones(2, 3, dtype=torch.bool, device="cuda")
    intervened = torch.tensor(
        [[False, False, True]], device="cuda"
    ).expand(2, -1)
    domain = alpha.SupportDomain.for_tensor(
        mask,
        domain_id="compiled-route-pulse",
        owner_ref="arti/pulse@2",
        partition_id="world",
        transition_id="fullgraph",
    )
    world = alpha.TensorEnvelope(alpha.EnvelopeRef.WORLD, value, mask, domain)
    supports = alpha.PulseSupports(
        alpha.SupportMask(alpha.SupportKind.OBSERVED, mask, domain),
        alpha.SupportMask(alpha.SupportKind.EXPOSED, mask, domain),
        alpha.SupportMask(alpha.SupportKind.INTERVENED, intervened, domain),
        validity=mask,
    )

    actual = compiled(world, supports).value
    actual.square().mean().backward()

    assert value.grad is not None and torch.isfinite(value.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_iterative_routed_formula_cuda_fullgraph(dtype: torch.dtype) -> None:
    routed, route_source, _compute = _stack()
    iterative = alpha.IterativeRoutedFormulaFabricCompute(
        routed,
        steps=3,
    ).cuda().to(dtype).train()
    compiled = torch.compile(iterative, backend="inductor", fullgraph=True)
    value = torch.tensor(
        [[[1.0, 2.0], [3.0, 5.0], [9.0, 11.0]]],
        device="cuda",
        dtype=dtype,
    ).expand(2, -1, -1).clone().requires_grad_(True)
    support = torch.ones(2, 3, dtype=torch.bool, device="cuda")
    workspace = alpha.ActiveWorkspace(value, support, support, support)

    actual = compiled(workspace).value
    actual.float().square().mean().backward()

    assert actual.is_cuda and actual.dtype is dtype
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert any(
        policy.banks[0].values.grad is not None
        for policy in route_source.policies
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_iterative_routed_formula_pulse_cuda_fullgraph_diagnostics(
    dtype: torch.dtype,
) -> None:
    routed, route_source, _compute = _stack()
    iterative = alpha.IterativeRoutedFormulaFabricCompute(routed, steps=3)
    topology = alpha.ReversibleTopology(active_count=3)
    fold, unfold = topology.operations()
    pulse = alpha.AdaptivePulse(
        fold=fold,
        selective_compute=iterative,
        unfold=unfold,
    ).cuda().to(dtype).train()
    compiled = torch.compile(pulse, backend="inductor", fullgraph=True)
    value = torch.tensor(
        [[[1.0, 2.0], [3.0, 5.0], [9.0, 11.0]]],
        device="cuda",
        dtype=dtype,
    ).expand(2, -1, -1).clone().requires_grad_(True)
    world, supports = _typed_inputs(value)

    result = compiled(world, supports)
    result.value.float().square().mean().backward()

    info = result.diagnostics.compute
    assert isinstance(info, alpha.IterativeRoutedFormulaFabricComputeInfo)
    assert info.executed_steps == 3
    assert len(info.iterations) == 3
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert any(
        policy.banks[0].values.grad is not None
        for policy in route_source.policies
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_dynamic_fold_weighted_routed_formula_cuda_fullgraph(
    dtype: torch.dtype,
) -> None:
    class CompiledJointPath(torch.nn.Module):
        def __init__(
            self,
            pulse: alpha.AdaptivePulse,
            topology_source: torch.nn.Module,
        ) -> None:
            super().__init__()
            self.pulse = pulse
            self.topology_source = topology_source

        def forward(
            self,
            world: alpha.TensorEnvelope,
            supports: alpha.PulseSupports,
            factors: torch.Tensor,
            prepared_inputs: object,
        ) -> alpha.PulseOutput:
            return self.pulse.forward(
                world,
                supports,
                compute_factors=factors,
                topology_source=self.topology_source,
                topology_source_inputs=prepared_inputs,
            )

    routed, route_source, _compute = _stack(weighted=True)
    topology = alpha.ReversibleTopology(
        active_count=3,
        surrogate=alpha.SoftTopKTopologySurrogate(),
    )
    fold, unfold = topology.operations()
    topology_source = _DynamicFoldSource().cuda().to(dtype).train()
    fold.bind_source_contract(topology_source)
    pulse = alpha.AdaptivePulse(
        fold=fold,
        selective_compute=routed,
        unfold=unfold,
    ).cuda().to(dtype).train()
    compiled = torch.compile(
        CompiledJointPath(pulse, topology_source),
        backend="inductor",
        fullgraph=True,
    )
    value = torch.tensor(
        [[[50.0, 60.0], [1.0, 2.0], [3.0, 5.0], [9.0, 11.0]]],
        device="cuda",
        dtype=dtype,
    ).expand(2, -1, -1).clone().requires_grad_(True)
    keys = torch.tensor(
        [[[0.0, 0.0], [4.0, 0.0], [3.0, 0.0], [2.0, 0.0]]],
        device="cuda",
        dtype=dtype,
    ).expand(2, -1, -1)
    query = torch.tensor(
        [[1.0, 0.0]],
        device="cuda",
        dtype=dtype,
    ).expand(2, -1)
    factors = torch.full(
        (2, 1, 1),
        0.5,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    world, supports = _typed_inputs(
        value,
        intervened=torch.tensor(
            [[False, True, True, True]], device="cuda"
        ).expand(2, -1),
    )

    prepared_inputs = fold.prepare_source_inputs(value, (keys, query))
    result = compiled(world, supports, factors, prepared_inputs)
    result.value.float().square().mean().backward()

    assert result.value.is_cuda and result.value.dtype is dtype
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert factors.grad is not None and torch.isfinite(factors.grad).all()
    assert topology_source.weight.grad is not None
    assert torch.isfinite(topology_source.weight.grad).all()
    assert any(
        policy.banks[0].values.grad is not None
        for policy in route_source.policies
    )
