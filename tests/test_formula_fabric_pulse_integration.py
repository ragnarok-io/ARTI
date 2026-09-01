from __future__ import annotations

import pytest
import torch

import arti
from arti import mechanisms


class _PulseTopologySource(torch.nn.Module):
    _component_reference = "example/pulse-topology-source@1"

    def __init__(self, dim: int = 2) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.eye(dim))
        self.calls = 0

    def propose(
        self,
        keys: torch.Tensor,
        query: torch.Tensor,
        *,
        mask: torch.Tensor,
    ) -> mechanisms.TopologyProposal:
        self.calls += 1
        priority = torch.einsum("bnd,bd->bn", keys @ self.weight, query)
        return mechanisms.TopologyProposal(mechanisms.TopologyAction(priority))

    def topology_contract(self) -> dict[str, object]:
        return {
            "ref": self._component_reference,
            "input_schema": ["keys[B,N,D]", "query[B,D]"],
            "input_instance_axes": [1, None],
            "output": "arti/topology-proposal@1",
        }


def _pulse(dim: int = 2) -> mechanisms.AdaptivePulse:
    topology = mechanisms.ReversibleTopology(active_count=3)
    fold, unfold = topology.operations()
    program = mechanisms.FormulaFabricProgram(
        arena_capacity=4,
        feature_dim=dim,
        steps=((mechanisms.FormulaInvocation(mechanisms.FormulaPrimitive.ADD, 2),),),
        domain="pulse-active",
    )
    return mechanisms.AdaptivePulse(
        fold=fold,
        intervention=mechanisms.FormulaAttention(
            mechanisms.MagnitudeInterventionPolicy(),
            mechanisms.StableTopKIntervention(1),
        ),
        selective_compute=mechanisms.FormulaFabricCompute(
            mechanisms.FormulaFabric(program), active_count=3
        ),
        unfold=unfold,
    )


def _inputs(value: torch.Tensor) -> tuple[mechanisms.TensorEnvelope, mechanisms.PulseSupports]:
    mask = torch.ones(value.shape[:-1], dtype=torch.bool, device=value.device)
    domain = mechanisms.SupportDomain.for_tensor(
        mask,
        domain_id="formula-pulse",
        owner_ref="arti/pulse@2",
        partition_id="world",
        transition_id="fixed-route",
    )
    world = mechanisms.TensorEnvelope(mechanisms.EnvelopeRef.WORLD, value, mask, domain)
    supports = mechanisms.PulseSupports(
        mechanisms.SupportMask(mechanisms.SupportKind.OBSERVED, mask, domain),
        mechanisms.SupportMask(mechanisms.SupportKind.EXPOSED, mask, domain),
        mechanisms.SupportMask(
            mechanisms.SupportKind.INTERVENED, torch.zeros_like(mask), domain
        ),
        validity=mask,
    )
    return world, supports


def _route(value: torch.Tensor) -> mechanisms.FormulaRoutePlan:
    weights = value.new_zeros((value.shape[0], 1, 1, 2, 4))
    weights[:, 0, 0, 0, 0] = 1
    weights[:, 0, 0, 1, 1] = 1
    enabled = torch.ones(value.shape[0], 1, 1, dtype=torch.bool, device=value.device)
    return mechanisms.FormulaRoutePlan(weights, enabled, enabled, enabled)


def _scratch_pulse(dim: int = 2) -> mechanisms.AdaptivePulse:
    topology = mechanisms.ReversibleTopology(active_count=3)
    fold, unfold = topology.operations()
    program = mechanisms.FormulaFabricProgram(
        arena_capacity=4,
        feature_dim=dim,
        steps=(
            (mechanisms.FormulaInvocation(mechanisms.FormulaPrimitive.ADD, 3),),
            (mechanisms.FormulaInvocation(mechanisms.FormulaPrimitive.ADD, 2),),
        ),
        domain="pulse-active",
    )
    return mechanisms.AdaptivePulse(
        fold=fold,
        intervention=mechanisms.FormulaAttention(
            mechanisms.MagnitudeInterventionPolicy(), mechanisms.StableTopKIntervention(1)
        ),
        selective_compute=mechanisms.FormulaFabricCompute(
            mechanisms.FormulaFabric(program), active_count=3
        ),
        unfold=unfold,
    )


def _scratch_route(value: torch.Tensor) -> mechanisms.FormulaRoutePlan:
    weights = value.new_zeros((value.shape[0], 2, 1, 2, 4))
    weights[:, 0, 0, 0, 0] = 1
    weights[:, 0, 0, 1, 1] = 1
    weights[:, 1, 0, 0, 3] = 1
    weights[:, 1, 0, 1, 0] = 1
    enabled = torch.ones(value.shape[0], 2, 1, dtype=torch.bool, device=value.device)
    return mechanisms.FormulaRoutePlan(weights, enabled, enabled, enabled)


def test_formula_fabric_compute_runs_inside_pulse_and_returns_trace() -> None:
    value = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [0.5, 0.5]]],
        requires_grad=True,
    )
    pulse = _pulse()
    world, supports = _inputs(value)

    result = pulse(world, supports, formula_route=_route(value))

    expected = value.detach().clone()
    expected[:, 2] = value.detach()[:, 0] + value.detach()[:, 1]
    torch.testing.assert_close(result.value, expected)
    assert isinstance(result.diagnostics.compute, mechanisms.FormulaFabricTrace)
    assert result.diagnostics.topology_record is not None
    result.value.sum().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()


def test_run_tensor_matches_typed_pulse_entry() -> None:
    value = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [10.0, 12.0], [0.5, 0.5]]]
    )
    pulse = _pulse()
    world, supports = _inputs(value)
    route = _route(value)

    typed = pulse(world, supports, formula_route=route)
    direct = pulse.run_tensor(value, formula_route=route)

    torch.testing.assert_close(direct.value, typed.value)
    assert direct.diagnostics.compute.program_fingerprint == (
        typed.diagnostics.compute.program_fingerprint
    )


def test_pulse_accepts_source_driven_topology_and_preserves_payload() -> None:
    value = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]]
    )
    source = _PulseTopologySource()
    keys = torch.tensor(
        [[[1.0, 0.0], [4.0, 0.0], [3.0, 0.0], [2.0, 0.0]]]
    )
    query = torch.tensor([[1.0, 0.0]])
    topology = mechanisms.ReversibleTopology(active_count=3)
    fold, unfold = topology.operations()
    pulse = mechanisms.AdaptivePulse(fold=fold, unfold=unfold)
    world, supports = _inputs(value)

    result = pulse(
        world,
        supports,
        topology_source=source,
        topology_source_inputs=(keys, query),
    )

    assert source.calls == 1
    assert result.diagnostics.topology_record is not None
    assert torch.equal(
        result.diagnostics.topology_record.active_index,
        torch.tensor([[1, 2, 3]]),
    )
    torch.testing.assert_close(result.value, value)


def test_source_driven_pulse_topology_is_bounded_by_observed_support() -> None:
    value = torch.randn(1, 4, 2)
    source = _PulseTopologySource()
    keys = torch.tensor(
        [[[100.0, 0.0], [4.0, 0.0], [3.0, 0.0], [2.0, 0.0]]]
    )
    query = torch.tensor([[1.0, 0.0]])
    pulse = mechanisms.AdaptivePulse(
        fold=mechanisms.Fold(active_count=2),
        unfold=mechanisms.UnFold(active_count=2),
    )

    result = pulse.run_tensor(
        value,
        observed=torch.tensor([[False, True, True, True]]),
        topology_source=source,
        topology_source_inputs=(keys, query),
    )

    assert result.diagnostics.topology_record is not None
    assert torch.equal(
        result.diagnostics.topology_record.active_index,
        torch.tensor([[1, 2]]),
    )


def test_pulse_rejects_partial_or_unbound_topology_source_arguments() -> None:
    value = torch.randn(1, 4, 2)
    source = _PulseTopologySource()
    keys = torch.randn(1, 4, 2)
    query = torch.randn(1, 2)

    with pytest.raises(ValueError, match="must be supplied together"):
        _pulse().run_tensor(value, topology_source=source)
    with pytest.raises(ValueError, match="requires an enabled Fold"):
        mechanisms.AdaptivePulse().run_tensor(
            value,
            topology_source=source,
            topology_source_inputs=(keys, query),
        )


def test_run_tensor_rejects_implicit_bank_write_authority() -> None:
    pulse = mechanisms.AdaptivePulse(
        aggregate=mechanisms.ReunionAggregate(mechanisms.SoftFoldAggregate(k=1, dim=2)),
        bank_update=mechanisms.TargetBankUpdater(hidden_dim=2, slots=2),
    )

    with pytest.raises(ValueError, match="cannot infer typed Bank write authority"):
        pulse.run_tensor(torch.randn(1, 3, 2))


def test_run_tensor_preserves_complex_identity_for_disabled_pulse() -> None:
    value = torch.randn(1, 3, 2, dtype=torch.complex64)

    result = mechanisms.AdaptivePulse().run_tensor(value)

    assert result.value is value
    assert result.value.dtype is torch.complex64


@pytest.mark.parametrize("argument", ["bank_state", "write_exposure", "write_policy"])
def test_run_tensor_rejects_write_arguments_without_authority(argument: str) -> None:
    value = torch.randn(1, 3, 2)

    with pytest.raises(ValueError, match="does not accept Bank write arguments"):
        mechanisms.AdaptivePulse().run_tensor(value, **{argument: object()})


def test_run_tensor_custom_supports_match_typed_forward() -> None:
    pulse = mechanisms.AdaptivePulse()
    value = torch.randn(1, 4, 2)
    mask = torch.tensor([[True, True, True, False]])
    observed = torch.tensor([[True, True, False, False]])
    exposed = torch.tensor([[True, False, False, False]])
    intervened = torch.zeros_like(mask)
    domain = mechanisms.SupportDomain.for_tensor(
        mask,
        domain_id="custom-supports",
        owner_ref="arti/pulse@2",
        partition_id="world",
        transition_id="custom",
    )
    world = mechanisms.TensorEnvelope(mechanisms.EnvelopeRef.WORLD, value, mask, domain)
    supports = mechanisms.PulseSupports(
        mechanisms.SupportMask(mechanisms.SupportKind.OBSERVED, observed, domain),
        mechanisms.SupportMask(mechanisms.SupportKind.EXPOSED, exposed, domain),
        mechanisms.SupportMask(mechanisms.SupportKind.INTERVENED, intervened, domain),
        validity=mask,
    )

    typed = pulse(world, supports)
    direct = pulse.run_tensor(
        value,
        mask=mask,
        observed=observed,
        exposed=exposed,
        intervened=intervened,
        domain_id="custom-supports",
        transition_id="custom",
    )

    assert direct.value is typed.value
    assert torch.equal(direct.source_supports.exposed.mask, exposed)


def test_run_tensor_rejects_invalid_support_subset() -> None:
    value = torch.randn(1, 3, 2)
    observed = torch.tensor([[True, False, False]])
    exposed = torch.tensor([[True, True, False]])

    with pytest.raises(ValueError, match="subset"):
        mechanisms.AdaptivePulse().run_tensor(
            value, observed=observed, exposed=exposed
        )


def test_formula_fabric_compute_supports_dynamic_batch() -> None:
    pulse = _pulse()
    for batch in (1, 3, 5):
        value = torch.tensor(
            [[[1.0, 2.0], [3.0, 4.0], [10.0, 12.0], [0.5, 0.5]]]
        ).expand(batch, -1, -1).clone()
        world, supports = _inputs(value)
        actual = pulse(world, supports, formula_route=_route(value)).value
        assert actual.shape == value.shape


def test_formula_fabric_compute_uses_scratch_without_leaking_it() -> None:
    value = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [10.0, 12.0], [0.5, 0.5]]]
    )
    pulse = _scratch_pulse()
    world, supports = _inputs(value)

    actual = pulse(world, supports, formula_route=_scratch_route(value)).value

    expected = value.clone()
    expected[:, 2] = 2 * value[:, 0] + value[:, 1]
    torch.testing.assert_close(actual, expected)


def test_disabled_formula_route_preserves_ragged_payload() -> None:
    value = torch.tensor(
        [[[1.0, 2.0], [9.0, 9.0], [5.0, 6.0], [7.0, 8.0]]]
    )
    validity = torch.tensor([[True, True, True, False]])
    exposed = torch.tensor([[True, False, True, False]])
    domain = mechanisms.SupportDomain.for_tensor(
        validity,
        domain_id="formula-ragged",
        owner_ref="arti/pulse@2",
        partition_id="world",
        transition_id="disabled-route",
    )
    world = mechanisms.TensorEnvelope(mechanisms.EnvelopeRef.WORLD, value, validity, domain)
    supports = mechanisms.PulseSupports(
        mechanisms.SupportMask(mechanisms.SupportKind.OBSERVED, validity, domain),
        mechanisms.SupportMask(mechanisms.SupportKind.EXPOSED, exposed, domain),
        mechanisms.SupportMask(
            mechanisms.SupportKind.INTERVENED, torch.zeros_like(validity), domain
        ),
        validity=validity,
    )
    route = _route(value)
    disabled = torch.zeros_like(route.fire_mask)
    route = mechanisms.FormulaRoutePlan(
        route.weights, route.valid_mask, disabled, disabled
    )

    actual = _pulse()(world, supports, formula_route=route).value

    torch.testing.assert_close(actual, value)


def test_formula_fabric_compute_rejects_unauthorized_commit() -> None:
    pulse = _pulse()
    value = torch.tensor([[[9.0, 9.0], [3.0, 4.0], [1.0, 1.0], [0.5, 0.5]]])
    world, supports = _inputs(value)

    with pytest.raises(ValueError, match="without intervention authority"):
        pulse(world, supports, formula_route=_route(value))


def test_formula_fabric_compute_rejects_ignored_runtime_operands() -> None:
    pulse = _pulse()
    value = torch.randn(1, 4, 2)
    value[:, 2].mul_(4)
    world, supports = _inputs(value)
    route = _route(value)

    with pytest.raises(ValueError, match="does not consume compute_factors"):
        pulse(
            world,
            supports,
            formula_route=route,
            compute_factors=torch.ones(1, 3, 1),
        )
    with pytest.raises(ValueError, match="does not consume visibility"):
        pulse(
            world,
            supports,
            formula_route=route,
            visibility=torch.ones(1, 3, 3, dtype=torch.bool),
        )


def test_formula_fabric_compute_rejects_arena_before_allocation(monkeypatch) -> None:
    program = mechanisms.FormulaFabricProgram(
        arena_capacity=4,
        feature_dim=2,
        steps=((mechanisms.FormulaInvocation(mechanisms.FormulaPrimitive.IDENTITY, 0),),),
    )
    limits = mechanisms.ContractLimits(max_elements=8)
    compute = mechanisms.FormulaFabricCompute(
        mechanisms.FormulaFabric(program, limits=limits), active_count=1
    )
    value = torch.ones(2, 1, 2)
    support = torch.ones(2, 1, dtype=torch.bool)
    workspace = mechanisms.ActiveWorkspace(value, support, support, support)
    weights = value.new_zeros((2, 1, 1, 1, 4))
    weights[..., 0] = 1
    enabled = torch.ones(2, 1, 1, dtype=torch.bool)
    route = mechanisms.FormulaRoutePlan(weights, enabled, enabled, enabled)
    called: list[bool] = []

    def fail_if_allocated(*args, **kwargs):
        called.append(True)
        raise AssertionError("arena allocation happened before admission")

    monkeypatch.setattr(mechanisms.FormulaArenaState, "from_tensor", fail_if_allocated)
    with pytest.raises(ValueError, match="arena exceeds allocation limits"):
        compute(workspace, formula_route=route)
    assert not called


def test_formula_route_is_rejected_by_standard_selective_compute() -> None:
    topology = mechanisms.ReversibleTopology(active_count=3)
    fold, unfold = topology.operations()
    pulse = mechanisms.AdaptivePulse(
        fold=fold,
        intervention=mechanisms.FormulaAttention(
            mechanisms.MagnitudeInterventionPolicy(), mechanisms.StableTopKIntervention(1)
        ),
        selective_compute=mechanisms.SelectiveCompute(
            mechanisms.ScaleShiftFormula(2), max_queries=1, max_sources=3
        ),
        unfold=unfold,
    )
    value = torch.randn(1, 4, 2)
    world, supports = _inputs(value)

    with pytest.raises(ValueError, match="does not consume formula_route"):
        pulse(world, supports, formula_route=_route(value))


def test_formula_fabric_pulse_component_graph_has_canonical_dependencies() -> None:
    refs = {
        node["ref"]
        for node in arti.component_provenance(_pulse())["components"]
    }
    assert refs >= {
        "arti/pulse@2",
        "arti/formula-fabric-compute@1",
        "arti/formula-fabric@1",
        "arti/fold@2",
        "arti/unfold@2",
    }


def test_formula_fabric_pulse_arti_st_round_trip(tmp_path) -> None:
    source = _pulse().eval()
    value = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [10.0, 12.0], [0.5, 0.5]]]
    )
    world, supports = _inputs(value)
    expected = source(world, supports, formula_route=_route(value)).value

    saved = arti.save(source, tmp_path / "formula-pulse.arti.st")
    target = _pulse().eval()
    loaded = arti.load(saved.weights_path, model=target)
    actual = target(world, supports, formula_route=_route(value)).value
    refs = {
        node["ref"]
        for node in loaded.manifest["architecture"]["component_graph"]["nodes"]
    }

    torch.testing.assert_close(actual, expected)
    assert refs >= {
        "arti/pulse@2",
        "arti/formula-fabric-compute@1",
        "arti/formula-fabric@1",
    }

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_formula_fabric_pulse_cuda_fullgraph_forward_backward(dtype: torch.dtype) -> None:
    pulse = _pulse().cuda().to(dtype).train()
    compiled = torch.compile(pulse, backend="inductor", fullgraph=True)
    value = torch.tensor(
        [[[1.0, 2.0], [3.0, 4.0], [10.0, 12.0], [0.5, 0.5]]],
        device="cuda",
        dtype=dtype,
    ).expand(2, -1, -1).clone().requires_grad_(True)
    world, supports = _inputs(value)

    actual = compiled(world, supports, formula_route=_route(value)).value
    actual.float().square().mean().backward()

    assert actual.is_cuda and actual.dtype is dtype
    assert value.grad is not None and torch.isfinite(value.grad).all()
