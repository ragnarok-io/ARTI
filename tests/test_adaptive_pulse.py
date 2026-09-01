from __future__ import annotations

import hashlib
import json
from types import MethodType

import pytest
import torch
from torch import nn

import arti
from arti.mechanisms import (
    AdaptiveObservation,
    AdaptivePulse,
    BankState,
    BankUpdateStatus,
    EnvelopeRef,
    FixedObservationPolicy,
    FormulaAttention,
    MagnitudeInterventionPolicy,
    OperandContract,
    OperandKind,
    PulseSupports,
    ReunionAggregate,
    ScaleShiftFormula,
    SelectiveCompute,
    SoftFoldAggregate,
    StableTopKIntervention,
    SupportDomain,
    SupportKind,
    SupportMask,
    TensorEnvelope,
)
from arti.component_registry import (
    ComponentCompatibilityError,
    component_graph_fingerprint,
    component_provenance,
    component_spec,
    validate_component_provenance,
)


def world_fixture(
    value: torch.Tensor,
    *,
    validity: torch.Tensor | None = None,
    exposed: torch.Tensor | None = None,
    intervened: torch.Tensor | None = None,
) -> tuple[TensorEnvelope, PulseSupports]:
    mask = (
        torch.ones(value.shape[:-1], dtype=torch.bool, device=value.device)
        if validity is None
        else validity
    )
    domain = SupportDomain.for_tensor(
        mask,
        domain_id="pulse-world",
        owner_ref="arti/pulse@2",
        partition_id="world",
        transition_id="pulse-transition",
    )
    world = TensorEnvelope(EnvelopeRef.WORLD, value, mask, domain)
    observed = mask
    exposed_mask = observed if exposed is None else exposed
    intervention_mask = torch.zeros_like(mask) if intervened is None else intervened
    supports = PulseSupports(
        SupportMask(SupportKind.OBSERVED, observed, domain),
        SupportMask(SupportKind.EXPOSED, exposed_mask, domain),
        SupportMask(SupportKind.INTERVENED, intervention_mask, domain),
        validity=mask,
    )
    return world, supports


def topology_pair(active_count: int) -> tuple[arti.mechanisms.Fold, arti.mechanisms.UnFold]:
    topology = arti.mechanisms.ReversibleTopology(
        active_count,
        policy=arti.mechanisms.FixedTopologyPolicy(),
    )
    return topology.operations()


def test_all_off_pulse_is_exact_object_identity() -> None:
    pulse = AdaptivePulse()
    value = torch.randn(2, 5, 3)
    world, supports = world_fixture(value)

    result = pulse(world, supports)

    assert result.envelope is world
    assert result.source_supports is supports
    assert result.value is value
    assert all(stage.mode.value == "off" for stage in pulse.manifest.stages)


def test_half_stage_uses_exposure_not_formula_intervention_support() -> None:
    value = torch.tensor([[[0.1], [2.0], [3.0], [4.0]]])
    exposed = torch.tensor([[True, True, False, False]])
    world, supports = world_fixture(value, exposed=exposed)
    half = arti.nn.Half(stochastic=False)
    pulse = AdaptivePulse(half=half)

    result = pulse(world, supports)
    expected_candidate = half(
        torch.where(exposed.unsqueeze(-1), value, torch.zeros_like(value))
    )
    expected = torch.where(exposed.unsqueeze(-1), expected_candidate, value)

    torch.testing.assert_close(result.value, expected)
    assert torch.equal(result.value[..., 2:, :], value[..., 2:, :])


def test_formula_attention_and_compute_change_only_selected_active_values() -> None:
    value = torch.tensor([[[1.0], [5.0], [3.0], [8.0], [9.0]]])
    world, supports = world_fixture(value)
    fold, unfold = topology_pair(3)
    pulse = AdaptivePulse(
        fold=fold,
        intervention=FormulaAttention(
            MagnitudeInterventionPolicy(),
            StableTopKIntervention(1),
        ),
        selective_compute=SelectiveCompute(
            ScaleShiftFormula(1),
            max_queries=1,
            max_sources=3,
        ),
        unfold=unfold,
    )
    factors = torch.zeros(1, 3, 2)
    factors[..., 1] = 10.0

    result = pulse(world, supports, compute_factors=factors)

    selected = result.source_supports.intervened.mask
    assert int(selected.sum()) == 1
    assert selected[0, 1]
    expected = torch.where(selected.unsqueeze(-1), value + 10.0, value)
    torch.testing.assert_close(result.value, expected)
    assert result.ref is EnvelopeRef.REUNITED


def test_external_intervention_is_allowed_when_attention_is_off() -> None:
    value = torch.arange(5.0).reshape(1, 5, 1)
    external = torch.tensor([[False, True, False, False, False]])
    world, supports = world_fixture(value, intervened=external)
    fold, unfold = topology_pair(3)
    pulse = AdaptivePulse(
        fold=fold,
        selective_compute=SelectiveCompute(
            ScaleShiftFormula(1),
            max_queries=1,
            max_sources=3,
        ),
        unfold=unfold,
    )
    factors = torch.zeros(1, 3, 2)
    factors[..., 1] = 4.0

    result = pulse(world, supports, compute_factors=factors)

    torch.testing.assert_close(
        result.value,
        torch.where(external.unsqueeze(-1), value + 4.0, value),
    )


def test_observation_preserves_t_until_post_reunion_aggregate() -> None:
    torch.manual_seed(19)
    value = torch.randn(2, 5, 3)
    world, supports = world_fixture(value)
    fold, unfold = topology_pair(3)
    pulse = AdaptivePulse(
        observation=AdaptiveObservation(
            FixedObservationPolicy(torch.ones(2, 1))
        ),
        fold=fold,
        unfold=unfold,
        aggregate=ReunionAggregate(SoftFoldAggregate(k=4, dim=3)),
    ).eval()

    result = pulse(world, supports)

    assert result.ref is EnvelopeRef.PULSE
    assert result.value.shape == (2, 4, 3)
    assert result.mask.shape == (2, 4)
    assert result.source_supports.observed.domain.shape == (2, 2, 5)
    assert result.domain.shape == (2, 4)


def test_aggregate_can_run_without_reversible_topology() -> None:
    value = torch.randn(2, 5, 3)
    world, supports = world_fixture(value)
    pulse = AdaptivePulse(
        aggregate=ReunionAggregate(SoftFoldAggregate(k=2, dim=3))
    )

    result = pulse(world, supports)

    assert result.ref is EnvelopeRef.PULSE
    assert result.value.shape == (2, 2, 3)


def test_pulse_rejects_incomplete_or_semantically_invalid_stage_sets() -> None:
    fold, unfold = topology_pair(2)
    attention = FormulaAttention(
        MagnitudeInterventionPolicy(), StableTopKIntervention(1)
    )

    with pytest.raises(ValueError, match="enabled together"):
        AdaptivePulse(fold=fold)
    with pytest.raises(ValueError, match="require reversible topology"):
        AdaptivePulse(selective_compute=SelectiveCompute(ScaleShiftFormula(2), max_queries=1))
    with pytest.raises(ValueError, match="requires SelectiveCompute"):
        AdaptivePulse(fold=fold, intervention=attention, unfold=unfold)


def test_pulse_component_owns_one_versioned_graph() -> None:
    fold, unfold = topology_pair(2)
    pulse = AdaptivePulse(fold=fold, unfold=unfold)
    spec = component_spec(pulse)

    assert spec.reference == "arti/pulse@2"
    assert spec.config["manifest_ref"] == "arti/pulse-stage-graph@1"
    assert spec.config["manifest_fingerprint"] == pulse.manifest.fingerprint
    assert "arti/fold@2" in spec.dependencies
    assert "arti/unfold@2" in spec.dependencies
    assert "arti/pulse@1" not in spec.dependencies


def test_pulse_rejects_component_drift_after_manifest_binding() -> None:
    fold, unfold = topology_pair(2)
    pulse = AdaptivePulse(fold=fold, unfold=unfold)
    pulse.half_stage = arti.Half(stochastic=False)
    value = torch.randn(1, 4, 3)
    world, supports = world_fixture(value)

    with pytest.raises(ValueError, match="enabled after manifest binding"):
        pulse(world, supports)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_compiled_pulse_rejects_component_drift_before_tracing() -> None:
    fold, unfold = topology_pair(2)
    pulse = AdaptivePulse(fold=fold, unfold=unfold).cuda()
    pulse.half_stage = arti.Half(stochastic=False).cuda()
    value = torch.randn(1, 4, 3, device="cuda")
    world, supports = world_fixture(value)

    compiled = torch.compile(pulse, backend="inductor", fullgraph=True)
    with pytest.raises(ValueError, match="enabled after manifest binding"):
        compiled(world, supports)


def test_pulse_provenance_recomputes_manifest_dependency_closure() -> None:
    fold, unfold = topology_pair(2)
    provenance = component_provenance(AdaptivePulse(fold=fold, unfold=unfold))
    root = next(item for item in provenance["components"] if item["path"] == "$")
    root["config"]["manifest_fingerprint"] = "0" * 64
    root["config_fingerprint"] = hashlib.sha256(
        json.dumps(
            root["config"],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")
    ).hexdigest()
    provenance["fingerprint"] = component_graph_fingerprint(provenance["components"])

    with pytest.raises(ComponentCompatibilityError, match="manifest fingerprint"):
        validate_component_provenance(provenance)


@pytest.mark.parametrize("field", ["validity", "exposed"])
def test_pulse_overlay_rejects_selective_support_mutation(field: str) -> None:
    value = torch.randn(1, 4, 3)
    world, supports = world_fixture(value)
    fold, unfold = topology_pair(2)
    intervention = FormulaAttention(
        MagnitudeInterventionPolicy(), StableTopKIntervention(1)
    )
    pulse = AdaptivePulse(
        fold=fold,
        intervention=intervention,
        selective_compute=SelectiveCompute(ScaleShiftFormula(1), max_queries=1),
        unfold=unfold,
    )

    def corrupt_support(self, workspace, _factors):
        del self
        validity = workspace.validity
        exposed = workspace.exposed
        if field == "validity":
            validity = torch.zeros_like(validity)
            exposed = torch.zeros_like(exposed)
        else:
            exposed = torch.zeros_like(exposed)
        return type(workspace)(
            workspace.value,
            validity,
            exposed,
            torch.zeros_like(workspace.intervened),
        )

    intervention.forward = MethodType(corrupt_support, intervention)
    with pytest.raises(ValueError, match=f"changed {field} support"):
        pulse(world, supports)


def test_learned_pulse_arti_st_round_trip_keeps_canonical_graph(tmp_path) -> None:
    topology = arti.mechanisms.ReversibleTopology(
        2, policy=arti.mechanisms.LearnedTopologyPolicy(dim=3)
    )
    fold, unfold = topology.operations()
    source = AdaptivePulse(fold=fold, unfold=unfold).eval()
    value = torch.randn(2, 5, 3)
    world, supports = world_fixture(value)
    expected = source(world, supports).value

    saved = arti.save(source, tmp_path / "pulse.arti.st")
    restored_topology = arti.mechanisms.ReversibleTopology(
        2, policy=arti.mechanisms.LearnedTopologyPolicy(dim=3)
    )
    restored_fold, restored_unfold = restored_topology.operations()
    target = AdaptivePulse(fold=restored_fold, unfold=restored_unfold).eval()
    loaded = arti.load(saved.weights_path, model=target)
    actual = target(world, supports).value
    refs = {
        node["ref"]
        for node in loaded.manifest["architecture"]["component_graph"]["nodes"]
    }
    fold_node = next(
        node
        for node in loaded.manifest["architecture"]["component_graph"]["nodes"]
        if node["ref"] == "arti/fold@2"
    )

    torch.testing.assert_close(actual, expected)
    assert refs >= {
        "arti/pulse@2",
        "arti/fold@2",
        "arti/unfold@2",
        "arti/topology-surrogate@1",
    }
    assert "arti/fold-record@1" in fold_node["dependencies"]
    assert "_ActiveTopologyOverlay" not in saved.manifest_path.read_text("utf-8")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_complete_pulse_cuda_fullgraph_forward_backward() -> None:
    topology = arti.mechanisms.ReversibleTopology(
        2, policy=arti.mechanisms.LearnedTopologyPolicy(dim=3)
    ).cuda()
    fold, unfold = topology.operations()
    pulse = AdaptivePulse(fold=fold, unfold=unfold).cuda().train()
    compiled = torch.compile(pulse, backend="inductor", fullgraph=True)
    value = torch.randn(2, 5, 3, device="cuda", requires_grad=True)
    world, supports = world_fixture(value)

    result = compiled(world, supports)
    result.value.sum().backward()

    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in pulse.parameters()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_full_stage_pulse_cuda_fullgraph_forward_backward() -> None:
    topology = arti.mechanisms.ReversibleTopology(
        3, policy=arti.mechanisms.LearnedTopologyPolicy(dim=3)
    ).cuda()
    fold, unfold = topology.operations()
    pulse = AdaptivePulse(
        observation=AdaptiveObservation(
            FixedObservationPolicy(torch.ones(2, 1, device="cuda"))
        ),
        fold=fold,
        intervention=FormulaAttention(
            MagnitudeInterventionPolicy(), StableTopKIntervention(1)
        ),
        selective_compute=SelectiveCompute(
            ScaleShiftFormula(3), max_queries=1, max_sources=3
        ),
        unfold=unfold,
        aggregate=ReunionAggregate(SoftFoldAggregate(k=2, dim=3)),
    ).cuda().train()
    compiled = torch.compile(pulse, backend="inductor", fullgraph=True)
    value = torch.randn(2, 5, 3, device="cuda", requires_grad=True)
    world, supports = world_fixture(value)
    factors = torch.zeros(2, 2, 3, 6, device="cuda")
    factors[..., 0] = 1.0

    result = compiled(world, supports, compute_factors=factors)
    result.value.sum().backward()

    assert result.value.shape == (2, 2, 3)
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in pulse.parameters()
    )


def with_write_authority(
    supports: PulseSupports,
    updater: nn.Module,
    write_mask: torch.Tensor,
    *,
    consumer_ref: str | None = None,
) -> PulseSupports:
    update_ref = arti.component_ref(updater)
    source_ref = "arti/recall-bank@1"
    domain = SupportDomain.for_tensor(
        write_mask,
        domain_id="pulse-bank",
        owner_ref=source_ref,
        partition_id="target-bank",
        transition_id=supports.observed.domain.transition_id,
    )
    contract = OperandContract(
        kind=OperandKind.WRITE,
        source_ref=source_ref,
        partition_id="target-bank",
        consumer_ref=update_ref if consumer_ref is None else consumer_ref,
        factor_dim=3,
        layout="dense",
        domain=domain,
        source_asset_fingerprint="a" * 64,
    )
    return PulseSupports(
        supports.observed,
        supports.exposed,
        supports.intervened,
        SupportMask(SupportKind.WRITE, write_mask, domain),
        contract,
        validity=supports.validity,
    )


def bank_pulse(updater: nn.Module) -> AdaptivePulse:
    return AdaptivePulse(
        aggregate=ReunionAggregate(SoftFoldAggregate(k=2, dim=3)),
        bank_update=updater,
    )


def test_bank_update_is_explicit_and_fixed_commit_preserves_unwritten_slots() -> None:
    value = torch.randn(1, 5, 3)
    world, supports = world_fixture(value)
    updater = arti.mechanisms.TargetBankUpdater(hidden_dim=3, slots=4)
    assert updater.shift_head is not None
    with torch.no_grad():
        updater.shift_head.weight.zero_()
        updater.shift_head.bias.fill_(1.0)
    write = torch.tensor([[True, False, True, False]])
    authorized = with_write_authority(supports, updater, write)
    bank = torch.randn(1, 4, 3)
    original = bank.clone()

    state = BankState(
        bank,
        torch.ones(1, 4, dtype=torch.bool),
        "arti/recall-bank@1",
        "target-bank",
        "a" * 64,
    )
    result = bank_pulse(updater)(world, authorized, bank_state=state)

    assert result.next_bank is not None
    assert result.bank.status is BankUpdateStatus.UPDATED
    assert torch.equal(result.next_bank.value[~write], original[~write])
    assert torch.equal(bank, original)
    assert result.next_bank.value.data_ptr() != bank.data_ptr()


def test_empty_write_support_bypasses_updater_and_returns_bank_identity() -> None:
    value = torch.randn(1, 5, 3)
    world, supports = world_fixture(value)
    updater = arti.mechanisms.TargetBankUpdater(hidden_dim=3, slots=4)
    write = torch.zeros(1, 4, dtype=torch.bool)
    authorized = with_write_authority(supports, updater, write)
    bank = torch.randn(1, 4, 3)
    called: list[bool] = []
    hook = updater.register_forward_pre_hook(lambda _module, _args: called.append(True))

    state = BankState(
        bank,
        torch.ones(1, 4, dtype=torch.bool),
        "arti/recall-bank@1",
        "target-bank",
        "a" * 64,
    )
    result = bank_pulse(updater)(world, authorized, bank_state=state)
    hook.remove()

    assert result.next_bank is state
    assert result.bank.status is BankUpdateStatus.NO_WRITE
    assert not called


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_bank_update_cuda_fullgraph_matches_eager(dtype: torch.dtype) -> None:
    torch.manual_seed(731)
    eager_updater = (
        arti.mechanisms.TargetBankUpdater(
            hidden_dim=3,
            slots=4,
            policy=arti.mechanisms.WriteRefinePolicy.fixed(2),
        )
        .cuda()
        .to(dtype)
    )
    compiled_updater = (
        arti.mechanisms.TargetBankUpdater(
            hidden_dim=3,
            slots=4,
            policy=arti.mechanisms.WriteRefinePolicy.fixed(2),
        )
        .cuda()
        .to(dtype)
    )
    compiled_updater.load_state_dict(eager_updater.state_dict())
    eager = bank_pulse(eager_updater).cuda().to(dtype).train()
    compiled_source = bank_pulse(compiled_updater).cuda().to(dtype).train()
    compiled_source.load_state_dict(eager.state_dict())
    compiled = torch.compile(compiled_source, backend="inductor", fullgraph=True)

    eager_value = torch.randn(
        2, 5, 3, device="cuda", dtype=dtype
    ).requires_grad_()
    compiled_value = eager_value.detach().clone().requires_grad_()
    eager_world, eager_supports = world_fixture(eager_value)
    compiled_world, compiled_supports = world_fixture(compiled_value)
    write = torch.ones(2, 4, dtype=torch.bool, device="cuda")
    eager_supports = with_write_authority(eager_supports, eager_updater, write)
    compiled_supports = with_write_authority(
        compiled_supports, compiled_updater, write
    )
    bank_value = torch.randn(2, 4, 3, device="cuda", dtype=dtype)
    eager_state = BankState(
        bank_value.clone(),
        torch.ones_like(write),
        "arti/recall-bank@1",
        "target-bank",
        "a" * 64,
    )
    compiled_state = BankState(
        bank_value.clone(),
        torch.ones_like(write),
        "arti/recall-bank@1",
        "target-bank",
        "a" * 64,
    )

    eager_result = eager(eager_world, eager_supports, bank_state=eager_state)
    compiled_result = compiled(
        compiled_world,
        compiled_supports,
        bank_state=compiled_state,
    )
    assert eager_result.next_bank is not None
    assert compiled_result.next_bank is not None
    torch.testing.assert_close(
        compiled_result.value, eager_result.value, rtol=2e-3, atol=2e-3
    )
    torch.testing.assert_close(
        compiled_result.next_bank.value,
        eager_result.next_bank.value,
        rtol=2e-3,
        atol=2e-3,
    )

    eager_result.next_bank.value.float().square().mean().backward()
    compiled_result.next_bank.value.float().square().mean().backward()
    torch.testing.assert_close(
        compiled_value.grad, eager_value.grad, rtol=2e-3, atol=2e-3
    )


def test_write_authority_cannot_address_invalid_bank_slots() -> None:
    value = torch.randn(1, 5, 3)
    world, supports = world_fixture(value)
    updater = arti.mechanisms.TargetBankUpdater(hidden_dim=3, slots=4)
    write = torch.tensor([[False, True, False, False]])
    authorized = with_write_authority(supports, updater, write)
    called: list[bool] = []
    hook = updater.register_forward_pre_hook(lambda _module, _args: called.append(True))

    with pytest.raises(ValueError, match="invalid Bank slots"):
        bank_pulse(updater)(
            world,
            authorized,
            bank_state=BankState(
                torch.randn(1, 4, 3),
                torch.tensor([[True, False, True, True]]),
                "arti/recall-bank@1",
                "target-bank",
                "a" * 64,
            ),
        )
    hook.remove()
    assert not called


def test_wrong_write_consumer_fails_before_updater_call() -> None:
    value = torch.randn(1, 5, 3)
    world, supports = world_fixture(value)
    updater = arti.mechanisms.TargetBankUpdater(hidden_dim=3, slots=4)
    write = torch.ones(1, 4, dtype=torch.bool)
    authorized = with_write_authority(
        supports,
        updater,
        write,
        consumer_ref="arti/target-bank-updater@2",
    )
    called: list[bool] = []
    hook = updater.register_forward_pre_hook(lambda _module, _args: called.append(True))

    with pytest.raises(ValueError, match="consumer"):
        bank_pulse(updater)(
            world,
            authorized,
            bank_state=BankState(
                torch.randn(1, 4, 3),
                torch.ones(1, 4, dtype=torch.bool),
                "arti/recall-bank@1",
                "target-bank",
                "a" * 64,
            ),
        )
    hook.remove()
    assert not called


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_pulse_cuda_forward_and_gradient_keep_selective_boundary() -> None:
    value = torch.randn(2, 6, 4, device="cuda", requires_grad=True)
    world, supports = world_fixture(value)
    fold, unfold = topology_pair(4)
    pulse = AdaptivePulse(
        fold=fold,
        intervention=FormulaAttention(
            MagnitudeInterventionPolicy(),
            StableTopKIntervention(2),
        ),
        selective_compute=SelectiveCompute(
            ScaleShiftFormula(4),
            max_queries=2,
            max_sources=4,
        ),
        unfold=unfold,
        aggregate=ReunionAggregate(SoftFoldAggregate(k=3, dim=4)),
    ).cuda()
    factors = torch.randn(2, 4, 8, device="cuda", requires_grad=True)

    result = pulse(world, supports, compute_factors=factors)
    result.value.float().square().mean().backward()

    assert result.value.is_cuda and result.value.shape == (2, 3, 4)
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert factors.grad is not None and torch.isfinite(factors.grad).all()
