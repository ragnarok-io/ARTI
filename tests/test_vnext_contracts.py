from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

import arti
from arti.mechanisms import (
    EnvelopeRef,
    InterventionOperator,
    OffSemantics,
    DEFAULT_CONTRACT_LIMITS,
    OperandContract,
    OperandKind,
    OperandOwnership,
    PulseStageGraph,
    PulseExecutor,
    PulseStageSpec,
    PulseSupports,
    StageMode,
    StageRole,
    SupportDomain,
    SupportKind,
    SupportMask,
    TypedOperands,
    TensorEnvelope,
    TopologyBinding,
    apply_intervention,
    assert_unsupported_identity,
    fold_pulse_supports,
    unfold_pulse_supports,
)
from arti.component_registry import (
    COMPONENT_PROVENANCE_VERSION,
    ComponentCompatibilityError,
    UnknownComponentError,
    component_spec,
    get_component_registry,
    validate_component_provenance,
)


def domain(
    mask: torch.Tensor,
    *,
    domain_id: str = "pulse-world",
    owner_ref: str = "arti/fold@2",
    partition_id: str = "world",
    transition_id: str = "transition-one",
) -> SupportDomain:
    return SupportDomain.for_tensor(
        mask,
        domain_id=domain_id,
        owner_ref=owner_ref,
        partition_id=partition_id,
        transition_id=transition_id,
    )


def support(
    kind: SupportKind,
    values: list[list[bool]],
    *,
    support_domain: SupportDomain | None = None,
) -> SupportMask:
    mask = torch.tensor(values, dtype=torch.bool)
    return SupportMask(kind, mask, support_domain or domain(mask))


_OFF = {
    StageRole.OBSERVATION: OffSemantics.IDENTITY_OBSERVATION,
    StageRole.HALF: OffSemantics.IDENTITY_VALUES,
    StageRole.FOLD: OffSemantics.ALL_OBSERVED_EXPOSED,
    StageRole.INTERVENTION: OffSemantics.PRESERVE_INTERVENTION_SUPPORT,
    StageRole.SELECTIVE_COMPUTE: OffSemantics.NO_COMPUTE,
    StageRole.UNFOLD: OffSemantics.REUNION_BYPASS,
    StageRole.AGGREGATE: OffSemantics.NO_AGGREGATION,
    StageRole.BANK_UPDATE: OffSemantics.NO_BANK_UPDATE,
}
_BINDING = TopologyBinding(
    topology_ref="arti/reversible-topology@1",
    topology_config_fingerprint="c" * 64,
    producer_provenance_fingerprint="e" * 64,
)


def stage(
    role: StageRole,
    input_schema: EnvelopeRef,
    output_schema: EnvelopeRef,
    *,
    enabled: bool,
    component_ref: str | None = None,
    pair_id: str | None = None,
    config: dict[str, object] | None = None,
    stage_id: str | None = None,
    topology_binding: TopologyBinding | None = None,
) -> PulseStageSpec:
    return PulseStageSpec(
        stage_id=stage_id or role.value.replace("_", "-"),
        role=role,
        mode=StageMode.ENABLED if enabled else StageMode.OFF,
        component_ref=component_ref if enabled else None,
        input_schema=input_schema,
        output_schema=output_schema,
        config=config,
        pair_id=pair_id,
        topology_binding=(
            (topology_binding or _BINDING)
            if enabled and role in {StageRole.FOLD, StageRole.UNFOLD}
            else None
        ),
        off_semantics=None if enabled else _OFF[role],
    )


def minimal_graph(
    *,
    topology_binding: TopologyBinding | None = None,
    stage_configs: dict[str, dict[str, object]] | None = None,
) -> PulseStageGraph:
    configs = stage_configs or {}
    return PulseStageGraph(
        (
            stage(
                StageRole.OBSERVATION,
                EnvelopeRef.WORLD,
                EnvelopeRef.WORLD,
                enabled=False,
            ),
            stage(
                StageRole.HALF,
                EnvelopeRef.WORLD,
                EnvelopeRef.WORLD,
                enabled=True,
                component_ref="arti/half@1",
                config=configs.get("half"),
            ),
            stage(
                StageRole.FOLD,
                EnvelopeRef.WORLD,
                EnvelopeRef.FOLDED,
                enabled=True,
                component_ref="arti/fold@2",
                pair_id="world-fold",
                config=configs.get("fold"),
                topology_binding=topology_binding,
            ),
            stage(
                StageRole.INTERVENTION,
                EnvelopeRef.FOLDED,
                EnvelopeRef.FOLDED,
                enabled=False,
            ),
            stage(
                StageRole.SELECTIVE_COMPUTE,
                EnvelopeRef.FOLDED,
                EnvelopeRef.FOLDED,
                enabled=False,
            ),
            stage(
                StageRole.UNFOLD,
                EnvelopeRef.FOLDED,
                EnvelopeRef.REUNITED,
                enabled=True,
                component_ref="arti/unfold@2",
                pair_id="world-fold",
                config=configs.get("unfold"),
                topology_binding=topology_binding,
            ),
            stage(
                StageRole.AGGREGATE,
                EnvelopeRef.REUNITED,
                EnvelopeRef.REUNITED,
                enabled=False,
            ),
        )
    )


def test_support_lattice_binds_domains_and_write_transition() -> None:
    world_mask = torch.tensor([[True, True, True, False]])
    world = domain(world_mask)
    write_mask = torch.tensor([[True, False]])
    write_domain = domain(
        write_mask,
        domain_id="bank-write",
        owner_ref="arti/target-bank-updater@2",
        partition_id="target-bank",
    )
    write_contract = OperandContract(
        kind=OperandKind.WRITE,
        source_ref="arti/target-bank-updater@2",
        partition_id="target-bank",
        consumer_ref="arti/target-bank-updater@2",
        factor_dim=4,
        layout="dense",
        domain=write_domain,
        source_asset_fingerprint="f" * 64,
    )
    supports = PulseSupports(
        support(SupportKind.OBSERVED, [[True, True, True, False]], support_domain=world),
        support(SupportKind.EXPOSED, [[True, False, True, False]], support_domain=world),
        support(SupportKind.INTERVENED, [[False, False, True, False]], support_domain=world),
        support(SupportKind.WRITE, [[True, False]], support_domain=write_domain),
        write_contract,
        validity=world_mask,
    )
    assert supports.observed.mask.sum().item() == 3
    assert supports.write is not None
    assert supports.write.domain.partition_id == "target-bank"

    other_transition = domain(
        write_mask,
        domain_id="bank-write",
        owner_ref="arti/target-bank-updater@2",
        partition_id="target-bank",
        transition_id="transition-two",
    )
    with pytest.raises(ValueError, match="same transition"):
        PulseSupports(
            supports.observed,
            supports.exposed,
            supports.intervened,
            support(SupportKind.WRITE, [[True, False]], support_domain=other_transition),
            write_contract,
            validity=world_mask,
        )


def test_support_snapshot_is_immutable_and_domains_cannot_drift() -> None:
    original = torch.tensor([[True, False]])
    world = domain(original)
    snapshot = SupportMask(SupportKind.OBSERVED, original, world)
    original.zero_()
    returned = snapshot.mask
    returned.zero_()
    assert snapshot.mask.tolist() == [[True, False]]
    with pytest.raises(AttributeError, match="immutable"):
        snapshot._mask = torch.ones_like(original)

    alien = domain(torch.ones(1, 2, dtype=torch.bool), domain_id="alien-world")
    with pytest.raises(ValueError, match="share domain"):
        PulseSupports(
            snapshot,
            support(SupportKind.EXPOSED, [[True, False]], support_domain=alien),
            support(SupportKind.INTERVENED, [[False, False]], support_domain=world),
            validity=torch.tensor([[True, False]]),
        )


@pytest.mark.parametrize(
    ("exposed", "intervened", "message"),
    [
        ([[True, True]], [[False, False]], "exposed support"),
        ([[True, False]], [[False, True]], "intervened support"),
    ],
)
def test_support_lattice_fails_closed_on_illegal_subset(
    exposed: list[list[bool]],
    intervened: list[list[bool]],
    message: str,
) -> None:
    world = domain(torch.tensor([[True, False]]))
    with pytest.raises(ValueError, match=message):
        PulseSupports(
            support(SupportKind.OBSERVED, [[True, False]], support_domain=world),
            support(SupportKind.EXPOSED, exposed, support_domain=world),
            support(SupportKind.INTERVENED, intervened, support_domain=world),
            validity=torch.tensor([[True, False]]),
        )


def test_identity_support_exposes_observed_but_intervenes_nowhere() -> None:
    observed = torch.tensor([[True, False, True]])
    supports = PulseSupports.identity(observed, domain(observed))
    assert torch.equal(supports.observed.mask, observed)
    assert torch.equal(supports.exposed.mask, observed)
    assert not supports.intervened.mask.any()


def test_fold_support_transport_uses_the_exact_topology_permutation() -> None:
    validity = torch.tensor([[True, True, True, False]])
    world = domain(validity)
    supports = PulseSupports(
        support(SupportKind.OBSERVED, [[True, True, True, False]], support_domain=world),
        support(SupportKind.EXPOSED, [[True, False, True, False]], support_domain=world),
        support(SupportKind.INTERVENED, [[False, False, True, False]], support_domain=world),
        validity=validity,
    )
    topology = arti.mechanisms.ReversibleTopology(
        active_count=2,
        policy=arti.mechanisms.FixedTopologyPolicy(order=[2, 0, 3, 1]),
    )
    record = topology.fold(torch.randn(1, 4, 3), validity).record

    transported = fold_pulse_supports(supports, record)
    packed_validity = torch.gather(validity, -1, record.permutation)

    assert transported.record is record
    assert torch.equal(transported.active.validity, packed_validity[..., :2])
    assert torch.equal(transported.preserved.validity, packed_validity[..., 2:])
    assert torch.equal(
        transported.active.intervened.mask,
        torch.gather(supports.intervened.mask, -1, record.permutation)[..., :2],
    )
    assert transported.active.observed.domain.layout == "packed"
    assert transported.active.observed.domain.shape == (1, 2)

    restored = unfold_pulse_supports(transported)
    assert restored.observed.domain == world
    assert torch.equal(restored.validity, validity)
    assert torch.equal(restored.observed.mask, supports.observed.mask)
    assert torch.equal(restored.exposed.mask, supports.exposed.mask)
    assert torch.equal(restored.intervened.mask, supports.intervened.mask)


def test_fold_support_transport_rejects_mask_lineage_drift() -> None:
    validity = torch.tensor([[True, True, False]])
    world = domain(validity)
    supports = PulseSupports.identity(validity, world)
    topology = arti.mechanisms.ReversibleTopology(
        active_count=2,
        policy=arti.mechanisms.FixedTopologyPolicy(order=[0, 1, 2]),
    )
    record = topology.fold(torch.randn(1, 3, 2), torch.ones_like(validity)).record
    with pytest.raises(ValueError, match="mask lineage"):
        fold_pulse_supports(supports, record)


def test_intervention_preserves_values_and_gradient_boundary() -> None:
    base = torch.randn(1, 3, 2, requires_grad=True)
    candidate = torch.randn(1, 3, 2, requires_grad=True)
    mask = torch.tensor([[False, True, False]])
    world = domain(mask)
    intervention = SupportMask(SupportKind.INTERVENED, mask, world)
    supports = PulseSupports(
        SupportMask(SupportKind.OBSERVED, torch.ones_like(mask), world),
        SupportMask(SupportKind.EXPOSED, torch.ones_like(mask), world),
        intervention,
        validity=torch.ones_like(mask),
    )
    validity = torch.ones_like(mask)
    base_envelope = TensorEnvelope(EnvelopeRef.FOLDED, base, validity, world)
    candidate_envelope = TensorEnvelope(EnvelopeRef.FOLDED, candidate, validity, world)
    result = apply_intervention(base_envelope, candidate_envelope, supports)
    assert torch.equal(result.value[:, [0, 2]], base[:, [0, 2]])
    cotangent = torch.randn_like(result.value)
    (result.value * cotangent).sum().backward()
    expected_base = torch.where(mask.unsqueeze(-1), 0.0, cotangent)
    expected_candidate = torch.where(mask.unsqueeze(-1), cotangent, 0.0)
    torch.testing.assert_close(base.grad, expected_base, rtol=0, atol=0)
    torch.testing.assert_close(candidate.grad, expected_candidate, rtol=0, atol=0)

    invalid = result.value.detach().clone()
    invalid[:, 0, 0] += 1.0
    invalid_envelope = TensorEnvelope(EnvelopeRef.FOLDED, invalid, validity, world)
    with pytest.raises(ValueError, match="exactly unchanged"):
        assert_unsupported_identity(base_envelope, invalid_envelope, supports)

    alien = domain(mask, domain_id="alien-folded")
    alien_candidate = TensorEnvelope(EnvelopeRef.FOLDED, candidate, validity, alien)
    with pytest.raises(ValueError, match="envelope identity and domain"):
        apply_intervention(base_envelope, alien_candidate, supports)


def test_intervention_hot_path_captures_as_one_full_graph() -> None:
    mask = torch.tensor([[False, True, False]])
    world = domain(mask)
    supports = PulseSupports(
        SupportMask(SupportKind.OBSERVED, torch.ones_like(mask), world),
        SupportMask(SupportKind.EXPOSED, torch.ones_like(mask), world),
        SupportMask(SupportKind.INTERVENED, mask, world),
        validity=torch.ones_like(mask),
    )

    operator = InterventionOperator(supports)
    compiled = torch.compile(operator, backend="eager", fullgraph=True)
    base = torch.randn(1, 3, 4)
    candidate = torch.randn_like(base)
    actual = compiled(base, candidate)
    torch.testing.assert_close(actual, operator(base, candidate), rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_intervention_operator_rebinds_domain_after_cuda_placement() -> None:
    mask = torch.tensor([[False, True, False]])
    cpu_domain = domain(mask)
    supports = PulseSupports(
        SupportMask(SupportKind.OBSERVED, torch.ones_like(mask), cpu_domain),
        SupportMask(SupportKind.EXPOSED, torch.ones_like(mask), cpu_domain),
        SupportMask(SupportKind.INTERVENED, mask, cpu_domain),
        validity=torch.ones_like(mask),
    )
    operator = InterventionOperator(supports).cuda()
    assert operator.domain.device == "cuda:0"
    compiled = torch.compile(operator, fullgraph=True)
    base = torch.randn(1, 3, 4, device="cuda")
    candidate = torch.randn_like(base)
    actual = compiled(base, candidate)
    torch.cuda.synchronize()
    torch.testing.assert_close(actual, operator(base, candidate), rtol=0, atol=0)


def test_intervention_operation_budget_cannot_be_relaxed() -> None:
    mask = torch.tensor([[False, True, False]])
    world = domain(mask)
    supports = PulseSupports(
        SupportMask(SupportKind.OBSERVED, torch.ones_like(mask), world),
        SupportMask(SupportKind.EXPOSED, torch.ones_like(mask), world),
        SupportMask(SupportKind.INTERVENED, mask, world),
        validity=torch.ones_like(mask),
    )
    values = torch.randn(1, 3, 4)
    envelope = TensorEnvelope(EnvelopeRef.FOLDED, values, torch.ones_like(mask), world)
    too_small = replace(DEFAULT_CONTRACT_LIMITS, max_operation_bytes=1)
    with pytest.raises(ValueError, match="max_operation_bytes"):
        apply_intervention(envelope, envelope, supports, limits=too_small)
    relaxed = replace(
        DEFAULT_CONTRACT_LIMITS,
        max_operation_bytes=DEFAULT_CONTRACT_LIMITS.max_operation_bytes + 1,
    )
    with pytest.raises(ValueError, match="cannot relax"):
        apply_intervention(envelope, envelope, supports, limits=relaxed)


def operand_contract(mask: torch.Tensor, *, fingerprint: str = "a" * 64) -> OperandContract:
    return OperandContract(
        kind=OperandKind.TOPOLOGY,
        source_ref="arti/topology-operand-bank@1",
        partition_id="topology-primary",
        consumer_ref="arti/topology-priority-formula@1",
        factor_dim=4,
        layout="dense",
        domain=domain(
            mask,
            domain_id="topology-operands",
            owner_ref="arti/topology-operand-bank@1",
            partition_id="topology-primary",
        ),
        source_asset_fingerprint=fingerprint,
    )


def test_typed_operands_bind_producer_consumer_asset_and_snapshot() -> None:
    mask = torch.ones(2, 3, dtype=torch.bool)
    values = torch.randn(2, 3, 4)
    operands = TypedOperands(
        operand_contract(mask),
        values,
        mask,
        ownership=OperandOwnership.OWNED_INFERENCE,
    )
    values.zero_()
    mask.zero_()
    assert operands.values.abs().sum().item() > 0
    assert operands.mask.all()
    returned = operands.snapshot_values()
    returned.zero_()
    assert operands.values.abs().sum().item() > 0
    public_values = operands.values
    public_values.zero_()
    assert operands.values.abs().sum().item() > 0
    assert operands.contract.consumer_ref == "arti/topology-priority-formula@1"


def test_training_operands_are_zero_copy_and_differentiable() -> None:
    mask = torch.ones(1, 2, dtype=torch.bool)
    values = torch.randn(1, 2, 4, requires_grad=True)
    operands = TypedOperands(operand_contract(mask), values, mask)
    source = arti.mechanisms.TopologyOperandBank(slots=2, key_dim=2, factor_dim=4)
    consumer = arti.mechanisms.TopologyPriorityFormula(factor_dim=4)
    consumed = operands.consume(
        consumer=consumer,
        kind=OperandKind.TOPOLOGY,
        source=source,
        partition_id="topology-primary",
        domain=operands.contract.domain,
        factor_dim=4,
        layout="dense",
        source_asset_fingerprint="a" * 64,
    )
    assert consumed.data_ptr() == values.data_ptr()
    consumed.square().sum().backward()
    torch.testing.assert_close(values.grad, 2 * values.detach())


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"consumer": arti.mechanisms.UnFold(active_count=1)}, "consumer"),
        ({"kind": OperandKind.RECALL}, "kind"),
        ({"source": arti.mechanisms.TopologyPriorityFormula(factor_dim=4)}, "source"),
        ({"partition_id": "other-partition"}, "partition"),
        (
            {"domain": domain(torch.ones(1, 2, dtype=torch.bool), domain_id="other")},
            "domain",
        ),
        ({"factor_dim": 3}, "factor_dim"),
        ({"layout": "packed"}, "layout"),
        ({"source_asset_fingerprint": "b" * 64}, "source asset"),
    ],
)
def test_typed_operands_fail_closed_for_wrong_consumer_authority(
    override: dict[str, object],
    message: str,
) -> None:
    mask = torch.ones(1, 2, dtype=torch.bool)
    operands = TypedOperands(operand_contract(mask), torch.randn(1, 2, 4), mask)
    request: dict[str, object] = {
        "consumer": arti.mechanisms.TopologyPriorityFormula(factor_dim=4),
        "kind": OperandKind.TOPOLOGY,
        "source": arti.mechanisms.TopologyOperandBank(slots=2, key_dim=2, factor_dim=4),
        "partition_id": "topology-primary",
        "domain": operands.contract.domain,
        "factor_dim": 4,
        "layout": "dense",
        "source_asset_fingerprint": "a" * 64,
    }
    request.update(override)
    with pytest.raises(ValueError, match=message):
        operands.consume(**request)  # type: ignore[arg-type]


def test_typed_operands_reject_shape_and_asset_contract_drift() -> None:
    mask = torch.ones(1, 2, dtype=torch.bool)
    with pytest.raises(ValueError, match="SHA-256"):
        operand_contract(mask, fingerprint="not-a-hash")
    with pytest.raises(ValueError, match="factor_dim"):
        TypedOperands(operand_contract(mask), torch.randn(1, 2, 3), mask)


def test_stage_graph_is_complete_composable_and_round_trips_strictly() -> None:
    graph = minimal_graph()
    payload = graph.to_dict()
    assert PulseStageGraph.from_dict(payload).to_dict() == payload
    assert graph.enabled_dependencies == (
        "arti/half@1",
        "arti/fold@2",
        "arti/unfold@2",
    )

    tampered = deepcopy(payload)
    tampered["stages"][1]["config"]["nested"] = True
    with pytest.raises(ValueError, match="fingerprint"):
        PulseStageGraph.from_dict(tampered)
    unknown = deepcopy(payload)
    unknown["unexpected"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        PulseStageGraph.from_dict(unknown)
    boolean_version = deepcopy(payload)
    boolean_version["schema_version"] = True
    with pytest.raises(ValueError, match="unsupported"):
        PulseStageGraph.from_dict(boolean_version)


def test_stage_config_rejects_non_json_keys_before_normalization() -> None:
    with pytest.raises(TypeError, match="keys must be strings"):
        stage(
            StageRole.HALF,
            EnvelopeRef.WORLD,
            EnvelopeRef.WORLD,
            enabled=True,
            component_ref="arti/half@1",
            config={1: "not-json"},  # type: ignore[dict-item]
        )


def test_stage_graph_allows_repeated_roles_with_unique_stage_ids() -> None:
    stages = list(minimal_graph().stages)
    stages.insert(
        2,
        stage(
            StageRole.HALF,
            EnvelopeRef.WORLD,
            EnvelopeRef.WORLD,
            enabled=True,
            component_ref="arti/half@1",
            stage_id="half-second",
        ),
    )
    graph = PulseStageGraph(tuple(stages))
    assert [item.role for item in graph.stages].count(StageRole.HALF) == 2


def test_fully_disabled_graph_is_a_typed_identity_path() -> None:
    roles = (
        StageRole.OBSERVATION,
        StageRole.HALF,
        StageRole.FOLD,
        StageRole.INTERVENTION,
        StageRole.SELECTIVE_COMPUTE,
        StageRole.UNFOLD,
        StageRole.AGGREGATE,
    )
    graph = PulseStageGraph(
        tuple(
            stage(role, EnvelopeRef.WORLD, EnvelopeRef.WORLD, enabled=False)
            for role in roles
        )
    )
    assert graph.enabled_dependencies == ()


def test_stage_config_is_immutable_and_reserved_keys_fail_closed() -> None:
    config = {"policy": {"thresholds": [0.25, 0.5]}}
    spec = stage(
        StageRole.HALF,
        EnvelopeRef.OBSERVATION,
        EnvelopeRef.OBSERVATION,
        enabled=True,
        component_ref="arti/half@1",
        config=config,
    )
    fingerprint = spec.config_fingerprint
    config["policy"]["thresholds"].append(1.0)  # type: ignore[index, union-attr]
    returned = spec.config
    returned["policy"]["thresholds"].append(2.0)
    assert spec.config_fingerprint == fingerprint
    assert spec.config == {"policy": {"thresholds": [0.25, 0.5]}}
    with pytest.raises(ValueError, match="reserved"):
        stage(
            StageRole.HALF,
            EnvelopeRef.OBSERVATION,
            EnvelopeRef.OBSERVATION,
            enabled=True,
            component_ref="arti/half@1",
            config={"enabled": False},
        )


def test_stage_role_off_and_graph_structure_cannot_be_spoofed() -> None:
    with pytest.raises(ValueError, match="authorized"):
        stage(
            StageRole.HALF,
            EnvelopeRef.OBSERVATION,
            EnvelopeRef.OBSERVATION,
            enabled=True,
            component_ref="arti/unfold@2",
        )
    with pytest.raises(UnknownComponentError):
        stage(
            StageRole.HALF,
            EnvelopeRef.OBSERVATION,
            EnvelopeRef.OBSERVATION,
            enabled=True,
            component_ref="third-party/not-registered@1",
        )
    with pytest.raises(ValueError, match="must not declare component_ref"):
        PulseStageSpec(
            stage_id="half",
            role=StageRole.HALF,
            mode=StageMode.OFF,
            component_ref="arti/half@1",
            input_schema=EnvelopeRef.OBSERVATION,
            output_schema=EnvelopeRef.OBSERVATION,
            off_semantics=OffSemantics.IDENTITY_VALUES,
        )
    with pytest.raises(ValueError, match="requires observation"):
        PulseStageGraph(
            (
                stage(
                    StageRole.AGGREGATE,
                    EnvelopeRef.REUNITED,
                    EnvelopeRef.REUNITED,
                    enabled=False,
                ),
            )
        )

    with pytest.raises(ValueError, match="transition does not match"):
        stage(
            StageRole.HALF,
            EnvelopeRef.WORLD,
            EnvelopeRef.OBSERVATION,
            enabled=True,
            component_ref="arti/half@1",
        )


def test_fold_unfold_pair_requires_identical_topology_binding() -> None:
    stages = list(minimal_graph().stages)
    stages[5] = PulseStageSpec(
        stage_id="unfold",
        role=StageRole.UNFOLD,
        mode=StageMode.ENABLED,
        component_ref="arti/unfold@2",
        input_schema=EnvelopeRef.FOLDED,
        output_schema=EnvelopeRef.REUNITED,
        pair_id="world-fold",
        topology_binding=TopologyBinding(
            topology_ref="arti/reversible-topology@1",
            topology_config_fingerprint="d" * 64,
            producer_provenance_fingerprint="e" * 64,
        ),
    )
    with pytest.raises(ValueError, match="topology binding"):
        PulseStageGraph(tuple(stages))


def test_topology_binding_validates_the_real_fold_record() -> None:
    topology = arti.mechanisms.ReversibleTopology(
        active_count=2,
        policy=arti.mechanisms.FixedTopologyPolicy(order=[2, 0, 3, 1]),
    )
    record = topology.fold(torch.randn(1, 4, 3)).record
    binding = TopologyBinding(
        topology_ref="arti/reversible-topology@1",
        topology_config_fingerprint=topology.contract_fingerprint,
        producer_provenance_fingerprint=topology.producer_provenance_fingerprint,
    )
    binding.validate_record(record)

    wrong = TopologyBinding(
        topology_ref="arti/reversible-topology@1",
        topology_config_fingerprint=topology.contract_fingerprint,
        producer_provenance_fingerprint="0" * 64,
    )
    with pytest.raises(ValueError, match="producer_provenance_fingerprint"):
        wrong.validate_record(record)


def test_pulse_executor_binds_and_runs_real_half_fold_unfold() -> None:
    topology = arti.mechanisms.ReversibleTopology(
        active_count=2,
        policy=arti.mechanisms.FixedTopologyPolicy(order=[2, 0, 3, 1]),
    )
    fold, unfold = topology.operations()
    binding = TopologyBinding(
        topology_ref="arti/reversible-topology@1",
        topology_config_fingerprint=topology.contract_fingerprint,
        producer_provenance_fingerprint=topology.producer_provenance_fingerprint,
    )
    half = arti.nn.Half(stochastic=False)
    modules = {"half": half, "fold": fold, "unfold": unfold}
    configs = {stage_id: dict(component_spec(module).config) for stage_id, module in modules.items()}
    executor = PulseExecutor(
        minimal_graph(topology_binding=binding, stage_configs=configs),
        modules,
    )
    value = torch.randn(2, 4, 3, requires_grad=True)
    mask = torch.tensor([[True, True, True, False], [True, True, True, True]])
    intervened = torch.tensor(
        [[True, False, False, False], [False, True, False, True]]
    )
    world = domain(
        mask,
        owner_ref="arti/pulse-executor@1",
        partition_id="world",
    )
    supports = PulseSupports(
        SupportMask(SupportKind.OBSERVED, mask, world),
        SupportMask(SupportKind.EXPOSED, mask, world),
        SupportMask(SupportKind.INTERVENED, intervened, world),
        validity=mask,
    )
    result = executor(TensorEnvelope(EnvelopeRef.WORLD, value, mask, world), supports)

    assert result.ref is EnvelopeRef.REUNITED
    expected = torch.where(intervened.unsqueeze(-1), half(value), value)
    torch.testing.assert_close(result.value, expected, rtol=0, atol=0)
    assert torch.equal(result.value[~intervened], value[~intervened])
    cotangent = torch.randn_like(result.value)
    (result.value * cotangent).sum().backward()
    torch.testing.assert_close(value.grad[~intervened], cotangent[~intervened], rtol=0, atol=0)
    assert torch.equal(result.mask, mask)
    assert result.domain == world
    assert arti.component_ref(executor) == "arti/pulse-executor@1"
    executor_spec = component_spec(executor)
    assert executor_spec.config["manifest"] == executor.manifest.to_dict()
    assert set(executor_spec.dependencies) == {
        "arti/pulse-stage-graph@1",
        "arti/half@1",
        "arti/fold@2",
        "arti/unfold@2",
    }
    provenance = {
        "schema_version": COMPONENT_PROVENANCE_VERSION,
        "components": [executor_spec.to_dict()],
    }
    provenance["fingerprint"] = arti.component_graph_fingerprint(provenance["components"])
    assert validate_component_provenance(provenance) == provenance
    missing_dependency = deepcopy(provenance)
    missing_dependency["components"][0]["dependencies"].remove("arti/fold@2")
    missing_dependency["fingerprint"] = arti.component_graph_fingerprint(
        missing_dependency["components"]
    )
    with pytest.raises(ComponentCompatibilityError, match="dependency closure"):
        validate_component_provenance(missing_dependency)


def test_contextual_half_cannot_read_outside_exposed_support() -> None:
    topology = arti.mechanisms.ReversibleTopology(
        active_count=2,
        policy=arti.mechanisms.FixedTopologyPolicy(order=[0, 1, 2]),
    )
    fold, unfold = topology.operations()
    binding = TopologyBinding(
        topology_ref="arti/reversible-topology@1",
        topology_config_fingerprint=topology.contract_fingerprint,
        producer_provenance_fingerprint=topology.producer_provenance_fingerprint,
    )
    half = arti.nn.Half(
        stochastic=False,
        context_mode="contextual",
        context_axes=-2,
        context_gain=1.0,
    )
    modules = {"half": half, "fold": fold, "unfold": unfold}
    configs = {stage_id: dict(component_spec(module).config) for stage_id, module in modules.items()}
    stages = list(minimal_graph(topology_binding=binding, stage_configs=configs).stages)
    stages[1] = stage(
        StageRole.HALF,
        EnvelopeRef.WORLD,
        EnvelopeRef.WORLD,
        enabled=True,
        component_ref="arti/half@2",
        config=configs["half"],
    )
    executor = PulseExecutor(PulseStageGraph(tuple(stages)), modules)
    validity = torch.ones(1, 3, dtype=torch.bool)
    world = domain(
        validity,
        owner_ref="arti/pulse-executor@1",
        partition_id="world",
    )
    supports = PulseSupports(
        SupportMask(SupportKind.OBSERVED, validity, world),
        SupportMask(SupportKind.EXPOSED, torch.tensor([[True, False, False]]), world),
        SupportMask(SupportKind.INTERVENED, torch.tensor([[True, False, False]]), world),
        validity=validity,
    )
    first = torch.tensor([[[0.25], [1.0], [2.0]]])
    second = torch.tensor([[[0.25], [1000.0], [-1000.0]]])

    first_result = executor(TensorEnvelope(EnvelopeRef.WORLD, first, validity, world), supports)
    second_result = executor(TensorEnvelope(EnvelopeRef.WORLD, second, validity, world), supports)

    torch.testing.assert_close(first_result.value[:, :1], second_result.value[:, :1])
    assert torch.equal(first_result.value[:, 1:], first[:, 1:])
    assert torch.equal(second_result.value[:, 1:], second[:, 1:])


def test_pulse_executor_rejects_missing_or_wrong_stage_modules() -> None:
    graph = minimal_graph()
    with pytest.raises(ValueError, match="enabled stage IDs exactly"):
        PulseExecutor(graph, {})
    with pytest.raises(ValueError, match="identity does not match"):
        PulseExecutor(
            graph,
            {
                "half": arti.mechanisms.Fold(active_count=1),
                "fold": arti.mechanisms.Fold(active_count=1),
                "unfold": arti.mechanisms.UnFold(active_count=1),
            },
        )


def test_pulse_executor_rejects_module_config_drift() -> None:
    topology = arti.mechanisms.ReversibleTopology(
        active_count=1,
        policy=arti.mechanisms.FixedTopologyPolicy(order=[0, 1]),
    )
    fold, unfold = topology.operations()
    binding = TopologyBinding(
        topology_ref="arti/reversible-topology@1",
        topology_config_fingerprint=topology.contract_fingerprint,
        producer_provenance_fingerprint=topology.producer_provenance_fingerprint,
    )
    with pytest.raises(ValueError, match="config does not match"):
        PulseExecutor(
            minimal_graph(topology_binding=binding),
            {
                "half": arti.nn.Half(stochastic=False),
                "fold": fold,
                "unfold": unfold,
            },
        )


def test_only_persistable_stage_contracts_are_registered_components() -> None:
    graph = minimal_graph()
    assert arti.component_ref(graph) == "arti/pulse-stage-graph@1"
    assert arti.component_ref(graph.stages[0]) == "arti/pulse-stage@1"
    with pytest.raises(UnknownComponentError):
        arti.component_ref(support(SupportKind.OBSERVED, [[True]]))


def test_stage_authority_comes_from_component_capabilities() -> None:
    registry = get_component_registry()
    half = registry.registration_for_reference("arti/half@1")
    unfold = registry.registration_for_reference("arti/unfold@2")
    assert half.capabilities == ("pulse.stage.half",)
    assert unfold.capabilities == ("pulse.stage.unfold",)


def test_stage_graph_component_spec_records_only_enabled_dependencies() -> None:
    spec = component_spec(minimal_graph())
    assert set(spec.dependencies) == {
        "arti/pulse-stage@1",
        "arti/half@1",
        "arti/fold@2",
        "arti/unfold@2",
    }
    payload = {"schema_version": COMPONENT_PROVENANCE_VERSION, "components": [spec.to_dict()]}
    payload["fingerprint"] = arti.component_graph_fingerprint(payload["components"])
    assert validate_component_provenance(payload) == payload

    unknown_top_level = deepcopy(payload)
    unknown_top_level["extra"] = True
    with pytest.raises(ComponentCompatibilityError, match="top-level"):
        validate_component_provenance(unknown_top_level)

    incomplete = deepcopy(payload)
    incomplete["components"][0]["dependencies"].remove("arti/half@1")
    incomplete["fingerprint"] = arti.component_graph_fingerprint(incomplete["components"])
    with pytest.raises(ComponentCompatibilityError, match="dependency closure"):
        validate_component_provenance(incomplete)

    drifted = deepcopy(payload)
    drifted["components"][0]["capabilities"] = ["pulse.stage.aggregate"]
    drifted["fingerprint"] = arti.component_graph_fingerprint(drifted["components"])
    with pytest.raises(ValueError, match="capability drift"):
        validate_component_provenance(drifted)
