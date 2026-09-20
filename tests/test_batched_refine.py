from __future__ import annotations

import hashlib
from itertools import product
import json
from pathlib import Path

import pytest
import torch

import arti
from arti import Half, Recall, RefinePolicy, alpha, component_ref
from arti.component_registry import (
    ComponentCompatibilityError,
    ComponentRegistryError,
    canonical_contract_reference,
    component_graph_fingerprint,
    validate_component_provenance,
)
from arti.recall_formula import FactorSpec, RecallFormulaContract
from arti.recall_registry import RecallFormulaId, register_formula
from arti.batched_refine import _joint_factor_topk


def _recall(*, topk: int = 3, group_size: int = 2) -> Recall:
    return Recall(
        dim=4,
        slots=12,
        formula="arti/delta@1",
        activation="none",
        routing="grouped",
        group_size=group_size,
        group_topk=topk,
        key_dim=4,
    )


def test_checked_cuda_receipt_covers_formula_topology_control() -> None:
    root = Path(__file__).resolve().parents[1]
    receipt_path = root / "docs/reference/batched-refine-cuda-profile.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    control = receipt["formula_topology_control"]

    assert receipt["schema"] == "arti.batched-refine.br6.cuda@2"
    assert receipt["logical_refine_token_work"]["wide"] == receipt[
        "logical_refine_token_work"
    ]["single_deep"]
    assert control["operation_ref"] == canonical_contract_reference(
        "arti/topology-formula-resident-operation@1"
    )
    assert control["topology_ref"] == [
        canonical_contract_reference("arti/fold@2"),
        canonical_contract_reference("arti/reversible-topology@1"),
        canonical_contract_reference("arti/unfold@2"),
    ]
    assert control["topology_unfold_verified"] is True
    assert control["deterministic_repeat_max_abs_error"] == 0.0
    assert control["formula_vs_topology_mse"] > 0.0
    for name in ("formula_control", "topology_formula_control"):
        assert receipt["physical"][name]["samples"] > 0
        assert receipt["physical"][name]["hbm_read_bytes"]["available"] is False
    script = root / "scripts/benchmark_batched_refine_cuda.py"
    assert receipt["provenance"]["script_sha256"] == hashlib.sha256(
        script.read_bytes()
    ).hexdigest()


def _copy_recall_prefix(source: Recall, target: Recall) -> None:
    with torch.no_grad():
        target.state.recall.query.weight.copy_(source.state.recall.query.weight)
        target.state.recall.group_bank[: source.state.recall.group_bank.shape[0]].copy_(
            source.state.recall.group_bank
        )
        target.state.recall.key_bank[: source.state.recall.key_bank.shape[0]].copy_(
            source.state.recall.key_bank
        )
        target.state.recall.bank[: source.state.recall.bank.shape[0]].copy_(
            source.state.recall.bank
        )


def test_batched_refine_rejects_unkeyed_stochastic_half_and_dropout() -> None:
    stochastic = Recall(
        dim=4,
        slots=12,
        activation="half",
        routing="grouped",
        group_size=2,
        group_topk=2,
        key_dim=4,
    )
    value = torch.randn(1, 3, 4)
    stochastic_candidates = alpha.query_recall_branches(stochastic, value)
    with pytest.raises(alpha.BatchedRefineContractError, match="ExecutionRNGPlan"):
        alpha.run_batched_refine(
            stochastic,
            value,
            candidates=stochastic_candidates,
        )

    dropout = _recall(topk=2)
    dropout.state.dropout.p = 0.25
    dropout.train()
    dropout_candidates = alpha.query_recall_branches(dropout, value)
    with pytest.raises(alpha.BatchedRefineContractError, match="ExecutionRNGPlan"):
        alpha.run_batched_refine(dropout, value, candidates=dropout_candidates)

    exploration = Recall(
        dim=4,
        slots=12,
        activation="none",
        routing="grouped",
        group_size=2,
        group_topk=2,
        key_dim=4,
        route_exploration=1.0,
    )
    exploration.state.recall.set_bank_gradient_enabled(True)
    exploration.train()
    with pytest.raises(alpha.BatchedRefineContractError, match="ExecutionRNGPlan"):
        alpha.query_recall_branches(exploration, torch.randn(1, 3, 4))


def test_keyed_route_exploration_replays_and_binds_candidate_query() -> None:
    recall = Recall(
        dim=4,
        slots=24,
        activation="none",
        routing="grouped",
        group_size=2,
        group_topk=3,
        key_dim=4,
        route_exploration=8.0,
    )
    recall.state.recall.set_bank_gradient_enabled(True)
    recall.train()
    value = torch.randn(2, 8, 4)
    rng_plan = alpha.ExecutionRNGPlan(
        seed=771,
        run_nonce="route-exploration",
        stream_key="encoder.block-3.recall",
        sample_keys=("sample-a", "sample-b"),
    )
    before = torch.random.get_rng_state().clone()
    first_candidates = alpha.query_recall_branches(
        recall,
        value,
        rng_plan=rng_plan,
    )
    second_candidates = alpha.query_recall_branches(
        recall,
        value,
        rng_plan=rng_plan,
    )
    assert torch.equal(torch.random.get_rng_state(), before)
    assert torch.equal(
        first_candidates.candidate_group_index,
        second_candidates.candidate_group_index,
    )
    assert first_candidates.query_rng_fingerprint == rng_plan.fingerprint

    result = alpha.run_batched_refine(
        recall,
        value,
        candidates=first_candidates,
        refine_policy=RefinePolicy.fixed(3, trace_level="routes"),
        rng_plan=rng_plan,
    )
    replay = alpha.run_batched_refine(
        recall,
        value,
        candidates=first_candidates,
        refine_policy=RefinePolicy.fixed(3, trace_level="routes"),
        rng_plan=rng_plan,
    )
    assert torch.equal(result.value, replay.value)
    assert torch.equal(
        result.branch_diagnostics["recall_route_history"],
        replay.branch_diagnostics["recall_route_history"],
    )
    assert result.execution_rng_stream_key == rng_plan.stream_key
    assert result.execution_rng_domains == (
        "candidate-route",
        "refine-route",
    )

    wrong_plan = alpha.ExecutionRNGPlan(
        seed=772,
        run_nonce="route-exploration-wrong",
        stream_key="encoder.block-3.recall",
        sample_keys=("sample-a", "sample-b"),
    )
    with pytest.raises(alpha.BatchedRefineContractError, match="query RNG"):
        alpha.run_batched_refine(
            recall,
            value,
            candidates=first_candidates,
            rng_plan=wrong_plan,
        )


def test_keyed_rng_stream_identity_decorrelates_distinct_callsites() -> None:
    first = alpha.ExecutionRNGPlan(
        seed=55,
        run_nonce="same-run",
        stream_key="encoder.block-1.recall",
        sample_keys=("sample",),
    )
    second = alpha.ExecutionRNGPlan(
        seed=55,
        run_nonce="same-run",
        stream_key="encoder.block-2.recall",
        sample_keys=("sample",),
    )

    assert first.fingerprint != second.fingerprint
    assert first.contract_ref == canonical_contract_reference(
        "arti/execution-rng-plan@2"
    )
    assert first.contract_ref.startswith("arti/execution-rng-plan@sha256:")
    assert first._derived_seed(
        sample_key="sample",
        branch_origin=0,
        phase="half-survival",
        refine_step=0,
    ) != second._derived_seed(
        sample_key="sample",
        branch_origin=0,
        phase="half-survival",
        refine_step=0,
    )


def test_unused_rng_plan_keeps_deterministic_execution_receipt() -> None:
    recall = _recall(topk=2)
    result = alpha.run_batched_refine(
        recall,
        torch.randn(1, 2, 4),
        rng_plan=alpha.ExecutionRNGPlan(
            seed=56,
            run_nonce="unused-plan",
            stream_key="encoder.block-3.recall",
            sample_keys=("sample",),
        ),
    )

    assert result.execution_rng_fingerprint is None
    assert result.execution_rng_stream_key is None
    assert result.execution_rng_domains == ()


def test_keyed_rng_replays_without_consuming_global_rng_and_is_permutation_covariant() -> None:
    recall = Recall(
        dim=4,
        slots=12,
        activation="half",
        routing="grouped",
        group_size=2,
        group_topk=3,
        key_dim=4,
        dropout=0.25,
    )
    recall.train()
    value = torch.randn(2, 3, 4)
    candidates = alpha.query_recall_branches(recall, value, max_k=3)
    rng_plan = alpha.ExecutionRNGPlan(
        seed=9127,
        run_nonce="rng-replay",
        stream_key="decoder.block-5.recall",
        sample_keys=("sample-a", "sample-b"),
    )
    policy = RefinePolicy.fixed(3, trace_level="routes")
    before = torch.random.get_rng_state().clone()
    first = alpha.run_batched_refine(
        recall,
        value,
        candidates=candidates,
        refine_policy=policy,
        rng_plan=rng_plan,
    )
    second = alpha.run_batched_refine(
        recall,
        value,
        candidates=candidates,
        refine_policy=policy,
        rng_plan=rng_plan,
    )
    assert torch.equal(torch.random.get_rng_state(), before)
    assert torch.equal(first.value, second.value)
    assert torch.equal(first.delta, second.delta)
    assert first.execution_rng_fingerprint == rng_plan.fingerprint

    permutation = torch.tensor([[2, 0, 1], [1, 2, 0]])
    permuted = candidates.permute_branches(permutation)
    permuted_result = alpha.run_batched_refine(
        recall,
        value,
        candidates=permuted,
        refine_policy=policy,
        rng_plan=rng_plan,
    )
    batch = torch.arange(value.shape[0]).unsqueeze(1)
    assert torch.equal(permuted_result.value, first.value[batch, permutation])
    assert torch.equal(permuted_result.delta, first.delta[batch, permutation])


def test_keyed_rng_plan_changes_stochastic_trajectory() -> None:
    recall = Recall(
        dim=4,
        slots=12,
        activation="half",
        routing="grouped",
        group_size=2,
        group_topk=2,
        key_dim=4,
    )
    value = torch.randn(1, 16, 4)
    candidates = alpha.query_recall_branches(recall, value)
    one = alpha.run_batched_refine(
        recall,
        value,
        candidates=candidates,
        rng_plan=alpha.ExecutionRNGPlan(
            seed=1,
            run_nonce="one",
            stream_key="decoder.block-7.recall",
            sample_keys=("sample",),
        ),
    )
    two = alpha.run_batched_refine(
        recall,
        value,
        candidates=candidates,
        rng_plan=alpha.ExecutionRNGPlan(
            seed=2,
            run_nonce="two",
            stream_key="decoder.block-7.recall",
            sample_keys=("sample",),
        ),
    )
    assert not torch.equal(one.value, two.value)


def test_batched_refine_binds_deterministic_learnable_half_state() -> None:
    recall = _recall(topk=2)
    recall.state.recall_activation = Half(stochastic=False, learnable=True)
    value = torch.randn(1, 3, 4)
    candidates = alpha.query_recall_branches(recall, value)

    with torch.no_grad():
        recall.state.recall_activation._threshold.add_(0.125)

    with pytest.raises(
        alpha.BatchedRefineContractError,
        match="execution tensors changed",
    ):
        alpha.run_batched_refine(recall, value, candidates=candidates)


def _formula_operation(
    *,
    batch: int,
    topology: bool,
    refine_steps: int = 1,
) -> alpha.FormulaResidentOperation | alpha.TopologyFormulaResidentOperation:
    program = alpha.FormulaFabricProgram(
        arena_capacity=3,
        feature_dim=4,
        steps=((alpha.FormulaInvocation(alpha.FormulaPrimitive.ADD, 2),),),
    )
    weights = torch.zeros(batch, 1, 1, 2, 3)
    weights[..., 0, 0] = 1
    weights[..., 1, 1] = 1
    enabled = torch.ones(batch, 1, 1, dtype=torch.bool)
    route = alpha.FormulaRoutePlan(weights, enabled, enabled, enabled)
    compute = alpha.FormulaFabricCompute(
        alpha.FormulaFabric(program),
        active_count=3,
    )
    if not topology:
        return alpha.FormulaResidentOperation(
            compute,
            route,
            refine_steps=refine_steps,
        )
    fold, unfold = alpha.ReversibleTopology(
        active_count=3,
        policy=alpha.FixedTopologyPolicy(order=[2, 0, 1]),
    ).operations()
    return alpha.TopologyFormulaResidentOperation(
        fold,
        unfold,
        compute,
        route,
        refine_steps=refine_steps,
    )


def test_query_recall_branches_preserves_topk_identity_mass_and_lineage() -> None:
    torch.manual_seed(1907)
    recall = _recall()
    value = torch.randn(2, 5, 4, requires_grad=True)
    mask = torch.tensor(
        [[True, True, True, False, False], [True, True, True, True, True]]
    )

    batch = alpha.query_recall_branches(recall, value, mask=mask)

    assert component_ref(batch).startswith("arti/recall-branch-batch@sha256:")
    assert batch.candidate_group_index.shape == (2, 5, 3)
    assert batch.candidate_slot_index.shape == (2, 5, 3, 2)
    assert batch.candidate_slot_weight.shape == (2, 5, 3, 2)
    assert batch.candidate_context.shape == (2, 5, 3, 4)
    assert batch.route_mass.shape == (2, 5, 3)
    assert batch.selection_weight.shape == (2, 5, 3)
    assert batch.candidate_mask.shape == (2, 5, 3)
    assert batch.branch_mask.shape == (2, 3)
    assert batch.source_ref.startswith("arti/recall@sha256:")
    assert len(batch.source_config_fingerprint) == 64
    groups = torch.sort(batch.candidate_group_index, dim=-1).values
    assert torch.all(groups[..., 1:] != groups[..., :-1])
    assert torch.all(
        batch.candidate_slot_index // 2
        == batch.candidate_group_index.unsqueeze(-1)
    )
    valid_weight = batch.selection_weight[mask]
    torch.testing.assert_close(
        valid_weight.sum(dim=-1),
        torch.ones_like(valid_weight[:, 0]),
    )
    assert torch.all(batch.route_mass[mask].sum(dim=-1) <= 1.0 + 1e-6)
    valid_slot_weight = batch.candidate_slot_weight[batch.candidate_mask]
    torch.testing.assert_close(
        valid_slot_weight.sum(dim=-1),
        torch.ones_like(valid_slot_weight[:, 0]),
    )
    assert not batch.candidate_mask[0, 3:].any()
    assert batch.route_mass.requires_grad
    assert batch.selection_weight.requires_grad


def test_per_bank_candidates_preserve_partition_identity_without_capacity_dilution() -> None:
    torch.manual_seed(1709)
    bank_a = Recall(
        dim=4,
        slots=8,
        activation="none",
        routing="grouped",
        group_size=2,
        group_topk=4,
        key_dim=4,
        routing_normalizer="per_bank",
    )
    bank_ab = Recall(
        dim=4,
        slots=16,
        activation="none",
        routing="grouped",
        group_size=2,
        group_topk=8,
        key_dim=4,
        routing_normalizer="per_bank",
    )
    bank_a.state.recall.configure_expert_routes(("a",), ((0, 4),))
    bank_ab.state.recall.configure_expert_routes(
        ("a", "b"),
        ((0, 4), (4, 8)),
        member_fingerprints=("1" * 64, "2" * 64),
    )
    _copy_recall_prefix(bank_a, bank_ab)
    value = torch.randn(2, 3, 4)

    candidates_a = alpha.query_recall_branches(bank_a, value, max_k=4)
    candidates_ab = alpha.query_recall_branches(bank_ab, value, max_k=8)
    a_mass = torch.zeros_like(candidates_a.route_mass)
    a_mass.scatter_(
        -1,
        candidates_a.candidate_group_index,
        candidates_a.route_mass,
    )
    ab_a_mass = torch.zeros_like(candidates_a.route_mass)
    ab_groups = candidates_ab.candidate_group_index[..., :]
    ab_values = torch.where(
        ab_groups < 4,
        candidates_ab.route_mass,
        torch.zeros_like(candidates_ab.route_mass),
    )
    ab_a_mass.scatter_add_(-1, ab_groups.clamp_max(3), ab_values)

    torch.testing.assert_close(ab_a_mass, a_mass)
    assert candidates_ab.routing_normalizer == "per_bank"
    assert candidates_ab.candidate_policy == "per-bank-reserved-topk@1"
    assert candidates_ab.partition_quota == (4, 4)
    assert candidates_ab.partition_names == ("a", "b")
    assert candidates_ab.partition_ranges == ((0, 4), (4, 8))
    assert candidates_ab.partition_member_fingerprints == (
        "1" * 64,
        "2" * 64,
    )
    assert torch.equal(
        candidates_ab.candidate_partition_index,
        (candidates_ab.candidate_group_index >= 4).to(torch.long),
    )

    permuted = candidates_ab.permute_branches(torch.tensor([7, 0, 6, 1, 5, 2, 4, 3]))
    assert permuted.partition_layout_fingerprint == candidates_ab.partition_layout_fingerprint
    assert torch.equal(
        permuted.candidate_partition_index,
        (permuted.candidate_group_index >= 4).to(torch.long),
    )


def test_per_bank_candidate_allocation_reserves_breadth_under_fixed_k() -> None:
    recall = Recall(
        dim=4,
        slots=16,
        activation="none",
        routing="grouped",
        group_size=2,
        group_topk=8,
        key_dim=4,
        routing_normalizer="per_bank",
    )
    recall.state.recall.configure_expert_routes(
        ("dominant", "quiet"),
        ((0, 4), (4, 8)),
    )
    recall.state.recall.set_expert_weights((1000.0, 0.001))
    value = torch.randn(2, 3, 4)

    global_candidates = alpha.query_recall_branches(
        recall,
        value,
        max_k=4,
        candidate_allocation="global_weighted_topk",
    )
    reserved_candidates = alpha.query_recall_branches(recall, value, max_k=4)
    quota_candidates = alpha.query_recall_branches(
        recall,
        value,
        max_k=4,
        candidate_allocation="explicit_bank_quota",
        bank_quotas=(1, 3),
    )

    assert global_candidates.candidate_policy == "global-weighted-topk@1"
    assert global_candidates.partition_quota == ()
    assert torch.all(global_candidates.candidate_partition_index == 0)
    assert reserved_candidates.candidate_policy == "per-bank-reserved-topk@1"
    assert reserved_candidates.partition_quota == (2, 2)
    assert torch.all(reserved_candidates.candidate_partition_index[..., :2] == torch.tensor([0, 1]))
    assert torch.all(
        torch.bincount(
            reserved_candidates.candidate_partition_index.flatten(), minlength=2
        )
        == torch.tensor([12, 12])
    )
    assert quota_candidates.candidate_policy == "per-bank-quota-topk@1"
    assert quota_candidates.partition_quota == (1, 3)
    assert torch.all(
        torch.bincount(quota_candidates.candidate_partition_index.flatten(), minlength=2)
        == torch.tensor([6, 18])
    )


def test_per_bank_reserved_requires_k_to_cover_each_enabled_bank() -> None:
    recall = Recall(
        dim=4,
        slots=8,
        activation="none",
        routing="grouped",
        group_size=2,
        group_topk=4,
        key_dim=4,
        routing_normalizer="per_bank",
    )
    recall.state.recall.configure_expert_routes(("a", "b"), ((0, 2), (2, 4)))

    with pytest.raises(alpha.BatchedRefineContractError, match="cover every enabled Bank"):
        alpha.query_recall_branches(recall, torch.randn(1, 2, 4), max_k=1)


def test_zero_weight_partition_is_padding_not_an_executable_branch() -> None:
    torch.manual_seed(1710)
    bank_a = Recall(
        dim=4,
        slots=8,
        activation="none",
        routing="grouped",
        group_size=2,
        group_topk=4,
        key_dim=4,
        routing_normalizer="per_bank",
    )
    bank_ab = Recall(
        dim=4,
        slots=16,
        activation="none",
        routing="grouped",
        group_size=2,
        group_topk=8,
        key_dim=4,
        routing_normalizer="per_bank",
    )
    bank_a.state.recall.configure_expert_routes(("a",), ((0, 4),))
    bank_ab.state.recall.configure_expert_routes(
        ("a", "b"),
        ((0, 4), (4, 8)),
    )
    bank_ab.state.recall.set_expert_weights((1.0, 0.0))
    _copy_recall_prefix(bank_a, bank_ab)
    value = torch.randn(1, 3, 4)

    torch.testing.assert_close(bank_ab(value), bank_a(value))
    candidates = alpha.query_recall_branches(bank_ab, value, max_k=8)

    assert candidates.max_k == 8
    assert candidates.requested_active_k.tolist() == [8]
    assert candidates.active_k.tolist() == [4]
    assert candidates.active_partition_count().tolist() == [1]
    assert candidates.active_partition_mask().tolist() == [[True, False]]
    assert candidates.active_branch_count_by_partition().tolist() == [[4, 0]]
    assert candidates.branch_mask.tolist() == [
        [True, True, True, True, False, False, False, False]
    ]
    assert torch.all(candidates.candidate_partition_index[..., :4] == 0)


def test_per_bank_reordering_preserves_named_expert_behavior() -> None:
    torch.manual_seed(1711)
    source = Recall(
        dim=4,
        slots=8,
        activation="none",
        routing="grouped",
        group_size=2,
        group_topk=4,
        key_dim=4,
        routing_normalizer="per_bank",
    )
    reordered = Recall(
        dim=4,
        slots=8,
        activation="none",
        routing="grouped",
        group_size=2,
        group_topk=4,
        key_dim=4,
        routing_normalizer="per_bank",
    )
    source.state.recall.configure_expert_routes(
        ("game", "animal"),
        ((0, 2), (2, 4)),
        member_fingerprints=("1" * 64, "2" * 64),
    )
    source.state.recall.set_expert_weights((0.75, 0.25))
    source.state.recall.set_expert_influences((1.0, -0.5))
    reordered.state.recall.configure_expert_routes(
        ("animal", "game"),
        ((0, 2), (2, 4)),
        member_fingerprints=("2" * 64, "1" * 64),
    )
    reordered.state.recall.set_expert_weights((0.25, 0.75))
    reordered.state.recall.set_expert_influences((-0.5, 1.0))
    with torch.no_grad():
        source_field = source.state.recall
        target_field = reordered.state.recall
        target_field.query.weight.copy_(source_field.query.weight)
        target_field.group_bank.copy_(
            torch.cat((source_field.group_bank[2:4], source_field.group_bank[0:2]))
        )
        target_field.key_bank.copy_(
            torch.cat((source_field.key_bank[4:8], source_field.key_bank[0:4]))
        )
        target_field.bank.copy_(
            torch.cat((source_field.bank[4:8], source_field.bank[0:4]))
        )

    value = torch.randn(2, 3, 4)
    torch.testing.assert_close(reordered(value), source(value))

    source_candidates = alpha.query_recall_branches(source, value, max_k=4)
    reordered_candidates = alpha.query_recall_branches(reordered, value, max_k=4)
    for name in ("game", "animal"):
        source_index = source_candidates.partition_names.index(name)
        reordered_index = reordered_candidates.partition_names.index(name)
        source_mass = torch.where(
            source_candidates.candidate_partition_index == source_index,
            source_candidates.route_mass,
            torch.zeros_like(source_candidates.route_mass),
        ).sum(dim=-1)
        reordered_mass = torch.where(
            reordered_candidates.candidate_partition_index == reordered_index,
            reordered_candidates.route_mass,
            torch.zeros_like(reordered_candidates.route_mass),
        ).sum(dim=-1)
        torch.testing.assert_close(reordered_mass, source_mass)


def test_branch_state_is_materialized_and_flattening_preserves_branch_axis() -> None:
    recall = _recall(topk=2)
    value = torch.randn(2, 3, 4)
    batch = alpha.query_recall_branches(recall, value)

    branches = batch.expand_state(value)
    assert branches.shape == (2, 2, 3, 4)
    branches[0, 0, 0, 0] += 10
    assert branches[0, 0, 0, 0] != branches[0, 1, 0, 0]
    assert branches[0, 1, 0, 0] == value[0, 0, 0]
    groups = batch.flattened_initial_groups()
    masks = batch.flattened_token_mask()
    contexts = batch.flattened_candidate_context()
    assert groups.shape == (4, 3, 1)
    assert masks.shape == (4, 3)
    assert contexts.shape == (4, 3, 4)
    assert torch.equal(
        groups.reshape(2, 2, 3, 1)[:, :, :, 0],
        batch.candidate_group_index.permute(0, 2, 1),
    )


def test_query_recall_branches_supports_fixed_max_k_and_k1_degeneration() -> None:
    recall = _recall(topk=3)
    value = torch.randn(1, 2, 4)

    batch = alpha.query_recall_branches(recall, value, max_k=1)

    assert batch.max_k == 1
    assert batch.requested_active_k.tolist() == [1]
    assert batch.candidate_group_index.shape == (1, 2, 1)
    assert batch.branch_mask.shape == (1, 1)
    assert batch.active_k.tolist() == [1]


def test_candidate_class_reference_cannot_cross_single_and_formula_schema() -> None:
    value = torch.randn(1, 2, 4)
    single = alpha.query_recall_branches(_recall(topk=2), value)
    with pytest.raises(
        alpha.BatchedRefineContractError,
        match="class/reference",
    ):
        type(single)(
            **{
                **single.__dict__,
                "value_composition": "product",
                "schema_version": 6,
            }
        )

    formula = alpha.query_recall_branches(
        Recall(
            dim=4,
            slots=16,
            formula="arti/affine@1",
            routing="grouped",
            group_size=2,
            group_topk=4,
            key_dim=4,
            activation="none",
        ),
        value,
        max_k=2,
        formula_beam_width=2,
    )
    with pytest.raises(
        alpha.BatchedRefineContractError,
        match="class/reference",
    ):
        type(formula)(
            **{
                **formula.__dict__,
                "value_composition": "single",
                "schema_version": 3,
            }
        )


def test_query_recall_branches_supports_per_sample_active_k() -> None:
    recall = _recall(topk=3)
    value = torch.randn(3, 2, 4)
    mask = torch.tensor(
        [[True, True], [True, False], [False, False]],
    )

    batch = alpha.query_recall_branches(
        recall,
        value,
        mask=mask,
        max_k=3,
        active_k=torch.tensor([1, 3, 2]),
    )

    assert batch.max_k == 3
    assert batch.requested_active_k.tolist() == [1, 3, 2]
    assert batch.active_k.tolist() == [1, 3, 0]
    assert batch.branch_mask.tolist() == [
        [True, False, False],
        [True, True, True],
        [False, False, False],
    ]
    expected_mask = mask.unsqueeze(-1) & batch.branch_mask.unsqueeze(1)
    assert torch.equal(batch.candidate_mask, expected_mask)
    assert not batch.candidate_context[~batch.candidate_mask].any()
    assert not batch.route_mass[~batch.candidate_mask].any()


def test_active_k_validity_follows_candidate_permutation() -> None:
    recall = _recall(topk=3)
    value = torch.randn(1, 2, 4)
    candidates = alpha.query_recall_branches(
        recall,
        value,
        max_k=3,
        active_k=2,
    )

    permuted = candidates.permute_branches(torch.tensor([2, 0, 1]))

    assert permuted.active_k.tolist() == [2]
    assert permuted.requested_active_k.tolist() == [2]
    assert permuted.branch_mask.tolist() == [[False, True, True]]
    assert permuted.branch_origin_index.tolist() == [[2, 0, 1]]
    assert torch.equal(
        permuted.candidate_mask,
        permuted.token_mask.unsqueeze(-1) & permuted.branch_mask.unsqueeze(1),
    )


def test_packed_active_executes_only_initially_eligible_rows() -> None:
    torch.manual_seed(2051)
    recall = _recall(topk=3)
    value = torch.randn(3, 2, 4)
    mask = torch.tensor(
        [[True, True], [True, False], [False, False]],
    )
    candidates = alpha.query_recall_branches(
        recall,
        value,
        mask=mask,
        max_k=3,
        active_k=torch.tensor([1, 3, 2]),
    )
    policy = RefinePolicy.fixed(2, trace_level="routes")
    static = alpha.run_batched_refine(
        recall,
        value,
        candidates=candidates,
        refine_policy=policy,
    )
    packed = alpha.run_batched_refine(
        recall,
        value,
        candidates=candidates,
        refine_policy=policy,
        plan=alpha.BatchedRefinePlan.recall_only(
            execution_layout="packed_active"
        ),
    )

    torch.testing.assert_close(packed.value, static.value)
    torch.testing.assert_close(packed.delta, static.delta)
    torch.testing.assert_close(
        packed.branch_diagnostics["recall_steps_committed"],
        static.branch_diagnostics["recall_steps_committed"],
    )
    assert packed.global_diagnostics["batched_physical_branch_rows"].item() == 4
    assert packed.global_diagnostics["batched_static_capacity_rows"].item() == 9
    assert packed.global_diagnostics["batched_initially_inactive_rows"].item() == 5
    assert packed.global_diagnostics["batched_packing_mode"].item() == 1
    assert packed.global_diagnostics[
        "batched_packed_active_flat_index"
    ].tolist() == [0, 3, 4, 5]


def test_packed_active_all_invalid_is_a_zero_row_fast_path() -> None:
    recall = _recall(topk=3)
    value = torch.randn(2, 2, 4)
    mask = torch.zeros(2, 2, dtype=torch.bool)
    candidates = alpha.query_recall_branches(
        recall,
        value,
        mask=mask,
        max_k=3,
        active_k=3,
    )
    original_forward = recall.state.forward

    def trap(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("zero-row packed execution called Recall state")

    recall.state.forward = trap  # type: ignore[method-assign]
    result = alpha.run_batched_refine(
        recall,
        value,
        candidates=candidates,
        refine_policy=RefinePolicy.fixed(3, trace_level="routes"),
        plan=alpha.BatchedRefinePlan.recall_only(
            execution_layout="packed_active"
        ),
    )
    recall.state.forward = original_forward  # type: ignore[method-assign]

    torch.testing.assert_close(
        result.value,
        value.unsqueeze(1).expand(2, 3, 2, 4),
    )
    assert not result.delta.any()
    assert result.global_diagnostics["batched_physical_branch_rows"].item() == 0
    assert result.global_diagnostics["recall_kernel_steps"].item() == 0
    assert not result.branch_diagnostics["recall_step_attempted"].any()
    assert not result.global_diagnostics["batched_executed_branch_mask"].any()
    assert result.execution_layout == "packed_active"


def test_packed_active_zero_rows_do_not_claim_unexecuted_rng_domains() -> None:
    recall = Recall(
        dim=4,
        slots=12,
        activation="half",
        routing="grouped",
        group_size=2,
        group_topk=3,
        key_dim=4,
        dropout=0.25,
        route_exploration=1.0,
    )
    recall.state.recall.set_bank_gradient_enabled(True)
    recall.train()
    value = torch.randn(1, 2, 4)
    rng_plan = alpha.ExecutionRNGPlan(
        seed=2057,
        run_nonce="zero-row-rng-receipt",
        stream_key="encoder.block-2.recall",
        sample_keys=("sample",),
    )
    candidates = alpha.query_recall_branches(
        recall,
        value,
        mask=torch.zeros(1, 2, dtype=torch.bool),
        rng_plan=rng_plan,
    )
    result = alpha.run_batched_refine(
        recall,
        value,
        candidates=candidates,
        rng_plan=rng_plan,
        plan=alpha.BatchedRefinePlan.recall_only(
            execution_layout="packed_active"
        ),
    )

    assert result.execution_rng_domains == ("candidate-route",)
    assert "half-survival" not in result.execution_rng_domains
    assert "recall-dropout" not in result.execution_rng_domains
    assert "refine-route" not in result.execution_rng_domains


@pytest.mark.parametrize("topology", [False, True])
def test_packed_active_all_invalid_preserves_composed_diagnostic_abi(
    topology: bool,
) -> None:
    recall = _recall(topk=3)
    value = torch.randn(2, 3, 4)
    candidates = alpha.query_recall_branches(
        recall,
        value,
        mask=torch.zeros(2, 3, dtype=torch.bool),
        active_k=3,
    )
    result = alpha.run_batched_refine(
        recall,
        value,
        candidates=candidates,
        refine_policy=RefinePolicy.fixed(2, trace_level="routes"),
        plan=alpha.BatchedRefinePlan.compose(
            _formula_operation(batch=6, topology=topology, refine_steps=2),
            execution_layout="packed_active",
        ),
    )

    for name in (
        "batched_formula_valid_trace",
        "batched_formula_fire_trace",
        "batched_formula_commit_trace",
    ):
        trace = result.branch_diagnostics[name]
        assert trace.shape[:4] == (2, 3, 2, 2)
        assert trace.dtype == torch.bool
        assert not trace.any()
    for name in (
        "batched_formula_cells",
        "batched_formula_fire_count",
        "batched_formula_commit_count",
        "batched_formula_invoked_cells",
        "batched_formula_effective_cells",
    ):
        assert result.branch_diagnostics[name].shape == (2, 3)
        assert not result.branch_diagnostics[name].any()
    if topology:
        permutation = result.branch_diagnostics[
            "batched_topology_permutation_trace"
        ]
        assert permutation.shape == (2, 3, 2, 3)
        assert (permutation == -1).all()
        assert not result.branch_diagnostics[
            "batched_topology_unfold_verified"
        ].any()
        assert result.global_diagnostics[
            "batched_topology_active_count"
        ].item() == 3


def test_packed_active_fails_closed_during_direct_compilation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recall = _recall(topk=2)
    value = torch.randn(1, 2, 4)
    candidates = alpha.query_recall_branches(recall, value)
    monkeypatch.setattr(torch.compiler, "is_compiling", lambda: True)

    with pytest.raises(
        alpha.BatchedRefineContractError,
        match="eager-only",
    ):
        alpha.run_batched_refine(
            recall,
            value,
            candidates=candidates,
            plan=alpha.BatchedRefinePlan.recall_only(
                execution_layout="packed_active"
            ),
        )


def test_packed_active_preserves_gradients_on_executed_rows() -> None:
    torch.manual_seed(2053)
    static_recall = _recall(topk=3)
    packed_recall = _recall(topk=3)
    packed_recall.load_state_dict(static_recall.state_dict())
    static_value = torch.randn(2, 2, 4, requires_grad=True)
    packed_value = static_value.detach().clone().requires_grad_(True)
    active_k = torch.tensor([1, 3])
    policy = RefinePolicy.fixed(2, trace_level="routes")

    static_candidates = alpha.query_recall_branches(
        static_recall, static_value, active_k=active_k
    )
    static = alpha.run_batched_refine(
        static_recall,
        static_value,
        candidates=static_candidates,
        refine_policy=policy,
    )
    static.value.sum().backward()

    packed_candidates = alpha.query_recall_branches(
        packed_recall, packed_value, active_k=active_k
    )
    packed = alpha.run_batched_refine(
        packed_recall,
        packed_value,
        candidates=packed_candidates,
        refine_policy=policy,
        plan=alpha.BatchedRefinePlan.recall_only(
            execution_layout="packed_active"
        ),
    )
    packed.value.sum().backward()

    torch.testing.assert_close(packed.value, static.value)
    torch.testing.assert_close(packed_value.grad, static_value.grad)
    torch.testing.assert_close(
        packed_recall.state.recall.bank.grad,
        static_recall.state.recall.bank.grad,
    )


def test_packed_active_keeps_keyed_stochastic_branch_identity() -> None:
    torch.manual_seed(2054)
    recall = Recall(
        dim=4,
        slots=12,
        formula="arti/delta@1",
        activation="half",
        routing="grouped",
        group_size=2,
        group_topk=3,
        key_dim=4,
    )
    recall.state.recall_activation = Half(stochastic=True)
    value = torch.randn(2, 2, 4)
    candidates = alpha.query_recall_branches(
        recall, value, active_k=torch.tensor([1, 3])
    )
    rng = alpha.ExecutionRNGPlan(
        seed=2054,
        run_nonce="packed-active-test",
        stream_key="main",
        sample_keys=("sample-a", "sample-b"),
    )
    policy = RefinePolicy.fixed(2, trace_level="routes")
    static = alpha.run_batched_refine(
        recall,
        value,
        candidates=candidates,
        refine_policy=policy,
        rng_plan=rng,
    )
    packed = alpha.run_batched_refine(
        recall,
        value,
        candidates=candidates,
        refine_policy=policy,
        rng_plan=rng,
        plan=alpha.BatchedRefinePlan.recall_only(
            execution_layout="packed_active"
        ),
    )

    torch.testing.assert_close(packed.value, static.value)
    torch.testing.assert_close(packed.delta, static.delta)


@pytest.mark.parametrize("topology", [False, True])
def test_packed_active_formula_topology_is_branch_permutation_equivariant(
    topology: bool,
) -> None:
    torch.manual_seed(2052)
    recall = _recall(topk=3)
    value = torch.randn(2, 3, 4)
    candidates = alpha.query_recall_branches(
        recall,
        value,
        active_k=torch.tensor([1, 3]),
    )
    operation = _formula_operation(batch=6, topology=topology)
    policy = RefinePolicy.fixed(2, trace_level="routes")
    packed = alpha.run_batched_refine(
        recall,
        value,
        candidates=candidates,
        refine_policy=policy,
        plan=alpha.BatchedRefinePlan.compose(
            operation,
            execution_layout="packed_active",
        ),
    )

    active = candidates.branch_mask
    torch.testing.assert_close(
        packed.value[~active],
        value.unsqueeze(1).expand(2, 3, 3, 4)[~active],
    )
    assert not packed.delta[~active].any()
    for name in (
        "batched_formula_cells",
        "batched_formula_fire_count",
        "batched_formula_commit_count",
    ):
        assert packed.branch_diagnostics[name][active].numel()
        assert not packed.branch_diagnostics[name][~active].any()
    if topology:
        assert packed.branch_diagnostics[
            "batched_topology_permutation_trace"
        ][active].numel()
        assert not packed.branch_diagnostics[
            "batched_topology_unfold_verified"
        ][~active].any()

    order = torch.tensor([2, 0, 1])
    permuted_candidates = candidates.permute_branches(order)
    permuted = alpha.run_batched_refine(
        recall,
        value,
        candidates=permuted_candidates,
        refine_policy=policy,
        plan=alpha.BatchedRefinePlan.compose(
            operation,
            execution_layout="packed_active",
        ),
    )
    torch.testing.assert_close(permuted.value, packed.value[:, order])
    torch.testing.assert_close(permuted.delta, packed.delta[:, order])


def test_per_sample_candidate_permutation_preserves_canonical_execution() -> None:
    recall = _recall(topk=3)
    value = torch.randn(2, 3, 4)
    candidates = alpha.query_recall_branches(
        recall,
        value,
        active_k=torch.tensor([2, 3]),
    )
    policy = alpha.BranchRefinePolicy(
        candidates,
        RefinePolicy.adaptive(
            max_steps=3,
            min_steps=0,
            absolute_tolerance=1e-7,
            relative_tolerance=0.0,
            trace_level="routes",
        ),
        min_steps=torch.tensor([[1, 2, 0], [1, 2, 3]]),
        max_steps=torch.tensor([[1, 2, 0], [1, 2, 3]]),
    )
    expected = alpha.run_batched_refine(
        recall,
        value,
        candidates=candidates,
        refine_policy=policy,
    )
    order = torch.tensor([[2, 0, 1], [1, 2, 0]])
    permuted_candidates = candidates.permute_branches(order)
    actual = alpha.run_batched_refine(
        recall,
        value,
        candidates=permuted_candidates,
        refine_policy=policy,
    )

    gather = order[:, :, None, None].expand_as(actual.value)
    torch.testing.assert_close(actual.value, expected.value.gather(1, gather))
    torch.testing.assert_close(actual.delta, expected.delta.gather(1, gather))
    assert (
        actual.candidates.canonical_manifest_fingerprint()
        == expected.candidates.canonical_manifest_fingerprint()
    )


@pytest.mark.parametrize("active_k", [-1, 4])
def test_active_k_rejects_values_outside_static_width(active_k: int) -> None:
    with pytest.raises(alpha.BatchedRefineContractError, match="active_k"):
        alpha.query_recall_branches(
            _recall(topk=3),
            torch.randn(1, 2, 4),
            max_k=3,
            active_k=active_k,
        )


def test_inactive_branches_remain_identity_during_refine() -> None:
    recall = _recall(topk=3)
    value = torch.randn(1, 3, 4)

    result = alpha.run_batched_refine(
        recall,
        value,
        max_k=3,
        active_k=1,
        refine_policy=RefinePolicy.fixed(2, trace_level="routes"),
    )

    torch.testing.assert_close(result.value[:, 1:], value.unsqueeze(1).expand(-1, 2, -1, -1))
    assert not result.delta[:, 1:].any()
    assert result.candidates.active_k.tolist() == [1]


def test_recall_only_plan_is_exactly_the_existing_executor() -> None:
    torch.manual_seed(2001)
    recall = _recall(topk=2)
    value = torch.randn(1, 3, 4)
    candidates = alpha.query_recall_branches(recall, value)
    policy = RefinePolicy.fixed(3, trace_level="routes")

    existing = alpha.run_batched_refine(
        recall,
        value,
        candidates=candidates,
        refine_policy=policy,
    )
    planned = alpha.run_batched_refine(
        recall,
        value,
        candidates=candidates,
        refine_policy=policy,
        plan=alpha.BatchedRefinePlan.recall_only(),
    )

    torch.testing.assert_close(planned.value, existing.value, rtol=0, atol=0)
    torch.testing.assert_close(planned.delta, existing.delta, rtol=0, atol=0)
    assert planned.operation_ref is None
    assert len(planned.plan_config_fingerprint) == 64


def test_topology_formula_plan_executes_inside_each_recall_refine_step() -> None:
    torch.manual_seed(2002)
    recall = _recall(topk=2)
    with torch.no_grad():
        recall.state.recall.bank.zero_()
    value = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]]
    )
    candidates = alpha.query_recall_branches(recall, value)
    plan = alpha.BatchedRefinePlan.compose(
        _formula_operation(batch=2, topology=True)
    )
    assert component_ref(plan).startswith("arti/batched-refine-plan@sha256:")
    assert component_ref(plan.operation).startswith("arti/batched-refine-operation@sha256:")

    result = alpha.run_batched_refine(
        recall,
        value,
        candidates=candidates,
        refine_policy=RefinePolicy.fixed(2, trace_level="routes"),
        plan=plan,
    )

    expected = value.clone()
    expected[:, 1] = value[:, 2] + value[:, 0]
    torch.testing.assert_close(result.value[:, 0], expected)
    torch.testing.assert_close(result.value[:, 1], expected)
    assert result.operation_ref.startswith(
        "arti/topology-formula-resident-operation@sha256:"
    )
    assert result.plan_config_fingerprint == plan.config_fingerprint
    route_history = result.branch_diagnostics["recall_route_history"]
    assert route_history.shape[2] == 2
    assert not torch.equal(route_history[:, :, 0], route_history[:, :, 1])


def test_batched_refine_plan_rejects_mutated_formula_route() -> None:
    recall = _recall(topk=2)
    value = torch.randn(1, 3, 4)
    operation = _formula_operation(batch=2, topology=False)
    plan = alpha.BatchedRefinePlan.compose(operation)
    operation.route.weights.add_(1)

    with pytest.raises(alpha.BatchedRefineContractError, match="operation changed"):
        alpha.run_batched_refine(recall, value, plan=plan)


def test_batched_refine_rejects_modules_constructed_inside_inference_mode() -> None:
    value = torch.randn(1, 2, 4)

    with torch.inference_mode():
        inference_recall = Recall(
            dim=4,
            slots=12,
            formula="arti/delta@1",
            activation="none",
            routing="grouped",
            group_size=2,
            group_topk=2,
            key_dim=4,
        )
        with pytest.raises(alpha.BatchedRefineContractError, match="no mutation version"):
            alpha.query_recall_branches(inference_recall, value)


def test_batched_refine_runs_with_normal_parameters_inside_inference_mode() -> None:
    recall = _recall(topk=2)

    with torch.inference_mode():
        value = torch.randn(1, 2, 4)
        candidates = alpha.query_recall_branches(recall, value)
        result = alpha.run_batched_refine(
            recall,
            value,
            candidates=candidates,
        )

    assert result.value.shape == (1, 2, 2, 4)
    assert result.value._version >= 0
    assert candidates.input_version == -1
    assert candidates.input_snapshot is not None


def test_batched_refine_rejects_source_mutation_inside_inference_mode() -> None:
    recall = _recall(topk=2)
    value = torch.randn(1, 2, 4)

    with torch.inference_mode():
        candidates = alpha.query_recall_branches(recall, value)
        recall.state.recall.bank.add_(1)
        with pytest.raises(alpha.BatchedRefineContractError, match="source changed"):
            alpha.run_batched_refine(recall, value, candidates=candidates)


def test_batched_refine_rejects_inference_input_mutation_after_query() -> None:
    recall = _recall(topk=2)

    with torch.inference_mode():
        value = torch.randn(1, 2, 4)
        candidates = alpha.query_recall_branches(recall, value)
        value.add_(1)
        with pytest.raises(alpha.BatchedRefineContractError, match="inference input changed"):
            alpha.run_batched_refine(recall, value, candidates=candidates)


def test_batched_refine_rejects_inference_candidate_mutation() -> None:
    recall = _recall(topk=2)
    value = torch.randn(1, 2, 4)

    with torch.inference_mode():
        candidates = alpha.query_recall_branches(recall, value)
        candidates.route_mass.add_(1)
        with pytest.raises(alpha.BatchedRefineContractError, match="bridge tensor changed"):
            alpha.run_batched_refine(recall, value, candidates=candidates)


def test_vector_input_can_materialize_branch_state() -> None:
    recall = _recall(topk=2)
    value = torch.randn(3, 4)

    batch = alpha.query_recall_branches(recall, value)

    assert batch.expand_state(value).shape == (3, 2, 1, 4)


def test_explicit_single_candidate_can_seed_existing_recall_kernel() -> None:
    recall = _recall(topk=3)
    value = torch.randn(2, 4, 4)
    batch = alpha.query_recall_branches(recall, value)
    branches = batch.expand_state(value).reshape(6, 4, 4)

    read = recall.state.recall(
        branches,
        batch.flattened_token_mask(),
        selected_groups=batch.flattened_initial_groups(),
    )

    assert read.context.shape == branches.shape
    selected_group = (
        read.indices[..., 0] // recall.state.recall.group_size
    ).squeeze(-1)
    assert torch.equal(selected_group, batch.flattened_initial_groups()[..., 0])


def test_batched_refine_matches_serial_candidates_and_requeries_after_seed() -> None:
    torch.manual_seed(1941)
    recall = _recall(topk=3)
    with torch.no_grad():
        recall.state.recall.bank.normal_(std=0.4)
    value = torch.randn(1, 3, 4)
    policy = RefinePolicy.fixed(2, trace_level="routes")
    candidates = alpha.query_recall_branches(recall, value)

    result = alpha.run_batched_refine(
        recall,
        value,
        candidates=candidates,
        refine_policy=policy,
    )

    assert result.value.shape == (1, 3, 3, 4)
    assert result.delta.shape == result.value.shape
    assert result.branch_diagnostics["recall_index_history"].shape[:4] == (1, 3, 2, 3)
    first_groups = (
        result.branch_diagnostics["recall_index_history"][:, :, 0, :, 0, 0]
        // recall.state.recall.group_size
    )
    expected = candidates.candidate_group_index.permute(0, 2, 1)
    assert torch.equal(first_groups, expected)

    serial_values = []
    serial_deltas = []
    serial_diagnostics: list[dict[str, torch.Tensor]] = []
    for branch in range(3):
        seed = candidates.candidate_group_index[:, :, branch : branch + 1].expand(
            -1, -1, candidates.source_topk
        )
        output, delta, diagnostics = recall.state(
            value,
            candidates.token_mask,
            refine_policy=policy,
            selected_groups=seed,
        )
        serial_values.append(output)
        serial_deltas.append(delta)
        serial_diagnostics.append(diagnostics)
    torch.testing.assert_close(result.value, torch.stack(serial_values, dim=1))
    torch.testing.assert_close(result.delta, torch.stack(serial_deltas, dim=1))
    for name in (
        "recall_index_history",
        "recall_route_history",
        "recall_steps_attempted",
        "recall_steps_committed",
        "recall_stop_reason",
    ):
        expected_diagnostic = torch.stack(
            [diagnostics[name] for diagnostics in serial_diagnostics],
            dim=1,
        )
        torch.testing.assert_close(result.branch_diagnostics[name], expected_diagnostic)


def test_batched_refine_is_equivariant_to_candidate_axis_permutation() -> None:
    torch.manual_seed(1993)
    recall = _recall(topk=3)
    value = torch.randn(2, 4, 4)
    candidates = alpha.query_recall_branches(recall, value)
    policy = RefinePolicy.fixed(2, trace_level="routes")
    original = alpha.run_batched_refine(
        recall,
        value,
        candidates=candidates,
        refine_policy=policy,
    )
    order = torch.tensor([2, 0, 1])
    permuted_candidates = candidates.permute_branches(order)
    assert (
        permuted_candidates.canonical_manifest_fingerprint()
        == candidates.canonical_manifest_fingerprint()
    )
    permuted = alpha.run_batched_refine(
        recall,
        value,
        candidates=permuted_candidates,
        refine_policy=policy,
    )

    torch.testing.assert_close(
        permuted.value,
        original.value.index_select(1, order),
    )


def test_batched_refine_gradients_reach_query_and_bank() -> None:
    recall = _recall(topk=2)
    recall.state.recall.set_bank_gradient_enabled(True)
    value = torch.randn(2, 3, 4, requires_grad=True)

    result = alpha.run_batched_refine(recall, value)
    result.value.square().mean().backward()

    assert value.grad is not None
    assert recall.state.recall.bank.grad is not None


def test_batched_refine_k3_bfloat16_forward_backward_is_finite() -> None:
    recall = _recall(topk=3).to(dtype=torch.bfloat16)
    recall.state.recall.set_bank_gradient_enabled(True)
    value = torch.randn(2, 3, 4, dtype=torch.bfloat16, requires_grad=True)

    result = alpha.run_batched_refine(
        recall,
        value,
        refine_policy=RefinePolicy.fixed(2, trace_level="routes"),
    )
    result.value.float().square().mean().backward()

    assert result.value.dtype == torch.bfloat16
    assert torch.isfinite(result.value).all()
    assert not recall.state.recall.query.weight.requires_grad
    assert recall.state.recall.query.weight.grad is None
    for tensor in (value, recall.state.recall.bank):
        assert tensor.grad is not None
        assert torch.isfinite(tensor.grad).all()
    for tensor in (recall.state.recall.group_bank, recall.state.recall.key_bank):
        if tensor.requires_grad:
            assert tensor.grad is not None
            assert torch.isfinite(tensor.grad).all()


def test_batched_refine_reuses_half_survival_independently_per_trajectory() -> None:
    torch.manual_seed(2017)
    recall = _recall(topk=2)
    recall.state.recall_activation = Half(stochastic=False)
    value = torch.randn(1, 3, 4)
    candidates = alpha.query_recall_branches(recall, value)
    policy = RefinePolicy.fixed(2, trace_level="routes")
    result = alpha.run_batched_refine(
        recall,
        value,
        candidates=candidates,
        refine_policy=policy,
    )
    serial = []
    for branch in range(2):
        groups = candidates.candidate_group_index[:, :, branch : branch + 1].expand(
            -1,
            -1,
            candidates.source_topk,
        )
        output, _, _ = recall.state(
            value,
            candidates.token_mask,
            refine_policy=policy,
            selected_groups=groups,
        )
        serial.append(output)
    torch.testing.assert_close(result.value, torch.stack(serial, dim=1))
    assert not torch.equal(result.value[:, 0], result.value[:, 1])


def test_batched_refine_uses_branch_local_adaptive_stopping() -> None:
    recall = _recall(topk=2)
    field = recall.state.recall
    value = torch.randn(1, 1, 4)
    with torch.no_grad():
        query = field._project_query(value)[0, 0]
        field.group_bank.copy_(-query)
        field.group_bank[0].copy_(2 * query)
        field.group_bank[1].copy_(query)
        field.bank.zero_()
        field.bank[2:4].fill_(0.25)
    candidates = alpha.query_recall_branches(recall, value)
    assert torch.equal(
        candidates.candidate_group_index[0, 0],
        torch.tensor([0, 1]),
    )
    policy = RefinePolicy.adaptive(
        max_steps=4,
        min_steps=1,
        scope="sample",
        absolute_tolerance=1e-7,
        relative_tolerance=0.0,
        trace_level="routes",
    )

    result = alpha.run_batched_refine(
        recall,
        value,
        candidates=candidates,
        refine_policy=policy,
    )

    steps = result.branch_diagnostics["recall_steps_committed"]
    assert steps.shape == (1, 2)
    assert steps[0, 0] == 1
    assert steps[0, 1] > steps[0, 0]


def test_branch_refine_policy_enforces_independent_budgets_and_permutation() -> None:
    torch.manual_seed(2021)
    recall = _recall(topk=3)
    value = torch.randn(1, 2, 4)
    candidates = alpha.query_recall_branches(recall, value)
    base = RefinePolicy.adaptive(
        max_steps=4,
        min_steps=0,
        scope="sample",
        absolute_tolerance=1e-7,
        relative_tolerance=0.0,
        trace_level="routes",
    )
    policy = alpha.BranchRefinePolicy(
        candidates,
        base,
        min_steps=torch.tensor([[0, 1, 4]]),
        max_steps=torch.tensor([[0, 1, 4]]),
    )

    result = alpha.run_batched_refine(
        recall,
        value,
        candidates=candidates,
        refine_policy=policy,
    )
    torch.testing.assert_close(
        result.branch_diagnostics["recall_steps_committed"],
        torch.tensor([[0, 1, 4]]),
    )
    torch.testing.assert_close(result.value[:, 0], value)
    assert result.branch_policy_fingerprint == policy.config_fingerprint
    assert result.global_diagnostics["batched_branch_capacity_k"].item() == 3
    assert result.global_diagnostics["batched_requested_k"].tolist() == [3]
    assert result.global_diagnostics["batched_eligible_k"].tolist() == [3]
    assert result.global_diagnostics["batched_active_partition_count"].tolist() == [1]
    assert result.global_diagnostics["batched_active_partition_mask"].tolist() == [
        [True]
    ]
    assert result.global_diagnostics[
        "batched_active_branch_count_by_partition"
    ].tolist() == [[3]]
    assert result.global_diagnostics["batched_attempted_k"].tolist() == [2]
    assert result.global_diagnostics["batched_committed_k"].tolist() == [2]
    assert result.global_diagnostics["batched_physical_branch_rows"].item() == 3

    order = torch.tensor([2, 0, 1])
    permuted_candidates = candidates.permute_branches(order)
    permuted = alpha.run_batched_refine(
        recall,
        value,
        candidates=permuted_candidates,
        refine_policy=policy,
    )
    torch.testing.assert_close(permuted.value, result.value.index_select(1, order))
    torch.testing.assert_close(
        permuted.branch_diagnostics["recall_steps_committed"],
        result.branch_diagnostics["recall_steps_committed"].index_select(1, order),
    )


def test_branch_refine_policy_rejects_foreign_candidate_input() -> None:
    recall = _recall(topk=2)
    value = torch.randn(1, 2, 4)
    candidates = alpha.query_recall_branches(recall, value)
    base = RefinePolicy.adaptive(
        max_steps=2,
        min_steps=1,
        absolute_tolerance=1e-6,
        relative_tolerance=0.0,
    )
    policy = alpha.BranchRefinePolicy(candidates, base)
    foreign = alpha.query_recall_branches(recall, value.clone())

    with pytest.raises(alpha.BatchedRefineContractError, match="does not belong"):
        alpha.run_batched_refine(
            recall,
            value.clone(),
            candidates=foreign,
            refine_policy=policy,
        )


def test_stopped_tokens_are_isolated_from_cross_token_state_operations() -> None:
    class CrossTokenOperation(torch.nn.Module):
        def forward(
            self,
            value: torch.Tensor,
            _validity: torch.Tensor,
            _exposed: torch.Tensor,
            _intervened: torch.Tensor,
        ) -> torch.Tensor:
            return value + value.sum(dim=1, keepdim=True)

    recall = _recall(topk=2)
    mask = torch.tensor([[True, False]])
    first = torch.tensor([[[1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0]]])
    second = first.clone()
    second[:, 1].mul_(1000)
    operation = CrossTokenOperation()

    first_result = recall.state._apply_refine_state_operation(first, mask, operation)
    second_result = recall.state._apply_refine_state_operation(second, mask, operation)

    torch.testing.assert_close(first_result[:, 0], second_result[:, 0])
    torch.testing.assert_close(first_result[:, 1], first[:, 1])
    torch.testing.assert_close(second_result[:, 1], second[:, 1])


def test_batched_refine_rejects_stale_or_foreign_candidates() -> None:
    recall = _recall(topk=2)
    value = torch.randn(1, 2, 4)
    candidates = alpha.query_recall_branches(recall, value)
    foreign = _recall(topk=2)

    with pytest.raises(alpha.BatchedRefineContractError, match="instance"):
        alpha.run_batched_refine(foreign, value, candidates=candidates)

    with torch.no_grad():
        recall.state.recall.bank.add_(1)
    with pytest.raises(alpha.BatchedRefineContractError, match="changed"):
        alpha.run_batched_refine(recall, value, candidates=candidates)

    recall = _recall(topk=2)
    candidates = alpha.query_recall_branches(recall, value)
    recall.state.recall.set_training_group_partitions(2, (0,))
    with pytest.raises(
        alpha.BatchedRefineContractError,
        match="layout|configuration",
    ):
        alpha.run_batched_refine(recall, value, candidates=candidates)


def test_batched_refine_candidates_bind_the_exact_unchanged_query_input() -> None:
    recall = _recall(topk=2)
    value = torch.randn(1, 2, 4)
    candidates = alpha.query_recall_branches(recall, value)

    with pytest.raises(alpha.BatchedRefineContractError, match="unchanged Tensor"):
        alpha.run_batched_refine(recall, value.clone(), candidates=candidates)
    value.add_(1)
    with pytest.raises(alpha.BatchedRefineContractError, match="unchanged Tensor"):
        alpha.run_batched_refine(recall, value, candidates=candidates)

    value = torch.randn(1, 2, 4)
    candidates = alpha.query_recall_branches(recall, value)
    candidates.candidate_group_index.add_(0)
    with pytest.raises(alpha.BatchedRefineContractError, match="bridge tensor changed"):
        alpha.run_batched_refine(recall, value, candidates=candidates)


def test_batched_refine_rejects_host_synchronized_early_break() -> None:
    recall = _recall(topk=2)
    value = torch.randn(1, 2, 4)
    policy = RefinePolicy.adaptive(
        max_steps=2,
        absolute_tolerance=1e-6,
        executor="early_break",
    )

    with pytest.raises(alpha.BatchedRefineContractError, match="static_masked"):
        alpha.run_batched_refine(recall, value, refine_policy=policy)


def test_query_recall_branches_supports_complete_composed_transitions() -> None:
    dense = Recall(dim=4, slots=8, routing="dense")
    with pytest.raises(alpha.BatchedRefineContractError, match="grouped"):
        alpha.query_recall_branches(dense, torch.randn(1, 4))

    composed = Recall(
        dim=4,
        slots=16,
        formula="arti/affine@1",
        routing="grouped",
        group_size=2,
        group_topk=2,
        key_dim=4,
        activation="none",
    )
    value = torch.randn(1, 3, 4)
    batch = alpha.query_recall_branches(composed, value)
    assert isinstance(batch, alpha.RecallFormulaBranchBatch)
    assert component_ref(batch).startswith("arti/recall-formula-branch-batch@sha256:")
    assert batch.schema_version == 6
    catalog = {entry["ref"]: entry for entry in arti.component_catalog()}
    assert catalog[component_ref(batch)]["config_schema_version"] == 6
    assert batch.value_composition == "product"
    assert batch.candidate_policy == "cross-bank-joint-factor-beam@1"
    assert batch.partition_coherence == "cross_bank"
    assert batch.max_k == 2
    assert batch.factor_count == 2
    assert batch.candidate_group_index.shape == (1, 3, 2, 2)
    assert batch.candidate_slot_index.shape == (1, 3, 2, 2, 2)
    assert batch.route_mass.shape == (1, 3, 2, 2)
    assert batch.candidate_log_score.shape == (1, 3, 2)
    assert batch.factor_candidate_rank is not None
    assert batch.factor_candidate_rank.shape == (1, 3, 2, 2)
    assert batch.factor_route_index is not None
    assert batch.flattened_execution_groups().shape == (2, 3, 2, 2)
    result = alpha.run_batched_refine(
        composed,
        value,
        candidates=batch,
        refine_policy=RefinePolicy.fixed(2, trace_level="routes"),
    )
    assert result.value.shape == (1, 2, 3, 4)
    permuted = batch.permute_branches(torch.tensor([1, 0]))
    permuted_result = alpha.run_batched_refine(
        composed,
        value,
        candidates=permuted,
        refine_policy=RefinePolicy.fixed(2, trace_level="routes"),
    )
    torch.testing.assert_close(permuted_result.value, result.value[:, [1, 0]])


def test_composed_per_bank_candidates_preserve_factor_partition_identity() -> None:
    recall = Recall(
        dim=4,
        slots=32,
        formula="arti/affine@1",
        routing="grouped",
        group_size=2,
        group_topk=8,
        key_dim=4,
        activation="none",
        routing_normalizer="per_bank",
    )
    recall.state.recall.configure_expert_routes(
        ("game", "animal"),
        ((0, 4), (4, 8)),
    )
    value = torch.randn(2, 3, 4)

    batch = alpha.query_recall_branches(
        recall,
        value,
        max_k=5,
        formula_beam_width=5,
    )

    assert component_ref(batch).startswith("arti/recall-formula-branch-batch@sha256:")
    assert batch.candidate_policy == "same-bank-joint-factor-beam@1"
    assert batch.partition_coherence == "same_bank"
    assert batch.partition_quota == (3, 2)
    assert batch.candidate_partition_index.shape == (2, 3, 5, 2)
    local_group = torch.remainder(batch.candidate_group_index, 8)
    assert torch.equal(
        batch.candidate_partition_index,
        (local_group >= 4).to(torch.long),
    )
    assert torch.all(
        batch.candidate_partition_index
        == batch.candidate_partition_index[..., :1]
    )
    permutation = torch.tensor([[4, 0, 3, 1, 2], [1, 4, 0, 2, 3]])
    permuted = batch.permute_branches(permutation)
    assert torch.equal(
        permuted.candidate_partition_index,
        (torch.remainder(permuted.candidate_group_index, 8) >= 4).to(torch.long),
    )


def test_same_bank_formula_quota_uses_complete_tuple_capacity() -> None:
    recall = Recall(
        dim=4,
        slots=16,
        formula="arti/affine@1",
        routing="grouped",
        group_size=2,
        group_topk=4,
        key_dim=4,
        activation="none",
        routing_normalizer="per_bank",
    )
    recall.state.recall.configure_expert_routes(
        ("enabled", "disabled"),
        ((0, 2), (2, 4)),
    )
    recall.state.recall.set_expert_weights((1.0, 0.0))
    value = torch.randn(1, 2, 4)

    candidates = alpha.query_recall_branches(
        recall,
        value,
        max_k=4,
        formula_beam_width=4,
        candidate_allocation="explicit_bank_quota",
        bank_quotas=(4, 0),
    )

    assert candidates.partition_quota == (4, 0)
    assert candidates.active_partition_mask().tolist() == [[True, False]]
    with pytest.raises(
        alpha.BatchedRefineContractError,
        match="quota exceeds its same-Bank Formula capacity",
    ):
        alpha.query_recall_branches(
            recall,
            value,
            max_k=5,
            formula_beam_width=5,
            candidate_allocation="explicit_bank_quota",
            bank_quotas=(5, 0),
        )


def test_composed_partition_coherence_distinguishes_cross_and_same_bank() -> None:
    recall = Recall(
        dim=4,
        slots=32,
        formula="arti/affine@1",
        routing="grouped",
        group_size=2,
        group_topk=8,
        key_dim=4,
        activation="none",
        routing_normalizer="per_bank",
    )
    recall.state.recall.configure_expert_routes(
        ("game", "animal"),
        ((0, 4), (4, 8)),
    )
    field = recall.state.recall
    field.factor_route_count = 2
    field.factor_route_names = ("scale", "shift")
    field.route_names = ("scale", "shift")
    field.factor_route_indices = (0, 1)
    field._factor_route_assignment = torch.eye(2)
    field._factor_route_index = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        field.query.weight.copy_(torch.eye(4))
        field.group_bank.zero_()
        field.group_bank[0, 0] = 10.0
        field.group_bank[1:4, 0] = -10.0
        field.group_bank[12, 0] = 10.0
        field.group_bank[13:16, 0] = -10.0
    value = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])

    cross = alpha.query_recall_branches(
        recall,
        value,
        max_k=2,
        formula_beam_width=2,
        partition_coherence="cross_bank",
        candidate_allocation="global_weighted_topk",
    )
    same = alpha.query_recall_branches(
        recall,
        value,
        max_k=2,
        formula_beam_width=2,
    )

    assert cross.candidate_policy == "cross-bank-joint-factor-beam@1"
    assert cross.partition_coherence == "cross_bank"
    assert cross.candidate_partition_index[0, 0, 0].tolist() == [0, 1]
    assert torch.any(
        cross.candidate_partition_index[..., 0]
        != cross.candidate_partition_index[..., 1]
    )
    assert same.candidate_policy == "same-bank-joint-factor-beam@1"
    assert same.partition_coherence == "same_bank"
    assert same.partition_quota == (1, 1)
    assert torch.all(
        same.candidate_partition_index == same.candidate_partition_index[..., :1]
    )


def test_same_bank_formula_candidates_match_partition_cartesian_oracle() -> None:
    torch.manual_seed(7421)
    recall = Recall(
        dim=4,
        slots=32,
        formula="arti/affine@1",
        routing="grouped",
        group_size=2,
        group_topk=8,
        key_dim=4,
        activation="none",
        routing_normalizer="per_bank",
    )
    recall.state.recall.configure_expert_routes(
        ("game", "animal"),
        ((0, 4), (4, 8)),
    )
    field = recall.state.recall
    field.factor_route_count = 2
    field.factor_route_names = ("scale", "shift")
    field.route_names = ("scale", "shift")
    field.factor_route_indices = (0, 1)
    field._factor_route_assignment = torch.eye(2)
    field._factor_route_index = torch.tensor([0, 1], dtype=torch.long)
    with torch.no_grad():
        field.query.weight.copy_(torch.eye(4))
        field.group_bank.copy_(torch.randn_like(field.group_bank))
    value = torch.tensor([[[0.7, -0.4, 0.2, 1.1]]])
    token_mask = torch.ones(1, 1, dtype=torch.bool)

    candidates = alpha.query_recall_branches(
        recall,
        value,
        max_k=4,
        formula_beam_width=4,
        candidate_allocation="explicit_bank_quota",
        bank_quotas=(2, 2),
    )
    route = field(value, token_mask).route.reshape(1, 1, 2, 8)[0, 0]
    expected_by_bank: list[list[tuple[float, tuple[int, int]]]] = []
    for start, stop in ((0, 4), (4, 8)):
        factor_groups = [
            torch.topk(route[factor, start:stop], 4).indices + start
            for factor in range(2)
        ]
        tuples = []
        for left, right in product(factor_groups[0].tolist(), factor_groups[1].tolist()):
            score = float(
                (
                    torch.log(
                        route[0, left].clamp_min(torch.finfo(route.dtype).tiny)
                    )
                    + torch.log(
                        route[1, right].clamp_min(torch.finfo(route.dtype).tiny)
                    )
                ).detach()
            )
            tuples.append((score, (left, right + 8)))
        expected_by_bank.append(sorted(tuples, key=lambda item: item[0], reverse=True)[:2])
    expected = [
        expected_by_bank[0][0],
        expected_by_bank[1][0],
        expected_by_bank[0][1],
        expected_by_bank[1][1],
    ]

    assert candidates.candidate_group_index[0, 0].tolist() == [
        list(groups) for _, groups in expected
    ]
    torch.testing.assert_close(
        candidates.candidate_log_score[0, 0],
        torch.tensor([score for score, _ in expected]),
    )
    assert candidates.active_branch_count_by_partition().tolist() == [[2, 2]]


def test_per_bank_candidate_rejects_route_weight_drift() -> None:
    recall = Recall(
        dim=4,
        slots=8,
        activation="none",
        routing="grouped",
        group_size=2,
        group_topk=4,
        key_dim=4,
        routing_normalizer="per_bank",
    )
    recall.state.recall.configure_expert_routes(
        ("a", "b"),
        ((0, 2), (2, 4)),
    )
    value = torch.randn(1, 2, 4)
    candidates = alpha.query_recall_branches(recall, value)

    recall.state.recall.set_expert_weights((1.0, 0.25))

    with pytest.raises(alpha.BatchedRefineContractError, match="configuration"):
        alpha.run_batched_refine(recall, value, candidates=candidates)


def test_per_bank_candidate_rejects_member_asset_identity_drift() -> None:
    recall = Recall(
        dim=4,
        slots=8,
        activation="none",
        routing="grouped",
        group_size=2,
        group_topk=4,
        key_dim=4,
        routing_normalizer="per_bank",
    )
    recall.state.recall.configure_expert_routes(
        ("a", "b"),
        ((0, 2), (2, 4)),
        member_fingerprints=("1" * 64, "2" * 64),
    )
    value = torch.randn(1, 2, 4)
    candidates = alpha.query_recall_branches(recall, value)

    recall.state.recall.configure_expert_routes(
        ("a", "b"),
        ((0, 2), (2, 4)),
        member_fingerprints=("1" * 64, "3" * 64),
    )

    with pytest.raises(
        alpha.BatchedRefineContractError,
        match="layout|configuration",
    ):
        alpha.run_batched_refine(recall, value, candidates=candidates)


@pytest.mark.parametrize("formula", ["arti/delta@1", "arti/affine@1"])
def test_recorded_candidate_context_matches_first_runtime_recall(
    formula: str,
) -> None:
    recall = Recall(
        dim=4,
        slots=16 if formula == "arti/affine@1" else 12,
        formula=formula,
        activation="none",
        routing="grouped",
        group_size=2,
        group_topk=2,
        key_dim=4,
    )
    value = torch.randn(2, 3, 4)
    candidates = alpha.query_recall_branches(recall, value)
    result = alpha.run_batched_refine(
        recall,
        value,
        candidates=candidates,
        refine_policy=RefinePolicy.fixed(1, trace_level="full"),
    )

    expected = candidates.candidate_context.permute(0, 2, 1, 3)
    torch.testing.assert_close(
        result.branch_diagnostics["recall_raw_context"],
        expected,
    )

    partitioned = _recall()
    partitioned.state.recall.set_training_group_partitions(2, (0,))
    with pytest.raises(alpha.BatchedRefineContractError, match="partition"):
        alpha.query_recall_branches(partitioned, torch.randn(1, 4))


@pytest.mark.parametrize("topology", [False, True])
def test_nonuniform_formula_topology_plan_is_branch_permutation_equivariant(
    topology: bool,
) -> None:
    recall = _recall(topk=3)
    value = torch.randn(1, 3, 4)
    operation = _formula_operation(batch=3, topology=topology)
    with torch.no_grad():
        operation._route_weights.zero_()
        operation._route_weights[0, ..., 0, 0] = 1
        operation._route_weights[0, ..., 1, 1] = 1
        operation._route_weights[1, ..., 0, 0] = 1
        operation._route_weights[1, ..., 1, 0] = 1
        operation._route_weights[2, ..., 0, 1] = 1
        operation._route_weights[2, ..., 1, 1] = 1
        operation._route_valid_mask[2].zero_()
        operation._route_fire_mask[2].zero_()
        operation._route_commit_mask[2].zero_()
    operation._bind_resident_route()
    plan = alpha.BatchedRefinePlan.compose(operation)
    candidates = alpha.query_recall_branches(recall, value, max_k=3)
    result = alpha.run_batched_refine(
        recall,
        value,
        candidates=candidates,
        plan=plan,
        refine_policy=RefinePolicy.fixed(1, trace_level="routes"),
    )

    order = torch.tensor([2, 0, 1])
    permuted = candidates.permute_branches(order)
    assert permuted.branch_origin_index.tolist() == [[2, 0, 1]]
    permuted_result = alpha.run_batched_refine(
        recall,
        value,
        candidates=permuted,
        plan=plan,
        refine_policy=RefinePolicy.fixed(1, trace_level="routes"),
    )

    torch.testing.assert_close(permuted_result.value, result.value[:, order])
    torch.testing.assert_close(permuted_result.delta, result.delta[:, order])
    for name in (
        "batched_formula_cells",
        "batched_formula_fire_count",
        "batched_formula_commit_count",
    ):
        torch.testing.assert_close(
            permuted_result.branch_diagnostics[name],
            result.branch_diagnostics[name][:, order],
        )
    for name in (
        "batched_formula_valid_trace",
        "batched_formula_fire_trace",
        "batched_formula_commit_trace",
    ):
        torch.testing.assert_close(
            permuted_result.branch_diagnostics[name],
            result.branch_diagnostics[name][:, order],
        )
    if topology:
        torch.testing.assert_close(
            permuted_result.branch_diagnostics[
                "batched_topology_permutation_trace"
            ],
            result.branch_diagnostics[
                "batched_topology_permutation_trace"
            ][:, order],
        )


def test_formula_topology_trace_records_outer_and_resident_steps() -> None:
    recall = _recall(topk=3)
    value = torch.randn(2, 3, 4)
    operation = _formula_operation(batch=6, topology=True, refine_steps=2)
    result = alpha.run_batched_refine(
        recall,
        value,
        plan=alpha.BatchedRefinePlan.compose(operation),
        refine_policy=RefinePolicy.fixed(2, trace_level="routes"),
    )

    valid = result.branch_diagnostics["batched_formula_valid_trace"]
    fire = result.branch_diagnostics["batched_formula_fire_trace"]
    commit = result.branch_diagnostics["batched_formula_commit_trace"]
    assert valid.shape == (2, 3, 2, 2, 1, 1)
    assert fire.shape == valid.shape
    assert commit.shape == valid.shape
    torch.testing.assert_close(
        result.branch_diagnostics["batched_formula_cells"],
        valid.sum(dim=(2, 3, 4, 5), dtype=torch.int64),
    )
    torch.testing.assert_close(
        result.branch_diagnostics["batched_formula_invoked_cells"],
        result.branch_diagnostics["batched_formula_declared_cells"],
    )
    torch.testing.assert_close(
        result.branch_diagnostics["batched_formula_effective_cells"],
        result.branch_diagnostics["batched_formula_invoked_cells"],
    )
    topology = result.branch_diagnostics["batched_topology_permutation_trace"]
    assert topology.shape == (2, 3, 2, 3)
    torch.testing.assert_close(
        topology,
        torch.tensor([2, 0, 1]).expand_as(topology),
    )
    assert result.global_diagnostics["batched_topology_active_count"].item() == 3
    assert result.branch_diagnostics["batched_topology_unfold_verified"].all()


def test_composed_query_keeps_cross_factor_joint_candidate() -> None:
    recall = Recall(
        dim=4,
        slots=24,
        formula="arti/affine@1",
        routing="grouped",
        group_size=2,
        group_topk=3,
        key_dim=4,
        activation="none",
    )
    field = recall.state.recall
    factor_groups = field.group_bank.shape[0] // 2
    with torch.no_grad():
        field.query.weight.zero_()
        field.query.weight[0, 0] = 1
        field.group_bank.fill_(-10)
        field.group_bank[:, 1:].zero_()
        field.group_bank[0, 0] = 4
        field.group_bank[1, 0] = 2
        field.group_bank[factor_groups, 0] = 4
        field.group_bank[factor_groups + 1, 0] = 3
        field.key_bank.zero_()

    batch = alpha.query_recall_branches(
        recall,
        torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]),
        max_k=2,
    )

    assert batch.factor_candidate_rank is not None
    ranks = batch.factor_candidate_rank[0, 0].tolist()
    assert ranks[0] == [0, 0]
    assert ranks[1] == [0, 1]
    assert ranks[1] != [1, 1]
    assert batch.candidate_log_score[0, 0, 0] > batch.candidate_log_score[0, 0, 1]


def test_composed_query_formula_beam_can_exceed_source_topk() -> None:
    recall = Recall(
        dim=4,
        slots=24,
        formula="arti/affine@1",
        routing="grouped",
        group_size=2,
        group_topk=3,
        key_dim=4,
        activation="none",
    )
    value = torch.randn(1, 2, 4)

    candidates = alpha.query_recall_branches(
        recall,
        value,
        formula_beam_width=7,
        max_k=7,
    )

    assert candidates.source_topk == 3
    assert candidates.formula_beam_width == 7
    assert candidates.max_k == 7
    assert candidates.factor_candidate_rank is not None
    assert candidates.factor_candidate_rank.shape == (1, 2, 7, 2)
    tuples = candidates.factor_candidate_rank[0, 0]
    assert len({tuple(item) for item in tuples.tolist()}) == 7
    assert int(tuples.max()) < candidates.source_topk

    result = alpha.run_batched_refine(
        recall,
        value,
        candidates=candidates,
        refine_policy=RefinePolicy.fixed(1, trace_level="routes"),
    )
    assert result.value.shape == (1, 7, 2, 4)


@pytest.mark.parametrize(
    "factors,candidates,output_k",
    [(2, 2, 2), (2, 3, 7), (2, 4, 3), (3, 3, 3)],
)
@pytest.mark.parametrize("case", ["random", "ties", "zeros"])
def test_joint_factor_beam_matches_cartesian_oracle(
    factors: int,
    candidates: int,
    output_k: int,
    case: str,
) -> None:
    generator = torch.Generator().manual_seed(7123 + factors * 10 + candidates)
    if case == "random":
        route_mass = torch.rand(2, 3, factors, candidates, generator=generator) + 0.01
    elif case == "ties":
        route_mass = torch.ones(2, 3, factors, candidates)
    else:
        route_mass = torch.rand(2, 3, factors, candidates, generator=generator)
        route_mass[..., -1] = 0

    paths, scores = _joint_factor_topk(route_mass, output_k)
    tiny = torch.finfo(route_mass.dtype).tiny
    log_mass = torch.log(route_mass.clamp_min(tiny))
    for batch in range(route_mass.shape[0]):
        for token in range(route_mass.shape[1]):
            oracle = []
            for ranks in product(range(candidates), repeat=factors):
                score = sum(
                    float(log_mass[batch, token, factor, rank])
                    for factor, rank in enumerate(ranks)
                )
                oracle.append((score, ranks))
            oracle.sort(key=lambda item: (-item[0], item[1]))
            expected = oracle[:output_k]
            assert paths[batch, token].tolist() == [list(item[1]) for item in expected]
            torch.testing.assert_close(
                scores[batch, token],
                torch.tensor([item[0] for item in expected]),
            )


class _VersionedParametricFormula(torch.nn.Module):
    recall_formula_contract = RecallFormulaContract(
        identity=RecallFormulaId.parse("tests/batched-parametric@1"),
        factors=(FactorSpec("content"),),
        composition="custom",
    )

    def __init__(self) -> None:
        super().__init__()
        self.gain = torch.nn.Parameter(torch.ones(()))
        self.mode = "add"

    def forward(self, state: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
        return state + self.gain * factors[..., 0, :]

    def recall_formula_config(self) -> dict[str, object]:
        return {"mode": self.mode}


register_formula(
    "tests/batched-parametric@1",
    factory=_VersionedParametricFormula,
    description="Batched Refine custom Formula provenance fixture",
)


@pytest.mark.parametrize(
    "formula",
    [_VersionedParametricFormula(), "tests/batched-parametric@1"],
)
def test_custom_formula_recall_is_runtime_only_at_artifact_boundary(
    tmp_path: object,
    formula: torch.nn.Module | str,
) -> None:
    recall = Recall(
        dim=4,
        slots=12,
        formula=formula,
        activation="none",
        routing="grouped",
        group_size=2,
        group_topk=2,
        key_dim=4,
    )
    with pytest.raises(ComponentRegistryError, match="runtime-only"):
        arti.save(recall, tmp_path / "custom-formula.st")


class _BuiltinIdentitySpoofFormula(torch.nn.Module):
    recall_formula_contract = RecallFormulaContract(
        identity=RecallFormulaId.parse("arti/affine@1"),
        factors=(FactorSpec("content"),),
        composition="custom",
    )

    def forward(self, state: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
        return state + factors[..., 0, :]


class _RegisteredProviderFormula(torch.nn.Module):
    recall_formula_contract = RecallFormulaContract(
        identity=RecallFormulaId.parse("tests/batched-provider-mismatch@1"),
        factors=(FactorSpec("content"),),
        composition="custom",
    )

    def forward(self, state: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
        return state + factors[..., 0, :]

    def recall_formula_config(self) -> dict[str, object]:
        return {}


class _MismatchedProviderFormula(torch.nn.Module):
    recall_formula_contract = _RegisteredProviderFormula.recall_formula_contract

    def forward(self, state: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
        return state - factors[..., 0, :]

    def recall_formula_config(self) -> dict[str, object]:
        return {}


register_formula(
    "tests/batched-provider-mismatch@1",
    factory=_RegisteredProviderFormula,
    description="Batched Refine provider mismatch fixture",
)


def test_formula_candidate_rejects_source_formula_mutation() -> None:
    recall = Recall(
        dim=4,
        slots=12,
        formula=_VersionedParametricFormula(),
        activation="none",
        routing="grouped",
        group_size=2,
        group_topk=2,
        key_dim=4,
    )
    value = torch.randn(1, 3, 4)
    candidates = alpha.query_recall_branches(recall, value)
    with torch.no_grad():
        recall.state.recall.formula.gain.add_(1)

    with pytest.raises(alpha.BatchedRefineContractError, match="execution tensors changed"):
        alpha.run_batched_refine(recall, value, candidates=candidates)


def test_formula_candidate_rejects_non_tensor_behavior_drift() -> None:
    recall = Recall(
        dim=4,
        slots=12,
        formula=_VersionedParametricFormula(),
        activation="none",
        routing="grouped",
        group_size=2,
        group_topk=2,
        key_dim=4,
    )
    value = torch.randn(1, 3, 4)
    candidates = alpha.query_recall_branches(recall, value)
    recall.state.recall.formula.mode = "subtract"

    with pytest.raises(alpha.BatchedRefineContractError, match="configuration changed"):
        alpha.run_batched_refine(recall, value, candidates=candidates)


def test_formula_candidate_rejects_builtin_identity_spoofing() -> None:
    recall = Recall(
        dim=4,
        slots=12,
        formula=_BuiltinIdentitySpoofFormula(),
        activation="none",
        routing="grouped",
        group_size=2,
        group_topk=2,
        key_dim=4,
    )

    with pytest.raises(alpha.BatchedRefineContractError, match="builtin arti identity"):
        alpha.query_recall_branches(recall, torch.randn(1, 3, 4))


def test_formula_candidate_rejects_registered_provider_mismatch() -> None:
    recall = Recall(
        dim=4,
        slots=12,
        formula=_MismatchedProviderFormula(),
        activation="none",
        routing="grouped",
        group_size=2,
        group_topk=2,
        key_dim=4,
    )

    with pytest.raises(alpha.BatchedRefineContractError, match="registered identity"):
        alpha.query_recall_branches(recall, torch.randn(1, 3, 4))


def test_runtime_candidate_and_result_identities_are_not_constructible() -> None:
    recall = _recall(topk=2)
    value = torch.randn(1, 3, 4)
    candidate = alpha.query_recall_branches(recall, value)
    result = alpha.run_batched_refine(recall, value, candidates=candidate)

    assert component_ref(result).startswith("arti/batched-refine-result@sha256:")
    catalog = {entry["ref"]: entry for entry in arti.component_catalog()}
    assert catalog[component_ref(candidate)]["constructible"] is False
    assert catalog[component_ref(result)]["constructible"] is False
    dependencies = set(arti.component_spec(result).dependencies)
    assert component_ref(candidate) in dependencies
    assert candidate.source_ref in dependencies
    assert candidate.formula_ref in dependencies
    assert candidate.source_ref.startswith("arti/recall@sha256:")
    assert "@sha256:" in candidate.formula_ref
    assert result.plan_ref in dependencies
    spec = arti.component_spec(result).to_dict()
    candidate_config = spec["config"]["candidate_config"]
    assert candidate_config["source_config_fingerprint"] == candidate.source_config_fingerprint
    assert candidate_config["formula_config_fingerprint"] == candidate.formula_config_fingerprint
    provenance = {
        "schema_version": 3,
        "components": [spec],
        "fingerprint": component_graph_fingerprint([spec]),
    }
    assert validate_component_provenance(provenance) == provenance

    tampered = dict(spec)
    tampered["dependencies"] = [
        dependency
        for dependency in spec["dependencies"]
        if dependency != candidate.formula_ref
    ]
    forged = {
        "schema_version": 3,
        "components": [tampered],
        "fingerprint": component_graph_fingerprint([tampered]),
    }
    with pytest.raises(ComponentCompatibilityError):
        validate_component_provenance(forged)
    with pytest.raises(ComponentRegistryError, match="runtime-only"):
        arti.resolve_component(component_ref(candidate))
    with pytest.raises(ComponentRegistryError, match="runtime-only"):
        arti.resolve_component(component_ref(result))


def test_runtime_candidate_and_result_cannot_be_saved(tmp_path: object) -> None:
    recall = _recall(topk=2)
    value = torch.randn(1, 3, 4)
    candidate = alpha.query_recall_branches(recall, value)
    result = alpha.run_batched_refine(recall, value, candidates=candidate)

    with pytest.raises(TypeError, match="torch.nn.Module"):
        arti.save(candidate, tmp_path / "candidate.arti.st")
    with pytest.raises(TypeError, match="torch.nn.Module"):
        arti.save(result, tmp_path / "result.arti.st")
    result_spec = arti.component_spec(result).to_dict()
    result_provenance = {
        "schema_version": 3,
        "components": [result_spec],
        "fingerprint": component_graph_fingerprint([result_spec]),
    }
    with pytest.raises(ComponentCompatibilityError, match="artifact_policy"):
        validate_component_provenance(
            result_provenance,
            artifact_scope=True,
        )


def test_plan_and_operation_provenance_reject_dependency_forgery() -> None:
    operation = alpha.BatchedRefineOperation(
        _formula_operation(batch=2, topology=False)
    )
    plan = alpha.BatchedRefinePlan.compose(operation.operation)

    for component in (operation, plan):
        root = arti.component_spec(component).to_dict()
        assert root["dependencies"]
        forged_root = dict(root)
        forged_root["dependencies"] = []
        forged = {
            "schema_version": 3,
            "components": [forged_root],
            "fingerprint": component_graph_fingerprint([forged_root]),
        }
        with pytest.raises(ComponentCompatibilityError):
            validate_component_provenance(forged)


@pytest.mark.parametrize("topology", [False, True])
def test_batched_operation_arti_st_restores_route_and_output(
    tmp_path: object,
    topology: bool,
) -> None:
    source = alpha.BatchedRefineOperation(
        _formula_operation(batch=2, topology=topology)
    )
    target = alpha.BatchedRefineOperation(
        _formula_operation(batch=2, topology=topology)
    )
    with torch.no_grad():
        target.operation._route_weights.zero_()
    value = torch.randn(2, 3, 4)
    mask = torch.ones(2, 3, dtype=torch.bool)
    expected = source(value, mask, mask, mask)
    assert component_ref(source.operation).split("@", maxsplit=1)[0] in {
        "arti/formula-resident-operation",
        "arti/topology-formula-resident-operation",
    }
    assert "@sha256:" in component_ref(source.operation)

    saved = arti.save(source, tmp_path / f"batched-{topology}.arti.st")
    arti.load(saved.weights_path, model=target)

    target.assert_unchanged()
    torch.testing.assert_close(target.operation.route.weights, source.operation.route.weights)
    torch.testing.assert_close(target(value, mask, mask, mask), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_query_recall_branches_stays_on_cuda() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    recall = _recall().to(device)
    value = torch.randn(2, 5, 4, device=device)

    batch = alpha.query_recall_branches(recall, value)

    assert batch.route_mass.is_cuda
    assert batch.candidate_group_index.is_cuda
    assert batch.expand_state(value).is_cuda
    result = alpha.run_batched_refine(recall, value, candidates=batch)
    assert result.value.is_cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_bfloat16_k3_ragged_policy_permutation_and_backward() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    recall = _recall(topk=3).to(device=device, dtype=torch.bfloat16)
    recall.state.recall.set_bank_gradient_enabled(True)
    value = torch.randn(
        2,
        3,
        4,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    candidates = alpha.query_recall_branches(
        recall,
        value,
        max_k=3,
        active_k=torch.tensor([1, 3], device=device),
    )
    base = RefinePolicy.adaptive(
        max_steps=3,
        min_steps=0,
        scope="sample",
        absolute_tolerance=1e-7,
        relative_tolerance=0.0,
        trace_level="routes",
    )
    policy = alpha.BranchRefinePolicy(
        candidates,
        base,
        min_steps=torch.tensor([[1, 0, 0], [1, 2, 3]], device=device),
        max_steps=torch.tensor([[1, 0, 0], [1, 2, 3]], device=device),
    )
    result = alpha.run_batched_refine(
        recall,
        value,
        candidates=candidates,
        refine_policy=policy,
        plan=alpha.BatchedRefinePlan.recall_only(
            execution_layout="packed_active"
        ),
    )

    order = torch.tensor([2, 0, 1], device=device)
    permuted = alpha.run_batched_refine(
        recall,
        value,
        candidates=candidates.permute_branches(order),
        refine_policy=policy,
        plan=alpha.BatchedRefinePlan.recall_only(
            execution_layout="packed_active"
        ),
    )
    torch.testing.assert_close(permuted.value, result.value.index_select(1, order))
    torch.testing.assert_close(permuted.delta, result.delta.index_select(1, order))
    result.value.float().square().mean().backward()

    assert result.value.is_cuda and result.value.dtype == torch.bfloat16
    assert value.grad is not None and torch.isfinite(value.grad).all()
    bank_grad = recall.state.recall.bank.grad
    assert bank_grad is not None and torch.isfinite(bank_grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cuda_keyed_rng_k3_route_half_dropout_replay_and_backward() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    recall = Recall(
        dim=4,
        slots=24,
        activation="half",
        routing="grouped",
        group_size=2,
        group_topk=3,
        key_dim=4,
        route_exploration=4.0,
        dropout=0.2,
    ).to(device=device, dtype=torch.bfloat16)
    recall.state.recall.set_bank_gradient_enabled(True)
    recall.train()
    value = torch.randn(
        2,
        5,
        4,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    rng_plan = alpha.ExecutionRNGPlan(
        seed=4049,
        run_nonce="cuda-keyed",
        stream_key="cuda.block-1.recall",
        sample_keys=("cuda-a", "cuda-b"),
    )
    candidates = alpha.query_recall_branches(
        recall,
        value,
        max_k=3,
        rng_plan=rng_plan,
    )
    policy = RefinePolicy.fixed(3, trace_level="routes")
    first = alpha.run_batched_refine(
        recall,
        value,
        candidates=candidates,
        refine_policy=policy,
        rng_plan=rng_plan,
    )
    replay = alpha.run_batched_refine(
        recall,
        value,
        candidates=candidates,
        refine_policy=policy,
        rng_plan=rng_plan,
    )
    order = torch.tensor([[2, 0, 1], [1, 2, 0]], device=device)
    permuted = alpha.run_batched_refine(
        recall,
        value,
        candidates=candidates.permute_branches(order),
        refine_policy=policy,
        rng_plan=rng_plan,
    )
    batch = torch.arange(value.shape[0], device=device).unsqueeze(1)

    assert torch.equal(first.value, replay.value)
    assert torch.equal(permuted.value, first.value[batch, order])
    first.value.float().square().mean().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert recall.state.recall.bank.grad is not None
    assert torch.isfinite(recall.state.recall.bank.grad).all()
