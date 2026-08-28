from __future__ import annotations

from dataclasses import replace
import pytest
import torch
from pathlib import Path

from arti import Recall, RefinePolicy, alpha, component_ref, component_spec
from arti.tensor_transaction import TensorTransactionContractError


HASH = "0" * 64
ABI = "3" * 64


def _recall() -> Recall:
    return Recall(
        dim=4,
        slots=12,
        formula="arti/delta@1",
        activation="none",
        routing="grouped",
        group_size=2,
        group_topk=3,
        key_dim=4,
    )


def _runtime(value: torch.Tensor) -> alpha.VolatileTensorRuntime:
    return alpha.VolatileTensorRuntime(
        {"state": value.detach().cpu().contiguous()},
        world_id="wide-search-world",
        store_instance_id="wide-search-store",
        abi_fingerprint=ABI,
        provenance_fingerprint=HASH,
    )


def _binding(
    store: alpha.VolatileTensorRuntime,
    snapshot: alpha.TensorSnapshot,
    executor: alpha.BatchedRefineExecutor,
) -> alpha.ExternalTensorBinding:
    return executor.bind(
        store,
        snapshot,
        "state",
        address_namespace="session",
        partition_id="main",
        logical_id="state",
        role="batched-refine-state",
        authority=alpha.TensorAuthority.READ_WRITE,
        state_schema_ref="arti/batched-refine-result@1",
        provenance_fingerprint=HASH,
    )


def _spec(
    result: alpha.BatchedRefineResult,
    executor: alpha.BatchedRefineExecutor,
    snapshot: alpha.TensorSnapshot,
    future: torch.Tensor,
    branch_ids: tuple[str, ...] = ("route-0", "route-1", "route-2"),
) -> alpha.BranchBatchSpecV2:
    return alpha.BranchBatchSpecV2.from_batched_refine(
        snapshot,
        executor,
        run_id="wide-search-1",
        branch_ids=branch_ids,
        input_fingerprint="6" * 64,
        future_tape_fingerprint=alpha.tensor_content_fingerprint(future),
        scorer_ref=alpha.FROZEN_MSE_SCORER_REF,
        scorer_config_fingerprint=alpha.frozen_mse_scorer_config_fingerprint(),
        allowed_write_keys=("state",),
        budgets=tuple(alpha.BranchBudget(1, 4) for _ in branch_ids),
    )


def _authority(
    result: alpha.BatchedRefineResult,
    store: alpha.VolatileTensorRuntime,
    snapshot: alpha.TensorSnapshot,
    future: torch.Tensor,
    branch_ids: tuple[str, ...] = ("route-0", "route-1", "route-2"),
) -> tuple[
    alpha.BatchedRefineExecutor,
    alpha.ExternalTensorBinding,
    alpha.BranchBatchSpecV2,
]:
    executor = alpha.BatchedRefineExecutor.from_result(result)
    return (
        executor,
        _binding(store, snapshot, executor),
        _spec(result, executor, snapshot, future, branch_ids),
    )


def _execution() -> tuple[torch.Tensor, alpha.BatchedRefineResult]:
    torch.manual_seed(2719)
    value = torch.randn(1, 3, 4)
    recall = _recall()
    with torch.no_grad():
        recall.state.recall.bank.normal_(std=0.5)
    result = alpha.run_batched_refine(
        recall,
        value,
        refine_policy=RefinePolicy.fixed(2, trace_level="routes"),
    )
    return value, result


def test_authority_manifest_rejects_mutated_candidate_lineage() -> None:
    _value, result = _execution()
    alpha.batched_refine_manifest_fingerprint(result)
    result.candidates.candidate_log_score.add_(1.0)
    with pytest.raises(alpha.BatchedRefineContractError, match="changed"):
        alpha.batched_refine_manifest_fingerprint(result)


def test_authority_manifest_rejects_mutated_requested_k_receipt() -> None:
    _value, result = _execution()
    result.candidates.requested_active_k.sub_(1)

    with pytest.raises(alpha.BatchedRefineContractError, match="changed"):
        alpha.batched_refine_manifest_fingerprint(result)


def test_executor_binds_explicit_execution_layout() -> None:
    value = torch.randn(1, 3, 4)
    result = alpha.run_batched_refine(
        _recall(),
        value,
        plan=alpha.BatchedRefinePlan.recall_only(
            execution_layout="packed_active"
        ),
    )
    executor = alpha.BatchedRefineExecutor.from_result(result)

    assert result.execution_layout == "packed_active"
    assert executor.execution_layout == "packed_active"
    assert component_spec(result).config["execution_layout"] == "packed_active"
    assert component_spec(executor).config["execution_layout"] == "packed_active"


def _formula_execution() -> tuple[torch.Tensor, alpha.BatchedRefineResult]:
    value = torch.randn(1, 3, 4)
    recall = _recall()
    program = alpha.FormulaFabricProgram(
        arena_capacity=3,
        feature_dim=4,
        steps=((alpha.FormulaInvocation(alpha.FormulaPrimitive.ADD, 2),),),
    )
    weights = torch.zeros(3, 1, 1, 2, 3)
    weights[..., 0, 0] = 1
    weights[..., 1, 1] = 1
    enabled = torch.ones(3, 1, 1, dtype=torch.bool)
    operation = alpha.FormulaResidentOperation(
        alpha.FormulaFabricCompute(
            alpha.FormulaFabric(program),
            active_count=3,
        ),
        alpha.FormulaRoutePlan(weights, enabled, enabled, enabled),
    )
    result = alpha.run_batched_refine(
        recall,
        value,
        refine_policy=RefinePolicy.fixed(2, trace_level="routes"),
        plan=alpha.BatchedRefinePlan.compose(operation),
    )
    return value, result


def _topology_execution() -> tuple[torch.Tensor, alpha.BatchedRefineResult]:
    value = torch.randn(1, 3, 4)
    recall = _recall()
    program = alpha.FormulaFabricProgram(
        arena_capacity=3,
        feature_dim=4,
        steps=((alpha.FormulaInvocation(alpha.FormulaPrimitive.ADD, 2),),),
    )
    weights = torch.zeros(3, 1, 1, 2, 3)
    weights[..., 0, 0] = 1
    weights[..., 1, 1] = 1
    enabled = torch.ones(3, 1, 1, dtype=torch.bool)
    fold, unfold = alpha.ReversibleTopology(
        active_count=3,
        policy=alpha.FixedTopologyPolicy(order=[2, 0, 1]),
    ).operations()
    operation = alpha.TopologyFormulaResidentOperation(
        fold,
        unfold,
        alpha.FormulaFabricCompute(
            alpha.FormulaFabric(program),
            active_count=3,
        ),
        alpha.FormulaRoutePlan(weights, enabled, enabled, enabled),
    )
    result = alpha.run_batched_refine(
        recall,
        value,
        refine_policy=RefinePolicy.fixed(2, trace_level="routes"),
        plan=alpha.BatchedRefinePlan.compose(operation),
    )
    return value, result


def test_k3_batched_refine_scores_and_publishes_only_the_frozen_winner() -> None:
    value, result = _execution()
    future = result.value[:, 1].detach().cpu().contiguous()
    store = _runtime(value)
    snapshot = store.snapshot()
    executor, binding, spec = _authority(result, store, snapshot, future)

    assert executor.execution_context.mode == "deterministic"
    assert executor.execution_context.algorithm == "none"
    assert spec.rng_fingerprint == executor.execution_context.fingerprint
    assert component_ref(executor.execution_context) == (
        "arti/execution-context-receipt@3"
    )

    proposals = alpha.stage_batched_refine_result(
        result,
        executor,
        spec,
        (binding, binding, binding),
    )
    assert len(proposals) == 3
    assert len({item.candidate_fingerprint for item in proposals}) == 3
    assert all(item.delta.shape == value.shape for item in proposals)
    assert all(len(item.delta_fingerprint) == 64 for item in proposals)
    harness = alpha.BranchBatchHarness(store, snapshot, spec)
    for proposal in proposals:
        harness.propose(proposal)
    assert store.snapshot().root_id == snapshot.root_id

    score = alpha.score_batched_refine_proposals(proposals, future, spec)
    assert score.scores[1] == 0.0
    receipt = harness.decide(score, idempotency_key="publish-wide-winner")

    assert receipt.status is alpha.BranchRunStatus.COMMITTED
    assert receipt._runtime_contract_ref == "arti/branch-run-receipt@2"
    assert receipt.winner_branch_id == "route-1"
    assert len(receipt.rollback_receipts) == 2
    torch.testing.assert_close(store.read(store.snapshot(), "state").value, future)
    assert harness.decide(score, idempotency_key="publish-wide-winner") is receipt


def test_keyed_execution_context_is_bound_to_the_rng_plan() -> None:
    value = torch.randn(1, 4, 4)
    recall = Recall(
        dim=4,
        slots=12,
        activation="half",
        routing="grouped",
        group_size=2,
        group_topk=3,
        key_dim=4,
    )
    rng_plan = alpha.ExecutionRNGPlan(
        seed=441,
        run_nonce="authority-keyed",
        stream_key="authority.block-1.recall",
        sample_keys=("sample-0",),
    )
    result = alpha.run_batched_refine(recall, value, rng_plan=rng_plan)
    executor = alpha.BatchedRefineExecutor.from_result(result)

    assert executor.execution_context.mode == "keyed"
    assert (
        executor.execution_context.algorithm
        == "sha256-seeded-torch-generator@2"
    )
    assert executor.execution_context.execution_rng_fingerprint == rng_plan.fingerprint
    assert executor.execution_context.execution_rng_stream_key == rng_plan.stream_key
    assert executor.execution_context.consumed_domains == ("half-survival",)
    executor.assert_matches(result)


def test_formula_work_receipts_come_from_the_composed_execution() -> None:
    value, result = _formula_execution()
    future = result.value[:, 0].detach().cpu().contiguous()
    store = _runtime(value)
    snapshot = store.snapshot()
    executor, binding, spec = _authority(result, store, snapshot, future)

    proposals = alpha.stage_batched_refine_result(
        result,
        executor,
        spec,
        (binding,) * 3,
    )

    assert {proposal.work.formula_cells for proposal in proposals} == {2}
    assert {proposal.work.fire_count for proposal in proposals} == {2}
    assert {proposal.work.commit_count for proposal in proposals} == {2}
    assert {proposal.work.route_applications for proposal in proposals} == {2}


def test_topology_result_enters_the_complete_authority_commit_path() -> None:
    value, result = _topology_execution()
    future = result.value[:, 2].detach().cpu().contiguous()
    store = _runtime(value)
    snapshot = store.snapshot()
    executor, binding, spec = _authority(result, store, snapshot, future)

    assert result.topology_refs == (
        "arti/fold@2",
        "arti/reversible-topology@1",
        "arti/unfold@2",
    )
    assert len(result.topology_contract_fingerprints) == 2
    proposals = alpha.stage_batched_refine_result(
        result,
        executor,
        spec,
        (binding,) * 3,
    )
    harness = alpha.BranchBatchHarness(store, snapshot, spec)
    for proposal in proposals:
        harness.propose(proposal)
    score = alpha.score_batched_refine_proposals(proposals, future, spec)
    receipt = harness.decide(score, idempotency_key="publish-topology-winner")

    assert receipt.status is alpha.BranchRunStatus.COMMITTED
    assert receipt.winner_branch_id == "route-2"
    torch.testing.assert_close(store.read(store.snapshot(), "state").value, future)


def test_topology_contract_fingerprint_drift_rejects_executor_match() -> None:
    _value, result = _topology_execution()
    executor = alpha.BatchedRefineExecutor.from_result(result)
    object.__setattr__(result, "topology_contract_fingerprints", ("f" * 64,) * 2)

    with pytest.raises(
        TensorTransactionContractError,
        match="does not match",
    ):
        executor.assert_matches(result)


def test_host_can_atomically_publish_a_k3_convex_mixture() -> None:
    value, result = _execution()
    future = result.value[:, 0].detach().cpu().contiguous()
    store = _runtime(value)
    snapshot = store.snapshot()
    executor, binding, spec = _authority(result, store, snapshot, future)
    proposals = alpha.stage_batched_refine_result(
        result,
        executor,
        spec,
        (binding,) * 3,
    )
    harness = alpha.BranchBatchHarness(store, snapshot, spec)
    for proposal in proposals:
        harness.propose(proposal)
    weights = (0.2, 0.3, 0.5)
    receipt = harness.mix(weights, idempotency_key="publish-wide-mixture")
    expected = sum(
        proposal.proposal.value * weight
        for proposal, weight in zip(proposals, weights, strict=True)
    )
    assert receipt.status is alpha.BranchRunStatus.COMMITTED
    assert receipt._runtime_contract_ref == "arti/branch-run-receipt@2"
    assert receipt.decision_kind == "mixture"
    assert receipt.winner_branch_id is None
    assert len(receipt.rollback_receipts) == 3
    torch.testing.assert_close(store.read(store.snapshot(), "state").value, expected)
    assert harness.mix(weights, idempotency_key="publish-wide-mixture") is receipt


def test_k1_degenerates_and_k3_discard_keeps_parent_immutable() -> None:
    value, result = _execution()
    future = result.value[:, 0].detach().cpu().contiguous()
    store = _runtime(value)
    snapshot = store.snapshot()
    executor, binding, spec = _authority(result, store, snapshot, future)
    proposals = alpha.stage_batched_refine_result(
        result,
        executor,
        spec,
        (binding,) * 3,
    )
    harness = alpha.BranchBatchHarness(store, snapshot, spec)
    for proposal in proposals:
        harness.propose(proposal)
    receipt = harness.discard(idempotency_key="discard-wide-search")
    assert receipt.status is alpha.BranchRunStatus.DISCARDED
    assert receipt._runtime_contract_ref == "arti/branch-run-receipt@2"
    assert len(receipt.rollback_receipts) == 3
    assert store.snapshot().root_id == snapshot.root_id

    k1 = alpha.query_recall_branches(_recall(), value, max_k=1)
    assert k1.max_k == 1


def test_authority_stages_only_active_ragged_branches() -> None:
    value = torch.randn(1, 3, 4)
    result = alpha.run_batched_refine(
        _recall(),
        value,
        max_k=3,
        active_k=2,
        refine_policy=RefinePolicy.fixed(1, trace_level="routes"),
    )
    future = result.value[:, 0].detach().cpu().contiguous()
    store = _runtime(value)
    snapshot = store.snapshot()
    executor, binding, spec = _authority(
        result,
        store,
        snapshot,
        future,
        ("route-0", "route-1"),
    )

    proposals = alpha.stage_batched_refine_result(
        result,
        executor,
        spec,
        (binding, binding),
    )

    assert len(proposals) == 2
    assert [proposal.branch_id for proposal in proposals] == ["route-0", "route-1"]


def test_authority_uniform_permutation_preserves_canonical_branch_identity() -> None:
    value = torch.randn(2, 3, 4)
    recall = _recall()
    candidates = alpha.query_recall_branches(recall, value, max_k=3)
    candidates = candidates.permute_branches(torch.tensor([2, 0, 1]))
    result = alpha.run_batched_refine(
        recall,
        value,
        candidates=candidates,
        refine_policy=RefinePolicy.fixed(1, trace_level="routes"),
    )
    future = result.value[:, 0].detach().cpu().contiguous()
    store = _runtime(value)
    snapshot = store.snapshot()
    executor, binding, spec = _authority(result, store, snapshot, future)

    proposals = alpha.stage_batched_refine_result(
        result,
        executor,
        spec,
        (binding,) * 3,
    )

    assert [proposal.branch_id for proposal in proposals] == [
        "route-0",
        "route-1",
        "route-2",
    ]
    physical_by_origin = {
        int(origin): physical
        for physical, origin in enumerate(candidates.branch_origin_index[0].tolist())
    }
    for origin, proposal in enumerate(proposals):
        torch.testing.assert_close(
            proposal.proposal.value,
            result.value[:, physical_by_origin[origin]].detach().cpu(),
        )


def test_authority_preserves_real_zero_step_receipts() -> None:
    value = torch.randn(1, 3, 4)
    result = alpha.run_batched_refine(
        _recall(),
        value,
        max_k=2,
        refine_policy=RefinePolicy.fixed(0, trace_level="routes"),
    )
    future = value.detach().cpu().contiguous()
    store = _runtime(value)
    snapshot = store.snapshot()
    executor, binding, spec = _authority(
        result,
        store,
        snapshot,
        future,
        ("route-0", "route-1"),
    )
    spec = replace(
        spec,
        budgets=(alpha.BranchBudget(0, 0), alpha.BranchBudget(0, 0)),
    )

    proposals = alpha.stage_batched_refine_result(
        result,
        executor,
        spec,
        (binding, binding),
    )

    for proposal in proposals:
        assert proposal.work.actual_steps == 0
        assert proposal.work.stop_reason == "max-steps"
        assert proposal.work.operation_count == 0
        torch.testing.assert_close(proposal.proposal.value, value)


def test_authority_receipt_separates_branch_and_kernel_depth() -> None:
    value = torch.randn(1, 3, 4)
    recall = _recall()
    candidates = alpha.query_recall_branches(recall, value, max_k=3)
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
        min_steps=torch.tensor([[1, 2, 4]]),
        max_steps=torch.tensor([[1, 2, 4]]),
    )
    result = alpha.run_batched_refine(
        recall,
        value,
        candidates=candidates,
        refine_policy=policy,
    )
    future = result.value[:, 0].detach().cpu().contiguous()
    store = _runtime(value)
    snapshot = store.snapshot()
    executor, binding, spec = _authority(result, store, snapshot, future)
    spec = replace(
        spec,
        budgets=tuple(alpha.BranchBudget(step, step) for step in (1, 2, 4)),
    )

    proposals = alpha.stage_batched_refine_result(
        result,
        executor,
        spec,
        (binding,) * 3,
    )

    assert [proposal.work.actual_steps for proposal in proposals] == [1, 2, 4]
    assert [proposal.work.outer_kernel_steps for proposal in proposals] == [4, 4, 4]
    assert [proposal.work.logical_steps_committed for proposal in proposals] == [
        3,
        6,
        12,
    ]
    assert all(proposal.work.resident_operation_steps == 0 for proposal in proposals)
    assert all(
        proposal.work.measurement_kind == "declared-estimate"
        for proposal in proposals
    )


def test_authority_canonicalizes_batch_dependent_branch_origins() -> None:
    value = torch.randn(2, 3, 4)
    recall = _recall()
    candidates = alpha.query_recall_branches(recall, value, max_k=3)
    order = torch.tensor([2, 0, 1])
    mixed_origin = candidates.branch_origin_index.clone()
    mixed_origin[1] = mixed_origin[1].index_select(0, order)
    candidate_values = [
        candidates.candidate_group_index,
        candidates.candidate_partition_index,
        candidates.candidate_slot_index,
        candidates.candidate_slot_weight,
        candidates.candidate_context,
        candidates.route_mass,
        candidates.selection_weight,
        candidates.candidate_log_score,
        candidates.candidate_mask,
        candidates.branch_mask,
        candidates.token_mask,
        mixed_origin,
        candidates.active_k,
        candidates.requested_active_k,
    ]
    mixed = type(candidates)(
        **{
            **candidates.__dict__,
            "branch_origin_index": mixed_origin,
            "candidate_tensor_tokens": tuple(id(tensor) for tensor in candidate_values),
            "candidate_tensor_versions": tuple(tensor._version for tensor in candidate_values),
        }
    )
    result = alpha.run_batched_refine(
        recall,
        value,
        candidates=mixed,
        refine_policy=RefinePolicy.fixed(1, trace_level="routes"),
    )
    future = result.value[:, 0].detach().cpu().contiguous()
    store = _runtime(value)
    snapshot = store.snapshot()
    executor, binding, spec = _authority(result, store, snapshot, future)

    proposals = alpha.stage_batched_refine_result(
        result,
        executor,
        spec,
        (binding,) * 3,
    )

    assert [proposal.branch_id for proposal in proposals] == [
        "route-0",
        "route-1",
        "route-2",
    ]
    for origin, proposal in enumerate(proposals):
        expected = torch.stack(
            [
                result.value[batch, int((mixed_origin[batch] == origin).nonzero()[0])]
                for batch in range(value.shape[0])
            ]
        ).detach().cpu()
        torch.testing.assert_close(proposal.proposal.value, expected)


def test_authority_accepts_sample_local_ragged_active_k() -> None:
    value = torch.randn(3, 3, 4)
    result = alpha.run_batched_refine(
        _recall(),
        value,
        max_k=3,
        active_k=torch.tensor([1, 3, 2]),
        refine_policy=RefinePolicy.fixed(1, trace_level="routes"),
    )
    future = result.value[:, 0].detach().cpu().contiguous()
    store = _runtime(value)
    snapshot = store.snapshot()
    executor, binding, spec = _authority(result, store, snapshot, future)

    proposals = alpha.stage_batched_refine_result(
        result,
        executor,
        spec,
        (binding,) * 3,
    )

    assert len(proposals) == 3
    torch.testing.assert_close(proposals[1].proposal.value[0], value[0])
    torch.testing.assert_close(proposals[2].proposal.value[0], value[0])
    torch.testing.assert_close(proposals[2].proposal.value[2], value[2])
    assert [proposal.work.logical_steps_committed for proposal in proposals] == [
        3,
        2,
        1,
    ]
    assert all(proposal.work.logical_step_unit == "sample-step" for proposal in proposals)


def test_score_and_authority_bindings_fail_closed() -> None:
    value, result = _execution()
    future = result.value[:, 0].detach().cpu().contiguous()
    store = _runtime(value)
    snapshot = store.snapshot()
    executor, binding, spec = _authority(result, store, snapshot, future)
    proposals = alpha.stage_batched_refine_result(
        result,
        executor,
        spec,
        (binding,) * 3,
    )
    with pytest.raises(TensorTransactionContractError, match="future tensor"):
        alpha.score_batched_refine_proposals(proposals, future + 1, spec)
    spoofed_scorer = replace(spec, scorer_config_fingerprint="8" * 64)
    with pytest.raises(TensorTransactionContractError, match="canonical frozen MSE"):
        alpha.score_batched_refine_proposals(proposals, future, spoofed_scorer)
    with pytest.raises(TensorTransactionContractError, match="must come from"):
        alpha.BranchScoreBatchReceipt(
            spec_fingerprint=spec.fingerprint,
            branch_ids=spec.branch_ids,
            overlay_fingerprints=(HASH,) * 3,
            candidate_fingerprints=(HASH,) * 3,
            future_fingerprint=spec.future_tape_fingerprint,
            scorer_ref=spec.scorer_ref,
            scorer_config_fingerprint=spec.scorer_config_fingerprint,
            tie_policy=spec.tie_policy,
            scores=(0.0, 1.0, 2.0),
            _factory_token=object(),
        )


def test_executor_identity_rejects_cross_result_and_tampered_spec() -> None:
    value, result = _execution()
    _other_value, other_result = _execution()
    with torch.no_grad():
        other_result.value.add_(0.25)
    store = _runtime(value)
    snapshot = store.snapshot()
    future = result.value[:, 0].detach().cpu().contiguous()
    executor, binding, spec = _authority(result, store, snapshot, future)

    with pytest.raises(
        (alpha.BatchedRefineContractError, TensorTransactionContractError),
        match="changed|does not match",
    ):
        alpha.stage_batched_refine_result(
            other_result,
            executor,
            spec,
            (binding,) * 3,
        )

    tampered = replace(spec, program_fingerprint="f" * 64)
    with pytest.raises(TensorTransactionContractError, match="does not match"):
        alpha.stage_batched_refine_result(
            result,
            executor,
            tampered,
            (binding,) * 3,
        )


def test_executor_identity_rejects_self_reported_binding_hashes() -> None:
    value, result = _execution()
    future = result.value[:, 0].detach().cpu().contiguous()
    store = _runtime(value)
    snapshot = store.snapshot()
    executor, _binding_value, spec = _authority(result, store, snapshot, future)
    forged = alpha.bind_external_tensor(
        store,
        snapshot,
        "state",
        address_namespace="session",
        partition_id="main",
        logical_id="state",
        role="batched-refine-state",
        authority=alpha.TensorAuthority.READ_WRITE,
        component_ref="arti/batched-refine@1",
        component_config_fingerprint="e" * 64,
        state_schema_ref="arti/batched-refine-result@1",
        producer_state_fingerprint="d" * 64,
        provenance_fingerprint=HASH,
    ).binding

    with pytest.raises(TensorTransactionContractError, match="executor identity"):
        alpha.stage_batched_refine_result(
            result,
            executor,
            spec,
            (forged,) * 3,
        )


def test_runtime_proposals_and_scores_cannot_be_saved(tmp_path: Path) -> None:
    value, result = _execution()
    future = result.value[:, 0].detach().cpu().contiguous()
    store = _runtime(value)
    snapshot = store.snapshot()
    executor, binding, spec = _authority(result, store, snapshot, future)
    proposals = alpha.stage_batched_refine_result(
        result,
        executor,
        spec,
        (binding,) * 3,
    )
    score = alpha.score_batched_refine_proposals(proposals, future, spec)

    with pytest.raises(TypeError, match="torch.nn.Module"):
        import arti

        arti.save(proposals[0], tmp_path / "proposal.arti.st")
    with pytest.raises(TypeError, match="torch.nn.Module"):
        arti.save(score, tmp_path / "score.arti.st")


def test_result_forgery_and_post_execution_mutation_fail_closed() -> None:
    value, result = _execution()
    with pytest.raises(alpha.BatchedRefineContractError, match="must come from"):
        alpha.BatchedRefineResult(
            candidates=result.candidates,
            value=result.value,
            delta=result.delta,
            branch_diagnostics=result.branch_diagnostics,
            global_diagnostics=result.global_diagnostics,
            _factory_token=object(),
        )

    future = result.value[:, 0].detach().cpu().contiguous()
    store = _runtime(value)
    snapshot = store.snapshot()
    executor, binding, spec = _authority(result, store, snapshot, future)
    with torch.no_grad():
        result.value.add_(1)
    with pytest.raises(alpha.BatchedRefineContractError, match="changed before staging"):
        alpha.stage_batched_refine_result(
            result,
            executor,
            spec,
            (binding,) * 3,
        )


def test_post_execution_diagnostic_mutation_fails_closed() -> None:
    value, result = _execution()
    future = result.value[:, 0].detach().cpu().contiguous()
    store = _runtime(value)
    snapshot = store.snapshot()
    executor, binding, spec = _authority(result, store, snapshot, future)
    diagnostic = next(iter(result.branch_diagnostics.values()))
    diagnostic.add_(1)

    with pytest.raises(alpha.BatchedRefineContractError, match="diagnostics changed"):
        alpha.stage_batched_refine_result(
            result,
            executor,
            spec,
            (binding,) * 3,
        )


def test_k3_conflict_never_publishes_a_second_root() -> None:
    value, result = _execution()
    future = result.value[:, 2].detach().cpu().contiguous()
    store = _runtime(value)
    snapshot = store.snapshot()
    executor, binding, spec = _authority(result, store, snapshot, future)
    proposals = alpha.stage_batched_refine_result(
        result,
        executor,
        spec,
        (binding,) * 3,
    )
    harness = alpha.BranchBatchHarness(store, snapshot, spec)
    for proposal in proposals:
        harness.propose(proposal)
    score = alpha.score_batched_refine_proposals(proposals, future, spec)

    outside = store.begin(snapshot, transaction_id="outside", branch_id="outside")
    outside.stage(
        "state",
        torch.full_like(value, 9.0),
        expected_version=1,
        provenance_fingerprint=HASH,
    )
    outside.commit(idempotency_key="outside")
    receipt = harness.decide(score, idempotency_key="stale-wide-winner")
    assert receipt.status is alpha.BranchRunStatus.CONFLICTED
    torch.testing.assert_close(
        store.read(store.snapshot(), "state").value,
        torch.full_like(value, 9.0),
    )


def test_committed_k3_winner_survives_fresh_runtime_reload(tmp_path: Path) -> None:
    value, result = _execution()
    future = result.value[:, 2].detach().cpu().contiguous()
    store = _runtime(value)
    snapshot = store.snapshot()
    executor, binding, spec = _authority(result, store, snapshot, future)
    proposals = alpha.stage_batched_refine_result(
        result,
        executor,
        spec,
        (binding,) * 3,
    )
    harness = alpha.BranchBatchHarness(store, snapshot, spec)
    for proposal in proposals:
        harness.propose(proposal)
    score = alpha.score_batched_refine_proposals(proposals, future, spec)
    receipt = harness.decide(score, idempotency_key="persist-wide-winner")
    assert receipt.winner_branch_id == "route-2"

    committed = store.snapshot()
    path = tmp_path / "wide.runtime.arti.st"
    saved = alpha.save_runtime_checkpoint(store, committed, path)
    restored = alpha.load_runtime_checkpoint(path, expected_abi_fingerprint=ABI)

    assert restored.snapshot.root_fingerprint == committed.root_fingerprint
    assert restored.artifact_sha256 == saved.artifact_sha256
    torch.testing.assert_close(
        restored.runtime.read(restored.snapshot, "state").value,
        future,
    )
    assert restored.runtime.committed_receipt("persist-wide-winner") == receipt.commit_receipt
