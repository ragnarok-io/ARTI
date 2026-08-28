from __future__ import annotations

import pytest
import torch

from arti import alpha
from arti.tensor_transaction import TensorTransactionContractError


HASH = "0" * 64
CONFIG = "1" * 64
STATE = "2" * 64
ABI = "3" * 64


def program() -> alpha.FormulaFabricProgram:
    return alpha.FormulaFabricProgram(
        arena_capacity=3,
        feature_dim=1,
        steps=((alpha.FormulaInvocation(alpha.FormulaPrimitive.ADD, 2),),),
    )


def route() -> alpha.FormulaRoutePlan:
    weights = torch.zeros(1, 1, 1, 2, 3)
    weights[..., 0, 0] = 1
    weights[..., 1, 1] = 1
    enabled = torch.ones(1, 1, 1, dtype=torch.bool)
    return alpha.FormulaRoutePlan(weights, enabled, enabled, enabled)


def workspace() -> alpha.ActiveWorkspace:
    value = torch.tensor([[[2.0], [3.0], [0.0]]])
    support = torch.ones(1, 3, dtype=torch.bool)
    return alpha.ActiveWorkspace(value, support, support, support)


def runtime() -> alpha.VolatileTensorRuntime:
    return alpha.VolatileTensorRuntime(
        {"state": workspace().value},
        world_id="formula-branch-world",
        store_instance_id="formula-branch-store",
        abi_fingerprint=ABI,
        provenance_fingerprint=HASH,
    )


def binding(
    store: alpha.VolatileTensorRuntime,
    snapshot: alpha.TensorSnapshot,
) -> alpha.BoundTensorRead:
    return alpha.bind_external_tensor(
        store,
        snapshot,
        "state",
        address_namespace="session",
        partition_id="main",
        logical_id="formula-workspace",
        role="formula-state",
        authority=alpha.TensorAuthority.READ_WRITE,
        component_ref="arti/formula-fabric-compute@1",
        component_config_fingerprint=CONFIG,
        state_schema_ref="arti/formula-workspace@1",
        producer_state_fingerprint=STATE,
        provenance_fingerprint=HASH,
    )


def proposal(
    bound: alpha.BoundTensorRead,
    value: torch.Tensor,
) -> alpha.ExternalTensorProposal:
    return alpha.ExternalTensorProposal(
        bound.binding,
        value,
        producer_ref="arti/formula-fabric-compute@1",
        producer_config_fingerprint=CONFIG,
        producer_state_fingerprint=STATE,
    )


def setup():
    store = runtime()
    snapshot = store.snapshot()
    source = workspace()
    fixed_route = route()
    compute = alpha.FormulaFabricCompute(
        alpha.FormulaCommitBlend(alpha.FormulaFabric(program())),
        active_count=3,
    )
    future = torch.tensor([[[2.0], [3.0], [5.0]]])
    spec = alpha.BranchBatchSpec.from_snapshot(
        snapshot,
        run_id="formula-k2",
        branch_ids=("left", "right"),
        executor_ref="arti/formula-fabric-compute@1",
        program_fingerprint=program().fingerprint,
        route_fingerprint=alpha.formula_route_fingerprint(fixed_route),
        input_fingerprint=alpha.workspace_fingerprint(source),
        rng_fingerprint=alpha.deterministic_formula_rng_fingerprint(),
        future_tape_fingerprint=alpha.tensor_content_fingerprint(future),
        budgets=(alpha.BranchBudget(2, 2), alpha.BranchBudget(2, 2)),
    )
    return store, snapshot, source, fixed_route, compute, future, spec


def test_fixed_formula_execution_derives_real_trace_and_step_lineage() -> None:
    _store, _snapshot, source, fixed_route, compute, _future, spec = setup()
    result = alpha.execute_fixed_formula_branch(
        compute,
        source,
        fixed_route,
        spec,
        branch_id="left",
        steps=2,
        factors=torch.ones(1, 1, 1),
    )

    assert result.work.actual_steps == 2
    assert result.work.formula_cells == 2
    assert result.work.route_applications == 2
    assert result.work.fire_count == 2
    assert result.work.commit_count == 2
    assert result.work.operation_count == 2
    assert len(result.trace_fingerprints) == 2
    assert result.step_input_fingerprints[0] == spec.input_fingerprint
    assert result.step_input_fingerprints[1] == result.step_output_fingerprints[0]
    torch.testing.assert_close(result.workspace.value[..., 2, :], torch.tensor([[5.0]]))


def test_same_future_scores_real_candidates_before_explicit_host_commit() -> None:
    store, snapshot, source, fixed_route, compute, future, spec = setup()
    left = alpha.execute_fixed_formula_branch(
        compute,
        source,
        fixed_route,
        spec,
        branch_id="left",
        steps=2,
        factors=torch.ones(1, 1, 1),
    )
    right = alpha.execute_fixed_formula_branch(
        compute,
        source,
        fixed_route,
        spec,
        branch_id="right",
        steps=2,
        factors=torch.zeros(1, 1, 1),
    )
    score = alpha.score_formula_branches((left, right), future, spec)
    assert score.scores[0] == 0.0
    assert score.scores[1] > 0.0

    bound = binding(store, snapshot)
    harness = alpha.K2BranchHarness(store, snapshot, spec)
    harness.propose(left.overlay(spec, (proposal(bound, left.workspace.value),)))
    harness.propose(right.overlay(spec, (proposal(bound, right.workspace.value),)))
    assert store.snapshot().root_id == snapshot.root_id
    receipt = alpha.select_scored_formula_branch(
        harness,
        score,
        idempotency_key="host-selected-left",
    )
    assert receipt.status is alpha.BranchRunStatus.COMMITTED
    torch.testing.assert_close(store.read(store.snapshot(), "state").value, future)


def test_future_is_score_only_and_cannot_be_substituted() -> None:
    _store, _snapshot, source, fixed_route, compute, future, spec = setup()
    candidates = tuple(
        alpha.execute_fixed_formula_branch(
            compute,
            source,
            fixed_route,
            spec,
            branch_id=branch_id,
            steps=2,
            factors=torch.full((1, 1, 1), factor),
        )
        for branch_id, factor in (("left", 1.0), ("right", 0.0))
    )
    with pytest.raises(TensorTransactionContractError, match="future tensor"):
        alpha.score_formula_branches(candidates, future + 1, spec)


def test_nonfinite_future_and_forged_receipts_fail_closed() -> None:
    _store, _snapshot, source, fixed_route, compute, _future, spec = setup()
    left = alpha.execute_fixed_formula_branch(
        compute,
        source,
        fixed_route,
        spec,
        branch_id="left",
        steps=2,
        factors=torch.ones(1, 1, 1),
    )
    with pytest.raises(TensorTransactionContractError, match="must come from"):
        alpha.FormulaBranchExecution(
            branch_id=left.branch_id,
            spec_fingerprint=left.spec_fingerprint,
            workspace=left.workspace,
            work=left.work,
            step_input_fingerprints=left.step_input_fingerprints,
            step_output_fingerprints=left.step_output_fingerprints,
            trace_fingerprints=left.trace_fingerprints,
            execution_fingerprint=left.execution_fingerprint,
            _factory_token=object(),
        )
    with pytest.raises(TensorTransactionContractError, match="must come from"):
        alpha.FrozenMSEScoreReceipt(
            spec_fingerprint=spec.fingerprint,
            candidate_execution_fingerprints=("1" * 64, "2" * 64),
            candidate_output_fingerprints=("3" * 64, "4" * 64),
            future_fingerprint="5" * 64,
            scores=(0.0, 1.0),
            _factory_token=object(),
        )

    nonfinite = torch.full_like(source.value, float("nan"))
    nonfinite_spec = alpha.BranchBatchSpec.from_snapshot(
        _snapshot,
        run_id="nonfinite-future",
        branch_ids=("left", "right"),
        executor_ref="arti/formula-fabric-compute@1",
        program_fingerprint=program().fingerprint,
        route_fingerprint=alpha.formula_route_fingerprint(fixed_route),
        input_fingerprint=alpha.workspace_fingerprint(source),
        rng_fingerprint=alpha.deterministic_formula_rng_fingerprint(),
        future_tape_fingerprint=alpha.tensor_content_fingerprint(nonfinite),
        budgets=(alpha.BranchBudget(2, 2), alpha.BranchBudget(2, 2)),
    )
    candidates = tuple(
        alpha.execute_fixed_formula_branch(
            compute,
            source,
            fixed_route,
            nonfinite_spec,
            branch_id=branch_id,
            steps=2,
            factors=torch.full((1, 1, 1), factor),
        )
        for branch_id, factor in (("left", 1.0), ("right", 0.0))
    )
    with pytest.raises(TensorTransactionContractError, match="finite"):
        alpha.score_formula_branches(candidates, nonfinite, nonfinite_spec)


def test_formula_overlay_rejects_candidate_not_produced_by_execution() -> None:
    store, snapshot, source, fixed_route, compute, _future, spec = setup()
    result = alpha.execute_fixed_formula_branch(
        compute,
        source,
        fixed_route,
        spec,
        branch_id="left",
        steps=2,
        factors=torch.ones(1, 1, 1),
    )
    bound = binding(store, snapshot)
    with pytest.raises(TensorTransactionContractError, match="must equal"):
        result.overlay(spec, (proposal(bound, torch.zeros_like(result.workspace.value)),))


def test_execution_rejects_non_hard_or_unbound_route() -> None:
    _store, _snapshot, source, fixed_route, compute, _future, spec = setup()
    soft_route = alpha.FormulaRoutePlan(
        fixed_route.weights,
        fixed_route.valid_mask,
        fixed_route.fire_mask,
        fixed_route.commit_mask,
        estimator="straight-through",
    )
    with pytest.raises(TensorTransactionContractError, match="fixed hard"):
        alpha.execute_fixed_formula_branch(
            compute,
            source,
            soft_route,
            spec,
            branch_id="left",
            steps=2,
            factors=torch.ones(1, 1, 1),
        )
