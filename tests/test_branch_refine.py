from __future__ import annotations

import pytest
import torch

import arti
from arti import alpha
from arti.tensor_transaction import (
    CommitReceipt,
    ConflictReceipt,
    TensorTransactionContractError,
)


HASH = "0" * 64
CONFIG = "1" * 64
STATE = "2" * 64
ABI = "3" * 64


def runtime() -> alpha.VolatileTensorRuntime:
    return alpha.VolatileTensorRuntime(
        {"state": torch.zeros(1, 2)},
        world_id="branch-world",
        store_instance_id="branch-store",
        abi_fingerprint=ABI,
        provenance_fingerprint=HASH,
    )


def bound(
    store: alpha.VolatileTensorRuntime,
    snapshot: alpha.TensorSnapshot,
) -> alpha.BoundTensorRead:
    return alpha.bind_external_tensor(
        store,
        snapshot,
        "state",
        address_namespace="session",
        partition_id="main",
        logical_id="state",
        role="formula-state",
        authority=alpha.TensorAuthority.READ_WRITE,
        component_ref="arti/formula-fabric@1",
        component_config_fingerprint=CONFIG,
        state_schema_ref="arti/formula-arena-value@1",
        producer_state_fingerprint=STATE,
        provenance_fingerprint=HASH,
    )


def external(
    item: alpha.BoundTensorRead,
    value: float,
) -> alpha.ExternalTensorProposal:
    return alpha.ExternalTensorProposal(
        item.binding,
        torch.full((1, 2), value),
        producer_ref="arti/formula-fabric@1",
        producer_config_fingerprint=CONFIG,
        producer_state_fingerprint=STATE,
    )


def work(*, steps: int = 2, operations: int = 8) -> alpha.BranchWorkReceipt:
    return alpha.BranchWorkReceipt(
        actual_steps=steps,
        formula_cells=4,
        route_applications=1,
        fire_count=4,
        commit_count=4,
        operation_count=operations,
        stop_reason="budget",
        residual_fingerprint=HASH,
        finite=True,
    )


def spec(snapshot: alpha.TensorSnapshot, *, run_id: str = "run-1") -> alpha.BranchBatchSpec:
    return alpha.BranchBatchSpec.from_snapshot(
        snapshot,
        run_id=run_id,
        branch_ids=("left", "right"),
        executor_ref="arti/formula-fabric-compute@1",
        program_fingerprint="4" * 64,
        route_fingerprint="5" * 64,
        input_fingerprint="6" * 64,
        rng_fingerprint="7" * 64,
        future_tape_fingerprint="8" * 64,
        budgets=(alpha.BranchBudget(1, 4), alpha.BranchBudget(1, 4)),
    )


def overlay(
    branch_id: str,
    item: alpha.BoundTensorRead,
    value: float,
    *,
    branch_spec: alpha.BranchBatchSpec,
    branch_work: alpha.BranchWorkReceipt | None = None,
) -> alpha.OverlayProposal:
    actual_work = work() if branch_work is None else branch_work
    inputs = [branch_spec.input_fingerprint]
    outputs = []
    for index in range(actual_work.actual_steps):
        output = f"{index + 10:x}" * 64
        outputs.append(output)
        if index + 1 < actual_work.actual_steps:
            inputs.append(output)
    return alpha.OverlayProposal(
        branch_id=branch_id,
        spec_fingerprint=branch_spec.fingerprint,
        proposals=(external(item, value),),
        work=actual_work,
        step_input_fingerprints=tuple(inputs),
        step_output_fingerprints=tuple(outputs),
        execution_fingerprint="9" * 64,
    )


def test_two_private_overlays_publish_only_explicit_winner() -> None:
    store = runtime()
    snapshot = store.snapshot()
    item = bound(store, snapshot)
    harness = alpha.K2BranchHarness(store, snapshot, spec(snapshot))
    harness.propose(overlay("left", item, 1.0, branch_spec=harness.spec))
    harness.propose(overlay("right", item, 2.0, branch_spec=harness.spec))
    assert store.snapshot().root_id == snapshot.root_id

    receipt = harness.select("right", idempotency_key="winner-right")
    assert receipt.status is alpha.BranchRunStatus.COMMITTED
    assert isinstance(receipt.commit_receipt, CommitReceipt)
    assert [item.branch_id for item in receipt.rollback_receipts] == ["left"]
    torch.testing.assert_close(store.read(store.snapshot(), "state").value, torch.full((1, 2), 2.0))
    assert store.read(store.snapshot(), "state").ref.version == 2
    assert harness.select("right", idempotency_key="winner-right") is receipt
    with pytest.raises(TensorTransactionContractError, match="another decision"):
        harness.select("right", idempotency_key="different-key")
    with pytest.raises(TensorTransactionContractError, match="another decision"):
        harness.select("left", idempotency_key="winner-left")


def test_discarded_matched_sham_executes_proposals_without_publication() -> None:
    store = runtime()
    snapshot = store.snapshot()
    item = bound(store, snapshot)
    harness = alpha.K2BranchHarness(store, snapshot, spec(snapshot, run_id="sham"))
    harness.propose(overlay("left", item, 1.0, branch_spec=harness.spec))
    harness.propose(overlay("right", item, 2.0, branch_spec=harness.spec))
    receipt = harness.select(None, idempotency_key="unused-sham-key")
    assert receipt.status is alpha.BranchRunStatus.DISCARDED
    assert receipt.decision_kind == "discard"
    assert receipt._runtime_contract_ref == "arti/branch-run-receipt@2"
    assert receipt.commit_receipt is None
    assert len(receipt.rollback_receipts) == 2
    assert store.snapshot().root_id == snapshot.root_id
    torch.testing.assert_close(store.read(store.snapshot(), "state").value, torch.zeros(1, 2))


def test_branch_run_receipt_cannot_be_forged_by_direct_construction() -> None:
    with pytest.raises(TensorTransactionContractError, match="factory owned"):
        alpha.BranchRunReceipt(
            spec_fingerprint=HASH,
            status=alpha.BranchRunStatus.DISCARDED,
            winner_branch_id=None,
            decision_idempotency_key="forged",
            decision_request_fingerprint=HASH,
            proposal_fingerprints=(),
            work_receipts=(),
            commit_receipt=None,
            rollback_receipts=(),
            decision_kind="discard",
            _factory_token=object(),
        )


def test_stale_winner_conflicts_and_loser_is_discarded() -> None:
    store = runtime()
    snapshot = store.snapshot()
    item = bound(store, snapshot)
    harness = alpha.K2BranchHarness(store, snapshot, spec(snapshot, run_id="stale"))
    harness.propose(overlay("left", item, 1.0, branch_spec=harness.spec))
    harness.propose(overlay("right", item, 2.0, branch_spec=harness.spec))

    outside = store.begin(snapshot, transaction_id="outside", branch_id="outside")
    outside.stage("state", torch.full((1, 2), 3.0), expected_version=1, provenance_fingerprint=HASH)
    outside.commit(idempotency_key="outside")
    receipt = harness.select("left", idempotency_key="stale-left")
    assert receipt.status is alpha.BranchRunStatus.CONFLICTED
    assert isinstance(receipt.commit_receipt, ConflictReceipt)
    assert [item.branch_id for item in receipt.rollback_receipts] == ["right"]
    torch.testing.assert_close(store.read(store.snapshot(), "state").value, torch.full((1, 2), 3.0))


def test_reconstructed_run_replays_original_commit_receipt() -> None:
    store = runtime()
    snapshot = store.snapshot()
    item = bound(store, snapshot)
    first = alpha.K2BranchHarness(store, snapshot, spec(snapshot, run_id="replay"))
    first.propose(overlay("left", item, 1.0, branch_spec=first.spec))
    first.propose(overlay("right", item, 2.0, branch_spec=first.spec))
    first_receipt = first.select("left", idempotency_key="replay-left")
    epoch = store.snapshot().epoch

    replay = alpha.K2BranchHarness(store, snapshot, spec(snapshot, run_id="replay"))
    replay.propose(overlay("left", item, 1.0, branch_spec=replay.spec))
    replay.propose(overlay("right", item, 2.0, branch_spec=replay.spec))
    replay_receipt = replay.select("left", idempotency_key="replay-left")
    assert replay_receipt.commit_receipt is first_receipt.commit_receipt
    assert store.snapshot().epoch == epoch


def test_branch_work_budget_and_matched_work_fail_closed() -> None:
    store = runtime()
    snapshot = store.snapshot()
    item = bound(store, snapshot)
    harness = alpha.K2BranchHarness(store, snapshot, spec(snapshot, run_id="work"))
    with pytest.raises(TensorTransactionContractError, match="violate its budget"):
        harness.propose(
            overlay(
                "left",
                item,
                1.0,
                branch_spec=harness.spec,
                branch_work=work(steps=5),
            )
        )

    harness.propose(overlay("left", item, 1.0, branch_spec=harness.spec))
    harness.propose(
        overlay(
            "right",
            item,
            2.0,
            branch_spec=harness.spec,
            branch_work=work(operations=9),
        )
    )
    with pytest.raises(TensorTransactionContractError, match="not matched"):
        harness.select("left", idempotency_key="mismatch")
    aborted = harness.abort(idempotency_key="abort-work")
    assert aborted.status is alpha.BranchRunStatus.DISCARDED
    assert aborted.decision_kind == "discard"
    assert len(aborted.rollback_receipts) == 2
    assert store.snapshot().root_id == snapshot.root_id


def test_step_lineage_must_cover_every_executed_refine_step() -> None:
    store = runtime()
    snapshot = store.snapshot()
    item = bound(store, snapshot)
    with pytest.raises(TensorTransactionContractError, match="match actual_steps"):
        alpha.OverlayProposal(
            branch_id="left",
            spec_fingerprint=spec(snapshot).fingerprint,
            proposals=(external(item, 1.0),),
            work=work(steps=2),
            step_input_fingerprints=("a" * 64,),
            step_output_fingerprints=("b" * 64, "c" * 64),
            execution_fingerprint="9" * 64,
        )


def test_partial_multi_page_failure_aborts_every_private_overlay() -> None:
    store = runtime()
    snapshot = store.snapshot()
    item = bound(store, snapshot)
    harness = alpha.K2BranchHarness(store, snapshot, spec(snapshot, run_id="atomic"))

    foreign = alpha.VolatileTensorRuntime(
        {"other": torch.zeros(1, 2)},
        world_id="branch-world",
        store_instance_id="foreign-store",
        abi_fingerprint=ABI,
        provenance_fingerprint=HASH,
    )
    foreign_snapshot = foreign.snapshot()
    foreign_item = alpha.bind_external_tensor(
        foreign,
        foreign_snapshot,
        "other",
        address_namespace="session",
        partition_id="main",
        logical_id="other",
        role="formula-state",
        authority=alpha.TensorAuthority.READ_WRITE,
        component_ref="arti/formula-fabric@1",
        component_config_fingerprint=CONFIG,
        state_schema_ref="arti/formula-arena-value@1",
        producer_state_fingerprint=STATE,
        provenance_fingerprint=HASH,
    )
    base = overlay("left", item, 1.0, branch_spec=harness.spec)
    invalid = alpha.OverlayProposal(
        branch_id="left",
        spec_fingerprint=harness.spec.fingerprint,
        proposals=(*base.proposals, external(foreign_item, 2.0)),
        work=base.work,
        step_input_fingerprints=base.step_input_fingerprints,
        step_output_fingerprints=base.step_output_fingerprints,
        execution_fingerprint=base.execution_fingerprint,
    )
    with pytest.raises(TensorTransactionContractError, match="parent snapshot"):
        harness.propose(invalid)
    receipt = harness.abort(idempotency_key="abort-atomic")
    assert receipt.status is alpha.BranchRunStatus.DISCARDED
    assert store.snapshot().root_id == snapshot.root_id


def test_branch_spec_and_lifecycle_reject_implicit_or_duplicate_paths() -> None:
    store = runtime()
    snapshot = store.snapshot()
    with pytest.raises(TensorTransactionContractError, match="exactly two"):
        alpha.BranchBatchSpec(
            run_id="bad",
            branch_ids=("same", "same"),
            parent_store_instance_id=snapshot.store_instance_id,
            parent_world_id=snapshot.world_id,
            parent_root_id=snapshot.root_id,
            parent_epoch=snapshot.epoch,
            parent_root_fingerprint=snapshot.root_fingerprint,
            executor_ref="arti/formula-fabric-compute@1",
            program_fingerprint=HASH,
            route_fingerprint=HASH,
            input_fingerprint=HASH,
            rng_fingerprint=HASH,
            future_tape_fingerprint=HASH,
            budgets=(alpha.BranchBudget(1, 1), alpha.BranchBudget(1, 1)),
        )
    item = bound(store, snapshot)
    harness = alpha.K2BranchHarness(store, snapshot, spec(snapshot, run_id="lifecycle"))
    harness.propose(overlay("left", item, 1.0, branch_spec=harness.spec))
    with pytest.raises(TensorTransactionContractError, match="already has"):
        harness.propose(overlay("left", item, 2.0, branch_spec=harness.spec))
    with pytest.raises(TensorTransactionContractError, match="both branches"):
        harness.select("left", idempotency_key="too-early")


def test_matched_work_receipts_can_compare_commit_and_sham() -> None:
    committed_store = runtime()
    committed_snapshot = committed_store.snapshot()
    committed_bound = bound(committed_store, committed_snapshot)
    committed = alpha.K2BranchHarness(
        committed_store,
        committed_snapshot,
        spec(committed_snapshot, run_id="commit-arm"),
    )
    committed.propose(
        overlay("left", committed_bound, 1.0, branch_spec=committed.spec)
    )
    committed.propose(
        overlay("right", committed_bound, 2.0, branch_spec=committed.spec)
    )
    committed_receipt = committed.select("left", idempotency_key="commit-arm")

    sham_store = runtime()
    sham_snapshot = sham_store.snapshot()
    sham_bound = bound(sham_store, sham_snapshot)
    sham = alpha.K2BranchHarness(
        sham_store,
        sham_snapshot,
        spec(sham_snapshot, run_id="sham-arm"),
    )
    sham.propose(overlay("left", sham_bound, 1.0, branch_spec=sham.spec))
    sham.propose(overlay("right", sham_bound, 2.0, branch_spec=sham.spec))
    sham_receipt = sham.select(None, idempotency_key="unused")
    alpha.assert_matched_branch_work(committed_receipt, sham_receipt)


def test_branch_harness_is_alpha_only_and_not_an_executor() -> None:
    assert not hasattr(arti, "K2BranchHarness")
    assert not hasattr(arti.nn, "K2BranchHarness")
    assert alpha.K2BranchHarness._runtime_contract_ref == "arti/k2-branch-harness@1"
    assert not hasattr(alpha.K2BranchHarness, "forward")
