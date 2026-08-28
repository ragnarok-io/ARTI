from __future__ import annotations

import pytest
import torch

import arti
from arti.tensor_transaction import (
    CommitReceipt,
    ConflictReason,
    ConflictReceipt,
    TensorOwnershipError,
    TensorTransactionContractError,
    TensorTransactionStateError,
    TensorTransactionStatus,
)


HASH = "0" * 64


def runtime(initial: dict[str, torch.Tensor] | None = None):
    return arti.alpha.VolatileTensorRuntime(
        initial,
        world_id="test-world",
        store_instance_id="test-store",
        abi_fingerprint="1" * 64,
        provenance_fingerprint=HASH,
    )


def test_runtime_owns_initial_storage_and_reads_return_owned_clones() -> None:
    source = torch.tensor([1.0, 2.0])
    store = runtime({"bank": source})
    source.add_(100)
    first = store.read(store.snapshot(), "bank")
    torch.testing.assert_close(first.value, torch.tensor([1.0, 2.0]))
    first.value.zero_()
    torch.testing.assert_close(store.read(store.snapshot(), "bank").value, torch.tensor([1.0, 2.0]))
    assert first.value.data_ptr() != store.read(store.snapshot(), "bank").value.data_ptr()


def test_commit_is_atomic_cow_and_old_snapshot_remains_unchanged() -> None:
    store = runtime({"left": torch.tensor([1.0]), "right": torch.tensor([2.0])})
    before = store.snapshot()
    tx = store.begin(before, transaction_id="tx-1", branch_id="branch-a")
    left = tx.read("left")
    right = tx.read("right")
    tx.stage("left", left.value + 10, expected_version=left.ref.version, provenance_fingerprint="2" * 64)
    tx.stage("right", right.value + 20, expected_version=right.ref.version, provenance_fingerprint="3" * 64)
    receipt = tx.commit(idempotency_key="commit-1")
    assert isinstance(receipt, CommitReceipt)
    after = store.snapshot()
    assert after.epoch == before.epoch + 1
    torch.testing.assert_close(store.read(before, "left").value, torch.tensor([1.0]))
    torch.testing.assert_close(store.read(before, "right").value, torch.tensor([2.0]))
    torch.testing.assert_close(store.read(after, "left").value, torch.tensor([11.0]))
    torch.testing.assert_close(store.read(after, "right").value, torch.tensor([22.0]))


def test_stale_root_conflict_has_no_partial_publish() -> None:
    store = runtime({"a": torch.tensor([1.0]), "b": torch.tensor([2.0])})
    base = store.snapshot()
    first = store.begin(base, transaction_id="first", branch_id="a")
    second = store.begin(base, transaction_id="second", branch_id="b")
    first.stage("a", torch.tensor([3.0]), expected_version=1, provenance_fingerprint=HASH)
    second.stage("b", torch.tensor([4.0]), expected_version=1, provenance_fingerprint=HASH)
    assert isinstance(first.commit(idempotency_key="first-key"), CommitReceipt)
    conflict = second.commit(idempotency_key="second-key")
    assert isinstance(conflict, ConflictReceipt)
    assert conflict.reason is ConflictReason.STALE_ROOT
    assert [(item.key, item.expected_version, item.current_version) for item in conflict.page_conflicts] == [("a", 1, 2)]
    assert second.status is TensorTransactionStatus.CONFLICTED
    torch.testing.assert_close(store.read(store.snapshot(), "b").value, torch.tensor([2.0]))


def test_rollback_is_idempotent_and_never_publishes() -> None:
    store = runtime({"bank": torch.tensor([1.0])})
    base = store.snapshot()
    tx = store.begin(base, transaction_id="rollback", branch_id="main")
    tx.stage("bank", torch.tensor([9.0]), expected_version=1, provenance_fingerprint=HASH)
    first = tx.rollback()
    assert tx.rollback() == first
    assert store.snapshot().root_id == base.root_id
    torch.testing.assert_close(store.read(store.snapshot(), "bank").value, torch.tensor([1.0]))
    with pytest.raises(TensorTransactionStateError):
        tx.commit(idempotency_key="after-rollback")


def test_same_idempotency_request_returns_original_receipt_without_new_epoch() -> None:
    store = runtime({"bank": torch.tensor([1.0])})
    tx = store.begin(store.snapshot(), transaction_id="retry", branch_id="main")
    tx.stage("bank", torch.tensor([2.0]), expected_version=1, provenance_fingerprint=HASH)
    first = tx.commit(idempotency_key="retry-key")
    epoch = store.snapshot().epoch
    assert tx.commit(idempotency_key="retry-key") is first
    assert store.snapshot().epoch == epoch


def test_reconstructed_replay_returns_original_receipt_without_publication() -> None:
    store = runtime({"bank": torch.tensor([1.0])})
    base = store.snapshot()
    first = store.begin(base, transaction_id="replay", branch_id="main")
    first.stage("bank", torch.tensor([2.0]), expected_version=1, provenance_fingerprint=HASH)
    receipt = first.commit(idempotency_key="replay-key")
    epoch = store.snapshot().epoch

    replay = store.begin(base, transaction_id="replay", branch_id="main")
    replay.stage("bank", torch.tensor([2.0]), expected_version=1, provenance_fingerprint=HASH)
    assert store.replay(replay, idempotency_key="replay-key") is receipt
    assert store.snapshot().epoch == epoch


def test_empty_transaction_cannot_create_a_new_epoch() -> None:
    store = runtime()
    tx = store.begin(store.snapshot(), transaction_id="empty", branch_id="main")
    with pytest.raises(TensorTransactionContractError, match="empty write set"):
        tx.commit(idempotency_key="empty-key")
    assert store.snapshot().epoch == 0


def test_same_idempotency_key_with_different_request_fails_closed() -> None:
    store = runtime({"bank": torch.tensor([1.0])})
    base = store.snapshot()
    first = store.begin(base, transaction_id="request-a", branch_id="main")
    first.stage("bank", torch.tensor([2.0]), expected_version=1, provenance_fingerprint=HASH)
    assert isinstance(first.commit(idempotency_key="shared-key"), CommitReceipt)
    current = store.snapshot()
    second = store.begin(current, transaction_id="request-b", branch_id="main")
    second.stage("bank", torch.tensor([3.0]), expected_version=2, provenance_fingerprint=HASH)
    conflict = second.commit(idempotency_key="shared-key")
    assert isinstance(conflict, ConflictReceipt)
    assert conflict.reason is ConflictReason.IDEMPOTENCY_MISMATCH
    assert store.snapshot().epoch == current.epoch


def test_nonfinite_tensor_is_rejected_at_runtime_and_stage_ingress() -> None:
    with pytest.raises(TensorOwnershipError, match="finite"):
        runtime({"bank": torch.tensor([float("nan")])})

    store = runtime({"bank": torch.tensor([1.0])})
    tx = store.begin(store.snapshot(), transaction_id="nonfinite", branch_id="main")
    with pytest.raises(TensorOwnershipError, match="finite"):
        tx.stage(
            "bank",
            torch.tensor([float("inf")]),
            expected_version=1,
            provenance_fingerprint=HASH,
        )


def test_conflicted_transaction_discards_private_overlay() -> None:
    store = runtime({"bank": torch.tensor([1.0])})
    snapshot = store.snapshot()
    stale = store.begin(snapshot, transaction_id="stale-clean", branch_id="stale")
    stale.stage(
        "bank",
        torch.tensor([9.0]),
        expected_version=1,
        provenance_fingerprint=HASH,
    )
    empty_before = store.begin(
        snapshot,
        transaction_id="empty-reference",
        branch_id="empty",
    )
    assert stale.write_set_digest != empty_before.write_set_digest

    outside = store.begin(snapshot, transaction_id="outside-clean", branch_id="outside")
    outside.stage(
        "bank",
        torch.tensor([3.0]),
        expected_version=1,
        provenance_fingerprint=HASH,
    )
    outside.commit(idempotency_key="outside-clean")
    assert isinstance(stale.commit(idempotency_key="stale-clean"), ConflictReceipt)

    empty_after = store.begin(
        store.snapshot(),
        transaction_id="empty-after",
        branch_id="empty-after",
    )
    assert stale.write_set_digest == empty_after.write_set_digest
    assert stale.read_set_digest == empty_after.read_set_digest


def test_stage_owns_candidate_and_rejects_blind_overwrite() -> None:
    store = runtime({"bank": torch.tensor([1.0])})
    tx = store.begin(store.snapshot(), transaction_id="ownership", branch_id="main")
    candidate = torch.tensor([5.0])
    tx.stage("bank", candidate, expected_version=1, provenance_fingerprint=HASH)
    candidate.add_(100)
    torch.testing.assert_close(tx.read("bank").value, torch.tensor([5.0]))
    other = store.begin(store.snapshot(), transaction_id="blind", branch_id="main")
    with pytest.raises(TensorTransactionContractError):
        other.stage("bank", torch.tensor([2.0]), expected_version=None, provenance_fingerprint=HASH)


def test_new_page_requires_none_version_and_is_published_at_version_one() -> None:
    store = runtime()
    tx = store.begin(store.snapshot(), transaction_id="new-page", branch_id="main")
    tx.stage("new", torch.arange(3), expected_version=None, provenance_fingerprint=HASH)
    assert isinstance(tx.commit(idempotency_key="new-page-key"), CommitReceipt)
    observed = store.read(store.snapshot(), "new")
    assert observed.ref.version == 1
    torch.testing.assert_close(observed.value, torch.arange(3))


def test_v1_rejects_grad_noncontiguous_and_non_cpu_storage() -> None:
    with pytest.raises(TensorOwnershipError, match="gradients"):
        runtime({"bank": torch.ones(2, requires_grad=True)})
    with pytest.raises(TensorOwnershipError, match="contiguous"):
        runtime({"bank": torch.ones(2, 3).transpose(0, 1)})
    if torch.cuda.is_available():
        with pytest.raises(TensorOwnershipError, match="CPU"):
            runtime({"bank": torch.ones(2, device="cuda")})


def test_snapshot_from_another_runtime_is_rejected() -> None:
    left = runtime({"bank": torch.ones(1)})
    right = arti.alpha.VolatileTensorRuntime(
        {"bank": torch.ones(1)},
        world_id="test-world",
        store_instance_id="other-store",
        abi_fingerprint="1" * 64,
    )
    with pytest.raises(TensorTransactionContractError, match="another runtime"):
        right.begin(left.snapshot(), transaction_id="foreign", branch_id="main")


def test_same_metadata_runtime_and_forged_snapshot_are_rejected() -> None:
    left = runtime({"bank": torch.ones(1)})
    right = runtime({"bank": torch.ones(1)})
    snapshot = left.snapshot()
    with pytest.raises(TensorTransactionContractError, match="another runtime"):
        right.begin(snapshot, transaction_id="same-metadata", branch_id="main")

    forged = arti.alpha.TensorSnapshot(
        store_instance_id=snapshot.store_instance_id,
        world_id=snapshot.world_id,
        root_id=snapshot.root_id,
        epoch=snapshot.epoch,
        abi_fingerprint=snapshot.abi_fingerprint,
        root_fingerprint=snapshot.root_fingerprint,
        page_refs=snapshot.page_refs,
        _owner_token=snapshot._owner_token,
    )
    with pytest.raises(TensorTransactionContractError, match="not registered"):
        left.begin(forged, transaction_id="forged", branch_id="main")


def test_transaction_constructor_cannot_bypass_runtime_factory() -> None:
    from arti.tensor_transaction import TensorTransaction

    store = runtime({"bank": torch.ones(1)})
    snapshot = store.snapshot()
    with pytest.raises(TensorTransactionContractError, match="must be created"):
        TensorTransaction(
            store,
            snapshot,
            store._published.root,
            _factory_token=object(),
            transaction_id="forged-transaction",
            branch_id="main",
        )
    assert not hasattr(arti.alpha, "TensorTransaction")


def test_snapshot_has_no_live_tensor_and_runtime_identity_is_read_only() -> None:
    store = runtime({"bank": torch.ones(1)})
    snapshot = store.snapshot()
    assert not hasattr(snapshot, "_root")
    assert all(not isinstance(value, torch.Tensor) for value in vars(snapshot).values())
    with pytest.raises(AttributeError):
        store.world_id = "changed"


@pytest.mark.parametrize("candidate", [torch.ones(2), torch.ones(1, dtype=torch.int64)])
def test_existing_page_rejects_shape_or_dtype_drift(candidate: torch.Tensor) -> None:
    store = runtime({"bank": torch.ones(1)})
    tx = store.begin(store.snapshot(), transaction_id="schema", branch_id="main")
    with pytest.raises(TensorTransactionContractError, match="same dtype and shape"):
        tx.stage("bank", candidate, expected_version=1, provenance_fingerprint=HASH)
    assert store.snapshot().epoch == 0


def test_alpha_surface_is_explicit_and_not_an_nn_module() -> None:
    store = runtime()
    assert store._runtime_contract_ref == "arti/volatile-tensor-runtime@1"
    assert not hasattr(store, "forward")
    assert arti.alpha.TensorRef._runtime_contract_ref == "arti/tensor-ref@1"
