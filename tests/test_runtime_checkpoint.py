from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from arti import alpha
from arti import runtime_checkpoint as checkpoint_module


HASH = "1" * 64
ABI = "2" * 64
CONFIG = "3" * 64
STATE = "4" * 64


def _fingerprint(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _rewrite_manifest(path: Path, mutate) -> None:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
    manifest = json.loads(metadata["manifest"])
    mutate(manifest)
    save_file(
        load_file(str(path), device="cpu"),
        str(path),
        metadata={
            "schema": metadata["schema"],
            "manifest": json.dumps(
                manifest,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ),
            "manifest_sha256": _fingerprint(manifest),
        },
    )


def _committed_runtime():
    runtime = alpha.VolatileTensorRuntime(
        {"bank": torch.arange(8, dtype=torch.float32).reshape(2, 4)},
        world_id="checkpoint-test",
        abi_fingerprint=ABI,
        provenance_fingerprint=HASH,
    )
    initial = runtime.snapshot()
    transaction = runtime.begin(
        initial,
        transaction_id="commit-1",
        branch_id="main",
    )
    observed = transaction.read("bank")
    transaction.stage(
        "bank",
        observed.value + 10,
        expected_version=observed.ref.version,
        provenance_fingerprint=HASH,
    )
    receipt = transaction.commit(idempotency_key="commit-1")
    return runtime, initial, runtime.snapshot(), receipt


def _binding(runtime, snapshot):
    return alpha.bind_external_tensor(
        runtime,
        snapshot,
        "bank",
        address_namespace="session",
        partition_id="target",
        logical_id="bank",
        role="target-bank",
        authority=alpha.TensorAuthority.READ_WRITE,
        component_ref="arti/target-bank-updater@2",
        component_config_fingerprint=CONFIG,
        state_schema_ref="arti/target-bank-state@1",
        producer_state_fingerprint=STATE,
        provenance_fingerprint=HASH,
    ).binding


def test_runtime_checkpoint_round_trip_preserves_root_receipts_and_bindings(
    tmp_path: Path,
) -> None:
    runtime, _initial, snapshot, receipt = _committed_runtime()
    binding = _binding(runtime, snapshot)
    target = tmp_path / "state.runtime.arti.st"

    saved = alpha.save_runtime_checkpoint(
        runtime,
        snapshot,
        target,
        bindings=[binding],
    )
    restored = alpha.load_runtime_checkpoint(
        target,
        expected_abi_fingerprint=ABI,
        expected_component_refs=["arti/target-bank-updater@2"],
    )

    assert saved.root_fingerprint == restored.snapshot.root_fingerprint
    assert saved.manifest_fingerprint == restored.manifest_fingerprint
    assert saved.artifact_sha256 == restored.artifact_sha256
    assert restored.snapshot.root_id == snapshot.root_id
    assert restored.snapshot.epoch == snapshot.epoch
    assert restored.bindings == (binding,)
    torch.testing.assert_close(
        restored.runtime.read(restored.snapshot, "bank").value,
        runtime.read(snapshot, "bank").value,
    )
    assert restored.runtime.committed_receipt("commit-1") == receipt

    transaction = restored.runtime.begin(
        restored.snapshot,
        transaction_id="commit-2",
        branch_id="main",
    )
    observed = transaction.read("bank")
    transaction.stage(
        "bank",
        observed.value - 2,
        expected_version=observed.ref.version,
        provenance_fingerprint=HASH,
    )
    next_receipt = transaction.commit(idempotency_key="commit-2")
    assert next_receipt.new_epoch == snapshot.epoch + 1


def test_repeated_checkpoint_save_is_deterministic_and_replaceable(
    tmp_path: Path,
) -> None:
    runtime, _initial, snapshot, _receipt = _committed_runtime()
    first_path = tmp_path / "first.runtime.arti.st"
    second_path = tmp_path / "second.runtime.arti.st"

    first = alpha.save_runtime_checkpoint(runtime, snapshot, first_path)
    second = alpha.save_runtime_checkpoint(runtime, snapshot, second_path)
    replaced = alpha.save_runtime_checkpoint(runtime, snapshot, first_path)

    assert first.manifest_fingerprint == second.manifest_fingerprint
    assert replaced.manifest_fingerprint == first.manifest_fingerprint
    assert replaced.root_fingerprint == first.root_fingerprint
    assert replaced.artifact_sha256 == hashlib.sha256(first_path.read_bytes()).hexdigest()
    first_restored = alpha.load_runtime_checkpoint(
        first_path,
        expected_abi_fingerprint=ABI,
    )
    second_restored = alpha.load_runtime_checkpoint(
        second_path,
        expected_abi_fingerprint=ABI,
    )
    torch.testing.assert_close(
        first_restored.runtime.read(first_restored.snapshot, "bank").value,
        second_restored.runtime.read(second_restored.snapshot, "bank").value,
    )
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_checkpoint_replace_failure_preserves_previous_artifact_and_releases_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _initial, snapshot, _receipt = _committed_runtime()
    target = tmp_path / "atomic.runtime.arti.st"
    baseline = alpha.save_runtime_checkpoint(runtime, snapshot, target)
    baseline_payload = target.read_bytes()

    def fail_replace(_source: Path, _target: Path) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(checkpoint_module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replace failure"):
        alpha.save_runtime_checkpoint(runtime, snapshot, target)

    assert target.read_bytes() == baseline_payload
    assert not tuple(tmp_path.glob(".*.tmp"))
    assert runtime.snapshot() == snapshot
    assert runtime.committed_receipt("commit-1") is not None
    assert baseline.artifact_sha256 == hashlib.sha256(baseline_payload).hexdigest()


def test_repeated_checkpoint_loads_restore_independent_runtime_state(
    tmp_path: Path,
) -> None:
    runtime, _initial, snapshot, _receipt = _committed_runtime()
    target = tmp_path / "independent.runtime.arti.st"
    alpha.save_runtime_checkpoint(runtime, snapshot, target)

    first = alpha.load_runtime_checkpoint(target, expected_abi_fingerprint=ABI)
    second = alpha.load_runtime_checkpoint(target, expected_abi_fingerprint=ABI)
    first_before = first.runtime.read(first.snapshot, "bank").value.clone()
    second_before = second.runtime.read(second.snapshot, "bank").value.clone()

    transaction = first.runtime.begin(
        first.snapshot,
        transaction_id="independent-commit",
        branch_id="main",
    )
    observed = transaction.read("bank")
    transaction.stage(
        "bank",
        observed.value + 123,
        expected_version=observed.ref.version,
        provenance_fingerprint=HASH,
    )
    transaction.commit(idempotency_key="independent-commit")

    torch.testing.assert_close(
        second.runtime.read(second.snapshot, "bank").value,
        second_before,
    )
    assert not torch.equal(
        first.runtime.read(first.runtime.snapshot(), "bank").value,
        first_before,
    )
    assert first.runtime is not second.runtime
    assert first.runtime._published is not second.runtime._published


def test_checkpoint_load_waits_for_process_local_path_authority(
    tmp_path: Path,
) -> None:
    runtime, _initial, snapshot, _receipt = _committed_runtime()
    target = tmp_path / "path-authority.runtime.arti.st"
    alpha.save_runtime_checkpoint(runtime, snapshot, target)
    started = Event()
    finished = Event()
    errors: list[BaseException] = []

    def load() -> None:
        started.set()
        try:
            alpha.load_runtime_checkpoint(
                target,
                expected_abi_fingerprint=ABI,
            )
        except BaseException as error:  # pragma: no cover - reported below
            errors.append(error)
        finally:
            finished.set()

    with checkpoint_module._checkpoint_path_lock(target):
        worker = Thread(target=load)
        worker.start()
        assert started.wait(1.0)
        assert not finished.wait(0.1)
    assert finished.wait(5.0)
    worker.join(timeout=1.0)
    assert not errors


def test_checkpoint_rejects_stale_root_and_excludes_open_overlay(tmp_path: Path) -> None:
    runtime, initial, snapshot, _receipt = _committed_runtime()
    with pytest.raises(alpha.RuntimeCheckpointError, match="current committed root"):
        alpha.save_runtime_checkpoint(
            runtime,
            initial,
            tmp_path / "stale.runtime.arti.st",
        )

    transaction = runtime.begin(
        snapshot,
        transaction_id="uncommitted",
        branch_id="private",
    )
    observed = transaction.read("bank")
    transaction.stage(
        "bank",
        observed.value + 1000,
        expected_version=observed.ref.version,
        provenance_fingerprint=HASH,
    )
    target = tmp_path / "committed-only.runtime.arti.st"
    alpha.save_runtime_checkpoint(runtime, snapshot, target)
    restored = alpha.load_runtime_checkpoint(
        target,
        expected_abi_fingerprint=ABI,
    )
    torch.testing.assert_close(
        restored.runtime.read(restored.snapshot, "bank").value,
        observed.value,
    )


def test_checkpoint_fails_closed_on_contract_or_payload_corruption(tmp_path: Path) -> None:
    runtime, _initial, snapshot, _receipt = _committed_runtime()
    binding = _binding(runtime, snapshot)
    target = tmp_path / "contract.runtime.arti.st"
    alpha.save_runtime_checkpoint(runtime, snapshot, target, bindings=[binding])

    with pytest.raises(alpha.RuntimeCheckpointError, match="ABI mismatch"):
        alpha.load_runtime_checkpoint(
            target,
            expected_abi_fingerprint="9" * 64,
            expected_component_refs=["arti/target-bank-updater@2"],
        )
    with pytest.raises(alpha.RuntimeCheckpointError, match="component refs mismatch"):
        alpha.load_runtime_checkpoint(
            target,
            expected_abi_fingerprint=ABI,
            expected_component_refs=[],
        )

    corrupted = tmp_path / "corrupt.runtime.arti.st"
    payload = bytearray(target.read_bytes())
    payload[-1] ^= 0xFF
    corrupted.write_bytes(payload)
    with pytest.raises(alpha.RuntimeCheckpointError, match="descriptor mismatch"):
        alpha.load_runtime_checkpoint(
            corrupted,
            expected_abi_fingerprint=ABI,
            expected_component_refs=["arti/target-bank-updater@2"],
        )

    truncated = tmp_path / "truncated.runtime.arti.st"
    truncated.write_bytes(target.read_bytes()[:64])
    with pytest.raises(alpha.RuntimeCheckpointError, match="cannot be opened"):
        alpha.load_runtime_checkpoint(
            truncated,
            expected_abi_fingerprint=ABI,
            expected_component_refs=["arti/target-bank-updater@2"],
        )


def test_checkpoint_rejects_binding_that_does_not_name_a_root_page(
    tmp_path: Path,
) -> None:
    runtime, _initial, snapshot, _receipt = _committed_runtime()
    binding = _binding(runtime, snapshot)
    foreign_ref = replace(binding.tensor_ref, key="foreign-bank")
    forged = replace(binding, tensor_ref=foreign_ref)

    with pytest.raises(alpha.RuntimeCheckpointError, match="checkpoint root page"):
        alpha.save_runtime_checkpoint(
            runtime,
            snapshot,
            tmp_path / "forged-binding.runtime.arti.st",
            bindings=[forged],
        )


def test_page_provenance_changes_root_fingerprint() -> None:
    common = {
        "bank": torch.arange(8, dtype=torch.float32).reshape(2, 4),
    }
    first = alpha.VolatileTensorRuntime(
        common,
        world_id="provenance-test",
        store_instance_id="shared-store",
        abi_fingerprint=ABI,
        provenance_fingerprint="5" * 64,
    )
    second = alpha.VolatileTensorRuntime(
        common,
        world_id="provenance-test",
        store_instance_id="shared-store",
        abi_fingerprint=ABI,
        provenance_fingerprint="6" * 64,
    )

    assert first.snapshot().root_id == second.snapshot().root_id
    assert first.snapshot().root_fingerprint != second.snapshot().root_fingerprint


def test_checkpoint_rejects_foreign_receipt_and_malformed_manifest(
    tmp_path: Path,
) -> None:
    runtime, _initial, snapshot, _receipt = _committed_runtime()
    foreign = tmp_path / "foreign-receipt.runtime.arti.st"
    alpha.save_runtime_checkpoint(runtime, snapshot, foreign)

    def forge_receipt(manifest: dict[str, object]) -> None:
        wrapper = manifest["idempotency_receipts"][0]
        receipt = wrapper["receipt"]
        receipt["store_instance_id"] = "foreign-store"
        content = dict(receipt)
        content.pop("receipt_fingerprint")
        receipt["receipt_fingerprint"] = _fingerprint(content)

    _rewrite_manifest(foreign, forge_receipt)
    with pytest.raises(alpha.RuntimeCheckpointError, match="does not belong"):
        alpha.load_runtime_checkpoint(
            foreign,
            expected_abi_fingerprint=ABI,
        )

    malformed = tmp_path / "malformed.runtime.arti.st"
    alpha.save_runtime_checkpoint(runtime, snapshot, malformed)
    _rewrite_manifest(malformed, lambda manifest: manifest["root"].pop("root_id"))
    with pytest.raises(alpha.RuntimeCheckpointError, match="manifest is malformed"):
        alpha.load_runtime_checkpoint(
            malformed,
            expected_abi_fingerprint=ABI,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_resident_checkpoint_restores_pool_metadata_and_executes(tmp_path: Path) -> None:
    runtime, _initial, snapshot, _receipt = _committed_runtime()
    binding = _binding(runtime, snapshot)
    device = torch.device("cuda", torch.cuda.current_device())
    pool = alpha.HotPagePool(
        torch.tensor([[[2.0], [3.0], [0.0], [9.0]]], device=device),
        generation=torch.tensor([[4, 4, 4, 4]], device=device),
        version=torch.tensor([[7, 8, 9, 10]], device=device),
    )
    support = torch.ones(1, 3, dtype=torch.bool)
    refs = alpha.FixedPageRefs(
        logical_slot=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        page_id=torch.zeros(1, 3, dtype=torch.int64),
        offset=torch.tensor([[0, 1, 2]], dtype=torch.int64),
        expected_generation=torch.full((1, 3), 4, dtype=torch.int64),
        read_mask=support,
        write_mask=support,
        commit_mask=support,
    )
    bucket = alpha.FixedResidentBucket(1, 3, 1, torch.float32, device)
    bound = alpha.bind_hot_page_pool(pool, bucket, refs)
    target = tmp_path / "resident.runtime.arti.st"
    alpha.save_runtime_checkpoint(
        runtime,
        snapshot,
        target,
        resident=bound,
        bindings=[binding],
    )

    restored = alpha.load_runtime_checkpoint(
        target,
        expected_abi_fingerprint=ABI,
        expected_component_refs=["arti/target-bank-updater@2"],
        resident_device=device,
    )
    assert restored.resident is not None
    torch.testing.assert_close(restored.resident.pool.value, bound.pool.value)
    assert torch.equal(restored.resident.pool.validity, bound.pool.validity)
    assert torch.equal(restored.resident.pool.generation, bound.pool.generation)
    assert torch.equal(restored.resident.pool.version, bound.pool.version)
    assert restored.resident.pointer_layout_receipt() != bound.pointer_layout_receipt()

    program = alpha.FormulaFabricProgram(
        arena_capacity=3,
        feature_dim=1,
        steps=((alpha.FormulaInvocation(alpha.FormulaPrimitive.ADD, 2),),),
    )
    weights = torch.zeros(1, 1, 1, 2, 3, device=device)
    weights[..., 0, 0] = 1
    weights[..., 1, 1] = 1
    enabled = torch.ones(1, 1, 1, dtype=torch.bool, device=device)
    route = alpha.FormulaRoutePlan(weights, enabled, enabled, enabled)
    operation = alpha.FormulaResidentOperation(
        alpha.FormulaFabricCompute(
            alpha.FormulaFabric(program).to(device),
            active_count=3,
        ),
        route,
    )
    before_value = restored.resident.pool.value.clone()
    before_version = restored.resident.pool.version.clone()
    no_write = restored.resident.eager_step(operation, commit=False).clone()
    torch.testing.assert_close(no_write[:, 2], torch.tensor([[5.0]], device=device))
    torch.testing.assert_close(restored.resident.pool.value, before_value)
    assert torch.equal(restored.resident.pool.version, before_version)
    committed = restored.resident.eager_step(operation, commit=True)
    torch.testing.assert_close(restored.resident.pool.value[:, :3], committed)
    assert torch.equal(
        restored.resident.pool.version[:, :3],
        before_version[:, :3] + 1,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_resident_checkpoint_requires_a_root_bound_component(tmp_path: Path) -> None:
    runtime, _initial, snapshot, _receipt = _committed_runtime()
    device = torch.device("cuda", torch.cuda.current_device())
    pool = alpha.HotPagePool(torch.zeros(1, 1, 1, device=device))
    support = torch.ones(1, 1, dtype=torch.bool)
    bound = alpha.bind_hot_page_pool(
        pool,
        alpha.FixedResidentBucket(1, 1, 1, torch.float32, device),
        alpha.FixedPageRefs(
            logical_slot=torch.zeros(1, 1, dtype=torch.int64),
            page_id=torch.zeros(1, 1, dtype=torch.int64),
            offset=torch.zeros(1, 1, dtype=torch.int64),
            expected_generation=torch.zeros(1, 1, dtype=torch.int64),
            read_mask=support,
            write_mask=support,
            commit_mask=support,
        ),
    )

    with pytest.raises(alpha.RuntimeCheckpointError, match="root-bound"):
        alpha.save_runtime_checkpoint(
            runtime,
            snapshot,
            tmp_path / "unbound.runtime.arti.st",
            resident=bound,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_resident_checkpoint_rejects_closed_pool(tmp_path: Path) -> None:
    runtime, _initial, snapshot, _receipt = _committed_runtime()
    binding = _binding(runtime, snapshot)
    device = torch.device("cuda", torch.cuda.current_device())
    support = torch.ones(1, 1, dtype=torch.bool)
    bound = alpha.bind_hot_page_pool(
        alpha.HotPagePool(torch.zeros(1, 1, 1, device=device)),
        alpha.FixedResidentBucket(1, 1, 1, torch.float32, device),
        alpha.FixedPageRefs(
            logical_slot=torch.zeros(1, 1, dtype=torch.int64),
            page_id=torch.zeros(1, 1, dtype=torch.int64),
            offset=torch.zeros(1, 1, dtype=torch.int64),
            expected_generation=torch.zeros(1, 1, dtype=torch.int64),
            read_mask=support,
            write_mask=support,
            commit_mask=support,
        ),
    )
    bound.close(close_pool=True)

    with pytest.raises(alpha.GPUResidentContractError, match="closed"):
        alpha.save_runtime_checkpoint(
            runtime,
            snapshot,
            tmp_path / "closed.runtime.arti.st",
            resident=bound,
            bindings=[binding],
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_resident_checkpoint_waits_for_authority_lock(tmp_path: Path) -> None:
    runtime, _initial, snapshot, _receipt = _committed_runtime()
    binding = _binding(runtime, snapshot)
    device = torch.device("cuda", torch.cuda.current_device())
    pool = alpha.HotPagePool(torch.zeros(1, 1, 1, device=device))
    support = torch.ones(1, 1, dtype=torch.bool)
    bound = alpha.bind_hot_page_pool(
        pool,
        alpha.FixedResidentBucket(1, 1, 1, torch.float32, device),
        alpha.FixedPageRefs(
            logical_slot=torch.zeros(1, 1, dtype=torch.int64),
            page_id=torch.zeros(1, 1, dtype=torch.int64),
            offset=torch.zeros(1, 1, dtype=torch.int64),
            expected_generation=torch.zeros(1, 1, dtype=torch.int64),
            read_mask=support,
            write_mask=support,
            commit_mask=support,
        ),
    )
    started = Event()
    finished = Event()
    errors: list[BaseException] = []

    def save() -> None:
        started.set()
        try:
            alpha.save_runtime_checkpoint(
                runtime,
                snapshot,
                tmp_path / "locked.runtime.arti.st",
                resident=bound,
                bindings=[binding],
            )
        except BaseException as error:  # pragma: no cover - reported below
            errors.append(error)
        finally:
            finished.set()

    with bound.authority_lock:
        worker = Thread(target=save)
        worker.start()
        assert started.wait(1.0)
        assert not finished.wait(0.1)
    assert finished.wait(5.0)
    worker.join(timeout=1.0)
    assert not errors


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_resident_checkpoint_quiesces_pool_before_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime, _initial, snapshot, _receipt = _committed_runtime()
    binding = _binding(runtime, snapshot)
    device = torch.device("cuda", torch.cuda.current_device())
    bound = alpha.bind_hot_page_pool(
        alpha.HotPagePool(torch.zeros(1, 1, 1, device=device)),
        alpha.FixedResidentBucket(1, 1, 1, torch.float32, device),
        alpha.FixedPageRefs(
            logical_slot=torch.zeros(1, 1, dtype=torch.int64),
            page_id=torch.zeros(1, 1, dtype=torch.int64),
            offset=torch.zeros(1, 1, dtype=torch.int64),
            expected_generation=torch.zeros(1, 1, dtype=torch.int64),
            read_mask=torch.ones(1, 1, dtype=torch.bool),
            write_mask=torch.ones(1, 1, dtype=torch.bool),
            commit_mask=torch.ones(1, 1, dtype=torch.bool),
        ),
    )
    original_quiesce = bound.pool._quiesce
    calls = 0

    def observed_quiesce() -> None:
        nonlocal calls
        calls += 1
        original_quiesce()

    monkeypatch.setattr(bound.pool, "_quiesce", observed_quiesce)
    alpha.save_runtime_checkpoint(
        runtime,
        snapshot,
        tmp_path / "quiesced.runtime.arti.st",
        resident=bound,
        bindings=[binding],
    )

    assert calls == 1
