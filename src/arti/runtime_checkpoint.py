"""Same-ABI clean-restart checkpoints for persistent tensor runtimes."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
from types import MappingProxyType
from typing import ClassVar, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file
from torch import Tensor

from .gpu_resident import (
    BoundHotPagePool,
    FixedPageRefs,
    FixedResidentBucket,
    HotPagePool,
    bind_hot_page_pool,
)
from .tensor_binding import ExternalTensorBinding, TensorAuthority
from .tensor_transaction import (
    CommitReceipt,
    TensorRef,
    TensorSnapshot,
    TensorTransactionContractError,
    VolatileTensorRuntime,
    _OwnedPage,
    _PublishedState,
    _WorldRoot,
    _root_fingerprint,
    _require_sha256,
    _tensor_hash,
)


RUNTIME_CHECKPOINT_SCHEMA = "arti/runtime-checkpoint@1"
_EMPTY_HASH = hashlib.sha256(b"").hexdigest()
_CHECKPOINT_PATH_LOCKS_GUARD = RLock()
_CHECKPOINT_PATH_LOCKS: dict[str, RLock] = {}


class RuntimeCheckpointError(RuntimeError):
    """Raised when a runtime checkpoint is malformed or incompatible."""


def _checkpoint_path_lock(path: Path) -> RLock:
    key = str(path.resolve())
    with _CHECKPOINT_PATH_LOCKS_GUARD:
        return _CHECKPOINT_PATH_LOCKS.setdefault(key, RLock())


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("ascii")).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_descriptor(value: Tensor) -> dict[str, object]:
    cpu = value.detach().to("cpu").contiguous()
    return {
        "dtype": str(cpu.dtype),
        "shape": list(cpu.shape),
        "content_sha256": _tensor_hash(cpu),
    }


def _owned_cpu(value: Tensor) -> Tensor:
    return value.detach().to("cpu").contiguous()


@dataclass(frozen=True)
class RuntimeCheckpointReceipt:
    """Receipt for one atomically replaced checkpoint artifact."""

    path: str
    artifact_sha256: str
    manifest_fingerprint: str
    root_fingerprint: str
    root_epoch: int
    page_count: int
    includes_resident_pool: bool
    _component_reference: ClassVar[str] = "arti/runtime-checkpoint-receipt@1"


@dataclass(frozen=True)
class RestoredRuntimeCheckpoint:
    """Fresh runtime state reconstructed from one verified artifact."""

    runtime: VolatileTensorRuntime
    snapshot: TensorSnapshot
    resident: BoundHotPagePool | None
    bindings: tuple[ExternalTensorBinding, ...]
    manifest_fingerprint: str
    artifact_sha256: str
    _component_reference: ClassVar[str] = "arti/restored-runtime-checkpoint@1"


def _serialize_refs(refs: FixedPageRefs) -> dict[str, object]:
    fields = (
        "logical_slot",
        "page_id",
        "offset",
        "expected_generation",
        "read_mask",
        "write_mask",
        "commit_mask",
    )
    return {
        name: _owned_cpu(getattr(refs, name)).tolist()
        for name in fields
    }


def _serialize_resident(
    resident: BoundHotPagePool,
    tensors: dict[str, Tensor],
    *,
    binding_fingerprints: Sequence[str],
) -> dict[str, object]:
    with resident.authority_lock:
        resident.assert_open()
        resident.pool._quiesce()
        pool = resident.pool
        entries = {
            "value": _owned_cpu(pool.value),
            "validity": _owned_cpu(pool.validity),
            "generation": _owned_cpu(pool.generation),
            "version": _owned_cpu(pool.version),
        }
        descriptors: dict[str, object] = {}
        for name, value in entries.items():
            tensor_key = f"resident.{name}"
            tensors[tensor_key] = value
            descriptors[name] = {
                "tensor_key": tensor_key,
                **_tensor_descriptor(value),
            }
        bucket = resident.bucket
        bucket_payload = {
            "batch_size": bucket.batch_size,
            "workset_slots": bucket.workset_slots,
            "feature_dim": bucket.feature_dim,
            "dtype": str(bucket.dtype),
            "refine_steps": bucket.refine_steps,
        }
        refs_payload = _serialize_refs(resident.refs)
        snapshot_payload = {
            "pool": descriptors,
            "bucket": bucket_payload,
            "refs": refs_payload,
            "binding_fingerprints": list(binding_fingerprints),
        }
        return {
            **snapshot_payload,
            "authority_snapshot_fingerprint": _fingerprint(snapshot_payload),
        }


def save_runtime_checkpoint(
    runtime: VolatileTensorRuntime,
    snapshot: TensorSnapshot,
    path: str | Path,
    *,
    resident: BoundHotPagePool | None = None,
    bindings: Sequence[ExternalTensorBinding] = (),
) -> RuntimeCheckpointReceipt:
    """Atomically save the current committed root and optional hot pool."""

    if not isinstance(runtime, VolatileTensorRuntime):
        raise RuntimeCheckpointError("runtime must be VolatileTensorRuntime@1")
    target = Path(path)
    if not target.name.endswith(".runtime.arti.st"):
        raise RuntimeCheckpointError("runtime checkpoint must end in .runtime.arti.st")
    target.parent.mkdir(parents=True, exist_ok=True)
    path_lock = _checkpoint_path_lock(target)
    path_lock.acquire()
    runtime._lock.acquire()
    try:
        try:
            root = runtime._resolve_snapshot(snapshot)
        except (TensorTransactionContractError, TypeError) as error:
            raise RuntimeCheckpointError("snapshot does not belong to runtime") from error
        if root is not runtime._published.root:
            raise RuntimeCheckpointError("checkpoint requires the current committed root")
        tensors: dict[str, Tensor] = {}
        pages: list[dict[str, object]] = []
        for index, key in enumerate(sorted(root.pages)):
            page = root.pages[key]
            tensor_key = f"page.{index:06d}"
            tensors[tensor_key] = page.value.detach().clone()
            pages.append(
                {
                    "tensor_key": tensor_key,
                    "ref": page.ref.to_dict(),
                    "provenance_fingerprint": page.provenance_fingerprint,
                }
            )
        binding_payload = []
        for binding in bindings:
            if not isinstance(binding, ExternalTensorBinding):
                raise RuntimeCheckpointError(
                    "bindings must contain ExternalTensorBinding@1"
                )
            bound_page = root.pages.get(binding.tensor_ref.key)
            if (
                binding.store_instance_id != root.store_instance_id
                or binding.world_id != root.world_id
                or binding.abi_fingerprint != root.abi_fingerprint
                or binding.root_id != root.root_id
                or binding.root_epoch != root.epoch
                or binding.root_fingerprint != root.fingerprint
                or bound_page is None
                or bound_page.ref != binding.tensor_ref
            ):
                raise RuntimeCheckpointError("binding does not match checkpoint root page")
            binding_payload.append(binding.to_dict())
        if resident is not None and not binding_payload:
            raise RuntimeCheckpointError(
                "resident checkpoint requires at least one root-bound external binding"
            )

        manifest: dict[str, object] = {
        "schema": RUNTIME_CHECKPOINT_SCHEMA,
        "scope": "same-abi-clean-restart",
        "unsupported": [
            "wal",
            "crash-consistent-incremental-checkpoint",
            "cold-page-migration",
            "cross-process-fencing",
            "cross-model-abi-migration",
            "distributed-agency",
        ],
        "root": {
            "store_instance_id": root.store_instance_id,
            "world_id": root.world_id,
            "root_id": root.root_id,
            "epoch": root.epoch,
            "abi_fingerprint": root.abi_fingerprint,
            "provenance_head": root.provenance_head,
            "root_fingerprint": root.fingerprint,
        },
        "pages": pages,
        "idempotency_receipts": [
            {
                "receipt": asdict(runtime._published.idempotency_index[key]),
                "abi_fingerprint": root.abi_fingerprint,
                "checkpoint_root_fingerprint": root.fingerprint,
            }
            for key in sorted(runtime._published.idempotency_index)
        ],
        "external_bindings": binding_payload,
        "resident": (
            None
            if resident is None
            else _serialize_resident(
                resident,
                tensors,
                binding_fingerprints=tuple(
                    binding.fingerprint for binding in bindings
                ),
            )
        ),
        }
        manifest_fingerprint = _fingerprint(manifest)
        temporary = target.with_name(f".{target.name}.{uuid.uuid4().hex}.tmp")
        try:
            save_file(
                tensors,
                str(temporary),
                metadata={
                    "schema": RUNTIME_CHECKPOINT_SCHEMA,
                    "manifest": _canonical_json(manifest),
                    "manifest_sha256": manifest_fingerprint,
                },
            )
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
        return RuntimeCheckpointReceipt(
            path=str(target),
            artifact_sha256=_file_hash(target),
            manifest_fingerprint=manifest_fingerprint,
            root_fingerprint=root.fingerprint,
            root_epoch=root.epoch,
            page_count=len(pages),
            includes_resident_pool=resident is not None,
        )
    finally:
        runtime._lock.release()
        path_lock.release()


def _require_mapping(value: object, name: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise RuntimeCheckpointError(f"{name} must be an object")
    return value


def _load_tensor(
    tensors: dict[str, Tensor],
    descriptor: dict[str, object],
    *,
    name: str,
) -> Tensor:
    tensor_key = descriptor.get("tensor_key")
    if not isinstance(tensor_key, str) or tensor_key not in tensors:
        raise RuntimeCheckpointError(f"missing tensor payload for {name}")
    value = tensors[tensor_key].detach().to("cpu").contiguous()
    if _tensor_descriptor(value) != {
        "dtype": descriptor.get("dtype"),
        "shape": descriptor.get("shape"),
        "content_sha256": descriptor.get("content_sha256"),
    }:
        raise RuntimeCheckpointError(f"tensor descriptor mismatch for {name}")
    return value


def _restore_runtime(manifest: dict[str, object], tensors: dict[str, Tensor]) -> VolatileTensorRuntime:
    root_data = _require_mapping(manifest.get("root"), "root")
    pages_data = manifest.get("pages")
    if not isinstance(pages_data, list):
        raise RuntimeCheckpointError("pages must be a list")
    pages: dict[str, _OwnedPage] = {}
    for item in pages_data:
        page_data = _require_mapping(item, "page")
        ref_data = _require_mapping(page_data.get("ref"), "page ref")
        ref = TensorRef(
            key=ref_data["key"],
            version=ref_data["version"],
            content_sha256=ref_data["content_sha256"],
            dtype=ref_data["dtype"],
            shape=tuple(ref_data["shape"]),
            layout=ref_data["layout"],
            device=ref_data["device"],
        )
        descriptor = {
            "tensor_key": page_data.get("tensor_key"),
            "dtype": ref.dtype,
            "shape": list(ref.shape),
            "content_sha256": ref.content_sha256,
        }
        value = _load_tensor(tensors, descriptor, name=f"page {ref.key}")
        provenance = page_data.get("provenance_fingerprint")
        try:
            provenance = _require_sha256(provenance, "page provenance fingerprint")
        except TensorTransactionContractError as error:
            raise RuntimeCheckpointError("page provenance fingerprint is invalid") from error
        if ref.key in pages:
            raise RuntimeCheckpointError("duplicate page key")
        pages[ref.key] = _OwnedPage(ref, value, provenance)
    frozen_pages = MappingProxyType(pages)
    expected_root = _root_fingerprint(
        store_instance_id=root_data["store_instance_id"],
        world_id=root_data["world_id"],
        root_id=root_data["root_id"],
        epoch=root_data["epoch"],
        abi_fingerprint=root_data["abi_fingerprint"],
        pages=frozen_pages,
        provenance_head=root_data["provenance_head"],
    )
    if expected_root != root_data.get("root_fingerprint"):
        raise RuntimeCheckpointError("root fingerprint mismatch")
    root = _WorldRoot(
        root_data["store_instance_id"],
        root_data["world_id"],
        root_data["root_id"],
        root_data["epoch"],
        root_data["abi_fingerprint"],
        frozen_pages,
        root_data["provenance_head"],
        expected_root,
    )
    receipts_data = manifest.get("idempotency_receipts")
    if not isinstance(receipts_data, list):
        raise RuntimeCheckpointError("idempotency_receipts must be a list")
    receipts: dict[str, CommitReceipt] = {}
    for item in receipts_data:
        wrapper = _require_mapping(item, "commit receipt wrapper")
        data = _require_mapping(wrapper.get("receipt"), "commit receipt")
        receipt = CommitReceipt(**data)
        content = dict(data)
        fingerprint = content.pop("receipt_fingerprint")
        if _fingerprint(content) != fingerprint:
            raise RuntimeCheckpointError("commit receipt fingerprint mismatch")
        if (
            wrapper.get("abi_fingerprint") != root.abi_fingerprint
            or wrapper.get("checkpoint_root_fingerprint") != root.fingerprint
            or receipt.store_instance_id != root.store_instance_id
            or receipt.world_id != root.world_id
            or receipt.base_epoch < 0
            or receipt.new_epoch <= receipt.base_epoch
            or receipt.new_epoch > root.epoch
            or (
                receipt.new_epoch == root.epoch
                and (
                    receipt.new_root_id != root.root_id
                    or receipt.provenance_head != root.provenance_head
                )
            )
        ):
            raise RuntimeCheckpointError("commit receipt does not belong to checkpoint root")
        if receipt.idempotency_key in receipts:
            raise RuntimeCheckpointError("duplicate idempotency receipt")
        receipts[receipt.idempotency_key] = receipt

    runtime = VolatileTensorRuntime.__new__(VolatileTensorRuntime)
    runtime._world_id = root.world_id
    runtime._store_instance_id = root.store_instance_id
    runtime._abi_fingerprint = root.abi_fingerprint
    runtime._owner_token = object()
    runtime._transaction_factory_token = object()
    runtime._snapshots = {}
    runtime._published = _PublishedState(root, MappingProxyType(receipts))
    runtime._lock = RLock()
    return runtime


def _restore_binding(data: dict[str, object]) -> ExternalTensorBinding:
    ref_data = _require_mapping(data.get("tensor_ref"), "binding tensor_ref")
    return ExternalTensorBinding(
        store_instance_id=data["store_instance_id"],
        world_id=data["world_id"],
        abi_fingerprint=data["abi_fingerprint"],
        root_id=data["root_id"],
        root_epoch=data["root_epoch"],
        root_fingerprint=data["root_fingerprint"],
        address_namespace=data["address_namespace"],
        partition_id=data["partition_id"],
        logical_id=data["logical_id"],
        tensor_ref=TensorRef(
            key=ref_data["key"],
            version=ref_data["version"],
            content_sha256=ref_data["content_sha256"],
            dtype=ref_data["dtype"],
            shape=tuple(ref_data["shape"]),
            layout=ref_data["layout"],
            device=ref_data["device"],
        ),
        role=data["role"],
        authority=TensorAuthority(data["authority"]),
        component_ref=data["component_ref"],
        component_config_fingerprint=data["component_config_fingerprint"],
        state_schema_ref=data["state_schema_ref"],
        producer_state_fingerprint=data["producer_state_fingerprint"],
        provenance_fingerprint=data["provenance_fingerprint"],
    )


def _restore_resident(
    data: object,
    tensors: dict[str, Tensor],
    device: torch.device | None,
    *,
    binding_fingerprints: Sequence[str],
) -> BoundHotPagePool | None:
    if data is None:
        return None
    if device is None or device.type != "cuda" or device.index is None:
        raise RuntimeCheckpointError("resident checkpoint requires an indexed CUDA device")
    resident = _require_mapping(data, "resident")
    claimed_snapshot = resident.get("authority_snapshot_fingerprint")
    snapshot_payload = {
        key: value
        for key, value in resident.items()
        if key != "authority_snapshot_fingerprint"
    }
    if (
        not isinstance(claimed_snapshot, str)
        or _fingerprint(snapshot_payload) != claimed_snapshot
    ):
        raise RuntimeCheckpointError("resident authority snapshot fingerprint mismatch")
    if resident.get("binding_fingerprints") != list(binding_fingerprints):
        raise RuntimeCheckpointError("resident binding fingerprints mismatch")
    pool_data = _require_mapping(resident.get("pool"), "resident pool")
    loaded = {
        name: _load_tensor(
            tensors,
            _require_mapping(pool_data.get(name), f"resident {name}"),
            name=f"resident {name}",
        ).to(device)
        for name in ("value", "validity", "generation", "version")
    }
    pool = HotPagePool(
        loaded["value"],
        validity=loaded["validity"],
        generation=loaded["generation"],
        version=loaded["version"],
    )
    bucket_data = _require_mapping(resident.get("bucket"), "resident bucket")
    dtype_by_name = {
        "torch.float16": torch.float16,
        "torch.bfloat16": torch.bfloat16,
        "torch.float32": torch.float32,
    }
    try:
        dtype = dtype_by_name[bucket_data["dtype"]]
    except KeyError as error:
        raise RuntimeCheckpointError("unsupported resident dtype") from error
    bucket = FixedResidentBucket(
        batch_size=bucket_data["batch_size"],
        workset_slots=bucket_data["workset_slots"],
        feature_dim=bucket_data["feature_dim"],
        dtype=dtype,
        device=device,
        refine_steps=bucket_data["refine_steps"],
    )
    refs_data = _require_mapping(resident.get("refs"), "resident refs")
    refs = FixedPageRefs(
        logical_slot=torch.tensor(refs_data["logical_slot"], dtype=torch.int64),
        page_id=torch.tensor(refs_data["page_id"], dtype=torch.int64),
        offset=torch.tensor(refs_data["offset"], dtype=torch.int64),
        expected_generation=torch.tensor(
            refs_data["expected_generation"], dtype=torch.int64
        ),
        read_mask=torch.tensor(refs_data["read_mask"], dtype=torch.bool),
        write_mask=torch.tensor(refs_data["write_mask"], dtype=torch.bool),
        commit_mask=torch.tensor(refs_data["commit_mask"], dtype=torch.bool),
    )
    return bind_hot_page_pool(pool, bucket, refs)


def _load_runtime_checkpoint_locked(
    path: str | Path,
    *,
    expected_abi_fingerprint: str,
    expected_component_refs: Sequence[str] = (),
    resident_device: str | torch.device | None = None,
) -> RestoredRuntimeCheckpoint:
    """Verify and restore a fresh same-ABI runtime checkpoint."""

    target = Path(path)
    try:
        with safe_open(str(target), framework="pt", device="cpu") as handle:
            metadata = handle.metadata()
    except Exception as error:
        raise RuntimeCheckpointError("runtime checkpoint cannot be opened") from error
    if metadata.get("schema") != RUNTIME_CHECKPOINT_SCHEMA:
        raise RuntimeCheckpointError("unsupported runtime checkpoint schema")
    raw_manifest = metadata.get("manifest")
    if not isinstance(raw_manifest, str):
        raise RuntimeCheckpointError("runtime checkpoint manifest is missing")
    try:
        manifest = json.loads(raw_manifest)
    except json.JSONDecodeError as error:
        raise RuntimeCheckpointError("runtime checkpoint manifest is invalid JSON") from error
    if not isinstance(manifest, dict):
        raise RuntimeCheckpointError("runtime checkpoint manifest must be an object")
    manifest_fingerprint = _fingerprint(manifest)
    if metadata.get("manifest_sha256") != manifest_fingerprint:
        raise RuntimeCheckpointError("runtime checkpoint manifest fingerprint mismatch")
    if manifest.get("schema") != RUNTIME_CHECKPOINT_SCHEMA:
        raise RuntimeCheckpointError("manifest schema mismatch")
    root_data = _require_mapping(manifest.get("root"), "root")
    if root_data.get("abi_fingerprint") != expected_abi_fingerprint:
        raise RuntimeCheckpointError("runtime checkpoint ABI mismatch")
    bindings_data = manifest.get("external_bindings")
    if not isinstance(bindings_data, list):
        raise RuntimeCheckpointError("external_bindings must be a list")
    try:
        bindings = tuple(
            _restore_binding(_require_mapping(item, "binding"))
            for item in bindings_data
        )
    except RuntimeCheckpointError:
        raise
    except (KeyError, TypeError, ValueError, TensorTransactionContractError) as error:
        raise RuntimeCheckpointError("runtime checkpoint binding is malformed") from error
    actual_refs = tuple(binding.component_ref for binding in bindings)
    if tuple(expected_component_refs) != actual_refs:
        raise RuntimeCheckpointError("runtime checkpoint component refs mismatch")
    try:
        tensors = load_file(str(target), device="cpu")
    except Exception as error:
        raise RuntimeCheckpointError("runtime checkpoint tensor payload is invalid") from error
    try:
        runtime = _restore_runtime(manifest, tensors)
    except RuntimeCheckpointError:
        raise
    except (KeyError, TypeError, ValueError, TensorTransactionContractError) as error:
        raise RuntimeCheckpointError("runtime checkpoint manifest is malformed") from error
    snapshot = runtime.snapshot()
    root = runtime._published.root
    for binding in bindings:
        page = root.pages.get(binding.tensor_ref.key)
        if (
            binding.store_instance_id != root.store_instance_id
            or binding.world_id != root.world_id
            or binding.abi_fingerprint != root.abi_fingerprint
            or binding.root_id != root.root_id
            or binding.root_epoch != root.epoch
            or binding.root_fingerprint != root.fingerprint
            or page is None
            or page.ref != binding.tensor_ref
        ):
            raise RuntimeCheckpointError(
                "runtime checkpoint binding does not match restored root page"
            )
    resident = _restore_resident(
        manifest.get("resident"),
        tensors,
        None if resident_device is None else torch.device(resident_device),
        binding_fingerprints=tuple(binding.fingerprint for binding in bindings),
    )
    return RestoredRuntimeCheckpoint(
        runtime=runtime,
        snapshot=snapshot,
        resident=resident,
        bindings=bindings,
        manifest_fingerprint=manifest_fingerprint,
        artifact_sha256=_file_hash(target),
    )


def load_runtime_checkpoint(
    path: str | Path,
    *,
    expected_abi_fingerprint: str,
    expected_component_refs: Sequence[str] = (),
    resident_device: str | torch.device | None = None,
) -> RestoredRuntimeCheckpoint:
    """Verify and restore one process-local linearized checkpoint view."""

    target = Path(path)
    with _checkpoint_path_lock(target):
        return _load_runtime_checkpoint_locked(
            target,
            expected_abi_fingerprint=expected_abi_fingerprint,
            expected_component_refs=expected_component_refs,
            resident_device=resident_device,
        )


__all__ = [
    "RUNTIME_CHECKPOINT_SCHEMA",
    "RestoredRuntimeCheckpoint",
    "RuntimeCheckpointError",
    "RuntimeCheckpointReceipt",
    "load_runtime_checkpoint",
    "save_runtime_checkpoint",
]
