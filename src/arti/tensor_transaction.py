"""Volatile, single-writer transactions for runtime-owned CPU tensors.

This module coordinates storage only. It does not execute Formula, Recall,
Fold, Updater, or any other neural operation.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
import weakref
from dataclasses import dataclass, field
from enum import Enum
from threading import RLock
from types import MappingProxyType
from typing import ClassVar, Mapping

import torch
from torch import Tensor


_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EMPTY_HASH = hashlib.sha256(b"").hexdigest()


class TensorTransactionError(RuntimeError):
    """Base error for the volatile transaction substrate."""


class TensorTransactionContractError(TensorTransactionError):
    """Raised when a transaction contract is malformed."""


class TensorOwnershipError(TensorTransactionContractError):
    """Raised when a tensor cannot become runtime-owned state."""


class TensorTransactionStateError(TensorTransactionError):
    """Raised when an operation is invalid for the transaction state."""


class TensorTransactionStatus(str, Enum):
    OPEN = "open"
    COMMITTED = "committed"
    CONFLICTED = "conflicted"
    ROLLED_BACK = "rolled_back"


class ConflictReason(str, Enum):
    STALE_ROOT = "stale_root"
    STALE_PAGE = "stale_page"
    IDEMPOTENCY_MISMATCH = "idempotency_mismatch"


@dataclass(frozen=True)
class PageConflict:
    """One page whose published reference differs from the base snapshot."""

    key: str
    expected_version: int | None
    current_version: int | None
    expected_content_sha256: str | None
    current_content_sha256: str | None
    _runtime_contract_ref: ClassVar[str] = "arti/page-conflict@1"


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _sha256_json(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise TensorTransactionContractError(f"{name} must be a canonical identifier")
    return value


def _require_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TensorTransactionContractError(f"{name} must be lowercase SHA-256 hex")
    return value


def _canonical_runtime_ref(reference: str) -> str:
    from .component_registry import canonical_contract_reference

    return canonical_contract_reference(reference)


def _own_tensor(value: Tensor, *, name: str) -> Tensor:
    if not isinstance(value, Tensor):
        raise TensorOwnershipError(f"{name} must be a Tensor")
    if value.device.type != "cpu":
        raise TensorOwnershipError(f"{name} must be a CPU Tensor in volatile runtime v1")
    if value.layout != torch.strided or not value.is_contiguous():
        raise TensorOwnershipError(f"{name} must be contiguous strided storage")
    if value.requires_grad:
        raise TensorOwnershipError(f"{name} must not require gradients")
    if (value.is_floating_point() or value.is_complex()) and not bool(
        torch.isfinite(value).all()
    ):
        raise TensorOwnershipError(f"{name} must contain only finite values")
    return value.detach().clone(memory_format=torch.contiguous_format)


def _tensor_hash(value: Tensor) -> str:
    raw = value.view(torch.uint8).numpy().tobytes()
    descriptor = {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "layout": "strided",
        "bytes_sha256": hashlib.sha256(raw).hexdigest(),
    }
    return _sha256_json(descriptor)


@dataclass(frozen=True)
class TensorRef:
    """Versioned metadata for one logical tensor page."""

    key: str
    version: int
    content_sha256: str
    dtype: str
    shape: tuple[int, ...]
    layout: str = "strided"
    device: str = "cpu"
    _runtime_contract_ref: ClassVar[str] = "arti/tensor-ref@1"

    def __post_init__(self) -> None:
        _require_identifier(self.key, "TensorRef key")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version <= 0:
            raise TensorTransactionContractError("TensorRef version must be positive")
        _require_sha256(self.content_sha256, "TensorRef content_sha256")
        if not isinstance(self.dtype, str) or not self.dtype:
            raise TensorTransactionContractError("TensorRef dtype must be non-empty")
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in self.shape):
            raise TensorTransactionContractError("TensorRef shape is invalid")
        if self.layout != "strided" or self.device != "cpu":
            raise TensorTransactionContractError("volatile TensorRef v1 is CPU strided only")

    def to_dict(self) -> dict[str, object]:
        return {
            "ref": _canonical_runtime_ref(self._runtime_contract_ref),
            "key": self.key,
            "version": self.version,
            "content_sha256": self.content_sha256,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "layout": self.layout,
            "device": self.device,
        }


@dataclass(frozen=True)
class TensorRead:
    """Caller-owned tensor copy plus its observed reference."""

    ref: TensorRef
    value: Tensor = field(repr=False, compare=False)
    _runtime_contract_ref: ClassVar[str] = "arti/tensor-read@1"


@dataclass(frozen=True)
class _OwnedPage:
    ref: TensorRef
    value: Tensor = field(repr=False, compare=False)
    provenance_fingerprint: str


@dataclass(frozen=True)
class _WorldRoot:
    store_instance_id: str
    world_id: str
    root_id: str
    epoch: int
    abi_fingerprint: str
    pages: Mapping[str, _OwnedPage] = field(repr=False, compare=False)
    provenance_head: str
    fingerprint: str


@dataclass(frozen=True)
class TensorSnapshot:
    """Opaque descriptor for one runtime-registered world root."""

    store_instance_id: str
    world_id: str
    root_id: str
    epoch: int
    abi_fingerprint: str
    root_fingerprint: str
    page_refs: tuple[TensorRef, ...]
    _owner_token: object = field(repr=False, compare=False)
    _runtime_contract_ref: ClassVar[str] = "arti/tensor-snapshot@1"


@dataclass(frozen=True)
class CommitReceipt:
    """Successful atomic publication receipt."""

    store_instance_id: str
    world_id: str
    transaction_id: str
    branch_id: str
    idempotency_key: str
    request_fingerprint: str
    base_root_id: str
    base_epoch: int
    new_root_id: str
    new_epoch: int
    read_set_digest: str
    write_set_digest: str
    provenance_head: str
    receipt_fingerprint: str
    _runtime_contract_ref: ClassVar[str] = "arti/commit-receipt@1"


@dataclass(frozen=True)
class ConflictReceipt:
    """Fail-closed receipt for a rejected publication."""

    store_instance_id: str
    world_id: str
    transaction_id: str
    branch_id: str
    idempotency_key: str
    request_fingerprint: str
    reason: ConflictReason
    base_root_id: str
    base_epoch: int
    current_root_id: str
    current_epoch: int
    page_conflicts: tuple[PageConflict, ...]
    _runtime_contract_ref: ClassVar[str] = "arti/conflict-receipt@1"


@dataclass(frozen=True)
class RollbackReceipt:
    """Receipt proving that a private overlay was discarded."""

    transaction_id: str
    branch_id: str
    base_root_id: str
    base_epoch: int
    _runtime_contract_ref: ClassVar[str] = "arti/rollback-receipt@1"


@dataclass(frozen=True)
class _StagedWrite:
    key: str
    expected_version: int | None
    base_hash: str | None
    value: Tensor = field(repr=False, compare=False)
    content_sha256: str
    provenance_fingerprint: str

    def descriptor(self) -> dict[str, object]:
        return {
            "key": self.key,
            "expected_version": self.expected_version,
            "base_hash": self.base_hash,
            "content_sha256": self.content_sha256,
            "provenance_fingerprint": self.provenance_fingerprint,
            "dtype": str(self.value.dtype),
            "shape": list(self.value.shape),
        }


@dataclass(frozen=True)
class _PublishedState:
    root: _WorldRoot
    idempotency_index: Mapping[str, CommitReceipt]


def _root_fingerprint(
    *,
    store_instance_id: str,
    world_id: str,
    root_id: str,
    epoch: int,
    abi_fingerprint: str,
    pages: Mapping[str, _OwnedPage],
    provenance_head: str,
) -> str:
    return _sha256_json(
        {
            "store_instance_id": store_instance_id,
            "world_id": world_id,
            "root_id": root_id,
            "epoch": epoch,
            "abi_fingerprint": abi_fingerprint,
            "pages": [
                {
                    "ref": pages[key].ref.to_dict(),
                    "provenance_fingerprint": pages[key].provenance_fingerprint,
                }
                for key in sorted(pages)
            ],
            "provenance_head": provenance_head,
        }
    )


class VolatileTensorRuntime:
    """Single-process, CPU-only, volatile tensor transaction coordinator."""

    _runtime_contract_ref: ClassVar[str] = "arti/volatile-tensor-runtime@1"

    def __init__(
        self,
        initial: Mapping[str, Tensor] | None = None,
        *,
        world_id: str = "default",
        store_instance_id: str | None = None,
        abi_fingerprint: str = _EMPTY_HASH,
        provenance_fingerprint: str = _EMPTY_HASH,
    ) -> None:
        self._world_id = _require_identifier(world_id, "world_id")
        self._store_instance_id = _require_identifier(
            store_instance_id or f"store-{uuid.uuid4().hex}", "store_instance_id"
        )
        self._abi_fingerprint = _require_sha256(abi_fingerprint, "abi_fingerprint")
        self._owner_token = object()
        self._transaction_factory_token = object()
        self._snapshots: dict[
            int, tuple[weakref.ReferenceType[TensorSnapshot], _WorldRoot]
        ] = {}
        provenance = _require_sha256(provenance_fingerprint, "provenance_fingerprint")
        pages: dict[str, _OwnedPage] = {}
        for key, raw in (initial or {}).items():
            _require_identifier(key, "page key")
            owned = _own_tensor(raw, name=f"initial page {key!r}")
            reference = TensorRef(
                key=key,
                version=1,
                content_sha256=_tensor_hash(owned),
                dtype=str(owned.dtype),
                shape=tuple(owned.shape),
            )
            pages[key] = _OwnedPage(reference, owned, provenance)
        root_id = _sha256_json(
            {
                "store_instance_id": self.store_instance_id,
                "world_id": self.world_id,
                "initial_pages": [pages[key].ref.to_dict() for key in sorted(pages)],
            }
        )
        frozen_pages = MappingProxyType(pages)
        root = _WorldRoot(
            self.store_instance_id,
            self.world_id,
            root_id,
            0,
            self.abi_fingerprint,
            frozen_pages,
            provenance,
            _root_fingerprint(
                store_instance_id=self.store_instance_id,
                world_id=self.world_id,
                root_id=root_id,
                epoch=0,
                abi_fingerprint=self.abi_fingerprint,
                pages=frozen_pages,
                provenance_head=provenance,
            ),
        )
        self._published = _PublishedState(root, MappingProxyType({}))
        self._lock = RLock()

    @property
    def world_id(self) -> str:
        return self._world_id

    @property
    def store_instance_id(self) -> str:
        return self._store_instance_id

    @property
    def abi_fingerprint(self) -> str:
        return self._abi_fingerprint

    def snapshot(self) -> TensorSnapshot:
        root = self._published.root
        snapshot = TensorSnapshot(
            store_instance_id=root.store_instance_id,
            world_id=root.world_id,
            root_id=root.root_id,
            epoch=root.epoch,
            abi_fingerprint=root.abi_fingerprint,
            root_fingerprint=root.fingerprint,
            page_refs=tuple(root.pages[key].ref for key in sorted(root.pages)),
            _owner_token=self._owner_token,
        )
        snapshot_id = id(snapshot)

        def discard(reference: weakref.ReferenceType[TensorSnapshot]) -> None:
            current = self._snapshots.get(snapshot_id)
            if current is not None and current[0] is reference:
                self._snapshots.pop(snapshot_id, None)

        self._snapshots[snapshot_id] = (weakref.ref(snapshot, discard), root)
        return snapshot

    def _resolve_snapshot(self, snapshot: TensorSnapshot) -> _WorldRoot:
        if not isinstance(snapshot, TensorSnapshot) or snapshot._owner_token is not self._owner_token:
            raise TensorTransactionContractError("snapshot belongs to another runtime")
        registered = self._snapshots.get(id(snapshot))
        if registered is None or registered[0]() is not snapshot:
            raise TensorTransactionContractError("snapshot is not registered by this runtime")
        root = registered[1]
        descriptor_matches = (
            snapshot.store_instance_id == root.store_instance_id
            and snapshot.world_id == root.world_id
            and snapshot.root_id == root.root_id
            and snapshot.epoch == root.epoch
            and snapshot.abi_fingerprint == root.abi_fingerprint
            and snapshot.root_fingerprint == root.fingerprint
            and snapshot.page_refs == tuple(root.pages[key].ref for key in sorted(root.pages))
        )
        if not descriptor_matches:
            raise TensorTransactionContractError("snapshot descriptor does not match its registered root")
        return root

    def read(self, snapshot: TensorSnapshot, key: str) -> TensorRead:
        """Return a caller-owned clone from one validated snapshot."""

        root = self._resolve_snapshot(snapshot)
        _require_identifier(key, "page key")
        try:
            page = root.pages[key]
        except KeyError as error:
            raise KeyError(f"unknown tensor page: {key}") from error
        return TensorRead(page.ref, page.value.detach().clone())

    def begin(
        self,
        snapshot: TensorSnapshot,
        *,
        transaction_id: str,
        branch_id: str,
    ) -> "TensorTransaction":
        root = self._resolve_snapshot(snapshot)
        return TensorTransaction(
            self,
            snapshot,
            root,
            _factory_token=self._transaction_factory_token,
            transaction_id=_require_identifier(transaction_id, "transaction_id"),
            branch_id=_require_identifier(branch_id, "branch_id"),
        )

    def replay(
        self,
        transaction: "TensorTransaction",
        *,
        idempotency_key: str,
    ) -> CommitReceipt | ConflictReceipt:
        """Replay one immutable storage proposal through the idempotency index."""

        if not isinstance(transaction, TensorTransaction) or transaction._runtime is not self:
            raise TensorTransactionContractError("replay transaction belongs to another runtime")
        return transaction.commit(idempotency_key=idempotency_key)

    def committed_receipt(self, idempotency_key: str) -> CommitReceipt | None:
        """Return one immutable committed receipt without replaying a proposal."""

        key = _require_identifier(idempotency_key, "idempotency_key")
        return self._published.idempotency_index.get(key)

    def _commit(
        self, transaction: "TensorTransaction", idempotency_key: str
    ) -> CommitReceipt | ConflictReceipt:
        request = transaction.request_fingerprint
        with self._lock:
            published = self._published
            previous = published.idempotency_index.get(idempotency_key)
            if previous is not None:
                if previous.request_fingerprint == request:
                    return previous
                return transaction._conflict(
                    idempotency_key,
                    request,
                    ConflictReason.IDEMPOTENCY_MISMATCH,
                    published.root,
                )
            base = transaction._snapshot
            current = published.root
            if (
                current.root_id != base.root_id
                or current.epoch != base.epoch
                or current.fingerprint != base.root_fingerprint
            ):
                return transaction._conflict(
                    idempotency_key, request, ConflictReason.STALE_ROOT, current
                )
            for key, observed in transaction._read_set.items():
                actual = current.pages.get(key)
                if actual is None or actual.ref != observed:
                    return transaction._conflict(
                        idempotency_key, request, ConflictReason.STALE_PAGE, current
                    )

            pages = dict(current.pages)
            for key, staged in transaction._write_set.items():
                old = pages.get(key)
                actual_version = None if old is None else old.ref.version
                actual_hash = None if old is None else old.ref.content_sha256
                if (
                    actual_version != staged.expected_version
                    or actual_hash != staged.base_hash
                ):
                    return transaction._conflict(
                        idempotency_key, request, ConflictReason.STALE_PAGE, current
                    )
                owned = staged.value.detach().clone()
                pages[key] = _OwnedPage(
                    TensorRef(
                        key=key,
                        version=1 if old is None else old.ref.version + 1,
                        content_sha256=staged.content_sha256,
                        dtype=str(owned.dtype),
                        shape=tuple(owned.shape),
                    ),
                    owned,
                    staged.provenance_fingerprint,
                )

            new_epoch = current.epoch + 1
            new_root_id = _sha256_json(
                {"base_root_id": current.root_id, "epoch": new_epoch, "request": request}
            )
            provenance_head = _sha256_json(
                {
                    "previous": current.provenance_head,
                    "request": request,
                    "writes": [
                        transaction._write_set[key].provenance_fingerprint
                        for key in sorted(transaction._write_set)
                    ],
                }
            )
            frozen_pages = MappingProxyType(pages)
            new_root = _WorldRoot(
                current.store_instance_id,
                current.world_id,
                new_root_id,
                new_epoch,
                current.abi_fingerprint,
                frozen_pages,
                provenance_head,
                _root_fingerprint(
                    store_instance_id=current.store_instance_id,
                    world_id=current.world_id,
                    root_id=new_root_id,
                    epoch=new_epoch,
                    abi_fingerprint=current.abi_fingerprint,
                    pages=frozen_pages,
                    provenance_head=provenance_head,
                ),
            )
            receipt_content = {
                "store_instance_id": self.store_instance_id,
                "world_id": self.world_id,
                "transaction_id": transaction.transaction_id,
                "branch_id": transaction.branch_id,
                "idempotency_key": idempotency_key,
                "request_fingerprint": request,
                "base_root_id": base.root_id,
                "base_epoch": base.epoch,
                "new_root_id": new_root.root_id,
                "new_epoch": new_root.epoch,
                "read_set_digest": transaction.read_set_digest,
                "write_set_digest": transaction.write_set_digest,
                "provenance_head": provenance_head,
            }
            receipt = CommitReceipt(
                **receipt_content,
                receipt_fingerprint=_sha256_json(receipt_content),
            )
            index = dict(published.idempotency_index)
            index[idempotency_key] = receipt
            self._published = _PublishedState(new_root, MappingProxyType(index))
            return receipt


class TensorTransaction:
    """Private COW overlay rooted at one immutable TensorSnapshot."""

    _runtime_contract_ref: ClassVar[str] = "arti/tensor-transaction@1"

    def __init__(
        self,
        runtime: VolatileTensorRuntime,
        snapshot: TensorSnapshot,
        base_root: _WorldRoot,
        *,
        _factory_token: object,
        transaction_id: str,
        branch_id: str,
    ) -> None:
        if _factory_token is not runtime._transaction_factory_token:
            raise TensorTransactionContractError(
                "transactions must be created by VolatileTensorRuntime.begin"
            )
        if runtime._resolve_snapshot(snapshot) is not base_root:
            raise TensorTransactionContractError(
                "transaction base root does not match the registered snapshot"
            )
        self._runtime = runtime
        self._snapshot = snapshot
        self._base_root = base_root
        self.transaction_id = transaction_id
        self.branch_id = branch_id
        self.status = TensorTransactionStatus.OPEN
        self._read_set: dict[str, TensorRef] = {}
        self._write_set: dict[str, _StagedWrite] = {}
        self._receipt: CommitReceipt | ConflictReceipt | RollbackReceipt | None = None

    def _require_open(self) -> None:
        if self.status is not TensorTransactionStatus.OPEN:
            raise TensorTransactionStateError(f"transaction is {self.status.value}")

    @property
    def read_set_digest(self) -> str:
        return _sha256_json(
            [self._read_set[key].to_dict() for key in sorted(self._read_set)]
        )

    @property
    def write_set_digest(self) -> str:
        return _sha256_json(
            [self._write_set[key].descriptor() for key in sorted(self._write_set)]
        )

    @property
    def request_fingerprint(self) -> str:
        return _sha256_json(
            {
                "transaction_id": self.transaction_id,
                "branch_id": self.branch_id,
                "base_root_id": self._snapshot.root_id,
                "base_epoch": self._snapshot.epoch,
                "read_set_digest": self.read_set_digest,
                "write_set_digest": self.write_set_digest,
            }
        )

    def read(self, key: str) -> TensorRead:
        self._require_open()
        _require_identifier(key, "page key")
        staged = self._write_set.get(key)
        if staged is not None:
            version = 1 if staged.expected_version is None else staged.expected_version + 1
            return TensorRead(
                TensorRef(
                    key,
                    version,
                    staged.content_sha256,
                    str(staged.value.dtype),
                    tuple(staged.value.shape),
                ),
                staged.value.detach().clone(),
            )
        observed = self._runtime.read(self._snapshot, key)
        self._read_set.setdefault(key, observed.ref)
        return observed

    def stage(
        self,
        key: str,
        value: Tensor,
        *,
        expected_version: int | None,
        provenance_fingerprint: str,
    ) -> None:
        self._require_open()
        _require_identifier(key, "page key")
        provenance = _require_sha256(provenance_fingerprint, "provenance_fingerprint")
        page = self._base_root.pages.get(key)
        if page is None:
            if expected_version is not None:
                raise TensorTransactionContractError(
                    "new pages require expected_version=None"
                )
            base_hash = None
        else:
            if expected_version != page.ref.version:
                raise TensorTransactionContractError(
                    "existing pages require their exact observed version"
                )
            base_hash = page.ref.content_sha256
            self._read_set.setdefault(key, page.ref)
        owned = _own_tensor(value, name=f"staged page {key!r}")
        if page is not None and (
            str(owned.dtype) != page.ref.dtype or tuple(owned.shape) != page.ref.shape
        ):
            raise TensorTransactionContractError(
                "existing pages require the same dtype and shape"
            )
        self._write_set[key] = _StagedWrite(
            key,
            expected_version,
            base_hash,
            owned,
            _tensor_hash(owned),
            provenance,
        )

    def commit(self, *, idempotency_key: str) -> CommitReceipt | ConflictReceipt:
        key = _require_identifier(idempotency_key, "idempotency_key")
        if self.status is TensorTransactionStatus.COMMITTED:
            if isinstance(self._receipt, CommitReceipt) and self._receipt.idempotency_key == key:
                return self._receipt
            raise TensorTransactionStateError("committed transaction used another idempotency key")
        self._require_open()
        if not self._write_set:
            raise TensorTransactionContractError("cannot commit an empty write set")
        receipt = self._runtime._commit(self, key)
        self._receipt = receipt
        self.status = (
            TensorTransactionStatus.COMMITTED
            if isinstance(receipt, CommitReceipt)
            else TensorTransactionStatus.CONFLICTED
        )
        if isinstance(receipt, ConflictReceipt):
            self._read_set.clear()
            self._write_set.clear()
        return receipt

    def _conflict(
        self,
        idempotency_key: str,
        request_fingerprint: str,
        reason: ConflictReason,
        current: _WorldRoot,
    ) -> ConflictReceipt:
        base_pages = self._base_root.pages
        changed_pages: list[PageConflict] = []
        for key in sorted(set(base_pages) | set(current.pages)):
            expected = base_pages.get(key)
            actual = current.pages.get(key)
            expected_ref = None if expected is None else expected.ref
            actual_ref = None if actual is None else actual.ref
            if expected_ref != actual_ref:
                changed_pages.append(
                    PageConflict(
                        key=key,
                        expected_version=None if expected_ref is None else expected_ref.version,
                        current_version=None if actual_ref is None else actual_ref.version,
                        expected_content_sha256=(
                            None if expected_ref is None else expected_ref.content_sha256
                        ),
                        current_content_sha256=(
                            None if actual_ref is None else actual_ref.content_sha256
                        ),
                    )
                )
        return ConflictReceipt(
            self._runtime.store_instance_id,
            self._runtime.world_id,
            self.transaction_id,
            self.branch_id,
            idempotency_key,
            request_fingerprint,
            reason,
            self._snapshot.root_id,
            self._snapshot.epoch,
            current.root_id,
            current.epoch,
            tuple(changed_pages),
        )

    def rollback(self) -> RollbackReceipt:
        if self.status is TensorTransactionStatus.ROLLED_BACK:
            assert isinstance(self._receipt, RollbackReceipt)
            return self._receipt
        self._require_open()
        self._read_set.clear()
        self._write_set.clear()
        receipt = RollbackReceipt(
            self.transaction_id,
            self.branch_id,
            self._snapshot.root_id,
            self._snapshot.epoch,
        )
        self._receipt = receipt
        self.status = TensorTransactionStatus.ROLLED_BACK
        return receipt


__all__ = [
    "CommitReceipt",
    "ConflictReason",
    "ConflictReceipt",
    "PageConflict",
    "RollbackReceipt",
    "TensorOwnershipError",
    "TensorRead",
    "TensorRef",
    "TensorSnapshot",
    "TensorTransactionContractError",
    "TensorTransactionError",
    "TensorTransactionStateError",
    "TensorTransactionStatus",
    "VolatileTensorRuntime",
]
