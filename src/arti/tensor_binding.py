"""Versioned host bindings between ARTI tensor results and volatile storage."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar, Sequence

import torch
from torch import Tensor

from .reversible_topology import FoldRecord
from .tensor_transaction import (
    TensorOwnershipError,
    TensorRead,
    TensorRef,
    TensorSnapshot,
    TensorTransaction,
    TensorTransactionContractError,
    VolatileTensorRuntime,
)


_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _require_identifier(value: str, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise TensorTransactionContractError(f"{name} must be a canonical identifier")
    return value


def _require_component_ref(value: str, name: str) -> str:
    """Accept an input declaration, but retain only a full component address."""

    try:
        from .component_registry import ComponentRef, canonical_contract_reference

        return ComponentRef.parse(canonical_contract_reference(value)).reference
    except (TypeError, ValueError) as error:
        raise TensorTransactionContractError(
            f"{name} must resolve to a full component contract address"
        ) from error


def _require_state_schema_ref(value: str, name: str) -> str:
    """Normalize a state schema to its immutable contract address."""

    return _require_component_ref(value, name)


def _require_sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TensorTransactionContractError(f"{name} must be lowercase SHA-256 hex")
    return value


def _owned_candidate(value: Tensor) -> Tensor:
    if not isinstance(value, Tensor):
        raise TensorOwnershipError("proposal value must be a Tensor")
    if value.device.type != "cpu" or value.layout != torch.strided or not value.is_contiguous():
        raise TensorOwnershipError("proposal value must be contiguous CPU strided storage")
    if value.requires_grad:
        raise TensorOwnershipError("proposal value must not require gradients")
    if (value.is_floating_point() or value.is_complex()) and not bool(
        torch.isfinite(value).all()
    ):
        raise TensorOwnershipError("proposal value must contain only finite values")
    return value.detach().clone(memory_format=torch.contiguous_format)


def _tensor_fingerprint(value: Tensor) -> str:
    raw = value.view(torch.uint8).numpy().tobytes()
    return _fingerprint(
        {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "bytes_sha256": hashlib.sha256(raw).hexdigest(),
        }
    )


class ProposalSemantics(str, Enum):
    """How a candidate relates to the page it may replace."""

    COMPLETE_NEXT_STATE = "complete_next_state"
    DELTA = "delta"


class TensorAuthority(str, Enum):
    """Host authority granted to one exact tensor binding."""

    READ_ONLY = "read_only"
    READ_WRITE = "read_write"


@dataclass(frozen=True)
class ExternalTensorBinding:
    """Exact host binding for one tensor page at one immutable snapshot."""

    store_instance_id: str
    world_id: str
    abi_fingerprint: str
    root_id: str
    root_epoch: int
    root_fingerprint: str
    address_namespace: str
    partition_id: str
    logical_id: str
    tensor_ref: TensorRef
    role: str
    authority: TensorAuthority
    component_ref: str
    component_config_fingerprint: str
    state_schema_ref: str
    producer_state_fingerprint: str
    provenance_fingerprint: str
    _runtime_contract_ref: ClassVar[str] = "arti/external-binding@1"

    def __post_init__(self) -> None:
        for value, name in (
            (self.store_instance_id, "store_instance_id"),
            (self.world_id, "world_id"),
            (self.address_namespace, "address_namespace"),
            (self.partition_id, "partition_id"),
            (self.logical_id, "logical_id"),
            (self.role, "role"),
        ):
            _require_identifier(value, name)
        for value, name in (
            (self.abi_fingerprint, "abi_fingerprint"),
            (self.root_id, "root_id"),
            (self.root_fingerprint, "root_fingerprint"),
            (self.component_config_fingerprint, "component_config_fingerprint"),
            (self.producer_state_fingerprint, "producer_state_fingerprint"),
            (self.provenance_fingerprint, "provenance_fingerprint"),
        ):
            _require_sha256(value, name)
        object.__setattr__(
            self, "component_ref", _require_component_ref(self.component_ref, "component_ref")
        )
        object.__setattr__(
            self,
            "state_schema_ref",
            _require_state_schema_ref(self.state_schema_ref, "state_schema_ref"),
        )
        if isinstance(self.root_epoch, bool) or not isinstance(self.root_epoch, int) or self.root_epoch < 0:
            raise TensorTransactionContractError("root_epoch must be a non-negative integer")
        if not isinstance(self.tensor_ref, TensorRef):
            raise TensorTransactionContractError("tensor_ref must be a TensorRef")
        if not isinstance(self.authority, TensorAuthority):
            raise TensorTransactionContractError("authority must be TensorAuthority")

    def to_dict(self) -> dict[str, object]:
        return {
            "ref": _require_component_ref(self._runtime_contract_ref, "binding_ref"),
            "store_instance_id": self.store_instance_id,
            "world_id": self.world_id,
            "abi_fingerprint": self.abi_fingerprint,
            "root_id": self.root_id,
            "root_epoch": self.root_epoch,
            "root_fingerprint": self.root_fingerprint,
            "address_namespace": self.address_namespace,
            "partition_id": self.partition_id,
            "logical_id": self.logical_id,
            "tensor_ref": self.tensor_ref.to_dict(),
            "role": self.role,
            "authority": self.authority.value,
            "component_ref": self.component_ref,
            "component_config_fingerprint": self.component_config_fingerprint,
            "state_schema_ref": self.state_schema_ref,
            "producer_state_fingerprint": self.producer_state_fingerprint,
            "provenance_fingerprint": self.provenance_fingerprint,
        }

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self.to_dict())


@dataclass(frozen=True)
class BoundTensorRead:
    """Caller-owned page value paired with its exact external binding."""

    binding: ExternalTensorBinding
    read: TensorRead
    _runtime_contract_ref: ClassVar[str] = "arti/bound-tensor-read@1"


class ExternalTensorProposal:
    """Owned complete-state proposal; never a commit authority."""

    __slots__ = (
        "_sealed",
        "_value",
        "candidate_fingerprint",
        "binding",
        "producer_ref",
        "producer_config_fingerprint",
        "producer_state_fingerprint",
        "semantics",
        "provenance_fingerprint",
    )
    _runtime_contract_ref: ClassVar[str] = "arti/external-proposal@1"

    def __setattr__(self, _name: str, _value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("ExternalTensorProposal is immutable")
        object.__setattr__(self, _name, _value)

    def __init__(
        self,
        binding: ExternalTensorBinding,
        value: Tensor,
        *,
        producer_ref: str,
        producer_config_fingerprint: str,
        producer_state_fingerprint: str,
        semantics: ProposalSemantics = ProposalSemantics.COMPLETE_NEXT_STATE,
    ) -> None:
        object.__setattr__(self, "_sealed", False)
        if not isinstance(binding, ExternalTensorBinding):
            raise TensorTransactionContractError("binding must be ExternalTensorBinding")
        producer_ref = _require_component_ref(producer_ref, "producer_ref")
        _require_sha256(producer_config_fingerprint, "producer_config_fingerprint")
        _require_sha256(producer_state_fingerprint, "producer_state_fingerprint")
        if producer_ref != binding.component_ref:
            raise TensorTransactionContractError("producer_ref must match the bound component")
        if producer_config_fingerprint != binding.component_config_fingerprint:
            raise TensorTransactionContractError(
                "producer config fingerprint must match the binding"
            )
        if producer_state_fingerprint != binding.producer_state_fingerprint:
            raise TensorTransactionContractError(
                "producer state fingerprint must match the binding"
            )
        if not isinstance(semantics, ProposalSemantics):
            raise TensorTransactionContractError("semantics must be ProposalSemantics")
        owned = _owned_candidate(value)
        if str(owned.dtype) != binding.tensor_ref.dtype or tuple(owned.shape) != binding.tensor_ref.shape:
            raise TensorTransactionContractError("proposal dtype and shape must match the bound page")
        self.binding = binding
        self._value = owned
        self.producer_ref = producer_ref
        self.producer_config_fingerprint = producer_config_fingerprint
        self.producer_state_fingerprint = producer_state_fingerprint
        self.semantics = semantics
        self.candidate_fingerprint = _tensor_fingerprint(owned)
        self.provenance_fingerprint = _fingerprint(
            {
                "ref": _require_component_ref(self._runtime_contract_ref, "proposal_ref"),
                "binding_fingerprint": binding.fingerprint,
                "producer_ref": producer_ref,
                "producer_config_fingerprint": producer_config_fingerprint,
                "producer_state_fingerprint": producer_state_fingerprint,
                "semantics": semantics.value,
                "candidate_fingerprint": self.candidate_fingerprint,
            }
        )
        object.__setattr__(self, "_sealed", True)

    @property
    def value(self) -> Tensor:
        return self._value.clone()

    def _trusted_value(self) -> Tensor:
        return self._value.clone()


@dataclass(frozen=True)
class FoldAddressBinding:
    """Bind stable logical identities to one FoldRecord transport."""

    source: ExternalTensorBinding
    logical_ids: tuple[str, ...]
    transported_logical_ids: tuple[tuple[str, ...], ...]
    write_logical_ids: tuple[str, ...]
    fold_record_fingerprint: str
    fold_record_ref: str
    producer_ref: str
    inverse_ref: str
    topology_ref: str
    topology_config_fingerprint: str
    _runtime_contract_ref: ClassVar[str] = "arti/fold-address-binding@1"

    def __post_init__(self) -> None:
        if not isinstance(self.source, ExternalTensorBinding):
            raise TensorTransactionContractError("source must be ExternalTensorBinding")
        normalized = tuple(self.logical_ids)
        object.__setattr__(self, "logical_ids", normalized)
        if not normalized or len(set(normalized)) != len(normalized):
            raise TensorTransactionContractError("logical_ids must be non-empty and unique")
        for value in normalized:
            _require_identifier(value, "logical_id")
        transported = tuple(tuple(row) for row in self.transported_logical_ids)
        object.__setattr__(self, "transported_logical_ids", transported)
        if not transported or any(
            len(row) != len(normalized) or set(row) != set(normalized)
            for row in transported
        ):
            raise TensorTransactionContractError(
                "transported_logical_ids must be per-instance permutations of logical_ids"
            )
        writes = tuple(self.write_logical_ids)
        object.__setattr__(self, "write_logical_ids", writes)
        if len(set(writes)) != len(writes) or not set(writes).issubset(normalized):
            raise TensorTransactionContractError(
                "write_logical_ids must be a unique subset of logical_ids"
            )
        if writes and self.source.authority is not TensorAuthority.READ_WRITE:
            raise TensorTransactionContractError(
                "write_logical_ids require a read_write source binding"
            )
        _require_sha256(self.fold_record_fingerprint, "fold_record_fingerprint")
        object.__setattr__(
            self,
            "fold_record_ref",
            _require_component_ref(self.fold_record_ref, "fold_record_ref"),
        )
        object.__setattr__(
            self, "producer_ref", _require_component_ref(self.producer_ref, "producer_ref")
        )
        object.__setattr__(
            self, "inverse_ref", _require_component_ref(self.inverse_ref, "inverse_ref")
        )
        object.__setattr__(
            self, "topology_ref", _require_component_ref(self.topology_ref, "topology_ref")
        )
        _require_sha256(self.topology_config_fingerprint, "topology_config_fingerprint")

    @classmethod
    def from_record(
        cls,
        source: ExternalTensorBinding,
        record: FoldRecord,
        *,
        logical_ids: Sequence[str],
        write_logical_ids: Sequence[str] = (),
    ) -> "FoldAddressBinding":
        if not isinstance(record, FoldRecord):
            raise TensorTransactionContractError("record must be FoldRecord@1")
        if source.tensor_ref.shape != record.original_shape:
            raise TensorTransactionContractError("source page shape must match FoldRecord original_shape")
        normalized = tuple(logical_ids)
        if len(normalized) != record.original_length:
            raise TensorTransactionContractError("logical_ids must match FoldRecord original length")
        permutation = record.permutation.reshape(-1, record.original_length).cpu().tolist()
        transported = tuple(
            tuple(normalized[index] for index in row)
            for row in permutation
        )
        return cls(
            source=source,
            logical_ids=normalized,
            transported_logical_ids=transported,
            write_logical_ids=tuple(write_logical_ids),
            fold_record_fingerprint=record.record_fingerprint,
            fold_record_ref=record._component_reference,
            producer_ref=record.producer_ref,
            inverse_ref=record.inverse_ref,
            topology_ref=record.topology_ref,
            topology_config_fingerprint=record.topology_config_fingerprint,
        )

    def validate_record(self, record: FoldRecord) -> None:
        candidate = type(self).from_record(
            self.source,
            record,
            logical_ids=self.logical_ids,
            write_logical_ids=self.write_logical_ids,
        )
        if candidate != self:
            raise TensorTransactionContractError("FoldRecord does not match FoldAddressBinding")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(
            {
                "ref": _require_component_ref(self._runtime_contract_ref, "fold_address_ref"),
                "source_binding_fingerprint": self.source.fingerprint,
                "logical_ids": list(self.logical_ids),
                "transported_logical_ids": [list(row) for row in self.transported_logical_ids],
                "write_logical_ids": list(self.write_logical_ids),
                "fold_record_fingerprint": self.fold_record_fingerprint,
                "fold_record_ref": self.fold_record_ref,
                "producer_ref": self.producer_ref,
                "inverse_ref": self.inverse_ref,
                "topology_ref": self.topology_ref,
                "topology_config_fingerprint": self.topology_config_fingerprint,
            }
        )


def bind_external_tensor(
    runtime: VolatileTensorRuntime,
    snapshot: TensorSnapshot,
    key: str,
    *,
    address_namespace: str,
    partition_id: str,
    logical_id: str,
    role: str,
    authority: TensorAuthority,
    component_ref: str,
    component_config_fingerprint: str,
    state_schema_ref: str,
    producer_state_fingerprint: str,
    provenance_fingerprint: str,
) -> BoundTensorRead:
    """Read one owned clone and bind it to the exact snapshot/page contract."""

    observed = runtime.read(snapshot, key)
    binding = ExternalTensorBinding(
        store_instance_id=snapshot.store_instance_id,
        world_id=snapshot.world_id,
        abi_fingerprint=snapshot.abi_fingerprint,
        root_id=snapshot.root_id,
        root_epoch=snapshot.epoch,
        root_fingerprint=snapshot.root_fingerprint,
        address_namespace=address_namespace,
        partition_id=partition_id,
        logical_id=logical_id,
        tensor_ref=observed.ref,
        role=role,
        authority=authority,
        component_ref=component_ref,
        component_config_fingerprint=component_config_fingerprint,
        state_schema_ref=state_schema_ref,
        producer_state_fingerprint=producer_state_fingerprint,
        provenance_fingerprint=provenance_fingerprint,
    )
    return BoundTensorRead(binding, observed)


def stage_external_proposal(transaction: TensorTransaction, proposal: ExternalTensorProposal) -> None:
    """Validate one complete-state proposal and stage it without committing."""

    if not isinstance(transaction, TensorTransaction):
        raise TensorTransactionContractError("transaction must come from VolatileTensorRuntime.begin")
    if not isinstance(proposal, ExternalTensorProposal):
        raise TensorTransactionContractError("proposal must be ExternalTensorProposal")
    if proposal.semantics is not ProposalSemantics.COMPLETE_NEXT_STATE:
        raise TensorTransactionContractError("v1 only stages complete_next_state proposals")
    binding = proposal.binding
    if binding.authority is not TensorAuthority.READ_WRITE:
        raise TensorTransactionContractError("binding does not grant write authority")
    if proposal.producer_state_fingerprint != binding.producer_state_fingerprint:
        raise TensorTransactionContractError("proposal producer state does not match binding")
    trusted_value = proposal._trusted_value()
    if _tensor_fingerprint(trusted_value) != proposal.candidate_fingerprint:
        raise TensorTransactionContractError("sealed proposal candidate was mutated")
    snapshot = transaction._snapshot
    if (
        binding.store_instance_id != snapshot.store_instance_id
        or binding.world_id != snapshot.world_id
        or binding.abi_fingerprint != snapshot.abi_fingerprint
        or binding.root_id != snapshot.root_id
        or binding.root_epoch != snapshot.epoch
        or binding.root_fingerprint != snapshot.root_fingerprint
    ):
        raise TensorTransactionContractError("binding does not match the transaction snapshot")
    current = transaction._base_root.pages.get(binding.tensor_ref.key)
    if current is None or current.ref != binding.tensor_ref:
        raise TensorTransactionContractError("binding TensorRef is stale or foreign")
    transaction.stage(
        binding.tensor_ref.key,
        trusted_value,
        expected_version=binding.tensor_ref.version,
        provenance_fingerprint=proposal.provenance_fingerprint,
    )


__all__ = [
    "BoundTensorRead",
    "ExternalTensorBinding",
    "ExternalTensorProposal",
    "FoldAddressBinding",
    "ProposalSemantics",
    "TensorAuthority",
    "bind_external_tensor",
    "stage_external_proposal",
]
