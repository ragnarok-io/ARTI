"""Reversible tensor-instance topology operations.

This module separates tensor values from their temporary computational
topology.  A fold partitions original instances into active and preserved
payloads using a recorded permutation.  Unfold applies the exact inverse
transport; it never predicts values or recomputes the topology.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import ClassVar, Sequence

import torch
from torch import Tensor, nn

from .topology import (
    SoftTopKTopologySurrogate,
    StablePriorityPartition,
    TopologyAction,
    TopologyProposal,
)


FOLD_RECORD_SCHEMA_VERSION = 1
FOLD_STATE_SCHEMA_VERSION = 1


class _HardValueSoftTopology(torch.autograd.Function):
    """Return hard values exactly while routing a VJP to soft topology."""

    @staticmethod
    def forward(_context: object, hard: Tensor, _soft: Tensor) -> Tensor:
        return hard

    @staticmethod
    def backward(_context: object, gradient: Tensor) -> tuple[Tensor, Tensor]:
        return gradient, gradient


def _sha256_json(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _transport_contract_fingerprint(*, active_count: int, axis: int) -> str:
    return _sha256_json(
        {
            "ref": "arti/reversible-topology@1",
            "active_count": active_count,
            "axis": axis,
            "operator_ref": "arti/stable-priority-partition@1",
            "tie_break": "stable-index",
            "inverse": "recorded-permutation",
            "original_instance_conservation": True,
        }
    )


def _validate_permutation(permutation: Tensor, *, length: int) -> None:
    if permutation.dtype != torch.long:
        raise ValueError("record permutation must use torch.int64")
    if permutation.ndim < 1 or permutation.shape[-1] != length:
        raise ValueError(f"record permutation must end with length {length}")
    if permutation.layout != torch.strided:
        raise ValueError("record permutation must use strided layout")
    compiler = getattr(torch, "compiler", None)
    is_compiling = bool(compiler is not None and compiler.is_compiling())
    expected = torch.arange(length, device=permutation.device, dtype=torch.long)
    expected = expected.expand_as(permutation)
    is_bijection = torch.eq(torch.sort(permutation, dim=-1).values, expected).all()
    if is_compiling:
        torch._assert_async(
            is_bijection,
            "record permutation must be a complete per-sample bijection",
        )
    elif not is_bijection:
        raise ValueError("record permutation must be a complete per-sample bijection")


class FoldRecord:
    """Immutable topology metadata produced by :class:`TopologyFold`.

    Tensor properties return clones so callers cannot mutate the canonical
    record through the public API.  The record stores topology only; folded
    values remain in :class:`FoldedTensor`.
    """

    __slots__ = (
        "_sealed",
        "_permutation",
        "_original_mask",
        "original_shape",
        "axis",
        "active_count",
        "producer_ref",
        "inverse_ref",
        "topology_ref",
        "topology_config_fingerprint",
        "producer_provenance_fingerprint",
        "schema_version",
    )

    _component_reference: ClassVar[str] = "arti/fold-record@1"

    def __setattr__(self, _name: str, _value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("FoldRecord is immutable")
        object.__setattr__(self, _name, _value)

    def __init__(
        self,
        *,
        permutation: Tensor,
        original_mask: Tensor,
        original_shape: Sequence[int],
        axis: int,
        active_count: int,
        producer_ref: str = "arti/fold@2",
        inverse_ref: str = "arti/unfold@2",
        topology_ref: str = "arti/reversible-topology@1",
        topology_config_fingerprint: str,
        producer_provenance_fingerprint: str = "fixed-producer",
        schema_version: int = FOLD_RECORD_SCHEMA_VERSION,
    ) -> None:
        object.__setattr__(self, "_sealed", False)
        shape = tuple(int(size) for size in original_shape)
        if len(shape) < 2:
            raise ValueError("original_shape must contain instance and feature axes")
        if axis not in {-2, len(shape) - 2}:
            raise ValueError("FoldRecord@1 only supports the penultimate instance axis")
        length = shape[-2]
        if active_count <= 0 or active_count > length:
            raise ValueError("active_count must be in the interval [1, original_length]")
        if permutation.shape != shape[:-1]:
            raise ValueError("record permutation shape must equal original_shape[:-1]")
        if original_mask.shape != shape[:-1] or original_mask.dtype != torch.bool:
            raise ValueError("original_mask must be boolean with shape original_shape[:-1]")
        if permutation.device != original_mask.device:
            raise ValueError("record permutation and original_mask must share a device")
        _validate_permutation(permutation, length=length)
        if schema_version != FOLD_RECORD_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported FoldRecord schema {schema_version}; "
                f"expected {FOLD_RECORD_SCHEMA_VERSION}"
            )
        if not topology_config_fingerprint:
            raise ValueError("topology_config_fingerprint must be non-empty")
        if not producer_provenance_fingerprint:
            raise ValueError("producer_provenance_fingerprint must be non-empty")

        object.__setattr__(self, "_permutation", permutation.detach().clone())
        object.__setattr__(self, "_original_mask", original_mask.detach().clone())
        object.__setattr__(self, "original_shape", shape)
        object.__setattr__(self, "axis", -2)
        object.__setattr__(self, "active_count", int(active_count))
        object.__setattr__(self, "producer_ref", producer_ref)
        object.__setattr__(self, "inverse_ref", inverse_ref)
        object.__setattr__(self, "topology_ref", topology_ref)
        object.__setattr__(self, "topology_config_fingerprint", topology_config_fingerprint)
        object.__setattr__(
            self, "producer_provenance_fingerprint", producer_provenance_fingerprint
        )
        object.__setattr__(self, "schema_version", int(schema_version))
        object.__setattr__(self, "_sealed", True)

    @property
    def permutation(self) -> Tensor:
        return self._permutation.clone()

    @property
    def original_mask(self) -> Tensor:
        return self._original_mask.clone()

    @property
    def original_length(self) -> int:
        return self.original_shape[-2]

    @property
    def folded_count(self) -> int:
        return self.original_length - self.active_count

    @property
    def active_index(self) -> Tensor:
        return self._permutation[..., : self.active_count].clone()

    @property
    def folded_index(self) -> Tensor:
        return self._permutation[..., self.active_count :].clone()

    @property
    def inverse_index(self) -> Tensor:
        return torch.argsort(self._permutation, dim=-1, stable=True)

    @property
    def record_fingerprint(self) -> str:
        # Explicit inspection may synchronize a CUDA record.  The hot unfold
        # path validates the bijection without moving topology to the host.
        return _sha256_json(
            {
                "schema_version": self.schema_version,
                "producer_ref": self.producer_ref,
                "inverse_ref": self.inverse_ref,
                "topology_ref": self.topology_ref,
                "topology_config_fingerprint": self.topology_config_fingerprint,
                "producer_provenance_fingerprint": self.producer_provenance_fingerprint,
                "original_shape": list(self.original_shape),
                "axis": self.axis,
                "active_count": self.active_count,
                "permutation": self._permutation.detach().cpu().tolist(),
                "original_mask": self._original_mask.detach().cpu().to(torch.uint8).tolist(),
            }
        )

    def _trusted_permutation(self) -> Tensor:
        return self._permutation

    def _trusted_original_mask(self) -> Tensor:
        return self._original_mask


@dataclass(frozen=True)
class FoldedTensor:
    """Active and preserved payloads paired with their exact topology record."""

    active: Tensor
    folded: Tensor
    active_mask: Tensor
    folded_mask: Tensor
    record: FoldRecord
    schema_version: int = FOLD_STATE_SCHEMA_VERSION

    _component_reference: ClassVar[str] = "arti/fold-state@1"

    def __post_init__(self) -> None:
        if self.schema_version != FOLD_STATE_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported FoldedTensor schema {self.schema_version}; "
                f"expected {FOLD_STATE_SCHEMA_VERSION}"
            )
        expected_active = (*self.record.original_shape[:-2], self.record.active_count)
        expected_folded = (*self.record.original_shape[:-2], self.record.folded_count)
        if self.active.shape != (*expected_active, self.record.original_shape[-1]):
            raise ValueError("active payload shape does not match the topology record")
        if self.folded.shape != (*expected_folded, self.record.original_shape[-1]):
            raise ValueError("folded payload shape does not match the topology record")
        if self.active_mask.shape != expected_active or self.active_mask.dtype != torch.bool:
            raise ValueError("active_mask must be boolean and match the active payload")
        if self.folded_mask.shape != expected_folded or self.folded_mask.dtype != torch.bool:
            raise ValueError("folded_mask must be boolean and match the folded payload")
        tensors = (self.active, self.folded, self.active_mask, self.folded_mask)
        if any(tensor.device != self.active.device for tensor in tensors):
            raise ValueError("FoldedTensor payloads and masks must share a device")
        if self.folded.dtype != self.active.dtype:
            raise ValueError("active and folded payloads must share a dtype")
        if self.record._trusted_permutation().device != self.active.device:
            raise ValueError("FoldedTensor and its record must share a device")

    def replace(
        self,
        *,
        active: Tensor | None = None,
        folded: Tensor | None = None,
    ) -> FoldedTensor:
        next_active = self.active if active is None else active
        next_folded = self.folded if folded is None else folded
        if next_active.shape != self.active.shape:
            raise ValueError("replacement active payload must preserve shape")
        if next_folded.shape != self.folded.shape:
            raise ValueError("replacement folded payload must preserve shape")
        if next_active.device != self.active.device or next_folded.device != self.folded.device:
            raise ValueError("replacement payloads must preserve device")
        if next_active.dtype != self.active.dtype or next_folded.dtype != self.folded.dtype:
            raise ValueError("replacement payloads must preserve dtype")
        return FoldedTensor(
            active=next_active,
            folded=next_folded,
            active_mask=self.active_mask,
            folded_mask=self.folded_mask,
            record=self.record,
            schema_version=self.schema_version,
        )


@dataclass(frozen=True)
class UnfoldedTensor:
    """A restored tensor and its mask in the original host topology."""

    value: Tensor
    mask: Tensor


def _unfold_state(
    state: FoldedTensor,
    *,
    active_count: int,
    axis: int,
    topology_ref: str,
    contract_fingerprint: str,
) -> UnfoldedTensor:
    if not isinstance(state, FoldedTensor):
        raise TypeError("unfold expects a FoldedTensor produced by Fold@2")
    record = state.record
    if record.producer_ref != "arti/fold@2" or record.inverse_ref != "arti/unfold@2":
        raise ValueError("FoldRecord operation identities are incompatible")
    if record.topology_ref != topology_ref:
        raise ValueError("FoldRecord topology identity is incompatible")
    if record.active_count != active_count or record.axis != axis:
        raise ValueError("FoldRecord transport dimensions are incompatible")
    if record.topology_config_fingerprint != contract_fingerprint:
        raise ValueError("FoldRecord was produced by a different topology contract")
    permutation = record._trusted_permutation()
    _validate_permutation(permutation, length=record.original_length)

    packed = torch.cat((state.active, state.folded), dim=-2)
    packed_mask = torch.cat((state.active_mask, state.folded_mask), dim=-1)
    scatter_index = permutation.unsqueeze(-1).expand(*permutation.shape, packed.shape[-1])
    restored = torch.zeros_like(packed).scatter(-2, scatter_index, packed)
    restored_mask = torch.zeros_like(packed_mask).scatter(-1, permutation, packed_mask)
    compiler = getattr(torch, "compiler", None)
    is_compiling = bool(compiler is not None and compiler.is_compiling())
    mask_lineage_matches = torch.eq(
        restored_mask, record._trusted_original_mask()
    ).all()
    if is_compiling:
        torch._assert_async(
            mask_lineage_matches,
            "FoldedTensor mask lineage does not match its record",
        )
    elif not mask_lineage_matches:
        raise ValueError("FoldedTensor mask lineage does not match its record")
    return UnfoldedTensor(restored, restored_mask)


class InverseTopologyContract(nn.Module):
    """Parameter-free contract for the recorded inverse used by UnFold@2."""

    _component_reference: ClassVar[str] = "arti/inverse-topology-contract@1"

    def __init__(self, active_count: int, *, axis: int = -2) -> None:
        super().__init__()
        if active_count <= 0:
            raise ValueError("active_count must be positive")
        if axis != -2:
            raise ValueError("InverseTopologyContract@1 only supports axis=-2")
        self.active_count = int(active_count)
        self.axis = int(axis)
        self.topology_ref = "arti/reversible-topology@1"
        self.contract_fingerprint = _transport_contract_fingerprint(
            active_count=self.active_count, axis=self.axis
        )

    def forward(self, state: FoldedTensor) -> UnfoldedTensor:
        return _unfold_state(
            state,
            active_count=self.active_count,
            axis=self.axis,
            topology_ref=self.topology_ref,
            contract_fingerprint=self.contract_fingerprint,
        )

    def topology_contract(self) -> dict[str, object]:
        return {
            "ref": self._component_reference,
            "active_count": self.active_count,
            "axis": self.axis,
            "topology_ref": self.topology_ref,
            "contract_fingerprint": self.contract_fingerprint,
        }


class FixedTopologyPolicy(nn.Module):
    """Deterministic topology policy backed by an optional full index order."""

    _component_reference: ClassVar[str] = "arti/fixed-topology-policy@1"

    def __init__(self, order: Sequence[int] | Tensor | None = None) -> None:
        super().__init__()
        if order is None:
            value = torch.empty(0, dtype=torch.long)
        else:
            value = torch.as_tensor(order, dtype=torch.long)
            if value.ndim != 1:
                raise ValueError("order must be a one-dimensional index sequence")
            _validate_permutation(value, length=value.numel())
        self.register_buffer("order", value, persistent=True)

    def forward(self, x: Tensor, mask: Tensor) -> TopologyProposal:
        if mask.ndim < 1 or mask.dtype != torch.bool:
            raise ValueError("mask must be a boolean tensor with an instance axis")
        if x.shape[:-1] != mask.shape or not x.is_floating_point():
            raise ValueError("x must be floating point with shape [..., N, D]")
        length = mask.shape[-1]
        if self.order.numel() == 0:
            order = torch.arange(length, device=mask.device, dtype=torch.long)
        else:
            if self.order.numel() != length:
                raise ValueError(
                    f"fixed topology order has length {self.order.numel()}, "
                    f"but the input length is {length}"
                )
            order = self.order.to(device=mask.device)
        rank = torch.empty(length, device=mask.device, dtype=torch.long)
        rank.scatter_(0, order, torch.arange(length, device=mask.device))
        priority = (length - rank).to(dtype=x.dtype)
        priority = priority.expand(*mask.shape[:-1], length)
        return TopologyProposal(TopologyAction(priority))

    def extra_repr(self) -> str:
        return "identity" if self.order.numel() == 0 else f"length={self.order.numel()}"

    def topology_contract(self) -> dict[str, object]:
        return {
            "ref": self._component_reference,
            "order": self.order.detach().cpu().tolist(),
            "tie_break": "stable-index",
            "validity": "valid-first",
        }


class ReversibleTopology(nn.Module):
    """Shared topology contract used by reversible Fold and UnFold operations."""

    _component_reference: ClassVar[str] = "arti/reversible-topology@1"

    def __init__(
        self,
        active_count: int,
        *,
        policy: nn.Module | None = None,
        operator: StablePriorityPartition | None = None,
        surrogate: SoftTopKTopologySurrogate | None = None,
        axis: int = -2,
    ) -> None:
        super().__init__()
        if active_count <= 0:
            raise ValueError("active_count must be positive")
        if axis != -2:
            raise ValueError("ReversibleTopology@1 only supports axis=-2")
        self.active_count = int(active_count)
        self.axis = int(axis)
        self.policy = FixedTopologyPolicy() if policy is None else policy
        self.operator = StablePriorityPartition() if operator is None else operator
        if not isinstance(self.policy, nn.Module):
            raise TypeError("ReversibleTopology@1 policy must be an nn.Module")
        if type(self.operator) is not StablePriorityPartition:
            raise TypeError("ReversibleTopology@1 requires StablePriorityPartition@1")
        has_trainable_policy = any(
            parameter.requires_grad for parameter in self.policy.parameters()
        )
        self.surrogate = (
            SoftTopKTopologySurrogate()
            if surrogate is None and has_trainable_policy
            else surrogate
        )
        if self.surrogate is not None and not isinstance(
            self.surrogate, SoftTopKTopologySurrogate
        ):
            raise TypeError(
                "ReversibleTopology@1 requires SoftTopKTopologySurrogate@1"
            )
        self._contract_fingerprint = self._make_contract_fingerprint()
        self._producer_provenance_fingerprint = self._make_producer_fingerprint()
        self.register_load_state_dict_post_hook(self._refresh_contract_fingerprint)

    def _make_contract_fingerprint(self) -> str:
        return _transport_contract_fingerprint(
            active_count=self.active_count, axis=self.axis
        )

    def _make_producer_fingerprint(self) -> str:
        contract = getattr(self.policy, "topology_contract", None)
        if not callable(contract):
            raise TypeError("topology policy must expose topology_contract()")
        return _sha256_json(
            {
                "policy": contract(),
                "operator_ref": self.operator._component_reference,
                "surrogate": (
                    None
                    if self.surrogate is None
                    else {
                        "ref": self.surrogate._component_reference,
                        "temperature": self.surrogate.temperature,
                        "path": "backward-only",
                    }
                ),
            }
        )

    @property
    def producer_provenance_fingerprint(self) -> str:
        return self._producer_provenance_fingerprint

    def _refresh_contract_fingerprint(
        self,
        _module: nn.Module,
        _incompatible_keys: object,
    ) -> None:
        self._contract_fingerprint = self._make_contract_fingerprint()
        self._producer_provenance_fingerprint = self._make_producer_fingerprint()

    @property
    def contract_fingerprint(self) -> str:
        return self._contract_fingerprint

    def fold(self, x: Tensor, mask: Tensor | None = None) -> FoldedTensor:
        if x.ndim < 2:
            raise ValueError("x must have shape [..., N, D]")
        length = x.shape[-2]
        if length == 0:
            raise ValueError("x must contain at least one tensor instance")
        if self.active_count > length:
            raise ValueError(
                f"active_count={self.active_count} exceeds input length {length}"
            )
        if mask is None:
            valid = torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)
        else:
            if mask.shape != x.shape[:-1] or mask.dtype != torch.bool:
                raise ValueError("mask must be boolean with shape x.shape[:-1]")
            if mask.device != x.device:
                raise ValueError("mask and x must share a device")
            valid = mask

        # Topology parameters learn through the declared surrogate. Tensor
        # values retain only the gradient of the executed hard lineage.
        proposal = self.policy(x.detach(), valid)
        if not isinstance(proposal, TopologyProposal):
            raise TypeError("topology policy must return TopologyProposal")
        permutation = self.operator(proposal.action, valid)
        needs_surrogate = (
            self.training
            and self.policy.training
            and torch.is_grad_enabled()
            and self.surrogate is not None
            and any(
                parameter.requires_grad for parameter in self.policy.parameters()
            )
        )
        surrogate_assignment = (
            self.surrogate(proposal.action, valid, self.active_count)
            if needs_surrogate
            else None
        )
        gather_index = permutation.unsqueeze(-1).expand(*permutation.shape, x.shape[-1])
        packed = torch.gather(x, -2, gather_index)
        packed_mask = torch.gather(valid, -1, permutation)
        active = packed[..., : self.active_count, :].clone()
        if surrogate_assignment is not None:
            expected = (*x.shape[:-2], self.active_count, x.shape[-2])
            if surrogate_assignment.shape != expected:
                raise ValueError(
                    "topology surrogate assignment must have shape [..., K, N]"
                )
            detached = x.detach()
            finite_detached = torch.where(
                torch.isfinite(detached), detached, torch.zeros_like(detached)
            )
            soft_active = torch.einsum(
                "...kn,...nd->...kd", surrogate_assignment, finite_detached
            )
            active = _HardValueSoftTopology.apply(active, soft_active)
        folded = packed[..., self.active_count :, :].clone()
        active_mask = packed_mask[..., : self.active_count].clone()
        folded_mask = packed_mask[..., self.active_count :].clone()
        record = FoldRecord(
            permutation=permutation,
            original_mask=valid,
            original_shape=x.shape,
            axis=self.axis,
            active_count=self.active_count,
            topology_config_fingerprint=self.contract_fingerprint,
            producer_provenance_fingerprint=self.producer_provenance_fingerprint,
        )
        return FoldedTensor(active, folded, active_mask, folded_mask, record)

    def unfold(self, state: FoldedTensor) -> UnfoldedTensor:
        return _unfold_state(
            state,
            active_count=self.active_count,
            axis=self.axis,
            topology_ref=self._component_reference,
            contract_fingerprint=self.contract_fingerprint,
        )

    def operations(self) -> tuple[TopologyFold, TopologyUnFold]:
        return TopologyFold(self), TopologyUnFold(self)

    def extra_repr(self) -> str:
        return f"active_count={self.active_count}, axis={self.axis}"


class TopologyFold(nn.Module):
    """Canonical ``arti/fold@2`` forward topology operation."""

    _component_reference: ClassVar[str] = "arti/fold@2"

    def __init__(
        self,
        topology: ReversibleTopology | None = None,
        *,
        active_count: int | None = None,
        policy: nn.Module | None = None,
        operator: StablePriorityPartition | None = None,
        axis: int = -2,
    ) -> None:
        super().__init__()
        if topology is None:
            if active_count is None:
                raise ValueError("active_count is required when topology is omitted")
            topology = ReversibleTopology(
                active_count, policy=policy, operator=operator, axis=axis
            )
        elif (
            active_count is not None
            or policy is not None
            or operator is not None
            or axis != -2
        ):
            raise ValueError("topology cannot be combined with topology construction options")
        self.topology = topology

    def forward(self, x: Tensor, mask: Tensor | None = None) -> FoldedTensor:
        return self.topology.fold(x, mask)


class TopologyUnFold(nn.Module):
    """Canonical ``arti/unfold@2`` exact recorded inverse operation."""

    _component_reference: ClassVar[str] = "arti/unfold@2"

    def __init__(
        self,
        topology: ReversibleTopology | None = None,
        *,
        active_count: int | None = None,
        axis: int = -2,
    ) -> None:
        super().__init__()
        if topology is None:
            if active_count is None:
                raise ValueError("active_count is required when topology is omitted")
            selected_active_count = active_count
            selected_axis = axis
        elif active_count is not None or axis != -2:
            raise ValueError("topology cannot be combined with topology construction options")
        else:
            selected_active_count = topology.active_count
            selected_axis = topology.axis
        self.inverse_contract = InverseTopologyContract(
            selected_active_count, axis=selected_axis
        )

    def forward(self, state: FoldedTensor) -> UnfoldedTensor:
        return self.inverse_contract(state)


__all__ = [
    "FOLD_RECORD_SCHEMA_VERSION",
    "FOLD_STATE_SCHEMA_VERSION",
    "FixedTopologyPolicy",
    "FoldRecord",
    "FoldedTensor",
    "InverseTopologyContract",
    "ReversibleTopology",
    "TopologyFold",
    "TopologyUnFold",
    "UnfoldedTensor",
]
