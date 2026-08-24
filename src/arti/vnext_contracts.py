"""Versioned contracts shared by the composable ARTI vNext pipeline.

Runtime tensor envelopes are not model components. Bank sources propose typed
operands, policies admit bounded proposals, and fixed operators own execution.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import re
from typing import Any, ClassVar, Mapping, Protocol

import torch
from torch import Tensor, nn

from .component_registry import ComponentRef, component_ref, get_component_registry

SUPPORT_SCHEMA_VERSION = 1
TYPED_OPERANDS_SCHEMA_VERSION = 1
PULSE_STAGE_SCHEMA_VERSION = 1
PULSE_STAGE_GRAPH_SCHEMA_VERSION = 1

_IDENTIFIER = re.compile(r"[a-z][a-z0-9]*(?:[-_.][a-z0-9]+)*")
_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class ContractLimits:
    """Admission limits checked before runtime copies or component calls."""

    max_rank: int = 8
    max_dimension: int = 1_048_576
    max_elements: int = 16_777_216
    max_tensor_bytes: int = 268_435_456
    max_operation_bytes: int = 1_610_612_736
    max_stages: int = 32
    max_config_bytes: int = 65_536
    max_string_bytes: int = 8_192
    max_json_depth: int = 16
    max_json_nodes: int = 4096

    def __post_init__(self) -> None:
        for name, value in self.__dict__.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

    def admit_tensor(self, value: Tensor, *, name: str) -> None:
        if not isinstance(value, Tensor):
            raise TypeError(f"{name} must be a Tensor")
        if value.layout != torch.strided:
            raise ValueError(f"{name} must use strided layout")
        if value.ndim > self.max_rank:
            raise ValueError(f"{name} exceeds max_rank")
        numel = 1
        for dimension in value.shape:
            if dimension < 0 or dimension > self.max_dimension:
                raise ValueError(f"{name} exceeds max_dimension")
            numel = math.prod((numel, dimension))
            if numel > self.max_elements:
                raise ValueError(f"{name} exceeds max_elements")
        if numel * value.element_size() > self.max_tensor_bytes:
            raise ValueError(f"{name} exceeds max_tensor_bytes")


DEFAULT_CONTRACT_LIMITS = ContractLimits()


def _assert_limits_do_not_relax_hard_ceiling(limits: ContractLimits) -> None:
    for name, hard_value in DEFAULT_CONTRACT_LIMITS.__dict__.items():
        if getattr(limits, name) > hard_value:
            raise ValueError(f"runtime limits cannot relax hard ceiling {name}")


def _admit_intervention_operation(
    base: Tensor,
    candidate: Tensor,
    mask: Tensor,
    limits: ContractLimits,
) -> None:
    if limits is not DEFAULT_CONTRACT_LIMITS:
        _assert_limits_do_not_relax_hard_ceiling(limits)
    limits.admit_tensor(base, name="intervention base")
    limits.admit_tensor(candidate, name="intervention candidate")
    output_bytes = base.numel() * base.element_size()
    mask_bytes = mask.numel() * mask.element_size()
    live_bytes = base.nbytes + candidate.nbytes + output_bytes + mask_bytes
    if torch.is_grad_enabled() and (base.requires_grad or candidate.requires_grad):
        live_bytes += base.nbytes + candidate.nbytes
    if live_bytes > limits.max_operation_bytes:
        raise ValueError("intervention exceeds max_operation_bytes")


class _JsonBudget:
    def __init__(self, limits: ContractLimits) -> None:
        self.limits = limits
        self.nodes = 0

    def normalize(self, value: Any, *, depth: int = 0) -> Any:
        self.nodes += 1
        if self.nodes > self.limits.max_json_nodes:
            raise ValueError("vNext config exceeds max_json_nodes")
        if depth > self.limits.max_json_depth:
            raise ValueError("vNext config exceeds max_json_depth")
        if isinstance(value, Mapping):
            if len(value) > self.limits.max_json_nodes:
                raise ValueError("vNext config mapping exceeds max_json_nodes")
            if any(not isinstance(key, str) for key in value):
                raise TypeError("vNext contract mapping keys must be strings")
            if any(
                len(key.encode("utf-8")) > self.limits.max_string_bytes
                for key in value
            ):
                raise ValueError("vNext contract key exceeds max_string_bytes")
            return {
                key: self.normalize(item, depth=depth + 1)
                for key, item in sorted(value.items())
            }
        if isinstance(value, (list, tuple)):
            return [self.normalize(item, depth=depth + 1) for item in value]
        if isinstance(value, (str, bool, int)) or value is None:
            if isinstance(value, str) and len(value.encode("utf-8")) > self.limits.max_string_bytes:
                raise ValueError("vNext contract string exceeds max_string_bytes")
            return value
        if isinstance(value, float):
            if not math.isfinite(value):
                raise ValueError("vNext contract floats must be finite")
            return value
        raise TypeError(f"unsupported vNext contract value: {type(value).__name__}")


def _canonical_json(value: Any, limits: ContractLimits = DEFAULT_CONTRACT_LIMITS) -> str:
    encoded = json.dumps(
        _JsonBudget(limits).normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    if len(encoded.encode("utf-8")) > limits.max_config_bytes:
        raise ValueError("vNext config exceeds max_config_bytes")
    return encoded


def _fingerprint_json(encoded: str) -> str:
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SupportDomain:
    """Coordinate-space identity for observed/exposed/intervened/write support."""

    domain_id: str
    owner_ref: str
    partition_id: str
    transition_id: str
    layout: str
    shape: tuple[int, ...]
    device: str
    axis: int = -1
    _runtime_contract_ref: ClassVar[str] = "arti/support-domain@1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(self.shape))
        for value, name in (
            (self.domain_id, "domain_id"),
            (self.partition_id, "partition_id"),
            (self.transition_id, "transition_id"),
        ):
            if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
                raise ValueError(f"{name} must be a stable lowercase identifier")
        ComponentRef.parse(self.owner_ref)
        if self.layout not in {"dense", "packed"}:
            raise ValueError("support layout must be 'dense' or 'packed'")
        if not self.shape or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in self.shape
        ):
            raise ValueError("support shape must contain non-negative integers")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("support device must be non-empty")
        if not isinstance(self.axis, int) or isinstance(self.axis, bool):
            raise TypeError("support axis must be an integer")

    @classmethod
    def for_tensor(
        cls,
        mask: Tensor,
        *,
        domain_id: str,
        owner_ref: str,
        partition_id: str,
        transition_id: str,
        layout: str = "dense",
        axis: int = -1,
    ) -> "SupportDomain":
        return cls(
            domain_id=domain_id,
            owner_ref=owner_ref,
            partition_id=partition_id,
            transition_id=transition_id,
            layout=layout,
            shape=tuple(mask.shape),
            device=str(mask.device),
            axis=axis,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": SUPPORT_SCHEMA_VERSION,
            "ref": self._runtime_contract_ref,
            "domain_id": self.domain_id,
            "owner_ref": self.owner_ref,
            "partition_id": self.partition_id,
            "transition_id": self.transition_id,
            "layout": self.layout,
            "shape": list(self.shape),
            "device": self.device,
            "axis": self.axis,
        }


class SupportKind(str, Enum):
    OBSERVED = "observed"
    EXPOSED = "exposed"
    INTERVENED = "intervened"
    WRITE = "write"


class SupportMask:
    """Immutable boolean authorization snapshot in one SupportDomain."""

    _runtime_contract_ref: ClassVar[str] = "arti/support-mask@1"
    __slots__ = ("_sealed", "kind", "domain", "_mask")

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("SupportMask is immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        kind: SupportKind,
        mask: Tensor,
        domain: SupportDomain,
        *,
        limits: ContractLimits | None = None,
    ) -> None:
        object.__setattr__(self, "_sealed", False)
        if not isinstance(kind, SupportKind):
            raise TypeError("kind must be a SupportKind")
        if not torch.compiler.is_compiling():
            (DEFAULT_CONTRACT_LIMITS if limits is None else limits).admit_tensor(
                mask, name="support mask"
            )
        if mask.dtype != torch.bool or mask.ndim < 1:
            raise TypeError("support mask must be a boolean Tensor with rank >= 1")
        if not isinstance(domain, SupportDomain):
            raise TypeError("domain must be a SupportDomain")
        if tuple(mask.shape) != domain.shape or str(mask.device) != domain.device:
            raise ValueError("support mask does not match its domain shape/device")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "_mask", mask.detach().clone())
        object.__setattr__(self, "_sealed", True)

    @property
    def mask(self) -> Tensor:
        return self._mask.clone()

    @property
    def shape(self) -> torch.Size:
        return self._mask.shape

    def metadata(self) -> dict[str, Any]:
        return {
            "schema_version": SUPPORT_SCHEMA_VERSION,
            "ref": self._runtime_contract_ref,
            "kind": self.kind.value,
            "domain": self.domain.metadata(),
        }


class PulseSupports:
    """Validated support lattice for one immutable transition snapshot."""

    _runtime_contract_ref: ClassVar[str] = "arti/pulse-supports@1"
    __slots__ = (
        "_sealed",
        "observed",
        "exposed",
        "intervened",
        "write",
        "write_contract",
        "_validity",
    )

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("PulseSupports is immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        observed: SupportMask,
        exposed: SupportMask,
        intervened: SupportMask,
        write: SupportMask | None = None,
        write_contract: OperandContract | None = None,
        *,
        validity: Tensor,
    ) -> None:
        object.__setattr__(self, "_sealed", False)
        for support, kind, name in (
            (observed, SupportKind.OBSERVED, "observed"),
            (exposed, SupportKind.EXPOSED, "exposed"),
            (intervened, SupportKind.INTERVENED, "intervened"),
        ):
            if not isinstance(support, SupportMask) or support.kind is not kind:
                raise TypeError(f"{name} must be a {kind.value} SupportMask")
        if not (observed.domain == exposed.domain == intervened.domain):
            raise ValueError("observed, exposed, and intervened supports must share domain")
        if (
            not isinstance(validity, Tensor)
            or validity.dtype != torch.bool
            or tuple(validity.shape) != observed.domain.shape
            or str(validity.device) != observed.domain.device
        ):
            raise ValueError("validity must be boolean and match the support domain")
        support_checks = (
            (
                ~(observed._mask & ~validity).any(),
                "observed support must be a subset of envelope validity",
            ),
            (
                ~(exposed._mask & ~observed._mask).any(),
                "exposed support must be a subset of observed support",
            ),
            (
                ~(intervened._mask & ~exposed._mask).any(),
                "intervened support must be a subset of exposed support",
            ),
        )
        for valid_support, message in support_checks:
            if torch.compiler.is_compiling():
                torch._assert_async(valid_support, message)
            elif not bool(valid_support):
                raise ValueError(message)
        if write is not None:
            if not isinstance(write, SupportMask) or write.kind is not SupportKind.WRITE:
                raise TypeError("write must be a write SupportMask")
            if write.domain.transition_id != observed.domain.transition_id:
                raise ValueError("write support must belong to the same transition")
            if not isinstance(write_contract, OperandContract):
                raise ValueError("write support requires an OperandContract")
            if write_contract.kind is not OperandKind.WRITE:
                raise ValueError("write support requires a WRITE operand contract")
            if write_contract.domain != write.domain:
                raise ValueError("write support and operand contract must share target domain")
        elif write_contract is not None:
            raise ValueError("write_contract requires write support")
        object.__setattr__(self, "observed", observed)
        object.__setattr__(self, "exposed", exposed)
        object.__setattr__(self, "intervened", intervened)
        object.__setattr__(self, "write", write)
        object.__setattr__(self, "write_contract", write_contract)
        object.__setattr__(self, "_validity", validity.detach().clone())
        object.__setattr__(self, "_sealed", True)

    @classmethod
    def identity(cls, observed: Tensor, domain: SupportDomain) -> "PulseSupports":
        return cls(
            SupportMask(SupportKind.OBSERVED, observed, domain),
            SupportMask(SupportKind.EXPOSED, observed, domain),
            SupportMask(SupportKind.INTERVENED, torch.zeros_like(observed), domain),
            validity=observed,
        )

    @property
    def validity(self) -> Tensor:
        return self._validity.clone()

    def replace_intervened(self, mask: Tensor) -> "PulseSupports":
        """Return the same support snapshot with a new intervention plane."""

        if not isinstance(mask, Tensor) or mask.dtype != torch.bool:
            raise TypeError("intervened support must be a boolean Tensor")
        if tuple(mask.shape) != self.observed.domain.shape:
            raise ValueError("intervened support must match the support domain")
        if str(mask.device) != self.observed.domain.device:
            raise ValueError("intervened support must share the support device")
        return PulseSupports(
            self.observed,
            self.exposed,
            SupportMask(SupportKind.INTERVENED, mask, self.observed.domain),
            self.write,
            self.write_contract,
            validity=self._validity,
        )


@dataclass(frozen=True)
class FoldedPulseSupports:
    """Runtime-only transport of support through one exact FoldRecord."""

    active: PulseSupports
    preserved: PulseSupports
    original_domain: SupportDomain
    record: object
    write: SupportMask | None = None
    write_contract: OperandContract | None = None
    _runtime_contract_ref: ClassVar[str] = "arti/folded-pulse-supports@1"

    def replace_active(self, active: PulseSupports) -> "FoldedPulseSupports":
        """Replace only the active support partition without changing lineage."""

        if not isinstance(active, PulseSupports):
            raise TypeError("active must be PulseSupports")
        if active.observed.domain != self.active.observed.domain:
            raise ValueError("replacement active support belongs to a different domain")
        if not torch.equal(active._validity, self.active._validity):
            raise ValueError("replacement active support changed FoldRecord validity")
        return FoldedPulseSupports(
            active=active,
            preserved=self.preserved,
            original_domain=self.original_domain,
            record=self.record,
            write=self.write,
            write_contract=self.write_contract,
        )


def _transport_domain(
    source: SupportDomain,
    mask: Tensor,
    *,
    suffix: str,
) -> SupportDomain:
    return SupportDomain.for_tensor(
        mask,
        domain_id=f"{source.domain_id}-{suffix}",
        owner_ref="arti/fold@2",
        partition_id=f"{source.partition_id}-{suffix}",
        transition_id=source.transition_id,
        layout="packed",
        axis=-1,
    )


def fold_pulse_supports(supports: PulseSupports, record: object) -> FoldedPulseSupports:
    """Gather the complete support lattice with the FoldRecord permutation."""

    from .reversible_topology import FoldRecord

    if not isinstance(supports, PulseSupports) or not isinstance(record, FoldRecord):
        raise TypeError("support transport requires PulseSupports and FoldRecord")
    if tuple(supports._validity.shape) != tuple(record.original_shape[:-1]):
        raise ValueError("support shape does not match FoldRecord")
    validity_matches = torch.eq(
        supports._validity, record._trusted_original_mask()
    ).all()
    if torch.compiler.is_compiling():
        torch._assert_async(
            validity_matches,
            "support validity does not match FoldRecord mask lineage",
        )
    elif not bool(validity_matches):
        raise ValueError("support validity does not match FoldRecord mask lineage")
    permutation = record._trusted_permutation()

    def gather(value: Tensor) -> Tensor:
        return torch.gather(value, -1, permutation)

    packed_validity = gather(supports._validity)
    split = record.active_count
    active_validity = packed_validity[..., :split]
    preserved_validity = packed_validity[..., split:]
    active_domain = _transport_domain(
        supports.observed.domain,
        active_validity,
        suffix="active",
    )
    preserved_domain = _transport_domain(
        supports.observed.domain,
        preserved_validity,
        suffix="preserved",
    )

    def partition(source: SupportMask, domain: SupportDomain, start: int, end: int) -> SupportMask:
        return SupportMask(source.kind, gather(source._mask)[..., start:end], domain)

    active = PulseSupports(
        partition(supports.observed, active_domain, 0, split),
        partition(supports.exposed, active_domain, 0, split),
        partition(supports.intervened, active_domain, 0, split),
        validity=active_validity,
    )
    preserved = PulseSupports(
        partition(supports.observed, preserved_domain, split, record.original_length),
        partition(supports.exposed, preserved_domain, split, record.original_length),
        partition(supports.intervened, preserved_domain, split, record.original_length),
        validity=preserved_validity,
    )
    return FoldedPulseSupports(
        active=active,
        preserved=preserved,
        original_domain=supports.observed.domain,
        record=record,
        write=supports.write,
        write_contract=supports.write_contract,
    )


def unfold_pulse_supports(state: FoldedPulseSupports) -> PulseSupports:
    """Scatter a transported support lattice back through its exact record."""

    from .reversible_topology import FoldRecord

    if not isinstance(state, FoldedPulseSupports) or not isinstance(state.record, FoldRecord):
        raise TypeError("support reunion requires FoldedPulseSupports")
    record = state.record
    permutation = record._trusted_permutation()

    def restore(active: Tensor, preserved: Tensor) -> Tensor:
        packed = torch.cat((active, preserved), dim=-1)
        return torch.zeros_like(packed).scatter(-1, permutation, packed)

    validity = restore(state.active._validity, state.preserved._validity)
    if not torch.equal(validity, record._trusted_original_mask()):
        raise ValueError("reunited support validity violates FoldRecord lineage")

    def support(kind: SupportKind, active: SupportMask, preserved: SupportMask) -> SupportMask:
        return SupportMask(
            kind,
            restore(active._mask, preserved._mask),
            state.original_domain,
        )

    return PulseSupports(
        support(SupportKind.OBSERVED, state.active.observed, state.preserved.observed),
        support(SupportKind.EXPOSED, state.active.exposed, state.preserved.exposed),
        support(
            SupportKind.INTERVENED,
            state.active.intervened,
            state.preserved.intervened,
        ),
        state.write,
        state.write_contract,
        validity=validity,
    )


def _gather_active_pulse_supports(
    supports: PulseSupports,
    record: object,
) -> PulseSupports:
    """Gather only the active K support plane for Pulse's overlay backend."""

    from .reversible_topology import FoldRecord

    if not isinstance(supports, PulseSupports) or not isinstance(record, FoldRecord):
        raise TypeError("active support transport requires PulseSupports and FoldRecord")
    if tuple(supports._validity.shape) != tuple(record.original_shape[:-1]):
        raise ValueError("support shape does not match FoldRecord")
    validity_matches = torch.eq(
        supports._validity, record._trusted_original_mask()
    ).all()
    if torch.compiler.is_compiling():
        torch._assert_async(
            validity_matches,
            "support validity does not match FoldRecord mask lineage",
        )
    elif not bool(validity_matches):
        raise ValueError("support validity does not match FoldRecord mask lineage")
    active_index = record._trusted_permutation()[..., : record.active_count]

    def gather(value: Tensor) -> Tensor:
        return torch.gather(value, -1, active_index)

    active_validity = gather(supports._validity)
    active_domain = _transport_domain(
        supports.observed.domain,
        active_validity,
        suffix="active",
    )
    selected = torch.zeros_like(supports._validity).scatter(
        -1, active_index, torch.ones_like(active_index, dtype=torch.bool)
    )
    intervention_is_active = ~(supports.intervened._mask & ~selected).any()
    if torch.compiler.is_compiling():
        torch._assert_async(
            intervention_is_active,
            "Fold topology left intervention support outside the active plane",
        )
    elif not bool(intervention_is_active):
        raise ValueError("Fold topology must place every intervened value in active")

    def support(source: SupportMask) -> SupportMask:
        return SupportMask(source.kind, gather(source._mask), active_domain)

    return PulseSupports(
        support(supports.observed),
        support(supports.exposed),
        support(supports.intervened),
        supports.write,
        supports.write_contract,
        validity=active_validity,
    )


def _restore_active_pulse_supports(
    original: PulseSupports,
    active: PulseSupports,
    record: object,
) -> PulseSupports:
    """Restore active intervention support without materializing N-K support."""

    from .reversible_topology import FoldRecord

    if not isinstance(original, PulseSupports) or not isinstance(active, PulseSupports):
        raise TypeError("active support reunion requires PulseSupports")
    if not isinstance(record, FoldRecord):
        raise TypeError("active support reunion requires FoldRecord")
    active_index = record._trusted_permutation()[..., : record.active_count]
    expected_validity = torch.gather(original._validity, -1, active_index)
    validity_matches = torch.eq(active._validity, expected_validity).all()
    if torch.compiler.is_compiling():
        torch._assert_async(
            validity_matches,
            "active support validity violates FoldRecord lineage",
        )
    elif not bool(validity_matches):
        raise ValueError("active support validity violates FoldRecord lineage")
    intervened = original.intervened._mask.scatter(
        -1, active_index, active.intervened._mask
    )
    return PulseSupports(
        original.observed,
        original.exposed,
        SupportMask(
            SupportKind.INTERVENED,
            intervened,
            original.observed.domain,
        ),
        original.write,
        original.write_contract,
        validity=original._validity,
    )


def lift_observation_supports(
    supports: PulseSupports,
    observation: "TensorEnvelope",
) -> PulseSupports:
    """Lift world supports over an observation axis without changing instance order."""

    if not isinstance(supports, PulseSupports):
        raise TypeError("observation support lift requires PulseSupports")
    if not isinstance(observation, TensorEnvelope) or observation.ref is not EnvelopeRef.OBSERVATION:
        raise TypeError("observation support lift requires an OBSERVATION envelope")
    observed_validity = observation._mask_for_execution()
    if observed_validity.ndim != supports._validity.ndim + 1:
        raise ValueError("observation envelope must add exactly one trajectory axis")
    if (
        observed_validity.shape[0] != supports._validity.shape[0]
        or observed_validity.shape[2:] != supports._validity.shape[1:]
    ):
        raise ValueError("observation envelope does not preserve world instance layout")
    expected_validity = observed_validity & supports._validity.unsqueeze(1)
    validity_matches = torch.eq(expected_validity, observed_validity).all()
    if torch.compiler.is_compiling():
        torch._assert_async(
            validity_matches,
            "observation validity contains instances outside the world",
        )
    elif not bool(validity_matches):
        raise ValueError("observation validity contains instances outside the world")

    def lift(source: SupportMask) -> SupportMask:
        lifted = source._mask.unsqueeze(1) & observed_validity
        return SupportMask(source.kind, lifted, observation.domain)

    return PulseSupports(
        lift(supports.observed),
        lift(supports.exposed),
        lift(supports.intervened),
        supports.write,
        supports.write_contract,
        validity=observed_validity,
    )


class OperandKind(str, Enum):
    OBSERVATION = "observation"
    TOPOLOGY = "topology"
    INTERVENTION = "intervention"
    RECALL = "recall"
    WRITE = "write"


class OperandOwnership(str, Enum):
    BORROWED_TRAINING = "borrowed_training"
    OWNED_INFERENCE = "owned_inference"


@dataclass(frozen=True)
class OperandContract:
    """Immutable binding between a Bank partition and one consumer."""

    kind: OperandKind
    source_ref: str
    partition_id: str
    consumer_ref: str
    factor_dim: int
    layout: str
    domain: SupportDomain
    source_asset_fingerprint: str
    schema_version: int = TYPED_OPERANDS_SCHEMA_VERSION
    _runtime_contract_ref: ClassVar[str] = "arti/operand-contract@1"

    def __post_init__(self) -> None:
        if not isinstance(self.kind, OperandKind):
            raise TypeError("kind must be an OperandKind")
        ComponentRef.parse(self.source_ref)
        ComponentRef.parse(self.consumer_ref)
        if _IDENTIFIER.fullmatch(self.partition_id) is None:
            raise ValueError("partition_id must be a stable lowercase identifier")
        if isinstance(self.factor_dim, bool) or not isinstance(self.factor_dim, int) or self.factor_dim <= 0:
            raise ValueError("factor_dim must be positive")
        if self.layout not in {"dense", "packed"}:
            raise ValueError("operand layout must be 'dense' or 'packed'")
        if not isinstance(self.domain, SupportDomain):
            raise TypeError("domain must be a SupportDomain")
        if self.source_ref != self.domain.owner_ref:
            raise ValueError("operand source_ref must match its support domain owner_ref")
        if self.partition_id != self.domain.partition_id:
            raise ValueError("operand partition_id must match its support domain partition_id")
        if self.layout != self.domain.layout:
            raise ValueError("operand layout must match its support domain layout")
        if _SHA256.fullmatch(self.source_asset_fingerprint) is None:
            raise ValueError("source_asset_fingerprint must be a SHA-256 hex digest")
        if type(self.schema_version) is not int or self.schema_version != TYPED_OPERANDS_SCHEMA_VERSION:
            raise ValueError("unsupported operand contract schema version")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "ref": self._runtime_contract_ref,
            "kind": self.kind.value,
            "source_ref": self.source_ref,
            "partition_id": self.partition_id,
            "consumer_ref": self.consumer_ref,
            "factor_dim": self.factor_dim,
            "layout": self.layout,
            "domain": self.domain.metadata(),
            "source_asset_fingerprint": self.source_asset_fingerprint,
        }


class TypedOperands:
    """Snapshot of tensor operands with a source/consumer contract."""

    _runtime_contract_ref: ClassVar[str] = "arti/typed-operands@1"
    __slots__ = ("_sealed", "contract", "ownership", "_values", "_mask")

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("TypedOperands is immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        contract: OperandContract,
        values: Tensor,
        mask: Tensor,
        *,
        ownership: OperandOwnership = OperandOwnership.BORROWED_TRAINING,
        limits: ContractLimits = DEFAULT_CONTRACT_LIMITS,
    ) -> None:
        object.__setattr__(self, "_sealed", False)
        if not isinstance(contract, OperandContract):
            raise TypeError("contract must be an OperandContract")
        if not isinstance(ownership, OperandOwnership):
            raise TypeError("ownership must be an OperandOwnership")
        limits.admit_tensor(values, name="operand values")
        limits.admit_tensor(mask, name="operand mask")
        if values.ndim < 2 or values.shape[-1] != contract.factor_dim:
            raise ValueError("operand values do not match contract factor_dim")
        if mask.dtype != torch.bool or mask.shape != values.shape[:-1]:
            raise ValueError("operand mask must be boolean with shape values.shape[:-1]")
        if values.device != mask.device or str(values.device) != contract.domain.device:
            raise ValueError("operand values, mask, and domain must share device")
        if tuple(mask.shape) != contract.domain.shape:
            raise ValueError("operand mask does not match contract domain")
        object.__setattr__(self, "contract", contract)
        object.__setattr__(self, "ownership", ownership)
        object.__setattr__(
            self,
            "_values",
            values if ownership is OperandOwnership.BORROWED_TRAINING else values.detach().clone(),
        )
        object.__setattr__(self, "_mask", mask.detach().clone())
        object.__setattr__(self, "_sealed", True)

    @property
    def values(self) -> Tensor:
        """Return an inspection snapshot; operators must use consume()."""

        return self._values.clone()

    def snapshot_values(self) -> Tensor:
        return self._values.detach().clone()

    def consume(
        self,
        *,
        consumer: object,
        kind: OperandKind,
        source: object,
        partition_id: str,
        domain: SupportDomain,
        factor_dim: int,
        layout: str,
        source_asset_fingerprint: str,
    ) -> Tensor:
        """Return the execution tensor only after exact authority validation."""

        if not isinstance(kind, OperandKind):
            raise TypeError("kind must be an OperandKind")
        if not isinstance(domain, SupportDomain):
            raise TypeError("domain must be a SupportDomain")
        if torch.compiler.is_compiling():
            consumer_ref = getattr(consumer, "_component_reference", None)
            source_ref = getattr(source, "_component_reference", None)
            if not isinstance(consumer_ref, str) or not isinstance(source_ref, str):
                raise TypeError(
                    "compiled operand consumers and sources require canonical references"
                )
        else:
            consumer_ref = component_ref(consumer)
            source_ref = component_ref(source)
        expected = self.contract
        if consumer_ref != expected.consumer_ref:
            raise ValueError("operand consumer does not match its contract")
        if kind is not expected.kind:
            raise ValueError("operand kind does not match its contract")
        if source_ref != expected.source_ref:
            raise ValueError("operand source does not match its contract")
        if partition_id != expected.partition_id:
            raise ValueError("operand partition does not match its contract")
        if domain != expected.domain:
            raise ValueError("operand domain does not match its contract")
        if factor_dim != expected.factor_dim:
            raise ValueError("operand factor_dim does not match its contract")
        if layout != expected.layout:
            raise ValueError("operand layout does not match its contract")
        if source_asset_fingerprint != expected.source_asset_fingerprint:
            raise ValueError("operand source asset does not match its contract")
        return self._values

    @property
    def mask(self) -> Tensor:
        return self._mask.clone()


class OperandSource(Protocol):
    _component_reference: str
    def read_operands(self, query: Tensor, mask: Tensor) -> TypedOperands: ...


class ProposalPolicy(Protocol):
    _component_reference: str
    def propose(self, state: Tensor, operands: TypedOperands | None = None) -> object: ...


class FixedOperator(Protocol):
    _component_reference: str
    def apply(self, state: Tensor, proposal: object) -> object: ...


class StageMode(str, Enum):
    OFF = "off"
    ENABLED = "enabled"


class StageRole(str, Enum):
    OBSERVATION = "observation"
    HALF = "half"
    FOLD = "fold"
    INTERVENTION = "intervention"
    SELECTIVE_COMPUTE = "selective_compute"
    UNFOLD = "unfold"
    AGGREGATE = "aggregate"
    BANK_UPDATE = "bank_update"


class EnvelopeRef(str, Enum):
    """Versioned runtime envelope identities used by the top-level Pulse graph."""

    WORLD = "arti/world-envelope@1"
    OBSERVATION = "arti/observation-envelope@1"
    FOLDED = "arti/fold-state@1"
    REUNITED = "arti/unfolded-state@1"
    PULSE = "arti/pulse-output@1"


class TensorEnvelope:
    """Zero-copy runtime binding between tensor values, validity, and a domain."""

    _runtime_contract_ref: ClassVar[str] = "arti/tensor-envelope@1"
    __slots__ = ("_sealed", "ref", "value", "_mask", "domain")

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("TensorEnvelope is immutable")
        object.__setattr__(self, name, value)

    def __init__(
        self,
        ref: EnvelopeRef,
        value: Tensor,
        mask: Tensor,
        domain: SupportDomain,
        *,
        limits: ContractLimits | None = None,
    ) -> None:
        object.__setattr__(self, "_sealed", False)
        if not isinstance(ref, EnvelopeRef):
            raise TypeError("ref must be an EnvelopeRef")
        if not torch.compiler.is_compiling():
            active_limits = DEFAULT_CONTRACT_LIMITS if limits is None else limits
            active_limits.admit_tensor(value, name="envelope value")
            active_limits.admit_tensor(mask, name="envelope mask")
        if value.ndim < 2 or not (value.is_floating_point() or value.is_complex()):
            raise TypeError("envelope value must be a floating or complex [..., N, D] tensor")
        if mask.dtype != torch.bool or mask.shape != value.shape[:-1]:
            raise ValueError("envelope mask must be boolean with shape value.shape[:-1]")
        if value.device != mask.device:
            raise ValueError("envelope value and mask must share a device")
        if not isinstance(domain, SupportDomain):
            raise TypeError("domain must be a SupportDomain")
        if domain.shape != tuple(mask.shape) or domain.device != str(value.device):
            raise ValueError("envelope tensor does not match its support domain")
        object.__setattr__(self, "ref", ref)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "_mask", mask.detach().clone())
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "_sealed", True)

    @property
    def mask(self) -> Tensor:
        return self._mask.clone()

    def _mask_for_execution(self) -> Tensor:
        return self._mask

    def replace(self, value: Tensor) -> "TensorEnvelope":
        return TensorEnvelope(self.ref, value, self._mask, self.domain)


class OffSemantics(str, Enum):
    IDENTITY_OBSERVATION = "identity_observation"
    IDENTITY_VALUES = "identity_values"
    ALL_OBSERVED_EXPOSED = "all_observed_exposed"
    NO_INTERVENTION = "no_intervention"
    PRESERVE_INTERVENTION_SUPPORT = "preserve_intervention_support"
    NO_COMPUTE = "no_compute"
    REUNION_BYPASS = "reunion_bypass"
    NO_AGGREGATION = "no_aggregation"
    NO_BANK_UPDATE = "no_bank_update"


_STAGE_ORDER = {role: index for index, role in enumerate(StageRole)}
_ROLE_CAPABILITY = {role: f"pulse.stage.{role.value.replace('_', '-')}" for role in StageRole}
_ROLE_TRANSITIONS = {
    StageRole.OBSERVATION: frozenset({(EnvelopeRef.WORLD, EnvelopeRef.OBSERVATION)}),
    StageRole.HALF: frozenset(
        {
            (EnvelopeRef.WORLD, EnvelopeRef.WORLD),
            (EnvelopeRef.OBSERVATION, EnvelopeRef.OBSERVATION),
        }
    ),
    StageRole.FOLD: frozenset(
        {
            (EnvelopeRef.WORLD, EnvelopeRef.FOLDED),
            (EnvelopeRef.OBSERVATION, EnvelopeRef.FOLDED),
        }
    ),
    StageRole.INTERVENTION: frozenset({(EnvelopeRef.FOLDED, EnvelopeRef.FOLDED)}),
    StageRole.SELECTIVE_COMPUTE: frozenset({(EnvelopeRef.FOLDED, EnvelopeRef.FOLDED)}),
    StageRole.UNFOLD: frozenset({(EnvelopeRef.FOLDED, EnvelopeRef.REUNITED)}),
    StageRole.AGGREGATE: frozenset(
        {
            (EnvelopeRef.WORLD, EnvelopeRef.PULSE),
            (EnvelopeRef.OBSERVATION, EnvelopeRef.PULSE),
            (EnvelopeRef.REUNITED, EnvelopeRef.PULSE),
        }
    ),
    StageRole.BANK_UPDATE: frozenset({(EnvelopeRef.PULSE, EnvelopeRef.PULSE)}),
}
_ROLE_OFF = {
    StageRole.OBSERVATION: OffSemantics.IDENTITY_OBSERVATION,
    StageRole.HALF: OffSemantics.IDENTITY_VALUES,
    StageRole.FOLD: OffSemantics.ALL_OBSERVED_EXPOSED,
    StageRole.INTERVENTION: OffSemantics.PRESERVE_INTERVENTION_SUPPORT,
    StageRole.SELECTIVE_COMPUTE: OffSemantics.NO_COMPUTE,
    StageRole.UNFOLD: OffSemantics.REUNION_BYPASS,
    StageRole.AGGREGATE: OffSemantics.NO_AGGREGATION,
    StageRole.BANK_UPDATE: OffSemantics.NO_BANK_UPDATE,
}
_RESERVED_CONFIG_KEYS = frozenset({"enabled", "mode", "ref", "role", "stage_id", "pair_id"})


@dataclass(frozen=True)
class TopologyBinding:
    """Exact Fold/UnFold transport identity shared by the paired operations."""

    topology_ref: str
    topology_config_fingerprint: str
    producer_provenance_fingerprint: str
    producer_ref: str = "arti/fold@2"
    inverse_ref: str = "arti/unfold@2"
    _runtime_contract_ref: ClassVar[str] = "arti/topology-binding@1"

    def __post_init__(self) -> None:
        for reference in (self.topology_ref, self.producer_ref, self.inverse_ref):
            ComponentRef.parse(reference)
        if _SHA256.fullmatch(self.topology_config_fingerprint) is None:
            raise ValueError("topology_config_fingerprint must be a SHA-256 hex digest")
        if _SHA256.fullmatch(self.producer_provenance_fingerprint) is None:
            raise ValueError("producer_provenance_fingerprint must be a SHA-256 hex digest")
        if self.topology_ref != "arti/reversible-topology@1":
            raise ValueError("TopologyBinding@1 requires ReversibleTopology@1")
        if self.producer_ref != "arti/fold@2" or self.inverse_ref != "arti/unfold@2":
            raise ValueError("TopologyBinding@1 requires Fold@2 and UnFold@2")

    def to_dict(self) -> dict[str, str]:
        return {
            "ref": self._runtime_contract_ref,
            "topology_ref": self.topology_ref,
            "topology_config_fingerprint": self.topology_config_fingerprint,
            "producer_provenance_fingerprint": self.producer_provenance_fingerprint,
            "producer_ref": self.producer_ref,
            "inverse_ref": self.inverse_ref,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TopologyBinding":
        required = {
            "ref",
            "topology_ref",
            "topology_config_fingerprint",
            "producer_provenance_fingerprint",
            "producer_ref",
            "inverse_ref",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("topology binding has missing or unknown fields")
        if value["ref"] != cls._runtime_contract_ref:
            raise ValueError("unsupported topology binding")
        return cls(
            topology_ref=value["topology_ref"],
            topology_config_fingerprint=value["topology_config_fingerprint"],
            producer_provenance_fingerprint=value["producer_provenance_fingerprint"],
            producer_ref=value["producer_ref"],
            inverse_ref=value["inverse_ref"],
        )

    def validate_record(self, record: object) -> None:
        expected = {
            "topology_ref": self.topology_ref,
            "topology_config_fingerprint": self.topology_config_fingerprint,
            "producer_provenance_fingerprint": self.producer_provenance_fingerprint,
            "producer_ref": self.producer_ref,
            "inverse_ref": self.inverse_ref,
        }
        for name, value in expected.items():
            if getattr(record, name, None) != value:
                raise ValueError(f"FoldRecord does not match topology binding field {name}")


@dataclass(frozen=True, init=False)
class PulseStageSpec:
    """One immutable, ordered Pulse stage declaration."""

    stage_id: str
    role: StageRole
    mode: StageMode
    component_ref: str | None
    input_schema: EnvelopeRef
    output_schema: EnvelopeRef
    pair_id: str | None
    topology_binding: TopologyBinding | None
    off_semantics: OffSemantics | None
    _config_json: str = field(repr=False)
    _config_fingerprint: str = field(repr=False)
    _component_reference: ClassVar[str] = "arti/pulse-stage@1"

    def __init__(
        self,
        *,
        stage_id: str,
        role: StageRole,
        mode: StageMode,
        component_ref: str | None,
        input_schema: EnvelopeRef,
        output_schema: EnvelopeRef,
        config: Mapping[str, Any] | None = None,
        pair_id: str | None = None,
        topology_binding: TopologyBinding | None = None,
        off_semantics: OffSemantics | None = None,
        limits: ContractLimits = DEFAULT_CONTRACT_LIMITS,
    ) -> None:
        if _IDENTIFIER.fullmatch(stage_id) is None:
            raise ValueError("stage_id must be a stable lowercase identifier")
        if not isinstance(role, StageRole) or not isinstance(mode, StageMode):
            raise TypeError("role and mode must be versioned enums")
        if not isinstance(input_schema, EnvelopeRef) or not isinstance(output_schema, EnvelopeRef):
            raise TypeError("input_schema and output_schema must be EnvelopeRef values")
        if mode is StageMode.OFF:
            if input_schema is not output_schema:
                raise ValueError("OFF stages must preserve their envelope exactly")
        else:
            if (input_schema, output_schema) not in _ROLE_TRANSITIONS[role]:
                raise ValueError("stage envelope transition does not match its role")
        if pair_id is not None and _IDENTIFIER.fullmatch(pair_id) is None:
            raise ValueError("pair_id must be a stable lowercase identifier")
        paired_role = role in {StageRole.FOLD, StageRole.UNFOLD}
        if mode is StageMode.ENABLED and paired_role and pair_id is None:
            raise ValueError("enabled Fold and UnFold stages require pair_id")
        if (not paired_role or mode is StageMode.OFF) and pair_id is not None:
            raise ValueError("pair_id is reserved for Fold and UnFold")
        if mode is StageMode.ENABLED and paired_role:
            if not isinstance(topology_binding, TopologyBinding):
                raise ValueError("enabled Fold and UnFold stages require topology_binding")
        elif topology_binding is not None:
            raise ValueError("topology_binding is reserved for enabled Fold and UnFold")
        if mode is StageMode.OFF:
            if component_ref is not None:
                raise ValueError("OFF stages must not declare component_ref")
            if off_semantics is not _ROLE_OFF[role]:
                raise ValueError("OFF stage does not declare required off semantics")
        else:
            if component_ref is None:
                raise ValueError("enabled stages require component_ref")
            registration = get_component_registry().registration_for_reference(component_ref)
            if _ROLE_CAPABILITY[role] not in registration.capabilities:
                raise ValueError("component_ref is not authorized for stage role")
            if role is StageRole.FOLD and component_ref != topology_binding.producer_ref:
                raise ValueError("Fold stage component does not match topology binding")
            if role is StageRole.UNFOLD and component_ref != topology_binding.inverse_ref:
                raise ValueError("UnFold stage component does not match topology binding")
            if off_semantics is not None:
                raise ValueError("enabled stages must not declare off semantics")
        config_value = {} if config is None else dict(config)
        if _RESERVED_CONFIG_KEYS & set(config_value):
            raise ValueError("stage config contains reserved structural keys")
        config_json = _canonical_json(config_value, limits)
        object.__setattr__(self, "stage_id", stage_id)
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "component_ref", component_ref)
        object.__setattr__(self, "input_schema", input_schema)
        object.__setattr__(self, "output_schema", output_schema)
        object.__setattr__(self, "pair_id", pair_id)
        object.__setattr__(self, "topology_binding", topology_binding)
        object.__setattr__(self, "off_semantics", off_semantics)
        object.__setattr__(self, "_config_json", config_json)
        object.__setattr__(self, "_config_fingerprint", _fingerprint_json(config_json))

    @property
    def config(self) -> dict[str, Any]:
        return json.loads(self._config_json)

    @property
    def config_fingerprint(self) -> str:
        return self._config_fingerprint

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": PULSE_STAGE_SCHEMA_VERSION,
            "ref": self._component_reference,
            "stage_id": self.stage_id,
            "role": self.role.value,
            "mode": self.mode.value,
            "component_ref": self.component_ref,
            "input_schema": self.input_schema.value,
            "output_schema": self.output_schema.value,
            "config": self.config,
            "config_fingerprint": self.config_fingerprint,
            "pair_id": self.pair_id,
            "topology_binding": (
                None if self.topology_binding is None else self.topology_binding.to_dict()
            ),
            "off_semantics": None if self.off_semantics is None else self.off_semantics.value,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PulseStageSpec":
        required = {
            "schema_version", "ref", "stage_id", "role", "mode", "component_ref",
            "input_schema", "output_schema", "config", "config_fingerprint", "pair_id",
            "topology_binding",
            "off_semantics",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ValueError("Pulse stage payload has missing or unknown fields")
        if (
            type(value["schema_version"]) is not int
            or value["schema_version"] != PULSE_STAGE_SCHEMA_VERSION
            or value["ref"] != cls._component_reference
        ):
            raise ValueError("unsupported Pulse stage payload")
        config_json = _canonical_json(value["config"])
        if value["config_fingerprint"] != _fingerprint_json(config_json):
            raise ValueError("Pulse stage config fingerprint is invalid")
        return cls(
            stage_id=value["stage_id"], role=StageRole(value["role"]),
            mode=StageMode(value["mode"]), component_ref=value["component_ref"],
            input_schema=EnvelopeRef(value["input_schema"]),
            output_schema=EnvelopeRef(value["output_schema"]),
            config=value["config"], pair_id=value["pair_id"],
            topology_binding=(
                None
                if value["topology_binding"] is None
                else TopologyBinding.from_dict(value["topology_binding"])
            ),
            off_semantics=None if value["off_semantics"] is None else OffSemantics(value["off_semantics"]),
        )


@dataclass(frozen=True)
class PulseStageGraph:
    """Immutable ordered Pulse composition contract with strict JSON decoding."""

    stages: tuple[PulseStageSpec, ...]
    schema_version: int = PULSE_STAGE_GRAPH_SCHEMA_VERSION
    _fingerprint: str = field(init=False, repr=False)
    _component_reference: ClassVar[str] = "arti/pulse-stage-graph@1"

    def __post_init__(self) -> None:
        stages = tuple(self.stages)
        object.__setattr__(self, "stages", stages)
        if type(self.schema_version) is not int or self.schema_version != PULSE_STAGE_GRAPH_SCHEMA_VERSION:
            raise ValueError("unsupported Pulse stage graph schema version")
        if not stages or len(stages) > DEFAULT_CONTRACT_LIMITS.max_stages:
            raise ValueError("Pulse stage graph has invalid stage count")
        if any(not isinstance(stage, PulseStageSpec) for stage in stages):
            raise TypeError("Pulse stage graph entries must be PulseStageSpec")
        ids, roles = [stage.stage_id for stage in stages], [stage.role for stage in stages]
        if len(set(ids)) != len(ids):
            raise ValueError("Pulse stage IDs must be unique")
        if [_STAGE_ORDER[role] for role in roles] != sorted(_STAGE_ORDER[role] for role in roles):
            raise ValueError("Pulse stages violate canonical execution order")
        required_roles = {
            StageRole.OBSERVATION,
            StageRole.FOLD,
            StageRole.UNFOLD,
            StageRole.AGGREGATE,
        }
        if not required_roles.issubset(roles):
            raise ValueError(
                "Pulse stage graph requires observation, fold, unfold, and aggregate"
            )
        if roles[0] is not StageRole.OBSERVATION:
            raise ValueError("Pulse stage graph must begin with observation")
        if roles[-1] not in {StageRole.AGGREGATE, StageRole.BANK_UPDATE}:
            raise ValueError("Pulse stage graph must end with aggregate or bank_update")
        for previous, current in zip(stages, stages[1:], strict=False):
            if previous.output_schema != current.input_schema:
                raise ValueError("adjacent Pulse stage schemas do not compose")
        folds = [
            stage for stage in stages
            if stage.role is StageRole.FOLD and stage.mode is StageMode.ENABLED
        ]
        unfolds = [
            stage for stage in stages
            if stage.role is StageRole.UNFOLD and stage.mode is StageMode.ENABLED
        ]
        if len(folds) != len(unfolds):
            raise ValueError("enabled Fold and UnFold stages must be paired")
        if [stage.pair_id for stage in unfolds] != [stage.pair_id for stage in reversed(folds)]:
            raise ValueError("enabled Fold and UnFold pairs must close in LIFO order")
        fold_by_pair = {stage.pair_id: stage for stage in folds}
        if len(fold_by_pair) != len(folds):
            raise ValueError("enabled Fold pair_id values must be unique")
        for unfold in unfolds:
            fold = fold_by_pair[unfold.pair_id]
            if fold.topology_binding != unfold.topology_binding:
                raise ValueError("Fold and UnFold must share topology binding")
        intervention_enabled = any(
            stage.role is StageRole.INTERVENTION and stage.mode is StageMode.ENABLED
            for stage in stages
        )
        selective_enabled = any(
            stage.role is StageRole.SELECTIVE_COMPUTE and stage.mode is StageMode.ENABLED
            for stage in stages
        )
        if intervention_enabled and not selective_enabled:
            raise ValueError("enabled intervention requires enabled selective compute")
        content = {"schema_version": self.schema_version, "ref": self._component_reference, "stages": [stage.to_dict() for stage in stages]}
        object.__setattr__(self, "_fingerprint", _fingerprint_json(_canonical_json(content)))

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    @property
    def enabled_dependencies(self) -> tuple[str, ...]:
        return tuple(stage.component_ref for stage in self.stages if stage.mode is StageMode.ENABLED and stage.component_ref is not None)

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "ref": self._component_reference, "stages": [stage.to_dict() for stage in self.stages], "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PulseStageGraph":
        if not isinstance(value, Mapping) or set(value) != {"schema_version", "ref", "stages", "fingerprint"}:
            raise ValueError("Pulse stage graph has missing or unknown fields")
        if (
            type(value["schema_version"]) is not int
            or value["schema_version"] != PULSE_STAGE_GRAPH_SCHEMA_VERSION
            or value["ref"] != cls._component_reference
        ):
            raise ValueError("unsupported Pulse stage graph payload")
        if not isinstance(value["stages"], list):
            raise ValueError("Pulse stage graph stages must be a list")
        if not value["stages"] or len(value["stages"]) > DEFAULT_CONTRACT_LIMITS.max_stages:
            raise ValueError("Pulse stage graph has invalid stage count")
        graph = cls(tuple(PulseStageSpec.from_dict(stage) for stage in value["stages"]))
        if value["fingerprint"] != graph.fingerprint:
            raise ValueError("Pulse stage graph fingerprint is invalid")
        return graph


def apply_intervention(
    base: TensorEnvelope,
    candidate: TensorEnvelope,
    supports: PulseSupports,
    *,
    limits: ContractLimits = DEFAULT_CONTRACT_LIMITS,
) -> TensorEnvelope:
    """Commit candidate values only where Formula intervention is authorized."""

    if not isinstance(supports, PulseSupports):
        raise TypeError("intervention merge requires an admitted PulseSupports lattice")
    if not isinstance(base, TensorEnvelope) or not isinstance(candidate, TensorEnvelope):
        raise TypeError("intervention merge requires TensorEnvelope inputs")
    _admit_intervention_operation(
        base.value,
        candidate.value,
        supports.intervened._mask,
        limits,
    )
    if base.ref is not candidate.ref or base.domain != candidate.domain:
        raise ValueError("base and candidate must share envelope identity and domain")
    if not torch.equal(candidate._mask, base._mask):
        raise ValueError("base and candidate must share the same validity snapshot")
    if (
        base.value.shape != candidate.value.shape
        or base.value.dtype != candidate.value.dtype
        or base.value.device != candidate.value.device
    ):
        raise ValueError("base and candidate must share shape, dtype, and device")
    support = supports.intervened
    if support.domain != base.domain:
        raise ValueError("intervention support domain does not match the value envelope")
    if not torch.equal(supports._validity, base._mask):
        raise ValueError("support validity does not match the value envelope")
    value = _apply_intervention_values(base.value, candidate.value, support._mask)
    return TensorEnvelope(base.ref, value, base._mask, base.domain, limits=limits)


def _apply_intervention_values(base: Tensor, candidate: Tensor, mask: Tensor) -> Tensor:
    """Compile-friendly kernel; admission and authority remain in apply_intervention."""

    return torch.where(mask.unsqueeze(-1), candidate, base)


class InterventionOperator(nn.Module):
    """Fixed, compile-friendly commit operator created from admitted support."""

    _component_reference: ClassVar[str] = "arti/intervention-operator@1"

    def __init__(self, supports: PulseSupports) -> None:
        super().__init__()
        if not isinstance(supports, PulseSupports):
            raise TypeError("InterventionOperator requires admitted PulseSupports")
        self._domain_template = supports.intervened.domain
        self._support_shape = supports.intervened.domain.shape
        self.register_buffer("support", supports.intervened._mask.clone(), persistent=False)

    @property
    def domain(self) -> SupportDomain:
        template = self._domain_template
        if template.device == str(self.support.device):
            return template
        return SupportDomain(
            domain_id=template.domain_id,
            owner_ref=template.owner_ref,
            partition_id=template.partition_id,
            transition_id=template.transition_id,
            layout=template.layout,
            shape=template.shape,
            device=str(self.support.device),
            axis=template.axis,
        )

    def forward(self, base: Tensor, candidate: Tensor) -> Tensor:
        if base.shape != candidate.shape or base.dtype != candidate.dtype:
            raise ValueError("base and candidate must share shape and dtype")
        if base.device != candidate.device or base.device != self.support.device:
            raise ValueError("base, candidate, and intervention support must share device")
        if tuple(base.shape[:-1]) != self._support_shape:
            raise ValueError("intervention values do not match admitted support shape")
        _admit_intervention_operation(
            base,
            candidate,
            self.support,
            DEFAULT_CONTRACT_LIMITS,
        )
        return _apply_intervention_values(base, candidate, self.support)


def assert_unsupported_identity(
    before: TensorEnvelope,
    after: TensorEnvelope,
    supports: PulseSupports,
) -> None:
    """Diagnostic assertion; not for compiled or latency-sensitive paths."""

    if not isinstance(before, TensorEnvelope) or not isinstance(after, TensorEnvelope):
        raise TypeError("identity validation requires TensorEnvelope inputs")
    if before.ref is not after.ref or before.domain != after.domain:
        raise ValueError("before and after must share envelope identity and domain")
    if (
        before.value.shape != after.value.shape
        or before.value.dtype != after.value.dtype
        or before.value.device != after.value.device
    ):
        raise ValueError("before and after must share shape, dtype, and device")
    support = supports.intervened
    if support.domain != before.domain:
        raise ValueError("intervention support domain does not match the value envelope")
    masked_before = torch.where(
        support._mask.unsqueeze(-1), torch.zeros_like(before.value), before.value
    )
    masked_after = torch.where(
        support._mask.unsqueeze(-1), torch.zeros_like(after.value), after.value
    )
    if not torch.equal(masked_before, masked_after):
        raise ValueError("values outside intervention support must remain exactly unchanged")


__all__ = [
    "ContractLimits", "DEFAULT_CONTRACT_LIMITS", "EnvelopeRef", "FixedOperator",
    "FoldedPulseSupports", "InterventionOperator", "OffSemantics",
    "OperandContract", "OperandKind", "OperandOwnership", "OperandSource",
    "PULSE_STAGE_GRAPH_SCHEMA_VERSION",
    "PULSE_STAGE_SCHEMA_VERSION", "ProposalPolicy", "PulseStageGraph", "PulseStageSpec",
    "PulseSupports", "StageMode", "StageRole", "SUPPORT_SCHEMA_VERSION", "SupportDomain",
    "SupportKind", "SupportMask", "TYPED_OPERANDS_SCHEMA_VERSION", "TensorEnvelope",
    "TopologyBinding", "TypedOperands", "apply_intervention", "assert_unsupported_identity",
    "fold_pulse_supports", "lift_observation_supports", "unfold_pulse_supports",
]
