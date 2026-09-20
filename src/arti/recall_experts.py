"""Immutable Recall bank assets and deterministic bank assemblies."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field as dataclass_field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, Iterable, Mapping

import torch
import torch.nn as nn
from torch import Tensor

from .component_registry import (
    canonical_contract_reference,
    component_provenance,
    validate_component_provenance,
)
from .serialization import ARTISaveResult, load, save


RECALL_BANK_ARTIFACT_KIND = "arti.recall-bank"
RECALL_BANK_ARTIFACT_VERSION = 4
RECALL_BANK_PROVENANCE_VERSION = 1
_OVERFLOW_POLICIES = ("abstain", "shard", "allow")
_PRIVATE_MODULE_NAME = "_arti_private"
_PRIVATE_STATE_PREFIX = f"{_PRIVATE_MODULE_NAME}."
_EXPERT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _canonical_formula_reference(reference: str) -> str:
    """Resolve a Formula input into the identity stored by Bank provenance."""

    from .recall_formula import resolve_builtin_formula
    from .recall_registry import RecallFormulaRegistryError, resolve_formula

    builtin = resolve_builtin_formula(reference)
    if builtin is not None:
        assert builtin.contract.identity is not None
        return builtin.contract.identity.reference
    try:
        return resolve_formula(reference).reference
    except RecallFormulaRegistryError:
        return canonical_contract_reference(reference)


@dataclass(frozen=True)
class RecallCapacityDecision:
    """A deterministic allocation decision for bounded Recall bank storage."""

    requested_items: int
    accepted_items: int
    dropped_items: int
    expert_item_counts: tuple[int, ...]
    overflowed: bool
    protected: bool


@dataclass(frozen=True)
class RecallCapacityPlan:
    """Declare bounded Recall bank storage without prescribing a training loop."""

    slots_per_expert: int
    experts: int = 1
    overflow_policy: str = "abstain"

    def validate(self) -> "RecallCapacityPlan":
        if self.slots_per_expert <= 0 or self.experts <= 0:
            raise ValueError("Recall capacity slots_per_expert and experts must be positive")
        if self.overflow_policy not in _OVERFLOW_POLICIES:
            raise ValueError(f"overflow_policy must be one of {_OVERFLOW_POLICIES}")
        return self

    @property
    def total_capacity(self) -> int:
        return self.slots_per_expert * self.experts

    def decide(self, item_count: int) -> RecallCapacityDecision:
        """Allocate an item count without inspecting tensor contents."""

        self.validate()
        if item_count < 0:
            raise ValueError("item_count must be non-negative")
        overflowed = item_count > self.total_capacity
        if overflowed and self.overflow_policy == "abstain":
            return RecallCapacityDecision(item_count, 0, item_count, (), True, True)
        accepted = item_count if self.overflow_policy == "allow" else min(item_count, self.total_capacity)
        counts = []
        remaining = accepted
        for _ in range(self.experts):
            count = min(self.slots_per_expert, remaining)
            counts.append(count)
            remaining -= count
        return RecallCapacityDecision(
            requested_items=item_count,
            accepted_items=accepted,
            dropped_items=item_count - accepted,
            expert_item_counts=tuple(counts),
            overflowed=overflowed,
            protected=self.overflow_policy != "allow",
        )


def module_structure_fingerprint(module: nn.Module) -> str:
    """Fingerprint module topology and state shapes without learned values."""

    payload = {
        "class": f"{module.__class__.__module__}.{module.__class__.__qualname__}",
        "state": [
            {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in module.state_dict().items()
        ],
    }
    return _sha256_json(payload)


def module_behavior_fingerprint(module: nn.Module) -> str:
    """Fingerprint declarative runtime behavior without learned tensor values."""

    fields = (
        "dim",
        "hidden_dim",
        "slots",
        "rank",
        "key_dim",
        "group_size",
        "group_topk",
        "composition_factor",
        "value_composition",
        "routing",
        "query_mode",
        "query_seed",
        "recognition_mode",
        "use_half",
        "output_semantics",
    )
    modules: list[dict[str, Any]] = []
    for name, child in module.named_modules():
        record: dict[str, Any] = {
            "name": name,
            "class": f"{child.__class__.__module__}.{child.__class__.__qualname__}",
        }
        for field_name in fields:
            value = getattr(child, field_name, None)
            if isinstance(value, (str, int, float, bool)):
                record[field_name] = value
        query_contract = getattr(child, "query_contract", None)
        if isinstance(query_contract, Mapping):
            record["query_contract"] = dict(query_contract)
        formula_contract = getattr(child, "formula_contract", None)
        if formula_contract is not None and callable(getattr(formula_contract, "to_dict", None)):
            record["formula_contract"] = formula_contract.to_dict()
        for name_attr in ("factor_names", "route_names", "factor_route_names"):
            value = getattr(child, name_attr, None)
            if isinstance(value, (tuple, list)):
                record[name_attr] = list(value)
        modules.append(record)
    return _sha256_json({"modules": modules})


def _stable_component_provenance(module: nn.Module) -> dict[str, Any]:
    """Capture component schema without binding it to transient trainability."""

    flags = [(parameter, parameter.requires_grad) for parameter in module.parameters()]
    try:
        for parameter, _requires_grad in flags:
            parameter.requires_grad_(True)
        return component_provenance(module)
    finally:
        for parameter, requires_grad in flags:
            parameter.requires_grad_(requires_grad)


def _module_provenance_descriptor(module: nn.Module) -> dict[str, Any]:
    content = {
        "module_class": f"{module.__class__.__module__}.{module.__class__.__qualname__}",
        "component_provenance": _stable_component_provenance(module),
        "structure_fingerprint": module_structure_fingerprint(module),
        "behavior_fingerprint": module_behavior_fingerprint(module),
    }
    return {**content, "fingerprint": _sha256_json(content)}


def _formula_provenance_descriptor(
    expert: nn.Module,
    formula: str | nn.Module | None,
) -> dict[str, Any] | None:
    manifest: Mapping[str, Any] | None = None
    lock: Mapping[str, Any] | None = None
    formula_module: nn.Module | None = None
    reference: str | None = None
    if formula is None:
        reference_value = getattr(expert, "formula_id", None)
        reference = reference_value if isinstance(reference_value, str) else None
        manifest_factory = getattr(expert, "formula_manifest", None)
        if callable(manifest_factory):
            candidate = manifest_factory()
            if callable(getattr(candidate, "to_dict", None)):
                manifest = candidate.to_dict()
        lock_factory = getattr(expert, "formula_lock", None)
        if callable(getattr(lock_factory, "to_dict", None)):
            lock = lock_factory.to_dict()
        candidate_module = getattr(expert, "formula", None)
        if isinstance(candidate_module, nn.Module):
            formula_module = candidate_module
    elif isinstance(formula, str):
        reference = formula
    elif isinstance(formula, nn.Module):
        formula_module = formula
        candidate_contract = getattr(formula, "recall_formula_contract", None)
        if candidate_contract is not None and callable(getattr(candidate_contract, "to_dict", None)):
            manifest = {"contract": candidate_contract.to_dict()}
    else:
        raise TypeError("formula must be a formula reference, module, or None")

    if reference == "custom":
        reference = None
    elif reference is not None:
        reference = _canonical_formula_reference(reference)

    if reference is None and manifest is None and formula_module is None:
        return None
    content: dict[str, Any] = {
        "reference": reference,
        "manifest": None if manifest is None else dict(manifest),
        "lock": None if lock is None else dict(lock),
        "module": (
            None
            if formula_module is None
            else _module_provenance_descriptor(formula_module)
        ),
    }
    return {**content, "fingerprint": _sha256_json(content)}


def _validate_provenance_descriptor(
    name: str,
    value: Mapping[str, Any],
    *,
    require_components: bool = False,
) -> None:
    if not isinstance(value, Mapping):
        raise TypeError(f"Recall Bank {name} provenance must be a mapping")
    fingerprint = value.get("fingerprint")
    if not isinstance(fingerprint, str) or not _is_sha256(fingerprint):
        raise ValueError(f"Recall Bank {name} provenance fingerprint is invalid")
    content = dict(value)
    content.pop("fingerprint", None)
    if fingerprint != _sha256_json(content):
        raise ValueError(f"Recall Bank {name} provenance fingerprint does not match its contents")
    if require_components:
        graph = value.get("component_provenance")
        if not isinstance(graph, Mapping):
            raise ValueError(f"Recall Bank {name} provenance is missing component provenance")
        validate_component_provenance(graph)
    module_class = value.get("module_class")
    if not isinstance(module_class, str) or not module_class:
        raise ValueError(f"Recall Bank {name} provenance module_class is invalid")
    for field_name in ("structure_fingerprint", "behavior_fingerprint"):
        if not _is_sha256(value.get(field_name, "")):
            raise ValueError(f"Recall Bank {name} provenance {field_name} is invalid")


def _validate_formula_descriptor(value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("Recall Bank formula provenance must be a mapping")
    required = {"reference", "manifest", "lock", "module", "fingerprint"}
    if set(value) != required:
        raise ValueError(
            "Recall Bank formula provenance keys must be exactly "
            f"{sorted(required)}"
        )
    fingerprint = value["fingerprint"]
    content = dict(value)
    content.pop("fingerprint", None)
    if not isinstance(fingerprint, str) or not _is_sha256(fingerprint):
        raise ValueError("Recall Bank formula provenance fingerprint is invalid")
    if fingerprint != _sha256_json(content):
        raise ValueError(
            "Recall Bank formula provenance fingerprint does not match its contents"
        )
    if value["reference"] is not None:
        if not isinstance(value["reference"], str):
            raise ValueError("Recall Bank formula provenance reference is invalid")
        if _canonical_formula_reference(value["reference"]) != value["reference"]:
            raise ValueError("Recall Bank formula provenance reference is not canonical")
    for name in ("manifest", "lock"):
        if value[name] is not None and not isinstance(value[name], Mapping):
            raise ValueError(f"Recall Bank formula provenance {name} is invalid")
    if value["module"] is not None:
        _validate_provenance_descriptor(
            "formula module",
            value["module"],
            require_components=True,
        )


def _validate_bank_layout_payload(value: Mapping[str, Any]) -> None:
    if not isinstance(value, Mapping):
        raise TypeError("Recall Bank layout provenance must be a mapping")
    required = {"banks", "dimensions"}
    if set(value) != required:
        raise ValueError(
            "Recall Bank layout provenance keys must be exactly "
            f"{sorted(required)}"
        )
    banks = value["banks"]
    dimensions = value["dimensions"]
    if not isinstance(banks, list) or any(not isinstance(item, Mapping) for item in banks):
        raise ValueError("Recall Bank layout banks must be a list of mappings")
    member_keys = {"name", "concat_dim", "shape", "dtype"}
    for item in banks:
        if set(item) != member_keys:
            raise ValueError(
                "Recall Bank layout bank keys must be exactly "
                f"{sorted(member_keys)}"
            )
        RecallBankMember(
            name=str(item["name"]),
            concat_dim=int(item["concat_dim"]),
            shape=tuple(int(size) for size in item["shape"]),
            dtype=str(item["dtype"]),
        ).validate()
    if not isinstance(dimensions, Mapping):
        raise ValueError("Recall Bank layout dimensions must be a mapping")
    required_dimensions = {
        "slots",
        "hidden_dim",
        "composition_factor",
        "factor_names",
        "route_names",
    }
    if set(dimensions) != required_dimensions:
        raise ValueError(
            "Recall Bank layout dimension keys must be exactly "
            f"{sorted(required_dimensions)}"
        )
    for name in ("slots", "hidden_dim"):
        if dimensions[name] is not None and (
            type(dimensions[name]) is not int or dimensions[name] <= 0
        ):
            raise ValueError(f"Recall Bank layout dimension {name} is invalid")
    if type(dimensions["composition_factor"]) is not int or dimensions["composition_factor"] <= 0:
        raise ValueError("Recall Bank layout composition_factor is invalid")
    for name in ("factor_names", "route_names"):
        if not isinstance(dimensions[name], list) or any(
            not isinstance(item, str) for item in dimensions[name]
        ):
            raise ValueError(f"Recall Bank layout {name} must be a list of strings")


def recall_bank_artifact_path(path: str | Path) -> Path:
    target = Path(path)
    if not target.name.endswith(".recall.arti.st"):
        raise ValueError(
            "Recall bank artifacts must end in '.recall.arti.st' "
            "(for example coder.recall.arti.st)"
        )
    return target


class RecallBankError(ValueError):
    """Structured rejection for an invalid or incompatible Recall bank asset."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        path: str | Path | None = None,
        field: str | None = None,
        expected: Any = None,
        actual: Any = None,
        action: str = "re-export the bank with the active ARTI version",
    ) -> None:
        self.code = code
        self.path = None if path is None else str(path)
        self.field = field
        self.expected = expected
        self.actual = actual
        self.action = action
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code,
            "path": self.path,
            "field": self.field,
            "expected": self.expected,
            "actual": self.actual,
            "action": self.action,
            "message": str(self),
        }


@dataclass(frozen=True)
class RecallBankMember:
    """Shape and concatenation contract for one independently trained bank."""

    name: str
    concat_dim: int
    shape: tuple[int, ...]
    dtype: str

    def validate(self) -> "RecallBankMember":
        if not self.name or not self.name.endswith("bank"):
            raise ValueError("Recall bank names must end in 'bank'")
        if not self.shape or any(size <= 0 for size in self.shape):
            raise ValueError("Recall bank shapes must contain positive dimensions")
        if self.concat_dim < 0 or self.concat_dim >= len(self.shape):
            raise ValueError("Recall bank concat_dim is out of range")
        if not self.dtype:
            raise ValueError("Recall bank dtype must not be empty")
        return self


def _recall_bank_member_payload(bank: RecallBankMember) -> dict[str, Any]:
    return {
        "name": bank.name,
        "concat_dim": bank.concat_dim,
        "shape": list(bank.shape),
        "dtype": bank.dtype,
    }


@dataclass(frozen=True)
class RecallBankProvenance:
    """Pure-data provenance for one independently saved Recall Bank asset."""

    reader: Mapping[str, Any]
    formula: Mapping[str, Any] | None
    updater: Mapping[str, Any] | None
    bank_layout: Mapping[str, Any]
    schema_fingerprint: str
    schema_version: int = RECALL_BANK_PROVENANCE_VERSION

    def _content_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "reader": dict(self.reader),
            "formula": None if self.formula is None else dict(self.formula),
            "updater": None if self.updater is None else dict(self.updater),
            "bank_layout": dict(self.bank_layout),
            "schema_fingerprint": self.schema_fingerprint,
        }

    @property
    def fingerprint(self) -> str:
        return _sha256_json(self._content_dict())

    def validate(self) -> "RecallBankProvenance":
        if self.schema_version != RECALL_BANK_PROVENANCE_VERSION:
            raise ValueError(
                "unsupported Recall Bank provenance schema version "
                f"{self.schema_version!r}; expected {RECALL_BANK_PROVENANCE_VERSION}"
            )
        _validate_provenance_descriptor("reader", self.reader, require_components=True)
        if self.formula is not None:
            _validate_formula_descriptor(self.formula)
        if self.updater is not None:
            _validate_provenance_descriptor("updater", self.updater, require_components=True)
        _validate_bank_layout_payload(self.bank_layout)
        expected_schema = _sha256_json(
            {"reader": dict(self.reader), "bank_layout": dict(self.bank_layout)}
        )
        if self.schema_fingerprint != expected_schema:
            raise ValueError("Recall Bank schema fingerprint does not match its contents")
        return self

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {**self._content_dict(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecallBankProvenance":
        if not isinstance(value, Mapping):
            raise TypeError("Recall Bank provenance must be a mapping")
        required = {
            "schema_version",
            "reader",
            "formula",
            "updater",
            "bank_layout",
            "schema_fingerprint",
            "fingerprint",
        }
        if set(value) != required:
            raise ValueError(
                "Recall Bank provenance keys must be exactly "
                f"{sorted(required)}"
            )
        result = cls(
            reader=dict(value["reader"]),
            formula=None if value["formula"] is None else dict(value["formula"]),
            updater=None if value["updater"] is None else dict(value["updater"]),
            bank_layout=dict(value["bank_layout"]),
            schema_fingerprint=str(value["schema_fingerprint"]),
            schema_version=int(value["schema_version"]),
        ).validate()
        fingerprint = value["fingerprint"]
        if not isinstance(fingerprint, str) or fingerprint != result.fingerprint:
            raise ValueError("Recall Bank provenance fingerprint does not match its contents")
        return result


@dataclass(frozen=True)
class RecallBankContract:
    """Immutable host, shared-reader, behavior, and bank compatibility data."""

    bank_id: str
    host_state_sha256: str
    shared_state_sha256: str
    shared_config_sha256: str
    banks: tuple[RecallBankMember, ...]
    behavior_fingerprint: str
    provenance: RecallBankProvenance

    def validate(self) -> "RecallBankContract":
        _validate_bank_id(self.bank_id)
        for name, value in (
            ("host_state_sha256", self.host_state_sha256),
            ("shared_state_sha256", self.shared_state_sha256),
            ("shared_config_sha256", self.shared_config_sha256),
        ):
            if not _is_sha256(value):
                raise ValueError(f"Recall bank {name} must be a SHA-256 value")
        if not self.banks:
            raise ValueError("Recall bank contracts require at least one bank")
        for bank in self.banks:
            bank.validate()
        if len({bank.name for bank in self.banks}) != len(self.banks):
            raise ValueError("Recall bank names must be unique")
        if not _is_sha256(self.behavior_fingerprint):
            raise ValueError("Recall bank behavior_fingerprint must be a SHA-256 value")
        self.provenance.validate()
        expected_layout = [_recall_bank_member_payload(bank) for bank in self.banks]
        actual_layout = self.provenance.bank_layout.get("banks")
        if actual_layout != expected_layout:
            raise ValueError(
                "Recall Bank provenance bank_layout does not match the contract banks"
            )
        return self

    @property
    def fingerprint(self) -> str:
        # ``bank_id`` names an asset, not the shared reader contract.  Keeping
        # it out of the compatibility fingerprint is what allows independently
        # named banks to compose without pretending their host/operator
        # contracts differ.
        payload = self.to_dict(include_fingerprint=False)
        payload.pop("bank_id", None)
        return _sha256_json(payload)

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        payload["banks"] = [_recall_bank_member_payload(bank) for bank in self.banks]
        payload["provenance"] = self.provenance.to_dict()
        if include_fingerprint:
            payload["fingerprint"] = self.fingerprint
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecallBankContract":
        payload = dict(value)
        expected = payload.pop("fingerprint", None)
        if not isinstance(expected, str) or not _is_sha256(expected):
            raise ValueError("Recall bank contract fingerprint is required")
        required = {
            "bank_id",
            "host_state_sha256",
            "shared_state_sha256",
            "shared_config_sha256",
            "banks",
            "behavior_fingerprint",
            "provenance",
        }
        if set(payload) != required:
            raise ValueError(
                "Recall bank contract keys must be exactly "
                f"{sorted(required | {'fingerprint'})}"
            )
        contract = cls(
            bank_id=str(payload.get("bank_id", "")),
            host_state_sha256=str(payload.get("host_state_sha256", "")),
            shared_state_sha256=str(payload.get("shared_state_sha256", "")),
            shared_config_sha256=str(payload.get("shared_config_sha256", "")),
            banks=tuple(
                RecallBankMember(
                    name=str(bank["name"]),
                    concat_dim=int(bank["concat_dim"]),
                    shape=tuple(int(size) for size in bank["shape"]),
                    dtype=str(bank["dtype"]),
                )
                for bank in payload.get("banks", ())
            ),
            behavior_fingerprint=str(payload.get("behavior_fingerprint", "")),
            provenance=RecallBankProvenance.from_dict(payload["provenance"]),
        ).validate()
        if expected != contract.fingerprint:
            raise ValueError("Recall bank contract fingerprint does not match its contents")
        return contract


@dataclass(frozen=True)
class RecallBankAsset:
    """Validated Recall bank asset with optional private extension tensors."""

    bank_id: str
    path: Path
    contract: RecallBankContract
    weights_sha256: str
    package_sha256: str
    bank_state_sha256: str
    _state_dict: Mapping[str, Tensor]
    training_metadata: Mapping[str, Any] | None = None
    artifact_version: int = RECALL_BANK_ARTIFACT_VERSION
    private_state_sha256: str | None = None
    private_metadata: Mapping[str, Any] | None = None
    _private_state: Mapping[str, Tensor] | None = None

    @property
    def state_dict(self) -> Mapping[str, Tensor]:
        return MappingProxyType(
            {name: tensor.detach().clone() for name, tensor in self._state_dict.items()}
        )

    @property
    def private_state(self) -> Mapping[str, Tensor] | None:
        if self._private_state is None:
            return None
        return MappingProxyType(
            {name: tensor.detach().clone() for name, tensor in self._private_state.items()}
        )


@dataclass(frozen=True)
class RecallBankLayout:
    """Deterministic logical and physical ownership for an expert assembly."""

    bank_ids: tuple[str, ...]
    artifact_sha256: tuple[str, ...]
    ranges: Mapping[str, Mapping[str, tuple[int, int]]]
    physical_ranges: Mapping[
        str,
        Mapping[str, tuple[tuple[int, int], ...]],
    ] = dataclass_field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "bank_ids": list(self.bank_ids),
            "artifact_sha256": list(self.artifact_sha256),
            "ranges": {
                expert: {name: list(bounds) for name, bounds in banks.items()}
                for expert, banks in self.ranges.items()
            },
            "physical_ranges": {
                expert: {
                    name: [list(bounds) for bounds in factor_ranges]
                    for name, factor_ranges in banks.items()
                }
                for expert, banks in self.physical_ranges.items()
            },
        }


def canonical_tensor_state_sha256(state: Mapping[str, Tensor]) -> str:
    """Hash tensor names, exact dtype/shape, and raw bytes in canonical order."""

    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name]
        if not isinstance(tensor, Tensor):
            raise TypeError(f"state value {name!r} is not a Tensor")
        contiguous = tensor.detach().to("cpu").contiguous()
        for value in (
            name,
            str(contiguous.dtype),
            json.dumps(list(contiguous.shape), separators=(",", ":")),
        ):
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "big"))
            digest.update(encoded)
        raw = contiguous.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def module_value_sha256(module: nn.Module, *, exclude_module: nn.Module | None = None) -> str:
    """Hash parameter and buffer values, including non-persistent buffers."""

    excluded_parameters = (
        set() if exclude_module is None else {id(value) for value in exclude_module.parameters()}
    )
    excluded_buffers = (
        set() if exclude_module is None else {id(value) for value in exclude_module.buffers()}
    )
    state: dict[str, Tensor] = {}
    for name, parameter in module.named_parameters():
        if id(parameter) not in excluded_parameters:
            state[f"parameter:{name}"] = parameter
    for name, buffer in module.named_buffers():
        if id(buffer) not in excluded_buffers:
            state[f"buffer:{name}"] = buffer
    return canonical_tensor_state_sha256(state)


def recall_bank_parameter_names(module: nn.Module) -> tuple[str, ...]:
    """Return the stable names of Recall bank parameters in a module tree."""

    names = tuple(
        name
        for name, _ in module.named_parameters()
        if name == "bank" or name.endswith(".bank") or name.rsplit(".", 1)[-1].endswith("_bank")
    )
    if not names:
        raise ValueError("Recall Bank module contains no bank parameters")
    return names


def freeze_for_recall_bank(
    host: nn.Module, expert: nn.Module
) -> tuple[tuple[str, nn.Parameter], ...]:
    """Freeze the host and shared reader, leaving only expert banks trainable."""

    for parameter in host.parameters():
        parameter.requires_grad_(False)
    bank_names = set(recall_bank_parameter_names(expert))
    selected: list[tuple[str, nn.Parameter]] = []
    for name, parameter in expert.named_parameters():
        trainable = name in bank_names
        parameter.requires_grad_(trainable)
        if trainable:
            selected.append((name, parameter))
    if {name for name, _ in selected} != bank_names:
        raise RuntimeError("failed to isolate Recall bank parameters")
    return tuple(selected)


def create_recall_bank_contract(
    host: nn.Module,
    expert: nn.Module,
    *,
    bank_id: str,
    shared_config: Mapping[str, Any] | None = None,
    formula: str | nn.Module | None = None,
    updater: nn.Module | None = None,
) -> RecallBankContract:
    """Bind one canonical reader, Formula, Updater, and host to a bank asset."""

    bank_names = recall_bank_parameter_names(expert)
    parameters = dict(expert.named_parameters())
    banks = tuple(
        RecallBankMember(name, 0, tuple(parameters[name].shape), str(parameters[name].dtype))
        for name in bank_names
    )
    shared_state = _module_tensor_state(expert, excluded_names=set(bank_names))
    behavior = {
        "module_behavior": _shared_behavior(expert),
        "caller_config": None if shared_config is None else dict(shared_config),
    }
    provenance = _build_recall_bank_provenance(
        expert,
        banks,
        formula=formula,
        updater=updater,
    )
    return RecallBankContract(
        bank_id=bank_id,
        host_state_sha256=module_value_sha256(host, exclude_module=expert),
        shared_state_sha256=canonical_tensor_state_sha256(shared_state),
        shared_config_sha256=_sha256_json(behavior),
        banks=banks,
        behavior_fingerprint=module_behavior_fingerprint(expert),
        provenance=provenance,
    ).validate()


def validate_recall_bank_contract(
    contract: RecallBankContract,
    host: nn.Module,
    expert: nn.Module,
    *,
    shared_config: Mapping[str, Any] | None = None,
    formula: str | nn.Module | None = None,
    updater: nn.Module | None = None,
) -> None:
    """Reject host, reader, Formula, Updater, or bank drift before loading an asset."""

    normalized = contract.validate()
    current = create_recall_bank_contract(
        host,
        expert,
        bank_id=normalized.bank_id,
        shared_config=shared_config,
        formula=formula,
        updater=updater,
    )
    if current != normalized:
        differences = [
            name
            for name in (
                "host_state_sha256",
                "shared_state_sha256",
                "shared_config_sha256",
                "banks",
                "behavior_fingerprint",
                "provenance",
            )
            if getattr(current, name) != getattr(normalized, name)
        ]
        raise ValueError(
            f"Recall Bank contract does not match active host/reader: {', '.join(differences)}"
        )


def save_recall_bank(
    expert: nn.Module,
    path: str | Path,
    *,
    host: nn.Module,
    bank_id: str,
    contract: RecallBankContract,
    shared_config: Mapping[str, Any] | None = None,
    formula: str | nn.Module | None = None,
    updater: nn.Module | None = None,
    training_metadata: Mapping[str, Any] | None = None,
    private_module: nn.Module | None = None,
    private_metadata: Mapping[str, Any] | None = None,
) -> ARTISaveResult:
    """Save trainable banks and an optional opaque private tensor extension."""

    _validate_bank_id(bank_id)
    target = recall_bank_artifact_path(path)
    expected = set(recall_bank_parameter_names(expert))
    trainable = {name for name, parameter in expert.named_parameters() if parameter.requires_grad}
    if trainable != expected:
        raise ValueError(
            "bank-only export requires exactly the Recall bank parameters to be trainable"
        )
    normalized_contract = contract.validate()
    validate_recall_bank_contract(
        normalized_contract,
        host,
        expert,
        shared_config=shared_config,
        formula=formula,
        updater=updater,
    )
    artifact_contract = replace(normalized_contract, bank_id=bank_id).validate()
    bank_state = {
        name: parameter for name, parameter in expert.named_parameters() if name in expected
    }
    artifact_version = RECALL_BANK_ARTIFACT_VERSION
    export_module = expert
    private_payload: dict[str, Any] | None = None
    if private_module is not None or private_metadata is not None:
        if private_module is None or private_metadata is None:
            raise ValueError("private_module and private_metadata must be provided together")
        private_state = _validate_private_module(private_module)
        private_payload = {
            "state_prefix": _PRIVATE_STATE_PREFIX,
            "state_sha256": canonical_tensor_state_sha256(private_state),
            "metadata": dict(private_metadata),
        }
        export_module = copy.deepcopy(expert)
        for parameter in export_module.parameters():
            parameter.requires_grad_(False)
        export_parameters = dict(export_module.named_parameters())
        for name in expected:
            export_parameters[name].requires_grad_(True)
        private_copy = copy.deepcopy(private_module)
        private_copy.requires_grad_(True)
        export_module.add_module(_PRIVATE_MODULE_NAME, private_copy)
    return save(
        export_module,
        target,
        scope="trainable",
        config={
            "artifact_kind": RECALL_BANK_ARTIFACT_KIND,
            "artifact_version": artifact_version,
            "recall_bank": {
                "bank_id": bank_id,
                "contract": artifact_contract.to_dict(),
                "provenance": artifact_contract.provenance.to_dict(),
                "bank_state_sha256": canonical_tensor_state_sha256(bank_state),
                "training_metadata": None if training_metadata is None else dict(training_metadata),
                "private": private_payload,
            },
        },
    )


def _repack_recall_expert_private_module(
    source: str | Path,
    output: str | Path,
    *,
    private_module: nn.Module,
    private_metadata: Mapping[str, Any],
) -> ARTISaveResult:
    """Append a private module to an already validated immutable bank asset."""

    asset = inspect_recall_bank(source)
    if asset.private_state is not None:
        raise ValueError("Recall Bank already contains a private tensor extension")
    target = recall_bank_artifact_path(output)
    if target.resolve() == asset.path.resolve():
        raise ValueError("private extension output must not overwrite its source asset")
    private_state = _validate_private_module(private_module)
    bank_state = asset.state_dict
    combined = {name: bank_state[name] for name in sorted(bank_state)}
    combined.update(
        {f"{_PRIVATE_STATE_PREFIX}{name}": private_state[name] for name in sorted(private_state)}
    )
    package = _TensorStateModule(combined)
    return save(
        package,
        target,
        scope="trainable",
        config={
            "artifact_kind": RECALL_BANK_ARTIFACT_KIND,
            "artifact_version": RECALL_BANK_ARTIFACT_VERSION,
            "recall_bank": {
                "bank_id": asset.bank_id,
                "contract": asset.contract.to_dict(),
                "provenance": asset.contract.provenance.to_dict(),
                "bank_state_sha256": asset.bank_state_sha256,
                "training_metadata": None
                if asset.training_metadata is None
                else dict(asset.training_metadata),
                "private": {
                    "state_prefix": _PRIVATE_STATE_PREFIX,
                    "state_sha256": canonical_tensor_state_sha256(private_state),
                    "metadata": dict(private_metadata),
                },
            },
        },
    )


def inspect_recall_bank(path: str | Path) -> RecallBankAsset:
    """Validate package integrity and return its bank-only metadata/state."""

    target = recall_bank_artifact_path(path)
    loaded = load(target, load_resources=False, load_checkpoint=False)
    config = loaded.manifest.get("architecture", {}).get("config", {})
    if config.get("artifact_kind") != RECALL_BANK_ARTIFACT_KIND:
        raise RecallBankError(
            "wrong_kind",
            "artifact is not a Recall bank",
            path=target,
            field="artifact_kind",
            expected=RECALL_BANK_ARTIFACT_KIND,
            actual=config.get("artifact_kind"),
        )
    artifact_version = config.get("artifact_version")
    if artifact_version != RECALL_BANK_ARTIFACT_VERSION:
        raise RecallBankError(
            "unsupported_version",
            f"unsupported Recall bank artifact version {artifact_version!r}; "
            f"expected {RECALL_BANK_ARTIFACT_VERSION}",
            path=target,
            field="artifact_version",
            expected=RECALL_BANK_ARTIFACT_VERSION,
            actual=artifact_version,
        )
    payload = config.get("recall_bank")
    if not isinstance(payload, Mapping):
        raise ValueError("Recall bank artifact is missing recall_bank metadata")
    bank_id = str(payload.get("bank_id", ""))
    _validate_bank_id(bank_id)
    contract_payload = payload.get("contract")
    if not isinstance(contract_payload, Mapping):
        raise ValueError("Recall bank artifact is missing its contract")
    contract = RecallBankContract.from_dict(contract_payload)
    if contract.bank_id != bank_id:
        raise RecallBankError(
            "bank_id_mismatch",
            "Recall bank artifact bank_id does not match its contract",
            path=target,
            field="bank_id",
            expected=contract.bank_id,
            actual=bank_id,
        )
    provenance_payload = payload.get("provenance")
    if not isinstance(provenance_payload, Mapping):
        raise RecallBankError(
            "missing_provenance",
            "Recall bank artifact is missing its provenance",
            path=target,
            field="provenance",
        )
    provenance = RecallBankProvenance.from_dict(provenance_payload)
    if provenance.to_dict() != contract.provenance.to_dict():
        raise RecallBankError(
            "provenance_mismatch",
            "Recall bank artifact provenance does not match its contract",
            path=target,
            field="provenance",
        )
    expected_names = {bank.name for bank in contract.banks}
    private_payload = payload.get("private")
    private_state: dict[str, Tensor] | None = None
    private_state_sha: str | None = None
    private_metadata: Mapping[str, Any] | None = None
    if private_payload is not None:
        if not isinstance(private_payload, Mapping):
            raise ValueError("Recall bank private metadata must be an object or null")
        if private_payload.get("state_prefix") != _PRIVATE_STATE_PREFIX:
            raise ValueError("Recall bank private tensor prefix is invalid")
        private_metadata = private_payload.get("metadata")
        if not isinstance(private_metadata, Mapping):
            raise ValueError("Recall bank private metadata must be an object")
        private_state = {
            name.removeprefix(_PRIVATE_STATE_PREFIX): tensor
            for name, tensor in sorted(loaded.state_dict.items())
            if name.startswith(_PRIVATE_STATE_PREFIX)
        }
        if not private_state:
            raise ValueError("Recall bank private tensor state is empty")
        private_state_sha = canonical_tensor_state_sha256(private_state)
        if private_payload.get("state_sha256") != private_state_sha:
            raise ValueError("Recall bank private state fingerprint does not match its metadata")
    stored_names = expected_names | {
        f"{_PRIVATE_STATE_PREFIX}{name}"
        for name in (() if private_state is None else private_state)
    }
    if set(loaded.state_dict) != stored_names:
        raise ValueError("Recall bank artifact tensor names do not match its contract")
    bank_state = {name: loaded.state_dict[name] for name in sorted(expected_names)}
    _validate_loaded_bank_state(contract, bank_state)
    state_sha = canonical_tensor_state_sha256(bank_state)
    if payload.get("bank_state_sha256") != state_sha:
        raise ValueError("Recall bank state fingerprint does not match its metadata")
    weights_sha = str(loaded.manifest.get("weights", {}).get("sha256", ""))
    if not _is_sha256(weights_sha):
        raise ValueError("Recall bank artifact is missing its weights SHA-256")
    metadata = payload.get("training_metadata")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise ValueError("Recall bank training_metadata must be an object")
    return RecallBankAsset(
        bank_id=bank_id,
        path=target,
        contract=contract,
        weights_sha256=weights_sha,
        package_sha256=_artifact_package_sha256(target),
        bank_state_sha256=state_sha,
        _state_dict=MappingProxyType(
            {
                name: tensor.detach().to("cpu").contiguous().clone()
                for name, tensor in bank_state.items()
            }
        ),
        training_metadata=metadata,
        artifact_version=int(artifact_version),
        private_state_sha256=private_state_sha,
        private_metadata=None
        if private_metadata is None
        else MappingProxyType(dict(private_metadata)),
        _private_state=None
        if private_state is None
        else MappingProxyType(
            {
                name: tensor.detach().to("cpu").contiguous().clone()
                for name, tensor in private_state.items()
            }
        ),
    )


def load_recall_bank(
    path: str | Path,
    expert: nn.Module,
    *,
    contract: RecallBankContract,
    formula: str | nn.Module | None = None,
    updater: nn.Module | None = None,
) -> RecallBankAsset:
    """Load only bank tensors into a compatible canonical shared reader."""

    asset = inspect_recall_bank(path)
    if asset.contract.fingerprint != contract.fingerprint:
        raise ValueError("Recall bank artifact belongs to a different expert contract")
    _validate_bank_specs(contract, expert)
    if contract.behavior_fingerprint != module_behavior_fingerprint(expert):
        raise ValueError("Recall bank behavior fingerprint does not match the active reader")
    _validate_active_recall_bank_provenance(
        expert,
        contract.provenance,
        formula=formula,
        updater=updater,
    )
    result = expert.load_state_dict(dict(asset.state_dict), strict=False)
    expected_missing = set(expert.state_dict()) - {bank.name for bank in contract.banks}
    if set(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise RuntimeError("Recall bank partial load touched an unexpected state surface")
    return asset


def migrate_recall_bank(
    source: str | Path,
    output: str | Path,
    *,
    target_expert: nn.Module,
    target_host: nn.Module,
    target_contract: RecallBankContract,
    state_transform: Callable[[Mapping[str, Tensor]], Mapping[str, Tensor]] | None = None,
    shared_config: Mapping[str, Any] | None = None,
    formula: str | nn.Module | None = None,
    updater: nn.Module | None = None,
    training_metadata: Mapping[str, Any] | None = None,
) -> ARTISaveResult:
    """Explicitly transform a validated bank into a different contract.

    No artifact version, shape, Formula, Updater, or reader migration is
    inferred here.  The caller must provide the target contract and, when
    needed, a named tensor transform.  Private extensions are intentionally
    not copied.
    """

    source_asset = inspect_recall_bank(source)
    normalized_target = target_contract.validate()
    if source_asset.contract.fingerprint == normalized_target.fingerprint:
        raise ValueError(
            "source and target Recall Bank contracts are identical; "
            "migration must be explicit and non-empty"
        )
    source_state = {
        name: tensor.detach().clone()
        for name, tensor in source_asset.state_dict.items()
    }
    transformed = (
        source_state
        if state_transform is None
        else state_transform(MappingProxyType(source_state))
    )
    if not isinstance(transformed, Mapping):
        raise TypeError("state_transform must return a tensor mapping")
    expected_names = {bank.name for bank in normalized_target.banks}
    if set(transformed) != expected_names:
        raise ValueError(
            "migrated Recall Bank state names must exactly match the target contract"
        )
    transformed_state = {
        name: value.detach().clone() if isinstance(value, Tensor) else value
        for name, value in transformed.items()
    }
    if any(not isinstance(value, Tensor) for value in transformed_state.values()):
        raise TypeError("migrated Recall Bank state values must be tensors")
    _validate_loaded_bank_state(normalized_target, transformed_state)
    freeze_for_recall_bank(target_host, target_expert)
    target_parameters = dict(target_expert.named_parameters())
    with torch.no_grad():
        for name, value in transformed_state.items():
            target_parameters[name].copy_(
                value.to(
                    device=target_parameters[name].device,
                    dtype=target_parameters[name].dtype,
                )
            )
    metadata = dict(training_metadata or {})
    if "migration" in metadata:
        raise ValueError("training_metadata cannot override migration provenance")
    metadata["migration"] = {
        "explicit": True,
        "source_package_sha256": source_asset.package_sha256,
        "source_contract_fingerprint": source_asset.contract.fingerprint,
        "target_contract_fingerprint": normalized_target.fingerprint,
    }
    return save_recall_bank(
        target_expert,
        output,
        host=target_host,
        bank_id=normalized_target.bank_id,
        contract=normalized_target,
        shared_config=shared_config,
        formula=formula,
        updater=updater,
        training_metadata=metadata,
    )


class RecallBankAssembly:
    """Rebuild native concatenated banks from immutable named assets."""

    def __init__(
        self,
        template: nn.Module,
        contract: RecallBankContract,
        *,
        formula: str | nn.Module | None = None,
        updater: nn.Module | None = None,
    ) -> None:
        self.template = copy.deepcopy(template)
        self.contract = contract.validate()
        _validate_bank_specs(contract, self.template)
        _validate_active_recall_bank_provenance(
            self.template,
            self.contract.provenance,
            formula=formula,
            updater=updater,
        )
        self.formula = formula
        self.updater = updater
        self._assets: dict[str, RecallBankAsset] = {}

    @property
    def bank_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._assets))

    def fork(self) -> "RecallBankAssembly":
        candidate = RecallBankAssembly(
            self.template,
            self.contract,
            formula=self.formula,
            updater=self.updater,
        )
        candidate._assets = dict(self._assets)
        return candidate

    def add(self, path: str | Path) -> RecallBankAsset:
        asset = inspect_recall_bank(path)
        if asset.contract.fingerprint != self.contract.fingerprint:
            raise ValueError("Recall Bank uses an incompatible shared contract")
        if asset.contract.behavior_fingerprint != module_behavior_fingerprint(self.template):
            raise ValueError("Recall Bank behavior fingerprint does not match the assembly template")
        _validate_active_recall_bank_provenance(
            self.template,
            asset.contract.provenance,
            formula=self.formula,
            updater=self.updater,
        )
        if asset.bank_id in self._assets:
            raise ValueError(f"Recall bank {asset.bank_id!r} is already present")
        self._assets[asset.bank_id] = asset
        return asset

    def replace(self, paths: Iterable[str | Path]) -> tuple[RecallBankAsset, ...]:
        """Atomically replace the asset set after validating every candidate."""

        assets = tuple(inspect_recall_bank(path) for path in paths)
        if any(asset.contract.fingerprint != self.contract.fingerprint for asset in assets):
            raise ValueError("Recall Bank uses an incompatible shared contract")
        for asset in assets:
            _validate_active_recall_bank_provenance(
                self.template,
                asset.contract.provenance,
                formula=self.formula,
                updater=self.updater,
            )
        if len({asset.bank_id for asset in assets}) != len(assets):
            raise ValueError("Recall bank IDs must be unique")
        self._assets = {asset.bank_id: asset for asset in assets}
        return assets

    def remove(self, bank_id: str) -> RecallBankAsset:
        if bank_id not in self._assets:
            raise KeyError(f"unknown Recall bank {bank_id!r}")
        return self._assets.pop(bank_id)

    def clear(self) -> tuple[RecallBankAsset, ...]:
        removed = tuple(self._assets[name] for name in self.bank_ids)
        self._assets.clear()
        return removed

    def materialize(self) -> tuple[nn.Module, RecallBankLayout]:
        """Return a fresh module; never mutate or slice a previous assembly."""

        result = copy.deepcopy(self.template)
        ordered = [self._assets[name] for name in self.bank_ids]
        verified_states = {asset.bank_id: _verified_asset_state(asset) for asset in ordered}
        ranges: dict[str, dict[str, tuple[int, int]]] = {asset.bank_id: {} for asset in ordered}
        physical_ranges: dict[
            str,
            dict[str, tuple[tuple[int, int], ...]],
        ] = {asset.bank_id: {} for asset in ordered}
        for spec in self.contract.banks:
            if not ordered:
                continue
            values: list[Tensor] = []
            offset = 0
            for asset in ordered:
                value = verified_states[asset.bank_id][spec.name]
                width = int(value.shape[spec.concat_dim])
                ranges[asset.bank_id][spec.name] = (offset, offset + width)
                offset += width
                values.append(value)
            parent, leaf = _resolve_parent(result, spec.name)
            current = getattr(parent, leaf)
            if not isinstance(current, nn.Parameter):
                raise ValueError(f"Recall bank {spec.name!r} is not a Parameter")
            joined, bank_physical_ranges = _concatenate_bank_values(
                parent,
                leaf,
                values,
                concat_dim=spec.concat_dim,
            )
            joined = joined.to(device=current.device, dtype=current.dtype)
            for asset, factor_ranges in zip(
                ordered,
                bank_physical_ranges,
                strict=True,
            ):
                physical_ranges[asset.bank_id][spec.name] = factor_ranges
            setattr(parent, leaf, nn.Parameter(joined, requires_grad=current.requires_grad))
            if spec.concat_dim == 0 and leaf == "bank" and hasattr(parent, "slots"):
                parent.slots = int(joined.shape[0])
                factor_count = int(getattr(parent, "composition_factor", 1))
                if factor_count > 1:
                    if parent.slots % factor_count:
                        raise ValueError(
                            "materialized Recall slots must be divisible by composition_factor"
                        )
                    slots_per_factor = parent.slots // factor_count
                    parent.factor_slices = tuple(
                        (
                            factor_index * slots_per_factor,
                            (factor_index + 1) * slots_per_factor,
                        )
                        for factor_index in range(factor_count)
                    )
        return result, RecallBankLayout(
            bank_ids=tuple(asset.bank_id for asset in ordered),
            artifact_sha256=tuple(asset.weights_sha256 for asset in ordered),
            ranges=ranges,
            physical_ranges=physical_ranges,
        )


def _concatenate_bank_values(
    parent: nn.Module,
    leaf: str,
    values: list[Tensor],
    *,
    concat_dim: int,
) -> tuple[Tensor, tuple[tuple[tuple[int, int], ...], ...]]:
    """Concatenate bank tensors while preserving contiguous factor regions."""

    factor_count = int(getattr(parent, "composition_factor", 1))
    factorized = concat_dim == 0 and factor_count > 1 and leaf in {"bank", "key_bank", "group_bank"}
    if not factorized:
        offsets: list[tuple[tuple[int, int], ...]] = []
        offset = 0
        for value in values:
            width = int(value.shape[concat_dim])
            offsets.append(((offset, offset + width),))
            offset += width
        return torch.cat(values, dim=concat_dim), tuple(offsets)

    widths: list[int] = []
    for value in values:
        rows = int(value.shape[0])
        if rows % factor_count:
            raise ValueError(f"Recall bank {leaf!r} rows must be divisible by composition_factor")
        widths.append(rows // factor_count)

    rows_per_factor = sum(widths)
    factors = [
        torch.cat(
            [
                value.narrow(0, factor_index * width, width)
                for value, width in zip(values, widths, strict=True)
            ],
            dim=0,
        )
        for factor_index in range(factor_count)
    ]
    physical: list[tuple[tuple[int, int], ...]] = []
    expert_offset = 0
    for width in widths:
        physical.append(
            tuple(
                (
                    factor_index * rows_per_factor + expert_offset,
                    factor_index * rows_per_factor + expert_offset + width,
                )
                for factor_index in range(factor_count)
            )
        )
        expert_offset += width
    return torch.cat(factors, dim=0), tuple(physical)


def _module_tensor_state(module: nn.Module, *, excluded_names: set[str]) -> dict[str, Tensor]:
    state: dict[str, Tensor] = {}
    for name, parameter in module.named_parameters():
        if name not in excluded_names:
            state[f"parameter:{name}"] = parameter
    for name, buffer in module.named_buffers():
        state[f"buffer:{name}"] = buffer
    return state


class _TensorStateModule(nn.Module):
    def __init__(self, state: Mapping[str, Tensor]) -> None:
        super().__init__()
        for name, value in state.items():
            parts = name.split(".")
            if not parts or any(not part for part in parts):
                raise ValueError(f"invalid tensor path {name!r}")
            parent = self
            for part in parts[:-1]:
                child = parent._modules.get(part)
                if child is None:
                    child = nn.Module()
                    parent.add_module(part, child)
                parent = child
            parent.register_parameter(parts[-1], nn.Parameter(value.detach().clone()))


def _validate_private_module(module: nn.Module) -> dict[str, Tensor]:
    private_state = dict(module.state_dict())
    private_parameters = dict(module.named_parameters())
    if not private_state:
        raise ValueError("private_module must contain at least one tensor")
    if set(private_state) != set(private_parameters):
        raise ValueError("private_module state must contain parameters only")
    if any(name.startswith(_PRIVATE_STATE_PREFIX) for name in private_state):
        raise ValueError("private_module tensor names use a reserved prefix")
    return private_state


def _shared_behavior(module: nn.Module) -> dict[str, Any]:
    modules: list[dict[str, Any]] = []
    fields = (
        "dim",
        "rank",
        "use_half",
        "recognition_mode",
        "combine",
        "threshold",
        "base",
        "scale",
        "stochastic",
    )
    for name, child in module.named_modules():
        record: dict[str, Any] = {
            "name": name,
            "class": f"{child.__class__.__module__}.{child.__class__.__qualname__}",
        }
        for field in fields:
            value = getattr(child, field, None)
            if isinstance(value, (str, int, float, bool)):
                record[field] = value
        modules.append(record)
    return {"modules": modules}


def _validate_bank_specs(contract: RecallBankContract, expert: nn.Module) -> None:
    parameters = dict(expert.named_parameters())
    expected = {bank.name for bank in contract.banks}
    if set(recall_bank_parameter_names(expert)) != expected:
        raise ValueError("active Recall bank names do not match the expert contract")
    for bank in contract.banks:
        value = parameters[bank.name]
        if tuple(value.shape) != bank.shape or str(value.dtype) != bank.dtype:
            raise ValueError(
                f"active Recall bank {bank.name!r} shape or dtype does not match the contract"
            )


def _recall_bank_layout_payload(
    expert: nn.Module,
    banks: tuple[RecallBankMember, ...],
) -> dict[str, Any]:
    def positive_int_or_none(value: Any) -> int | None:
        return value if type(value) is int and value > 0 else None

    dimensions = {
        "slots": positive_int_or_none(getattr(expert, "slots", None)),
        "hidden_dim": positive_int_or_none(
            getattr(expert, "hidden_dim", getattr(expert, "dim", None))
        ),
        "composition_factor": positive_int_or_none(
            getattr(expert, "composition_factor", 1)
        )
        or 1,
        "factor_names": [
            str(value)
            for value in getattr(expert, "factor_names", ())
            if isinstance(value, str)
        ],
        "route_names": [
            str(value)
            for value in getattr(expert, "route_names", ())
            if isinstance(value, str)
        ],
    }
    return {
        "banks": [_recall_bank_member_payload(bank) for bank in banks],
        "dimensions": dimensions,
    }


def _build_recall_bank_provenance(
    expert: nn.Module,
    banks: tuple[RecallBankMember, ...],
    *,
    formula: str | nn.Module | None = None,
    updater: nn.Module | None = None,
) -> RecallBankProvenance:
    if updater is not None and not isinstance(updater, nn.Module):
        raise TypeError("updater must be a torch.nn.Module or None")
    reader = _module_provenance_descriptor(expert)
    bank_layout = _recall_bank_layout_payload(expert, banks)
    provenance = RecallBankProvenance(
        reader=reader,
        formula=_formula_provenance_descriptor(expert, formula),
        updater=None if updater is None else _module_provenance_descriptor(updater),
        bank_layout=bank_layout,
        schema_fingerprint=_sha256_json(
            {"reader": reader, "bank_layout": bank_layout}
        ),
    )
    return provenance.validate()


def _validate_active_recall_bank_provenance(
    expert: nn.Module,
    expected: RecallBankProvenance,
    *,
    formula: str | nn.Module | None = None,
    updater: nn.Module | None = None,
) -> None:
    expected.validate()
    expected_members = tuple(
        RecallBankMember(
            name=str(item["name"]),
            concat_dim=int(item["concat_dim"]),
            shape=tuple(int(size) for size in item["shape"]),
            dtype=str(item["dtype"]),
        )
        for item in expected.bank_layout["banks"]
    )
    parameters = dict(expert.named_parameters())
    if any(member.name not in parameters for member in expected_members):
        raise ValueError("active Recall bank names do not match provenance")
    current_banks = tuple(
        RecallBankMember(
            member.name,
            member.concat_dim,
            tuple(parameters[member.name].shape),
            str(parameters[member.name].dtype),
        )
        for member in expected_members
    )
    actual = _build_recall_bank_provenance(
        expert,
        current_banks,
        formula=formula,
        updater=updater,
    )
    if actual.to_dict() != expected.to_dict():
        raise ValueError(
            "Recall Bank provenance does not match the active reader, Formula, "
            "Updater, or layout"
        )


def _validate_loaded_bank_state(
    contract: RecallBankContract, state: Mapping[str, Tensor]
) -> None:
    for bank in contract.banks:
        value = state[bank.name]
        if tuple(value.shape) != bank.shape or str(value.dtype) != bank.dtype:
            raise ValueError(
                f"stored Recall bank {bank.name!r} shape or dtype does not match the contract"
            )


def _verified_asset_state(asset: RecallBankAsset) -> Mapping[str, Tensor]:
    if canonical_tensor_state_sha256(asset._state_dict) != asset.bank_state_sha256:
        raise RuntimeError(
            f"in-memory Recall bank {asset.bank_id!r} no longer matches its recorded bank hash"
        )
    return asset._state_dict


def _resolve_parent(module: nn.Module, path: str) -> tuple[nn.Module, str]:
    parts = path.split(".")
    parent = module
    for part in parts[:-1]:
        parent = (
            parent[int(part)]
            if part.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList))
            else getattr(parent, part)
        )
    return parent, parts[-1]


def _artifact_package_sha256(target: Path) -> str:
    stem = target.with_suffix("")
    members = (target, stem.with_suffix(".json"), stem.with_suffix(".lock.json"))
    digest = hashlib.sha256()
    for member in members:
        raw = member.read_bytes()
        encoded = member.name.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def _validate_bank_id(value: str) -> None:
    if not _EXPERT_ID.fullmatch(value):
        raise ValueError(
            "bank_id must be 1-128 letters, digits, '.', '_', or '-' and start with a letter or digit"
        )


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "RECALL_BANK_ARTIFACT_KIND",
    "RECALL_BANK_ARTIFACT_VERSION",
    "RECALL_BANK_PROVENANCE_VERSION",
    "RecallCapacityDecision",
    "RecallCapacityPlan",
    "RecallBankError",
    "RecallBankAsset",
    "RecallBankAssembly",
    "RecallBankContract",
    "RecallBankLayout",
    "RecallBankMember",
    "RecallBankProvenance",
    "canonical_tensor_state_sha256",
    "create_recall_bank_contract",
    "save_recall_bank",
    "freeze_for_recall_bank",
    "inspect_recall_bank",
    "load_recall_bank",
    "migrate_recall_bank",
    "module_behavior_fingerprint",
    "module_structure_fingerprint",
    "module_value_sha256",
    "recall_bank_artifact_path",
    "recall_bank_parameter_names",
    "validate_recall_bank_contract",
]
