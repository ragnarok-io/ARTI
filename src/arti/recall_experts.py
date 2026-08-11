"""Immutable bank-only Recall expert assets and deterministic assemblies."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from dataclasses import asdict, dataclass, field as dataclass_field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn
from torch import Tensor

from .recall_artifacts import recall_artifact_path
from .serialization import ARTISaveResult, load, save


RECALL_BANK_ARTIFACT_KIND = "arti.recall-bank"
RECALL_BANK_ARTIFACT_VERSION = 2
_SUPPORTED_RECALL_BANK_ARTIFACT_VERSIONS = frozenset({1, RECALL_BANK_ARTIFACT_VERSION})
_PRIVATE_MODULE_NAME = "_arti_private"
_PRIVATE_STATE_PREFIX = f"{_PRIVATE_MODULE_NAME}."
_EXPERT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


@dataclass(frozen=True)
class RecallBankSpec:
    """Shape and concatenation contract for one independently trained bank."""

    name: str
    concat_dim: int
    shape: tuple[int, ...]
    dtype: str

    def validate(self) -> "RecallBankSpec":
        if not self.name or not self.name.endswith("bank"):
            raise ValueError("Recall bank names must end in 'bank'")
        if not self.shape or any(size <= 0 for size in self.shape):
            raise ValueError("Recall bank shapes must contain positive dimensions")
        if self.concat_dim < 0 or self.concat_dim >= len(self.shape):
            raise ValueError("Recall bank concat_dim is out of range")
        if not self.dtype:
            raise ValueError("Recall bank dtype must not be empty")
        return self


@dataclass(frozen=True)
class RecallExpertContract:
    """Immutable host, shared-reader, behavior, and bank compatibility data."""

    preset_id: str
    host_state_sha256: str
    shared_state_sha256: str
    shared_config_sha256: str
    banks: tuple[RecallBankSpec, ...]
    model_id: str | None = None
    revision: str | None = None
    format_version: int = 1

    def validate(self) -> "RecallExpertContract":
        if not self.preset_id.strip():
            raise ValueError("Recall expert preset_id must not be empty")
        for name, value in (
            ("host_state_sha256", self.host_state_sha256),
            ("shared_state_sha256", self.shared_state_sha256),
            ("shared_config_sha256", self.shared_config_sha256),
        ):
            if not _is_sha256(value):
                raise ValueError(f"Recall expert {name} must be a SHA-256 value")
        if not self.banks:
            raise ValueError("Recall expert contracts require at least one bank")
        for bank in self.banks:
            bank.validate()
        if len({bank.name for bank in self.banks}) != len(self.banks):
            raise ValueError("Recall expert bank names must be unique")
        if self.format_version != 1:
            raise ValueError("unsupported Recall expert contract version")
        return self

    @property
    def fingerprint(self) -> str:
        return _sha256_json(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        payload["banks"] = [asdict(bank) for bank in self.banks]
        if include_fingerprint:
            payload["fingerprint"] = self.fingerprint
        return payload

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecallExpertContract":
        payload = dict(value)
        expected = payload.pop("fingerprint", None)
        contract = cls(
            preset_id=str(payload.get("preset_id", "")),
            host_state_sha256=str(payload.get("host_state_sha256", "")),
            shared_state_sha256=str(payload.get("shared_state_sha256", "")),
            shared_config_sha256=str(payload.get("shared_config_sha256", "")),
            banks=tuple(
                RecallBankSpec(
                    name=str(bank["name"]),
                    concat_dim=int(bank["concat_dim"]),
                    shape=tuple(int(size) for size in bank["shape"]),
                    dtype=str(bank["dtype"]),
                )
                for bank in payload.get("banks", ())
            ),
            model_id=None if payload.get("model_id") is None else str(payload["model_id"]),
            revision=None if payload.get("revision") is None else str(payload["revision"]),
            format_version=int(payload.get("format_version", 0)),
        ).validate()
        if expected is not None and expected != contract.fingerprint:
            raise ValueError("Recall expert contract fingerprint does not match its contents")
        return contract


@dataclass(frozen=True)
class RecallExpertAsset:
    """Validated Recall bank asset with optional private extension tensors."""

    expert_id: str
    path: Path
    contract: RecallExpertContract
    weights_sha256: str
    package_sha256: str
    bank_state_sha256: str
    _state_dict: Mapping[str, Tensor]
    training_metadata: Mapping[str, Any] | None = None
    artifact_version: int = 1
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
class RecallExpertLayout:
    """Deterministic logical and physical ownership for an expert assembly."""

    expert_ids: tuple[str, ...]
    artifact_sha256: tuple[str, ...]
    ranges: Mapping[str, Mapping[str, tuple[int, int]]]
    physical_ranges: Mapping[
        str,
        Mapping[str, tuple[tuple[int, int], ...]],
    ] = dataclass_field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "expert_ids": list(self.expert_ids),
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
        raise ValueError("Recall expert module contains no bank parameters")
    return names


def freeze_for_recall_expert(
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


def create_recall_expert_contract(
    host: nn.Module,
    expert: nn.Module,
    *,
    preset_id: str,
    model_id: str | None = None,
    revision: str | None = None,
    shared_config: Mapping[str, Any] | None = None,
) -> RecallExpertContract:
    """Bind one canonical shared reader and host to independently trained banks."""

    bank_names = recall_bank_parameter_names(expert)
    parameters = dict(expert.named_parameters())
    banks = tuple(
        RecallBankSpec(name, 0, tuple(parameters[name].shape), str(parameters[name].dtype))
        for name in bank_names
    )
    shared_state = _module_tensor_state(expert, excluded_names=set(bank_names))
    behavior = {
        "module_behavior": _shared_behavior(expert),
        "caller_config": None if shared_config is None else dict(shared_config),
    }
    return RecallExpertContract(
        preset_id=preset_id,
        model_id=model_id,
        revision=revision,
        host_state_sha256=module_value_sha256(host, exclude_module=expert),
        shared_state_sha256=canonical_tensor_state_sha256(shared_state),
        shared_config_sha256=_sha256_json(behavior),
        banks=banks,
    ).validate()


def validate_recall_expert_contract(
    contract: RecallExpertContract,
    host: nn.Module,
    expert: nn.Module,
    *,
    shared_config: Mapping[str, Any] | None = None,
) -> None:
    """Reject host, reader, behavior, or bank drift before loading an asset."""

    normalized = contract.validate()
    current = create_recall_expert_contract(
        host,
        expert,
        preset_id=normalized.preset_id,
        model_id=normalized.model_id,
        revision=normalized.revision,
        shared_config=shared_config,
    )
    if current != normalized:
        differences = [
            name
            for name in (
                "host_state_sha256",
                "shared_state_sha256",
                "shared_config_sha256",
                "banks",
            )
            if getattr(current, name) != getattr(normalized, name)
        ]
        raise ValueError(
            f"Recall expert contract does not match active host/reader: {', '.join(differences)}"
        )


def export_recall_expert_bank(
    expert: nn.Module,
    path: str | Path,
    *,
    host: nn.Module,
    expert_id: str,
    contract: RecallExpertContract,
    shared_config: Mapping[str, Any] | None = None,
    training_metadata: Mapping[str, Any] | None = None,
    private_module: nn.Module | None = None,
    private_metadata: Mapping[str, Any] | None = None,
) -> ARTISaveResult:
    """Save trainable banks and an optional opaque private tensor extension."""

    _validate_expert_id(expert_id)
    target = recall_artifact_path(path)
    expected = set(recall_bank_parameter_names(expert))
    trainable = {name for name, parameter in expert.named_parameters() if parameter.requires_grad}
    if trainable != expected:
        raise ValueError(
            "bank-only export requires exactly the Recall bank parameters to be trainable"
        )
    validate_recall_expert_contract(contract, host, expert, shared_config=shared_config)
    bank_state = {
        name: parameter for name, parameter in expert.named_parameters() if name in expected
    }
    artifact_version = 1
    export_module = expert
    private_payload: dict[str, Any] | None = None
    if private_module is not None or private_metadata is not None:
        if private_module is None or private_metadata is None:
            raise ValueError("private_module and private_metadata must be provided together")
        private_state = _validate_private_module(private_module)
        artifact_version = RECALL_BANK_ARTIFACT_VERSION
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
                "expert_id": expert_id,
                "contract": contract.to_dict(),
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

    asset = inspect_recall_expert_bank(source)
    if asset.private_state is not None:
        raise ValueError("Recall expert already contains a private tensor extension")
    target = recall_artifact_path(output)
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
                "expert_id": asset.expert_id,
                "contract": asset.contract.to_dict(),
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


def inspect_recall_expert_bank(path: str | Path) -> RecallExpertAsset:
    """Validate package integrity and return its bank-only metadata/state."""

    target = recall_artifact_path(path)
    loaded = load(target, load_resources=False, load_checkpoint=False)
    config = loaded.manifest.get("architecture", {}).get("config", {})
    if config.get("artifact_kind") != RECALL_BANK_ARTIFACT_KIND:
        raise ValueError("artifact is not a Recall bank expert")
    artifact_version = config.get("artifact_version")
    if artifact_version not in _SUPPORTED_RECALL_BANK_ARTIFACT_VERSIONS:
        raise ValueError("unsupported Recall bank artifact version")
    payload = config.get("recall_bank")
    if not isinstance(payload, Mapping):
        raise ValueError("Recall bank artifact is missing recall_bank metadata")
    expert_id = str(payload.get("expert_id", ""))
    _validate_expert_id(expert_id)
    contract_payload = payload.get("contract")
    if not isinstance(contract_payload, Mapping):
        raise ValueError("Recall bank artifact is missing its contract")
    contract = RecallExpertContract.from_dict(contract_payload)
    expected_names = {bank.name for bank in contract.banks}
    private_payload = payload.get("private")
    private_state: dict[str, Tensor] | None = None
    private_state_sha: str | None = None
    private_metadata: Mapping[str, Any] | None = None
    if artifact_version == 1:
        if private_payload is not None:
            raise ValueError("Recall bank artifact v1 cannot contain private metadata")
    else:
        if not isinstance(private_payload, Mapping):
            raise ValueError("Recall bank artifact v2 is missing private metadata")
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
    return RecallExpertAsset(
        expert_id=expert_id,
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


def load_recall_expert_bank(
    path: str | Path,
    expert: nn.Module,
    *,
    contract: RecallExpertContract,
) -> RecallExpertAsset:
    """Load only bank tensors into a compatible canonical shared reader."""

    asset = inspect_recall_expert_bank(path)
    if asset.contract.fingerprint != contract.fingerprint:
        raise ValueError("Recall bank artifact belongs to a different expert contract")
    _validate_bank_specs(contract, expert)
    result = expert.load_state_dict(dict(asset.state_dict), strict=False)
    expected_missing = set(expert.state_dict()) - {bank.name for bank in contract.banks}
    if set(result.missing_keys) != expected_missing or result.unexpected_keys:
        raise RuntimeError("Recall bank partial load touched an unexpected state surface")
    return asset


class RecallExpertAssembly:
    """Rebuild native concatenated banks from immutable named assets."""

    def __init__(self, template: nn.Module, contract: RecallExpertContract) -> None:
        self.template = copy.deepcopy(template)
        self.contract = contract.validate()
        _validate_bank_specs(contract, self.template)
        self._assets: dict[str, RecallExpertAsset] = {}

    @property
    def expert_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._assets))

    def fork(self) -> "RecallExpertAssembly":
        candidate = RecallExpertAssembly(self.template, self.contract)
        candidate._assets = dict(self._assets)
        return candidate

    def add(self, path: str | Path) -> RecallExpertAsset:
        asset = inspect_recall_expert_bank(path)
        if asset.contract.fingerprint != self.contract.fingerprint:
            raise ValueError("Recall expert uses an incompatible shared contract")
        if asset.expert_id in self._assets:
            raise ValueError(f"Recall expert {asset.expert_id!r} is already present")
        self._assets[asset.expert_id] = asset
        return asset

    def replace(self, paths: Iterable[str | Path]) -> tuple[RecallExpertAsset, ...]:
        """Atomically replace the asset set after validating every candidate."""

        assets = tuple(inspect_recall_expert_bank(path) for path in paths)
        if any(asset.contract.fingerprint != self.contract.fingerprint for asset in assets):
            raise ValueError("Recall expert uses an incompatible shared contract")
        if len({asset.expert_id for asset in assets}) != len(assets):
            raise ValueError("Recall expert IDs must be unique")
        self._assets = {asset.expert_id: asset for asset in assets}
        return assets

    def remove(self, expert_id: str) -> RecallExpertAsset:
        if expert_id not in self._assets:
            raise KeyError(f"unknown Recall expert {expert_id!r}")
        return self._assets.pop(expert_id)

    def clear(self) -> tuple[RecallExpertAsset, ...]:
        removed = tuple(self._assets[name] for name in self.expert_ids)
        self._assets.clear()
        return removed

    def materialize(self) -> tuple[nn.Module, RecallExpertLayout]:
        """Return a fresh module; never mutate or slice a previous assembly."""

        result = copy.deepcopy(self.template)
        ordered = [self._assets[name] for name in self.expert_ids]
        verified_states = {asset.expert_id: _verified_asset_state(asset) for asset in ordered}
        ranges: dict[str, dict[str, tuple[int, int]]] = {asset.expert_id: {} for asset in ordered}
        physical_ranges: dict[
            str,
            dict[str, tuple[tuple[int, int], ...]],
        ] = {asset.expert_id: {} for asset in ordered}
        for spec in self.contract.banks:
            if not ordered:
                continue
            values: list[Tensor] = []
            offset = 0
            for asset in ordered:
                value = verified_states[asset.expert_id][spec.name]
                width = int(value.shape[spec.concat_dim])
                ranges[asset.expert_id][spec.name] = (offset, offset + width)
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
                physical_ranges[asset.expert_id][spec.name] = factor_ranges
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
        return result, RecallExpertLayout(
            expert_ids=tuple(asset.expert_id for asset in ordered),
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


def _validate_bank_specs(contract: RecallExpertContract, expert: nn.Module) -> None:
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


def _validate_loaded_bank_state(
    contract: RecallExpertContract, state: Mapping[str, Tensor]
) -> None:
    for bank in contract.banks:
        value = state[bank.name]
        if tuple(value.shape) != bank.shape or str(value.dtype) != bank.dtype:
            raise ValueError(
                f"stored Recall bank {bank.name!r} shape or dtype does not match the contract"
            )


def _verified_asset_state(asset: RecallExpertAsset) -> Mapping[str, Tensor]:
    if canonical_tensor_state_sha256(asset._state_dict) != asset.bank_state_sha256:
        raise RuntimeError(
            f"in-memory Recall expert {asset.expert_id!r} no longer matches its recorded bank hash"
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


def _validate_expert_id(value: str) -> None:
    if not _EXPERT_ID.fullmatch(value):
        raise ValueError(
            "expert_id must be 1-128 letters, digits, '.', '_', or '-' and start with a letter or digit"
        )


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


__all__ = [
    "RECALL_BANK_ARTIFACT_KIND",
    "RECALL_BANK_ARTIFACT_VERSION",
    "RecallBankSpec",
    "RecallExpertAsset",
    "RecallExpertAssembly",
    "RecallExpertContract",
    "RecallExpertLayout",
    "canonical_tensor_state_sha256",
    "create_recall_expert_contract",
    "export_recall_expert_bank",
    "freeze_for_recall_expert",
    "inspect_recall_expert_bank",
    "load_recall_expert_bank",
    "module_value_sha256",
    "recall_bank_parameter_names",
    "validate_recall_expert_contract",
]
