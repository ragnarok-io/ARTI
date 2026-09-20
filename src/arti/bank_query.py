"""Bank-owned Query contracts, sealing, and portable assets."""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import ClassVar, Mapping

import torch
from torch import Tensor, nn

from .component_registry import (
    ComponentRef,
    canonical_contract_reference,
    component_ref,
)
from .recall_experts import canonical_tensor_state_sha256
from .serialization import ARTISaveResult, load, save
from .tensor_schema import GradientContract, TensorSchema


BANK_QUERY_ARTIFACT_KIND = "arti.bank-query"
BANK_QUERY_ARTIFACT_VERSION = 1
QUERY_EXECUTION_SIGNATURE_VERSION = 1

_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class BankQueryError(ValueError):
    """Raised when a Bank-owned Query violates its sealed contract."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_sha256(value: str, *, field: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise BankQueryError(f"{field} must be a SHA-256 hex digest")


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise BankQueryError("contract mapping keys must be strings")
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in sorted(value.items())}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        if not torch.isfinite(torch.tensor(value)):
            raise BankQueryError("contract floats must be finite")
        return value
    raise BankQueryError(f"unsupported contract value: {type(value).__name__}")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _query_state(query: nn.Module) -> dict[str, Tensor]:
    state = dict(query.state_dict())
    declared = {
        *(name for name, _parameter in query.named_parameters()),
        *(name for name, _buffer in query.named_buffers()),
    }
    if set(state) != declared:
        raise BankQueryError(
            "Bank Query state must be fully persistent; non-persistent buffers are unsupported"
        )
    if not state:
        raise BankQueryError("Bank Query must contain at least one persistent tensor")
    return state


def _query_api_identity(query: nn.Module) -> str:
    query_type = type(query)
    return f"{query_type.__module__}.{query_type.__qualname__}"


def _query_state_schema_fingerprint(query: nn.Module) -> str:
    entries = [
        {
            "name": name,
            "kind": "parameter",
            "dtype": str(parameter.dtype),
            "shape": list(parameter.shape),
        }
        for name, parameter in query.named_parameters()
    ]
    entries.extend(
        {
            "name": name,
            "kind": "buffer",
            "dtype": str(buffer.dtype),
            "shape": list(buffer.shape),
        }
        for name, buffer in query.named_buffers()
    )
    return _fingerprint(sorted(entries, key=lambda item: item["name"]))


def _query_tensor_versions(query: nn.Module) -> tuple[tuple[str, str, int], ...]:
    versions = [
        ("parameter", name, int(parameter._version))
        for name, parameter in query.named_parameters()
    ]
    versions.extend(
        ("buffer", name, int(buffer._version))
        for name, buffer in query.named_buffers()
    )
    return tuple(sorted(versions))


def _validate_fixed_query_state(
    query: nn.Module,
    *,
    expected_versions: tuple[tuple[str, str, int], ...] | None = None,
) -> None:
    if any(parameter.requires_grad for parameter in query.parameters()):
        raise BankQueryError("sealed Bank Query parameters must remain frozen")
    if any(parameter.grad is not None for parameter in query.parameters()):
        raise BankQueryError("sealed Bank Query parameters must not retain gradients")
    if (
        expected_versions is not None
        and _query_tensor_versions(query) != expected_versions
    ):
        raise BankQueryError("sealed Bank Query tensor state was modified after mounting")


@dataclass(frozen=True)
class BankQueryResult:
    """One Bank-local Query representation produced from the current state."""

    value: Tensor

    _runtime_contract_ref: ClassVar[str] = "arti/bank-query-result@1"

    def __post_init__(self) -> None:
        if not isinstance(self.value, Tensor) or not self.value.is_floating_point():
            raise TypeError("BankQueryResult.value must be a floating-point Tensor")
        if self.value.ndim < 1:
            raise BankQueryError("BankQueryResult.value must preserve a batch dimension")


class BankQuery(nn.Module, ABC):
    """A Bank-owned Query that may train before sealing and is fixed after mounting."""

    _component_reference: ClassVar[str] = "arti/bank-query@1"

    def __init__(
        self,
        *,
        input_schema: TensorSchema,
        output_schema: TensorSchema,
        retrieval_contract: Mapping[str, object],
        normalization_contract: Mapping[str, object],
        gradient_contract: GradientContract | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(input_schema, TensorSchema) or not isinstance(
            output_schema, TensorSchema
        ):
            raise TypeError("BankQuery schemas must be TensorSchema values")
        retrieval = _freeze_json(retrieval_contract)
        normalization = _freeze_json(normalization_contract)
        if not isinstance(retrieval, Mapping) or not isinstance(normalization, Mapping):
            raise BankQueryError("retrieval and normalization contracts must be mappings")
        if normalization.get("scope") != "bank_local":
            raise BankQueryError("Bank Query normalization must remain Bank-local")
        gradient = GradientContract.autograd() if gradient_contract is None else gradient_contract
        if not isinstance(gradient, GradientContract):
            raise TypeError("gradient_contract must be GradientContract")
        if gradient.mode != "autograd":
            raise BankQueryError(
                "BankQuery@1 supports only the autograd gradient contract"
            )
        self.input_schema = input_schema
        self.output_schema = output_schema
        # Draft Queries remain deepcopy-friendly while training.  Sealing
        # replaces these dictionaries with immutable mappings.
        self.retrieval_contract = dict(_thaw_json(retrieval))
        self.normalization_contract = dict(_thaw_json(normalization))
        self.gradient_contract = gradient

    def contract_config(self) -> dict[str, object]:
        return {
            "input_schema": self.input_schema.to_dict(),
            "output_schema": self.output_schema.to_dict(),
            "retrieval_contract": _thaw_json(self.retrieval_contract),
            "normalization_contract": _thaw_json(self.normalization_contract),
            "gradient_contract": self.gradient_contract.to_dict(),
        }

    @abstractmethod
    def forward(self, value: Tensor) -> BankQueryResult:
        """Query the Bank-local coordinate system from the latest state."""


class LinearBankQuery(BankQuery):
    """A small trainable projection suitable for Bank-local Query pretraining."""

    _component_reference: ClassVar[str] = "arti/linear-bank-query@1"

    def __init__(
        self,
        input_schema: TensorSchema,
        output_schema: TensorSchema,
        *,
        input_dim: int,
        query_dim: int,
        bias: bool = True,
        retrieval_contract: Mapping[str, object] | None = None,
        normalization_contract: Mapping[str, object] | None = None,
    ) -> None:
        if input_dim <= 0 or query_dim <= 0:
            raise ValueError("input_dim and query_dim must be positive")
        super().__init__(
            input_schema=input_schema,
            output_schema=output_schema,
            retrieval_contract={"kind": "projection"}
            if retrieval_contract is None
            else retrieval_contract,
            normalization_contract={"scope": "bank_local", "kind": "none"}
            if normalization_contract is None
            else normalization_contract,
        )
        self.input_dim = input_dim
        self.query_dim = query_dim
        self.bias = bool(bias)
        self.projection = nn.Linear(input_dim, query_dim, bias=self.bias)

    def contract_config(self) -> dict[str, object]:
        return {
            **super().contract_config(),
            "input_dim": self.input_dim,
            "query_dim": self.query_dim,
            "bias": self.bias,
        }

    def forward(self, value: Tensor) -> BankQueryResult:
        self.input_schema.validate_tensor(value, name="bank_query.input")
        result = self.projection(value)
        self.output_schema.validate_tensor(result, name="bank_query.output")
        return BankQueryResult(result)


@dataclass(frozen=True)
class QueryExecutionSignature:
    """Portable identity for one sealed Bank-owned Query implementation and state."""

    query_ref: str
    api_identity: str
    config_fingerprint: str
    state_fingerprint: str
    state_schema_fingerprint: str
    input_schema: TensorSchema
    output_schema: TensorSchema
    retrieval_contract: Mapping[str, object]
    normalization_contract: Mapping[str, object]
    gradient_contract: GradientContract
    capabilities: tuple[str, ...] = ()
    runtime_mode: str = "sealed"
    state_owner: str = "bank"
    schema_version: int = QUERY_EXECUTION_SIGNATURE_VERSION

    _component_reference: ClassVar[str] = "arti/query-execution-signature@1"

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != QUERY_EXECUTION_SIGNATURE_VERSION
        ):
            raise BankQueryError("unsupported QueryExecutionSignature version")
        try:
            reference = canonical_contract_reference(self.query_ref)
            ComponentRef.parse(reference)
        except (TypeError, ValueError) as error:
            raise BankQueryError("query_ref must be a contract reference") from error
        object.__setattr__(self, "query_ref", reference)
        if not isinstance(self.api_identity, str) or not self.api_identity:
            raise BankQueryError("api_identity must be a non-empty qualified name")
        _require_sha256(self.config_fingerprint, field="config_fingerprint")
        _require_sha256(self.state_fingerprint, field="state_fingerprint")
        _require_sha256(
            self.state_schema_fingerprint, field="state_schema_fingerprint"
        )
        if not isinstance(self.input_schema, TensorSchema) or not isinstance(
            self.output_schema, TensorSchema
        ):
            raise TypeError("QueryExecutionSignature schemas must be TensorSchema values")
        if not isinstance(self.gradient_contract, GradientContract):
            raise TypeError("gradient_contract must be GradientContract")
        if self.gradient_contract.mode != "autograd":
            raise BankQueryError(
                "QueryExecutionSignature@1 supports only autograd gradients"
            )
        retrieval = _freeze_json(self.retrieval_contract)
        normalization = _freeze_json(self.normalization_contract)
        if not isinstance(retrieval, Mapping) or not isinstance(normalization, Mapping):
            raise BankQueryError("Query contracts must be mappings")
        if normalization.get("scope") != "bank_local":
            raise BankQueryError("Bank Query normalization must remain Bank-local")
        if self.runtime_mode != "sealed" or self.state_owner != "bank":
            raise BankQueryError("Query must be sealed and Bank-owned at runtime")
        capabilities = tuple(self.capabilities)
        if tuple(sorted(set(capabilities))) != capabilities or any(
            not isinstance(value, str) or not value for value in capabilities
        ):
            raise BankQueryError("capabilities must be sorted unique non-empty strings")
        object.__setattr__(self, "retrieval_contract", retrieval)
        object.__setattr__(self, "normalization_contract", normalization)
        object.__setattr__(self, "capabilities", capabilities)

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "ref": canonical_contract_reference(self._component_reference),
            "query_ref": self.query_ref,
            "api_identity": self.api_identity,
            "config_fingerprint": self.config_fingerprint,
            "state_fingerprint": self.state_fingerprint,
            "state_schema_fingerprint": self.state_schema_fingerprint,
            "input_schema": self.input_schema.to_dict(),
            "output_schema": self.output_schema.to_dict(),
            "retrieval_contract": _thaw_json(self.retrieval_contract),
            "normalization_contract": _thaw_json(self.normalization_contract),
            "gradient_contract": self.gradient_contract.to_dict(),
            "capabilities": list(self.capabilities),
            "runtime_mode": self.runtime_mode,
            "state_owner": self.state_owner,
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> QueryExecutionSignature:
        required = {
            "schema_version",
            "ref",
            "query_ref",
            "api_identity",
            "config_fingerprint",
            "state_fingerprint",
            "state_schema_fingerprint",
            "input_schema",
            "output_schema",
            "retrieval_contract",
            "normalization_contract",
            "gradient_contract",
            "capabilities",
            "runtime_mode",
            "state_owner",
            "fingerprint",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise BankQueryError(
                "QueryExecutionSignature payload contains missing or unknown fields"
            )
        if value["ref"] != canonical_contract_reference(cls._component_reference):
            raise BankQueryError("QueryExecutionSignature reference is invalid")
        capabilities = value["capabilities"]
        if not isinstance(capabilities, (list, tuple)):
            raise BankQueryError("capabilities must be a sequence")
        result = cls(
            query_ref=value["query_ref"],
            api_identity=value["api_identity"],
            config_fingerprint=value["config_fingerprint"],
            state_fingerprint=value["state_fingerprint"],
            state_schema_fingerprint=value["state_schema_fingerprint"],
            input_schema=TensorSchema.from_dict(value["input_schema"]),
            output_schema=TensorSchema.from_dict(value["output_schema"]),
            retrieval_contract=value["retrieval_contract"],
            normalization_contract=value["normalization_contract"],
            gradient_contract=GradientContract.from_dict(value["gradient_contract"]),
            capabilities=tuple(capabilities),
            runtime_mode=value["runtime_mode"],
            state_owner=value["state_owner"],
            schema_version=value["schema_version"],
        )
        if value["fingerprint"] != result.fingerprint:
            raise BankQueryError("QueryExecutionSignature fingerprint is invalid")
        return result

    def validate_query(self, query: BankQuery) -> None:
        if not isinstance(query, BankQuery):
            raise TypeError("query must be BankQuery")
        if component_ref(query) != self.query_ref:
            raise BankQueryError("Query component identity does not match its signature")
        if _query_api_identity(query) != self.api_identity:
            raise BankQueryError("Query API implementation does not match its signature")
        if _fingerprint(query.contract_config()) != self.config_fingerprint:
            raise BankQueryError("Query configuration does not match its signature")
        if canonical_tensor_state_sha256(_query_state(query)) != self.state_fingerprint:
            raise BankQueryError("Query tensor state does not match its signature")
        if (
            _query_state_schema_fingerprint(query)
            != self.state_schema_fingerprint
        ):
            raise BankQueryError("Query tensor roles do not match its signature")
        if query.input_schema.fingerprint != self.input_schema.fingerprint:
            raise BankQueryError("Query input schema does not match its signature")
        if query.output_schema.fingerprint != self.output_schema.fingerprint:
            raise BankQueryError("Query output schema does not match its signature")
        if _thaw_json(query.retrieval_contract) != _thaw_json(
            self.retrieval_contract
        ):
            raise BankQueryError("Query retrieval contract does not match its signature")
        if _thaw_json(query.normalization_contract) != _thaw_json(
            self.normalization_contract
        ):
            raise BankQueryError(
                "Query normalization contract does not match its signature"
            )
        if query.gradient_contract.fingerprint != self.gradient_contract.fingerprint:
            raise BankQueryError("Query gradient contract does not match its signature")


class SealedBankQuery(nn.Module):
    """A fixed deployment wrapper whose gradients may still flow to its input."""

    _component_reference: ClassVar[str] = "arti/sealed-bank-query@1"

    def __init__(self, query: BankQuery, signature: QueryExecutionSignature) -> None:
        super().__init__()
        if not isinstance(query, BankQuery) or not isinstance(
            signature, QueryExecutionSignature
        ):
            raise TypeError("SealedBankQuery requires a BankQuery and signature")
        signature.validate_query(query)
        query.retrieval_contract = _freeze_json(query.retrieval_contract)
        query.normalization_contract = _freeze_json(query.normalization_contract)
        self.query = query
        self.signature = signature
        self.query.requires_grad_(False)
        self.query.zero_grad(set_to_none=True)
        self.query.eval()
        super().train(False)
        self._sealed_tensor_versions = _query_tensor_versions(self.query)
        self.register_load_state_dict_post_hook(self._validate_loaded_state)

    def _validate_loaded_state(
        self,
        _module: nn.Module,
        _incompatible_keys: object,
    ) -> None:
        self.query.requires_grad_(False)
        self.query.zero_grad(set_to_none=True)
        self.query.eval()
        self.signature.validate_query(self.query)
        self._sealed_tensor_versions = _query_tensor_versions(self.query)

    def _apply(self, fn: object, recurse: bool = True) -> SealedBankQuery:
        if hasattr(self, "_sealed_tensor_versions"):
            self.validate_runtime_state()
        result = super()._apply(fn, recurse=recurse)
        self.signature.validate_query(self.query)
        self._sealed_tensor_versions = _query_tensor_versions(self.query)
        return result

    def train(self, mode: bool = True) -> SealedBankQuery:
        super().train(False)
        self.query.eval()
        return self

    def requires_grad_(self, requires_grad: bool = True) -> SealedBankQuery:
        if requires_grad:
            raise BankQueryError("sealed Bank Query parameters cannot be unfrozen")
        super().requires_grad_(False)
        self.query.zero_grad(set_to_none=True)
        return self

    def contract_config(self) -> dict[str, object]:
        return {"signature": self.signature.to_dict()}

    def validate_runtime_state(self) -> None:
        _validate_fixed_query_state(
            self.query,
            expected_versions=self._sealed_tensor_versions,
        )

    def forward(self, value: Tensor) -> BankQueryResult:
        self.validate_runtime_state()
        self.signature.input_schema.validate_tensor(value, name="sealed_bank_query.input")
        result = self.query(value)
        if not isinstance(result, BankQueryResult):
            raise TypeError("BankQuery must return BankQueryResult")
        self.validate_runtime_state()
        self.signature.output_schema.validate_tensor(
            result.value, name="sealed_bank_query.output"
        )
        return result


def seal_bank_query(
    query: BankQuery,
    *,
    capabilities: tuple[str, ...] = (),
) -> SealedBankQuery:
    """Clone a trained Query, freeze it, and bind its exact config and tensor state."""

    if not isinstance(query, BankQuery):
        raise TypeError("query must be BankQuery")
    sealed = copy.deepcopy(query)
    sealed.requires_grad_(False)
    sealed.zero_grad(set_to_none=True)
    sealed.eval()
    state = _query_state(sealed)
    signature = QueryExecutionSignature(
        query_ref=component_ref(sealed),
        api_identity=_query_api_identity(sealed),
        config_fingerprint=_fingerprint(sealed.contract_config()),
        state_fingerprint=canonical_tensor_state_sha256(state),
        state_schema_fingerprint=_query_state_schema_fingerprint(sealed),
        input_schema=sealed.input_schema,
        output_schema=sealed.output_schema,
        retrieval_contract=sealed.retrieval_contract,
        normalization_contract=sealed.normalization_contract,
        gradient_contract=sealed.gradient_contract,
        capabilities=tuple(capabilities),
    )
    return SealedBankQuery(sealed, signature)


@dataclass(frozen=True)
class BankQueryAsset:
    """Validated metadata and tensor state from one portable sealed Query artifact."""

    path: Path
    signature: QueryExecutionSignature
    weights_sha256: str
    _state_dict: Mapping[str, Tensor]
    artifact_version: int = BANK_QUERY_ARTIFACT_VERSION

    @property
    def state_dict(self) -> Mapping[str, Tensor]:
        return MappingProxyType(
            {
                name: tensor.detach().to("cpu").contiguous().clone()
                for name, tensor in self._state_dict.items()
            }
        )


def bank_query_artifact_path(path: str | Path) -> Path:
    target = Path(path)
    if target.suffix == ".st":
        return target
    return target.with_name(f"{target.name}.arti.st")


def save_bank_query(query: SealedBankQuery, path: str | Path) -> ARTISaveResult:
    """Save a sealed Bank-owned Query as a portable ARTI artifact."""

    if not isinstance(query, SealedBankQuery):
        raise TypeError("save_bank_query requires SealedBankQuery")
    query.validate_runtime_state()
    query.signature.validate_query(query.query)
    return save(
        query,
        bank_query_artifact_path(path),
        scope="all",
        config={
            "artifact_kind": BANK_QUERY_ARTIFACT_KIND,
            "artifact_version": BANK_QUERY_ARTIFACT_VERSION,
            "bank_query": {"signature": query.signature.to_dict()},
        },
    )


def inspect_bank_query(path: str | Path) -> BankQueryAsset:
    """Validate a Bank Query artifact without executing or constructing its module."""

    target = bank_query_artifact_path(path)
    loaded = load(target, load_resources=False, load_checkpoint=False)
    config = loaded.manifest.get("architecture", {}).get("config", {})
    if config.get("artifact_kind") != BANK_QUERY_ARTIFACT_KIND:
        raise BankQueryError("artifact is not a Bank Query")
    artifact_version = config.get("artifact_version")
    if (
        type(artifact_version) is not int
        or artifact_version != BANK_QUERY_ARTIFACT_VERSION
    ):
        raise BankQueryError("unsupported Bank Query artifact version")
    payload = config.get("bank_query")
    if not isinstance(payload, Mapping) or not isinstance(payload.get("signature"), Mapping):
        raise BankQueryError("Bank Query artifact is missing its signature")
    signature = QueryExecutionSignature.from_dict(payload["signature"])
    if loaded.manifest.get("weight_scope") != "all":
        raise BankQueryError("Bank Query artifacts must contain their complete state")
    architecture = loaded.manifest.get("architecture")
    provenance = architecture.get("component_provenance") if isinstance(
        architecture, Mapping
    ) else None
    components = provenance.get("components") if isinstance(provenance, Mapping) else None
    if not isinstance(components, list):
        raise BankQueryError("Bank Query artifact provenance is missing")
    roots = [item for item in components if isinstance(item, Mapping) and item.get("path") == "$"]
    children = [
        item
        for item in components
        if isinstance(item, Mapping) and item.get("path") == "query"
    ]
    expected_root_config = {"signature": signature.to_dict()}
    if (
        len(roots) != 1
        or roots[0].get("ref")
        != canonical_contract_reference(SealedBankQuery._component_reference)
        or roots[0].get("config") != expected_root_config
        or len(children) != 1
        or children[0].get("ref") != signature.query_ref
    ):
        raise BankQueryError(
            "Bank Query artifact provenance does not match its sealed signature"
        )
    prefix = "query."
    if not loaded.state_dict or any(not name.startswith(prefix) for name in loaded.state_dict):
        raise BankQueryError("Bank Query artifact tensor names are invalid")
    state = {
        name.removeprefix(prefix): tensor
        for name, tensor in sorted(loaded.state_dict.items())
    }
    if canonical_tensor_state_sha256(state) != signature.state_fingerprint:
        raise BankQueryError("Bank Query artifact state fingerprint is invalid")
    weights_sha256 = str(loaded.manifest.get("weights", {}).get("sha256", ""))
    _require_sha256(weights_sha256, field="weights_sha256")
    return BankQueryAsset(
        path=target,
        signature=signature,
        weights_sha256=weights_sha256,
        _state_dict=MappingProxyType(
            {
                name: tensor.detach().to("cpu").contiguous().clone()
                for name, tensor in state.items()
            }
        ),
    )


def load_bank_query(path: str | Path, query: BankQuery) -> SealedBankQuery:
    """Load one sealed asset into an explicitly constructed compatible Query."""

    if not isinstance(query, BankQuery):
        raise TypeError("query must be BankQuery")
    asset = inspect_bank_query(path)
    restored = copy.deepcopy(query)
    try:
        result = restored.load_state_dict(dict(asset.state_dict), strict=True)
    except RuntimeError as exc:
        raise BankQueryError("Bank Query artifact state surface is incompatible") from exc
    if result.missing_keys or result.unexpected_keys:
        raise BankQueryError("Bank Query artifact state surface is incompatible")
    restored.requires_grad_(False)
    restored.zero_grad(set_to_none=True)
    restored.eval()
    asset.signature.validate_query(restored)
    return SealedBankQuery(restored, asset.signature)


__all__ = [
    "BANK_QUERY_ARTIFACT_KIND",
    "BANK_QUERY_ARTIFACT_VERSION",
    "QUERY_EXECUTION_SIGNATURE_VERSION",
    "BankQuery",
    "BankQueryAsset",
    "BankQueryError",
    "BankQueryResult",
    "LinearBankQuery",
    "QueryExecutionSignature",
    "SealedBankQuery",
    "bank_query_artifact_path",
    "inspect_bank_query",
    "load_bank_query",
    "save_bank_query",
    "seal_bank_query",
]
