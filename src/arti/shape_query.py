"""Bank-owned shape-polymorphic Query components."""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import ClassVar, Mapping, Sequence

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .component_registry import ComponentRef, canonical_contract_reference, component_ref
from .recall_experts import canonical_tensor_state_sha256
from .tensor_view import TensorView, TensorViewPattern


TENSOR_VIEW_QUERY_SIGNATURE_VERSION = 2


class ShapeQueryError(ValueError):
    """Raised when a shape-polymorphic Bank Query violates its contract."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ShapeQueryError("Query contracts must be string-keyed mappings")
    frozen: dict[str, object] = {}
    for key, item in sorted(value.items()):
        if isinstance(item, Mapping):
            frozen[key] = _freeze_mapping(item)
        elif isinstance(item, (list, tuple)):
            frozen[key] = tuple(item)
        elif isinstance(item, (str, bool, int, float)) or item is None:
            frozen[key] = item
        else:
            raise ShapeQueryError("Query contract contains a non-serializable value")
    return MappingProxyType(frozen)


def _thaw(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw(item) for item in value]
    return value


def _state_schema_fingerprint(module: nn.Module) -> str:
    items = []
    for name, parameter in module.named_parameters():
        items.append(("parameter", name, str(parameter.dtype), tuple(parameter.shape)))
    for name, buffer in module.named_buffers():
        items.append(("buffer", name, str(buffer.dtype), tuple(buffer.shape)))
    return _fingerprint(sorted(items))


def _tensor_versions(module: nn.Module) -> tuple[tuple[str, str, int], ...]:
    result = [
        ("parameter", name, int(parameter._version))
        for name, parameter in module.named_parameters()
    ]
    result.extend(
        ("buffer", name, int(buffer._version)) for name, buffer in module.named_buffers()
    )
    return tuple(sorted(result))


def _validate_frozen(
    module: nn.Module,
    *,
    expected_versions: tuple[tuple[str, str, int], ...] | None = None,
) -> None:
    if any(parameter.requires_grad for parameter in module.parameters()):
        raise ShapeQueryError("sealed TensorView Query parameters must remain frozen")
    if any(parameter.grad is not None for parameter in module.parameters()):
        raise ShapeQueryError("sealed TensorView Query must not retain parameter gradients")
    if expected_versions is not None and _tensor_versions(module) != expected_versions:
        raise ShapeQueryError("sealed TensorView Query state changed after mounting")


def _role_code(name: str, role: str) -> float:
    digest = hashlib.sha256(f"{role}:{name}".encode("utf-8")).digest()
    raw = int.from_bytes(digest[:8], "big") / float(2**64 - 1)
    return 2.0 * raw - 1.0


@dataclass(frozen=True)
class TensorViewObservation:
    """Variable-length, coordinate-aware observations of one TensorView."""

    tokens: Tensor
    mask: Tensor
    view_fingerprint: str

    _runtime_contract_ref: ClassVar[str] = "arti/tensor-view-observation@1"

    def __post_init__(self) -> None:
        if not isinstance(self.tokens, Tensor) or not self.tokens.is_floating_point():
            raise TypeError("observation tokens must be a floating Tensor")
        if self.tokens.ndim != 3 or self.tokens.shape[0] <= 0 or self.tokens.shape[2] <= 0:
            raise ShapeQueryError("observation tokens must have shape [B, T, Q]")
        if (
            not isinstance(self.mask, Tensor)
            or self.mask.dtype != torch.bool
            or self.mask.shape != self.tokens.shape[:2]
            or self.mask.device != self.tokens.device
        ):
            raise ShapeQueryError("observation mask must be bool [B, T] on the token device")
        if not isinstance(self.view_fingerprint, str) or len(self.view_fingerprint) != 64:
            raise ShapeQueryError("view_fingerprint must be a SHA-256 digest")


class TensorViewObserver(nn.Module, ABC):
    """Convert arbitrary admitted TensorViews into a variable observation stream."""

    _component_reference: ClassVar[str] = "arti/tensor-view-observer@1"

    @property
    @abstractmethod
    def query_dim(self) -> int:
        """Return the observation embedding width."""

    @abstractmethod
    def forward(self, view: TensorView) -> TensorViewObservation:
        """Observe the latest view without replacing or transforming its payload."""


class CoordinateTensorViewObserver(TensorViewObserver):
    """A shared pointwise observer over values, axes, extents, and source coordinates."""

    _component_reference: ClassVar[str] = "arti/coordinate-tensor-view-observer@1"

    def __init__(
        self,
        *,
        max_rank: int,
        query_dim: int,
        hidden_dim: int | None = None,
        max_observations: int = 4096,
    ) -> None:
        super().__init__()
        if type(max_rank) is not int or max_rank < 1:
            raise ShapeQueryError("max_rank must be a positive integer")
        if type(query_dim) is not int or query_dim < 1:
            raise ShapeQueryError("query_dim must be a positive integer")
        if hidden_dim is None:
            hidden_dim = max(query_dim, 16)
        if type(hidden_dim) is not int or hidden_dim < 1:
            raise ShapeQueryError("hidden_dim must be a positive integer")
        if type(max_observations) is not int or max_observations < 1:
            raise ShapeQueryError("max_observations must be a positive integer")
        self.max_rank = max_rank
        self._query_dim = query_dim
        self.hidden_dim = hidden_dim
        self.max_observations = max_observations
        feature_dim = 1 + 6 * max_rank
        self.encoder = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, query_dim),
        )

    @property
    def query_dim(self) -> int:
        return self._query_dim

    def contract_config(self) -> dict[str, object]:
        return {
            "max_rank": self.max_rank,
            "query_dim": self.query_dim,
            "hidden_dim": self.hidden_dim,
            "max_observations": self.max_observations,
            "logical_order": "canonical-axis-role-name",
            "storage_flattening": "physical-only",
        }

    @staticmethod
    def _mesh(shape: tuple[int, ...], *, device: torch.device) -> Tensor:
        if not shape:
            return torch.zeros((1, 0), device=device, dtype=torch.int64)
        axes = [torch.arange(size, device=device, dtype=torch.int64) for size in shape]
        return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1).reshape(-1, len(shape))

    def _canonical_payload(
        self,
        view: TensorView,
    ) -> tuple[Tensor, Tensor | None, tuple[object, ...], tuple[int, ...]]:
        batch_axis = view.batch_axis
        logical_axes = [index for index in range(view.value.ndim) if index != batch_axis]
        logical_axes.sort(key=lambda index: (view.axes[index].role, view.axes[index].name))
        order = (batch_axis, *logical_axes)
        value = view.value.permute(order)
        mask = None if view.mask is None else view.mask.permute(order)
        axes = tuple(view.axes[index] for index in logical_axes)
        return value, mask, axes, tuple(logical_axes)

    def forward(self, view: TensorView) -> TensorViewObservation:
        if not isinstance(view, TensorView):
            raise TypeError("CoordinateTensorViewObserver expects TensorView")
        if view.value.ndim > self.max_rank:
            raise ShapeQueryError("TensorView rank exceeds observer max_rank")
        value, value_mask, axes, logical_order = self._canonical_payload(view)
        batch = int(value.shape[0])
        logical_shape = tuple(int(size) for size in value.shape[1:])
        token_count = math.prod(logical_shape) if logical_shape else 1
        flat = value.reshape(batch, token_count, 1)
        current = self._mesh(logical_shape, device=value.device)

        if view.index_map is None or view.index_map.is_identity:
            source = current
            source_shape = logical_shape
        else:
            assert view.index_map.coordinates is not None
            coordinates = view.index_map.coordinates
            nonbatch_order = tuple(
                index - (1 if index > view.batch_axis else 0) for index in logical_order
            )
            if coordinates.ndim == len(view.index_map.target_shape) + 1:
                source = coordinates.permute(*nonbatch_order, coordinates.ndim - 1).reshape(
                    token_count, -1
                )
            else:
                source = coordinates.permute(
                    0,
                    *(index + 1 for index in nonbatch_order),
                    coordinates.ndim - 1,
                ).reshape(batch, token_count, -1)
            source_shape = view.index_map.source_shape

        if token_count > self.max_observations:
            sample = torch.linspace(
                0,
                token_count - 1,
                self.max_observations,
                device=value.device,
            ).round().to(torch.int64)
            flat = flat.index_select(1, sample)
            current = current.index_select(0, sample)
            source = source.index_select(-2, sample)
            if value_mask is not None:
                value_mask = value_mask.reshape(batch, token_count).index_select(1, sample)
            token_count = self.max_observations
        elif value_mask is not None:
            value_mask = value_mask.reshape(batch, token_count)

        current_features = flat.new_zeros((token_count, 4 * self.max_rank))
        for axis_index, axis in enumerate(axes):
            extent = max(axis.extent - 1, 1)
            current_features[:, axis_index] = current[:, axis_index].to(flat.dtype) / extent
            current_features[:, self.max_rank + axis_index] = math.log1p(axis.extent) / 16.0
            current_features[:, 2 * self.max_rank + axis_index] = _role_code(
                axis.name, axis.role
            )
            current_features[:, 3 * self.max_rank + axis_index] = 1.0
        current_features = current_features.unsqueeze(0).expand(batch, -1, -1)

        source_features = flat.new_zeros((batch, token_count, 2 * self.max_rank))
        source_rank = min(len(source_shape), self.max_rank)
        if source.ndim == 2:
            source = source.unsqueeze(0).expand(batch, -1, -1)
        for axis_index in range(source_rank):
            extent = max(source_shape[axis_index] - 1, 1)
            source_features[:, :, axis_index] = source[:, :, axis_index].to(flat.dtype) / extent
            source_features[:, :, self.max_rank + axis_index] = 1.0

        features = torch.cat((flat, current_features, source_features), dim=-1)
        tokens = self.encoder(features)
        mask = (
            torch.ones((batch, token_count), device=value.device, dtype=torch.bool)
            if value_mask is None
            else value_mask
        )
        return TensorViewObservation(tokens, mask, view.descriptor_fingerprint)


class BankMemberMatcher(nn.Module):
    """Late-interaction member scoring with append-only candidate semantics."""

    _component_reference: ClassVar[str] = "arti/bank-member-matcher@1"

    def __init__(self, keys: Tensor, *, member_ids: Sequence[str]) -> None:
        super().__init__()
        if not isinstance(keys, Tensor) or not keys.is_floating_point():
            raise TypeError("member keys must be a floating Tensor")
        if keys.ndim != 2 or keys.shape[0] < 1 or keys.shape[1] < 1:
            raise ShapeQueryError("member keys must have shape [A, Q]")
        members = tuple(member_ids)
        if (
            len(members) != keys.shape[0]
            or len(set(members)) != len(members)
            or any(not isinstance(item, str) or not item for item in members)
        ):
            raise ShapeQueryError("member_ids must uniquely identify every member")
        self.keys = nn.Parameter(keys.detach().clone())
        self.member_ids = members
        self.query_dim = int(keys.shape[1])

    @property
    def member_count(self) -> int:
        return len(self.member_ids)

    def contract_config(self) -> dict[str, object]:
        return {
            "member_ids": list(self.member_ids),
            "member_count": self.member_count,
            "query_dim": self.query_dim,
            "interaction": "normalized-token-member-max",
            "normalization_scope": "member_local",
        }

    def forward(self, observation: TensorViewObservation) -> Tensor:
        if not isinstance(observation, TensorViewObservation):
            raise TypeError("BankMemberMatcher expects TensorViewObservation")
        if observation.tokens.shape[-1] != self.query_dim:
            raise ShapeQueryError("observation width does not match member keys")
        if observation.tokens.device != self.keys.device or observation.tokens.dtype != self.keys.dtype:
            raise ShapeQueryError("observation and member keys must share device and dtype")
        token = F.normalize(observation.tokens, dim=-1)
        keys = F.normalize(self.keys, dim=-1)
        similarity = torch.einsum("btq,aq->bta", token, keys)
        floor = torch.finfo(similarity.dtype).min
        similarity = similarity.masked_fill(~observation.mask.unsqueeze(-1), floor)
        return similarity.max(dim=1).values

    @classmethod
    def concatenate(cls, *matchers: BankMemberMatcher) -> BankMemberMatcher:
        if not matchers or any(not isinstance(item, cls) for item in matchers):
            raise TypeError("concatenate requires BankMemberMatcher values")
        query_dim = matchers[0].query_dim
        device = matchers[0].keys.device
        dtype = matchers[0].keys.dtype
        if any(
            item.query_dim != query_dim
            or item.keys.device != device
            or item.keys.dtype != dtype
            for item in matchers
        ):
            raise ShapeQueryError("concatenated matchers must share query width, device, and dtype")
        members = tuple(member for item in matchers for member in item.member_ids)
        if len(set(members)) != len(members):
            raise ShapeQueryError("concatenated member_ids must remain unique")
        return cls(torch.cat([item.keys.detach() for item in matchers], dim=0), member_ids=members)


@dataclass(frozen=True)
class TensorViewQueryResult:
    """Bank-local member scores plus the observations that produced them."""

    scores: Tensor
    member_ids: tuple[str, ...]
    observation: TensorViewObservation

    _runtime_contract_ref: ClassVar[str] = "arti/tensor-view-query-result@1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "member_ids", tuple(self.member_ids))
        if not isinstance(self.scores, Tensor) or not self.scores.is_floating_point():
            raise TypeError("TensorViewQueryResult.scores must be a floating Tensor")
        if self.scores.ndim != 2 or self.scores.shape[1] != len(self.member_ids):
            raise ShapeQueryError("TensorViewQueryResult scores must have shape [B, A]")
        if self.scores.shape[0] != self.observation.tokens.shape[0]:
            raise ShapeQueryError("Query score and observation batches disagree")


class TensorViewBankQuery(nn.Module):
    """A trainable Bank Query over arbitrary admitted TensorView shapes."""

    _component_reference: ClassVar[str] = "arti/tensor-view-bank-query@2"

    def __init__(
        self,
        *,
        pattern: TensorViewPattern,
        observer: TensorViewObserver,
        matcher: BankMemberMatcher,
        normalization_contract: Mapping[str, object] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(pattern, TensorViewPattern):
            raise TypeError("pattern must be TensorViewPattern")
        if not isinstance(observer, TensorViewObserver):
            raise TypeError("observer must be TensorViewObserver")
        if not isinstance(matcher, BankMemberMatcher):
            raise TypeError("matcher must be BankMemberMatcher")
        if observer.query_dim != matcher.query_dim:
            raise ShapeQueryError("observer and matcher query dimensions must match")
        normalization = normalization_contract or {
            "scope": "bank_local",
            "candidate_axis": "A",
        }
        frozen = _freeze_mapping(normalization)
        if frozen.get("scope") != "bank_local":
            raise ShapeQueryError("TensorView Query normalization must remain Bank-local")
        self.pattern = pattern
        self.observer = observer
        self.matcher = matcher
        self.normalization_contract = dict(_thaw(frozen))

    def contract_config(self) -> dict[str, object]:
        return {
            "pattern": self.pattern.to_dict(),
            "observer_ref": component_ref(self.observer),
            "matcher_ref": component_ref(self.matcher),
            "member_ids": list(self.matcher.member_ids),
            "normalization_contract": _thaw(self.normalization_contract),
            "query_semantics": "latest-tensor-view-read-only",
        }

    def forward(self, view: TensorView) -> TensorViewQueryResult:
        self.pattern.validate(view)
        observation = self.observer(view)
        scores = self.matcher(observation)
        return TensorViewQueryResult(scores, self.matcher.member_ids, observation)


@dataclass(frozen=True)
class TensorViewQueryExecutionSignature:
    """Immutable identity of a pretrained, shape-polymorphic Bank Query."""

    query_ref: str
    api_identity: str
    config_fingerprint: str
    state_schema_fingerprint: str
    state_fingerprint: str
    pattern: TensorViewPattern
    observer_ref: str
    matcher_ref: str
    member_ids: tuple[str, ...]
    normalization_contract: Mapping[str, object]
    schema_version: int = TENSOR_VIEW_QUERY_SIGNATURE_VERSION

    _component_reference: ClassVar[str] = "arti/query-execution-signature@2"

    def __post_init__(self) -> None:
        object.__setattr__(self, "member_ids", tuple(self.member_ids))
        if self.schema_version != TENSOR_VIEW_QUERY_SIGNATURE_VERSION:
            raise ShapeQueryError("unsupported TensorView Query signature version")
        for value, name in (
            (self.query_ref, "query_ref"),
            (self.observer_ref, "observer_ref"),
            (self.matcher_ref, "matcher_ref"),
            (self.api_identity, "api_identity"),
        ):
            if not isinstance(value, str) or not value:
                raise ShapeQueryError(f"{name} must be non-empty")
        try:
            for field in ("query_ref", "observer_ref", "matcher_ref"):
                reference = canonical_contract_reference(getattr(self, field))
                ComponentRef.parse(reference)
                object.__setattr__(self, field, reference)
        except (TypeError, ValueError) as error:
            raise ShapeQueryError("TensorView Query references must be content-addressed") from error
        for value in (
            self.config_fingerprint,
            self.state_schema_fingerprint,
            self.state_fingerprint,
        ):
            if not isinstance(value, str) or len(value) != 64:
                raise ShapeQueryError("Query signature fingerprints must be SHA-256 digests")
        if not isinstance(self.pattern, TensorViewPattern):
            raise TypeError("pattern must be TensorViewPattern")
        if not self.member_ids or len(set(self.member_ids)) != len(self.member_ids):
            raise ShapeQueryError("signature member_ids must be non-empty and unique")
        normalization = _freeze_mapping(self.normalization_contract)
        if normalization.get("scope") != "bank_local":
            raise ShapeQueryError("signature normalization must remain Bank-local")
        object.__setattr__(self, "normalization_contract", normalization)

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
            "state_schema_fingerprint": self.state_schema_fingerprint,
            "state_fingerprint": self.state_fingerprint,
            "pattern": self.pattern.to_dict(),
            "observer_ref": self.observer_ref,
            "matcher_ref": self.matcher_ref,
            "member_ids": list(self.member_ids),
            "normalization_contract": _thaw(self.normalization_contract),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "fingerprint": self.fingerprint}

    def __deepcopy__(self, memo: dict[int, object]) -> "TensorViewQueryExecutionSignature":
        """Clone immutable signature data for independent attached Federal graphs."""

        result = type(self).from_dict(self.to_dict())
        memo[id(self)] = result
        return result

    @classmethod
    def from_dict(
        cls,
        value: Mapping[str, object],
    ) -> TensorViewQueryExecutionSignature:
        required = {
            "schema_version",
            "ref",
            "query_ref",
            "api_identity",
            "config_fingerprint",
            "state_schema_fingerprint",
            "state_fingerprint",
            "pattern",
            "observer_ref",
            "matcher_ref",
            "member_ids",
            "normalization_contract",
            "fingerprint",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise ShapeQueryError(
                "TensorView Query signature contains missing or unknown fields"
            )
        if value["ref"] != canonical_contract_reference(cls._component_reference):
            raise ShapeQueryError("TensorView Query signature reference is invalid")
        member_ids = value["member_ids"]
        if not isinstance(member_ids, (list, tuple)):
            raise ShapeQueryError("TensorView Query member_ids must be a sequence")
        result = cls(
            query_ref=value["query_ref"],
            api_identity=value["api_identity"],
            config_fingerprint=value["config_fingerprint"],
            state_schema_fingerprint=value["state_schema_fingerprint"],
            state_fingerprint=value["state_fingerprint"],
            pattern=TensorViewPattern.from_dict(value["pattern"]),
            observer_ref=value["observer_ref"],
            matcher_ref=value["matcher_ref"],
            member_ids=tuple(member_ids),
            normalization_contract=value["normalization_contract"],
            schema_version=value["schema_version"],
        )
        if value["fingerprint"] != result.fingerprint:
            raise ShapeQueryError("TensorView Query signature fingerprint is invalid")
        return result

    def validate_query(self, query: TensorViewBankQuery) -> None:
        if not isinstance(query, TensorViewBankQuery):
            raise TypeError("query must be TensorViewBankQuery")
        if component_ref(query) != self.query_ref:
            raise ShapeQueryError("Query component identity changed")
        identity = f"{type(query).__module__}.{type(query).__qualname__}"
        if identity != self.api_identity:
            raise ShapeQueryError("Query API identity changed")
        if _fingerprint(query.contract_config()) != self.config_fingerprint:
            raise ShapeQueryError("Query configuration changed")
        if _state_schema_fingerprint(query) != self.state_schema_fingerprint:
            raise ShapeQueryError("Query state schema changed")
        if canonical_tensor_state_sha256(query.state_dict()) != self.state_fingerprint:
            raise ShapeQueryError("Query tensor state changed")


class SealedTensorViewBankQuery(nn.Module):
    """A fixed TensorView Query asset mounted by one autonomous Bank."""

    _component_reference: ClassVar[str] = "arti/sealed-bank-query@2"

    def __init__(
        self,
        query: TensorViewBankQuery,
        signature: TensorViewQueryExecutionSignature,
    ) -> None:
        super().__init__()
        if not isinstance(query, TensorViewBankQuery) or not isinstance(
            signature, TensorViewQueryExecutionSignature
        ):
            raise TypeError("sealed TensorView Query requires a Query and signature")
        signature.validate_query(query)
        self.query = query
        self.signature = signature
        self.query.eval()
        self.query.requires_grad_(False)
        self._versions = _tensor_versions(self.query)
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
        self._versions = _tensor_versions(self.query)

    def _apply(self, fn: object, recurse: bool = True) -> SealedTensorViewBankQuery:
        if hasattr(self, "_versions"):
            self.validate_runtime_state()
        result = super()._apply(fn, recurse=recurse)
        self.signature.validate_query(self.query)
        self._versions = _tensor_versions(self.query)
        return result

    def contract_config(self) -> dict[str, object]:
        return {"signature": self.signature.to_dict()}

    def train(self, mode: bool = True) -> SealedTensorViewBankQuery:
        super().train(False)
        self.query.train(False)
        return self

    def requires_grad_(self, requires_grad: bool = True) -> SealedTensorViewBankQuery:
        if requires_grad:
            raise ShapeQueryError("sealed TensorView Query parameters cannot be unfrozen")
        super().requires_grad_(False)
        self.query.zero_grad(set_to_none=True)
        return self

    def validate_runtime_state(self) -> None:
        _validate_frozen(self.query, expected_versions=self._versions)

    def forward(self, view: TensorView) -> TensorViewQueryResult:
        self.validate_runtime_state()
        result = self.query(view)
        if result.member_ids != self.signature.member_ids:
            raise ShapeQueryError("runtime Query member identities changed")
        self.validate_runtime_state()
        return result


def seal_tensor_view_bank_query(query: TensorViewBankQuery) -> SealedTensorViewBankQuery:
    """Freeze a pretrained TensorView Query without detaching input gradients."""

    if not isinstance(query, TensorViewBankQuery):
        raise TypeError("query must be TensorViewBankQuery")
    sealed = copy.deepcopy(query)
    sealed.eval()
    sealed.requires_grad_(False)
    signature = TensorViewQueryExecutionSignature(
        query_ref=component_ref(sealed),
        api_identity=f"{type(sealed).__module__}.{type(sealed).__qualname__}",
        config_fingerprint=_fingerprint(sealed.contract_config()),
        state_schema_fingerprint=_state_schema_fingerprint(sealed),
        state_fingerprint=canonical_tensor_state_sha256(sealed.state_dict()),
        pattern=sealed.pattern,
        observer_ref=component_ref(sealed.observer),
        matcher_ref=component_ref(sealed.matcher),
        member_ids=sealed.matcher.member_ids,
        normalization_contract=sealed.normalization_contract,
    )
    return SealedTensorViewBankQuery(sealed, signature)


__all__ = [
    "TENSOR_VIEW_QUERY_SIGNATURE_VERSION",
    "BankMemberMatcher",
    "CoordinateTensorViewObserver",
    "SealedTensorViewBankQuery",
    "ShapeQueryError",
    "TensorViewBankQuery",
    "TensorViewObservation",
    "TensorViewObserver",
    "TensorViewQueryExecutionSignature",
    "TensorViewQueryResult",
    "seal_tensor_view_bank_query",
]
