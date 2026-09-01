"""Python-owned identities and provenance for ARTI components.

Component provenance is deliberately separate from tensor execution.  A
component reference identifies a mathematical/API contract; configuration and
parameter-schema fingerprints identify one concrete instance.  The registry
never imports code named by an artifact and never infers a version from a
class name or a tensor shape.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from threading import RLock
from typing import Any, Literal

import torch
from torch import Tensor, nn


COMPONENT_PROVENANCE_VERSION = 2
COMPONENT_STATE_CONTRACT_VERSION = 1
ComponentLifecycle = Literal["stable", "alpha", "legacy", "deprecated"]
ArtifactPolicy = Literal["portable", "runtime_only", "host_bound"]
_LIFECYCLES = frozenset({"stable", "alpha", "legacy", "deprecated"})
_ARTIFACT_POLICIES = frozenset({"portable", "runtime_only", "host_bound"})
_COMPONENT_NAME = r"[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?"
_REFERENCE = re.compile(
    rf"^(?P<namespace>{_COMPONENT_NAME})/(?P<name>{_COMPONENT_NAME})@(?P<version>[1-9][0-9]*)$"
)


class ComponentRegistryError(ValueError):
    """Base error for component identity and provenance failures."""


class InvalidComponentRefError(ComponentRegistryError):
    """Raised when a component reference is not canonical."""


class DuplicateComponentError(ComponentRegistryError):
    """Raised when a component identity or alias is registered twice."""


class UnknownComponentError(ComponentRegistryError):
    """Raised when an exact component identity is not registered."""


class ComponentCompatibilityError(ComponentRegistryError):
    """Raised when an artifact component cannot be loaded safely."""


def _canonical_json(value: Any) -> str:
    return json.dumps(
        _normalize(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _tensor_descriptor(value: Tensor) -> dict[str, Any]:
    try:
        shape: list[int | str] = list(value.shape)
    except RuntimeError:
        shape = ["uninitialized"]
    return {
        "dtype": str(value.dtype),
        "shape": shape,
        "device": str(value.device),
    }


def _normalize(value: Any) -> Any:
    """Convert metadata to deterministic JSON data without storing tensor values."""

    if isinstance(value, Tensor):
        return {"__tensor__": _tensor_descriptor(value)}
    if is_dataclass(value):
        return _normalize(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _normalize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normalize(item) for item in value]
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("component metadata floats must be finite")
        return value
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    return str(value)


@dataclass(frozen=True, order=True)
class ComponentRef:
    """Canonical ``namespace/name@version`` component identity."""

    namespace: str
    name: str
    version: int

    def __post_init__(self) -> None:
        if re.fullmatch(_COMPONENT_NAME, self.namespace) is None:
            raise InvalidComponentRefError("component namespace is invalid")
        if re.fullmatch(_COMPONENT_NAME, self.name) is None:
            raise InvalidComponentRefError("component name is invalid")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version <= 0:
            raise InvalidComponentRefError("component version must be a positive integer")

    @property
    def mechanism_id(self) -> str:
        return f"{self.namespace}/{self.name}"

    @property
    def reference(self) -> str:
        return f"{self.mechanism_id}@{self.version}"

    @classmethod
    def parse(cls, reference: str) -> "ComponentRef":
        if not isinstance(reference, str):
            raise InvalidComponentRefError("component reference must be a string")
        match = _REFERENCE.fullmatch(reference)
        if match is None:
            raise InvalidComponentRefError(
                "component reference must use namespace/name@version syntax"
            )
        return cls(
            namespace=match.group("namespace"),
            name=match.group("name"),
            version=int(match.group("version")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "name": self.name,
            "version": self.version,
        }


ConfigBuilder = Callable[[Any], Mapping[str, Any]]
DependencyBuilder = Callable[[Any], Sequence[str]]
Factory = Callable[..., Any]


@dataclass(frozen=True)
class ComponentRegistration:
    """A code-side registration for one exact component contract."""

    identity: ComponentRef
    component_type: type[Any]
    lifecycle: ComponentLifecycle
    variant: str
    config_schema_version: int = 1
    state_schema_version: int = 1
    aliases: tuple[str, ...] = ()
    deprecated_aliases: tuple[str, ...] = ()
    constructible: bool = True
    artifact_policy: ArtifactPolicy = "portable"
    factory: Factory | None = None
    config_builder: ConfigBuilder | None = None
    dependency_builder: DependencyBuilder | None = None
    capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.lifecycle not in _LIFECYCLES:
            raise ValueError(f"unsupported component lifecycle: {self.lifecycle!r}")
        if not isinstance(self.component_type, type):
            raise TypeError("component_type must be a type")
        if not isinstance(self.variant, str) or not self.variant:
            raise ValueError("component variant must be a non-empty string")
        if set(self.aliases) & set(self.deprecated_aliases):
            raise ValueError("component aliases cannot also be deprecated aliases")
        if type(self.constructible) is not bool:
            raise TypeError("component constructible must be boolean")
        if not self.constructible and self.factory is not None:
            raise ValueError("a non-constructible component cannot expose a factory")
        if self.artifact_policy not in _ARTIFACT_POLICIES:
            raise ValueError(f"unsupported artifact policy: {self.artifact_policy!r}")
        for value, name in (
            (self.config_schema_version, "config_schema_version"),
            (self.state_schema_version, "state_schema_version"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        capabilities = tuple(self.capabilities)
        if any(
            not isinstance(value, str)
            or re.fullmatch(r"[a-z][a-z0-9]*(?:[.-][a-z0-9]+)*", value) is None
            for value in capabilities
        ):
            raise ValueError("component capabilities must be canonical lowercase names")
        if tuple(sorted(set(capabilities))) != capabilities:
            raise ValueError("component capabilities must be sorted and unique")

    @property
    def reference(self) -> str:
        return self.identity.reference

    def config(self, component: Any) -> dict[str, Any]:
        builder = self.config_builder or _default_config
        return dict(_normalize(builder(component)))

    def dependencies(self, component: Any) -> tuple[str, ...]:
        builder = self.dependency_builder
        if builder is None:
            return ()
        values = tuple(builder(component))
        if any(not isinstance(value, str) for value in values):
            raise ValueError(f"dependencies for {self.reference} must be strings")
        return tuple(sorted(set(values)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "module",
            "ref": self.reference,
            "mechanism_id": self.identity.mechanism_id,
            "mechanism_version": self.identity.version,
            "variant": self.variant,
            "lifecycle": self.lifecycle,
            "config_schema_version": self.config_schema_version,
            "state_schema_version": self.state_schema_version,
            "aliases": list(self.aliases),
            "deprecated_aliases": list(self.deprecated_aliases),
            "constructible": self.constructible,
            "artifact_policy": self.artifact_policy,
            "capabilities": list(self.capabilities),
        }


@dataclass(frozen=True)
class ComponentSpec:
    """Serializable identity and schema metadata for one module instance."""

    path: str
    reference: str
    api: str
    variant: str
    lifecycle: ComponentLifecycle
    config_schema_version: int
    state_schema_version: int
    config: Mapping[str, Any]
    config_fingerprint: str
    parameter_schema_fingerprint: str
    dependencies: tuple[str, ...] = ()
    capabilities: tuple[str, ...] = ()

    @property
    def identity(self) -> ComponentRef:
        return ComponentRef.parse(self.reference)

    def to_dict(self) -> dict[str, Any]:
        identity = self.identity
        return {
            "path": self.path,
            "api": self.api,
            "ref": self.reference,
            "mechanism_id": identity.mechanism_id,
            "mechanism_version": identity.version,
            "variant": self.variant,
            "lifecycle": self.lifecycle,
            "config_schema_version": self.config_schema_version,
            "state_schema_version": self.state_schema_version,
            "config": _normalize(self.config),
            "config_fingerprint": self.config_fingerprint,
            "parameter_schema_fingerprint": self.parameter_schema_fingerprint,
            "dependencies": list(self.dependencies),
            "capabilities": list(self.capabilities),
        }


def _default_config(component: Any) -> Mapping[str, Any]:
    candidate = getattr(component, "config", None)
    if candidate is not None and not isinstance(candidate, (str, bytes)):
        if is_dataclass(candidate) or isinstance(candidate, Mapping):
            return candidate
    if is_dataclass(component):
        return asdict(component)
    return {}


def _attr(component: Any, name: str, default: Any = None) -> Any:
    current = component
    for part in name.split("."):
        if isinstance(current, Mapping):
            current = current.get(part, default)
        else:
            current = getattr(current, part, default)
        if current is default:
            return default
    return current


def _fields(*names: str) -> ConfigBuilder:
    def build(component: Any) -> Mapping[str, Any]:
        return {name: _attr(component, name) for name in names}

    return build


def _half_config(component: Any) -> Mapping[str, Any]:
    config = {
        "threshold": _attr(component, "_threshold_init"),
        "base": _attr(component, "_base_init"),
        "scale": _attr(component, "_scale_init"),
        "stochastic": _attr(component, "stochastic"),
        "learnable": _attr(component, "learnable"),
    }
    if _attr(component, "context_mode", "none") != "none":
        config.update(
            {
                "context_mode": _attr(component, "context_mode"),
                "context_axes": list(_attr(component, "context_axes")),
                "context_gain": _attr(component, "context_gain"),
            }
        )
    metadata = getattr(component, "survival_metadata", None)
    if isinstance(metadata, Mapping):
        survival = dict(metadata)
        if survival.get("runtime_only"):
            raise ComponentCompatibilityError(
                "custom survival is runtime-only and cannot be stored in an ARTI artifact; "
                "use a builtin survival or an explicitly authorized portable implementation"
            )
        config["survival"] = survival
    return config


def _dropout_config(component: Any, *fields: str) -> Mapping[str, Any]:
    result = {field: _attr(component, field) for field in fields}
    result["dropout"] = _attr(component, "dropout.p")
    return result


def _unfold_config(component: Any) -> Mapping[str, Any]:
    return {
        name: _attr(component, name)
        for name in (
            "dim",
            "exposed",
            "guide_dim",
            "condition_dim",
            "hidden_dim",
            "temperature",
            "sinkhorn_steps",
            "max_length",
            "hard_backend",
            "layout_mode",
            "value_operators",
            "value_rank",
            "query_chunk_size",
            "operator_chunk_size",
            "validate_values",
            "_operator_schedule",
        )
    }


def _fold_config(component: Any) -> Mapping[str, Any]:
    return _dropout_config(
        component,
        "k",
        "dim",
        "hidden_dim",
        "temperature",
        "mode",
        "topk",
        "heads",
        "eps",
    )


def _fixed_topology_policy_config(component: Any) -> Mapping[str, Any]:
    order = _attr(component, "order")
    return {
        "order": order.detach().cpu().tolist(),
        "tie_break": "stable-index",
        "validity": "valid-first",
    }


def _topology_surrogate_config(component: Any) -> Mapping[str, Any]:
    if getattr(component, "_component_reference", None) == "arti/topology-surrogate@2":
        return _attr(component, "topology_contract")()
    return {"temperature": _attr(component, "temperature"), "path": "backward-only"}


def _learned_topology_policy_config(component: Any) -> Mapping[str, Any]:
    return {
        "dim": _attr(component, "dim"),
        "hidden_dim": _attr(component, "hidden_dim"),
        "value_input": "detached",
    }


def _learned_topology_policy_dependencies(component: Any) -> Sequence[str]:
    return ()


def _topology_formula_config(component: Any) -> Mapping[str, Any]:
    contract = _attr(component, "contract")
    return {
        "factor_dim": _attr(contract, "factor_dim"),
        "mode": _attr(contract, "mode"),
        "output_semantics": _attr(contract, "output_semantics"),
        "api_version": _attr(contract, "api_version"),
        "trainable": _attr(component, "trainable"),
    }


def _topology_operand_bank_config(component: Any) -> Mapping[str, Any]:
    return {
        **_attr(component, "structure_contract"),
        "query": "fixed-address",
        "state": "trainable-values",
    }


def _fixed_topology_query_config(component: Any) -> Mapping[str, Any]:
    return _attr(component, "topology_contract")()


def _bank_formula_topology_policy_config(component: Any) -> Mapping[str, Any]:
    return {
        "dim": _attr(component, "dim"),
        "key_dim": _attr(component, "key_dim"),
        "query_seed": _attr(component, "query_seed"),
        "bank_count": len(_attr(component, "banks")),
        "ordered_banks": [
            _attr(bank, "structure_contract") for bank in _attr(component, "banks")
        ],
        "bank_weights": _attr(component, "bank_weights").detach().cpu().tolist(),
        "query": _attr(component, "query.topology_contract")(),
        "formula": {
            "ref": _attr(component, "formula._component_reference"),
            "contract_fingerprint": _attr(component, "formula.contract.fingerprint"),
            "factor_dim": _attr(component, "formula.contract.factor_dim"),
        },
        "normalization": "per-bank",
        "merge": "explicit-weighted-sum",
        "diagnostics": _attr(component, "diagnostics"),
        "diagnostic_slot_limit": _attr(component, "diagnostic_slot_limit"),
    }


def _bank_formula_topology_policy_dependencies(component: Any) -> Sequence[str]:
    dependencies = [
        _attr(component, "query._component_reference"),
        _attr(component, "formula._component_reference"),
    ]
    dependencies.extend(
        _attr(bank, "_component_reference") for bank in _attr(component, "banks")
    )
    return tuple(dependencies)


def _bank_formula_route_source_config(component: Any) -> Mapping[str, Any]:
    return {
        "program": _attr(component, "program.to_dict")(),
        "program_fingerprint": _attr(component, "program.fingerprint"),
        "active_count": _attr(component, "active_count"),
        "estimator": _attr(component, "estimator"),
        "policy_order": [
            {
                "ref": component_ref(policy),
                "config_fingerprint": component_spec(policy).config_fingerprint,
            }
            for policy in _attr(component, "policies")
        ],
        "candidate_mask_shape": list(_attr(component, "_candidate_mask.shape")),
        "route_source_config_fingerprint": _attr(
            component, "config_fingerprint"
        ),
        "limits": dict(_attr(component, "limits.__dict__")),
        "route_plan": "pre-execution-static-ssa-availability",
        "write_authority": "host-intervened-support",
    }


def _bank_formula_route_source_dependencies(component: Any) -> Sequence[str]:
    return tuple(component_ref(policy) for policy in _attr(component, "policies"))


def _routed_formula_fabric_compute_config(component: Any) -> Mapping[str, Any]:
    return {
        "compute": component_spec(_attr(component, "compute")).to_dict(),
        "route_source": component_spec(_attr(component, "route_source")).to_dict(),
        "active_count": _attr(component, "active_count"),
        "explicit_route": "per-call-override",
        "route_contract": "bound-source-or-explicit-override",
        "adapter_config_fingerprint": _attr(component, "config_fingerprint"),
    }


def _routed_formula_fabric_compute_dependencies(component: Any) -> Sequence[str]:
    return (
        component_ref(_attr(component, "compute")),
        component_ref(_attr(component, "route_source")),
    )


def _reversible_topology_config(component: Any) -> Mapping[str, Any]:
    return {
        "active_count": _attr(component, "active_count"),
        "axis": _attr(component, "axis"),
        "policy_ref": _attr(component, "policy._component_reference"),
        "operator_ref": _attr(component, "operator._component_reference"),
        "surrogate_ref": (
            None
            if getattr(component, "surrogate", None) is None
            else _attr(component, "surrogate._component_reference")
        ),
        "inverse": "recorded-permutation",
        "original_instance_conservation": True,
        "record_ref": "arti/fold-record@1",
        "state_ref": "arti/fold-state@1",
    }


def _reversible_topology_dependencies(component: Any) -> Sequence[str]:
    policy = getattr(component, "policy", None)
    registration = get_component_registry().registration_for(policy)
    dependencies = [] if registration is None else [registration.reference]
    operator = getattr(component, "operator", None)
    operator_registration = get_component_registry().registration_for(operator)
    if operator_registration is not None:
        dependencies.append(operator_registration.reference)
    surrogate = getattr(component, "surrogate", None)
    surrogate_registration = get_component_registry().registration_for(surrogate)
    if surrogate_registration is not None:
        dependencies.append(surrogate_registration.reference)
    return tuple(dependencies)


def _topology_fold_config(component: Any) -> Mapping[str, Any]:
    topology = _attr(component, "topology")
    return {
        "topology_ref": _attr(topology, "_component_reference"),
        "active_count": _attr(topology, "active_count"),
        "axis": _attr(topology, "axis"),
        "record_ref": "arti/fold-record@1",
        "source_contract_binding": _attr(component, "source_contract_binding"),
    }


def _topology_fold_dependencies(component: Any) -> Sequence[str]:
    topology = getattr(component, "topology", None)
    registration = get_component_registry().registration_for(topology)
    dependencies = [] if registration is None else [registration.reference]
    dependencies.extend(("arti/fold-record@1", "arti/fold-state@1"))
    return tuple(dependencies)


def _inverse_topology_contract_config(component: Any) -> Mapping[str, Any]:
    return _attr(component, "topology_contract")()


def _topology_unfold_config(component: Any) -> Mapping[str, Any]:
    inverse = _attr(component, "inverse_contract")
    return {
        "inverse_contract_ref": _attr(inverse, "_component_reference"),
        "active_count": _attr(inverse, "active_count"),
        "axis": _attr(inverse, "axis"),
        "record_ref": "arti/fold-record@1",
    }


def _topology_unfold_dependencies(component: Any) -> Sequence[str]:
    inverse = getattr(component, "inverse_contract", None)
    registration = get_component_registry().registration_for(inverse)
    dependencies = [] if registration is None else [registration.reference]
    dependencies.extend(("arti/fold-record@1", "arti/fold-state@1"))
    return tuple(dependencies)


def _fold_record_config(component: Any) -> Mapping[str, Any]:
    return {
        "schema_version": _attr(component, "schema_version"),
        "producer_ref": _attr(component, "producer_ref"),
        "inverse_ref": _attr(component, "inverse_ref"),
        "topology_ref": _attr(component, "topology_ref"),
        "transport_contract_fingerprint": _attr(
            component, "topology_config_fingerprint"
        ),
        "producer_provenance_fingerprint": _attr(
            component, "producer_provenance_fingerprint"
        ),
        "axis": _attr(component, "axis"),
        "active_count": _attr(component, "active_count"),
        "original_shape": list(_attr(component, "original_shape")),
    }


def _fold_state_config(component: Any) -> Mapping[str, Any]:
    return {
        "schema_version": _attr(component, "schema_version"),
        "record_ref": _attr(component, "record._component_reference"),
        "active_shape": list(_attr(component, "active.shape")),
        "folded_shape": list(_attr(component, "folded.shape")),
    }


def _adaptive_pulse_config(component: Any) -> Mapping[str, Any]:
    _attr(component, "_validate_manifest_binding")()
    manifest = _attr(component, "manifest")
    return {
        "manifest_ref": "arti/pulse-stage-graph@1",
        "manifest": manifest.to_dict(),
        "manifest_fingerprint": manifest.fingerprint,
    }


def _adaptive_pulse_dependencies(component: Any) -> Sequence[str]:
    _attr(component, "_validate_manifest_binding")()
    return (
        "arti/pulse-stage-graph@1",
        *_attr(component, "enabled_components"),
    )


def _arti_layer_config(component: Any) -> Mapping[str, Any]:
    return {
        "pulse": component_spec(_attr(component, "pulse")).to_dict(),
    }


def _arti_layer_dependencies(component: Any) -> Sequence[str]:
    return (component_ref(_attr(component, "pulse")),)


def _learned_pulse_config(component: Any) -> Mapping[str, Any]:
    fold = getattr(component, "fold", None)
    return _dropout_config(
        component,
        "k",
        "dim",
        "hidden_dim",
        "refine_enabled",
        "refine_mode",
        "fold_mode",
        "fold_topk",
        "q_topk",
        "use_half",
        "temperature",
        "eps",
    ) | {"fold_heads": _attr(fold, "heads", 1)}


def _recall_config(component: Any) -> Mapping[str, Any]:
    state_config = _attr(component, "state.config")
    result = {
        "dim": _attr(component, "dim"),
        "slots": _attr(component, "slots"),
        "formula": _attr(component, "formula_id"),
        "formula_origin": _attr(component, "formula_origin"),
        "formula_portable": _attr(component, "formula_portable"),
        "activation": _attr(state_config, "recall_activation"),
        "recognition": _attr(state_config, "recall_recognition_mode"),
        "routing": _attr(state_config, "recall_routing"),
        "key_dim": _attr(state_config, "recall_key_dim"),
        "group_size": _attr(state_config, "recall_group_size"),
        "group_topk": _attr(state_config, "recall_group_topk"),
        "value_composition": _attr(state_config, "recall_value_composition"),
        "route_exploration": _attr(state_config, "recall_route_exploration"),
        "dropout": _attr(state_config, "dropout"),
        "routing_normalizer": _attr(component, "routing_normalizer"),
        "expert_names": list(_attr(component, "state.recall.expert_names")),
        "expert_route_ranges": [
            list(value)
            for value in _attr(component, "state.recall._expert_route_ranges")
        ],
        "expert_member_fingerprints": list(
            _attr(component, "state.recall.expert_member_fingerprints")
        ),
        "expert_weights": list(_attr(component, "state.recall.expert_weights")),
        "expert_influences": list(
            _attr(component, "state.recall.expert_influences")
        ),
    }
    if _attr(component, "_component_reference") == "arti/recall@4":
        result.update(
            {
                "breadth": _attr(component, "breadth"),
                "breadth_mode": _attr(component, "breadth_mode"),
                "breadth_aggregation": _attr(component, "breadth_aggregation"),
            }
        )
    if result["formula_portable"] is not True:
        raise ComponentCompatibilityError(
            "custom Recall Formula is runtime-only and cannot be stored in an ARTI artifact; "
            "use a builtin Formula or an explicitly authorized portable implementation"
        )
    return result


def _recall_dependencies(component: Any) -> Sequence[str]:
    result: list[str] = []
    formula = _attr(component, "formula_id")
    if isinstance(formula, str) and formula != "custom":
        result.append(formula)
    if _attr(component, "state.config.recall_activation") == "half":
        result.append("arti/half@1")
    return result


def _half_dependencies(component: Any) -> Sequence[str]:
    metadata = getattr(component, "survival_metadata", None)
    if not isinstance(metadata, Mapping):
        return ()
    reference = metadata.get("ref")
    return (reference,) if isinstance(reference, str) else ()


def _refiner_dependencies(component: Any) -> Sequence[str]:
    result: list[str] = []
    recall = getattr(component, "recall_layer", None)
    registration = get_component_registry().registration_for(recall)
    if registration is not None:
        result.append(registration.reference)
    activation = getattr(component, "activation", None)
    registration = get_component_registry().registration_for(activation)
    if registration is not None:
        result.append(registration.reference)
    return result


def _bank_execution_signature_v2_dependencies(component: Any) -> Sequence[str]:
    result = {
        "arti/gradient-contract@1",
        "arti/query-execution-signature@1",
        "arti/shape-relation@1",
        "arti/tensor-schema@1",
        "arti/terminal-output-abi@1",
        component.program_ref,
        component.query_signature.query_ref,
        component.terminal_adapter_ref,
    }
    if component.local_formula_ref is not None:
        result.add(component.local_formula_ref)
    if component.local_refine_ref is not None:
        result.add(component.local_refine_ref)
    return tuple(sorted(result))


def _federal_recall_v2_dependencies(component: Any) -> Sequence[str]:
    result = {
        "arti/bank-execution-signature@2",
        "arti/sealed-bank-query@1",
        "arti/terminal-output-abi@1",
    }
    for bank_id in component.banks:
        result.update(
            _bank_execution_signature_v2_dependencies(
                component.banks[bank_id].signature
            )
        )
    return tuple(sorted(result))


def _target_bank_updater_config(component: Any) -> Mapping[str, Any]:
    policy = _attr(component, "policy")
    return {
        "hidden_dim": _attr(component, "hidden_dim"),
        "slots": _attr(component, "slots"),
        "workspace_dim": _attr(component, "workspace_dim"),
        "private_slots": _attr(component, "private_slots"),
        "query_seed": _attr(component, "query_seed"),
        "target_coupling": _attr(component, "target_coupling"),
        "epsilon": _attr(component, "epsilon"),
        "policy": {
            "budget": _attr(policy, "budget"),
            "stop": _attr(policy, "stop"),
            "exposure_schedule": _attr(policy, "exposure_schedule"),
        },
    }


def _target_bank_updater_dependencies(component: Any) -> Sequence[str]:
    result = [
        "arti/write-refine-policy@1",
        "arti/refine-budget@1",
    ]
    if _attr(component, "policy.stop") is not None:
        result.append("arti/refine-stop@1")
    formula = getattr(component, "formula", None)
    registration = get_component_registry().registration_for(formula)
    if registration is not None:
        result.append(registration.reference)
    return result


def _route_stack_config(component: Any) -> Mapping[str, Any]:
    items = []
    for item in _attr(component, "items"):
        spec = component_spec(item)
        items.append(
            {
                "reference": spec.reference,
                "config_fingerprint": spec.config_fingerprint,
            }
        )
    return {
        "schema_version": _attr(component, "schema_version"),
        "axis": _attr(component, "axis"),
        "count": len(_attr(component, "items")),
        "items": items,
    }


def _route_stack_dependencies(component: Any) -> Sequence[str]:
    result = []
    for item in _attr(component, "items"):
        registration = get_component_registry().registration_for(item)
        if registration is None:
            raise ValueError("RecallRouteStack contains an unregistered item")
        result.append(registration.reference)
    return result


def _context_config(component: Any) -> Mapping[str, Any]:
    if component.__class__.__name__ == "FrameContext":
        return {
            "mode": _attr(component, "mode"),
            "coord": _tensor_descriptor(_attr(component, "coord"))
            if isinstance(_attr(component, "coord"), Tensor)
            else None,
            "observer_coord": _tensor_descriptor(_attr(component, "observer_coord"))
            if isinstance(_attr(component, "observer_coord"), Tensor)
            else None,
            "frame_operators": _tensor_descriptor(_attr(component, "frame_operators"))
            if isinstance(_attr(component, "frame_operators"), Tensor)
            else None,
            "rotation_tolerance": _attr(component, "rotation_tolerance"),
        }
    return {
        "valid_mask": _tensor_descriptor(_attr(component, "valid_mask"))
        if isinstance(_attr(component, "valid_mask"), Tensor)
        else None,
        "visibility": _tensor_descriptor(_attr(component, "visibility"))
        if isinstance(_attr(component, "visibility"), Tensor)
        else None,
        "frame_present": _attr(component, "frame") is not None,
    }


def _recall_state_config(component: Any) -> Mapping[str, Any]:
    value = _attr(component, "value")
    if not isinstance(value, Tensor) or value.ndim not in {2, 3}:
        raise ValueError("RecallState value must have rank 2 or 3")
    return {
        "rank": value.ndim,
        "slots": int(value.shape[-2]),
        "hidden_dim": int(value.shape[-1]),
        "dtype": str(value.dtype),
        "schema_version": _attr(component, "schema_version"),
        "bound": _attr(component, "contract_fingerprint") is not None,
    }


def _updater_config(component: Any) -> Mapping[str, Any]:
    return _fields(
        "hidden_dim",
        "slots",
        "workspace_dim",
        "depth",
        "interface_slots",
        "recall_slots",
        "recall_group_topk",
        "recall_route_exploration",
        "recall_steps",
        "recall_min_steps",
        "recall_tolerance",
    )(component)


def _normalised_updater_config(component: Any) -> Mapping[str, Any]:
    return _fields("hidden_dim", "slots", "workspace_dim", "factors", "epsilon")(component)


def _affine_updater_config(component: Any) -> Mapping[str, Any]:
    return _fields("hidden_dim", "slots", "workspace_dim")(component)


def _recall_runtime_config(component: Any) -> Mapping[str, Any]:
    return {
        "slots": _attr(component, "slots"),
        "hidden_dim": _attr(component, "hidden_dim"),
        "bank_layout": "values-only",
        "state_ref": "arti/recall-state@1",
        "recall_state_schema_version": _attr(component, "contract.state_schema_version"),
        "runtime_contract_schema_version": _attr(component, "contract.schema_version"),
    }


def _recall_runtime_dependencies(component: Any) -> Sequence[str]:
    result = ["arti/recall-state@1"]
    registry = get_component_registry()
    for child in (getattr(component, "reader", None), getattr(component, "updater", None)):
        registration = registry.registration_for(child)
        if registration is not None:
            result.append(registration.reference)
    return tuple(dict.fromkeys(result))


def _resident_formula_operation_config(component: Any) -> Mapping[str, Any]:
    route = _attr(component, "route")
    factors = _attr(component, "factors")
    return {
        "refine_steps": _attr(component, "resident_refine_steps"),
        "route_estimator": _attr(route, "estimator"),
        "route_shape": list(_attr(route, "weights").shape),
        "factor_shape": None if factors is None else list(factors.shape),
        "factor_dtype": None if factors is None else str(factors.dtype),
    }


def _topology_resident_operation_config(component: Any) -> Mapping[str, Any]:
    return {
        **_resident_formula_operation_config(component),
        "fold_ref": component_ref(_attr(component, "fold")),
        "unfold_ref": component_ref(_attr(component, "unfold")),
    }


def _fixed_page_refs_config(component: Any) -> Mapping[str, Any]:
    return {
        name: _tensor_descriptor(_attr(component, name))
        for name in (
            "logical_slot",
            "page_id",
            "offset",
            "expected_generation",
            "read_mask",
            "write_mask",
            "commit_mask",
        )
    }


def _hot_page_pool_config(component: Any) -> Mapping[str, Any]:
    return {
        "lifecycle_state": component.lifecycle_state,
        "value": _tensor_descriptor(component.value),
        "validity": _tensor_descriptor(component.validity),
        "generation": _tensor_descriptor(component.generation),
        "version": _tensor_descriptor(component.version),
    }


def _bound_hot_page_pool_config(component: Any) -> Mapping[str, Any]:
    receipt = component.pointer_layout_receipt()
    return {
        "lifecycle_state": component.lifecycle_state,
        "bucket_ref": component_ref(component.bucket),
        "refs_ref": component_ref(component.refs),
        "pool_ref": component_ref(component.pool),
        "shape": list(receipt.shape),
        "stride": list(receipt.stride),
        "dtype": receipt.dtype,
        "device": receipt.device,
    }


def _runtime_checkpoint_receipt_config(component: Any) -> Mapping[str, Any]:
    return {
        "artifact_sha256": component.artifact_sha256,
        "manifest_fingerprint": component.manifest_fingerprint,
        "root_fingerprint": component.root_fingerprint,
        "root_epoch": component.root_epoch,
        "page_count": component.page_count,
        "includes_resident_pool": component.includes_resident_pool,
    }


def _restored_runtime_checkpoint_config(component: Any) -> Mapping[str, Any]:
    return {
        "manifest_fingerprint": component.manifest_fingerprint,
        "artifact_sha256": component.artifact_sha256,
        "resident": component.resident is not None,
        "binding_count": len(component.bindings),
        "binding_fingerprints": [binding.fingerprint for binding in component.bindings],
    }


def _recall_branch_batch_config(component: Any) -> Mapping[str, Any]:
    return {
        "schema_version": _attr(component, "schema_version"),
        "source_ref": _attr(component, "source_ref"),
        "source_config_fingerprint": _attr(component, "source_config_fingerprint"),
        "layout_fingerprint": _attr(component, "layout_fingerprint"),
        "max_k": _attr(component, "max_k"),
        "formula_beam_width": _attr(component, "formula_beam_width"),
        "active_k": [
            int(value)
            for value in _attr(component, "active_k").detach().to("cpu").tolist()
        ],
        "requested_active_k": [
            int(value)
            for value in _attr(component, "requested_active_k")
            .detach()
            .to("cpu")
            .tolist()
        ],
        "active_partition_count": [
            int(value)
            for value in component.active_partition_count()
            .detach()
            .to("cpu")
            .tolist()
        ],
        "active_partition_mask": component.active_partition_mask()
        .detach()
        .to("cpu")
        .tolist(),
        "active_branch_count_by_partition": component.active_branch_count_by_partition()
        .detach()
        .to("cpu")
        .tolist(),
        "source_topk": _attr(component, "source_topk"),
        "group_count": _attr(component, "group_count"),
        "group_size": _attr(component, "group_size"),
        "routing_normalizer": _attr(component, "routing_normalizer"),
        "partition_names": list(_attr(component, "partition_names")),
        "partition_ranges": [
            list(value) for value in _attr(component, "partition_ranges")
        ],
        "partition_member_fingerprints": list(
            _attr(component, "partition_member_fingerprints")
        ),
        "partition_layout_fingerprint": _attr(
            component, "partition_layout_fingerprint"
        ),
        "candidate_partition_index": _attr(component, "candidate_partition_index")
        .detach()
        .to("cpu")
        .tolist(),
        "value_composition": _attr(component, "value_composition"),
        "candidate_policy": _attr(component, "candidate_policy"),
        "partition_quota": list(_attr(component, "partition_quota")),
        "partition_coherence": _attr(component, "partition_coherence"),
        "factor_count": _attr(component, "factor_count"),
        "formula_ref": _attr(component, "formula_ref"),
        "formula_config_fingerprint": _attr(
            component, "formula_config_fingerprint"
        ),
        "topology_lineage": list(_attr(component, "topology_lineage")),
    }


def _batched_refine_result_config(component: Any) -> Mapping[str, Any]:
    candidate = component.candidates
    candidate_config = _recall_branch_batch_config(candidate)
    return {
        "schema_version": _attr(component, "schema_version"),
        "candidate_ref": component_ref(candidate),
        "candidate_config": candidate_config,
        "candidate_config_fingerprint": _sha256_json(candidate_config),
        "plan_ref": component.plan_ref,
        "plan_config_fingerprint": component.plan_config_fingerprint,
        "execution_layout": component.execution_layout,
        "operation_wrapper_ref": (
            None
            if component.operation_ref is None
            else "arti/batched-refine-operation@1"
        ),
        "operation_ref": component.operation_ref,
        "formula_route_fingerprint": component.formula_route_fingerprint,
        "topology_refs": list(component.topology_refs),
        "topology_contract_fingerprints": list(
            component.topology_contract_fingerprints
        ),
        "branch_policy_fingerprint": component.branch_policy_fingerprint,
        "execution_rng_fingerprint": component.execution_rng_fingerprint,
        "execution_rng_stream_key": component.execution_rng_stream_key,
        "execution_rng_domains": list(component.execution_rng_domains),
        "max_k": candidate.max_k,
    }


def _batched_refine_result_dependencies(component: Any) -> Sequence[str]:
    candidate = component.candidates
    return tuple(
        reference
        for reference in (
            component_ref(candidate),
            candidate.source_ref,
            candidate.formula_ref,
            component.plan_ref,
            (
                None
                if component.operation_ref is None
                else "arti/batched-refine-operation@1"
            ),
            component.operation_ref,
            *component.topology_refs,
            (
                None
                if component.branch_policy_fingerprint is None
                else "arti/branch-refine-policy@1"
            ),
        )
        if reference is not None
    )


class ComponentRegistry:
    """Thread-safe exact registry for component contracts and aliases."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._by_reference: dict[str, ComponentRegistration] = {}
        self._by_alias: dict[str, ComponentRegistration] = {}

    def register(
        self,
        reference: str,
        *,
        component_type: type[Any],
        lifecycle: ComponentLifecycle,
        variant: str = "default",
        aliases: Sequence[str] = (),
        deprecated_aliases: Sequence[str] = (),
        config_schema_version: int = 1,
        state_schema_version: int = 1,
        constructible: bool = True,
        artifact_policy: ArtifactPolicy = "portable",
        factory: Factory | None = None,
        config_builder: ConfigBuilder | None = None,
        dependency_builder: DependencyBuilder | None = None,
        capabilities: Sequence[str] = (),
    ) -> ComponentRegistration:
        identity = ComponentRef.parse(reference)
        if type(constructible) is not bool:
            raise TypeError("constructible must be boolean")
        if not constructible and factory is not None:
            raise ValueError("a non-constructible component cannot expose a factory")
        if artifact_policy not in _ARTIFACT_POLICIES:
            raise ValueError(f"unsupported artifact policy: {artifact_policy!r}")
        registration = ComponentRegistration(
            identity=identity,
            component_type=component_type,
            lifecycle=lifecycle,
            variant=variant,
            aliases=tuple(aliases),
            deprecated_aliases=tuple(deprecated_aliases),
            config_schema_version=config_schema_version,
            state_schema_version=state_schema_version,
            constructible=constructible,
            artifact_policy=artifact_policy,
            factory=(
                None
                if not constructible
                else component_type if factory is None else factory
            ),
            config_builder=config_builder,
            dependency_builder=dependency_builder,
            capabilities=tuple(capabilities),
        )
        with self._lock:
            if reference in self._by_reference or reference in self._by_alias:
                raise DuplicateComponentError(f"component reference is already registered: {reference}")
            aliases_to_add = set(registration.aliases) | set(registration.deprecated_aliases)
            for alias in {identity.mechanism_id, identity.name, component_type.__name__}:
                if alias not in self._by_alias and alias not in self._by_reference:
                    aliases_to_add.add(alias)
            if reference in aliases_to_add:
                raise DuplicateComponentError(
                    f"component reference cannot also be an alias: {reference}"
                )
            for alias in aliases_to_add:
                if not isinstance(alias, str) or not alias:
                    raise InvalidComponentRefError("component aliases must be non-empty strings")
            if any(alias in self._by_reference or alias in self._by_alias for alias in aliases_to_add):
                raise DuplicateComponentError(f"component alias is already registered: {sorted(aliases_to_add)}")
            self._by_reference[reference] = registration
            for alias in aliases_to_add:
                self._by_alias[alias] = registration
        return registration

    def registration_for_reference(self, reference: str) -> ComponentRegistration:
        try:
            with self._lock:
                return self._by_reference[reference]
        except KeyError as error:
            raise UnknownComponentError(f"unknown component reference: {reference!r}") from error

    def resolve_registration(self, reference_or_alias: str) -> ComponentRegistration:
        with self._lock:
            registration = self._by_reference.get(reference_or_alias) or self._by_alias.get(reference_or_alias)
        if registration is None:
            raise UnknownComponentError(f"unknown component reference or alias: {reference_or_alias!r}")
        return registration

    def registration_for(self, value: Any) -> ComponentRegistration | None:
        if value is None:
            return None
        with self._lock:
            preferred_reference = getattr(value, "_component_reference", None)
            if isinstance(preferred_reference, str):
                preferred = self._by_reference.get(preferred_reference)
                if preferred is not None and isinstance(value, preferred.component_type):
                    return preferred
            for registration in self._by_reference.values():
                if type(value) is registration.component_type:
                    return registration
            for base in type(value).__mro__[1:]:
                for registration in self._by_reference.values():
                    if base is registration.component_type:
                        return registration
        return None

    def registrations(self) -> tuple[ComponentRegistration, ...]:
        with self._lock:
            return tuple(sorted(self._by_reference.values(), key=lambda item: item.reference))

    def catalog(self) -> tuple[dict[str, Any], ...]:
        """Return the deterministic public identity catalog."""

        return tuple(registration.to_dict() for registration in self.registrations())

    def resolve(self, reference_or_alias: str, **kwargs: Any) -> Any:
        registration = self.resolve_registration(reference_or_alias)
        if registration.factory is None:
            raise ComponentRegistryError(
                f"component {registration.reference!r} is runtime-only and cannot be constructed"
            )
        return registration.factory(**kwargs)


_DEFAULT_REGISTRY: ComponentRegistry | None = None


def _build_default_registry() -> ComponentRegistry:
    registry = ComponentRegistry()
    from ._recall_state import (
        AffineRecallValueUpdater,
        NormalizedDeltaRecallValueUpdater,
        RecallState,
        RECALL_STATE_SCHEMA_VERSION,
        RecallValueUpdater,
        StackedRecallValueUpdater,
    )
    from .context import FrameContext, TensorContext
    from .emission import EmissionRouter, EmissionRouterConfig
    from .arti_layer import ARTILayer
    from .layers import (
        ARTIDynamicStateLayer,
        ARTILatentRecallField,
        ARTILatentTensorLayer,
        ARTILayer as LegacyARTILayer,
        ARTIPhaseMixer,
        ARTIVirtualInterfaceMixer,
    )
    from .membrane import MembraneVisibilityRouter
    from .aggregate import ReunionAggregate, SoftFoldAggregate
    from .adaptive_pulse import AdaptivePulse
    from .batched_refine import (
        BatchedRefineOperation,
        BatchedRefinePlan,
        BatchedRefineResult,
        BranchRefinePolicy,
        ExecutionRNGPlan,
        RecallBranchBatch,
        RecallFormulaBranchBatch,
    )
    from .branch_refine import (
        BatchedRefineExecutor,
        ExecutionContextReceipt,
        ResidentBranchCommitReceipt,
        ResidentBranchDecision,
        ResidentBranchRun,
        ResidentBranchScoreReceipt,
    )
    from .formula_attention import (
        FactorInterventionPolicy,
        FormulaAttention,
        MagnitudeInterventionPolicy,
        ScaleShiftFormula,
        SelectiveCompute,
        StableTopKIntervention,
    )
    from .formula_fabric import (
        BankFormulaRouteSource,
        FormulaCommitBlend,
        FormulaFabric,
        FormulaFabricCompute,
        RoutedFormulaFabricCompute,
        IterativeRoutedFormulaFabricCompute,
    )
    from .formula_v2 import (
        AddAtom,
        ContractAtom,
        FORMULA_EXECUTION_PLAN_V1_SCHEMA_REF,
        FORMULA_EXECUTION_PLAN_V1_SCHEMA_VERSION,
        FormulaExecutionPlanV2,
        FormulaFabricV2,
        GatherAtom,
        PermuteAtom,
        ReduceAtom,
        ReshapeAtom,
        ScaleAtom,
        ScatterAtom,
    )
    from .formula_learning import FormulaOperandBank
    from .gpu_resident import (
        BoundHotPagePool,
        CUDAActivityReceipt,
        CapturedHotStep,
        FixedPageRefs,
        FixedResidentBucket,
        FormulaResidentOperation,
        HotPagePool,
        ResidentLatencyReceipt,
        TopologyFormulaResidentOperation,
    )
    from .runtime_checkpoint import (
        RestoredRuntimeCheckpoint,
        RuntimeCheckpointReceipt,
    )
    from .selective_recall import SelectiveRecallKernel
    from .nn import Fold, FusionPulse, Half, LearnedPulse, Recall, RecallRefiner, UnFold
    from .pulse import PulseCompressor
    from .reversible_topology import (
        FOLD_RECORD_SCHEMA_VERSION,
        FOLD_STATE_SCHEMA_VERSION,
        FixedTopologyPolicy,
        FoldRecord,
        FoldedTensor,
        InverseTopologyContract,
        ReversibleTopology,
        TopologyFold,
        TopologyUnFold,
    )
    from .topology import (
        BankFormulaTopologyPolicy,
        FixedTopologyQuery,
        LearnedTopologyPolicy,
        PairwiseRankTopologySurrogate,
        SoftTopKTopologySurrogate,
        StablePriorityPartition,
        TopologyAction,
        TopologyProposal,
        TopologyFormulaLock,
        TopologyOperandBank,
        TopologyPriorityFormula,
    )
    from .typed_topology import (
        TypedBankFormulaTopologyPolicy,
        TypedTopologyOperandBank,
        TypedTopologyPriorityFormula,
    )
    from .tensor_schema import GradientContract, ShapeRelation, TensorSchema
    from .bank_query import (
        LinearBankQuery,
        QueryExecutionSignature,
        SealedBankQuery,
    )
    from .bank_local_program import (
        BankLocalFormulaAction,
        BankLocalFormulaProgram,
        BankLocalTerminalAction,
        DetachedBankLocalProgramTraining,
        DetachedBankLocalRollout,
        ExactBankLocalProgramTraining,
        ValueTerminalAdapter,
    )
    from .terminal_abi import (
        BankExecutionSignature,
        BankExecutionSignatureV2,
        TerminalOutputABI,
    )
    from .federal_recall import BankLocalRefinePolicy, FederalRecall, FederalRecallV2
    from .recall_refine import (
        AdaptiveRefinePolicy,
        RecallRoutePlan,
        RecallRouteStack,
        RecallTraceV3,
        RefineBudget,
        RefinePolicy,
        RefineStop,
    )
    from .refine_training import RefineRollout, RefineStepTraining
    from .refine_exit import FormulaRefineExit, RefineExitControl, RefineExitRequest
    from .refine_exit_training import RefineExitCurve, RefineExitTraining
    from .target_bank import TargetBankUpdater, WriteRefinePolicy
    from .objective_bank import ObjectiveExposureBank
    from .objective_formula import ObjectiveFormulaFabricCompute
    from .recall_runtime import RecallRuntime
    from .vnext_contracts import (
        PULSE_STAGE_GRAPH_SCHEMA_VERSION,
        PULSE_STAGE_SCHEMA_VERSION,
        PulseStageGraph,
        PulseStageSpec,
    )
    from .vnext_pipeline import PulseExecutor
    from .observation import (
        AdaptiveObservation,
        FixedObservationPolicy,
        FourierShiftObservationOperator,
        IdentityObservationOperator,
        LearnedObservationPolicy,
        StateAffineObservationOperator,
    )
    from .observation_bank import (
        BankConditionedObservationPolicy,
        FixedObservationQuery,
        ObservationOperandBank,
        ObservationTrajectoryFormula,
    )
    from .tensor_operation import (
        OperableTensorPort,
        PortSnapshot,
        PortSpec,
        ReaderRefineSchedule,
        SharedCanvas,
        SharedCanvasFold,
        TensorEditInstruction,
        TensorEditFormula,
        TensorEditResult,
        TensorEditSurrogate,
        TensorInvocation,
        TensorInvocationResult,
        TensorOperation,
        TensorOperationBank,
        TensorOperationDecision,
        TensorOperationFieldSpec,
        TensorOperationLoop,
        TensorOperationQuery,
        TensorOperationResult,
        TensorOperationSchedule,
        TensorOperationSelector,
        TensorOperationStepResult,
        TensorOperationStopPolicy,
        TensorOperationTrace,
    )

    def add(reference: str, component_type: type[Any], **kwargs: Any) -> None:
        registry.register(reference, component_type=component_type, **kwargs)

    def contextual_half_factory(**kwargs: Any) -> Half:
        requested_mode = kwargs.get("context_mode", "contextual")
        if requested_mode != "contextual":
            raise ValueError("arti/half@2 requires context_mode='contextual'")
        kwargs["context_mode"] = "contextual"
        return Half(**kwargs)

    def scalar_half_factory(**kwargs: Any) -> Half:
        requested_mode = kwargs.get("context_mode", "none")
        if requested_mode != "none":
            raise ValueError("arti/half@1 requires context_mode='none'")
        kwargs["context_mode"] = "none"
        return Half(**kwargs)

    def recall_factory(
        normalizer: str,
        kwargs: dict[str, Any],
        *,
        breadth_mode: str,
    ) -> Recall:
        requested = kwargs.pop("routing_normalizer", normalizer)
        if requested != normalizer:
            version = 3 if normalizer == "per_bank" else 2
            raise ValueError(
                f"arti/recall@{version} requires routing_normalizer={normalizer!r}"
            )
        kwargs.pop("formula_origin", None)
        kwargs.pop("formula_portable", None)
        kwargs.pop("value_composition", None)
        requested_breadth_mode = kwargs.pop("breadth_mode", breadth_mode)
        if requested_breadth_mode != breadth_mode:
            raise ValueError(
                f"this Recall reference requires breadth_mode={breadth_mode!r}"
            )
        kwargs["breadth_mode"] = breadth_mode
        if breadth_mode == "mixed":
            kwargs.pop("breadth", None)
            kwargs.pop("breadth_aggregation", None)
            kwargs["breadth"] = 1
        expert_names = tuple(kwargs.pop("expert_names", ()))
        expert_ranges = tuple(
            tuple(int(item) for item in value)
            for value in kwargs.pop("expert_route_ranges", ())
        )
        expert_member_fingerprints = tuple(
            str(value) for value in kwargs.pop("expert_member_fingerprints", ())
        )
        expert_weights = tuple(float(value) for value in kwargs.pop("expert_weights", ()))
        expert_influences = tuple(
            float(value) for value in kwargs.pop("expert_influences", ())
        )
        module = Recall(routing_normalizer=normalizer, **kwargs)
        if (
            expert_names
            or expert_ranges
            or expert_member_fingerprints
            or expert_weights
            or expert_influences
        ):
            if not expert_names or len(expert_names) != len(expert_ranges):
                raise ValueError("Recall expert assembly config is incomplete")
            module.state.recall.configure_expert_routes(
                expert_names,
                expert_ranges,
                member_fingerprints=(
                    expert_member_fingerprints or None
                ),
            )
            if expert_weights:
                module.state.recall.set_expert_weights(expert_weights)
            if expert_influences:
                module.state.recall.set_expert_influences(expert_influences)
        return module

    def global_recall_factory(**kwargs: Any) -> Recall:
        return recall_factory("global", kwargs, breadth_mode="mixed")

    def per_bank_recall_factory(**kwargs: Any) -> Recall:
        return recall_factory("per_bank", kwargs, breadth_mode="mixed")

    def wide_recall_factory(**kwargs: Any) -> Recall:
        normalizer = kwargs.pop("routing_normalizer", "global")
        if normalizer not in {"global", "per_bank"}:
            raise ValueError("arti/recall@4 requires a supported routing_normalizer")
        return recall_factory(normalizer, kwargs, breadth_mode="independent")

    def coupled_target_bank_updater_factory(**kwargs: Any) -> TargetBankUpdater:
        requested = kwargs.get("target_coupling", "required_after_bootstrap")
        if requested != "required_after_bootstrap":
            raise ValueError(
                "arti/target-bank-updater@2 requires "
                "target_coupling='required_after_bootstrap'"
            )
        kwargs["target_coupling"] = "required_after_bootstrap"
        return TargetBankUpdater(**kwargs)

    stable = "stable"
    add(
        "arti/half@1",
        Half,
        lifecycle=stable,
        capabilities=("pulse.stage.half",),
        config_schema_version=2,
        factory=scalar_half_factory,
        config_builder=_half_config,
        dependency_builder=_half_dependencies,
    )
    add(
        "arti/half@2",
        Half,
        lifecycle=stable,
        capabilities=("pulse.stage.half",),
        variant="contextual",
        config_schema_version=3,
        factory=contextual_half_factory,
        config_builder=_half_config,
        dependency_builder=_half_dependencies,
    )
    add("arti/fold@1", Fold, lifecycle=stable, config_builder=_fold_config)
    add("arti/unfold@1", UnFold, lifecycle=stable, config_builder=_unfold_config)
    add(
        "arti/fixed-topology-policy@1",
        FixedTopologyPolicy,
        lifecycle=stable,
        variant="fixed-index",
        config_builder=_fixed_topology_policy_config,
    )
    add(
        "arti/topology-action@1",
        TopologyAction,
        lifecycle=stable,
        variant="priority-operands",
        config_builder=lambda component: {
            "shape": list(_attr(component, "priority.shape")),
            "dtype": str(_attr(component, "priority.dtype")),
        },
    )
    add(
        "arti/topology-proposal@1",
        TopologyProposal,
        lifecycle=stable,
        variant="continuous-priority-proposal",
        config_builder=lambda component: {
            "action_ref": component_ref(_attr(component, "action")),
        },
        dependency_builder=lambda _component: ("arti/topology-action@1",),
    )
    add(
        "arti/stable-priority-partition@1",
        StablePriorityPartition,
        lifecycle=stable,
        variant="valid-first-stable-sort",
    )
    add(
        "arti/topology-surrogate@1",
        SoftTopKTopologySurrogate,
        lifecycle=stable,
        variant="soft-top-k-vjp",
        config_builder=_topology_surrogate_config,
    )
    add(
        "arti/topology-surrogate@2",
        PairwiseRankTopologySurrogate,
        lifecycle=stable,
        variant="pairwise-soft-rank-position-vjp",
        config_builder=_topology_surrogate_config,
    )
    add(
        "arti/learned-topology-policy@1",
        LearnedTopologyPolicy,
        lifecycle=stable,
        variant="equivariant-scorer",
        config_builder=_learned_topology_policy_config,
        dependency_builder=_learned_topology_policy_dependencies,
    )
    add(
        "arti/topology-priority-formula@1",
        TopologyPriorityFormula,
        lifecycle=stable,
        variant="affine-priority",
        config_builder=_topology_formula_config,
    )
    add(
        "arti/topology-priority-formula@2",
        TypedTopologyPriorityFormula,
        lifecycle=stable,
        variant="typed-affine-priority",
        config_builder=_topology_formula_config,
    )
    add(
        "arti/topology-formula-lock@1",
        TopologyFormulaLock,
        lifecycle=stable,
        variant="formula-binding",
    )
    add(
        "arti/fixed-topology-query@1",
        FixedTopologyQuery,
        lifecycle=stable,
        variant="deterministic-projection",
        config_builder=_fixed_topology_query_config,
    )
    add(
        "arti/topology-operand-bank@1",
        TopologyOperandBank,
        lifecycle=stable,
        variant="fixed-address-values",
        config_builder=_topology_operand_bank_config,
    )
    add(
        "arti/topology-operand-bank@2",
        TypedTopologyOperandBank,
        lifecycle=stable,
        variant="typed-fixed-address-values",
        config_builder=_topology_operand_bank_config,
    )
    add(
        "arti/bank-formula-topology-policy@1",
        BankFormulaTopologyPolicy,
        lifecycle=stable,
        variant="fixed-query-bank-formula",
        config_builder=_bank_formula_topology_policy_config,
        dependency_builder=_bank_formula_topology_policy_dependencies,
    )
    add(
        "arti/bank-formula-topology-policy@2",
        TypedBankFormulaTopologyPolicy,
        lifecycle=stable,
        variant="typed-fixed-query-bank-formula",
        config_builder=_bank_formula_topology_policy_config,
        dependency_builder=_bank_formula_topology_policy_dependencies,
    )
    add(
        "arti/reversible-topology@1",
        ReversibleTopology,
        lifecycle=stable,
        variant="permutation-partition",
        config_builder=_reversible_topology_config,
        dependency_builder=_reversible_topology_dependencies,
    )
    add(
        "arti/inverse-topology-contract@1",
        InverseTopologyContract,
        lifecycle=stable,
        variant="recorded-permutation-inverse",
        config_builder=_inverse_topology_contract_config,
    )
    add(
        "arti/fold@2",
        TopologyFold,
        lifecycle=stable,
        capabilities=("pulse.stage.fold",),
        variant="reversible-forward",
        config_schema_version=3,
        config_builder=_topology_fold_config,
        dependency_builder=_topology_fold_dependencies,
    )
    add(
        "arti/unfold@2",
        TopologyUnFold,
        lifecycle=stable,
        capabilities=("pulse.stage.unfold",),
        variant="recorded-inverse",
        config_schema_version=2,
        config_builder=_topology_unfold_config,
        dependency_builder=_topology_unfold_dependencies,
    )
    add(
        "arti/fold-record@1",
        FoldRecord,
        lifecycle=stable,
        variant="runtime-record",
        state_schema_version=FOLD_RECORD_SCHEMA_VERSION,
        config_builder=_fold_record_config,
    )
    add(
        "arti/fold-state@1",
        FoldedTensor,
        lifecycle=stable,
        variant="runtime-state",
        state_schema_version=FOLD_STATE_SCHEMA_VERSION,
        config_builder=_fold_state_config,
        dependency_builder=lambda _component: ("arti/fold-record@1",),
    )
    add(
        "arti/pulse-stage@1",
        PulseStageSpec,
        lifecycle=stable,
        variant="ordered-stage-spec",
        state_schema_version=PULSE_STAGE_SCHEMA_VERSION,
        config_builder=lambda component: component.to_dict(),
        dependency_builder=lambda component: (
            (component.component_ref,)
            if component.mode.value == "enabled"
            else ()
        ),
    )
    add(
        "arti/fixed-observation-policy@1",
        FixedObservationPolicy,
        lifecycle=stable,
        variant="bounded-fixed-trajectory",
        config_builder=lambda component: {
            "max_observations": component.max_observations,
            "active_count": component.active_count,
            "state_dim": component.state_dim,
            "learnable": isinstance(component.states, torch.nn.Parameter),
        },
    )
    add(
        "arti/identity-observation-operator@1",
        IdentityObservationOperator,
        lifecycle=stable,
        variant="identity-substrate-observation",
    )
    add(
        "arti/fourier-observation-operator@1",
        FourierShiftObservationOperator,
        lifecycle=stable,
        variant="circular-subpixel-phase-shift",
        config_builder=lambda component: {
            "spatial_shape": list(component.spatial_shape),
            "state_mode": component.state_mode,
            "direction_epsilon": component.direction_epsilon,
            "boundary": component.boundary,
            "compile_policy": component.compile_policy,
        },
    )
    add(
        "arti/learned-observation-policy@1",
        LearnedObservationPolicy,
        lifecycle=stable,
        variant="input-conditioned-bounded-trajectory",
        config_builder=lambda component: {
            "input_dim": component.input_dim,
            "state_dim": component.state_dim,
            "hidden_dim": component.hidden_dim,
            "max_observations": component.max_observations,
            "min_observations": component.min_observations,
            "stop_threshold": component.stop_threshold,
            "temperature": component.temperature,
        },
    )
    add(
        "arti/state-affine-observation-operator@1",
        StateAffineObservationOperator,
        lifecycle=stable,
        variant="bounded-state-conditioned-feature-frame",
        config_builder=lambda component: {
            "dim": component.dim,
            "state_dim": component.state_dim,
            "scale": component.scale,
        },
    )
    add(
        "arti/fixed-observation-query@1",
        FixedObservationQuery,
        lifecycle=stable,
        variant="deterministic-input-trajectory-query",
        config_builder=lambda component: component.observation_query_contract(),
    )
    add(
        "arti/observation-operand-bank@1",
        ObservationOperandBank,
        lifecycle=stable,
        variant="typed-fixed-address-observation-values",
        config_builder=lambda component: {
            **component.structure_contract,
            "query": "fixed-address",
            "state": "trainable-values",
        },
    )
    add(
        "arti/observation-trajectory-formula@1",
        ObservationTrajectoryFormula,
        lifecycle=stable,
        variant="typed-state-and-continuation",
        config_builder=lambda component: {
            "state_dim": component.state_dim,
            "factor_dim": component.factor_dim,
            "state_scale": component.state_scale,
            "continuation_scale": component.continuation_scale,
        },
    )
    add(
        "arti/bank-observation-policy@1",
        BankConditionedObservationPolicy,
        lifecycle=stable,
        variant="fixed-query-typed-bank-trajectory",
        config_builder=lambda component: {
            "input_dim": component.input_dim,
            "state_dim": component.state_dim,
            "max_observations": component.max_observations,
            "min_observations": component.min_observations,
            "key_dim": component.key_dim,
            "query_seed": component.query_seed,
            "stop_threshold": component.stop_threshold,
            "temperature": component.temperature,
            "ordered_banks": [
                bank.structure_contract for bank in component.banks
            ],
            "bank_weights": component.bank_weights.detach().cpu().tolist(),
            "normalization": "per-bank",
            "merge": "explicit-weighted-sum",
        },
        dependency_builder=lambda component: (
            component_ref(component.query),
            component_ref(component.formula),
            *(component_ref(bank) for bank in component.banks),
        ),
    )
    add(
        "arti/adaptive-observation@1",
        AdaptiveObservation,
        lifecycle=stable,
        variant="bounded-original-substrate-trajectory",
        capabilities=("pulse.stage.observation",),
        config_builder=lambda component: {
            "policy": component_spec(component.policy).to_dict(),
            "operator": component_spec(component.operator).to_dict(),
            "executor": component.executor,
            "limits": dict(component.limits.__dict__),
        },
        dependency_builder=lambda component: (
            component_ref(component.policy),
            component_ref(component.operator),
        ),
    )
    add(
        "arti/scale-shift-formula@1",
        ScaleShiftFormula,
        lifecycle=stable,
        variant="per-feature-next-state",
        capabilities=("selective.compute.kernel",),
        config_builder=lambda component: {"dim": component.dim},
    )
    add(
        "arti/magnitude-intervention-policy@1",
        MagnitudeInterventionPolicy,
        lifecycle=stable,
        variant="feature-strength-priority",
        capabilities=("formula.intervention.policy",),
    )
    add(
        "arti/factor-intervention-policy@1",
        FactorInterventionPolicy,
        lifecycle=stable,
        variant="typed-factor-priority",
        capabilities=("formula.intervention.policy",),
        config_builder=lambda component: {"factor_index": component.factor_index},
    )
    add(
        "arti/stable-topk-intervention@1",
        StableTopKIntervention,
        lifecycle=stable,
        variant="bounded-stable-support-selection",
        capabilities=("formula.intervention.operator",),
        config_builder=lambda component: {
            "max_interventions": component.max_interventions,
        },
    )
    add(
        "arti/formula-attention@1",
        FormulaAttention,
        lifecycle=stable,
        variant="formula-intervention-support",
        capabilities=("pulse.stage.intervention",),
        config_builder=lambda component: {
            "policy": component_spec(component.policy).to_dict(),
            "operator": component_spec(component.operator).to_dict(),
        },
        dependency_builder=lambda component: (
            component_ref(component.policy),
            component_ref(component.operator),
        ),
    )
    add(
        "arti/formula-fabric@1",
        FormulaFabric,
        lifecycle=stable,
        variant="bounded-hard-ssa-formula-executor",
        config_schema_version=2,
        capabilities=("formula.fabric.executor",),
        config_builder=lambda component: {
            "program": component.program.to_dict(),
            "program_fingerprint": component.program.fingerprint,
            "limits": dict(component.limits.__dict__),
        },
    )
    add(
        "arti/formula-atom-contract@1",
        ContractAtom,
        lifecycle=stable,
        variant="named-axis-parameter-free-contraction",
        capabilities=("formula.fabric.typed-atom",),
        config_builder=lambda component: {
            "left_type": component.left_type.to_dict(),
            "right_type": component.right_type.to_dict(),
            "output_type": component.output_type.to_dict(),
            "reduce_axes": [list(pair) for pair in component.reduce_axes],
            "output_axes": list(component.output_axes),
            "accumulation_dtype": component.accumulation_dtype,
        },
    )
    add(
        "arti/formula-atom-scale@1",
        ScaleAtom,
        lifecycle=stable,
        variant="named-axis-explicit-operand-scale",
        capabilities=("formula.fabric.typed-atom",),
        config_builder=lambda component: {
            "value_type": component.value_type.to_dict(),
            "factor_type": component.factor_type.to_dict(),
            "accumulation_dtype": component.accumulation_dtype,
        },
    )
    add(
        "arti/formula-atom-add@1",
        AddAtom,
        lifecycle=stable,
        variant="typed-binary-add",
        capabilities=("formula.fabric.typed-atom",),
        config_builder=lambda component: {
            "value_type": component.value_type.to_dict(),
            "accumulation_dtype": component.accumulation_dtype,
        },
    )
    add(
        "arti/formula-atom-reduce@1",
        ReduceAtom,
        lifecycle=stable,
        variant="ordered-named-axis-sum",
        capabilities=("formula.fabric.typed-atom",),
        config_builder=lambda component: {
            "value_type": component.value_type.to_dict(),
            "output_type": component.output_type.to_dict(),
            "axis": component.axis,
            "mode": "sum",
            "accumulation_dtype": component.accumulation_dtype,
        },
    )
    add(
        "arti/formula-atom-reshape@1",
        ReshapeAtom,
        lifecycle=stable,
        variant="typed-element-preserving-shape-repartition",
        capabilities=("formula.fabric.shape", "formula.fabric.typed-atom"),
        config_builder=lambda component: {
            "value_type": component.value_type.to_dict(),
            "output_type": component.output_type.to_dict(),
            "output_axes": list(component.output_axes),
            "output_sizes": list(component.output_sizes),
        },
    )
    add(
        "arti/formula-atom-permute@1",
        PermuteAtom,
        lifecycle=stable,
        variant="typed-named-axis-permutation",
        capabilities=("formula.fabric.shape", "formula.fabric.typed-atom"),
        config_builder=lambda component: {
            "value_type": component.value_type.to_dict(),
            "output_type": component.output_type.to_dict(),
            "output_axes": list(component.output_axes),
        },
    )
    add(
        "arti/formula-atom-gather@1",
        GatherAtom,
        lifecycle=stable,
        variant="typed-indexed-workset-selection",
        capabilities=("formula.fabric.topology", "formula.fabric.typed-atom"),
        config_builder=lambda component: {
            "value_type": component.value_type.to_dict(),
            "index_type": component.index_type.to_dict(),
            "output_type": component.output_type.to_dict(),
            "axis": component.axis,
            "index_axis": component.index_axis,
        },
    )
    add(
        "arti/formula-atom-scatter@1",
        ScatterAtom,
        lifecycle=stable,
        variant="typed-indexed-workset-restoration",
        capabilities=("formula.fabric.topology", "formula.fabric.typed-atom"),
        config_builder=lambda component: {
            "base_type": component.base_type.to_dict(),
            "index_type": component.index_type.to_dict(),
            "update_type": component.update_type.to_dict(),
            "output_type": component.output_type.to_dict(),
            "axis": component.axis,
            "index_axis": component.index_axis,
            "mode": component.mode,
        },
    )
    add(
        "arti/formula-fabric@2",
        FormulaFabricV2,
        lifecycle=stable,
        variant="typed-heterogeneous-ssa-formula-executor",
        config_schema_version=2,
        capabilities=("formula.fabric.executor", "formula.fabric.typed-executor"),
        config_builder=lambda component: {
            "program": component.program.to_dict(),
            "program_fingerprint": component.program.fingerprint,
        },
        dependency_builder=lambda component: tuple(
            sorted({item.atom_ref for item in component.program.instructions})
        ),
    )
    add(
        "arti/formula-execution-plan@1",
        FormulaExecutionPlanV2,
        lifecycle=stable,
        variant="typed-static-positional-formula-lowering",
        config_builder=lambda component: {
            "schema_ref": FORMULA_EXECUTION_PLAN_V1_SCHEMA_REF,
            "schema_version": FORMULA_EXECUTION_PLAN_V1_SCHEMA_VERSION,
            "program": component.program.to_dict(),
            "program_fingerprint": component.program_fingerprint,
            "binding_names": list(component.binding_names),
        },
        dependency_builder=lambda component: tuple(
            sorted({item.atom_ref for item in component.program.instructions})
        ),
        capabilities=("formula.fabric.compilable-plan", "formula.fabric.typed-executor"),
    )
    add(
        "arti/formula-operand-bank@1",
        FormulaOperandBank,
        lifecycle=stable,
        variant="joint-typed-formula-operand-candidates",
        capabilities=("formula.fabric.learned-route", "formula.fabric.operand-bank"),
        config_builder=lambda component: {
            "source_ref": component.source_ref,
            "asset_fingerprint": component.asset_fingerprint,
            "bundle_id": component.bundle_id,
            "member_ids": list(component.member_ids),
            "candidate_count": component.candidate_count,
            "key_dim": component.key_dim,
            "key_dtype": str(component.keys.dtype).removeprefix("torch."),
            "operand_shapes": {
                name: list(value.shape) for name, value in component.operands.items()
            },
            "operand_dtypes": {
                name: str(value.dtype).removeprefix("torch.")
                for name, value in component.operands.items()
            },
        },
    )
    add(
        "arti/formula-commit-blend@1",
        FormulaCommitBlend,
        lifecycle=stable,
        variant="bounded-continuous-formula-commit",
        capabilities=("formula.fabric.executor",),
        config_builder=lambda component: {
            "fabric": component_spec(component.fabric).to_dict(),
            "blend": "old+alpha*(candidate-old)",
            "range": [0.0, 1.0],
        },
        dependency_builder=lambda component: (component_ref(component.fabric),),
    )
    add(
        "arti/formula-fabric-compute@1",
        FormulaFabricCompute,
        lifecycle=stable,
        variant="folded-workspace-formula-executor",
        capabilities=("pulse.stage.selective-compute",),
        config_schema_version=2,
        config_builder=lambda component: {
            "fabric": component_spec(component.fabric).to_dict(),
            "active_count": component.active_count,
            "commit_mode": component.commit_mode,
            "factor_contract": component.factor_contract,
            "route_contract": component.route_contract,
            "arena_layout": component.arena_layout,
            "visibility": component.visibility_contract,
            "execution_config_fingerprint": component.execution_config_fingerprint,
        },
        dependency_builder=lambda component: (component_ref(component.fabric),),
    )
    add(
        "arti/fixed-resident-bucket@1",
        FixedResidentBucket,
        lifecycle=stable,
        variant="runtime-only-static-cuda-bucket-contract",
        constructible=False,
        artifact_policy="runtime_only",
        capabilities=("tensor.resident.bucket",),
        config_builder=lambda component: {
            "batch_size": component.batch_size,
            "workset_slots": component.workset_slots,
            "feature_dim": component.feature_dim,
            "dtype": str(component.dtype),
            "device": str(component.device),
            "refine_steps": component.refine_steps,
        },
    )
    add(
        "arti/fixed-page-refs@1",
        FixedPageRefs,
        lifecycle=stable,
        variant="runtime-only-fixed-resident-page-references",
        constructible=False,
        artifact_policy="runtime_only",
        capabilities=("tensor.resident.page-refs",),
        config_builder=_fixed_page_refs_config,
    )
    add(
        "arti/hot-page-pool@1",
        HotPagePool,
        lifecycle=stable,
        variant="host-bound-cuda-page-pool",
        constructible=False,
        artifact_policy="host_bound",
        capabilities=("tensor.authority.host-mediated", "tensor.resident.pool"),
        config_builder=_hot_page_pool_config,
    )
    add(
        "arti/bound-hot-page-pool@1",
        BoundHotPagePool,
        lifecycle=stable,
        variant="host-bound-fixed-page-workset",
        constructible=False,
        artifact_policy="host_bound",
        capabilities=("tensor.authority.host-mediated", "tensor.resident.pool"),
        config_builder=_bound_hot_page_pool_config,
        dependency_builder=lambda component: (
            component_ref(component.pool),
            component_ref(component.bucket),
            component_ref(component.refs),
        ),
    )
    add(
        "arti/captured-hot-step@1",
        CapturedHotStep,
        lifecycle=stable,
        variant="host-bound-read-only-cuda-graph-step",
        constructible=False,
        artifact_policy="host_bound",
        capabilities=("tensor.resident.graph-replay",),
        config_builder=lambda component: {
            "bound_ref": component_ref(component.bound),
            "commit": component.commit,
            "lifecycle_state": component.lifecycle_state,
        },
        dependency_builder=lambda component: (component_ref(component.bound),),
    )
    add(
        "arti/resident-latency-receipt@1",
        ResidentLatencyReceipt,
        lifecycle=stable,
        variant="runtime-only-scoped-latency-receipt",
        constructible=False,
        artifact_policy="runtime_only",
        capabilities=("tensor.resident.measurement",),
        config_builder=lambda component: asdict(component),
    )
    add(
        "arti/cuda-activity-receipt@1",
        CUDAActivityReceipt,
        lifecycle=stable,
        variant="runtime-only-cupti-activity-receipt",
        constructible=False,
        artifact_policy="runtime_only",
        capabilities=("tensor.resident.measurement",),
        config_builder=lambda component: asdict(component),
    )
    add(
        "arti/runtime-checkpoint-receipt@1",
        RuntimeCheckpointReceipt,
        lifecycle=stable,
        variant="runtime-only-atomic-checkpoint-receipt",
        constructible=False,
        artifact_policy="runtime_only",
        capabilities=("tensor.runtime.checkpoint",),
        config_builder=_runtime_checkpoint_receipt_config,
    )
    add(
        "arti/restored-runtime-checkpoint@1",
        RestoredRuntimeCheckpoint,
        lifecycle=stable,
        variant="host-bound-restored-runtime-root",
        constructible=False,
        artifact_policy="host_bound",
        capabilities=("tensor.authority.host-mediated", "tensor.runtime.checkpoint"),
        config_builder=_restored_runtime_checkpoint_config,
        dependency_builder=lambda component: tuple(
            dict.fromkeys(
                (
                    *(
                        ()
                        if component.resident is None
                        else (component_ref(component.resident),)
                    ),
                    *(binding.component_ref for binding in component.bindings),
                )
            )
        ),
    )
    add(
        "arti/formula-resident-operation@1",
        FormulaResidentOperation,
        lifecycle=stable,
        variant="fixed-route-resident-formula",
        capabilities=("formula.fabric.resident-operation",),
        config_builder=_resident_formula_operation_config,
        dependency_builder=lambda component: (component_ref(component.compute),),
    )
    add(
        "arti/topology-formula-resident-operation@1",
        TopologyFormulaResidentOperation,
        lifecycle=stable,
        variant="fold-formula-unfold-resident-operation",
        capabilities=("formula.fabric.resident-operation", "topology.reversible"),
        config_builder=_topology_resident_operation_config,
        dependency_builder=lambda component: (
            component_ref(component.fold),
            component_ref(component.unfold),
            component_ref(component.compute),
        ),
    )
    add(
        "arti/batched-refine-operation@1",
        BatchedRefineOperation,
        lifecycle=stable,
        variant="existing-formula-topology-step-operation",
        capabilities=("recall.batched-refine.operation",),
        config_builder=lambda component: {
            "operation_ref": component.operation_ref,
            "operation_config_fingerprint": component_spec(
                component.operation
            ).config_fingerprint,
        },
        dependency_builder=lambda component: (component.operation_ref,),
    )
    add(
        "arti/execution-rng-plan@2",
        ExecutionRNGPlan,
        lifecycle=stable,
        variant="runtime-only-callsite-and-branch-origin-keyed-rng-plan",
        constructible=False,
        artifact_policy="runtime_only",
        capabilities=("recall.batched-refine.rng-plan",),
        config_builder=lambda component: {
            "algorithm": component.algorithm,
            "stream_key": component.stream_key,
            "fingerprint": component.fingerprint,
            "sample_count": len(component.sample_keys),
        },
    )
    add(
        "arti/execution-context-receipt@3",
        ExecutionContextReceipt,
        lifecycle=stable,
        variant="runtime-only-keyed-or-deterministic-execution-context",
        constructible=False,
        artifact_policy="runtime_only",
        capabilities=("recall.batched-refine.execution-context",),
        config_builder=lambda component: {
            "schema_version": component.schema_version,
            "mode": component.mode,
            "algorithm": component.algorithm,
            "execution_rng_fingerprint": component.execution_rng_fingerprint,
            "execution_rng_stream_key": component.execution_rng_stream_key,
            "consumed_domains": list(component.consumed_domains),
            "fingerprint": component.fingerprint,
        },
    )
    add(
        "arti/batched-refine@1",
        BatchedRefineExecutor,
        lifecycle=stable,
        variant="factory-owned-authority-closure",
        constructible=False,
        artifact_policy="runtime_only",
        capabilities=("recall.batched-refine.executor",),
        config_schema_version=2,
        config_builder=lambda component: {
            "schema_version": component.schema_version,
            "identity_mode": component.identity_mode,
            "candidate_ref": component.candidate_ref,
            "candidate_manifest_fingerprint": (
                component.candidate_manifest_fingerprint
            ),
            "plan_ref": component.plan_ref,
            "plan_config_fingerprint": component.plan_config_fingerprint,
            "execution_layout": component.execution_layout,
            "operation_ref": component.operation_ref,
            "route_fingerprint": component.route_fingerprint,
            "topology_refs": list(component.topology_refs),
            "topology_contract_fingerprints": list(
                component.topology_contract_fingerprints
            ),
            "branch_policy_fingerprint": component.branch_policy_fingerprint,
            "execution_context_ref": component.execution_context._runtime_contract_ref,
            "execution_context_fingerprint": component.execution_context.fingerprint,
        },
        dependency_builder=lambda component: tuple(
            reference
            for reference in (
                component.candidate_ref,
                component.plan_ref,
                component.operation_ref,
                *component.topology_refs,
                component.execution_context._runtime_contract_ref,
                (
                    None
                    if component.branch_policy_fingerprint is None
                    else "arti/branch-refine-policy@1"
                ),
            )
            if reference is not None
        ),
    )
    add(
        "arti/resident-branch-run@1",
        ResidentBranchRun,
        lifecycle=stable,
        variant="host-bound-gpu-resident-branch-authority",
        constructible=False,
        artifact_policy="host_bound",
        capabilities=(
            "recall.batched-refine.resident",
            "tensor.authority.host-mediated",
        ),
        config_builder=lambda component: {
            "run_instance_token": component._run_instance_token,
            "result_manifest_fingerprint": component._manifest,
            "spec_fingerprint": component.spec.fingerprint,
            "pool_layout_fingerprint": component._pool_layout_fingerprint,
            "active_count": component._active_count,
        },
        dependency_builder=lambda component: (
            component_ref(component.result),
            component_ref(component.executor),
        ),
    )
    add(
        "arti/resident-branch-score@1",
        ResidentBranchScoreReceipt,
        lifecycle=stable,
        variant="runtime-only-host-visible-k-score-receipt",
        constructible=False,
        artifact_policy="runtime_only",
        capabilities=("recall.batched-refine.resident-score",),
        config_builder=lambda component: {
            "spec_fingerprint": component.spec_fingerprint,
            "run_instance_token": component.run_instance_token,
            "result_manifest_fingerprint": component.result_manifest_fingerprint,
            "branch_ids": list(component.branch_ids),
            "future_fingerprint": component.future_fingerprint,
            "scorer_ref": component.scorer_ref,
            "scorer_config_fingerprint": component.scorer_config_fingerprint,
            "tie_policy": component.tie_policy,
            "scores": list(component.scores),
            "receipt_fingerprint": component.receipt_fingerprint,
        },
        dependency_builder=lambda component: (component.scorer_ref,),
    )
    add(
        "arti/resident-branch-decision@1",
        ResidentBranchDecision,
        lifecycle=stable,
        variant="runtime-only-host-authority-decision",
        constructible=False,
        artifact_policy="runtime_only",
        capabilities=("recall.batched-refine.resident-decision",),
        config_builder=lambda component: {
            "kind": component.kind,
            "run_instance_token": component.run_instance_token,
            "spec_fingerprint": component.spec_fingerprint,
            "score_receipt_fingerprint": component.score_receipt_fingerprint,
            "winner_origin": component.winner_origin,
            "weights": list(component.weights),
            "idempotency_key": component.idempotency_key,
            "decision_fingerprint": component.decision_fingerprint,
        },
    )
    add(
        "arti/resident-branch-commit@1",
        ResidentBranchCommitReceipt,
        lifecycle=stable,
        variant="runtime-only-gpu-resident-publication-receipt",
        constructible=False,
        artifact_policy="runtime_only",
        capabilities=("recall.batched-refine.resident-commit",),
        config_builder=lambda component: {
            "status": component.status,
            "decision_fingerprint": component.decision_fingerprint,
            "idempotency_key": component.idempotency_key,
            "result_manifest_fingerprint": component.result_manifest_fingerprint,
            "pool_layout_fingerprint": component.pool_layout_fingerprint,
            "committed_linear_refs": list(component.committed_linear_refs),
            "generations": list(component.generations),
            "versions_before": list(component.versions_before),
            "versions_after": list(component.versions_after),
            "receipt_fingerprint": component.receipt_fingerprint,
        },
    )
    add(
        "arti/batched-refine-plan@1",
        BatchedRefinePlan,
        lifecycle=stable,
        variant="candidate-recall-operation-requery",
        capabilities=("recall.batched-refine.plan",),
        config_builder=lambda component: {
            "schema_version": component.schema_version,
            "execution_layout": component.execution_layout,
            "config_fingerprint": component.config_fingerprint,
            "operation_ref": (
                None
                if component.operation is None
                else component.operation.operation_ref
            ),
        },
        dependency_builder=lambda component: (
            ()
            if component.operation is None
            else (component_ref(component.operation),)
        ),
    )
    add(
        "arti/branch-refine-policy@1",
        BranchRefinePolicy,
        lifecycle=stable,
        variant="runtime-only-candidate-bound-branch-policy",
        artifact_policy="runtime_only",
        capabilities=("recall.batched-refine.branch-policy",),
        config_builder=lambda component: {
            "config_fingerprint": component.config_fingerprint,
            "base_ref": component.base._component_reference,
        },
        dependency_builder=lambda component: (component.base._component_reference,),
    )
    add(
        "arti/recall-branch-batch@3",
        RecallBranchBatch,
        lifecycle=stable,
        variant="runtime-only-single-value-candidate-batch",
        constructible=False,
        artifact_policy="runtime_only",
        capabilities=("recall.batched-refine.candidates",),
        config_schema_version=3,
        config_builder=_recall_branch_batch_config,
        dependency_builder=lambda component: (component.source_ref,),
    )
    add(
        "arti/recall-formula-branch-batch@3",
        RecallFormulaBranchBatch,
        lifecycle=stable,
        variant="runtime-only-joint-formula-candidate-batch",
        constructible=False,
        artifact_policy="runtime_only",
        capabilities=(
            "recall.batched-refine.candidates",
            "recall.formula.factor-aware",
        ),
        config_schema_version=6,
        config_builder=_recall_branch_batch_config,
        dependency_builder=lambda component: (
            component.source_ref,
            component.formula_ref,
        ),
    )
    add(
        "arti/batched-refine-result@1",
        BatchedRefineResult,
        lifecycle=stable,
        variant="runtime-only-branch-result",
        constructible=False,
        artifact_policy="runtime_only",
        capabilities=("recall.batched-refine.result",),
        config_schema_version=4,
        config_builder=_batched_refine_result_config,
        dependency_builder=_batched_refine_result_dependencies,
    )
    add(
        "arti/bank-formula-route-source@1",
        BankFormulaRouteSource,
        lifecycle=stable,
        variant="fixed-query-bank-formula-route-plan",
        capabilities=("formula.fabric.route-source",),
        config_builder=_bank_formula_route_source_config,
        dependency_builder=_bank_formula_route_source_dependencies,
    )
    add(
        "arti/routed-formula-fabric-compute@1",
        RoutedFormulaFabricCompute,
        lifecycle=stable,
        variant="bank-routed-formula-fabric-adapter",
        capabilities=("pulse.stage.selective-compute",),
        config_schema_version=2,
        config_builder=_routed_formula_fabric_compute_config,
        dependency_builder=_routed_formula_fabric_compute_dependencies,
    )
    add(
        "arti/iterative-routed-formula-fabric-compute@1",
        IterativeRoutedFormulaFabricCompute,
        lifecycle=stable,
        variant="state-conditioned-route-refinement",
        capabilities=("pulse.stage.selective-compute",),
        config_builder=lambda component: {
            "routed": component_spec(component.routed).to_dict(),
            "steps": component.steps,
            "route_semantics": "requery-after-program",
            "executor_reuse": True,
            "config_fingerprint": component.config_fingerprint,
        },
        dependency_builder=lambda component: (component_ref(component.routed),),
    )
    add(
        "arti/selective-compute@1",
        SelectiveCompute,
        lifecycle=stable,
        variant="pack-apply-scatter",
        capabilities=("pulse.stage.selective-compute",),
        config_builder=lambda component: {
            "kernel": component_spec(component.kernel).to_dict(),
            "max_queries": component.max_queries,
            "max_sources": component.max_sources,
            "limits": dict(component.limits.__dict__),
        },
        dependency_builder=lambda component: (component_ref(component.kernel),),
    )
    add(
        "arti/selective-recall-kernel@1",
        SelectiveRecallKernel,
        lifecycle=stable,
        variant="packed-canonical-recall-kernel",
        capabilities=("selective.compute.kernel",),
        config_builder=lambda component: {
            "recall": component_spec(component.recall).to_dict(),
            "refine_policy": component_spec(component.refine_policy).to_dict(),
        },
        dependency_builder=lambda component: (
            component_ref(component.recall),
            component_ref(component.refine_policy),
        ),
    )
    add(
        "arti/soft-fold-aggregate@1",
        SoftFoldAggregate,
        lifecycle=stable,
        variant="soft-slot-aggregation",
        capabilities=("pulse.aggregate.kernel",),
        config_builder=lambda component: {
            "k": component.k,
            "dim": component.dim,
            "fold": component_spec(component.fold).to_dict(),
        },
        dependency_builder=lambda _component: ("arti/fold@1",),
    )
    add(
        "arti/reunion-aggregate@1",
        ReunionAggregate,
        lifecycle=stable,
        variant="reunion-only-aggregate-host",
        capabilities=("pulse.stage.aggregate",),
        config_builder=lambda component: {
            "kernel": component_spec(component.kernel).to_dict(),
        },
        dependency_builder=lambda component: (component_ref(component.kernel),),
    )
    add(
        "arti/tensor-schema@1",
        TensorSchema,
        lifecycle=stable,
        variant="shape-autonomous-logical-schema",
        config_builder=lambda component: component.to_dict(),
        capabilities=("federal.contract.tensor-schema",),
    )
    add(
        "arti/shape-relation@1",
        ShapeRelation,
        lifecycle=stable,
        variant="declared-shape-relation",
        config_builder=lambda component: component.to_dict(),
        capabilities=("federal.contract.shape-relation",),
    )
    add(
        "arti/gradient-contract@1",
        GradientContract,
        lifecycle=stable,
        variant="declared-gradient-boundary",
        config_builder=lambda component: component.to_dict(),
        capabilities=("federal.contract.gradient",),
    )
    add(
        "arti/terminal-output-abi@1",
        TerminalOutputABI,
        lifecycle=stable,
        variant="named-terminal-tensor-alliance",
        config_builder=lambda component: component.to_dict(),
        dependency_builder=lambda _component: (
            "arti/gradient-contract@1",
            "arti/tensor-schema@1",
        ),
        capabilities=("federal.contract.terminal-abi",),
    )
    add(
        "arti/bank-execution-signature@1",
        BankExecutionSignature,
        lifecycle=stable,
        variant="shape-autonomous-bank-program",
        config_builder=lambda component: component.to_dict(),
        dependency_builder=lambda _component: (
            "arti/gradient-contract@1",
            "arti/shape-relation@1",
            "arti/tensor-schema@1",
            "arti/terminal-output-abi@1",
        ),
        capabilities=("federal.contract.bank-signature",),
    )
    add(
        "arti/linear-bank-query@1",
        LinearBankQuery,
        lifecycle=stable,
        variant="trainable-bank-local-projection",
        config_builder=lambda component: component.contract_config(),
        dependency_builder=lambda _component: (
            "arti/gradient-contract@1",
            "arti/tensor-schema@1",
        ),
        capabilities=("federal.query.pretrainable",),
    )
    add(
        "arti/query-execution-signature@1",
        QueryExecutionSignature,
        lifecycle=stable,
        variant="sealed-bank-query-identity",
        config_builder=lambda component: component.to_dict(),
        dependency_builder=lambda component: (
            "arti/gradient-contract@1",
            "arti/tensor-schema@1",
            component.query_ref,
        ),
        capabilities=("federal.query.sealed-signature",),
    )
    add(
        "arti/sealed-bank-query@1",
        SealedBankQuery,
        lifecycle=stable,
        variant="bank-owned-pretrained-then-sealed",
        config_builder=lambda component: component.contract_config(),
        dependency_builder=lambda component: (
            component_ref(component.query),
            "arti/query-execution-signature@1",
        ),
        capabilities=(
            "federal.query.bank-owned",
            "federal.query.runtime-fixed",
        ),
    )
    add(
        "arti/bank-execution-signature@2",
        BankExecutionSignatureV2,
        lifecycle=stable,
        variant="shape-autonomous-bank-owned-query-program",
        config_builder=lambda component: component.to_dict(),
        dependency_builder=_bank_execution_signature_v2_dependencies,
        capabilities=("federal.contract.bank-owned-query-signature",),
    )
    add(
        "arti/bank-local-refine-policy@1",
        BankLocalRefinePolicy,
        lifecycle=stable,
        variant="latest-local-state-requery-budget",
        config_builder=lambda component: component.contract_config(),
        capabilities=("federal.bank-local-refine.policy",),
    )
    add(
        "arti/bank-local-formula-action@1",
        BankLocalFormulaAction,
        lifecycle=stable,
        variant="typed-bank-owned-formula-transition",
        constructible=False,
        config_builder=lambda component: component.contract_config(),
        dependency_builder=lambda component: (
            component_ref(component.fabric),
            "arti/tensor-schema@1",
        ),
        capabilities=("federal.bank-local.formula-action",),
    )
    add(
        "arti/value-terminal-adapter@1",
        ValueTerminalAdapter,
        lifecycle=stable,
        variant="value-validity-score-terminal-adapter",
        config_builder=lambda component: component.contract_config(),
        capabilities=("federal.terminal.adapter",),
    )
    add(
        "arti/bank-local-terminal-action@1",
        BankLocalTerminalAction,
        lifecycle=stable,
        variant="formula-requested-terminal-action",
        constructible=False,
        config_builder=lambda component: component.contract_config(),
        dependency_builder=lambda component: (
            component_ref(component.adapter),
            component_ref(component.exit_atom),
            "arti/tensor-schema@1",
        ),
        capabilities=("federal.bank-local.terminal-action",),
    )
    add(
        "arti/bank-local-formula-program@1",
        BankLocalFormulaProgram,
        lifecycle=stable,
        variant="declarative-latest-state-formula-program",
        constructible=False,
        config_builder=lambda component: component.contract_config(),
        dependency_builder=lambda component: tuple(
            sorted(
                {
                    *(component_ref(action) for action in component.actions),
                    component_ref(component.terminal_action),
                    component_ref(component.query),
                    component_ref(component.local_refine),
                    "arti/bank-execution-signature@2",
                    "arti/terminal-output-abi@1",
                }
            )
        ),
        capabilities=(
            "federal.bank-local.latest-state-requery",
            "federal.bank-local.typed-formula-program",
        ),
    )
    add(
        "arti/exact-bank-local-program-training@1",
        ExactBankLocalProgramTraining,
        lifecycle=stable,
        variant="exact-expected-policy-final-task-loss",
        artifact_policy="runtime_only",
        config_builder=lambda component: component.contract_config(),
        dependency_builder=lambda _component: (
            "arti/bank-local-formula-action@1",
            "arti/bank-local-terminal-action@1",
        ),
        capabilities=("federal.bank-local.training.exact-expected-policy",),
    )
    add(
        "arti/detached-bank-local-rollout@1",
        DetachedBankLocalRollout,
        lifecycle=stable,
        variant="variable-shape-detached-on-policy-states",
        constructible=False,
        artifact_policy="runtime_only",
        config_builder=lambda component: {
            "state_count": len(component.states),
            "min_steps": component.min_steps,
            "max_steps": component.max_steps,
            "terminated": component.terminated,
            "route_cache": False,
            "transition_teacher": False,
        },
        capabilities=("federal.bank-local.training.detached-rollout",),
    )
    add(
        "arti/detached-bank-local-program-training@1",
        DetachedBankLocalProgramTraining,
        lifecycle=stable,
        variant="fresh-one-step-final-task-loss",
        artifact_policy="runtime_only",
        config_builder=lambda component: component.contract_config(),
        dependency_builder=lambda _component: (
            "arti/detached-bank-local-rollout@1",
            "arti/bank-local-formula-action@1",
            "arti/bank-local-terminal-action@1",
        ),
        capabilities=("federal.bank-local.training.detached-on-policy",),
    )
    add(
        "arti/federal-recall@1",
        FederalRecall,
        lifecycle=stable,
        variant="shape-autonomous-fixed-k-eager-reference",
        config_builder=lambda component: component.contract_config(),
        dependency_builder=lambda _component: (
            "arti/bank-execution-signature@1",
            "arti/terminal-output-abi@1",
        ),
        capabilities=(
            "federal.execution.eager-reference",
            "federal.execution.fixed-k",
            "federal.execution.hard-one-winner",
        ),
    )
    add(
        "arti/federal-recall@2",
        FederalRecallV2,
        lifecycle=stable,
        variant="bank-owned-query-serial-requery",
        config_builder=lambda component: component.contract_config(),
        dependency_builder=_federal_recall_v2_dependencies,
        capabilities=(
            "federal.execution.eager-reference",
            "federal.execution.latest-state-requery",
            "federal.execution.serial-k1",
        ),
    )
    add(
        "arti/pulse-stage-graph@1",
        PulseStageGraph,
        lifecycle=stable,
        variant="ordered-stage-graph",
        state_schema_version=PULSE_STAGE_GRAPH_SCHEMA_VERSION,
        config_builder=lambda component: component.to_dict(),
        dependency_builder=lambda component: (
            "arti/pulse-stage@1",
            *component.enabled_dependencies,
        ),
    )
    add(
        "arti/pulse-executor@1",
        PulseExecutor,
        lifecycle=stable,
        variant="manifest-bound-executor",
        config_builder=lambda component: {
            "manifest_ref": "arti/pulse-stage-graph@1",
            "manifest": component.manifest.to_dict(),
            "manifest_fingerprint": component.manifest.fingerprint,
            "enabled_stage_ids": sorted(component.enabled_stage_ids),
        },
        dependency_builder=lambda component: (
            "arti/pulse-stage-graph@1",
            *component.manifest.enabled_dependencies,
        ),
    )
    add(
        "arti/pulse@1",
        LearnedPulse,
        lifecycle=stable,
        variant="learned",
        aliases=("Pulse", "LearnedPulse"),
        deprecated_aliases=("arti/learned-pulse@1",),
        config_builder=_learned_pulse_config,
    )
    add(
        "arti/pulse@2",
        AdaptivePulse,
        lifecycle=stable,
        variant="adaptive-composable-stage-graph",
        config_schema_version=2,
        config_builder=_adaptive_pulse_config,
        dependency_builder=_adaptive_pulse_dependencies,
    )
    add(
        "arti/pulse-legacy@1",
        PulseCompressor,
        lifecycle="legacy",
        variant="explicit",
        aliases=("PulseCompressor",),
    )
    add("arti/fusion-pulse@1", FusionPulse, lifecycle=stable, config_builder=_fields("k", "dim", "hidden_dim", "salience_heads", "half_threshold", "salience_scale", "similarity_threshold", "representative_target", "redundancy_weight", "support_weight", "representative_weight", "value_operators", "value_rank", "eps"))
    add(
        "arti/recall@2",
        Recall,
        lifecycle=stable,
        config_schema_version=2,
        factory=global_recall_factory,
        config_builder=_recall_config,
        dependency_builder=_recall_dependencies,
    )
    add(
        "arti/recall@3",
        Recall,
        lifecycle=stable,
        variant="per-bank-route-normalization",
        config_schema_version=4,
        factory=per_bank_recall_factory,
        config_builder=_recall_config,
        dependency_builder=_recall_dependencies,
    )
    add(
        "arti/recall@4",
        Recall,
        lifecycle=stable,
        variant="independent-k-wide-refine",
        config_schema_version=1,
        factory=wide_recall_factory,
        config_builder=_recall_config,
        dependency_builder=_recall_dependencies,
    )
    add(
        "arti/recall-refiner@2",
        RecallRefiner,
        lifecycle=stable,
        variant="runtime-policy-adapter",
        config_builder=lambda _component: {},
        dependency_builder=_refiner_dependencies,
    )
    add(
        "arti/target-bank-updater@1",
        TargetBankUpdater,
        lifecycle=stable,
        variant="runtime-policy-adapter",
        capabilities=("pulse.stage.bank-update",),
        config_builder=_target_bank_updater_config,
        dependency_builder=_target_bank_updater_dependencies,
    )
    add(
        "arti/target-bank-updater@2",
        TargetBankUpdater,
        lifecycle=stable,
        variant="target-coupled-after-bootstrap",
        config_schema_version=2,
        capabilities=("pulse.stage.bank-update",),
        factory=coupled_target_bank_updater_factory,
        config_builder=_target_bank_updater_config,
        dependency_builder=_target_bank_updater_dependencies,
    )
    add(
        "arti/write-refine-policy@1",
        WriteRefinePolicy,
        lifecycle=stable,
        variant="runtime-only",
        config_builder=_fields("budget", "stop", "exposure_schedule"),
        dependency_builder=lambda component: (
            "arti/refine-budget@1",
            *(("arti/refine-stop@1",) if _attr(component, "stop") is not None else ()),
        ),
    )
    add(
        "arti/objective-exposure-bank@1",
        ObjectiveExposureBank,
        lifecycle=stable,
        variant="fixed-query-trainable-value-exposure",
        config_builder=_fields(
            "slots",
            "query_dim",
            "key_layout",
            "key_seed",
            "temperature",
            "min_exposure",
            "max_exposure",
            "init_scale",
        ),
    )
    add(
        "arti/objective-formula-fabric-compute@1",
        ObjectiveFormulaFabricCompute,
        lifecycle=stable,
        variant="objective-controlled-formula-commit",
        config_schema_version=1,
        capabilities=("pulse.stage.selective-compute",),
        config_builder=lambda component: {
            "compute": component_spec(component.compute).to_dict(),
            "objective": component_spec(component.objective).to_dict(),
            "program_fingerprint": component.program_fingerprint,
            "control_semantics": "formula-commit-weight",
            "control_scope": "per-pulse-fixed",
            "factor_contract": component.factor_contract,
            "limits": dict(component.limits.__dict__),
        },
        dependency_builder=lambda component: (
            component_ref(component.compute),
            component_ref(component.objective),
        ),
    )
    add(
        "arti/refine-policy@1",
        RefinePolicy,
        lifecycle=stable,
        variant="runtime-only",
        config_builder=_fields(
            "max_steps",
            "min_steps",
            "tolerance",
            "checkpoints",
            "trace_level",
            "cycle_tolerance",
            "cycle_periods",
            "check_finite",
            "nonfinite_action",
            "checkpoint_mode",
        ),
    )
    add(
        "arti/refine-budget@1",
        RefineBudget,
        lifecycle=stable,
        variant="runtime-only",
        config_builder=_fields("max_steps", "min_steps"),
    )
    add(
        "arti/refine-stop@1",
        RefineStop,
        lifecycle=stable,
        variant="runtime-only",
        config_builder=_fields(
            "scope",
            "absolute_tolerance",
            "relative_tolerance",
            "route_tolerance",
            "patience",
            "cycle_tolerance",
            "cycle_periods",
        ),
    )
    add(
        "arti/refine-policy@2",
        AdaptiveRefinePolicy,
        lifecycle=stable,
        variant="adaptive-runtime",
        config_schema_version=2,
        factory=RefinePolicy.adaptive,
        config_builder=_fields(
            "budget",
            "stop",
            "checkpoints",
            "trace_level",
            "check_finite",
            "nonfinite_action",
            "checkpoint_mode",
            "executor",
        ),
        dependency_builder=lambda _component: (
            "arti/refine-budget@1",
            "arti/refine-stop@1",
        ),
    )
    add(
        "arti/refine-rollout@1",
        RefineRollout,
        lifecycle=stable,
        variant="detached-on-policy-adjacent-steps",
        constructible=False,
        artifact_policy="runtime_only",
        config_builder=lambda _component: {
            "trajectory_source": "on_policy_snapshot",
            "training_view": "flattened_adjacent_steps",
            "hidden_teacher": "forbidden",
            "route_cache": "forbidden",
            "query_trainable": False,
        },
        dependency_builder=lambda component: (component.source_ref,),
    )
    add(
        "arti/refine-step-training@1",
        RefineStepTraining,
        lifecycle=stable,
        variant="fresh-query-one-step-task-loss",
        artifact_policy="runtime_only",
        config_builder=_fields("max_snapshot_staleness"),
    )
    add(
        "arti/formula-atom-refine-exit@1",
        FormulaRefineExit,
        lifecycle=stable,
        variant="post-transition-hard-refine-exit",
        capabilities=("refine.exit.atom", "refine.exit.request"),
        config_builder=_fields("input_kind", "scope", "threshold"),
    )
    add(
        "arti/refine-exit-request@1",
        RefineExitRequest,
        lifecycle=stable,
        variant="tensor-only-post-transition-request",
        constructible=False,
        artifact_policy="runtime_only",
        config_builder=lambda _component: {},
        dependency_builder=lambda _component: ("arti/formula-atom-refine-exit@1",),
    )
    add(
        "arti/refine-exit-control@1",
        RefineExitControl,
        lifecycle=stable,
        variant="neural-source-with-typed-exit-atom",
        constructible=False,
        artifact_policy="runtime_only",
        capabilities=("refine.exit.control",),
        config_builder=lambda component: {
            "source_api": (
                f"{type(component.source).__module__}."
                f"{type(component.source).__qualname__}"
            ),
            "input_kind": component.atom.input_kind,
            "scope": component.atom.scope,
            "threshold": component.atom.threshold,
        },
        dependency_builder=lambda component: (component_ref(component.atom),),
    )
    add(
        "arti/refine-exit-curve@1",
        RefineExitCurve,
        lifecycle=stable,
        variant="detached-full-depth-task-loss-curve",
        constructible=False,
        artifact_policy="runtime_only",
        config_builder=lambda component: {
            "scope": component.scope,
            "depth": component.depth,
            "breadth": component.breadth,
            "sampling_policy": component.sampling_policy,
        },
        dependency_builder=lambda component: (
            component.source_ref,
            "arti/refine-step-training@1",
        ),
    )
    add(
        "arti/refine-exit-training@1",
        RefineExitTraining,
        lifecycle=stable,
        variant="quality-constrained-task-loss-hazard",
        artifact_policy="runtime_only",
        config_builder=_fields(
            "temperature",
            "compute_weight",
            "quality_tolerance",
            "quality_weight",
        ),
        dependency_builder=lambda _component: (
            "arti/refine-exit-curve@1",
            "arti/refine-exit-control@1",
        ),
    )
    add(
        "arti/operable-tensor-port-spec@3",
        PortSpec,
        lifecycle=stable,
        variant="logical-tensor-shape-and-folded-view-spec",
        constructible=False,
        config_builder=lambda component: {
            "canvas_tokens": component.canvas_tokens,
            "tensor_shape": component.tensor_shape,
            "dim": component.dim,
            "tensor_to_canvas": component.tensor_to_canvas,
            "folded_tensor_coordinates": component.folded_tensor_coordinates,
            "dtype": str(component.dtype).removeprefix("torch."),
            "empty_value": component.empty_value,
            "default_visible": component.default_visible,
            "coordinate_frame": component.coordinate_frame,
        },
    )
    add(
        "arti/operable-tensor-port@2",
        OperableTensorPort,
        lifecycle=stable,
        variant="stable-runtime-owned-default-or-external-backing",
        constructible=False,
        artifact_policy="runtime_only",
        config_builder=lambda component: {
            "batch_size": component.batch_size,
        },
        dependency_builder=lambda component: (component_ref(component.spec),),
    )
    add(
        "arti/operable-tensor-snapshot@2",
        PortSnapshot,
        lifecycle=stable,
        variant="resolved-pre-step-backing",
        constructible=False,
        artifact_policy="runtime_only",
        config_builder=lambda component: {
            "source": component.source,
            "backing_epoch": component.backing_epoch,
            "step_index": component.step_index,
        },
        dependency_builder=lambda _component: ("arti/operable-tensor-port@2",),
    )
    add(
        "arti/shared-canvas@3",
        SharedCanvas,
        lifecycle=stable,
        variant="world-shaped-masked-overlay",
        constructible=False,
        artifact_policy="runtime_only",
        config_builder=lambda component: {
            "shape": tuple(component.values.shape),
            "backing_epoch": component.backing_epoch,
            "step_index": component.step_index,
        },
        dependency_builder=lambda _component: ("arti/shared-canvas-fold@3",),
    )
    add(
        "arti/shared-canvas-fold@3",
        SharedCanvasFold,
        lifecycle=stable,
        variant="partial-fixed-map-masked-overlay",
        constructible=False,
        config_builder=lambda _component: {},
        dependency_builder=lambda component: (component_ref(component.spec),),
    )
    add(
        "arti/tensor-operation-field-spec@2",
        TensorOperationFieldSpec,
        lifecycle=stable,
        variant="bounded-synchronous-operation-field",
        constructible=False,
        config_builder=lambda component: component.contract(),
        dependency_builder=lambda component: (component_ref(component.port),),
    )
    add(
        "arti/tensor-edit-instruction@3",
        TensorEditInstruction,
        lifecycle=stable,
        variant="complete-hard-operation-field",
        constructible=False,
        artifact_policy="runtime_only",
        config_builder=lambda _component: {},
    )
    add(
        "arti/tensor-edit-formula@3",
        TensorEditFormula,
        lifecycle=stable,
        variant="synchronous-pre-state-operation-field",
        constructible=False,
        config_builder=lambda _component: {
            "operations": ("KEEP", "COPY", "ERASE"),
            "collision_policy": "last_element",
        },
        dependency_builder=lambda component: (
            component_ref(component.spec),
            "arti/tensor-edit-instruction@3",
        ),
    )
    add(
        "arti/tensor-edit-result@3",
        TensorEditResult,
        lifecycle=stable,
        variant="functional-next-backing",
        constructible=False,
        artifact_policy="runtime_only",
        config_builder=lambda component: {
            "shape": tuple(component.value.shape),
        },
        dependency_builder=lambda _component: (
            "arti/tensor-edit-formula@3",
            "arti/tensor-edit-instruction@3",
        ),
    )
    add(
        "arti/tensor-edit-surrogate@3",
        TensorEditSurrogate,
        lifecycle=stable,
        variant="exact-hard-forward-continuous-backward",
        constructible=False,
        config_builder=lambda component: {"temperature": component.temperature},
        dependency_builder=lambda component: (
            component_ref(component.spec),
            "arti/tensor-edit-formula@3",
        ),
    )
    add(
        "arti/tensor-operation-query@4",
        TensorOperationQuery,
        lifecycle=stable,
        variant="fixed-complete-world-and-backing-projection",
        constructible=False,
        config_builder=lambda component: component.operation_query_contract(),
        dependency_builder=lambda component: (component_ref(component.spec),),
    )
    add(
        "arti/tensor-operation-bank@3",
        TensorOperationBank,
        lifecycle=stable,
        variant="concat-native-complete-operation-fields",
        constructible=False,
        config_builder=lambda component: component.operation_bank_contract(),
        dependency_builder=lambda component: (
            component_ref(component.spec),
            component_ref(component.field_spec),
        ),
    )
    add(
        "arti/tensor-operation-decision@3",
        TensorOperationDecision,
        lifecycle=stable,
        variant="hard-instruction-with-training-logits",
        constructible=False,
        artifact_policy="runtime_only",
        config_builder=lambda component: {
            "batch_size": int(component.instruction.operation.shape[0]),
            "support_size": int(component.instruction.operation.shape[1]),
            "candidate_count": int(component.route.route.shape[-1]),
            "bank_count": len(component.route.bank_ids),
        },
        dependency_builder=lambda _component: (
            "arti/tensor-edit-instruction@3",
            "arti/tensor-operation-bank@3",
        ),
    )
    add(
        "arti/tensor-operation-selector@3",
        TensorOperationSelector,
        lifecycle=stable,
        variant="fixed-query-bank-selected-hard-edit",
        constructible=False,
        config_builder=lambda component: {
            "estimator": component.estimator,
            "temperature": component.temperature,
        },
        dependency_builder=lambda component: (
            component_ref(component.spec),
            component_ref(component.query),
            component_ref(component.bank),
        ),
    )
    add(
        "arti/tensor-operation@3",
        TensorOperation,
        lifecycle=stable,
        variant="fixed-query-bank-selected-hard-transition",
        config_schema_version=2,
        constructible=False,
        config_builder=lambda component: {
            "surrogate": (
                None
                if component.surrogate is None
                else component_ref(component.surrogate)
            ),
        },
        dependency_builder=lambda component: (
            component_ref(component.spec),
            component_ref(component.selector),
            component_ref(component.fold),
            component_ref(component.formula),
        )
        + (() if component.surrogate is None else (component_ref(component.surrogate),)),
    )
    add(
        "arti/tensor-operation-step-result@3",
        TensorOperationStepResult,
        lifecycle=stable,
        variant="one-local-shadow-transition",
        constructible=False,
        artifact_policy="runtime_only",
        config_builder=lambda component: {"shape": tuple(component.edit.value.shape)},
        dependency_builder=lambda _component: (
            "arti/tensor-operation@3",
            "arti/tensor-operation-decision@3",
            "arti/tensor-edit-result@3",
        ),
    )
    add(
        "arti/tensor-operation-stop@1",
        TensorOperationStopPolicy,
        lifecycle=stable,
        variant="bounded-post-transition-stop",
        constructible=False,
        config_builder=lambda component: {
            "min_operation_steps": component.min_operation_steps,
            "stop_on_stable": component.stop_on_stable,
        },
    )
    add(
        "arti/tensor-operation-schedule@1",
        TensorOperationSchedule,
        lifecycle=stable,
        variant="independent-operation-depth",
        config_schema_version=2,
        constructible=False,
        artifact_policy="runtime_only",
        config_builder=lambda component: {
            "scalar_steps": (
                component.operation_steps
                if isinstance(component.operation_steps, int)
                else None
            ),
            "max_steps": component.max_steps,
        },
    )
    add(
        "arti/tensor-operation-trace@3",
        TensorOperationTrace,
        lifecycle=stable,
        variant="bounded-operation-axis-trace",
        constructible=False,
        artifact_policy="runtime_only",
        config_builder=lambda component: {
            "steps": int(component.attempted.shape[0]),
            "batch_size": int(component.requested_steps.shape[0]),
        },
    )
    add(
        "arti/tensor-operation-loop@3",
        TensorOperationLoop,
        lifecycle=stable,
        variant="private-shadow-fresh-query-loop",
        config_schema_version=2,
        constructible=False,
        config_builder=lambda component: {"executor": component.executor},
        dependency_builder=lambda component: (
            component_ref(component.operation),
            component_ref(component.stop),
        ),
    )
    add(
        "arti/tensor-operation-result@3",
        TensorOperationResult,
        lifecycle=stable,
        variant="next-call-backing-proposal",
        constructible=False,
        artifact_policy="runtime_only",
        config_builder=lambda component: {"shape": tuple(component.value.shape)},
        dependency_builder=lambda _component: (
            "arti/tensor-operation-loop@3",
            "arti/tensor-operation-trace@3",
        ),
    )
    add(
        "arti/reader-refine-schedule@1",
        ReaderRefineSchedule,
        lifecycle=stable,
        variant="independent-reader-depth",
        constructible=False,
        artifact_policy="runtime_only",
        config_builder=lambda component: {"reader_steps": component.reader_steps},
    )
    add(
        "arti/tensor-invocation@2",
        TensorInvocation,
        lifecycle=stable,
        variant="same-root-independent-reader-operation-axes",
        constructible=False,
        config_builder=lambda component: {
            "reader_api": (
                f"{type(component.reader).__module__}."
                f"{type(component.reader).__qualname__}"
            ),
        },
        dependency_builder=lambda component: (
            component_ref(component.spec),
            component_ref(component.fold),
            component_ref(component.operation),
        ),
    )
    add(
        "arti/tensor-invocation-result@2",
        TensorInvocationResult,
        lifecycle=stable,
        variant="independent-reader-output-and-operation-proposal",
        constructible=False,
        artifact_policy="runtime_only",
        config_builder=lambda component: {
            "output_shape": tuple(component.output.shape),
            "reader_steps": component.reader_steps,
        },
        dependency_builder=lambda _component: (
            "arti/tensor-invocation@2",
            "arti/tensor-operation-result@3",
        ),
    )
    add(
        "arti/recall-trace@3",
        RecallTraceV3,
        lifecycle=stable,
        variant="post-transition-neural-exit-trace",
        constructible=False,
        artifact_policy="runtime_only",
        config_builder=lambda component: {"schema_version": component.schema_version},
        dependency_builder=lambda _component: ("arti/refine-exit-request@1",),
    )
    add(
        "arti/recall-route-plan@1",
        RecallRoutePlan,
        lifecycle=stable,
        variant="runtime-only",
        config_schema_version=1,
        config_builder=_fields(
            "schema_version",
            "routing",
            "value_composition",
            "slots",
            "composition_factor",
            "group_size",
            "layout_fingerprint",
        ),
    )
    add(
        "arti/recall-route-stack@1",
        RecallRouteStack,
        lifecycle=stable,
        variant="runtime-only",
        config_schema_version=1,
        config_builder=_route_stack_config,
        dependency_builder=_route_stack_dependencies,
    )
    add(
        "arti/recall-runtime@1",
        RecallRuntime,
        lifecycle=stable,
        variant="values-only-session",
        config_builder=_recall_runtime_config,
        dependency_builder=_recall_runtime_dependencies,
    )
    add(
        "arti/recall-state@1",
        RecallState,
        lifecycle=stable,
        variant="values-only",
        state_schema_version=RECALL_STATE_SCHEMA_VERSION,
        config_builder=_recall_state_config,
    )
    add("arti/updater@1", RecallValueUpdater, lifecycle=stable, config_builder=_updater_config)
    add("arti/affine-updater@1", AffineRecallValueUpdater, lifecycle=stable, config_builder=_affine_updater_config)
    add("arti/normalized-updater@1", NormalizedDeltaRecallValueUpdater, lifecycle=stable, config_builder=_normalised_updater_config)
    add("arti/stacked-updater@1", StackedRecallValueUpdater, lifecycle=stable, config_builder=_fields("site_count", "hidden_dim", "slots", "workspace_dim", "depth", "recall_group_topk"))
    add("arti/tensor-context@1", TensorContext, lifecycle=stable, config_builder=_context_config, factory=TensorContext)
    add("arti/frame-context@1", FrameContext, lifecycle=stable, config_builder=_context_config, factory=FrameContext)
    add(
        "arti/emission-router@1",
        EmissionRouter,
        lifecycle=stable,
        config_builder=_default_config,
        factory=lambda **kwargs: EmissionRouter(
            kwargs.pop("config", None) or EmissionRouterConfig(**kwargs)
        ),
    )
    add("arti/membrane-router@1", MembraneVisibilityRouter, lifecycle="legacy", variant="adapter", config_builder=_default_config)
    add(
        "arti/layer@2",
        ARTILayer,
        lifecycle=stable,
        variant="adaptive-pulse-host",
        config_schema_version=2,
        aliases=("ARTILayer",),
        config_builder=_arti_layer_config,
        dependency_builder=_arti_layer_dependencies,
    )
    add(
        "arti/classic-layer@1",
        LegacyARTILayer,
        lifecycle=stable,
        variant="internal-composed-layer",
        config_builder=_default_config,
    )
    add("arti/layer@1", LegacyARTILayer, lifecycle="legacy", config_builder=_default_config)
    add("arti/latent-tensor-layer@1", ARTILatentTensorLayer, lifecycle=stable, config_builder=_default_config)
    add("arti/dynamic-state@1", ARTIDynamicStateLayer, lifecycle=stable, config_builder=_default_config)
    add("arti/phase-mixer@1", ARTIPhaseMixer, lifecycle=stable, config_builder=_fields("hidden_dim", "operator_count"))
    add("arti/virtual-interface@1", ARTIVirtualInterfaceMixer, lifecycle=stable, config_builder=_fields("scale"))
    add("arti/latent-recall-field@1", ARTILatentRecallField, lifecycle=stable, config_builder=_fields("hidden_dim", "slots", "routing", "key_dim"))
    return registry


def get_component_registry() -> ComponentRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = _build_default_registry()
    return _DEFAULT_REGISTRY


def component_catalog() -> list[dict[str, Any]]:
    """Return the canonical component and alias catalog for release checks."""

    catalog = [dict(item) for item in get_component_registry().catalog()]
    # Formula identities are pure tensor contracts rather than nn.Module
    # registrations, but they belong to the same release-facing catalog.
    from .recall_registry import list_formulas
    from .survival import list_survivals

    for formula in list_formulas():
        catalog.append(
            {
                "kind": "formula",
                "ref": formula.reference,
                "mechanism_id": f"{formula.namespace}/{formula.name}",
                "mechanism_version": formula.version,
                "variant": formula.provider_kind,
                "lifecycle": "stable" if formula.origin in {"builtin", "registered"} else "deprecated",
                "config_schema_version": 1,
                "state_schema_version": 1,
                "aliases": [],
                "deprecated_aliases": [],
                "constructible": True,
                "artifact_policy": "portable",
            }
        )
    for survival in list_survivals():
        catalog.append(
            {
                "kind": "survival",
                "ref": survival.reference,
                "mechanism_id": f"{survival.namespace}/{survival.name}",
                "mechanism_version": survival.version,
                "variant": "survival",
                "lifecycle": "stable",
                "config_schema_version": 1,
                "state_schema_version": 1,
                "aliases": [],
                "deprecated_aliases": [],
                "constructible": True,
                "artifact_policy": "portable",
            }
        )
    return sorted(catalog, key=lambda item: item["ref"])


def register_component(reference: str, **kwargs: Any) -> ComponentRegistration:
    """Register an application-owned component in the process-local registry."""

    return get_component_registry().register(reference, **kwargs)


def resolve_component(reference_or_alias: str, **kwargs: Any) -> Any:
    """Resolve an exact component reference or explicit convenience alias."""

    return get_component_registry().resolve(reference_or_alias, **kwargs)


def component_ref(value: Any) -> str:
    registration = get_component_registry().registration_for(value)
    if registration is None:
        raise UnknownComponentError(f"no component registration for {type(value).__name__}")
    return registration.reference


def _parameter_schema_fingerprint(module: nn.Module) -> str:
    def schema_descriptor(value: Tensor) -> dict[str, Any]:
        try:
            shape: list[int | str] = list(value.shape)
        except RuntimeError:
            shape = ["uninitialized"]
        return {"dtype": str(value.dtype), "shape": shape}

    entries: list[dict[str, Any]] = []
    for name, parameter in module.named_parameters():
        entries.append(
            {
                "kind": "parameter",
                "name": name,
                **schema_descriptor(parameter),
            }
        )
    for name, buffer in module.named_buffers():
        entries.append({"kind": "buffer", "name": name, **schema_descriptor(buffer)})
    return _sha256_json(entries)


def _api_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _direct_registered_dependencies(module: nn.Module, registry: ComponentRegistry) -> tuple[str, ...]:
    result: set[str] = set()
    for child in module.children():
        registration = registry.registration_for(child)
        if registration is not None:
            result.add(registration.reference)
    return tuple(sorted(result))


def _make_component_spec(
    value: Any,
    registration: ComponentRegistration,
    *,
    path: str,
    dependencies: Sequence[str] = (),
) -> ComponentSpec:
    config = registration.config(value)
    return ComponentSpec(
        path=path,
        reference=registration.reference,
        api=_api_name(value),
        variant=registration.variant,
        lifecycle=registration.lifecycle,
        config_schema_version=registration.config_schema_version,
        state_schema_version=registration.state_schema_version,
        config=config,
        config_fingerprint=_sha256_json(config),
        parameter_schema_fingerprint=(
            _parameter_schema_fingerprint(value) if isinstance(value, nn.Module) else _sha256_json(config)
        ),
        dependencies=tuple(sorted(set(dependencies) | set(registration.dependencies(value)))),
        capabilities=registration.capabilities,
    )


def component_spec(value: Any, *, path: str = "$") -> ComponentSpec:
    """Return provenance for one registered component instance."""

    registry = get_component_registry()
    registration = registry.registration_for(value)
    if registration is None:
        raise UnknownComponentError(f"no component registration for {type(value).__name__}")
    dependencies: Sequence[str] = ()
    if isinstance(value, nn.Module):
        dependencies = _direct_registered_dependencies(value, registry)
    return _make_component_spec(value, registration, path=path, dependencies=dependencies)


def component_manifest(model: nn.Module) -> list[dict[str, Any]]:
    """Return the registered component graph for a module tree."""

    if not isinstance(model, nn.Module):
        raise TypeError("component_manifest expects a torch.nn.Module")
    registry = get_component_registry()
    registered: list[tuple[str, nn.Module, ComponentRegistration]] = []
    for raw_path, module in model.named_modules():
        registration = registry.registration_for(module)
        if registration is not None:
            registered.append((raw_path, module, registration))
    specs: list[ComponentSpec] = []
    for raw_path, module, registration in registered:
        child_refs: set[str] = set(registration.dependencies(module))
        for child_path, _child, child_registration in registered:
            if child_path == raw_path:
                continue
            if raw_path:
                if not child_path.startswith(raw_path + "."):
                    continue
                relative = child_path[len(raw_path) + 1 :]
            else:
                relative = child_path
            between = relative.split(".")
            if len(between) == 1:
                child_refs.add(child_registration.reference)
        specs.append(
            _make_component_spec(
                module,
                registration,
                path="$" if raw_path == "" else raw_path,
                dependencies=tuple(child_refs),
            )
        )
    return [spec.to_dict() for spec in sorted(specs, key=lambda item: item.path)]


def component_provenance(model: nn.Module) -> dict[str, Any]:
    components = component_manifest(model)
    return {
        "schema_version": COMPONENT_PROVENANCE_VERSION,
        "components": components,
        "fingerprint": component_graph_fingerprint(components),
    }


def component_graph_fingerprint(components: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_json(list(components))


def _state_schema_entries(state_dict: Mapping[str, Tensor]) -> list[dict[str, Any]]:
    if not isinstance(state_dict, Mapping):
        raise TypeError("state_dict must be a mapping of tensor names to tensors")
    entries: list[dict[str, Any]] = []
    for name, value in state_dict.items():
        if not isinstance(name, str) or not name:
            raise ComponentCompatibilityError("state_dict keys must be non-empty strings")
        if not isinstance(value, Tensor):
            raise ComponentCompatibilityError(f"state_dict value {name!r} must be a tensor")
        if value.layout != torch.strided:
            raise ComponentCompatibilityError(f"state_dict value {name!r} must be strided")
        entries.append({"name": name, "dtype": str(value.dtype), "shape": list(value.shape)})
    return sorted(entries, key=lambda item: item["name"])


def state_dict_schema(state_dict: Mapping[str, Tensor]) -> dict[str, Any]:
    """Describe state names, shapes and dtypes without hashing tensor values."""

    entries = _state_schema_entries(state_dict)
    content = {"schema_version": COMPONENT_STATE_CONTRACT_VERSION, "tensors": entries}
    return {**content, "fingerprint": _sha256_json(content)}


def component_state_contract(
    model: nn.Module,
    state_dict: Mapping[str, Tensor],
    *,
    scope: Literal["all", "trainable", "custom"] = "custom",
) -> dict[str, Any]:
    """Bind a tensor state schema to a model's canonical component graph."""

    if not isinstance(model, nn.Module):
        raise TypeError("model must be a torch.nn.Module")
    if scope not in {"all", "trainable", "custom"}:
        raise ValueError("scope must be 'all', 'trainable', or 'custom'")
    content = {
        "schema_version": COMPONENT_STATE_CONTRACT_VERSION,
        "scope": scope,
        "component_provenance": component_provenance(model),
        "state_schema": state_dict_schema(state_dict),
    }
    return {**content, "fingerprint": _sha256_json(content)}


def validate_component_state_contract(
    value: Mapping[str, Any],
    *,
    state_dict: Mapping[str, Tensor] | None = None,
    model: nn.Module | None = None,
    allow_legacy: bool = False,
) -> dict[str, Any]:
    """Validate a state schema and, when supplied, its model and tensors."""

    if not isinstance(value, Mapping):
        raise ComponentCompatibilityError("component state contract must be a mapping")
    required = {"schema_version", "scope", "component_provenance", "state_schema", "fingerprint"}
    if set(value) != required:
        raise ComponentCompatibilityError("component state contract has missing or unknown fields")
    if value["schema_version"] != COMPONENT_STATE_CONTRACT_VERSION:
        raise ComponentCompatibilityError("unsupported component state contract version")
    if value["scope"] not in {"all", "trainable", "custom"}:
        raise ComponentCompatibilityError("component state contract scope is invalid")
    normalized_provenance = validate_component_provenance(
        value["component_provenance"], allow_legacy=allow_legacy
    )
    state_schema = value["state_schema"]
    if not isinstance(state_schema, Mapping) or set(state_schema) != {"schema_version", "tensors", "fingerprint"}:
        raise ComponentCompatibilityError("component state schema is invalid")
    if state_schema["schema_version"] != COMPONENT_STATE_CONTRACT_VERSION:
        raise ComponentCompatibilityError("unsupported component state schema version")
    if not isinstance(state_schema["tensors"], list) or state_schema["fingerprint"] != _sha256_json(
        {"schema_version": state_schema["schema_version"], "tensors": state_schema["tensors"]}
    ):
        raise ComponentCompatibilityError("component state schema fingerprint is invalid")
    content = {key: value[key] for key in required if key != "fingerprint"}
    if value["fingerprint"] != _sha256_json(content):
        raise ComponentCompatibilityError("component state contract fingerprint is invalid")
    if state_dict is not None and state_dict_schema(state_dict) != dict(state_schema):
        raise ComponentCompatibilityError("component state schema does not match supplied state_dict")
    if model is not None:
        actual_graph = component_provenance(model)
        if actual_graph != normalized_provenance:
            raise ComponentCompatibilityError("component state contract does not match target model")
    return dict(value)


def _is_known_dependency(reference: str, registry: ComponentRegistry) -> bool:
    try:
        registry.registration_for_reference(reference)
        return True
    except UnknownComponentError:
        try:
            from .recall_registry import describe_formula

            describe_formula(reference)
            return True
        except Exception:
            try:
                from .survival import describe_survival

                describe_survival(reference)
                return True
            except Exception:
                return False


def _validate_tensor_operation_dependency_closure(
    reference: str,
    config: Any,
    dependencies: list[str],
) -> bool:
    refs = {
        "arti/tensor-edit-surrogate@3",
        "arti/tensor-operation-query@4",
        "arti/tensor-operation-bank@3",
        "arti/tensor-operation-selector@3",
        "arti/tensor-operation@3",
        "arti/tensor-operation-stop@1",
        "arti/tensor-operation-schedule@1",
        "arti/tensor-operation-loop@3",
    }
    if reference not in refs:
        return False
    if not isinstance(config, Mapping):
        raise ComponentCompatibilityError("TensorOperation config must be a mapping")

    spec_ref = "arti/operable-tensor-port-spec@3"
    expected: set[str]
    if reference == "arti/tensor-edit-surrogate@3":
        if set(config) != {"temperature"} or not _is_finite_positive(config["temperature"]):
            raise ComponentCompatibilityError("TensorEditSurrogate config is invalid")
        expected = {spec_ref, "arti/tensor-edit-formula@3"}
    elif reference == "arti/tensor-operation-query@4":
        required = {
            "ref",
            "key_dim",
            "seed",
            "basis_hash",
            "backing_basis_hash",
            "mask_basis_hash",
            "backing_mask_basis_hash",
            "input_view",
            "fixed",
            "deterministic",
            "stateful",
        }
        if (
            set(config) != required
            or config["ref"] != reference
            or not _is_positive_int(config["key_dim"])
            or type(config["seed"]) is not int
            or not _is_sha256(config["basis_hash"])
            or not _is_sha256(config["backing_basis_hash"])
            or not _is_sha256(config["mask_basis_hash"])
            or not _is_sha256(config["backing_mask_basis_hash"])
            or config["input_view"] != "complete_world_and_backing"
            or config["fixed"] is not True
            or config["deterministic"] is not True
            or config["stateful"] is not False
        ):
            raise ComponentCompatibilityError("TensorOperationQuery config is invalid")
        expected = {spec_ref}
    elif reference == "arti/tensor-operation-bank@3":
        required = {
            "ref",
            "schema_version",
            "candidate_count",
            "key_dim",
            "field",
            "operand_schema",
            "member_ids",
            "bank_ids",
            "group_slices",
            "group_influences",
            "route_normalizer",
            "composition_kind",
            "parent_fingerprints",
        }
        field = config.get("field")
        operands = config.get("operand_schema")
        member_ids = config.get("member_ids")
        bank_ids = config.get("bank_ids")
        group_slices = config.get("group_slices")
        influences = config.get("group_influences")
        candidate_count = config.get("candidate_count")
        if (
            set(config) != required
            or config["ref"] != reference
            or config["schema_version"] != 3
            or not _is_positive_int(candidate_count)
            or not _is_positive_int(config["key_dim"])
            or not isinstance(field, Mapping)
            or set(field) != {
                "ref",
                "support_size",
                "source_capacity",
                "collision_policy",
                "atomic_snapshot",
                "operations",
                "index_dtype",
                "value_dtype",
            }
            or field["ref"] != "arti/tensor-operation-field-spec@2"
            or not _is_positive_int(field["support_size"])
            or not _is_positive_int(field["source_capacity"])
            or field["collision_policy"] != "last_element"
            or field["atomic_snapshot"] is not True
            or tuple(field["operations"]) != ("KEEP", "COPY", "ERASE")
            or field["index_dtype"] != "int64"
            or not isinstance(field["value_dtype"], str)
            or not isinstance(operands, Mapping)
            or set(operands) != {"active", "operation", "source", "destination"}
            or not isinstance(member_ids, (list, tuple))
            or len(member_ids) != candidate_count
            or any(not isinstance(value, str) or not value for value in member_ids)
            or len(set(member_ids)) != len(member_ids)
            or not isinstance(bank_ids, (list, tuple))
            or not bank_ids
            or any(not isinstance(value, str) or not value for value in bank_ids)
            or len(set(bank_ids)) != len(bank_ids)
            or not isinstance(group_slices, (list, tuple))
            or len(group_slices) != len(bank_ids)
            or not isinstance(influences, (list, tuple))
            or len(influences) != len(bank_ids)
            or any(
                type(value) not in {int, float}
                or not math.isfinite(float(value))
                or value < 0
                for value in influences
            )
            or not any(value > 0 for value in influences)
            or config["route_normalizer"] != "per_bank_local"
            or config["composition_kind"] not in {"native", "concat"}
            or not isinstance(config["parent_fingerprints"], (list, tuple))
            or any(not _is_sha256(value) for value in config["parent_fingerprints"])
        ):
            raise ComponentCompatibilityError("TensorOperationBank config is invalid")
        expected_start = 0
        for pair in group_slices:
            if (
                not isinstance(pair, (list, tuple))
                or len(pair) != 2
                or type(pair[0]) is not int
                or type(pair[1]) is not int
                or pair[0] != expected_start
                or pair[1] <= pair[0]
            ):
                raise ComponentCompatibilityError("TensorOperationBank group slices are invalid")
            expected_start = pair[1]
        if expected_start != candidate_count:
            raise ComponentCompatibilityError("TensorOperationBank groups must cover all members")
        expected_shapes = {
            "active": (candidate_count, field["support_size"]),
            "operation": (candidate_count, field["support_size"], 3),
            "source": (
                candidate_count,
                field["support_size"],
                field["source_capacity"],
            ),
        }
        for name, expected_shape in expected_shapes.items():
            entry = operands[name]
            if (
                not isinstance(entry, Mapping)
                or set(entry) != {"shape", "dtype"}
                or tuple(entry["shape"]) != expected_shape
                or entry["dtype"] != field["value_dtype"]
            ):
                raise ComponentCompatibilityError("TensorOperationBank operand schema is invalid")
        destination = operands["destination"]
        destination_shape = (
            tuple(destination.get("shape", ())) if isinstance(destination, Mapping) else ()
        )
        if (
            not isinstance(destination, Mapping)
            or set(destination) != {"shape", "dtype"}
            or len(destination_shape) != 3
            or destination_shape[:2] != (candidate_count, field["support_size"])
            or not _is_positive_int(destination_shape[2])
            or destination["dtype"] != field["value_dtype"]
        ):
            raise ComponentCompatibilityError("TensorOperationBank destination schema is invalid")
        expected = {spec_ref, "arti/tensor-operation-field-spec@2"}
    elif reference == "arti/tensor-operation-selector@3":
        if (
            set(config) != {"estimator", "temperature"}
            or config["estimator"] not in {"hard", "straight-through"}
            or not _is_finite_positive(config["temperature"])
        ):
            raise ComponentCompatibilityError("TensorOperationSelector config is invalid")
        expected = {
            spec_ref,
            "arti/tensor-operation-query@4",
            "arti/tensor-operation-bank@3",
        }
    elif reference == "arti/tensor-operation@3":
        if set(config) != {"surrogate"} or config["surrogate"] not in {
            None,
            "arti/tensor-edit-surrogate@3",
        }:
            raise ComponentCompatibilityError("TensorOperation config is invalid")
        expected = {
            spec_ref,
            "arti/tensor-operation-selector@3",
            "arti/shared-canvas-fold@3",
            "arti/tensor-edit-formula@3",
        }
        if config["surrogate"] is not None:
            expected.add(config["surrogate"])
    elif reference == "arti/tensor-operation-stop@1":
        if (
            set(config) != {"min_operation_steps", "stop_on_stable"}
            or not _is_non_negative_int(config["min_operation_steps"])
            or type(config["stop_on_stable"]) is not bool
        ):
            raise ComponentCompatibilityError("TensorOperationStopPolicy config is invalid")
        expected = set()
    elif reference == "arti/tensor-operation-schedule@1":
        scalar = config.get("scalar_steps")
        maximum = config.get("max_steps")
        if (
            set(config) != {"scalar_steps", "max_steps"}
            or (scalar is not None and not _is_non_negative_int(scalar))
            or (maximum is not None and not _is_non_negative_int(maximum))
            or (scalar is not None and maximum is not None and scalar > maximum)
        ):
            raise ComponentCompatibilityError("TensorOperationSchedule config is invalid")
        expected = set()
    else:
        if set(config) != {"executor"} or config["executor"] not in {
            "static_masked",
            "early_break",
        }:
            raise ComponentCompatibilityError("TensorOperationLoop config is invalid")
        expected = {
            "arti/tensor-operation@3",
            "arti/tensor-operation-stop@1",
        }

    if dependencies != sorted(expected):
        raise ComponentCompatibilityError(
            f"TensorOperation dependency closure is invalid for {reference!r}"
        )
    return True


def _is_non_negative_int(value: Any) -> bool:
    return type(value) is int and value >= 0


def _is_positive_int(value: Any) -> bool:
    return type(value) is int and value > 0


def _is_finite_positive(value: Any) -> bool:
    return type(value) in {int, float} and math.isfinite(float(value)) and value > 0


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _validate_vnext_dependency_closure(
    reference: str,
    config: Any,
    dependencies: list[str],
) -> None:
    """Recompute dependency closure for manifest-owned vNext components."""

    if _validate_tensor_operation_dependency_closure(reference, config, dependencies):
        return

    if reference == "arti/linear-bank-query@1":
        from .tensor_schema import GradientContract, TensorSchema

        required = {
            "input_schema",
            "output_schema",
            "retrieval_contract",
            "normalization_contract",
            "gradient_contract",
            "input_dim",
            "query_dim",
            "bias",
        }
        try:
            if not isinstance(config, Mapping) or set(config) != required:
                raise ValueError("linear Bank Query fields")
            TensorSchema.from_dict(config["input_schema"])
            TensorSchema.from_dict(config["output_schema"])
            gradient = GradientContract.from_dict(config["gradient_contract"])
            if (
                gradient.mode != "autograd"
                or not _is_positive_int(config["input_dim"])
                or not _is_positive_int(config["query_dim"])
                or type(config["bias"]) is not bool
                or not isinstance(config["retrieval_contract"], Mapping)
                or not isinstance(config["normalization_contract"], Mapping)
                or config["normalization_contract"].get("scope") != "bank_local"
            ):
                raise ValueError("linear Bank Query contract")
        except (TypeError, ValueError, KeyError) as exc:
            raise ComponentCompatibilityError(
                "LinearBankQuery config is invalid"
            ) from exc
        expected = sorted(
            {"arti/gradient-contract@1", "arti/tensor-schema@1"}
        )
        if dependencies != expected:
            raise ComponentCompatibilityError(
                "LinearBankQuery dependency closure is invalid"
            )
        return

    if reference == "arti/query-execution-signature@1":
        from .bank_query import QueryExecutionSignature

        try:
            signature = QueryExecutionSignature.from_dict(config)
        except (TypeError, ValueError, KeyError) as exc:
            raise ComponentCompatibilityError(
                "QueryExecutionSignature config is invalid"
            ) from exc
        expected = sorted(
            {
                "arti/gradient-contract@1",
                "arti/tensor-schema@1",
                signature.query_ref,
            }
        )
        if dependencies != expected:
            raise ComponentCompatibilityError(
                "QueryExecutionSignature dependency closure is invalid"
            )
        return

    if reference == "arti/sealed-bank-query@1":
        from .bank_query import QueryExecutionSignature

        try:
            if not isinstance(config, Mapping) or set(config) != {"signature"}:
                raise ValueError("sealed Bank Query fields")
            signature = QueryExecutionSignature.from_dict(config["signature"])
        except (TypeError, ValueError, KeyError) as exc:
            raise ComponentCompatibilityError(
                "SealedBankQuery config is invalid"
            ) from exc
        expected = sorted(
            {"arti/query-execution-signature@1", signature.query_ref}
        )
        if dependencies != expected:
            raise ComponentCompatibilityError(
                "SealedBankQuery dependency closure is invalid"
            )
        return

    if reference == "arti/bank-execution-signature@2":
        from .terminal_abi import BankExecutionSignatureV2

        try:
            signature = BankExecutionSignatureV2.from_dict(config)
        except (TypeError, ValueError, KeyError) as exc:
            raise ComponentCompatibilityError(
                "BankExecutionSignatureV2 config is invalid"
            ) from exc
        expected = sorted(
            {
                "arti/gradient-contract@1",
                "arti/query-execution-signature@1",
                "arti/shape-relation@1",
                "arti/tensor-schema@1",
                "arti/terminal-output-abi@1",
                signature.program_ref,
                signature.query_signature.query_ref,
                signature.terminal_adapter_ref,
                *(
                    ()
                    if signature.local_formula_ref is None
                    else (signature.local_formula_ref,)
                ),
                *(
                    ()
                    if signature.local_refine_ref is None
                    else (signature.local_refine_ref,)
                ),
            }
        )
        if dependencies != expected:
            raise ComponentCompatibilityError(
                "BankExecutionSignatureV2 dependency closure is invalid"
            )
        return

    if reference == "arti/bank-local-refine-policy@1":
        required = {
            "min_steps",
            "max_steps",
            "state_source",
            "exit_semantics",
        }
        if (
            not isinstance(config, Mapping)
            or set(config) != required
            or dependencies
            or not _is_positive_int(config["min_steps"])
            or not _is_positive_int(config["max_steps"])
            or config["max_steps"] < config["min_steps"]
            or config["state_source"] != "latest-local-state"
            or config["exit_semantics"] != "formula-request-after-min-steps"
        ):
            raise ComponentCompatibilityError(
                "BankLocalRefinePolicy config is invalid"
            )
        return

    if reference == "arti/federal-recall@2":
        from .terminal_abi import BankExecutionSignatureV2, TerminalOutputABI

        required = {
            "terminal_abi",
            "root_bank_ids",
            "bank_signatures",
            "max_levels",
            "max_k",
            "winner_policy",
        }
        try:
            if not isinstance(config, Mapping) or set(config) != required:
                raise ValueError("FederalRecall@2 fields")
            TerminalOutputABI.from_dict(config["terminal_abi"])
            roots = config["root_bank_ids"]
            signatures = config["bank_signatures"]
            if (
                not isinstance(roots, list)
                or not roots
                or len(roots) != len(set(roots))
                or any(not isinstance(root, str) or not root for root in roots)
                or not isinstance(signatures, Mapping)
                or not signatures
                or any(root not in signatures for root in roots)
                or not _is_positive_int(config["max_levels"])
                or config["max_k"] != 1
                or type(config["max_k"]) is not int
                or config["winner_policy"] != "hard_one_winner"
            ):
                raise ValueError("FederalRecall@2 contract")
            parsed_signatures = []
            for bank_id, payload in signatures.items():
                if not isinstance(bank_id, str) or not bank_id:
                    raise ValueError("FederalRecall@2 Bank id")
                parsed_signatures.append(BankExecutionSignatureV2.from_dict(payload))
        except (TypeError, ValueError, KeyError) as exc:
            raise ComponentCompatibilityError(
                "FederalRecall@2 config is invalid"
            ) from exc
        expected_set = {
            "arti/bank-execution-signature@2",
            "arti/sealed-bank-query@1",
            "arti/terminal-output-abi@1",
        }
        for signature in parsed_signatures:
            expected_set.update(
                _bank_execution_signature_v2_dependencies(signature)
            )
        expected = sorted(expected_set)
        if dependencies != expected:
            raise ComponentCompatibilityError(
                "FederalRecall@2 dependency closure is invalid"
            )
        return

    formula_atom_refs = {
        "arti/formula-atom-contract@1",
        "arti/formula-atom-scale@1",
        "arti/formula-atom-add@1",
        "arti/formula-atom-reduce@1",
        "arti/formula-atom-reshape@1",
        "arti/formula-atom-permute@1",
        "arti/formula-atom-gather@1",
        "arti/formula-atom-scatter@1",
    }
    formula_instruction_refs = formula_atom_refs | {
        "arti/fold@2",
        "arti/unfold@2",
    }
    if reference == "arti/formula-operand-bank@1":
        required = {
            "source_ref",
            "asset_fingerprint",
            "bundle_id",
            "member_ids",
            "candidate_count",
            "key_dim",
            "key_dtype",
            "operand_shapes",
            "operand_dtypes",
        }
        if not isinstance(config, Mapping) or set(config) != required or dependencies:
            raise ComponentCompatibilityError("FormulaOperandBank config is incomplete")
        member_ids = config["member_ids"]
        shapes = config["operand_shapes"]
        dtypes = config["operand_dtypes"]
        candidate_count = config["candidate_count"]
        key_dim = config["key_dim"]
        if (
            not isinstance(config["source_ref"], str)
            or not config["source_ref"]
            or (
                config["asset_fingerprint"] is not None
                and (
                    not isinstance(config["asset_fingerprint"], str)
                    or len(config["asset_fingerprint"]) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in config["asset_fingerprint"]
                    )
                )
            )
            or not isinstance(config["bundle_id"], str)
            or not config["bundle_id"]
            or isinstance(candidate_count, bool)
            or not isinstance(candidate_count, int)
            or candidate_count <= 0
            or isinstance(key_dim, bool)
            or not isinstance(key_dim, int)
            or key_dim <= 0
            or not isinstance(config["key_dtype"], str)
            or not isinstance(member_ids, list)
            or len(member_ids) != candidate_count
            or len(set(member_ids)) != candidate_count
            or any(not isinstance(item, str) or not item for item in member_ids)
            or not isinstance(shapes, Mapping)
            or not shapes
            or not isinstance(dtypes, Mapping)
            or set(shapes) != set(dtypes)
        ):
            raise ComponentCompatibilityError("FormulaOperandBank config is invalid")
        try:
            ComponentRef.parse(config["source_ref"])
        except InvalidComponentRefError as exc:
            raise ComponentCompatibilityError(
                "FormulaOperandBank source reference is invalid"
            ) from exc
        for name, shape in shapes.items():
            if (
                not isinstance(name, str)
                or not name
                or not isinstance(shape, list)
                or not shape
                or shape[0] != candidate_count
                or any(
                    isinstance(size, bool) or not isinstance(size, int) or size <= 0
                    for size in shape
                )
                or not isinstance(dtypes[name], str)
                or not dtypes[name]
            ):
                raise ComponentCompatibilityError("FormulaOperandBank operand schema is invalid")
        return
    if reference in {"arti/formula-fabric@2", "arti/formula-execution-plan@1"}:
        from .formula_v2 import FormulaProgram

        required = {"program", "program_fingerprint"}
        if reference == "arti/formula-execution-plan@1":
            required.update(
                {
                    "schema_ref",
                    "schema_version",
                    "binding_names",
                }
            )
        if not isinstance(config, Mapping) or set(config) != required:
            raise ComponentCompatibilityError("Formula execution config is incomplete")
        try:
            program = FormulaProgram.from_dict(config["program"])
        except (TypeError, ValueError, KeyError) as exc:
            raise ComponentCompatibilityError("Formula execution program is invalid") from exc
        if config["program_fingerprint"] != program.fingerprint:
            raise ComponentCompatibilityError("Formula execution program fingerprint is invalid")
        if reference == "arti/formula-execution-plan@1":
            from .formula_v2 import (
                FORMULA_EXECUTION_PLAN_V1_SCHEMA_REF,
                FORMULA_EXECUTION_PLAN_V1_SCHEMA_VERSION,
            )

            if (
                config["schema_ref"] != FORMULA_EXECUTION_PLAN_V1_SCHEMA_REF
                or config["schema_version"]
                != FORMULA_EXECUTION_PLAN_V1_SCHEMA_VERSION
            ):
                raise ComponentCompatibilityError("Formula execution plan schema is invalid")
            if config["binding_names"] != [binding.name for binding in program.bindings]:
                raise ComponentCompatibilityError("Formula execution binding order is invalid")
        expected_dependencies = sorted({item.atom_ref for item in program.instructions})
        if dependencies != expected_dependencies or not set(dependencies).issubset(
            formula_instruction_refs
        ):
            raise ComponentCompatibilityError("Formula execution dependency closure is invalid")
        return
    if reference in formula_atom_refs:
        from .formula_v2 import (
            AddAtom,
            ContractAtom,
            GatherAtom,
            PermuteAtom,
            ReduceAtom,
            ReshapeAtom,
            ScaleAtom,
            ScatterAtom,
            TensorType,
        )

        if not isinstance(config, Mapping) or dependencies:
            raise ComponentCompatibilityError("Formula atom config or dependency closure is invalid")
        try:
            if reference == "arti/formula-atom-contract@1":
                required = {
                    "left_type",
                    "right_type",
                    "output_type",
                    "reduce_axes",
                    "output_axes",
                    "accumulation_dtype",
                }
                if set(config) != required:
                    raise ValueError("contract config fields")
                atom = ContractAtom(
                    TensorType.from_dict(config["left_type"]),
                    TensorType.from_dict(config["right_type"]),
                    reduce_axes=config["reduce_axes"],
                    output_axes=config["output_axes"],
                    accumulation_dtype=config["accumulation_dtype"],
                )
                if atom.output_type.to_dict() != config["output_type"]:
                    raise ValueError("contract output type")
            elif reference == "arti/formula-atom-scale@1":
                if set(config) != {"value_type", "factor_type", "accumulation_dtype"}:
                    raise ValueError("scale config fields")
                ScaleAtom(
                    TensorType.from_dict(config["value_type"]),
                    TensorType.from_dict(config["factor_type"]),
                    accumulation_dtype=config["accumulation_dtype"],
                )
            elif reference == "arti/formula-atom-add@1":
                if set(config) != {"value_type", "accumulation_dtype"}:
                    raise ValueError("add config fields")
                AddAtom(
                    TensorType.from_dict(config["value_type"]),
                    accumulation_dtype=config["accumulation_dtype"],
                )
            elif reference == "arti/formula-atom-reduce@1":
                required = {
                    "value_type",
                    "output_type",
                    "axis",
                    "mode",
                    "accumulation_dtype",
                }
                if set(config) != required or config["mode"] != "sum":
                    raise ValueError("reduce config fields")
                atom = ReduceAtom(
                    TensorType.from_dict(config["value_type"]),
                    axis=config["axis"],
                    accumulation_dtype=config["accumulation_dtype"],
                )
                if atom.output_type.to_dict() != config["output_type"]:
                    raise ValueError("reduce output type")
            elif reference == "arti/formula-atom-reshape@1":
                required = {
                    "value_type",
                    "output_type",
                    "output_axes",
                    "output_sizes",
                }
                if set(config) != required:
                    raise ValueError("reshape config fields")
                atom = ReshapeAtom(
                    TensorType.from_dict(config["value_type"]),
                    output_axes=config["output_axes"],
                    output_sizes=config["output_sizes"],
                )
                if atom.output_type.to_dict() != config["output_type"]:
                    raise ValueError("reshape output type")
            elif reference == "arti/formula-atom-permute@1":
                required = {"value_type", "output_type", "output_axes"}
                if set(config) != required:
                    raise ValueError("permute config fields")
                atom = PermuteAtom(
                    TensorType.from_dict(config["value_type"]),
                    output_axes=config["output_axes"],
                )
                if atom.output_type.to_dict() != config["output_type"]:
                    raise ValueError("permute output type")
            elif reference == "arti/formula-atom-gather@1":
                required = {
                    "value_type",
                    "index_type",
                    "output_type",
                    "axis",
                    "index_axis",
                }
                if set(config) != required:
                    raise ValueError("gather config fields")
                atom = GatherAtom(
                    TensorType.from_dict(config["value_type"]),
                    TensorType.from_dict(config["index_type"]),
                    axis=config["axis"],
                    index_axis=config["index_axis"],
                )
                if atom.output_type.to_dict() != config["output_type"]:
                    raise ValueError("gather output type")
            elif reference == "arti/formula-atom-scatter@1":
                required = {
                    "base_type",
                    "index_type",
                    "update_type",
                    "output_type",
                    "axis",
                    "index_axis",
                    "mode",
                }
                if set(config) != required or config["mode"] != "replace":
                    raise ValueError("scatter config fields")
                atom = ScatterAtom(
                    TensorType.from_dict(config["base_type"]),
                    TensorType.from_dict(config["index_type"]),
                    TensorType.from_dict(config["update_type"]),
                    axis=config["axis"],
                    index_axis=config["index_axis"],
                )
                if atom.output_type.to_dict() != config["output_type"]:
                    raise ValueError("scatter output type")
            else:
                raise ValueError("unknown Formula atom reference")
        except (TypeError, ValueError, KeyError) as exc:
            raise ComponentCompatibilityError("Formula atom config is invalid") from exc
        return

    if reference == "arti/batched-refine-operation@1":
        required = {"operation_ref", "operation_config_fingerprint"}
        if not isinstance(config, Mapping) or set(config) != required:
            raise ComponentCompatibilityError(
                "BatchedRefineOperation config is incomplete"
            )
        operation_ref = config["operation_ref"]
        operation_fingerprint = config["operation_config_fingerprint"]
        if (
            not isinstance(operation_ref, str)
            or not operation_ref.startswith("arti/")
            or "@" not in operation_ref
            or not isinstance(operation_fingerprint, str)
            or len(operation_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in operation_fingerprint
            )
            or dependencies != [operation_ref]
        ):
            raise ComponentCompatibilityError(
                "BatchedRefineOperation dependency closure is invalid"
            )
        return
    if reference == "arti/batched-refine-plan@1":
        required = {
            "schema_version",
            "execution_layout",
            "config_fingerprint",
            "operation_ref",
        }
        if not isinstance(config, Mapping) or set(config) != required:
            raise ComponentCompatibilityError("BatchedRefinePlan config is incomplete")
        fingerprint = config["config_fingerprint"]
        operation_ref = config["operation_ref"]
        if (
            config["schema_version"] != 1
            or config["execution_layout"] not in {"static_capacity", "packed_active"}
            or not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise ComponentCompatibilityError("BatchedRefinePlan config is invalid")
        expected = [] if operation_ref is None else ["arti/batched-refine-operation@1"]
        if operation_ref is not None and (
            not isinstance(operation_ref, str)
            or not operation_ref.startswith("arti/")
            or "@" not in operation_ref
        ):
            raise ComponentCompatibilityError(
                "BatchedRefinePlan operation reference is invalid"
            )
        if dependencies != expected:
            raise ComponentCompatibilityError(
                "BatchedRefinePlan dependency closure is invalid"
            )
        return

    if reference == "arti/batched-refine-result@1":
        required = {
            "schema_version",
            "candidate_ref",
            "candidate_config",
            "candidate_config_fingerprint",
            "plan_ref",
            "plan_config_fingerprint",
            "execution_layout",
            "operation_wrapper_ref",
            "operation_ref",
            "formula_route_fingerprint",
            "topology_refs",
            "topology_contract_fingerprints",
            "branch_policy_fingerprint",
            "execution_rng_fingerprint",
            "execution_rng_stream_key",
            "execution_rng_domains",
            "max_k",
        }
        if not isinstance(config, Mapping) or set(config) != required:
            raise ComponentCompatibilityError(
                "BatchedRefineResult config is incomplete"
            )
        if config["candidate_config_fingerprint"] != _sha256_json(
            config["candidate_config"]
        ):
            raise ComponentCompatibilityError(
                "BatchedRefineResult candidate config fingerprint is invalid"
            )
        if config["execution_layout"] not in {"static_capacity", "packed_active"}:
            raise ComponentCompatibilityError(
                "BatchedRefineResult execution layout is invalid"
            )
        candidate = config["candidate_config"]
        if not isinstance(candidate, Mapping):
            raise ComponentCompatibilityError(
                "BatchedRefineResult candidate config must be a mapping"
            )
        expected = {
            config["candidate_ref"],
            candidate.get("source_ref"),
            candidate.get("formula_ref"),
            config["plan_ref"],
        }
        branch_policy_fingerprint = config["branch_policy_fingerprint"]
        if branch_policy_fingerprint is not None:
            if (
                not isinstance(branch_policy_fingerprint, str)
                or len(branch_policy_fingerprint) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in branch_policy_fingerprint
                )
            ):
                raise ComponentCompatibilityError(
                    "BatchedRefineResult branch policy fingerprint is invalid"
                )
            expected.add("arti/branch-refine-policy@1")
        rng_fingerprint = config["execution_rng_fingerprint"]
        rng_stream = config["execution_rng_stream_key"]
        rng_domains = config["execution_rng_domains"]
        if (rng_fingerprint is None) != (rng_stream is None):
            raise ComponentCompatibilityError(
                "BatchedRefineResult RNG identity is incomplete"
            )
        if rng_fingerprint is None:
            if rng_domains != []:
                raise ComponentCompatibilityError(
                    "deterministic BatchedRefineResult cannot consume RNG domains"
                )
        elif (
            not isinstance(rng_fingerprint, str)
            or len(rng_fingerprint) != 64
            or not isinstance(rng_stream, str)
            or not rng_stream
            or not isinstance(rng_domains, list)
            or not rng_domains
        ):
            raise ComponentCompatibilityError(
                "BatchedRefineResult RNG identity is invalid"
            )
        operation_ref = config["operation_ref"]
        wrapper_ref = config["operation_wrapper_ref"]
        topology_refs = config["topology_refs"]
        topology_contracts = config["topology_contract_fingerprints"]
        if (
            not isinstance(topology_refs, list)
            or any(
                not isinstance(reference, str)
                or not reference.startswith("arti/")
                or "@" not in reference
                for reference in topology_refs
            )
            or not isinstance(topology_contracts, list)
            or any(
                not isinstance(fingerprint, str)
                or len(fingerprint) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in fingerprint
                )
                for fingerprint in topology_contracts
            )
        ):
            raise ComponentCompatibilityError(
                "BatchedRefineResult topology lineage is invalid"
            )
        expected.update(topology_refs)
        if operation_ref is None:
            if wrapper_ref is not None:
                raise ComponentCompatibilityError(
                    "BatchedRefineResult operation wrapper requires an operation"
                )
        else:
            if wrapper_ref != "arti/batched-refine-operation@1":
                raise ComponentCompatibilityError(
                    "BatchedRefineResult operation wrapper is invalid"
                )
            expected.update({wrapper_ref, operation_ref})
        if None in expected or dependencies != sorted(expected):
            raise ComponentCompatibilityError(
                "BatchedRefineResult dependency closure is invalid"
            )
        return
    if reference not in {
        "arti/pulse-stage-graph@1",
        "arti/pulse-executor@1",
        "arti/pulse@2",
    }:
        return
    from .vnext_contracts import PulseStageGraph

    if reference == "arti/pulse-stage-graph@1":
        graph = PulseStageGraph.from_dict(config)
    elif reference == "arti/pulse-executor@1":
        required = {
            "manifest_ref",
            "manifest",
            "manifest_fingerprint",
            "enabled_stage_ids",
        }
        if not isinstance(config, Mapping) or set(config) != required:
            raise ComponentCompatibilityError("PulseExecutor config is incomplete")
        if config["manifest_ref"] != "arti/pulse-stage-graph@1":
            raise ComponentCompatibilityError("PulseExecutor manifest reference is invalid")
        graph = PulseStageGraph.from_dict(config["manifest"])
        if config["manifest_fingerprint"] != graph.fingerprint:
            raise ComponentCompatibilityError("PulseExecutor manifest fingerprint is invalid")
        expected_ids = sorted(
            stage.stage_id for stage in graph.stages if stage.mode.value == "enabled"
        )
        if config["enabled_stage_ids"] != expected_ids:
            raise ComponentCompatibilityError("PulseExecutor enabled stages are invalid")
    else:
        required = {"manifest_ref", "manifest", "manifest_fingerprint"}
        if not isinstance(config, Mapping) or set(config) != required:
            raise ComponentCompatibilityError("Pulse@2 config is incomplete")
        if config["manifest_ref"] != "arti/pulse-stage-graph@1":
            raise ComponentCompatibilityError("Pulse@2 manifest reference is invalid")
        graph = PulseStageGraph.from_dict(config["manifest"])
        if config["manifest_fingerprint"] != graph.fingerprint:
            raise ComponentCompatibilityError("Pulse@2 manifest fingerprint is invalid")
    root_dependency = (
        "arti/pulse-stage@1"
        if reference == "arti/pulse-stage-graph@1"
        else "arti/pulse-stage-graph@1"
    )
    expected_dependencies = sorted({root_dependency, *graph.enabled_dependencies})
    if dependencies != expected_dependencies:
        raise ComponentCompatibilityError(
            f"component dependency closure is invalid for {reference!r}"
        )


def validate_component_provenance(
    value: Mapping[str, Any],
    *,
    allow_legacy: bool = False,
    artifact_scope: bool = False,
) -> dict[str, Any]:
    """Validate an artifact component graph against the local registry."""

    if not isinstance(value, Mapping):
        raise ComponentCompatibilityError("component provenance must be a mapping")
    if set(value) != {"schema_version", "components", "fingerprint"}:
        raise ComponentCompatibilityError(
            "component provenance has missing or unknown top-level fields"
        )
    schema_version = value.get("schema_version")
    if type(schema_version) is not int or schema_version not in {1, COMPONENT_PROVENANCE_VERSION}:
        raise ComponentCompatibilityError("unsupported component provenance schema version")
    components = value.get("components")
    if not isinstance(components, list) or any(not isinstance(item, Mapping) for item in components):
        raise ComponentCompatibilityError("component provenance components must be a list of mappings")
    fingerprint = value.get("fingerprint")
    if not isinstance(fingerprint, str) or fingerprint != component_graph_fingerprint(components):
        raise ComponentCompatibilityError("component provenance fingerprint is invalid")
    registry = get_component_registry()
    normalized: list[dict[str, Any]] = []
    paths: set[str] = set()
    for item in components:
        required = {
            "path",
            "api",
            "ref",
            "mechanism_id",
            "mechanism_version",
            "variant",
            "lifecycle",
            "config_schema_version",
            "state_schema_version",
            "config",
            "config_fingerprint",
            "parameter_schema_fingerprint",
            "dependencies",
        }
        if schema_version >= 2:
            required.add("capabilities")
        if set(item) != required:
            raise ComponentCompatibilityError("component provenance entry has missing or unknown fields")
        path = item["path"]
        if not isinstance(path, str) or not path or path in paths:
            raise ComponentCompatibilityError("component provenance paths must be unique non-empty strings")
        paths.add(path)
        identity = ComponentRef.parse(item["ref"])
        if item["mechanism_id"] != identity.mechanism_id or item["mechanism_version"] != identity.version:
            raise ComponentCompatibilityError(f"component identity fields disagree at {path!r}")
        try:
            registration = registry.registration_for_reference(identity.reference)
        except UnknownComponentError as error:
            raise ComponentCompatibilityError(
                f"unknown component reference in artifact: {identity.reference!r}"
            ) from error
        if registration.lifecycle in {"legacy", "deprecated"} and not allow_legacy:
            raise ComponentCompatibilityError(
                f"component {identity.reference!r} is {registration.lifecycle}; pass allow_legacy=True explicitly"
            )
        if artifact_scope and registration.artifact_policy != "portable":
            raise ComponentCompatibilityError(
                f"component {identity.reference!r} has artifact_policy="
                f"{registration.artifact_policy!r} and cannot appear in arti.st"
            )
        if item["variant"] != registration.variant or item["lifecycle"] != registration.lifecycle:
            raise ComponentCompatibilityError(f"component lifecycle or variant drift at {path!r}")
        if item["config_schema_version"] != registration.config_schema_version or item["state_schema_version"] != registration.state_schema_version:
            raise ComponentCompatibilityError(f"component schema version drift at {path!r}")
        capabilities = (
            item["capabilities"]
            if schema_version >= 2
            else list(registration.capabilities)
        )
        if capabilities != list(registration.capabilities):
            raise ComponentCompatibilityError(f"component capability drift at {path!r}")
        if item["config_fingerprint"] != _sha256_json(item["config"]):
            raise ComponentCompatibilityError(f"component config fingerprint is invalid at {path!r}")
        dependencies = item["dependencies"]
        if not isinstance(dependencies, list) or any(not isinstance(dep, str) or not _is_known_dependency(dep, registry) for dep in dependencies):
            raise ComponentCompatibilityError(f"component dependencies are invalid at {path!r}")
        _validate_vnext_dependency_closure(identity.reference, item["config"], dependencies)
        normalized.append({**dict(item), "capabilities": capabilities})
    normalized_fingerprint = component_graph_fingerprint(normalized)
    return {
        "schema_version": COMPONENT_PROVENANCE_VERSION,
        "components": normalized,
        "fingerprint": normalized_fingerprint,
    }


def verify_component_provenance(
    model: nn.Module,
    expected: Mapping[str, Any],
    *,
    allow_legacy: bool = False,
) -> None:
    normalized_expected = validate_component_provenance(
        expected, allow_legacy=allow_legacy
    )
    actual = component_provenance(model)
    if actual != normalized_expected:
        raise ComponentCompatibilityError(
            "arti.st component provenance does not match target model; "
            "construct the exact component versions/configuration or use an explicit migration"
        )


__all__ = [
    "COMPONENT_PROVENANCE_VERSION",
    "COMPONENT_STATE_CONTRACT_VERSION",
    "ArtifactPolicy",
    "ComponentCompatibilityError",
    "ComponentLifecycle",
    "ComponentRef",
    "ComponentRegistration",
    "ComponentRegistry",
    "ComponentRegistryError",
    "ComponentSpec",
    "DuplicateComponentError",
    "InvalidComponentRefError",
    "UnknownComponentError",
    "component_graph_fingerprint",
    "component_catalog",
    "component_manifest",
    "component_provenance",
    "component_ref",
    "component_spec",
    "component_state_contract",
    "get_component_registry",
    "register_component",
    "resolve_component",
    "validate_component_provenance",
    "validate_component_state_contract",
    "verify_component_provenance",
    "state_dict_schema",
]
