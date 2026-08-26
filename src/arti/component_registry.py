"""Small, explicit registry for public ARTI component contracts.

The registry identifies public tensor modules without importing classes from an
artifact.  Applications may register their own modules explicitly; ordinary
``torch.nn.Module`` composition remains valid without registration.
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


COMPONENT_STATE_CONTRACT_VERSION = 1
COMPONENT_PROVENANCE_VERSION = 2
ComponentLifecycle = Literal["stable", "alpha", "legacy", "deprecated"]
_LIFECYCLES = frozenset({"stable", "alpha", "legacy", "deprecated"})
_NAME = r"[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?"
_REFERENCE = re.compile(rf"^(?P<namespace>{_NAME})/(?P<name>{_NAME})@(?P<version>[1-9][0-9]*)$")


class ComponentRegistryError(ValueError):
    """Base error for component registry failures."""


class InvalidComponentRefError(ComponentRegistryError):
    """Raised when a component reference is not canonical."""


class DuplicateComponentError(ComponentRegistryError):
    """Raised when an identity or alias is registered twice."""


class UnknownComponentError(ComponentRegistryError):
    """Raised when an exact component identity is not registered."""


class ComponentCompatibilityError(ComponentRegistryError):
    """Raised when a serialized component contract cannot be trusted."""


def _normalize(value: Any) -> Any:
    if isinstance(value, Tensor):
        try:
            shape: list[int | str] = list(value.shape)
        except RuntimeError:
            shape = ["uninitialized"]
        return {"__tensor__": {"dtype": str(value.dtype), "shape": shape, "device": str(value.device)}}
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
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    return str(value)


def _canonical_json(value: Any) -> str:
    return json.dumps(_normalize(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _default_config(component: Any) -> Mapping[str, Any]:
    candidate = getattr(component, "serialization_config", None)
    if callable(candidate):
        value = candidate()
        if isinstance(value, Mapping):
            return value
    candidate = getattr(component, "config", None)
    if is_dataclass(candidate) or isinstance(candidate, Mapping):
        return candidate
    return {}


def _fields(*names: str) -> Callable[[Any], Mapping[str, Any]]:
    def build(component: Any) -> Mapping[str, Any]:
        result: dict[str, Any] = {}
        for name in names:
            value = getattr(component, name, None)
            if isinstance(value, (str, bool, int, float)) or value is None:
                result[name] = value
        return result

    return build


@dataclass(frozen=True, order=True)
class ComponentRef:
    """Canonical ``namespace/name@version`` identity."""

    namespace: str
    name: str
    version: int

    def __post_init__(self) -> None:
        if re.fullmatch(_NAME, self.namespace) is None or re.fullmatch(_NAME, self.name) is None:
            raise InvalidComponentRefError("component namespace and name are invalid")
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
            raise InvalidComponentRefError("component reference must use namespace/name@version syntax")
        return cls(match.group("namespace"), match.group("name"), int(match.group("version")))


ConfigBuilder = Callable[[Any], Mapping[str, Any]]
DependencyBuilder = Callable[[Any], Sequence[str]]
Factory = Callable[..., Any]


@dataclass(frozen=True)
class ComponentRegistration:
    identity: ComponentRef
    component_type: type[Any]
    lifecycle: ComponentLifecycle
    variant: str = "default"
    config_schema_version: int = 1
    state_schema_version: int = 1
    aliases: tuple[str, ...] = ()
    deprecated_aliases: tuple[str, ...] = ()
    factory: Factory | None = None
    config_builder: ConfigBuilder | None = None
    dependency_builder: DependencyBuilder | None = None
    capabilities: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if set(self.aliases) & set(self.deprecated_aliases):
            raise ValueError("component aliases cannot also be deprecated aliases")
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
        if self.dependency_builder is None:
            return ()
        values = tuple(self.dependency_builder(component))
        if any(not isinstance(value, str) for value in values):
            raise ComponentCompatibilityError(f"dependencies for {self.reference} must be strings")
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
            "capabilities": list(self.capabilities),
        }


@dataclass(frozen=True)
class ComponentSpec:
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

    def to_dict(self) -> dict[str, Any]:
        identity = ComponentRef.parse(self.reference)
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


class ComponentRegistry:
    """Thread-safe exact registry for public component contracts."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._by_reference: dict[str, ComponentRegistration] = {}
        self._by_alias: dict[str, ComponentRegistration] = {}

    def register(
        self,
        reference: str,
        *,
        component_type: type[Any],
        lifecycle: ComponentLifecycle = "alpha",
        variant: str = "default",
        aliases: Sequence[str] = (),
        deprecated_aliases: Sequence[str] = (),
        config_schema_version: int = 1,
        state_schema_version: int = 1,
        factory: Factory | None = None,
        config_builder: ConfigBuilder | None = None,
        dependency_builder: DependencyBuilder | None = None,
        capabilities: Sequence[str] = (),
    ) -> ComponentRegistration:
        identity = ComponentRef.parse(reference)
        if lifecycle not in _LIFECYCLES:
            raise ValueError(f"unsupported component lifecycle: {lifecycle!r}")
        registration = ComponentRegistration(
            identity=identity,
            component_type=component_type,
            lifecycle=lifecycle,
            variant=variant,
            aliases=tuple(aliases),
            deprecated_aliases=tuple(deprecated_aliases),
            config_schema_version=config_schema_version,
            state_schema_version=state_schema_version,
            factory=component_type if factory is None else factory,
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
            if any(alias in self._by_alias or alias in self._by_reference for alias in aliases_to_add):
                raise DuplicateComponentError(f"component alias is already registered: {sorted(aliases_to_add)}")
            self._by_reference[reference] = registration
            for alias in aliases_to_add:
                self._by_alias[alias] = registration
        return registration

    def registration_for_reference(self, reference: str) -> ComponentRegistration:
        try:
            return self._by_reference[reference]
        except KeyError as error:
            raise UnknownComponentError(f"unknown component reference: {reference!r}") from error

    def resolve_registration(self, reference_or_alias: str) -> ComponentRegistration:
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
        return None

    def registrations(self) -> tuple[ComponentRegistration, ...]:
        return tuple(sorted(self._by_reference.values(), key=lambda item: item.reference))

    def catalog(self) -> tuple[dict[str, Any], ...]:
        return tuple(registration.to_dict() for registration in self.registrations())


_DEFAULT_REGISTRY: ComponentRegistry | None = None


def _build_default_registry() -> ComponentRegistry:
    registry = ComponentRegistry()
    from .adaptive_pulse import AdaptivePulse
    from .aggregate import ReunionAggregate, SoftFoldAggregate
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
        IterativeRoutedFormulaFabricCompute,
        RoutedFormulaFabricCompute,
    )
    from .layers import ARTIDynamicStateLayer, ARTILatentRecallField, ARTILatentTensorLayer, ARTILayer, ARTIPhaseMixer, ARTIVirtualInterfaceMixer
    from .nn import Fold, FusionPulse, Half, LearnedPulse, Recall, RecallRefiner, UnFold
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
    from .pulse import PulseCompressor
    from .recall_refine import RefineBudget, RefinePolicy, RefineStop
    from .target_bank import TargetBankUpdater, WriteRefinePolicy
    from .objective_bank import ObjectiveExposureBank
    from .objective_formula import ObjectiveFormulaFabricCompute
    from .selective_recall import SelectiveRecallKernel
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
    from .vnext_contracts import (
        PULSE_STAGE_GRAPH_SCHEMA_VERSION,
        PULSE_STAGE_SCHEMA_VERSION,
        PulseStageGraph,
        PulseStageSpec,
    )
    from .vnext_pipeline import PulseExecutor

    def add(reference: str, component_type: type[Any], **kwargs: Any) -> None:
        registry.register(reference, component_type=component_type, **kwargs)

    def coupled_target_bank_updater_factory(**kwargs: Any) -> TargetBankUpdater:
        requested = kwargs.get("target_coupling", "required_after_bootstrap")
        if requested != "required_after_bootstrap":
            raise ValueError(
                "arti/target-bank-updater@2 requires "
                "target_coupling='required_after_bootstrap'"
            )
        kwargs["target_coupling"] = "required_after_bootstrap"
        return TargetBankUpdater(**kwargs)

    def target_bank_config(component: TargetBankUpdater) -> Mapping[str, Any]:
        return {
            "hidden_dim": component.hidden_dim,
            "slots": component.slots,
            "workspace_dim": component.workspace_dim,
            "private_slots": component.private_slots,
            "query_seed": component.query_seed,
            "target_coupling": component.target_coupling,
            "policy": component.policy,
        }

    def target_bank_dependencies(component: TargetBankUpdater) -> Sequence[str]:
        result = ["arti/write-refine-policy@1", "arti/refine-budget@1"]
        if component.policy.stop is not None:
            result.append("arti/refine-stop@1")
        return result

    def adaptive_pulse_config(component: AdaptivePulse) -> Mapping[str, Any]:
        component._validate_manifest_binding()
        return {
            "manifest_ref": "arti/pulse-stage-graph@1",
            "manifest": component.manifest.to_dict(),
            "manifest_fingerprint": component.manifest.fingerprint,
        }

    def adaptive_pulse_dependencies(component: AdaptivePulse) -> Sequence[str]:
        component._validate_manifest_binding()
        return ("arti/pulse-stage-graph@1", *component.enabled_components)

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

    def legacy_recall_factory(**kwargs: Any) -> Recall:
        component = Recall(**kwargs)
        component._component_reference = "arti/recall@1"
        return component

    def half_config(component: Half) -> Mapping[str, Any]:
        config: dict[str, Any] = {
            "threshold": component._threshold_init,
            "base": component._base_init,
            "scale": component._scale_init,
            "stochastic": component.stochastic,
            "learnable": component.learnable,
        }
        metadata = getattr(component, "survival_metadata", None)
        if isinstance(metadata, Mapping):
            if metadata.get("runtime_only"):
                raise ComponentCompatibilityError(
                    "runtime-only survival implementations cannot be stored in an artifact"
                )
            config["survival"] = dict(metadata)
        else:
            config["survival"] = {
                "ref": "arti/survival@1",
                "origin": "builtin",
                "portable": True,
                "runtime_only": False,
                "config": {
                    "threshold": component._threshold_init,
                    "base": component._base_init,
                    "scale": component._scale_init,
                    "learnable": component.learnable,
                },
            }
        if component.context_mode != "none":
            config.update(
                {
                    "context_mode": component.context_mode,
                    "context_axes": list(component.context_axes),
                    "context_gain": component.context_gain,
                }
            )
        return config

    def half_dependencies(component: Half) -> Sequence[str]:
        metadata = component.survival_metadata
        reference = metadata.get("ref")
        return (reference,) if isinstance(reference, str) else ()

    def recall_dependencies(component: Recall) -> Sequence[str]:
        formula = getattr(component, "formula_id", None)
        result = [formula] if isinstance(formula, str) and formula != "custom" else []
        state = getattr(component, "state", None)
        config = getattr(state, "config", None)
        if getattr(config, "recall_activation", None) == "half":
            result.append("arti/half@1")
        return result

    def refiner_dependencies(component: RecallRefiner) -> Sequence[str]:
        result: list[str] = []
        for child in (
            getattr(component, "recall_layer", None),
            getattr(component, "activation", None),
        ):
            registration = registry.registration_for(child)
            if registration is not None:
                result.append(registration.reference)
        return result

    alpha = "alpha"
    add(
        "arti/half@1",
        Half,
        lifecycle=alpha,
        capabilities=("pulse.stage.half",),
        config_schema_version=2,
        factory=scalar_half_factory,
        config_builder=half_config,
        dependency_builder=half_dependencies,
    )
    add(
        "arti/half@2",
        Half,
        lifecycle=alpha,
        capabilities=("pulse.stage.half",),
        variant="contextual",
        config_schema_version=3,
        factory=contextual_half_factory,
        config_builder=half_config,
        dependency_builder=half_dependencies,
    )
    add("arti/fold@1", Fold, lifecycle=alpha, config_builder=_fields("k", "dim", "hidden_dim", "temperature", "mode", "topk", "heads", "eps"))
    add("arti/unfold@1", UnFold, lifecycle=alpha, config_builder=_fields("dim", "exposed", "guide_dim", "condition_dim", "hidden_dim", "temperature", "sinkhorn_steps", "max_length", "layout_mode"))
    add(
        "arti/fixed-topology-policy@1",
        FixedTopologyPolicy,
        lifecycle=alpha,
        variant="fixed-index",
        config_builder=lambda component: component.topology_contract(),
    )
    add(
        "arti/topology-action@1",
        TopologyAction,
        lifecycle=alpha,
        variant="priority-operands",
        config_builder=lambda component: {
            "shape": list(component.priority.shape),
            "dtype": str(component.priority.dtype),
        },
    )
    add(
        "arti/topology-proposal@1",
        TopologyProposal,
        lifecycle=alpha,
        variant="continuous-priority-proposal",
        config_builder=lambda component: {
            "action_ref": component_ref(component.action),
        },
        dependency_builder=lambda _component: ("arti/topology-action@1",),
    )
    add(
        "arti/stable-priority-partition@1",
        StablePriorityPartition,
        lifecycle=alpha,
        variant="valid-first-stable-sort",
    )
    add(
        "arti/topology-surrogate@1",
        SoftTopKTopologySurrogate,
        lifecycle=alpha,
        variant="soft-top-k-vjp",
        config_builder=_fields("temperature"),
    )
    add(
        "arti/topology-surrogate@2",
        PairwiseRankTopologySurrogate,
        lifecycle=alpha,
        variant="pairwise-soft-rank-position-vjp",
        config_builder=lambda component: component.topology_contract(),
    )
    add(
        "arti/learned-topology-policy@1",
        LearnedTopologyPolicy,
        lifecycle=alpha,
        variant="equivariant-scorer",
        config_builder=lambda component: component.topology_contract(),
        dependency_builder=lambda _component: ("arti/topology-surrogate@1",),
    )
    add(
        "arti/topology-priority-formula@1",
        TopologyPriorityFormula,
        lifecycle=alpha,
        variant="affine-priority",
        config_builder=lambda component: {
            "contract": component.contract,
            "trainable": component.trainable,
        },
    )
    add(
        "arti/topology-priority-formula@2",
        TypedTopologyPriorityFormula,
        lifecycle=alpha,
        variant="typed-affine-priority",
        config_builder=lambda component: {
            "factor_dim": component.contract.factor_dim,
            "mode": component.contract.mode,
            "output_semantics": component.contract.output_semantics,
            "api_version": component.contract.api_version,
            "trainable": component.trainable,
        },
    )
    add(
        "arti/topology-formula-lock@1",
        TopologyFormulaLock,
        lifecycle=alpha,
        variant="formula-binding",
    )
    add(
        "arti/fixed-topology-query@1",
        FixedTopologyQuery,
        lifecycle=alpha,
        variant="deterministic-projection",
        config_builder=lambda component: component.topology_contract(),
    )
    add(
        "arti/topology-operand-bank@1",
        TopologyOperandBank,
        lifecycle=alpha,
        variant="fixed-address-values",
        config_builder=lambda component: component.structure_contract,
    )
    add(
        "arti/topology-operand-bank@2",
        TypedTopologyOperandBank,
        lifecycle=alpha,
        variant="typed-fixed-address-values",
        config_builder=lambda component: component.structure_contract,
    )
    add(
        "arti/bank-formula-topology-policy@1",
        BankFormulaTopologyPolicy,
        lifecycle=alpha,
        variant="fixed-query-bank-formula",
        config_builder=lambda component: {
            "dim": component.dim,
            "key_dim": component.key_dim,
            "query_seed": component.query_seed,
            "ordered_banks": [bank.structure_contract for bank in component.banks],
            "bank_weights": component.bank_weights.detach().cpu().tolist(),
            "query": component.query.topology_contract(),
            "formula": {
                "ref": component.formula._component_reference,
                "contract_fingerprint": component.formula.contract.fingerprint,
                "factor_dim": component.formula.contract.factor_dim,
            },
            "diagnostics": component.diagnostics,
            "diagnostic_slot_limit": component.diagnostic_slot_limit,
        },
        dependency_builder=lambda component: (
            component.query._component_reference,
            component.formula._component_reference,
            *(bank._component_reference for bank in component.banks),
        ),
    )
    add(
        "arti/bank-formula-topology-policy@2",
        TypedBankFormulaTopologyPolicy,
        lifecycle=alpha,
        variant="typed-fixed-query-bank-formula",
        config_builder=lambda component: {
            "dim": component.dim,
            "key_dim": component.key_dim,
            "query_seed": component.query_seed,
            "ordered_banks": [bank.structure_contract for bank in component.banks],
            "bank_weights": component.bank_weights.detach().cpu().tolist(),
            "query": component.query.topology_contract(),
            "formula": {
                "ref": component.formula._component_reference,
                "contract_fingerprint": component.formula.contract.fingerprint,
                "factor_dim": component.formula.contract.factor_dim,
            },
            "diagnostics": component.diagnostics,
            "diagnostic_slot_limit": component.diagnostic_slot_limit,
        },
        dependency_builder=lambda component: (
            component.query._component_reference,
            component.formula._component_reference,
            *(bank._component_reference for bank in component.banks),
        ),
    )
    add(
        "arti/reversible-topology@1",
        ReversibleTopology,
        lifecycle=alpha,
        variant="permutation-partition",
        config_builder=lambda component: {
            "active_count": component.active_count,
            "axis": component.axis,
            "policy_ref": component.policy._component_reference,
            "operator_ref": component.operator._component_reference,
            "surrogate_ref": (
                None
                if component.surrogate is None
                else component.surrogate._component_reference
            ),
            "inverse": "recorded-permutation",
            "record_ref": "arti/fold-record@1",
        },
        dependency_builder=lambda component: tuple(
            child._component_reference
            for child in (component.policy, component.operator, component.surrogate)
            if child is not None and hasattr(child, "_component_reference")
        ),
    )
    add(
        "arti/inverse-topology-contract@1",
        InverseTopologyContract,
        lifecycle=alpha,
        variant="recorded-permutation-inverse",
        config_builder=lambda component: component.topology_contract(),
    )
    add(
        "arti/fold@2",
        TopologyFold,
        lifecycle=alpha,
        capabilities=("pulse.stage.fold",),
        variant="reversible-forward",
        config_schema_version=3,
        config_builder=lambda component: {
            "topology_ref": component.topology._component_reference,
            "active_count": component.topology.active_count,
            "axis": component.topology.axis,
            "record_ref": "arti/fold-record@1",
            "source_contract_binding": component.source_contract_binding,
        },
        dependency_builder=lambda _component: (
            "arti/reversible-topology@1",
            "arti/fold-record@1",
            "arti/fold-state@1",
        ),
    )
    add(
        "arti/unfold@2",
        TopologyUnFold,
        lifecycle=alpha,
        capabilities=("pulse.stage.unfold",),
        variant="recorded-inverse",
        config_schema_version=2,
        config_builder=lambda component: {
            "inverse_contract_ref": component.inverse_contract._component_reference,
            "active_count": component.inverse_contract.active_count,
            "axis": component.inverse_contract.axis,
            "record_ref": "arti/fold-record@1",
        },
        dependency_builder=lambda _component: (
            "arti/inverse-topology-contract@1",
            "arti/fold-record@1",
            "arti/fold-state@1",
        ),
    )
    add(
        "arti/fold-record@1",
        FoldRecord,
        lifecycle=alpha,
        variant="runtime-record",
        state_schema_version=FOLD_RECORD_SCHEMA_VERSION,
    )
    add(
        "arti/fold-state@1",
        FoldedTensor,
        lifecycle=alpha,
        variant="runtime-state",
        state_schema_version=FOLD_STATE_SCHEMA_VERSION,
        dependency_builder=lambda _component: ("arti/fold-record@1",),
    )
    add(
        "arti/pulse-stage@1",
        PulseStageSpec,
        lifecycle=alpha,
        variant="ordered-stage-spec",
        state_schema_version=PULSE_STAGE_SCHEMA_VERSION,
        config_builder=lambda component: component.to_dict(),
        dependency_builder=lambda component: (
            (component.component_ref,) if component.mode.value == "enabled" else ()
        ),
    )
    add(
        "arti/fixed-observation-policy@1",
        FixedObservationPolicy,
        lifecycle=alpha,
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
        lifecycle=alpha,
        variant="identity-substrate-observation",
    )
    add(
        "arti/fourier-observation-operator@1",
        FourierShiftObservationOperator,
        lifecycle=alpha,
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
        lifecycle=alpha,
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
        lifecycle=alpha,
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
        lifecycle=alpha,
        variant="deterministic-input-trajectory-query",
        config_builder=lambda component: component.observation_query_contract(),
    )
    add(
        "arti/observation-operand-bank@1",
        ObservationOperandBank,
        lifecycle=alpha,
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
        lifecycle=alpha,
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
        lifecycle=alpha,
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
            "ordered_banks": [bank.structure_contract for bank in component.banks],
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
        lifecycle=alpha,
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
        lifecycle=alpha,
        variant="per-feature-next-state",
        capabilities=("selective.compute.kernel",),
        config_builder=lambda component: {"dim": component.dim},
    )
    add(
        "arti/magnitude-intervention-policy@1",
        MagnitudeInterventionPolicy,
        lifecycle=alpha,
        variant="feature-strength-priority",
        capabilities=("formula.intervention.policy",),
    )
    add(
        "arti/factor-intervention-policy@1",
        FactorInterventionPolicy,
        lifecycle=alpha,
        variant="typed-factor-priority",
        capabilities=("formula.intervention.policy",),
        config_builder=lambda component: {"factor_index": component.factor_index},
    )
    add(
        "arti/stable-topk-intervention@1",
        StableTopKIntervention,
        lifecycle=alpha,
        variant="bounded-stable-support-selection",
        capabilities=("formula.intervention.operator",),
        config_builder=lambda component: {"max_interventions": component.max_interventions},
    )
    add(
        "arti/formula-attention@1",
        FormulaAttention,
        lifecycle=alpha,
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
        lifecycle=alpha,
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
        "arti/formula-commit-blend@1",
        FormulaCommitBlend,
        lifecycle=alpha,
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
        lifecycle=alpha,
        variant="folded-workspace-formula-executor",
        config_schema_version=2,
        capabilities=("pulse.stage.selective-compute",),
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
        "arti/bank-formula-route-source@1",
        BankFormulaRouteSource,
        lifecycle=alpha,
        variant="fixed-query-bank-formula-route-plan",
        capabilities=("formula.fabric.route-source",),
        config_builder=lambda component: {
            "program": component.program.to_dict(),
            "program_fingerprint": component.program.fingerprint,
            "active_count": component.active_count,
            "estimator": component.estimator,
            "policy_order": [
                {
                    "ref": component_ref(policy),
                    "config_fingerprint": component_spec(policy).config_fingerprint,
                }
                for policy in component.policies
            ],
            "candidate_mask_shape": list(component._candidate_mask.shape),
            "route_source_config_fingerprint": component.config_fingerprint,
            "limits": dict(component.limits.__dict__),
            "route_plan": "pre-execution-static-ssa-availability",
            "write_authority": "host-intervened-support",
        },
        dependency_builder=lambda component: tuple(
            component_ref(policy) for policy in component.policies
        ),
    )
    add(
        "arti/routed-formula-fabric-compute@1",
        RoutedFormulaFabricCompute,
        lifecycle=alpha,
        variant="bank-routed-formula-fabric-adapter",
        capabilities=("pulse.stage.selective-compute",),
        config_schema_version=2,
        config_builder=lambda component: {
            "compute": component_spec(component.compute).to_dict(),
            "route_source": component_spec(component.route_source).to_dict(),
            "active_count": component.active_count,
            "explicit_route": "per-call-override",
            "route_contract": "bound-source-or-explicit-override",
            "adapter_config_fingerprint": component.config_fingerprint,
        },
        dependency_builder=lambda component: (
            component_ref(component.compute),
            component_ref(component.route_source),
        ),
    )
    add(
        "arti/iterative-routed-formula-fabric-compute@1",
        IterativeRoutedFormulaFabricCompute,
        lifecycle=alpha,
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
        lifecycle=alpha,
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
        lifecycle=alpha,
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
        lifecycle=alpha,
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
        lifecycle=alpha,
        variant="reunion-only-aggregate-host",
        capabilities=("pulse.stage.aggregate",),
        config_builder=lambda component: {"kernel": component_spec(component.kernel).to_dict()},
        dependency_builder=lambda component: (component_ref(component.kernel),),
    )
    add(
        "arti/pulse-stage-graph@1",
        PulseStageGraph,
        lifecycle=alpha,
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
        lifecycle=alpha,
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
        lifecycle=alpha,
        variant="learned",
        aliases=("Pulse", "LearnedPulse"),
        deprecated_aliases=("arti/learned-pulse@1",),
    )
    add(
        "arti/pulse@2",
        AdaptivePulse,
        lifecycle=alpha,
        variant="adaptive-composable-stage-graph",
        config_schema_version=2,
        config_builder=adaptive_pulse_config,
        dependency_builder=adaptive_pulse_dependencies,
    )
    add("arti/pulse-legacy@1", PulseCompressor, lifecycle="legacy", variant="explicit", aliases=("PulseCompressor",))
    add("arti/fusion-pulse@1", FusionPulse, lifecycle=alpha)
    add(
        "arti/recall@2",
        Recall,
        lifecycle=alpha,
        config_builder=_fields("dim", "slots", "formula_id", "formula_origin"),
        dependency_builder=recall_dependencies,
    )
    add(
        "arti/recall@1",
        Recall,
        lifecycle=alpha,
        factory=legacy_recall_factory,
        config_builder=_fields("dim", "slots", "formula_id", "formula_origin"),
        dependency_builder=recall_dependencies,
    )
    add(
        "arti/recall-refiner@1",
        RecallRefiner,
        lifecycle=alpha,
        config_builder=_fields("steps", "learnable_step_scale"),
        dependency_builder=refiner_dependencies,
    )
    add(
        "arti/target-bank-updater@1",
        TargetBankUpdater,
        lifecycle=alpha,
        variant="target-addressable",
        capabilities=("pulse.stage.bank-update",),
        config_builder=target_bank_config,
        dependency_builder=target_bank_dependencies,
    )
    add(
        "arti/target-bank-updater@2",
        TargetBankUpdater,
        lifecycle=alpha,
        variant="target-coupled-after-bootstrap",
        config_schema_version=2,
        capabilities=("pulse.stage.bank-update",),
        factory=coupled_target_bank_updater_factory,
        config_builder=target_bank_config,
        dependency_builder=target_bank_dependencies,
    )
    add(
        "arti/objective-exposure-bank@1",
        ObjectiveExposureBank,
        lifecycle=alpha,
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
        lifecycle=alpha,
        variant="objective-controlled-formula-commit",
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
        "arti/write-refine-policy@1",
        WriteRefinePolicy,
        lifecycle=alpha,
        variant="runtime-only",
        config_builder=lambda component: {
            "budget": component.budget,
            "stop": component.stop,
            "exposure_schedule": component.exposure_schedule,
        },
        dependency_builder=lambda component: (
            "arti/refine-budget@1",
            *(("arti/refine-stop@1",) if component.stop is not None else ()),
        ),
    )
    add(
        "arti/refine-budget@1",
        RefineBudget,
        lifecycle=alpha,
        variant="runtime-only",
        config_builder=lambda component: {
            "max_steps": component.max_steps,
            "min_steps": component.min_steps,
        },
    )
    add(
        "arti/refine-policy@1",
        RefinePolicy,
        lifecycle=alpha,
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
        "arti/refine-stop@1",
        RefineStop,
        lifecycle=alpha,
        variant="runtime-only",
        config_builder=lambda component: asdict(component),
    )
    add("arti/layer@1", ARTILayer, lifecycle=alpha)
    add("arti/latent-tensor-layer@1", ARTILatentTensorLayer, lifecycle=alpha)
    add("arti/dynamic-state@1", ARTIDynamicStateLayer, lifecycle=alpha)
    add("arti/phase-mixer@1", ARTIPhaseMixer, lifecycle=alpha)
    add("arti/virtual-interface@1", ARTIVirtualInterfaceMixer, lifecycle=alpha)
    add("arti/latent-recall-field@1", ARTILatentRecallField, lifecycle=alpha)
    return registry


def get_component_registry() -> ComponentRegistry:
    global _DEFAULT_REGISTRY
    if _DEFAULT_REGISTRY is None:
        _DEFAULT_REGISTRY = _build_default_registry()
    return _DEFAULT_REGISTRY


def register_component(reference: str, **kwargs: Any) -> ComponentRegistration:
    return get_component_registry().register(reference, **kwargs)


def resolve_component(reference_or_alias: str, **kwargs: Any) -> Any:
    registration = get_component_registry().resolve_registration(reference_or_alias)
    if registration.factory is None:
        raise ComponentRegistryError(
            f"component {registration.reference!r} has no executable factory"
        )
    return registration.factory(**kwargs)


def component_ref(value: Any) -> str:
    registration = get_component_registry().registration_for(value)
    if registration is None:
        raise UnknownComponentError(f"no component registration for {type(value).__name__}")
    return registration.reference


def _api_name(value: Any) -> str:
    cls = type(value)
    return f"{cls.__module__}.{cls.__qualname__}"


def _direct_registered_dependencies(module: nn.Module, registry: ComponentRegistry) -> tuple[str, ...]:
    return tuple(sorted({registration.reference for child in module.children() if (registration := registry.registration_for(child)) is not None}))


def component_spec(value: Any, *, path: str = "$") -> ComponentSpec:
    registration = get_component_registry().registration_for(value)
    if registration is None:
        raise UnknownComponentError(f"no component registration for {type(value).__name__}")
    config = registration.config(value)
    dependencies = _direct_registered_dependencies(value, get_component_registry()) if isinstance(value, nn.Module) else ()
    dependencies = tuple(sorted(set(dependencies) | set(registration.dependencies(value))))
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
            _parameter_schema_fingerprint(value)
            if isinstance(value, nn.Module)
            else _sha256_json(config)
        ),
        dependencies=dependencies,
        capabilities=registration.capabilities,
    )
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
        child_refs = set(registration.dependencies(module))
        for child_path, _child, child_registration in registered:
            if child_path == raw_path:
                continue
            if raw_path:
                if not child_path.startswith(raw_path + "."):
                    continue
                relative = child_path[len(raw_path) + 1 :]
            else:
                relative = child_path
            if len(relative.split(".")) == 1:
                child_refs.add(child_registration.reference)
        config = registration.config(module)
        specs.append(
            ComponentSpec(
                path="$" if raw_path == "" else raw_path,
                reference=registration.reference,
                api=_api_name(module),
                variant=registration.variant,
                lifecycle=registration.lifecycle,
                config_schema_version=registration.config_schema_version,
                state_schema_version=registration.state_schema_version,
                config=config,
                config_fingerprint=_sha256_json(config),
                parameter_schema_fingerprint=_parameter_schema_fingerprint(module),
                dependencies=tuple(sorted(child_refs)),
                capabilities=registration.capabilities,
            )
        )
    return [spec.to_dict() for spec in sorted(specs, key=lambda item: item.path)]


def component_graph_fingerprint(components: Sequence[Mapping[str, Any]]) -> str:
    return _sha256_json(list(components))


def component_provenance(model: nn.Module) -> dict[str, Any]:
    components = component_manifest(model)
    return {
        "schema_version": COMPONENT_PROVENANCE_VERSION,
        "components": components,
        "fingerprint": component_graph_fingerprint(components),
    }


def component_catalog() -> list[dict[str, Any]]:
    catalog = list(get_component_registry().catalog())
    for formula in __import__("arti.recall_registry", fromlist=["list_formulas"]).list_formulas():
        catalog.append({"kind": "formula", "ref": formula.reference, "mechanism_id": f"{formula.namespace}/{formula.name}", "mechanism_version": formula.version, "variant": formula.provider_kind, "lifecycle": "alpha", "config_schema_version": 1, "state_schema_version": 1, "aliases": [], "deprecated_aliases": [], "capabilities": []})
    for survival in __import__("arti.survival", fromlist=["list_survivals"]).list_survivals():
        catalog.append({"kind": "survival", "ref": survival.reference, "mechanism_id": f"{survival.namespace}/{survival.name}", "mechanism_version": survival.version, "variant": "survival", "lifecycle": "alpha", "config_schema_version": 1, "state_schema_version": 1, "aliases": [], "deprecated_aliases": [], "capabilities": []})
    return sorted(catalog, key=lambda item: item["ref"])


def _parameter_schema_fingerprint(module: nn.Module) -> str:
    entries = []
    for name, value in module.named_parameters():
        entries.append({"kind": "parameter", "name": name, "dtype": str(value.dtype), "shape": list(value.shape)})
    for name, value in module.named_buffers():
        entries.append({"kind": "buffer", "name": name, "dtype": str(value.dtype), "shape": list(value.shape)})
    return _sha256_json(sorted(entries, key=lambda item: (item["kind"], item["name"])))


def state_dict_schema(state_dict: Mapping[str, Tensor]) -> dict[str, Any]:
    entries = [{"name": name, "dtype": str(value.dtype), "shape": list(value.shape)} for name, value in state_dict.items()]
    content = {"schema_version": COMPONENT_STATE_CONTRACT_VERSION, "tensors": sorted(entries, key=lambda item: item["name"])}
    return {**content, "fingerprint": _sha256_json(content)}


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


def _validate_vnext_dependency_closure(
    reference: str,
    config: Any,
    dependencies: list[str],
) -> None:
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
    expected = sorted({root_dependency, *graph.enabled_dependencies})
    if dependencies != expected:
        raise ComponentCompatibilityError(
            f"component dependency closure is invalid for {reference!r}"
        )


def validate_component_provenance(
    value: Mapping[str, Any],
    *,
    allow_legacy: bool = False,
) -> dict[str, Any]:
    """Validate and normalize a component provenance graph."""

    if not isinstance(value, Mapping):
        raise ComponentCompatibilityError("component provenance must be a mapping")
    if set(value) != {"schema_version", "components", "fingerprint"}:
        raise ComponentCompatibilityError(
            "component provenance has missing or unknown top-level fields"
        )
    schema_version = value["schema_version"]
    if type(schema_version) is not int or schema_version not in {
        1,
        COMPONENT_PROVENANCE_VERSION,
    }:
        raise ComponentCompatibilityError("unsupported component provenance schema version")
    components = value["components"]
    if not isinstance(components, list) or any(
        not isinstance(item, Mapping) for item in components
    ):
        raise ComponentCompatibilityError(
            "component provenance components must be a list of mappings"
        )
    if value["fingerprint"] != component_graph_fingerprint(components):
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
            raise ComponentCompatibilityError(
                "component provenance entry has missing or unknown fields"
            )
        path = item["path"]
        if not isinstance(path, str) or not path or path in paths:
            raise ComponentCompatibilityError(
                "component provenance paths must be unique non-empty strings"
            )
        paths.add(path)
        identity = ComponentRef.parse(item["ref"])
        if (
            item["mechanism_id"] != identity.mechanism_id
            or item["mechanism_version"] != identity.version
        ):
            raise ComponentCompatibilityError(
                f"component identity fields disagree at {path!r}"
            )
        try:
            registration = registry.registration_for_reference(identity.reference)
        except UnknownComponentError as error:
            raise ComponentCompatibilityError(
                f"unknown component reference in artifact: {identity.reference!r}"
            ) from error
        if registration.lifecycle in {"legacy", "deprecated"} and not allow_legacy:
            raise ComponentCompatibilityError(
                f"component {identity.reference!r} is {registration.lifecycle}; "
                "pass allow_legacy=True explicitly"
            )
        if item["variant"] != registration.variant or item["lifecycle"] != registration.lifecycle:
            raise ComponentCompatibilityError(
                f"component lifecycle or variant drift at {path!r}"
            )
        if (
            item["config_schema_version"] != registration.config_schema_version
            or item["state_schema_version"] != registration.state_schema_version
        ):
            raise ComponentCompatibilityError(f"component schema version drift at {path!r}")
        capabilities = (
            item["capabilities"]
            if schema_version >= 2
            else list(registration.capabilities)
        )
        if capabilities != list(registration.capabilities):
            raise ComponentCompatibilityError(f"component capability drift at {path!r}")
        if item["config_fingerprint"] != _sha256_json(item["config"]):
            raise ComponentCompatibilityError(
                f"component config fingerprint is invalid at {path!r}"
            )
        dependencies = item["dependencies"]
        if not isinstance(dependencies, list) or any(
            not isinstance(dependency, str)
            or not _is_known_dependency(dependency, registry)
            for dependency in dependencies
        ):
            raise ComponentCompatibilityError(f"component dependencies are invalid at {path!r}")
        _validate_vnext_dependency_closure(identity.reference, item["config"], dependencies)
        normalized.append({**dict(item), "capabilities": capabilities})

    return {
        "schema_version": COMPONENT_PROVENANCE_VERSION,
        "components": normalized,
        "fingerprint": component_graph_fingerprint(normalized),
    }


__all__ = [
    "COMPONENT_PROVENANCE_VERSION",
    "COMPONENT_STATE_CONTRACT_VERSION",
    "ComponentCompatibilityError",
    "ComponentRef",
    "ComponentRegistration",
    "ComponentRegistry",
    "ComponentRegistryError",
    "ComponentSpec",
    "DuplicateComponentError",
    "InvalidComponentRefError",
    "UnknownComponentError",
    "component_catalog",
    "component_graph_fingerprint",
    "component_manifest",
    "component_provenance",
    "component_ref",
    "component_spec",
    "get_component_registry",
    "register_component",
    "resolve_component",
    "state_dict_schema",
    "validate_component_provenance",
]
