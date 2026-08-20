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
    factory: Factory | None = None
    config_builder: ConfigBuilder | None = None
    dependency_builder: DependencyBuilder | None = None

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
    dependencies: tuple[str, ...] = ()

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
            "dependencies": list(self.dependencies),
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
        config_schema_version: int = 1,
        state_schema_version: int = 1,
        factory: Factory | None = None,
        config_builder: ConfigBuilder | None = None,
        dependency_builder: DependencyBuilder | None = None,
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
            config_schema_version=config_schema_version,
            state_schema_version=state_schema_version,
            factory=component_type if factory is None else factory,
            config_builder=config_builder,
            dependency_builder=dependency_builder,
        )
        with self._lock:
            if reference in self._by_reference or reference in self._by_alias:
                raise DuplicateComponentError(f"component reference is already registered: {reference}")
            aliases_to_add = set(registration.aliases)
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
    from .layers import ARTIDynamicStateLayer, ARTILatentRecallField, ARTILatentTensorLayer, ARTILayer, ARTIPhaseMixer, ARTIVirtualInterfaceMixer
    from .nn import Fold, FusionPulse, Half, LearnedPulse, Recall, RecallRefiner, UnFold
    from .pulse import PulseCompressor
    from .recall_refine import RefineBudget, RefineStop
    from .target_bank import TargetBankUpdater, WriteRefinePolicy

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

    alpha = "alpha"
    add("arti/half@1", Half, lifecycle=alpha, config_schema_version=2, config_builder=_fields("_threshold_init", "_base_init", "_scale_init", "stochastic", "learnable"))
    add("arti/fold@1", Fold, lifecycle=alpha, config_builder=_fields("k", "dim", "hidden_dim", "temperature", "mode", "topk", "heads", "eps"))
    add("arti/unfold@1", UnFold, lifecycle=alpha, config_builder=_fields("dim", "exposed", "guide_dim", "condition_dim", "hidden_dim", "temperature", "sinkhorn_steps", "max_length", "layout_mode"))
    add("arti/pulse@1", LearnedPulse, lifecycle=alpha, variant="learned", aliases=("Pulse", "LearnedPulse"))
    add("arti/pulse-legacy@1", PulseCompressor, lifecycle="legacy", variant="explicit", aliases=("PulseCompressor",))
    add("arti/fusion-pulse@1", FusionPulse, lifecycle=alpha)
    add("arti/recall@1", Recall, lifecycle=alpha, config_builder=_fields("dim", "slots", "formula_id", "formula_origin"))
    add("arti/recall-refiner@1", RecallRefiner, lifecycle=alpha, config_builder=_fields("steps", "learnable_step_scale"))
    add(
        "arti/target-bank-updater@1",
        TargetBankUpdater,
        lifecycle=alpha,
        variant="target-addressable",
        config_builder=target_bank_config,
        dependency_builder=target_bank_dependencies,
    )
    add(
        "arti/target-bank-updater@2",
        TargetBankUpdater,
        lifecycle=alpha,
        variant="target-coupled-after-bootstrap",
        config_schema_version=2,
        factory=coupled_target_bank_updater_factory,
        config_builder=target_bank_config,
        dependency_builder=target_bank_dependencies,
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
        dependencies=dependencies,
    )


def component_catalog() -> list[dict[str, Any]]:
    catalog = list(get_component_registry().catalog())
    for formula in __import__("arti.recall_registry", fromlist=["list_formulas"]).list_formulas():
        catalog.append({"kind": "formula", "ref": formula.reference, "mechanism_id": f"{formula.namespace}/{formula.name}", "mechanism_version": formula.version, "variant": formula.provider_kind, "lifecycle": "alpha", "config_schema_version": 1, "state_schema_version": 1, "aliases": [], "deprecated_aliases": []})
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


__all__ = [
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
    "component_ref",
    "component_spec",
    "get_component_registry",
    "register_component",
    "resolve_component",
    "state_dict_schema",
]
