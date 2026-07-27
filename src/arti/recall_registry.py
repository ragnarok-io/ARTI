"""Explicit, process-local registration for Recall formulas.

The registry never scans entry points and never imports or installs code named
by an artifact. Applications must import their formula implementation and
register it explicitly before resolving its stable identity.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable, Literal
from weakref import WeakSet

from torch import nn


FormulaFactory = Callable[[], nn.Module]
FormulaOrigin = Literal["builtin", "registered", "custom"]
FormulaProviderKind = Literal["factory", "instance"]

_COMPONENT = r"[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?"
_REFERENCE_PATTERN = re.compile(
    rf"^(?P<namespace>{_COMPONENT})/(?P<name>{_COMPONENT})@(?P<version>[1-9][0-9]*)$"
)
_BUILTIN_NAMESPACE = "arti"


class RecallFormulaRegistryError(ValueError):
    """Base class for Recall formula registry errors."""


class InvalidRecallFormulaIdError(RecallFormulaRegistryError):
    """Raised when a formula identity is not canonical."""


class DuplicateRecallFormulaError(RecallFormulaRegistryError):
    """Raised when an exact formula identity is already registered."""


class UnknownRecallFormulaError(RecallFormulaRegistryError):
    """Raised when an exact formula identity has not been registered."""


class FrozenRecallFormulaRegistryError(RecallFormulaRegistryError):
    """Raised when code attempts to mutate a frozen registry snapshot."""


@dataclass(frozen=True, order=True)
class RecallFormulaId:
    """A canonical, exact Recall formula identity."""

    namespace: str
    name: str
    version: int

    @property
    def base_id(self) -> str:
        return f"{self.namespace}/{self.name}"

    @property
    def reference(self) -> str:
        return f"{self.base_id}@{self.version}"

    @classmethod
    def parse(cls, reference: str) -> "RecallFormulaId":
        if not isinstance(reference, str):
            raise InvalidRecallFormulaIdError(
                f"Recall formula reference must be a string, got {type(reference).__name__}"
            )
        match = _REFERENCE_PATTERN.fullmatch(reference)
        if match is None:
            raise InvalidRecallFormulaIdError(
                "Recall formula reference must use canonical "
                "'namespace/name@version' syntax with lowercase ASCII names and "
                "a positive integer version; for example 'arti/state@1' or "
                "'acme/signed-gate@2'"
            )
        return cls(
            namespace=match.group("namespace"),
            name=match.group("name"),
            version=int(match.group("version")),
        )


@dataclass(frozen=True)
class RecallFormulaDescription:
    """Serializable metadata describing one registered formula."""

    reference: str
    namespace: str
    name: str
    version: int
    origin: FormulaOrigin
    provider_kind: FormulaProviderKind
    portable: bool
    description: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "reference": self.reference,
            "namespace": self.namespace,
            "name": self.name,
            "version": self.version,
            "origin": self.origin,
            "provider_kind": self.provider_kind,
            "portable": self.portable,
            "description": self.description,
        }


@dataclass(frozen=True)
class RecallFormulaRegistration:
    """A resolved formula provider and its stable metadata."""

    identity: RecallFormulaId
    origin: FormulaOrigin
    provider_kind: FormulaProviderKind
    portable: bool
    description: str | None = None
    _provider: Any = field(default=None, repr=False, compare=False)
    _instances: WeakSet[nn.Module] = field(
        default_factory=WeakSet,
        repr=False,
        compare=False,
    )
    _instance_lock: RLock = field(default_factory=RLock, repr=False, compare=False)

    @property
    def reference(self) -> str:
        return self.identity.reference

    def instantiate(self) -> nn.Module:
        """Return the registered instance or create one through its factory."""

        if self.provider_kind == "instance":
            return self._provider
        formula = self._provider()
        if not isinstance(formula, nn.Module):
            raise RecallFormulaRegistryError(
                f"factory for Recall formula {self.reference!r} must return torch.nn.Module"
            )
        with self._instance_lock:
            if formula in self._instances:
                raise RecallFormulaRegistryError(
                    f"factory for Recall formula {self.reference!r} returned a shared "
                    "module instance; factories must create a fresh module"
                )
            self._instances.add(formula)
        return formula

    def describe(self) -> RecallFormulaDescription:
        return RecallFormulaDescription(
            reference=self.reference,
            namespace=self.identity.namespace,
            name=self.identity.name,
            version=self.identity.version,
            origin=self.origin,
            provider_kind=self.provider_kind,
            portable=self.portable,
            description=self.description,
        )


class RecallFormulaRegistry:
    """Thread-safe explicit registry with exact-version lookup."""

    def __init__(
        self,
        registrations: tuple[RecallFormulaRegistration, ...] = (),
        *,
        frozen: bool = False,
    ) -> None:
        self._lock = RLock()
        self._registrations = {
            registration.reference: registration for registration in registrations
        }
        if len(self._registrations) != len(registrations):
            raise DuplicateRecallFormulaError(
                "cannot construct a Recall formula registry with duplicate references"
            )
        self._frozen = bool(frozen)

    @property
    def is_frozen(self) -> bool:
        return self._frozen

    def register_builtin(
        self,
        reference: str,
        *,
        factory: FormulaFactory,
        description: str | None = None,
    ) -> RecallFormulaRegistration:
        """Register a core-owned formula under the reserved ``arti`` namespace."""

        identity = RecallFormulaId.parse(reference)
        if identity.namespace != _BUILTIN_NAMESPACE:
            raise InvalidRecallFormulaIdError(
                f"builtin Recall formula {reference!r} must use the "
                f"{_BUILTIN_NAMESPACE!r} namespace"
            )
        return self._register(
            identity,
            provider=factory,
            provider_kind="factory",
            origin="builtin",
            portable=True,
            description=description,
        )

    def register(
        self,
        reference: str,
        *,
        factory: FormulaFactory | None = None,
        instance: nn.Module | None = None,
        portable: bool | None = None,
        description: str | None = None,
    ) -> RecallFormulaRegistration:
        """Explicitly register a process-local third-party Formula factory."""

        identity = RecallFormulaId.parse(reference)
        if identity.namespace == _BUILTIN_NAMESPACE:
            raise InvalidRecallFormulaIdError(
                f"namespace {_BUILTIN_NAMESPACE!r} is reserved for builtin formulas; "
                "use register_builtin() from ARTI core"
            )
        if instance is not None:
            raise RecallFormulaRegistryError(
                "registered Formula instances are not supported because they "
                "silently share parameters; pass the module directly to Recall "
                "or register a factory"
            )
        if factory is None:
            raise RecallFormulaRegistryError("register() requires factory=")
        if not callable(factory):
            raise RecallFormulaRegistryError("factory must be callable")
        if portable:
            raise RecallFormulaRegistryError(
                "third-party Recall formulas are process-local and cannot declare "
                "portable=true until artifact authorization is implemented"
            )
        return self._register(
            identity,
            provider=factory,
            provider_kind="factory",
            origin="registered",
            portable=False,
            description=description,
        )

    def _register(
        self,
        identity: RecallFormulaId,
        *,
        provider: Any,
        provider_kind: FormulaProviderKind,
        origin: FormulaOrigin,
        portable: bool,
        description: str | None,
    ) -> RecallFormulaRegistration:
        if description is not None and not isinstance(description, str):
            raise RecallFormulaRegistryError("description must be a string or None")
        registration = RecallFormulaRegistration(
            identity=identity,
            origin=origin,
            provider_kind=provider_kind,
            portable=portable,
            description=description,
            _provider=provider,
        )
        with self._lock:
            if self._frozen:
                raise FrozenRecallFormulaRegistryError(
                    "cannot register a Recall formula in a frozen registry snapshot"
                )
            if registration.reference in self._registrations:
                raise DuplicateRecallFormulaError(
                    f"Recall formula {registration.reference!r} is already registered"
                )
            self._registrations[registration.reference] = registration
        return registration

    def resolve(self, reference: str) -> RecallFormulaRegistration:
        """Resolve an exact registered identity without importing any code."""

        identity = RecallFormulaId.parse(reference)
        with self._lock:
            registration = self._registrations.get(identity.reference)
            known = tuple(sorted(self._registrations))
        if registration is None:
            suffix = f"; registered formulas: {', '.join(known)}" if known else ""
            raise UnknownRecallFormulaError(
                f"Recall formula {identity.reference!r} is not explicitly registered{suffix}"
            )
        return registration

    def list(self) -> tuple[RecallFormulaDescription, ...]:
        """Return immutable descriptions ordered by canonical reference."""

        with self._lock:
            references = sorted(self._registrations)
            return tuple(
                self._registrations[reference].describe() for reference in references
            )

    def describe(self, reference: str) -> RecallFormulaDescription:
        return self.resolve(reference).describe()

    def freeze(self) -> "RecallFormulaRegistry":
        """Return an isolated, immutable snapshot of current registrations."""

        with self._lock:
            registrations = tuple(self._registrations.values())
        return RecallFormulaRegistry(registrations, frozen=True)


_FORMULA_REGISTRY = RecallFormulaRegistry()


def register_formula(
    reference: str,
    *,
    factory: FormulaFactory | None = None,
    instance: nn.Module | None = None,
    portable: bool | None = None,
    description: str | None = None,
) -> RecallFormulaRegistration:
    return _FORMULA_REGISTRY.register(
        reference,
        factory=factory,
        instance=instance,
        portable=portable,
        description=description,
    )


def resolve_formula(reference: str) -> RecallFormulaRegistration:
    return _FORMULA_REGISTRY.resolve(reference)


def list_formulas() -> tuple[RecallFormulaDescription, ...]:
    from .recall_formula import BUILTIN_RECALL_FORMULAS

    builtins = tuple(
        RecallFormulaDescription(
            reference=formula_id,
            namespace="arti",
            name=description.contract.identity.name,
            version=description.contract.identity.version,
            origin="builtin",
            provider_kind="factory",
            portable=True,
            description=description.summary,
        )
        for formula_id, description in BUILTIN_RECALL_FORMULAS.items()
        if description.contract.identity is not None
    )
    return tuple(
        sorted((*builtins, *_FORMULA_REGISTRY.list()), key=lambda item: item.reference)
    )


def describe_formula(reference: str) -> RecallFormulaDescription:
    from .recall_formula import (
        BUILTIN_RECALL_FORMULA_ALIASES,
        BUILTIN_RECALL_FORMULAS,
    )

    builtin_id = BUILTIN_RECALL_FORMULA_ALIASES.get(reference, reference)
    builtin = BUILTIN_RECALL_FORMULAS.get(builtin_id)
    if builtin is not None:
        assert builtin.contract.identity is not None
        return RecallFormulaDescription(
            reference=builtin_id,
            namespace="arti",
            name=builtin.contract.identity.name,
            version=builtin.contract.identity.version,
            origin="builtin",
            provider_kind="factory",
            portable=True,
            description=builtin.summary,
        )
    return _FORMULA_REGISTRY.describe(reference)


def freeze_formula_registry() -> RecallFormulaRegistry:
    return _FORMULA_REGISTRY.freeze()


__all__ = [
    "DuplicateRecallFormulaError",
    "FrozenRecallFormulaRegistryError",
    "InvalidRecallFormulaIdError",
    "RecallFormulaDescription",
    "RecallFormulaId",
    "RecallFormulaRegistration",
    "RecallFormulaRegistry",
    "RecallFormulaRegistryError",
    "UnknownRecallFormulaError",
    "describe_formula",
    "freeze_formula_registry",
    "list_formulas",
    "register_formula",
    "resolve_formula",
]
