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
_CONTENT_REFERENCE_PATTERN = re.compile(
    rf"^(?P<namespace>{_COMPONENT})/(?P<name>{_COMPONENT})@sha256:(?P<digest>[0-9a-f]{{64}})$"
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
    """A Formula declaration input or resolved immutable contract identity."""

    namespace: str
    name: str
    contract_sha256: str | None = None
    source_version: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, str) or re.fullmatch(_COMPONENT, self.namespace) is None:
            raise InvalidRecallFormulaIdError("Recall formula namespace is invalid")
        if not isinstance(self.name, str) or re.fullmatch(_COMPONENT, self.name) is None:
            raise InvalidRecallFormulaIdError("Recall formula name is invalid")
        resolved = self.contract_sha256 is not None
        sourced = self.source_version is not None
        if resolved == sourced:
            raise InvalidRecallFormulaIdError(
                "Recall formula identity must contain either a contract digest or a source declaration"
            )
        if resolved and (
            not isinstance(self.contract_sha256, str)
            or re.fullmatch(r"[0-9a-f]{64}", self.contract_sha256) is None
        ):
            raise InvalidRecallFormulaIdError("Recall formula contract digest is invalid")
        if sourced and (
            isinstance(self.source_version, bool)
            or not isinstance(self.source_version, int)
            or self.source_version <= 0
        ):
            raise InvalidRecallFormulaIdError("Recall formula source version must be a positive integer")

    @property
    def base_id(self) -> str:
        return f"{self.namespace}/{self.name}"

    @property
    def is_canonical(self) -> bool:
        return self.contract_sha256 is not None

    @property
    def source_reference(self) -> str:
        if self.source_version is None:
            raise InvalidRecallFormulaIdError("resolved Formula identities have no source declaration")
        return f"{self.base_id}@{self.source_version}"

    @property
    def version(self) -> int | None:
        """Source declaration metadata, never part of a resolved artifact ref."""

        return self.source_version

    @property
    def reference(self) -> str:
        if self.contract_sha256 is not None:
            return f"{self.base_id}@sha256:{self.contract_sha256}"
        return self.source_reference

    def to_dict(self) -> dict[str, object]:
        if self.contract_sha256 is None:
            raise InvalidRecallFormulaIdError(
                "source Formula declarations cannot be serialized; resolve a Formula contract first"
            )
        return {
            "namespace": self.namespace,
            "name": self.name,
            "contract_sha256": self.contract_sha256,
        }

    @classmethod
    def from_dict(cls, value: Any) -> "RecallFormulaId":
        """Restore an exact identity from a validated, code-free payload."""

        if not isinstance(value, dict):
            raise InvalidRecallFormulaIdError(
                "Recall formula identity must be a mapping"
            )
        required = {"namespace", "name", "contract_sha256"}
        unknown = set(value) - required
        missing = required - set(value)
        if missing or unknown:
            raise InvalidRecallFormulaIdError(
                "Recall formula identity must contain exactly namespace, name, and contract_sha256"
            )
        return cls(
            namespace=value["namespace"],
            name=value["name"],
            contract_sha256=value["contract_sha256"],
        )

    @classmethod
    def parse(cls, reference: str) -> "RecallFormulaId":
        if not isinstance(reference, str):
            raise InvalidRecallFormulaIdError(
                f"Recall formula reference must be a string, got {type(reference).__name__}"
            )
        content = _CONTENT_REFERENCE_PATTERN.fullmatch(reference)
        if content is not None:
            return cls(
                namespace=content.group("namespace"),
                name=content.group("name"),
                contract_sha256=content.group("digest"),
            )
        declaration = _REFERENCE_PATTERN.fullmatch(reference)
        if declaration is None:
            raise InvalidRecallFormulaIdError(
                "Recall formula reference must use a full SHA-256 contract address "
                "or a source declaration at an explicit input boundary"
            )
        return cls(
            namespace=declaration.group("namespace"),
            name=declaration.group("name"),
            source_version=int(declaration.group("version")),
        )


@dataclass(frozen=True)
class RecallFormulaDescription:
    """Serializable metadata describing one registered formula."""

    reference: str
    namespace: str
    name: str
    contract_sha256: str
    origin: FormulaOrigin
    provider_kind: FormulaProviderKind
    portable: bool
    description: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "reference": self.reference,
            "namespace": self.namespace,
            "name": self.name,
            "contract_sha256": self.contract_sha256,
            "origin": self.origin,
            "provider_kind": self.provider_kind,
            "portable": self.portable,
            "description": self.description,
        }


@dataclass(frozen=True)
class RecallFormulaRegistration:
    """A resolved formula provider and its stable metadata."""

    identity: RecallFormulaId
    source_declaration: str
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
        self._validate_formula_contract(formula)
        with self._instance_lock:
            if formula in self._instances:
                raise RecallFormulaRegistryError(
                    f"factory for Recall formula {self.reference!r} returned a shared "
                    "module instance; factories must create a fresh module"
                )
            self._instances.add(formula)
        return formula

    def _validate_formula_contract(self, formula: nn.Module) -> None:
        from .recall_formula import RecallFormulaContract

        contract = getattr(formula, "recall_formula_contract", None)
        if not isinstance(contract, RecallFormulaContract):
            raise RecallFormulaRegistryError(
                f"factory for Recall formula {self.reference!r} must return a module "
                "with a RecallFormulaContract"
            )
        if contract.identity != self.identity:
            raise RecallFormulaRegistryError(
                f"factory for Recall formula {self.reference!r} returned a module "
                "with a different immutable Formula contract"
            )

    def describe(self) -> RecallFormulaDescription:
        return RecallFormulaDescription(
            reference=self.reference,
            namespace=self.identity.namespace,
            name=self.identity.name,
            contract_sha256=self.identity.contract_sha256 or "",
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
        self._declarations = {
            registration.source_declaration: registration.reference
            for registration in registrations
        }
        if len(self._declarations) != len(registrations):
            raise DuplicateRecallFormulaError(
                "cannot construct a Recall formula registry with duplicate source declarations"
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
        if identity.source_version is None:
            raise InvalidRecallFormulaIdError(
                "Formula registration requires an explicit source declaration at its input boundary"
            )
        if description is not None and not isinstance(description, str):
            raise RecallFormulaRegistryError("description must be a string or None")
        canonical_identity = self._provider_identity(identity, provider)
        registration = RecallFormulaRegistration(
            identity=canonical_identity,
            source_declaration=identity.source_reference,
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
            if (
                registration.reference in self._registrations
                or registration.source_declaration in self._declarations
            ):
                raise DuplicateRecallFormulaError(
                    f"Recall formula {registration.source_declaration!r} is already registered"
                )
            self._registrations[registration.reference] = registration
            self._declarations[registration.source_declaration] = registration.reference
        return registration

    @staticmethod
    def _provider_identity(
        declaration: RecallFormulaId,
        provider: Any,
    ) -> RecallFormulaId:
        from .recall_formula import RecallFormulaContract

        probe = provider()
        if not isinstance(probe, nn.Module):
            raise RecallFormulaRegistryError(
                f"factory for Recall formula {declaration.source_reference!r} must return "
                "torch.nn.Module"
            )
        contract = getattr(probe, "recall_formula_contract", None)
        if not isinstance(contract, RecallFormulaContract) or contract.identity is None:
            raise RecallFormulaRegistryError(
                f"factory for Recall formula {declaration.source_reference!r} must declare "
                "a RecallFormulaContract with an immutable identity"
            )
        if contract.identity.base_id != declaration.base_id:
            raise RecallFormulaRegistryError(
                f"factory Formula contract {contract.identity.base_id!r} does not match "
                f"registration declaration {declaration.base_id!r}"
            )
        if not contract.identity.is_canonical:
            raise RecallFormulaRegistryError(
                "Formula contracts must resolve to immutable SHA-256 identities"
            )
        return contract.identity

    def resolve(self, reference: str) -> RecallFormulaRegistration:
        """Resolve an exact registered identity without importing any code."""

        identity = RecallFormulaId.parse(reference)
        with self._lock:
            resolved_reference = (
                identity.reference
                if identity.is_canonical
                else self._declarations.get(identity.source_reference)
            )
            registration = (
                None
                if resolved_reference is None
                else self._registrations.get(resolved_reference)
            )
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
            reference=description.contract.identity.reference,
            namespace=description.contract.identity.namespace,
            name=description.contract.identity.name,
            contract_sha256=description.contract.identity.contract_sha256 or "",
            origin="builtin",
            provider_kind="factory",
            portable=True,
            description=description.summary,
        )
        for description in BUILTIN_RECALL_FORMULAS.values()
        if description.contract.identity is not None
    )
    return tuple(
        sorted((*builtins, *_FORMULA_REGISTRY.list()), key=lambda item: item.reference)
    )


def describe_formula(reference: str) -> RecallFormulaDescription:
    from .recall_formula import resolve_builtin_formula

    builtin = resolve_builtin_formula(reference)
    if builtin is not None:
        assert builtin.contract.identity is not None
        return RecallFormulaDescription(
            reference=builtin.contract.identity.reference,
            namespace=builtin.contract.identity.namespace,
            name=builtin.contract.identity.name,
            contract_sha256=builtin.contract.identity.contract_sha256 or "",
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
