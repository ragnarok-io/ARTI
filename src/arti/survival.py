"""Versioned, same-shape survival operators for :class:`arti.nn.Half`.

The survival operator computes a differentiable probability tensor. ``Half``
owns the execution policy (deterministic multiplication or Bernoulli
sampling); this module owns the salience rule and its reproducible identity.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from threading import RLock
from typing import Any, Literal

import torch
from torch import nn

from .functional import _half_contextual_survival, _half_survival


SurvivalFactory = Callable[[Mapping[str, Any]], nn.Module]
SurvivalOrigin = Literal["builtin", "registered"]
_COMPONENT = r"[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?"
_REFERENCE_PATTERN = re.compile(
    rf"^(?P<namespace>{_COMPONENT})/(?P<name>{_COMPONENT})@(?P<version>[1-9][0-9]*)$"
)


class SurvivalRegistryError(ValueError):
    """Base error for survival identity and registration failures."""


class InvalidSurvivalRefError(SurvivalRegistryError):
    """Raised when a survival reference is not canonical."""


class DuplicateSurvivalError(SurvivalRegistryError):
    """Raised when a survival identity is registered twice."""


class UnknownSurvivalError(SurvivalRegistryError):
    """Raised when a survival identity is not registered."""


@dataclass(frozen=True, order=True)
class SurvivalRef:
    """Canonical ``namespace/name@version`` identity for a survival rule."""

    namespace: str
    name: str
    version: int

    def __post_init__(self) -> None:
        if re.fullmatch(_COMPONENT, self.namespace) is None:
            raise InvalidSurvivalRefError("survival namespace is invalid")
        if re.fullmatch(_COMPONENT, self.name) is None:
            raise InvalidSurvivalRefError("survival name is invalid")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version <= 0:
            raise InvalidSurvivalRefError("survival version must be a positive integer")

    @property
    def reference(self) -> str:
        return f"{self.namespace}/{self.name}@{self.version}"

    @classmethod
    def parse(cls, reference: str) -> "SurvivalRef":
        if not isinstance(reference, str):
            raise InvalidSurvivalRefError("survival reference must be a string")
        match = _REFERENCE_PATTERN.fullmatch(reference)
        if match is None:
            raise InvalidSurvivalRefError(
                "survival reference must use namespace/name@version syntax"
            )
        return cls(
            namespace=match.group("namespace"),
            name=match.group("name"),
            version=int(match.group("version")),
        )


@dataclass(frozen=True)
class SurvivalDescription:
    """Serializable metadata for one registered survival rule."""

    reference: str
    namespace: str
    name: str
    version: int
    origin: SurvivalOrigin
    portable: bool
    description: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "reference": self.reference,
            "namespace": self.namespace,
            "name": self.name,
            "version": self.version,
            "origin": self.origin,
            "portable": self.portable,
            "description": self.description,
        }


@dataclass(frozen=True)
class SurvivalRegistration:
    """A factory and reproducibility metadata for one survival identity."""

    identity: SurvivalRef
    origin: SurvivalOrigin
    portable: bool
    description: str | None
    _factory: SurvivalFactory

    @property
    def reference(self) -> str:
        return self.identity.reference

    def instantiate(self, config: Mapping[str, Any] | None = None) -> nn.Module:
        values = {} if config is None else dict(config)
        operator = self._factory(values)
        if not isinstance(operator, nn.Module):
            raise SurvivalRegistryError(
                f"factory for survival {self.reference!r} must return torch.nn.Module"
            )
        operator_reference = getattr(operator, "reference", None)
        if operator_reference is not None and operator_reference != self.reference:
            raise SurvivalRegistryError(
                f"factory for survival {self.reference!r} returned {operator_reference!r}"
            )
        return operator

    def describe(self) -> SurvivalDescription:
        return SurvivalDescription(
            reference=self.reference,
            namespace=self.identity.namespace,
            name=self.identity.name,
            version=self.identity.version,
            origin=self.origin,
            portable=self.portable,
            description=self.description,
        )


class SurvivalOperator(nn.Module):
    """Base class for a same-shape differentiable survival operator."""

    reference: str | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class SurvivalRegistry:
    """Thread-safe explicit registry with exact-version lookup."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._registrations: dict[str, SurvivalRegistration] = {}

    def register_builtin(
        self,
        reference: str,
        *,
        factory: SurvivalFactory,
        description: str | None = None,
    ) -> SurvivalRegistration:
        identity = SurvivalRef.parse(reference)
        if identity.namespace != "arti":
            raise InvalidSurvivalRefError("builtin survival must use the 'arti' namespace")
        return self._register(
            identity,
            factory=factory,
            origin="builtin",
            portable=True,
            description=description,
        )

    def register(
        self,
        reference: str,
        *,
        factory: SurvivalFactory,
        portable: bool | None = None,
        description: str | None = None,
    ) -> SurvivalRegistration:
        """Register a process-local application survival factory.

        Third-party implementations remain non-portable until an artifact
        authorization contract exists. They can still be used for local
        forward and training experiments.
        """

        identity = SurvivalRef.parse(reference)
        if identity.namespace == "arti":
            raise InvalidSurvivalRefError(
                "the 'arti' namespace is reserved for builtin survival rules"
            )
        if portable:
            raise SurvivalRegistryError(
                "third-party survival cannot declare portable=True yet"
            )
        if not callable(factory):
            raise SurvivalRegistryError("survival factory must be callable")
        return self._register(
            identity,
            factory=factory,
            origin="registered",
            portable=False,
            description=description,
        )

    def _register(
        self,
        identity: SurvivalRef,
        *,
        factory: SurvivalFactory,
        origin: SurvivalOrigin,
        portable: bool,
        description: str | None,
    ) -> SurvivalRegistration:
        if description is not None and not isinstance(description, str):
            raise SurvivalRegistryError("description must be a string or None")
        registration = SurvivalRegistration(
            identity=identity,
            origin=origin,
            portable=portable,
            description=description,
            _factory=factory,
        )
        with self._lock:
            if registration.reference in self._registrations:
                raise DuplicateSurvivalError(
                    f"survival {registration.reference!r} is already registered"
                )
            self._registrations[registration.reference] = registration
        return registration

    def resolve(self, reference: str) -> SurvivalRegistration:
        identity = SurvivalRef.parse(reference)
        with self._lock:
            registration = self._registrations.get(identity.reference)
            known = tuple(sorted(self._registrations))
        if registration is None:
            suffix = f"; registered: {', '.join(known)}" if known else ""
            raise UnknownSurvivalError(
                f"survival {identity.reference!r} is not registered{suffix}"
            )
        return registration

    def list(self) -> tuple[SurvivalDescription, ...]:
        with self._lock:
            return tuple(
                self._registrations[reference].describe()
                for reference in sorted(self._registrations)
            )


def _inverse_base(base: float) -> float:
    eps = 1e-6
    clipped = min(max(float(base), eps), 1.0 - eps)
    normalized = min(max((clipped - eps) / (1.0 - 2.0 * eps), eps), 1.0 - eps)
    return math.log(normalized / (1.0 - normalized))


def _inverse_softplus(value: float) -> float:
    target = max(float(value) - 1e-6, 1e-6)
    return target if target > 20.0 else math.log(math.expm1(target))


class ExponentialSurvival(SurvivalOperator):
    """Built-in ``q = base ** D`` survival rule.

    This standalone module is useful when a developer wants the salience
    curve itself as a configurable component. ``Half`` remains responsible
    for deterministic or stochastic application of the returned ``q``.
    """

    def __init__(
        self,
        threshold: float = 1.0,
        base: float = 0.5,
        scale: float = 1.0,
        *,
        learnable: bool = False,
        context_mode: str = "none",
        context_axes: int | tuple[int, ...] = -1,
        context_gain: float = 0.25,
    ) -> None:
        super().__init__()
        if not math.isfinite(threshold):
            raise ValueError("threshold must be finite")
        if not math.isfinite(base) or not 0 < base <= 1:
            raise ValueError("base must be in the interval (0, 1]")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("scale must be positive")
        if context_mode not in {"none", "contextual"}:
            raise ValueError("context_mode must be 'none' or 'contextual'")
        if not math.isfinite(context_gain) or context_gain < 0:
            raise ValueError("context_gain must be finite and non-negative")
        raw_axes = (context_axes,) if isinstance(context_axes, int) else tuple(context_axes)
        if not raw_axes or any(not isinstance(axis, int) for axis in raw_axes):
            raise ValueError("context_axes must contain at least one integer")
        self.learnable = bool(learnable)
        self.context_mode = context_mode
        self.context_axes = raw_axes
        self.context_gain = float(context_gain)
        self._threshold_init = float(threshold)
        self._base_init = float(base)
        self._scale_init = float(scale)
        if self.learnable:
            self._threshold = nn.Parameter(torch.tensor(float(threshold)))
            self._base_logit = nn.Parameter(torch.tensor(_inverse_base(base)))
            self._scale_raw = nn.Parameter(torch.tensor(_inverse_softplus(scale)))
        else:
            self._threshold_value = float(threshold)
            self._base_value = float(base)
            self._scale_value = float(scale)

    @property
    def reference(self) -> str:
        return (
            "arti/survival@2"
            if self.context_mode == "contextual"
            else "arti/survival@1"
        )

    @property
    def threshold(self) -> float | torch.Tensor:
        return self._threshold if self.learnable else self._threshold_value

    @property
    def base(self) -> float | torch.Tensor:
        if not self.learnable:
            return self._base_value
        eps = 1e-6
        return eps + (1.0 - 2.0 * eps) * torch.sigmoid(self._base_logit)

    @property
    def scale(self) -> float | torch.Tensor:
        if not self.learnable:
            return self._scale_value
        return torch.nn.functional.softplus(self._scale_raw) + 1e-6

    @property
    def config(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "threshold": self._threshold_init,
            "base": self._base_init,
            "scale": self._scale_init,
            "learnable": self.learnable,
        }
        if self.context_mode != "none":
            result.update(
                {
                    "context_mode": self.context_mode,
                    "context_axes": list(self.context_axes),
                    "context_gain": self.context_gain,
                }
            )
        return result

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.context_mode == "contextual":
            return _half_contextual_survival(
                x,
                threshold=self.threshold,
                base=self.base,
                scale=self.scale,
                context_axes=self.context_axes,
                context_gain=self.context_gain,
            )
        return _half_survival(
            x,
            threshold=self.threshold,
            base=self.base,
            scale=self.scale,
        )

    def extra_repr(self) -> str:
        args = [
            f"threshold={float(self.threshold.detach() if isinstance(self.threshold, torch.Tensor) else self.threshold):g}",
            f"base={float(self.base.detach() if isinstance(self.base, torch.Tensor) else self.base):g}",
            f"scale={float(self.scale.detach() if isinstance(self.scale, torch.Tensor) else self.scale):g}",
        ]
        if self.learnable:
            args.append("learnable=True")
        if self.context_mode != "none":
            args.append(f"context_mode={self.context_mode!r}")
        return ", ".join(args)


def _contextual_exponential_factory(config: Mapping[str, Any]) -> nn.Module:
    values = dict(config)
    values["context_mode"] = "contextual"
    return ExponentialSurvival(**values)


_SURVIVAL_REGISTRY = SurvivalRegistry()
_SURVIVAL_REGISTRY.register_builtin(
    "arti/survival@1",
    factory=lambda config: ExponentialSurvival(**config),
    description="Pointwise exponential salience survival.",
)
_SURVIVAL_REGISTRY.register_builtin(
    "arti/survival@2",
    factory=_contextual_exponential_factory,
    description="Same-shape contextual exponential salience survival.",
)


def register_survival(
    reference: str,
    *,
    factory: SurvivalFactory,
    portable: bool | None = None,
    description: str | None = None,
) -> SurvivalRegistration:
    """Register an application-owned survival factory."""

    return _SURVIVAL_REGISTRY.register(
        reference,
        factory=factory,
        portable=portable,
        description=description,
    )


def resolve_survival(reference: str) -> SurvivalRegistration:
    """Resolve an exact survival identity without importing code."""

    return _SURVIVAL_REGISTRY.resolve(reference)


def list_survivals() -> tuple[SurvivalDescription, ...]:
    """Return registered survival descriptions in canonical order."""

    return _SURVIVAL_REGISTRY.list()


def describe_survival(reference: str) -> SurvivalDescription:
    """Describe one exact survival identity."""

    return resolve_survival(reference).describe()


def survival_is_registered(reference: str) -> bool:
    try:
        resolve_survival(reference)
    except (InvalidSurvivalRefError, UnknownSurvivalError):
        return False
    return True


def validate_survival_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate JSON-compatible constructor configuration."""

    values = {} if config is None else dict(config)
    try:
        json.dumps(values, ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise TypeError("survival_config must be JSON-compatible and finite") from error
    return values


__all__ = [
    "DuplicateSurvivalError",
    "ExponentialSurvival",
    "InvalidSurvivalRefError",
    "SurvivalDescription",
    "SurvivalOperator",
    "SurvivalRef",
    "SurvivalRegistration",
    "SurvivalRegistry",
    "SurvivalRegistryError",
    "UnknownSurvivalError",
    "describe_survival",
    "list_survivals",
    "register_survival",
    "resolve_survival",
    "survival_is_registered",
    "validate_survival_config",
]
