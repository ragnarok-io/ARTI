"""Pure-data manifests for versioned Recall formula contracts."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import Any, Mapping


RECALL_FORMULA_API_VERSION = 1
RECALL_LAYOUT_VERSION = 1

_ORIGINS = frozenset({"builtin", "registered", "custom"})
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._/-]{0,127}$")
_FACTOR_NAME = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_EXACT_VERSION = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._+-]{0,63}$")
_CAPABILITY = re.compile(r"^[a-z0-9][a-z0-9._-]{0,63}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def canonical_json(value: Any) -> str:
    """Encode a validated manifest payload using its stable JSON representation."""

    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    except (TypeError, ValueError) as error:
        raise TypeError("Recall manifest values must be finite JSON data") from error


def canonical_sha256(value: Any) -> str:
    """Return the lowercase SHA-256 digest of canonical JSON data."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class RecallLayoutManifest:
    """Stable ordering contract for factors read from a Recall bank."""

    factor_order: tuple[str, ...]
    layout_version: int = RECALL_LAYOUT_VERSION

    def __post_init__(self) -> None:
        _require_plain_int("layout_version", self.layout_version)
        if self.layout_version != RECALL_LAYOUT_VERSION:
            raise ValueError(
                f"unsupported Recall layout_version={self.layout_version!r}; "
                f"expected {RECALL_LAYOUT_VERSION}"
            )
        _validate_factor_names("factor_order", self.factor_order)

    @property
    def fingerprint(self) -> str:
        """Content fingerprint excluding the fingerprint field itself."""

        return canonical_sha256(self._content_dict())

    def _content_dict(self) -> dict[str, Any]:
        return {
            "layout_version": self.layout_version,
            "factor_order": list(self.factor_order),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe payload with a verifiable layout fingerprint."""

        return {**self._content_dict(), "fingerprint": self.fingerprint}

    def canonical_json(self) -> str:
        """Return the canonical JSON form used by artifact hashing."""

        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecallLayoutManifest":
        """Restore and strictly validate a pure-data layout manifest."""

        payload = _require_mapping("Recall layout manifest", value)
        _require_exact_keys(
            "Recall layout manifest",
            payload,
            required={"layout_version", "factor_order", "fingerprint"},
        )
        factor_order = _require_string_list("factor_order", payload["factor_order"])
        fingerprint = _require_string("fingerprint", payload["fingerprint"])
        if not _SHA256.fullmatch(fingerprint):
            raise ValueError("Recall layout fingerprint must be a lowercase SHA-256 digest")
        result = cls(
            factor_order=factor_order,
            layout_version=_require_plain_int("layout_version", payload["layout_version"]),
        )
        if fingerprint != result.fingerprint:
            raise ValueError("Recall layout fingerprint does not match its contents")
        return result


@dataclass(frozen=True)
class RecallFormulaManifest:
    """Portable identity and capability contract for one Recall formula."""

    id: str
    version: str
    factor_names: tuple[str, ...]
    origin: str
    portable: bool
    layout: RecallLayoutManifest
    capabilities: tuple[str, ...] = ()
    api_version: int = RECALL_FORMULA_API_VERSION

    def __post_init__(self) -> None:
        _require_plain_int("api_version", self.api_version)
        if self.api_version != RECALL_FORMULA_API_VERSION:
            raise ValueError(
                f"unsupported Recall formula api_version={self.api_version!r}; "
                f"expected {RECALL_FORMULA_API_VERSION}"
            )
        _validate_identifier("id", self.id, _IDENTIFIER)
        _validate_identifier("version", self.version, _EXACT_VERSION)
        if any(character in self.version for character in "<>=*^~!, \t\r\n"):
            raise ValueError("Recall formula version must be one exact version, not a range")
        _validate_factor_names("factor_names", self.factor_names)
        origin = _require_string("origin", self.origin)
        if origin not in _ORIGINS:
            raise ValueError("Recall formula origin must be 'builtin', 'registered', or 'custom'")
        if type(self.portable) is not bool:
            raise TypeError("portable must be a boolean")
        if origin != "builtin" and self.portable:
            raise ValueError(
                "only builtin Recall formulas can declare portable=true in formula API v1"
            )
        if not isinstance(self.layout, RecallLayoutManifest):
            raise TypeError("layout must be a RecallLayoutManifest")
        if self.factor_names != self.layout.factor_order:
            raise ValueError("factor_names must exactly match layout.factor_order")
        _validate_capabilities(self.capabilities)
        if origin == "builtin" and self.portable:
            self._validate_builtin_portability()

    def _validate_builtin_portability(self) -> None:
        from .recall_formula import BUILTIN_RECALL_FORMULAS

        matches = []
        for description in BUILTIN_RECALL_FORMULAS.values():
            identity = description.contract.identity
            if identity is None:
                continue
            if identity.name == self.id and str(identity.version) == self.version:
                matches.append(description.contract.factor_names)
        if self.factor_names not in matches:
            raise ValueError(
                "portable builtin Recall manifest does not match an immutable "
                "builtin formula identity and factor layout"
            )

    @property
    def layout_fingerprint(self) -> str:
        """Return the exact layout identity bound by this formula."""

        return self.layout.fingerprint

    @property
    def sha256(self) -> str:
        """Return the canonical manifest SHA-256 digest."""

        return canonical_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe manifest containing no executable references."""

        return {
            "api_version": self.api_version,
            "id": self.id,
            "version": self.version,
            "factor_names": list(self.factor_names),
            "origin": self.origin,
            "portable": self.portable,
            "layout": self.layout.to_dict(),
            "layout_fingerprint": self.layout_fingerprint,
            "capabilities": list(self.capabilities),
        }

    def canonical_json(self) -> str:
        """Return the exact canonical JSON used by :attr:`sha256`."""

        return canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecallFormulaManifest":
        """Restore a formula manifest without importing or resolving code."""

        payload = _require_mapping("Recall formula manifest", value)
        _require_exact_keys(
            "Recall formula manifest",
            payload,
            required={
                "api_version",
                "id",
                "version",
                "factor_names",
                "origin",
                "portable",
                "layout",
                "layout_fingerprint",
                "capabilities",
            },
        )
        layout_payload = _require_mapping("layout", payload["layout"])
        layout = RecallLayoutManifest.from_dict(layout_payload)
        layout_fingerprint = _require_string(
            "layout_fingerprint", payload["layout_fingerprint"]
        )
        if not _SHA256.fullmatch(layout_fingerprint):
            raise ValueError("layout_fingerprint must be a lowercase SHA-256 digest")
        if layout_fingerprint != layout.fingerprint:
            raise ValueError("layout_fingerprint does not match the embedded layout")
        return cls(
            api_version=_require_plain_int("api_version", payload["api_version"]),
            id=_require_string("id", payload["id"]),
            version=_require_string("version", payload["version"]),
            factor_names=_require_string_list("factor_names", payload["factor_names"]),
            origin=_require_string("origin", payload["origin"]),
            portable=_require_bool("portable", payload["portable"]),
            layout=layout,
            capabilities=_require_string_list("capabilities", payload["capabilities"]),
        )


def _require_mapping(name: str, value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    for key in value:
        if type(key) is not str:
            raise TypeError(f"{name} keys must be strings")
    return dict(value)


def _require_exact_keys(name: str, payload: Mapping[str, Any], *, required: set[str]) -> None:
    missing = sorted(required - set(payload))
    unknown = sorted(set(payload) - required)
    if missing:
        raise ValueError(f"{name} is missing required fields: {missing}")
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {unknown}")


def _require_plain_int(name: str, value: Any) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    return value


def _require_bool(name: str, value: Any) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{name} must be a boolean")
    return value


def _require_string(name: str, value: Any) -> str:
    if type(value) is not str:
        raise TypeError(f"{name} must be a string")
    return value


def _require_string_list(name: str, value: Any) -> tuple[str, ...]:
    if type(value) is not list:
        raise TypeError(f"{name} must be a JSON array")
    if any(type(item) is not str for item in value):
        raise TypeError(f"{name} must contain only strings")
    return tuple(value)


def _validate_identifier(name: str, value: Any, pattern: re.Pattern[str]) -> None:
    text = _require_string(name, value)
    if not pattern.fullmatch(text):
        raise ValueError(f"Recall formula {name} has an invalid format")


def _validate_factor_names(name: str, value: Any) -> None:
    if type(value) is not tuple:
        raise TypeError(f"{name} must be a tuple of strings")
    if not value:
        raise ValueError(f"{name} must contain at least one factor")
    if any(type(item) is not str or not _FACTOR_NAME.fullmatch(item) for item in value):
        raise ValueError(f"{name} contains an invalid factor name")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} must not contain duplicate factors")


def _validate_capabilities(value: Any) -> None:
    if type(value) is not tuple:
        raise TypeError("capabilities must be a tuple of strings")
    if any(type(item) is not str or not _CAPABILITY.fullmatch(item) for item in value):
        raise ValueError("capabilities contains an invalid capability")
    if len(set(value)) != len(value):
        raise ValueError("capabilities must not contain duplicates")
    if tuple(sorted(value)) != value:
        raise ValueError("capabilities must be sorted for canonical serialization")


__all__ = [
    "RECALL_FORMULA_API_VERSION",
    "RECALL_LAYOUT_VERSION",
    "RecallFormulaManifest",
    "RecallLayoutManifest",
    "canonical_json",
    "canonical_sha256",
]
