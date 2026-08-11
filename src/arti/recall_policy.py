"""Auditable optimization metadata for custom Recall formulas.

This module deliberately contains no optimizer or training-loop behavior. Formula
requirements validate a resolved host policy, while formula hints participate only
when the host explicitly accepts them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Union


JSONValue = Union[
    None,
    bool,
    int,
    float,
    str,
    tuple["JSONValue", ...],
    Mapping[str, "JSONValue"],
]

_FORMULA_POLICY_KEYS = frozenset(
    {
        "clip_grad_norm",
        "clip_scope",
        "compute_dtype",
        "fp32_master_weights",
        "gradient_layout_by_role",
        "lr_scale_by_role",
        "parameter_dtype",
        "parameter_dtype_by_role",
        "reduction_dtype",
        "sparse_gradients",
        "sparse_update_mode",
        "weight_decay",
    }
)
_FORBIDDEN_FORMULA_FIELD_PARTS = (
    "optimizer",
    "hook",
    "requires_grad",
    "grad_mutation",
    "gradient_mutation",
    "mutate_grad",
    "custom_backward",
)
_GRADIENT_LAYOUTS = frozenset({"dense", "row_sparse"})
_POLICY_SOURCES = ("framework", "formula", "project", "user")


def _normalize_field_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def _reject_formula_field(name: str) -> None:
    normalized = _normalize_field_name(name)
    if any(part in normalized for part in _FORBIDDEN_FORMULA_FIELD_PARTS):
        raise ValueError(
            f"formula optimization metadata cannot define unsafe field {name!r}; "
            "formulas cannot inject optimizers, hooks, custom backward behavior, "
            "requires_grad changes, or gradient mutation"
        )
    if normalized not in _FORMULA_POLICY_KEYS:
        supported = ", ".join(sorted(_FORMULA_POLICY_KEYS))
        raise ValueError(
            f"unsupported formula optimization field {name!r}; expected one of: {supported}"
        )


def _freeze_json(value: Any, *, path: str = "value") -> JSONValue:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, JSONValue] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise TypeError(f"{path} mapping keys must be non-empty strings")
            frozen[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]") for index, item in enumerate(value)
        )
    raise TypeError(
        f"{path} must be JSON-safe; received {type(value).__name__}. "
        "Only null, booleans, finite numbers, strings, arrays, and string-keyed objects are allowed"
    )


def _thaw_json(value: JSONValue) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _freeze_mapping(value: Mapping[str, Any] | None, *, path: str) -> Mapping[str, JSONValue]:
    if value is not None and not isinstance(value, Mapping):
        raise TypeError(f"{path} must be a string-keyed mapping")
    frozen = _freeze_json({} if value is None else value, path=path)
    assert isinstance(frozen, Mapping)
    return frozen


def _validate_formula_values(values: Mapping[str, JSONValue]) -> None:
    for name in values:
        _reject_formula_field(name)


def _flatten(value: Mapping[str, JSONValue], prefix: str = "") -> dict[str, JSONValue]:
    flattened: dict[str, JSONValue] = {}
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else key
        if isinstance(item, Mapping):
            flattened.update(_flatten(item, path))
        else:
            flattened[path] = item
    return flattened


def _lookup_path(values: Mapping[str, JSONValue], path: str) -> tuple[bool, JSONValue]:
    current: JSONValue = values
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return False, None
        current = current[part]
    return True, current


def _merge_policy(
    target: dict[str, JSONValue],
    incoming: Mapping[str, JSONValue],
    *,
    source: str,
    sources: dict[str, str],
    prefix: str = "",
) -> None:
    for key, value in incoming.items():
        path = f"{prefix}.{key}" if prefix else key
        existing = target.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            mutable = dict(existing)
            _merge_policy(mutable, value, source=source, sources=sources, prefix=path)
            target[key] = MappingProxyType(mutable)
        else:
            for stale_path in tuple(sources):
                if stale_path == path or stale_path.startswith(f"{path}."):
                    del sources[stale_path]
            target[key] = value
            if isinstance(value, Mapping):
                for child_path in _flatten(value, path):
                    sources[child_path] = source
            else:
                sources[path] = source


@dataclass(frozen=True)
class FormulaOptimizationRequirements:
    """Policy values a formula requires but never supplies.

    Missing values fail validation. This is intentional: a requirement is not a
    hidden default and cannot silently change the host's optimization policy.
    """

    required_values: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frozen = _freeze_mapping(self.required_values, path="required_values")
        _validate_formula_values(frozen)
        object.__setattr__(self, "required_values", frozen)

    def validate(self, policy: Mapping[str, Any]) -> None:
        resolved = _freeze_mapping(policy, path="policy")
        for path, expected in _flatten(self.required_values).items():
            present, actual = _lookup_path(resolved, path)
            if not present:
                raise ValueError(
                    f"resolved optimization policy is missing required field {path!r}; "
                    "requirements validate policy and do not provide defaults"
                )
            if actual != expected:
                raise ValueError(
                    f"resolved optimization policy field {path!r} must equal "
                    f"{_thaw_json(expected)!r}, received {_thaw_json(actual)!r}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {"required_values": _thaw_json(self.required_values)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FormulaOptimizationRequirements":
        unknown = set(value) - {"required_values"}
        if unknown:
            raise ValueError(f"unknown FormulaOptimizationRequirements fields: {sorted(unknown)}")
        return cls(required_values=value.get("required_values", {}))


@dataclass(frozen=True)
class FormulaOptimizationHints:
    """Optional formula advice applied only with explicit host acceptance."""

    values: Mapping[str, JSONValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        frozen = _freeze_mapping(self.values, path="values")
        _validate_formula_values(frozen)
        object.__setattr__(self, "values", frozen)

    def to_dict(self) -> dict[str, Any]:
        return {"values": _thaw_json(self.values)}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FormulaOptimizationHints":
        unknown = set(value) - {"values"}
        if unknown:
            raise ValueError(f"unknown FormulaOptimizationHints fields: {sorted(unknown)}")
        return cls(values=value.get("values", {}))


@dataclass(frozen=True)
class RecallParameterTag:
    """Serializable ownership metadata for one named Recall parameter."""

    parameter_name: str
    role: str
    storage_group: str
    factor: str | None = None
    gradient_layout: str = "dense"

    def __post_init__(self) -> None:
        for field_name in ("parameter_name", "role", "storage_group"):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
        if self.factor is not None and (
            not isinstance(self.factor, str) or not self.factor.strip()
        ):
            raise ValueError("factor must be a non-empty string when provided")
        if self.gradient_layout not in _GRADIENT_LAYOUTS:
            raise ValueError(f"gradient_layout must be one of {sorted(_GRADIENT_LAYOUTS)}")

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "parameter_name": self.parameter_name,
            "role": self.role,
            "storage_group": self.storage_group,
            "factor": self.factor,
            "gradient_layout": self.gradient_layout,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RecallParameterTag":
        allowed = {"parameter_name", "role", "storage_group", "factor", "gradient_layout"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown RecallParameterTag fields: {sorted(unknown)}")
        try:
            return cls(
                parameter_name=value["parameter_name"],
                role=value["role"],
                storage_group=value["storage_group"],
                factor=value.get("factor"),
                gradient_layout=value.get("gradient_layout", "dense"),
            )
        except KeyError as error:
            raise ValueError(f"missing RecallParameterTag field: {error.args[0]}") from error


def validate_parameter_tags(
    tags: Iterable[RecallParameterTag],
    *,
    parameter_names: Iterable[str] | None = None,
) -> tuple[RecallParameterTag, ...]:
    """Validate unique tag ownership and optional exact parameter coverage."""

    resolved = tuple(tags)
    names = [tag.parameter_name for tag in resolved]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"parameters must have exactly one RecallParameterTag: {duplicates}")
    if parameter_names is not None:
        expected = set(parameter_names)
        actual = set(names)
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        if missing or unexpected:
            raise ValueError(
                "parameter tag coverage mismatch: "
                f"missing={missing or 'none'}, unexpected={unexpected or 'none'}"
            )
    return resolved


@dataclass(frozen=True)
class ResolvedFormulaOptimizationPolicy:
    """Resolved values plus a leaf-level provenance map."""

    values: Mapping[str, JSONValue]
    sources: Mapping[str, str]
    formula_hints_accepted: bool

    def __post_init__(self) -> None:
        values = _freeze_mapping(self.values, path="values")
        sources = dict(self.sources)
        leaf_paths = set(_flatten(values))
        if set(sources) != leaf_paths:
            raise ValueError("sources must identify every resolved policy leaf exactly once")
        if any(source not in _POLICY_SOURCES for source in sources.values()):
            raise ValueError(f"policy sources must be one of {_POLICY_SOURCES}")
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "sources", MappingProxyType(sources))

    def to_dict(self) -> dict[str, Any]:
        return {
            "values": _thaw_json(self.values),
            "sources": dict(self.sources),
            "formula_hints_accepted": self.formula_hints_accepted,
        }


def resolve_formula_optimization_policy(
    *,
    framework: Mapping[str, Any] | None = None,
    formula_hints: FormulaOptimizationHints | None = None,
    accept_formula_hints: bool = False,
    project: Mapping[str, Any] | None = None,
    user: Mapping[str, Any] | None = None,
    requirements: FormulaOptimizationRequirements | None = None,
) -> ResolvedFormulaOptimizationPolicy:
    """Resolve policy with ``user > project > accepted formula > framework`` priority."""

    layers: list[tuple[str, Mapping[str, JSONValue]]] = [
        ("framework", _freeze_mapping(framework, path="framework")),
    ]
    if formula_hints is not None and accept_formula_hints:
        layers.append(("formula", formula_hints.values))
    layers.extend(
        (
            ("project", _freeze_mapping(project, path="project")),
            ("user", _freeze_mapping(user, path="user")),
        )
    )

    values: dict[str, JSONValue] = {}
    sources: dict[str, str] = {}
    for source, layer in layers:
        _merge_policy(values, layer, source=source, sources=sources)
    resolved = ResolvedFormulaOptimizationPolicy(
        values=values,
        sources=sources,
        formula_hints_accepted=formula_hints is not None and accept_formula_hints,
    )
    if requirements is not None:
        requirements.validate(resolved.values)
    return resolved


__all__ = [
    "FormulaOptimizationHints",
    "FormulaOptimizationRequirements",
    "RecallParameterTag",
    "ResolvedFormulaOptimizationPolicy",
    "resolve_formula_optimization_policy",
    "validate_parameter_tags",
]
