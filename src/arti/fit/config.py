"""Declarative configuration for ARTI fit projects."""

from __future__ import annotations

import json
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._toml import loads as load_toml
from .objectives import resolve_objectives
from .plugins import get_plugin
from .profiles import AdapterProfile, resolve_profile
from .runtime import RuntimeFieldConfig
from .scales import AdapterScale, resolve_scale


FIT_CONFIG_SCHEMA_PATH = "docs/reference/fit-config.schema.json"


@dataclass(frozen=True)
class MechanismOverrides:
    """Optional fine-grained overrides for resolved ARTI mechanisms."""

    coord_dim: int | None = None
    coord_frame_mode: str | None = None
    observer_phase: bool | None = None
    virtual_recall: bool | None = None
    operator_count: int | None = None
    interface_slots: int | None = None
    recall_slots: int | None = None
    recall_steps: int | None = None
    recall_activation: str | None = None
    hidden_multiplier: float | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "MechanismOverrides":
        if not payload:
            return cls()
        return cls(
            coord_dim=_optional_int(payload.get("coord_dim")),
            coord_frame_mode=None if payload.get("coord_frame_mode") is None else str(payload["coord_frame_mode"]),
            observer_phase=_optional_bool(payload.get("observer_phase")),
            virtual_recall=_optional_bool(payload.get("virtual_recall")),
            operator_count=_optional_int(payload.get("operator_count")),
            interface_slots=_optional_int(payload.get("interface_slots")),
            recall_slots=_optional_int(payload.get("recall_slots")),
            recall_steps=_optional_int(payload.get("recall_steps")),
            recall_activation=None if payload.get("recall_activation") is None else str(payload["recall_activation"]),
            hidden_multiplier=None if payload.get("hidden_multiplier") is None else float(payload["hidden_multiplier"]),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "coord_dim": self.coord_dim,
            "coord_frame_mode": self.coord_frame_mode,
            "observer_phase": self.observer_phase,
            "virtual_recall": self.virtual_recall,
            "operator_count": self.operator_count,
            "interface_slots": self.interface_slots,
            "recall_slots": self.recall_slots,
            "recall_steps": self.recall_steps,
            "recall_activation": self.recall_activation,
            "hidden_multiplier": self.hidden_multiplier,
        }

    def has_values(self) -> bool:
        return any(value is not None for value in self.to_dict().values())

    def validate(self) -> "MechanismOverrides":
        if self.coord_dim is not None and self.coord_dim < 0:
            raise ValueError("ARTI fit config mechanism.coord_dim must be non-negative")
        if self.coord_frame_mode is not None and self.coord_frame_mode not in {"none", "paired_rotation", "operator_bank"}:
            raise ValueError("ARTI fit config mechanism.coord_frame_mode must be 'none', 'paired_rotation', or 'operator_bank'")
        for key in ("operator_count", "interface_slots", "recall_slots"):
            value = getattr(self, key)
            if value is not None and value <= 0:
                raise ValueError(f"ARTI fit config mechanism.{key} must be positive")
        if self.recall_steps is not None and self.recall_steps < 0:
            raise ValueError("ARTI fit config mechanism.recall_steps must be non-negative")
        if self.recall_activation is not None and self.recall_activation not in {"half", "none"}:
            raise ValueError("ARTI fit config mechanism.recall_activation must be 'half' or 'none'")
        if self.hidden_multiplier is not None and self.hidden_multiplier <= 0:
            raise ValueError("ARTI fit config mechanism.hidden_multiplier must be positive")
        return self


@dataclass(frozen=True)
class FitProjectConfig:
    """Serializable Gradle-like configuration for an ARTI adaptation project."""

    plugins: tuple[str, ...] = ("torch",)
    profile: str = "latent-adapt"
    phases: int | None = None
    scale: str = "small"
    mechanism: MechanismOverrides = MechanismOverrides()
    causal: bool = False
    runtime_fields: RuntimeFieldConfig = RuntimeFieldConfig()
    objectives: tuple[str, ...] = ()
    where: tuple[str, ...] | None = None
    every: int = 1
    freeze_base: bool = True
    max_adapters: int | None = None
    max_extra_params: int | str | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "FitProjectConfig":
        fit = _section(payload, "fit")
        runtime = _section(payload, "runtime")
        mechanism = _section(payload, "mechanism")
        insertion = _section(payload, "insertion")
        where = insertion.get("where", fit.get("target_modules"))
        return cls(
            plugins=_as_tuple(fit.get("plugins", payload.get("plugins", ("torch",)))),
            profile=str(fit.get("profile", payload.get("profile", "latent-adapt"))),
            phases=_optional_int(fit.get("phases", payload.get("phases"))),
            scale=str(fit.get("scale", payload.get("scale", "small"))),
            mechanism=MechanismOverrides.from_mapping(mechanism),
            causal=bool(runtime.get("causal", fit.get("causal", payload.get("causal", False)))),
            runtime_fields=RuntimeFieldConfig.from_mapping(runtime),
            objectives=_as_tuple(fit.get("objectives", fit.get("objective", payload.get("objectives", payload.get("objective", ()))))),
            where=None if where is None else _as_tuple(where),
            every=int(insertion.get("every", 1)),
            freeze_base=bool(insertion.get("freeze_base", True)),
            max_adapters=_optional_int(insertion.get("max_adapters")),
            max_extra_params=insertion.get("max_extra_params"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "plugins": list(self.plugins),
            "profile": self.profile,
            "phases": self.phases,
            "scale": self.scale,
            "mechanism": self.mechanism.to_dict(),
            "runtime": {"causal": self.causal, **self.runtime_fields.to_dict()},
            "objectives": list(self.objectives),
            "insertion": {
                "where": None if self.where is None else list(self.where),
                "every": self.every,
                "freeze_base": self.freeze_base,
                "max_adapters": self.max_adapters,
                "max_extra_params": self.max_extra_params,
            },
        }

    @property
    def fingerprint(self) -> str:
        """Stable SHA-256 fingerprint for the normalized config."""

        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def validate(self) -> "FitProjectConfig":
        """Validate this config against ARTI registries and numeric constraints."""

        for plugin in self.plugins:
            get_plugin(plugin)
        resolve_profile(self.profile, phases=self.phases)
        resolve_scale(self.scale)
        self.mechanism.validate()
        resolve_objectives(self.objectives)
        if self.every <= 0:
            raise ValueError("ARTI fit config insertion.every must be positive")
        if self.max_adapters is not None and self.max_adapters < 0:
            raise ValueError("ARTI fit config insertion.max_adapters must be non-negative")
        if isinstance(self.max_extra_params, int) and self.max_extra_params < 0:
            raise ValueError("ARTI fit config insertion.max_extra_params must be non-negative")
        if isinstance(self.max_extra_params, str):
            stripped = self.max_extra_params.strip()
            if stripped.endswith("%"):
                if float(stripped[:-1]) < 0:
                    raise ValueError("ARTI fit config insertion.max_extra_params percent must be non-negative")
            elif int(stripped.replace("_", "")) < 0:
                raise ValueError("ARTI fit config insertion.max_extra_params must be non-negative")
        return self


def load_fit_config(path: str | Path) -> FitProjectConfig:
    """Load an ARTI fit project config from JSON or TOML."""

    target = Path(path)
    suffix = target.suffix.lower()
    if suffix == ".json":
        payload = json.loads(target.read_text(encoding="utf-8"))
    elif suffix == ".toml":
        payload = load_toml(target.read_text(encoding="utf-8"))
    else:
        raise ValueError("ARTI fit config must be a .json or .toml file")
    if not isinstance(payload, dict):
        raise ValueError("ARTI fit config must contain a mapping at the top level")
    return validate_fit_config(FitProjectConfig.from_mapping(payload))


def validate_fit_config(config: FitProjectConfig | dict[str, Any]) -> FitProjectConfig:
    """Validate and return a normalized ARTI fit project config."""

    resolved = FitProjectConfig.from_mapping(config) if isinstance(config, dict) else config
    return resolved.validate()


def apply_mechanism_overrides(profile: AdapterProfile, scale: AdapterScale, overrides: MechanismOverrides) -> tuple[AdapterProfile, AdapterScale]:
    """Apply fine-grained mechanism overrides to resolved profile and scale objects."""

    overrides.validate()
    if not overrides.has_values():
        return profile, scale
    observer_phase = profile.observer_phase if overrides.observer_phase is None else overrides.observer_phase
    coord_dim = profile.coord_dim if overrides.coord_dim is None else overrides.coord_dim
    coord_frame_mode = profile.coord_frame_mode if overrides.coord_frame_mode is None else overrides.coord_frame_mode
    if not observer_phase:
        coord_dim = 0
        coord_frame_mode = "none"
    return (
        AdapterProfile(
            name=profile.name,
            coord_frame_mode=coord_frame_mode,
            coord_dim=coord_dim,
            virtual_recall=profile.virtual_recall if overrides.virtual_recall is None else overrides.virtual_recall,
            observer_phase=observer_phase,
        ),
        AdapterScale(
            hidden_multiplier=scale.hidden_multiplier if overrides.hidden_multiplier is None else overrides.hidden_multiplier,
            interface_slots=scale.interface_slots if overrides.interface_slots is None else overrides.interface_slots,
            recall_slots=scale.recall_slots if overrides.recall_slots is None else overrides.recall_slots,
            recall_steps=scale.recall_steps if overrides.recall_steps is None else overrides.recall_steps,
            recall_activation=scale.recall_activation if overrides.recall_activation is None else overrides.recall_activation,
            operator_count=scale.operator_count if overrides.operator_count is None else overrides.operator_count,
        ),
    )


def resolve_fit_config_mechanism(config: FitProjectConfig | dict[str, Any]) -> tuple[AdapterProfile, AdapterScale]:
    """Resolve a fit config into concrete profile and scale mechanism objects."""

    resolved = validate_fit_config(config)
    profile = resolve_profile(resolved.profile, phases=resolved.phases)
    scale = resolve_scale(resolved.scale)
    return apply_mechanism_overrides(profile, scale, resolved.mechanism)


def template_fit_config(*, profile: str = "latent-adapt", scale: str = "small") -> FitProjectConfig:
    """Return a conservative starter config for an ARTI fit project."""

    return FitProjectConfig(
        plugins=("torch",),
        profile=profile,
        scale=scale,
        mechanism=MechanismOverrides(),
        runtime_fields=RuntimeFieldConfig(),
        objectives=(),
        where=("*",),
        max_adapters=4,
        max_extra_params="1%",
    ).validate()


def write_fit_config_template(path: str | Path, *, profile: str = "latent-adapt", scale: str = "small", overwrite: bool = False) -> Path:
    """Write a JSON or TOML starter config."""

    target = Path(path)
    if target.exists() and not overwrite:
        raise FileExistsError(f"ARTI fit config already exists: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    config = template_fit_config(profile=profile, scale=scale)
    suffix = target.suffix.lower()
    if suffix == ".json":
        payload = {"$schema": FIT_CONFIG_SCHEMA_PATH, **config.to_dict()}
        target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    elif suffix == ".toml":
        target.write_text(_to_toml(config), encoding="utf-8")
    else:
        raise ValueError("ARTI fit config template path must end with .json or .toml")
    return target


def _to_toml(config: FitProjectConfig) -> str:
    data = config.to_dict()
    insertion = data["insertion"]
    mechanism = data["mechanism"]
    runtime = data["runtime"]
    mechanism_lines = ["[mechanism]"]
    mechanism_lines.extend(f"{key} = {_toml_value(value)}" for key, value in mechanism.items() if value is not None)
    mechanism_lines.append("")
    runtime_lines = ["[runtime]", f"causal = {str(runtime['causal']).lower()}"]
    runtime_lines.extend(
        f"{key} = {_toml_value(value)}"
        for key, value in runtime.items()
        if key != "causal" and value is not None
    )
    runtime_lines.append("")
    lines = [
        "[fit]",
        f"plugins = {_toml_list(data['plugins'])}",
        f"profile = {_toml_string(data['profile'])}",
        f"scale = {_toml_string(data['scale'])}",
        f"objectives = {_toml_list(data['objectives'])}",
        "",
        *mechanism_lines,
        *runtime_lines,
        "[insertion]",
        f"where = {_toml_value(insertion['where'])}",
        f"every = {insertion['every']}",
        f"freeze_base = {str(insertion['freeze_base']).lower()}",
        f"max_adapters = {_toml_value(insertion['max_adapters'])}",
        f"max_extra_params = {_toml_value(insertion['max_extra_params'])}",
        "",
    ]
    if data["phases"] is not None:
        lines.insert(4, f"phases = {data['phases']}")
    return "\n".join(lines)


def _toml_value(value: Any) -> str:
    if value is None:
        return '""'
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, list):
        return _toml_list(value)
    return _toml_string(str(value))


def _toml_list(values: list[Any]) -> str:
    return "[" + ", ".join(_toml_value(value) for value in values) + "]"


def _toml_string(value: str) -> str:
    return json.dumps(value)


def _section(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"ARTI fit config section {key!r} must be a mapping")
    return value


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, (list, tuple)):
        return tuple(str(item) for item in value)
    raise ValueError("ARTI fit config value must be a string or list of strings")


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    raise ValueError("ARTI fit config mechanism boolean values must be booleans")
