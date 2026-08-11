"""Declarative configuration for ARTI fit projects."""

from __future__ import annotations

import json
import hashlib
import math
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
    recall_min_steps: int | None = None
    recall_tolerance: float | None = None
    recall_activation: str | None = None
    recall_recognition_mode: str | None = None
    recall_bank_fraction: float | None = None
    recall_routing: str | None = None
    recall_key_dim: int | None = None
    recall_group_size: int | None = None
    recall_group_topk: int | None = None
    recall_value_composition: str | None = None
    recall_formula: str | None = None
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
            recall_min_steps=_optional_int(payload.get("recall_min_steps")),
            recall_tolerance=None
            if payload.get("recall_tolerance") is None
            else float(payload["recall_tolerance"]),
            recall_activation=None if payload.get("recall_activation") is None else str(payload["recall_activation"]),
            recall_recognition_mode=None
            if payload.get("recall_recognition_mode") is None
            else str(payload["recall_recognition_mode"]),
            recall_bank_fraction=None
            if payload.get("recall_bank_fraction") is None
            else float(payload["recall_bank_fraction"]),
            recall_routing=None
            if payload.get("recall_routing") is None
            else str(payload["recall_routing"]),
            recall_key_dim=_optional_int(payload.get("recall_key_dim")),
            recall_group_size=_optional_int(payload.get("recall_group_size")),
            recall_group_topk=_optional_int(payload.get("recall_group_topk")),
            recall_value_composition=None
            if payload.get("recall_value_composition") is None
            else str(payload["recall_value_composition"]),
            recall_formula=None
            if payload.get("recall_formula") is None
            else str(payload["recall_formula"]),
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
            "recall_min_steps": self.recall_min_steps,
            "recall_tolerance": self.recall_tolerance,
            "recall_activation": self.recall_activation,
            "recall_recognition_mode": self.recall_recognition_mode,
            "recall_bank_fraction": self.recall_bank_fraction,
            "recall_routing": self.recall_routing,
            "recall_key_dim": self.recall_key_dim,
            "recall_group_size": self.recall_group_size,
            "recall_group_topk": self.recall_group_topk,
            "recall_value_composition": self.recall_value_composition,
            "recall_formula": self.recall_formula,
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
        if self.recall_min_steps is not None and self.recall_min_steps < 0:
            raise ValueError("ARTI fit config mechanism.recall_min_steps must be non-negative")
        if self.recall_tolerance is not None and (
            not math.isfinite(self.recall_tolerance) or self.recall_tolerance < 0
        ):
            raise ValueError(
                "ARTI fit config mechanism.recall_tolerance must be finite and non-negative"
            )
        if (
            self.recall_steps is not None
            and self.recall_steps > 0
            and self.recall_min_steps is not None
            and not 1 <= self.recall_min_steps <= self.recall_steps
        ):
            raise ValueError(
                "ARTI fit config mechanism.recall_min_steps must be in "
                "[1, recall_steps]"
            )
        if self.recall_activation is not None and self.recall_activation not in {"half", "none"}:
            raise ValueError("ARTI fit config mechanism.recall_activation must be 'half' or 'none'")
        if self.recall_recognition_mode is not None and self.recall_recognition_mode not in {
            "explicit",
            "alignment",
            "none",
        }:
            raise ValueError(
                "ARTI fit config mechanism.recall_recognition_mode must be "
                "'explicit', 'alignment', or 'none'"
            )
        if self.recall_bank_fraction is not None and not 0.5 < self.recall_bank_fraction < 1.0:
            raise ValueError(
                "ARTI fit config mechanism.recall_bank_fraction must be in (0.5, 1)"
            )
        if self.recall_bank_fraction is not None and self.recall_slots is not None:
            raise ValueError(
                "ARTI fit config mechanism.recall_bank_fraction and "
                "mechanism.recall_slots are mutually exclusive; use the fraction "
                "for automatic bank sizing or slots for an explicit shape"
            )
        if self.recall_routing is not None and self.recall_routing not in {
            "dense",
            "grouped",
        }:
            raise ValueError(
                "ARTI fit config mechanism.recall_routing must be 'dense' or 'grouped'"
            )
        if self.recall_value_composition is not None and self.recall_value_composition not in {
            "single",
            "product",
            "state",
        }:
            raise ValueError(
                "ARTI fit config mechanism.recall_value_composition must be "
                "'single', 'product', or 'state'"
            )
        if self.recall_formula is not None:
            from ..recall_registry import RecallFormulaId

            RecallFormulaId.parse(self.recall_formula)
            if self.recall_value_composition not in {None, "single"}:
                raise ValueError(
                    "ARTI fit config mechanism.recall_formula cannot be combined "
                    "with a non-single legacy recall_value_composition"
                )
        for key in ("recall_key_dim", "recall_group_size", "recall_group_topk"):
            value = getattr(self, key)
            if value is not None and value <= 0:
                raise ValueError(f"ARTI fit config mechanism.{key} must be positive")
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
    exclude: tuple[str, ...] = ()
    positions: tuple[str, ...] = ("output",)
    scale_pattern: tuple[tuple[str, str], ...] = ()
    every: int = 1
    freeze_base: bool = True
    identity_gate: bool = False
    zero_init_output: bool = False
    bridge_mode: str = "radial"
    boundary_mask_key: str | None = None
    require_runtime_context: bool = False
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
            exclude=_as_tuple(insertion.get("exclude", fit.get("exclude_modules", ()))),
            positions=_as_tuple(insertion.get("positions", ("output",))),
            scale_pattern=_as_scale_pattern(insertion.get("scale_pattern", {})),
            every=int(insertion.get("every", 1)),
            freeze_base=bool(insertion.get("freeze_base", True)),
            identity_gate=bool(insertion.get("identity_gate", False)),
            zero_init_output=bool(insertion.get("zero_init_output", False)),
            bridge_mode=str(insertion.get("bridge_mode", "radial")),
            boundary_mask_key=_optional_str(insertion.get("boundary_mask_key")),
            require_runtime_context=bool(
                insertion.get("require_runtime_context", False)
            ),
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
                "exclude": list(self.exclude),
                "positions": list(self.positions),
                "scale_pattern": dict(self.scale_pattern),
                "every": self.every,
                "freeze_base": self.freeze_base,
                "identity_gate": self.identity_gate,
                "zero_init_output": self.zero_init_output,
                "bridge_mode": self.bridge_mode,
                "boundary_mask_key": self.boundary_mask_key,
                "require_runtime_context": self.require_runtime_context,
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
        if self.identity_gate and self.zero_init_output:
            raise ValueError(
                "ARTI fit config insertion.identity_gate and zero_init_output are mutually exclusive"
            )
        if self.bridge_mode not in {"radial", "dense"}:
            raise ValueError("ARTI fit config insertion.bridge_mode must be 'radial' or 'dense'")
        if self.boundary_mask_key == "":
            raise ValueError("ARTI fit config insertion.boundary_mask_key must not be empty")
        if self.every <= 0:
            raise ValueError("ARTI fit config insertion.every must be positive")
        if not self.positions or set(self.positions) - {"input", "output"}:
            raise ValueError("ARTI fit config insertion.positions must contain 'input' and/or 'output'")
        for _, scale_name in self.scale_pattern:
            resolve_scale(scale_name)
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
    explicit_recall_shape = (
        overrides.recall_slots is not None or overrides.hidden_multiplier is not None
    )
    recall_bank_fraction = (
        overrides.recall_bank_fraction
        if overrides.recall_bank_fraction is not None
        else None
        if explicit_recall_shape
        else scale.recall_bank_fraction
    )
    recall_routing = (
        overrides.recall_routing
        if overrides.recall_routing is not None
        else "dense"
        if recall_bank_fraction is None and explicit_recall_shape
        else scale.recall_routing
    )
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
            recall_min_steps=scale.recall_min_steps
            if overrides.recall_min_steps is None
            else overrides.recall_min_steps,
            recall_tolerance=scale.recall_tolerance
            if overrides.recall_tolerance is None
            else overrides.recall_tolerance,
            recall_activation=scale.recall_activation if overrides.recall_activation is None else overrides.recall_activation,
            recall_recognition_mode=scale.recall_recognition_mode
            if overrides.recall_recognition_mode is None
            else overrides.recall_recognition_mode,
            recall_bank_fraction=recall_bank_fraction,
            recall_routing=recall_routing,
            recall_key_dim=scale.recall_key_dim
            if overrides.recall_key_dim is None
            else overrides.recall_key_dim,
            recall_group_size=scale.recall_group_size
            if overrides.recall_group_size is None
            else overrides.recall_group_size,
            recall_group_topk=scale.recall_group_topk
            if overrides.recall_group_topk is None
            else overrides.recall_group_topk,
            recall_value_composition=scale.recall_value_composition
            if overrides.recall_value_composition is None
            else overrides.recall_value_composition,
            recall_formula=scale.recall_formula
            if overrides.recall_formula is None
            else overrides.recall_formula,
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
        f"exclude = {_toml_value(insertion['exclude'])}",
        f"positions = {_toml_value(insertion['positions'])}",
        f"every = {insertion['every']}",
        f"freeze_base = {str(insertion['freeze_base']).lower()}",
        f"identity_gate = {str(insertion['identity_gate']).lower()}",
        f"zero_init_output = {str(insertion['zero_init_output']).lower()}",
        f'bridge_mode = "{insertion["bridge_mode"]}"',
        *(
            []
            if insertion["boundary_mask_key"] is None
            else [f'boundary_mask_key = "{insertion["boundary_mask_key"]}"']
        ),
        f"require_runtime_context = {str(insertion['require_runtime_context']).lower()}",
        f"max_adapters = {_toml_value(insertion['max_adapters'])}",
        f"max_extra_params = {_toml_value(insertion['max_extra_params'])}",
        "",
    ]
    if insertion["scale_pattern"]:
        lines.extend(["[insertion.scale_pattern]"])
        lines.extend(
            f"{_toml_string(pattern)} = {_toml_string(scale)}"
            for pattern, scale in insertion["scale_pattern"].items()
        )
        lines.append("")
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


def _as_scale_pattern(value: Any) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    if not isinstance(value, dict):
        raise ValueError("ARTI fit config insertion.scale_pattern must be a mapping")
    return tuple((str(pattern), str(scale)) for pattern, scale in value.items())


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)


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
