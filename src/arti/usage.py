"""Progressive, transparent construction helpers for ARTI layers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields, replace
from typing import Any, Mapping

from .config import ARTIConfig
from .layers import ARTILatentTensorLayer


@dataclass(frozen=True)
class FeatureConfig:
    """Orthogonal feature selection compiled into :class:`ARTIConfig`."""

    phase: bool = False
    coord_dim: int = 0
    coord_frame_mode: str = "none"
    fallback_context: str = "none"
    operator_count: int = 4
    visibility: bool = False
    virtual_interface: bool = False
    interface_slots: int = 8
    pairwise_context: bool = False
    recall: bool = False
    recall_slots: int = 4
    recall_steps: int = 1
    recall_activation: str = "half"
    recall_recognition_mode: str = "explicit"
    virtual_recall: bool = False
    layer_norm: bool = True
    dropout: float = 0.0

    def __post_init__(self) -> None:
        if self.coord_dim < 0:
            raise ValueError("coord_dim must be non-negative")
        if self.coord_dim == 0 and self.coord_frame_mode != "none":
            raise ValueError("coordinate inversion requires coord_dim > 0")
        if self.coord_dim > 0 and not self.phase:
            raise ValueError("coord_dim > 0 requires phase=True")
        if not self.phase and self.coord_frame_mode != "none":
            raise ValueError("coordinate inversion requires phase=True")
        if self.fallback_context != "none" and self.coord_dim == 0:
            raise ValueError("fallback context requires coord_dim > 0")
        if self.recall_steps < 0:
            raise ValueError("recall_steps must be non-negative")
        if self.visibility and not (self.pairwise_context or self.virtual_interface):
            raise ValueError("visibility=True requires pairwise_context or virtual_interface")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureConfig":
        known = {field.name for field in fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"unknown FeatureConfig fields: {unknown}")
        return cls(**dict(payload))

    def with_overrides(self, **overrides: Any) -> "FeatureConfig":
        return replace(self, **{key: value for key, value in overrides.items() if value is not None})

    def compile(self, *, input_dim: int, hidden_dim: int | None = None) -> ARTIConfig:
        """Compile feature intent into the exact execution configuration."""

        recall_enabled = self.recall and self.recall_steps > 0
        return ARTIConfig(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            coord_dim=self.coord_dim,
            operator_count=self.operator_count if self.phase else 0,
            interface_slots=self.interface_slots if self.virtual_interface else 0,
            recall_slots=self.recall_slots if recall_enabled else 0,
            recall_steps=self.recall_steps if recall_enabled else 0,
            recall_activation=self.recall_activation,
            recall_recognition_mode=self.recall_recognition_mode,
            dropout=self.dropout,
            use_layer_norm=self.layer_norm,
            use_phase_mixer=self.phase,
            use_virtual_interface=self.virtual_interface,
            use_pairwise_context=self.pairwise_context,
            use_recall=recall_enabled,
            use_virtual_recall=self.virtual_recall,
            require_coord=self.phase and self.coord_dim > 0 and self.fallback_context == "none",
            require_visibility=self.visibility,
            coord_frame_mode=self.coord_frame_mode,
            fallback_context=self.fallback_context,
        )

    def explain(self, *, input_dim: int, hidden_dim: int | None = None) -> dict[str, Any]:
        return self.compile(input_dim=input_dim, hidden_dim=hidden_dim).explain()

    def diff(self, other: "FeatureConfig") -> dict[str, dict[str, Any]]:
        if not isinstance(other, FeatureConfig):
            raise TypeError("other must be a FeatureConfig")
        left = self.to_dict()
        right = other.to_dict()
        return {key: {"self": left[key], "other": right[key]} for key in left if left[key] != right[key]}


_PROFILES: dict[str, FeatureConfig] = {
    "minimal": FeatureConfig(),
    "recall": FeatureConfig(recall=True, virtual_recall=True),
    "multisource": FeatureConfig(
        phase=True,
        coord_dim=4,
        coord_frame_mode="operator_bank",
        visibility=True,
        virtual_interface=True,
        pairwise_context=False,
    ),
}


def features(
    *,
    phase: bool = False,
    coord_dim: int = 0,
    coord_frame_mode: str = "none",
    fallback_context: str = "none",
    operator_count: int = 4,
    visibility: bool = False,
    virtual_interface: bool = False,
    interface_slots: int = 8,
    pairwise_context: bool = False,
    recall: bool = False,
    recall_slots: int = 4,
    recall_steps: int = 1,
    recall_activation: str = "half",
    recall_recognition_mode: str = "explicit",
    virtual_recall: bool = False,
    layer_norm: bool = True,
    dropout: float = 0.0,
) -> FeatureConfig:
    """Build a JSON-safe, orthogonal ARTI feature declaration."""

    return FeatureConfig(
        phase=phase,
        coord_dim=coord_dim,
        coord_frame_mode=coord_frame_mode,
        fallback_context=fallback_context,
        operator_count=operator_count,
        visibility=visibility,
        virtual_interface=virtual_interface,
        interface_slots=interface_slots,
        pairwise_context=pairwise_context,
        recall=recall,
        recall_slots=recall_slots,
        recall_steps=recall_steps,
        recall_activation=recall_activation,
        recall_recognition_mode=recall_recognition_mode,
        virtual_recall=virtual_recall,
        layer_norm=layer_norm,
        dropout=dropout,
    )


def profile(name: str, **overrides: Any) -> FeatureConfig:
    """Resolve a transparent layer profile and apply explicit overrides."""

    if name not in _PROFILES:
        raise ValueError(f"unknown ARTI layer profile {name!r}; expected one of {sorted(_PROFILES)}")
    return _PROFILES[name].with_overrides(**overrides)


def layer_profiles() -> tuple[str, ...]:
    return tuple(_PROFILES)


class Layer(ARTILatentTensorLayer):
    """Progressive ARTI layer constructor with no mechanism enabled implicitly."""

    def __init__(
        self,
        dim: int,
        *,
        hidden_dim: int | None = None,
        features: FeatureConfig | Mapping[str, Any] | None = None,
        profile: str | None = None,
        **overrides: Any,
    ) -> None:
        if features is not None and profile is not None:
            raise ValueError("provide either features or profile, not both")
        if features is None:
            selected = globals()["profile"](profile or "minimal", **overrides)
        else:
            selected = FeatureConfig.from_dict(features) if isinstance(features, Mapping) else features
            if not isinstance(selected, FeatureConfig):
                raise TypeError("features must be a FeatureConfig or mapping")
            selected = selected.with_overrides(**overrides)
        self.features = selected
        super().__init__(selected.compile(input_dim=dim, hidden_dim=hidden_dim))


__all__ = ["FeatureConfig", "Layer", "features", "profile", "layer_profiles"]
