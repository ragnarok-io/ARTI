"""Configuration objects for ARTI layers."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping


STATE_RECALL_CONTENT_FACTORS = 2
STATE_RECALL_MODULATION_FACTORS = 13
STATE_RECALL_COMPOSITION_FACTOR = (
    STATE_RECALL_CONTENT_FACTORS + STATE_RECALL_MODULATION_FACTORS + 2
)


@dataclass(frozen=True)
class ARTIConfig:
    """Configuration for dynamic latent tensor layers."""

    input_dim: int
    hidden_dim: int | None = None
    coord_dim: int = 0
    operator_count: int = 4
    interface_slots: int = 8
    recall_slots: int = 4
    recall_steps: int = 1
    recall_min_steps: int = 1
    recall_tolerance: float | None = None
    recall_activation: str = "half"
    recall_recognition_mode: str = "none"
    recall_recognition_threshold: float = 0.5
    recall_recognition_temperature: float = 0.1
    recall_routing: str = "dense"
    recall_key_dim: int = 32
    recall_query_mode: str = "fixed"
    recall_query_seed: int = 0
    recall_group_size: int = 16
    recall_group_topk: int = 2
    recall_value_composition: str = "single"
    recall_route_exploration: float = 0.0
    dropout: float = 0.0
    use_layer_norm: bool = True
    use_phase_mixer: bool = True
    use_virtual_interface: bool = True
    use_pairwise_context: bool = True
    use_recall: bool = True
    use_virtual_recall: bool = True
    require_coord: bool = False
    require_visibility: bool = False
    return_input_shape: bool = True
    coord_frame_mode: str = "none"
    fallback_context: str = "none"
    fallback_slots: int = 32

    def __post_init__(self) -> None:
        hidden_dim = self.input_dim if self.hidden_dim is None else self.hidden_dim
        object.__setattr__(self, "hidden_dim", hidden_dim)

        if self.input_dim <= 0:
            raise ValueError("input_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        if self.coord_dim < 0:
            raise ValueError("coord_dim must be non-negative")
        if self.operator_count < 0 or (self.use_phase_mixer and self.operator_count == 0):
            raise ValueError("operator_count must be positive when phase mixing is enabled")
        if self.interface_slots < 0 or (self.use_virtual_interface and self.interface_slots == 0):
            raise ValueError(
                "interface_slots must be positive when the virtual interface is enabled"
            )
        if self.recall_slots < 0 or (
            self.use_recall and self.recall_steps > 0 and self.recall_slots == 0
        ):
            raise ValueError("recall_slots must be positive when recall is enabled")
        if self.recall_steps < 0:
            raise ValueError("recall_steps must be non-negative")
        if self.recall_min_steps < 0:
            raise ValueError("recall_min_steps must be non-negative")
        if self.recall_steps > 0 and not 1 <= self.recall_min_steps <= self.recall_steps:
            raise ValueError("recall_min_steps must be in [1, recall_steps] when recall is enabled")
        if self.recall_tolerance is not None and (
            not math.isfinite(self.recall_tolerance) or self.recall_tolerance < 0
        ):
            raise ValueError("recall_tolerance must be finite and non-negative")
        if self.recall_activation not in {"half", "none"}:
            raise ValueError("recall_activation must be 'half' or 'none'")
        if self.recall_recognition_mode not in {"explicit", "alignment", "none"}:
            raise ValueError("recall_recognition_mode must be 'explicit', 'alignment', or 'none'")
        if not -1.0 <= self.recall_recognition_threshold <= 1.0:
            raise ValueError("recall_recognition_threshold must be in [-1, 1]")
        if self.recall_recognition_temperature <= 0:
            raise ValueError("recall_recognition_temperature must be positive")
        if self.recall_routing not in {"dense", "grouped"}:
            raise ValueError("recall_routing must be 'dense' or 'grouped'")
        if self.recall_key_dim <= 0:
            raise ValueError("recall_key_dim must be positive")
        if self.recall_query_mode not in {"fixed", "legacy_learned"}:
            raise ValueError("recall_query_mode must be 'fixed' or 'legacy_learned'")
        if isinstance(self.recall_query_seed, bool) or not isinstance(
            self.recall_query_seed, int
        ):
            raise TypeError("recall_query_seed must be an integer")
        if not 0 <= self.recall_query_seed < 2**63:
            raise ValueError("recall_query_seed must be in [0, 2**63)")
        if self.recall_group_size <= 0:
            raise ValueError("recall_group_size must be positive")
        if self.recall_group_topk <= 0:
            raise ValueError("recall_group_topk must be positive")
        if self.recall_value_composition not in {"single", "product", "state"}:
            raise ValueError("recall_value_composition must be 'single', 'product', or 'state'")
        if not math.isfinite(self.recall_route_exploration) or self.recall_route_exploration < 0:
            raise ValueError("recall_route_exploration must be finite and non-negative")
        recall_enabled = self.use_recall and self.recall_steps > 0
        if recall_enabled:
            if self.recall_routing == "grouped":
                if self.recall_slots % self.recall_group_size:
                    raise ValueError("grouped recall_slots must be divisible by recall_group_size")
                groups = self.recall_slots // self.recall_group_size
                if self.recall_group_topk > groups:
                    raise ValueError(
                        "recall_group_topk must not exceed the number of recall groups"
                    )
                composition_factor = {
                    "single": 1,
                    "product": 2,
                    "state": STATE_RECALL_COMPOSITION_FACTOR,
                }[self.recall_value_composition]
                if composition_factor > 1:
                    if groups % composition_factor:
                        raise ValueError(
                            f"{self.recall_value_composition} Recall requires recall groups "
                            f"divisible by {composition_factor}"
                        )
                    if self.recall_group_topk > groups // composition_factor:
                        raise ValueError(
                            "recall_group_topk must fit within each composed Recall bank"
                        )
            else:
                composition_factor = {
                    "single": 1,
                    "product": 2,
                    "state": STATE_RECALL_COMPOSITION_FACTOR,
                }[self.recall_value_composition]
                if self.recall_slots % composition_factor:
                    raise ValueError(
                        f"{self.recall_value_composition} Recall requires recall_slots "
                        f"divisible by {composition_factor}"
                    )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.coord_frame_mode not in {"none", "paired_rotation", "operator_bank"}:
            raise ValueError(
                "coord_frame_mode must be 'none', 'paired_rotation', or 'operator_bank'"
            )
        if self.fallback_context not in {"none", "random_coord", "random_context"}:
            raise ValueError("fallback_context must be 'none', 'random_coord', or 'random_context'")
        if self.fallback_slots <= 0:
            raise ValueError("fallback_slots must be positive")
        if self.coord_frame_mode == "paired_rotation" and self.coord_dim < 2:
            raise ValueError("paired_rotation coord_frame_mode requires coord_dim >= 2")
        if self.coord_frame_mode == "paired_rotation" and self.input_dim % 2 != 0:
            raise ValueError("paired_rotation coord_frame_mode requires an even input_dim")
        if self.require_visibility and not (
            self.use_pairwise_context or self.use_virtual_interface
        ):
            raise ValueError("require_visibility needs pairwise context or the virtual interface")
        if self.require_coord and self.coord_dim == 0:
            raise ValueError("require_coord needs coord_dim > 0")

    def to_dict(self) -> dict[str, Any]:
        """Return the complete JSON-safe execution configuration."""

        return asdict(self)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ARTIConfig":
        """Restore a configuration while rejecting unknown keys."""

        known = {field.name for field in fields(cls)}
        unknown = sorted(set(payload) - known)
        if unknown:
            raise ValueError(f"unknown ARTIConfig fields: {unknown}")
        return cls(**dict(payload))

    def explain(self) -> dict[str, Any]:
        """Describe enabled mechanisms, required context, and allocated capacities."""

        recall_enabled = self.use_recall and self.recall_steps > 0
        mechanisms = {
            "phase": self.use_phase_mixer,
            "coordinate_inverse": self.coord_frame_mode != "none",
            "virtual_interface": self.use_virtual_interface,
            "pairwise_context": self.use_pairwise_context,
            "recall": recall_enabled,
            "adaptive_recall": recall_enabled and self.recall_tolerance is not None,
            "virtual_recall": self.use_virtual_recall,
            "half": recall_enabled and self.recall_activation == "half",
            "pulse": False,
            "carrier": False,
        }
        required_inputs = {
            "x": True,
            "coord": self.require_coord,
            "frame_operators": self.coord_frame_mode == "operator_bank",
            "observer_coord": False,
            "mask": False,
            "visibility": self.require_visibility,
            "recall": False,
        }
        accepted_inputs = {
            **required_inputs,
            "observer_coord": self.coord_frame_mode != "none",
            "mask": True,
            "visibility": self.use_pairwise_context or self.use_virtual_interface,
            "recall": recall_enabled,
        }
        capacities = {
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
            "coord_dim": self.coord_dim if self.coord_dim > 0 else 0,
            "operator_count": self.operator_count if self.use_phase_mixer else 0,
            "interface_slots": self.interface_slots if self.use_virtual_interface else 0,
            "recall_slots": self.recall_slots if recall_enabled else 0,
            "recall_steps": self.recall_steps if recall_enabled else 0,
            "recall_min_steps": self.recall_min_steps if recall_enabled else 0,
            "recall_tolerance": self.recall_tolerance if recall_enabled else None,
            "recall_routing": self.recall_routing if recall_enabled else None,
            "recall_key_dim": self.recall_key_dim if recall_enabled else 0,
            "recall_query_mode": self.recall_query_mode if recall_enabled else None,
            "recall_query_seed": self.recall_query_seed if recall_enabled else 0,
            "recall_group_size": self.recall_group_size
            if recall_enabled and self.recall_routing == "grouped"
            else 0,
            "recall_group_topk": self.recall_group_topk
            if recall_enabled and self.recall_routing == "grouped"
            else 0,
            "recall_value_composition": self.recall_value_composition if recall_enabled else None,
            "recall_route_exploration": self.recall_route_exploration
            if recall_enabled and self.recall_routing == "grouped"
            else 0.0,
        }
        return {
            "mechanisms": mechanisms,
            "required_inputs": required_inputs,
            "accepted_inputs": accepted_inputs,
            "capacities": capacities,
            "fallback_context": self.fallback_context,
            "synthetic_context": self.fallback_context != "none",
        }

    def diff(self, other: "ARTIConfig") -> dict[str, dict[str, Any]]:
        """Return field-level differences against another execution config."""

        if not isinstance(other, ARTIConfig):
            raise TypeError("other must be an ARTIConfig")
        left = self.to_dict()
        right = other.to_dict()
        return {
            key: {"self": left[key], "other": right[key]} for key in left if left[key] != right[key]
        }
