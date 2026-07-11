"""Optional fit plugin registry.

Plugins are metadata-only unless their optional dependency is installed. This
keeps the core package lightweight while still giving ``arti.project`` a
Gradle-like extension point.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec


@dataclass(frozen=True)
class FitPlugin:
    name: str
    kind: str
    default_strategy: str
    optional_dependency: str | None
    capabilities: tuple[str, ...]

    @property
    def available(self) -> bool:
        return self.optional_dependency is None or find_spec(self.optional_dependency) is not None

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind,
            "default_strategy": self.default_strategy,
            "optional_dependency": self.optional_dependency,
            "available": self.available,
            "capabilities": list(self.capabilities),
        }


PLUGIN_REGISTRY = {
    "torch": FitPlugin(
        name="torch",
        kind="backend",
        default_strategy="all-linear",
        optional_dependency=None,
        capabilities=("scan-linear", "insert-wrapper", "freeze-base", "adapter-artifact"),
    ),
    "transformers": FitPlugin(
        name="transformers",
        kind="model-family",
        default_strategy="transformer",
        optional_dependency="transformers",
        capabilities=("attention-output-strategy", "mlp-output-strategy", "pretrained-module-names"),
    ),
    "timm": FitPlugin(
        name="timm",
        kind="model-family",
        default_strategy="vision-transformer",
        optional_dependency="timm",
        capabilities=("vision-transformer-names", "classifier-head-strategy", "patch-embedding-strategy"),
    ),
    "vision-cnn": FitPlugin(
        name="vision-cnn",
        kind="model-family",
        default_strategy="vision-cnn",
        optional_dependency=None,
        capabilities=("conv2d-output-strategy", "spatial-token-bridge", "resnet-style-names"),
    ),
    "recurrent": FitPlugin(
        name="recurrent",
        kind="model-family",
        default_strategy="recurrent",
        optional_dependency=None,
        capabilities=("rnn-output-strategy", "lstm-output-strategy", "gru-output-strategy", "tuple-output-adaptation"),
    ),
}


def get_plugin(name: str) -> FitPlugin:
    if name not in PLUGIN_REGISTRY:
        raise ValueError(f"unknown ARTI fit plugin {name!r}; expected one of {sorted(PLUGIN_REGISTRY)}")
    return PLUGIN_REGISTRY[name]


def plugin_report(names: tuple[str, ...] | list[str]) -> tuple[dict[str, object], ...]:
    return tuple(get_plugin(name).to_dict() for name in names)


def default_strategy_for(names: tuple[str, ...] | list[str]) -> str:
    for name in reversed(tuple(names)):
        plugin = get_plugin(name)
        if plugin.default_strategy != "all-linear":
            return plugin.default_strategy
    return "all-linear"
