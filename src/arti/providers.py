"""Provider boundary for pretrained model ecosystems.

Providers load an external object and expose the ``torch.nn.Module`` components
that ARTI can scan and adapt. Heavy optional dependencies stay behind lazy
imports so the core package remains importable without the integration extras.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from importlib.util import find_spec
from typing import Any, Mapping

import torch.nn as nn


class ARTIProviderError(RuntimeError):
    """Actionable provider failure with a stable stage and remediation hint."""

    def __init__(self, provider: str, stage: str, message: str, *, hint: str | None = None) -> None:
        self.provider = provider
        self.stage = stage
        self.hint = hint
        detail = f"[{provider}:{stage}] {message}"
        if hint:
            detail += f" Hint: {hint}"
        super().__init__(detail)


@dataclass(frozen=True)
class LoadedPretrained:
    """Object loaded by a provider plus optional ecosystem assets."""

    root: Any
    tokenizer: Any | None = None
    processor: Any | None = None


@dataclass(frozen=True)
class ProviderInspection:
    """Normalized provider view consumed by the pretrained workflow."""

    provider: str
    components: Mapping[str, nn.Module]
    native_capabilities: tuple[str, ...]
    metadata: Mapping[str, Any]


class PretrainedProvider:
    """Base class for a pretrained ecosystem provider."""

    name = "base"
    dependency: str | None = None
    extra: str | None = None

    @property
    def available(self) -> bool:
        return self.dependency is None or find_spec(self.dependency) is not None

    def require(self) -> None:
        if not self.available:
            extra = self.extra or self.name
            raise ARTIProviderError(
                self.name,
                "dependency",
                f"optional dependency {self.dependency!r} is not installed",
                hint=f"run `uv sync --extra {extra}`",
            )

    def matches(self, value: Any) -> bool:
        return False

    def load(self, source: str, *, task: str | None, revision: str | None, kwargs: Mapping[str, Any]) -> LoadedPretrained:
        raise NotImplementedError

    def inspect(self, root: Any, *, components: tuple[str, ...] | None = None) -> ProviderInspection:
        if not isinstance(root, nn.Module):
            raise ARTIProviderError(self.name, "inspect", "provider root is not a torch.nn.Module")
        return ProviderInspection(
            provider=self.name,
            components={"model": root},
            native_capabilities=_native_capabilities(root),
            metadata=_model_metadata(root),
        )


class TorchProvider(PretrainedProvider):
    name = "torch"

    def matches(self, value: Any) -> bool:
        return isinstance(value, nn.Module)

    def load(self, source: str, *, task: str | None, revision: str | None, kwargs: Mapping[str, Any]) -> LoadedPretrained:
        raise ARTIProviderError(
            self.name,
            "load",
            "the torch provider accepts an instantiated nn.Module, not a model id",
            hint="pass the model object to arti.pretrained(...) or select a model-family provider",
        )


class TransformersProvider(PretrainedProvider):
    name = "transformers"
    dependency = "transformers"
    extra = "qwen"

    def matches(self, value: Any) -> bool:
        module = value.__class__.__module__
        return module.startswith("transformers.") or (isinstance(value, nn.Module) and hasattr(value, "config") and hasattr(value, "generate"))

    def load(self, source: str, *, task: str | None, revision: str | None, kwargs: Mapping[str, Any]) -> LoadedPretrained:
        self.require()
        _reject_remote_code(kwargs)
        transformers = import_module("transformers")
        task_name = task or "causal-lm"
        class_names = {
            "causal-lm": "AutoModelForCausalLM",
            "seq2seq-lm": "AutoModelForSeq2SeqLM",
            "masked-lm": "AutoModelForMaskedLM",
            "sequence-classification": "AutoModelForSequenceClassification",
            "image-classification": "AutoModelForImageClassification",
            "base": "AutoModel",
        }
        if task_name not in class_names:
            raise ARTIProviderError(
                self.name,
                "load",
                f"unsupported task {task_name!r}",
                hint=f"choose one of {sorted(class_names)}",
            )
        loader = getattr(transformers, class_names[task_name])
        load_kwargs = dict(kwargs)
        if revision is not None:
            load_kwargs["revision"] = revision
        try:
            model = loader.from_pretrained(source, **load_kwargs)
        except Exception as exc:
            raise ARTIProviderError(self.name, "load", f"failed to load {source!r}: {exc}") from exc
        tokenizer = None
        if task_name != "image-classification" and hasattr(transformers, "AutoTokenizer"):
            tokenizer_kwargs = {
                key: load_kwargs[key]
                for key in ("revision", "trust_remote_code", "token", "local_files_only", "cache_dir")
                if key in load_kwargs
            }
            try:
                tokenizer = transformers.AutoTokenizer.from_pretrained(source, **tokenizer_kwargs)
            except Exception:
                tokenizer = None
        return LoadedPretrained(root=model, tokenizer=tokenizer)


class PEFTProvider(PretrainedProvider):
    name = "peft"
    dependency = "peft"
    extra = "peft"

    def matches(self, value: Any) -> bool:
        return value.__class__.__module__.startswith("peft.") or hasattr(value, "peft_config")

    def load(self, source: str, *, task: str | None, revision: str | None, kwargs: Mapping[str, Any]) -> LoadedPretrained:
        self.require()
        _reject_remote_code(kwargs)
        peft = import_module("peft")
        task_name = task or "causal-lm"
        loader_name = "AutoPeftModelForCausalLM" if task_name == "causal-lm" else "AutoPeftModel"
        loader = getattr(peft, loader_name, None)
        if loader is None:
            raise ARTIProviderError(self.name, "load", f"installed PEFT does not expose {loader_name}")
        load_kwargs = dict(kwargs)
        if revision is not None:
            load_kwargs["revision"] = revision
        try:
            return LoadedPretrained(root=loader.from_pretrained(source, **load_kwargs))
        except Exception as exc:
            raise ARTIProviderError(self.name, "load", f"failed to load {source!r}: {exc}") from exc

    def inspect(self, root: Any, *, components: tuple[str, ...] | None = None) -> ProviderInspection:
        inspection = super().inspect(root, components=components)
        metadata = dict(inspection.metadata)
        peft_config = getattr(root, "peft_config", {})
        metadata["peft_adapters"] = sorted(str(name) for name in peft_config)
        return ProviderInspection(self.name, inspection.components, inspection.native_capabilities, metadata)


class DiffusersProvider(PretrainedProvider):
    name = "diffusers"
    dependency = "diffusers"
    extra = "sd"

    def matches(self, value: Any) -> bool:
        return value.__class__.__module__.startswith("diffusers.") or hasattr(value, "components") and not isinstance(value, nn.Module)

    def load(self, source: str, *, task: str | None, revision: str | None, kwargs: Mapping[str, Any]) -> LoadedPretrained:
        self.require()
        _reject_remote_code(kwargs)
        diffusers = import_module("diffusers")
        load_kwargs = dict(kwargs)
        if revision is not None:
            load_kwargs["revision"] = revision
        try:
            pipeline = diffusers.DiffusionPipeline.from_pretrained(source, **load_kwargs)
        except Exception as exc:
            raise ARTIProviderError(self.name, "load", f"failed to load {source!r}: {exc}") from exc
        return LoadedPretrained(root=pipeline)

    def inspect(self, root: Any, *, components: tuple[str, ...] | None = None) -> ProviderInspection:
        available = {
            str(name): value
            for name, value in dict(getattr(root, "components", {})).items()
            if isinstance(value, nn.Module)
        }
        preferred = components or tuple(name for name in ("transformer", "unet", "text_encoder", "text_encoder_2", "vae") if name in available)
        selected = {name: available[name] for name in preferred if name in available}
        if not selected:
            raise ARTIProviderError(
                self.name,
                "inspect",
                "pipeline exposes no selected torch components",
                hint=f"available components: {sorted(available)}",
            )
        return ProviderInspection(
            provider=self.name,
            components=selected,
            native_capabilities=_native_capabilities(root) + ("pipeline-components",),
            metadata={
                "pipeline_class": _class_path(root),
                "available_components": sorted(available),
                "model_id": dict(getattr(root, "config", {})).get("_name_or_path"),
                "resolved_revision": dict(getattr(root, "config", {})).get("_commit_hash"),
            },
        )


_PROVIDERS: dict[str, PretrainedProvider] = {
    provider.name: provider
    for provider in (TorchProvider(), TransformersProvider(), PEFTProvider(), DiffusersProvider())
}


def register_provider(provider: PretrainedProvider, *, replace: bool = False) -> None:
    """Register a custom provider without importing it from ARTI core."""

    if not provider.name or provider.name == "auto":
        raise ValueError("provider.name must be a non-empty name other than 'auto'")
    if provider.name in _PROVIDERS and not replace:
        raise ValueError(f"ARTI provider {provider.name!r} is already registered")
    _PROVIDERS[provider.name] = provider


def get_provider(name: str) -> PretrainedProvider:
    if name not in _PROVIDERS:
        raise ValueError(f"unknown ARTI provider {name!r}; expected one of {sorted(_PROVIDERS)}")
    return _PROVIDERS[name]


def resolve_provider(value: Any, name: str = "auto") -> PretrainedProvider:
    if name != "auto":
        return get_provider(name)
    if isinstance(value, str):
        return get_provider("transformers")
    for candidate in ("peft", "diffusers", "transformers", "torch"):
        provider = get_provider(candidate)
        if provider.matches(value):
            return provider
    raise ValueError("could not infer an ARTI provider; pass provider= explicitly")


def provider_report() -> tuple[dict[str, Any], ...]:
    return tuple(
        {
            "name": provider.name,
            "dependency": provider.dependency,
            "extra": provider.extra,
            "available": provider.available,
        }
        for provider in _PROVIDERS.values()
    )


def _native_capabilities(value: Any) -> tuple[str, ...]:
    names = []
    for name in ("forward", "generate", "save_pretrained", "push_to_hub", "enable_adapters", "disable_adapters", "__call__"):
        if callable(getattr(value, name, None)):
            names.append(name)
    if hasattr(value, "past_key_values") or hasattr(getattr(value, "config", None), "use_cache"):
        names.append("kv-cache")
    return tuple(names)


def _model_metadata(value: Any) -> dict[str, Any]:
    config = getattr(value, "config", None)
    return {
        "class": _class_path(value),
        "config_class": None if config is None else _class_path(config),
        "model_id": None if config is None else getattr(config, "_name_or_path", None),
        "resolved_revision": None if config is None else getattr(config, "_commit_hash", None),
    }


def _reject_remote_code(kwargs: Mapping[str, Any]) -> None:
    if kwargs.get("trust_remote_code") is True:
        raise ARTIProviderError(
            "pretrained",
            "security",
            "declarative loading does not allow trust_remote_code=True",
            hint="review and instantiate the trusted model yourself, then pass the nn.Module to ARTI",
        )


def _class_path(value: Any) -> str:
    return f"{value.__class__.__module__}.{value.__class__.__qualname__}"


__all__ = [
    "ARTIProviderError",
    "DiffusersProvider",
    "LoadedPretrained",
    "PEFTProvider",
    "PretrainedProvider",
    "ProviderInspection",
    "TorchProvider",
    "TransformersProvider",
    "get_provider",
    "provider_report",
    "register_provider",
    "resolve_provider",
]
