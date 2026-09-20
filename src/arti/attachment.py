"""High-level, reversible ARTILayer attachment for existing PyTorch models."""

from __future__ import annotations

import fnmatch
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn

from .arti_layer import ARTILayer
from .attachment_config import (
    ARTIAttachConfig,
    ARTIAttachTrainingConfig,
    attach_config_from_dict,
    load_attach_config,
    validate_attach_lock,
    write_attach_lock,
)
from .attachment_layer import (
    AttachedARTILayer,
    AttachedARTILayerConfig,
    AttachedARTILayerSpec,
    AttachedARTIModel,
    LayerFactory,
    _make_layer,
    infer_module_dim,
    runtime_path_dims,
)
from .fit.insertion import get_parent_module, set_child_module
from .component_registry import canonical_contract_reference
from .serialization import ARTILoadResult, ARTISaveResult, load as load_arti, save as save_arti


@dataclass(frozen=True)
class ARTILayerInfo:
    """One compatible insertion point discovered in a model."""

    path: str
    dim: int | None
    module_type: str


@dataclass(frozen=True)
class ARTIAttachmentSummary:
    """Static resource description for a proposed or active attachment."""

    layers: tuple[ARTILayerInfo, ...]
    trainable_parameters: int
    backbone_parameters: int
    parameter_fraction: float
    multiply_adds_per_token: int
    estimated_parameter_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "layers": [asdict(layer) for layer in self.layers],
            "trainable_parameters": self.trainable_parameters,
            "backbone_parameters": self.backbone_parameters,
            "parameter_fraction": self.parameter_fraction,
            "multiply_adds_per_token": self.multiply_adds_per_token,
            "estimated_parameter_bytes": self.estimated_parameter_bytes,
        }

    def __str__(self) -> str:
        paths = ", ".join(layer.path for layer in self.layers)
        return (
            f"ARTILayer: {len(self.layers)} layer(s) [{paths}] | "
            f"{self.trainable_parameters:,} trainable parameters "
            f"({self.parameter_fraction:.3%} of backbone)"
        )


class _LayerBundle(nn.Module):
    def __init__(self, layers: Iterable[ARTILayer]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(tuple(layers))


class ARTIAttachment:
    """Control surface installed as ``model.arti`` by :meth:`ARTI.attach`."""

    def __init__(
        self,
        model: nn.Module,
        attached_layers: AttachedARTIModel,
        config: AttachedARTILayerConfig,
        prior_trainability: Mapping[str, bool],
        declaration: ARTIAttachConfig | None = None,
    ) -> None:
        self._model = model
        self._layers = attached_layers
        self.config = config
        self.declaration = declaration
        self._prior_trainability = dict(prior_trainability)
        self._attached = True

    @property
    def attached(self) -> bool:
        return self._attached

    @property
    def paths(self) -> tuple[str, ...]:
        return self.config.paths

    @property
    def layers(self) -> Mapping[str, ARTILayer]:
        self._require_attached()
        return {path: wrapper.layer for path, wrapper in self._layers.wrappers.items()}

    def summary(self) -> ARTIAttachmentSummary:
        self._require_attached()
        return _summary(self._model, self.config)

    def parameters(self) -> Iterable[nn.Parameter]:
        """Iterate trainable parameters owned by attached ARTILayer graphs."""

        self._require_attached()
        return self._layers.arti_parameters(trainable_only=True)

    def set_enabled(
        self,
        enabled: bool = True,
        *,
        paths: Iterable[str] | None = None,
    ) -> None:
        self._require_attached()
        self._layers.set_enabled(enabled, paths=paths)

    def enable(self, *, paths: Iterable[str] | None = None) -> None:
        self.set_enabled(True, paths=paths)

    def disable(self, *, paths: Iterable[str] | None = None) -> None:
        self.set_enabled(False, paths=paths)

    def set_capture(self, enabled: bool = True) -> None:
        """Enable or disable typed Federal runtime diagnostics at every attached layer."""

        self._require_attached()
        for wrapper in self._layers.wrappers.values():
            wrapper.capture = bool(enabled)
            if not enabled:
                wrapper.clear_trace()

    def diagnostics(self):
        self._require_attached()
        return self._layers.diagnostics()

    def doctor(self):
        from .attachment_hub import attachment_doctor

        self._require_attached()
        return attachment_doctor(self)

    def save_pretrained(
        self,
        directory: str | Path,
        *,
        base_model: str | Path | None = None,
        revision: str | None = None,
        training_session: Any | None = None,
    ):
        from .attachment_hub import save_attachment_pretrained

        self._require_attached()
        return save_attachment_pretrained(
            self,
            directory,
            base_model=base_model,
            revision=revision,
            training_session=training_session,
        )

    def save(
        self,
        path: str | Path,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        training_state: Any | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> ARTISaveResult:
        """Save attached ARTILayer state without host model parameters."""

        self._require_attached()
        target = Path(path)
        if not target.name.endswith(".arti.st"):
            raise ValueError("attachment artifacts must end in '.arti.st'")
        extra = dict(metadata or {})
        if "unified_attachment" in extra:
            raise ValueError("attachment metadata cannot override unified_attachment")
        artifact_metadata = {
            "unified_attachment": {
                "version": 3,
                "config": _config_payload(self.config),
                "declaration": None
                if self.declaration is None
                else self.declaration.to_dict(include_source=True),
                "host_structure": _host_structure_fingerprint(self._model),
                "execution_surface": {
                    "kind": "federal-bank-attachment",
                    "layers": {
                        path: layer.runtime_provenance() for path, layer in self.layers.items()
                    },
                    "federal_compiler_ref": canonical_contract_reference(
                        "arti/federal-static-compiler@1"
                    ),
                    "compiled_artifact_is_separate": True,
                },
            },
            **extra,
        }
        return save_arti(
            self._bundle(),
            target,
            config=artifact_metadata,
            scope="all",
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=training_state,
        )

    def load(
        self,
        path: str | Path,
        *,
        map_location: str | torch.device | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        load_checkpoint: bool = False,
    ) -> ARTILoadResult:
        self._require_attached()
        inspected = load_arti(path, load_resources=False, load_checkpoint=False)
        metadata = _attachment_metadata(inspected.manifest)
        if _config_from_payload(metadata["config"]) != self.config:
            raise ValueError("ARTILayer artifact topology does not match this attachment")
        if metadata["host_structure"] != _host_structure_fingerprint(self._model):
            raise ValueError("ARTILayer artifact host structure does not match this model")
        return load_arti(
            path,
            model=self._bundle(),
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=map_location or _model_device(self._model),
            load_resources=False,
            load_checkpoint=load_checkpoint,
        )

    def trainer(
        self,
        *,
        engine: str | None = None,
        objective: str | Any | None = None,
        learning_rate: float | None = None,
        steps: int | None = None,
        gradient_accumulation_steps: int | None = None,
        mixed_precision: str | None = None,
        corruption_probability: float | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        resume_from_checkpoint: str | Path | bool | None = None,
    ):
        from .attachment_training import ARTITrainingSession

        base = (
            self.declaration.training
            if self.declaration is not None
            else ARTIAttachTrainingConfig()
        )
        config = ARTIAttachTrainingConfig(
            engine=engine or base.engine,
            objective=base.objective
            if callable(objective) or objective is None
            else str(objective),
            learning_rate=base.learning_rate if learning_rate is None else learning_rate,
            steps=base.steps if steps is None else steps,
            gradient_accumulation_steps=base.gradient_accumulation_steps
            if gradient_accumulation_steps is None
            else gradient_accumulation_steps,
            mixed_precision=base.mixed_precision if mixed_precision is None else mixed_precision,
            corruption_probability=base.corruption_probability
            if corruption_probability is None
            else corruption_probability,
        )
        return ARTITrainingSession(
            self,
            config=config,
            objective=objective,
            optimizer=optimizer,
            scheduler=scheduler,
            resume_from_checkpoint=resume_from_checkpoint,
        )

    def write_lock(self, path: str | Path) -> Path:
        declaration = self.declaration or ARTIAttachConfig(layer={"layers": list(self.paths)})
        return write_attach_lock(
            path,
            config=declaration,
            resolved_layer=_config_payload(self.config),
            host_structure=_host_structure_fingerprint(self._model),
        )

    def validate_lock(self, path: str | Path) -> dict[str, Any]:
        declaration = self.declaration or ARTIAttachConfig(layer={"layers": list(self.paths)})
        return validate_attach_lock(
            path,
            config=declaration,
            resolved_layer=_config_payload(self.config),
            host_structure=_host_structure_fingerprint(self._model),
        )

    def detach(self) -> nn.Module:
        self._require_attached()
        for path in reversed(self.paths):
            parent, leaf = get_parent_module(self._model, path)
            wrapper = _child(parent, leaf)
            if not isinstance(wrapper, AttachedARTILayer):
                raise RuntimeError(f"layer {path!r} no longer contains its ARTI attachment")
            set_child_module(parent, leaf, wrapper.base)
        for name, parameter in self._model.named_parameters():
            if name in self._prior_trainability:
                parameter.requires_grad_(self._prior_trainability[name])
        object.__delattr__(self._model, "arti")
        self._attached = False
        return self._model

    def _bundle(self) -> _LayerBundle:
        return _LayerBundle(self._layers.wrappers[path].layer for path in self.paths)

    def _require_attached(self) -> None:
        if not self._attached:
            raise RuntimeError("ARTI attachment has been detached")


class ARTI:
    """Attach federated-program-backed ARTILayer instances to an existing model."""

    @staticmethod
    def discover(
        model: nn.Module,
        layers: str | Iterable[str] | None = None,
    ) -> tuple[ARTILayerInfo, ...]:
        return discover_layers(model, layers)

    @staticmethod
    def preview(
        model: nn.Module,
        layer: ARTILayer | LayerFactory | None = None,
        *,
        layers: str | Iterable[str] | None = None,
        freeze_backbone: bool = True,
        sample_batch: Any | None = None,
    ) -> ARTIAttachmentSummary:
        resolved = _resolve_config(
            model,
            layers=layers,
            freeze_backbone=freeze_backbone,
            sample_batch=sample_batch,
        )
        return _summary(model, resolved, layer=layer)

    @staticmethod
    def attach(
        model: nn.Module,
        layer: ARTILayer | LayerFactory | None = None,
        *,
        layers: str | Iterable[str] | None = None,
        freeze_backbone: bool = True,
        config: str | Path | ARTIAttachConfig | None = None,
        sample_batch: Any | None = None,
    ) -> nn.Module:
        """Attach ARTILayer@3 in place and return the original model object."""

        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if hasattr(model, "arti"):
            raise ValueError("model already has an ARTI attachment")
        declaration = None
        batch_axis = None
        feature_axis = None
        if config is not None:
            if layers is not None or freeze_backbone is not True:
                raise ValueError("config owns layers and freeze_backbone")
            declaration = load_attach_config(config) if isinstance(config, (str, Path)) else config
            options = dict(declaration.layer)
            layers = options.pop("layers", None)
            freeze_backbone = bool(options.pop("freeze_backbone", True))
            batch_axis = options.pop("batch_axis", None)
            feature_axis = options.pop("feature_axis", None)
            if options:
                raise ValueError(f"unknown ARTILayer attachment options: {sorted(options)}")
        resolved = _resolve_config(
            model,
            layers=layers,
            freeze_backbone=freeze_backbone,
            sample_batch=sample_batch,
            batch_axis=batch_axis,
            feature_axis=feature_axis,
        )
        prior = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
        try:
            attached_layers = AttachedARTIModel.from_config(
                model,
                resolved,
                layer=layer,
                sample_batch=sample_batch,
            )
        except Exception:
            _rollback_attachment(model, resolved.paths, prior)
            raise
        controller = ARTIAttachment(
            model,
            attached_layers,
            attached_layers.config,
            prior,
            declaration,
        )
        object.__setattr__(model, "arti", controller)
        return model

    @staticmethod
    def load(
        model: nn.Module,
        path: str | Path,
        *,
        layer: ARTILayer | LayerFactory | None = None,
        sample_batch: Any | None = None,
        map_location: str | torch.device | None = None,
    ) -> nn.Module:
        """Reconstruct one attachment using a matching ARTILayer graph."""

        inspected = load_arti(path, load_resources=False, load_checkpoint=False)
        metadata = _attachment_metadata(inspected.manifest)
        config = _config_from_payload(metadata["config"])
        ARTI.attach(
            model,
            layer,
            layers=config.paths,
            freeze_backbone=config.freeze_backbone,
            sample_batch=sample_batch,
        )
        declaration = metadata.get("declaration")
        if isinstance(declaration, Mapping):
            model.arti.declaration = attach_config_from_dict(declaration)
        try:
            model.arti.load(path, map_location=map_location)
        except Exception:
            model.arti.detach()
            raise
        return model

    @staticmethod
    def from_pretrained(
        directory: str | Path,
        *,
        model: nn.Module | None = None,
        layer: ARTILayer | LayerFactory | None = None,
        map_location: str | torch.device | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
    ) -> nn.Module:
        from .attachment_hub import load_attachment_pretrained

        return load_attachment_pretrained(
            directory,
            model=model,
            layer=layer,
            map_location=map_location,
            model_kwargs=model_kwargs,
        )


def discover_layers(
    model: nn.Module,
    layers: str | Iterable[str] | None = None,
) -> tuple[ARTILayerInfo, ...]:
    """Discover block-level insertion points without running the model."""

    named = tuple((name, module) for name, module in model.named_modules() if name)
    if layers is not None:
        patterns = (layers,) if isinstance(layers, str) else tuple(layers)
        selected = [
            (name, module)
            for name, module in named
            if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)
        ]
        missing = [
            pattern
            for pattern in patterns
            if not any(fnmatch.fnmatchcase(name, pattern) for name, _ in named)
        ]
        if missing:
            raise ValueError(f"layer patterns matched no modules: {missing}")
    else:
        selected = [(name, module) for name, module in named if _is_transformer_block(name, module)]
        if not selected and isinstance(model, nn.Sequential):
            selected = [
                (name, module)
                for name, module in named
                if name.count(".") == 0
                and infer_module_dim(module) is not None
                and _is_shape_preserving(module)
            ]
    if not selected:
        raise ValueError("no compatible layers discovered; pass layers='path.or.glob'")
    paths: list[tuple[str, nn.Module]] = []
    for name, module in selected:
        if not any(name.startswith(parent + ".") for parent, _ in paths):
            paths.append((name, module))
    return tuple(
        ARTILayerInfo(name, _infer_attachment_dim(model, module), type(module).__name__)
        for name, module in paths
    )


def _resolve_config(
    model: nn.Module,
    *,
    layers: str | Iterable[str] | None,
    freeze_backbone: bool,
    sample_batch: Any | None,
    batch_axis: int | Mapping[str, int] | None = None,
    feature_axis: int | Mapping[str, int] | None = None,
) -> AttachedARTILayerConfig:
    discovered = discover_layers(model, layers)
    dims = {item.path: item.dim for item in discovered if item.dim is not None}
    unresolved = tuple(item.path for item in discovered if item.dim is None)
    if unresolved and sample_batch is not None:
        dims.update(
            runtime_path_dims(
                model,
                unresolved,
                sample_batch,
                batch_axis=batch_axis,
                feature_axis=feature_axis,
            )
        )
    unknown = tuple(path for path in unresolved if path not in dims)
    if unknown:
        raise ValueError(f"cannot infer hidden dimensions for {unknown}; pass sample_batch")
    return AttachedARTILayerConfig(
        tuple(
            AttachedARTILayerSpec(
                item.path,
                dim=dims[item.path],
                batch_axis=None if batch_axis is None else int(_path_option(batch_axis, item.path)),
                feature_axis=None
                if feature_axis is None
                else int(_path_option(feature_axis, item.path)),
            )
            for item in discovered
        ),
        freeze_backbone=bool(freeze_backbone),
    )


def _summary(
    model: nn.Module,
    config: AttachedARTILayerConfig,
    *,
    layer: ARTILayer | LayerFactory | None = None,
) -> ARTIAttachmentSummary:
    wrappers = {
        path: module
        for path, module in model.named_modules()
        if isinstance(module, AttachedARTILayer)
    }
    infos: list[ARTILayerInfo] = []
    trainable = 0
    parameter_bytes = 0
    layer_parameter_ids: set[int] = set()
    for spec in config.layers:
        module = dict(model.named_modules()).get(spec.path)
        wrapper = wrappers.get(spec.path)
        if wrapper is not None:
            module = wrapper.base
            instance = wrapper.layer
        else:
            instance = _make_layer(layer, spec)
        infos.append(ARTILayerInfo(spec.path, spec.dim, type(module).__name__))
        for parameter in instance.parameters():
            layer_parameter_ids.add(id(parameter))
            if parameter.requires_grad:
                trainable += parameter.numel()
                parameter_bytes += parameter.numel() * parameter.element_size()
    backbone = sum(
        parameter.numel()
        for parameter in model.parameters()
        if id(parameter) not in layer_parameter_ids
    )
    return ARTIAttachmentSummary(
        layers=tuple(infos),
        trainable_parameters=trainable,
        backbone_parameters=backbone,
        parameter_fraction=trainable / max(backbone, 1),
        multiply_adds_per_token=0,
        estimated_parameter_bytes=parameter_bytes,
    )


def _config_payload(config: AttachedARTILayerConfig) -> dict[str, Any]:
    return {
        "freeze_backbone": config.freeze_backbone,
        "layers": [asdict(spec) for spec in config.layers],
    }


def _config_from_payload(payload: Mapping[str, Any]) -> AttachedARTILayerConfig:
    return AttachedARTILayerConfig(
        tuple(AttachedARTILayerSpec(**item) for item in payload["layers"]),
        freeze_backbone=bool(payload.get("freeze_backbone", True)),
    )


def _attachment_metadata(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = manifest.get("architecture", {}).get("config", {}).get("unified_attachment")
    if not isinstance(metadata, Mapping) or metadata.get("version") != 3:
        raise ValueError("artifact is not an ARTILayer@3 attachment")
    return metadata


def _rollback_attachment(
    model: nn.Module,
    paths: Iterable[str],
    prior: Mapping[str, bool],
) -> None:
    for path in paths:
        try:
            parent, leaf = get_parent_module(model, path)
            current = _child(parent, leaf)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            continue
        if isinstance(current, AttachedARTILayer):
            set_child_module(parent, leaf, current.base)
    for name, parameter in model.named_parameters():
        if name in prior:
            parameter.requires_grad_(prior[name])


def _model_device(model: nn.Module) -> torch.device:
    return next(model.parameters(), torch.empty(0)).device


def _host_structure_fingerprint(model: nn.Module) -> str:
    payload = {
        "class": f"{type(model).__module__}.{type(model).__qualname__}",
        "modules": [
            {"path": name, "class": f"{type(module).__module__}.{type(module).__qualname__}"}
            for name, module in model.named_modules()
        ],
        "state": [
            {"name": name, "shape": list(tensor.shape), "dtype": str(tensor.dtype)}
            for name, tensor in model.state_dict().items()
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _path_option(value: Any, path: str) -> Any:
    if isinstance(value, Mapping):
        if path not in value:
            raise ValueError(f"option mapping is missing layer {path!r}")
        return value[path]
    return value


def _is_shape_preserving(module: nn.Module) -> bool:
    return not isinstance(module, nn.Linear) or module.in_features == module.out_features


def _is_transformer_block(name: str, module: nn.Module) -> bool:
    cls = type(module).__name__.lower()
    block_class = any(
        token in cls for token in ("decoderlayer", "encoderlayer", "transformerblock")
    ) or cls.endswith("block")
    indexed_layer = ".layers." in f".{name}." and name.rsplit(".", 1)[-1].isdigit()
    return block_class or indexed_layer


def _infer_attachment_dim(model: nn.Module, module: nn.Module) -> int | None:
    direct = infer_module_dim(module)
    if direct is not None:
        return direct
    config = getattr(model, "config", None)
    for attribute in ("hidden_size", "d_model", "n_embd"):
        value = (
            config.get(attribute)
            if isinstance(config, Mapping)
            else getattr(config, attribute, None)
        )
        if isinstance(value, int) and value > 0:
            return value
    candidates = []
    for child in module.modules():
        if isinstance(child, nn.Linear) and child.in_features == child.out_features:
            candidates.append(child.out_features)
        else:
            value = infer_module_dim(child)
            if value is not None:
                candidates.append(value)
    return max(set(candidates), key=candidates.count) if candidates else None


def _child(parent: nn.Module, leaf: str) -> nn.Module:
    if leaf.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        return parent[int(leaf)]
    return getattr(parent, leaf)


__all__ = [
    "ARTI",
    "ARTIAttachment",
    "ARTIAttachmentSummary",
    "ARTILayerInfo",
    "AttachedARTILayerConfig",
    "AttachedARTILayerSpec",
    "discover_layers",
]
