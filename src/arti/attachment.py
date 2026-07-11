"""High-level, reversible ARTI attachment for existing PyTorch models."""

from __future__ import annotations

import fnmatch
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
import torch.nn as nn

from .fit.insertion import get_parent_module, set_child_module
from .attachment_config import (
    ARTIAttachConfig,
    ARTIAttachTrainingConfig,
    attach_config_from_dict,
    load_attach_config,
    validate_attach_lock,
    write_attach_lock,
)
from .layered_recall import (
    LayerRecall,
    LayerRecallSpec,
    LayerRecallStack,
    LayerRecallWrapper,
    LayeredRecallConfig,
    LayeredRecallModel,
    _infer_module_dim,
    _runtime_path_dims,
)
from .recall_topology import LayeredRecallCandidate, estimate_layered_recall_cost
from .serialization import ARTILoadResult, ARTISaveResult, load as load_arti, save as save_arti


@dataclass(frozen=True)
class ARTILayerInfo:
    """One compatible insertion point discovered in a model."""

    path: str
    dim: int | None
    module_type: str


@dataclass(frozen=True)
class ARTIAttachmentSummary:
    """Static resource estimate for a proposed or active attachment."""

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
            f"ARTI Recall: {len(self.layers)} layer(s) [{paths}] | "
            f"{self.trainable_parameters:,} trainable parameters "
            f"({self.parameter_fraction:.3%} of backbone) | "
            f"{self.multiply_adds_per_token:,} multiply-adds/token"
        )


class _RecallBundle(nn.Module):
    def __init__(self, recalls: Iterable[nn.Module]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(tuple(recalls))


class ARTIAttachment:
    """Control surface installed as ``model.arti`` by :meth:`ARTI.attach`."""

    def __init__(
        self,
        model: nn.Module,
        layered: LayeredRecallModel,
        config: LayeredRecallConfig,
        prior_trainability: Mapping[str, bool],
        declaration: ARTIAttachConfig | None = None,
    ) -> None:
        self._model = model
        self._layered = layered
        self.config = config
        self.declaration = declaration
        self._prior_trainability = dict(prior_trainability)
        self._attached = True
        self._recognition_modes = {
            path: tuple(branch.recognition_mode for branch in _branches(wrapper.recall))
            for path, wrapper in layered.wrappers.items()
        }

    @property
    def attached(self) -> bool:
        return self._attached

    @property
    def paths(self) -> tuple[str, ...]:
        return self.config.paths

    def summary(self) -> ARTIAttachmentSummary:
        self._require_attached()
        return _summary(self._model, self.config)

    def parameters(self) -> Iterable[nn.Parameter]:
        self._require_attached()
        return self._layered.recall_parameters()

    def set_enabled(self, feature: str, enabled: bool = True, *, paths: Iterable[str] | None = None) -> None:
        """Independently toggle Recall, Half, or recognition."""

        self._require_attached()
        selected = self.paths if paths is None else self._layered._validate_paths(tuple(paths))
        if feature == "recall":
            self._layered.set_enabled(enabled, paths=selected)
            return
        if feature not in {"half", "recognition"}:
            raise ValueError("feature must be 'recall', 'half', or 'recognition'")
        for path in selected:
            for index, branch in enumerate(_branches(self._layered.wrappers[path].recall)):
                if feature == "half":
                    branch.use_half = bool(enabled)
                    branch.survival = branch.survival if enabled and branch.survival.__class__.__name__ == "Half" else (
                        _new_half(branch) if enabled else nn.Identity()
                    )
                else:
                    branch.recognition_mode = self._recognition_modes[path][index] if enabled else "none"

    def enable(self, feature: str = "recall", *, paths: Iterable[str] | None = None) -> None:
        self.set_enabled(feature, True, paths=paths)

    def disable(self, feature: str = "recall", *, paths: Iterable[str] | None = None) -> None:
        self.set_enabled(feature, False, paths=paths)

    def diagnostics(self) -> dict[str, dict[str, torch.Tensor]]:
        self._require_attached()
        return self._layered.diagnostics()

    def doctor(self):
        """Inspect placement, dtype, freezing, and wrapper compatibility."""

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
        """Write a Hub-compatible ARTI bundle without base model weights."""

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
    ) -> ARTISaveResult:
        """Save only attached Recall weights and topology as ``*.recall.arti.st``."""

        self._require_attached()
        target = Path(path)
        if not target.name.endswith(".recall.arti.st"):
            raise ValueError("attachment artifacts must end in '.recall.arti.st'")
        bundle = self._bundle()
        metadata = {
            "unified_attachment": {
                "version": 1,
                "config": _config_payload(self.config),
                "declaration": None if self.declaration is None else self.declaration.to_dict(include_source=True),
                "host_structure": _host_structure_fingerprint(self._model),
            }
        }
        return save_arti(
            bundle,
            target,
            config=metadata,
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
        """Restore a compatible attachment artifact into this model."""

        self._require_attached()
        inspected = load_arti(path, load_resources=False, load_checkpoint=False)
        metadata = _attachment_metadata(inspected.manifest)
        if _config_from_payload(metadata["config"]) != self.config:
            raise ValueError("Recall artifact topology does not match this attachment")
        if metadata["host_structure"] != _host_structure_fingerprint(self._model):
            raise ValueError("Recall artifact host structure does not match this model")
        device = map_location or _model_device(self._model)
        return load_arti(
            path,
            model=self._bundle(),
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
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
        """Create a Torch, Transformers, or Accelerate training session."""

        from .attachment_training import ARTITrainingSession

        base = self.declaration.training if self.declaration is not None else ARTIAttachTrainingConfig()
        config = ARTIAttachTrainingConfig(
            engine=engine or base.engine,
            objective=base.objective if callable(objective) or objective is None else str(objective),
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
        session = ARTITrainingSession(
            self,
            config=config,
            objective=objective,
            optimizer=optimizer,
            scheduler=scheduler,
            resume_from_checkpoint=resume_from_checkpoint,
        )
        return session

    def write_lock(self, path: str | Path) -> Path:
        if self.declaration is None:
            declaration = ARTIAttachConfig(recall=_config_payload(self.config))
        else:
            declaration = self.declaration
        return write_attach_lock(
            path,
            config=declaration,
            resolved_recall=_config_payload(self.config),
            host_structure=_host_structure_fingerprint(self._model),
        )

    def validate_lock(self, path: str | Path) -> dict[str, Any]:
        if self.declaration is None:
            declaration = ARTIAttachConfig(recall=_config_payload(self.config))
        else:
            declaration = self.declaration
        return validate_attach_lock(
            path,
            config=declaration,
            resolved_recall=_config_payload(self.config),
            host_structure=_host_structure_fingerprint(self._model),
        )

    def detach(self) -> nn.Module:
        """Remove every inserted branch and restore original trainability."""

        self._require_attached()
        for path in self.paths:
            parent, leaf = get_parent_module(self._model, path)
            wrapper = _child(parent, leaf)
            if not isinstance(wrapper, LayerRecallWrapper):
                raise RuntimeError(f"attached layer {path!r} was replaced outside ARTI")
            set_child_module(parent, leaf, wrapper.base)
        for name, parameter in self._model.named_parameters():
            if name in self._prior_trainability:
                parameter.requires_grad_(self._prior_trainability[name])
        object.__delattr__(self._model, "arti")
        self._attached = False
        return self._model

    def _bundle(self) -> _RecallBundle:
        return _RecallBundle(self._layered.wrappers[path].recall for path in self.paths)

    def _require_attached(self) -> None:
        if not self._attached:
            raise RuntimeError("ARTI attachment has been detached")


class ARTI:
    """Developer-friendly entry point for attaching ARTI to an existing model."""

    @staticmethod
    def discover(model: nn.Module, layers: str | Iterable[str] | None = None) -> tuple[ARTILayerInfo, ...]:
        return discover_layers(model, layers)

    @staticmethod
    def preview(
        model: nn.Module,
        recall: bool | Mapping[str, Any] | LayeredRecallConfig = True,
        *,
        sample_batch: Any | None = None,
    ) -> ARTIAttachmentSummary:
        config = _resolve_config(model, recall, sample_batch=sample_batch)
        return _summary(model, config, sample_batch=sample_batch)

    @staticmethod
    def attach(
        model: nn.Module,
        recall: bool | Mapping[str, Any] | LayeredRecallConfig = True,
        *,
        config: str | Path | ARTIAttachConfig | None = None,
        sample_batch: Any | None = None,
    ) -> nn.Module:
        """Attach Recall in place and return the original model object."""

        if not isinstance(model, nn.Module):
            raise TypeError("model must be a torch.nn.Module")
        if hasattr(model, "arti"):
            raise ValueError("model already has an ARTI attachment")
        declaration = None
        if config is not None:
            if recall is not True:
                raise ValueError("pass either recall=... or config=..., not both")
            declaration = load_attach_config(config) if isinstance(config, (str, Path)) else config
            recall = declaration.recall
        resolved_config = _resolve_config(model, recall, sample_batch=sample_batch)
        prior = {name: parameter.requires_grad for name, parameter in model.named_parameters()}
        try:
            layered = LayeredRecallModel.from_config(model, resolved_config, sample_batch=sample_batch)
        except Exception:
            _rollback_attachment(model, resolved_config.paths, prior)
            raise
        controller = ARTIAttachment(model, layered, resolved_config, prior, declaration)
        object.__setattr__(model, "arti", controller)
        return model

    @staticmethod
    def load(
        model: nn.Module,
        path: str | Path,
        *,
        sample_batch: Any | None = None,
        map_location: str | torch.device | None = None,
    ) -> nn.Module:
        """Reconstruct an attachment from its artifact and restore its weights."""

        inspected = load_arti(path, load_resources=False, load_checkpoint=False)
        metadata = _attachment_metadata(inspected.manifest)
        ARTI.attach(model, _config_from_payload(metadata["config"]), sample_batch=sample_batch)
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
        map_location: str | torch.device | None = None,
        model_kwargs: Mapping[str, Any] | None = None,
    ) -> nn.Module:
        """Load a base-model reference plus its independent ARTI artifact."""

        from .attachment_hub import load_attachment_pretrained

        return load_attachment_pretrained(
            directory,
            model=model,
            map_location=map_location,
            model_kwargs=model_kwargs,
        )


def discover_layers(model: nn.Module, layers: str | Iterable[str] | None = None) -> tuple[ARTILayerInfo, ...]:
    """Discover stable block-level insertion points without running the model."""

    named = tuple((name, module) for name, module in model.named_modules() if name)
    if layers is not None:
        patterns = (layers,) if isinstance(layers, str) else tuple(layers)
        selected = [(name, module) for name, module in named if any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)]
        missing = [pattern for pattern in patterns if not any(fnmatch.fnmatchcase(name, pattern) for name, _ in named)]
        if missing:
            raise ValueError(f"layer patterns matched no modules: {missing}")
    else:
        selected = [(name, module) for name, module in named if _is_transformer_block(name, module)]
        if not selected and isinstance(model, nn.Sequential):
            selected = [
                (name, module)
                for name, module in named
                if name.count(".") == 0 and _infer_module_dim(module) is not None and _is_shape_preserving(module)
            ]
    if not selected:
        raise ValueError("no compatible layers discovered; pass recall={'layers': 'path.or.glob'}")
    # A pattern may match both a block and its descendants. Keep the shallowest match.
    paths: list[tuple[str, nn.Module]] = []
    for name, module in selected:
        if not any(name.startswith(parent + ".") for parent, _ in paths):
            paths.append((name, module))
    return tuple(ARTILayerInfo(name, _infer_attachment_dim(model, module), type(module).__name__) for name, module in paths)


def _resolve_config(
    model: nn.Module,
    recall: bool | Mapping[str, Any] | LayeredRecallConfig,
    *,
    sample_batch: Any | None,
) -> LayeredRecallConfig:
    if recall is False:
        raise ValueError("recall=False creates no attachment; leave the model unchanged instead")
    if isinstance(recall, LayeredRecallConfig):
        return recall
    options = {} if recall is True else dict(recall)
    if options.pop("enabled", True) is False:
        raise ValueError("recall enabled=False creates no attachment; leave the model unchanged instead")
    layer_selector = options.pop("layers", options.pop("layer_paths", None))
    discovered = discover_layers(model, layer_selector)
    dims = {item.path: item.dim for item in discovered if item.dim is not None}
    unresolved = tuple(item.path for item in discovered if item.dim is None)
    if unresolved and sample_batch is not None:
        dims.update(_runtime_path_dims(model, unresolved, sample_batch))
    unknown = tuple(path for path in unresolved if path not in dims)
    if unknown:
        raise ValueError(f"cannot infer hidden dimensions for {unknown}; pass sample_batch or explicit LayeredRecallConfig")
    allowed = {"rank", "slots", "half", "use_half", "recognition", "recognition_mode", "recognition_threshold", "recognition_temperature", "copies", "combine", "freeze_backbone"}
    extra = set(options) - allowed
    if extra:
        raise ValueError(f"unknown Recall attachment options: {sorted(extra)}")
    rank = options.get("rank", 16)
    slots = options.get("slots", 8)
    use_half = options.get("half", options.get("use_half", True))
    recognition = options.get("recognition", options.get("recognition_mode", "alignment"))
    if recognition is False:
        recognition = "none"
    elif recognition is True:
        recognition = "alignment"
    specs = tuple(
        LayerRecallSpec(
            item.path,
            dim=dims[item.path],
            rank=int(_path_option(rank, item.path)),
            slots=int(_path_option(slots, item.path)),
            use_half=bool(_path_option(use_half, item.path)),
            recognition_mode=str(_path_option(recognition, item.path)),
            recognition_threshold=float(_path_option(options.get("recognition_threshold", 0.5), item.path)),
            recognition_temperature=float(_path_option(options.get("recognition_temperature", 0.1), item.path)),
            copies=int(_path_option(options.get("copies", 1), item.path)),
            combine=str(_path_option(options.get("combine", "sum"), item.path)),
        )
        for item in discovered
    )
    return LayeredRecallConfig(layers=specs, freeze_backbone=bool(options.get("freeze_backbone", True)))


def _summary(model: nn.Module, config: LayeredRecallConfig, *, sample_batch: Any | None = None) -> ARTIAttachmentSummary:
    infos = []
    dims = {}
    for spec in config.layers:
        dim = spec.dim
        if dim is None:
            module = dict(model.named_modules()).get(spec.path)
            dim = _infer_module_dim(module) if module is not None else None
        module = dict(model.named_modules()).get(spec.path)
        if isinstance(module, LayerRecallWrapper):
            module = module.base
        infos.append(ARTILayerInfo(spec.path, dim, type(module).__name__))
        if dim is not None:
            dims[spec.path] = dim
    unresolved = tuple(info.path for info in infos if info.dim is None)
    if unresolved and sample_batch is not None:
        dims.update(_runtime_path_dims(model, unresolved, sample_batch))
        infos = [ARTILayerInfo(info.path, dims.get(info.path), info.module_type) for info in infos]
    cost = estimate_layered_recall_cost(LayeredRecallCandidate("attachment", config), layer_dims=dims)
    total = sum(parameter.numel() for parameter in model.parameters())
    backbone = total - cost.parameters if hasattr(model, "arti") else total
    dtype_bytes = max((parameter.element_size() for parameter in model.parameters() if parameter.is_floating_point()), default=4)
    return ARTIAttachmentSummary(
        layers=tuple(infos),
        trainable_parameters=cost.parameters,
        backbone_parameters=backbone,
        parameter_fraction=cost.parameters / max(backbone, 1),
        multiply_adds_per_token=cost.token_multiply_adds,
        estimated_parameter_bytes=cost.parameters * dtype_bytes,
    )


def _config_payload(config: LayeredRecallConfig) -> dict[str, Any]:
    return {"freeze_backbone": config.freeze_backbone, "layers": [asdict(spec) for spec in config.layers]}


def _config_from_payload(payload: Mapping[str, Any]) -> LayeredRecallConfig:
    return LayeredRecallConfig(
        layers=tuple(LayerRecallSpec(**item) for item in payload["layers"]),
        freeze_backbone=bool(payload.get("freeze_backbone", True)),
    )


def _attachment_metadata(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    metadata = manifest.get("architecture", {}).get("config", {}).get("unified_attachment")
    if not isinstance(metadata, Mapping) or metadata.get("version") != 1:
        raise ValueError("artifact is not an ARTI unified Recall attachment")
    return metadata


def _branches(recall: LayerRecall | LayerRecallStack) -> tuple[LayerRecall, ...]:
    return tuple(recall.branches) if isinstance(recall, LayerRecallStack) else (recall,)


def _new_half(branch: LayerRecall) -> nn.Module:
    from .nn import Half

    return Half().to(device=branch.bank.device, dtype=branch.bank.dtype)


def _child(parent: nn.Module, leaf: str) -> nn.Module:
    return parent[int(leaf)] if leaf.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)) else getattr(parent, leaf)


def _rollback_attachment(model: nn.Module, paths: Iterable[str], prior: Mapping[str, bool]) -> None:
    for path in paths:
        try:
            parent, leaf = get_parent_module(model, path)
            current = _child(parent, leaf)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            continue
        if isinstance(current, LayerRecallWrapper):
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
    if isinstance(module, nn.Linear):
        return module.in_features == module.out_features
    return True


def _is_transformer_block(name: str, module: nn.Module) -> bool:
    cls = type(module).__name__.lower()
    block_class = any(token in cls for token in ("decoderlayer", "encoderlayer", "transformerblock")) or cls.endswith("block")
    indexed_layer = ".layers." in f".{name}." and name.rsplit(".", 1)[-1].isdigit()
    return block_class or indexed_layer


def _infer_attachment_dim(model: nn.Module, module: nn.Module) -> int | None:
    direct = _infer_module_dim(module)
    if direct is not None:
        return direct
    config = getattr(model, "config", None)
    for attribute in ("hidden_size", "d_model", "n_embd"):
        value = config.get(attribute) if isinstance(config, Mapping) else getattr(config, attribute, None)
        if isinstance(value, int) and value > 0:
            return value
    candidates = []
    for child in module.modules():
        if isinstance(child, nn.Linear) and child.in_features == child.out_features:
            candidates.append(child.out_features)
        else:
            value = _infer_module_dim(child)
            if value is not None:
                candidates.append(value)
    return max(set(candidates), key=candidates.count) if candidates else None


__all__ = ["ARTI", "ARTIAttachment", "ARTIAttachmentSummary", "ARTILayerInfo", "discover_layers"]
