"""Declarative pretrained-model adaptation workflow.

This module compiles provider-specific model objects into a stable ``ARTIPlan``
before mutating them. The same workflow then applies ARTI adapters, delegates
training to a selected engine, and exports ``arti.st`` plus a reproducibility
lock that binds source, structure, plan, environment, and weights.
"""

from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import torch
import torch.nn as nn

from ._version import __version__
from .fit import ARTIProject, project
from .fit.project import default_loss, run_model, split_batch
from .fit.runtime import adapter_context
from .providers import LoadedPretrained, ProviderInspection, provider_report, resolve_provider
from .serialization import ARTILoadResult, ARTISaveResult, load as load_st, save as save_st


PRETRAINED_PLAN_FORMAT = "arti.pretrained-plan"
PRETRAINED_PLAN_VERSION = 1
PRETRAINED_LOCK_FORMAT = "arti.pretrained-lock"
PRETRAINED_LOCK_VERSION = 1


@dataclass(frozen=True)
class TrainingSpec:
    """Portable training settings captured by an ``ARTIPlan``."""

    engine: str = "torch"
    learning_rate: float = 3e-4
    steps: int = 0
    mixed_precision: str = "no"
    gradient_accumulation_steps: int = 1
    distributed: bool = False
    device: str = "auto"

    @classmethod
    def from_value(cls, value: "TrainingSpec | Mapping[str, Any] | None") -> "TrainingSpec":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value.validate()
        return cls(
            engine=str(value.get("engine", "torch")),
            learning_rate=float(value.get("learning_rate", value.get("lr", 3e-4))),
            steps=int(value.get("steps", 0)),
            mixed_precision=str(value.get("mixed_precision", "no")),
            gradient_accumulation_steps=int(value.get("gradient_accumulation_steps", 1)),
            distributed=bool(value.get("distributed", False)),
            device=str(value.get("device", "auto")),
        ).validate()

    def validate(self) -> "TrainingSpec":
        if self.engine not in {"torch", "transformers", "accelerate"}:
            raise ValueError("training engine must be 'torch', 'transformers', or 'accelerate'")
        if self.learning_rate <= 0:
            raise ValueError("training learning_rate must be positive")
        if self.steps < 0:
            raise ValueError("training steps must be non-negative")
        if self.mixed_precision not in {"no", "fp16", "bf16"}:
            raise ValueError("training mixed_precision must be 'no', 'fp16', or 'bf16'")
        if self.gradient_accumulation_steps <= 0:
            raise ValueError("gradient_accumulation_steps must be positive")
        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("training device must be 'auto', 'cpu', or 'cuda'")
        return self


@dataclass(frozen=True)
class ComponentPlan:
    name: str
    class_path: str
    structure_fingerprint: str
    candidates: int
    selected: tuple[dict[str, Any], ...]
    adapter_parameters: int


@dataclass(frozen=True)
class ARTIPlan:
    """Serializable, immutable plan between pretrained discovery and mutation."""

    provider: str
    source: dict[str, Any]
    components: tuple[ComponentPlan, ...]
    mechanisms: dict[str, Any]
    insertion: dict[str, Any]
    training: TrainingSpec
    environment: dict[str, str | None]
    native_capabilities: tuple[str, ...]
    provider_metadata: dict[str, Any]
    format: str = PRETRAINED_PLAN_FORMAT
    format_version: int = PRETRAINED_PLAN_VERSION

    @property
    def fingerprint(self) -> str:
        return _stable_hash(self.to_dict(include_fingerprint=False))

    def to_dict(self, *, include_fingerprint: bool = True) -> dict[str, Any]:
        payload = {
            "format": self.format,
            "format_version": self.format_version,
            "provider": self.provider,
            "source": self.source,
            "components": [asdict(component) for component in self.components],
            "mechanisms": self.mechanisms,
            "insertion": self.insertion,
            "training": asdict(self.training),
            "environment": self.environment,
            "native_capabilities": list(self.native_capabilities),
            "provider_metadata": self.provider_metadata,
        }
        if include_fingerprint:
            payload["fingerprint"] = self.fingerprint
        return payload

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return target

    @classmethod
    def read(cls, path: str | Path) -> "ARTIPlan":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format") != PRETRAINED_PLAN_FORMAT or payload.get("format_version") != PRETRAINED_PLAN_VERSION:
            raise ValueError("unsupported ARTI pretrained plan format or version")
        expected = payload.pop("fingerprint", None)
        result = cls(
            provider=str(payload["provider"]),
            source=dict(payload["source"]),
            components=tuple(ComponentPlan(**component) for component in payload["components"]),
            mechanisms=dict(payload["mechanisms"]),
            insertion=dict(payload["insertion"]),
            training=TrainingSpec.from_value(payload["training"]),
            environment=dict(payload["environment"]),
            native_capabilities=tuple(payload["native_capabilities"]),
            provider_metadata=dict(payload["provider_metadata"]),
        )
        if expected != result.fingerprint:
            raise ValueError("ARTI pretrained plan fingerprint does not match its contents")
        return result


@dataclass(frozen=True)
class PretrainedFitResult:
    workflow: "ARTIPretrained"
    engine: str
    steps: int
    loss_history: tuple[float, ...]
    trainer: Any | None = None
    optimizer: torch.optim.Optimizer | None = None
    scheduler: Any | None = None

    @property
    def model(self) -> Any:
        return self.workflow.model

    def export(self, path: str | Path = "arti.st", *, include_base: bool = False) -> "PretrainedExportResult":
        return self.workflow.export(path, include_base=include_base)


@dataclass(frozen=True)
class PretrainedExportResult:
    saved: ARTISaveResult
    plan_path: Path
    lock_path: Path


class _ComponentBundle(nn.Module):
    """Stable state container for multi-component pipelines."""

    def __init__(self, components: Mapping[str, nn.Module]) -> None:
        super().__init__()
        self.components = nn.ModuleDict(dict(components))


class ARTIPretrained:
    """Stateful ``scan -> plan -> apply -> fit -> export`` workflow."""

    def __init__(
        self,
        source: str | Any,
        *,
        provider: str = "auto",
        task: str | None = None,
        revision: str | None = None,
        loader_kwargs: Mapping[str, Any] | None = None,
        components: Iterable[str] | None = None,
    ) -> None:
        self.source = source
        self.provider = resolve_provider(source, provider)
        self.task = task
        self.revision = revision
        self.loader_kwargs = dict(loader_kwargs or {})
        self.component_names = None if components is None else tuple(components)
        self.assets: LoadedPretrained | None = None
        self.inspection: ProviderInspection | None = None
        self.projects: dict[str, ARTIProject] = {}
        self.plan_value: ARTIPlan | None = None
        self.applied = False
        self.fit_result: PretrainedFitResult | None = None

    @property
    def model(self) -> Any:
        return self._ensure_loaded().root

    @property
    def tokenizer(self) -> Any | None:
        return self._ensure_loaded().tokenizer

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        root = self.model
        try:
            return getattr(root, name)
        except AttributeError:
            raise AttributeError(name) from None

    @contextmanager
    def context(
        self,
        *,
        mask: torch.Tensor | None = None,
        visibility: torch.Tensor | None = None,
        coord: torch.Tensor | None = None,
        observer_coord: torch.Tensor | None = None,
        frame_operators: torch.Tensor | None = None,
        causal: bool = False,
    ):
        """Bind mask/visibility/phase tensors while retaining the native model API."""

        with adapter_context(
            mask=mask,
            visibility=visibility,
            coord=coord,
            observer_coord=observer_coord,
            frame_operators=frame_operators,
            causal=causal,
        ):
            yield self.model

    def generate(self, *args: Any, arti_context: Mapping[str, Any] | None = None, **kwargs: Any) -> Any:
        """Call native ``generate`` with an optional ARTI runtime context."""

        native = getattr(self.model, "generate", None)
        if not callable(native):
            raise AttributeError("provider model does not expose generate()")
        if arti_context is None:
            return native(*args, **kwargs)
        context = dict(arti_context)
        if "mask" not in context and "attention_mask" in kwargs:
            context["mask"] = kwargs["attention_mask"]
        with self.context(**context):
            return native(*args, **kwargs)

    def scan(
        self,
        sample_batch: Any | None = None,
        *,
        component_samples: Mapping[str, Any] | None = None,
    ) -> "ARTIPretrained":
        assets = self._ensure_loaded()
        self.inspection = self.provider.inspect(assets.root, components=self.component_names)
        self.projects = {}
        samples = dict(component_samples or {})
        for name, component in self.inspection.components.items():
            builder = project(component)
            if self.provider.name in {"transformers", "peft"}:
                builder.plugin("transformers")
            elif self.provider.name == "diffusers":
                builder.plugin("vision-cnn" if name in {"unet", "vae"} else "transformers")
            builder.scan(samples.get(name, sample_batch if len(self.inspection.components) == 1 else None))
            self.projects[name] = builder
        return self

    def plan(
        self,
        *,
        features: Mapping[str, Any] | None = None,
        where: str | Iterable[str] | Mapping[str, str | Iterable[str]] | None = None,
        scale: str = "small",
        freeze_base: bool = True,
        max_adapters: int | None = None,
        max_extra_params: int | str | None = None,
        allow_empty: bool = False,
        training: TrainingSpec | Mapping[str, Any] | None = None,
    ) -> ARTIPlan:
        if not self.projects:
            self.scan()
        assert self.inspection is not None
        mechanisms = _normalize_features(features)
        components = []
        for name, builder in self.projects.items():
            builder.scale(scale)
            builder.mechanism(**_mechanism_kwargs(mechanisms))
            selected_where = where.get(name) if isinstance(where, Mapping) else where
            if selected_where is None and self.provider.name == "diffusers":
                selected_where = "vision-cnn" if name in {"unet", "vae"} else "attention"
            insertion = builder.plan_insert(
                where=selected_where,
                freeze_base=freeze_base,
                max_adapters=max_adapters,
                max_extra_params=max_extra_params,
                identity_gate=True,
                require_runtime_context=bool(mechanisms["phase"]["enabled"] and mechanisms["phase"]["mode"] == "external"),
            )
            if not insertion.selected and not allow_empty:
                candidates = [candidate.name for candidate in builder.scan_report.candidates[:8]]
                raise ValueError(
                    f"ARTIPlan selected no insertion points for component {name!r}; "
                    f"where={selected_where!r}, candidate examples={candidates}. "
                    "Choose a matching selector or pass allow_empty=True for inspection-only plans."
                )
            components.append(
                ComponentPlan(
                    name=name,
                    class_path=_class_path(builder.model),
                    structure_fingerprint=model_structure_fingerprint(builder.model),
                    candidates=len(builder.scan_report.candidates),
                    selected=tuple(dict(item) for item in insertion.to_dict()["selected"]),
                    adapter_parameters=insertion.adapter_parameters,
                )
            )
        source = {
            "model_id": self.source if isinstance(self.source, str) else self.inspection.metadata.get("model_id"),
            "requested_revision": self.revision,
            "resolved_revision": self.inspection.metadata.get("resolved_revision") or self.revision,
            "task": self.task,
            "components": None if self.component_names is None else list(self.component_names),
            "loader": _safe_loader_kwargs(self.loader_kwargs),
        }
        self.plan_value = ARTIPlan(
            provider=self.provider.name,
            source=source,
            components=tuple(components),
            mechanisms=mechanisms,
            insertion={
                "where": _json_where(where),
                "scale": scale,
                "freeze_base": freeze_base,
                "max_adapters": max_adapters,
                "max_extra_params": max_extra_params,
                "identity_gate": True,
                "allow_empty": allow_empty,
                "require_runtime_context": bool(mechanisms["phase"]["enabled"] and mechanisms["phase"]["mode"] == "external"),
            },
            training=TrainingSpec.from_value(training),
            environment=_environment_versions(),
            native_capabilities=self.inspection.native_capabilities,
            provider_metadata=dict(self.inspection.metadata),
        )
        return self.plan_value

    def apply(self, plan: ARTIPlan | str | Path | None = None) -> "ARTIPretrained":
        selected_plan = ARTIPlan.read(plan) if isinstance(plan, (str, Path)) else plan or self.plan_value
        if selected_plan is None:
            selected_plan = self.plan()
        if selected_plan.provider != self.provider.name:
            raise ValueError("ARTIPlan provider does not match the loaded workflow provider")
        if not self.projects:
            self.scan()
        for component_plan in selected_plan.components:
            if component_plan.name not in self.projects:
                raise ValueError(f"ARTIPlan component {component_plan.name!r} is absent from the provider inspection")
            builder = self.projects[component_plan.name]
            actual_fingerprint = model_structure_fingerprint(builder.model)
            if actual_fingerprint != component_plan.structure_fingerprint:
                raise ValueError(f"component {component_plan.name!r} structure changed after planning")
            builder.scale(str(selected_plan.insertion["scale"]))
            builder.mechanism(**_mechanism_kwargs(selected_plan.mechanisms))
            names = [str(row["name"]) for row in component_plan.selected]
            builder.insert(
                where=names or ("__arti_no_match__",),
                freeze_base=bool(selected_plan.insertion["freeze_base"]),
                max_adapters=len(names),
                max_extra_params=selected_plan.insertion.get("max_extra_params"),
                identity_gate=bool(selected_plan.insertion.get("identity_gate", True)),
                require_runtime_context=bool(selected_plan.insertion.get("require_runtime_context", False)),
            )
            if [item.name for item in builder.inserted] != names:
                raise ValueError(f"component {component_plan.name!r} insertion result does not match ARTIPlan")
        current_capabilities = self.provider.inspect(self.model, components=self.component_names).native_capabilities
        missing = sorted(set(selected_plan.native_capabilities) - set(current_capabilities))
        if missing:
            raise ValueError(f"applying ARTI removed native provider capabilities: {missing}")
        self.plan_value = selected_plan
        self.applied = True
        return self

    def fit(
        self,
        train_data: Any,
        *,
        loss_fn: Callable[[Any, Any], torch.Tensor] | None = None,
        step_fn: Callable[[Any, Any], torch.Tensor] | None = None,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
        trainer_kwargs: Mapping[str, Any] | None = None,
    ) -> PretrainedFitResult:
        if not self.applied:
            self.apply()
        assert self.plan_value is not None
        spec = self.plan_value.training.validate()
        if spec.engine == "transformers":
            result = self._fit_transformers(train_data, spec=spec, trainer_kwargs=trainer_kwargs)
        elif spec.engine == "accelerate":
            result = self._fit_accelerate(train_data, spec=spec, loss_fn=loss_fn, step_fn=step_fn, optimizer=optimizer, scheduler=scheduler)
        else:
            result = self._fit_torch(train_data, spec=spec, loss_fn=loss_fn, step_fn=step_fn, optimizer=optimizer, scheduler=scheduler)
        self.fit_result = result
        return result

    def export(self, path: str | Path = "arti.st", *, include_base: bool = False) -> PretrainedExportResult:
        if not self.applied or self.plan_value is None:
            raise ValueError("call apply() before export()")
        target = Path(path)
        state_model = self._state_model()
        saved = save_st(
            state_model,
            target,
            config={"pretrained_plan_fingerprint": self.plan_value.fingerprint},
            scope="all" if include_base else "trainable",
            optimizer=None if self.fit_result is None else self.fit_result.optimizer,
            scheduler=None if self.fit_result is None else self.fit_result.scheduler,
            training_state=None
            if self.fit_result is None
            else {
                "engine": self.fit_result.engine,
                "steps": self.fit_result.steps,
                "loss_history": list(self.fit_result.loss_history),
                "plan_fingerprint": self.plan_value.fingerprint,
            },
        )
        stem = target.with_suffix("")
        plan_path = stem.with_name(f"{stem.name}.plan.json")
        lock_path = stem.with_name(f"{stem.name}.pretrained.lock.json")
        self.plan_value.write(plan_path)
        lock = {
            "format": PRETRAINED_LOCK_FORMAT,
            "format_version": PRETRAINED_LOCK_VERSION,
            "arti_version": __version__,
            "plan_file": plan_path.name,
            "plan_sha256": _file_sha256(plan_path),
            "plan_fingerprint": self.plan_value.fingerprint,
            "weights_file": saved.weights_path.name,
            "weights_sha256": saved.weights_sha256,
            "weights_lock_file": saved.lock_path.name,
            "weights_lock_sha256": _file_sha256(saved.lock_path),
            "source": self.plan_value.source,
            "environment": self.plan_value.environment,
            "component_structures": {component.name: component.structure_fingerprint for component in self.plan_value.components},
            "training": asdict(self.plan_value.training),
        }
        lock_path.write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return PretrainedExportResult(saved=saved, plan_path=plan_path, lock_path=lock_path)

    def load_weights(
        self,
        path: str | Path = "arti.st",
        *,
        map_location: str | torch.device = "cpu",
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: Any | None = None,
    ) -> ARTILoadResult:
        if not self.applied:
            raise ValueError("apply the matching ARTIPlan before loading pretrained ARTI weights")
        result = load_st(path, model=self._state_model(), optimizer=optimizer, scheduler=scheduler, map_location=map_location)
        state = result.training_state
        if isinstance(state, Mapping) and state.get("plan_fingerprint") != self.plan_value.fingerprint:
            raise ValueError("arti.st training checkpoint belongs to a different ARTIPlan")
        return result

    def doctor(self) -> dict[str, Any]:
        inspection = self.inspection
        return {
            "ok": self.provider.available,
            "provider": self.provider.name,
            "provider_available": self.provider.available,
            "providers": list(provider_report()),
            "loaded": self.assets is not None,
            "scanned": bool(self.projects),
            "planned": self.plan_value is not None,
            "applied": self.applied,
            "components": [] if inspection is None else list(inspection.components),
            "native_capabilities": [] if inspection is None else list(inspection.native_capabilities),
        }

    def _ensure_loaded(self) -> LoadedPretrained:
        if self.assets is None:
            if isinstance(self.source, str):
                self.assets = self.provider.load(self.source, task=self.task, revision=self.revision, kwargs=self.loader_kwargs)
            else:
                self.assets = LoadedPretrained(root=self.source)
        return self.assets

    def _state_model(self) -> nn.Module:
        assert self.inspection is not None
        if isinstance(self.model, nn.Module) and len(self.inspection.components) == 1:
            return self.model
        return _ComponentBundle(self.inspection.components)

    def _fit_torch(
        self,
        train_data: Iterable[Any],
        *,
        spec: TrainingSpec,
        loss_fn: Callable[[Any, Any], torch.Tensor] | None,
        step_fn: Callable[[Any, Any], torch.Tensor] | None,
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any | None,
    ) -> PretrainedFitResult:
        return self._manual_train(train_data, spec=spec, loss_fn=loss_fn, step_fn=step_fn, optimizer=optimizer, scheduler=scheduler)

    def _fit_accelerate(
        self,
        train_data: Iterable[Any],
        *,
        spec: TrainingSpec,
        loss_fn: Callable[[Any, Any], torch.Tensor] | None,
        step_fn: Callable[[Any, Any], torch.Tensor] | None,
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any | None,
    ) -> PretrainedFitResult:
        try:
            from accelerate import Accelerator, DistributedDataParallelKwargs
        except ImportError as exc:
            raise RuntimeError("accelerate training requires `uv sync --extra qwen` or `uv sync --extra sd`") from exc
        state_model = self._state_model()
        params = [parameter for parameter in state_model.parameters() if parameter.requires_grad]
        opt = optimizer or torch.optim.AdamW(params, lr=spec.learning_rate)
        accelerator = Accelerator(
            mixed_precision=spec.mixed_precision,
            gradient_accumulation_steps=spec.gradient_accumulation_steps,
            cpu=spec.device == "cpu",
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)] if spec.distributed else None,
        )
        if spec.distributed and accelerator.num_processes < 2:
            raise RuntimeError("distributed=True requires launching with `torchrun` or a multi-process `accelerate launch`")
        if scheduler is None:
            prepared_model, opt, prepared_loader = accelerator.prepare(state_model, opt, train_data)
            prepared_scheduler = None
        else:
            prepared_model, opt, prepared_loader, prepared_scheduler = accelerator.prepare(state_model, opt, train_data, scheduler)
        history = _training_loop(
            prepared_model,
            prepared_loader,
            steps=spec.steps,
            optimizer=opt,
            loss_fn=loss_fn,
            step_fn=step_fn,
            backward=accelerator.backward,
            accumulation_context=lambda: accelerator.accumulate(prepared_model),
            scheduler=prepared_scheduler,
        )
        return PretrainedFitResult(
            self,
            "accelerate",
            spec.steps,
            tuple(history),
            trainer=accelerator,
            optimizer=opt,
            scheduler=prepared_scheduler,
        )

    def _fit_transformers(self, train_data: Any, *, spec: TrainingSpec, trainer_kwargs: Mapping[str, Any] | None) -> PretrainedFitResult:
        if not isinstance(self.model, nn.Module):
            raise ValueError("the transformers engine requires a single nn.Module model")
        try:
            from transformers import Trainer, TrainingArguments
        except ImportError as exc:
            raise RuntimeError("transformers training requires `uv sync --extra qwen`") from exc
        kwargs = dict(trainer_kwargs or {})
        arguments = kwargs.pop("args", None)
        if arguments is None:
            argument_values = dict(kwargs.pop("training_args", {}))
            fields = getattr(TrainingArguments, "__dataclass_fields__", {})
            for key in tuple(kwargs):
                if key in fields:
                    argument_values[key] = kwargs.pop(key)
            resolved_arguments = {
                "output_dir": str(argument_values.pop("output_dir", ".arti-trainer")),
                "learning_rate": spec.learning_rate,
                "max_steps": spec.steps,
                "gradient_accumulation_steps": spec.gradient_accumulation_steps,
                "fp16": spec.mixed_precision == "fp16",
                "bf16": spec.mixed_precision == "bf16",
                "report_to": [],
            }
            resolved_arguments.update(argument_values)
            arguments = TrainingArguments(**resolved_arguments)
        trainer = Trainer(model=self.model, args=arguments, train_dataset=train_data, **kwargs)
        if spec.distributed and int(getattr(trainer.args, "world_size", 1)) < 2:
            raise RuntimeError("distributed=True requires launching Trainer with torchrun or accelerate launch on at least two processes")
        output = trainer.train()
        history = tuple(float(row["loss"]) for row in trainer.state.log_history if "loss" in row)
        steps = int(getattr(output, "global_step", spec.steps))
        return PretrainedFitResult(
            self,
            "transformers",
            steps,
            history,
            trainer=trainer,
            optimizer=trainer.optimizer,
            scheduler=trainer.lr_scheduler,
        )

    def _manual_train(
        self,
        train_data: Iterable[Any],
        *,
        spec: TrainingSpec,
        loss_fn: Callable[[Any, Any], torch.Tensor] | None,
        step_fn: Callable[[Any, Any], torch.Tensor] | None,
        optimizer: torch.optim.Optimizer | None,
        scheduler: Any | None,
    ) -> PretrainedFitResult:
        state_model = self._state_model()
        params = [parameter for parameter in state_model.parameters() if parameter.requires_grad]
        if not params:
            raise ValueError("no trainable ARTI parameters are available")
        opt = optimizer or torch.optim.AdamW(params, lr=spec.learning_rate)
        history = _training_loop(
            state_model,
            train_data,
            steps=spec.steps,
            optimizer=opt,
            loss_fn=loss_fn,
            step_fn=step_fn,
            scheduler=scheduler,
        )
        return PretrainedFitResult(self, "torch", spec.steps, tuple(history), optimizer=opt, scheduler=scheduler)


def pretrained(
    source: str | Any,
    *,
    provider: str = "auto",
    task: str | None = None,
    revision: str | None = None,
    loader_kwargs: Mapping[str, Any] | None = None,
    components: Iterable[str] | None = None,
) -> ARTIPretrained:
    """Create a declarative pretrained-model workflow without mutating it."""

    return ARTIPretrained(
        source,
        provider=provider,
        task=task,
        revision=revision,
        loader_kwargs=loader_kwargs,
        components=components,
    )


def from_pretrained(
    source: str | Any,
    *,
    provider: str = "auto",
    task: str | None = None,
    revision: str | None = None,
    loader_kwargs: Mapping[str, Any] | None = None,
    components: Iterable[str] | None = None,
    sample_batch: Any | None = None,
    features: Mapping[str, Any] | None = None,
    where: str | Iterable[str] | Mapping[str, str | Iterable[str]] | None = None,
    scale: str = "small",
    freeze_base: bool = True,
    max_adapters: int | None = None,
    max_extra_params: int | str | None = None,
    allow_empty: bool = False,
    training: TrainingSpec | Mapping[str, Any] | None = None,
) -> ARTIPretrained:
    """Load, scan, plan, and apply ARTI to a pretrained model in one call."""

    workflow = pretrained(
        source,
        provider=provider,
        task=task,
        revision=revision,
        loader_kwargs=loader_kwargs,
        components=components,
    )
    workflow.scan(sample_batch).plan(
        features=features,
        where=where,
        scale=scale,
        freeze_base=freeze_base,
        max_adapters=max_adapters,
        max_extra_params=max_extra_params,
        allow_empty=allow_empty,
        training=training,
    )
    workflow.apply()
    return workflow


def validate_pretrained_lock(path: str | Path, *, strict_environment: bool = True) -> dict[str, Any]:
    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload.get("format") != PRETRAINED_LOCK_FORMAT or payload.get("format_version") != PRETRAINED_LOCK_VERSION:
        raise ValueError("unsupported ARTI pretrained lock format or version")
    plan_path = target.parent / payload["plan_file"]
    weights_path = target.parent / payload["weights_file"]
    weights_lock_path = target.parent / payload["weights_lock_file"]
    for member, expected in (
        (plan_path, payload["plan_sha256"]),
        (weights_path, payload["weights_sha256"]),
        (weights_lock_path, payload["weights_lock_sha256"]),
    ):
        if not member.exists() or _file_sha256(member) != expected:
            raise ValueError(f"ARTI pretrained lock SHA-256 mismatch for {member.name}")
    plan = ARTIPlan.read(plan_path)
    if plan.fingerprint != payload.get("plan_fingerprint"):
        raise ValueError("ARTI pretrained lock plan fingerprint mismatch")
    expected_fields = {
        "source": plan.source,
        "environment": plan.environment,
        "component_structures": {component.name: component.structure_fingerprint for component in plan.components},
        "training": asdict(plan.training),
    }
    for key, expected in expected_fields.items():
        if payload.get(key) != expected:
            raise ValueError(f"ARTI pretrained lock {key} does not match ARTIPlan")
    load_st(weights_path, load_resources=False, load_checkpoint=False)
    if strict_environment:
        current = _environment_versions()
        mismatches = {
            name: {"locked": locked, "current": current.get(name)}
            for name, locked in plan.environment.items()
            if current.get(name) != locked
        }
        if mismatches:
            raise ValueError(f"ARTI pretrained lock environment mismatch: {mismatches}")
    return payload


def model_structure_fingerprint(model: nn.Module) -> str:
    """Hash module topology and tensor shapes without hashing learned values."""

    modules = [
        {
            "name": name,
            "class": _class_path(module),
            "parameters": [
                {"name": param_name, "shape": list(parameter.shape), "dtype": str(parameter.dtype)}
                for param_name, parameter in module.named_parameters(recurse=False)
            ],
            "buffers": [
                {"name": buffer_name, "shape": list(buffer.shape), "dtype": str(buffer.dtype)}
                for buffer_name, buffer in module.named_buffers(recurse=False)
            ],
        }
        for name, module in model.named_modules()
    ]
    return _stable_hash(modules)


def _normalize_features(value: Mapping[str, Any] | None) -> dict[str, Any]:
    payload = dict(value or {})
    recall = payload.get("recall", {})
    phase = payload.get("phase", {})
    half = payload.get("half", {})
    recall = {"enabled": bool(recall)} if isinstance(recall, bool) else dict(recall)
    phase = {"enabled": bool(phase)} if isinstance(phase, bool) else dict(phase)
    half = {"enabled": bool(half)} if isinstance(half, bool) else dict(half)
    result = {
        "recall": {
            "enabled": bool(recall.get("enabled", bool(recall))),
            "steps": int(recall.get("steps", 1 if recall else 0)),
            "slots": int(recall.get("slots", 4)),
        },
        "half": {"enabled": bool(half.get("enabled", True))},
        "phase": {
            "enabled": bool(phase.get("enabled", bool(phase))),
            "coord_dim": int(phase.get("coord_dim", phase.get("dim", 16))),
            "frame_mode": str(phase.get("frame_mode", "operator_bank")),
            "mode": str(phase.get("mode", "external")),
        },
    }
    if result["recall"]["steps"] < 0 or result["recall"]["slots"] <= 0:
        raise ValueError("recall steps must be non-negative and slots must be positive")
    if result["phase"]["coord_dim"] <= 0:
        raise ValueError("phase coord_dim must be positive")
    if result["phase"]["mode"] != "external":
        raise ValueError("pretrained phase mode currently supports only 'external'")
    return result


def _mechanism_kwargs(features: Mapping[str, Any]) -> dict[str, Any]:
    recall = features["recall"]
    phase = features["phase"]
    return {
        "recall_steps": recall["steps"] if recall["enabled"] else 0,
        "recall_slots": recall["slots"],
        "recall_activation": "half" if features["half"]["enabled"] else "none",
        "observer_phase": phase["enabled"],
        "coord_dim": phase["coord_dim"] if phase["enabled"] else 0,
        "coord_frame_mode": phase["frame_mode"] if phase["enabled"] else "none",
    }


def _training_loop(
    model: nn.Module,
    loader: Iterable[Any],
    *,
    steps: int,
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[Any, Any], torch.Tensor] | None,
    step_fn: Callable[[Any, Any], torch.Tensor] | None,
    backward: Callable[[torch.Tensor], None] | None = None,
    accumulation_context: Callable[[], Any] | None = None,
    scheduler: Any | None = None,
) -> list[float]:
    if steps == 0:
        return []
    iterator = iter(loader)
    history = []
    model.train()
    for _ in range(steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        context = accumulation_context() if accumulation_context is not None else _NullContext()
        with context:
            if step_fn is not None:
                loss = step_fn(model, batch)
            else:
                inputs, target = split_batch(batch)
                output = run_model(model, inputs)
                loss = default_loss(output, target) if loss_fn is None else loss_fn(output, target)
            optimizer.zero_grad(set_to_none=True)
            (backward or torch.Tensor.backward)(loss)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
        history.append(float(loss.detach().cpu()))
    return history


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type, exc, traceback) -> bool:
        return False


def _environment_versions() -> dict[str, str | None]:
    result: dict[str, str | None] = {"arti": __version__, "torch": torch.__version__}
    for name in ("transformers", "peft", "diffusers", "accelerate", "safetensors"):
        try:
            result[name] = version(name)
        except PackageNotFoundError:
            result[name] = None
    return result


def _json_where(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_where(item) for key, item in value.items()}
    if isinstance(value, str) or value is None:
        return value
    return list(value)


def _safe_loader_kwargs(value: Mapping[str, Any]) -> dict[str, Any]:
    result = {}
    for key, item in value.items():
        if key in {"token", "use_auth_token"}:
            result[key] = "<redacted>"
        elif item is None or isinstance(item, (bool, int, float, str)):
            result[key] = item
        elif isinstance(item, (list, tuple)):
            result[key] = [str(part) for part in item]
        else:
            result[key] = str(item)
    return result


def _class_path(value: Any) -> str:
    return f"{value.__class__.__module__}.{value.__class__.__qualname__}"


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "ARTIPlan",
    "ARTIPretrained",
    "ComponentPlan",
    "PRETRAINED_LOCK_FORMAT",
    "PRETRAINED_LOCK_VERSION",
    "PRETRAINED_PLAN_FORMAT",
    "PRETRAINED_PLAN_VERSION",
    "PretrainedExportResult",
    "PretrainedFitResult",
    "TrainingSpec",
    "from_pretrained",
    "model_structure_fingerprint",
    "pretrained",
    "validate_pretrained_lock",
]
