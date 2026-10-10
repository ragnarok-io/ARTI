"""Declarative pretrained-model adaptation workflow.

This module compiles provider-specific model objects into a stable ``ARTIPlan``
before mutating them. The same workflow then applies ARTI adapters, delegates
training to a selected engine, and exports ``arti.st`` plus a reproducibility
lock that binds source, structure, plan, environment, and weights.
"""

from __future__ import annotations

import hashlib
import inspect
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
from .training_binding import TrainingBinding
from .training_config import ARTITrainingConfig
from .training_run import ARTITrainingRunConfig, RunReporter


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
        ARTITrainingConfig(**asdict(self))
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
    session: Any | None = None
    checkpoint_path: Path | None = None
    run_directory: Path | None = None
    stop_reason: str = "steps_completed"

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


class _PretrainedAdapterBundle(nn.Module):
    def __init__(self, workflow: "ARTIPretrained") -> None:
        super().__init__()
        self.adapters = nn.ModuleDict()
        self.gates = nn.ParameterDict()
        self.paths = []
        for component in workflow.plan_value.components:
            modules = dict(workflow.projects[component.name].model.named_modules())
            for selected in component.selected:
                wrapper = modules[selected["name"]]
                key = f"adapter_{len(self.paths)}"
                self.adapters[key] = wrapper.adapter
                if wrapper.output_gate is not None:
                    self.gates[key] = wrapper.output_gate
                self.paths.append(f"{component.name}:{selected['name']}")


class ARTIPretrained:
    """Stateful ``scan -> plan -> apply -> fit -> export -> detach`` workflow."""

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
        self._training_session: Any | None = None
        self._resume_artifact: Path | None = None

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
        identity_gate: bool = False,
        zero_init_output: bool = True,
        bridge_mode: str = "radial",
        boundary_mask_key: str | None = None,
        allow_empty: bool = False,
        training: TrainingSpec | Mapping[str, Any] | None = None,
    ) -> ARTIPlan:
        if not self.projects:
            self.scan()
        assert self.inspection is not None
        if identity_gate and zero_init_output:
            raise ValueError("identity_gate and zero_init_output are mutually exclusive")
        if bridge_mode not in {"radial", "dense"}:
            raise ValueError("bridge_mode must be 'radial' or 'dense'")
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
                identity_gate=identity_gate,
                zero_init_output=zero_init_output,
                bridge_mode=bridge_mode,
                boundary_mask_key=boundary_mask_key,
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
                "identity_gate": identity_gate,
                "zero_init_output": zero_init_output,
                "bridge_mode": bridge_mode,
                "boundary_mask_key": boundary_mask_key,
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
                zero_init_output=bool(selected_plan.insertion.get("zero_init_output", False)),
                bridge_mode=str(selected_plan.insertion.get("bridge_mode", "dense")),
                boundary_mask_key=selected_plan.insertion.get("boundary_mask_key"),
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
        steps: int | None = None,
        run: ARTITrainingRunConfig | None = None,
        eval_data: Iterable[Any] | None = None,
        evaluation_weight: Callable[[Any], float] | None = None,
        reporter: RunReporter | None = None,
        stop_requested: Callable[[], bool] | None = None,
        resume_from_checkpoint: str | Path | None = None,
    ) -> PretrainedFitResult:
        if not self.applied:
            self.apply()
        assert self.plan_value is not None
        session = self.trainer(loss_fn=loss_fn, step_fn=step_fn, optimizer=optimizer, scheduler=scheduler, resume_from_checkpoint=resume_from_checkpoint)
        fitted = session.fit(train_data, steps=steps, trainer_kwargs=trainer_kwargs, run=run, eval_data=eval_data, evaluation_weight=evaluation_weight, reporter=reporter, stop_requested=stop_requested)
        result = PretrainedFitResult(self, fitted.engine, fitted.steps, fitted.loss_history, trainer=session.engine_object, optimizer=session.optimizer, scheduler=session.scheduler, session=session, checkpoint_path=fitted.checkpoint_path, run_directory=fitted.run_directory, stop_reason=fitted.stop_reason)
        self.fit_result = result
        return result

    def trainer(self, *, objective=None, loss_fn=None, step_fn=None, optimizer=None, scheduler=None, resume_from_checkpoint=None, reset=False):
        from .attachment_training import ARTITrainingSession, _bind_objective, _optimizer_parameter_ids, resolve_attachment_objective

        if not self.applied:
            self.apply()
        if objective is not None and (loss_fn is not None or step_fn is not None):
            raise ValueError("choose objective or the loss_fn/step_fn adaptation")
        if self._training_session is not None and not reset:
            session = self._training_session
            if optimizer is not None:
                if _optimizer_parameter_ids(optimizer) != {id(p) for p in session.binding.parameters()}:
                    raise ValueError("optimizer parameters must match the pretrained trainable role")
                session.optimizer = optimizer
            if scheduler is not None:
                session.scheduler = scheduler
            if objective is not None or loss_fn is not None or step_fn is not None:
                session.objective = resolve_attachment_objective(objective) if objective is not None else _pretrained_objective(self._state_model(), loss_fn, step_fn)
                session._objective = _bind_objective(session.objective)
            if resume_from_checkpoint is not None:
                if session.config.engine == "transformers":
                    session.resume_from_checkpoint = resume_from_checkpoint
                else:
                    session.load_checkpoint(resume_from_checkpoint)
            return session
        spec = self.plan_value.training.validate()
        config = ARTITrainingConfig(engine=spec.engine, learning_rate=spec.learning_rate, steps=spec.steps, mixed_precision=spec.mixed_precision, gradient_accumulation_steps=spec.gradient_accumulation_steps, device=spec.device, distributed=spec.distributed)
        model = self._state_model()
        binding = TrainingBinding(
            model,
            owner=self,
            kind="pretrained",
            state_module=self._training_state_model,
            save_checkpoint=self._save_training_checkpoint,
            load_checkpoint=self._load_training_checkpoint,
            checkpoint_metadata={"plan_fingerprint": self.plan_value.fingerprint},
        )
        self._training_session = ARTITrainingSession(binding, config=config, objective=objective or _pretrained_objective(model, loss_fn, step_fn), optimizer=optimizer, scheduler=scheduler, resume_from_checkpoint=resume_from_checkpoint or (None if reset else self._resume_artifact))
        return self._training_session

    def export(self, path: str | Path = "arti.st", *, include_base: bool = False) -> PretrainedExportResult:
        if not self.applied or self.plan_value is None:
            raise ValueError("call apply() before export()")
        target = Path(path)
        state_model = self._state_model() if include_base else self._training_state_model()
        session = self._training_session
        saved = save_st(
            state_model,
            target,
            config={"pretrained_plan_fingerprint": self.plan_value.fingerprint, "pretrained_state_scope": "model" if include_base or not self.plan_value.insertion["freeze_base"] else "adapters"},
            scope="all",
            optimizer=None if session is None else session.optimizer,
            scheduler=None if session is None else session.scheduler,
            training_state=None
            if session is None
            else session._checkpoint_state(),
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

    def detach(self) -> Any:
        """Remove all ARTI insertions and restore the provider model in place."""

        for builder in self.projects.values():
            builder.detach()
        self.applied = False
        self.fit_result = None
        self._training_session = None
        self._resume_artifact = None
        return self.model

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
        result = self._load_training_checkpoint(path, optimizer=optimizer, scheduler=scheduler, map_location=map_location)
        state = result.training_state
        if isinstance(state, Mapping) and state.get("plan_fingerprint") != self.plan_value.fingerprint:
            raise ValueError("arti.st training checkpoint belongs to a different ARTIPlan")
        if state is not None:
            self._resume_artifact = Path(path)
            if self._training_session is not None:
                self._training_session.load_checkpoint(path, map_location=map_location)
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

    def _training_state_model(self) -> nn.Module:
        return _PretrainedAdapterBundle(self) if self.plan_value.insertion["freeze_base"] else self._state_model()

    def _save_training_checkpoint(self, path, *, optimizer=None, scheduler=None, training_state=None):
        return save_st(self._training_state_model(), path, config={"pretrained_plan_fingerprint": self.plan_value.fingerprint, "pretrained_state_scope": "adapters" if self.plan_value.insertion["freeze_base"] else "model"}, optimizer=optimizer, scheduler=scheduler, training_state=training_state)

    def _load_training_checkpoint(self, path, *, optimizer=None, scheduler=None, map_location=None, load_checkpoint=True):
        inspected = load_st(path, load_resources=False, load_checkpoint=False)
        config = inspected.manifest["architecture"]["config"]
        if config.get("pretrained_plan_fingerprint") != self.plan_value.fingerprint:
            raise ValueError("arti.st training checkpoint belongs to a different ARTIPlan")
        model = self._training_state_model() if config.get("pretrained_state_scope") == "adapters" else self._state_model()
        device = next(self._state_model().parameters()).device if map_location is None else map_location
        return load_st(path, model=model, optimizer=optimizer, scheduler=scheduler, map_location=device, load_checkpoint=load_checkpoint)

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
    identity_gate: bool = False,
    zero_init_output: bool = True,
    bridge_mode: str = "radial",
    boundary_mask_key: str | None = None,
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
        identity_gate=identity_gate,
        zero_init_output=zero_init_output,
        bridge_mode=bridge_mode,
        boundary_mask_key=boundary_mask_key,
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


def _pretrained_objective(model: nn.Module, loss_fn=None, step_fn=None):
    if loss_fn is not None and step_fn is not None:
        raise ValueError("choose loss_fn or step_fn, not both")
    if step_fn is not None:
        return step_fn
    signature = inspect.signature(model.forward)
    accepts_labels = "labels" in signature.parameters or any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())

    def objective(host, batch):
        inputs, target = split_batch(batch)
        if isinstance(batch, Mapping) and "labels" in batch and accepts_labels:
            output = host(**batch)
        else:
            output = run_model(host, inputs)
        native_loss = output.get("loss") if isinstance(output, Mapping) else getattr(output, "loss", None)
        if loss_fn is None and isinstance(native_loss, torch.Tensor):
            return native_loss
        return default_loss(output, target) if loss_fn is None else loss_fn(output, target)

    objective._arti_native_model_loss = loss_fn is None and accepts_labels
    return objective


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
