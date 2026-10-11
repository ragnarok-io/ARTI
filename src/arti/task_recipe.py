"""Model-independent task recipes using the existing training lifecycle."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping
import json

import torch

from .python_factory import PythonFactoryConfig
from .recipe_data import json_fingerprint
from .recipe_workflow import fit_task_recipe, read_recipe_config, write_json
from .serialization import (
    _read_and_validate_package,
    _sidecar_paths,
    load,
    save,
)
from .training_binding import TrainingBinding, trainer
from .training_config import ARTITrainingConfig
from .training_run import ARTITrainingRunConfig


def _json_object(value, name):
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    return json.loads(json.dumps(dict(value), allow_nan=False))


@dataclass(frozen=True)
class TaskRecipeConfig:
    task: str
    factory: PythonFactoryConfig | str
    options: Mapping[str, Any] = field(default_factory=dict)
    resources: Mapping[str, Any] = field(default_factory=dict)
    training: ARTITrainingConfig = field(default_factory=ARTITrainingConfig)
    run: ARTITrainingRunConfig = field(default_factory=ARTITrainingRunConfig)

    def __post_init__(self):
        if not isinstance(self.task, str) or not self.task.strip():
            raise ValueError("task must be a nonempty application task name")
        if isinstance(self.factory, str):
            object.__setattr__(self, "factory", PythonFactoryConfig(self.factory))
        if not isinstance(self.factory, PythonFactoryConfig):
            raise ValueError("factory must be an installed reference or PythonFactoryConfig")
        object.__setattr__(self, "options", _json_object(self.options, "options"))
        object.__setattr__(self, "resources", _json_object(self.resources, "resources"))
        if not isinstance(self.training, ARTITrainingConfig) or not isinstance(
            self.run, ARTITrainingRunConfig
        ):
            raise ValueError("training/run must use ARTI's shared configuration types")

    @classmethod
    def read(cls, path):
        return cls.from_dict(read_recipe_config(path))

    @classmethod
    def from_dict(cls, payload):
        payload = dict(payload)
        if payload.pop("format", "arti.task-recipe-config") != "arti.task-recipe-config":
            raise ValueError("not a generic task recipe configuration")
        factory = payload.get("factory")
        if isinstance(factory, Mapping):
            payload["factory"] = PythonFactoryConfig(**factory)
        payload["training"] = ARTITrainingConfig(**payload.get("training", {}))
        payload["run"] = ARTITrainingRunConfig(**payload.get("run", {}))
        return cls(**payload)

    def to_dict(self):
        return {
            "format": "arti.task-recipe-config",
            "task": self.task,
            "factory": self.factory.to_dict(),
            "options": _json_object(self.options, "options"),
            "resources": _json_object(self.resources, "resources"),
            "training": asdict(self.training),
            "run": self.run.to_dict(),
        }


def _factory(config, supplied=None):
    if supplied is None:
        descriptor = config.factory
    elif isinstance(supplied, str):
        descriptor = PythonFactoryConfig(supplied, config.factory.kwargs)
    else:
        descriptor = supplied
    if (
        not isinstance(descriptor, PythonFactoryConfig)
        or descriptor.to_dict() != config.factory.to_dict()
    ):
        raise ValueError("task factory reference/kwargs differ from the task declaration")
    return descriptor.resolve()


def _manifest(path):
    from .recipe_deployment import _artifact
    path = target = _artifact(path)
    manifest, _lock = _read_and_validate_package(target, _sidecar_paths(target), allow_legacy=False)
    contract = manifest["architecture"]["config"].get("task_recipe", {})
    return path, contract, manifest


class TaskRecipe:
    """A task supplies its semantics; ARTI owns the shared run lifecycle.

    TrainingBinding selects execution, parameters and tensor-state ownership.
    This recipe stores that state with the standard arti.st codec rather than
    adopting a provider-specific checkpoint format.
    """

    def __init__(
        self,
        config: TaskRecipeConfig,
        model,
        *,
        train_data,
        validation_data,
        dataset_manifest,
        objective,
        objective_id: str,
        evaluate_fn,
        predict_fn,
        metadata=None,
        optimizer=None,
        scheduler=None,
        training_weight=None,
        evaluation_weight=None,
        max_grad_norm=None,
        validate_records=None,
        data_preparation=None,
    ):
        if not isinstance(objective_id, str) or not objective_id:
            raise ValueError("objective_id must describe the actual task objective")
        if not callable(evaluate_fn) or not callable(predict_fn):
            raise ValueError("tasks require explicit evaluate_fn and predict_fn")
        self.config, self.task_name = config, config.task
        source = model if isinstance(model, TrainingBinding) else TrainingBinding(model)
        self.model = source.model
        self.train_data, self.validation_data = train_data, validation_data
        self.dataset_manifest = _json_object(dataset_manifest, "dataset_manifest")
        self.data_preparation = (
            None if data_preparation is None else _json_object(data_preparation, "data_preparation")
        )
        if (
            not isinstance(self.dataset_manifest.get("sha256"), str)
            or not self.dataset_manifest["sha256"]
        ):
            raise ValueError("dataset_manifest requires its declared content/revision sha256")
        self._evaluate_fn, self._predict_fn = evaluate_fn, predict_fn
        self._validate_records = validate_records
        self.training_weight, self.evaluation_weight = training_weight, evaluation_weight
        self.max_grad_norm = max_grad_norm
        self._evaluation_kind = "builtin"
        self._resume_evaluation_kind = None
        selected_ids = {id(parameter) for parameter in source.parameters()}
        self.task_contract = {
            "format": "arti.task-recipe",
            "format_version": 1,
            "task": config.task,
            "factory": config.factory.to_dict(),
            "options": dict(config.options),
            "dataset": self.dataset_manifest,
            "objective": objective_id,
            "application": _json_object(metadata or {}, "metadata"),
            "training": {
                key: value
                for key, value in asdict(config.training).items()
                if key not in {"steps", "device"}
            },
            "optimizer_type": "torch.optim.adamw.AdamW"
            if optimizer is None
            else _type_name(optimizer),
            "scheduler_type": None if scheduler is None else _type_name(scheduler),
            "max_grad_norm": max_grad_norm,
            "trainable_parameters": [
                name
                for name, parameter in source.state_module().named_parameters()
                if id(parameter) in selected_ids
            ],
        }
        self.task_contract = _json_object(self.task_contract, "task contract")
        self.task_fingerprint = json_fingerprint(self.task_contract)
        self.task_contract["sha256"] = self.task_fingerprint
        self.binding = TrainingBinding(
            self.model,
            owner=source.owner,
            kind="task-recipe",
            parameters=source.parameters,
            state_module=source.state_module,
            summary=source.describe,
            save_checkpoint=self._save_checkpoint,
            load_checkpoint=self._load_checkpoint,
            checkpoint_metadata={
                "task_recipe_sha256": self.task_fingerprint,
                "training_config": asdict(config.training),
            },
        )
        self.session = trainer(
            self.binding,
            config=config.training,
            objective=objective,
            optimizer=optimizer,
            scheduler=scheduler,
        )

    @classmethod
    def from_config(cls, config, *, task_factory=None):
        config = config if isinstance(config, TaskRecipeConfig) else TaskRecipeConfig.read(config)
        recipe = _factory(config, task_factory)(config, mode="train")
        if not isinstance(recipe, cls) or recipe.config != config:
            raise ValueError("task factory must return TaskRecipe with the supplied configuration")
        return recipe

    def plan(self):
        return {
            **self.session.plan(),
            "task_recipe": self.task_contract,
            "deployment": {
                "format": "arti.st",
                "requires_task_factory": True,
                "state_scope": "read-only",
            },
        }

    def check(self, batch=None):
        import copy

        loader = self.train_data
        state = (
            copy.deepcopy(loader.state_dict())
            if callable(getattr(loader, "state_dict", None))
            else None
        )
        if batch is None and state is None and isinstance(loader, Iterator):
            raise ValueError("check a one-pass iterator with an explicit batch")
        try:
            with self.binding.evaluation_context():
                return {
                    **self.session.check(next(iter(loader)) if batch is None else batch),
                    "task_recipe_sha256": self.task_fingerprint,
                }
        finally:
            if state is not None:
                loader.load_state_dict(state)

    def fit(self, **kwargs):
        return fit_task_recipe(self, **kwargs)

    def validate_run(self, run, validation_fn=None):
        if (
            run.validation is not None
            and self._resume_evaluation_kind == "caller-supplied"
            and validation_fn is None
        ):
            raise ValueError("continuation of caller-supplied validation requires validation_fn")
        self._evaluation_kind = "builtin" if validation_fn is None else "caller-supplied"
        return {"kind": self._evaluation_kind}

    def resume_run(self, source_run, **kwargs):
        from .recipe_recovery import resume_task_recipe

        return resume_task_recipe(self, source_run, **kwargs)

    def evaluate(self, split="validation", *, output=None, checkpoint=None):
        context = (
            self.session.evaluation_context()
            if checkpoint is None
            else self.session.checkpoint_context(checkpoint)
        )
        with context, torch.no_grad():
            self.model.eval()
            metrics = _json_object(self._evaluate_fn(self.model, split), "evaluation metrics")
        report = {
            **metrics,
            "task": self.task_name,
            "split": split,
            "task_recipe_sha256": self.task_fingerprint,
            "dataset_sha256": self.dataset_manifest["sha256"],
        }
        if output is not None:
            write_json(output, report)
        return report

    def export(self, directory, *, selection=None):
        return save(
            self.binding.state_module(),
            Path(directory) / "model.st",
            config={
                "task_recipe": self.task_contract,
                **({"selection": dict(selection)} if selection is not None else {}),
            },
        ).weights_path

    def load_model(self, path):
        self._load_checkpoint(path, load_checkpoint=False)

    def _save_checkpoint(self, path, *, optimizer=None, scheduler=None, training_state=None):
        return save(
            self.binding.state_module(),
            path,
            config={"task_recipe": self.task_contract},
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=None
            if training_state is None
            else {
                **training_state,
                "task_evaluator_kind": self._evaluation_kind,
            },
        )

    def _load_checkpoint(
        self, path, *, optimizer=None, scheduler=None, map_location=None, load_checkpoint=True
    ):
        path, contract, _metadata = _manifest(path)
        if contract != self.task_contract:
            raise ValueError(
                "checkpoint task/data/objective/factory contract differs from this recipe"
            )
        device = (
            next(self.model.parameters(), torch.empty(0)).device
            if map_location is None
            else map_location
        )
        result = load(
            path,
            model=self.binding.state_module(),
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
            load_checkpoint=load_checkpoint,
        )
        if load_checkpoint and result.training_state is not None:
            self._resume_evaluation_kind = result.training_state.get(
                "task_evaluator_kind", "builtin"
            )
        return result

    def predictor(self):
        return TaskPredictor(
            self.binding,
            self.task_contract,
            predict_fn=self._predict_fn,
            validate_records=self._validate_records,
        )


def _type_name(value):
    return f"{type(value).__module__}.{type(value).__qualname__}"


class TaskPredictor:
    """Explicit task code plus saved tensor state, without a training session."""

    serving_state_scope = "read-only"

    def __init__(self, model, contract, *, predict_fn, evaluate_fn=None, validate_records=None):
        self.binding = model if isinstance(model, TrainingBinding) else TrainingBinding(model)
        self.model, self.contract = self.binding.model, _json_object(contract, "task contract")
        self._predict_fn, self._evaluate_fn = predict_fn, evaluate_fn
        self._validate_records = validate_records

    @classmethod
    def from_pretrained(cls, path, *, task_factory=None, device="cpu"):
        path, contract, manifest = _manifest(path)
        if contract.get("format") != "arti.task-recipe" or contract.get("format_version") != 1:
            raise ValueError("asset is not a generic task recipe")
        if contract.get("sha256") != json_fingerprint(
            {key: value for key, value in contract.items() if key != "sha256"}
        ):
            raise ValueError("task contract fingerprint differs")
        from .recipe_deployment import local_task_resources
        resources, deployment_sha256 = local_task_resources(path, manifest)
        if task_factory is None:
            raise ValueError(f"explicit task_factory required: {contract['factory']['reference']}")
        config = TaskRecipeConfig(
            task=contract["task"],
            factory=PythonFactoryConfig(**contract["factory"]),
            options=contract["options"],
            resources=resources,
            training=ARTITrainingConfig(device=device),
        )
        predictor = _factory(config, task_factory)(config, mode="predict", contract=contract)
        if not isinstance(predictor, cls) or predictor.contract != contract:
            raise ValueError("task factory must return TaskPredictor with the saved contract")
        predictor.model.to(device)
        load(
            path, model=predictor.binding.state_module(), map_location=device, load_checkpoint=False
        )
        predictor.model_source_sha256 = json_fingerprint(manifest)
        if deployment_sha256 is not None:
            predictor.deployment_sha256 = deployment_sha256
        return predictor

    def validate_records(self, records):
        if self._validate_records is not None:
            self._validate_records(records)

    def predict_records(self, records, *, batch_size=None, **predict_kwargs):
        self.validate_records(records)
        kwargs = {**predict_kwargs, **({} if batch_size is None else {"batch_size": batch_size})}
        with self.binding.evaluation_context(), torch.no_grad():
            self.model.eval()
            result = list(self._predict_fn(self.model, records, **kwargs))
        if len(result) != len(records):
            raise ValueError("task prediction must return one result per supplied record")
        for source, value in zip(records, result, strict=True):
            if (
                isinstance(source, Mapping)
                and isinstance(value, Mapping)
                and "id" in source
                and value.get("id") != source["id"]
            ):
                raise ValueError("task prediction must preserve input record ids and order")
        return result

    def evaluate_records(self, records, *, output=None, dataset_manifest=None):
        if self._evaluate_fn is None:
            raise ValueError("this task factory does not provide independent evaluation")
        self.validate_records(records)
        dataset = (
            {"sha256": json_fingerprint(records)}
            if dataset_manifest is None
            else _json_object(dataset_manifest, "dataset_manifest")
        )
        if not isinstance(dataset.get("sha256"), str) or not dataset["sha256"]:
            raise ValueError("dataset_manifest requires a declared content/revision sha256")
        with self.binding.evaluation_context(), torch.no_grad():
            self.model.eval()
            metrics = _json_object(self._evaluate_fn(self.model, records), "evaluation metrics")
        report = {
            **metrics,
            "task": self.contract["task"],
            "task_recipe_sha256": self.contract["sha256"],
            "dataset_sha256": dataset["sha256"],
            **({"deployment_sha256": self.deployment_sha256} if hasattr(self, "deployment_sha256") else {}),
            **(
                {"model_source_sha256": self.model_source_sha256}
                if hasattr(self, "model_source_sha256")
                else {}
            ),
        }
        if output is not None:
            write_json(output, report)
        return report


__all__ = ["TaskRecipeConfig", "TaskRecipe", "TaskPredictor"]
