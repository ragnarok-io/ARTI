"""Executable supervised task recipes using ARTI's shared training service."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .recipe_workflow import fit_task_recipe, read_recipe_config
from .recipe_workflow import write_json as _write_json, write_jsonl as _write_jsonl
from .formula_v2 import (
    BankBinding,
    FormulaFabricV2,
    FormulaProgram,
    InputBinding,
    TensorType,
    add,
    broadcast,
    contract,
    scalar_map,
)
from .recipe_data import (
    ClassificationBatches,
    ClassificationDataset,
    ClassificationFields,
    _positive_int,
    feature_vector,
    json_fingerprint,
    read_jsonl,
)
from .resource_graph import (
    FormulaProgramNode,
    ProgramGraph,
    ResourceGraphCompiler,
    ResourcePort,
    TensorResource,
    TensorResourceSpec,
)
from .serialization import load, save
from .tensor_view import TensorView, TensorViewPattern
from .training_binding import TrainingBinding, trainer
from .training_config import ARTITrainingConfig
from .training_run import ARTITrainingRunConfig


@dataclass(frozen=True)
class ClassificationRecipeConfig:
    data_path: str | Path
    fields: ClassificationFields = field(default_factory=ClassificationFields)
    labels: tuple[str, ...] | None = None
    hidden_dim: int = 64
    preprocessing: str = "standardize"
    batch_size: int = 64
    seed: int = 17
    training: ARTITrainingConfig = field(
        default_factory=lambda: ARTITrainingConfig(device="cpu", steps=200, learning_rate=0.003)
    )
    run: ARTITrainingRunConfig = field(default_factory=ARTITrainingRunConfig)

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_path", Path(self.data_path).resolve())
        _positive_int(self.hidden_dim, "hidden_dim")
        _positive_int(self.batch_size, "batch_size")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("recipe.seed must be a nonnegative integer")
        if self.preprocessing not in {"standardize", "none"}:
            raise ValueError("preprocessing must be standardize or none")
        if self.training.engine not in {"torch", "accelerate"} or self.training.distributed:
            raise ValueError(
                "the local classification recipe supports single-process Torch or Accelerate"
            )
        if self.training.objective != "model_loss":
            raise ValueError(
                "classification uses its real cross-entropy task loss, not tensor_alignment"
            )

    @classmethod
    def read(cls, path: str | Path) -> ClassificationRecipeConfig:
        path = Path(path)
        payload = read_recipe_config(path)
        if not isinstance(payload, dict) or payload.get("task") != "classification":
            raise ValueError("recipe.task must be classification")
        unknown = payload.keys() - {"task", "data", "model", "training", "run", "seed"}
        if unknown:
            raise ValueError(f"unknown recipe fields: {sorted(unknown)}")
        data, model = dict(payload.get("data", {})), dict(payload.get("model", {}))
        unknown = data.keys() - {"path", "fields", "labels", "batch_size", "preprocessing"}
        if unknown or model.keys() - {"hidden_dim"}:
            raise ValueError("unknown recipe data/model settings")
        if not isinstance(data.get("path"), str) or not data["path"]:
            raise ValueError("recipe.data.path must name a JSONL file")
        data_path = Path(data["path"])
        if not data_path.is_absolute():
            data_path = path.parent / data_path
        run = dict(payload.get("run", {}))
        if run.get("output_dir") is not None:
            directory = Path(run["output_dir"])
            run["output_dir"] = str(
                directory if directory.is_absolute() else path.parent / directory
            )
        labels = data.get("labels")
        if labels is not None and not isinstance(labels, list):
            raise ValueError("recipe.data.labels must be an array of class names")
        return cls(
            data_path=data_path,
            fields=ClassificationFields(**data.get("fields", {})),
            labels=None if labels is None else tuple(labels),
            hidden_dim=model.get("hidden_dim", 64),
            batch_size=data.get("batch_size", 64),
            preprocessing=data.get("preprocessing", "standardize"),
            seed=payload.get("seed", 17),
            training=replace(
                ARTITrainingConfig(device="cpu", steps=200, learning_rate=0.003),
                **payload.get("training", {}),
            ),
            run=ARTITrainingRunConfig(**run),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": "classification",
            "seed": self.seed,
            "data": {
                "path": str(self.data_path),
                "fields": asdict(self.fields),
                "labels": None if self.labels is None else list(self.labels),
                "batch_size": self.batch_size,
                "preprocessing": self.preprocessing,
            },
            "model": {"hidden_dim": self.hidden_dim},
            "training": asdict(self.training),
            "run": self.run.to_dict(),
        }


class _FormulaClassifier(nn.Module):
    """A fixed task recipe, not a claim of learned architecture superiority.

    Numerical computation is existing Contract/Add/Broadcast/GELU Formula IR.
    The resource graph is lowered once; it is not interpreted per minibatch.
    """

    def __init__(self, input_dim: int, hidden_dim: int, classes: int) -> None:
        super().__init__()
        self.input_dim, self.hidden_dim, self.classes = input_dim, hidden_dim, classes
        self.operands = nn.ParameterDict(
            {
                "encoder": nn.Parameter(torch.randn(input_dim, hidden_dim) / input_dim**0.5),
                "encoder_bias": nn.Parameter(torch.zeros(hidden_dim)),
                "head": nn.Parameter(torch.randn(hidden_dim, classes) / hidden_dim**0.5),
                "head_bias": nn.Parameter(torch.zeros(classes)),
            }
        )
        self.register_buffer("feature_mean", torch.zeros(input_dim))
        self.register_buffer("feature_scale", torch.ones(input_dim))
        resources = []
        for name, axis, width in (
            ("features", "input_feature", input_dim),
            ("hidden", "hidden_feature", hidden_dim),
            ("logits", "class", classes),
        ):
            view = TensorView.from_tensor(
                torch.zeros(1, width), axis_names=("batch", axis), axis_roles=("batch", "feature")
            )
            resources.append(
                TensorResource(
                    TensorResourceSpec(
                        name,
                        TensorViewPattern(
                            min_rank=2, max_rank=2, allowed_axis_roles=("batch", "feature")
                        ),
                    ),
                    view,
                )
            )
        nodes = []
        for (
            node_id,
            source,
            destination,
            input_axis,
            output_axis,
            width_in,
            width_out,
            weight_name,
            bias_name,
        ) in (
            (
                "encode",
                "features",
                "hidden",
                "input_feature",
                "hidden_feature",
                input_dim,
                hidden_dim,
                "encoder",
                "encoder_bias",
            ),
            (
                "classify",
                "hidden",
                "logits",
                "hidden_feature",
                "class",
                hidden_dim,
                classes,
                "head",
                "head_bias",
            ),
        ):
            x = InputBinding("x", TensorType(("batch", input_axis), ("B", width_in)))
            weight = BankBinding(
                "weight",
                "arti/formula-operand-bank@1",
                weight_name,
                TensorType((input_axis, output_axis), (width_in, width_out)),
            )
            bias = BankBinding(
                "bias",
                "arti/formula-operand-bank@1",
                bias_name,
                TensorType((output_axis,), (width_out,)),
            )
            projected = contract(x, weight, reduce_axes=((input_axis, input_axis),))
            value = add(
                projected,
                broadcast(bias, output_axes=("batch", output_axis), output_sizes=("B", width_out)),
            )
            if node_id == "encode":
                value = scalar_map(value, mode="gelu")
            program = FormulaProgram.build(outputs=(value,))
            nodes.append(
                FormulaProgramNode(
                    node_id,
                    FormulaFabricV2(program),
                    input_ports={"x": ResourcePort(source)},
                    output_ports={"value": ResourcePort(destination)},
                    output_slots={"value": program.outputs[0]},
                    banks={
                        "weight": weight.bind(self.operands[weight_name]),
                        "bias": bias.bind(self.operands[bias_name]),
                    },
                    axis_roles={"value": {"batch": "batch", output_axis: "feature"}},
                )
            )
        self.graph = ProgramGraph(
            resources, (), nodes=nodes, programs={"classify": ("encode", "classify")}
        )
        # The plan shares the registered operands, but is not a second owner of
        # their serialized state. The graph itself remains the portable declaration.
        object.__setattr__(
            self,
            "_plan",
            ResourceGraphCompiler.compile_program_outputs(self.graph, "classify", ("logits",)),
        )

    def forward(self, features: Tensor, labels: Tensor | None = None) -> dict[str, Tensor]:
        features = features.to(device=self.feature_mean.device, dtype=self.feature_mean.dtype)
        normalized = (features - self.feature_mean) / self.feature_scale
        logits = self._plan(
            normalized,
            normalized.new_zeros((features.shape[0], self.hidden_dim)),
            normalized.new_zeros((features.shape[0], self.classes)),
        )[0]
        result = {"logits": logits}
        if labels is not None:
            result["loss"] = F.cross_entropy(
                logits, labels.to(device=logits.device, dtype=torch.long)
            )
        return result


class ClassificationRecipe:
    """One data/task contract from supervised training to a deployable asset."""

    task_name = "classification"

    @property
    def dataset_manifest(self):
        return self.data.manifest(include_ids=True)

    @property
    def validation_data(self):
        return self.data.splits["validation"].batches(self.config.batch_size)

    @staticmethod
    def evaluation_weight(batch):
        return batch["labels"].numel()

    def __init__(self, config: ClassificationRecipeConfig) -> None:
        self.config = config
        self.data = ClassificationDataset(
            config.data_path, fields=config.fields, labels=config.labels
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            self.model = _FormulaClassifier(
                self.data.input_dim, config.hidden_dim, len(self.data.labels)
            )
        mean, scale = self.data.preprocessing(config.preprocessing)
        self.model.feature_mean.copy_(mean)
        self.model.feature_scale.copy_(scale)
        self.task_contract = {
            "task": "classification",
            "format": "arti.classification-recipe",
            "format_version": 1,
            "input_dim": self.data.input_dim,
            "hidden_dim": config.hidden_dim,
            "labels": list(self.data.labels),
            "fields": asdict(config.fields),
            "preprocessing": config.preprocessing,
            "dataset": self.data.manifest(),
            "objective": "cross_entropy",
            "metrics": ["sample_mean_cross_entropy", "accuracy", "macro_f1"],
            "checkpoint_selection": "final",
            "execution": "torch-native-program-graph",
        }
        self.task_fingerprint = json_fingerprint(self.task_contract)
        self.task_contract["sha256"] = self.task_fingerprint
        binding = TrainingBinding(
            self.model,
            kind="classification-recipe",
            save_checkpoint=self._save_checkpoint,
            load_checkpoint=self._load_checkpoint,
            checkpoint_metadata={
                "task_recipe_sha256": self.task_fingerprint,
                "training_config": asdict(config.training),
            },
        )
        self.session = trainer(binding, config=config.training)
        self.train_data = ClassificationBatches(
            self.data.splits["train"], batch_size=config.batch_size, seed=config.seed
        )

    @classmethod
    def from_config(cls, config: str | Path | ClassificationRecipeConfig) -> ClassificationRecipe:
        return cls(
            config
            if isinstance(config, ClassificationRecipeConfig)
            else ClassificationRecipeConfig.read(config)
        )

    def plan(self) -> dict[str, Any]:
        return {
            **self.session.plan(),
            "task_recipe": self.task_contract,
            "batch_size": self.config.batch_size,
            "data_resume": "epoch_shuffle_and_cursor",
            "validation_policy": self.config.run.to_dict()["validation"],
            "deployment": {
                "format": "arti.st",
                "preprocessing_included": True,
                "labels_included": True,
                "sklearn_required": False,
            },
        }

    def check(self) -> dict[str, Any]:
        batch = self.data.splits["train"].batches(self.config.batch_size)[0]
        return {**self.session.check(batch), "task_recipe_sha256": self.task_fingerprint}

    def fit(
        self,
        *,
        steps: int | None = None,
        run: ARTITrainingRunConfig | None = None,
        resume_from_checkpoint: str | Path | None = None,
        reporter=None,
        stop_requested=None,
        validation_fn=None,
        on_checkpoint=None,
    ) -> dict[str, Any]:
        return fit_task_recipe(
            self,
            steps=steps,
            run=run,
            resume_from_checkpoint=resume_from_checkpoint,
            reporter=reporter,
            stop_requested=stop_requested,
            validation_fn=validation_fn,
            on_checkpoint=on_checkpoint,
        )

    def resume_run(
        self,
        source_run: str | Path,
        *,
        run: ARTITrainingRunConfig | None = None,
        output_dir: str | Path | None = None,
        max_seconds: float | None = None,
        refresh_completed: bool = False,
        reporter=None,
        stop_requested=None,
        validation_fn=None,
        on_checkpoint=None,
    ) -> dict[str, Any]:
        """Finish the original task horizon and remaining workflow stages."""
        from .recipe_recovery import resume_task_recipe

        return resume_task_recipe(
            self,
            source_run,
            run=run,
            output_dir=output_dir,
            max_seconds=max_seconds,
            refresh_completed=refresh_completed,
            reporter=reporter,
            stop_requested=stop_requested,
            validation_fn=validation_fn,
            on_checkpoint=on_checkpoint,
        )

    def evaluate(
        self, split: str = "validation", *, output: str | Path | None = None, checkpoint: str | Path | None = None
    ) -> dict[str, Any]:
        if split not in self.data.splits or not len(self.data.splits[split]):
            raise ValueError(f"no labeled records for evaluation split {split!r}")
        data = self.data.splits[split]
        context = self.session.evaluation_context() if checkpoint is None else self.session.checkpoint_context(checkpoint)
        with context:
            logits = _logits(self.model, data.features, self.config.batch_size)
        return _classification_evaluation(
            logits,
            data.labels,
            data.ids,
            self.data.labels,
            split=split,
            dataset_fingerprint=data.fingerprint,
            task_fingerprint=self.task_fingerprint,
            output=output,
        )

    def export(self, directory: str | Path, *, selection: Mapping[str, Any] | None = None) -> Path:
        path = Path(directory) / "model.st"
        save(self.model, path, config={"task_recipe": self.task_contract, **({"selection": dict(selection)} if selection is not None else {})})
        return path

    def load_model(self, path: str | Path) -> None:
        self._load_checkpoint(path, load_checkpoint=False)

    def _save_checkpoint(self, path, *, optimizer=None, scheduler=None, training_state=None):
        return save(
            self.model,
            path,
            config={"task_recipe": self.task_contract},
            optimizer=optimizer,
            scheduler=scheduler,
            training_state=training_state,
        )

    def _load_checkpoint(
        self, path, *, optimizer=None, scheduler=None, map_location=None, load_checkpoint=True
    ):
        path = _model_path(path)
        preview = load(path, load_checkpoint=load_checkpoint)
        contract = preview.manifest["architecture"]["config"].get("task_recipe", {})
        if contract != self.task_contract:
            raise ValueError(
                "model/checkpoint task, labels, preprocessing or dataset does not match this recipe"
            )
        if load_checkpoint:
            state = preview.training_state
            if (
                not isinstance(state, Mapping)
                or state.get("task_recipe_sha256") != self.task_fingerprint
            ):
                raise ValueError("model asset has no matching resumable recipe state")
            previous = state.get("training_config", {})
            current = asdict(self.config.training)
            for key in (
                "engine",
                "objective",
                "learning_rate",
                "gradient_accumulation_steps",
                "mixed_precision",
            ):
                if previous.get(key) != current[key]:
                    raise ValueError(f"resume training.{key} does not match the checkpoint")
            cursor = state.get("data_state")
            if cursor is not None:
                expected = self.train_data.state_dict()
                if any(
                    cursor.get(key) != expected[key]
                    for key in ("seed", "batch_size", "split_sha256")
                ):
                    raise ValueError("resume data/order does not match the checkpoint")
        device = self.model.feature_mean.device if map_location is None else map_location
        return load(
            path,
            model=self.model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
            load_checkpoint=load_checkpoint,
        )


class ClassificationPredictor:
    """Load once and batch raw records, without the original data/config path."""
    serving_state_scope = "read-only"

    def validate_records(self, records) -> None:
        limit = torch.finfo(torch.float32).max
        for row in records:
            values = feature_vector(row.get(self.fields.features), expected_size=self.model.input_dim)
            if any(abs(value) > limit for value in values):
                raise ValueError("features must be representable as finite float32 values")

    def __init__(self, model: _FormulaClassifier, contract: Mapping[str, Any]) -> None:
        self.model = model.eval()
        self.contract = dict(contract)
        self.labels = tuple(contract["labels"])
        self.fields = ClassificationFields(**contract["fields"])

    @classmethod
    def from_pretrained(cls, path: str | Path, *, device: str = "cpu") -> ClassificationPredictor:
        path = _model_path(path)
        from .task_recipe import _manifest
        from .recipe_deployment import local_task_resources
        path, contract, manifest = _manifest(path)
        if (
            contract.get("format") != "arti.classification-recipe"
            or contract.get("format_version") != 1
            or contract.get("task") != "classification"
        ):
            raise ValueError("asset is not a supported classification recipe")
        digest = contract.get("sha256")
        if digest != json_fingerprint(
            {key: value for key, value in contract.items() if key != "sha256"}
        ):
            raise ValueError("task recipe fingerprint does not match its declaration")
        _resources, deployment_sha256 = local_task_resources(path, manifest)
        with torch.random.fork_rng(devices=[]):
            model = _FormulaClassifier(
                contract["input_dim"], contract["hidden_dim"], len(contract["labels"])
            )
        load(path, model=model, map_location=device, load_checkpoint=False)
        result = cls(model, contract)
        result.model_source_sha256 = json_fingerprint(manifest)
        if deployment_sha256 is not None:
            result.deployment_sha256 = deployment_sha256
        return result

    def predict(
        self, features: Sequence[Sequence[float]] | Tensor, *, batch_size: int = 128
    ) -> list[dict[str, Any]]:
        _positive_int(batch_size, "batch_size")
        if isinstance(features, Tensor):
            if features.ndim != 2 or features.shape[1] != self.model.input_dim:
                raise ValueError(f"features must have shape [batch, {self.model.input_dim}]")
            if not bool(torch.isfinite(features).all()):
                raise ValueError("features must be finite")
            values = features
        else:
            values = torch.tensor(
                [feature_vector(row, expected_size=self.model.input_dim) for row in features],
                dtype=torch.float32,
            ).reshape(-1, self.model.input_dim)
            if not bool(torch.isfinite(values).all()):
                raise ValueError("features must be representable as finite float32 values")
        probabilities = _probabilities(self.model, values, batch_size)
        return [
            {
                "label": self.labels[int(row.argmax())],
                "score": float(row.max()),
                "probabilities": row.tolist(),
            }
            for row in probabilities
        ]

    def predict_records(
        self, records: Sequence[Mapping[str, Any]], *, batch_size: int = 128
    ) -> list[dict[str, Any]]:
        results = self.predict(
            [row.get(self.fields.features) for row in records], batch_size=batch_size
        )
        for record, result in zip(records, results, strict=True):
            if self.fields.id in record:
                result["id"] = record[self.fields.id]
        return results

    def evaluate_records(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        batch_size: int = 128,
        output: str | Path | None = None,
    ) -> dict[str, Any]:
        """Evaluate fresh labeled records using saved preprocessing, never refit it."""
        _positive_int(batch_size, "batch_size")
        if not records:
            raise ValueError("evaluation records must not be empty")
        ids, targets, vectors = [], [], []
        lookup = {label: index for index, label in enumerate(self.labels)}
        for row in records:
            identifier, label = row.get(self.fields.id), row.get(self.fields.label)
            if not isinstance(identifier, str) or not identifier:
                raise ValueError("evaluation records require nonempty string ids")
            if not isinstance(label, str) or label not in lookup:
                raise ValueError("evaluation label is absent from the model's label mapping")
            ids.append(identifier)
            targets.append(lookup[label])
            vectors.append(
                feature_vector(row.get(self.fields.features), expected_size=self.model.input_dim)
            )
        if len(set(ids)) != len(ids):
            raise ValueError("evaluation record ids must be unique")
        features = torch.tensor(vectors, dtype=torch.float32)
        if not bool(torch.isfinite(features).all()):
            raise ValueError("features must be representable as finite float32 values")
        fingerprint = json_fingerprint(
            [
                {"id": identifier, "features": vector, "label": self.labels[target]}
                for identifier, vector, target in zip(ids, vectors, targets, strict=True)
            ]
        )
        return _classification_evaluation(
            _logits(self.model, features, batch_size),
            torch.tensor(targets, dtype=torch.long),
            ids,
            self.labels,
            split="evaluation",
            dataset_fingerprint=fingerprint,
            task_fingerprint=self.contract["sha256"],
            output=output,
        )


def _classification_evaluation(
    logits, targets, ids, labels, *, split, dataset_fingerprint, task_fingerprint, output
):
    predictions = logits.softmax(-1)
    chosen = predictions.argmax(-1)
    classes = len(labels)
    matrix = torch.bincount(targets * classes + chosen, minlength=classes * classes).reshape(
        classes, classes
    )
    true_positive = matrix.diag().float()
    denominator = matrix.sum(0) + matrix.sum(1)
    f1 = torch.where(
        denominator > 0, 2 * true_positive / denominator, torch.zeros_like(true_positive)
    )
    report = {
        "split": split,
        "samples": len(ids),
        "dataset_sha256": dataset_fingerprint,
        "loss": float(F.cross_entropy(logits, targets)),
        "accuracy": float((chosen == targets).float().mean()),
        "macro_f1": float(f1.mean()),
        "labels": list(labels),
        "confusion_matrix": matrix.tolist(),
        "task_recipe_sha256": task_fingerprint,
    }
    if output is not None:
        target = Path(output)
        _write_json(target, report)
        _write_jsonl(
            target.with_suffix(".predictions.jsonl"),
            (
                {
                    "id": identifier,
                    "label": labels[int(truth)],
                    "prediction": labels[int(prediction)],
                    "probabilities": probability.tolist(),
                }
                for identifier, truth, prediction, probability in zip(
                    ids, targets, chosen, predictions, strict=True
                )
            ),
        )
    return report


def _probabilities(model: _FormulaClassifier, features: Tensor, batch_size: int) -> Tensor:
    return _logits(model, features, batch_size).softmax(-1)


def _logits(model: _FormulaClassifier, features: Tensor, batch_size: int) -> Tensor:
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        with torch.inference_mode():
            result = [
                model(features=features[start : start + batch_size])["logits"].float().cpu()
                for start in range(0, len(features), batch_size)
            ]
        return torch.cat(result) if result else torch.empty(0, model.classes)
    finally:
        for module, training in modes:
            module.training = training


def _model_path(path: str | Path) -> Path:
    from .recipe_deployment import _artifact
    return _artifact(path)


def _recipe_factory(options):
    reference = options.get("layer_factory")
    kwargs_path = options.get("factory_kwargs_json")
    if kwargs_path is not None:
        if reference is None:
            raise ValueError("factory kwargs require an explicit --layer-factory")
        from .recipe_adapters import LayerFactoryConfig

        return LayerFactoryConfig(reference, json.loads(Path(kwargs_path).read_text(encoding="utf-8")))
    return reference


def _task_factory(options):
    reference = options.get("task_factory")
    kwargs_path = options.get("task_factory_kwargs_json")
    if kwargs_path is not None:
        if reference is None:
            raise ValueError("task factory kwargs require --task-factory")
        from .python_factory import PythonFactoryConfig
        return PythonFactoryConfig(reference, json.loads(Path(kwargs_path).read_text(encoding="utf-8")))
    return reference


def _prediction_files(options):
    from .recipe_files import PredictionFilesConfig
    from .python_factory import PythonFactoryConfig

    if not any(options.get(name) is not None for name in ("output_files_field", "file_validator", "file_validator_id", "file_validator_kwargs_json")):
        return None
    kwargs = options.get("file_validator_kwargs_json")
    if kwargs is not None and options.get("file_validator") is None:
        raise ValueError("file validator kwargs require a validator")
    validator = None if options.get("file_validator") is None else PythonFactoryConfig(options["file_validator"], {} if kwargs is None else json.loads(Path(kwargs).read_text(encoding="utf-8")))
    return PredictionFilesConfig(options.get("output_files_field") or "files", validator, options.get("file_validator_id"))


def _prediction_episodes(options):
    path = options.get("episode_config")
    if path is None:
        return None
    from .recipe_episode import EpisodeRunConfig

    return EpisodeRunConfig.read(path)


def _evaluation_external(options, *, comparison=False):
    path = options.get("external_json")
    if path is None:
        return {}
    path = Path(path).resolve()
    values = json.loads(path.read_text(encoding="utf-8"))
    def local(value):
        file = Path(value)
        return file if file.is_absolute() else path.parent / file
    return ({arm: {name: local(file) for name, file in mapping.items()} for arm, mapping in values.items()}
            if comparison else {name: local(file) for name, file in values.items()})


def recipe_cli_report(action: str, path: str | Path, **options) -> dict[str, Any]:
    if action in {"export-results", "inspect-results"}:
        from .recipe_export import export_prediction_results, inspect_result_export

        if action == "inspect-results":
            return inspect_result_export(path)

        def mappings(name):
            file = options.get(name)
            if file is None:
                return None
            file = Path(file).resolve()
            return {
                key: str(Path(value) if Path(value).is_absolute() else file.parent / value)
                for key, value in json.loads(file.read_text(encoding="utf-8")).items()
            }

        expected = options.get("expected_ids_json")
        return export_prediction_results(
            path,
            options["export_config"],
            options["output"],
            expected_ids=None
            if expected is None
            else json.loads(Path(expected).read_text(encoding="utf-8")),
            evidence=mappings("evidence_json"),
            attachments=mappings("attachments_json"),
            deployment=options.get("deployment"),
        )
    if action in {"inspect-runtime", "run-runtime"}:
        from .recipe_runtime import inspect_task_runtime, run_task_runtime
        if action == "inspect-runtime":
            return inspect_task_runtime(path, python=options.get("python"), output=options.get("output"), inspect_backend=options.get("inspect_backend", False))
        def binding(name):
            file = Path(options[name]).resolve()
            values = json.loads(file.read_text(encoding="utf-8"))
            return {key: str(Path(value) if Path(value).is_absolute() else file.parent / value) for key, value in values.items()}
        kwargs = options.get("options_json")
        return run_task_runtime(path, inputs=binding("inputs_json"), outputs=binding("outputs_json"),
                                python=options.get("python"), options=None if kwargs is None else json.loads(Path(kwargs).read_text(encoding="utf-8")), output=options["output"])
    if action in {"score-suite", "compare-suites", "import-evaluation"}:
        from .recipe_suite import evaluate_prediction_suite, compare_prediction_suites, import_evaluation_evidence

        common = {"allow_partial": options.get("allow_partial", False), "output": options["output"]}
        if action == "import-evaluation":
            report = import_evaluation_evidence(path, options["suite"], options["view"], options["evidence"], **common)
            return {key: value for key, value in report.items() if key not in {"scores", "evidence"}}
        external = _evaluation_external(options, comparison=action == "compare-suites")
        common.update(references=options.get("references"), external=external)
        if action == "compare-suites":
            report = compare_prediction_suites(path, options["baseline"], options["suite"], pairing=options.get("pairing", "exact"), **common)
            report = {**report, "arms": {name: {**arm, "views": {view: {key: value for key, value in result.items() if key not in {"scores", "outcomes"}} for view, result in arm["views"].items()}} for name, arm in report["arms"].items()}}
        else:
            report = evaluate_prediction_suite(path, options["suite"], **common)
            report = {**report, "views": {name: {key: value for key, value in view.items() if key not in {"scores", "outcomes"}} for name, view in report["views"].items()}}
        return report
    if action == "validate-deployment":
        from .recipe_prediction import PredictionRunConfig
        from .recipe_scoring import PredictionScoringConfig
        from .python_factory import PythonFactoryConfig
        from .recipe_validation import validate_recipe_deployment

        def values(name, default):
            value = options.get(name)
            return default if value is None else json.loads(Path(value).read_text(encoding="utf-8"))

        if options.get("evaluation_suite") is not None:
            from .recipe_suite import EvaluationSuiteConfig
            scoring = EvaluationSuiteConfig.read(options["evaluation_suite"])
            if any(options.get(name) is not None for name in ("scorer_id", "scorer_kwargs_json", "target_id_field")):
                raise ValueError("suite scoring declarations belong in the suite configuration")
        else:
            scoring = PredictionScoringConfig(
                PythonFactoryConfig(options["scorer"], values("scorer_kwargs_json", {})),
                options.get("scorer_id"), options.get("target_id_field"),
            )
            if options.get("external_json") is not None:
                raise ValueError("external evidence requires an evaluation suite")
        run = PredictionRunConfig(
            options["run_dir"], batch_size=options.get("batch_size"),
            predict_kwargs=values("predict_kwargs_json", {}), id_field=options.get("id_field"),
            output_id_field=options.get("output_id_field"), max_seconds=options.get("max_seconds"),
            files=_prediction_files(options),
            episodes=_prediction_episodes(options),
        )
        return validate_recipe_deployment(
            path, options.get("data"), run, scoring, criteria=values("criteria_json", None),
            targets=options.get("targets"), resume=options.get("resume", False),
            output=options.get("output"), device=options.get("device") or "cpu",
            task_factory=_task_factory(options), layer_factory=_recipe_factory(options),
            base_model=options.get("base_model"),
            external=_evaluation_external(options) if options.get("evaluation_suite") is not None else None,
        )
    if action in {"score-prediction", "compare-predictions"}:
        from .python_factory import PythonFactoryConfig
        from .recipe_scoring import (
            PredictionScoringConfig,
            score_prediction_run,
            compare_prediction_runs,
        )

        kwargs_path = options.get("scorer_kwargs_json")
        kwargs = {} if kwargs_path is None else json.loads(
            Path(kwargs_path).read_text(encoding="utf-8")
        )
        scoring = PredictionScoringConfig(
            PythonFactoryConfig(options["scorer"], kwargs), options["scorer_id"],
            options.get("target_id_field"),
        )
        common = {
            "targets": options.get("targets"),
            "allow_partial": options.get("allow_partial", False),
            "output": options.get("output"),
        }
        if action == "score-prediction":
            result = score_prediction_run(path, scoring, **common)
        else:
            result = compare_prediction_runs(
                path, options["baseline"], scoring,
                pairing=options.get("pairing", "exact"), **common,
            )
        return {key: value for key, value in result.items() if key != "scores"}
    if action == "inspect-prediction":
        from .recipe_prediction import inspect_prediction

        return inspect_prediction(path)
    if action == "predict-run":
        from .recipe_prediction import CausalLMPredictionRunConfig, PredictionRunConfig, run_prediction, _model_declaration, _read_plan
        kwargs_path = options.get("predict_kwargs_json")
        prediction_kwargs = {} if kwargs_path is None else json.loads(Path(kwargs_path).read_text(encoding="utf-8"))
        files = _prediction_files(options)
        episodes = _prediction_episodes(options)
        if options.get("resume"):
            legacy = _read_plan(Path(options["run_dir"]))["format"] == "arti.causal-lm-prediction-run"
        else:
            _asset, _manifest, declaration = _model_declaration(path)
            legacy = declaration["format"] == "arti.causal-lm-recipe" and kwargs_path is None and files is None and episodes is None
        if legacy:
            run = CausalLMPredictionRunConfig(options["run_dir"], batch_size=options.get("batch_size"), max_new_tokens=options.get("max_new_tokens"), max_seconds=options.get("max_seconds"))
            if prediction_kwargs or options.get("id_field") is not None or options.get("output_id_field") is not None or files is not None or episodes is not None:
                raise ValueError("generic task prediction options do not apply to an existing causal-LM run")
        else:
            if options.get("max_new_tokens") is not None:
                prediction_kwargs["max_new_tokens"] = options["max_new_tokens"]
            run = PredictionRunConfig(options["run_dir"], batch_size=options.get("batch_size"), predict_kwargs=prediction_kwargs, id_field=options.get("id_field"), output_id_field=options.get("output_id_field"), max_seconds=options.get("max_seconds"), files=files, episodes=episodes)
        return run_prediction(path, options.get("data"), run, resume=options.get("resume", False), device=options.get("device") or "cpu", base_model=options.get("base_model"), layer_factory=_recipe_factory(options), task_factory=_task_factory(options))
    if action == "import-data":
        from .recipe_import import import_causal_lm_data

        return import_causal_lm_data(path, options["output"])
    if action in {"snapshot-data", "inspect-data", "prepare-assets"}:
        from .recipe_assets import create_dataset_snapshot, inspect_dataset_snapshot, prepare_dataset_assets
        from .serialization import _atomic_json

        if action == "snapshot-data":
            return {"ok": True, **create_dataset_snapshot(path, options["output"]).manifest()}
        if action == "inspect-data":
            return {"ok": True, **inspect_dataset_snapshot(path, output=options.get("output"))}
        preparation = prepare_dataset_assets(path, options["cache_dir"], options["transform"], roles=options.get("role"), batch_size=options.get("batch_size", 16))
        report = {"ok": True, **preparation.report, "dataset": preparation.manifest()}
        if options.get("output") is not None:
            output = Path(options["output"])
            if output.resolve().is_relative_to(Path(path).resolve()) or output.resolve().is_relative_to(Path(options["cache_dir"]).resolve() / "entries"):
                raise ValueError("write preparation reports outside immutable input/cache entries")
            output.parent.mkdir(parents=True, exist_ok=True)
            _atomic_json(output, report)
        return report
    if action == "compare-evaluations":
        from .recipe_evaluation import compare_causal_lm_evaluations

        result = compare_causal_lm_evaluations(path, options["baseline"], comparison=options.get("comparison"), pairing=options.get("pairing", "exact"), output=options.get("output"))
        return {key: value for key, value in result.items() if key != "predictions"}
    if action == "inspect-deployment":
        from .recipe_deployment import inspect_recipe_deployment

        return inspect_recipe_deployment(path, base_model=options.get("base_model"), output=options.get("output"))
    if action == "package-deployment":
        from .recipe_deployment import package_recipe_deployment

        resources = options.get("resources_json")
        requirements = options.get("requirements_json")
        def code_bindings(name):
            source = options.get(name)
            if source is None:
                return None
            source = Path(source).resolve()
            return {key: str(Path(value) if Path(value).is_absolute() else source.parent / value) for key, value in json.loads(source.read_text(encoding="utf-8")).items()}
        return package_recipe_deployment(path, options["output"], include_base=options.get("include_base", False), base_model=options.get("base_model"), resources=None if resources is None else json.loads(Path(resources).read_text(encoding="utf-8")), requirements=None if requirements is None else json.loads(Path(requirements).read_text(encoding="utf-8")), runtime=options.get("runtime_profile"), code=code_bindings("code_json"), wheels=code_bindings("wheels_json"))
    if action == "prepare-data":
        from .language_recipe import prepare_causal_lm_data

        return prepare_causal_lm_data(path, output=options.get("output"))
    if action == "resume-run":
        from .recipe_recovery import plan_recipe_resume, resume_recipe_run

        resources = {name: options.get(option) for name, option in (("data_path", "data"), ("cache_dir", "cache_dir"), ("source_manifest", "source_manifest"), ("base_model", "base_model"))}
        if options.get("resources_json") is not None:
            resources["resources"] = json.loads(Path(options["resources_json"]).read_text(encoding="utf-8"))
        if options.get("plan_only"):
            plan = plan_recipe_resume(path, refresh_completed=options.get("refresh_completed", False), **resources)
            if options.get("output") is not None:
                _write_json(options["output"], plan)
            return plan
        result = resume_recipe_run(path, output_dir=options.get("run_dir"), max_seconds=options.get("max_seconds"), refresh_completed=options.get("refresh_completed", False), layer_factory=_recipe_factory(options), **resources)
        if options.get("output") is not None:
            _write_json(options["output"], result)
        return result
    if action == "plan" and read_recipe_config(path).get("task") == "causal-lm" and read_recipe_config(path).get("format") != "arti.task-recipe-config":
        from .language_recipe import plan_causal_lm_recipe
        report = plan_causal_lm_recipe(path)
        if options.get("output") is not None:
            _write_json(options["output"], report)
        return report
    if action == "evaluate" and options.get("data") is not None:
        predictor = load_recipe_predictor(path, device=options.get("device") or "cpu", layer_factory=_recipe_factory(options), task_factory=_task_factory(options), **({"base_model": options["base_model"]} if options.get("base_model") is not None else {}))
        kwargs = {"output": options.get("output")}
        if predictor.contract.get("task") == "causal-lm" and predictor.contract.get("format") != "arti.task-recipe":
            kwargs["generate_samples"] = options.get("generate_samples")
        report = predictor.evaluate_records(list(read_jsonl(options["data"])), **kwargs)
        return {"ok": True, "kind": "recipe-evaluate", **report}
    if action == "predict":
        predictor = load_recipe_predictor(path, device=options.get("device") or "cpu", layer_factory=_recipe_factory(options), task_factory=_task_factory(options), **({"base_model": options["base_model"]} if options.get("base_model") is not None else {}))
        task = predictor.contract["task"]
        kwargs = {"batch_size": options.get("batch_size")}
        if task == "causal-lm" and predictor.contract.get("format") != "arti.task-recipe":
            kwargs["max_new_tokens"] = options.get("max_new_tokens")
        elif predictor.contract.get("format") != "arti.task-recipe":
            kwargs["batch_size"] = options.get("batch_size") or 128
        predictions = predictor.predict_records(list(read_jsonl(options["data"])), **kwargs)
        _write_jsonl(options["output"], predictions)
        report = {
            "ok": True,
            "kind": "recipe-predict",
            "samples": len(predictions),
            "task": predictor.contract["task"],
            "output": str(options["output"]),
        }
        if task == "classification" and hasattr(predictor, "labels"):
            report["labels"] = list(predictor.labels)
        return report
    if action == "evaluate" and read_recipe_config(path).get("task") == "causal-lm" and read_recipe_config(path).get("format") != "arti.task-recipe-config":
        from .language_recipe import evaluate_causal_lm_deployment

        return evaluate_causal_lm_deployment(
            options["model"], path, split=options.get("split") or "validation",
            output=options.get("output"), generate_samples=options.get("generate_samples"),
            device=options.get("device") or "cpu", base_model=options.get("base_model"),
            layer_factory=_recipe_factory(options),
        )
    else:
        if action == "evaluate" and _recipe_factory(options) is not None:
            raise ValueError("layer_factory applies only to causal-LM deployments")
        recipe = recipe_from_config(path)
    if action == "plan":
        report = {"ok": True, "kind": "recipe-plan", **recipe.plan()}
    elif action == "check":
        report = {"kind": "recipe-check", **recipe.check()}
    elif action == "train":
        run = recipe.config.run
        if options.get("run_dir") is not None:
            run = replace(run, output_dir=options["run_dir"])
        return recipe.fit(
            steps=options.get("steps"),
            run=run,
            resume_from_checkpoint=options.get("resume_from_checkpoint"),
        )
    elif action == "evaluate":
        recipe.load_model(options["model"])
        kwargs = {"output": options.get("output")}
        if recipe.task_name == "causal-lm" and recipe.task_contract.get("format") != "arti.task-recipe":
            kwargs["generate_samples"] = options.get("generate_samples")
        return {
            "ok": True,
            "kind": "recipe-evaluate",
            **recipe.evaluate(options.get("split") or "validation", **kwargs),
        }
    elif action == "export":
        recipe.load_model(options["checkpoint"])
        exported = recipe.export(options["output"])
        return {"ok": True, "kind": "recipe-export", "model": str(exported)}
    else:
        raise ValueError(f"unknown recipe action {action!r}")
    if options.get("output") is not None:
        _write_json(options["output"], report)
    return report


def recipe_from_config(path):
    payload = read_recipe_config(path)
    if payload.get("format") == "arti.task-recipe-config":
        from .task_recipe import TaskRecipe
        return TaskRecipe.from_config(path)
    task = payload.get("task")
    if task == "classification":
        return ClassificationRecipe.from_config(path)
    if task == "causal-lm":
        from .language_recipe import CausalLMRecipe

        return CausalLMRecipe.from_config(path)
    raise ValueError(f"unsupported recipe task {task!r}")


def load_recipe_predictor(path, *, device="cpu", task_factory=None, **kwargs):
    from .recipe_deployment import _artifact
    path = _artifact(path)
    from .task_recipe import _manifest
    _path, contract, _metadata = _manifest(path)
    if contract.get("format") == "arti.task-recipe":
        from .task_recipe import TaskPredictor
        if any(value is not None for value in kwargs.values()):
            raise ValueError("base_model/layer_factory options belong to causal-LM deployments")
        return TaskPredictor.from_pretrained(path, device=device, task_factory=task_factory)
    if task_factory is not None:
        raise ValueError("task_factory applies only to generic task deployments")
    task = contract.get("task")
    if task == "classification":
        if kwargs.get("base_model") is not None:
            raise ValueError("base_model applies only to causal-LM deployments")
        if kwargs.get("layer_factory") is not None:
            raise ValueError("layer_factory applies only to causal-LM deployments")
        return ClassificationPredictor.from_pretrained(path, device=device)
    if task == "causal-lm":
        from .language_recipe import CausalLMPredictor

        return CausalLMPredictor.from_pretrained(path, device=device, **kwargs)
    raise ValueError(f"unsupported inference recipe task {task!r}")


__all__ = [
    "ClassificationRecipeConfig",
    "ClassificationRecipe",
    "ClassificationPredictor",
    "recipe_from_config",
    "load_recipe_predictor",
]
