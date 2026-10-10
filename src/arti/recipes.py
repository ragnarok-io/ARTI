"""Executable supervised task recipes using ARTI's shared training service."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
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
        if self.training.engine != "torch" or self.training.distributed:
            raise ValueError(
                "the local classification recipe currently supports the single-process Torch engine"
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
    ) -> dict[str, Any]:
        return fit_task_recipe(
            self,
            steps=steps,
            run=run,
            resume_from_checkpoint=resume_from_checkpoint,
            reporter=reporter,
        )

    def evaluate(
        self, split: str = "validation", *, output: str | Path | None = None
    ) -> dict[str, Any]:
        if split not in self.data.splits or not len(self.data.splits[split]):
            raise ValueError(f"no labeled records for evaluation split {split!r}")
        data = self.data.splits[split]
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

    def export(self, directory: str | Path) -> Path:
        path = Path(directory) / "model.st"
        save(self.model, path, config={"task_recipe": self.task_contract})
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

    def __init__(self, model: _FormulaClassifier, contract: Mapping[str, Any]) -> None:
        self.model = model.eval()
        self.contract = dict(contract)
        self.labels = tuple(contract["labels"])
        self.fields = ClassificationFields(**contract["fields"])

    @classmethod
    def from_pretrained(cls, path: str | Path, *, device: str = "cpu") -> ClassificationPredictor:
        path = _model_path(path)
        artifact = load(path, load_checkpoint=False)
        contract = artifact.manifest["architecture"]["config"].get("task_recipe", {})
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
        with torch.random.fork_rng(devices=[]):
            model = _FormulaClassifier(
                contract["input_dim"], contract["hidden_dim"], len(contract["labels"])
            )
        load(path, model=model, map_location=device, load_checkpoint=False)
        return cls(model, contract)

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
    path = Path(path)
    return path / "model.st" if path.is_dir() else path


def recipe_cli_report(action: str, path: str | Path, **options) -> dict[str, Any]:
    if action == "plan" and read_recipe_config(path).get("task") == "causal-lm":
        from .language_recipe import plan_causal_lm_recipe
        report = plan_causal_lm_recipe(path)
        if options.get("output") is not None:
            _write_json(options["output"], report)
        return report
    if action == "evaluate" and options.get("data") is not None:
        predictor = load_recipe_predictor(path, device=options.get("device") or "cpu")
        kwargs = {"output": options.get("output")}
        if getattr(predictor, "contract", {}).get("task") == "causal-lm":
            kwargs["generate_samples"] = options.get("generate_samples") or 0
        report = predictor.evaluate_records(list(read_jsonl(options["data"])), **kwargs)
        return {"ok": True, "kind": "recipe-evaluate", **report}
    if action == "predict":
        predictor = load_recipe_predictor(path, device=options.get("device") or "cpu")
        task = predictor.contract["task"]
        kwargs = {"batch_size": options.get("batch_size")}
        if task == "causal-lm":
            kwargs["max_new_tokens"] = options.get("max_new_tokens")
        else:
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
        if task == "classification":
            report["labels"] = list(predictor.labels)
        return report
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
        if recipe.task_name == "causal-lm":
            kwargs["generate_samples"] = options.get("generate_samples") or 0
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
    task = read_recipe_config(path).get("task")
    if task == "classification":
        return ClassificationRecipe.from_config(path)
    if task == "causal-lm":
        from .language_recipe import CausalLMRecipe

        return CausalLMRecipe.from_config(path)
    raise ValueError(f"unsupported recipe task {task!r}")


def load_recipe_predictor(path, *, device="cpu", **kwargs):
    path = Path(path)
    if path.is_dir():
        from .attachment_hub import DEFAULT_ARTIFACT

        path = path / (DEFAULT_ARTIFACT if (path / DEFAULT_ARTIFACT).exists() else "model.st")
    task = (
        load(path, load_checkpoint=False)
        .manifest["architecture"]["config"]
        .get("task_recipe", {})
        .get("task")
    )
    if task == "classification":
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
