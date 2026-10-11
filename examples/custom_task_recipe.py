"""A native tabular task, not a language-model adapter.

Run: python examples/custom_task_recipe.py --output outputs/diabetes --steps 30
The bundled scikit-learn dataset is needed only to create the example inputs.
The installed factory, PyTorch and ARTI suffice for deployed inference.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

import arti
from arti import mechanisms as m
from arti.recipe_data import json_fingerprint, read_jsonl
from arti.recipe_workflow import write_json, write_jsonl


FACTORY = "examples.custom_task_recipe:build_task"


def _program(dim):
    pattern = m.TensorViewPattern(min_rank=2, max_rank=2, allowed_axis_roles=("batch", "feature"))
    resources = [
        m.TensorResource(
            m.TensorResourceSpec(name, pattern),
            m.TensorView.from_tensor(
                torch.zeros(1, dim),
                axis_names=("batch", "feature"),
                axis_roles=("batch", "feature"),
            ),
        )
        for name in ("inputs", "outputs")
    ]
    graph = m.ProgramGraph(
        resources,
        (
            m.Connection(
                "adapt",
                m.ResourcePort("inputs"),
                m.ResourcePort("outputs"),
                transfer=m.LearnableAffineTransfer(gain=1.0),
            ),
        ),
        programs={"task": ("adapt",)},
    )
    return arti.ARTILayer(
        graph=graph,
        graph_program_id="task",
        graph_input_resource_id="inputs",
        graph_output_resource_id="outputs",
        axis_names=("batch", "feature"),
        axis_roles=("batch", "feature"),
    )


class ExampleModel(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.kind = options.get("kind", "regression")
        self.input_shape = tuple(options["input_shape"])
        dim = math.prod(self.input_shape)
        self.program = _program(dim)
        self.dropout = nn.Dropout(options.get("dropout", 0.1))
        self.output = nn.Linear(dim, 1 if self.kind == "regression" else options["classes"])
        self.register_buffer("feature_mean", torch.zeros(dim))
        self.register_buffer("feature_scale", torch.ones(dim))
        self.register_buffer("target_scale", torch.ones(()))

    def forward(self, features):
        features = (features.flatten(1) - self.feature_mean) / self.feature_scale
        return self.output(self.dropout(self.program(features)))


def _features(model, records):
    values = torch.tensor(
        [row["features"] for row in records], dtype=torch.float32, device=model.feature_mean.device
    )
    if tuple(values.shape[1:]) != model.input_shape:
        raise ValueError("record features differ from the declared input shape")
    return values


def _predict(model, records, *, batch_size=128):
    if type(batch_size) is not int or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")
    for start in range(0, len(records), batch_size):
        rows = records[start : start + batch_size]
        value = model(_features(model, rows))
        predictions = (
            (value[:, 0] * model.target_scale).tolist()
            if model.kind == "regression"
            else value.argmax(-1).tolist()
        )
        for row, prediction in zip(rows, predictions, strict=True):
            yield {"id": row["id"], "prediction": prediction}


def _metrics(model, records):
    if not records:
        raise ValueError("evaluation requires labeled records")
    logits = model(_features(model, records))
    target = torch.tensor([row["target"] for row in records], device=logits.device)
    if model.kind == "regression":
        prediction = logits[:, 0] * model.target_scale
        return {
            "loss": float(F.mse_loss(logits[:, 0], target.float() / model.target_scale)),
            "mse": float(F.mse_loss(prediction, target.float())),
            "mae": float((prediction - target).abs().mean()),
            "samples": len(records),
        }
    return {
        "loss": float(F.cross_entropy(logits, target.long())),
        "accuracy": float((logits.argmax(-1) == target).float().mean()),
        "samples": len(records),
    }


def _objective(model, batch):
    value = model(batch["features"])
    if model.kind == "regression":
        return F.mse_loss(value[:, 0], batch["target"] / model.target_scale)
    return F.cross_entropy(value, batch["target"].long())


def _validate(records):
    for row in records:
        if not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("prediction records require their original string id")
        if "features" not in row:
            raise ValueError("prediction records require features")


def build_task(config, *, mode, contract=None):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.options.get("seed", 17))
        model = ExampleModel(config.options)
    source = model
    if config.options.get("use_binding", False):
        model.output.requires_grad_(False)
        source = arti.TrainingBinding(
            model, parameters=model.program.parameters, state_module=model
        )
    if mode == "predict":
        return arti.TaskPredictor(
            source, contract, predict_fn=_predict, evaluate_fn=_metrics, validate_records=_validate
        )
    records = list(read_jsonl(config.resources["data"]))
    splits = {
        name: [row for row in records if row["split"] == name]
        for name in ("train", "validation", "test")
    }
    if not splits["train"] or not splits["validation"]:
        raise ValueError("explicit train and validation records are required")
    train = _features(model, splits["train"]).flatten(1)
    model.feature_mean.copy_(train.mean(0))
    model.feature_scale.copy_(train.std(0, unbiased=False).clamp_min(1e-6))
    if model.kind == "regression":
        scale = torch.tensor([row["target"] for row in splits["train"]]).float().std(unbiased=False)
        model.target_scale.copy_(scale.clamp_min(1.0))
    batch_size = config.options.get("batch_size", 16)

    def batches(rows):
        return [
            {
                "features": _features(model, rows[start : start + batch_size]),
                "target": torch.tensor(
                    [row["target"] for row in rows[start : start + batch_size]], dtype=torch.float32
                ),
            }
            for start in range(0, len(rows), batch_size)
        ]

    optimizer = torch.optim.AdamW(
        (source.parameters() if isinstance(source, arti.TrainingBinding) else model.parameters()),
        lr=config.training.learning_rate,
        weight_decay=0.01,
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.9)
    return arti.TaskRecipe(
        config,
        source,
        train_data=batches(splits["train"]),
        validation_data=batches(splits["validation"]),
        dataset_manifest={
            "sha256": json_fingerprint(records),
            "splits": {
                name: {"samples": len(rows), "ids": [row["id"] for row in rows]}
                for name, rows in splits.items()
            },
        },
        objective=_objective,
        objective_id="scaled-mse" if model.kind == "regression" else "cross-entropy",
        evaluate_fn=lambda host, split: _metrics(host, splits[split]),
        predict_fn=_predict,
        metadata={
            "input_shape": list(model.input_shape),
            "target_field": "target",
            "preprocessing": "train-only standardization in model buffers",
            "target_units": "original values in predictions; scale-normalized regression training",
        },
        optimizer=optimizer,
        scheduler=scheduler,
        training_weight=lambda batch: len(batch["target"]),
        evaluation_weight=lambda batch: len(batch["target"]),
        validate_records=_validate,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--engine", choices=("torch", "accelerate"), default="torch")
    args = parser.parse_args()
    from sklearn.datasets import load_diabetes

    data = load_diabetes()
    order = torch.randperm(len(data.target), generator=torch.Generator().manual_seed(17)).tolist()
    records = [
        {
            "id": str(index),
            "features": data.data[index].tolist(),
            "target": float(data.target[index]),
            "split": "train" if position < 300 else "validation" if position < 371 else "test",
        }
        for position, index in enumerate(order)
    ]
    write_jsonl(args.output / "data.jsonl", records)
    config = arti.TaskRecipeConfig(
        "diabetes-regression",
        FACTORY,
        options={"input_shape": [10], "kind": "regression", "batch_size": 16},
        resources={"data": str((args.output / "data.jsonl").resolve())},
        training=arti.ARTITrainingConfig(
            device="cpu", steps=args.steps, learning_rate=0.01, engine=args.engine
        ),
    )
    write_json(args.output / "recipe.json", config.to_dict())
    recipe = arti.TaskRecipe.from_config(config)
    before = recipe.evaluate("validation")
    result = recipe.fit(
        run=arti.ARTITrainingRunConfig(output_dir=args.output / "run", checkpoint_steps=5)
    )
    predictor = arti.TaskPredictor.from_pretrained(result["model"], task_factory=FACTORY)
    test = [row for row in records if row["split"] == "test"]
    write_jsonl(args.output / "test.jsonl", test)
    evaluation = predictor.evaluate_records(test, output=args.output / "evaluation.json")
    prediction_run = arti.run_prediction(
        result["model"],
        args.output / "test.jsonl",
        arti.PredictionRunConfig(
            args.output / "prediction-run", batch_size=32, predict_kwargs={"batch_size": 16}
        ),
        task_factory=FACTORY,
    )
    offline = arti.score_prediction_run(
        args.output / "prediction-run",
        arti.PredictionScoringConfig(
            "examples.prediction_scorers:regression", "original-unit-mse-mae-v1"
        ),
        output=args.output / "prediction-scores.json",
    )
    print(
        json.dumps(
            {
                "ok": True,
                "before": before,
                "after": result["validation"],
                "held_out": evaluation,
                "model": result["model"],
                "run_directory": result["run_directory"],
                "predictions": prediction_run["predictions"],
                "offline_prediction_metrics": offline["metrics"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
