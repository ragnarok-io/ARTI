"""Application-owned model, binary codec, objective and result callbacks.

Replace these declarations with your own task semantics. No repository example
module or training dataset is needed by the packaged predict-mode factory.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import struct

import torch
from torch import nn
from torch.nn import functional as F

import arti
from arti import mechanisms as m
from arti.recipe_data import json_fingerprint


FACTORY = "task_app:build_task"


def decode_vectors(requests, *, options):
    metadata = []
    for request in requests:
        count = math.prod(request.metadata["shape"])
        values = struct.unpack(f"<{count}f", request.source.read_bytes())
        tensor = torch.tensor(values).reshape(request.metadata["shape"]) * options["scale"]
        (request.directory / "vector.json").write_text(
            json.dumps(tensor.tolist()), encoding="utf-8"
        )
        metadata.append({"shape": request.metadata["shape"], "dtype": "float32"})
    return metadata


class Model(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = tuple(shape)
        dim = math.prod(shape)
        pattern = m.TensorViewPattern(
            min_rank=2, max_rank=2, allowed_axis_roles=("batch", "feature")
        )
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
        self.program = arti.ARTILayer(
            graph=graph,
            graph_program_id="task",
            graph_input_resource_id="inputs",
            graph_output_resource_id="outputs",
            axis_names=("batch", "feature"),
            axis_roles=("batch", "feature"),
        )
        self.output = nn.Linear(dim, 1)
        self.dropout = nn.Dropout(0.1)
        self.register_buffer("feature_mean", torch.zeros(dim))
        self.register_buffer("feature_scale", torch.ones(dim))

    def forward(self, features):
        values = (features.flatten(1) - self.feature_mean) / self.feature_scale
        return self.output(self.dropout(self.program(values)))[:, 0]


def objective(model, batch):
    return F.mse_loss(model(batch["features"]), batch["target"])


def build_task(config, *, mode, contract=None):
    snapshot = arti.DatasetSnapshot(config.resources["snapshot"])
    transform = arti.AssetTransformConfig(
        "task_app:decode_vectors", "vectors-v1", config.options["transform"]
    )
    prepared = arti.prepare_dataset_assets(
        snapshot, config.resources["cache"], transform, batch_size=3
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.options["seed"])
        model = Model(config.options["input_shape"])

    def native_rows(records):
        result = []
        for row in records:
            refs = snapshot.references(row["id"], role="input")
            if len(refs) != 1:
                raise ValueError("this task requires one input vector per record")
            values = json.loads(
                prepared.files(refs[0]["asset_id"])["vector.json"].read_text(encoding="utf-8")
            )
            result.append({**row, "features": values})
        return result

    def features(rows):
        values = torch.tensor(
            [row["features"] for row in rows], dtype=torch.float32, device=model.feature_mean.device
        )
        if tuple(values.shape[1:]) != model.shape:
            raise ValueError("features differ from the application shape")
        return values

    def metrics(host, rows):
        prediction = host(features(rows))
        target = torch.tensor([row["target"] for row in rows], device=prediction.device)
        return {"loss": float(F.mse_loss(prediction, target)), "samples": len(rows)}

    def predict(host, records, *, output_dir=None):
        prediction = host(features(native_rows(records))).tolist()
        result = []
        for row, value in zip(records, prediction, strict=True):
            output = {"id": row["id"], "prediction": value}
            if output_dir is not None:
                relative = f"{json_fingerprint(row['id'])}/value.bin"
                path = Path(output_dir) / relative
                path.parent.mkdir(parents=True)
                path.write_bytes(struct.pack("<d", value))
                output["files"] = {
                    "value": {"file": relative, "metadata": {"encoding": "little-endian-float64"}}
                }
            result.append(output)
        return result

    if mode == "predict":
        return arti.TaskPredictor(
            model,
            contract,
            predict_fn=predict,
            evaluate_fn=lambda host, rows: metrics(host, native_rows(rows)),
        )
    splits = {
        name: native_rows(list(snapshot.records(name))) for name in ("train", "validation", "test")
    }
    training_features = features(splits["train"]).flatten(1)
    model.feature_mean.copy_(training_features.mean(0))
    model.feature_scale.copy_(training_features.std(0, unbiased=False).clamp_min(1e-6))

    def batches(rows):
        size = config.options["batch_size"]
        return [
            {
                "features": features(rows[start : start + size]),
                "target": torch.tensor(
                    [row["target"] for row in rows[start : start + size]], dtype=torch.float32
                ),
            }
            for start in range(0, len(rows), size)
        ]

    source, preparation = snapshot.manifest(), prepared.manifest()
    return arti.TaskRecipe(
        config,
        model,
        train_data=batches(splits["train"]),
        validation_data=batches(splits["validation"]),
        dataset_manifest={
            "sha256": json_fingerprint({"source": source, "preparation": preparation}),
            "source": source,
            "preparation": preparation,
            "splits": source["splits"],
        },
        data_preparation=prepared.report,
        objective=objective,
        objective_id="native-mse",
        evaluate_fn=lambda host, split: metrics(host, splits[split]),
        predict_fn=predict,
        metadata={
            "input_shape": list(model.shape),
            "preprocessing": "train-only buffers",
            "target_units": "original",
        },
        training_weight=lambda batch: len(batch["target"]),
        evaluation_weight=lambda batch: len(batch["target"]),
    )


def validate_files(inputs, predictions, *, directory, field="files"):
    for row in predictions:
        value = struct.unpack("<d", (directory / row[field]["value"]["file"]).read_bytes())[0]
        if value != row["prediction"]:
            raise ValueError("file value differs from native prediction")
    return {"ok": True, "records": len(inputs)}


def reference_scores(pairs):
    scores = [
        {
            "id": pair["id"],
            "squared_error": (pair["prediction"]["prediction"] - pair["target"]["target"]) ** 2,
        }
        for pair in pairs
    ]
    return {
        "metrics": {"mse": sum(row["squared_error"] for row in scores) / len(scores)},
        "scores": scores,
    }


def file_scores(pairs, *, directory, reference_directory, files_field):
    scores = []
    for pair in pairs:
        row = pair["prediction"]
        value = struct.unpack("<d", (directory / row[files_field]["value"]["file"]).read_bytes())[0]
        scores.append({"id": pair["id"], "valid": value == row["prediction"]})
    return {
        "metrics": {"valid_fraction": sum(row["valid"] for row in scores) / len(scores)},
        "scores": scores,
    }


def runtime(context):
    return arti.validate_recipe_deployment(
        context.package,
        context.inputs["requests"],
        arti.PredictionRunConfig(
            context.outputs["results"],
            batch_size=context.options["batch_size"],
            files=arti.PredictionFilesConfig.from_dict(context.options["files"]),
        ),
        arti.EvaluationSuiteConfig.read(context.inputs["suite"]),
        targets=context.inputs["references"],
        task_factory=context.options["task_factory"],
        resume=context.options.get("resume", False),
        output=context.outputs["evaluation"],
    )


def write_results(context, items):
    results = []
    for index, item in enumerate(items):
        relative = f"records/{index}.json"
        path = context.directory / relative
        path.parent.mkdir(exist_ok=True)
        path.write_text(
            json.dumps({"id": item["id"], "value": item["prediction"]["prediction"]}),
            encoding="utf-8",
        )
        results.append({"id": item["id"], "files": {"record": relative}})
    return results


def validate_results(results, *, directory):
    for row in results:
        if (
            json.loads((directory / row["files"]["record"]["file"]).read_text(encoding="utf-8"))[
                "id"
            ]
            != row["id"]
        ):
            raise ValueError("formatted result lost its original ID")
    return {"ok": True, "records": len(results)}
