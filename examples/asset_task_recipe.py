"""Explicit application-side binary decoding and collation for TaskRecipe.

The core snapshot/cache has no format-specific loader. This example owns a
little-endian float array format, native tensor shape and regression objective.
It reuses the native model/example helpers rather than building another trainer.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
import struct

import torch

import arti
from arti.recipe_data import json_fingerprint
from examples.custom_task_recipe import ExampleModel, _features, _metrics, _objective, _predict


FACTORY = "examples.asset_task_recipe:build_task"
TRANSFORM = "examples.asset_task_recipe:decode_arrays"


def decode_arrays(requests, *, options):
    metadata = []
    for request in requests:
        shape = request.metadata["shape"]
        count = math.prod(shape)
        values = struct.unpack(f"<{count}f", request.source.read_bytes())
        tensor = torch.tensor(values).reshape(shape) * options.get("scale", 1.0)
        (request.directory / "array.json").write_text(
            json.dumps(tensor.tolist(), allow_nan=False), encoding="utf-8"
        )
        metadata.append({"shape": shape, "dtype": "float32"})
    return metadata


def _write_prediction_files(directory, prediction, *, formats, side):
    directory.mkdir(parents=True)
    files = {}
    if "array" in formats:
        (directory / "array.json").write_text(json.dumps([prediction]), encoding="utf-8")
        files["array"] = {"file": "array.json", "metadata": {"shape": [1], "units": "original"}}
    if "binary" in formats:
        (directory / "value.bin").write_bytes(struct.pack("<d", prediction))
        files["binary"] = {"file": "value.bin", "metadata": {"encoding": "little-endian-float64"}}
    if "image" in formats:
        pixel = int(min(abs(prediction) * 20, 255))
        (directory / "image.pgm").write_bytes(
            f"P5\n{side} {side}\n255\n".encode("ascii") + bytes([pixel]) * (side * side)
        )
        files["image"] = {
            "file": "image.pgm",
            "metadata": {"shape": [side, side], "encoding": "gray-pgm"},
        }
    return files


def validate_files(inputs, predictions, *, directory, field="files"):
    for row in predictions:
        for name, file in row[field].items():
            path = directory / file["file"]
            if name == "array":
                if json.loads(path.read_text(encoding="utf-8")) != [row["prediction"]]:
                    raise ValueError("array output differs from the native prediction")
            elif name == "binary":
                if struct.unpack("<d", path.read_bytes())[0] != row["prediction"]:
                    raise ValueError("binary output differs from the native prediction")
            elif name == "image":
                header, size, maximum, pixels = path.read_bytes().split(b"\n", 3)
                width, height = map(int, size.split())
                if header != b"P5" or maximum != b"255" or len(pixels) != width * height:
                    raise ValueError("invalid application bitmap output")
    return {
        "ok": True,
        "records": len(inputs),
        "files": sum(len(row[field]) for row in predictions),
    }


def build_task(config, *, mode, contract=None):
    snapshot = arti.DatasetSnapshot(config.resources["snapshot"])
    transform = arti.AssetTransformConfig(
        TRANSFORM, "float-array-v1", config.options.get("transform", {})
    )
    prepared = arti.prepare_dataset_assets(
        snapshot,
        config.resources["cache"],
        transform,
        roles=("input",),
        batch_size=config.options.get("preparation_batch_size", 16),
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(config.options.get("seed", 17))
        model = ExampleModel(config.options)

    def native_rows(records):
        result = []
        for row in records:
            references = snapshot.references(row["id"], role="input")
            if len(references) != 1:
                raise ValueError("this application expects one input array per record")
            path = prepared.files(references[0]["asset_id"])["array.json"]
            features = json.loads(path.read_text(encoding="utf-8"))
            result.append({**row, "features": features})
        return result

    def predict(host, records, *, output_dir=None, **kwargs):
        predictions = list(_predict(host, native_rows(records), **kwargs))
        if output_dir is not None:
            for row in predictions:
                folder = json_fingerprint(row["id"])
                files = _write_prediction_files(
                    Path(output_dir) / folder,
                    row["prediction"],
                    formats=config.options.get("file_formats", ("array", "binary", "image")),
                    side=config.options.get("image_side", 2),
                )
                row[config.options.get("file_field", "files")] = {
                    name: {**item, "file": f"{folder}/{item['file']}"}
                    for name, item in files.items()
                }
        return predictions

    if mode == "predict":
        return arti.TaskPredictor(
            model,
            contract,
            predict_fn=predict,
            evaluate_fn=lambda host, rows: _metrics(host, native_rows(rows)),
        )
    splits = {
        name: native_rows(list(snapshot.records(name))) for name in ("train", "validation", "test")
    }
    train = _features(model, splits["train"]).flatten(1)
    model.feature_mean.copy_(train.mean(0))
    model.feature_scale.copy_(train.std(0, unbiased=False).clamp_min(1e-6))
    target_scale = torch.tensor([row["target"] for row in splits["train"]]).std(unbiased=False)
    model.target_scale.copy_(target_scale.clamp_min(1.0))
    batch_size = config.options.get("batch_size", 4)

    def batches(rows):
        return [
            {
                "features": _features(model, rows[start : start + batch_size]),
                "target": torch.tensor([row["target"] for row in rows[start : start + batch_size]]),
            }
            for start in range(0, len(rows), batch_size)
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
        objective=_objective,
        objective_id="scaled-mse",
        evaluate_fn=lambda host, split: _metrics(host, splits[split]),
        predict_fn=predict,
        metadata={
            "input_shape": list(model.input_shape),
            "preprocessing": "train-only model buffers",
        },
        training_weight=lambda batch: len(batch["target"]),
        evaluation_weight=lambda batch: len(batch["target"]),
    )
