"""Real offline digit recognition: prepare, train, resume, test and reload.

The preparation step needs scikit-learn. Training and deployment use core ARTI.
Run: python examples/classification_workflow.py --output .tmp/digits-workflow
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import time

import torch

import arti


def prepare(output: Path, *, seed: int = 17, validation=None) -> Path:
    import sklearn
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    data = load_digits()
    indices = list(range(len(data.target)))
    train, remainder = train_test_split(
        indices, test_size=0.3, stratify=data.target, random_state=seed
    )
    validation_indices, test = train_test_split(
        remainder, test_size=0.5, stratify=data.target[remainder], random_state=seed + 1
    )
    roles = {
        index: role
        for role, rows in (("train", train), ("validation", validation_indices), ("test", test))
        for index in rows
    }
    output.mkdir(parents=True, exist_ok=True)
    path = output / "digits.jsonl"
    with path.open("w", encoding="utf-8") as stream:
        for index in indices:
            row = {
                "id": f"sklearn-digits-{index:04d}",
                "features": data.data[index].tolist(),
                "label": str(int(data.target[index])),
                "split": roles[index],
            }
            stream.write(json.dumps(row) + "\n")
    inputs = output / "unlabeled.jsonl"
    with inputs.open("w", encoding="utf-8") as stream:
        for index in test[:16]:
            stream.write(
                json.dumps(
                    {"id": f"sklearn-digits-{index:04d}", "features": data.data[index].tolist()}
                )
                + "\n"
            )
    config = {
        "task": "classification",
        "seed": seed,
        "data": {
            "path": "digits.jsonl",
            "labels": [str(value) for value in range(10)],
            "batch_size": 64,
            "preprocessing": "standardize",
        },
        "model": {"hidden_dim": 64},
        "training": {"device": "cpu", "learning_rate": 0.003, "steps": 200},
        "run": {"checkpoint_steps": 50, "evaluation_steps": 50, "output_dir": "train"},
    }
    config_path = output / "recipe.json"
    if validation is not None:
        config["run"]["validation"] = asdict(validation)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (output / "source.json").write_text(
        json.dumps(
            {
                "source": "sklearn.datasets.load_digits",
                "sklearn_version": sklearn.__version__,
                "documentation": "https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html",
                "input_layout": "8x8 grayscale, row-major 64 features, original pixel values 0..16",
                "split_method": "stratified 70/15/15 by original row id",
                "seed": seed,
                "samples": len(indices),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return config_path


def run(output: Path, *, reporter=None, validation=None) -> dict:
    started = time.monotonic()
    config_path = prepare(output, validation=validation)
    recipe = arti.ClassificationRecipe.from_config(config_path)
    initial = recipe.evaluate("validation")
    assert recipe.check()["ok"]
    first = recipe.fit(reporter=reporter)
    continued = arti.ClassificationRecipe.from_config(config_path)
    second = continued.fit(
        steps=40,
        resume_from_checkpoint=first["checkpoint"],
        run=replace(continued.config.run, output_dir=output / "resume"),
        reporter=reporter,
    )
    test = continued.evaluate("test", output=output / "test.json", checkpoint=second["selected_checkpoint"])
    predictor = arti.ClassificationPredictor.from_pretrained(second["model"])
    inputs = [json.loads(line) for line in (output / "unlabeled.jsonl").read_text().splitlines()]
    predictions = predictor.predict_records(inputs)
    with continued.session.checkpoint_context(second["selected_checkpoint"]):
        reference = continued.model(continued.data.splits["test"].features[:16])["logits"].detach().softmax(-1)
    reloaded = torch.tensor(
        [
            row["probabilities"]
            for row in predictor.predict(continued.data.splits["test"].features[:16])
        ]
    )
    torch.testing.assert_close(reloaded, reference, atol=0, rtol=0)
    (output / "predictions.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in predictions), encoding="utf-8"
    )
    report = {
        "initial_validation": initial,
        "final_validation": second["validation"],
        "test": test,
        "optimizer_steps": second["optimizer_steps"],
        "deploy_asset": second["model"],
        "resume_checkpoint": second["checkpoint"],
        "selected_checkpoint": second["selected_checkpoint"],
        "selection": second["selection"],
        "inference_reload_max_abs_error": float((reloaded - reference).abs().max()),
        "elapsed_seconds": time.monotonic() - started,
        "scope": "real labeled CPU workflow, not an architecture-superiority or CUDA benchmark",
    }
    (output / "workflow_result.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(".tmp/digits-workflow"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--select-best", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.cpu_threads)
    validation = arti.ARTIValidationConfig() if args.select_best else None
    print(
        json.dumps(
            {"config": str(prepare(args.output, validation=validation))} if args.prepare_only else run(args.output, validation=validation),
            indent=2,
        )
    )
