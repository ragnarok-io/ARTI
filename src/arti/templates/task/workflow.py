"""Explicit application commands; existing ARTI journals own lifecycle state."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import arti
from arti.recipe_data import read_jsonl
from arti.recipe_workflow import read_recipe_config, write_json
from arti.task_project import read_task_project


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "code"))


def _project():
    return read_task_project(ROOT / "project.json")


def _pin_training(report):
    # Update explicit selected artifact pointers, without changing any native
    # recipe/objective/configuration or introducing a second run authority.
    value = _project().to_dict()
    value["artifacts"]["training_run"] = Path(report["run_directory"]).relative_to(ROOT).as_posix()
    if report.get("model"):
        value["artifacts"]["model"] = Path(report["model"]).relative_to(ROOT).as_posix()
    write_json(ROOT / "project.json", value)


def _stop_after_step(step):
    def callback(session, _checkpoint):
        if step is not None and session.global_step >= step:
            raise KeyboardInterrupt

    return callback


def prepare(args):
    project = _project()
    resources, configs = project.resolved_paths("resources"), project.resolved_paths("configs")
    report = arti.prepare_dataset_assets(
        arti.DatasetSnapshot(resources["snapshot"]),
        resources["cache"],
        arti.AssetTransformConfig.read(configs["transform"]),
        batch_size=3,
    ).report
    write_json(project.resolved_paths("artifacts")["preparation_report"], report)
    return report


def train(args):
    project = _project()
    recipe = arti.TaskRecipe.from_config(project.resolved_recipe())
    try:
        report = recipe.fit(on_checkpoint=_stop_after_step(args.stop_after))
    except KeyboardInterrupt:
        if args.stop_after is None:
            raise
        report = read_recipe_config(Path(recipe.config.run.output_dir) / "task_result.json")
    _pin_training(report)
    return report


def recover(args):
    project = _project()
    paths = project.resolved_paths("artifacts")
    report = arti.resume_recipe_run(
        paths["training_run"],
        output_dir=ROOT / "runs/recovered",
        resources={key: str(path) for key, path in project.resolved_paths("resources").items()},
    )
    _pin_training(report)
    return report


def _prediction_config(project):
    configs = project.resolved_paths("configs")
    payload = read_recipe_config(configs["prediction"])
    payload.pop("format", None)
    payload["output_dir"] = project.resolved_paths("artifacts")["prediction_run"]
    file_config = read_recipe_config(configs["files"])
    file_config.pop("format", None)
    payload["files"] = arti.PredictionFilesConfig.from_dict(file_config)
    return arti.PredictionRunConfig(**payload)


def predict(args):
    project = _project()
    artifacts, inputs = project.resolved_paths("artifacts"), project.resolved_paths("inputs")
    batches = 0

    def committed(_progress, _receipt):
        nonlocal batches
        batches += 1
        if args.stop_after is not None and batches >= args.stop_after:
            raise KeyboardInterrupt

    try:
        return arti.run_prediction(
            artifacts["deployment"],
            inputs["requests"] if not args.resume else None,
            _prediction_config(project),
            resume=args.resume,
            task_factory=project.resolved_recipe().factory,
            on_batch=committed,
        )
    except KeyboardInterrupt:
        if args.stop_after is None:
            raise
        return {"status": "interrupted", **arti.inspect_prediction(artifacts["prediction_run"])}


def evaluate(args):
    project = _project()
    return arti.evaluate_prediction_suite(
        project.resolved_paths("artifacts")["prediction_run"],
        arti.EvaluationSuiteConfig.read(project.resolved_paths("configs")["suite"]),
        references=project.resolved_paths("inputs")["references"],
        output=project.resolved_paths("artifacts")["evaluation"],
    )


def package(args):
    project = _project()
    artifacts = project.resolved_paths("artifacts")
    return arti.package_recipe_deployment(
        artifacts["model"],
        artifacts["deployment"],
        resources=project.resolved_paths("resources"),
        runtime=arti.TaskRuntimeProfile.read(project.resolved_paths("configs")["runtime"]),
        code={"application": ROOT / "code/task_app.py"},
    )


def runtime(args):
    project = _project()
    artifacts = project.resolved_paths("artifacts")
    # A fresh cold run is distinct from an existing model-only prediction plan.
    # --resume explicitly reuses the prior cold journal; it does not silently
    # reinterpret model-only evidence as package-qualified inference.
    return arti.run_task_runtime(
        artifacts["deployment"],
        inputs={
            **project.resolved_paths("inputs"),
            "suite": project.resolved_paths("configs")["suite"],
        },
        outputs={
            "results": ROOT / "artifacts/cold-prediction",
            "evaluation": ROOT / "artifacts/cold-evaluation.json",
        },
        options={"resume": args.resume},
        output=artifacts["runtime_receipt"],
    )


def export(args):
    project = _project()
    artifacts = project.resolved_paths("artifacts")
    return arti.export_prediction_results(
        artifacts["prediction_run"],
        arti.ResultExportConfig.read(project.resolved_paths("configs")["export"]),
        artifacts["result_export"],
        expected_ids=[
            row["id"] for row in read_jsonl(project.resolved_paths("inputs")["requests"])
        ],
        deployment=artifacts["deployment"],
        evidence={"evaluation": artifacts["evaluation"], "runtime": artifacts["runtime_receipt"]},
        attachments={
            "inputs": ROOT / "data/requests.jsonl",
            "references": ROOT / "data/references.jsonl",
            "suite": project.resolved_paths("configs")["suite"],
            "journal": artifacts["prediction_run"],
            "cold_journal": ROOT / "artifacts/cold-prediction",
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for command in (
        "prepare",
        "train",
        "recover",
        "predict",
        "evaluate",
        "package",
        "runtime",
        "export",
    ):
        subparser = commands.add_parser(command)
        if command in {"train", "predict"}:
            subparser.add_argument(
                "--stop-after",
                type=int,
                help="Illustrate interruption after an optimizer checkpoint or committed prediction batch",
            )
        if command in {"predict", "runtime"}:
            subparser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if getattr(args, "stop_after", None) is not None and args.stop_after <= 0:
        parser.error("--stop-after must be positive")
    result = globals()[args.command](args)
    print(json.dumps(result, indent=2, allow_nan=False))
    if isinstance(result, dict) and result.get("status") in {"not_run", "failed"}:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
