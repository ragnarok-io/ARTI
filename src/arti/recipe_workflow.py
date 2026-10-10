"""Shared data-to-model lifecycle for task recipes, outside neural execution."""

from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
from typing import Any

from ._toml import loads as load_toml
from .training_run import _prepare_run_directory


def read_recipe_config(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if target.suffix.lower() == ".json":
        payload = json.loads(target.read_text(encoding="utf-8"))
    elif target.suffix.lower() == ".toml":
        payload = load_toml(target.read_text(encoding="utf-8"))
    else:
        raise ValueError("a recipe config must be JSON or TOML")
    if not isinstance(payload, dict):
        raise ValueError("a recipe config must contain an object")
    return payload


def write_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
    )


def write_jsonl(path: str | Path, records) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")


def fit_task_recipe(
    recipe, *, steps=None, run=None, resume_from_checkpoint=None, reporter=None, stop_requested=None
) -> dict[str, Any]:
    """Run one task through the existing session and truthful workflow stages."""
    if resume_from_checkpoint is not None:
        recipe.session.load_checkpoint(resume_from_checkpoint)
    run = run or recipe.config.run
    run = replace(
        run,
        metadata={
            **dict(run.metadata),
            "task": recipe.task_name,
            "task_recipe_sha256": recipe.task_fingerprint,
            "dataset_sha256": recipe.data.fingerprint,
        },
    )
    requested = recipe.config.training.steps if steps is None else int(steps)
    training = replace(recipe.config.training, steps=requested)
    directory = _prepare_run_directory(run).resolve()
    run = replace(run, output_dir=directory)
    resolved = replace(recipe.config, training=training, run=run)
    write_json(directory / "resolved_recipe.json", resolved.to_dict())
    write_json(directory / "split_manifest.json", recipe.data.manifest(include_ids=True))
    result_path = directory / "task_result.json"
    report = {
        "ok": False,
        "kind": "recipe-train",
        "task": recipe.task_name,
        "status": "running",
        "stage": "training",
        "run_directory": str(directory),
        "task_recipe_sha256": recipe.task_fingerprint,
    }
    write_json(result_path, report)
    try:
        prepare = getattr(recipe, "write_run_artifacts", None)
        if prepare is not None:
            prepare(directory)
        result = recipe.session.fit(
            recipe.train_data,
            steps=steps,
            run=run,
            reporter=reporter,
            eval_data=recipe.data.splits["validation"].batches(recipe.config.batch_size),
            evaluation_weight=recipe.evaluation_weight,
            objective_weight=getattr(recipe, "training_weight", None),
            max_grad_norm=getattr(recipe.config, "max_grad_norm", None),
            stop_requested=stop_requested,
        )
        report.update(
            optimizer_steps=result.steps,
            microbatches=recipe.session.micro_batches_seen,
            checkpoint=str(result.checkpoint_path),
            stop_reason=result.stop_reason,
            stage="evaluation",
        )
        write_json(result_path, report)
        report["validation"] = recipe.evaluate("validation", output=directory / "validation.json")
        report["stage"] = "export"
        write_json(result_path, report)
        asset = recipe.export(directory / "model")
        report.update(ok=True, status="completed", stage="completed", model=str(asset))
        write_json(result_path, report)
        return report
    except BaseException as error:
        report.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
        )
        write_json(result_path, report)
        raise


__all__ = ["fit_task_recipe", "read_recipe_config", "write_json", "write_jsonl"]
