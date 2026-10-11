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
    recipe, *, steps=None, run=None, resume_from_checkpoint=None, reporter=None, stop_requested=None, validation_fn=None, on_checkpoint=None, recovery=None
) -> dict[str, Any]:
    """Run one task through the existing session and truthful workflow stages."""
    if resume_from_checkpoint is not None:
        recipe.session.load_checkpoint(resume_from_checkpoint)
    run = run or recipe.config.run
    validate_run = getattr(recipe, "validate_run", None)
    evaluator = None if validate_run is None else validate_run(run, validation_fn)
    requested = recipe.config.training.steps if steps is None else steps
    initial_step = recipe.session.global_step
    dataset_manifest = recipe.dataset_manifest
    run = replace(
        run,
        metadata={
            **dict(run.metadata),
            "task": recipe.task_name,
            "task_recipe_sha256": recipe.task_fingerprint,
            "dataset_sha256": dataset_manifest["sha256"],
            "target_optimizer_step": initial_step + requested if recovery is None else recovery["target_optimizer_step"],
        },
    )
    training = replace(recipe.config.training, steps=requested)
    directory = _prepare_run_directory(run).resolve()
    run = replace(run, output_dir=directory)
    resolved = replace(recipe.config, training=training, run=run)
    write_json(directory / "resolved_recipe.json", resolved.to_dict())
    write_json(directory / "split_manifest.json", dataset_manifest)
    result_path = directory / "task_result.json"
    report = {
        "ok": False,
        "kind": "recipe-train",
        "task": recipe.task_name,
        "status": "running",
        "stage": "training",
        "run_directory": str(directory),
        "task_recipe_sha256": recipe.task_fingerprint,
        "initial_optimizer_step": initial_step,
        "target_optimizer_step": run.metadata["target_optimizer_step"],
        **({"recovery": dict(recovery)} if recovery is not None else {}),
        **({"evaluation": evaluator} if evaluator is not None else {}),
    }
    write_json(result_path, report)
    preparation = getattr(recipe, "data_preparation", None)
    if preparation is not None:
        report["data_preparation"] = dict(preparation)
        write_json(directory / "data_preparation.json", report["data_preparation"])
        write_json(result_path, report)

    def emit(kind, **values):
        active = recipe.session._run
        if active is not None and active.directory.resolve() == directory:
            active.emit(kind, step=recipe.session.global_step, **values)
            active.write()

    try:
        prepare = getattr(recipe, "write_run_artifacts", None)
        if prepare is not None:
            prepare(directory)
        result = recipe.session.fit(
            recipe.train_data,
            steps=steps,
            run=run,
            reporter=reporter,
            eval_data=recipe.validation_data,
            evaluation_weight=recipe.evaluation_weight,
            validation_fn=validation_fn or ((lambda _session: recipe.evaluate("validation")) if run.validation is not None else None),
            objective_weight=getattr(recipe, "training_weight", None),
            max_grad_norm=getattr(recipe.config, "max_grad_norm", None),
            stop_requested=stop_requested,
            on_checkpoint=on_checkpoint,
        )
        report.update(
            optimizer_steps=result.steps,
            optimizer_updates_this_attempt=result.steps - initial_step,
            microbatches=recipe.session.micro_batches_seen,
            checkpoint=str(result.checkpoint_path),
            stop_reason=result.stop_reason,
            run_id=recipe.session._run.payload["run_id"],
            best_checkpoint=None if result.best_checkpoint_path is None else str(result.best_checkpoint_path),
            selected_checkpoint=str(result.selected_checkpoint_path),
            selection=dict(result.selection),
            stage="evaluation",
        )
        if callable(getattr(recipe.train_data, "state_dict", None)):
            report["data_cursor"] = recipe.session._data_state if recipe.session._data_state is not None else recipe.train_data.state_dict()
        write_json(result_path, report)
        emit("workflow_stage", stage="evaluation")
        with recipe.session.checkpoint_context(result.selected_checkpoint_path):
            report["validation"] = recipe.evaluate("validation", output=directory / "validation.json")
            report["validation"]["model_step"] = result.selection["selected_step"]
            write_json(directory / "validation.json", report["validation"])
            emit("evaluation", split="validation", phase="final", metrics=report["validation"], model_step=result.selection["selected_step"])
            report["stage"] = "export"
            write_json(result_path, report)
            emit("workflow_stage", stage="export")
            asset = recipe.export(directory / "model", selection={**dict(result.selection), "run_id": report["run_id"]})
        emit("artifact", name="model", path=str(Path(asset).parent.resolve()))
        report.update(ok=True, status="completed", stage="completed", model=str(asset))
        write_json(result_path, report)
        emit("workflow_finished", status="completed", stage="completed")
        return report
    except BaseException as error:
        report.update(
            status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
        )
        write_json(result_path, report)
        emit("workflow_finished", status=report["status"], stage=report["stage"], error_type=report["error_type"])
        raise


__all__ = ["fit_task_recipe", "read_recipe_config", "write_json", "write_jsonl"]
