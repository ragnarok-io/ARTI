"""Resume an existing task horizon and finish interrupted recipe stages."""

from __future__ import annotations

from dataclasses import replace
import hashlib
from pathlib import Path

from .serialization import (
    _decode_tree,
    _load_json,
    _member_path,
    _read_and_validate_package,
    _sidecar_paths,
    _weight_path,
)
from .training_run import ARTITrainingRunConfig


def _encoded_member(value, name):
    if not isinstance(value, dict) or set(value) != {"__arti_dict__"}:
        raise ValueError("checkpoint has no structured training state")
    fields = {_decode_tree(key, {}): item for key, item in value["__arti_dict__"]}
    return fields.get(name)


def _checkpoint_metadata(path):
    target = _weight_path(path)
    manifest, _lock = _read_and_validate_package(target, _sidecar_paths(target), allow_legacy=False)
    checkpoint = manifest.get("checkpoint")
    if checkpoint is None:
        raise ValueError("run recovery requires a training checkpoint, not an inference asset")
    encoded = _load_json(_member_path(target.parent, checkpoint["metadata_file"]))["state"]
    state = _encoded_member(encoded, "training_state")
    fields = ("global_step", "engine", "binding_kind", "task_recipe_sha256", "run_id", "validation")
    return manifest, {name: _decode_tree(_encoded_member(state, name), {}) for name in fields}


def _output_path(value, directory, report):
    path = Path(value)
    if not path.is_absolute():
        return directory / path
    original = report.get("run_directory")
    if original is not None:
        try:
            return directory / path.relative_to(Path(original))
        except ValueError:
            pass
    return path


def plan_recipe_resume(
    source_run: str | Path,
    *,
    refresh_completed: bool = False,
    data_path=None,
    cache_dir=None,
    source_manifest=None,
    base_model=None,
    resources=None,
) -> dict:
    """Inspect validated metadata without model construction or tensor loading."""
    directory = Path(source_run).resolve()
    config_path = directory / "resolved_recipe.json"
    run_path = directory / "run.json"
    resolved = _load_json(config_path)
    record = _load_json(run_path)
    task_path = directory / "task_result.json"
    report = _load_json(task_path) if task_path.exists() else {}
    if record.get("format") != "arti.training-run" or record.get("format_version") != 1:
        raise ValueError("source directory is not a supported ARTI training run")
    latest = record.get("latest_checkpoint")
    if latest is None:
        raise ValueError(
            "source run has no completed checkpoint; start training from its resolved recipe"
        )
    checkpoint = (directory / latest).resolve()
    manifest, state = _checkpoint_metadata(checkpoint)
    task_contract = manifest["architecture"]["config"].get("task_recipe", {})
    if not task_contract or task_contract.get("task") != resolved.get("task"):
        raise ValueError("source checkpoint and resolved recipe have different tasks")
    completed = state["global_step"]
    if (
        type(completed) is not int
        or completed < 0
        or state["engine"] not in {"torch", "accelerate"}
        or resolved["training"].get("distributed", False)
    ):
        raise ValueError(
            "run recovery requires a local Torch or Accelerate optimizer-step checkpoint"
        )
    metadata = record.get("run", {}).get("metadata", {})
    planned = record["plan"]
    target = metadata.get(
        "target_optimizer_step", planned["global_step"] + planned["training"]["steps"]
    )
    if type(target) is not int or target < completed:
        raise ValueError("checkpoint exceeds the recorded task horizon")
    terminal = metadata.get("training_terminal_reason")
    if (
        terminal is None
        and record.get("status") == "completed"
        and record.get("stop_reason") in {"steps_completed", "early_stopping"}
    ):
        terminal = record["stop_reason"]
    remaining = 0 if terminal is not None else target - completed
    inference = None
    ready = False
    deployment_error = None
    if report.get("ok") and report.get("status") == "completed" and report.get("model"):
        inference = _output_path(report["model"], directory, report)
        if inference.exists():
            try:
                _read_and_validate_package(
                    _weight_path(inference),
                    _sidecar_paths(_weight_path(inference)),
                    allow_legacy=False,
                )
                ready = True
            except (ValueError, FileNotFoundError) as error:
                deployment_error = type(error).__name__
    action = (
        "resume_training"
        if remaining
        else "already_completed"
        if ready and not refresh_completed
        else "finalize"
    )
    result = {
        "ok": True,
        "kind": "recipe-resume-plan",
        "source_run": str(directory),
        "source_run_id": record.get("run_id"),
        "source_run_sha256": hashlib.sha256(run_path.read_bytes()).hexdigest(),
        "config": str(config_path),
        "task": resolved["task"],
        "recipe_format": resolved.get("format"),
        "task_recipe_sha256": state["task_recipe_sha256"],
        "checkpoint": str(checkpoint),
        "checkpoint_step": completed,
        "target_optimizer_step": target,
        "remaining_optimizer_steps": remaining,
        "training_terminal_reason": terminal,
        "source_status": report.get("status", record["status"]),
        "source_stage": report.get("stage", "training"),
        "action": action,
        "model": None if inference is None else str(inference),
        "deployment_error": deployment_error,
        "budget_scope": "new_attempt",
        "max_seconds": record["run"].get("max_seconds"),
        "loads_model_weights": False,
        "loads_checkpoint_tensors": False,
        "configured_device": resolved["training"].get("device", "auto"),
    }
    overrides = {
        "data_path": data_path,
        "cache_dir": cache_dir,
        "source_manifest": source_manifest,
        "base_model": base_model,
    }
    if resolved.get("format") == "arti.task-recipe-config":
        if any(value is not None for value in overrides.values()):
            raise ValueError("recovery resource overrides apply only to causal-LM runs")
        result["resources"] = {**resolved.get("resources", {}), **(resources or {})}
        result["resource_scope"] = "task-factory"
        return result
    if resolved["task"] != "causal-lm":
        if any(value is not None for value in overrides.values()):
            raise ValueError("recovery resource overrides apply only to causal-LM runs")
        if resources is not None:
            raise ValueError("named resources overrides apply to generic task recipes")
        return result
    if resources is not None:
        raise ValueError("named resources overrides apply to generic task recipes")
    original_directory = (
        report.get("run_directory") or record.get("run", {}).get("output_dir") or str(directory)
    )
    reference = {"run_directory": original_directory}
    moved = Path(original_directory).resolve() != directory
    resources, origins = {}, {}
    for name, key in (
        ("data_path", "path"),
        ("cache_dir", "cache_dir"),
        ("source_manifest", "source_manifest"),
    ):
        recorded = resolved["data"].get(key)
        if overrides[name] is not None:
            resources[name], origins[name] = str(Path(overrides[name]).resolve()), "caller"
        elif recorded is None:
            resources[name], origins[name] = None, "disabled"
        else:
            relocated = _output_path(recorded, directory, reference).resolve()
            resources[name], origins[name] = (
                str(relocated),
                "relocated" if str(relocated) != recorded else "recorded",
            )
    stored_manifest = directory / "data_source_manifest.json"
    if (
        source_manifest is None
        and resources["source_manifest"] is not None
        and not Path(resources["source_manifest"]).exists()
        and stored_manifest.exists()
    ):
        resources["source_manifest"], origins["source_manifest"] = str(stored_manifest), "run-copy"
    if (
        cache_dir is None
        and resources["cache_dir"] is not None
        and origins["cache_dir"] == "recorded"
        and moved
        and not Path(resources["cache_dir"]).exists()
    ):
        resources["cache_dir"], origins["cache_dir"] = (
            str(directory / "prepared-chat"),
            "recovery-local",
        )
    saved_tokenizer = directory / "tokenizer"
    resources["tokenizer_path"] = str(saved_tokenizer) if saved_tokenizer.is_dir() else None
    origins["tokenizer_path"] = "run-copy" if resources["tokenizer_path"] else "pinned-model"
    recorded_base = metadata.get("recovery_resources", {}).get("base_model")
    if base_model is not None:
        resources["base_model"], origins["base_model"] = str(Path(base_model).resolve()), "caller"
    elif recorded_base is not None:
        resources["base_model"] = str(_output_path(recorded_base, directory, reference).resolve())
        origins["base_model"] = (
            "recorded" if resources["base_model"] == recorded_base else "relocated"
        )
    else:
        resources["base_model"], origins["base_model"] = None, "pinned-model"
    result["resources"] = {
        **resources,
        "origins": origins,
        "missing_local_paths": [
            name
            for name in ("data_path", "source_manifest", "base_model")
            if resources[name] is not None and not Path(resources[name]).exists()
        ],
        "validation": "not-needed-for-completed"
        if action == "already_completed"
        else "dataset-and-tokenizer-checked-before-host",
    }
    result["expected_dataset"] = task_contract["dataset"]
    return result


def resume_recipe_run(
    source_run,
    *,
    output_dir=None,
    max_seconds=None,
    refresh_completed=False,
    data_path=None,
    cache_dir=None,
    source_manifest=None,
    base_model=None,
    layer_factory=None,
    reporter=None,
    stop_requested=None,
    validation_fn=None,
    on_checkpoint=None,
    resources=None,
):
    """Restore an existing recipe using current local resource locations."""
    plan = plan_recipe_resume(
        source_run,
        refresh_completed=refresh_completed,
        data_path=data_path,
        cache_dir=cache_dir,
        source_manifest=source_manifest,
        base_model=base_model,
        resources=resources,
    )
    if plan["action"] == "already_completed":
        return _completed_report(plan)
    if plan["recipe_format"] == "arti.task-recipe-config":
        if layer_factory is not None:
            raise ValueError("layer_factory recovery applies only to causal-LM runs")
        from .task_recipe import TaskRecipe, TaskRecipeConfig
        config = TaskRecipeConfig.read(plan["config"])
        recipe = TaskRecipe.from_config(replace(config, resources=plan["resources"]))
    elif plan["task"] == "causal-lm":
        if resources is not None:
            raise ValueError("named resources overrides apply to generic task recipes")
        from .language_recipe import CausalLMRecipe, CausalLMRecipeConfig, _load_tokenizer

        resources = plan["resources"]
        if resources["missing_local_paths"]:
            raise ValueError(
                f"recovery local resources are unavailable: {resources['missing_local_paths']}"
            )
        config = CausalLMRecipeConfig.read(plan["config"])
        config = replace(
            config,
            **{name: resources[name] for name in ("data_path", "cache_dir", "source_manifest")},
        )
        tokenizer = (
            None
            if resources["tokenizer_path"] is None
            else _load_tokenizer(resources["tokenizer_path"], local_files_only=True)
        )
        recipe = CausalLMRecipe(
            config,
            tokenizer=tokenizer,
            base_model=resources["base_model"],
            layer_factory=layer_factory,
            _expected_dataset=plan["expected_dataset"],
        )
    else:
        if layer_factory is not None:
            raise ValueError("layer_factory recovery applies only to causal-LM runs")
        from .recipes import recipe_from_config

        if resources is not None:
            raise ValueError("named resources overrides apply to generic task recipes")
        recipe = recipe_from_config(plan["config"])
    run = recovery_run_config(plan, output_dir=output_dir, max_seconds=max_seconds)
    if "resources" in plan:
        run = replace(run, metadata={**dict(run.metadata), "recovery_resources": plan["resources"]})
    result = recipe.resume_run(
        source_run,
        run=run,
        refresh_completed=refresh_completed,
        reporter=reporter,
        stop_requested=stop_requested,
        validation_fn=validation_fn,
        on_checkpoint=on_checkpoint,
    )
    if "resources" in plan:
        result["recovery"]["resources"] = plan["resources"]
        from .recipe_workflow import write_json

        write_json(Path(result["run_directory"]) / "task_result.json", result)
    return result


def _completed_report(plan):
    report = _load_json(Path(plan["source_run"]) / "task_result.json")
    directory = Path(plan["source_run"])
    paths = {
        name: str(_output_path(report[name], directory, report))
        for name in ("selected_checkpoint", "best_checkpoint")
        if report.get(name) is not None
    }
    return {
        **report,
        **paths,
        "kind": "recipe-resume-run",
        "recovery": plan,
        "run_directory": str(directory),
        "model": plan["model"],
        "checkpoint": plan["checkpoint"],
        "optimizer_updates_this_attempt": 0,
    }


def recovery_run_config(plan, *, output_dir=None, max_seconds=None):
    recorded = _load_json(Path(plan["source_run"]) / "run.json")["run"]
    values = {**recorded, "output_dir": output_dir}
    if max_seconds is not None:
        values["max_seconds"] = max_seconds
    return ARTITrainingRunConfig(**values)


def resume_task_recipe(
    recipe,
    source_run,
    *,
    run=None,
    output_dir=None,
    max_seconds=None,
    refresh_completed=False,
    reporter=None,
    stop_requested=None,
    validation_fn=None,
    on_checkpoint=None,
):
    from .recipe_workflow import fit_task_recipe

    plan = plan_recipe_resume(source_run, refresh_completed=refresh_completed)
    if recipe.task_fingerprint != plan["task_recipe_sha256"]:
        raise ValueError("source run does not match this recipe's task/data contract")
    if plan["action"] == "already_completed":
        return _completed_report(plan)
    policy = run or recovery_run_config(plan, output_dir=output_dir, max_seconds=max_seconds)
    if run is not None and (output_dir is not None or max_seconds is not None):
        policy = replace(
            run,
            **({"output_dir": output_dir} if output_dir is not None else {}),
            **({"max_seconds": max_seconds} if max_seconds is not None else {}),
        )
    recipe.session.load_checkpoint(plan["checkpoint"])
    if recipe.session.global_step != plan["checkpoint_step"]:
        raise ValueError("loaded checkpoint step differs from its recovery plan")
    policy = replace(
        policy,
        metadata={
            **dict(policy.metadata),
            "target_optimizer_step": plan["target_optimizer_step"],
            "source_run_id": plan["source_run_id"],
            "source_run_sha256": plan["source_run_sha256"],
            "training_terminal_reason": plan["training_terminal_reason"],
            "budget_scope": "new_attempt",
        },
    )
    return fit_task_recipe(
        recipe,
        steps=plan["remaining_optimizer_steps"],
        run=policy,
        reporter=reporter,
        stop_requested=stop_requested,
        validation_fn=validation_fn,
        on_checkpoint=on_checkpoint,
        recovery=plan,
    )


__all__ = ["plan_recipe_resume", "resume_recipe_run"]
