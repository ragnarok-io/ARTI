from __future__ import annotations

import copy
from dataclasses import replace
import json

import pytest
import torch

import arti
from arti.cli import main
from arti.recipes import recipe_cli_report


def recipe(kind, directory, *, steps=6):
    directory.mkdir(parents=True, exist_ok=True)
    if kind == "classification":
        from test_task_recipes import _config

        cfg = _config(directory)
        return arti.ClassificationRecipe(replace(cfg, training=replace(cfg.training, steps=steps)))
    pytest.importorskip("transformers")
    from test_language_recipe import config, base_model, tokenizer

    cfg = config(directory)
    cfg = replace(cfg, training=replace(cfg.training, steps=steps))
    return arti.CausalLMRecipe(cfg, model=base_model(), tokenizer=tokenizer())


def policy(path, **kwargs):
    return arti.ARTITrainingRunConfig(output_dir=path, checkpoint_steps=1, **kwargs)


def equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            equal(a, b)
    else:
        assert left == right


def interrupt_after(step):
    def callback(session, _path):
        if session.global_step == step:
            raise KeyboardInterrupt()

    return callback


@pytest.mark.parametrize("kind", ["classification", "causal-lm"])
def test_run_recovery_keeps_original_horizon_and_exact_training_state(tmp_path, kind):
    current = recipe(kind, tmp_path / "data")
    full = recipe(kind, tmp_path / "data")
    full.fit(run=policy(tmp_path / "full"))
    with pytest.raises(KeyboardInterrupt):
        current.fit(run=policy(tmp_path / "interrupted"), on_checkpoint=interrupt_after(2))
    plan = arti.plan_recipe_resume(tmp_path / "interrupted")
    assert plan["checkpoint_step"] == 2
    assert plan["target_optimizer_step"] == 6 and plan["remaining_optimizer_steps"] == 4
    recovered = recipe(kind, tmp_path / "data")
    result = recovered.resume_run(tmp_path / "interrupted", run=policy(tmp_path / "recovered"))
    assert result["optimizer_steps"] == 6 and result["optimizer_updates_this_attempt"] == 4
    equal(
        full.session.binding.state_module().state_dict(),
        recovered.session.binding.state_module().state_dict(),
    )
    equal(full.session.optimizer.state_dict(), recovered.session.optimizer.state_dict())
    if full.session.scheduler is not None:
        equal(full.session.scheduler.state_dict(), recovered.session.scheduler.state_dict())
    assert full.session.loss_history == recovered.session.loss_history
    assert full.train_data.state_dict() == recovered.train_data.state_dict()
    state = json.loads((tmp_path / "recovered" / "run.json").read_text())
    assert state["parent_run_id"] == plan["source_run_id"]


@pytest.mark.parametrize("stage", ["evaluation", "export"])
@pytest.mark.parametrize("kind", ["classification", "causal-lm"])
def test_post_training_failure_recovery_performs_zero_updates(tmp_path, monkeypatch, stage, kind):
    current = recipe(kind, tmp_path / "data", steps=2)
    method = "evaluate" if stage == "evaluation" else "export"

    def fail(*_args, **_kwargs):
        raise OSError("stage unavailable")

    monkeypatch.setattr(current, method, fail)
    with pytest.raises(OSError):
        current.fit(run=policy(tmp_path / "source"))
    plan = arti.plan_recipe_resume(tmp_path / "source")
    assert plan["action"] == "finalize" and plan["remaining_optimizer_steps"] == 0
    original_weights = copy.deepcopy(current.session.binding.state_module().state_dict())
    original_optimizer = copy.deepcopy(current.session.optimizer.state_dict())
    original_cursor = current.train_data.state_dict()
    recovered = recipe(kind, tmp_path / "data", steps=2)
    result = recovered.resume_run(tmp_path / "source", run=policy(tmp_path / "finalized"))
    assert result["ok"] and result["optimizer_updates_this_attempt"] == 0
    equal(original_weights, recovered.session.binding.state_module().state_dict())
    equal(original_optimizer, recovered.session.optimizer.state_dict())
    assert result["data_cursor"] == original_cursor
    assert arti.plan_recipe_resume(tmp_path / "finalized")["action"] == "already_completed"


def test_repeated_recovery_preserves_one_absolute_target(tmp_path):
    current = recipe("classification", tmp_path / "data")
    with pytest.raises(KeyboardInterrupt):
        current.fit(run=policy(tmp_path / "first"), on_checkpoint=interrupt_after(2))
    a = recipe("classification", tmp_path / "data")
    with pytest.raises(KeyboardInterrupt):
        a.resume_run(
            tmp_path / "first", run=policy(tmp_path / "second"), on_checkpoint=interrupt_after(3)
        )
    plan = arti.plan_recipe_resume(tmp_path / "second")
    assert plan["target_optimizer_step"] == 6 and plan["remaining_optimizer_steps"] == 3
    b = recipe("classification", tmp_path / "data")
    result = b.resume_run(tmp_path / "second", run=policy(tmp_path / "third"))
    assert result["optimizer_steps"] == 6 and result["optimizer_updates_this_attempt"] == 3


def test_early_stopping_is_terminal_even_if_task_finalize_fails(tmp_path, monkeypatch):
    current = recipe("classification", tmp_path / "data")
    run = policy(
        tmp_path / "source", evaluation_steps=1, validation=arti.ARTIValidationConfig(patience=1)
    )
    monkeypatch.setattr(
        current, "export", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("export"))
    )
    with pytest.raises(OSError):
        current.fit(run=run, validation_fn=lambda _session: {"loss": 1.0})
    plan = arti.plan_recipe_resume(tmp_path / "source")
    assert plan["training_terminal_reason"] == "early_stopping"
    assert plan["checkpoint_step"] == 2 and plan["remaining_optimizer_steps"] == 0
    recovered = recipe("classification", tmp_path / "data")
    result = recovered.resume_run(
        tmp_path / "source", run=replace(run, output_dir=tmp_path / "finalized")
    )
    assert result["optimizer_updates_this_attempt"] == 0
    again = arti.plan_recipe_resume(tmp_path / "finalized", refresh_completed=True)
    assert again["target_optimizer_step"] == 6 and again["remaining_optimizer_steps"] == 0


def test_completed_run_is_idempotent_without_model_loading(tmp_path, monkeypatch, capsys):
    current = recipe("classification", tmp_path / "data", steps=2)
    result = current.fit(run=policy(tmp_path / "source"))
    import arti.recipes as module

    monkeypatch.setattr(
        module,
        "recipe_from_config",
        lambda *_args: (_ for _ in ()).throw(AssertionError("model creation")),
    )
    assert main(["recipe", "resume-run", str(tmp_path / "source")]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["model"] == result["model"] and report["optimizer_updates_this_attempt"] == 0
    assert main(["recipe", "resume-run", str(tmp_path / "source"), "--plan-only"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["loads_model_weights"] is False and plan["loads_checkpoint_tensors"] is False


def test_static_plan_does_not_decode_tensor_payloads(tmp_path, monkeypatch):
    current = recipe("classification", tmp_path / "data", steps=1)
    current.fit(run=policy(tmp_path / "source"))
    import arti.serialization as codec

    monkeypatch.setattr(
        codec,
        "_load_safetensors",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("tensor load")),
    )
    assert arti.plan_recipe_resume(tmp_path / "source")["checkpoint_step"] == 1


def test_missing_export_is_rebuilt_without_more_training(tmp_path):
    current = recipe("classification", tmp_path / "data", steps=1)
    result = current.fit(run=policy(tmp_path / "source"))
    from pathlib import Path

    Path(result["model"]).unlink()
    assert arti.plan_recipe_resume(tmp_path / "source")["action"] == "finalize"
    recovered = recipe("classification", tmp_path / "data", steps=1)
    report = recovered.resume_run(tmp_path / "source", run=policy(tmp_path / "repaired"))
    assert report["optimizer_updates_this_attempt"] == 0 and report["ok"]


def test_source_log_tail_is_not_treated_as_saved_progress(tmp_path, monkeypatch):
    current = recipe("classification", tmp_path / "data")
    forward = current.model.forward
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("unsaved tail")
        return forward(*args, **kwargs)

    monkeypatch.setattr(current.model, "forward", interrupted)
    with pytest.raises(RuntimeError):
        current.fit(
            run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "source", checkpoint_steps=50)
        )
    record = json.loads((tmp_path / "source" / "run.json").read_text())
    assert record["global_step"] == 2
    plan = arti.plan_recipe_resume(tmp_path / "source")
    assert plan["checkpoint_step"] == 1 and plan["remaining_optimizer_steps"] == 5


def test_budget_stop_with_an_export_is_not_task_horizon_completion(tmp_path):
    current = recipe("classification", tmp_path / "data")
    result = current.fit(
        run=policy(tmp_path / "source"), stop_requested=lambda: current.session.global_step >= 2
    )
    assert result["ok"] and result["stop_reason"] == "requested"
    plan = arti.plan_recipe_resume(tmp_path / "source")
    assert plan["action"] == "resume_training" and plan["remaining_optimizer_steps"] == 4
    recovered = recipe("classification", tmp_path / "data")
    outcome = recovered.resume_run(
        tmp_path / "source", output_dir=tmp_path / "attempt", max_seconds=5
    )
    assert outcome["optimizer_steps"] == 6
    assert outcome["recovery"]["budget_scope"] == "new_attempt"


def test_cli_refresh_respects_original_selection_and_new_budget(tmp_path):
    current = recipe("classification", tmp_path / "data", steps=2)
    run = policy(tmp_path / "source", evaluation_steps=1, validation=arti.ARTIValidationConfig())
    current.fit(run=run)
    result = recipe_cli_report(
        "resume-run",
        tmp_path / "source",
        run_dir=tmp_path / "refresh",
        refresh_completed=True,
        max_seconds=4,
    )
    assert result["optimizer_updates_this_attempt"] == 0
    recorded = json.loads((tmp_path / "refresh" / "run.json").read_text())
    assert recorded["run"]["max_seconds"] == 4
    assert recorded["run"]["validation"] == run.to_dict()["validation"]
    assert arti.torch.plan_recipe_resume is arti.plan_recipe_resume


def test_corrupt_derived_export_can_be_rebuilt_but_checkpoint_is_required(tmp_path):
    from pathlib import Path

    current = recipe("classification", tmp_path / "data", steps=1)
    result = current.fit(run=policy(tmp_path / "source"))
    model = Path(result["model"])
    model.with_suffix(".json").unlink()
    plan = arti.plan_recipe_resume(tmp_path / "source")
    assert plan["action"] == "finalize" and plan["deployment_error"] == "ValueError"
    Path(plan["checkpoint"]).unlink()
    with pytest.raises(FileNotFoundError):
        arti.plan_recipe_resume(tmp_path / "source")


def test_finalize_keeps_last_optimizer_and_exports_earlier_best(tmp_path, monkeypatch):
    current = recipe("classification", tmp_path / "data", steps=3)
    run = policy(tmp_path / "source", evaluation_steps=1, validation=arti.ARTIValidationConfig())
    monkeypatch.setattr(
        current, "export", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("export"))
    )
    with pytest.raises(OSError):
        current.fit(run=run, validation_fn=lambda session: {"loss": float(session.global_step)})
    last_weights = copy.deepcopy(current.model.state_dict())
    moments = copy.deepcopy(current.session.optimizer.state_dict())
    recovered = recipe("classification", tmp_path / "data", steps=3)
    result = recovered.resume_run(tmp_path / "source", output_dir=tmp_path / "finalized")
    assert result["optimizer_steps"] == 3 and result["optimizer_updates_this_attempt"] == 0
    assert result["selection"]["selected_step"] == 1
    equal(last_weights, recovered.model.state_dict())
    equal(moments, recovered.session.optimizer.state_dict())
    deployed = arti.ClassificationPredictor.from_pretrained(result["model"])
    with recovered.session.checkpoint_context(result["selected_checkpoint"]):
        equal(deployed.model.state_dict(), recovered.model.state_dict())


def test_completed_result_uses_relocated_run_artifact_paths(tmp_path, monkeypatch):
    from pathlib import Path
    import shutil
    import arti.recipes as module

    current = recipe("classification", tmp_path / "data", steps=2)
    current.fit(run=policy(tmp_path / "source"))
    relocated = tmp_path / "relocated"
    shutil.copytree(tmp_path / "source", relocated)
    monkeypatch.setattr(
        module,
        "recipe_from_config",
        lambda *_args: (_ for _ in ()).throw(AssertionError("model creation")),
    )
    result = recipe_cli_report("resume-run", relocated)
    assert result["optimizer_updates_this_attempt"] == 0
    assert Path(result["run_directory"]) == relocated
    for name in ("model", "checkpoint", "selected_checkpoint"):
        assert Path(result[name]).is_relative_to(relocated) and Path(result[name]).is_file()
