from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
from torch import nn

import arti
from arti.cli import main
from test_recipe_recovery import equal, interrupt_after, policy

pytest.importorskip("accelerate")


def qwen(directory, *, steps=4, accumulation=2, dropout=False, factory=None):
    pytest.importorskip("transformers")
    from test_language_recipe import base_model, config, tokenizer

    directory.mkdir(parents=True, exist_ok=True)
    cfg = config(directory, **({"layer_factory": factory} if factory else {}))
    cfg = replace(
        cfg,
        training=replace(
            cfg.training, engine="accelerate", steps=steps, gradient_accumulation_steps=accumulation
        ),
    )
    model = base_model()
    if dropout:
        model.config.attention_dropout = 0.2
        for layer in model.model.layers:
            layer.self_attn.attention_dropout = 0.2
    return arti.CausalLMRecipe(cfg, model=model, tokenizer=tokenizer())


class Regression(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.25))

    def forward(self, x, y):
        return {"loss": ((self.weight * x - y) ** 2).mean()}


def test_weighted_native_accumulation_matches_a_joint_batch_and_prepares_once(tmp_path):
    model = Regression()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    session = arti.trainer(
        model,
        engine="accelerate",
        device="cpu",
        gradient_accumulation_steps=2,
        optimizer=optimizer,
        scheduler=scheduler,
    )
    batches = [
        {"x": torch.tensor([1.0]), "y": torch.tensor([2.0])},
        {"x": torch.tensor([3.0, 4.0, 5.0]), "y": torch.tensor([1.0, 2.0, 3.0])},
    ]
    expected = Regression()
    joint_optimizer = torch.optim.SGD(expected.parameters(), lr=0.1)
    joint = {key: torch.cat([batch[key] for batch in batches]) for key in ("x", "y")}
    expected(**joint)["loss"].backward()
    joint_optimizer.step()
    result = session.fit(
        batches,
        steps=1,
        objective_weight=lambda batch: len(batch["x"]),
        run=policy(tmp_path / "first"),
    )
    torch.testing.assert_close(model.weight, expected.weight, rtol=0, atol=0)
    assert result.steps == 1 and session.micro_batches_seen == 2
    assert scheduler.last_epoch == 1
    engine, prepared_optimizer = session.engine_object, session.optimizer
    session.fit(
        batches,
        steps=1,
        objective_weight=lambda batch: len(batch["x"]),
        run=policy(tmp_path / "second"),
    )
    assert session.engine_object is engine and session.optimizer is prepared_optimizer
    assert len(engine._optimizers) == 1 and scheduler.last_epoch == 2


def test_qwen_real_updates_and_exact_interrupted_continuation_with_dropout(tmp_path):
    full = qwen(tmp_path / "data", dropout=True)
    split = qwen(tmp_path / "data", dropout=True)
    frozen = {
        name: parameter.detach().clone()
        for name, parameter in full.model.named_parameters()
        if not parameter.requires_grad
    }
    before = [parameter.detach().clone() for parameter in full.model.arti.parameters()]
    initial_rng = torch.get_rng_state().clone()
    full.fit(run=policy(tmp_path / "full"))
    torch.set_rng_state(initial_rng)
    with pytest.raises(KeyboardInterrupt):
        split.fit(run=policy(tmp_path / "split"), on_checkpoint=interrupt_after(2))
    plan = arti.plan_recipe_resume(tmp_path / "split")
    assert plan["remaining_optimizer_steps"] == 2 and not plan["loads_checkpoint_tensors"]
    restored = qwen(tmp_path / "data", dropout=True)
    result = restored.resume_run(tmp_path / "split", run=policy(tmp_path / "restored"))
    assert result["optimizer_updates_this_attempt"] == 2
    equal(full.model.arti._bundle().state_dict(), restored.model.arti._bundle().state_dict())
    equal(full.session.optimizer.state_dict(), restored.session.optimizer.state_dict())
    equal(full.session.scheduler.state_dict(), restored.session.scheduler.state_dict())
    assert full.session.loss_history == restored.session.loss_history
    assert full.train_data.state_dict() == restored.train_data.state_dict()
    assert any(
        not torch.equal(a, b) for a, b in zip(before, full.model.arti.parameters(), strict=True)
    )
    assert all(
        torch.equal(value, dict(full.model.named_parameters())[name])
        for name, value in frozen.items()
    )


def test_validation_best_export_preserves_last_training_state_and_periodic_assets(tmp_path):
    from test_language_recipe import base_model

    recipe = qwen(tmp_path / "data", steps=3)
    run = policy(
        tmp_path / "run",
        evaluation_steps=1,
        validation=arti.ARTIValidationConfig(monitor="fixture_score"),
    )
    result = recipe.fit(
        run=run, validation_fn=lambda session: {"fixture_score": float(session.global_step)}
    )
    assert result["selection"]["selected_step"] == 1 and recipe.session.global_step == 3
    assert (tmp_path / "run" / "checkpoints" / "step-00000001" / "model.arti.st").exists()
    loaded = arti.CausalLMPredictor.from_pretrained(result["model"], model=base_model())
    batch = recipe.data.splits["validation"].batches(1)[0]
    with recipe.session.checkpoint_context(result["selected_checkpoint"]), torch.no_grad():
        expected = recipe.model(**batch, use_cache=False).logits
    with torch.no_grad():
        torch.testing.assert_close(
            loaded.model(**batch, use_cache=False).logits, expected, rtol=0, atol=0
        )
    assert recipe.session.global_step == 3


def test_early_stop_and_zero_update_finalization(tmp_path, monkeypatch):
    recipe = qwen(tmp_path / "data", steps=6)
    run = policy(
        tmp_path / "source", evaluation_steps=1, validation=arti.ARTIValidationConfig(patience=1)
    )
    monkeypatch.setattr(
        recipe,
        "export",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("export unavailable")),
    )
    with pytest.raises(OSError):
        recipe.fit(run=run, validation_fn=lambda _session: {"loss": 1.0})
    plan = arti.plan_recipe_resume(tmp_path / "source")
    assert (
        plan["training_terminal_reason"] == "early_stopping"
        and plan["remaining_optimizer_steps"] == 0
    )
    restored = qwen(tmp_path / "data", steps=6)
    result = restored.resume_run(
        tmp_path / "source",
        run=replace(run, output_dir=tmp_path / "finalized"),
        validation_fn=lambda _session: {"loss": 1.0},
    )
    assert result["optimizer_updates_this_attempt"] == 0 and restored.session.global_step == 2
    assert arti.plan_recipe_resume(tmp_path / "finalized")["action"] == "already_completed"


def test_budget_stop_happens_at_completed_update_and_saves_assets(tmp_path):
    recipe = qwen(tmp_path / "data", steps=5)
    result = recipe.fit(
        run=policy(tmp_path / "budget", max_seconds=10),
        stop_requested=lambda: recipe.session.global_step >= 1,
    )
    assert result["optimizer_steps"] == 1 and result["stop_reason"] == "requested"
    assert Path(result["checkpoint"]).exists() and Path(result["model"]).exists()


def test_elapsed_budget_stops_before_the_next_accumulation_group(tmp_path):
    recipe = qwen(tmp_path / "data", steps=5)

    def expire_after_save(session, _path):
        session._run.started -= 120

    result = recipe.fit(
        run=policy(tmp_path / "budget", max_seconds=60), on_checkpoint=expire_after_save
    )
    assert result["optimizer_steps"] == 1 and result["stop_reason"] == "time_budget"
    assert recipe.session.micro_batches_seen == 2
    assert recipe.train_data.state_dict()["samples_seen"] == 2 * recipe.config.batch_size


def test_classification_recipe_accelerate_train_reload_and_resume(tmp_path):
    from test_task_recipes import _config

    cfg = _config(tmp_path)
    cfg = replace(cfg, training=replace(cfg.training, engine="accelerate", steps=3))
    recipe = arti.ClassificationRecipe(cfg)
    result = recipe.fit(run=policy(tmp_path / "run"))
    assert result["optimizer_steps"] == 3
    loaded = arti.ClassificationPredictor.from_pretrained(result["model"])
    assert loaded.labels == recipe.data.labels
    restored = arti.ClassificationRecipe(cfg)
    no_op = restored.resume_run(tmp_path / "run")
    assert no_op["optimizer_updates_this_attempt"] == 0


def test_native_standalone_and_attachment_share_the_local_accelerate_loop(tmp_path):
    from test_shared_training import NativeProgram
    from test_training_run import TaskModel, graph_layer, data

    native = NativeProgram()
    attached = arti.ARTI.attach(TaskModel(), graph_layer(), layers="features")
    a = arti.trainer(native, engine="accelerate", device="cpu", gradient_accumulation_steps=2)
    b = attached.arti.trainer(engine="accelerate", device="cpu", gradient_accumulation_steps=2)
    a.fit(data(), steps=2, run=policy(tmp_path / "native"))
    b.fit(data(), steps=2, run=policy(tmp_path / "attached"))
    assert a.loss_history == b.loss_history
    for left, right in zip(native.parameters(), attached.arti.parameters(), strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)


def test_cli_accelerate_recipe_with_custom_factory_and_offline_reload(
    tmp_path, monkeypatch, capsys
):
    import arti.language_recipe as module
    from test_language_recipe import base_model, tokenizer

    descriptor = arti.LayerFactoryConfig("examples.qwen_program_factory:build_layer", {"width": 4})
    recipe = qwen(tmp_path / "data", steps=2, factory=descriptor)
    cfg = tmp_path / "recipe.json"
    cfg.write_text(json.dumps(recipe.config.to_dict()))
    monkeypatch.setattr(module, "_load_model", lambda *_args, **_kwargs: base_model())
    monkeypatch.setattr(module, "_load_tokenizer", lambda *_args, **_kwargs: tokenizer())
    assert main(["recipe", "train", str(cfg), "--run-dir", str(tmp_path / "run")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["optimizer_steps"] == 2 and result["ok"]
    assert main(["recipe", "resume-run", str(tmp_path / "run")]) == 0
    assert json.loads(capsys.readouterr().out)["optimizer_updates_this_attempt"] == 0
    native = tmp_path / "native"
    base_model().save_pretrained(native)
    package = arti.package_causal_lm_deployment(
        result["model"], tmp_path / "package", include_base=True, base_model=native
    )
    predictor = arti.CausalLMPredictor.from_pretrained(
        package["directory"], layer_factory=descriptor.reference
    )
    assert predictor.contract["layer_factory"] == descriptor.to_dict()


def test_cpu_bf16_native_accelerator_in_a_clean_process(tmp_path):
    code = "import torch,arti,sys;from torch import nn; model=nn.Linear(4,4); session=arti.trainer(model,engine='accelerate',device='cpu',mixed_precision='bf16',objective=lambda m,b:(m(b['x'])-b['y']).square().mean()); result=session.fit([{'x':torch.ones(2,4),'y':torch.zeros(2,4)}],steps=2,run=arti.ARTITrainingRunConfig(output_dir=sys.argv[1])); assert result.steps==2; assert session.engine_object.mixed_precision=='bf16'; print('bf16-local-pass')"
    env = dict(
        os.environ,
        PYTHONPATH=str(Path(arti.__file__).resolve().parent.parent),
        CUDA_VISIBLE_DEVICES="-1",
        ACCELERATE_USE_CPU="true",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
    )
    child = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path / "bf16")],
        env=env,
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=90,
        check=True,
    )
    assert "bf16-local-pass" in child.stdout


def test_qwen_unequal_answer_lengths_match_joint_token_normalization(tmp_path):
    from test_language_recipe import base_model, config, tokenizer
    from arti.language_recipe import causal_lm_model_loss
    from arti.recipe_chat import collate_chat, target_token_weight
    from arti.recipe_data import read_jsonl
    from arti.recipe_workflow import write_jsonl

    cfg = config(tmp_path)
    rows = list(read_jsonl(cfg.data_path))
    train = [row for row in rows if row["split"] == "train"]
    train[1]["messages"][-1]["content"] = "red blue green red"
    write_jsonl(cfg.data_path, rows)
    reference = arti.CausalLMRecipe(cfg, model=base_model(), tokenizer=tokenizer())
    actual = arti.CausalLMRecipe(cfg, model=base_model(), tokenizer=tokenizer())
    examples = reference.data.splits["train"].examples[:2]
    assert examples[0].target_tokens != examples[1].target_tokens
    joint = collate_chat(
        examples, pad_token_id=reference.tokenizer.pad_token_id, pad_to_multiple_of=8
    )
    separate = [
        collate_chat([example], pad_token_id=reference.tokenizer.pad_token_id, pad_to_multiple_of=8)
        for example in examples
    ]
    expected_optimizer = torch.optim.SGD(reference.model.arti.parameters(), lr=0.1)
    causal_lm_model_loss(reference.model, joint).backward()
    expected_optimizer.step()
    optimizer = torch.optim.SGD(actual.model.arti.parameters(), lr=0.1)
    session = actual.model.arti.trainer(
        engine="accelerate",
        device="cpu",
        objective=causal_lm_model_loss,
        gradient_accumulation_steps=2,
        optimizer=optimizer,
    )
    session.fit(
        separate, steps=1, objective_weight=target_token_weight, run=policy(tmp_path / "normalized")
    )
    for expected, observed in zip(
        reference.model.arti.parameters(), actual.model.arti.parameters(), strict=True
    ):
        torch.testing.assert_close(expected, observed, rtol=1e-6, atol=1e-7)
