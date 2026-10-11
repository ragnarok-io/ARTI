from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

import pytest
import torch

import arti
from arti.cli import main
from test_recipe_recovery import equal, interrupt_after, policy


def setup_source(tmp_path, *, engine="torch", custom=False, steps=4):
    pytest.importorskip("transformers")
    if engine == "accelerate":
        pytest.importorskip("accelerate")
    from test_language_recipe import base_model, tokenizer
    from test_recipe_factory import factory
    from test_recipe_import import imported

    old = tmp_path / "old"
    _declaration, _imported, cfg = imported(old / "data")
    cfg = replace(
        cfg,
        batch_size=5,
        max_tokens_per_batch=48,
        layer_factory=factory() if custom else None,
        optimizer=replace(cfg.optimizer, scheduler="linear", warmup_steps=1, total_steps=steps),
        training=replace(cfg.training, steps=steps, engine=engine, gradient_accumulation_steps=2),
    )
    native = old / "native"
    host = base_model()
    host.config.attention_dropout = 0.2
    host.save_pretrained(native)

    def construct():
        return arti.CausalLMRecipe(cfg, tokenizer=tokenizer(), base_model=native)

    return cfg, construct


def relocated(tmp_path, cfg):
    old, moved = tmp_path / "old", tmp_path / "moved"
    data_relative = cfg.data_path.relative_to(old)
    shutil.move(str(old), moved)
    assert not old.exists()
    return moved / "run", {"data_path": moved / data_relative, "base_model": moved / "native"}


def interrupted(tmp_path, *, engine="torch", custom=False):
    cfg, construct = setup_source(tmp_path, engine=engine, custom=custom)
    current = construct()
    with pytest.raises(KeyboardInterrupt):
        current.fit(run=policy(tmp_path / "old" / "run"), on_checkpoint=interrupt_after(2))
    source, resources = relocated(tmp_path, cfg)
    return current, source, resources


def forbid(*_args, **_kwargs):
    raise AssertionError("must not load host weights, optimizer, or checkpoint tensors")


@pytest.mark.parametrize("engine", ["torch", "accelerate"])
@pytest.mark.parametrize("custom", [False, True])
def test_actual_offline_relocation_matches_uninterrupted_training(
    tmp_path, monkeypatch, engine, custom
):
    import arti.language_recipe as language

    cfg, construct = setup_source(tmp_path, engine=engine, custom=custom)
    full, split = construct(), construct()
    before = [parameter.detach().clone() for parameter in full.model.arti.parameters()]
    initial_rng = torch.get_rng_state().clone()
    shapes = []
    full.model.register_forward_pre_hook(
        lambda _model, _args, kwargs: shapes.append(tuple(kwargs["input_ids"].shape)),
        with_kwargs=True,
    )
    full.fit(run=policy(tmp_path / "full"))
    final_rng = torch.get_rng_state().clone()
    torch.set_rng_state(initial_rng)
    with pytest.raises(KeyboardInterrupt):
        split.fit(run=policy(tmp_path / "old" / "run"), on_checkpoint=interrupt_after(2))
    source, resources = relocated(tmp_path, cfg)
    checkpoint = Path(arti.plan_recipe_resume(source, **resources)["checkpoint"])
    original_checkpoint = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    original_recipe = (source / "resolved_recipe.json").read_bytes()
    captured, tokenizers, hosts = [], [], []
    resume, load_tokenizer, load_model = (
        arti.CausalLMRecipe.resume_run,
        language._load_tokenizer,
        language._load_model,
    )

    def capture(self, *args, **kwargs):
        captured.append(self)
        return resume(self, *args, **kwargs)

    def local_tokenizer(path, **kwargs):
        assert Path(path) == source / "tokenizer" and kwargs == {"local_files_only": True}
        tokenizers.append(path)
        return load_tokenizer(path, **kwargs)

    def local_host(config, *, source_override=None):
        assert source_override == str(resources["base_model"])
        hosts.append(source_override)
        return load_model(config, source_override=source_override)

    monkeypatch.setattr(arti.CausalLMRecipe, "resume_run", capture)
    monkeypatch.setattr(language, "_load_tokenizer", local_tokenizer)
    monkeypatch.setattr(language, "_load_model", local_host)
    result = arti.resume_recipe_run(
        source,
        output_dir=tmp_path / "continued",
        layer_factory=cfg.layer_factory,
        **resources,
    )
    recovered = captured[0]
    assert len(captured) == len(tokenizers) == len(hosts) == 1
    assert result["optimizer_steps"] == 4 and result["optimizer_updates_this_attempt"] == 2
    assert result["recovery"]["target_optimizer_step"] == 4
    assert result["recovery"]["resources"]["origins"]["source_manifest"] == "run-copy"
    assert result["recovery"]["resources"]["origins"]["cache_dir"] == "recovery-local"
    assert recovered.config.training.engine == engine
    assert recovered.task_fingerprint == full.task_fingerprint
    equal(full.model.arti._bundle().state_dict(), recovered.model.arti._bundle().state_dict())
    equal(full.session.optimizer.state_dict(), recovered.session.optimizer.state_dict())
    equal(full.session.scheduler.state_dict(), recovered.session.scheduler.state_dict())
    assert full.session.loss_history == recovered.session.loss_history
    assert full.train_data.state_dict() == recovered.train_data.state_dict()
    assert torch.equal(torch.get_rng_state(), final_rng)
    assert all(rows * length <= 48 for rows, length in shapes)
    assert len({rows for rows, _length in shapes}) > 1
    assert any(
        not torch.equal(a, b) for a, b in zip(before, full.model.arti.parameters(), strict=True)
    )
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == original_checkpoint
    assert (source / "resolved_recipe.json").read_bytes() == original_recipe
    journal = json.loads((tmp_path / "continued" / "run.json").read_text())
    assert journal["run"]["metadata"]["recovery_resources"] == result["recovery"]["resources"]


def test_plan_reports_relocations_and_missing_paths_without_loading(tmp_path, monkeypatch, capsys):
    _current, source, resources = interrupted(tmp_path, custom=True)
    import arti.language_recipe as language
    import arti.serialization as codec

    monkeypatch.setattr(language, "_load_model", forbid)
    monkeypatch.setattr(language, "_load_tokenizer", forbid)
    monkeypatch.setattr(torch.optim, "AdamW", forbid)
    monkeypatch.setattr(codec, "_load_safetensors", forbid)
    monkeypatch.setattr(arti.LayerFactoryConfig, "resolve", forbid)
    output = tmp_path / "plan.json"
    assert (
        main(
            [
                "recipe",
                "resume-run",
                str(source),
                "--plan-only",
                "--output",
                str(output),
                "--data",
                str(resources["data_path"]),
                "--base-model",
                str(resources["base_model"]),
                "--cache-dir",
                str(tmp_path / "cache"),
            ]
        )
        == 0
    )
    plan = json.loads(capsys.readouterr().out)
    assert json.loads(output.read_text()) == plan
    assert plan["resources"]["missing_local_paths"] == []
    assert plan["resources"]["tokenizer_path"] == str(source / "tokenizer")
    assert plan["loads_model_weights"] is plan["loads_checkpoint_tensors"] is False
    missing = arti.plan_recipe_resume(source)
    assert missing["resources"]["missing_local_paths"] == ["data_path"]
    with pytest.raises(ValueError, match="local resources.*data_path"):
        arti.resume_recipe_run(source)
    bad_base = {**resources, "base_model": tmp_path / "missing-native"}
    assert arti.plan_recipe_resume(source, **bad_base)["resources"]["missing_local_paths"] == [
        "base_model"
    ]
    with pytest.raises(ValueError, match="local resources.*base_model"):
        arti.resume_recipe_run(source, **bad_base)


@pytest.mark.parametrize("changed", ["data", "tokenizer", "source_manifest"])
def test_changed_recovery_semantics_rejected_before_host(tmp_path, monkeypatch, changed):
    _current, source, resources = interrupted(tmp_path)
    import arti.language_recipe as language

    if changed == "data":
        path = resources["data_path"]
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        rows[0]["messages"][-1]["content"] = "green"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    elif changed == "source_manifest":
        path = source / "data_source_manifest.json"
        payload = json.loads(path.read_text())
        payload["sources"][0]["sha256"] = "0" * 64
        path.write_text(json.dumps(payload))
    else:
        from test_language_recipe import tokenizer

        changed_tokenizer = tokenizer()
        changed_tokenizer.add_tokens(["fresh"])
        changed_tokenizer.save_pretrained(source / "tokenizer")
    monkeypatch.setattr(language, "_load_model", forbid)
    monkeypatch.setattr(torch.optim, "AdamW", forbid)
    with pytest.raises(ValueError, match="source_manifest|dataset/tokenizer"):
        arti.resume_recipe_run(source, **resources)


def test_cli_recovery_explicit_factory_and_warm_cache(tmp_path, capsys):
    current, source, resources = interrupted(tmp_path, custom=True)
    from test_recipe_factory import REFERENCE

    moved_cache = source.parent / "data" / "tokens"
    manifest = source / "data_source_manifest.json"
    kwargs = tmp_path / "factory.json"
    kwargs.write_text(json.dumps({"width": 4}))
    output = tmp_path / "result.json"
    assert (
        main(
            [
                "recipe",
                "resume-run",
                str(source),
                "--run-dir",
                str(tmp_path / "cli-run"),
                "--data",
                str(resources["data_path"]),
                "--base-model",
                str(resources["base_model"]),
                "--cache-dir",
                str(moved_cache),
                "--source-manifest",
                str(manifest),
                "--layer-factory",
                REFERENCE,
                "--factory-kwargs-json",
                str(kwargs),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result == json.loads(output.read_text())
    assert result["optimizer_updates_this_attempt"] == 2 and result["optimizer_steps"] == 4
    prepared = json.loads((tmp_path / "cli-run" / "plan.json").read_text())
    assert prepared["task_recipe"]["sha256"] == current.task_fingerprint
    assert prepared["data_preparation"]["encoded_samples"] == 0
    assert result["recovery"]["resources"]["origins"]["cache_dir"] == "caller"
    assert arti.torch.resume_recipe_run is arti.resume_recipe_run


def test_resources_inside_run_relocate_without_data_override(tmp_path):
    from test_language_recipe import base_model, config, tokenizer

    old = tmp_path / "old"
    old.mkdir()
    cfg = config(old)
    cfg = replace(cfg, cache_dir=old / "cache")
    current = arti.CausalLMRecipe(cfg, model=base_model(), tokenizer=tokenizer())
    with pytest.raises(KeyboardInterrupt):
        current.fit(run=policy(old / "run"), on_checkpoint=interrupt_after(2))
    run = old / "run"
    shutil.copy2(cfg.data_path, run / "chat.jsonl")
    cfg = replace(cfg, data_path=run / "chat.jsonl", cache_dir=run / "cache")
    (run / "resolved_recipe.json").write_text(json.dumps(cfg.to_dict()))
    native = tmp_path / "native"
    base_model().save_pretrained(native)
    moved = tmp_path / "moved"
    shutil.move(str(run), moved)
    shutil.rmtree(old)
    plan = arti.plan_recipe_resume(moved, base_model=native)
    assert plan["resources"]["data_path"] == str(moved / "chat.jsonl")
    assert plan["resources"]["cache_dir"] == str(moved / "cache")
    assert plan["resources"]["origins"]["data_path"] == "relocated"
    result = arti.resume_recipe_run(moved, base_model=native, output_dir=tmp_path / "result")
    assert result["optimizer_updates_this_attempt"] == 2


def test_saved_tokenizer_absence_keeps_pinned_loading_fallback(tmp_path, monkeypatch):
    _current, source, resources = interrupted(tmp_path)
    import arti.language_recipe as language
    from test_language_recipe import tokenizer

    shutil.rmtree(source / "tokenizer")
    calls = []

    def pinned(path, **kwargs):
        calls.append((path, kwargs))
        return tokenizer()

    monkeypatch.setattr(language, "_load_tokenizer", pinned)
    result = arti.resume_recipe_run(source, output_dir=tmp_path / "continued", **resources)
    assert result["optimizer_steps"] == 4
    assert calls == [("example/tiny-qwen", {"revision": "cpu-fixture", "local_files_only": True})]
    assert result["recovery"]["resources"]["origins"]["tokenizer_path"] == "pinned-model"


def test_moved_terminal_best_finalize_and_completed_noop(tmp_path, monkeypatch):
    cfg, construct = setup_source(tmp_path)
    current = construct()
    run = policy(
        tmp_path / "old" / "run",
        evaluation_steps=1,
        validation=arti.ARTIValidationConfig(patience=1),
    )
    monkeypatch.setattr(
        current, "export", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("export"))
    )
    with pytest.raises(OSError, match="export"):
        current.fit(run=run, validation_fn=lambda _session: {"loss": 1.0})
    source, resources = relocated(tmp_path, cfg)
    plan = arti.plan_recipe_resume(source, **resources)
    assert plan["training_terminal_reason"] == "early_stopping" and plan["action"] == "finalize"
    captured = []
    resume = arti.CausalLMRecipe.resume_run

    def capture(self, *args, **kwargs):
        captured.append(self)
        return resume(self, *args, **kwargs)

    monkeypatch.setattr(arti.CausalLMRecipe, "resume_run", capture)
    result = arti.resume_recipe_run(
        source,
        output_dir=tmp_path / "finalized",
        validation_fn=lambda _session: {"loss": 1.0},
        **resources,
    )
    assert (
        result["optimizer_updates_this_attempt"] == 0 and result["selection"]["selected_step"] == 1
    )
    equal(current.session.optimizer.state_dict(), captured[0].session.optimizer.state_dict())
    equal(current.model.arti._bundle().state_dict(), captured[0].model.arti._bundle().state_dict())
    equal(current.train_data.state_dict(), result["data_cursor"])
    equal(result["data_cursor"], captured[0].session._data_state)
    import arti.language_recipe as language

    monkeypatch.setattr(language, "_load_model", forbid)
    monkeypatch.setattr(language, "_load_tokenizer", forbid)
    monkeypatch.setattr(torch.optim, "AdamW", forbid)
    noop = arti.resume_recipe_run(tmp_path / "finalized", data_path=tmp_path / "missing")
    assert noop["optimizer_updates_this_attempt"] == 0 and len(captured) == 1
    assert noop["recovery"]["resources"]["missing_local_paths"] == ["data_path"]


def test_classification_helper_remains_compatible_and_rejects_causal_resources(tmp_path):
    from test_recipe_recovery import recipe

    current = recipe("classification", tmp_path / "data", steps=3)
    with pytest.raises(KeyboardInterrupt):
        current.fit(run=policy(tmp_path / "source"), on_checkpoint=interrupt_after(1))
    with pytest.raises(ValueError, match="only to causal-LM"):
        arti.plan_recipe_resume(tmp_path / "source", data_path=tmp_path / "other.jsonl")
    result = arti.resume_recipe_run(tmp_path / "source", output_dir=tmp_path / "continued")
    assert result["optimizer_steps"] == 3 and result["optimizer_updates_this_attempt"] == 2


@pytest.mark.parametrize("engine", ["torch", "accelerate"])
def test_second_recovery_inherits_native_base_and_keeps_absolute_target(
    tmp_path, monkeypatch, engine
):
    cfg, construct = setup_source(tmp_path, engine=engine)
    full, split = construct(), construct()
    initial_rng = torch.get_rng_state().clone()
    full.fit(run=policy(tmp_path / "full"))
    final_rng = torch.get_rng_state().clone()
    torch.set_rng_state(initial_rng)
    with pytest.raises(KeyboardInterrupt):
        split.fit(run=policy(tmp_path / "old" / "run"), on_checkpoint=interrupt_after(2))
    source, resources = relocated(tmp_path, cfg)
    with pytest.raises(KeyboardInterrupt):
        arti.resume_recipe_run(
            source,
            output_dir=tmp_path / "first-attempt",
            on_checkpoint=interrupt_after(3),
            **resources,
        )
    plan = arti.plan_recipe_resume(tmp_path / "first-attempt")
    assert plan["target_optimizer_step"] == 4 and plan["remaining_optimizer_steps"] == 1
    assert plan["resources"]["base_model"] == str(resources["base_model"])
    assert plan["resources"]["origins"]["base_model"] == "recorded"
    captured = []
    resume = arti.CausalLMRecipe.resume_run

    def capture(self, *args, **kwargs):
        captured.append(self)
        return resume(self, *args, **kwargs)

    monkeypatch.setattr(arti.CausalLMRecipe, "resume_run", capture)
    result = arti.resume_recipe_run(
        tmp_path / "first-attempt",
        output_dir=tmp_path / "second-attempt",
    )
    assert result["optimizer_steps"] == 4 and result["optimizer_updates_this_attempt"] == 1
    assert result["recovery"]["resources"]["base_model"] == str(resources["base_model"])
    recovered = captured[0]
    equal(full.model.arti._bundle().state_dict(), recovered.model.arti._bundle().state_dict())
    equal(full.session.optimizer.state_dict(), recovered.session.optimizer.state_dict())
    equal(full.session.scheduler.state_dict(), recovered.session.scheduler.state_dict())
    assert full.session.loss_history == recovered.session.loss_history
    assert full.train_data.state_dict() == recovered.train_data.state_dict()
    assert torch.equal(torch.get_rng_state(), final_rng)
