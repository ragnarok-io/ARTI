from __future__ import annotations

from dataclasses import replace
import json

import pytest
import torch

import arti
from arti.cli import main
from arti.recipe_chat import ChatBatches, ChatExample, ChatSplit
from arti.recipe_workflow import write_jsonl
from test_recipe_recovery import equal, interrupt_after, policy


def split():
    examples = []
    for index, length in enumerate([8, 8, 8, 48, 8, 8]):
        ids = torch.full((length,), index + 4, dtype=torch.long)
        ids[-1] = 3
        labels = ids.clone()
        labels[:-2] = -100
        examples.append(
            ChatExample(
                str(index),
                ({"role": "user", "content": "color"}, {"role": "assistant", "content": "red"}),
                ids,
                labels,
                length - 2,
            )
        )
    return ChatSplit(tuple(examples), "mixed", 3, 8)


def batches(max_tokens=None):
    return ChatBatches(
        split(), batch_size=6, seed=17, group_by_length=True, max_tokens_per_batch=max_tokens
    )


def config(directory, *, budget=64, steps=4, engine="torch"):
    pytest.importorskip("transformers")
    from test_language_recipe import config as original, records

    directory.mkdir(parents=True, exist_ok=True)
    cfg = original(directory)
    rows = records()
    for index, row in enumerate(rows):
        row["messages"][-2]["content"] += " old" * [0, 3, 24, 0, 12][index % 5]
    write_jsonl(cfg.data_path, rows)
    return replace(
        cfg,
        batch_size=5,
        max_tokens_per_batch=budget,
        training=replace(cfg.training, steps=steps, engine=engine, gradient_accumulation_steps=2),
    )


def recipe(cfg, *, dropout=False):
    from test_language_recipe import base_model, tokenizer

    model = base_model()
    if dropout:
        for layer in model.model.layers:
            layer.self_attn.attention_dropout = 0.2
    return arti.CausalLMRecipe(cfg, model=model, tokenizer=tokenizer())


def test_dense_budget_allows_more_short_rows_and_reduces_padding_without_packing():
    fixed, budget = batches(), batches(64)
    before = fixed.state_dict()
    original_plan, plan = fixed.plan(), budget.plan()
    assert fixed.state_dict() == before
    assert original_plan["padded_tokens_per_epoch"] == 288
    assert plan["padded_tokens_per_epoch"] == 88 and plan["padding_efficiency"] == 1.0
    assert plan["min_batch_rows"] == 1 and plan["max_batch_rows"] == 5
    assert plan["max_padded_tokens"] == 48
    observed = []
    for batch in budget:
        assert batch["input_ids"].numel() <= 64
        for ids, labels, mask in zip(
            batch["input_ids"], batch["labels"], batch["attention_mask"], strict=True
        ):
            identifier = str(int(ids[0]) - 4)
            expected = split().examples[int(identifier)]
            count = len(expected.input_ids)
            assert torch.equal(ids[:count], expected.input_ids)
            assert torch.equal(labels[:count], expected.labels)
            assert labels[count - 1] == 3 and labels[count:].eq(-100).all()
            assert mask[:count].all() and mask[count:].eq(0).all()
            observed.append(identifier)
    assert sorted(observed) == list(split().ids)


def test_variable_batch_sample_cursor_and_none_legacy_state():
    current = batches(64)
    iterator = iter(current)
    first = next(iterator)
    state = current.state_dict()
    restored = batches(64)
    restored.load_state_dict(state)
    expected, actual = list(iterator), list(restored)
    assert first["input_ids"].shape[0] == 1
    assert len(expected) == len(actual)
    for left, right in zip(expected, actual, strict=True):
        equal(left, right)
    assert current.state_dict() == restored.state_dict()
    with pytest.raises(ValueError, match="token budget"):
        batches(128).load_state_dict(state)
    legacy = batches().state_dict()
    assert "max_tokens_per_batch" not in legacy
    batches().load_state_dict(legacy)


def test_padding_alignment_and_single_example_error_before_host(tmp_path, monkeypatch):
    import arti.language_recipe as module
    from test_language_recipe import tokenizer

    cfg = config(tmp_path, budget=8)
    monkeypatch.setattr(
        module,
        "_load_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("host allocation")),
    )
    with pytest.raises(ValueError, match="record .*needs .*padded tokens alone"):
        arti.CausalLMRecipe(cfg, tokenizer=tokenizer())
    with pytest.raises(ValueError, match="48 padded tokens"):
        batches(47)


@pytest.mark.parametrize("engine", ["torch", "accelerate"])
def test_native_mixed_length_training_and_exact_budget_continuation(tmp_path, engine):
    if engine == "accelerate":
        pytest.importorskip("accelerate")
    cfg = config(tmp_path / "data", engine=engine)
    full, interrupted = recipe(cfg, dropout=True), recipe(cfg, dropout=True)
    shapes = []
    full.model.register_forward_pre_hook(
        lambda _model, _args, kwargs: shapes.append(tuple(kwargs["input_ids"].shape)),
        with_kwargs=True,
    )
    initial = [parameter.detach().clone() for parameter in full.model.arti.parameters()]
    initial_rng = torch.get_rng_state().clone()
    full.fit(run=policy(tmp_path / "full"))
    final_rng = torch.get_rng_state().clone()
    torch.set_rng_state(initial_rng)
    with pytest.raises(KeyboardInterrupt):
        interrupted.fit(run=policy(tmp_path / "partial"), on_checkpoint=interrupt_after(2))
    restored = recipe(cfg, dropout=True)
    result = restored.resume_run(tmp_path / "partial", run=policy(tmp_path / "restored"))
    assert result["optimizer_updates_this_attempt"] == 2
    equal(full.model.arti._bundle().state_dict(), restored.model.arti._bundle().state_dict())
    equal(full.session.optimizer.state_dict(), restored.session.optimizer.state_dict())
    equal(full.session.scheduler.state_dict(), restored.session.scheduler.state_dict())
    assert full.session.loss_history == restored.session.loss_history
    assert full.train_data.state_dict() == restored.train_data.state_dict()
    assert torch.equal(torch.get_rng_state(), final_rng)
    assert any(
        not torch.equal(a, b) for a, b in zip(initial, full.model.arti.parameters(), strict=True)
    )
    assert all(rows * length <= 64 for rows, length in shapes)
    assert len(set(rows for rows, _length in shapes)) > 1
    changed = recipe(replace(cfg, max_tokens_per_batch=128))
    with pytest.raises(ValueError, match="token budget"):
        changed.session.load_checkpoint(result["checkpoint"])


def test_teacher_forced_evaluation_parity_and_deployment_budget(tmp_path):
    from test_language_recipe import base_model, tokenizer

    cfg = config(tmp_path / "data")
    current = recipe(cfg)
    unlimited = recipe(replace(cfg, max_tokens_per_batch=None))
    assert unlimited.task_fingerprint == current.task_fingerprint
    actual, expected = current.evaluate("validation"), unlimited.evaluate("validation")
    assert actual["loss"] == pytest.approx(expected["loss"], abs=1e-6)
    assert actual["supervised_tokens"] == expected["supervised_tokens"]
    assert actual["batching_config"]["max_padded_tokens"] <= 64
    asset = current.export(tmp_path / "asset")
    loaded = arti.CausalLMPredictor.from_pretrained(
        asset, model=base_model(), tokenizer=tokenizer()
    )
    assert loaded.max_tokens_per_batch == 64
    from arti.recipe_data import read_jsonl

    records = [row for row in read_jsonl(cfg.data_path) if row["split"] == "validation"]
    report = loaded.evaluate_records(records, batch_size=5)
    assert report["loss"] == pytest.approx(actual["loss"], abs=1e-6)
    assert report["batching_config"]["max_tokens_per_batch"] == 64
    deployment = json.loads((asset.parent / "deployment.json").read_text())
    assert deployment["batching"]["max_tokens_per_batch"] == 64


def test_budget_is_not_a_preparation_cache_key_or_task_identity(tmp_path):
    from test_language_recipe import tokenizer

    cfg = replace(config(tmp_path / "data"), cache_dir=tmp_path / "prepared")
    first = arti.prepare_causal_lm_data(
        replace(cfg, max_tokens_per_batch=None), tokenizer=tokenizer()
    )
    second = arti.prepare_causal_lm_data(cfg, tokenizer=tokenizer())
    assert first["data_preparation"]["cache_key"] == second["data_preparation"]["cache_key"]
    assert (
        second["data_preparation"]["cache_hit"]
        and second["data_preparation"]["encoded_samples"] == 0
    )
    assert second["batching_plan"]["max_padded_tokens"] <= 64


def test_cli_config_roundtrip_and_budget_evaluation(tmp_path, monkeypatch, capsys):
    from test_language_recipe import base_model, tokenizer
    import arti.language_recipe as module

    cfg = config(tmp_path / "data", steps=2)
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(cfg.to_dict()))
    assert arti.CausalLMRecipeConfig.read(path) == cfg
    monkeypatch.setattr(module, "_load_model", lambda *_args, **_kwargs: base_model())
    monkeypatch.setattr(module, "_load_tokenizer", lambda *_args, **_kwargs: tokenizer())
    assert main(["recipe", "train", str(path), "--run-dir", str(tmp_path / "run")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["optimizer_steps"] == 2
    assert main(["recipe", "evaluate", result["model"], "--data", str(cfg.data_path)]) == 0
    assert json.loads(capsys.readouterr().out)["batching_config"]["max_padded_tokens"] <= 64


def test_preflight_collates_only_one_batch(tmp_path, monkeypatch):
    import arti.language_recipe as module

    current = recipe(config(tmp_path))
    calls = []
    original = module.collate_chat

    def capture(examples, **kwargs):
        batch = original(examples, **kwargs)
        calls.append(batch["input_ids"].numel())
        return batch

    monkeypatch.setattr(module, "collate_chat", capture)
    assert current.check()["ok"] and len(calls) == 1 and calls[0] <= 64


def test_native_package_reload_preserves_evaluation_budget(tmp_path):
    from test_language_recipe import base_model

    cfg = config(tmp_path / "data", steps=2)
    current = recipe(cfg)
    result = current.fit(run=policy(tmp_path / "run"))
    native = tmp_path / "native"
    base_model().save_pretrained(native)
    package = arti.package_causal_lm_deployment(
        result["model"], tmp_path / "package", include_base=True, base_model=native
    )
    loaded = arti.CausalLMPredictor.from_pretrained(package["directory"], local_files_only=True)
    assert loaded.max_tokens_per_batch == 64
    assert (
        arti.inspect_causal_lm_deployment(package["directory"])["batching"]["max_tokens_per_batch"]
        == 64
    )
