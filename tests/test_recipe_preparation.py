from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

import pytest
import torch

import arti
from arti.cli import main
from arti.language_recipe import CausalLMRecipe, CausalLMRecipeConfig, _prepare_chat_dataset

pytest.importorskip("transformers")
from test_language_recipe import base_model, config, records, tokenizer, write
from test_recipe_recovery import equal


def prepared(cfg, tok=None):
    return _prepare_chat_dataset(cfg, tok if tok is not None else tokenizer())


def same_data(left, right):
    assert left.manifest(include_ids=True) == right.manifest(include_ids=True)
    for role, split in left.splits.items():
        for a, b in zip(split.examples, right.splits[role].examples, strict=True):
            assert a.id == b.id and a.messages == b.messages and a.prompt_length == b.prompt_length
            equal(a.input_ids, b.input_ids)
            equal(a.labels, b.labels)


def test_warm_cache_preserves_tokens_and_does_not_render_or_tokenize(tmp_path, monkeypatch):
    cfg = replace(config(tmp_path), cache_dir=tmp_path / "cache")
    tok = tokenizer()
    cold = prepared(cfg, tok)

    def fail(*_args, **_kwargs):
        raise AssertionError("warm data must not encode")

    monkeypatch.setattr(type(tok), "__call__", fail)
    monkeypatch.setattr(tok, "apply_chat_template", fail)
    warm = prepared(cfg, tok)
    same_data(cold, warm)
    assert cold.preparation["encoded_samples"] == 10
    assert warm.preparation["encoded_samples"] == 0 and warm.preparation["cache_hit"]
    assert "cache_directory" not in warm.manifest()


@pytest.mark.parametrize(
    "change", ["source", "template", "vocab", "kwargs", "length", "overlength"]
)
def test_encoding_identity_changes_invalidate_cache(tmp_path, change):
    cfg = replace(config(tmp_path), cache_dir=tmp_path / "cache")
    tok = tokenizer()
    before = prepared(cfg, tok)
    if change == "source":
        rows = records()
        rows[0]["messages"][-1]["content"] = "blue"
        write(cfg.data_path, rows)
    elif change == "template":
        tok.chat_template = tok.chat_template.replace("m['content']", "'please ' + m['content']")
    elif change == "vocab":
        tok.add_tokens(["unseen-word"])
    elif change == "kwargs":
        cfg = replace(cfg, template_kwargs={"enable_thinking": True})
    elif change == "length":
        cfg = replace(cfg, max_length=128)
    else:
        cfg = replace(cfg, overlength="drop")
    after = prepared(cfg, tok)
    assert not after.preparation["cache_hit"]
    assert before.preparation["cache_key"] != after.preparation["cache_key"]
    assert prepared(cfg, tok).preparation["cache_hit"]


def test_cache_key_excludes_padding_batch_order_and_optimizer(tmp_path):
    cfg = replace(config(tmp_path), cache_dir=tmp_path / "cache")
    before = prepared(cfg)
    changed = replace(cfg, pad_to_multiple_of=16, batch_size=4, seed=33, group_by_length=False)
    after = prepared(changed)
    assert after.preparation["cache_hit"]
    assert before.preparation["cache_key"] == after.preparation["cache_key"]
    assert before.splits["train"].fingerprint == after.splits["train"].fingerprint
    assert after.splits["train"].batches(4)[0]["input_ids"].shape[1] % 16 == 0


def test_drop_records_and_empty_optional_split_survive_warm_load(tmp_path):
    cfg = replace(config(tmp_path), cache_dir=tmp_path / "cache", overlength="drop")
    rows = [row for row in records() if row["split"] != "test"]
    rows.append(
        {
            "id": "long",
            "split": "train",
            "messages": [
                {"role": "user", "content": "color " * 100},
                {"role": "assistant", "content": "red"},
            ],
        }
    )
    write(cfg.data_path, rows)
    cold, warm = prepared(cfg), prepared(cfg)
    same_data(cold, warm)
    assert cold.dropped["train"] == ["long"]
    assert not len(warm.splits["test"])
    assert warm.preparation["cache_hit"]


def test_failed_source_preparation_does_not_publish_cache(tmp_path):
    cfg = replace(config(tmp_path), cache_dir=tmp_path / "cache", max_length=1)
    with pytest.raises(ValueError, match="no answer was silently truncated"):
        prepared(cfg)
    assert not cfg.cache_dir.exists()


def test_incomplete_or_invalid_cache_rebuilds_from_source(tmp_path, monkeypatch):
    import arti.recipe_chat_cache as module

    cfg = replace(config(tmp_path), cache_dir=tmp_path / "cache")
    original = module._replace_atomic_file
    monkeypatch.setattr(
        module, "_replace_atomic_file", lambda *_args: (_ for _ in ()).throw(OSError("publish"))
    )
    with pytest.raises(OSError, match="publish"):
        prepared(cfg)
    assert not list(cfg.cache_dir.glob("*/manifest.json"))
    monkeypatch.setattr(module, "_replace_atomic_file", original)
    cold = prepared(cfg)
    (Path(cold.preparation["cache_directory"]) / "tokens.safetensors").write_bytes(b"incomplete")
    rebuilt = prepared(cfg)
    same_data(cold, rebuilt)
    assert rebuilt.preparation["cache_status"] == "invalid"
    assert prepared(cfg).preparation["cache_hit"]


def test_prepare_api_cli_do_not_construct_model_or_optimizer(tmp_path, monkeypatch, capsys):
    import arti.language_recipe as module

    cfg = replace(config(tmp_path), cache_dir=tmp_path / "cache")
    tok = tokenizer()
    monkeypatch.setattr(module, "_load_tokenizer", lambda *_args, **_kwargs: tok)
    def fail(*_args, **_kwargs):
        raise AssertionError("model or optimizer")
    monkeypatch.setattr(module, "_load_model", fail)
    monkeypatch.setattr(CausalLMRecipe, "__init__", fail)
    monkeypatch.setattr(torch.optim, "AdamW", fail)
    cfg_path = tmp_path / "recipe.json"
    cfg_path.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    assert (
        main(["recipe", "prepare-data", str(cfg_path), "--output", str(tmp_path / "prepared.json")])
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["weights_loaded"] is False and report["optimizer_created"] is False
    assert report["model_allocation"] == "none"
    assert arti.prepare_causal_lm_data(cfg, tokenizer=tok)["data_preparation"]["cache_hit"]
    assert arti.torch.prepare_causal_lm_data is arti.prepare_causal_lm_data


def test_cache_location_is_runtime_only_and_resume_matches_no_cache(tmp_path):
    cfg = config(tmp_path)
    uncached = CausalLMRecipe(cfg, model=base_model(), tokenizer=tokenizer())
    cached_cfg = replace(cfg, cache_dir=tmp_path / "cache")
    cached = CausalLMRecipe(cached_cfg, model=base_model(), tokenizer=tokenizer())
    assert cached.task_fingerprint == uncached.task_fingerprint
    assert "cache_dir" not in json.dumps(cached.task_contract)
    uncached.fit(run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "full", checkpoint_steps=1))

    def interrupt(session, _path):
        if session.global_step == 1:
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        cached.fit(
            run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "first", checkpoint_steps=1),
            on_checkpoint=interrupt,
        )
    restored = CausalLMRecipe(cached_cfg, model=base_model(), tokenizer=tokenizer())
    result = restored.resume_run(tmp_path / "first", output_dir=tmp_path / "restored")
    equal(uncached.model.state_dict(), restored.model.state_dict())
    equal(uncached.session.optimizer.state_dict(), restored.session.optimizer.state_dict())
    assert uncached.session.loss_history == restored.session.loss_history
    assert uncached.train_data.state_dict() == restored.train_data.state_dict()
    assert result["data_preparation"]["cache_hit"]
    assert (tmp_path / "restored" / "data_preparation.json").exists()


def test_config_cache_path_is_relative_to_recipe_and_nullable(tmp_path):
    cfg_path = tmp_path / "recipe.json"
    cfg = config(tmp_path).to_dict()
    cfg["data"]["cache_dir"] = "cache"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    restored = CausalLMRecipeConfig.read(cfg_path)
    assert restored.cache_dir == tmp_path / "cache"
    cfg["data"]["cache_dir"] = None
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")
    assert CausalLMRecipeConfig.read(cfg_path).cache_dir is None


@pytest.mark.parametrize("value", [True, 4, ""])
def test_config_rejects_invalid_cache_location(tmp_path, value):
    with pytest.raises(ValueError, match="cache_dir"):
        replace(config(tmp_path), cache_dir=value)
