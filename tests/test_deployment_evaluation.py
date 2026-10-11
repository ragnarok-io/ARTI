from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import random
import shutil

import pytest
import torch

import arti
from arti.cli import main
from arti.language_recipe import _prepare_chat_dataset
from arti.recipe_data import read_jsonl

pytest.importorskip("transformers")
from test_language_recipe import base_model, config, records, tokenizer, write
from test_recipe_recovery import equal


def trained(tmp_path, *, engine="torch", factory=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = config(tmp_path, layer_factory=factory)
    cfg = replace(cfg, training=replace(cfg.training, steps=2, engine=engine))
    current = arti.CausalLMRecipe(cfg, model=base_model(), tokenizer=tokenizer())
    result = current.fit(
        run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "train", checkpoint_steps=1)
    )
    return current, Path(result["model"]).parent


def evaluation_config(current, tmp_path, rows=None):
    rows = (
        [{**row, "id": f"heldout-{index}"} for index, row in enumerate(records()[-2:])]
        if rows is None
        else rows
    )
    return replace(
        current.config,
        data_path=write(tmp_path / "heldout.jsonl", rows),
        cache_dir=tmp_path / "eval-cache",
        source_manifest=None,
        evaluation=arti.CausalLMEvaluationConfig(generate_samples=2, sampling="seeded"),
    )


def forbid_training(monkeypatch):
    import arti.language_recipe as language

    def fail(*_args, **_kwargs):
        raise AssertionError("independent evaluation must not create a trainer/optimizer/scheduler")

    monkeypatch.setattr(arti.CausalLMRecipe, "__init__", fail)
    monkeypatch.setattr(arti.ARTITrainingSession, "__init__", fail)
    monkeypatch.setattr(language, "trainer", fail)
    monkeypatch.setattr(torch.optim, "AdamW", fail)
    monkeypatch.setattr(torch.optim.lr_scheduler, "LambdaLR", fail)


@pytest.mark.parametrize("engine", ["torch", "accelerate"])
def test_new_test_only_data_is_evaluated_without_training_objects(tmp_path, monkeypatch, engine):
    if engine == "accelerate":
        pytest.importorskip("accelerate")
    current, asset = trained(tmp_path, engine=engine)
    cfg = evaluation_config(current, tmp_path)
    # Training-only choices cannot change the already exported architecture.
    cfg = replace(
        cfg,
        adapter=replace(cfg.adapter, rank=1),
        optimizer=replace(cfg.optimizer, weight_decay=0.7),
        training=replace(cfg.training, learning_rate=0.9),
    )
    native = base_model()
    tok = tokenizer()
    forbid_training(monkeypatch)
    rng, python_rng = torch.get_rng_state().clone(), random.getstate()
    report = arti.evaluate_causal_lm_deployment(
        asset, cfg, split="test", model=native, tokenizer=tok, output=tmp_path / "test.json"
    )
    assert report["ok"] and report["samples"] == 2
    assert report["task_recipe_sha256"] == current.task_fingerprint
    assert report["dataset"]["selected_splits"] == ["test"]
    assert report["dataset"]["splits"]["train"]["samples"] == 0
    assert report["dataset"]["splits"]["validation"]["samples"] == 0
    assert (
        not report["optimizer_created"]
        and not report["training_session_created"]
        and not report["training_state_loaded"]
    )
    assert report["execution"] == "read-only-deployment"
    assert report["generated_samples"] == 2 and report["supervised_tokens"] > 0
    equal(torch.get_rng_state(), rng)
    assert random.getstate() == python_rng
    rows = list(read_jsonl(tmp_path / "test.predictions.jsonl"))
    assert {row["id"] for row in rows} == {"heldout-0", "heldout-1"}
    assert arti.torch.evaluate_causal_lm_deployment is arti.evaluate_causal_lm_deployment


def test_split_selection_skips_encoding_other_splits_and_reuses_cache(tmp_path, monkeypatch):
    import arti.recipe_chat as chat

    current, asset = trained(tmp_path)
    rows = records()
    rows[0]["messages"][-2]["content"] = "color " * 1000
    cfg = evaluation_config(current, tmp_path, rows)
    predictor = arti.CausalLMPredictor.from_pretrained(
        asset, model=base_model(), tokenizer=tokenizer()
    )
    encoded, original = [], chat.encode_final_answer

    def encode(identifier, *args, **kwargs):
        assert identifier.startswith("test-")
        encoded.append(identifier)
        return original(identifier, *args, **kwargs)

    monkeypatch.setattr(chat, "encode_final_answer", encode)
    first = predictor.evaluate_recipe(cfg, split="test")
    assert len(encoded) == 2 and first["data_preparation"]["encoded_samples"] == 2
    assert first["dataset"]["source_dataset_sha256"] != first["dataset"]["sha256"]
    second = predictor.evaluate_recipe(cfg, split="test")
    assert len(encoded) == 2 and second["data_preparation"]["cache_hit"]
    assert second["data_preparation"]["encoded_samples"] == 0
    assert first["loss"] == second["loss"]
    assert first["generation_sample_ids"] == second["generation_sample_ids"]


def test_training_still_requires_train_and_validation(tmp_path):
    cfg = config(tmp_path)
    write(cfg.data_path, records()[-2:])
    with pytest.raises(ValueError, match="train and validation"):
        _prepare_chat_dataset(cfg, tokenizer())
    assert _prepare_chat_dataset(cfg, tokenizer(), split="test").manifest()["selected_splits"] == [
        "test"
    ]


def test_selected_preparation_keeps_full_history_final_answer_and_native_eos(tmp_path):
    cfg = config(tmp_path)
    messages = [
        {"role": "user", "content": "old color blue"},
        {"role": "assistant", "content": "blue"},
        {"role": "user", "content": "new color red"},
        {"role": "assistant", "content": "red"},
    ]
    write(cfg.data_path, [{"id": "history", "split": "test", "messages": messages}])
    tok = tokenizer()
    example = _prepare_chat_dataset(cfg, tok, split="test").splits["test"].examples[0]
    assert list(example.messages) == messages
    assert "blue" in tok.decode(example.input_ids[: example.prompt_length])
    assert example.labels[: example.prompt_length].eq(-100).all()
    assert example.labels[example.labels.ne(-100)].tolist() == tok("red", add_special_tokens=False)[
        "input_ids"
    ] + [tok.eos_token_id]


@pytest.mark.parametrize(
    "change",
    [
        "model",
        "template",
        "tokenizer",
        "source",
        "overlength",
        "missing-split",
        "numeric",
        "budget",
    ],
)
def test_invalid_evaluation_fails_before_host_loading(tmp_path, monkeypatch, change):
    import arti.language_recipe as language

    current, asset = trained(tmp_path)
    cfg = evaluation_config(current, tmp_path)
    tok = tokenizer()
    split = "test"
    if change == "model":
        cfg = replace(cfg, model=replace(cfg.model, revision="other"))
    elif change == "template":
        cfg = replace(cfg, template_kwargs={"enable_thinking": True})
    elif change == "tokenizer":
        tok.add_tokens(["other"])
    elif change == "source":
        cfg = replace(cfg, source_manifest=tmp_path / "invalid-manifest.json")
        cfg.source_manifest.write_text("{}", encoding="utf-8")
    elif change == "overlength":
        cfg = replace(cfg, max_length=1)
    elif change == "missing-split":
        split = "validation"
    elif change == "numeric":
        cfg = replace(cfg, evaluation=replace(cfg.evaluation, comparison="numeric"))
    else:
        cfg = replace(cfg, max_tokens_per_batch=1)

    def fail(*_args, **_kwargs):
        raise AssertionError("invalid data/contract must precede host allocation")

    monkeypatch.setattr(language, "_load_model", fail)
    with pytest.raises(ValueError):
        arti.evaluate_causal_lm_deployment(asset, cfg, split=split, tokenizer=tok)


def test_drop_policy_reports_excluded_ids_without_truncating_answers(tmp_path):
    current, asset = trained(tmp_path)
    rows = records()[-2:]
    rows.append(
        {
            "id": "long-test",
            "split": "test",
            "messages": [
                {"role": "user", "content": "color " * 100},
                {"role": "assistant", "content": "blue"},
            ],
        }
    )
    cfg = replace(evaluation_config(current, tmp_path, rows), overlength="drop")
    report = arti.evaluate_causal_lm_deployment(
        asset, cfg, split="test", model=base_model(), tokenizer=tokenizer()
    )
    assert report["samples"] == 2
    assert report["dataset"]["splits"]["test"]["dropped_ids"] == ["long-test"]
    assert report["dataset"]["splits"]["test"]["dropped_samples"] == 1
    assert report["data_config"]["overlength"] == "drop"


def test_helper_and_reused_predictor_produce_identical_receipts(tmp_path):
    current, asset = trained(tmp_path)
    cfg = evaluation_config(current, tmp_path)
    first = arti.evaluate_causal_lm_deployment(
        asset,
        cfg,
        split="test",
        model=base_model(),
        tokenizer=tokenizer(),
        output=tmp_path / "first.json",
    )
    predictor = arti.CausalLMPredictor.from_pretrained(
        asset, model=base_model(), tokenizer=tokenizer()
    )
    second = predictor.evaluate_recipe(cfg, split="test", output=tmp_path / "second.json")
    assert first["loss"] == second["loss"] and first["perplexity"] == second["perplexity"]
    assert first["dataset_sha256"] == second["dataset_sha256"]
    assert first["model_source"] == second["model_source"]
    assert list(read_jsonl(tmp_path / "first.predictions.jsonl")) == list(
        read_jsonl(tmp_path / "second.predictions.jsonl")
    )
    paired = arti.compare_causal_lm_evaluations(tmp_path / "first.json", tmp_path / "second.json")
    assert paired["paired_samples"] == 2
    assert paired["teacher_forced_comparison"]["delta"]["loss"] == 0
    assert paired["conditions"]["model_source"]["status"] == "matched"


def test_model_source_distinguishes_different_weights_with_the_same_task(tmp_path):
    current, first_asset = trained(tmp_path)
    cfg = evaluation_config(current, tmp_path)
    first = arti.evaluate_causal_lm_deployment(
        first_asset, cfg, split="test", model=base_model(), tokenizer=tokenizer()
    )
    result = current.fit(steps=1, run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "continued"))
    second = arti.evaluate_causal_lm_deployment(
        Path(result["model"]).parent, cfg, split="test", model=base_model(), tokenizer=tokenizer()
    )
    assert first["task_recipe_sha256"] == second["task_recipe_sha256"]
    assert first["model_source"]["manifest_sha256"] != second["model_source"]["manifest_sha256"]


def test_model_and_runtime_state_rng_and_modes_are_read_only(tmp_path):
    from arti.attachment_training import _rng_state
    from test_recipe_factory import factory

    current, asset = trained(tmp_path, factory=factory())
    cfg = evaluation_config(current, tmp_path)
    predictor = arti.CausalLMPredictor.from_pretrained(
        asset, model=base_model(), tokenizer=tokenizer(), layer_factory=factory()
    )
    predictor.model.train()
    predictor.model.register_buffer("observed_calls", torch.zeros((), dtype=torch.long))
    graphs = [
        module
        for module in predictor.model.arti._bundle().modules()
        if isinstance(module, arti.mechanisms.ProgramGraph)
    ]
    states = [copy.deepcopy(graph.state()) for graph in graphs]
    model_state = copy.deepcopy(predictor.model.state_dict())
    modes = [module.training for module in predictor.model.modules()]
    rng = _rng_state()

    def observe(model, _args, _output):
        model.observed_calls.add_(1)
        torch.rand(1)
        random.random()

    handle = predictor.model.register_forward_hook(observe)
    try:
        predictor.evaluate_recipe(cfg, split="test")
    finally:
        handle.remove()
    equal(model_state, predictor.model.state_dict())
    equal(rng, _rng_state())
    assert modes == [module.training for module in predictor.model.modules()]
    for state, graph in zip(states, graphs, strict=True):
        restored = graph.state()
        assert state.contract_fingerprint == restored.contract_fingerprint
        assert state.join_cursors == restored.join_cursors
        for before, after in zip(state.resources, restored.resources, strict=True):
            assert before.spec == after.spec
            assert (before.epoch, before.step_index, before.active_source) == (
                after.epoch,
                after.step_index,
                after.active_source,
            )
            equal(before.default_view.value, after.default_view.value)
            equal(before.active_view.value, after.active_view.value)


def test_no_generation_override_and_batched_token_budget(tmp_path, monkeypatch):
    import arti.language_recipe as language

    current, asset = trained(tmp_path)
    cfg = replace(evaluation_config(current, tmp_path), batch_size=8, max_tokens_per_batch=16)
    predictor = arti.CausalLMPredictor.from_pretrained(
        asset, model=base_model(), tokenizer=tokenizer()
    )
    monkeypatch.setattr(
        language,
        "_generate_chat",
        lambda *_args: (_ for _ in ()).throw(AssertionError("generation disabled")),
    )
    report = predictor.evaluate_recipe(cfg, split="test", generate_samples=0)
    assert "generated_samples" not in report
    assert report["batching_config"]["max_padded_tokens"] <= 16
    assert report["evaluation"]["generate_samples"] == 0


@pytest.mark.parametrize("custom", [False, True])
def test_cli_config_evaluation_uses_asset_not_training_factory(
    tmp_path, monkeypatch, capsys, custom
):
    import arti.language_recipe as language
    from test_recipe_factory import factory

    current, asset = trained(tmp_path, factory=factory() if custom else None)
    cfg = evaluation_config(current, tmp_path)
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(cfg.to_dict()), encoding="utf-8")
    monkeypatch.setattr(language, "_load_model", lambda *_args, **_kwargs: base_model())
    forbid_training(monkeypatch)
    assert (
        main(
            [
                "recipe",
                "evaluate",
                str(path),
                "--model",
                str(asset),
                "--split",
                "test",
                "--output",
                str(tmp_path / "report.json"),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["samples"] == 2 and report["generated_samples"] == 2
    assert report["execution"] == "read-only-deployment"
    assert not report["optimizer_created"]


def test_native_moved_package_evaluation_needs_no_training_data(tmp_path, monkeypatch):
    current, asset = trained(tmp_path)
    cfg = evaluation_config(current, tmp_path)
    base = tmp_path / "native"
    base_model().save_pretrained(base)
    result = arti.package_causal_lm_deployment(
        asset, tmp_path / "package", include_base=True, base_model=base
    )
    moved = tmp_path / "moved"
    shutil.copytree(result["directory"], moved)
    current.config.data_path.unlink()
    forbid_training(monkeypatch)
    report = arti.evaluate_causal_lm_deployment(moved, cfg, split="test", local_files_only=True)
    assert report["samples"] == 2 and report["supervised_tokens"] > 0


def test_source_manifest_is_full_source_but_receipt_names_selected_subset(tmp_path):
    from test_recipe_import import imported

    current, asset = trained(tmp_path)
    _declared, snapshot, cfg = imported(tmp_path / "import")
    cfg = replace(cfg, model=current.config.model)
    report = arti.evaluate_causal_lm_deployment(
        asset, cfg, split="test", model=base_model(), tokenizer=tokenizer()
    )
    assert report["samples"] == 2
    assert report["source_manifest_sha256"] == snapshot["manifest"]["sha256"]
    assert (
        report["dataset"]["source_dataset_sha256"]
        == snapshot["manifest"]["output"]["dataset_sha256"]
    )
    assert report["dataset"]["sha256"] != report["dataset"]["source_dataset_sha256"]
    assert report["dataset"]["selected_splits"] == ["test"]


def test_new_data_field_mapping_does_not_change_the_deployed_model(tmp_path):
    current, asset = trained(tmp_path)
    rows = [
        {"record": f"new-{index}", "role": "test", "turns": row["messages"]}
        for index, row in enumerate(records()[-2:])
    ]
    cfg = replace(
        evaluation_config(current, tmp_path, rows),
        fields=arti.ChatFields(id="record", split="role", messages="turns"),
    )
    report = arti.evaluate_causal_lm_deployment(
        asset,
        cfg,
        split="test",
        model=base_model(),
        tokenizer=tokenizer(),
        output=tmp_path / "remapped.json",
    )
    assert report["task_recipe_sha256"] == current.task_fingerprint
    assert report["dataset"]["splits"]["test"]["ids"] == ["new-0", "new-1"]
    assert report["data_config"]["fields"]["messages"] == "turns"


def test_custom_factory_is_explicit_and_missing_dependency_precedes_loading(tmp_path, monkeypatch):
    import arti.language_recipe as language
    from test_recipe_factory import factory

    current, asset = trained(tmp_path, factory=factory())
    cfg = replace(evaluation_config(current, tmp_path), layer_factory=None)

    def fail(*_args, **_kwargs):
        raise AssertionError("factory declaration must precede tokenizer/host loading")

    monkeypatch.setattr(language, "_load_tokenizer", fail)
    monkeypatch.setattr(language, "_load_model", fail)
    with pytest.raises(ValueError, match="layer_factory"):
        arti.evaluate_causal_lm_deployment(asset, cfg, split="test")
    report = arti.evaluate_causal_lm_deployment(
        asset,
        cfg,
        split="test",
        layer_factory=factory().reference,
        model=base_model(),
        tokenizer=tokenizer(),
    )
    assert report["samples"] == 2
