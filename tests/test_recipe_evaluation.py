from __future__ import annotations

import copy
from dataclasses import asdict, replace
import json
import random

import pytest
import torch

import arti
from arti.recipe_evaluation import (
    CausalLMEvaluationConfig,
    score_causal_lm_predictions,
    select_generation_examples,
)


def prediction(text, target, identifier="row"):
    return {
        "id": identifier,
        "target": target,
        "generation": {
            "text": text,
            "token_ids": [1],
            "generated_tokens": 1,
            "finish_reason": "stop",
        },
    }


@pytest.mark.parametrize(
    "text,target,matched",
    [
        ("42.0", "42", True),
        ("+1,000.00", "1e3", True),
        ("-.5", "-0.50", True),
        ("The answer is 42", "42", False),
        ("7 then 42", "42", False),
        ("```42```", "42", False),
        ("42 kg", "42", False),
        ("4,2", "42", False),
        ("NaN", "42", False),
        ("Infinity", "42", False),
        ("43", "42", False),
        ("1e99999999999999999999999999999", "1", False),
    ],
)
def test_numeric_scoring_uses_whole_reply_and_preserves_literal_metric(text, target, matched):
    row = prediction(text, target)
    original = copy.deepcopy(row)
    result = score_causal_lm_predictions([row], comparison="numeric")
    assert result["metrics"]["generation_numeric_exact_match"] == float(matched)
    assert result["metrics"]["generation_exact_match"] == float(text.strip() == target.strip())
    assert row == original and result["predictions"][0]["generation"] == original["generation"]


def test_offline_scoring_counts_actual_denominator_and_truncation():
    rows = [prediction("42", "42", "a"), prediction("wrong", "7", "b")]
    rows[1]["generation"]["finish_reason"] = "length"
    report = arti.score_causal_lm_predictions(rows)
    assert report["metrics"]["generated_samples"] == 2
    assert report["metrics"]["generation_score"] == 0.5
    assert report["metrics"]["generation_truncated_samples"] == 1
    assert arti.torch.score_causal_lm_predictions is arti.score_causal_lm_predictions


@pytest.mark.parametrize(
    "kwargs",
    [
        {"generate_samples": True},
        {"generate_samples": -1},
        {"generate_samples": 1.2},
        {"sample_seed": True},
        {"sample_seed": -1},
        {"sampling": "random"},
        {"comparison": "extract_last_number"},
    ],
)
def test_profile_rejects_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        CausalLMEvaluationConfig(**kwargs)


def make_recipe(tmp_path, profile=None, *, steps=4):
    pytest.importorskip("transformers")
    from test_language_recipe import config, base_model, tokenizer

    cfg = config(tmp_path, evaluation=profile or CausalLMEvaluationConfig())
    cfg = replace(cfg, training=replace(cfg.training, steps=steps))
    return arti.CausalLMRecipe(cfg, model=base_model(), tokenizer=tokenizer())


def echo_generation(_model, tok, conversations, _kwargs, _config):
    return [
        {
            "text": messages[-1]["content"].split()[-1],
            "token_ids": tok(messages[-1]["content"].split()[-1])["input_ids"],
            "generated_tokens": 1,
            "finish_reason": "stop",
        }
        for messages in conversations
    ]


def test_seeded_samples_are_stable_by_id_and_do_not_consume_rng(tmp_path):
    current = make_recipe(tmp_path)
    examples = current.data.splits["validation"].examples
    profile = CausalLMEvaluationConfig(generate_samples=2, sampling="seeded", sample_seed=5)
    state = torch.get_rng_state().clone(), random.getstate()
    a = select_generation_examples(examples, profile)
    b = select_generation_examples(tuple(reversed(examples)), profile)
    assert [item.id for item in a] == [item.id for item in b]
    assert torch.equal(state[0], torch.get_rng_state()) and state[1] == random.getstate()


def test_configured_generation_attaches_results_to_correct_id(tmp_path, monkeypatch):
    import arti.language_recipe as module

    current = make_recipe(
        tmp_path, CausalLMEvaluationConfig(generate_samples=2, sampling="seeded", sample_seed=5)
    )
    observed = []

    def capture(model, tok, conversations, kwargs, config):
        observed.extend(conversations)
        return echo_generation(model, tok, conversations, kwargs, config)

    monkeypatch.setattr(module, "_generate_chat", capture)
    report = current.evaluate(output=tmp_path / "evaluation.json")
    assert report["generated_samples"] == 2 and report["generation_score"] == 1
    assert all(messages[-1]["role"] == "user" for messages in observed)
    rows = [
        json.loads(line)
        for line in (tmp_path / "evaluation.predictions.jsonl").read_text().splitlines()
    ]
    selected = {row["id"] for row in rows if "generation" in row}
    assert selected == set(report["generation_sample_ids"])
    planned = current.plan()["evaluation_plan"]
    assert planned["sample_ids"] == report["generation_sample_ids"]
    assert planned["max_generated_tokens_per_check"] == 2 * current.config.generation.max_new_tokens
    assert planned["cost_is_static_upper_bound"]
    assert all(row["exact_match"] for row in rows if "generation" in row)
    assert "generated_samples" not in current.evaluate(generate_samples=0)


def test_gold_forward_state_and_rng_do_not_leak_to_generation(tmp_path, monkeypatch):
    import arti.language_recipe as module
    from test_recipe_recovery import equal

    current = make_recipe(tmp_path, CausalLMEvaluationConfig(generate_samples=2))
    current.model.register_buffer("evaluation_counter", torch.tensor(0))
    owned = next(iter(current.model.arti.parameters()))
    original_forward = module._forward_batch
    weights = copy.deepcopy(current.model.arti._bundle().state_dict())
    moments = copy.deepcopy(current.session.optimizer.state_dict())
    rng, python_rng = torch.get_rng_state().clone(), random.getstate()
    cursor = current.train_data.state_dict()

    def teacher_forward(model, batch):
        result = original_forward(model, batch)
        with torch.no_grad():
            owned.add_(1)
            model.evaluation_counter.add_(1)
        torch.rand(2)
        random.random()
        return result

    def generation(model, tok, conversations, kwargs, config):
        assert model.evaluation_counter.item() == 0
        equal(weights, model.arti._bundle().state_dict())
        assert torch.equal(rng, torch.get_rng_state()) and python_rng == random.getstate()
        with torch.no_grad():
            owned.add_(2)
            model.evaluation_counter.add_(2)
        torch.rand(3)
        return echo_generation(model, tok, conversations, kwargs, config)

    monkeypatch.setattr(module, "_forward_batch", teacher_forward)
    monkeypatch.setattr(module, "_generate_chat", generation)
    assert current.evaluate()["generation_score"] == 1
    equal(weights, current.model.arti._bundle().state_dict())
    equal(moments, current.session.optimizer.state_dict())
    assert current.model.evaluation_counter.item() == 0 and current.model.training
    assert torch.equal(rng, torch.get_rng_state()) and python_rng == random.getstate()
    assert cursor == current.train_data.state_dict()


@pytest.mark.parametrize(
    "monitor,profile",
    [
        ("generation_score", CausalLMEvaluationConfig()),
        ("generation_numeric_exact_match", CausalLMEvaluationConfig(generate_samples=1)),
        ("typo", CausalLMEvaluationConfig()),
    ],
)
def test_builtin_monitor_error_occurs_before_optimizer_updates(tmp_path, monitor, profile):
    current = make_recipe(tmp_path, profile)
    run = arti.ARTITrainingRunConfig(
        output_dir=tmp_path / "bad",
        evaluation_steps=1,
        validation=arti.ARTIValidationConfig(monitor=monitor, mode="max"),
    )
    with pytest.raises(ValueError):
        current.fit(run=run)
    assert current.session.global_step == 0 and not (tmp_path / "bad").exists()


@pytest.mark.parametrize("with_selection", [False, True])
def test_final_numeric_evaluation_is_checked_before_any_updates(tmp_path, with_selection):
    current = make_recipe(
        tmp_path, CausalLMEvaluationConfig(generate_samples=1, comparison="numeric")
    )
    run = arti.ARTITrainingRunConfig(
        output_dir=tmp_path / "bad",
        evaluation_steps=1,
        validation=arti.ARTIValidationConfig() if with_selection else None,
    )
    with pytest.raises(ValueError, match="numeric targets"):
        current.fit(run=run)
    assert current.session.global_step == 0 and not (tmp_path / "bad").exists()


def test_generation_metric_selects_earlier_best_and_early_stops(tmp_path, monkeypatch):
    import arti.language_recipe as module
    from test_recipe_recovery import equal

    current = make_recipe(tmp_path, CausalLMEvaluationConfig(generate_samples=2))
    first_state = None

    def generation(model, tok, conversations, kwargs, config):
        nonlocal first_state
        vector = torch.cat([p.detach().flatten() for p in model.arti.parameters()])
        if first_state is None:
            first_state = vector.clone()
        results = echo_generation(model, tok, conversations, kwargs, config)
        if not torch.equal(vector, first_state):
            for result in results:
                result["text"] = "wrong"
        return results

    monkeypatch.setattr(module, "_generate_chat", generation)
    run = arti.ARTITrainingRunConfig(
        output_dir=tmp_path / "run",
        evaluation_steps=1,
        checkpoint_steps=1,
        validation=arti.ARTIValidationConfig(monitor="generation_score", mode="max", patience=1),
    )
    result = current.fit(run=run)
    assert result["stop_reason"] == "early_stopping" and result["optimizer_steps"] == 2
    assert (
        result["selection"]["selected_step"] == 1 and result["validation"]["generation_score"] == 1
    )
    last = copy.deepcopy(current.model.arti._bundle().state_dict())
    from test_language_recipe import base_model

    deployed = arti.CausalLMPredictor.from_pretrained(result["model"], model=base_model())
    assert deployed.evaluation == current.config.evaluation
    with current.session.checkpoint_context(result["selected_checkpoint"]):
        equal(deployed.model.arti._bundle().state_dict(), current.model.arti._bundle().state_dict())
    equal(last, current.model.arti._bundle().state_dict())


def test_builtin_profile_resume_identity_is_separate_from_task_identity(tmp_path, monkeypatch):
    import arti.language_recipe as module
    from test_language_recipe import base_model, tokenizer

    current = make_recipe(tmp_path, CausalLMEvaluationConfig(generate_samples=1))
    monkeypatch.setattr(module, "_generate_chat", echo_generation)
    run = arti.ARTITrainingRunConfig(
        output_dir=tmp_path / "run",
        evaluation_steps=1,
        validation=arti.ARTIValidationConfig(monitor="generation_score", mode="max"),
    )
    result = current.fit(steps=1, run=run)
    changed = arti.CausalLMRecipe(
        replace(current.config, evaluation=CausalLMEvaluationConfig(generate_samples=2)),
        model=base_model(),
        tokenizer=tokenizer(),
    )
    assert changed.task_fingerprint == current.task_fingerprint
    with pytest.raises(ValueError, match="evaluation profile"):
        changed.session.load_checkpoint(result["checkpoint"])
    changed.load_model(result["model"])
    assert changed.evaluate()["generated_samples"] == 2


def test_config_read_roundtrip_and_default_no_generation(tmp_path):
    pytest.importorskip("transformers")
    from test_language_recipe import config

    cfg = config(
        tmp_path,
        evaluation=CausalLMEvaluationConfig(
            generate_samples=2, sampling="seeded", comparison="numeric"
        ),
    )
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(cfg.to_dict()))
    assert arti.CausalLMRecipeConfig.read(path).evaluation == cfg.evaluation
    assert asdict(CausalLMEvaluationConfig()) == {
        "generate_samples": 0,
        "sampling": "first",
        "sample_seed": 17,
        "comparison": "exact_text",
    }
    assert arti.torch.CausalLMEvaluationConfig is arti.CausalLMEvaluationConfig


def test_generated_validation_keeps_exact_interrupted_training_trajectory(tmp_path, monkeypatch):
    import arti.language_recipe as module
    from test_language_recipe import base_model, tokenizer
    from test_recipe_recovery import equal

    monkeypatch.setattr(module, "_generate_chat", echo_generation)
    profile = CausalLMEvaluationConfig(generate_samples=2, sampling="seeded", sample_seed=5)
    full = make_recipe(tmp_path, profile)
    current = arti.CausalLMRecipe(full.config, model=base_model(), tokenizer=tokenizer())
    policy = arti.ARTITrainingRunConfig(
        output_dir=tmp_path / "full",
        checkpoint_steps=1,
        evaluation_steps=1,
        validation=arti.ARTIValidationConfig(monitor="generation_score", mode="max"),
    )
    full.fit(run=policy)

    def stop(session, _path):
        if session.global_step == 2:
            raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        current.fit(run=replace(policy, output_dir=tmp_path / "first"), on_checkpoint=stop)
    restored = arti.CausalLMRecipe(full.config, model=base_model(), tokenizer=tokenizer())
    result = restored.resume_run(tmp_path / "first", output_dir=tmp_path / "restored")
    equal(full.model.state_dict(), restored.model.state_dict())
    equal(full.session.optimizer.state_dict(), restored.session.optimizer.state_dict())
    assert full.session.loss_history == restored.session.loss_history
    assert full.train_data.state_dict() == restored.train_data.state_dict()
    assert result["optimizer_updates_this_attempt"] == 2
    assert result["validation"]["generation_sample_ids"] == full.evaluate()["generation_sample_ids"]


def test_finalization_reuses_profile_without_new_updates(tmp_path, monkeypatch):
    import arti.language_recipe as module
    from test_language_recipe import base_model, tokenizer
    from test_recipe_recovery import equal

    monkeypatch.setattr(module, "_generate_chat", echo_generation)
    current = make_recipe(tmp_path, CausalLMEvaluationConfig(generate_samples=2), steps=1)
    monkeypatch.setattr(
        current, "export", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("export"))
    )
    with pytest.raises(OSError):
        current.fit(run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "source"))
    state = copy.deepcopy(current.model.state_dict())
    restored = arti.CausalLMRecipe(current.config, model=base_model(), tokenizer=tokenizer())
    result = restored.resume_run(tmp_path / "source", output_dir=tmp_path / "finalized")
    assert (
        result["optimizer_updates_this_attempt"] == 0
        and result["validation"]["generated_samples"] == 2
    )
    equal(state, restored.model.state_dict())


def test_caller_validation_requires_explicit_callback_on_continuation(tmp_path):
    from test_language_recipe import base_model, tokenizer

    current = make_recipe(tmp_path)
    policy = arti.ARTITrainingRunConfig(
        output_dir=tmp_path / "source", evaluation_steps=1, validation=arti.ARTIValidationConfig()
    )
    result = current.fit(steps=1, run=policy, validation_fn=lambda _session: {"loss": 1.0})
    restored = arti.CausalLMRecipe(current.config, model=base_model(), tokenizer=tokenizer())
    with pytest.raises(ValueError, match="caller-supplied validation"):
        restored.fit(
            steps=1,
            resume_from_checkpoint=result["checkpoint"],
            run=replace(policy, output_dir=tmp_path / "invalid"),
        )
    before = restored.session.global_step
    report = restored.fit(
        steps=1,
        run=replace(policy, output_dir=tmp_path / "valid"),
        validation_fn=lambda _session: {"loss": 1.0},
    )
    assert report["optimizer_steps"] == before + 1


def test_cli_uses_configured_generation_and_asset_profile(tmp_path, monkeypatch, capsys):
    import arti.language_recipe as module
    from arti.cli import main
    from test_language_recipe import base_model, tokenizer

    tok = tokenizer()
    monkeypatch.setattr(module, "_load_model", lambda *_args, **_kwargs: base_model())
    monkeypatch.setattr(module, "_load_tokenizer", lambda *_args, **_kwargs: tok)
    monkeypatch.setattr(module, "_generate_chat", echo_generation)
    current = make_recipe(
        tmp_path, CausalLMEvaluationConfig(generate_samples=2, sampling="seeded"), steps=1
    )
    config_path = tmp_path / "recipe.json"
    config_path.write_text(json.dumps(current.config.to_dict()))
    assert main(["recipe", "train", str(config_path), "--run-dir", str(tmp_path / "run")]) == 0
    trained = json.loads(capsys.readouterr().out)
    assert trained["validation"]["generated_samples"] == 2
    assert main(["recipe", "evaluate", str(config_path), "--model", trained["model"]]) == 0
    assert json.loads(capsys.readouterr().out)["generated_samples"] == 2
    assert (
        main(["recipe", "evaluate", trained["model"], "--data", str(current.config.data_path)]) == 0
    )
    assert json.loads(capsys.readouterr().out)["generated_samples"] == 2
    assert (
        main(
            [
                "recipe",
                "evaluate",
                trained["model"],
                "--data",
                str(current.config.data_path),
                "--generate-samples",
                "0",
            ]
        )
        == 0
    )
    assert "generated_samples" not in json.loads(capsys.readouterr().out)
