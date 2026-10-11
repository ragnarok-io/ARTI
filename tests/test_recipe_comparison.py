from __future__ import annotations

import copy
import json

import pytest
import torch

import arti
from arti.cli import main
from arti.recipe_data import read_jsonl
from arti.recipe_workflow import write_json, write_jsonl


def reply(identifier, text, target="42", *, tokens=1, input_id="same-input", finish="stop"):
    return {
        "id": identifier,
        "target": target,
        "input_sha256": input_id,
        "generation": {
            "text": text,
            "token_ids": [1] * tokens,
            "generated_tokens": tokens,
            "finish_reason": finish,
        },
    }


def receipt(path, rows, **overrides):
    metadata = {
        "objective": "native-causal-lm-loss",
        "weighting": "supervised_tokens",
        "dataset_sha256": "same-encoded-data",
        "task_recipe_sha256": path.stem,
        "samples": len(rows),
        "supervised_tokens": 12,
        "loss": 2.0,
        "perplexity": 7.4,
        "teacher_forced_token_accuracy": 0.2,
        "generation_config": {"max_new_tokens": 8, "batch_size": 2, "do_sample": False},
        "input_config": {"tokenizer_sha256": "same-tokenizer", "template_kwargs": {}},
        "evaluation": {"comparison": "exact_text", "sampling": "first", "sample_seed": 17},
        **overrides,
    }
    write_json(path, metadata)
    write_jsonl(path.with_suffix(".predictions.jsonl"), rows)
    return path


def test_pairing_is_by_id_rescored_and_does_not_mutate_inputs():
    candidate = [
        reply("c", "wrong"),
        reply("a", "42", tokens=2),
        reply("d", "wrong"),
        reply("b", "42"),
    ]
    baseline = [
        reply("a", "wrong", tokens=4, finish="length"),
        reply("b", "42"),
        reply("c", "42"),
        reply("d", "wrong"),
    ]
    candidate[0]["generation_match"] = True
    before = copy.deepcopy((candidate, baseline))
    result = arti.compare_causal_lm_predictions(candidate, baseline)
    assert (candidate, baseline) == before
    assert [row["id"] for row in result["predictions"]] == ["a", "b", "c", "d"]
    assert result["outcomes"] == {
        "improved": 1,
        "regressed": 1,
        "both_correct": 1,
        "both_incorrect": 1,
    }
    assert result["metrics"]["candidate"]["generation_score"] == 0.5
    assert result["delta"]["generation_score"] == 0
    assert result["delta"]["generated_tokens"] == -2
    assert result["delta"]["generation_truncated_samples"] == -1
    assert result["input_identity"]["fully_verified"]
    assert result["changed_reply_samples"] == 2
    assert result["predictions"][0]["baseline"]["generation"] == baseline[0]["generation"]
    assert arti.torch.compare_causal_lm_predictions is arti.compare_causal_lm_predictions


def test_partial_pairing_is_explicit_and_reports_actual_coverage():
    candidate = [reply("a", "42"), reply("candidate-only", "42")]
    baseline = [reply("a", "wrong"), reply("baseline-only", "wrong")]
    with pytest.raises(ValueError, match="IDs differ"):
        arti.compare_causal_lm_predictions(candidate, baseline)
    result = arti.compare_causal_lm_predictions(candidate, baseline, pairing="intersection")
    assert result["paired_samples"] == 1 and result["outcomes"]["improved"] == 1
    assert result["coverage"]["candidate"] == {
        "generated_samples": 2,
        "paired_fraction": 0.5,
        "excluded_ids": ["candidate-only"],
    }
    assert result["coverage"]["baseline"]["excluded_ids"] == ["baseline-only"]


@pytest.mark.parametrize(
    "change,match",
    [({"target": "7"}, "targets differ"), ({"input_sha256": "other"}, "inputs differ")],
)
def test_same_id_must_refer_to_the_same_known_input_and_target(change, match):
    baseline = reply("a", "42")
    candidate = {**baseline, **change}
    with pytest.raises(ValueError, match=match):
        arti.compare_causal_lm_predictions([candidate], [baseline])


def test_legacy_input_and_missing_token_counts_are_not_invented():
    baseline = reply("a", "wrong")
    baseline.pop("input_sha256")
    baseline["generation"].pop("generated_tokens")
    result = arti.compare_causal_lm_predictions([reply("a", "42")], [baseline])
    assert result["input_identity"]["unrecorded_samples"] == 1
    assert not result["input_identity"]["fully_verified"]
    assert result["metrics"]["baseline"]["generated_tokens"] is None
    assert "generated_tokens" not in result["delta"]


def test_numeric_rule_remains_whole_response_and_literal_scores_remain_separate():
    result = arti.compare_causal_lm_predictions(
        [reply("a", "42.0"), reply("b", "The answer is 42")],
        [reply("a", "wrong"), reply("b", "42")],
        comparison="numeric",
    )
    assert result["outcomes"]["improved"] == result["outcomes"]["regressed"] == 1
    assert result["metrics"]["candidate"]["generation_exact_match"] == 0
    assert result["metrics"]["candidate"]["generation_score"] == 0.5


def test_receipts_preserve_sources_conditions_and_write_separate_pairs(tmp_path, monkeypatch):
    import arti.language_recipe as module

    def fail(*_args, **_kwargs):
        raise AssertionError("comparison must not load a model or run generation")

    monkeypatch.setattr(module, "_load_model", fail)
    monkeypatch.setattr(module, "_generate_chat", fail)
    baseline = receipt(
        tmp_path / "baseline.json", [reply("a", "wrong"), {"id": "no-generation", "target": "7"}]
    )
    candidate = receipt(
        tmp_path / "candidate.json",
        [reply("a", "42"), {"id": "no-generation", "target": "7"}],
        loss=1.0,
    )
    inputs = {
        path: path.read_bytes()
        for path in (
            baseline,
            candidate,
            baseline.with_suffix(".predictions.jsonl"),
            candidate.with_suffix(".predictions.jsonl"),
        )
    }
    output = tmp_path / "comparison.json"
    result = arti.compare_causal_lm_evaluations(candidate, baseline, output=output)
    assert result["paired_samples"] == 1
    assert result["conditions"]["generation_config"]["status"] == "matched"
    assert result["conditions"]["input_config"]["status"] == "matched"
    assert result["teacher_forced_comparison"]["delta"]["loss"] == -1
    assert result["sources"]["candidate"]["report_metadata"]["task_recipe_sha256"] == "candidate"
    assert all(path.read_bytes() == content for path, content in inputs.items())
    saved = json.loads(output.read_text())
    assert "predictions" not in saved and saved["outcomes"]["improved"] == 1
    assert list(read_jsonl(output.with_suffix(".pairs.jsonl"))) == result["predictions"]
    assert arti.torch.compare_causal_lm_evaluations is arti.compare_causal_lm_evaluations


@pytest.mark.parametrize(
    "override",
    [{"dataset_sha256": "different"}, {"supervised_tokens": 13}, {"weighting": "batch_mean"}],
)
def test_token_metric_deltas_are_not_reported_for_different_denominators(tmp_path, override):
    baseline = receipt(tmp_path / "baseline.json", [reply("a", "wrong")])
    candidate = receipt(tmp_path / "candidate.json", [reply("a", "42")], **override)
    result = arti.compare_causal_lm_evaluations(candidate, baseline)
    assert result["outcomes"]["improved"] == 1
    assert not result["teacher_forced_comparison"]["comparable"]
    assert result["teacher_forced_comparison"]["delta"] == {}


def test_legacy_receipts_and_different_budgets_are_disclosed(tmp_path):
    baseline = receipt(tmp_path / "baseline.json", [reply("a", "wrong")])
    candidate = receipt(
        tmp_path / "candidate.json", [reply("a", "42")], generation_config={"max_new_tokens": 32}
    )
    result = arti.compare_causal_lm_evaluations(candidate, baseline)
    assert result["conditions"]["generation_config"]["status"] == "different"
    metadata = json.loads(baseline.read_text())
    for key in ("generation_config", "input_config", "evaluation"):
        metadata.pop(key)
    write_json(baseline, metadata)
    result = arti.compare_causal_lm_evaluations(candidate, baseline)
    assert all(value["status"] == "unrecorded" for value in result["conditions"].values())


def test_different_declared_scoring_rules_require_an_explicit_common_rule(tmp_path):
    baseline = receipt(tmp_path / "baseline.json", [reply("a", "wrong")])
    candidate = receipt(
        tmp_path / "candidate.json", [reply("a", "42.0")], evaluation={"comparison": "numeric"}
    )
    with pytest.raises(ValueError, match="shared comparison"):
        arti.compare_causal_lm_evaluations(candidate, baseline)
    assert (
        arti.compare_causal_lm_evaluations(candidate, baseline, comparison="numeric")["outcomes"][
            "improved"
        ]
        == 1
    )


def test_cli_writes_durable_comparison_without_models(tmp_path, capsys):
    candidate = receipt(tmp_path / "candidate.json", [reply("a", "42")])
    baseline = receipt(tmp_path / "baseline.json", [reply("a", "wrong")])
    output = tmp_path / "comparison.json"
    assert (
        main(
            [
                "recipe",
                "compare-evaluations",
                str(candidate),
                "--baseline",
                str(baseline),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["outcomes"]["improved"] == 1 and "predictions" not in result
    assert output.exists() and output.with_suffix(".pairs.jsonl").exists()


def test_comparison_does_not_replace_its_source_report(tmp_path):
    candidate = receipt(tmp_path / "candidate.json", [reply("a", "42")])
    baseline = receipt(tmp_path / "baseline.json", [reply("a", "wrong")])
    original = candidate.read_bytes()
    with pytest.raises(ValueError, match="input receipts"):
        arti.compare_causal_lm_evaluations(candidate, baseline, output=candidate)
    assert candidate.read_bytes() == original


def test_native_qwen_training_export_offline_reload_and_paired_receipts(tmp_path):
    pytest.importorskip("transformers")
    from test_language_recipe import base_model, config, tokenizer

    native = tmp_path / "native"
    base_model().save_pretrained(native)
    cfg = config(tmp_path, evaluation=arti.CausalLMEvaluationConfig(generate_samples=2))
    current = arti.CausalLMRecipe(cfg, model=base_model(), tokenizer=tokenizer())
    baseline_path, candidate_path = tmp_path / "baseline.json", tmp_path / "candidate.json"
    current.evaluate("test", output=baseline_path)
    before = [parameter.detach().clone() for parameter in current.model.arti.parameters()]
    fitted = current.fit(steps=2, run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "run"))
    assert any(
        not torch.equal(a, b) for a, b in zip(before, current.model.arti.parameters(), strict=True)
    )
    current.evaluate("test", output=candidate_path)
    compared = arti.compare_causal_lm_evaluations(candidate_path, baseline_path)
    assert compared["paired_samples"] == 2 and compared["input_identity"]["fully_verified"]
    assert compared["conditions"]["generation_config"]["status"] == "matched"
    packaged = arti.package_causal_lm_deployment(
        fitted["model"], tmp_path / "package", include_base=True, base_model=native
    )
    loaded = arti.CausalLMPredictor.from_pretrained(packaged["directory"], local_files_only=True)
    records = [row for row in read_jsonl(cfg.data_path) if row["split"] == "test"]
    reloaded_path = tmp_path / "reloaded.json"
    loaded.evaluate_records(records, batch_size=cfg.batch_size, output=reloaded_path)
    parity = arti.compare_causal_lm_evaluations(reloaded_path, candidate_path)
    assert parity["changed_reply_samples"] == 0 and parity["delta"]["generation_score"] == 0
    assert parity["input_identity"]["fully_verified"]
