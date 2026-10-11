from __future__ import annotations

import json
from pathlib import Path
import shutil

import pytest

import arti
from arti.cli import main
from arti.recipe_data import json_fingerprint
from arti.recipe_workflow import write_jsonl
from test_custom_task_workflow import forbid
from test_task_prediction_run import interrupt, loaded, trained


def config(kind="regression", **kwargs):
    return arti.PredictionScoringConfig(
        f"examples.prediction_scorers:{kind}", f"{kind}-original-units-v1", **kwargs
    )


def job(tmp_path, *, kind="custom", size=2, partial=False):
    current, asset, rows = trained(tmp_path, kind)
    data = tmp_path / "requests.jsonl"
    write_jsonl(data, rows)
    run = arti.PredictionRunConfig(tmp_path / "job", batch_size=size)
    kwargs = {"on_batch": interrupt} if partial else {}
    try:
        arti.run_prediction(asset, data, run, predictor=loaded(asset, kind), **kwargs)
    except KeyboardInterrupt:
        if not partial:
            raise
    return current, asset, rows, run.output_dir


@pytest.mark.parametrize("kind", ["custom", "classification"])
def test_actual_results_score_after_deleting_model_without_replay(tmp_path, monkeypatch, kind):
    _current, asset, rows, run = job(tmp_path, kind=kind)
    before = {
        str(path.relative_to(run)): path.read_bytes() for path in run.rglob("*") if path.is_file()
    }
    Path(asset).unlink()
    (run / "predictions.jsonl").unlink()
    import arti.recipes as recipes

    monkeypatch.setattr(recipes, "load_recipe_predictor", forbid)
    monkeypatch.setattr(arti.TaskRecipe, "from_config", forbid)
    result = arti.score_prediction_run(
        run,
        config("classification" if kind == "classification" else "regression"),
        output=tmp_path / "scores.json",
    )
    assert result["samples"] == len(rows) == 7 and result["coverage"] == 1
    assert [item["id"] for item in result["scores"]] == [row["id"] for row in rows]
    assert [item["target"] for item in result["scores"]] == rows
    assert result["sha256"] == json_fingerprint(
        {key: value for key, value in result.items() if key != "sha256"}
    )
    assert json.loads((tmp_path / "scores.json").read_text()) == result
    for path in run.rglob("*"):
        if path.is_file():
            assert path.read_bytes() == before[str(path.relative_to(run))]
    assert not (run / "predictions.jsonl").exists()
    assert arti.torch.score_prediction_run is arti.score_prediction_run
    assert arti.torch.compare_prediction_runs is arti.compare_prediction_runs


def test_original_target_ids_reordered_extra_and_custom_field(tmp_path):
    _current, _asset, rows, run = job(tmp_path)
    targets = tmp_path / "targets.jsonl"
    labels = [{"sample_id": row["id"], "target": row["target"]} for row in reversed(rows)]
    labels.append({"sample_id": "unused", "target": 42})
    write_jsonl(targets, labels)
    result = arti.score_prediction_run(run, config(target_id_field="sample_id"), targets=targets)
    assert result["targets"]["unused_ids"] == ["unused"]
    assert [item["target"]["target"] for item in result["scores"]] == [
        row["target"] for row in rows
    ]
    labels[0]["target"] += 1
    write_jsonl(targets, labels)
    changed = arti.score_prediction_run(run, config(target_id_field="sample_id"), targets=targets)
    assert result["targets"]["records_sha256"] != changed["targets"]["records_sha256"]
    assert result["metrics"] != changed["metrics"]


@pytest.mark.parametrize("error", ["missing", "duplicate"])
def test_invalid_original_labels_fail_before_scorer(tmp_path, monkeypatch, error):
    _current, _asset, rows, run = job(tmp_path)
    targets = tmp_path / "targets.jsonl"
    write_jsonl(targets, rows[:-1] if error == "missing" else rows + rows[:1])
    monkeypatch.setattr(arti.PythonFactoryConfig, "resolve", forbid)
    with pytest.raises(ValueError, match="missing|distinct"):
        arti.score_prediction_run(run, config(), targets=targets)


def test_partial_scoring_uses_committed_prefix_not_report_or_final_merge(tmp_path):
    _current, _asset, rows, run = job(tmp_path, partial=True)
    with pytest.raises(ValueError, match="unfinished"):
        arti.score_prediction_run(run, config())
    result = arti.score_prediction_run(run, config(), allow_partial=True)
    assert result["samples"] == 2 and result["coverage"] == 2 / 7
    assert result["source"]["uncommitted_records"] == 5
    assert result["targets"]["unused_ids"] == [row["id"] for row in rows[2:]]
    assert arti.inspect_prediction(run)["action"] == "resume_prediction"
    stored = arti.score_prediction_run(
        run, config(), allow_partial=True, output=run / "evaluation" / "scores.json"
    )
    assert stored == result
    assert arti.inspect_prediction(run)["committed_records"] == 2


def test_pairing_rechecks_same_inputs_and_reduces_on_shared_samples(tmp_path):
    _current, asset, rows, candidate = job(tmp_path / "candidate")
    baseline = tmp_path / "baseline"
    data = tmp_path / "baseline-input.jsonl"
    write_jsonl(data, list(reversed(rows)))
    arti.run_prediction(
        asset, data, arti.PredictionRunConfig(baseline, batch_size=4), predictor=loaded(asset)
    )
    result = arti.compare_prediction_runs(
        candidate, baseline, config(), output=tmp_path / "comparison.json"
    )
    assert result["paired_samples"] == 7
    assert result["delta"] == {"mae": 0, "mse": 0}
    assert (
        result["sources"]["candidate"]["plan_sha256"]
        != result["sources"]["baseline"]["plan_sha256"]
    )
    partial = tmp_path / "partial"
    with pytest.raises(KeyboardInterrupt):
        arti.run_prediction(
            asset,
            data,
            arti.PredictionRunConfig(partial, batch_size=3),
            predictor=loaded(asset),
            on_batch=interrupt,
        )
    with pytest.raises(ValueError, match="IDs differ"):
        arti.compare_prediction_runs(candidate, partial, config(), allow_partial=True)
    compared = arti.compare_prediction_runs(
        candidate, partial, config(), allow_partial=True, pairing="intersection"
    )
    assert compared["paired_samples"] == 3
    assert compared["coverage"]["candidate"]["paired_fraction"] == 3 / 7
    assert compared["coverage"]["baseline"]["uncommitted_records"] == 4
    assert compared["delta"] == {"mae": 0, "mse": 0}
    assert len(compared["coverage"]["candidate"]["excluded_ids"]) == 4
    rows[0]["features"][0] += 1
    write_jsonl(data, rows)
    different = tmp_path / "different"
    arti.run_prediction(asset, data, arti.PredictionRunConfig(different), predictor=loaded(asset))
    with pytest.raises(ValueError, match="inputs differ"):
        arti.compare_prediction_runs(candidate, different, config())


def test_pairing_uses_json_identity_not_python_numeric_coercion(tmp_path):
    _current, asset, rows, _run = job(tmp_path)
    data = tmp_path / "typed-input.jsonl"
    paths = []
    for index, value in enumerate((1, True)):
        records = [{**row, "application_marker": value} for row in rows]
        write_jsonl(data, records)
        path = tmp_path / f"typed-{index}"
        arti.run_prediction(asset, data, arti.PredictionRunConfig(path), predictor=loaded(asset))
        paths.append(path)
    with pytest.raises(ValueError, match="inputs differ"):
        arti.compare_prediction_runs(*paths, config())


def test_empty_committed_prefix_cannot_be_scored(tmp_path):
    _current, asset, rows, _run = job(tmp_path)
    data = tmp_path / "requests.jsonl"
    write_jsonl(data, rows)
    empty = tmp_path / "empty"
    arti.run_prediction(asset, data, arti.PredictionRunConfig(empty), stop_requested=lambda: True)
    with pytest.raises(ValueError, match="at least one committed"):
        arti.score_prediction_run(empty, config(), allow_partial=True)


def mutating_scorer(pairs):
    result = {
        "metrics": {"description": "no direction guessed", "score": 1},
        "scores": [
            {"id": pair["id"], "nested": {"observed": pair["target"]["target"]}}
            for pair in reversed(pairs)
        ],
    }
    pairs[0]["target"]["target"] = -100
    pairs[0]["prediction"]["prediction"] = -100
    return result


def invalid_scorer(pairs, *, failure):
    if failure == "exception":
        raise RuntimeError("task scoring failed")
    if failure == "nonfinite":
        return {"metrics": {"value": float("nan")}, "scores": []}
    return {"metrics": {}, "scores": [{"id": pairs[0]["id"]}]}


def test_scorer_mutation_and_output_order_preserve_original_evidence(tmp_path):
    _current, _asset, rows, run = job(tmp_path)
    result = arti.score_prediction_run(
        run,
        arti.PredictionScoringConfig(
            "test_prediction_scoring:mutating_scorer", "explicit-test-rule"
        ),
    )
    assert [row["target"] for row in result["scores"]] == rows
    assert [row["id"] for row in result["scores"]] == [row["id"] for row in rows]
    compared = arti.compare_prediction_runs(
        run,
        run,
        arti.PredictionScoringConfig(
            "test_prediction_scoring:mutating_scorer", "explicit-test-rule"
        ),
    )
    assert compared["delta"] == {"score": 0}
    assert "description" not in compared["delta"]


@pytest.mark.parametrize("failure", ["exception", "nonfinite", "coverage"])
def test_scorer_failure_keeps_existing_report_and_predictions(tmp_path, failure):
    _current, _asset, _rows, run = job(tmp_path)
    output = tmp_path / "scores.json"
    accepted = arti.score_prediction_run(run, config(), output=output)
    content = output.read_bytes()
    scorer = arti.PredictionScoringConfig(
        arti.PythonFactoryConfig("test_prediction_scoring:invalid_scorer", {"failure": failure}),
        "invalid-rule",
    )
    with pytest.raises((RuntimeError, ValueError)):
        arti.score_prediction_run(run, scorer, output=output)
    assert output.read_bytes() == content
    assert arti.score_prediction_run(run, config()) == accepted


def test_atomic_write_and_sources_are_not_overwritten(tmp_path, monkeypatch):
    _current, _asset, _rows, run = job(tmp_path)
    output = tmp_path / "scores.json"
    arti.score_prediction_run(run, config(), output=output)
    content = output.read_bytes()
    import arti.serialization as serialization

    monkeypatch.setattr(
        serialization,
        "_replace_atomic_file",
        lambda *_args: (_ for _ in ()).throw(OSError("publication failed")),
    )
    with pytest.raises(OSError, match="publication failed"):
        arti.score_prediction_run(run, config(), output=output)
    assert output.read_bytes() == content
    with pytest.raises(ValueError, match="overwrite"):
        arti.score_prediction_run(run, config(), output=run / "run.json")
    with pytest.raises(ValueError, match="overwrite"):
        arti.score_prediction_run(
            run, config(), targets=run / "input.jsonl", output=run / "input.jsonl"
        )


def test_relocated_source_and_cli_only_need_scorer_code(tmp_path, monkeypatch, capsys):
    _current, asset, _rows, run = job(tmp_path)
    Path(asset).unlink()
    relocated = tmp_path / "moved"
    shutil.move(str(run), str(relocated))
    import arti.recipes as recipes

    monkeypatch.setattr(recipes, "load_recipe_predictor", forbid)
    output = tmp_path / "new-reports" / "cli.json"
    common = [
        "--scorer",
        "examples.prediction_scorers:regression",
        "--scorer-id",
        "regression-v1",
        "--output",
        str(output),
    ]
    assert main(["recipe", "score-prediction", str(relocated), *common]) == 0
    assert json.loads(capsys.readouterr().out)["samples"] == 7
    assert (
        main(
            ["recipe", "compare-predictions", str(relocated), "--baseline", str(relocated), *common]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["paired_samples"] == 7


def reply_scorer(pairs):
    from arti.recipe_evaluation import score_causal_lm_predictions

    result = score_causal_lm_predictions(
        [
            {"id": pair["id"], "generation": pair["prediction"], "target": pair["target"]["target"]}
            for pair in pairs
        ]
    )
    return {
        "metrics": result["metrics"],
        "scores": [
            {"id": row["id"], "match": row["generation_match"]} for row in result["predictions"]
        ],
    }


def test_legacy_language_run_uses_same_offline_scoring_without_language_defaults(tmp_path):
    from test_prediction_run import (
        loaded as load_language,
        requests,
        trained as train_language,
        policy,
    )

    _current, asset = train_language(tmp_path)
    data = tmp_path / "requests.jsonl"
    rows = requests(3)
    write_jsonl(data, rows)
    arti.run_causal_lm_prediction(
        asset, data, policy(tmp_path / "language"), predictor=load_language(asset)
    )
    targets = tmp_path / "targets.jsonl"
    write_jsonl(targets, [{"id": row["id"], "target": "red"} for row in rows])
    result = arti.score_prediction_run(
        tmp_path / "language",
        arti.PredictionScoringConfig("test_prediction_scoring:reply_scorer", "whole-reply-v1"),
        targets=targets,
    )
    assert result["samples"] == 3
    assert "generation_score" in result["metrics"] and "mse" not in result["metrics"]
