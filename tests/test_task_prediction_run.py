from __future__ import annotations

import copy
from dataclasses import replace
import json
from pathlib import Path
import shutil

import pytest
import torch

import arti
from arti.cli import main
from arti.recipe_data import read_jsonl
from arti.recipe_workflow import write_json, write_jsonl
from examples.custom_task_recipe import FACTORY
from test_custom_task_workflow import setup, forbid
from test_recipe_recovery import equal, policy


def trained(tmp_path, kind="custom"):
    if kind == "classification":
        from test_recipe_recovery import recipe
        from test_task_recipes import _records

        current = recipe("classification", tmp_path / "data", steps=2)
        rows = _records()[-3:]
    else:
        cfg, rows = setup(tmp_path / "data", steps=2)
        current = arti.TaskRecipe.from_config(cfg)
        rows = rows[-3:]
    asset = current.fit(run=policy(tmp_path / "training"))["model"]
    requests = [
        {**copy.deepcopy(rows[index % len(rows)]), "id": f"request-{index}"} for index in range(7)
    ]
    return current, asset, requests


def loaded(asset, kind="custom"):
    from arti.recipes import load_recipe_predictor

    return load_recipe_predictor(asset, task_factory=FACTORY if kind == "custom" else None)


def run_config(path, **kwargs):
    return arti.PredictionRunConfig(path, batch_size=2, predict_kwargs={"batch_size": 2}, **kwargs)


def interrupt(_report, _receipt):
    raise KeyboardInterrupt()


@pytest.mark.parametrize("kind", ["custom", "classification"])
def test_actual_nonlanguage_prediction_resume_preserves_raw_results_and_does_not_replay(
    tmp_path, monkeypatch, kind
):
    _current, asset, rows = trained(tmp_path, kind)
    predictor = loaded(asset, kind)
    expected = predictor.predict_records(rows, batch_size=2)
    original, calls = predictor.predict_records, []

    def predict(records, **kwargs):
        calls.append([row["id"] for row in records])
        return original(records, **kwargs)

    monkeypatch.setattr(predictor, "predict_records", predict)
    data = tmp_path / "requests.jsonl"
    write_jsonl(data, rows)
    config = run_config(tmp_path / "job")
    with pytest.raises(KeyboardInterrupt):
        arti.run_prediction(asset, data, config, predictor=predictor, on_batch=interrupt)
    assert arti.inspect_prediction(config.output_dir)["committed_records"] == 2
    data.unlink()
    result = arti.run_prediction(asset, None, config, resume=True, predictor=predictor)
    assert result["newly_predicted_records"] == 5 and result["reused_records"] == 2
    assert result["committed_batches"] == 4
    assert calls == [[row["id"] for row in rows[start : start + 2]] for start in range(0, 7, 2)]
    assert list(read_jsonl(result["predictions"])) == expected
    assert "generated_tokens" not in result
    for path in (config.output_dir / "parts").glob("*/receipt.json"):
        assert "generated_tokens" not in json.loads(path.read_text())
    assert arti.torch.run_prediction is arti.run_prediction
    assert arti.torch.inspect_prediction is arti.inspect_prediction


@pytest.mark.parametrize("kind", ["custom", "classification"])
def test_load_once_and_complete_finalize_without_model_or_factory(tmp_path, monkeypatch, kind):
    _current, asset, rows = trained(tmp_path, kind)
    import arti.recipes as recipes
    import arti.recipe_prediction as module

    predictor, loads = loaded(asset, kind), []

    def load(*_args, **_kwargs):
        loads.append(1)
        return predictor

    monkeypatch.setattr(recipes, "load_recipe_predictor", load)
    source = tmp_path / "requests.jsonl"
    write_jsonl(source, rows)
    config = run_config(tmp_path / "job")
    result = arti.run_prediction(
        asset, source, config, task_factory=FACTORY if kind == "custom" else None
    )
    assert len(loads) == 1 and result["status"] == "completed"
    monkeypatch.setattr(recipes, "load_recipe_predictor", forbid)
    monkeypatch.setattr(arti.PythonFactoryConfig, "resolve", forbid)
    monkeypatch.setattr(torch.optim, "AdamW", forbid)
    Path(result["predictions"]).unlink()
    with monkeypatch.context() as patch:
        patch.setattr(module, "_model_declaration", forbid)
        finished = arti.run_prediction(None, None, config, resume=True)
        assert finished["newly_predicted_records"] == 0 and finished["reused_records"] == 7
        assert arti.run_prediction(None, None, config, resume=True)["newly_predicted_records"] == 0


@pytest.mark.parametrize("failure", ["progress", "merge"])
def test_shared_journal_retains_committed_work_when_progress_or_merge_fails(
    tmp_path, monkeypatch, failure
):
    _current, asset, rows = trained(tmp_path)
    import arti.recipe_prediction as module

    predictor = loaded(asset)
    data = tmp_path / "requests.jsonl"
    write_jsonl(data, rows)
    config = run_config(tmp_path / "job")
    atomic, merge, fired = module._atomic_json, module._merge_predictions, []

    def write(path, payload):
        if Path(path).name == "run.json" and payload.get("committed_records") == 2 and not fired:
            fired.append(1)
            raise OSError("progress")
        return atomic(path, payload)

    if failure == "progress":
        monkeypatch.setattr(module, "_atomic_json", write)
    else:
        monkeypatch.setattr(
            module, "_merge_predictions", lambda *_args: (_ for _ in ()).throw(OSError("merge"))
        )
    with pytest.raises(OSError):
        arti.run_prediction(asset, data, config, predictor=predictor)
    progress = arti.inspect_prediction(config.output_dir)
    assert progress["committed_records"] == (2 if failure == "progress" else 7)
    monkeypatch.setattr(module, "_atomic_json", atomic)
    monkeypatch.setattr(module, "_merge_predictions", merge)
    calls, original = [], predictor.predict_records
    monkeypatch.setattr(
        predictor,
        "predict_records",
        lambda records, **kwargs: (calls.append(len(records)), original(records, **kwargs))[1],
    )
    result = arti.run_prediction(
        asset if failure == "progress" else None, None, config, resume=True, predictor=predictor
    )
    assert result["status"] == "completed"
    assert calls == ([2, 2, 1] if failure == "progress" else [])


def test_outer_partition_and_task_batch_options_are_independent_and_pinned(tmp_path, monkeypatch):
    _current, asset, rows = trained(tmp_path)
    predictor = loaded(asset)
    seen, original = [], predictor.predict_records
    monkeypatch.setattr(
        predictor,
        "predict_records",
        lambda records, **kwargs: (
            seen.append((len(records), kwargs)),
            original(records, **kwargs),
        )[1],
    )
    data = tmp_path / "requests.jsonl"
    write_jsonl(data, rows)
    config = arti.PredictionRunConfig(
        tmp_path / "job", batch_size=4, predict_kwargs={"batch_size": 2}
    )
    arti.run_prediction(asset, data, config, predictor=predictor)
    assert seen == [(4, {"batch_size": 2}), (3, {"batch_size": 2})]
    with pytest.raises(ValueError, match="option batch_size"):
        arti.run_prediction(
            asset, None, replace(config, predict_kwargs={"batch_size": 1}), resume=True
        )
    with pytest.raises(ValueError, match="batch_size differs"):
        arti.run_prediction(asset, None, replace(config, batch_size=3), resume=True)


def test_metadata_source_and_input_identity_checked_before_loading(tmp_path, monkeypatch):
    _current, asset, rows = trained(tmp_path)
    import arti.recipes as recipes
    import arti.serialization as serialization

    data = tmp_path / "requests.jsonl"
    write_jsonl(data, rows)
    config = run_config(tmp_path / "job", max_seconds=0)
    monkeypatch.setattr(recipes, "load_recipe_predictor", forbid)
    result = arti.run_prediction(asset, data, config, task_factory=FACTORY)
    assert result["status"] == "stopped" and result["newly_predicted_records"] == 0
    with monkeypatch.context() as patch:
        patch.setattr(serialization, "_load_safetensors", forbid)
        assert arti.inspect_prediction(config.output_dir)["loads_model_weights"] is False
    changed = tmp_path / "changed.jsonl"
    write_jsonl(changed, rows[:-1])
    with pytest.raises(ValueError, match="data differs"):
        arti.run_prediction(asset, changed, config, resume=True)
    with pytest.raises(ValueError, match="requires the pinned model"):
        arti.run_prediction(None, None, config, resume=True)


def test_relocated_run_and_model_do_not_need_original_data(tmp_path):
    _current, asset, rows = trained(tmp_path)
    data = tmp_path / "requests.jsonl"
    write_jsonl(data, rows)
    config = run_config(tmp_path / "job")
    with pytest.raises(KeyboardInterrupt):
        arti.run_prediction(asset, data, config, task_factory=FACTORY, on_batch=interrupt)
    data.unlink()
    model = tmp_path / "moved-model"
    shutil.move(str(Path(asset).parent), model)
    moved = tmp_path / "moved-job"
    shutil.move(str(config.output_dir), moved)
    result = arti.run_prediction(
        model, None, replace(config, output_dir=moved), resume=True, task_factory=FACTORY
    )
    assert result["committed_records"] == 7 and result["reused_records"] == 2


def test_generic_cli_and_model_free_finalize(tmp_path, capsys, monkeypatch):
    _current, asset, rows = trained(tmp_path)
    data, options = tmp_path / "requests.jsonl", tmp_path / "kwargs.json"
    write_jsonl(data, rows)
    write_json(options, {"batch_size": 2})
    job = tmp_path / "job"
    assert (
        main(
            [
                "recipe",
                "predict-run",
                asset,
                "--data",
                str(data),
                "--run-dir",
                str(job),
                "--task-factory",
                FACTORY,
                "--batch-size",
                "3",
                "--predict-kwargs-json",
                str(options),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "completed" and result["committed_batches"] == 3
    assert main(["recipe", "inspect-prediction", str(job)]) == 0
    assert json.loads(capsys.readouterr().out)["remaining_records"] == 0
    import arti.recipes as recipes

    monkeypatch.setattr(recipes, "load_recipe_predictor", forbid)
    Path(result["predictions"]).unlink()
    assert main(["recipe", "predict-run", "--run-dir", str(job), "--resume"]) == 0
    assert json.loads(capsys.readouterr().out)["newly_predicted_records"] == 0


def test_callback_input_mutation_does_not_change_committed_input_identity(tmp_path, monkeypatch):
    _current, asset, rows = trained(tmp_path)
    predictor = loaded(asset)
    original = predictor.predict_records

    def predict(records, **kwargs):
        values = original(records, **kwargs)
        for row in records:
            row["temporary"] = "local preprocessing state"
        return values

    monkeypatch.setattr(predictor, "predict_records", predict)
    data = tmp_path / "requests.jsonl"
    write_jsonl(data, rows)
    config = run_config(tmp_path / "job")
    arti.run_prediction(asset, data, config, predictor=predictor)
    assert arti.inspect_prediction(config.output_dir)["action"] == "already_completed"


def test_custom_identity_fields_and_owned_state_are_preserved(tmp_path, monkeypatch):
    from arti.attachment_training import _rng_state

    _current, asset, rows = trained(tmp_path)
    for row in rows:
        row["source_key"] = row["id"]
    predictor = loaded(asset)
    state, rng = copy.deepcopy(predictor.model.state_dict()), _rng_state()
    original = predictor.predict_records

    def predict(records, **kwargs):
        result = original(records, **kwargs)
        predictor.model.feature_mean.add_(1)
        torch.rand(1)
        for value in result:
            value["result_key"] = value["id"]
        return result

    monkeypatch.setattr(predictor, "predict_records", predict)
    data = tmp_path / "requests.jsonl"
    write_jsonl(data, rows)
    config = run_config(tmp_path / "job", id_field="source_key", output_id_field="result_key")
    result = arti.run_prediction(asset, data, config, predictor=predictor)
    assert [row["result_key"] for row in read_jsonl(result["predictions"])] == [
        row["source_key"] for row in rows
    ]
    equal(state, predictor.model.state_dict())
    equal(rng, _rng_state())


def test_model_failure_retains_prefix_and_only_replays_uncommitted_work(tmp_path, monkeypatch):
    _current, asset, rows = trained(tmp_path)
    predictor = loaded(asset)
    original, calls = predictor.predict_records, []

    def predict(records, **kwargs):
        calls.append([row["id"] for row in records])
        if len(calls) == 2:
            raise RuntimeError("model batch")
        return original(records, **kwargs)

    monkeypatch.setattr(predictor, "predict_records", predict)
    source = tmp_path / "requests.jsonl"
    write_jsonl(source, rows)
    config = run_config(tmp_path / "job")
    with pytest.raises(RuntimeError, match="model batch"):
        arti.run_prediction(asset, source, config, predictor=predictor)
    record = json.loads((config.output_dir / "run.json").read_text())
    assert record["status"] == "failed" and record["committed_records"] == 2
    assert record["attempts"][-1]["uncommitted_requested_records"] == 2
    result = arti.run_prediction(asset, None, config, resume=True, predictor=predictor)
    assert result["status"] == "completed" and result["reused_records"] == 2
    assert calls[2] == calls[1] and calls[0] not in calls[1:]


def test_changed_model_source_rejected_before_loader(tmp_path, monkeypatch):
    current, asset, rows = trained(tmp_path)
    source = tmp_path / "requests.jsonl"
    write_jsonl(source, rows)
    config = run_config(tmp_path / "job", max_seconds=0)
    arti.run_prediction(asset, source, config)
    next_asset = current.fit(steps=1, run=policy(tmp_path / "next-training"))["model"]
    import arti.recipes as recipes

    monkeypatch.setattr(recipes, "load_recipe_predictor", forbid)
    with pytest.raises(ValueError, match="model source differs"):
        arti.run_prediction(next_asset, None, config, resume=True)


def test_nested_task_options_remain_pinned_when_callback_uses_local_mutation(tmp_path):
    _current, asset, rows = trained(tmp_path)
    predictor = loaded(asset)
    seen, original = [], predictor._predict_fn

    def predict(model, records, *, settings):
        seen.append(settings["offset"])
        result = list(original(model, records))
        for value in result:
            value["prediction"] += settings["offset"]
        settings["offset"] = 99
        return result

    predictor._predict_fn = predict
    source = tmp_path / "requests.jsonl"
    write_jsonl(source, rows)
    config = arti.PredictionRunConfig(
        tmp_path / "job", batch_size=2, predict_kwargs={"settings": {"offset": 0.5}}
    )
    result = arti.run_prediction(asset, source, config, predictor=predictor)
    assert seen == [0.5] * 4
    assert config.predict_kwargs == {"settings": {"offset": 0.5}}
    assert arti.inspect_prediction(config.output_dir)["action"] == "already_completed"
    assert "generated_tokens" not in result
