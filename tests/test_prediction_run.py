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
from arti.recipe_data import read_jsonl

pytest.importorskip("transformers")
from test_deployment_evaluation import forbid_training, trained
from test_language_recipe import base_model, tokenizer, write
from test_recipe_recovery import equal


def requests(count=7):
    return [
        {
            "id": f"query-{index}",
            "messages": [
                {"role": "user", "content": "old color blue"},
                {"role": "assistant", "content": "blue"},
                {"role": "user", "content": "new color red"},
            ],
        }
        for index in range(count)
    ]


def loaded(asset, factory=None):
    return arti.CausalLMPredictor.from_pretrained(
        asset, model=base_model(), tokenizer=tokenizer(), layer_factory=factory
    )


def policy(path, **kwargs):
    return arti.CausalLMPredictionRunConfig(path, batch_size=2, max_new_tokens=4, **kwargs)


def stop_after_batch(_report, _receipt):
    raise KeyboardInterrupt()


def test_committed_batches_resume_without_repeating_model_work(tmp_path, monkeypatch):
    _current, asset = trained(tmp_path)
    data = write(tmp_path / "requests.jsonl", requests())
    predictor = loaded(asset)
    expected = predictor.predict_records(requests(), batch_size=2, max_new_tokens=4)
    original, calls = predictor.predict_records, []

    def predict(rows, **kwargs):
        calls.append([row["id"] for row in rows])
        assert len(rows) <= 2
        return original(rows, **kwargs)

    monkeypatch.setattr(predictor, "predict_records", predict)
    run = policy(tmp_path / "job")
    with pytest.raises(KeyboardInterrupt):
        arti.run_causal_lm_prediction(
            asset, data, run, predictor=predictor, on_batch=stop_after_batch
        )
    pending = arti.inspect_causal_lm_prediction(run.output_dir)
    assert pending["committed_records"] == 2 and pending["remaining_records"] == 5
    assert pending["action"] == "resume_prediction" and not pending["loads_model_weights"]
    first = (run.output_dir / "parts" / "00000000" / "predictions.jsonl").read_bytes()
    data.unlink()
    report = arti.run_causal_lm_prediction(asset, None, run, resume=True, predictor=predictor)
    assert report["status"] == "completed" and report["committed_records"] == 7
    assert report["newly_predicted_records"] == 5 and report["reused_records"] == 2
    assert calls == [
        [f"query-{i}" for i in range(start, min(start + 2, 7))] for start in range(0, 7, 2)
    ]
    assert list(read_jsonl(report["predictions"])) == expected
    assert (run.output_dir / "parts" / "00000000" / "predictions.jsonl").read_bytes() == first
    assert arti.torch.run_causal_lm_prediction is arti.run_causal_lm_prediction


def test_loader_runs_once_and_complete_retry_never_loads(tmp_path, monkeypatch):
    _current, asset = trained(tmp_path)
    data = write(tmp_path / "requests.jsonl", requests())
    import arti.recipe_prediction as module

    predictor, calls = loaded(asset), []

    def load(*_args, **_kwargs):
        calls.append(1)
        return predictor

    monkeypatch.setattr(module.CausalLMPredictor, "from_pretrained", load)
    report = arti.run_causal_lm_prediction(asset, data, policy(tmp_path / "job"))
    assert len(calls) == 1 and report["committed_batches"] == 4
    monkeypatch.setattr(
        module.CausalLMPredictor,
        "from_pretrained",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("completed job must not load")
        ),
    )
    again = arti.run_causal_lm_prediction(asset, None, policy(tmp_path / "job"), resume=True)
    assert again["newly_predicted_records"] == 0 and again["status"] == "completed"
    assert arti.inspect_causal_lm_prediction(tmp_path / "job")["action"] == "already_completed"


@pytest.mark.parametrize("failure", ["progress", "finalize"])
def test_committed_receipts_are_authoritative_over_failed_progress_or_merge(
    tmp_path, monkeypatch, failure
):
    import arti.recipe_prediction as module

    _current, asset = trained(tmp_path)
    data = write(tmp_path / "requests.jsonl", requests(4))
    predictor = loaded(asset)
    run = policy(tmp_path / "job")
    original = module._atomic_json if failure == "progress" else module._merge_predictions
    fired = []

    def write_record(path, payload):
        if Path(path).name == "run.json" and payload.get("committed_records") == 2 and not fired:
            fired.append(1)
            raise OSError("progress write")
        return original(path, payload)

    if failure == "progress":
        monkeypatch.setattr(module, "_atomic_json", write_record)
    else:
        monkeypatch.setattr(
            module, "_merge_predictions", lambda *_args: (_ for _ in ()).throw(OSError("merge"))
        )
    with pytest.raises(OSError):
        arti.run_causal_lm_prediction(asset, data, run, predictor=predictor)
    pending = arti.inspect_causal_lm_prediction(run.output_dir)
    assert pending["committed_records"] == (2 if failure == "progress" else 4)
    monkeypatch.setattr(
        module, "_atomic_json" if failure == "progress" else "_merge_predictions", original
    )
    calls, predict = [], predictor.predict_records
    monkeypatch.setattr(
        predictor,
        "predict_records",
        lambda rows, **kwargs: (calls.append([row["id"] for row in rows]), predict(rows, **kwargs))[
            1
        ],
    )
    result = arti.run_causal_lm_prediction(asset, None, run, resume=True, predictor=predictor)
    assert result["status"] == "completed"
    assert calls == ([["query-2", "query-3"]] if failure == "progress" else [])
    if failure == "finalize":
        assert result["newly_predicted_records"] == 0


def test_missing_or_changed_merged_result_is_rebuilt_without_a_model(tmp_path, monkeypatch):
    import arti.recipe_prediction as module

    _current, asset = trained(tmp_path)
    data = write(tmp_path / "requests.jsonl", requests(4))
    run = policy(tmp_path / "job")
    first = arti.run_causal_lm_prediction(asset, data, run, predictor=loaded(asset))
    result = Path(first["predictions"])
    expected = result.read_bytes()
    result.write_text("partial", encoding="utf-8")
    assert arti.inspect_causal_lm_prediction(run.output_dir)["action"] == "finalize"
    monkeypatch.setattr(
        module.CausalLMPredictor,
        "from_pretrained",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("no model")),
    )
    second = arti.run_causal_lm_prediction(asset, None, run, resume=True)
    assert second["newly_predicted_records"] == 0 and result.read_bytes() == expected


def test_published_batch_is_adopted_after_interrupt_before_progress_update(tmp_path, monkeypatch):
    import arti.recipe_prediction as module

    _current, asset = trained(tmp_path)
    data = write(tmp_path / "requests.jsonl", requests(4))
    run, predictor = policy(tmp_path / "job"), loaded(asset)
    replace_file = module._replace_atomic_file

    def interrupted(source, destination):
        replace_file(source, destination)
        if Path(destination).name == "00000000":
            raise KeyboardInterrupt()

    monkeypatch.setattr(module, "_replace_atomic_file", interrupted)
    with pytest.raises(KeyboardInterrupt):
        arti.run_causal_lm_prediction(asset, data, run, predictor=predictor)
    assert json.loads((run.output_dir / "run.json").read_text())["committed_records"] == 0
    assert arti.inspect_causal_lm_prediction(run.output_dir)["committed_records"] == 2
    monkeypatch.setattr(module, "_replace_atomic_file", replace_file)
    result = arti.run_causal_lm_prediction(asset, None, run, resume=True, predictor=predictor)
    assert result["reused_records"] == 2 and result["newly_predicted_records"] == 2


def test_recording_failure_does_not_replace_the_original_model_error(tmp_path, monkeypatch):
    import arti.recipe_prediction as module

    _current, asset = trained(tmp_path)
    data = write(tmp_path / "requests.jsonl", requests(4))
    predictor, write_record = loaded(asset), module._atomic_json
    monkeypatch.setattr(
        predictor,
        "predict_records",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("prediction failed")),
    )

    def journal(path, payload):
        if payload.get("status") == "failed":
            raise OSError("journal full")
        return write_record(path, payload)

    monkeypatch.setattr(module, "_atomic_json", journal)
    with pytest.raises(RuntimeError, match="prediction failed") as failure:
        arti.run_causal_lm_prediction(asset, data, policy(tmp_path / "job"), predictor=predictor)
    assert isinstance(failure.value.__cause__, OSError)


def test_completed_output_repairs_stale_derived_status_without_loading(tmp_path, monkeypatch):
    import arti.recipe_prediction as module

    _current, asset = trained(tmp_path)
    data = write(tmp_path / "requests.jsonl", requests(4))
    run = policy(tmp_path / "job")
    arti.run_causal_lm_prediction(asset, data, run, predictor=loaded(asset))
    record = json.loads((run.output_dir / "run.json").read_text())
    record["status"] = "running"
    (run.output_dir / "run.json").write_text(json.dumps(record), encoding="utf-8")
    monkeypatch.setattr(
        module.CausalLMPredictor,
        "from_pretrained",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("no model")),
    )
    assert arti.run_causal_lm_prediction(asset, None, run, resume=True)["status"] == "completed"
    assert json.loads((run.output_dir / "run.json").read_text())["status"] == "completed"


def test_uncommitted_inference_failure_keeps_prior_outputs(tmp_path, monkeypatch):
    _current, asset = trained(tmp_path)
    data = write(tmp_path / "requests.jsonl", requests(5))
    predictor = loaded(asset)
    predict, calls = predictor.predict_records, []

    def fail_second(rows, **kwargs):
        calls.append([row["id"] for row in rows])
        if len(calls) == 2:
            raise RuntimeError("device execution")
        return predict(rows, **kwargs)

    monkeypatch.setattr(predictor, "predict_records", fail_second)
    run = policy(tmp_path / "job")
    with pytest.raises(RuntimeError):
        arti.run_causal_lm_prediction(asset, data, run, predictor=predictor)
    assert arti.inspect_causal_lm_prediction(run.output_dir)["committed_records"] == 2
    record = json.loads((run.output_dir / "run.json").read_text())
    assert record["attempts"][-1]["uncommitted_requested_records"] == 2
    monkeypatch.setattr(predictor, "predict_records", predict)
    assert (
        arti.run_causal_lm_prediction(asset, None, run, resume=True, predictor=predictor)[
            "newly_predicted_records"
        ]
        == 3
    )


@pytest.mark.parametrize("changed", ["data", "model", "batch_size", "max_new_tokens", "snapshot"])
def test_changed_identity_is_rejected_before_model_loading(tmp_path, monkeypatch, changed):
    import arti.recipe_prediction as module

    current, asset = trained(tmp_path)
    data = write(tmp_path / "requests.jsonl", requests())
    run = policy(tmp_path / "job")
    with pytest.raises(KeyboardInterrupt):
        arti.run_causal_lm_prediction(
            asset, data, run, predictor=loaded(asset), on_batch=stop_after_batch
        )
    if changed == "data":
        write(data, requests(8))
    elif changed == "model":
        result = current.fit(
            steps=1, run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "new-model")
        )
        asset = Path(result["model"]).parent
    elif changed in {"batch_size", "max_new_tokens"}:
        run = replace(run, **{changed: 3})
    else:
        write(run.output_dir / "input.jsonl", requests(6))
    monkeypatch.setattr(
        module.CausalLMPredictor,
        "from_pretrained",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("identity check before model")
        ),
    )
    with pytest.raises(ValueError):
        arti.run_causal_lm_prediction(asset, data if changed == "data" else None, run, resume=True)


@pytest.mark.parametrize(
    "invalid", ["duplicate", "missing-id", "completed-chat", "malformed", "empty"]
)
def test_bad_input_does_not_publish_a_run_or_load_model(tmp_path, monkeypatch, invalid):
    import arti.recipe_prediction as module

    _current, asset = trained(tmp_path)
    rows = requests(2)
    if invalid == "duplicate":
        rows[1]["id"] = rows[0]["id"]
    elif invalid == "missing-id":
        rows[0].pop("id")
    elif invalid == "completed-chat":
        rows[0]["messages"].append({"role": "assistant", "content": "red"})
    elif invalid == "empty":
        rows = []
    data = write(tmp_path / "requests.jsonl", rows)
    if invalid == "malformed":
        data.write_text("{broken}\n", encoding="utf-8")
    monkeypatch.setattr(
        module.CausalLMPredictor,
        "from_pretrained",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("invalid input before host")
        ),
    )
    with pytest.raises(ValueError):
        arti.run_causal_lm_prediction(asset, data, policy(tmp_path / "job"))
    assert not (tmp_path / "job").exists()


@pytest.mark.parametrize("stop", ["requested", "zero-budget"])
def test_stop_before_loading_preserves_prepared_job(tmp_path, monkeypatch, stop):
    import arti.recipe_prediction as module

    _current, asset = trained(tmp_path)
    data = write(tmp_path / "requests.jsonl", requests())
    monkeypatch.setattr(
        module.CausalLMPredictor,
        "from_pretrained",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("no load for stopped job")),
    )
    run = policy(tmp_path / "job", max_seconds=0 if stop == "zero-budget" else None)
    result = arti.run_causal_lm_prediction(
        asset, data, run, stop_requested=lambda: stop == "requested"
    )
    assert result["status"] == "stopped" and result["committed_records"] == 0
    assert result["stop_reason"] == ("time_budget" if stop == "zero-budget" else "requested_stop")


def test_elapsed_allowance_stops_new_batches_not_current_generation(tmp_path, monkeypatch):
    import arti.recipe_prediction as module

    _current, asset = trained(tmp_path)
    data = write(tmp_path / "requests.jsonl", requests())
    predictor = loaded(asset)
    now = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    predict = predictor.predict_records

    def forward(rows, **kwargs):
        values = predict(rows, **kwargs)
        now[0] += 2.0
        return values

    monkeypatch.setattr(predictor, "predict_records", forward)
    result = arti.run_causal_lm_prediction(
        asset, data, policy(tmp_path / "job", max_seconds=1), predictor=predictor
    )
    assert result["status"] == "stopped" and result["committed_records"] == 2
    assert result["budget_scope"] == "new-batch-starts-after-model-loading"


def test_owned_state_rng_and_modes_are_preserved_without_optimizer(tmp_path, monkeypatch):
    from arti.attachment_training import _rng_state
    from test_recipe_factory import factory

    _current, asset = trained(tmp_path, factory=factory())
    data = write(tmp_path / "requests.jsonl", requests(3))
    predictor = loaded(asset, factory())
    predictor.model.train()
    predictor.model.register_buffer("observations", torch.zeros((), dtype=torch.long))
    state, modes, rng = (
        copy.deepcopy(predictor.model.state_dict()),
        [m.training for m in predictor.model.modules()],
        _rng_state(),
    )
    forbid_training(monkeypatch)

    def observe(model, _args, _output):
        model.observations.add_(1)
        torch.rand(1)
        random.random()

    handle = predictor.model.register_forward_hook(observe)
    try:
        arti.run_causal_lm_prediction(asset, data, policy(tmp_path / "job"), predictor=predictor)
    finally:
        handle.remove()
    equal(state, predictor.model.state_dict())
    equal(rng, _rng_state())
    assert modes == [m.training for m in predictor.model.modules()]


def test_native_package_and_job_relocation_resume_after_original_data_deletion(tmp_path):
    current, asset = trained(tmp_path)
    native = tmp_path / "native"
    base_model().save_pretrained(native)
    packaged = arti.package_causal_lm_deployment(
        asset, tmp_path / "package", include_base=True, base_model=native
    )
    data = write(tmp_path / "requests.jsonl", requests(5))
    with pytest.raises(KeyboardInterrupt):
        arti.run_causal_lm_prediction(
            packaged["directory"],
            data,
            policy(tmp_path / "job"),
            local_files_only=True,
            on_batch=stop_after_batch,
        )
    moved_model, moved_job = tmp_path / "moved-model", tmp_path / "moved-job"
    shutil.copytree(packaged["directory"], moved_model)
    shutil.copytree(tmp_path / "job", moved_job)
    data.unlink()
    current.config.data_path.unlink()
    result = arti.run_causal_lm_prediction(
        moved_model, None, policy(moved_job), resume=True, local_files_only=True
    )
    assert result["newly_predicted_records"] == 3 and result["reused_records"] == 2


def test_cli_predict_run_and_metadata_inspection(tmp_path, monkeypatch, capsys):
    import arti.language_recipe as language

    _current, asset = trained(tmp_path)
    data = write(tmp_path / "requests.jsonl", requests(5))
    monkeypatch.setattr(language, "_load_model", lambda *_args, **_kwargs: base_model())
    forbid_training(monkeypatch)
    assert (
        main(
            [
                "recipe",
                "predict-run",
                str(asset),
                "--data",
                str(data),
                "--run-dir",
                str(tmp_path / "job"),
                "--batch-size",
                "2",
                "--max-new-tokens",
                "4",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["committed_records"] == 5
    assert main(["recipe", "inspect-prediction", str(tmp_path / "job")]) == 0
    assert json.loads(capsys.readouterr().out)["action"] == "already_completed"
    monkeypatch.setattr(
        language,
        "_load_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("no model on completion")),
    )
    assert (
        main(["recipe", "predict-run", str(asset), "--run-dir", str(tmp_path / "job"), "--resume"])
        == 0
    )
    assert json.loads(capsys.readouterr().out)["newly_predicted_records"] == 0


def test_completed_outputs_can_be_scored_without_inventing_labels(tmp_path):
    _current, asset = trained(tmp_path)
    from test_language_recipe import records

    labeled = records()[-2:]
    inputs = [{"id": row["id"], "messages": row["messages"][:-1]} for row in labeled]
    data = write(tmp_path / "requests.jsonl", inputs)
    result = arti.run_causal_lm_prediction(
        asset, data, policy(tmp_path / "job"), predictor=loaded(asset)
    )
    targets = {row["id"]: row["messages"][-1]["content"] for row in labeled}
    outputs = list(read_jsonl(result["predictions"]))
    scored = arti.score_causal_lm_predictions(
        [{"id": row["id"], "target": targets[row["id"]], "generation": row} for row in outputs]
    )
    assert scored["metrics"]["generated_samples"] == 2
    assert [row["generation"]["token_ids"] for row in scored["predictions"]] == [
        row["token_ids"] for row in outputs
    ]
