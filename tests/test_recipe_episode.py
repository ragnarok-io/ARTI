from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import shutil
import zipfile

import pytest
import torch

import arti
from arti.cli import main
from arti.recipe_data import json_fingerprint, read_jsonl
from arti.recipe_workflow import write_json, write_jsonl
from examples import episode_task as application
from examples.file_lifecycle_workflow import _cold_runtime, _equal, _files


@pytest.fixture(scope="module")
def native_asset(tmp_path_factory):
    root = tmp_path_factory.mktemp("episode-native-model")
    rows = [
        {
            "id": f"native-{index}", "features": [float(index - 6)],
            "target": float((index - 6) * 2),
            "split": "train" if index < 8 else "validation" if index < 11 else "test",
        }
        for index in range(14)
    ]
    data = root / "training-data.jsonl"
    write_jsonl(data, rows)
    task = arti.TaskRecipe.from_config(
        arti.TaskRecipeConfig(
            "generated-regression-policy", "examples.custom_task_recipe:build_task",
            options={"input_shape": [1], "batch_size": 4, "dropout": 0.1, "use_binding": True},
            resources={"data": str(data)},
            training=arti.ARTITrainingConfig(device="cpu", steps=3, learning_rate=0.02),
        )
    )
    initial = {name: parameter.detach().clone() for name, parameter in task.model.named_parameters()}
    result = task.fit(run=arti.ARTITrainingRunConfig(output_dir=root / "training", checkpoint_steps=1))
    assert task.task_contract["objective"] == "scaled-mse"
    assert any(
        not torch.equal(initial[name], parameter)
        for name, parameter in task.model.named_parameters() if name.startswith("program.")
    )
    for name, parameter in task.model.named_parameters():
        if name.startswith("output."):
            assert torch.equal(initial[name], parameter)
    package = root / "package"
    arti.package_recipe_deployment(result["model"], package, resources={})
    data.unlink()
    return package


def episodes(**kwargs):
    return arti.EpisodeRunConfig(
        application.ADAPTER, policy_id="native-policy-v1", environment_id="local-value-v1",
        evaluator_id="local-outcome-v1", initial_state_field="initial", seed_field="seed",
        trial_field="trial", **{"independent_batch": True, **kwargs},
    )


def records(*, horizons=(3, 1, 2, 3, 1)):
    return [
        {
            "id": f"trial-{index}",
            "state": {"value": 0, "goal": 1000, "horizon": horizon},
            "initial": json_fingerprint({"value": 0, "goal": 1000, "horizon": horizon}),
            "seed": index % 3, "trial": index, "reference": {"private_label": "evaluator-only"},
        }
        for index, horizon in enumerate(horizons)
    ]


def run(native_asset, tmp_path, *, config=None, rows=None, predictor=None, **kwargs):
    source = tmp_path / "requests.jsonl"
    if not kwargs.get("resume"):
        write_jsonl(source, records() if rows is None else rows)
    return arti.run_prediction(
        native_asset, None if kwargs.get("resume") else source,
        arti.PredictionRunConfig(tmp_path / "prediction", batch_size=2, episodes=config or episodes()),
        task_factory="examples.custom_task_recipe:build_task", predictor=predictor, **kwargs,
    )


def outputs(tmp_path):
    return list(read_jsonl(tmp_path / "prediction/predictions.jsonl"))


def normalized(rows):
    result = deepcopy(rows)
    for row in result:
        evidence = row["episode"]
        evidence.pop("execution")
        evidence.pop("recovery")
        for entry in evidence["trace"]:
            entry.pop("phase")
            entry.pop("attempt")
            entry.pop("policy_phase")
            entry.pop("policy_attempt")
    return result


def test_evolving_observations_independent_actual_batches_partial_tail_and_native_state(
    native_asset, tmp_path, monkeypatch,
):
    predictor = arti.TaskPredictor.from_pretrained(
        native_asset, task_factory="examples.custom_task_recipe:build_task"
    )
    state = deepcopy(predictor.model.state_dict())
    mode, rng = predictor.model.training, torch.get_rng_state().clone()
    forward, policy = predictor.model.forward, application.policy
    batches, policy_inputs = [], []

    def tracked_forward(features):
        batches.append(features.shape[0])
        return forward(features)

    def tracked_policy(model, observations, states):
        policy_inputs.append(deepcopy((observations, states)))
        assert all(set(observation) == {"value", "distance"} for observation in observations)
        return policy(model, observations, states)

    monkeypatch.setattr(predictor.model, "forward", tracked_forward)
    monkeypatch.setattr(application, "policy", tracked_policy)
    result = run(native_asset, tmp_path, predictor=predictor)
    assert result["status"] == "completed" and result["committed_records"] == 5
    assert result["committed_batches"] == 3 and batches == [2, 1, 1, 2, 2, 1, 1]
    assert json.loads((tmp_path / "prediction/plan.json").read_text())["episodes"]["independent_batch"] is True
    receipts = [json.loads(path.read_text()) for path in sorted((tmp_path / "prediction/parts").glob("*/receipt.json"))]
    assert [receipt["execution"]["policy_batch_calls"] for receipt in receipts] == [3, 3, 1]
    assert [receipt["execution"]["executed_environment_calls"] for receipt in receipts] == [4, 5, 1]
    assert all(receipt["execution"]["replay_environment_calls"] == 0 for receipt in receipts)
    assert [state["calls"] for state in policy_inputs[0][1]] == [0, 0]
    assert [state["calls"] for state in policy_inputs[1][1]] == [1]
    assert [state["calls"] for state in policy_inputs[3][1]] == [0, 0]
    repeated = outputs(tmp_path)
    assert repeated[0]["id"] != repeated[3]["id"]
    assert repeated[0]["episode"]["initial_state"] == repeated[3]["episode"]["initial_state"]
    assert repeated[0]["episode"]["seed"] == repeated[3]["episode"]["seed"]
    assert repeated[0]["episode"]["trace"] == repeated[3]["episode"]["trace"]
    for source, row in zip(records(), outputs(tmp_path), strict=True):
        episode = row["episode"]
        assert row["id"] == source["id"] and episode["initial_state"] == source["initial"]
        assert episode["seed"] == source["seed"] and episode["trial"] == source["trial"]
        assert episode["terminated"] and not episode["truncated"]
        assert episode["termination_reason"] == "environment_horizon"
        assert len(episode["trace"]) == source["state"]["horizon"]
        assert episode["resources"]["steps"] == episode["outcome"]["environment_steps"]
        assert episode["outcome"]["reached"] is False
        for previous, current in zip(episode["trace"], episode["trace"][1:]):
            assert current["observation"] == previous["next_observation"]
        for entry in episode["trace"]:
            assert entry["action"]["claimed_success"] is True
            assert entry["response"]["value"] == entry["observation"]["value"] + entry["action"]["delta"]
    _equal(state, predictor.model.state_dict())
    _equal(rng, torch.get_rng_state())
    assert predictor.model.training == mode


def test_undeclared_independent_batch_dispatches_singletons_and_keeps_pooled_policy_trial_owned(
    native_asset, tmp_path, monkeypatch,
):
    declaration = episodes().to_dict()
    declaration.pop("independent_batch")
    config = arti.EpisodeRunConfig.from_dict(declaration)
    assert config.independent_batch is False
    received = []

    def initial(record):
        return {"calls": 0, "owner": record["id"]}

    def pooled(model, observations, states):
        # This callback deliberately mixes every supplied observation. Singleton
        # dispatch prevents an unrelated trial from contributing to its action.
        received.append(deepcopy((observations, states)))
        total = sum(observation["value"] for observation in observations)
        proposals = []
        for state in states:
            state["calls"] += 1
            proposals.append({
                "action": {"delta": total, "owner": state["owner"], "calls": state["calls"]},
                "state": state,
            })
        return proposals

    monkeypatch.setattr(application, "initial_policy", initial)
    monkeypatch.setattr(application, "policy", pooled)
    rows = records(horizons=(2, 1, 2))
    for row, value in zip(rows, (1, 50, 3), strict=True):
        row["state"]["value"] = value
        row["initial"] = json_fingerprint(row["state"])
    result = run(native_asset, tmp_path, config=config, rows=rows)
    assert [len(observations) for observations, _states in received] == [1] * 5
    assert result["policy_batch_calls"] == 5
    assert json.loads((tmp_path / "prediction/plan.json").read_text())["episodes"]["independent_batch"] is False
    actual = outputs(tmp_path)
    assert [row["episode"]["outcome"]["position"] for row in actual] == [4, 100, 12]
    for source, output in zip(rows, actual, strict=True):
        trace = output["episode"]["trace"]
        assert all(entry["action"]["owner"] == source["id"] for entry in trace)
        assert [entry["action"]["calls"] for entry in trace] == list(range(1, len(trace) + 1))
        assert all(entry["action"]["delta"] == entry["observation"]["value"] for entry in trace)


def test_step_scoped_policy_state_resets_without_resetting_environment(native_asset, tmp_path, monkeypatch):
    callback, calls = application.policy, []

    def tracked(model, observations, states):
        calls.extend(state["calls"] for state in states)
        return callback(model, observations, states)

    monkeypatch.setattr(application, "policy", tracked)
    run(native_asset, tmp_path, config=episodes(policy_state_lifetime="step"), rows=records(horizons=(3,)))
    episode = outputs(tmp_path)[0]["episode"]
    assert calls == [0, 0, 0] and episode["outcome"]["environment_steps"] == 3
    assert episode["trace"][1]["observation"]["value"] != 0


@pytest.mark.parametrize("limit", ["max_steps", "max_calls", "host"])
def test_task_termination_remains_distinct_from_host_and_call_step_truncation(
    native_asset, tmp_path, limit,
):
    settings = {} if limit == "host" else {limit: 1}
    events = []

    def observed(event):
        events.append(event["kind"])

    run(
        native_asset, tmp_path, config=episodes(**settings), rows=records(horizons=(3,)),
        on_episode_event=observed,
        **({"stop_requested": lambda: "response" in events} if limit == "host" else {}),
    )
    episode = outputs(tmp_path)[0]["episode"]
    assert not episode["terminated"] and episode["truncated"]
    assert episode["outcome"]["environment_steps"] == 1
    assert episode["outcome"]["reached"] is False
    assert episode["termination_reason"] != "environment_horizon"


@pytest.mark.parametrize("recovery", ["snapshot", "replay"])
def test_durable_response_recovery_preserves_state_trace_and_native_outcomes(
    native_asset, tmp_path, recovery,
):
    full, partial = tmp_path / "full", tmp_path / "partial"
    config, rows = episodes(recovery=recovery), records(horizons=(3, 2, 1))
    run(native_asset, full, config=config, rows=rows)

    def interrupt(event):
        if event["kind"] == "response":
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run(native_asset, partial, config=config, rows=rows, on_episode_event=interrupt)
    work = partial / "prediction/work/episodes/00000000.json"
    assert work.is_file() and json.loads(work.read_text())["sha256"]
    run(native_asset, partial, config=config, resume=True)
    actual = outputs(partial)
    assert normalized(actual) == normalized(outputs(full))
    first = actual[0]["episode"]
    assert first["trace"][0]["phase"] == "initial"
    assert first["trace"][-1]["phase"] == "recovered"
    assert first["execution"]["phase"] == "recovered"
    assert first["execution"]["outcomes"] == [
        {
            "attempt": 1, "phase": "initial", "availability": "unavailable",
            "reason": "no_terminal_outcome_recorded", "outcome": None,
        },
        {
            "attempt": 2, "phase": "recovered", "availability": "observed",
            "reason": "environment_outcome_evaluated", "outcome": first["outcome"],
        },
    ]
    assert first["recovery"]
    result = arti.inspect_prediction(partial / "prediction")
    assert result["replay_environment_calls"] == (1 if recovery == "replay" else 0)
    assert result["local_replay_step_calls"] == (1 if recovery == "replay" else 0)


def test_local_replay_checks_actual_prior_responses_instead_of_trusting_declaration(
    native_asset, tmp_path, monkeypatch,
):
    config = episodes(recovery="replay")
    with pytest.raises(KeyboardInterrupt):
        run(
            native_asset, tmp_path, config=config, rows=records(horizons=(3,)),
            on_episode_event=lambda event: (_ for _ in ()).throw(KeyboardInterrupt())
            if event["kind"] == "response" else None,
        )
    original = application.step

    def changed(environment, action):
        value = original(environment, action)
        value["response"]["value"] += 1
        return value

    monkeypatch.setattr(application, "step", changed)
    with pytest.raises((ValueError, RuntimeError), match="replay|response|differs"):
        run(native_asset, tmp_path, config=config, resume=True)
    assert not list((tmp_path / "prediction/parts").iterdir())


def test_terminal_response_recovery_finalizes_outcome_without_repeating_action(
    native_asset, tmp_path, monkeypatch,
):
    config = episodes(recovery="snapshot")
    with pytest.raises(KeyboardInterrupt):
        run(
            native_asset, tmp_path, config=config, rows=records(horizons=(1,)),
            on_episode_event=lambda event: (_ for _ in ()).throw(KeyboardInterrupt())
            if event["kind"] == "response" else None,
        )

    def forbidden(*args, **kwargs):
        raise AssertionError("a durable terminal response must not repeat the action or policy")

    monkeypatch.setattr(application, "step", forbidden)
    monkeypatch.setattr(application, "policy", forbidden)
    result = run(native_asset, tmp_path, config=config, resume=True)
    assert result["committed_records"] == 1
    episode = outputs(tmp_path)[0]["episode"]
    assert episode["terminated"] and episode["outcome"]["environment_steps"] == 1
    assert len(episode["trace"]) == 1 and episode["trace"][0]["phase"] == "initial"


def test_recovered_pending_action_rechecks_host_stop_after_snapshot_restore(
    native_asset, tmp_path, monkeypatch,
):
    config, initialized, step_calls = episodes(recovery="snapshot"), [], []
    original_step, original_restore = application.step, application.restore

    def tracked_step(environment, action):
        step_calls.append(deepcopy(action))
        return original_step(environment, action)

    def interrupt(event):
        if event["kind"] == "initialized":
            initialized.append(event["id"])
        if event["kind"] == "intent" and initialized:
            raise KeyboardInterrupt

    monkeypatch.setattr(application, "step", tracked_step)
    with pytest.raises(KeyboardInterrupt):
        run(
            native_asset, tmp_path, config=config, rows=records(horizons=(3,)),
            on_episode_event=interrupt,
        )
    saved = json.loads((tmp_path / "prediction/work/episodes/00000000.json").read_text())["trials"][0]
    assert saved["pending"]["kind"] == "action" and saved["trace"] == []
    pending, stop, restored = deepcopy(saved["pending"]), {"requested": False}, []

    def restore_and_stop(record, state):
        restored.append(deepcopy(state))
        stop["requested"] = True
        return original_restore(record, state)

    def forbidden(*args, **kwargs):
        raise AssertionError("a stopped saved proposal must not rerun policy inference")

    monkeypatch.setattr(application, "restore", restore_and_stop)
    monkeypatch.setattr(application, "policy", forbidden)
    result = run(
        native_asset, tmp_path, config=config, resume=True,
        stop_requested=lambda: stop["requested"],
    )
    assert result["status"] == "completed" and len(restored) == 1 and step_calls == []
    episode = outputs(tmp_path)[0]["episode"]
    assert episode["truncated"] and not episode["terminated"]
    assert episode["termination_reason"] == "requested_stop" and episode["trace"] == []
    unexecuted = episode["unexecuted_action"]
    assert unexecuted["status"] == "not_executed" and unexecuted["reason"] == "requested_stop"
    assert unexecuted["action"] == pending["action"] and unexecuted["observation"] == saved["observation"]
    assert unexecuted["policy_attempt"] == 1 and unexecuted["attempt"] == 2
    assert unexecuted["prior_effect_outcome"] == "unrecorded"
    assert "response" not in unexecuted and "outcome" not in unexecuted
    assert episode["outcome"] == application.evaluate(restored[0], [])
    assert episode["resources"]["steps"] == episode["resources"]["environment_calls"] == 0


@pytest.mark.parametrize("state", ["policy", "environment"])
def test_unsupported_native_state_codec_fails_before_any_action(
    native_asset, tmp_path, monkeypatch, state,
):
    calls = []
    if state == "policy":
        callback = application.policy

        def native_state(model, observations, states):
            result = callback(model, observations, states)
            result[0]["state"] = torch.tensor(1)
            return result

        monkeypatch.setattr(application, "policy", native_state)
    else:
        monkeypatch.setattr(application, "snapshot", lambda env: {"unsupported": torch.tensor(1)})
    monkeypatch.setattr(application, "step", lambda *_: calls.append("action"))
    with pytest.raises(ValueError, match="JSON|codec"):
        run(native_asset, tmp_path, config=episodes(recovery="snapshot"), rows=records(horizons=(1,)))
    assert calls == []


@pytest.mark.parametrize("recovery,boundary", [("none", "intent"), ("none", "response"), ("snapshot", "intent")])
def test_nonresumable_external_declaration_never_automatically_repeats_ambiguous_actions(
    native_asset, tmp_path, monkeypatch, recovery, boundary,
):
    # This is only a declaration over a local fixture; no external effect is executed.
    config = episodes(recovery=recovery, effects="external")
    callback, calls = application.step, []

    def tracked(environment, action):
        calls.append(deepcopy(action))
        return callback(environment, action)

    monkeypatch.setattr(application, "step", tracked)
    initialized = []

    def interrupt(event):
        if event["kind"] == "initialized":
            initialized.append(event["id"])
        if event["kind"] == boundary and initialized:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run(
            native_asset, tmp_path, config=config, rows=records(horizons=(3,)),
            on_episode_event=interrupt,
        )
    before = deepcopy(calls)
    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported recovery must be diagnosed before application import")

    monkeypatch.setattr(arti.PythonFactoryConfig, "resolve", forbidden)
    result = run(native_asset, tmp_path, config=config, resume=True)
    assert result["status"] == "not_run" and result["reason"] == "unsupported_episode_recovery"
    assert result["episode_recovery"]["blockers"]
    assert calls == before
    assert len(calls) == (0 if boundary == "intent" else 1)


@pytest.mark.parametrize("observation", [None, 0])
def test_explicit_initialization_recovers_null_or_zero_observations_without_reset(
    native_asset, tmp_path, monkeypatch, observation,
):
    callback = application.policy
    monkeypatch.setattr(application, "observe", lambda env: observation)
    monkeypatch.setattr(
        application, "policy",
        lambda model, observations, states: callback(
            model, [{"value": 0, "distance": 1} for _ in observations], states
        ),
    )
    config = episodes(recovery="snapshot")
    with pytest.raises(KeyboardInterrupt):
        run(
            native_asset, tmp_path, config=config, rows=records(horizons=(1,)),
            on_episode_event=lambda event: (_ for _ in ()).throw(KeyboardInterrupt())
            if event["kind"] == "initialized" else None,
        )

    def forbidden(*args, **kwargs):
        raise AssertionError("restoring an initialized environment must not reset it")

    monkeypatch.setattr(application, "start", forbidden)
    run(native_asset, tmp_path, config=config, resume=True)
    episode = outputs(tmp_path)[0]["episode"]
    assert episode["trace"][0]["observation"] == observation
    assert episode["terminated"] and episode["outcome"]["environment_steps"] == 1


def test_completed_parts_move_without_source_and_need_no_adapter_model_or_replay(
    native_asset, tmp_path, monkeypatch,
):
    source = tmp_path / "source"
    shutil.copytree(native_asset, source)
    run(source, tmp_path)
    journal = tmp_path / "prediction"
    before = _files(journal / "parts")
    expected = outputs(tmp_path)
    moved = tmp_path / "moved"
    shutil.move(journal, moved)
    (tmp_path / "requests.jsonl").unlink()
    assert source.resolve().is_relative_to(tmp_path.resolve())
    shutil.rmtree(source)

    def forbidden(*args, **kwargs):
        raise AssertionError("complete output must not reconstruct a model or episode adapter")

    monkeypatch.setattr(arti.PythonFactoryConfig, "resolve", forbidden)
    resumed = arti.run_prediction(None, None, arti.PredictionRunConfig(moved), resume=True)
    assert resumed["newly_predicted_records"] == 0 and _files(moved / "parts") == before
    (moved / "predictions.jsonl").unlink()
    arti.run_prediction(None, None, arti.PredictionRunConfig(moved), resume=True)
    assert list(read_jsonl(moved / "predictions.jsonl")) == expected
    assert arti.inspect_prediction(moved)["committed_records"] == 5


def test_fully_finished_unpublished_trials_adopt_parts_without_model_adapter_or_evaluator(
    native_asset, tmp_path, monkeypatch,
):
    finished = []

    def interrupt(event):
        if event["kind"] == "finished":
            finished.append(event["id"])
            if len(finished) == 2:
                raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run(native_asset, tmp_path, rows=records(horizons=(1, 1)), on_episode_event=interrupt)
    journal = tmp_path / "prediction"
    work = json.loads((journal / "work/episodes/00000000.json").read_text())
    expected = [trial["result"] for trial in work["trials"]]
    assert all(result is not None for result in expected)
    assert arti.inspect_prediction(journal)["action"] == "publish_episode_part"
    (tmp_path / "requests.jsonl").unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("finished durable trials require publication only")

    monkeypatch.setattr(arti.PythonFactoryConfig, "resolve", forbidden)
    result = arti.run_prediction(None, None, arti.PredictionRunConfig(journal), resume=True)
    assert result["status"] == "completed" and result["newly_predicted_records"] == 0
    assert result["reused_records"] == result["committed_records"] == 2
    assert outputs(tmp_path) == expected
    assert all(row["episode"]["execution"]["phase"] == "initial" for row in expected)


def test_reported_call_consumption_and_library_callbacks_remain_distinct_with_allowance_overrun(
    native_asset, tmp_path, monkeypatch,
):
    callback = application.step

    def reported_calls(environment, action):
        response = callback(environment, action)
        response["calls"] = 3
        return response

    monkeypatch.setattr(application, "step", reported_calls)
    result = run(native_asset, tmp_path, config=episodes(max_calls=2), rows=records(horizons=(3,)))
    assert result["completed_environment_step_callbacks"] == 1
    assert result["executed_environment_calls"] == 3
    episode = outputs(tmp_path)[0]["episode"]
    assert episode["truncated"] and not episode["terminated"]
    assert episode["resources"]["environment_calls"] == 3 and episode["resources"]["steps"] == 1
    assert episode["allowances"]["max_calls"] == 2 and not episode["allowances"]["host_enforced"]
    assert "overrun" in episode["allowances"]["scope"]
    assert episode["resource_attribution"]["environment_calls"] == "application-response-reported-consumption"


def test_unsupported_state_lifetimes_are_reported_before_execution():
    with pytest.raises(ValueError, match="state|lifetime|codec"):
        episodes(native_model_state_lifetime="persistent-global")
    with pytest.raises(ValueError, match="state|lifetime"):
        episodes(environment_state_lifetime="shared-run")


def episode_suite(config, *, references=False, external=False):
    views = [
        arti.EvaluationViewConfig(
            "local",
            arti.PredictionScoringConfig("examples.episode_task:score_outcomes", "local-terminal-v1"),
            requires="output",
        )
    ]
    if references:
        views.append(
            arti.EvaluationViewConfig(
                "reference",
                arti.PredictionScoringConfig("examples.episode_task:score_references", "reference-v1"),
                reference_field="reached", availability_field="available",
            )
        )
    if external:
        views.append(
            arti.EvaluationViewConfig("external", requires="external", external_protocol="fixture-outcome-v1")
        )
    return arti.EvaluationSuiteConfig(
        tuple(views),
        trials=arti.EvaluationTrialFields(
            config.initial_state_field, config.seed_field, config.trial_field
        ),
        aggregator=arti.PythonFactoryConfig(
            "examples.evaluation_suite_scorers:aggregate_views",
            {"coefficients": [["local", "terminal_fraction", 1]]},
        ),
        aggregation_id="caller-terminal-v1",
    )


def test_episode_suite_original_trial_coverage_phase_and_paired_state_seed_evidence(
    native_asset, tmp_path, monkeypatch,
):
    config, baseline, recovered = episodes(recovery="snapshot"), tmp_path / "baseline", tmp_path / "recovered"
    run(native_asset, baseline, config=config)
    with pytest.raises(KeyboardInterrupt):
        run(
            native_asset, recovered, config=config,
            on_batch=lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
    run(native_asset, recovered, config=config, resume=True)
    suite = episode_suite(config, references=True)
    references = tmp_path / "references.jsonl"
    write_jsonl(references, [
        {"id": row["id"], "reached": False, "available": index < 3}
        for index, row in enumerate(records())
    ])
    different = tmp_path / "different"
    changed = records()
    changed[0]["seed"] += 1
    run(native_asset, different, config=config, rows=changed)
    import arti.recipes as recipes

    monkeypatch.setattr(
        recipes, "load_recipe_predictor",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("suite must not replay episodes")),
    )
    scored = arti.evaluate_prediction_suite(recovered / "prediction", suite, references=references)
    assert scored["source"]["conditions"]["episodes"] == config.to_dict()
    local = scored["views"]["local"]
    assert local["coverage"]["total"] == local["coverage"]["scored"] == 5
    assert [row["execution"]["phase"] for row in local["outcomes"]] == ["initial"] * 2 + ["recovered"] * 3
    assert [row["trial"] for row in local["outcomes"]] == [
        {"initial_state": row["initial"], "seed": row["seed"], "trial": row["trial"]}
        for row in records()
    ]
    assert all(row["score"]["reached"] is False for row in local["outcomes"])
    assert scored["views"]["reference"]["coverage"]["scored"] == 3
    assert scored["views"]["reference"]["coverage"]["total_fraction"] == 3 / 5
    assert all(row["score"] is None for row in scored["views"]["reference"]["outcomes"][3:])
    assert not scored["complete_evidence"] and scored["criteria_passed"] is None
    rejected = arti.evaluate_prediction_suite(
        recovered / "prediction", replace(suite, criteria={"minimum_coverage": {"minimum": 1}}),
        references=references,
    )
    assert rejected["criteria_passed"] is False
    paired = arti.compare_prediction_suites(
        recovered / "prediction", baseline / "prediction", suite, references=references
    )
    assert paired["views"]["local"]["comparable"]
    assert paired["views"]["local"]["delta"]["terminal_fraction"] == 0
    with pytest.raises(ValueError, match="paired|trial|input"):
        arti.compare_prediction_suites(
            recovered / "prediction", different / "prediction", suite, references=references
        )


def test_episode_external_evidence_stays_attributed_separately_from_local_environment_outcomes(
    native_asset, tmp_path,
):
    config = episodes()
    run(native_asset, tmp_path, config=config)
    directory, suite = tmp_path / "prediction", episode_suite(config, external=True)
    payload = {
        "source": arti.evaluation_source_identity(directory), "protocol": "fixture-outcome-v1",
        "producer": {"name": "declared-fixture", "revision": "fixture-v1"},
        "metrics": {"observed_measurement": 9},
        "scores": [{"id": row["id"], "measurement": 9, "unit": "caller-unit"} for row in records()[:-1]],
    }
    evidence = tmp_path / "external.json"
    imported = arti.import_evaluation_evidence(directory, suite, "external", payload, output=evidence)
    assert imported["origin"].startswith("external")
    report = arti.evaluate_prediction_suite(directory, suite, external={"external": evidence})
    assert report["views"]["local"]["origin"] == "local"
    assert report["views"]["external"]["origin"] == "external"
    assert report["views"]["external"]["coverage"]["scored"] == 4
    assert report["views"]["external"]["coverage"]["total"] == 5
    assert report["views"]["external"]["metrics"]["observed_measurement"] == 9
    assert report["views"]["external"]["scores"][0]["score"]["unit"] == "caller-unit"
    assert all(row["score"]["reached"] is False for row in report["views"]["local"]["outcomes"])


def test_pairing_uses_committed_episode_state_seed_trial_despite_identical_input_aliases(
    native_asset, tmp_path, monkeypatch,
):
    rows = records(horizons=(1, 1))
    for index, row in enumerate(rows):
        alternate = {**row["state"], "value": index + 8}
        alternate_identity = json_fingerprint(alternate)
        row.update(
            initial_a=row["initial"], seed_a=row["seed"], trial_a=row["trial"],
            initial_b=alternate_identity, seed_b=row["seed"] + 7, trial_b=row["trial"] + 10,
            initial_c=row["initial"], seed_c=row["seed"], trial_c=row["trial"],
            state_options={row["initial"]: row["state"], alternate_identity: alternate},
        )
    callback = application.start

    def selected(record, *, seed, initial_state):
        return callback(
            {**record, "state": record["state_options"][initial_state]},
            seed=seed, initial_state=initial_state,
        )

    monkeypatch.setattr(application, "start", selected)
    configurations = {
        alias: replace(
            episodes(), initial_state_field=f"initial_{alias}",
            seed_field=f"seed_{alias}", trial_field=f"trial_{alias}",
        )
        for alias in ("a", "b", "c")
    }
    for alias, config in configurations.items():
        run(native_asset, tmp_path / alias, config=config, rows=rows)
    assert (tmp_path / "a/requests.jsonl").read_bytes() == (tmp_path / "b/requests.jsonl").read_bytes()
    assert (tmp_path / "a/requests.jsonl").read_bytes() == (tmp_path / "c/requests.jsonl").read_bytes()
    for fields in (
        arti.EvaluationTrialFields(),
        arti.EvaluationTrialFields("initial", "seed", "trial"),
    ):
        suite = replace(episode_suite(episodes()), trials=fields)
        with pytest.raises(ValueError, match="paired committed episode state/seed/trial"):
            arti.compare_prediction_suites(
                tmp_path / "a/prediction", tmp_path / "b/prediction", suite
            )
        comparable = arti.compare_prediction_suites(
            tmp_path / "a/prediction", tmp_path / "c/prediction", suite
        )
        assert comparable["paired_samples"] == 2 and comparable["views"]["local"]["comparable"]
        assert comparable["views"]["local"]["delta"]["terminal_fraction"] == 0


def test_episode_suite_cold_selected_code_handoff_export_and_source_free_archive_reuse(
    native_asset, tmp_path,
):
    code = tmp_path / "original/examples"
    code.mkdir(parents=True)
    (code / "__init__.py").write_text("", encoding="utf-8")
    for name in (
        "runtime_task.py", "custom_task_recipe.py", "episode_task.py", "evaluation_suite_scorers.py",
    ):
        shutil.copyfile(Path(application.__file__).with_name(name), code / name)
    config, suite = episodes(recovery="snapshot"), episode_suite(episodes(recovery="snapshot"))
    runtime = arti.TaskRuntimeProfile(
        "examples.runtime_task:validate", "local-episode-runtime-v1", network="disabled",
        requirements=("torch>=2",),
        inputs={"requests": "file", "references": "file", "suite": "file"},
        outputs={"results": "directory", "evaluation": "file"},
        options={
            "task_factory": "examples.custom_task_recipe:build_task", "batch_size": 2,
            "episodes": config.to_dict(),
        },
    )
    package, inputs = tmp_path / "package", tmp_path / "inputs"
    arti.package_recipe_deployment(native_asset, package, runtime=runtime, code={"application": code.parent})
    write_jsonl(inputs / "requests.jsonl", records())
    write_jsonl(inputs / "references.jsonl", [{"id": row["id"]} for row in records()])
    write_json(inputs / "suite.json", suite.to_dict())
    assert code.parent.resolve().is_relative_to(tmp_path.resolve())
    shutil.rmtree(code.parent)
    journal, evaluation, receipt = tmp_path / "prediction", tmp_path / "validation.json", tmp_path / "runtime.json"
    execution = arti.run_task_runtime(
        package,
        inputs={name: inputs / filename for name, filename in (
            ("requests", "requests.jsonl"), ("references", "references.jsonl"), ("suite", "suite.json")
        )},
        outputs={"results": journal, "evaluation": evaluation}, output=receipt,
    )
    assert execution["status"] == "completed"
    assert execution["runtime_identity"]["framework_root"] == str(Path(arti.__file__).resolve().parent.parent)
    assert not execution["network_enforced"] and not execution["installs_dependencies"]
    validation = execution["worker"]["result"]
    assert validation["prediction"]["newly_predicted_records"] == 5
    native = list(read_jsonl(journal / "predictions.jsonl"))
    assert len(native) == 5 and all(row["episode"]["terminated"] for row in native)
    archive = tmp_path / "results.zip"
    exported = arti.export_prediction_results(
        journal,
        arti.ResultExportConfig(
            "examples.result_writers:write_results", "episode-layout-v1", archive="zip",
            options={"value_field": "episode"}, validator="examples.result_writers:validate_results",
            validation_id="episode-layout-validator-v1", instructions="Extract and reuse the declared journal and inputs.",
            unresolved_checks=("external effects and host resource enforcement are unqualified",),
        ),
        archive, expected_ids=[row["id"] for row in records()], deployment=package,
        evidence={"runtime": receipt, "validation": evaluation, "suite": validation["scores"]["file"]},
        attachments={"journal": journal, "inputs": inputs},
    )
    assert not exported["regenerates_predictions"] and not exported["submits_externally"]
    for directory in (package, inputs, journal):
        assert directory.resolve().is_relative_to(tmp_path.resolve())
        shutil.rmtree(directory)
    extracted = tmp_path / "extracted"
    with zipfile.ZipFile(archive) as stream:
        stream.extractall(extracted)
    manifest = json.loads((extracted / ".arti-export/manifest.json").read_text())
    attached = manifest["attachments"]
    recovered = tmp_path / "recovered"
    shutil.copytree(extracted / attached["journal"]["path"], recovered)
    before = _files(recovered / "parts")
    repeated = _cold_runtime(
        extracted / ".arti-export/deployment", extracted / attached["inputs"]["path"], recovered,
        tmp_path / "recovered-validation.json", tmp_path / "recovered-runtime.json", tmp_path / "consumer",
    )
    assert repeated["worker"]["result"]["prediction"]["newly_predicted_records"] == 0
    assert _files(recovered / "parts") == before
    assert list(read_jsonl(recovered / "predictions.jsonl")) == native
    assert arti.inspect_result_export(archive)["sha256"] == exported["sha256"]
    assert arti.inspect_result_export(extracted)["sha256"] == exported["sha256"]


def test_cli_episode_declaration_and_completed_model_free_inspect_resume(
    native_asset, tmp_path, monkeypatch, capsys,
):
    model, data, declaration, journal = (
        tmp_path / "model", tmp_path / "requests.jsonl", tmp_path / "episodes.json", tmp_path / "prediction"
    )
    shutil.copytree(native_asset, model)
    write_jsonl(data, records(horizons=(2, 1, 2)))
    config = episodes(recovery="snapshot")
    write_json(declaration, config.to_dict())
    assert arti.EpisodeRunConfig.read(declaration) == config
    assert main([
        "recipe", "predict-run", str(model), "--data", str(data), "--run-dir", str(journal),
        "--episode-config", str(declaration), "--batch-size", "2",
        "--task-factory", "examples.custom_task_recipe:build_task",
    ]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["committed_records"] == 3 and report["status"] == "completed"
    before = _files(journal / "parts")
    assert model.resolve().is_relative_to(tmp_path.resolve())
    shutil.rmtree(model)
    data.unlink()
    declaration.unlink()

    def forbidden(*args, **kwargs):
        raise AssertionError("completed CLI inspection/recovery must not import application code")

    import arti.recipes as recipes

    monkeypatch.setattr(arti.PythonFactoryConfig, "resolve", forbidden)
    monkeypatch.setattr(recipes, "load_recipe_predictor", forbidden)
    assert main(["recipe", "inspect-prediction", str(journal)]) == 0
    assert json.loads(capsys.readouterr().out)["committed_records"] == 3
    assert main(["recipe", "predict-run", "--run-dir", str(journal), "--resume"]) == 0
    assert json.loads(capsys.readouterr().out)["newly_predicted_records"] == 0
    assert _files(journal / "parts") == before
