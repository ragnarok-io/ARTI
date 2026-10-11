"""Explicit application episodes inside the existing prediction run lifecycle."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass, asdict
import json
import math
from pathlib import Path

import torch

from .python_factory import PythonFactoryConfig
from .recipe_data import _positive_int, json_fingerprint
from .recipe_workflow import read_recipe_config
from .serialization import _atomic_json, _load_json


def _json(value, name="episode state"):
    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"{name} requires supported JSON state; no native state codec is declared"
        ) from error


def _name(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"episode {name} must be a nonempty string")


@dataclass(frozen=True)
class EpisodeRunConfig:
    adapter: PythonFactoryConfig | str
    policy_id: str
    environment_id: str
    evaluator_id: str
    initial_state_field: str
    seed_field: str
    recovery: str = "none"
    effects: str = "local"
    policy_state_lifetime: str = "trial"
    environment_state_lifetime: str = "trial"
    native_model_state_lifetime: str = "read-only-policy-step"
    max_steps: int | None = None
    max_calls: int | None = None
    field: str = "episode"
    trial_field: str | None = None
    independent_batch: bool = False

    def __post_init__(self):
        if isinstance(self.adapter, str):
            object.__setattr__(self, "adapter", PythonFactoryConfig(self.adapter))
        if not isinstance(self.adapter, PythonFactoryConfig):
            raise ValueError("episode adapter must be an installed PythonFactoryConfig")
        for name in (
            "policy_id",
            "environment_id",
            "evaluator_id",
            "initial_state_field",
            "seed_field",
            "field",
        ):
            _name(getattr(self, name), name)
        if self.trial_field is not None:
            _name(self.trial_field, "trial_field")
        if type(self.independent_batch) is not bool:
            raise ValueError("episode independent_batch capability must be an explicit boolean")
        if self.recovery not in {"snapshot", "replay", "none"}:
            raise ValueError("episode recovery must be snapshot, replay or none")
        if self.effects not in {"local", "external"}:
            raise ValueError("episode effects must be local or external")
        if self.effects == "external" and self.recovery == "replay":
            raise ValueError("external effects cannot use automatic local replay")
        if self.policy_state_lifetime not in {"step", "trial"}:
            raise ValueError("episode policy state lifetime must be step or trial")
        if self.environment_state_lifetime != "trial":
            raise ValueError("independent episodes require trial-owned environment state")
        if self.native_model_state_lifetime != "read-only-policy-step":
            raise ValueError(
                "native model persistence requires an explicitly supported codec; only read-only-policy-step is supported"
            )
        for name in ("max_steps", "max_calls"):
            if getattr(self, name) is not None:
                _positive_int(getattr(self, name), f"episode.{name}")

    def to_dict(self):
        return {
            "format": "arti.episode-run-config",
            **asdict(self),
            "adapter": self.adapter.to_dict(),
        }

    @property
    def dispatch_scope(self):
        return (
            "declared-independent-batch" if self.independent_batch else "singleton-policy-dispatch"
        )

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if value.pop("format", "arti.episode-run-config") != "arti.episode-run-config":
            raise ValueError("not an episode run configuration")
        if isinstance(value.get("adapter"), Mapping):
            value["adapter"] = PythonFactoryConfig(**value["adapter"])
        return cls(**value)

    @classmethod
    def read(cls, path):
        return cls.from_dict(read_recipe_config(path))

    def trial(self, record):
        initial = record.get(self.initial_state_field)
        _name(initial, "initial-state identity")
        seed = record.get(self.seed_field)
        if type(seed) is not int or seed < 0:
            raise ValueError("episode seed must be a nonnegative integer")
        result = {"initial_state": initial, "seed": seed}
        if self.trial_field is not None:
            if self.trial_field not in record or record[self.trial_field] is None:
                raise ValueError("declared episode trial identity is missing")
            result["trial"] = _json(record[self.trial_field], "trial identity")
        return result


class EpisodeAdapter:
    """Trusted caller callbacks; policy receives observations and individual states."""

    def __init__(
        self,
        *,
        start,
        observe,
        policy,
        step,
        evaluate,
        initial_policy=None,
        snapshot=None,
        restore=None,
    ):
        for name, callback in locals().copy().items():
            if name == "self":
                continue
            if callback is not None and not callable(callback):
                raise ValueError(f"episode {name} must be callable")
            setattr(self, name, callback)
        if any(
            getattr(self, name) is None
            for name in ("start", "observe", "policy", "step", "evaluate")
        ):
            raise ValueError(
                "episode adapter requires start/observe/policy/step/evaluate callbacks"
            )


def _work_path(directory, index):
    return Path(directory) / "work" / "episodes" / f"{index:08d}.json"


def _save(path, state):
    payload = {key: value for key, value in state.items() if key != "sha256"}
    payload = _json(payload)
    payload["sha256"] = json_fingerprint(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_json(path, payload)


def _read(path, plan):
    state = _load_json(path)
    if (
        state.get("format") != "arti.prediction-episode-work"
        or state.get("plan_sha256") != plan["sha256"]
        or state.get("sha256")
        != json_fingerprint({key: value for key, value in state.items() if key != "sha256"})
    ):
        raise ValueError("episode work differs from its pinned prediction plan")
    return state


def _blocker(trial, config):
    if trial.get("result") is not None:
        return None
    if trial.get("pending") is not None and config.effects == "external":
        return "ambiguous_external_effect"
    if trial.get("initialized") and config.recovery == "none":
        return "non_resumable_episode_state"
    return None


def inspect_episode_work(directory, plan, committed_batches):
    if "episodes" not in plan:
        return {}
    config = EpisodeRunConfig.from_dict(plan["episodes"])
    blockers, pending, finished, counters = [], 0, 0, {}
    path = _work_path(directory, committed_batches)
    if path.is_file():
        state = _read(path, plan)
        counters = _counters(state)
        for trial in state["trials"]:
            pending += int(trial.get("pending") is not None)
            finished += int(trial.get("result") is not None)
            reason = _blocker(trial, config)
            if reason is not None:
                blockers.append({"id": trial["id"], "reason": reason})
    return {
        "episode_recovery": {
            "mode": config.recovery,
            "effects": config.effects,
            "blockers": blockers,
            "pending_intents": pending,
            "finished_unpublished_trials": finished,
            "finished_part_ready": bool(finished) and finished == len(state["trials"])
            if path.is_file()
            else False,
            "work_counters": counters,
            "exactly_once_external_effects": False,
        }
    }


def _counters(state):
    return {
        key: state[key]
        for key in (
            "policy_batch_calls",
            "completed_environment_step_callbacks",
            "executed_environment_calls",
            "replay_environment_calls",
            "local_replay_step_calls",
        )
    }


def finished_episode_results(directory, plan, index, records):
    """Read already completed work without model, adapter or evaluator code."""
    path = _work_path(directory, index)
    if not path.is_file():
        return None
    state = _read(path, plan)
    if state["input_sha256"] != json_fingerprint(records) or state["batch"] != index:
        raise ValueError("episode work input population differs")
    if any(trial["result"] is None for trial in state["trials"]):
        return None
    rows = [trial["result"] for trial in state["trials"]]
    if len(rows) != len(records) or [trial["id"] for trial in state["trials"]] != [
        row[plan["fields"]["id"]] for row in records
    ]:
        raise ValueError("episode completed work ID population differs")
    verify_episode_results(rows, records, EpisodeRunConfig.from_dict(plan["episodes"]))
    return rows, _counters(state)


def verify_episode_results(predictions, records, config):
    for row, record in zip(predictions, records, strict=True):
        episode = row.get(config.field)
        expected = config.trial(record)
        if not isinstance(episode, dict) or json_fingerprint(
            {key: episode.get(key) for key in expected}
        ) != json_fingerprint(expected):
            raise ValueError("episode result initial-state/seed/trial differs from original input")
        if (
            type(episode.get("terminated")) is not bool
            or type(episode.get("truncated")) is not bool
            or not (episode["terminated"] or episode["truncated"])
        ):
            raise ValueError("episode result requires truthful termination or truncation")
        if not isinstance(episode.get("trace"), list) or not isinstance(
            episode.get("execution"), dict
        ):
            raise ValueError("episode result requires original trace and execution evidence")


def _resources(value):
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str)
        or not key
        or type(item) not in {int, float}
        or not math.isfinite(item)
        or item < 0
        for key, item in value.items()
    ):
        raise ValueError("episode resource consumption requires named finite nonnegative values")
    if set(value) & {"steps", "environment_calls", "policy_participations"}:
        raise ValueError("application resources cannot replace observed episode counters")
    return dict(value)


def run_episode_batch(
    directory, plan, index, records, predictor, *, attempt, stop_requested=None, on_event=None
):
    # This boundary preserves the caller's surrounding owned state/RNG. The
    # declared native lifetime is separately enforced at every policy step;
    # application trial state lives in the explicit work record, not the model.
    with predictor.binding.evaluation_context():
        return _run_episode_batch(
            directory,
            plan,
            index,
            records,
            predictor,
            attempt=attempt,
            stop_requested=stop_requested,
            on_event=on_event,
        )


def _run_episode_batch(
    directory, plan, index, records, predictor, *, attempt, stop_requested=None, on_event=None
):
    """Advance independent trial states and stage results for the existing part commit."""
    config = EpisodeRunConfig.from_dict(plan["episodes"])
    path = _work_path(directory, index)
    input_sha = json_fingerprint(records)
    phase = "initial" if attempt == 1 else "recovered"
    if path.is_file():
        state = _read(path, plan)
        if (
            state["input_sha256"] != input_sha
            or state["batch"] != index
            or [trial["id"] for trial in state["trials"]]
            != [row[plan["fields"]["id"]] for row in records]
        ):
            raise ValueError("episode work input population differs")
        blocked = [
            {"id": trial["id"], "reason": reason}
            for trial in state["trials"]
            if (reason := _blocker(trial, config)) is not None
        ]
        if blocked:
            raise ValueError(f"episode recovery is unsupported: {blocked}")
    else:
        state = {
            "format": "arti.prediction-episode-work",
            "plan_sha256": plan["sha256"],
            "input_sha256": input_sha,
            "batch": index,
            "policy_batch_calls": 0,
            "executed_environment_calls": 0,
            "completed_environment_step_callbacks": 0,
            "replay_environment_calls": 0,
            "local_replay_step_calls": 0,
            "trials": [],
        }
        for record in records:
            state["trials"].append(
                {
                    "id": record[plan["fields"]["id"]],
                    **config.trial(record),
                    "observation": None,
                    "initialized": False,
                    "policy_state": None,
                    "snapshot": None,
                    "trace": [],
                    "pending": None,
                    "result": None,
                    "resources": {"steps": 0, "environment_calls": 0, "policy_participations": 0},
                    "attempts": [],
                }
            )
        _save(path, state)
    if all(trial["result"] is not None for trial in state["trials"]):
        return [trial["result"] for trial in state["trials"]], _counters(state)
    adapter = config.adapter.resolve()(config, predictor=predictor)
    if not isinstance(adapter, EpisodeAdapter):
        raise ValueError("episode factory must return EpisodeAdapter")
    if config.recovery == "snapshot" and (adapter.snapshot is None or adapter.restore is None):
        raise ValueError(
            "snapshot recovery requires declared environment snapshot and restore callbacks"
        )
    environments = {}

    def event(kind, trial):
        if on_event is not None:
            on_event(
                {
                    "kind": kind,
                    "id": trial["id"],
                    "batch": index,
                    "step": len(trial["trace"]),
                    "attempt": attempt,
                    "phase": phase,
                }
            )

    def persist(kind=None, trial=None):
        _save(path, state)
        if kind is not None:
            event(kind, trial)

    def snapshot(env):
        return (
            _json(adapter.snapshot(env), "environment snapshot")
            if config.recovery == "snapshot"
            else None
        )

    for record, trial in zip(records, state["trials"], strict=True):
        if trial["result"] is not None:
            continue
        if attempt not in trial["attempts"]:
            trial["attempts"].append(attempt)
        if trial["initialized"]:
            if config.recovery == "snapshot":
                env = adapter.restore(deepcopy(record), deepcopy(trial["snapshot"]))
            else:
                env = adapter.start(
                    deepcopy(record), seed=trial["seed"], initial_state=trial["initial_state"]
                )
                for transition in trial["trace"]:
                    if json_fingerprint(_json(adapter.observe(env))) != json_fingerprint(
                        transition["observation"]
                    ):
                        raise ValueError(
                            "local environment replay differs from original observation"
                        )
                    replayed = adapter.step(env, deepcopy(transition["action"]))
                    state["local_replay_step_calls"] += 1
                    persist()
                    replayed = _json(replayed, "replayed response")
                    observed = _json(adapter.observe(env), "replayed observation")
                    calls = replayed.get("calls", 1)
                    if type(calls) is not int or calls < 0:
                        raise ValueError("replayed environment calls must be a nonnegative integer")
                    actual = {
                        "response": replayed.get("response"),
                        "terminated": replayed.get("terminated"),
                        "next_observation": observed,
                        "reason": replayed.get("reason"),
                        "calls": calls,
                        "resources": _resources(replayed.get("resources", {})),
                    }
                    state["replay_environment_calls"] += calls
                    persist()
                    if json_fingerprint(actual) != json_fingerprint(
                        {key: transition[key] for key in actual}
                    ):
                        raise ValueError(
                            "local environment replay differs from original response/observation"
                        )
                trial.setdefault("recovery_events", []).append(
                    {
                        "attempt": attempt,
                        "mode": "local-replay",
                        "replayed_actions": len(trial["trace"]),
                    }
                )
            if json_fingerprint(_json(adapter.observe(env))) != json_fingerprint(
                trial["observation"]
            ):
                raise ValueError("restored environment observation differs from checkpoint")
            if config.recovery == "snapshot":
                trial.setdefault("recovery_events", []).append(
                    {"attempt": attempt, "mode": "snapshot", "external_effects_rolled_back": False}
                )
            environments[trial["id"]] = env
            persist()
        else:
            trial["pending"] = {"kind": "initialize", "attempt": attempt, "phase": phase}
            persist("intent", trial)
            env = adapter.start(
                deepcopy(record), seed=trial["seed"], initial_state=trial["initial_state"]
            )
            trial.update(
                observation=_json(adapter.observe(env), "observation"),
                policy_state=_json(
                    adapter.initial_policy(deepcopy(record)) if adapter.initial_policy else None,
                    "policy state",
                ),
                snapshot=snapshot(env),
                pending=None,
                initialized=True,
            )
            environments[trial["id"]] = env
            persist("initialized", trial)

    def finish(trial, *, terminated, reason):
        env = environments[trial["id"]]
        outcome = _json(adapter.evaluate(env, deepcopy(trial["trace"])), "environment outcome")
        episode = {
            **{key: trial[key] for key in ("initial_state", "seed", "trace", "resources")},
            **({"trial": trial["trial"]} if "trial" in trial else {}),
            "state_lifetimes": {
                "policy": config.policy_state_lifetime,
                "environment": config.environment_state_lifetime,
                "native_model": config.native_model_state_lifetime,
            },
            "terminated": terminated,
            "truncated": not terminated,
            "termination_reason": reason,
            "outcome": outcome,
            "policy_dispatch": {
                "independent_batch": config.independent_batch,
                "scope": config.dispatch_scope,
                "counter": "policy_batch_calls counts callback dispatches including singletons",
            },
            "resource_attribution": {
                "steps": "library-completed-transitions",
                "environment_calls": "application-response-reported-consumption",
                "policy_participations": "trial-participations-in-policy-batch-dispatches",
                "other_resources": "application-response-reported",
            },
            "allowances": {
                "max_steps": config.max_steps,
                "max_calls": config.max_calls,
                "scope": "episode-step-starts; one callback may overrun reported-call allowance",
                "host_enforced": False,
            },
            "execution": {
                "phase": phase,
                "attempt": attempt,
                "attempts": trial["attempts"],
                "trace_phases": list(dict.fromkeys(item["phase"] for item in trial["trace"])),
                "outcomes": [
                    {
                        "attempt": number,
                        "phase": "initial" if number == 1 else "recovered",
                        "availability": "observed" if number == attempt else "unavailable",
                        "reason": "environment_outcome_evaluated"
                        if number == attempt
                        else "no_terminal_outcome_recorded",
                        "outcome": deepcopy(outcome) if number == attempt else None,
                    }
                    for number in trial["attempts"]
                ],
            },
            "recovery": {
                "mode": config.recovery,
                "effects": config.effects,
                "events": trial.get("recovery_events", []),
                "exactly_once_external_effects": False,
            },
        }
        unissued = trial.get("pending") or trial.get("proposal")
        if not terminated and unissued is not None and unissued.get("kind") == "action":
            episode["unexecuted_action"] = {
                "action": deepcopy(unissued["action"]),
                "observation": deepcopy(trial["observation"]),
                "status": "not_executed",
                "attempt": attempt,
                "phase": phase,
                "reason": reason,
                "policy_attempt": unissued["attempt"],
                "policy_phase": unissued["phase"],
                "prior_effect_outcome": "unrecorded"
                if trial.get("pending") is not None
                else "not_issued",
            }
        trial["result"] = {plan["output_id_field"]: trial["id"], config.field: _json(episode)}
        persist("finished", trial)

    def execute(trial):
        pending = trial["pending"]
        env = environments[trial["id"]]
        response = adapter.step(env, deepcopy(pending["action"]))
        state["completed_environment_step_callbacks"] += 1
        persist()
        response = _json(response, "environment response")
        if (
            not isinstance(response, dict)
            or "response" not in response
            or type(response.get("terminated")) is not bool
        ):
            raise ValueError(
                "environment step must return response and explicit terminated boolean"
            )
        calls = response.get("calls", 1)
        if type(calls) is not int or calls < 0:
            raise ValueError("observed environment calls must be a nonnegative integer")
        consumption = _resources(response.get("resources", {}))
        reason = response.get("reason")
        if reason is not None:
            _name(reason, "termination reason")
        observation = _json(adapter.observe(env), "observation")
        next_snapshot = snapshot(env)
        transition = {
            "step": len(trial["trace"]),
            "observation": trial["observation"],
            "action": pending["action"],
            "response": response["response"],
            "next_observation": observation,
            "terminated": response["terminated"],
            "reason": reason,
            "resources": consumption,
            "calls": calls,
            "attempt": attempt,
            "phase": phase,
            "policy_attempt": pending["attempt"],
            "policy_phase": pending["phase"],
        }
        trial["trace"].append(transition)
        trial["resources"]["steps"] += 1
        trial["resources"]["environment_calls"] += calls
        for name, value in consumption.items():
            trial["resources"][name] = trial["resources"].get(name, 0) + value
        state["executed_environment_calls"] += calls
        trial.update(observation=observation, snapshot=next_snapshot, pending=None)
        persist("response", trial)
        if response["terminated"]:
            finish(trial, terminated=True, reason=reason or "environment_terminated")

    def stop_reason():
        value = stop_requested() if stop_requested is not None else False
        return value if isinstance(value, str) and value else "requested_stop" if value else None

    def proposed(trial):
        reason = stop_reason()
        if reason is not None:
            finish(trial, terminated=False, reason=reason)
            return
        trial["pending"] = trial.pop("proposal")
        persist("intent", trial)
        execute(trial)

    # A recoverable local pending intent reuses its already produced action/state.
    for trial in state["trials"]:
        if trial["result"] is None and trial["pending"] is not None:
            reason = stop_reason()
            if reason is not None:
                finish(trial, terminated=False, reason=reason)
            else:
                execute(trial)
        elif trial["result"] is None and trial["trace"] and trial["trace"][-1]["terminated"]:
            finish(
                trial,
                terminated=True,
                reason=trial["trace"][-1]["reason"] or "environment_terminated",
            )
        elif trial["result"] is None and trial.get("proposal") is not None:
            proposed(trial)
    while active := [trial for trial in state["trials"] if trial["result"] is None]:
        requested = stop_reason()
        for trial in active:
            reason = (
                requested
                if requested
                else "step_budget"
                if config.max_steps is not None and trial["resources"]["steps"] >= config.max_steps
                else "call_budget"
                if config.max_calls is not None
                and trial["resources"]["environment_calls"] >= config.max_calls
                else None
            )
            if reason is not None:
                finish(trial, terminated=False, reason=reason)
        active = [trial for trial in active if trial["result"] is None]
        if not active:
            continue
        groups = [active] if config.independent_batch else [[trial] for trial in active]
        for group in groups:
            policy_states = []
            for trial in group:
                value = trial["policy_state"]
                if config.policy_state_lifetime == "step":
                    record = records[state["trials"].index(trial)]
                    value = (
                        adapter.initial_policy(deepcopy(record)) if adapter.initial_policy else None
                    )
                policy_states.append(_json(value, "policy state"))
                trial["resources"]["policy_participations"] += 1
            state["policy_batch_calls"] += 1
            persist()
            with predictor.binding.evaluation_context(), torch.no_grad():
                predictor.model.eval()
                proposals = adapter.policy(
                    predictor.model,
                    deepcopy([trial["observation"] for trial in group]),
                    deepcopy(policy_states),
                )
            if not isinstance(proposals, list) or len(proposals) != len(group):
                raise ValueError(
                    "episode policy batch must return one action/state per active trial"
                )
            proposals = _json(proposals, "policy action/state")
            if any(
                not isinstance(value, dict) or "action" not in value or "state" not in value
                for value in proposals
            ):
                raise ValueError("policy proposals require explicit action and state")
            for trial, proposal in zip(group, proposals, strict=True):
                trial["policy_state"] = proposal["state"]
                trial["proposal"] = {
                    "kind": "action",
                    "action": proposal["action"],
                    "attempt": attempt,
                    "phase": phase,
                }
            # Retain successful callback results before another callback/effect.
            # Unissued proposals remain distinct from ambiguous effect intents.
            persist()
        for trial in active:
            proposed(trial)
    return [trial["result"] for trial in state["trials"]], _counters(state)


__all__ = ["EpisodeRunConfig", "EpisodeAdapter"]
