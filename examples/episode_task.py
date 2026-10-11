"""Application-owned local episodes over an existing native regression predictor.

The environment is an in-memory numeric process. It performs no external tools
or real-world effects. The policy's claim is deliberately separate from the
environment's measured outcome; its estimate comes from the supplied model.
"""

from __future__ import annotations

from copy import deepcopy

import arti
from arti.recipe_data import json_fingerprint
from examples.custom_task_recipe import _predict


ADAPTER = "examples.episode_task:build_adapter"


def start(record, *, seed, initial_state):
    values = record["state"]
    if initial_state != json_fingerprint(values):
        raise ValueError("declared initial state differs from application state")
    return {
        "id": record["id"],
        "value": values["value"],
        "goal": values["goal"],
        "horizon": values["horizon"],
        "steps": 0,
        "seed": seed,
    }


def observe(environment):
    return {
        "value": environment["value"],
        "distance": environment["goal"] - environment["value"],
    }


def initial_policy(record):
    return {"calls": 0}


def policy(model, observations, states):
    rows = [
        {"id": str(index), "features": [observation["distance"]]}
        for index, observation in enumerate(observations)
    ]
    predictions = list(_predict(model, rows))
    return [
        {
            "action": {
                "delta": 1 if prediction["prediction"] >= 0 else -1,
                "estimate": prediction["prediction"],
                "claimed_success": True,
            },
            "state": {"calls": state["calls"] + 1},
        }
        for prediction, state in zip(predictions, states, strict=True)
    ]


def step(environment, action):
    environment["value"] += action["delta"]
    environment["steps"] += 1
    terminated = environment["steps"] >= environment["horizon"]
    return {
        "response": {"value": environment["value"], "step": environment["steps"]},
        "terminated": terminated,
        "reason": "environment_horizon" if terminated else None,
        "resources": {"local_updates": 1},
    }


def evaluate(environment, trace):
    return {
        "position": environment["value"],
        "reached": environment["value"] == environment["goal"],
        "environment_steps": environment["steps"],
        "origin": "local environment fixture",
    }


def snapshot(environment):
    return deepcopy(environment)


def restore(record, state):
    return deepcopy(state)


def build_adapter(config, *, predictor):
    return arti.EpisodeAdapter(
        start=start,
        observe=observe,
        policy=policy,
        step=step,
        evaluate=evaluate,
        initial_policy=initial_policy,
        snapshot=snapshot,
        restore=restore,
    )


def score_outcomes(pairs):
    """Task-owned scorer for terminal evidence, without evaluating the model."""
    scores = [
        {
            "id": pair["id"],
            "reached": pair["prediction"]["episode"]["outcome"]["reached"],
            "terminated": pair["prediction"]["episode"]["terminated"],
            "truncated": pair["prediction"]["episode"]["truncated"],
        }
        for pair in pairs
    ]
    return {
        "metrics": {"terminal_fraction": sum(row["terminated"] for row in scores) / len(scores)},
        "scores": scores,
    }


def score_references(pairs):
    scores = [
        {
            "id": pair["id"],
            "agrees": pair["prediction"]["episode"]["outcome"]["reached"]
            == pair["target"]["reached"],
        }
        for pair in pairs
    ]
    return {
        "metrics": {"agreement_fraction": sum(row["agrees"] for row in scores) / len(scores)},
        "scores": scores,
    }
