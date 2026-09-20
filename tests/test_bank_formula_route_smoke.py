from __future__ import annotations

import importlib.util
import json
import sys
from argparse import Namespace
from pathlib import Path

import torch


SCRIPT = Path(__file__).parents[1] / "benchmarks" / "train_bank_formula_route_smoke.py"
SPEC = importlib.util.spec_from_file_location("bank_formula_route_smoke", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def _score(**changes: float) -> object:
    values = {
        "seed": 1,
        "initial_mse": 0.7,
        "learned_mse": 0.0,
        "fixed_mse": 0.7,
        "random_mse": 0.7,
        "oracle_mse": 0.0,
        "left_route_accuracy": 1.0,
        "right_route_accuracy": 1.0,
        "initial_left_route_accuracy": 0.5,
        "initial_right_route_accuracy": 0.5,
        "elapsed_seconds": 1.0,
    }
    values.update(changes)
    return MODULE.Score(**values)


def test_route_smoke_assessment_is_fail_closed() -> None:
    assert MODULE.assess([_score()]) == []
    assert MODULE.assess([_score(learned_mse=0.2)])
    assert MODULE.assess([_score(left_route_accuracy=0.5)])
    assert MODULE.assess([_score(learned_mse=float("nan"))])


def test_route_smoke_score_is_strict_json() -> None:
    encoded = json.dumps(MODULE.asdict(_score()), allow_nan=False)
    assert json.loads(encoded)["seed"] == 1


def test_route_smoke_cpu_run_is_reproducible_and_passes() -> None:
    args = Namespace(
        steps=80,
        batch_size=64,
        eval_size=256,
        dim=8,
        bank_slots=32,
        learning_rate=0.03,
    )
    first = MODULE.run_seed(args, 7, torch.device("cpu"))
    second = MODULE.run_seed(args, 7, torch.device("cpu"))

    assert MODULE.assess([first]) == []
    assert MODULE.assess([second]) == []
    first_values = MODULE.asdict(first)
    second_values = MODULE.asdict(second)
    first_values.pop("elapsed_seconds")
    second_values.pop("elapsed_seconds")
    assert first_values == second_values
