from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_SPEC = importlib.util.spec_from_file_location(
    "build_hyperparameter_trial_plan", ROOT / "benchmarks" / "build_hyperparameter_trial_plan.py"
)
build_hyperparameter_trial_plan = importlib.util.module_from_spec(BUILD_SPEC)
assert BUILD_SPEC.loader is not None
BUILD_SPEC.loader.exec_module(build_hyperparameter_trial_plan)

VERIFY_SPEC = importlib.util.spec_from_file_location(
    "verify_hyperparameter_trial_plan", ROOT / "benchmarks" / "verify_hyperparameter_trial_plan.py"
)
verify_hyperparameter_trial_plan = importlib.util.module_from_spec(VERIFY_SPEC)
assert VERIFY_SPEC.loader is not None
VERIFY_SPEC.loader.exec_module(verify_hyperparameter_trial_plan)


def budget() -> dict:
    return {
        "selection_metric": "validation_accuracy",
        "max_trials_per_model": 2,
        "shared_training_budget": {"optimizer": "AdamW", "steps": [100], "learning_rates": [0.003]},
        "model_families": {
            "arti_full": {"hidden_dim": [32, 64]},
            "arti_ablation": {"hidden_dim": [32, 64]},
            "mlp": {"hidden_dim": [32, 64]},
            "transformer": {"hidden_dim": [32, 64]},
        },
    }


def protocol() -> dict:
    return {
        "selection_metric": "validation_accuracy",
        "primary_metric": "test_accuracy",
        "minimum_locked_public_benchmarks": 2,
    }


def test_trial_plan_passes() -> None:
    plan = build_hyperparameter_trial_plan.build_plan(budget(), protocol())
    assert verify_hyperparameter_trial_plan.verify(plan, budget(), protocol()) == []
    assert plan["trial_count"] == 16


def test_trial_plan_rejects_duplicate_ids() -> None:
    plan = build_hyperparameter_trial_plan.build_plan(budget(), protocol())
    plan["trials"][1]["trial_id"] = plan["trials"][0]["trial_id"]
    failures = verify_hyperparameter_trial_plan.verify(plan, budget(), protocol())
    assert any("duplicate trial IDs" in failure for failure in failures)


def test_trial_plan_rejects_missing_family() -> None:
    plan = build_hyperparameter_trial_plan.build_plan(budget(), protocol())
    plan["model_families"] = ["mlp"]
    failures = verify_hyperparameter_trial_plan.verify(plan, budget(), protocol())
    assert any("model families mismatch" in failure for failure in failures)
