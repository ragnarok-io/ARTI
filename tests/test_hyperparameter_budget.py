from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_hyperparameter_budget", ROOT / "benchmarks" / "verify_hyperparameter_budget.py"
)
assert SPEC is not None
verify_hyperparameter_budget = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_hyperparameter_budget)


def valid_budget() -> dict:
    return {
        "status": "planned_not_executed",
        "selection_metric": "validation_accuracy",
        "max_trials_per_model": 6,
        "shared_training_budget": {"optimizer": "AdamW", "learning_rates": [0.003], "steps": [100]},
        "model_families": {
            "arti_full": {"hidden_dim": [32]},
            "arti_ablation": {"hidden_dim": [32]},
            "mlp": {"hidden_dim": [32]},
            "transformer": {"hidden_dim": [32]},
        },
        "selection_rule": "report the selected config, all trial scores, and test score exactly once.",
    }


def test_valid_hyperparameter_budget_passes() -> None:
    assert verify_hyperparameter_budget.verify(valid_budget()) == []


def test_missing_model_family_fails() -> None:
    budget = valid_budget()
    del budget["model_families"]["transformer"]

    failures = verify_hyperparameter_budget.verify(budget)

    assert any("missing model families" in failure for failure in failures)


def test_excessive_trial_budget_fails() -> None:
    budget = valid_budget()
    budget["max_trials_per_model"] = 100

    failures = verify_hyperparameter_budget.verify(budget)

    assert any("max_trials" in failure for failure in failures)
