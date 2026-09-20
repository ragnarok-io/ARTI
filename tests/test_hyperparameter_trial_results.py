from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_SPEC = importlib.util.spec_from_file_location(
    "build_hyperparameter_results_packet", ROOT / "benchmarks" / "build_hyperparameter_results_packet.py"
)
build_hyperparameter_results_packet = importlib.util.module_from_spec(BUILD_SPEC)
assert BUILD_SPEC.loader is not None
BUILD_SPEC.loader.exec_module(build_hyperparameter_results_packet)

VERIFY_SPEC = importlib.util.spec_from_file_location(
    "verify_hyperparameter_trial_results", ROOT / "benchmarks" / "verify_hyperparameter_trial_results.py"
)
verify_hyperparameter_trial_results = importlib.util.module_from_spec(VERIFY_SPEC)
assert VERIFY_SPEC.loader is not None
VERIFY_SPEC.loader.exec_module(verify_hyperparameter_trial_results)


def plan() -> dict:
    return {
        "trial_count": 4,
        "benchmarks": [{"benchmark_id": "b"}],
        "model_families": ["mlp", "transformer"],
        "trials": [
            {"trial_id": "b::mlp::00", "benchmark_id": "b", "model_family": "mlp", "config_index": 0},
            {"trial_id": "b::mlp::01", "benchmark_id": "b", "model_family": "mlp", "config_index": 1},
            {"trial_id": "b::transformer::00", "benchmark_id": "b", "model_family": "transformer", "config_index": 0},
            {"trial_id": "b::transformer::01", "benchmark_id": "b", "model_family": "transformer", "config_index": 1},
        ],
    }


def protocol() -> dict:
    return {
        "selection_metric": "validation_accuracy",
        "primary_metric": "test_accuracy",
        "minimum_reporting": [
            "all_trial_scores",
            "selected_config_per_model_family",
            "single_test_score_per_selected_config",
        ],
    }


def results() -> dict:
    return {
        "trial_results": [
            {
                "trial_id": "b::mlp::00",
                "benchmark_id": "b",
                "model_family": "mlp",
                "config_index": 0,
                "validation_accuracy": 0.8,
                "status": "completed",
            },
            {
                "trial_id": "b::mlp::01",
                "benchmark_id": "b",
                "model_family": "mlp",
                "config_index": 1,
                "validation_accuracy": 0.7,
                "status": "completed",
            },
            {
                "trial_id": "b::transformer::00",
                "benchmark_id": "b",
                "model_family": "transformer",
                "config_index": 0,
                "validation_accuracy": 0.7,
                "status": "completed",
            },
            {
                "trial_id": "b::transformer::01",
                "benchmark_id": "b",
                "model_family": "transformer",
                "config_index": 1,
                "validation_accuracy": 0.6,
                "status": "completed",
            },
        ],
        "selected_configs": [
            {
                "benchmark_id": "b",
                "model_family": "mlp",
                "selected_trial_id": "b::mlp::00",
                "selection_metric": "validation_accuracy",
                "validation_accuracy": 0.8,
                "test_accuracy": 0.75,
            },
            {
                "benchmark_id": "b",
                "model_family": "transformer",
                "selected_trial_id": "b::transformer::00",
                "selection_metric": "validation_accuracy",
                "validation_accuracy": 0.7,
                "test_accuracy": 0.72,
            },
        ],
    }


def test_schema_packet_passes() -> None:
    packet = build_hyperparameter_results_packet.build_packet(plan(), protocol())
    assert verify_hyperparameter_trial_results.verify_schema_packet(packet, plan(), protocol()) == []


def test_trial_results_pass() -> None:
    assert verify_hyperparameter_trial_results.verify_results(results(), plan(), protocol()) == []


def test_trial_results_reject_unselected_test_scores() -> None:
    payload = results()
    payload["trial_results"][0]["test_accuracy"] = 0.75
    failures = verify_hyperparameter_trial_results.verify_results(payload, plan(), protocol())
    assert any("must not report test_accuracy" in failure for failure in failures)


def test_trial_results_require_all_planned_trials() -> None:
    payload = results()
    payload["trial_results"].pop()
    failures = verify_hyperparameter_trial_results.verify_results(payload, plan(), protocol())
    assert any("trial_results length" in failure for failure in failures)


def test_partial_trial_results_can_be_verified_explicitly() -> None:
    payload = results()
    payload["trial_results"] = payload["trial_results"][:1]
    payload["selected_configs"] = payload["selected_configs"][:1]
    failures = verify_hyperparameter_trial_results.verify_results(payload, plan(), protocol(), require_complete=False)
    assert failures == []


def test_trial_results_require_validation_best_selection() -> None:
    payload = results()
    payload["trial_results"][1]["validation_accuracy"] = 0.9
    failures = verify_hyperparameter_trial_results.verify_results(payload, plan(), protocol())
    assert any("not validation-best" in failure for failure in failures)
