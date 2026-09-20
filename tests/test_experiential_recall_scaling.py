from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_experiential_recall_scaling",
    ROOT / "benchmarks" / "verify_experiential_recall_scaling.py",
)
verify_experiential_recall_scaling = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_experiential_recall_scaling)


def payload() -> dict:
    return {
        "provenance": {"seed_values": [0, 1]},
        "scope": "mechanism-level experiential recall scaling test; no answer memory or external recall tensor is provided",
        "summary": [
            {
                "model": "no_recall",
                "capacity_index": 2,
                "mean_final_clean_latent_trace_mse": 2.0,
                "mean_corrupt_accuracy": 0.95,
            },
            {
                "model": "shallow_recall",
                "capacity_index": 4,
                "mean_final_clean_latent_trace_mse": 0.50,
                "mean_corrupt_accuracy": 0.96,
            },
            {
                "model": "deep_recall",
                "capacity_index": 16,
                "mean_final_clean_latent_trace_mse": 0.45,
                "mean_corrupt_accuracy": 0.96,
            },
            {
                "model": "wide_deep_recall",
                "capacity_index": 384,
                "mean_final_clean_latent_trace_mse": 0.35,
                "mean_corrupt_accuracy": 0.94,
            },
        ],
        "runs": [{"seed": 0}, {"seed": 1}],
    }


def test_valid_experiential_recall_scaling_passes() -> None:
    assert verify_experiential_recall_scaling.verify(payload()) == []


def test_experiential_recall_scaling_rejects_weak_wide_capacity() -> None:
    data = payload()
    data["summary"][-1]["mean_final_clean_latent_trace_mse"] = 0.49
    failures = verify_experiential_recall_scaling.verify(data)
    assert any("does not improve enough over shallow" in failure for failure in failures)


def test_experiential_recall_scaling_rejects_accuracy_regression() -> None:
    data = payload()
    data["summary"][-1]["mean_corrupt_accuracy"] = 0.80
    failures = verify_experiential_recall_scaling.verify(data)
    assert any("corrupt accuracy regression" in failure for failure in failures)
