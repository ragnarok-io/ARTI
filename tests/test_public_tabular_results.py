from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_public_tabular_results", ROOT / "benchmarks" / "verify_public_tabular_results.py"
)
verify_public_tabular_results = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_public_tabular_results)


def valid_payloads() -> tuple[dict, dict, dict]:
    lock = {
        "input_csv": "benchmarks/results/public_wdbc.csv",
        "sha256": "abc",
        "target": "diagnosis",
        "rows": 569,
        "feature_count": 30,
        "feature_names": [f"f{i}" for i in range(30)],
        "positive_rate": 0.37,
        "adapter_status": "external_public_locked",
    }
    results = {
        "provenance": {"seed_values": [0, 1, 2]},
        "audit": dict(lock),
        "summary": [
            {"model": "arti", "mean_accuracy": 0.88},
            {"model": "mlp", "mean_accuracy": 0.96},
        ],
    }
    registry = {"benchmarks": [{"target": "diagnosis", "expected_feature_count": 30}]}
    return results, lock, registry


def test_valid_public_tabular_results_pass() -> None:
    assert verify_public_tabular_results.verify(*valid_payloads()) == []


def test_downstream_public_tabular_requires_preregistered_models() -> None:
    results, lock, registry = valid_payloads()
    results["provenance"]["model_set"] = "downstream"
    results["summary"] = [
        {"model": "arti_full", "mean_accuracy": 0.88},
        {"model": "arti_ablation", "mean_accuracy": 0.84},
        {"model": "mlp", "mean_accuracy": 0.96},
        {"model": "transformer", "mean_accuracy": 0.90},
    ]
    assert verify_public_tabular_results.verify(results, lock, registry) == []


def test_downstream_public_tabular_rejects_missing_ablation() -> None:
    results, lock, registry = valid_payloads()
    results["provenance"]["model_set"] = "downstream"
    results["summary"] = [
        {"model": "arti_full", "mean_accuracy": 0.88},
        {"model": "mlp", "mean_accuracy": 0.96},
        {"model": "transformer", "mean_accuracy": 0.90},
    ]
    failures = verify_public_tabular_results.verify(results, lock, registry)
    assert any("arti_ablation" in failure for failure in failures)


def test_public_tabular_results_require_external_lock() -> None:
    results, lock, registry = valid_payloads()
    lock["adapter_status"] = "smoke_generated"
    results["audit"] = dict(lock)
    failures = verify_public_tabular_results.verify(results, lock, registry)
    assert "dataset lock adapter_status must be external_public_locked" in failures


def test_public_tabular_results_require_registered_target() -> None:
    results, lock, registry = valid_payloads()
    registry["benchmarks"] = [{"target": "other", "expected_feature_count": 30}]
    failures = verify_public_tabular_results.verify(results, lock, registry)
    assert "dataset lock target/feature_count must match a registered public benchmark" in failures
