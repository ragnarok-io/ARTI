from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_downstream_results", ROOT / "benchmarks" / "verify_downstream_results.py")
verify_downstream_results = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_downstream_results)


def result_payload(model_set: str = "downstream") -> dict:
    models = {
        "arti_full": [0.8, 0.82, 0.81],
        "arti_ablation": [0.79, 0.80, 0.80],
        "mlp": [0.9, 0.91, 0.92],
        "transformer": [0.88, 0.87, 0.89],
    }
    runs = []
    summary = []
    for model, values in models.items():
        for seed, value in enumerate(values):
            runs.append({"model": model, "seed": seed, "validation_accuracy": value - 0.01, "accuracy": value})
        mean = sum(values) / len(values)
        summary.append({"model": model, "mean_accuracy": mean, "mean_validation_accuracy": mean - 0.01, "std_accuracy": 0.01})
    return {"provenance": {"model_set": model_set}, "audit": {"target": "target", "rows": 10}, "summary": summary, "runs": runs}


def trial_plan(path: Path) -> dict:
    benchmark_id = "bench"
    return {
        "benchmarks": [{"benchmark_id": benchmark_id, "results_path": path.as_posix()}],
        "trials": [
            {
                "benchmark_id": benchmark_id,
                "model_family": model,
                "config_index": 0,
                "trial_id": f"{benchmark_id}::{model}::00",
            }
            for model in ("arti_full", "arti_ablation", "mlp", "transformer")
        ],
    }


def test_downstream_report_negative_superiority_passes_verifier(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text("{}", encoding="utf-8")
    report = {
        "minimum_benchmarks": 1,
        "benchmark_count": 1,
        "benchmarks": [verify_downstream_results.evaluate_result(path, result_payload(), trial_plan(path))],
    }
    assert verify_downstream_results.verify(report) == []
    assert report["benchmarks"][0]["mechanism_utility_supported"] is True
    assert report["benchmarks"][0]["baseline_superiority_supported"] is False
    margin = report["benchmarks"][0]["paired_margins"][1]
    assert "bootstrap_ci95" in margin
    assert "sign_test_p_two_sided" in margin


def test_downstream_sign_test_is_exact_two_sided() -> None:
    assert verify_downstream_results.sign_test_p_value([1.0, 1.0, 1.0]) == 0.25
    assert verify_downstream_results.sign_test_p_value([1.0, -1.0, 1.0, -1.0]) == 1.0
    assert verify_downstream_results.sign_test_p_value([0.0, 0.0]) is None


def test_downstream_verifier_requires_validation_metric(tmp_path: Path) -> None:
    payload = result_payload()
    del payload["summary"][0]["mean_validation_accuracy"]
    path = tmp_path / "result.json"
    report = {
        "minimum_benchmarks": 1,
        "benchmark_count": 1,
        "benchmarks": [verify_downstream_results.evaluate_result(path, payload, trial_plan(path))],
    }
    failures = verify_downstream_results.verify(report)
    assert any("lacks mean_validation_accuracy" in failure for failure in failures)


def test_downstream_verifier_requires_downstream_model_set(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    report = {
        "minimum_benchmarks": 1,
        "benchmark_count": 1,
        "benchmarks": [verify_downstream_results.evaluate_result(path, result_payload(model_set="adapter"), trial_plan(path))],
    }
    failures = verify_downstream_results.verify(report)
    assert any("model_set=downstream" in failure for failure in failures)


def test_downstream_verifier_requires_uncertainty_fields(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    benchmark = verify_downstream_results.evaluate_result(path, result_payload(), trial_plan(path))
    del benchmark["paired_margins"][0]["bootstrap_ci95"]
    del benchmark["paired_margins"][1]["sign_test_p_two_sided"]
    report = {"minimum_benchmarks": 1, "benchmark_count": 1, "benchmarks": [benchmark]}

    failures = verify_downstream_results.verify(report)

    assert any("lacks bootstrap_ci95" in failure for failure in failures)
    assert any("lacks sign-test p-value" in failure for failure in failures)


def test_downstream_verifier_requires_trial_plan_mapping(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    report = {
        "minimum_benchmarks": 1,
        "benchmark_count": 1,
        "benchmarks": [verify_downstream_results.evaluate_result(path, result_payload(), None)],
    }

    failures = verify_downstream_results.verify(report)

    assert any("not mapped to a trial-plan entry" in failure for failure in failures)
