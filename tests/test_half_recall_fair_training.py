from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_half_recall_fair_training",
    ROOT / "benchmarks" / "verify_half_recall_fair_training.py",
)
verify_half_recall_fair_training = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verify_half_recall_fair_training
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_half_recall_fair_training)


def payload() -> dict:
    return json.loads((ROOT / "benchmarks" / "results" / "half_recall_fair_training_results.json").read_text(encoding="utf-8"))


def test_current_half_recall_fair_training_passes() -> None:
    assert verify_half_recall_fair_training.verify(payload()) == []


def test_half_recall_fair_training_rejects_wrong_loss_contract() -> None:
    data = payload()
    data["fairness"] = dict(data["fairness"])
    data["fairness"]["loss"] = "downstream probe loss"

    failures = verify_half_recall_fair_training.verify(data)

    assert any("direct hidden-state reconstruction loss" in failure for failure in failures)


def test_half_recall_fair_training_rejects_parameter_mismatch() -> None:
    data = payload()
    data["summary"] = copy.deepcopy(data["summary"])
    data["summary"][0]["parameters"] += 1

    failures = verify_half_recall_fair_training.verify(data)

    assert any("parameters" in failure for failure in failures)


def test_half_recall_fair_training_rejects_bad_replicate_throughput() -> None:
    data = payload()
    data["summary"] = copy.deepcopy(data["summary"])
    data["summary"][0]["seed_metrics"][0]["samples_per_second"] = 0.0

    failures = verify_half_recall_fair_training.verify(data)

    assert any("samples_per_second must be positive" in failure for failure in failures)
    assert any("samples_per_second must equal" in failure for failure in failures)


def test_half_recall_fair_training_rejects_stale_resource_summary() -> None:
    data = payload()
    data["summary"] = copy.deepcopy(data["summary"])
    data["summary"][0]["train_seconds"] += 1.0
    data["resource_profile"]["min_samples_per_second"] = 1.0

    failures = verify_half_recall_fair_training.verify(data)

    assert any("train_seconds summary" in failure for failure in failures)
    assert any("min_samples_per_second" in failure for failure in failures)


def test_half_recall_fair_training_rejects_identity_regression() -> None:
    data = payload()
    rows = {row["variant"]: row for row in data["summary"]}
    rows["half"]["final_mse"] = rows["identity"]["final_mse"] * 0.95

    failures = verify_half_recall_fair_training.verify(data)

    assert any("target MSE versus identity" in failure for failure in failures)


def test_half_recall_fair_training_rejects_softshrink_retention_loss() -> None:
    data = payload()
    rows = {row["variant"]: row for row in data["summary"]}
    rows["half"]["strong_trace_retention"] = rows["softshrink"]["strong_trace_retention"] + 0.01
    comparisons = {row["comparison"]: row for row in data["comparisons"]}
    comparisons["half_vs_softshrink"]["strong_trace_retention_delta"] = 0.01

    failures = verify_half_recall_fair_training.verify(data)

    assert any("softshrink" in failure for failure in failures)
