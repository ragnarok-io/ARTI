from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_decisive_recall", ROOT / "benchmarks" / "verify_decisive_recall.py")
verify_decisive_recall = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_decisive_recall)


def payload() -> dict:
    return {
        "provenance": {"seed_values": [0, 1]},
        "scope": "synthetic recall-only expressivity test; not a real-world task benchmark",
        "summary": [
            {"model": "mlp_no_recall", "mean_accuracy": 0.5, "mean_zero_recall_drop": 0.0, "mean_shuffled_recall_drop": 0.0},
            {"model": "arti_no_recall", "mean_accuracy": 0.5, "mean_zero_recall_drop": 0.0, "mean_shuffled_recall_drop": 0.0},
            {"model": "recall_baseline", "mean_accuracy": 0.95, "mean_zero_recall_drop": 0.45, "mean_shuffled_recall_drop": 0.45},
            {"model": "arti_full", "mean_accuracy": 0.95, "mean_zero_recall_drop": 0.45, "mean_shuffled_recall_drop": 0.45},
        ],
    }


def test_valid_decisive_recall_passes() -> None:
    assert verify_decisive_recall.verify(payload()) == []


def test_decisive_recall_rejects_weak_arti_full() -> None:
    data = payload()
    data["summary"][-1]["mean_accuracy"] = 0.7
    failures = verify_decisive_recall.verify(data)
    assert any("accuracy below decisive gate" in failure for failure in failures)
