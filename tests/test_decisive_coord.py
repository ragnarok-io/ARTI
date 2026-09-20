from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_decisive_coord", ROOT / "benchmarks" / "verify_decisive_coord.py")
verify_decisive_coord = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_decisive_coord)


def payload() -> dict:
    return {
        "provenance": {"seed_values": [0, 1]},
        "scope": "synthetic coordinate-only expressivity test; not a real-world task benchmark",
        "summary": [
            {"model": "mlp_no_coord", "mean_accuracy": 0.5, "mean_zero_coord_drop": 0.0, "mean_shuffled_coord_drop": 0.0},
            {"model": "arti_no_coord", "mean_accuracy": 0.5, "mean_zero_coord_drop": 0.0, "mean_shuffled_coord_drop": 0.0},
            {"model": "coord_baseline", "mean_accuracy": 0.95, "mean_zero_coord_drop": 0.45, "mean_shuffled_coord_drop": 0.45},
            {"model": "arti_full", "mean_accuracy": 0.95, "mean_zero_coord_drop": 0.45, "mean_shuffled_coord_drop": 0.45},
        ],
    }


def test_valid_decisive_coord_passes() -> None:
    assert verify_decisive_coord.verify(payload()) == []


def test_decisive_coord_rejects_missing_coord_drop() -> None:
    data = payload()
    data["summary"][-1]["mean_zero_coord_drop"] = 0.1
    failures = verify_decisive_coord.verify(data)
    assert any("zero-coord drop below gate" in failure for failure in failures)
