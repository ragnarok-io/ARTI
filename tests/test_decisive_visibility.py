from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_decisive_visibility", ROOT / "benchmarks" / "verify_decisive_visibility.py")
verify_decisive_visibility = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_decisive_visibility)


def payload() -> dict:
    return {
        "provenance": {"seed_values": [0, 1]},
        "scope": "synthetic visibility-only expressivity test; not a real-world task benchmark",
        "summary": [
            {"model": "mlp_no_visibility", "mean_accuracy": 0.5, "mean_no_visibility_drop": 0.0, "mean_all_visibility_drop": 0.0},
            {"model": "transformer_no_visibility", "mean_accuracy": 0.5, "mean_no_visibility_drop": 0.0, "mean_all_visibility_drop": 0.0},
            {"model": "arti_no_visibility", "mean_accuracy": 0.5, "mean_no_visibility_drop": 0.0, "mean_all_visibility_drop": 0.0},
            {"model": "visibility_transformer", "mean_accuracy": 0.95, "mean_no_visibility_drop": 0.45, "mean_all_visibility_drop": 0.45},
            {"model": "arti_full", "mean_accuracy": 0.95, "mean_no_visibility_drop": 0.45, "mean_all_visibility_drop": 0.45},
        ],
    }


def test_valid_decisive_visibility_passes() -> None:
    assert verify_decisive_visibility.verify(payload()) == []


def test_decisive_visibility_rejects_informative_no_visibility_baseline() -> None:
    data = payload()
    data["summary"][0]["mean_accuracy"] = 0.8
    failures = verify_decisive_visibility.verify(data)
    assert any("near chance" in failure for failure in failures)
