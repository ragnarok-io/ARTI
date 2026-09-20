from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_decisive_mask", ROOT / "benchmarks" / "verify_decisive_mask.py")
verify_decisive_mask = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_decisive_mask)


def payload() -> dict:
    return {
        "provenance": {"seed_values": [0, 1]},
        "scope": "synthetic mask-only expressivity test; not a real-world task benchmark",
        "summary": [
            {"model": "mlp_no_mask", "mean_accuracy": 0.5, "mean_all_mask_drop": 0.0},
            {"model": "arti_no_mask", "mean_accuracy": 0.5, "mean_all_mask_drop": 0.0},
            {"model": "mask_baseline", "mean_accuracy": 0.95, "mean_all_mask_drop": 0.45},
            {"model": "arti_full", "mean_accuracy": 0.95, "mean_all_mask_drop": 0.45},
        ],
    }


def test_valid_decisive_mask_passes() -> None:
    assert verify_decisive_mask.verify(payload()) == []


def test_decisive_mask_rejects_weak_drop() -> None:
    data = payload()
    data["summary"][-1]["mean_all_mask_drop"] = 0.1
    failures = verify_decisive_mask.verify(data)
    assert any("all-mask drop below gate" in failure for failure in failures)
