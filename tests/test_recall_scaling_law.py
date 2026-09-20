from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_recall_scaling_law", ROOT / "benchmarks" / "verify_recall_scaling_law.py")
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def payload() -> dict:
    rows = []
    for spec, fields, protected in (("single_allow", 1, False), ("single_abstain", 1, True), ("deep_2", 2, True), ("sharded_2", 2, True), ("deep_sharded_4", 4, True)):
        for count in (4, 8, 24):
            overflowed = count > fields * 8
            rows.append({
                "spec": spec,
                "item_count": count,
                "overflowed": overflowed,
                "protected": protected,
                "mean_mse_improvement": 0.1,
                "mean_coverage": 1.0 if not overflowed else (0.0 if spec == "single_abstain" else min(1.0, fields * 8 / count)),
                "mean_rejected_false_recall_rate": 0.0,
            })
    return {
        "status": "completed",
        "claim_boundary": "The host is frozen. Each cell receives the same optimizer-step budget. Adaptation receives complete support traces and internally corrupted support views only; query targets are evaluation-only.",
        "config": {"seeds": [13, 29, 47], "train_steps": 10},
        "runs": [{"query_targets_exposed_to_adaptation": False, "train_steps": 10, "accepted_recall_misidentification_rate": 0.1, "peak_cuda_memory_bytes": None}],
        "summary": rows,
        "recommendation": {"overflow_is_a_calibration_boundary": True, "protected_overflow_has_no_recalled_output_for_rejected_items": True},
    }


def test_valid_scaling_payload_passes() -> None:
    assert module.verify(payload()) == []


def test_scaling_verifier_rejects_query_target_leakage() -> None:
    data = payload()
    data["runs"][0]["query_targets_exposed_to_adaptation"] = True
    assert any("query targets" in failure for failure in module.verify(data))
