from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compare_evidence_bundle_locks", ROOT / "benchmarks" / "compare_evidence_bundle_locks.py"
)
assert SPEC is not None
compare_evidence_bundle_locks = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(compare_evidence_bundle_locks)


def artifact(path: str, **extra) -> dict:
    row = {"path": path, "sha256": "a" * 64, "bytes": 1, "seed_count": None}
    row.update(extra)
    return row


def valid_lock() -> dict:
    return {
        "artifacts": [
            artifact(
                "benchmarks/results/high_power_nature_results.json",
                seed_count=6,
                claim_verdicts={
                    "coordinate_routing": "supported",
                    "visibility_reasoning": "supported",
                },
            ),
            artifact("benchmarks/results/high_power_proxy_results.json", seed_count=6),
            artifact("benchmarks/results/high_power_visibility_proxy_results.json", seed_count=6),
            artifact("benchmarks/results/high_power_statistical_audit.json"),
            artifact("benchmarks/high_power_protocol.json", status="generated_local_not_independent"),
            artifact("benchmarks/results/environment_snapshot.json"),
            artifact("benchmarks/results/validation_suite_report.json", passed=True),
        ]
    }


def test_same_bundle_lock_passes() -> None:
    result = compare_evidence_bundle_locks.compare(valid_lock(), valid_lock())

    assert result["passed"] is True
    assert result["failures"] == []


def test_hash_change_is_review_item_not_failure() -> None:
    reference = valid_lock()
    current = valid_lock()
    current["artifacts"][0]["sha256"] = "b" * 64

    result = compare_evidence_bundle_locks.compare(reference, current)

    assert result["passed"] is True
    assert result["hash_change_count"] == 1


def test_low_seed_count_fails() -> None:
    current = valid_lock()
    current["artifacts"][1]["seed_count"] = 3

    result = compare_evidence_bundle_locks.compare(valid_lock(), current)

    assert result["passed"] is False
    assert any("seed count" in failure for failure in result["failures"])


def test_unsupported_verdict_fails() -> None:
    current = valid_lock()
    current["artifacts"][0]["claim_verdicts"]["coordinate_routing"] = "unsupported"

    result = compare_evidence_bundle_locks.compare(valid_lock(), current)

    assert result["passed"] is False
    assert any("verdicts" in failure for failure in result["failures"])
