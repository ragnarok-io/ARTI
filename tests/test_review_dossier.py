from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

BUILD_SPEC = importlib.util.spec_from_file_location(
    "build_review_dossier", ROOT / "benchmarks" / "build_review_dossier.py"
)
build_review_dossier = importlib.util.module_from_spec(BUILD_SPEC)
assert BUILD_SPEC.loader is not None
BUILD_SPEC.loader.exec_module(build_review_dossier)

VERIFY_SPEC = importlib.util.spec_from_file_location(
    "verify_review_dossier", ROOT / "benchmarks" / "verify_review_dossier.py"
)
verify_review_dossier = importlib.util.module_from_spec(VERIFY_SPEC)
assert VERIFY_SPEC.loader is not None
VERIFY_SPEC.loader.exec_module(verify_review_dossier)


def minimal_payload() -> dict:
    evidence_summary = {
        "high_power": {
            "available": True,
            "bundle_lock_artifact_count": 18,
            "statistical_records": {
                "task: paired margin arti_full - ablation": {
                    "n": 6,
                    "mean": 0.1,
                    "positive_fraction": 1.0,
                    "sign_test_p_two_sided": 0.03125,
                    "bootstrap_ci95": [0.05, 0.15],
                }
            },
        }
    }
    gap_audit = {
        "overall_status": "incomplete",
        "met_count": 3,
        "partial_count": 1,
        "unmet_count": 0,
        "items": [{"name": "independent_second_machine_reproduction", "status": "partial"}],
    }
    claim_ledger = {
        "supported_local": [{"claim": "local claim"}],
        "planning_only": [{"claim": "planned claim"}],
        "prohibited": [{"claim": "ARTI has Nature-level validation.", "reason": "gap audit incomplete"}],
    }
    return build_review_dossier.build_dossier(evidence_summary, gap_audit, claim_ledger, {}, {"artifacts": [], "commands": []})


def test_valid_review_dossier_passes() -> None:
    assert verify_review_dossier.verify(minimal_payload()) == []


def test_review_dossier_requires_conservative_verdict() -> None:
    payload = minimal_payload()
    payload["verdict"] = "nature_complete"
    failures = verify_review_dossier.verify(payload)
    assert "review dossier verdict must remain not_nature_complete" in failures


def test_review_dossier_requires_entrypoints() -> None:
    payload = minimal_payload()
    payload["required_entrypoints"] = []
    failures = verify_review_dossier.verify(payload)
    assert any("missing required entrypoints" in failure for failure in failures)
