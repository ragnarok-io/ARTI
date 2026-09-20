from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_claim_ledger", ROOT / "benchmarks" / "verify_claim_ledger.py")
assert SPEC is not None
verify_claim_ledger = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_claim_ledger)


def ledger() -> dict:
    return {
        "status": "active",
        "supported_local": [{"claim": "local", "evidence": ["a"], "required_qualifier": "local"}],
        "planning_only": [{"claim": "plan", "evidence": ["b"], "required_qualifier": "not executed"}],
        "prohibited": [
            {"claim": "Transformer superiority", "reason": "no"},
            {"claim": "downstream improvement", "reason": "no"},
            {"claim": "Nature-level validation", "reason": "no"},
            {"claim": "public benchmark", "reason": "no"},
            {"claim": "CUDA evidence generalizes", "reason": "no"},
        ],
    }


def test_valid_claim_ledger_passes() -> None:
    assert verify_claim_ledger.verify(ledger(), {"overall_status": "incomplete"}) == []


def test_claim_ledger_requires_prohibited_coverage() -> None:
    payload = ledger()
    payload["prohibited"] = []

    failures = verify_claim_ledger.verify(payload, {"overall_status": "incomplete"})

    assert any("prohibited" in failure for failure in failures)


def test_planning_claim_must_be_qualified_as_not_executed() -> None:
    payload = ledger()
    payload["planning_only"][0]["required_qualifier"] = "registered"

    failures = verify_claim_ledger.verify(payload, {"overall_status": "incomplete"})

    assert any("not executed" in failure for failure in failures)
