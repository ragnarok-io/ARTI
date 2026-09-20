from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_independent_reproduction_protocol", ROOT / "benchmarks" / "verify_independent_reproduction_protocol.py"
)
assert SPEC is not None
verify_independent_reproduction_protocol = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_independent_reproduction_protocol)


def protocol() -> dict:
    return {
        "status": "planned_not_executed",
        "required_return_artifacts": sorted(verify_independent_reproduction_protocol.REQUIRED_RETURN_ARTIFACTS),
        "required_commands": [
            "uv run --extra dev python benchmarks/run_validation_suite.py --quick --fail-fast",
            "uv run --extra dev python benchmarks/verify_evidence_bundle_lock.py",
        ],
        "comparison_commands": [
            "uv run --extra dev python benchmarks/compare_reproduction.py",
            "uv run --extra dev python benchmarks/compare_evidence_bundle_locks.py",
        ],
        "acceptance_rules": sorted(verify_independent_reproduction_protocol.REQUIRED_ACCEPTANCE_RULES),
        "interpretation": "This is not complete until artifacts from a separate machine are returned.",
    }


def test_valid_independent_reproduction_protocol_passes() -> None:
    assert verify_independent_reproduction_protocol.verify(protocol()) == []


def test_protocol_requires_bundle_lock_comparison() -> None:
    payload = protocol()
    payload["comparison_commands"] = ["uv run --extra dev python benchmarks/compare_reproduction.py"]

    failures = verify_independent_reproduction_protocol.verify(payload)

    assert any("compare_evidence_bundle_locks" in failure for failure in failures)


def test_protocol_requires_return_artifacts() -> None:
    payload = protocol()
    payload["required_return_artifacts"] = []

    failures = verify_independent_reproduction_protocol.verify(payload)

    assert any("missing return artifacts" in failure for failure in failures)


def test_protocol_rejects_completed_status_without_artifacts() -> None:
    payload = protocol()
    payload["status"] = "complete"

    failures = verify_independent_reproduction_protocol.verify(payload)

    assert any("status" in failure for failure in failures)
