from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_SPEC = importlib.util.spec_from_file_location(
    "build_reproduction_packet", ROOT / "benchmarks" / "build_reproduction_packet.py"
)
build_reproduction_packet = importlib.util.module_from_spec(BUILD_SPEC)
assert BUILD_SPEC.loader is not None
BUILD_SPEC.loader.exec_module(build_reproduction_packet)

VERIFY_SPEC = importlib.util.spec_from_file_location(
    "verify_reproduction_packet", ROOT / "benchmarks" / "verify_reproduction_packet.py"
)
verify_reproduction_packet = importlib.util.module_from_spec(VERIFY_SPEC)
assert VERIFY_SPEC.loader is not None
VERIFY_SPEC.loader.exec_module(verify_reproduction_packet)


def packet() -> dict:
    artifacts = [
        {"path": path, "sha256": f"hash-{index}", "seed_count": None}
        for index, path in enumerate(verify_reproduction_packet.REQUIRED_ARTIFACTS)
    ]
    return build_reproduction_packet.build_packet(
        {
            "required_commands": ["uv run --extra dev python benchmarks/run_validation_suite.py --quick --fail-fast"],
            "required_return_artifacts": list(verify_reproduction_packet.REQUIRED_ARTIFACTS),
            "comparison_commands": [
                "uv run --extra dev python benchmarks/compare_reproduction.py",
                "uv run --extra dev python benchmarks/compare_evidence_bundle_locks.py",
            ],
            "acceptance_rules": ["second_machine_validation_suite_passed"],
        },
        {"artifacts": artifacts},
        {"overall_status": "incomplete"},
        {"supported_local": [{"claim": "local"}], "prohibited": [{"claim": "Nature-level validation"}]},
    )


def test_valid_reproduction_packet_passes() -> None:
    assert verify_reproduction_packet.verify(packet()) == []


def test_reproduction_packet_requires_separate_machine_boundary() -> None:
    payload = packet()
    payload["interpretation"] = "local only"
    failures = verify_reproduction_packet.verify(payload)
    assert any("separate machine" in failure for failure in failures)


def test_reproduction_packet_requires_reference_hashes() -> None:
    payload = packet()
    artifact = next(iter(verify_reproduction_packet.REQUIRED_ARTIFACTS))
    payload["reference_artifacts"][artifact]["sha256"] = None
    failures = verify_reproduction_packet.verify(payload)
    assert any("lacks sha256" in failure for failure in failures)
