from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_SPEC = importlib.util.spec_from_file_location(
    "build_cuda_evidence_packet", ROOT / "benchmarks" / "build_cuda_evidence_packet.py"
)
build_cuda_evidence_packet = importlib.util.module_from_spec(BUILD_SPEC)
assert BUILD_SPEC.loader is not None
BUILD_SPEC.loader.exec_module(build_cuda_evidence_packet)

VERIFY_SPEC = importlib.util.spec_from_file_location(
    "verify_cuda_evidence_packet", ROOT / "benchmarks" / "verify_cuda_evidence_packet.py"
)
verify_cuda_evidence_packet = importlib.util.module_from_spec(VERIFY_SPEC)
assert VERIFY_SPEC.loader is not None
VERIFY_SPEC.loader.exec_module(verify_cuda_evidence_packet)


def protocol() -> dict:
    return {
        "command": "uv run --extra dev python benchmarks/profile_scaling.py --device cuda --tokens 16",
        "verification_command": "uv run --extra dev python benchmarks/verify_cuda_scaling.py",
        "required_models": ["arti_interface_only", "arti_pairwise", "transformer"],
        "required_tokens": [16],
        "required_fields": ["mean_ms", "estimated_activation_bytes", "cuda_peak_allocated_bytes"],
        "required_device_report_fields": ["cuda_available", "device_name", "device_capability", "cuda_version"],
    }


def test_planned_cuda_packet_passes() -> None:
    packet = build_cuda_evidence_packet.build_packet(protocol(), None)
    assert verify_cuda_evidence_packet.verify(packet) == []
    assert packet["status"] == "planned_not_executed"


def test_cuda_packet_requires_peak_memory_field() -> None:
    payload = build_cuda_evidence_packet.build_packet(protocol(), None)
    payload["required_fields"] = ["mean_ms"]
    failures = verify_cuda_evidence_packet.verify(payload)
    assert any("cuda_peak_allocated_bytes" in failure for failure in failures)


def test_generated_cuda_packet_requires_cuda_available() -> None:
    payload = build_cuda_evidence_packet.build_packet(protocol(), {"cuda_device_report": {"cuda_available": False}})
    failures = verify_cuda_evidence_packet.verify(payload)
    assert any("cuda_available=true" in failure for failure in failures)
