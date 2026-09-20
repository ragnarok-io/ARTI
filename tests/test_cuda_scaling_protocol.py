from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_cuda_scaling", ROOT / "benchmarks" / "verify_cuda_scaling.py")
assert SPEC is not None
verify_cuda_scaling = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_cuda_scaling)


def protocol() -> dict:
    return {
        "status": "planned_not_executed",
        "required_device_prefix": "cuda",
        "required_models": ["arti_interface_only", "arti_pairwise", "transformer"],
        "required_tokens": [16],
        "required_fields": ["mean_ms", "estimated_activation_bytes", "cuda_peak_allocated_bytes"],
        "required_device_report_fields": ["cuda_available", "device_name", "device_capability", "cuda_version"],
    }


def profile(device: str = "cuda", peak: int | None = 1234) -> dict:
    return {
        "config": {"device": device},
        "cuda_device_report": {
            "cuda_available": device.startswith("cuda"),
            "device": device,
            "device_name": "Test CUDA Device" if device.startswith("cuda") else None,
            "device_capability": [8, 0] if device.startswith("cuda") else None,
            "cuda_version": "12.1" if device.startswith("cuda") else None,
        },
        "rows": [
            {
                "model": model,
                "tokens": 16,
                "mean_ms": 1.0,
                "estimated_activation_bytes": 10,
                "cuda_peak_allocated_bytes": peak,
            }
            for model in ("arti_interface_only", "arti_pairwise", "transformer")
        ],
    }


def test_cuda_protocol_passes_without_profile_when_planned() -> None:
    assert verify_cuda_scaling.verify_protocol(protocol()) == []


def test_cuda_profile_passes_when_all_required_fields_exist() -> None:
    assert verify_cuda_scaling.verify_results(protocol(), profile()) == []


def test_cuda_profile_fails_on_cpu_device() -> None:
    failures = verify_cuda_scaling.verify_results(protocol(), profile(device="cpu"))

    assert any("not cuda" in failure for failure in failures)


def test_cuda_profile_fails_on_missing_peak_memory() -> None:
    failures = verify_cuda_scaling.verify_results(protocol(), profile(peak=None))

    assert any("cuda_peak_allocated_bytes" in failure for failure in failures)


def test_cuda_profile_requires_device_report() -> None:
    payload = profile()
    payload["cuda_device_report"] = {}

    failures = verify_cuda_scaling.verify_results(protocol(), payload)

    assert any("cuda_device_report" in failure for failure in failures)
