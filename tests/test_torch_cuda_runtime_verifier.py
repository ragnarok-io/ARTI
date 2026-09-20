import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_torch_cuda_runtime",
    ROOT / "benchmarks" / "verify_torch_cuda_runtime.py",
)
verify_torch_cuda_runtime = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = verify_torch_cuda_runtime
SPEC.loader.exec_module(verify_torch_cuda_runtime)


def hardware_visible_cpu_torch_report() -> dict:
    return {
        "nvidia_devices": [{"index": 0, "name": "NVIDIA GeForce RTX 5070 Ti"}],
        "torch_cuda_available": False,
        "torch_cuda_version": None,
        "torch_version": "2.12.1+cpu",
        "torch_cuda_devices": [],
    }


def test_cuda_runtime_verifier_rejects_visible_gpu_with_cpu_torch():
    failures = verify_torch_cuda_runtime.verify(hardware_visible_cpu_torch_report())
    assert failures
    assert "cannot use CUDA" in failures[0]


def test_cuda_runtime_verifier_allows_cpu_torch_when_explicitly_allowed():
    failures = verify_torch_cuda_runtime.verify(hardware_visible_cpu_torch_report(), allow_cpu_torch=True)
    assert failures == []


def test_cuda_runtime_verifier_accepts_torch_cuda_report():
    report = {
        "nvidia_devices": [{"index": 0, "name": "NVIDIA GeForce RTX 5070 Ti"}],
        "torch_cuda_available": True,
        "torch_cuda_version": "12.8",
        "torch_version": "2.7.0+cu128",
        "torch_cuda_devices": [{"index": 0, "name": "NVIDIA GeForce RTX 5070 Ti"}],
    }
    assert verify_torch_cuda_runtime.verify(report) == []
