from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_pretrained_distributed_smoke",
    ROOT / "benchmarks" / "verify_pretrained_distributed_smoke.py",
)
verify_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_module)


def valid_payload() -> dict:
    return {
        "status": "completed",
        "engine": "accelerate",
        "processes": 2,
        "distributed_type": "DistributedType.MULTI_CPU",
        "device": "cpu:0",
        "steps": 2,
        "finite_losses": True,
    }


def test_distributed_smoke_verifier_accepts_two_process_training() -> None:
    assert verify_module.verify(valid_payload()) == []


def test_distributed_smoke_verifier_rejects_single_process() -> None:
    payload = valid_payload()
    payload["processes"] = 1
    assert any("fewer than two" in failure for failure in verify_module.verify(payload))
