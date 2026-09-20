from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_pretrained_ecosystem_smoke", ROOT / "benchmarks" / "verify_pretrained_ecosystem_smoke.py")
verify_module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_module)


def valid_payload() -> dict:
    common = {
        "model_id": "model",
        "resolved_revision": "abc123",
        "adapter_count": 1,
        "identity_max_abs_error": 0.0,
        "roundtrip_max_abs_error": 0.0,
        "lock_valid": True,
    }
    return {
        "status": "completed",
        "scope": "integration and reproducibility smoke; not downstream quality superiority",
        "qwen": {**common, "provider": "peft", "peft_api_present": True, "generation_identical": True, "kv_cache_present": True},
        "vit": {
            **common,
            "training_engine": "accelerate",
            "training_steps": 2,
            "mixed_precision": "bf16",
            "finite_losses": True,
            "trainer_compatible": True,
        },
        "diffusers": {**common, "pipeline_api_preserved": True, "pipeline_identity_max_abs_error": 0.0},
    }


def test_pretrained_ecosystem_smoke_verifier_accepts_complete_evidence() -> None:
    assert verify_module.verify(valid_payload()) == []


def test_pretrained_ecosystem_smoke_verifier_rejects_non_identity_apply() -> None:
    payload = valid_payload()
    payload["qwen"]["identity_max_abs_error"] = 0.01
    assert any("identity" in failure for failure in verify_module.verify(payload))
