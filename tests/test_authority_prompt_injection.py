from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_authority_prompt_injection",
    ROOT / "benchmarks" / "verify_authority_prompt_injection.py",
)
verify_authority_prompt_injection = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_authority_prompt_injection)


def payload() -> dict:
    return {
        "scope": "synthetic phase-authority prompt-injection proxy; all tokens visible; not a production security guarantee",
        "summary": [
            {
                "model": "content_only_mlp",
                "mean_accuracy": 0.50,
                "mean_zero_phase_accuracy": 0.50,
                "mean_authority_drop_when_zero_phase": 0.0,
            },
            {
                "model": "content_coord_mlp",
                "mean_accuracy": 1.0,
                "mean_zero_phase_accuracy": 0.50,
                "mean_authority_drop_when_zero_phase": 0.50,
            },
            {
                "model": "arti_phase_authority",
                "mean_accuracy": 1.0,
                "mean_zero_phase_accuracy": 0.50,
                "mean_authority_drop_when_zero_phase": 0.50,
            },
        ],
        "runs": [{"seed": 0}, {"seed": 1}],
    }


def test_valid_authority_prompt_injection_passes() -> None:
    assert verify_authority_prompt_injection.verify(payload()) == []


def test_authority_prompt_injection_rejects_content_leakage() -> None:
    data = payload()
    data["summary"][0]["mean_accuracy"] = 0.75
    failures = verify_authority_prompt_injection.verify(data)
    assert any("content-only baseline" in failure for failure in failures)


def test_authority_prompt_injection_rejects_weak_zero_phase_ablation() -> None:
    data = payload()
    data["summary"][2]["mean_authority_drop_when_zero_phase"] = 0.1
    failures = verify_authority_prompt_injection.verify(data)
    assert any("zero-phase ablation" in failure for failure in failures)
