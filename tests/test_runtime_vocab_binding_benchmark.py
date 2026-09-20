from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_runtime_vocab_binding",
    ROOT / "benchmarks" / "verify_runtime_vocab_binding.py",
)
verify_runtime_vocab_binding = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_runtime_vocab_binding)


def payload() -> dict:
    return {
        "scope": "synthetic runtime vocab binding effect test; not a language benchmark",
        "summary": [
            {"model": "fixed_head", "mean_accuracy": 0.04},
            {"model": "runtime_vocab_head", "mean_accuracy": 0.98},
        ],
        "runs": [{"seed": 0}, {"seed": 1}],
    }


def test_valid_runtime_vocab_binding_passes() -> None:
    assert verify_runtime_vocab_binding.verify(payload()) == []


def test_runtime_vocab_binding_rejects_strong_fixed_head() -> None:
    data = payload()
    data["summary"][0]["mean_accuracy"] = 0.40
    failures = verify_runtime_vocab_binding.verify(data)
    assert any("fixed-head baseline" in failure for failure in failures)


def test_runtime_vocab_binding_rejects_weak_runtime_head() -> None:
    data = payload()
    data["summary"][1]["mean_accuracy"] = 0.50
    failures = verify_runtime_vocab_binding.verify(data)
    assert any("runtime vocab head" in failure for failure in failures)
