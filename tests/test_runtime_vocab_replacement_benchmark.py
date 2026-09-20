from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_runtime_vocab_replacement",
    ROOT / "benchmarks" / "verify_runtime_vocab_replacement.py",
)
verify_runtime_vocab_replacement = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_runtime_vocab_replacement)


def payload() -> dict:
    return {
        "scope": "synthetic runtime vocab replacement effect test; copy-by-symbol, not a language benchmark",
        "summary": [
            {"model": "fixed_head", "mean_heldout_replacement_accuracy": 0.06},
            {"model": "runtime_vocab_head", "mean_heldout_replacement_accuracy": 0.94},
        ],
        "runs": [{"seed": 0}, {"seed": 1}],
    }


def test_valid_runtime_vocab_replacement_passes() -> None:
    assert verify_runtime_vocab_replacement.verify(payload()) == []


def test_runtime_vocab_replacement_rejects_strong_fixed_head() -> None:
    data = payload()
    data["summary"][0]["mean_heldout_replacement_accuracy"] = 0.50
    failures = verify_runtime_vocab_replacement.verify(data)
    assert any("fixed-head heldout" in failure for failure in failures)


def test_runtime_vocab_replacement_rejects_weak_runtime_head() -> None:
    data = payload()
    data["summary"][1]["mean_heldout_replacement_accuracy"] = 0.40
    failures = verify_runtime_vocab_replacement.verify(data)
    assert any("runtime vocab heldout" in failure for failure in failures)
