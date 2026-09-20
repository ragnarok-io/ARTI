from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_qwen_runtime_vocab_pulse_semantic",
    ROOT / "benchmarks" / "verify_qwen_runtime_vocab_pulse_semantic.py",
)
verify_qwen_runtime_vocab_pulse_semantic = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verify_qwen_runtime_vocab_pulse_semantic
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_qwen_runtime_vocab_pulse_semantic)


def load_result() -> dict:
    return json.loads((ROOT / "benchmarks" / "results" / "qwen_runtime_vocab_pulse_semantic_results.json").read_text(encoding="utf-8"))


def test_qwen_runtime_vocab_pulse_semantic_passes() -> None:
    assert verify_qwen_runtime_vocab_pulse_semantic.verify(load_result()) == []


def test_qwen_runtime_vocab_pulse_semantic_rejects_tiny_effective_batch() -> None:
    payload = copy.deepcopy(load_result())
    payload["resource_efficiency"]["effective_batch_size"] = 32

    failures = verify_qwen_runtime_vocab_pulse_semantic.verify(payload)

    assert any("effective batch size" in failure for failure in failures)


def test_qwen_runtime_vocab_pulse_semantic_rejects_tiny_runtime_view() -> None:
    payload = copy.deepcopy(load_result())
    payload["config"]["view_size"] = 2

    failures = verify_qwen_runtime_vocab_pulse_semantic.verify(payload)

    assert any("view_size" in failure for failure in failures)


def test_qwen_runtime_vocab_pulse_semantic_rejects_missing_heldout_tokenization_split() -> None:
    payload = copy.deepcopy(load_result())
    payload["dataset"]["heldout_tokenization_variants"] = []

    failures = verify_qwen_runtime_vocab_pulse_semantic.verify(payload)

    assert any("held-out evaluation" in failure for failure in failures)


def test_qwen_runtime_vocab_pulse_semantic_rejects_tiny_vocab_bank() -> None:
    payload = copy.deepcopy(load_result())
    payload["resource_efficiency"]["precompute"]["raw_vocab_items"] = 8

    failures = verify_qwen_runtime_vocab_pulse_semantic.verify(payload)

    assert any("raw runtime vocab items" in failure for failure in failures)


def test_qwen_runtime_vocab_pulse_semantic_rejects_missing_grad_accumulation() -> None:
    payload = copy.deepcopy(load_result())
    payload["resource_efficiency"]["grad_accum_steps"] = 1
    rows = {row["model"]: row for row in payload["runs"]}
    rows["runtime_vocab_pulse"]["grad_accum_steps"] = 1

    failures = verify_qwen_runtime_vocab_pulse_semantic.verify(payload)

    assert any("grad_accum_steps" in failure for failure in failures)


def test_qwen_runtime_vocab_pulse_semantic_rejects_short_training_window() -> None:
    payload = copy.deepcopy(load_result())
    rows = {row["model"]: row for row in payload["runs"]}
    rows["runtime_vocab_pulse"]["steps"] = 20

    failures = verify_qwen_runtime_vocab_pulse_semantic.verify(payload)

    assert any("at least 200 training steps" in failure for failure in failures)
