from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_qwen_learned_pulse_adapter", ROOT / "benchmarks" / "verify_qwen_learned_pulse_adapter.py"
)
verify_qwen_learned_pulse_adapter = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verify_qwen_learned_pulse_adapter
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_qwen_learned_pulse_adapter)


def load_result() -> dict:
    return json.loads((ROOT / "benchmarks" / "results" / "qwen_learned_pulse_adapter_results.json").read_text(encoding="utf-8"))


def test_qwen_learned_pulse_adapter_passes() -> None:
    assert verify_qwen_learned_pulse_adapter.verify(load_result()) == []


def test_qwen_learned_pulse_adapter_rejects_weak_refined_pulse() -> None:
    payload = copy.deepcopy(load_result())
    rows = {row["model"]: row for row in payload["runs"]}
    rows["pulse_refine"]["heldout_tokenization_accuracy"] = 0.50

    failures = verify_qwen_learned_pulse_adapter.verify(payload)

    assert any("pulse_refine heldout" in failure for failure in failures)


def test_qwen_learned_pulse_adapter_rejects_missing_reference() -> None:
    payload = copy.deepcopy(load_result())
    payload["runs"] = [row for row in payload["runs"] if row["model"] != "pulse_refine_reference"]

    failures = verify_qwen_learned_pulse_adapter.verify(payload)

    assert any("missing models" in failure for failure in failures)


def test_qwen_learned_pulse_adapter_rejects_slow_optimization() -> None:
    payload = copy.deepcopy(load_result())
    rows = {row["model"]: row for row in payload["runs"]}
    rows["pulse_refine"]["samples_per_second"] = rows["pulse_refine_reference"]["samples_per_second"] * 0.5

    failures = verify_qwen_learned_pulse_adapter.verify(payload)

    assert any("speedup" in failure for failure in failures)
