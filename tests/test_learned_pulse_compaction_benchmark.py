from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_learned_pulse_compaction", ROOT / "benchmarks" / "verify_learned_pulse_compaction.py"
)
verify_learned_pulse_compaction = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verify_learned_pulse_compaction
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_learned_pulse_compaction)


def load_result() -> dict:
    return json.loads((ROOT / "benchmarks" / "results" / "learned_pulse_compaction_results.json").read_text(encoding="utf-8"))


def test_learned_pulse_compaction_benchmark_passes() -> None:
    assert verify_learned_pulse_compaction.verify(load_result()) == []


def test_learned_pulse_compaction_rejects_weak_learned_pulse() -> None:
    payload = copy.deepcopy(load_result())
    rows = {row["variant"]: row for row in payload["variants"]}
    rows["pulse_k2"]["accuracy"] = 0.50

    failures = verify_learned_pulse_compaction.verify(payload)

    assert any("pulse_k2 accuracy" in failure for failure in failures)


def test_learned_pulse_compaction_rejects_small_gain_over_fixed_pulse() -> None:
    payload = copy.deepcopy(load_result())
    rows = {row["variant"]: row for row in payload["variants"]}
    rows["legacy_explicit_pulse"]["accuracy"] = rows["pulse_k2"]["accuracy"] - 0.01

    failures = verify_learned_pulse_compaction.verify(payload)

    assert any("legacy explicit pulse pooling" in failure for failure in failures)


def test_learned_pulse_compaction_rejects_missing_variant() -> None:
    payload = copy.deepcopy(load_result())
    payload["variants"] = [row for row in payload["variants"] if row["variant"] != "pulse_refine_k4"]

    failures = verify_learned_pulse_compaction.verify(payload)

    assert any("missing variants" in failure for failure in failures)


def test_learned_pulse_compaction_rejects_slow_optimized_pulse() -> None:
    payload = copy.deepcopy(load_result())
    rows = {row["variant"]: row for row in payload["variants"]}
    rows["pulse_refine_k4"]["samples_per_second"] = rows["pulse_refine_reference_k4"]["samples_per_second"] * 0.5

    failures = verify_learned_pulse_compaction.verify(payload)

    assert any("speed ratio" in failure for failure in failures)
