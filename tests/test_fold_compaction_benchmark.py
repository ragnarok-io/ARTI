from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_fold_compaction", ROOT / "benchmarks" / "verify_fold_compaction.py")
verify_fold_compaction = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verify_fold_compaction
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_fold_compaction)


def load_result() -> dict:
    return json.loads((ROOT / "benchmarks" / "results" / "fold_compaction_results.json").read_text(encoding="utf-8"))


def test_fold_compaction_benchmark_passes() -> None:
    assert verify_fold_compaction.verify(load_result()) == []


def test_fold_compaction_benchmark_rejects_weak_fold_q() -> None:
    payload = copy.deepcopy(load_result())
    rows = {row["variant"]: row for row in payload["variants"]}
    rows["fold_q"]["accuracy"] = 0.50

    failures = verify_fold_compaction.verify(payload)

    assert any("fold_q accuracy" in failure for failure in failures)


def test_fold_compaction_benchmark_rejects_small_gain_over_q_mean() -> None:
    payload = copy.deepcopy(load_result())
    rows = {row["variant"]: row for row in payload["variants"]}
    rows["q_mean_pool"]["accuracy"] = rows["fold_q"]["accuracy"] - 0.01

    failures = verify_fold_compaction.verify(payload)

    assert any("q-guided mean pooling" in failure for failure in failures)


def test_fold_compaction_benchmark_rejects_missing_variant() -> None:
    payload = copy.deepcopy(load_result())
    payload["variants"] = [row for row in payload["variants"] if row["variant"] != "half_fold_q"]

    failures = verify_fold_compaction.verify(payload)

    assert any("missing variants" in failure for failure in failures)
