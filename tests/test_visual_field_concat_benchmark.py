from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_visual_field_concat", ROOT / "benchmarks" / "verify_visual_field_concat.py")
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


def test_committed_visual_field_concat_result_passes() -> None:
    payload = json.loads((ROOT / "benchmarks" / "results" / "visual_field_concat.json").read_text(encoding="utf-8"))
    assert module.verify(payload) == []


def test_visual_field_verifier_rejects_parameter_growth() -> None:
    payload = json.loads((ROOT / "benchmarks" / "results" / "visual_field_concat.json").read_text(encoding="utf-8"))
    concat = next(row for row in payload["summary"] if row["condition"] == "concat")
    concat["parameters"] += 1
    assert any("parameter budget" in failure for failure in module.verify(payload))
