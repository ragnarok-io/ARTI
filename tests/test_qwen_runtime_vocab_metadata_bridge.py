from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_qwen_runtime_vocab_metadata_bridge",
    ROOT / "benchmarks" / "verify_qwen_runtime_vocab_metadata_bridge.py",
)
verify_qwen_runtime_vocab_metadata_bridge = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verify_qwen_runtime_vocab_metadata_bridge
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_qwen_runtime_vocab_metadata_bridge)


def load_result() -> dict:
    return json.loads(
        (ROOT / "benchmarks" / "results" / "qwen_runtime_vocab_pulse_metadata_bridge_probe.json").read_text(encoding="utf-8")
    )


def test_qwen_runtime_vocab_metadata_bridge_passes() -> None:
    assert verify_qwen_runtime_vocab_metadata_bridge.verify(load_result()) == []


def test_qwen_runtime_vocab_metadata_bridge_rejects_zero_glyph_channel() -> None:
    payload = copy.deepcopy(load_result())
    payload["dataset"]["vocab_glyph_scale"] = 0.0

    failures = verify_qwen_runtime_vocab_metadata_bridge.verify(payload)

    assert any("nonzero" in failure for failure in failures)


def test_qwen_runtime_vocab_metadata_bridge_rejects_missing_bridge() -> None:
    payload = copy.deepcopy(load_result())
    payload["dataset"]["semantic_bridge"] = False

    failures = verify_qwen_runtime_vocab_metadata_bridge.verify(payload)

    assert any("semantic_bridge" in failure for failure in failures)


def test_qwen_runtime_vocab_metadata_bridge_rejects_short_training_window() -> None:
    payload = copy.deepcopy(load_result())
    rows = {row["model"]: row for row in payload["runs"]}
    rows["runtime_vocab_pulse"]["steps"] = 20

    failures = verify_qwen_runtime_vocab_metadata_bridge.verify(payload)

    assert any("500 steps" in failure for failure in failures)
