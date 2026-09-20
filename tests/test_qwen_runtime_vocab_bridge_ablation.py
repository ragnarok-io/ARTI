from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_qwen_runtime_vocab_bridge_ablation",
    ROOT / "benchmarks" / "verify_qwen_runtime_vocab_bridge_ablation.py",
)
verify_qwen_runtime_vocab_bridge_ablation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verify_qwen_runtime_vocab_bridge_ablation
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_qwen_runtime_vocab_bridge_ablation)


def load_result(name: str) -> dict:
    return json.loads((ROOT / "benchmarks" / "results" / name).read_text(encoding="utf-8"))


def payloads() -> dict[str, dict]:
    return {
        "no_bridge": load_result("qwen_runtime_vocab_pulse_strict_surface_probe.json"),
        "metadata_bridge": load_result("qwen_runtime_vocab_pulse_metadata_bridge_probe.json"),
    }


def test_qwen_runtime_vocab_bridge_ablation_passes() -> None:
    assert verify_qwen_runtime_vocab_bridge_ablation.verify(**payloads()) == []


def test_qwen_runtime_vocab_bridge_ablation_rejects_high_no_bridge() -> None:
    data = payloads()
    data["no_bridge"] = copy.deepcopy(data["no_bridge"])
    rows = {row["model"]: row for row in data["no_bridge"]["runs"]}
    rows["runtime_vocab_pulse"]["heldout_surface_accuracy"] = 0.9

    failures = verify_qwen_runtime_vocab_bridge_ablation.verify(**data)

    assert any("no-bridge" in failure and "high" in failure for failure in failures)


def test_qwen_runtime_vocab_bridge_ablation_rejects_low_bridge() -> None:
    data = payloads()
    data["metadata_bridge"] = copy.deepcopy(data["metadata_bridge"])
    rows = {row["model"]: row for row in data["metadata_bridge"]["runs"]}
    rows["runtime_vocab_pulse"]["heldout_surface_accuracy"] = 0.4

    failures = verify_qwen_runtime_vocab_bridge_ablation.verify(**data)

    assert any("metadata bridge strict" in failure for failure in failures)
