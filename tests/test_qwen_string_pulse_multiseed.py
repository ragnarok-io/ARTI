from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_qwen_string_pulse_multiseed",
    ROOT / "benchmarks" / "verify_qwen_string_pulse_multiseed.py",
)
verifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verifier
assert SPEC.loader is not None
SPEC.loader.exec_module(verifier)


def load_result() -> dict:
    return json.loads((ROOT / "benchmarks" / "results" / "qwen_string_pulse_multiseed_results.json").read_text(encoding="utf-8"))


def test_qwen_string_pulse_multiseed_passes() -> None:
    assert verifier.verify(load_result()) == []


def test_qwen_string_pulse_multiseed_rejects_hidden_collapse() -> None:
    payload = copy.deepcopy(load_result())
    payload["summary"][0]["pulse_mode_collapse_count"] = 1

    failures = verifier.verify(payload)

    assert any("pulse_mode_collapse_count" in failure for failure in failures)


def test_qwen_string_pulse_multiseed_rejects_stale_exact_flag() -> None:
    payload = copy.deepcopy(load_result())
    payload["runs"][0]["answer_samples"][0]["generated"][1]["exact"] = False

    failures = verifier.verify(payload)

    assert any("exact flag" in failure for failure in failures)


def test_qwen_string_pulse_multiseed_rejects_resource_drift() -> None:
    payload = copy.deepcopy(load_result())
    payload["resource_profile"]["max_peak_cuda_memory_bytes"] += 1

    failures = verifier.verify(payload)

    assert any("max_peak_cuda_memory_bytes" in failure for failure in failures)


def test_qwen_string_pulse_multiseed_rejects_decoupled_identity_objective() -> None:
    payload = copy.deepcopy(load_result())
    payload["training_objective_contract"]["answer_identity_target"] = "diagnostic_condition_head"
    payload["fairness"]["prompt_balanced_sampling"] = False

    failures = verifier.verify(payload)

    assert any("directly bind answer identity" in failure for failure in failures)
    assert any("prompt-balanced sampling" in failure for failure in failures)
