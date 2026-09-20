from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_pulse_refiner_multiseed",
    ROOT / "benchmarks" / "verify_pulse_refiner_multiseed.py",
)
verifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verifier
assert SPEC.loader is not None
SPEC.loader.exec_module(verifier)


def load_result() -> dict:
    return json.loads((ROOT / "benchmarks" / "results" / "pulse_refiner_multiseed_results.json").read_text(encoding="utf-8"))


def test_pulse_refiner_multiseed_passes() -> None:
    assert verifier.verify(load_result()) == []


def test_pulse_refiner_multiseed_rejects_selectivity_regression() -> None:
    payload = copy.deepcopy(load_result())
    payload["summary"][0]["pulse_better_signal_only_count"] = 2
    payload["summary"][0]["pulse_lower_leakage_count"] = 2

    failures = verifier.verify(payload)

    assert any("signal-only" in failure for failure in failures)
    assert any("leakage" in failure for failure in failures)


def test_pulse_refiner_multiseed_rejects_hidden_final_mse_claim() -> None:
    payload = copy.deepcopy(load_result())
    payload["summary"][0]["final_mse_dominance"] = True

    failures = verifier.verify(payload)

    assert any("final-MSE dominance" in failure for failure in failures)


def test_pulse_refiner_multiseed_rejects_resource_drift() -> None:
    payload = copy.deepcopy(load_result())
    payload["resource_profile"]["max_peak_cuda_memory_bytes"] += 1

    failures = verifier.verify(payload)

    assert any("max_peak_cuda_memory_bytes" in failure for failure in failures)
