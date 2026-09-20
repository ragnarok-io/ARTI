from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_qwen_hidden_refiner_half",
    ROOT / "benchmarks" / "verify_qwen_hidden_refiner_half.py",
)
verify_qwen_hidden_refiner_half = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verify_qwen_hidden_refiner_half
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_qwen_hidden_refiner_half)


def load_result() -> dict:
    return json.loads((ROOT / "benchmarks" / "results" / "qwen_hidden_refiner_half_results.json").read_text(encoding="utf-8"))


def test_qwen_hidden_refiner_half_passes() -> None:
    assert verify_qwen_hidden_refiner_half.verify(load_result()) == []


def test_qwen_hidden_refiner_half_rejects_unfair_parameters() -> None:
    payload = copy.deepcopy(load_result())
    rows = {row["variant"]: row for row in payload["runs"]}
    rows["refiner_half"]["parameters"] += 1

    failures = verify_qwen_hidden_refiner_half.verify(payload)

    assert any("equal parameters" in failure or "parameter count" in failure for failure in failures)


def test_qwen_hidden_refiner_half_rejects_unshared_seed() -> None:
    payload = copy.deepcopy(load_result())
    rows = {row["variant"]: row for row in payload["runs"]}
    rows["refiner_half"]["init_seed"] += 1

    failures = verify_qwen_hidden_refiner_half.verify(payload)

    assert any("initialization seed" in failure for failure in failures)


def test_qwen_hidden_refiner_half_rejects_weak_noise_regression() -> None:
    payload = copy.deepcopy(load_result())
    payload["half_effect"]["weak_noise_gain_ratio"] = 1.0

    failures = verify_qwen_hidden_refiner_half.verify(payload)

    assert any("weak_noise_gain_ratio does not match" in failure or "weak-noise gain" in failure for failure in failures)


def test_qwen_hidden_refiner_half_rejects_clean_drift_regression() -> None:
    payload = copy.deepcopy(load_result())
    rows = {row["variant"]: row for row in payload["runs"]}
    rows["refiner_half"]["clean_start_drift_mse"] = rows["refiner_no_half"]["clean_start_drift_mse"]
    payload["half_effect"]["clean_start_drift_ratio"] = 1.0

    failures = verify_qwen_hidden_refiner_half.verify(payload)

    assert any("clean-start drift" in failure for failure in failures)


def test_qwen_hidden_refiner_half_rejects_final_mse_regression() -> None:
    payload = copy.deepcopy(load_result())
    rows = {row["variant"]: row for row in payload["runs"]}
    rows["refiner_half"]["final_mse"] = rows["refiner_no_half"]["final_mse"] + 0.01
    payload["half_effect"]["final_mse_delta"] = 0.01

    failures = verify_qwen_hidden_refiner_half.verify(payload)

    assert any("final hidden repair MSE" in failure for failure in failures)


def test_qwen_hidden_refiner_half_rejects_wrong_training_objective() -> None:
    payload = copy.deepcopy(load_result())
    payload["task"]["training_objective"]["uses_generated_string_loss"] = True

    failures = verify_qwen_hidden_refiner_half.verify(payload)

    assert any("generated strings" in failure for failure in failures)


def test_qwen_hidden_refiner_half_rejects_bad_final_loss_formula() -> None:
    payload = copy.deepcopy(load_result())
    rows = {row["variant"]: row for row in payload["runs"]}
    rows["refiner_half"]["final_loss"] += 0.25

    failures = verify_qwen_hidden_refiner_half.verify(payload)

    assert any("final_loss does not match" in failure for failure in failures)


def test_qwen_hidden_refiner_half_rejects_bad_throughput_formula() -> None:
    payload = copy.deepcopy(load_result())
    rows = {row["variant"]: row for row in payload["runs"]}
    rows["refiner_half"]["samples_per_second"] *= 0.5

    failures = verify_qwen_hidden_refiner_half.verify(payload)

    assert any("samples_per_second does not match" in failure for failure in failures)
