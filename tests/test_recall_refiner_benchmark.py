from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_recall_refiner", ROOT / "benchmarks" / "verify_recall_refiner.py")
verify_recall_refiner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verify_recall_refiner
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_recall_refiner)


def load_result() -> dict:
    return json.loads((ROOT / "benchmarks" / "results" / "recall_refiner_results.json").read_text(encoding="utf-8"))


def test_recall_refiner_benchmark_passes() -> None:
    assert verify_recall_refiner.verify(load_result()) == []


def test_recall_refiner_benchmark_rejects_weak_gain_over_single_step() -> None:
    payload = copy.deepcopy(load_result())
    rows = {row["variant"]: row for row in payload["runs"]}
    rows["recall_refiner_half"]["final_mse"] = rows["single_recall"]["final_mse"] - 0.001

    failures = verify_recall_refiner.verify(payload)

    assert any("gain over single-step recall" in failure for failure in failures)


def test_recall_refiner_benchmark_rejects_missing_variant() -> None:
    payload = copy.deepcopy(load_result())
    payload["runs"] = [row for row in payload["runs"] if row["variant"] != "recall_refiner_half"]

    failures = verify_recall_refiner.verify(payload)

    assert any("missing variants" in failure for failure in failures)


def test_recall_refiner_benchmark_rejects_unchecked_half_drift() -> None:
    payload = copy.deepcopy(load_result())
    rows = {row["variant"]: row for row in payload["runs"]}
    rows["recall_refiner_half"]["clean_start_drift_mse"] = rows["recall_refiner_no_half"]["clean_start_drift_mse"]

    failures = verify_recall_refiner.verify(payload)

    assert any("weak correction drift" in failure for failure in failures)


def test_recall_refiner_benchmark_rejects_unfair_parameter_budget() -> None:
    payload = copy.deepcopy(load_result())
    rows = {row["variant"]: row for row in payload["runs"]}
    rows["recall_refiner_half"]["parameters"] = rows["direct_mlp"]["parameters"] * 2
    payload["fairness"]["max_trainable_parameters"] = rows["recall_refiner_half"]["parameters"]
    payload["fairness"]["max_parameter_ratio"] = 2.0

    failures = verify_recall_refiner.verify(payload)

    assert any("parameter ratio" in failure for failure in failures)


def test_recall_refiner_benchmark_rejects_unfair_training_examples() -> None:
    payload = copy.deepcopy(load_result())
    rows = {row["variant"]: row for row in payload["runs"]}
    rows["recall_refiner_half"]["train_examples"] += 1

    failures = verify_recall_refiner.verify(payload)

    assert any("train example budget" in failure for failure in failures)


def test_recall_refiner_benchmark_rejects_unpaired_refiner_seed() -> None:
    payload = copy.deepcopy(load_result())
    rows = {row["variant"]: row for row in payload["runs"]}
    rows["recall_refiner_half"]["init_seed"] += 1

    failures = verify_recall_refiner.verify(payload)

    assert any("init seed" in failure for failure in failures)


def test_recall_refiner_benchmark_rejects_unshared_training_batches() -> None:
    payload = copy.deepcopy(load_result())
    rows = {row["variant"]: row for row in payload["runs"]}
    rows["recall_refiner_half"]["train_seed"] += 1

    failures = verify_recall_refiner.verify(payload)

    assert any("train seed" in failure for failure in failures)


def test_recall_refiner_benchmark_rejects_missing_resource_profile() -> None:
    payload = copy.deepcopy(load_result())
    payload.pop("resource_profile", None)

    failures = verify_recall_refiner.verify(payload)

    assert any("resource_profile must include requested_device" in failure for failure in failures)


def test_recall_refiner_benchmark_rejects_stale_half_effect() -> None:
    payload = copy.deepcopy(load_result())
    payload["half_effect"]["final_mse_delta"] += 0.1

    failures = verify_recall_refiner.verify(payload)

    assert any("half_effect final_mse_delta" in failure for failure in failures)
