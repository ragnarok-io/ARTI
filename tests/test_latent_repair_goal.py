from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_latent_repair_goal", ROOT / "benchmarks" / "verify_latent_repair_goal.py")
verify_latent_repair_goal = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verify_latent_repair_goal
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_latent_repair_goal)


def load_result(name: str) -> dict:
    return json.loads((ROOT / "benchmarks" / "results" / name).read_text(encoding="utf-8"))


def payloads() -> dict[str, dict]:
    return {
        "half_trace": load_result("half_recall_trace_survival_results.json"),
        "half_training": load_result("half_recall_fair_training_results.json"),
        "recall_refiner": load_result("recall_refiner_results.json"),
        "pulse_repair": load_result("pulse_refiner_repair_results.json"),
        "pulse_multiseed": load_result("pulse_refiner_multiseed_results.json"),
    }


def test_latent_repair_goal_passes() -> None:
    assert verify_latent_repair_goal.verify(**payloads()) == []


def test_latent_repair_goal_rejects_direct_refiner_regression() -> None:
    data = payloads()
    data["recall_refiner"] = copy.deepcopy(data["recall_refiner"])
    rows = {row["variant"]: row for row in data["recall_refiner"]["runs"]}
    rows["recall_refiner_half"]["final_mse"] = rows["recall_refiner_no_half"]["final_mse"] + 0.01
    data["recall_refiner"]["half_effect"]["final_mse_delta"] = 0.01

    failures = verify_latent_repair_goal.verify(**data)

    assert any("recall refiner" in failure and "final MSE" in failure for failure in failures)


def test_latent_repair_goal_rejects_half_training_regression() -> None:
    data = payloads()
    data["half_training"] = copy.deepcopy(data["half_training"])
    rows = {row["variant"]: row for row in data["half_training"]["summary"]}
    rows["half"]["weak_noise_gain"] = rows["identity"]["weak_noise_gain"]

    failures = verify_latent_repair_goal.verify(**data)

    assert any("weak-trace gain" in failure or "weak gain" in failure for failure in failures)


def test_latent_repair_goal_rejects_compact_weak_noise_regression() -> None:
    data = payloads()
    data["pulse_repair"] = copy.deepcopy(data["pulse_repair"])
    rows = {row["variant"]: row for row in data["pulse_repair"]["runs"]}
    rows["pulse_refiner_half"]["weak_noise_gain"] = rows["fold_refiner_no_half"]["weak_noise_gain"]
    data["pulse_repair"]["pulse_effect"]["weak_noise_gain_ratio"] = 1.0

    failures = verify_latent_repair_goal.verify(**data)

    assert any("pulse repair" in failure and "weak-noise gain" in failure for failure in failures)


def test_latent_repair_goal_rejects_missing_overcomplete_pressure() -> None:
    data = payloads()
    data["pulse_repair"] = copy.deepcopy(data["pulse_repair"])
    data["pulse_repair"]["input_pressure"]["overcomplete_ratio"] = 1.0

    failures = verify_latent_repair_goal.verify(**data)

    assert any("pulse repair" in failure and "overcomplete" in failure for failure in failures)


def test_latent_repair_goal_rejects_multiseed_selectivity_regression() -> None:
    data = payloads()
    data["pulse_multiseed"] = copy.deepcopy(data["pulse_multiseed"])
    data["pulse_multiseed"]["summary"][0]["pulse_lower_leakage_count"] = 2

    failures = verify_latent_repair_goal.verify(**data)

    assert any("pulse multi-seed" in failure and "leakage" in failure for failure in failures)
