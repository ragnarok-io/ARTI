from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_pulse_refiner_repair",
    ROOT / "benchmarks" / "verify_pulse_refiner_repair.py",
)
verify_pulse_refiner_repair = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verify_pulse_refiner_repair
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_pulse_refiner_repair)


def load_result() -> dict:
    return json.loads((ROOT / "benchmarks" / "results" / "pulse_refiner_repair_results.json").read_text(encoding="utf-8"))


def test_pulse_refiner_repair_passes() -> None:
    assert verify_pulse_refiner_repair.verify(load_result()) == []


def test_pulse_refiner_repair_rejects_unfair_parameters() -> None:
    payload = copy.deepcopy(load_result())
    rows = {row["variant"]: row for row in payload["runs"]}
    rows["pulse_refiner_half"]["parameters"] += 1

    failures = verify_pulse_refiner_repair.verify(payload)

    assert any("equal parameters" in failure for failure in failures)


def test_pulse_refiner_repair_rejects_unshared_seed() -> None:
    payload = copy.deepcopy(load_result())
    rows = {row["variant"]: row for row in payload["runs"]}
    rows["pulse_refiner_half"]["train_seed"] += 1

    failures = verify_pulse_refiner_repair.verify(payload)

    assert any("train seed" in failure for failure in failures)


def test_pulse_refiner_repair_rejects_final_mse_regression() -> None:
    payload = copy.deepcopy(load_result())
    rows = {row["variant"]: row for row in payload["runs"]}
    rows["pulse_refiner_half"]["final_mse"] = rows["fold_refiner_no_half"]["final_mse"] + 0.01
    payload["pulse_effect"]["final_mse_delta"] = 0.01

    failures = verify_pulse_refiner_repair.verify(payload)

    assert any("final repair MSE" in failure for failure in failures)


def test_pulse_refiner_repair_rejects_noise_leakage_regression() -> None:
    payload = copy.deepcopy(load_result())
    payload["pulse_effect"]["noise_leakage_ratio"] = 1.0

    failures = verify_pulse_refiner_repair.verify(payload)

    assert any("noise_leakage_ratio does not match" in failure or "reduce leakage" in failure for failure in failures)


def test_pulse_refiner_repair_rejects_weak_gain_regression() -> None:
    payload = copy.deepcopy(load_result())
    rows = {row["variant"]: row for row in payload["runs"]}
    rows["pulse_refiner_half"]["weak_noise_gain"] = rows["fold_refiner_no_half"]["weak_noise_gain"]
    payload["pulse_effect"]["weak_noise_gain_ratio"] = 1.0

    failures = verify_pulse_refiner_repair.verify(payload)

    assert any("weak-noise gain" in failure for failure in failures)


def test_pulse_refiner_repair_rejects_signal_only_regression() -> None:
    payload = copy.deepcopy(load_result())
    rows = {row["variant"]: row for row in payload["runs"]}
    rows["pulse_refiner_half"]["signal_only_mse"] = rows["fold_refiner_no_half"]["signal_only_mse"] + 0.01
    payload["pulse_effect"]["signal_only_mse_delta"] = 0.01

    failures = verify_pulse_refiner_repair.verify(payload)

    assert any("strong signal-only repair" in failure for failure in failures)


def test_pulse_refiner_repair_rejects_missing_resource_record() -> None:
    payload = copy.deepcopy(load_result())
    payload["resource_profile"]["cuda_runtime_used"] = True
    payload["resource_profile"]["records_peak_cuda_memory"] = True
    payload["runs"][0]["peak_cuda_memory_bytes"] = None

    failures = verify_pulse_refiner_repair.verify(payload)

    assert any("CUDA peak memory" in failure for failure in failures)


def test_pulse_refiner_repair_allows_cpu_run_on_cuda_host_without_peak_memory() -> None:
    payload = copy.deepcopy(load_result())
    payload["provenance"]["cuda_available"] = True
    payload["config"]["device"] = "cpu"
    payload["resource_profile"]["requested_device"] = "cpu"
    payload["resource_profile"]["device_type"] = "cpu"
    payload["resource_profile"]["cuda_runtime_used"] = False
    payload["resource_profile"]["records_peak_cuda_memory"] = False
    for row in payload["runs"]:
        row["requested_device"] = "cpu"
        row["device_type"] = "cpu"
        row["peak_cuda_memory_bytes"] = None
    payload["provenance"]["command"] = payload["provenance"]["command"].replace("--device cuda", "--device cpu")

    failures = verify_pulse_refiner_repair.verify(payload)

    assert not any("CUDA peak memory" in failure for failure in failures)


def test_pulse_refiner_repair_rejects_missing_resource_profile() -> None:
    payload = copy.deepcopy(load_result())
    payload.pop("resource_profile", None)

    failures = verify_pulse_refiner_repair.verify(payload)

    assert any("resource_profile must include requested_device" in failure for failure in failures)


def test_pulse_refiner_repair_rejects_resource_profile_budget_mismatch() -> None:
    payload = copy.deepcopy(load_result())
    payload["resource_profile"]["effective_batch_size"] += 1

    failures = verify_pulse_refiner_repair.verify(payload)

    assert any("effective_batch_size" in failure for failure in failures)


def test_pulse_refiner_repair_rejects_row_eval_budget_mismatch() -> None:
    payload = copy.deepcopy(load_result())
    payload["runs"][0]["eval_examples"] += 1

    failures = verify_pulse_refiner_repair.verify(payload)

    assert any("eval example budget" in failure for failure in failures)


def test_pulse_refiner_repair_rejects_stale_summary() -> None:
    payload = copy.deepcopy(load_result())
    payload["summary"][0]["final_mse"] += 0.1

    failures = verify_pulse_refiner_repair.verify(payload)

    assert any("summary" in failure and "does not match runs" in failure for failure in failures)


def test_pulse_refiner_repair_rejects_missing_loss_weights() -> None:
    payload = copy.deepcopy(load_result())
    payload["fairness"]["loss_weights"]["consistency"] = 0.0

    failures = verify_pulse_refiner_repair.verify(payload)

    assert any("loss weight consistency" in failure for failure in failures)


def test_pulse_refiner_repair_rejects_inconsistent_loss_components() -> None:
    payload = copy.deepcopy(load_result())
    payload["runs"][0]["final_repair_loss"] += 1.0

    failures = verify_pulse_refiner_repair.verify(payload)

    assert any("weighted loss components" in failure for failure in failures)


def test_pulse_refiner_repair_rejects_wrong_training_objective() -> None:
    payload = copy.deepcopy(load_result())
    payload["training_objective_contract"]["target_source"] = "corrupted_workspace"
    payload["training_objective_contract"]["uses_evaluation_metrics_as_training_loss"] = True

    failures = verify_pulse_refiner_repair.verify(payload)

    assert any("target_source" in failure for failure in failures)
    assert any("uses_evaluation_metrics_as_training_loss" in failure for failure in failures)


def test_pulse_refiner_repair_rejects_bad_throughput_and_resource_aggregates() -> None:
    payload = copy.deepcopy(load_result())
    payload["runs"][0]["samples_per_second"] += 10.0
    payload["resource_profile"]["min_samples_per_second"] += 1.0
    payload["resource_profile"]["max_samples_per_second"] += 1.0
    payload["resource_profile"]["max_peak_cuda_memory_bytes"] += 1

    failures = verify_pulse_refiner_repair.verify(payload)

    assert any("samples_per_second must equal" in failure for failure in failures)
    assert any("min_samples_per_second must match runs" in failure for failure in failures)
    assert any("max_samples_per_second must match runs" in failure for failure in failures)
    assert any("max_peak_cuda_memory_bytes must match runs" in failure for failure in failures)


def test_pulse_refiner_repair_rejects_incomplete_reproducer_command() -> None:
    payload = copy.deepcopy(load_result())
    payload["provenance"]["command"] = "benchmarks/run_pulse_refiner_repair.py"

    failures = verify_pulse_refiner_repair.verify(payload)

    assert any("provenance command" in failure and "--train-steps" in failure for failure in failures)


def test_pulse_refiner_repair_rejects_missing_input_pressure() -> None:
    payload = copy.deepcopy(load_result())
    payload.pop("input_pressure", None)

    failures = verify_pulse_refiner_repair.verify(payload)

    assert any("input_pressure must include overcomplete_ratio" in failure for failure in failures)


def test_pulse_refiner_repair_rejects_trivial_input_pressure() -> None:
    payload = copy.deepcopy(load_result())
    payload["input_pressure"]["observed_signal_keep_fraction"] = 1.0
    payload["input_pressure"]["observed_weak_trace_norm"] = 0.0

    failures = verify_pulse_refiner_repair.verify(payload)

    assert any("observed signal keep fraction" in failure for failure in failures)
    assert any("weak trace norm" in failure for failure in failures)


def test_pulse_refiner_repair_rejects_input_pressure_config_mismatch() -> None:
    payload = copy.deepcopy(load_result())
    payload["input_pressure"]["noise_std"] = payload["config"]["noise_std"] + 0.1

    failures = verify_pulse_refiner_repair.verify(payload)

    assert any("noise_std must match config" in failure for failure in failures)


def test_pulse_refiner_repair_rejects_config_fairness_mismatch() -> None:
    payload = copy.deepcopy(load_result())
    payload["fairness"]["train_examples_per_variant"] += 1

    failures = verify_pulse_refiner_repair.verify(payload)

    assert any("train_examples_per_variant" in failure and "config" in failure for failure in failures)


def test_pulse_refiner_repair_rejects_stale_depth_effect() -> None:
    payload = copy.deepcopy(load_result())
    payload["depth_effect"]["per_step_mse_delta"][0] += 0.1

    failures = verify_pulse_refiner_repair.verify(payload)

    assert any("per_step_mse_delta does not match" in failure for failure in failures)


def test_pulse_refiner_repair_rejects_unstable_depth_curve() -> None:
    payload = copy.deepcopy(load_result())
    rows = {row["variant"]: row for row in payload["runs"]}
    pulse_steps = rows["pulse_refiner_half"]["per_step_mse"]
    pulse_steps[1] = pulse_steps[0] + 0.01
    payload["depth_effect"]["pulse_per_step_mse"] = list(pulse_steps)
    payload["depth_effect"]["pulse_nonincreasing"] = False

    failures = verify_pulse_refiner_repair.verify(payload)

    assert any("depth curve" in failure or "non-increasing" in failure for failure in failures)


def test_pulse_refiner_repair_rejects_stale_update_norm_effect() -> None:
    payload = copy.deepcopy(load_result())
    payload["depth_effect"]["per_step_update_norm_ratio"][0] += 0.1

    failures = verify_pulse_refiner_repair.verify(payload)

    assert any("per_step_update_norm_ratio does not match" in failure for failure in failures)


def test_pulse_refiner_repair_rejects_stale_depth_summary_ratio() -> None:
    payload = copy.deepcopy(load_result())
    payload["depth_effect"]["pulse_final_to_first_ratio"] += 0.1
    payload["depth_effect"]["pulse_early_step_ratio_max"] += 0.1

    failures = verify_pulse_refiner_repair.verify(payload)

    assert any("pulse_final_to_first_ratio does not match" in failure for failure in failures)
    assert any("pulse_early_step_ratio_max does not match" in failure for failure in failures)


def test_pulse_refiner_repair_rejects_stale_depth_boolean() -> None:
    payload = copy.deepcopy(load_result())
    payload["depth_effect"]["fold_update_norm_nonincreasing"] = not payload["depth_effect"]["fold_update_norm_nonincreasing"]

    failures = verify_pulse_refiner_repair.verify(payload)

    assert any("fold_update_norm_nonincreasing does not match" in failure for failure in failures)


def test_pulse_refiner_repair_rejects_unstable_update_norm_curve() -> None:
    payload = copy.deepcopy(load_result())
    rows = {row["variant"]: row for row in payload["runs"]}
    update_norms = rows["pulse_refiner_half"]["per_step_update_norm"]
    update_norms[1] = update_norms[0] + 1.0
    rows["pulse_refiner_half"]["max_update_norm"] = max(update_norms)
    payload["depth_effect"]["pulse_per_step_update_norm"] = list(update_norms)
    payload["depth_effect"]["pulse_update_norm_nonincreasing"] = False

    failures = verify_pulse_refiner_repair.verify(payload)

    assert any("update norms" in failure or "update norm" in failure for failure in failures)
