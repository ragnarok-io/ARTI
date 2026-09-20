from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_qwen_string_recall_refiner_answering",
    ROOT / "benchmarks" / "verify_qwen_string_recall_refiner_answering.py",
)
verify_qwen_string_recall_refiner_answering = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verify_qwen_string_recall_refiner_answering
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_qwen_string_recall_refiner_answering)


def load_result() -> dict:
    return json.loads((ROOT / "benchmarks" / "results" / "qwen_string_recall_refiner_answering_results.json").read_text(encoding="utf-8"))


def test_qwen_string_recall_refiner_answering_passes() -> None:
    assert verify_qwen_string_recall_refiner_answering.verify(load_result()) == []


def test_qwen_string_recall_refiner_answering_rejects_token_input_boundary() -> None:
    payload = copy.deepcopy(load_result())
    payload["claim_boundary"] = "ARTI reads token ids and reports hidden loss."

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("rendered strings" in failure for failure in failures)
    assert any("conditional-NLL measurement" in failure for failure in failures)
    assert any("generated strings" in failure or "generated answer string" in failure for failure in failures)
    assert any("coherence metrics" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_pulse_regression() -> None:
    payload = copy.deepcopy(load_result())
    rows = {row["model"]: row for row in payload["runs"]}
    rows["pulse"]["open_loop_exact_rate"] = rows["mean_string"]["open_loop_exact_rate"]

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("Pulse should beat" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_pulse_matched_baseline_regression() -> None:
    payload = copy.deepcopy(load_result())
    rows = {row["model"]: row for row in payload["runs"]}
    rows["pulse"]["open_loop_exact_rate"] = rows["mean_wide"]["open_loop_exact_rate"]

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("parameter-matched mean_wide" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_pulse_matched_quality_regression() -> None:
    payload = copy.deepcopy(load_result())
    rows = {row["model"]: row for row in payload["runs"]}
    rows["pulse"]["open_loop_conditional_nll_mean"] = rows["mean_wide"]["open_loop_conditional_nll_mean"]
    rows["pulse"]["open_loop_coherence_mean"] = rows["mean_wide"]["open_loop_coherence_mean"] - 0.01
    rows["pulse"]["open_loop_loop_penalty_mean"] = rows["mean_wide"]["open_loop_loop_penalty_mean"] + 0.01

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("conditional NLL" in failure for failure in failures)
    assert any("coherence" in failure for failure in failures)
    assert any("loop penalty" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_missing_half_ablation() -> None:
    payload = copy.deepcopy(load_result())
    payload.pop("half_ablation", None)

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("half_ablation must include outcome" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_unpaired_half_ablation() -> None:
    payload = copy.deepcopy(load_result())
    payload["half_ablation"]["shared_init_seed"] = False
    payload["half_ablation"]["shared_train_seed"] = False

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("initialization seed" in failure for failure in failures)
    assert any("train seed" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_unmatched_half_parameters() -> None:
    payload = copy.deepcopy(load_result())
    payload["half_ablation"]["half_parameters"] += 1
    payload["half_ablation"]["parameter_ratio"] = 1.1
    payload["half_ablation"]["equal_trainable_parameters"] = False

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("equal trainable parameters" in failure for failure in failures)
    assert any("parameter ratio" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_inconsistent_half_delta() -> None:
    payload = copy.deepcopy(load_result())
    payload["half_ablation"]["open_loop_exact_rate_delta"] += 0.5

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("exact-rate delta" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_missing_half_context() -> None:
    payload = copy.deepcopy(load_result())
    payload.pop("half_mechanism_context", None)

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("half_mechanism_context must include string_probe_role" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_string_only_half_verdict() -> None:
    payload = copy.deepcopy(load_result())
    payload["half_mechanism_context"]["verdict_rule"] = "Use this result as the Half verdict."
    payload["half_mechanism_context"]["positive_half_evidence"] = [
        item
        for item in payload["half_mechanism_context"]["positive_half_evidence"]
        if item["gate"] not in {"qwen_hidden_refiner_half", "half_recall_fair_training"}
    ]

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("verdict rule must forbid string-only Half verdicts" in failure for failure in failures)
    assert any("missing controlled Half evidence gates" in failure for failure in failures)
    assert any("half_recall_fair_training" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_missing_text_tensor_contract() -> None:
    payload = copy.deepcopy(load_result())
    payload.pop("text_tensor_contract", None)

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("text_tensor_contract must include identity_mode" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_codepoint_visible_identity() -> None:
    payload = copy.deepcopy(load_result())
    payload["text_tensor_contract"]["identity_mode"] = "glyph_plus_codepoint_aux"
    payload["text_tensor_contract"]["known_visible_codepoint_aux_max_abs"] = 0.5

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("identity_mode must be control_codepoint_aux" in failure for failure in failures)
    assert any("known visible characters must have zero codepoint aux" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_text_tensor_dim_mismatch() -> None:
    payload = copy.deepcopy(load_result())
    payload["text_tensor_contract"]["text_tensor_dim"] += 1

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("text_tensor_dim" in failure and "must" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_missing_input_boundary_contract() -> None:
    payload = copy.deepcopy(load_result())
    payload.pop("input_boundary_contract", None)

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("input_boundary_contract must include arti_training_inputs" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_token_ids_as_arti_input() -> None:
    payload = copy.deepcopy(load_result())
    payload["input_boundary_contract"]["token_ids_are_arti_input"] = True
    payload["input_boundary_contract"]["arti_training_inputs"] = ["token_ids", "mask"]

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("rendered text tensor plus mask" in failure for failure in failures)
    assert any("token ids are not ARTI input" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_unfrozen_qwen_boundary() -> None:
    payload = copy.deepcopy(load_result())
    payload["qwen"]["qwen_trainable_parameters"] = 1
    payload["qwen"]["lm_head_trainable_parameters"] = 1

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("Qwen must remain frozen" in failure for failure in failures)
    assert any("lm_head must remain frozen" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_missing_training_objective_contract() -> None:
    payload = copy.deepcopy(load_result())
    payload.pop("training_objective_contract", None)

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("training_objective_contract must include supervision_unit" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_hidden_reconstruction_objective() -> None:
    payload = copy.deepcopy(load_result())
    payload["training_objective_contract"]["uses_hidden_reconstruction_loss"] = True
    payload["training_objective_contract"]["uses_open_loop_strings_as_training_loss"] = True
    payload["training_objective_contract"]["open_loop_strings_are_evaluation_only"] = False

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("must not use hidden reconstruction loss" in failure for failure in failures)
    assert any("must not use open-loop generated strings" in failure for failure in failures)
    assert any("open-loop strings as evaluation only" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_training_target_count_mismatch() -> None:
    payload = copy.deepcopy(load_result())
    payload["training_objective_contract"]["target_count"] += 1
    payload["training_objective_contract"]["prompt_condition_count"] += 1

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("target_count must match prefix_examples" in failure for failure in failures)
    assert any("prompt_condition_count must match prefix_examples" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_teacher_generated_targets() -> None:
    payload = copy.deepcopy(load_result())
    payload["training_objective_contract"]["target_source"] = "Qwen teacher generated next-token ids"
    payload["resource_profile"]["qwen_body_used_for_teacher_generation"] = True

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("canonical answer strings" in failure for failure in failures)
    assert any("teacher generation is not used" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_wrong_canonical_answer() -> None:
    payload = copy.deepcopy(load_result())
    payload["data"]["target_answers"][2] = "There are 2 letter r in the word strawberry."
    payload["answer_samples"][2]["target_answer"] = "There are 2 letter r in the word strawberry."

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("target_answers must match" in failure for failure in failures)
    assert any("canonical answer" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_missing_budget() -> None:
    payload = copy.deepcopy(load_result())
    payload.pop("fairness", None)

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("fairness must include train_examples_per_model" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_unshared_train_batches() -> None:
    payload = copy.deepcopy(load_result())
    rows = {row["model"]: row for row in payload["runs"]}
    rows["pulse"]["train_batch_group"] = "pulse"
    rows["pulse"]["train_seed"] += 1

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("mean_wide and Pulse" in failure for failure in failures)
    assert any("train_batch_group" in failure or "train seed differs" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_unmatched_wide_baseline() -> None:
    payload = copy.deepcopy(load_result())
    payload["fairness"]["matched_baseline_to_pulse_parameter_ratio"] = 1.25

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("mean_wide parameter ratio" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_untracked_cuda_memory() -> None:
    payload = copy.deepcopy(load_result())
    payload["resource_profile"]["cuda_runtime_used"] = True
    payload["resource_profile"]["records_peak_cuda_memory"] = True
    payload["runs"][0]["peak_cuda_memory_bytes"] = None

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("CUDA peak memory" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_allows_cpu_run_on_cuda_host_without_peak_memory() -> None:
    payload = copy.deepcopy(load_result())
    payload["provenance"]["cuda_available"] = True
    payload["config"]["device"] = "cpu"
    payload["resource_profile"]["requested_device"] = "cpu"
    payload["resource_profile"]["device_type"] = "cpu"
    payload["resource_profile"]["cuda_runtime_used"] = False
    payload["resource_profile"]["records_peak_cuda_memory"] = False
    payload["resource_profile"]["amp_enabled"] = False
    for row in payload["runs"]:
        row["requested_device"] = "cpu"
        row["device_type"] = "cpu"
        row["amp_enabled"] = False
        row["peak_cuda_memory_bytes"] = None
    payload["provenance"]["command"] = payload["provenance"]["command"].replace("--device cuda", "--device cpu")

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert not any("CUDA peak memory" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_missing_resource_profile() -> None:
    payload = copy.deepcopy(load_result())
    payload.pop("resource_profile", None)

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("resource_profile must include requested_device" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_resource_profile_budget_mismatch() -> None:
    payload = copy.deepcopy(load_result())
    payload["resource_profile"]["effective_batch_size"] += 1
    payload["resource_profile"]["prefix_examples"] += 1

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("effective_batch_size" in failure for failure in failures)
    assert any("prefix_examples" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_row_resource_mismatch() -> None:
    payload = copy.deepcopy(load_result())
    payload["runs"][0]["eval_prefix_examples"] += 1

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("eval_prefix_examples" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_bad_parameter_sum() -> None:
    payload = copy.deepcopy(load_result())
    payload["runs"][0]["parameters"] += 1

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("parameter count" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_bad_throughput() -> None:
    payload = copy.deepcopy(load_result())
    payload["runs"][0]["examples_per_second"] = 0.0

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("examples_per_second must be positive" in failure for failure in failures)
    assert any("examples_per_second must equal" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_stale_resource_aggregates() -> None:
    payload = copy.deepcopy(load_result())
    payload["resource_profile"]["min_examples_per_second"] += 1.0
    payload["resource_profile"]["max_examples_per_second"] += 1.0
    payload["resource_profile"]["max_peak_cuda_memory_bytes"] += 1

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("min_examples_per_second must match runs" in failure for failure in failures)
    assert any("max_examples_per_second must match runs" in failure for failure in failures)
    assert any("max_peak_cuda_memory_bytes must match runs" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_stale_summary_throughput() -> None:
    payload = copy.deepcopy(load_result())
    payload["summary"][0]["examples_per_second"] += 1.0

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("summary" in failure and "examples_per_second" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_stale_summary_train_seconds() -> None:
    payload = copy.deepcopy(load_result())
    payload["summary"][0]["train_seconds"] += 1.0

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("summary" in failure and "train_seconds" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_inconsistent_training_loss() -> None:
    payload = copy.deepcopy(load_result())
    payload["runs"][0]["final_loss"] += 1.0

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("final_loss must equal token, prompt-conditioning, and answer-identity losses" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_inconsistent_eval_joint_loss() -> None:
    payload = copy.deepcopy(load_result())
    payload["runs"][0]["eval_joint_loss"] += 1.0

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("eval_joint_loss must equal token, prompt-conditioning, and answer-identity losses" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_stale_summary() -> None:
    payload = copy.deepcopy(load_result())
    payload["summary"][0]["open_loop_exact_rate"] = 1.0

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("summary" in failure and "does not match runs" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_decoupled_identity_loss() -> None:
    payload = copy.deepcopy(load_result())
    payload["training_objective_contract"]["loss_formula"] = (
        "cross_entropy(frozen_lm_head(arti_side_model(rendered_text_tensor, mask)), target_next_token_id) "
        "+ condition_loss_weight * cross_entropy(prompt_condition_head(hidden), prompt_index)"
    )
    payload["fairness"]["prompt_balanced_sampling"] = False

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("directly bind answer identity" in failure for failure in failures)
    assert any("prompt-balanced sampling" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_stale_sample_aggregation() -> None:
    payload = copy.deepcopy(load_result())
    for sample in payload["answer_samples"]:
        generated = {row["model"]: row for row in sample["generated"]}
        if generated["pulse"]["exact"] is True:
            generated["pulse"]["exact"] = False
            break

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("pulse open_loop_exact_rate must match answer_samples aggregation" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_missing_sample_model() -> None:
    payload = copy.deepcopy(load_result())
    payload["answer_samples"][0]["generated"] = [
        row for row in payload["answer_samples"][0]["generated"] if row["model"] != "mean_string"
    ]

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("mean_string sample 0 missing generated answer" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_degenerate_positive_sample() -> None:
    payload = copy.deepcopy(load_result())
    generated = {row["model"]: row for row in payload["answer_samples"][0]["generated"]}
    generated["pulse"]["exact"] = False
    generated["pulse"]["coherence"] = 0.1
    generated["pulse"]["loop_penalty"] = 0.9
    generated["pulse"]["degenerate"] = True

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("pulse sample 0 must exactly match" in failure for failure in failures)
    assert any("pulse sample 0 coherence is too low" in failure for failure in failures)
    assert any("pulse sample 0 loop penalty is too high" in failure for failure in failures)
    assert any("pulse sample 0 must not be marked degenerate" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_high_sample_nll() -> None:
    payload = copy.deepcopy(load_result())
    generated = {row["model"]: row for row in payload["answer_samples"][0]["generated"]}
    generated["pulse_refiner_no_half"]["conditional_nll"] = 3.0

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("pulse_refiner_no_half sample 0 conditional NLL is too high" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_nonfinite_metrics() -> None:
    payload = copy.deepcopy(load_result())
    generated = {row["model"]: row for row in payload["answer_samples"][0]["generated"]}
    generated["mean_string"]["conditional_nll"] = float("nan")
    rows = {row["model"]: row for row in payload["runs"]}
    rows["mean_string"]["open_loop_conditional_nll_mean"] = float("nan")

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("conditional_nll" in failure and "finite" in failure for failure in failures)
    assert any("open_loop_conditional_nll_mean" in failure and "finite" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_missing_task_coverage() -> None:
    payload = copy.deepcopy(load_result())
    payload["data"].pop("task_types", None)

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("data must include task_types" in failure for failure in failures)
    assert any("data task_types must cover" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_unknown_sample_task() -> None:
    payload = copy.deepcopy(load_result())
    payload["answer_samples"][0]["task_type"] = "unknown"

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("answer samples must cover task types" in failure for failure in failures)
    assert any("answer sample 0 has unknown task_type" in failure for failure in failures)


def test_qwen_string_recall_refiner_answering_rejects_incomplete_reproducer_command() -> None:
    payload = copy.deepcopy(load_result())
    payload["provenance"]["command"] = "benchmarks/run_qwen_string_recall_refiner_answering.py"

    failures = verify_qwen_string_recall_refiner_answering.verify(payload)

    assert any("provenance command" in failure and "--steps" in failure for failure in failures)
