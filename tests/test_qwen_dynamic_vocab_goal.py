from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_qwen_dynamic_vocab_goal", ROOT / "benchmarks" / "verify_qwen_dynamic_vocab_goal.py")
verify_qwen_dynamic_vocab_goal = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verify_qwen_dynamic_vocab_goal
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_qwen_dynamic_vocab_goal)


def load_result(name: str) -> dict:
    return json.loads((ROOT / "benchmarks" / "results" / name).read_text(encoding="utf-8"))


def payloads() -> dict[str, dict]:
    return {
        "input_head": load_result("qwen_input_head_replacement_results.json"),
        "closed_loop": load_result("qwen_closed_loop_replacement_results.json"),
        "runtime_pulse": load_result("qwen_runtime_vocab_pulse_semantic_results.json"),
        "no_bridge": load_result("qwen_runtime_vocab_pulse_strict_surface_probe.json"),
        "metadata_bridge": load_result("qwen_runtime_vocab_pulse_metadata_bridge_probe.json"),
        "dialogue": load_result("qwen_oov_dialogue_preservation_results.json"),
        "string_answer": load_result("qwen_string_recall_refiner_answering_results.json"),
        "string_multiseed": load_result("qwen_string_pulse_multiseed_results.json"),
        "literal_output_context": load_result("qwen_literal_output_context_results.json"),
        "unseen_literal_transfer": load_result("qwen_unseen_literal_transfer_results.json"),
        "literal_segmentation_generation": load_result("qwen_literal_segmentation_generation_results.json"),
    }


def test_qwen_dynamic_vocab_goal_evidence_chain_passes() -> None:
    assert verify_qwen_dynamic_vocab_goal.verify(**payloads()) == []


def test_qwen_dynamic_vocab_goal_rejects_missing_glyph_runtime_vocab() -> None:
    data = payloads()
    data["runtime_pulse"] = copy.deepcopy(data["runtime_pulse"])
    data["runtime_pulse"]["dataset"]["runtime_vocab_tensor_mode"] = "analytic"
    failures = verify_qwen_dynamic_vocab_goal.verify(**data)
    assert any("glyph tensors" in failure for failure in failures)


def test_qwen_dynamic_vocab_goal_rejects_tiny_closed_loop_candidate_set() -> None:
    data = payloads()
    data["closed_loop"] = copy.deepcopy(data["closed_loop"])
    data["closed_loop"]["config"]["candidate_count"] = 8
    failures = verify_qwen_dynamic_vocab_goal.verify(**data)
    assert any("output candidates" in failure for failure in failures)


def test_qwen_dynamic_vocab_goal_rejects_tiny_input_readable_probe() -> None:
    data = payloads()
    data["input_head"] = copy.deepcopy(data["input_head"])
    data["input_head"]["readable_probe"] = data["input_head"]["readable_probe"][:1]
    failures = verify_qwen_dynamic_vocab_goal.verify(**data)
    assert any("input head replacement" in failure and "readable probe" in failure for failure in failures)


def test_qwen_dynamic_vocab_goal_rejects_short_closed_loop_training() -> None:
    data = payloads()
    data["closed_loop"] = copy.deepcopy(data["closed_loop"])
    data["closed_loop"]["config"]["input_steps"] = 20
    data["closed_loop"]["config"]["output_steps"] = 20
    failures = verify_qwen_dynamic_vocab_goal.verify(**data)
    assert any("input head" in failure and "500 steps" in failure for failure in failures)
    assert any("output head" in failure and "150 steps" in failure for failure in failures)


def test_qwen_dynamic_vocab_goal_rejects_unsolved_closed_loop_probe() -> None:
    data = payloads()
    data["closed_loop"] = copy.deepcopy(data["closed_loop"])
    for row in data["closed_loop"]["scenario_probe"]:
        if row["scenario"] == "permuted_vocab":
            row["slot_match"] = False
    failures = verify_qwen_dynamic_vocab_goal.verify(**data)
    assert any("scenario probe" in failure and "permuted_vocab" in failure for failure in failures)


def test_qwen_dynamic_vocab_goal_rejects_weak_closed_loop_probe_majority() -> None:
    data = payloads()
    data["closed_loop"] = copy.deepcopy(data["closed_loop"])
    for row in data["closed_loop"]["scenario_probe"]:
        row["slot_match"] = False
    failures = verify_qwen_dynamic_vocab_goal.verify(**data)
    assert any("closed-loop replacement" in failure and "mostly match" in failure for failure in failures)


def test_qwen_dynamic_vocab_goal_rejects_missing_heldout_tokenization_split() -> None:
    data = payloads()
    data["runtime_pulse"] = copy.deepcopy(data["runtime_pulse"])
    data["runtime_pulse"]["dataset"]["heldout_tokenization_variants"] = []
    failures = verify_qwen_dynamic_vocab_goal.verify(**data)
    assert any("held-out evaluation" in failure for failure in failures)


def test_qwen_dynamic_vocab_goal_rejects_tiny_runtime_vocab_bank() -> None:
    data = payloads()
    data["runtime_pulse"] = copy.deepcopy(data["runtime_pulse"])
    data["runtime_pulse"]["resource_efficiency"]["precompute"]["pulse_vocab_items"] = 8
    failures = verify_qwen_dynamic_vocab_goal.verify(**data)
    assert any("pulse vocab bank" in failure for failure in failures)


def test_qwen_dynamic_vocab_goal_rejects_runtime_pulse_without_pulse_usage() -> None:
    data = payloads()
    data["runtime_pulse"] = copy.deepcopy(data["runtime_pulse"])
    rows = {row["model"]: row for row in data["runtime_pulse"]["runs"]}
    rows["runtime_vocab_pulse"]["uses_pulse"] = False
    failures = verify_qwen_dynamic_vocab_goal.verify(**data)
    assert any("runtime vocab pulse" in failure and "both runtime vocab and pulse" in failure for failure in failures)


def test_qwen_dynamic_vocab_goal_rejects_sampling_bottleneck() -> None:
    data = payloads()
    data["runtime_pulse"] = copy.deepcopy(data["runtime_pulse"])
    rows = {row["model"]: row for row in data["runtime_pulse"]["runs"]}
    rows["runtime_vocab_pulse"]["sample_seconds"] = rows["runtime_vocab_pulse"]["train_seconds"]
    failures = verify_qwen_dynamic_vocab_goal.verify(**data)
    assert any("batch sampling" in failure for failure in failures)


def test_qwen_dynamic_vocab_goal_rejects_tiny_training_window() -> None:
    data = payloads()
    data["runtime_pulse"] = copy.deepcopy(data["runtime_pulse"])
    rows = {row["model"]: row for row in data["runtime_pulse"]["runs"]}
    rows["runtime_vocab_pulse"]["steps"] = 20
    failures = verify_qwen_dynamic_vocab_goal.verify(**data)
    assert any("at least 200 steps" in failure for failure in failures)


def test_qwen_dynamic_vocab_goal_rejects_missing_grad_accumulation() -> None:
    data = payloads()
    data["runtime_pulse"] = copy.deepcopy(data["runtime_pulse"])
    data["runtime_pulse"]["resource_efficiency"]["grad_accum_steps"] = 1
    rows = {row["model"]: row for row in data["runtime_pulse"]["runs"]}
    rows["runtime_vocab_pulse"]["grad_accum_steps"] = 1
    failures = verify_qwen_dynamic_vocab_goal.verify(**data)
    assert any("grad_accum_steps" in failure for failure in failures)


def test_qwen_dynamic_vocab_goal_rejects_short_metadata_bridge_training() -> None:
    data = payloads()
    data["metadata_bridge"] = copy.deepcopy(data["metadata_bridge"])
    rows = {row["model"]: row for row in data["metadata_bridge"]["runs"]}
    rows["runtime_vocab_pulse"]["steps"] = 20
    failures = verify_qwen_dynamic_vocab_goal.verify(**data)
    assert any("metadata bridge" in failure and "500 steps" in failure for failure in failures)


def test_qwen_dynamic_vocab_goal_rejects_unsolved_strict_surface_probe() -> None:
    data = payloads()
    data["metadata_bridge"] = load_result("qwen_runtime_vocab_pulse_strict_surface_probe.json")
    failures = verify_qwen_dynamic_vocab_goal.verify(**data)
    assert any("semantic_bridge" in failure or "held-out surface accuracy" in failure for failure in failures)


def test_qwen_dynamic_vocab_goal_rejects_missing_negative_control_gap() -> None:
    data = payloads()
    data["no_bridge"] = copy.deepcopy(data["no_bridge"])
    rows = {row["model"]: row for row in data["no_bridge"]["runs"]}
    rows["runtime_vocab_pulse"]["heldout_surface_accuracy"] = 0.90
    failures = verify_qwen_dynamic_vocab_goal.verify(**data)
    assert any("bridge ablation" in failure and "no-bridge" in failure for failure in failures)


def test_qwen_dynamic_vocab_goal_rejects_zero_glyph_bridge() -> None:
    data = payloads()
    data["metadata_bridge"] = copy.deepcopy(data["metadata_bridge"])
    data["metadata_bridge"]["dataset"]["vocab_glyph_scale"] = 0.0
    failures = verify_qwen_dynamic_vocab_goal.verify(**data)
    assert any("nonzero" in failure for failure in failures)


def test_qwen_dynamic_vocab_goal_rejects_missing_metadata_bridge_dim() -> None:
    data = payloads()
    data["metadata_bridge"] = copy.deepcopy(data["metadata_bridge"])
    data["metadata_bridge"]["dataset"]["semantic_bridge_dim"] = 0
    failures = verify_qwen_dynamic_vocab_goal.verify(**data)
    assert any("metadata bridge" in failure and "semantic bridge dim" in failure for failure in failures)


def test_qwen_dynamic_vocab_goal_rejects_dialogue_side_head_that_does_not_learn_oov() -> None:
    data = payloads()
    data["dialogue"] = copy.deepcopy(data["dialogue"])
    rows = {row["model"]: row for row in data["dialogue"]["runs"]}
    rows["glyph_runtime_head"]["oov_accuracy"] = 0.20
    failures = verify_qwen_dynamic_vocab_goal.verify(**data)
    assert any("dialogue preservation" in failure and "OOV glyph" in failure for failure in failures)


def test_qwen_dynamic_vocab_goal_rejects_string_answer_hidden_objective() -> None:
    data = payloads()
    data["string_answer"] = copy.deepcopy(data["string_answer"])
    data["string_answer"]["training_objective_contract"]["uses_hidden_reconstruction_loss"] = True

    failures = verify_qwen_dynamic_vocab_goal.verify(**data)

    assert any("string-first answer" in failure and "hidden reconstruction loss" in failure for failure in failures)


def test_qwen_dynamic_vocab_goal_rejects_string_answer_loss_drift() -> None:
    data = payloads()
    data["string_answer"] = copy.deepcopy(data["string_answer"])
    data["string_answer"]["runs"][0]["final_loss"] += 1.0

    failures = verify_qwen_dynamic_vocab_goal.verify(**data)

    assert any("string-first answer" in failure and "final_loss must equal token, prompt-conditioning, and answer-identity losses" in failure for failure in failures)


def test_qwen_dynamic_vocab_goal_rejects_multiseed_mode_collapse() -> None:
    data = payloads()
    data["string_multiseed"] = copy.deepcopy(data["string_multiseed"])
    data["string_multiseed"]["summary"][0]["pulse_mode_collapse_count"] = 1

    failures = verify_qwen_dynamic_vocab_goal.verify(**data)

    assert any("string multi-seed" in failure and "mode_collapse_count" in failure for failure in failures)


def test_qwen_dynamic_vocab_goal_rejects_tiny_dialogue_preservation_probe_set() -> None:
    data = payloads()
    data["dialogue"] = copy.deepcopy(data["dialogue"])
    data["dialogue"]["dialogue_preservation"]["prompt_count"] = 1
    failures = verify_qwen_dynamic_vocab_goal.verify(**data)
    assert any("dialogue preservation" in failure and "multiple prompts" in failure for failure in failures)
