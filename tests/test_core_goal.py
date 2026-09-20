from __future__ import annotations

import copy
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_core_goal", ROOT / "benchmarks" / "verify_core_goal.py")
verify_core_goal = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verify_core_goal
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_core_goal)


def load_result(name: str) -> dict:
    return json.loads((ROOT / "benchmarks" / "results" / name).read_text(encoding="utf-8"))


def payloads() -> dict[str, dict[str, dict]]:
    return {
        "qwen": {
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
        },
        "latent_repair": {
            "half_trace": load_result("half_recall_trace_survival_results.json"),
            "half_training": load_result("half_recall_fair_training_results.json"),
            "recall_refiner": load_result("recall_refiner_results.json"),
            "pulse_repair": load_result("pulse_refiner_repair_results.json"),
            "pulse_multiseed": load_result("pulse_refiner_multiseed_results.json"),
        },
    }


def test_core_goal_passes() -> None:
    assert verify_core_goal.verify(**payloads()) == []


def test_core_goal_rejects_qwen_string_objective_drift() -> None:
    data = payloads()
    data["qwen"] = copy.deepcopy(data["qwen"])
    data["qwen"]["string_answer"]["runs"][0]["final_loss"] += 1.0

    failures = verify_core_goal.verify(**data)

    assert any("qwen goal" in failure and "final_loss must equal token, prompt-conditioning, and answer-identity losses" in failure for failure in failures)


def test_core_goal_rejects_latent_repair_regression() -> None:
    data = payloads()
    data["latent_repair"] = copy.deepcopy(data["latent_repair"])
    rows = {row["variant"]: row for row in data["latent_repair"]["pulse_repair"]["runs"]}
    rows["pulse_refiner_half"]["weak_noise_gain"] = rows["fold_refiner_no_half"]["weak_noise_gain"]
    data["latent_repair"]["pulse_repair"]["pulse_effect"]["weak_noise_gain_ratio"] = 1.0

    failures = verify_core_goal.verify(**data)

    assert any("latent repair goal" in failure and "weak-noise gain" in failure for failure in failures)


def test_core_goal_rejects_missing_payload_group() -> None:
    data = payloads()
    data["qwen"].pop("string_answer")

    failures = verify_core_goal.verify(**data)

    assert any("missing Qwen evidence payloads" in failure for failure in failures)
