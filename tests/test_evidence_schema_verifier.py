from __future__ import annotations

import importlib.util
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_evidence_schema", ROOT / "benchmarks" / "verify_evidence_schema.py")
verify_schema = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_schema)


def complete_payload() -> dict:
    return {
        "status": "completed",
        "scope": "unit test benchmark",
        "claim_boundary": "unit test only",
        "provenance": {"command": "test"},
        "config": {"seed": 0},
        "runs": [{"model": "a", "score": 1.0}],
        "summary": [{"model": "a", "score": 1.0}],
        "metrics": {"primary": "score"},
    }


def test_evidence_schema_accepts_complete_payload(tmp_path: Path) -> None:
    path = tmp_path / "result.json"

    assert verify_schema.verify_payload(path, complete_payload()) == []


def test_evidence_schema_rejects_missing_core_field(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    payload = complete_payload()
    payload.pop("claim_boundary")

    failures = verify_schema.verify_payload(path, payload)

    assert any("missing claim_boundary" in failure for failure in failures)


def test_evidence_schema_rejects_nonfinite_numbers(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    payload = complete_payload()
    payload["runs"][0]["score"] = float("nan")

    failures = verify_schema.verify_payload(path, payload)

    assert any("non-finite numeric value" in failure for failure in failures)


def test_evidence_schema_requires_qwen_string_contracts(tmp_path: Path) -> None:
    path = tmp_path / "qwen_string_recall_refiner_answering_results.json"
    payload = complete_payload()

    failures = verify_schema.verify_payload(path, payload)

    assert any("resource_profile must be present" in failure for failure in failures)
    assert any("training_objective_contract must be present" in failure for failure in failures)


def test_evidence_schema_rejects_bad_qwen_string_objective(tmp_path: Path) -> None:
    path = tmp_path / "qwen_string_recall_refiner_answering_results.json"
    payload = complete_payload()
    payload["resource_profile"] = {
        "requested_device": "cuda",
        "device_type": "cuda",
        "cuda_runtime_used": True,
        "amp_enabled": True,
        "effective_batch_size": 64,
        "throughput_metric": "training_examples_per_second",
        "records_peak_cuda_memory": True,
        "qwen_body_used_during_adapter_training": True,
        "qwen_body_used_for_teacher_generation": True,
        "qwen_body_used_for_conditional_nll": True,
        "lm_head_used_during_adapter_training": True,
    }
    payload["training_objective_contract"] = {
        "supervision_unit": "prefix_next_token",
        "loss_formula": "hidden_mse",
        "target_source": "hidden",
        "target_sequence_terminator": "none",
        "uses_hidden_reconstruction_loss": True,
        "uses_open_loop_strings_as_training_loss": True,
        "open_loop_strings_are_evaluation_only": False,
        "trainable_components": ["arti_side_model"],
    }

    failures = verify_schema.verify_payload(path, payload)

    assert any("Qwen body must not be used during adapter training" in failure for failure in failures)
    assert any("teacher generation must not be used" in failure for failure in failures)
    assert any("canonical answer string targets" in failure for failure in failures)
    assert any("eos_token_id" in failure for failure in failures)
    assert any("must not use hidden reconstruction loss" in failure for failure in failures)
    assert any("open-loop strings must not be the training loss" in failure for failure in failures)
    assert any("open-loop strings must be evaluation-only" in failure for failure in failures)


def test_evidence_schema_requires_pulse_repair_resource_profile(tmp_path: Path) -> None:
    path = tmp_path / "pulse_refiner_repair_results.json"
    payload = complete_payload()

    failures = verify_schema.verify_payload(path, payload)

    assert any("resource_profile must be present" in failure for failure in failures)
    assert any("training_objective_contract must be present" in failure for failure in failures)


def test_evidence_schema_requires_recall_refiner_resource_profile(tmp_path: Path) -> None:
    path = tmp_path / "recall_refiner_results.json"
    payload = complete_payload()

    failures = verify_schema.verify_payload(path, payload)

    assert any("resource_profile must be present" in failure for failure in failures)


def test_gate_schema_accepts_quality_gate_payload(tmp_path: Path) -> None:
    path = tmp_path / "quality_gate.json"
    payload = {
        "python": "3.14.0",
        "gates": ["quick"],
        "passed": True,
        "runs": [{"command": "python test.py", "returncode": 0, "seconds": 0.1}],
    }

    assert verify_schema.verify_gate_payload(path, payload) == []


def test_gate_schema_rejects_nonfinite_seconds(tmp_path: Path) -> None:
    path = tmp_path / "quality_gate.json"
    payload = {
        "python": "3.14.0",
        "gates": ["quick"],
        "passed": True,
        "runs": [{"command": "python test.py", "returncode": 0, "seconds": float("inf")}],
    }

    failures = verify_schema.verify_gate_payload(path, payload)

    assert any("non-finite numeric value" in failure for failure in failures)


def test_default_results_include_qwen_bridge_ablation_probes() -> None:
    paths = {path.as_posix() for path in verify_schema.DEFAULT_RESULTS}

    assert "benchmarks/results/qwen_runtime_vocab_pulse_strict_surface_probe.json" in paths
    assert "benchmarks/results/qwen_runtime_vocab_pulse_metadata_bridge_probe.json" in paths
    assert "benchmarks/results/qwen_learned_pulse_adapter_results.json" in paths
    assert "benchmarks/results/qwen_string_recall_refiner_answering_results.json" in paths
    assert "benchmarks/results/recall_refiner_results.json" in paths


def test_evidence_schema_reference_matches_default_results() -> None:
    docs_path = ROOT / "docs" / "reference" / "evidence-schema.md"
    text = docs_path.read_text(encoding="utf-8")
    core_section = text.split("The verifier currently checks these core result files:", maxsplit=1)[1]
    core_section = core_section.split("It also checks these gate reports:", maxsplit=1)[0]
    documented = set(re.findall(r"`(benchmarks/results/[^`]+\.json)`", core_section))
    defaults = {path.as_posix() for path in verify_schema.DEFAULT_RESULTS}

    assert documented == defaults


def test_default_gate_results_include_mainline_gate() -> None:
    paths = {path.as_posix() for path in verify_schema.DEFAULT_GATE_RESULTS}

    assert "benchmarks/results/quality_gate_mainline.json" in paths
