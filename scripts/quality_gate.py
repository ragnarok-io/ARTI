"""Run ARTI industrial package quality gates."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "benchmarks" / "results" / "quality_gate_report.json"

PY = sys.executable
_RUN_TEMP_SUFFIX = f"{os.getpid()}-{time.time_ns()}"

QUICK_COMMANDS = [
    [PY, "-m", "pytest", "--basetemp=.tmp/quality-pytest", "tests/test_backend_capabilities.py", "tests/test_torch_cuda_runtime_verifier.py", "tests/test_package_metadata.py", "tests/test_backend_namespaces.py", "tests/test_torch_backend_runtime.py", "tests/test_generated_docs.py", "tests/test_fit_api.py", "tests/test_arti_st.py", "tests/test_layers.py", "tests/test_layered_recall.py", "tests/test_layered_recall_benchmark.py", "tests/test_recall_topology.py", "tests/test_recall_topology_confirmation.py", "tests/test_recall_topology_evidence.py", "tests/test_qwen_layered_recall_protocol.py", "tests/test_qwen_layered_recall_v2_protocol.py", "tests/test_usage_api.py", "tests/test_inspection.py", "tests/test_feature_matrix.py", "tests/test_masking.py", "tests/test_device.py", "tests/test_serialization.py", "tests/test_text_tensor.py", "tests/test_half_activation.py", "tests/test_half_recall_trace_survival.py", "tests/test_half_recall_fair_training.py", "tests/test_recall_refiner.py", "tests/test_recall_refiner_benchmark.py", "tests/test_pulse_refiner_repair.py", "tests/test_pulse_refiner_multiseed.py", "tests/test_latent_repair_goal.py", "tests/test_core_goal.py", "tests/test_membrane.py", "tests/test_qwen_integration.py", "tests/test_qwen_dynamic_vocab_goal.py", "tests/test_qwen_string_recall_refiner_answering.py", "tests/test_qwen_string_pulse_multiseed.py", "tests/test_qwen_runtime_vocab_metadata_bridge.py", "tests/test_qwen_runtime_vocab_bridge_ablation.py", "tests/test_objective_bank_engineering_gate.py", "tests/test_quality_gate.py", "tests/test_evidence_schema_verifier.py"],
    [PY, "-m", "pytest", "--basetemp=.tmp/quality-objective-formula", "tests/test_objective_formula_control.py", "tests/test_objective_formula_engineering_gate.py"],
    [PY, "-m", "pytest", "--basetemp=.tmp/quality-attachment", "tests/test_attachment.py", "tests/test_attachment_training.py", "tests/test_attachment_hub.py", "tests/test_recall_scaling_screen.py", "tests/test_recall_scaling_evidence.py", "tests/test_qwen_recall_scaling_protocol.py", "tests/test_qwen_unified_attachment_protocol.py", "tests/test_qwen_unified_training_protocol.py", "tests/test_qwen_hub_lifecycle_protocol.py"],
    [PY, "scripts/validate_arti.py"],
    [PY, "benchmarks/report_backend_capabilities.py"],
    [PY, "benchmarks/verify_backend_capabilities.py"],
    [PY, "benchmarks/verify_torch_cuda_runtime.py", "--allow-cpu-torch"],
    [PY, "benchmarks/verify_evidence_schema.py"],
    [PY, "examples/coord_mask_visibility_recall.py"],
    [PY, "examples/arti_st_roundtrip.py"],
]

DOCS_COMMANDS = [
    [PY, "-m", "arti.cli", "docs", "check"],
    [PY, "-m", "arti.cli", "schema", "fit-config", "check"],
    [PY, "-m", "arti.cli", "schema", "task-graph", "check"],
    [PY, "-m", "mkdocs", "build", "--strict"],
]

PACKAGE_COMMANDS = [
    [PY, "scripts/check_lifecycle_contract.py"],
    [PY, "scripts/check_package.py"],
    [PY, "scripts/check_release_readiness.py"],
]

JAX_COMMANDS = [
    [PY, "-m", "pytest", "--basetemp=.tmp/quality-jax", "-q", "tests/test_jax_contract.py", "tests/test_jax_torch_parity.py"],
    [PY, "-m", "arti.cli", "doctor", "--allow-cpu-torch", "--require-jax-smoke"],
]

MAINLINE_COMMANDS = [
    [PY, "benchmarks/verify_qwen_hidden_refiner_half.py"],
    [PY, "benchmarks/verify_core_goal.py"],
    [PY, "benchmarks/verify_evidence_schema.py", "--skip-gates"],
]

MECHANISM_COMMANDS = [
    [PY, "benchmarks/verify_virtual_recall_alignment.py"],
    [PY, "benchmarks/verify_experiential_recall_scaling.py"],
    [PY, "benchmarks/verify_recall_scaling_law.py"],
    [PY, "benchmarks/verify_recall_recognition_modes.py"],
    [PY, "benchmarks/verify_layered_recall_trajectory.py"],
    [PY, "benchmarks/verify_visual_field_concat.py"],
    [PY, "benchmarks/verify_visual_scan_superresolution.py"],
    [PY, "benchmarks/run_half_recall_trace_survival.py", "--output", "benchmarks/results/half_recall_trace_survival_results.json"],
    [PY, "benchmarks/verify_half_recall_trace_survival.py"],
    [PY, "benchmarks/run_half_recall_fair_training.py", "--output", "benchmarks/results/half_recall_fair_training_results.json"],
    [PY, "benchmarks/verify_half_recall_fair_training.py"],
    [PY, "benchmarks/run_recall_refiner.py", "--output", "benchmarks/results/recall_refiner_results.json"],
    [PY, "benchmarks/verify_recall_refiner.py"],
    [PY, "benchmarks/run_fold_compaction.py", "--output", "benchmarks/results/fold_compaction_results.json"],
    [PY, "benchmarks/verify_fold_compaction.py"],
    [PY, "benchmarks/run_learned_pulse_compaction.py", "--output", "benchmarks/results/learned_pulse_compaction_results.json"],
    [PY, "benchmarks/verify_learned_pulse_compaction.py"],
    [PY, "benchmarks/verify_pulse_refiner_repair.py"],
    [PY, "benchmarks/verify_runtime_vocab_binding.py"],
    [PY, "benchmarks/verify_runtime_vocab_replacement.py"],
    [PY, "benchmarks/verify_runtime_vocab_permutation.py"],
    [PY, "benchmarks/verify_pulse_vocab_invariance.py"],
    [PY, "benchmarks/verify_pulse_distinctness.py"],
    [PY, "benchmarks/run_micro_glyph_distinctness.py", "--output", "benchmarks/results/micro_glyph_distinctness_results.json"],
    [PY, "benchmarks/verify_micro_glyph_distinctness.py"],
    [PY, "benchmarks/verify_text_bitmap_vocab.py"],
    [PY, "benchmarks/verify_source_integrity_stereo.py"],
    [PY, "benchmarks/verify_source_integrity_multisource.py"],
    [PY, "benchmarks/verify_multisource_reliability_arena.py"],
    [PY, "benchmarks/verify_authority_prompt_injection.py"],
    [PY, "benchmarks/verify_coord_virtual_recall_phase_corruption.py"],
    [PY, "benchmarks/run_membrane_visibility_routing.py"],
    [PY, "benchmarks/verify_membrane_visibility_routing.py"],
    [PY, "benchmarks/verify_operator_bank_image.py"],
    [PY, "benchmarks/verify_operator_bank_image.py", "--results", "benchmarks/results/operator_bank_image_large_results.json"],
    [PY, "benchmarks/verify_operator_bank_image.py", "--results", "benchmarks/results/operator_bank_image_pressure.json"],
    [PY, "benchmarks/verify_autoregressive_observer_phase.py"],
]

CUDA_COMMANDS = [
    [PY, "benchmarks/report_backend_capabilities.py"],
    [PY, "benchmarks/verify_backend_capabilities.py"],
    [PY, "benchmarks/verify_torch_cuda_runtime.py"],
    [PY, "benchmarks/profile_scaling.py", "--device", "cuda", "--output", "benchmarks/results/cuda_scaling_profile.json"],
    [PY, "benchmarks/verify_cuda_scaling.py", "--profile", "benchmarks/results/cuda_scaling_profile.json"],
    [PY, "benchmarks/build_cuda_evidence_packet.py", "--profile", "benchmarks/results/cuda_scaling_profile.json"],
    [PY, "benchmarks/verify_cuda_evidence_packet.py", "--packet", "benchmarks/results/cuda_evidence_packet.json"],
    [PY, "benchmarks/run_pulse_refiner_repair.py", "--device", "cuda", "--output", "benchmarks/results/pulse_refiner_repair_results.json"],
    [PY, "benchmarks/verify_pulse_refiner_repair.py"],
    [PY, "benchmarks/run_pulse_refiner_multiseed.py", "--device", "cuda", "--output", "benchmarks/results/pulse_refiner_multiseed_results.json"],
    [PY, "benchmarks/verify_pulse_refiner_multiseed.py"],
]

QWEN_COMMANDS = [
    [PY, "benchmarks/run_qwen_arti_smoke.py", "--output", "benchmarks/results/qwen_arti_smoke_results.json"],
    [PY, "benchmarks/verify_qwen_arti_smoke.py"],
    [PY, "benchmarks/run_qwen_glyph_runtime_adapter_api.py", "--output", "benchmarks/results/qwen_glyph_runtime_adapter_api_results.json"],
    [PY, "benchmarks/verify_qwen_glyph_runtime_adapter_api.py"],
    [PY, "benchmarks/run_qwen_runtime_vocab_replacement.py", "--output", "benchmarks/results/qwen_runtime_vocab_replacement_results.json"],
    [PY, "benchmarks/verify_qwen_runtime_vocab_replacement.py"],
    [PY, "benchmarks/run_qwen_runtime_vocab_training.py", "--steps", "120", "--eval-batches", "16", "--output", "benchmarks/results/qwen_runtime_vocab_training_results.json"],
    [PY, "benchmarks/verify_qwen_runtime_vocab_training.py"],
    [PY, "benchmarks/run_qwen_output_head_replacement.py", "--steps", "180", "--eval-batches", "24", "--output", "benchmarks/results/qwen_output_head_replacement_results.json"],
    [PY, "benchmarks/verify_qwen_output_head_replacement.py"],
    [PY, "benchmarks/run_qwen_literal_output_context.py", "--output", "benchmarks/results/qwen_literal_output_context_results.json"],
    [PY, "benchmarks/verify_qwen_literal_output_context.py"],
    [PY, "benchmarks/run_qwen_unseen_literal_transfer.py", "--output", "benchmarks/results/qwen_unseen_literal_transfer_results.json"],
    [PY, "benchmarks/verify_qwen_unseen_literal_transfer.py"],
    [PY, "benchmarks/run_qwen_literal_segmentation_generation.py", "--output", "benchmarks/results/qwen_literal_segmentation_generation_results.json"],
    [PY, "benchmarks/verify_qwen_literal_segmentation_generation.py"],
    [PY, "benchmarks/run_qwen_input_head_replacement.py", "--steps", "260", "--output", "benchmarks/results/qwen_input_head_replacement_results.json"],
    [PY, "benchmarks/verify_qwen_input_head_replacement.py"],
    [PY, "benchmarks/run_qwen_closed_loop_replacement.py", "--input-steps", "520", "--output-steps", "180", "--eval-batches", "24", "--output", "benchmarks/results/qwen_closed_loop_replacement_results.json"],
    [PY, "benchmarks/verify_qwen_closed_loop_replacement.py"],
    [PY, "benchmarks/run_qwen_autoregressive_closed_loop.py", "--input-steps", "520", "--output-steps", "180", "--eval-batches", "16", "--output", "benchmarks/results/qwen_autoregressive_closed_loop_results.json"],
    [PY, "benchmarks/verify_qwen_autoregressive_closed_loop.py"],
    [PY, "benchmarks/run_qwen_controlled_open_generation.py", "--input-steps", "640", "--output-steps", "260", "--eval-batches", "16", "--candidate-count", "48", "--output", "benchmarks/results/qwen_controlled_open_generation_results.json"],
    [PY, "benchmarks/verify_qwen_controlled_open_generation.py"],
    [PY, "benchmarks/run_qwen_runtime_vocab_pulse_semantic.py", "--steps", "220", "--eval-batches", "16", "--batch-size", "128", "--grad-accum-steps", "2", "--output", "benchmarks/results/qwen_runtime_vocab_pulse_semantic_results.json"],
    [PY, "benchmarks/verify_qwen_runtime_vocab_pulse_semantic.py"],
    [PY, "benchmarks/run_qwen_learned_pulse_adapter.py", "--steps", "220", "--eval-batches", "16", "--batch-size", "128", "--output", "benchmarks/results/qwen_learned_pulse_adapter_results.json"],
    [PY, "benchmarks/verify_qwen_learned_pulse_adapter.py"],
    [PY, "benchmarks/verify_qwen_string_recall_refiner_answering.py"],
    [PY, "benchmarks/verify_qwen_string_pulse_multiseed.py"],
    [PY, "benchmarks/run_qwen_runtime_vocab_pulse_semantic.py", "--steps", "220", "--eval-batches", "16", "--batch-size", "128", "--grad-accum-steps", "2", "--strict-heldout-surfaces", "--output", "benchmarks/results/qwen_runtime_vocab_pulse_strict_surface_probe.json"],
    [PY, "benchmarks/run_qwen_runtime_vocab_pulse_semantic.py", "--steps", "520", "--eval-batches", "16", "--batch-size", "128", "--grad-accum-steps", "2", "--strict-heldout-surfaces", "--semantic-bridge", "--semantic-bridge-source", "metadata", "--vocab-glyph-scale", "0.02", "--output", "benchmarks/results/qwen_runtime_vocab_pulse_metadata_bridge_probe.json"],
    [PY, "benchmarks/verify_qwen_runtime_vocab_metadata_bridge.py"],
    [PY, "benchmarks/verify_qwen_runtime_vocab_bridge_ablation.py"],
    [PY, "benchmarks/run_qwen_oov_text_vocab_finetune.py", "--steps", "120", "--eval-batches", "16", "--output", "benchmarks/results/qwen_oov_text_vocab_finetune_results.json"],
    [PY, "benchmarks/verify_qwen_oov_text_vocab_finetune.py"],
    [PY, "benchmarks/run_qwen_external_glyph_answering.py", "--input-steps", "560", "--output-steps", "220", "--eval-batches", "16", "--candidate-count", "24", "--output", "benchmarks/results/qwen_external_glyph_answering_results.json"],
    [PY, "benchmarks/verify_qwen_external_glyph_answering.py"],
    [PY, "benchmarks/run_qwen_multitoken_glyph_decoder.py", "--input-steps", "560", "--output-steps", "260", "--eval-batches", "16", "--candidate-count", "32", "--output", "benchmarks/results/qwen_multitoken_glyph_decoder_results.json"],
    [PY, "benchmarks/verify_qwen_multitoken_glyph_decoder.py"],
    [PY, "benchmarks/run_qwen_oov_dialogue_preservation.py", "--steps", "120", "--eval-batches", "16", "--output", "benchmarks/results/qwen_oov_dialogue_preservation_results.json"],
    [PY, "benchmarks/verify_qwen_oov_dialogue_preservation.py"],
    [PY, "benchmarks/verify_qwen_dynamic_vocab_goal.py"],
    [PY, "benchmarks/run_qwen_membrane_routing_training.py", "--steps", "220", "--eval-batches", "24", "--output", "benchmarks/results/qwen_membrane_routing_training_results.json"],
    [PY, "benchmarks/verify_qwen_membrane_routing_training.py"],
    [PY, "benchmarks/run_qwen_membrane_guessing_game.py", "--steps", "260", "--eval-batches", "24", "--output", "benchmarks/results/qwen_membrane_guessing_game_results.json"],
    [PY, "benchmarks/verify_qwen_membrane_guessing_game.py"],
    [PY, "benchmarks/run_qwen_membrane_guessing_game_real_forward.py", "--steps", "30", "--train-games", "12", "--eval-games", "6", "--output", "benchmarks/results/qwen_membrane_guessing_game_real_forward_results.json"],
    [PY, "benchmarks/verify_qwen_membrane_guessing_game_real_forward.py"],
]

PRETRAINED_COMMANDS = [
    [PY, "-m", "pytest", "--basetemp=.tmp/quality-pretrained", "tests/test_pretrained_workflow.py", "tests/test_pretrained_ecosystem_smoke.py", "tests/test_pretrained_distributed_smoke.py"],
    [PY, "benchmarks/run_pretrained_distributed_gate.py", "--output", "benchmarks/results/pretrained_distributed_smoke.json"],
    [PY, "benchmarks/verify_pretrained_distributed_smoke.py"],
    [PY, "benchmarks/run_pretrained_ecosystem_smoke.py", "--device", "auto", "--output", "benchmarks/results/pretrained_ecosystem_smoke.json"],
    [PY, "benchmarks/verify_pretrained_ecosystem_smoke.py"],
]

GATES = {
    "quick": QUICK_COMMANDS,
    "docs": DOCS_COMMANDS,
    "package": PACKAGE_COMMANDS,
    "jax": JAX_COMMANDS,
    "mainline": MAINLINE_COMMANDS,
    "mechanism": MECHANISM_COMMANDS,
    "cuda": CUDA_COMMANDS,
    "qwen": QWEN_COMMANDS,
    "pretrained": PRETRAINED_COMMANDS,
}


def display_command(command: list[str]) -> str:
    return " ".join(command)


def runtime_command(command: list[str]) -> list[str]:
    """Give pytest a fresh basetemp while keeping reports reproducible."""

    return [
        f"{part}-{_RUN_TEMP_SUFFIX}" if part.startswith("--basetemp=") else part
        for part in command
    ]


def generated_output_paths(command: list[str]) -> list[Path]:
    if not command:
        return []
    script = Path(command[0]).name if len(command) == 1 else Path(command[1]).name
    if script.startswith(("verify_", "check_")):
        return []
    if "--output" not in command:
        return []
    output_index = command.index("--output") + 1
    if output_index >= len(command):
        return []
    output = (ROOT / command[output_index]).resolve()
    results_root = (ROOT / "benchmarks" / "results").resolve()
    if results_root not in output.parents:
        return []
    return [output, output.with_suffix(".md")]


def remove_stale_outputs(command: list[str]) -> None:
    for path in generated_output_paths(command):
        if path.exists():
            path.unlink()


def reusable_passing_row(command: list[str], previous_runs: list[dict]) -> dict | None:
    output_paths = generated_output_paths(command)
    if not output_paths or not all(path.exists() for path in output_paths):
        return None
    expected_tail = " ".join(command[1:]).replace("\\", "/")
    for row in previous_runs:
        previous_command = str(row.get("command", "")).replace("\\", "/")
        if row.get("returncode") == 0 and expected_tail in previous_command:
            return row
    return None


def run_or_reuse_command(command: list[str], previous_runs: list[dict], *, reuse_passing_producers: bool) -> dict:
    if reuse_passing_producers:
        previous = reusable_passing_row(command, previous_runs)
        if previous is not None:
            display = display_command(command)
            print(f"REUSE {display}")
            return {
                "command": display,
                "returncode": 0,
                "seconds": 0.0,
                "stdout_tail": previous.get("stdout_tail", ""),
                "stderr_tail": previous.get("stderr_tail", ""),
                "reused": True,
                "reused_from_seconds": previous.get("seconds"),
            }
    return run_command(command)


def run_command(command: list[str]) -> dict:
    started = time.perf_counter()
    display = display_command(command)
    print(f"RUN {display}")
    remove_stale_outputs(command)
    result = subprocess.run(
        runtime_command(command),
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    seconds = time.perf_counter() - started
    print(f"  returncode={result.returncode} seconds={seconds:.3f}")
    return {
        "command": display,
        "returncode": result.returncode,
        "seconds": seconds,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def commands_for(gates: list[str]) -> list[list[str]]:
    selected = []
    expanded = ["quick", "docs", "package", "mainline", "mechanism", "cuda"] if "all" in gates else gates
    for gate in expanded:
        if gate not in GATES:
            raise SystemExit(f"unknown gate: {gate}")
        selected.extend(GATES[gate])
    return selected


def write_markdown(path: Path, payload: dict) -> None:
    has_reused = any(row.get("reused") for row in payload["runs"])
    lines = [
        "# ARTI Quality Gate Report",
        "",
        f"Overall: `{'PASS' if payload['passed'] else 'FAIL'}`",
        "",
        f"Gates: `{payload['gates']}`",
        "",
    ]
    if has_reused:
        lines.extend(["| Command | Return Code | Seconds | Mode |", "| --- | ---: | ---: | --- |"])
    else:
        lines.extend(["| Command | Return Code | Seconds |", "| --- | ---: | ---: |"])
    for row in payload["runs"]:
        if has_reused:
            mode = "reused producer output" if row.get("reused") else "run"
            lines.append(f"| `{row['command']}` | {row['returncode']} | {row['seconds']:.3f} | {mode} |")
        else:
            lines.append(f"| `{row['command']}` | {row['returncode']} | {row['seconds']:.3f} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("gates", nargs="*", default=["quick"], choices=[*GATES.keys(), "all"])
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--reuse-passing-producers",
        action="store_true",
        help="Reuse passing producer outputs from the existing gate report while rerunning current verifiers.",
    )
    args = parser.parse_args()

    previous_runs: list[dict] = []
    if args.reuse_passing_producers and args.output.exists():
        previous_payload = json.loads(args.output.read_text(encoding="utf-8"))
        previous_runs = [row for row in previous_payload.get("runs", []) if isinstance(row, dict)]

    runs = []
    for command in commands_for(args.gates):
        row = run_or_reuse_command(command, previous_runs, reuse_passing_producers=args.reuse_passing_producers)
        runs.append(row)
        if args.fail_fast and row["returncode"] != 0:
            break

    payload = {
        "python": sys.version.split()[0],
        "gates": args.gates,
        "passed": all(row["returncode"] == 0 for row in runs),
        "runs": runs,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    write_markdown(args.output.with_suffix(".md"), payload)
    print(f"Wrote {args.output}")
    print(f"Wrote {args.output.with_suffix('.md')}")
    raise SystemExit(0 if payload["passed"] else 1)


if __name__ == "__main__":
    main()
