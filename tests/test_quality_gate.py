from __future__ import annotations

from pathlib import Path

import scripts.quality_gate as quality_gate


def test_runtime_command_uniquifies_basetemp_without_mutating_report_command() -> None:
    command = [quality_gate.PY, "-m", "pytest", "--basetemp=.tmp/quality-test"]

    runtime = quality_gate.runtime_command(command)

    assert command[-1] == "--basetemp=.tmp/quality-test"
    assert runtime[-1].startswith("--basetemp=.tmp/quality-test-")
    assert runtime[-1] != command[-1]


def test_jax_gate_is_bounded_and_zero_skip() -> None:
    commands = quality_gate.GATES["jax"]
    assert len(commands) == 2
    pytest_command = commands[0]
    assert "tests/test_jax_contract.py" in pytest_command
    assert "tests/test_jax_torch_parity.py" in pytest_command
    assert not any("backend_namespaces" in part for part in pytest_command)


def test_generated_output_paths_only_targets_result_producers() -> None:
    command = [
        quality_gate.PY,
        "benchmarks/run_example.py",
        "--output",
        "benchmarks/results/example.json",
    ]

    assert quality_gate.generated_output_paths(command) == [
        (quality_gate.ROOT / "benchmarks/results/example.json").resolve(),
        (quality_gate.ROOT / "benchmarks/results/example.md").resolve(),
    ]


def test_generated_output_paths_accepts_non_run_result_producers() -> None:
    command = [
        quality_gate.PY,
        "benchmarks/profile_scaling.py",
        "--device",
        "cuda",
        "--output",
        "benchmarks/results/cuda_scaling_profile.json",
    ]

    assert quality_gate.generated_output_paths(command) == [
        (quality_gate.ROOT / "benchmarks/results/cuda_scaling_profile.json").resolve(),
        (quality_gate.ROOT / "benchmarks/results/cuda_scaling_profile.md").resolve(),
    ]


def test_generated_output_paths_ignores_verifiers() -> None:
    command = [
        quality_gate.PY,
        "benchmarks/verify_example.py",
        "--output",
        "benchmarks/results/example.json",
    ]

    assert quality_gate.generated_output_paths(command) == []


def test_remove_stale_outputs_removes_json_and_markdown() -> None:
    json_path = quality_gate.ROOT / "benchmarks/results/tmp_quality_gate_stale.json"
    markdown_path = json_path.with_suffix(".md")
    json_path.write_text("old", encoding="utf-8")
    markdown_path.write_text("old", encoding="utf-8")

    quality_gate.remove_stale_outputs(
        [
            quality_gate.PY,
            "benchmarks/run_example.py",
            "--output",
            str(Path("benchmarks/results/tmp_quality_gate_stale.json")),
        ]
    )

    assert not json_path.exists()
    assert not markdown_path.exists()


def test_reuse_passing_producer_requires_existing_output_and_prior_success() -> None:
    json_path = quality_gate.ROOT / "benchmarks/results/tmp_quality_gate_reuse.json"
    markdown_path = json_path.with_suffix(".md")
    json_path.write_text("fresh", encoding="utf-8")
    markdown_path.write_text("fresh", encoding="utf-8")
    command = [
        quality_gate.PY,
        "benchmarks/run_example.py",
        "--output",
        str(Path("benchmarks/results/tmp_quality_gate_reuse.json")),
    ]
    previous_runs = [
        {
            "command": "old-python benchmarks/run_example.py --output benchmarks/results/tmp_quality_gate_reuse.json",
            "returncode": 0,
            "seconds": 12.5,
            "stdout_tail": "PASS\n",
            "stderr_tail": "",
        }
    ]

    try:
        row = quality_gate.run_or_reuse_command(command, previous_runs, reuse_passing_producers=True)

        assert row["returncode"] == 0
        assert row["reused"] is True
        assert row["reused_from_seconds"] == 12.5
        assert json_path.read_text(encoding="utf-8") == "fresh"
        assert markdown_path.read_text(encoding="utf-8") == "fresh"
    finally:
        json_path.unlink(missing_ok=True)
        markdown_path.unlink(missing_ok=True)


def test_reuse_passing_producer_does_not_reuse_verifier(monkeypatch) -> None:
    calls = []

    def fake_run_command(command: list[str]) -> dict:
        calls.append(command)
        return {"command": quality_gate.display_command(command), "returncode": 0, "seconds": 0.1, "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr(quality_gate, "run_command", fake_run_command)
    command = [
        quality_gate.PY,
        "benchmarks/verify_example.py",
        "--output",
        str(Path("benchmarks/results/tmp_quality_gate_reuse.json")),
    ]

    row = quality_gate.run_or_reuse_command(
        command,
        [{"command": "old-python benchmarks/verify_example.py --output benchmarks/results/tmp_quality_gate_reuse.json", "returncode": 0}],
        reuse_passing_producers=True,
    )

    assert row["returncode"] == 0
    assert "reused" not in row
    assert calls == [command]


def test_qwen_gate_runs_metadata_bridge_probe_before_goal_verifier() -> None:
    commands = [" ".join(command) for command in quality_gate.QWEN_COMMANDS]
    run_index = next(index for index, command in enumerate(commands) if "qwen_runtime_vocab_pulse_metadata_bridge_probe.json" in command)
    verify_index = next(index for index, command in enumerate(commands) if "verify_qwen_runtime_vocab_metadata_bridge.py" in command)
    goal_index = next(index for index, command in enumerate(commands) if "verify_qwen_dynamic_vocab_goal.py" in command)

    assert "--strict-heldout-surfaces" in commands[run_index]
    assert "--semantic-bridge" in commands[run_index]
    assert "--semantic-bridge-source metadata" in commands[run_index]
    assert "--vocab-glyph-scale 0.02" in commands[run_index]
    assert run_index < verify_index < goal_index


def test_mechanism_gate_runs_learned_pulse_after_fold() -> None:
    commands = [" ".join(command) for command in quality_gate.MECHANISM_COMMANDS]
    fold_index = next(index for index, command in enumerate(commands) if "run_fold_compaction.py" in command)
    learned_index = next(index for index, command in enumerate(commands) if "run_learned_pulse_compaction.py" in command)
    learned_verify_index = next(index for index, command in enumerate(commands) if "verify_learned_pulse_compaction.py" in command)

    assert fold_index < learned_index < learned_verify_index


def test_mechanism_gate_runs_recall_refiner_after_half_recall_trace() -> None:
    commands = [" ".join(command) for command in quality_gate.MECHANISM_COMMANDS]
    half_index = next(index for index, command in enumerate(commands) if "verify_half_recall_trace_survival.py" in command)
    fair_index = next(index for index, command in enumerate(commands) if "run_half_recall_fair_training.py" in command)
    fair_verify_index = next(index for index, command in enumerate(commands) if "verify_half_recall_fair_training.py" in command)
    refiner_index = next(index for index, command in enumerate(commands) if "run_recall_refiner.py" in command)
    refiner_verify_index = next(index for index, command in enumerate(commands) if "verify_recall_refiner.py" in command)

    assert half_index < fair_index < fair_verify_index < refiner_index < refiner_verify_index


def test_mechanism_gate_verifies_recall_mechanisms() -> None:
    commands = [" ".join(command) for command in quality_gate.MECHANISM_COMMANDS]
    assert any("verify_recall_scaling_law.py" in command for command in commands)
    assert any("verify_recall_recognition_modes.py" in command for command in commands)
    assert any("verify_layered_recall_trajectory.py" in command for command in commands)
    assert any("verify_visual_field_concat.py" in command for command in commands)
    assert any("verify_visual_scan_superresolution.py" in command for command in commands)


def test_quick_gate_protects_progressive_usage_surface() -> None:
    commands = [" ".join(command) for command in quality_gate.QUICK_COMMANDS]
    pytest_command = commands[0]

    assert "tests/test_usage_api.py" in pytest_command
    assert "tests/test_inspection.py" in pytest_command
    assert "tests/test_feature_matrix.py" in pytest_command
    assert "tests/test_layered_recall.py" in pytest_command
    assert "tests/test_layered_recall_benchmark.py" in pytest_command
    assert "tests/test_qwen_layered_recall_protocol.py" in pytest_command
    assert "tests/test_qwen_layered_recall_v2_protocol.py" in pytest_command
    assert "tests/test_recall_topology.py" in pytest_command
    assert "tests/test_recall_topology_confirmation.py" in pytest_command
    assert "tests/test_recall_topology_evidence.py" in pytest_command


def test_quick_gate_protects_unified_attachment_surface() -> None:
    commands = [" ".join(command) for command in quality_gate.QUICK_COMMANDS]
    assert any("tests/test_attachment.py" in command for command in commands)
    assert any("tests/test_qwen_unified_attachment_protocol.py" in command for command in commands)
    assert any("tests/test_attachment_training.py" in command for command in commands)
    assert any("tests/test_qwen_unified_training_protocol.py" in command for command in commands)
    assert any("tests/test_attachment_hub.py" in command for command in commands)
    assert any("tests/test_qwen_hub_lifecycle_protocol.py" in command for command in commands)
    assert any("tests/test_recall_scaling_screen.py" in command for command in commands)
    assert any("tests/test_recall_scaling_evidence.py" in command for command in commands)


def test_qwen_gate_runs_learned_pulse_probe_after_runtime_pulse() -> None:
    commands = [" ".join(command) for command in quality_gate.QWEN_COMMANDS]
    runtime_index = next(index for index, command in enumerate(commands) if "qwen_runtime_vocab_pulse_semantic_results.json" in command)
    learned_index = next(index for index, command in enumerate(commands) if "run_qwen_learned_pulse_adapter.py" in command)
    learned_verify_index = next(index for index, command in enumerate(commands) if "verify_qwen_learned_pulse_adapter.py" in command)

    assert runtime_index < learned_index < learned_verify_index


def test_mechanism_gate_verifies_pulse_refiner_repair() -> None:
    commands = [" ".join(command) for command in quality_gate.MECHANISM_COMMANDS]
    learned_verify_index = next(index for index, command in enumerate(commands) if "verify_learned_pulse_compaction.py" in command)
    repair_verify_index = next(index for index, command in enumerate(commands) if "verify_pulse_refiner_repair.py" in command)

    assert learned_verify_index < repair_verify_index


def test_cuda_gate_regenerates_pulse_refiner_repair() -> None:
    commands = [" ".join(command) for command in quality_gate.CUDA_COMMANDS]
    run_index = next(index for index, command in enumerate(commands) if "run_pulse_refiner_repair.py" in command)
    verify_index = next(index for index, command in enumerate(commands) if "verify_pulse_refiner_repair.py" in command)

    assert "--device cuda" in commands[run_index]
    assert "benchmarks/results/pulse_refiner_repair_results.json" in commands[run_index]
    assert run_index < verify_index


def test_cuda_gate_regenerates_pulse_refiner_multiseed() -> None:
    commands = [" ".join(command) for command in quality_gate.CUDA_COMMANDS]
    run_index = next(index for index, command in enumerate(commands) if "run_pulse_refiner_multiseed.py" in command)
    verify_index = next(index for index, command in enumerate(commands) if "verify_pulse_refiner_multiseed.py" in command)

    assert "--device cuda" in commands[run_index]
    assert "benchmarks/results/pulse_refiner_multiseed_results.json" in commands[run_index]
    assert run_index < verify_index


def test_qwen_gate_verifies_string_recall_refiner_answering() -> None:
    commands = [" ".join(command) for command in quality_gate.QWEN_COMMANDS]
    learned_verify_index = next(index for index, command in enumerate(commands) if "verify_qwen_learned_pulse_adapter.py" in command)
    string_verify_index = next(index for index, command in enumerate(commands) if "verify_qwen_string_recall_refiner_answering.py" in command)
    multiseed_verify_index = next(index for index, command in enumerate(commands) if "verify_qwen_string_pulse_multiseed.py" in command)

    assert learned_verify_index < string_verify_index < multiseed_verify_index


def test_mainline_gate_covers_qwen_and_recall_evidence() -> None:
    commands = [" ".join(command) for command in quality_gate.MAINLINE_COMMANDS]

    assert any("verify_qwen_hidden_refiner_half.py" in command for command in commands)
    assert any("verify_core_goal.py" in command for command in commands)
    assert any("verify_evidence_schema.py --skip-gates" in command for command in commands)


def test_quick_gate_covers_qwen_string_verifier_tests() -> None:
    commands = [" ".join(command) for command in quality_gate.QUICK_COMMANDS]

    assert any("tests/test_qwen_string_recall_refiner_answering.py" in command for command in commands)
    assert any("tests/test_qwen_string_pulse_multiseed.py" in command for command in commands)


def test_pretrained_gate_runs_contracts_real_models_and_verifier() -> None:
    commands = [" ".join(command) for command in quality_gate.PRETRAINED_COMMANDS]

    assert "tests/test_pretrained_workflow.py" in commands[0]
    assert "tests/test_pretrained_distributed_smoke.py" in commands[0]
    assert "run_pretrained_distributed_gate.py" in commands[1]
    assert "verify_pretrained_distributed_smoke.py" in commands[2]
    assert "run_pretrained_ecosystem_smoke.py" in commands[3]
    assert "verify_pretrained_ecosystem_smoke.py" in commands[4]


def test_quick_gate_covers_pulse_refiner_repair_verifier_tests() -> None:
    commands = [" ".join(command) for command in quality_gate.QUICK_COMMANDS]

    assert any("tests/test_pulse_refiner_repair.py" in command for command in commands)
    assert any("tests/test_pulse_refiner_multiseed.py" in command for command in commands)


def test_quick_gate_covers_latent_repair_goal_tests() -> None:
    commands = [" ".join(command) for command in quality_gate.QUICK_COMMANDS]

    assert any("tests/test_latent_repair_goal.py" in command for command in commands)


def test_quick_gate_covers_half_fair_training_tests() -> None:
    commands = [" ".join(command) for command in quality_gate.QUICK_COMMANDS]

    assert any("tests/test_half_recall_fair_training.py" in command for command in commands)


def test_quick_gate_covers_core_goal_tests() -> None:
    commands = [" ".join(command) for command in quality_gate.QUICK_COMMANDS]

    assert any("tests/test_core_goal.py" in command for command in commands)


def test_all_gate_includes_mainline_before_heavy_mechanism_and_cuda() -> None:
    commands = [" ".join(command) for command in quality_gate.commands_for(["all"])]
    mainline_index = next(index for index, command in enumerate(commands) if "verify_core_goal.py" in command)
    mechanism_index = next(index for index, command in enumerate(commands) if "run_half_recall_trace_survival.py" in command)
    cuda_index = next(index for index, command in enumerate(commands) if "profile_scaling.py" in command)

    assert mainline_index < mechanism_index < cuda_index
