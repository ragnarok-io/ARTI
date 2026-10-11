from pathlib import Path

from scripts import quality_gate as gate


def test_public_gates_only_reference_existing_public_files():
    for commands in gate.GATES.values():
        for command in commands:
            assert not any("benchmarks/" in part for part in command)
            for part in command:
                if part.endswith(".py"):
                    assert (gate.ROOT / part).is_file(), part


def test_recipe_gate_contains_real_task_lifecycle_and_service_regressions():
    assert {
        "tests/test_task_recipes.py", "tests/test_language_recipe.py",
        "tests/test_recipe_serving.py", "tests/test_training_run.py",
        "tests/test_shared_training.py",
    } <= set(gate.RECIPE_COMMANDS[0])


def test_jax_gate_includes_native_bindings_and_training_example():
    assert "tests/test_jax_program.py" in gate.JAX_COMMANDS[0]
    assert "tests/test_jax_formula.py" in gate.JAX_COMMANDS[0]
    assert gate.JAX_COMMANDS[1] == [gate.PY, "examples/jax_native_training.py"]


def test_default_all_never_launches_gpu_training():
    commands = gate.commands_for(["all"])
    assert gate.CUDA_COMMANDS[0] not in commands
    assert all("run_qwen" not in " ".join(command) for command in commands)
    assert gate.LIFECYCLE_COMMANDS[0] in commands
    assert gate.INTEGRATION_COMMANDS[1] not in commands


def test_lifecycle_gate_includes_native_recovery_outputs_and_installed_user_contracts():
    assert {
        "tests/test_custom_task_workflow.py", "tests/test_recipe_assets.py",
        "tests/test_prediction_files.py", "tests/test_recipe_episode.py",
        "tests/test_recipe_suite.py", "tests/test_recipe_runtime.py",
        "tests/test_runtime_backend.py", "tests/test_recipe_export.py",
        "tests/test_file_lifecycle_workflow.py", "tests/test_task_project.py",
        "tests/test_task_project_cli.py", "tests/test_task_scaffold.py",
        "tests/test_task_status.py", "tests/test_training_validation.py",
        "tests/test_atomic_checkpoint_write.py",
    } <= set(gate.LIFECYCLE_COMMANDS[0])


def test_docs_gate_checks_every_published_native_workflow_schema():
    from arti.task_project import SCHEMA_KINDS

    checked = {
        command[-1] for command in gate.DOCS_COMMANDS
        if command[1:7] == ["-m", "arti.cli", "schema", "task-workflow", "check", "--kind"]
    }
    assert checked == set(SCHEMA_KINDS)


def test_optional_cpu_sdk_gate_runs_real_tracker_and_native_trainer_cases():
    assert {
        "tests/test_transformers_training.py", "tests/test_trl_training.py",
        "tests/test_ray_training.py", "tests/test_reporting.py",
    } <= set(gate.INTEGRATION_COMMANDS[1])
    assert "'skipped'" in gate.INTEGRATION_COMMANDS[-1][-1]


def test_gate_reports_do_not_embed_interpreter_or_user_path():
    assert gate.display_command([gate.PY, "scripts/check_package.py"]) == "python scripts/check_package.py"
    assert gate.portable_output_tail(str(gate.ROOT) + "/run") == "./run"
    assert gate.portable_output_tail(str(Path.home()) + "/cache") == "~/cache"


def test_gate_creates_temporary_parent_in_a_fresh_checkout(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "ROOT", tmp_path)
    assert not (tmp_path / ".tmp").exists()
    row = gate.run_command([
        gate.PY, "-c", "from pathlib import Path; assert Path('.tmp').is_dir()",
    ])
    assert row["returncode"] == 0
