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


def test_gate_reports_do_not_embed_interpreter_or_user_path():
    assert gate.display_command([gate.PY, "scripts/check_package.py"]) == "python scripts/check_package.py"
    assert gate.portable_output_tail(str(gate.ROOT) + "/run") == "./run"
    assert gate.portable_output_tail(str(Path.home()) + "/cache") == "~/cache"
