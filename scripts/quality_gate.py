"""Run public software gates without private data or benchmark dependencies."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
DEFAULT_OUTPUT = ROOT / ".tmp" / "quality_gate_report.json"


def pytest_command(name, *files):
    return [PY, "-m", "pytest", "--basetemp=.tmp/quality-" + name, *files]


QUICK_COMMANDS = [pytest_command(
    "public", "tests/test_package_metadata.py", "tests/test_backend_namespaces.py",
    "tests/test_component_registry.py", "tests/test_lifecycle_contract.py",
    "tests/test_component_graph.py", "tests/test_execution_contracts.py",
    "tests/test_public_execution_vocabulary.py", "tests/test_arti_st.py",
    "tests/test_serialization.py", "tests/test_runtime_checkpoint.py",
    "tests/test_generated_docs.py", "tests/test_release_security.py",
    "tests/test_release_readiness.py", "tests/test_quality_gate.py",
)]
MAINLINE_COMMANDS = [pytest_command(
    "programs", "tests/test_resource_graph.py", "tests/test_program_graph_nodes.py",
    "tests/test_program_graph_artifact.py", "tests/test_branch_search_compilation.py",
    "tests/test_federal_contracts.py", "tests/test_federal_recall.py",
    "tests/test_federal_compiler.py", "tests/test_federal_stateful_graph.py",
    "tests/test_formula_fabric.py", "tests/test_formula_v2.py",
    "tests/test_credit_boundary.py", "tests/test_learning_regions.py",
    "tests/test_formula_position.py", "tests/test_tensor_binding.py",
    "tests/test_tensor_transaction.py", "tests/test_tensor_operation.py",
    "tests/test_reversible_topology.py", "tests/test_fold_module.py",
    "tests/test_unfold_module.py", "tests/test_adaptive_observation.py",
    "tests/test_observation_optional.py", "tests/test_rcc_host.py",
)]
RECIPE_COMMANDS = [pytest_command(
    "recipes", "tests/test_task_recipes.py", "tests/test_language_recipe.py",
    "tests/test_recipe_serving.py", "tests/test_shared_training.py",
    "tests/test_training_run.py", "tests/test_attachment_training.py",
    "tests/test_attachment_training_engines.py", "tests/test_pretrained_workflow.py",
    "tests/test_attachment_hub.py",
)]
PRETRAINED_COMMANDS = [pytest_command("pretrained", "tests/test_pretrained_workflow.py")]
JAX_COMMANDS = [
    pytest_command("jax", "tests/test_jax_contract.py", "tests/test_jax_torch_parity.py",
                   "tests/test_jax_formula.py", "tests/test_jax_program.py"),
    [PY, "examples/jax_native_training.py"],
    [PY, "-m", "arti.cli", "doctor", "--allow-cpu-torch", "--require-jax-smoke"],
]
DOCS_COMMANDS = [
    [PY, "-m", "arti.cli", "docs", "check"],
    [PY, "-m", "arti.cli", "schema", "fit-config", "check"],
    [PY, "-m", "arti.cli", "schema", "task-graph", "check"],
    [PY, "-m", "mkdocs", "build", "--strict"],
]
PACKAGE_COMMANDS = [
    [PY, "scripts/check_lifecycle_contract.py"],
    [PY, "scripts/check_release_readiness.py"],
    [PY, "scripts/check_package.py"],
]
CUDA_COMMANDS = [[PY, "-m", "arti.cli", "doctor", "--require-cuda-smoke"]]
GATES = {
    "quick": QUICK_COMMANDS, "mainline": MAINLINE_COMMANDS, "recipes": RECIPE_COMMANDS,
    "pretrained": PRETRAINED_COMMANDS, "jax": JAX_COMMANDS, "docs": DOCS_COMMANDS,
    "package": PACKAGE_COMMANDS, "mechanism": MAINLINE_COMMANDS,
    "qwen": RECIPE_COMMANDS, "cuda": CUDA_COMMANDS,
}


def display_command(command):
    return " ".join(["python", *command[1:]])


def portable_output_tail(value):
    for path, replacement in ((str(ROOT), "."), (str(Path.home()), "~")):
        value = value.replace(path.replace("\\", "\\\\"), replacement).replace(path, replacement)
    return value[-4000:]


def commands_for(gates):
    names = ["quick", "mainline", "recipes", "docs", "package"] if "all" in gates else gates
    return [command for name in names for command in GATES[name]]


def run_command(command):
    runtime = [
        f"{part}-{os.getpid()}-{time.time_ns()}" if part.startswith("--basetemp=") else part
        for part in command
    ]
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    print("RUN " + display_command(command), flush=True)
    started = time.perf_counter()
    result = subprocess.run(runtime, cwd=ROOT, text=True, capture_output=True, env=env)
    row = {
        "command": display_command(command), "returncode": result.returncode,
        "seconds": time.perf_counter() - started,
        "stdout_tail": portable_output_tail(result.stdout),
        "stderr_tail": portable_output_tail(result.stderr),
    }
    print(f"{row['command']}: exit={result.returncode} ({row['seconds']:.2f}s)", flush=True)
    if result.returncode:
        print(row["stdout_tail"])
        print(row["stderr_tail"])
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gates", nargs="+", choices=[*GATES, "all"])
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reuse-passing-producers", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    rows = []
    for command in commands_for(args.gates):
        rows.append(run_command(command))
        if rows[-1]["returncode"] and args.fail_fast:
            break
    payload = {"gates": args.gates, "passed": all(row["returncode"] == 0 for row in rows),
               "runs": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(
        "# Public Quality Gate\n\n" + "\n".join(
            f"- `{row['command']}`: exit {row['returncode']}" for row in rows
        ) + "\n", encoding="utf-8",
    )
    print(json.dumps({"passed": payload["passed"], "output": str(args.output)}))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
