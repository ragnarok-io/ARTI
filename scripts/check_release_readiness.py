"""Check whether the local tree has alpha-release evidence attached."""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


REQUIRED_GATE_REPORTS = {
    "quick": ROOT / "benchmarks" / "results" / "quality_gate_quick.json",
    "mainline": ROOT / "benchmarks" / "results" / "quality_gate_mainline.json",
    "docs": ROOT / "benchmarks" / "results" / "quality_gate_docs.json",
    "package": ROOT / "benchmarks" / "results" / "quality_gate_package.json",
    "pretrained": ROOT / "benchmarks" / "results" / "quality_gate_pretrained.json",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def project_version() -> str:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', pyproject, flags=re.MULTILINE)
    if not match:
        raise ValueError("project version not found in pyproject.toml")
    return match.group(1)


def package_version() -> str:
    version_file = (ROOT / "src" / "arti" / "_version.py").read_text(encoding="utf-8")
    match = re.search(r'^__version__ = "([^"]+)"$', version_file, flags=re.MULTILINE)
    if not match:
        raise ValueError("package version not found in src/arti/_version.py")
    return match.group(1)


def check_gate_reports(require_cuda: bool = False, require_mechanism: bool = False, require_qwen: bool = False) -> list[str]:
    failures = []
    for name, path in REQUIRED_GATE_REPORTS.items():
        if not path.exists():
            failures.append(f"missing {name} gate report: {path.relative_to(ROOT)}")
            continue
        payload = load_json(path)
        if payload.get("passed") is not True:
            failures.append(f"{name} gate report is not passing")
        if name == "mainline":
            failures.extend(check_mainline_gate_report(payload))
        if name == "pretrained":
            failures.extend(check_pretrained_gate_report(payload))
    if require_cuda:
        cuda_path = ROOT / "benchmarks" / "results" / "quality_gate_cuda.json"
        if not cuda_path.exists():
            failures.append("missing cuda gate report")
        elif load_json(cuda_path).get("passed") is not True:
            failures.append("cuda gate report is not passing")
    if require_mechanism:
        mechanism_path = ROOT / "benchmarks" / "results" / "quality_gate_mechanism.json"
        if not mechanism_path.exists():
            failures.append("missing mechanism gate report")
        elif load_json(mechanism_path).get("passed") is not True:
            failures.append("mechanism gate report is not passing")
    if require_qwen:
        qwen_path = ROOT / "benchmarks" / "results" / "quality_gate_qwen.json"
        if not qwen_path.exists():
            failures.append("missing qwen gate report")
        else:
            qwen_payload = load_json(qwen_path)
            if qwen_payload.get("passed") is not True:
                failures.append("qwen gate report is not passing")
            failures.extend(check_qwen_gate_report(qwen_payload))
    return failures


def check_mainline_gate_report(payload: dict) -> list[str]:
    runs = payload.get("runs", [])
    commands = [str(row.get("command", "")) for row in runs if isinstance(row, dict)]
    quality_gate = load_module("quality_gate_for_mainline_release_readiness", ROOT / "scripts" / "quality_gate.py")
    failures = []
    if payload.get("gates") != ["mainline"]:
        failures.append("mainline gate report must have gates ['mainline']")
    if len(runs) != len(quality_gate.MAINLINE_COMMANDS):
        failures.append(
            f"mainline gate report run count {len(runs)} does not match current MAINLINE_COMMANDS {len(quality_gate.MAINLINE_COMMANDS)}"
        )
    for index, row in enumerate(runs):
        if not isinstance(row, dict):
            failures.append(f"mainline gate report run {index} is not an object")
            continue
        if row.get("returncode") != 0:
            failures.append(f"mainline gate report run {index} did not pass")
    for expected in quality_gate.MAINLINE_COMMANDS:
        expected_tail = " ".join(str(part) for part in expected[1:])
        if not any(expected_tail in command for command in commands):
            failures.append(f"mainline gate report missing current MAINLINE_COMMANDS entry: {expected_tail}")
    return failures


def check_pretrained_gate_report(payload: dict) -> list[str]:
    runs = payload.get("runs", [])
    commands = [str(row.get("command", "")) for row in runs if isinstance(row, dict)]
    quality_gate = load_module("quality_gate_for_pretrained_release_readiness", ROOT / "scripts" / "quality_gate.py")
    failures = []
    if payload.get("gates") != ["pretrained"]:
        failures.append("pretrained gate report must have gates ['pretrained']")
    if len(runs) != len(quality_gate.PRETRAINED_COMMANDS):
        failures.append(
            f"pretrained gate report run count {len(runs)} does not match current PRETRAINED_COMMANDS {len(quality_gate.PRETRAINED_COMMANDS)}"
        )
    for index, row in enumerate(runs):
        if not isinstance(row, dict) or row.get("returncode") != 0:
            failures.append(f"pretrained gate report run {index} did not pass")
    for expected in quality_gate.PRETRAINED_COMMANDS:
        expected_tail = " ".join(str(part) for part in expected[1:])
        if not any(expected_tail in command for command in commands):
            failures.append(f"pretrained gate report missing current PRETRAINED_COMMANDS entry: {expected_tail}")
    return failures


def check_claim_boundaries(require_cuda: bool = False, require_mechanism: bool = False, require_qwen: bool = False) -> list[str]:
    failures = []
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    release = (ROOT / "docs" / "guides" / "release.md").read_text(encoding="utf-8")
    combined = "\n".join([readme, changelog, release])
    required_phrases = [
        "alpha",
        "Public API is not yet frozen",
        "not broad downstream superiority",
        "mainline gate",
        "full JAX parity",
        "CUDA gate",
    ]
    for phrase in required_phrases:
        if phrase not in combined:
            failures.append(f"release boundary text missing phrase: {phrase}")
    forbidden_phrases = [
        "Nature-level validation is complete",
        "broad downstream superiority is supported",
        "Full JAX parity is implemented",
    ]
    for phrase in forbidden_phrases:
        if phrase in combined:
            failures.append(f"release boundary text contains forbidden phrase: {phrase}")
    if require_cuda:
        packet = ROOT / "benchmarks" / "results" / "cuda_evidence_packet.json"
        if not packet.exists():
            failures.append("CUDA is required but cuda_evidence_packet.json is missing")
        else:
            payload = load_json(packet)
            if payload.get("status") != "generated_local_cuda":
                failures.append("CUDA is required but cuda_evidence_packet status is not generated_local_cuda")
    if require_mechanism:
        mechanism_report = ROOT / "benchmarks" / "results" / "quality_gate_mechanism.json"
        if not mechanism_report.exists():
            failures.append("mechanism evidence is required but quality_gate_mechanism.json is missing")
    if require_qwen and "Qwen gate" not in combined:
        failures.append("Qwen evidence is required but release docs do not mention the Qwen gate")
    return failures


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def check_qwen_gate_report(payload: dict) -> list[str]:
    runs = payload.get("runs", [])
    commands = [str(row.get("command", "")) for row in runs if isinstance(row, dict)]
    quality_gate = load_module("quality_gate_for_release_readiness", ROOT / "scripts" / "quality_gate.py")
    required_snippets = [
        "qwen_runtime_vocab_pulse_semantic_results.json",
        "verify_qwen_runtime_vocab_pulse_semantic.py",
        "qwen_runtime_vocab_pulse_strict_surface_probe.json",
        "qwen_runtime_vocab_pulse_metadata_bridge_probe.json",
        "verify_qwen_runtime_vocab_bridge_ablation.py",
        "verify_qwen_runtime_vocab_metadata_bridge.py",
        "verify_qwen_dynamic_vocab_goal.py",
    ]
    failures = []
    if payload.get("gates") != ["qwen"]:
        failures.append("qwen gate report must have gates ['qwen']")
    if len(runs) != len(quality_gate.QWEN_COMMANDS):
        failures.append(f"qwen gate report run count {len(runs)} does not match current QWEN_COMMANDS {len(quality_gate.QWEN_COMMANDS)}")
    for index, row in enumerate(runs):
        if not isinstance(row, dict):
            failures.append(f"qwen gate report run {index} is not an object")
            continue
        if row.get("returncode") != 0:
            failures.append(f"qwen gate report run {index} did not pass")
    for snippet in required_snippets:
        if not any(snippet in command for command in commands):
            failures.append(f"qwen gate report missing command: {snippet}")
    for expected in quality_gate.QWEN_COMMANDS:
        expected_tail = " ".join(str(part) for part in expected[1:])
        if not any(expected_tail in command for command in commands):
            failures.append(f"qwen gate report missing current QWEN_COMMANDS entry: {expected_tail}")
    return failures


def load_qwen_goal_payloads() -> dict[str, dict]:
    return {
        "input_head": load_json(ROOT / "benchmarks" / "results" / "qwen_input_head_replacement_results.json"),
        "closed_loop": load_json(ROOT / "benchmarks" / "results" / "qwen_closed_loop_replacement_results.json"),
        "runtime_pulse": load_json(ROOT / "benchmarks" / "results" / "qwen_runtime_vocab_pulse_semantic_results.json"),
        "no_bridge": load_json(ROOT / "benchmarks" / "results" / "qwen_runtime_vocab_pulse_strict_surface_probe.json"),
        "metadata_bridge": load_json(ROOT / "benchmarks" / "results" / "qwen_runtime_vocab_pulse_metadata_bridge_probe.json"),
        "dialogue": load_json(ROOT / "benchmarks" / "results" / "qwen_oov_dialogue_preservation_results.json"),
        "string_answer": load_json(ROOT / "benchmarks" / "results" / "qwen_string_recall_refiner_answering_results.json"),
        "string_multiseed": load_json(ROOT / "benchmarks" / "results" / "qwen_string_pulse_multiseed_results.json"),
        "literal_output_context": load_json(ROOT / "benchmarks" / "results" / "qwen_literal_output_context_results.json"),
        "unseen_literal_transfer": load_json(ROOT / "benchmarks" / "results" / "qwen_unseen_literal_transfer_results.json"),
        "literal_segmentation_generation": load_json(ROOT / "benchmarks" / "results" / "qwen_literal_segmentation_generation_results.json"),
    }


def load_latent_repair_goal_payloads() -> dict[str, dict]:
    return {
        "half_trace": load_json(ROOT / "benchmarks" / "results" / "half_recall_trace_survival_results.json"),
        "half_training": load_json(ROOT / "benchmarks" / "results" / "half_recall_fair_training_results.json"),
        "recall_refiner": load_json(ROOT / "benchmarks" / "results" / "recall_refiner_results.json"),
        "pulse_repair": load_json(ROOT / "benchmarks" / "results" / "pulse_refiner_repair_results.json"),
        "pulse_multiseed": load_json(ROOT / "benchmarks" / "results" / "pulse_refiner_multiseed_results.json"),
    }


def check_qwen_evidence() -> list[str]:
    failures = []
    dynamic = load_module("verify_qwen_dynamic_vocab_goal", ROOT / "benchmarks" / "verify_qwen_dynamic_vocab_goal.py")
    bridge = load_module("verify_qwen_runtime_vocab_metadata_bridge", ROOT / "benchmarks" / "verify_qwen_runtime_vocab_metadata_bridge.py")
    dynamic_failures = dynamic.verify(**load_qwen_goal_payloads())
    failures.extend(f"qwen dynamic vocab evidence: {failure}" for failure in dynamic_failures)
    bridge_failures = bridge.verify(load_json(ROOT / "benchmarks" / "results" / "qwen_runtime_vocab_pulse_metadata_bridge_probe.json"))
    failures.extend(f"qwen metadata bridge evidence: {failure}" for failure in bridge_failures)
    return failures


def check_core_goal_evidence() -> list[str]:
    core = load_module("verify_core_goal", ROOT / "benchmarks" / "verify_core_goal.py")
    failures = core.verify(qwen=load_qwen_goal_payloads(), latent_repair=load_latent_repair_goal_payloads())
    return [f"core goal evidence: {failure}" for failure in failures]


def check_version_consistency() -> list[str]:
    failures = []
    project = project_version()
    package = package_version()
    if project != package:
        failures.append(f"version mismatch: pyproject={project}, package={package}")
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    if f"## {project} " not in changelog and f"## {project} -" not in changelog:
        failures.append(f"CHANGELOG.md does not mention version {project}")
    return failures


def check_backend_capabilities(require_cuda: bool = False) -> list[str]:
    path = ROOT / "benchmarks" / "results" / "backend_capabilities.json"
    if not path.exists():
        return ["missing backend_capabilities.json"]
    payload = load_json(path)
    failures = []
    if "torch" not in payload.get("available_backends", []):
        failures.append("backend capabilities do not list torch as available")
    if payload.get("jax_backend_status") not in {"available", "broken", "unavailable"}:
        failures.append("backend capabilities has invalid jax_backend_status")
    if payload.get("jax_smoke_status") not in {"passed", "skipped", "failed"}:
        failures.append("backend capabilities has invalid jax_smoke_status")
    jax_smoke = payload.get("jax_smoke")
    if not isinstance(jax_smoke, dict):
        failures.append("backend capabilities missing jax_smoke")
    elif jax_smoke.get("smoke_status") != payload.get("jax_smoke_status"):
        failures.append("backend capabilities jax_smoke_status does not match smoke report")
    if payload.get("jax_backend_status") == "available" and payload.get("jax_smoke_status") != "passed":
        failures.append("backend capabilities JAX backend is available but jax_smoke_status is not passed")
    if payload.get("jax_backend_status") == "unavailable" and payload.get("jax_smoke_status") != "skipped":
        failures.append("backend capabilities JAX backend is unavailable but jax_smoke_status is not skipped")
    if payload.get("jax_backend_status") == "broken" and payload.get("jax_smoke_status") != "failed":
        failures.append("backend capabilities JAX backend is broken but jax_smoke_status is not failed")
    if payload.get("gpu_readiness_level") not in {
        "cpu_only",
        "nvidia_hardware_detected_torch_cpu",
        "torch_cuda_runtime_available",
    }:
        failures.append("backend capabilities has invalid gpu_readiness_level")
    if payload.get("torch_cuda_smoke_status") not in {"passed", "skipped", "failed"}:
        failures.append("backend capabilities has invalid torch_cuda_smoke_status")
    smoke = payload.get("torch_cuda_smoke")
    if not isinstance(smoke, dict):
        failures.append("backend capabilities missing torch_cuda_smoke")
    elif smoke.get("smoke_status") != payload.get("torch_cuda_smoke_status"):
        failures.append("backend capabilities torch_cuda_smoke_status does not match smoke report")
    if require_cuda and payload.get("torch_cuda_smoke_status") != "passed":
        failures.append("CUDA is required but torch_cuda_smoke_status is not passed")
    return failures


def check_ci_workflow() -> list[str]:
    workflow = ROOT / ".github" / "workflows" / "ci.yml"
    if not workflow.exists():
        return ["missing CI workflow: .github/workflows/ci.yml"]

    text = workflow.read_text(encoding="utf-8")
    required_snippets = [
        "name: CI",
        "scripts/quality_gate.py quick --fail-fast",
        "scripts/quality_gate.py mainline --fail-fast",
        "scripts/quality_gate.py docs --fail-fast",
        "scripts/quality_gate.py package --fail-fast",
        "scripts/check_lifecycle_contract.py",
        "name: JAX optional backend",
        "uv sync --locked --extra dev --extra jax",
        "scripts/quality_gate.py jax --fail-fast",
        "name: Mechanism evidence",
        "scripts/quality_gate.py mechanism --fail-fast",
        "scripts/check_release_readiness.py --require-mechanism",
        "name: Pretrained provider contracts",
        "uv sync --extra dev --extra qwen --extra peft --extra sd",
        "tests/test_pretrained_workflow.py tests/test_pretrained_ecosystem_smoke.py tests/test_pretrained_distributed_smoke.py",
        "benchmarks/verify_pretrained_ecosystem_smoke.py",
    ]
    return [f"CI workflow missing required snippet: {snippet}" for snippet in required_snippets if snippet not in text]


def required_gate_names(*, require_cuda: bool = False, require_mechanism: bool = False, require_qwen: bool = False) -> list[str]:
    gates = set(REQUIRED_GATE_REPORTS)
    if require_cuda:
        gates.add("cuda")
    if require_mechanism:
        gates.add("mechanism")
    if require_qwen:
        gates.add("qwen")
    return sorted(gates)


def check_readiness(require_cuda: bool = False, require_mechanism: bool = False, require_qwen: bool = False) -> dict:
    failures = []
    failures.extend(check_version_consistency())
    failures.extend(check_gate_reports(require_cuda=require_cuda, require_mechanism=require_mechanism, require_qwen=require_qwen))
    failures.extend(check_claim_boundaries(require_cuda=require_cuda, require_mechanism=require_mechanism, require_qwen=require_qwen))
    failures.extend(check_backend_capabilities(require_cuda=require_cuda))
    failures.extend(check_core_goal_evidence())
    if require_qwen:
        failures.extend(check_qwen_evidence())
    failures.extend(check_ci_workflow())
    return {
        "ok": not failures,
        "kind": "release-readiness",
        "version": project_version(),
        "required_gates": required_gate_names(
            require_cuda=require_cuda,
            require_mechanism=require_mechanism,
            require_qwen=require_qwen,
        ),
        "ci_workflow": ".github/workflows/ci.yml",
        "require_cuda": require_cuda,
        "require_mechanism": require_mechanism,
        "require_qwen": require_qwen,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-cuda", action="store_true", help="Require local CUDA evidence in addition to alpha gates.")
    parser.add_argument("--require-mechanism", action="store_true", help="Require mechanism gate evidence in addition to alpha gates.")
    parser.add_argument("--require-qwen", action="store_true", help="Require Qwen dynamic runtime-vocab evidence in addition to alpha gates.")
    args = parser.parse_args()

    payload = check_readiness(require_cuda=args.require_cuda, require_mechanism=args.require_mechanism, require_qwen=args.require_qwen)
    print(json.dumps(payload, indent=2, sort_keys=True))
    raise SystemExit(0 if payload["ok"] else 1)


if __name__ == "__main__":
    main()
