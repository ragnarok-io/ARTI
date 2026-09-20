from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path

import pytest

from benchmarks._formula_topology_v4_arm_contract import (
    build_c0_binding,
    build_lineage,
    build_workload_identity_document,
    canonical_digest,
    canonical_json,
    load_contract_documents,
)
from benchmarks.verify_formula_topology_v4_arm_characterization import (
    ArmVerificationError,
    verify_arm_artifact,
)


ROOT = Path(__file__).resolve().parents[1]
HERE = ROOT / "benchmarks"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value), encoding="utf-8", newline="")


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def _write_sums(path: Path, values: dict[str, str]) -> None:
    path.write_text(
        "".join(f"{values[name]}  {name}\n" for name in sorted(values)),
        encoding="ascii",
        newline="",
    )


def _authority() -> dict[str, bool]:
    return {
        "formal_authorization": False,
        "formal_seeds_consumed": False,
        "reusable_as_preflight": False,
        "scientific_score": False,
    }


def _c0_binding(environment: dict[str, object]) -> dict[str, object]:
    documents = load_contract_documents()
    binding = documents.prereg["calibration_binding"]
    completion = {
        "classification": binding["completion_classification"],
        "completed": True,
        "formal_authorization": False,
        "formal_seeds_consumed": False,
        "reusable_as_preflight": False,
        "scientific_score": False,
        "prereg_sha256": binding["prereg_sha256"],
        "worker_manifest_sha256": binding["manifest_sha256"],
    }
    metric_names = {**binding["primary_metrics"], **binding["secondary_metrics"]}
    units = {**binding["primary_units"], **binding["secondary_units"]}
    metrics = {
        logical: {"metric": metric, "unit": units[logical], "sum": 1.0}
        for logical, metric in metric_names.items()
    }
    manifest = {
        "classification": binding["manifest_classification"],
        "formal_authorization": False,
        "formal_seeds_consumed": False,
        "reusable_as_preflight": False,
        "scientific_score": False,
        "candidate_metrics": metric_names,
        "primary_metrics": {"read": "dram_read", "write": "dram_write"},
        "gpu_name": environment["gpu"]["name"],
        "compute_capability": environment["gpu"]["compute_capability"],
        "cuda_runtime": environment["cuda"]["runtime"],
        "torch": environment["torch"],
        "ncu_launcher_sha256": environment["ncu"]["launcher_sha256"],
        "ncu_executable_sha256": environment["ncu"]["executable_sha256"],
        "observations": {
            "first": {"metrics": copy.deepcopy(metrics)},
            "second": {"metrics": copy.deepcopy(metrics)},
        },
    }
    hashes = {
        "prereg": binding["prereg_sha256"],
        "completion": binding["completion_sha256"],
        "manifest": binding["manifest_sha256"],
        "outer_sha256s": binding["outer_sha256s_sha256"],
        "worker_sha256s": binding["worker_sha256s_sha256"],
    }
    return build_c0_binding(completion=completion, manifest=manifest, artifact_hashes=hashes)


def _workload_identity(environment: dict[str, object], arm: str) -> dict[str, object]:
    return {
        "arm": arm,
        "batch_size": 128,
        "seed": 19031,
        "depth": 2,
        "formula_scale": 1.0,
        "dtype": "torch.float32",
        "device": "cuda:0",
        "episode_sha256": "0" * 64,
        "target_sha256": "1" * 64,
        "task_transform_sha256": "2" * 64,
        "formula_contract_sha256": "3" * 64,
        "topology_contract_sha256": "4" * 64,
        "initial_model_state_sha256": "5" * 64,
        "initial_optimizer_state_sha256": "6" * 64,
        "optimizer_config_sha256": "7" * 64,
        "parameter_layout_sha256": "8" * 64,
        "optimizer_layout_sha256": "9" * 64,
        "operator_graph_sha256": "a" * 64,
        "source_identity_sha256": "b" * 64,
        "environment_identity_sha256": canonical_digest(environment),
    }


def _receipt(
    name: str,
    argv: list[str],
    stdout: str,
    stderr: str,
    bundle: Path,
    *,
    timeout: float,
) -> dict[str, object]:
    return {
        "format": "arti.bounded-command-receipt.v1",
        "name": name,
        "argv": argv,
        "argv_sha256": canonical_digest(argv),
        "host_pid": 321,
        "started_utc_seconds": 1.0,
        "duration_seconds": 1.0,
        "timeout_seconds": timeout,
        "cleanup_timeout_seconds": 2.0,
        "cause": "COMPLETED",
        "returncode": 0,
        "job_ownership_established": True,
        "job_close_failed": False,
        "fallback_tree_kill_succeeded": False,
        "cleanup_confirmed": True,
        "stdout_file": Path(stdout).name,
        "stdout_sha256": _sha256(bundle / stdout),
        "stderr_file": Path(stderr).name,
        "stderr_sha256": _sha256(bundle / stderr),
    }


def _seal_bundle(bundle: Path) -> None:
    documents = load_contract_documents()
    contract = documents.artifact_contract
    artifact = bundle / "artifact"
    manifest_path = artifact / "manifest.json"
    manifest = _read_json(manifest_path)
    direct = {
        "c0_binding_sha256": "c0-binding.json",
        "identity_sha256": "identity.json",
        "environment_sha256": "environment.json",
        "semantic_probes_sha256": "semantic-probes.json",
    }
    for field, relative in direct.items():
        manifest[field] = _sha256(artifact / relative)
    manifest["lineage_sha256"] = _sha256(bundle / "lineage.json")
    evidence = []
    for relative in sorted(set(contract["artifact_files"]) - {"manifest.json"}):
        evidence.append({"path": relative, "sha256": _sha256(artifact / relative)})
    manifest["evidence"] = evidence
    _write_json(manifest_path, manifest)

    completion_path = bundle / "completion.json"
    completion = _read_json(completion_path)
    completion["worker_receipt_sha256"] = _sha256(bundle / "arm-worker.command.json")
    completion["worker_manifest_sha256"] = _sha256(manifest_path)
    completion["lineage_sha256"] = _sha256(bundle / "lineage.json")
    _write_json(completion_path, completion)

    inner = {
        relative: _sha256(artifact / relative) for relative in contract["artifact_files"]
    }
    _write_sums(artifact / "SHA256SUMS.txt", inner)
    outer_paths = set(contract["outer_root_files"]) | {contract["artifact_hash_file"]}
    outer = {relative: _sha256(bundle / relative) for relative in outer_paths}
    _write_sums(bundle / "SHA256SUMS.txt", outer)


def _valid_bundle(tmp_path: Path, arm: str = "sham") -> Path:
    bundle = tmp_path / f"arm-{arm}"
    artifact = bundle / "artifact"
    documents = load_contract_documents()
    prereg = documents.prereg
    phase = prereg["budgets_seconds"]["phase_limits"]
    for relative in documents.artifact_contract["artifact_files"]:
        path = artifact / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix not in {".json"}:
            path.write_bytes(b"")
    for relative in documents.artifact_contract["outer_root_files"]:
        path = bundle / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix != ".json":
            path.write_bytes(b"")

    environment = {
        "format": "arti.formula-topology-arm-environment.v1",
        "git": {"head": "c" * 40, "checkout_signature": "d" * 64},
        "python": "3.12.0",
        "torch": "2.11.0+cu128",
        "cuda": {"torch_cuda": "12.8", "runtime": "12.8"},
        "gpu": {
            "name": "test gpu",
            "uuid": "GPU-test",
            "pci_bus_id": "0000:01:00.0",
            "driver": "999.0",
            "compute_capability": [12, 0],
        },
        "ncu": {
            "version": "2026.1.1",
            "launcher_path": "C:/ncu.bat",
            "launcher_sha256": "e" * 64,
            "executable_path": "C:/ncu.exe",
            "executable_sha256": "f" * 64,
        },
    }
    identity = build_workload_identity_document(
        _workload_identity(environment, arm),
        attempt_index=1,
        arm_semantics_sha256=canonical_digest(prereg["arms"][arm]),
    )
    lineage = build_lineage(
        arm=arm,
        attempt_index=1,
        same_arm_predecessor=None,
        locked_prior_arms=[],
    )
    _write_json(artifact / "environment.json", environment)
    _write_json(artifact / "identity.json", identity)
    _write_json(artifact / "c0-binding.json", _c0_binding(environment))
    _write_json(bundle / "lineage.json", lineage)

    identity_sha = canonical_digest(identity)
    _write_json(
        artifact / "attributed/attributed.json",
        {
            "format": "arti.formula-topology-arm-attributed.v1",
            "arm": arm,
            "forward_flops": 100,
            "backward_flops": 200,
            "operator_signature_sha256": "1" * 64,
            "optimizer_parameter_elements": 10,
            "optimizer_parameter_tensors": 2,
            "optimizer_state_tensor_count": 4,
            "identity_sha256": identity_sha,
        },
    )
    _write_json(
        artifact / "latency/latency.json",
        {
            "format": "arti.formula-topology-arm-latency.v1",
            "arm": arm,
            "cold_host_seconds": 0.1,
            "cold_cuda_seconds": 0.08,
            "warm_host_seconds": [0.05] * 50,
            "warm_cuda_seconds": [0.04] * 50,
            "warmup_steps": 10,
            "snapshot_restores": 50,
            "peak_allocated_memory_bytes": 1024,
            "peak_reserved_memory_bytes": 2048,
            "identity_sha256": identity_sha,
        },
    )
    process_id = 777
    _write_json(
        artifact / "ncu/ncu-target.json",
        {
            "format": "arti.formula-topology-arm-ncu-target.v1",
            "arm": arm,
            "process_id": process_id,
            "completed": True,
            "identity_sha256": identity_sha,
            "loss_sha256": "2" * 64,
        },
    )
    selected = "static" if arm == "static" else "dynamic"
    multiplier = 0.0 if arm == "sham" else 1.0
    dynamic_hash = "4" * 64
    static_hash = "5" * 64
    selected_hash = static_hash if selected == "static" else dynamic_hash
    before_gradient_hash = "6" * 64
    after_gradient_hash = "7" * 64 if arm == "sham" else before_gradient_hash
    _write_json(
        artifact / "semantic-probes.json",
        {
            "format": "arti.formula-topology-arm-semantic-probes.v1",
            "arm": arm,
            "both_priority_paths_executed": True,
            "selected_priority": selected,
            "selected_priority_sha256": selected_hash,
            "dynamic_priority_sha256": dynamic_hash,
            "static_priority_sha256": static_hash,
            "gradient_multiplier": multiplier,
            "pre_postprocess_nonzero_count": 4,
            "postprocess_nonzero_count": 0 if arm == "sham" else 4,
            "pre_postprocess_gradient_sha256": before_gradient_hash,
            "postprocess_gradient_sha256": after_gradient_hash,
            "all_gradients_materialized": True,
            "optimizer_state_materialized": True,
            "static_input_invariance": True if arm == "static" else None,
            "dynamic_sham_forward_pair_sha256": "3" * 64,
            "passed": True,
        },
    )

    report = artifact / "ncu/profile.ncu-rep"
    report.write_bytes(b"raw-ncu-report")
    (artifact / "ncu/capture.csv").write_text(
        f"==PROF== Connected to process {process_id}\n"
        f"==PROF== Disconnected from process {process_id}\n"
        f"==PROF== Report: {report.resolve()}\n",
        encoding="utf-8",
    )
    metrics = prereg["measurement"]["ncu_command_contract"]["metrics"]
    report_rows = []
    for index, phase_name in enumerate(
        ("FORWARD", "BACKWARD", "GRADIENT_POSTPROCESS", "OPTIMIZER"), start=1
    ):
        report_rows.append(
            ",".join(
                [
                    str(process_id),
                    str(index),
                    f"test_kernel_{index}",
                    f"ARTI_FORMULA_TOPOLOGY_V3_COST/{phase_name}",
                    "10",
                    "11",
                    "12",
                    "13",
                ]
            )
        )
    (artifact / "ncu/report-derived.csv").write_text(
        ",".join(["Process ID", "ID", "Kernel Name", "NVTX Push/Pop_Range", *metrics])
        + "\n"
        + ",".join(["", "", "", "", "byte", "byte", "sector", "sector"])
        + "\n"
        + "\n".join(report_rows)
        + "\n",
        encoding="utf-8",
    )

    def target_argv(mode: str, output: Path) -> list[str]:
        return [
            "python",
            str(HERE / "formula_topology_v4_arm_characterization_target.py"),
            "--mode",
            mode,
            "--arm",
            arm,
            "--output",
            str(output),
            "--attempt-index",
            "1",
            "--environment-identity-sha256",
            canonical_digest(environment),
            "--source-identity-sha256",
            identity["full_workload_identity"]["source_identity_sha256"],
        ]

    ncu_argv = [
        "ncu",
        "--replay-mode",
        "kernel",
        "--nvtx",
        "--nvtx-include",
        "ARTI_FORMULA_TOPOLOGY_V3_COST/",
        "--metrics",
        ",".join(metrics),
        "--csv",
        "--log-file",
        str(artifact / "ncu/capture.csv"),
        "--export",
        str(artifact / "ncu/profile"),
        *target_argv("ncu", artifact / "ncu/ncu-target.json"),
    ]
    import_argv = [
        "ncu",
        "--import",
        str(report),
        "--csv",
        "--page",
        "raw",
        "--print-units",
        "base",
        "--log-file",
        str(artifact / "ncu/report-derived.csv"),
    ]
    receipts = {
        "artifact/attributed/attributed.command.json": (
            "attributed",
            target_argv("attributed", artifact / "attributed/attributed.json"),
            "artifact/attributed/attributed.stdout.txt",
            "artifact/attributed/attributed.stderr.txt",
            phase["attributed_profile"],
        ),
        "artifact/latency/latency.command.json": (
            "latency",
            target_argv("latency", artifact / "latency/latency.json"),
            "artifact/latency/latency.stdout.txt",
            "artifact/latency/latency.stderr.txt",
            phase["live_latency_and_memory"],
        ),
        "artifact/ncu/ncu.command.json": (
            "ncu",
            ncu_argv,
            "artifact/ncu/ncu.stdout.txt",
            "artifact/ncu/ncu.stderr.txt",
            phase["ncu_capture"],
        ),
        "artifact/ncu/report-import.command.json": (
            "report-import",
            import_argv,
            "artifact/ncu/report-import.stdout.txt",
            "artifact/ncu/report-import.stderr.txt",
            phase["report_import"],
        ),
        "arm-worker.command.json": (
            "arm-worker",
            ["python", "worker.py", "--arm", arm],
            "arm-worker.stdout.txt",
            "arm-worker.stderr.txt",
            prereg["budgets_seconds"]["deadlines_from_host_start"]["cleanup"],
        ),
    }
    for relative, (name, argv, stdout, stderr, timeout) in receipts.items():
        _write_json(
            bundle / relative,
            _receipt(name, argv, stdout, stderr, bundle, timeout=timeout),
        )

    source_hashes = {
        "prereg_sha256": _sha256(
            HERE / "formula_topology_v4_arm_characterization_prereg.json"
        ),
        "design_sha256": _sha256(HERE / "formula_topology_v4_arm_characterization_design.md"),
        "artifact_contract_sha256": _sha256(
            HERE / "formula_topology_v4_arm_characterization_artifact_contract.json"
        ),
        "schema_sha256": _sha256(
            HERE / "formula_topology_v4_arm_characterization_schema.json"
        ),
    }
    _write_json(
        artifact / "manifest.json",
        {
            "format": "arti.formula-topology-arm-characterization.v1",
            "classification": "VALID_CHARACTERIZED",
            "arm": arm,
            "attempt_index": 1,
            "authority": _authority(),
            **source_hashes,
            "c0_binding_sha256": "0" * 64,
            "identity_sha256": "0" * 64,
            "environment_sha256": "0" * 64,
            "semantic_probes_sha256": "0" * 64,
            "lineage_sha256": "0" * 64,
            "evidence": [{"path": "identity.json", "sha256": "0" * 64}],
            "duration_seconds": 10.0,
        },
    )
    _write_json(
        bundle / "completion.json",
        {
            "format": "arti.formula-topology-arm-characterization-completion.v1",
            "completed": True,
            "classification": "VALID_CHARACTERIZED",
            "arm": arm,
            "attempt_index": 1,
            "authority": _authority(),
            "host_limit_seconds": 540,
            "host_duration_seconds": 20.0,
            "worker_receipt_sha256": "0" * 64,
            "worker_manifest_sha256": "0" * 64,
            "lineage_sha256": "0" * 64,
            **source_hashes,
        },
    )
    _seal_bundle(bundle)
    return bundle


def test_verifier_accepts_one_complete_bundle_without_cross_arm_ratio(tmp_path: Path) -> None:
    report = verify_arm_artifact(_valid_bundle(tmp_path), verify_current_environment=False)
    assert report["verified"] is True
    assert report["arm"] == "sham"
    assert report["cross_arm_ratios_computed"] is False
    assert report["ncu_kernel_launches"] == 4
    assert report["ncu_metric_totals"]["dram__bytes_op_read.sum"] == 40.0


def test_verifier_rejects_extra_file(tmp_path: Path) -> None:
    bundle = _valid_bundle(tmp_path)
    (bundle / "unexpected.txt").write_text("leak", encoding="utf-8")
    with pytest.raises(ArmVerificationError, match="allowlist"):
        verify_arm_artifact(bundle, verify_current_environment=False)


def test_verifier_rejects_payload_tampering_before_trusting_json(tmp_path: Path) -> None:
    bundle = _valid_bundle(tmp_path)
    (bundle / "artifact/latency/latency.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ArmVerificationError, match="SHA-256 mismatch"):
        verify_arm_artifact(bundle, verify_current_environment=False)


def test_verifier_recomputes_arm_semantics_and_ignores_passed_as_proof(tmp_path: Path) -> None:
    bundle = _valid_bundle(tmp_path)
    probes_path = bundle / "artifact/semantic-probes.json"
    probes = _read_json(probes_path)
    probes["selected_priority"] = "static"
    probes["passed"] = True
    _write_json(probes_path, probes)
    _seal_bundle(bundle)
    with pytest.raises(ArmVerificationError, match="contradict"):
        verify_arm_artifact(bundle, verify_current_environment=False)


def test_verifier_rejects_rehashed_c0_projection_tampering(tmp_path: Path) -> None:
    bundle = _valid_bundle(tmp_path)
    path = bundle / "artifact/c0-binding.json"
    value = _read_json(path)
    value["projection"]["observed_units"]["dram_read"] = "sector"
    value["canonical_projection_sha256"] = canonical_digest(value["projection"])
    _write_json(path, value)
    _seal_bundle(bundle)
    with pytest.raises(ArmVerificationError, match="c0Binding|C0"):
        verify_arm_artifact(bundle, verify_current_environment=False)


def test_verifier_rejects_rehashed_identity_projection_tampering(tmp_path: Path) -> None:
    bundle = _valid_bundle(tmp_path)
    path = bundle / "artifact/identity.json"
    value = _read_json(path)
    value["shared_workload_identity"]["seed"] = 7
    value["shared_workload_identity_sha256"] = canonical_digest(
        value["shared_workload_identity"]
    )
    _write_json(path, value)
    identity_sha = canonical_digest(value)
    for relative in (
        "artifact/attributed/attributed.json",
        "artifact/latency/latency.json",
        "artifact/ncu/ncu-target.json",
    ):
        child = _read_json(bundle / relative)
        child["identity_sha256"] = identity_sha
        _write_json(bundle / relative, child)
    _seal_bundle(bundle)
    with pytest.raises(ArmVerificationError, match="shared_workload|shared workload"):
        verify_arm_artifact(bundle, verify_current_environment=False)


def test_verifier_rejects_success_after_global_deadline(tmp_path: Path) -> None:
    bundle = _valid_bundle(tmp_path)
    path = bundle / "completion.json"
    value = _read_json(path)
    value["host_duration_seconds"] = 540.001
    _write_json(path, value)
    _seal_bundle(bundle)
    with pytest.raises(ArmVerificationError, match="global host deadline"):
        verify_arm_artifact(bundle, verify_current_environment=False)


def test_verifier_rejects_unclean_command_even_when_rehashed(tmp_path: Path) -> None:
    bundle = _valid_bundle(tmp_path)
    path = bundle / "artifact/ncu/ncu.command.json"
    value = _read_json(path)
    value["cleanup_confirmed"] = False
    _write_json(path, value)
    _seal_bundle(bundle)
    with pytest.raises(ArmVerificationError, match="lifecycle"):
        verify_arm_artifact(bundle, verify_current_environment=False)


def test_verifier_rejects_missing_ncu_metric_column(tmp_path: Path) -> None:
    bundle = _valid_bundle(tmp_path)
    path = bundle / "artifact/ncu/report-derived.csv"
    rows = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(",".join(row.split(",")[:-1]) for row in rows) + "\n")
    _seal_bundle(bundle)
    with pytest.raises(ArmVerificationError, match="metric|lts__|header"):
        verify_arm_artifact(bundle, verify_current_environment=False)


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="symlink API unavailable")
def test_verifier_rejects_redirected_artifact(tmp_path: Path) -> None:
    bundle = _valid_bundle(tmp_path)
    path = bundle / "artifact/ncu/ncu.stdout.txt"
    target = tmp_path / "outside.txt"
    target.write_text("outside", encoding="utf-8")
    path.unlink()
    try:
        path.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is not permitted")
    with pytest.raises(ArmVerificationError, match="redirected"):
        verify_arm_artifact(bundle, verify_current_environment=False)
