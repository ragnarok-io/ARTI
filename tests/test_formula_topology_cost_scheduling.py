from __future__ import annotations

import importlib.util
import hashlib
import inspect
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"


def load_module(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, BENCHMARKS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def test_all_non_nsight_sidecars_share_one_process_but_not_one_workload(
    tmp_path: Path, monkeypatch
) -> None:
    target = load_module(
        "formula_topology_v3_cost_target_test",
        "formula_topology_same_executor_v3_cost_target.py",
    )
    calls: list[tuple[str, str, int, int]] = []

    def attributed(arm: str, batch_size: int, seed: int) -> dict[str, object]:
        calls.append(("attributed", arm, batch_size, seed))
        return {"kind": "attributed", "arm": arm}

    def latency(arm: str, batch_size: int, seed: int) -> dict[str, object]:
        calls.append(("latency", arm, batch_size, seed))
        return {"kind": "latency", "arm": arm}

    monkeypatch.setattr(target, "profile_attributed", attributed)
    monkeypatch.setattr(target, "profile_latency", latency)
    target.profile_all_sidecars(tmp_path, batch_size=128, seed=19031)

    assert calls == [
        (kind, arm, 128, 19031) for arm in target.ARMS for kind in ("attributed", "latency")
    ]
    assert sorted(path.name for path in tmp_path.iterdir()) == sorted(
        [f"{kind}-{arm}.json" for arm in target.ARMS for kind in ("attributed", "latency")]
    )
    for arm in target.ARMS:
        assert json.loads((tmp_path / f"attributed-{arm}.json").read_text(encoding="utf-8")) == {
            "arm": arm,
            "kind": "attributed",
        }
        assert json.loads((tmp_path / f"latency-{arm}.json").read_text(encoding="utf-8")) == {
            "arm": arm,
            "kind": "latency",
        }


def test_all_arm_process_map_uses_actual_child_receipt_pid(tmp_path: Path, monkeypatch) -> None:
    target = load_module(
        "formula_topology_v3_cost_target_pid_test",
        "formula_topology_same_executor_v3_cost_target.py",
    )
    launched: list[int] = []

    class FakeProcess:
        returncode = 0

        def __init__(self, command, **kwargs) -> None:
            del kwargs
            self.pid = 100 + len(launched)
            actual_pid = 900 + len(launched)
            launched.append(actual_pid)
            receipt_path = Path(command[command.index("--nvtx-receipt") + 1])
            receipt_path.write_text(json.dumps({"process_id": actual_pid}), encoding="utf-8")

        def communicate(self):
            return "", ""

    monkeypatch.setattr(target.subprocess, "Popen", FakeProcess)
    map_path = tmp_path / "ncu-process-map.json"
    target.execute_all_arm_children(map_path, batch_size=128, seed=19031)

    process_map = json.loads(map_path.read_text(encoding="utf-8"))
    assert [item["process_id"] for item in process_map["processes"]] == launched
    assert launched == [900, 901, 902]


def test_collector_uses_one_batched_sidecar_launch_and_remaining_deadline() -> None:
    collector = load_module(
        "formula_topology_v3_cost_collector_test",
        "profile_formula_topology_same_executor_v3_cost.py",
    )
    source = inspect.getsource(collector.collect)
    assert source.count('"--sidecars-dir"') == 1
    assert '"--attributed-json"' not in source
    assert '"--latency-json"' not in source
    assert source.count("timeout=remaining_timeout()") == 2


def test_orchestrator_does_not_predivide_the_physical_stage_deadline() -> None:
    orchestrator = load_module(
        "formula_topology_v3_gate_test",
        "run_formula_topology_same_executor_v3_gate.py",
    )
    source = inspect.getsource(orchestrator.main)
    assert "remaining_preflight / 4.0" not in source
    assert "remaining_preflight - 2.0" in source
    assert source.index("validate_cost(") < source.index("finalized_preflight_seconds")


def test_finalization_deadline_revokes_canonical_preflight(tmp_path: Path, monkeypatch) -> None:
    orchestrator = load_module(
        "formula_topology_v3_gate_finalization_test",
        "run_formula_topology_same_executor_v3_gate.py",
    )
    ticks = iter((119.9, 120.1))
    monkeypatch.setattr(
        orchestrator,
        "atomic_json",
        lambda path, value: path.write_text(json.dumps(value), encoding="utf-8"),
    )
    receipt = {"valid": True, "cost_preflight_passed": True}
    path = tmp_path / "preflight.json"

    elapsed = orchestrator._finalize_preflight(
        path, receipt, started=0.0, cap=120.0, clock=lambda: next(ticks)
    )

    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert elapsed == 120.1
    assert persisted["valid"] is False
    assert persisted["cost_preflight_passed"] is False
    assert "finalization budget" in persisted["revocation_reason"]


def test_seed_runner_rejects_preflight_denied_by_sibling_manifest(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    runner = load_module(
        "formula_topology_v3_seed_manifest_test",
        "run_formula_topology_same_executor_v3_seed.py",
    )
    provenance = {
        "formula_program_sha256": runner.FORMULA_PROGRAM_SHA256,
        "task_transform_sha256": runner.tensor_sha256(runner.TASK_TRANSFORM),
    }
    monkeypatch.setattr(runner, "_current_static_bindings", lambda: {})
    monkeypatch.setitem(
        runner.ARTIFACT_CONTRACT,
        "provenance_required_keys",
        sorted(provenance),
    )
    preflight = {
        "format": runner.ARTIFACT_CONTRACT["formats"]["preflight"],
        "valid": True,
        "mode": "formal",
        "cost_preflight_passed": True,
        "counter_status": "AVAILABLE",
        "orchestrator_capability_sha256": hashlib.sha256(b"live-capability").hexdigest(),
        "provenance": provenance,
    }
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(preflight), encoding="utf-8")
    (tmp_path / "run-manifest.json").write_text(
        json.dumps({"decision": "INVALID_COST_PROFILE"}), encoding="utf-8"
    )

    monkeypatch.setenv("ARTI_FORMULA_GATE_CAPABILITY", "live-capability")
    with pytest.raises(RuntimeError, match="denied by its run manifest"):
        runner.load_preflight(path, development=False)


def test_persisted_passed_receipt_is_not_an_execution_authority(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    runner = load_module(
        "formula_topology_v3_seed_ephemeral_capability_test",
        "run_formula_topology_same_executor_v3_seed.py",
    )
    provenance = {
        "formula_program_sha256": runner.FORMULA_PROGRAM_SHA256,
        "task_transform_sha256": runner.tensor_sha256(runner.TASK_TRANSFORM),
    }
    monkeypatch.setattr(runner, "_current_static_bindings", lambda: {})
    monkeypatch.setitem(
        runner.ARTIFACT_CONTRACT,
        "provenance_required_keys",
        sorted(provenance),
    )
    receipt = {
        "format": runner.ARTIFACT_CONTRACT["formats"]["preflight"],
        "valid": True,
        "mode": "formal",
        "cost_preflight_passed": True,
        "counter_status": "AVAILABLE",
        "orchestrator_capability_sha256": hashlib.sha256(b"now-lost-capability").hexdigest(),
        "provenance": provenance,
    }
    path = tmp_path / "preflight.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    monkeypatch.delenv("ARTI_FORMULA_GATE_CAPABILITY", raising=False)

    with pytest.raises(RuntimeError, match="live orchestrator capability"):
        runner.load_preflight(path, development=False)

    monkeypatch.setenv("ARTI_FORMULA_GATE_CAPABILITY", "wrong-capability")
    with pytest.raises(RuntimeError, match="live orchestrator capability"):
        runner.load_preflight(path, development=True)


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
@pytest.mark.parametrize(
    ("module_name", "filename", "function_name"),
    [
        (
            "formula_topology_v3_cost_collector_job_test",
            "profile_formula_topology_same_executor_v3_cost.py",
            "_run",
        ),
        (
            "formula_topology_v3_gate_job_test",
            "run_formula_topology_same_executor_v3_gate.py",
            "run_with_timeout",
        ),
    ],
)
def test_timeout_closes_the_complete_windows_process_job(
    module_name: str, filename: str, function_name: str
) -> None:
    module = load_module(module_name, filename)
    command = (
        sys.executable,
        "-c",
        (
            "import subprocess,sys,time;"
            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);"
            "print(p.pid,flush=True);time.sleep(30)"
        ),
    )
    with pytest.raises(subprocess.TimeoutExpired) as caught:
        getattr(module, function_name)(command, timeout=0.75)
    output = caught.value.output or ""
    if isinstance(output, bytes):
        output = output.decode("utf-8", errors="replace")
    match = re.search(r"\b(\d+)\b", output)
    assert match is not None
    child_pid = int(match.group(1))
    time.sleep(0.1)
    tasklist = subprocess.run(
        ("tasklist", "/FI", f"PID eq {child_pid}", "/FO", "CSV", "/NH"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert tasklist.returncode == 0
    assert str(child_pid) not in tasklist.stdout


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_ncu_diagnostic_timeout_is_receipted_and_cleans_children(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    diagnostic = load_module(
        "formula_topology_v3_ncu_diagnostic_timeout_test",
        "diagnose_formula_topology_v3_ncu.py",
    )
    command = (
        sys.executable,
        "-c",
        (
            "import subprocess,sys,time;"
            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);"
            "print(p.pid,flush=True);time.sleep(30)"
        ),
    )
    receipt = diagnostic.run_bounded(command, timeout=0.75, output_dir=tmp_path, name="timeout")
    assert receipt["cause"] == "TIMEOUT"
    assert receipt["cleanup_confirmed"] is True
    output = (tmp_path / "timeout.stdout.txt").read_text(encoding="utf-8")
    match = re.search(r"\b(\d+)\b", output)
    assert match is not None
    child_pid = int(match.group(1))
    tasklist = subprocess.run(
        ("tasklist", "/FI", f"PID eq {child_pid}", "/FO", "CSV", "/NH"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert str(child_pid) not in tasklist.stdout


@pytest.mark.skipif(os.name != "nt", reason="Windows Job Object contract")
def test_v4_dynamic_runner_timeout_is_bounded_and_cleans_children(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    diagnostic = load_module(
        "formula_topology_v4_runner_timeout_test",
        "diagnose_formula_topology_v4_dynamic.py",
    )
    command = (
        sys.executable,
        "-c",
        (
            "import subprocess,sys,time;"
            "p=subprocess.Popen([sys.executable,'-c','import time;time.sleep(30)']);"
            "print(p.pid,flush=True);time.sleep(30)"
        ),
    )
    started = time.perf_counter()
    receipt = diagnostic.run_bounded(
        command,
        timeout=0.75,
        cleanup_timeout=2.0,
        output_dir=tmp_path,
        name="v4-timeout",
    )
    assert time.perf_counter() - started < 3.5
    assert receipt["cause"] == "TIMEOUT"
    assert receipt["cleanup_confirmed"] is True
    assert receipt["job_ownership_established"] is True
    assert receipt["job_close_failed"] is False
    assert receipt["fallback_tree_kill_succeeded"] is False
    output = (tmp_path / "v4-timeout.stdout.txt").read_text(encoding="utf-8")
    match = re.search(r"\b(\d+)\b", output)
    assert match is not None
    child_pid = int(match.group(1))
    tasklist = subprocess.run(
        ("tasklist", "/FI", f"PID eq {child_pid}", "/FO", "CSV", "/NH"),
        capture_output=True,
        text=True,
        check=False,
    )
    assert str(child_pid) not in tasklist.stdout


def test_v4_dynamic_runner_job_setup_failure_has_only_bounded_waits(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    diagnostic = load_module(
        "formula_topology_v4_runner_setup_failure_test",
        "diagnose_formula_topology_v4_dynamic.py",
    )
    observed: list[tuple[str, float | None]] = []

    class FakeProcess:
        pid = 73
        returncode = None

        def wait(self, timeout=None):
            observed.append(("wait", timeout))
            self.returncode = -9
            return self.returncode

        def poll(self):
            return self.returncode

        def kill(self):
            observed.append(("kill", None))
            self.returncode = -9

    monkeypatch.setattr(diagnostic.subprocess, "Popen", lambda *args, **kwargs: FakeProcess())
    monkeypatch.setattr(
        diagnostic,
        "WindowsKillJob",
        lambda process: (_ for _ in ()).throw(RuntimeError("job setup")),
    )

    def bounded_taskkill(*args, timeout=None, **kwargs):
        observed.append(("taskkill", timeout))
        return subprocess.CompletedProcess(args[0], 0, "", "")

    monkeypatch.setattr(diagnostic.subprocess, "run", bounded_taskkill)
    receipt = diagnostic.run_bounded(
        ("fake",),
        timeout=1.0,
        cleanup_timeout=2.0,
        output_dir=tmp_path,
        name="setup-failure",
    )
    assert receipt["cause"] == "JOB_SETUP_FAILURE"
    assert receipt["cleanup_confirmed"] is True
    assert receipt["job_ownership_established"] is False
    assert receipt["fallback_tree_kill_succeeded"] is True
    assert observed
    assert all(timeout is not None and timeout <= 2.0 for name, timeout in observed if name != "kill")

    monkeypatch.setattr(
        diagnostic.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 1, "", "failed"),
    )
    failed_receipt = diagnostic.run_bounded(
        ("fake",),
        timeout=1.0,
        cleanup_timeout=2.0,
        output_dir=tmp_path,
        name="setup-failure-nonzero",
    )
    assert failed_receipt["cleanup_confirmed"] is False
    assert failed_receipt["job_ownership_established"] is False
    assert failed_receipt["fallback_tree_kill_succeeded"] is False

    def timed_out_taskkill(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs.get("timeout", 0.0))

    monkeypatch.setattr(diagnostic.subprocess, "run", timed_out_taskkill)
    timeout_receipt = diagnostic.run_bounded(
        ("fake",),
        timeout=1.0,
        cleanup_timeout=2.0,
        output_dir=tmp_path,
        name="setup-failure-timeout",
    )
    assert timeout_receipt["cleanup_confirmed"] is False
    assert timeout_receipt["job_ownership_established"] is False
    assert timeout_receipt["fallback_tree_kill_succeeded"] is False


def test_ncu_diagnostic_reconstructs_completed_prefix_and_isolates_paths(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    diagnostic = load_module(
        "formula_topology_v3_ncu_diagnostic_prefix_test",
        "diagnose_formula_topology_v3_ncu.py",
    )
    for index, arm in enumerate(("dynamic", "static")):
        path = tmp_path / f"ncu-child-{arm}.json"
        argv = [
            "--arm",
            arm,
            "--batch-size",
            "128",
            "--seed",
            "19031",
            "--nvtx-step",
            "--nvtx-receipt",
            str(path.resolve()),
        ]
        path.write_text(
            json.dumps(
                {
                    "format": "arti.formula-topology-same-executor-ncu-child.v3",
                    "process_id": index + 1,
                    "arm": arm,
                    "batch_size": 128,
                    "seed": 19031,
                    "scope": "complete_forward_backward_gradient_postprocess_optimizer_step",
                    "argv_sha256": hashlib.sha256(
                        json.dumps(argv, ensure_ascii=True, separators=(",", ":")).encode()
                    ).hexdigest(),
                    "workload_identity": {
                        "format": "arti.formula-topology-same-executor-cost-workload.v3",
                        "arm": arm,
                        "batch_size": 128,
                        "depth": 2,
                        "device_type": "cuda",
                        "dtype": "torch.float32",
                        "episode": {
                            "batch_index": 0,
                            "keys_sha256": "1" * 64,
                            "mask_sha256": "2" * 64,
                            "payload_sha256": "3" * 64,
                            "role_index_sha256": "4" * 64,
                            "role_queries_sha256": "5" * 64,
                            "slot_permutation_sha256": "6" * 64,
                            "split": "cost",
                            "stream_seeds": {"keys": 1},
                        },
                        "formula_scale_sha256": "7" * 64,
                        "initial_model_state_sha256": "8" * 64,
                        "initial_optimizer_state_sha256": "9" * 64,
                        "optimizer_config_sha256": "a" * 64,
                        "target_sha256": "b" * 64,
                    },
                    "completed": True,
                }
            ),
            encoding="utf-8",
        )
    assert diagnostic.completed_prefix(tmp_path) == ["dynamic", "static"]
    command = diagnostic.ncu_command(Path("ncu.BAT"), tmp_path / "single-dynamic", arm="dynamic")
    assert str(tmp_path / "single-dynamic" / "single-dynamic.csv") in command
    assert str(tmp_path / "single-dynamic" / "single-dynamic-child.json") in command


def test_ncu_diagnostic_requires_every_pid_metric_pair(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    diagnostic = load_module(
        "formula_topology_v3_ncu_diagnostic_metric_test",
        "diagnose_formula_topology_v3_ncu.py",
    )
    path = tmp_path / "incomplete.csv"
    path.write_text(
        '"ID","Process ID","Kernel Name","Metric Name","Metric Unit","Metric Value"\n'
        '"0","11","kernel","dram__bytes_op_read.sum","byte","10"\n'
        '"1","12","kernel","dram__bytes_op_write.sum","byte","10"\n',
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="required PID/metric rows"):
        diagnostic.validate_ncu_csv(path, {11, 12})


def test_v4_dynamic_heartbeat_chain_is_strict_and_tamper_evident(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    target = load_module(
        "formula_topology_v4_heartbeat_target_test",
        "formula_topology_same_executor_v3_cost_target.py",
    )
    diagnostic = load_module(
        "formula_topology_v4_heartbeat_diagnostic_test",
        "diagnose_formula_topology_v4_dynamic.py",
    )
    previous = None
    identity = "a" * 64
    for sequence, phase in enumerate(diagnostic.PHASES):
        previous = target._heartbeat(
            tmp_path,
            sequence=sequence,
            phase=phase,
            process_id=73,
            identity_sha256=None if sequence == 0 else identity,
            previous_sha256=previous,
        )
    chain = diagnostic.load_heartbeat_chain(tmp_path)
    assert [entry["phase"] for entry in chain] == list(diagnostic.PHASES)
    assert {entry["process_id"] for entry in chain} == {73}

    tampered = tmp_path / "heartbeat-04-forward_completed.json"
    value = json.loads(tampered.read_text(encoding="utf-8"))
    value["phase"] = "BACKWARD_COMPLETED"
    tampered.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid heartbeat chain"):
        diagnostic.load_heartbeat_chain(tmp_path)


def test_v4_heartbeat_extension_preserves_v3_no_heartbeat_operator_order(
    monkeypatch,
) -> None:
    target = load_module(
        "formula_topology_v4_legacy_order_test",
        "formula_topology_same_executor_v3_cost_target.py",
    )

    class Workload:
        def __init__(self) -> None:
            self.module = torch.nn.Linear(1, 1, bias=False)
            self.optimizer = torch.optim.SGD(self.module.parameters(), lr=0.1)
            self.events: list[str] = []
            original_step = self.optimizer.step

            def step(*args, **kwargs):
                self.events.append("optimizer")
                return original_step(*args, **kwargs)

            self.optimizer.step = step

        def forward_loss(self):
            self.events.append("forward")
            return self.module(torch.ones(1, 1)).sum()

        def postprocess_gradients(self):
            self.events.append("postprocess")

        def identity_receipt(self):
            return {"identity": "same"}

    reference = Workload()
    candidate = Workload()
    candidate.module.load_state_dict(reference.module.state_dict())
    target._complete_step(reference)

    monkeypatch.setattr(target, "_build", lambda arm, batch_size, seed: candidate)
    monkeypatch.setattr(target.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(target.torch.cuda.nvtx, "range_push", lambda name: None)
    monkeypatch.setattr(target.torch.cuda.nvtx, "range_pop", lambda: None)
    target.execute_nvtx_step("dynamic", 128, 19031)

    assert reference.events == candidate.events == [
        "forward",
        "postprocess",
        "optimizer",
    ]
    assert torch.equal(reference.module.weight, candidate.module.weight)


def test_v4_dynamic_diagnostic_classifies_attach_and_phase_boundaries(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    target = load_module(
        "formula_topology_v4_classification_target_test",
        "formula_topology_same_executor_v3_cost_target.py",
    )
    diagnostic = load_module(
        "formula_topology_v4_classification_diagnostic_test",
        "diagnose_formula_topology_v4_dynamic.py",
    )
    receipt = {"cleanup_confirmed": True, "cause": "TIMEOUT", "returncode": 1}

    no_attach = tmp_path / "no-attach"
    (no_attach / "heartbeats").mkdir(parents=True)
    (no_attach / "ncu-dynamic.stderr.txt").write_text("timeout", encoding="utf-8")
    assert diagnostic.classify_ncu(no_attach, receipt, plain_identity_sha256="b" * 64) == (
        "NCU_ATTACH_EVIDENCE_MISSING",
        None,
    )

    attached = tmp_path / "attached"
    heartbeats = attached / "heartbeats"
    heartbeats.mkdir(parents=True)
    (attached / "ncu-dynamic.stderr.txt").write_text("Connected to process 73", encoding="utf-8")
    previous = target._heartbeat(
        heartbeats,
        sequence=0,
        phase=diagnostic.PHASES[0],
        process_id=73,
        identity_sha256=None,
        previous_sha256=None,
    )
    target._heartbeat(
        heartbeats,
        sequence=1,
        phase=diagnostic.PHASES[1],
        process_id=73,
        identity_sha256="b" * 64,
        previous_sha256=previous,
    )
    assert diagnostic.classify_ncu(attached, receipt, plain_identity_sha256="b" * 64) == (
        "NCU_PROFILE_OR_REPLAY_TIMEOUT",
        "WORKLOAD_IDENTITY_FROZEN",
    )

    (attached / "ncu-dynamic.stderr.txt").write_text("Connected to process 999", encoding="utf-8")
    assert diagnostic.classify_ncu(attached, receipt, plain_identity_sha256="b" * 64) == (
        "NCU_ATTACH_PID_MISMATCH",
        "WORKLOAD_IDENTITY_FROZEN",
    )

    (attached / "ncu-dynamic.stderr.txt").write_text("Connected to process 73", encoding="utf-8")
    assert diagnostic.classify_ncu(attached, receipt, plain_identity_sha256="c" * 64) == (
        "WORKLOAD_IDENTITY_DRIFT",
        "WORKLOAD_IDENTITY_FROZEN",
    )

    (attached / "ncu-dynamic.stderr.txt").write_text(
        "Connected to process 73\nConnected to process 74", encoding="utf-8"
    )
    assert diagnostic.classify_ncu(attached, receipt, plain_identity_sha256="b" * 64) == (
        "NCU_ATTACH_PID_AMBIGUOUS",
        "WORKLOAD_IDENTITY_FROZEN",
    )

    process_failure = {
        "cleanup_confirmed": True,
        "cause": "COMPLETED",
        "returncode": 9,
    }
    assert diagnostic.classify_ncu(attached, process_failure, plain_identity_sha256="b" * 64) == (
        "NCU_PROCESS_FAILURE",
        "WORKLOAD_IDENTITY_FROZEN",
    )


def test_v4_dynamic_diagnostic_stage_timeout_uses_shared_deadline(
    monkeypatch,
) -> None:
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    diagnostic = load_module(
        "formula_topology_v4_deadline_test",
        "diagnose_formula_topology_v4_dynamic.py",
    )
    assert (
        diagnostic.remaining_stage_timeout(requested=420.0, process_deadline=465.0, now=44.0)
        == 420.0
    )
    assert (
        diagnostic.remaining_stage_timeout(requested=420.0, process_deadline=465.0, now=60.0)
        == 405.0
    )
    assert (
        diagnostic.remaining_stage_timeout(requested=420.0, process_deadline=465.0, now=466.0)
        == 0.0
    )
    assert (
        diagnostic.deadline_classification(
            "DYNAMIC_NCU_CHARACTERIZABLE",
            now=540.01,
            global_deadline=540.0,
            finalization_duration=20.0,
            finalization_budget=45.0,
        )
        == "INVALID_DIAGNOSTIC_TIMEOUT"
    )


def test_v4_dynamic_diagnostic_prereg_rejects_schema_and_budget_drift(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.syspath_prepend(str(BENCHMARKS))
    diagnostic = load_module(
        "formula_topology_v4_prereg_test",
        "diagnose_formula_topology_v4_dynamic.py",
    )
    prereg = json.loads(
        (BENCHMARKS / "formula_topology_v4_dynamic_diagnostic_prereg.json").read_text(
            encoding="utf-8"
        )
    )
    path = tmp_path / "prereg.json"
    diagnostic.PREREG = path

    prereg["unexpected"] = True
    path.write_text(json.dumps(prereg), encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid v4 dynamic diagnostic"):
        diagnostic.load_prereg()

    prereg.pop("unexpected")
    prereg["budgets_seconds"]["ncu_dynamic"] = 421
    path.write_text(json.dumps(prereg), encoding="utf-8")
    with pytest.raises(RuntimeError, match="invalid v4 dynamic diagnostic"):
        diagnostic.load_prereg()
    assert (
        diagnostic.deadline_classification(
            "DYNAMIC_NCU_CHARACTERIZABLE",
            now=500.0,
            global_deadline=540.0,
            finalization_duration=45.01,
            finalization_budget=45.0,
        )
        == "INVALID_DIAGNOSTIC_TIMEOUT"
    )
