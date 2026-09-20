from __future__ import annotations

from contextlib import contextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import weakref

import pytest
import torch
from torch import Tensor, nn


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"


def load_target():
    spec = importlib.util.spec_from_file_location(
        "formula_topology_v4_arm_target_test",
        BENCHMARKS / "formula_topology_v4_arm_characterization_target.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class FakeTopology:
    active_count = 2
    axis = 1
    contract_fingerprint = "topology"

    class Operator:
        @staticmethod
        def topology_contract():
            return {"operator": "fake"}

    operator = Operator()
    surrogate = Operator()


class FakeFormula:
    @staticmethod
    def composition_contract():
        return {"formula": "fake"}


class FakeModule(nn.Module):
    def __init__(self, starts: list[float]) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))
        self.starts = starts
        self.topology = FakeTopology()
        self.formula = FakeFormula()

    def operator_signature(self):
        return {"operator": "fake", "shape": []}


class FakeWorkload:
    def __init__(self, arm: str, starts: list[float]) -> None:
        self.arm = arm
        self.module = FakeModule(starts)
        self.optimizer = torch.optim.AdamW(
            self.module.parameters(), lr=0.1, weight_decay=0.0
        )
        self.starts = starts
        self.scale = torch.ones(())

    def forward_loss(self) -> Tensor:
        self.starts.append(float(self.module.weight.detach()))
        return (self.module.weight - 3.0).square()

    def postprocess_gradients(self) -> None:
        assert self.module.weight.grad is not None
        if self.arm == "sham":
            self.module.weight.grad.mul_(0)

    def identity_receipt(self):
        scale_header = json.dumps(
            {"dtype": str(self.scale.dtype), "shape": list(self.scale.shape)},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        scale_sha256 = hashlib.sha256(
            scale_header
            + b"\n"
            + self.scale.detach().reshape(-1).view(torch.uint8).numpy().tobytes()
        ).hexdigest()
        return {
            "arm": self.arm,
            "episode": {"payload_sha256": "1" * 64, "mask_sha256": "2" * 64},
            "target_sha256": "3" * 64,
            "initial_model_state_sha256": "4" * 64,
            "initial_optimizer_state_sha256": "5" * 64,
            "optimizer_config_sha256": "6" * 64,
            "depth": 2,
            "formula_scale_sha256": scale_sha256,
            "dtype": "torch.float32",
            "device_type": "cuda",
            "batch_size": 128,
        }


def fake_runner():
    class Runner:
        EXPERIMENT_CONTRACT = {"task_transform_sha256": "7" * 64}

    return Runner()


def configure_identity_mocks(monkeypatch: pytest.MonkeyPatch, target) -> None:
    monkeypatch.setattr(target, "_load_v3_runner", fake_runner)


IDENTITY_KWARGS = {
    "attempt_index": 1,
    "environment_identity_sha256": "9" * 64,
    "source_identity_sha256": "8" * 64,
}


def test_identity_persists_exact_arm_projection_and_digests(monkeypatch) -> None:
    target = load_target()
    configure_identity_mocks(monkeypatch, target)
    workload = FakeWorkload("dynamic", [])
    document = target._identity_document(
        workload,
        arm="dynamic",
        attempt_index=2,
        environment_identity_sha256="9" * 64,
        source_identity_sha256="8" * 64,
    )

    full = document["full_workload_identity"]
    shared = document["shared_workload_identity"]
    assert full["arm"] == "dynamic"
    assert shared == {key: value for key, value in full.items() if key != "arm"}
    assert document["full_workload_identity_sha256"] == target.canonical_sha256(full)
    assert document["shared_workload_identity_sha256"] == target.canonical_sha256(shared)
    assert document["attempt_index"] == 2
    prereg_semantic = target._load_arm_contract().load_contract_documents().prereg["arms"][
        "dynamic"
    ]
    assert document["arm_semantics_sha256"] == target.canonical_sha256(prereg_semantic)
    assert full["source_identity_sha256"] == "8" * 64


def test_latency_restores_same_snapshot_outside_each_timed_sample(monkeypatch) -> None:
    target = load_target()
    configure_identity_mocks(monkeypatch, target)
    workload_refs: list[weakref.ReferenceType[FakeWorkload]] = []
    starts: list[list[float]] = []

    def build(arm: str, batch_size: int, seed: int):
        assert batch_size == 128
        assert seed == 19031
        if len(workload_refs) == 4:
            assert all(reference() is None for reference in workload_refs)
        workload_starts: list[float] = []
        starts.append(workload_starts)
        workload = FakeWorkload(arm, workload_starts)
        workload_refs.append(weakref.ref(workload))
        return workload

    def timed(workload: FakeWorkload):
        target._complete_step(workload)
        return 0.01, 0.009

    monkeypatch.setattr(target, "_timed_complete_step", timed)
    monkeypatch.setattr(target, "_cuda_sync", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 1024)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 2048)

    result = target.run_latency(
        "dynamic",
        128,
        19031,
        **IDENTITY_KWARGS,
        build=build,
    )

    assert len(workload_refs) == 5
    assert starts[3] == [1.0] * 50
    assert len(starts[2]) == 10
    assert result["snapshot_restores"] == 50
    assert result["warm_host_seconds"] == [0.01] * 50
    assert result["warm_cuda_seconds"] == [0.009] * 50


def test_attributed_uses_fresh_real_workloads_and_exact_schema(monkeypatch) -> None:
    target = load_target()
    configure_identity_mocks(monkeypatch, target)
    workloads: list[FakeWorkload] = []

    def build(arm: str, batch_size: int, seed: int):
        workload = FakeWorkload(arm, [])
        workloads.append(workload)
        return workload

    monkeypatch.setattr(
        target,
        "_profile_flops",
        lambda workload, phase: 100 if phase == "forward" else 200,
    )

    def complete_signature(workload):
        target._complete_step(workload)
        return {"aten::mul": 4}

    monkeypatch.setattr(target, "_profile_complete_signature", complete_signature)
    result = target.run_attributed(
        "sham",
        128,
        19031,
        **IDENTITY_KWARGS,
        build=build,
    )

    assert len(workloads) == 4
    assert result["format"] == "arti.formula-topology-arm-attributed.v1"
    assert result["forward_flops"] == 100
    assert result["backward_flops"] == 200
    assert result["optimizer_state_tensor_count"] > 0
    assert set(result) == {
        "format",
        "arm",
        "forward_flops",
        "backward_flops",
        "operator_signature_sha256",
        "optimizer_parameter_elements",
        "optimizer_parameter_tensors",
        "optimizer_state_tensor_count",
        "identity_sha256",
    }


def test_ncu_runs_exact_nested_ranges_and_writes_exact_receipt(monkeypatch) -> None:
    target = load_target()
    configure_identity_mocks(monkeypatch, target)
    workload = FakeWorkload("dynamic", [])
    ranges: list[tuple[str, str]] = []

    @contextmanager
    def nvtx(name: str):
        ranges.append(("enter", name))
        try:
            yield
        finally:
            ranges.append(("exit", name))

    monkeypatch.setattr(target, "_nvtx_range", nvtx)
    monkeypatch.setattr(target, "_cuda_sync", lambda: None)
    monkeypatch.setattr(target.os, "getpid", lambda: 73)
    receipt = target.run_ncu(
        "dynamic",
        128,
        19031,
        **IDENTITY_KWARGS,
        build=lambda arm, batch_size, seed: workload,
    )

    assert ranges == [
        ("enter", target.NVTX_ROOT),
        ("enter", "FORWARD"),
        ("exit", "FORWARD"),
        ("enter", "BACKWARD"),
        ("exit", "BACKWARD"),
        ("enter", "GRADIENT_POSTPROCESS"),
        ("exit", "GRADIENT_POSTPROCESS"),
        ("enter", "OPTIMIZER"),
        ("exit", "OPTIMIZER"),
        ("exit", target.NVTX_ROOT),
    ]
    assert receipt["process_id"] == 73
    assert receipt["completed"] is True
    assert set(receipt) == {
        "format",
        "arm",
        "process_id",
        "completed",
        "identity_sha256",
        "loss_sha256",
    }


def test_backward_flops_excludes_gradient_postprocess(monkeypatch) -> None:
    target = load_target()
    workload = FakeWorkload("dynamic", [])
    profiler_active = False
    calls: list[tuple[str, bool]] = []

    class FakeProfile:
        def __enter__(self):
            nonlocal profiler_active
            profiler_active = True
            return self

        def __exit__(self, exc_type, exc, traceback):
            nonlocal profiler_active
            profiler_active = False

    monkeypatch.setattr(torch.profiler, "profile", lambda **kwargs: FakeProfile())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(target, "_profile_events", lambda profile: (123, {"op": 1}))
    monkeypatch.setattr(
        target,
        "_backward",
        lambda candidate, loss: calls.append(("backward", profiler_active)),
    )
    monkeypatch.setattr(
        target,
        "_postprocess",
        lambda candidate: calls.append(("postprocess", profiler_active)),
    )

    assert target._profile_flops(workload, "backward") == 123
    assert calls == [("backward", True), ("postprocess", False)]


def test_latency_rejects_workload_identity_drift(monkeypatch) -> None:
    target = load_target()
    configure_identity_mocks(monkeypatch, target)
    build_count = 0

    class DriftWorkload(FakeWorkload):
        def identity_receipt(self):
            receipt = super().identity_receipt()
            receipt["target_sha256"] = "a" * 64
            return receipt

    def build(arm: str, batch_size: int, seed: int):
        nonlocal build_count
        build_count += 1
        if build_count == 2:
            return DriftWorkload(arm, [])
        return FakeWorkload(arm, [])

    with pytest.raises(RuntimeError, match="latency cold workload identity drift"):
        target.run_latency(
            "dynamic",
            128,
            19031,
            **IDENTITY_KWARGS,
            build=build,
        )


def test_atomic_json_is_canonical_evidence_file(tmp_path: Path) -> None:
    target = load_target()
    path = tmp_path / "receipt.json"
    target._atomic_json(path, {"b": 2, "a": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {"a": 1, "b": 2}
    assert not list(tmp_path.glob("*.tmp"))
