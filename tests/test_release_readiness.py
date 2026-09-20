from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("check_release_readiness", ROOT / "scripts" / "check_release_readiness.py")
check_release_readiness = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(check_release_readiness)


def test_version_sources_match() -> None:
    assert check_release_readiness.check_version_consistency() == []


def test_ci_workflow_covers_release_gates() -> None:
    assert check_release_readiness.check_ci_workflow() == []


def test_release_readiness_current_tree_passes() -> None:
    payload = check_release_readiness.check_readiness()
    assert payload["ok"] is True
    assert payload["ci_workflow"] == ".github/workflows/ci.yml"
    assert payload["version"] == check_release_readiness.project_version()
    assert payload["required_gates"] == ["docs", "mainline", "package", "pretrained", "quick"]


def test_release_readiness_rejects_failed_mainline_gate(monkeypatch) -> None:
    original_load_json = check_release_readiness.load_json

    def fake_load_json(path: Path) -> dict[str, Any]:
        payload = original_load_json(path)
        if path.name == "quality_gate_mainline.json":
            payload = dict(payload)
            payload["passed"] = False
        return payload

    monkeypatch.setattr(check_release_readiness, "load_json", fake_load_json)

    failures = check_release_readiness.check_gate_reports()

    assert any("mainline gate report is not passing" in failure for failure in failures)


def test_release_readiness_rejects_mainline_gate_truncated_runs(monkeypatch) -> None:
    original_load_json = check_release_readiness.load_json

    def fake_load_json(path: Path) -> dict[str, Any]:
        payload = original_load_json(path)
        if path.name == "quality_gate_mainline.json":
            payload = dict(payload)
            payload["runs"] = payload.get("runs", [])[:-1]
        return payload

    monkeypatch.setattr(check_release_readiness, "load_json", fake_load_json)

    failures = check_release_readiness.check_gate_reports()

    assert any("mainline gate report run count" in failure for failure in failures)


def test_release_readiness_can_require_cuda_when_packet_exists() -> None:
    payload = check_release_readiness.check_readiness(require_cuda=True)
    assert payload["ok"] is True
    assert "cuda" in payload["required_gates"]


def test_release_readiness_can_require_mechanism_when_gate_exists() -> None:
    payload = check_release_readiness.check_readiness(require_mechanism=True)
    assert payload["ok"] is True
    assert "mechanism" in payload["required_gates"]


def test_release_readiness_records_qwen_requirement_when_requested() -> None:
    payload = check_release_readiness.check_readiness(require_qwen=True)
    assert "qwen" in payload["required_gates"]
    assert payload["require_qwen"] is True


def test_release_readiness_rejects_failed_qwen_gate(monkeypatch) -> None:
    original_load_json = check_release_readiness.load_json

    def fake_load_json(path: Path) -> dict[str, Any]:
        payload = original_load_json(path)
        if path.name == "quality_gate_qwen.json":
            payload = dict(payload)
            payload["passed"] = False
        return payload

    monkeypatch.setattr(check_release_readiness, "load_json", fake_load_json)

    failures = check_release_readiness.check_gate_reports(require_qwen=True)

    assert any("qwen gate report is not passing" in failure for failure in failures)


def test_release_readiness_rejects_stale_qwen_gate_without_metadata_bridge(monkeypatch) -> None:
    original_load_json = check_release_readiness.load_json

    def fake_load_json(path: Path) -> dict[str, Any]:
        payload = original_load_json(path)
        if path.name == "quality_gate_qwen.json":
            payload = dict(payload)
            payload["runs"] = [
                row
                for row in payload.get("runs", [])
                if "qwen_runtime_vocab_pulse_metadata_bridge_probe.json" not in row.get("command", "")
                and "verify_qwen_runtime_vocab_metadata_bridge.py" not in row.get("command", "")
            ]
        return payload

    monkeypatch.setattr(check_release_readiness, "load_json", fake_load_json)

    failures = check_release_readiness.check_gate_reports(require_qwen=True)

    assert any("qwen_runtime_vocab_pulse_metadata_bridge_probe.json" in failure for failure in failures)
    assert any("verify_qwen_runtime_vocab_metadata_bridge.py" in failure for failure in failures)


def test_release_readiness_rejects_qwen_gate_missing_current_command(monkeypatch) -> None:
    original_load_json = check_release_readiness.load_json

    def fake_load_json(path: Path) -> dict[str, Any]:
        payload = original_load_json(path)
        if path.name == "quality_gate_qwen.json":
            payload = dict(payload)
            payload["runs"] = [
                row
                for row in payload.get("runs", [])
                if "benchmarks/verify_qwen_runtime_vocab_training.py" not in row.get("command", "")
            ]
        return payload

    monkeypatch.setattr(check_release_readiness, "load_json", fake_load_json)

    failures = check_release_readiness.check_gate_reports(require_qwen=True)

    assert any("current QWEN_COMMANDS entry" in failure and "verify_qwen_runtime_vocab_training.py" in failure for failure in failures)


def test_release_readiness_rejects_qwen_gate_wrong_gate_name(monkeypatch) -> None:
    original_load_json = check_release_readiness.load_json

    def fake_load_json(path: Path) -> dict[str, Any]:
        payload = original_load_json(path)
        if path.name == "quality_gate_qwen.json":
            payload = dict(payload)
            payload["gates"] = ["quick"]
        return payload

    monkeypatch.setattr(check_release_readiness, "load_json", fake_load_json)

    failures = check_release_readiness.check_gate_reports(require_qwen=True)

    assert any("gates ['qwen']" in failure for failure in failures)


def test_release_readiness_rejects_qwen_gate_truncated_runs(monkeypatch) -> None:
    original_load_json = check_release_readiness.load_json

    def fake_load_json(path: Path) -> dict[str, Any]:
        payload = original_load_json(path)
        if path.name == "quality_gate_qwen.json":
            payload = dict(payload)
            payload["runs"] = payload.get("runs", [])[:-1]
        return payload

    monkeypatch.setattr(check_release_readiness, "load_json", fake_load_json)

    failures = check_release_readiness.check_gate_reports(require_qwen=True)

    assert any("run count" in failure for failure in failures)


def test_release_readiness_rejects_qwen_gate_failed_run(monkeypatch) -> None:
    original_load_json = check_release_readiness.load_json

    def fake_load_json(path: Path) -> dict[str, Any]:
        payload = original_load_json(path)
        if path.name == "quality_gate_qwen.json":
            payload = dict(payload)
            payload["runs"] = [dict(row) for row in payload.get("runs", [])]
            payload["runs"][0]["returncode"] = 1
        return payload

    monkeypatch.setattr(check_release_readiness, "load_json", fake_load_json)

    failures = check_release_readiness.check_gate_reports(require_qwen=True)

    assert any("run 0 did not pass" in failure for failure in failures)


def test_release_readiness_rejects_failed_qwen_evidence(monkeypatch) -> None:
    original_load_module = check_release_readiness.load_module

    class FakeBridge:
        @staticmethod
        def verify(payload: dict) -> list[str]:
            return ["metadata bridge failed"]

    def fake_load_module(name: str, path: Path):
        if name == "verify_qwen_runtime_vocab_metadata_bridge":
            return FakeBridge
        return original_load_module(name, path)

    monkeypatch.setattr(check_release_readiness, "load_module", fake_load_module)

    failures = check_release_readiness.check_qwen_evidence()

    assert any("metadata bridge failed" in failure for failure in failures)


def test_release_readiness_rejects_failed_core_goal_evidence(monkeypatch) -> None:
    original_load_module = check_release_readiness.load_module

    class FakeCore:
        @staticmethod
        def verify(*, qwen: dict[str, dict], latent_repair: dict[str, dict]) -> list[str]:
            return ["goal drifted"]

    def fake_load_module(name: str, path: Path):
        if name == "verify_core_goal":
            return FakeCore
        return original_load_module(name, path)

    monkeypatch.setattr(check_release_readiness, "load_module", fake_load_module)

    payload = check_release_readiness.check_readiness()

    assert payload["ok"] is False
    assert any("core goal evidence: goal drifted" in failure for failure in payload["failures"])


def test_backend_capabilities_reject_available_jax_without_passing_smoke(monkeypatch) -> None:
    original_load_json = check_release_readiness.load_json

    def fake_load_json(path: Path) -> dict[str, Any]:
        payload = original_load_json(path)
        if path.name == "backend_capabilities.json":
            payload = dict(payload)
            payload["jax_backend_status"] = "available"
            payload["jax_smoke_status"] = "failed"
            payload["jax_smoke"] = {"smoke_status": "failed"}
        return payload

    monkeypatch.setattr(check_release_readiness, "load_json", fake_load_json)

    failures = check_release_readiness.check_backend_capabilities()

    assert any("JAX backend is available" in failure for failure in failures)


def test_backend_capabilities_require_cuda_smoke_when_cuda_required(monkeypatch) -> None:
    original_load_json = check_release_readiness.load_json

    def fake_load_json(path: Path) -> dict[str, Any]:
        payload = original_load_json(path)
        if path.name == "backend_capabilities.json":
            payload = dict(payload)
            payload["torch_cuda_smoke_status"] = "skipped"
            payload["torch_cuda_smoke"] = {"smoke_status": "skipped"}
        return payload

    monkeypatch.setattr(check_release_readiness, "load_json", fake_load_json)

    failures = check_release_readiness.check_backend_capabilities(require_cuda=True)

    assert any("CUDA is required" in failure for failure in failures)
