import importlib.util
import json
import sys
from pathlib import Path

import arti
from arti.backend import jax_backend_status


ROOT = Path(__file__).resolve().parents[1]
REPORT_SPEC = importlib.util.spec_from_file_location(
    "report_backend_capabilities",
    ROOT / "benchmarks" / "report_backend_capabilities.py",
)
VERIFY_SPEC = importlib.util.spec_from_file_location(
    "verify_backend_capabilities",
    ROOT / "benchmarks" / "verify_backend_capabilities.py",
)
report_backend_capabilities = importlib.util.module_from_spec(REPORT_SPEC)
verify_backend_capabilities = importlib.util.module_from_spec(VERIFY_SPEC)
assert REPORT_SPEC.loader is not None
assert VERIFY_SPEC.loader is not None
sys.modules[REPORT_SPEC.name] = report_backend_capabilities
sys.modules[VERIFY_SPEC.name] = verify_backend_capabilities
REPORT_SPEC.loader.exec_module(report_backend_capabilities)
VERIFY_SPEC.loader.exec_module(verify_backend_capabilities)


def test_backend_capability_report_verifies_current_machine():
    payload = report_backend_capabilities.asdict(report_backend_capabilities.build_report())
    assert verify_backend_capabilities.verify(payload) == []
    assert arti.validate_backend_capabilities(payload, allow_cpu_torch=True) == []
    assert arti.doctor_report(allow_cpu_torch=True)["ok"] is True
    assert payload["torch_cuda_smoke_status"] in {"passed", "skipped"}
    assert payload["torch_cuda_smoke"]["smoke_status"] == payload["torch_cuda_smoke_status"]
    assert payload["jax_smoke_status"] in {"passed", "skipped", "failed"}
    assert payload["jax_smoke"]["smoke_status"] == payload["jax_smoke_status"]


def test_backend_capability_report_rejects_missing_jax_status():
    payload = report_backend_capabilities.asdict(report_backend_capabilities.build_report())
    payload["jax_backend_status"] = "ready"
    failures = verify_backend_capabilities.verify(payload)
    assert any("jax_backend_status" in failure for failure in failures)
    assert any("jax_backend_status" in failure for failure in arti.validate_backend_capabilities(payload))


def test_backend_capability_report_rejects_inconsistent_cuda_smoke():
    payload = report_backend_capabilities.asdict(report_backend_capabilities.build_report())
    payload["torch_cuda_smoke_status"] = "failed"
    payload["torch_cuda_smoke"] = {"smoke_status": "passed", "allocation_ok": True, "compute_ok": True}
    failures = verify_backend_capabilities.verify(payload)
    assert any("torch_cuda_smoke_status" in failure for failure in failures)
    assert any("torch_cuda_smoke_status" in failure for failure in arti.validate_backend_capabilities(payload, allow_cpu_torch=True))


def test_backend_capability_report_rejects_inconsistent_jax_smoke():
    payload = report_backend_capabilities.asdict(report_backend_capabilities.build_report())
    payload["jax_smoke_status"] = "failed"
    payload["jax_smoke"] = {"smoke_status": "passed", "backend_status": payload["jax_backend_status"]}
    failures = verify_backend_capabilities.verify(payload)
    assert any("jax_smoke_status" in failure for failure in failures)
    assert any("jax_smoke_status" in failure for failure in arti.validate_backend_capabilities(payload, allow_cpu_torch=True))


def test_cli_doctor_reports_backend_capabilities(capsys):
    from arti.cli import main

    args = ["doctor", "--allow-cpu-torch"]
    if jax_backend_status() == "available":
        args.append("--require-jax-smoke")
    assert main(args) == 0
    captured = capsys.readouterr()

    assert '"kind": "doctor"' in captured.out
    assert '"capabilities"' in captured.out
    assert '"torch_cuda_smoke_status"' in captured.out
    assert '"jax_smoke_status"' in captured.out


def test_cli_doctor_requirement_helper_rejects_missing_smoke():
    from arti.cli import enforce_doctor_requirements

    summary = {
        "ok": True,
        "kind": "doctor",
        "failures": [],
        "capabilities": {
            "torch_cuda_smoke_status": "skipped",
            "jax_smoke_status": "skipped",
        },
    }
    enforce_doctor_requirements(summary, require_cuda_smoke=True, require_jax_smoke=True)

    assert summary["ok"] is False
    assert any("CUDA smoke" in failure for failure in summary["failures"])
    assert any("JAX smoke" in failure for failure in summary["failures"])


def test_cli_doctor_writes_json_and_markdown_reports(tmp_path: Path, capsys):
    from arti.cli import main

    json_path = tmp_path / "doctor.json"
    md_path = tmp_path / "doctor.md"

    assert main(["doctor", "--allow-cpu-torch", "--output", str(json_path)]) == 0
    capsys.readouterr()
    assert main(["doctor", "--allow-cpu-torch", "--output", str(md_path)]) == 0
    captured = capsys.readouterr()

    assert json.loads(json_path.read_text(encoding="utf-8"))["kind"] == "doctor"
    assert md_path.read_text(encoding="utf-8").startswith("# ARTI Doctor Report")
    assert "CUDA smoke status" in md_path.read_text(encoding="utf-8")
    assert "JAX smoke" in md_path.read_text(encoding="utf-8")
    assert json.loads(captured.out)["output"] == str(md_path)
    assert "ARTI Doctor Report" in arti.doctor_report_markdown(arti.doctor_report(allow_cpu_torch=True))
