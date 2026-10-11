from __future__ import annotations

from dataclasses import replace
import json
import os
import shutil
import sys

import pytest

import arti
from arti.cli import main
from arti.recipe_data import json_fingerprint
from arti.recipe_runtime import TaskRuntimeBackendConfig, TaskRuntimeBackendContext
from test_recipe_runtime import fixture, profile
from test_recipe_assets import forbidden


def backend(name="local", *, revision="caller-host-v1"):
    return TaskRuntimeBackendConfig(f"examples.runtime_backend:{name}", revision)


def selected(tmp_path):
    source, portable, requests, rows, local, _report = fixture(tmp_path)
    runtime = replace(local, backend=backend())
    destination = tmp_path / "selected"
    result = arti.package_recipe_deployment(portable, destination, runtime=runtime)
    return source, portable, destination, requests, rows, runtime, result


def test_optional_backend_keeps_legacy_profile_serialization_and_identity():
    legacy = {
        "format": "arti.task-runtime-profile",
        "entrypoint": {"reference": "examples.runtime_task:describe", "kwargs": {}},
        "revision": "application-runtime-v1",
        "python": ">=3.10",
        "requirements": ["torch>=2"],
        "inputs": {},
        "outputs": {},
        "options": {},
        "environment": {},
        "network": "inherit",
        "limits": {},
        "require_enforcement": [],
        "allowed_dependencies": None,
        "container": None,
    }
    original = arti.TaskRuntimeProfile.from_dict(legacy)
    assert original.to_dict() == legacy
    assert json_fingerprint(original.to_dict()) == json_fingerprint(legacy)
    explicit_null = arti.TaskRuntimeProfile.from_dict({**legacy, "backend": None})
    assert explicit_null.to_dict() == legacy
    extended = replace(original, backend=backend())
    assert arti.TaskRuntimeProfile.from_dict(extended.to_dict()) == extended
    assert extended.to_dict()["backend"] == backend().to_dict()
    assert json_fingerprint(extended.to_dict()) != json_fingerprint(legacy)
    with pytest.raises(ValueError, match="explicit revision"):
        TaskRuntimeBackendConfig("examples.runtime_backend:local", "")


def test_selected_backend_metadata_inspection_is_code_free_and_explicit_inspection_discloses_trust(
    tmp_path, monkeypatch
):
    _source, _portable, package, _requests, _rows, runtime, _report = selected(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(arti.PythonFactoryConfig, "resolve", forbidden)
        observed = arti.inspect_task_runtime(package)
    assert observed["status"] == "not_run" and not observed["ok"]
    assert not observed["imports_backend"] and not observed["imports_application"]
    assert observed["declared"] == runtime.to_dict()
    explicit = arti.inspect_task_runtime(package, inspect_backend=True)
    assert explicit["ok"] and explicit["imports_backend"]
    assert not explicit["imports_application"] and not explicit["loads_model"]
    assert explicit["evidence_origin"] == "backend-reported"
    assert explicit["local_evidence"]["origin"] == "local-interpreter-probe"
    assert explicit["observed"]["python"]
    assert explicit["backend"]["declaration"] == backend().to_dict()
    assert explicit["backend"]["observation"]["container_verification"] == "not_run"
    assert explicit["profile_sha256"] == json_fingerprint(runtime.to_dict())


def test_backend_api_and_cli_require_explicit_trusted_inspection(tmp_path, monkeypatch, capsys):
    _source, _portable, package, _requests, _rows, _runtime, _report = selected(tmp_path)
    assert arti.TaskRuntimeBackendConfig is TaskRuntimeBackendConfig
    assert arti.TaskRuntimeBackendContext is TaskRuntimeBackendContext
    assert arti.torch.TaskRuntimeBackendConfig is TaskRuntimeBackendConfig
    assert arti.torch.TaskRuntimeBackendContext is TaskRuntimeBackendContext
    with monkeypatch.context() as patch:
        patch.setattr(arti.PythonFactoryConfig, "resolve", forbidden)
        assert main(["recipe", "inspect-runtime", str(package)]) == 1
    metadata = json.loads(capsys.readouterr().out)
    assert metadata["status"] == "not_run" and not metadata["imports_backend"]
    assert main(["recipe", "inspect-runtime", str(package), "--inspect-backend"]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["status"] == "compatible" and inspected["imports_backend"]
    assert inspected["trust"] == "explicit installed caller-owned host code"
    assert inspected["evidence_origin"] == "backend-reported"


def test_selected_backend_reuses_original_package_and_actual_cold_worker_without_prediction_replay(
    tmp_path,
):
    source, portable, package, requests, rows, runtime, packaged = selected(tmp_path)
    moved = tmp_path / "moved"
    inherited = arti.package_recipe_deployment(package, moved)
    assert inherited["deployment_sha256"] == packaged["deployment_sha256"]
    shutil.rmtree(source)
    shutil.rmtree(portable)
    shutil.rmtree(package)
    job, receipt = tmp_path / "job", tmp_path / "runtime.json"
    first = arti.run_task_runtime(
        moved, inputs={"requests": requests}, outputs={"results": job}, output=receipt
    )
    assert first["status"] == "completed" and first["application_executed"]
    assert first["observed"]["pid"] != os.getpid()
    assert first["worker"]["result"]["committed_records"] == len(rows)
    assert first["worker"]["result"]["output_files"] == 21
    assert first["deployment_sha256"] == packaged["deployment_sha256"]
    assert first["profile_sha256"] == json_fingerprint(runtime.to_dict())
    assert first["backend_identity"] == backend().to_dict()
    assert not first["locally_verified_execution"]
    assert first["enforced"] == {} and not first["network_enforced"]
    assert first["container_verification"] == "not_run" and not first["installs_dependencies"]
    initial = {
        path.relative_to(job): path.read_bytes()
        for path in (job / "parts").rglob("*")
        if path.is_file()
    }
    second = arti.run_task_runtime(
        moved,
        inputs={"requests": requests},
        outputs={"results": job},
        options={"resume": True},
        output=receipt,
    )
    assert second["status"] == "completed"
    assert second["worker"]["result"]["newly_predicted_records"] == 0
    assert initial == {
        path.relative_to(job): path.read_bytes()
        for path in (job / "parts").rglob("*")
        if path.is_file()
    }
    assert len(list((tmp_path / "runtime.reports").iterdir())) == 2
    assert arti.inspect_recipe_deployment(moved)["ok"]
    assert not list(moved.rglob("__pycache__"))


@pytest.mark.parametrize("constraint", ["memory", "network", "container", "dependency", "python"])
def test_selected_backend_cannot_hide_unsupported_or_incompatible_execution(
    tmp_path, monkeypatch, constraint
):
    runtime = profile("describe", backend=backend())
    if constraint == "memory":
        runtime = replace(
            runtime, limits={"memory_bytes": 4096}, require_enforcement=("memory_bytes",)
        )
    elif constraint == "network":
        runtime = replace(runtime, network="disabled", require_enforcement=("network",))
    elif constraint == "container":
        runtime = replace(runtime, container={"image": "caller-declared-container@digest"})
    elif constraint == "dependency":
        runtime = replace(runtime, requirements=("missing-caller-host-fixture-package>=1",))
    else:
        runtime = replace(runtime, python=">=99")
    _source, package, _requests, _rows, _runtime, _report = fixture(tmp_path, runtime=runtime)
    monkeypatch.setattr(TaskRuntimeBackendContext, "run_local", forbidden)
    result = arti.run_task_runtime(package, inputs={}, outputs={})
    assert result["status"] == "not_run" and result["application_executed"] is False
    assert result["inspection"]["imports_backend"]
    assert not result["inspection"]["ok"] and not result["installs_dependencies"]
    assert result["inspection"]["unsupported_enforcement"] or (
        result["inspection"]["missing_dependencies"]
        or result["inspection"]["incompatible_dependencies"]
    )


def test_unavailable_or_unobserved_target_environment_is_not_runtime_completion(
    tmp_path, monkeypatch
):
    runtime = profile("describe", backend=backend("unavailable"))
    _source, package, _requests, _rows, _runtime, _report = fixture(tmp_path, runtime=runtime)
    missing = arti.run_task_runtime(package, inputs={}, outputs={})
    assert missing["status"] == "not_run" and not missing["application_executed"]
    assert missing["inspection"]["observed"] is None
    import examples.runtime_backend as application

    calls = []

    def unobserved(context, *, operation):
        calls.append(operation)
        return {"available": True, "capabilities": [], "environment": {"python": "3.12"}}

    monkeypatch.setattr(application, "unavailable", unobserved)
    invalid = arti.run_task_runtime(package, inputs={}, outputs={})
    assert calls == ["inspect"] and invalid["status"] == "not_run"
    assert invalid["inspection"]["backend_error"]["type"] == "ValueError"


@pytest.mark.parametrize("applied", ["missing", "boolean-value"])
def test_declared_caps_do_not_fabricate_applied_enforcement_receipts(
    tmp_path, monkeypatch, applied
):
    runtime = profile(
        "describe",
        backend=backend(),
        limits={"memory_bytes": 1},
        require_enforcement=("memory_bytes",),
    )
    _source, package, _requests, _rows, _runtime, _report = fixture(tmp_path, runtime=runtime)
    import examples.runtime_backend as application

    original = application.local

    def no_enforcement(context, *, operation):
        if operation == "inspect":
            value = original(context, operation=operation)
            value["capabilities"].append("memory_bytes")
            return value
        return {
            "status": "completed",
            "ok": True,
            "application_executed": False,
            "inspection": {"deployment_sha256": context.deployment_sha256},
            "runtime_identity": {"interpreter": context.inspect_local()["observed"]},
            "enforced": {}
            if applied == "missing"
            else {"memory_bytes": {"value": True, "scope": "caller-reported fixture"}},
        }

    monkeypatch.setattr(application, "local", no_enforcement)
    result = arti.run_task_runtime(package, inputs={}, outputs={})
    assert result["status"] == "failed" and not result["ok"]
    expected = "required enforcement evidence" if applied == "missing" else "declared limit/scope"
    assert expected in result["backend_error"]["message"]
    if applied == "missing":
        assert result["backend_report"]["enforced"] == {}
    assert result["application_executed"] is None


def test_actual_delegated_deadline_has_direct_worker_scope_and_reported_origin(tmp_path):
    runtime = profile(
        "wait",
        backend=backend(),
        outputs={"marker": "file"},
        options={"seconds": 60},
        limits={"process_wall_seconds": 10},
        require_enforcement=("process_wall_seconds",),
    )
    _source, package, _requests, _rows, _runtime, _report = fixture(tmp_path, runtime=runtime)
    marker = tmp_path / "started.txt"
    result = arti.run_task_runtime(package, inputs={}, outputs={"marker": marker})
    assert result["status"] == "terminated_by_host" and not result["ok"]
    assert result["observed"]["pid"] != os.getpid()
    assert result["observed"]["returncode"] is not None
    assert result["enforced"]["process_wall_seconds"]["triggered"]
    assert (
        "descendants/startup probe excluded" in result["enforced"]["process_wall_seconds"]["scope"]
    )
    assert result["enforced"]["process_wall_seconds"]["evidence_origin"] == "backend-reported"
    assert not result["network_enforced"] and result["container_verification"] == "not_run"


def test_backend_failure_after_effect_is_ambiguous_and_never_automatically_repeated(
    tmp_path, monkeypatch
):
    runtime = profile("describe", backend=backend("fail_after_marker"), outputs={"marker": "file"})
    _source, package, _requests, _rows, _runtime, _report = fixture(tmp_path, runtime=runtime)
    import examples.runtime_backend as application

    original, calls = application.fail_after_marker, []

    def observed(context, *, operation):
        calls.append(operation)
        return original(context, operation=operation)

    monkeypatch.setattr(application, "fail_after_marker", observed)
    marker = tmp_path / "marker.txt"
    result = arti.run_task_runtime(package, inputs={}, outputs={"marker": marker})
    assert calls == ["inspect", "run"] and marker.is_file()
    assert result["status"] == "failed" and result["application_executed"] is None
    assert result["execution_outcome"].startswith("unknown")
    assert result["recovery"] == "caller decision required; no automatic repeat"


def test_foreign_execution_receipt_keeps_failure_and_original_identity(tmp_path, monkeypatch):
    runtime = profile("describe", backend=backend())
    _source, package, _requests, _rows, _runtime, packaged = fixture(tmp_path, runtime=runtime)
    import examples.runtime_backend as application

    original = application.local

    def foreign(context, *, operation):
        value = original(context, operation=operation)
        if operation == "run":
            value["inspection"]["deployment_sha256"] = "0" * 64
        return value

    monkeypatch.setattr(application, "local", foreign)
    result = arti.run_task_runtime(package, inputs={}, outputs={})
    assert result["status"] == "failed" and result["application_executed"] is None
    assert result["deployment_sha256"] == packaged["deployment_sha256"]
    assert "different package" in result["backend_error"]["message"]
    assert result["backend_report"]["observed"]["pid"] != os.getpid()
    assert not list(package.rglob("__pycache__"))


@pytest.mark.parametrize("receipt", ["metadata-only", "installation-observation"])
def test_backend_completion_and_installation_fields_keep_their_actual_claim_scope(
    tmp_path, monkeypatch, receipt
):
    runtime = profile("describe", backend=backend())
    _source, package, _requests, _rows, _runtime, _report = fixture(tmp_path, runtime=runtime)
    import examples.runtime_backend as application

    original = application.local

    def reported(context, *, operation):
        if operation == "inspect":
            return original(context, operation=operation)
        if receipt == "metadata-only":
            return {
                "status": "completed",
                "ok": True,
                "application_executed": False,
                "inspection": {"deployment_sha256": context.deployment_sha256},
                "runtime_identity": {"interpreter": context.inspect_local()["observed"]},
                "enforced": {},
                "observed": {"note": "no application was executed"},
            }
        value = context.run_local()
        # A synthetic backend attribution tests the receipt field only; no installer runs.
        value["installs_dependencies"] = True
        return value

    monkeypatch.setattr(application, "local", reported)
    result = arti.run_task_runtime(package, inputs={}, outputs={})
    assert result["core_installs_dependencies"] is False
    if receipt == "metadata-only":
        assert result["status"] == "failed" and not result["ok"]
        assert "actual application execution observations" in result["backend_error"]["message"]
    else:
        assert result["status"] == "completed"
        assert result["installs_dependencies"] is True
        assert result["backend_report"]["installs_dependencies"] is True
        assert result["evidence_origin"] == "backend-reported; trusted caller code"
        assert result["origin"] == "backend-reported"


def test_unavailable_controller_probe_does_not_admit_or_block_the_selected_target(
    tmp_path, monkeypatch
):
    runtime = profile(
        "describe",
        backend=TaskRuntimeBackendConfig(
            arti.PythonFactoryConfig(
                "examples.runtime_backend:independent_local", {"interpreter": sys.executable}
            ),
            "explicit-existing-target-v1",
        ),
    )
    _source, package, _requests, _rows, _runtime, _report = fixture(tmp_path, runtime=runtime)
    unavailable = str(tmp_path / "missing-controller-interpreter.exe")
    with monkeypatch.context() as patch:
        patch.setattr(arti.PythonFactoryConfig, "resolve", forbidden)
        metadata = arti.inspect_task_runtime(package, python=unavailable)
    assert metadata["status"] == "not_run" and not metadata["imports_backend"]
    assert metadata["local_evidence"]["status"] == "unavailable"
    assert metadata["local_evidence"]["error"]["type"] == "FileNotFoundError"
    inspected = arti.inspect_task_runtime(package, python=unavailable, inspect_backend=True)
    assert inspected["ok"] and inspected["imports_backend"]
    assert inspected["local_evidence"]["status"] == "unavailable"
    assert inspected["observed"]["executable"] == sys.executable
    executed = arti.run_task_runtime(package, python=unavailable, inputs={}, outputs={})
    assert executed["status"] == "completed" and executed["application_executed"]
    assert executed["inspection"]["local_evidence"]["status"] == "unavailable"
    assert executed["runtime_identity"]["interpreter"]["executable"] == sys.executable
    assert executed["worker"]["pid"] != os.getpid()
    assert executed["origin"] == "backend-reported" and not executed["core_installs_dependencies"]
    # The original local entrance retains its ordinary missing-interpreter failure.
    local = tmp_path / "local"
    arti.package_recipe_deployment(package, local, runtime=replace(runtime, backend=None))
    with pytest.raises(FileNotFoundError):
        arti.inspect_task_runtime(local, python=unavailable)
