from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

import arti
from arti.cli import main
from arti.recipe_workflow import write_json
from test_prediction_files import setup
from test_recipe_assets import forbidden


def profile(entrypoint="predict", **kwargs):
    settings = {
        "requirements": ("torch>=2",),
        "inputs": {"requests": "file"} if entrypoint == "predict" else {},
        "outputs": {"results": "directory"} if entrypoint == "predict" else {},
    }
    return arti.TaskRuntimeProfile(
        f"examples.runtime_task:{entrypoint}", "application-runtime-v1", **{**settings, **kwargs}
    )


def fixture(tmp_path, *, runtime=None):
    source = tmp_path / "source"
    package, requests, files, rows = setup(source)
    code = source / "application/examples"
    code.mkdir(parents=True)
    (code / "__init__.py").write_text("", encoding="utf-8")
    for name in (
        "runtime_task.py",
        "asset_task_recipe.py",
        "custom_task_recipe.py",
        "evaluation_suite_scorers.py",
        "prediction_scorers.py",
    ):
        shutil.copyfile(Path("examples") / name, code / name)
    runtime = runtime or profile(
        options={
            "task_factory": "examples.asset_task_recipe:build_task",
            "batch_size": 3,
            "files": files.to_dict(),
        }
    )
    if runtime.entrypoint.reference == "examples.runtime_task:validate":
        runtime = replace(runtime, options={**runtime.options, "files": files.to_dict()})
    portable = tmp_path / "portable"
    report = arti.package_recipe_deployment(
        package, portable, runtime=runtime, code={"application": code.parent}
    )
    data = tmp_path / "requests.jsonl"
    shutil.copyfile(requests, data)
    return source, portable, data, rows, runtime, report


def test_runtime_code_resource_closure_relocation_and_cold_native_prediction_without_trainer_tree(
    tmp_path,
):
    source, portable, requests, rows, runtime, packaged = fixture(tmp_path)
    inspection = arti.inspect_task_runtime(portable)
    assert (
        inspection["ok"] and not inspection["imports_application"] and not inspection["loads_model"]
    )
    copied = tmp_path / "repacked"
    inherited = arti.package_recipe_deployment(portable, copied)
    assert inherited["deployment_sha256"] == packaged["deployment_sha256"]
    shutil.rmtree(source)
    shutil.rmtree(portable)
    moved = tmp_path / "moved"
    shutil.move(copied, moved)
    job = tmp_path / "prediction"
    result = arti.run_task_runtime(
        moved,
        inputs={"requests": requests},
        outputs={"results": job},
        output=tmp_path / "runtime.json",
    )
    assert result["status"] == "completed" and result["application_executed"]
    assert result["worker"]["result"]["committed_records"] == len(rows) == 7
    assert result["worker"]["result"]["output_files"] == 21
    assert result["observed"]["pid"] != os.getpid()
    assert not result["network_enforced"] and result["enforced"] == {}
    assert not result["installs_dependencies"]
    assert result["inspection"]["profile_sha256"] == arti.recipe_data.json_fingerprint(
        runtime.to_dict()
    )
    assert len(result["runtime_identity"]["code_files"]) == 6
    repeat = arti.run_task_runtime(
        moved,
        inputs={"requests": requests},
        outputs={"results": job},
        options={"resume": True},
        output=tmp_path / "runtime.json",
    )
    assert repeat["worker"]["result"]["newly_predicted_records"] == 0
    assert len(list((tmp_path / "runtime.reports").iterdir())) == 2


@pytest.mark.parametrize(
    "change", ["python", "dependency", "whitelist", "memory", "network", "container"]
)
def test_incompatible_profile_is_not_executed_or_automatically_installed(tmp_path, change):
    runtime = profile("describe")
    if change == "python":
        runtime = replace(runtime, python=">=99")
    elif change == "dependency":
        runtime = replace(runtime, requirements=("missing-runtime-fixture-package>=1",))
    elif change == "whitelist":
        runtime = replace(runtime, allowed_dependencies=())
    elif change == "memory":
        runtime = replace(
            runtime, limits={"memory_bytes": 4096}, require_enforcement=("memory_bytes",)
        )
    elif change == "network":
        runtime = replace(runtime, network="disabled", require_enforcement=("network",))
    else:
        runtime = replace(runtime, container={"image": "caller-selected-image@declared-digest"})
    _source, portable, _requests, _rows, _runtime, _report = fixture(tmp_path, runtime=runtime)
    inspected = arti.inspect_task_runtime(portable)
    assert not inspected["ok"] and inspected["status"] == "incompatible"
    result = arti.run_task_runtime(portable, inputs={}, outputs={})
    assert result["status"] == "not_run" and not result["application_executed"]
    assert (
        not result["installs_dependencies"]
        and result["inspection"]["backend"]["container_verification"] == "not_run"
    )


def test_declared_limits_are_not_host_caps_and_environment_is_observed_in_a_distinct_cwd(tmp_path):
    runtime = profile(
        "describe",
        network="disabled",
        limits={"memory_bytes": 4096, "cpu_threads": 2},
        environment={"RUNTIME_FIXTURE_FLAG": "declared"},
        options={"environment_keys": ["RUNTIME_FIXTURE_FLAG", "HF_HUB_OFFLINE"]},
    )
    _source, portable, _requests, _rows, _runtime, _report = fixture(tmp_path, runtime=runtime)
    report = arti.run_task_runtime(portable, inputs={}, outputs={})
    assert report["status"] == "completed"
    assert report["unenforced"] == dict(runtime.limits) and report["enforced"] == {}
    assert report["worker"]["result"]["variables"] == {
        "RUNTIME_FIXTURE_FLAG": "declared",
        "HF_HUB_OFFLINE": "1",
    }
    assert Path(report["worker"]["result"]["cwd"]) != Path.cwd()
    assert not report["network_enforced"]


def test_application_failure_preserves_diagnostics_and_never_claims_quality(tmp_path):
    _source, portable, _requests, _rows, _runtime, _report = fixture(
        tmp_path, runtime=profile("fail")
    )
    report = arti.run_task_runtime(portable, inputs={}, outputs={})
    assert report["status"] == "failed" and not report["ok"]
    assert report["worker"]["error"]["type"] == "RuntimeError"
    assert "task_quality_evaluated" not in report


def test_worker_deadline_is_actual_termination_not_a_declared_memory_or_network_limit(tmp_path):
    runtime = profile(
        "wait",
        outputs={"marker": "file"},
        options={"seconds": 60},
        limits={"process_wall_seconds": 10},
        require_enforcement=("process_wall_seconds",),
    )
    _source, portable, _requests, _rows, _runtime, _report = fixture(tmp_path, runtime=runtime)
    marker = tmp_path / "started.txt"
    report = arti.run_task_runtime(portable, inputs={}, outputs={"marker": marker})
    assert (
        report["status"] == "terminated_by_host"
        and report["enforced"]["process_wall_seconds"]["triggered"]
    )
    assert report["observed"]["returncode"] is not None
    assert not report["network_enforced"]
    assert report["observed"]["elapsed_seconds"] < 60
    assert report["observed"]["launch"]["selected_executable"] == report["inspection"]["observed"]["executable"]


def test_direct_selected_interpreter_preserves_environment_and_actual_process_pid():
    from arti.recipe_runtime import _local_python_launch, _python_probe
    observed = _python_probe(sys.executable, ["torch"])
    executable, environment, launch = _local_python_launch(
        observed, {**os.environ, "ARTI_TEST_ENVIRONMENT": "caller-value", "PYTHONDONTWRITEBYTECODE": "1"}
    )
    code = """
import json,os,sys,importlib.metadata as m,importlib.util
print(json.dumps({'pid':os.getpid(),'executable':sys.executable,'prefix':sys.prefix,
 'torch':m.version('torch'),'torch_origin':importlib.util.find_spec('torch').origin,
 'environment':os.environ['ARTI_TEST_ENVIRONMENT']}))
"""
    with subprocess.Popen([executable, "-c", code], env=environment,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as process:
        stdout, stderr = process.communicate(timeout=30)
        assert process.returncode == 0, stderr
        worker = json.loads(stdout)
        assert worker["pid"] == process.pid
    assert worker["executable"] == observed["executable"]
    assert worker["prefix"] == observed["prefix"]
    assert worker["torch"] == observed["versions"]["torch"]
    assert worker["environment"] == "caller-value"
    reference = subprocess.run([sys.executable, "-c", "import importlib.util;print(importlib.util.find_spec('torch').origin)"],
                               capture_output=True, text=True, check=True)
    assert worker["torch_origin"] == reference.stdout.strip()
    if os.name == "nt" and observed["prefix"] != observed["base_prefix"]:
        assert launch["venv_redirector_bypassed"]
        assert launch["process_executable"] == observed["base_executable"]


def test_missing_observed_venv_base_never_selects_an_alternative_runtime():
    from arti.recipe_runtime import _local_python_launch
    if os.name != "nt":
        return
    observed = {"executable": sys.executable, "base_executable": "missing-runtime.exe",
                "prefix": "selected-env", "base_prefix": "base-env",
                "markers": {"implementation_name": "cpython"}}
    with pytest.raises(ValueError, match="runtime was not launched"):
        _local_python_launch(observed, {})


def test_framework_dependencies_use_the_distribution_that_owns_the_import(monkeypatch, tmp_path):
    import arti.recipe_runtime as runtime
    from types import SimpleNamespace
    source = Path(runtime.__file__).with_name("__init__.py").resolve()
    calls = []

    def distribution(name):
        calls.append(name)
        return SimpleNamespace(
            locate_file=lambda _member: source if name == "arti-fit" else tmp_path / "other.py",
            requires=["packaging>=23"] if name == "arti-fit" else ["unrelated-distribution>=1"],
        )

    monkeypatch.setattr(runtime.importlib.metadata, "distribution", distribution)
    assert runtime._framework_requirements() == ["packaging>=23"]
    assert calls == ["arti-fit"]


def test_private_distribution_remains_available_when_public_distribution_is_absent(monkeypatch):
    import arti.recipe_runtime as runtime
    from types import SimpleNamespace
    source = Path(runtime.__file__).with_name("__init__.py").resolve()

    def distribution(name):
        if name == "arti-fit":
            raise runtime.importlib.metadata.PackageNotFoundError(name)
        return SimpleNamespace(locate_file=lambda _member: source, requires=["packaging>=23"])

    monkeypatch.setattr(runtime.importlib.metadata, "distribution", distribution)
    assert runtime._framework_requirements() == ["packaging>=23"]


def test_installed_core_markers_are_preserved_for_the_target_environment(monkeypatch):
    import arti.recipe_runtime as runtime
    from types import SimpleNamespace
    source = Path(runtime.__file__).with_name("__init__.py").resolve()
    core = [
        "packaging>=23", "tomli>=2; python_version < '3.11'",
        "platform-helper>=1; platform_machine == 'extra'",
        "platform-helper>=1; 'extra' == platform_machine",
    ]
    extra = [
        "optional-helper>=1; extra == 'optional'",
        "optional-helper>=1; 'optional' == extra and python_version < '3.14'",
        "optional-helper>=1; python_version >= '3.10' and (extra == 'optional' or extra == 'other')",
    ]
    monkeypatch.setattr(runtime.importlib.metadata, "distribution", lambda _name: SimpleNamespace(
        locate_file=lambda _member: source, requires=core + extra,
    ))
    requirements = runtime._framework_requirements()
    assert requirements == core
    environment = {
        "python": "3.10.1", "markers": {"python_version": "3.10", "platform_machine": "standard"},
        "versions": {"packaging": "23.0", "tomli": None, "platform-helper": None},
    }
    missing, incompatible = runtime._compatibility(profile(), requirements, environment)
    assert missing == [core[1]] and incompatible == []
    environment.update(python="3.12.1", markers={"python_version": "3.12", "platform_machine": "standard"})
    assert runtime._compatibility(profile(), requirements, environment) == ([], [])


def test_missing_or_changed_code_is_detected_before_application_import(tmp_path, monkeypatch):
    _source, portable, _requests, _rows, _runtime, _report = fixture(tmp_path)
    code = next(portable.glob("code/*/examples/runtime_task.py"))
    code.write_text("changed", encoding="utf-8")
    monkeypatch.setattr(arti.PythonFactoryConfig, "resolve", forbidden)
    with pytest.raises(ValueError, match="files are unavailable"):
        arti.inspect_task_runtime(portable)


def test_cli_runtime_profile_and_explicit_code_distribution(tmp_path, monkeypatch, capsys):
    source, portable, requests, _rows, runtime, _report = fixture(tmp_path)
    profile_file = tmp_path / "profile.json"
    write_json(profile_file, runtime.to_dict())
    assert arti.TaskRuntimeProfile.read(profile_file) == runtime
    monkeypatch.setattr(arti.PythonFactoryConfig, "resolve", forbidden)
    assert main(["recipe", "inspect-runtime", str(portable)]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["ok"] and not inspected["imports_application"]
    inputs, outputs = tmp_path / "inputs.json", tmp_path / "outputs.json"
    write_json(inputs, {"requests": "requests.jsonl"})
    write_json(outputs, {"results": "job"})
    shutil.rmtree(source)
    assert (
        main(
            [
                "recipe",
                "run-runtime",
                str(portable),
                "--inputs-json",
                str(inputs),
                "--outputs-json",
                str(outputs),
                "--output",
                str(tmp_path / "receipt.json"),
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["worker"]["result"]["committed_records"] == 7
    assert arti.torch.TaskRuntimeProfile is arti.TaskRuntimeProfile


def test_runtime_entrypoint_connects_prediction_suite_evaluation_and_cold_reuse(tmp_path):
    from test_recipe_suite import suite
    from arti.recipe_workflow import write_jsonl

    runtime = profile(
        "validate",
        inputs={"requests": "file", "suite": "file", "references": "file"},
        outputs={"results": "directory", "evaluation": "file"},
        options={"task_factory": "examples.asset_task_recipe:build_task", "batch_size": 3},
    )
    source, portable, requests, rows, _runtime, _report = fixture(tmp_path, runtime=runtime)
    references, config = tmp_path / "references.jsonl", tmp_path / "suite.json"
    write_jsonl(references, [{**row, "available": True} for row in rows])
    write_json(config, suite(aggregate=True).to_dict())
    job, evaluation = tmp_path / "job", tmp_path / "evaluation.json"
    shutil.rmtree(source)
    result = arti.run_task_runtime(
        portable,
        inputs={"requests": requests, "suite": config, "references": references},
        outputs={"results": job, "evaluation": evaluation},
    )
    assert result["status"] == "completed"
    scored = json.loads(evaluation.read_text())
    assert scored["complete_evidence"] and scored["status"] == "evaluated"
    again = arti.run_task_runtime(
        portable,
        inputs={"requests": requests, "suite": config, "references": references},
        outputs={"results": job, "evaluation": evaluation},
        options={"resume": True},
    )
    assert again["worker"]["result"]["prediction"]["newly_predicted_records"] == 0
    assert (
        not list(portable.rglob("__pycache__")) and arti.inspect_recipe_deployment(portable)["ok"]
    )


def test_explicit_offline_wheel_is_pinned_but_not_installed(tmp_path):
    source, portable, _requests, _rows, runtime, _report = fixture(
        tmp_path, runtime=profile("describe")
    )
    wheel = tmp_path / "offline_fixture_dependency-1.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("offline_fixture_dependency/__init__.py", "VALUE = 1\n")
        archive.writestr(
            "offline_fixture_dependency-1.0.dist-info/METADATA",
            "Metadata-Version: 2.1\nName: offline-fixture-dependency\nVersion: 1.0\n",
        )
        archive.writestr(
            "offline_fixture_dependency-1.0.dist-info/WHEEL",
            "Wheel-Version: 1.0\nGenerator: fixture\nRoot-Is-Purelib: true\nTag: py3-none-any\n",
        )
    dependency_runtime = replace(runtime, requirements=("offline-fixture-dependency==1.0",))
    destination = tmp_path / "with-wheel"
    arti.package_recipe_deployment(
        portable, destination, runtime=dependency_runtime, wheels={"dependency": wheel}
    )
    metadata = arti.inspect_recipe_deployment(destination)["runtime"]["wheels"]["dependency"]
    assert metadata["distribution"]["name"] == "offline-fixture-dependency"
    assert metadata["distribution"]["installed_automatically"] is False
    assert not arti.inspect_task_runtime(destination)["ok"]
    repacked = tmp_path / "with-wheel-again"
    arti.package_recipe_deployment(destination, repacked)
    wheel.unlink()
    shutil.rmtree(source)
    assert arti.inspect_recipe_deployment(repacked)["ok"]


def test_runtime_io_and_receipt_do_not_mutate_immutable_inputs(tmp_path):
    _source, portable, requests, _rows, _runtime, _report = fixture(tmp_path)
    with pytest.raises(ValueError, match="exactly the declared ports"):
        arti.run_task_runtime(portable, inputs={}, outputs={"results": tmp_path / "job"})
    with pytest.raises(ValueError, match="separate from input|wrong file kind"):
        arti.run_task_runtime(
            portable, inputs={"requests": requests}, outputs={"results": requests}
        )
    with pytest.raises(ValueError, match="must not mutate"):
        arti.run_task_runtime(
            portable, inputs={"requests": requests}, outputs={"results": portable / "job"}
        )
