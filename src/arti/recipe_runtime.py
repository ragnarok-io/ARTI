"""Explicit trusted application entrypoints and qualified local handoff."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
import tokenize

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name

from .python_factory import PythonFactoryConfig
from .recipe_data import json_fingerprint
from .recipe_deployment import (
    _requirements,
    _resource_path,
    _task_deployment,
    inspect_recipe_deployment,
)
from .recipe_validation import _publish
from .recipe_workflow import read_recipe_config
from .serialization import _atomic_json, _load_json


def _json(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _log_tail(path, limit=8192):
    with path.open("rb") as stream:
        stream.seek(max(0, path.stat().st_size - limit))
        return stream.read(limit).decode("utf-8", errors="replace")


def _named_paths(value, name):
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) or not key or kind not in {"file", "directory"}
        for key, kind in value.items()
    ):
        raise ValueError(f"runtime {name} must map named ports to file/directory kinds")
    return dict(value)


@dataclass(frozen=True)
class TaskRuntimeBackendConfig:
    """Explicit installed trusted host callback; never installed by the runtime."""

    factory: PythonFactoryConfig | str
    revision: str

    def __post_init__(self):
        if isinstance(self.factory, str):
            object.__setattr__(self, "factory", PythonFactoryConfig(self.factory))
        if (
            not isinstance(self.factory, PythonFactoryConfig)
            or not isinstance(self.revision, str)
            or not self.revision.strip()
        ):
            raise ValueError("runtime backend needs an installed callback and explicit revision")

    def to_dict(self):
        return {"factory": self.factory.to_dict(), "revision": self.revision}

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if isinstance(value.get("factory"), Mapping):
            value["factory"] = PythonFactoryConfig(**value["factory"])
        return cls(**value)


@dataclass(frozen=True)
class TaskRuntimeProfile:
    entrypoint: PythonFactoryConfig | str
    revision: str
    python: str = ">=3.10"
    requirements: tuple[str, ...] = ()
    inputs: Mapping[str, str] = field(default_factory=dict)
    outputs: Mapping[str, str] = field(default_factory=dict)
    options: Mapping = field(default_factory=dict)
    environment: Mapping[str, str] = field(default_factory=dict)
    network: str = "inherit"
    limits: Mapping[str, int | float] = field(default_factory=dict)
    require_enforcement: tuple[str, ...] = ()
    allowed_dependencies: tuple[str, ...] | None = None
    container: Mapping | None = None
    backend: TaskRuntimeBackendConfig | None = None

    def __post_init__(self):
        if isinstance(self.entrypoint, str):
            object.__setattr__(self, "entrypoint", PythonFactoryConfig(self.entrypoint))
        if (
            not isinstance(self.entrypoint, PythonFactoryConfig)
            or not isinstance(self.revision, str)
            or not self.revision.strip()
        ):
            raise ValueError("runtime requires an installed entrypoint and explicit revision")
        SpecifierSet(self.python)
        object.__setattr__(self, "requirements", tuple(_requirements(self.requirements)))
        object.__setattr__(self, "inputs", _named_paths(self.inputs, "inputs"))
        object.__setattr__(self, "outputs", _named_paths(self.outputs, "outputs"))
        object.__setattr__(self, "options", _json(dict(self.options)))
        if not isinstance(self.environment, Mapping) or any(
            not isinstance(key, str) or not key or not isinstance(value, str)
            for key, value in self.environment.items()
        ):
            raise ValueError("runtime environment must map names to strings")
        object.__setattr__(self, "environment", dict(self.environment))
        if self.network not in {"inherit", "disabled"}:
            raise ValueError("runtime network must be inherit or disabled")
        if any(
            not isinstance(key, str)
            or not key
            or type(value) not in {int, float}
            or not math.isfinite(value)
            or value <= 0
            for key, value in self.limits.items()
        ):
            raise ValueError("runtime limits must have named finite positive values")
        object.__setattr__(self, "limits", dict(self.limits))
        object.__setattr__(self, "require_enforcement", tuple(self.require_enforcement))
        if set(self.require_enforcement) - (set(self.limits) | {"network"}):
            raise ValueError("required enforcement must name a declared limit or network")
        if "network" in self.require_enforcement and self.network != "disabled":
            raise ValueError("required network isolation needs network=disabled")
        if self.allowed_dependencies is not None:
            if any(not isinstance(name, str) or not name for name in self.allowed_dependencies):
                raise ValueError("allowed_dependencies must name Python distributions")
            object.__setattr__(
                self,
                "allowed_dependencies",
                tuple(canonicalize_name(name) for name in self.allowed_dependencies),
            )
        if self.container is not None:
            if not isinstance(self.container, Mapping) or not self.container.get("image"):
                raise ValueError("container handoff must declare a caller-owned image identity")
            object.__setattr__(self, "container", _json(dict(self.container)))
        if self.backend is not None and not isinstance(self.backend, TaskRuntimeBackendConfig):
            raise ValueError("runtime backend must be an explicit TaskRuntimeBackendConfig")

    def to_dict(self):
        result = {
            "format": "arti.task-runtime-profile",
            "entrypoint": self.entrypoint.to_dict(),
            "revision": self.revision,
            "python": self.python,
            "requirements": list(self.requirements),
            "inputs": dict(self.inputs),
            "outputs": dict(self.outputs),
            "options": _json(self.options),
            "environment": dict(self.environment),
            "network": self.network,
            "limits": dict(self.limits),
            "require_enforcement": list(self.require_enforcement),
            "allowed_dependencies": None
            if self.allowed_dependencies is None
            else list(self.allowed_dependencies),
            "container": self.container,
        }
        # An omitted extension must preserve existing portable profile fingerprints.
        if self.backend is not None:
            result["backend"] = self.backend.to_dict()
        return result

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if value.pop("format", "arti.task-runtime-profile") != "arti.task-runtime-profile":
            raise ValueError("not a task runtime profile")
        if isinstance(value.get("entrypoint"), Mapping):
            value["entrypoint"] = PythonFactoryConfig(**value["entrypoint"])
        if isinstance(value.get("backend"), Mapping):
            value["backend"] = TaskRuntimeBackendConfig.from_dict(value["backend"])
        return cls(**value)

    @classmethod
    def read(cls, path):
        return cls.from_dict(read_recipe_config(path))


@dataclass(frozen=True)
class TaskRuntimeContext:
    package: Path
    inputs: Mapping[str, Path]
    outputs: Mapping[str, Path]
    options: Mapping
    profile: TaskRuntimeProfile


@dataclass(frozen=True)
class TaskRuntimeBackendContext:
    package: Path
    deployment_sha256: str
    profile: TaskRuntimeProfile
    requirements: tuple[str, ...]
    python: str | Path | None
    inputs: Mapping[str, Path] = field(default_factory=dict)
    outputs: Mapping[str, Path] = field(default_factory=dict)
    options: Mapping = field(default_factory=dict)

    def inspect_local(self):
        """Observe the local interpreter; no backend/application code is imported."""
        root, deployment, profile = _runtime(self.package)
        if deployment["sha256"] != self.deployment_sha256:
            raise ValueError("runtime package changed before backend inspection")
        return _inspect_local_runtime(root, deployment, profile, self.python)

    def run_local(self):
        """Delegate to the existing actual cold worker, retaining original identities."""
        _root, deployment, _profile = _runtime(self.package)
        if deployment["sha256"] != self.deployment_sha256:
            raise ValueError("runtime package changed before backend execution")
        return _run_local_task_runtime(
            self.package,
            inputs=self.inputs,
            outputs=self.outputs,
            python=self.python,
            options=self.options,
        )


def _uses_distribution_extra(requirement):
    marker = Requirement(requirement).marker
    if marker is None:
        return False
    # Canonical PEP 508 markers use names for variables and quoted strings for
    # values. Tokenize those separately so a literal value "extra" stays core.
    tokens = tokenize.generate_tokens(io.StringIO(str(marker)).readline)
    return any(token.type == tokenize.NAME and token.string == "extra" for token in tokens)


def _framework_requirements():
    for name in ("arti-fit", "arti"):
        try:
            distribution = importlib.metadata.distribution(name)
            if (
                distribution.locate_file("arti/__init__.py").resolve()
                == Path(__file__).with_name("__init__.py").resolve()
            ):
                return [
                    value for value in distribution.requires or [] if not _uses_distribution_extra(value)
                ]
        except importlib.metadata.PackageNotFoundError:
            continue
    project = Path(__file__).resolve().parents[2] / "pyproject.toml"
    if project.is_file():
        return read_recipe_config(project)["project"]["dependencies"]
    raise ValueError("framework dependency metadata is unavailable")


def _runtime(path):
    inspection = inspect_recipe_deployment(path)
    if not inspection["files_available"]:
        raise ValueError("runtime deployment files are unavailable")
    root = Path(inspection["artifact"]).parent
    deployment = _task_deployment(root)
    if deployment is None or "runtime" not in deployment:
        raise ValueError("deployment has no declared runtime profile")
    profile = TaskRuntimeProfile.from_dict(deployment["runtime"]["profile"])
    return root, deployment, profile


def _python_probe(python, names):
    code = """
import json, sys, os, platform, importlib.metadata as metadata
versions = {}
for name in json.loads(sys.argv[1]):
    try:
        versions[name] = metadata.version(name)
    except metadata.PackageNotFoundError:
        versions[name] = None
markers = {
    "implementation_name": sys.implementation.name,
    "implementation_version": platform.python_version(),
    "os_name": os.name, "sys_platform": sys.platform,
    "platform_machine": platform.machine(), "platform_system": platform.system(),
    "platform_release": platform.release(), "platform_version": platform.version(),
    "platform_python_implementation": platform.python_implementation(),
    "python_version": ".".join(platform.python_version().split(".")[:2]),
    "python_full_version": platform.python_version(), "extra": "",
}
print(json.dumps({"python": platform.python_version(), "executable": sys.executable,
                  "base_executable": getattr(sys, "_base_executable", sys.executable),
                  "prefix": sys.prefix, "base_prefix": sys.base_prefix,
                  "platform": platform.platform(), "markers": markers, "versions": versions}))
"""
    result = subprocess.run(
        [str(python), "-c", code, json.dumps(names)], text=True, capture_output=True, check=True
    )
    return json.loads(result.stdout)


def _local_python_launch(observed, environment):
    """Launch the observed interpreter directly, retaining its selected venv.

    Windows venv redirectors spawn the actual interpreter. Killing that
    redirector does not terminate the worker holding the inherited log handles.
    CPython's own launcher environment preserves the selected venv when its
    observed base interpreter is invoked directly instead.
    """
    selected = observed["executable"]
    actual = selected
    environment = dict(environment)
    direct_venv = (
        os.name == "nt"
        and observed.get("markers", {}).get("implementation_name") == "cpython"
        and observed.get("prefix") is not None
        and observed.get("base_prefix") is not None
        and observed["prefix"] != observed["base_prefix"]
    )
    if direct_venv:
        actual = observed.get("base_executable")
        if not isinstance(actual, str) or not Path(actual).is_file():
            raise ValueError("observed Windows venv base executable is unavailable; runtime was not launched")
        environment["__PYVENV_LAUNCHER__"] = selected
    return actual, environment, {
        "selected_executable": selected,
        "process_executable": actual,
        "venv_redirector_bypassed": direct_venv,
        "selected_prefix": observed.get("prefix"),
    }


def _compatibility(profile, requirements, environment):
    names = sorted({canonicalize_name(Requirement(value).name) for value in requirements})
    if (
        not isinstance(environment, Mapping)
        or not isinstance(environment.get("python"), str)
        or not isinstance(environment.get("markers"), Mapping)
        or not isinstance(environment.get("versions"), Mapping)
        or any(name not in environment["versions"] for name in names)
    ):
        raise ValueError("backend must observe Python, markers and all requested distributions")
    missing, incompatible = [], []
    for value in requirements:
        requirement = Requirement(value)
        if requirement.marker is not None and not requirement.marker.evaluate(
            environment["markers"]
        ):
            continue
        version = environment["versions"][canonicalize_name(requirement.name)]
        if version is None:
            missing.append(value)
        elif requirement.url is not None:
            incompatible.append(f"direct URL origin not verified: {value}")
        elif not requirement.specifier.contains(version):
            incompatible.append(value)
    if not SpecifierSet(profile.python).contains(environment["python"]):
        incompatible.append(f"Python {environment['python']} does not satisfy {profile.python}")
    if profile.allowed_dependencies is not None:
        forbidden = sorted(set(names) - set(profile.allowed_dependencies))
        if forbidden:
            incompatible.append(
                f"required runtime distributions outside allowed_dependencies: {forbidden}"
            )
    return missing, incompatible


def _inspect_local_runtime(root, deployment, profile, python):
    requirements = _requirements(
        [*_framework_requirements(), *deployment["python_requirements"], *profile.requirements]
    )
    names = sorted({canonicalize_name(Requirement(value).name) for value in requirements})
    environment = _python_probe(python or sys.executable, names)
    missing, incompatible = _compatibility(profile, requirements, environment)
    unsupported = sorted(set(profile.require_enforcement) - {"process_wall_seconds"})
    if profile.container is not None:
        unsupported.append("container execution requires a caller-selected host backend")
    result = {
        "ok": not missing and not incompatible and not unsupported,
        "kind": "task-runtime-inspection",
        "format": "arti.task-runtime-inspection",
        "format_version": 1,
        "status": "compatible"
        if not missing and not incompatible and not unsupported
        else "incompatible",
        "deployment_sha256": deployment["sha256"],
        "profile_sha256": json_fingerprint(profile.to_dict()),
        "declared": profile.to_dict(),
        "observed": environment,
        "requirements": requirements,
        "missing_dependencies": missing,
        "incompatible_dependencies": incompatible,
        "unsupported_enforcement": unsupported,
        "backend": {
            "name": "local-subprocess",
            "capabilities": ["process_wall_seconds"],
            "network_isolation": False,
            "memory_cap": False,
            "process_tree_isolation": False,
            "container_verification": "not_run",
        },
        "loads_model": False,
        "imports_application": False,
        "installs_dependencies": False,
    }
    return result


def _backend_context(root, deployment, profile, python, *, bindings=None, options=None):
    return TaskRuntimeBackendContext(
        root,
        deployment["sha256"],
        profile,
        tuple(
            _requirements(
                [
                    *_framework_requirements(),
                    *deployment["python_requirements"],
                    *profile.requirements,
                ]
            )
        ),
        python,
        {} if bindings is None else bindings["inputs"],
        {} if bindings is None else bindings["outputs"],
        {**profile.options, **_json(dict(options or {}))},
    )


def _selected_local_observation(root, deployment, profile, python):
    """A failed controller probe is evidence, not admission for a different host."""
    try:
        return _inspect_local_runtime(root, deployment, profile, python)
    except (OSError, subprocess.SubprocessError, ValueError) as error:
        return {
            "ok": False,
            "kind": "task-runtime-inspection",
            "format": "arti.task-runtime-inspection",
            "format_version": 1,
            "status": "not_run",
            "deployment_sha256": deployment["sha256"],
            "profile_sha256": json_fingerprint(profile.to_dict()),
            "declared": profile.to_dict(),
            "observed": None,
            "requirements": _requirements(
                [
                    *_framework_requirements(),
                    *deployment["python_requirements"],
                    *profile.requirements,
                ]
            ),
            "missing_dependencies": [],
            "incompatible_dependencies": [],
            "unsupported_enforcement": [],
            "local_probe_error": {"type": type(error).__name__, "message": str(error)},
            "loads_model": False,
            "imports_application": False,
            "installs_dependencies": False,
        }


def _inspect_selected_backend(context, local, callback=None):
    result = {
        **local,
        "ok": False,
        "status": "not_run",
        "observed": None,
        "local_evidence": {
            "origin": "local-interpreter-probe",
            "status": "unavailable" if "local_probe_error" in local else "observed",
            "observed": local["observed"],
            **({"error": local["local_probe_error"]} if "local_probe_error" in local else {}),
        },
        "evidence_origin": "backend-reported",
        "origin": "backend-reported",
        "imports_backend": False,
        "installs_dependencies": None,
        "core_installs_dependencies": False,
        "trust": "explicit installed caller-owned host code",
        "backend": {"declaration": context.profile.backend.to_dict(), "status": "not_run"},
    }
    try:
        result["imports_backend"] = True
        callback = callback or context.profile.backend.factory.resolve()
        observation = _json(callback(context, operation="inspect"))
        if not isinstance(observation, dict) or type(observation.get("available")) is not bool:
            raise ValueError("backend inspection must report actual availability")
        capabilities = observation.get("capabilities", [])
        if not isinstance(capabilities, list) or any(
            not isinstance(name, str) or not name for name in capabilities
        ):
            raise ValueError("backend capabilities must be explicitly named")
        missing, incompatible = (
            _compatibility(context.profile, context.requirements, observation.get("environment"))
            if observation["available"]
            else ([], [])
        )
        unsupported = sorted(set(context.profile.require_enforcement) - set(capabilities))
        if context.profile.container is not None and (
            observation.get("container_verification") != "verified"
            or observation.get("container_identity") != context.profile.container
        ):
            unsupported.append("container execution was not verified by the selected backend")
        compatible = (
            observation["available"] and not missing and not incompatible and not unsupported
        )
        result.update(
            ok=compatible,
            status="compatible" if compatible else "incompatible",
            observed=observation.get("environment"),
            installs_dependencies=observation.get("installs_dependencies"),
            missing_dependencies=missing,
            incompatible_dependencies=incompatible,
            unsupported_enforcement=unsupported,
            backend={"declaration": context.profile.backend.to_dict(), "observation": observation},
        )
    except Exception as error:
        result["backend_error"] = {"type": type(error).__name__, "message": str(error)}
    return result, callback


def inspect_task_runtime(path, *, python=None, output=None, inspect_backend=False):
    """Code-free local/default inspection; explicit backend inspection imports trusted code."""
    root, deployment, profile = _runtime(path)
    result = (
        _inspect_local_runtime(root, deployment, profile, python)
        if profile.backend is None
        else _selected_local_observation(root, deployment, profile, python)
    )
    if profile.backend is not None:
        if inspect_backend:
            result, _callback = _inspect_selected_backend(
                _backend_context(root, deployment, profile, python), result
            )
        else:
            result = {
                **result,
                "ok": False,
                "status": "not_run",
                "imports_backend": False,
                "backend": {"declaration": profile.backend.to_dict(), "status": "not_run"},
                "evidence_origin": "local-interpreter-probe; selected backend not inspected",
                "local_evidence": {
                    "origin": "local-interpreter-probe",
                    "status": "unavailable" if "local_probe_error" in result else "observed",
                    "observed": result["observed"],
                    **(
                        {"error": result["local_probe_error"]}
                        if "local_probe_error" in result
                        else {}
                    ),
                },
            }
    if output is not None:
        target = Path(output).resolve()
        if target.is_relative_to(root):
            raise ValueError("runtime inspection must be outside its immutable package")
        _publish(target, result)
    return result


def _io(profile, inputs, outputs, root):
    values = {}
    for scope, specifications, supplied in (
        ("inputs", profile.inputs, inputs),
        ("outputs", profile.outputs, outputs),
    ):
        if set(supplied) != set(specifications):
            raise ValueError(f"runtime {scope} must bind exactly the declared ports")
        values[scope] = {name: Path(path).resolve() for name, path in supplied.items()}
        for name, path in values[scope].items():
            if scope == "inputs":
                if not (path.is_file() if specifications[name] == "file" else path.is_dir()):
                    raise ValueError(f"runtime input {name!r} has missing/wrong file kind")
            elif path.is_relative_to(root):
                raise ValueError("runtime outputs must not mutate the deployment package")
            elif path.exists() and not (
                path.is_file() if specifications[name] == "file" else path.is_dir()
            ):
                raise ValueError(f"runtime output {name!r} has wrong file kind")
    for output in values["outputs"].values():
        if any(
            output == path or (path.is_dir() and output.is_relative_to(path))
            for path in values["inputs"].values()
        ):
            raise ValueError("runtime outputs must be separate from input bindings")
    return values


def _worker(request_path, response_path):
    request = _load_json(Path(request_path))
    root, deployment, profile = _runtime(request["package"])
    if deployment["sha256"] != request["deployment_sha256"]:
        raise ValueError("runtime package changed before worker admission")
    context = TaskRuntimeContext(
        root,
        {name: Path(path) for name, path in request["inputs"].items()},
        {name: Path(path) for name, path in request["outputs"].items()},
        request["options"],
        profile,
    )
    response = {
        "pid": os.getpid(),
        "python": platform.python_version(),
        "phase": "entrypoint",
        "entrypoint": profile.entrypoint.to_dict(),
        "entrypoint_revision": profile.revision,
        "application_executed": False,
    }
    _atomic_json(Path(response_path), response)
    try:
        entrypoint = profile.entrypoint.resolve()
        response["application_executed"] = True
        _atomic_json(Path(response_path), response)
        value = entrypoint(context)
        response.update(status="completed", result=_json(value))
    except Exception as error:
        response.update(
            status="failed", error={"type": type(error).__name__, "message": str(error)}
        )
    _atomic_json(Path(response_path), response)


def _run_local_task_runtime(path, *, inputs, outputs, python=None, options=None, output=None):
    """Cold trusted application process; underlying task journals remain authoritative."""
    root, deployment, profile = _runtime(path)
    bindings = _io(profile, dict(inputs), dict(outputs), root)
    inspection = _inspect_local_runtime(root, deployment, profile, python)
    if output is not None:
        report_path = Path(output).resolve()
        if report_path.is_relative_to(root) or any(
            report_path == path or report_path.is_relative_to(path)
            for mapping in bindings.values()
            for path in mapping.values()
        ):
            raise ValueError("runtime receipt must be outside package and IO bindings")
    else:
        report_path = None
    if not inspection["ok"]:
        report = {
            "ok": False,
            "kind": "task-runtime-execution",
            "status": "not_run",
            "inspection": inspection,
            "application_executed": False,
            "installs_dependencies": False,
        }
        return (
            {**report, "sha256": json_fingerprint(report)}
            if report_path is None
            else _publish(report_path, report)
        )
    options = {**profile.options, **_json(dict(options or {}))}
    profile_environment = {**os.environ, **profile.environment}
    code_roots = [
        str(_resource_path(root, item["path"]))
        if item["kind"] == "directory"
        else str(_resource_path(root, item["path"]).parent)
        for item in deployment["runtime"]["code"].values()
    ]
    framework = str(Path(__file__).resolve().parent.parent)
    profile_environment["PYTHONPATH"] = os.pathsep.join([framework, *code_roots])
    profile_environment["PYTHONDONTWRITEBYTECODE"] = "1"
    if profile.network == "disabled":
        profile_environment.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    timeout = profile.limits.get("process_wall_seconds")
    executable, profile_environment, launch = _local_python_launch(
        inspection["observed"], profile_environment
    )
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="arti-runtime-") as temporary:
        workspace = Path(temporary)
        request, response = workspace / "request.json", workspace / "response.json"
        _atomic_json(
            request,
            {
                "package": str(root),
                "deployment_sha256": deployment["sha256"],
                "inputs": {name: str(path) for name, path in bindings["inputs"].items()},
                "outputs": {name: str(path) for name, path in bindings["outputs"].items()},
                "options": options,
            },
        )
        command = [
            executable,
            "-c",
            "from arti.recipe_runtime import _worker;import sys;_worker(sys.argv[1],sys.argv[2])",
            str(request),
            str(response),
        ]
        stdout, stderr = workspace / "stdout.log", workspace / "stderr.log"
        with stdout.open("wb") as out, stderr.open("wb") as err:
            process = subprocess.Popen(
                command,
                cwd=workspace,
                env=profile_environment,
                stdout=out,
                stderr=err,
                **({"creationflags": subprocess.CREATE_NO_WINDOW} if os.name == "nt" else {}),
            )
            terminated = False
            try:
                process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
                terminated = True
            except BaseException:
                process.kill()
                process.wait()
                raise
        worker = _load_json(response) if response.is_file() else None
        success = (
            not terminated
            and process.returncode == 0
            and worker is not None
            and worker.get("status") == "completed"
        )
        report = {
            "ok": success,
            "kind": "task-runtime-execution",
            "format": "arti.task-runtime-execution",
            "format_version": 1,
            "status": "completed" if success else "terminated_by_host" if terminated else "failed",
            "inspection": inspection,
            "application_executed": worker is not None
            and worker.get("application_executed", False),
            "runtime_identity": {
                "interpreter": inspection["observed"],
                "framework_root": framework,
                "code_files": {
                    file: value
                    for file, value in deployment["files"].items()
                    if file.startswith("code/")
                },
            },
            "io": {
                name: {port: str(path) for port, path in mapping.items()}
                for name, mapping in bindings.items()
            },
            "options_sha256": json_fingerprint(options),
            "worker": worker,
            "observed": {
                "launch": launch,
                "elapsed_seconds": time.monotonic() - started,
                "pid": process.pid,
                "returncode": process.returncode,
            },
            "enforced": {
                "process_wall_seconds": {
                    "value": timeout,
                    "scope": "worker after process creation; descendants/startup probe excluded",
                    "triggered": terminated,
                }
            }
            if timeout is not None
            else {},
            "unenforced": {
                key: value for key, value in profile.limits.items() if key != "process_wall_seconds"
            },
            "network_enforced": False,
            "container_verification": "not_run",
            "installs_dependencies": False,
            "diagnostics": {
                "stdout_tail": _log_tail(stdout),
                "stderr_tail": _log_tail(stderr),
            },
        }
    return (
        {**report, "sha256": json_fingerprint(report)}
        if report_path is None
        else _publish(report_path, report)
    )


def run_task_runtime(path, *, inputs, outputs, python=None, options=None, output=None):
    """Run the original local worker or an explicitly selected trusted host callback."""
    root, deployment, profile = _runtime(path)
    if profile.backend is None:
        return _run_local_task_runtime(
            path, inputs=inputs, outputs=outputs, python=python, options=options, output=output
        )
    bindings = _io(profile, dict(inputs), dict(outputs), root)
    report_path = None if output is None else Path(output).resolve()
    if report_path is not None and (
        report_path.is_relative_to(root)
        or any(
            report_path == path or report_path.is_relative_to(path)
            for mapping in bindings.values()
            for path in mapping.values()
        )
    ):
        raise ValueError("runtime receipt must be outside package and IO bindings")
    context = _backend_context(
        root, deployment, profile, python, bindings=bindings, options=options
    )
    local = _selected_local_observation(root, deployment, profile, python)
    inspection, callback = _inspect_selected_backend(context, local)
    report = {
        "ok": False,
        "kind": "task-runtime-execution",
        "format": "arti.task-runtime-execution",
        "format_version": 1,
        "status": "not_run",
        "inspection": inspection,
        "application_executed": False,
        "backend_identity": profile.backend.to_dict(),
        "deployment_sha256": deployment["sha256"],
        "profile_sha256": json_fingerprint(profile.to_dict()),
        "io": {
            name: {port: str(path) for port, path in mapping.items()}
            for name, mapping in bindings.items()
        },
        "options_sha256": json_fingerprint(context.options),
        "evidence_origin": "backend-reported; trusted caller code",
        "origin": "backend-reported",
        "locally_verified_execution": False,
        "core_installs_dependencies": False,
        "installs_dependencies": inspection.get("installs_dependencies"),
    }
    if inspection["ok"]:
        try:
            executed = _json(callback(context, operation="run"))
            if not isinstance(executed, dict) or executed.get("status") not in {
                "completed",
                "failed",
                "terminated_by_host",
                "not_run",
            }:
                raise ValueError(
                    "backend execution must return a runtime receipt with explicit status"
                )
            report["backend_report"] = executed
            if executed.get("inspection", {}).get("deployment_sha256") != deployment["sha256"]:
                raise ValueError("backend execution receipt describes a different package")
            if executed.get("status") != "not_run" and (
                executed.get("runtime_identity", {}).get("interpreter") != inspection["observed"]
            ):
                raise ValueError(
                    "backend execution environment differs from its admission observation"
                )
            enforced = executed.get("enforced", {})
            if not isinstance(enforced, dict):
                raise ValueError("backend enforced limits must have explicit applied receipts")
            for name, receipt in enforced.items():
                declared = "disabled" if name == "network" else profile.limits.get(name)
                if (
                    declared is None
                    or not isinstance(receipt, dict)
                    or json_fingerprint(receipt.get("value")) != json_fingerprint(declared)
                    or not isinstance(receipt.get("scope"), str)
                    or not receipt["scope"]
                ):
                    raise ValueError(
                        "backend enforcement receipt does not describe a declared limit/scope"
                    )
            if ("network" in enforced) != (executed.get("network_enforced") is True):
                raise ValueError("backend network enforcement needs consistent applied evidence")
            if executed.get("status") != "not_run" and (
                set(profile.require_enforcement) - set(enforced)
                or profile.container is not None
                and executed.get("container_verification") != "verified"
                or profile.container is not None
                and executed.get("container_identity") != profile.container
            ):
                raise ValueError("backend execution did not supply required enforcement evidence")
            if executed.get("status") == "completed" and (
                executed.get("application_executed") is not True
                or not isinstance(executed.get("observed"), dict)
                or not executed["observed"]
            ):
                raise ValueError(
                    "backend completion needs actual application execution observations"
                )
            success = executed.get("status") == "completed" and executed.get("ok") is True
            report.update(
                ok=success,
                status=executed["status"]
                if executed["status"] != "completed" or success
                else "failed",
                application_executed=executed.get("application_executed"),
                worker=executed.get("worker"),
                observed=executed.get("observed"),
                runtime_identity=executed.get("runtime_identity"),
                enforced={
                    name: {**receipt, "evidence_origin": "backend-reported"}
                    for name, receipt in enforced.items()
                },
                unenforced={
                    name: value for name, value in profile.limits.items() if name not in enforced
                },
                network_enforced=executed.get("network_enforced", False),
                container_verification=executed.get("container_verification", "not_run"),
                installs_dependencies=executed.get("installs_dependencies"),
            )
        except Exception as error:
            report.update(
                status="failed",
                application_executed=None,
                execution_outcome="unknown; backend failure may follow external effects",
                backend_error={"type": type(error).__name__, "message": str(error)},
                recovery="caller decision required; no automatic repeat",
                installs_dependencies=report.get("backend_report", {}).get("installs_dependencies"),
            )
    return (
        {**report, "sha256": json_fingerprint(report)}
        if report_path is None
        else _publish(report_path, report)
    )


__all__ = [
    "TaskRuntimeBackendConfig",
    "TaskRuntimeBackendContext",
    "TaskRuntimeProfile",
    "TaskRuntimeContext",
    "inspect_task_runtime",
    "run_task_runtime",
]
