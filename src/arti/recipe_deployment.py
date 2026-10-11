"""Local native-base handoff using the existing ARTI deployment protocol."""

from __future__ import annotations

import hashlib
from email.parser import BytesParser
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath, PureWindowsPath
import tempfile
import zipfile

from .attachment_hub import DEFAULT_ARTIFACT, HUB_MANIFEST, _read_manifest
from .recipe_chat import tokenizer_fingerprint
from .recipe_data import json_fingerprint
from .serialization import (
    _atomic_json,
    _file_sha256,
    _load_json,
    _manifest_file_records,
    _member_path,
    _read_and_validate_package,
    _replace_atomic_file,
    _sidecar_paths,
    _weight_path,
)


def _wheel_identity(path):
    from packaging.utils import parse_wheel_filename, canonicalize_name

    name, version, _build, tags = parse_wheel_filename(Path(path).name)
    with zipfile.ZipFile(path) as archive:
        metadata = [
            member for member in archive.namelist() if member.endswith(".dist-info/METADATA")
        ]
        if len(metadata) != 1:
            raise ValueError("offline wheel must contain one distribution metadata record")
        message = BytesParser().parsebytes(archive.read(metadata[0]))
        if canonicalize_name(message["Name"]) != name or message["Version"] != str(version):
            raise ValueError("offline wheel filename and metadata identity differ")
    return {
        "name": str(name),
        "version": str(version),
        "tags": sorted(str(tag) for tag in tags),
        "installed_automatically": False,
    }


_TOKENIZER_FILES = {
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
    "chat_template.jinja",
}


def _artifact(path):
    target = Path(path).resolve()
    if target.is_dir():
        deployment = _task_deployment(target)
        target = (
            _member_path(target, deployment["artifact"])
            if deployment is not None
            else target / (DEFAULT_ARTIFACT if (target / DEFAULT_ARTIFACT).exists() else "model.st")
        )
    return _weight_path(target)


def _task_deployment(directory):
    path = Path(directory) / "deployment.json"
    if not path.is_file():
        return None
    value = _load_json(path)
    if value.get("format") == "arti.causal-lm-deployment":
        return None
    if value.get("format") != "arti.task-deployment" or value.get("format_version") != 1:
        raise ValueError("unsupported task deployment metadata")
    if value.get("sha256") != json_fingerprint(
        {key: item for key, item in value.items() if key != "sha256"}
    ):
        raise ValueError("task deployment fingerprint differs")
    return value


def _resource_path(root, name):
    if not isinstance(name, str) or not name or "\\" in name:
        raise ValueError("deployment resource paths must be relative POSIX paths")
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or path.as_posix() != name:
        raise ValueError("deployment resource paths must be relative POSIX paths")
    result = Path(root).joinpath(*path.parts)
    if not result.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError("deployment resource path leaves its package")
    return result


def _task_package(artifact, manifest):
    deployment = _task_deployment(artifact.parent)
    if deployment is None:
        return None, {}, []
    declaration = manifest["architecture"]["config"]["task_recipe"]
    if (
        deployment.get("artifact") != artifact.name
        or deployment.get("task_recipe_sha256") != declaration["sha256"]
        or deployment.get("model_manifest_sha256") != json_fingerprint(manifest)
    ):
        raise ValueError("task deployment does not describe this inference asset")
    missing = []
    for name, record in deployment["files"].items():
        path = _resource_path(artifact.parent, name)
        if not path.is_file():
            missing.append(name)
        elif path.stat().st_size != record["bytes"] or _file_sha256(path) != record["sha256"]:
            missing.append(f"{name} (checksum differs)")
    bindings = {}
    collections = [
        deployment["resources"],
        *[deployment.get("runtime", {}).get(name, {}) for name in ("code", "wheels")],
    ]
    for index, collection in enumerate(collections):
        for name, resource in collection.items():
            path = _resource_path(artifact.parent, resource["path"])
            if index == 0:
                bindings[name] = str(path.resolve())
            if resource["kind"] == "directory":
                if not path.is_dir():
                    missing.append(resource["path"])
                else:
                    files = {
                        item.relative_to(artifact.parent).as_posix()
                        for item in path.rglob("*")
                        if item.is_file()
                    }
                    directories = {
                        item.relative_to(artifact.parent).as_posix()
                        for item in path.rglob("*")
                        if item.is_dir()
                    }
                    if files != set(resource["files"]) or directories != set(
                        resource["directories"]
                    ):
                        missing.append(f"{resource['path']} (directory inventory differs)")
            elif resource["kind"] != "file" or not path.is_file():
                missing.append(resource["path"])
    return deployment, bindings, missing


def local_task_resources(artifact, manifest):
    deployment, bindings, missing = _task_package(Path(artifact).resolve(), manifest)
    if missing:
        raise ValueError(f"task deployment dependencies unavailable: {missing}")
    return bindings, None if deployment is None else deployment["sha256"]


def _declaration(path, *, require_inference=True):
    artifact = _artifact(path)
    manifest, _lock = _read_and_validate_package(
        artifact, _sidecar_paths(artifact), allow_legacy=False
    )
    declaration = manifest["architecture"]["config"].get("task_recipe", {})
    if (
        declaration.get("format") != "arti.causal-lm-recipe"
        or declaration.get("format_version") != 1
    ):
        raise ValueError("deployment inspection requires a causal-LM recipe inference asset")
    if declaration.get("sha256") != json_fingerprint(
        {key: value for key, value in declaration.items() if key != "sha256"}
    ):
        raise ValueError("causal-LM task declaration fingerprint differs")
    if require_inference and manifest.get("checkpoint") is not None:
        raise ValueError(
            "export an inference asset before packaging; training checkpoints are separate"
        )
    return artifact, manifest, declaration


def _deployment(artifact, declaration):
    path = artifact.parent / "deployment.json"
    if not path.exists():
        return {}
    value = _load_json(path)
    if (
        value.get("format") != "arti.causal-lm-deployment"
        or value.get("format_version") != 1
        or value.get("task_recipe_sha256") != declaration["sha256"]
    ):
        raise ValueError("deployment metadata does not describe this causal-LM artifact")
    if value.get("artifact") != artifact.name:
        raise ValueError("deployment artifact name differs")
    return value


def _base_directory(artifact, declaration, deployment, override=None):
    if override is not None:
        return Path(override).resolve(), "caller-local"
    local = deployment.get("base_model_directory")
    if local is not None:
        return _member_path(artifact.parent, local), "bundled"
    from huggingface_hub import try_to_load_from_cache

    model = declaration["base_model"]
    source = model["id"]
    if Path(source).is_dir():
        return Path(source).resolve(), "declared-local"
    try:
        config = try_to_load_from_cache(source, "config.json", revision=model["revision"])
    except ValueError:
        config = None
    return (
        (Path(config).parent, "hub-cache")
        if isinstance(config, str)
        else (None, "external-reference")
    )


def _native_files(directory):
    if directory is None or not directory.is_dir():
        return [], ["base_model/config.json", "base_model/SafeTensors weights"]
    files, missing = [], []
    config = directory / "config.json"
    if config.is_file():
        _load_json(config)
        files.append(config)
    else:
        missing.append("base_model/config.json")
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        mapping = _load_json(index).get("weight_map")
        if not isinstance(mapping, dict) or not mapping:
            raise ValueError("native SafeTensors index requires a nonempty weight_map")
        files.append(index)
        members = [_member_path(directory, name) for name in sorted(set(mapping.values()))]
    else:
        members = [directory / "model.safetensors"]
    for member in members:
        if member.is_file():
            files.append(member)
        else:
            missing.append(f"base_model/{member.name}")
    generation = directory / "generation_config.json"
    if generation.is_file():
        files.append(generation)
    return files, missing


def _tokenizer_files(root):
    files = [root / name for name in sorted(_TOKENIZER_FILES) if (root / name).is_file()]
    templates = root / "chat_templates"
    if templates.is_dir():
        files.extend(sorted(templates.glob("*.jinja")))
    return files


def _check_tokenizer(root, declaration):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(root, local_files_only=True, use_fast=True)
    if tokenizer_fingerprint(tokenizer) != declaration["tokenizer_sha256"]:
        raise ValueError("deployment tokenizer differs from its task declaration")


def inspect_causal_lm_deployment(path, *, base_model=None, output=None):
    """Check dependency files without constructing a host or reading its tensors."""
    artifact, manifest, declaration = _declaration(path)
    deployment = _deployment(artifact, declaration)
    missing = []
    hub = artifact.parent / HUB_MANIFEST
    if hub.is_file():
        _read_manifest(artifact.parent)
    else:
        missing.append(HUB_MANIFEST)
    tokenizer = artifact.parent / "tokenizer"
    if tokenizer.is_dir():
        _check_tokenizer(tokenizer, declaration)
    else:
        missing.append("tokenizer/")
    base, source = _base_directory(artifact, declaration, deployment, base_model)
    native, absent = _native_files(base)
    missing.extend(absent)
    records = [{"file": file.name, "bytes": file.stat().st_size} for file in native]
    expected = deployment.get("base_files", {}) if source == "bundled" else {}
    for record in records:
        previous = expected.get(record["file"])
        if previous is not None and previous["bytes"] != record["bytes"]:
            missing.append(f"base_model/{record['file']} (size differs)")
    result = {
        "ok": not missing,
        "kind": "causal-lm-deployment-inspection",
        "files_available": not missing,
        "missing_dependencies": missing,
        "task_recipe_sha256": declaration["sha256"],
        "artifact": str(artifact),
        "base_model": {
            "id": declaration["base_model"]["id"],
            "revision": declaration["base_model"]["revision"],
        },
        "base_source": source,
        "base_directory": None if base is None else str(base),
        "base_files": records,
        "base_bytes": sum(item["bytes"] for item in records),
        "requires_layer_factory": declaration.get("adapter_factory") == "caller-supplied",
        "layer_factory": declaration.get("layer_factory"),
        "base_weights_included": bool(deployment.get("base_weights_included", False)),
        "runtime_reload_verified": False,
        "task_quality_evaluated": False,
        "loads_base_model": False,
        "loads_base_weight_tensors": False,
        "evaluation": manifest["architecture"]["config"].get("evaluation", {}),
        "batching": manifest["architecture"]["config"].get("batching", {}),
        "selection": deployment.get("selection"),
    }
    if output is not None:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(Path(output), result)
    return result


def _copy_file(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with source.open("rb") as reader, destination.open("wb") as writer:
        for chunk in iter(lambda: reader.read(1024 * 1024), b""):
            writer.write(chunk)
            digest.update(chunk)
    return {"sha256": digest.hexdigest(), "bytes": destination.stat().st_size}


def package_causal_lm_deployment(path, directory, *, include_base=False, base_model=None):
    """Publish a fresh movable package; never export a patched host as native base."""
    if type(include_base) is not bool:
        raise ValueError("include_base must be boolean")
    artifact, manifest, declaration = _declaration(path)
    deployment = _deployment(artifact, declaration)
    if not deployment:
        raise ValueError("package a complete recipe export with deployment.json")
    if (
        Path(declaration["base_model"]["id"]).is_absolute()
        or PureWindowsPath(declaration["base_model"]["id"]).is_absolute()
    ):
        raise ValueError("a portable package requires a logical base model ID, not a machine path")
    _read_manifest(artifact.parent)
    _check_tokenizer(artifact.parent / "tokenizer", declaration)
    base, source = _base_directory(artifact, declaration, deployment, base_model)
    native, missing = _native_files(base)
    if include_base and missing:
        raise ValueError(f"offline packaging requires complete native SafeTensors files: {missing}")
    target = Path(directory).resolve()
    if target.exists():
        raise FileExistsError("deployment package needs a fresh output directory")
    target.parent.mkdir(parents=True, exist_ok=True)
    hub = _read_manifest(artifact.parent)
    members = {record["file"] for record in _manifest_file_records(manifest)}
    sidecars = _sidecar_paths(artifact)
    members.update({sidecars["manifest"].name, sidecars["lock"].name})
    members.update({HUB_MANIFEST, hub["config"]["file"], hub["lock"]["file"]})
    with tempfile.TemporaryDirectory(prefix=".arti-deployment-", dir=target.parent) as temporary:
        staging = Path(temporary) / "package"
        staging.mkdir()
        for name in sorted(members):
            _copy_file(_member_path(artifact.parent, name), _member_path(staging, name))
        for file in _tokenizer_files(artifact.parent / "tokenizer"):
            _copy_file(
                file, staging / "tokenizer" / file.relative_to(artifact.parent / "tokenizer")
            )
        packaged = dict(deployment)
        packaged.pop("base_model_directory", None)
        packaged.pop("base_files", None)
        packaged["base_weights_included"] = include_base
        if include_base:
            packaged["base_model_directory"] = "base_model"
            packaged["base_files"] = {
                file.name: _copy_file(file, staging / "base_model" / file.name) for file in native
            }
        _atomic_json(staging / "deployment.json", packaged)
        _check_tokenizer(staging / "tokenizer", declaration)
        _replace_atomic_file(staging, target)
    return {
        "ok": True,
        "kind": "causal-lm-deployment-package",
        "directory": str(target),
        "artifact": str(target / artifact.name),
        "base_weights_included": include_base,
        "base_source": source,
        "task_recipe_sha256": declaration["sha256"],
        "runtime_reload_verified": False,
    }


def local_deployment_base(artifact, declaration):
    deployment = _deployment(Path(artifact), declaration)
    if deployment.get("base_model_directory") is None:
        return None
    base, _source = _base_directory(Path(artifact), declaration, deployment)
    _files, missing = _native_files(base)
    if missing:
        raise ValueError(f"bundled base dependency files are unavailable: {missing}")
    for file in _files:
        expected = deployment.get("base_files", {}).get(file.name)
        if expected is not None and expected["bytes"] != file.stat().st_size:
            raise ValueError(f"bundled base dependency size differs: {file.name}")
    return base


def _recipe_declaration(path):
    artifact = _artifact(path)
    manifest, _lock = _read_and_validate_package(
        artifact, _sidecar_paths(artifact), allow_legacy=False
    )
    declaration = manifest["architecture"]["config"].get("task_recipe", {})
    if (
        declaration.get("format")
        not in {"arti.task-recipe", "arti.classification-recipe", "arti.causal-lm-recipe"}
        or declaration.get("format_version") != 1
    ):
        raise ValueError("deployment requires a supported task recipe inference asset")
    if declaration.get("sha256") != json_fingerprint(
        {key: value for key, value in declaration.items() if key != "sha256"}
    ):
        raise ValueError("task declaration fingerprint differs")
    if manifest.get("checkpoint") is not None:
        raise ValueError(
            "export an inference asset before packaging; training checkpoints are separate"
        )
    return artifact, manifest, declaration


def inspect_recipe_deployment(path, *, base_model=None, output=None):
    """Inspect native task files; this does not execute or install task code."""
    artifact, manifest, declaration = _recipe_declaration(path)
    if declaration["format"] == "arti.causal-lm-recipe":
        return inspect_causal_lm_deployment(artifact, base_model=base_model, output=output)
    if base_model is not None:
        raise ValueError("base_model applies only to native causal-LM deployments")
    deployment, bindings, missing = _task_package(artifact, manifest)
    result = {
        "ok": not missing,
        "kind": "task-deployment-inspection",
        "artifact": str(artifact),
        "task": declaration["task"],
        "task_recipe_sha256": declaration["sha256"],
        "model_manifest_sha256": json_fingerprint(manifest),
        "deployment_sha256": None if deployment is None else deployment["sha256"],
        "packaged": deployment is not None,
        "files_available": not missing,
        "missing_dependencies": missing,
        "resources": bindings,
        "python_requirements": [] if deployment is None else deployment["python_requirements"],
        "task_factory": declaration.get("factory"),
        "requires_task_factory": declaration["format"] == "arti.task-recipe",
        "selection": manifest["architecture"]["config"].get("selection"),
        "runtime_reload_verified": False,
        "task_quality_evaluated": False,
        "code_dependencies_verified": False,
        **(
            {"runtime": deployment["runtime"]}
            if deployment is not None and "runtime" in deployment
            else {}
        ),
        "loads_model": False,
        "loads_weight_tensors": False,
    }
    if output is not None:
        target = Path(output)
        protected = {
            artifact.parent / record["file"] for record in _manifest_file_records(manifest)
        } | {*_sidecar_paths(artifact).values(), artifact.parent / "deployment.json"}
        if target.resolve() in protected or target.resolve().is_relative_to(
            artifact.parent / "resources"
        ):
            raise ValueError("inspection output must not overwrite deployment dependencies")
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(target, result)
    return result


def _requirements(values):
    from packaging.requirements import InvalidRequirement, Requirement

    if isinstance(values, str) or not isinstance(values, Sequence):
        raise ValueError("requirements must be a sequence of Python requirement strings")
    result = []
    for value in values:
        try:
            if not isinstance(value, str):
                raise ValueError("requirements must contain strings")
            Requirement(value)
        except (InvalidRequirement, ValueError) as error:
            raise ValueError(f"invalid Python requirement declaration: {value!r}") from error
        result.append(value)
    return result


def _copy_resources(resources, staging, files, *, prefix="resources"):
    result = {}
    for name, source in resources.items():
        if (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
        ):
            raise ValueError("resource names must be nonempty single path components")
        source = Path(source).resolve()
        if not source.exists():
            raise FileNotFoundError(source)
        root = staging / prefix / name
        if source.is_file():
            destination = root / source.name
            relative = destination.relative_to(staging).as_posix()
            files[relative] = _copy_file(source, destination)
            result[name] = {
                "kind": "file",
                "path": relative,
                "files": [relative],
                "directories": [],
            }
        elif source.is_dir():
            root.mkdir(parents=True)
            resource_files, directories = [], []
            for member in sorted(source.rglob("*")):
                if member.is_symlink() and member.is_dir():
                    raise ValueError(
                        "materialize nested directory symlinks before packaging resources"
                    )
                destination = root / member.relative_to(source)
                relative = destination.relative_to(staging).as_posix()
                if member.is_file():
                    files[relative] = _copy_file(member, destination)
                    resource_files.append(relative)
                elif member.is_dir():
                    destination.mkdir(parents=True, exist_ok=True)
                    directories.append(relative)
            result[name] = {
                "kind": "directory",
                "path": root.relative_to(staging).as_posix(),
                "files": resource_files,
                "directories": directories,
            }
        else:
            raise ValueError("inference resources must be regular files or directories")
    return result


def package_recipe_deployment(
    path,
    directory,
    *,
    resources=None,
    requirements=None,
    include_base=False,
    base_model=None,
    runtime=None,
    code=None,
    wheels=None,
):
    """Publish the declared inference closure with relative resource bindings."""
    artifact, manifest, declaration = _recipe_declaration(path)
    if declaration["format"] == "arti.causal-lm-recipe":
        if (
            resources is not None
            or requirements is not None
            or any(value is not None for value in (runtime, code, wheels))
        ):
            raise ValueError("native causal-LM packages use their existing tokenizer/base contract")
        return package_causal_lm_deployment(
            artifact, directory, include_base=include_base, base_model=base_model
        )
    if type(include_base) is not bool or include_base or base_model is not None:
        raise ValueError("include_base/base_model apply only to native causal-LM deployments")
    previous, bindings, missing = _task_package(artifact, manifest)
    if missing:
        raise ValueError(f"task deployment dependencies unavailable: {missing}")
    resources = bindings if resources is None else resources
    if not isinstance(resources, Mapping):
        raise ValueError("resources must map inference resource names to files/directories")
    if resources and declaration["format"] != "arti.task-recipe":
        raise ValueError("named resources require a generic task predict-mode factory")
    requirements = _requirements(
        previous["python_requirements"]
        if requirements is None and previous is not None
        else []
        if requirements is None
        else requirements
    )
    previous_runtime = {} if previous is None else previous.get("runtime", {})
    if runtime is None and previous_runtime:
        from .recipe_runtime import TaskRuntimeProfile

        runtime = TaskRuntimeProfile.from_dict(previous_runtime["profile"])
    if code is None:
        code = {
            name: _resource_path(artifact.parent, value["path"])
            for name, value in previous_runtime.get("code", {}).items()
        }
    if wheels is None:
        wheels = {
            name: _resource_path(artifact.parent, value["path"])
            for name, value in previous_runtime.get("wheels", {}).items()
        }
    if runtime is None and (code or wheels):
        raise ValueError("code/wheel distribution requires an explicit runtime profile")
    if runtime is not None:
        from .recipe_runtime import TaskRuntimeProfile

        runtime = (
            runtime if isinstance(runtime, TaskRuntimeProfile) else TaskRuntimeProfile.read(runtime)
        )
        if not isinstance(code, Mapping) or not isinstance(wheels, Mapping):
            raise ValueError("code/wheels must be explicitly named file/directory mappings")
        if any(
            not Path(value).is_file() or Path(value).suffix != ".whl" for value in wheels.values()
        ):
            raise ValueError("offline wheels must name existing .whl files; they are not installed")
    target = Path(directory).resolve()
    if target.exists():
        raise FileExistsError("deployment package needs a fresh output directory")
    if any(
        Path(source).is_dir() and target.is_relative_to(Path(source).resolve())
        for source in [*resources.values(), *code.values(), *wheels.values()]
    ):
        raise ValueError("deployment output must be outside its directory resources")
    members = {record["file"] for record in _manifest_file_records(manifest)}
    sidecars = _sidecar_paths(artifact)
    members.update({sidecars["manifest"].name, sidecars["lock"].name})
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".arti-task-deployment-", dir=target.parent
    ) as temporary:
        staging = Path(temporary) / "package"
        staging.mkdir()
        files = {
            name: _copy_file(_member_path(artifact.parent, name), _member_path(staging, name))
            for name in sorted(members)
        }
        resource_records = _copy_resources(resources, staging, files)
        code_records = (
            _copy_resources(code, staging, files, prefix="code") if runtime is not None else {}
        )
        wheel_records = (
            _copy_resources(wheels, staging, files, prefix="wheels") if runtime is not None else {}
        )
        for name, record in wheel_records.items():
            record["distribution"] = _wheel_identity(wheels[name])
        deployment = {
            "format": "arti.task-deployment",
            "format_version": 1,
            "artifact": artifact.name,
            "task_recipe_sha256": declaration["sha256"],
            "model_manifest_sha256": json_fingerprint(manifest),
            "resources": resource_records,
            "files": files,
            "python_requirements": requirements,
            "artifact_package_version": manifest["package_version"],
        }
        if runtime is not None:
            deployment["runtime"] = {
                "profile": runtime.to_dict(),
                "code": code_records,
                "wheels": wheel_records,
            }
        deployment["sha256"] = json_fingerprint(deployment)
        _atomic_json(staging / "deployment.json", deployment)
        if not inspect_recipe_deployment(staging)["ok"]:
            raise ValueError("copied deployment dependencies are incomplete")
        _replace_atomic_file(staging, target)
    return {
        "ok": True,
        "kind": "task-deployment-package",
        "directory": str(target),
        "artifact": str(target / artifact.name),
        "task_recipe_sha256": declaration["sha256"],
        "model_manifest_sha256": json_fingerprint(manifest),
        "deployment_sha256": deployment["sha256"],
        "runtime_reload_verified": False,
    }


__all__ = [
    "inspect_causal_lm_deployment",
    "package_causal_lm_deployment",
    "inspect_recipe_deployment",
    "package_recipe_deployment",
]
