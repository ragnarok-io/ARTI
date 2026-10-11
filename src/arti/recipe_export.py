"""Portable application-owned layouts over already committed task results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import tempfile
import zipfile

from .python_factory import PythonFactoryConfig
from .recipe_assets import _member, _verify_file
from .recipe_data import json_fingerprint
from .recipe_deployment import (
    _copy_file,
    _copy_resources,
    _recipe_declaration,
    inspect_recipe_deployment,
    package_recipe_deployment,
)
from .recipe_files import _declared_files
from .recipe_suite import _population, _binding
from .recipe_workflow import read_recipe_config, write_jsonl
from .serialization import _atomic_json, _load_json, _replace_atomic_file


_META = ".arti-export"


def _json(value):
    return json.loads(json.dumps(value, allow_nan=False))


@dataclass(frozen=True)
class ResultExportConfig:
    writer: PythonFactoryConfig | str
    revision: str
    options: Mapping = field(default_factory=dict)
    validator: PythonFactoryConfig | str | None = None
    validation_id: str | None = None
    archive: str = "directory"
    instructions: str = ""
    unresolved_checks: tuple[str, ...] = ()
    examples: tuple[str, ...] = ()

    def __post_init__(self):
        for name in ("writer", "validator"):
            if isinstance(getattr(self, name), str):
                object.__setattr__(self, name, PythonFactoryConfig(getattr(self, name)))
        if (
            not isinstance(self.writer, PythonFactoryConfig)
            or not isinstance(self.revision, str)
            or not self.revision.strip()
        ):
            raise ValueError("result export requires an installed writer and explicit revision")
        if (self.validator is None) != (self.validation_id is None) or (
            self.validator is not None
            and (
                not isinstance(self.validator, PythonFactoryConfig)
                or not isinstance(self.validation_id, str)
                or not self.validation_id.strip()
            )
        ):
            raise ValueError("result validator and its revision must be supplied together")
        if self.archive not in {"directory", "zip"}:
            raise ValueError("result archive must be directory or zip")
        if not isinstance(self.instructions, str):
            raise ValueError("result instructions must be text supplied by the application")
        object.__setattr__(self, "options", _json(dict(self.options)))
        for name in ("unresolved_checks", "examples"):
            values = getattr(self, name)
            if (
                isinstance(values, str)
                or not isinstance(values, Sequence)
                or any(not isinstance(value, str) or not value for value in values)
            ):
                raise ValueError(f"export {name} must contain nonempty strings")
            object.__setattr__(self, name, tuple(values))

    def to_dict(self):
        return {
            "format": "arti.result-export-config",
            "writer": self.writer.to_dict(),
            "revision": self.revision,
            "options": _json(self.options),
            "validator": None if self.validator is None else self.validator.to_dict(),
            "validation_id": self.validation_id,
            "archive": self.archive,
            "instructions": self.instructions,
            "unresolved_checks": list(self.unresolved_checks),
            "examples": list(self.examples),
        }

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if value.pop("format", "arti.result-export-config") != "arti.result-export-config":
            raise ValueError("not a result export configuration")
        for name in ("writer", "validator"):
            if isinstance(value.get(name), Mapping):
                value[name] = PythonFactoryConfig(**value[name])
        return cls(**value)

    @classmethod
    def read(cls, path):
        return cls.from_dict(read_recipe_config(path))


@dataclass(frozen=True)
class ResultExportContext:
    directory: Path
    source_directory: Path
    options: Mapping


def _layout(rows, expected, producer, staging):
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("writer must return a list of per-ID result declarations")
    identifiers = [row.get("id") for row in rows]
    if (
        any(not isinstance(identifier, str) for identifier in identifiers)
        or len(identifiers) != len(expected)
        or set(identifiers) != set(expected)
    ):
        raise ValueError("writer must cover each expected ID exactly once")
    copied, result = {}, _json(rows)
    for row in result:
        files = row.get("files")
        if not isinstance(files, dict):
            raise ValueError("export result needs a named file mapping")
        for name, declaration in files.items():
            if not isinstance(name, str) or not name:
                raise ValueError("export file names must be nonempty strings")
            declaration = {"file": declaration} if isinstance(declaration, str) else declaration
            if (
                not isinstance(declaration, dict)
                or not isinstance(declaration.get("file"), str)
                or Path(declaration["file"]).is_absolute()
            ):
                raise ValueError("export files must be relative to the writer directory")
            source = _member(producer, declaration["file"])
            relative = source.relative_to(producer).as_posix()
            if relative == _META or relative.startswith(f"{_META}/"):
                raise ValueError("writer paths cannot overwrite export provenance")
            if relative not in copied:
                if not source.is_file():
                    raise ValueError("writer declared a missing result file")
                copied[relative] = _copy_file(source, staging / relative)
            metadata = declaration.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError("export file metadata must be a JSON object")
            files[name] = {"file": relative, **copied[relative], "metadata": metadata}
    _declared_files(result, "files")
    return result


def _copy_native(population, staging):
    root, plan, _inputs, predictions, source, execution = population
    rows = deepcopy(list(predictions.values()))
    copied = {}
    if "files" in plan:
        for file in _declared_files(rows, plan["files"]["field"]):
            relative = f"{_META}/native/{file['file']}"
            actual = _copy_file(_verify_file(root, file), staging / relative)
            if actual != {key: file[key] for key in ("sha256", "bytes")}:
                raise ValueError("native output changed during export")
            copied[file["file"]] = relative
        for row in rows:
            for file in row[plan["files"]["field"]].values():
                file["file"] = copied[file["file"]]
    write_jsonl(staging / _META / "native-results.jsonl", rows)
    copied_input = _copy_file(
        _member(root, plan["input"]["file"]), staging / _META / "inputs.jsonl"
    )
    if copied_input["sha256"] != plan["input"]["sha256"] or (
        copied_input["sha256"] != source["input_sha256"]
    ):
        raise ValueError("pinned input snapshot changed during export")
    _atomic_json(staging / _META / "plan.json", plan)
    _atomic_json(staging / _META / "execution.json", execution)
    return rows


def _source_binding(origin, source, *, suite=False):
    """Match result provenance while keeping legacy score receipts readable."""
    keys = {
        "plan_sha256",
        "model_manifest_sha256",
        "task_recipe_sha256",
        "input_sha256",
        "predictions_sha256",
        "total_records",
        "committed_records",
        "uncommitted_records",
        "conditions",
    }
    if suite or "execution_sha256" in origin:
        keys.update({"execution_sha256", "kind"})
    if "files" in origin or suite and "files" in source:
        keys.add("files")
    if any(key not in origin or origin[key] != source[key] for key in keys) or (
        origin.get("deployment_sha256") != source.get("deployment_sha256")
    ):
        raise ValueError("evaluation evidence describes different result content")


def _evidence_bytes(payload, source):
    value = json.loads(payload)
    if value.get("sha256") != json_fingerprint(
        {key: item for key, item in value.items() if key != "sha256"}
    ):
        raise ValueError("export evidence fingerprint differs")
    format = value.get("format")
    scope = "result-content"
    if format in {"arti.evaluation-suite", "arti.prediction-scores"}:
        _source_binding(value["source"], source, suite=format == "arti.evaluation-suite")
    elif format == "arti.deployment-validation":
        if value["prediction"]["plan_sha256"] != source["plan_sha256"]:
            raise ValueError("validation evidence describes a different result plan")
        if "source" in value:
            _source_binding(value["source"], source)
        else:
            # A stopped or failed validation may establish only the original plan.
            scope = "plan-only"
    elif format == "arti.task-runtime-execution":
        deployment = value.get("inspection", {}).get("deployment_sha256")
        if deployment is None or deployment != source.get("deployment_sha256"):
            raise ValueError("runtime evidence describes a different deployment")
        scope = "deployment-only"
    elif format == "arti.external-evaluation-evidence":
        if value["source"] != _binding(source):
            raise ValueError("external evidence describes different result content")
    else:
        raise ValueError(
            "attach unknown user documents explicitly, not as verified evaluation evidence"
        )
    return {
        "format": format,
        "sha256": value["sha256"],
        "status": value.get("status"),
        "origin": value.get("origin", value.get("evidence_origin", "local")),
        "source_binding": "verified-declared-identities",
        "binding_scope": scope,
        "content": {"sha256": hashlib.sha256(payload).hexdigest(), "bytes": len(payload)},
    }


def _evidence(path, source):
    return _evidence_bytes(Path(path).read_bytes(), source)


def _inventory(root):
    result = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError("result closure must contain materialized files, not links")
        if path.is_file():
            from .serialization import _file_sha256

            result[path.relative_to(root).as_posix()] = {
                "sha256": _file_sha256(path),
                "bytes": path.stat().st_size,
            }
    return result


def _directories(root):
    members = list(root.rglob("*"))
    if any(path.is_symlink() for path in members):
        raise ValueError("result closure must contain materialized files, not links")
    return sorted(path.relative_to(root).as_posix() for path in members if path.is_dir())


def export_prediction_results(
    directory,
    config,
    destination,
    *,
    expected_ids=None,
    evidence=None,
    attachments=None,
    deployment=None,
):
    """Format committed outputs once; publish verified portable closure, never infer."""
    config = config if isinstance(config, ResultExportConfig) else ResultExportConfig.read(config)
    population = _population(directory, False)
    root, plan, inputs, predictions, source, execution = population
    expected = list(inputs) if expected_ids is None else list(expected_ids)
    if (
        not expected
        or any(not isinstance(value, str) or not value for value in expected)
        or len(set(expected)) != len(expected)
        or set(expected) != set(inputs)
    ):
        raise ValueError("expected IDs must match the complete original input population")
    if set(predictions) != set(expected):
        raise ValueError("result export requires a complete committed result for every expected ID")
    if set(config.examples) - set(expected):
        raise ValueError("chosen example IDs must belong to the result population")
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("result export requires a fresh destination")
    if destination.is_relative_to(root):
        raise ValueError("export destination must be outside its original result journal")
    evidence, attachments = dict(evidence or {}), dict(attachments or {})
    if any(
        Path(path).is_dir() and destination.is_relative_to(Path(path).resolve())
        for path in attachments.values()
    ):
        raise ValueError("export destination must be outside directory attachments")
    inspected = None if deployment is None else inspect_recipe_deployment(deployment)
    if inspected is not None and "model_manifest_sha256" not in inspected:
        _artifact, model_manifest, _declaration = _recipe_declaration(deployment)
        inspected = {**inspected, "model_manifest_sha256": json_fingerprint(model_manifest)}
    if inspected is not None and (
        not inspected["files_available"]
        or inspected["model_manifest_sha256"] != source["model_manifest_sha256"]
        or inspected["task_recipe_sha256"] != source["task_recipe_sha256"]
        or inspected.get("deployment_sha256") != source.get("deployment_sha256")
    ):
        raise ValueError("export model deployment differs from the committed results")
    evidence_records = {name: _evidence(path, source) for name, path in evidence.items()}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{destination.name}-", dir=destination.parent
    ) as temporary:
        temporary = Path(temporary)
        producer, staging = temporary / "writer", temporary / "handoff"
        producer.mkdir()
        staging.mkdir()
        items = [
            {
                "id": identifier,
                "input": inputs[identifier],
                "prediction": predictions[identifier],
                "execution": execution[identifier],
            }
            for identifier in expected
        ]
        rows = config.writer.resolve()(
            ResultExportContext(producer, root, _json(config.options)), deepcopy(items)
        )
        rows = _layout(rows, expected, producer, staging)
        validation = None
        if config.validator is not None:
            validation = config.validator.resolve()(deepcopy(rows), directory=staging)
            if not isinstance(validation, Mapping) or validation.get("ok") is not True:
                raise ValueError("export layout validator rejected the formatted result")
            validation = _json(dict(validation))
        for file in _declared_files(rows, "files"):
            _verify_file(staging, file)
        native = _copy_native(population, staging)
        write_jsonl(staging / _META / "formatted-results.jsonl", rows)
        _atomic_json(
            staging / _META / "source.json",
            {key: value for key, value in source.items() if key != "directory"},
        )
        (staging / _META / "instructions.txt").write_text(config.instructions, encoding="utf-8")
        evidence_bindings = _copy_resources(evidence, staging, {}, prefix=f"{_META}/evidence")
        for name, record in evidence_records.items():
            binding = evidence_bindings[name]
            if binding["kind"] != "file":
                raise ValueError("verified evaluation evidence must be a single JSON file")
            copied = _verify_file(staging, {"file": binding["path"], **record["content"]})
            if _evidence(copied, source) != record:
                raise ValueError("evaluation evidence changed during export")
        attachment_bindings = _copy_resources(
            attachments, staging, {}, prefix=f"{_META}/attachments"
        )
        packaged = None
        if deployment is not None:
            packaged = package_recipe_deployment(
                deployment,
                staging / _META / "deployment",
                # Preserve a selected portable native base; never discover/download one.
                include_base=inspected.get("base_source") == "bundled",
            )
            if source.get("deployment_sha256") is not None and (
                packaged.get("deployment_sha256") != source["deployment_sha256"]
            ):
                raise ValueError("packaged deployment differs from the committed result identity")
            packaged = {
                key: value
                for key, value in packaged.items()
                if key not in {"directory", "artifact"}
            }
        unresolved = list(config.unresolved_checks)
        if deployment is None:
            unresolved.append(
                "model/runtime package was not included; source identity is recorded only"
            )
        elif packaged.get("base_weights_included") is False:
            unresolved.append(
                "native base weights were not included; external local base is required"
            )
        unresolved.append(
            "export is not external submission, task-quality acceptance or host enforcement"
        )
        manifest = {
            "format": "arti.result-export",
            "format_version": 1,
            "config": config.to_dict(),
            "source": {key: value for key, value in source.items() if key != "directory"},
            "expected_ids": expected,
            "records": len(rows),
            "native_records": len(native),
            "formatted_results": f"{_META}/formatted-results.jsonl",
            "native_results": f"{_META}/native-results.jsonl",
            "inputs": f"{_META}/inputs.jsonl",
            "validation": validation,
            "evidence": evidence_records,
            "evidence_bindings": evidence_bindings,
            "attachments": attachment_bindings,
            "deployment": packaged,
            "examples": [
                {
                    "id": identifier,
                    "input": inputs[identifier],
                    "prediction": next(
                        row
                        for row in native
                        if row.get(source["conditions"].get("output_id_field", "id")) == identifier
                    ),
                    "execution": execution[identifier],
                }
                for identifier in config.examples
            ],
            "unresolved_checks": unresolved,
            "files": _inventory(staging),
            "directories": _directories(staging),
        }
        manifest["sha256"] = json_fingerprint(manifest)
        _atomic_json(staging / _META / "manifest.json", manifest)
        inspect_result_export(staging)
        if config.archive == "zip":
            archive = temporary / "result.zip"
            with zipfile.ZipFile(
                archive, "w", compression=zipfile.ZIP_STORED, allowZip64=True
            ) as package:
                for path in sorted(staging.rglob("*")):
                    package.write(
                        path, path.relative_to(staging).as_posix() + ("/" if path.is_dir() else "")
                    )
            inspect_result_export(archive)
            _replace_atomic_file(archive, destination)
        else:
            _replace_atomic_file(staging, destination)
    return {
        "ok": True,
        "kind": "result-export",
        "destination": str(destination),
        "archive": config.archive,
        "sha256": manifest["sha256"],
        "records": len(expected),
        "unresolved_checks": unresolved,
        "loads_model": False,
        "regenerates_predictions": False,
        "submits_externally": False,
    }


def inspect_result_export(path):
    """Verify all exported bytes and expected IDs without writer/model code."""
    path = Path(path).resolve()
    if path.is_dir():
        manifest = _load_json(path / _META / "manifest.json")

        def reader(name):
            return _member(path, name).open("rb")

        names = {
            member.relative_to(path).as_posix() for member in path.rglob("*") if member.is_file()
        }
        directories = _directories(path)
        archive = None
    else:
        archive = zipfile.ZipFile(path)
        manifest = json.loads(archive.read(f"{_META}/manifest.json"))
        reader = archive.open
        names = {item.filename for item in archive.infolist() if not item.is_dir()}
        directories = sorted(
            item.filename.rstrip("/") for item in archive.infolist() if item.is_dir()
        )
        if len(names) != sum(not item.is_dir() for item in archive.infolist()):
            archive.close()
            raise ValueError("result archive has duplicate file members")
    try:
        if (
            manifest.get("format") != "arti.result-export"
            or manifest.get("format_version") != 1
            or manifest.get("sha256")
            != json_fingerprint({key: value for key, value in manifest.items() if key != "sha256"})
        ):
            raise ValueError("result export manifest differs")
        if names != set(manifest["files"]) | {f"{_META}/manifest.json"}:
            raise ValueError("result export file inventory differs")
        if directories != manifest["directories"]:
            raise ValueError("result export directory inventory differs")
        expected = manifest["expected_ids"]
        if (
            not isinstance(expected, list)
            or not expected
            or any(not isinstance(identifier, str) or not identifier for identifier in expected)
            or len(set(expected)) != len(expected)
            or manifest["records"] != len(expected)
            or manifest["native_records"] != len(expected)
        ):
            raise ValueError("result export expected ID population differs")
        for name, declaration in manifest["files"].items():
            digest, size = hashlib.sha256(), 0
            with reader(name) as file:
                for chunk in iter(lambda: file.read(1024 * 1024), b""):
                    digest.update(chunk)
                    size += len(chunk)
            if declaration != {"sha256": digest.hexdigest(), "bytes": size}:
                raise ValueError("result export file content differs")
        with reader(f"{_META}/plan.json") as file:
            plan = json.load(file)
        if plan.get("sha256") != manifest["source"]["plan_sha256"] or (
            plan.get("sha256")
            != json_fingerprint({key: value for key, value in plan.items() if key != "sha256"})
        ):
            raise ValueError("result export prediction plan identity differs")
        input_digest = manifest["files"][manifest["inputs"]]["sha256"]
        if input_digest != plan["input"]["sha256"] or (
            input_digest != manifest["source"]["input_sha256"]
        ):
            raise ValueError("result export pinned input snapshot differs")
        with reader(manifest["inputs"]) as file:
            input_ids = [
                json.loads(line).get(plan["fields"]["id"]) for line in file if line.strip()
            ]
        if (
            len(input_ids) != len(expected)
            or set(input_ids) != set(expected)
            or plan["input"]["records"] != len(expected)
            or manifest["source"]["total_records"] != len(expected)
            or manifest["source"]["committed_records"] != len(expected)
        ):
            raise ValueError("result export original input ID population differs")
        closures = {}
        for name in ("formatted_results", "native_results"):
            with reader(manifest[name]) as file:
                rows = [json.loads(line) for line in file if line.strip()]
            ids = (
                [row.get("id") for row in rows]
                if name == "formatted_results"
                else [
                    row.get(manifest["source"]["conditions"].get("output_id_field", "id"))
                    for row in rows
                ]
            )
            if len(ids) != len(expected) or set(ids) != set(expected):
                raise ValueError("result export ID coverage differs")
            field = (
                "files"
                if name == "formatted_results"
                else manifest["source"].get("files", {}).get("field")
            )
            files = [] if field is None else _declared_files(rows, field)
            closures[name] = {file["file"] for file in files}
            for file in files:
                if manifest["files"].get(file["file"]) != {
                    key: file[key] for key in ("sha256", "bytes")
                }:
                    raise ValueError("result export declared file closure differs")
        if closures["formatted_results"] != {
            name for name in names if name != _META and not name.startswith(f"{_META}/")
        } or closures["native_results"] != {
            name for name in names if name.startswith(f"{_META}/native/")
        }:
            raise ValueError("result export declared file closure differs")
        for name, record in manifest["evidence"].items():
            binding = manifest["evidence_bindings"][name]
            if binding["kind"] != "file":
                raise ValueError("verified evaluation evidence must be a single JSON file")
            with reader(binding["path"]) as file:
                if _evidence_bytes(file.read(), manifest["source"]) != record:
                    raise ValueError("result export evidence binding differs")
        packaged = manifest["deployment"]
        if packaged is not None:
            prefix = f"{_META}/deployment/"
            with reader(f"{prefix}deployment.json") as file:
                deployment = json.load(file)
            if deployment.get("task_recipe_sha256") != manifest["source"]["task_recipe_sha256"]:
                raise ValueError("result export deployment task identity differs")
            if deployment.get("format") == "arti.task-deployment":
                identity = deployment.get("sha256")
                if (
                    identity != packaged.get("deployment_sha256")
                    or identity
                    != json_fingerprint(
                        {key: value for key, value in deployment.items() if key != "sha256"}
                    )
                    or deployment.get("model_manifest_sha256")
                    != manifest["source"]["model_manifest_sha256"]
                    or manifest["source"].get("deployment_sha256") not in {None, identity}
                ):
                    raise ValueError("result export deployment content identity differs")
                if {name for name in names if name.startswith(prefix)} != {
                    f"{prefix}{name}" for name in deployment["files"]
                } | {f"{prefix}deployment.json"}:
                    raise ValueError("result export deployment file closure differs")
                for name, record in deployment["files"].items():
                    if manifest["files"].get(f"{prefix}{name}") != record:
                        raise ValueError("result export deployment file closure differs")
            elif deployment.get("format") != "arti.causal-lm-deployment" or deployment.get(
                "base_weights_included"
            ) != packaged.get("base_weights_included"):
                raise ValueError("result export deployment declaration differs")
        return {
            "ok": True,
            "kind": "result-export-inspection",
            "sha256": manifest["sha256"],
            "records": manifest["records"],
            "source": manifest["source"],
            "files": len(manifest["files"]),
            "validation": manifest["validation"],
            "unresolved_checks": manifest["unresolved_checks"],
            "loads_model": False,
            "imports_writer": False,
            "submits_externally": False,
        }
    finally:
        if archive is not None:
            archive.close()


__all__ = [
    "ResultExportConfig",
    "ResultExportContext",
    "export_prediction_results",
    "inspect_result_export",
]
