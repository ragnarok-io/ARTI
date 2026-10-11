"""Named task files inside the existing atomic prediction-part boundary."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import time

from .python_factory import PythonFactoryConfig
from .recipe_assets import _member, _verify_file
from .recipe_data import json_fingerprint
from .serialization import _file_sha256, _valid_sha


@dataclass(frozen=True)
class PredictionFilesConfig:
    field: str = "files"
    validator: PythonFactoryConfig | str | None = None
    revision: str | None = None

    def __post_init__(self):
        if not isinstance(self.field, str) or not self.field.strip():
            raise ValueError("prediction files field must be a nonempty string")
        if isinstance(self.validator, str):
            object.__setattr__(self, "validator", PythonFactoryConfig(self.validator))
        if self.validator is not None and not isinstance(self.validator, PythonFactoryConfig):
            raise ValueError("file validator must be an installed PythonFactoryConfig")
        if (self.validator is None) != (self.revision is None) or (
            self.revision is not None
            and (not isinstance(self.revision, str) or not self.revision.strip())
        ):
            raise ValueError("file validator and its declared revision must be supplied together")

    def to_dict(self):
        return {
            "field": self.field,
            "validator": None if self.validator is None else self.validator.to_dict(),
            "revision": self.revision,
        }

    @classmethod
    def from_dict(cls, value):
        value = dict(value)
        if isinstance(value.get("validator"), Mapping):
            value["validator"] = PythonFactoryConfig(**value["validator"])
        return cls(**value)


def _declared_files(predictions, field):
    files = {}
    for row in predictions:
        declared = row.get(field)
        if not isinstance(declared, Mapping):
            raise ValueError("file predictions require a named file mapping per result")
        for name, value in declared.items():
            if not isinstance(name, str) or not name or not isinstance(value, Mapping):
                raise ValueError("prediction files need nonempty names and file declarations")
            if (
                not isinstance(value.get("file"), str)
                or not value["file"]
                or not _valid_sha(value.get("sha256"))
                or type(value.get("bytes")) is not int
                or value["bytes"] < 0
                or not isinstance(value.get("metadata"), dict)
            ):
                raise ValueError("prediction file declaration is incomplete")
            record = {key: value[key] for key in ("file", "sha256", "bytes")}
            if record["file"] in files and record != files[record["file"]]:
                raise ValueError("aliased prediction file content differs")
            files[record["file"]] = record
    return [files[name] for name in sorted(files)]


def _stage_prediction_files(predictions, source, staging, prefix, config, records):
    started = time.monotonic()
    result = json.loads(json.dumps(predictions, allow_nan=False))
    (staging / "files").mkdir()
    copied = set()
    for row in result:
        declared = row.get(config.field)
        if not isinstance(declared, dict):
            raise ValueError("file predictions require a named file mapping per result")
        for name, value in declared.items():
            if not isinstance(name, str) or not name:
                raise ValueError("prediction file names must be nonempty strings")
            value = {"file": value} if isinstance(value, str) else value
            if not isinstance(value, dict) or not isinstance(value.get("file"), str):
                raise ValueError("producer file must name a relative path")
            if Path(value["file"]).is_absolute():
                raise ValueError("producer files must be relative to output_dir")
            path = _member(source, value["file"])
            if not path.is_file():
                raise ValueError("a declared prediction file is missing")
            relative = path.relative_to(source).as_posix()
            target = staging / "files" / relative
            if relative not in copied:
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(path, target)
                copied.add(relative)
            metadata = value.get("metadata", {})
            if not isinstance(metadata, dict):
                raise ValueError("prediction file metadata must be a JSON object")
            declared[name] = {
                "file": relative,
                "sha256": _file_sha256(target),
                "bytes": target.stat().st_size,
                "metadata": metadata,
            }
    validation = None
    if config.validator is not None:
        validation = config.validator.resolve()(
            json.loads(json.dumps(records)),
            json.loads(json.dumps(result)),
            directory=staging / "files",
        )
        if not isinstance(validation, Mapping) or validation.get("ok") is not True:
            raise ValueError("file validator rejected prediction outputs")
        validation = json.loads(json.dumps(dict(validation), allow_nan=False))
    # Validation may inspect results, but cannot rewrite the committed declaration.
    for row in result:
        for value in row[config.field].values():
            _verify_file(staging / "files", value)
            value["file"] = f"{prefix}/files/{value['file']}"
    files = _declared_files(result, config.field)
    return result, {
        "files": files,
        "files_sha256": json_fingerprint(files),
        "file_validation": validation,
        "file_validation_sha256": json_fingerprint(validation),
        "file_preparation_seconds": time.monotonic() - started,
    }


def _verify_prediction_files(directory, part, predictions, plan, receipt):
    files = _declared_files(predictions, plan["files"]["field"])
    if receipt.get("files") != files or receipt.get("files_sha256") != json_fingerprint(files):
        raise ValueError("prediction file manifest differs from the committed result")
    validation = receipt.get("file_validation")
    if receipt.get("file_validation_sha256") != json_fingerprint(validation) or (
        plan["files"]["validator"] is not None
        and (not isinstance(validation, Mapping) or validation.get("ok") is not True)
    ):
        raise ValueError("prediction file validation receipt differs")
    root = (part / "files").resolve()
    for file in files:
        path = _verify_file(directory, file)
        if not path.is_relative_to(root):
            raise ValueError("prediction files must belong to their committed part")
    return {
        "output_files": len(files),
        "output_bytes": sum(file["bytes"] for file in files),
        "file_preparation_seconds": receipt["file_preparation_seconds"],
    }


__all__ = ["PredictionFilesConfig"]
