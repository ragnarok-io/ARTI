"""Explicit local SFT data import into the existing chat recipe contract."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
import tempfile

from .recipe_chat import normalized_messages
from .recipe_data import json_fingerprint
from .recipe_workflow import read_recipe_config, write_json, write_jsonl
from .serialization import _file_sha256, _replace_atomic_file


@dataclass(frozen=True)
class CausalLMImportFields:
    id: str | None = None
    messages: str = "messages"
    prompt: str = "prompt"
    completion: str = "completion"

    def __post_init__(self):
        names = [value for value in asdict(self).values() if value is not None]
        if any(not isinstance(value, str) or not value for value in names):
            raise ValueError("import fields must be nonempty strings")
        if any(getattr(self, name) is None for name in ("messages", "prompt", "completion")):
            raise ValueError("only the import id field may be null")


@dataclass(frozen=True)
class CausalLMDataSource:
    path: str | Path
    name: str
    split: str
    format: str = "messages"
    file_format: str = "jsonl"
    fields: CausalLMImportFields = field(default_factory=CausalLMImportFields)

    def __post_init__(self):
        if not isinstance(self.path, (str, Path)) or not str(self.path).strip():
            raise ValueError("source.path is required")
        object.__setattr__(self, "path", Path(self.path).resolve())
        if not isinstance(self.name, str) or not self.name.strip() or ":" in self.name:
            raise ValueError("source.name must be a nonempty logical name without ':'")
        if self.split not in {"train", "validation", "test"}:
            raise ValueError("source.split must be train, validation or test")
        if self.format not in {"messages", "prompt_completion"}:
            raise ValueError("source.format must be messages or prompt_completion")
        if self.file_format not in {"jsonl", "json"}:
            raise ValueError("source.file_format must be jsonl or json (a top-level array)")
        if not isinstance(self.fields, CausalLMImportFields):
            raise ValueError("source.fields must be CausalLMImportFields")
        active_fields = (
            [self.fields.messages]
            if self.format == "messages"
            else [self.fields.prompt, self.fields.completion]
        )
        if self.fields.id is not None:
            active_fields.append(self.fields.id)
        if len(set(active_fields)) != len(active_fields):
            raise ValueError("active source fields must be distinct")


@dataclass(frozen=True)
class CausalLMDataImportConfig:
    sources: tuple[CausalLMDataSource, ...]

    def __post_init__(self):
        object.__setattr__(self, "sources", tuple(self.sources))
        if not self.sources or any(
            not isinstance(item, CausalLMDataSource) for item in self.sources
        ):
            raise ValueError("import requires explicit CausalLMDataSource entries")
        if len({item.name for item in self.sources}) != len(self.sources):
            raise ValueError("source names must be unique")

    @classmethod
    def read(cls, path: str | Path):
        path = Path(path).resolve()
        payload = read_recipe_config(path)
        if payload.keys() != {"task", "sources"} or payload["task"] != "causal-lm":
            raise ValueError("import config requires task=causal-lm and sources")
        if not isinstance(payload["sources"], list):
            raise ValueError("import sources must be an array")
        sources = []
        for entry in payload["sources"]:
            if not isinstance(entry, Mapping):
                raise ValueError("each source must be an object")
            settings = dict(entry)
            source_path = settings.get("path")
            if not isinstance(source_path, str) or not source_path.strip():
                raise ValueError("source.path is required")
            settings["path"] = path.parent / source_path
            settings["fields"] = CausalLMImportFields(**settings.get("fields", {}))
            sources.append(CausalLMDataSource(**settings))
        return cls(tuple(sources))

    def to_dict(self):
        return {
            "task": "causal-lm",
            "sources": [{**asdict(source), "path": str(source.path)} for source in self.sources],
        }


def _source_rows(source):
    digest = hashlib.sha256()
    if source.file_format == "json":
        payload = source.path.read_bytes()
        digest.update(payload)
        rows = json.loads(payload)
        if not isinstance(rows, list):
            raise ValueError("JSON data must be a top-level array")
    else:
        rows = []
        with source.path.open("rb") as stream:
            for line_number, line in enumerate(stream, 1):
                digest.update(line)
                if not line.strip():
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError as error:
                    raise ValueError(f"invalid JSONL at line {line_number}: {error.msg}") from error
    if not rows:
        raise ValueError("source contains no records")
    return rows, digest.hexdigest()


def _source_messages(source, row):
    if source.format == "messages":
        return normalized_messages(row.get(source.fields.messages), completed=True)
    prompt, completion = row.get(source.fields.prompt), row.get(source.fields.completion)
    if isinstance(prompt, str):
        prompt = [{"role": "user", "content": prompt}]
    prompt = normalized_messages(prompt, completed=False)
    if isinstance(completion, str):
        completion = [{"role": "assistant", "content": completion}]
    if not isinstance(completion, list) or len(completion) != 1:
        raise ValueError("completion must be text or one assistant message")
    return normalized_messages([*prompt, *completion], completed=True)


def import_causal_lm_data(config, output: str | Path):
    """Publish a fresh data snapshot, without tokenizer, model, or optimizer work."""
    config = (
        config
        if isinstance(config, CausalLMDataImportConfig)
        else CausalLMDataImportConfig.read(config)
    )
    target = Path(output).resolve()
    if target.exists():
        raise FileExistsError(f"choose a new snapshot directory: {target}")
    records, sources, ids, content_splits = [], [], set(), {}
    grouped = {role: [] for role in ("train", "validation", "test")}
    repeated = {role: 0 for role in grouped}
    for source in config.sources:
        try:
            rows, source_sha256 = _source_rows(source)
        except (ValueError, OSError) as error:
            raise ValueError(f"source {source.name!r}: {error}") from error
        for number, row in enumerate(rows, 1):
            try:
                if not isinstance(row, Mapping):
                    raise ValueError("record must be an object")
                local_id = str(number) if source.fields.id is None else row.get(source.fields.id)
                if type(local_id) is int:
                    local_id = str(local_id)
                if not isinstance(local_id, str) or not local_id:
                    raise ValueError("configured source id must be a nonempty string or integer")
                identifier = f"{source.name}:{local_id}"
                if identifier in ids:
                    raise ValueError(f"duplicate id {identifier!r}")
                messages = _source_messages(source, row)
                content = json_fingerprint(messages)
                previous = content_splits.get(content)
                if previous is not None and previous != source.split:
                    raise ValueError(
                        f"identical completed conversation also occurs in {previous!r}; assign disjoint source splits"
                    )
                if previous is not None:
                    repeated[source.split] += 1
                content_splits[content] = source.split
                ids.add(identifier)
                records.append({"id": identifier, "split": source.split, "messages": messages})
                grouped[source.split].append({"id": identifier, "messages": messages})
            except ValueError as error:
                raise ValueError(f"source {source.name!r}, record {number}: {error}") from error
        sources.append(
            {
                "name": source.name,
                "split": source.split,
                "format": source.format,
                "file_format": source.file_format,
                "fields": asdict(source.fields),
                "sha256": source_sha256,
                "records": len(rows),
                "id_rule": "source-name:record-ordinal"
                if source.fields.id is None
                else "source-name:source-id",
            }
        )
    if not grouped["train"] or not grouped["validation"]:
        raise ValueError("import requires nonempty explicit train and validation splits")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}-", dir=target.parent) as temporary:
        staging = Path(temporary) / "snapshot"
        staging.mkdir()
        write_jsonl(staging / "dataset.jsonl", records)
        manifest = {
            "format": "arti.causal-lm-data-import",
            "format_version": 1,
            "sources": sources,
            "transform": "complete-chat-final-assistant",
            "split_policy": "explicit-source-role",
            "output": {
                "file": "dataset.jsonl",
                "sha256": _file_sha256(staging / "dataset.jsonl"),
                "dataset_sha256": json_fingerprint(grouped),
            },
            "splits": {
                role: {
                    "records": len(items),
                    "sha256": json_fingerprint(items),
                    "repeated_conversations": repeated[role],
                }
                for role, items in grouped.items()
            },
            "records_dropped": 0,
        }
        manifest["sha256"] = json_fingerprint(manifest)
        write_json(staging / "manifest.json", manifest)
        _replace_atomic_file(staging, target)
    return {
        "ok": True,
        "kind": "recipe-import-data",
        "task": "causal-lm",
        "data": {
            "path": str(target / "dataset.jsonl"),
            "source_manifest": str(target / "manifest.json"),
        },
        "manifest": manifest,
        "weights_loaded": False,
        "optimizer_created": False,
    }


def _load_source_manifest(path, dataset_sha256):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != "arti.causal-lm-data-import"
        or manifest.get("format_version") != 1
    ):
        raise ValueError("source_manifest must be an ARTI causal-LM import manifest")
    identity = {key: value for key, value in manifest.items() if key != "sha256"}
    if manifest.get("sha256") != json_fingerprint(identity):
        raise ValueError("source_manifest content fingerprint differs")
    if manifest.get("output", {}).get("dataset_sha256") != dataset_sha256:
        raise ValueError("source_manifest does not describe this dataset snapshot")
    return manifest


__all__ = [
    "CausalLMImportFields",
    "CausalLMDataSource",
    "CausalLMDataImportConfig",
    "import_causal_lm_data",
]
