"""Portable native records, immutable file inputs and batched derived assets."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

from .python_factory import PythonFactoryConfig
from .recipe_data import _positive_int, json_fingerprint, read_jsonl
from .recipe_workflow import read_recipe_config, write_jsonl
from .serialization import _atomic_json, _file_sha256, _load_json, _replace_atomic_file


def _json(value):
    return json.loads(json.dumps(value, allow_nan=False))


def _name(value, name):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _seal(value):
    return {**value, "sha256": json_fingerprint(value)}


def _read_sealed(path, format):
    value = _load_json(path)
    if (
        value.get("format") != format
        or value.get("format_version") != 1
        or value.get("sha256")
        != json_fingerprint({key: item for key, item in value.items() if key != "sha256"})
    ):
        raise ValueError(f"invalid {format} manifest")
    return value


def _member(root, relative):
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("asset file must be inside its committed directory")
    return path


def _verify_file(root, declaration):
    path = _member(root, declaration["file"])
    if (
        not path.is_file()
        or path.stat().st_size != declaration["bytes"]
        or _file_sha256(path) != declaration["sha256"]
    ):
        raise ValueError(f"asset content differs: {declaration['file']}")
    return path


@dataclass(frozen=True)
class AssetReference:
    """A record's use of a logical asset; absent optional evidence stays absent."""

    asset_id: str
    role: str = "input"
    required: bool = True

    def __post_init__(self):
        _name(self.asset_id, "asset_id")
        _name(self.role, "asset role")
        if type(self.required) is not bool:
            raise ValueError("asset required must be a boolean")

    def to_dict(self):
        return {"asset_id": self.asset_id, "role": self.role, "required": self.required}


@dataclass(frozen=True)
class DatasetSnapshotConfig:
    records: str | Path
    assets: Mapping[str, str | Path | None] = field(default_factory=dict)
    bindings: Mapping[str, Sequence[AssetReference | Mapping[str, Any]]] = field(
        default_factory=dict
    )
    asset_metadata: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    id_field: str = "id"
    split_field: str | None = "split"
    group_field: str | None = None
    source_revision: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "records", Path(self.records).resolve())
        for key in ("id_field", "split_field", "group_field", "source_revision"):
            if getattr(self, key) is not None:
                _name(getattr(self, key), key)
        if self.id_field is None:
            raise ValueError("id_field is required")
        if not isinstance(self.assets, Mapping) or not isinstance(self.bindings, Mapping):
            raise ValueError("assets/bindings must be mappings")
        object.__setattr__(
            self,
            "assets",
            {
                _name(name, "asset_id"): None if path is None else Path(path).resolve()
                for name, path in self.assets.items()
            },
        )
        object.__setattr__(
            self,
            "bindings",
            {
                _name(identifier, "record id"): tuple(
                    item if isinstance(item, AssetReference) else AssetReference(**item)
                    for item in references
                )
                for identifier, references in self.bindings.items()
            },
        )
        object.__setattr__(self, "asset_metadata", _json(dict(self.asset_metadata)))
        object.__setattr__(self, "metadata", _json(dict(self.metadata)))
        if self.asset_metadata.keys() - self.assets.keys():
            raise ValueError("asset metadata names an undeclared asset")
        if any(not isinstance(value, dict) for value in self.asset_metadata.values()):
            raise ValueError("each asset metadata value must be a JSON object")

    @classmethod
    def read(cls, path):
        path = Path(path).resolve()
        value = dict(read_recipe_config(path))
        if value.pop("format", "arti.dataset-snapshot-config") != "arti.dataset-snapshot-config":
            raise ValueError("not a dataset snapshot configuration")

        def local(name):
            target = Path(name)
            return target if target.is_absolute() else path.parent / target

        value["records"] = local(value["records"])
        value["assets"] = {
            name: None if source is None else local(source)
            for name, source in value.get("assets", {}).items()
        }
        return cls(**value)


class DatasetSnapshot:
    """A verified, movable file closure; application records remain unchanged.

    This local adapter indexes record IDs in memory. It does not decode assets,
    infer labels, assign splits, or decide an application's collation/sampler.
    """

    def __init__(self, directory):
        self.directory = Path(directory).resolve()
        self._manifest = _read_sealed(self.directory / "dataset.json", "arti.dataset-snapshot")
        _verify_file(self.directory, self._manifest["records"])
        seen = set()
        for asset in self._manifest["assets"].values():
            if asset["available"] and asset["file"] not in seen:
                _verify_file(self.directory, asset)
                seen.add(asset["file"])

    def manifest(self):
        return _json(self._manifest)

    def records(self, split=None):
        field = self._manifest["fields"]["split"]
        if split is not None and field is None:
            raise ValueError("this snapshot has no declared split field")
        for row in read_jsonl(self.directory / "records.jsonl"):
            if split is None or row[field] == split:
                yield row

    def asset(self, asset_id):
        asset = self._manifest["assets"][asset_id]
        return _member(self.directory, asset["file"]) if asset["available"] else None

    def references(self, record_id, *, role=None):
        if record_id not in self._manifest["bindings"]:
            raise KeyError(record_id)
        return [
            {**_json(item), "available": self._manifest["assets"][item["asset_id"]]["available"]}
            for item in self._manifest["bindings"][record_id]
            if role is None or item["role"] == role
        ]


def create_dataset_snapshot(config, directory):
    """Copy an explicit local source to a fresh, atomically published snapshot."""
    config = (
        config if isinstance(config, DatasetSnapshotConfig) else DatasetSnapshotConfig.read(config)
    )
    directory = Path(directory).resolve()
    if directory.exists():
        raise FileExistsError("dataset snapshot already exists")
    rows, seen, splits, groups, bindings = [], set(), {}, {}, {}
    for row in read_jsonl(config.records):
        identifier = _name(row.get(config.id_field), "record id")
        if identifier in seen:
            raise ValueError("dataset record IDs must be unique")
        seen.add(identifier)
        role = (
            "unsplit" if config.split_field is None else _name(row.get(config.split_field), "split")
        )
        splits.setdefault(role, []).append(identifier)
        if config.group_field is not None:
            group = _name(row.get(config.group_field), "record group")
            if group in groups and groups[group] != role:
                raise ValueError("a declared group cannot cross dataset splits")
            groups[group] = role
        references = config.bindings.get(identifier, ())
        for reference in references:
            if reference.asset_id not in config.assets:
                raise ValueError("record references an undeclared asset")
            if reference.required and config.assets[reference.asset_id] is None:
                raise ValueError("a required record asset is unavailable")
        bindings[identifier] = [item.to_dict() for item in references]
        rows.append(row)
    if not rows:
        raise ValueError("dataset snapshot requires records")
    if config.bindings.keys() - seen:
        raise ValueError("asset bindings name an unknown record ID")
    used = {item["asset_id"] for items in bindings.values() for item in items}
    if used != config.assets.keys():
        raise ValueError("declare only assets referenced by dataset records")
    directory.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=f".{directory.name}-", dir=directory.parent
    ) as temporary:
        staging = Path(temporary) / "snapshot"
        staging.mkdir()
        write_jsonl(staging / "records.jsonl", rows)
        assets, unique = {}, {}
        for name, source in config.assets.items():
            declaration = {
                "available": source is not None,
                "metadata": config.asset_metadata.get(name, {}),
            }
            if source is not None:
                if not source.is_file():
                    raise ValueError(f"asset {name!r} must name an existing regular file")
                digest = _file_sha256(source)
                relative = f"assets/{digest}"
                destination = staging / relative
                if digest not in unique:
                    destination.parent.mkdir(exist_ok=True)
                    shutil.copyfile(source, destination)
                    if _file_sha256(destination) != digest:
                        raise ValueError("asset changed while creating the snapshot")
                    unique[digest] = destination.stat().st_size
                declaration.update(file=relative, sha256=digest, bytes=unique[digest])
            assets[name] = declaration
        references = [item for items in bindings.values() for item in items]
        manifest = _seal(
            {
                "format": "arti.dataset-snapshot",
                "format_version": 1,
                "source_revision": config.source_revision,
                "fields": {
                    "id": config.id_field,
                    "split": config.split_field,
                    "group": config.group_field,
                },
                "metadata": config.metadata,
                "records": {
                    "file": "records.jsonl",
                    "sha256": _file_sha256(staging / "records.jsonl"),
                    "bytes": (staging / "records.jsonl").stat().st_size,
                    "count": len(rows),
                },
                "assets": assets,
                "bindings": bindings,
                "splits": {name: {"samples": len(ids), "ids": ids} for name, ids in splits.items()},
                "groups": groups,
                "inventory": {
                    "references": len(references),
                    "logical_assets": len(assets),
                    "unique_contents": len(unique),
                    "unique_bytes": sum(unique.values()),
                    "referenced_bytes": sum(
                        assets[item["asset_id"]].get("bytes", 0) for item in references
                    ),
                    "unavailable_references": sum(
                        not assets[item["asset_id"]]["available"] for item in references
                    ),
                },
            }
        )
        _atomic_json(staging / "dataset.json", manifest)
        _replace_atomic_file(staging, directory)
    return DatasetSnapshot(directory)


def inspect_dataset_snapshot(directory, *, output=None):
    result = DatasetSnapshot(directory).manifest()
    if output is not None:
        target = Path(output)
        if target.resolve().is_relative_to(Path(directory).resolve()):
            raise ValueError("write inspection reports outside the immutable snapshot")
        target.parent.mkdir(parents=True, exist_ok=True)
        _atomic_json(target, result)
    return result


@dataclass(frozen=True)
class AssetTransformConfig:
    factory: PythonFactoryConfig | str
    revision: str
    options: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if isinstance(self.factory, str):
            object.__setattr__(self, "factory", PythonFactoryConfig(self.factory))
        if not isinstance(self.factory, PythonFactoryConfig):
            raise ValueError("transform factory must be an installed PythonFactoryConfig")
        _name(self.revision, "transform revision")
        object.__setattr__(self, "options", _json(dict(self.options)))

    def to_dict(self):
        return {
            "factory": self.factory.to_dict(),
            "revision": self.revision,
            "options": _json(self.options),
        }

    @classmethod
    def read(cls, path):
        value = dict(read_recipe_config(path))
        if value.pop("format", "arti.asset-transform-config") != "arti.asset-transform-config":
            raise ValueError("not an asset transform configuration")
        if isinstance(value.get("factory"), Mapping):
            value["factory"] = PythonFactoryConfig(**value["factory"])
        return cls(**value)


@dataclass(frozen=True)
class AssetPreparationInput:
    source: Path
    directory: Path
    sha256: str
    metadata: Mapping[str, Any]


def _inventory(directory):
    files = []
    for path in sorted(directory.rglob("*")):
        if path.is_symlink():
            raise ValueError("derived assets must contain files, not external links")
        if path.is_file():
            files.append(
                {
                    "file": path.relative_to(directory).as_posix(),
                    "sha256": _file_sha256(path),
                    "bytes": path.stat().st_size,
                }
            )
    if not files:
        raise ValueError("an asset transform must produce at least one file")
    if any(item["file"] == "entry.json" for item in files):
        raise ValueError("entry.json is reserved for the preparation receipt")
    return files


def _entry(cache, key):
    directory = cache / "entries" / key
    value = _read_sealed(directory / "entry.json", "arti.prepared-asset")
    if value["key"] != key or key != json_fingerprint(
        {
            "source_sha256": value["source_sha256"],
            "source_metadata": value["source_metadata"],
            "transform": value["transform"],
        }
    ):
        raise ValueError("preparation cache key differs")
    for item in value["files"]:
        _verify_file(directory, item)
    return value


class DatasetAssetPreparation:
    def __init__(self, cache, report):
        self.cache = Path(cache).resolve()
        self.report = _json(report)

    def files(self, asset_id):
        key = self.report["assets"][asset_id]
        if key is None:
            return None
        value = _entry(self.cache, key)
        return {
            item["file"]: _member(self.cache / "entries" / key, item["file"])
            for item in value["files"]
        }

    def manifest(self):
        if not self.report["complete"]:
            raise ValueError("preparation is incomplete; do not use it as a training dataset")
        return _seal(
            {
                "format": "arti.prepared-dataset",
                "format_version": 1,
                "dataset_sha256": self.report["dataset_sha256"],
                "transform": self.report["transform"],
                "roles": self.report["roles"],
                "assets": {
                    name: None
                    if key is None
                    else {"key": key, "entry_sha256": _entry(self.cache, key)["sha256"]}
                    for name, key in self.report["assets"].items()
                },
            }
        )


def prepare_dataset_assets(
    snapshot, cache, transform, *, roles=None, batch_size=16, stop_requested=None, on_batch=None
):
    """Cache deterministic asset-local transforms; commit each complete file closure.

    The installed callback receives a batch of AssetPreparationInput and options,
    writes inside each request.directory, and returns one JSON metadata object
    per request. It must depend only on source bytes/metadata and the declared
    transform revision/options, not record labels, batch peers or source paths.
    """
    snapshot = DatasetSnapshot(
        snapshot.directory if isinstance(snapshot, DatasetSnapshot) else snapshot
    )
    transform = (
        transform
        if isinstance(transform, AssetTransformConfig)
        else AssetTransformConfig.read(transform)
    )
    _positive_int(batch_size, "preparation batch_size")
    if roles is not None:
        if isinstance(roles, str) or not roles:
            raise ValueError("roles must be a nonempty sequence or None")
        roles = sorted({_name(role, "preparation role") for role in roles})
    cache = Path(cache).resolve()
    if cache.is_relative_to(snapshot.directory) or snapshot.directory.is_relative_to(cache):
        raise ValueError("use a separate preparation cache, not the immutable snapshot")
    manifest = snapshot.manifest()
    selected = {
        item["asset_id"]
        for items in manifest["bindings"].values()
        for item in items
        if roles is None or item["role"] in roles
    }
    assets, sources = {}, {}
    for name in sorted(selected):
        asset = manifest["assets"][name]
        if not asset["available"]:
            assets[name] = None
            continue
        key = json_fingerprint(
            {
                "source_sha256": asset["sha256"],
                "source_metadata": asset["metadata"],
                "transform": transform.to_dict(),
            }
        )
        assets[name] = key
        sources.setdefault(key, name)
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "entries").mkdir(exist_ok=True)
    pending, reused = [], 0
    for key, name in sources.items():
        if (cache / "entries" / key).exists():
            _entry(cache, key)
            reused += 1
        else:
            pending.append((key, name))
    produced, batches, callback = 0, 0, None
    for start in range(0, len(pending), batch_size):
        if stop_requested is not None and stop_requested():
            break
        if callback is None:
            callback = transform.factory.resolve()
        batch = pending[start : start + batch_size]
        with tempfile.TemporaryDirectory(prefix=".prepare-", dir=cache) as temporary:
            requests = []
            for index, (key, name) in enumerate(batch):
                directory = Path(temporary) / str(index)
                directory.mkdir()
                asset = manifest["assets"][name]
                requests.append(
                    AssetPreparationInput(
                        snapshot.asset(name), directory, asset["sha256"], _json(asset["metadata"])
                    )
                )
            metadata = list(callback(requests, options=_json(transform.options)))
            if len(metadata) != len(batch) or any(
                not isinstance(item, Mapping) for item in metadata
            ):
                raise ValueError("asset transform must return one JSON metadata object per source")
            # Finish all closures before publishing any part of this callback batch.
            entries = [
                _seal(
                    {
                        "format": "arti.prepared-asset",
                        "format_version": 1,
                        "key": key,
                        "source_sha256": request.sha256,
                        "source_metadata": _json(request.metadata),
                        "transform": transform.to_dict(),
                        "metadata": _json(dict(item)),
                        "files": _inventory(request.directory),
                    }
                )
                for (key, _name_unused), request, item in zip(
                    batch, requests, metadata, strict=True
                )
            ]
            for (key, _name_unused), request, entry in zip(batch, requests, entries, strict=True):
                _atomic_json(request.directory / "entry.json", entry)
                _replace_atomic_file(request.directory, cache / "entries" / key)
                produced += 1
        batches += 1
        if on_batch is not None:
            on_batch({"produced": produced, "reused": reused, "batches": batches})
    ready = {key for key in sources if (cache / "entries" / key).exists()}
    return DatasetAssetPreparation(
        cache,
        {
            "format": "arti.asset-preparation-report",
            "format_version": 1,
            "dataset_sha256": manifest["sha256"],
            "transform": transform.to_dict(),
            "roles": roles,
            "assets": assets,
            "complete": len(ready) == len(sources),
            "unique_sources": len(sources),
            "produced": produced,
            "reused": reused,
            "batches": batches,
            "unavailable_assets": sum(key is None for key in assets.values()),
        },
    )


__all__ = [
    "AssetReference",
    "DatasetSnapshotConfig",
    "DatasetSnapshot",
    "create_dataset_snapshot",
    "inspect_dataset_snapshot",
    "AssetTransformConfig",
    "AssetPreparationInput",
    "DatasetAssetPreparation",
    "prepare_dataset_assets",
]
