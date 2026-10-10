"""Local supervised records and resumable batches for task recipes."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class ClassificationFields:
    id: str = "id"
    features: str = "features"
    label: str = "label"
    split: str = "split"

    def __post_init__(self) -> None:
        names = tuple(asdict(self).values())
        if any(not isinstance(name, str) or not name for name in names):
            raise ValueError("classification fields must be nonempty strings")
        if len(set(names)) != len(names):
            raise ValueError("classification fields must be distinct")


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at line {number}: {error.msg}") from error
            if not isinstance(row, dict):
                raise ValueError(f"JSONL line {number} must be an object")
            yield row


def feature_vector(value: Any, *, expected_size: int | None = None) -> list[float]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("features must be a nonempty numeric vector")
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in value):
        raise ValueError("features must contain numbers, not labels or booleans")
    result = [float(item) for item in value]
    if any(not math.isfinite(item) for item in result):
        raise ValueError("features must contain finite values")
    if expected_size is not None and len(result) != expected_size:
        raise ValueError(f"expected {expected_size} features, got {len(result)}")
    return result


def json_fingerprint(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ClassificationSplit:
    ids: tuple[str, ...]
    features: Tensor
    labels: Tensor
    fingerprint: str

    def __len__(self) -> int:
        return len(self.ids)

    def batches(self, batch_size: int) -> list[dict[str, Tensor]]:
        _positive_int(batch_size, "batch_size")
        return [
            {
                "features": self.features[start : start + batch_size],
                "labels": self.labels[start : start + batch_size],
            }
            for start in range(0, len(self), batch_size)
        ]


class ClassificationDataset:
    """Dense JSONL classification data with caller-supplied split membership.

    Label inference and preprocessing use train records only. This first local
    adapter is in-memory; it is not a streaming/sharded data backend.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        fields: ClassificationFields | None = None,
        labels: Sequence[str] | None = None,
    ) -> None:
        self.fields = fields or ClassificationFields()
        records: dict[str, list[dict[str, Any]]] = {
            name: [] for name in ("train", "validation", "test")
        }
        seen: set[str] = set()
        size: int | None = None
        for row in read_jsonl(path):
            identifier = row.get(self.fields.id)
            if not isinstance(identifier, str) or not identifier:
                raise ValueError("every supervised record requires a nonempty string id")
            if identifier in seen:
                raise ValueError(f"duplicate record id {identifier!r}; splits must not overlap")
            seen.add(identifier)
            role = row.get(self.fields.split)
            if role not in records:
                raise ValueError("split must be train, validation or test")
            label = row.get(self.fields.label)
            if not isinstance(label, str) or not label:
                raise ValueError("classification labels must be nonempty strings")
            vector = feature_vector(row.get(self.fields.features), expected_size=size)
            size = len(vector)
            records[role].append({"id": identifier, "features": vector, "label": label})
        if not records["train"] or not records["validation"]:
            raise ValueError(
                "a classification recipe requires separate train and validation records"
            )
        inferred = tuple(sorted({row["label"] for row in records["train"]}))
        self.labels = inferred if labels is None else tuple(labels)
        if (
            len(self.labels) < 2
            or len(set(self.labels)) != len(self.labels)
            or any(not isinstance(label, str) or not label for label in self.labels)
        ):
            raise ValueError("labels must name at least two distinct classes")
        if set(inferred) != set(self.labels):
            raise ValueError("declared classes must match classes present in the training split")
        lookup = {label: index for index, label in enumerate(self.labels)}
        self.input_dim = int(size)
        self.fingerprint = json_fingerprint(records)
        self.splits = {}
        for role, rows in records.items():
            unknown = {row["label"] for row in rows} - lookup.keys()
            if unknown:
                raise ValueError(f"{role} contains classes absent from train: {sorted(unknown)}")
            features = torch.tensor([row["features"] for row in rows], dtype=torch.float32).reshape(
                -1, self.input_dim
            )
            if not bool(torch.isfinite(features).all()):
                raise ValueError("features must be representable as finite float32 values")
            self.splits[role] = ClassificationSplit(
                tuple(row["id"] for row in rows),
                features,
                torch.tensor([lookup[row["label"]] for row in rows], dtype=torch.long),
                json_fingerprint(rows),
            )

    def manifest(self, *, include_ids: bool = False) -> dict[str, Any]:
        result = {
            "sha256": self.fingerprint,
            "input_dim": self.input_dim,
            "labels": list(self.labels),
            "fields": asdict(self.fields),
            "splits": {
                role: {"samples": len(split), "sha256": split.fingerprint}
                for role, split in self.splits.items()
            },
        }
        if include_ids:
            for role, split in self.splits.items():
                result["splits"][role]["ids"] = list(split.ids)
        return result

    def preprocessing(self, mode: str) -> tuple[Tensor, Tensor]:
        if mode == "none":
            return torch.zeros(self.input_dim), torch.ones(self.input_dim)
        if mode != "standardize":
            raise ValueError("preprocessing must be standardize or none")
        features = self.splits["train"].features
        mean = features.mean(dim=0)
        scale = features.std(dim=0, unbiased=False)
        return mean, torch.where(scale > 0, scale, torch.ones_like(scale))


class ClassificationBatches:
    """Epoch-shuffled batches whose order/cursor are part of a checkpoint."""

    def __init__(self, split: ClassificationSplit, *, batch_size: int, seed: int) -> None:
        _positive_int(batch_size, "batch_size")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        self.split = split
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.cursor = 0

    def __iter__(self) -> Iterator[dict[str, Tensor]]:
        if self.cursor == len(self.split):
            self.epoch += 1
            self.cursor = 0
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.split), generator=generator)
        while self.cursor < len(self.split):
            indices = order[self.cursor : self.cursor + self.batch_size]
            self.cursor += len(indices)
            yield {"features": self.split.features[indices], "labels": self.split.labels[indices]}
        self.epoch += 1
        self.cursor = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "epoch": self.epoch,
            "cursor": self.cursor,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "split_sha256": self.split.fingerprint,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        expected = self.state_dict()
        if any(state.get(key) != expected[key] for key in ("seed", "batch_size", "split_sha256")):
            raise ValueError("resume data/order does not match the checkpoint")
        epoch, cursor = state.get("epoch"), state.get("cursor")
        if (
            type(epoch) is not int
            or epoch < 0
            or type(cursor) is not int
            or not 0 <= cursor <= len(self.split)
        ):
            raise ValueError("invalid classification data cursor")
        self.epoch, self.cursor = epoch, cursor


def _positive_int(value: int, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


__all__ = [
    "ClassificationBatches",
    "ClassificationDataset",
    "ClassificationFields",
    "ClassificationSplit",
]
