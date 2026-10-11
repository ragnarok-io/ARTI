"""Derived local chat tokens, independent of task checkpoints and model assets."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile

from safetensors import SafetensorError
from safetensors.torch import load_file, save_file
import torch

from .serialization import _atomic_json, _file_sha256, _replace_atomic_file


def load_prepared_chat(directory: Path, records, identity):
    from .recipe_chat import ChatExample

    manifest_path = directory / "manifest.json"
    if not manifest_path.exists():
        return None, "missing"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if (
            manifest["format"] != "arti.prepared-chat"
            or manifest["format_version"] != 1
            or manifest["identity"] != identity
        ):
            raise ValueError("prepared chat identity differs")
        tensor_path = directory / "tokens.safetensors"
        if _file_sha256(tensor_path) != manifest["tensors_sha256"]:
            raise ValueError("prepared chat tensors differ")
        tensors = load_file(str(tensor_path), device="cpu")
        examples, dropped = {}, {}
        for role, rows in records.items():
            split = manifest["splits"][role]
            dropped[role] = split["dropped_ids"]
            excluded = set(dropped[role])
            if len(excluded) != len(dropped[role]) or not excluded <= {row["id"] for row in rows}:
                raise ValueError("invalid prepared chat dropped IDs")
            retained = [row for row in rows if row["id"] not in excluded]
            if [row["id"] for row in retained] != split["ids"]:
                raise ValueError("prepared chat record order differs")
            ids, labels = tensors[f"{role}.input_ids"], tensors[f"{role}.labels"]
            offsets = tensors[f"{role}.offsets"]
            if (
                ids.dtype != torch.long
                or labels.dtype != torch.long
                or offsets.dtype != torch.long
                or ids.ndim != 1
                or labels.shape != ids.shape
                or offsets.shape != (len(retained) + 1,)
                or len(split["prompt_lengths"]) != len(retained)
            ):
                raise ValueError("invalid prepared chat tensor layout")
            positions = offsets.tolist()
            if positions[0] != 0 or positions[-1] != len(ids):
                raise ValueError("invalid prepared chat offsets")
            items = []
            for index, row in enumerate(retained):
                start, end = positions[index : index + 2]
                prompt = split["prompt_lengths"][index]
                if not start < end or type(prompt) is not int or not 0 < prompt < end - start:
                    raise ValueError("invalid prepared chat prompt boundary")
                items.append(
                    ChatExample(
                        row["id"], tuple(row["messages"]), ids[start:end], labels[start:end], prompt
                    )
                )
            examples[role] = items
        return (examples, dropped), "hit"
    except (OSError, ValueError, KeyError, TypeError, IndexError, SafetensorError):
        return None, "invalid"


def save_prepared_chat(directory: Path, examples, dropped, identity):
    directory.parent.mkdir(parents=True, exist_ok=True)
    tensors, splits = {}, {}
    for role, items in examples.items():
        offsets = [0]
        for item in items:
            offsets.append(offsets[-1] + len(item.input_ids))
        tensors[f"{role}.input_ids"] = (
            torch.cat([item.input_ids for item in items])
            if items
            else torch.empty(0, dtype=torch.long)
        )
        tensors[f"{role}.labels"] = (
            torch.cat([item.labels for item in items])
            if items
            else torch.empty(0, dtype=torch.long)
        )
        tensors[f"{role}.offsets"] = torch.tensor(offsets, dtype=torch.long)
        splits[role] = {
            "ids": [item.id for item in items],
            "prompt_lengths": [item.prompt_length for item in items],
            "dropped_ids": list(dropped[role]),
        }
    # A unique staging directory keeps concurrent identical-key writers independent.
    with tempfile.TemporaryDirectory(prefix=".chat-", dir=directory.parent) as temporary:
        staging = Path(temporary)
        save_file(tensors, str(staging / "tokens.safetensors"))
        _atomic_json(
            staging / "manifest.json",
            {
                "format": "arti.prepared-chat",
                "format_version": 1,
                "identity": identity,
                "splits": splits,
                "tensors_sha256": _file_sha256(staging / "tokens.safetensors"),
            },
        )
        directory.mkdir(parents=True, exist_ok=True)
        _replace_atomic_file(staging / "tokens.safetensors", directory / "tokens.safetensors")
        _replace_atomic_file(staging / "manifest.json", directory / "manifest.json")
