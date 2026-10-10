"""Text-only final-answer supervision with a model's native chat template."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .recipe_data import _positive_int, json_fingerprint, read_jsonl


@dataclass(frozen=True)
class ChatFields:
    id: str = "id"
    messages: str = "messages"
    split: str = "split"

    def __post_init__(self):
        values = tuple(asdict(self).values())
        if any(not isinstance(value, str) or not value for value in values) or len(
            set(values)
        ) != len(values):
            raise ValueError("chat fields must be distinct nonempty names")


def normalized_messages(value: Any, *, completed: bool) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise ValueError("messages must be a nonempty conversation")
    messages = []
    for message in value:
        if not isinstance(message, Mapping) or message.get("role") not in {
            "system",
            "user",
            "assistant",
        }:
            raise ValueError("this text recipe accepts system, user and assistant messages")
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError("message content must be nonempty text")
        if set(message) - {"role", "content"}:
            raise ValueError("reasoning/tool/multimodal fields require an explicit task adapter")
        messages.append({"role": message["role"], "content": message["content"]})
    if not any(message["role"] == "user" for message in messages):
        raise ValueError("a conversation requires a complete user message")
    if completed and (len(messages) < 2 or messages[-1]["role"] != "assistant"):
        raise ValueError("supervised records must end in a real assistant answer")
    if not completed and messages[-1]["role"] != "user":
        raise ValueError("generation records must end in a user message")
    return messages


def tokenizer_fingerprint(tokenizer) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        raise ValueError("the text recipe currently requires a fast tokenizer")
    declaration = json.loads(backend.to_str())
    declaration.pop("padding", None)
    declaration.pop("truncation", None)
    return json_fingerprint(
        {
            "backend": declaration,
            "special_tokens": tokenizer.special_tokens_map,
            "chat_template": tokenizer.chat_template,
            "eos_token_id": tokenizer.eos_token_id,
            "pad_token_id": tokenizer.pad_token_id,
        }
    )


@dataclass(frozen=True)
class ChatExample:
    id: str
    messages: tuple[dict[str, str], ...]
    input_ids: Tensor
    labels: Tensor
    prompt_length: int

    @property
    def target_tokens(self) -> int:
        return int(self.labels[1:].ne(-100).sum())


def encode_final_answer(
    identifier: str, messages, tokenizer, *, template_kwargs: Mapping[str, Any]
) -> ChatExample:
    messages = normalized_messages(messages, completed=True)
    if (
        not tokenizer.chat_template
        or tokenizer.eos_token_id is None
        or tokenizer.pad_token_id is None
    ):
        raise ValueError("a native chat template, EOS and PAD are required")
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True, **dict(template_kwargs)
    )
    completed = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, **dict(template_kwargs)
    )
    if not completed.startswith(prompt):
        raise ValueError(
            "chat template is not prefix-preserving for the final answer; use matching training/generation template kwargs"
        )
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    completion_ids = tokenizer(completed[len(prompt) :], add_special_tokens=False)["input_ids"]
    endings = [
        index for index, token in enumerate(completion_ids) if token == tokenizer.eos_token_id
    ]
    if not prompt_ids or not endings:
        raise ValueError("native chat template must terminate the final assistant answer with EOS")
    end = endings[-1] + 1
    if tokenizer.decode(completion_ids[end:], skip_special_tokens=False).strip():
        raise ValueError("native chat template adds non-whitespace content after the final EOS")
    completion_ids = completion_ids[:end]
    # Separate prompt/completion tokenization preserves the exact generation
    # prefix even when a BPE token could otherwise cross the response boundary.
    return ChatExample(
        identifier,
        tuple(messages),
        torch.tensor(prompt_ids + completion_ids, dtype=torch.long),
        torch.tensor([-100] * len(prompt_ids) + completion_ids, dtype=torch.long),
        len(prompt_ids),
    )


def collate_chat(
    examples: Sequence[ChatExample], *, pad_token_id: int, pad_to_multiple_of: int
) -> dict[str, Tensor]:
    if not examples:
        raise ValueError("chat batches must not be empty")
    length = max(len(example.input_ids) for example in examples)
    length = ((length + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
    ids = torch.full((len(examples), length), pad_token_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    labels = torch.full_like(ids, -100)
    for index, example in enumerate(examples):
        count = len(example.input_ids)
        ids[index, :count] = example.input_ids
        mask[index, :count] = 1
        labels[index, :count] = example.labels
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


@dataclass(frozen=True)
class ChatSplit:
    examples: tuple[ChatExample, ...]
    fingerprint: str
    pad_token_id: int
    pad_to_multiple_of: int

    def __len__(self):
        return len(self.examples)

    @property
    def ids(self):
        return tuple(example.id for example in self.examples)

    def batches(self, batch_size: int):
        _positive_int(batch_size, "batch_size")
        return [
            collate_chat(
                self.examples[start : start + batch_size],
                pad_token_id=self.pad_token_id,
                pad_to_multiple_of=self.pad_to_multiple_of,
            )
            for start in range(0, len(self), batch_size)
        ]


class ChatDataset:
    def __init__(
        self,
        path: str | Path,
        tokenizer,
        *,
        fields: ChatFields,
        template_kwargs: Mapping[str, Any],
        max_length: int,
        overlength: str,
        pad_to_multiple_of: int,
    ):
        records = {role: [] for role in ("train", "validation", "test")}
        examples = {role: [] for role in records}
        self.dropped = {role: [] for role in records}
        seen = set()
        for row in read_jsonl(path):
            identifier = row.get(fields.id)
            role = row.get(fields.split)
            if not isinstance(identifier, str) or not identifier or identifier in seen:
                raise ValueError("chat records require globally unique nonempty string ids")
            seen.add(identifier)
            if not isinstance(role, str) or role not in records:
                raise ValueError("chat split must be train, validation or test")
            messages = normalized_messages(row.get(fields.messages), completed=True)
            records[role].append({"id": identifier, "messages": messages})
            example = encode_final_answer(
                identifier, messages, tokenizer, template_kwargs=template_kwargs
            )
            if len(example.input_ids) > max_length:
                if overlength == "error":
                    raise ValueError(
                        f"record {identifier!r} has {len(example.input_ids)} tokens, exceeding max_length={max_length}; no answer was silently truncated"
                    )
                self.dropped[role].append(identifier)
                continue
            examples[role].append(example)
        if not examples["train"] or not examples["validation"]:
            raise ValueError("SFT requires nonempty independent train and validation splits")
        self.fields = fields
        self.fingerprint = json_fingerprint(records)
        self.tokenizer_fingerprint = tokenizer_fingerprint(tokenizer)
        self.splits = {
            role: ChatSplit(
                tuple(items),
                json_fingerprint(
                    [
                        {
                            "id": item.id,
                            "input_ids": item.input_ids.tolist(),
                            "labels": item.labels.tolist(),
                        }
                        for item in items
                    ]
                ),
                tokenizer.pad_token_id,
                pad_to_multiple_of,
            )
            for role, items in examples.items()
        }

    def manifest(self, *, include_ids=False):
        splits = {}
        for role, split in self.splits.items():
            splits[role] = {
                "samples": len(split),
                "sha256": split.fingerprint,
                "input_tokens": sum(len(example.input_ids) for example in split.examples),
                "supervised_tokens": sum(example.target_tokens for example in split.examples),
                "max_sequence_length": max(
                    (len(example.input_ids) for example in split.examples), default=0
                ),
                "dropped_samples": len(self.dropped[role]),
            }
            if include_ids:
                splits[role].update(ids=list(split.ids), dropped_ids=self.dropped[role])
        return {
            "sha256": self.fingerprint,
            "tokenizer_sha256": self.tokenizer_fingerprint,
            "fields": asdict(self.fields),
            "splits": splits,
        }


class ChatBatches:
    """Pretokenized, length-grouped epoch batches with complete data cursor."""

    def __init__(self, split: ChatSplit, *, batch_size: int, seed: int, group_by_length: bool):
        self.split, self.batch_size, self.seed = split, batch_size, seed
        self.group_by_length = group_by_length
        self.epoch = self.cursor = self.samples_seen = self.input_tokens_seen = (
            self.target_tokens_seen
        ) = 0

    def __iter__(self):
        if self.cursor == len(self.split):
            self.epoch += 1
            self.cursor = 0
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        if self.group_by_length:
            from transformers.trainer_pt_utils import get_length_grouped_indices

            order = get_length_grouped_indices(
                [len(example.input_ids) for example in self.split.examples],
                self.batch_size,
                generator=generator,
            )
        else:
            order = torch.randperm(len(self.split), generator=generator).tolist()
        while self.cursor < len(self.split):
            examples = [
                self.split.examples[index]
                for index in order[self.cursor : self.cursor + self.batch_size]
            ]
            self.cursor += len(examples)
            self.samples_seen += len(examples)
            self.input_tokens_seen += sum(len(example.input_ids) for example in examples)
            self.target_tokens_seen += sum(example.target_tokens for example in examples)
            yield collate_chat(
                examples,
                pad_token_id=self.split.pad_token_id,
                pad_to_multiple_of=self.split.pad_to_multiple_of,
            )
        self.epoch += 1
        self.cursor = 0

    def state_dict(self):
        return {
            "epoch": self.epoch,
            "cursor": self.cursor,
            "seed": self.seed,
            "batch_size": self.batch_size,
            "group_by_length": self.group_by_length,
            "split_sha256": self.split.fingerprint,
            "samples_seen": self.samples_seen,
            "input_tokens_seen": self.input_tokens_seen,
            "target_tokens_seen": self.target_tokens_seen,
        }

    def validate_state(self, state):
        current = self.state_dict()
        if any(
            state.get(key) != current[key]
            for key in ("seed", "batch_size", "group_by_length", "split_sha256")
        ):
            raise ValueError("SFT resume data/order does not match its checkpoint")
        for key in ("epoch", "cursor", "samples_seen", "input_tokens_seen", "target_tokens_seen"):
            if type(state.get(key)) is not int or state[key] < 0:
                raise ValueError("invalid SFT data cursor")
        if state["cursor"] > len(self.split):
            raise ValueError("invalid SFT data cursor")

    def load_state_dict(self, state):
        self.validate_state(state)
        for key in ("epoch", "cursor", "samples_seen", "input_tokens_seen", "target_tokens_seen"):
            setattr(self, key, state[key])


def target_token_weight(batch):
    return int(batch["labels"][..., 1:].ne(-100).sum())


__all__ = ["ChatDataset", "ChatFields", "ChatExample", "ChatSplit", "ChatBatches"]
