"""Batch schema helpers for pretrained-model adaptation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class TensorField:
    name: str
    shape: tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class BatchSchema:
    kind: str
    tensor_fields: tuple[TensorField, ...]
    input_keys: tuple[str, ...]
    label_key: str | None = None
    mask_key: str | None = None
    token_key: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "tensor_fields": [field.__dict__ for field in self.tensor_fields],
            "input_keys": list(self.input_keys),
            "label_key": self.label_key,
            "mask_key": self.mask_key,
            "token_key": self.token_key,
        }


def infer_batch_schema(sample_batch: Any | None) -> BatchSchema | None:
    if sample_batch is None:
        return None
    if isinstance(sample_batch, dict):
        fields = tuple(
            TensorField(name=key, shape=tuple(int(dim) for dim in value.shape), dtype=str(value.dtype))
            for key, value in sample_batch.items()
            if torch.is_tensor(value)
        )
        label_key = next((key for key in ("labels", "label", "y") if key in sample_batch), None)
        mask_key = next((key for key in ("attention_mask", "mask", "padding_mask") if key in sample_batch), None)
        token_key = next((key for key in ("input_ids", "tokens", "token_ids") if key in sample_batch), None)
        input_keys = tuple(key for key in sample_batch if key != label_key)
        return BatchSchema(kind="dict", tensor_fields=fields, input_keys=input_keys, label_key=label_key, mask_key=mask_key, token_key=token_key)
    if isinstance(sample_batch, (tuple, list)):
        fields = tuple(
            TensorField(name=str(index), shape=tuple(int(dim) for dim in value.shape), dtype=str(value.dtype))
            for index, value in enumerate(sample_batch)
            if torch.is_tensor(value)
        )
        return BatchSchema(kind="tuple", tensor_fields=fields, input_keys=tuple(str(index) for index in range(max(0, len(sample_batch) - 1))), label_key=str(len(sample_batch) - 1) if sample_batch else None)
    if torch.is_tensor(sample_batch):
        return BatchSchema(kind="tensor", tensor_fields=(TensorField(name="0", shape=tuple(int(dim) for dim in sample_batch.shape), dtype=str(sample_batch.dtype)),), input_keys=("0",))
    return BatchSchema(kind=type(sample_batch).__name__, tensor_fields=(), input_keys=())


def attention_mask_to_visibility(attention_mask: Tensor, *, causal: bool = False) -> Tensor:
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [B, N]")
    mask = attention_mask.to(dtype=torch.bool)
    visibility = mask.unsqueeze(1) & mask.unsqueeze(2)
    if causal:
        tokens = mask.shape[1]
        causal_mask = torch.ones(tokens, tokens, dtype=torch.bool, device=mask.device).tril()
        visibility = visibility & causal_mask.unsqueeze(0)
    return visibility
