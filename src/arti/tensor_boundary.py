"""Model-agnostic tensor boundary discovery and layout conversion."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch.nn as nn
from torch import Tensor


TensorPath = tuple[Any, ...]
PRIMARY_TENSOR_KEYS = (
    "last_hidden_state",
    "hidden_state",
    "sample",
    "logits",
    "output",
)


@dataclass(frozen=True)
class TensorLayout:
    """Reversible view of an arbitrary batched tensor as latent tokens."""

    shape: tuple[int, ...]
    batch_axis: int
    feature_axis: int

    def __post_init__(self) -> None:
        if len(self.shape) < 2 or any(value <= 0 for value in self.shape):
            raise ValueError("tensor layout shape must contain at least two positive dimensions")
        rank = len(self.shape)
        batch_axis = self.batch_axis % rank
        feature_axis = self.feature_axis % rank
        if batch_axis == feature_axis:
            raise ValueError("batch_axis and feature_axis must be different")
        object.__setattr__(self, "batch_axis", batch_axis)
        object.__setattr__(self, "feature_axis", feature_axis)

    @classmethod
    def infer(cls, tensor: Tensor, module: nn.Module | None = None) -> "TensorLayout":
        if not isinstance(tensor, Tensor) or not tensor.is_floating_point() or tensor.ndim < 2:
            raise ValueError("tensor boundary requires a floating Tensor with rank >= 2")
        batch_axis = 1 if _time_first_recurrent(module, tensor) else 0
        feature_axis = _infer_feature_axis(module, tensor, batch_axis=batch_axis)
        return cls(tuple(int(value) for value in tensor.shape), batch_axis, feature_axis)

    @property
    def feature_dim(self) -> int:
        return self.shape[self.feature_axis]

    @property
    def token_count(self) -> int:
        count = 1
        for axis, value in enumerate(self.shape):
            if axis not in {self.batch_axis, self.feature_axis}:
                count *= value
        return count

    def pack(self, tensor: Tensor) -> Tensor:
        if tuple(tensor.shape) != self.shape:
            raise ValueError(f"tensor shape {tuple(tensor.shape)} does not match layout {self.shape}")
        order = self._packed_axes()
        packed = tensor.permute(order)
        batch = self.shape[self.batch_axis]
        if self.token_count == 1 and tensor.ndim == 2:
            return packed.reshape(batch, self.feature_dim)
        return packed.reshape(batch, self.token_count, self.feature_dim)

    def restore(self, sequence: Tensor) -> Tensor:
        batch = self.shape[self.batch_axis]
        expected = (
            (batch, self.feature_dim)
            if len(self.shape) == 2
            else (batch, self.token_count, self.feature_dim)
        )
        if tuple(sequence.shape) != expected:
            raise ValueError(
                f"latent sequence shape {tuple(sequence.shape)} does not match expected {expected}"
            )
        order = self._packed_axes()
        packed_shape = tuple(self.shape[axis] for axis in order)
        packed = sequence.reshape(packed_shape)
        inverse = [0] * len(order)
        for packed_axis, original_axis in enumerate(order):
            inverse[original_axis] = packed_axis
        return packed.permute(tuple(inverse)).contiguous()

    def _packed_axes(self) -> tuple[int, ...]:
        middle = tuple(
            axis
            for axis in range(len(self.shape))
            if axis not in {self.batch_axis, self.feature_axis}
        )
        return (self.batch_axis, *middle, self.feature_axis)


def find_primary_tensor(output: Any) -> tuple[Tensor, TensorPath] | None:
    """Find the first compatible tensor leaf without assuming a model family."""

    if isinstance(output, Tensor):
        if output.is_floating_point() and output.ndim >= 2:
            return output, ()
        return None
    if isinstance(output, Mapping):
        keys = tuple(output)
        ordered = tuple(key for key in PRIMARY_TENSOR_KEYS if key in output) + tuple(
            key for key in keys if key not in PRIMARY_TENSOR_KEYS
        )
        for key in ordered:
            found = find_primary_tensor(output[key])
            if found is not None:
                tensor, path = found
                return tensor, (key, *path)
        return None
    if isinstance(output, (tuple, list)):
        for index, value in enumerate(output):
            found = find_primary_tensor(value)
            if found is not None:
                tensor, path = found
                return tensor, (index, *path)
    return None


def replace_tensor_at_path(output: Any, path: TensorPath, replacement: Tensor) -> Any:
    """Return an output tree with one tensor leaf replaced."""

    if not path:
        return replacement
    head, *tail = path
    remainder = tuple(tail)
    if isinstance(output, Mapping):
        value = replace_tensor_at_path(output[head], remainder, replacement)
        return _replace_mapping_value(output, head, value)
    if isinstance(output, list) and isinstance(head, int):
        values = list(output)
        values[head] = replace_tensor_at_path(values[head], remainder, replacement)
        return values
    if isinstance(output, tuple) and isinstance(head, int):
        values = list(output)
        values[head] = replace_tensor_at_path(values[head], remainder, replacement)
        if hasattr(output, "_fields"):
            return type(output)(*values)
        return tuple(values)
    raise TypeError(f"tensor path {path!r} does not match output type {type(output).__name__}")


def tensor_at_path(tree: Any, path: TensorPath) -> Tensor:
    """Return the tensor leaf at an exact path discovered by ``find_primary_tensor``."""

    value = tree
    for part in path:
        if isinstance(value, Mapping):
            value = value[part]
        elif isinstance(value, (tuple, list)) and isinstance(part, int):
            value = value[part]
        else:
            raise TypeError(f"tensor path {path!r} does not match value type {type(value).__name__}")
    if not isinstance(value, Tensor):
        raise TypeError(f"tensor path {path!r} resolves to {type(value).__name__}, not Tensor")
    return value


def _replace_mapping_value(output: Mapping, key: Any, value: Any) -> Any:
    if type(output) is dict:
        replaced = dict(output)
        replaced[key] = value
        return replaced
    try:
        replaced = output.copy()
        replaced[key] = value
        return replaced
    except Exception:
        replaced = dict(output)
        replaced[key] = value
        return replaced


def _time_first_recurrent(module: nn.Module | None, tensor: Tensor) -> bool:
    return (
        tensor.ndim == 3
        and isinstance(module, (nn.RNN, nn.LSTM, nn.GRU))
        and not bool(getattr(module, "batch_first", False))
    )


def _infer_feature_axis(
    module: nn.Module | None,
    tensor: Tensor,
    *,
    batch_axis: int,
) -> int:
    if tensor.ndim <= 3:
        return tensor.ndim - 1
    channel_dims = _declared_channel_dims(module)
    matching = [
        axis
        for axis, value in enumerate(tensor.shape)
        if axis != batch_axis and int(value) in channel_dims
    ]
    if 1 in matching:
        return 1
    if len(matching) == 1:
        return matching[0]
    if isinstance(
        module,
        (
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            nn.GroupNorm,
        ),
    ):
        return 1
    return tensor.ndim - 1


def _declared_channel_dims(module: nn.Module | None) -> set[int]:
    if module is None:
        return set()
    values: set[int] = set()
    config = getattr(module, "config", None)
    for owner in (module, config):
        if owner is None:
            continue
        for name in ("out_channels", "in_channels", "num_channels", "num_features"):
            value = owner.get(name) if isinstance(owner, Mapping) else getattr(owner, name, None)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                values.add(value)
    return values


__all__ = [
    "PRIMARY_TENSOR_KEYS",
    "TensorLayout",
    "TensorPath",
    "find_primary_tensor",
    "replace_tensor_at_path",
    "tensor_at_path",
]
