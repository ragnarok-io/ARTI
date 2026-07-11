"""Runtime context for fit-inserted adapters."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

from torch import Tensor

from .batch_schema import attention_mask_to_visibility


@dataclass(frozen=True)
class FitRuntimeContext:
    mask: Tensor | None = None
    visibility: Tensor | None = None
    coord: Tensor | None = None
    observer_coord: Tensor | None = None
    frame_operators: Tensor | None = None


@dataclass(frozen=True)
class RuntimeFieldConfig:
    mask_key: str | None = None
    coord_key: str | None = None
    observer_coord_key: str | None = None
    frame_operators_key: str | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, object]) -> "RuntimeFieldConfig":
        return cls(
            mask_key=_optional_str(payload.get("mask_key")),
            coord_key=_optional_str(payload.get("coord_key")),
            observer_coord_key=_optional_str(payload.get("observer_coord_key")),
            frame_operators_key=_optional_str(payload.get("frame_operators_key")),
        )

    def to_dict(self) -> dict[str, str | None]:
        return {
            "mask_key": self.mask_key,
            "coord_key": self.coord_key,
            "observer_coord_key": self.observer_coord_key,
            "frame_operators_key": self.frame_operators_key,
        }

    def has_values(self) -> bool:
        return any(value is not None for value in self.to_dict().values())


_CONTEXT: ContextVar[FitRuntimeContext | None] = ContextVar("arti_fit_runtime_context", default=None)


def current_context() -> FitRuntimeContext | None:
    return _CONTEXT.get()


@contextmanager
def adapter_context(
    *,
    attention_mask: Tensor | None = None,
    mask: Tensor | None = None,
    visibility: Tensor | None = None,
    causal: bool = False,
    coord: Tensor | None = None,
    observer_coord: Tensor | None = None,
    frame_operators: Tensor | None = None,
) -> Iterator[None]:
    if attention_mask is not None and mask is not None:
        raise ValueError("adapter_context accepts either attention_mask or mask, not both")
    resolved_mask = attention_mask if attention_mask is not None else mask
    resolved_visibility = visibility
    if resolved_visibility is None and resolved_mask is not None:
        resolved_visibility = attention_mask_to_visibility(resolved_mask, causal=causal)
    token = _CONTEXT.set(
        FitRuntimeContext(
            mask=resolved_mask,
            visibility=resolved_visibility,
            coord=coord,
            observer_coord=observer_coord,
            frame_operators=frame_operators,
        )
    )
    try:
        yield
    finally:
        _CONTEXT.reset(token)


def runtime_kwargs_from_batch(batch: dict[str, object], fields: RuntimeFieldConfig | None = None) -> dict[str, Tensor | None]:
    fields = RuntimeFieldConfig() if fields is None else fields
    return {
        "attention_mask": _tensor_value(batch, *(field_aliases(fields.mask_key, "attention_mask", "mask", "padding_mask"))),
        "coord": _tensor_value(batch, *(field_aliases(fields.coord_key, "coord", "arti_coord"))),
        "observer_coord": _tensor_value(batch, *(field_aliases(fields.observer_coord_key, "observer_coord", "arti_observer_coord", "next_token_coord"))),
        "frame_operators": _tensor_value(batch, *(field_aliases(fields.frame_operators_key, "frame_operators", "arti_frame_operators", "inverse_bank"))),
    }


def runtime_keys(fields: RuntimeFieldConfig | None = None) -> set[str]:
    fields = RuntimeFieldConfig() if fields is None else fields
    keys = {
        "coord",
        "arti_coord",
        "observer_coord",
        "arti_observer_coord",
        "next_token_coord",
        "frame_operators",
        "arti_frame_operators",
        "inverse_bank",
    }
    keys.update(value for value in fields.to_dict().values() if value is not None)
    return keys


def field_aliases(preferred: str | None, *aliases: str) -> tuple[str, ...]:
    if preferred is None:
        return aliases
    return (preferred, *tuple(alias for alias in aliases if alias != preferred))


def _tensor_value(batch: dict[str, object], *keys: str) -> Tensor | None:
    for key in keys:
        value = batch.get(key)
        if isinstance(value, Tensor):
            return value
    return None


def _optional_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)
