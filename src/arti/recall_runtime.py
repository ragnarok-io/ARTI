"""Explicit forward-only composition of a Recall writer and reader."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from ._recall_state import (
    RECALL_STATE_COMPONENT_REF,
    RECALL_STATE_SCHEMA_VERSION,
    RecallState,
    migrate_recall_state,
)
from .component_registry import (
    ComponentRef,
    component_provenance,
    component_spec,
    validate_component_provenance,
)
from .recall_experts import module_behavior_fingerprint, module_structure_fingerprint


RECALL_RUNTIME_CONTRACT_VERSION = 1


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _is_sha256(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _component_descriptor(module: nn.Module) -> dict[str, Any]:
    """Capture identity and schema without binding a runtime to trainability."""

    descriptor = {
        **component_spec(module).to_dict(),
        "component_provenance": component_provenance(module),
    }
    descriptor["structure_fingerprint"] = module_structure_fingerprint(module)
    descriptor["behavior_fingerprint"] = module_behavior_fingerprint(module)
    return {**descriptor, "fingerprint": _sha256_json(descriptor)}


def _formula_descriptor(reader: nn.Module) -> dict[str, Any]:
    formula = getattr(reader, "formula", None)
    manifest_factory = getattr(reader, "formula_manifest", None)
    manifest = None
    if callable(manifest_factory):
        candidate = manifest_factory()
        if callable(getattr(candidate, "to_dict", None)):
            manifest = candidate.to_dict()
    lock = getattr(reader, "formula_lock", None)
    content = {
        "reference": getattr(reader, "formula_id", None),
        "manifest": manifest,
        "lock": lock.to_dict() if callable(getattr(lock, "to_dict", None)) else None,
        "module": (
            None
            if not isinstance(formula, nn.Module)
            else {
                "class": f"{formula.__class__.__module__}.{formula.__class__.__qualname__}",
                "structure_fingerprint": module_structure_fingerprint(formula),
                "behavior_fingerprint": module_behavior_fingerprint(formula),
            }
        ),
    }
    return {**content, "fingerprint": _sha256_json(content)}


def _bank_layout_descriptor(slots: int, hidden_dim: int) -> dict[str, Any]:
    content = {
        "kind": "values-only",
        "shape": ["B", slots, hidden_dim],
        "slots": slots,
        "hidden_dim": hidden_dim,
        "concat_axis": 1,
    }
    return {**content, "fingerprint": _sha256_json(content)}


def _state_descriptor(slots: int, hidden_dim: int) -> dict[str, Any]:
    content = {
        "ref": RECALL_STATE_COMPONENT_REF,
        "variant": "values-only",
        "lifecycle": "alpha",
        "schema_version": RECALL_STATE_SCHEMA_VERSION,
        "shape": ["B", slots, hidden_dim],
        "fields": {
            "value": {
                "kind": "tensor",
                "dtype": "runtime",
                "shape": ["B", slots, hidden_dim],
            },
            "step": {"kind": "scalar", "dtype": "int64"},
            "contract_fingerprint": {"kind": "bytes", "length": 32},
            "schema_version": {"kind": "scalar", "dtype": "int64"},
        },
    }
    return {**content, "fingerprint": _sha256_json(content)}


def _validate_component_descriptor(name: str, value: Mapping[str, Any]) -> None:
    required = {
        "path",
        "api",
        "ref",
        "mechanism_id",
        "mechanism_version",
        "variant",
        "lifecycle",
        "config_schema_version",
        "state_schema_version",
        "config",
        "config_fingerprint",
        "parameter_schema_fingerprint",
        "dependencies",
        "capabilities",
        "component_provenance",
        "structure_fingerprint",
        "behavior_fingerprint",
        "fingerprint",
    }
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError(f"Recall runtime {name} descriptor has missing or unknown fields")
    identity = ComponentRef.parse(str(value["ref"]))
    if value["mechanism_id"] != identity.mechanism_id or value["mechanism_version"] != identity.version:
        raise ValueError(f"Recall runtime {name} component identity is inconsistent")
    if value["config_fingerprint"] != _sha256_json(value["config"]):
        raise ValueError(f"Recall runtime {name} config fingerprint is invalid")
    for field in ("structure_fingerprint", "behavior_fingerprint", "fingerprint"):
        if not _is_sha256(value[field]):
            raise ValueError(f"Recall runtime {name} {field} is invalid")
    validate_component_provenance(value["component_provenance"])
    root = next(
        (
            item
            for item in value["component_provenance"]["components"]
            if item["path"] == "$"
        ),
        None,
    )
    if root is None or any(
        value[field] != root[field]
        for field in (
            "api",
            "ref",
            "mechanism_id",
            "mechanism_version",
            "variant",
            "lifecycle",
            "config_schema_version",
            "state_schema_version",
            "config",
            "config_fingerprint",
            "parameter_schema_fingerprint",
            "dependencies",
            "capabilities",
        )
    ):
        raise ValueError(f"Recall runtime {name} descriptor disagrees with its component graph")
    if value["fingerprint"] != _sha256_json(
        {key: value[key] for key in required if key != "fingerprint"}
    ):
        raise ValueError(f"Recall runtime {name} fingerprint does not match its contents")


def _validate_formula_descriptor(value: Mapping[str, Any]) -> None:
    required = {"reference", "manifest", "lock", "module", "fingerprint"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ValueError("Recall runtime Formula descriptor has missing or unknown fields")
    if value["reference"] is not None and not isinstance(value["reference"], str):
        raise ValueError("Recall runtime Formula reference is invalid")
    if value["fingerprint"] != _sha256_json(
        {key: value[key] for key in required if key != "fingerprint"}
    ):
        raise ValueError("Recall runtime Formula fingerprint does not match its contents")
    if value["module"] is not None:
        module = value["module"]
        module_keys = {"class", "structure_fingerprint", "behavior_fingerprint"}
        if not isinstance(module, Mapping) or set(module) != module_keys:
            raise ValueError("Recall runtime Formula module descriptor is invalid")
        if not all(_is_sha256(module[key]) for key in ("structure_fingerprint", "behavior_fingerprint")):
            raise ValueError("Recall runtime Formula module fingerprints are invalid")


def _validate_fingerprinted_payload(
    name: str,
    value: Mapping[str, Any],
    fields: set[str],
) -> None:
    if not isinstance(value, Mapping) or set(value) != fields | {"fingerprint"}:
        raise ValueError(f"Recall runtime {name} descriptor has missing or unknown fields")
    if value["fingerprint"] != _sha256_json({key: value[key] for key in fields}):
        raise ValueError(f"Recall runtime {name} fingerprint does not match its contents")


@dataclass(frozen=True)
class RecallRuntimeContract:
    """Serializable structural compatibility contract for one writer/reader pair.

    The contract identifies declared behavior and tensor schemas. It does not
    hash learned tensor contents, because an Updater may continue training after
    Runtime construction; exact asset identity belongs to the enclosing
    artifact lock.
    """

    reader: dict[str, Any]
    updater: dict[str, Any]
    formula: dict[str, Any]
    bank_layout: dict[str, Any]
    state: dict[str, Any]
    slots: int
    hidden_dim: int
    state_schema_version: int = RECALL_STATE_SCHEMA_VERSION
    schema_version: int = RECALL_RUNTIME_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != RECALL_RUNTIME_CONTRACT_VERSION:
            raise ValueError("unsupported Recall runtime contract schema version")
        if self.state_schema_version != RECALL_STATE_SCHEMA_VERSION:
            raise ValueError("unsupported Recall state schema version")
        if self.slots <= 0 or self.hidden_dim <= 0:
            raise ValueError("Recall runtime slots and hidden_dim must be positive")
        _validate_component_descriptor("reader", self.reader)
        _validate_component_descriptor("updater", self.updater)
        _validate_formula_descriptor(self.formula)
        _validate_fingerprinted_payload(
            "bank layout",
            self.bank_layout,
            {"kind", "shape", "slots", "hidden_dim", "concat_axis"},
        )
        _validate_fingerprinted_payload(
            "state",
            self.state,
            {"ref", "variant", "lifecycle", "schema_version", "shape", "fields"},
        )
        if self.bank_layout["slots"] != self.slots or self.bank_layout["hidden_dim"] != self.hidden_dim:
            raise ValueError("Recall runtime bank layout does not match its dimensions")
        if self.state["ref"] != RECALL_STATE_COMPONENT_REF or self.state["schema_version"] != self.state_schema_version:
            raise ValueError("Recall runtime state descriptor does not match its schema")

    def _content(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "state_schema_version": self.state_schema_version,
            "reader": self.reader,
            "updater": self.updater,
            "formula": self.formula,
            "bank_layout": self.bank_layout,
            "state": self.state,
            "slots": self.slots,
            "hidden_dim": self.hidden_dim,
        }

    @property
    def fingerprint(self) -> str:
        return _sha256_json(self._content())

    def to_dict(self) -> dict[str, Any]:
        return {**self._content(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RecallRuntimeContract":
        if not isinstance(value, dict):
            raise TypeError("Recall runtime contract must be a mapping")
        required = {
            "schema_version",
            "state_schema_version",
            "reader",
            "updater",
            "formula",
            "bank_layout",
            "state",
            "slots",
            "hidden_dim",
            "fingerprint",
        }
        if set(value) != required:
            raise ValueError("Recall runtime contract has missing or unknown fields")
        contract = cls(
            reader=dict(value["reader"]),
            updater=dict(value["updater"]),
            formula=dict(value["formula"]),
            bank_layout=dict(value["bank_layout"]),
            state=dict(value["state"]),
            slots=int(value["slots"]),
            hidden_dim=int(value["hidden_dim"]),
            state_schema_version=int(value["state_schema_version"]),
            schema_version=int(value["schema_version"]),
        )
        if value["fingerprint"] != contract.fingerprint:
            raise ValueError("Recall runtime contract fingerprint does not match its contents")
        return contract


class RecallRuntime(nn.Module):
    """Compose an offline-trained state writer with a Recall reader.

    ``updater`` owns trainable writer parameters. ``RecallState`` owns the
    dynamic values-only Bank. ``reader`` consumes that Bank through its
    ``memory=`` argument. No state is mutated implicitly by this wrapper.
    """

    _component_reference = "arti/recall-runtime@1"

    def __init__(self, updater: nn.Module, reader: nn.Module) -> None:
        super().__init__()
        if not isinstance(updater, nn.Module) or not isinstance(reader, nn.Module):
            raise TypeError("updater and reader must be torch.nn.Module instances")
        self.updater = updater
        self.reader = reader
        slots = self._matching_dimension("slots")
        hidden_dim = self._matching_dimension("hidden_dim")
        self.contract = RecallRuntimeContract(
            reader=_component_descriptor(reader),
            updater=_component_descriptor(updater),
            formula=_formula_descriptor(reader),
            bank_layout=_bank_layout_descriptor(slots, hidden_dim),
            state=_state_descriptor(slots, hidden_dim),
            slots=slots,
            hidden_dim=hidden_dim,
        )
        for name in ("slots", "hidden_dim"):
            updater_value = getattr(updater, name, None)
            reader_value = getattr(reader, name, None)
            if (
                isinstance(updater_value, int)
                and isinstance(reader_value, int)
                and updater_value != reader_value
            ):
                raise ValueError(
                    f"updater and reader {name} must match; "
                    f"got {updater_value} and {reader_value}"
                )

    def _matching_dimension(self, name: str) -> int:
        values = [getattr(module, name, None) for module in (self.updater, self.reader)]
        known = [value for value in values if isinstance(value, int) and not isinstance(value, bool)]
        if not known or any(value <= 0 for value in known) or len(set(known)) != 1:
            raise ValueError(f"updater and reader must expose one matching positive {name}")
        return known[0]

    @property
    def contract_fingerprint(self) -> str:
        return self.contract.fingerprint

    @property
    def slots(self) -> int:
        return self._required_dimension("slots")

    @property
    def hidden_dim(self) -> int:
        return self._required_dimension("hidden_dim")

    def _required_dimension(self, name: str) -> int:
        value = getattr(self.updater, name, None)
        if value is None:
            value = getattr(self.reader, name, None)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise AttributeError(
                f"{name} must be exposed as a positive integer by updater or reader"
            )
        return int(value)

    def initial_state(
        self,
        batch_size: int,
        *,
        reference: Tensor | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> RecallState:
        """Create a zero dynamic Bank state for this runtime."""

        return RecallState.zeros(
            batch_size,
            self.slots,
            self.hidden_dim,
            reference=reference,
            device=device,
            dtype=dtype,
            contract_fingerprint=self.contract_fingerprint,
        )

    def migrate_state(
        self,
        payload: RecallState | Mapping[str, Tensor],
        *,
        map_location: torch.device | str | None = None,
    ) -> RecallState:
        """Explicitly adopt an old or unbound state for this runtime."""

        return migrate_recall_state(
            payload,
            contract_fingerprint=self.contract_fingerprint,
            slots=self.slots,
            hidden_dim=self.hidden_dim,
            map_location=map_location,
        )

    @staticmethod
    def _coerce_state(state: RecallState | Tensor) -> RecallState:
        if isinstance(state, RecallState):
            return state
        if isinstance(state, Tensor):
            return RecallState(state)
        raise TypeError("state must be a RecallState or floating-point Tensor")

    def _validate_state(self, state: RecallState) -> None:
        if state.schema_version != self.contract.state_schema_version:
            raise ValueError(
                "state schema version does not match the runtime; "
                f"expected {self.contract.state_schema_version}, got {state.schema_version}"
            )
        if state.contract_fingerprint != self.contract_fingerprint:
            raise ValueError(
                "state contract fingerprint does not match the runtime; "
                "create the state with runtime.initial_state() or use an explicit migration"
            )
        if state.slots != self.slots or state.hidden_dim != self.hidden_dim:
            raise ValueError(
                "state dimensions must match the runtime; "
                f"expected ({self.slots}, {self.hidden_dim}), "
                f"got ({state.slots}, {state.hidden_dim})"
            )

    def update(
        self,
        trace: Tensor,
        state: RecallState | Tensor,
        *,
        trace_mask: Tensor | None = None,
        detach_state: bool = False,
    ) -> RecallState:
        """Apply one forward state transition and return a new state."""

        current = self._coerce_state(state)
        self._validate_state(current)
        next_value = self.updater(trace, current.value, mask=trace_mask)
        if isinstance(next_value, tuple):
            raise TypeError(
                "updater must return one next-state Tensor; use its low-level "
                "forward API when transition diagnostics are required"
            )
        return current.advance(next_value, detach=detach_state)

    def write(
        self,
        trace: Tensor,
        state: RecallState | Tensor,
        *,
        mask: Tensor | None = None,
        detach_state: bool = False,
    ) -> RecallState:
        """Alias for :meth:`update` using the concise single-trace spelling."""

        return self.update(
            trace,
            state,
            trace_mask=mask,
            detach_state=detach_state,
        )

    def scan(
        self,
        traces: Tensor,
        state: RecallState | Tensor,
        *,
        mask: Tensor | None = None,
        order: Sequence[int] | None = None,
        detach_state: bool = False,
        return_snapshots: bool = False,
    ) -> RecallState | tuple[RecallState, Tensor]:
        """Apply an ordered sequence of trace updates.

        Unbatched inputs use ``traces=[L, T, H]`` and ``state=[S, H]``.
        Batched inputs use ``traces=[B, L, T, H]`` and ``state=[B, S, H]``.
        The order axis is intentionally sequential; it is not averaged or
        treated as an ordinary batch dimension.
        """

        current = self._coerce_state(state)
        if traces.ndim not in {3, 4} or traces.shape[-1] != current.hidden_dim:
            raise ValueError(
                "traces must have shape [L, T, H] or [B, L, T, H] and match state H"
            )
        batched = traces.ndim == 4
        if batched != (current.value.ndim == 3):
            raise ValueError("traces and state must use the same batch convention")
        update_count = traces.shape[1] if batched else traces.shape[0]
        if update_count <= 0:
            raise ValueError("traces must contain at least one update")
        expected_mask_shape = (
            (traces.shape[0], update_count, traces.shape[2])
            if batched
            else (update_count, traces.shape[1])
        )
        if mask is not None and (mask.dtype != torch.bool or tuple(mask.shape) != expected_mask_shape):
            raise ValueError(f"mask must be boolean with shape {expected_mask_shape}")
        if order is None:
            indices = tuple(range(update_count))
        else:
            indices = tuple(int(index) for index in order)
            if len(indices) != update_count or sorted(indices) != list(range(update_count)):
                raise ValueError("order must be a permutation of the update axis")

        snapshots: list[Tensor] = []
        for index in indices:
            trace_one = traces[:, index] if batched else traces[index]
            mask_one = None if mask is None else (mask[:, index] if batched else mask[index])
            current = self.update(
                trace_one,
                current,
                trace_mask=mask_one,
                detach_state=detach_state,
            )
            if return_snapshots:
                snapshots.append(current.value.clone())
        if not return_snapshots:
            return current
        return current, torch.stack(snapshots, dim=1 if batched else 0)

    def read(self, hidden: Tensor, state: RecallState | Tensor, **kwargs: Any) -> Any:
        """Read a dynamic Bank state through the configured Formula reader."""

        current = self._coerce_state(state)
        self._validate_state(current)
        return self.reader(hidden, memory=current.value, **kwargs)

    def forward(
        self,
        hidden: Tensor,
        trace: Tensor,
        state: RecallState | Tensor,
        *,
        trace_mask: Tensor | None = None,
        detach_state: bool = False,
        **reader_kwargs: Any,
    ) -> tuple[Any, RecallState]:
        """Write a trace, then read the resulting state into ``hidden``."""

        next_state = self.update(
            trace,
            state,
            trace_mask=trace_mask,
            detach_state=detach_state,
        )
        return self.read(hidden, next_state, **reader_kwargs), next_state


__all__ = [
    "RECALL_RUNTIME_CONTRACT_VERSION",
    "RecallRuntime",
    "RecallRuntimeContract",
    "RecallState",
    "migrate_recall_state",
]
