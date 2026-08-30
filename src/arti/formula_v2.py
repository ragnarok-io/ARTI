"""Typed, parameter-free Formula atoms and a bounded SSA reference executor."""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass, replace
from typing import ClassVar, Literal, Mapping, NamedTuple, Sequence

import torch
from torch import Tensor, nn


FORMULA_PROGRAM_V2_SCHEMA_VERSION = 2
FORMULA_TRACE_V1_SCHEMA_VERSION = 1
FORMULA_LIMITS_V1_SCHEMA_VERSION = 1
FORMULA_PROGRAM_V2_SCHEMA_REF = "arti/formula-program@2"
FORMULA_TENSOR_TYPE_V1_SCHEMA_REF = "arti/formula-tensor-type@1"
FORMULA_TRACE_V1_SCHEMA_REF = "arti/formula-trace@1"
FORMULA_LIMITS_V1_SCHEMA_REF = "arti/formula-limits@1"
FORMULA_EXECUTION_PLAN_V1_SCHEMA_REF = "arti/formula-execution-plan@1"
FORMULA_EXECUTION_PLAN_V1_SCHEMA_VERSION = 1

_AXIS_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_NAME_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]*$")
_COMPONENT_REF_RE = re.compile(
    r"^[a-z0-9][a-z0-9_.-]*/[a-z0-9][a-z0-9_.-]*@[1-9][0-9]*$"
)
_DTYPES = frozenset({"floating", "float16", "bfloat16", "float32", "float64"})
_ACCUMULATION_DTYPES = frozenset({"activation", "float32"})
_ATOM_SIGNATURES: dict[str, tuple[int, frozenset[str]]] = {
    "arti/formula-atom-contract@1": (
        2,
        frozenset({"reduce_axes", "output_axes", "accumulation_dtype"}),
    ),
    "arti/formula-atom-scale@1": (
        2,
        frozenset({"factor_axes", "accumulation_dtype"}),
    ),
    "arti/formula-atom-add@1": (2, frozenset({"accumulation_dtype"})),
    "arti/formula-atom-reduce@1": (
        1,
        frozenset({"axis", "mode", "accumulation_dtype"}),
    ),
}


class FormulaV2Error(ValueError):
    """Base error carrying a stable FormulaFabric@2 diagnostic code."""

    def __init__(self, code: str, message: str, *, path: str = "$") -> None:
        super().__init__(message)
        self.code = code
        self.path = path


class FormulaSchemaError(FormulaV2Error):
    pass


class FormulaTypeError(FormulaV2Error):
    pass


class FormulaBindingError(FormulaV2Error):
    pass


class FormulaProgramError(FormulaV2Error):
    pass


@dataclass(frozen=True)
class _FrozenJsonObject:
    items: tuple[tuple[str, object], ...]


@dataclass(frozen=True)
class FormulaLimits:
    """Static admission limits embedded in a FormulaProgram fingerprint."""

    max_bindings: int = 32
    max_slots: int = 128
    max_instructions: int = 96
    max_steps: int = 64
    max_axes: int = 8
    max_axis_extent: int = 1_048_576
    max_tensor_elements: int = 268_435_456
    max_tensor_bytes: int = 1_073_741_824
    max_working_bytes: int = 2_147_483_648
    schema_version: int = FORMULA_LIMITS_V1_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field in (
            "max_bindings",
            "max_slots",
            "max_instructions",
            "max_steps",
            "max_axes",
            "max_axis_extent",
            "max_tensor_elements",
            "max_tensor_bytes",
            "max_working_bytes",
        ):
            value = getattr(self, field)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise FormulaSchemaError("FF2_INVALID_LIMIT", f"{field} must be positive")
        if self.schema_version != FORMULA_LIMITS_V1_SCHEMA_VERSION:
            raise FormulaSchemaError("FF2_INVALID_LIMIT", "unsupported FormulaLimits schema")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_ref": FORMULA_LIMITS_V1_SCHEMA_REF,
            "schema_version": self.schema_version,
            "max_bindings": self.max_bindings,
            "max_slots": self.max_slots,
            "max_instructions": self.max_instructions,
            "max_steps": self.max_steps,
            "max_axes": self.max_axes,
            "max_axis_extent": self.max_axis_extent,
            "max_tensor_elements": self.max_tensor_elements,
            "max_tensor_bytes": self.max_tensor_bytes,
            "max_working_bytes": self.max_working_bytes,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> FormulaLimits:
        required = {
            "schema_ref",
            "schema_version",
            "max_bindings",
            "max_slots",
            "max_instructions",
            "max_steps",
            "max_axes",
            "max_axis_extent",
            "max_tensor_elements",
            "max_tensor_bytes",
            "max_working_bytes",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise FormulaSchemaError("FF2_LIMIT_SCHEMA", "limits have missing or unknown fields")
        if value["schema_ref"] != FORMULA_LIMITS_V1_SCHEMA_REF:
            raise FormulaSchemaError("FF2_LIMIT_SCHEMA", "limits schema reference is invalid")
        return cls(**{key: item for key, item in value.items() if key != "schema_ref"})


DEFAULT_FORMULA_LIMITS = FormulaLimits()


@dataclass(frozen=True)
class TensorType:
    """A named-axis tensor type with optional static extents."""

    axis_names: tuple[str, ...]
    sizes: tuple[int | None, ...]
    dtype: str = "floating"
    domain: str = "anonymous"

    def __post_init__(self) -> None:
        if isinstance(self.axis_names, (str, bytes)) or isinstance(self.sizes, (str, bytes)):
            raise FormulaSchemaError(
                "FF2_TYPE_SCHEMA",
                "TensorType axes and sizes must be sequences, not text",
                path="$.type",
            )
        axes = tuple(self.axis_names)
        sizes = tuple(self.sizes)
        object.__setattr__(self, "axis_names", axes)
        object.__setattr__(self, "sizes", sizes)
        if len(axes) != len(sizes):
            raise FormulaSchemaError(
                "FF2_TYPE_RANK_MISMATCH",
                "axis_names and sizes must have equal length",
                path="$.type",
            )
        if len(set(axes)) != len(axes):
            raise FormulaSchemaError(
                "FF2_DUPLICATE_AXIS", "TensorType axes must be unique", path="$.type.axes"
            )
        if any(not isinstance(axis, str) or not _AXIS_RE.fullmatch(axis) for axis in axes):
            raise FormulaSchemaError(
                "FF2_INVALID_AXIS", "TensorType axes must be valid identifiers", path="$.type.axes"
            )
        if any(
            size is not None
            and (isinstance(size, bool) or not isinstance(size, int) or size <= 0)
            for size in sizes
        ):
            raise FormulaSchemaError(
                "FF2_INVALID_EXTENT",
                "TensorType sizes must be positive integers or None",
                path="$.type.sizes",
            )
        if self.dtype not in _DTYPES:
            raise FormulaSchemaError(
                "FF2_INVALID_DTYPE", f"unsupported dtype contract {self.dtype!r}", path="$.type.dtype"
            )
        if not isinstance(self.domain, str) or not self.domain:
            raise FormulaSchemaError(
                "FF2_INVALID_DOMAIN", "domain must be a non-empty string", path="$.type.domain"
            )

    @classmethod
    def axes(
        cls,
        names: Sequence[str],
        *,
        sizes: Sequence[int | None] | None = None,
        dtype: str = "floating",
        domain: str = "anonymous",
    ) -> TensorType:
        names_tuple = tuple(names)
        return cls(
            names_tuple,
            (None,) * len(names_tuple) if sizes is None else tuple(sizes),
            dtype=dtype,
            domain=domain,
        )

    @classmethod
    def scalar(
        cls, *, dtype: str = "floating", domain: str = "anonymous"
    ) -> TensorType:
        return cls((), (), dtype=dtype, domain=domain)

    def size_for(self, axis: str) -> int | None:
        try:
            return self.sizes[self.axis_names.index(axis)]
        except ValueError as exc:
            raise FormulaTypeError(
                "FF2_AXIS_MISMATCH", f"axis {axis!r} is not present in {self.axis_names}"
            ) from exc

    def with_axes(self, axes: Sequence[str]) -> TensorType:
        names = tuple(axes)
        return TensorType(
            names,
            tuple(self.size_for(axis) for axis in names),
            dtype=self.dtype,
            domain=self.domain,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_ref": FORMULA_TENSOR_TYPE_V1_SCHEMA_REF,
            "axes": list(self.axis_names),
            "sizes": list(self.sizes),
            "dtype": self.dtype,
            "domain": self.domain,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TensorType:
        required = {"schema_ref", "axes", "sizes", "dtype", "domain"}
        if not isinstance(value, Mapping) or set(value) != required:
            raise FormulaSchemaError(
                "FF2_TYPE_SCHEMA", "TensorType payload contains missing or unknown fields"
            )
        if value["schema_ref"] != FORMULA_TENSOR_TYPE_V1_SCHEMA_REF:
            raise FormulaSchemaError("FF2_TYPE_SCHEMA", "TensorType schema reference is invalid")
        axes = value["axes"]
        sizes = value["sizes"]
        if not isinstance(axes, Sequence) or isinstance(axes, (str, bytes)):
            raise FormulaSchemaError("FF2_TYPE_SCHEMA", "TensorType axes must be a sequence")
        if not isinstance(sizes, Sequence) or isinstance(sizes, (str, bytes)):
            raise FormulaSchemaError("FF2_TYPE_SCHEMA", "TensorType sizes must be a sequence")
        return cls(tuple(axes), tuple(sizes), dtype=value["dtype"], domain=value["domain"])


@dataclass(frozen=True)
class InputBinding:
    name: str
    value_type: TensorType

    def __post_init__(self) -> None:
        _validate_name(self.name, field="InputBinding.name")
        if not isinstance(self.value_type, TensorType):
            raise TypeError("InputBinding.value_type must be TensorType")

    def to_dict(self) -> dict[str, object]:
        return {"kind": "input", "name": self.name, "value_type": self.value_type.to_dict()}


@dataclass(frozen=True)
class BankBinding:
    name: str
    source_ref: str
    partition_id: str
    value_type: TensorType
    asset_fingerprint: str | None = None
    route_ref: str | None = None
    bundle_id: str | None = None
    member_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _validate_name(self.name, field="BankBinding.name")
        _validate_component_ref(self.source_ref, field="BankBinding.source_ref")
        _validate_name(self.partition_id, field="BankBinding.partition_id")
        if isinstance(self.member_ids, (str, bytes)):
            raise FormulaSchemaError(
                "FF2_INVALID_BANK_BUNDLE", "member_ids must be a sequence of names"
            )
        object.__setattr__(self, "member_ids", tuple(self.member_ids))
        if not isinstance(self.value_type, TensorType):
            raise TypeError("BankBinding.value_type must be TensorType")
        for value, field in (
            (self.asset_fingerprint, "asset_fingerprint"),
            (self.route_ref, "route_ref"),
        ):
            if value is not None and (not isinstance(value, str) or not value):
                raise FormulaSchemaError("FF2_INVALID_BANK_REF", f"{field} must be None or non-empty")
        if self.asset_fingerprint is not None and (
            len(self.asset_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in self.asset_fingerprint)
        ):
            raise FormulaSchemaError(
                "FF2_INVALID_BANK_REF", "asset_fingerprint must be a SHA-256 hex digest"
            )
        if self.route_ref is not None:
            raise FormulaSchemaError(
                "FF2_UNSUPPORTED_ROUTE", "FormulaFabric@2 reference execution has no routed Bank binding"
            )
        if self.bundle_id is None:
            if self.member_ids:
                raise FormulaSchemaError(
                    "FF2_INVALID_BANK_BUNDLE", "member_ids require a bundle_id"
                )
        else:
            _validate_name(self.bundle_id, field="BankBinding.bundle_id")
            if not self.member_ids or len(self.member_ids) != len(set(self.member_ids)):
                raise FormulaSchemaError(
                    "FF2_INVALID_BANK_BUNDLE", "bundled Bank operands require unique member_ids"
                )
            for member_id in self.member_ids:
                _validate_name(member_id, field="BankBinding.member_ids")
            expected_members = (
                self.value_type.size_for("K") if "K" in self.value_type.axis_names else 1
            )
            if expected_members is not None and len(self.member_ids) != expected_members:
                raise FormulaSchemaError(
                    "FF2_INVALID_BANK_BUNDLE",
                    "Bank bundle member_ids must match the declared member axis",
                )

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": "bank",
            "name": self.name,
            "source_ref": self.source_ref,
            "partition_id": self.partition_id,
            "value_type": self.value_type.to_dict(),
            "asset_fingerprint": self.asset_fingerprint,
            "route_ref": self.route_ref,
            "bundle_id": self.bundle_id,
            "member_ids": list(self.member_ids),
        }

    def bind(self, value: Tensor) -> FormulaBankOperand:
        """Bind one runtime tensor to this exact static Bank identity."""

        return FormulaBankOperand(
            value=value,
            source_ref=self.source_ref,
            partition_id=self.partition_id,
            asset_fingerprint=self.asset_fingerprint,
            route_ref=self.route_ref,
            bundle_id=self.bundle_id,
            member_ids=self.member_ids,
        )


@dataclass(frozen=True)
class FormulaBankOperand:
    """Zero-copy runtime value carrying the identity declared by BankBinding."""

    value: Tensor
    source_ref: str
    partition_id: str
    asset_fingerprint: str | None = None
    route_ref: str | None = None
    bundle_id: str | None = None
    member_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.value, Tensor):
            raise TypeError("FormulaBankOperand.value must be a Tensor")
        if not isinstance(self.source_ref, str) or not _COMPONENT_REF_RE.fullmatch(
            self.source_ref
        ):
            raise FormulaBindingError(
                "FF2_INVALID_BANK_REF", "source_ref must be a canonical component reference"
            )
        _validate_name(self.partition_id, field="FormulaBankOperand.partition_id")
        if isinstance(self.member_ids, (str, bytes)):
            raise FormulaBindingError(
                "FF2_INVALID_BANK_BUNDLE", "member_ids must be a sequence of names"
            )
        object.__setattr__(self, "member_ids", tuple(self.member_ids))
        for value, field in (
            (self.asset_fingerprint, "asset_fingerprint"),
            (self.route_ref, "route_ref"),
        ):
            if value is not None and (not isinstance(value, str) or not value):
                raise FormulaBindingError(
                    "FF2_INVALID_BANK_REF", f"{field} must be None or non-empty"
                )
        if self.asset_fingerprint is not None and (
            len(self.asset_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in self.asset_fingerprint)
        ):
            raise FormulaBindingError(
                "FF2_INVALID_BANK_REF", "asset_fingerprint must be a SHA-256 hex digest"
            )
        if self.bundle_id is None:
            if self.member_ids:
                raise FormulaBindingError(
                    "FF2_INVALID_BANK_BUNDLE", "member_ids require a bundle_id"
                )
        else:
            _validate_name(self.bundle_id, field="FormulaBankOperand.bundle_id")
            if not self.member_ids or len(self.member_ids) != len(set(self.member_ids)):
                raise FormulaBindingError(
                    "FF2_INVALID_BANK_BUNDLE", "runtime bundle member_ids must be unique"
                )
            for member_id in self.member_ids:
                _validate_name(member_id, field="FormulaBankOperand.member_ids")

    def consume(self, binding: BankBinding) -> Tensor:
        expected = (
            binding.source_ref,
            binding.partition_id,
            binding.asset_fingerprint,
            binding.route_ref,
            binding.bundle_id,
            binding.member_ids,
        )
        received = (
            self.source_ref,
            self.partition_id,
            self.asset_fingerprint,
            self.route_ref,
            self.bundle_id,
            self.member_ids,
        )
        if received != expected:
            raise FormulaBindingError(
                "FF2_BANK_IDENTITY_MISMATCH",
                f"runtime Bank identity for {binding.name!r} does not match its program binding",
            )
        return self.value


FormulaBinding = InputBinding | BankBinding


@dataclass(frozen=True)
class _FormulaExpr:
    value_type: TensorType
    atom_ref: str | None = None
    operands: tuple[_FormulaExpr | FormulaBinding, ...] = ()
    attributes: tuple[tuple[str, object], ...] = ()
    binding: FormulaBinding | None = None


FormulaOperand = _FormulaExpr | FormulaBinding


@dataclass(frozen=True)
class FormulaSlotSpec:
    slot_id: str
    value_type: TensorType
    producer: Literal["input", "bank", "instruction"]
    producer_id: str
    role: Literal["input", "bank_operand", "temporary", "output"]

    def __post_init__(self) -> None:
        if not isinstance(self.slot_id, str) or not self.slot_id:
            raise FormulaSchemaError("FF2_INVALID_SLOT", "slot_id must be non-empty")
        if not isinstance(self.value_type, TensorType):
            raise TypeError("FormulaSlotSpec.value_type must be TensorType")
        if self.producer not in {"input", "bank", "instruction"}:
            raise FormulaSchemaError("FF2_INVALID_SLOT", "slot producer is invalid")
        if not isinstance(self.producer_id, str) or not self.producer_id:
            raise FormulaSchemaError("FF2_INVALID_SLOT", "producer_id must be non-empty")
        allowed_roles = {
            "input": {"input"},
            "bank": {"bank_operand"},
            "instruction": {"temporary", "output"},
        }
        if self.role not in allowed_roles[self.producer]:
            raise FormulaSchemaError("FF2_INVALID_SLOT", "slot role does not match its producer")

    def to_dict(self) -> dict[str, object]:
        return {
            "slot_id": self.slot_id,
            "value_type": self.value_type.to_dict(),
            "producer": self.producer,
            "producer_id": self.producer_id,
            "role": self.role,
        }


@dataclass(frozen=True)
class FormulaInstructionV2:
    instruction_id: str
    step: int
    atom_ref: str
    input_slots: tuple[str, ...]
    output_slot: str
    attributes: tuple[tuple[str, object], ...]

    def __post_init__(self) -> None:
        if isinstance(self.input_slots, (str, bytes)):
            raise FormulaSchemaError(
                "FF2_INVALID_INSTRUCTION", "input_slots must be a sequence of names"
            )
        object.__setattr__(self, "input_slots", tuple(self.input_slots))
        raw_attributes = tuple(self.attributes)
        if any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            for item in raw_attributes
        ):
            raise FormulaSchemaError(
                "FF2_INVALID_INSTRUCTION", "attributes must contain string-key pairs"
            )
        object.__setattr__(
            self,
            "attributes",
            tuple(
                sorted(
                    ((key, _freeze_json(value)) for key, value in raw_attributes),
                    key=lambda item: item[0],
                )
            ),
        )
        if not isinstance(self.instruction_id, str) or not self.instruction_id:
            raise FormulaSchemaError("FF2_INVALID_INSTRUCTION", "instruction_id must be non-empty")
        if isinstance(self.step, bool) or not isinstance(self.step, int) or self.step <= 0:
            raise FormulaSchemaError("FF2_INVALID_STEP", "step must be a positive integer")
        if not isinstance(self.atom_ref, str) or not self.atom_ref:
            raise FormulaSchemaError("FF2_INVALID_ATOM", "atom_ref must be non-empty")
        if not self.input_slots or any(not isinstance(slot, str) or not slot for slot in self.input_slots):
            raise FormulaSchemaError("FF2_INVALID_INSTRUCTION", "input_slots must be non-empty names")
        if not isinstance(self.output_slot, str) or not self.output_slot:
            raise FormulaSchemaError("FF2_INVALID_INSTRUCTION", "output_slot must be non-empty")
        keys = [key for key, _value in self.attributes]
        if len(keys) != len(set(keys)) or any(not isinstance(key, str) for key in keys):
            raise FormulaSchemaError("FF2_INVALID_INSTRUCTION", "attribute keys must be unique strings")
        signature = _ATOM_SIGNATURES.get(self.atom_ref)
        if signature is None:
            raise FormulaProgramError("FF2_UNKNOWN_ATOM", f"unknown atom {self.atom_ref!r}")
        arity, attribute_keys = signature
        if len(self.input_slots) != arity or set(keys) != attribute_keys:
            raise FormulaProgramError(
                "FF2_ATOM_SIGNATURE",
                f"atom {self.atom_ref!r} requires arity {arity} and attributes {sorted(attribute_keys)}",
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "instruction_id": self.instruction_id,
            "step": self.step,
            "atom_ref": self.atom_ref,
            "input_slots": list(self.input_slots),
            "output_slot": self.output_slot,
            "attributes": {key: _thaw_json(value) for key, value in self.attributes},
        }


@dataclass(frozen=True)
class FormulaProgram:
    """A code-free typed SSA Formula program."""

    bindings: tuple[FormulaBinding, ...]
    slots: tuple[FormulaSlotSpec, ...]
    instructions: tuple[FormulaInstructionV2, ...]
    outputs: tuple[str, ...]
    limits: FormulaLimits = DEFAULT_FORMULA_LIMITS
    schema_version: int = FORMULA_PROGRAM_V2_SCHEMA_VERSION

    def __post_init__(self) -> None:
        sequence_fields = {
            "bindings": self.bindings,
            "slots": self.slots,
            "instructions": self.instructions,
            "outputs": self.outputs,
        }
        for field, value in sequence_fields.items():
            if isinstance(value, (str, bytes)):
                raise FormulaSchemaError(
                    "FF2_PROGRAM_SCHEMA",
                    f"FormulaProgram.{field} must be a sequence, not text",
                )
        object.__setattr__(self, "bindings", tuple(self.bindings))
        object.__setattr__(self, "slots", tuple(self.slots))
        object.__setattr__(self, "instructions", tuple(self.instructions))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        if not isinstance(self.limits, FormulaLimits):
            raise TypeError("FormulaProgram.limits must be FormulaLimits")
        if self.schema_version != FORMULA_PROGRAM_V2_SCHEMA_VERSION:
            raise FormulaSchemaError("FF2_UNSUPPORTED_SCHEMA", "unsupported FormulaProgram schema")
        binding_names = [binding.name for binding in self.bindings]
        slot_ids = [slot.slot_id for slot in self.slots]
        instruction_ids = [item.instruction_id for item in self.instructions]
        if len(binding_names) != len(set(binding_names)):
            raise FormulaProgramError("FF2_DUPLICATE_BINDING", "binding names must be unique")
        bundles: dict[str, tuple[str, str | None, tuple[str, ...]]] = {}
        for binding in self.bindings:
            if not isinstance(binding, BankBinding) or binding.bundle_id is None:
                continue
            identity = (binding.source_ref, binding.asset_fingerprint, binding.member_ids)
            previous = bundles.setdefault(binding.bundle_id, identity)
            if previous != identity:
                raise FormulaProgramError(
                    "FF2_BANK_BUNDLE_MISMATCH",
                    f"Bank bundle {binding.bundle_id!r} has inconsistent source or member order",
                )
        if len(slot_ids) != len(set(slot_ids)):
            raise FormulaProgramError("FF2_DUPLICATE_SLOT", "slot ids must be unique")
        if len(instruction_ids) != len(set(instruction_ids)):
            raise FormulaProgramError("FF2_DUPLICATE_INSTRUCTION", "instruction ids must be unique")
        if len(self.bindings) > self.limits.max_bindings:
            raise FormulaProgramError("FF2_LIMIT_EXCEEDED", "binding count exceeds Formula limits")
        if len(self.slots) > self.limits.max_slots:
            raise FormulaProgramError("FF2_LIMIT_EXCEEDED", "slot count exceeds Formula limits")
        if len(self.instructions) > self.limits.max_instructions:
            raise FormulaProgramError(
                "FF2_LIMIT_EXCEEDED", "instruction count exceeds Formula limits"
            )
        if self.instructions and max(item.step for item in self.instructions) > self.limits.max_steps:
            raise FormulaProgramError("FF2_LIMIT_EXCEEDED", "step count exceeds Formula limits")
        for slot in self.slots:
            if len(slot.value_type.axis_names) > self.limits.max_axes:
                raise FormulaProgramError("FF2_LIMIT_EXCEEDED", "axis count exceeds Formula limits")
            if any(
                size is not None and size > self.limits.max_axis_extent
                for size in slot.value_type.sizes
            ):
                raise FormulaProgramError("FF2_LIMIT_EXCEEDED", "axis extent exceeds Formula limits")
            if all(size is not None for size in slot.value_type.sizes) and math.prod(
                size for size in slot.value_type.sizes if size is not None
            ) > self.limits.max_tensor_elements:
                raise FormulaProgramError(
                    "FF2_LIMIT_EXCEEDED", "static tensor size exceeds Formula limits"
                )
        slot_set = set(slot_ids)
        if not self.outputs or any(output not in slot_set for output in self.outputs):
            raise FormulaProgramError("FF2_INVALID_OUTPUT", "outputs must name existing slots")
        if len(self.outputs) != len(set(self.outputs)):
            raise FormulaProgramError("FF2_INVALID_OUTPUT", "outputs must not contain duplicates")
        slot_by_id = {slot.slot_id: slot for slot in self.slots}
        for slot in self.slots:
            if slot.producer == "instruction":
                expected_role = "output" if slot.slot_id in self.outputs else "temporary"
                if slot.role != expected_role:
                    raise FormulaProgramError(
                        "FF2_INVALID_OUTPUT",
                        "instruction slot roles must exactly match program outputs",
                    )
        if any(slot_by_id[output].producer != "instruction" for output in self.outputs):
            raise FormulaProgramError(
                "FF2_INVALID_OUTPUT", "program outputs must be instruction-produced values"
            )
        binding_by_name = {binding.name: binding for binding in self.bindings}
        instruction_by_id = {item.instruction_id: item for item in self.instructions}
        for slot in self.slots:
            if slot.producer in {"input", "bank"}:
                binding = binding_by_name.get(slot.producer_id)
                expected_kind = "input" if isinstance(binding, InputBinding) else "bank"
                if binding is None or slot.slot_id != binding.name or slot.producer != expected_kind:
                    raise FormulaProgramError(
                        "FF2_SLOT_PRODUCER", f"slot {slot.slot_id!r} has an invalid binding producer"
                    )
            else:
                instruction = instruction_by_id.get(slot.producer_id)
                if instruction is None or instruction.output_slot != slot.slot_id:
                    raise FormulaProgramError(
                        "FF2_SLOT_PRODUCER", f"slot {slot.slot_id!r} has an invalid instruction producer"
                    )
        available = {slot.slot_id for slot in self.slots if slot.producer != "instruction"}
        previous_step = 0
        pending: list[FormulaInstructionV2] = []
        for instruction in self.instructions:
            if instruction.step <= 0 or instruction.step < previous_step:
                raise FormulaProgramError("FF2_INVALID_STEP", "instruction steps must be positive and sorted")
            if instruction.step != previous_step:
                for item in pending:
                    available.add(item.output_slot)
                pending = []
                previous_step = instruction.step
            if any(slot not in available for slot in instruction.input_slots):
                raise FormulaProgramError(
                    "FF2_SSA_VISIBILITY",
                    "an instruction may only read bindings or outputs from earlier steps",
                )
            output_slot = slot_by_id.get(instruction.output_slot)
            if output_slot is None:
                raise FormulaProgramError(
                    "FF2_INVALID_OUTPUT", f"instruction {instruction.instruction_id!r} output is absent"
                )
            if (
                output_slot.producer != "instruction"
                or output_slot.producer_id != instruction.instruction_id
            ):
                raise FormulaProgramError(
                    "FF2_SLOT_PRODUCER",
                    f"instruction {instruction.instruction_id!r} must write its own produced slot",
                )
            expected_type = _infer_instruction_output_type(
                instruction,
                tuple(slot_by_id[slot].value_type for slot in instruction.input_slots),
            )
            if output_slot.value_type != expected_type:
                raise FormulaProgramError(
                    "FF2_INSTRUCTION_TYPE",
                    f"instruction {instruction.instruction_id!r} output type is invalid",
                )
            pending.append(instruction)

    @classmethod
    def build(
        cls,
        *,
        outputs: Sequence[FormulaOperand],
        limits: FormulaLimits = DEFAULT_FORMULA_LIMITS,
    ) -> FormulaProgram:
        outputs_tuple = tuple(_as_expr(output) for output in outputs)
        if not outputs_tuple:
            raise FormulaProgramError("FF2_NO_OUTPUTS", "FormulaProgram requires outputs")

        bindings: dict[str, FormulaBinding] = {}
        expression_slots: dict[int, str] = {}
        expression_depth: dict[int, int] = {}
        slots: list[FormulaSlotSpec] = []
        instructions: list[FormulaInstructionV2] = []

        def visit(expr: _FormulaExpr) -> tuple[str, int]:
            key = id(expr)
            if key in expression_slots:
                return expression_slots[key], expression_depth[key]
            if expr.binding is not None:
                binding = expr.binding
                prior = bindings.get(binding.name)
                if prior is not None and prior != binding:
                    raise FormulaProgramError(
                        "FF2_BINDING_COLLISION",
                        f"binding {binding.name!r} has conflicting declarations",
                    )
                bindings[binding.name] = binding
                if prior is None:
                    producer = "input" if isinstance(binding, InputBinding) else "bank"
                    role = "input" if producer == "input" else "bank_operand"
                    slots.append(
                        FormulaSlotSpec(
                            binding.name,
                            binding.value_type,
                            producer,
                            binding.name,
                            role,
                        )
                    )
                expression_slots[key] = binding.name
                expression_depth[key] = 0
                return binding.name, 0

            operand_results = [visit(_as_expr(operand)) for operand in expr.operands]
            depth = 1 + max((item[1] for item in operand_results), default=0)
            instruction_id = f"i{len(instructions)}"
            slot_id = f"%{len(instructions)}"
            instructions.append(
                FormulaInstructionV2(
                    instruction_id,
                    depth,
                    expr.atom_ref or "",
                    tuple(item[0] for item in operand_results),
                    slot_id,
                    expr.attributes,
                )
            )
            slots.append(
                FormulaSlotSpec(
                    slot_id,
                    expr.value_type,
                    "instruction",
                    instruction_id,
                    "temporary",
                )
            )
            expression_slots[key] = slot_id
            expression_depth[key] = depth
            return slot_id, depth

        output_slots = tuple(visit(output)[0] for output in outputs_tuple)
        output_set = set(output_slots)
        slots = [
            replace(slot, role="output")
            if slot.producer == "instruction" and slot.slot_id in output_set
            else slot
            for slot in slots
        ]
        instructions.sort(key=lambda item: (item.step, int(item.instruction_id[1:])))
        return cls(
            tuple(bindings.values()),
            tuple(slots),
            tuple(instructions),
            output_slots,
            limits,
        )

    @property
    def slot_types(self) -> dict[str, TensorType]:
        return {slot.slot_id: slot.value_type for slot in self.slots}

    @property
    def input_names(self) -> tuple[str, ...]:
        return tuple(binding.name for binding in self.bindings if isinstance(binding, InputBinding))

    @property
    def bank_names(self) -> tuple[str, ...]:
        return tuple(binding.name for binding in self.bindings if isinstance(binding, BankBinding))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_ref": FORMULA_PROGRAM_V2_SCHEMA_REF,
            "schema_version": self.schema_version,
            "bindings": [binding.to_dict() for binding in self.bindings],
            "slots": [slot.to_dict() for slot in self.slots],
            "instructions": [instruction.to_dict() for instruction in self.instructions],
            "outputs": list(self.outputs),
            "limits": self.limits.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> FormulaProgram:
        required = {
            "schema_ref",
            "schema_version",
            "bindings",
            "slots",
            "instructions",
            "outputs",
            "limits",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise FormulaSchemaError(
                "FF2_PROGRAM_SCHEMA", "FormulaProgram payload contains missing or unknown fields"
            )
        if value["schema_version"] != FORMULA_PROGRAM_V2_SCHEMA_VERSION:
            raise FormulaSchemaError("FF2_UNSUPPORTED_SCHEMA", "unsupported FormulaProgram schema")
        if value["schema_ref"] != FORMULA_PROGRAM_V2_SCHEMA_REF:
            raise FormulaSchemaError(
                "FF2_UNSUPPORTED_SCHEMA", "FormulaProgram schema reference is invalid"
            )
        raw_bindings = _require_record_sequence(value["bindings"], path="$.bindings")
        bindings: list[FormulaBinding] = []
        for index, raw in enumerate(raw_bindings):
            kind = raw.get("kind")
            if kind == "input" and set(raw) == {"kind", "name", "value_type"}:
                bindings.append(
                    InputBinding(raw["name"], TensorType.from_dict(raw["value_type"]))
                )
            elif kind == "bank" and set(raw) == {
                "kind",
                "name",
                "source_ref",
                "partition_id",
                "value_type",
                "asset_fingerprint",
                "route_ref",
                "bundle_id",
                "member_ids",
            }:
                member_ids = _require_name_sequence(
                    raw["member_ids"], path=f"$.bindings[{index}].member_ids"
                )
                bindings.append(
                    BankBinding(
                        raw["name"],
                        raw["source_ref"],
                        raw["partition_id"],
                        TensorType.from_dict(raw["value_type"]),
                        raw["asset_fingerprint"],
                        raw["route_ref"],
                        raw["bundle_id"],
                        member_ids,
                    )
                )
            else:
                raise FormulaSchemaError(
                    "FF2_BINDING_SCHEMA", f"invalid binding payload at index {index}"
                )

        raw_slots = _require_record_sequence(value["slots"], path="$.slots")
        slots: list[FormulaSlotSpec] = []
        for index, raw in enumerate(raw_slots):
            if set(raw) != {"slot_id", "value_type", "producer", "producer_id", "role"}:
                raise FormulaSchemaError("FF2_SLOT_SCHEMA", f"invalid slot payload at index {index}")
            slots.append(
                FormulaSlotSpec(
                    raw["slot_id"],
                    TensorType.from_dict(raw["value_type"]),
                    raw["producer"],
                    raw["producer_id"],
                    raw["role"],
                )
            )

        raw_instructions = _require_record_sequence(value["instructions"], path="$.instructions")
        instructions: list[FormulaInstructionV2] = []
        for index, raw in enumerate(raw_instructions):
            if set(raw) != {
                "instruction_id",
                "step",
                "atom_ref",
                "input_slots",
                "output_slot",
                "attributes",
            } or not isinstance(raw["attributes"], Mapping):
                raise FormulaSchemaError(
                    "FF2_INSTRUCTION_SCHEMA", f"invalid instruction payload at index {index}"
                )
            raw_input_slots = raw["input_slots"]
            if not isinstance(raw_input_slots, Sequence) or isinstance(
                raw_input_slots, (str, bytes)
            ):
                raise FormulaSchemaError(
                    "FF2_INSTRUCTION_SCHEMA",
                    f"instruction input_slots must be a sequence at index {index}",
                )
            instructions.append(
                FormulaInstructionV2(
                    raw["instruction_id"],
                    raw["step"],
                    raw["atom_ref"],
                    tuple(raw_input_slots),
                    raw["output_slot"],
                    tuple(
                        (str(key), _freeze_json(item))
                        for key, item in sorted(raw["attributes"].items())
                    ),
                )
            )
        outputs = value["outputs"]
        if not isinstance(outputs, Sequence) or isinstance(outputs, (str, bytes)):
            raise FormulaSchemaError("FF2_PROGRAM_SCHEMA", "outputs must be a sequence")
        return cls(
            tuple(bindings),
            tuple(slots),
            tuple(instructions),
            tuple(outputs),
            FormulaLimits.from_dict(value["limits"]),
        )

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


def contract(
    left: FormulaOperand,
    right: FormulaOperand,
    *,
    reduce_axes: Sequence[tuple[str, str]],
    output_axes: Sequence[str] | None = None,
    accumulation_dtype: str = "float32",
) -> _FormulaExpr:
    left_expr = _as_expr(left)
    right_expr = _as_expr(right)
    pairs = tuple(tuple(pair) for pair in reduce_axes)
    if len(pairs) != 1 or any(len(pair) != 2 for pair in pairs):
        raise FormulaTypeError(
            "FF2_INVALID_CONTRACTION",
            "Contract@1 requires exactly one reduction axis pair",
        )
    if accumulation_dtype not in _ACCUMULATION_DTYPES:
        raise FormulaTypeError("FF2_INVALID_ACCUMULATION", "unsupported accumulation dtype")
    _require_compatible_domains(left_expr.value_type, right_expr.value_type)
    _require_compatible_dtypes(left_expr.value_type, right_expr.value_type)
    left_reduced = tuple(pair[0] for pair in pairs)
    right_reduced = tuple(pair[1] for pair in pairs)
    if len(set(left_reduced)) != len(left_reduced) or len(set(right_reduced)) != len(right_reduced):
        raise FormulaTypeError("FF2_INVALID_CONTRACTION", "Contract axes may be reduced once")
    for left_axis, right_axis in pairs:
        _require_axis_extent_equal(left_expr.value_type, left_axis, right_expr.value_type, right_axis)
        if left_axis != right_axis and (
            left_axis in right_expr.value_type.axis_names
            or right_axis in left_expr.value_type.axis_names
        ):
            raise FormulaTypeError(
                "FF2_AMBIGUOUS_AXIS_ALIAS",
                "differently named reduction axes may not alias preserved axes",
            )

    inferred: list[str] = [
        axis for axis in left_expr.value_type.axis_names if axis not in left_reduced
    ]
    for axis in right_expr.value_type.axis_names:
        if axis in right_reduced:
            continue
        if axis in inferred:
            _require_axis_extent_equal(left_expr.value_type, axis, right_expr.value_type, axis)
        else:
            inferred.append(axis)
    output = tuple(inferred if output_axes is None else output_axes)
    if len(output) != len(set(output)) or set(output) != set(inferred):
        raise FormulaTypeError(
            "FF2_OUTPUT_AXES", f"output_axes must be a permutation of {tuple(inferred)}"
        )
    sizes = []
    for axis in output:
        if axis in left_expr.value_type.axis_names and axis not in left_reduced:
            sizes.append(left_expr.value_type.size_for(axis))
        else:
            sizes.append(right_expr.value_type.size_for(axis))
    output_type = TensorType(
        output,
        tuple(sizes),
        dtype=left_expr.value_type.dtype,
        domain=left_expr.value_type.domain,
    )
    return _FormulaExpr(
        output_type,
        "arti/formula-atom-contract@1",
        (left_expr, right_expr),
        (
            ("reduce_axes", tuple((str(a), str(b)) for a, b in pairs)),
            ("output_axes", output),
            ("accumulation_dtype", accumulation_dtype),
        ),
    )


def dot(
    left: FormulaOperand,
    right: FormulaOperand,
    *,
    left_axis: str,
    right_axis: str,
    output_axes: Sequence[str] | None = None,
    accumulation_dtype: str = "float32",
) -> _FormulaExpr:
    """Canonical convenience alias that expands to Contract."""

    return contract(
        left,
        right,
        reduce_axes=((left_axis, right_axis),),
        output_axes=output_axes,
        accumulation_dtype=accumulation_dtype,
    )


def scale(
    value: FormulaOperand,
    factor: FormulaOperand,
    *,
    accumulation_dtype: str = "activation",
) -> _FormulaExpr:
    value_expr = _as_expr(value)
    factor_expr = _as_expr(factor)
    _require_compatible_domains(value_expr.value_type, factor_expr.value_type)
    _require_compatible_dtypes(value_expr.value_type, factor_expr.value_type)
    if accumulation_dtype not in _ACCUMULATION_DTYPES:
        raise FormulaTypeError("FF2_INVALID_ACCUMULATION", "unsupported accumulation dtype")
    for axis in factor_expr.value_type.axis_names:
        _require_axis_extent_equal(value_expr.value_type, axis, factor_expr.value_type, axis)
    return _FormulaExpr(
        value_expr.value_type,
        "arti/formula-atom-scale@1",
        (value_expr, factor_expr),
        (
            ("factor_axes", factor_expr.value_type.axis_names),
            ("accumulation_dtype", accumulation_dtype),
        ),
    )


def add(
    left: FormulaOperand,
    right: FormulaOperand,
    *,
    accumulation_dtype: str = "activation",
) -> _FormulaExpr:
    left_expr = _as_expr(left)
    right_expr = _as_expr(right)
    _require_exact_type(left_expr.value_type, right_expr.value_type)
    if accumulation_dtype not in _ACCUMULATION_DTYPES:
        raise FormulaTypeError("FF2_INVALID_ACCUMULATION", "unsupported accumulation dtype")
    return _FormulaExpr(
        left_expr.value_type,
        "arti/formula-atom-add@1",
        (left_expr, right_expr),
        (("accumulation_dtype", accumulation_dtype),),
    )


def reduce_sum(
    value: FormulaOperand, *, axis: str, accumulation_dtype: str = "float32"
) -> _FormulaExpr:
    value_expr = _as_expr(value)
    if axis not in value_expr.value_type.axis_names:
        raise FormulaTypeError("FF2_AXIS_MISMATCH", f"Reduce axis {axis!r} is absent")
    if accumulation_dtype not in _ACCUMULATION_DTYPES:
        raise FormulaTypeError("FF2_INVALID_ACCUMULATION", "unsupported accumulation dtype")
    output_axes = tuple(item for item in value_expr.value_type.axis_names if item != axis)
    return _FormulaExpr(
        value_expr.value_type.with_axes(output_axes),
        "arti/formula-atom-reduce@1",
        (value_expr,),
        (("axis", axis), ("mode", "sum"), ("accumulation_dtype", accumulation_dtype)),
    )


@dataclass(frozen=True)
class FormulaTraceV2:
    """Deterministic program diagnostics, not an attestation of Bank tensor contents."""

    program_fingerprint: str
    instruction_ids: tuple[str, ...]
    atom_refs: tuple[str, ...]
    output_slots: tuple[str, ...]
    schema_version: int = FORMULA_TRACE_V1_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if any(
            isinstance(value, (str, bytes))
            for value in (self.instruction_ids, self.atom_refs, self.output_slots)
        ):
            raise FormulaSchemaError(
                "FF2_TRACE_SCHEMA", "trace identifiers must be sequences, not text"
            )
        object.__setattr__(self, "instruction_ids", tuple(self.instruction_ids))
        object.__setattr__(self, "atom_refs", tuple(self.atom_refs))
        object.__setattr__(self, "output_slots", tuple(self.output_slots))
        if self.schema_version != FORMULA_TRACE_V1_SCHEMA_VERSION:
            raise FormulaSchemaError("FF2_UNSUPPORTED_TRACE", "unsupported FormulaTraceV2 schema")
        if (
            not isinstance(self.program_fingerprint, str)
            or len(self.program_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in self.program_fingerprint)
        ):
            raise FormulaSchemaError("FF2_TRACE_SCHEMA", "trace program fingerprint is invalid")
        if len(self.instruction_ids) != len(self.atom_refs):
            raise FormulaSchemaError("FF2_TRACE_SCHEMA", "trace instruction and atom counts differ")
        if any(not isinstance(item, str) or not item for item in (*self.instruction_ids, *self.atom_refs)):
            raise FormulaSchemaError("FF2_TRACE_SCHEMA", "trace identifiers must be non-empty")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_ref": FORMULA_TRACE_V1_SCHEMA_REF,
            "schema_version": self.schema_version,
            "program_fingerprint": self.program_fingerprint,
            "instruction_ids": list(self.instruction_ids),
            "atom_refs": list(self.atom_refs),
            "output_slots": list(self.output_slots),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> FormulaTraceV2:
        required = {
            "schema_ref",
            "schema_version",
            "program_fingerprint",
            "instruction_ids",
            "atom_refs",
            "output_slots",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise FormulaSchemaError("FF2_TRACE_SCHEMA", "trace has missing or unknown fields")
        if value["schema_ref"] != FORMULA_TRACE_V1_SCHEMA_REF:
            raise FormulaSchemaError("FF2_TRACE_SCHEMA", "trace schema reference is invalid")
        sequences = (value["instruction_ids"], value["atom_refs"], value["output_slots"])
        if any(
            not isinstance(item, Sequence) or isinstance(item, (str, bytes))
            for item in sequences
        ):
            raise FormulaSchemaError("FF2_TRACE_SCHEMA", "trace lists must be sequences")
        return cls(
            value["program_fingerprint"],
            tuple(value["instruction_ids"]),
            tuple(value["atom_refs"]),
            tuple(value["output_slots"]),
            schema_version=value["schema_version"],
        )

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    def verify(self, program: FormulaProgram) -> None:
        if not isinstance(program, FormulaProgram):
            raise TypeError("program must be FormulaProgram")
        expected_instruction_ids = tuple(
            instruction.instruction_id for instruction in program.instructions
        )
        expected_atom_refs = tuple(instruction.atom_ref for instruction in program.instructions)
        if (
            self.program_fingerprint != program.fingerprint
            or self.instruction_ids != expected_instruction_ids
            or self.atom_refs != expected_atom_refs
            or self.output_slots != program.outputs
        ):
            raise FormulaSchemaError(
                "FF2_TRACE_PROGRAM_MISMATCH",
                "FormulaTraceV2 does not match the supplied FormulaProgram",
            )


@dataclass(frozen=True)
class FormulaFabricV2Result:
    names: tuple[str, ...]
    values: tuple[Tensor, ...]
    trace: FormulaTraceV2 | None = None

    def output(self, name: str) -> Tensor:
        try:
            return self.values[self.names.index(name)]
        except ValueError as exc:
            raise KeyError(name) from exc


class ContractAtom(nn.Module):
    _component_reference: ClassVar[str] = "arti/formula-atom-contract@1"

    def __init__(
        self,
        left_type: TensorType,
        right_type: TensorType,
        *,
        reduce_axes: Sequence[tuple[str, str]],
        output_axes: Sequence[str],
        accumulation_dtype: str = "float32",
    ) -> None:
        super().__init__()
        probe_left = InputBinding("left", left_type)
        probe_right = InputBinding("right", right_type)
        expression = contract(
            probe_left,
            probe_right,
            reduce_axes=reduce_axes,
            output_axes=output_axes,
            accumulation_dtype=accumulation_dtype,
        )
        self.left_type = left_type
        self.right_type = right_type
        self.output_type = expression.value_type
        self.reduce_axes = tuple(tuple(pair) for pair in reduce_axes)
        self.output_axes = tuple(output_axes)
        self.accumulation_dtype = accumulation_dtype

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        _validate_atom_operands((left, right), (self.left_type, self.right_type))
        result = _named_contract(
            left,
            right,
            self.left_type,
            self.right_type,
            self.reduce_axes,
            self.output_axes,
            self.accumulation_dtype,
        )
        _validate_tensor_against_type(result, self.output_type, name="output")
        return result


class ScaleAtom(nn.Module):
    _component_reference: ClassVar[str] = "arti/formula-atom-scale@1"

    def __init__(
        self,
        value_type: TensorType,
        factor_type: TensorType,
        *,
        accumulation_dtype: str = "activation",
    ) -> None:
        super().__init__()
        scale(
            InputBinding("value", value_type),
            InputBinding("factor", factor_type),
            accumulation_dtype=accumulation_dtype,
        )
        self.value_type = value_type
        self.factor_type = factor_type
        self.accumulation_dtype = accumulation_dtype

    def forward(self, value: Tensor, factor: Tensor) -> Tensor:
        _validate_atom_operands((value, factor), (self.value_type, self.factor_type))
        result = _binary_with_accumulation(
            value,
            _align_tensor(factor, self.factor_type.axis_names, self.value_type.axis_names),
            operation="multiply",
            accumulation_dtype=self.accumulation_dtype,
        )
        _validate_tensor_against_type(result, self.value_type, name="output")
        return result


class AddAtom(nn.Module):
    _component_reference: ClassVar[str] = "arti/formula-atom-add@1"

    def __init__(
        self,
        value_type: TensorType,
        *,
        accumulation_dtype: str = "activation",
    ) -> None:
        super().__init__()
        add(
            InputBinding("left", value_type),
            InputBinding("right", value_type),
            accumulation_dtype=accumulation_dtype,
        )
        self.value_type = value_type
        self.accumulation_dtype = accumulation_dtype

    def forward(self, left: Tensor, right: Tensor) -> Tensor:
        _validate_atom_operands((left, right), (self.value_type, self.value_type))
        result = _binary_with_accumulation(
            left,
            right,
            operation="add",
            accumulation_dtype=self.accumulation_dtype,
        )
        _validate_tensor_against_type(result, self.value_type, name="output")
        return result


class ReduceAtom(nn.Module):
    _component_reference: ClassVar[str] = "arti/formula-atom-reduce@1"

    def __init__(
        self,
        value_type: TensorType,
        *,
        axis: str,
        accumulation_dtype: str = "float32",
    ) -> None:
        super().__init__()
        expression = reduce_sum(
            InputBinding("value", value_type),
            axis=axis,
            accumulation_dtype=accumulation_dtype,
        )
        self.value_type = value_type
        self.output_type = expression.value_type
        self.axis = axis
        self.accumulation_dtype = accumulation_dtype

    def forward(self, value: Tensor) -> Tensor:
        _validate_atom_operands((value,), (self.value_type,))
        result = _ordered_sum(
            value,
            self.value_type.axis_names.index(self.axis),
            self.accumulation_dtype,
        )
        _validate_tensor_against_type(result, self.output_type, name="output")
        return result


class PreparedFormulaBindings(NamedTuple):
    """Host-admitted positional bindings for one exact Formula program."""

    program_fingerprint: str
    binding_names: tuple[str, ...]
    values: tuple[Tensor, ...]

    def verify(
        self,
        program_fingerprint: str,
        binding_names: tuple[str, ...],
    ) -> None:
        if self.program_fingerprint != program_fingerprint:
            raise FormulaBindingError(
                "FF2_PREPARED_PROGRAM_MISMATCH",
                "prepared Formula bindings target a different program",
            )
        if self.binding_names != binding_names or len(self.values) != len(binding_names):
            raise FormulaBindingError(
                "FF2_PREPARED_BINDING_MISMATCH",
                "prepared Formula binding order does not match the program",
            )


class FormulaExecutionPlanV2(nn.Module):
    """Static lowering that only consumes host-admitted Formula bindings."""

    _component_reference: ClassVar[str] = "arti/formula-execution-plan@1"

    def __init__(self, program: FormulaProgram) -> None:
        super().__init__()
        if not isinstance(program, FormulaProgram):
            raise TypeError("FormulaExecutionPlanV2 requires FormulaProgram")
        self.program = program
        self.program_fingerprint = program.fingerprint
        slot_indices = {
            binding.name: index for index, binding in enumerate(program.bindings)
        }
        slot_types = program.slot_types
        operations = []
        for instruction in program.instructions:
            output_index = len(slot_indices)
            slot_indices[instruction.output_slot] = output_index
            operations.append(
                (
                    instruction,
                    tuple(slot_indices[name] for name in instruction.input_slots),
                    tuple(slot_types[name] for name in instruction.input_slots),
                )
            )
        self.binding_names = tuple(binding.name for binding in program.bindings)
        self._operations = tuple(operations)
        self._output_indices = tuple(slot_indices[name] for name in program.outputs)

    def forward(self, prepared: PreparedFormulaBindings) -> tuple[Tensor, ...]:
        if not isinstance(prepared, PreparedFormulaBindings):
            raise TypeError("FormulaExecutionPlanV2 requires PreparedFormulaBindings")
        prepared.verify(self.program_fingerprint, self.binding_names)
        bindings = prepared.values
        slots = list(bindings)
        for instruction, input_indices, input_types in self._operations:
            operands = tuple(slots[index] for index in input_indices)
            slots.append(_execute_instruction(instruction, operands, input_types))
        return tuple(slots[index] for index in self._output_indices)


class FormulaFabricV2(nn.Module):
    """Execute a typed FormulaProgram with no hidden trainable parameters."""

    _component_reference: ClassVar[str] = "arti/formula-fabric@2"

    def __init__(self, program: FormulaProgram) -> None:
        super().__init__()
        if not isinstance(program, FormulaProgram):
            raise TypeError("FormulaFabricV2 requires FormulaProgram")
        self.program = program

    def execution_plan(self) -> FormulaExecutionPlanV2:
        return FormulaExecutionPlanV2(self.program)

    def bind_tensors(
        self,
        *,
        inputs: Mapping[str, Tensor],
        banks: Mapping[str, FormulaBankOperand],
    ) -> PreparedFormulaBindings:
        slots, axis_extents = _bind_runtime_values(self.program, inputs=inputs, banks=banks)
        _preflight_program_shapes(self.program, slots, axis_extents)
        names = tuple(binding.name for binding in self.program.bindings)
        return PreparedFormulaBindings(
            self.program.fingerprint,
            names,
            tuple(slots[name] for name in names),
        )

    def forward(
        self,
        *,
        inputs: Mapping[str, Tensor],
        banks: Mapping[str, FormulaBankOperand],
        return_trace: bool = False,
    ) -> FormulaFabricV2Result:
        slots, axis_extents = _bind_runtime_values(self.program, inputs=inputs, banks=banks)
        _preflight_program_shapes(self.program, slots, axis_extents)

        slot_types = self.program.slot_types
        grouped: dict[int, list[FormulaInstructionV2]] = {}
        for instruction in self.program.instructions:
            grouped.setdefault(instruction.step, []).append(instruction)
        for step in sorted(grouped):
            snapshot = dict(slots)
            candidates: dict[str, Tensor] = {}
            for instruction in grouped[step]:
                operands = tuple(snapshot[slot] for slot in instruction.input_slots)
                input_types = tuple(slot_types[slot] for slot in instruction.input_slots)
                output_type = slot_types[instruction.output_slot]
                if len({operand.dtype for operand in operands}) != 1:
                    raise FormulaBindingError(
                        "FF2_RUNTIME_DTYPE_MISMATCH",
                        f"instruction {instruction.instruction_id!r} operands have different dtypes",
                    )
                _preflight_output_allocation(
                    output_type,
                    axis_extents,
                    operands[0].element_size(),
                    self.program.limits,
                    name=instruction.output_slot,
                )
                candidate = _execute_instruction(instruction, operands, input_types)
                _validate_tensor_against_type(
                    candidate,
                    output_type,
                    name=instruction.output_slot,
                )
                _validate_tensor_limits(
                    candidate,
                    self.program.limits,
                    name=instruction.output_slot,
                )
                _bind_axis_extents(
                    axis_extents,
                    candidate,
                    output_type,
                    name=instruction.output_slot,
                )
                candidates[instruction.output_slot] = candidate
            slots.update(candidates)

        names = tuple(self.program.outputs)
        trace = None
        if return_trace:
            trace = FormulaTraceV2(
                self.program.fingerprint,
                tuple(item.instruction_id for item in self.program.instructions),
                tuple(item.atom_ref for item in self.program.instructions),
                names,
            )
        return FormulaFabricV2Result(names, tuple(slots[name] for name in names), trace)


def build_lora_program(
    *,
    input_dim: int,
    output_dim: int,
    rank: int,
    source_ref: str,
    asset_fingerprint: str | None = None,
    member_count: int | None = None,
    bundle_id: str = "lora",
    member_ids: Sequence[str] | None = None,
    dtype: str = "floating",
    domain: str = "anonymous",
    contract_accumulation_dtype: str = "float32",
    pointwise_accumulation_dtype: str = "activation",
    limits: FormulaLimits = DEFAULT_FORMULA_LIMITS,
) -> FormulaProgram:
    """Expand LoRA apply into atoms; ``rank`` is the rank of each Bank member.

    With ``member_count=K`` the deterministic sum has a total rank upper bound
    of ``K * rank``. No rank normalization is implicit; ``lora.gain`` carries
    the complete caller-selected scale.
    """

    for value, name in ((input_dim, "input_dim"), (output_dim, "output_dim"), (rank, "rank")):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if member_count is not None and (
        isinstance(member_count, bool) or not isinstance(member_count, int) or member_count <= 0
    ):
        raise ValueError("member_count must be None or a positive integer")
    expected_members = 1 if member_count is None else member_count
    if member_ids is None:
        resolved_member_ids = tuple(f"member-{index:03d}" for index in range(expected_members))
    else:
        resolved_member_ids = _require_name_sequence(member_ids, path="member_ids")
        if len(resolved_member_ids) != expected_members:
            raise ValueError("member_ids must match member_count")

    x = InputBinding(
        "x",
        TensorType.axes(("B", "S", "Din"), sizes=(None, None, input_dim), dtype=dtype, domain=domain),
    )
    base = InputBinding(
        "base",
        TensorType.axes(("B", "S", "Dout"), sizes=(None, None, output_dim), dtype=dtype, domain=domain),
    )
    if member_count is None:
        a_type = TensorType.axes(("R", "Din"), sizes=(rank, input_dim), dtype=dtype, domain=domain)
        b_type = TensorType.axes(("Dout", "R"), sizes=(output_dim, rank), dtype=dtype, domain=domain)
        gain_type = TensorType.scalar(dtype=dtype, domain=domain)
    else:
        a_type = TensorType.axes(
            ("K", "R", "Din"), sizes=(member_count, rank, input_dim), dtype=dtype, domain=domain
        )
        b_type = TensorType.axes(
            ("K", "Dout", "R"), sizes=(member_count, output_dim, rank), dtype=dtype, domain=domain
        )
        gain_type = TensorType.axes(("K",), sizes=(member_count,), dtype=dtype, domain=domain)
    a = BankBinding(
        "lora.A",
        source_ref,
        "A",
        a_type,
        asset_fingerprint,
        bundle_id=bundle_id,
        member_ids=resolved_member_ids,
    )
    b = BankBinding(
        "lora.B",
        source_ref,
        "B",
        b_type,
        asset_fingerprint,
        bundle_id=bundle_id,
        member_ids=resolved_member_ids,
    )
    gain = InputBinding("lora.gain", gain_type)

    first_axes = ("B", "S", "R") if member_count is None else ("B", "S", "K", "R")
    delta_axes = ("B", "S", "Dout") if member_count is None else ("B", "S", "K", "Dout")
    hidden = contract(
        x,
        a,
        reduce_axes=(("Din", "Din"),),
        output_axes=first_axes,
        accumulation_dtype=contract_accumulation_dtype,
    )
    delta = contract(
        hidden,
        b,
        reduce_axes=(("R", "R"),),
        output_axes=delta_axes,
        accumulation_dtype=contract_accumulation_dtype,
    )
    scaled = scale(delta, gain, accumulation_dtype=pointwise_accumulation_dtype)
    if member_count is not None:
        scaled = reduce_sum(
            scaled,
            axis="K",
            accumulation_dtype=contract_accumulation_dtype,
        )
    return FormulaProgram.build(
        outputs=(
            add(base, scaled, accumulation_dtype=pointwise_accumulation_dtype),
        ),
        limits=limits,
    )


def _infer_instruction_output_type(
    instruction: FormulaInstructionV2, operand_types: tuple[TensorType, ...]
) -> TensorType:
    attributes = dict(instruction.attributes)
    if instruction.atom_ref == "arti/formula-atom-contract@1" and len(operand_types) == 2:
        try:
            reduce_axes = tuple(tuple(pair) for pair in attributes["reduce_axes"])
            output_axes = tuple(attributes["output_axes"])
        except (TypeError, KeyError) as exc:
            raise FormulaProgramError(
                "FF2_ATOM_ATTRIBUTES", "Contract attributes are invalid"
            ) from exc
        return contract(
            InputBinding("left", operand_types[0]),
            InputBinding("right", operand_types[1]),
            reduce_axes=reduce_axes,
            output_axes=output_axes,
            accumulation_dtype=attributes["accumulation_dtype"],
        ).value_type
    if instruction.atom_ref == "arti/formula-atom-scale@1" and len(operand_types) == 2:
        try:
            factor_axes = tuple(attributes["factor_axes"])
        except (TypeError, KeyError) as exc:
            raise FormulaProgramError(
                "FF2_ATOM_ATTRIBUTES", "Scale factor_axes are invalid"
            ) from exc
        if factor_axes != operand_types[1].axis_names:
            raise FormulaProgramError(
                "FF2_ATOM_ATTRIBUTES",
                "Scale factor_axes must exactly match the factor operand type",
            )
        return scale(
            InputBinding("value", operand_types[0]),
            InputBinding("factor", operand_types[1]),
            accumulation_dtype=attributes["accumulation_dtype"],
        ).value_type
    if instruction.atom_ref == "arti/formula-atom-add@1" and len(operand_types) == 2:
        return add(
            InputBinding("left", operand_types[0]),
            InputBinding("right", operand_types[1]),
            accumulation_dtype=attributes["accumulation_dtype"],
        ).value_type
    if instruction.atom_ref == "arti/formula-atom-reduce@1" and len(operand_types) == 1:
        if attributes["mode"] != "sum":
            raise FormulaProgramError(
                "FF2_ATOM_ATTRIBUTES", "Reduce@1 only supports mode='sum'"
            )
        return reduce_sum(
            InputBinding("value", operand_types[0]),
            axis=attributes["axis"],
            accumulation_dtype=attributes["accumulation_dtype"],
        ).value_type
    raise FormulaProgramError(
        "FF2_UNKNOWN_ATOM",
        f"atom {instruction.atom_ref!r} has an unknown signature or invalid arity",
    )


def _execute_instruction(
    instruction: FormulaInstructionV2,
    operands: tuple[Tensor, ...],
    operand_types: tuple[TensorType, ...],
) -> Tensor:
    attributes = dict(instruction.attributes)
    if instruction.atom_ref == "arti/formula-atom-contract@1":
        return _named_contract(
            operands[0],
            operands[1],
            operand_types[0],
            operand_types[1],
            tuple(tuple(pair) for pair in attributes["reduce_axes"]),
            tuple(attributes["output_axes"]),
            attributes["accumulation_dtype"],
        )
    if instruction.atom_ref == "arti/formula-atom-scale@1":
        return _binary_with_accumulation(
            operands[0],
            _align_tensor(
                operands[1], operand_types[1].axis_names, operand_types[0].axis_names
            ),
            operation="multiply",
            accumulation_dtype=attributes["accumulation_dtype"],
        )
    if instruction.atom_ref == "arti/formula-atom-add@1":
        return _binary_with_accumulation(
            operands[0],
            operands[1],
            operation="add",
            accumulation_dtype=attributes["accumulation_dtype"],
        )
    if instruction.atom_ref == "arti/formula-atom-reduce@1":
        axis = operand_types[0].axis_names.index(attributes["axis"])
        return _ordered_sum(operands[0], axis, attributes["accumulation_dtype"])
    raise FormulaProgramError(
        "FF2_UNKNOWN_ATOM", f"unsupported atom reference {instruction.atom_ref!r}"
    )


def _named_contract(
    left: Tensor,
    right: Tensor,
    left_type: TensorType,
    right_type: TensorType,
    reduce_axes: tuple[tuple[str, str], ...],
    output_axes: tuple[str, ...],
    accumulation_dtype: str,
) -> Tensor:
    for left_axis, right_axis in reduce_axes:
        left_extent = left.shape[left_type.axis_names.index(left_axis)]
        right_extent = right.shape[right_type.axis_names.index(right_axis)]
        if left_extent != right_extent:
            raise FormulaBindingError(
                "FF2_AXIS_MISMATCH",
                f"Contract reduction extents differ: {left_axis}={left_extent}, "
                f"{right_axis}={right_extent}",
            )
    labels = []
    for axis in (*left_type.axis_names, *right_type.axis_names, *output_axes):
        if axis not in labels:
            labels.append(axis)
    if len(labels) > 52:
        raise FormulaTypeError("FF2_TOO_MANY_AXES", "Contract supports at most 52 named axes")
    characters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
    label_map = {axis: characters[index] for index, axis in enumerate(labels)}
    right_alias = dict(label_map)
    for left_axis, right_axis in reduce_axes:
        right_alias[right_axis] = label_map[left_axis]
    equation = (
        "".join(label_map[axis] for axis in left_type.axis_names)
        + ","
        + "".join(right_alias[axis] for axis in right_type.axis_names)
        + "->"
        + "".join(label_map[axis] for axis in output_axes)
    )
    original_dtype = left.dtype
    compute_dtype = _accumulation_dtype(original_dtype, accumulation_dtype)
    result = torch.einsum(equation, left.to(compute_dtype), right.to(compute_dtype))
    return result.to(original_dtype)


def _ordered_sum(value: Tensor, axis: int, accumulation_dtype: str) -> Tensor:
    if value.shape[axis] == 0:
        raise FormulaTypeError("FF2_EMPTY_REDUCTION", "Reduce does not accept an empty axis")
    original_dtype = value.dtype
    compute_dtype = _accumulation_dtype(original_dtype, accumulation_dtype)
    terms = value.to(compute_dtype).unbind(axis)
    result = terms[0]
    for term in terms[1:]:
        result = result + term
    return result.to(original_dtype)


def _align_tensor(value: Tensor, source_axes: tuple[str, ...], target_axes: tuple[str, ...]) -> Tensor:
    if not source_axes:
        return value.reshape((1,) * len(target_axes))
    positions = [target_axes.index(axis) for axis in source_axes]
    permutation = sorted(range(len(source_axes)), key=lambda index: positions[index])
    if permutation != list(range(len(source_axes))):
        value = value.permute(permutation)
    ordered_axes = tuple(source_axes[index] for index in permutation)
    shape = [1] * len(target_axes)
    for index, axis in enumerate(ordered_axes):
        shape[target_axes.index(axis)] = value.shape[index]
    return value.reshape(shape)


def _as_expr(value: FormulaOperand) -> _FormulaExpr:
    if isinstance(value, _FormulaExpr):
        return value
    if isinstance(value, (InputBinding, BankBinding)):
        return _FormulaExpr(value.value_type, binding=value)
    raise TypeError("Formula operands must be bindings or Formula expressions")


def _validate_name(value: str, *, field: str) -> None:
    if not isinstance(value, str) or not _NAME_RE.fullmatch(value):
        raise FormulaSchemaError("FF2_INVALID_NAME", f"{field} is invalid")


def _validate_component_ref(value: str, *, field: str) -> None:
    if not isinstance(value, str) or not _COMPONENT_REF_RE.fullmatch(value):
        raise FormulaSchemaError(
            "FF2_INVALID_BANK_REF", f"{field} must be a canonical component reference"
        )


def _require_compatible_domains(left: TensorType, right: TensorType) -> None:
    if left.domain != right.domain:
        raise FormulaTypeError("FF2_DOMAIN_MISMATCH", "Formula operands have different domains")


def _require_compatible_dtypes(left: TensorType, right: TensorType) -> None:
    if left.dtype != right.dtype:
        raise FormulaTypeError("FF2_DTYPE_MISMATCH", "Formula operands have different dtypes")


def _require_exact_type(left: TensorType, right: TensorType) -> None:
    _require_compatible_domains(left, right)
    _require_compatible_dtypes(left, right)
    if left.axis_names != right.axis_names or left.sizes != right.sizes:
        raise FormulaTypeError("FF2_SLOT_TYPE_MISMATCH", "Add operands must have identical types")


def _require_axis_extent_equal(
    left: TensorType, left_axis: str, right: TensorType, right_axis: str
) -> None:
    left_size = left.size_for(left_axis)
    right_size = right.size_for(right_axis)
    if left_size is not None and right_size is not None and left_size != right_size:
        raise FormulaTypeError(
            "FF2_AXIS_MISMATCH",
            f"axis extents differ: {left_axis}={left_size}, {right_axis}={right_size}",
        )


def _validate_tensor_against_type(value: Tensor, value_type: TensorType, *, name: str) -> None:
    if not isinstance(value, Tensor):
        raise FormulaBindingError("FF2_BINDING_NOT_TENSOR", f"{name!r} must be a Tensor")
    if value.ndim != len(value_type.axis_names):
        raise FormulaBindingError(
            "FF2_BINDING_RANK", f"{name!r} rank does not match {value_type.axis_names}"
        )
    for size, expected, axis in zip(value.shape, value_type.sizes, value_type.axis_names):
        if size <= 0:
            raise FormulaBindingError("FF2_EMPTY_AXIS", f"{name!r} axis {axis!r} is empty")
        if expected is not None and size != expected:
            raise FormulaBindingError(
                "FF2_BINDING_SHAPE",
                f"{name!r} axis {axis!r} expected {expected}, received {size}",
            )
    if value_type.dtype == "floating":
        valid_dtype = value.is_floating_point()
    else:
        valid_dtype = str(value.dtype).removeprefix("torch.") == value_type.dtype
    if not valid_dtype:
        raise FormulaBindingError(
            "FF2_BINDING_DTYPE", f"{name!r} does not satisfy dtype {value_type.dtype!r}"
        )
    if not bool(torch.isfinite(value).all()):
        raise FormulaBindingError("FF2_NONFINITE", f"{name!r} must contain finite values")


def _bind_runtime_values(
    program: FormulaProgram,
    *,
    inputs: Mapping[str, Tensor],
    banks: Mapping[str, FormulaBankOperand],
) -> tuple[dict[str, Tensor], dict[str, int]]:
    if not isinstance(inputs, Mapping) or not isinstance(banks, Mapping):
        raise TypeError("inputs and banks must be mappings")
    _require_exact_keys(inputs, program.input_names, source="inputs")
    _require_exact_keys(banks, program.bank_names, source="banks")
    slots: dict[str, Tensor] = {}
    axis_extents: dict[str, int] = {}
    execution_device: torch.device | None = None
    for binding in program.bindings:
        source = inputs if isinstance(binding, InputBinding) else banks
        bound_value = source[binding.name]
        if isinstance(binding, BankBinding):
            if not isinstance(bound_value, FormulaBankOperand):
                raise FormulaBindingError(
                    "FF2_UNBOUND_BANK",
                    f"Bank {binding.name!r} requires FormulaBankOperand identity binding",
                )
            value = bound_value.consume(binding)
        else:
            value = bound_value
        _validate_tensor_against_type(value, binding.value_type, name=binding.name)
        _validate_tensor_limits(value, program.limits, name=binding.name)
        _bind_axis_extents(axis_extents, value, binding.value_type, name=binding.name)
        if execution_device is None:
            execution_device = value.device
        elif value.device != execution_device:
            raise FormulaBindingError(
                "FF2_DEVICE_MISMATCH", "all Formula bindings must use the same device"
            )
        slots[binding.name] = value
    return slots, axis_extents


def _preflight_program_shapes(
    program: FormulaProgram,
    binding_slots: Mapping[str, Tensor],
    axis_extents: Mapping[str, int],
) -> None:
    slot_shapes = {name: tuple(value.shape) for name, value in binding_slots.items()}
    slot_element_sizes = {
        name: value.element_size() for name, value in binding_slots.items()
    }
    slot_dtypes = {name: value.dtype for name, value in binding_slots.items()}
    persistent_bytes = sum(
        value.numel() * value.element_size() for value in binding_slots.values()
    )
    _validate_working_bytes(
        persistent_bytes,
        program.limits,
        name="Formula bindings",
    )
    slot_types = program.slot_types
    for instruction in program.instructions:
        input_shapes = tuple(slot_shapes[name] for name in instruction.input_slots)
        input_types = tuple(slot_types[name] for name in instruction.input_slots)
        input_dtypes = tuple(slot_dtypes[name] for name in instruction.input_slots)
        if len(set(input_dtypes)) != 1:
            raise FormulaBindingError(
                "FF2_RUNTIME_DTYPE_MISMATCH",
                f"instruction {instruction.instruction_id!r} operands have different dtypes",
            )
        if instruction.atom_ref == "arti/formula-atom-contract@1":
            reduce_axes = tuple(tuple(pair) for pair in dict(instruction.attributes)["reduce_axes"])
            for left_axis, right_axis in reduce_axes:
                left_extent = input_shapes[0][input_types[0].axis_names.index(left_axis)]
                right_extent = input_shapes[1][input_types[1].axis_names.index(right_axis)]
                if left_extent != right_extent:
                    raise FormulaBindingError(
                        "FF2_AXIS_MISMATCH",
                        f"Contract reduction extents differ: {left_axis}={left_extent}, "
                        f"{right_axis}={right_extent}",
                    )
        output_type = slot_types[instruction.output_slot]
        output_shape = _resolved_type_shape(
            output_type,
            axis_extents,
            name=instruction.output_slot,
        )
        attributes = dict(instruction.attributes)
        compute_dtype = _accumulation_dtype(
            input_dtypes[0],
            attributes["accumulation_dtype"],
        )
        compute_element_size = torch.empty((), dtype=compute_dtype).element_size()
        for input_slot, input_shape in zip(instruction.input_slots, input_shapes):
            _preflight_shape_bytes(
                input_shape,
                compute_element_size,
                program.limits,
                name=f"{instruction.instruction_id}:{input_slot}:working",
            )
        _preflight_output_allocation(
            output_type,
            axis_extents,
            compute_element_size,
            program.limits,
            name=f"{instruction.output_slot}:working",
        )
        output_elements = math.prod(output_shape)
        output_compute_bytes = output_elements * compute_element_size
        output_storage_bytes = output_elements * slot_element_sizes[
            instruction.input_slots[0]
        ]
        working_bytes = persistent_bytes + sum(
            math.prod(shape) * compute_element_size for shape in input_shapes
        )
        # Runtime accumulation can hold both the compute-dtype result and its
        # activation-dtype cast until the instruction returns.
        working_bytes += output_compute_bytes + output_storage_bytes
        if instruction.atom_ref == "arti/formula-atom-reduce@1":
            # Ordered reduction can briefly retain the previous accumulator
            # while materializing the next one.
            working_bytes += output_compute_bytes
        _validate_working_bytes(
            working_bytes,
            program.limits,
            name=f"instruction {instruction.instruction_id!r}",
        )
        slot_shapes[instruction.output_slot] = output_shape
        slot_element_sizes[instruction.output_slot] = slot_element_sizes[
            instruction.input_slots[0]
        ]
        slot_dtypes[instruction.output_slot] = slot_dtypes[instruction.input_slots[0]]
        persistent_bytes += output_elements * slot_element_sizes[instruction.output_slot]
        _validate_working_bytes(
            persistent_bytes,
            program.limits,
            name=f"slot {instruction.output_slot!r}",
        )


def _validate_tensor_limits(value: Tensor, limits: FormulaLimits, *, name: str) -> None:
    if value.numel() > limits.max_tensor_elements:
        raise FormulaBindingError(
            "FF2_LIMIT_EXCEEDED", f"{name!r} exceeds the Formula tensor element limit"
        )
    if any(size > limits.max_axis_extent for size in value.shape):
        raise FormulaBindingError(
            "FF2_LIMIT_EXCEEDED", f"{name!r} exceeds the Formula axis extent limit"
        )
    if value.numel() * value.element_size() > limits.max_tensor_bytes:
        raise FormulaBindingError(
            "FF2_LIMIT_EXCEEDED", f"{name!r} exceeds the Formula tensor byte limit"
        )


def _validate_working_bytes(value: int, limits: FormulaLimits, *, name: str) -> None:
    if value > limits.max_working_bytes:
        raise FormulaBindingError(
            "FF2_LIMIT_EXCEEDED",
            f"{name} exceeds the Formula aggregate working byte limit",
        )


def _preflight_output_allocation(
    value_type: TensorType,
    axis_extents: Mapping[str, int],
    element_size: int,
    limits: FormulaLimits,
    *,
    name: str,
) -> None:
    shape = _resolved_type_shape(value_type, axis_extents, name=name)
    _preflight_shape_bytes(shape, element_size, limits, name=name)


def _preflight_shape_bytes(
    shape: Sequence[int],
    element_size: int,
    limits: FormulaLimits,
    *,
    name: str,
) -> None:
    if any(size > limits.max_axis_extent for size in shape):
        raise FormulaBindingError(
            "FF2_LIMIT_EXCEEDED", f"{name!r} exceeds the Formula axis extent limit"
        )
    elements = math.prod(shape)
    if elements > limits.max_tensor_elements:
        raise FormulaBindingError(
            "FF2_LIMIT_EXCEEDED",
            f"{name!r} output allocation exceeds the Formula tensor element limit",
        )
    if elements * element_size > limits.max_tensor_bytes:
        raise FormulaBindingError(
            "FF2_LIMIT_EXCEEDED",
            f"{name!r} output allocation exceeds the Formula tensor byte limit",
        )


def _resolved_type_shape(
    value_type: TensorType,
    axis_extents: Mapping[str, int],
    *,
    name: str,
) -> tuple[int, ...]:
    shape: list[int] = []
    for axis, declared in zip(value_type.axis_names, value_type.sizes):
        resolved = declared if declared is not None else axis_extents.get(axis)
        if resolved is None:
            raise FormulaBindingError(
                "FF2_SYMBOL_UNBOUND",
                f"output {name!r} axis {axis!r} has no runtime extent",
            )
        shape.append(int(resolved))
    return tuple(shape)


def _validate_atom_operands(
    values: tuple[Tensor, ...], value_types: tuple[TensorType, ...]
) -> None:
    if len(values) != len(value_types):
        raise FormulaBindingError("FF2_ATOM_ARITY", "atom values and types have different arity")
    extents: dict[str, int] = {}
    device: torch.device | None = None
    dtype: torch.dtype | None = None
    for index, (value, value_type) in enumerate(zip(values, value_types)):
        name = f"operand[{index}]"
        _validate_tensor_against_type(value, value_type, name=name)
        _bind_axis_extents(extents, value, value_type, name=name)
        if device is None:
            device = value.device
            dtype = value.dtype
        elif value.device != device:
            raise FormulaBindingError(
                "FF2_DEVICE_MISMATCH", "all atom operands must use the same device"
            )
        elif value.dtype != dtype:
            raise FormulaBindingError(
                "FF2_RUNTIME_DTYPE_MISMATCH", "all atom operands must use the same dtype"
            )


def _bind_axis_extents(
    extents: dict[str, int], value: Tensor, value_type: TensorType, *, name: str
) -> None:
    for axis, size in zip(value_type.axis_names, value.shape):
        previous = extents.setdefault(axis, int(size))
        if previous != size:
            raise FormulaBindingError(
                "FF2_SYMBOL_UNBOUND",
                f"axis {axis!r} has conflicting extents {previous} and {size} at {name!r}",
            )


def _require_exact_keys(
    values: Mapping[str, Tensor], expected: Sequence[str], *, source: str
) -> None:
    expected_set = set(expected)
    received_set = set(values)
    if received_set != expected_set:
        raise FormulaBindingError(
            "FF2_BINDING_KEYS",
            f"{source} keys must be {sorted(expected_set)}, received {sorted(received_set)}",
        )


def _accumulation_dtype(dtype: torch.dtype, policy: str) -> torch.dtype:
    if policy == "activation":
        return dtype
    if dtype == torch.float64:
        return torch.float64
    return torch.float32


def _binary_with_accumulation(
    left: Tensor,
    right: Tensor,
    *,
    operation: Literal["add", "multiply"],
    accumulation_dtype: str,
) -> Tensor:
    compute_dtype = _accumulation_dtype(left.dtype, accumulation_dtype)
    left_compute = left.to(dtype=compute_dtype)
    right_compute = right.to(dtype=compute_dtype)
    if operation == "add":
        result = left_compute + right_compute
    else:
        result = left_compute * right_compute
    return result.to(dtype=left.dtype)


def _require_record_sequence(value: object, *, path: str) -> tuple[Mapping[str, object], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise FormulaSchemaError("FF2_PROGRAM_SCHEMA", f"{path} must be a sequence")
    if any(not isinstance(item, Mapping) for item in value):
        raise FormulaSchemaError("FF2_PROGRAM_SCHEMA", f"{path} must contain records")
    return tuple(value)


def _require_name_sequence(value: object, *, path: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise FormulaSchemaError("FF2_PROGRAM_SCHEMA", f"{path} must be a sequence")
    result = tuple(value)
    if any(not isinstance(item, str) for item in result):
        raise FormulaSchemaError("FF2_PROGRAM_SCHEMA", f"{path} must contain names")
    return result


def _freeze_json(value: object) -> object:
    if isinstance(value, _FrozenJsonObject):
        return value
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise FormulaSchemaError("FF2_ATTRIBUTE_SCHEMA", "attributes must be finite JSON")
        return value
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise FormulaSchemaError(
                "FF2_ATTRIBUTE_SCHEMA", "attribute object keys must be strings"
            )
        return _FrozenJsonObject(
            tuple((str(key), _freeze_json(item)) for key, item in sorted(value.items()))
        )
    raise FormulaSchemaError("FF2_ATTRIBUTE_SCHEMA", "attributes must contain JSON values")


def _thaw_json(value: object) -> object:
    if isinstance(value, _FrozenJsonObject):
        return {key: _thaw_json(item) for key, item in value.items}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


__all__ = [
    "AddAtom",
    "BankBinding",
    "ContractAtom",
    "DEFAULT_FORMULA_LIMITS",
    "FORMULA_LIMITS_V1_SCHEMA_VERSION",
    "FORMULA_LIMITS_V1_SCHEMA_REF",
    "FORMULA_EXECUTION_PLAN_V1_SCHEMA_REF",
    "FORMULA_EXECUTION_PLAN_V1_SCHEMA_VERSION",
    "FORMULA_PROGRAM_V2_SCHEMA_VERSION",
    "FORMULA_PROGRAM_V2_SCHEMA_REF",
    "FORMULA_TENSOR_TYPE_V1_SCHEMA_REF",
    "FORMULA_TRACE_V1_SCHEMA_VERSION",
    "FORMULA_TRACE_V1_SCHEMA_REF",
    "FormulaBindingError",
    "FormulaBankOperand",
    "FormulaExecutionPlanV2",
    "FormulaFabricV2",
    "FormulaFabricV2Result",
    "FormulaInstructionV2",
    "FormulaLimits",
    "FormulaProgram",
    "FormulaProgramError",
    "FormulaSchemaError",
    "FormulaSlotSpec",
    "FormulaTraceV2",
    "FormulaTypeError",
    "FormulaV2Error",
    "InputBinding",
    "PreparedFormulaBindings",
    "ReduceAtom",
    "ScaleAtom",
    "TensorType",
    "add",
    "build_lora_program",
    "contract",
    "dot",
    "reduce_sum",
    "scale",
]
