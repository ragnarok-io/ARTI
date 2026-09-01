"""Terminal alliance contracts for shape-autonomous ARTI Bank programs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
import re
from typing import ClassVar, Mapping

from torch import Tensor, nn

from .bank_query import QueryExecutionSignature
from .component_registry import ComponentRef
from .tensor_schema import GradientContract, ShapeRelation, TensorSchema, TensorSchemaError


TERMINAL_OUTPUT_ABI_VERSION = 1
BANK_EXECUTION_SIGNATURE_VERSION = 1
BANK_EXECUTION_SIGNATURE_V2_VERSION = 2

_NAME = re.compile(r"^[a-z][a-z0-9]*(?:[-_.][a-z0-9]+)*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class TerminalABIError(ValueError):
    """Raised when an autonomous Bank cannot satisfy its terminal alliance."""


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _fingerprint(value: object) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _require_name(value: str, *, field: str) -> None:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise TerminalABIError(f"{field} must be a canonical lowercase name")


def _require_text(value: str, *, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise TerminalABIError(f"{field} must be non-empty text")


def _require_sha256(value: str, *, field: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise TerminalABIError(f"{field} must be a SHA-256 hex digest")


def _freeze_json(value: object) -> object:
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TerminalABIError("contract mapping keys must be strings")
        return MappingProxyType(
            {key: _freeze_json(item) for key, item in sorted(value.items())}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item) for item in value)
    if isinstance(value, (str, bool, int)) or value is None:
        return value
    if isinstance(value, float):
        if value != value or value in {float("inf"), float("-inf")}:
            raise TerminalABIError("contract floats must be finite")
        return value
    raise TerminalABIError(f"unsupported contract value: {type(value).__name__}")


def _thaw_json(value: object) -> object:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True)
class TerminalField:
    """One named tensor field exposed at an autonomous Bank's terminal boundary."""

    name: str
    schema: TensorSchema
    semantic_role: str
    factor_index: int | None = None

    _runtime_contract_ref: ClassVar[str] = "arti/terminal-field@1"

    def __post_init__(self) -> None:
        _require_name(self.name, field="TerminalField.name")
        if not isinstance(self.schema, TensorSchema):
            raise TypeError("TerminalField.schema must be TensorSchema")
        _require_name(self.semantic_role, field="TerminalField.semantic_role")
        if self.factor_index is not None and (
            isinstance(self.factor_index, bool)
            or not isinstance(self.factor_index, int)
            or self.factor_index < 0
        ):
            raise TerminalABIError("factor_index must be None or a non-negative integer")

    def to_dict(self) -> dict[str, object]:
        return {
            "ref": self._runtime_contract_ref,
            "name": self.name,
            "schema": self.schema.to_dict(),
            "semantic_role": self.semantic_role,
            "factor_index": self.factor_index,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TerminalField:
        required = {"ref", "name", "schema", "semantic_role", "factor_index"}
        if not isinstance(value, Mapping) or set(value) != required:
            raise TerminalABIError("TerminalField payload contains missing or unknown fields")
        if value["ref"] != cls._runtime_contract_ref:
            raise TerminalABIError("TerminalField reference is invalid")
        return cls(
            name=value["name"],
            schema=TensorSchema.from_dict(value["schema"]),
            semantic_role=value["semantic_role"],
            factor_index=value["factor_index"],
        )


@dataclass(frozen=True)
class TerminalOutputABI:
    """The only tensor compatibility boundary shared by federated Bank programs."""

    fields: tuple[TerminalField, ...]
    factor_order: tuple[str, ...]
    validity_contract: str
    packing_contract: str
    score_contract: str
    consumer_contract: str
    gradient_contract: GradientContract
    schema_version: int = TERMINAL_OUTPUT_ABI_VERSION

    _component_reference: ClassVar[str] = "arti/terminal-output-abi@1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", tuple(self.fields))
        object.__setattr__(self, "factor_order", tuple(self.factor_order))
        if self.schema_version != TERMINAL_OUTPUT_ABI_VERSION:
            raise TerminalABIError("unsupported TerminalOutputABI version")
        if not self.fields:
            raise TerminalABIError("TerminalOutputABI requires at least one field")
        if any(not isinstance(field, TerminalField) for field in self.fields):
            raise TypeError("TerminalOutputABI fields must be TerminalField values")
        names = tuple(field.name for field in self.fields)
        if len(names) != len(set(names)):
            raise TerminalABIError("TerminalOutputABI field names must be unique")
        for factor in self.factor_order:
            _require_name(factor, field="factor_order")
        if len(self.factor_order) != len(set(self.factor_order)):
            raise TerminalABIError("factor_order must be unique")
        if any(
            field.factor_index is not None and field.factor_index >= len(self.factor_order)
            for field in self.fields
        ):
            raise TerminalABIError("TerminalField factor_index exceeds factor_order")
        for value, name in (
            (self.validity_contract, "validity_contract"),
            (self.packing_contract, "packing_contract"),
            (self.score_contract, "score_contract"),
            (self.consumer_contract, "consumer_contract"),
        ):
            _require_text(value, field=name)
        if not isinstance(self.gradient_contract, GradientContract):
            raise TypeError("gradient_contract must be GradientContract")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "ref": self._component_reference,
            "fields": [field.to_dict() for field in self.fields],
            "factor_order": list(self.factor_order),
            "validity_contract": self.validity_contract,
            "packing_contract": self.packing_contract,
            "score_contract": self.score_contract,
            "consumer_contract": self.consumer_contract,
            "gradient_contract": self.gradient_contract.to_dict(),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> TerminalOutputABI:
        required = {
            "schema_version",
            "ref",
            "fields",
            "factor_order",
            "validity_contract",
            "packing_contract",
            "score_contract",
            "consumer_contract",
            "gradient_contract",
            "fingerprint",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise TerminalABIError("TerminalOutputABI payload contains missing or unknown fields")
        if value["ref"] != cls._component_reference:
            raise TerminalABIError("TerminalOutputABI reference is invalid")
        fields = value["fields"]
        factors = value["factor_order"]
        if not isinstance(fields, (list, tuple)) or not isinstance(factors, (list, tuple)):
            raise TerminalABIError("TerminalOutputABI fields and factors must be sequences")
        result = cls(
            fields=tuple(TerminalField.from_dict(field) for field in fields),
            factor_order=tuple(factors),
            validity_contract=value["validity_contract"],
            packing_contract=value["packing_contract"],
            score_contract=value["score_contract"],
            consumer_contract=value["consumer_contract"],
            gradient_contract=GradientContract.from_dict(value["gradient_contract"]),
            schema_version=value["schema_version"],
        )
        if value["fingerprint"] != result.fingerprint:
            raise TerminalABIError("TerminalOutputABI fingerprint is invalid")
        return result

    def validate_outputs(self, outputs: Mapping[str, Tensor]) -> dict[str, int]:
        if not isinstance(outputs, Mapping):
            raise TypeError("terminal outputs must be a mapping")
        expected = {field.name for field in self.fields}
        if set(outputs) != expected:
            raise TerminalABIError("terminal output names do not match the ABI")
        symbols: dict[str, int] = {}
        for field in self.fields:
            try:
                symbols = field.schema.validate_tensor(
                    outputs[field.name], symbols=symbols, name=field.name
                )
            except TensorSchemaError as exc:
                raise TerminalABIError(str(exc)) from exc
        return symbols

    def compatible_with(self, other: TerminalOutputABI) -> bool:
        return isinstance(other, TerminalOutputABI) and self.fingerprint == other.fingerprint


@dataclass(frozen=True)
class BankExecutionSignature:
    """A portable declaration of one autonomous Bank program and its terminal ABI."""

    program_ref: str
    program_config_fingerprint: str
    program_state_fingerprint: str
    input_schema: TensorSchema
    output_schema: TensorSchema
    shape_relation: ShapeRelation
    query_contract: Mapping[str, object]
    local_normalization_contract: Mapping[str, object]
    terminal_adapter_ref: str
    terminal_abi_ref: str
    terminal_abi_fingerprint: str
    score_contract: str
    gradient_contract: GradientContract
    local_formula_ref: str | None = None
    local_refine_ref: str | None = None
    execution_capabilities: tuple[str, ...] = ()
    schema_version: int = BANK_EXECUTION_SIGNATURE_VERSION

    _component_reference: ClassVar[str] = "arti/bank-execution-signature@1"

    def __post_init__(self) -> None:
        if self.schema_version != BANK_EXECUTION_SIGNATURE_VERSION:
            raise TerminalABIError("unsupported BankExecutionSignature version")
        ComponentRef.parse(self.program_ref)
        ComponentRef.parse(self.terminal_adapter_ref)
        if self.terminal_abi_ref != TerminalOutputABI._component_reference:
            raise TerminalABIError("terminal_abi_ref must name TerminalOutputABI@1")
        for value, name in (
            (self.program_config_fingerprint, "program_config_fingerprint"),
            (self.program_state_fingerprint, "program_state_fingerprint"),
            (self.terminal_abi_fingerprint, "terminal_abi_fingerprint"),
        ):
            _require_sha256(value, field=name)
        for value, name in (
            (self.input_schema, "input_schema"),
            (self.output_schema, "output_schema"),
        ):
            if not isinstance(value, TensorSchema):
                raise TypeError(f"{name} must be TensorSchema")
        if not isinstance(self.shape_relation, ShapeRelation):
            raise TypeError("shape_relation must be ShapeRelation")
        self.shape_relation.validate_schemas(self.input_schema, self.output_schema)
        if not isinstance(self.gradient_contract, GradientContract):
            raise TypeError("gradient_contract must be GradientContract")
        for value, name in (
            (self.local_formula_ref, "local_formula_ref"),
            (self.local_refine_ref, "local_refine_ref"),
        ):
            if value is not None:
                ComponentRef.parse(value)
        query = _freeze_json(self.query_contract)
        normalization = _freeze_json(self.local_normalization_contract)
        if not isinstance(query, Mapping) or not isinstance(normalization, Mapping):
            raise TerminalABIError("query and normalization contracts must be mappings")
        if (
            query.get("fixed") is not True
            or query.get("deterministic") is not True
            or query.get("trainable") is not False
        ):
            raise TerminalABIError("federal Query must be fixed, deterministic, and non-trainable")
        if normalization.get("scope") != "bank_local":
            raise TerminalABIError("federal normalization must remain Bank-local")
        object.__setattr__(self, "query_contract", query)
        object.__setattr__(self, "local_normalization_contract", normalization)
        capabilities = tuple(self.execution_capabilities)
        for capability in capabilities:
            _require_name(capability, field="execution_capability")
        if tuple(sorted(set(capabilities))) != capabilities:
            raise TerminalABIError("execution_capabilities must be sorted and unique")
        object.__setattr__(self, "execution_capabilities", capabilities)
        _require_text(self.score_contract, field="score_contract")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self._payload())

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "ref": self._component_reference,
            "program_ref": self.program_ref,
            "program_config_fingerprint": self.program_config_fingerprint,
            "program_state_fingerprint": self.program_state_fingerprint,
            "input_schema": self.input_schema.to_dict(),
            "output_schema": self.output_schema.to_dict(),
            "shape_relation": self.shape_relation.to_dict(),
            "query_contract": _thaw_json(self.query_contract),
            "local_normalization_contract": _thaw_json(
                self.local_normalization_contract
            ),
            "local_formula_ref": self.local_formula_ref,
            "local_refine_ref": self.local_refine_ref,
            "terminal_adapter_ref": self.terminal_adapter_ref,
            "terminal_abi_ref": self.terminal_abi_ref,
            "terminal_abi_fingerprint": self.terminal_abi_fingerprint,
            "score_contract": self.score_contract,
            "gradient_contract": self.gradient_contract.to_dict(),
            "execution_capabilities": list(self.execution_capabilities),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> BankExecutionSignature:
        required = {
            "schema_version",
            "ref",
            "program_ref",
            "program_config_fingerprint",
            "program_state_fingerprint",
            "input_schema",
            "output_schema",
            "shape_relation",
            "query_contract",
            "local_normalization_contract",
            "local_formula_ref",
            "local_refine_ref",
            "terminal_adapter_ref",
            "terminal_abi_ref",
            "terminal_abi_fingerprint",
            "score_contract",
            "gradient_contract",
            "execution_capabilities",
            "fingerprint",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise TerminalABIError(
                "BankExecutionSignature payload contains missing or unknown fields"
            )
        if value["ref"] != cls._component_reference:
            raise TerminalABIError("BankExecutionSignature reference is invalid")
        capabilities = value["execution_capabilities"]
        if not isinstance(capabilities, (list, tuple)):
            raise TerminalABIError("execution_capabilities must be a sequence")
        result = cls(
            program_ref=value["program_ref"],
            program_config_fingerprint=value["program_config_fingerprint"],
            program_state_fingerprint=value["program_state_fingerprint"],
            input_schema=TensorSchema.from_dict(value["input_schema"]),
            output_schema=TensorSchema.from_dict(value["output_schema"]),
            shape_relation=ShapeRelation.from_dict(value["shape_relation"]),
            query_contract=value["query_contract"],
            local_normalization_contract=value["local_normalization_contract"],
            local_formula_ref=value["local_formula_ref"],
            local_refine_ref=value["local_refine_ref"],
            terminal_adapter_ref=value["terminal_adapter_ref"],
            terminal_abi_ref=value["terminal_abi_ref"],
            terminal_abi_fingerprint=value["terminal_abi_fingerprint"],
            score_contract=value["score_contract"],
            gradient_contract=GradientContract.from_dict(value["gradient_contract"]),
            execution_capabilities=tuple(capabilities),
            schema_version=value["schema_version"],
        )
        if value["fingerprint"] != result.fingerprint:
            raise TerminalABIError("BankExecutionSignature fingerprint is invalid")
        return result

    def validate_terminal_abi(self, abi: TerminalOutputABI) -> None:
        if not isinstance(abi, TerminalOutputABI):
            raise TypeError("abi must be TerminalOutputABI")
        if (
            self.terminal_abi_ref != abi._component_reference
            or self.terminal_abi_fingerprint != abi.fingerprint
        ):
            raise TerminalABIError("Bank signature does not match the terminal ABI")


@dataclass(frozen=True)
class BankExecutionSignatureV2:
    """A Bank program signature bound to an exact sealed Bank-owned Query."""

    program_ref: str
    program_api_identity: str
    program_config_fingerprint: str
    program_state_schema_fingerprint: str
    input_schema: TensorSchema
    output_schema: TensorSchema
    shape_relation: ShapeRelation
    query_signature: QueryExecutionSignature
    local_normalization_contract: Mapping[str, object]
    terminal_adapter_ref: str
    terminal_abi_ref: str
    terminal_abi_fingerprint: str
    score_contract: str
    gradient_contract: GradientContract
    local_formula_ref: str | None = None
    local_refine_ref: str | None = None
    execution_capabilities: tuple[str, ...] = ()
    schema_version: int = BANK_EXECUTION_SIGNATURE_V2_VERSION

    _component_reference: ClassVar[str] = "arti/bank-execution-signature@2"

    def __post_init__(self) -> None:
        if (
            type(self.schema_version) is not int
            or self.schema_version != BANK_EXECUTION_SIGNATURE_V2_VERSION
        ):
            raise TerminalABIError("unsupported BankExecutionSignatureV2 version")
        ComponentRef.parse(self.program_ref)
        if not isinstance(self.program_api_identity, str) or not self.program_api_identity:
            raise TerminalABIError("program_api_identity must be a qualified name")
        ComponentRef.parse(self.terminal_adapter_ref)
        if self.terminal_abi_ref != TerminalOutputABI._component_reference:
            raise TerminalABIError("terminal_abi_ref must name TerminalOutputABI@1")
        for value, name in (
            (self.program_config_fingerprint, "program_config_fingerprint"),
            (
                self.program_state_schema_fingerprint,
                "program_state_schema_fingerprint",
            ),
            (self.terminal_abi_fingerprint, "terminal_abi_fingerprint"),
        ):
            _require_sha256(value, field=name)
        for value, name in (
            (self.input_schema, "input_schema"),
            (self.output_schema, "output_schema"),
        ):
            if not isinstance(value, TensorSchema):
                raise TypeError(f"{name} must be TensorSchema")
        if not isinstance(self.shape_relation, ShapeRelation):
            raise TypeError("shape_relation must be ShapeRelation")
        self.shape_relation.validate_schemas(self.input_schema, self.output_schema)
        if not isinstance(self.query_signature, QueryExecutionSignature):
            raise TypeError("query_signature must be QueryExecutionSignature")
        if self.query_signature.input_schema.fingerprint != self.input_schema.fingerprint:
            raise TerminalABIError("Bank input schema must match its Query input schema")
        if not isinstance(self.gradient_contract, GradientContract):
            raise TypeError("gradient_contract must be GradientContract")
        if self.gradient_contract.mode != "autograd":
            raise TerminalABIError(
                "BankExecutionSignatureV2 supports only autograd gradients"
            )
        if (
            self.gradient_contract.fingerprint
            != self.query_signature.gradient_contract.fingerprint
        ):
            raise TerminalABIError(
                "Bank gradient contract must match its Query gradient contract"
            )
        for value, name in (
            (self.local_formula_ref, "local_formula_ref"),
            (self.local_refine_ref, "local_refine_ref"),
        ):
            if value is not None:
                ComponentRef.parse(value)
        normalization = _freeze_json(self.local_normalization_contract)
        if not isinstance(normalization, Mapping):
            raise TerminalABIError("local_normalization_contract must be a mapping")
        if normalization.get("scope") != "bank_local":
            raise TerminalABIError("federal normalization must remain Bank-local")
        object.__setattr__(self, "local_normalization_contract", normalization)
        capabilities = tuple(self.execution_capabilities)
        for capability in capabilities:
            _require_name(capability, field="execution_capability")
        if tuple(sorted(set(capabilities))) != capabilities:
            raise TerminalABIError("execution_capabilities must be sorted and unique")
        object.__setattr__(self, "execution_capabilities", capabilities)
        _require_text(self.score_contract, field="score_contract")

    @property
    def fingerprint(self) -> str:
        return _fingerprint(self._payload())

    @classmethod
    def from_program(
        cls,
        program: nn.Module,
        *,
        input_schema: TensorSchema,
        output_schema: TensorSchema,
        shape_relation: ShapeRelation,
        query_signature: QueryExecutionSignature,
        local_normalization_contract: Mapping[str, object],
        terminal_adapter_ref: str,
        terminal_abi_ref: str,
        terminal_abi_fingerprint: str,
        score_contract: str,
        gradient_contract: GradientContract,
        local_formula_ref: str | None = None,
        local_refine_ref: str | None = None,
        execution_capabilities: tuple[str, ...] = (),
    ) -> BankExecutionSignatureV2:
        """Bind a v2 signature to a fully constructed registered Bank program."""

        if not isinstance(program, nn.Module):
            raise TypeError("program must be an nn.Module")
        from .component_registry import component_spec

        spec = component_spec(program)
        return cls(
            program_ref=spec.reference,
            program_api_identity=spec.api,
            program_config_fingerprint=spec.config_fingerprint,
            program_state_schema_fingerprint=spec.parameter_schema_fingerprint,
            input_schema=input_schema,
            output_schema=output_schema,
            shape_relation=shape_relation,
            query_signature=query_signature,
            local_normalization_contract=local_normalization_contract,
            terminal_adapter_ref=terminal_adapter_ref,
            terminal_abi_ref=terminal_abi_ref,
            terminal_abi_fingerprint=terminal_abi_fingerprint,
            score_contract=score_contract,
            gradient_contract=gradient_contract,
            local_formula_ref=local_formula_ref,
            local_refine_ref=local_refine_ref,
            execution_capabilities=execution_capabilities,
        )

    def _payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "ref": self._component_reference,
            "program_ref": self.program_ref,
            "program_api_identity": self.program_api_identity,
            "program_config_fingerprint": self.program_config_fingerprint,
            "program_state_schema_fingerprint": self.program_state_schema_fingerprint,
            "input_schema": self.input_schema.to_dict(),
            "output_schema": self.output_schema.to_dict(),
            "shape_relation": self.shape_relation.to_dict(),
            "query_signature": self.query_signature.to_dict(),
            "local_normalization_contract": _thaw_json(
                self.local_normalization_contract
            ),
            "local_formula_ref": self.local_formula_ref,
            "local_refine_ref": self.local_refine_ref,
            "terminal_adapter_ref": self.terminal_adapter_ref,
            "terminal_abi_ref": self.terminal_abi_ref,
            "terminal_abi_fingerprint": self.terminal_abi_fingerprint,
            "score_contract": self.score_contract,
            "gradient_contract": self.gradient_contract.to_dict(),
            "execution_capabilities": list(self.execution_capabilities),
        }

    def to_dict(self) -> dict[str, object]:
        return {**self._payload(), "fingerprint": self.fingerprint}

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> BankExecutionSignatureV2:
        required = {
            "schema_version",
            "ref",
            "program_ref",
            "program_api_identity",
            "program_config_fingerprint",
            "program_state_schema_fingerprint",
            "input_schema",
            "output_schema",
            "shape_relation",
            "query_signature",
            "local_normalization_contract",
            "local_formula_ref",
            "local_refine_ref",
            "terminal_adapter_ref",
            "terminal_abi_ref",
            "terminal_abi_fingerprint",
            "score_contract",
            "gradient_contract",
            "execution_capabilities",
            "fingerprint",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise TerminalABIError(
                "BankExecutionSignatureV2 payload contains missing or unknown fields"
            )
        if value["ref"] != cls._component_reference:
            raise TerminalABIError("BankExecutionSignatureV2 reference is invalid")
        capabilities = value["execution_capabilities"]
        if not isinstance(capabilities, (list, tuple)):
            raise TerminalABIError("execution_capabilities must be a sequence")
        result = cls(
            program_ref=value["program_ref"],
            program_api_identity=value["program_api_identity"],
            program_config_fingerprint=value["program_config_fingerprint"],
            program_state_schema_fingerprint=value[
                "program_state_schema_fingerprint"
            ],
            input_schema=TensorSchema.from_dict(value["input_schema"]),
            output_schema=TensorSchema.from_dict(value["output_schema"]),
            shape_relation=ShapeRelation.from_dict(value["shape_relation"]),
            query_signature=QueryExecutionSignature.from_dict(value["query_signature"]),
            local_normalization_contract=value["local_normalization_contract"],
            local_formula_ref=value["local_formula_ref"],
            local_refine_ref=value["local_refine_ref"],
            terminal_adapter_ref=value["terminal_adapter_ref"],
            terminal_abi_ref=value["terminal_abi_ref"],
            terminal_abi_fingerprint=value["terminal_abi_fingerprint"],
            score_contract=value["score_contract"],
            gradient_contract=GradientContract.from_dict(value["gradient_contract"]),
            execution_capabilities=tuple(capabilities),
            schema_version=value["schema_version"],
        )
        if value["fingerprint"] != result.fingerprint:
            raise TerminalABIError("BankExecutionSignatureV2 fingerprint is invalid")
        return result

    def validate_terminal_abi(self, abi: TerminalOutputABI) -> None:
        if not isinstance(abi, TerminalOutputABI):
            raise TypeError("abi must be TerminalOutputABI")
        if (
            self.terminal_abi_ref != abi._component_reference
            or self.terminal_abi_fingerprint != abi.fingerprint
        ):
            raise TerminalABIError("Bank signature does not match the terminal ABI")


__all__ = [
    "BANK_EXECUTION_SIGNATURE_VERSION",
    "BANK_EXECUTION_SIGNATURE_V2_VERSION",
    "TERMINAL_OUTPUT_ABI_VERSION",
    "BankExecutionSignature",
    "BankExecutionSignatureV2",
    "TerminalABIError",
    "TerminalField",
    "TerminalOutputABI",
]
