"""Effect-aware Formula execution for Bank Federation self-modification."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import ClassVar, Mapping

import torch
from torch import Tensor, nn

from .formula_v2 import (
    BankBinding,
    FormulaBankOperand,
    FormulaFabricV2,
    FormulaInstructionV2,
    FormulaProgram,
    FormulaProgramError,
    FormulaTraceV2,
    InputBinding,
    TensorType,
    formula_program_dependency_refs,
    formula_program_effect_refs,
)


FORMULA_EFFECT_PROGRAM_V1_SCHEMA_REF = "arti/formula-effect-program@1"
FORMULA_EFFECT_PROGRAM_V1_SCHEMA_VERSION = 1
FORMULA_EFFECT_PROGRAM_V2_SCHEMA_REF = "arti/formula-effect-program@2"
FORMULA_EFFECT_PROGRAM_V2_SCHEMA_VERSION = 2
FORMULA_EFFECT_PROGRAM_V3_SCHEMA_REF = "arti/formula-effect-program@3"
FORMULA_EFFECT_PROGRAM_V3_SCHEMA_VERSION = 3
NEURAL_PLASTICITY_ATOM_REF = "arti/formula-atom-neural-plasticity@1"
NEURAL_PLASTICITY_BLEND_ATOM_REF = "arti/formula-atom-neural-plasticity-blend@1"
NEURAL_PLASTICITY_OUTER_ATOM_REF = "arti/formula-atom-neural-plasticity-outer@1"
NEURAL_PLASTICITY_OUTER_V2_ATOM_REF = "arti/formula-atom-neural-plasticity-outer@2"
NEURAL_PLASTICITY_TRANSPORT_ATOM_REF = "arti/formula-atom-neural-plasticity-transport@1"
NEURAL_PLASTICITY_POLYNOMIAL_ATOM_REF = "arti/formula-atom-neural-plasticity-polynomial@1"
NEURAL_PLASTICITY_PROXIMAL_ATOM_REF = "arti/formula-atom-neural-plasticity-proximal@1"
NEURAL_PLASTICITY_EFFECT_REFS = frozenset(
    {
        NEURAL_PLASTICITY_ATOM_REF,
        NEURAL_PLASTICITY_BLEND_ATOM_REF,
        NEURAL_PLASTICITY_OUTER_ATOM_REF,
        NEURAL_PLASTICITY_OUTER_V2_ATOM_REF,
        NEURAL_PLASTICITY_TRANSPORT_ATOM_REF,
        NEURAL_PLASTICITY_POLYNOMIAL_ATOM_REF,
        NEURAL_PLASTICITY_PROXIMAL_ATOM_REF,
    }
)


def _slot_dependencies(program: FormulaProgram) -> dict[str, frozenset[str]]:
    dependencies: dict[str, frozenset[str]] = {
        binding.name: frozenset((binding.name,)) for binding in program.bindings
    }
    for instruction in program.instructions:
        dependencies[instruction.output_slot] = frozenset(
            dependency
            for input_slot in instruction.input_slots
            for dependency in dependencies[input_slot]
        )
    return dependencies


def _slot_lineage(program: FormulaProgram) -> dict[str, frozenset[str]]:
    lineage: dict[str, frozenset[str]] = {
        binding.name: frozenset((binding.name,)) for binding in program.bindings
    }
    for instruction in program.instructions:
        lineage[instruction.output_slot] = frozenset(
            {
                instruction.output_slot,
                *(
                    dependency
                    for input_slot in instruction.input_slots
                    for dependency in ({input_slot} | set(lineage[input_slot]))
                ),
            }
        )
    return lineage


@dataclass(frozen=True)
class FormulaEffectProgram:
    """Validate a Formula SSA program whose effect targets its owning site."""

    program: FormulaProgram
    data_input_name: str
    state_type: TensorType
    schema_version: int = FORMULA_EFFECT_PROGRAM_V1_SCHEMA_VERSION

    _component_reference: ClassVar[str] = FORMULA_EFFECT_PROGRAM_V1_SCHEMA_REF

    def __post_init__(self) -> None:
        if not isinstance(self.program, FormulaProgram):
            raise TypeError("FormulaEffectProgram.program must be FormulaProgram")
        if not isinstance(self.state_type, TensorType):
            raise TypeError("FormulaEffectProgram.state_type must be TensorType")
        if self.schema_version != FORMULA_EFFECT_PROGRAM_V1_SCHEMA_VERSION:
            raise FormulaProgramError(
                "FF3_UNSUPPORTED_SCHEMA", "unsupported FormulaEffectProgram schema"
            )
        bindings = {binding.name: binding for binding in self.program.bindings}
        if not isinstance(bindings.get(self.data_input_name), InputBinding):
            raise FormulaProgramError(
                "FF3_DATA_BINDING", "data_input_name must identify a Formula input binding"
            )
        if any(
            isinstance(binding, BankBinding)
            and binding.source_ref == NEURAL_PLASTICITY_ATOM_REF
            for binding in self.program.bindings
        ):
            raise FormulaProgramError(
                "FF3_EXPLICIT_SELF_STATE",
                "NeuralPlasticity self state cannot appear as a Formula binding",
            )
        effects = tuple(
            instruction
            for instruction in self.program.instructions
            if instruction.atom_ref == NEURAL_PLASTICITY_ATOM_REF
        )
        if len(effects) != 1 or formula_program_effect_refs(self.program) != (
            NEURAL_PLASTICITY_ATOM_REF,
        ):
            raise FormulaProgramError(
                "FF3_EFFECT_COUNT", "FormulaEffectProgram@1 requires exactly one effect atom"
            )
        effect = effects[0]
        if effect.input_slots[0] != self.data_input_name:
            raise FormulaProgramError(
                "FF3_DATA_IDENTITY",
                "NeuralPlasticity@1 must consume the unmodified current data binding",
            )
        if self.program.outputs != (effect.output_slot,):
            raise FormulaProgramError(
                "FF3_DATA_IDENTITY",
                "the effect identity output must be the only public Formula output",
            )
        additive_slot, multiplicative_slot = effect.input_slots[1:]
        if (
            self.program.slot_types[additive_slot] != self.state_type
            or self.program.slot_types[multiplicative_slot] != self.state_type
        ):
            raise FormulaProgramError(
                "FF3_EFFECT_TYPE",
                "effect operands must preserve the execution site's state type",
            )
        dependencies = _slot_dependencies(self.program)
        if self.data_input_name not in (
            dependencies[additive_slot] | dependencies[multiplicative_slot]
        ):
            raise FormulaProgramError(
                "FF3_DATA_CONDITIONED_EFFECT",
                "at least one effect operand must depend on current data",
            )

    @property
    def effect_instruction(self) -> FormulaInstructionV2:
        return next(
            instruction
            for instruction in self.program.instructions
            if instruction.atom_ref == NEURAL_PLASTICITY_ATOM_REF
        )

    @property
    def dependency_refs(self) -> tuple[str, ...]:
        return formula_program_dependency_refs(self.program)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_ref": FORMULA_EFFECT_PROGRAM_V1_SCHEMA_REF,
            "schema_version": self.schema_version,
            "program": self.program.to_dict(),
            "program_fingerprint": self.program.fingerprint,
            "data_input_name": self.data_input_name,
            "state_type": self.state_type.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> FormulaEffectProgram:
        required = {
            "schema_ref",
            "schema_version",
            "program",
            "program_fingerprint",
            "data_input_name",
            "state_type",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise FormulaProgramError(
                "FF3_EFFECT_PROGRAM_SCHEMA",
                "FormulaEffectProgram payload contains missing or unknown fields",
            )
        if value["schema_ref"] != FORMULA_EFFECT_PROGRAM_V1_SCHEMA_REF:
            raise FormulaProgramError(
                "FF3_EFFECT_PROGRAM_SCHEMA", "FormulaEffectProgram schema reference is invalid"
            )
        program = FormulaProgram.from_dict(value["program"])
        if value["program_fingerprint"] != program.fingerprint:
            raise FormulaProgramError(
                "FF3_EFFECT_PROGRAM_SCHEMA", "Formula body fingerprint is invalid"
            )
        return cls(
            program=program,
            data_input_name=value["data_input_name"],
            state_type=TensorType.from_dict(value["state_type"]),
            schema_version=value["schema_version"],
        )

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class FormulaEffectProgramV2:
    """Validate one of the versioned execution-site NeuralPlasticity effects."""

    program: FormulaProgram
    data_input_name: str
    state_type: TensorType
    schema_version: int = FORMULA_EFFECT_PROGRAM_V2_SCHEMA_VERSION

    _component_reference: ClassVar[str] = FORMULA_EFFECT_PROGRAM_V2_SCHEMA_REF

    def __post_init__(self) -> None:
        if not isinstance(self.program, FormulaProgram):
            raise TypeError("FormulaEffectProgramV2.program must be FormulaProgram")
        if not isinstance(self.state_type, TensorType):
            raise TypeError("FormulaEffectProgramV2.state_type must be TensorType")
        if self.schema_version != FORMULA_EFFECT_PROGRAM_V2_SCHEMA_VERSION:
            raise FormulaProgramError(
                "FF4_UNSUPPORTED_SCHEMA", "unsupported FormulaEffectProgramV2 schema"
            )
        bindings = {binding.name: binding for binding in self.program.bindings}
        if not isinstance(bindings.get(self.data_input_name), InputBinding):
            raise FormulaProgramError(
                "FF4_DATA_BINDING", "data_input_name must identify a Formula input binding"
            )
        if any(
            isinstance(binding, BankBinding)
            and binding.source_ref in NEURAL_PLASTICITY_EFFECT_REFS
            for binding in self.program.bindings
        ):
            raise FormulaProgramError(
                "FF4_EXPLICIT_SELF_STATE",
                "NeuralPlasticity self state cannot appear as a Formula binding",
            )
        effects = tuple(
            instruction
            for instruction in self.program.instructions
            if instruction.atom_ref in NEURAL_PLASTICITY_EFFECT_REFS
        )
        if len(effects) != 1 or formula_program_effect_refs(self.program) != (
            effects[0].atom_ref,
        ):
            raise FormulaProgramError(
                "FF4_EFFECT_COUNT", "FormulaEffectProgram@2 requires exactly one effect atom"
            )
        effect = effects[0]
        if effect.input_slots[0] != self.data_input_name:
            raise FormulaProgramError(
                "FF4_DATA_IDENTITY",
                "NeuralPlasticity effects must consume the unmodified current data binding",
            )
        if self.program.outputs != (effect.output_slot,):
            raise FormulaProgramError(
                "FF4_DATA_IDENTITY",
                "the effect identity output must be the only public Formula output",
            )
        self._validate_effect_types(effect)
        dependencies = _slot_dependencies(self.program)
        effect_dependencies = frozenset(
            dependency
            for slot in effect.input_slots[1:]
            for dependency in dependencies[slot]
        )
        if self.data_input_name not in effect_dependencies:
            raise FormulaProgramError(
                "FF4_DATA_CONDITIONED_EFFECT",
                "at least one effect operand must depend on current data",
            )

    def _validate_effect_types(self, effect: FormulaInstructionV2) -> None:
        operand_types = tuple(self.program.slot_types[slot] for slot in effect.input_slots[1:])
        if effect.atom_ref == NEURAL_PLASTICITY_ATOM_REF:
            if operand_types != (self.state_type, self.state_type):
                raise FormulaProgramError(
                    "FF4_EFFECT_TYPE",
                    "affine effect operands must preserve the execution-site state type",
                )
            return
        if effect.atom_ref == NEURAL_PLASTICITY_BLEND_ATOM_REF:
            if operand_types != (self.state_type, self.state_type):
                raise FormulaProgramError(
                    "FF4_EFFECT_TYPE",
                    "blend target and amount must preserve the execution-site state type",
                )
            return
        if effect.atom_ref in {
            NEURAL_PLASTICITY_OUTER_ATOM_REF,
            NEURAL_PLASTICITY_OUTER_V2_ATOM_REF,
        }:
            left_type, right_type, rate_type = operand_types[:3]
            count_type = None if len(operand_types) == 3 else operand_types[3]
            expected_axes = (*left_type.axis_names, *right_type.axis_names)
            expected_sizes = (*left_type.sizes, *right_type.sizes)
            if (
                len(left_type.axis_names) != 1
                or len(right_type.axis_names) != 1
                or rate_type.axis_names
                or rate_type.sizes
                or self.state_type.axis_names != expected_axes
                or self.state_type.sizes != expected_sizes
                or len({left_type.dtype, right_type.dtype, rate_type.dtype, self.state_type.dtype})
                != 1
                or len(
                    {
                        left_type.domain,
                        right_type.domain,
                        rate_type.domain,
                        self.state_type.domain,
                    }
                )
                != 1
                or (
                    count_type is not None
                    and (
                        count_type.axis_names
                        or count_type.sizes
                        or count_type.dtype != self.state_type.dtype
                        or count_type.domain != self.state_type.domain
                    )
                )
            ):
                raise FormulaProgramError(
                    "FF4_EFFECT_TYPE",
                    "outer factors must form the rank-two execution-site state type",
                )
            if effect.atom_ref == NEURAL_PLASTICITY_OUTER_V2_ATOM_REF:
                max_executions = dict(effect.attributes).get("max_executions")
                if (
                    count_type is None
                    or isinstance(max_executions, bool)
                    or not isinstance(max_executions, int)
                    or max_executions <= 0
                ):
                    raise FormulaProgramError(
                        "FF4_EFFECT_TYPE",
                        "Outer@2 requires a scalar execution_count and positive max_executions",
                    )
            return
        if effect.atom_ref == NEURAL_PLASTICITY_TRANSPORT_ATOM_REF:
            bias_type, output_type, input_type, rate_type = operand_types
            self._validate_axis_factor_effect(
                effect=effect,
                bias_type=bias_type,
                factor_types=(output_type, input_type),
                rate_type=rate_type,
                effect_name="transport",
            )
            return
        if effect.atom_ref == NEURAL_PLASTICITY_POLYNOMIAL_ATOM_REF:
            bias_type, output_type, left_type, right_type, rate_type = operand_types
            self._validate_axis_factor_effect(
                effect=effect,
                bias_type=bias_type,
                factor_types=(output_type, left_type, right_type),
                rate_type=rate_type,
                effect_name="polynomial",
            )
            return
        if effect.atom_ref == NEURAL_PLASTICITY_PROXIMAL_ATOM_REF:
            bias_type, strength_type = operand_types
            if bias_type != self.state_type or strength_type != self.state_type:
                raise FormulaProgramError(
                    "FF4_EFFECT_TYPE",
                    "proximal bias and strength must preserve the execution-site state type",
                )
            return
        raise AssertionError("unreachable NeuralPlasticity effect kind")

    def _validate_axis_factor_effect(
        self,
        *,
        effect: FormulaInstructionV2,
        bias_type: TensorType,
        factor_types: tuple[TensorType, ...],
        rate_type: TensorType,
        effect_name: str,
    ) -> None:
        state_axis = dict(effect.attributes).get("state_axis")
        if not isinstance(state_axis, str) or state_axis not in self.state_type.axis_names:
            raise FormulaProgramError(
                "FF4_EFFECT_TYPE",
                f"{effect_name} state_axis must identify an execution-site state axis",
            )
        factor_type = factor_types[0]
        state_size = self.state_type.size_for(state_axis)
        if (
            bias_type != self.state_type
            or any(item != factor_type for item in factor_types[1:])
            or len(factor_type.axis_names) != 2
            or factor_type.axis_names[0] != state_axis
            or factor_type.sizes[0] != state_size
            or rate_type.axis_names
            or rate_type.sizes
            or len(
                {
                    self.state_type.dtype,
                    bias_type.dtype,
                    factor_type.dtype,
                    rate_type.dtype,
                }
            )
            != 1
            or factor_type.domain != self.state_type.domain
            or rate_type.domain != self.state_type.domain
        ):
            raise FormulaProgramError(
                "FF4_EFFECT_TYPE",
                f"{effect_name} factors must use [state-axis, rank] and match state dtype",
            )

    @property
    def effect_instruction(self) -> FormulaInstructionV2:
        return next(
            instruction
            for instruction in self.program.instructions
            if instruction.atom_ref in NEURAL_PLASTICITY_EFFECT_REFS
        )

    @property
    def dependency_refs(self) -> tuple[str, ...]:
        return formula_program_dependency_refs(self.program)

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_ref": FORMULA_EFFECT_PROGRAM_V2_SCHEMA_REF,
            "schema_version": self.schema_version,
            "program": self.program.to_dict(),
            "program_fingerprint": self.program.fingerprint,
            "data_input_name": self.data_input_name,
            "state_type": self.state_type.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> FormulaEffectProgramV2:
        required = {
            "schema_ref",
            "schema_version",
            "program",
            "program_fingerprint",
            "data_input_name",
            "state_type",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise FormulaProgramError(
                "FF4_EFFECT_PROGRAM_SCHEMA",
                "FormulaEffectProgramV2 payload contains missing or unknown fields",
            )
        if value["schema_ref"] != FORMULA_EFFECT_PROGRAM_V2_SCHEMA_REF:
            raise FormulaProgramError(
                "FF4_EFFECT_PROGRAM_SCHEMA", "FormulaEffectProgramV2 schema reference is invalid"
            )
        program = FormulaProgram.from_dict(value["program"])
        if value["program_fingerprint"] != program.fingerprint:
            raise FormulaProgramError(
                "FF4_EFFECT_PROGRAM_SCHEMA", "Formula body fingerprint is invalid"
            )
        return cls(
            program=program,
            data_input_name=value["data_input_name"],
            state_type=TensorType.from_dict(value["state_type"]),
            schema_version=value["schema_version"],
        )

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class FormulaEffectProgramV3(FormulaEffectProgramV2):
    """Interleave self-network effects through one ordinary Formula data path."""

    schema_version: int = FORMULA_EFFECT_PROGRAM_V3_SCHEMA_VERSION

    _component_reference: ClassVar[str] = FORMULA_EFFECT_PROGRAM_V3_SCHEMA_REF

    def __post_init__(self) -> None:
        if not isinstance(self.program, FormulaProgram):
            raise TypeError("FormulaEffectProgramV3.program must be FormulaProgram")
        if not isinstance(self.state_type, TensorType):
            raise TypeError("FormulaEffectProgramV3.state_type must be TensorType")
        if self.schema_version != FORMULA_EFFECT_PROGRAM_V3_SCHEMA_VERSION:
            raise FormulaProgramError(
                "FF5_UNSUPPORTED_SCHEMA", "unsupported FormulaEffectProgramV3 schema"
            )
        bindings = {binding.name: binding for binding in self.program.bindings}
        if not isinstance(bindings.get(self.data_input_name), InputBinding):
            raise FormulaProgramError(
                "FF5_DATA_BINDING", "data_input_name must identify a Formula input binding"
            )
        if any(
            isinstance(binding, BankBinding)
            and binding.source_ref in NEURAL_PLASTICITY_EFFECT_REFS
            for binding in self.program.bindings
        ):
            raise FormulaProgramError(
                "FF5_EXPLICIT_SELF_STATE",
                "NeuralPlasticity self state cannot appear as a Formula binding",
            )
        effects = tuple(
            instruction
            for instruction in self.program.instructions
            if instruction.atom_ref in NEURAL_PLASTICITY_EFFECT_REFS
        )
        if not effects or formula_program_effect_refs(self.program) != tuple(
            sorted({effect.atom_ref for effect in effects})
        ):
            raise FormulaProgramError(
                "FF5_EFFECT_COUNT", "FormulaEffectProgram@3 requires one or more effect atoms"
            )
        if len(self.program.outputs) != 1:
            raise FormulaProgramError(
                "FF5_OUTPUT_COUNT", "FormulaEffectProgram@3 requires one public data output"
            )

        instructions_by_output = {
            instruction.output_slot: instruction for instruction in self.program.instructions
        }
        public_output = self.program.outputs[0]
        output_instruction = instructions_by_output.get(public_output)
        if (
            public_output in {effect.output_slot for effect in effects}
            or output_instruction is None
            or output_instruction.atom_ref in NEURAL_PLASTICITY_EFFECT_REFS
        ):
            raise FormulaProgramError(
                "FF5_EFFECT_POSITION",
                "the NeuralPlasticity effect must be followed by an ordinary Formula instruction",
            )

        dependencies = _slot_lineage(self.program)
        previous_effect_slot: str | None = None
        for effect in effects:
            data_predecessor = instructions_by_output.get(effect.input_slots[0])
            if (
                data_predecessor is None
                or data_predecessor.atom_ref in NEURAL_PLASTICITY_EFFECT_REFS
            ):
                raise FormulaProgramError(
                    "FF5_EFFECT_POSITION",
                    "each NeuralPlasticity effect must follow an ordinary Formula instruction",
                )
            self._validate_effect_types(effect)
            data_dependencies = dependencies[effect.input_slots[0]]
            if self.data_input_name not in data_dependencies:
                raise FormulaProgramError(
                    "FF5_DATA_PATH", "each effect data operand must derive from current data"
                )
            if previous_effect_slot is not None and previous_effect_slot not in data_dependencies:
                raise FormulaProgramError(
                    "FF5_EFFECT_CHAIN",
                    "NeuralPlasticity effects must form one ordered Formula data path",
                )
            effect_dependencies = frozenset(
                dependency
                for slot in effect.input_slots[1:]
                for dependency in dependencies[slot]
            )
            if self.data_input_name not in effect_dependencies:
                raise FormulaProgramError(
                    "FF5_DATA_CONDITIONED_EFFECT",
                    "each effect requires an operand that depends on current data",
                )
            previous_effect_slot = effect.output_slot
        if previous_effect_slot not in dependencies[public_output]:
            raise FormulaProgramError(
                "FF5_DATA_PATH", "the public output must execute downstream of every effect"
            )

    @property
    def effect_instructions(self) -> tuple[FormulaInstructionV2, ...]:
        return tuple(
            instruction
            for instruction in self.program.instructions
            if instruction.atom_ref in NEURAL_PLASTICITY_EFFECT_REFS
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_ref": FORMULA_EFFECT_PROGRAM_V3_SCHEMA_REF,
            "schema_version": self.schema_version,
            "program": self.program.to_dict(),
            "program_fingerprint": self.program.fingerprint,
            "data_input_name": self.data_input_name,
            "state_type": self.state_type.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> FormulaEffectProgramV3:
        required = {
            "schema_ref",
            "schema_version",
            "program",
            "program_fingerprint",
            "data_input_name",
            "state_type",
        }
        if not isinstance(value, Mapping) or set(value) != required:
            raise FormulaProgramError(
                "FF5_EFFECT_PROGRAM_SCHEMA",
                "FormulaEffectProgramV3 payload contains missing or unknown fields",
            )
        if value["schema_ref"] != FORMULA_EFFECT_PROGRAM_V3_SCHEMA_REF:
            raise FormulaProgramError(
                "FF5_EFFECT_PROGRAM_SCHEMA", "FormulaEffectProgramV3 schema reference is invalid"
            )
        program = FormulaProgram.from_dict(value["program"])
        if value["program_fingerprint"] != program.fingerprint:
            raise FormulaProgramError(
                "FF5_EFFECT_PROGRAM_SCHEMA", "Formula body fingerprint is invalid"
            )
        return cls(
            program=program,
            data_input_name=value["data_input_name"],
            state_type=TensorType.from_dict(value["state_type"]),
            schema_version=value["schema_version"],
        )


@dataclass(frozen=True)
class NeuralPlasticityEffect:
    """Update operands emitted for the current execution site."""

    instruction_id: str
    additive_update: Tensor
    multiplicative_update: Tensor


@dataclass(frozen=True)
class NeuralPlasticityEffectV2:
    """Typed operands emitted by one versioned execution-site effect."""

    instruction_id: str
    atom_ref: str
    operands: tuple[Tensor, ...]
    attributes: tuple[tuple[str, object], ...] = ()

    def __post_init__(self) -> None:
        if self.atom_ref not in NEURAL_PLASTICITY_EFFECT_REFS:
            raise FormulaProgramError("FF4_EFFECT_KIND", "unknown NeuralPlasticity effect atom")
        object.__setattr__(self, "operands", tuple(self.operands))
        object.__setattr__(self, "attributes", tuple(self.attributes))


def apply_neural_plasticity_effect(
    effect: NeuralPlasticityEffectV2,
    state: Tensor,
    *,
    state_type: TensorType,
    execution_count: Tensor | None = None,
    max_executions: int = 16,
) -> Tensor:
    """Apply one Formula effect to implicit execution-site state.

    The Formula atom emits operands but never receives ``state`` as a binding.
    Keeping the application here gives Bank-local execution and architecture
    search one implementation of the self-state transition algebra.
    """

    if not isinstance(effect, NeuralPlasticityEffectV2):
        raise TypeError("effect must be NeuralPlasticityEffectV2")
    if not isinstance(state, Tensor) or not state.is_floating_point():
        raise TypeError("NeuralPlasticity state must be a floating Tensor")
    if not isinstance(state_type, TensorType):
        raise TypeError("state_type must be TensorType")

    from .formula_v2 import _validate_tensor_against_type

    _validate_tensor_against_type(state, state_type, name="neural-plasticity-state")

    if execution_count is not None:
        if effect.atom_ref == NEURAL_PLASTICITY_OUTER_V2_ATOM_REF:
            raise FormulaProgramError(
                "FF_EFFECT_EXECUTION_COUNT",
                "generic execution_count cannot wrap Outer@2's own count operand",
            )
        if (
            execution_count.ndim != 0
            or execution_count.dtype != state.dtype
            or execution_count.device != state.device
        ):
            raise FormulaProgramError(
                "FF_EFFECT_EXECUTION_COUNT",
                "execution_count must be a scalar matching execution-site dtype and device",
            )
        if (
            isinstance(max_executions, bool)
            or not isinstance(max_executions, int)
            or max_executions <= 0
        ):
            raise FormulaProgramError(
                "FF_EFFECT_EXECUTION_COUNT",
                "max_executions must be a positive integer",
            )
        bounded = execution_count.clamp(0.0, float(max_executions))
        hard_steps = int(bounded.detach().round().item())
        count_surrogate = torch.is_grad_enabled() and execution_count.requires_grad
        result = state
        states = [state.detach()] if count_surrogate else []
        for _ in range(hard_steps):
            result = apply_neural_plasticity_effect(
                effect,
                result,
                state_type=state_type,
            )
            if count_surrogate:
                states.append(result.detach())
        if not count_surrogate or not bool(torch.isfinite(result).all()):
            # Do not repair an invalid *selected* hard transition with a surrogate.
            return result

        # Only the selected hard path carries state/operand gradients. Extend its
        # detached snapshots under no_grad: even a finite unselected output can
        # have an invalid backward path. Never put that path in the autograd graph.
        # A non-finite trial ends the count surrogate's admissible prefix; neither
        # that trial nor higher counts enter the zero-valued correction below.
        with torch.no_grad():
            current = states[-1]
            for _ in range(hard_steps, max_executions):
                successor = apply_neural_plasticity_effect(
                    effect,
                    current,
                    state_type=state_type,
                )
                if not bool(torch.isfinite(successor).all()):
                    break
                states.append(successor)
                current = successor
        choices = torch.arange(
            len(states),
            dtype=bounded.dtype,
            device=bounded.device,
        )
        soft_weights = torch.softmax(-4.0 * (choices - bounded).square(), dim=0)
        # Subtract weights before the contraction, rather than subtracting two
        # potentially overflowing state mixtures. All stacked values are finite
        # and detached, so this adds count credit without changing the hard value.
        zero_weights = soft_weights - soft_weights.detach()
        return result + torch.einsum("k,k...->...", zero_weights, torch.stack(states, dim=0))

    def matching_state_operand(value: Tensor, *, name: str) -> Tensor:
        _validate_tensor_against_type(value, state_type, name=name)
        if (
            value.shape != state.shape
            or value.dtype != state.dtype
            or value.device != state.device
        ):
            raise FormulaProgramError(
                "FF_EFFECT_STATE_MISMATCH",
                f"{name} must exactly match execution-site state",
            )
        return value

    def matching_low_rank_factors(
        factors: tuple[Tensor, ...],
        *,
        rate: Tensor,
        state_axis: str,
        effect_name: str,
    ) -> int:
        if state_axis not in state_type.axis_names:
            raise FormulaProgramError(
                "FF_EFFECT_STATE_AXIS",
                f"{effect_name} state_axis must identify an execution-site state axis",
            )
        state_axis_index = state_type.axis_names.index(state_axis)
        if (
            not factors
            or any(item.ndim != 2 for item in factors)
            or any(item.shape != factors[0].shape for item in factors[1:])
            or factors[0].shape[0] != state.shape[state_axis_index]
            or rate.ndim != 0
            or any(item.dtype != state.dtype for item in factors)
            or any(item.device != state.device for item in factors)
            or rate.dtype != state.dtype
            or rate.device != state.device
        ):
            raise FormulaProgramError(
                "FF_EFFECT_FACTOR_MISMATCH",
                f"{effect_name} factors must use [state-axis, rank] and match state",
            )
        return state_axis_index

    if effect.atom_ref == NEURAL_PLASTICITY_ATOM_REF:
        additive = matching_state_operand(
            effect.operands[0], name="additive-update"
        )
        multiplicative = matching_state_operand(
            effect.operands[1], name="multiplicative-update"
        )
        return state + additive + state * multiplicative
    if effect.atom_ref == NEURAL_PLASTICITY_BLEND_ATOM_REF:
        target = matching_state_operand(effect.operands[0], name="blend-target")
        amount = matching_state_operand(effect.operands[1], name="blend-amount")
        return state + amount * (target - state)
    if effect.atom_ref in {
        NEURAL_PLASTICITY_OUTER_ATOM_REF,
        NEURAL_PLASTICITY_OUTER_V2_ATOM_REF,
    }:
        left, right, rate = effect.operands[:3]
        execution_count = None if len(effect.operands) == 3 else effect.operands[3]
        if (
            left.ndim != 1
            or right.ndim != 1
            or rate.ndim != 0
            or left.dtype != state.dtype
            or right.dtype != state.dtype
            or rate.dtype != state.dtype
            or left.device != state.device
            or right.device != state.device
            or rate.device != state.device
            or (
                execution_count is not None
                and (
                    execution_count.ndim != 0
                    or execution_count.dtype != state.dtype
                    or execution_count.device != state.device
                )
            )
        ):
            raise FormulaProgramError(
                "FF_EFFECT_OUTER_MISMATCH",
                "outer operands must match execution-site dtype and device",
            )
        update = rate * left.unsqueeze(-1) * right.unsqueeze(-2)
        if update.shape != state.shape:
            raise FormulaProgramError(
                "FF_EFFECT_OUTER_MISMATCH",
                "outer product must exactly match execution-site state",
            )
        if effect.atom_ref == NEURAL_PLASTICITY_OUTER_V2_ATOM_REF:
            max_executions = dict(effect.attributes).get("max_executions")
            if (
                isinstance(max_executions, bool)
                or not isinstance(max_executions, int)
                or max_executions <= 0
            ):
                raise FormulaProgramError(
                    "FF_EFFECT_EXECUTION_COUNT",
                    "Outer@2 max_executions must be a positive integer",
                )
            assert execution_count is not None
            bounded_count = execution_count.clamp(0.0, float(max_executions))
            hard_count = bounded_count.round()
            applied_count = bounded_count + (hard_count - bounded_count).detach()
            return state + applied_count * update
        return state + update
    if effect.atom_ref == NEURAL_PLASTICITY_TRANSPORT_ATOM_REF:
        bias = matching_state_operand(effect.operands[0], name="transport-bias")
        output_factor, input_factor, rate = effect.operands[1:]
        state_axis = dict(effect.attributes)["state_axis"]
        state_axis_index = matching_low_rank_factors(
            (output_factor, input_factor),
            rate=rate,
            state_axis=state_axis,
            effect_name="transport",
        )
        axis_last = state.movedim(state_axis_index, -1)
        transported = (axis_last @ input_factor) @ output_factor.transpose(0, 1)
        return state + bias + rate * transported.movedim(-1, state_axis_index)
    if effect.atom_ref == NEURAL_PLASTICITY_POLYNOMIAL_ATOM_REF:
        bias = matching_state_operand(effect.operands[0], name="polynomial-bias")
        output_factor, left_factor, right_factor, rate = effect.operands[1:]
        state_axis = dict(effect.attributes)["state_axis"]
        state_axis_index = matching_low_rank_factors(
            (output_factor, left_factor, right_factor),
            rate=rate,
            state_axis=state_axis,
            effect_name="polynomial",
        )
        axis_last = state.movedim(state_axis_index, -1)
        left_projection = axis_last @ left_factor
        right_projection = axis_last @ right_factor
        feedback = (left_projection * right_projection) @ output_factor.transpose(0, 1)
        return state + bias + rate * feedback.movedim(-1, state_axis_index)
    if effect.atom_ref == NEURAL_PLASTICITY_PROXIMAL_ATOM_REF:
        bias = matching_state_operand(effect.operands[0], name="proximal-bias")
        raw_strength = matching_state_operand(
            effect.operands[1], name="proximal-raw-strength"
        )
        candidate = state + bias
        threshold = torch.nn.functional.softplus(raw_strength)
        return torch.sign(candidate) * torch.relu(torch.abs(candidate) - threshold)
    raise AssertionError("unreachable NeuralPlasticity effect kind")


class NeuralPlasticityAtom(nn.Module):
    """Component identity for the execution-site-bound Formula effect atom."""

    _component_reference: ClassVar[str] = NEURAL_PLASTICITY_ATOM_REF

    def forward(self, *args: object, **kwargs: object) -> Tensor:
        raise RuntimeError(
            "NeuralPlasticity@1 can only execute inside FormulaFabric@3"
        )


class NeuralPlasticityBlendAtom(nn.Module):
    """Component identity for target-directed execution-site plasticity."""

    _component_reference: ClassVar[str] = NEURAL_PLASTICITY_BLEND_ATOM_REF

    def forward(self, *args: object, **kwargs: object) -> Tensor:
        raise RuntimeError(
            "NeuralPlasticityBlend@1 can only execute inside FormulaFabric@4"
        )


class NeuralPlasticityOuterAtom(nn.Module):
    """Component identity for rank-one execution-site plasticity."""

    _component_reference: ClassVar[str] = NEURAL_PLASTICITY_OUTER_ATOM_REF

    def forward(self, *args: object, **kwargs: object) -> Tensor:
        raise RuntimeError(
            "NeuralPlasticityOuter@1 can only execute inside FormulaFabric@4"
        )


class NeuralPlasticityOuterAtomV2(nn.Module):
    """Rank-one self effect with a direct trainable execution-count operand."""

    _component_reference: ClassVar[str] = NEURAL_PLASTICITY_OUTER_V2_ATOM_REF

    def forward(self, *args: object, **kwargs: object) -> Tensor:
        raise RuntimeError(
            "NeuralPlasticityOuter@2 can only execute inside FormulaFabric@4"
        )


class NeuralPlasticityTransportAtom(nn.Module):
    """Component identity for low-rank cross-coordinate state transport."""

    _component_reference: ClassVar[str] = NEURAL_PLASTICITY_TRANSPORT_ATOM_REF

    def forward(self, *args: object, **kwargs: object) -> Tensor:
        raise RuntimeError(
            "NeuralPlasticityTransport@1 can only execute inside FormulaFabric@4"
        )


class NeuralPlasticityPolynomialAtom(nn.Module):
    """Component identity for quadratic implicit-state feedback."""

    _component_reference: ClassVar[str] = NEURAL_PLASTICITY_POLYNOMIAL_ATOM_REF

    def forward(self, *args: object, **kwargs: object) -> Tensor:
        raise RuntimeError(
            "NeuralPlasticityPolynomial@1 can only execute inside FormulaFabric@4"
        )


class NeuralPlasticityProximalAtom(nn.Module):
    """Component identity for L1 proximal execution-site plasticity."""

    _component_reference: ClassVar[str] = NEURAL_PLASTICITY_PROXIMAL_ATOM_REF

    def forward(self, *args: object, **kwargs: object) -> Tensor:
        raise RuntimeError(
            "NeuralPlasticityProximal@1 can only execute inside FormulaFabric@4"
        )


@dataclass(frozen=True)
class FormulaFabricV3Result:
    """Formula data output plus site-owned NeuralPlasticity operands."""

    value: Tensor
    effect: NeuralPlasticityEffect
    trace: FormulaTraceV2 | None = None

class FormulaFabricV3(nn.Module):
    """Run typed SSA and its NeuralPlasticity atom in one Formula dispatcher."""

    _component_reference: ClassVar[str] = "arti/formula-fabric@3"

    def __init__(self, effect_program: FormulaEffectProgram) -> None:
        super().__init__()
        if not isinstance(effect_program, FormulaEffectProgram):
            raise TypeError("FormulaFabricV3 requires FormulaEffectProgram")
        self.effect_program = effect_program
        object.__setattr__(
            self,
            "_executor",
            FormulaFabricV2(effect_program.program, _allow_effects=True),
        )

    def _execute_owned(
        self,
        *,
        inputs: Mapping[str, Tensor],
        banks: Mapping[str, FormulaBankOperand],
        return_trace: bool = False,
    ) -> FormulaFabricV3Result:
        effects: list[NeuralPlasticityEffect] = []

        def capture(
            instruction: FormulaInstructionV2,
            updates: tuple[Tensor, Tensor],
        ) -> None:
            effects.append(
                NeuralPlasticityEffect(instruction.instruction_id, updates[0], updates[1])
            )

        result = self._executor._execute(
            inputs=inputs,
            banks=banks,
            return_trace=return_trace,
            effect_sink=capture,
        )
        if len(effects) != 1:
            raise FormulaProgramError(
                "FF3_EFFECT_COUNT", "FormulaFabric@3 must execute one NeuralPlasticity effect"
            )
        return FormulaFabricV3Result(result.values[0], effects[0], result.trace)

    def forward(self, *args: object, **kwargs: object) -> FormulaFabricV3Result:
        raise RuntimeError(
            "FormulaFabric@3 is execution-site-bound and can only run through a Bank-local action"
        )

    def contract_config(self) -> dict[str, object]:
        return {
            "effect_program": self.effect_program.to_dict(),
            "effect_program_fingerprint": self.effect_program.fingerprint,
            "data_lane": "identity",
            "target_binding": "runtime-predecessor-bank",
            "state_access": "effect-operands-only",
            "state_transition": "additive-plus-state-scaled",
            "state_visibility": "next-dispatch",
            "execution_mode": "eager",
        }


@dataclass(frozen=True)
class FormulaFabricV4Result:
    """Formula data output plus one versioned self-network effect."""

    value: Tensor
    effect: NeuralPlasticityEffectV2
    trace: FormulaTraceV2 | None = None


class FormulaFabricV4(nn.Module):
    """Execute the extensible NeuralPlasticity effect algebra at an owned site."""

    _component_reference: ClassVar[str] = "arti/formula-fabric@4"

    def __init__(self, effect_program: FormulaEffectProgramV2) -> None:
        super().__init__()
        if not isinstance(effect_program, FormulaEffectProgramV2):
            raise TypeError("FormulaFabricV4 requires FormulaEffectProgramV2")
        self.effect_program = effect_program
        object.__setattr__(
            self,
            "_executor",
            FormulaFabricV2(effect_program.program, _allow_effects=True),
        )

    def _execute_owned(
        self,
        *,
        inputs: Mapping[str, Tensor],
        banks: Mapping[str, FormulaBankOperand],
        return_trace: bool = False,
    ) -> FormulaFabricV4Result:
        effects: list[NeuralPlasticityEffectV2] = []

        def capture(
            instruction: FormulaInstructionV2,
            operands: tuple[Tensor, ...],
        ) -> None:
            effects.append(
                NeuralPlasticityEffectV2(
                    instruction.instruction_id,
                    instruction.atom_ref,
                    operands,
                    instruction.attributes,
                )
            )

        result = self._executor._execute(
            inputs=inputs,
            banks=banks,
            return_trace=return_trace,
            effect_sink=capture,
        )
        if len(effects) != 1:
            raise FormulaProgramError(
                "FF4_EFFECT_COUNT", "FormulaFabric@4 must execute one NeuralPlasticity effect"
            )
        return FormulaFabricV4Result(result.values[0], effects[0], result.trace)

    def forward(self, *args: object, **kwargs: object) -> FormulaFabricV4Result:
        raise RuntimeError(
            "FormulaFabric@4 is execution-site-bound and can only run through a Bank-local action"
        )

    def contract_config(self) -> dict[str, object]:
        return {
            "effect_program": self.effect_program.to_dict(),
            "effect_program_fingerprint": self.effect_program.fingerprint,
            "effect_atom_ref": self.effect_program.effect_instruction.atom_ref,
            "data_lane": "identity",
            "target_binding": "runtime-predecessor-bank",
            "state_access": "effect-operands-only",
            "state_visibility": "next-dispatch",
            "execution_mode": "eager",
        }


@dataclass(frozen=True)
class FormulaFabricV5Result:
    """Ordinary Formula output plus its ordered intermediate self-effects."""

    value: Tensor
    effects: tuple[NeuralPlasticityEffectV2, ...]
    trace: FormulaTraceV2 | None = None


class FormulaFabricV5(nn.Module):
    """Execute self-network effects inside one ordinary Formula instruction path."""

    _component_reference: ClassVar[str] = "arti/formula-fabric@5"

    def __init__(self, effect_program: FormulaEffectProgramV3) -> None:
        super().__init__()
        if not isinstance(effect_program, FormulaEffectProgramV3):
            raise TypeError("FormulaFabricV5 requires FormulaEffectProgramV3")
        self.effect_program = effect_program
        object.__setattr__(
            self,
            "_executor",
            FormulaFabricV2(effect_program.program, _allow_effects=True),
        )

    def _execute_owned(
        self,
        *,
        inputs: Mapping[str, Tensor],
        banks: Mapping[str, FormulaBankOperand],
        return_trace: bool = False,
    ) -> FormulaFabricV5Result:
        effects: list[NeuralPlasticityEffectV2] = []

        def capture(
            instruction: FormulaInstructionV2,
            operands: tuple[Tensor, ...],
        ) -> None:
            effects.append(
                NeuralPlasticityEffectV2(
                    instruction.instruction_id,
                    instruction.atom_ref,
                    operands,
                    instruction.attributes,
                )
            )

        result = self._executor._execute(
            inputs=inputs,
            banks=banks,
            return_trace=return_trace,
            effect_sink=capture,
        )
        if len(effects) != len(self.effect_program.effect_instructions):
            raise FormulaProgramError(
                "FF5_EFFECT_COUNT", "FormulaFabric@5 did not execute every NeuralPlasticity effect"
            )
        return FormulaFabricV5Result(result.values[0], tuple(effects), result.trace)

    def forward(self, *args: object, **kwargs: object) -> FormulaFabricV5Result:
        raise RuntimeError(
            "FormulaFabric@5 is execution-site-bound and can only run through a Bank-local action"
        )

    def contract_config(self) -> dict[str, object]:
        return {
            "effect_program": self.effect_program.to_dict(),
            "effect_program_fingerprint": self.effect_program.fingerprint,
            "effect_atom_refs": [
                effect.atom_ref for effect in self.effect_program.effect_instructions
            ],
            "effect_count": len(self.effect_program.effect_instructions),
            "data_lane": "ordinary-formula-with-intermediate-effects",
            "effect_position": "intermediate",
            "target_binding": "runtime-predecessor-bank",
            "state_access": "effect-operands-only",
            "state_visibility": "next-dispatch",
            "execution_mode": "eager",
        }


__all__ = [
    "FORMULA_EFFECT_PROGRAM_V1_SCHEMA_REF",
    "FORMULA_EFFECT_PROGRAM_V1_SCHEMA_VERSION",
    "FORMULA_EFFECT_PROGRAM_V2_SCHEMA_REF",
    "FORMULA_EFFECT_PROGRAM_V2_SCHEMA_VERSION",
    "FORMULA_EFFECT_PROGRAM_V3_SCHEMA_REF",
    "FORMULA_EFFECT_PROGRAM_V3_SCHEMA_VERSION",
    "NEURAL_PLASTICITY_ATOM_REF",
    "NEURAL_PLASTICITY_BLEND_ATOM_REF",
    "NEURAL_PLASTICITY_EFFECT_REFS",
    "NEURAL_PLASTICITY_OUTER_ATOM_REF",
    "NEURAL_PLASTICITY_OUTER_V2_ATOM_REF",
    "NEURAL_PLASTICITY_POLYNOMIAL_ATOM_REF",
    "NEURAL_PLASTICITY_PROXIMAL_ATOM_REF",
    "NEURAL_PLASTICITY_TRANSPORT_ATOM_REF",
    "FormulaEffectProgram",
    "FormulaEffectProgramV2",
    "FormulaEffectProgramV3",
    "FormulaFabricV3",
    "FormulaFabricV3Result",
    "FormulaFabricV4",
    "FormulaFabricV4Result",
    "FormulaFabricV5",
    "FormulaFabricV5Result",
    "NeuralPlasticityAtom",
    "NeuralPlasticityBlendAtom",
    "NeuralPlasticityEffect",
    "NeuralPlasticityEffectV2",
    "NeuralPlasticityOuterAtom",
    "NeuralPlasticityOuterAtomV2",
    "NeuralPlasticityPolynomialAtom",
    "NeuralPlasticityProximalAtom",
    "NeuralPlasticityTransportAtom",
    "apply_neural_plasticity_effect",
]
