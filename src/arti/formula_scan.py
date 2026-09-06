"""Ordered shared-body Scan lowered into the existing typed Formula IR."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
from typing import ClassVar, Mapping

from torch import nn

from .formula_v2 import (
    DEFAULT_FORMULA_LIMITS, FormulaBindingError, FormulaFabricV2,
    FormulaInstructionV2, FormulaLimits, FormulaProgram, FormulaProgramError,
    FormulaSchemaError, FormulaSlotSpec, FormulaTypeError, InputBinding, TensorType,
    _infer_instruction_output_type, _validate_name, _validate_tensor_metadata_against_type,
    concat, formula_program_effect_refs, permute, reshape, slice_tensor,
)


@dataclass(frozen=True)
class _ScanDefinition:
    body: FormulaProgram
    axis: str
    sequence_types: tuple[tuple[str, TensorType], ...]
    carry_outputs: tuple[tuple[str, str], ...]
    emissions: tuple[tuple[str, str], ...]
    max_length: int


class FormulaScan(nn.Module):
    """A pure body with explicit sequence inputs, simultaneous carry and emissions.

    ``carry_outputs`` maps body input names to body output slots. ``emissions``
    names independently stacked body outputs. The new sequence axis is leading
    on every emitted output; input sequences may place it anywhere.
    """

    _component_reference: ClassVar[str] = "arti/formula-scan@1"
    body: FormulaProgram
    axis: str
    sequence_types: tuple[tuple[str, TensorType], ...]
    carry_outputs: tuple[tuple[str, str], ...]
    emissions: tuple[tuple[str, str], ...]
    max_length: int

    def __init__(self, body: FormulaProgram, *, axis: str,
                 sequence_types: Mapping[str, TensorType], carry_outputs: Mapping[str, str],
                 emissions: Mapping[str, str], max_length: int):
        super().__init__()
        if not isinstance(body, FormulaProgram):
            raise TypeError("Scan body must be a FormulaProgram")
        if formula_program_effect_refs(body):
            raise FormulaTypeError("FF2_SCAN_EFFECT", "Scan@1 body is pure; carry is an explicit return, not a Bank effect")
        if type(max_length) is not int or max_length <= 0:
            raise FormulaTypeError("FF2_SCAN_LENGTH", "max_length must be positive")
        sequences, carry, emits = tuple(sequence_types.items()), tuple(carry_outputs.items()), tuple(emissions.items())
        if not sequences or not carry or not emits:
            raise FormulaTypeError("FF2_SCAN_PORTS", "Scan requires sequence, carry and emission ports")
        if set(sequence_types) & set(carry_outputs):
            raise FormulaTypeError("FF2_SCAN_PORTS", "sequence and carry input names must be distinct")
        inputs = {b.name: b.value_type for b in body.bindings if isinstance(b, InputBinding)}
        for name, value_type in sequences:
            if name not in inputs or not isinstance(value_type, TensorType) or axis not in value_type.axis_names:
                raise FormulaTypeError("FF2_SCAN_PORTS", "sequence ports must name body inputs and contain the sequence axis")
            sliced = TensorType(
                tuple(a for a in value_type.axis_names if a != axis),
                tuple(s for a, s in zip(value_type.axis_names, value_type.sizes) if a != axis),
                dtype=value_type.dtype, domain=value_type.domain,
            )
            if sliced != inputs[name]:
                raise FormulaTypeError("FF2_SCAN_TYPE", "removing the sequence axis must give the exact body input type")
        for name, output in carry:
            if name not in inputs or output not in body.outputs or inputs[name] != body.slot_types[output]:
                raise FormulaTypeError("FF2_SCAN_CARRY", "carry output must match its input shape, dtype and domain")
        for name, output in emits:
            _validate_name(name, field="Scan emission")
            if output not in body.outputs or axis in body.slot_types[output].axis_names:
                raise FormulaTypeError("FF2_SCAN_OUTPUT", "emissions must name body outputs without the sequence axis")
        if set(carry_outputs.values()) | set(emissions.values()) != set(body.outputs):
            raise FormulaTypeError("FF2_SCAN_OUTPUT", "every body output must have a carry or emission role")
        for name, value in (("body", body), ("axis", axis), ("sequence_types", sequences),
                            ("carry_outputs", carry), ("emissions", emits), ("max_length", max_length)):
            setattr(self, name, value)

    def _definition(self):
        return _ScanDefinition(self.body, self.axis, self.sequence_types, self.carry_outputs, self.emissions, self.max_length)

    @property
    def output_names(self):
        return tuple("carry." + name for name, _ in self.carry_outputs) + tuple("emit." + name for name, _ in self.emissions)

    def to_dict(self):
        return {"schema_ref": self._component_reference, "body": self.body.to_dict(), "axis": self.axis,
                "sequence_types": {name: t.to_dict() for name, t in self.sequence_types},
                "carry_outputs": dict(self.carry_outputs), "emissions": dict(self.emissions), "max_length": self.max_length}

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, Mapping) or set(payload) != {
            "schema_ref", "body", "axis", "sequence_types", "carry_outputs", "emissions", "max_length",
        } or payload["schema_ref"] != cls._component_reference:
            raise FormulaSchemaError("FF2_SCAN_SCHEMA", "invalid Scan@1 payload")
        return cls(FormulaProgram.from_dict(payload["body"]), axis=payload["axis"],
                   sequence_types={n: TensorType.from_dict(t) for n, t in payload["sequence_types"].items()},
                   carry_outputs=payload["carry_outputs"], emissions=payload["emissions"], max_length=payload["max_length"])

    def lower(self, length: int, *, limits: FormulaLimits = DEFAULT_FORMULA_LIMITS) -> FormulaProgram:
        """Specialize sequence length on the host; return ordinary finite SSA."""
        return _lower_scan(self._definition(), length, limits)

    def prepare(self, *, inputs, banks, limits: FormulaLimits = DEFAULT_FORMULA_LIMITS):
        length = None
        for name, value_type in self.sequence_types:
            value = inputs[name]
            _validate_tensor_metadata_against_type(value, value_type, name=name)
            current = value.shape[value_type.axis_names.index(self.axis)]
            if length is not None and current != length:
                raise FormulaBindingError("FF2_SCAN_LENGTH", "sequence lengths must match")
            length = int(current)
        plan = _scan_plan(self._definition(), length, limits)
        prepared = FormulaFabricV2(plan.program).bind_tensors(inputs=inputs, banks=banks)
        return plan, prepared

    def forward(self, *, inputs, banks, limits: FormulaLimits = DEFAULT_FORMULA_LIMITS):
        plan, prepared = self.prepare(inputs=inputs, banks=banks, limits=limits)
        return dict(zip(self.output_names, plan(prepared), strict=True))


def _specialize(value_type, extents):
    return replace(value_type, sizes=tuple(extents.get(s, s) if isinstance(s, str) else s for s in value_type.sizes))


def _specialize_attributes(attributes, extents):
    attrs = dict(attributes)
    if "output_sizes" in attrs:
        attrs["output_sizes"] = tuple(extents.get(s, s) if isinstance(s, str) else s for s in attrs["output_sizes"])
    if isinstance(attrs.get("output_size"), str):
        attrs["output_size"] = extents.get(attrs["output_size"], attrs["output_size"])
    return tuple(attrs.items())


@lru_cache(maxsize=64)
def _lower_scan(scan: _ScanDefinition, length: int, limits: FormulaLimits):
    if type(length) is not int or not 1 <= length <= scan.max_length:
        raise FormulaTypeError("FF2_SCAN_LENGTH", "length must be within the positive Scan horizon")
    if not isinstance(limits, FormulaLimits):
        raise TypeError("Scan lowering limits must be FormulaLimits")
    # Count every body occurrence and balanced-concat node before constructing IR.
    count = length * (len(scan.body.instructions) + 2 * len(scan.sequence_types))
    count += (2 * length - 1) * len(scan.emissions) + len(scan.carry_outputs)
    if count > limits.max_instructions or count + len(scan.body.bindings) > limits.max_slots:
        raise FormulaProgramError("FF2_LIMIT_EXCEEDED", "expanded Scan instruction/slot count exceeds Formula limits")
    sequences = dict(scan.sequence_types)
    extents = {}
    for t in sequences.values():
        declared = t.size_for(scan.axis)
        if isinstance(declared, int) and declared != length:
            raise FormulaTypeError("FF2_SCAN_LENGTH", "specialized length conflicts with sequence declaration")
        if isinstance(declared, str):
            extents[declared] = length
    bindings = tuple(replace(b, value_type=_specialize(sequences.get(b.name, b.value_type), extents)) for b in scan.body.bindings)
    types = {b.name: b.value_type for b in bindings}
    slots = [FormulaSlotSpec(b.name, b.value_type, "input" if isinstance(b, InputBinding) else "bank",
                            b.name, "input" if isinstance(b, InputBinding) else "bank_operand") for b in bindings]
    depths = {b.name: 0 for b in bindings}
    instructions = []

    def emit(reference, operands, attributes):
        index = len(instructions)
        output = f"%{index}"
        depth = 1 + max(depths[n] for n in operands)
        if depth > limits.max_steps:
            raise FormulaProgramError("FF2_LIMIT_EXCEEDED", "expanded Scan dependency depth exceeds Formula limits")
        instruction = FormulaInstructionV2(f"i{index}", depth, reference, tuple(operands), output, tuple(attributes))
        value_type = _infer_instruction_output_type(instruction, tuple(types[n] for n in operands))
        instructions.append(instruction)
        slots.append(FormulaSlotSpec(output, value_type, "instruction", instruction.instruction_id, "temporary"))
        types[output], depths[output] = value_type, depth
        return output

    def unary(builder, name, **kwargs):
        expression = builder(InputBinding("value", types[name]), **kwargs)
        return emit(expression.atom_ref, (name,), expression.attributes)

    current = {name: name for name, _ in scan.carry_outputs}
    emissions = [[] for _ in scan.emissions]
    for time in range(length):
        frame = {b.name: b.name for b in bindings}
        frame.update(current)
        for name, seq_type in scan.sequence_types:
            sliced = unary(slice_tensor, name, axis=scan.axis, start=time, stop=time + 1)
            cell_type = _specialize(next(b.value_type for b in scan.body.bindings if b.name == name), extents)
            frame[name] = unary(reshape, sliced, output_axes=cell_type.axis_names, output_sizes=cell_type.sizes)
        for instruction in scan.body.instructions:
            frame[instruction.output_slot] = emit(instruction.atom_ref,
                tuple(frame[n] for n in instruction.input_slots), _specialize_attributes(instruction.attributes, extents))
        # All carry ports advance from the same completed body invocation.
        current = {name: frame[output] for name, output in scan.carry_outputs}
        for (_, output), values in zip(scan.emissions, emissions, strict=True):
            name = frame[output]
            values.append(unary(reshape, name, output_axes=(scan.axis, *types[name].axis_names),
                                output_sizes=(1, *types[name].sizes)))
    outputs = [unary(permute, name, output_axes=types[name].axis_names) for name in current.values()]
    for values in emissions:
        while len(values) > 1:
            combined = []
            for i in range(0, len(values), 2):
                if i + 1 == len(values):
                    combined.append(values[i])
                else:
                    a, b = values[i:i + 2]
                    expression = concat(InputBinding("left", types[a]), InputBinding("right", types[b]), axis=scan.axis)
                    combined.append(emit(expression.atom_ref, (a, b), expression.attributes))
            values = combined
        outputs.append(values[0])
    slots = tuple(replace(s, role="output") if s.slot_id in outputs else s for s in slots)
    instructions.sort(key=lambda i: (i.step, int(i.instruction_id[1:])))
    return FormulaProgram(bindings, slots, tuple(instructions), tuple(outputs), limits)


@lru_cache(maxsize=64)
def _scan_plan(scan, length, limits):
    return FormulaFabricV2(_lower_scan(scan, length, limits)).execution_plan()


__all__ = ["FormulaScan"]
