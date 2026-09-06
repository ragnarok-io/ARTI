"""Host-side typed wiring expansion into existing shared candidate occurrences."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import product
from math import prod

from .formula_program_query import _require_name
from .formula_program_query_v5 import FormulaProgramTensorCandidateV4
from .formula_v2 import InputBinding, TensorType


def expand_candidate_bindings(
    template: FormulaProgramTensorCandidateV4, *,
    slot_types: Mapping[str, TensorType],
    input_choices: Mapping[str, Sequence[str]],
    output_choices: Mapping[str, Sequence[str]],
    prefix: str | None = None,
    max_candidates: int = 4096,
    requires_empty_slots: Sequence[str] | None = None,
) -> tuple[FormulaProgramTensorCandidateV4, ...]:
    """Enumerate legal named connections without cloning a template's Bank.

    Choices name existing/planned SSA slots with exact declared TensorTypes.
    Every output port, including continuation responses, must be bound. This
    prepares a finite catalog, not a new Query or a runtime grammar interpreter.
    Concrete tensor admission still belongs to the ordinary executor. Omitted
    empty-slot guards are inherited; an explicit sequence replaces them.
    """
    if not isinstance(template, FormulaProgramTensorCandidateV4):
        raise TypeError("binding expansion requires TensorCandidate@4")
    if type(max_candidates) is not int or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer")
    prefix = template.candidate_id if prefix is None else prefix
    _require_name(prefix, field="candidate prefix")
    for name, value_type in slot_types.items():
        _require_name(name, field="SSA slot")
        if not isinstance(value_type, TensorType):
            raise TypeError("slot_types must contain existing TensorType contracts")
    empty = tuple(template.requires_empty_slots if requires_empty_slots is None else requires_empty_slots)
    if len(set(empty)) != len(empty) or not set(empty) <= slot_types.keys():
        raise ValueError("required empty slots must name unique declared slots")
    program = template.candidate.program
    input_types = {b.name: b.value_type for b in program.bindings if isinstance(b, InputBinding)}
    output_types = {slot: program.slot_types[slot] for slot in program.outputs}

    def choices(mapping, expected):
        if not isinstance(mapping, Mapping) or set(mapping) != set(expected):
            raise ValueError("choices must bind every named input/output port exactly once")
        result = []
        for port in sorted(expected):
            values = mapping[port]
            if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
                raise TypeError("each port needs a sequence of SSA choices")
            if not values or len(set(values)) != len(values) or not set(values) <= slot_types.keys():
                raise ValueError("choices must name nonempty unique declared SSA slots")
            compatible = tuple(sorted(v for v in values if slot_types[v] == expected[port]))
            if not compatible:
                raise ValueError(f"no exact-type choices for port {port!r}")
            result.append((port, compatible))
        return tuple(result)

    inputs, outputs = choices(input_choices, input_types), choices(output_choices, output_types)
    ports = (*inputs, *outputs)
    if prod(len(values) for _, values in ports) > max_candidates:
        raise ValueError("typed wiring product exceeds max_candidates before expansion")
    variants = []
    for values in product(*(values for _, values in ports)):
        bound_inputs = dict(zip((p for p, _ in inputs), values[:len(inputs)], strict=True))
        bound_outputs = dict(zip((p for p, _ in outputs), values[len(inputs):], strict=True))
        destinations = tuple(bound_outputs.values())
        if (len(set(destinations)) != len(destinations)
                or set(bound_inputs.values()).intersection((*destinations, *empty))):
            continue
        variants.append(template.with_bindings(
            f"{prefix}.{len(variants)}", input_slots=bound_inputs, output_slots=bound_outputs,
            requires_empty_slots=empty,
        ))
    if not variants:
        raise ValueError("no legal SSA wiring remains after read/write constraints")
    return tuple(variants)


__all__ = ["expand_candidate_bindings"]
