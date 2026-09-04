"""Pure numeric candidate batching; ownership and effects stay in QueryV4."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import Sequence

import torch
from torch import Tensor
from torch.nn.modules import module as module_runtime

from .formula_program_query_v4 import (
    FormulaProgramEffectCandidateV3,
    FormulaProgramSearchCandidateV4,
    FormulaProgramTensorCandidateV3,
    _FormulaProgramExecutionArenaV4,
)
from .formula_v2 import (
    FormulaExecutionPlanV2,
    FormulaBindingError,
    FormulaFabricV2,
    FormulaProgram,
    PreparedFormulaBindings,
    _validate_instruction_dtypes,
    _validate_tensor_against_type,
    formula_program_effect_refs,
)
from .formula_v3 import FormulaFabricV4, NeuralPlasticityEffectV2


Request = tuple[FormulaProgramSearchCandidateV4, _FormulaProgramExecutionArenaV4]
_PURE_ATOMS = frozenset(
    f"arti/formula-atom-{name}@1"
    for name in (
        "contract", "scale", "add", "reduce", "reshape", "permute",
        "scalar-map", "broadcast", "concat",
    )
)


@lru_cache(maxsize=64)
def _checked_plan(program: FormulaProgram) -> FormulaExecutionPlanV2 | None:
    """Lower only pure numerical work through the existing checked plan."""
    effects = frozenset(formula_program_effect_refs(program))
    effect_outputs = {
        instruction.output_slot
        for instruction in program.instructions
        if instruction.atom_ref in effects
    }
    numeric = tuple(
        instruction for instruction in program.instructions
        if instruction.atom_ref not in effects
    )
    if not numeric or any(
        instruction.atom_ref not in _PURE_ATOMS
        or effect_outputs.intersection(instruction.input_slots)
        for instruction in numeric
    ):
        return None
    if not effects:
        return FormulaExecutionPlanV2(program)
    binding_names = {binding.name for binding in program.bindings}
    outputs = tuple(dict.fromkeys(
        name
        for instruction in program.instructions if instruction.atom_ref in effects
        for name in instruction.input_slots[1:] if name not in binding_names
    ))
    if not outputs:
        return None
    checked = replace(
        program,
        instructions=numeric,
        slots=tuple(
            replace(slot, role="output" if slot.slot_id in outputs else "temporary")
            if slot.producer == "instruction" else slot
            for slot in program.slots if slot.slot_id not in effect_outputs
        ),
        outputs=outputs,
    )
    return FormulaExecutionPlanV2(checked)


def _program(candidate: FormulaProgramSearchCandidateV4) -> FormulaProgram:
    if isinstance(candidate, FormulaProgramEffectCandidateV3):
        return candidate.effect_program.program
    return candidate.candidate.program


def _prepare(
    candidate: FormulaProgramSearchCandidateV4,
    arena: _FormulaProgramExecutionArenaV4,
) -> PreparedFormulaBindings | None:
    inputs, banks = candidate._bindings(arena)
    values = (*inputs.values(), *(operand.value for operand in banks.values()))
    if any(
        value.layout != torch.strided or not value.is_floating_point()
        or value.device.type not in {"cpu", "cuda"}
        for value in values
    ):
        return None
    if isinstance(candidate, FormulaProgramEffectCandidateV3):
        _lineage, previous, _revision = candidate._target(arena)
        _validate_tensor_against_type(
            previous, candidate.effect_program.state_type,
            name=f"{candidate.candidate_id}.predecessor_bank_slot",
        )
        return candidate.fabric._executor._bind_for_checked_execution(inputs=inputs, banks=banks)
    return candidate.candidate.fabric._bind_for_checked_execution(inputs=inputs, banks=banks)


def _native_only(candidate: FormulaProgramSearchCandidateV4) -> bool:
    """Custom candidate/fabric behavior is not a numerical-plan optimization."""
    if (
        module_runtime._global_forward_hooks or module_runtime._global_forward_pre_hooks
        or module_runtime._global_backward_hooks or module_runtime._global_backward_pre_hooks
    ):
        return True
    if type(candidate) not in {FormulaProgramTensorCandidateV3, FormulaProgramEffectCandidateV3}:
        return True
    if any(name in candidate.__dict__ for name in (
        "forward", "_bindings", "_target", "_proposal", "_finish",
    )):
        return True
    fabric = (
        candidate.fabric if isinstance(candidate, FormulaProgramEffectCandidateV3)
        else candidate.candidate.fabric
    )
    expected = FormulaFabricV4 if isinstance(candidate, FormulaProgramEffectCandidateV3) else FormulaFabricV2
    if type(fabric) is not expected or any(
        name in fabric.__dict__ for name in ("forward", "_execute", "_execute_owned")
    ):
        return True
    if isinstance(candidate, FormulaProgramEffectCandidateV3):
        executor = fabric._executor
        if type(executor) is not FormulaFabricV2 or "_execute" in executor.__dict__:
            return True
    return any(
        module._forward_hooks or module._forward_pre_hooks
        or module._backward_hooks or module._backward_pre_hooks
        for module in (candidate, fabric)
    )


def _run_group(
    requests: Sequence[Request],
    rows: Sequence[tuple[int, PreparedFormulaBindings]],
    plan: FormulaExecutionPlanV2,
) -> tuple[_FormulaProgramExecutionArenaV4, ...]:
    # Chunking happens before stack, so it also bounds extra input copies.
    stacked = tuple(
        torch.stack(tuple(prepared.values[index] for _, prepared in rows))
        for index in range(len(plan.binding_names))
    )

    def run(*values: Tensor) -> tuple[tuple[Tensor, ...], Tensor]:
        return plan.forward_checked(PreparedFormulaBindings(
            plan.program_fingerprint, plan.binding_names, values
        ))

    # Unsupported programs were excluded before this call. Do not catch real
    # numerical, device, or allocation errors and disguise them as fallback.
    outputs, finite = torch.vmap(run, randomness="error")(*stacked)
    if not bool(finite.all()):
        raise FormulaBindingError(
            "FF2_NONFINITE", "batched Formula intermediate values must be finite"
        )
    numeric = dict(zip(plan.program.outputs, outputs, strict=True))
    return tuple(
        _finish(
            requests[request_index], prepared,
            {name: value[row_index] for name, value in numeric.items()},
        )
        for row_index, (request_index, prepared) in enumerate(rows)
    )


def _run_single(
    prepared: PreparedFormulaBindings, plan: FormulaExecutionPlanV2,
) -> tuple[tuple[Tensor, ...], Tensor]:
    return plan.forward_checked(PreparedFormulaBindings(
        plan.program_fingerprint, plan.binding_names, prepared.values
    ))


def _run_independent(
    requests: Sequence[Request],
    rows: Sequence[tuple[int, PreparedFormulaBindings]],
    plan: FormulaExecutionPlanV2,
) -> tuple[_FormulaProgramExecutionArenaV4, ...]:
    # Never stack graph-carrying values across requests: a loss consuming one
    # result must leave the other request's exclusive parameters at grad=None.
    executed = tuple(_run_single(prepared, plan) for _, prepared in rows)
    finite = torch.stack(tuple(flag for _, flag in executed))
    if not bool(finite.all()):
        raise FormulaBindingError(
            "FF2_NONFINITE", "Formula intermediate values must be finite"
        )
    return tuple(
        _finish(
            requests[index], prepared,
            dict(zip(plan.program.outputs, outputs, strict=True)),
        )
        for (index, prepared), (outputs, _) in zip(rows, executed, strict=True)
    )


def _finish(
    request: Request, prepared: PreparedFormulaBindings, numeric: dict[str, Tensor],
) -> _FormulaProgramExecutionArenaV4:
    candidate, arena = request
    if isinstance(candidate, FormulaProgramEffectCandidateV3):
        instruction = candidate.effect_program.effect_instruction
        originals = dict(zip(prepared.binding_names, prepared.values, strict=True))
        effect = NeuralPlasticityEffectV2(
            instruction.instruction_id, instruction.atom_ref,
            tuple(
                originals[name] if name in originals else numeric[name]
                for name in instruction.input_slots[1:]
            ),
            instruction.attributes,
        )
        value = originals[candidate.effect_program.data_input_name]
        if value is not arena.values.get(candidate.input_slot):
            raise RuntimeError("NeuralPlasticity data lane must preserve Tensor identity")
        _validate_instruction_dtypes(
            instruction, (value, *effect.operands),
            tuple(candidate.effect_program.program.slot_types[name]
                  for name in instruction.input_slots),
        )
        return candidate._finish(arena, value, effect)
    return candidate._finish(arena, numeric[candidate.candidate.program.outputs[0]])


def execute_many(
    requests: Sequence[Request], *, chunk_size: int | None = None, serial: bool = False
) -> tuple[_FormulaProgramExecutionArenaV4, ...]:
    if chunk_size is not None and (
        isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer or None")
    if not isinstance(serial, bool):
        raise TypeError("serial must be a bool")
    groups: dict[tuple[object, ...], list[tuple[int, PreparedFormulaBindings]]] = {}
    plans: dict[tuple[object, ...], FormulaExecutionPlanV2] = {}
    native: list[int] = []
    for index, (candidate, arena) in enumerate(requests):
        if arena.values.get(candidate.output_slot) is not None:
            raise ValueError(f"SSA output slot {candidate.output_slot!r} is already occupied")
        if any(arena.values.get(slot) is not None for slot in candidate.requires_empty_slots):
            raise ValueError("candidate requires empty SSA slots")
        if serial or _native_only(candidate):
            native.append(index)
            continue
        program = _program(candidate)
        plan = _checked_plan(program)
        if plan is None:
            native.append(index)
            continue
        prepared = _prepare(candidate, arena)
        if prepared is None:
            native.append(index)
            continue
        # Metadata/provenance/preflight are unchanged. Input finiteness is
        # deferred only here, where every admitted row must run forward_checked.
        if prepared.binding_names != plan.binding_names:
            raise RuntimeError("checked plan must preserve original binding order")
        key = (
            program.fingerprint,
            tuple((tuple(value.shape), value.dtype, value.device) for value in prepared.values),
        )
        groups.setdefault(key, []).append((index, prepared))
        plans[key] = plan
    results: list[_FormulaProgramExecutionArenaV4 | None] = [None] * len(requests)
    for key, group in groups.items():
        width = len(group) if chunk_size is None else chunk_size
        for start in range(0, len(group), width):
            rows = group[start:start + width]
            run = _run_independent if torch.is_grad_enabled() or len(rows) == 1 else _run_group
            for (index, _), result in zip(
                rows, run(requests, rows, plans[key]), strict=True
            ):
                results[index] = result
    for index in sorted(native):
        candidate, arena = requests[index]
        results[index] = candidate(arena)
    assert all(result is not None for result in results)
    return tuple(results)  # type: ignore[return-value]
