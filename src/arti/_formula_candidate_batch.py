"""Pure numeric candidate batching; ownership and effects stay in QueryV4."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
from typing import Literal, Sequence, overload

import torch
from torch import Tensor
from torch.nn.modules import module as module_runtime

from . import formula_program_query_v4 as query_runtime
from ._formula_finite_rows import _FiniteTensorRows
from ._formula_effect_execution import _EFFECT_BACKEND, _effect_plan
from ._formula_grouped_training import execute_grouped_training, use_grouped_training
from .formula_program_query_v4 import (
    FormulaProgramEffectCandidateV3,
    FormulaProgramSearchCandidateV4,
    FormulaProgramTensorCandidateV3,
    _FormulaProgramExecutionArenaV4,
)
from .formula_program_query_v5 import FormulaProgramTensorCandidateV4
from .formula_v2 import (
    FormulaExecutionPlanV2,
    FormulaBindingError,
    FormulaFabricV2,
    FormulaProgram,
    FormulaProgramError,
    PreparedFormulaBindings,
    _validate_instruction_dtypes,
    _validate_tensor_against_type,
    _capture_finite_validation,
    _OBSERVATION_ATOM_SIGNATURES,
    _INDEX_ATOM_SIGNATURES,
    formula_program_effect_refs,
)
from .formula_v3 import (
    FormulaFabricV4, NeuralPlasticityEffectV2, NEURAL_PLASTICITY_OUTER_V2_ATOM_REF,
    _validate_neural_plasticity_step,
)


Request = tuple[FormulaProgramSearchCandidateV4, _FormulaProgramExecutionArenaV4]
_PURE_ATOMS = frozenset(
    f"arti/formula-atom-{name}@1"
    for name in (
        "contract", "scale", "add", "reduce", "reshape", "permute",
        "scalar-map", "broadcast", "concat", "slice", "select", "lookup",
    )
) | frozenset(_OBSERVATION_ATOM_SIGNATURES) | frozenset(_INDEX_ATOM_SIGNATURES) | frozenset({
    "arti/formula-atom-reduce@2",
    "arti/formula-atom-scalar-map@2",
    "arti/formula-atom-cast@1",
    "arti/formula-atom-window@1",
    "arti/formula-atom-masked-softmax@1",
})
_EFFECT_METHODS = {
    name: getattr(FormulaProgramEffectCandidateV3, name)
    for name in ("_finish", "_target", "_proposal", "execution_count_tensor")
}
_APPLY_EFFECT = query_runtime.apply_neural_plasticity_effect


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
        value.layout != torch.strided or not (value.is_floating_point() or value.dtype in {torch.bool, torch.int64})
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
    if type(candidate) not in {
        FormulaProgramTensorCandidateV3, FormulaProgramTensorCandidateV4, FormulaProgramEffectCandidateV3,
    }:
        return True
    if any(name in candidate.__dict__ for name in (
        "forward", "_bindings", "_target", "_proposal", "_finish", "_finish_outputs",
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
    *, reject_nonfinite: bool = False,
) -> tuple[_FormulaProgramExecutionArenaV4 | None, ...]:
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
    if not reject_nonfinite and not bool(finite.all()):
        raise FormulaBindingError(
            "FF2_NONFINITE", "batched Formula intermediate values must be finite"
        )
    numeric = dict(zip(plan.program.outputs, outputs, strict=True))
    valid_rows = finite.tolist() if reject_nonfinite else [True] * len(rows)
    return _finish_many_checked(tuple(
        (requests[request_index], prepared,
            {name: value[row_index] for name, value in numeric.items()},
        ) if valid_rows[row_index] else None
        for row_index, (request_index, prepared) in enumerate(rows)
    ), reject_nonfinite=reject_nonfinite)


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
    *, reject_nonfinite: bool = False,
) -> tuple[_FormulaProgramExecutionArenaV4 | None, ...]:
    # Never stack graph-carrying values across requests: a loss consuming one
    # result must leave the other request's exclusive parameters at grad=None.
    executed = tuple(_run_single(prepared, plan) for _, prepared in rows)
    finite = torch.stack(tuple(flag for _, flag in executed))
    if not reject_nonfinite and not bool(finite.all()):
        raise FormulaBindingError(
            "FF2_NONFINITE", "Formula intermediate values must be finite"
        )
    valid_rows = finite.tolist() if reject_nonfinite else [True] * len(rows)
    return _finish_many_checked(tuple(
        (requests[index], prepared,
            dict(zip(plan.program.outputs, outputs, strict=True)),
        ) if valid else None
        for (index, prepared), (outputs, _), valid in zip(rows, executed, valid_rows, strict=True)
    ), reject_nonfinite=reject_nonfinite)


def _run_training_group(requests, rows, plan, *, reject_nonfinite=False):
    executed, finite = execute_grouped_training(plan, tuple(prepared for _, prepared in rows))
    validity = finite.tolist()
    if not reject_nonfinite and not all(validity):
        raise FormulaBindingError("FF2_NONFINITE", "Formula intermediate values must be finite")
    return _finish_many_checked(tuple(
        (requests[index], prepared, dict(zip(plan.program.outputs, outputs, strict=True))) if valid else None
        for (index, prepared), outputs, valid in zip(rows, executed, validity, strict=True)
    ), reject_nonfinite=reject_nonfinite)


def _finish(
    request: Request, prepared: PreparedFormulaBindings, numeric: dict[str, Tensor],
) -> _FormulaProgramExecutionArenaV4:
    candidate, arena = request
    if isinstance(candidate, FormulaProgramEffectCandidateV3):
        value, effect = _effect_inputs(request, prepared, numeric)
        return candidate._finish(arena, value, effect)
    if isinstance(candidate, FormulaProgramTensorCandidateV4):
        return candidate._finish_outputs(arena, numeric)
    return candidate._finish(arena, numeric[candidate.candidate.program.outputs[0]])


def _effect_inputs(request, prepared, numeric):
    candidate, arena = request
    instruction = candidate.effect_program.effect_instruction
    originals = dict(zip(prepared.binding_names, prepared.values, strict=True))
    effect = NeuralPlasticityEffectV2(
        instruction.instruction_id, instruction.atom_ref,
        tuple(originals[name] if name in originals else numeric[name] for name in instruction.input_slots[1:]),
        instruction.attributes,
    )
    value = originals[candidate.effect_program.data_input_name]
    if value is not arena.values.get(candidate.input_slot):
        raise RuntimeError("NeuralPlasticity data lane must preserve Tensor identity")
    _validate_instruction_dtypes(
        instruction, (value, *effect.operands),
        tuple(candidate.effect_program.program.slot_types[name] for name in instruction.input_slots),
    )
    return value, effect


def _finish_checked(
    request: Request, prepared: PreparedFormulaBindings, numeric: dict[str, Tensor],
    *, reject_nonfinite: bool,
) -> _FormulaProgramExecutionArenaV4 | None:
    try:
        return _finish(request, prepared, numeric)
    except FormulaBindingError as error:
        if not reject_nonfinite or error.code != "FF2_NONFINITE":
            raise
        return None


def _finish_many_checked(
    rows: Sequence[tuple[Request, PreparedFormulaBindings, dict[str, Tensor]] | None],
    *, reject_nonfinite: bool,
) -> tuple[_FormulaProgramExecutionArenaV4 | None, ...]:
    if _EFFECT_BACKEND.get() is not None and not torch.is_grad_enabled():
        return _finish_many_tensor_effects(rows, reject_nonfinite=reject_nonfinite)
    return _finish_many_reference(rows, reject_nonfinite=reject_nonfinite)


def _finish_many_reference(rows, *, reject_nonfinite):
    results = [None] * len(rows)
    indices = []
    finite_rows = _FiniteTensorRows()
    device = None
    for index, row in enumerate(rows):
        if row is None:
            continue
        request, prepared, numeric = row
        candidate, arena = request
        defer = reject_nonfinite and type(candidate) is FormulaProgramEffectCandidateV3 and all(
            name not in candidate.__dict__ and getattr(type(candidate), name) is method
            for name, method in _EFFECT_METHODS.items()
        ) and query_runtime.apply_neural_plasticity_effect is _APPLY_EFFECT
        if defer:
            # Native finishing only constructs a branch-local successor; it
            # cannot become a later search input until these checks pass.
            with _capture_finite_validation() as checks:
                result = _finish_checked(request, prepared, numeric, reject_nonfinite=reject_nonfinite)
            if result is not None:
                indices.append(index)
                finite_rows.add(checks)
                device = arena.device
        else:
            result = _finish_checked(request, prepared, numeric, reject_nonfinite=reject_nonfinite)
        results[index] = result
    if indices:
        finite = finite_rows.evaluate(device=device)
        for index, valid in zip(indices, finite.tolist(), strict=True):
            if not valid:
                results[index] = None
    return tuple(results)


def _finish_many_tensor_effects(rows, *, reject_nonfinite):
    groups = {}
    remaining = list(rows)
    for index, row in enumerate(rows):
        if row is None:
            continue
        request, prepared, numeric = row
        candidate, arena = request
        if type(candidate) is not FormulaProgramEffectCandidateV3 or any(
            name in candidate.__dict__ or getattr(type(candidate), name) is not method
            for name, method in _EFFECT_METHODS.items()
        ) or query_runtime.apply_neural_plasticity_effect is not _APPLY_EFFECT:
            continue
        value, effect = _effect_inputs(request, prepared, numeric)
        target = candidate._target(arena)
        previous = target[1]
        try:
            with _capture_finite_validation() as checks:
                axis, maximum = _validate_neural_plasticity_step(effect, previous, candidate.effect_program.state_type)
        except (FormulaProgramError, FormulaBindingError):
            # A zero native count does not execute the law or validate its
            # state/operand shape relationship. Keep that native boundary.
            continue
        count = None if effect.atom_ref == NEURAL_PLASTICITY_OUTER_V2_ATOM_REF else candidate.execution_count_tensor()
        if count is not None and (count.ndim != 0 or count.dtype != previous.dtype or count.device != previous.device):
            raise FormulaProgramError("FF_EFFECT_EXECUTION_COUNT", "execution_count must match predecessor Bank dtype and device")
        key = (
            effect.atom_ref, axis, maximum, candidate.max_executions, count is not None,
            tuple((tuple(item.shape), item.dtype, item.device) for item in (previous, *effect.operands)),
        )
        groups.setdefault(key, []).append((index, request, value, effect, target, count, checks))
        remaining[index] = None
    results = list(_finish_many_reference(remaining, reject_nonfinite=reject_nonfinite))
    for key, group in groups.items():
        atom, axis, maximum, repeats, counted, _metadata = key
        plan = _effect_plan(atom, axis, maximum, repeats, counted, _EFFECT_BACKEND.get())
        states = torch.stack(tuple(row[4][1] for row in group))
        counts = torch.stack(tuple(row[5] for row in group)) if counted else states.new_ones(len(group))
        operands = tuple(torch.stack(tuple(row[3].operands[i] for row in group)) for i in range(len(group[0][3].operands)))
        capacity = 1 << (len(group) - 1).bit_length()
        if capacity != len(group) and _EFFECT_BACKEND.get() != "eager":
            def pad(value):
                return torch.cat((value, value.new_zeros((capacity - len(group), *value.shape[1:]))))
            states, counts, *operands = tuple(pad(value) for value in (states, counts, *operands))
        successors, finite = plan(states, counts, *operands)
        # Graph outputs are reusable scratch; a retained branch owns its values
        # beyond the next graph invocation. One copy retains the entire group.
        if _EFFECT_BACKEND.get() != "eager":
            successors = successors.clone()
        admitted = _FiniteTensorRows()
        for row in group:
            admitted.add(row[6])
        finite = finite[:len(group)] & admitted.evaluate(device=states.device)
        packet = torch.stack((finite, torch.isnan(counts[:len(group)])), dim=1).tolist()
        if any(invalid_count for _valid, invalid_count in packet):
            raise ValueError("cannot convert float NaN to integer")
        validity = [valid for valid, _invalid_count in packet]
        if not reject_nonfinite and not all(validity):
            raise FormulaBindingError("FF2_NONFINITE", "Formula Bank transitions must be finite")
        for row_index, (row, valid) in enumerate(zip(group, validity, strict=True)):
            if not valid:
                continue
            index, (candidate, arena), value, effect, target, _count, _checks = row
            # Numeric validity was checked as one device packet. The original
            # lineage/proposal builder still owns every branch-local successor.
            with _capture_finite_validation():
                results[index] = candidate._finish(
                    arena, value, effect, target=target, successor=successors[row_index],
                )
    return tuple(results)


@overload
def execute_many(
    requests: Sequence[Request], *, chunk_size: int | None = None, serial: bool = False,
    reject_nonfinite: Literal[False] = False,
) -> tuple[_FormulaProgramExecutionArenaV4, ...]:
    ...


@overload
def execute_many(
    requests: Sequence[Request], *, chunk_size: int | None = None, serial: bool = False,
    reject_nonfinite: Literal[True],
) -> tuple[_FormulaProgramExecutionArenaV4 | None, ...]:
    ...


def execute_many(
    requests: Sequence[Request], *, chunk_size: int | None = None, serial: bool = False,
    reject_nonfinite: bool = False,
) -> tuple[_FormulaProgramExecutionArenaV4 | None, ...]:
    """Execute each row once; search may explicitly reject numerical-invalid rows.

    None identifies a rejected row, never a valid identity or zero update.
    Structural, dtype, device and allocation errors keep their normal behavior.
    """
    if chunk_size is not None and (
        isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0
    ):
        raise ValueError("chunk_size must be a positive integer or None")
    if not isinstance(serial, bool):
        raise TypeError("serial must be a bool")
    if not isinstance(reject_nonfinite, bool):
        raise TypeError("reject_nonfinite must be a bool")
    groups: dict[tuple[object, ...], list[tuple[int, PreparedFormulaBindings]]] = {}
    plans: dict[tuple[object, ...], FormulaExecutionPlanV2] = {}
    native: list[int] = []
    for index, (candidate, arena) in enumerate(requests):
        for slot in candidate.output_slot_ids:
            if arena.values.get(slot) is not None:
                raise ValueError(f"SSA output slot {slot!r} is already occupied")
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
        try:
            prepared = _prepare(candidate, arena)
        except FormulaBindingError as error:
            if not reject_nonfinite or error.code != "FF2_NONFINITE":
                raise
            continue
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
            if torch.is_grad_enabled() and use_grouped_training(rows[0][1].values[0].device):
                run = _run_training_group
            options = {"reject_nonfinite": True} if reject_nonfinite else {}
            for (index, _), result in zip(
                rows, run(requests, rows, plans[key], **options), strict=True
            ):
                results[index] = result
    for index in sorted(native):
        candidate, arena = requests[index]
        try:
            results[index] = candidate(arena)
        except FormulaBindingError as error:
            if not reject_nonfinite or error.code != "FF2_NONFINITE":
                raise
    if not reject_nonfinite:
        assert all(result is not None for result in results)
    return tuple(results)
