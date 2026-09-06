"""Device-resident numerical admission for metadata-compatible candidates."""

import torch
from torch import Tensor, nn

from .formula_program_query_v4 import FormulaProgramEffectCandidateV3, FormulaProgramTensorCandidateV3, FormulaProgramQueryV4
from .formula_program_query_v5 import FormulaProgramTensorCandidateV4, FormulaProgramQueryV5
from .formula_program_query_v6 import FormulaProgramQueryV6
from .formula_v2 import BankBinding, FormulaBankOperand, FormulaFabricV2, _capture_finite_validation
from .formula_program_query import _ProgramOperandStore
from ._formula_candidate_admission_plan import _EXTENSION_TENSOR, prepare_tensor_admission
from ._formula_finite_rows import _FiniteTensorRows


_NATIVE_ELIGIBLE = FormulaProgramQueryV4._candidate_eligible
_NATIVE_CANDIDATES = (FormulaProgramTensorCandidateV3, FormulaProgramTensorCandidateV4, FormulaProgramEffectCandidateV3)
_NATIVE_ACCEPTS = {cls: cls.accepts for cls in _NATIVE_CANDIDATES}
_NATIVE_BINDINGS = {cls: cls._bindings for cls in _NATIVE_CANDIDATES}
_NATIVE_BUDGET = {cls: cls._candidate_budget_eligible for cls in (FormulaProgramQueryV4, FormulaProgramQueryV5, FormulaProgramQueryV6)}
_NATIVE_BIND_TENSORS = FormulaFabricV2.bind_tensors
_NATIVE_BANK_BIND = BankBinding.bind
_NATIVE_BANK_CONSUME = FormulaBankOperand.consume
_NATIVE_STORE_TENSOR = _ProgramOperandStore.tensor
_NATIVE_STORE_TENSORS = _ProgramOperandStore.tensors
_NATIVE_TARGET = FormulaProgramEffectCandidateV3._target
_NATIVE_STOP = {cls: cls._stop_eligible for cls in (FormulaProgramQueryV4, FormulaProgramQueryV5, FormulaProgramQueryV6)}


def _prepare_native_candidate(query, candidate):
    capture = type(candidate) in _NATIVE_CANDIDATES and not any(
        name in candidate.__dict__ for name in ("accepts", "_bindings", "_target")
    ) and type(query) in (FormulaProgramQueryV4, FormulaProgramQueryV5, FormulaProgramQueryV6) and not any(
        name in query.__dict__ for name in ("_candidate_eligible", "_candidate_budget_eligible")
    ) and type(query)._candidate_eligible is _NATIVE_ELIGIBLE and (
        type(query)._candidate_budget_eligible is _NATIVE_BUDGET[type(query)]
        and type(candidate).accepts is _NATIVE_ACCEPTS[type(candidate)]
        and type(candidate)._bindings is _NATIVE_BINDINGS[type(candidate)]
        and BankBinding.bind is _NATIVE_BANK_BIND
        and FormulaBankOperand.consume is _NATIVE_BANK_CONSUME
        and _ProgramOperandStore.tensor is _NATIVE_STORE_TENSOR
        and _ProgramOperandStore.tensors is _NATIVE_STORE_TENSORS
    )
    if capture:
        implementation = candidate if isinstance(candidate, FormulaProgramEffectCandidateV3) else candidate.candidate
        store = implementation.operand_store
        capture = type(store) is _ProgramOperandStore and not any(
            name in store.__dict__ for name in ("tensor", "tensors")
        ) and all(type(binding) is BankBinding for binding in implementation._bank_bindings.values())
        if capture:
            capture = all(type(value) in (Tensor, nn.Parameter) for value in store.tensors().values())
    if capture and isinstance(candidate, FormulaProgramEffectCandidateV3):
        capture = type(candidate)._target is _NATIVE_TARGET
    if capture:
        fabric = candidate.fabric._executor if isinstance(candidate, FormulaProgramEffectCandidateV3) else candidate.candidate.fabric
        capture = type(fabric) is FormulaFabricV2 and "bind_tensors" not in fabric.__dict__ and (
            type(fabric).bind_tensors is _NATIVE_BIND_TENSORS
        )
    plan = prepare_tensor_admission(candidate, query.slot_ids) if capture and type(candidate) in (
        FormulaProgramTensorCandidateV3, FormulaProgramTensorCandidateV4,
    ) else None
    return capture, plan


def candidate_mask(query, arenas, candidates, *, steps: int, include_stop: bool) -> Tensor:
    mask = _candidate_mask(query, arenas, candidates, steps=steps, include_stop=include_stop)
    if not query._uses_external_query:
        columns = [query.candidate_ids.index(candidate.candidate_id) for candidate in candidates]
        if include_stop:
            columns.append(len(query.candidates))
        frontier = torch.stack(tuple(query.routing_mask(arena, steps=steps) for arena in arenas))
        mask = mask & frontier[:, columns]
    return mask


def _candidate_mask(query, arenas, candidates, *, steps: int, include_stop: bool) -> Tensor:
    """Keep finite predicates on device, separate from host-only wiring checks.

    Shape/layout/Bank lineage admission retains the native contract. Python
    extension candidates and returning calls retain their native admission.
    No device scalar is read for built-in numerical candidate admission.
    """
    from ._formula_admission_table import indexed_candidate_mask

    indexed = indexed_candidate_mask(query, arenas, candidates, steps=steps, include_stop=include_stop)
    if indexed is not None:
        return indexed
    device = arenas[0].device
    width = len(candidates) + int(include_stop)
    mask = torch.zeros((len(arenas), width), dtype=torch.bool, device=device)
    cells = []
    finite_rows = _FiniteTensorRows()
    structural = query._structural_candidates(arenas, candidates)
    preparation = {}
    metadata_cache = {}

    def finish_checks():
        nonlocal finite_rows
        if cells:
            valid = finite_rows.evaluate(device=device)
            mask.view(-1)[torch.tensor(cells, device=device)] = valid
            cells.clear()
            finite_rows = _FiniteTensorRows()

    for row, (arena, ready) in enumerate(zip(arenas, structural, strict=True)):
        native_inputs = all(value is None or type(value) in (Tensor, nn.Parameter) for value in arena.values.values)
        for column in ready:
            candidate = candidates[column]
            if column not in preparation:
                preparation[column] = _prepare_native_candidate(query, candidate)
            capture, plan = preparation[column]
            if not native_inputs:
                capture, plan = False, None
            checks = None
            if plan is not None and query._candidate_budget_eligible(candidate, arena, steps=steps):
                try:
                    checks = plan.checked_values(arena, metadata_cache=metadata_cache)
                except (KeyError, TypeError, ValueError):
                    pass  # Use the original validator for a metadata miss/error.
            if checks is _EXTENSION_TENSOR:
                capture, checks = False, None
            if capture and isinstance(candidate, FormulaProgramEffectCandidateV3):
                try:
                    _, target, _ = candidate._target(arena)
                except (KeyError, TypeError, ValueError):
                    pass  # Normal eligibility still diagnoses invalid lineage.
                else:
                    capture = type(target) in (Tensor, nn.Parameter)
            if checks is not None:
                allowed = True
            elif capture:
                with _capture_finite_validation() as checks:
                    allowed = query._candidate_eligible(candidate, arena, steps=steps)
            else:
                # Extensions may replace their neighboring Python bindings.
                finish_checks()
                preparation.clear()
                metadata_cache.clear()
                try:
                    allowed = query._candidate_eligible(candidate, arena, steps=steps)
                finally:
                    preparation.clear()
                    metadata_cache.clear()
                checks = ()
            if not allowed:
                continue
            cells.append(row * width + column)
            finite_rows.add(checks)
        if include_stop:
            native_stop = "_stop_eligible" not in query.__dict__ and (
                type(query)._stop_eligible is _NATIVE_STOP.get(type(query))
            )
            if not native_stop:
                finish_checks()
                preparation.clear()
                metadata_cache.clear()
            try:
                can_stop = query._stop_eligible(arena, steps=steps)
            finally:
                if not native_stop:
                    preparation.clear()
                    metadata_cache.clear()
            if can_stop:
                cells.append(row * width + len(candidates))
                finite_rows.add(())
    finish_checks()
    return mask
