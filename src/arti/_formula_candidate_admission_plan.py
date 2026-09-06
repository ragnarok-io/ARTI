"""Native tensor-candidate binding sources for one read-only admission batch."""

from torch import Tensor, nn

from .formula_program_query import _ProgramOperandStore
from .formula_v2 import BankBinding, InputBinding


_STORE_TENSOR = _ProgramOperandStore.tensor
_EXTENSION_TENSOR = object()


class _TensorAdmissionPlan:
    def __init__(self, candidate, slot_ids):
        implementation = candidate.candidate
        program = implementation.program
        self.candidate = candidate
        self.binding_plan = program._binding_plan
        self.store = implementation.operand_store
        self.bank_ref = candidate.bank_slot_ref
        self.broadcasts = {}
        self.empty = tuple(slot_ids.index(slot) for slot in
                           (*candidate.output_slot_ids, *candidate.requires_empty_slots))
        self.sources = tuple(
            ("input", slot_ids.index(candidate.input_slots[binding.name]), False)
            if type(binding) is InputBinding else
            ("plastic" if binding.name == candidate.plastic_bank_slot else "operand",
             binding.name, binding.name in implementation.batch_broadcast_operands)
            for binding in program.bindings
        )

    def checked_values(self, arena, *, metadata_cache=None):
        if any(arena.values.values[index] is not None for index in self.empty):
            return None
        values = []
        for source, key, broadcast in self.sources:
            if source == "input":
                value = arena.values.values[key]
            elif source == "plastic":
                value, _revision = arena.effect_state(self.bank_ref)
            else:
                value = self.store.tensor(key)
            if value is not None and type(value) not in (Tensor, nn.Parameter):
                return _EXTENSION_TENSOR
            if broadcast:
                broadcast_key = (id(value), arena.batch_size)
                cached = self.broadcasts.get(broadcast_key)
                if cached is None:
                    if value.ndim < 1 or value.shape[0] != 1:
                        return None
                    expanded = value.expand(arena.batch_size, *value.shape[1:])
                    self.broadcasts[broadcast_key] = (value, expanded)
                    value = expanded
                else:
                    value = cached[1]
            values.append(value)
        return values if self.binding_plan.has_checked_metadata(values, metadata_cache=metadata_cache) else None


def prepare_tensor_admission(candidate, slot_ids):
    """Only reuse metadata established by the ordinary binding validator."""
    implementation = candidate.candidate
    store = implementation.operand_store
    if type(store) is not _ProgramOperandStore or "tensor" in store.__dict__ or type(store).tensor is not _STORE_TENSOR:
        return None
    program = implementation.program
    if any(type(binding) not in (InputBinding, BankBinding) for binding in program.bindings):
        return None
    # Native _bindings creates each operand from this declared Bank identity.
    declared = {binding.name: binding for binding in program.bindings if type(binding) is BankBinding}
    if implementation._bank_bindings != declared or set(candidate.input_slots) != set(program.input_names):
        return None
    try:
        return _TensorAdmissionPlan(candidate, slot_ids)
    except (KeyError, ValueError):
        return None
