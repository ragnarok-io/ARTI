"""Searchable Formula self-operation over predecessor-owned Bank slots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import ClassVar
from typing import Callable

import torch
from torch import Tensor, nn

from .formula_program_query import (
    FormulaProgramArena,
    FormulaProgramCandidate,
    FormulaProgramQueryResult,
    _ProgramOperandStore,
)
from .formula_v2 import (
    BankBinding,
    FormulaBankOperand,
    FormulaFabricV2,
    FormulaProgram,
    FormulaV2Error,
    InputBinding,
    formula_program_dependency_refs,
)
from .formula_v3 import (
    FormulaEffectProgramV2,
    FormulaFabricV4,
    NeuralPlasticityEffectV2,
    apply_neural_plasticity_effect,
)


_NAME = re.compile(r"^[a-z][a-z0-9]*(?:[-_.][a-z0-9]+)*$")
_SUMMARY_WIDTH = 8


def _require_name(value: str, *, field: str) -> None:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise ValueError(f"{field} must be a canonical lowercase name")


@dataclass(frozen=True, order=True)
class BankSlotRef:
    """Stable identity of one ordinary Formula producer's Bank operand."""

    producer_id: str
    producer_fingerprint: str
    binding_name: str
    source_ref: str
    partition_id: str
    asset_fingerprint: str | None = None

    _runtime_contract_ref: ClassVar[str] = "arti/bank-slot-ref@1"

    def __post_init__(self) -> None:
        _require_name(self.producer_id, field="producer_id")
        _require_name(self.binding_name, field="binding_name")
        _require_name(self.partition_id, field="partition_id")
        if len(self.producer_fingerprint) != 64 or any(
            character not in "0123456789abcdef"
            for character in self.producer_fingerprint
        ):
            raise ValueError("producer_fingerprint must be a SHA-256 hex digest")
        if not isinstance(self.source_ref, str) or not self.source_ref:
            raise ValueError("source_ref must be non-empty")
        if self.asset_fingerprint is not None and (
            len(self.asset_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in self.asset_fingerprint)
        ):
            raise ValueError("asset_fingerprint must be a SHA-256 hex digest")

    def to_dict(self) -> dict[str, object]:
        return {
            "ref": self._runtime_contract_ref,
            "producer_id": self.producer_id,
            "producer_fingerprint": self.producer_fingerprint,
            "binding_name": self.binding_name,
            "source_ref": self.source_ref,
            "partition_id": self.partition_id,
            "asset_fingerprint": self.asset_fingerprint,
        }


@dataclass(frozen=True)
class FormulaProgramBankState:
    """Immutable values and revisions for producer-owned Bank slots."""

    slot_refs: tuple[BankSlotRef, ...]
    values: tuple[Tensor, ...]
    revisions: tuple[int, ...]

    _runtime_contract_ref: ClassVar[str] = "arti/formula-program-bank-state@1"

    def __post_init__(self) -> None:
        refs = tuple(self.slot_refs)
        values = tuple(self.values)
        revisions = tuple(self.revisions)
        if len(refs) != len(values) or len(values) != len(revisions):
            raise ValueError("Bank state refs, values, and revisions must be aligned")
        if len(set(refs)) != len(refs):
            raise ValueError("Bank state slot_refs must be unique")
        entries = sorted(zip(refs, values, revisions, strict=True), key=lambda item: item[0])
        object.__setattr__(self, "slot_refs", tuple(item[0] for item in entries))
        object.__setattr__(self, "values", tuple(item[1] for item in entries))
        object.__setattr__(self, "revisions", tuple(item[2] for item in entries))
        for value, revision in zip(self.values, self.revisions, strict=True):
            if not isinstance(value, Tensor) or not value.is_floating_point():
                raise TypeError("plastic Bank slots must be floating Tensors")
            if type(revision) is not int or revision < 0:
                raise ValueError("Bank slot revisions must be non-negative integers")

    @classmethod
    def empty(cls) -> FormulaProgramBankState:
        return cls((), (), ())

    def _index(self, slot_ref: BankSlotRef) -> int:
        try:
            return self.slot_refs.index(slot_ref)
        except ValueError as exc:
            raise KeyError(slot_ref) from exc

    def value(self, slot_ref: BankSlotRef) -> Tensor:
        return self.values[self._index(slot_ref)]

    def revision(self, slot_ref: BankSlotRef) -> int:
        return self.revisions[self._index(slot_ref)]

    def replace(
        self,
        slot_ref: BankSlotRef,
        value: Tensor,
        *,
        revision: int,
    ) -> FormulaProgramBankState:
        index = self._index(slot_ref)
        current = self.values[index]
        if (
            not isinstance(value, Tensor)
            or value.shape != current.shape
            or value.dtype != current.dtype
            or value.device != current.device
        ):
            raise ValueError("successor must exactly match its predecessor Bank slot")
        if type(revision) is not int or revision <= self.revisions[index]:
            raise ValueError("successor revision must advance the Bank slot")
        values = list(self.values)
        revisions = list(self.revisions)
        values[index] = value
        revisions[index] = revision
        return FormulaProgramBankState(self.slot_refs, tuple(values), tuple(revisions))


class FormulaProgramCandidateV2(nn.Module):
    """One bounded Formula subprogram with explicit SSA wiring.

    Candidate@1 intentionally represents one atom. Candidate@2 keeps the same
    external SSA contract while allowing a multi-atom Formula to remain one
    ordinary producer. This is required when the producer-owned plastic Bank
    participates inside a composed operation such as a low-rank residual.
    """

    _component_reference: ClassVar[str] = "arti/formula-program-candidate@2"

    def __init__(
        self,
        candidate_id: str,
        program: FormulaProgram,
        *,
        input_slots: Mapping[str, str],
        output_slot: str,
        requires_empty_slots: Sequence[str] = (),
        operands: Mapping[str, Tensor] | None = None,
        trainable_operands: Sequence[str] = (),
        batch_broadcast_operands: Sequence[str] = (),
    ) -> None:
        super().__init__()
        _require_name(candidate_id, field="candidate_id")
        _require_name(output_slot, field="output_slot")
        if not isinstance(program, FormulaProgram):
            raise TypeError("program must be FormulaProgram")
        if len(program.outputs) != 1:
            raise ValueError("FormulaProgramCandidateV2 requires exactly one public output")
        input_bindings = tuple(
            binding for binding in program.bindings if isinstance(binding, InputBinding)
        )
        normalized_inputs = dict(input_slots)
        if set(normalized_inputs) != {binding.name for binding in input_bindings}:
            raise ValueError("input_slots must bind every Formula InputBinding exactly once")
        for slot_id in normalized_inputs.values():
            _require_name(slot_id, field="input slot")
        normalized_empty = tuple(requires_empty_slots)
        if len(set(normalized_empty)) != len(normalized_empty):
            raise ValueError("requires_empty_slots must not contain duplicates")
        for slot_id in normalized_empty:
            _require_name(slot_id, field="required empty slot")
        bank_bindings = {
            binding.name: binding
            for binding in program.bindings
            if isinstance(binding, BankBinding)
        }
        normalized_operands = {} if operands is None else dict(operands)
        if set(normalized_operands) != set(bank_bindings):
            raise ValueError("operands must bind every Formula BankBinding exactly once")
        batch_broadcast = frozenset(batch_broadcast_operands)
        if not batch_broadcast.issubset(bank_bindings):
            raise ValueError("batch_broadcast_operands must name Formula BankBindings")

        self.candidate_id = candidate_id
        self.input_slots = normalized_inputs
        self.output_slot = output_slot
        self.requires_empty_slots = normalized_empty
        self.fabric = FormulaFabricV2(program)
        self._bank_bindings = bank_bindings
        self.operand_store = _ProgramOperandStore(
            normalized_operands,
            trainable=tuple(trainable_operands),
        )
        self.batch_broadcast_operands = batch_broadcast

    @property
    def program(self) -> FormulaProgram:
        return self.fabric.program

    @property
    def atom_ref(self) -> str:
        refs = formula_program_dependency_refs(self.program)
        return refs[0] if len(refs) == 1 else "arti/formula-program@2"

    def contract_config(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "program": self.program.to_dict(),
            "program_fingerprint": self.program.fingerprint,
            "atom_ref": self.atom_ref,
            "atom_refs": list(formula_program_dependency_refs(self.program)),
            "input_slots": dict(sorted(self.input_slots.items())),
            "output_slot": self.output_slot,
            "requires_empty_slots": list(self.requires_empty_slots),
            "operands": self.operand_store.contract_config(),
            "batch_broadcast_operands": sorted(self.batch_broadcast_operands),
        }

    def _bindings(
        self,
        arena: FormulaProgramArena,
    ) -> tuple[dict[str, Tensor], dict[str, FormulaBankOperand]]:
        inputs: dict[str, Tensor] = {}
        for input_name, slot_id in self.input_slots.items():
            value = arena.get(slot_id)
            if value is None:
                raise ValueError(f"candidate input slot {slot_id!r} is empty")
            inputs[input_name] = value
        tensors = self.operand_store.tensors()
        for name in self.batch_broadcast_operands:
            value = tensors[name]
            if value.ndim < 1 or value.shape[0] != 1:
                raise ValueError(
                    f"batch-broadcast operand {name!r} must have a leading singleton axis"
                )
            tensors[name] = value.expand(arena.batch_size, *value.shape[1:])
        banks = {
            name: binding.bind(tensors[name])
            for name, binding in self._bank_bindings.items()
        }
        return inputs, banks

    def accepts(self, arena: FormulaProgramArena) -> bool:
        if arena.get(self.output_slot) is not None:
            return False
        try:
            if any(arena.get(slot_id) is not None for slot_id in self.requires_empty_slots):
                return False
            inputs, banks = self._bindings(arena)
            self.fabric.bind_tensors(inputs=inputs, banks=banks)
        except (FormulaV2Error, KeyError, TypeError, ValueError):
            return False
        return True

    def forward(self, arena: FormulaProgramArena) -> FormulaProgramArena:
        if arena.get(self.output_slot) is not None:
            raise ValueError(f"SSA output slot {self.output_slot!r} is already occupied")
        inputs, banks = self._bindings(arena)
        value = self.fabric(inputs=inputs, banks=banks).values[0]
        return arena.write(self.output_slot, value)


@dataclass(frozen=True)
class _FormulaProducerLineage:
    """Runtime proof that an SSA value came from one ordinary Formula node."""

    producer_id: str
    output_slot: str
    plastic_slot: BankSlotRef | None
    plastic_revision: int | None
    plastic_value: Tensor | None

    _runtime_contract_ref: ClassVar[str] = "arti/formula-producer-lineage@1"


@dataclass(frozen=True)
class _BankSlotEffectProposal:
    """A branch-local, write-only successor for one predecessor Bank slot."""

    target: BankSlotRef
    predecessor_id: str
    effect_candidate_id: str
    effect_instruction_id: str
    effect_atom_ref: str
    previous_revision: int
    successor_revision: int
    previous: Tensor
    successor: Tensor

    _runtime_contract_ref: ClassVar[str] = "arti/bank-slot-effect-proposal@1"

    def __post_init__(self) -> None:
        if self.predecessor_id != self.target.producer_id:
            raise ValueError("effect proposal target must belong to its predecessor")
        if self.successor_revision != self.previous_revision + 1:
            raise ValueError("effect proposal must advance exactly one logical revision")
        if (
            self.previous.shape != self.successor.shape
            or self.previous.dtype != self.successor.dtype
            or self.previous.device != self.successor.device
        ):
            raise ValueError("effect proposal must preserve Bank slot type")


@dataclass(frozen=True)
class _FormulaProgramExecutionArena:
    """Immutable SSA snapshot with producer lineage and write-only proposals."""

    values: FormulaProgramArena
    producers: tuple[_FormulaProducerLineage | None, ...]
    bank_state: FormulaProgramBankState
    proposals: tuple[_BankSlotEffectProposal, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "producers", tuple(self.producers))
        object.__setattr__(self, "proposals", tuple(self.proposals))
        if len(self.producers) != len(self.values.slot_ids):
            raise ValueError("producer lineage must align with SSA slots")
        if not isinstance(self.bank_state, FormulaProgramBankState):
            raise TypeError("bank_state must be FormulaProgramBankState")

    @property
    def batch_size(self) -> int:
        return self.values.batch_size

    @property
    def device(self) -> torch.device:
        return self.values.device

    def producer(self, slot_id: str) -> _FormulaProducerLineage | None:
        try:
            return self.producers[self.values.slot_ids.index(slot_id)]
        except ValueError as exc:
            raise KeyError(slot_id) from exc

    def write(
        self,
        slot_id: str,
        value: Tensor,
        *,
        producer: _FormulaProducerLineage | None,
    ) -> _FormulaProgramExecutionArena:
        index = self.values.slot_ids.index(slot_id)
        producers = list(self.producers)
        producers[index] = producer
        return _FormulaProgramExecutionArena(
            self.values.write(slot_id, value),
            tuple(producers),
            self.bank_state,
            self.proposals,
        )

    def effect_state(self, slot_ref: BankSlotRef) -> tuple[Tensor, int]:
        for proposal in reversed(self.proposals):
            if proposal.target == slot_ref:
                return proposal.successor, proposal.successor_revision
        return self.bank_state.value(slot_ref), self.bank_state.revision(slot_ref)

    def append_proposal(
        self,
        proposal: _BankSlotEffectProposal,
    ) -> _FormulaProgramExecutionArena:
        return _FormulaProgramExecutionArena(
            self.values,
            self.producers,
            self.bank_state,
            (*self.proposals, proposal),
        )

    def committed_state(self) -> FormulaProgramBankState:
        result = self.bank_state
        for proposal in self.proposals:
            result = result.replace(
                proposal.target,
                proposal.successor,
                revision=proposal.successor_revision,
            )
        return result


def _program_output_dependencies(program: FormulaProgram) -> frozenset[str]:
    dependencies: dict[str, frozenset[str]] = {
        binding.name: frozenset((binding.name,)) for binding in program.bindings
    }
    for instruction in program.instructions:
        dependencies[instruction.output_slot] = frozenset(
            dependency
            for input_slot in instruction.input_slots
            for dependency in dependencies[input_slot]
        )
    return dependencies[program.outputs[0]]


FormulaProgramCandidateLike = FormulaProgramCandidate | FormulaProgramCandidateV2


class FormulaProgramTensorCandidateV2(nn.Module):
    """Ordinary Formula node that may own one plastic Bank binding."""

    _component_reference: ClassVar[str] = "arti/formula-program-tensor-candidate@2"

    def __init__(
        self,
        candidate: FormulaProgramCandidateLike,
        *,
        plastic_bank_slot: str | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(candidate, (FormulaProgramCandidate, FormulaProgramCandidateV2)):
            raise TypeError("candidate must be FormulaProgramCandidate@1 or @2")
        if plastic_bank_slot is not None:
            _require_name(plastic_bank_slot, field="plastic_bank_slot")
            if plastic_bank_slot not in candidate._bank_bindings:
                raise ValueError("plastic_bank_slot must name a used Formula BankBinding")
            if plastic_bank_slot in candidate.operand_store.trainable_names:
                raise ValueError(
                    "plastic_bank_slot must be forward-written state, not a trainable operand"
                )
            if plastic_bank_slot not in _program_output_dependencies(candidate.program):
                raise ValueError(
                    "plastic_bank_slot must contribute to the producer's public output"
                )
        self.candidate = candidate
        self.plastic_bank_slot = plastic_bank_slot
        self.register_buffer("bank_revision", torch.zeros((), dtype=torch.int64), persistent=True)

    @property
    def candidate_id(self) -> str:
        return self.candidate.candidate_id

    @property
    def atom_ref(self) -> str:
        return self.candidate.atom_ref

    @property
    def input_slots(self) -> Mapping[str, str]:
        return self.candidate.input_slots

    @property
    def output_slot(self) -> str:
        return self.candidate.output_slot

    @property
    def requires_empty_slots(self) -> tuple[str, ...]:
        return self.candidate.requires_empty_slots

    @property
    def bank_slot_ref(self) -> BankSlotRef | None:
        if self.plastic_bank_slot is None:
            return None
        binding = self.candidate._bank_bindings[self.plastic_bank_slot]
        return BankSlotRef(
            self.candidate_id,
            self._producer_fingerprint(),
            binding.name,
            binding.source_ref,
            binding.partition_id,
            binding.asset_fingerprint,
        )

    def _producer_fingerprint(self) -> str:
        payload = {
            "candidate_id": self.candidate_id,
            "program_fingerprint": self.candidate.program.fingerprint,
            "input_slots": dict(sorted(self.input_slots.items())),
            "output_slot": self.output_slot,
            "plastic_bank_slot": self.plastic_bank_slot,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def initial_bank_value(self) -> Tensor:
        if self.plastic_bank_slot is None:
            raise RuntimeError("candidate has no plastic Bank slot")
        return self.candidate.operand_store.tensor(self.plastic_bank_slot).clone()

    def initial_revision(self) -> int:
        return int(self.bank_revision.detach().cpu())

    def install_(self, state: FormulaProgramBankState) -> None:
        slot_ref = self.bank_slot_ref
        if slot_ref is None:
            return
        self.candidate.operand_store.install_(slot_ref.binding_name, state.value(slot_ref))
        with torch.no_grad():
            self.bank_revision.fill_(state.revision(slot_ref))

    def contract_config(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.contract_config(),
            "plastic_bank_slot": self.plastic_bank_slot,
            "bank_slot_ref": None if self.bank_slot_ref is None else self.bank_slot_ref.to_dict(),
        }

    def _bindings(
        self,
        arena: _FormulaProgramExecutionArena,
    ) -> tuple[dict[str, Tensor], dict[str, FormulaBankOperand]]:
        inputs, banks = self.candidate._bindings(arena.values)
        slot_ref = self.bank_slot_ref
        if slot_ref is not None:
            binding = self.candidate._bank_bindings[slot_ref.binding_name]
            banks[slot_ref.binding_name] = binding.bind(arena.bank_state.value(slot_ref))
        return inputs, banks

    def accepts(self, arena: _FormulaProgramExecutionArena) -> bool:
        if arena.values.get(self.output_slot) is not None:
            return False
        try:
            if any(arena.values.get(slot_id) is not None for slot_id in self.requires_empty_slots):
                return False
            inputs, banks = self._bindings(arena)
            self.candidate.fabric.bind_tensors(inputs=inputs, banks=banks)
        except (FormulaV2Error, KeyError, TypeError, ValueError):
            return False
        return True

    def forward(self, arena: _FormulaProgramExecutionArena) -> _FormulaProgramExecutionArena:
        inputs, banks = self._bindings(arena)
        value = self.candidate.fabric(inputs=inputs, banks=banks).values[0]
        slot_ref = self.bank_slot_ref
        lineage = _FormulaProducerLineage(
            self.candidate_id,
            self.output_slot,
            slot_ref,
            None if slot_ref is None else arena.bank_state.revision(slot_ref),
            None if slot_ref is None else arena.bank_state.value(slot_ref),
        )
        return arena.write(self.output_slot, value, producer=lineage)


class FormulaProgramEffectCandidateV2(nn.Module):
    """Identity-data effect that targets its dynamic ordinary predecessor."""

    _component_reference: ClassVar[str] = "arti/formula-program-effect-candidate@2"

    def __init__(
        self,
        candidate_id: str,
        effect_program: FormulaEffectProgramV2,
        *,
        input_slot: str,
        output_slot: str,
        requires_empty_slots: Sequence[str] = (),
        operands: Mapping[str, Tensor],
        trainable_operands: Sequence[str] = (),
        batch_broadcast_operands: Sequence[str] = (),
    ) -> None:
        super().__init__()
        for value, field in (
            (candidate_id, "candidate_id"),
            (input_slot, "input_slot"),
            (output_slot, "output_slot"),
        ):
            _require_name(value, field=field)
        if not isinstance(effect_program, FormulaEffectProgramV2):
            raise TypeError("effect_program must be FormulaEffectProgramV2")
        input_bindings = tuple(
            item for item in effect_program.program.bindings if isinstance(item, InputBinding)
        )
        if len(input_bindings) != 1 or input_bindings[0].name != effect_program.data_input_name:
            raise ValueError("effect candidate requires one current-data InputBinding")
        bank_bindings = {
            item.name: item
            for item in effect_program.program.bindings
            if isinstance(item, BankBinding)
        }
        normalized_empty = tuple(requires_empty_slots)
        if len(set(normalized_empty)) != len(normalized_empty):
            raise ValueError("requires_empty_slots must not contain duplicates")
        for slot_id in normalized_empty:
            _require_name(slot_id, field="required empty slot")
        broadcast = frozenset(batch_broadcast_operands)
        if not broadcast.issubset(bank_bindings):
            raise ValueError("batch_broadcast_operands must name Formula Bank bindings")

        self.candidate_id = candidate_id
        self.effect_program = effect_program
        self.input_slot = input_slot
        self.output_slot = output_slot
        self.requires_empty_slots = normalized_empty
        self.fabric = FormulaFabricV4(effect_program)
        self._bank_bindings = bank_bindings
        self.operand_store = _ProgramOperandStore(dict(operands), trainable=trainable_operands)
        if set(self.operand_store.names) != set(bank_bindings):
            raise ValueError("operands must bind every Formula BankBinding exactly once")
        self.batch_broadcast_operands = broadcast

    @property
    def atom_ref(self) -> str:
        return self.effect_program.effect_instruction.atom_ref

    @property
    def input_slots(self) -> Mapping[str, str]:
        return {self.effect_program.data_input_name: self.input_slot}

    def contract_config(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "effect_program": self.effect_program.to_dict(),
            "input_slot": self.input_slot,
            "output_slot": self.output_slot,
            "requires_empty_slots": list(self.requires_empty_slots),
            "operands": self.operand_store.contract_config(),
            "batch_broadcast_operands": sorted(self.batch_broadcast_operands),
            "target_resolution": "dynamic-immediate-predecessor-bank-slot",
            "data_lane": "identity",
        }

    def _bindings(
        self,
        arena: _FormulaProgramExecutionArena,
    ) -> tuple[dict[str, Tensor], dict[str, FormulaBankOperand]]:
        value = arena.values.get(self.input_slot)
        if value is None:
            raise ValueError(f"candidate input slot {self.input_slot!r} is empty")
        tensors = self.operand_store.tensors()
        for name in self.batch_broadcast_operands:
            operand = tensors[name]
            if operand.ndim < 1 or operand.shape[0] != 1:
                raise ValueError(
                    f"batch-broadcast operand {name!r} must have a leading singleton axis"
                )
            tensors[name] = operand.expand(arena.batch_size, *operand.shape[1:])
        banks = {name: binding.bind(tensors[name]) for name, binding in self._bank_bindings.items()}
        return {self.effect_program.data_input_name: value}, banks

    def _target(
        self,
        arena: _FormulaProgramExecutionArena,
    ) -> tuple[_FormulaProducerLineage, Tensor, int]:
        lineage = arena.producer(self.input_slot)
        if lineage is None or lineage.plastic_slot is None:
            raise ValueError("effect input must come from an ordinary plastic Formula producer")
        entry_value = arena.bank_state.value(lineage.plastic_slot)
        entry_revision = arena.bank_state.revision(lineage.plastic_slot)
        if (
            lineage.plastic_value is not entry_value
            or lineage.plastic_revision != entry_revision
        ):
            raise ValueError("effect input producer does not match the entry Bank snapshot")
        state, revision = arena.effect_state(lineage.plastic_slot)
        return lineage, state, revision

    def accepts(self, arena: _FormulaProgramExecutionArena) -> bool:
        if arena.values.get(self.output_slot) is not None:
            return False
        try:
            if any(arena.values.get(slot_id) is not None for slot_id in self.requires_empty_slots):
                return False
            inputs, banks = self._bindings(arena)
            self.fabric._executor.bind_tensors(inputs=inputs, banks=banks)
            _lineage, state, _revision = self._target(arena)
            from .formula_v2 import _validate_tensor_against_type

            _validate_tensor_against_type(
                state,
                self.effect_program.state_type,
                name=f"{self.candidate_id}.predecessor_bank_slot",
            )
        except (FormulaV2Error, KeyError, TypeError, ValueError):
            return False
        return True

    def _proposal(
        self,
        effect: NeuralPlasticityEffectV2,
        lineage: _FormulaProducerLineage,
        previous: Tensor,
        revision: int,
    ) -> _BankSlotEffectProposal:
        assert lineage.plastic_slot is not None
        successor = apply_neural_plasticity_effect(
            effect,
            previous,
            state_type=self.effect_program.state_type,
        )
        return _BankSlotEffectProposal(
            lineage.plastic_slot,
            lineage.producer_id,
            self.candidate_id,
            effect.instruction_id,
            effect.atom_ref,
            revision,
            revision + 1,
            previous,
            successor,
        )

    def forward(self, arena: _FormulaProgramExecutionArena) -> _FormulaProgramExecutionArena:
        inputs, banks = self._bindings(arena)
        lineage, previous, revision = self._target(arena)
        result = self.fabric._execute_owned(inputs=inputs, banks=banks)
        value = inputs[self.effect_program.data_input_name]
        if result.value is not value:
            raise RuntimeError("NeuralPlasticity data lane must preserve Tensor identity")
        proposal = self._proposal(result.effect, lineage, previous, revision)
        updated = arena.append_proposal(proposal)
        return updated.write(self.output_slot, result.value, producer=lineage)


FormulaProgramSearchCandidateV3 = (
    FormulaProgramTensorCandidateV2 | FormulaProgramEffectCandidateV2
)
FormulaProgramTaskLossV3 = Callable[[Tensor, object], Tensor]


@dataclass(frozen=True)
class FormulaProgramQueryTraceStepV3:
    step: int
    candidate_id: str
    atom_ref: str | None
    input_slots: tuple[str, ...]
    output_slot: str | None
    predecessor_id: str | None = None
    target_slot: BankSlotRef | None = None
    target_revision: int | None = None


@dataclass(frozen=True)
class FormulaProgramQueryTraceV3:
    steps: tuple[FormulaProgramQueryTraceStepV3, ...]
    stopped: bool


@dataclass(frozen=True)
class FormulaProgramQueryExecutionV3:
    value: Tensor
    bank_state: FormulaProgramBankState
    proposals: tuple[_BankSlotEffectProposal, ...]
    trace: FormulaProgramQueryTraceV3
    _owner_token: object


@dataclass(frozen=True)
class FormulaProgramQueryTrainingLossV3:
    total: Tensor
    task: Tensor
    invalid: Tensor
    success_probability: Tensor
    visited_states: int
    per_row_total: Tensor


class FormulaProgramQueryV3(nn.Module):
    """Search ordinary and predecessor-owned self-operation nodes."""

    _component_reference: ClassVar[str] = "arti/formula-program-query@3"

    def __init__(
        self,
        *,
        slot_ids: Sequence[str],
        candidates: Sequence[FormulaProgramSearchCandidateV3],
        terminal_slot: str,
        min_steps: int = 1,
        max_steps: int = 8,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        normalized_slots = tuple(slot_ids)
        if not normalized_slots or len(set(normalized_slots)) != len(normalized_slots):
            raise ValueError("slot_ids must be a non-empty unique sequence")
        for slot_id in normalized_slots:
            _require_name(slot_id, field="slot_id")
        _require_name(terminal_slot, field="terminal_slot")
        if terminal_slot not in normalized_slots:
            raise ValueError("terminal_slot must name one declared slot")
        normalized_candidates = tuple(candidates)
        if not normalized_candidates or any(
            type(item)
            not in (FormulaProgramTensorCandidateV2, FormulaProgramEffectCandidateV2)
            for item in normalized_candidates
        ):
            raise TypeError("candidates must contain corrected Formula program candidates")
        candidate_ids = tuple(item.candidate_id for item in normalized_candidates)
        if len(set(candidate_ids)) != len(candidate_ids) or "stop" in candidate_ids:
            raise ValueError("candidate ids must be unique and must not use 'stop'")
        for candidate in normalized_candidates:
            referenced = {
                *candidate.input_slots.values(),
                candidate.output_slot,
                *candidate.requires_empty_slots,
            }
            if not referenced.issubset(normalized_slots):
                raise ValueError("candidate wiring must reference declared SSA slots")
        if (
            isinstance(min_steps, bool)
            or not isinstance(min_steps, int)
            or min_steps < 0
            or isinstance(max_steps, bool)
            or not isinstance(max_steps, int)
            or max_steps < max(1, min_steps)
        ):
            raise ValueError("ProgramQuery step bounds are invalid")
        if isinstance(hidden_dim, bool) or not isinstance(hidden_dim, int) or hidden_dim <= 0:
            raise ValueError("hidden_dim must be a positive integer")

        self.slot_ids = normalized_slots
        self.candidates = nn.ModuleList(normalized_candidates)
        self.terminal_slot = terminal_slot
        self.min_steps = int(min_steps)
        self.max_steps = int(max_steps)
        self.hidden_dim = int(hidden_dim)
        self._owner_token = object()
        self.network = nn.Sequential(
            nn.Linear(len(normalized_slots) * _SUMMARY_WIDTH, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, len(normalized_candidates) + 1),
        )
        action_ids = candidate_ids + ("stop",)
        lexical_rank = {item: rank for rank, item in enumerate(sorted(action_ids))}
        self.register_buffer(
            "_action_priority",
            torch.tensor([lexical_rank[item] for item in action_ids], dtype=torch.int64),
            persistent=False,
        )

    @property
    def candidate_ids(self) -> tuple[str, ...]:
        return tuple(item.candidate_id for item in self.candidates)

    @property
    def action_ids(self) -> tuple[str, ...]:
        return self.candidate_ids + ("stop",)

    @property
    def plastic_candidates(self) -> tuple[FormulaProgramTensorCandidateV2, ...]:
        return tuple(
            item
            for item in self.candidates
            if isinstance(item, FormulaProgramTensorCandidateV2)
            and item.bank_slot_ref is not None
        )

    def initial_bank_state(self) -> FormulaProgramBankState:
        candidates = self.plastic_candidates
        refs = tuple(item.bank_slot_ref for item in candidates)
        assert all(item is not None for item in refs)
        return FormulaProgramBankState(
            refs,  # type: ignore[arg-type]
            tuple(item.initial_bank_value() for item in candidates),
            tuple(item.initial_revision() for item in candidates),
        )

    def commit_(self, execution: FormulaProgramQueryExecutionV3) -> None:
        if not isinstance(execution, FormulaProgramQueryExecutionV3):
            raise TypeError("execution must be a FormulaProgramQueryExecutionV3 winner")
        if execution._owner_token is not self._owner_token or not execution.trace.stopped:
            raise ValueError("only this ProgramQuery's stopped winner can be committed")
        for candidate in self.plastic_candidates:
            candidate.install_(execution.bank_state)

    def reexecute(
        self,
        candidate_id: str,
        values: Mapping[str, Tensor],
        *,
        bank_state: FormulaProgramBankState,
    ) -> Tensor:
        """Run one ordinary producer against a committed or functional Bank state."""

        matches = tuple(
            candidate
            for candidate in self.plastic_candidates
            if candidate.candidate_id == candidate_id
        )
        if len(matches) != 1:
            raise ValueError("candidate_id must name one plastic ordinary producer")
        arena = self._arena(values, bank_state=bank_state)
        producer = matches[0]
        if not producer.accepts(arena):
            raise ValueError("values do not satisfy the selected ordinary producer")
        executed = producer(arena)
        output = executed.values.get(producer.output_slot)
        assert output is not None
        return output

    def contract_config(self) -> dict[str, object]:
        return {
            "slot_ids": list(self.slot_ids),
            "candidate_ids": list(self.candidate_ids),
            "terminal_slot": self.terminal_slot,
            "min_steps": self.min_steps,
            "max_steps": self.max_steps,
            "hidden_dim": self.hidden_dim,
            "selection": "hard-one-shape-valid",
            "topology": "query-selected-ordinary-and-predecessor-effect-nodes",
            "effect_target": "dynamic-immediate-predecessor-bank-slot",
            "pending_visibility": "write-only-until-winner-commit",
        }

    def _arena(
        self,
        values: Mapping[str, Tensor],
        *,
        bank_state: FormulaProgramBankState | None = None,
    ) -> _FormulaProgramExecutionArena:
        state = self.initial_bank_state() if bank_state is None else bank_state
        expected = self.initial_bank_state().slot_refs
        if state.slot_refs != expected:
            raise ValueError("bank_state does not match ProgramQuery producer-owned slots")
        arena = FormulaProgramArena.from_mapping(self.slot_ids, values)
        return _FormulaProgramExecutionArena(arena, (None,) * len(self.slot_ids), state)

    def eligible(self, arena: _FormulaProgramExecutionArena, *, steps: int) -> Tensor:
        if arena.values.slot_ids != self.slot_ids:
            raise ValueError("arena layout does not match ProgramQuery@3")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        candidates = [
            steps < self.max_steps and candidate.accepts(arena)
            for candidate in self.candidates
        ]
        stop = steps >= self.min_steps and arena.values.get(self.terminal_slot) is not None
        return torch.tensor((*candidates, stop), dtype=torch.bool, device=arena.device)

    def _summarize(self, arena: _FormulaProgramExecutionArena) -> Tensor:
        parameter = next(self.network.parameters())
        if arena.device != parameter.device:
            raise ValueError("arena and FormulaProgramQuery must share device")
        rows: list[Tensor] = []
        for value in arena.values.values:
            if value is None:
                rows.append(
                    torch.zeros(
                        arena.batch_size,
                        _SUMMARY_WIDTH,
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                )
                continue
            numeric = value.to(dtype=parameter.dtype).reshape(arena.batch_size, -1)
            if not bool(torch.isfinite(numeric).all()):
                raise ValueError("ProgramQuery arena values must be finite")
            rows.append(
                torch.cat(
                    (
                        numeric.new_ones((arena.batch_size, 1)),
                        numeric.mean(dim=-1, keepdim=True),
                        numeric.std(dim=-1, unbiased=False, keepdim=True),
                        numeric.abs().mean(dim=-1, keepdim=True),
                        numeric.amax(dim=-1, keepdim=True),
                        numeric.amin(dim=-1, keepdim=True),
                        numeric.square().mean(dim=-1, keepdim=True).sqrt(),
                        numeric.new_full((arena.batch_size, 1), math.log1p(numeric.shape[-1])),
                    ),
                    dim=-1,
                )
            )
        return torch.cat(rows, dim=-1)

    def query(
        self,
        arena: _FormulaProgramExecutionArena,
        *,
        steps: int,
    ) -> FormulaProgramQueryResult:
        eligible = self.eligible(arena, steps=steps)
        if not bool(eligible.any()):
            raise RuntimeError("ProgramQuery has no shape-valid candidate or valid stop")
        logits = self.network(self._summarize(arena))
        return FormulaProgramQueryResult(logits, logits.masked_fill(~eligible.unsqueeze(0), -torch.inf), eligible)

    def _hard_index(self, masked_logits: Tensor) -> int:
        if masked_logits.shape[0] != 1:
            raise ValueError("hard ProgramQuery execution currently requires batch size one")
        maxima = masked_logits[0] == masked_logits[0].max()
        sentinel = torch.full_like(self._action_priority, len(self.action_ids))
        priority = torch.where(maxima, self._action_priority, sentinel)
        return int(priority.argmin().item())

    def forward(
        self,
        values: Mapping[str, Tensor],
        *,
        bank_state: FormulaProgramBankState | None = None,
    ) -> FormulaProgramQueryExecutionV3:
        if not isinstance(values, Mapping):
            raise TypeError("values must be an SSA input mapping")
        arena = self._arena(values, bank_state=bank_state)
        if arena.batch_size != 1:
            raise ValueError("hard ProgramQuery execution currently requires batch size one")
        trace: list[FormulaProgramQueryTraceStepV3] = []
        steps = 0
        while True:
            result = self.query(arena, steps=steps)
            selected = self._hard_index(result.masked_logits)
            if selected == len(self.candidates):
                value = arena.values.get(self.terminal_slot)
                assert value is not None
                trace.append(FormulaProgramQueryTraceStepV3(steps, "stop", None, (), None))
                return FormulaProgramQueryExecutionV3(
                    value,
                    arena.committed_state(),
                    arena.proposals,
                    FormulaProgramQueryTraceV3(tuple(trace), True),
                    self._owner_token,
                )
            candidate = self.candidates[selected]
            arena = candidate(arena)
            proposal = arena.proposals[-1] if isinstance(candidate, FormulaProgramEffectCandidateV2) else None
            trace.append(
                FormulaProgramQueryTraceStepV3(
                    steps,
                    candidate.candidate_id,
                    candidate.atom_ref,
                    tuple(candidate.input_slots.values()),
                    candidate.output_slot,
                    None if proposal is None else proposal.predecessor_id,
                    None if proposal is None else proposal.target,
                    None if proposal is None else proposal.successor_revision,
                )
            )
            steps += 1


class ExactFormulaProgramQueryTrainingV3:
    """Train event-1 topology through event-2 predecessor re-execution."""

    _component_reference: ClassVar[str] = "arti/exact-formula-program-query-training@3"

    def __init__(
        self,
        *,
        invalid_weight: float = 2.0,
        max_states: int = 512,
        exploration_probability: float = 0.0,
    ) -> None:
        if (
            isinstance(invalid_weight, bool)
            or not isinstance(invalid_weight, (int, float))
            or not math.isfinite(float(invalid_weight))
            or invalid_weight < 0
        ):
            raise ValueError("invalid_weight must be finite and non-negative")
        if isinstance(max_states, bool) or not isinstance(max_states, int) or max_states <= 0:
            raise ValueError("max_states must be a positive integer")
        if (
            isinstance(exploration_probability, bool)
            or not isinstance(exploration_probability, (int, float))
            or not math.isfinite(float(exploration_probability))
            or not 0.0 <= float(exploration_probability) < 1.0
        ):
            raise ValueError("exploration_probability must be finite in [0, 1)")
        self.invalid_weight = float(invalid_weight)
        self.max_states = int(max_states)
        self.exploration_probability = float(exploration_probability)

    def contract_config(self) -> dict[str, object]:
        return {
            "invalid_weight": self.invalid_weight,
            "max_states": self.max_states,
            "exploration_probability": self.exploration_probability,
            "supervision": "event-2-final-task-loss-only",
            "route_teacher": False,
            "state_teacher": False,
            "state_readout": False,
            "event_boundary": "winner-bank-state-then-predecessor-reexecution",
            "estimator": "exact-expected-two-event-program-policy",
            "state_merging": "none-bank-roots-are-path-specific",
        }

    @staticmethod
    def _producer(
        program_query: FormulaProgramQueryV3,
        candidate_id: str,
    ) -> FormulaProgramTensorCandidateV2:
        matches = tuple(
            candidate
            for candidate in program_query.plastic_candidates
            if candidate.candidate_id == candidate_id
        )
        if len(matches) != 1:
            raise ValueError("event2_candidate_id must name one plastic ordinary producer")
        return matches[0]

    @staticmethod
    def _event2_output(
        program_query: FormulaProgramQueryV3,
        producer: FormulaProgramTensorCandidateV2,
        *,
        values: Mapping[str, Tensor],
        bank_state: FormulaProgramBankState,
    ) -> Tensor:
        return program_query.reexecute(
            producer.candidate_id,
            values,
            bank_state=bank_state,
        )

    @staticmethod
    def _validate_row_loss(
        row_loss: Tensor,
        *,
        batch: int,
        device: torch.device,
    ) -> None:
        if (
            not isinstance(row_loss, Tensor)
            or not row_loss.is_floating_point()
            or row_loss.shape != (batch,)
            or row_loss.device != device
            or not bool(torch.isfinite(row_loss).all())
        ):
            raise TypeError("task_loss must return finite floating [B] values")

    def loss(
        self,
        program_query: FormulaProgramQueryV3,
        *,
        event1: Mapping[str, Tensor],
        event2: Mapping[str, Tensor],
        event2_candidate_id: str,
        target: object,
        task_loss: FormulaProgramTaskLossV3,
        bank_state: FormulaProgramBankState | None = None,
    ) -> FormulaProgramQueryTrainingLossV3:
        if not isinstance(program_query, FormulaProgramQueryV3):
            raise TypeError("program_query must be FormulaProgramQueryV3")
        producer = self._producer(program_query, event2_candidate_id)
        arena = program_query._arena(event1, bank_state=bank_state)
        batch = arena.batch_size
        if batch != 1:
            raise ValueError(
                "exact ProgramQuery@3 training currently requires batch size one; "
                "independent rows cannot share one branch-local Bank root"
            )
        reference = next(value for value in arena.values.values if value is not None)
        expected_task = reference.new_zeros((batch,), dtype=torch.float32)
        success = reference.new_zeros((batch,), dtype=torch.float32)
        visited_states = 0
        frontier: list[tuple[_FormulaProgramExecutionArena, Tensor]] = [
            (arena, expected_task.new_ones((batch,)))
        ]
        steps = 0
        while frontier:
            next_frontier: list[tuple[_FormulaProgramExecutionArena, Tensor]] = []
            for state, probability in frontier:
                visited_states += 1
                if visited_states > self.max_states:
                    raise RuntimeError("ProgramQuery path graph exceeded max_states")
                try:
                    result = program_query.query(state, steps=steps)
                except RuntimeError:
                    continue
                action_probability = result.masked_logits.softmax(dim=-1)
                if self.exploration_probability:
                    uniform = result.eligible.to(action_probability)
                    uniform = uniform / uniform.sum()
                    action_probability = (
                        1.0 - self.exploration_probability
                    ) * action_probability + self.exploration_probability * uniform.unsqueeze(0)
                stop_index = len(program_query.candidates)
                if bool(result.eligible[stop_index]):
                    output = self._event2_output(
                        program_query,
                        producer,
                        values=event2,
                        bank_state=state.committed_state(),
                    )
                    row_loss = task_loss(output, target)
                    self._validate_row_loss(row_loss, batch=batch, device=arena.device)
                    stop_probability = probability * action_probability[:, stop_index]
                    expected_task = expected_task + stop_probability * row_loss.float()
                    success = success + stop_probability
                for candidate_index, candidate in enumerate(program_query.candidates):
                    if not bool(result.eligible[candidate_index]):
                        continue
                    next_frontier.append(
                        (
                            candidate(state),
                            probability * action_probability[:, candidate_index],
                        )
                    )
            frontier = next_frontier
            steps += 1
        invalid = (1.0 - success).clamp_min(0.0)
        conditional_task = expected_task / success.clamp_min(1e-8)
        task = conditional_task.mean()
        invalid_loss = invalid.mean()
        per_row = conditional_task + self.invalid_weight * invalid
        return FormulaProgramQueryTrainingLossV3(
            total=task + self.invalid_weight * invalid_loss,
            task=task,
            invalid=invalid_loss,
            success_probability=success.mean(),
            visited_states=visited_states,
            per_row_total=per_row,
        )


__all__ = [
    "BankSlotRef",
    "ExactFormulaProgramQueryTrainingV3",
    "FormulaProgramBankState",
    "FormulaProgramCandidateV2",
    "FormulaProgramEffectCandidateV2",
    "FormulaProgramQueryExecutionV3",
    "FormulaProgramQueryTraceStepV3",
    "FormulaProgramQueryTraceV3",
    "FormulaProgramQueryTrainingLossV3",
    "FormulaProgramQueryV3",
    "FormulaProgramSearchCandidateV3",
    "FormulaProgramTensorCandidateV2",
    "FormulaProgramTaskLossV3",
]
