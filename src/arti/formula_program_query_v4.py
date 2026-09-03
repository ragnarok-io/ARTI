"""Branch-visible Formula self-operation over stable producer-owned Bank slots."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import ClassVar

import torch
from torch import Tensor, nn

from .formula_program_query import FormulaProgramArena, FormulaProgramQueryResult
from .formula_program_query_v3 import (
    BankSlotRef,
    FormulaProgramBankState,
    FormulaProgramCandidateLike,
    FormulaProgramEffectCandidateV2,
    FormulaProgramTensorCandidateV2,
    _require_name,
    _SUMMARY_WIDTH,
)
from .formula_v2 import FormulaBankOperand, FormulaV2Error
from .formula_v3 import (
    FormulaEffectProgramV2,
    NEURAL_PLASTICITY_OUTER_V2_ATOM_REF,
    NeuralPlasticityEffectV2,
    apply_neural_plasticity_effect,
)


@dataclass(frozen=True)
class _FormulaProducerLineageV2:
    """One SSA occurrence and the exact branch-local Bank snapshot it observed."""

    execution_id: str
    owner_id: str
    output_slot: str
    plastic_slot: BankSlotRef | None
    plastic_revision: int | None
    plastic_value: Tensor | None

    _runtime_contract_ref: ClassVar[str] = "arti/formula-producer-lineage@2"


@dataclass(frozen=True)
class _BankSlotEffectProposalV2:
    """A branch-local successor retaining both occurrence and owner identity."""

    target: BankSlotRef
    predecessor_execution_id: str
    predecessor_owner_id: str
    effect_candidate_id: str
    effect_instruction_id: str
    effect_atom_ref: str
    previous_revision: int
    successor_revision: int
    previous: Tensor
    successor: Tensor

    _runtime_contract_ref: ClassVar[str] = "arti/bank-slot-effect-proposal@2"

    def __post_init__(self) -> None:
        if self.predecessor_owner_id != self.target.producer_id:
            raise ValueError("effect proposal target must belong to its predecessor owner")
        if self.successor_revision != self.previous_revision + 1:
            raise ValueError("effect proposal must advance exactly one logical revision")
        if (
            self.previous.shape != self.successor.shape
            or self.previous.dtype != self.successor.dtype
            or self.previous.device != self.successor.device
        ):
            raise ValueError("effect proposal must preserve Bank slot type")


@dataclass(frozen=True)
class _FormulaProgramExecutionArenaV4:
    """Immutable SSA values plus a branch-local proposal overlay."""

    values: FormulaProgramArena
    producers: tuple[_FormulaProducerLineageV2 | None, ...]
    bank_state: FormulaProgramBankState
    proposals: tuple[_BankSlotEffectProposalV2, ...] = ()

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

    def producer(self, slot_id: str) -> _FormulaProducerLineageV2 | None:
        try:
            return self.producers[self.values.slot_ids.index(slot_id)]
        except ValueError as exc:
            raise KeyError(slot_id) from exc

    def write(
        self,
        slot_id: str,
        value: Tensor,
        *,
        producer: _FormulaProducerLineageV2 | None,
    ) -> _FormulaProgramExecutionArenaV4:
        index = self.values.slot_ids.index(slot_id)
        producers = list(self.producers)
        producers[index] = producer
        return _FormulaProgramExecutionArenaV4(
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
        proposal: _BankSlotEffectProposalV2,
    ) -> _FormulaProgramExecutionArenaV4:
        current, revision = self.effect_state(proposal.target)
        if proposal.previous is not current or proposal.previous_revision != revision:
            raise ValueError("effect proposal must extend the latest branch-local Bank revision")
        return _FormulaProgramExecutionArenaV4(
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


class FormulaProgramBankOwnerV1(nn.Module):
    """The single persistent fast-state owner shared by SSA occurrences."""

    _component_reference: ClassVar[str] = "arti/formula-program-bank-owner@1"

    def __init__(self, slot_ref: BankSlotRef, value: Tensor, *, revision: int = 0) -> None:
        super().__init__()
        if not isinstance(slot_ref, BankSlotRef):
            raise TypeError("slot_ref must be a BankSlotRef")
        if not isinstance(value, Tensor) or not value.is_floating_point():
            raise TypeError("Bank owner value must be a floating Tensor")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 0:
            raise ValueError("Bank owner revision must be a non-negative integer")
        self.slot_ref = slot_ref
        self.register_buffer("value", value.detach().clone(), persistent=True)
        self.register_buffer(
            "revision",
            torch.tensor(revision, dtype=torch.int64, device=value.device),
            persistent=True,
        )

    def install_(self, state: FormulaProgramBankState) -> None:
        value = state.value(self.slot_ref)
        revision = state.revision(self.slot_ref)
        if value.shape != self.value.shape or value.dtype != self.value.dtype:
            raise ValueError("installed Bank owner value must preserve shape and dtype")
        with torch.no_grad():
            self.value.copy_(value.detach())
            self.revision.fill_(revision)

    def contract_config(self) -> dict[str, object]:
        return {
            "slot_ref": self.slot_ref.to_dict(),
            "shape": list(self.value.shape),
            "dtype": str(self.value.dtype),
            "state": "forward-written-not-optimizer-parameter",
        }


class FormulaProgramQueryTensorEncoderV1(nn.Module):
    """Content-sensitive fixed-width summary for dynamic-length SSA tensors."""

    _component_reference: ClassVar[str] = "arti/formula-program-query-tensor-encoder@1"

    def __init__(self, input_dim: int, width: int) -> None:
        super().__init__()
        for value, name in ((input_dim, "input_dim"), (width, "width")):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.input_dim = int(input_dim)
        self.width = int(width)
        self.network = nn.Sequential(
            nn.Linear(self.input_dim, self.width),
            nn.SiLU(),
            nn.Linear(self.width, self.width),
        )

    @property
    def output_width(self) -> int:
        return 4 * self.width + 2

    def contract_config(self) -> dict[str, object]:
        return {
            "input_dim": self.input_dim,
            "width": self.width,
            "output_width": self.output_width,
            "reduction": "first-last-mean-standard-deviation",
            "dynamic_axes": "all-between-leading-batch-and-feature",
            "observes": "ssa-tensor-values-only",
        }

    def forward(self, value: Tensor) -> Tensor:
        parameter = next(self.network.parameters())
        if (
            not isinstance(value, Tensor)
            or not value.is_floating_point()
            or value.ndim < 2
            or value.shape[-1] != self.input_dim
        ):
            raise ValueError(
                "Query tensor encoder expects floating [B,...,input_dim] values"
            )
        if value.device != parameter.device:
            raise ValueError("Query tensor encoder and input must share device")
        tokens = value.to(dtype=parameter.dtype).reshape(
            value.shape[0],
            -1,
            self.input_dim,
        )
        if tokens.shape[1] == 0 or not bool(torch.isfinite(tokens).all()):
            raise ValueError("Query tensor encoder input must be non-empty and finite")
        encoded = self.network(tokens)
        return torch.cat(
            (
                encoded.new_ones((encoded.shape[0], 1)),
                encoded[:, 0],
                encoded[:, -1],
                encoded.mean(dim=1),
                encoded.std(dim=1, unbiased=False),
                encoded.new_full(
                    (encoded.shape[0], 1),
                    math.log1p(tokens.shape[1]),
                ),
            ),
            dim=-1,
        )


class FormulaProgramTensorCandidateV3(FormulaProgramTensorCandidateV2):
    """An ordinary SSA occurrence that may share a stable plastic Bank owner."""

    _component_reference: ClassVar[str] = "arti/formula-program-tensor-candidate@3"

    def __init__(
        self,
        candidate: FormulaProgramCandidateLike,
        *,
        plastic_bank_slot: str | None = None,
        bank_owner_id: str | None = None,
    ) -> None:
        super().__init__(candidate, plastic_bank_slot=plastic_bank_slot)
        owner_id = candidate.candidate_id if bank_owner_id is None else bank_owner_id
        _require_name(owner_id, field="bank_owner_id")
        if plastic_bank_slot is None and bank_owner_id is not None:
            raise ValueError("bank_owner_id requires plastic_bank_slot")
        self.bank_owner_id = owner_id
        inherited_revision = self._buffers.pop("bank_revision")
        owner = None
        if plastic_bank_slot is not None:
            slot_ref = self._make_bank_slot_ref()
            owner = FormulaProgramBankOwnerV1(
                slot_ref,
                self.candidate.operand_store.tensor(plastic_bank_slot),
                revision=int(inherited_revision.detach().cpu()),
            )
            self._bind_bank_owner(owner)
        object.__setattr__(self, "_bank_owner", owner)

    @property
    def bank_slot_ref(self) -> BankSlotRef | None:
        if self.plastic_bank_slot is None:
            return None
        owner = self.bank_owner
        return owner.slot_ref

    @property
    def bank_owner(self) -> FormulaProgramBankOwnerV1:
        owner = self._bank_owner
        if owner is None:
            raise RuntimeError("candidate has no plastic Bank owner")
        return owner

    def _make_bank_slot_ref(self) -> BankSlotRef:
        assert self.plastic_bank_slot is not None
        binding = self.candidate._bank_bindings[self.plastic_bank_slot]
        return BankSlotRef(
            self.bank_owner_id,
            self._owner_fingerprint(),
            binding.name,
            binding.source_ref,
            binding.partition_id,
            binding.asset_fingerprint,
        )

    def _owner_fingerprint(self) -> str:
        payload = {
            "bank_owner_id": self.bank_owner_id,
            "program_fingerprint": self.candidate.program.fingerprint,
            "plastic_bank_slot": self.plastic_bank_slot,
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def _bind_bank_owner(self, owner: FormulaProgramBankOwnerV1) -> None:
        if self.plastic_bank_slot is None or owner.slot_ref != self._make_bank_slot_ref():
            raise ValueError("Bank owner does not match this occurrence contract")
        self.candidate.operand_store._bind_external_buffer(
            self.plastic_bank_slot, owner, "value"
        )
        object.__setattr__(self, "_bank_owner", owner)

    def initial_bank_value(self) -> Tensor:
        return self.bank_owner.value.clone()

    def initial_revision(self) -> int:
        return int(self.bank_owner.revision.detach().cpu())

    def install_(self, state: FormulaProgramBankState) -> None:
        if self.plastic_bank_slot is not None:
            self.bank_owner.install_(state)

    def contract_config(self) -> dict[str, object]:
        return {
            "candidate": self.candidate.contract_config(),
            "execution_id": self.candidate_id,
            "bank_owner_id": self.bank_owner_id,
            "plastic_bank_slot": self.plastic_bank_slot,
            "bank_slot_ref": None if self.bank_slot_ref is None else self.bank_slot_ref.to_dict(),
            "bank_read": "latest-branch-local-proposal-overlay",
        }

    def _bindings(
        self,
        arena: _FormulaProgramExecutionArenaV4,
    ) -> tuple[dict[str, Tensor], dict[str, FormulaBankOperand]]:
        inputs: dict[str, Tensor] = {}
        for input_name, slot_id in self.input_slots.items():
            value = arena.values.get(slot_id)
            if value is None:
                raise ValueError(f"candidate input slot {slot_id!r} is empty")
            inputs[input_name] = value
        tensors: dict[str, Tensor] = {}
        for name in self.candidate._bank_bindings:
            if name == self.plastic_bank_slot:
                continue
            operand = self.candidate.operand_store.tensor(name)
            if name in self.candidate.batch_broadcast_operands:
                if operand.ndim < 1 or operand.shape[0] != 1:
                    raise ValueError(
                        f"batch-broadcast operand {name!r} must have a leading singleton axis"
                    )
                operand = operand.expand(arena.batch_size, *operand.shape[1:])
            tensors[name] = operand
        slot_ref = self.bank_slot_ref
        if slot_ref is not None:
            current, _revision = arena.effect_state(slot_ref)
            if slot_ref.binding_name in self.candidate.batch_broadcast_operands:
                if current.ndim < 1 or current.shape[0] != 1:
                    raise ValueError(
                        f"batch-broadcast operand {slot_ref.binding_name!r} "
                        "must have a leading singleton axis"
                    )
                current = current.expand(arena.batch_size, *current.shape[1:])
            tensors[slot_ref.binding_name] = current
        banks = {
            name: binding.bind(tensors[name])
            for name, binding in self.candidate._bank_bindings.items()
        }
        return inputs, banks

    def accepts(self, arena: _FormulaProgramExecutionArenaV4) -> bool:
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

    def forward(
        self,
        arena: _FormulaProgramExecutionArenaV4,
    ) -> _FormulaProgramExecutionArenaV4:
        if arena.values.get(self.output_slot) is not None:
            raise ValueError(f"SSA output slot {self.output_slot!r} is already occupied")
        inputs, banks = self._bindings(arena)
        value = self.candidate.fabric(inputs=inputs, banks=banks).values[0]
        slot_ref = self.bank_slot_ref
        current: Tensor | None = None
        revision: int | None = None
        if slot_ref is not None:
            current, revision = arena.effect_state(slot_ref)
        lineage = _FormulaProducerLineageV2(
            self.candidate_id,
            self.bank_owner_id,
            self.output_slot,
            slot_ref,
            revision,
            current,
        )
        return arena.write(self.output_slot, value, producer=lineage)


class FormulaProgramEffectCandidateV3(FormulaProgramEffectCandidateV2):
    """Identity-data effect that advances the latest predecessor Bank revision."""

    _component_reference: ClassVar[str] = "arti/formula-program-effect-candidate@3"

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
        execution_count: Tensor | None = None,
        max_executions: int = 16,
        trainable_execution_count: bool = False,
    ) -> None:
        super().__init__(
            candidate_id,
            effect_program,
            input_slot=input_slot,
            output_slot=output_slot,
            requires_empty_slots=requires_empty_slots,
            operands=operands,
            trainable_operands=trainable_operands,
            batch_broadcast_operands=batch_broadcast_operands,
        )
        if (
            isinstance(max_executions, bool)
            or not isinstance(max_executions, int)
            or max_executions <= 0
        ):
            raise ValueError("max_executions must be a positive integer")
        if trainable_execution_count and execution_count is None:
            raise ValueError("trainable_execution_count requires execution_count")
        if execution_count is not None and (
            not isinstance(execution_count, Tensor)
            or not execution_count.is_floating_point()
            or execution_count.ndim != 0
        ):
            raise TypeError("execution_count must be a floating scalar Tensor")
        if execution_count is not None and "execution.count" in self.operand_store.names:
            raise ValueError("effect occurrence cannot declare two execution counts")
        parameter = (
            None
            if execution_count is None
            else nn.Parameter(
                execution_count.detach().clone(),
                requires_grad=trainable_execution_count,
            )
        )
        self.register_parameter("execution_count", parameter)
        self.max_executions = int(max_executions)

    def execution_count_tensor(self) -> Tensor | None:
        if self.execution_count is not None:
            return self.execution_count
        if "execution.count" in self.operand_store.names:
            return self.operand_store.tensor("execution.count")
        return None

    def hard_execution_count(self) -> int:
        count = self.execution_count_tensor()
        if count is None:
            return 1
        maximum = (
            dict(self.effect_program.effect_instruction.attributes).get(
                "max_executions",
                self.max_executions,
            )
            if self.atom_ref == NEURAL_PLASTICITY_OUTER_V2_ATOM_REF
            else self.max_executions
        )
        return int(count.detach().clamp(0.0, float(maximum)).round().item())

    def contract_config(self) -> dict[str, object]:
        config = super().contract_config()
        config.update(
            {
                "lineage_ref": "arti/formula-producer-lineage@2",
                "proposal_visibility": "branch-local-on-next-ordinary-execution",
                "execution_count": {
                    "source": (
                        "atom-operand"
                        if "execution.count" in self.operand_store.names
                        else "occurrence-operand"
                        if self.execution_count is not None
                        else "fixed-one"
                    ),
                    "max_executions": self.max_executions,
                    "trainable": bool(
                        self.execution_count is not None
                        and self.execution_count.requires_grad
                    )
                    or "execution.count" in self.operand_store.trainable_names,
                    "semantics": "bounded-hard-repeat-predecessor-bank-slot",
                },
            }
        )
        return config

    def _target(
        self,
        arena: _FormulaProgramExecutionArenaV4,
    ) -> tuple[_FormulaProducerLineageV2, Tensor, int]:
        lineage = arena.producer(self.input_slot)
        if lineage is None or lineage.plastic_slot is None:
            raise ValueError("effect input must come from an ordinary plastic Formula producer")
        current, revision = arena.effect_state(lineage.plastic_slot)
        if lineage.plastic_value is not current or lineage.plastic_revision != revision:
            raise ValueError("effect input lineage is stale for the branch-local Bank overlay")
        return lineage, current, revision

    def accepts(self, arena: _FormulaProgramExecutionArenaV4) -> bool:
        if arena.values.get(self.output_slot) is not None:
            return False
        try:
            if any(arena.values.get(slot_id) is not None for slot_id in self.requires_empty_slots):
                return False
            inputs, banks = self._bindings(arena)  # type: ignore[arg-type]
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
        lineage: _FormulaProducerLineageV2,
        previous: Tensor,
        revision: int,
    ) -> _BankSlotEffectProposalV2:
        assert lineage.plastic_slot is not None
        count = self.execution_count_tensor()
        successor = apply_neural_plasticity_effect(
            effect,
            previous,
            state_type=self.effect_program.state_type,
            execution_count=(
                None
                if self.atom_ref == NEURAL_PLASTICITY_OUTER_V2_ATOM_REF
                else count
            ),
            max_executions=self.max_executions,
        )
        return _BankSlotEffectProposalV2(
            lineage.plastic_slot,
            lineage.execution_id,
            lineage.owner_id,
            self.candidate_id,
            effect.instruction_id,
            effect.atom_ref,
            revision,
            revision + 1,
            previous,
            successor,
        )

    def forward(
        self,
        arena: _FormulaProgramExecutionArenaV4,
    ) -> _FormulaProgramExecutionArenaV4:
        if arena.values.get(self.output_slot) is not None:
            raise ValueError(f"SSA output slot {self.output_slot!r} is already occupied")
        inputs, banks = self._bindings(arena)  # type: ignore[arg-type]
        lineage, previous, revision = self._target(arena)
        result = self.fabric._execute_owned(inputs=inputs, banks=banks)
        value = inputs[self.effect_program.data_input_name]
        if result.value is not value:
            raise RuntimeError("NeuralPlasticity data lane must preserve Tensor identity")
        proposal = self._proposal(result.effect, lineage, previous, revision)
        updated = arena.append_proposal(proposal)
        next_lineage = replace(
            lineage,
            output_slot=self.output_slot,
            plastic_revision=proposal.successor_revision,
            plastic_value=proposal.successor,
        )
        return updated.write(self.output_slot, result.value, producer=next_lineage)


FormulaProgramSearchCandidateV4 = (
    FormulaProgramTensorCandidateV3 | FormulaProgramEffectCandidateV3
)


@dataclass(frozen=True)
class FormulaProgramQueryTraceStepV4:
    step: int
    candidate_id: str
    atom_ref: str | None
    input_slots: tuple[str, ...]
    output_slot: str | None
    execution_id: str | None = None
    bank_owner_id: str | None = None
    target_slot: BankSlotRef | None = None
    target_revision: int | None = None


@dataclass(frozen=True)
class FormulaProgramQueryTraceV4:
    steps: tuple[FormulaProgramQueryTraceStepV4, ...]
    stopped: bool


@dataclass(frozen=True)
class FormulaProgramQueryExecutionV4:
    value: Tensor
    bank_state: FormulaProgramBankState
    proposals: tuple[_BankSlotEffectProposalV2, ...]
    trace: FormulaProgramQueryTraceV4
    _owner_token: object


class FormulaProgramQueryV4(nn.Module):
    """Search ordinary and self-operation occurrences with branch-visible state."""

    _component_reference: ClassVar[str] = "arti/formula-program-query@4"

    def __init__(
        self,
        *,
        slot_ids: Sequence[str],
        candidates: Sequence[FormulaProgramSearchCandidateV4],
        terminal_slot: str,
        min_steps: int = 1,
        max_steps: int = 8,
        hidden_dim: int = 64,
        tensor_encoder: FormulaProgramQueryTensorEncoderV1 | None = None,
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
            not isinstance(item, (FormulaProgramTensorCandidateV3, FormulaProgramEffectCandidateV3))
            for item in normalized_candidates
        ):
            raise TypeError("candidates must contain Formula ProgramQuery@4 candidates")
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
        if tensor_encoder is not None and not isinstance(
            tensor_encoder,
            FormulaProgramQueryTensorEncoderV1,
        ):
            raise TypeError("tensor_encoder must be FormulaProgramQueryTensorEncoderV1")

        self.slot_ids = normalized_slots
        self.candidates = nn.ModuleList(normalized_candidates)
        self.terminal_slot = terminal_slot
        self.min_steps = int(min_steps)
        self.max_steps = int(max_steps)
        self.hidden_dim = int(hidden_dim)
        self.tensor_encoder = tensor_encoder
        self._owner_token = object()
        self._bind_shared_bank_owners()
        summary_width = (
            _SUMMARY_WIDTH
            if self.tensor_encoder is None
            else self.tensor_encoder.output_width
        )
        self.network = nn.Sequential(
            nn.Linear(len(normalized_slots) * summary_width, self.hidden_dim),
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
    def plastic_candidates(self) -> tuple[FormulaProgramTensorCandidateV3, ...]:
        return tuple(
            item
            for item in self.candidates
            if isinstance(item, FormulaProgramTensorCandidateV3)
            and item.bank_slot_ref is not None
        )

    @property
    def bank_owners(self) -> tuple[FormulaProgramTensorCandidateV3, ...]:
        owners: dict[BankSlotRef, FormulaProgramTensorCandidateV3] = {}
        for candidate in self.plastic_candidates:
            assert candidate.bank_slot_ref is not None
            owners.setdefault(candidate.bank_slot_ref, candidate)
        return tuple(owners[slot_ref] for slot_ref in sorted(owners))

    def _validate_bank_owners(self) -> None:
        owner_refs: dict[str, BankSlotRef] = {}
        owner_values: dict[BankSlotRef, tuple[Tensor, int]] = {}
        for candidate in self.plastic_candidates:
            slot_ref = candidate.bank_slot_ref
            assert slot_ref is not None
            previous_ref = owner_refs.setdefault(candidate.bank_owner_id, slot_ref)
            if previous_ref != slot_ref:
                raise ValueError("one bank_owner_id must identify one compatible Bank slot")
            value = candidate.initial_bank_value()
            revision = candidate.initial_revision()
            previous = owner_values.setdefault(slot_ref, (value, revision))
            if (
                previous[1] != revision
                or previous[0].shape != value.shape
                or previous[0].dtype != value.dtype
                or previous[0].device != value.device
                or not torch.equal(previous[0], value)
            ):
                raise ValueError("shared Bank owner occurrences must start from identical state")

    def _bind_shared_bank_owners(self) -> None:
        owners: dict[BankSlotRef, FormulaProgramBankOwnerV1] = {}
        owner_ids: dict[str, BankSlotRef] = {}
        for candidate in self.plastic_candidates:
            slot_ref = candidate.bank_slot_ref
            assert slot_ref is not None
            previous_ref = owner_ids.setdefault(candidate.bank_owner_id, slot_ref)
            if previous_ref != slot_ref:
                raise ValueError("one bank_owner_id must identify one compatible Bank slot")
            owner = owners.get(slot_ref)
            if owner is None:
                owners[slot_ref] = candidate.bank_owner
                continue
            value = candidate.bank_owner.value
            if (
                int(owner.revision.detach().cpu())
                != int(candidate.bank_owner.revision.detach().cpu())
                or owner.value.shape != value.shape
                or owner.value.dtype != value.dtype
                or owner.value.device != value.device
                or not torch.equal(owner.value, value)
            ):
                raise ValueError("shared Bank owner occurrences must start from identical state")
            candidate._bind_bank_owner(owner)
        self.owner_states = nn.ModuleList(tuple(owners[slot_ref] for slot_ref in sorted(owners)))
        self._validate_bank_owners()

    def initial_bank_state(self) -> FormulaProgramBankState:
        owners = tuple(self.owner_states)
        return FormulaProgramBankState(
            tuple(item.slot_ref for item in owners),
            tuple(item.value.clone() for item in owners),
            tuple(int(item.revision.detach().cpu()) for item in owners),
        )

    def commit_(self, execution: FormulaProgramQueryExecutionV4) -> None:
        if not isinstance(execution, FormulaProgramQueryExecutionV4):
            raise TypeError("execution must be a FormulaProgramQueryExecutionV4 winner")
        if execution._owner_token is not self._owner_token or not execution.trace.stopped:
            raise ValueError("only this ProgramQuery's stopped winner can be committed")
        for owner in self.owner_states:
            owner.install_(execution.bank_state)

    def reexecute(
        self,
        candidate_id: str,
        values: Mapping[str, Tensor],
        *,
        bank_state: FormulaProgramBankState,
    ) -> Tensor:
        matches = tuple(
            candidate
            for candidate in self.plastic_candidates
            if candidate.candidate_id == candidate_id
        )
        if len(matches) != 1:
            raise ValueError("candidate_id must name one plastic ordinary occurrence")
        arena = self._arena(values, bank_state=bank_state)
        producer = matches[0]
        if not producer.accepts(arena):
            raise ValueError("values do not satisfy the selected ordinary occurrence")
        executed = producer(arena)
        output = executed.values.get(producer.output_slot)
        assert output is not None
        return output

    def contract_config(self) -> dict[str, object]:
        return {
            "slot_ids": list(self.slot_ids),
            "candidate_ids": list(self.candidate_ids),
            "bank_owner_ids": [item.bank_owner_id for item in self.bank_owners],
            "terminal_slot": self.terminal_slot,
            "min_steps": self.min_steps,
            "max_steps": self.max_steps,
            "hidden_dim": self.hidden_dim,
            "tensor_encoder_ref": (
                None
                if self.tensor_encoder is None
                else self.tensor_encoder._component_reference
            ),
            "selection": "hard-one-shape-valid",
            "topology": "query-selected-versioned-ssa-occurrences",
            "effect_target": "dynamic-immediate-predecessor-bank-slot",
            "pending_visibility": "branch-local-proposal-overlay",
            "authoritative_commit": "stopped-hard-winner-only",
        }

    def _arena(
        self,
        values: Mapping[str, Tensor],
        *,
        bank_state: FormulaProgramBankState | None = None,
    ) -> _FormulaProgramExecutionArenaV4:
        state = self.initial_bank_state() if bank_state is None else bank_state
        expected = tuple(owner.slot_ref for owner in self.owner_states)
        if state.slot_refs != expected:
            raise ValueError("bank_state does not match ProgramQuery producer-owned slots")
        arena = FormulaProgramArena.from_mapping(self.slot_ids, values)
        return _FormulaProgramExecutionArenaV4(arena, (None,) * len(self.slot_ids), state)

    def eligible(self, arena: _FormulaProgramExecutionArenaV4, *, steps: int) -> Tensor:
        if arena.values.slot_ids != self.slot_ids:
            raise ValueError("arena layout does not match ProgramQuery@4")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        candidates = [
            steps < self.max_steps and candidate.accepts(arena)
            for candidate in self.candidates
        ]
        stop = steps >= self.min_steps and arena.values.get(self.terminal_slot) is not None
        return torch.tensor((*candidates, stop), dtype=torch.bool, device=arena.device)

    def _summarize(self, arena: _FormulaProgramExecutionArenaV4) -> Tensor:
        return self._summarize_values(arena.values)

    def _summarize_values(self, arena: FormulaProgramArena) -> Tensor:
        parameter = next(self.network.parameters())
        if arena.device != parameter.device:
            raise ValueError("arena and FormulaProgramQuery must share device")
        rows: list[Tensor] = []
        for value in arena.values:
            if value is None:
                width = (
                    _SUMMARY_WIDTH
                    if self.tensor_encoder is None
                    else self.tensor_encoder.output_width
                )
                rows.append(
                    torch.zeros(
                        arena.batch_size,
                        width,
                        device=parameter.device,
                        dtype=parameter.dtype,
                    )
                )
                continue
            if self.tensor_encoder is not None:
                rows.append(self.tensor_encoder(value))
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
                        numeric.new_full(
                            (arena.batch_size, 1),
                            math.log1p(numeric.shape[-1]),
                        ),
                    ),
                    dim=-1,
                )
            )
        return torch.cat(rows, dim=-1)

    def query(
        self,
        arena: _FormulaProgramExecutionArenaV4,
        *,
        steps: int,
    ) -> FormulaProgramQueryResult:
        eligible = self.eligible(arena, steps=steps)
        if not bool(eligible.any()):
            raise RuntimeError("ProgramQuery has no shape-valid candidate or valid stop")
        logits = self.network(self._summarize(arena))
        return FormulaProgramQueryResult(
            logits,
            logits.masked_fill(~eligible.unsqueeze(0), -torch.inf),
            eligible,
        )

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
    ) -> FormulaProgramQueryExecutionV4:
        if not isinstance(values, Mapping):
            raise TypeError("values must be an SSA input mapping")
        arena = self._arena(values, bank_state=bank_state)
        if arena.batch_size != 1:
            raise ValueError("hard ProgramQuery execution currently requires batch size one")
        trace: list[FormulaProgramQueryTraceStepV4] = []
        steps = 0
        while True:
            result = self.query(arena, steps=steps)
            selected = self._hard_index(result.masked_logits)
            if selected == len(self.candidates):
                value = arena.values.get(self.terminal_slot)
                assert value is not None
                trace.append(FormulaProgramQueryTraceStepV4(steps, "stop", None, (), None))
                return FormulaProgramQueryExecutionV4(
                    value,
                    arena.committed_state(),
                    arena.proposals,
                    FormulaProgramQueryTraceV4(tuple(trace), True),
                    self._owner_token,
                )
            candidate = self.candidates[selected]
            arena = candidate(arena)
            proposal = (
                arena.proposals[-1]
                if isinstance(candidate, FormulaProgramEffectCandidateV3)
                else None
            )
            lineage = arena.producer(candidate.output_slot)
            trace.append(
                FormulaProgramQueryTraceStepV4(
                    steps,
                    candidate.candidate_id,
                    candidate.atom_ref,
                    tuple(candidate.input_slots.values()),
                    candidate.output_slot,
                    None if lineage is None else lineage.execution_id,
                    None if lineage is None else lineage.owner_id,
                    None if proposal is None else proposal.target,
                    None if proposal is None else proposal.successor_revision,
                )
            )
            steps += 1


__all__ = [
    "FormulaProgramBankOwnerV1",
    "FormulaProgramEffectCandidateV3",
    "FormulaProgramQueryExecutionV4",
    "FormulaProgramQueryTraceStepV4",
    "FormulaProgramQueryTraceV4",
    "FormulaProgramQueryTensorEncoderV1",
    "FormulaProgramQueryV4",
    "FormulaProgramSearchCandidateV4",
    "FormulaProgramTensorCandidateV3",
]
