"""Branch-visible Formula self-operation over stable producer-owned Bank slots."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import TYPE_CHECKING, ClassVar

import torch
from torch import Tensor, nn
from torch.nn.modules import module as module_runtime

from ._formula_finite_rows import _FiniteTensorRows
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
from .formula_v2 import FormulaBankOperand, FormulaV2Error, TensorType, _validate_tensor_against_type
from .formula_v3 import (
    FormulaEffectProgramV2,
    NEURAL_PLASTICITY_OUTER_V2_ATOM_REF,
    NeuralPlasticityEffectV2,
    apply_neural_plasticity_effect,
)

if TYPE_CHECKING:
    from .formula_program_query_v5 import FormulaProgramQueryTraceV5


_CANDIDATE_STRUCTURE_PLANS: ContextVar[dict | None] = ContextVar(
    "arti_candidate_structure_plans", default=None,
)


@contextmanager
def _candidate_structure_scope() -> Iterator[None]:
    """Reuse declared wiring during one synchronous, fixed-graph search.

    Candidate wiring and action identities stay fixed inside this private scope. Bank values,
    revisions, operands, scores and selected paths remain fully dynamic.
    Plans contain only structural groups/layouts/action indices, not admission verdicts.
    """
    token = _CANDIDATE_STRUCTURE_PLANS.set({})
    try:
        yield
    finally:
        _CANDIDATE_STRUCTURE_PLANS.reset(token)


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
class _BankSlotEffectTransition:
    """An executed Formula's operands, reusable without querying another Bank.

    This is an in-memory autograd record, not a second persistent state. Its
    operands retain the generating branch's computation and execution count.
    """

    effect: NeuralPlasticityEffectV2
    state_type: TensorType
    execution_count: Tensor | None
    max_executions: int

    def apply(self, value: Tensor) -> Tensor:
        return apply_neural_plasticity_effect(
            self.effect, value, state_type=self.state_type,
            execution_count=self.execution_count, max_executions=self.max_executions,
        )


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
    transition: _BankSlotEffectTransition | None = None

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
    tensor_steps: int = 0
    effect_steps: int = 0
    invocation_path: tuple[str, ...] = ()
    call_traces: tuple[FormulaProgramQueryTraceV5, ...] = ()

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

    def execution_id(self, candidate_id: str) -> str:
        return "/".join((*self.invocation_path, candidate_id))

    def write(
        self,
        slot_id: str,
        value: Tensor,
        *,
        producer: _FormulaProducerLineageV2 | None,
        is_effect: bool = False,
    ) -> _FormulaProgramExecutionArenaV4:
        return self.write_many({slot_id: value}, producers={slot_id: producer}, is_effect=is_effect)

    def write_many(
        self,
        outputs: Mapping[str, Tensor],
        *,
        producers: Mapping[str, _FormulaProducerLineageV2 | None],
        is_effect: bool = False,
    ) -> _FormulaProgramExecutionArenaV4:
        if set(outputs) != set(producers):
            raise ValueError("each SSA output must have an explicit producer entry")
        lineage = list(self.producers)
        for slot_id, producer in producers.items():
            index = self.values.slot_ids.index(slot_id)
            lineage[index] = producer
        return replace(
            self,
            values=self.values.write_many(outputs),
            producers=tuple(lineage),
            tensor_steps=self.tensor_steps + int(not is_effect),
            effect_steps=self.effect_steps + int(is_effect),
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
        return replace(self, proposals=(*self.proposals, proposal))

    def committed_state(self) -> FormulaProgramBankState:
        if not self.proposals:
            return self.bank_state
        return self.bank_state._replace_sequence(
            (proposal.target, proposal.successor, proposal.successor_revision)
            for proposal in self.proposals
        )


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


class _ScaledPopulationStd(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(value: Tensor, dim: int):
        # Bound Welford's unnormalized second moment, preserving original units.
        peak = value.detach().abs().amax(dim=dim, keepdim=True)
        limit = math.sqrt(torch.finfo(value.dtype).max / (4 * value.shape[dim]))
        scale = torch.where(peak > limit, peak, torch.ones_like(peak))
        deviation = (value / scale).std(dim=dim, unbiased=False, keepdim=True)
        return (deviation * scale).squeeze(dim), scale, deviation

    @staticmethod
    def setup_context(ctx, inputs, output):
        value, ctx.dim = inputs
        _, scale, deviation = output
        ctx.save_for_backward(value, scale, deviation)
        ctx.mark_non_differentiable(scale, deviation)

    @staticmethod
    def backward(ctx, gradient: Tensor, _scale_gradient, _deviation_gradient) -> tuple[Tensor, None]:
        value, scale, deviation = ctx.saved_tensors
        dim = ctx.dim
        normalized = value / scale
        if torch.is_grad_enabled():
            deviation = normalized.std(dim=dim, unbiased=False, keepdim=True)
        denominator = torch.where(deviation == 0, torch.ones_like(deviation), deviation)
        standardized = (normalized - normalized.mean(dim=dim, keepdim=True)) / denominator
        standardized = torch.where(deviation == 0, torch.zeros_like(standardized), standardized)
        # The forward scale cancels analytically. Multiplying gradient by scale
        # first could overflow even when the correct input gradient is finite.
        return standardized * (gradient.unsqueeze(dim) / value.shape[dim]), None


def _scaled_population_std(value: Tensor, *, dim: int) -> Tensor:
    return _ScaledPopulationStd.apply(value, dim)[0]


class _ScaledRootMeanSquare(torch.autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(value: Tensor, dim: int):
        peak = value.detach().abs().amax(dim=dim, keepdim=True)
        limit = math.sqrt(torch.finfo(value.dtype).max / value.shape[dim])
        scale = torch.where(peak > limit, peak, torch.ones_like(peak))
        rms = (value / scale).square().mean(dim=dim, keepdim=True).sqrt()
        return (rms * scale).squeeze(dim), scale, rms

    @staticmethod
    def setup_context(ctx, inputs, output):
        value, ctx.dim = inputs
        _, scale, rms = output
        ctx.save_for_backward(value, scale, rms)
        ctx.mark_non_differentiable(scale, rms)

    @staticmethod
    def backward(ctx, gradient: Tensor, _scale_gradient, _rms_gradient) -> tuple[Tensor, None]:
        value, scale, rms = ctx.saved_tensors
        dim = ctx.dim
        normalized = value / scale
        if torch.is_grad_enabled():
            rms = normalized.square().mean(dim=dim, keepdim=True).sqrt()
        denominator = torch.where(rms == 0, torch.ones_like(rms), rms)
        # Cancel the scale before multiplying by upstream gradients, as in std.
        return (normalized / denominator) * (gradient.unsqueeze(dim) / value.shape[dim]), None


def _scaled_root_mean_square(value: Tensor, *, dim: int) -> Tensor:
    return _ScaledRootMeanSquare.apply(value, dim)[0]


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

    def _prepare_tokens(self, value: Tensor) -> Tensor:
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
        if tokens.shape[1] == 0:
            raise ValueError("Query tensor encoder input must be non-empty and finite")
        return tokens

    def _encode_tokens(self, tokens: Tensor) -> Tensor:
        encoded = self.network(tokens)
        return torch.cat(
            (
                encoded.new_ones((encoded.shape[0], 1)),
                encoded[:, 0],
                encoded[:, -1],
                encoded.mean(dim=1),
                _scaled_population_std(encoded, dim=1),
                encoded.new_full(
                    (encoded.shape[0], 1),
                    math.log1p(tokens.shape[1]),
                ),
            ),
            dim=-1,
        )

    def forward(self, value: Tensor) -> Tensor:
        tokens = self._prepare_tokens(value)
        if not bool(torch.isfinite(tokens).all()):
            raise ValueError("Query tensor encoder input must be non-empty and finite")
        return self._encode_tokens(tokens)


_NATIVE_ENCODER_METHODS = {
    name: getattr(FormulaProgramQueryTensorEncoderV1, name)
    for name in ("forward", "_prepare_tokens", "_encode_tokens")
}
_NATIVE_ENCODER_FORWARDS = {cls: cls.forward for cls in (nn.Sequential, nn.Linear, nn.SiLU)}


def _native_summary_encoder(encoder: nn.Module | None) -> bool:
    if encoder is None:
        return True
    if type(encoder) is not FormulaProgramQueryTensorEncoderV1 or any(
        name in encoder.__dict__ or getattr(type(encoder), name) is not method
        for name, method in _NATIVE_ENCODER_METHODS.items()
    ):
        return False
    if any((module_runtime._global_forward_hooks, module_runtime._global_forward_pre_hooks,
            module_runtime._global_backward_hooks, module_runtime._global_backward_pre_hooks)):
        return False
    if any(type(parameter) not in (Tensor, nn.Parameter) for parameter in encoder.parameters()):
        return False
    return all(
        (module is encoder or type(module) in _NATIVE_ENCODER_FORWARDS
         and type(module).forward is _NATIVE_ENCODER_FORWARDS[type(module)])
        and not any(getattr(module, name) for name in (
            "_forward_hooks", "_forward_pre_hooks", "_backward_hooks", "_backward_pre_hooks",
        )) and "forward" not in module.__dict__
        for module in encoder.modules()
    )


def _numeric_summary(numeric: Tensor) -> Tensor:
    return torch.cat((
        numeric.new_ones((numeric.shape[0], 1)),
        (numeric / numeric.shape[-1]).sum(dim=-1, keepdim=True),
        _scaled_population_std(numeric, dim=-1).unsqueeze(-1),
        (numeric.abs() / numeric.shape[-1]).sum(dim=-1, keepdim=True),
        numeric.amax(dim=-1, keepdim=True),
        numeric.amin(dim=-1, keepdim=True),
        _scaled_root_mean_square(numeric, dim=-1).unsqueeze(-1),
        numeric.new_full((numeric.shape[0], 1), math.log1p(numeric.shape[-1])),
    ), dim=-1)


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

    @property
    def output_slot_ids(self) -> tuple[str, ...]:
        return (self.output_slot,)

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
        input_values: Mapping[str, Tensor] | None = None,
    ) -> tuple[dict[str, Tensor], dict[str, FormulaBankOperand]]:
        if input_values is not None and set(input_values) != set(self.input_slots):
            raise ValueError("input values must bind every named Formula port")
        inputs: dict[str, Tensor] = {}
        for input_name, slot_id in self.input_slots.items():
            value = arena.values.get(slot_id) if input_values is None else input_values[input_name]
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
        *, _input_values: Mapping[str, Tensor] | None = None,
    ) -> _FormulaProgramExecutionArenaV4:
        if arena.values.get(self.output_slot) is not None:
            raise ValueError(f"SSA output slot {self.output_slot!r} is already occupied")
        inputs, banks = self._bindings(arena) if _input_values is None else self._bindings(arena, _input_values)
        value = self.candidate.fabric(inputs=inputs, banks=banks).values[0]
        return self._finish(arena, value)

    def _finish(
        self, arena: _FormulaProgramExecutionArenaV4, value: Tensor
    ) -> _FormulaProgramExecutionArenaV4:
        slot_ref = self.bank_slot_ref
        current: Tensor | None = None
        revision: int | None = None
        if slot_ref is not None:
            current, revision = arena.effect_state(slot_ref)
        lineage = _FormulaProducerLineageV2(
            arena.execution_id(self.candidate_id),
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

    @property
    def output_slot_ids(self) -> tuple[str, ...]:
        return (self.output_slot,)

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
        *,
        successor: Tensor | None = None,
    ) -> _BankSlotEffectProposalV2:
        assert lineage.plastic_slot is not None
        count = self.execution_count_tensor()
        transition = _BankSlotEffectTransition(
            effect, self.effect_program.state_type,
            None if self.atom_ref == NEURAL_PLASTICITY_OUTER_V2_ATOM_REF else count,
            self.max_executions,
        )
        if successor is None:
            successor = transition.apply(previous)
        _validate_tensor_against_type(
            successor, self.effect_program.state_type,
            name=f"{self.candidate_id}.successor_bank_slot",
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
            transition,
        )

    def forward(
        self,
        arena: _FormulaProgramExecutionArenaV4,
    ) -> _FormulaProgramExecutionArenaV4:
        if arena.values.get(self.output_slot) is not None:
            raise ValueError(f"SSA output slot {self.output_slot!r} is already occupied")
        inputs, banks = self._bindings(arena)  # type: ignore[arg-type]
        target = self._target(arena)
        result = self.fabric._execute_owned(inputs=inputs, banks=banks)
        if result.value is not inputs[self.effect_program.data_input_name]:
            raise RuntimeError("NeuralPlasticity data lane must preserve Tensor identity")
        return self._finish(arena, result.value, result.effect, target=target)

    def _finish(
        self,
        arena: _FormulaProgramExecutionArenaV4,
        value: Tensor,
        effect: NeuralPlasticityEffectV2,
        *,
        target: tuple[_FormulaProducerLineageV2, Tensor, int] | None = None,
        successor: Tensor | None = None,
    ) -> _FormulaProgramExecutionArenaV4:
        lineage, previous, revision = self._target(arena) if target is None else target
        proposal = (
            self._proposal(effect, lineage, previous, revision) if successor is None
            else self._proposal(effect, lineage, previous, revision, successor=successor)
        )
        if arena.invocation_path:
            proposal = replace(proposal, effect_candidate_id=arena.execution_id(self.candidate_id))
        updated = arena.append_proposal(proposal)
        next_lineage = replace(
            lineage,
            output_slot=self.output_slot,
            plastic_revision=proposal.successor_revision,
            plastic_value=proposal.successor,
        )
        return updated.write(self.output_slot, value, producer=next_lineage, is_effect=True)


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
    """Search ordinary and self-operation occurrences with branch-visible state.

    ``max_steps`` bounds non-STOP dispatches. Optional tensor/effect bounds
    count their respective occurrences, not an effect's internal repetitions.
    A populated terminal enables STOP; it does not force the search to stop.
    """

    _component_reference: ClassVar[str] = "arti/formula-program-query@4"
    _allows_multiple_outputs: ClassVar[bool] = False
    _uses_external_query: ClassVar[bool] = True
    _candidate_types: ClassVar[tuple[type[nn.Module], ...]] = (
        FormulaProgramTensorCandidateV3, FormulaProgramEffectCandidateV3,
    )

    def __init__(
        self,
        *,
        slot_ids: Sequence[str],
        candidates: Sequence[FormulaProgramSearchCandidateV4],
        terminal_slot: str,
        min_steps: int = 1,
        max_steps: int = 8,
        min_tensor_steps: int = 0,
        max_tensor_steps: int | None = None,
        max_effect_steps: int | None = None,
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
            not isinstance(item, self._candidate_types)
            for item in normalized_candidates
        ):
            raise TypeError("candidates must contain Formula ProgramQuery@4 candidates")
        candidate_ids = tuple(item.candidate_id for item in normalized_candidates)
        if len(set(candidate_ids)) != len(candidate_ids) or "stop" in candidate_ids:
            raise ValueError("candidate ids must be unique and must not use 'stop'")
        for candidate in normalized_candidates:
            if len(candidate.output_slot_ids) != 1 and not self._allows_multiple_outputs:
                raise ValueError("multiple outputs require ProgramQuery@5")
            referenced = {
                *candidate.input_slots.values(),
                *candidate.output_slot_ids,
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
        for name, limit in (
            ("min_tensor_steps", min_tensor_steps),
            ("max_tensor_steps", max_tensor_steps),
            ("max_effect_steps", max_effect_steps),
        ):
            if limit is not None and (
                isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None")
        if min_tensor_steps is None or (
            max_tensor_steps is not None and min_tensor_steps > max_tensor_steps
        ):
            raise ValueError("tensor step bounds are invalid")
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
        self.min_tensor_steps = min_tensor_steps
        self.max_tensor_steps = max_tensor_steps
        self.max_effect_steps = max_effect_steps
        self._terminal_requires_tensor = all(
            isinstance(item, FormulaProgramTensorCandidateV3)
            for item in normalized_candidates if item.output_slot == terminal_slot
        )
        self._terminal_closes_tensor = all(
            item.output_slot == terminal_slot or terminal_slot in item.requires_empty_slots
            for item in normalized_candidates
            if isinstance(item, FormulaProgramTensorCandidateV3)
        )
        self.hidden_dim = int(hidden_dim)
        self.tensor_encoder = tensor_encoder
        self._owner_token = object()
        self._bind_shared_bank_owners()
        summary_width = (
            _SUMMARY_WIDTH
            if self.tensor_encoder is None
            else self.tensor_encoder.output_width
        )
        if self._uses_external_query:
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
        for owner in self.owner_states:
            owner._is_query_owned = True
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

    def execute_many(
        self,
        requests: Sequence[
            tuple[FormulaProgramSearchCandidateV4, _FormulaProgramExecutionArenaV4]
        ],
        *,
        chunk_size: int | None = None,
        serial: bool = False,
    ) -> tuple[_FormulaProgramExecutionArenaV4, ...]:
        """Execute independent candidate requests, without selecting or committing.

        Under no_grad/inference_mode, compatible pure-tensor programs share
        the existing execution plan. Grad-enabled calls execute each checked plan
        independently so unused candidates keep absent (not zero) gradients.
        Other programs also retain native execution. ``chunk_size`` bounds the
        number of requests stacked together, not the search width or depth.
        Set ``serial=True`` to use the original candidate calls throughout.
        """
        from ._formula_candidate_batch import execute_many

        rows = tuple(requests)
        candidates = {id(candidate) for candidate in self.candidates}
        expected_refs = tuple(owner.slot_ref for owner in self.owner_states)
        for candidate, arena in rows:
            if id(candidate) not in candidates:
                raise ValueError("batch candidate must belong to this ProgramQuery")
            if not isinstance(arena, _FormulaProgramExecutionArenaV4):
                raise TypeError("batch requests require ProgramQuery@4 execution arenas")
            if arena.values.slot_ids != self.slot_ids or arena.bank_state.slot_refs != expected_refs:
                raise ValueError("batch arena does not match this ProgramQuery")
        return execute_many(rows, chunk_size=chunk_size, serial=serial)

    def contract_config(self) -> dict[str, object]:
        return {
            "slot_ids": list(self.slot_ids),
            "candidate_ids": list(self.candidate_ids),
            "bank_owner_ids": [item.bank_owner_id for item in self.bank_owners],
            "terminal_slot": self.terminal_slot,
            "min_steps": self.min_steps,
            "max_steps": self.max_steps,
            "min_tensor_steps": self.min_tensor_steps,
            "max_tensor_steps": self.max_tensor_steps,
            "max_effect_steps": self.max_effect_steps,
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

    def _candidate_eligible(
        self,
        candidate: FormulaProgramSearchCandidateV4,
        arena: _FormulaProgramExecutionArenaV4,
        *,
        steps: int,
    ) -> bool:
        return self._candidate_budget_eligible(candidate, arena, steps=steps) and candidate.accepts(arena)

    def _candidate_budget_eligible(
        self, candidate: FormulaProgramSearchCandidateV4, arena: _FormulaProgramExecutionArenaV4,
        *, steps: int,
    ) -> bool:
        if steps >= self.max_steps:
            return False
        if (
            self._terminal_closes_tensor
            and candidate.output_slot == self.terminal_slot
            and arena.tensor_steps + int(isinstance(candidate, FormulaProgramTensorCandidateV3))
            < self.min_tensor_steps
        ):
            return False
        if isinstance(candidate, FormulaProgramEffectCandidateV3):
            if self.max_effect_steps is not None and arena.effect_steps >= self.max_effect_steps:
                return False
        elif self.max_tensor_steps is not None:
            remaining = self.max_tensor_steps - arena.tensor_steps
            if remaining <= 0:
                return False
            if (
                remaining == 1 and self._terminal_requires_tensor
                and arena.values.get(self.terminal_slot) is None
                and candidate.output_slot != self.terminal_slot
            ):
                return False
        return True

    def _stop_eligible(self, arena: _FormulaProgramExecutionArenaV4, *, steps: int) -> bool:
        return (
            steps >= self.min_steps and arena.tensor_steps >= self.min_tensor_steps
            and arena.values.get(self.terminal_slot) is not None
        )

    def _structural_candidates(
        self, arenas: Sequence[_FormulaProgramExecutionArenaV4],
        candidates: Sequence[FormulaProgramSearchCandidateV4] | None = None,
    ) -> tuple[tuple[int, ...], ...]:
        """Prefilter wiring; never reuse Bank or numerical admission."""
        from .formula_program_query_v5 import FormulaProgramQueryV5
        from .formula_program_query_v6 import FormulaProgramQueryV6

        candidates = tuple(self.candidates if candidates is None else candidates)
        all_indices = tuple(range(len(candidates)))
        if type(self) not in (FormulaProgramQueryV4, FormulaProgramQueryV5, FormulaProgramQueryV6) or "_candidate_eligible" in self.__dict__:
            return (all_indices,) * len(arenas)
        cache = _CANDIDATE_STRUCTURE_PLANS.get()
        key = (self, self.slot_ids, tuple(map(id, candidates)))
        plan = None if cache is None else cache.get(key)
        if plan is None:
            plan = (self._compile_candidate_wiring(candidates), {}, candidates)
            if cache is not None:
                cache[key] = plan
        groups, layouts, _ = plan
        rows = []
        for arena in arenas:
            if arena.values.slot_ids != self.slot_ids:
                raise ValueError("arena layout does not match ProgramQuery@4")
            occupied = sum(1 << i for i, value in enumerate(arena.values.values) if value is not None)
            if occupied not in layouts:
                layouts[occupied] = tuple(sorted(
                    i for (required, empty), indices in groups.items()
                    if occupied & required == required and not occupied & empty
                    for i in indices
                ))
            rows.append(layouts[occupied])
        return tuple(rows)

    def _compile_candidate_wiring(self, candidates):
        from .formula_program_call import FormulaProgramCallCandidateV1
        from .formula_program_query_v5 import FormulaProgramTensorCandidateV4

        builtins = (FormulaProgramTensorCandidateV3, FormulaProgramTensorCandidateV4,
                    FormulaProgramEffectCandidateV3, FormulaProgramCallCandidateV1)
        bits = {slot: 1 << i for i, slot in enumerate(self.slot_ids)}
        groups: dict[tuple[int, int], list[int]] = {}
        for i, candidate in enumerate(candidates):
            if type(candidate) in builtins and not any(
                name in candidate.__dict__ for name in ("accepts", "_bindings", "_entry")
            ):
                try:
                    required = sum(bits[slot] for slot in set(candidate.input_slots.values()))
                    empty = sum(bits[slot] for slot in set((*candidate.output_slot_ids, *candidate.requires_empty_slots)))
                except KeyError:
                    required = empty = 0
            else:
                required = empty = 0
            groups.setdefault((required, empty), []).append(i)
        return groups

    def _has_eligible(self, arena: _FormulaProgramExecutionArenaV4, *, steps: int) -> bool:
        """Existence check only; actual dispatch still scores the entire legal set."""
        if self._stop_eligible(arena, steps=steps):
            return True
        return any(self._candidate_eligible(self.candidates[i], arena, steps=steps)
                   for i in self._structural_candidates((arena,))[0])

    def eligible(self, arena: _FormulaProgramExecutionArenaV4, *, steps: int) -> Tensor:
        from ._formula_candidate_admission import candidate_mask

        if arena.values.slot_ids != self.slot_ids:
            raise ValueError("arena layout does not match ProgramQuery@4")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
            raise ValueError("steps must be a non-negative integer")
        return candidate_mask(self, (arena,), self.candidates, steps=steps, include_stop=True)[0]

    def _summarize(self, arena: _FormulaProgramExecutionArenaV4) -> Tensor:
        return self._summarize_values(arena.values)

    def _summarize_values(self, arena: FormulaProgramArena) -> Tensor:
        parameter = next(self.network.parameters())
        if arena.device != parameter.device:
            raise ValueError("arena and FormulaProgramQuery must share device")
        if _native_summary_encoder(self.tensor_encoder) and all(
            value is None or type(value) in (Tensor, nn.Parameter) and value.layout == torch.strided
            for value in arena.values
        ):
            return self._summarize_native_values(arena, parameter)
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
            rows.append(_numeric_summary(numeric))
        return torch.cat(rows, dim=-1)

    def _summarize_native_values(self, arena: FormulaProgramArena, parameter: Tensor) -> Tensor:
        encoder = self.tensor_encoder
        numeric = tuple(
            None if value is None else
            value.to(dtype=parameter.dtype).reshape(arena.batch_size, -1) if encoder is None else
            encoder._prepare_tokens(value)
            for value in arena.values
        )
        checks = _FiniteTensorRows()
        checks.add(value for value in numeric if value is not None)
        # One boundary read for the complete summary, not one per live SSA slot.
        if not bool(checks.evaluate(device=parameter.device).all()):
            raise ValueError("ProgramQuery arena values must be finite" if encoder is None
                             else "Query tensor encoder input must be non-empty and finite")
        width = _SUMMARY_WIDTH if encoder is None else encoder.output_width
        empty = parameter.new_zeros((arena.batch_size, width))
        rows = [empty] * len(numeric)
        if encoder is not None:
            # Preserve each GEMM shape and parameter-gradient accumulation order.
            for index, value in enumerate(numeric):
                if value is not None:
                    rows[index] = encoder._encode_tokens(value)
        else:
            groups = {}
            for index, value in enumerate(numeric):
                if value is not None:
                    groups.setdefault(value.shape[-1], []).append((index, value))
            for group in groups.values():
                # Bound the temporary copy independently of arena capacity.
                count = max(1, 262144 // max(1, group[0][1].numel()))
                for start in range(0, len(group), count):
                    chunk = group[start:start + count]
                    packed = chunk[0][1] if len(chunk) == 1 else torch.cat(tuple(value for _, value in chunk))
                    summaries = _numeric_summary(packed).split(arena.batch_size)
                    for (index, _), summary in zip(chunk, summaries, strict=True):
                        rows[index] = summary
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
        logits = self.query_logits(arena)
        return FormulaProgramQueryResult(
            logits,
            logits.masked_fill(~eligible.unsqueeze(0), -torch.inf),
            eligible,
        )

    def query_logits(self, arena: _FormulaProgramExecutionArenaV4) -> Tensor:
        """Produce scores without dispatching a candidate or applying an effect."""
        return self.network(self._summarize(arena))

    def routing_mask(self, arena: _FormulaProgramExecutionArenaV4, *, steps: int) -> Tensor:
        return torch.ones(len(self.action_ids), dtype=torch.bool, device=arena.device)

    def _hard_index(self, masked_logits: Tensor) -> int:
        if masked_logits.shape[0] != 1:
            raise ValueError("hard ProgramQuery execution currently requires batch size one")
        maxima = masked_logits[0] == masked_logits[0].max()
        sentinel = torch.full_like(self._action_priority, len(self.action_ids))
        priority = torch.where(maxima, self._action_priority, sentinel)
        return int(priority.argmin().item())

    def _walk(
        self,
        values: Mapping[str, Tensor],
        *,
        bank_state: FormulaProgramBankState | None = None,
    ) -> Iterator[tuple[int, FormulaProgramSearchCandidateV4 | None, _FormulaProgramExecutionArenaV4]]:
        if not isinstance(values, Mapping):
            raise TypeError("values must be an SSA input mapping")
        arena = self._arena(values, bank_state=bank_state)
        yield from self._walk_arena(arena)

    def _walk_arena(
        self, arena: _FormulaProgramExecutionArenaV4,
    ) -> Iterator[tuple[int, FormulaProgramSearchCandidateV4 | None, _FormulaProgramExecutionArenaV4]]:
        if arena.batch_size != 1:
            raise ValueError("hard ProgramQuery execution currently requires batch size one")
        steps = 0
        while True:
            result = self.query(arena, steps=steps)
            selected = self._hard_index(result.masked_logits)
            if selected == len(self.candidates):
                yield steps, None, arena
                return
            candidate = self.candidates[selected]
            arena = candidate(arena)
            yield steps, candidate, arena
            steps += 1

    def forward(
        self,
        values: Mapping[str, Tensor],
        *,
        bank_state: FormulaProgramBankState | None = None,
    ) -> FormulaProgramQueryExecutionV4:
        trace: list[FormulaProgramQueryTraceStepV4] = []
        for steps, candidate, arena in self._walk(values, bank_state=bank_state):
            if candidate is None:
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
        raise RuntimeError("ProgramQuery execution ended without STOP")


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
