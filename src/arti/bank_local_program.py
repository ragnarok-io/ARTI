"""Declarative Bank-local Formula programs and final-loss route training."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import re
from typing import ClassVar, Literal

import torch
from torch import Tensor, nn

from .bank_query import BankQuery, BankQueryResult, SealedBankQuery
from .federal_recall import (
    BankLocalRefinePolicy,
    BankOwnedQueryProgram,
    FederalBankStep,
    FederalCandidate,
    FederalRecallError,
)
from .formula_v2 import (
    BankBinding,
    FormulaFabricV2,
    FormulaProgram,
    InputBinding,
    _validate_tensor_against_type,
)
from .formula_v3 import (
    FormulaEffectProgramV2,
    FormulaFabricV4,
    apply_neural_plasticity_effect,
)
from .formula_program_query_v3 import BankSlotRef, FormulaProgramBankState
from .refine_exit import FormulaRefineExit
from .tensor_schema import GradientContract, ShapeRelation, TensorSchema, TensorSchemaError
from .terminal_abi import BankExecutionSignatureV2, TerminalOutputABI


BankLocalActionKind = Literal["continue", "descend"]
_NAME = re.compile(r"^[a-z][a-z0-9]*(?:[-_.][a-z0-9]+)*$")


def _require_name(value: str, *, field: str) -> None:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise ValueError(f"{field} must be a canonical lowercase name")


def _tensor_contract(value: Tensor, *, trainable: bool) -> dict[str, object]:
    return {
        "dtype": str(value.dtype),
        "shape": list(value.shape),
        "trainable": trainable,
    }


class _FormulaOperandStore(nn.Module):
    def __init__(
        self,
        bindings: Mapping[str, BankBinding],
        operands: Mapping[str, Tensor],
        *,
        trainable: Sequence[str],
    ) -> None:
        super().__init__()
        names = tuple(sorted(bindings))
        if set(operands) != set(names):
            raise ValueError("operands must bind every Formula Bank input exactly once")
        trainable_names = frozenset(trainable)
        if not trainable_names.issubset(names):
            raise ValueError("trainable operands must name declared Formula Bank inputs")
        self.names = names
        self.trainable_names = trainable_names
        self._attributes: dict[str, str] = {}
        for index, name in enumerate(names):
            value = operands[name]
            if not isinstance(value, Tensor):
                raise TypeError("Formula operands must be Tensor values")
            attribute = f"operand_{index:04d}"
            self._attributes[name] = attribute
            cloned = value.detach().clone()
            if name in trainable_names:
                if not (cloned.is_floating_point() or cloned.is_complex()):
                    raise TypeError("trainable Formula operands must be floating or complex")
                self.register_parameter(attribute, nn.Parameter(cloned))
            else:
                self.register_buffer(attribute, cloned, persistent=True)

    def tensors(self) -> dict[str, Tensor]:
        return {name: getattr(self, self._attributes[name]) for name in self.names}

    def tensor(self, name: str) -> Tensor:
        try:
            return getattr(self, self._attributes[name])
        except KeyError as exc:
            raise KeyError(f"unknown Formula Bank operand {name!r}") from exc

    def install_(self, name: str, value: Tensor) -> None:
        current = self.tensor(name)
        if (
            not isinstance(value, Tensor)
            or value.shape != current.shape
            or value.dtype != current.dtype
            or value.device != current.device
        ):
            raise ValueError("installed Formula Bank operand must exactly match its slot")
        with torch.no_grad():
            current.copy_(value.detach())

    def tensors_for(
        self,
        value: Tensor,
        *,
        batch_broadcast: frozenset[str],
    ) -> dict[str, Tensor]:
        result = self.tensors()
        for name in batch_broadcast:
            operand = result[name]
            if operand.ndim < 1 or operand.shape[0] != 1:
                raise ValueError(
                    f"batch-broadcast operand {name!r} must have a leading singleton axis"
                )
            result[name] = operand.expand(value.shape[0], *operand.shape[1:])
        return result

    def contract_config(self) -> dict[str, object]:
        return {
            name: _tensor_contract(
                getattr(self, self._attributes[name]),
                trainable=name in self.trainable_names,
            )
            for name in self.names
        }


class BankLocalFormulaAction(nn.Module):
    """One typed Formula transition selectable by a Bank-owned Query."""

    _component_reference: ClassVar[str] = "arti/bank-local-formula-action@1"

    def __init__(
        self,
        action_id: str,
        program: FormulaProgram,
        *,
        input_schema: TensorSchema,
        output_schema: TensorSchema,
        operands: Mapping[str, Tensor],
        result_kind: BankLocalActionKind = "continue",
        next_bank_id: str | None = None,
        input_name: str = "value",
        output_index: int = 0,
        trainable_operands: Sequence[str] = (),
        batch_broadcast_operands: Sequence[str] = (),
        plastic_bank_slot: str | None = None,
    ) -> None:
        super().__init__()
        _require_name(action_id, field="action_id")
        if not isinstance(program, FormulaProgram):
            raise TypeError("program must be FormulaProgram")
        if not isinstance(input_schema, TensorSchema) or not isinstance(
            output_schema, TensorSchema
        ):
            raise TypeError("action schemas must be TensorSchema values")
        if result_kind not in {"continue", "descend"}:
            raise ValueError("result_kind must be 'continue' or 'descend'")
        if result_kind == "descend":
            _require_name(next_bank_id, field="next_bank_id")
        elif next_bank_id is not None:
            raise ValueError("continue actions cannot declare next_bank_id")
        if isinstance(output_index, bool) or not isinstance(output_index, int):
            raise TypeError("output_index must be an integer")
        if output_index < 0 or output_index >= len(program.outputs):
            raise ValueError("output_index is outside FormulaProgram outputs")

        input_bindings = tuple(
            binding for binding in program.bindings if isinstance(binding, InputBinding)
        )
        if len(input_bindings) != 1 or input_bindings[0].name != input_name:
            raise ValueError(
                "BankLocalFormulaAction@1 requires exactly one current-state InputBinding"
            )
        bank_bindings = {
            binding.name: binding
            for binding in program.bindings
            if isinstance(binding, BankBinding)
        }
        self.action_id = action_id
        self.input_schema = input_schema
        self.output_schema = output_schema
        self.result_kind = result_kind
        self.next_bank_id = next_bank_id
        self.input_name = input_name
        self.output_index = int(output_index)
        self.fabric = FormulaFabricV2(program)
        self._bank_bindings = bank_bindings
        self.operand_store = _FormulaOperandStore(
            bank_bindings,
            operands,
            trainable=tuple(trainable_operands),
        )
        if plastic_bank_slot is not None:
            _require_name(plastic_bank_slot, field="plastic_bank_slot")
            if plastic_bank_slot not in bank_bindings:
                raise ValueError("plastic_bank_slot must name a Formula Bank input")
            if plastic_bank_slot in self.operand_store.trainable_names:
                raise ValueError(
                    "plastic_bank_slot must be forward-written state, not a trainable operand"
                )
            output_slot = program.outputs[output_index]
            output_instruction = next(
                item for item in program.instructions if item.output_slot == output_slot
            )
            if plastic_bank_slot not in output_instruction.input_slots:
                raise ValueError(
                    "plastic_bank_slot must be consumed by the producer output instruction"
                )
        self.plastic_bank_slot = plastic_bank_slot
        self.register_buffer(
            "bank_revision",
            torch.zeros((), dtype=torch.int64),
            persistent=True,
        )
        broadcast_names = frozenset(batch_broadcast_operands)
        if not broadcast_names.issubset(bank_bindings):
            raise ValueError(
                "batch_broadcast_operands must name declared Formula Bank inputs"
            )
        for name in broadcast_names:
            operand = self.operand_store.tensors()[name]
            if operand.ndim < 1 or operand.shape[0] != 1:
                raise ValueError(
                    f"batch-broadcast operand {name!r} must have a leading singleton axis"
                )
        self.batch_broadcast_operands = broadcast_names

    @property
    def program(self) -> FormulaProgram:
        return self.fabric.program

    def contract_config(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "program_fingerprint": self.program.fingerprint,
            "input_schema": self.input_schema.to_dict(),
            "output_schema": self.output_schema.to_dict(),
            "result_kind": self.result_kind,
            "next_bank_id": self.next_bank_id,
            "input_name": self.input_name,
            "output_index": self.output_index,
            "operands": self.operand_store.contract_config(),
            "batch_broadcast_operands": sorted(self.batch_broadcast_operands),
            "plastic_bank_slot": self.plastic_bank_slot,
        }

    def bank_slot_ref(self, owner_bank_id: str) -> BankSlotRef | None:
        if self.plastic_bank_slot is None:
            return None
        _require_name(owner_bank_id, field="owner_bank_id")
        binding = self._bank_bindings[self.plastic_bank_slot]
        payload = {
            "owner_bank_id": owner_bank_id,
            "action_id": self.action_id,
            "program_fingerprint": self.program.fingerprint,
            "input_name": self.input_name,
            "output_index": self.output_index,
            "plastic_bank_slot": self.plastic_bank_slot,
        }
        fingerprint = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return BankSlotRef(
            f"{owner_bank_id}.{self.action_id}",
            fingerprint,
            binding.name,
            binding.source_ref,
            binding.partition_id,
            binding.asset_fingerprint,
        )

    def initial_bank_value(self) -> Tensor:
        if self.plastic_bank_slot is None:
            raise RuntimeError("action has no plastic Bank slot")
        return self.operand_store.tensor(self.plastic_bank_slot).detach().clone()

    def initial_revision(self) -> int:
        return int(self.bank_revision.detach().cpu())

    def install_(self, owner_bank_id: str, state: FormulaProgramBankState) -> None:
        slot_ref = self.bank_slot_ref(owner_bank_id)
        if slot_ref is None:
            return
        self.operand_store.install_(slot_ref.binding_name, state.value(slot_ref))
        with torch.no_grad():
            self.bank_revision.fill_(state.revision(slot_ref))

    def accepts(self, value: Tensor) -> bool:
        try:
            self.input_schema.validate_tensor(value, name=f"{self.action_id}.input")
        except TensorSchemaError:
            return False
        return True

    def _execute_with_operand_overrides(
        self,
        value: Tensor,
        overrides: Mapping[str, Tensor] | None = None,
    ) -> Tensor:
        self.input_schema.validate_tensor(value, name=f"{self.action_id}.input")
        tensors = self.operand_store.tensors_for(
            value,
            batch_broadcast=self.batch_broadcast_operands,
        )
        overrides = {} if overrides is None else dict(overrides)
        unknown = set(overrides).difference(self._bank_bindings)
        if unknown:
            raise ValueError(
                "Formula operand overrides name undeclared Bank inputs: "
                + ", ".join(sorted(unknown))
            )
        tensors.update(overrides)
        result = self.fabric(
            inputs={self.input_name: value},
            banks={
                name: binding.bind(tensors[name])
                for name, binding in self._bank_bindings.items()
            },
        ).values[self.output_index]
        self.output_schema.validate_tensor(result, name=f"{self.action_id}.output")
        return result

    def _execute_from_bank_state(
        self,
        value: Tensor,
        *,
        owner_bank_id: str,
        bank_state: FormulaProgramBankState,
    ) -> Tensor:
        slot_ref = self.bank_slot_ref(owner_bank_id)
        if slot_ref is None:
            return self._execute_with_operand_overrides(value)
        return self._execute_with_operand_overrides(
            value,
            {slot_ref.binding_name: bank_state.value(slot_ref)},
        )

    def forward(self, value: Tensor) -> Tensor:
        return self._execute_with_operand_overrides(value)


@dataclass(frozen=True)
class BankLocalFormulaEffectResult:
    """Identity data plus a successor for a runtime-resolved predecessor slot."""

    value: Tensor
    successor: Tensor
    update: Tensor
    previous_revision: int
    successor_revision: int


class BankLocalFormulaEffectAction(nn.Module):
    """A write-only Formula effect with no owned state or caller target."""

    _component_reference: ClassVar[str] = "arti/bank-local-formula-effect-action@1"

    def __init__(
        self,
        action_id: str,
        effect_program: FormulaEffectProgramV2,
        *,
        input_schema: TensorSchema,
        operands: Mapping[str, Tensor],
        result_kind: BankLocalActionKind = "continue",
        next_bank_id: str | None = None,
        trainable_operands: Sequence[str] = (),
        batch_broadcast_operands: Sequence[str] = (),
    ) -> None:
        super().__init__()
        _require_name(action_id, field="action_id")
        if not isinstance(effect_program, FormulaEffectProgramV2):
            raise TypeError("effect_program must be FormulaEffectProgramV2")
        if not isinstance(input_schema, TensorSchema):
            raise TypeError("input_schema must be TensorSchema")
        if result_kind not in {"continue", "descend"}:
            raise ValueError("result_kind must be 'continue' or 'descend'")
        if result_kind == "descend":
            _require_name(next_bank_id, field="next_bank_id")
        elif next_bank_id is not None:
            raise ValueError("continue actions cannot declare next_bank_id")

        program = effect_program.program
        input_bindings = tuple(
            binding for binding in program.bindings if isinstance(binding, InputBinding)
        )
        if len(input_bindings) != 1 or input_bindings[0].name != effect_program.data_input_name:
            raise ValueError("Formula effect action requires one current-data InputBinding")
        bank_bindings = {
            binding.name: binding
            for binding in program.bindings
            if isinstance(binding, BankBinding)
        }
        self.action_id = action_id
        self.input_schema = input_schema
        self.output_schema = input_schema
        self.result_kind = result_kind
        self.next_bank_id = next_bank_id
        self.fabric = FormulaFabricV4(effect_program)
        self._bank_bindings = bank_bindings
        self.operand_store = _FormulaOperandStore(
            bank_bindings,
            operands,
            trainable=tuple(trainable_operands),
        )
        broadcast_names = frozenset(batch_broadcast_operands)
        if not broadcast_names.issubset(bank_bindings):
            raise ValueError("batch_broadcast_operands must name Formula Bank inputs")
        self.batch_broadcast_operands = broadcast_names

    @property
    def effect_program(self) -> FormulaEffectProgramV2:
        return self.fabric.effect_program

    def contract_config(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "effect_program_fingerprint": self.effect_program.fingerprint,
            "input_schema": self.input_schema.to_dict(),
            "result_kind": self.result_kind,
            "next_bank_id": self.next_bank_id,
            "operands": self.operand_store.contract_config(),
            "batch_broadcast_operands": sorted(self.batch_broadcast_operands),
            "target_resolution": "dynamic-immediate-predecessor-bank-slot",
            "data_lane": "identity",
            "pending_visibility": "write-only-until-winner-commit",
        }

    def accepts(self, value: Tensor) -> bool:
        try:
            self.input_schema.validate_tensor(value, name=f"{self.action_id}.input")
        except TensorSchemaError:
            return False
        return True

    def _execute_against(
        self,
        value: Tensor,
        predecessor: Tensor,
        *,
        previous_revision: int,
    ) -> BankLocalFormulaEffectResult:
        self.input_schema.validate_tensor(value, name=f"{self.action_id}.input")
        _validate_tensor_against_type(
            predecessor,
            self.effect_program.state_type,
            name=f"{self.action_id}.predecessor_bank_slot",
        )
        tensors = self.operand_store.tensors_for(
            value,
            batch_broadcast=self.batch_broadcast_operands,
        )
        result = self.fabric._execute_owned(
            inputs={self.effect_program.data_input_name: value},
            banks={
                name: binding.bind(tensors[name])
                for name, binding in self._bank_bindings.items()
            },
        )
        if result.value is not value:
            raise FederalRecallError("Formula effect data lane must preserve Tensor identity")
        successor = apply_neural_plasticity_effect(
            result.effect,
            predecessor,
            state_type=self.effect_program.state_type,
        )
        return BankLocalFormulaEffectResult(
            value,
            successor,
            successor - predecessor,
            previous_revision,
            previous_revision + 1,
        )

    def forward(self, value: Tensor) -> BankLocalFormulaEffectResult:
        raise RuntimeError(
            "Formula effects require runtime-resolved ordinary predecessor lineage"
        )

class ValueTerminalAdapter(nn.Module):
    """Adapt one tensor to the common value/validity/score terminal ABI."""

    _component_reference: ClassVar[str] = "arti/value-terminal-adapter@1"

    def __init__(
        self,
        *,
        value_field: str = "value",
        validity_field: str = "validity",
        score_field: str = "score",
    ) -> None:
        super().__init__()
        for value, field in (
            (value_field, "value_field"),
            (validity_field, "validity_field"),
            (score_field, "score_field"),
        ):
            _require_name(value, field=field)
        if len({value_field, validity_field, score_field}) != 3:
            raise ValueError("terminal field names must be unique")
        self.value_field = value_field
        self.validity_field = validity_field
        self.score_field = score_field

    def contract_config(self) -> dict[str, object]:
        return {
            "value_field": self.value_field,
            "validity_field": self.validity_field,
            "score_field": self.score_field,
        }

    def forward(self, value: Tensor, local_log_score: Tensor) -> dict[str, Tensor]:
        if not isinstance(value, Tensor) or value.ndim < 1:
            raise TypeError("terminal value must preserve a batch dimension")
        if not isinstance(local_log_score, Tensor) or local_log_score.numel() != 1:
            raise TypeError("local_log_score must be one scalar Tensor")
        batch = value.shape[0]
        return {
            self.value_field: value,
            self.validity_field: torch.ones(
                batch, dtype=torch.bool, device=value.device
            ),
            self.score_field: local_log_score.exp().reshape(1).expand(batch),
        }


class BankLocalTerminalAction(nn.Module):
    """A neural Refine exit action with an explicit terminal adapter."""

    _component_reference: ClassVar[str] = "arti/bank-local-terminal-action@1"

    def __init__(
        self,
        action_id: str,
        *,
        input_schema: TensorSchema,
        adapter: nn.Module | None = None,
        exit_atom: FormulaRefineExit | None = None,
    ) -> None:
        super().__init__()
        _require_name(action_id, field="action_id")
        if not isinstance(input_schema, TensorSchema):
            raise TypeError("input_schema must be TensorSchema")
        if adapter is not None and not isinstance(adapter, nn.Module):
            raise TypeError("adapter must be an nn.Module")
        if exit_atom is not None and not isinstance(exit_atom, FormulaRefineExit):
            raise TypeError("exit_atom must be FormulaRefineExit")
        self.action_id = action_id
        self.input_schema = input_schema
        self.adapter = ValueTerminalAdapter() if adapter is None else adapter
        self.exit_atom = (
            FormulaRefineExit(input_kind="logit", scope="branch")
            if exit_atom is None
            else exit_atom
        )
        if self.exit_atom.input_kind != "logit" or self.exit_atom.scope != "branch":
            raise ValueError("Bank-local terminal actions require a branch logit exit atom")

    def contract_config(self) -> dict[str, object]:
        from .component_registry import component_ref

        return {
            "action_id": self.action_id,
            "input_schema": self.input_schema.to_dict(),
            "adapter_ref": component_ref(self.adapter),
            "exit_atom_ref": component_ref(self.exit_atom),
        }

    def accepts(self, value: Tensor) -> bool:
        try:
            self.input_schema.validate_tensor(value, name=f"{self.action_id}.input")
        except TensorSchemaError:
            return False
        return True

    def requested(self, exit_logit: Tensor, formula_logits: Tensor, value: Tensor) -> bool:
        if exit_logit.ndim != 1 or formula_logits.ndim != 2:
            raise ValueError("Bank-local action logits must preserve batch dimensions")
        if exit_logit.shape[0] != value.shape[0] or formula_logits.shape[0] != value.shape[0]:
            raise ValueError("Bank-local action logits must match the current batch")
        margin = exit_logit - formula_logits.amax(dim=-1)
        tokens = value.shape[1] if value.ndim >= 2 else 1
        request = self.exit_atom(
            margin,
            mask=torch.ones(
                value.shape[0], tokens, dtype=torch.bool, device=value.device
            ),
        )
        return bool(request.requested.all().detach())

    def forward(self, value: Tensor, local_log_score: Tensor) -> Mapping[str, Tensor]:
        self.input_schema.validate_tensor(value, name=f"{self.action_id}.input")
        outputs = self.adapter(value, local_log_score)
        if not isinstance(outputs, Mapping) or any(
            not isinstance(name, str) or not isinstance(item, Tensor)
            for name, item in outputs.items()
        ):
            raise TypeError("terminal adapter must return a Tensor mapping")
        return outputs


class BankLocalFormulaProgram(BankOwnedQueryProgram):
    """Execute declarative Formula actions selected from the latest local state."""

    _component_reference: ClassVar[str] = "arti/bank-local-formula-program@1"

    def __init__(
        self,
        *,
        bank_id: str,
        query: SealedBankQuery,
        actions: Sequence[BankLocalFormulaAction],
        terminal_action: BankLocalTerminalAction,
        local_refine: BankLocalRefinePolicy,
        output_schema: TensorSchema,
        terminal_abi: TerminalOutputABI,
    ) -> None:
        if not isinstance(local_refine, BankLocalRefinePolicy):
            raise TypeError("local_refine must be BankLocalRefinePolicy")
        super().__init__(bank_id=bank_id, query=query, local_refine=local_refine)
        normalized = tuple(actions)
        if not normalized or any(
            not isinstance(action, BankLocalFormulaAction) for action in normalized
        ):
            raise TypeError("actions must contain BankLocalFormulaAction values")
        if not isinstance(terminal_action, BankLocalTerminalAction):
            raise TypeError("terminal_action must be BankLocalTerminalAction")
        if not isinstance(output_schema, TensorSchema):
            raise TypeError("output_schema must be TensorSchema")
        if not isinstance(terminal_abi, TerminalOutputABI):
            raise TypeError("terminal_abi must be TerminalOutputABI")
        action_ids = tuple(action.action_id for action in normalized) + (
            terminal_action.action_id,
        )
        if len(action_ids) != len(set(action_ids)):
            raise ValueError("Bank-local action ids must be unique")
        declared_members = tuple(query.signature.retrieval_contract.get("member_ids", ()))
        if declared_members != action_ids:
            raise ValueError(
                "sealed Query member_ids must exactly match Formula and terminal actions"
            )
        self.actions = nn.ModuleList(normalized)
        self.terminal_action = terminal_action
        self.output_schema = output_schema
        self.terminal_abi = terminal_abi

        from .component_registry import component_ref

        signature = BankExecutionSignatureV2.from_program(
            self,
            input_schema=query.signature.input_schema,
            output_schema=output_schema,
            shape_relation=ShapeRelation.maps_shape(
                "Bank-local Formula actions may change shape before exit or descent"
            ),
            query_signature=query.signature,
            local_normalization_contract=query.signature.normalization_contract,
            local_formula_ref="arti/formula-fabric@2",
            local_refine_ref=component_ref(local_refine),
            terminal_adapter_ref=component_ref(terminal_action.adapter),
            terminal_abi_ref="arti/terminal-output-abi@1",
            terminal_abi_fingerprint=terminal_abi.fingerprint,
            score_contract="sum of Bank-local action log probabilities",
            gradient_contract=GradientContract.autograd(),
            execution_capabilities=("eager", "fixed-k-wide", "latest-state-requery"),
        )
        self.bind_signature(signature)

    @property
    def action_ids(self) -> tuple[str, ...]:
        return tuple(action.action_id for action in self.actions) + (
            self.terminal_action.action_id,
        )

    def contract_config(self) -> dict[str, object]:
        return {
            "bank_id": self.bank_id,
            "actions": [action.contract_config() for action in self.actions],
            "terminal_action": self.terminal_action.contract_config(),
            "local_refine": self.local_refine.contract_config(),
            "output_schema": self.output_schema.to_dict(),
            "terminal_abi_fingerprint": self.terminal_abi.fingerprint,
        }

    def execute(
        self,
        value: Tensor,
        *,
        query_result: BankQueryResult,
        max_candidates: int,
    ) -> FederalBankStep:
        if value.shape[0] != 1:
            raise FederalRecallError(
                "BankLocalFormulaProgram@1 requires one sample per Bank invocation"
            )
        if (
            isinstance(max_candidates, bool)
            or not isinstance(max_candidates, int)
            or max_candidates <= 0
        ):
            raise FederalRecallError("max_candidates must be a positive integer")
        logits = query_result.value
        if logits.shape != (1, len(self.action_ids)):
            raise FederalRecallError("Bank-local Query returned the wrong action shape")
        log_probability = logits.log_softmax(dim=-1)
        formula_logits = logits[:, : len(self.actions)]
        exit_logit = logits[:, len(self.actions)]
        exit_requested = self.terminal_action.requested(
            exit_logit, formula_logits, value
        )
        if max_candidates == 1 and exit_requested:
            score = log_probability[0, len(self.actions)]
            candidate = FederalCandidate.terminal(
                self.terminal_action.action_id,
                local_log_score=score,
                outputs=self.terminal_action(value, score),
            )
            return FederalBankStep((candidate,))

        eligible = [
            index
            for index, action in enumerate(self.actions)
            if action.accepts(value)
        ]
        if (
            (max_candidates > 1 or exit_requested)
            and self.terminal_action.accepts(value)
        ):
            eligible.append(len(self.actions))
        if not eligible:
            raise FederalRecallError(
                "no Bank-local Formula or terminal action accepts the current tensor"
            )
        eligible.sort(
            key=lambda index: (
                -float(log_probability[0, index].detach().cpu()),
                self.action_ids[index],
            )
        )
        candidates: list[FederalCandidate] = []
        for action_index in eligible[:max_candidates]:
            score = log_probability[0, action_index]
            if action_index == len(self.actions):
                candidates.append(
                    FederalCandidate.terminal(
                        self.terminal_action.action_id,
                        local_log_score=score,
                        outputs=self.terminal_action(value, score),
                    )
                )
                continue
            action = self.actions[action_index]
            next_value = action(value)
            if action.result_kind == "continue":
                candidates.append(
                    FederalCandidate.local(
                        action.action_id,
                        local_log_score=score,
                        next_value=next_value,
                    )
                )
            else:
                assert action.next_bank_id is not None
                candidates.append(
                    FederalCandidate.child(
                        action.action_id,
                        local_log_score=score,
                        next_bank_id=action.next_bank_id,
                        next_value=next_value,
                    )
                )
        return FederalBankStep(tuple(candidates))


BankLocalTaskLoss = Callable[[Mapping[str, Tensor], object], Tensor]
BankLocalStepTaskLoss = Callable[[Tensor, object, int], Tensor]
BankLocalDescendLoss = Callable[[str, Tensor, object], Tensor]


@dataclass(frozen=True)
class BankLocalProgramTrainingLoss:
    """Exact expected final task loss over a bounded local Formula graph."""

    total: Tensor
    task: Tensor
    invalid: Tensor
    success_probability: Tensor
    visited_states: int
    per_row_total: Tensor


class ExactBankLocalProgramTraining:
    """Train a draft Query from final loss without route or step teachers."""

    _component_reference: ClassVar[str] = "arti/exact-bank-local-program-training@1"

    def __init__(self, *, invalid_weight: float = 2.0, max_states: int = 256) -> None:
        if (
            isinstance(invalid_weight, bool)
            or not isinstance(invalid_weight, (int, float))
            or not torch.isfinite(torch.tensor(float(invalid_weight)))
            or invalid_weight < 0
        ):
            raise ValueError("invalid_weight must be finite and non-negative")
        if isinstance(max_states, bool) or not isinstance(max_states, int) or max_states <= 0:
            raise ValueError("max_states must be a positive integer")
        self.invalid_weight = float(invalid_weight)
        self.max_states = int(max_states)

    def contract_config(self) -> dict[str, object]:
        return {
            "invalid_weight": self.invalid_weight,
            "max_states": self.max_states,
            "supervision": "final-task-loss-only",
            "route_teacher": False,
            "estimator": "exact-expected-policy",
            "descend_credit": "downstream-final-task-loss",
        }

    def loss(
        self,
        query: BankQuery,
        *,
        actions: Sequence[BankLocalFormulaAction],
        terminal_action: BankLocalTerminalAction,
        initial: Tensor,
        target: object,
        task_loss: BankLocalTaskLoss,
        min_steps: int,
        max_steps: int,
        descend_loss: BankLocalDescendLoss | None = None,
    ) -> BankLocalProgramTrainingLoss:
        if not isinstance(query, BankQuery):
            raise TypeError("query must be an unsealed BankQuery")
        normalized = tuple(actions)
        if not normalized or any(
            not isinstance(action, BankLocalFormulaAction) for action in normalized
        ):
            raise TypeError("actions must contain BankLocalFormulaAction values")
        if any(action.result_kind == "descend" for action in normalized) and descend_loss is None:
            raise ValueError("descend actions require a downstream final-task-loss evaluator")
        if not isinstance(terminal_action, BankLocalTerminalAction):
            raise TypeError("terminal_action must be BankLocalTerminalAction")
        if (
            isinstance(min_steps, bool)
            or not isinstance(min_steps, int)
            or min_steps <= 0
            or isinstance(max_steps, bool)
            or not isinstance(max_steps, int)
            or max_steps < min_steps
        ):
            raise ValueError("training refine bounds are invalid")
        action_ids = tuple(action.action_id for action in normalized) + (
            terminal_action.action_id,
        )
        declared_members = tuple(query.retrieval_contract.get("member_ids", ()))
        if declared_members != action_ids:
            raise ValueError("draft Query member_ids must match the training action graph")

        batch = initial.shape[0]
        expected_task = initial.new_zeros((batch,))
        success_probability = initial.new_zeros((batch,))
        visited_states = 0

        def visit(state: Tensor, probability: Tensor, depth: int) -> None:
            nonlocal expected_task, success_probability, visited_states
            visited_states += 1
            if visited_states > self.max_states:
                raise RuntimeError("exact Bank-local path graph exceeded max_states")
            logits = query(state).value
            if logits.shape != (batch, len(action_ids)):
                raise ValueError("draft Query returned the wrong action shape")
            action_probability = logits.softmax(dim=-1)

            if depth >= min_steps and terminal_action.accepts(state):
                terminal_probability = probability * action_probability[:, -1]
                outputs = terminal_action(state, state.new_zeros(()))
                row_loss = task_loss(outputs, target)
                if (
                    not isinstance(row_loss, Tensor)
                    or not row_loss.is_floating_point()
                    or row_loss.shape != (batch,)
                    or row_loss.device != initial.device
                ):
                    raise TypeError("task_loss must return floating [B] on the task device")
                expected_task = expected_task + terminal_probability * row_loss
                success_probability = success_probability + terminal_probability

            for index, action in enumerate(normalized):
                if not action.accepts(state):
                    continue
                if action.result_kind == "continue" and depth >= max_steps:
                    continue
                next_state = action(state)
                next_probability = probability * action_probability[:, index]
                if action.result_kind == "continue":
                    visit(
                        next_state,
                        next_probability,
                        depth + 1,
                    )
                elif depth >= min_steps:
                    assert action.next_bank_id is not None
                    assert descend_loss is not None
                    row_loss = descend_loss(action.next_bank_id, next_state, target)
                    if (
                        not isinstance(row_loss, Tensor)
                        or not row_loss.is_floating_point()
                        or row_loss.shape != (batch,)
                        or row_loss.device != initial.device
                    ):
                        raise TypeError(
                            "descend_loss must return floating [B] on the task device"
                        )
                    expected_task = expected_task + next_probability * row_loss
                    success_probability = success_probability + next_probability

        visit(initial, initial.new_ones((batch,)), 1)
        invalid = (1.0 - success_probability).clamp_min(0.0)
        task = expected_task.mean()
        invalid_loss = invalid.mean()
        return BankLocalProgramTrainingLoss(
            total=task + self.invalid_weight * invalid_loss,
            task=task,
            invalid=invalid_loss,
            success_probability=success_probability.mean(),
            visited_states=visited_states,
            per_row_total=expected_task + self.invalid_weight * invalid,
        )


@dataclass(frozen=True)
class DetachedBankLocalRollout:
    """Variable-shape on-policy states without routes or transition teachers."""

    _component_reference: ClassVar[str] = "arti/detached-bank-local-rollout@1"

    states: tuple[Tensor, ...]
    min_steps: int
    max_steps: int
    terminated: bool

    def __post_init__(self) -> None:
        if not self.states:
            raise ValueError("rollout must contain at least one local state")
        if (
            isinstance(self.min_steps, bool)
            or not isinstance(self.min_steps, int)
            or self.min_steps <= 0
            or isinstance(self.max_steps, bool)
            or not isinstance(self.max_steps, int)
            or self.max_steps < self.min_steps
        ):
            raise ValueError("rollout refine bounds are invalid")
        if len(self.states) > self.max_steps:
            raise ValueError("rollout contains more states than max_steps")
        for state in self.states:
            if (
                not isinstance(state, Tensor)
                or not state.is_floating_point()
                or state.ndim < 2
                or state.shape[0] != 1
            ):
                raise TypeError("rollout states must be floating tensors with batch size one")
            if state.requires_grad or state.grad_fn is not None:
                raise ValueError("rollout states must be detached")


class DetachedBankLocalProgramTraining:
    """Flatten deep local Refine into fresh one-step final-loss decisions."""

    _component_reference: ClassVar[str] = "arti/detached-bank-local-program-training@1"

    def __init__(self, *, invalid_weight: float = 2.0) -> None:
        if (
            isinstance(invalid_weight, bool)
            or not isinstance(invalid_weight, (int, float))
            or not torch.isfinite(torch.tensor(float(invalid_weight)))
            or invalid_weight < 0
        ):
            raise ValueError("invalid_weight must be finite and non-negative")
        self.invalid_weight = float(invalid_weight)

    def contract_config(self) -> dict[str, object]:
        return {
            "invalid_weight": self.invalid_weight,
            "supervision": "final-task-loss-only",
            "trajectory_source": "detached-on-policy",
            "route_teacher": False,
            "transition_teacher": False,
            "estimator": "fresh-one-step-expected-policy",
        }

    @staticmethod
    def _validate_graph(
        query: BankQuery,
        actions: Sequence[BankLocalFormulaAction],
        terminal_action: BankLocalTerminalAction,
    ) -> tuple[BankLocalFormulaAction, ...]:
        if not isinstance(query, BankQuery):
            raise TypeError("query must be an unsealed BankQuery")
        normalized = tuple(actions)
        if not normalized or any(
            not isinstance(action, BankLocalFormulaAction) for action in normalized
        ):
            raise TypeError("actions must contain BankLocalFormulaAction values")
        if not isinstance(terminal_action, BankLocalTerminalAction):
            raise TypeError("terminal_action must be BankLocalTerminalAction")
        action_ids = tuple(action.action_id for action in normalized) + (
            terminal_action.action_id,
        )
        if tuple(query.retrieval_contract.get("member_ids", ())) != action_ids:
            raise ValueError("draft Query member_ids must match the local action graph")
        return normalized

    def capture(
        self,
        query: BankQuery,
        *,
        actions: Sequence[BankLocalFormulaAction],
        terminal_action: BankLocalTerminalAction,
        initial: Tensor,
        min_steps: int,
        max_steps: int,
    ) -> DetachedBankLocalRollout:
        """Capture the current hard trajectory without retaining its route."""

        normalized = self._validate_graph(query, actions, terminal_action)
        if initial.shape[0] != 1:
            raise ValueError("detached Bank-local rollout capture is serial with batch size one")
        if (
            isinstance(min_steps, bool)
            or not isinstance(min_steps, int)
            or min_steps <= 0
            or isinstance(max_steps, bool)
            or not isinstance(max_steps, int)
            or max_steps < min_steps
        ):
            raise ValueError("capture refine bounds are invalid")
        states: list[Tensor] = []
        current = initial.detach()
        terminated = False
        with torch.no_grad():
            for depth in range(1, max_steps + 1):
                states.append(current.detach().clone())
                logits = query(current).value
                if logits.shape != (1, len(normalized) + 1):
                    raise ValueError("draft Query returned the wrong action shape")
                if terminal_action.requested(
                    logits[:, -1],
                    logits[:, :-1],
                    current,
                ):
                    if depth < min_steps or not terminal_action.accepts(current):
                        raise RuntimeError("on-policy Query requested an invalid early exit")
                    terminated = True
                    break
                action = normalized[int(logits[:, :-1].argmax(dim=-1).item())]
                if not action.accepts(current):
                    raise RuntimeError("on-policy Query selected a shape-incompatible action")
                current = action(current).detach()
        return DetachedBankLocalRollout(
            states=tuple(states),
            min_steps=min_steps,
            max_steps=max_steps,
            terminated=terminated,
        )

    def loss(
        self,
        query: BankQuery,
        rollout: DetachedBankLocalRollout,
        *,
        actions: Sequence[BankLocalFormulaAction],
        terminal_action: BankLocalTerminalAction,
        target: object,
        task_loss: BankLocalStepTaskLoss,
    ) -> BankLocalProgramTrainingLoss:
        """Re-query every detached state and train from real downstream loss."""

        normalized = self._validate_graph(query, actions, terminal_action)
        if not isinstance(rollout, DetachedBankLocalRollout):
            raise TypeError("rollout must be DetachedBankLocalRollout")
        expected_terms: list[Tensor] = []
        invalid_terms: list[Tensor] = []
        valid_mass_terms: list[Tensor] = []
        for depth, state in enumerate(rollout.states, start=1):
            logits = query(state).value
            if logits.shape != (1, len(normalized) + 1):
                raise ValueError("draft Query returned the wrong action shape")
            probability = logits.softmax(dim=-1)
            expected = state.new_zeros((1,))
            valid_mass = state.new_zeros((1,))
            if depth < rollout.max_steps:
                for index, action in enumerate(normalized):
                    if not action.accepts(state):
                        continue
                    row_loss = task_loss(action(state), target, depth)
                    self._validate_row_loss(row_loss, state)
                    expected = expected + probability[:, index] * row_loss
                    valid_mass = valid_mass + probability[:, index]
            if depth >= rollout.min_steps and terminal_action.accepts(state):
                row_loss = task_loss(state, target, depth)
                self._validate_row_loss(row_loss, state)
                expected = expected + probability[:, -1] * row_loss
                valid_mass = valid_mass + probability[:, -1]
            expected_terms.append(expected)
            invalid_terms.append((1.0 - valid_mass).clamp_min(0.0))
            valid_mass_terms.append(valid_mass)
        task = torch.cat(expected_terms).mean()
        invalid = torch.cat(invalid_terms).mean()
        return BankLocalProgramTrainingLoss(
            total=task + self.invalid_weight * invalid,
            task=task,
            invalid=invalid,
            success_probability=torch.cat(valid_mass_terms).mean(),
            visited_states=len(rollout.states),
            per_row_total=torch.cat(expected_terms)
            + self.invalid_weight * torch.cat(invalid_terms),
        )

    @staticmethod
    def _validate_row_loss(value: Tensor, state: Tensor) -> None:
        if (
            not isinstance(value, Tensor)
            or not value.is_floating_point()
            or value.shape != (1,)
            or value.device != state.device
            or not bool(torch.isfinite(value).all())
        ):
            raise TypeError("task_loss must return one finite floating value on the state device")


__all__ = [
    "BankLocalActionKind",
    "BankLocalDescendLoss",
    "BankLocalFormulaAction",
    "BankLocalFormulaEffectAction",
    "BankLocalFormulaEffectResult",
    "BankLocalFormulaProgram",
    "BankLocalProgramTrainingLoss",
    "BankLocalTaskLoss",
    "BankLocalStepTaskLoss",
    "BankLocalTerminalAction",
    "DetachedBankLocalProgramTraining",
    "DetachedBankLocalRollout",
    "ExactBankLocalProgramTraining",
    "ValueTerminalAdapter",
]
