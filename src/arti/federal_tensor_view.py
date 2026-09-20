"""Shape-polymorphic federated Banks with TensorView-local iteration."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import math
import re
from types import MappingProxyType
from typing import ClassVar, Literal

import torch
from torch import Tensor, nn

from .bank_local_program import (
    BankLocalFormulaEffectAction,
    BankLocalFormulaAction,
    BankLocalTerminalAction,
)
from .component_registry import canonical_contract_reference, component_ref
from .federal_recall import (
    LocalIterationPolicy,
    LocalIterationTraceStep,
    FederalBankStep,
    FederalCandidate,
    FederalRecallError,
    FederalTerminalRecord,
    FederalTraceStep,
)
from .shape_query import (
    SealedTensorViewBankQuery,
    TensorViewQueryResult,
)
from .formula_program_query_v3 import BankSlotRef, FormulaProgramBankState
from .formula_v2 import FormulaV2Error, _validate_tensor_against_type
from .resource_graph import (
    ProgramNode,
    ProgramNodeInvocation,
    ProgramGraph,
    ProgramGraphState,
    ResourceGraphError,
)
from .tensor_schema import GradientContract
from .tensor_view import AxisDescriptor, TensorIndexMap, TensorView, TensorViewPattern
from .terminal_abi import BankExecutionSignatureV3, TerminalOutputABI


FEDERAL_RECALL_V3_VERSION = 3
TENSOR_VIEW_FEDERAL_TRACE_VERSION = 3

_NAME = re.compile(r"^[a-z][a-z0-9]*(?:[-_.][a-z0-9]+)*$")
TensorViewActionKind = Literal["continue", "descend"]
IndexTransitionKind = Literal["identity", "preserve_flat", "permute"]


@dataclass(frozen=True)
class _TensorViewProducerLineage:
    owner_bank_id: str
    action_id: str
    slot_ref: BankSlotRef | None
    bound_revision: int | None
    bound_value: Tensor | None


@dataclass(frozen=True)
class _BankSlotEffectProposal:
    target: BankSlotRef
    owner_bank_id: str
    predecessor_action_id: str
    effect_action_id: str
    effect_program_fingerprint: str
    instruction_id: str
    effect_atom_ref: str
    previous_revision: int
    successor_revision: int
    previous: Tensor
    successor: Tensor

    def __post_init__(self) -> None:
        if self.target.producer_id != f"{self.owner_bank_id}.{self.predecessor_action_id}":
            raise FederalRecallError("effect target must belong to its dynamic predecessor")
        if self.successor_revision != self.previous_revision + 1:
            raise FederalRecallError("effect proposal must advance one logical revision")
        if (
            self.previous.shape != self.successor.shape
            or self.previous.dtype != self.successor.dtype
            or self.previous.device != self.successor.device
        ):
            raise FederalRecallError("effect proposal must preserve Bank slot type")


_BankSlotProposals = tuple[_BankSlotEffectProposal, ...]


@dataclass(frozen=True)
class _ResourceGraphStateProposal:
    """A branch-local resource state produced by one declared graph call.

    This is deliberately a proposal rather than a live write.  A federated
    search can therefore explore several calls from the same resource snapshot
    and publish only the selected terminal path.
    """

    graph: ProgramGraph
    state: ProgramGraphState
    action_id: str
    connection_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.graph, ProgramGraph):
            raise TypeError("resource proposal graph must be ProgramGraph")
        if not isinstance(self.state, ProgramGraphState):
            raise TypeError("resource proposal state must be ProgramGraphState")
        if self.state.contract_fingerprint != self.graph.contract_fingerprint:
            raise FederalRecallError("resource proposal must match its graph contract")
        _require_name(self.action_id, field="resource proposal action_id")
        object.__setattr__(self, "connection_ids", tuple(self.connection_ids))
        if not self.connection_ids:
            raise FederalRecallError("resource proposal must record declared connections")


_ResourceGraphProposals = tuple[_ResourceGraphStateProposal, ...]


def _resource_graph_state(
    graph: ProgramGraph,
    proposals: _ResourceGraphProposals,
) -> ProgramGraphState | None:
    """Return the latest branch-local state for one graph identity."""

    for proposal in reversed(proposals):
        if proposal.graph is graph:
            return proposal.state
    return None


def _replace_resource_graph_state(
    proposals: _ResourceGraphProposals,
    proposal: _ResourceGraphStateProposal,
) -> _ResourceGraphProposals:
    """Advance one graph inside a branch without touching another graph."""

    return (
        *(item for item in proposals if item.graph is not proposal.graph),
        proposal,
    )


def _effect_state(
    entry: FormulaProgramBankState,
    proposals: _BankSlotProposals,
    slot_ref: BankSlotRef,
) -> tuple[Tensor, int]:
    for proposal in reversed(proposals):
        if proposal.target == slot_ref:
            return proposal.successor, proposal.successor_revision
    return entry.value(slot_ref), entry.revision(slot_ref)


def _committed_state(
    entry: FormulaProgramBankState,
    proposals: _BankSlotProposals,
) -> FormulaProgramBankState:
    state = entry
    for proposal in proposals:
        state = state.replace(
            proposal.target,
            proposal.successor,
            revision=proposal.successor_revision,
        )
    return state


def _require_name(value: str, *, field: str) -> None:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise FederalRecallError(f"{field} must be a canonical lowercase name")


def _mesh(shape: tuple[int, ...], *, device: torch.device) -> Tensor:
    if not shape:
        return torch.zeros((1, 0), dtype=torch.int64, device=device)
    axes = [torch.arange(size, device=device, dtype=torch.int64) for size in shape]
    return torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)


@dataclass(frozen=True)
class TensorViewLayoutTransition:
    """Describe the logical view produced by one Formula output tensor."""

    axis_names: tuple[str, ...]
    axis_roles: tuple[str, ...]
    index_transition: IndexTransitionKind = "preserve_flat"
    axis_permutation: tuple[int, ...] = ()

    _component_reference: ClassVar[str] = "arti/tensor-view-layout-transition@1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "axis_names", tuple(self.axis_names))
        object.__setattr__(self, "axis_roles", tuple(self.axis_roles))
        object.__setattr__(self, "axis_permutation", tuple(self.axis_permutation))
        if not self.axis_names or len(self.axis_names) != len(self.axis_roles):
            raise FederalRecallError("layout axis names and roles must be non-empty and aligned")
        if len(set(self.axis_names)) != len(self.axis_names):
            raise FederalRecallError("layout axis names must be unique")
        if self.axis_roles.count("batch") != 1:
            raise FederalRecallError("layout transition must contain one batch axis")
        if self.index_transition not in {"identity", "preserve_flat", "permute"}:
            raise FederalRecallError("unsupported TensorView index transition")
        nonbatch_rank = len(self.axis_names) - 1
        if self.index_transition == "permute":
            if set(self.axis_permutation) != set(range(nonbatch_rank)):
                raise FederalRecallError("axis_permutation must permute every non-batch axis")
        elif self.axis_permutation:
            raise FederalRecallError("axis_permutation is only valid for permute transitions")

    def to_dict(self) -> dict[str, object]:
        return {
            "ref": canonical_contract_reference(self._component_reference),
            "axis_names": list(self.axis_names),
            "axis_roles": list(self.axis_roles),
            "index_transition": self.index_transition,
            "axis_permutation": list(self.axis_permutation),
        }

    @staticmethod
    def _source_coordinates(view: TensorView) -> tuple[Tensor, tuple[str, ...], tuple[int, ...]]:
        batch_axis = view.batch_axis
        target_shape = tuple(
            int(size) for index, size in enumerate(view.value.shape) if index != batch_axis
        )
        if view.index_map is None or view.index_map.is_identity:
            source_axes = tuple(
                axis.name for index, axis in enumerate(view.axes) if index != batch_axis
            )
            return _mesh(target_shape, device=view.value.device), source_axes, target_shape
        assert view.index_map.coordinates is not None
        return (
            view.index_map.coordinates,
            view.index_map.source_axes,
            view.index_map.source_shape,
        )

    def apply(self, source: TensorView, output: Tensor) -> TensorView:
        if not isinstance(source, TensorView) or not isinstance(output, Tensor):
            raise TypeError("layout transitions require a TensorView and output Tensor")
        if output.ndim != len(self.axis_names):
            raise FederalRecallError("Formula output rank does not match its layout transition")
        output_batch_axis = self.axis_roles.index("batch")
        if output.shape[output_batch_axis] != source.value.shape[source.batch_axis]:
            raise FederalRecallError("Formula output changed the batch extent")
        axes = tuple(
            AxisDescriptor(name, role, int(extent))
            for name, role, extent in zip(
                self.axis_names, self.axis_roles, output.shape, strict=True
            )
        )
        source_target_shape = tuple(
            int(size) for index, size in enumerate(source.value.shape) if index != source.batch_axis
        )
        output_target_shape = tuple(
            int(size) for index, size in enumerate(output.shape) if index != output_batch_axis
        )
        canonical_mask = None if source.mask is None else source.mask.movedim(source.batch_axis, 0)
        if self.index_transition == "identity":
            if source_target_shape != output_target_shape:
                raise FederalRecallError("identity index transitions must preserve non-batch shape")
            output_mask = (
                None if canonical_mask is None else canonical_mask.movedim(0, output_batch_axis)
            )
            return TensorView(
                output,
                axes,
                index_map=source.index_map,
                mask=output_mask,
            )

        coordinates, source_axes, source_shape = self._source_coordinates(source)
        batched = coordinates.ndim == len(source_target_shape) + 2
        if self.index_transition == "preserve_flat":
            if math.prod(source_target_shape) != math.prod(output_target_shape):
                raise FederalRecallError(
                    "preserve_flat requires Formula input/output element-count preservation"
                )
            if batched:
                coordinates = coordinates.reshape(
                    coordinates.shape[0], *output_target_shape, len(source_shape)
                )
            else:
                coordinates = coordinates.reshape(*output_target_shape, len(source_shape))
            output_mask = (
                None
                if canonical_mask is None
                else canonical_mask.reshape(
                    canonical_mask.shape[0],
                    *output_target_shape,
                ).movedim(0, output_batch_axis)
            )
        else:
            if len(source_target_shape) != len(output_target_shape):
                raise FederalRecallError("permute index transitions must preserve non-batch rank")
            if batched:
                coordinates = coordinates.permute(
                    0,
                    *(index + 1 for index in self.axis_permutation),
                    coordinates.ndim - 1,
                )
            else:
                coordinates = coordinates.permute(
                    *self.axis_permutation,
                    coordinates.ndim - 1,
                )
            output_mask = (
                None
                if canonical_mask is None
                else canonical_mask.permute(
                    0,
                    *(index + 1 for index in self.axis_permutation),
                ).movedim(0, output_batch_axis)
            )
        index_map = TensorIndexMap(
            source_axes=source_axes,
            source_shape=source_shape,
            target_shape=output_target_shape,
            coordinates=coordinates,
        )
        return TensorView(output, axes, index_map=index_map, mask=output_mask)


class TensorViewFormulaAction(nn.Module):
    """Run one ordinary Formula action and stamp its producer-owned Bank slot."""

    _component_reference: ClassVar[str] = "arti/tensor-view-formula-action@2"

    def __init__(
        self,
        action: BankLocalFormulaAction,
        *,
        layout: TensorViewLayoutTransition,
    ) -> None:
        super().__init__()
        if not isinstance(action, BankLocalFormulaAction):
            raise TypeError("action must be BankLocalFormulaAction")
        if not isinstance(layout, TensorViewLayoutTransition):
            raise TypeError("layout must be TensorViewLayoutTransition")
        if action.output_schema.rank != len(layout.axis_names):
            raise FederalRecallError("action output schema and TensorView layout ranks disagree")
        self.action = action
        self.layout = layout
        self._owner_bank_id: str | None = None

    def _bind_owner_program(self, program_id: str) -> None:
        _require_name(program_id, field="program_id")
        if self._owner_bank_id is not None and self._owner_bank_id != program_id:
            raise FederalRecallError("a Formula action cannot be owned by multiple Programs")
        self._owner_bank_id = program_id

    @property
    def owner_program_id(self) -> str:
        if self._owner_bank_id is None:
            raise FederalRecallError("TensorView Formula action is not mounted in a Program")
        return self._owner_bank_id

    @property
    def action_id(self) -> str:
        return self.action.action_id

    @property
    def result_kind(self) -> TensorViewActionKind:
        return self.action.result_kind

    @property
    def next_program_id(self) -> str | None:
        return self.action.next_bank_id

    @property
    def bank_slot_ref(self) -> BankSlotRef | None:
        if self.action.plastic_bank_slot is None:
            return None
        return self.action.bank_slot_ref(self.owner_program_id)

    def accepts(self, view: TensorView) -> bool:
        return isinstance(view, TensorView) and self.action.accepts(view.value)

    def contract_config(self) -> dict[str, object]:
        return {
            "action": self.action.contract_config(),
            "layout": self.layout.to_dict(),
            "owner_program_id": self.owner_program_id,
            "bank_slot_ref": (
                None if self.bank_slot_ref is None else self.bank_slot_ref.to_dict()
            ),
            "state_transform_owner": "formula-fabric",
        }

    def _execute(
        self,
        view: TensorView,
        *,
        bank_state: FormulaProgramBankState,
        proposals: _BankSlotProposals,
        resource_proposals: _ResourceGraphProposals = (),
    ) -> _TensorViewActionExecution:
        if not isinstance(view, TensorView):
            raise TypeError("TensorViewFormulaAction expects TensorView")
        output = self.action._execute_from_bank_state(
            view.value,
            owner_bank_id=self.owner_program_id,
            bank_state=bank_state,
        )
        slot_ref = self.bank_slot_ref
        producer = _TensorViewProducerLineage(
            self.owner_program_id,
            self.action_id,
            slot_ref,
            None if slot_ref is None else bank_state.revision(slot_ref),
            None if slot_ref is None else bank_state.value(slot_ref),
        )
        return _TensorViewActionExecution(
            self.layout.apply(view, output),
            proposals,
            resource_proposals,
            producer,
        )

    def forward(self, view: TensorView) -> TensorView:
        if self._owner_bank_id is None:
            if self.action.plastic_bank_slot is not None:
                raise FederalRecallError(
                    "a plastic TensorView Formula action must be mounted in a Bank"
                )
            return self.layout.apply(view, self.action(view.value))
        slot_ref = self.bank_slot_ref
        state = (
            FormulaProgramBankState.empty()
            if slot_ref is None
            else FormulaProgramBankState(
                (slot_ref,),
                (self.action.initial_bank_value(),),
                (self.action.initial_revision(),),
            )
        )
        return self._execute(view, bank_state=state, proposals=()).view


class TensorViewResourceGraphAction(nn.Module):
    """Call a declared resource subgraph as one local federated action.

    The incoming TensorView is mounted only in an ephemeral branch state.  The
    resulting state is carried with the candidate and becomes visible to the
    graph only after the federal winner has produced a valid terminal output.
    """

    _component_reference: ClassVar[str] = "arti/tensor-view-resource-graph-action@1"

    def __init__(
        self,
        *,
        action_id: str,
        graph: ProgramGraph,
        connection_ids: Sequence[str] | None = None,
        program_id: str | None = None,
        input_resource_id: str,
        output_resource_id: str,
        result_kind: TensorViewActionKind = "continue",
        next_program_id: str | None = None,
        use_input_context: bool = False,
        iterations: int = 1,
    ) -> None:
        super().__init__()
        _require_name(action_id, field="action_id")
        if not isinstance(graph, ProgramGraph):
            raise TypeError("graph must be ProgramGraph")
        if (connection_ids is None) == (program_id is None):
            raise FederalRecallError(
                "resource action requires exactly one of connection_ids or program_id"
            )
        if program_id is not None:
            if not isinstance(program_id, str):
                raise TypeError("program_id must be a string or None")
            try:
                connection_ids = graph.program(program_id)
            except ResourceGraphError as error:
                raise FederalRecallError(
                    f"resource action references an unknown graph program {program_id!r}"
                ) from error
        assert connection_ids is not None
        connection_ids = tuple(connection_ids)
        if not connection_ids or any(not isinstance(item, str) for item in connection_ids):
            raise FederalRecallError("connection_ids must be declared connection names")
        for connection_id in connection_ids:
            graph.connection(connection_id)
        if input_resource_id not in graph.resources or output_resource_id not in graph.resources:
            raise FederalRecallError("resource action endpoints must name graph resources")
        if result_kind not in {"continue", "descend"}:
            raise FederalRecallError("resource action result_kind must be continue or descend")
        if type(use_input_context) is not bool:
            raise TypeError("use_input_context must be boolean")
        if type(iterations) is not int or iterations <= 0:
            raise FederalRecallError("iterations must be a positive integer")
        if result_kind == "continue":
            if next_program_id is not None:
                raise FederalRecallError("continuing resource actions cannot name a child Program")
        else:
            if next_program_id is None:
                raise FederalRecallError("descending resource actions require next_program_id")
            _require_name(next_program_id, field="next_program_id")
        self.action_id = action_id
        self.graph = graph
        self.connection_ids = connection_ids
        self.program_id = program_id
        self.input_resource_id = input_resource_id
        self.output_resource_id = output_resource_id
        self.result_kind = result_kind
        self.next_program_id = next_program_id
        self.use_input_context = use_input_context
        self.iterations = iterations
        self._owner_bank_id: str | None = None

    def _bind_owner_program(self, program_id: str) -> None:
        _require_name(program_id, field="program_id")
        if self._owner_bank_id is not None and self._owner_bank_id != program_id:
            raise FederalRecallError("a resource action cannot be owned by multiple Programs")
        self._owner_bank_id = program_id

    @property
    def owner_program_id(self) -> str:
        if self._owner_bank_id is None:
            raise FederalRecallError("TensorView resource action is not mounted in a Program")
        return self._owner_bank_id

    def accepts(self, view: TensorView) -> bool:
        if not isinstance(view, TensorView):
            return False
        try:
            self.graph.resource(self.input_resource_id).spec.validate(
                view, name=f"{self.action_id}.input"
            )
        except ResourceGraphError:
            return False
        return True

    def contract_config(self) -> dict[str, object]:
        return {
            "action_id": self.action_id,
            "owner_program_id": self._owner_bank_id,
            "graph_contract_fingerprint": self.graph.contract_fingerprint,
            "connection_ids": list(self.connection_ids),
            "program_id": self.program_id,
            "input_resource_id": self.input_resource_id,
            "output_resource_id": self.output_resource_id,
            "result_kind": self.result_kind,
            "next_program_id": self.next_program_id,
            "connection_context": "current-input-view" if self.use_input_context else None,
            "iterations": self.iterations,
            "state_visibility": "winner-committed-after-terminal-validation",
        }

    def _execute(
        self,
        view: TensorView,
        *,
        bank_proposals: _BankSlotProposals,
        proposals: _ResourceGraphProposals,
    ) -> _TensorViewActionExecution:
        if not self.accepts(view):
            raise FederalRecallError("resource action does not accept the current TensorView")
        try:
            invocation = self.graph.invoke_functional(
                self.connection_ids,
                input_resource_id=self.input_resource_id,
                input_view=view,
                output_resource_id=self.output_resource_id,
                state=_resource_graph_state(self.graph, proposals),
                contexts=(
                    {connection_id: view.value for connection_id in self.connection_ids}
                    if self.use_input_context
                    else None
                ),
                iterations=self.iterations,
            )
        except ResourceGraphError as error:
            raise FederalRecallError(
                f"resource action {self.action_id!r} failed its declared graph call"
            ) from error
        proposal = _ResourceGraphStateProposal(
            self.graph,
            invocation.state,
            self.action_id,
            self.connection_ids,
        )
        return _TensorViewActionExecution(
            invocation.output,
            bank_proposals,
            _replace_resource_graph_state(proposals, proposal),
            None,
        )

    def forward(self, view: TensorView) -> TensorView:
        """Evaluate purely; callers must explicitly publish the returned state."""

        return self._execute(view, bank_proposals=(), proposals=()).view


@dataclass(frozen=True)
class _TensorViewActionExecution:
    view: TensorView
    bank_proposals: _BankSlotProposals
    resource_proposals: _ResourceGraphProposals
    producer: _TensorViewProducerLineage | None
    effect_proposal: _BankSlotEffectProposal | None = None


class TensorViewFormulaEffectAction(nn.Module):
    """Apply an identity-data effect to the dynamic ordinary predecessor Bank slot."""

    _component_reference: ClassVar[str] = "arti/tensor-view-formula-effect-action@1"

    def __init__(
        self,
        action: BankLocalFormulaEffectAction,
        *,
        layout: TensorViewLayoutTransition,
    ) -> None:
        super().__init__()
        if not isinstance(action, BankLocalFormulaEffectAction):
            raise TypeError("action must be BankLocalFormulaEffectAction")
        if not isinstance(layout, TensorViewLayoutTransition):
            raise TypeError("layout must be TensorViewLayoutTransition")
        if layout.index_transition != "identity":
            raise FederalRecallError("Formula effect data identity requires identity layout")
        if action.input_schema.rank != len(layout.axis_names):
            raise FederalRecallError("effect schema and TensorView layout ranks disagree")
        self.action = action
        self.layout = layout
        self._owner_bank_id: str | None = None

    def _bind_owner_program(self, program_id: str) -> None:
        _require_name(program_id, field="program_id")
        if self._owner_bank_id is not None and self._owner_bank_id != program_id:
            raise FederalRecallError("a Formula effect action cannot be owned by multiple Programs")
        self._owner_bank_id = program_id

    @property
    def owner_program_id(self) -> str:
        if self._owner_bank_id is None:
            raise FederalRecallError("TensorView Formula effect is not mounted in a Program")
        return self._owner_bank_id

    @property
    def action_id(self) -> str:
        return self.action.action_id

    @property
    def result_kind(self) -> TensorViewActionKind:
        return self.action.result_kind

    @property
    def next_program_id(self) -> str | None:
        return self.action.next_bank_id

    def accepts(
        self,
        view: TensorView,
        producer: _TensorViewProducerLineage | None = None,
    ) -> bool:
        if (
            not isinstance(view, TensorView)
            or not self.action.accepts(view.value)
            or producer is None
            or producer.slot_ref is None
            or producer.bound_revision is None
            or producer.bound_value is None
        ):
            return False
        try:
            _validate_tensor_against_type(
                producer.bound_value,
                self.action.effect_program.state_type,
                name=f"{self.action_id}.predecessor_bank_slot",
            )
        except FormulaV2Error:
            return False
        return True

    def contract_config(self) -> dict[str, object]:
        return {
            "action": self.action.contract_config(),
            "layout": self.layout.to_dict(),
            "owner_program_id": self.owner_program_id,
            "data_lane": "identity",
            "target_resolution": "dynamic-immediate-predecessor-bank-slot",
        }

    def _execute(
        self,
        view: TensorView,
        *,
        bank_state: FormulaProgramBankState,
        proposals: _BankSlotProposals,
        producer: _TensorViewProducerLineage | None,
        resource_proposals: _ResourceGraphProposals = (),
    ) -> _TensorViewActionExecution:
        if not self.accepts(view, producer):
            raise FederalRecallError(
                "Formula effect requires one unambiguous plastic ordinary predecessor"
            )
        assert producer is not None
        assert producer.slot_ref is not None
        previous, previous_revision = _effect_state(
            bank_state,
            proposals,
            producer.slot_ref,
        )
        result = self.action._execute_against(
            view.value,
            previous,
            previous_revision=previous_revision,
        )
        instruction = self.action.effect_program.effect_instruction
        proposal = _BankSlotEffectProposal(
            producer.slot_ref,
            producer.owner_bank_id,
            producer.action_id,
            self.action_id,
            self.action.effect_program.fingerprint,
            instruction.instruction_id,
            canonical_contract_reference(instruction.atom_ref),
            result.previous_revision,
            result.successor_revision,
            previous,
            result.successor,
        )
        next_view = self.layout.apply(view, result.value)
        if next_view.value is not view.value:
            raise FederalRecallError("Formula effect must preserve the current Tensor identity")
        return _TensorViewActionExecution(
            next_view,
            (*proposals, proposal),
            resource_proposals,
            producer,
            proposal,
        )

    def forward(self, view: TensorView) -> TensorView:
        raise RuntimeError(
            "Formula effects can only execute with runtime-resolved predecessor lineage"
        )

@dataclass(frozen=True)
class TensorViewFederalCandidate(FederalCandidate):
    """A federal candidate with producer lineage and write-only Bank proposals."""

    next_view: TensorView | None = None
    bank_proposals: _BankSlotProposals = ()
    resource_proposals: _ResourceGraphProposals = ()
    producer: _TensorViewProducerLineage | None = None
    effect_proposal: _BankSlotEffectProposal | None = None

    _runtime_contract_ref: ClassVar[str] = "arti/federal-candidate@3"

    @property
    def next_program_id(self) -> str | None:
        """Return the child Program identity for the canonical tensor-view API."""

        return self.next_bank_id

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "bank_proposals", tuple(self.bank_proposals))
        object.__setattr__(self, "resource_proposals", tuple(self.resource_proposals))
        if any(not isinstance(item, _BankSlotEffectProposal) for item in self.bank_proposals):
            raise TypeError("candidate proposals must be Bank-slot effect proposals")
        if any(
            not isinstance(item, _ResourceGraphStateProposal)
            for item in self.resource_proposals
        ):
            raise TypeError("candidate resource proposals must be graph-state proposals")
        if self.effect_proposal is not None:
            if (
                not self.bank_proposals
                or self.bank_proposals[-1] is not self.effect_proposal
            ):
                raise FederalRecallError(
                    "candidate effect proposal must be the latest branch proposal"
                )
        if self.next_value is None:
            if self.next_view is not None:
                raise FederalRecallError("terminal candidates cannot carry a successor view")
            return
        if not isinstance(self.next_view, TensorView):
            raise TypeError("non-terminal TensorView candidates require next_view")
        if self.next_view.value is not self.next_value:
            raise FederalRecallError("candidate next_value must be the TensorView payload")

    @classmethod
    def local_view(
        cls,
        candidate_id: str,
        *,
        local_log_score: Tensor,
        next_view: TensorView,
        bank_proposals: _BankSlotProposals = (),
        resource_proposals: _ResourceGraphProposals = (),
        producer: _TensorViewProducerLineage | None = None,
        effect_proposal: _BankSlotEffectProposal | None = None,
    ) -> TensorViewFederalCandidate:
        return cls(
            candidate_id,
            local_log_score,
            next_value=next_view.value,
            next_view=next_view,
            bank_proposals=bank_proposals,
            resource_proposals=resource_proposals,
            producer=producer,
            effect_proposal=effect_proposal,
        )

    @classmethod
    def child_view(
        cls,
        candidate_id: str,
        *,
        local_log_score: Tensor,
        next_program_id: str,
        next_view: TensorView,
        bank_proposals: _BankSlotProposals = (),
        resource_proposals: _ResourceGraphProposals = (),
        producer: _TensorViewProducerLineage | None = None,
        effect_proposal: _BankSlotEffectProposal | None = None,
    ) -> TensorViewFederalCandidate:
        return cls(
            candidate_id,
            local_log_score,
            next_bank_id=next_program_id,
            next_value=next_view.value,
            next_view=next_view,
            bank_proposals=bank_proposals,
            resource_proposals=resource_proposals,
            producer=producer,
            effect_proposal=effect_proposal,
        )

    @classmethod
    def terminal_view(
        cls,
        candidate_id: str,
        *,
        local_log_score: Tensor,
        outputs: Mapping[str, Tensor],
        bank_proposals: _BankSlotProposals = (),
        resource_proposals: _ResourceGraphProposals = (),
        producer: _TensorViewProducerLineage | None = None,
    ) -> TensorViewFederalCandidate:
        return cls(
            candidate_id,
            local_log_score,
            terminal_outputs=outputs,
            bank_proposals=bank_proposals,
            resource_proposals=resource_proposals,
            producer=producer,
        )

@dataclass(frozen=True)
class TensorViewLocalIterationTraceStep(LocalIterationTraceStep):
    """Local iteration evidence including dynamic predecessor Bank ownership."""

    input_view_fingerprint: str = ""
    output_view_fingerprint: str | None = None
    input_axes: tuple[str, ...] = ()
    output_axes: tuple[str, ...] | None = None
    effect_action_id: str | None = None
    effect_action_ref: str | None = None
    predecessor_action_id: str | None = None
    target_bank_slot: BankSlotRef | None = None
    effect_program_fingerprint: str | None = None
    effect_instruction_id: str | None = None
    effect_atom_ref: str | None = None
    effect_visibility: str | None = None
    effect_state_change_norm: float | None = None
    effect_previous_revision: int | None = None
    effect_successor_revision: int | None = None

    _runtime_contract_ref: ClassVar[str] = "arti/local-iteration-trace-step@3"

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "input_axes", tuple(self.input_axes))
        if self.output_axes is not None:
            object.__setattr__(self, "output_axes", tuple(self.output_axes))

    def to_dict(self) -> dict[str, object]:
        return {
            **super().to_dict(),
            "ref": canonical_contract_reference(self._runtime_contract_ref),
            "input_view_fingerprint": self.input_view_fingerprint,
            "output_view_fingerprint": self.output_view_fingerprint,
            "input_axes": list(self.input_axes),
            "output_axes": None if self.output_axes is None else list(self.output_axes),
            "effect_action_id": self.effect_action_id,
            "effect_action_ref": self.effect_action_ref,
            "predecessor_action_id": self.predecessor_action_id,
            "target_bank_slot": (
                None if self.target_bank_slot is None else self.target_bank_slot.to_dict()
            ),
            "effect_program_fingerprint": self.effect_program_fingerprint,
            "effect_instruction_id": self.effect_instruction_id,
            "effect_atom_ref": self.effect_atom_ref,
            "effect_visibility": self.effect_visibility,
            "effect_state_change_norm": self.effect_state_change_norm,
            "effect_previous_revision": self.effect_previous_revision,
            "effect_successor_revision": self.effect_successor_revision,
        }


@dataclass(frozen=True)
class NeuralPlasticityCommitReceipt:
    """JSON-safe evidence for a winner proposal installed into its producer slot."""

    bank_id: str
    producer_action_id: str
    effect_action_id: str
    bank_slot_ref: BankSlotRef
    effect_program_fingerprint: str
    instruction_id: str
    effect_atom_ref: str
    previous_revision: int
    successor_revision: int
    state_change_norm: float
    winner_path: str
    visibility: str = "next-dispatch"

    _runtime_contract_ref: ClassVar[str] = "arti/neural-plasticity-commit-receipt@2"

    def __post_init__(self) -> None:
        _require_name(self.bank_id, field="bank_id")
        _require_name(self.producer_action_id, field="producer_action_id")
        _require_name(self.effect_action_id, field="effect_action_id")
        if self.bank_slot_ref.producer_id != f"{self.bank_id}.{self.producer_action_id}":
            raise FederalRecallError("commit receipt Bank slot must belong to its producer")
        if not isinstance(self.effect_program_fingerprint, str) or len(
            self.effect_program_fingerprint
        ) != 64:
            raise FederalRecallError("effect_program_fingerprint must be SHA-256")
        if not isinstance(self.instruction_id, str) or not self.instruction_id:
            raise FederalRecallError("instruction_id must be non-empty")
        if not isinstance(self.effect_atom_ref, str) or not self.effect_atom_ref:
            raise FederalRecallError("effect_atom_ref must be non-empty")
        if canonical_contract_reference(self.effect_atom_ref) != self.effect_atom_ref:
            raise FederalRecallError("effect_atom_ref must be a canonical contract reference")
        if (
            type(self.previous_revision) is not int
            or type(self.successor_revision) is not int
            or self.previous_revision < 0
            or self.successor_revision != self.previous_revision + 1
        ):
            raise FederalRecallError("commit receipt revisions must advance one step")
        if not math.isfinite(self.state_change_norm) or self.state_change_norm < 0.0:
            raise FederalRecallError("state_change_norm must be finite and non-negative")
        if not isinstance(self.winner_path, str) or not self.winner_path:
            raise FederalRecallError("winner_path must be non-empty")
        if self.visibility != "next-dispatch":
            raise FederalRecallError("unsupported NeuralPlasticity visibility boundary")

    def to_dict(self) -> dict[str, object]:
        return {
            "ref": canonical_contract_reference(self._runtime_contract_ref),
            "bank_id": self.bank_id,
            "producer_action_id": self.producer_action_id,
            "effect_action_id": self.effect_action_id,
            "bank_slot_ref": self.bank_slot_ref.to_dict(),
            "effect_program_fingerprint": self.effect_program_fingerprint,
            "instruction_id": self.instruction_id,
            "effect_atom_ref": self.effect_atom_ref,
            "previous_revision": self.previous_revision,
            "successor_revision": self.successor_revision,
            "state_change_norm": self.state_change_norm,
            "winner_path": self.winner_path,
            "visibility": self.visibility,
        }

@dataclass(frozen=True)
class ResourceGraphCommitReceipt:
    """JSON-safe publication evidence for a winning resource graph branch."""

    graph_contract_fingerprint: str
    action_id: str
    connection_ids: tuple[str, ...]
    resource_ids: tuple[str, ...]
    winner_path: str
    visibility: str = "after-terminal-validation"

    _runtime_contract_ref: ClassVar[str] = "arti/resource-graph-commit-receipt@1"

    def __post_init__(self) -> None:
        if not isinstance(self.graph_contract_fingerprint, str) or len(
            self.graph_contract_fingerprint
        ) != 64:
            raise FederalRecallError("resource graph commit fingerprint must be SHA-256")
        _require_name(self.action_id, field="resource commit action_id")
        object.__setattr__(self, "connection_ids", tuple(self.connection_ids))
        object.__setattr__(self, "resource_ids", tuple(self.resource_ids))
        if not self.connection_ids or any(not isinstance(item, str) for item in self.connection_ids):
            raise FederalRecallError("resource commit must record connection ids")
        if not self.resource_ids or any(not isinstance(item, str) for item in self.resource_ids):
            raise FederalRecallError("resource commit must record changed resource ids")
        if not isinstance(self.winner_path, str) or not self.winner_path:
            raise FederalRecallError("resource commit winner_path must be non-empty")
        if self.visibility != "after-terminal-validation":
            raise FederalRecallError("unsupported resource graph visibility boundary")

    def to_dict(self) -> dict[str, object]:
        return {
            "ref": canonical_contract_reference(self._runtime_contract_ref),
            "graph_contract_fingerprint": self.graph_contract_fingerprint,
            "action_id": self.action_id,
            "connection_ids": list(self.connection_ids),
            "resource_ids": list(self.resource_ids),
            "winner_path": self.winner_path,
            "visibility": self.visibility,
        }


@dataclass(frozen=True)
class TensorViewFederalTrace:
    """FederalRecall@3 trace with explicit NeuralPlasticity publications."""

    max_k: int
    max_levels: int
    steps: tuple[FederalTraceStep, ...]
    winner_paths: tuple[str, ...]
    committed_effects: tuple[NeuralPlasticityCommitReceipt, ...] = ()
    committed_resources: tuple[ResourceGraphCommitReceipt, ...] = ()
    schema_version: int = TENSOR_VIEW_FEDERAL_TRACE_VERSION

    _runtime_contract_ref: ClassVar[str] = "arti/federal-trace@3"

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "winner_paths", tuple(self.winner_paths))
        object.__setattr__(self, "committed_effects", tuple(self.committed_effects))
        object.__setattr__(self, "committed_resources", tuple(self.committed_resources))
        if self.schema_version != TENSOR_VIEW_FEDERAL_TRACE_VERSION:
            raise FederalRecallError("unsupported TensorView FederalTrace version")
        if any(not isinstance(item, FederalTraceStep) for item in self.steps):
            raise TypeError("steps must contain FederalTraceStep values")
        if any(
            not isinstance(item, NeuralPlasticityCommitReceipt)
            for item in self.committed_effects
        ):
            raise TypeError(
                "committed_effects must contain NeuralPlasticityCommitReceipt values"
            )
        if any(
            not isinstance(item, ResourceGraphCommitReceipt)
            for item in self.committed_resources
        ):
            raise TypeError("committed_resources must contain ResourceGraphCommitReceipt values")

    @property
    def maximum_kept_paths(self) -> int:
        return max((step.kept_count for step in self.steps), default=0)

    def to_dict(self) -> dict[str, object]:
        return {
            "ref": canonical_contract_reference(self._runtime_contract_ref),
            "schema_version": self.schema_version,
            "max_k": self.max_k,
            "max_levels": self.max_levels,
            "maximum_kept_paths": self.maximum_kept_paths,
            "steps": [item.to_dict() for item in self.steps],
            "winner_paths": list(self.winner_paths),
            "committed_effects": [item.to_dict() for item in self.committed_effects],
            "committed_resources": [item.to_dict() for item in self.committed_resources],
        }


@dataclass(frozen=True)
class _LocalViewPath:
    view: TensorView
    cumulative_log_score: Tensor
    lineage: tuple[str, ...]
    bank_proposals: _BankSlotProposals = ()
    resource_proposals: _ResourceGraphProposals = ()
    producer: _TensorViewProducerLineage | None = None


class RoutedProgram(nn.Module):
    """A local program that re-queries its latest view during iteration."""

    _component_reference: ClassVar[str] = "arti/routed-program@1"

    def __init__(
        self,
        *,
        program_id: str,
        query: SealedTensorViewBankQuery,
        actions: Sequence[
            TensorViewFormulaAction
            | TensorViewFormulaEffectAction
            | TensorViewResourceGraphAction
        ],
        terminal_action: BankLocalTerminalAction,
        local_iteration: LocalIterationPolicy,
        input_pattern: TensorViewPattern,
        exit_pattern: TensorViewPattern,
        terminal_abi: TerminalOutputABI,
        direct_actions: Sequence[TensorViewResourceGraphAction] = (),
    ) -> None:
        super().__init__()
        _require_name(program_id, field="program_id")
        if not isinstance(query, SealedTensorViewBankQuery):
            raise TypeError("query must be SealedTensorViewBankQuery")
        normalized = tuple(actions)
        if not normalized or any(
            not isinstance(
                item,
                (
                    TensorViewFormulaAction,
                    TensorViewFormulaEffectAction,
                    TensorViewResourceGraphAction,
                ),
            )
            for item in normalized
        ):
            raise TypeError("actions must contain TensorView Formula or effect actions")
        direct = tuple(direct_actions)
        if any(not isinstance(item, TensorViewResourceGraphAction) for item in direct):
            raise TypeError("direct_actions must contain TensorView resource actions")
        if any(item.result_kind != "continue" for item in direct):
            raise FederalRecallError("direct resource actions must remain in the current Program")
        if not isinstance(terminal_action, BankLocalTerminalAction):
            raise TypeError("terminal_action must be BankLocalTerminalAction")
        if not isinstance(local_iteration, LocalIterationPolicy):
            raise TypeError("local_iteration must be LocalIterationPolicy")
        if not isinstance(input_pattern, TensorViewPattern) or not isinstance(
            exit_pattern, TensorViewPattern
        ):
            raise TypeError("input and exit patterns must be TensorViewPattern")
        if query.signature.pattern.fingerprint != input_pattern.fingerprint:
            raise FederalRecallError("Program input pattern must match its sealed Query")
        if not isinstance(terminal_abi, TerminalOutputABI):
            raise TypeError("terminal_abi must be TerminalOutputABI")
        action_ids = tuple(item.action_id for item in normalized) + (terminal_action.action_id,)
        if action_ids != query.signature.member_ids:
            raise FederalRecallError(
                "sealed TensorView Query members must exactly match local actions"
            )
        for item in normalized:
            item._bind_owner_program(program_id)
        for item in direct:
            item._bind_owner_program(program_id)
        if len({item.action_id for item in (*normalized, *direct)}) != len(
            (*normalized, *direct)
        ):
            raise FederalRecallError("direct and selected local action ids must remain unique")
        self.program_id = program_id
        self.query = query
        self.actions = nn.ModuleList(normalized)
        self.direct_actions = nn.ModuleList(direct)
        self.terminal_action = terminal_action
        self.local_iteration = local_iteration
        self.input_pattern = input_pattern
        self.exit_pattern = exit_pattern
        self.terminal_abi = terminal_abi
        effectful = bool(self.effect_actions)
        resourceful = bool(self.all_resource_actions)
        self._signature = BankExecutionSignatureV3.from_program(
            self,
            input_pattern=input_pattern,
            exit_pattern=exit_pattern,
            query_signature=query.signature,
            local_normalization_contract=query.signature.normalization_contract,
            terminal_adapter_ref=component_ref(terminal_action.adapter),
            terminal_abi_ref="arti/terminal-output-abi@1",
            terminal_abi_fingerprint=terminal_abi.fingerprint,
            score_contract="sum of Bank-local TensorView action log probabilities",
            gradient_contract=GradientContract.autograd(),
            local_formula_ref=(
                "arti/formula-fabric@4" if effectful else "arti/formula-fabric@2"
            ),
            local_iteration_ref=component_ref(local_iteration),
            execution_capabilities=tuple(
                sorted(
                    (
                        "eager",
                        "fixed-k-wide",
                        "latest-tensor-view-requery",
                        "variable-rank-local-iteration",
                        *(("winner-committed-predecessor-bank-slots",) if effectful else ()),
                        *(("winner-committed-resource-graphs",) if resourceful else ()),
                        *(("direct-resource-connections",) if self.direct_actions else ()),
                    )
                )
            ),
        )

    @property
    def signature(self) -> BankExecutionSignatureV3:
        return self._signature

    @property
    def action_ids(self) -> tuple[str, ...]:
        return tuple(item.action_id for item in self.actions) + (self.terminal_action.action_id,)

    @property
    def plastic_actions(self) -> tuple[TensorViewFormulaAction, ...]:
        return tuple(
            item
            for item in self.actions
            if isinstance(item, TensorViewFormulaAction) and item.bank_slot_ref is not None
        )

    @property
    def effect_actions(self) -> tuple[TensorViewFormulaEffectAction, ...]:
        return tuple(
            item for item in self.actions if isinstance(item, TensorViewFormulaEffectAction)
        )

    @property
    def resource_actions(self) -> tuple[TensorViewResourceGraphAction, ...]:
        return tuple(
            item for item in self.actions if isinstance(item, TensorViewResourceGraphAction)
        )

    @property
    def all_resource_actions(self) -> tuple[TensorViewResourceGraphAction, ...]:
        return (*self.direct_actions, *self.resource_actions)

    def initial_bank_state(self) -> FormulaProgramBankState:
        actions = self.plastic_actions
        refs = tuple(item.bank_slot_ref for item in actions)
        assert all(item is not None for item in refs)
        return FormulaProgramBankState(
            refs,  # type: ignore[arg-type]
            tuple(item.action.initial_bank_value() for item in actions),
            tuple(item.action.initial_revision() for item in actions),
        )

    def contract_config(self) -> dict[str, object]:
        return {
            "program_id": self.program_id,
            "actions": [item.contract_config() for item in self.actions],
            "direct_actions": [item.contract_config() for item in self.direct_actions],
            "terminal_action": self.terminal_action.contract_config(),
            "local_iteration": self.local_iteration.contract_config(),
            "input_pattern": self.input_pattern.to_dict(),
            "exit_pattern": self.exit_pattern.to_dict(),
            "terminal_abi_fingerprint": self.terminal_abi.fingerprint,
            "pending_visibility": "write-only-until-federal-winner-commit",
            "resource_state_visibility": (
                "winner-committed-after-terminal-validation"
                if self.all_resource_actions
                else None
            ),
        }

    @staticmethod
    def _choice_order(
        item: tuple[str, _LocalViewPath | TensorViewFederalCandidate, tuple[str, ...]],
    ) -> tuple[float, str]:
        _kind, choice, lineage = item
        score = (
            choice.cumulative_log_score
            if isinstance(choice, _LocalViewPath)
            else choice.local_log_score
        )
        return (-float(score.detach().reshape(()).cpu()), "/".join(lineage))

    @staticmethod
    def _lineage_candidate(
        candidate: TensorViewFederalCandidate,
        *,
        lineage: tuple[str, ...],
        preserve_identity: bool,
    ) -> TensorViewFederalCandidate:
        if preserve_identity or len(lineage) == 1:
            return candidate
        digest = hashlib.sha256("/".join(lineage).encode("utf-8")).hexdigest()[:12]
        return replace(candidate, candidate_id=f"{candidate.candidate_id}-{digest}")

    def _trace(
        self,
        *,
        step: int,
        candidate: TensorViewFederalCandidate,
        source: TensorView,
        query: TensorViewQueryResult,
        action: str,
        exit_reason: str | None,
    ) -> TensorViewLocalIterationTraceStep:
        target = candidate.next_view
        proposal = candidate.effect_proposal
        formula_ref = (
            "arti/formula-fabric@4"
            if proposal is not None
            else "arti/formula-fabric@2"
        )
        return TensorViewLocalIterationTraceStep(
            bank_id=self.program_id,
            iteration=step,
            candidate_id=candidate.candidate_id,
            action=action,
            input_shape=tuple(int(size) for size in source.value.shape),
            query_shape=tuple(int(size) for size in query.scores.shape),
            output_shape=(
                None if target is None else tuple(int(size) for size in target.value.shape)
            ),
            query_ref=self.query.signature.query_ref,
            query_state_fingerprint=self.query.signature.state_fingerprint,
            formula_ref=formula_ref,
            exit_reason=exit_reason,
            input_view_fingerprint=source.descriptor_fingerprint,
            output_view_fingerprint=(None if target is None else target.descriptor_fingerprint),
            input_axes=tuple(axis.name for axis in source.axes),
            output_axes=(None if target is None else tuple(axis.name for axis in target.axes)),
            effect_action_id=None if proposal is None else proposal.effect_action_id,
            effect_action_ref=(
                None if proposal is None else f"{self.program_id}/{proposal.effect_action_id}"
            ),
            predecessor_action_id=(
                None if proposal is None else proposal.predecessor_action_id
            ),
            target_bank_slot=None if proposal is None else proposal.target,
            effect_program_fingerprint=(
                None if proposal is None else proposal.effect_program_fingerprint
            ),
            effect_instruction_id=None if proposal is None else proposal.instruction_id,
            effect_atom_ref=None if proposal is None else proposal.effect_atom_ref,
            effect_visibility=None if proposal is None else "next-dispatch",
            effect_state_change_norm=(
                None
                if proposal is None
                else float(
                    torch.linalg.vector_norm(
                        (proposal.successor - proposal.previous).detach().to(torch.float32)
                    )
                    .cpu()
                    .reshape(())
                )
            ),
            effect_previous_revision=(
                None if proposal is None else proposal.previous_revision
            ),
            effect_successor_revision=(
                None if proposal is None else proposal.successor_revision
            ),
        )

    def _execute_once(
        self,
        view: TensorView,
        query: TensorViewQueryResult,
        *,
        max_candidates: int,
        bank_state: FormulaProgramBankState,
        bank_proposals: _BankSlotProposals,
        resource_proposals: _ResourceGraphProposals,
        producer: _TensorViewProducerLineage | None,
    ) -> tuple[TensorViewFederalCandidate, ...]:
        if view.value.shape[view.batch_axis] != 1:
            raise FederalRecallError("TensorView Formula programs execute one sample at a time")
        logits = query.scores
        if logits.shape != (1, len(self.action_ids)):
            raise FederalRecallError("TensorView Query returned the wrong action shape")
        log_probability = logits.log_softmax(dim=-1)
        formula_logits = logits[:, : len(self.actions)]
        exit_logit = logits[:, len(self.actions)]
        terminal_accepted = self.terminal_action.accepts(view.value)
        exit_requested = terminal_accepted and self.terminal_action.requested(
            exit_logit, formula_logits, view.value
        )
        if max_candidates == 1 and exit_requested:
            score = log_probability[0, len(self.actions)]
            return (
                TensorViewFederalCandidate.terminal_view(
                    self.terminal_action.action_id,
                    local_log_score=score,
                    outputs=self.terminal_action(view.value, score),
                    bank_proposals=bank_proposals,
                    resource_proposals=resource_proposals,
                    producer=producer,
                ),
            )
        eligible = [
            index
            for index, candidate in enumerate(self.actions)
            if (
                candidate.accepts(view, producer)
                if isinstance(candidate, TensorViewFormulaEffectAction)
                else candidate.accepts(view)
            )
        ]
        if (max_candidates > 1 or exit_requested) and terminal_accepted:
            eligible.append(len(self.actions))
        if not eligible:
            raise FederalRecallError("no local Formula or terminal accepts the latest TensorView")
        eligible.sort(
            key=lambda index: (
                -float(log_probability[0, index].detach().cpu()),
                self.action_ids[index],
            )
        )
        result: list[TensorViewFederalCandidate] = []
        for index in eligible[:max_candidates]:
            score = log_probability[0, index]
            if index == len(self.actions):
                result.append(
                    TensorViewFederalCandidate.terminal_view(
                        self.terminal_action.action_id,
                        local_log_score=score,
                    outputs=self.terminal_action(view.value, score),
                    bank_proposals=bank_proposals,
                    resource_proposals=resource_proposals,
                    producer=producer,
                    )
                )
                continue
            selected = self.actions[index]
            if isinstance(selected, TensorViewFormulaEffectAction):
                execution = selected._execute(
                    view,
                    bank_state=bank_state,
                    proposals=bank_proposals,
                    producer=producer,
                    resource_proposals=resource_proposals,
                )
            elif isinstance(selected, TensorViewResourceGraphAction):
                execution = selected._execute(
                    view,
                    bank_proposals=bank_proposals,
                    proposals=resource_proposals,
                )
            else:
                execution = selected._execute(
                    view,
                    bank_state=bank_state,
                    proposals=bank_proposals,
                    resource_proposals=resource_proposals,
                )
            if selected.result_kind == "continue":
                result.append(
                    TensorViewFederalCandidate.local_view(
                        selected.action_id,
                        local_log_score=score,
                        next_view=execution.view,
                        bank_proposals=execution.bank_proposals,
                        resource_proposals=execution.resource_proposals,
                        producer=execution.producer,
                        effect_proposal=execution.effect_proposal,
                    )
                )
            else:
                assert selected.next_program_id is not None
                self.exit_pattern.validate(execution.view, name=f"{self.program_id}.exit")
                result.append(
                    TensorViewFederalCandidate.child_view(
                        selected.action_id,
                        local_log_score=score,
                        next_program_id=selected.next_program_id,
                        next_view=execution.view,
                        bank_proposals=execution.bank_proposals,
                        resource_proposals=execution.resource_proposals,
                        producer=execution.producer,
                        effect_proposal=execution.effect_proposal,
                    )
                )
        return tuple(result)

    def forward(
        self,
        view: TensorView,
        *,
        max_candidates: int,
        _bank_state: FormulaProgramBankState | None = None,
        _bank_proposals: _BankSlotProposals = (),
        _resource_proposals: _ResourceGraphProposals = (),
        _producer: _TensorViewProducerLineage | None = None,
    ) -> FederalBankStep:
        self.input_pattern.validate(view, name=f"{self.program_id}.input")
        if type(max_candidates) is not int or max_candidates <= 0:
            raise FederalRecallError("max_candidates must be a positive integer")
        bank_state = self.initial_bank_state() if _bank_state is None else _bank_state
        active = (
            _LocalViewPath(
                view,
                view.value.new_zeros(()),
                (),
                _bank_proposals,
                _resource_proposals,
                _producer,
            ),
        )
        completed: tuple[tuple[TensorViewFederalCandidate, tuple[str, ...]], ...] = ()
        trace: list[TensorViewLocalIterationTraceStep] = []
        for iteration in range(1, self.local_iteration.max_steps + 1):
            next_active: list[_LocalViewPath] = []
            next_completed = list(completed)
            saw_early_exit = False
            for branch in active:
                current_view = branch.view
                current_bank_proposals = branch.bank_proposals
                current_resource_proposals = branch.resource_proposals
                current_producer = branch.producer
                for direct in self.direct_actions:
                    execution = direct._execute(
                        current_view,
                        bank_proposals=current_bank_proposals,
                        proposals=current_resource_proposals,
                    )
                    current_view = execution.view
                    current_bank_proposals = execution.bank_proposals
                    current_resource_proposals = execution.resource_proposals
                    current_producer = execution.producer
                query = self.query(current_view)
                candidates = self._execute_once(
                    current_view,
                    query,
                    max_candidates=max_candidates,
                    bank_state=bank_state,
                    bank_proposals=current_bank_proposals,
                    resource_proposals=current_resource_proposals,
                    producer=current_producer,
                )
                for candidate in candidates:
                    lineage = (*branch.lineage, candidate.candidate_id)
                    trace_candidate = self._lineage_candidate(
                        candidate,
                        lineage=lineage,
                        preserve_identity=max_candidates == 1,
                    )
                    cumulative = branch.cumulative_log_score + candidate.local_log_score.to(
                        branch.cumulative_log_score
                    ).reshape(())
                    is_local = candidate.next_view is not None and candidate.next_program_id is None
                    if is_local:
                        trace.append(
                            self._trace(
                                step=iteration,
                                candidate=trace_candidate,
                                source=current_view,
                                query=query,
                                action="continue-local",
                                exit_reason=None,
                            )
                        )
                        if iteration < self.local_iteration.max_steps:
                            assert candidate.next_view is not None
                            next_active.append(
                                _LocalViewPath(
                                    candidate.next_view,
                                    cumulative,
                                    lineage,
                                    candidate.bank_proposals,
                                    candidate.resource_proposals,
                                    candidate.producer,
                                )
                            )
                        continue
                    if iteration < self.local_iteration.min_steps:
                        saw_early_exit = True
                        continue
                    trace.append(
                        self._trace(
                                step=iteration,
                                candidate=trace_candidate,
                                source=current_view,
                            query=query,
                            action=(
                                "terminal"
                                if candidate.terminal_outputs is not None
                                else "descend"
                            ),
                            exit_reason="formula-exit",
                        )
                    )
                    next_completed.append(
                        (replace(trace_candidate, local_log_score=cumulative), lineage)
                    )
            choices: list[
                tuple[str, _LocalViewPath | TensorViewFederalCandidate, tuple[str, ...]]
            ] = [("active", item, item.lineage) for item in next_active] + [
                ("completed", item, lineage) for item, lineage in next_completed
            ]
            choices.sort(key=self._choice_order)
            choices = choices[:max_candidates]
            active = tuple(
                item
                for kind, item, _lineage in choices
                if kind == "active" and isinstance(item, _LocalViewPath)
            )
            completed = tuple(
                (item, lineage)
                for kind, item, lineage in choices
                if kind == "completed" and isinstance(item, TensorViewFederalCandidate)
            )
            if not active:
                if completed:
                    return FederalBankStep(
                        tuple(item for item, _lineage in completed),
                        tuple(trace),
                    )
                if saw_early_exit:
                    raise FederalRecallError("Bank-local exit is invalid before min_steps")
                break
        if completed:
            return FederalBankStep(
                tuple(item for item, _lineage in completed),
                tuple(trace),
            )
        raise FederalRecallError(
            "TensorView Bank-local iteration reached max_steps without a valid exit"
        )


class _TensorViewProgramCollection(nn.Module, Mapping[str, RoutedProgram]):
    __hash__ = object.__hash__

    def __init__(self, programs: Mapping[str, RoutedProgram]) -> None:
        super().__init__()
        ordered = tuple(sorted(programs.items()))
        self._ids = tuple(program_id for program_id, _program in ordered)
        self._keys = {
            program_id: f"program_{hashlib.sha256(program_id.encode('utf-8')).hexdigest()}"
            for program_id in self._ids
        }
        self._programs = nn.ModuleDict(
            {self._keys[program_id]: program for program_id, program in ordered}
        )

    def __getitem__(self, program_id: str) -> RoutedProgram:
        return self._programs[self._keys[program_id]]

    def __iter__(self) -> Iterator[str]:
        return iter(self._ids)

    def __len__(self) -> int:
        return len(self._ids)


@dataclass(frozen=True)
class _ViewFederalPath:
    program_id: str
    view: TensorView | None
    cumulative_log_score: Tensor
    path: tuple[str, ...]
    terminal: FederalTerminalRecord | None = None
    bank_proposals: _BankSlotProposals = ()
    resource_proposals: _ResourceGraphProposals = ()
    producer: _TensorViewProducerLineage | None = None


class FederatedProgram(nn.Module):
    """K-wide composition of routed programs with winner-owned state writes."""

    _component_reference: ClassVar[str] = "arti/federated-program@1"
    recommended_breadth: ClassVar[int] = 8

    def __init__(
        self,
        programs: Mapping[str, RoutedProgram],
        *,
        terminal_abi: TerminalOutputABI,
        root_program_ids: tuple[str, ...],
        max_levels: int = 8,
        max_k: int = recommended_breadth,
        winner_policy: str = "hard_one_winner",
    ) -> None:
        super().__init__()
        if not isinstance(programs, Mapping) or not programs:
            raise FederalRecallError("programs must be a non-empty mapping")
        if not isinstance(terminal_abi, TerminalOutputABI):
            raise TypeError("terminal_abi must be TerminalOutputABI")
        modules: dict[str, RoutedProgram] = {}
        for program_id, program in programs.items():
            _require_name(program_id, field="program_id")
            if not isinstance(program, RoutedProgram):
                raise TypeError("FederatedProgram requires RoutedProgram values")
            if program.program_id != program_id:
                raise FederalRecallError("Program mapping key must match program.program_id")
            program.signature.validate_terminal_abi(terminal_abi)
            modules[program_id] = program
        roots = tuple(root_program_ids)
        if not roots or len(roots) != len(set(roots)) or any(root not in modules for root in roots):
            raise FederalRecallError("root_program_ids must be unique declared Programs")
        if type(max_levels) is not int or max_levels <= 0:
            raise FederalRecallError("max_levels must be a positive integer")
        if type(max_k) is not int or max_k <= 0:
            raise FederalRecallError("max_k must be a positive integer")
        if winner_policy != "hard_one_winner":
            raise FederalRecallError("FederatedProgram supports only hard_one_winner")
        score_fields = tuple(
            field.name for field in terminal_abi.fields if field.semantic_role == "terminal-score"
        )
        validity_fields = tuple(
            field.name
            for field in terminal_abi.fields
            if field.semantic_role == "terminal-validity"
        )
        if len(score_fields) != 1 or len(validity_fields) != 1:
            raise FederalRecallError(
                "TerminalOutputABI must define one score and one validity field"
            )
        plastic_actions = tuple(
            action for program in modules.values() for action in program.plastic_actions
        )
        slot_refs = tuple(action.bank_slot_ref for action in plastic_actions)
        assert all(item is not None for item in slot_refs)
        if len(set(slot_refs)) != len(slot_refs):
            raise FederalRecallError("Federation plastic Bank-slot references must be unique")
        self.programs = _TensorViewProgramCollection(modules)
        self.terminal_abi = terminal_abi
        self.root_program_ids = roots
        self.max_levels = max_levels
        self.max_k = max_k
        self.winner_policy = winner_policy
        self._score_field = score_fields[0]
        self._validity_field = validity_fields[0]
        self._plastic_actions = plastic_actions

    @property
    def effect_actions(self) -> tuple[TensorViewFormulaEffectAction, ...]:
        return tuple(
            action for program in self.programs.values() for action in program.effect_actions
        )

    @property
    def resource_actions(self) -> tuple[TensorViewResourceGraphAction, ...]:
        return tuple(
            action for program in self.programs.values() for action in program.all_resource_actions
        )

    def initial_bank_state(self) -> FormulaProgramBankState:
        refs = tuple(action.bank_slot_ref for action in self._plastic_actions)
        assert all(item is not None for item in refs)
        return FormulaProgramBankState(
            refs,  # type: ignore[arg-type]
            tuple(action.action.initial_bank_value() for action in self._plastic_actions),
            tuple(action.action.initial_revision() for action in self._plastic_actions),
        )

    def _install_bank_state(self, state: FormulaProgramBankState) -> None:
        for action in self._plastic_actions:
            action.action.install_(action.owner_program_id, state)

    def contract_config(self) -> dict[str, object]:
        return {
            "terminal_abi": self.terminal_abi.to_dict(),
            "root_program_ids": list(self.root_program_ids),
            "program_signatures": {
                program_id: self.programs[program_id].signature.to_dict()
                for program_id in sorted(self.programs)
            },
            "max_levels": self.max_levels,
            "max_k": self.max_k,
            "winner_policy": self.winner_policy,
        }

    @staticmethod
    def _path_order(path: _ViewFederalPath) -> tuple[float, str]:
        return (-float(path.cumulative_log_score.detach().cpu()), "/".join(path.path))

    def _terminal_record(
        self,
        *,
        bank_id: str,
        path: tuple[str, ...],
        outputs: Mapping[str, Tensor],
        cumulative_log_score: Tensor,
    ) -> FederalTerminalRecord:
        self.terminal_abi.validate_outputs(outputs)
        score = outputs[self._score_field]
        validity = outputs[self._validity_field]
        return FederalTerminalRecord(
            bank_id=bank_id,
            path=path,
            outputs=outputs,
            cumulative_log_score=cumulative_log_score,
            terminal_score=score.reshape(()),
            valid=bool(validity.reshape(()).detach().cpu()),
        )

    def _run_sample(
        self,
        view: TensorView,
        *,
        sample_index: int,
        root_program_id: str,
        max_levels: int,
        max_k: int,
        entry_bank_state: FormulaProgramBankState,
    ) -> tuple[
        FederalTerminalRecord,
        tuple[FederalTraceStep, ...],
        _BankSlotProposals,
        _ResourceGraphProposals,
    ]:
        paths = (
            _ViewFederalPath(
                root_program_id,
                view,
                view.value.new_zeros(()),
                (root_program_id,),
            ),
        )
        receipts: list[FederalTraceStep] = []
        for depth in range(1, max_levels + 1):
            expanded: list[_ViewFederalPath] = []
            local_receipts: list[LocalIterationTraceStep] = []
            for current in paths:
                if current.terminal is not None:
                    expanded.append(current)
                    continue
                assert current.view is not None
                step = self.programs[current.program_id](
                    current.view,
                    max_candidates=max_k,
                    _bank_state=entry_bank_state,
                    _bank_proposals=current.bank_proposals,
                    _resource_proposals=current.resource_proposals,
                    _producer=current.producer,
                )
                local_receipts.extend(step.local_trace)
                for raw_candidate in step.candidates:
                    if not isinstance(raw_candidate, TensorViewFederalCandidate):
                        raise FederalRecallError("FederalRecall@3 received a legacy candidate")
                    candidate = raw_candidate
                    cumulative = current.cumulative_log_score + candidate.local_log_score.to(
                        current.cumulative_log_score
                    ).reshape(())
                    next_path = (*current.path, candidate.candidate_id)
                    if candidate.terminal_outputs is not None:
                        terminal = self._terminal_record(
                            bank_id=current.program_id,
                            path=next_path,
                            outputs=candidate.terminal_outputs,
                            cumulative_log_score=cumulative,
                        )
                        expanded.append(
                            _ViewFederalPath(
                                current.program_id,
                                None,
                                cumulative,
                                next_path,
                                terminal=terminal,
                                bank_proposals=candidate.bank_proposals,
                                resource_proposals=candidate.resource_proposals,
                                producer=candidate.producer,
                            )
                        )
                        continue
                    assert candidate.next_program_id is not None
                    assert candidate.next_view is not None
                    if candidate.next_program_id not in self.programs:
                        raise FederalRecallError("candidate references an unknown child Program")
                    self.programs[candidate.next_program_id].input_pattern.validate(
                        candidate.next_view,
                        name=f"{candidate.next_program_id}.input",
                    )
                    expanded.append(
                        _ViewFederalPath(
                            candidate.next_program_id,
                            candidate.next_view,
                            cumulative,
                            (*next_path, candidate.next_program_id),
                            bank_proposals=candidate.bank_proposals,
                            resource_proposals=candidate.resource_proposals,
                            producer=candidate.producer,
                        )
                    )
            if not expanded:
                raise FederalRecallError("all federated paths ended without terminal output")
            expanded.sort(key=self._path_order)
            paths = tuple(expanded[:max_k])
            receipts.append(
                FederalTraceStep(
                    sample_index=sample_index,
                    depth=depth,
                    expanded_count=len(expanded),
                    kept_count=len(paths),
                    bank_ids=tuple(item.program_id for item in paths),
                    path_ids=tuple("/".join(item.path) for item in paths),
                    terminal_mask=tuple(item.terminal is not None for item in paths),
                    cumulative_log_scores=tuple(
                        float(item.cumulative_log_score.detach().cpu()) for item in paths
                    ),
                    local_iteration=tuple(local_receipts),
                )
            )
            if all(item.terminal is not None for item in paths):
                break
        terminal_paths = tuple(
            item for item in paths if item.terminal is not None and item.terminal.valid
        )
        if not terminal_paths:
            raise FederalRecallError("no valid terminal output was reached within max_levels")
        winner_path = min(
            terminal_paths,
            key=lambda item: (
                -float(item.terminal.terminal_score.detach().cpu()),
                "/".join(item.path),
            ),
        )
        assert winner_path.terminal is not None
        return (
            winner_path.terminal,
            tuple(receipts),
            winner_path.bank_proposals,
            winner_path.resource_proposals,
        )

    def forward(
        self,
        view: TensorView,
        *,
        root_program_id: str | None = None,
        max_levels: int | None = None,
        max_k: int | None = None,
        return_trace: bool = False,
    ) -> Mapping[str, Tensor] | tuple[Mapping[str, Tensor], TensorViewFederalTrace]:
        if not isinstance(view, TensorView):
            raise TypeError("FederalRecall@3 expects a TensorView")
        root = root_program_id
        if root is None:
            if len(self.root_program_ids) != 1:
                raise FederalRecallError("root_program_id is required when multiple roots exist")
            root = self.root_program_ids[0]
        if root not in self.root_program_ids:
            raise FederalRecallError("root_program_id is not declared by this Federation")
        levels = self.max_levels if max_levels is None else max_levels
        width = self.max_k if max_k is None else max_k
        if type(levels) is not int or not 0 < levels <= self.max_levels:
            raise FederalRecallError("max_levels must be within the configured bound")
        if type(width) is not int or not 0 < width <= self.max_k:
            raise FederalRecallError("max_k must be within the configured bound")
        self.programs[root].input_pattern.validate(view, name=f"{root}.input")
        if (self.effect_actions or self.resource_actions) and view.value.shape[view.batch_axis] != 1:
            raise FederalRecallError(
                "winner-owned mutable state commits currently require one invocation row"
            )
        entry_bank_state = self.initial_bank_state()
        winners: list[FederalTerminalRecord] = []
        winner_proposals: list[_BankSlotProposals] = []
        winner_resource_proposals: list[_ResourceGraphProposals] = []
        steps: list[FederalTraceStep] = []
        for sample_index in range(view.value.shape[view.batch_axis]):
            winner, sample_steps, proposals, resource_proposals = self._run_sample(
                view.slice_batch(sample_index),
                sample_index=sample_index,
                root_program_id=root,
                max_levels=levels,
                max_k=width,
                entry_bank_state=entry_bank_state,
            )
            winners.append(winner)
            winner_proposals.append(proposals)
            winner_resource_proposals.append(resource_proposals)
            steps.extend(sample_steps)
        outputs: dict[str, Tensor] = {}
        for field in self.terminal_abi.fields:
            try:
                outputs[field.name] = torch.cat(
                    [winner.outputs[field.name] for winner in winners], dim=0
                )
            except RuntimeError as exc:
                raise FederalRecallError(
                    f"terminal field {field.name!r} cannot be explicitly batched"
                ) from exc
        self.terminal_abi.validate_outputs(outputs)

        committed_effects: list[NeuralPlasticityCommitReceipt] = []
        for winner, proposals in zip(winners, winner_proposals, strict=True):
            if not proposals:
                continue
            successor_state = _committed_state(entry_bank_state, proposals)
            if return_trace:
                for proposal in proposals:
                    committed_effects.append(
                        NeuralPlasticityCommitReceipt(
                            bank_id=proposal.owner_bank_id,
                            producer_action_id=proposal.predecessor_action_id,
                            effect_action_id=proposal.effect_action_id,
                            bank_slot_ref=proposal.target,
                            effect_program_fingerprint=proposal.effect_program_fingerprint,
                            instruction_id=proposal.instruction_id,
                            effect_atom_ref=proposal.effect_atom_ref,
                            previous_revision=proposal.previous_revision,
                            successor_revision=proposal.successor_revision,
                            state_change_norm=float(
                                torch.linalg.vector_norm(
                                    (proposal.successor - proposal.previous)
                                    .detach()
                                    .to(torch.float32)
                                )
                                .cpu()
                                .reshape(())
                            ),
                            winner_path="/".join(winner.path),
                        )
                    )
            self._install_bank_state(successor_state)

        committed_resources: list[ResourceGraphCommitReceipt] = []
        for winner, resource_proposals in zip(
            winners, winner_resource_proposals, strict=True
        ):
            for proposal in resource_proposals:
                previous = proposal.graph.state() if return_trace else None
                proposal.graph.restore_state(proposal.state)
                if previous is not None:
                    previous_by_id = {
                        item.spec.resource_id: item for item in previous.resources
                    }
                    changed_resource_ids = tuple(
                        item.spec.resource_id
                        for item in proposal.state.resources
                        if (
                            item.epoch != previous_by_id[item.spec.resource_id].epoch
                            or item.step_index
                            != previous_by_id[item.spec.resource_id].step_index
                            or item.active_source
                            != previous_by_id[item.spec.resource_id].active_source
                        )
                    )
                    committed_resources.append(
                        ResourceGraphCommitReceipt(
                            proposal.graph.contract_fingerprint,
                            proposal.action_id,
                            proposal.connection_ids,
                            changed_resource_ids,
                            "/".join(winner.path),
                        )
                    )

        frozen = MappingProxyType(outputs)
        if not return_trace:
            return frozen
        return frozen, TensorViewFederalTrace(
            max_k=width,
            max_levels=levels,
            steps=tuple(steps),
            winner_paths=tuple("/".join(winner.path) for winner in winners),
            committed_effects=tuple(committed_effects),
            committed_resources=tuple(committed_resources),
        )


class FederatedProgramNode(ProgramNode):
    """Mount one federated region as a typed ``ProgramGraph`` node.

    The node deliberately owns no resource state.  Its enclosing graph supplies
    the input view and publishes the returned view into the declared output
    resource, so direct connections and local routing share one functional
    state trajectory.
    """

    _component_reference: ClassVar[str] = "arti/federated-program-node@1"

    def __init__(
        self,
        node_id: str,
        program: FederatedProgram,
        *,
        input_resource_id: str,
        output_resource_id: str,
        root_program_id: str | None = None,
        max_levels: int | None = None,
        max_k: int | None = None,
        value_field: str = "value",
        output_axis_names: tuple[str, ...] | None = None,
        output_axis_roles: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(
            node_id,
            input_resource_id=input_resource_id,
            output_resource_id=output_resource_id,
        )
        if not isinstance(program, FederatedProgram):
            raise TypeError("program must be FederatedProgram")
        if program.effect_actions or program.resource_actions:
            raise FederalRecallError(
                "FederatedProgramNode currently requires a state-free FederatedProgram; "
                "graph-owned functional publication for mutable program effects is not yet implemented"
            )
        if root_program_id is not None and not isinstance(root_program_id, str):
            raise TypeError("root_program_id must be a string or None")
        for name, value in (("max_levels", max_levels), ("max_k", max_k)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None")
        if not isinstance(value_field, str) or not value_field:
            raise ValueError("value_field must be a non-empty string")
        if (output_axis_names is None) != (output_axis_roles is None):
            raise ValueError("output_axis_names and output_axis_roles must be provided together")
        if output_axis_names is not None:
            if not output_axis_names or len(output_axis_names) != len(output_axis_roles or ()):
                raise ValueError("output-axis names and roles must have equal non-zero length")
            if len(set(output_axis_names)) != len(output_axis_names):
                raise ValueError("output_axis_names must be unique")
            if tuple(output_axis_roles or ()).count("batch") != 1:
                raise ValueError("output_axis_roles must contain exactly one batch role")
        self.program = program
        self.root_program_id = root_program_id
        self.max_levels = max_levels
        self.max_k = max_k
        self.value_field = value_field
        self.output_axis_names = None if output_axis_names is None else tuple(output_axis_names)
        self.output_axis_roles = None if output_axis_roles is None else tuple(output_axis_roles)

    def contract_config(self) -> dict[str, object]:
        return {
            **super().contract_config(),
            "program_ref": canonical_contract_reference(self.program._component_reference),
            "program_contract": self.program.contract_config(),
            "root_program_id": self.root_program_id,
            "max_levels": self.max_levels,
            "max_k": self.max_k,
            "value_field": self.value_field,
            "output_axis_names": None
            if self.output_axis_names is None
            else list(self.output_axis_names),
            "output_axis_roles": None
            if self.output_axis_roles is None
            else list(self.output_axis_roles),
        }

    def _output_view(self, value: Tensor, source: TensorView) -> TensorView:
        if self.output_axis_names is None:
            if tuple(value.shape) != tuple(source.value.shape):
                raise FederalRecallError(
                    "FederatedProgramNode requires output axes for a terminal value with a new shape"
                )
            return TensorView(value, source.axes, index_map=source.index_map, mask=source.mask)
        if len(self.output_axis_names) != value.ndim:
            raise FederalRecallError(
                "FederatedProgramNode output axis contract does not match terminal tensor rank"
            )
        assert self.output_axis_roles is not None
        axes = tuple(
            AxisDescriptor(name, role, int(extent))
            for name, role, extent in zip(
                self.output_axis_names, self.output_axis_roles, value.shape, strict=True
            )
        )
        return TensorView(value, axes)

    def invoke(self, view: TensorView) -> ProgramNodeInvocation:
        raw = self.program(
            view,
            root_program_id=self.root_program_id,
            max_levels=self.max_levels,
            max_k=self.max_k,
            return_trace=True,
        )
        outputs, trace = raw
        try:
            value = outputs[self.value_field]
        except KeyError as error:
            raise FederalRecallError(
                f"Program terminal output does not contain value field {self.value_field!r}"
            ) from error
        if not isinstance(value, Tensor):
            raise TypeError("FederatedProgramNode terminal value must be a Tensor")
        return ProgramNodeInvocation(self._output_view(value, view), trace)


class RoutedProgramNode(ProgramNode):
    """Mount one terminalizing local ``RoutedProgram`` into a graph.

    A local program keeps its own Query and execution loop.  The graph sees only
    its declared input view and one terminal output view; cross-program
    traversal belongs to :class:`FederatedProgramNode` instead.
    """

    _component_reference: ClassVar[str] = "arti/routed-program-node@1"

    def __init__(
        self,
        node_id: str,
        program: RoutedProgram,
        *,
        input_resource_id: str,
        output_resource_id: str,
        value_field: str = "value",
        output_axis_names: tuple[str, ...] | None = None,
        output_axis_roles: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__(
            node_id,
            input_resource_id=input_resource_id,
            output_resource_id=output_resource_id,
        )
        if not isinstance(program, RoutedProgram):
            raise TypeError("program must be RoutedProgram")
        if program.effect_actions or program.all_resource_actions:
            raise FederalRecallError(
                "RoutedProgramNode currently requires a state-free RoutedProgram; "
                "graph-owned functional publication for mutable program effects is not yet implemented"
            )
        if not isinstance(value_field, str) or not value_field:
            raise ValueError("value_field must be a non-empty string")
        if (output_axis_names is None) != (output_axis_roles is None):
            raise ValueError("output_axis_names and output_axis_roles must be provided together")
        if output_axis_names is not None:
            if not output_axis_names or len(output_axis_names) != len(output_axis_roles or ()):
                raise ValueError("output-axis names and roles must have equal non-zero length")
            if len(set(output_axis_names)) != len(output_axis_names):
                raise ValueError("output_axis_names must be unique")
            if tuple(output_axis_roles or ()).count("batch") != 1:
                raise ValueError("output_axis_roles must contain exactly one batch role")
        self.program = program
        self.value_field = value_field
        self.output_axis_names = None if output_axis_names is None else tuple(output_axis_names)
        self.output_axis_roles = None if output_axis_roles is None else tuple(output_axis_roles)

    def contract_config(self) -> dict[str, object]:
        return {
            **super().contract_config(),
            "program_ref": canonical_contract_reference(self.program._component_reference),
            "program_contract": self.program.contract_config(),
            "value_field": self.value_field,
            "output_axis_names": None
            if self.output_axis_names is None
            else list(self.output_axis_names),
            "output_axis_roles": None
            if self.output_axis_roles is None
            else list(self.output_axis_roles),
            "terminalization": "single-local-terminal",
        }

    def _output_view(self, value: Tensor, source: TensorView) -> TensorView:
        if self.output_axis_names is None:
            if tuple(value.shape) != tuple(source.value.shape):
                raise FederalRecallError(
                    "RoutedProgramNode requires output axes for a terminal value with a new shape"
                )
            return TensorView(value, source.axes, index_map=source.index_map, mask=source.mask)
        if len(self.output_axis_names) != value.ndim:
            raise FederalRecallError(
                "RoutedProgramNode output axis contract does not match terminal tensor rank"
            )
        assert self.output_axis_roles is not None
        axes = tuple(
            AxisDescriptor(name, role, int(extent))
            for name, role, extent in zip(
                self.output_axis_names, self.output_axis_roles, value.shape, strict=True
            )
        )
        return TensorView(value, axes)

    def invoke(self, view: TensorView) -> ProgramNodeInvocation:
        step = self.program(view, max_candidates=1)
        if len(step.candidates) != 1:
            raise FederalRecallError("RoutedProgramNode requires exactly one terminal candidate")
        candidate = step.candidates[0]
        if candidate.terminal_outputs is None:
            raise FederalRecallError(
                "RoutedProgramNode local execution did not reach a terminal output; "
                "mount a FederatedProgramNode for cross-program traversal"
            )
        try:
            value = candidate.terminal_outputs[self.value_field]
        except KeyError as error:
            raise FederalRecallError(
                f"Program terminal output does not contain value field {self.value_field!r}"
            ) from error
        if not isinstance(value, Tensor):
            raise TypeError("RoutedProgramNode terminal value must be a Tensor")
        return ProgramNodeInvocation(self._output_view(value, view), step.local_trace)


__all__ = [
    "FEDERAL_RECALL_V3_VERSION",
    "TENSOR_VIEW_FEDERAL_TRACE_VERSION",
    "FederatedProgram",
    "FederatedProgramNode",
    "NeuralPlasticityCommitReceipt",
    "ResourceGraphCommitReceipt",
    "TensorViewFederalCandidate",
    "TensorViewFormulaEffectAction",
    "TensorViewFormulaAction",
    "RoutedProgram",
    "RoutedProgramNode",
    "TensorViewResourceGraphAction",
    "TensorViewFederalTrace",
    "TensorViewLayoutTransition",
    "TensorViewLocalIterationTraceStep",
]
