"""Shape-polymorphic federated Banks with TensorView-local Refine."""

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

from .bank_local_program import BankLocalFormulaAction, BankLocalTerminalAction
from .component_registry import component_ref
from .federal_recall import (
    BankLocalRefinePolicy,
    BankLocalRefineTraceStep,
    FederalBankStep,
    FederalCandidate,
    FederalRecallError,
    FederalTerminalRecord,
    FederalTrace,
    FederalTraceStep,
)
from .shape_query import (
    SealedTensorViewBankQuery,
    TensorViewQueryResult,
)
from .tensor_schema import GradientContract
from .tensor_view import AxisDescriptor, TensorIndexMap, TensorView, TensorViewPattern
from .terminal_abi import BankExecutionSignatureV3, TerminalOutputABI


FEDERAL_RECALL_V3_VERSION = 3

_NAME = re.compile(r"^[a-z][a-z0-9]*(?:[-_.][a-z0-9]+)*$")
TensorViewActionKind = Literal["continue", "descend"]
IndexTransitionKind = Literal["identity", "preserve_flat", "permute"]


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
            "ref": self._component_reference,
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
    """Run one typed Formula action and expose its successor TensorView."""

    _component_reference: ClassVar[str] = "arti/tensor-view-formula-action@1"

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

    @property
    def action_id(self) -> str:
        return self.action.action_id

    @property
    def result_kind(self) -> TensorViewActionKind:
        return self.action.result_kind

    @property
    def next_bank_id(self) -> str | None:
        return self.action.next_bank_id

    def accepts(self, view: TensorView) -> bool:
        return isinstance(view, TensorView) and self.action.accepts(view.value)

    def contract_config(self) -> dict[str, object]:
        return {
            "action": self.action.contract_config(),
            "layout": self.layout.to_dict(),
            "state_transform_owner": "formula-fabric",
        }

    def forward(self, view: TensorView) -> TensorView:
        if not isinstance(view, TensorView):
            raise TypeError("TensorViewFormulaAction expects TensorView")
        output = self.action(view.value)
        return self.layout.apply(view, output)


@dataclass(frozen=True)
class TensorViewFederalCandidate(FederalCandidate):
    """A federal candidate that retains the Formula-produced successor view."""

    next_view: TensorView | None = None

    _runtime_contract_ref: ClassVar[str] = "arti/federal-candidate@2"

    def __post_init__(self) -> None:
        super().__post_init__()
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
    ) -> TensorViewFederalCandidate:
        return cls(
            candidate_id,
            local_log_score,
            next_value=next_view.value,
            next_view=next_view,
        )

    @classmethod
    def child_view(
        cls,
        candidate_id: str,
        *,
        local_log_score: Tensor,
        next_bank_id: str,
        next_view: TensorView,
    ) -> TensorViewFederalCandidate:
        return cls(
            candidate_id,
            local_log_score,
            next_bank_id=next_bank_id,
            next_value=next_view.value,
            next_view=next_view,
        )

    @classmethod
    def terminal_view(
        cls,
        candidate_id: str,
        *,
        local_log_score: Tensor,
        outputs: Mapping[str, Tensor],
    ) -> TensorViewFederalCandidate:
        return cls(
            candidate_id,
            local_log_score,
            terminal_outputs=outputs,
        )


@dataclass(frozen=True)
class TensorViewLocalRefineTraceStep(BankLocalRefineTraceStep):
    """Local refine evidence including logical view identities."""

    input_view_fingerprint: str = ""
    output_view_fingerprint: str | None = None
    input_axes: tuple[str, ...] = ()
    output_axes: tuple[str, ...] | None = None

    _runtime_contract_ref: ClassVar[str] = "arti/bank-local-refine-trace-step@2"

    def __post_init__(self) -> None:
        super().__post_init__()
        object.__setattr__(self, "input_axes", tuple(self.input_axes))
        if self.output_axes is not None:
            object.__setattr__(self, "output_axes", tuple(self.output_axes))

    def to_dict(self) -> dict[str, object]:
        return {
            **super().to_dict(),
            "ref": self._runtime_contract_ref,
            "input_view_fingerprint": self.input_view_fingerprint,
            "output_view_fingerprint": self.output_view_fingerprint,
            "input_axes": list(self.input_axes),
            "output_axes": None if self.output_axes is None else list(self.output_axes),
        }


@dataclass(frozen=True)
class _LocalViewPath:
    view: TensorView
    cumulative_log_score: Tensor
    lineage: tuple[str, ...]


class TensorViewFormulaProgram(nn.Module):
    """A Bank that re-queries its latest Formula-produced TensorView."""

    _component_reference: ClassVar[str] = "arti/tensor-view-formula-program@1"

    def __init__(
        self,
        *,
        bank_id: str,
        query: SealedTensorViewBankQuery,
        actions: Sequence[TensorViewFormulaAction],
        terminal_action: BankLocalTerminalAction,
        local_refine: BankLocalRefinePolicy,
        input_pattern: TensorViewPattern,
        exit_pattern: TensorViewPattern,
        terminal_abi: TerminalOutputABI,
    ) -> None:
        super().__init__()
        _require_name(bank_id, field="bank_id")
        if not isinstance(query, SealedTensorViewBankQuery):
            raise TypeError("query must be SealedTensorViewBankQuery")
        normalized = tuple(actions)
        if not normalized or any(
            not isinstance(item, TensorViewFormulaAction) for item in normalized
        ):
            raise TypeError("actions must contain TensorViewFormulaAction values")
        if not isinstance(terminal_action, BankLocalTerminalAction):
            raise TypeError("terminal_action must be BankLocalTerminalAction")
        if not isinstance(local_refine, BankLocalRefinePolicy):
            raise TypeError("local_refine must be BankLocalRefinePolicy")
        if not isinstance(input_pattern, TensorViewPattern) or not isinstance(
            exit_pattern, TensorViewPattern
        ):
            raise TypeError("input and exit patterns must be TensorViewPattern")
        if query.signature.pattern.fingerprint != input_pattern.fingerprint:
            raise FederalRecallError("Bank input pattern must match its sealed Query")
        if not isinstance(terminal_abi, TerminalOutputABI):
            raise TypeError("terminal_abi must be TerminalOutputABI")
        action_ids = tuple(item.action_id for item in normalized) + (terminal_action.action_id,)
        if action_ids != query.signature.member_ids:
            raise FederalRecallError(
                "sealed TensorView Query members must exactly match local actions"
            )
        self.bank_id = bank_id
        self.query = query
        self.actions = nn.ModuleList(normalized)
        self.terminal_action = terminal_action
        self.local_refine = local_refine
        self.input_pattern = input_pattern
        self.exit_pattern = exit_pattern
        self.terminal_abi = terminal_abi
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
            local_formula_ref="arti/formula-fabric@2",
            local_refine_ref=component_ref(local_refine),
            execution_capabilities=(
                "eager",
                "fixed-k-wide",
                "latest-tensor-view-requery",
                "variable-rank-local-refine",
            ),
        )

    @property
    def signature(self) -> BankExecutionSignatureV3:
        return self._signature

    @property
    def action_ids(self) -> tuple[str, ...]:
        return tuple(item.action_id for item in self.actions) + (self.terminal_action.action_id,)

    def contract_config(self) -> dict[str, object]:
        return {
            "bank_id": self.bank_id,
            "actions": [item.contract_config() for item in self.actions],
            "terminal_action": self.terminal_action.contract_config(),
            "local_refine": self.local_refine.contract_config(),
            "input_pattern": self.input_pattern.to_dict(),
            "exit_pattern": self.exit_pattern.to_dict(),
            "terminal_abi_fingerprint": self.terminal_abi.fingerprint,
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
    ) -> TensorViewLocalRefineTraceStep:
        target = candidate.next_view
        return TensorViewLocalRefineTraceStep(
            bank_id=self.bank_id,
            local_step=step,
            candidate_id=candidate.candidate_id,
            action=action,
            input_shape=tuple(int(size) for size in source.value.shape),
            query_shape=tuple(int(size) for size in query.scores.shape),
            output_shape=(
                None if target is None else tuple(int(size) for size in target.value.shape)
            ),
            query_ref=self.query.signature.query_ref,
            query_state_fingerprint=self.query.signature.state_fingerprint,
            formula_ref="arti/formula-fabric@2",
            exit_reason=exit_reason,
            input_view_fingerprint=source.descriptor_fingerprint,
            output_view_fingerprint=(None if target is None else target.descriptor_fingerprint),
            input_axes=tuple(axis.name for axis in source.axes),
            output_axes=(None if target is None else tuple(axis.name for axis in target.axes)),
        )

    def _execute_once(
        self,
        view: TensorView,
        query: TensorViewQueryResult,
        *,
        max_candidates: int,
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
                ),
            )
        eligible = [index for index, action in enumerate(self.actions) if action.accepts(view)]
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
                    )
                )
                continue
            action = self.actions[index]
            next_view = action(view)
            if action.result_kind == "continue":
                result.append(
                    TensorViewFederalCandidate.local_view(
                        action.action_id,
                        local_log_score=score,
                        next_view=next_view,
                    )
                )
            else:
                assert action.next_bank_id is not None
                self.exit_pattern.validate(next_view, name=f"{self.bank_id}.exit")
                result.append(
                    TensorViewFederalCandidate.child_view(
                        action.action_id,
                        local_log_score=score,
                        next_bank_id=action.next_bank_id,
                        next_view=next_view,
                    )
                )
        return tuple(result)

    def forward(self, view: TensorView, *, max_candidates: int) -> FederalBankStep:
        self.input_pattern.validate(view, name=f"{self.bank_id}.input")
        if type(max_candidates) is not int or max_candidates <= 0:
            raise FederalRecallError("max_candidates must be a positive integer")
        active = (_LocalViewPath(view, view.value.new_zeros(()), ()),)
        completed: tuple[tuple[TensorViewFederalCandidate, tuple[str, ...]], ...] = ()
        trace: list[TensorViewLocalRefineTraceStep] = []
        for local_step in range(1, self.local_refine.max_steps + 1):
            next_active: list[_LocalViewPath] = []
            next_completed = list(completed)
            saw_early_exit = False
            for branch in active:
                query = self.query(branch.view)
                candidates = self._execute_once(
                    branch.view,
                    query,
                    max_candidates=max_candidates,
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
                    is_local = candidate.next_view is not None and candidate.next_bank_id is None
                    if is_local:
                        trace.append(
                            self._trace(
                                step=local_step,
                                candidate=trace_candidate,
                                source=branch.view,
                                query=query,
                                action="continue-local",
                                exit_reason=None,
                            )
                        )
                        if local_step < self.local_refine.max_steps:
                            assert candidate.next_view is not None
                            next_active.append(
                                _LocalViewPath(candidate.next_view, cumulative, lineage)
                            )
                        continue
                    if local_step < self.local_refine.min_steps:
                        saw_early_exit = True
                        continue
                    trace.append(
                        self._trace(
                            step=local_step,
                            candidate=trace_candidate,
                            source=branch.view,
                            query=query,
                            action=(
                                "terminal" if candidate.terminal_outputs is not None else "descend"
                            ),
                            exit_reason="formula-exit",
                        )
                    )
                    next_completed.append(
                        (
                            replace(trace_candidate, local_log_score=cumulative),
                            lineage,
                        )
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
            "TensorView Bank-local Refine reached max_steps without a valid exit"
        )


class _TensorViewBankCollection(nn.Module, Mapping[str, TensorViewFormulaProgram]):
    __hash__ = object.__hash__

    def __init__(self, programs: Mapping[str, TensorViewFormulaProgram]) -> None:
        super().__init__()
        ordered = tuple(sorted(programs.items()))
        self._ids = tuple(bank_id for bank_id, _program in ordered)
        self._keys = {
            bank_id: f"bank_{hashlib.sha256(bank_id.encode('utf-8')).hexdigest()}"
            for bank_id in self._ids
        }
        self._programs = nn.ModuleDict(
            {self._keys[bank_id]: program for bank_id, program in ordered}
        )

    def __getitem__(self, bank_id: str) -> TensorViewFormulaProgram:
        return self._programs[self._keys[bank_id]]

    def __iter__(self) -> Iterator[str]:
        return iter(self._ids)

    def __len__(self) -> int:
        return len(self._ids)


@dataclass(frozen=True)
class _ViewFederalPath:
    bank_id: str
    view: TensorView | None
    cumulative_log_score: Tensor
    path: tuple[str, ...]
    terminal: FederalTerminalRecord | None = None


class FederalRecallV3(nn.Module):
    """K-wide Federation over Banks with arbitrary-rank local TensorView Refine."""

    _component_reference: ClassVar[str] = "arti/federal-recall@3"
    recommended_breadth: ClassVar[int] = 8

    def __init__(
        self,
        banks: Mapping[str, TensorViewFormulaProgram],
        *,
        terminal_abi: TerminalOutputABI,
        root_bank_ids: tuple[str, ...],
        max_levels: int = 8,
        max_k: int = recommended_breadth,
        winner_policy: str = "hard_one_winner",
    ) -> None:
        super().__init__()
        if not isinstance(banks, Mapping) or not banks:
            raise FederalRecallError("banks must be a non-empty mapping")
        if not isinstance(terminal_abi, TerminalOutputABI):
            raise TypeError("terminal_abi must be TerminalOutputABI")
        modules: dict[str, TensorViewFormulaProgram] = {}
        for bank_id, program in banks.items():
            _require_name(bank_id, field="bank_id")
            if not isinstance(program, TensorViewFormulaProgram):
                raise TypeError("FederalRecall@3 requires TensorViewFormulaProgram values")
            if program.bank_id != bank_id:
                raise FederalRecallError("Bank mapping key must match program.bank_id")
            program.signature.validate_terminal_abi(terminal_abi)
            modules[bank_id] = program
        roots = tuple(root_bank_ids)
        if not roots or len(roots) != len(set(roots)) or any(root not in modules for root in roots):
            raise FederalRecallError("root_bank_ids must be unique declared Banks")
        if type(max_levels) is not int or max_levels <= 0:
            raise FederalRecallError("max_levels must be a positive integer")
        if type(max_k) is not int or max_k <= 0:
            raise FederalRecallError("max_k must be a positive integer")
        if winner_policy != "hard_one_winner":
            raise FederalRecallError("FederalRecall@3 supports only hard_one_winner")
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
        self.banks = _TensorViewBankCollection(modules)
        self.terminal_abi = terminal_abi
        self.root_bank_ids = roots
        self.max_levels = max_levels
        self.max_k = max_k
        self.winner_policy = winner_policy
        self._score_field = score_fields[0]
        self._validity_field = validity_fields[0]

    def contract_config(self) -> dict[str, object]:
        return {
            "terminal_abi": self.terminal_abi.to_dict(),
            "root_bank_ids": list(self.root_bank_ids),
            "bank_signatures": {
                bank_id: self.banks[bank_id].signature.to_dict() for bank_id in sorted(self.banks)
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
        root_bank_id: str,
        max_levels: int,
        max_k: int,
    ) -> tuple[FederalTerminalRecord, tuple[FederalTraceStep, ...]]:
        paths = (
            _ViewFederalPath(
                root_bank_id,
                view,
                view.value.new_zeros(()),
                (root_bank_id,),
            ),
        )
        receipts: list[FederalTraceStep] = []
        for depth in range(1, max_levels + 1):
            expanded: list[_ViewFederalPath] = []
            local_receipts: list[BankLocalRefineTraceStep] = []
            for current in paths:
                if current.terminal is not None:
                    expanded.append(current)
                    continue
                assert current.view is not None
                step = self.banks[current.bank_id](
                    current.view,
                    max_candidates=max_k,
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
                            bank_id=current.bank_id,
                            path=next_path,
                            outputs=candidate.terminal_outputs,
                            cumulative_log_score=cumulative,
                        )
                        expanded.append(
                            _ViewFederalPath(
                                current.bank_id,
                                None,
                                cumulative,
                                next_path,
                                terminal,
                            )
                        )
                        continue
                    assert candidate.next_bank_id is not None
                    assert candidate.next_view is not None
                    if candidate.next_bank_id not in self.banks:
                        raise FederalRecallError("candidate references an unknown child Bank")
                    self.banks[candidate.next_bank_id].input_pattern.validate(
                        candidate.next_view,
                        name=f"{candidate.next_bank_id}.input",
                    )
                    expanded.append(
                        _ViewFederalPath(
                            candidate.next_bank_id,
                            candidate.next_view,
                            cumulative,
                            (*next_path, candidate.next_bank_id),
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
                    bank_ids=tuple(item.bank_id for item in paths),
                    path_ids=tuple("/".join(item.path) for item in paths),
                    terminal_mask=tuple(item.terminal is not None for item in paths),
                    cumulative_log_scores=tuple(
                        float(item.cumulative_log_score.detach().cpu()) for item in paths
                    ),
                    local_refine=tuple(local_receipts),
                )
            )
            if all(item.terminal is not None for item in paths):
                break
        terminals = tuple(
            item.terminal for item in paths if item.terminal is not None and item.terminal.valid
        )
        if not terminals:
            raise FederalRecallError("no valid terminal output was reached within max_levels")
        winner = min(
            terminals,
            key=lambda item: (
                -float(item.terminal_score.detach().cpu()),
                "/".join(item.path),
            ),
        )
        return winner, tuple(receipts)

    def forward(
        self,
        view: TensorView,
        *,
        root_bank_id: str | None = None,
        max_levels: int | None = None,
        max_k: int | None = None,
        return_trace: bool = False,
    ) -> Mapping[str, Tensor] | tuple[Mapping[str, Tensor], FederalTrace]:
        if not isinstance(view, TensorView):
            raise TypeError("FederalRecall@3 expects a TensorView")
        root = root_bank_id
        if root is None:
            if len(self.root_bank_ids) != 1:
                raise FederalRecallError("root_bank_id is required when multiple roots exist")
            root = self.root_bank_ids[0]
        if root not in self.root_bank_ids:
            raise FederalRecallError("root_bank_id is not declared by this Federation")
        levels = self.max_levels if max_levels is None else max_levels
        width = self.max_k if max_k is None else max_k
        if type(levels) is not int or not 0 < levels <= self.max_levels:
            raise FederalRecallError("max_levels must be within the configured bound")
        if type(width) is not int or not 0 < width <= self.max_k:
            raise FederalRecallError("max_k must be within the configured bound")
        self.banks[root].input_pattern.validate(view, name=f"{root}.input")
        winners: list[FederalTerminalRecord] = []
        steps: list[FederalTraceStep] = []
        for sample_index in range(view.value.shape[view.batch_axis]):
            winner, sample_steps = self._run_sample(
                view.slice_batch(sample_index),
                sample_index=sample_index,
                root_bank_id=root,
                max_levels=levels,
                max_k=width,
            )
            winners.append(winner)
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
        frozen = MappingProxyType(outputs)
        if not return_trace:
            return frozen
        return frozen, FederalTrace(
            max_k=width,
            max_levels=levels,
            steps=tuple(steps),
            winner_paths=tuple("/".join(winner.path) for winner in winners),
        )


__all__ = [
    "FEDERAL_RECALL_V3_VERSION",
    "FederalRecallV3",
    "TensorViewFederalCandidate",
    "TensorViewFormulaAction",
    "TensorViewFormulaProgram",
    "TensorViewLayoutTransition",
    "TensorViewLocalRefineTraceStep",
]
