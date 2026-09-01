"""Shape-autonomous eager reference execution for federated ARTI Banks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
import hashlib
import math
import re
from types import MappingProxyType
from collections.abc import Iterator
from typing import ClassVar, Mapping

import torch
from torch import Tensor, nn

from .bank_query import BankQueryError, BankQueryResult, SealedBankQuery
from .terminal_abi import (
    BankExecutionSignature,
    BankExecutionSignatureV2,
    TerminalOutputABI,
)


FEDERAL_RECALL_VERSION = 1
FEDERAL_RECALL_V2_VERSION = 2
FEDERAL_TRACE_VERSION = 2

_NAME = re.compile(r"^[a-z][a-z0-9]*(?:[-_.][a-z0-9]+)*$")


class FederalRecallError(ValueError):
    """Raised when a federated Bank path violates its declared contracts."""


class BankLocalRefinePolicy(nn.Module):
    """Bound the number of latest-state re-queries performed inside one Bank."""

    _component_reference: ClassVar[str] = "arti/bank-local-refine-policy@1"

    def __init__(self, *, min_steps: int = 1, max_steps: int = 8) -> None:
        super().__init__()
        if (
            isinstance(min_steps, bool)
            or not isinstance(min_steps, int)
            or min_steps <= 0
        ):
            raise FederalRecallError("min_steps must be a positive integer")
        if (
            isinstance(max_steps, bool)
            or not isinstance(max_steps, int)
            or max_steps < min_steps
        ):
            raise FederalRecallError("max_steps must be an integer no smaller than min_steps")
        self.min_steps = int(min_steps)
        self.max_steps = int(max_steps)

    def contract_config(self) -> dict[str, object]:
        return {
            "min_steps": self.min_steps,
            "max_steps": self.max_steps,
            "state_source": "latest-local-state",
            "exit_semantics": "formula-request-after-min-steps",
        }


def _require_name(value: str, *, field: str) -> None:
    if not isinstance(value, str) or _NAME.fullmatch(value) is None:
        raise FederalRecallError(f"{field} must be a canonical lowercase name")


def _require_scalar_score(value: Tensor, *, field: str) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point() or value.numel() != 1:
        raise FederalRecallError(f"{field} must be one floating-point scalar")
    if not bool(torch.isfinite(value.detach()).all()):
        raise FederalRecallError(f"{field} must be finite")


def _freeze_outputs(outputs: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    if not isinstance(outputs, Mapping):
        raise TypeError("terminal_outputs must be a tensor mapping")
    frozen: dict[str, Tensor] = {}
    for name, value in sorted(outputs.items()):
        _require_name(name, field="terminal output name")
        if not isinstance(value, Tensor):
            raise TypeError("terminal output values must be Tensor values")
        frozen[name] = value
    return MappingProxyType(frozen)


@dataclass(frozen=True)
class FederalCandidate:
    """One local Bank result that either descends or terminalizes."""

    candidate_id: str
    local_log_score: Tensor
    next_bank_id: str | None = None
    next_value: Tensor | None = None
    terminal_outputs: Mapping[str, Tensor] | None = None

    _runtime_contract_ref: ClassVar[str] = "arti/federal-candidate@1"

    def __post_init__(self) -> None:
        _require_name(self.candidate_id, field="candidate_id")
        _require_scalar_score(self.local_log_score, field="local_log_score")
        is_local = self.next_bank_id is None and self.next_value is not None
        is_child = self.next_bank_id is not None and self.next_value is not None
        is_terminal = self.terminal_outputs is not None
        if sum((is_local, is_child, is_terminal)) != 1:
            raise FederalRecallError(
                "candidate must provide exactly one local, child, or terminal transition"
            )
        if is_local or is_child:
            assert self.next_value is not None
            if is_child:
                assert self.next_bank_id is not None
                _require_name(self.next_bank_id, field="next_bank_id")
            elif self.next_bank_id is not None:
                raise FederalRecallError("local candidate cannot select a child Bank")
            if not isinstance(self.next_value, Tensor) or not self.next_value.is_floating_point():
                raise TypeError("next_value must be a floating-point Tensor")
            if self.next_value.ndim < 1 or self.next_value.shape[0] != 1:
                raise FederalRecallError("next_value must describe one branch with batch size 1")
        else:
            object.__setattr__(self, "terminal_outputs", _freeze_outputs(self.terminal_outputs))

    @classmethod
    def child(
        cls,
        candidate_id: str,
        *,
        local_log_score: Tensor,
        next_bank_id: str,
        next_value: Tensor,
    ) -> FederalCandidate:
        return cls(
            candidate_id=candidate_id,
            local_log_score=local_log_score,
            next_bank_id=next_bank_id,
            next_value=next_value,
        )

    @classmethod
    def local(
        cls,
        candidate_id: str,
        *,
        local_log_score: Tensor,
        next_value: Tensor,
    ) -> FederalCandidate:
        return cls(
            candidate_id=candidate_id,
            local_log_score=local_log_score,
            next_value=next_value,
        )

    @classmethod
    def terminal(
        cls,
        candidate_id: str,
        *,
        local_log_score: Tensor,
        outputs: Mapping[str, Tensor],
    ) -> FederalCandidate:
        return cls(
            candidate_id=candidate_id,
            local_log_score=local_log_score,
            terminal_outputs=outputs,
        )


@dataclass(frozen=True)
class BankLocalRefineTraceStep:
    """JSON-safe evidence for one Query/Formula transition inside a Bank."""

    bank_id: str
    local_step: int
    candidate_id: str
    action: str
    input_shape: tuple[int, ...]
    query_shape: tuple[int, ...]
    output_shape: tuple[int, ...] | None
    query_ref: str
    query_state_fingerprint: str
    formula_ref: str | None
    exit_reason: str | None

    _runtime_contract_ref: ClassVar[str] = "arti/bank-local-refine-trace-step@1"

    def __post_init__(self) -> None:
        _require_name(self.bank_id, field="bank_id")
        _require_name(self.candidate_id, field="candidate_id")
        if self.action not in {"continue-local", "descend", "terminal"}:
            raise FederalRecallError("local refine trace action is invalid")
        if isinstance(self.local_step, bool) or not isinstance(self.local_step, int) or self.local_step <= 0:
            raise FederalRecallError("local_step must be a positive integer")
        object.__setattr__(self, "input_shape", tuple(self.input_shape))
        object.__setattr__(self, "query_shape", tuple(self.query_shape))
        if self.output_shape is not None:
            object.__setattr__(self, "output_shape", tuple(self.output_shape))

    def to_dict(self) -> dict[str, object]:
        return {
            "ref": self._runtime_contract_ref,
            "bank_id": self.bank_id,
            "local_step": self.local_step,
            "candidate_id": self.candidate_id,
            "action": self.action,
            "input_shape": list(self.input_shape),
            "query_shape": list(self.query_shape),
            "output_shape": None if self.output_shape is None else list(self.output_shape),
            "query_ref": self.query_ref,
            "query_state_fingerprint": self.query_state_fingerprint,
            "formula_ref": self.formula_ref,
            "exit_reason": self.exit_reason,
        }


@dataclass(frozen=True)
class FederalBankStep:
    """The bounded candidates returned by one autonomous Bank invocation."""

    candidates: tuple[FederalCandidate, ...]
    local_trace: tuple[BankLocalRefineTraceStep, ...] = ()

    _runtime_contract_ref: ClassVar[str] = "arti/federal-bank-step@1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "local_trace", tuple(self.local_trace))
        if any(not isinstance(item, FederalCandidate) for item in self.candidates):
            raise TypeError("candidates must contain FederalCandidate values")
        if any(not isinstance(item, BankLocalRefineTraceStep) for item in self.local_trace):
            raise TypeError("local_trace must contain BankLocalRefineTraceStep values")
        identities = tuple(item.candidate_id for item in self.candidates)
        if len(identities) != len(set(identities)):
            raise FederalRecallError("candidate_id values must be unique within one Bank step")


class AutonomousBankProgram(nn.Module, ABC):
    """Base class for one shape-autonomous Bank program used by FederalRecall."""

    _component_reference: ClassVar[str] = "arti/autonomous-bank-program@1"

    def __init__(self, *, bank_id: str, signature: BankExecutionSignature) -> None:
        super().__init__()
        _require_name(bank_id, field="bank_id")
        if not isinstance(signature, BankExecutionSignature):
            raise TypeError("signature must be BankExecutionSignature")
        self.bank_id = bank_id
        self.signature = signature

    @abstractmethod
    def forward(self, value: Tensor, *, max_candidates: int) -> FederalBankStep:
        """Execute one local Bank step without imposing a shared internal shape."""


class BankOwnedQueryProgram(nn.Module, ABC):
    """A Bank program whose sealed local Query is executed on every invocation."""

    _component_reference: ClassVar[str] = "arti/bank-owned-query-program@1"

    def __init__(
        self,
        *,
        bank_id: str,
        query: SealedBankQuery,
        local_refine: BankLocalRefinePolicy | None = None,
    ) -> None:
        super().__init__()
        _require_name(bank_id, field="bank_id")
        if not isinstance(query, SealedBankQuery):
            raise TypeError("query must be SealedBankQuery")
        if local_refine is not None and not isinstance(
            local_refine, BankLocalRefinePolicy
        ):
            raise TypeError("local_refine must be BankLocalRefinePolicy or None")
        self.bank_id = bank_id
        self.query = query
        self.local_refine = local_refine
        self._signature: BankExecutionSignatureV2 | None = None

    @property
    def signature(self) -> BankExecutionSignatureV2:
        if self._signature is None:
            raise FederalRecallError(
                "Bank-owned Query program must bind its execution signature after construction"
            )
        return self._signature

    def bind_signature(
        self,
        signature: BankExecutionSignatureV2,
    ) -> BankOwnedQueryProgram:
        if self._signature is not None:
            raise FederalRecallError("Bank execution signature is already bound")
        if not isinstance(signature, BankExecutionSignatureV2):
            raise TypeError("signature must be BankExecutionSignatureV2")
        try:
            self.query.signature.validate_query(self.query.query)
        except BankQueryError as exc:
            raise FederalRecallError("mounted Bank Query asset is invalid") from exc
        if self.query.signature != signature.query_signature:
            raise FederalRecallError(
                "mounted Bank Query does not match the Bank execution signature"
            )
        if self.local_refine is not None:
            from .component_registry import component_ref

            if signature.local_refine_ref != component_ref(self.local_refine):
                raise FederalRecallError(
                    "Bank execution signature does not match its local refine policy"
                )
        if (
            dict(signature.local_normalization_contract)
            != dict(self.query.signature.normalization_contract)
        ):
            raise FederalRecallError(
                "Bank execution and owned Query normalization contracts disagree"
            )
        from .component_registry import component_spec

        spec = component_spec(self)
        if (
            signature.program_ref != spec.reference
            or signature.program_api_identity != spec.api
            or signature.program_config_fingerprint != spec.config_fingerprint
            or signature.program_state_schema_fingerprint
            != spec.parameter_schema_fingerprint
        ):
            raise FederalRecallError(
                "Bank execution signature does not match the constructed program"
            )
        self._signature = signature
        return self

    @abstractmethod
    def execute(
        self,
        value: Tensor,
        *,
        query_result: BankQueryResult,
        max_candidates: int,
    ) -> FederalBankStep:
        """Execute one local step using this invocation's latest Query result."""

    def forward(self, value: Tensor, *, max_candidates: int) -> FederalBankStep:
        self.signature.input_schema.validate_tensor(
            value,
            name=f"{self.bank_id}.input",
        )
        policy = self.local_refine
        if policy is not None and max_candidates != 1:
            raise FederalRecallError("Bank-local Refine@1 requires serial K=1 execution")
        min_steps = 1 if policy is None else policy.min_steps
        max_steps = 1 if policy is None else policy.max_steps
        current = value
        accumulated_score = value.new_zeros(())
        trace: list[BankLocalRefineTraceStep] = []
        for local_step in range(1, max_steps + 1):
            query_result = self.query(current)
            step = self.execute(
                current,
                query_result=query_result,
                max_candidates=max_candidates,
            )
            if not isinstance(step, FederalBankStep):
                raise TypeError("BankOwnedQueryProgram.execute must return FederalBankStep")
            if step.local_trace:
                raise FederalRecallError(
                    "BankOwnedQueryProgram.execute cannot forge local refine receipts"
                )
            if len(step.candidates) > max_candidates:
                raise FederalRecallError("local Bank returned more than the global K limit")
            local_candidates = tuple(
                candidate
                for candidate in step.candidates
                if candidate.next_value is not None and candidate.next_bank_id is None
            )
            if local_candidates:
                if policy is None:
                    raise FederalRecallError(
                        "local continuation requires BankLocalRefinePolicy@1"
                    )
                if len(step.candidates) != 1:
                    raise FederalRecallError(
                        "local continuation must be the only serial Bank candidate"
                    )
                candidate = local_candidates[0]
                assert candidate.next_value is not None
                trace.append(
                    self._local_trace_step(
                        local_step=local_step,
                        candidate=candidate,
                        input_value=current,
                        query_result=query_result,
                        action="continue-local",
                        exit_reason=None,
                    )
                )
                accumulated_score = accumulated_score + candidate.local_log_score.to(
                    device=accumulated_score.device,
                    dtype=accumulated_score.dtype,
                ).reshape(())
                if local_step >= max_steps:
                    raise FederalRecallError(
                        "Bank-local Refine reached max_steps without a valid exit"
                    )
                current = candidate.next_value
                self.signature.input_schema.validate_tensor(
                    current,
                    name=f"{self.bank_id}.local[{local_step}]",
                )
                continue

            if local_step < min_steps:
                raise FederalRecallError(
                    "Bank-local exit is invalid before min_steps"
                )
            adjusted: list[FederalCandidate] = []
            for candidate in step.candidates:
                if candidate.next_value is not None:
                    self.signature.output_schema.validate_tensor(
                        candidate.next_value,
                        name=f"{self.bank_id}.output",
                    )
                action = "terminal" if candidate.terminal_outputs is not None else "descend"
                trace.append(
                    self._local_trace_step(
                        local_step=local_step,
                        candidate=candidate,
                        input_value=current,
                        query_result=query_result,
                        action=action,
                        exit_reason="formula-exit",
                    )
                )
                adjusted.append(
                    replace(
                        candidate,
                        local_log_score=(
                            accumulated_score
                            + candidate.local_log_score.to(
                                device=accumulated_score.device,
                                dtype=accumulated_score.dtype,
                            ).reshape(())
                        ),
                    )
                )
            return FederalBankStep(tuple(adjusted), tuple(trace))
        raise AssertionError("unreachable Bank-local Refine state")

    def _local_trace_step(
        self,
        *,
        local_step: int,
        candidate: FederalCandidate,
        input_value: Tensor,
        query_result: BankQueryResult,
        action: str,
        exit_reason: str | None,
    ) -> BankLocalRefineTraceStep:
        output_shape = (
            None
            if candidate.next_value is None
            else tuple(int(size) for size in candidate.next_value.shape)
        )
        return BankLocalRefineTraceStep(
            bank_id=self.bank_id,
            local_step=local_step,
            candidate_id=candidate.candidate_id,
            action=action,
            input_shape=tuple(int(size) for size in input_value.shape),
            query_shape=tuple(int(size) for size in query_result.value.shape),
            output_shape=output_shape,
            query_ref=self.query.signature.query_ref,
            query_state_fingerprint=self.query.signature.state_fingerprint,
            formula_ref=self.signature.local_formula_ref,
            exit_reason=exit_reason,
        )


class _AutonomousBankCollection(nn.Module, Mapping[str, AutonomousBankProgram]):
    """Register programs without leaking ModuleDict key restrictions into Bank IDs."""

    __hash__ = object.__hash__

    def __init__(self, programs: Mapping[str, AutonomousBankProgram]) -> None:
        super().__init__()
        ordered = tuple(sorted(programs.items()))
        self._bank_ids = tuple(bank_id for bank_id, _program in ordered)
        self._lookup = {bank_id: index for index, bank_id in enumerate(self._bank_ids)}
        self._programs = nn.ModuleList(program for _bank_id, program in ordered)

    def __getitem__(self, bank_id: str) -> AutonomousBankProgram:
        try:
            return self._programs[self._lookup[bank_id]]
        except KeyError as exc:
            raise KeyError(bank_id) from exc

    def __iter__(self) -> Iterator[str]:
        return iter(self._bank_ids)

    def __len__(self) -> int:
        return len(self._bank_ids)


class _BankOwnedQueryCollection(nn.Module, Mapping[str, BankOwnedQueryProgram]):
    """Register v2 Banks under stable ID-derived state paths."""

    __hash__ = object.__hash__

    def __init__(self, programs: Mapping[str, BankOwnedQueryProgram]) -> None:
        super().__init__()
        ordered = tuple(sorted(programs.items()))
        self._bank_ids = tuple(bank_id for bank_id, _program in ordered)
        self._lookup = {
            bank_id: f"bank_{hashlib.sha256(bank_id.encode('utf-8')).hexdigest()}"
            for bank_id in self._bank_ids
        }
        if len(set(self._lookup.values())) != len(self._lookup):
            raise FederalRecallError("Bank identifier digest collision")
        self._programs = nn.ModuleDict(
            {
                self._lookup[bank_id]: program
                for bank_id, program in ordered
            }
        )

    def __getitem__(self, bank_id: str) -> BankOwnedQueryProgram:
        try:
            return self._programs[self._lookup[bank_id]]
        except KeyError as exc:
            raise KeyError(bank_id) from exc

    def __iter__(self) -> Iterator[str]:
        return iter(self._bank_ids)

    def __len__(self) -> int:
        return len(self._bank_ids)


@dataclass(frozen=True)
class FederalTerminalRecord:
    """One validated terminal result retained in a runtime path."""

    bank_id: str
    path: tuple[str, ...]
    outputs: Mapping[str, Tensor]
    cumulative_log_score: Tensor
    terminal_score: Tensor
    valid: bool

    _runtime_contract_ref: ClassVar[str] = "arti/federal-terminal-record@1"

    def __post_init__(self) -> None:
        _require_name(self.bank_id, field="bank_id")
        object.__setattr__(self, "path", tuple(self.path))
        if not self.path:
            raise FederalRecallError("terminal path must not be empty")
        object.__setattr__(self, "outputs", _freeze_outputs(self.outputs))
        _require_scalar_score(
            self.cumulative_log_score,
            field="cumulative_log_score",
        )
        _require_scalar_score(self.terminal_score, field="terminal_score")
        if not isinstance(self.valid, bool):
            raise TypeError("valid must be a bool")


@dataclass(frozen=True)
class FederalTraceStep:
    """A JSON-safe receipt for one sample and one Federation depth."""

    sample_index: int
    depth: int
    expanded_count: int
    kept_count: int
    bank_ids: tuple[str, ...]
    path_ids: tuple[str, ...]
    terminal_mask: tuple[bool, ...]
    cumulative_log_scores: tuple[float, ...]
    local_refine: tuple[BankLocalRefineTraceStep, ...] = ()

    _runtime_contract_ref: ClassVar[str] = "arti/federal-trace-step@2"

    def __post_init__(self) -> None:
        object.__setattr__(self, "local_refine", tuple(self.local_refine))
        if any(not isinstance(item, BankLocalRefineTraceStep) for item in self.local_refine):
            raise TypeError("local_refine must contain BankLocalRefineTraceStep values")

    def to_dict(self) -> dict[str, object]:
        return {
            "ref": self._runtime_contract_ref,
            "sample_index": self.sample_index,
            "depth": self.depth,
            "expanded_count": self.expanded_count,
            "kept_count": self.kept_count,
            "bank_ids": list(self.bank_ids),
            "path_ids": list(self.path_ids),
            "terminal_mask": list(self.terminal_mask),
            "cumulative_log_scores": list(self.cumulative_log_scores),
            "local_refine": [item.to_dict() for item in self.local_refine],
        }


@dataclass(frozen=True)
class FederalTrace:
    """Runtime-only traversal evidence; it is never part of Bank persistence."""

    max_k: int
    max_levels: int
    steps: tuple[FederalTraceStep, ...]
    winner_paths: tuple[str, ...]
    schema_version: int = FEDERAL_TRACE_VERSION

    _component_reference: ClassVar[str] = "arti/federal-trace@2"

    def __post_init__(self) -> None:
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "winner_paths", tuple(self.winner_paths))
        if self.schema_version != FEDERAL_TRACE_VERSION:
            raise FederalRecallError("unsupported FederalTrace version")

    @property
    def maximum_kept_paths(self) -> int:
        return max((step.kept_count for step in self.steps), default=0)

    def to_dict(self) -> dict[str, object]:
        return {
            "ref": self._component_reference,
            "schema_version": self.schema_version,
            "max_k": self.max_k,
            "max_levels": self.max_levels,
            "maximum_kept_paths": self.maximum_kept_paths,
            "steps": [step.to_dict() for step in self.steps],
            "winner_paths": list(self.winner_paths),
        }


@dataclass(frozen=True)
class _FederalPath:
    bank_id: str
    value: Tensor | None
    cumulative_log_score: Tensor
    path: tuple[str, ...]
    terminal: FederalTerminalRecord | None = None


class FederalRecall(nn.Module):
    """Eager fixed-K traversal across shape-autonomous Bank programs."""

    _component_reference: ClassVar[str] = "arti/federal-recall@1"

    def __init__(
        self,
        banks: Mapping[str, AutonomousBankProgram],
        *,
        terminal_abi: TerminalOutputABI,
        root_bank_ids: tuple[str, ...],
        max_levels: int = 8,
        max_k: int = 32,
        winner_policy: str = "hard_one_winner",
    ) -> None:
        super().__init__()
        if not isinstance(banks, Mapping) or not banks:
            raise FederalRecallError("banks must be a non-empty mapping")
        if not isinstance(terminal_abi, TerminalOutputABI):
            raise TypeError("terminal_abi must be TerminalOutputABI")
        modules: dict[str, AutonomousBankProgram] = {}
        for bank_id, program in banks.items():
            _require_name(bank_id, field="bank_id")
            if not isinstance(program, AutonomousBankProgram):
                raise TypeError("banks must contain AutonomousBankProgram values")
            if program.bank_id != bank_id:
                raise FederalRecallError("Bank mapping key must match program.bank_id")
            program.signature.validate_terminal_abi(terminal_abi)
            modules[bank_id] = program
        roots = tuple(root_bank_ids)
        if not roots or len(roots) != len(set(roots)):
            raise FederalRecallError("root_bank_ids must be non-empty and unique")
        if any(root not in modules for root in roots):
            raise FederalRecallError("every root Bank must be present")
        if isinstance(max_levels, bool) or not isinstance(max_levels, int) or max_levels <= 0:
            raise FederalRecallError("max_levels must be a positive integer")
        if isinstance(max_k, bool) or not isinstance(max_k, int) or max_k <= 0:
            raise FederalRecallError("max_k must be a positive integer")
        if winner_policy != "hard_one_winner":
            raise FederalRecallError("FederalRecall@1 supports only hard_one_winner")
        score_fields = tuple(
            field.name
            for field in terminal_abi.fields
            if field.semantic_role == "terminal-score"
        )
        validity_fields = tuple(
            field.name
            for field in terminal_abi.fields
            if field.semantic_role == "terminal-validity"
        )
        if len(score_fields) != 1 or len(validity_fields) != 1:
            raise FederalRecallError(
                "TerminalOutputABI must define one terminal-score and one terminal-validity field"
            )
        self.banks = _AutonomousBankCollection(modules)
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
                name: self.banks[name].signature.to_dict()
                for name in sorted(self.banks)
            },
            "max_levels": self.max_levels,
            "max_k": self.max_k,
            "winner_policy": self.winner_policy,
        }

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
        _require_scalar_score(score, field="terminal score output")
        if not isinstance(validity, Tensor) or validity.dtype != torch.bool:
            raise FederalRecallError("terminal validity output must be boolean")
        valid = bool(validity.detach().all())
        return FederalTerminalRecord(
            bank_id=bank_id,
            path=path,
            outputs=outputs,
            cumulative_log_score=cumulative_log_score,
            terminal_score=score.reshape(()),
            valid=valid,
        )

    @staticmethod
    def _path_order(path: _FederalPath) -> tuple[float, str]:
        score = float(path.cumulative_log_score.detach().reshape(()).cpu())
        if path.terminal is not None and not path.terminal.valid:
            score = -math.inf
        return (-score, "/".join(path.path))

    def _run_sample(
        self,
        value: Tensor,
        *,
        sample_index: int,
        root_bank_id: str,
        max_levels: int,
        max_k: int,
    ) -> tuple[FederalTerminalRecord, tuple[FederalTraceStep, ...]]:
        zero = value.new_zeros(())
        paths = (
            _FederalPath(
                bank_id=root_bank_id,
                value=value,
                cumulative_log_score=zero,
                path=(root_bank_id,),
            ),
        )
        receipts: list[FederalTraceStep] = []
        for depth in range(max_levels):
            expanded: list[_FederalPath] = []
            local_receipts: list[BankLocalRefineTraceStep] = []
            for current in paths:
                if current.terminal is not None:
                    expanded.append(current)
                    continue
                program = self.banks[current.bank_id]
                assert current.value is not None
                program.signature.input_schema.validate_tensor(
                    current.value,
                    name=f"{current.bank_id}.input",
                )
                step = program(current.value, max_candidates=max_k)
                if not isinstance(step, FederalBankStep):
                    raise TypeError("autonomous Bank must return FederalBankStep")
                if len(step.candidates) > max_k:
                    raise FederalRecallError("local Bank returned more than the global K limit")
                local_receipts.extend(step.local_trace)
                for candidate in step.candidates:
                    local_score = candidate.local_log_score.to(
                        device=current.cumulative_log_score.device,
                        dtype=current.cumulative_log_score.dtype,
                    ).reshape(())
                    cumulative = current.cumulative_log_score + local_score
                    next_path = (*current.path, candidate.candidate_id)
                    if candidate.terminal_outputs is not None:
                        terminal = self._terminal_record(
                            bank_id=current.bank_id,
                            path=next_path,
                            outputs=candidate.terminal_outputs,
                            cumulative_log_score=cumulative,
                        )
                        expanded.append(
                            _FederalPath(
                                bank_id=current.bank_id,
                                value=None,
                                cumulative_log_score=cumulative,
                                path=next_path,
                                terminal=terminal,
                            )
                        )
                        continue
                    assert candidate.next_bank_id is not None
                    assert candidate.next_value is not None
                    if candidate.next_bank_id not in self.banks:
                        raise FederalRecallError("candidate references an unknown child Bank")
                    child = self.banks[candidate.next_bank_id]
                    child.signature.input_schema.validate_tensor(
                        candidate.next_value,
                        name=f"{candidate.next_bank_id}.input",
                    )
                    expanded.append(
                        _FederalPath(
                            bank_id=candidate.next_bank_id,
                            value=candidate.next_value,
                            cumulative_log_score=cumulative,
                            path=(*next_path, candidate.next_bank_id),
                        )
                    )
            if not expanded:
                raise FederalRecallError("all federated paths ended without a terminal output")
            expanded.sort(key=self._path_order)
            paths = tuple(expanded[:max_k])
            receipts.append(
                FederalTraceStep(
                    sample_index=sample_index,
                    depth=depth,
                    expanded_count=len(expanded),
                    kept_count=len(paths),
                    bank_ids=tuple(path.bank_id for path in paths),
                    path_ids=tuple("/".join(path.path) for path in paths),
                    terminal_mask=tuple(path.terminal is not None for path in paths),
                    cumulative_log_scores=tuple(
                        float(path.cumulative_log_score.detach().reshape(()).cpu())
                        for path in paths
                    ),
                    local_refine=tuple(local_receipts),
                )
            )
            if all(path.terminal is not None for path in paths):
                break
        terminals = tuple(
            path.terminal
            for path in paths
            if path.terminal is not None and path.terminal.valid
        )
        if not terminals:
            raise FederalRecallError("no valid terminal output was reached within max_levels")
        winner = min(
            terminals,
            key=lambda record: (
                -float(record.terminal_score.detach().cpu()),
                "/".join(record.path),
            ),
        )
        return winner, tuple(receipts)

    def forward(
        self,
        value: Tensor,
        *,
        root_bank_id: str | None = None,
        max_levels: int | None = None,
        max_k: int | None = None,
        return_trace: bool = False,
    ) -> Mapping[str, Tensor] | tuple[Mapping[str, Tensor], FederalTrace]:
        if not isinstance(value, Tensor) or not value.is_floating_point() or value.ndim < 1:
            raise TypeError("value must be a batched floating-point Tensor")
        if value.shape[0] == 0:
            raise FederalRecallError("FederalRecall requires a non-empty batch")
        root = root_bank_id
        if root is None:
            if len(self.root_bank_ids) != 1:
                raise FederalRecallError("root_bank_id is required when multiple roots exist")
            root = self.root_bank_ids[0]
        if root not in self.root_bank_ids:
            raise FederalRecallError("root_bank_id is not declared by this Federation")
        levels = self.max_levels if max_levels is None else max_levels
        width = self.max_k if max_k is None else max_k
        if isinstance(levels, bool) or not isinstance(levels, int) or not 0 < levels <= self.max_levels:
            raise FederalRecallError("max_levels must be within the configured bound")
        if isinstance(width, bool) or not isinstance(width, int) or not 0 < width <= self.max_k:
            raise FederalRecallError("max_k must be within the configured bound")
        self.banks[root].signature.input_schema.validate_tensor(value, name=f"{root}.input")
        winners: list[FederalTerminalRecord] = []
        trace_steps: list[FederalTraceStep] = []
        for sample_index in range(value.shape[0]):
            winner, steps = self._run_sample(
                value[sample_index : sample_index + 1],
                sample_index=sample_index,
                root_bank_id=root,
                max_levels=levels,
                max_k=width,
            )
            winners.append(winner)
            trace_steps.extend(steps)
        outputs: dict[str, Tensor] = {}
        for field in self.terminal_abi.fields:
            try:
                outputs[field.name] = torch.cat(
                    [winner.outputs[field.name] for winner in winners],
                    dim=0,
                )
            except RuntimeError as exc:
                raise FederalRecallError(
                    f"terminal field {field.name!r} cannot be explicitly batched"
                ) from exc
        self.terminal_abi.validate_outputs(outputs)
        if not return_trace:
            return MappingProxyType(outputs)
        trace = FederalTrace(
            max_k=width,
            max_levels=levels,
            steps=tuple(trace_steps),
            winner_paths=tuple("/".join(winner.path) for winner in winners),
        )
        return MappingProxyType(outputs), trace


class FederalRecallV2(FederalRecall):
    """K=1 Federation whose autonomous Banks own sealed local Queries."""

    _component_reference: ClassVar[str] = "arti/federal-recall@2"

    def __init__(
        self,
        banks: Mapping[str, BankOwnedQueryProgram],
        *,
        terminal_abi: TerminalOutputABI,
        root_bank_ids: tuple[str, ...],
        max_levels: int = 8,
        max_k: int = 1,
        winner_policy: str = "hard_one_winner",
    ) -> None:
        nn.Module.__init__(self)
        if not isinstance(banks, Mapping) or not banks:
            raise FederalRecallError("banks must be a non-empty mapping")
        if not isinstance(terminal_abi, TerminalOutputABI):
            raise TypeError("terminal_abi must be TerminalOutputABI")
        modules: dict[str, BankOwnedQueryProgram] = {}
        for bank_id, program in banks.items():
            _require_name(bank_id, field="bank_id")
            if not isinstance(program, BankOwnedQueryProgram):
                raise TypeError("FederalRecall@2 requires BankOwnedQueryProgram values")
            if program.bank_id != bank_id:
                raise FederalRecallError("Bank mapping key must match program.bank_id")
            program.signature.validate_terminal_abi(terminal_abi)
            modules[bank_id] = program
        roots = tuple(root_bank_ids)
        if not roots or len(roots) != len(set(roots)):
            raise FederalRecallError("root_bank_ids must be non-empty and unique")
        if any(root not in modules for root in roots):
            raise FederalRecallError("every root Bank must be present")
        if (
            isinstance(max_levels, bool)
            or not isinstance(max_levels, int)
            or max_levels <= 0
        ):
            raise FederalRecallError("max_levels must be a positive integer")
        if type(max_k) is not int or max_k != 1:
            raise FederalRecallError(
                "FederalRecall@2 initially supports only serial K=1 execution"
            )
        if winner_policy != "hard_one_winner":
            raise FederalRecallError("FederalRecall@2 supports only hard_one_winner")
        score_fields = tuple(
            field.name
            for field in terminal_abi.fields
            if field.semantic_role == "terminal-score"
        )
        validity_fields = tuple(
            field.name
            for field in terminal_abi.fields
            if field.semantic_role == "terminal-validity"
        )
        if len(score_fields) != 1 or len(validity_fields) != 1:
            raise FederalRecallError(
                "TerminalOutputABI must define one terminal-score and one terminal-validity field"
            )
        self.banks = _BankOwnedQueryCollection(modules)
        self.terminal_abi = terminal_abi
        self.root_bank_ids = roots
        self.max_levels = max_levels
        self.max_k = max_k
        self.winner_policy = winner_policy
        self._score_field = score_fields[0]
        self._validity_field = validity_fields[0]


__all__ = [
    "FEDERAL_RECALL_VERSION",
    "FEDERAL_RECALL_V2_VERSION",
    "FEDERAL_TRACE_VERSION",
    "AutonomousBankProgram",
    "BankLocalRefinePolicy",
    "BankLocalRefineTraceStep",
    "BankOwnedQueryProgram",
    "FederalBankStep",
    "FederalCandidate",
    "FederalRecall",
    "FederalRecallV2",
    "FederalRecallError",
    "FederalTerminalRecord",
    "FederalTrace",
    "FederalTraceStep",
]
