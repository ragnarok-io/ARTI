"""Bank-owned operands and explicit hard routing for FormulaFabric@2."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import ClassVar, Literal, Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .component_registry import ComponentRef, canonical_contract_reference
from .formula_v2 import (
    DEFAULT_FORMULA_LIMITS,
    BankBinding,
    FormulaBankOperand,
    FormulaBindingError,
    FormulaLimits,
    FormulaProgram,
    InputBinding,
    TensorType,
    add,
    contract,
    scale,
)


_PARTITION_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_-]*$")


def _canonical_source_ref(value: str) -> str:
    try:
        return ComponentRef.parse(canonical_contract_reference(value)).reference
    except (TypeError, ValueError) as error:
        raise ValueError("source_ref must resolve to a full component contract reference") from error


def _require_finite(value: Tensor, message: str, *, code: str) -> None:
    condition = torch.isfinite(value).all()
    if torch.compiler.is_compiling() or value.device.type != "cpu":
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise FormulaBindingError(code, message)


@dataclass(frozen=True)
class FormulaRouteSelection:
    """One explicit candidate route and its training diagnostics."""

    route: Tensor
    hard_indices: Tensor
    logits: Tensor
    entropy: Tensor
    estimator: Literal["hard", "straight-through"]


def hard_formula_route(
    logits: Tensor,
    *,
    estimator: Literal["hard", "straight-through"] = "straight-through",
    temperature: float = 1.0,
    member_ids: Sequence[str] | None = None,
    member_priority: Tensor | None = None,
) -> FormulaRouteSelection:
    """Choose one candidate with hard forward values and an optional soft surrogate."""

    if not isinstance(logits, Tensor) or not logits.is_floating_point():
        raise TypeError("Formula route logits must be a floating Tensor")
    if logits.ndim != 2 or logits.shape[0] <= 0 or logits.shape[1] <= 0:
        raise ValueError("Formula route logits must have shape [B, K]")
    if estimator not in {"hard", "straight-through"}:
        raise ValueError("estimator must be 'hard' or 'straight-through'")
    if not isinstance(temperature, (int, float)) or isinstance(temperature, bool):
        raise TypeError("temperature must be a finite positive number")
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be a finite positive number")
    _require_finite(
        logits,
        "Formula route logits must be finite",
        code="FF2_NONFINITE",
    )
    if member_ids is not None:
        normalized_ids = tuple(member_ids)
        if (
            len(normalized_ids) != logits.shape[-1]
            or len(set(normalized_ids)) != len(normalized_ids)
            or any(not isinstance(item, str) or not item for item in normalized_ids)
        ):
            raise ValueError("member_ids must uniquely identify every route candidate")
        if member_priority is not None and (
            not isinstance(member_priority, Tensor)
            or member_priority.shape != (logits.shape[-1],)
            or member_priority.dtype != torch.int64
            or member_priority.device != logits.device
        ):
            raise ValueError(
                "member_priority must be int64 [K] on the logits device"
            )
    elif member_priority is not None:
        raise ValueError("member_priority requires member_ids")

    soft = torch.softmax(logits / temperature, dim=-1)
    if member_ids is None:
        hard_indices = logits.argmax(dim=-1)
    else:
        if member_priority is None:
            lexical_rank = {
                item: rank for rank, item in enumerate(sorted(normalized_ids))
            }
            priority = torch.tensor(
                [lexical_rank[item] for item in normalized_ids],
                device=logits.device,
                dtype=torch.int64,
            )
        else:
            priority = member_priority
        maxima = logits == logits.max(dim=-1, keepdim=True).values
        sentinel = torch.full_like(priority, len(normalized_ids))
        hard_indices = torch.where(maxima, priority, sentinel).argmin(dim=-1)
    hard = F.one_hot(hard_indices, num_classes=logits.shape[-1]).to(dtype=logits.dtype)
    route = hard if estimator == "hard" else hard + soft - soft.detach()
    entropy = -(soft * soft.clamp_min(torch.finfo(soft.dtype).tiny).log()).sum(dim=-1)
    return FormulaRouteSelection(route, hard_indices, logits, entropy, estimator)


class FormulaOperandBank(nn.Module):
    """A joint candidate Bank whose heterogeneous values remain explicit operands.

    The query is a runtime input, not module state. Gradients may flow through
    the query to the current hidden state, but this Bank never owns, saves, or
    adds a Query producer to its optimizer-visible parameters.
    """

    _component_reference: ClassVar[str] = "arti/formula-operand-bank@1"

    def __init__(
        self,
        *,
        keys: Tensor,
        operands: Mapping[str, Tensor],
        source_ref: str = "arti/formula-operand-bank@1",
        bundle_id: str = "formula",
        member_ids: Sequence[str] | None = None,
        asset_fingerprint: str | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(keys, Tensor) or not keys.is_floating_point():
            raise TypeError("keys must be a floating Tensor")
        if keys.ndim != 2 or keys.shape[0] <= 0 or keys.shape[1] <= 0:
            raise ValueError("keys must have shape [K, Q]")
        if not isinstance(operands, Mapping) or not operands:
            raise ValueError("operands must be a non-empty mapping")
        resolved_source_ref = _canonical_source_ref(source_ref)
        if not isinstance(bundle_id, str) or not _PARTITION_RE.fullmatch(bundle_id):
            raise ValueError("bundle_id is invalid")
        if not bool(torch.isfinite(keys).all()):
            raise ValueError("keys must contain finite values")
        if asset_fingerprint is not None and (
            not isinstance(asset_fingerprint, str)
            or len(asset_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in asset_fingerprint)
        ):
            raise ValueError("asset_fingerprint must be a SHA-256 hex digest")

        candidate_count = int(keys.shape[0])
        normalized_operands: dict[str, Tensor] = {}
        for name, value in operands.items():
            if not isinstance(name, str) or not _PARTITION_RE.fullmatch(name):
                raise ValueError("operand names must be valid partition identifiers")
            if not isinstance(value, Tensor) or not value.is_floating_point():
                raise TypeError(f"operand {name!r} must be a floating Tensor")
            if value.ndim < 1 or value.shape[0] != candidate_count:
                raise ValueError(f"operand {name!r} must use candidate axis K={candidate_count}")
            if value.device != keys.device or value.dtype != keys.dtype:
                raise ValueError("keys and operands must share device and dtype")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"operand {name!r} must contain finite values")
            normalized_operands[name] = value.detach().clone()

        if member_ids is None:
            normalized_members = tuple(
                f"member-{index:03d}" for index in range(candidate_count)
            )
        else:
            normalized_members = tuple(member_ids)
        if (
            len(normalized_members) != candidate_count
            or len(set(normalized_members)) != candidate_count
            or any(
                not isinstance(item, str) or not _PARTITION_RE.fullmatch(item)
                for item in normalized_members
            )
        ):
            raise ValueError("member_ids must uniquely identify every candidate")

        self.keys = nn.Parameter(keys.detach().clone())
        self.operands = nn.ParameterDict(
            {name: nn.Parameter(value) for name, value in sorted(normalized_operands.items())}
        )
        self.source_ref = resolved_source_ref
        self.bundle_id = bundle_id
        self.member_ids = normalized_members
        lexical_rank = {
            item: rank for rank, item in enumerate(sorted(normalized_members))
        }
        self.register_buffer(
            "_member_priority",
            torch.tensor(
                [lexical_rank[item] for item in normalized_members],
                dtype=torch.int64,
                device=keys.device,
            ),
            persistent=False,
        )
        self.asset_fingerprint = asset_fingerprint
        self.candidate_count = candidate_count
        self.key_dim = int(keys.shape[1])

    def route(
        self,
        query: Tensor,
        *,
        estimator: Literal["hard", "straight-through"] = "straight-through",
        temperature: float = 1.0,
    ) -> FormulaRouteSelection:
        if not isinstance(query, Tensor) or not query.is_floating_point():
            raise TypeError("query must be a floating Tensor")
        if query.ndim != 2 or query.shape[0] <= 0 or query.shape[1] != self.key_dim:
            raise ValueError(f"query must have shape [B, {self.key_dim}]")
        if query.device != self.keys.device or query.dtype != self.keys.dtype:
            raise ValueError("query and FormulaOperandBank keys must share device and dtype")
        _require_finite(
            query,
            "query must contain finite values",
            code="FF2_NONFINITE_QUERY",
        )
        normalized_query = F.normalize(query, dim=-1)
        normalized_keys = F.normalize(self.keys, dim=-1)
        logits = normalized_query @ normalized_keys.transpose(0, 1)
        return hard_formula_route(
            logits,
            estimator=estimator,
            temperature=temperature,
            member_ids=self.member_ids,
            member_priority=self._member_priority,
        )

    def bind(self, program: FormulaProgram) -> dict[str, FormulaBankOperand]:
        """Bind this Bank's role-specific tensors to matching program declarations."""

        if not isinstance(program, FormulaProgram):
            raise TypeError("program must be FormulaProgram")
        result: dict[str, FormulaBankOperand] = {}
        for binding in program.bindings:
            if not isinstance(binding, BankBinding):
                continue
            if (
                binding.source_ref != self.source_ref
                or binding.asset_fingerprint != self.asset_fingerprint
                or binding.bundle_id != self.bundle_id
                or binding.member_ids != self.member_ids
            ):
                raise FormulaBindingError(
                    "FF2_BANK_IDENTITY_MISMATCH",
                    f"program Bank binding {binding.name!r} does not match this operand Bank",
                )
            try:
                value = self.operands[binding.partition_id]
            except KeyError as exc:
                raise FormulaBindingError(
                    "FF2_BANK_PARTITION_MISSING",
                    f"operand Bank has no partition {binding.partition_id!r}",
                ) from exc
            result[binding.name] = binding.bind(value)
        return result


def build_routed_lora_program(
    *,
    input_dim: int,
    output_dim: int,
    rank: int,
    candidate_count: int,
    source_ref: str = "arti/formula-operand-bank@1",
    bundle_id: str = "lora",
    member_ids: Sequence[str] | None = None,
    asset_fingerprint: str | None = None,
    dtype: str = "floating",
    domain: str = "anonymous",
    contract_accumulation_dtype: str = "float32",
    pointwise_accumulation_dtype: str = "activation",
    limits: FormulaLimits = DEFAULT_FORMULA_LIMITS,
) -> FormulaProgram:
    """Build a joint-bundle hard-route program and expand its LoRA arithmetic to atoms."""

    for value, name in (
        (input_dim, "input_dim"),
        (output_dim, "output_dim"),
        (rank, "rank"),
        (candidate_count, "candidate_count"),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if member_ids is None:
        resolved_member_ids = tuple(
            f"member-{index:03d}" for index in range(candidate_count)
        )
    else:
        resolved_member_ids = tuple(member_ids)
        if len(resolved_member_ids) != candidate_count:
            raise ValueError("member_ids must match candidate_count")

    tensor = lambda axes, sizes: TensorType.axes(  # noqa: E731
        axes, sizes=sizes, dtype=dtype, domain=domain
    )
    x = InputBinding("x", tensor(("B", "S", "Din"), ("B", "S", input_dim)))
    base = InputBinding(
        "base", tensor(("B", "S", "Dout"), ("B", "S", output_dim))
    )
    route = InputBinding(
        "formula.route", tensor(("B", "K"), ("B", candidate_count))
    )
    bank_kwargs = {
        "source_ref": source_ref,
        "asset_fingerprint": asset_fingerprint,
        "bundle_id": bundle_id,
        "member_ids": resolved_member_ids,
    }
    a = BankBinding(
        "lora.A",
        partition_id="A",
        value_type=tensor(("K", "R", "Din"), (candidate_count, rank, input_dim)),
        **bank_kwargs,
    )
    b = BankBinding(
        "lora.B",
        partition_id="B",
        value_type=tensor(("K", "Dout", "R"), (candidate_count, output_dim, rank)),
        **bank_kwargs,
    )
    gain = BankBinding(
        "lora.gain",
        partition_id="gain",
        value_type=tensor(("K",), (candidate_count,)),
        **bank_kwargs,
    )

    selected_a = contract(
        route,
        a,
        reduce_axes=(("K", "K"),),
        output_axes=("B", "R", "Din"),
        accumulation_dtype=contract_accumulation_dtype,
    )
    selected_b = contract(
        route,
        b,
        reduce_axes=(("K", "K"),),
        output_axes=("B", "Dout", "R"),
        accumulation_dtype=contract_accumulation_dtype,
    )
    selected_gain = contract(
        route,
        gain,
        reduce_axes=(("K", "K"),),
        output_axes=("B",),
        accumulation_dtype=contract_accumulation_dtype,
    )
    hidden = contract(
        x,
        selected_a,
        reduce_axes=(("Din", "Din"),),
        output_axes=("B", "S", "R"),
        accumulation_dtype=contract_accumulation_dtype,
    )
    delta = contract(
        hidden,
        selected_b,
        reduce_axes=(("R", "R"),),
        output_axes=("B", "S", "Dout"),
        accumulation_dtype=contract_accumulation_dtype,
    )
    return FormulaProgram.build(
        outputs=(
            add(
                base,
                scale(
                    delta,
                    selected_gain,
                    accumulation_dtype=pointwise_accumulation_dtype,
                ),
                accumulation_dtype=pointwise_accumulation_dtype,
            ),
        ),
        limits=limits,
    )


__all__ = [
    "FormulaOperandBank",
    "FormulaRouteSelection",
    "build_routed_lora_program",
    "hard_formula_route",
]
