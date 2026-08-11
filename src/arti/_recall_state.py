"""Experimental value-state transition used to study forward-only Recall TTT."""

from __future__ import annotations

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.func import functional_call, stack_module_state, vmap

from .layers import ARTILayer


@dataclass(frozen=True)
class AffineRecallTransition:
    """One composable values-only Recall state transition.

    ``retention`` and ``write`` define ``V' = retention * V + write``.
    Applying ``first`` and then ``second`` is represented exactly by
    ``first.then(second)``; no learned consistency approximation is involved.
    """

    retention: Tensor
    write: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.retention, Tensor) or not isinstance(self.write, Tensor):
            raise TypeError("retention and write must be Tensors")
        if not self.retention.is_floating_point() or not self.write.is_floating_point():
            raise TypeError("retention and write must be floating-point Tensors")
        if self.retention.device != self.write.device:
            raise ValueError("retention and write must share a device")
        try:
            torch.broadcast_shapes(self.retention.shape, self.write.shape)
        except RuntimeError as error:
            raise ValueError("retention and write must be broadcast-compatible") from error

    def apply(self, value: Tensor) -> Tensor:
        """Apply this transition to a values-only Bank state."""

        if not isinstance(value, Tensor) or not value.is_floating_point():
            raise TypeError("value must be a floating-point Tensor")
        if value.device != self.write.device:
            raise ValueError("value and transition must share a device")
        return self.retention.to(value) * value + self.write.to(value)

    def then(self, second: "AffineRecallTransition") -> "AffineRecallTransition":
        """Compose this transition followed by ``second``."""

        if self.write.device != second.write.device:
            raise ValueError("composed transitions must share a device")
        retention = second.retention * self.retention
        write = second.retention * self.write + second.write
        return AffineRecallTransition(retention, write)


@dataclass(frozen=True)
class MatrixAffineRecallTransition:
    """A composable affine transition that can mix Bank slots.

    ``matrix`` and ``write`` define ``V' = matrix @ V + write``. Unlike
    :class:`AffineRecallTransition`, this transition can replace one keyed
    association without applying the same elementwise retention to every slot.
    """

    matrix: Tensor
    write: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.matrix, Tensor) or not isinstance(self.write, Tensor):
            raise TypeError("matrix and write must be Tensors")
        if not self.matrix.is_floating_point() or not self.write.is_floating_point():
            raise TypeError("matrix and write must be floating-point Tensors")
        if self.matrix.device != self.write.device:
            raise ValueError("matrix and write must share a device")
        if self.matrix.ndim < 2 or self.matrix.shape[-1] != self.matrix.shape[-2]:
            raise ValueError("matrix must end in square [S, S] dimensions")
        if self.write.ndim < 2 or self.write.shape[-2] != self.matrix.shape[-1]:
            raise ValueError("write must end in [S, H] with the same slot count")
        if self.matrix.shape[:-2] != self.write.shape[:-2]:
            raise ValueError("matrix and write must share leading dimensions")

    def apply(self, value: Tensor) -> Tensor:
        """Apply this transition to a values-only Bank state."""

        if not isinstance(value, Tensor) or not value.is_floating_point():
            raise TypeError("value must be a floating-point Tensor")
        if value.device != self.write.device:
            raise ValueError("value and transition must share a device")
        if (
            value.shape[:-2] != self.write.shape[:-2]
            or value.shape[-2:] != self.write.shape[-2:]
        ):
            raise ValueError("value must have the same shape as write")
        return torch.matmul(self.matrix.to(value), value) + self.write.to(value)

    def then(
        self,
        second: "MatrixAffineRecallTransition",
    ) -> "MatrixAffineRecallTransition":
        """Compose this transition followed by ``second``."""

        if self.write.device != second.write.device:
            raise ValueError("composed transitions must share a device")
        matrix = torch.matmul(second.matrix, self.matrix)
        write = torch.matmul(second.matrix, self.write) + second.write
        return MatrixAffineRecallTransition(matrix, write)


class FixedFeatureValueBank(nn.Module):
    """Read a values-only Bank through one frozen feature basis.

    The basis defines ``phi(x)`` and is stored as a buffer. Only ``values`` is
    dynamic, so adding a state increment has an exactly additive host effect.
    """

    state_semantics = "values_only"
    feature_semantics = "fixed"

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        rank: int,
        *,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if input_dim <= 0 or output_dim <= 0 or rank <= 0:
            raise ValueError("input_dim, output_dim, and rank must be positive")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.rank = int(rank)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        basis = torch.empty(rank, input_dim)
        basis.normal_(std=input_dim**-0.5, generator=generator)
        self.register_buffer("feature_basis", basis, persistent=True)

    def initial_values(
        self,
        batch_size: int,
        *,
        reference: Tensor | None = None,
    ) -> Tensor:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        values = torch.zeros(batch_size, self.output_dim, self.rank)
        return values if reference is None else values.to(reference)

    def forward(
        self,
        x: Tensor,
        values: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> Tensor:
        if x.ndim != 3 or x.shape[-1] != self.input_dim:
            raise ValueError(f"x must have shape [B, T, {self.input_dim}]")
        expected = (x.shape[0], self.output_dim, self.rank)
        if tuple(values.shape) != expected:
            raise ValueError(f"values must have shape {expected}")
        if values.device != x.device or not values.is_floating_point():
            raise ValueError("values must be floating-point on the input device")
        features = torch.einsum("bti,ri->btr", x, self.feature_basis.to(x))
        result = torch.einsum("btr,bor->bto", features.to(values), values)
        if mask is not None:
            if (
                mask.dtype != torch.bool
                or mask.device != x.device
                or tuple(mask.shape) != tuple(x.shape[:2])
            ):
                raise ValueError("mask must be boolean [B, T] on the input device")
            result = result * mask.unsqueeze(-1).to(result.dtype)
        return result


class RecallStateController(nn.Module):
    """Read a values-only Bank as a decoupled cross-attention condition."""

    state_semantics = "values_only"
    control_semantics = "decoupled_cross_attention"

    def __init__(
        self,
        hidden_dim: int,
        *,
        control_dim: int | None = None,
        heads: int = 8,
    ) -> None:
        super().__init__()
        resolved_dim = hidden_dim if control_dim is None else int(control_dim)
        if hidden_dim <= 0 or resolved_dim <= 0 or heads <= 0:
            raise ValueError("hidden_dim, control_dim, and heads must be positive")
        if resolved_dim % heads:
            raise ValueError("control_dim must be divisible by heads")
        self.hidden_dim = int(hidden_dim)
        self.control_dim = resolved_dim
        self.heads = int(heads)
        self.head_dim = resolved_dim // heads
        self.query = nn.Linear(hidden_dim, resolved_dim, bias=False)
        self.key = nn.Linear(hidden_dim, resolved_dim, bias=False)
        self.value = nn.Linear(hidden_dim, resolved_dim, bias=False)
        self.output = nn.Linear(resolved_dim, hidden_dim, bias=False)
        self.control_scale = nn.Parameter(torch.tensor(0.25))

    def forward(
        self,
        hidden: Tensor,
        bank_values: Tensor,
        *,
        mask: Tensor | None = None,
        bank_mask: Tensor | None = None,
    ) -> Tensor:
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_dim:
            raise ValueError(f"hidden must have shape [B, T, {self.hidden_dim}]")
        if (
            bank_values.ndim != 3
            or bank_values.shape[0] != hidden.shape[0]
            or bank_values.shape[-1] != self.hidden_dim
        ):
            raise ValueError("bank_values must have shape [B, S, H]")
        if bank_values.device != hidden.device:
            raise ValueError("hidden and bank_values must share a device")
        if mask is not None and (
            mask.dtype != torch.bool
            or mask.device != hidden.device
            or tuple(mask.shape) != tuple(hidden.shape[:2])
        ):
            raise ValueError("mask must be boolean [B, T] on the hidden device")
        if bank_mask is not None and (
            bank_mask.dtype != torch.bool
            or bank_mask.device != hidden.device
            or tuple(bank_mask.shape) != tuple(bank_values.shape[:2])
        ):
            raise ValueError("bank_mask must be boolean [B, S] on the hidden device")
        batch_size, token_count, _ = hidden.shape
        slot_count = bank_values.shape[1]
        query = self.query(hidden).reshape(
            batch_size, token_count, self.heads, self.head_dim
        ).transpose(1, 2)
        key = self.key(bank_values).reshape(
            batch_size, slot_count, self.heads, self.head_dim
        ).transpose(1, 2)
        value = self.value(bank_values).reshape(
            batch_size, slot_count, self.heads, self.head_dim
        ).transpose(1, 2)
        attention_mask = None
        if bank_mask is not None:
            attention_mask = bank_mask.reshape(batch_size, 1, 1, slot_count)
        controlled = torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
        )
        controlled = controlled.transpose(1, 2).reshape(
            batch_size, token_count, self.control_dim
        )
        controlled = self.output(controlled)
        controlled = controlled * torch.tanh(self.control_scale).to(controlled)
        if mask is not None:
            controlled = controlled * mask.unsqueeze(-1).to(controlled.dtype)
        return controlled


@dataclass(frozen=True)
class RecallExpertRead:
    """One sparse read from independently normalized Recall experts."""

    delta: Tensor
    route_weights: Tensor
    selected_experts: Tensor
    route_logits: Tensor


class RecallStateExpertAssembly(nn.Module):
    """Route across independent Recall controller Q/K/V branches.

    Each expert performs its own cross-attention normalization before routing.
    Adding an expert therefore adds a candidate branch without changing the
    numerical read performed by an existing branch.
    """

    def __init__(
        self,
        experts: Sequence[RecallStateController],
        *,
        topk: int = 1,
    ) -> None:
        super().__init__()
        if not experts:
            raise ValueError("Recall expert assembly requires at least one expert")
        hidden_dims = {expert.hidden_dim for expert in experts}
        control_dims = {expert.control_dim for expert in experts}
        heads = {expert.heads for expert in experts}
        if len(hidden_dims) != 1 or len(control_dims) != 1 or len(heads) != 1:
            raise ValueError("Recall experts must share hidden/control dimensions and heads")
        if not 1 <= topk <= len(experts):
            raise ValueError("topk must be in [1, expert_count]")
        self.experts = nn.ModuleList(experts)
        self.hidden_dim = hidden_dims.pop()
        self.control_dim = control_dims.pop()
        self.heads = heads.pop()
        self.topk = int(topk)
        self.route_weight = nn.Parameter(torch.eye(len(experts)))
        self.route_bias = nn.Parameter(torch.zeros(len(experts)))

    @property
    def expert_count(self) -> int:
        return len(self.experts)

    def _validate(
        self,
        hidden: Tensor,
        bank_values: Tensor,
        mask: Tensor | None,
        bank_mask: Tensor | None,
    ) -> None:
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_dim:
            raise ValueError(f"hidden must have shape [B, T, {self.hidden_dim}]")
        expected_prefix = (hidden.shape[0], self.expert_count)
        if (
            bank_values.ndim != 4
            or tuple(bank_values.shape[:2]) != expected_prefix
            or bank_values.shape[-1] != self.hidden_dim
        ):
            raise ValueError("bank_values must have shape [B, E, S, H]")
        if bank_values.device != hidden.device or not bank_values.is_floating_point():
            raise ValueError("bank_values must be floating-point on the hidden device")
        if mask is not None and (
            mask.dtype != torch.bool
            or mask.device != hidden.device
            or tuple(mask.shape) != tuple(hidden.shape[:2])
        ):
            raise ValueError("mask must be boolean [B, T] on the hidden device")
        if bank_mask is not None and (
            bank_mask.dtype != torch.bool
            or bank_mask.device != hidden.device
            or tuple(bank_mask.shape) != tuple(bank_values.shape[:3])
        ):
            raise ValueError("bank_mask must be boolean [B, E, S] on the hidden device")

    def route_logits(
        self,
        hidden: Tensor,
        bank_values: Tensor,
        *,
        bank_mask: Tensor | None = None,
    ) -> Tensor:
        """Score expert compatibility from each branch's own Q/K projections."""

        self._validate(hidden, bank_values, None, bank_mask)
        scores = []
        for index, expert in enumerate(self.experts):
            values = bank_values[:, index]
            batch_size, token_count, _ = hidden.shape
            slot_count = values.shape[1]
            query = expert.query(hidden).reshape(
                batch_size, token_count, expert.heads, expert.head_dim
            )
            key = expert.key(values).reshape(
                batch_size, slot_count, expert.heads, expert.head_dim
            )
            compatibility = torch.einsum("bthd,bshd->bths", query, key)
            compatibility = compatibility * (expert.head_dim**-0.5)
            if bank_mask is not None:
                valid = bank_mask[:, index].reshape(batch_size, 1, 1, slot_count)
                compatibility = compatibility.masked_fill(~valid, -torch.inf)
            evidence = torch.logsumexp(compatibility, dim=-1).mean(dim=-1)
            if bank_mask is None:
                evidence = evidence - math.log(slot_count)
            else:
                valid_count = bank_mask[:, index].sum(dim=-1).clamp_min(1)
                evidence = evidence - valid_count.log().unsqueeze(-1)
            scores.append(evidence)
        logits = torch.stack(scores, dim=-1)
        return F.linear(
            logits,
            self.route_weight.to(logits),
            self.route_bias.to(logits),
        )

    def forward(
        self,
        hidden: Tensor,
        bank_values: Tensor,
        *,
        mask: Tensor | None = None,
        bank_mask: Tensor | None = None,
        route_weights: Tensor | None = None,
        return_info: bool = False,
    ) -> Tensor | RecallExpertRead:
        self._validate(hidden, bank_values, mask, bank_mask)
        logits: Tensor
        if route_weights is None:
            logits = self.route_logits(hidden, bank_values, bank_mask=bank_mask)
            selected_logits, selected = torch.topk(logits, self.topk, dim=-1)
            selected_weights = torch.softmax(selected_logits, dim=-1)
            weights = torch.zeros_like(logits).scatter(-1, selected, selected_weights)
        else:
            expected = (*hidden.shape[:2], self.expert_count)
            if (
                route_weights.shape != expected
                or route_weights.device != hidden.device
                or not route_weights.is_floating_point()
            ):
                raise ValueError(f"route_weights must be floating-point with shape {expected}")
            if bool(torch.any(route_weights < 0)):
                raise ValueError("route_weights must be non-negative")
            total = route_weights.sum(dim=-1, keepdim=True)
            if bool(torch.any(total <= 0)):
                raise ValueError("route_weights must select at least one expert")
            weights = route_weights / total
            selected = torch.topk(weights, min(self.topk, self.expert_count), dim=-1).indices
            logits = torch.log(weights.clamp_min(torch.finfo(weights.dtype).tiny))

        expert_deltas = []
        for index, expert in enumerate(self.experts):
            expert_deltas.append(
                expert(
                    hidden,
                    bank_values[:, index],
                    mask=mask,
                    bank_mask=None if bank_mask is None else bank_mask[:, index],
                )
            )
        stacked = torch.stack(expert_deltas, dim=-2)
        delta = torch.sum(stacked * weights.unsqueeze(-1).to(stacked), dim=-2)
        if not return_info:
            return delta
        return RecallExpertRead(delta, weights, selected, logits)


def reduce_affine_recall_transitions(
    retention: Tensor,
    write: Tensor,
    *,
    mask: Tensor | None = None,
) -> AffineRecallTransition:
    """Reduce ordered token transitions into one exactly equivalent transition."""

    if retention.ndim < 3 or write.ndim < 3:
        raise ValueError("retention and write require batch and token dimensions")
    if retention.shape[:2] != write.shape[:2]:
        raise ValueError("retention and write must share [B, T]")
    if mask is None:
        mask = torch.ones(retention.shape[:2], dtype=torch.bool, device=retention.device)
    elif (
        mask.dtype != torch.bool
        or mask.device != retention.device
        or tuple(mask.shape) != tuple(retention.shape[:2])
    ):
        raise ValueError("mask must be boolean [B, T] on the transition device")
    aggregate = AffineRecallTransition(
        torch.ones_like(retention[:, 0]),
        torch.zeros_like(write[:, 0]),
    )
    expand = (1,) * (write.ndim - 2)
    for position in range(retention.shape[1]):
        valid = mask[:, position].reshape(-1, *expand)
        step = AffineRecallTransition(
            torch.where(valid, retention[:, position], torch.ones_like(retention[:, position])),
            torch.where(valid, write[:, position], torch.zeros_like(write[:, position])),
        )
        aggregate = aggregate.then(step)
    return aggregate


class AffineRecallValueUpdater(nn.Module):
    """Produce strictly composable transitions for a values-only Recall Bank.

    Token transitions depend on the trace but never on the previous Bank. The
    resulting state update is therefore affine in ``V`` and can be composed in
    token, chunk, or tree order without changing the result.
    """

    state_semantics = "values_only"
    transition_semantics = "affine_monoid"

    def __init__(
        self,
        hidden_dim: int,
        slots: int,
        *,
        workspace_dim: int = 256,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or slots <= 0 or workspace_dim <= 0:
            raise ValueError("hidden_dim, slots, and workspace_dim must be positive")
        self.hidden_dim = int(hidden_dim)
        self.slots = int(slots)
        self.workspace_dim = int(workspace_dim)
        self.recall_slots = 0
        self.recall_group_topk = 0
        self.trace_projection = nn.Linear(hidden_dim, workspace_dim, bias=False)
        self.slot_identity = nn.Parameter(torch.empty(slots, workspace_dim))
        self.retention_weight = nn.Parameter(torch.empty(slots, workspace_dim))
        self.retention_bias = nn.Parameter(torch.empty(slots))
        self.value_weight = nn.Parameter(torch.empty(slots, workspace_dim, hidden_dim))
        self.output_scale = workspace_dim**-0.5
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.slot_identity, std=self.workspace_dim**-0.5)
        nn.init.zeros_(self.retention_weight)
        nn.init.constant_(self.retention_bias, -6.0)
        nn.init.zeros_(self.value_weight)

    def transitions(
        self,
        trace: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> AffineRecallTransition:
        trace_b, mask_b, _ = self._normalize_trace(trace, mask)
        workspace = torch.tanh(
            self.trace_projection(trace_b).unsqueeze(2)
            * self.slot_identity.to(trace_b).reshape(1, 1, self.slots, -1)
        )
        retention_logits = torch.einsum(
            "btsw,sw->bts",
            workspace,
            self.retention_weight.to(workspace),
        )
        retention_logits = retention_logits + self.retention_bias.to(workspace)
        retention = torch.exp(-torch.nn.functional.softplus(retention_logits)).unsqueeze(-1)
        write = torch.einsum(
            "btsw,swh->btsh",
            workspace,
            self.value_weight.to(workspace),
        )
        write = write * self.output_scale
        valid = mask_b.reshape(*mask_b.shape, 1, 1)
        retention = torch.where(valid, retention, torch.ones_like(retention))
        write = torch.where(valid, write, torch.zeros_like(write))
        return AffineRecallTransition(retention, write)

    def forward(
        self,
        trace: Tensor,
        previous_value: Tensor,
        *,
        mask: Tensor | None = None,
        return_transition: bool = False,
    ) -> Tensor | tuple[Tensor, AffineRecallTransition]:
        trace_b, mask_b, squeeze = self._normalize_trace(trace, mask)
        value_b = previous_value.unsqueeze(0) if squeeze else previous_value
        expected = (trace_b.shape[0], self.slots, self.hidden_dim)
        if tuple(value_b.shape) != expected:
            raise ValueError(f"previous_value must have shape {expected}")
        token_transitions = self.transitions(trace_b, mask=mask_b)
        transition = reduce_affine_recall_transitions(
            token_transitions.retention,
            token_transitions.write,
            mask=mask_b,
        )
        result = transition.apply(value_b)
        if squeeze:
            result = result.squeeze(0)
            transition = AffineRecallTransition(
                transition.retention.squeeze(0),
                transition.write.squeeze(0),
            )
        if return_transition:
            return result, transition
        return result

    def _normalize_trace(
        self,
        trace: Tensor,
        mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, bool]:
        if not isinstance(trace, Tensor) or not trace.is_floating_point():
            raise TypeError("trace must be a floating-point Tensor")
        if trace.ndim not in {2, 3} or trace.shape[-1] != self.hidden_dim:
            raise ValueError("trace must have shape [T, H] or [B, T, H]")
        squeeze = trace.ndim == 2
        trace_b = trace.unsqueeze(0) if squeeze else trace
        if mask is None:
            mask_b = torch.ones(trace_b.shape[:2], dtype=torch.bool, device=trace.device)
        else:
            mask_b = mask.unsqueeze(0) if squeeze and mask.ndim == 1 else mask
            if (
                mask_b.dtype != torch.bool
                or mask_b.device != trace.device
                or tuple(mask_b.shape) != tuple(trace_b.shape[:2])
            ):
                raise ValueError("mask must match the trace batch and token dimensions")
        return trace_b, mask_b, squeeze


def reduce_matrix_affine_recall_transitions(
    matrix: Tensor,
    write: Tensor,
    *,
    mask: Tensor | None = None,
) -> MatrixAffineRecallTransition:
    """Reduce ordered full-matrix token transitions exactly."""

    if matrix.ndim != 4 or write.ndim != 4:
        raise ValueError("matrix and write must have shapes [B, T, S, S/H]")
    if matrix.shape[:2] != write.shape[:2] or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("matrix and write must share [B, T, S]")
    if matrix.shape[-1] != write.shape[-2]:
        raise ValueError("matrix and write slot dimensions must match")
    if mask is None:
        mask = torch.ones(matrix.shape[:2], dtype=torch.bool, device=matrix.device)
    elif (
        mask.dtype != torch.bool
        or mask.device != matrix.device
        or tuple(mask.shape) != tuple(matrix.shape[:2])
    ):
        raise ValueError("mask must be boolean [B, T] on the transition device")
    batch_size, token_count, slots, _ = matrix.shape
    aggregate = MatrixAffineRecallTransition(
        torch.eye(slots, device=matrix.device, dtype=matrix.dtype)
        .reshape(1, slots, slots)
        .expand(batch_size, -1, -1),
        torch.zeros_like(write[:, 0]),
    )
    for position in range(token_count):
        valid = mask[:, position].reshape(batch_size, 1, 1)
        identity = torch.eye(slots, device=matrix.device, dtype=matrix.dtype)
        step = MatrixAffineRecallTransition(
            torch.where(valid, matrix[:, position], identity),
            torch.where(valid, write[:, position], torch.zeros_like(write[:, position])),
        )
        aggregate = aggregate.then(step)
    return aggregate


def compose_matrix_affine_recall_transitions(
    matrix: Tensor,
    write: Tensor,
    *,
    order: Sequence[int] | None = None,
) -> MatrixAffineRecallTransition:
    """Compose a batch of independent transitions in an explicit order.

    ``reduce_matrix_affine_recall_transitions`` reduces the token axis and
    keeps each batch item independent.  This helper is the second reduction
    needed by a stateful TTT writer: each batch item represents one prompt
    (or one support trace), while the result is one shared state after those
    prompts have been applied in order.

    For transitions ``T_i(V) = A_i @ V + b_i``, the returned transition is
    exactly ``T_order[-1] o ... o T_order[0]``.  It is therefore equivalent to
    serial state updates, without averaging states or introducing a learned
    merge.  The batch axis is reduced only here; the updater itself can still
    compute trace projections and per-token transitions in parallel.

    Args:
        matrix: Transition matrices with shape ``[B, S, S]``.
        write: Transition writes with shape ``[B, S, H]``.
        order: Optional prompt order.  Defaults to ``range(B)``.

    Returns:
        A single ``MatrixAffineRecallTransition`` with shapes ``[S, S]`` and
        ``[S, H]``.
    """

    if matrix.ndim != 3 or write.ndim != 3:
        raise ValueError("matrix and write must have shapes [B, S, S/H]")
    if matrix.shape[0] != write.shape[0]:
        raise ValueError("matrix and write must share the batch dimension")
    if matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError("matrix must end in square [S, S] dimensions")
    if matrix.shape[-1] != write.shape[-2]:
        raise ValueError("matrix and write must share the slot dimension")
    if matrix.device != write.device:
        raise ValueError("matrix and write must share a device")
    if not matrix.is_floating_point() or not write.is_floating_point():
        raise TypeError("matrix and write must be floating-point Tensors")

    batch_size, slots, _ = matrix.shape
    if batch_size <= 0:
        raise ValueError("matrix and write must contain at least one transition")
    if order is None:
        indices = tuple(range(batch_size))
    else:
        indices = tuple(int(index) for index in order)
        if len(indices) != batch_size:
            raise ValueError("order must contain exactly one index per batch item")
        if sorted(indices) != list(range(batch_size)):
            raise ValueError("order must be a permutation of the batch indices")

    aggregate = MatrixAffineRecallTransition(
        torch.eye(slots, device=matrix.device, dtype=matrix.dtype),
        torch.zeros_like(write[0]),
    )
    for index in indices:
        aggregate = aggregate.then(
            MatrixAffineRecallTransition(matrix[index], write[index])
        )
    return aggregate


class NormalizedDeltaRecallValueUpdater(nn.Module):
    """Write keyed prediction residuals into a values-only Recall Bank.

    Each token defines one or more normalized slot keys ``k``, target values
    ``v``, and update rates ``beta``. Each factor applies the update

    ``V' = V + beta * k (v - k.T @ V) / (eps + ||k||^2)``

    is affine in ``V`` and therefore composes exactly across tokens or chunks.
    """

    state_semantics = "values_only"
    transition_semantics = "matrix_affine_monoid"

    def __init__(
        self,
        hidden_dim: int,
        slots: int,
        *,
        workspace_dim: int = 256,
        factors: int = 1,
        epsilon: float = 1e-6,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or slots <= 0 or workspace_dim <= 0 or factors <= 0:
            raise ValueError(
                "hidden_dim, slots, workspace_dim, and factors must be positive"
            )
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.hidden_dim = int(hidden_dim)
        self.slots = int(slots)
        self.workspace_dim = int(workspace_dim)
        self.factors = int(factors)
        self.epsilon = float(epsilon)
        self.recall_slots = 0
        self.recall_group_topk = 0
        self.trace_projection = nn.Linear(hidden_dim, workspace_dim, bias=False)
        if self.factors == 1:
            self.key_weight = nn.Parameter(torch.empty(slots, workspace_dim))
            self.value_weight = nn.Parameter(torch.empty(workspace_dim, hidden_dim))
            self.rate_weight = nn.Parameter(torch.empty(workspace_dim))
            self.rate_bias = nn.Parameter(torch.zeros(()))
        else:
            self.key_weight = nn.Parameter(
                torch.empty(self.factors, slots, workspace_dim)
            )
            self.value_weight = nn.Parameter(
                torch.empty(self.factors, workspace_dim, hidden_dim)
            )
            self.rate_weight = nn.Parameter(torch.empty(self.factors, workspace_dim))
            self.rate_bias = nn.Parameter(torch.zeros(self.factors))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.key_weight, std=self.workspace_dim**-0.5)
        nn.init.zeros_(self.value_weight)
        nn.init.zeros_(self.rate_weight)
        nn.init.zeros_(self.rate_bias)

    def transitions(
        self,
        trace: Tensor,
        *,
        mask: Tensor | None = None,
    ) -> MatrixAffineRecallTransition:
        trace_b, mask_b, _ = self._normalize_trace(trace, mask)
        slots = self.slots
        key, target, beta = self._write_parameters(trace_b)
        identity = torch.eye(slots, device=trace.device, dtype=key.dtype)
        if self.factors == 1:
            matrix, write = self._transition_terms(key, target, beta, identity)
        else:
            batch_size, token_count = trace_b.shape[:2]
            matrix = identity.reshape(1, 1, slots, slots).expand(
                batch_size, token_count, -1, -1
            )
            write = torch.zeros(
                batch_size,
                token_count,
                slots,
                self.hidden_dim,
                device=trace.device,
                dtype=key.dtype,
            )
            for factor in range(self.factors):
                step_matrix, step_write = self._transition_terms(
                    key[:, :, factor],
                    target[:, :, factor],
                    beta[:, :, factor],
                    identity,
                )
                write = torch.matmul(step_matrix, write) + step_write
                matrix = torch.matmul(step_matrix, matrix)
        valid = mask_b.reshape(*mask_b.shape, 1, 1)
        matrix = torch.where(valid, matrix, identity)
        write = torch.where(valid, write, torch.zeros_like(write))
        return MatrixAffineRecallTransition(matrix, write)

    def _transition_terms(
        self,
        key: Tensor,
        target: Tensor,
        beta: Tensor,
        identity: Tensor,
    ) -> tuple[Tensor, Tensor]:
        denominator = key.square().sum(dim=-1, keepdim=True) + self.epsilon
        coefficient = beta.unsqueeze(-1) / denominator
        outer = key.unsqueeze(-1) * key.unsqueeze(-2)
        matrix = identity - coefficient.unsqueeze(-1) * outer
        write = coefficient.unsqueeze(-1) * key.unsqueeze(-1) * target.unsqueeze(-2)
        return matrix, write

    def _write_parameters(self, trace: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        workspace = torch.tanh(self.trace_projection(trace))
        if self.factors == 1:
            key = torch.einsum("btw,sw->bts", workspace, self.key_weight.to(workspace))
            key = F.normalize(key, dim=-1, eps=self.epsilon)
            target = torch.einsum(
                "btw,wh->bth", workspace, self.value_weight.to(workspace)
            )
            beta = torch.sigmoid(
                torch.einsum("btw,w->bt", workspace, self.rate_weight.to(workspace))
                + self.rate_bias.to(workspace)
            )
        else:
            key = torch.einsum(
                "btw,fsw->btfs", workspace, self.key_weight.to(workspace)
            )
            key = F.normalize(key, dim=-1, eps=self.epsilon)
            target = torch.einsum(
                "btw,fwh->btfh", workspace, self.value_weight.to(workspace)
            )
            beta = torch.sigmoid(
                torch.einsum("btw,fw->btf", workspace, self.rate_weight.to(workspace))
                + self.rate_bias.to(workspace)
            )
        return key, target, beta

    def forward(
        self,
        trace: Tensor,
        previous_value: Tensor,
        *,
        mask: Tensor | None = None,
        return_transition: bool = False,
    ) -> Tensor | tuple[Tensor, MatrixAffineRecallTransition]:
        trace_b, mask_b, squeeze = self._normalize_trace(trace, mask)
        value_b = previous_value.unsqueeze(0) if squeeze else previous_value
        expected = (trace_b.shape[0], self.slots, self.hidden_dim)
        if tuple(value_b.shape) != expected:
            raise ValueError(f"previous_value must have shape {expected}")
        if return_transition:
            token_transitions = self.transitions(trace_b, mask=mask_b)
            transition = reduce_matrix_affine_recall_transitions(
                token_transitions.matrix,
                token_transitions.write,
                mask=mask_b,
            )
            result = transition.apply(value_b)
        else:
            key, target, beta = self._write_parameters(trace_b)
            if self.factors == 1:
                key = key.unsqueeze(2)
                target = target.unsqueeze(2)
                beta = beta.unsqueeze(2)
            result = value_b
            for position in range(trace_b.shape[1]):
                for factor in range(self.factors):
                    key_t = key[:, position, factor]
                    target_t = target[:, position, factor]
                    prediction = torch.einsum("bs,bsh->bh", key_t, result)
                    coefficient = beta[:, position, factor] / (
                        key_t.square().sum(dim=-1) + self.epsilon
                    )
                    update = (
                        coefficient.reshape(-1, 1, 1)
                        * key_t.unsqueeze(-1)
                        * (target_t - prediction).unsqueeze(1)
                    )
                    valid = mask_b[:, position].reshape(-1, 1, 1)
                    result = torch.where(valid, result + update, result)
        if squeeze:
            result = result.squeeze(0)
            if return_transition:
                transition = MatrixAffineRecallTransition(
                    transition.matrix.squeeze(0),
                    transition.write.squeeze(0),
                )
        if return_transition:
            return result, transition
        return result

    def _normalize_trace(
        self,
        trace: Tensor,
        mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, bool]:
        if not isinstance(trace, Tensor) or not trace.is_floating_point():
            raise TypeError("trace must be a floating-point Tensor")
        if trace.ndim not in {2, 3} or trace.shape[-1] != self.hidden_dim:
            raise ValueError("trace must have shape [T, H] or [B, T, H]")
        squeeze = trace.ndim == 2
        trace_b = trace.unsqueeze(0) if squeeze else trace
        if mask is None:
            mask_b = torch.ones(trace_b.shape[:2], dtype=torch.bool, device=trace.device)
        else:
            mask_b = mask.unsqueeze(0) if squeeze and mask.ndim == 1 else mask
            if (
                mask_b.dtype != torch.bool
                or mask_b.device != trace.device
                or tuple(mask_b.shape) != tuple(trace_b.shape[:2])
            ):
                raise ValueError("mask must match the trace batch and token dimensions")
        return trace_b, mask_b, squeeze


class _RecallWorkspaceBlock(nn.Module):
    """An ARTI block with a linear-size virtual-interface interaction path."""

    def __init__(
        self,
        dim: int,
        *,
        interface_slots: int,
        dropout: float,
        recall_slots: int,
        recall_group_topk: int,
        recall_route_exploration: float,
        recall_steps: int,
        recall_min_steps: int,
        recall_tolerance: float | None,
        recall_seed: int,
    ) -> None:
        super().__init__()
        self.layer = ARTILayer(
            input_dim=dim,
            hidden_dim=dim,
            dropout=dropout,
            interface_slots=interface_slots,
            recall_slots=recall_slots,
            recall_steps=recall_steps,
            recall_min_steps=recall_min_steps,
            recall_tolerance=recall_tolerance,
            recall_routing="grouped",
            recall_key_dim=min(32, dim),
            recall_query_mode="fixed",
            recall_query_seed=recall_seed,
            recall_group_size=1,
            recall_group_topk=recall_group_topk,
            recall_route_exploration=recall_route_exploration,
            use_phase_mixer=False,
            use_virtual_interface=True,
            use_pairwise_context=False,
            use_recall=recall_steps > 0,
            use_virtual_recall=False,
        )
        if self.layer.state.recall is not None:
            with torch.no_grad():
                self.layer.state.recall.bank.zero_()
        self.norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: Tensor,
        mask: Tensor,
        *,
        recall_steps: int | None = None,
        selected_recall_groups: Tensor | None = None,
        return_route: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        output = self.layer(
            x,
            mask=mask,
            recall_steps=recall_steps,
            _selected_recall_groups=selected_recall_groups,
            _selected_groups_normalized=selected_recall_groups is not None,
        )
        updated = self.norm(x + output.y)
        if not return_route:
            return updated
        indices = output.diagnostics["recall_bank_indices"]
        groups = indices if indices.ndim == x.ndim else indices[..., 0]
        return updated, groups


class RecallValueUpdater(nn.Module):
    """Map a processed tensor trace and the previous Bank values to new values.

    This is deliberately an internal research primitive. It does not prescribe
    addresses, erase gates, write gates, update budgets, or an online optimizer.
    The caller owns the returned state and decides when to detach or persist it.

    The public mathematical contract is a complete state transition::

        next_value = updater(trace, previous_value, mask=mask)

    Internally the output is identity-centred at initialization so an untrained
    updater leaves the state unchanged. The learned update is otherwise
    unrestricted and may preserve, replace, amplify, or cancel previous values.
    """

    def __init__(
        self,
        hidden_dim: int,
        slots: int,
        *,
        workspace_dim: int = 256,
        depth: int = 2,
        interface_slots: int = 8,
        dropout: float = 0.0,
        recall_slots: int = 8,
        recall_group_topk: int | None = None,
        recall_route_exploration: float = 0.0,
        recall_steps: int = 0,
        recall_min_steps: int = 1,
        recall_tolerance: float | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or slots <= 0 or workspace_dim <= 0:
            raise ValueError("hidden_dim, slots, and workspace_dim must be positive")
        if depth <= 0:
            raise ValueError("depth must be positive")
        if interface_slots <= 0:
            raise ValueError("interface_slots must be positive")
        if recall_slots <= 0:
            raise ValueError("recall_slots must be positive")
        resolved_recall_group_topk = (
            min(2, recall_slots)
            if recall_group_topk is None
            else int(recall_group_topk)
        )
        if not 1 <= resolved_recall_group_topk <= recall_slots:
            raise ValueError("recall_group_topk must be in [1, recall_slots]")
        if recall_steps < 0:
            raise ValueError("recall_steps must be non-negative")
        if not math.isfinite(recall_route_exploration) or recall_route_exploration < 0.0:
            raise ValueError("recall_route_exploration must be finite and non-negative")
        if recall_steps > 0 and not 1 <= recall_min_steps <= recall_steps:
            raise ValueError("recall_min_steps must be in [1, recall_steps]")
        if recall_tolerance is not None and (
            not math.isfinite(recall_tolerance) or recall_tolerance < 0.0
        ):
            raise ValueError("recall_tolerance must be finite and non-negative")
        if not math.isfinite(dropout) or not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the interval [0, 1)")

        self.hidden_dim = int(hidden_dim)
        self.slots = int(slots)
        self.workspace_dim = int(workspace_dim)
        self.depth = int(depth)
        self.recall_slots = int(recall_slots)
        self.recall_group_topk = resolved_recall_group_topk
        self.recall_route_exploration = float(recall_route_exploration)
        self.recall_steps = int(recall_steps)
        self.recall_min_steps = int(recall_min_steps)
        self.recall_tolerance = recall_tolerance
        self.output_scale = workspace_dim**-0.5

        self.state_projection = nn.Linear(hidden_dim, workspace_dim)
        self.trace_projection = nn.Linear(hidden_dim, workspace_dim)
        self.slot_identity = nn.Parameter(torch.empty(slots, workspace_dim))
        self.state_kind = nn.Parameter(torch.empty(workspace_dim))
        self.trace_kind = nn.Parameter(torch.empty(workspace_dim))
        self.workspace = nn.ModuleList(
            _RecallWorkspaceBlock(
                workspace_dim,
                interface_slots=min(interface_slots, slots),
                dropout=dropout,
                recall_slots=recall_slots,
                recall_group_topk=resolved_recall_group_topk,
                recall_route_exploration=recall_route_exploration,
                recall_steps=recall_steps,
                recall_min_steps=recall_min_steps,
                recall_tolerance=recall_tolerance,
                recall_seed=9173 + block_index,
            )
            for block_index in range(depth)
        )

        # Each Bank slot owns a full independent W x H factor generator. These
        # values are never applied to the host directly: the attached Recall
        # Formula owns the actual host-state perturbation.
        self.value_weight = nn.Parameter(torch.empty(slots, workspace_dim, hidden_dim))
        self.value_bias = nn.Parameter(torch.empty(slots, hidden_dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize a live workspace with an identity state transition."""

        nn.init.normal_(self.slot_identity, std=self.workspace_dim**-0.5)
        nn.init.normal_(self.state_kind, std=self.workspace_dim**-0.5)
        nn.init.normal_(self.trace_kind, std=self.workspace_dim**-0.5)
        nn.init.zeros_(self.value_weight)
        nn.init.zeros_(self.value_bias)

    def forward(
        self,
        trace: Tensor,
        previous_value: Tensor,
        *,
        mask: Tensor | None = None,
        recall_steps: int | None = None,
        route_plan: Tensor | None = None,
        return_info: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        """Return the complete next Bank value tensor.

        ``trace`` has shape ``[T, H]`` or ``[B, T, H]`` and ``previous_value``
        has shape ``[S, H]`` or ``[B, S, H]``. Their batch conventions must
        match. Invalid trace positions are excluded from the workspace, and a
        sample with no valid trace positions leaves its value state unchanged.
        """

        slot_workspace, value_b, has_trace, squeeze, selected_route_plan = (
            self._workspace_state(
            trace,
            previous_value,
            mask=mask,
            recall_steps=recall_steps,
            route_plan=route_plan,
            return_route_plan=return_info,
            )
        )
        update = torch.einsum(
            "bsw,swh->bsh",
            slot_workspace,
            self.value_weight.to(slot_workspace),
        )
        update = update * self.output_scale
        update = update + self.value_bias.to(update).unsqueeze(0)
        update = update * has_trace.to(update.dtype)
        next_value = value_b + update.to(value_b)

        if squeeze:
            next_value = next_value.squeeze(0)
            update = update.squeeze(0)
            slot_workspace = slot_workspace.squeeze(0)
            if selected_route_plan.numel():
                selected_route_plan = selected_route_plan.squeeze(0)
        if not return_info:
            return next_value
        return next_value, {
            "update": update,
            "update_norm": torch.linalg.vector_norm(update.float(), dim=(-2, -1)),
            "slot_workspace": slot_workspace,
            "route_plan": selected_route_plan,
        }

    def _workspace_state(
        self,
        trace: Tensor,
        previous_value: Tensor,
        *,
        mask: Tensor | None,
        recall_steps: int | None,
        route_plan: Tensor | None = None,
        return_route_plan: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, bool, Tensor]:
        """Return the state workspace before the independent output map."""

        trace_b, value_b, mask_b, squeeze = self._normalize_inputs(
            trace,
            previous_value,
            mask,
        )
        state_tokens = (
            self.state_projection(value_b)
            + self.slot_identity.to(value_b).unsqueeze(0)
            + self.state_kind.to(value_b).reshape(1, 1, -1)
        )
        trace_tokens = self.trace_projection(trace_b)
        trace_tokens = trace_tokens + self.trace_kind.to(trace_b).reshape(1, 1, -1)
        trace_tokens = trace_tokens * mask_b.unsqueeze(-1).to(trace_tokens.dtype)

        workspace = torch.cat((state_tokens, trace_tokens), dim=1)
        state_mask = torch.ones(
            trace_b.shape[0],
            self.slots,
            dtype=torch.bool,
            device=trace_b.device,
        )
        workspace_mask = torch.cat((state_mask, mask_b), dim=1)
        route_plan_b = route_plan
        if squeeze and route_plan is not None and route_plan.ndim == 3:
            route_plan_b = route_plan.unsqueeze(0)
        if route_plan_b is not None:
            expected = (
                trace_b.shape[0],
                self.depth,
                workspace.shape[1],
                self.recall_group_topk,
            )
            if tuple(route_plan_b.shape) != expected:
                raise ValueError(f"route_plan must have shape {expected}")
            if route_plan_b.dtype != torch.long or route_plan_b.device != trace_b.device:
                raise ValueError("route_plan must be a long Tensor on the trace device")
        selected_routes = []
        for block_index, block in enumerate(self.workspace):
            selected = None if route_plan_b is None else route_plan_b[:, block_index]
            if return_route_plan:
                workspace, selected = block(
                    workspace,
                    mask=workspace_mask,
                    recall_steps=recall_steps,
                    selected_recall_groups=selected,
                    return_route=True,
                )
                selected_routes.append(selected)
            else:
                workspace = block(
                    workspace,
                    mask=workspace_mask,
                    recall_steps=recall_steps,
                    selected_recall_groups=selected,
                )

        slot_workspace = workspace[:, : self.slots]
        has_trace = mask_b.any(dim=-1).reshape(-1, 1, 1)
        selected_route_plan = (
            torch.stack(selected_routes, dim=1)
            if selected_routes
            else torch.empty(0, dtype=torch.long, device=trace_b.device)
        )
        return slot_workspace, value_b, has_trace, squeeze, selected_route_plan

    def freeze(self) -> "RecallValueUpdater":
        """Freeze the offline-trained transition for forward-only use."""

        self.eval()
        self.requires_grad_(False)
        return self

    def transition_parameter_count(self) -> int:
        """Return the number of parameters in the independent slot output maps."""

        return self.value_weight.numel() + self.value_bias.numel()

    def _normalize_inputs(
        self,
        trace: Tensor,
        previous_value: Tensor,
        mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, bool]:
        if not isinstance(trace, Tensor) or not trace.is_floating_point():
            raise TypeError("trace must be a floating-point Tensor")
        if not isinstance(previous_value, Tensor) or not previous_value.is_floating_point():
            raise TypeError("previous_value must be a floating-point Tensor")
        if trace.ndim not in {2, 3}:
            raise ValueError("trace must have shape [T, H] or [B, T, H]")
        if previous_value.ndim not in {2, 3}:
            raise ValueError("previous_value must have shape [S, H] or [B, S, H]")
        if trace.ndim != previous_value.ndim:
            raise ValueError("trace and previous_value must use the same batch convention")
        if trace.shape[-1] != self.hidden_dim:
            raise ValueError(f"trace feature dim must be {self.hidden_dim}, got {trace.shape[-1]}")
        if previous_value.shape[-2:] != (self.slots, self.hidden_dim):
            raise ValueError(
                "previous_value trailing shape must be "
                f"({self.slots}, {self.hidden_dim}), got {tuple(previous_value.shape[-2:])}"
            )
        if trace.device != previous_value.device or trace.dtype != previous_value.dtype:
            raise ValueError("trace and previous_value must share device and dtype")

        squeeze = trace.ndim == 2
        trace_b = trace.unsqueeze(0) if squeeze else trace
        value_b = previous_value.unsqueeze(0) if squeeze else previous_value
        if trace_b.shape[0] != value_b.shape[0]:
            raise ValueError("trace and previous_value batch sizes must match")
        if mask is None:
            mask_b = torch.ones(trace_b.shape[:2], dtype=torch.bool, device=trace.device)
        else:
            expected = trace.shape[:-1]
            if mask.shape != expected:
                raise ValueError(f"mask must have shape {tuple(expected)}, got {tuple(mask.shape)}")
            if mask.dtype != torch.bool:
                raise ValueError("mask must be a boolean Tensor")
            if mask.device != trace.device:
                raise ValueError("mask must be on the trace device")
            mask_b = mask.unsqueeze(0) if squeeze else mask
        return trace_b, value_b, mask_b, squeeze

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, slots={self.slots}, "
            f"workspace_dim={self.workspace_dim}, depth={self.depth}, "
            f"recall_slots={self.recall_slots}, "
            f"recall_group_topk={self.recall_group_topk}, "
            f"recall_route_exploration={self.recall_route_exploration}, "
            f"recall_steps={self.recall_steps}"
        )


_RECALL_CAPACITY_SUFFIXES = (
    ".layer.state.recall.bank",
    ".layer.state.recall.group_bank",
    ".layer.state.recall.key_bank",
)


def adapt_recall_capacity_state(
    updater: RecallValueUpdater,
    state: dict[str, Tensor],
    *,
    symmetry_break: float = 0.0,
) -> tuple[dict[str, Tensor], dict[str, Any]]:
    """Expand internal Recall rows while preserving the inherited read.

    Capacity expansion is only valid by an integer factor. Each old route is
    repeated contiguously; callers must increase ``recall_group_topk`` by the
    same factor. With zero symmetry breaking this makes the grouped weighted
    read mathematically equivalent before further training.
    """

    if not math.isfinite(symmetry_break) or symmetry_break < 0.0:
        raise ValueError("symmetry_break must be finite and non-negative")
    target = updater.state_dict()
    adapted = dict(state)
    expanded_names: list[str] = []
    expansion_factor: int | None = None
    source_slots: int | None = None
    inferred_source_topk: int | None = None
    for name, source in state.items():
        destination = target.get(name)
        if destination is None or source.shape == destination.shape:
            continue
        if not name.endswith(_RECALL_CAPACITY_SUFFIXES):
            continue
        if source.ndim < 1 or source.shape[1:] != destination.shape[1:]:
            continue
        if destination.shape[0] % source.shape[0]:
            raise ValueError(
                f"Recall capacity for {name} must expand by an integer factor"
            )
        factor = destination.shape[0] // source.shape[0]
        if factor <= 1:
            raise ValueError("Recall capacity migration only supports expansion")
        if expansion_factor is not None and factor != expansion_factor:
            raise ValueError("inconsistent Recall capacity expansion factors")
        expansion_factor = factor
        source_slots = int(source.shape[0])
        expanded = source.repeat_interleave(factor, dim=0)
        if symmetry_break and name.endswith(("group_bank", "key_bank")):
            feature_shape = source.shape[1:]
            direction = torch.arange(
                math.prod(feature_shape),
                device=source.device,
                dtype=torch.float32,
            ).reshape(feature_shape)
            direction = torch.sin(direction + 1.0)
            direction = direction / direction.square().mean().sqrt().clamp_min(1e-12)
            offsets = torch.arange(factor, device=source.device, dtype=torch.float32)
            offsets = offsets - offsets.mean()
            perturbation = offsets.reshape(1, factor, *([1] * len(feature_shape)))
            perturbation = perturbation * direction.reshape(1, 1, *feature_shape)
            expanded = expanded.reshape(source.shape[0], factor, *feature_shape)
            expanded = expanded + perturbation.to(expanded) * symmetry_break
            expanded = expanded.reshape_as(destination)
        adapted[name] = expanded
        expanded_names.append(name)

    if expansion_factor is not None:
        if updater.recall_group_topk % expansion_factor:
            raise ValueError(
                "behavior-preserving Recall expansion requires "
                "recall_group_topk to increase by the capacity factor"
            )
        inferred_source_topk = updater.recall_group_topk // expansion_factor
        if not 1 <= inferred_source_topk <= (source_slots or 0):
            raise ValueError("expanded Recall group_topk implies an invalid source top-k")
    return adapted, {
        "source_slots": source_slots,
        "target_slots": updater.recall_slots,
        "factor": expansion_factor or 1,
        "inferred_source_topk": inferred_source_topk,
        "target_topk": updater.recall_group_topk,
        "expanded_parameters": tuple(expanded_names),
        "symmetry_break": float(symmetry_break),
    }


class StackedRecallValueUpdater(nn.Module):
    """Execute independent, structurally identical updaters over a site axis."""

    def __init__(self, modules: Sequence[RecallValueUpdater]) -> None:
        super().__init__()
        if not modules:
            raise ValueError("at least one updater is required")
        reference = modules[0]
        if any(
            module.hidden_dim != reference.hidden_dim
            or module.slots != reference.slots
            or module.workspace_dim != reference.workspace_dim
            for module in modules[1:]
        ):
            raise ValueError("stacked updaters must share structural dimensions")

        parameters, buffers = stack_module_state(list(modules))
        template = copy.deepcopy(reference).to("meta")
        object.__setattr__(self, "_functional_template", template)
        persistent_state_names = frozenset(reference.state_dict())
        self._parameter_names = tuple(parameters)
        self._buffer_names = tuple(buffers)
        self._persistent_buffer_names = tuple(
            name for name in buffers if name in persistent_state_names
        )
        self._parameter_attributes: dict[str, str] = {}
        self._buffer_attributes: dict[str, str] = {}
        for index, (name, parameter) in enumerate(parameters.items()):
            attribute = f"stacked_parameter_{index}"
            self.register_parameter(
                attribute,
                nn.Parameter(parameter, requires_grad=parameter.requires_grad),
            )
            self._parameter_attributes[name] = attribute
        for index, (name, buffer) in enumerate(buffers.items()):
            attribute = f"stacked_buffer_{index}"
            self.register_buffer(
                attribute,
                buffer,
                persistent=name in persistent_state_names,
            )
            self._buffer_attributes[name] = attribute

        self.site_count = len(modules)
        self.hidden_dim = reference.hidden_dim
        self.slots = reference.slots
        self.workspace_dim = reference.workspace_dim
        self.depth = reference.depth
        self.recall_group_topk = reference.recall_group_topk

    def _parameters_by_name(self) -> dict[str, Tensor]:
        return {
            name: getattr(self, self._parameter_attributes[name])
            for name in self._parameter_names
        }

    def _buffers_by_name(self) -> dict[str, Tensor]:
        return {
            name: getattr(self, self._buffer_attributes[name])
            for name in self._buffer_names
        }

    def named_updater_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        return tuple(
            (name, getattr(self, self._parameter_attributes[name]))
            for name in self._parameter_names
        )

    def parameter_for(self, name: str) -> nn.Parameter:
        try:
            return getattr(self, self._parameter_attributes[name])
        except KeyError as error:
            raise KeyError(f"unknown updater parameter: {name}") from error

    def site_state_dict(self, site: int) -> dict[str, Tensor]:
        if not 0 <= site < self.site_count:
            raise IndexError("stacked updater site is out of range")
        result = {
            name: tensor[site].detach()
            for name, tensor in self._parameters_by_name().items()
        }
        result.update(
            {
                name: tensor[site].detach()
                for name, tensor in self._buffers_by_name().items()
                if name in self._persistent_buffer_names
            }
        )
        return result

    def train(self, mode: bool = True) -> "StackedRecallValueUpdater":
        super().train(mode)
        template = object.__getattribute__(self, "_functional_template")
        template.train(mode)
        return self

    def forward(
        self,
        trace: Tensor,
        previous_value: Tensor,
        *,
        mask: Tensor,
        recall_steps: int | None = None,
        route_plan: Tensor | None = None,
        return_route_plan: bool = False,
    ) -> Tensor | tuple[Tensor, Tensor]:
        if trace.ndim != 4 or previous_value.ndim != 4 or mask.ndim != 3:
            raise ValueError(
                "stacked updater expects trace [B,L,T,H], value [B,L,S,H], "
                "and mask [B,L,T]"
            )
        if trace.shape[:2] != previous_value.shape[:2] or trace.shape[:3] != mask.shape:
            raise ValueError("stacked updater batch, site, or token axes do not match")
        if trace.shape[1] != self.site_count:
            raise ValueError("stacked updater site axis does not match its parameters")
        if route_plan is not None:
            expected_prefix = (trace.shape[0], trace.shape[1])
            if route_plan.shape[:2] != expected_prefix:
                raise ValueError("stacked route_plan batch and site axes do not match")
            if route_plan.dtype != torch.long or route_plan.device != trace.device:
                raise ValueError("stacked route_plan must be a long Tensor on the trace device")

        template = object.__getattribute__(self, "_functional_template")

        def apply_one(
            parameters: dict[str, Tensor],
            buffers: dict[str, Tensor],
            trace_one: Tensor,
            previous_one: Tensor,
            mask_one: Tensor,
        ) -> Tensor:
            return functional_call(
                template,
                (parameters, buffers),
                (trace_one, previous_one),
                {"mask": mask_one, "recall_steps": recall_steps},
            )

        parameters = self._parameters_by_name()
        buffers = self._buffers_by_name()
        trace_by_site = trace.transpose(0, 1)
        previous_by_site = previous_value.transpose(0, 1)
        mask_by_site = mask.transpose(0, 1)

        if return_route_plan:
            def apply_with_route(
                parameters_one: dict[str, Tensor],
                buffers_one: dict[str, Tensor],
                trace_one: Tensor,
                previous_one: Tensor,
                mask_one: Tensor,
                route_one: Tensor | None,
            ) -> tuple[Tensor, Tensor]:
                result, info = functional_call(
                    template,
                    (parameters_one, buffers_one),
                    (trace_one, previous_one),
                    {
                        "mask": mask_one,
                        "recall_steps": recall_steps,
                        "route_plan": route_one,
                        "return_info": True,
                    },
                )
                return result, info["route_plan"]

            if route_plan is None:
                def apply_without_input_route(
                    parameters_one: dict[str, Tensor],
                    buffers_one: dict[str, Tensor],
                    trace_one: Tensor,
                    previous_one: Tensor,
                    mask_one: Tensor,
                ) -> tuple[Tensor, Tensor]:
                    return apply_with_route(
                        parameters_one,
                        buffers_one,
                        trace_one,
                        previous_one,
                        mask_one,
                        None,
                    )

                output, selected = vmap(
                    apply_without_input_route,
                    randomness="different",
                )(parameters, buffers, trace_by_site, previous_by_site, mask_by_site)
            else:
                output, selected = vmap(apply_with_route, randomness="different")(
                    parameters,
                    buffers,
                    trace_by_site,
                    previous_by_site,
                    mask_by_site,
                    route_plan.transpose(0, 1),
                )
            return output.transpose(0, 1), selected.transpose(0, 1)

        if route_plan is None:
            output = vmap(apply_one, randomness="different")(
                parameters,
                buffers,
                trace_by_site,
                previous_by_site,
                mask_by_site,
            )
        else:
            def apply_replay(
                parameters_one: dict[str, Tensor],
                buffers_one: dict[str, Tensor],
                trace_one: Tensor,
                previous_one: Tensor,
                mask_one: Tensor,
                route_one: Tensor,
            ) -> Tensor:
                return functional_call(
                    template,
                    (parameters_one, buffers_one),
                    (trace_one, previous_one),
                    {
                        "mask": mask_one,
                        "recall_steps": recall_steps,
                        "route_plan": route_one,
                    },
                )

            output = vmap(apply_replay, randomness="different")(
                parameters,
                buffers,
                trace_by_site,
                previous_by_site,
                mask_by_site,
                route_plan.transpose(0, 1),
            )
        return output.transpose(0, 1)


__all__ = ["RecallValueUpdater", "StackedRecallValueUpdater"]
