"""Core tensor-in / tensor-out ARTI layers."""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Mapping, Sequence

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from ._compile import CompiledProductTail, grouped_product_tail
from .config import (
    ARTIConfig,
    STATE_RECALL_CONTENT_FACTORS,
    STATE_RECALL_COMPOSITION_FACTOR,
    STATE_RECALL_MODULATION_FACTORS,
)
from .functional import (
    as_sequence,
    apply_coord_frame_inverse,
    ensure_coord,
    ensure_mask,
    ensure_visibility,
    mask_coverage,
    masked_mean,
    masked_softmax,
    restore_input_rank,
)
from .init import init_arti_module
from .nn import Half

if TYPE_CHECKING:
    from .recall_policy import RecallParameterTag
from .outputs import ARTIOutput
from .recall_formula import RecallFormulaContract, check_recall_formula
from .utils import assert_floating_tensor, detach_diagnostics


_FIXED_QUERY_ALGORITHM = "rademacher-shake256-v1"


def _fixed_query_weight(hidden_dim: int, key_dim: int, seed: int) -> Tensor:
    """Build a cross-process deterministic normalized query projection."""

    if key_dim == hidden_dim:
        return torch.eye(hidden_dim, dtype=torch.float32)
    elements = hidden_dim * key_dim
    payload = f"{_FIXED_QUERY_ALGORITHM}:{seed}:{hidden_dim}:{key_dim}".encode()
    packed = torch.tensor(
        list(hashlib.shake_256(payload).digest((elements + 7) // 8)),
        dtype=torch.uint8,
    )
    shifts = torch.arange(8, dtype=torch.uint8)
    bits = torch.bitwise_and(
        torch.bitwise_right_shift(packed.unsqueeze(-1), shifts),
        1,
    )
    signs = bits.flatten()[:elements].to(torch.float32).mul_(2.0).sub_(1.0)
    return signs.reshape(key_dim, hidden_dim).mul_(hidden_dim**-0.5)


@dataclass
class _ARTIRecallRead:
    context: Tensor
    weights: Tensor
    influence: Tensor
    recognition: Tensor
    indices: Tensor
    route: Tensor

    def __iter__(self):
        yield self.context
        yield self.weights
        yield self.influence
        yield self.recognition


class _CoalescedSparseEmbeddingBag(torch.autograd.Function):
    """Embedding-bag sum with group-coalesced sparse weight gradients."""

    @staticmethod
    def forward(
        ctx,
        indices: Tensor,
        weight: Tensor,
        per_sample_weights: Tensor,
        group_size: int,
    ) -> Tensor:
        ctx.group_size = int(group_size)
        ctx.per_sample_dtype = per_sample_weights.dtype
        ctx.save_for_backward(indices, weight, per_sample_weights)
        return torch.nn.functional.embedding_bag(
            indices,
            weight,
            mode="sum",
            per_sample_weights=per_sample_weights.to(weight.dtype),
        )

    @staticmethod
    def _grouped_mm_backward(
        grouped_indices: Tensor,
        grouped_weights: Tensor,
        grad_output: Tensor,
        weight: Tensor,
    ) -> tuple[Tensor, Tensor] | None:
        grouped_mm = getattr(torch.nn.functional, "grouped_mm", None)
        if (
            grouped_mm is None
            or not grad_output.is_cuda
            or weight.dtype not in {torch.float32, torch.bfloat16}
        ):
            return None
        major, _minor = torch.cuda.get_device_capability(grad_output.device)
        if major < 8:
            return None

        token_count, reads_per_token, group_size = grouped_indices.shape
        feature_dim = weight.shape[-1]
        flat_group_ids = (
            grouped_indices[..., 0]
            .div(
                group_size,
                rounding_mode="floor",
            )
            .reshape(-1)
        )
        order = torch.argsort(flat_group_ids, stable=True)
        sorted_group_ids = flat_group_ids.index_select(0, order)
        unique_groups, counts = torch.unique_consecutive(
            sorted_group_ids,
            return_counts=True,
        )
        offsets = counts.cumsum(0).to(torch.int32)

        flat_weights = grouped_weights.reshape(-1, group_size)
        sorted_weights = flat_weights.index_select(0, order).contiguous()
        token_indices = (
            torch.arange(token_count, device=grad_output.device)
            .unsqueeze(1)
            .expand(token_count, reads_per_token)
            .reshape(-1)
            .index_select(0, order)
        )
        sorted_grad = grad_output.index_select(0, token_indices).contiguous()

        alignment = max(1, 16 // weight.element_size())
        padded_group_size = math.ceil(group_size / alignment) * alignment
        padded_feature_dim = math.ceil(feature_dim / alignment) * alignment
        sorted_weights = torch.nn.functional.pad(
            sorted_weights,
            (0, padded_group_size - group_size),
        )
        sorted_grad = torch.nn.functional.pad(
            sorted_grad,
            (0, padded_feature_dim - feature_dim),
        )
        slot_offsets = torch.arange(group_size, device=grad_output.device)
        grouped_rows = unique_groups.unsqueeze(1) * group_size + slot_offsets
        grouped_bank = torch.nn.functional.embedding(grouped_rows, weight)
        grouped_bank = torch.nn.functional.pad(
            grouped_bank,
            (
                0,
                padded_feature_dim - feature_dim,
                0,
                padded_group_size - group_size,
            ),
        )

        try:
            grouped_values = grouped_mm(
                sorted_weights.T,
                sorted_grad,
                offs=offsets,
            )
            sorted_per_sample = grouped_mm(
                sorted_grad,
                grouped_bank.transpose(-1, -2),
                offs=offsets,
            )
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise
            return None

        row_indices = grouped_rows.reshape(-1)
        row_values = grouped_values[
            :,
            :group_size,
            :feature_dim,
        ].reshape(-1, feature_dim)
        grad_weight = torch.sparse_coo_tensor(
            row_indices.unsqueeze(0),
            row_values,
            weight.shape,
            device=weight.device,
            dtype=weight.dtype,
            check_invariants=False,
            is_coalesced=True,
        )
        flat_per_sample = torch.empty_like(flat_weights)
        flat_per_sample.index_copy_(
            0,
            order,
            sorted_per_sample[:, :group_size],
        )
        return grad_weight, flat_per_sample.reshape_as(grouped_weights)

    @staticmethod
    def backward(ctx, grad_output: Tensor):
        indices, weight, per_sample_weights = ctx.saved_tensors
        group_size = ctx.group_size
        if indices.shape[-1] % group_size:
            raise RuntimeError("sparse Recall indices must contain complete groups")
        token_count = indices.shape[0]
        reads_per_token = indices.shape[-1] // group_size
        grouped_indices = indices.reshape(token_count, reads_per_token, group_size)
        grouped_weights = per_sample_weights.reshape(
            token_count,
            reads_per_token,
            group_size,
        ).to(weight.dtype)
        grad_output = grad_output.to(weight.dtype)
        grouped_result = _CoalescedSparseEmbeddingBag._grouped_mm_backward(
            grouped_indices,
            grouped_weights,
            grad_output,
            weight,
        )
        if grouped_result is not None:
            grad_weight, grad_per_sample = grouped_result
            return (
                None,
                grad_weight,
                grad_per_sample.reshape_as(per_sample_weights).to(ctx.per_sample_dtype),
                None,
            )

        group_ids = grouped_indices[..., 0].div(group_size, rounding_mode="floor")
        unique_groups = torch.unique(group_ids)
        grad_per_sample = torch.empty_like(grouped_weights)
        sparse_rows: list[Tensor] = []
        sparse_values: list[Tensor] = []
        slot_offsets = torch.arange(group_size, device=indices.device)
        for group_id in unique_groups:
            occurrences = (group_ids == group_id).nonzero(as_tuple=False)
            token_indices = occurrences[:, 0]
            read_indices = occurrences[:, 1]
            occurrence_weights = grouped_weights[token_indices, read_indices]
            occurrence_grad = grad_output.index_select(0, token_indices)
            sparse_rows.append(group_id * group_size + slot_offsets)
            sparse_values.append(occurrence_weights.transpose(0, 1) @ occurrence_grad)
            bank_group = torch.nn.functional.embedding(
                group_id * group_size + slot_offsets,
                weight,
            )
            grad_per_sample[token_indices, read_indices] = occurrence_grad @ bank_group.T
        row_indices = torch.cat(sparse_rows)
        row_values = torch.cat(sparse_values)
        grad_weight = torch.sparse_coo_tensor(
            row_indices.unsqueeze(0),
            row_values,
            weight.shape,
            device=weight.device,
            dtype=weight.dtype,
            check_invariants=False,
            is_coalesced=True,
        )
        return (
            None,
            grad_weight,
            grad_per_sample.reshape_as(per_sample_weights).to(ctx.per_sample_dtype),
            None,
        )


class ARTIPhaseMixer(nn.Module):
    """Mix cone-like latent receptors according to hidden state and coordinates."""

    def __init__(
        self, hidden_dim: int, coord_dim: int, operator_count: int, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.operator_count = operator_count
        self.hidden_dim = hidden_dim
        self.operators = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                )
                for _ in range(operator_count)
            ]
        )
        self.router = nn.Linear(hidden_dim + coord_dim, operator_count)
        self.coord_receptors = (
            nn.Linear(coord_dim, operator_count * hidden_dim) if coord_dim > 0 else None
        )

    def forward(self, z: Tensor, coord: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        route_input = torch.cat([z, coord], dim=-1)
        weights = torch.softmax(self.router(route_input), dim=-1)
        if self.coord_receptors is None:
            receptor_gain = torch.ones(
                *z.shape[:2], self.operator_count, self.hidden_dim, device=z.device, dtype=z.dtype
            )
        else:
            receptor_gain = 1.0 + torch.tanh(self.coord_receptors(coord)).view(
                *z.shape[:2], self.operator_count, self.hidden_dim
            )
        candidates = torch.stack(
            [op(z * receptor_gain[:, :, index, :]) for index, op in enumerate(self.operators)],
            dim=-2,
        )
        mixed = (candidates * weights.unsqueeze(-1)).sum(dim=-2)
        return mixed, weights, receptor_gain


class ARTIVirtualInterfaceMixer(nn.Module):
    """Fixed-size virtual interface for scalable token synchronization."""

    def __init__(
        self,
        hidden_dim: int,
        slots: int,
        *,
        recognition_mode: str = "explicit",
        recognition_threshold: float = 0.5,
        recognition_temperature: float = 0.1,
    ) -> None:
        super().__init__()
        self.slots = nn.Parameter(torch.randn(slots, hidden_dim) * hidden_dim**-0.5)
        self.read = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.write = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.out = nn.Linear(hidden_dim, hidden_dim)
        self.scale = hidden_dim**-0.5

    def forward(
        self, z: Tensor, mask: Tensor, visibility: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        read_logits = torch.einsum("bnd,sd->bns", self.read(z), self.slots) * self.scale
        slot_read_weights = masked_softmax(
            read_logits, mask.unsqueeze(-1).expand_as(read_logits), dim=1
        )

        written = self.write(z)
        if visibility is None:
            interface_state = torch.einsum("bns,bnd->bsd", slot_read_weights, written)
            read_weights = slot_read_weights
        else:
            pair_weights = slot_read_weights.unsqueeze(1) * visibility.unsqueeze(-1).to(z.dtype)
            normalizer = pair_weights.sum(dim=2, keepdim=True).clamp_min(torch.finfo(z.dtype).eps)
            pair_weights = pair_weights / normalizer
            interface_state = torch.einsum("bnms,bmd->bnsd", pair_weights, written)
            read_weights = pair_weights.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(
                1
            ).unsqueeze(-1).to(z.dtype)

        write_logits = (
            torch.einsum("bnd,bnsd->bns", z, interface_state) * self.scale
            if interface_state.ndim == 4
            else torch.einsum("bnd,bsd->bns", z, interface_state) * self.scale
        )
        write_weights = torch.softmax(write_logits, dim=-1)
        context = (
            torch.einsum("bns,bnsd->bnd", write_weights, interface_state)
            if interface_state.ndim == 4
            else torch.einsum("bns,bsd->bnd", write_weights, interface_state)
        )
        return self.out(context), read_weights, write_weights


class ARTILatentRecallField(nn.Module):
    """Private latent recall slots used as low-channel internal condition."""

    def __init__(
        self,
        hidden_dim: int,
        slots: int,
        *,
        recognition_mode: str = "none",
        recognition_threshold: float = 0.5,
        recognition_temperature: float = 0.1,
        routing: str = "dense",
        key_dim: int = 32,
        query_mode: str = "fixed",
        query_seed: int = 0,
        group_size: int = 16,
        group_topk: int = 2,
        value_composition: str = "single",
        formula: nn.Module | None = None,
        factor_activation: str = "none",
        route_exploration: float = 0.0,
        project_external: bool = True,
    ) -> None:
        super().__init__()
        if routing not in {"dense", "grouped"}:
            raise ValueError("routing must be 'dense' or 'grouped'")
        if key_dim <= 0 or group_size <= 0 or group_topk <= 0:
            raise ValueError("key_dim, group_size, and group_topk must be positive")
        if query_mode not in {"fixed", "legacy_learned"}:
            raise ValueError("query_mode must be 'fixed' or 'legacy_learned'")
        if isinstance(query_seed, bool) or not isinstance(query_seed, int):
            raise TypeError("query_seed must be an integer")
        if not 0 <= query_seed < 2**63:
            raise ValueError("query_seed must be in [0, 2**63)")
        if not math.isfinite(route_exploration) or route_exploration < 0:
            raise ValueError("route_exploration must be finite and non-negative")
        if value_composition not in {"single", "product", "state"}:
            raise ValueError("value_composition must be 'single', 'product', or 'state'")
        if formula is not None and value_composition != "single":
            raise ValueError(
                "custom Recall formula cannot be combined with legacy value_composition"
            )
        if factor_activation not in {"half", "none"}:
            raise ValueError("factor_activation must be 'half' or 'none'")
        formula_contract: RecallFormulaContract | None = None
        if formula is None:
            composition_factor = {
                "single": 1,
                "product": 2,
                "state": STATE_RECALL_COMPOSITION_FACTOR,
            }[value_composition]
        else:
            reference = next(formula.parameters(), None)
            if reference is None:
                reference = next(formula.buffers(), None)
            probe_kwargs = {}
            if reference is not None:
                probe_kwargs["device"] = reference.device
                if reference.is_floating_point():
                    probe_kwargs["dtype"] = reference.dtype
            probe = torch.zeros(hidden_dim, **probe_kwargs)
            formula_contract = check_recall_formula(formula, probe)
            composition_factor = formula_contract.factor_count
        composition_name = (
            "custom Recall formula" if formula is not None else f"{value_composition} Recall"
        )
        if slots % composition_factor:
            raise ValueError(f"{composition_name} requires slots divisible by {composition_factor}")
        if routing == "grouped":
            if slots % group_size:
                raise ValueError("grouped slots must be divisible by group_size")
            if group_topk > slots // group_size:
                raise ValueError("group_topk must not exceed the number of groups")
            if composition_factor > 1:
                groups = slots // group_size
                if groups % composition_factor:
                    raise ValueError(
                        f"{composition_name} requires groups divisible by {composition_factor}"
                    )
                if group_topk > groups // composition_factor:
                    raise ValueError("group_topk must fit within each composed Recall bank")
        self.hidden_dim = int(hidden_dim)
        self.slots = int(slots)
        self.routing = routing
        self.key_dim = int(key_dim if routing == "grouped" else hidden_dim)
        self.query_mode = query_mode
        self.query_seed = query_seed
        self.group_size = int(group_size)
        self.group_topk = int(group_topk)
        self.value_composition = "custom" if formula is not None else value_composition
        self.composition_factor = composition_factor
        self.formula = formula
        self.formula_contract = formula_contract
        if formula_contract is None:
            self.factor_names = {
                "single": ("content",),
                "product": ("scale", "shift"),
                "state": (
                    "coarse_content",
                    "fine_content",
                    *tuple(
                        f"modulation_{index:02d}"
                        for index in range(STATE_RECALL_MODULATION_FACTORS)
                    ),
                    "direction",
                    "opacity",
                ),
            }[value_composition]
        else:
            self.factor_names = formula_contract.factor_names
        if formula_contract is None:
            self.factor_route_names = self.factor_names
        else:
            self.factor_route_names = tuple(factor.route for factor in formula_contract.factors)
        route_order = tuple(dict.fromkeys(self.factor_route_names))
        self.route_names = route_order
        route_indices = {name: index for index, name in enumerate(route_order)}
        self.factor_route_indices = tuple(route_indices[name] for name in self.factor_route_names)
        self.factor_route_count = len(route_order)
        route_assignment = torch.zeros(
            self.factor_route_count,
            composition_factor,
        )
        for factor_index, route_index in enumerate(self.factor_route_indices):
            route_assignment[route_index, factor_index] = 1.0
        route_assignment /= route_assignment.sum(dim=-1, keepdim=True)
        self.register_buffer(
            "_factor_route_assignment",
            route_assignment,
            persistent=False,
        )
        self.register_buffer(
            "_factor_route_index",
            torch.tensor(self.factor_route_indices, dtype=torch.long),
            persistent=False,
        )
        slots_per_factor = slots // composition_factor
        self.factor_slices = tuple(
            (index * slots_per_factor, (index + 1) * slots_per_factor)
            for index in range(composition_factor)
        )
        self.factor_activation = Half() if factor_activation == "half" else nn.Identity()
        self.route_exploration = float(route_exploration)
        self._bank_gradient_enabled = True
        self._training_group_partitions: tuple[int, tuple[int, ...]] | None = None
        self.register_buffer(
            "_route_log_prior",
            torch.empty(0),
            persistent=False,
        )
        self.register_buffer(
            "_route_influence",
            torch.empty(0),
            persistent=False,
        )
        self._expert_names: tuple[str, ...] = ()
        self._expert_route_ranges: tuple[tuple[int, int], ...] = ()
        self._expert_weights: tuple[float, ...] = ()
        self._expert_influences: tuple[float, ...] = ()
        # Recall values are host-dimensional writes. The caller applies them
        # directly to the current state instead of exposing a residual branch.
        self.bank = nn.Parameter(torch.randn(slots, hidden_dim) * hidden_dim**-0.5)
        if formula_contract is not None:
            with torch.no_grad():
                for factor, (start, stop) in zip(
                    formula_contract.factors,
                    self.factor_slices,
                    strict=True,
                ):
                    factor_bank = self.bank[start:stop]
                    if factor.init == "zero":
                        factor_bank.fill_(factor.identity)
                    else:
                        scale = hidden_dim**-0.5 if factor.init_scale is None else factor.init_scale
                        factor_bank.normal_(mean=factor.identity, std=scale)
        self.query = nn.Linear(hidden_dim, self.key_dim, bias=False)
        if self.query_mode == "fixed":
            self.query._arti_fixed_query = True
            self.reset_query()
        if routing == "grouped":
            groups = slots // group_size
            self.key_bank = nn.Parameter(torch.randn(slots, self.key_dim) * self.key_dim**-0.5)
            self.group_bank = nn.Parameter(torch.randn(groups, self.key_dim) * self.key_dim**-0.5)
        else:
            self.register_parameter("key_bank", None)
            self.register_parameter("group_bank", None)
        self.external = (
            nn.Linear(hidden_dim, hidden_dim, bias=False) if project_external else nn.Identity()
        )
        if recognition_mode not in {"explicit", "alignment", "none"}:
            raise ValueError("recognition_mode must be 'explicit', 'alignment', or 'none'")
        self.recognition_mode = recognition_mode
        self.register_buffer(
            "recognition_threshold", torch.tensor(float(recognition_threshold)), persistent=False
        )
        self.register_buffer(
            "recognition_temperature",
            torch.tensor(float(recognition_temperature)),
            persistent=False,
        )
        self.alignment_recognizer = (
            nn.Linear(hidden_dim * 2, 1) if recognition_mode == "alignment" else None
        )
        self.scale = self.key_dim**-0.5

    @property
    def query_contract(self) -> dict[str, int | str]:
        """Return the stable coordinate contract used by Recall routing."""

        return {
            "mode": self.query_mode,
            "algorithm": (
                _FIXED_QUERY_ALGORITHM if self.query_mode == "fixed" else "legacy-learned"
            ),
            "seed": self.query_seed,
            "hidden_dim": self.hidden_dim,
            "key_dim": self.key_dim,
        }

    def recall_parameter_tags(self) -> tuple[RecallParameterTag, ...]:
        """Return authoritative ownership tags for every Recall parameter."""

        from .recall_policy import RecallParameterTag, validate_parameter_tags

        tags = []
        for name, _parameter in self.named_parameters():
            if name == "bank":
                role = "value_bank"
                storage_group = "formula_bank"
            elif name == "query.weight":
                role = "fixed_query" if self.query_mode == "fixed" else "query"
                storage_group = "query"
            elif name in {"key_bank", "group_bank"}:
                role = "routing"
                storage_group = "routing"
            elif name.startswith("formula."):
                role = "formula"
                storage_group = "formula"
            elif name.startswith("alignment_recognizer."):
                role = "recognition"
                storage_group = "recognition"
            elif name.startswith("external."):
                role = "external_projection"
                storage_group = "external"
            else:
                role = "frozen_state"
                storage_group = "module"
            tags.append(
                RecallParameterTag(
                    name,
                    role=role,
                    storage_group=storage_group,
                )
            )
        return validate_parameter_tags(
            tags,
            parameter_names=dict(self.named_parameters()),
        )

    def reset_query(self, *, seed: int | None = None) -> None:
        """Reset Recall to its deterministic fixed query coordinate system."""

        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise TypeError("query seed must be an integer")
            if not 0 <= seed < 2**63:
                raise ValueError("query seed must be in [0, 2**63)")
            self.query_seed = seed
        self.query_mode = "fixed"
        self.query._arti_fixed_query = True
        with torch.no_grad():
            self.query.weight.copy_(
                _fixed_query_weight(
                    self.hidden_dim,
                    self.key_dim,
                    self.query_seed,
                ).to(self.query.weight)
            )
        self.query.weight.requires_grad_(False)

    def set_bank_gradient_enabled(self, enabled: bool) -> None:
        """Enable or detach bank gradients without changing the forward value."""

        self._bank_gradient_enabled = bool(enabled)

    @property
    def expert_names(self) -> tuple[str, ...]:
        """Return runtime expert names configured by a bank assembly."""

        return self._expert_names

    @property
    def expert_weights(self) -> tuple[float, ...]:
        """Return the current non-negative routing prior for each expert."""

        return self._expert_weights

    @property
    def expert_influences(self) -> tuple[float, ...]:
        """Return signed multipliers applied to routed expert writes."""

        return self._expert_influences

    def configure_expert_routes(
        self,
        names: Sequence[str],
        ranges: Sequence[tuple[int, int]],
    ) -> None:
        """Bind named experts to contiguous positions in one routing axis."""

        resolved_names = tuple(str(name) for name in names)
        resolved_ranges = tuple((int(start), int(stop)) for start, stop in ranges)
        if not resolved_names or len(resolved_names) != len(resolved_ranges):
            raise ValueError("expert names and ranges must have the same non-zero length")
        if any(not name.strip() for name in resolved_names):
            raise ValueError("expert names must not be empty")
        if len(set(resolved_names)) != len(resolved_names):
            raise ValueError("expert names must be unique")
        route_width = self._route_width()
        cursor = 0
        for start, stop in resolved_ranges:
            if start != cursor or stop <= start or stop > route_width:
                raise ValueError(
                    "expert route ranges must form a contiguous partition of the routing axis"
                )
            cursor = stop
        if cursor != route_width:
            raise ValueError("expert route ranges must cover the routing axis")
        self._expert_names = resolved_names
        self._expert_route_ranges = resolved_ranges
        self.set_expert_weights((1.0,) * len(resolved_names))
        self.set_expert_influences((1.0,) * len(resolved_names))

    def set_expert_weights(
        self,
        weights: Sequence[float] | Mapping[str, float],
    ) -> None:
        """Set multiplicative routing priors without scaling recalled values."""

        if not self._expert_names:
            raise RuntimeError("Recall field has no configured expert assembly")
        if isinstance(weights, Mapping):
            unknown = set(weights) - set(self._expert_names)
            missing = set(self._expert_names) - set(weights)
            if unknown or missing:
                raise ValueError("expert weight mapping must exactly match configured names")
            resolved = tuple(float(weights[name]) for name in self._expert_names)
        else:
            resolved = tuple(float(weight) for weight in weights)
            if len(resolved) != len(self._expert_names):
                raise ValueError(
                    f"expected {len(self._expert_names)} expert weights, got {len(resolved)}"
                )
        if any(not math.isfinite(weight) or weight < 0.0 for weight in resolved):
            raise ValueError("expert weights must be finite and non-negative")

        route_prior = torch.zeros(self._route_width(), dtype=torch.float32)
        for weight, (start, stop) in zip(
            resolved,
            self._expert_route_ranges,
            strict=True,
        ):
            route_prior[start:stop] = weight
        required = self.group_topk if self.routing == "grouped" else 1
        if int(torch.count_nonzero(route_prior)) < required:
            raise ValueError(
                f"expert weights must leave at least {required} routing positions enabled"
            )
        self._expert_weights = resolved
        if all(weight == 1.0 for weight in resolved):
            self._route_log_prior = torch.empty(
                0,
                device=self._route_log_prior.device,
            )
            return
        log_prior = torch.full_like(route_prior, -torch.inf)
        positive = route_prior > 0
        log_prior[positive] = torch.log(route_prior[positive])
        self._route_log_prior = log_prior.to(device=self._route_log_prior.device)

    def set_expert_influences(
        self,
        influences: Sequence[float] | Mapping[str, float],
    ) -> None:
        """Set signed expert-write multipliers without changing routing."""

        if not self._expert_names:
            raise RuntimeError("Recall field has no configured expert assembly")
        if isinstance(influences, Mapping):
            unknown = set(influences) - set(self._expert_names)
            missing = set(self._expert_names) - set(influences)
            if unknown or missing:
                raise ValueError("expert influence mapping must exactly match configured names")
            resolved = tuple(float(influences[name]) for name in self._expert_names)
        else:
            resolved = tuple(float(influence) for influence in influences)
            if len(resolved) != len(self._expert_names):
                raise ValueError(
                    f"expected {len(self._expert_names)} expert influences, got {len(resolved)}"
                )
        if any(not math.isfinite(influence) for influence in resolved):
            raise ValueError("expert influences must be finite")

        route_influence = torch.ones(self._route_width(), dtype=torch.float32)
        for influence, (start, stop) in zip(
            resolved,
            self._expert_route_ranges,
            strict=True,
        ):
            route_influence[start:stop] = influence
        self._expert_influences = resolved
        if all(influence == 1.0 for influence in resolved):
            self._route_influence = torch.empty(
                0,
                device=self._route_influence.device,
            )
            return
        self._route_influence = route_influence.to(device=self._route_influence.device)

    def _route_width(self) -> int:
        if self.routing == "grouped":
            assert self.group_bank is not None
            width = int(self.group_bank.shape[0])
        else:
            width = self.slots
        if width % self.composition_factor:
            raise RuntimeError("Recall routing width is not factor-aligned")
        return width // self.composition_factor

    def _apply_route_prior(
        self,
        logits: Tensor,
        *,
        active_indices: Tensor | None = None,
        offset: int = 0,
        count: int | None = None,
    ) -> Tensor:
        if self._route_log_prior.numel() == 0:
            return logits
        prior = self._route_log_prior
        if offset or count is not None:
            stop = prior.numel() if count is None else offset + count
            prior = prior[offset:stop]
        if active_indices is not None:
            prior = prior.index_select(0, active_indices)
        if prior.numel() != logits.shape[-1]:
            raise RuntimeError("Recall route prior is stale for the current bank assembly")
        return logits + prior.to(device=logits.device, dtype=logits.dtype)

    def _routed_influence(self, route: Tensor) -> Tensor:
        """Resolve one signed write multiplier from per-expert route mass."""

        if self._route_influence.numel() == 0:
            return route.new_ones(route.shape[:-1])
        route_width = self._route_width()
        if route.shape[-1] == 0 or route.shape[-1] % route_width:
            raise RuntimeError("Recall route is unavailable for signed expert influence")
        factor_count = route.shape[-1] // route_width
        factor_route = route.reshape(*route.shape[:-1], factor_count, route_width)
        influence = self._route_influence.to(device=route.device, dtype=route.dtype)
        return torch.sum(factor_route * influence, dim=-1).mean(dim=-1)

    def set_training_group_partitions(
        self,
        partition_count: int,
        active_partitions: Sequence[int],
    ) -> None:
        """Restrict automatic grouped routing to selected training partitions."""

        if self.routing != "grouped":
            raise ValueError("training group partitions require grouped Recall routing")
        if (
            isinstance(partition_count, bool)
            or not isinstance(partition_count, int)
            or partition_count <= 0
        ):
            raise ValueError("partition_count must be a positive integer")
        active = tuple(int(value) for value in active_partitions)
        if not active or len(active) != len(set(active)):
            raise ValueError("active_partitions must contain unique partition indices")
        if any(value < 0 or value >= partition_count for value in active):
            raise ValueError("active_partitions contains an out-of-range partition index")
        assert self.group_bank is not None
        available_groups = self.group_bank.shape[0]
        available_groups //= self.composition_factor
        if partition_count > available_groups:
            raise ValueError("partition_count must not exceed Recall groups per value bank")
        self._training_group_partitions = (partition_count, active)

    def clear_training_group_partitions(self) -> None:
        """Restore unrestricted grouped routing."""

        self._training_group_partitions = None

    def forward(
        self,
        z: Tensor,
        mask: Tensor,
        recall: Tensor | None = None,
        selected_groups: Tensor | None = None,
        route_assignment: Tensor | None = None,
        memory: Tensor | None = None,
        _selected_groups_normalized: bool = False,
    ) -> _ARTIRecallRead:
        if recall is not None and (
            recall.ndim != 3 or recall.shape[0] != z.shape[0] or recall.shape[2] != z.shape[2]
        ):
            raise ValueError("recall must have shape [B, K, H]")
        if selected_groups is not None and self.routing != "grouped":
            raise ValueError("selected_groups is only supported by grouped Recall routing")

        context, weights, indices, route = self._read_bank(
            z,
            recall=recall,
            selected_groups=selected_groups,
            route_assignment=route_assignment,
            memory=memory,
            selected_groups_normalized=_selected_groups_normalized,
            return_route=True,
        )

        if self.recognition_mode == "explicit":
            similarity = torch.cosine_similarity(z, context, dim=-1, eps=1e-6)
            temperature = self.recognition_temperature.to(z).clamp_min(torch.finfo(z.dtype).eps)
            recognition = torch.sigmoid(
                (similarity - self.recognition_threshold.to(z)) / temperature
            )
        elif self.recognition_mode == "alignment":
            assert self.alignment_recognizer is not None
            query = self._project_query(z)
            recognition_query = query if query.shape[-1] == context.shape[-1] else z
            recognition = torch.sigmoid(
                self.alignment_recognizer(torch.cat([recognition_query, context], dim=-1))
            ).squeeze(-1)
        else:
            recognition = torch.ones(
                z.shape[:2],
                device=z.device,
                dtype=z.dtype,
            )
        recognition = recognition * mask.to(z.dtype)
        routed_influence = self._routed_influence(route)
        influence = (recognition * routed_influence).unsqueeze(-1).expand_as(context)
        return _ARTIRecallRead(
            context=influence * context,
            weights=weights,
            influence=influence,
            recognition=recognition,
            indices=indices,
            route=route,
        )

    def read_context(
        self,
        z: Tensor,
        mask: Tensor | None = None,
        recall: Tensor | None = None,
        selected_groups: Tensor | None = None,
        route_assignment: Tensor | None = None,
        memory: Tensor | None = None,
        _selected_groups_normalized: bool = False,
    ) -> Tensor:
        """Return only the recalled value for state-write hot paths."""

        if recall is not None and (
            recall.ndim != 3 or recall.shape[0] != z.shape[0] or recall.shape[2] != z.shape[2]
        ):
            raise ValueError("recall must have shape [B, K, H]")
        if selected_groups is not None and self.routing != "grouped":
            raise ValueError("selected_groups is only supported by grouped Recall routing")

        context, _weights, _indices, route = self._read_bank(
            z,
            recall=recall,
            selected_groups=selected_groups,
            route_assignment=route_assignment,
            memory=memory,
            selected_groups_normalized=_selected_groups_normalized,
            return_route=self._route_influence.numel() > 0,
        )

        routed_influence = self._routed_influence(route)
        if self.recognition_mode == "none":
            context = context * routed_influence.unsqueeze(-1)
            return context if mask is None else context * mask.unsqueeze(-1).to(context.dtype)
        if self.recognition_mode == "explicit":
            similarity = torch.cosine_similarity(z, context, dim=-1, eps=1e-6)
            temperature = self.recognition_temperature.to(z).clamp_min(torch.finfo(z.dtype).eps)
            recognition = torch.sigmoid(
                (similarity - self.recognition_threshold.to(z)) / temperature
            )
        else:
            assert self.alignment_recognizer is not None
            query = self._project_query(z)
            recognition_query = query if query.shape[-1] == context.shape[-1] else z
            recognition = torch.sigmoid(
                self.alignment_recognizer(torch.cat([recognition_query, context], dim=-1))
            ).squeeze(-1)
        if mask is not None:
            recognition = recognition * mask.to(recognition.dtype)
        return (recognition * routed_influence).unsqueeze(-1) * context

    def _read_bank(
        self,
        z: Tensor,
        *,
        recall: Tensor | None,
        selected_groups: Tensor | None,
        route_assignment: Tensor | None,
        memory: Tensor | None,
        selected_groups_normalized: bool,
        return_route: bool,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if self.value_composition != "single" and selected_groups is not None:
            raise ValueError("selected_groups is not supported by composed Recall")
        if route_assignment is not None and self.value_composition != "custom":
            raise ValueError(
                "route_assignment requires a Recall Formula with explicit factor routes"
            )
        if memory is not None:
            if self.routing == "grouped" and self.group_size != 1:
                raise ValueError(
                    "explicit memory with grouped Recall currently requires group_size=1"
                )
            if not isinstance(memory, Tensor) or not memory.is_floating_point():
                raise TypeError("memory must be a floating-point Tensor")
            if memory.ndim not in {2, 3}:
                raise ValueError("memory must have shape [S, D] or [B, S, D]")
            if memory.shape[-2:] != (self.slots, self.hidden_dim):
                raise ValueError(
                    "memory trailing shape must be "
                    f"({self.slots}, {self.hidden_dim}), got {tuple(memory.shape[-2:])}"
                )
            if memory.ndim == 3 and memory.shape[0] != z.shape[0]:
                raise ValueError("batched memory must match the input batch size")
            if memory.device != z.device:
                raise ValueError("memory must be on the input device")

        if self.routing == "grouped":
            if self.value_composition == "single":
                context, weights, indices, route = self._grouped_read(
                    z,
                    selected_groups=selected_groups,
                    memory=memory,
                    selected_groups_normalized=selected_groups_normalized,
                    return_route=return_route,
                )
            else:
                factors, weights, indices, route = self._grouped_factor_read(
                    z,
                    factor_count=self.composition_factor,
                    memory=memory,
                    return_route=return_route,
                    compose_state=self.value_composition == "state",
                )
                if self.value_composition == "product":
                    context = self._compose_product_write(
                        z,
                        factors[..., 0, :],
                        factors[..., 1, :],
                    )
                elif self.value_composition == "custom":
                    context = self._compose_custom_write(
                        z,
                        self._assign_factor_route_gradients(
                            z,
                            factors,
                            route_assignment,
                        ),
                    )
                else:
                    context = factors
        else:
            bank = memory
            if bank is None:
                bank = self.bank if self._bank_gradient_enabled else self.bank.detach()
            if not torch.is_autocast_enabled(z.device.type):
                bank = bank.to(dtype=z.dtype)
            query = self._project_query(z)
            if self.value_composition == "single":
                internal_logits = (
                    torch.einsum("bnd,kd->bnk", query, bank)
                    if bank.ndim == 2
                    else torch.einsum("bnd,bkd->bnk", query, bank)
                ) * self.scale
                internal_logits = self._apply_route_prior(internal_logits)
                if recall is None:
                    logits = internal_logits
                    weights = torch.softmax(logits, dim=-1)
                    context = (
                        torch.einsum("bnk,kd->bnd", weights, bank)
                        if bank.ndim == 2
                        else torch.einsum("bnk,bkd->bnd", weights, bank)
                    )
                else:
                    internal_bank = (
                        bank.unsqueeze(0).expand(z.shape[0], -1, -1)
                        if bank.ndim == 2
                        else bank
                    )
                    routed_bank = torch.cat(
                        [internal_bank, recall.to(z)],
                        dim=1,
                    )
                    external_logits = torch.einsum("bnd,bkd->bnk", query, recall.to(z)) * self.scale
                    logits = torch.cat((internal_logits, external_logits), dim=-1)
                    weights = torch.softmax(logits, dim=-1)
                    context = torch.einsum("bnk,bkd->bnd", weights, routed_bank)
            else:
                factor_banks = tuple(
                    bank[start:stop] if bank.ndim == 2 else bank[:, start:stop]
                    for start, stop in self.factor_slices
                )
                factor_logits = torch.stack(
                    [
                        (
                            torch.einsum("bnd,kd->bnk", query, factor_bank)
                            if factor_bank.ndim == 2
                            else torch.einsum("bnd,bkd->bnk", query, factor_bank)
                        )
                        * self.scale
                        for factor_bank in factor_banks
                    ],
                    dim=-2,
                )
                factor_logits = self._share_factor_route_logits(factor_logits)
                factor_logits = self._apply_route_prior(factor_logits)
                factor_weights_tensor = torch.softmax(factor_logits, dim=-1)
                factors = torch.stack(
                    [
                        torch.einsum("bnk,kd->bnd", weights, factor_bank)
                        if factor_bank.ndim == 2
                        else torch.einsum("bnk,bkd->bnd", weights, factor_bank)
                        for weights, factor_bank in zip(
                            factor_weights_tensor.unbind(dim=-2),
                            factor_banks,
                            strict=True,
                        )
                    ],
                    dim=-2,
                )
                if self.value_composition == "product":
                    context = self._compose_product_write(
                        z,
                        factors[..., 0, :],
                        factors[..., 1, :],
                    )
                elif self.value_composition == "custom":
                    context = self._compose_custom_write(
                        z,
                        self._assign_factor_route_gradients(
                            z,
                            factors,
                            route_assignment,
                        ),
                    )
                else:
                    context = self._compose_state_write(z, factors)
                weights = factor_weights_tensor.flatten(-2)
            indices = torch.empty(*weights.shape, 0, device=z.device, dtype=torch.long)
            route = weights

        if recall is not None:
            external_context = self.external(recall.to(z).mean(dim=1, keepdim=True)).expand_as(
                context
            )
            context = context + external_context
        return context, weights, indices, route

    @staticmethod
    def _compose_product_write(z: Tensor, scale: Tensor, shift: Tensor) -> Tensor:
        """Return a bounded affine write from two independently recalled values."""

        gain = 1.0 + torch.tanh(scale)
        return gain * (z + shift) - z

    def _compose_custom_write(self, z: Tensor, factors: Tensor) -> Tensor:
        """Apply a custom next-state formula and return its write tensor."""

        if self.formula is None:
            raise RuntimeError("custom Recall composition requires a formula module")
        flat_state = z.reshape(-1, z.shape[-1])
        flat_factors = factors.reshape(
            -1,
            self.composition_factor,
            factors.shape[-1],
        )
        next_state = torch.vmap(self.formula)(flat_state, flat_factors).reshape_as(z)
        if not isinstance(next_state, Tensor):
            raise TypeError("Recall formula must return one next-state Tensor")
        if next_state.shape != z.shape:
            raise ValueError(
                "Recall formula next-state shape must match the host state; "
                f"expected {tuple(z.shape)}, got {tuple(next_state.shape)}"
            )
        if next_state.device != z.device:
            raise ValueError("Recall formula next state must remain on the host device")
        if next_state.dtype != z.dtype:
            raise ValueError("Recall formula next state must preserve the host dtype")
        return next_state - z

    def _assign_factor_route_gradients(
        self,
        z: Tensor,
        factors: Tensor,
        route_assignment: Tensor | None,
    ) -> Tensor:
        """Route Formula-factor gradients without changing their forward values."""

        if route_assignment is None:
            return factors
        if not isinstance(route_assignment, Tensor) or not route_assignment.is_floating_point():
            raise TypeError("route_assignment must be a floating-point Tensor")
        if route_assignment.ndim != 2:
            raise ValueError("route_assignment must have shape [B, R]")
        expected = (z.shape[0], self.factor_route_count)
        if route_assignment.shape != expected:
            raise ValueError(
                f"route_assignment must have shape {expected}, got {tuple(route_assignment.shape)}"
            )
        assignment = route_assignment.to(device=z.device, dtype=z.dtype)
        factor_assignment = (
            assignment.index_select(
                -1,
                self._factor_route_index.to(device=assignment.device),
            )
            .unsqueeze(1)
            .unsqueeze(-1)
        )
        return factors.detach() + factor_assignment * (factors - factors.detach())

    @staticmethod
    def _compose_state_content(coarse: Tensor, fine: Tensor) -> Tensor:
        """Combine a coarse value with an independently routed relative fraction."""

        unit = coarse.detach().abs().clamp_min(1.0)
        return coarse + unit * torch.tanh(fine)

    @staticmethod
    def _compose_state_write(z: Tensor, factors: Tensor) -> Tensor:
        """Compose a constrained polynomial state from independently routed factors."""

        coarse_content = factors[..., 0, :]
        content = ARTILatentRecallField._compose_state_content(
            coarse_content,
            factors[..., 1, :],
        )
        modulation = (
            1.0
            + torch.tanh(
                factors[
                    ...,
                    STATE_RECALL_CONTENT_FACTORS : (
                        STATE_RECALL_CONTENT_FACTORS + STATE_RECALL_MODULATION_FACTORS
                    ),
                    :,
                ]
            )
            / STATE_RECALL_MODULATION_FACTORS
        )
        recalled_content = content * modulation.prod(dim=-2)
        write_direction = torch.tanh(factors[..., -2, :])
        memory_opacity = torch.tanh(factors[..., -1, :]).square()
        host_weight = 1.0 - memory_opacity
        memory_weight = memory_opacity * write_direction
        return host_weight * z + memory_weight * recalled_content

    def _project_query(self, z: Tensor) -> Tensor:
        weight = self.query.weight if self._bank_gradient_enabled else self.query.weight.detach()
        if not torch.is_autocast_enabled(z.device.type):
            weight = weight.to(dtype=z.dtype)
        return torch.nn.functional.linear(z, weight)

    def _grouped_factor_read(
        self,
        z: Tensor,
        *,
        factor_count: int,
        memory: Tensor | None,
        return_route: bool,
        compose_state: bool = False,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Read independently routed factors, optionally composing state online."""

        assert self.key_bank is not None
        assert self.group_bank is not None
        query = self._project_query(z)
        autocast_enabled = torch.is_autocast_enabled(z.device.type)
        value_bank_parameter = self.bank if self._bank_gradient_enabled else self.bank.detach()
        group_bank_parameter = (
            self.group_bank if self._bank_gradient_enabled else self.group_bank.detach()
        )
        groups = group_bank_parameter.shape[0]
        factor_groups = groups // factor_count
        active_groups = self._active_training_groups(factor_groups, z.device)
        if active_groups is None:
            group_bank = (
                group_bank_parameter
                if autocast_enabled
                else group_bank_parameter.to(dtype=query.dtype)
            ).reshape(factor_count, factor_groups, self.key_dim)
        else:
            factor_offsets = (
                torch.arange(factor_count, device=z.device).unsqueeze(-1) * factor_groups
            )
            active_global_groups = active_groups.unsqueeze(0) + factor_offsets
            group_bank = torch.nn.functional.embedding(
                active_global_groups,
                group_bank_parameter,
                sparse=True,
            )
            if not autocast_enabled:
                group_bank = group_bank.to(dtype=query.dtype)
        group_logits = torch.einsum("bnd,fgd->bnfg", query, group_bank) * self.scale
        group_logits = self._share_factor_route_logits(group_logits)
        group_logits = self._apply_route_prior(
            group_logits,
            active_indices=active_groups,
        )
        route = (
            torch.softmax(group_logits, dim=-1).flatten(-2)
            if return_route
            else group_logits.new_empty((*group_logits.shape[:2], 0))
        )

        if self.training and self._bank_gradient_enabled and self.route_exploration > 0:
            normalized_logits = self._standardize_route_logits(group_logits)
            uniform = torch.rand(
                normalized_logits.shape[0],
                1,
                self.factor_route_count,
                normalized_logits.shape[-1],
                device=normalized_logits.device,
                dtype=torch.float32,
            ).clamp(torch.finfo(torch.float32).eps, 1.0 - torch.finfo(torch.float32).eps)
            uniform = uniform.index_select(
                -2,
                self._factor_route_index.to(device=uniform.device),
            )
            gumbel = -torch.log(-torch.log(uniform))
            selection_logits = normalized_logits + self.route_exploration * gumbel
            selected_groups = torch.topk(
                selection_logits,
                self.group_topk,
                dim=-1,
            ).indices
            selected_logits = normalized_logits.gather(-1, selected_groups)
        elif self.group_topk == 1:
            selected_logits, selected_groups = torch.max(
                group_logits,
                dim=-1,
                keepdim=True,
            )
        else:
            selected_logits, selected_groups = torch.topk(
                group_logits,
                self.group_topk,
                dim=-1,
            )

        group_weights = self._selected_group_weights(
            group_logits,
            selected_groups,
            selected_logits,
        )
        if active_groups is not None:
            selected_groups = active_groups[selected_groups]
        factor_offsets = torch.arange(factor_count, device=z.device).view(
            1,
            1,
            factor_count,
            1,
        )
        global_selected_groups = selected_groups + factor_offsets * factor_groups
        if self.group_size == 1 and not compose_state:
            value_bank = memory
            if value_bank is None:
                value_bank = value_bank_parameter
            selected_values = self._select_explicit_values(
                value_bank,
                global_selected_groups,
                sparse=active_groups is not None,
            )
            factors = torch.sum(
                selected_values.to(dtype=z.dtype) * group_weights.to(dtype=z.dtype).unsqueeze(-1),
                dim=-2,
            )
            if return_route:
                diagnostic_weights = group_weights.unsqueeze(-1)
                diagnostic_indices = global_selected_groups.unsqueeze(-1)
            else:
                diagnostic_weights = group_weights.new_empty((*z.shape[:2], 0, 1))
                diagnostic_indices = global_selected_groups.new_empty((*z.shape[:2], 0, 1))
            return factors, diagnostic_weights, diagnostic_indices, route
        offsets = torch.arange(self.group_size, device=z.device)
        indices = global_selected_groups.unsqueeze(-1) * self.group_size + offsets
        key_bank = self.key_bank if self._bank_gradient_enabled else self.key_bank.detach()
        selected_keys = torch.nn.functional.embedding(
            indices,
            key_bank,
            sparse=active_groups is not None,
        )
        if not autocast_enabled:
            selected_keys = selected_keys.to(dtype=query.dtype)
        slot_logits = (
            torch.einsum(
                "bnd,bnftmd->bnftm",
                query,
                selected_keys,
            )
            * self.scale
        )
        slot_weights = torch.softmax(slot_logits, dim=-1)
        packed_weights = group_weights.to(slot_weights.dtype).unsqueeze(-1) * slot_weights
        value_bank = value_bank_parameter
        native_grouped_mm = active_groups is None and self._native_grouped_mm_available(
            packed_weights
        )

        def read_factor_range(start: int, stop: int) -> Tensor:
            factor_groups = global_selected_groups[:, :, start:stop]
            factor_weights = packed_weights[:, :, start:stop]
            if native_grouped_mm:
                return self._grouped_mm_value_read(
                    factor_groups,
                    factor_weights,
                    value_bank,
                    output_dtype=z.dtype,
                )
            factor_indices = indices[:, :, start:stop]
            return self._weighted_value_read(
                factor_indices,
                factor_weights,
                value_bank,
                output_dtype=z.dtype,
                sparse=active_groups is not None,
            )

        if compose_state:
            modulation_start = STATE_RECALL_CONTENT_FACTORS
            modulation_stop = modulation_start + STATE_RECALL_MODULATION_FACTORS
            chunk_size = 4
            first_stop = min(chunk_size, modulation_stop)
            first_chunk = read_factor_range(0, first_stop)
            content = self._compose_state_content(
                first_chunk[..., 0, :],
                first_chunk[..., 1, :],
            )
            polynomial = (
                1.0
                + torch.tanh(first_chunk[..., modulation_start:, :])
                / STATE_RECALL_MODULATION_FACTORS
            ).prod(dim=-2)
            for start in range(first_stop, modulation_stop, chunk_size):
                modulation = read_factor_range(
                    start,
                    min(start + chunk_size, modulation_stop),
                )
                polynomial = polynomial * (
                    1.0 + torch.tanh(modulation) / STATE_RECALL_MODULATION_FACTORS
                ).prod(dim=-2)
            controls = read_factor_range(modulation_stop, factor_count)
            write_direction = torch.tanh(controls[..., 0, :])
            memory_opacity = torch.tanh(controls[..., 1, :]).square()
            host_weight = 1.0 - memory_opacity
            memory_weight = memory_opacity * write_direction
            factors = host_weight * z + memory_weight * content * polynomial
        elif native_grouped_mm:
            factors = self._grouped_mm_value_read(
                global_selected_groups,
                packed_weights,
                value_bank,
                output_dtype=z.dtype,
            )
        elif active_groups is not None and factor_count > 4:
            factors = torch.cat(
                [
                    read_factor_range(start, min(start + 4, factor_count))
                    for start in range(0, factor_count, 4)
                ],
                dim=-2,
            )
        else:
            factors = self._weighted_value_read(
                indices,
                packed_weights,
                value_bank,
                output_dtype=z.dtype,
                sparse=active_groups is not None,
            )
        if return_route:
            diagnostic_weights = packed_weights.flatten(-3, -2)
            diagnostic_indices = indices.flatten(-3, -2)
        else:
            diagnostic_weights = packed_weights.new_empty((*z.shape[:2], 0, self.group_size))
            diagnostic_indices = global_selected_groups.new_empty(
                (*z.shape[:2], 0, self.group_size)
            )
        return factors, diagnostic_weights, diagnostic_indices, route

    def _share_factor_route_logits(self, logits: Tensor) -> Tensor:
        """Share one routing decision across factors with the same route name."""

        if self.factor_route_count == self.composition_factor:
            return logits
        unique_logits = torch.einsum(
            "...fg,rf->...rg",
            logits,
            self._factor_route_assignment.to(device=logits.device, dtype=logits.dtype),
        )
        return unique_logits.index_select(
            -2,
            self._factor_route_index.to(device=unique_logits.device),
        )

    @staticmethod
    def _selected_group_weights(
        group_logits: Tensor,
        selected_groups: Tensor,
        selected_logits: Tensor,
    ) -> Tensor:
        """Keep hard top-1 routing while preserving soft routing gradients."""

        if selected_groups.shape[-1] != 1:
            return torch.softmax(selected_logits, dim=-1)
        probability = torch.softmax(group_logits, dim=-1).gather(
            -1,
            selected_groups,
        )
        return torch.ones_like(probability) + probability - probability.detach()

    def _grouped_read(
        self,
        z: Tensor,
        *,
        selected_groups: Tensor | None = None,
        memory: Tensor | None = None,
        selected_groups_normalized: bool = False,
        return_route: bool = True,
        group_offset: int = 0,
        group_count: int | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        assert self.key_bank is not None
        assert self.group_bank is not None
        query = self._project_query(z)
        autocast_enabled = torch.is_autocast_enabled(z.device.type)
        group_bank_parameter = (
            self.group_bank if self._bank_gradient_enabled else self.group_bank.detach()
        )
        total_groups = group_bank_parameter.shape[0]
        resolved_group_count = total_groups - group_offset if group_count is None else group_count
        if (
            group_offset < 0
            or resolved_group_count <= 0
            or group_offset + resolved_group_count > total_groups
        ):
            raise ValueError("invalid Recall group partition")
        active_groups = (
            None
            if selected_groups is not None
            else self._active_training_groups(resolved_group_count, z.device)
        )
        if active_groups is None:
            group_bank_parameter = group_bank_parameter[
                group_offset : group_offset + resolved_group_count
            ]
            group_bank = (
                group_bank_parameter
                if autocast_enabled
                else group_bank_parameter.to(dtype=query.dtype)
            )
        else:
            group_bank = torch.nn.functional.embedding(
                active_groups + group_offset,
                group_bank_parameter,
                sparse=True,
            )
            if not autocast_enabled:
                group_bank = group_bank.to(dtype=query.dtype)
        group_logits = torch.einsum("bnd,gd->bng", query, group_bank) * self.scale
        group_logits = self._apply_route_prior(
            group_logits,
            active_indices=active_groups,
            offset=group_offset,
            count=resolved_group_count,
        )
        route = (
            torch.softmax(group_logits, dim=-1)
            if return_route
            else group_logits.new_empty((*group_logits.shape[:-1], 0))
        )
        if selected_groups is not None:
            expected_shape = (*z.shape[:2], self.group_topk)
            if selected_groups.shape != expected_shape:
                raise ValueError(
                    "selected_groups must have shape "
                    f"{expected_shape}, got {tuple(selected_groups.shape)}"
                )
            if selected_groups.dtype != torch.long:
                raise ValueError("selected_groups must use torch.long dtype")
            if selected_groups.device != z.device:
                raise ValueError("selected_groups must be on the same device as z")
            is_batched = torch._C._functorch.is_batchedtensor(selected_groups)
            if not is_batched and (
                bool(torch.any(selected_groups < 0))
                or bool(torch.any(selected_groups >= resolved_group_count))
            ):
                raise ValueError("selected_groups contains an out-of-range group index")
            selection_logits = (
                self._standardize_route_logits(group_logits)
                if selected_groups_normalized
                else group_logits
            )
            selected_logits = selection_logits.gather(-1, selected_groups)
        elif self.training and self._bank_gradient_enabled and self.route_exploration > 0:
            normalized_logits = self._standardize_route_logits(group_logits)
            uniform = torch.rand(
                normalized_logits.shape[0],
                1,
                normalized_logits.shape[-1],
                device=normalized_logits.device,
                dtype=torch.float32,
            ).clamp(torch.finfo(torch.float32).eps, 1.0 - torch.finfo(torch.float32).eps)
            gumbel = -torch.log(-torch.log(uniform))
            selection_logits = normalized_logits + self.route_exploration * gumbel
            selected_groups = torch.topk(
                selection_logits,
                self.group_topk,
                dim=-1,
            ).indices
            selected_logits = normalized_logits.gather(-1, selected_groups)
        else:
            selected_logits, selected_groups = torch.topk(
                group_logits,
                self.group_topk,
                dim=-1,
            )
        group_weights = self._selected_group_weights(
            group_logits,
            selected_groups,
            selected_logits,
        )
        if active_groups is not None:
            selected_groups = active_groups[selected_groups]
        global_selected_groups = selected_groups + group_offset
        if self.group_size == 1:
            value_bank = memory
            if value_bank is None:
                value_bank = self.bank if self._bank_gradient_enabled else self.bank.detach()
            selected_values = self._select_explicit_values(
                value_bank,
                global_selected_groups,
                sparse=active_groups is not None,
            )
            context = torch.sum(
                selected_values.to(dtype=z.dtype) * group_weights.to(dtype=z.dtype).unsqueeze(-1),
                dim=-2,
            )
            return (
                context,
                group_weights.unsqueeze(-1),
                global_selected_groups.unsqueeze(-1),
                route,
            )
        offsets = torch.arange(self.group_size, device=z.device)
        indices = global_selected_groups.unsqueeze(-1) * self.group_size + offsets
        key_bank = self.key_bank if self._bank_gradient_enabled else self.key_bank.detach()
        selected_keys = torch.nn.functional.embedding(
            indices,
            key_bank,
            sparse=active_groups is not None,
        )
        if not autocast_enabled:
            selected_keys = selected_keys.to(dtype=query.dtype)
        slot_logits = (
            torch.einsum(
                "bnd,bntmd->bntm",
                query,
                selected_keys,
            )
            * self.scale
        )
        slot_weights = torch.softmax(slot_logits, dim=-1)
        weights = group_weights.to(slot_weights.dtype).unsqueeze(-1) * slot_weights
        value_bank = self.bank if self._bank_gradient_enabled else self.bank.detach()
        if active_groups is None and self._native_grouped_mm_available(weights):
            context = self._grouped_mm_value_read(
                global_selected_groups,
                weights,
                value_bank,
                output_dtype=z.dtype,
            )
        else:
            context = self._weighted_value_read(
                indices,
                weights,
                value_bank,
                output_dtype=z.dtype,
                sparse=active_groups is not None,
            )
        return context, weights, indices, route

    @staticmethod
    def _select_explicit_values(
        value_bank: Tensor,
        indices: Tensor,
        *,
        sparse: bool,
    ) -> Tensor:
        """Gather shared or per-sample group_size=1 Recall values."""

        if value_bank.ndim == 2:
            return torch.nn.functional.embedding(indices, value_bank, sparse=sparse)
        batch_index = torch.arange(value_bank.shape[0], device=value_bank.device)
        batch_index = batch_index.reshape(value_bank.shape[0], *([1] * (indices.ndim - 1)))
        return value_bank[batch_index, indices]

    @staticmethod
    def _native_grouped_mm_available(weights: Tensor) -> bool:
        grouped_mm = getattr(torch.nn.functional, "grouped_mm", None)
        if (
            grouped_mm is None
            or not weights.is_cuda
            or weights.dtype != torch.bfloat16
            or weights.shape[-1] % 8
        ):
            return False
        major, _minor = torch.cuda.get_device_capability(weights.device)
        return major >= 8

    @staticmethod
    def _grouped_mm_value_read(
        selected_groups: Tensor,
        weights: Tensor,
        value_bank: Tensor,
        *,
        output_dtype: torch.dtype,
    ) -> Tensor:
        """Read contiguous Recall groups through PyTorch's native grouped GEMM."""

        batch_shape = selected_groups.shape[:-1]
        group_topk = selected_groups.shape[-1]
        group_size = weights.shape[-1]
        groups = value_bank.shape[0] // group_size
        token_count = math.prod(batch_shape)
        flat_groups = selected_groups.reshape(-1)
        flat_weights = weights.reshape(-1, group_size)
        order = torch.argsort(flat_groups, stable=True)
        sorted_groups = flat_groups.index_select(0, order)
        sorted_weights = flat_weights.index_select(0, order).contiguous()
        offsets = (
            torch.bincount(
                sorted_groups,
                minlength=groups,
            )
            .cumsum(0)
            .to(torch.int32)
        )
        grouped_values = value_bank.to(torch.bfloat16).reshape(
            groups,
            group_size,
            value_bank.shape[-1],
        )
        partial = torch.nn.functional.grouped_mm(
            sorted_weights,
            grouped_values,
            offs=offsets,
        )
        if group_topk == 1:
            context = torch.empty_like(partial)
            context.index_copy_(0, order, partial)
            return context.reshape(*batch_shape, value_bank.shape[-1]).to(output_dtype)

        token_indices = torch.arange(
            token_count,
            device=selected_groups.device,
        ).repeat_interleave(group_topk)
        context = torch.zeros(
            token_count,
            value_bank.shape[-1],
            device=value_bank.device,
            dtype=weights.dtype,
        )
        context.index_add_(
            0,
            token_indices.index_select(0, order),
            partial,
        )
        return context.reshape(*batch_shape, value_bank.shape[-1]).to(output_dtype)

    @staticmethod
    def _weighted_value_read(
        indices: Tensor,
        weights: Tensor,
        value_bank: Tensor,
        *,
        output_dtype: torch.dtype,
        sparse: bool = False,
    ) -> Tensor:
        """Reduce sparse Recall values without materializing [B,N,T,M,H]."""

        selected_per_token = indices.shape[-2] * indices.shape[-1]
        flat_indices = indices.reshape(-1, selected_per_token)
        flat_weights = weights.reshape(-1, selected_per_token).to(value_bank.dtype)
        if sparse:
            context = _CoalescedSparseEmbeddingBag.apply(
                flat_indices,
                value_bank,
                flat_weights,
                indices.shape[-1],
            )
        else:
            embedding_weight = value_bank
            if value_bank.is_cuda and value_bank.dtype in {
                torch.float16,
                torch.bfloat16,
            }:
                # CUDA embedding_bag cannot backpropagate per-sample weights
                # for low-precision values on all supported PyTorch builds.
                embedding_weight = value_bank.float()
                flat_weights = flat_weights.float()
            context = torch.nn.functional.embedding_bag(
                flat_indices,
                embedding_weight,
                mode="sum",
                per_sample_weights=flat_weights,
            )
        return context.reshape(*indices.shape[:-2], value_bank.shape[-1]).to(output_dtype)

    @staticmethod
    def _standardize_route_logits(logits: Tensor) -> Tensor:
        values = logits.float()
        centered = values - values.mean(dim=-1, keepdim=True)
        variance = centered.square().mean(dim=-1, keepdim=True).clamp_min(1e-6)
        scale = variance.sqrt()
        return centered / scale

    def _active_training_groups(
        self,
        group_count: int,
        device: torch.device,
    ) -> Tensor | None:
        selection = self._training_group_partitions
        if not self.training or selection is None:
            return None
        partition_count, active_partitions = selection
        if len(active_partitions) == partition_count:
            return None
        group_indices = torch.arange(group_count, device=device)
        group_partitions = group_indices.remainder(partition_count)
        active = torch.tensor(
            active_partitions,
            device=device,
            dtype=group_partitions.dtype,
        )
        allowed = torch.isin(group_partitions, active)
        return group_indices[allowed]


class ARTIDynamicStateLayer(nn.Module):
    """Runtime latent state update with phase, interface, recall, and gated residuals."""

    def __init__(self, config: ARTIConfig) -> None:
        super().__init__()
        self.config = config
        hidden_dim = int(config.hidden_dim)
        self.phase = (
            ARTIPhaseMixer(hidden_dim, config.coord_dim, config.operator_count, config.dropout)
            if config.use_phase_mixer
            else None
        )
        self.interface = (
            ARTIVirtualInterfaceMixer(hidden_dim, config.interface_slots)
            if config.use_virtual_interface
            else None
        )
        self.recall = (
            ARTILatentRecallField(
                hidden_dim,
                config.recall_slots,
                recognition_mode=config.recall_recognition_mode,
                recognition_threshold=config.recall_recognition_threshold,
                recognition_temperature=config.recall_recognition_temperature,
                routing=config.recall_routing,
                key_dim=config.recall_key_dim,
                query_mode=config.recall_query_mode,
                query_seed=config.recall_query_seed,
                group_size=config.recall_group_size,
                group_topk=config.recall_group_topk,
                value_composition=config.recall_value_composition,
                factor_activation=config.recall_activation,
                route_exploration=config.recall_route_exploration,
            )
            if config.use_recall and config.recall_steps > 0
            else None
        )
        self.recall_activation = (
            nn.Identity()
            if config.recall_value_composition == "state"
            else Half()
            if config.recall_activation == "half"
            else nn.Identity()
        )
        self.update_gate = nn.Linear(hidden_dim * 5 + config.coord_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim) if config.use_layer_norm else nn.Identity()
        self.dropout = nn.Dropout(config.dropout)
        init_arti_module(self)

    def forward(
        self,
        z: Tensor,
        coord: Tensor,
        mask: Tensor,
        visibility: Tensor | None = None,
        recall: Tensor | None = None,
        recall_steps: int | None = None,
        _selected_recall_groups: Tensor | None = None,
        _selected_groups_normalized: bool = False,
    ) -> tuple[Tensor, dict[str, Tensor], Tensor, Tensor]:
        pairwise_visibility = (
            ensure_visibility(visibility, mask) if self.config.use_pairwise_context else visibility
        )

        if self.config.use_pairwise_context:
            visible_logits = torch.einsum("bnd,bmd->bnm", z, z) * (z.shape[-1] ** -0.5)
            visible_weights = masked_softmax(visible_logits, pairwise_visibility, dim=-1)
            visible_context = torch.einsum("bnm,bmd->bnd", visible_weights, z)
        else:
            visible_weights = torch.empty(z.shape[0], z.shape[1], 0, device=z.device, dtype=z.dtype)
            visible_context = torch.zeros_like(z)

        if self.config.use_phase_mixer:
            assert self.phase is not None
            phase_context, operator_weights, phase_receptor_gain = self.phase(z, coord)
        else:
            phase_context = torch.zeros_like(z)
            operator_weights = torch.empty(
                z.shape[0], z.shape[1], 0, device=z.device, dtype=z.dtype
            )
            phase_receptor_gain = torch.empty(
                z.shape[0], z.shape[1], 0, z.shape[-1], device=z.device, dtype=z.dtype
            )

        if self.config.use_virtual_interface:
            assert self.interface is not None
            interface_context, read_weights, write_weights = self.interface(
                z, mask, pairwise_visibility
            )
        else:
            interface_context = torch.zeros_like(z)
            read_weights = torch.empty(z.shape[0], z.shape[1], 0, device=z.device, dtype=z.dtype)
            write_weights = torch.empty(z.shape[0], z.shape[1], 0, device=z.device, dtype=z.dtype)

        recall_context = torch.zeros_like(z)
        recall_weights = torch.empty(z.shape[0], z.shape[1], 0, device=z.device, dtype=z.dtype)
        recall_indices = torch.empty(z.shape[0], z.shape[1], 0, device=z.device, dtype=torch.long)
        recall_route = torch.empty(z.shape[0], z.shape[1], 0, device=z.device, dtype=z.dtype)
        recall_influence = torch.zeros_like(z)
        recall_recognition = torch.zeros(*z.shape[:2], device=z.device, dtype=z.dtype)
        raw_recall_context = torch.zeros_like(z)
        recall_input = z
        recalled_state = z
        recall_write = torch.zeros_like(z)
        configured_recall_steps = self.config.recall_steps if self.config.use_recall else 0
        if recall_steps is None:
            recall_steps = configured_recall_steps
        else:
            if isinstance(recall_steps, bool) or not isinstance(recall_steps, int):
                raise TypeError("recall_steps must be an integer or None")
            if recall_steps < 0:
                raise ValueError("recall_steps must be non-negative")
            if recall_steps > configured_recall_steps:
                raise ValueError(
                    "runtime recall_steps cannot exceed the configured recall_steps "
                    f"({configured_recall_steps})"
                )
        recall_active = mask.any(dim=1)
        recall_steps_executed = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        recall_update_ratio = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        recall_step_active: list[Tensor] = []
        recall_step_update_ratio: list[Tensor] = []
        selected_recall_groups = _selected_recall_groups
        for step in range(recall_steps):
            assert self.recall is not None
            read = self.recall(
                z,
                mask,
                recall,
                selected_groups=selected_recall_groups,
                _selected_groups_normalized=_selected_groups_normalized and step == 0,
            )
            (
                raw_recall_context,
                recall_weights,
                recall_influence,
                recall_recognition,
            ) = read
            recall_indices = read.indices
            recall_route = read.route
            if selected_recall_groups is None and self.recall.routing == "grouped":
                selected_recall_groups = (read.indices[..., 0] // self.recall.group_size).detach()
            active = recall_active.view(-1, 1, 1).to(z.dtype)
            raw_recall_context = raw_recall_context * active
            recall_influence = recall_influence * active
            recall_recognition = recall_recognition * recall_active.unsqueeze(-1).to(z.dtype)
            recall_context = self.recall_activation(raw_recall_context)
            applied_write = self.dropout(recall_context)
            previous = z
            active_tokens = (recall_active.unsqueeze(-1) & mask).unsqueeze(-1)
            step_write = torch.where(active_tokens, applied_write, torch.zeros_like(applied_write))
            if self.config.recall_value_composition == "state":
                z = torch.where(active_tokens, step_write, z)
                recall_write = recall_write + (z - previous)
            else:
                z = torch.where(active_tokens, z + step_write, z)
                recall_write = recall_write + step_write
            recalled_state = z
            recall_effect = z - previous
            valid = mask.unsqueeze(-1).to(torch.float32)
            recall_update_ratio = (
                (recall_effect.float() * valid).flatten(1).norm(dim=-1)
                / (previous.float() * valid)
                .flatten(1)
                .norm(dim=-1)
                .clamp_min(torch.finfo(torch.float32).eps)
            ).to(z.dtype)
            recall_steps_executed = recall_steps_executed + recall_active.to(z.dtype)
            recall_step_active.append(recall_active.to(z.dtype))
            recall_step_update_ratio.append(recall_update_ratio)
            if (
                self.config.recall_tolerance is not None
                and step + 1 >= self.config.recall_min_steps
            ):
                recall_active = recall_active & (
                    recall_update_ratio.detach() > self.config.recall_tolerance
                )
                if not bool(torch.any(recall_active)):
                    break

        if recall_steps > 0:
            padding = recall_steps - len(recall_step_active)
            recall_step_active.extend(
                torch.zeros_like(recall_steps_executed) for _ in range(padding)
            )
            recall_step_update_ratio.extend(
                torch.zeros_like(recall_update_ratio) for _ in range(padding)
            )
            step_active_diagnostic = torch.stack(recall_step_active, dim=1)
            step_ratio_diagnostic = torch.stack(recall_step_update_ratio, dim=1)
        else:
            step_active_diagnostic = torch.empty(z.shape[0], 0, device=z.device, dtype=z.dtype)
            step_ratio_diagnostic = torch.empty(z.shape[0], 0, device=z.device, dtype=z.dtype)

        update_input = torch.cat(
            [z, coord, phase_context, interface_context, visible_context, recall_context], dim=-1
        )
        gate = torch.sigmoid(self.update_gate(update_input)) * mask.unsqueeze(-1).to(z.dtype)
        candidate = phase_context + interface_context + visible_context
        updated = self.norm(z + self.dropout(gate * candidate))
        updated = updated * mask.unsqueeze(-1).to(updated.dtype)

        diagnostics = {
            "operator_weights": operator_weights,
            "phase_receptor_gain": phase_receptor_gain,
            "interface_read_weights": read_weights,
            "interface_write_weights": write_weights,
            "visibility_weights": visible_weights,
            "recall_bank_weights": recall_weights,
            "recall_bank_indices": recall_indices,
            "recall_route": recall_route,
            "recall_influence": recall_influence,
            "recall_recognition": recall_recognition,
            "recall_steps_executed": recall_steps_executed,
            "recall_step_active": step_active_diagnostic,
            "recall_step_update_ratio": step_ratio_diagnostic,
            "recall_update_ratio": recall_update_ratio,
            "recall_effect_norm": (recalled_state - recall_input).norm(dim=-1),
            "recall_activation_half": torch.full(
                (z.shape[0],),
                1.0 if self.config.recall_activation == "half" and recall_steps > 0 else 0.0,
                device=z.device,
                dtype=z.dtype,
            ),
            "recall_activation_survival_ratio": recall_context.norm(dim=-1)
            / raw_recall_context.norm(dim=-1).clamp_min(torch.finfo(z.dtype).eps),
            "residual_gate": gate,
            "residual_norm": (updated - z).norm(dim=-1),
            "mask_coverage": mask_coverage(mask),
        }
        return updated, diagnostics, recall_context, recall_route, recall_write


class ARTIRecallWriteState(nn.Module):
    """Query host-dimensional writes and apply them to the current state."""

    def __init__(
        self,
        config: ARTIConfig,
        *,
        identity_init_bank: bool = False,
        formula: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if not config.use_recall or config.recall_steps <= 0:
            raise ValueError("direct Recall requires use_recall=True and recall_steps > 0")
        hidden_dim = int(config.hidden_dim)
        if hidden_dim != config.input_dim:
            raise ValueError("direct Recall values must use the host input dimension")
        self.config = config
        self.recall = ARTILatentRecallField(
            hidden_dim,
            config.recall_slots,
            recognition_mode=config.recall_recognition_mode,
            recognition_threshold=config.recall_recognition_threshold,
            recognition_temperature=config.recall_recognition_temperature,
            routing=config.recall_routing,
            key_dim=config.recall_key_dim,
            query_mode=config.recall_query_mode,
            query_seed=config.recall_query_seed,
            group_size=config.recall_group_size,
            group_topk=config.recall_group_topk,
            value_composition=config.recall_value_composition,
            formula=formula,
            factor_activation=config.recall_activation,
            route_exploration=config.recall_route_exploration,
            project_external=False,
        )
        self._formula_replaces_state = formula is not None
        self._replaces_state = (
            config.recall_value_composition == "state" or self._formula_replaces_state
        )
        self.recall_activation = (
            nn.Identity()
            if config.recall_value_composition == "state"
            else Half()
            if config.recall_activation == "half"
            else nn.Identity()
        )
        self.dropout = nn.Dropout(config.dropout)
        self._compiled_product_tail: CompiledProductTail | None = None
        initial_retention = 1.0 if identity_init_bank and self._replaces_state else 0.0
        self.register_buffer(
            "_state_input_retention",
            torch.tensor(initial_retention, dtype=torch.float32),
            persistent=True,
        )
        self._state_bank_needs_calibration = bool(
            identity_init_bank and config.recall_value_composition == "state"
        )
        if identity_init_bank:
            if formula is not None:
                # Formula factor initialization is declared by the host
                # contract. The temporary input-retention path preserves the
                # pretrained host while those factors begin differentiating.
                pass
            elif config.recall_value_composition == "state":
                (
                    coarse_content,
                    fine_content,
                    *modulations,
                    write_direction,
                    memory_opacity,
                ) = self.recall.bank.chunk(
                    STATE_RECALL_COMPOSITION_FACTOR,
                    dim=0,
                )
                nn.init.normal_(coarse_content, mean=0.0, std=1.0)
                nn.init.zeros_(fine_content)
                for modulation in modulations:
                    nn.init.normal_(modulation, mean=0.0, std=0.05)
                nn.init.normal_(write_direction, mean=0.0, std=0.05)
                nn.init.normal_(memory_opacity, mean=0.0, std=0.05)
            else:
                nn.init.zeros_(self.recall.bank)

    @property
    def state_input_retention(self) -> float:
        """Return the temporary input bypass around multi-factor state Recall."""

        return float(self._state_input_retention.detach().cpu())

    @property
    def replaces_state(self) -> bool:
        """Return whether Recall yields a complete next state."""

        return self._replaces_state

    def set_state_input_retention(self, value: float) -> None:
        """Set the temporary state-Recall input bypass in the range [0, 1]."""

        value = float(value)
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("state input retention must be finite and in [0, 1]")
        self._state_input_retention.fill_(value)

    @property
    def state_bank_calibrated(self) -> bool:
        """Return whether state content values were matched to a host tensor."""

        return not self._state_bank_needs_calibration

    def mark_state_bank_calibrated(self) -> None:
        """Prevent data-dependent initialization from changing loaded values."""

        self._state_bank_needs_calibration = False

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        bank_key = f"{prefix}recall.bank"
        retention_key = f"{prefix}_state_input_retention"
        legacy_key_bank_key = f"{prefix}recall.key_bank"

        # Public 1.x Recall checkpoints predate the state-retention buffer and
        # stored a learned routing key bank. Current Recall keeps the value
        # Bank and query basis, but fixed routing no longer owns that key bank.
        if retention_key not in state_dict:
            state_dict[retention_key] = self._state_input_retention.detach().clone()
        if self.recall.key_bank is None:
            state_dict.pop(legacy_key_bank_key, None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )
        if self.config.recall_value_composition == "state" and bank_key in state_dict:
            self.mark_state_bank_calibrated()

    @torch.no_grad()
    def _calibrate_state_bank_once(
        self,
        z: Tensor,
        mask: Tensor | None,
    ) -> None:
        if not self._state_bank_needs_calibration:
            return
        values = z.detach().float()
        if mask is not None and bool(torch.any(mask)):
            values = values[mask]
        else:
            values = values.reshape(-1, values.shape[-1])
        feature_mean = values.mean(dim=0)
        feature_std = values.std(dim=0, unbiased=False)
        global_std = values.std(unbiased=False).clamp_min(torch.finfo(torch.float32).eps)
        feature_std = torch.where(
            feature_std > torch.finfo(torch.float32).eps,
            feature_std,
            global_std,
        )
        content = self.recall.bank.chunk(
            STATE_RECALL_COMPOSITION_FACTOR,
            dim=0,
        )[0]
        noise = content.detach().float().clamp_(-2.0, 2.0)
        noise = noise - noise.mean(dim=0, keepdim=True)
        noise_scale = noise.std(dim=0, unbiased=False).clamp_min(torch.finfo(torch.float32).eps)
        noise = noise / noise_scale
        calibrated = feature_mean + noise * feature_std
        content.copy_(calibrated.to(device=content.device, dtype=content.dtype))
        self._state_bank_needs_calibration = False

    def _compose_state_transition(self, previous: Tensor, recalled: Tensor) -> Tensor:
        retention = self._state_input_retention.to(
            device=previous.device,
            dtype=previous.dtype,
        )
        return previous * retention + recalled * (1.0 - retention)

    def _recalled_state(self, previous: Tensor, write: Tensor) -> Tensor:
        return previous + write if self._formula_replaces_state else write

    def compile_write_hotpath(
        self,
        *,
        mode: str,
        dynamic: bool,
        fullgraph: bool,
    ) -> bool:
        """Enable the shared route-stable product Recall compiler path."""

        activation = self.recall_activation
        if not (
            self.recall.routing == "grouped"
            and self.recall.value_composition == "product"
            and self.recall.group_topk == 1
            and self.recall.route_exploration == 0.0
            and self.recall.recognition_mode == "none"
            and isinstance(activation, Half)
            and not activation.stochastic
        ):
            return False
        assert self.recall.group_bank is not None
        self._compiled_product_tail = grouped_product_tail(
            group_size=self.recall.group_size,
            factor_groups=self.recall.group_bank.shape[0] // 2,
            key_dim=self.recall.key_dim,
            threshold=activation.threshold,
            base=activation.base,
            half_scale=activation.scale,
            mode=mode,
            dynamic=dynamic,
            fullgraph=fullgraph,
        )
        return True

    def _compiled_product_write(
        self,
        z: Tensor,
        mask: Tensor | None,
        recall: Tensor | None,
    ) -> Tensor | None:
        tail = self._compiled_product_tail
        field = self.recall
        if (
            tail is None
            or z.device.type != "cuda"
            or mask is not None
            or recall is not None
            or field._route_influence.numel() > 0
            or not torch.is_autocast_enabled(z.device.type)
        ):
            return None
        assert field.group_bank is not None
        assert field.key_bank is not None
        query = field._project_query(z)
        group_bank_parameter = (
            field.group_bank if field._bank_gradient_enabled else field.group_bank.detach()
        )
        factor_groups = group_bank_parameter.shape[0] // 2
        group_bank = group_bank_parameter.reshape(
            2,
            factor_groups,
            field.key_dim,
        )
        group_logits = torch.einsum("bnd,fgd->bnfg", query, group_bank) * field.scale
        group_logits = field._apply_route_prior(group_logits)
        selected_groups = torch.max(
            group_logits,
            dim=-1,
            keepdim=True,
        ).indices
        factor_offsets = torch.arange(2, device=z.device).view(1, 1, 2, 1)
        selected_groups = selected_groups + factor_offsets * factor_groups
        key_bank = field.key_bank if field._bank_gradient_enabled else field.key_bank.detach()
        value_bank = field.bank if field._bank_gradient_enabled else field.bank.detach()
        try:
            return tail(
                z,
                query,
                selected_groups,
                key_bank,
                value_bank,
            )
        except torch._dynamo.exc.BackendCompilerFailed:
            # A missing or incompatible compiler backend must not make an
            # otherwise valid eager adapter unusable.
            self._compiled_product_tail = None
            return None

    def forward(
        self,
        z: Tensor,
        mask: Tensor,
        recall: Tensor | None = None,
        route_assignment: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, dict[str, Tensor]]:
        self._calibrate_state_bank_once(z, mask)
        initial = z
        cumulative_write = torch.zeros_like(z)
        active = mask.any(dim=1)
        steps_executed = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        update_ratio = torch.zeros(z.shape[0], device=z.device, dtype=z.dtype)
        step_active: list[Tensor] = []
        step_update_ratio: list[Tensor] = []
        last_read: _ARTIRecallRead | None = None

        for step in range(self.config.recall_steps):
            # Re-route after every update so iterative Recall can select a new trace.
            read = self.recall(
                z,
                mask,
                recall,
                route_assignment=route_assignment,
            )
            last_read = read
            raw_write = read.context
            write = self.dropout(self.recall_activation(raw_write))
            previous = z
            active_tokens = (active.unsqueeze(-1) & mask).unsqueeze(-1)
            step_write = torch.where(active_tokens, write, torch.zeros_like(write))
            if self._replaces_state:
                candidate = self._compose_state_transition(
                    initial,
                    self._recalled_state(previous, step_write),
                )
                z = torch.where(active_tokens, candidate, z)
                cumulative_write = cumulative_write + (z - previous)
            else:
                z = torch.where(active_tokens, z + step_write, z)
                cumulative_write = cumulative_write + step_write
            effect = z - previous
            valid = mask.unsqueeze(-1).to(torch.float32)
            update_ratio = (
                (effect.float() * valid).flatten(1).norm(dim=-1)
                / (previous.float() * valid)
                .flatten(1)
                .norm(dim=-1)
                .clamp_min(torch.finfo(torch.float32).eps)
            ).to(z.dtype)
            steps_executed = steps_executed + active.to(z.dtype)
            step_active.append(active.to(z.dtype))
            step_update_ratio.append(update_ratio)
            if (
                self.config.recall_tolerance is not None
                and step + 1 >= self.config.recall_min_steps
            ):
                active = active & (update_ratio.detach() > self.config.recall_tolerance)
                if not bool(torch.any(active)):
                    break

        padding = self.config.recall_steps - len(step_active)
        step_active.extend(torch.zeros_like(steps_executed) for _ in range(padding))
        step_update_ratio.extend(torch.zeros_like(update_ratio) for _ in range(padding))
        if self._replaces_state:
            cumulative_write = z - initial
        assert last_read is not None
        diagnostics = {
            "recall_bank_weights": last_read.weights,
            "recall_bank_indices": last_read.indices,
            "recall_route": last_read.route,
            "recall_influence": last_read.influence,
            "recall_recognition": last_read.recognition,
            "recall_steps_executed": steps_executed,
            "recall_step_active": torch.stack(step_active, dim=1),
            "recall_step_update_ratio": torch.stack(step_update_ratio, dim=1),
            "recall_update_ratio": update_ratio,
            "recall_effect_norm": (z - initial).norm(dim=-1),
            "recall_write_norm": cumulative_write.norm(dim=-1),
        }
        return z, cumulative_write, diagnostics

    def forward_write(
        self,
        z: Tensor,
        mask: Tensor | None,
        recall: Tensor | None = None,
        route_assignment: Tensor | None = None,
        *,
        static_steps: bool = False,
    ) -> Tensor:
        """Return the Recall-written host tensor without diagnostic reductions."""

        self._calibrate_state_bank_once(z, mask)
        initial = z
        if self.config.recall_tolerance is None:
            for _step in range(self.config.recall_steps):
                write = (
                    None
                    if route_assignment is not None
                    else self._compiled_product_write(z, mask, recall)
                )
                if write is None:
                    write = self.recall_activation(
                        self.recall.read_context(
                            z,
                            mask,
                            recall,
                            route_assignment=route_assignment,
                        )
                    )
                write = self.dropout(write)
                if self._replaces_state:
                    candidate = self._compose_state_transition(
                        initial,
                        self._recalled_state(z, write),
                    )
                    z = candidate if mask is None else torch.where(mask.unsqueeze(-1), candidate, z)
                else:
                    z = z + write
            return z

        active = (
            torch.ones(z.shape[0], device=z.device, dtype=torch.bool)
            if mask is None
            else mask.any(dim=1)
        )
        tolerance = self.config.recall_tolerance
        token_active = None if mask is None else mask.unsqueeze(-1)

        for step in range(self.config.recall_steps):
            # A static loop avoids device-to-host synchronization and remains
            # compiler/capture friendly. Inactive samples contribute exact zeros.
            write = (
                None
                if route_assignment is not None
                else self._compiled_product_write(z, mask, recall)
            )
            if write is None:
                write = self.recall_activation(
                    self.recall.read_context(
                        z,
                        mask,
                        recall,
                        route_assignment=route_assignment,
                    )
                )
            write = self.dropout(write)
            previous = z
            if step < self.config.recall_min_steps:
                if self._replaces_state:
                    candidate = self._compose_state_transition(
                        initial,
                        self._recalled_state(previous, write),
                    )
                    z = (
                        candidate
                        if token_active is None
                        else torch.where(token_active, candidate, z)
                    )
                else:
                    z = z + write
            else:
                active_tokens = active.view(-1, 1, 1)
                if token_active is not None:
                    active_tokens = active_tokens & token_active
                candidate = (
                    self._compose_state_transition(
                        initial,
                        self._recalled_state(previous, write),
                    )
                    if self._replaces_state
                    else z + write
                )
                z = torch.where(active_tokens, candidate, z)
            if step + 1 >= self.config.recall_min_steps and step + 1 < self.config.recall_steps:
                valid = (
                    torch.ones_like(z, dtype=torch.float32)
                    if mask is None
                    else mask.unsqueeze(-1).to(torch.float32)
                )
                update_ratio = ((z - previous).float() * valid).flatten(1).norm(dim=-1) / (
                    previous.float() * valid
                ).flatten(1).norm(dim=-1).clamp_min(torch.finfo(torch.float32).eps)
                active = active & (update_ratio.detach() > tolerance)
                if not static_steps and not bool(torch.any(active)):
                    break

        return z


class ARTIRecallWriteLayer(nn.Module):
    """Apply Recall writes in the input space and return the modified tensor."""

    def __init__(
        self,
        config: ARTIConfig,
        *,
        identity_init_bank: bool = False,
        formula: nn.Module | None = None,
    ) -> None:
        super().__init__()
        if config.hidden_dim != config.input_dim:
            raise ValueError("direct Recall requires hidden_dim == input_dim")
        self.config = config
        self.state = ARTIRecallWriteState(
            config,
            identity_init_bank=identity_init_bank,
            formula=formula,
        )
        self._compiled_state_write: Callable[..., Tensor] | None = None
        self._write_hooks: OrderedDict[
            int,
            Callable[[ARTIRecallWriteLayer, Tensor], None],
        ] = OrderedDict()

    def _register_write_hook(
        self,
        hook: Callable[[ARTIRecallWriteLayer, Tensor], None],
    ) -> RemovableHandle:
        handle = RemovableHandle(self._write_hooks)
        self._write_hooks[handle.id] = hook
        return handle

    def compile_write_hotpath(
        self,
        *,
        mode: str = "default",
        dynamic: bool = False,
        fullgraph: bool = False,
    ) -> None:
        """Compile Recall writes while leaving hooks and routing eager."""

        if self.state.compile_write_hotpath(
            mode=mode,
            dynamic=dynamic,
            fullgraph=fullgraph,
        ):
            return
        if self._compiled_state_write is None:
            self._compiled_state_write = torch.compile(
                self.state.forward_write,
                mode=mode,
                dynamic=dynamic,
                fullgraph=fullgraph,
            )

    @property
    def write_hotpath_compiled(self) -> bool:
        return (
            self.state._compiled_product_tail is not None or self._compiled_state_write is not None
        )

    def forward(
        self,
        x: Tensor,
        coord: Tensor | None = None,
        mask: Tensor | None = None,
        visibility: Tensor | None = None,
        recall: Tensor | None = None,
        frame_operators: Tensor | None = None,
        observer_coord: Tensor | None = None,
        route_assignment: Tensor | None = None,
    ) -> ARTIOutput:
        assert_floating_tensor("x", x)
        seq, was_vector = as_sequence(x)
        if seq.shape[-1] != self.config.input_dim:
            raise ValueError(f"x last dim must be {self.config.input_dim}, got {seq.shape[-1]}")
        batch, tokens, _ = seq.shape
        token_mask = ensure_mask(mask, batch, tokens, seq.device)
        if self.config.require_coord and coord is None:
            raise ValueError("coord is required by this ARTI configuration")
        token_coord = ensure_coord(
            coord, batch, tokens, self.config.coord_dim, seq.device, seq.dtype
        )
        if self.config.require_visibility and visibility is None:
            raise ValueError("visibility is required by this ARTI configuration")
        if visibility is not None:
            ensure_visibility(visibility, token_mask)
        query_state = apply_coord_frame_inverse(
            seq,
            token_coord,
            self.config.coord_frame_mode,
            frame_operators,
            observer_coord=observer_coord,
        )
        written_seq, write_seq, diagnostics = self.state(
            query_state,
            token_mask,
            recall,
            route_assignment,
        )
        pooled = masked_mean(written_seq, token_mask, dim=1)
        written = restore_input_rank(
            written_seq,
            was_vector and self.config.return_input_shape,
        )
        trace = written if self.config.use_virtual_recall else None
        diagnostics["pooled_mean"] = pooled.mean(dim=-1)
        diagnostics["pooled_std"] = pooled.std(dim=-1, unbiased=False)
        return ARTIOutput(
            y=written,
            pooled=pooled,
            virtual_y=None,
            recall_trace=trace,
            recall_prediction=trace,
            recall_influence=restore_input_rank(
                write_seq,
                was_vector and self.config.return_input_shape,
            ),
            recall_context=restore_input_rank(
                write_seq,
                was_vector and self.config.return_input_shape,
            ),
            recall_route=restore_input_rank(
                diagnostics["recall_route"],
                was_vector and self.config.return_input_shape,
            ),
            diagnostics=detach_diagnostics(diagnostics),
        )

    def forward_write(
        self,
        x: Tensor,
        coord: Tensor | None = None,
        mask: Tensor | None = None,
        visibility: Tensor | None = None,
        recall: Tensor | None = None,
        frame_operators: Tensor | None = None,
        observer_coord: Tensor | None = None,
        route_assignment: Tensor | None = None,
        *,
        static_steps: bool = False,
    ) -> Tensor:
        """Return only the Recall-written host tensor for hot insertion paths."""

        assert_floating_tensor("x", x)
        seq, was_vector = as_sequence(x)
        if seq.shape[-1] != self.config.input_dim:
            raise ValueError(f"x last dim must be {self.config.input_dim}, got {seq.shape[-1]}")
        batch, tokens, _ = seq.shape
        token_mask = (
            None
            if mask is None and visibility is None
            else ensure_mask(mask, batch, tokens, seq.device)
        )
        if self.config.require_coord and coord is None:
            raise ValueError("coord is required by this ARTI configuration")
        token_coord = ensure_coord(
            coord,
            batch,
            tokens,
            self.config.coord_dim,
            seq.device,
            seq.dtype,
        )
        if self.config.require_visibility and visibility is None:
            raise ValueError("visibility is required by this ARTI configuration")
        if visibility is not None:
            assert token_mask is not None
            ensure_visibility(visibility, token_mask)
        query_state = apply_coord_frame_inverse(
            seq,
            token_coord,
            self.config.coord_frame_mode,
            frame_operators,
            observer_coord=observer_coord,
        )
        write_state = (
            self._compiled_state_write
            if self._compiled_state_write is not None and torch.is_grad_enabled()
            else self.state.forward_write
        )
        written = write_state(
            query_state,
            token_mask,
            recall,
            route_assignment,
            static_steps=static_steps,
        )
        restored = restore_input_rank(
            written,
            was_vector and self.config.return_input_shape,
        )
        for hook in tuple(self._write_hooks.values()):
            hook(self, restored)
        return restored


class ARTILatentTensorLayer(nn.Module):
    """Project anonymous hidden tensors into a dynamic latent space."""

    def __init__(self, config: ARTIConfig) -> None:
        super().__init__()
        self.config = config
        hidden_dim = int(config.hidden_dim)
        self.in_proj = nn.Linear(config.input_dim, hidden_dim)
        self.state = ARTIDynamicStateLayer(config)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.virtual_recall_proj = (
            nn.Linear(hidden_dim, hidden_dim) if config.use_virtual_recall else None
        )
        if config.coord_dim > 0 and config.fallback_context in {"random_coord", "random_context"}:
            fallback_coord = torch.randn(config.fallback_slots, config.coord_dim) * (
                config.coord_dim**-0.5
            )
            self.register_buffer("fallback_coord_bank", fallback_coord)
        else:
            self.register_buffer("fallback_coord_bank", torch.empty(0, config.coord_dim))
        init_arti_module(self.in_proj)
        init_arti_module(self.out_proj)
        if self.virtual_recall_proj is not None:
            init_arti_module(self.virtual_recall_proj)

    def forward(
        self,
        x: Tensor,
        coord: Tensor | None = None,
        mask: Tensor | None = None,
        visibility: Tensor | None = None,
        recall: Tensor | None = None,
        frame_operators: Tensor | None = None,
        observer_coord: Tensor | None = None,
        recall_steps: int | None = None,
        _selected_recall_groups: Tensor | None = None,
        _selected_groups_normalized: bool = False,
    ) -> ARTIOutput:
        assert_floating_tensor("x", x)
        seq, was_vector = as_sequence(x)
        batch, tokens, _ = seq.shape
        token_mask = ensure_mask(mask, batch, tokens, seq.device)
        if self.config.require_coord and coord is None:
            raise ValueError("coord is required by this ARTI configuration")
        token_coord = self._resolve_coord(coord, batch, tokens, seq.device, seq.dtype)
        if self.config.require_visibility and visibility is None:
            raise ValueError("visibility is required by this ARTI configuration")
        visibility = self._resolve_visibility(visibility, token_mask)

        seq_canonical = apply_coord_frame_inverse(
            seq,
            token_coord,
            self.config.coord_frame_mode,
            frame_operators,
            observer_coord=observer_coord,
        ) * token_mask.unsqueeze(-1).to(seq.dtype)
        z = self.in_proj(seq_canonical) * token_mask.unsqueeze(-1).to(seq.dtype)
        virtual_seq = None
        private_recall = recall
        if self.virtual_recall_proj is not None:
            virtual_input = (
                torch.nn.functional.layer_norm(z, (z.shape[-1],))
                if self.config.use_layer_norm
                else z
            )
            virtual_seq = self.virtual_recall_proj(virtual_input) * token_mask.unsqueeze(-1).to(
                z.dtype
            )
            if self.state.recall is not None:
                self_signal = masked_mean(virtual_seq, token_mask, dim=1).unsqueeze(1)
                private_recall = (
                    self_signal
                    if recall is None
                    else torch.cat([self_signal, recall.to(self_signal)], dim=1)
                )
        z, diagnostics, recall_context, recall_route, recall_write = self.state(
            z,
            token_coord,
            token_mask,
            visibility,
            private_recall,
            recall_steps,
            _selected_recall_groups,
            _selected_groups_normalized,
        )
        y_seq = self.out_proj(z) * token_mask.unsqueeze(-1).to(z.dtype)
        recall_trace = None
        recall_prediction = None
        if virtual_seq is not None:
            recall_trace = virtual_seq
            recall_prediction = (
                self.out_proj(recall_write) * token_mask.unsqueeze(-1).to(recall_write.dtype)
                if self.state.recall is not None
                else virtual_seq
            )
        pooled = masked_mean(y_seq, token_mask, dim=1)
        diagnostics["pooled_mean"] = pooled.mean(dim=-1)
        diagnostics["pooled_std"] = pooled.std(dim=-1, unbiased=False)
        if recall_trace is not None and recall_prediction is not None:
            diagnostics["experiential_recall_trace_norm"] = recall_trace.norm(dim=-1)
            diagnostics["experiential_recall_prediction_norm"] = recall_prediction.norm(dim=-1)
            diagnostics["experiential_recall_familiarity"] = torch.cosine_similarity(
                recall_prediction,
                y_seq.detach(),
                dim=-1,
                eps=1e-6,
            ) * token_mask.to(y_seq.dtype)
            diagnostics["experiential_recall_self_input_norm"] = masked_mean(
                recall_trace,
                token_mask,
                dim=1,
            ).norm(dim=-1)
        diagnostics["coord_frame_delta"] = (seq_canonical - seq).norm(dim=-1)
        diagnostics["observer_frame_active"] = torch.full(
            (batch,),
            1.0 if observer_coord is not None and self.config.coord_frame_mode != "none" else 0.0,
            device=seq.device,
            dtype=seq.dtype,
        )

        y = restore_input_rank(y_seq, was_vector and self.config.return_input_shape)
        virtual_y = (
            None
            if virtual_seq is None
            else restore_input_rank(virtual_seq, was_vector and self.config.return_input_shape)
        )
        trace = (
            None
            if recall_trace is None
            else restore_input_rank(recall_trace, was_vector and self.config.return_input_shape)
        )
        prediction = (
            None
            if recall_prediction is None
            else restore_input_rank(
                recall_prediction, was_vector and self.config.return_input_shape
            )
        )
        influence = restore_input_rank(
            recall_context,
            was_vector and self.config.return_input_shape,
        )
        route = restore_input_rank(
            recall_route,
            was_vector and self.config.return_input_shape,
        )
        return ARTIOutput(
            y=y,
            pooled=pooled,
            virtual_y=virtual_y,
            recall_trace=trace,
            recall_prediction=prediction,
            recall_influence=influence,
            recall_context=influence,
            recall_route=route,
            diagnostics=detach_diagnostics(diagnostics),
        )

    def _resolve_coord(
        self,
        coord: Tensor | None,
        batch: int,
        tokens: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tensor:
        if (
            coord is not None
            or self.config.fallback_context == "none"
            or self.config.coord_dim == 0
        ):
            return ensure_coord(coord, batch, tokens, self.config.coord_dim, device, dtype)
        bank = self.fallback_coord_bank.to(device=device, dtype=dtype)
        if bank.shape[0] == 0:
            return ensure_coord(coord, batch, tokens, self.config.coord_dim, device, dtype)
        index = torch.arange(tokens, device=device) % bank.shape[0]
        return bank.index_select(0, index).unsqueeze(0).expand(batch, -1, -1)

    def _resolve_visibility(self, visibility: Tensor | None, mask: Tensor) -> Tensor | None:
        if visibility is not None or self.config.fallback_context != "random_context":
            return visibility
        return mask.unsqueeze(1) & mask.unsqueeze(2)


class ARTILayer(ARTILatentTensorLayer):
    """Convenience constructor for the default latent tensor layer."""

    def __init__(
        self,
        input_dim: int,
        coord_dim: int = 0,
        hidden_dim: int | None = None,
        operator_count: int = 4,
        interface_slots: int = 8,
        recall_slots: int = 4,
        recall_steps: int = 1,
        recall_min_steps: int = 1,
        recall_tolerance: float | None = None,
        recall_activation: str = "half",
        recall_recognition_mode: str = "none",
        recall_recognition_threshold: float = 0.5,
        recall_recognition_temperature: float = 0.1,
        recall_routing: str = "dense",
        recall_key_dim: int = 32,
        recall_query_mode: str = "fixed",
        recall_query_seed: int = 0,
        recall_group_size: int = 16,
        recall_group_topk: int = 2,
        recall_value_composition: str = "single",
        recall_route_exploration: float = 0.0,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        use_phase_mixer: bool = True,
        use_virtual_interface: bool = True,
        use_pairwise_context: bool = True,
        use_recall: bool = True,
        use_virtual_recall: bool = True,
        require_coord: bool = False,
        require_visibility: bool = False,
        coord_frame_mode: str = "none",
        fallback_context: str = "none",
        fallback_slots: int = 32,
    ) -> None:
        super().__init__(
            ARTIConfig(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                coord_dim=coord_dim,
                operator_count=operator_count,
                interface_slots=interface_slots,
                recall_slots=recall_slots,
                recall_steps=recall_steps,
                recall_min_steps=recall_min_steps,
                recall_tolerance=recall_tolerance,
                recall_activation=recall_activation,
                recall_recognition_mode=recall_recognition_mode,
                recall_recognition_threshold=recall_recognition_threshold,
                recall_recognition_temperature=recall_recognition_temperature,
                recall_routing=recall_routing,
                recall_key_dim=recall_key_dim,
                recall_query_mode=recall_query_mode,
                recall_query_seed=recall_query_seed,
                recall_group_size=recall_group_size,
                recall_group_topk=recall_group_topk,
                recall_value_composition=recall_value_composition,
                recall_route_exploration=recall_route_exploration,
                dropout=dropout,
                use_layer_norm=use_layer_norm,
                use_phase_mixer=use_phase_mixer,
                use_virtual_interface=use_virtual_interface,
                use_pairwise_context=use_pairwise_context,
                use_recall=use_recall,
                use_virtual_recall=use_virtual_recall,
                require_coord=require_coord,
                require_visibility=require_visibility,
                coord_frame_mode=coord_frame_mode,
                fallback_context=fallback_context,
                fallback_slots=fallback_slots,
            )
        )
