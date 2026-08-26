"""Formula-intervention support and bounded selective execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

import torch
from torch import Tensor, nn

from .reversible_topology import FoldedTensor
from .vnext_contracts import ContractLimits, DEFAULT_CONTRACT_LIMITS, FoldedPulseSupports


@dataclass(frozen=True)
class ActiveWorkspace:
    """The only folded partition visible to intervention and compute plugins."""

    value: Tensor
    validity: Tensor
    exposed: Tensor
    intervened: Tensor

    def __post_init__(self) -> None:
        if not isinstance(self.value, Tensor) or self.value.ndim < 3:
            raise ValueError("ActiveWorkspace value must be a [..., K, D] Tensor")
        expected = self.value.shape[:-1]
        for mask, name in (
            (self.validity, "validity"),
            (self.exposed, "exposed"),
            (self.intervened, "intervened"),
        ):
            if not isinstance(mask, Tensor) or mask.dtype != torch.bool or mask.shape != expected:
                raise ValueError(
                    f"ActiveWorkspace {name} must be boolean with shape value.shape[:-1]"
                )
            if mask.device != self.value.device:
                raise ValueError(f"ActiveWorkspace {name} must share the value device")
        if torch.compiler.is_compiling():
            torch._assert_async(
                (~self.intervened | self.exposed).all(),
                "intervened support must be exposed",
            )
            torch._assert_async(
                (~self.exposed | self.validity).all(),
                "exposed support must be valid",
            )
        else:
            if bool((self.intervened & ~self.exposed).any()):
                raise ValueError("intervened support must be a subset of exposed support")
            if bool((self.exposed & ~self.validity).any()):
                raise ValueError("exposed support must be a subset of validity")

    def replace(
        self,
        *,
        value: Tensor | None = None,
        intervened: Tensor | None = None,
    ) -> ActiveWorkspace:
        return ActiveWorkspace(
            self.value if value is None else value,
            self.validity,
            self.exposed,
            self.intervened if intervened is None else intervened,
        )


@dataclass(frozen=True)
class FoldedWorkspace:
    """Runtime-only Fold state whose plugin boundary exposes only the active plane."""

    state: FoldedTensor
    supports: FoldedPulseSupports

    def __post_init__(self) -> None:
        if not isinstance(self.state, FoldedTensor) or not isinstance(
            self.supports, FoldedPulseSupports
        ):
            raise TypeError("FoldedWorkspace requires FoldedTensor and FoldedPulseSupports")
        if self.supports.record is not self.state.record:
            raise ValueError("Folded support lineage does not match FoldedTensor record")
        active = self.supports.active
        preserved = self.supports.preserved
        active_validity = torch.eq(active._validity, self.state.active_mask).all()
        preserved_validity = torch.eq(preserved._validity, self.state.folded_mask).all()
        intervention_is_active = ~preserved.intervened._mask.any()
        if torch.compiler.is_compiling():
            torch._assert_async(active_validity, "active support validity mismatch")
            torch._assert_async(preserved_validity, "preserved support validity mismatch")
            torch._assert_async(
                intervention_is_active,
                "Fold topology left intervention support outside the active plane",
            )
        else:
            if not bool(active_validity) or not bool(preserved_validity):
                raise ValueError("Folded support validity does not match FoldedTensor")
            if not bool(intervention_is_active):
                raise ValueError("Fold topology must place every intervened value in active")

    def active_workspace(self) -> ActiveWorkspace:
        active = self.supports.active
        return ActiveWorkspace(
            self.state.active,
            active._validity,
            active.exposed._mask,
            active.intervened._mask,
        )

    def commit(self, active: ActiveWorkspace) -> FoldedWorkspace:
        if not isinstance(active, ActiveWorkspace):
            raise TypeError("FoldedWorkspace.commit requires ActiveWorkspace")
        if active.value.shape != self.state.active.shape:
            raise ValueError("committed active value shape does not match FoldedTensor")
        for actual, expected, name in (
            (active.validity, self.supports.active._validity, "validity"),
            (active.exposed, self.supports.active.exposed._mask, "exposed"),
        ):
            equal = torch.eq(actual, expected).all()
            if torch.compiler.is_compiling():
                torch._assert_async(equal, f"selective stage changed {name} support")
            elif not bool(equal):
                raise ValueError(f"selective stage changed {name} support")
        active_supports = self.supports.active.replace_intervened(active.intervened)
        transported = self.supports.replace_active(active_supports)
        return FoldedWorkspace(self.state.replace(active=active.value), transported)


@dataclass(frozen=True)
class InterventionProposal:
    """Per-active-instance priority proposed to the fixed selection operator."""

    priority: Tensor


class MagnitudeInterventionPolicy(nn.Module):
    """Reference policy that proposes intervention from active feature strength."""

    _component_reference: ClassVar[str] = "arti/magnitude-intervention-policy@1"

    def forward(
        self,
        value: Tensor,
        _factors: Tensor,
        *,
        exposed: Tensor,
    ) -> InterventionProposal:
        redacted = torch.where(exposed.unsqueeze(-1), value, torch.zeros_like(value))
        return InterventionProposal(redacted.square().mean(dim=-1))


class FactorInterventionPolicy(nn.Module):
    """Use one typed factor channel as intervention priority."""

    _component_reference: ClassVar[str] = "arti/factor-intervention-policy@1"

    def __init__(self, factor_index: int = 0) -> None:
        super().__init__()
        if isinstance(factor_index, bool) or not isinstance(factor_index, int) or factor_index < 0:
            raise ValueError("factor_index must be a non-negative integer")
        self.factor_index = factor_index

    def forward(
        self,
        _value: Tensor,
        factors: Tensor,
        *,
        exposed: Tensor,
    ) -> InterventionProposal:
        if factors.shape[-1] <= self.factor_index:
            raise ValueError("intervention factors do not contain configured factor_index")
        priority = torch.where(
            exposed,
            factors[..., self.factor_index],
            torch.zeros_like(factors[..., self.factor_index]),
        )
        return InterventionProposal(priority)


class StableTopKIntervention(nn.Module):
    """Select a bounded intervention support from exposed instances."""

    _component_reference: ClassVar[str] = "arti/stable-topk-intervention@1"

    def __init__(self, max_interventions: int) -> None:
        super().__init__()
        if (
            isinstance(max_interventions, bool)
            or not isinstance(max_interventions, int)
            or max_interventions <= 0
        ):
            raise ValueError("max_interventions must be a positive integer")
        self.max_interventions = max_interventions

    def forward(self, proposal: InterventionProposal, exposed: Tensor) -> Tensor:
        if not isinstance(proposal, InterventionProposal):
            raise TypeError("intervention policy must return InterventionProposal")
        priority = proposal.priority
        if priority.shape != exposed.shape or not priority.is_floating_point():
            raise ValueError("intervention priority must be floating and match exposed support")
        if self.max_interventions > priority.shape[-1]:
            raise ValueError("max_interventions exceeds active workspace length")
        invalid = exposed & ~torch.isfinite(priority)
        if torch.compiler.is_compiling():
            torch._assert_async(~invalid.any(), "exposed intervention priority must be finite")
        elif bool(invalid.any()):
            raise ValueError("exposed intervention priority must be finite")
        ranked = torch.where(
            exposed,
            priority,
            torch.full_like(priority, torch.finfo(priority.dtype).min),
        )
        order = torch.argsort(ranked, dim=-1, descending=True, stable=True)
        index = order[..., : self.max_interventions]
        selected = torch.gather(exposed, -1, index)
        return torch.zeros_like(exposed).scatter(-1, index, selected)


class FormulaAttention(nn.Module):
    """Select the active instances that may receive Formula-induced change."""

    _component_reference: ClassVar[str] = "arti/formula-attention@1"

    def __init__(self, policy: nn.Module, operator: StableTopKIntervention) -> None:
        super().__init__()
        if not isinstance(policy, nn.Module):
            raise TypeError("intervention policy must be an nn.Module")
        if not isinstance(operator, StableTopKIntervention):
            raise TypeError("FormulaAttention requires StableTopKIntervention@1")
        self.policy = policy
        self.operator = operator

    def forward(
        self,
        workspace: ActiveWorkspace,
        factors: Tensor | None = None,
    ) -> ActiveWorkspace:
        if not isinstance(workspace, ActiveWorkspace):
            raise TypeError("FormulaAttention requires ActiveWorkspace")
        if factors is None:
            factors = workspace.value.new_empty((*workspace.value.shape[:-1], 0))
        if factors.shape[:-1] != workspace.value.shape[:-1]:
            raise ValueError("intervention factors must share active prefix shape")
        if factors.device != workspace.value.device:
            raise ValueError("intervention factors must share active device")
        if not factors.is_floating_point():
            raise TypeError("intervention factors must be floating point")
        if torch.compiler.is_compiling():
            torch._assert_async(
                ~workspace.intervened.any(),
                "FormulaAttention requires empty incoming intervention support",
            )
        elif bool(workspace.intervened.any()):
            raise ValueError(
                "FormulaAttention requires empty incoming intervention support; "
                "disable it to use externally supplied intervention support"
            )
        redacted = torch.where(
            workspace.exposed.unsqueeze(-1),
            workspace.value,
            torch.zeros_like(workspace.value),
        )
        redacted_factors = torch.where(
            workspace.exposed.unsqueeze(-1),
            factors,
            torch.zeros_like(factors),
        )
        proposal = self.policy(redacted, redacted_factors, exposed=workspace.exposed)
        intervened = self.operator(proposal, workspace.exposed)
        return workspace.replace(intervened=intervened)


@dataclass(frozen=True)
class SelectiveComputeInfo:
    """Runtime-only record of the physically packed Formula workspace."""

    query_index: Tensor
    source_index: Tensor
    query_mask: Tensor
    source_mask: Tensor
    active_length: int
    query_length: int
    source_length: int


class ScaleShiftFormula(nn.Module):
    """Reference next-state Formula using per-feature scale and shift factors."""

    _component_reference: ClassVar[str] = "arti/scale-shift-formula@1"

    def __init__(self, dim: int) -> None:
        super().__init__()
        if dim <= 0:
            raise ValueError("dim must be positive")
        self.dim = int(dim)

    def forward(
        self,
        query: Tensor,
        _source: Tensor,
        factors: Tensor,
        *,
        query_mask: Tensor,
        source_mask: Tensor,
        visibility: Tensor | None = None,
    ) -> Tensor:
        del source_mask, visibility
        if query.shape[-1] != self.dim or factors.shape != (
            *query.shape[:-1],
            2 * self.dim,
        ):
            raise ValueError("ScaleShiftFormula factors must end with 2 * dim")
        scale, shift = factors.split(self.dim, dim=-1)
        candidate = query * (1.0 + scale) + shift
        return torch.where(query_mask.unsqueeze(-1), candidate, query)


class SelectiveCompute(nn.Module):
    """Pack I/E supports, run a bounded kernel, and return an active-only update."""

    _component_reference: ClassVar[str] = "arti/selective-compute@1"

    def __init__(
        self,
        kernel: nn.Module,
        *,
        max_queries: int,
        max_sources: int | None = None,
        limits: ContractLimits = DEFAULT_CONTRACT_LIMITS,
    ) -> None:
        super().__init__()
        if not isinstance(kernel, nn.Module):
            raise TypeError("kernel must be an nn.Module")
        from .component_registry import get_component_registry

        registration = get_component_registry().registration_for(kernel)
        if registration is None or "selective.compute.kernel" not in registration.capabilities:
            raise ValueError("kernel must declare selective.compute.kernel capability")
        source_count = max_queries if max_sources is None else max_sources
        for value, name in ((max_queries, "max_queries"), (source_count, "max_sources")):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
            if value > limits.max_dimension:
                raise ValueError(f"{name} exceeds max_dimension")
        if limits is not DEFAULT_CONTRACT_LIMITS:
            for name, hard_value in DEFAULT_CONTRACT_LIMITS.__dict__.items():
                if getattr(limits, name) > hard_value:
                    raise ValueError(f"SelectiveCompute limits cannot relax {name}")
        self.kernel = kernel
        self.max_queries = max_queries
        self.max_sources = source_count
        self.limits = limits

    @staticmethod
    def _admit_count(support: Tensor, limit: int, *, name: str) -> None:
        overflow = support.sum(dim=-1) > limit
        if torch.compiler.is_compiling():
            torch._assert_async(~overflow.any(), f"{name} support exceeds K")
        elif bool(overflow.any()):
            raise ValueError(f"{name} support exceeds configured K")

    @staticmethod
    def _pack(value: Tensor, support: Tensor, length: int) -> tuple[Tensor, Tensor, Tensor]:
        order = torch.argsort(support.to(torch.int8), dim=-1, descending=True, stable=True)
        index = order[..., :length]
        gather_index = index.unsqueeze(-1).expand(*index.shape, value.shape[-1])
        return torch.gather(value, -2, gather_index), torch.gather(support, -1, index), index

    def forward(
        self,
        workspace: ActiveWorkspace,
        factors: Tensor | None = None,
        *,
        visibility: Tensor | None = None,
        formula_route: object | None = None,
        return_info: bool = False,
    ) -> ActiveWorkspace | tuple[ActiveWorkspace, SelectiveComputeInfo]:
        if not isinstance(workspace, ActiveWorkspace):
            raise TypeError("SelectiveCompute requires ActiveWorkspace")
        if formula_route is not None:
            raise ValueError("SelectiveCompute does not consume formula_route")
        self.limits.admit_tensor(workspace.value, name="SelectiveCompute active")
        if factors is None:
            factors = workspace.value.new_empty((*workspace.value.shape[:-1], 0))
        self.limits.admit_tensor(factors, name="SelectiveCompute factors")
        if factors.shape[:-1] != workspace.value.shape[:-1] or factors.device != workspace.value.device:
            raise ValueError("factors must share active prefix shape and device")
        active_length = workspace.value.shape[-2]
        if self.max_queries > active_length or self.max_sources > active_length:
            raise ValueError("configured packed lengths must not exceed active length")
        prefix_count = workspace.value.numel() // (active_length * workspace.value.shape[-1])
        query_values = prefix_count * self.max_queries * workspace.value.shape[-1]
        source_values = prefix_count * self.max_sources * workspace.value.shape[-1]
        query_factors = prefix_count * self.max_queries * factors.shape[-1]
        operation_bytes = (
            workspace.value.numel() * workspace.value.element_size() * 2
            + factors.numel() * factors.element_size()
            + (query_values * 3 + source_values) * workspace.value.element_size()
            + query_factors * factors.element_size()
        )
        if operation_bytes > self.limits.max_operation_bytes:
            raise ValueError("SelectiveCompute exceeds max_operation_bytes")
        self._admit_count(workspace.intervened, self.max_queries, name="query")
        self._admit_count(workspace.exposed, self.max_sources, name="source")

        query, query_mask, query_index = self._pack(
            workspace.value, workspace.intervened, self.max_queries
        )
        source, source_mask, source_index = self._pack(
            workspace.value, workspace.exposed, self.max_sources
        )
        if not torch.compiler.is_compiling() and not bool(query_mask.any()):
            if not return_info:
                return workspace
            return workspace, SelectiveComputeInfo(
                query_index=query_index,
                source_index=source_index,
                query_mask=query_mask,
                source_mask=source_mask,
                active_length=active_length,
                query_length=self.max_queries,
                source_length=self.max_sources,
            )
        factor_index = query_index.unsqueeze(-1).expand(*query_index.shape, factors.shape[-1])
        packed_factors = torch.gather(factors, -2, factor_index)
        packed_visibility = None
        if visibility is not None:
            self.limits.admit_tensor(visibility, name="SelectiveCompute visibility")
            expected = (*workspace.value.shape[:-2], active_length, active_length)
            if visibility.dtype != torch.bool or visibility.shape != expected:
                raise ValueError("visibility must be boolean over the active workspace")
            rows = torch.gather(
                visibility,
                -2,
                query_index.unsqueeze(-1).expand(*query_index.shape, active_length),
            )
            packed_visibility = torch.gather(
                rows,
                -1,
                source_index.unsqueeze(-2).expand(
                    *source_index.shape[:-1], self.max_queries, self.max_sources
                ),
            )

        candidate = self.kernel(
            query,
            source,
            packed_factors,
            query_mask=query_mask,
            source_mask=source_mask,
            visibility=packed_visibility,
        )
        if not isinstance(candidate, Tensor) or candidate.shape != query.shape:
            raise ValueError("selective kernel must return a replacement query tensor")
        committed = torch.where(query_mask.unsqueeze(-1), candidate, query)
        scatter_index = query_index.unsqueeze(-1).expand(
            *query_index.shape, workspace.value.shape[-1]
        )
        active = workspace.value.scatter(-2, scatter_index, committed)
        result = workspace.replace(value=active)
        if not return_info:
            return result
        return result, SelectiveComputeInfo(
            query_index=query_index,
            source_index=source_index,
            query_mask=query_mask,
            source_mask=source_mask,
            active_length=active_length,
            query_length=self.max_queries,
            source_length=self.max_sources,
        )


__all__ = [
    "ActiveWorkspace",
    "FactorInterventionPolicy",
    "FoldedWorkspace",
    "FormulaAttention",
    "InterventionProposal",
    "MagnitudeInterventionPolicy",
    "ScaleShiftFormula",
    "SelectiveCompute",
    "SelectiveComputeInfo",
    "StableTopKIntervention",
]
