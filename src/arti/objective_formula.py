"""Objective-controlled commit strength for the existing Formula executor."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from torch import Tensor, nn

from .formula_fabric import (
    FormulaCommitBlend,
    FormulaFabricCompute,
    FormulaRoutePlan,
    RoutedFormulaFabricCompute,
)
from .objective_bank import ObjectiveExposureBank, ObjectiveExposureOutput
from .runtime_contracts import ContractLimits


@dataclass(frozen=True)
class ObjectiveFormulaFabricComputeInfo:
    """Objective diagnostics paired with the real Formula execution trace."""

    objective: ObjectiveExposureOutput
    commit_weights: Tensor
    formula: object
    _component_reference: ClassVar[str] = (
        "arti/objective-formula-fabric-compute-info@1"
    )


class ObjectiveFormulaFabricCompute(nn.Module):
    """Use bounded Objective exposure as Formula commit strength.

    The adapter does not choose routes, Formula primitives, masks, or execution
    depth. It delegates all Formula work to an existing executor and only
    supplies one current/past-conditioned commit weight per batch item.
    """

    _component_reference: ClassVar[str] = "arti/objective-formula-fabric-compute@1"
    objective_query_contract: ClassVar[str] = "required"

    def __init__(
        self,
        compute: FormulaFabricCompute | RoutedFormulaFabricCompute,
        objective: ObjectiveExposureBank,
    ) -> None:
        super().__init__()
        if not isinstance(compute, (FormulaFabricCompute, RoutedFormulaFabricCompute)):
            raise TypeError(
                "compute must be FormulaFabricCompute or RoutedFormulaFabricCompute"
            )
        if not isinstance(objective, ObjectiveExposureBank):
            raise TypeError("objective must be ObjectiveExposureBank@1")
        base = compute.compute if isinstance(compute, RoutedFormulaFabricCompute) else compute
        if not isinstance(base.fabric, FormulaCommitBlend):
            raise ValueError(
                "Objective Formula control requires FormulaCommitBlend@1"
            )
        self.compute = compute
        self.objective = objective
        self.active_count = compute.active_count
        self.program_fingerprint = base.fabric.program.fingerprint
        self.program_steps = len(base.fabric.program.steps)
        self.max_cells = base.fabric.program.max_cells
        self.factor_contract = "objective-exposure"
        self.route_contract = (
            "bound-source-or-explicit-override"
            if isinstance(compute, RoutedFormulaFabricCompute)
            else compute.route_contract
        )
        self.limits: ContractLimits = base.fabric.limits

    def operation_bytes_upper_bound(
        self,
        workspace: object,
        objective_query: Tensor,
        *,
        formula_route: FormulaRoutePlan | None = None,
        return_info: bool = False,
    ) -> int:
        """Bound the complete Objective plus Formula operation before execution."""

        from .formula_attention import ActiveWorkspace

        if not isinstance(workspace, ActiveWorkspace):
            raise TypeError("ObjectiveFormulaFabricCompute requires ActiveWorkspace")
        flat_batch = workspace.value.numel() // (
            self.active_count * workspace.value.shape[-1]
        )
        self.limits.admit_tensor(
            objective_query, name="Objective Formula query"
        )
        route_elements = flat_batch * self.objective.slots
        factor_elements = flat_batch * self.program_steps * self.max_cells
        if max(route_elements, factor_elements) > self.limits.max_elements:
            raise ValueError("Objective Formula control exceeds allocation limits")
        itemsize = workspace.value.element_size()
        if max(route_elements, factor_elements) * itemsize > self.limits.max_tensor_bytes:
            raise ValueError("Objective Formula control exceeds tensor byte limits")

        base = (
            self.compute.compute
            if isinstance(self.compute, RoutedFormulaFabricCompute)
            else self.compute
        )
        formula_bytes = base.operation_bytes_upper_bound(workspace)
        formula_bytes += factor_elements * itemsize * 2
        route_bytes = 0
        if isinstance(self.compute, RoutedFormulaFabricCompute) and formula_route is None:
            route_bytes = self.compute.route_source.operation_bytes_upper_bound(workspace)
        # logits, softmax route, operand/exposure, expanded commit factors, and
        # the detached diagnostic copies requested by the public info path.
        objective_bytes = route_elements * itemsize * 4
        objective_bytes += flat_batch * itemsize * 3
        objective_bytes += factor_elements * itemsize
        if return_info:
            objective_bytes += route_elements * itemsize
            objective_bytes += factor_elements * itemsize
        return int(formula_bytes + route_bytes + objective_bytes)

    def _admit_operation(
        self,
        workspace: object,
        objective_query: Tensor,
        *,
        formula_route: FormulaRoutePlan | None,
        return_info: bool,
    ) -> None:
        operation_bytes = self.operation_bytes_upper_bound(
            workspace,
            objective_query,
            formula_route=formula_route,
            return_info=return_info,
        )
        operation_limit = self.limits.max_operation_bytes
        if isinstance(self.compute, RoutedFormulaFabricCompute):
            operation_limit = min(
                operation_limit,
                self.compute.route_source.limits.max_operation_bytes,
            )
        if operation_bytes > operation_limit:
            raise ValueError(
                "Objective Formula compute exceeds cumulative operation byte limits"
            )

    def _commit_weights(
        self,
        workspace: object,
        objective_query: Tensor,
        *,
        return_info: bool,
    ) -> tuple[Tensor, ObjectiveExposureOutput | None]:
        from .formula_attention import ActiveWorkspace

        if not isinstance(workspace, ActiveWorkspace):
            raise TypeError("ObjectiveFormulaFabricCompute requires ActiveWorkspace")
        flat_batch = workspace.value.numel() // (
            self.active_count * workspace.value.shape[-1]
        )
        if (
            not isinstance(objective_query, Tensor)
            or not objective_query.is_floating_point()
            or objective_query.shape != (flat_batch, self.objective.query_dim)
        ):
            raise ValueError(
                "objective_query must have shape "
                f"[{flat_batch}, {self.objective.query_dim}]"
            )
        if objective_query.device != workspace.value.device:
            raise ValueError("objective_query and workspace must share device")
        if objective_query.dtype != workspace.value.dtype:
            raise ValueError("objective_query and workspace must share dtype")
        if return_info:
            objective = self.objective(objective_query, return_info=True)
            exposure = objective.exposure
        else:
            objective = None
            exposure = self.objective(objective_query)
        weights = exposure.to(workspace.value).reshape(flat_batch, 1, 1)
        weights = weights.expand(flat_batch, self.program_steps, self.max_cells)
        return weights, objective

    def forward(
        self,
        workspace: object,
        factors: Tensor | None = None,
        *,
        objective_query: Tensor | None = None,
        visibility: Tensor | None = None,
        formula_route: FormulaRoutePlan | None = None,
        return_info: bool = False,
    ) -> object:
        if factors is not None:
            raise ValueError(
                "ObjectiveFormulaFabricCompute does not accept external compute_factors"
            )
        if objective_query is None:
            raise ValueError("ObjectiveFormulaFabricCompute requires objective_query")
        self._admit_operation(
            workspace,
            objective_query,
            formula_route=formula_route,
            return_info=return_info,
        )
        commit_weights, objective = self._commit_weights(
            workspace,
            objective_query,
            return_info=return_info,
        )
        if not return_info:
            return self.compute(
                workspace,
                commit_weights,
                visibility=visibility,
                formula_route=formula_route,
            )
        updated, formula = self.compute(
            workspace,
            commit_weights,
            visibility=visibility,
            formula_route=formula_route,
            return_info=True,
        )
        assert objective is not None
        return updated, ObjectiveFormulaFabricComputeInfo(
            objective=objective,
            commit_weights=commit_weights.detach().clone(),
            formula=formula,
        )


__all__ = [
    "ObjectiveFormulaFabricCompute",
    "ObjectiveFormulaFabricComputeInfo",
]
