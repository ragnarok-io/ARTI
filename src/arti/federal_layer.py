"""Program execution boundary for :class:`arti.ARTILayer`."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Mapping

import torch
from torch import Tensor, nn

from .component_registry import canonical_contract_reference
from .federal_tensor_view import FederatedProgram, TensorViewFederalTrace
from .resource_graph import ProgramGraph, ProgramGraphExecution
from .tensor_view import TensorView


def _fingerprint(value: Mapping[str, object]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ProgramLayerResult:
    """Terminal value and optional receipt from one program-layer invocation."""

    value: Tensor
    outputs: Mapping[str, Tensor] | None
    trace: TensorViewFederalTrace | ProgramGraphExecution | None
    view: TensorView | None
    value_identity: bool


class ProgramRuntime(nn.Module):
    """Adapt one tensor boundary to an optional Program or ``ProgramGraph``.

    A missing program is an explicit, parameter-free identity shell. It is
    useful for host insertion before a ProgramGraph region is configured, but it never
    falls back to an AdaptivePulse or any other execution path.
    """

    def __init__(
        self,
        program: FederatedProgram | None = None,
        *,
        graph: ProgramGraph | None = None,
        graph_program_id: str | None = None,
        graph_input_resource_id: str | None = None,
        graph_output_resource_id: str | None = None,
        root_program_id: str | None = None,
        max_levels: int | None = None,
        max_k: int | None = None,
        axis_names: tuple[str, ...] | None = None,
        axis_roles: tuple[str, ...] | None = None,
        value_field: str = "value",
    ) -> None:
        super().__init__()
        if program is not None and not isinstance(program, FederatedProgram):
            raise TypeError("program must be FederatedProgram or None")
        if graph is not None and not isinstance(graph, ProgramGraph):
            raise TypeError("graph must be ProgramGraph or None")
        if program is not None and graph is not None:
            raise ValueError("program and graph are mutually exclusive execution regions")
        graph_fields = (graph_program_id, graph_input_resource_id, graph_output_resource_id)
        if graph is None:
            if any(value is not None for value in graph_fields):
                raise ValueError("graph resource ids require a ProgramGraph")
        else:
            if not all(isinstance(value, str) and value for value in graph_fields):
                raise ValueError(
                    "ProgramGraph execution requires graph_program_id, graph_input_resource_id, "
                    "and graph_output_resource_id"
                )
            assert graph_program_id is not None
            assert graph_input_resource_id is not None
            assert graph_output_resource_id is not None
            graph.program(graph_program_id)
            graph.resource(graph_input_resource_id)
            graph.resource(graph_output_resource_id)
        if root_program_id is not None and not isinstance(root_program_id, str):
            raise TypeError("root_program_id must be a string or None")
        for name, value in (("max_levels", max_levels), ("max_k", max_k)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value <= 0):
                raise ValueError(f"{name} must be a positive integer or None")
        if (axis_names is None) != (axis_roles is None):
            raise ValueError("axis_names and axis_roles must be provided together")
        if axis_names is not None:
            if not axis_names or len(axis_names) != len(axis_roles or ()):
                raise ValueError("axis_names and axis_roles must have equal non-zero length")
            if len(set(axis_names)) != len(axis_names):
                raise ValueError("axis_names must be unique")
            if tuple(axis_roles or ()).count("batch") != 1:
                raise ValueError("axis_roles must contain exactly one batch role")
        if not isinstance(value_field, str) or not value_field:
            raise ValueError("value_field must be a non-empty string")
        self.program = program
        self.graph = graph
        self.graph_program_id = graph_program_id
        self.graph_input_resource_id = graph_input_resource_id
        self.graph_output_resource_id = graph_output_resource_id
        self.root_program_id = root_program_id
        self.max_levels = max_levels
        self.max_k = max_k
        self.axis_names = None if axis_names is None else tuple(axis_names)
        self.axis_roles = None if axis_roles is None else tuple(axis_roles)
        self.value_field = value_field

    @property
    def configured(self) -> bool:
        return self.program is not None or self.graph is not None

    def contract_config(self) -> dict[str, object]:
        program = self.program
        graph = self.graph
        return {
            "program_ref": (
                None
                if program is None
                else canonical_contract_reference(program._component_reference)
            ),
            "program_contract": None if program is None else program.contract_config(),
            "graph_ref": (
                None
                if graph is None
                else canonical_contract_reference(graph._component_reference)
            ),
            "graph_contract": None if graph is None else graph.contract_config(),
            "graph_program_id": self.graph_program_id,
            "graph_input_resource_id": self.graph_input_resource_id,
            "graph_output_resource_id": self.graph_output_resource_id,
            "root_program_id": self.root_program_id,
            "max_levels": self.max_levels,
            "max_k": self.max_k,
            "axis_names": None if self.axis_names is None else list(self.axis_names),
            "axis_roles": None if self.axis_roles is None else list(self.axis_roles),
            "value_field": self.value_field,
        }

    @property
    def contract_fingerprint(self) -> str:
        return _fingerprint(self.contract_config())

    def runtime_provenance(self) -> dict[str, object]:
        return {
            "surface": "program-runtime",
            "runtime_ref": canonical_contract_reference("arti/program-runtime@1"),
            "configured": self.configured,
            "program_ref": None
            if self.program is None
            else canonical_contract_reference(self.program._component_reference),
            "graph_ref": None
            if self.graph is None
            else canonical_contract_reference(self.graph._component_reference),
            "program_contract_fingerprint": self.contract_fingerprint,
            "identity_when_unconfigured": True,
            "graph_execution": self.graph is not None,
            "functional_graph_state": self.graph is not None,
            "terminal_value_field": self.value_field,
        }

    def _axis_spec(self, x: Tensor) -> tuple[tuple[str, ...], tuple[str, ...]]:
        if self.axis_names is None:
            return (
                tuple(f"axis{index}" for index in range(x.ndim)),
                tuple("batch" if index == 0 else "generic" for index in range(x.ndim)),
            )
        if len(self.axis_names) != x.ndim:
            raise ValueError(
                f"Federated layer axis contract has rank {len(self.axis_names)}, got tensor rank {x.ndim}"
            )
        assert self.axis_roles is not None
        return self.axis_names, self.axis_roles

    @staticmethod
    def _normalize_mask(mask: Tensor | None, x: Tensor) -> Tensor | None:
        if mask is None:
            return None
        if not isinstance(mask, Tensor):
            raise TypeError("mask must be a Tensor or None")
        if mask.device != x.device:
            raise ValueError("mask must share the layer input device")
        if tuple(mask.shape) == tuple(x.shape):
            return mask.to(dtype=torch.bool)
        if x.ndim >= 3 and tuple(mask.shape) == tuple(x.shape[:2]):
            expanded = mask.to(dtype=torch.bool)
            return expanded[(...,) + (None,) * (x.ndim - 2)].expand_as(x)
        raise ValueError(
            "Federated layer mask must match the input tensor or its [batch, sequence] prefix"
        )

    def forward(
        self,
        x: Tensor,
        *,
        mask: Tensor | None = None,
        return_trace: bool = False,
    ) -> ProgramLayerResult:
        if not isinstance(x, Tensor) or not x.is_floating_point():
            raise TypeError("ProgramRuntime expects a floating Tensor")
        if self.program is None and self.graph is None:
            return ProgramLayerResult(x, None, None, None, True)

        names, roles = self._axis_spec(x)
        view = TensorView.from_tensor(
            x,
            axis_names=names,
            axis_roles=roles,
            mask=self._normalize_mask(mask, x),
        )
        if self.graph is not None:
            assert self.graph_program_id is not None
            assert self.graph_input_resource_id is not None
            assert self.graph_output_resource_id is not None
            execution = self.graph.execute_program_functional(
                self.graph_program_id,
                input_views={self.graph_input_resource_id: view},
            )
            output_state = next(
                (
                    item
                    for item in execution.state.resources
                    if item.spec.resource_id == self.graph_output_resource_id
                ),
                None,
            )
            if output_state is None:  # pragma: no cover - constructor validates the resource id.
                raise AssertionError("validated ProgramGraph output resource was missing")
            output = output_state.active_view
            value = output.value
            if tuple(value.shape) != tuple(x.shape):
                raise ValueError(
                    "ProgramGraph output shape does not match the ARTILayer boundary: "
                    f"expected {tuple(x.shape)}, got {tuple(value.shape)}"
                )
            if value.dtype != x.dtype or value.device != x.device:
                raise ValueError("ProgramGraph output must preserve ARTILayer dtype and device")
            return ProgramLayerResult(
                value,
                None,
                execution if return_trace else None,
                view,
                torch.equal(value, x),
            )

        assert self.program is not None
        raw = self.program(
            view,
            root_program_id=self.root_program_id,
            max_levels=self.max_levels,
            max_k=self.max_k,
            return_trace=return_trace,
        )
        if return_trace:
            outputs, trace = raw
        else:
            outputs, trace = raw, None
        if self.value_field not in outputs:
            raise ValueError(
                f"Program terminal output does not contain configured value field {self.value_field!r}"
            )
        value = outputs[self.value_field]
        if not isinstance(value, Tensor):
            raise TypeError("Program terminal value must be a Tensor")
        if tuple(value.shape) != tuple(x.shape):
            raise ValueError(
                "Program terminal value shape does not match the ARTILayer boundary: "
                f"expected {tuple(x.shape)}, got {tuple(value.shape)}. "
                "Use a TITO terminal ABI for ARTI.attach; arbitrary-shape "
                "programs run directly through FederatedProgram."
            )
        if value.dtype != x.dtype or value.device != x.device:
            raise ValueError("Federated terminal value must preserve ARTILayer dtype and device")
        return ProgramLayerResult(value, outputs, trace, view, torch.equal(value, x))


# Historical aliases remain module-private compatibility names. New code uses
# the role-oriented names above or :class:`arti.ARTILayer`.
FederatedLayerResult = ProgramLayerResult
FederatedProgramRuntime = ProgramRuntime
FederalLayerResult = ProgramLayerResult
FederalLayerRuntime = ProgramRuntime

__all__ = ["ProgramLayerResult", "ProgramRuntime"]
