"""Program host layer for tensor-in/tensor-out integration."""

from __future__ import annotations

from typing import ClassVar

from torch import Tensor, nn

from .component_registry import canonical_contract_reference
from .federal_layer import ProgramLayerResult, ProgramRuntime
from .federal_tensor_view import FederatedProgram
from .resource_graph import ProgramGraph


class ARTILayer(nn.Module):
    """Tensor boundary whose configured execution region is a Program or graph.

    ``ARTILayer`` is a program shell. A supplied
    :class:`FederatedProgram` owns routing, local iteration, Formula Fabric and
    cross-program traversal. Constructing the layer without a Program creates
    an explicit identity shell for safe host attachment; it never dispatches
    through AdaptivePulse.
    """

    _component_reference: ClassVar[str] = "arti/layer@3"
    output_semantics: ClassVar[str] = "next_state"

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
        self.runtime = ProgramRuntime(
            program,
            graph=graph,
            graph_program_id=graph_program_id,
            graph_input_resource_id=graph_input_resource_id,
            graph_output_resource_id=graph_output_resource_id,
            root_program_id=root_program_id,
            max_levels=max_levels,
            max_k=max_k,
            axis_names=axis_names,
            axis_roles=axis_roles,
            value_field=value_field,
        )

    @property
    def program(self) -> FederatedProgram | None:
        """Return the configured program, or ``None`` for the identity shell."""

        return self.runtime.program

    @property
    def graph(self) -> ProgramGraph | None:
        """Return the configured functional graph, if this layer hosts one."""

        return self.runtime.graph

    @property
    def program_contract_fingerprint(self) -> str:
        """Stable fingerprint for attachment and artifact compatibility checks."""

        return self.runtime.contract_fingerprint

    def contract_config(self) -> dict[str, object]:
        """Return the versioned program declaration for this layer."""

        return self.runtime.contract_config()

    def runtime_provenance(self) -> dict[str, object]:
        """Describe the actual program execution boundary."""

        return {
            "layer_ref": canonical_contract_reference(self._component_reference),
            **self.runtime.runtime_provenance(),
        }

    def run(
        self,
        x: Tensor,
        *,
        mask: Tensor | None = None,
        return_trace: bool = False,
    ) -> ProgramLayerResult:
        """Execute the configured program runtime and retain its typed receipt."""

        return self.runtime(x, mask=mask, return_trace=return_trace)

    def forward(
        self,
        x: Tensor,
        *,
        mask: Tensor | None = None,
        return_info: bool = False,
        return_trace: bool = False,
    ) -> Tensor | tuple[Tensor, ProgramLayerResult]:
        """Return the program terminal value, optionally with its receipt."""

        result = self.run(x, mask=mask, return_trace=return_trace)
        return (result.value, result) if return_info else result.value

    def extra_repr(self) -> str:
        program = self.program
        graph = self.graph
        if graph is not None:
            return f"graph={graph._component_reference}, program={self.runtime.graph_program_id}"
        return (
            "program=identity-shell"
            if program is None
            else f"program={program._component_reference}, roots={program.root_program_ids}"
        )


__all__ = ["ARTILayer", "ProgramLayerResult"]
