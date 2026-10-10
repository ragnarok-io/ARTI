"""Portable Formula program profiles for pretrained-model task recipes."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import Tensor, nn

from .arti_layer import ARTILayer
from .federal_layer import ProgramLayerResult
from .formula_v2 import (
    BankBinding,
    FormulaFabricV2,
    FormulaProgram,
    InputBinding,
    TensorType,
    add,
    contract,
    scalar_map,
)
from .resource_graph import (
    FormulaProgramNode,
    ProgramGraph,
    ResourceGraphCompiler,
    ResourcePort,
    TensorResource,
    TensorResourceSpec,
)
from .tensor_view import TensorView, TensorViewPattern


@dataclass(frozen=True)
class FormulaAdapterConfig:
    rank: int = 32
    activation: str = "silu"
    last_layers: int = 2
    layers: tuple[str, ...] | None = None

    def __post_init__(self):
        if (
            type(self.rank) is not int
            or self.rank <= 0
            or type(self.last_layers) is not int
            or self.last_layers <= 0
        ):
            raise ValueError("adapter rank/last_layers must be positive integers")
        if self.activation not in {"none", "silu", "gelu", "relu"}:
            raise ValueError("adapter activation must be none, silu, gelu or relu")
        if self.layers is not None and (
            not self.layers or len(set(self.layers)) != len(self.layers)
        ):
            raise ValueError("adapter layers must name distinct module paths")


class _FormulaAdapterLayer(ARTILayer):
    """One explicit residual Formula profile with a tensor-only default path.

    The graph is still the canonical portable declaration. Its lowering shares
    FP32 operands; training does not construct views or synchronize for equality
    diagnostics at every host invocation.
    """

    def __init__(self, dim: int, config: FormulaAdapterConfig):
        operands = nn.ParameterDict(
            {
                "down": nn.Parameter(torch.randn(dim, config.rank) / dim**0.5),
                "up": nn.Parameter(torch.zeros(config.rank, dim)),
            }
        )
        hidden = TensorType(("batch", "token", "feature"), ("B", "S", dim))
        x = InputBinding("x", hidden)
        down = BankBinding(
            "down",
            "arti/formula-operand-bank@1",
            "down",
            TensorType(("feature", "rank"), (dim, config.rank)),
        )
        up = BankBinding(
            "up",
            "arti/formula-operand-bank@1",
            "up",
            TensorType(("rank", "feature"), (config.rank, dim)),
        )
        projected = contract(x, down, reduce_axes=(("feature", "feature"),))
        if config.activation != "none":
            projected = scalar_map(projected, mode=config.activation)
        update = contract(projected, up, reduce_axes=(("rank", "rank"),))
        program = FormulaProgram.build(outputs=(add(x, update),))
        pattern = TensorViewPattern(
            min_rank=3, max_rank=3, allowed_axis_roles=("batch", "sequence", "feature")
        )
        resources = [
            TensorResource(
                TensorResourceSpec(name, pattern),
                TensorView.from_tensor(
                    torch.zeros(1, 1, dim),
                    axis_names=("batch", "token", "feature"),
                    axis_roles=("batch", "sequence", "feature"),
                ),
            )
            for name in ("source", "adapted")
        ]
        node = FormulaProgramNode(
            "transform",
            FormulaFabricV2(program),
            input_ports={"x": ResourcePort("source")},
            output_ports={"value": ResourcePort("adapted")},
            output_slots={"value": program.outputs[0]},
            banks={"down": down.bind(operands["down"]), "up": up.bind(operands["up"])},
            axis_roles={"value": {"batch": "batch", "token": "sequence", "feature": "feature"}},
        )
        graph = ProgramGraph(resources, (), nodes=(node,), programs={"adapt": ("transform",)})
        super().__init__(
            graph=graph,
            graph_program_id="adapt",
            graph_input_resource_id="source",
            graph_output_resource_id="adapted",
            axis_names=("batch", "token", "feature"),
            axis_roles=("batch", "sequence", "feature"),
        )
        self.operands = operands
        object.__setattr__(
            self,
            "_plan",
            ResourceGraphCompiler.compile_program_outputs(graph, "adapt", ("adapted",)),
        )

    def run(self, x: Tensor, *, mask=None, return_trace=False):
        if return_trace:
            result = super().run(x.float(), mask=mask, return_trace=True)
            return replace(result, value=result.value.to(x.dtype))
        with torch.autocast(device_type=x.device.type, enabled=False):
            value = self._plan(x.float(), torch.zeros_like(x, dtype=torch.float32))[0]
        return ProgramLayerResult(value.to(x.dtype), None, None, None, False)


def formula_adapter_factory(config: FormulaAdapterConfig):
    def create(spec):
        return _FormulaAdapterLayer(spec.dim, config)

    return create


__all__ = ["FormulaAdapterConfig", "formula_adapter_factory"]
