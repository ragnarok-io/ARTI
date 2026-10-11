"""Explicit local Qwen layer factory with two Fabric stages and a workspace resource.

Use reference examples.qwen_program_factory:build_layer from this source checkout,
or install this module in your own project. The deployment package does not copy
Python source. This example has no persistent forward mutation.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch import nn

import arti
from arti import mechanisms as m
from arti.federal_layer import ProgramLayerResult


class WorkspaceProgramLayer(arti.ARTILayer):
    def __init__(self, dim, width):
        if type(width) is not int or width <= 0:
            raise ValueError("width must be a positive integer")
        operands = nn.ParameterDict(
            {
                "encode": nn.Parameter(torch.randn(dim, width) / dim**0.5),
                "decode": nn.Parameter(torch.zeros(width, dim)),
            }
        )
        hidden = m.TensorType(("batch", "token", "feature"), ("B", "S", dim))
        work = m.TensorType(("batch", "token", "work"), ("B", "S", width))
        encode = m.BankBinding(
            "encode",
            "arti/formula-operand-bank@1",
            "encode",
            m.TensorType(("feature", "work"), (dim, width)),
        )
        decode = m.BankBinding(
            "decode",
            "arti/formula-operand-bank@1",
            "decode",
            m.TensorType(("work", "feature"), (width, dim)),
        )
        x = m.InputBinding("x", hidden)
        first = m.FormulaProgram.build(
            outputs=(
                m.scalar_map(
                    m.contract(x, encode, reduce_axes=(("feature", "feature"),)), mode="gelu"
                ),
            )
        )
        h = m.InputBinding("h", work)
        second = m.FormulaProgram.build(
            outputs=(m.add(x, m.contract(h, decode, reduce_axes=(("work", "work"),))),)
        )
        pattern = m.TensorViewPattern(
            min_rank=3, max_rank=3, allowed_axis_roles=("batch", "sequence", "feature")
        )
        resources = [
            m.TensorResource(
                m.TensorResourceSpec(name, pattern),
                m.TensorView.from_tensor(
                    torch.zeros(1, 1, size),
                    axis_names=("batch", "token", axis),
                    axis_roles=("batch", "sequence", "feature"),
                ),
            )
            for name, size, axis in (
                ("source", dim, "feature"),
                ("workspace", width, "work"),
                ("adapted", dim, "feature"),
            )
        ]
        nodes = (
            m.FormulaProgramNode(
                "encode",
                m.FormulaFabricV2(first),
                input_ports={"x": m.ResourcePort("source")},
                output_ports={"h": m.ResourcePort("workspace")},
                output_slots={"h": first.outputs[0]},
                banks={"encode": encode.bind(operands["encode"])},
                axis_roles={"h": {"batch": "batch", "token": "sequence", "work": "feature"}},
            ),
            m.FormulaProgramNode(
                "decode",
                m.FormulaFabricV2(second),
                input_ports={"x": m.ResourcePort("source"), "h": m.ResourcePort("workspace")},
                output_ports={"y": m.ResourcePort("adapted")},
                output_slots={"y": second.outputs[0]},
                banks={"decode": decode.bind(operands["decode"])},
                axis_roles={"y": {"batch": "batch", "token": "sequence", "feature": "feature"}},
            ),
        )
        graph = m.ProgramGraph(resources, (), nodes=nodes, programs={"adapt": ("encode", "decode")})
        super().__init__(
            graph=graph,
            graph_program_id="adapt",
            graph_input_resource_id="source",
            graph_output_resource_id="adapted",
            axis_names=("batch", "token", "feature"),
            axis_roles=("batch", "sequence", "feature"),
        )
        self.operands = operands
        self.width = width
        object.__setattr__(
            self,
            "_plan",
            m.ResourceGraphCompiler.compile_program_outputs(graph, "adapt", ("adapted",)),
        )

    def run(self, x, *, mask=None, return_trace=False):
        if return_trace:
            result = super().run(x.float(), mask=mask, return_trace=True)
            return replace(result, value=result.value.to(x.dtype))
        with torch.autocast(device_type=x.device.type, enabled=False):
            value = x.float()
            workspace = value.new_zeros((*value.shape[:-1], self.width))
            result = self._plan(value, workspace, torch.zeros_like(value))[0]
        return ProgramLayerResult(result.to(x.dtype), None, None, None, False)


def build_layer(spec, *, width=8):
    return WorkspaceProgramLayer(spec.dim, width)
