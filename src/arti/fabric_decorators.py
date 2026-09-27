"""Decorator bridge from ordinary PyTorch modules into the ARTI Fabric graph.

The bridge deliberately leaves a custom layer's numerical implementation in
PyTorch.  It adds named TensorView ports and a declared ABI, so the layer can
participate in a ProgramGraph without being rewritten as Formula IR.  Formula
IR remains the strict static-lowering path; custom modules may later publish a
dedicated lowering without changing this runtime contract.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Callable, Literal, TypeVar

from torch import Tensor, nn

from .resource_graph import (
    DifferentiableFabricNode,
    LocalVJP,
    LocalVJPResult,
    MultiPortProgramNode,
    MultiPortProgramNodeInvocation,
    ResourceGraphError,
    ResourcePort,
    autograd_local_vjp,
)
from .tensor_view import TensorView


_LayerT = TypeVar("_LayerT", bound=type[nn.Module])
_REGISTRATION_ATTRIBUTE = "__arti_fabric_layer_registration__"
_FATE_ATTRIBUTE = "__arti_differentiable_fate_id__"


def _identifier(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value or not value.replace("_", "a").isalnum() or not value[0].isalpha():
        raise ValueError(f"{field} must be an identifier")
    return value


@dataclass(frozen=True)
class FabricLayerRegistration:
    """Named ABI declaration attached to a custom ``nn.Module`` class.

    ``outputs`` maps each output port to the input port whose TensorView
    metadata is reused when the module returns a raw Tensor.  A module that
    changes logical layout must instead return a TensorView for that output.
    """

    input_names: tuple[str, ...]
    output_sources: Mapping[str, str]
    local_vjp: LocalVJP | Literal["autograd"] = "autograd"

    def __post_init__(self) -> None:
        inputs = tuple(self.input_names)
        outputs = dict(self.output_sources)
        if not inputs or not outputs:
            raise ValueError("Fabric layer registrations require input and output ports")
        if len(set(inputs)) != len(inputs):
            raise ValueError("Fabric layer input names must be unique")
        for name in inputs:
            _identifier(name, field="input port")
        for name, source in outputs.items():
            _identifier(name, field="output port")
            if source not in inputs:
                raise ValueError("Fabric layer output sources must name declared input ports")
        if self.local_vjp != "autograd" and not callable(self.local_vjp):
            raise TypeError("Fabric layer local_vjp must be 'autograd' or a callable")
        object.__setattr__(self, "input_names", inputs)
        object.__setattr__(self, "output_sources", outputs)


def fabric_layer(
    *,
    inputs: Sequence[str],
    outputs: Mapping[str, str],
    local_vjp: LocalVJP | Literal["autograd"] = "autograd",
) -> Callable[[_LayerT], _LayerT]:
    """Register an arbitrary ``nn.Module`` class as a runtime Fabric region.

    The decorated class remains an ordinary PyTorch layer.  Construct it as
    usual, then call :func:`as_fabric_node` with resource ports.
    """

    registration = FabricLayerRegistration(tuple(inputs), dict(outputs), local_vjp)

    def decorate(layer_type: _LayerT) -> _LayerT:
        if not isinstance(layer_type, type) or not issubclass(layer_type, nn.Module):
            raise TypeError("fabric_layer decorates nn.Module classes")
        setattr(layer_type, _REGISTRATION_ATTRIBUTE, registration)
        return layer_type

    return decorate


def formula_layer(
    *,
    inputs: Sequence[str],
    outputs: Mapping[str, str],
    local_vjp: LocalVJP | Literal["autograd"] = "autograd",
) -> Callable[[_LayerT], _LayerT]:
    """Alias for :func:`fabric_layer` for formula-like custom modules.

    The name describes graph participation, not Formula IR lowering.  The
    module is still a PyTorch implementation until it supplies a lowering.
    """

    return fabric_layer(inputs=inputs, outputs=outputs, local_vjp=local_vjp)


def differentiable_fate(fate_id: str) -> Callable[[_LayerT], _LayerT]:
    """Mark a registered custom layer as one candidate node fate.

    A fate is only made trainable when instances are assembled with
    :func:`as_differentiable_fabric_node`; the decorator itself creates no
    hidden architecture parameters or execution path.
    """

    canonical = _identifier(fate_id, field="fate_id")

    def decorate(layer_type: _LayerT) -> _LayerT:
        if not isinstance(layer_type, type) or not issubclass(layer_type, nn.Module):
            raise TypeError("differentiable_fate decorates nn.Module classes")
        setattr(layer_type, _FATE_ATTRIBUTE, canonical)
        return layer_type

    return decorate


def fabric_registration(layer: nn.Module | type[nn.Module]) -> FabricLayerRegistration:
    """Return the declared Fabric ABI for a decorated module or module class."""

    layer_type = layer if isinstance(layer, type) else type(layer)
    registration = getattr(layer_type, _REGISTRATION_ATTRIBUTE, None)
    if not isinstance(registration, FabricLayerRegistration):
        raise TypeError("module class is not registered with fabric_layer")
    return registration


class FabricModuleNode(MultiPortProgramNode):
    """A named-port ProgramGraph adapter for one decorated PyTorch module."""

    _component_reference = "arti/fabric-module-node@1"

    def __init__(
        self,
        node_id: str,
        module: nn.Module,
        *,
        input_ports: Mapping[str, ResourcePort],
        output_ports: Mapping[str, ResourcePort],
    ) -> None:
        if not isinstance(module, nn.Module):
            raise TypeError("module must be an nn.Module")
        registration = fabric_registration(module)
        if tuple(input_ports) != registration.input_names:
            raise ResourceGraphError("FabricModuleNode input ports must match the registered order")
        if set(output_ports) != set(registration.output_sources):
            raise ResourceGraphError("FabricModuleNode output ports must match the registered declaration")
        super().__init__(node_id, input_ports=input_ports, output_ports=output_ports)
        self.module = module
        self.registration = registration

    def contract_config(self) -> dict[str, object]:
        return {
            **super().contract_config(),
            "module_type": f"{type(self.module).__module__}.{type(self.module).__qualname__}",
            "input_names": list(self.registration.input_names),
            "output_sources": dict(self.registration.output_sources),
            "local_vjp": "autograd" if self.registration.local_vjp == "autograd" else "custom",
            "fate_id": getattr(type(self.module), _FATE_ATTRIBUTE, None),
        }

    def _normalize_outputs(
        self, raw: object, inputs: Mapping[str, TensorView]
    ) -> Mapping[str, TensorView]:
        output_names = tuple(self.output_ports)
        if isinstance(raw, Mapping):
            values = dict(raw)
            if set(values) != set(output_names):
                raise ResourceGraphError("custom Fabric module returned an incomplete output mapping")
        elif len(output_names) == 1:
            values = {output_names[0]: raw}
        elif isinstance(raw, (tuple, list)) and len(raw) == len(output_names):
            values = dict(zip(output_names, raw, strict=True))
        else:
            raise ResourceGraphError(
                "custom Fabric module with multiple outputs must return a mapping or tuple"
            )
        outputs: dict[str, TensorView] = {}
        for name in output_names:
            value = values[name]
            if isinstance(value, TensorView):
                outputs[name] = value
            elif isinstance(value, Tensor):
                source = inputs[self.registration.output_sources[name]]
                if value.shape != source.value.shape:
                    raise ResourceGraphError(
                        "a raw Tensor output may only reuse its declared source layout; "
                        "return TensorView explicitly for a changed ABI"
                    )
                outputs[name] = TensorView(
                    value, source.axes, index_map=source.index_map, mask=source.mask
                )
            else:
                raise TypeError("custom Fabric module outputs must be Tensor or TensorView")
        return outputs

    def invoke_ports(self, inputs: Mapping[str, TensorView]) -> MultiPortProgramNodeInvocation:
        if tuple(inputs) != self.registration.input_names:
            raise ResourceGraphError("FabricModuleNode received an incomplete input port set")
        if any(not isinstance(view, TensorView) for view in inputs.values()):
            raise TypeError("FabricModuleNode inputs must be TensorView values")
        raw = self.module(*(inputs[name].value for name in self.registration.input_names))
        return MultiPortProgramNodeInvocation(self._normalize_outputs(raw, inputs))

    def lower_static(self) -> StaticFabricModuleNode:
        """Freeze the named ABI as a tensor-only program fragment."""

        return StaticFabricModuleNode(self)


class StaticFabricModuleNode(nn.Module):
    """Tensor-only lowering of one :class:`FabricModuleNode`.

    Port ordering is fixed when the graph is compiled.  This adapter does not
    manufacture TensorView objects during invocation; it merely runs the
    user module and normalizes its declared named result into tuple order.
    """

    def __init__(self, node: FabricModuleNode) -> None:
        super().__init__()
        if not isinstance(node, FabricModuleNode):
            raise TypeError("node must be FabricModuleNode")
        self.module = node.module
        self.input_names = node.registration.input_names
        self.output_names = tuple(node.output_ports)

    def forward(self, *inputs: Tensor) -> tuple[Tensor, ...]:
        if len(inputs) != len(self.input_names):
            raise ResourceGraphError("static Fabric module received an incorrect input count")
        raw = self.module(*inputs)
        if isinstance(raw, Mapping):
            if set(raw) != set(self.output_names):
                raise ResourceGraphError("static Fabric module returned an incomplete output mapping")
            values = tuple(raw[name] for name in self.output_names)
        elif len(self.output_names) == 1:
            values = (raw,)
        elif isinstance(raw, (tuple, list)) and len(raw) == len(self.output_names):
            values = tuple(raw)
        else:
            raise ResourceGraphError(
                "static Fabric module with multiple outputs must return a mapping or tuple"
            )
        tensors: list[Tensor] = []
        for value in values:
            if isinstance(value, TensorView):
                tensors.append(value.value)
            elif isinstance(value, Tensor):
                tensors.append(value)
            else:
                raise TypeError("static Fabric module outputs must be Tensor or TensorView")
        return tuple(tensors)

    def local_vjp(
        self,
        inputs: tuple[Tensor, ...],
        outputs: tuple[Tensor, ...],
        output_cotangents: tuple[Tensor, ...],
        parameters: tuple[nn.Parameter, ...],
        create_graph: bool,
    ) -> LocalVJPResult:
        """Apply this layer's registered local pullback."""

        registration = fabric_registration(self.module)
        if registration.local_vjp == "autograd":
            return autograd_local_vjp(
                self.module,
                inputs,
                outputs,
                output_cotangents,
                parameters,
                create_graph,
            )
        return registration.local_vjp(
            self.module,
            inputs,
            outputs,
            output_cotangents,
            parameters,
            create_graph,
        )


def as_fabric_node(
    node_id: str,
    module: nn.Module,
    *,
    input_ports: Mapping[str, ResourcePort],
    output_ports: Mapping[str, ResourcePort],
) -> FabricModuleNode:
    """Mount one decorated module as a ProgramGraph node."""

    return FabricModuleNode(
        node_id, module, input_ports=input_ports, output_ports=output_ports
    )


def as_differentiable_fabric_node(
    node_id: str,
    candidates: Mapping[str, nn.Module],
    *,
    input_ports: Mapping[str, ResourcePort],
    output_ports: Mapping[str, ResourcePort],
    temperature: float = 1.0,
) -> DifferentiableFabricNode:
    """Assemble decorated module instances into one trainable node-kind choice."""

    nodes: dict[str, FabricModuleNode] = {}
    for candidate_id, module in candidates.items():
        _identifier(candidate_id, field="candidate_id")
        fabric_registration(module)
        fate_id = getattr(type(module), _FATE_ATTRIBUTE, None)
        if fate_id != candidate_id:
            raise ResourceGraphError(
                "differentiable candidate id must match its differentiable_fate registration"
            )
        nodes[candidate_id] = as_fabric_node(
            f"{node_id}_{candidate_id}",
            module,
            input_ports=input_ports,
            output_ports=output_ports,
        )
    return DifferentiableFabricNode(node_id, nodes, temperature=temperature)


__all__ = [
    "FabricLayerRegistration",
    "LocalVJP",
    "LocalVJPResult",
    "FabricModuleNode",
    "StaticFabricModuleNode",
    "as_differentiable_fabric_node",
    "as_fabric_node",
    "differentiable_fate",
    "fabric_layer",
    "fabric_registration",
    "formula_layer",
]
