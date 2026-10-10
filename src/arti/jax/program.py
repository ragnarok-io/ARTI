"""Functional JAX bindings for the shared ARTI resource/program graph."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import hashlib
import json
from types import CodeType
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from ..credit_boundary import CreditBoundaryMode
from ..component_registry import canonical_contract_reference
from ..resource_graph import (
    DifferentiableFabricNode,
    FormulaProgramNode,
    LearnableAffineTransfer,
    MultiPortProgramNode,
    ProgramGraph,
    ProgramJoin,
    ProgramNode,
    ProgramRoute,
    ProgramStage,
    ResourceLifetime,
    ResourcePort,
    TensorResource,
    TensorResourceSpec,
)
from ..tensor_view import AxisDescriptor, TensorView
from .formula import _array, _compute, _scalar_map, bind_formula
from .contracts import register_binding


@dataclass(frozen=True)
class NativeModule:
    """A native init/apply pair with explicit parameters and forward state.

    ``init(key, *inputs)`` returns ``(params, state)``. ``apply`` receives
    those trees, positional input arrays and ``rng=``; it returns
    ``(outputs, next_state)``. Output names are the shared graph's port names.
    """

    init: Callable
    apply: Callable
    input_names: tuple[str, ...] = ("value",)
    output_names: tuple[str, ...] = ("value",)
    requires_rng: bool = False
    mutable_state: bool = True
    state_batch_axis: int | None = None
    parameter_ids: Any = None
    implementation_ref: str | None = None

    def __post_init__(self):
        registration = getattr(self.apply, "__arti_jax_ports__", None)
        if registration is not None:
            object.__setattr__(self, "input_names", registration[0])
            object.__setattr__(self, "output_names", registration[1])
        for names in (self.input_names, self.output_names):
            if not names or len(set(names)) != len(names):
                raise ValueError("native module ports must be non-empty and unique")
        if not callable(self.init) or not callable(self.apply):
            raise TypeError("NativeModule requires callable init and apply")
        if self.implementation_ref is not None:
            object.__setattr__(
                self, "implementation_ref", canonical_contract_reference(self.implementation_ref)
            )

    def outputs(self, raw):
        if isinstance(raw, Mapping):
            result = dict(raw)
        elif len(self.output_names) == 1:
            result = {self.output_names[0]: raw}
        else:
            result = dict(zip(self.output_names, raw, strict=True))
        if set(result) != set(self.output_names):
            raise ValueError("native module outputs do not match its named ports")
        return result


def fabric_layer(*, inputs, outputs):
    """Declare named ports on a native JAX apply function without wrapping it."""
    input_names = tuple(inputs)
    output_names = tuple(outputs)

    def decorate(function):
        function.__arti_jax_ports__ = (input_names, output_names)
        return function

    return decorate


class NativeProgramNode(MultiPortProgramNode):
    """A native module using the ordinary graph's named resource ports."""

    _component_reference = "arti/native-program-node@1"

    def __init__(self, node_id, module, *, input_ports, output_ports):
        if not isinstance(module, NativeModule):
            raise TypeError("NativeProgramNode requires NativeModule")
        if tuple(input_ports) != module.input_names or tuple(output_ports) != module.output_names:
            raise ValueError("native node ports must match the native module declaration")
        super().__init__(node_id, input_ports=input_ports, output_ports=output_ports)
        self.native_module = module

    def invoke_ports(self, inputs):
        raise TypeError("NativeProgramNode executes through arti.jax.bind_resource_graph")

    def contract_config(self):
        return {
            **super().contract_config(),
            "backend": "jax",
            "input_names": self.native_module.input_names,
            "output_names": self.native_module.output_names,
            "implementation_ref": self.native_module.implementation_ref,
            "implementation": _implementation_descriptor(self.native_module),
        }


def as_fabric_node(node_id, module, *, input_ports, output_ports):
    return NativeProgramNode(node_id, module, input_ports=input_ports, output_ports=output_ports)


class NativeModuleResult(NamedTuple):
    outputs: Any
    state: Any
    valid: Any
    masks: Any = None


def _unpack(result):
    if isinstance(result, NativeModuleResult):
        return result.outputs, result.state, result.valid, result.masks
    outputs, state = result
    return outputs, state, jnp.asarray(True), None


def _tree_get(tree, path):
    for key in path:
        tree = tree[key]
    return tree


def _tree_set(tree, path, value, *, remove=False):
    result = dict(tree)
    key, *rest = path
    if rest:
        result[key] = _tree_set(result[key], rest, value, remove=remove)
    elif remove:
        result.pop(key)
    else:
        result[key] = value
    return result


def _identity_leaves(tree, prefix=()):
    if isinstance(tree, Mapping):
        for name, child in tree.items():
            yield from _identity_leaves(child, (*prefix, name))
    elif tree is not None:
        yield prefix, tree


def _code_payload(value):
    if isinstance(value, CodeType):
        return {
            "code": value.co_code.hex(),
            "names": value.co_names,
            "constants": tuple(_code_payload(item) for item in value.co_consts),
        }
    if isinstance(value, (str, int, float, bool, type(None))):
        return value
    if isinstance(value, tuple):
        return tuple(_code_payload(item) for item in value)
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}"}


def _implementation_descriptor(module):
    function = module.apply
    code = getattr(function, "__code__", None)
    payload = _code_payload(code) if code is not None else None
    return {
        "ref": module.implementation_ref,
        "function": f"{getattr(function, '__module__', type(function).__module__)}.{getattr(function, '__qualname__', type(function).__qualname__)}",
        "code_sha256": hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest(),
        "input_names": module.input_names,
        "output_names": module.output_names,
        "mutable_state": module.mutable_state,
        "state_batch_axis": module.state_batch_axis,
        "requires_rng": module.requires_rng,
    }


def tensor_resource(spec: TensorResourceSpec, value, *, axis_names, axis_roles=None):
    """Build a shared graph resource from a native JAX/NumPy example backing."""
    import torch

    array = np.array(value, copy=True)
    dtype = str(array.dtype)
    tensor = torch.from_numpy(array.astype(np.float32) if dtype == "bfloat16" else array)
    if dtype == "bfloat16":
        tensor = tensor.to(torch.bfloat16)
    names = tuple(axis_names)
    roles = (
        tuple(axis_roles)
        if axis_roles is not None
        else ("batch",) + ("feature",) * (array.ndim - 1)
    )
    axes = tuple(
        AxisDescriptor(name, role, size)
        for name, role, size in zip(names, roles, array.shape, strict=True)
    )
    return TensorResource(spec, TensorView(tensor, axes))


@jax.custom_vjp
def credit_boundary(value, scale):
    """Exact forward identity with the shared boundary's explicit VJP scale."""
    return value


def _credit_forward(value, scale):
    return value, scale


def _credit_backward(scale, cotangent):
    return jnp.where(scale == 0, 0, cotangent * scale), jnp.zeros_like(scale)


credit_boundary.defvjp(_credit_forward, _credit_backward)


class ProgramState(NamedTuple):
    values: dict[str, Any]
    defaults: dict[str, Any]
    masks: dict[str, Any]
    default_masks: dict[str, Any]
    mounted: dict[str, Any]
    epochs: dict[str, Any]
    steps: dict[str, Any]
    modules: dict[str, Any]
    joins: dict[str, dict[str, Any]]
    valid: Any

    def mount(self, resource_id, value):
        value = jnp.asarray(value)
        if (
            value.shape != self.values[resource_id].shape
            or value.dtype != self.values[resource_id].dtype
        ):
            raise ValueError("mounting requires the current compiled backing shape and dtype")
        return self._replace(
            values={**self.values, resource_id: value},
            mounted={**self.mounted, resource_id: jnp.asarray(True)},
            epochs={**self.epochs, resource_id: self.epochs[resource_id] + 1},
        )

    def unmount(self, resource_id):
        return self._replace(
            values={**self.values, resource_id: self.defaults[resource_id]},
            masks={**self.masks, resource_id: self.default_masks[resource_id]},
            mounted={**self.mounted, resource_id: jnp.asarray(False)},
            epochs={**self.epochs, resource_id: self.epochs[resource_id] + 1},
        )


class ProgramResult(NamedTuple):
    outputs: dict[str, Any]
    state: ProgramState
    receipt: dict[str, Any]


def _parameter_module(module, names, operation):
    trainable = {}
    fixed = {}
    identities = {}
    for name in names:
        value = getattr(module, name)
        if value is not None:
            target = trainable if getattr(value, "requires_grad", False) else fixed
            target[name] = _array(value)
            if target is trainable:
                identities[name] = id(value)

    def initialize(key, *inputs):
        del key, inputs
        return dict(trainable), dict(fixed)

    def apply(params, state, *inputs, rng=None):
        del rng
        weights = {**{n: jax.lax.stop_gradient(v) for n, v in state.items()}, **params}
        return operation(weights, *inputs), state

    return NativeModule(initialize, apply, mutable_state=False, parameter_ids=identities)


def _torch_module(module):
    """Lower common numeric modules once; execution below contains JAX only."""
    import torch.nn as nn

    if isinstance(module, nn.Linear):
        return _parameter_module(
            module, ("weight", "bias"), lambda p, x: x @ p["weight"].T + p.get("bias", 0)
        )
    if isinstance(module, LearnableAffineTransfer):
        return _parameter_module(module, ("gain", "bias"), lambda p, x: x * p["gain"] + p["bias"])
    if isinstance(module, nn.LayerNorm):
        axes = tuple(range(-len(module.normalized_shape), 0))

        def normalize(params, value):
            compute = _compute(value)
            mean = jnp.mean(compute, axis=axes, keepdims=True)
            variance = jnp.mean(jnp.square(compute - mean), axis=axes, keepdims=True)
            weight = jnp.asarray(params.get("weight", 1), dtype=compute.dtype)
            bias = jnp.asarray(params.get("bias", 0), dtype=compute.dtype)
            result = (compute - mean) * jax.lax.rsqrt(variance + module.eps) * weight + bias
            return result.astype(value.dtype)

        return _parameter_module(module, ("weight", "bias"), normalize)
    modes = {nn.ReLU: "relu", nn.SiLU: "silu", nn.Tanh: "tanh", nn.Sigmoid: "sigmoid"}
    mode = next((mode for kind, mode in modes.items() if isinstance(module, kind)), None)
    if isinstance(module, nn.GELU):

        def operation(x):
            return jax.nn.gelu(x, approximate=module.approximate == "tanh")
    elif mode is not None:

        def operation(x):
            return _scalar_map(x, mode)
    elif isinstance(module, nn.Identity):

        def operation(x):
            return x
    elif isinstance(module, nn.Sequential):
        members = tuple(_torch_module(member) for member in module)

        def initialize(key, value):
            keys = jax.random.split(key, max(len(members), 1))
            params, states = {}, {}
            for index, member in enumerate(members):
                params[str(index)], states[str(index)] = member.init(keys[index], value)
                value, _ = member.apply(
                    params[str(index)], states[str(index)], value, rng=keys[index]
                )
            return params, states

        def apply(params, state, value, rng=None):
            states = {}
            for index, member in enumerate(members):
                key = None if rng is None else jax.random.fold_in(rng, index)
                value, states[str(index)] = member.apply(
                    params[str(index)], state[str(index)], value, rng=key
                )
            return value, states

        return NativeModule(
            initialize,
            apply,
            mutable_state=any(member.mutable_state for member in members),
            parameter_ids={
                str(index): member.parameter_ids for index, member in enumerate(members)
            },
        )
    else:
        raise TypeError(f"{type(module).__name__} needs an explicit NativeModule lowering")
    return NativeModule(
        lambda key, *inputs: ({}, {}),
        lambda params, state, x, rng=None: (operation(x), state),
        mutable_state=False,
    )


def _formula_module(node):
    formula = bind_formula(node.fabric.program)
    initial = {name: operand.value for name, operand in node.banks.items()}
    inputs = tuple(node.input_ports)
    outputs = tuple(node.output_ports)
    floating = {
        name for name, value in initial.items() if value.is_floating_point() and value.requires_grad
    }

    def initialize(key, *values):
        del values
        operands = formula.init(key, bank_initializers=initial)
        return {name: value for name, value in operands.items() if name in floating}, {
            "operands": {name: value for name, value in operands.items() if name not in floating}
        }

    def apply(params, state, *values, rng=None):
        del rng
        fixed = {name: jax.lax.stop_gradient(value) for name, value in state["operands"].items()}
        result = formula.apply_checked({**fixed, **params}, dict(zip(inputs, values, strict=True)))
        values = {name: result.outputs[node.output_slots[name]] for name in outputs}
        return NativeModuleResult(
            values,
            state,
            result.valid,
            {name: jnp.ones(value.shape, dtype=bool) for name, value in values.items()},
        )

    return NativeModule(
        initialize,
        apply,
        inputs,
        outputs,
        mutable_state=False,
        parameter_ids={name: id(value) for name, value in initial.items() if name in floating},
    )


def _node_ports(node):
    if isinstance(node, ProgramNode):
        return {"value": ResourcePort(node.input_resource_id)}, {
            "value": ResourcePort(node.output_resource_id)
        }
    return node.input_ports, node.output_ports


class BoundResourceGraph:
    """A shared graph bound to a native, functional JAX execution region."""

    _component_reference = "arti/jax-resource-graph-binding@1"

    def __init__(
        self,
        graph: ProgramGraph,
        program_id: str,
        *,
        native_modules=None,
        native_transfers=None,
        native_conditions=None,
        example_contexts=None,
    ):
        if not isinstance(graph, ProgramGraph):
            raise TypeError("bind_resource_graph requires the shared ProgramGraph")
        if program_id not in graph._programs and program_id not in graph._loops:
            raise ValueError("program_id must name a declared graph program or loop")
        self.graph = graph
        self.program_id = program_id
        self.source_fingerprint = graph.contract_fingerprint
        self._resources = graph.resources
        self._ports = {name: _node_ports(node) for name, node in graph.nodes.items()}
        self._programs = dict(graph._programs)
        self._loops = dict(graph._loops)
        self._joins = dict(graph._joins)
        for steps in self._programs.values():
            for step in steps:
                if (
                    isinstance(step, ProgramRoute)
                    and step.selection_scope == "sample"
                    and step.execution_mode != "all_candidates"
                ):
                    raise ValueError(
                        "native JAX sample routes require execution_mode='all_candidates'"
                    )
        self._example_contexts = dict(example_contexts or {})
        self._axes = {
            name: tuple(axis.name for axis in resource.resolve().view.axes)
            for name, resource in self._resources.items()
        }
        self._batch_axes = {
            name: resource.resolve().view.batch_axis for name, resource in self._resources.items()
        }
        self._modules: dict[str, NativeModule] = {}
        self._owners: dict[str, str] = {}
        owners = {}
        cache = {}
        native_modules = dict(native_modules or {})

        def lower(node, path):
            if path in native_modules:
                return native_modules[path]
            if isinstance(node, NativeProgramNode):
                return node.native_module
            if isinstance(node, FormulaProgramNode):
                return _formula_module(node)
            if isinstance(node, DifferentiableFabricNode):
                members = {
                    name: lower(node.candidate(name), path + "." + name)
                    for name in node.candidate_ids
                }
                input_names, output_names = tuple(node.input_ports), tuple(node.output_ports)

                def initialize(key, *values):
                    keys = jax.random.split(key, len(members))
                    params, state = {}, {}
                    for index, (name, member) in enumerate(members.items()):
                        params[name], state[name] = member.init(keys[index], *values)
                    return {"logits": _array(node.logits), "candidates": params}, state

                def apply(params, state, *values, rng=None):
                    weights = jax.nn.softmax(params["logits"] / node.temperature)
                    results, next_state, valid = [], {}, jnp.asarray(True)
                    for index, (name, member) in enumerate(members.items()):
                        key = None if rng is None else jax.random.fold_in(rng, index)
                        raw, next_state[name], condition, _ = _unpack(
                            member.apply(params["candidates"][name], state[name], *values, rng=key)
                        )
                        valid = valid & condition
                        results.append(member.outputs(raw))
                    result = {
                        name: sum(weights[index] * item[name] for index, item in enumerate(results))
                        for name in output_names
                    }
                    return NativeModuleResult(
                        result,
                        next_state,
                        valid,
                        {name: jnp.ones(value.shape, dtype=bool) for name, value in result.items()},
                    )

                return NativeModule(
                    initialize,
                    apply,
                    input_names,
                    output_names,
                    any(member.requires_rng for member in members.values()),
                    any(member.mutable_state for member in members.values()),
                    parameter_ids={
                        "logits": id(node.logits),
                        "candidates": {
                            name: member.parameter_ids for name, member in members.items()
                        },
                    },
                )
            if hasattr(node, "module"):
                identity = id(node.module)
                if identity not in cache:
                    member = _torch_module(node.module)
                    input_names, output_names = tuple(node.input_ports), tuple(node.output_ports)
                    cache[identity] = NativeModule(
                        member.init,
                        member.apply,
                        input_names,
                        output_names,
                        member.requires_rng,
                        member.mutable_state,
                        member.state_batch_axis,
                        member.parameter_ids,
                    )
                return cache[identity]
            raise TypeError(f"node {path!r} needs a native_modules binding")

        for name, node in graph.nodes.items():
            module = lower(node, name)
            if not isinstance(module, NativeModule):
                raise TypeError("native_modules must contain NativeModule values")
            input_ports, output_ports = self._ports[name]
            if (
                tuple(input_ports) != module.input_names
                or tuple(output_ports) != module.output_names
            ):
                raise ValueError(f"native module {name!r} ports differ from the shared graph")
            owner = owners.setdefault(id(module), name)
            self._owners[name] = owner
            self._modules[owner] = module
        self._transfers = {}
        self._conditions = {}
        for name, connection in graph.connections.items():
            if connection.transfer is not None:
                self._transfers[name] = (native_transfers or {}).get(name) or _torch_module(
                    connection.transfer
                )
            if connection.activation is not None:
                self._conditions[name] = (native_conditions or {}).get(name) or _torch_module(
                    connection.activation
                )
            if connection.operand_views:
                raise TypeError(
                    "connections with extra operand views need a native graph node binding"
                )
        self._module_requires_rng = any(
            module.requires_rng
            for modules in (self._modules, self._transfers, self._conditions)
            for module in modules.values()
        )
        self._stochastic_edges = {
            name
            for name, edge in graph.connections.items()
            if edge.credit_boundary is not None
            and edge.credit_boundary.mode is CreditBoundaryMode.BERNOULLI
        }
        self.requires_rng = self._module_requires_rng or bool(self._stochastic_edges)
        identities = {}
        for section, modules in (
            ("nodes", self._modules),
            ("transfers", self._transfers),
            ("conditions", self._conditions),
        ):
            for name, module in modules.items():
                for path, identity in _identity_leaves(module.parameter_ids):
                    identities.setdefault(identity, []).append((section, name, *path))
        for name, edge in graph.connections.items():
            if edge.credit_boundary is not None and edge.credit_boundary.alpha.requires_grad:
                identities.setdefault(id(edge.credit_boundary.alpha), []).append(
                    ("boundaries", name)
                )
        self._aliases = {
            path: ".".join(paths[0])
            for paths in identities.values()
            if len(paths) > 1
            for path in paths
        }
        descriptor = {
            "source": self.source_fingerprint,
            "binding_ref": canonical_contract_reference(self._component_reference),
            "program_id": program_id,
            "modules": {
                name: _implementation_descriptor(module) for name, module in self._modules.items()
            },
            "transfers": {
                name: _implementation_descriptor(module) for name, module in self._transfers.items()
            },
            "conditions": {
                name: _implementation_descriptor(module)
                for name, module in self._conditions.items()
            },
            "parameter_aliases": {".".join(path): key for path, key in self._aliases.items()},
        }
        self.fingerprint = hashlib.sha256(
            json.dumps(descriptor, sort_keys=True).encode()
        ).hexdigest()

    def _parameters(self, params, section, owner):
        value = params[section].get(owner)
        prefix = (section, owner)
        for path, shared in self._aliases.items():
            if path[:2] == prefix:
                if len(path) == 2:
                    value = params["shared"][shared]
                else:
                    value = _tree_set(value, path[2:], params["shared"][shared])
        return value

    def _boundary_alpha(self, params, state, name):
        if name in params["boundaries"] or ("boundaries", name) in self._aliases:
            return self._parameters(params, "boundaries", name)
        return jax.lax.stop_gradient(state.modules["boundaries"][name])

    def _credit_samples(self, params, state, key, supplied):
        result = dict(supplied)
        for index, name in enumerate(sorted(self._stochastic_edges)):
            if name not in result:
                edge = self.graph.connections[name]
                target = self._read(state, edge.destination, edge.destination_view)
                probability = jax.nn.sigmoid(self._boundary_alpha(params, state, name))
                result[name] = jax.random.bernoulli(
                    jax.random.fold_in(key, index), probability, target.shape
                )
        return result

    def _batch_mask(self, mask, value, resource_id):
        shape = [1] * value.ndim
        shape[self._batch_axes[resource_id]] = mask.shape[0]
        return mask.reshape(shape)

    def _read(self, state, port, region=None):
        value = (
            state.values[port.resource_id]
            if port.port == "value"
            else state.masks[port.resource_id]
        )
        if port.port not in {"value", "mask"}:
            raise ValueError("native resource ports must be value or mask")
        return value[self._slices(region, value)] if region is not None else value

    def _slices(self, region, value):
        ranges = {item.axis: item for item in region.ranges}
        return tuple(
            ranges[axis].resolve(size) if axis in ranges else slice(None)
            for axis, size in zip(self._axes[region.resource_id], value.shape, strict=True)
        )

    def _validate_resource(self, name, value, expected):
        pattern = self._resources[name].spec.view_pattern
        if value.shape != expected.shape:
            raise ValueError(
                f"resource {name!r} output must preserve its compiled shape {expected.shape}"
            )
        if value.dtype != expected.dtype:
            raise ValueError(
                f"resource {name!r} output must preserve its compiled dtype {expected.dtype}"
            )
        if not pattern.min_rank <= value.ndim <= pattern.max_rank:
            raise ValueError(f"resource {name!r} rank is outside its declared contract")
        if pattern.dtype not in {"floating", "any"} and value.dtype != jnp.dtype(pattern.dtype):
            raise ValueError(f"resource {name!r} dtype is outside its declared contract")
        for axis, capacity in self._resources[name].spec.axis_capacity:
            if value.shape[self._axes[name].index(axis)] > capacity:
                raise ValueError(f"resource {name!r} exceeds its axis capacity")

    @staticmethod
    def _mask(value, mask):
        mask = jnp.asarray(mask, dtype=bool)
        while mask.ndim < value.ndim:
            mask = mask[..., None]
        return jnp.broadcast_to(mask, value.shape)

    def init(self, key, example_inputs=None, *, example_contexts=None):
        inputs = dict(example_inputs or {})
        contexts = {**self._example_contexts, **dict(example_contexts or {})}
        values, masks, defaults, default_masks, mounted, epochs, steps = {}, {}, {}, {}, {}, {}, {}
        batch = None
        for name, value in inputs.items():
            batch = jnp.asarray(value).shape[self._batch_axes[name]]
            break
        for name, resource in self._resources.items():
            snapshot = resource.resolve()
            value = _array(snapshot.view.value)
            axis = self._batch_axes[name]
            if batch is None:
                batch = value.shape[axis]
            if value.shape[axis] != batch:
                if value.shape[axis] != 1:
                    raise ValueError("graph resources must share their batch extent")
                shape = list(value.shape)
                shape[axis] = batch
                value = jnp.broadcast_to(value, shape)
            default_view = resource.state().default_view
            default = _array(default_view.value)
            default = jnp.broadcast_to(default, value.shape)
            raw_mask = snapshot.view.mask
            mask = jnp.ones(value.shape, dtype=bool) if raw_mask is None else _array(raw_mask)
            while mask.ndim < value.ndim:
                mask = mask[..., None]
            values[name] = value
            masks[name] = jnp.broadcast_to(mask, value.shape)
            default_mask = (
                jnp.ones(default.shape, dtype=bool)
                if default_view.mask is None
                else _array(default_view.mask)
            )
            while default_mask.ndim < default.ndim:
                default_mask = default_mask[..., None]
            default_masks[name] = jnp.broadcast_to(default_mask, default.shape)
            defaults[name] = default
            mounted[name] = jnp.asarray(snapshot.binding.source == "external")
            epochs[name] = jnp.full((batch,), snapshot.binding.epoch, dtype=jnp.int32)
            steps[name] = jnp.full((batch,), snapshot.binding.step_index, dtype=jnp.int32)
        state = ProgramState(
            values,
            defaults,
            masks,
            default_masks,
            mounted,
            epochs,
            steps,
            {},
            {},
            jnp.asarray(True),
        )
        examples = state._replace(
            values={**values, **{name: jnp.asarray(value) for name, value in inputs.items()}}
        )
        params = {"nodes": {}, "transfers": {}, "conditions": {}, "boundaries": {}}
        module_states = {"nodes": {}, "transfers": {}, "conditions": {}, "boundaries": {}}
        keys = jax.random.split(
            key, max(len(self._modules) + len(self._transfers) + len(self._conditions), 1)
        )
        index = 0
        for name, module in self._modules.items():
            ports = self._ports[name][0]
            params["nodes"][name], module_states["nodes"][name] = module.init(
                keys[index], *(self._read(examples, port) for port in ports.values())
            )
            index += 1
        for section, modules in (("transfers", self._transfers), ("conditions", self._conditions)):
            for name, module in modules.items():
                edge = self.graph.connections[name]
                example = self._read(examples, edge.source, edge.source_view)
                if section == "conditions":
                    if name not in contexts:
                        raise ValueError("conditional connections require example_contexts")
                    example = jnp.asarray(contexts[name])
                params[section][name], module_states[section][name] = module.init(
                    keys[index], example
                )
                index += 1
        for name, edge in self.graph.connections.items():
            if edge.credit_boundary is not None:
                target = (
                    params["boundaries"]
                    if edge.credit_boundary.alpha.requires_grad
                    else module_states["boundaries"]
                )
                target[name] = _array(edge.credit_boundary.alpha)
        shared = {}
        for path, key_name in self._aliases.items():
            shared.setdefault(key_name, _tree_get(params, path))
        for path in self._aliases:
            params = _tree_set(params, path, None, remove=True)
        params["shared"] = shared
        joins = {
            name: {
                port_name: epochs[port.resource_id]
                for port_name, port in self._ports[join.node_id][0].items()
            }
            for name, join in self._joins.items()
        }
        state = state._replace(modules=module_states, joins=joins)
        if self.program_id in self._loops:
            self._validate_batch_state(
                state, self._mutable_owners(self._loops[self.program_id].program_id), batch
            )
        for join in self._joins.values():
            self._validate_batch_state(state, self._mutable_owners(join.node_id), batch)
        return params, state

    def _write(self, state, port, value, region=None, mask=None, active=None):
        name = port.resource_id
        value = jnp.asarray(value)
        if port.port == "value" and not jnp.issubdtype(value.dtype, jnp.floating):
            raise TypeError("shared TensorResource values must be floating-point arrays")
        values, masks = dict(state.values), dict(state.masks)
        target = values[name] if port.port == "value" else masks[name]
        selected = target[self._slices(region, target)] if region is not None else target
        if region is None and port.port == "value":
            self._validate_resource(name, value, target)
        elif value.shape != selected.shape or value.dtype != selected.dtype:
            raise ValueError("a resource-region write must preserve its selected shape and dtype")
        result = target.at[self._slices(region, target)].set(value) if region is not None else value
        if active is not None:
            result = jnp.where(self._batch_mask(active, result, name), result, target)
        if port.port == "value":
            values[name] = result
            if mask is not None:
                payload_mask = self._mask(value, mask)
                next_mask = (
                    masks[name].at[self._slices(region, masks[name])].set(payload_mask)
                    if region is not None
                    else payload_mask
                )
            else:
                next_mask = (
                    masks[name] if region is not None else jnp.ones(result.shape, dtype=bool)
                )
            if active is not None:
                next_mask = jnp.where(
                    self._batch_mask(active, next_mask, name), next_mask, masks[name]
                )
            masks[name] = next_mask
        else:
            masks[name] = result
        defaults, default_masks = dict(state.defaults), dict(state.default_masks)
        persistent = self._resources[name].spec.lifetime is not ResourceLifetime.CALL
        if port.port == "value" and persistent:
            defaults[name] = jnp.where(state.mounted[name], defaults[name], result)
            default_masks[name] = jnp.where(state.mounted[name], default_masks[name], masks[name])
        elif port.port == "mask" and persistent:
            default_masks[name] = jnp.where(state.mounted[name], default_masks[name], result)
        increment = (
            jnp.ones_like(state.epochs[name]) if active is None else active.astype(jnp.int32)
        )
        return state._replace(
            values=values,
            masks=masks,
            defaults=defaults,
            default_masks=default_masks,
            epochs={**state.epochs, name: state.epochs[name] + increment},
            steps={**state.steps, name: state.steps[name] + increment},
        )

    def _invoke(self, name, params, state, key, contexts, credit_masks):
        if name in self._ports:
            inputs, outputs = self._ports[name]
            owner = self._owners[name]
            module = self._modules[owner]
            raw, next_state, valid, output_masks = _unpack(
                module.apply(
                    self._parameters(params, "nodes", owner),
                    state.modules["nodes"][owner],
                    *(self._read(state, port) for port in inputs.values()),
                    rng=key,
                )
            )
            values = module.outputs(raw)
            node = self.graph.nodes[name]
            sources = getattr(getattr(node, "registration", None), "output_sources", {})
            writes = []
            for output_name, port in outputs.items():
                mask = None if output_masks is None else output_masks.get(output_name)
                if mask is None and inputs:
                    source = inputs[sources.get(output_name, next(iter(inputs)))]
                    source_value = self._read(state, source)
                    if source_value.shape == values[output_name].shape:
                        mask = state.masks[source.resource_id]
                writes.append((port, values[output_name], None, mask))
            updates = {("nodes", owner): next_state} if module.mutable_state else {}
            return tuple(writes), updates, valid
        edge = self.graph.connections[name]
        value = self._read(state, edge.source, edge.source_view)
        updates = {}
        valid = jnp.asarray(True)
        transfer_mask = None
        if name in self._transfers:
            raw, next_state, condition, transfer_masks = _unpack(
                self._transfers[name].apply(
                    self._parameters(params, "transfers", name),
                    state.modules["transfers"][name],
                    value,
                    rng=key,
                )
            )
            output_name = self._transfers[name].output_names[0]
            value = self._transfers[name].outputs(raw)[output_name]
            if transfer_masks is not None:
                transfer_mask = (
                    transfer_masks.get(output_name)
                    if isinstance(transfer_masks, Mapping)
                    else transfer_masks
                )
            if self._transfers[name].mutable_state:
                updates[("transfers", name)] = next_state
            valid = valid & condition
        if name in self._conditions:
            raw, next_state, condition, _ = _unpack(
                self._conditions[name].apply(
                    self._parameters(params, "conditions", name),
                    state.modules["conditions"][name],
                    contexts[name],
                    rng=key,
                )
            )
            gate = self._conditions[name].outputs(raw)[self._conditions[name].output_names[0]]
            if self._conditions[name].mutable_state:
                updates[("conditions", name)] = next_state
            valid = valid & condition
            while gate.ndim < value.ndim:
                gate = gate[..., None]
            value = value * gate
        boundary = edge.credit_boundary
        if boundary is not None:
            if boundary.mode is CreditBoundaryMode.OPEN:
                scale = jnp.asarray(1, value.dtype)
            elif boundary.mode is CreditBoundaryMode.CLOSED:
                scale = jnp.asarray(0, value.dtype)
            else:
                alpha = self._boundary_alpha(params, state, name)
                probability = jax.nn.sigmoid(alpha)
                if boundary.mode is CreditBoundaryMode.MEAN:
                    scale = probability
                elif name in credit_masks:
                    scale = credit_masks[name].astype(value.dtype)
                else:
                    scale = jax.random.bernoulli(key, probability, value.shape).astype(value.dtype)
            while scale.ndim < value.ndim:
                scale = scale[..., None]
            value = credit_boundary(value, scale)
        mask = transfer_mask
        if mask is None:
            source_mask = self._read(
                state, ResourcePort(edge.source.resource_id, "mask"), edge.source_view
            )
            mask = (
                source_mask
                if source_mask.shape == value.shape
                else jnp.ones(value.shape, dtype=bool)
            )
        return ((edge.destination, value, edge.destination_view, mask),), updates, valid

    def _install_modules(self, state, updates, active_rows=None):
        modules = {section: dict(values) for section, values in state.modules.items()}
        for (section, owner), value in updates.items():
            if active_rows is not None and jax.tree_util.tree_leaves(value):
                axis = self._module(section, owner).state_batch_axis
                self._validate_batch_state(state, {(section, owner)}, active_rows.shape[0])

                def preserve(next_value, previous):
                    shape = [1] * next_value.ndim
                    shape[axis] = active_rows.shape[0]
                    return jnp.where(active_rows.reshape(shape), next_value, previous)

                value = jax.tree_util.tree_map(preserve, value, state.modules[section][owner])
            modules[section][owner] = value
        return state._replace(modules=modules)

    def _step(
        self, step, params, state, key, contexts, credit_masks, active_rows=None, route_rows=None
    ):
        if isinstance(step, str):
            if step in self._programs:
                return self._program(
                    step, params, state, key, contexts, credit_masks, active_rows, route_rows
                )
            writes, updates, valid = self._invoke(step, params, state, key, contexts, credit_masks)
            result = state
            for port, value, region, mask in writes:
                result = self._write(result, port, value, region, mask, active=active_rows)
            return self._install_modules(result, updates, active_rows)._replace(
                valid=state.valid & valid
            )
        if isinstance(step, ProgramStage):
            writes, updates, valid = [], {}, state.valid
            writers = {}
            read_sets = {}
            for name in step.step_ids:
                if name in self._ports:
                    inputs, outputs = self._ports[name]
                    read_sets[name] = {port.resource_id for port in inputs.values()}
                    destinations = {port.resource_id for port in outputs.values()}
                else:
                    edge = self.graph.connections[name]
                    read_sets[name] = {edge.source.resource_id}
                    destinations = {edge.destination.resource_id}
                    if set(edge.depends_on) & set(step.step_ids):
                        raise ValueError("parallel connections cannot depend on stage peers")
                for destination in destinations:
                    if destination in writers:
                        raise ValueError("ProgramStage writes require distinct resources")
                    writers[destination] = name
            for name, resources in read_sets.items():
                if any(resource in writers and writers[resource] != name for resource in resources):
                    raise ValueError("ProgramStage peers cannot read another peer's publication")
            for index, name in enumerate(step.step_ids):
                pending, module_updates, condition = self._invoke(
                    name,
                    params,
                    state,
                    None if key is None else jax.random.fold_in(key, index),
                    contexts,
                    credit_masks,
                )
                valid = valid & condition
                writes.extend(pending)
                if updates.keys() & module_updates.keys():
                    raise ValueError("parallel nodes cannot implicitly merge shared module state")
                updates.update(module_updates)
            result = state
            for port, value, region, mask in writes:
                result = self._write(result, port, value, region, mask, active=active_rows)
            return self._install_modules(result, updates, active_rows)._replace(valid=valid)
        if isinstance(step, ProgramJoin):
            ports = self._ports[step.node_id][0]
            ready = jnp.ones_like(next(iter(state.epochs.values())), dtype=bool)
            for name, port in ports.items():
                ready = ready & (state.epochs[port.resource_id] > state.joins[step.join_id][name])
            if active_rows is not None:
                ready = ready & active_rows

            def fire(current):
                result = self._step(
                    step.node_id, params, current, key, contexts, credit_masks, ready
                )
                cursors = {
                    name: jnp.where(
                        ready, current.epochs[port.resource_id], current.joins[step.join_id][name]
                    )
                    for name, port in ports.items()
                }
                return result._replace(joins={**result.joins, step.join_id: cursors})

            return jax.lax.cond(jnp.any(ready), fire, lambda current: current, state)
        if isinstance(step, ProgramRoute):
            scores = state.values[step.score_resource_id]
            if scores.shape[-1] != len(step.candidates):
                raise ValueError("route score width must equal the candidate count")
            if step.selection_scope == "batch":
                scoring_rows = active_rows if route_rows is None else route_rows
                if scoring_rows is None:
                    totals = jnp.mean(scores.reshape(-1, len(step.candidates)), axis=0)
                else:
                    weights = scoring_rows.reshape(
                        (scoring_rows.shape[0],) + (1,) * (scores.ndim - 1)
                    )
                    totals = jnp.sum(
                        jnp.where(weights, scores, 0), axis=tuple(range(scores.ndim - 1))
                    )
                index = jnp.argmax(totals)
                if step.execution_mode == "all_candidates":
                    candidates = tuple(
                        self._step(
                            candidate,
                            params,
                            state,
                            key,
                            contexts,
                            credit_masks,
                            active_rows,
                            route_rows,
                        )
                        for candidate in step.candidates
                    )
                    return jax.tree_util.tree_map(
                        lambda *values: jnp.take(jnp.stack(values), index, axis=0), *candidates
                    )
                branches = tuple(
                    lambda current, candidate=candidate: self._step(
                        candidate,
                        params,
                        current,
                        key,
                        contexts,
                        credit_masks,
                        active_rows,
                        route_rows,
                    )
                    for candidate in step.candidates
                )
                return jax.lax.switch(index, branches, state)
            choices = jnp.argmax(scores, axis=-1)
            if choices.ndim != 1:
                raise ValueError("sample routes require scores [batch, candidates]")
            owners = set().union(
                *(self._mutable_owners(candidate) for candidate in step.candidates)
            )
            self._validate_batch_state(state, owners, choices.shape[0])
            cohort = active_rows if route_rows is None else route_rows
            candidates = tuple(
                self._step(
                    candidate,
                    params,
                    state,
                    None if key is None else jax.random.fold_in(key, index),
                    contexts,
                    credit_masks,
                    active_rows,
                    (choices == index) if cohort is None else cohort & (choices == index),
                )
                for index, candidate in enumerate(step.candidates)
            )
            return self._select_candidates(candidates, choices, owners)
        raise TypeError("unsupported shared graph step")

    def _module(self, section, owner):
        return {
            "nodes": self._modules,
            "transfers": self._transfers,
            "conditions": self._conditions,
        }[section][owner]

    def _mutable_owners(self, step):
        if isinstance(step, str):
            if step in self._programs:
                return set().union(
                    *(self._mutable_owners(member) for member in self._programs[step])
                )
            if step in self._ports:
                owner = self._owners[step]
                return {("nodes", owner)} if self._modules[owner].mutable_state else set()
            return {
                (section, step)
                for section, modules in (
                    ("transfers", self._transfers),
                    ("conditions", self._conditions),
                )
                if step in modules and modules[step].mutable_state
            }
        members = (
            step.step_ids
            if isinstance(step, ProgramStage)
            else step.candidates
            if isinstance(step, ProgramRoute)
            else (step.node_id,)
        )
        return set().union(*(self._mutable_owners(member) for member in members))

    def _validate_batch_state(self, state, owners, batch):
        for section, owner in owners:
            leaves = jax.tree_util.tree_leaves(state.modules[section][owner])
            axis = self._module(section, owner).state_batch_axis
            if leaves and (
                axis is None
                or any(value.ndim <= axis or value.shape[axis] != batch for value in leaves)
            ):
                raise ValueError(
                    "per-sample execution requires a declared batch axis on mutable module state"
                )

    def _select_candidates(self, candidates, choices, mutable_owners=()):
        def select(values, axis):
            stacked = jnp.stack(values)
            shape = [1] * stacked.ndim
            shape[axis + 1] = choices.shape[0]
            indices = jnp.broadcast_to(choices.reshape(shape), (1, *stacked.shape[1:]))
            return jnp.take_along_axis(stacked, indices, axis=0)[0]

        result = candidates[0]
        fields = {}
        for field in ("values", "defaults", "masks", "default_masks"):
            fields[field] = {
                name: select(
                    tuple(getattr(state, field)[name] for state in candidates),
                    self._batch_axes[name],
                )
                for name in self._resources
            }
        for field in ("epochs", "steps"):
            fields[field] = {
                name: select(tuple(getattr(state, field)[name] for state in candidates), 0)
                for name in self._resources
            }
        fields["joins"] = {
            join_id: {
                name: select(tuple(state.joins[join_id][name] for state in candidates), 0)
                for name in ports
            }
            for join_id, ports in result.joins.items()
        }
        modules = {section: dict(values) for section, values in result.modules.items()}
        for section, owner in mutable_owners:
            axis = self._module(section, owner).state_batch_axis
            trees = tuple(state.modules[section][owner] for state in candidates)
            if jax.tree_util.tree_leaves(trees[0]):
                modules[section][owner] = jax.tree_util.tree_map(
                    lambda *values: select(values, axis), *trees
                )
        fields["modules"] = modules
        used = jnp.any(choices[None, :] == jnp.arange(len(candidates))[:, None], axis=1)
        fields["valid"] = jnp.all(
            jnp.where(used, jnp.stack(tuple(state.valid for state in candidates)), True)
        )
        return result._replace(**fields)

    def _program(
        self, name, params, state, key, contexts, credit_masks, active_rows=None, route_rows=None
    ):
        result = state
        for index, step in enumerate(self._programs[name]):
            result = self._step(
                step,
                params,
                result,
                None if key is None else jax.random.fold_in(key, index),
                contexts,
                credit_masks,
                active_rows,
                route_rows,
            )
        return result

    def apply(
        self,
        params,
        state: ProgramState,
        inputs=None,
        *,
        rng=None,
        contexts=None,
        credit_masks=None,
        input_masks=None,
    ):
        contexts, credit_masks = dict(contexts or {}), dict(credit_masks or {})
        if rng is None and (
            self._module_requires_rng or self._stochastic_edges - credit_masks.keys()
        ):
            raise ValueError("this program requires an explicit JAX RNG key")
        values, masks = dict(state.values), dict(state.masks)
        for name, resource in self._resources.items():
            if resource.spec.lifetime is ResourceLifetime.CALL:
                values[name] = jnp.where(state.mounted[name], values[name], state.defaults[name])
                masks[name] = jnp.where(state.mounted[name], masks[name], state.default_masks[name])
        current = state._replace(values=values, masks=masks, valid=jnp.asarray(True))
        for name, value in (inputs or {}).items():
            value = jnp.asarray(value)
            self._validate_resource(name, value, current.values[name])
            current = current.mount(name, value)
            mask = (input_masks or {}).get(name, jnp.ones(value.shape, dtype=bool))
            current = current._replace(masks={**current.masks, name: self._mask(value, mask)})
        receipt = {}
        if self.program_id in self._loops:
            loop = self._loops[self.program_id]
            if loop.max_iterations is None:
                raise ValueError("native differentiable loops require a finite max_iterations")
            batch = next(iter(current.epochs.values())).shape[0]
            active = jnp.ones((batch,), dtype=bool)
            counts = jnp.zeros((batch,), dtype=jnp.int32)

            def iteration(carry, index):
                old, enabled, count = carry
                key = None if rng is None else jax.random.fold_in(rng, index)
                candidate = jax.lax.cond(
                    jnp.any(enabled),
                    lambda current: self._program(
                        loop.program_id,
                        params,
                        current,
                        key,
                        contexts,
                        self._credit_samples(params, current, key, credit_masks),
                        enabled,
                    ),
                    lambda current: current,
                    old,
                )
                result = self._select_candidates(
                    (old, candidate),
                    enabled.astype(jnp.int32),
                    self._mutable_owners(loop.program_id),
                )
                continuation = candidate.values[loop.continue_resource_id]
                if continuation.shape != (batch,):
                    raise ValueError(
                        "ProgramLoop continuation must be a rank-1 floating batch mask"
                    )
                continuation = continuation > 0
                enabled = enabled & ((index + 1 < loop.min_iterations) | continuation)
                return (result, enabled, count + carry[1].astype(jnp.int32)), None

            (current, active, counts), _ = jax.lax.scan(
                iteration, (current, active, counts), jnp.arange(loop.max_iterations)
            )
            receipt["iterations"] = counts
            receipt["host_limited"] = active
        else:
            samples = self._credit_samples(params, current, rng, credit_masks)
            current = self._program(self.program_id, params, current, rng, contexts, samples)
        receipt["valid"] = current.valid & jnp.all(
            jnp.stack(tuple(jnp.all(jnp.isfinite(value)) for value in current.values.values()))
        )
        return ProgramResult(current.values, current, receipt)


def bind_resource_graph(graph, program_id, **kwargs):
    return BoundResourceGraph(graph, program_id, **kwargs)


class ARTILayer:
    """Native JAX tensor host with explicit init/apply state."""

    _component_reference = "arti/jax-layer-binding@1"

    def __init__(
        self,
        program: BoundResourceGraph | None = None,
        *,
        input_resource_id=None,
        output_resource_id=None,
    ):
        if program is not None and (input_resource_id is None or output_resource_id is None):
            raise ValueError("configured JAX ARTILayer requires input and output resource ids")
        self.program = program
        self.input_resource_id = input_resource_id
        self.output_resource_id = output_resource_id

    def init(self, key, x):
        if self.program is None:
            return {}, {}
        return self.program.init(key, {self.input_resource_id: x})

    def apply(self, params, state, x, *, rng=None, mask=None):
        if self.program is None:
            return x, state
        result = self.program.apply(
            params,
            state,
            {self.input_resource_id: x},
            rng=rng,
            input_masks=None if mask is None else {self.input_resource_id: mask},
        )
        return result.outputs[self.output_resource_id], result.state


register_binding(
    NativeProgramNode,
    NativeProgramNode._component_reference,
    config_builder=lambda value: value.contract_config(),
)
register_binding(
    BoundResourceGraph,
    BoundResourceGraph._component_reference,
    config_builder=lambda value: {
        "backend": "jax",
        "graph_fingerprint": value.fingerprint,
        "program_id": value.program_id,
    },
)
register_binding(
    ARTILayer,
    ARTILayer._component_reference,
    config_builder=lambda value: {
        "backend": "jax",
        "graph_fingerprint": None if value.program is None else value.program.fingerprint,
        "input_resource_id": value.input_resource_id,
        "output_resource_id": value.output_resource_id,
    },
)


__all__ = [
    "ARTILayer",
    "BoundResourceGraph",
    "NativeModule",
    "NativeModuleResult",
    "NativeProgramNode",
    "ProgramResult",
    "ProgramState",
    "as_fabric_node",
    "bind_resource_graph",
    "credit_boundary",
    "fabric_layer",
    "tensor_resource",
]
