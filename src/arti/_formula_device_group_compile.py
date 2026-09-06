"""Prepared compilation islands for the existing no-grad dispatch groups."""

import inspect

import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.utils._pytree import tree_flatten, tree_unflatten


def _signature(args):
    leaves, spec = tree_flatten(args)
    key = tuple((tuple(v.shape), v.stride(), v.dtype, v.device, v.requires_grad) if isinstance(v, torch.Tensor)
                else v for v in leaves)
    devices = sorted({v.device.type for v in leaves if isinstance(v, torch.Tensor)})
    context = (torch.is_inference_mode_enabled(), tuple(
        (device, torch.is_autocast_enabled(device), torch.get_autocast_dtype(device)) for device in devices
    ))
    return (spec, key, context), leaves, spec


class _CompiledGroup:
    def __init__(self, group, calls, backend):
        self.group = group
        self.backend = backend
        self.calls = {}
        self.graph_nodes = 0
        for args in calls:
            key, leaves, spec = _signature(args)
            if key in self.calls:
                continue
            indices = tuple(i for i, value in enumerate(leaves) if isinstance(value, torch.Tensor))
            constants = tuple(None if i in indices else value for i, value in enumerate(leaves))

            def run(*values, indices=indices, constants=constants, spec=spec):
                current = list(constants)
                for i, value in zip(indices, values, strict=True):
                    current[i] = value
                return group.forward(*tree_unflatten(current, spec))

            # Ordinary groups do not mutate inputs. Keep the actual scratch
            # layout, including expanded/strided views, for compile guards.
            samples = tuple(leaves[i] for i in indices)
            graph = make_fx(run)(*samples)
            if any("_local_scalar_dense" in str(n.target) for n in graph.graph.nodes):
                raise ValueError("compiled dispatch groups cannot read device scalars on the host")
            options = {"options": {"triton.cudagraphs": False}} if backend == "inductor" else {}
            compiled = torch.compile(graph, backend=backend, fullgraph=True, **options)
            compiled(*samples)
            self.calls[key] = compiled
            self.graph_nodes += len(tuple(graph.graph.nodes))

    def prepare_variant(self, args):
        key = _signature(args)[0]
        if key not in self.calls:
            variant = _CompiledGroup(self.group, (args,), self.backend)
            self.calls.update(variant.calls)
            self.graph_nodes += variant.graph_nodes
            if len(self.calls) > 32:
                self.calls.pop(next(iter(self.calls)))

    def __call__(self, *args):
        if torch.is_grad_enabled():
            raise RuntimeError("prepared device search is no-grad; use live Formula replay for gradients")
        key, leaves, _ = _signature(args)
        if key not in self.calls:
            raise ValueError("dispatch tensor metadata changed; prepare compiled groups again")
        return self.calls[key](*(v for v in leaves if isinstance(v, torch.Tensor)))


@torch.no_grad()
def prepare_groups(dispatch, sample_run, group_ids, backend):
    """Observe actual calls in a caller-owned scratch run, then compile once."""
    selected = set(group_ids)
    groups = {g.group_id: g for g in dispatch.groups}
    if not selected <= groups.keys() or any(groups[i].is_effect for i in selected):
        raise ValueError("compiled islands currently require existing ordinary group IDs")
    calls = {i: [] for i in selected}
    handles = []
    dispatch._compiled_groups = {}
    dispatch._compiled_dispatch = False
    try:
        for i in selected:
            def record(module, args, i=i):
                calls[i].append(args)
            handles.append(groups[i].register_forward_pre_hook(record))
        sample_run()
    finally:
        for handle in handles:
            handle.remove()
    if any(not values for values in calls.values()):
        raise ValueError("scratch run did not visit each requested dispatch group")
    compiled = {i: _CompiledGroup(groups[i], calls[i], backend) for i in sorted(selected)}
    dispatch._compiled_groups = compiled
    return {i: {"variants": len(g.calls), "graph_nodes": g.graph_nodes} for i, g in compiled.items()}


@torch.no_grad()
def prepare_dispatch(dispatch, sample_run, backend):
    """Compile the original pure typed dispatch, including output packing."""
    if dispatch.data_layout is None or any(g.is_effect for g in dispatch.groups):
        raise ValueError("whole-dispatch compilation requires ordinary typed groups")
    dispatch._compiled_groups = {}
    dispatch._compiled_dispatch = False
    calls = []
    signature = inspect.signature(dispatch.forward)
    def record(module, args, kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        calls.append(module._normalize_call(*bound.arguments.values()))
    handle = dispatch.register_forward_pre_hook(record, with_kwargs=True)
    try:
        sample_run()
    finally:
        handle.remove()
    if not calls:
        raise ValueError("scratch run did not visit the numerical dispatch")
    compiled = _CompiledGroup(dispatch, calls, backend)
    dispatch._compiled_dispatch = compiled
    return {"variants": len(compiled.calls), "graph_nodes": compiled.graph_nodes}
