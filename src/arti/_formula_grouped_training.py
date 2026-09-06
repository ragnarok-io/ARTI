"""Grouped native Formula arithmetic with sparse output-gradient ownership.

The adapter restores autograd's absent-versus-zero gradient contract around
compiled groups. PyTorch computes every derivative; no Formula law is copied.
"""

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
import gc

import torch
from torch import nn
from torch.fx.experimental.proxy_tensor import make_fx

from .formula_v2 import PreparedFormulaBindings


_GROUPED_TRAINING = ContextVar("arti_grouped_formula_training", default=None)
_DEFAULT_GROUPED_CACHE = OrderedDict()


def automatic_cuda_compilation(device):
    """The automatic compiled path has a narrower floor than base eager ARTI."""
    return device.type == "cuda" and tuple(int(n) for n in torch.__version__.split(".")[:2]) >= (2, 11)


def use_grouped_training(device):
    scope = _GROUPED_TRAINING.get()
    return scope[0] != "native" if scope is not None else automatic_cuda_compilation(device)


@contextmanager
def grouped_formula_training(*, backend="inductor"):
    """Compile numerical groups; keep original candidate/Bank finishing.

    Static programs are cached for this scope; graph-owned scratch receives
    current operand values on every call, never caller-owned Parameters. Higher
    derivatives use eager PyTorch transforms because compiled double backward
    is not supported by this execution path.
    """
    if backend not in {"native", "eager", "aot_eager", "inductor", "captured"}:
        raise ValueError("grouped backend must be native, eager, aot_eager, inductor or captured")
    token = _GROUPED_TRAINING.set((backend, {}))
    try:
        yield
    finally:
        _GROUPED_TRAINING.reset(token)


def grouped_training_cache_stats():
    """Read host-only cache counters at a training/checkpoint boundary."""
    scope = _GROUPED_TRAINING.get()
    cache = _DEFAULT_GROUPED_CACHE if scope is None else scope[1]
    return {
        name: {"buckets": len(plan.buckets), "builds": plan.cache_builds,
               "hits": plan.cache_hits, "evictions": plan.cache_evictions}
        for name, plan in cache.items()
    }


class _NumericGroup(nn.Module):
    def __init__(self, plan):
        super().__init__()
        self.plan = plan

    def row(self, *values):
        return self.plan.forward_checked(PreparedFormulaBindings(
            self.plan.program_fingerprint, self.plan.binding_names, values,
        ))

    def forward(self, *values):
        return torch.vmap(self.row, randomness="error")(*values)


class _GroupVJP(nn.Module):
    def __init__(self, numeric, inputs, outputs):
        super().__init__()
        self.numeric = numeric
        self.input_indices = inputs
        self.output_indices = outputs
        self.binding_count = len(numeric.plan.binding_names)

    def forward(self, *values):
        bindings = values[:self.binding_count]
        cotangents = values[self.binding_count:]

        def selected(*active):
            current = list(bindings)
            for index, value in zip(self.input_indices, active, strict=True):
                current[index] = value
            outputs, _finite = self.numeric(*current)
            return tuple(outputs[index] for index in self.output_indices)

        _, pullback = torch.func.vjp(selected, *(bindings[index] for index in self.input_indices))
        return pullback(cotangents)


class _GroupedPlan:
    def __init__(self, plan, backend):
        self.numeric = _NumericGroup(plan)
        self.backend = backend
        self.buckets = OrderedDict()
        self.cache_hits = self.cache_builds = self.cache_evictions = 0
        self.pullbacks = {}
        self.bindings = len(plan.binding_names)
        self.outputs = len(plan.program.outputs)
        # Track differentiable reachability, not whether a numerical gradient is
        # zero. Identity observation uses states only to determine trajectory size.
        ancestors = {
            binding.name: (frozenset() if binding.value_type.dtype in {"boolean", "int64"} else frozenset((i,)))
            for i, binding in enumerate(plan.program.bindings)
        }
        for instruction in plan.program.instructions:
            inputs = instruction.input_slots
            if plan.program.slot_types[instruction.output_slot].dtype in {"boolean", "int64"}:
                inputs = ()
            elif instruction.atom_ref == "arti/formula-atom-select@1":
                inputs = inputs[1:]
            elif instruction.atom_ref in {"arti/formula-atom-gather@2", "arti/formula-atom-segment@1"}:
                inputs = inputs[:1]
            elif instruction.atom_ref == "arti/formula-atom-scatter@2":
                inputs = (inputs[0], inputs[2])
            if instruction.atom_ref == "arti/formula-atom-observe-identity@1":
                inputs = (inputs[0], inputs[3])
            ancestors[instruction.output_slot] = frozenset().union(*(ancestors[name] for name in inputs))
        self.dependencies = tuple(ancestors[name] for name in plan.program.outputs)
        self.native_fft_training = any(
            instruction.atom_ref == "arti/formula-atom-observe-fourier@1"
            for instruction in plan.program.instructions
        )

    def execute(self, module, values, *, higher_order=False):
        device = values[0].device
        capture = self.backend == "captured"
        if self.backend == "eager" or higher_order or (capture and device.type != "cuda"):
            return module(*values)
        if self.native_fft_training and self.backend in {"aot_eager", "inductor"}:
            # Preserve the existing native FFT backward boundary. CUDA Graph
            # capture can replay those kernels without compiling complex VJPs.
            return module(*values)
        autocast = torch.is_autocast_enabled(device.type)
        amp_dtype = torch.get_autocast_dtype(device.type) if autocast else None
        key = (module, tuple((tuple(value.shape), value.stride(), value.dtype, value.device) for value in values),
               autocast, amp_dtype,
               torch.cuda.current_stream(device).cuda_stream if capture else None)
        if key not in self.buckets:
            if device.type == "cuda" and torch.cuda.is_current_stream_capturing():
                raise RuntimeError("warm up the Formula forward and actual backward outside CUDA capture")
            if capture:
                # The outer Function owns gradients. Capture only native numeric
                # kernels (including the explicit PyTorch VJP), never Parameters.
                samples = tuple(value.detach().clone() for value in values)
                with torch.cuda.device(device), torch.autocast("cuda", enabled=autocast, dtype=amp_dtype, cache_enabled=False):
                    capture_module = make_fx(module)(*samples) if isinstance(module, _GroupVJP) else module
                    # Cyclic collection may destroy an unrelated old CUDA graph.
                    # Defer it only during capture, not over replay or training.
                    collecting = gc.isenabled()
                    gc.disable()
                    try:
                        self.buckets[key] = torch.cuda.make_graphed_callables(
                            lambda *inputs: capture_module(*inputs), samples,
                        )
                    finally:
                        if collecting:
                            gc.enable()
            else:
                # export's symbol validation does not currently unwrap VJP tracking
                # tensors. make_fx traces the same PyTorch transform to ATen first.
                graph = make_fx(module)(*values) if isinstance(module, _GroupVJP) else torch.export.export(module, values, strict=True).module()
                # The caller owns whole-step capture. Internal automatic graphs
                # cannot be replayed while an enclosing graph is being captured.
                options = {"options": {"triton.cudagraphs": False}} if self.backend == "inductor" else {}
                self.buckets[key] = torch.compile(graph, backend=self.backend, fullgraph=True, **options)
            self.cache_builds += 1
            if len(self.buckets) > 32:
                self.buckets.popitem(last=False)
                self.cache_evictions += 1
        else:
            self.cache_hits += 1
        self.buckets.move_to_end(key)
        return self.buckets[key](*values)

    def pullback(self, inputs, outputs):
        key = (inputs, outputs)
        if key not in self.pullbacks:
            self.pullbacks[key] = _GroupVJP(self.numeric, inputs, outputs)
        return self.pullbacks[key]


class _GroupedFormula(torch.autograd.Function):
    @staticmethod
    def forward(ctx, plan, rows, *values):
        ctx.plan, ctx.rows = plan, rows
        ctx.amp_device = values[0].device.type
        ctx.amp_enabled = torch.is_autocast_enabled(ctx.amp_device)
        ctx.amp_dtype = torch.get_autocast_dtype(ctx.amp_device) if ctx.amp_enabled else None
        ctx.save_for_backward(*values)
        ctx.set_materialize_grads(False)
        stacked = tuple(torch.stack(values[index::plan.bindings]) for index in range(plan.bindings))
        outputs, finite = plan.execute(plan.numeric, stacked)
        # Compiled outputs are scratch. Each returned row outlives this call,
        # and must have a separate autograd edge for unused-output detection.
        result = tuple(value[row].clone() for row in range(rows) for value in outputs)
        nondifferentiable = tuple(result[row * plan.outputs + out] for row in range(rows)
            for out, dependencies in enumerate(plan.dependencies)
            if not any(ctx.needs_input_grad[2 + row * plan.bindings + index] for index in dependencies))
        finite = finite.clone()
        ctx.mark_non_differentiable(finite, *nondifferentiable)
        return (*result, finite)

    @staticmethod
    def backward(ctx, *gradients):
        plan, values = ctx.plan, ctx.saved_tensors
        higher_order = torch.is_grad_enabled()
        groups = {}
        for row in range(ctx.rows):
            outputs = tuple(index for index in range(plan.outputs) if gradients[row * plan.outputs + index] is not None)
            dependencies = frozenset().union(*(plan.dependencies[index] for index in outputs))
            inputs = tuple(index for index in range(plan.bindings)
                if index in dependencies and ctx.needs_input_grad[2 + row * plan.bindings + index])
            if inputs:
                groups.setdefault((inputs, outputs), []).append(row)
        result = [None] * len(values)
        for (inputs, outputs), rows in groups.items():
            # No inactive row enters the VJP, including rejected nonfinite rows.
            bindings = tuple(torch.stack(tuple(values[row * plan.bindings + index] for row in rows))
                             for index in range(plan.bindings))
            cotangents = tuple(torch.stack(tuple(gradients[row * plan.outputs + index] for row in rows)) for index in outputs)
            with torch.autocast(ctx.amp_device, enabled=ctx.amp_enabled, dtype=ctx.amp_dtype, cache_enabled=False):
                grads = plan.execute(plan.pullback(inputs, outputs), (*bindings, *cotangents), higher_order=higher_order)
            for index, value in zip(inputs, grads, strict=True):
                # Preserve returned gradients across later graph replay too.
                value = value.clone()
                for position, row in enumerate(rows):
                    result[row * plan.bindings + index] = value[position]
        return (None, None, *result)


def execute_grouped_training(plan, prepared):
    scope = _GROUPED_TRAINING.get()
    backend, cache = ("inductor", _DEFAULT_GROUPED_CACHE) if scope is None else scope
    if backend == "native":
        raise RuntimeError("native execution must bypass grouped differentiation")
    if plan.program_fingerprint not in cache:
        cache[plan.program_fingerprint] = _GroupedPlan(plan, backend)
        if cache is _DEFAULT_GROUPED_CACHE and len(cache) > 32:
            cache.popitem(last=False)
    runtime = cache[plan.program_fingerprint]
    result = _GroupedFormula.apply(runtime, len(prepared), *(value for row in prepared for value in row.values))
    outputs, finite = result[:-1], result[-1]
    return tuple(tuple(outputs[row * runtime.outputs + index] for index in range(runtime.outputs))
                 for row in range(len(prepared))), finite
