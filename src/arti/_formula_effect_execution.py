"""Tensor-only, bounded effect execution for the native candidate backend.

Plans carry no Bank values or ownership. The caller admits operands and keeps
branch-local proposal/commit semantics; only transition arithmetic runs here.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from collections import OrderedDict

import torch
from torch import Tensor, nn

from .formula_v3 import _neural_plasticity_tensor_step


_EFFECT_BACKEND: ContextVar[str | None] = ContextVar("arti_tensor_effect_backend", default=None)


@contextmanager
def tensor_effect_execution(*, backend: str = "inductor"):
    """Opt into lowered no-grad search; gradient replay keeps native semantics."""
    if backend not in {"eager", "aot_eager", "inductor"}:
        raise ValueError("effect backend must be eager, aot_eager or inductor")
    token = _EFFECT_BACKEND.set(backend)
    try:
        yield
    finally:
        _EFFECT_BACKEND.reset(token)


class _TensorEffectPlan(nn.Module):
    def __init__(self, atom_ref: str, axis: int, maximum: int, repeats: int, counted: bool):
        super().__init__()
        self.atom_ref = atom_ref
        self.axis = axis
        self.maximum = maximum
        self.repeats = repeats
        self.counted = counted

    def _row(self, state: Tensor, count: Tensor, *operands: Tensor) -> tuple[Tensor, Tensor]:
        if not self.counted:
            result = _neural_plasticity_tensor_step(
                self.atom_ref, state, operands, axis=self.axis, maximum=self.maximum,
            )
            finite = torch.isfinite(state).all() & torch.isfinite(result).all()
            for value in operands:
                finite = finite & torch.isfinite(value).all()
            return result, finite

        hard = count.detach().clamp(0.0, float(self.repeats)).round()
        finite = torch.isfinite(state).all() & ~torch.isnan(count)
        for value in operands:
            finite = finite & ((hard == 0) | torch.isfinite(value).all())
        result = state
        # Fixed loop structure is compiled once. Active counts stay on device;
        # inactive rows never feed invalid operands into a later transition.
        for step in range(self.repeats):
            active = (step < hard) & finite
            safe_state = torch.where(active, result, torch.zeros_like(result))
            safe_operands = tuple(torch.where(active, value, torch.zeros_like(value)) for value in operands)
            successor = _neural_plasticity_tensor_step(
                self.atom_ref, safe_state, safe_operands, axis=self.axis, maximum=self.maximum,
            )
            finite = finite & (~active | torch.isfinite(successor).all())
            result = torch.where(active, successor, result)
        return result, finite

    def forward(self, states: Tensor, counts: Tensor, *operands: Tensor) -> tuple[Tensor, Tensor]:
        if torch.is_grad_enabled():
            raise RuntimeError("lowered effect plans require no_grad; use native gradient replay")
        return torch.vmap(self._row, randomness="error")(states, counts, *operands)


@lru_cache(maxsize=64)
def _effect_plan(atom_ref: str, axis: int, maximum: int, repeats: int, counted: bool, backend: str):
    plan = _TensorEffectPlan(atom_ref, axis, maximum, repeats, counted)
    if backend == "eager":
        return plan
    return _CompiledEffectPlan(plan, backend)


class _CompiledEffectPlan:
    """Compile distinct static buckets, not one Python frame with many guards."""
    def __init__(self, plan, backend):
        self.plan = plan
        self.backend = backend
        self.buckets = OrderedDict()

    def __call__(self, *values):
        if torch.is_grad_enabled():
            raise RuntimeError("lowered effect plans require no_grad; use native gradient replay")
        key = tuple((tuple(value.shape), value.stride(), value.dtype, value.device) for value in values)
        if key not in self.buckets:
            # Public export produces a distinct static tensor graph per bucket.
            # There are no parameters/buffers: all changing Bank data are inputs.
            graph = torch.export.export(self.plan, values, strict=True).graph_module
            options = {"mode": "reduce-overhead"} if self.backend == "inductor" else {}
            self.buckets[key] = torch.compile(graph, backend=self.backend, fullgraph=True, **options)
            if len(self.buckets) > 16:
                self.buckets.popitem(last=False)
        self.buckets.move_to_end(key)
        return self.buckets[key](*values)
