"""Adapter insertion utilities."""

from __future__ import annotations

import fnmatch
import hashlib
import inspect
import math
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Iterator

import torch
import torch.nn as nn
from torch import Tensor

from ..blocks import ARTIResidualBlock
from ..config import ARTIConfig, STATE_RECALL_COMPOSITION_FACTOR
from ..layers import ARTILatentRecallField
from ..recall_formula import RecallFormulaContract
from ..recall_registry import resolve_formula
from ..tensor_boundary import (
    TensorLayout,
    find_primary_tensor,
    replace_tensor_at_path,
    tensor_at_path,
)
from .profiles import AdapterProfile
from .runtime import current_context
from .scales import AdapterScale, resolve_scale
from .scanner import InsertionCandidate


@dataclass(frozen=True)
class InsertionSpec:
    where: tuple[str, ...] = ("*",)
    exclude: tuple[str, ...] = ()
    positions: tuple[str, ...] = ("output",)
    scale_pattern: tuple[tuple[str, str], ...] = ()
    every: int = 1
    freeze_base: bool = True
    max_adapters: int | None = None
    max_extra_params: int | None = None
    identity_gate: bool = False
    zero_init_output: bool = False
    bridge_mode: str = "radial"
    boundary_mask_key: str | None = None
    require_runtime_context: bool = False


@dataclass(frozen=True)
class InsertedAdapter:
    name: str
    module_path: str
    position: str
    tensor_path: tuple[str | int, ...]
    dim: int
    parameters: int
    profile: str
    scale: str
    bridge_mode: str
    residual_budget: float
    hidden_dim: int = 0
    recall_slots: int = 0
    recall_bank_parameters: int = 0
    recall_bank_fraction: float = 0.0
    recall_routing: str = "dense"
    recall_key_dim: int = 0
    recall_group_size: int = 0
    recall_group_topk: int = 0


@dataclass(frozen=True)
class AdapterInsertionPlan:
    """Dry-run plan for adapter insertion without mutating the model."""

    selected: tuple[InsertedAdapter, ...]
    skipped_budget: tuple[InsertedAdapter, ...]
    excluded: tuple[str, ...]
    spec: InsertionSpec

    @property
    def adapter_parameters(self) -> int:
        return sum(adapter.parameters for adapter in self.selected)

    def to_dict(self) -> dict[str, object]:
        return {
            "selected": [adapter.__dict__ for adapter in self.selected],
            "skipped_budget": [adapter.__dict__ for adapter in self.skipped_budget],
            "excluded": list(self.excluded),
            "adapter_parameters": self.adapter_parameters,
            "spec": {
                "where": list(self.spec.where),
                "exclude": list(self.spec.exclude),
                "positions": list(self.spec.positions),
                "scale_pattern": dict(self.spec.scale_pattern),
                "every": self.spec.every,
                "freeze_base": self.spec.freeze_base,
                "max_adapters": self.spec.max_adapters,
                "max_extra_params": self.spec.max_extra_params,
                "identity_gate": self.spec.identity_gate,
                "zero_init_output": self.spec.zero_init_output,
                "bridge_mode": self.spec.bridge_mode,
                "boundary_mask_key": self.spec.boundary_mask_key,
                "require_runtime_context": self.spec.require_runtime_context,
            },
        }


class ARTIAdapterWrapper(nn.Module):
    """Wrap one exact input or output tensor boundary with ARTI."""

    def __init__(
        self,
        base: nn.Module,
        adapter: ARTIResidualBlock,
        *,
        freeze_base: bool = True,
        identity_gate: bool = False,
        boundary_mask_key: str | None = None,
        require_runtime_context: bool = False,
        batch_axis: int | None = None,
        feature_axis: int | None = None,
        position: str = "output",
        tensor_path: tuple[str | int, ...] | None = None,
    ) -> None:
        super().__init__()
        self.base = base
        self.adapter = adapter
        self.enabled = True
        self.influence_scale = 1.0
        self._route_assignment_override: Tensor | None = None
        self.output_gate = nn.Parameter(torch.zeros(())) if identity_gate else None
        self.boundary_mask_key = boundary_mask_key
        self._boundary_mask_position = None
        if boundary_mask_key is not None:
            try:
                parameters = tuple(inspect.signature(base.forward).parameters.values())
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"cannot inspect host forward for boundary mask {boundary_mask_key!r}"
                ) from exc
            positional = [
                parameter
                for parameter in parameters
                if parameter.kind
                in {inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD}
            ]
            names = {parameter.name for parameter in parameters}
            if boundary_mask_key not in names:
                raise ValueError(
                    f"host forward has no boundary mask parameter {boundary_mask_key!r}"
                )
            self._boundary_mask_position = next(
                (
                    index
                    for index, parameter in enumerate(positional)
                    if parameter.name == boundary_mask_key
                ),
                None,
            )
        self.require_runtime_context = require_runtime_context
        self.batch_axis = batch_axis
        self.feature_axis = feature_axis
        if position not in {"input", "output"}:
            raise ValueError("ARTI adapter position must be 'input' or 'output'")
        self.position = position
        self.tensor_path = tensor_path
        if freeze_base:
            for param in self.base.parameters():
                param.requires_grad = False

    def forward(self, *args, **kwargs):
        boundary_mask = self._resolve_boundary_mask(args, kwargs) if self.enabled else None
        if self.enabled and self.position == "input":
            tree = {"args": args, "kwargs": kwargs}
            tree = self._adapt_tree(tree, boundary_mask=boundary_mask)
            args, kwargs = tree["args"], tree["kwargs"]
        output = self.base(*args, **kwargs)
        if not self.enabled or self.position != "output":
            return output
        return self._adapt_tree(output, boundary_mask=boundary_mask)

    def _resolve_boundary_mask(self, args, kwargs) -> Tensor | None:
        if self.boundary_mask_key is None:
            return None
        value = kwargs.get(self.boundary_mask_key)
        if (
            value is None
            and self._boundary_mask_position is not None
            and self._boundary_mask_position < len(args)
        ):
            value = args[self._boundary_mask_position]
        if not isinstance(value, Tensor):
            raise ValueError(
                f"host boundary mask {self.boundary_mask_key!r} must resolve to a Tensor"
            )
        return value

    def _adapt_tree(self, tree, *, boundary_mask: Tensor | None = None):
        if self.tensor_path is None:
            found = find_primary_tensor(tree)
            if found is None:
                return tree
            tensor, path = found
        else:
            path = self.tensor_path
            tensor = tensor_at_path(tree, path)
        return replace_tensor_at_path(tree, path, self._adapt(tensor, boundary_mask=boundary_mask))

    def _adapt(self, tensor: Tensor, *, boundary_mask: Tensor | None = None) -> Tensor:
        inferred = TensorLayout.infer(tensor, self.base)
        layout = TensorLayout(
            tuple(int(value) for value in tensor.shape),
            inferred.batch_axis if self.batch_axis is None else self.batch_axis,
            inferred.feature_axis if self.feature_axis is None else self.feature_axis,
        )
        return layout.restore(
            self._adapt_sequence(layout.pack(tensor), boundary_mask=boundary_mask)
        )

    def _adapt_sequence(self, tensor: Tensor, *, boundary_mask: Tensor | None = None) -> Tensor:
        context = current_context()
        if self.require_runtime_context and context is None:
            raise ValueError(
                "ARTI pretrained phase adapter requires workflow.context(...) or arti_context= at runtime"
            )
        if (
            self.output_gate is not None
            and not self.training
            and not torch.is_grad_enabled()
            and bool(torch.count_nonzero(self.output_gate.detach()) == 0)
        ):
            return tensor
        resolved_mask = None
        if boundary_mask is not None:
            if tensor.ndim != 3 or boundary_mask.shape != tensor.shape[:2]:
                raise ValueError(
                    f"host boundary mask must have shape {tuple(tensor.shape[:2])}, "
                    f"got {tuple(boundary_mask.shape)}"
                )
            resolved_mask = boundary_mask.to(device=tensor.device, dtype=torch.bool)
        if tensor.ndim != 3 or (
            context is None
            and resolved_mask is None
            and self._route_assignment_override is None
        ):
            return self._blend(tensor, self._forward_adapter(tensor))
        kwargs = {}
        if (
            context is not None
            and context.mask is not None
            and context.mask.shape == tensor.shape[:2]
        ):
            context_mask = context.mask.to(device=tensor.device, dtype=torch.bool)
            resolved_mask = context_mask if resolved_mask is None else resolved_mask & context_mask
        if resolved_mask is not None:
            kwargs["mask"] = resolved_mask
        if (
            context is not None
            and context.visibility is not None
            and context.visibility.shape == (*tensor.shape[:2], tensor.shape[1])
        ):
            kwargs["visibility"] = context.visibility.to(device=tensor.device)
        route_assignment = (
            context.route_assignment
            if context is not None and context.route_assignment is not None
            else self._route_assignment_override
        )
        layer = getattr(self.adapter, "layer", None)
        config = getattr(layer, "config", None)
        supports_recall_route = bool(getattr(self.adapter, "direct_recall", False)) or int(
            getattr(config, "recall_steps", 0)
        ) > 0
        if route_assignment is not None and supports_recall_route:
            if (
                route_assignment.ndim != 2
                or route_assignment.shape[0] != tensor.shape[0]
            ):
                raise ValueError(
                    "route_assignment must have shape [B, R] with the adapter batch size"
                )
            kwargs["route_assignment"] = route_assignment.to(
                device=tensor.device,
                dtype=tensor.dtype,
            )
        coord_dim = int(getattr(config, "coord_dim", 0))
        if coord_dim > 0 and context is not None:
            if context.coord is None or context.coord.shape != (*tensor.shape[:2], coord_dim):
                raise ValueError(
                    f"observer-phase ARTI adapter requires coord with shape {(*tensor.shape[:2], coord_dim)}"
                )
            kwargs["coord"] = context.coord.to(device=tensor.device, dtype=tensor.dtype)
            if context.observer_coord is not None:
                kwargs["observer_coord"] = context.observer_coord.to(
                    device=tensor.device, dtype=tensor.dtype
                )
            if getattr(config, "coord_frame_mode", "none") == "operator_bank":
                if context.frame_operators is None:
                    raise ValueError(
                        "operator_bank ARTI adapter requires frame_operators in the runtime context"
                    )
                kwargs["frame_operators"] = context.frame_operators.to(
                    device=tensor.device, dtype=tensor.dtype
                )
        blended = self._blend(tensor, self._forward_adapter(tensor, **kwargs))
        if resolved_mask is not None:
            blended = torch.where(resolved_mask.unsqueeze(-1), blended, tensor)
        return blended

    def _forward_adapter(self, tensor: Tensor, **kwargs) -> Tensor:
        use_mixed_precision = tensor.device.type == "cuda" and tensor.dtype in {
            torch.float16,
            torch.bfloat16,
        }
        with torch.autocast(
            device_type=tensor.device.type,
            dtype=tensor.dtype,
            enabled=use_mixed_precision,
        ):
            return self.adapter(tensor, **kwargs)

    def _blend(self, original: Tensor, adapted: Tensor) -> Tensor:
        adapted = adapted.to(device=original.device, dtype=original.dtype)
        if (
            bool(getattr(self.adapter, "direct_recall", False))
            and self.output_gate is None
            and self.influence_scale == 1.0
        ):
            return adapted
        delta = self.influence_scale * (adapted - original)
        if self.output_gate is None:
            return original + delta
        return original + torch.tanh(self.output_gate).to(dtype=original.dtype) * delta


def iter_adapter_wrappers(model: nn.Module) -> Iterator[ARTIAdapterWrapper]:
    for module in model.modules():
        if isinstance(module, ARTIAdapterWrapper):
            yield module


def set_adapter_scale(model: nn.Module, scale: float) -> int:
    """Set runtime interpolation strength for every attached ARTI adapter."""

    value = float(scale)
    if not math.isfinite(value) or value < 0:
        raise ValueError("ARTI adapter scale must be finite and non-negative")
    wrappers = tuple(iter_adapter_wrappers(model))
    for wrapper in wrappers:
        wrapper.influence_scale = value
    return len(wrappers)


def set_recall_refine_steps(
    model: nn.Module,
    steps: int,
    *,
    min_steps: int | None = None,
    tolerance: float | None = None,
) -> int:
    """Set runtime Recall refinement depth for every attached adapter.

    ``steps`` is the maximum depth. With no other arguments it is also the
    exact depth. Set ``min_steps`` and ``tolerance`` to enable data-dependent
    early stopping. A depth of zero is an exact Recall bypass.
    """

    resolved = _resolve_recall_refine_settings(steps, min_steps, tolerance)
    return sum(
        _set_wrapper_recall_refine_settings(wrapper, *resolved)
        for wrapper in iter_adapter_wrappers(model)
    )


def _resolve_recall_refine_settings(
    steps: int,
    min_steps: int | None,
    tolerance: float | None,
) -> tuple[int, int, float | None]:
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("Recall refine steps must be a non-negative integer")
    if tolerance is not None:
        tolerance = float(tolerance)
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError("Recall refine tolerance must be finite and non-negative")
    if steps == 0:
        if min_steps not in {None, 0} or tolerance is not None:
            raise ValueError(
                "zero Recall refine steps require min_steps unset or zero and no tolerance"
            )
        resolved_min_steps = 0
    else:
        resolved_min_steps = (
            (1 if tolerance is not None else steps)
            if min_steps is None
            else min_steps
        )
        if (
            isinstance(resolved_min_steps, bool)
            or not isinstance(resolved_min_steps, int)
            or not 1 <= resolved_min_steps <= steps
        ):
            raise ValueError("Recall refine min_steps must be in [1, steps]")
    return steps, resolved_min_steps, tolerance


def _set_wrapper_recall_refine_settings(
    wrapper: ARTIAdapterWrapper,
    steps: int,
    resolved_min_steps: int,
    tolerance: float | None,
) -> int:
    adapter = wrapper.adapter
    layer = getattr(adapter, "layer", None)
    state = getattr(layer, "state", None)
    layer_config = getattr(layer, "config", None)
    state_config = getattr(state, "config", None)
    if not isinstance(layer_config, ARTIConfig) or not isinstance(
        state_config,
        ARTIConfig,
    ):
        return 0
    layer.config = replace(
        layer_config,
        recall_steps=steps,
        recall_min_steps=resolved_min_steps,
        recall_tolerance=tolerance,
    )
    state.config = replace(
        state_config,
        recall_steps=steps,
        recall_min_steps=resolved_min_steps,
        recall_tolerance=tolerance,
    )
    # A compiled loop captures its previous static depth.
    if hasattr(layer, "_compiled_state_write"):
        layer._compiled_state_write = None
    if hasattr(state, "_compiled_product_tail"):
        state._compiled_product_tail = None
    if hasattr(adapter, "_compiled_static_recall_steps"):
        adapter._compiled_static_recall_steps = False
    return 1


def set_recall_refine_schedule(
    model: nn.Module,
    steps: Sequence[int] | Mapping[str, int],
) -> int:
    """Set an exact Recall refinement depth for each attached adapter.

    A sequence follows ``model.named_modules()`` traversal order. A mapping
    addresses adapters by their full module paths and must cover every attached
    adapter exactly, preventing stale per-layer depths from surviving a schedule
    update.
    """

    named_wrappers = tuple(
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, ARTIAdapterWrapper)
    )
    if isinstance(steps, Mapping):
        expected = {name for name, _wrapper in named_wrappers}
        provided = set(steps)
        missing = sorted(expected - provided)
        unexpected = sorted(provided - expected)
        if missing or unexpected:
            raise ValueError(
                "Recall refine schedule paths must match attached adapters exactly; "
                f"missing={missing}, unexpected={unexpected}"
            )
        depths = tuple(steps[name] for name, _wrapper in named_wrappers)
    else:
        if isinstance(steps, (str, bytes)):
            raise TypeError("Recall refine schedule must be a sequence of integers")
        depths = tuple(steps)
        if len(depths) != len(named_wrappers):
            raise ValueError(
                "Recall refine schedule length must match attached adapters; "
                f"expected {len(named_wrappers)}, found {len(depths)}"
            )

    # Validate the complete schedule before touching any adapter.
    settings = tuple(
        _resolve_recall_refine_settings(depth, None, None) for depth in depths
    )
    return sum(
        _set_wrapper_recall_refine_settings(wrapper, *resolved)
        for (_name, wrapper), resolved in zip(named_wrappers, settings, strict=True)
    )


def reset_recall_queries(model: nn.Module) -> int:
    """Reset every Recall field to its configured deterministic query."""

    fields = tuple(
        module for module in model.modules() if isinstance(module, ARTILatentRecallField)
    )
    for field in fields:
        field.reset_query()
    return len(fields)


def set_recall_state_input_retention(model: nn.Module, retention: float) -> int:
    """Set the temporary input bypass for attached state-replacing Recall."""

    value = float(retention)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("Recall state input retention must be finite and in [0, 1]")
    updated = 0
    for wrapper in iter_adapter_wrappers(model):
        state = getattr(getattr(wrapper.adapter, "layer", None), "state", None)
        setter = getattr(state, "set_state_input_retention", None)
        if callable(setter) and bool(getattr(state, "replaces_state", False)):
            setter(value)
            updated += 1
    return updated


def mark_recall_state_banks_calibrated(model: nn.Module) -> int:
    """Protect loaded state-Recall content values from lazy initialization."""

    updated = 0
    for wrapper in iter_adapter_wrappers(model):
        state = getattr(getattr(wrapper.adapter, "layer", None), "state", None)
        marker = getattr(state, "mark_state_bank_calibrated", None)
        config = getattr(state, "config", None)
        if callable(marker) and getattr(config, "recall_value_composition", None) == "state":
            marker()
            updated += 1
    return updated


def compile_adapter_hotpaths(
    model: nn.Module,
    *,
    mode: str = "default",
    dynamic: bool = False,
    fullgraph: bool = False,
) -> int:
    """Compile attached ARTI adapters without compiling the host model.

    Compilation is runtime-only: adapter parameters and exported artifacts keep
    their ordinary eager representation. Compatible product Recall adapters
    share one route-stable compiled tail while routing remains eager; other
    adapter shapes use the generic state-write compiler.
    """

    compiled = 0
    for wrapper in iter_adapter_wrappers(model):
        adapter = wrapper.adapter
        layer = getattr(adapter, "layer", None)
        compile_write = getattr(layer, "compile_write_hotpath", None)
        if not callable(compile_write):
            continue
        if getattr(layer, "write_hotpath_compiled", False):
            continue
        adapter._compiled_static_recall_steps = True
        compile_write(mode=mode, dynamic=dynamic, fullgraph=fullgraph)
        compiled += 1
    return compiled


def replace_mapping_value(output: Mapping, key: str, value: Tensor):
    if type(output) is dict:
        replaced = dict(output)
        replaced[key] = value
        return replaced
    try:
        replaced = output.copy()
        replaced[key] = value
        return replaced
    except Exception:
        replaced = dict(output)
        replaced[key] = value
        return replaced


@contextmanager
def adapters_enabled(model: nn.Module, enabled: bool) -> Iterator[None]:
    wrappers = list(iter_adapter_wrappers(model))
    previous = [wrapper.enabled for wrapper in wrappers]
    for wrapper in wrappers:
        wrapper.enabled = enabled
    try:
        yield
    finally:
        for wrapper, value in zip(wrappers, previous):
            wrapper.enabled = value


def select_candidates(
    candidates: tuple[InsertionCandidate, ...], spec: InsertionSpec
) -> tuple[InsertionCandidate, ...]:
    matched = []
    seen = set()
    repeated_composites = _repeated_composite_types(candidates, spec.positions)
    repeated_stage = _repeated_stage_paths(candidates, spec.positions)
    for pattern in spec.where:
        for candidate in candidates:
            matches = _matches_pattern(
                candidate,
                pattern,
                repeated_composites=repeated_composites,
                repeated_stage=repeated_stage,
            )
            excluded = any(
                _matches_pattern(
                    candidate,
                    excluded_pattern,
                    repeated_composites=repeated_composites,
                    repeated_stage=repeated_stage,
                )
                for excluded_pattern in spec.exclude
            )
            if (
                candidate.position in spec.positions
                and candidate.name not in seen
                and matches
                and not excluded
            ):
                matched.append(candidate)
                seen.add(candidate.name)
    selected = (
        matched
        if spec.every <= 1
        else [candidate for index, candidate in enumerate(matched) if index % spec.every == 0]
    )
    if spec.max_adapters is not None:
        selected = selected[: spec.max_adapters]
    return tuple(selected)


def _matches_pattern(
    candidate: InsertionCandidate,
    pattern: str,
    *,
    repeated_composites: set[str] | None = None,
    repeated_stage: set[str] | None = None,
) -> bool:
    if pattern == "@tensor-leaves":
        return candidate.is_leaf
    if pattern == "@all-tensor-boundaries":
        return True
    if pattern == "@linear":
        return candidate.module_type == "Linear"
    if pattern == "@repeated-composites":
        return not candidate.is_leaf and candidate.module_type in (repeated_composites or set())
    if pattern == "@repeated-stages":
        return not candidate.is_leaf and candidate.module_path in (repeated_stage or set())
    return fnmatch.fnmatch(candidate.name, pattern) or fnmatch.fnmatch(
        candidate.module_path, pattern
    )


def _scale_for_candidate(
    candidate: InsertionCandidate,
    spec: InsertionSpec,
    default: AdapterScale,
    default_name: str,
) -> tuple[AdapterScale, str]:
    resolved = default
    resolved_name = default_name
    for pattern, scale_name in spec.scale_pattern:
        if _matches_pattern(candidate, pattern):
            resolved = resolve_scale(scale_name)
            resolved_name = scale_name
    return resolved, resolved_name


def get_parent_module(model: nn.Module, qualified_name: str) -> tuple[nn.Module, str]:
    parent = model
    parts = qualified_name.split(".")
    for part in parts[:-1]:
        parent = (
            parent[int(part)]
            if part.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList))
            else getattr(parent, part)
        )
    return parent, parts[-1]


def set_child_module(parent: nn.Module, child_name: str, module: nn.Module) -> None:
    if child_name.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList)):
        parent[int(child_name)] = module
    else:
        setattr(parent, child_name, module)


def make_adapter(
    candidate: InsertionCandidate,
    profile: AdapterProfile,
    scale: AdapterScale,
    *,
    zero_init_output: bool = False,
    bridge_mode: str = "radial",
    residual_budget: float = 1.0,
) -> ARTIResidualBlock:
    coord_dim = profile.coord_dim if profile.observer_phase else 0
    hidden_dim = max(1, int(round(candidate.dim * scale.hidden_multiplier)))
    recall_formula = _instantiate_recall_formula(scale)
    query_seed = int.from_bytes(
        hashlib.sha256(
            (
                "arti-fixed-query-v1|"
                f"{candidate.name}|{candidate.module_path}|{candidate.position}|"
                f"{candidate.dim}|{min(scale.recall_key_dim, hidden_dim)}"
            ).encode("utf-8")
        ).digest()[:8],
        "big",
    ) % (2**63)
    return ARTIResidualBlock(
        dim=candidate.dim,
        coord_dim=coord_dim,
        hidden_dim=hidden_dim,
        operator_count=scale.operator_count,
        interface_slots=scale.interface_slots,
        recall_slots=scale.recall_slots,
        recall_steps=scale.recall_steps,
        recall_min_steps=scale.recall_min_steps,
        recall_tolerance=scale.recall_tolerance,
        recall_activation=scale.recall_activation,
        recall_recognition_mode=scale.recall_recognition_mode,
        recall_routing=scale.recall_routing,
        recall_key_dim=min(scale.recall_key_dim, hidden_dim),
        recall_query_mode="fixed",
        recall_query_seed=query_seed,
        recall_group_size=scale.recall_group_size,
        recall_group_topk=scale.recall_group_topk,
        recall_value_composition=scale.recall_value_composition,
        recall_formula=recall_formula,
        coord_frame_mode=profile.coord_frame_mode if profile.observer_phase else "none",
        zero_init_output=zero_init_output,
        bridge_mode=bridge_mode,
        residual_budget=residual_budget,
        direct_recall=scale.recall_steps > 0,
    )


def _instantiate_recall_formula(scale: AdapterScale) -> nn.Module | None:
    if scale.recall_formula is None:
        return None
    formula = resolve_formula(scale.recall_formula).instantiate()
    contract = getattr(formula, "recall_formula_contract", None)
    if not isinstance(contract, RecallFormulaContract):
        raise ValueError(
            "ARTI Fit Recall formulas must declare recall_formula_contract "
            "so parameter sizing and artifact reconstruction remain deterministic"
        )
    return formula


def _recall_composition_factor(scale: AdapterScale) -> int:
    if scale.recall_formula is not None:
        formula = _instantiate_recall_formula(scale)
        assert formula is not None
        contract = formula.recall_formula_contract
        assert isinstance(contract, RecallFormulaContract)
        return contract.factor_count
    return {
        "single": 1,
        "product": 2,
        "state": STATE_RECALL_COMPOSITION_FACTOR,
    }[scale.recall_value_composition]


def _is_recall_bank_parameter(name: str) -> bool:
    leaf = name.rsplit(".", 1)[-1]
    return leaf == "bank" or leaf.endswith("_bank")


def _recall_bank_parameter_count(module: nn.Module) -> int:
    return sum(
        parameter.numel()
        for name, parameter in module.named_parameters()
        if _is_recall_bank_parameter(name)
    )


def _adapter_parameter_counts(
    candidate: InsertionCandidate,
    profile: AdapterProfile,
    scale: AdapterScale,
    *,
    zero_init_output: bool,
    bridge_mode: str,
    residual_budget: float,
) -> tuple[int, int]:
    """Count a prospective adapter without changing caller RNG state."""

    with torch.random.fork_rng(devices=[]):
        adapter = make_adapter(
            candidate,
            profile,
            scale,
            zero_init_output=zero_init_output,
            bridge_mode=bridge_mode,
            residual_budget=residual_budget,
        )
    total = sum(parameter.numel() for parameter in adapter.parameters())
    return total, _recall_bank_parameter_count(adapter)


def _resolve_bank_first_scale(
    candidate: InsertionCandidate,
    profile: AdapterProfile,
    scale: AdapterScale,
    *,
    target_parameters: int,
    zero_init_output: bool,
    bridge_mode: str,
    residual_budget: float,
) -> AdapterScale:
    fraction = scale.recall_bank_fraction
    if scale.recall_steps <= 0 or fraction is None:
        return scale
    if scale.recall_routing != "grouped":
        raise ValueError("recall_bank_fraction requires grouped recall routing")
    if target_parameters <= 0:
        raise ValueError("bank-first target_parameters must be positive")

    requested_hidden = max(1, int(round(candidate.dim * scale.hidden_multiplier)))
    group_size = scale.recall_group_size
    composition_factor = _recall_composition_factor(scale)
    minimum_groups = scale.recall_group_topk * composition_factor

    def template(hidden_dim: int) -> tuple[AdapterScale, int]:
        key_dim = min(scale.recall_key_dim, hidden_dim)
        resolved = replace(
            scale,
            hidden_multiplier=hidden_dim / candidate.dim,
            recall_slots=group_size * minimum_groups,
            recall_key_dim=key_dim,
        )
        total_parameters, bank_parameters = _adapter_parameter_counts(
            candidate,
            profile,
            resolved,
            zero_init_output=zero_init_output,
            bridge_mode=bridge_mode,
            residual_budget=residual_budget,
        )
        return resolved, total_parameters - bank_parameters

    controller_limit = int(target_parameters * (1.0 - fraction))
    low = 1
    high = requested_hidden
    best: tuple[AdapterScale, int] | None = None
    while low <= high:
        middle = (low + high) // 2
        resolved, nonbank_parameters = template(middle)
        if nonbank_parameters <= controller_limit:
            best = (resolved, nonbank_parameters)
            low = middle + 1
        else:
            high = middle - 1
    if best is None:
        raise ValueError(
            f"adapter budget {target_parameters} is too small for a bank-first "
            f"controller at {candidate.name!r}"
        )

    resolved, nonbank_parameters = best
    available = target_parameters - nonbank_parameters
    resolved_group_size = group_size
    while True:
        minimum_unit = replace(
            resolved,
            recall_slots=resolved_group_size * composition_factor,
            recall_group_size=resolved_group_size,
            recall_group_topk=1,
        )
        _, bank_parameters_per_unit = _adapter_parameter_counts(
            candidate,
            profile,
            minimum_unit,
            zero_init_output=zero_init_output,
            bridge_mode=bridge_mode,
            residual_budget=residual_budget,
        )
        bank_parameters_per_group = bank_parameters_per_unit // composition_factor
        if available >= minimum_groups * bank_parameters_per_group or resolved_group_size == 1:
            break
        resolved_group_size = max(1, resolved_group_size // 2)
    groups = available // bank_parameters_per_group
    groups -= groups % composition_factor
    if groups < minimum_groups:
        raise ValueError(
            f"adapter budget {target_parameters} cannot allocate "
            f"{minimum_groups} Recall groups at {candidate.name!r}"
        )
    return replace(
        resolved,
        recall_slots=int(groups * resolved_group_size),
        recall_group_size=resolved_group_size,
    )


def plan_adapters(
    candidates: tuple[InsertionCandidate, ...],
    spec: InsertionSpec,
    profile: AdapterProfile,
    scale: AdapterScale,
    *,
    scale_name: str,
) -> AdapterInsertionPlan:
    selected = []
    skipped_budget = []
    used_params = 0
    selected_candidates = select_candidates(candidates, spec)
    initial_budget = 1.0 / math.sqrt(max(1, len(selected_candidates)))
    target_per_candidate = (
        None
        if spec.max_extra_params is None or not selected_candidates
        else spec.max_extra_params // len(selected_candidates)
    )
    baseline_count_cache: dict[tuple[int, AdapterProfile, AdapterScale], int] = {}
    resolved_scale_cache: dict[tuple[int, AdapterProfile, AdapterScale, int], AdapterScale] = {}
    parameter_count_cache: dict[tuple[int, AdapterProfile, AdapterScale], tuple[int, int]] = {}
    module_paths = [candidate.module_path for candidate in selected_candidates]
    duplicates = sorted({path for path in module_paths if module_paths.count(path) > 1})
    if duplicates:
        raise ValueError(
            "select at most one tensor boundary per module in one insertion pass; "
            f"conflicting modules: {duplicates}"
        )
    for candidate in selected_candidates:
        candidate_scale, candidate_scale_name = _scale_for_candidate(
            candidate, spec, scale, scale_name
        )
        if target_per_candidate is None:
            baseline_scale = replace(
                candidate_scale,
                recall_bank_fraction=None,
                recall_routing="dense",
            )
            baseline_key = (candidate.dim, profile, baseline_scale)
            candidate_target = baseline_count_cache.get(baseline_key, 0)
            if candidate_target == 0:
                candidate_target, _ = _adapter_parameter_counts(
                    candidate,
                    profile,
                    baseline_scale,
                    zero_init_output=spec.zero_init_output,
                    bridge_mode=spec.bridge_mode,
                    residual_budget=initial_budget,
                )
                baseline_count_cache[baseline_key] = candidate_target
        else:
            candidate_target = target_per_candidate
        resolved_key = (candidate.dim, profile, candidate_scale, candidate_target)
        candidate_scale = resolved_scale_cache.get(resolved_key) or _resolve_bank_first_scale(
            candidate,
            profile,
            candidate_scale,
            target_parameters=candidate_target,
            zero_init_output=spec.zero_init_output,
            bridge_mode=spec.bridge_mode,
            residual_budget=initial_budget,
        )
        resolved_scale_cache[resolved_key] = candidate_scale
        count_key = (candidate.dim, profile, candidate_scale)
        counts = parameter_count_cache.get(count_key)
        if counts is None:
            counts = _adapter_parameter_counts(
                candidate,
                profile,
                candidate_scale,
                zero_init_output=spec.zero_init_output,
                bridge_mode=spec.bridge_mode,
                residual_budget=initial_budget,
            )
            parameter_count_cache[count_key] = counts
        adapter_params, bank_params = counts
        adapter_params += 1 if spec.identity_gate else 0
        hidden_dim = (
            candidate.dim
            if candidate_scale.recall_steps > 0
            else int(round(candidate.dim * candidate_scale.hidden_multiplier))
        )
        planned = InsertedAdapter(
            name=candidate.name,
            module_path=candidate.module_path,
            position=candidate.position,
            tensor_path=candidate.tensor_path,
            dim=candidate.dim,
            parameters=adapter_params,
            profile=profile.name,
            scale=candidate_scale_name,
            bridge_mode=("recall-write" if candidate_scale.recall_steps > 0 else spec.bridge_mode),
            residual_budget=1.0 if candidate_scale.recall_steps > 0 else initial_budget,
            hidden_dim=hidden_dim,
            recall_slots=candidate_scale.recall_slots,
            recall_bank_parameters=bank_params,
            recall_bank_fraction=0.0 if adapter_params <= 0 else bank_params / adapter_params,
            recall_routing=candidate_scale.recall_routing,
            recall_key_dim=min(candidate_scale.recall_key_dim, candidate.dim),
            recall_group_size=candidate_scale.recall_group_size,
            recall_group_topk=candidate_scale.recall_group_topk,
        )
        if (
            spec.max_extra_params is not None
            and used_params + adapter_params > spec.max_extra_params
        ):
            skipped_budget.append(planned)
            continue
        selected.append(planned)
        used_params += adapter_params
    final_budget = 1.0 / math.sqrt(max(1, len(selected)))
    selected = [
        replace(
            item,
            residual_budget=1.0 if item.bridge_mode == "recall-write" else final_budget,
        )
        for item in selected
    ]
    repeated_composites = _repeated_composite_types(candidates, spec.positions)
    repeated_stage = _repeated_stage_paths(candidates, spec.positions)
    excluded = tuple(
        candidate.name
        for candidate in candidates
        if candidate.position in spec.positions
        and any(
            _matches_pattern(
                candidate,
                pattern,
                repeated_composites=repeated_composites,
                repeated_stage=repeated_stage,
            )
            for pattern in spec.where
        )
        and any(
            _matches_pattern(
                candidate,
                pattern,
                repeated_composites=repeated_composites,
                repeated_stage=repeated_stage,
            )
            for pattern in spec.exclude
        )
    )
    return AdapterInsertionPlan(
        selected=tuple(selected),
        skipped_budget=tuple(skipped_budget),
        excluded=excluded,
        spec=spec,
    )


def _repeated_composite_types(
    candidates: tuple[InsertionCandidate, ...],
    positions: tuple[str, ...],
) -> set[str]:
    paths_by_type: dict[str, set[str]] = {}
    for candidate in candidates:
        if candidate.is_leaf or candidate.position not in positions:
            continue
        paths_by_type.setdefault(candidate.module_type, set()).add(candidate.module_path)
    return {
        module_type for module_type, module_paths in paths_by_type.items() if len(module_paths) > 1
    }


def _repeated_stage_paths(
    candidates: tuple[InsertionCandidate, ...],
    positions: tuple[str, ...],
) -> set[str]:
    """Identify one shallow, dominant sibling stack of repeated composites."""

    paths_by_family: dict[tuple[str, str], set[str]] = {}
    depths_by_family: dict[tuple[str, str], list[int]] = {}
    for candidate in candidates:
        if candidate.is_leaf or candidate.position not in positions:
            continue
        parent, separator, child = candidate.module_path.rpartition(".")
        if not separator:
            parent, child = "", candidate.module_path
        if not child.isdigit():
            continue
        family = (parent, candidate.module_type)
        paths_by_family.setdefault(family, set()).add(candidate.module_path)
        depths_by_family.setdefault(family, []).append(candidate.path_depth)
    repeated = [family for family, paths in paths_by_family.items() if len(paths) > 1]
    if not repeated:
        return set()
    selected = min(
        repeated,
        key=lambda family: (
            sum(depths_by_family[family]) / len(depths_by_family[family]),
            -len(paths_by_family[family]),
            family,
        ),
    )
    return paths_by_family[selected]


def insert_adapters(
    model: nn.Module,
    candidates: tuple[InsertionCandidate, ...],
    spec: InsertionSpec,
    profile: AdapterProfile,
    scale: AdapterScale,
    *,
    scale_name: str,
) -> tuple[InsertedAdapter, ...]:
    inserted = []
    planned = plan_adapters(candidates, spec, profile, scale, scale_name=scale_name)
    planned_names = {adapter.name: adapter for adapter in planned.selected}
    residual_budget = 1.0 / math.sqrt(max(1, len(planned.selected)))
    if spec.freeze_base:
        for parameter in model.parameters():
            parameter.requires_grad = False
    for candidate in select_candidates(candidates, spec):
        if candidate.name not in planned_names:
            continue
        parent, child_name = get_parent_module(model, candidate.module_path)
        base = (
            parent[int(child_name)]
            if child_name.isdigit() and isinstance(parent, (nn.Sequential, nn.ModuleList))
            else getattr(parent, child_name)
        )
        candidate_scale, _ = _scale_for_candidate(candidate, spec, scale, scale_name)
        planned_adapter = planned_names[candidate.name]
        candidate_scale = replace(
            candidate_scale,
            hidden_multiplier=planned_adapter.hidden_dim / candidate.dim,
            recall_slots=planned_adapter.recall_slots,
            recall_routing=planned_adapter.recall_routing,
            recall_key_dim=planned_adapter.recall_key_dim,
            recall_group_size=planned_adapter.recall_group_size,
            recall_group_topk=planned_adapter.recall_group_topk,
        )
        adapter = make_adapter(
            candidate,
            profile,
            candidate_scale,
            zero_init_output=spec.zero_init_output,
            bridge_mode=spec.bridge_mode,
            residual_budget=residual_budget,
        )
        wrapper = ARTIAdapterWrapper(
            base,
            adapter,
            freeze_base=spec.freeze_base,
            identity_gate=spec.identity_gate,
            boundary_mask_key=spec.boundary_mask_key,
            require_runtime_context=spec.require_runtime_context,
            batch_axis=candidate.batch_axis,
            feature_axis=candidate.feature_axis,
            position=candidate.position,
            tensor_path=candidate.tensor_path,
        )
        reference = next(
            (parameter for parameter in base.parameters() if parameter.is_floating_point()), None
        )
        if reference is not None:
            wrapper.to(device=reference.device, dtype=reference.dtype)
        elif candidate.device is not None and candidate.dtype is not None:
            dtype = getattr(torch, candidate.dtype.removeprefix("torch."), None)
            if isinstance(dtype, torch.dtype):
                wrapper.to(device=torch.device(candidate.device), dtype=dtype)
        set_child_module(parent, child_name, wrapper)
        inserted.append(planned_names[candidate.name])
    return tuple(inserted)
