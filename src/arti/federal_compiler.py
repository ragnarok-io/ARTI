"""Compile Federal paths and explicit stateful Tensor graphs.

Static paths, state/effect recurrences, and bounded logical-shape carriers are
separate contracts.  None of them snapshots hidden Python runtime state or
mutates a parameter during forward.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import ClassVar, Literal, Mapping, Sequence

import torch
from torch import Tensor, nn

from .formula_fabric import FormulaArenaState, FormulaFabric, FormulaRoutePlan
from .reversible_topology import FixedTopologyPolicy, TopologyFold, TopologyUnFold


class FederalCompileError(ValueError):
    """Raised when a Federal path cannot be specialized safely."""


def _json_fingerprint(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _module_ref(module: nn.Module) -> str:
    reference = getattr(module, "_component_reference", None)
    if isinstance(reference, str) and reference:
        return reference
    return f"torch/{module.__class__.__module__}.{module.__class__.__qualname__}@1"


def _shape_matches(actual: Sequence[int], expected: Sequence[int | None]) -> bool:
    return len(actual) == len(expected) and all(
        expected_value is None or actual_value == expected_value
        for actual_value, expected_value in zip(actual, expected, strict=True)
    )


def _static_operation_reasons(module: nn.Module, *, prefix: str = "") -> tuple[str, ...]:
    """Find dynamic operation nodes that cannot be erased by specialization."""

    reasons: list[str] = []
    module_name = module.__class__.__name__
    module_path = f"{prefix}.{module_name}" if prefix else module_name
    if getattr(module, "_static_compile_compatible", True) is not True:
        reasons.append(f"operation {_module_ref(module)!r} is not static-compatible")
    if module.__class__.__module__ == "arti.nn" and module_name in {"Fold", "UnFold"}:
        reasons.append(
            f"dynamic arti.nn.{module_name} at {module_path} requires a fixed transport specialization"
        )
    for child_name, child in module.named_children():
        reasons.extend(_static_operation_reasons(child, prefix=f"{module_path}.{child_name}"))
    return tuple(reasons)


@dataclass(frozen=True)
class FederalCompileAssessment:
    """Static classification of a path before compilation."""

    classification: Literal["exact-static", "gated-static", "uncompilable"]
    reasons: tuple[str, ...] = ()

    @property
    def compilable(self) -> bool:
        return self.classification == "exact-static"

    def to_dict(self) -> dict[str, object]:
        return {
            "classification": self.classification,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class FederalCompileManifest:
    """Provenance for an ordinary network produced from one Federal path."""

    schema: str
    compiler_ref: str
    mode: str
    source_ref: str
    source_snapshot_fingerprint: str
    path_ids: tuple[str, ...]
    operation_refs: tuple[str, ...]
    dependency_refs: tuple[str, ...]
    terminal_abi_ref: str
    refine_steps: int
    input_shape: tuple[int | None, ...] | None = None
    output_shape: tuple[int | None, ...] | None = None

    _component_reference: ClassVar[str] = "arti/federal-compile-manifest@1"

    def __post_init__(self) -> None:
        if self.schema != self._component_reference:
            raise FederalCompileError("unsupported Federal compile manifest schema")
        if self.mode != "exact-static":
            raise FederalCompileError("compiled manifest must use exact-static mode")
        if not self.source_ref or not self.source_snapshot_fingerprint:
            raise FederalCompileError("compiled manifest requires Federal source provenance")
        if not self.path_ids:
            raise FederalCompileError("compiled manifest requires a non-empty path")
        if len(self.operation_refs) == 0:
            raise FederalCompileError("compiled manifest requires operations")
        if any(not isinstance(ref, str) or not ref for ref in self.dependency_refs):
            raise FederalCompileError("compiled manifest dependencies must be named refs")
        if not self.terminal_abi_ref:
            raise FederalCompileError("compiled manifest requires terminal ABI provenance")
        if isinstance(self.refine_steps, bool) or self.refine_steps <= 0:
            raise FederalCompileError("refine_steps must be positive")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "compiler_ref": self.compiler_ref,
            "mode": self.mode,
            "source_ref": self.source_ref,
            "source_snapshot_fingerprint": self.source_snapshot_fingerprint,
            "path_ids": list(self.path_ids),
            "operation_refs": list(self.operation_refs),
            "dependency_refs": list(self.dependency_refs),
            "terminal_abi_ref": self.terminal_abi_ref,
            "refine_steps": self.refine_steps,
            "input_shape": None if self.input_shape is None else list(self.input_shape),
            "output_shape": None if self.output_shape is None else list(self.output_shape),
        }

    @property
    def fingerprint(self) -> str:
        return _json_fingerprint(self.to_dict())


class FederalParallel(nn.Module):
    """An ordinary PyTorch parallel branch used by a frozen Federal path."""

    _component_reference = "arti/federal-parallel@1"

    def __init__(self, branches: Sequence[nn.Module], *, merge: str = "sum", dim: int = -1):
        super().__init__()
        if not branches or any(not isinstance(branch, nn.Module) for branch in branches):
            raise TypeError("branches must contain at least one nn.Module")
        if merge not in {"sum", "concat"}:
            raise ValueError("merge must be 'sum' or 'concat'")
        self.branches = nn.ModuleList(branches)
        self.merge = merge
        self.dim = dim
        self._component_dependencies = tuple(
            dict.fromkeys(
                dependency
                for branch in branches
                for dependency in (
                    (_module_ref(branch),)
                    + tuple(getattr(branch, "_component_dependencies", ()))
                )
            )
        )

    def forward(self, value: Tensor) -> Tensor:
        outputs = tuple(branch(value) for branch in self.branches)
        if self.merge == "sum":
            first = outputs[0]
            if any(output.shape != first.shape for output in outputs[1:]):
                raise FederalCompileError("sum branches must have identical output shapes")
            return torch.stack(outputs, dim=0).sum(dim=0)
        return torch.cat(outputs, dim=self.dim)


class FederalResidual(nn.Module):
    """An ordinary residual branch extracted from a Federal path."""

    _component_reference = "arti/federal-residual@1"

    def __init__(self, branch: nn.Module) -> None:
        super().__init__()
        if not isinstance(branch, nn.Module):
            raise TypeError("branch must be an nn.Module")
        self.branch = branch
        self._static_compile_compatible = (
            getattr(branch, "_static_compile_compatible", True) is True
        )
        self._component_dependencies = tuple(getattr(branch, "_component_dependencies", ()))

    def forward(self, value: Tensor) -> Tensor:
        branch_value = self.branch(value)
        if not isinstance(branch_value, Tensor):
            raise TypeError("residual branch must return a Tensor")
        if branch_value.shape != value.shape:
            raise FederalCompileError("residual branch must preserve the input shape")
        return value + branch_value


class FederalRefine(nn.Module):
    """A finite, statically expandable Refine sequence."""

    _component_reference = "arti/federal-refine@1"

    def __init__(self, operation: nn.Module, *, steps: int) -> None:
        super().__init__()
        if not isinstance(operation, nn.Module):
            raise TypeError("operation must be an nn.Module")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("steps must be a positive integer")
        self.operation = operation
        self.steps = steps
        self._static_compile_compatible = (
            getattr(operation, "_static_compile_compatible", True) is True
        )
        self._component_dependencies = tuple(getattr(operation, "_component_dependencies", ()))

    def forward(self, value: Tensor) -> Tensor:
        for _ in range(self.steps):
            value = self.operation(value)
            if not isinstance(value, Tensor):
                raise TypeError("Refine operation must return a Tensor")
        return value


class FederalFormulaBlock(nn.Module):
    """Expose one fixed Formula Fabric route as a Tensor-to-Tensor module."""

    _component_reference = "arti/federal-formula-block@1"
    _component_dependencies = ("arti/formula-fabric@1",)

    def __init__(
        self,
        fabric: FormulaFabric,
        route: FormulaRoutePlan,
        *,
        output_slots: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(fabric, FormulaFabric):
            raise TypeError("fabric must be a FormulaFabric")
        if not isinstance(route, FormulaRoutePlan):
            raise TypeError("route must be a FormulaRoutePlan")
        expected = (
            1,
            len(fabric.program.steps),
            fabric.program.max_cells,
            fabric.program.max_arity,
            fabric.program.arena_capacity,
        )
        if route.weights.shape != expected:
            raise ValueError(f"fixed Formula route must have shape {expected}")
        if route.estimator != "hard":
            raise ValueError("FederalFormulaBlock requires a hard forward route")
        selected = tuple(output_slots or ())
        if any(
            isinstance(slot, bool)
            or not isinstance(slot, int)
            or not 0 <= slot < fabric.program.arena_capacity
            for slot in selected
        ):
            raise ValueError("output_slots must contain valid arena slots")
        self.fabric = fabric
        self.output_slots = selected
        self.register_buffer("_route_weights", route.weights, persistent=True)
        self.register_buffer("_route_valid", route.valid_mask, persistent=True)
        self.register_buffer("_route_fire", route.fire_mask, persistent=True)
        self.register_buffer("_route_commit", route.commit_mask, persistent=True)
        self.register_buffer(
            "_output_slots",
            torch.tensor(selected, dtype=torch.long),
            persistent=True,
        )

    def forward(self, value: Tensor) -> Tensor:
        if not isinstance(value, Tensor) or value.ndim != 3:
            raise TypeError("FederalFormulaBlock expects a [B, S, D] Tensor")
        expected = (self.fabric.program.arena_capacity, self.fabric.program.feature_dim)
        if value.shape[-2:] != expected:
            raise FederalCompileError(
                f"Formula input must end in {expected}, got {tuple(value.shape[-2:])}"
            )
        mask = torch.ones(value.shape[:-1], dtype=torch.bool, device=value.device)
        state = FormulaArenaState.from_tensor(
            value,
            mask,
            capacity=self.fabric.program.arena_capacity,
            domain=self.fabric.program.domain,
        )
        batch = value.shape[0]
        route = FormulaRoutePlan(
            self._route_weights.expand(batch, *self._route_weights.shape[1:]),
            self._route_valid.expand(batch, *self._route_valid.shape[1:]),
            self._route_fire.expand(batch, *self._route_fire.shape[1:]),
            self._route_commit.expand(batch, *self._route_commit.shape[1:]),
            estimator="hard",
        )
        result = self.fabric(state, route)
        if self._output_slots.numel() == 0:
            return result.state.value
        return result.state.value.index_select(-2, self._output_slots)


class FederalTopologyBlock(nn.Module):
    """A compiled Fold@2 -> active operation -> UnFold@2 block.

    The topology must be fixed for exact static compilation.  A learned or
    input-dependent topology remains a Federal runtime concern and is marked
    incompatible instead of being silently treated as a fixed route.
    """

    _component_reference = "arti/federal-topology-block@1"
    _component_dependencies = ("arti/fold@2", "arti/unfold@2")

    def __init__(
        self,
        fold: TopologyFold,
        operation: nn.Module,
        unfold: TopologyUnFold,
    ) -> None:
        super().__init__()
        if not isinstance(fold, TopologyFold):
            raise TypeError("fold must be a TopologyFold")
        if not isinstance(unfold, TopologyUnFold):
            raise TypeError("unfold must be a TopologyUnFold")
        if not isinstance(operation, nn.Module):
            raise TypeError("operation must be an nn.Module")
        if fold.topology.active_count != unfold.inverse_contract.active_count:
            raise FederalCompileError("Fold and UnFold active counts must match")
        if fold.topology.axis != unfold.inverse_contract.axis:
            raise FederalCompileError("Fold and UnFold axes must match")
        if fold.topology.contract_fingerprint != unfold.inverse_contract.contract_fingerprint:
            raise FederalCompileError("Fold and UnFold topology contracts must match")
        self.fold = fold
        self.operation = operation
        self.unfold = unfold
        self._static_compile_compatible = (
            isinstance(fold.topology.policy, FixedTopologyPolicy)
            and getattr(operation, "_static_compile_compatible", True) is True
        )

    def forward(self, value: Tensor) -> Tensor:
        folded = self.fold(value)
        active = self.operation(folded.active)
        if not isinstance(active, Tensor):
            raise TypeError("topology active operation must return a Tensor")
        restored = self.unfold(folded.replace(active=active))
        return restored.value


class FederalStaticFold(nn.Module):
    """A fixed gather specialization of ``arti/fold@2``.

    This is the shape-changing form used when a Federal path has already
    frozen its selected input positions.  It has no learned Query and no
    runtime routing: the compiler is specializing a recorded selection into
    an ordinary tensor operation.
    """

    _component_reference = "arti/federal-static-fold@1"
    _component_dependencies = ("arti/fold@2",)
    _static_compile_compatible = True

    def __init__(self, input_length: int, active_indices: Sequence[int]) -> None:
        super().__init__()
        if isinstance(input_length, bool) or not isinstance(input_length, int) or input_length <= 0:
            raise ValueError("input_length must be a positive integer")
        indices = tuple(active_indices)
        if not indices:
            raise ValueError("active_indices must not be empty")
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < input_length
            for index in indices
        ):
            raise ValueError("active_indices must be valid input positions")
        if len(set(indices)) != len(indices):
            raise ValueError("active_indices must be unique for a fixed Fold specialization")
        self.input_length = input_length
        self.active_length = len(indices)
        self.register_buffer(
            "_active_indices", torch.tensor(indices, dtype=torch.long), persistent=True
        )

    def forward(self, value: Tensor) -> Tensor:
        if not isinstance(value, Tensor) or value.ndim < 2:
            raise TypeError("FederalStaticFold expects a tensor with a sequence axis")
        if value.shape[-2] != self.input_length:
            raise FederalCompileError(
                f"fixed Fold expects length {self.input_length}, got {value.shape[-2]}"
            )
        return value.index_select(-2, self._active_indices.to(device=value.device))


class FederalStaticUnFold(nn.Module):
    """A fixed gather/insert specialization of ``arti/unfold@2``.

    ``source_indices`` maps every output position to an input position.  A
    value of ``-1`` inserts a fixed (or zero) value.  This deliberately keeps
    shape-changing transport explicit and finite; learned layout, dynamic
    target lengths, and input-dependent routing remain Federal runtime paths.
    """

    _component_reference = "arti/federal-static-unfold@1"
    _component_dependencies = ("arti/unfold@2",)
    _static_compile_compatible = True

    def __init__(
        self,
        input_length: int,
        source_indices: Sequence[int],
        *,
        insert_values: Tensor | None = None,
    ) -> None:
        super().__init__()
        if isinstance(input_length, bool) or not isinstance(input_length, int) or input_length <= 0:
            raise ValueError("input_length must be a positive integer")
        indices = tuple(source_indices)
        if not indices:
            raise ValueError("source_indices must not be empty")
        if any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < -1
            or index >= input_length
            for index in indices
        ):
            raise ValueError("source_indices must contain -1 or valid input positions")
        if insert_values is not None:
            if not isinstance(insert_values, Tensor) or insert_values.ndim != 2:
                raise TypeError("insert_values must have shape [output_length, feature]")
            if insert_values.shape[0] != len(indices):
                raise ValueError("insert_values must match output_length")
            if any(index == -1 for index in indices) is False:
                raise ValueError("insert_values are only valid when an output position is inserted")
        self.input_length = input_length
        self.output_length = len(indices)
        self.register_buffer(
            "_source_indices", torch.tensor(indices, dtype=torch.long), persistent=True
        )
        if insert_values is not None:
            self.register_buffer("_insert_values", insert_values.detach().clone(), persistent=True)
        else:
            self._insert_values = None

    def forward(self, value: Tensor) -> Tensor:
        if not isinstance(value, Tensor) or value.ndim < 2:
            raise TypeError("FederalStaticUnFold expects a tensor with a sequence axis")
        if value.shape[-2] != self.input_length:
            raise FederalCompileError(
                f"fixed UnFold expects length {self.input_length}, got {value.shape[-2]}"
            )
        source = self._source_indices.to(device=value.device)
        safe_source = source.clamp_min(0)
        gathered = value.index_select(-2, safe_source)
        inserted = source < 0
        if not bool(inserted.any()):
            return gathered
        if self._insert_values is None:
            fill = torch.zeros_like(gathered)
        else:
            if self._insert_values.shape[-1] != value.shape[-1]:
                raise FederalCompileError(
                    "fixed UnFold insert feature dimension does not match input"
                )
            fill = self._insert_values.to(device=value.device, dtype=value.dtype)
            fill = fill.reshape((1,) * (value.ndim - 2) + fill.shape).expand_as(gathered)
        mask = inserted.reshape((1,) * (value.ndim - 2) + (self.output_length, 1))
        return torch.where(mask, fill, gathered)


class FederalPath(nn.Module):
    """Explicit frozen execution path extracted from a Federal snapshot.

    The path contains ordinary tensor operations only.  ``route_mode='gated'``
    is retained for assessment, but is intentionally not compiled by the first
    reference compiler because its gate still belongs to the Federal runtime.
    """

    _component_reference = "arti/federal-path@1"

    @classmethod
    def from_trace(
        cls,
        trace: object,
        operations: Sequence[nn.Module],
        *,
        source_ref: str,
        source_snapshot_fingerprint: str,
        terminal_abi_ref: str,
        sample_index: int = 0,
        **kwargs: object,
    ) -> "FederalPath":
        """Build explicit compile evidence from one Federal traversal receipt.

        A trace supplies only the selected path evidence.  It does not supply
        executable Python or mutable Bank state; callers must still provide
        the frozen operation modules and snapshot fingerprint explicitly.
        """

        winner_paths = getattr(trace, "winner_paths", None)
        if not isinstance(winner_paths, (tuple, list)) or not winner_paths:
            raise FederalCompileError("Federal trace has no winner path evidence")
        if (
            isinstance(sample_index, bool)
            or not isinstance(sample_index, int)
            or not 0 <= sample_index < len(winner_paths)
        ):
            raise FederalCompileError("trace sample_index is outside winner path evidence")
        winner = winner_paths[sample_index]
        if not isinstance(winner, str) or not winner:
            raise FederalCompileError("Federal trace winner path must be a named string")
        path_ids = tuple(part for part in winner.split("/") if part)
        if not path_ids:
            raise FederalCompileError("Federal trace winner path is empty")
        if not hasattr(trace, "steps"):
            raise FederalCompileError("Federal trace is missing traversal steps")
        return cls(
            operations,
            source_ref=source_ref,
            source_snapshot_fingerprint=source_snapshot_fingerprint,
            path_ids=path_ids,
            terminal_abi_ref=terminal_abi_ref,
            **kwargs,
        )

    def __init__(
        self,
        operations: Sequence[nn.Module],
        *,
        source_ref: str,
        source_snapshot_fingerprint: str,
        path_ids: Sequence[str],
        terminal_abi_ref: str,
        refine_steps: int = 1,
        route_mode: Literal["frozen", "gated", "dynamic"] = "frozen",
        data_dependent_shape: bool = False,
        mutable_state: bool = False,
        runtime_callback: bool = False,
        unbounded_refine: bool = False,
        operation_refs: Sequence[str] | None = None,
        dependency_refs: Sequence[str] | None = None,
        input_shape: Sequence[int | None] | None = None,
        output_shape: Sequence[int | None] | None = None,
    ) -> None:
        super().__init__()
        normalized = tuple(operations)
        if not normalized or any(not isinstance(operation, nn.Module) for operation in normalized):
            raise TypeError("operations must contain at least one nn.Module")
        if isinstance(refine_steps, bool) or not isinstance(refine_steps, int) or refine_steps <= 0:
            raise ValueError("refine_steps must be a positive integer")
        if route_mode not in {"frozen", "gated", "dynamic"}:
            raise ValueError("route_mode must be frozen, gated, or dynamic")
        refs = tuple(operation_refs or (_module_ref(operation) for operation in normalized))
        if len(refs) != len(normalized) or any(not isinstance(ref, str) or not ref for ref in refs):
            raise ValueError("operation_refs must match operations and be non-empty")
        if not source_ref or not source_snapshot_fingerprint or not terminal_abi_ref:
            raise ValueError("source and terminal ABI provenance are required")
        normalized_paths = tuple(path_ids)
        if not normalized_paths:
            raise ValueError("path_ids must be non-empty")
        dependencies = tuple(
            dependency_refs
            if dependency_refs is not None
            else (
                dependency
                for operation in normalized
                for dependency in getattr(operation, "_component_dependencies", ())
            )
        )
        if any(not isinstance(ref, str) or not ref for ref in dependencies):
            raise ValueError("dependency_refs must contain non-empty strings")

        def normalize_shape(
            shape: Sequence[int | None] | None, *, field: str
        ) -> tuple[int | None, ...] | None:
            if shape is None:
                return None
            normalized_shape = tuple(shape)
            if not normalized_shape:
                raise ValueError(f"{field} must not be empty")
            if any(
                value is not None
                and (isinstance(value, bool) or not isinstance(value, int) or value <= 0)
                for value in normalized_shape
            ):
                raise ValueError(f"{field} entries must be positive integers or None")
            return normalized_shape

        self.operations = nn.ModuleList(normalized)
        self.source_ref = source_ref
        self.source_snapshot_fingerprint = source_snapshot_fingerprint
        self.path_ids = normalized_paths
        self.terminal_abi_ref = terminal_abi_ref
        self.refine_steps = refine_steps
        self.route_mode = route_mode
        self.data_dependent_shape = bool(data_dependent_shape)
        self.mutable_state = bool(mutable_state)
        self.runtime_callback = bool(runtime_callback)
        self.unbounded_refine = bool(unbounded_refine)
        self.operation_refs = refs
        self.dependency_refs = dependencies
        self.input_shape = normalize_shape(input_shape, field="input_shape")
        self.output_shape = normalize_shape(output_shape, field="output_shape")

    def forward(self, value: Tensor) -> Tensor:
        value, _ = self.forward_with_intermediates(value)
        return value

    def forward_with_intermediates(self, value: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
        if not isinstance(value, Tensor):
            raise TypeError("Federal path input must be a Tensor")
        if self.input_shape is not None and not _shape_matches(value.shape, self.input_shape):
            raise FederalCompileError("Federal path input does not match its shape ABI")
        intermediates: list[Tensor] = []
        for operation in self.operations:
            value = operation(value)
            if not isinstance(value, Tensor):
                raise TypeError("every Federal path operation must return a Tensor")
            intermediates.append(value)
        if self.output_shape is not None and not _shape_matches(value.shape, self.output_shape):
            raise FederalCompileError("Federal path output does not match its shape ABI")
        return value, tuple(intermediates)

    def assessment(self) -> FederalCompileAssessment:
        reasons: list[str] = []
        if self.route_mode == "gated":
            reasons.append("runtime gate is not frozen")
        if self.route_mode == "dynamic":
            reasons.append("dynamic routing is not static")
        if self.data_dependent_shape:
            reasons.append("data-dependent shape is not statically specialized")
        if self.mutable_state:
            reasons.append("mutable path state is not snapshot-frozen")
        if self.runtime_callback:
            reasons.append("runtime callback is not part of the compiled graph")
        if self.unbounded_refine:
            reasons.append("unbounded Refine cannot be expanded")
        for operation in self.operations:
            reasons.extend(_static_operation_reasons(operation))
        if self.route_mode == "gated" and not reasons[1:]:
            return FederalCompileAssessment("gated-static", tuple(reasons))
        if reasons:
            return FederalCompileAssessment("uncompilable", tuple(reasons))
        return FederalCompileAssessment("exact-static")


class CompiledFederalNetwork(nn.Module):
    """Ordinary layerwise network with no Federal runtime dependency."""

    _component_reference = "arti/federal-compiled-network@1"

    def __init__(self, operations: Sequence[nn.Module], manifest: FederalCompileManifest) -> None:
        super().__init__()
        if not operations:
            raise ValueError("compiled network requires operations")
        if len(operations) != len(manifest.operation_refs):
            raise ValueError("compiled operations do not match manifest")
        self.operations = nn.ModuleList(operations)
        self.manifest = manifest

    def forward(self, value: Tensor) -> Tensor:
        value, _ = self.forward_with_intermediates(value)
        return value

    def forward_with_intermediates(self, value: Tensor) -> tuple[Tensor, tuple[Tensor, ...]]:
        if not isinstance(value, Tensor):
            raise TypeError("compiled network input must be a Tensor")
        if self.manifest.input_shape is not None and not _shape_matches(
            value.shape, self.manifest.input_shape
        ):
            raise FederalCompileError("compiled input does not match its shape ABI")
        intermediates: list[Tensor] = []
        for operation in self.operations:
            value = operation(value)
            if not isinstance(value, Tensor):
                raise TypeError("compiled operation must return a Tensor")
            intermediates.append(value)
        if self.manifest.output_shape is not None and not _shape_matches(
            value.shape, self.manifest.output_shape
        ):
            raise FederalCompileError("compiled output does not match its shape ABI")
        return value, tuple(intermediates)

    def save(self, directory: str | Path) -> None:
        from safetensors.torch import save_model

        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        save_model(self, str(path / "model.safetensors"))
        (path / "manifest.json").write_text(
            json.dumps(self.manifest.to_dict(), indent=2), encoding="utf-8"
        )

    @classmethod
    def load(cls, directory: str | Path, path: FederalPath) -> CompiledFederalNetwork:
        from safetensors.torch import load_model

        directory_path = Path(directory)
        manifest_data = json.loads((directory_path / "manifest.json").read_text(encoding="utf-8"))
        expected = FederalPathCompiler.compile(path)
        if manifest_data != expected.manifest.to_dict():
            raise FederalCompileError("compiled manifest does not match the supplied path")
        load_model(expected, str(directory_path / "model.safetensors"), strict=True)
        return expected


@dataclass(frozen=True)
class FederalTensorQueryManifest:
    """Provenance for a tensorized dynamic-Query graph.

    Unlike :class:`FederalCompileManifest`, this manifest describes a bounded
    *query graph*, not one already selected Federal winner path.  The route is
    still input-dependent; only its representation has been lowered to Tensor
    operators and a finite unrolled horizon.
    """

    schema: str
    compiler_ref: str
    source_ref: str
    source_snapshot_fingerprint: str
    query_ref: str
    operation_refs: tuple[str, ...]
    route_mode: str
    max_steps: int
    input_shape: tuple[int | None, ...] | None
    output_shape: tuple[int | None, ...] | None
    early_stop_semantics: str = "logical-mask-fixed-horizon"

    _component_reference: ClassVar[str] = "arti/federal-tensor-query-manifest@1"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "compiler_ref": self.compiler_ref,
            "source_ref": self.source_ref,
            "source_snapshot_fingerprint": self.source_snapshot_fingerprint,
            "query_ref": self.query_ref,
            "operation_refs": list(self.operation_refs),
            "route_mode": self.route_mode,
            "max_steps": self.max_steps,
            "input_shape": None if self.input_shape is None else list(self.input_shape),
            "output_shape": None if self.output_shape is None else list(self.output_shape),
            "early_stop_semantics": self.early_stop_semantics,
        }


class FederalTensorQueryAdapter(nn.Module):
    """Erase a Bank Query result wrapper at compile time.

    ``BankQuery`` deliberately returns a ``BankQueryResult`` so eager Federal
    execution can carry its contract.  A tensorized graph only needs the
    numeric Query output.  This adapter is intentionally tiny: validation and
    sealing happen before compilation, while its traced forward is only
    ``query(value).value``.
    """

    _component_reference = "arti/federal-tensor-query-adapter@1"

    def __init__(self, query: nn.Module) -> None:
        super().__init__()
        if not isinstance(query, nn.Module):
            raise TypeError("query must be an nn.Module")
        self.query = query

    def forward(self, value: Tensor) -> Tensor:
        result = self.query(value)
        return result.value


class FederalTensorQueryBlock(nn.Module):
    """A dynamic Query and finite Refine loop expressed only as Tensor ops.

    The Query is evaluated again after every selected operation.  Candidate
    operations must share one Tensor ABI within this block; heterogeneous
    shapes are connected by explicit transport blocks before entering or after
    leaving it.  ``max_steps`` is statically unrolled.  An exit is represented
    by the identity candidate, so no Python ``break`` or Bank-ID dispatch is
    present in the lowered graph.
    """

    _component_reference = "arti/federal-tensor-query-block@1"

    def __init__(
        self,
        query: nn.Module,
        operations: Sequence[nn.Module],
        *,
        max_steps: int,
        route_mode: Literal["hard", "soft", "straight_through"] = "hard",
        manifest: FederalTensorQueryManifest | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(query, nn.Module):
            raise TypeError("query must be an nn.Module")
        normalized = tuple(operations)
        if not normalized or any(not isinstance(operation, nn.Module) for operation in normalized):
            raise TypeError("operations must contain at least one nn.Module")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        if route_mode not in {"hard", "soft", "straight_through"}:
            raise ValueError("route_mode must be hard, soft, or straight_through")
        self.query = query
        self.operations = nn.ModuleList(normalized)
        self.max_steps = int(max_steps)
        self.route_mode = route_mode
        self.exit_index = len(normalized)
        self.manifest = manifest

    def forward(self, value: Tensor) -> Tensor:
        state = value
        batch = value.shape[0]
        done = torch.zeros((batch,), dtype=torch.bool, device=value.device)
        broadcast_shape = (batch, *([1] * (state.ndim - 1)))
        for _ in range(self.max_steps):
            logits = self.query(state)
            candidate_values = [operation(state) for operation in self.operations]
            candidate_values.append(state)
            candidates = torch.stack(candidate_values, dim=1)
            raw_route = torch.argmax(logits, dim=-1)
            route = torch.where(
                done,
                torch.full_like(raw_route, self.exit_index),
                raw_route,
            )
            index = route.reshape(batch, 1, *([1] * (state.ndim - 1)))
            index = index.expand(batch, 1, *state.shape[1:])
            hard_selected = candidates.gather(1, index).squeeze(1)
            if self.route_mode == "hard":
                selected = hard_selected
            else:
                probabilities = torch.softmax(logits, dim=-1)
                weights = probabilities.reshape(batch, probabilities.shape[-1], *([1] * (state.ndim - 1)))
                if self.route_mode == "straight_through":
                    hard_weights = torch.nn.functional.one_hot(
                        route,
                        num_classes=self.exit_index + 1,
                    ).to(probabilities.dtype)
                    weights = hard_weights + probabilities - probabilities.detach()
                    weights = weights.reshape(
                        batch,
                        probabilities.shape[-1],
                        *([1] * (state.ndim - 1)),
                    )
                selected = (candidates * weights).sum(dim=1)
            active = (~done).reshape(broadcast_shape)
            state = torch.where(active, selected, state)
            done = done | (route == self.exit_index)
        return state


class FederalTensorQueryCompiler:
    """Lower a bounded dynamic Query graph into an ordinary Tensor module."""

    _component_reference = "arti/federal-tensor-query-compiler@1"

    @classmethod
    def compile(
        cls,
        query: nn.Module,
        operations: Sequence[nn.Module],
        *,
        example_input: Tensor,
        source_ref: str,
        source_snapshot_fingerprint: str,
        max_steps: int,
        route_mode: Literal["hard", "soft", "straight_through"] = "hard",
        input_shape: Sequence[int | None] | None = None,
        output_shape: Sequence[int | None] | None = None,
        mutable_state: bool = False,
        data_dependent_shape: bool = False,
        runtime_callback: bool = False,
    ) -> FederalTensorQueryBlock:
        if not isinstance(example_input, Tensor) or not example_input.is_floating_point():
            raise TypeError("example_input must be a floating-point Tensor")
        if not source_ref or not source_snapshot_fingerprint:
            raise ValueError("source provenance is required")
        if mutable_state:
            raise FederalCompileError("mutable Query state is not tensorized")
        if data_dependent_shape:
            raise FederalCompileError("data-dependent Query shape is not tensorized")
        if runtime_callback:
            raise FederalCompileError("runtime Query callbacks are not tensorized")
        normalized = tuple(operations)
        if not normalized:
            raise ValueError("operations must not be empty")
        with torch.no_grad():
            logits = query(example_input)
            if not isinstance(logits, Tensor) or logits.ndim != 2:
                raise FederalCompileError("tensorized Query must return [B, action] logits")
            if logits.shape[0] != example_input.shape[0] or logits.shape[1] != len(normalized) + 1:
                raise FederalCompileError(
                    "tensorized Query logits must have one exit action plus one logit per operation"
                )
            expected_shape = tuple(example_input.shape)
            for operation in normalized:
                result = operation(example_input)
                if not isinstance(result, Tensor) or tuple(result.shape) != expected_shape:
                    raise FederalCompileError(
                        "all tensorized Query operations must preserve the example Tensor ABI"
                    )

        def normalize_shape(
            shape: Sequence[int | None] | None,
            *,
            field: str,
        ) -> tuple[int | None, ...] | None:
            if shape is None:
                return None
            result = tuple(shape)
            if not result or any(
                item is not None
                and (isinstance(item, bool) or not isinstance(item, int) or item <= 0)
                for item in result
            ):
                raise ValueError(f"{field} entries must be positive integers or None")
            if not _shape_matches(example_input.shape, result):
                raise FederalCompileError(f"{field} does not match example_input")
            return result

        manifest = FederalTensorQueryManifest(
            schema=FederalTensorQueryManifest._component_reference,
            compiler_ref=cls._component_reference,
            source_ref=source_ref,
            source_snapshot_fingerprint=source_snapshot_fingerprint,
            query_ref=_module_ref(query),
            operation_refs=tuple(_module_ref(operation) for operation in normalized),
            route_mode=route_mode,
            max_steps=max_steps,
            input_shape=normalize_shape(input_shape, field="input_shape"),
            output_shape=normalize_shape(output_shape, field="output_shape"),
        )
        return FederalTensorQueryBlock(
            deepcopy(query),
            deepcopy(normalized),
            max_steps=max_steps,
            route_mode=route_mode,
            manifest=manifest,
        )

    @staticmethod
    def export(
        graph: FederalTensorQueryBlock,
        example_input: Tensor,
        *,
        dynamic_shapes: object | None = None,
    ) -> object:
        if not isinstance(graph, FederalTensorQueryBlock):
            raise TypeError("graph must be a FederalTensorQueryBlock")
        export = getattr(torch, "export", None)
        if export is None or not hasattr(export, "export"):
            raise FederalCompileError("this PyTorch runtime does not provide torch.export")
        return export.export(graph, (example_input,), dynamic_shapes=dynamic_shapes)


class FederalTensorBank(nn.Module):
    """One Bank node in a tensorized finite Federation.

    ``operations[index]`` is the action selected by the corresponding Query
    logit.  ``next_bank_ids[index]`` names the next Bank, or ``None`` for a
    terminal action.  All actions in one compiled Federation share the
    Federation's Tensor ABI; shape-changing edges must be explicit transport
    modules in a separate ABI family.
    """

    _component_reference = "arti/federal-tensor-bank@1"

    def __init__(
        self,
        bank_id: str,
        query: nn.Module,
        operations: Sequence[nn.Module],
        next_bank_ids: Sequence[str | None],
        *,
        mutable_state: bool = False,
        data_dependent_shape: bool = False,
        runtime_callback: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(bank_id, str) or not bank_id:
            raise ValueError("bank_id must be a non-empty string")
        if not isinstance(query, nn.Module):
            raise TypeError("query must be an nn.Module")
        normalized_operations = tuple(operations)
        normalized_next = tuple(next_bank_ids)
        if not normalized_operations or any(
            not isinstance(operation, nn.Module) for operation in normalized_operations
        ):
            raise TypeError("operations must contain at least one nn.Module")
        if len(normalized_operations) != len(normalized_next):
            raise ValueError("next_bank_ids must match operations")
        self.bank_id = bank_id
        self.query = query
        self.operations = nn.ModuleList(normalized_operations)
        self.next_bank_ids = normalized_next
        self.mutable_state = bool(mutable_state)
        self.data_dependent_shape = bool(data_dependent_shape)
        self.runtime_callback = bool(runtime_callback)


@dataclass(frozen=True)
class FederalTensorFederationManifest:
    """Provenance for a tensorized multi-Bank Federation graph."""

    schema: str
    compiler_ref: str
    source_ref: str
    source_snapshot_fingerprint: str
    bank_ids: tuple[str, ...]
    root_bank_id: str
    max_levels: int
    beam_width: int
    max_actions: int
    input_shape: tuple[int | None, ...] | None
    output_shape: tuple[int | None, ...] | None
    early_stop_semantics: str = "logical-carry-fixed-horizon"

    _component_reference: ClassVar[str] = "arti/federal-tensor-federation-manifest@1"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "compiler_ref": self.compiler_ref,
            "source_ref": self.source_ref,
            "source_snapshot_fingerprint": self.source_snapshot_fingerprint,
            "bank_ids": list(self.bank_ids),
            "root_bank_id": self.root_bank_id,
            "max_levels": self.max_levels,
            "beam_width": self.beam_width,
            "max_actions": self.max_actions,
            "input_shape": None if self.input_shape is None else list(self.input_shape),
            "output_shape": None if self.output_shape is None else list(self.output_shape),
            "early_stop_semantics": self.early_stop_semantics,
        }


class FederalTensorFederation(nn.Module):
    """Ordinary Tensor graph for a finite dynamic-Query Federation.

    Every Bank Query and action is evaluated in parallel for the current
    tensor beam.  Bank selection, action selection, K-wide expansion and
    terminal carry are Tensor indexing operations.  The Python objects used to
    construct the graph do not participate in its forward execution.
    """

    _component_reference = "arti/federal-tensor-federation@1"

    def __init__(
        self,
        banks: Sequence[FederalTensorBank],
        *,
        root_bank_index: int,
        max_levels: int,
        beam_width: int,
        next_bank_indices: Tensor,
        terminal_mask: Tensor,
        manifest: FederalTensorFederationManifest,
    ) -> None:
        super().__init__()
        normalized = tuple(banks)
        if not normalized:
            raise ValueError("banks must not be empty")
        self.banks = nn.ModuleList(normalized)
        self.root_bank_index = int(root_bank_index)
        self.max_levels = int(max_levels)
        self.beam_width = int(beam_width)
        self.max_actions = int(next_bank_indices.shape[1])
        self.register_buffer("_next_bank_indices", next_bank_indices.to(torch.long), persistent=True)
        self.register_buffer("_terminal_mask", terminal_mask.to(torch.bool), persistent=True)
        self.manifest = manifest

    def _bank_query_logits(self, state: Tensor) -> Tensor:
        batch, beam = state.shape[:2]
        flat = state.reshape(batch * beam, *state.shape[2:])
        padded: list[Tensor] = []
        for bank in self.banks:
            logits = bank.query(flat)
            padded.append(
                torch.nn.functional.pad(
                    logits,
                    (0, self.max_actions - len(bank.operations)),
                    value=float("-inf"),
                )
            )
        return torch.stack(padded, dim=1).reshape(
            batch,
            beam,
            len(self.banks),
            self.max_actions,
        )

    def _bank_action_values(self, state: Tensor) -> Tensor:
        batch, beam = state.shape[:2]
        flat = state.reshape(batch * beam, *state.shape[2:])
        padded: list[Tensor] = []
        for bank in self.banks:
            values = torch.stack([operation(flat) for operation in bank.operations], dim=1)
            if values.shape[1] < self.max_actions:
                zero_shape = (values.shape[0], self.max_actions - values.shape[1], *values.shape[2:])
                values = torch.cat((values, values.new_zeros(zero_shape)), dim=1)
            padded.append(values)
        return torch.stack(padded, dim=1).reshape(
            batch,
            beam,
            len(self.banks),
            self.max_actions,
            *state.shape[2:],
        )

    def forward(self, value: Tensor) -> Tensor:
        batch = value.shape[0]
        tail = value.shape[1:]
        state = value.unsqueeze(1).expand(batch, self.beam_width, *tail)
        bank_indices = torch.full(
            (batch, self.beam_width),
            self.root_bank_index,
            dtype=torch.long,
            device=value.device,
        )
        scores = value.new_zeros((batch, self.beam_width))
        done = torch.zeros(
            (batch, self.beam_width),
            dtype=torch.bool,
            device=value.device,
        )
        negative_infinity = float("-inf")
        for _ in range(self.max_levels):
            query_logits = self._bank_query_logits(state)
            action_values = self._bank_action_values(state)
            bank_index = bank_indices[..., None, None].expand(
                batch,
                self.beam_width,
                1,
                self.max_actions,
            )
            logits = query_logits.gather(2, bank_index).squeeze(2)
            values_index = bank_indices[..., None, None, None]
            values_index = values_index.expand(
                batch,
                self.beam_width,
                1,
                self.max_actions,
                *tail,
            )
            values = action_values.gather(2, values_index).squeeze(2)
            log_probability = torch.log_softmax(logits, dim=-1)
            action_scores = scores.unsqueeze(-1) + log_probability
            action_scores = torch.where(
                done.unsqueeze(-1),
                torch.full_like(action_scores, negative_infinity),
                action_scores,
            )
            next_banks = self._next_bank_indices[bank_indices]
            terminals = self._terminal_mask[bank_indices]
            carry_scores = torch.where(
                done,
                scores,
                torch.full_like(scores, negative_infinity),
            ).unsqueeze(-1)
            candidate_scores = torch.cat((action_scores, carry_scores), dim=-1)
            candidate_values = torch.cat((values, state.unsqueeze(2)), dim=2)
            candidate_banks = torch.cat((next_banks, bank_indices.unsqueeze(-1)), dim=-1)
            candidate_done = torch.cat((terminals, done.unsqueeze(-1)), dim=-1)
            flat_scores = candidate_scores.reshape(batch, -1)
            top_scores, top_indices = torch.topk(flat_scores, k=self.beam_width, dim=-1)
            flat_values = candidate_values.reshape(batch, -1, *tail)
            gather_values = top_indices.reshape(batch, self.beam_width, *([1] * len(tail)))
            gather_values = gather_values.expand(batch, self.beam_width, *tail)
            state = flat_values.gather(1, gather_values)
            flat_banks = candidate_banks.reshape(batch, -1)
            flat_done = candidate_done.reshape(batch, -1)
            bank_indices = flat_banks.gather(1, top_indices)
            done = flat_done.gather(1, top_indices)
            scores = top_scores
        terminal_scores = torch.where(
            done,
            scores,
            torch.full_like(scores, negative_infinity),
        )
        best_terminal = torch.argmax(terminal_scores, dim=-1)
        best_any = torch.argmax(scores, dim=-1)
        best = torch.where(done.any(dim=-1), best_terminal, best_any)
        gather_best = best.reshape(batch, 1, *([1] * len(tail))).expand(
            batch,
            1,
            *tail,
        )
        return state.gather(1, gather_best).squeeze(1)


class FederalTensorFederationCompiler:
    """Lower finite Bank/Query/action graphs without Python runtime dispatch."""

    _component_reference = "arti/federal-tensor-federation-compiler@1"

    @classmethod
    def compile(
        cls,
        banks: Mapping[str, FederalTensorBank] | Sequence[FederalTensorBank],
        *,
        root_bank_id: str,
        example_input: Tensor,
        source_ref: str,
        source_snapshot_fingerprint: str,
        max_levels: int,
        beam_width: int = 1,
        input_shape: Sequence[int | None] | None = None,
        output_shape: Sequence[int | None] | None = None,
    ) -> FederalTensorFederation:
        if not isinstance(example_input, Tensor) or not example_input.is_floating_point():
            raise TypeError("example_input must be a floating-point Tensor")
        if not source_ref or not source_snapshot_fingerprint:
            raise ValueError("source provenance is required")
        if isinstance(banks, Mapping):
            normalized = tuple(banks.values())
        else:
            normalized = tuple(banks)
        if not normalized or any(not isinstance(bank, FederalTensorBank) for bank in normalized):
            raise TypeError("banks must contain FederalTensorBank values")
        for bank in normalized:
            if bank.mutable_state:
                raise FederalCompileError(
                    f"Bank {bank.bank_id!r} has mutable state and remains runtime-only"
                )
            if bank.data_dependent_shape:
                raise FederalCompileError(
                    f"Bank {bank.bank_id!r} has data-dependent shape and remains runtime-only"
                )
            if bank.runtime_callback:
                raise FederalCompileError(
                    f"Bank {bank.bank_id!r} has a runtime callback and remains runtime-only"
                )
        bank_ids = tuple(bank.bank_id for bank in normalized)
        if len(set(bank_ids)) != len(bank_ids):
            raise FederalCompileError("tensorized Federation Bank IDs must be unique")
        if root_bank_id not in bank_ids:
            raise FederalCompileError("root_bank_id is not present in the tensorized Federation")
        if isinstance(max_levels, bool) or not isinstance(max_levels, int) or max_levels <= 0:
            raise ValueError("max_levels must be a positive integer")
        if isinstance(beam_width, bool) or not isinstance(beam_width, int) or beam_width <= 0:
            raise ValueError("beam_width must be a positive integer")
        bank_index = {bank_id: index for index, bank_id in enumerate(bank_ids)}
        max_actions = max(len(bank.operations) for bank in normalized)
        for bank in normalized:
            with torch.no_grad():
                logits = bank.query(example_input)
                if not isinstance(logits, Tensor) or logits.shape != (
                    example_input.shape[0],
                    len(bank.operations),
                ):
                    raise FederalCompileError(
                        f"Bank {bank.bank_id!r} Query must return [B, action] logits"
                    )
                for operation in bank.operations:
                    result = operation(example_input)
                    if not isinstance(result, Tensor) or tuple(result.shape) != tuple(example_input.shape):
                        raise FederalCompileError(
                            "tensorized Federation actions must preserve the compiled Tensor ABI"
                        )
            for next_bank_id in bank.next_bank_ids:
                if next_bank_id is not None and next_bank_id not in bank_index:
                    raise FederalCompileError(
                        f"Bank {bank.bank_id!r} references unknown child {next_bank_id!r}"
                    )
        next_bank_indices = torch.zeros(
            (len(normalized), max_actions), dtype=torch.long, device=example_input.device
        )
        terminal_mask = torch.ones(
            (len(normalized), max_actions), dtype=torch.bool, device=example_input.device
        )
        for bank_index_value, bank in enumerate(normalized):
            for action_index, next_bank_id in enumerate(bank.next_bank_ids):
                if next_bank_id is not None:
                    next_bank_indices[bank_index_value, action_index] = bank_index[next_bank_id]
                    terminal_mask[bank_index_value, action_index] = False

        def normalize_shape(
            shape: Sequence[int | None] | None,
            *,
            field: str,
        ) -> tuple[int | None, ...] | None:
            if shape is None:
                return None
            result = tuple(shape)
            if not result or any(
                item is not None
                and (isinstance(item, bool) or not isinstance(item, int) or item <= 0)
                for item in result
            ):
                raise ValueError(f"{field} entries must be positive integers or None")
            if not _shape_matches(example_input.shape, result):
                raise FederalCompileError(f"{field} does not match example_input")
            return result

        manifest = FederalTensorFederationManifest(
            schema=FederalTensorFederationManifest._component_reference,
            compiler_ref=cls._component_reference,
            source_ref=source_ref,
            source_snapshot_fingerprint=source_snapshot_fingerprint,
            bank_ids=bank_ids,
            root_bank_id=root_bank_id,
            max_levels=max_levels,
            beam_width=beam_width,
            max_actions=max_actions,
            input_shape=normalize_shape(input_shape, field="input_shape"),
            output_shape=normalize_shape(output_shape, field="output_shape"),
        )
        return FederalTensorFederation(
            deepcopy(normalized),
            root_bank_index=bank_index[root_bank_id],
            max_levels=max_levels,
            beam_width=beam_width,
            next_bank_indices=next_bank_indices,
            terminal_mask=terminal_mask,
            manifest=manifest,
        )

    @staticmethod
    def export(
        graph: FederalTensorFederation,
        example_input: Tensor,
        *,
        dynamic_shapes: object | None = None,
    ) -> object:
        if not isinstance(graph, FederalTensorFederation):
            raise TypeError("graph must be a FederalTensorFederation")
        export = getattr(torch, "export", None)
        if export is None or not hasattr(export, "export"):
            raise FederalCompileError("this PyTorch runtime does not provide torch.export")
        return export.export(graph, (example_input,), dynamic_shapes=dynamic_shapes)


@dataclass(frozen=True)
class FederalStatefulGraphManifest:
    """Provenance for a graph with explicit mutable state lanes.

    The graph never mutates a parameter or a Python object during forward.  A
    Bank and an effect lane are ordinary tensor inputs and their next values
    are ordinary tensor outputs.  ``max_steps`` is a host budget, while the
    graph also contains a tensor stop predicate.
    """

    schema: str
    compiler_ref: str
    source_ref: str
    source_snapshot_fingerprint: str
    transition_ref: str
    max_steps: int
    value_shape: tuple[int | None, ...]
    bank_state_shape: tuple[int | None, ...]
    effect_state_shape: tuple[int | None, ...]
    state_semantics: str = "explicit-bank-effect-input-output"
    refine_semantics: str = "while-loop-stop-or-host-bound"

    _component_reference: ClassVar[str] = "arti/federal-stateful-graph-manifest@1"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "compiler_ref": self.compiler_ref,
            "source_ref": self.source_ref,
            "source_snapshot_fingerprint": self.source_snapshot_fingerprint,
            "transition_ref": self.transition_ref,
            "max_steps": self.max_steps,
            "value_shape": list(self.value_shape),
            "bank_state_shape": list(self.bank_state_shape),
            "effect_state_shape": list(self.effect_state_shape),
            "state_semantics": self.state_semantics,
            "refine_semantics": self.refine_semantics,
        }


class FederalStatefulRefineGraph(nn.Module):
    """Compile a stateful Refine recurrence into a tensor while-loop.

    ``transition`` receives ``(value, bank_state, effect_state)`` and returns
    ``(next_value, next_bank_state, next_effect_state, stop)``.  The caller's
    input tensors are therefore the only mutable state visible to the graph;
    the module's parameters remain immutable during forward.  ``stop`` is a
    boolean tensor with one entry per batch item.

    The loop is open-ended with respect to the learned stop predicate, but a
    finite host budget is always present as a termination guard.  This is the
    exportable meaning of unbounded Refine: no static unrolling, no Python
    ``break``, and no claim of literally infinite physical execution.
    """

    _component_reference = "arti/federal-stateful-refine-graph@1"

    def __init__(
        self,
        transition: nn.Module,
        *,
        max_steps: int,
        manifest: FederalStatefulGraphManifest | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(transition, nn.Module):
            raise TypeError("transition must be an nn.Module")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        self.transition = transition
        self.max_steps = int(max_steps)
        self.register_buffer(
            "_max_steps_tensor",
            torch.tensor(self.max_steps, dtype=torch.int64),
            persistent=True,
        )
        self.manifest = manifest

    def forward(
        self,
        value: Tensor,
        bank_state: Tensor,
        effect_state: Tensor,
        max_steps: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        batch = value.shape[0]
        budget = torch.minimum(
            max_steps.to(dtype=torch.int64),
            self._max_steps_tensor.to(device=value.device),
        )
        step = torch.zeros((), dtype=torch.int64, device=value.device)
        done = torch.zeros((batch,), dtype=torch.bool, device=value.device)
        steps = torch.zeros((batch,), dtype=torch.int64, device=value.device)

        def condition(
            current_step: Tensor,
            current_value: Tensor,
            current_bank: Tensor,
            current_effect: Tensor,
            current_done: Tensor,
            current_steps: Tensor,
        ) -> Tensor:
            del current_value, current_bank, current_effect, current_steps
            return (current_step < budget) & (~current_done).any()

        def body(
            current_step: Tensor,
            current_value: Tensor,
            current_bank: Tensor,
            current_effect: Tensor,
            current_done: Tensor,
            current_steps: Tensor,
        ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
            next_value, next_bank, next_effect, stop = self.transition(
                current_value,
                current_bank,
                current_effect,
            )
            active = ~current_done
            value_active = active.reshape(batch, *([1] * (current_value.ndim - 1)))
            bank_active = active.reshape(batch, *([1] * (current_bank.ndim - 1)))
            effect_active = active.reshape(batch, *([1] * (current_effect.ndim - 1)))
            next_value = torch.where(value_active, next_value, current_value)
            next_bank = torch.where(bank_active, next_bank, current_bank)
            next_effect = torch.where(effect_active, next_effect, current_effect)
            next_done = current_done | stop
            next_steps = current_steps + active.to(dtype=torch.int64)
            return (
                current_step + 1,
                next_value,
                next_bank,
                next_effect,
                next_done,
                next_steps,
            )

        _, value, bank_state, effect_state, done, steps = torch.while_loop(
            condition,
            body,
            (step, value, bank_state, effect_state, done, steps),
        )
        return value, bank_state, effect_state, done, steps


class FederalStatefulGraphCompiler:
    """Lower explicit Bank/effect state and open-ended Refine to a graph."""

    _component_reference = "arti/federal-stateful-graph-compiler@1"

    @classmethod
    def compile(
        cls,
        transition: nn.Module,
        *,
        example_value: Tensor,
        example_bank_state: Tensor,
        example_effect_state: Tensor,
        source_ref: str,
        source_snapshot_fingerprint: str,
        max_steps: int,
    ) -> FederalStatefulRefineGraph:
        examples = (example_value, example_bank_state, example_effect_state)
        if any(not isinstance(item, Tensor) for item in examples):
            raise TypeError("stateful graph examples must be Tensors")
        if not isinstance(example_value, Tensor) or not example_value.is_floating_point():
            raise TypeError("example_value must be a floating-point Tensor")
        if not source_ref or not source_snapshot_fingerprint:
            raise ValueError("source provenance is required")
        if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
            raise ValueError("max_steps must be a positive integer")
        batch = example_value.shape[0]
        if any(item.ndim == 0 or item.shape[0] != batch for item in examples):
            raise FederalCompileError("state lanes must have the same batch dimension")
        if not isinstance(transition, nn.Module):
            raise TypeError("transition must be an nn.Module")

        snapshots = tuple(item.detach().clone() for item in examples)
        with torch.no_grad():
            result = transition(*examples)
        if not isinstance(result, (tuple, list)) or len(result) != 4:
            raise FederalCompileError(
                "stateful transition must return next_value, next_bank, next_effect, stop"
            )
        next_value, next_bank, next_effect, stop = result
        next_states = (next_value, next_bank, next_effect)
        if any(not isinstance(item, Tensor) for item in next_states):
            raise FederalCompileError("stateful transition outputs must be Tensors")
        for current, updated in zip(examples, next_states, strict=True):
            if tuple(updated.shape) != tuple(current.shape):
                raise FederalCompileError(
                    "stateful transition must preserve each explicit state lane shape"
                )
        if not isinstance(stop, Tensor) or stop.dtype is not torch.bool or stop.shape != (batch,):
            raise FederalCompileError("stateful transition stop must be a bool Tensor shaped [B]")
        for before, after in zip(snapshots, examples, strict=True):
            if not torch.equal(before, after):
                raise FederalCompileError("stateful transition mutated an input Tensor in place")

        def shape_abi(value: Tensor) -> tuple[int | None, ...]:
            return (None, *tuple(int(item) for item in value.shape[1:]))

        manifest = FederalStatefulGraphManifest(
            schema=FederalStatefulGraphManifest._component_reference,
            compiler_ref=cls._component_reference,
            source_ref=source_ref,
            source_snapshot_fingerprint=source_snapshot_fingerprint,
            transition_ref=_module_ref(transition),
            max_steps=max_steps,
            value_shape=shape_abi(example_value),
            bank_state_shape=shape_abi(example_bank_state),
            effect_state_shape=shape_abi(example_effect_state),
        )
        return FederalStatefulRefineGraph(
            deepcopy(transition),
            max_steps=max_steps,
            manifest=manifest,
        )

    @staticmethod
    def export(
        graph: FederalStatefulRefineGraph,
        example_value: Tensor,
        example_bank_state: Tensor,
        example_effect_state: Tensor,
        max_steps: Tensor | int,
        *,
        dynamic_shapes: object | None = None,
    ) -> object:
        if not isinstance(graph, FederalStatefulRefineGraph):
            raise TypeError("graph must be a FederalStatefulRefineGraph")
        if isinstance(max_steps, int):
            max_steps = torch.tensor(max_steps, dtype=torch.int64, device=example_value.device)
        export = getattr(torch, "export", None)
        if export is None or not hasattr(export, "export"):
            raise FederalCompileError("this PyTorch runtime does not provide torch.export")
        return export.export(
            graph,
            (example_value, example_bank_state, example_effect_state, max_steps),
            dynamic_shapes=dynamic_shapes,
        )


@dataclass(frozen=True)
class FederalRaggedShapeManifest:
    """Provenance for a bounded logical-shape graph."""

    schema: str
    compiler_ref: str
    source_ref: str
    source_snapshot_fingerprint: str
    transform_ref: str
    capacity: int
    value_rank: int
    shape_semantics: str = "bounded-ragged-logical-shape"

    _component_reference: ClassVar[str] = "arti/federal-ragged-shape-manifest@1"

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": self.schema,
            "compiler_ref": self.compiler_ref,
            "source_ref": self.source_ref,
            "source_snapshot_fingerprint": self.source_snapshot_fingerprint,
            "transform_ref": self.transform_ref,
            "capacity": self.capacity,
            "value_rank": self.value_rank,
            "shape_semantics": self.shape_semantics,
        }


class FederalRaggedShapeGraph(nn.Module):
    """Represent data-dependent sequence shapes as values plus logical lengths.

    Physical storage is bounded by ``capacity`` so it is exportable.  The
    logical length is a Tensor produced by the transform and can depend on the
    data.  Values outside that length are masked to zero.  This is the graph
    contract for dynamic/ragged shapes; it does not pretend that an arbitrary
    data-dependent allocation or rank change is a static Tensor ABI.
    """

    _component_reference = "arti/federal-ragged-shape-graph@1"

    def __init__(
        self,
        transform: nn.Module,
        *,
        capacity: int,
        manifest: FederalRaggedShapeManifest | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(transform, nn.Module):
            raise TypeError("transform must be an nn.Module")
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.transform = transform
        self.capacity = int(capacity)
        self.manifest = manifest

    def forward(self, values: Tensor, lengths: Tensor) -> tuple[Tensor, Tensor]:
        next_values, next_lengths = self.transform(values, lengths)
        bounded_lengths = torch.clamp(next_lengths.to(dtype=torch.int64), 0, self.capacity)
        positions = torch.arange(self.capacity, device=values.device)
        active = positions.unsqueeze(0) < bounded_lengths.unsqueeze(1)
        active = active.reshape(values.shape[0], self.capacity, *([1] * (values.ndim - 2)))
        return torch.where(active, next_values, torch.zeros_like(next_values)), bounded_lengths


class FederalRaggedShapeCompiler:
    """Lower bounded data-dependent logical shapes without Python allocation."""

    _component_reference = "arti/federal-ragged-shape-compiler@1"

    @classmethod
    def compile(
        cls,
        transform: nn.Module,
        *,
        example_values: Tensor,
        example_lengths: Tensor,
        source_ref: str,
        source_snapshot_fingerprint: str,
    ) -> FederalRaggedShapeGraph:
        if not isinstance(transform, nn.Module):
            raise TypeError("transform must be an nn.Module")
        if not isinstance(example_values, Tensor) or example_values.ndim < 2:
            raise TypeError("example_values must have shape [B, capacity, ...]")
        if not isinstance(example_lengths, Tensor) or example_lengths.shape != (
            example_values.shape[0],
        ):
            raise TypeError("example_lengths must have shape [B]")
        if example_lengths.dtype not in (torch.int32, torch.int64):
            raise TypeError("example_lengths must be an integer Tensor")
        if not source_ref or not source_snapshot_fingerprint:
            raise ValueError("source provenance is required")
        with torch.no_grad():
            result = transform(example_values, example_lengths)
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise FederalCompileError("ragged transform must return values and logical lengths")
        next_values, next_lengths = result
        if not isinstance(next_values, Tensor) or tuple(next_values.shape) != tuple(
            example_values.shape
        ):
            raise FederalCompileError("ragged transform must preserve the padded value ABI")
        if not isinstance(next_lengths, Tensor) or next_lengths.shape != example_lengths.shape:
            raise FederalCompileError("ragged transform must return one logical length per batch item")
        if next_lengths.dtype not in (torch.int32, torch.int64):
            raise FederalCompileError("ragged logical lengths must be integer Tensors")
        manifest = FederalRaggedShapeManifest(
            schema=FederalRaggedShapeManifest._component_reference,
            compiler_ref=cls._component_reference,
            source_ref=source_ref,
            source_snapshot_fingerprint=source_snapshot_fingerprint,
            transform_ref=_module_ref(transform),
            capacity=int(example_values.shape[1]),
            value_rank=example_values.ndim,
        )
        return FederalRaggedShapeGraph(
            deepcopy(transform),
            capacity=int(example_values.shape[1]),
            manifest=manifest,
        )

    @staticmethod
    def export(
        graph: FederalRaggedShapeGraph,
        example_values: Tensor,
        example_lengths: Tensor,
        *,
        dynamic_shapes: object | None = None,
    ) -> object:
        if not isinstance(graph, FederalRaggedShapeGraph):
            raise TypeError("graph must be a FederalRaggedShapeGraph")
        export = getattr(torch, "export", None)
        if export is None or not hasattr(export, "export"):
            raise FederalCompileError("this PyTorch runtime does not provide torch.export")
        return export.export(
            graph,
            (example_values, example_lengths),
            dynamic_shapes=dynamic_shapes,
        )


class FederalPathCompiler:
    """Reference compiler for a path whose Bank choices are already frozen."""

    _component_reference = "arti/federal-static-compiler@1"

    @classmethod
    def assess(cls, path: FederalPath) -> FederalCompileAssessment:
        if not isinstance(path, FederalPath):
            raise TypeError("path must be FederalPath")
        return path.assessment()

    @classmethod
    def compile(cls, path: FederalPath) -> CompiledFederalNetwork:
        assessment = cls.assess(path)
        if not assessment.compilable:
            reason = "; ".join(assessment.reasons) or assessment.classification
            raise FederalCompileError(f"path is not exact-static compilable: {reason}")
        manifest = FederalCompileManifest(
            schema=FederalCompileManifest._component_reference,
            compiler_ref=cls._component_reference,
            mode="exact-static",
            source_ref=path.source_ref,
            source_snapshot_fingerprint=path.source_snapshot_fingerprint,
            path_ids=path.path_ids,
            operation_refs=path.operation_refs,
            dependency_refs=path.dependency_refs,
            terminal_abi_ref=path.terminal_abi_ref,
            refine_steps=path.refine_steps,
            input_shape=path.input_shape,
            output_shape=path.output_shape,
        )
        # Deep-copying makes the compiled artifact independent from the live
        # Federal path and its mutable runtime ownership.
        return CompiledFederalNetwork(deepcopy(tuple(path.operations)), manifest)


def compile_federal_path(path: FederalPath) -> CompiledFederalNetwork:
    """Compile one frozen Federal path into an ordinary PyTorch network."""

    return FederalPathCompiler.compile(path)


__all__ = [
    "CompiledFederalNetwork",
    "FederalCompileAssessment",
    "FederalCompileError",
    "FederalCompileManifest",
    "FederalTensorQueryAdapter",
    "FederalTensorQueryBlock",
    "FederalTensorQueryCompiler",
    "FederalTensorQueryManifest",
    "FederalTensorBank",
    "FederalTensorFederation",
    "FederalTensorFederationCompiler",
    "FederalTensorFederationManifest",
    "FederalFormulaBlock",
    "FederalParallel",
    "FederalPath",
    "FederalPathCompiler",
    "FederalRefine",
    "FederalResidual",
    "FederalStaticFold",
    "FederalStaticUnFold",
    "FederalTopologyBlock",
    "compile_federal_path",
]
