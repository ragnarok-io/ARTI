"""Versioned Recall Top-K branch batches for true batched refine."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar
from collections.abc import Mapping

import torch
from torch import Tensor, nn


class BatchedRefineContractError(ValueError):
    """Raised when a candidate batch cannot represent independent branches."""


_RNG_IDENTIFIER_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-"
)
_RNG_PHASES = frozenset(
    {"candidate-route", "refine-route", "half-survival", "recall-dropout"}
)


@dataclass(frozen=True)
class ExecutionRNGPlan:
    """Run-before identity for branch-origin keyed stochastic execution.

    The plan deliberately keys samples by caller-provided stable identities and
    branches by canonical candidate origin. Physical batch or branch position
    therefore does not define the random stream.
    """

    seed: int
    run_nonce: str
    stream_key: str
    sample_keys: tuple[str, ...]
    algorithm: str = "sha256-seeded-torch-generator@2"
    _runtime_contract_ref: ClassVar[str] = "arti/execution-rng-plan@2"

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not 0 <= self.seed < 2**63:
            raise ValueError("seed must be in [0, 2**63)")
        if (
            not isinstance(self.run_nonce, str)
            or not self.run_nonce
            or len(self.run_nonce) > 128
            or any(character not in _RNG_IDENTIFIER_CHARS for character in self.run_nonce)
        ):
            raise ValueError("run_nonce must be a canonical non-empty identifier")
        if (
            not isinstance(self.stream_key, str)
            or not self.stream_key
            or len(self.stream_key) > 128
            or any(
                character not in _RNG_IDENTIFIER_CHARS
                for character in self.stream_key
            )
        ):
            raise ValueError("stream_key must be a canonical non-empty identifier")
        keys = tuple(self.sample_keys)
        if not keys or len(set(keys)) != len(keys):
            raise ValueError("sample_keys must contain unique stable identities")
        for key in keys:
            if (
                not isinstance(key, str)
                or not key
                or len(key) > 128
                or any(character not in _RNG_IDENTIFIER_CHARS for character in key)
            ):
                raise ValueError("sample_keys must be canonical non-empty identifiers")
        if self.algorithm != "sha256-seeded-torch-generator@2":
            raise ValueError("unsupported execution RNG algorithm")
        object.__setattr__(self, "sample_keys", keys)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(
                {
                    "ref": self._runtime_contract_ref,
                    "algorithm": self.algorithm,
                    "seed": self.seed,
                    "run_nonce": self.run_nonce,
                    "stream_key": self.stream_key,
                    "sample_keys": self.sample_keys,
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode("ascii")
        ).hexdigest()

    def _derived_seed(
        self,
        *,
        sample_key: str,
        branch_origin: int,
        phase: str,
        refine_step: int,
    ) -> int:
        if phase not in _RNG_PHASES:
            raise BatchedRefineContractError(f"unsupported execution RNG phase {phase!r}")
        payload = json.dumps(
            {
                "algorithm": self.algorithm,
                "seed": self.seed,
                "run_nonce": self.run_nonce,
                "stream_key": self.stream_key,
                "sample_key": sample_key,
                "branch_origin": branch_origin,
                "phase": phase,
                "refine_step": refine_step,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63)

    def bind(self, branch_origin_index: Tensor) -> "_KeyedBranchRandomSource":
        if (
            not isinstance(branch_origin_index, Tensor)
            or branch_origin_index.dtype != torch.long
            or branch_origin_index.ndim != 2
        ):
            raise BatchedRefineContractError(
                "branch_origin_index must be torch.long [B,K]"
            )
        if branch_origin_index.shape[0] != len(self.sample_keys):
            raise BatchedRefineContractError(
                "execution RNG sample_keys must match the Batched Refine batch"
            )
        origins = branch_origin_index.detach().to("cpu").contiguous()
        return _KeyedBranchRandomSource(self, origins)

    def bind_candidate(self, batch_size: int) -> "_KeyedBranchRandomSource":
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        if batch_size != len(self.sample_keys):
            raise BatchedRefineContractError(
                "execution RNG sample_keys must match the candidate query batch"
            )
        return _KeyedBranchRandomSource(
            self,
            torch.full((batch_size, 1), -1, dtype=torch.long),
        )


class _KeyedBranchRandomSource:
    def __init__(self, plan: ExecutionRNGPlan, origins: Tensor) -> None:
        self.plan = plan
        self.origins = origins
        self.batch, self.branches = origins.shape
        self.row_sample_index = torch.arange(self.batch, dtype=torch.long).repeat_interleave(
            self.branches
        )
        self.row_origin = origins.reshape(-1).contiguous()

    def select_rows(self, flat_index: Tensor) -> "_KeyedBranchRandomSource":
        """Keep RNG identity tied to canonical sample and branch origin."""

        if not isinstance(flat_index, Tensor) or flat_index.dtype != torch.long:
            raise BatchedRefineContractError("packed RNG row index must be torch.long")
        cpu_index = flat_index.detach().to("cpu").contiguous()
        selected = object.__new__(_KeyedBranchRandomSource)
        selected.plan = self.plan
        selected.origins = self.origins
        selected.batch = int(cpu_index.numel())
        selected.branches = 1
        selected.row_sample_index = self.row_sample_index.index_select(0, cpu_index)
        selected.row_origin = self.row_origin.index_select(0, cpu_index)
        return selected

    def uniform(
        self,
        phase: str,
        refine_step: int,
        reference: Tensor,
    ) -> Tensor:
        if not isinstance(refine_step, int) or refine_step < 0:
            raise BatchedRefineContractError("refine_step must be non-negative")
        if reference.shape[0] != self.row_origin.numel():
            raise BatchedRefineContractError(
                "keyed random reference must match the bound execution rows"
            )
        tail = tuple(reference.shape[1:])
        rows: list[Tensor] = []
        for row in range(self.row_origin.numel()):
            sample = int(self.row_sample_index[row])
            origin = int(self.row_origin[row])
            generator = torch.Generator(device=reference.device)
            generator.manual_seed(
                self.plan._derived_seed(
                    sample_key=self.plan.sample_keys[sample],
                    branch_origin=origin,
                    phase=phase,
                    refine_step=refine_step,
                )
            )
            rows.append(
                torch.rand(
                    tail,
                    device=reference.device,
                    dtype=torch.float32,
                    generator=generator,
                )
            )
        uniform = torch.stack(rows, dim=0).to(dtype=reference.dtype)
        # Casting a float32 value just below one to float16/bfloat16 can round
        # it to exactly one. Keep the public RNG contract in [0, 1) in the
        # destination dtype used by Half and route exploration.
        upper = torch.nextafter(
            torch.ones((), device=reference.device, dtype=reference.dtype),
            torch.zeros((), device=reference.device, dtype=reference.dtype),
        )
        return torch.minimum(uniform, upper)


def _tensor_content_fingerprint(value: Tensor) -> str:
    payload = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def _operation_owned_tensors(operation: nn.Module) -> tuple[tuple[str, Tensor], ...]:
    owned: list[tuple[str, Tensor]] = [
        *((f"parameter:{name}", value) for name, value in operation.named_parameters()),
        *((f"buffer:{name}", value) for name, value in operation.named_buffers()),
    ]
    route = getattr(operation, "route", None)
    if route is not None:
        for name in ("weights", "valid_mask", "fire_mask", "commit_mask"):
            value = getattr(route, name, None)
            if isinstance(value, Tensor):
                owned.append((f"route:{name}", value))
    factors = getattr(operation, "factors", None)
    if isinstance(factors, Tensor):
        owned.append(("factors", factors))
    deduplicated: dict[int, tuple[str, Tensor]] = {}
    for name, value in owned:
        deduplicated.setdefault(id(value), (name, value))
    return tuple(deduplicated.values())


def _flatten_branch_origin(
    branch_origin_index: Tensor,
    *,
    route_batch: int,
) -> Tensor:
    if (
        not isinstance(branch_origin_index, Tensor)
        or branch_origin_index.dtype != torch.long
        or branch_origin_index.ndim != 2
        or branch_origin_index.numel() != route_batch
    ):
        raise BatchedRefineContractError(
            "branch_origin_index must be torch.long [B,K] matching the resident route"
        )
    batch, branches = branch_origin_index.shape
    base = torch.arange(
        batch,
        device=branch_origin_index.device,
        dtype=torch.long,
    ).unsqueeze(1)
    return (base * branches + branch_origin_index).reshape(-1).contiguous()


class BatchedRefineOperation(nn.Module):
    """Versioned adapter for an existing Formula or Topology operation."""

    _component_reference: ClassVar[str] = "arti/batched-refine-operation@1"

    def __init__(self, operation: nn.Module) -> None:
        super().__init__()
        from .gpu_resident import (
            FormulaResidentOperation,
            TopologyFormulaResidentOperation,
        )

        if not isinstance(
            operation,
            (FormulaResidentOperation, TopologyFormulaResidentOperation),
        ):
            raise BatchedRefineContractError(
                "BatchedRefineOperation supports existing FormulaResidentOperation "
                "or TopologyFormulaResidentOperation only"
            )
        self.operation = operation
        self.operation_ref = operation._component_reference
        self._capture_operation_lineage()
        self.register_load_state_dict_post_hook(self._after_state_load)

    def _capture_operation_lineage(self) -> None:
        operation = self.operation
        lineage = []
        tensor_config = []
        for name, value in _operation_owned_tensors(operation):
            lineage.append(
                (name, id(value), _require_tensor_version(value, name=name))
            )
            tensor_config.append(
                {
                    "name": name,
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "content": _tensor_content_fingerprint(value),
                }
            )
        self._lineage = tuple(lineage)
        route = operation.route
        self.formula_route_fingerprint = _config_fingerprint(
            {
                "weights": _tensor_content_fingerprint(route.weights),
                "valid": _tensor_content_fingerprint(route.valid_mask),
                "fire": _tensor_content_fingerprint(route.fire_mask),
                "commit": _tensor_content_fingerprint(route.commit_mask),
                "estimator": route.estimator,
            }
        )
        topology_refs = []
        topology_contract_fingerprints = []
        fold = getattr(operation, "fold", None)
        unfold = getattr(operation, "unfold", None)
        if fold is not None:
            topology_refs.append(fold._component_reference)
            topology_refs.append(fold.topology._component_reference)
            topology_contract_fingerprints.append(
                fold.topology.contract_fingerprint
            )
        if unfold is not None:
            topology_refs.append(unfold._component_reference)
            topology_contract_fingerprints.append(
                unfold.inverse_contract.contract_fingerprint
            )
        self.topology_refs = tuple(topology_refs)
        self.topology_contract_fingerprints = tuple(
            topology_contract_fingerprints
        )
        self.config_fingerprint = _config_fingerprint(
            {
                "ref": self._component_reference,
                "operation_ref": self.operation_ref,
                "resident_refine_steps": operation.resident_refine_steps,
                "compute_config_fingerprint": operation.compute.execution_config_fingerprint,
                "formula_route_fingerprint": self.formula_route_fingerprint,
                "topology_refs": self.topology_refs,
                "topology_contract_fingerprints": (
                    self.topology_contract_fingerprints
                ),
                "tensor_config": tensor_config,
            }
        )

    def _after_state_load(self, _module: nn.Module, _incompatible_keys: object) -> None:
        self._capture_operation_lineage()

    def work_per_execution(
        self,
        branch_origin_index: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Return per-flat-branch Formula cell, fire, and commit counts."""

        route = self.operation.route
        multiplier = self.operation.resident_refine_steps
        cells = route.valid_mask.sum(dim=(1, 2), dtype=torch.int64) * multiplier
        fire = route.fire_mask.sum(dim=(1, 2), dtype=torch.int64) * multiplier
        commit = route.commit_mask.sum(dim=(1, 2), dtype=torch.int64) * multiplier
        if branch_origin_index is not None:
            index = _flatten_branch_origin(
                branch_origin_index,
                route_batch=route.weights.shape[0],
            )
            cells = cells.index_select(0, index)
            fire = fire.index_select(0, index)
            commit = commit.index_select(0, index)
        return cells, fire, commit

    def indexed(
        self,
        branch_origin_index: Tensor,
        *,
        active_flat_index: Tensor | None = None,
        trace_records: list[tuple[tuple[object, ...], Tensor | None, int]] | None = None,
    ) -> nn.Module:
        """Bind one runtime-only branch ordering to the resident operation."""

        return _BranchIndexedBatchedRefineOperation(
            self,
            branch_origin_index,
            active_flat_index=active_flat_index,
            trace_records=trace_records,
        )

    def assert_unchanged(self) -> None:
        current = {
            name: value for name, value in _operation_owned_tensors(self.operation)
        }
        if set(current) != {name for name, _, _ in self._lineage}:
            raise BatchedRefineContractError("Batched Refine operation lineage changed")
        for name, token, version in self._lineage:
            value = current[name]
            if id(value) != token or _tensor_version(value) != version:
                raise BatchedRefineContractError(
                    "Batched Refine operation changed after plan construction"
                )

    def forward(
        self,
        value: Tensor,
        validity: Tensor,
        exposed: Tensor,
        intervened: Tensor,
    ) -> Tensor:
        return self.operation(value, validity, exposed, intervened)


class _BranchIndexedBatchedRefineOperation(nn.Module):
    """Runtime-only Formula/Topology view aligned to candidate branch origins."""

    def __init__(
        self,
        source: BatchedRefineOperation,
        branch_origin_index: Tensor,
        *,
        active_flat_index: Tensor | None,
        trace_records: list[tuple[tuple[object, ...], Tensor | None, int]] | None,
    ) -> None:
        super().__init__()
        route_batch = source.operation.route.weights.shape[0]
        flat_index = _flatten_branch_origin(
            branch_origin_index,
            route_batch=route_batch,
        )
        if active_flat_index is not None:
            flat_index = flat_index.index_select(0, active_flat_index)
        if active_flat_index is None:
            route, factors = source.operation.indexed_inputs(flat_index)
        else:
            route, factors = source.operation.selected_inputs(flat_index)
        self.source = source
        self.route = route
        self.factors = factors
        self.trace_records = trace_records

    def _record_trace(
        self,
        traces: tuple[object, ...],
        permutation: Tensor | None,
        active_count: int,
    ) -> None:
        if self.trace_records is not None:
            self.trace_records.append((traces, permutation, active_count))

    def forward(
        self,
        value: Tensor,
        validity: Tensor,
        exposed: Tensor,
        intervened: Tensor,
    ) -> Tensor:
        return self.source.operation._forward_with_route(
            value,
            validity,
            exposed,
            intervened,
            route=self.route,
            factors=self.factors,
            trace_callback=(
                None if self.trace_records is None else self._record_trace
            ),
        )


@dataclass(frozen=True)
class BatchedRefinePlan:
    """Versioned composition plan for K Recall trajectories."""

    operation: BatchedRefineOperation | None = None
    execution_layout: str = "static_capacity"
    schema_version: int = 1
    _component_reference: ClassVar[str] = "arti/batched-refine-plan@1"

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise BatchedRefineContractError("unsupported BatchedRefinePlan schema")
        if self.operation is not None and not isinstance(
            self.operation, BatchedRefineOperation
        ):
            raise TypeError("operation must be BatchedRefineOperation or None")
        if self.execution_layout not in {"static_capacity", "packed_active"}:
            raise BatchedRefineContractError(
                "execution_layout must be 'static_capacity' or 'packed_active'"
            )
        if self.operation is not None:
            self.operation.assert_unchanged()

    @classmethod
    def recall_only(
        cls, *, execution_layout: str = "static_capacity"
    ) -> BatchedRefinePlan:
        return cls(execution_layout=execution_layout)

    @classmethod
    def compose(
        cls,
        operation: nn.Module,
        *,
        execution_layout: str = "static_capacity",
    ) -> BatchedRefinePlan:
        return cls(
            BatchedRefineOperation(operation),
            execution_layout=execution_layout,
        )

    @property
    def config_fingerprint(self) -> str:
        return _config_fingerprint(
            {
                "ref": self._component_reference,
                "schema_version": self.schema_version,
                "execution_layout": self.execution_layout,
                "step_order": [
                    "candidate-seed",
                    "recall",
                    "optional-formula-topology-operation",
                    "branch-state-update",
                    "recall-requery",
                ],
                "operation_ref": (
                    None if self.operation is None else self.operation.operation_ref
                ),
                "operation_config_fingerprint": (
                    None
                    if self.operation is None
                    else self.operation.config_fingerprint
                ),
                "formula_route_fingerprint": (
                    None
                    if self.operation is None
                    else self.operation.formula_route_fingerprint
                ),
                "topology_refs": (
                    ()
                    if self.operation is None
                    else self.operation.topology_refs
                ),
                "topology_contract_fingerprints": (
                    ()
                    if self.operation is None
                    else self.operation.topology_contract_fingerprints
                ),
            }
        )

    def assert_unchanged(self) -> None:
        if self.operation is not None:
            self.operation.assert_unchanged()


def _config_fingerprint(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _assert_tensor(condition: Tensor, message: str) -> None:
    """Fail closed without synchronizing a CUDA hot path."""

    scalar = torch.all(condition)
    if scalar.device.type == "cpu":
        if not bool(scalar):
            raise BatchedRefineContractError(message)
        return
    torch._assert_async(scalar, message)


def _tensor_version(value: Tensor) -> int:
    """Return PyTorch mutation lineage, or -1 for inference-mode tensors."""

    try:
        return value._version
    except RuntimeError:
        return -1


def _require_tensor_version(value: Tensor, *, name: str) -> int:
    version = _tensor_version(value)
    if version < 0:
        raise BatchedRefineContractError(
            f"{name} has no mutation version; construct and load Recall outside "
            "torch.inference_mode()"
        )
    return version


def _versioned_runtime_tensor(value: Tensor) -> Tensor:
    """Keep ordinary tensors zero-copy and snapshot inference tensors."""

    if _tensor_version(value) >= 0:
        return value
    with torch.inference_mode(False), torch.no_grad():
        snapshot = value.detach().clone()
    if _tensor_version(snapshot) < 0:  # pragma: no cover - defensive boundary
        raise BatchedRefineContractError("failed to create versioned runtime storage")
    return snapshot


def _source_config_fingerprint(
    recall: object,
    field: object,
    *,
    max_k: int,
    formula_beam_width: int,
    candidate_policy: str,
    partition_quota: tuple[int, ...],
    partition_coherence: str,
) -> str:
    activation = recall.state.recall_activation
    activation_metadata = getattr(activation, "survival_metadata", None)
    activation_config = {
        "ref": getattr(activation, "_component_reference", "torch/identity@1"),
        "stochastic": bool(getattr(activation, "stochastic", False)),
        "learnable": bool(getattr(activation, "learnable", False)),
        "survival": (
            dict(activation_metadata)
            if isinstance(activation_metadata, Mapping)
            else None
        ),
    }
    candidate_ref = (
        RecallBranchBatch._component_reference
        if field.value_composition == "single"
        else "arti/recall-formula-branch-batch@3"
    )
    return _config_fingerprint(
        {
            "ref": candidate_ref,
            "source_ref": recall._component_reference,
            "layout_fingerprint": field._route_layout_fingerprint(),
            "query_contract": field.query_contract,
            "routing": field.routing,
            "routing_normalizer": field.routing_normalizer,
            "formula_ref": _recall_formula_reference(field),
            "formula_behavior": _recall_formula_behavior_config(field),
            "formula_provider": _recall_formula_provider_metadata(field),
            "value_composition": field.value_composition,
            "candidate_policy": candidate_policy,
            "partition_quota": partition_quota,
            "partition_coherence": partition_coherence,
            "composition_factor": field.composition_factor,
            "factor_names": field.factor_names,
            "factor_route_names": field.factor_route_names,
            "factor_route_indices": field.factor_route_indices,
            "group_size": field.group_size,
            "available_k": field.group_topk,
            "max_k": max_k,
            "formula_beam_width": formula_beam_width,
            "route_exploration": field.route_exploration,
            "training": field.training,
            "bank_gradient_enabled": field._bank_gradient_enabled,
            "training_group_partitions": field._training_group_partitions,
            "trace_activation": activation_config,
            "dropout": float(recall.state.dropout.p),
            "state_training": bool(recall.state.training),
            "expert_names": field.expert_names,
            "expert_route_ranges": field._expert_route_ranges,
            "expert_member_fingerprints": field.expert_member_fingerprints,
            "expert_weights": field.expert_weights,
            "expert_influences": field.expert_influences,
        }
    )


def _recall_formula_reference(field: object) -> str:
    composition = field.value_composition
    if composition == "single":
        return "arti/delta@1"
    if composition == "product":
        return "arti/affine@1"
    if composition == "state":
        return "arti/state@1"
    if composition != "custom":
        raise BatchedRefineContractError("unsupported Recall value composition")
    contract = field.formula_contract
    identity = None if contract is None else contract.identity
    if identity is None:
        raise BatchedRefineContractError(
            "custom Recall Formula candidates require a versioned Formula identity"
        )
    if identity.namespace == "arti":
        raise BatchedRefineContractError(
            "custom Recall Formula candidates cannot claim a builtin arti identity"
        )
    from .recall_registry import RecallFormulaRegistryError, resolve_formula

    try:
        registration = resolve_formula(identity.reference)
        provider = registration.instantiate()
    except RecallFormulaRegistryError as error:
        raise BatchedRefineContractError(
            "custom Recall Formula candidates require an explicitly registered provider"
        ) from error
    if type(provider) is not type(field.formula):
        raise BatchedRefineContractError(
            "custom Recall Formula provider does not match the registered identity"
        )
    provider_contract = getattr(provider, "recall_formula_contract", None)
    if provider_contract != contract:
        raise BatchedRefineContractError(
            "custom Recall Formula contract does not match the registered provider"
        )
    return identity.reference


def _partition_contract(
    field: object,
) -> tuple[
    tuple[str, ...],
    tuple[tuple[int, int], ...],
    tuple[str, ...],
    str,
]:
    route_width = field._route_width()
    names = field.expert_names or ("default",)
    ranges = field._expert_route_ranges or ((0, route_width),)
    member_fingerprints = field.expert_member_fingerprints
    fingerprint = _config_fingerprint(
        {
            "ref": "arti/recall-partition-layout@1",
            "routing_normalizer": field.routing_normalizer,
            "names": names,
            "ranges": ranges,
            "member_fingerprints": member_fingerprints,
            "route_width": route_width,
            "composition_factor": field.composition_factor,
        }
    )
    return names, ranges, member_fingerprints, fingerprint


def _candidate_partition_index(
    group_index: Tensor,
    *,
    route_width: int,
    ranges: tuple[tuple[int, int], ...],
) -> Tensor:
    local_group = torch.remainder(group_index, route_width)
    partition = torch.full_like(group_index, -1)
    for index, (start, stop) in enumerate(ranges):
        partition = torch.where(
            (local_group >= start) & (local_group < stop),
            torch.full_like(partition, index),
            partition,
        )
    _assert_tensor(partition >= 0, "candidate group has no Bank partition")
    return partition


def _effective_candidate_capacity(
    field: object,
    *,
    available_k: int,
    resolved_k: int,
    factor_count: int,
    candidate_policy: str,
) -> int:
    """Return executable candidate width after zero-weight partitions are removed."""

    if field.routing_normalizer != "per_bank":
        return resolved_k
    ranges = field._expert_route_ranges or ((0, field._route_width()),)
    weights = field.expert_weights or (1.0,)
    enabled_groups = sum(
        stop - start
        for weight, (start, stop) in zip(weights, ranges, strict=True)
        if weight > 0.0
    )
    per_factor = min(available_k, enabled_groups)
    if field.value_composition == "single":
        return min(resolved_k, per_factor)
    if candidate_policy == "same-bank-joint-factor-beam@1":
        return min(
            resolved_k,
            sum(
                min(available_k, stop - start) ** factor_count
                for weight, (start, stop) in zip(weights, ranges, strict=True)
                if weight > 0.0
            ),
        )
    return min(resolved_k, per_factor**factor_count)


def _resolve_single_candidate_allocation(
    field: object,
    *,
    resolved_k: int,
    candidate_allocation: str | None,
    bank_quotas: tuple[int, ...] | None,
) -> tuple[str, tuple[int, ...]]:
    """Resolve one explicit fixed-K allocation contract for single-value Recall."""

    if field.routing_normalizer == "global":
        if candidate_allocation not in {None, "global_weighted_topk"}:
            raise BatchedRefineContractError(
                "global Recall only supports candidate_allocation='global_weighted_topk'"
            )
        if bank_quotas is not None:
            raise BatchedRefineContractError(
                "bank_quotas require per_bank Recall normalization"
            )
        return "global-weighted-topk@1", ()

    allocation = "per_bank_reserved" if candidate_allocation is None else candidate_allocation
    if allocation not in {
        "global_weighted_topk",
        "per_bank_reserved",
        "explicit_bank_quota",
    }:
        raise BatchedRefineContractError(
            "candidate_allocation must be 'global_weighted_topk', "
            "'per_bank_reserved', or 'explicit_bank_quota'"
        )
    if allocation == "global_weighted_topk":
        if bank_quotas is not None:
            raise BatchedRefineContractError(
                "bank_quotas cannot be combined with global_weighted_topk"
            )
        return "global-weighted-topk@1", ()

    ranges = field._expert_route_ranges or ((0, field._route_width()),)
    weights = field.expert_weights or (1.0,)
    capacities = tuple(stop - start for start, stop in ranges)
    enabled = tuple(index for index, weight in enumerate(weights) if weight > 0.0)
    if not enabled:
        raise BatchedRefineContractError("candidate allocation has no enabled Bank")
    if allocation == "explicit_bank_quota":
        if bank_quotas is None:
            raise BatchedRefineContractError(
                "explicit_bank_quota requires one quota per Bank"
            )
        quotas = tuple(bank_quotas)
        if len(quotas) != len(ranges) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in quotas
        ):
            raise BatchedRefineContractError(
                "bank_quotas must contain one non-negative integer per Bank"
            )
        if sum(quotas) != resolved_k:
            raise BatchedRefineContractError("bank_quotas must sum to max_k")
        if any(quota > capacity for quota, capacity in zip(quotas, capacities, strict=True)):
            raise BatchedRefineContractError("a Bank quota exceeds its route capacity")
        if any(quota and weights[index] == 0.0 for index, quota in enumerate(quotas)):
            raise BatchedRefineContractError("a zero-weight Bank must have zero quota")
        return "per-bank-quota-topk@1", quotas

    if bank_quotas is not None:
        raise BatchedRefineContractError(
            "bank_quotas require candidate_allocation='explicit_bank_quota'"
        )
    if resolved_k < len(enabled):
        raise BatchedRefineContractError(
            "per_bank_reserved requires max_k to cover every enabled Bank"
        )
    quotas = [0] * len(ranges)
    for index in enabled:
        quotas[index] = 1
    remaining = resolved_k - len(enabled)
    while remaining and any(quotas[index] < capacities[index] for index in enabled):
        progressed = False
        for index in enabled:
            if quotas[index] < capacities[index]:
                quotas[index] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:  # pragma: no cover - guarded by capacity validation
            raise BatchedRefineContractError("unable to allocate the requested Bank quota")
    disabled = tuple(index for index in range(len(ranges)) if index not in enabled)
    while remaining:
        progressed = False
        for index in disabled:
            if quotas[index] < capacities[index]:
                quotas[index] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise BatchedRefineContractError(
                "max_k exceeds the complete per-Bank candidate capacity"
            )
    return "per-bank-reserved-topk@1", tuple(quotas)


def _resolve_composed_candidate_allocation(
    field: object,
    *,
    resolved_k: int,
    available_k: int,
    factor_count: int,
    candidate_allocation: str | None,
    bank_quotas: tuple[int, ...] | None,
    partition_coherence: str | None,
) -> tuple[str, str, tuple[int, ...]]:
    coherence = (
        "same_bank" if field.routing_normalizer == "per_bank" else "cross_bank"
    ) if partition_coherence is None else partition_coherence
    if coherence not in {"cross_bank", "same_bank"}:
        raise BatchedRefineContractError(
            "partition_coherence must be 'cross_bank' or 'same_bank'"
        )
    if coherence == "cross_bank":
        if candidate_allocation not in {None, "global_weighted_topk"}:
            raise BatchedRefineContractError(
                "cross_bank Formula candidates only support global_weighted_topk"
            )
        if bank_quotas is not None:
            raise BatchedRefineContractError(
                "cross_bank Formula candidates do not accept Bank quotas"
            )
        return "cross-bank-joint-factor-beam@1", coherence, ()
    if field.routing_normalizer != "per_bank":
        raise BatchedRefineContractError(
            "same_bank Formula candidates require per_bank Recall normalization"
        )

    allocation = "per_bank_reserved" if candidate_allocation is None else candidate_allocation
    if allocation not in {"per_bank_reserved", "explicit_bank_quota"}:
        raise BatchedRefineContractError(
            "same_bank Formula candidates require per_bank_reserved or "
            "explicit_bank_quota allocation"
        )
    ranges = field._expert_route_ranges or ((0, field._route_width()),)
    weights = field.expert_weights or (1.0,)
    capacities = tuple(
        min(available_k, stop - start) ** factor_count for start, stop in ranges
    )
    enabled = tuple(index for index, weight in enumerate(weights) if weight > 0.0)
    if resolved_k < len(enabled):
        raise BatchedRefineContractError(
            "same_bank per_bank_reserved requires max_k to cover every enabled Bank"
        )
    if allocation == "explicit_bank_quota":
        if bank_quotas is None:
            raise BatchedRefineContractError(
                "explicit_bank_quota requires one quota per Bank"
            )
        quotas = tuple(bank_quotas)
        if len(quotas) != len(ranges) or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in quotas
        ):
            raise BatchedRefineContractError(
                "bank_quotas must contain one non-negative integer per Bank"
            )
        if sum(quotas) != resolved_k:
            raise BatchedRefineContractError("bank_quotas must sum to max_k")
        if any(quota > capacity for quota, capacity in zip(quotas, capacities, strict=True)):
            raise BatchedRefineContractError(
                "a Bank quota exceeds its same-Bank Formula capacity"
            )
        if any(quota and weights[index] == 0.0 for index, quota in enumerate(quotas)):
            raise BatchedRefineContractError("a zero-weight Bank must have zero quota")
        return "same-bank-joint-factor-beam@1", coherence, quotas

    if bank_quotas is not None:
        raise BatchedRefineContractError(
            "bank_quotas require candidate_allocation='explicit_bank_quota'"
        )
    quotas = [0] * len(ranges)
    for index in enabled:
        quotas[index] = 1
    remaining = resolved_k - len(enabled)
    ordered_partitions = (*enabled, *(i for i in range(len(ranges)) if i not in enabled))
    while remaining:
        progressed = False
        for index in ordered_partitions:
            if quotas[index] < capacities[index]:
                quotas[index] += 1
                remaining -= 1
                progressed = True
                if remaining == 0:
                    break
        if not progressed:
            raise BatchedRefineContractError(
                "max_k exceeds the complete same-Bank Formula capacity"
            )
    return "same-bank-joint-factor-beam@1", coherence, tuple(quotas)


def _select_partition_quota_groups(
    route_mass: Tensor,
    *,
    ranges: tuple[tuple[int, int], ...],
    quotas: tuple[int, ...],
    active_partitions: tuple[bool, ...],
) -> Tensor:
    """Select per-Bank Top-K and interleave ranks to preserve prefix breadth."""

    selected_by_partition: list[Tensor | None] = []
    for (start, stop), quota in zip(ranges, quotas, strict=True):
        if quota == 0:
            selected_by_partition.append(None)
            continue
        local = torch.topk(route_mass[..., start:stop], quota, dim=-1).indices + start
        selected_by_partition.append(local)
    ordered: list[Tensor] = []
    for active in (True, False):
        for rank in range(max(quotas, default=0)):
            for quota, selected, is_active in zip(
                quotas, selected_by_partition, active_partitions, strict=True
            ):
                if is_active == active and selected is not None and rank < quota:
                    ordered.append(selected[..., rank])
    if not ordered:
        raise BatchedRefineContractError("candidate allocation selected no routes")
    return torch.stack(ordered, dim=-1)


def _recall_formula_behavior_config(field: object) -> dict[str, object] | None:
    """Return explicit non-tensor behavior that binds a custom Formula query."""

    if field.value_composition != "custom":
        return None
    provider = field.formula
    builder = getattr(provider, "recall_formula_config", None)
    if not callable(builder):
        raise BatchedRefineContractError(
            "custom Recall Formula candidates require recall_formula_config()"
        )
    config = builder()
    if not isinstance(config, dict) or any(not isinstance(key, str) for key in config):
        raise BatchedRefineContractError(
            "custom Recall Formula recall_formula_config() must return a string-keyed dict"
        )
    # Reuse the canonical JSON normalizer as a validation boundary.
    _config_fingerprint(config)
    return config


def _recall_formula_provider_metadata(field: object) -> dict[str, object] | None:
    if field.value_composition != "custom":
        return None
    from .recall_registry import describe_formula

    metadata = describe_formula(_recall_formula_reference(field)).to_dict()
    return {
        "reference": metadata["reference"],
        "origin": metadata["origin"],
        "provider_kind": metadata["provider_kind"],
        "portable": metadata["portable"],
    }


def _joint_factor_topk(route_mass: Tensor, output_k: int) -> tuple[Tensor, Tensor]:
    """Return exact Top-K tuples under an additive independent-factor score.

    ``route_mass`` is ``[B,N,F,C]`` where ``C`` is the source Top-K width.
    The width-K beam is exact because factor scores are independent and every
    prefix keeps its K best partial tuples. Complexity is ``O(F*K*C)`` rather
    than materializing the ``C**F`` Cartesian product.
    """

    if route_mass.ndim != 4 or output_k < 1:
        raise BatchedRefineContractError("joint factor route mass has an invalid shape")
    candidate_count = route_mass.shape[-1]
    factor_count = route_mass.shape[-2]
    if output_k > candidate_count**factor_count:
        raise BatchedRefineContractError(
            "joint Formula beam exceeds the factor Cartesian space"
        )
    tiny = torch.finfo(route_mass.dtype).tiny
    factor_score = torch.log(route_mass.clamp_min(tiny))
    candidate_count = factor_score.shape[-1]
    first = torch.argsort(
        factor_score[:, :, 0],
        dim=-1,
        descending=True,
        stable=True,
    )[..., :output_k]
    paths = first.unsqueeze(-1)
    scores = torch.gather(factor_score[:, :, 0], -1, first)
    for factor in range(1, route_mass.shape[2]):
        combined = scores.unsqueeze(-1) + factor_score[:, :, factor].unsqueeze(-2)
        flat = combined.flatten(-2)
        selected = torch.argsort(
            flat,
            dim=-1,
            descending=True,
            stable=True,
        )[..., :output_k]
        parent = torch.div(selected, candidate_count, rounding_mode="floor")
        rank = torch.remainder(selected, candidate_count)
        paths = torch.cat(
            (
                torch.gather(
                    paths,
                    2,
                    parent.unsqueeze(-1).expand(-1, -1, -1, paths.shape[-1]),
                ),
                rank.unsqueeze(-1),
            ),
            dim=-1,
        )
        scores = torch.gather(flat, -1, selected)
    return paths.contiguous(), scores.contiguous()


def _gather_factor_candidates(value: Tensor, factor_rank: Tensor) -> Tensor:
    """Gather ``[B,N,F,C,...]`` values into ``[B,N,K,F,...]`` tuples."""

    if value.ndim < 4 or factor_rank.ndim != 4:
        raise BatchedRefineContractError("factor candidate gather rank is invalid")
    if value.shape[:3] != (
        factor_rank.shape[0],
        factor_rank.shape[1],
        factor_rank.shape[3],
    ):
        raise BatchedRefineContractError("factor candidate gather shape is invalid")
    factor_first_rank = factor_rank.permute(0, 1, 3, 2)
    index = factor_first_rank
    for _ in value.shape[4:]:
        index = index.unsqueeze(-1)
    index = index.expand(*factor_first_rank.shape, *value.shape[4:])
    gathered = torch.gather(value, 3, index)
    permutation = (0, 1, 3, 2, *range(4, gathered.ndim))
    return gathered.permute(permutation).contiguous()


def _same_bank_formula_candidates(
    field: object,
    sequence: Tensor,
    token_mask: Tensor,
    factor_route: Tensor,
    *,
    available_k: int,
    partition_quota: tuple[int, ...],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Build complete Formula tuples inside each Bank before K allocation."""

    factor_count = factor_route.shape[2]
    ranges = field._expert_route_ranges or ((0, field._route_width()),)
    active_partitions = tuple(
        weight > 0.0 for weight in (field.expert_weights or (1.0,))
    )
    pieces: list[tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor] | None] = []
    for (start, stop), quota in zip(ranges, partition_quota, strict=True):
        if quota == 0:
            pieces.append(None)
            continue
        source_width = min(available_k, stop - start)
        selected_groups = (
            torch.topk(
                factor_route[..., start:stop],
                source_width,
                dim=-1,
            ).indices
            + start
        )
        partition_read = field(
            sequence,
            token_mask,
            selected_groups=selected_groups,
        )
        prefix = (
            *sequence.shape[:2],
            factor_count,
            source_width,
            field.group_size,
        )
        factor_slot_index = partition_read.indices.reshape(prefix)
        factor_weights = partition_read.weights.reshape(prefix)
        factor_group_index = torch.div(
            factor_slot_index[..., 0],
            field.group_size,
            rounding_mode="floor",
        )
        local_route_mass = factor_route.gather(-1, selected_groups)
        factor_rank, joint_score = _joint_factor_topk(local_route_mass, quota)
        pieces.append(
            (
                _gather_factor_candidates(factor_slot_index, factor_rank),
                _gather_factor_candidates(factor_weights, factor_rank),
                _gather_factor_candidates(factor_group_index, factor_rank),
                _gather_factor_candidates(local_route_mass, factor_rank),
                factor_rank,
                joint_score,
            )
        )

    order: list[tuple[int, int]] = []
    for active in (True, False):
        for rank in range(max(partition_quota, default=0)):
            for partition, (quota, is_active) in enumerate(
                zip(partition_quota, active_partitions, strict=True)
            ):
                if is_active == active and rank < quota:
                    order.append((partition, rank))
    if len(order) != sum(partition_quota):
        raise BatchedRefineContractError("same-Bank Formula allocation is incomplete")

    outputs: list[Tensor] = []
    for value_index in range(6):
        selected: list[Tensor] = []
        for partition, rank in order:
            piece = pieces[partition]
            if piece is None:  # pragma: no cover - order excludes zero quotas
                raise BatchedRefineContractError("missing same-Bank Formula candidate")
            selected.append(piece[value_index].narrow(2, rank, 1))
        outputs.append(torch.cat(selected, dim=2).contiguous())
    return tuple(outputs)  # type: ignore[return-value]


@dataclass(frozen=True)
class RecallBranchBatch:
    """One Recall query expanded into K addressable branch seeds.

    The branch axis is always explicit. Execution adapters may flatten
    ``[B, K, ...]`` to ``[B*K, ...]`` for existing ARTI kernels, but receipts
    and state ownership must restore the K axis.
    """

    source_ref: str
    source_instance_token: int
    source_config_fingerprint: str
    layout_fingerprint: str
    query_version: int
    key_bank_version: int
    group_bank_version: int
    value_bank_version: int
    input_instance_token: int
    input_version: int
    input_shape: tuple[int, ...]
    input_dtype: str
    input_device: str
    input_snapshot: Tensor | None
    input_snapshot_token: int
    input_snapshot_version: int
    candidate_tensor_tokens: tuple[int, ...]
    candidate_tensor_versions: tuple[int, ...]
    max_k: int
    source_topk: int
    formula_beam_width: int
    group_count: int
    group_size: int
    partition_names: tuple[str, ...]
    partition_ranges: tuple[tuple[int, int], ...]
    partition_member_fingerprints: tuple[str, ...]
    partition_layout_fingerprint: str
    candidate_group_index: Tensor
    candidate_partition_index: Tensor
    candidate_slot_index: Tensor
    candidate_slot_weight: Tensor
    candidate_context: Tensor
    route_mass: Tensor
    selection_weight: Tensor
    candidate_log_score: Tensor
    candidate_mask: Tensor
    branch_mask: Tensor
    token_mask: Tensor
    branch_origin_index: Tensor
    active_k: Tensor
    requested_active_k: Tensor
    query_rng_fingerprint: str | None = None
    routing_normalizer: str = "global"
    value_composition: str = "single"
    factor_count: int = 1
    factor_candidate_rank: Tensor | None = None
    factor_route_index: Tensor | None = None
    candidate_policy: str = "global-weighted-topk@1"
    formula_ref: str = "arti/recall-single@1"
    formula_config_fingerprint: str = "0" * 64
    topology_lineage: tuple[str, ...] = ()
    source_execution_tensor_lineage: tuple[tuple[str, int, int], ...] = ()
    partition_quota: tuple[int, ...] = ()
    partition_coherence: str = "not_applicable"
    schema_version: int = 3
    _component_reference: ClassVar[str] = "arti/recall-branch-batch@3"

    def __post_init__(self) -> None:
        if self.schema_version not in {3, 6}:
            raise BatchedRefineContractError("unsupported Recall branch-batch schema")
        composed = self.value_composition != "single"
        if self.schema_version != (6 if composed else 3):
            raise BatchedRefineContractError(
                "Recall branch schema does not match value composition"
            )
        expected_reference = (
            "arti/recall-formula-branch-batch@3"
            if composed
            else "arti/recall-branch-batch@3"
        )
        if self._component_reference != expected_reference:
            raise BatchedRefineContractError(
                "Recall branch class/reference does not match value composition"
            )
        if self.routing_normalizer not in {"global", "per_bank"}:
            raise BatchedRefineContractError("invalid Recall routing normalizer")
        expected_policies = (
            {
                "cross-bank-joint-factor-beam@1",
                "same-bank-joint-factor-beam@1",
            }
            if composed
            else {"global-weighted-topk@1"}
            if self.routing_normalizer == "global"
            else {
                "global-weighted-topk@1",
                "per-bank-reserved-topk@1",
                "per-bank-quota-topk@1",
            }
        )
        if self.candidate_policy not in expected_policies:
            raise BatchedRefineContractError(
                "candidate policy does not match the Recall value composition"
            )
        if composed:
            if self.partition_coherence not in {"cross_bank", "same_bank"}:
                raise BatchedRefineContractError(
                    "composed candidates require explicit partition coherence"
                )
            if (
                self.candidate_policy == "same-bank-joint-factor-beam@1"
            ) != (self.partition_coherence == "same_bank"):
                raise BatchedRefineContractError(
                    "Formula candidate policy and partition coherence disagree"
                )
        elif self.partition_coherence != "not_applicable":
            raise BatchedRefineContractError(
                "single-value candidates do not have partition coherence"
            )
        if not composed and self.factor_count != 1:
            raise BatchedRefineContractError("single Recall must expose one factor")
        if composed and self.factor_count <= 0:
            raise BatchedRefineContractError(
                "Formula-aware Recall must expose at least one factor"
            )
        if not isinstance(self.formula_ref, str) or not self.formula_ref:
            raise BatchedRefineContractError("formula_ref must be non-empty")
        if not isinstance(self.topology_lineage, tuple) or any(
            not isinstance(item, str) or not item for item in self.topology_lineage
        ):
            raise BatchedRefineContractError("topology_lineage must contain strings")
        if not isinstance(self.source_execution_tensor_lineage, tuple):
            raise BatchedRefineContractError(
                "source_execution_tensor_lineage must be a tuple"
            )
        for entry in self.source_execution_tensor_lineage:
            if (
                not isinstance(entry, tuple)
                or len(entry) != 3
                or not isinstance(entry[0], str)
                or not entry[0]
                or isinstance(entry[1], bool)
                or not isinstance(entry[1], int)
                or entry[1] <= 0
                or isinstance(entry[2], bool)
                or not isinstance(entry[2], int)
                or entry[2] < 0
            ):
                raise BatchedRefineContractError(
                    "source execution tensor lineage is invalid"
                )
        if not isinstance(self.source_ref, str) or not self.source_ref:
            raise BatchedRefineContractError("source_ref must be non-empty")
        for value, name in (
            (self.source_instance_token, "source_instance_token"),
            (self.query_version, "query_version"),
            (self.key_bank_version, "key_bank_version"),
            (self.group_bank_version, "group_bank_version"),
            (self.value_bank_version, "value_bank_version"),
            (self.input_instance_token, "input_instance_token"),
            (self.input_version, "input_version"),
        ):
            minimum = -1 if name == "input_version" else 0
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise BatchedRefineContractError(
                    f"{name} must be an integer greater than or equal to {minimum}"
                )
        if self.input_shape not in {
            (self.batch_size, self.candidate_context.shape[-1]),
            (self.batch_size, self.token_count, self.candidate_context.shape[-1]),
        }:
            raise BatchedRefineContractError("input_shape does not match branch candidates")
        if not self.input_dtype or not self.input_device:
            raise BatchedRefineContractError("input dtype and device must be recorded")
        if self.input_version < 0:
            if not isinstance(self.input_snapshot, Tensor):
                raise BatchedRefineContractError(
                    "an inference input requires a versioned immutable snapshot"
                )
            if (
                tuple(self.input_snapshot.shape) != self.input_shape
                or str(self.input_snapshot.dtype) != self.input_dtype
                or str(self.input_snapshot.device) != self.input_device
                or id(self.input_snapshot) != self.input_snapshot_token
                or _tensor_version(self.input_snapshot) != self.input_snapshot_version
                or self.input_snapshot_version < 0
            ):
                raise BatchedRefineContractError("input snapshot contract is invalid")
        elif (
            self.input_snapshot is not None
            or self.input_snapshot_token != 0
            or self.input_snapshot_version != -1
        ):
            raise BatchedRefineContractError(
                "a versioned input must not carry a redundant snapshot"
            )
        expected_lineage = 16 if composed else 14
        if (
            len(self.candidate_tensor_tokens) != expected_lineage
            or len(self.candidate_tensor_versions) != expected_lineage
        ):
            raise BatchedRefineContractError(
                "candidate tensor lineage does not cover the complete bridge"
            )
        for value, name in (
            (self.source_config_fingerprint, "source_config_fingerprint"),
            (self.layout_fingerprint, "layout_fingerprint"),
            (self.formula_config_fingerprint, "formula_config_fingerprint"),
            (self.partition_layout_fingerprint, "partition_layout_fingerprint"),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise BatchedRefineContractError(f"{name} must be lowercase SHA-256 hex")
        if self.query_rng_fingerprint is not None and (
            len(self.query_rng_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.query_rng_fingerprint
            )
        ):
            raise BatchedRefineContractError(
                "query_rng_fingerprint must be lowercase SHA-256 hex"
            )
        for value, name in (
            (self.max_k, "max_k"),
            (self.source_topk, "source_topk"),
            (self.formula_beam_width, "formula_beam_width"),
            (self.group_count, "group_count"),
            (self.group_size, "group_size"),
            (self.factor_count, "factor_count"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise BatchedRefineContractError(f"{name} must be a positive integer")
        if (
            not self.partition_names
            or len(self.partition_names) != len(self.partition_ranges)
            or len(set(self.partition_names)) != len(self.partition_names)
        ):
            raise BatchedRefineContractError(
                "partition names and ranges must form one non-empty layout"
            )
        if self.partition_member_fingerprints and (
            len(self.partition_member_fingerprints) != len(self.partition_names)
            or any(
                len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
                for value in self.partition_member_fingerprints
            )
        ):
            raise BatchedRefineContractError(
                "partition member fingerprints must be one SHA-256 per partition"
            )
        route_width = self.group_count // self.factor_count
        cursor = 0
        for name, (start, stop) in zip(
            self.partition_names,
            self.partition_ranges,
            strict=True,
        ):
            if not isinstance(name, str) or not name or start != cursor or stop <= start:
                raise BatchedRefineContractError("invalid candidate partition layout")
            cursor = stop
        if cursor != route_width:
            raise BatchedRefineContractError(
                "candidate partitions must cover the per-factor route axis"
            )
        if (
            self.candidate_policy
            in {"global-weighted-topk@1", "cross-bank-joint-factor-beam@1"}
        ):
            if self.partition_quota:
                raise BatchedRefineContractError(
                    "this candidate policy must not declare a partition quota"
                )
        else:
            if (
                len(self.partition_quota) != len(self.partition_names)
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, int)
                    or value < 0
                    for value in self.partition_quota
                )
                or sum(self.partition_quota) != self.max_k
            ):
                raise BatchedRefineContractError(
                    "partition quota must contain one non-negative allocation per Bank "
                    "and sum to max_k"
                )
            partition_capacities = tuple(
                (
                    min(self.source_topk, stop - start) ** self.factor_count
                    if composed
                    else stop - start
                )
                for start, stop in self.partition_ranges
            )
            if any(
                quota > capacity
                for quota, capacity in zip(
                    self.partition_quota, partition_capacities, strict=True
                )
            ):
                raise BatchedRefineContractError(
                    "partition quota exceeds a Bank route capacity"
                )
        if not composed and self.max_k > self.source_topk:
            raise BatchedRefineContractError("single Recall max_k must not exceed source_topk")
        if not composed and self.formula_beam_width != 1:
            raise BatchedRefineContractError(
                "single Recall does not have a Formula beam"
            )
        if composed and not (
            self.max_k
            <= self.formula_beam_width
            <= self.source_topk**self.factor_count
        ):
            raise BatchedRefineContractError(
                "Formula beam width must cover branch K within the Cartesian space"
            )
        if (
            not isinstance(self.candidate_group_index, Tensor)
            or self.candidate_group_index.dtype != torch.long
            or self.candidate_group_index.ndim != (4 if composed else 3)
        ):
            raise BatchedRefineContractError(
                "candidate_group_index must be torch.long [B,N,K] or [B,N,K,F]"
            )
        batch, tokens, branches = self.candidate_group_index.shape[:3]
        if branches != self.max_k:
            raise BatchedRefineContractError("candidate group K axis must equal max_k")
        if composed and self.candidate_group_index.shape[3] != self.factor_count:
            raise BatchedRefineContractError("candidate factor axis must equal factor_count")
        if (
            not isinstance(self.candidate_partition_index, Tensor)
            or self.candidate_partition_index.dtype != torch.long
            or self.candidate_partition_index.shape != self.candidate_group_index.shape
        ):
            raise BatchedRefineContractError(
                "candidate_partition_index must match candidate_group_index"
            )
        slot_shape = (
            (batch, tokens, branches, self.factor_count, self.group_size)
            if composed
            else (batch, tokens, branches, self.group_size)
        )
        if (
            not isinstance(self.candidate_slot_index, Tensor)
            or self.candidate_slot_index.dtype != torch.long
            or self.candidate_slot_index.shape != slot_shape
        ):
            raise BatchedRefineContractError(
                "candidate_slot_index has the wrong transition/factor shape"
            )
        if (
            not isinstance(self.candidate_slot_weight, Tensor)
            or not self.candidate_slot_weight.is_floating_point()
            or self.candidate_slot_weight.shape != slot_shape
        ):
            raise BatchedRefineContractError(
                "candidate_slot_weight has the wrong transition/factor shape"
            )
        if (
            not isinstance(self.candidate_context, Tensor)
            or not self.candidate_context.is_floating_point()
            or self.candidate_context.ndim != 4
            or self.candidate_context.shape[:3] != (batch, tokens, branches)
        ):
            raise BatchedRefineContractError(
                "candidate_context must be floating [B,N,K,D]"
            )
        for value, name in (
            (self.route_mass, "route_mass"),
            (self.selection_weight, "selection_weight"),
        ):
            expected_weight_shape = (
                (batch, tokens, branches, self.factor_count)
                if composed
                else (batch, tokens, branches)
            )
            if (
                not isinstance(value, Tensor)
                or not value.is_floating_point()
                or value.shape != expected_weight_shape
            ):
                raise BatchedRefineContractError(
                    f"{name} has the wrong transition/factor shape"
                )
        if (
            not isinstance(self.candidate_log_score, Tensor)
            or not self.candidate_log_score.is_floating_point()
            or self.candidate_log_score.shape != (batch, tokens, branches)
        ):
            raise BatchedRefineContractError(
                "candidate_log_score must be floating [B,N,K]"
            )
        if composed:
            if (
                not isinstance(self.factor_candidate_rank, Tensor)
                or self.factor_candidate_rank.dtype != torch.long
                or self.factor_candidate_rank.shape
                != (batch, tokens, branches, self.factor_count)
            ):
                raise BatchedRefineContractError(
                    "factor_candidate_rank must be torch.long [B,N,K,F]"
                )
            if (
                not isinstance(self.factor_route_index, Tensor)
                or self.factor_route_index.dtype != torch.long
                or self.factor_route_index.shape != (self.factor_count,)
            ):
                raise BatchedRefineContractError(
                    "factor_route_index must be torch.long [F]"
                )
        elif self.factor_candidate_rank is not None or self.factor_route_index is not None:
            raise BatchedRefineContractError(
                "single-value Recall must not carry factor candidate metadata"
            )
        for value, shape, name in (
            (self.candidate_mask, (batch, tokens, branches), "candidate_mask"),
            (self.branch_mask, (batch, branches), "branch_mask"),
            (self.token_mask, (batch, tokens), "token_mask"),
        ):
            if not isinstance(value, Tensor) or value.dtype != torch.bool or value.shape != shape:
                raise BatchedRefineContractError(f"{name} must be boolean {shape}")
        if (
            not isinstance(self.branch_origin_index, Tensor)
            or self.branch_origin_index.dtype != torch.long
            or self.branch_origin_index.shape != (batch, branches)
        ):
            raise BatchedRefineContractError(
                "branch_origin_index must be torch.long [B,K]"
            )
        if (
            not isinstance(self.active_k, Tensor)
            or self.active_k.dtype != torch.long
            or self.active_k.shape != (batch,)
        ):
            raise BatchedRefineContractError("active_k must be torch.long [B]")
        if (
            not isinstance(self.requested_active_k, Tensor)
            or self.requested_active_k.dtype != torch.long
            or self.requested_active_k.shape != (batch,)
        ):
            raise BatchedRefineContractError(
                "requested_active_k must be torch.long [B]"
            )
        device = self.route_mass.device
        tensors = (
            self.candidate_group_index,
            self.candidate_partition_index,
            self.candidate_slot_index,
            self.candidate_slot_weight,
            self.candidate_context,
            self.selection_weight,
            self.candidate_log_score,
            self.candidate_mask,
            self.branch_mask,
            self.token_mask,
            self.branch_origin_index,
            self.active_k,
            self.requested_active_k,
        )
        if self.factor_candidate_rank is not None:
            tensors = (*tensors, self.factor_candidate_rank)
        if self.factor_route_index is not None:
            tensors = (*tensors, self.factor_route_index)
        if any(value.device != device for value in tensors):
            raise BatchedRefineContractError("branch-batch tensors must use one device")
        expected_origins = torch.arange(branches, device=device).expand(batch, -1)
        _assert_tensor(
            torch.sort(self.branch_origin_index, dim=1).values == expected_origins,
            "branch_origin_index must be a permutation of [0,K) per sample",
        )
        _assert_tensor(
            (self.active_k >= 0) & (self.active_k <= self.max_k),
            "active_k must be within [0,max_k]",
        )
        _assert_tensor(
            (self.requested_active_k >= 0)
            & (self.requested_active_k <= self.max_k)
            & (self.active_k <= self.requested_active_k),
            "requested_active_k must be within [active_k,max_k]",
        )
        valid_group = (self.candidate_group_index >= 0) & (
            self.candidate_group_index < self.group_count
        )
        _assert_tensor(valid_group, "candidate group index is out of range")
        expected_partition = _candidate_partition_index(
            self.candidate_group_index,
            route_width=route_width,
            ranges=self.partition_ranges,
        )
        _assert_tensor(
            self.candidate_partition_index == expected_partition,
            "candidate partition identity does not match its group",
        )
        if self.partition_coherence == "same_bank":
            _assert_tensor(
                self.candidate_partition_index
                == self.candidate_partition_index[..., :1],
                "same-Bank Formula candidate crosses Bank partitions",
            )
        expected_group = torch.div(
            self.candidate_slot_index,
            self.group_size,
            rounding_mode="floor",
        )
        _assert_tensor(
            expected_group == self.candidate_group_index.unsqueeze(-1),
            "candidate slot index crosses a Recall group",
        )
        if self.group_size > 1:
            ordered_slots = torch.sort(self.candidate_slot_index, dim=-1).values
            _assert_tensor(
                ordered_slots[..., 1:] != ordered_slots[..., :-1],
                "candidate slots must be unique within each Recall group",
            )
        if self.max_k > 1 and not composed:
            ordered = torch.sort(self.candidate_group_index, dim=2).values
            _assert_tensor(
                ordered[:, :, 1:] != ordered[:, :, :-1],
                "candidate groups must be unique per token",
            )
        if composed:
            assert self.factor_candidate_rank is not None
            _assert_tensor(
                (self.factor_candidate_rank >= 0)
                & (self.factor_candidate_rank < self.source_topk),
                "factor candidate rank is out of range",
            )
            if self.max_k > 1:
                same_tuple = (
                    self.candidate_group_index.unsqueeze(3)
                    == self.candidate_group_index.unsqueeze(2)
                ).all(dim=-1)
                diagonal = torch.eye(
                    self.max_k,
                    device=same_tuple.device,
                    dtype=torch.bool,
                ).view(1, 1, self.max_k, self.max_k)
                _assert_tensor(
                    ~(same_tuple & ~diagonal),
                    "complete Formula candidate tuples must be unique per token",
                )
        finite_non_negative = (
            torch.isfinite(self.route_mass)
            & (self.route_mass >= 0)
            & torch.isfinite(self.selection_weight)
            & (self.selection_weight >= 0)
            & torch.isfinite(self.candidate_slot_weight).all(dim=-1)
            & (self.candidate_slot_weight >= 0).all(dim=-1)
        )
        if composed:
            finite_non_negative = finite_non_negative.all(dim=-1)
        finite_non_negative = finite_non_negative & torch.isfinite(
            self.candidate_context
        ).all(dim=-1)
        finite_non_negative = finite_non_negative & torch.isfinite(
            self.candidate_log_score
        )
        _assert_tensor(
            finite_non_negative,
            "candidate weights must be finite and non-negative; context must be finite",
        )
        expected_mask = self.token_mask.unsqueeze(-1) & self.branch_mask.unsqueeze(1)
        _assert_tensor(
            self.candidate_mask == expected_mask,
            "candidate_mask must combine token and branch validity",
        )
        _assert_tensor(
            self.branch_mask == self.candidate_mask.any(dim=1),
            "branch_mask must match candidate token coverage",
        )
        _assert_tensor(
            self.branch_mask.sum(dim=1) == self.active_k,
            "active_k must equal the number of active branches",
        )
        normalization_atol = max(
            1e-5,
            4.0 * torch.finfo(self.candidate_slot_weight.dtype).eps,
        )
        valid_slot_weight = self.candidate_slot_weight[self.candidate_mask]
        if valid_slot_weight.numel():
            slot_sums = valid_slot_weight.sum(dim=-1)
            _assert_tensor(
                torch.isclose(
                slot_sums,
                torch.ones_like(slot_sums),
                rtol=1e-4,
                atol=normalization_atol,
                ),
                "candidate slot weights must sum to one per candidate",
            )

    @property
    def batch_size(self) -> int:
        return int(self.candidate_group_index.shape[0])

    @property
    def token_count(self) -> int:
        return int(self.candidate_group_index.shape[1])

    def active_partition_mask(self) -> Tensor:
        """Return which Bank partitions actively participate per sample."""

        partition_mask = self.candidate_mask
        if self.candidate_partition_index.ndim == 4:
            partition_mask = partition_mask.unsqueeze(-1)
        presence = tuple(
            (
                (self.candidate_partition_index == partition_index)
                & partition_mask
            )
            .flatten(start_dim=1)
            .any(dim=1)
            for partition_index in range(len(self.partition_names))
        )
        return torch.stack(presence, dim=1)

    def active_partition_count(self) -> Tensor:
        """Return the number of Bank partitions active per sample."""

        return self.active_partition_mask().sum(dim=1, dtype=torch.long)

    def active_branch_count_by_partition(self) -> Tensor:
        """Count active candidate branches that touch each Bank partition."""

        branch_presence = tuple(
            (
                (
                    (self.candidate_partition_index == partition_index).any(dim=-1)
                    if self.candidate_partition_index.ndim == 4
                    else self.candidate_partition_index == partition_index
                )
                & self.candidate_mask
            )
            .any(dim=1)
            .sum(dim=1, dtype=torch.long)
            for partition_index in range(len(self.partition_names))
        )
        return torch.stack(branch_presence, dim=1)

    def expand_state(self, value: Tensor) -> Tensor:
        """Materialize independent ``[B,K,N,D]`` branch storage."""

        sequence = self.source_state(value)
        return sequence.unsqueeze(1).repeat(1, self.max_k, 1, 1).contiguous()

    def source_state(self, value: Tensor) -> Tensor:
        """Validate and return the canonical ``[B,N,D]`` source without K expansion."""

        if not isinstance(value, Tensor) or not value.is_floating_point():
            raise BatchedRefineContractError(
                "value must be a floating Tensor"
            )
        invalid_identity = (
            id(value) != self.input_instance_token
            or tuple(value.shape) != self.input_shape
            or str(value.dtype) != self.input_dtype
            or str(value.device) != self.input_device
        )
        if self.input_version >= 0:
            invalid_identity = invalid_identity or _tensor_version(value) != self.input_version
            source = value
        else:
            assert self.input_snapshot is not None
            if _tensor_version(self.input_snapshot) != self.input_snapshot_version:
                raise BatchedRefineContractError("input snapshot changed after query")
            unchanged = (value == self.input_snapshot) | (
                torch.isnan(value) & torch.isnan(self.input_snapshot)
            )
            _assert_tensor(
                unchanged,
                "inference input changed after its Recall query",
            )
            source = self.input_snapshot
        if invalid_identity:
            raise BatchedRefineContractError(
                "value is not the unchanged Tensor used for the candidate query"
            )
        sequence = source.unsqueeze(1) if source.ndim == 2 else source
        if (
            sequence.ndim != 3
            or sequence.shape[:2] != (self.batch_size, self.token_count)
            or sequence.device != self.route_mass.device
        ):
            raise BatchedRefineContractError(
                "value must be floating [B,D] or [B,N,D] on the branch-batch device"
            )
        return sequence

    def assert_unchanged(self) -> None:
        """Fail closed if any candidate bridge tensor was replaced or mutated."""

        tensors = (
            self.candidate_group_index,
            self.candidate_partition_index,
            self.candidate_slot_index,
            self.candidate_slot_weight,
            self.candidate_context,
            self.route_mass,
            self.selection_weight,
            self.candidate_log_score,
            self.candidate_mask,
            self.branch_mask,
            self.token_mask,
            self.branch_origin_index,
            self.active_k,
            self.requested_active_k,
        )
        if self.factor_candidate_rank is not None:
            tensors = (*tensors, self.factor_candidate_rank)
        if self.factor_route_index is not None:
            tensors = (*tensors, self.factor_route_index)
        if tuple(id(tensor) for tensor in tensors) != self.candidate_tensor_tokens:
            raise BatchedRefineContractError("candidate bridge tensor identity changed")
        current_versions = tuple(_tensor_version(tensor) for tensor in tensors)
        if current_versions != self.candidate_tensor_versions:
            raise BatchedRefineContractError("candidate bridge tensor changed after query")

    def canonical_manifest_fingerprint(self) -> str:
        """Fingerprint candidate content independently of physical K ordering."""

        self.assert_unchanged()
        order = torch.argsort(self.branch_origin_index, dim=1, stable=True)

        def canonical(tensor: Tensor, branch_dim: int) -> Tensor:
            shape = [tensor.shape[0]] + [1] * (tensor.ndim - 1)
            shape[branch_dim] = self.max_k
            index = order.reshape(shape)
            expand = list(tensor.shape)
            expand[branch_dim] = self.max_k
            return tensor.gather(branch_dim, index.expand(expand)).contiguous()

        branch_tensors = {
            "candidate_group_index": canonical(self.candidate_group_index, 2),
            "candidate_partition_index": canonical(
                self.candidate_partition_index, 2
            ),
            "candidate_slot_index": canonical(self.candidate_slot_index, 2),
            "candidate_slot_weight": canonical(self.candidate_slot_weight, 2),
            "candidate_context": canonical(self.candidate_context, 2),
            "route_mass": canonical(self.route_mass, 2),
            "selection_weight": canonical(self.selection_weight, 2),
            "candidate_log_score": canonical(self.candidate_log_score, 2),
            "candidate_mask": canonical(self.candidate_mask, 2),
            "branch_mask": canonical(self.branch_mask, 1),
        }
        if self.factor_candidate_rank is not None:
            branch_tensors["factor_candidate_rank"] = canonical(
                self.factor_candidate_rank, 2
            )
        payload = {
            "ref": "arti/recall-candidate-manifest@2",
            "source_ref": self.source_ref,
            "source_config_fingerprint": self.source_config_fingerprint,
            "layout_fingerprint": self.layout_fingerprint,
            "formula_ref": self.formula_ref,
            "formula_config_fingerprint": self.formula_config_fingerprint,
            "schema_version": self.schema_version,
            "query_rng_fingerprint": self.query_rng_fingerprint,
            "routing_normalizer": self.routing_normalizer,
            "partition_names": self.partition_names,
            "partition_ranges": self.partition_ranges,
            "partition_member_fingerprints": self.partition_member_fingerprints,
            "partition_layout_fingerprint": self.partition_layout_fingerprint,
            "max_k": self.max_k,
            "source_topk": self.source_topk,
            "formula_beam_width": self.formula_beam_width,
            "active_k": _tensor_content_fingerprint(self.active_k),
            "requested_active_k": _tensor_content_fingerprint(
                self.requested_active_k
            ),
            "active_partition_count": _tensor_content_fingerprint(
                self.active_partition_count()
            ),
            "active_partition_mask": _tensor_content_fingerprint(
                self.active_partition_mask()
            ),
            "active_branch_count_by_partition": _tensor_content_fingerprint(
                self.active_branch_count_by_partition()
            ),
            "token_mask": _tensor_content_fingerprint(self.token_mask),
            "factor_route_index": (
                None
                if self.factor_route_index is None
                else _tensor_content_fingerprint(self.factor_route_index)
            ),
            "candidate_tensors": {
                name: _tensor_content_fingerprint(tensor)
                for name, tensor in sorted(branch_tensors.items())
            },
        }
        return _config_fingerprint(payload)

    def flattened_candidate_context(self) -> Tensor:
        """Return raw candidate Bank values as ``[B*K,N,D]``."""

        batch, tokens, branches, dim = self.candidate_context.shape
        return (
            self.candidate_context.permute(0, 2, 1, 3)
            .reshape(batch * branches, tokens, dim)
            .contiguous()
        )

    def flattened_initial_groups(self) -> Tensor:
        """Return complete candidate seeds for existing Recall kernels."""

        if self.value_composition == "single":
            return (
                self.candidate_group_index.permute(0, 2, 1)
                .reshape(self.batch_size * self.max_k, self.token_count, 1)
                .contiguous()
            )
        factor_groups = self.group_count // self.factor_count
        offsets = torch.arange(
            self.factor_count,
            device=self.candidate_group_index.device,
        ).view(1, 1, 1, -1)
        local_groups = self.candidate_group_index - offsets * factor_groups
        return (
            local_groups.permute(0, 2, 1, 3)
            .reshape(
                self.batch_size * self.max_k,
                self.token_count,
                self.factor_count,
                1,
            )
            .contiguous()
        )

    def flattened_execution_groups(self) -> Tensor:
        """Return fixed-width first-step seeds as ``[B*K,N,source_topk]``.

        Repeating one candidate group preserves its value while keeping route
        trace shapes static when later steps return to automatic routing.
        """

        return self.flattened_initial_groups().expand(
            -1,
            -1,
            *(() if self.value_composition == "single" else (-1,)),
            self.source_topk,
        )

    def flattened_token_mask(self) -> Tensor:
        """Return branch-aware token validity as ``[B*K,N]``."""

        valid = self.candidate_mask.permute(0, 2, 1)
        return valid.reshape(self.batch_size * self.max_k, self.token_count).contiguous()

    def permute_branches(self, order: Tensor) -> "RecallBranchBatch":
        """Return the same candidate set under global or per-sample K ordering."""

        if (
            not isinstance(order, Tensor)
            or order.dtype != torch.long
            or order.ndim not in {1, 2}
            or order.shape
            not in {(self.max_k,), (self.batch_size, self.max_k)}
            or order.device != self.route_mass.device
        ):
            raise BatchedRefineContractError(
                "order must be torch.long [K] or [B,K] on the candidate device"
            )
        expected = torch.arange(self.max_k, device=order.device)
        _assert_tensor(
            torch.sort(order, dim=-1).values == expected,
            "each order row must be a permutation",
        )

        def select(tensor: Tensor, branch_dim: int) -> Tensor:
            if order.ndim == 1:
                return tensor.index_select(branch_dim, order).contiguous()
            shape = [self.batch_size] + [1] * (tensor.ndim - 1)
            shape[branch_dim] = self.max_k
            index = order.reshape(shape)
            return tensor.gather(branch_dim, index.expand(tensor.shape)).contiguous()

        branch_tensors = tuple(
            _versioned_runtime_tensor(tensor)
            for tensor in (
            select(self.candidate_group_index, 2),
            select(self.candidate_partition_index, 2),
            select(self.candidate_slot_index, 2),
            select(self.candidate_slot_weight, 2),
            select(self.candidate_context, 2),
            select(self.route_mass, 2),
            select(self.selection_weight, 2),
            select(self.candidate_log_score, 2),
            select(self.candidate_mask, 2),
            select(self.branch_mask, 1),
            self.token_mask,
            select(self.branch_origin_index, 1),
            self.active_k,
            self.requested_active_k,
            *((
                select(self.factor_candidate_rank, 2),
            ) if self.factor_candidate_rank is not None else ()),
            *((self.factor_route_index,) if self.factor_route_index is not None else ()),
            )
        )
        return type(self)(
            source_ref=self.source_ref,
            source_instance_token=self.source_instance_token,
            source_config_fingerprint=self.source_config_fingerprint,
            layout_fingerprint=self.layout_fingerprint,
            query_version=self.query_version,
            key_bank_version=self.key_bank_version,
            group_bank_version=self.group_bank_version,
            value_bank_version=self.value_bank_version,
            input_instance_token=self.input_instance_token,
            input_version=self.input_version,
            input_shape=self.input_shape,
            input_dtype=self.input_dtype,
            input_device=self.input_device,
            input_snapshot=self.input_snapshot,
            input_snapshot_token=self.input_snapshot_token,
            input_snapshot_version=self.input_snapshot_version,
            candidate_tensor_tokens=tuple(id(tensor) for tensor in branch_tensors),
            candidate_tensor_versions=tuple(_tensor_version(tensor) for tensor in branch_tensors),
            max_k=self.max_k,
            source_topk=self.source_topk,
            formula_beam_width=self.formula_beam_width,
            group_count=self.group_count,
            group_size=self.group_size,
            partition_names=self.partition_names,
            partition_ranges=self.partition_ranges,
            partition_member_fingerprints=self.partition_member_fingerprints,
            partition_layout_fingerprint=self.partition_layout_fingerprint,
            candidate_group_index=branch_tensors[0],
            candidate_partition_index=branch_tensors[1],
            candidate_slot_index=branch_tensors[2],
            candidate_slot_weight=branch_tensors[3],
            candidate_context=branch_tensors[4],
            route_mass=branch_tensors[5],
            selection_weight=branch_tensors[6],
            candidate_log_score=branch_tensors[7],
            candidate_mask=branch_tensors[8],
            branch_mask=branch_tensors[9],
            token_mask=branch_tensors[10],
            branch_origin_index=branch_tensors[11],
            active_k=branch_tensors[12],
            requested_active_k=branch_tensors[13],
            query_rng_fingerprint=self.query_rng_fingerprint,
            routing_normalizer=self.routing_normalizer,
            value_composition=self.value_composition,
            factor_count=self.factor_count,
            factor_candidate_rank=(
                None if self.factor_candidate_rank is None else branch_tensors[14]
            ),
            factor_route_index=(
                None
                if self.factor_route_index is None
                else branch_tensors[14 + (self.factor_candidate_rank is not None)]
            ),
            candidate_policy=self.candidate_policy,
            partition_quota=self.partition_quota,
            partition_coherence=self.partition_coherence,
            formula_ref=self.formula_ref,
            formula_config_fingerprint=self.formula_config_fingerprint,
            topology_lineage=self.topology_lineage,
            source_execution_tensor_lineage=self.source_execution_tensor_lineage,
            schema_version=self.schema_version,
        )


@dataclass(frozen=True)
class RecallFormulaBranchBatch(RecallBranchBatch):
    """K complete, factor-aware Recall Formula transition candidates."""

    schema_version: int = 6
    _component_reference: ClassVar[str] = "arti/recall-formula-branch-batch@3"


class BranchRefinePolicy:
    """Runtime-only per-branch policy for one Recall candidate batch.

    Values supplied to the constructor follow the candidates' current branch
    order. They are stored in canonical ``branch_origin_index`` order so the
    same policy remains equivariant when the candidate axis is permuted.
    """

    _component_reference: ClassVar[str] = "arti/branch-refine-policy@1"

    def __init__(
        self,
        candidates: RecallBranchBatch,
        base: object,
        *,
        min_steps: int | Tensor | None = None,
        max_steps: int | Tensor | None = None,
        absolute_tolerance: float | Tensor | None = None,
        relative_tolerance: float | Tensor | None = None,
        route_tolerance: float | Tensor | None = None,
        patience: int | Tensor | None = None,
        cycle_tolerance: float | Tensor | None = None,
    ) -> None:
        from .recall_refine import AdaptiveRefinePolicy

        if not isinstance(candidates, RecallBranchBatch):
            raise TypeError("candidates must be RecallBranchBatch")
        if not isinstance(base, AdaptiveRefinePolicy):
            raise TypeError("base must be AdaptiveRefinePolicy")
        candidates.assert_unchanged()
        batch, branches = candidates.branch_mask.shape
        device = candidates.branch_mask.device

        def normalize(
            value: int | float | Tensor,
            *,
            name: str,
            dtype: torch.dtype,
        ) -> Tensor:
            if isinstance(value, bool):
                raise TypeError(f"{name} must not be boolean")
            if isinstance(value, (int, float)):
                tensor = torch.full((batch, branches), value, device=device, dtype=dtype)
            elif (
                isinstance(value, Tensor)
                and value.shape == (batch, branches)
                and value.device == device
            ):
                tensor = value.to(dtype=dtype)
            else:
                raise TypeError(
                    f"{name} must be a scalar or Tensor [B,K] on the candidate device"
                )
            canonical = torch.empty_like(tensor)
            canonical.scatter_(1, candidates.branch_origin_index, tensor)
            return _versioned_runtime_tensor(canonical.contiguous())

        self.base = base
        self.min_steps = normalize(
            base.min_steps if min_steps is None else min_steps,
            name="min_steps",
            dtype=torch.int64,
        )
        self.max_steps = normalize(
            base.max_steps if max_steps is None else max_steps,
            name="max_steps",
            dtype=torch.int64,
        )
        self.absolute_tolerance = normalize(
            base.stop.absolute_tolerance
            if absolute_tolerance is None
            else absolute_tolerance,
            name="absolute_tolerance",
            dtype=torch.float32,
        )
        self.relative_tolerance = normalize(
            base.stop.relative_tolerance
            if relative_tolerance is None
            else relative_tolerance,
            name="relative_tolerance",
            dtype=torch.float32,
        )
        self.patience = normalize(
            base.stop.patience if patience is None else patience,
            name="patience",
            dtype=torch.int64,
        )
        resolved_route = (
            base.stop.route_tolerance if route_tolerance is None else route_tolerance
        )
        self.route_tolerance = (
            None
            if resolved_route is None
            else normalize(
                resolved_route,
                name="route_tolerance",
                dtype=torch.float32,
            )
        )
        resolved_cycle = (
            base.stop.cycle_tolerance if cycle_tolerance is None else cycle_tolerance
        )
        self.cycle_tolerance = (
            None
            if resolved_cycle is None
            else normalize(
                resolved_cycle,
                name="cycle_tolerance",
                dtype=torch.float32,
            )
        )
        _assert_tensor(self.min_steps >= 0, "branch min_steps must be non-negative")
        _assert_tensor(
            self.max_steps >= self.min_steps,
            "branch max_steps must be greater than or equal to min_steps",
        )
        _assert_tensor(
            self.max_steps <= base.max_steps,
            "branch max_steps must not exceed the base policy maximum",
        )
        _assert_tensor(self.patience > 0, "branch patience must be positive")
        _assert_tensor(
            (self.absolute_tolerance >= 0) & (self.relative_tolerance >= 0),
            "branch state tolerances must be non-negative",
        )
        _assert_tensor(
            (self.max_steps == 0)
            | (self.absolute_tolerance > 0)
            | (self.relative_tolerance > 0),
            "each executable branch needs a positive state tolerance",
        )
        if self.route_tolerance is not None:
            _assert_tensor(
                self.route_tolerance >= 0,
                "branch route_tolerance must be non-negative",
            )
        if self.cycle_tolerance is not None:
            _assert_tensor(
                self.cycle_tolerance >= 0,
                "branch cycle_tolerance must be non-negative",
            )

        self._source_binding = (
            candidates.source_ref,
            candidates.source_instance_token,
            candidates.source_config_fingerprint,
            candidates.layout_fingerprint,
            candidates.input_instance_token,
            candidates.input_version,
            candidates.input_snapshot_token,
            candidates.input_snapshot_version,
            candidates.max_k,
            candidates.schema_version,
            candidates.canonical_manifest_fingerprint(),
        )
        owned = [
            self.min_steps,
            self.max_steps,
            self.absolute_tolerance,
            self.relative_tolerance,
            self.patience,
            *((self.route_tolerance,) if self.route_tolerance is not None else ()),
            *((self.cycle_tolerance,) if self.cycle_tolerance is not None else ()),
        ]
        self._tensor_lineage = tuple((id(value), _require_tensor_version(value, name="branch policy")) for value in owned)
        self.config_fingerprint = _config_fingerprint(
            {
                "ref": self._component_reference,
                "base_ref": base._component_reference,
                "base": {
                    "max_steps": base.max_steps,
                    "scope": base.stop.scope,
                    "cycle_periods": list(base.stop.cycle_periods),
                    "trace_level": base.trace_level,
                    "check_finite": base.check_finite,
                    "nonfinite_action": base.nonfinite_action,
                    "executor": base.executor,
                },
                "source": list(self._source_binding),
                "min_steps": _tensor_content_fingerprint(self.min_steps),
                "max_steps": _tensor_content_fingerprint(self.max_steps),
                "absolute_tolerance": _tensor_content_fingerprint(self.absolute_tolerance),
                "relative_tolerance": _tensor_content_fingerprint(self.relative_tolerance),
                "patience": _tensor_content_fingerprint(self.patience),
                "route_tolerance": None if self.route_tolerance is None else _tensor_content_fingerprint(self.route_tolerance),
                "cycle_tolerance": None if self.cycle_tolerance is None else _tensor_content_fingerprint(self.cycle_tolerance),
            }
        )

    def _owned_tensors(self) -> tuple[Tensor, ...]:
        return (
            self.min_steps,
            self.max_steps,
            self.absolute_tolerance,
            self.relative_tolerance,
            self.patience,
            *((self.route_tolerance,) if self.route_tolerance is not None else ()),
            *((self.cycle_tolerance,) if self.cycle_tolerance is not None else ()),
        )

    def assert_matches(self, candidates: RecallBranchBatch) -> None:
        candidates.assert_unchanged()
        current = (
            candidates.source_ref,
            candidates.source_instance_token,
            candidates.source_config_fingerprint,
            candidates.layout_fingerprint,
            candidates.input_instance_token,
            candidates.input_version,
            candidates.input_snapshot_token,
            candidates.input_snapshot_version,
            candidates.max_k,
            candidates.schema_version,
            candidates.canonical_manifest_fingerprint(),
        )
        if current != self._source_binding:
            raise BatchedRefineContractError(
                "branch policy does not belong to these Recall candidates"
            )
        for value, (token, version) in zip(
            self._owned_tensors(), self._tensor_lineage, strict=True
        ):
            if id(value) != token or _tensor_version(value) != version:
                raise BatchedRefineContractError("branch policy changed after construction")

    def bind(self, candidates: RecallBranchBatch) -> object:
        """Return a flattened runtime schedule in current candidate order."""

        from .recall_refine import AdaptiveRefineSchedule

        self.assert_matches(candidates)
        origin = candidates.branch_origin_index

        def gather(value: Tensor) -> Tensor:
            return value.gather(1, origin).reshape(-1).contiguous()

        return AdaptiveRefineSchedule(
            min_steps=gather(self.min_steps),
            max_steps=gather(self.max_steps),
            absolute_tolerance=gather(self.absolute_tolerance),
            relative_tolerance=gather(self.relative_tolerance),
            patience=gather(self.patience),
            route_tolerance=(
                None if self.route_tolerance is None else gather(self.route_tolerance)
            ),
            cycle_tolerance=(
                None if self.cycle_tolerance is None else gather(self.cycle_tolerance)
            ),
        )


def query_recall_branches(
    recall: object,
    value: Tensor,
    *,
    mask: Tensor | None = None,
    max_k: int | None = None,
    formula_beam_width: int | None = None,
    active_k: int | Tensor | None = None,
    candidate_allocation: str | None = None,
    bank_quotas: tuple[int, ...] | None = None,
    partition_coherence: str | None = None,
    rng_plan: ExecutionRNGPlan | None = None,
) -> RecallBranchBatch:
    """Query one grouped Recall and preserve its Top-K as branch seeds.

    This function performs the existing Recall read only. It does not apply a
    Formula, refine a branch, score a future, or authorize a persistent write.
    """

    from .functional import ensure_mask
    from .nn import Recall

    if not isinstance(recall, Recall):
        raise TypeError("recall must be arti.nn.Recall")
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise TypeError("value must be a floating-point Tensor")
    if value.ndim not in {2, 3} or value.shape[-1] != recall.dim:
        raise BatchedRefineContractError("value must have shape [B,D] or [B,N,D]")
    field = recall.state.recall
    if field.routing != "grouped":
        raise BatchedRefineContractError("Batched Refine requires grouped Recall routing")
    if field.training and field._training_group_partitions is not None:
        raise BatchedRefineContractError(
            "Batched Refine does not support partition-restricted training routes"
        )
    available_k = field.group_topk
    composed = field.value_composition != "single"
    factor_count = field.composition_factor
    if not composed:
        if formula_beam_width is not None:
            raise BatchedRefineContractError(
                "formula_beam_width is only valid for composed Recall Formula"
            )
        resolved_formula_beam_width = 1
        resolved_k = available_k if max_k is None else max_k
        maximum_k = available_k
    else:
        resolved_formula_beam_width = (
            available_k if formula_beam_width is None else formula_beam_width
        )
        maximum_formula_beam = available_k**factor_count
        if (
            isinstance(resolved_formula_beam_width, bool)
            or not isinstance(resolved_formula_beam_width, int)
            or resolved_formula_beam_width <= 0
            or resolved_formula_beam_width > maximum_formula_beam
        ):
            raise BatchedRefineContractError(
                "formula_beam_width must be within the factor Cartesian space"
            )
        resolved_k = (
            resolved_formula_beam_width if max_k is None else max_k
        )
        maximum_k = resolved_formula_beam_width
    if (
        isinstance(resolved_k, bool)
        or not isinstance(resolved_k, int)
        or resolved_k <= 0
        or resolved_k > maximum_k
    ):
        raise BatchedRefineContractError(
            "max_k must be positive and not exceed its candidate width"
        )
    if composed:
        candidate_policy, resolved_partition_coherence, partition_quota = (
            _resolve_composed_candidate_allocation(
                field,
                resolved_k=resolved_k,
                available_k=available_k,
                factor_count=factor_count,
                candidate_allocation=candidate_allocation,
                bank_quotas=bank_quotas,
                partition_coherence=partition_coherence,
            )
        )
    else:
        if partition_coherence is not None:
            raise BatchedRefineContractError(
                "partition_coherence is only valid for composed Recall Formula"
            )
        candidate_policy, partition_quota = _resolve_single_candidate_allocation(
            field,
            resolved_k=resolved_k,
            candidate_allocation=candidate_allocation,
            bank_quotas=bank_quotas,
        )
        resolved_partition_coherence = "not_applicable"

    was_vector = value.ndim == 2
    sequence = value.unsqueeze(1) if was_vector else value
    if was_vector and mask is not None and mask.shape == value.shape[:1]:
        mask = mask.unsqueeze(1)
    token_mask = ensure_mask(
        mask,
        sequence.shape[0],
        sequence.shape[1],
        sequence.device,
    )
    if active_k is None:
        active_k_tensor = torch.full(
            (sequence.shape[0],),
            resolved_k,
            dtype=torch.long,
            device=sequence.device,
        )
    elif isinstance(active_k, bool):
        raise BatchedRefineContractError("active_k must be an integer or torch.long [B]")
    elif isinstance(active_k, int):
        active_k_tensor = torch.full(
            (sequence.shape[0],),
            active_k,
            dtype=torch.long,
            device=sequence.device,
        )
    elif (
        isinstance(active_k, Tensor)
        and active_k.dtype == torch.long
        and active_k.shape == (sequence.shape[0],)
        and active_k.device == sequence.device
    ):
        active_k_tensor = active_k
    else:
        raise BatchedRefineContractError(
            "active_k must be an integer or torch.long [B] on the input device"
        )
    _assert_tensor(
        (active_k_tensor >= 0) & (active_k_tensor <= resolved_k),
        "active_k must be within [0,max_k]",
    )
    requested_active_k_tensor = active_k_tensor
    effective_capacity = _effective_candidate_capacity(
        field,
        available_k=available_k,
        resolved_k=resolved_k,
        factor_count=factor_count,
        candidate_policy=candidate_policy,
    )
    active_k_tensor = torch.clamp(active_k_tensor, max=effective_capacity)
    # A sample with no valid tokens has no executable branches regardless of
    # its requested breadth.
    active_k_tensor = torch.where(
        token_mask.any(dim=1),
        active_k_tensor,
        torch.zeros_like(active_k_tensor),
    )
    active_branch_mask = torch.arange(
        resolved_k,
        device=sequence.device,
    ).unsqueeze(0) < active_k_tensor.unsqueeze(1)
    source_versions = tuple(
        _require_tensor_version(tensor, name=name)
        for tensor, name in (
            (field.query.weight, "Recall query"),
            (field.key_bank, "Recall key Bank"),
            (field.group_bank, "Recall group Bank"),
            (field.bank, "Recall value Bank"),
        )
    )
    activation = recall.state.recall_activation
    if bool(getattr(activation, "survival_runtime_only", False)):
        raise BatchedRefineContractError(
            "Batched Refine requires a versioned portable Half survival"
        )
    route_exploration = (
        field.training and field._bank_gradient_enabled and field.route_exploration > 0
    )
    if route_exploration and rng_plan is None:
        raise BatchedRefineContractError(
            "training route exploration requires ExecutionRNGPlan"
        )
    if route_exploration and partition_quota:
        raise BatchedRefineContractError(
            "per-Bank candidate allocation does not yet support route exploration"
        )
    if rng_plan is not None and not isinstance(rng_plan, ExecutionRNGPlan):
        raise TypeError("rng_plan must be ExecutionRNGPlan or None")
    formula_tensors = (
        ()
        if not isinstance(field.formula, nn.Module)
        else _operation_owned_tensors(field.formula)
    )
    activation_tensors = _operation_owned_tensors(activation)
    execution_tensors = tuple(
        (f"formula:{name}", tensor) for name, tensor in formula_tensors
    ) + tuple(
        (f"activation:{name}", tensor) for name, tensor in activation_tensors
    )
    source_execution_tensor_lineage = tuple(
        (
            name,
            id(tensor),
            _require_tensor_version(tensor, name=f"Recall execution {name}"),
        )
        for name, tensor in execution_tensors
    )
    candidate_random_source = (
        None if rng_plan is None else rng_plan.bind_candidate(sequence.shape[0])
    )
    read = field(
        sequence,
        token_mask,
        _random_source=candidate_random_source,
        _random_phase="candidate-route",
        _random_step=0,
    )
    if not composed and partition_quota:
        ranges = field._expert_route_ranges or ((0, field._route_width()),)
        selected_groups = _select_partition_quota_groups(
            read.route,
            ranges=ranges,
            quotas=partition_quota,
            active_partitions=tuple(
                weight > 0.0 for weight in (field.expert_weights or (1.0,))
            ),
        )
        read = field(
            sequence,
            token_mask,
            selected_groups=selected_groups,
        )
    value_bank = field.bank if field._bank_gradient_enabled else field.bank.detach()
    if not composed:
        selected_width = resolved_k if partition_quota else available_k
        if read.indices.shape[-2:] != (selected_width, field.group_size):
            raise BatchedRefineContractError(
                "Recall query did not expose grouped Top-K indices"
            )
        slot_index = read.indices[..., :resolved_k, :]
        weights = read.weights[..., :resolved_k, :]
        group_index = torch.div(
            slot_index[..., 0], field.group_size, rounding_mode="floor"
        )
        selection_weight = weights.sum(dim=-1)
        candidate_slot_weight = weights / selection_weight.unsqueeze(-1).clamp_min(
            torch.finfo(weights.dtype).eps
        )
        route_mass = read.route.gather(-1, group_index)
        candidate_log_score = torch.log(
            route_mass.clamp_min(torch.finfo(route_mass.dtype).tiny)
        )
        finite = (
            torch.isfinite(route_mass)
            & (route_mass >= 0)
            & torch.isfinite(selection_weight)
            & (selection_weight >= 0)
        )
        candidate_mask = token_mask.unsqueeze(-1) & active_branch_mask.unsqueeze(1)
        selected_values = torch.nn.functional.embedding(slot_index, value_bank)
        candidate_context = (
            selected_values.to(dtype=sequence.dtype)
            * candidate_slot_weight.to(dtype=sequence.dtype).unsqueeze(-1)
        ).sum(dim=-2)
        factor_candidate_rank = None
        factor_route_index = None
    else:
        factor_groups = field.group_bank.shape[0] // factor_count
        factor_route = read.route.reshape(
            sequence.shape[0], sequence.shape[1], factor_count, factor_groups
        )
        if resolved_partition_coherence == "same_bank":
            (
                slot_index,
                weights,
                group_index,
                route_mass,
                factor_candidate_rank,
                candidate_log_score,
            ) = _same_bank_formula_candidates(
                field,
                sequence,
                token_mask,
                factor_route,
                available_k=available_k,
                partition_quota=partition_quota,
            )
        else:
            expected_candidates = factor_count * available_k
            structured_shape = (
                factor_count,
                available_k,
                field.group_size,
            )
            if read.indices.shape[-3:] == structured_shape:
                read_indices = read.indices.flatten(-3, -2)
                read_weights = read.weights.flatten(-3, -2)
            else:
                read_indices = read.indices
                read_weights = read.weights
            if read_indices.shape[-2:] != (expected_candidates, field.group_size):
                raise BatchedRefineContractError(
                    "composed Recall query did not expose factor Top-K identities"
                )
            prefix = (*sequence.shape[:2], factor_count, available_k, field.group_size)
            factor_slot_index = read_indices.reshape(prefix)
            factor_weights = read_weights.reshape(prefix)
            factor_group_index = torch.div(
                factor_slot_index[..., 0], field.group_size, rounding_mode="floor"
            )
            offsets = torch.arange(factor_count, device=sequence.device).view(
                1, 1, -1, 1
            )
            local_group_index = factor_group_index - offsets * factor_groups
            factor_route_mass = factor_route.gather(-1, local_group_index)
            factor_candidate_rank, candidate_log_score = _joint_factor_topk(
                factor_route_mass,
                resolved_formula_beam_width,
            )
            factor_candidate_rank = factor_candidate_rank[..., :resolved_k, :]
            candidate_log_score = candidate_log_score[..., :resolved_k]
            slot_index = _gather_factor_candidates(
                factor_slot_index,
                factor_candidate_rank,
            )
            weights = _gather_factor_candidates(
                factor_weights, factor_candidate_rank
            )
            group_index = _gather_factor_candidates(
                factor_group_index,
                factor_candidate_rank,
            )
            route_mass = _gather_factor_candidates(
                factor_route_mass,
                factor_candidate_rank,
            )
        selection_weight = weights.sum(dim=-1)
        candidate_slot_weight = weights / selection_weight.unsqueeze(-1).clamp_min(
            torch.finfo(weights.dtype).eps
        )
        finite = (
            torch.isfinite(route_mass)
            & (route_mass >= 0)
            & torch.isfinite(selection_weight)
            & (selection_weight >= 0)
        ).all(dim=-1)
        candidate_mask = token_mask.unsqueeze(-1) & active_branch_mask.unsqueeze(1)
        selected_values = torch.nn.functional.embedding(slot_index, value_bank)
        factors = (
            selected_values.to(dtype=sequence.dtype)
            * candidate_slot_weight.to(dtype=sequence.dtype).unsqueeze(-1)
        ).sum(dim=-2)
        branch_input = sequence.unsqueeze(2).expand(-1, -1, resolved_k, -1)
        if field.value_composition == "product":
            candidate_context = field._compose_product_write(
                branch_input,
                factors[..., 0, :],
                factors[..., 1, :],
            )
        elif field.value_composition == "custom":
            candidate_context = field._compose_custom_write(branch_input, factors)
        else:
            candidate_context = field._compose_state_write(branch_input, factors)
        factor_route_index = field._factor_route_index
    torch._assert_async(
        torch.all(finite | ~token_mask.unsqueeze(-1)),
        "Recall Top-K candidate tensors must be finite and non-negative",
    )
    route_mask = candidate_mask.unsqueeze(-1) if composed else candidate_mask
    route_mass = torch.where(route_mask, route_mass, torch.zeros_like(route_mass))
    selection_weight = torch.where(
        route_mask,
        selection_weight,
        torch.zeros_like(selection_weight),
    )
    candidate_log_score = torch.where(
        candidate_mask,
        candidate_log_score,
        torch.zeros_like(candidate_log_score),
    )
    candidate_slot_weight = torch.where(
        route_mask.unsqueeze(-1),
        candidate_slot_weight,
        torch.zeros_like(candidate_slot_weight),
    )
    candidate_context = torch.where(
        candidate_mask.unsqueeze(-1),
        candidate_context,
        torch.zeros_like(candidate_context),
    )
    branch_mask = candidate_mask.any(dim=1)
    branch_origin_index = torch.arange(
        resolved_k,
        device=sequence.device,
        dtype=torch.long,
    ).expand(sequence.shape[0], -1)
    source_config = _source_config_fingerprint(
        recall,
        field,
        max_k=resolved_k,
        formula_beam_width=resolved_formula_beam_width,
        candidate_policy=candidate_policy,
        partition_quota=partition_quota,
        partition_coherence=resolved_partition_coherence,
    )
    (
        partition_names,
        partition_ranges,
        partition_member_fingerprints,
        partition_layout_fingerprint,
    ) = _partition_contract(field)
    candidate_partition_index = _candidate_partition_index(
        group_index,
        route_width=field._route_width(),
        ranges=partition_ranges,
    )
    candidate_values = (
        group_index,
        candidate_partition_index,
        slot_index,
        candidate_slot_weight,
        candidate_context,
        route_mass,
        selection_weight,
        candidate_log_score,
        candidate_mask,
        branch_mask,
        token_mask,
        branch_origin_index,
        active_k_tensor,
        requested_active_k_tensor,
        *((factor_candidate_rank,) if factor_candidate_rank is not None else ()),
        *((factor_route_index,) if factor_route_index is not None else ()),
    )
    candidate_tensors = tuple(
        _versioned_runtime_tensor(tensor)
        for tensor in candidate_values
    )
    (
        group_index,
        candidate_partition_index,
        slot_index,
        candidate_slot_weight,
        candidate_context,
        route_mass,
        selection_weight,
        candidate_log_score,
        candidate_mask,
        branch_mask,
        token_mask,
    ) = candidate_tensors[:11]
    branch_origin_index = candidate_tensors[11]
    active_k_tensor = candidate_tensors[12]
    requested_active_k_tensor = candidate_tensors[13]
    factor_candidate_rank = None if not composed else candidate_tensors[14]
    factor_route_index = None if not composed else candidate_tensors[15]
    input_version = _tensor_version(value)
    input_snapshot = _versioned_runtime_tensor(value) if input_version < 0 else None
    formula_ref = _recall_formula_reference(field)
    formula_config_fingerprint = _config_fingerprint(
        {
            "formula_ref": formula_ref,
            "formula_behavior": _recall_formula_behavior_config(field),
            "formula_provider": _recall_formula_provider_metadata(field),
            "provider_type": (
                f"{type(field.formula).__module__}.{type(field.formula).__qualname__}"
            ),
            "contract": (
                None
                if field.formula_contract is None
                else field.formula_contract.to_dict()
            ),
            "value_composition": field.value_composition,
            "factor_names": field.factor_names,
            "factor_route_names": field.factor_route_names,
            "factor_route_indices": field.factor_route_indices,
            "layout_fingerprint": field._route_layout_fingerprint(),
        }
    )
    batch_type = RecallFormulaBranchBatch if composed else RecallBranchBatch
    return batch_type(
        source_ref=recall._component_reference,
        source_instance_token=id(field),
        source_config_fingerprint=source_config,
        layout_fingerprint=field._route_layout_fingerprint(),
        query_version=source_versions[0],
        key_bank_version=source_versions[1],
        group_bank_version=source_versions[2],
        value_bank_version=source_versions[3],
        input_instance_token=id(value),
        input_version=input_version,
        input_shape=tuple(value.shape),
        input_dtype=str(value.dtype),
        input_device=str(value.device),
        input_snapshot=input_snapshot,
        input_snapshot_token=0 if input_snapshot is None else id(input_snapshot),
        input_snapshot_version=(
            -1 if input_snapshot is None else _tensor_version(input_snapshot)
        ),
        candidate_tensor_tokens=tuple(id(tensor) for tensor in candidate_tensors),
        candidate_tensor_versions=tuple(_tensor_version(tensor) for tensor in candidate_tensors),
        max_k=resolved_k,
        source_topk=available_k,
        formula_beam_width=resolved_formula_beam_width,
        group_count=field.group_bank.shape[0],
        group_size=field.group_size,
        partition_names=partition_names,
        partition_ranges=partition_ranges,
        partition_member_fingerprints=partition_member_fingerprints,
        partition_layout_fingerprint=partition_layout_fingerprint,
        candidate_group_index=group_index,
        candidate_partition_index=candidate_partition_index,
        candidate_slot_index=slot_index,
        candidate_slot_weight=candidate_slot_weight,
        candidate_context=candidate_context,
        route_mass=route_mass,
        selection_weight=selection_weight,
        candidate_log_score=candidate_log_score,
        candidate_mask=candidate_mask,
        branch_mask=branch_mask,
        token_mask=token_mask,
        branch_origin_index=branch_origin_index,
        active_k=active_k_tensor,
        requested_active_k=requested_active_k_tensor,
        query_rng_fingerprint=(
            None if not route_exploration else rng_plan.fingerprint
        ),
        routing_normalizer=field.routing_normalizer,
        value_composition=field.value_composition,
        factor_count=factor_count,
        factor_candidate_rank=factor_candidate_rank,
        factor_route_index=factor_route_index,
        candidate_policy=candidate_policy,
        partition_quota=partition_quota,
        partition_coherence=resolved_partition_coherence,
        formula_ref=formula_ref,
        formula_config_fingerprint=formula_config_fingerprint,
        source_execution_tensor_lineage=source_execution_tensor_lineage,
        schema_version=6 if composed else 3,
    )


_GLOBAL_DIAGNOSTICS = frozenset(
    {
        "recall_trace_schema",
        "recall_active_fraction",
        "recall_kernel_steps",
        "recall_logical_token_steps",
    }
)


_BATCHED_RESULT_FACTORY_TOKEN = object()


@dataclass(frozen=True, init=False)
class BatchedRefineResult:
    """K independent Recall trajectories produced by one candidate query."""

    candidates: RecallBranchBatch
    value: Tensor
    delta: Tensor
    branch_diagnostics: Mapping[str, Tensor]
    global_diagnostics: Mapping[str, Tensor]
    value_version: int
    delta_version: int
    diagnostic_lineage: tuple[tuple[str, int, int], ...]
    plan_ref: str
    plan_config_fingerprint: str
    execution_layout: str
    operation_ref: str | None
    formula_route_fingerprint: str | None
    topology_refs: tuple[str, ...]
    topology_contract_fingerprints: tuple[str, ...]
    branch_policy_fingerprint: str | None
    execution_rng_fingerprint: str | None
    execution_rng_stream_key: str | None
    execution_rng_domains: tuple[str, ...]
    schema_version: int = 1
    _component_reference: ClassVar[str] = "arti/batched-refine-result@1"

    def __init__(
        self,
        *,
        candidates: RecallBranchBatch,
        value: Tensor,
        delta: Tensor,
        branch_diagnostics: Mapping[str, Tensor],
        global_diagnostics: Mapping[str, Tensor],
        _factory_token: object,
        plan_ref: str = "arti/batched-refine-plan@1",
        plan_config_fingerprint: str = "0" * 64,
        execution_layout: str = "static_capacity",
        operation_ref: str | None = None,
        formula_route_fingerprint: str | None = None,
        topology_refs: tuple[str, ...] = (),
        topology_contract_fingerprints: tuple[str, ...] = (),
        branch_policy_fingerprint: str | None = None,
        execution_rng_fingerprint: str | None = None,
        execution_rng_stream_key: str | None = None,
        execution_rng_domains: tuple[str, ...] = (),
    ) -> None:
        if _factory_token is not _BATCHED_RESULT_FACTORY_TOKEN:
            raise BatchedRefineContractError(
                "BatchedRefineResult must come from run_batched_refine"
            )
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "value", value)
        object.__setattr__(self, "delta", delta)
        object.__setattr__(self, "branch_diagnostics", branch_diagnostics)
        object.__setattr__(self, "global_diagnostics", global_diagnostics)
        object.__setattr__(self, "plan_ref", plan_ref)
        object.__setattr__(
            self, "plan_config_fingerprint", plan_config_fingerprint
        )
        object.__setattr__(self, "execution_layout", execution_layout)
        object.__setattr__(self, "operation_ref", operation_ref)
        object.__setattr__(
            self, "formula_route_fingerprint", formula_route_fingerprint
        )
        object.__setattr__(self, "topology_refs", tuple(topology_refs))
        object.__setattr__(
            self,
            "topology_contract_fingerprints",
            tuple(topology_contract_fingerprints),
        )
        object.__setattr__(
            self, "branch_policy_fingerprint", branch_policy_fingerprint
        )
        object.__setattr__(
            self, "execution_rng_fingerprint", execution_rng_fingerprint
        )
        object.__setattr__(self, "execution_rng_stream_key", execution_rng_stream_key)
        object.__setattr__(self, "execution_rng_domains", execution_rng_domains)
        object.__setattr__(
            self,
            "value_version",
            _require_tensor_version(value, name="Batched Refine value"),
        )
        object.__setattr__(
            self,
            "delta_version",
            _require_tensor_version(delta, name="Batched Refine delta"),
        )
        diagnostic_lineage = tuple(
            (f"branch:{name}", id(tensor), _require_tensor_version(tensor, name=name))
            for name, tensor in sorted(branch_diagnostics.items())
        ) + tuple(
            (f"global:{name}", id(tensor), _require_tensor_version(tensor, name=name))
            for name, tensor in sorted(global_diagnostics.items())
        )
        object.__setattr__(self, "diagnostic_lineage", diagnostic_lineage)
        object.__setattr__(self, "schema_version", 1)
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise BatchedRefineContractError("unsupported Batched Refine result schema")
        if self.plan_ref != BatchedRefinePlan._component_reference:
            raise BatchedRefineContractError("unsupported Batched Refine plan reference")
        if self.execution_layout not in {"static_capacity", "packed_active"}:
            raise BatchedRefineContractError(
                "unsupported Batched Refine execution layout"
            )
        if (
            len(self.plan_config_fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in self.plan_config_fingerprint)
        ):
            raise BatchedRefineContractError("invalid Batched Refine plan fingerprint")
        if self.formula_route_fingerprint is not None and (
            len(self.formula_route_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.formula_route_fingerprint
            )
        ):
            raise BatchedRefineContractError("invalid Formula route fingerprint")
        if any(not isinstance(item, str) or not item for item in self.topology_refs):
            raise BatchedRefineContractError("invalid topology component reference")
        for fingerprint in self.topology_contract_fingerprints:
            if (
                not isinstance(fingerprint, str)
                or len(fingerprint) != 64
                or any(character not in "0123456789abcdef" for character in fingerprint)
            ):
                raise BatchedRefineContractError(
                    "invalid topology contract fingerprint"
                )
        if self.branch_policy_fingerprint is not None and (
            len(self.branch_policy_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.branch_policy_fingerprint
            )
        ):
            raise BatchedRefineContractError("invalid branch policy fingerprint")
        if self.execution_rng_fingerprint is not None and (
            len(self.execution_rng_fingerprint) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.execution_rng_fingerprint
            )
        ):
            raise BatchedRefineContractError("invalid execution RNG fingerprint")
        if (self.execution_rng_fingerprint is None) != (
            self.execution_rng_stream_key is None
        ):
            raise BatchedRefineContractError(
                "execution RNG fingerprint and stream key must be present together"
            )
        if self.execution_rng_stream_key is not None and (
            not self.execution_rng_stream_key
            or any(
                character not in _RNG_IDENTIFIER_CHARS
                for character in self.execution_rng_stream_key
            )
        ):
            raise BatchedRefineContractError("invalid execution RNG stream key")
        if (
            tuple(sorted(set(self.execution_rng_domains)))
            != self.execution_rng_domains
            or any(domain not in _RNG_PHASES for domain in self.execution_rng_domains)
        ):
            raise BatchedRefineContractError("invalid execution RNG domain receipt")
        if bool(self.execution_rng_domains) != (
            self.execution_rng_fingerprint is not None
        ):
            raise BatchedRefineContractError(
                "keyed execution must record at least one consumed RNG domain"
            )
        expected = (
            self.candidates.batch_size,
            self.candidates.max_k,
            self.candidates.token_count,
        )
        if (
            not isinstance(self.value, Tensor)
            or not self.value.is_floating_point()
            or self.value.ndim != 4
            or self.value.shape[:3] != expected
        ):
            raise BatchedRefineContractError("value must be floating [B,K,N,D]")
        if not isinstance(self.delta, Tensor) or self.delta.shape != self.value.shape:
            raise BatchedRefineContractError("delta must match the branch value tensor")
        if self.delta.device != self.value.device:
            raise BatchedRefineContractError("result tensors must use one device")
        for name, tensor in self.branch_diagnostics.items():
            if not isinstance(name, str) or not isinstance(tensor, Tensor):
                raise BatchedRefineContractError(
                    "branch diagnostics must map strings to tensors"
                )
            if tensor.ndim < 2 or tensor.shape[:2] != expected[:2]:
                raise BatchedRefineContractError(
                    f"branch diagnostic {name!r} must begin with [B,K]"
                )
        for name, tensor in self.global_diagnostics.items():
            if not isinstance(name, str) or not isinstance(tensor, Tensor):
                raise BatchedRefineContractError(
                    "global diagnostics must map strings to tensors"
                )

    def assert_unchanged(self) -> None:
        """Fail closed when versioned result tensors changed after execution."""

        for tensor, expected in (
            (self.value, self.value_version),
            (self.delta, self.delta_version),
        ):
            if _tensor_version(tensor) != expected:
                raise BatchedRefineContractError(
                    "Batched Refine result changed before staging"
                )
        current_diagnostics = {
            **{f"branch:{name}": tensor for name, tensor in self.branch_diagnostics.items()},
            **{f"global:{name}": tensor for name, tensor in self.global_diagnostics.items()},
        }
        if set(current_diagnostics) != {name for name, _, _ in self.diagnostic_lineage}:
            raise BatchedRefineContractError("Batched Refine diagnostics changed before staging")
        for name, token, version in self.diagnostic_lineage:
            tensor = current_diagnostics[name]
            if id(tensor) != token or _tensor_version(tensor) != version:
                raise BatchedRefineContractError(
                    "Batched Refine diagnostics changed before staging"
                )


def run_batched_refine(
    recall: object,
    value: Tensor,
    *,
    mask: Tensor | None = None,
    candidates: RecallBranchBatch | None = None,
    max_k: int | None = None,
    formula_beam_width: int | None = None,
    active_k: int | Tensor | None = None,
    candidate_allocation: str | None = None,
    bank_quotas: tuple[int, ...] | None = None,
    partition_coherence: str | None = None,
    refine_policy: object | None = None,
    plan: BatchedRefinePlan | None = None,
    rng_plan: ExecutionRNGPlan | None = None,
) -> BatchedRefineResult:
    """Run K Recall trajectories from one Top-K query in one tensor call."""

    from .nn import Recall
    from .recall_refine import (
        AdaptiveRefinePolicy,
        AdaptiveRefineSchedule,
        RecallStopReason,
        RefinePolicy,
    )

    if not isinstance(recall, Recall):
        raise TypeError("recall must be arti.nn.Recall")
    plan = BatchedRefinePlan.recall_only() if plan is None else plan
    if not isinstance(plan, BatchedRefinePlan):
        raise TypeError("plan must be BatchedRefinePlan or None")
    plan.assert_unchanged()
    if candidates is None:
        candidates = query_recall_branches(
            recall,
            value,
            mask=mask,
            max_k=max_k,
            formula_beam_width=formula_beam_width,
            active_k=active_k,
            candidate_allocation=candidate_allocation,
            bank_quotas=bank_quotas,
            partition_coherence=partition_coherence,
            rng_plan=rng_plan,
        )
    elif (
        max_k is not None
        or formula_beam_width is not None
        or active_k is not None
        or candidate_allocation is not None
        or bank_quotas is not None
        or partition_coherence is not None
    ):
        raise BatchedRefineContractError(
            "candidate query options cannot be combined with candidates"
        )
    if not isinstance(candidates, RecallBranchBatch):
        raise TypeError("candidates must be RecallBranchBatch")
    candidates.assert_unchanged()
    if candidates.query_rng_fingerprint is not None and (
        rng_plan is None or rng_plan.fingerprint != candidates.query_rng_fingerprint
    ):
        raise BatchedRefineContractError(
            "candidate query RNG does not match the Batched Refine execution plan"
        )
    field = recall.state.recall
    if candidates.source_ref != recall._component_reference:
        raise BatchedRefineContractError("candidate source does not match Recall")
    if candidates.source_instance_token != id(field):
        raise BatchedRefineContractError("candidate source instance does not match Recall")
    if candidates.layout_fingerprint != field._route_layout_fingerprint():
        raise BatchedRefineContractError("candidate layout does not match Recall")
    if candidates.source_config_fingerprint != _source_config_fingerprint(
        recall,
        field,
        max_k=candidates.max_k,
        formula_beam_width=candidates.formula_beam_width,
        candidate_policy=candidates.candidate_policy,
        partition_quota=candidates.partition_quota,
        partition_coherence=candidates.partition_coherence,
    ):
        raise BatchedRefineContractError(
            "candidate routing configuration changed after its Recall query"
        )
    current_versions = (
        _tensor_version(field.query.weight),
        _tensor_version(field.key_bank),
        _tensor_version(field.group_bank),
        _tensor_version(field.bank),
    )
    candidate_versions = (
        candidates.query_version,
        candidates.key_bank_version,
        candidates.group_bank_version,
        candidates.value_bank_version,
    )
    if candidate_versions != current_versions:
        raise BatchedRefineContractError("candidate source changed after its Recall query")
    current_formula_tensors = (
        ()
        if not isinstance(field.formula, nn.Module)
        else _operation_owned_tensors(field.formula)
    )
    current_activation_tensors = _operation_owned_tensors(
        recall.state.recall_activation
    )
    current_execution_tensors = tuple(
        (f"formula:{name}", tensor) for name, tensor in current_formula_tensors
    ) + tuple(
        (f"activation:{name}", tensor)
        for name, tensor in current_activation_tensors
    )
    current_execution_lineage = tuple(
        (name, id(tensor), _tensor_version(tensor))
        for name, tensor in current_execution_tensors
    )
    if current_execution_lineage != candidates.source_execution_tensor_lineage:
        raise BatchedRefineContractError(
            "candidate source execution tensors changed after its Recall query"
        )
    branch_policy = refine_policy if isinstance(refine_policy, BranchRefinePolicy) else None
    policy = (
        branch_policy.base
        if branch_policy is not None
        else refine_policy or RefinePolicy.fixed(1, trace_level="routes")
    )
    if not isinstance(policy, (RefinePolicy, AdaptiveRefinePolicy)):
        raise TypeError("refine_policy must be RefinePolicy or AdaptiveRefinePolicy")
    if isinstance(policy, AdaptiveRefinePolicy) and policy.executor != "static_masked":
        raise BatchedRefineContractError(
            "Batched Refine requires static_masked execution to avoid host synchronization"
        )
    refine_schedule = (
        None if branch_policy is None else branch_policy.bind(candidates)
    )

    source_value = candidates.source_state(value)
    batch, tokens, dim = source_value.shape
    branches = candidates.max_k
    static_capacity_rows = batch * branches
    packed_active = plan.execution_layout == "packed_active"
    if packed_active:
        compiler = getattr(torch, "compiler", None)
        is_compiling = getattr(compiler, "is_compiling", None)
        if callable(is_compiling) and is_compiling():
            raise BatchedRefineContractError(
                "packed_active execution is eager-only until a fixed-P bucket "
                "executor is selected"
            )
        if source_value.is_cuda and torch.cuda.is_current_stream_capturing():
            raise BatchedRefineContractError(
                "packed_active execution cannot be captured directly; select a "
                "fixed-P bucket before CUDA Graph capture"
            )
    active_flat_index = (
        torch.nonzero(candidates.branch_mask.reshape(-1), as_tuple=False)
        .reshape(-1)
        .contiguous()
        if packed_active
        else None
    )
    if active_flat_index is None:
        flat_value = candidates.expand_state(value).reshape(
            static_capacity_rows, tokens, dim
        )
        flat_mask = candidates.flattened_token_mask()
    else:
        sample_index = torch.div(active_flat_index, branches, rounding_mode="floor")
        flat_value = source_value.index_select(0, sample_index).contiguous()
        flat_mask = candidates.flattened_token_mask().index_select(
            0, active_flat_index
        )
    physical_rows = int(flat_value.shape[0])
    activation = recall.state.recall_activation
    stochastic_half = isinstance(activation, nn.Module) and bool(
        getattr(activation, "stochastic", False)
    )
    stochastic_dropout = recall.state.training and recall.state.dropout.p > 0
    if (stochastic_half or stochastic_dropout) and rng_plan is None:
        raise BatchedRefineContractError(
            "stochastic Half and training dropout require ExecutionRNGPlan"
        )
    if rng_plan is not None and not isinstance(rng_plan, ExecutionRNGPlan):
        raise TypeError("rng_plan must be ExecutionRNGPlan or None")
    random_source = None
    if rng_plan is not None:
        random_source = rng_plan.bind(candidates.branch_origin_index)
        if active_flat_index is not None:
            random_source = random_source.select_rows(active_flat_index)
    rng_domains = tuple(
        sorted(
            {
                *(
                    ("candidate-route",)
                    if candidates.query_rng_fingerprint is not None
                    else ()
                ),
                *(
                    ("refine-route",)
                    if physical_rows > 0
                    and field.training
                    and field._bank_gradient_enabled
                    and field.route_exploration > 0
                    else ()
                ),
                *(("half-survival",) if physical_rows > 0 and stochastic_half else ()),
                *(("recall-dropout",) if physical_rows > 0 and stochastic_dropout else ()),
            }
        )
    )
    operation_trace_records: list[
        tuple[tuple[object, ...], Tensor | None, int]
    ] = []
    state_operation = (
        None
        if plan.operation is None
        else plan.operation.indexed(
            candidates.branch_origin_index,
            active_flat_index=active_flat_index,
            trace_records=operation_trace_records,
        )
    )
    selected_groups = candidates.flattened_execution_groups()
    if active_flat_index is not None:
        selected_groups = selected_groups.index_select(0, active_flat_index)
        if refine_schedule is not None:
            def select_schedule(value: Tensor | None) -> Tensor | None:
                return (
                    None
                    if value is None
                    else value.index_select(0, active_flat_index)
                )

            refine_schedule = AdaptiveRefineSchedule(
                min_steps=select_schedule(refine_schedule.min_steps),
                max_steps=select_schedule(refine_schedule.max_steps),
                absolute_tolerance=select_schedule(
                    refine_schedule.absolute_tolerance
                ),
                relative_tolerance=select_schedule(
                    refine_schedule.relative_tolerance
                ),
                patience=select_schedule(refine_schedule.patience),
                route_tolerance=select_schedule(refine_schedule.route_tolerance),
                cycle_tolerance=select_schedule(refine_schedule.cycle_tolerance),
            )
    if physical_rows:
        refined, delta, diagnostics = recall.state(
            flat_value,
            flat_mask,
            refine_policy=policy,
            selected_groups=selected_groups,
            selected_groups_first_step_only=True,
            state_operation=state_operation,
            refine_schedule=refine_schedule,
            _random_source=random_source,
        )
    else:
        refine_steps = policy.max_steps
        token_shape = (0, tokens)
        token_step_shape = (0, refine_steps, tokens)
        diagnostics = {
            "recall_trace_schema": torch.tensor(
                2, device=source_value.device, dtype=torch.int64
            ),
            "recall_token_steps_attempted": torch.zeros(
                token_shape, device=source_value.device, dtype=torch.int64
            ),
            "recall_token_steps_committed": torch.zeros(
                token_shape, device=source_value.device, dtype=torch.int64
            ),
            "recall_token_stop_reason": torch.full(
                token_shape,
                int(RecallStopReason.MASKED),
                device=source_value.device,
                dtype=torch.int64,
            ),
            "recall_token_step_attempted": torch.zeros(
                token_step_shape, device=source_value.device, dtype=torch.bool
            ),
            "recall_token_step_committed": torch.zeros(
                token_step_shape, device=source_value.device, dtype=torch.bool
            ),
            "recall_token_step_update_ratio": source_value.new_zeros(
                token_step_shape
            ),
            "recall_token_step_route_change": source_value.new_zeros(
                token_step_shape
            ),
            "recall_token_step_effective_read_change": source_value.new_zeros(
                token_step_shape
            ),
            "recall_active_fraction": source_value.new_zeros((refine_steps,)),
            "recall_kernel_steps": torch.zeros(
                (), device=source_value.device, dtype=torch.int64
            ),
            "recall_logical_token_steps": torch.zeros(
                (), device=source_value.device, dtype=torch.int64
            ),
            "recall_steps_attempted": torch.zeros(
                (0,), device=source_value.device, dtype=torch.int64
            ),
            "recall_steps_committed": torch.zeros(
                (0,), device=source_value.device, dtype=torch.int64
            ),
            "recall_step_attempted": torch.zeros(
                (0, refine_steps), device=source_value.device, dtype=torch.bool
            ),
            "recall_step_committed": torch.zeros(
                (0, refine_steps), device=source_value.device, dtype=torch.bool
            ),
            "recall_step_update_ratio": source_value.new_zeros((0, refine_steps)),
            "recall_step_route_change": source_value.new_zeros((0, refine_steps)),
            "recall_step_effective_read_change": source_value.new_zeros(
                (0, refine_steps)
            ),
            "recall_update_ratio": source_value.new_zeros(token_shape),
            "recall_raw_context": source_value.new_zeros((0, tokens, dim)),
            "recall_context": source_value.new_zeros((0, tokens, dim)),
            "recall_effect_norm": source_value.new_zeros(token_shape),
            "recall_write_norm": source_value.new_zeros(token_shape),
        }
        refined = source_value.new_empty((0, tokens, dim))
        delta = source_value.new_empty((0, tokens, dim))
    plan.assert_unchanged()
    branch_diagnostics: dict[str, Tensor] = {}
    global_diagnostics: dict[str, Tensor] = {}

    def scatter_rows(tensor: Tensor, *, fill_value: int | float = 0) -> Tensor:
        if active_flat_index is None:
            return tensor.reshape(batch, branches, *tensor.shape[1:])
        full = torch.full(
            (static_capacity_rows, *tensor.shape[1:]),
            fill_value,
            device=tensor.device,
            dtype=tensor.dtype,
        )
        if physical_rows:
            full.index_copy_(0, active_flat_index, tensor)
        return full.reshape(batch, branches, *tensor.shape[1:])

    for name, tensor in diagnostics.items():
        if name in _GLOBAL_DIAGNOSTICS:
            global_diagnostics[name] = _versioned_runtime_tensor(tensor)
        elif tensor.ndim > 0 and tensor.shape[0] == physical_rows:
            fill_value = (
                int(RecallStopReason.MASKED)
                if name == "recall_token_stop_reason"
                else -1
                if "index" in name
                else 0
            )
            branch_diagnostics[name] = _versioned_runtime_tensor(
                scatter_rows(tensor, fill_value=fill_value)
            )
        else:
            global_diagnostics[name] = _versioned_runtime_tensor(tensor)
    kernel_steps = diagnostics.get("recall_kernel_steps")
    if kernel_steps is None and isinstance(policy, RefinePolicy):
        kernel_steps = torch.tensor(
            policy.max_steps,
            device=flat_value.device,
            dtype=torch.int64,
        )
        global_diagnostics["recall_kernel_steps"] = _versioned_runtime_tensor(
            kernel_steps
        )
    if plan.operation is not None and physical_rows:
        if not isinstance(kernel_steps, Tensor) or kernel_steps.numel() != 1:
            raise BatchedRefineContractError(
                "composed Batched Refine requires a scalar kernel-step receipt"
            )
        resident_steps = plan.operation.operation.resident_refine_steps
        if len(operation_trace_records) != policy.max_steps:
            raise BatchedRefineContractError(
                "resident operation trace count does not match outer kernel depth"
            )
        if any(len(traces) != resident_steps for traces, _permutation, _active in operation_trace_records):
            raise BatchedRefineContractError(
                "resident Formula trace count does not match configured inner depth"
            )

        def formula_trace(trace: object) -> object:
            return getattr(trace, "formula", trace)

        def stack_formula_field(name: str) -> Tensor:
            outer = torch.stack(
                [
                    torch.stack(
                        [getattr(formula_trace(trace), name) for trace in traces],
                        dim=0,
                    )
                    for traces, _permutation, _active in operation_trace_records
                ],
                dim=0,
            )
            # [outer, inner, P, ...] -> [P, outer, inner, ...] -> [B,K,...]
            permuted = outer.permute(2, 0, 1, *range(3, outer.ndim))
            return scatter_rows(permuted)

        valid_trace = stack_formula_field("valid_mask")
        fire_trace = stack_formula_field("fire_mask")
        commit_trace = stack_formula_field("commit_mask")
        formula_dims = tuple(range(2, valid_trace.ndim))
        formula_cells = valid_trace.sum(dim=formula_dims, dtype=torch.int64)
        formula_fire = fire_trace.sum(dim=formula_dims, dtype=torch.int64)
        formula_commit = commit_trace.sum(dim=formula_dims, dtype=torch.int64)
        attempted_trace = branch_diagnostics.get("recall_step_attempted")
        if attempted_trace is None or attempted_trace.shape[:3] != (
            batch,
            branches,
            policy.max_steps,
        ):
            raise BatchedRefineContractError(
                "composed Batched Refine requires [B,K,R,...] attempt trace"
            )
        outer_active = attempted_trace.reshape(
            batch, branches, policy.max_steps, -1
        ).any(dim=-1)
        operation_active = outer_active.reshape(
            batch,
            branches,
            policy.max_steps,
            *([1] * (valid_trace.ndim - 3)),
        )
        effective_valid_trace = valid_trace & operation_active
        effective_fire_trace = fire_trace & operation_active
        effective_commit_trace = commit_trace & operation_active
        effective_dims = tuple(range(2, effective_valid_trace.ndim))
        effective_cells = effective_valid_trace.sum(
            dim=effective_dims, dtype=torch.int64
        )
        effective_fire = effective_fire_trace.sum(
            dim=effective_dims, dtype=torch.int64
        )
        effective_commit = effective_commit_trace.sum(
            dim=effective_dims, dtype=torch.int64
        )
        declared_cells, declared_fire, declared_commit = (
            count.reshape(batch, branches) * policy.max_steps
            for count in plan.operation.work_per_execution(
                candidates.branch_origin_index
            )
        )
        global_diagnostics["batched_resident_operation_steps"] = (
            _versioned_runtime_tensor(
                torch.tensor(
                    resident_steps,
                    device=flat_value.device,
                    dtype=torch.int64,
                )
            )
        )
        branch_diagnostics.update(
            {
                "batched_formula_cells": _versioned_runtime_tensor(
                    formula_cells
                ),
                "batched_formula_fire_count": _versioned_runtime_tensor(
                    formula_fire
                ),
                "batched_formula_commit_count": _versioned_runtime_tensor(
                    formula_commit
                ),
                "batched_formula_route_applications": _versioned_runtime_tensor(
                    candidates.branch_mask.to(torch.int64)
                    * (policy.max_steps * resident_steps)
                ),
                "batched_formula_valid_trace": _versioned_runtime_tensor(valid_trace),
                "batched_formula_fire_trace": _versioned_runtime_tensor(fire_trace),
                "batched_formula_commit_trace": _versioned_runtime_tensor(commit_trace),
                "batched_formula_declared_cells": _versioned_runtime_tensor(
                    declared_cells
                ),
                "batched_formula_declared_fire_count": _versioned_runtime_tensor(
                    declared_fire
                ),
                "batched_formula_declared_commit_count": _versioned_runtime_tensor(
                    declared_commit
                ),
                "batched_formula_invoked_cells": _versioned_runtime_tensor(
                    formula_cells.clone()
                ),
                "batched_formula_invoked_fire_count": _versioned_runtime_tensor(
                    formula_fire.clone()
                ),
                "batched_formula_invoked_commit_count": _versioned_runtime_tensor(
                    formula_commit.clone()
                ),
                "batched_formula_effective_cells": _versioned_runtime_tensor(
                    effective_cells
                ),
                "batched_formula_effective_fire_count": _versioned_runtime_tensor(
                    effective_fire
                ),
                "batched_formula_effective_commit_count": _versioned_runtime_tensor(
                    effective_commit
                ),
            }
        )
        permutations = [
            permutation
            for _traces, permutation, _active_count in operation_trace_records
        ]
        if any(permutation is not None for permutation in permutations):
            if any(permutation is None for permutation in permutations):
                raise BatchedRefineContractError(
                    "resident topology trace is incomplete"
                )
            stacked_permutation = torch.stack(
                [permutation for permutation in permutations if permutation is not None],
                dim=1,
            )
            topology_trace = scatter_rows(stacked_permutation)
            active_counts = {
                active_count
                for _traces, _permutation, active_count in operation_trace_records
            }
            if len(active_counts) != 1:
                raise BatchedRefineContractError(
                    "resident topology active_count changed during execution"
                )
            branch_diagnostics["batched_topology_permutation_trace"] = (
                _versioned_runtime_tensor(topology_trace)
            )
            branch_diagnostics["batched_topology_unfold_verified"] = (
                _versioned_runtime_tensor(
                    scatter_rows(
                        torch.ones(
                            (physical_rows, policy.max_steps),
                            device=flat_value.device,
                            dtype=torch.bool,
                        )
                    )
                )
            )
            global_diagnostics["batched_topology_active_count"] = (
                _versioned_runtime_tensor(
                    torch.tensor(
                        active_counts.pop(),
                        device=flat_value.device,
                        dtype=torch.int64,
                    )
                )
            )
    elif plan.operation is not None:
        operation = plan.operation.operation
        resident_steps = operation.resident_refine_steps
        route_shape = tuple(operation.route.valid_mask.shape[1:])
        formula_trace_shape = (
            batch,
            branches,
            policy.max_steps,
            resident_steps,
            *route_shape,
        )
        declared_cells, declared_fire, declared_commit = (
            count.reshape(batch, branches) * policy.max_steps
            for count in plan.operation.work_per_execution(
                candidates.branch_origin_index
            )
        )
        zero_count = torch.zeros(
            (batch, branches), device=source_value.device, dtype=torch.int64
        )
        branch_diagnostics.update(
            {
                "batched_formula_cells": _versioned_runtime_tensor(
                    zero_count.clone()
                ),
                "batched_formula_fire_count": _versioned_runtime_tensor(
                    zero_count.clone()
                ),
                "batched_formula_commit_count": _versioned_runtime_tensor(
                    zero_count.clone()
                ),
                "batched_formula_valid_trace": _versioned_runtime_tensor(
                    torch.zeros(
                        formula_trace_shape,
                        device=source_value.device,
                        dtype=torch.bool,
                    )
                ),
                "batched_formula_fire_trace": _versioned_runtime_tensor(
                    torch.zeros(
                        formula_trace_shape,
                        device=source_value.device,
                        dtype=torch.bool,
                    )
                ),
                "batched_formula_commit_trace": _versioned_runtime_tensor(
                    torch.zeros(
                        formula_trace_shape,
                        device=source_value.device,
                        dtype=torch.bool,
                    )
                ),
                "batched_formula_route_applications": _versioned_runtime_tensor(
                    zero_count.clone()
                ),
                "batched_formula_declared_cells": _versioned_runtime_tensor(
                    declared_cells
                ),
                "batched_formula_declared_fire_count": _versioned_runtime_tensor(
                    declared_fire
                ),
                "batched_formula_declared_commit_count": _versioned_runtime_tensor(
                    declared_commit
                ),
                "batched_formula_invoked_cells": _versioned_runtime_tensor(
                    zero_count.clone()
                ),
                "batched_formula_invoked_fire_count": _versioned_runtime_tensor(
                    zero_count.clone()
                ),
                "batched_formula_invoked_commit_count": _versioned_runtime_tensor(
                    zero_count.clone()
                ),
                "batched_formula_effective_cells": _versioned_runtime_tensor(
                    zero_count.clone()
                ),
                "batched_formula_effective_fire_count": _versioned_runtime_tensor(
                    zero_count.clone()
                ),
                "batched_formula_effective_commit_count": _versioned_runtime_tensor(
                    zero_count.clone()
                ),
            }
        )
        global_diagnostics["batched_resident_operation_steps"] = (
            _versioned_runtime_tensor(
                torch.tensor(
                    resident_steps,
                    device=source_value.device,
                    dtype=torch.int64,
                )
            )
        )
        fold = getattr(operation, "fold", None)
        if fold is not None:
            branch_diagnostics["batched_topology_permutation_trace"] = (
                _versioned_runtime_tensor(
                    torch.full(
                        (batch, branches, policy.max_steps, tokens),
                        -1,
                        device=source_value.device,
                        dtype=torch.int64,
                    )
                )
            )
            branch_diagnostics["batched_topology_unfold_verified"] = (
                _versioned_runtime_tensor(
                    torch.zeros(
                        (batch, branches, policy.max_steps),
                        device=source_value.device,
                        dtype=torch.bool,
                    )
                )
            )
            global_diagnostics["batched_topology_active_count"] = (
                _versioned_runtime_tensor(
                    torch.tensor(
                        fold.topology.active_count,
                        device=source_value.device,
                        dtype=torch.int64,
                    )
                )
            )

    attempted_steps = branch_diagnostics.get("recall_steps_attempted")
    committed_steps = branch_diagnostics.get("recall_steps_committed")
    if attempted_steps is None or committed_steps is None:
        raise BatchedRefineContractError(
            "Batched Refine requires attempted and committed step diagnostics"
        )

    def executed_width(steps: Tensor) -> Tensor:
        if steps.shape[:2] != (batch, branches):
            raise BatchedRefineContractError(
                "branch step diagnostics must begin with [B,K]"
            )
        return (steps.reshape(batch, branches, -1) > 0).any(dim=-1).sum(
            dim=1, dtype=torch.int64
        )

    global_diagnostics.update(
        {
            "batched_branch_capacity_k": _versioned_runtime_tensor(
                torch.tensor(branches, device=flat_value.device, dtype=torch.int64)
            ),
            "batched_requested_k": _versioned_runtime_tensor(
                candidates.requested_active_k.clone()
            ),
            "batched_eligible_k": _versioned_runtime_tensor(
                candidates.active_k.clone()
            ),
            "batched_active_partition_count": _versioned_runtime_tensor(
                candidates.active_partition_count()
            ),
            "batched_active_partition_mask": _versioned_runtime_tensor(
                candidates.active_partition_mask()
            ),
            "batched_active_branch_count_by_partition": _versioned_runtime_tensor(
                candidates.active_branch_count_by_partition()
            ),
            "batched_attempted_k": _versioned_runtime_tensor(
                executed_width(attempted_steps)
            ),
            "batched_committed_k": _versioned_runtime_tensor(
                executed_width(committed_steps)
            ),
            "batched_physical_branch_rows": _versioned_runtime_tensor(
                torch.tensor(
                    physical_rows,
                    device=flat_value.device,
                    dtype=torch.int64,
                )
            ),
            "batched_static_capacity_rows": _versioned_runtime_tensor(
                torch.tensor(
                    static_capacity_rows,
                    device=flat_value.device,
                    dtype=torch.int64,
                )
            ),
            "batched_eligible_branch_rows": _versioned_runtime_tensor(
                candidates.branch_mask.sum(dtype=torch.int64)
            ),
            "batched_initially_inactive_rows": _versioned_runtime_tensor(
                torch.tensor(
                    static_capacity_rows - physical_rows,
                    device=flat_value.device,
                    dtype=torch.int64,
                )
            ),
            "batched_packing_mode": _versioned_runtime_tensor(
                torch.tensor(
                    1 if packed_active else 0,
                    device=flat_value.device,
                    dtype=torch.int64,
                )
            ),
            "batched_packed_active_flat_index": _versioned_runtime_tensor(
                (
                    torch.arange(
                        static_capacity_rows,
                        device=flat_value.device,
                        dtype=torch.int64,
                    )
                    if active_flat_index is None
                    else active_flat_index.clone()
                )
            ),
            "batched_executed_branch_mask": _versioned_runtime_tensor(
                candidates.branch_mask.clone()
            ),
        }
    )
    if active_flat_index is None:
        full_refined = refined.reshape(batch, branches, tokens, dim)
        full_delta = delta.reshape(batch, branches, tokens, dim)
    else:
        full_refined_flat = (
            source_value.unsqueeze(1)
            .expand(batch, branches, tokens, dim)
            .clone()
            .reshape(static_capacity_rows, tokens, dim)
        )
        full_delta_flat = source_value.new_zeros(
            (static_capacity_rows, tokens, dim)
        )
        if physical_rows:
            full_refined_flat.index_copy_(0, active_flat_index, refined)
            full_delta_flat.index_copy_(0, active_flat_index, delta)
        full_refined = full_refined_flat.reshape(batch, branches, tokens, dim)
        full_delta = full_delta_flat.reshape(batch, branches, tokens, dim)
    result_value = _versioned_runtime_tensor(full_refined)
    result_delta = _versioned_runtime_tensor(full_delta)
    return BatchedRefineResult(
        candidates=candidates,
        value=result_value,
        delta=result_delta,
        branch_diagnostics=MappingProxyType(branch_diagnostics),
        global_diagnostics=MappingProxyType(global_diagnostics),
        plan_ref=plan._component_reference,
        plan_config_fingerprint=plan.config_fingerprint,
        execution_layout=plan.execution_layout,
        operation_ref=(
            None if plan.operation is None else plan.operation.operation_ref
        ),
        formula_route_fingerprint=(
            None
            if plan.operation is None
            else plan.operation.formula_route_fingerprint
        ),
        topology_refs=(
            () if plan.operation is None else plan.operation.topology_refs
        ),
        topology_contract_fingerprints=(
            ()
            if plan.operation is None
            else plan.operation.topology_contract_fingerprints
        ),
        branch_policy_fingerprint=(
            None if branch_policy is None else branch_policy.config_fingerprint
        ),
        execution_rng_fingerprint=(
            None if not rng_domains else rng_plan.fingerprint
        ),
        execution_rng_stream_key=(
            None if not rng_domains else rng_plan.stream_key
        ),
        execution_rng_domains=rng_domains,
        _factory_token=_BATCHED_RESULT_FACTORY_TOKEN,
    )


__all__ = [
    "BatchedRefineContractError",
    "BranchRefinePolicy",
    "ExecutionRNGPlan",
    "BatchedRefineOperation",
    "BatchedRefinePlan",
    "BatchedRefineResult",
    "RecallBranchBatch",
    "RecallFormulaBranchBatch",
    "query_recall_branches",
    "run_batched_refine",
]
