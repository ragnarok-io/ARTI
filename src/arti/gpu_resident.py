"""Single-GPU hot-only fixed-bucket tensor residency primitives."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import tempfile
import weakref
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from typing import Callable, ClassVar

import torch
from torch import Tensor, nn

from .formula_attention import ActiveWorkspace
from .formula_fabric import FormulaFabricCompute, FormulaRoutePlan
from .reversible_topology import TopologyFold, TopologyUnFold


class GPUResidentContractError(RuntimeError):
    """Raised when the fixed hot-only execution contract is violated."""


_BOUND_HOT_PAGE_POOL_FACTORY_TOKEN = object()
_CAPTURED_HOT_STEP_FACTORY_TOKEN = object()


@dataclass(frozen=True)
class FixedResidentBucket:
    """Static shape and execution identity for one resident bucket."""

    batch_size: int
    workset_slots: int
    feature_dim: int
    dtype: torch.dtype
    device: torch.device
    refine_steps: int = 1
    _component_reference: ClassVar[str] = "arti/fixed-resident-bucket@1"

    def __post_init__(self) -> None:
        for value, name in (
            (self.batch_size, "batch_size"),
            (self.workset_slots, "workset_slots"),
            (self.feature_dim, "feature_dim"),
            (self.refine_steps, "refine_steps"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise GPUResidentContractError(f"{name} must be a positive integer")
        device = torch.device(self.device)
        object.__setattr__(self, "device", device)
        if device.type != "cuda" or device.index is None:
            raise GPUResidentContractError("FixedResidentBucket requires an indexed CUDA device")
        if self.dtype not in {torch.float16, torch.bfloat16, torch.float32}:
            raise GPUResidentContractError("resident dtype must be float16, bfloat16, or float32")


@dataclass(frozen=True)
class FixedPageRefs:
    """Owned device-side references for one fixed resident workset."""

    logical_slot: Tensor
    page_id: Tensor
    offset: Tensor
    expected_generation: Tensor
    read_mask: Tensor
    write_mask: Tensor
    commit_mask: Tensor
    _component_reference: ClassVar[str] = "arti/fixed-page-refs@1"


@dataclass(frozen=True)
class PointerLayoutReceipt:
    """Pointer and layout snapshot for fixed hot-path buffers."""

    pool_value_ptr: int
    pool_validity_ptr: int
    pool_generation_ptr: int
    pool_version_ptr: int
    workset_input_ptr: int
    workset_output_ptr: int
    commit_buffer_ptr: int
    linear_ref_ptr: int
    shape: tuple[int, int, int]
    stride: tuple[int, int, int]
    dtype: str
    device: str


@dataclass(frozen=True)
class PhysicalCounter:
    """One measured counter or an explicit unavailable result."""

    value: int | float | None
    available: bool
    source: str
    reason: str | None = None

    def __post_init__(self) -> None:
        if self.available != (self.value is not None):
            raise GPUResidentContractError(
                "available physical counters require a value; unavailable counters require None"
            )


@dataclass(frozen=True)
class ResidentLatencyReceipt:
    """CUDA-event timing plus explicitly scoped allocator observations."""

    samples: int
    warmups: int
    p50_ms: float
    p95_ms: float
    p99_ms: float
    allocated_before: int
    allocated_after: int
    reserved_before: int
    reserved_after: int
    max_allocated: int
    pointer_stable: bool
    h2d_bytes: PhysicalCounter
    d2h_bytes: PhysicalCounter
    hbm_read_bytes: PhysicalCounter
    hbm_write_bytes: PhysicalCounter
    kernel_count: PhysicalCounter
    host_sync_count: PhysicalCounter
    allocation_count: PhysicalCounter
    page_miss_count: PhysicalCounter
    eviction_count: PhysicalCounter
    prefetch_count: PhysicalCounter
    _component_reference: ClassVar[str] = "arti/resident-latency-receipt@1"


@dataclass(frozen=True)
class CUDAActivityReceipt:
    """Scoped CUPTI activity counts and transfer bytes from a profiler trace."""

    replays: int
    trace_events: int
    kernel_count: int
    h2d_bytes: int
    d2h_bytes: int
    d2d_bytes: int
    cuda_sync_api_count: int
    trace_export_complete: bool
    trace_fingerprint: str
    dropped_record_count: PhysicalCounter
    source: str = "torch.profiler-cupti-chrome-trace"
    _component_reference: ClassVar[str] = "arti/cuda-activity-receipt@1"


class HotPagePool:
    """GPU-owned page payload with fixed generation and version metadata."""

    _component_reference: ClassVar[str] = "arti/hot-page-pool@1"

    def __init__(
        self,
        value: Tensor,
        *,
        validity: Tensor | None = None,
        generation: Tensor | None = None,
        version: Tensor | None = None,
    ) -> None:
        if not isinstance(value, Tensor) or value.ndim != 3 or value.device.type != "cuda":
            raise GPUResidentContractError("page pool value must be CUDA [P,C,D]")
        if value.layout != torch.strided or not value.is_contiguous():
            raise GPUResidentContractError("page pool value must be contiguous strided storage")
        if value.requires_grad:
            raise GPUResidentContractError("hot-only page pool is inference-only")
        pages, capacity, _dim = value.shape
        expected = (pages, capacity)
        if validity is None:
            validity = torch.ones(expected, dtype=torch.bool, device=value.device)
        if generation is None:
            generation = torch.zeros(expected, dtype=torch.int64, device=value.device)
        if version is None:
            version = torch.zeros(expected, dtype=torch.int64, device=value.device)
        for tensor, name, dtype in (
            (validity, "validity", torch.bool),
            (generation, "generation", torch.int64),
            (version, "version", torch.int64),
        ):
            if (
                not isinstance(tensor, Tensor)
                or tensor.shape != expected
                or tensor.dtype != dtype
                or tensor.device != value.device
                or not tensor.is_contiguous()
            ):
                raise GPUResidentContractError(
                    f"page pool {name} must be contiguous {dtype} {expected} on pool device"
                )
        self.value = value.detach().clone()
        self.validity = validity.detach().clone()
        self.generation = generation.detach().clone()
        self.version = version.detach().clone()
        self._authority_lock = RLock()
        self._lifecycle_state = "open"
        self._bindings: weakref.WeakSet[BoundHotPagePool] = weakref.WeakSet()
        self._authority_event = torch.cuda.Event(blocking=False, interprocess=False)
        self._authority_event_recorded = False

    @property
    def lifecycle_state(self) -> str:
        return self._lifecycle_state

    @property
    def closed(self) -> bool:
        return self._lifecycle_state == "closed"

    def assert_open(self) -> None:
        if self._lifecycle_state != "open":
            raise GPUResidentContractError(
                f"hot page pool is {self._lifecycle_state}"
            )

    def _register_binding(self, binding: BoundHotPagePool) -> None:
        self.assert_open()
        self._bindings.add(binding)

    def _begin_cuda_use(self) -> torch.cuda.Stream:
        """Order one physical use after every earlier use of this pool."""

        self.assert_open()
        stream = torch.cuda.current_stream(self.value.device)
        if self._authority_event_recorded:
            stream.wait_event(self._authority_event)
        return stream

    def _end_cuda_use(self, stream: torch.cuda.Stream) -> None:
        self._authority_event.record(stream)
        self._authority_event_recorded = True

    def _quiesce(self) -> None:
        """Finish every stream use submitted through the pool authority."""

        if self._authority_event_recorded:
            self._authority_event.synchronize()

    def close(self) -> bool:
        """Synchronize, invalidate borrowed bindings, and release pool storage."""

        with self._authority_lock:
            if self._lifecycle_state == "closed":
                return False
            if self._lifecycle_state not in {"open", "closing"}:
                raise GPUResidentContractError(
                    f"hot page pool cannot close from {self._lifecycle_state}"
                )
            self._lifecycle_state = "closing"
            device = self.value.device
            self._quiesce()
            torch.cuda.synchronize(device)
            for binding in tuple(self._bindings):
                binding._close_from_pool()
            self._bindings.clear()
            self.value = torch.empty(0, dtype=self.value.dtype, device=device)
            self.validity = torch.empty(0, dtype=torch.bool, device=device)
            self.generation = torch.empty(0, dtype=torch.int64, device=device)
            self.version = torch.empty(0, dtype=torch.int64, device=device)
            self._lifecycle_state = "closed"
            return True

    def __enter__(self) -> HotPagePool:
        self.assert_open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def authority_lock(self) -> RLock:
        """Shared single-process guard for every binding of this pool."""

        return self._authority_lock

    @property
    def page_count(self) -> int:
        self.assert_open()
        return self.value.shape[0]

    @property
    def page_capacity(self) -> int:
        self.assert_open()
        return self.value.shape[1]

    @property
    def feature_dim(self) -> int:
        self.assert_open()
        return self.value.shape[2]


def _host_owned(value: Tensor, *, name: str, dtype: torch.dtype, shape: tuple[int, ...]) -> Tensor:
    if not isinstance(value, Tensor) or value.dtype != dtype or tuple(value.shape) != shape:
        raise GPUResidentContractError(f"{name} must have dtype {dtype} and shape {shape}")
    if value.device.type != "cpu" or not value.is_contiguous():
        raise GPUResidentContractError(f"{name} must be a contiguous CPU tensor at bind time")
    return value.detach().clone()


class BoundHotPagePool:
    """Preallocated gather/operation/scatter buffers for one immutable bucket."""

    _component_reference: ClassVar[str] = "arti/bound-hot-page-pool@1"

    def __init__(
        self,
        pool: HotPagePool,
        bucket: FixedResidentBucket,
        refs: FixedPageRefs,
        *,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _BOUND_HOT_PAGE_POOL_FACTORY_TOKEN:
            raise GPUResidentContractError(
                "BoundHotPagePool must come from bind_hot_page_pool"
            )
        pool.assert_open()
        if pool.value.device != bucket.device:
            raise GPUResidentContractError("pool and bucket devices differ")
        if pool.value.dtype != bucket.dtype or pool.feature_dim != bucket.feature_dim:
            raise GPUResidentContractError("pool dtype or feature dimension differs from bucket")
        self.pool = pool
        self.bucket = bucket
        shape = (bucket.batch_size, bucket.workset_slots)
        page_id = _host_owned(refs.page_id, name="page_id", dtype=torch.int64, shape=shape)
        offset = _host_owned(refs.offset, name="offset", dtype=torch.int64, shape=shape)
        logical = _host_owned(
            refs.logical_slot,
            name="logical_slot",
            dtype=torch.int64,
            shape=shape,
        )
        expected_generation = _host_owned(
            refs.expected_generation,
            name="expected_generation",
            dtype=torch.int64,
            shape=shape,
        )
        read_mask = _host_owned(refs.read_mask, name="read_mask", dtype=torch.bool, shape=shape)
        write_mask = _host_owned(
            refs.write_mask,
            name="write_mask",
            dtype=torch.bool,
            shape=shape,
        )
        commit_mask = _host_owned(
            refs.commit_mask,
            name="commit_mask",
            dtype=torch.bool,
            shape=shape,
        )
        self.refs = FixedPageRefs(
            logical_slot=logical.clone(),
            page_id=page_id.clone(),
            offset=offset.clone(),
            expected_generation=expected_generation.clone(),
            read_mask=read_mask.clone(),
            write_mask=write_mask.clone(),
            commit_mask=commit_mask.clone(),
        )
        if bool((write_mask & ~read_mask).any()):
            raise GPUResidentContractError("write_mask must be a subset of read_mask")
        if bool((commit_mask & ~write_mask).any()):
            raise GPUResidentContractError("commit_mask must be a subset of write_mask")
        if bool((page_id < 0).any()) or bool((page_id >= pool.page_count).any()):
            raise GPUResidentContractError("page_id is out of range")
        if bool((offset < 0).any()) or bool((offset >= pool.page_capacity).any()):
            raise GPUResidentContractError("offset is out of range")
        linear = page_id * pool.page_capacity + offset
        writable = linear[commit_mask]
        if writable.numel() != torch.unique(writable).numel():
            raise GPUResidentContractError("duplicate writable physical references are forbidden")
        actual_generation = pool.generation.detach().cpu().reshape(-1).index_select(
            0, linear.reshape(-1)
        ).reshape(shape)
        if not torch.equal(actual_generation, expected_generation):
            raise GPUResidentContractError("stale page generation")

        device = bucket.device
        self.logical_slot = logical.to(device)
        self.linear_ref = linear.reshape(-1).to(device)
        self.expected_generation = expected_generation.to(device)
        self.read_mask = read_mask.to(device)
        self.write_mask = write_mask.to(device)
        self.commit_mask = commit_mask.to(device)
        commit_positions = torch.nonzero(commit_mask.reshape(-1), as_tuple=False).reshape(-1)
        self.commit_positions = commit_positions.to(device)
        self.commit_linear_ref = writable.to(device)
        self.commit_ones = torch.ones_like(self.commit_linear_ref, dtype=torch.int64)
        value_shape = (bucket.batch_size, bucket.workset_slots, bucket.feature_dim)
        self.workset_input = torch.empty(value_shape, dtype=bucket.dtype, device=device)
        self.workset_output = torch.empty_like(self.workset_input)
        self.commit_buffer = torch.empty(
            (writable.numel(), bucket.feature_dim),
            dtype=bucket.dtype,
            device=device,
        )
        self.gathered_validity = torch.empty(shape, dtype=torch.bool, device=device)
        self.gathered_generation = torch.empty(
            shape,
            dtype=torch.int64,
            device=device,
        )
        self.effective_validity = torch.empty(shape, dtype=torch.bool, device=device)
        self.intervened = torch.empty(shape, dtype=torch.bool, device=device)
        self._lifecycle_state = "open"
        self._captures: weakref.WeakSet[CapturedHotStep] = weakref.WeakSet()
        pool._register_binding(self)

    @property
    def lifecycle_state(self) -> str:
        return self._lifecycle_state

    @property
    def closed(self) -> bool:
        return self._lifecycle_state == "closed"

    def assert_open(self) -> None:
        if self._lifecycle_state != "open":
            raise GPUResidentContractError(
                f"bound hot page pool is {self._lifecycle_state}"
            )
        self.pool.assert_open()

    def _register_capture(self, captured: CapturedHotStep) -> None:
        self.assert_open()
        self._captures.add(captured)

    def _release_buffers(self) -> None:
        for name in (
            "logical_slot",
            "linear_ref",
            "expected_generation",
            "read_mask",
            "write_mask",
            "commit_mask",
            "commit_positions",
            "commit_linear_ref",
            "commit_ones",
            "workset_input",
            "workset_output",
            "commit_buffer",
            "gathered_validity",
            "gathered_generation",
            "effective_validity",
            "intervened",
        ):
            value = getattr(self, name)
            setattr(
                self,
                name,
                torch.empty(0, dtype=value.dtype, device=value.device),
            )

    def _finish_close(self) -> None:
        for captured in tuple(self._captures):
            captured._close_from_bound()
        self._captures.clear()
        self._release_buffers()
        self._lifecycle_state = "closed"

    def _close_from_pool(self) -> None:
        if self._lifecycle_state == "closed":
            return
        self._lifecycle_state = "closing"
        self._finish_close()

    def close(self, *, close_pool: bool = False) -> bool:
        """Release this binding; the shared pool remains open by default."""

        with self.authority_lock:
            if self._lifecycle_state == "closed":
                if close_pool:
                    self.pool.close()
                return False
            if self._lifecycle_state not in {"open", "closing"}:
                raise GPUResidentContractError(
                    f"bound hot page pool cannot close from {self._lifecycle_state}"
                )
            self._lifecycle_state = "closing"
            self.pool._quiesce()
            torch.cuda.synchronize(self.bucket.device)
            self._finish_close()
        if close_pool:
            self.pool.close()
        return True

    def __enter__(self) -> BoundHotPagePool:
        self.assert_open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def authority_lock(self) -> RLock:
        """Shared single-process guard for check-and-commit authority."""

        return self.pool.authority_lock

    def pointer_layout_receipt(self) -> PointerLayoutReceipt:
        with self.authority_lock:
            self.assert_open()
            return PointerLayoutReceipt(
                pool_value_ptr=self.pool.value.data_ptr(),
                pool_validity_ptr=self.pool.validity.data_ptr(),
                pool_generation_ptr=self.pool.generation.data_ptr(),
                pool_version_ptr=self.pool.version.data_ptr(),
                workset_input_ptr=self.workset_input.data_ptr(),
                workset_output_ptr=self.workset_output.data_ptr(),
                commit_buffer_ptr=self.commit_buffer.data_ptr(),
                linear_ref_ptr=self.linear_ref.data_ptr(),
                shape=tuple(self.workset_input.shape),
                stride=tuple(self.workset_input.stride()),
                dtype=str(self.workset_input.dtype),
                device=str(self.workset_input.device),
            )

    def synchronize(self) -> None:
        """Wait for every physical pool use submitted through this authority."""

        with self.authority_lock:
            self.assert_open()
            self.pool._quiesce()

    def _execute(self, operation: nn.Module, *, commit: bool) -> Tensor:
        self.assert_open()
        flat_pool = self.pool.value.view(-1, self.pool.feature_dim)
        torch.index_select(
            flat_pool,
            0,
            self.linear_ref,
            out=self.workset_input.view(-1, self.pool.feature_dim),
        )
        torch.index_select(
            self.pool.validity.view(-1),
            0,
            self.linear_ref,
            out=self.gathered_validity.view(-1),
        )
        torch.index_select(
            self.pool.generation.view(-1),
            0,
            self.linear_ref,
            out=self.gathered_generation.view(-1),
        )
        torch._assert_async(
            (self.gathered_generation == self.expected_generation).all(),
            "resident page generation changed after binding",
        )
        torch.logical_and(
            self.gathered_validity,
            self.read_mask,
            out=self.effective_validity,
        )
        torch.logical_and(
            self.effective_validity,
            self.write_mask,
            out=self.intervened,
        )
        self.workset_input.masked_fill_(~self.effective_validity.unsqueeze(-1), 0)
        result = operation(
            self.workset_input,
            self.effective_validity,
            self.effective_validity,
            self.intervened,
        )
        if not isinstance(result, Tensor) or result.shape != self.workset_output.shape:
            raise GPUResidentContractError("resident operation must return the fixed value shape")
        self.workset_output.copy_(result)
        if commit and self.commit_positions.numel() > 0:
            torch._assert_async(
                (~self.commit_mask | self.gathered_validity).all(),
                "resident commit target became invalid after binding",
            )
            torch.index_select(
                self.workset_output.view(-1, self.pool.feature_dim),
                0,
                self.commit_positions,
                out=self.commit_buffer,
            )
            flat_pool.index_copy_(0, self.commit_linear_ref, self.commit_buffer)
            self.pool.version.view(-1).index_add_(
                0,
                self.commit_linear_ref,
                self.commit_ones,
            )
        return self.workset_output

    def _validate_operation(self, operation: nn.Module) -> None:
        self.assert_open()
        if not isinstance(operation, nn.Module):
            raise GPUResidentContractError("resident operation must be an nn.Module")
        refine_steps = getattr(operation, "resident_refine_steps", None)
        if refine_steps is not None and refine_steps != self.bucket.refine_steps:
            raise GPUResidentContractError(
                "resident operation refine_steps differ from the fixed bucket"
            )

    def eager_step(
        self,
        operation: nn.Module,
        *,
        commit: bool = True,
        copy_output: bool = False,
    ) -> Tensor:
        """Run one eager fixed-bucket step and optionally return owned output."""

        with self.authority_lock:
            self._validate_operation(operation)
            stream = self.pool._begin_cuda_use()
            result = self._execute(operation, commit=commit)
            if copy_output:
                result = result.clone()
            self.pool._end_cuda_use(stream)
            if commit:
                self.pool._quiesce()
            return result

    def capture(self, operation: nn.Module, *, commit: bool = True) -> CapturedHotStep:
        """Capture one fixed-address hot step after caller-owned warmup."""

        with self.authority_lock:
            self._validate_operation(operation)
            if commit:
                raise GPUResidentContractError(
                    "commit-capable CUDA Graph capture is unsupported because host "
                    "authority cannot be released before asynchronous replay completes"
                )
            self.pool._quiesce()
            stream = self.pool._begin_cuda_use()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._execute(operation, commit=False)
            self.pool._end_cuda_use(stream)
            captured = CapturedHotStep(
                self,
                graph,
                commit=False,
                _factory_token=_CAPTURED_HOT_STEP_FACTORY_TOKEN,
            )
            self._register_capture(captured)
            return captured


class CapturedHotStep:
    """Fixed-address CUDA Graph handle for one BoundHotPagePool."""

    _component_reference: ClassVar[str] = "arti/captured-hot-step@1"

    def __init__(
        self,
        bound: BoundHotPagePool,
        graph: torch.cuda.CUDAGraph,
        *,
        commit: bool,
        _factory_token: object | None = None,
    ) -> None:
        if _factory_token is not _CAPTURED_HOT_STEP_FACTORY_TOKEN:
            raise GPUResidentContractError(
                "CapturedHotStep must come from BoundHotPagePool.capture"
            )
        self.bound = bound
        self.graph = graph
        self.commit = commit
        self._lifecycle_state = "open"

    @property
    def lifecycle_state(self) -> str:
        return self._lifecycle_state

    @property
    def closed(self) -> bool:
        return self._lifecycle_state == "closed"

    def assert_open(self) -> None:
        if self._lifecycle_state != "open":
            raise GPUResidentContractError(
                f"captured hot step is {self._lifecycle_state}"
            )
        self.bound.assert_open()

    def _close_from_bound(self) -> None:
        if self._lifecycle_state == "closed":
            return
        self._lifecycle_state = "closing"
        self.graph = None
        self._lifecycle_state = "closed"

    def close(self) -> bool:
        with self.bound.authority_lock:
            if self._lifecycle_state == "closed":
                return False
            if self._lifecycle_state not in {"open", "closing"}:
                raise GPUResidentContractError(
                    f"captured hot step cannot close from {self._lifecycle_state}"
                )
            self._lifecycle_state = "closing"
            self.bound.pool._quiesce()
            torch.cuda.synchronize(self.bound.bucket.device)
            self.graph = None
            self._lifecycle_state = "closed"
            return True

    def __enter__(self) -> CapturedHotStep:
        self.assert_open()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def replay(self, *, copy_output: bool = False) -> Tensor:
        with self.bound.authority_lock:
            self.assert_open()
            assert self.graph is not None
            stream = self.bound.pool._begin_cuda_use()
            self.graph.replay()
            result = self.bound.workset_output
            if copy_output:
                result = result.clone()
            self.bound.pool._end_cuda_use(stream)
            return result


def _resident_route_view(
    weights: Tensor,
    valid_mask: Tensor,
    fire_mask: Tensor,
    commit_mask: Tensor,
    estimator: str,
) -> FormulaRoutePlan:
    """Bind a validated FormulaRoutePlan view without cloning resident buffers."""

    route = object.__new__(FormulaRoutePlan)
    object.__setattr__(route, "weights", weights)
    object.__setattr__(route, "valid_mask", valid_mask)
    object.__setattr__(route, "fire_mask", fire_mask)
    object.__setattr__(route, "commit_mask", commit_mask)
    object.__setattr__(route, "estimator", estimator)
    return route


def _validate_resident_route(
    weights: Tensor,
    valid_mask: Tensor,
    fire_mask: Tensor,
    commit_mask: Tensor,
    estimator: str,
) -> None:
    # FormulaRoutePlan owns clones, so this validates loaded state without
    # replacing the resident buffers used by the hot path.
    FormulaRoutePlan(weights, valid_mask, fire_mask, commit_mask, estimator)


def _reindex_resident_inputs(
    operation: object,
    batch_index: Tensor,
) -> tuple[FormulaRoutePlan, Tensor | None]:
    route = operation.route
    batch = route.weights.shape[0]
    if (
        not isinstance(batch_index, Tensor)
        or batch_index.dtype != torch.long
        or batch_index.ndim != 1
        or batch_index.shape[0] != batch
        or batch_index.device != route.weights.device
    ):
        raise GPUResidentContractError(
            "resident batch_index must be torch.long [B] on the route device"
        )
    expected = torch.arange(batch, device=batch_index.device)
    condition = torch.sort(batch_index).values == expected
    if condition.device.type == "cpu":
        if not bool(condition.all()):
            raise GPUResidentContractError(
                "resident batch_index must be a permutation of [0,B)"
            )
    else:
        torch._assert_async(
            condition.all(),
            "resident batch_index must be a permutation of [0,B)",
        )
    reindexed = _resident_route_view(
        route.weights.index_select(0, batch_index).contiguous(),
        route.valid_mask.index_select(0, batch_index).contiguous(),
        route.fire_mask.index_select(0, batch_index).contiguous(),
        route.commit_mask.index_select(0, batch_index).contiguous(),
        route.estimator,
    )
    factors = operation.factors
    if factors is not None:
        if factors.ndim < 1 or factors.shape[0] != batch:
            raise GPUResidentContractError(
                "resident factors must have the same batch axis as the route"
            )
        factors = factors.index_select(0, batch_index).contiguous()
    return reindexed, factors


def _select_resident_inputs(
    operation: object,
    batch_index: Tensor,
) -> tuple[FormulaRoutePlan, Tensor | None]:
    """Select a packed subset without weakening indexed permutation semantics."""

    route = operation.route
    batch = route.weights.shape[0]
    if (
        not isinstance(batch_index, Tensor)
        or batch_index.dtype != torch.long
        or batch_index.ndim != 1
        or batch_index.device != route.weights.device
    ):
        raise GPUResidentContractError(
            "resident packed index must be torch.long [P] on the route device"
        )
    in_range = (batch_index >= 0) & (batch_index < batch)
    if in_range.device.type == "cpu":
        if not bool(in_range.all()):
            raise GPUResidentContractError("resident packed index is out of range")
    else:
        torch._assert_async(
            in_range.all(),
            "resident packed index is out of range",
        )
    selected = _resident_route_view(
        route.weights.index_select(0, batch_index).contiguous(),
        route.valid_mask.index_select(0, batch_index).contiguous(),
        route.fire_mask.index_select(0, batch_index).contiguous(),
        route.commit_mask.index_select(0, batch_index).contiguous(),
        route.estimator,
    )
    factors = operation.factors
    if factors is not None:
        if factors.ndim < 1 or factors.shape[0] != batch:
            raise GPUResidentContractError(
                "resident factors must have the same batch axis as the route"
            )
        factors = factors.index_select(0, batch_index).contiguous()
    return selected, factors


class FormulaResidentOperation(nn.Module):
    """Thin adapter from fixed resident buffers to existing FormulaFabricCompute."""

    _component_reference: ClassVar[str] = "arti/formula-resident-operation@1"

    def __init__(
        self,
        compute: FormulaFabricCompute,
        route: FormulaRoutePlan,
        *,
        refine_steps: int = 1,
        factors: Tensor | None = None,
    ) -> None:
        super().__init__()
        if route.estimator != "hard":
            raise GPUResidentContractError("resident Formula route must be hard")
        if isinstance(refine_steps, bool) or not isinstance(refine_steps, int) or refine_steps <= 0:
            raise GPUResidentContractError("refine_steps must be positive")
        self.compute = compute
        self.refine_steps = refine_steps
        self._route_estimator = route.estimator
        self.register_buffer("_route_weights", route.weights)
        self.register_buffer("_route_valid_mask", route.valid_mask)
        self.register_buffer("_route_fire_mask", route.fire_mask)
        self.register_buffer("_route_commit_mask", route.commit_mask)
        self.register_buffer(
            "_resident_factors",
            None if factors is None else factors.clone(),
        )
        self._bind_resident_route()
        self.register_load_state_dict_post_hook(self._after_state_load)

    @property
    def route(self) -> FormulaRoutePlan:
        return self._route

    @property
    def factors(self) -> Tensor | None:
        return self._resident_factors

    def _bind_resident_route(self) -> None:
        self._route = _resident_route_view(
            self._route_weights,
            self._route_valid_mask,
            self._route_fire_mask,
            self._route_commit_mask,
            self._route_estimator,
        )

    def _after_state_load(self, _module: nn.Module, _incompatible_keys: object) -> None:
        _validate_resident_route(
            self._route_weights,
            self._route_valid_mask,
            self._route_fire_mask,
            self._route_commit_mask,
            self._route_estimator,
        )
        self._bind_resident_route()

    def _apply(self, fn: object, recurse: bool = True) -> FormulaResidentOperation:
        result = super()._apply(fn, recurse=recurse)
        self._bind_resident_route()
        return result

    @property
    def resident_refine_steps(self) -> int:
        return self.refine_steps

    def indexed_inputs(
        self, batch_index: Tensor
    ) -> tuple[FormulaRoutePlan, Tensor | None]:
        return _reindex_resident_inputs(self, batch_index)

    def selected_inputs(
        self, batch_index: Tensor
    ) -> tuple[FormulaRoutePlan, Tensor | None]:
        return _select_resident_inputs(self, batch_index)

    def forward(
        self,
        value: Tensor,
        validity: Tensor,
        exposed: Tensor,
        intervened: Tensor,
    ) -> Tensor:
        return self._forward_with_route(
            value,
            validity,
            exposed,
            intervened,
            route=self.route,
            factors=self.factors,
        )

    def forward_indexed(
        self,
        value: Tensor,
        validity: Tensor,
        exposed: Tensor,
        intervened: Tensor,
        batch_index: Tensor,
    ) -> Tensor:
        """Execute a branch-reindexed view without mutating resident buffers."""

        route, factors = _reindex_resident_inputs(self, batch_index)
        return self._forward_with_route(
            value,
            validity,
            exposed,
            intervened,
            route=route,
            factors=factors,
        )

    def _forward_with_route(
        self,
        value: Tensor,
        validity: Tensor,
        exposed: Tensor,
        intervened: Tensor,
        *,
        route: FormulaRoutePlan,
        factors: Tensor | None,
        trace_callback: Callable[[tuple[object, ...], Tensor | None, int], None]
        | None = None,
    ) -> Tensor:
        workspace = ActiveWorkspace(value, validity, exposed, intervened)
        traces: list[object] = []
        for _step in range(self.refine_steps):
            computed = self.compute(
                workspace,
                factors,
                formula_route=route,
                return_info=trace_callback is not None,
            )
            if trace_callback is None:
                workspace = computed
            else:
                workspace, trace = computed
                traces.append(trace)
        if trace_callback is not None:
            trace_callback(tuple(traces), None, 0)
        return workspace.value


class TopologyFormulaResidentOperation(nn.Module):
    """Compose existing Fold@2, Formula Fabric, and UnFold@2 on a fixed workset."""

    _component_reference: ClassVar[str] = "arti/topology-formula-resident-operation@1"

    def __init__(
        self,
        fold: TopologyFold,
        unfold: TopologyUnFold,
        compute: FormulaFabricCompute,
        route: FormulaRoutePlan,
        *,
        refine_steps: int = 1,
        factors: Tensor | None = None,
    ) -> None:
        super().__init__()
        if route.estimator != "hard":
            raise GPUResidentContractError("resident Formula route must be hard")
        if isinstance(refine_steps, bool) or not isinstance(refine_steps, int) or refine_steps <= 0:
            raise GPUResidentContractError("refine_steps must be positive")
        if fold.topology.active_count != unfold.inverse_contract.active_count:
            raise GPUResidentContractError("Fold@2 and UnFold@2 active_count differ")
        if fold.topology.axis != unfold.inverse_contract.axis:
            raise GPUResidentContractError("Fold@2 and UnFold@2 axes differ")
        self.fold = fold
        self.unfold = unfold
        self.compute = compute
        self.refine_steps = refine_steps
        self._route_estimator = route.estimator
        self.register_buffer("_route_weights", route.weights)
        self.register_buffer("_route_valid_mask", route.valid_mask)
        self.register_buffer("_route_fire_mask", route.fire_mask)
        self.register_buffer("_route_commit_mask", route.commit_mask)
        self.register_buffer(
            "_resident_factors",
            None if factors is None else factors.clone(),
        )
        self._bind_resident_route()
        self.register_load_state_dict_post_hook(self._after_state_load)

    @property
    def route(self) -> FormulaRoutePlan:
        return self._route

    @property
    def factors(self) -> Tensor | None:
        return self._resident_factors

    def _bind_resident_route(self) -> None:
        self._route = _resident_route_view(
            self._route_weights,
            self._route_valid_mask,
            self._route_fire_mask,
            self._route_commit_mask,
            self._route_estimator,
        )

    def _after_state_load(self, _module: nn.Module, _incompatible_keys: object) -> None:
        _validate_resident_route(
            self._route_weights,
            self._route_valid_mask,
            self._route_fire_mask,
            self._route_commit_mask,
            self._route_estimator,
        )
        self._bind_resident_route()

    def _apply(
        self,
        fn: object,
        recurse: bool = True,
    ) -> TopologyFormulaResidentOperation:
        result = super()._apply(fn, recurse=recurse)
        self._bind_resident_route()
        return result

    @property
    def resident_refine_steps(self) -> int:
        return self.refine_steps

    def indexed_inputs(
        self, batch_index: Tensor
    ) -> tuple[FormulaRoutePlan, Tensor | None]:
        return _reindex_resident_inputs(self, batch_index)

    def selected_inputs(
        self, batch_index: Tensor
    ) -> tuple[FormulaRoutePlan, Tensor | None]:
        return _select_resident_inputs(self, batch_index)

    def forward(
        self,
        value: Tensor,
        validity: Tensor,
        exposed: Tensor,
        intervened: Tensor,
    ) -> Tensor:
        return self._forward_with_route(
            value,
            validity,
            exposed,
            intervened,
            route=self.route,
            factors=self.factors,
        )

    def forward_indexed(
        self,
        value: Tensor,
        validity: Tensor,
        exposed: Tensor,
        intervened: Tensor,
        batch_index: Tensor,
    ) -> Tensor:
        """Execute a branch-reindexed topology view over resident inputs."""

        route, factors = _reindex_resident_inputs(self, batch_index)
        return self._forward_with_route(
            value,
            validity,
            exposed,
            intervened,
            route=route,
            factors=factors,
        )

    def _forward_with_route(
        self,
        value: Tensor,
        validity: Tensor,
        exposed: Tensor,
        intervened: Tensor,
        *,
        route: FormulaRoutePlan,
        factors: Tensor | None,
        trace_callback: Callable[[tuple[object, ...], Tensor | None, int], None]
        | None = None,
    ) -> Tensor:
        state = self.fold(value, validity)
        permutation = state.record._trusted_permutation()
        active_index = permutation[
            ..., : state.record.active_count
        ]
        active_exposed = torch.gather(exposed, -1, active_index)
        active_intervened = torch.gather(intervened, -1, active_index)
        workspace = ActiveWorkspace(
            state.active,
            state.active_mask,
            active_exposed,
            active_intervened,
        )
        traces: list[object] = []
        for _step in range(self.refine_steps):
            computed = self.compute(
                workspace,
                factors,
                formula_route=route,
                return_info=trace_callback is not None,
            )
            if trace_callback is None:
                workspace = computed
            else:
                workspace, trace = computed
                traces.append(trace)
        result = self.unfold(state.replace(active=workspace.value)).value
        if trace_callback is not None:
            trace_callback(tuple(traces), permutation, state.record.active_count)
        return result


def bind_hot_page_pool(
    pool: HotPagePool,
    bucket: FixedResidentBucket,
    refs: FixedPageRefs,
) -> BoundHotPagePool:
    """Bind and preallocate one all-hot fixed bucket."""

    with pool.authority_lock:
        return BoundHotPagePool(
            pool,
            bucket,
            refs,
            _factory_token=_BOUND_HOT_PAGE_POOL_FACTORY_TOKEN,
        )


def measure_captured_replays(
    captured: CapturedHotStep,
    *,
    warmups: int = 10,
    samples: int = 50,
) -> ResidentLatencyReceipt:
    """Measure graph replay latency while leaving unavailable counters unknown."""

    if warmups < 1 or samples < 2:
        raise GPUResidentContractError("measurement requires warmups >= 1 and samples >= 2")
    device = captured.bound.bucket.device
    for _ in range(warmups):
        captured.replay()
    torch.cuda.synchronize(device)
    before_pointer = captured.bound.pointer_layout_receipt()
    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    torch.cuda.reset_peak_memory_stats(device)
    timings: list[float] = []
    events = [
        (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        for _ in range(samples)
    ]
    for start, end in events:
        start.record()
        captured.replay()
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end)))
    allocated_after = torch.cuda.memory_allocated(device)
    reserved_after = torch.cuda.memory_reserved(device)
    after_pointer = captured.bound.pointer_layout_receipt()
    ordered = sorted(timings)

    def percentile(q: float) -> float:
        index = (len(ordered) - 1) * q
        low = math.floor(index)
        high = math.ceil(index)
        if low == high:
            return ordered[low]
        return ordered[low] * (high - index) + ordered[high] * (index - low)

    def unavailable(reason: str) -> PhysicalCounter:
        return PhysicalCounter(None, False, "not-collected", reason)

    return ResidentLatencyReceipt(
        samples=samples,
        warmups=warmups,
        p50_ms=statistics.median(timings),
        p95_ms=percentile(0.95),
        p99_ms=percentile(0.99),
        allocated_before=allocated_before,
        allocated_after=allocated_after,
        reserved_before=reserved_before,
        reserved_after=reserved_after,
        max_allocated=torch.cuda.max_memory_allocated(device),
        pointer_stable=before_pointer == after_pointer,
        h2d_bytes=unavailable("requires complete CUPTI/Nsight transfer activity"),
        d2h_bytes=unavailable("requires complete CUPTI/Nsight transfer activity"),
        hbm_read_bytes=unavailable("requires an available hardware counter"),
        hbm_write_bytes=unavailable("requires an available hardware counter"),
        kernel_count=unavailable("requires complete CUPTI/Nsight kernel activity"),
        host_sync_count=PhysicalCounter(
            samples,
            True,
            "measurement-policy",
            "one end-event synchronization per timed sample; outside captured hot step",
        ),
        allocation_count=unavailable(
            "PyTorch allocator byte snapshots do not expose allocation/free call counts"
        ),
        page_miss_count=PhysicalCounter(
            0,
            True,
            "hot-only-contract",
            "this runtime has no cold-page lookup path",
        ),
        eviction_count=PhysicalCounter(
            0,
            True,
            "hot-only-contract",
            "this runtime has no eviction path",
        ),
        prefetch_count=PhysicalCounter(
            0,
            True,
            "hot-only-contract",
            "this runtime has no prefetch path",
        ),
    )


def profile_captured_cuda_activity(
    captured: CapturedHotStep,
    *,
    replays: int = 10,
) -> CUDAActivityReceipt:
    """Collect scoped CUDA kernel and memcpy activity for graph replay."""

    if isinstance(replays, bool) or not isinstance(replays, int) or replays <= 0:
        raise GPUResidentContractError("replays must be a positive integer")
    def replay() -> None:
        for _ in range(replays):
            captured.replay()

    return profile_cuda_callable_activity(
        replay,
        device=captured.bound.bucket.device,
        invocations=replays,
    )


def profile_cuda_callable_activity(
    operation: Callable[[], object],
    *,
    device: torch.device | str,
    invocations: int = 1,
) -> CUDAActivityReceipt:
    """Profile one bounded callable without inventing unavailable counters."""

    if not callable(operation):
        raise TypeError("operation must be callable")
    if isinstance(invocations, bool) or not isinstance(invocations, int) or invocations <= 0:
        raise GPUResidentContractError("invocations must be a positive integer")
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda" or resolved_device.index is None:
        raise GPUResidentContractError("CUDA activity profiling requires an indexed device")

    from torch.profiler import ProfilerActivity, profile

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
    ) as profiler:
        operation()
        torch.cuda.synchronize(resolved_device)

    temporary = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
    trace_path = Path(temporary.name)
    temporary.close()
    try:
        profiler.export_chrome_trace(str(trace_path))
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
    finally:
        trace_path.unlink(missing_ok=True)
    events = payload.get("traceEvents")
    if not isinstance(events, list):
        raise GPUResidentContractError("profiler trace does not contain traceEvents")
    trace_fingerprint = hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("ascii")
    ).hexdigest()
    h2d_bytes = 0
    d2h_bytes = 0
    d2d_bytes = 0
    kernel_count = 0
    sync_count = 0
    for event in events:
        if not isinstance(event, dict):
            continue
        category = str(event.get("cat", ""))
        name = str(event.get("name", ""))
        if category in {"kernel", "gpu_kernel"}:
            kernel_count += 1
        if category == "gpu_memcpy":
            arguments = event.get("args", {})
            byte_count = (
                int(arguments.get("bytes", 0))
                if isinstance(arguments, dict)
                else 0
            )
            if "HtoD" in name:
                h2d_bytes += byte_count
            elif "DtoH" in name:
                d2h_bytes += byte_count
            elif "DtoD" in name:
                d2d_bytes += byte_count
        if category == "cuda_runtime" and name in {
            "cudaDeviceSynchronize",
            "cudaStreamSynchronize",
            "cudaEventSynchronize",
        }:
            sync_count += 1
    return CUDAActivityReceipt(
        replays=invocations,
        trace_events=len(events),
        kernel_count=kernel_count,
        h2d_bytes=h2d_bytes,
        d2h_bytes=d2h_bytes,
        d2d_bytes=d2d_bytes,
        cuda_sync_api_count=sync_count,
        trace_export_complete=True,
        trace_fingerprint=trace_fingerprint,
        dropped_record_count=PhysicalCounter(
            None,
            False,
            "torch.profiler-cupti-chrome-trace",
            "the exported trace does not report CUPTI dropped-record counts",
        ),
    )


__all__ = [
    "BoundHotPagePool",
    "CUDAActivityReceipt",
    "CapturedHotStep",
    "FixedPageRefs",
    "FixedResidentBucket",
    "FormulaResidentOperation",
    "GPUResidentContractError",
    "HotPagePool",
    "PhysicalCounter",
    "PointerLayoutReceipt",
    "ResidentLatencyReceipt",
    "TopologyFormulaResidentOperation",
    "bind_hot_page_pool",
    "measure_captured_replays",
    "profile_captured_cuda_activity",
    "profile_cuda_callable_activity",
]
