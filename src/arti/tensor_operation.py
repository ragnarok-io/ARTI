"""Stable mounted tensor port and exact shared-canvas edit semantics."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import ClassVar, Literal, Sequence

import torch
from torch import Tensor, nn

from .component_registry import canonical_contract_reference
from .formula_learning import FormulaOperandBank, hard_formula_route


BackingSource = Literal["default", "external"]
TensorOperationExecutor = Literal["static_masked", "early_break"]


def _require_tensor(condition: Tensor, message: str) -> None:
    if torch.compiler.is_compiling() or condition.device.type != "cpu":
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


class _HardForwardRelaxedBackward(torch.autograd.Function):
    """Return the hard tensor bit-exactly while differentiating the relaxation."""

    @staticmethod
    def forward(_ctx: object, hard: Tensor, _relaxed: Tensor) -> Tensor:
        return hard.clone()

    @staticmethod
    def backward(_ctx: object, gradient: Tensor) -> tuple[None, Tensor]:
        return None, gradient


class CanvasSource(IntEnum):
    """Logical source recorded for each shared-canvas position."""

    EMPTY = -1
    WORLD = 0
    BACKING = 1


class EditOperation(IntEnum):
    """Hard field operations over one backing snapshot."""

    KEEP = 0
    COPY = 1
    ERASE = 2


@dataclass(frozen=True)
class PortSpec:
    """Immutable logical-tensor and folded-view contract for one tensor port."""

    _component_reference: ClassVar[str] = "arti/operable-tensor-port-spec@3"

    canvas_tokens: int
    tensor_shape: tuple[int, ...]
    dim: int
    tensor_to_canvas: tuple[int, ...]
    folded_tensor_coordinates: tuple[tuple[int, ...], ...] | None = None
    dtype: torch.dtype = torch.float32
    empty_value: float = 0.0
    default_visible: bool = False
    coordinate_frame: str = "flat"

    def __post_init__(self) -> None:
        for name in ("canvas_tokens", "dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        shape = tuple(self.tensor_shape)
        if not shape or any(
            isinstance(size, bool) or not isinstance(size, int) or size <= 0
            for size in shape
        ):
            raise ValueError("tensor_shape must contain positive integer dimensions")
        object.__setattr__(self, "tensor_shape", shape)
        tensor_to_canvas = tuple(self.tensor_to_canvas)
        object.__setattr__(self, "tensor_to_canvas", tensor_to_canvas)
        folded = (
            tuple(self.unravel_offset(index) for index in range(len(tensor_to_canvas)))
            if self.folded_tensor_coordinates is None
            else tuple(tuple(coordinate) for coordinate in self.folded_tensor_coordinates)
        )
        object.__setattr__(self, "folded_tensor_coordinates", folded)
        if len(tensor_to_canvas) != len(folded):
            raise ValueError(
                "tensor_to_canvas and folded_tensor_coordinates must contain the same number of entries"
            )
        if len(set(tensor_to_canvas)) != len(tensor_to_canvas):
            raise ValueError("tensor_to_canvas must be injective")
        if any(index < 0 or index >= self.canvas_tokens for index in tensor_to_canvas):
            raise ValueError("tensor_to_canvas contains an out-of-range canvas index")
        if len(set(folded)) != len(folded):
            raise ValueError("folded_tensor_coordinates must be injective")
        for coordinate in folded:
            self.ravel_coordinate(coordinate)
        if not isinstance(self.dtype, torch.dtype) or not torch.empty((), dtype=self.dtype).is_floating_point():
            raise TypeError("dtype must be a floating-point torch dtype")
        if isinstance(self.empty_value, bool) or not isinstance(self.empty_value, (int, float)):
            raise TypeError("empty_value must be a finite number")
        if not torch.isfinite(torch.tensor(float(self.empty_value))):
            raise ValueError("empty_value must be finite")
        if not isinstance(self.default_visible, bool):
            raise TypeError("default_visible must be boolean")
        if not isinstance(self.coordinate_frame, str) or not self.coordinate_frame:
            raise ValueError("coordinate_frame must be a non-empty string")

    @property
    def canvas_shape(self) -> tuple[int, int]:
        return self.canvas_tokens, self.dim

    @property
    def element_count(self) -> int:
        return math.prod(self.tensor_shape)

    @property
    def backing_shape(self) -> tuple[int, ...]:
        return (*self.tensor_shape, self.dim)

    @property
    def backing_mask_shape(self) -> tuple[int, ...]:
        return self.tensor_shape

    @property
    def folded_tensor_offsets(self) -> tuple[int, ...]:
        assert self.folded_tensor_coordinates is not None
        return tuple(self.ravel_coordinate(value) for value in self.folded_tensor_coordinates)

    def ravel_coordinate(self, coordinate: Sequence[int]) -> int:
        resolved = tuple(coordinate)
        if len(resolved) != len(self.tensor_shape):
            raise ValueError(
                f"tensor coordinate must have rank {len(self.tensor_shape)}, got {len(resolved)}"
            )
        offset = 0
        for axis, (index, size) in enumerate(zip(resolved, self.tensor_shape, strict=True)):
            if isinstance(index, bool) or not isinstance(index, int):
                raise TypeError("tensor coordinates must contain integers")
            if index < 0 or index >= size:
                raise ValueError(f"tensor coordinate axis {axis} is out of range")
            offset = offset * size + index
        return offset

    def unravel_offset(self, offset: int) -> tuple[int, ...]:
        if isinstance(offset, bool) or not isinstance(offset, int):
            raise TypeError("tensor offset must be an integer")
        if offset < 0 or offset >= self.element_count:
            raise ValueError("tensor offset is out of range")
        result = [0] * len(self.tensor_shape)
        remainder = offset
        for axis in range(len(self.tensor_shape) - 1, -1, -1):
            size = self.tensor_shape[axis]
            result[axis] = remainder % size
            remainder //= size
        return tuple(result)

    def region_offsets(self, *axes: slice | int) -> tuple[int, ...]:
        """Compile one logical tensor region into row-major element offsets."""

        if len(axes) != len(self.tensor_shape):
            raise ValueError(f"region must provide {len(self.tensor_shape)} axes")
        coordinates: list[tuple[int, ...]] = [()]
        for axis, (selector, size) in enumerate(zip(axes, self.tensor_shape, strict=True)):
            if isinstance(selector, int) and not isinstance(selector, bool):
                if selector < 0 or selector >= size:
                    raise ValueError(f"region axis {axis} is out of range")
                selected = (selector,)
            elif isinstance(selector, slice):
                if selector.step == 0:
                    raise ValueError("region slices cannot have a zero step")
                selected = tuple(range(*selector.indices(size)))
            else:
                raise TypeError("region axes must be integers or slices")
            coordinates = [prefix + (index,) for prefix in coordinates for index in selected]
        return tuple(self.ravel_coordinate(value) for value in coordinates)

    def flatten_value(self, value: Tensor) -> Tensor:
        return value.reshape(value.shape[0], self.element_count, self.dim)

    def flatten_mask(self, mask: Tensor) -> Tensor:
        return mask.reshape(mask.shape[0], self.element_count)

    def restore_value(self, value: Tensor) -> Tensor:
        return value.reshape(value.shape[0], *self.backing_shape)

    def restore_mask(self, mask: Tensor) -> Tensor:
        return mask.reshape(mask.shape[0], *self.backing_mask_shape)

    def validate_backing(self, value: Tensor, mask: Tensor, *, batch_size: int) -> None:
        expected_value = (batch_size, *self.backing_shape)
        if not isinstance(value, Tensor) or tuple(value.shape) != expected_value:
            raise ValueError(
                f"backing must have shape {list(expected_value)}"
            )
        if value.dtype != self.dtype:
            raise TypeError(f"backing dtype must be {self.dtype}")
        if not value.is_contiguous():
            raise ValueError("backing must be contiguous")
        expected_mask = (batch_size, *self.backing_mask_shape)
        if not isinstance(mask, Tensor) or tuple(mask.shape) != expected_mask:
            raise ValueError(
                f"backing mask must have shape {list(expected_mask)}"
            )
        if mask.dtype != torch.bool:
            raise TypeError("backing mask must be boolean")
        if value.device != mask.device:
            raise ValueError("backing and mask must use the same device")


@dataclass(frozen=True)
class TensorOperationFieldSpec:
    """Bounded shape and synchronous-write contract for one operation field."""

    _component_reference: ClassVar[str] = "arti/tensor-operation-field-spec@2"

    port: PortSpec
    support_size: int
    collision_policy: Literal["last_element"] = "last_element"
    atomic_snapshot: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.port, PortSpec):
            raise TypeError("port must be PortSpec")
        if (
            isinstance(self.support_size, bool)
            or not isinstance(self.support_size, int)
            or self.support_size <= 0
        ):
            raise ValueError("support_size must be a positive integer")
        if self.collision_policy != "last_element":
            raise ValueError("collision_policy must be 'last_element'")
        if self.atomic_snapshot is not True:
            raise ValueError("operation fields require atomic_snapshot=True")

    @property
    def source_capacity(self) -> int:
        return self.port.canvas_tokens + self.port.element_count

    def contract(self) -> dict[str, object]:
        return {
            "ref": canonical_contract_reference(self._component_reference),
            "support_size": self.support_size,
            "source_capacity": self.source_capacity,
            "collision_policy": self.collision_policy,
            "atomic_snapshot": self.atomic_snapshot,
            "operations": tuple(operation.name for operation in EditOperation),
            "index_dtype": "int64",
            "value_dtype": str(self.port.dtype),
        }


@dataclass(frozen=True)
class PortSnapshot:
    """Resolved pre-step backing selected at a call boundary."""

    _component_reference: ClassVar[str] = "arti/operable-tensor-snapshot@2"

    value: Tensor
    mask: Tensor
    source: BackingSource
    backing_epoch: int
    step_index: int


class OperableTensorPort:
    """Runtime-owned stable port with an always-present default backing."""

    _component_reference: ClassVar[str] = "arti/operable-tensor-port@2"

    def __init__(
        self,
        spec: PortSpec,
        *,
        batch_size: int,
        device: torch.device | str | None = None,
        default_value: Tensor | None = None,
        default_mask: Tensor | None = None,
    ) -> None:
        if not isinstance(spec, PortSpec):
            raise TypeError("spec must be PortSpec")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer")
        self.spec = spec
        self.batch_size = batch_size
        resolved_device = torch.device("cpu" if device is None else device)
        if default_value is None:
            default_value = torch.full(
                (batch_size, *spec.backing_shape),
                spec.empty_value,
                dtype=spec.dtype,
                device=resolved_device,
            )
        if default_mask is None:
            default_mask = torch.full(
                (batch_size, *spec.backing_mask_shape),
                spec.default_visible,
                dtype=torch.bool,
                device=default_value.device,
            )
        spec.validate_backing(default_value, default_mask, batch_size=batch_size)
        self._default_value = default_value.detach().clone()
        self._default_mask = default_mask.detach().clone()
        self._external_value: Tensor | None = None
        self._external_mask: Tensor | None = None
        self._backing_epoch = 0
        self._step_index = 0

    @property
    def mounted(self) -> bool:
        return self._external_value is not None

    @property
    def backing_epoch(self) -> int:
        return self._backing_epoch

    @property
    def step_index(self) -> int:
        return self._step_index

    def resolve(self) -> PortSnapshot:
        if self._external_value is None:
            return PortSnapshot(
                self._default_value,
                self._default_mask,
                "default",
                self._backing_epoch,
                self._step_index,
            )
        assert self._external_mask is not None
        return PortSnapshot(
            self._external_value,
            self._external_mask,
            "external",
            self._backing_epoch,
            self._step_index,
        )

    def mount(self, value: Tensor, mask: Tensor) -> None:
        if self.mounted:
            raise RuntimeError("an external backing is already mounted; use replace")
        self._set_external(value, mask)

    def replace(self, value: Tensor, mask: Tensor) -> None:
        if not self.mounted:
            raise RuntimeError("no external backing is mounted; use mount")
        self._set_external(value, mask)

    def detach_to_default(self) -> None:
        if self.mounted:
            self._external_value = None
            self._external_mask = None
            self._backing_epoch += 1

    def advance(self, value: Tensor, mask: Tensor) -> None:
        """Install a validated functional edit for the next call."""

        self.spec.validate_backing(value, mask, batch_size=self.batch_size)
        current = self.resolve()
        if value.device != current.value.device:
            raise ValueError("next backing must remain on the current backing device")
        if current.source == "external":
            self._external_value = value.detach()
            self._external_mask = mask.detach()
        else:
            self._default_value = value.detach()
            self._default_mask = mask.detach()
        self._step_index += 1

    def _set_external(self, value: Tensor, mask: Tensor) -> None:
        self.spec.validate_backing(value, mask, batch_size=self.batch_size)
        if value.device != self._default_value.device:
            raise ValueError("external backing must use the port device")
        self._external_value = value.detach()
        self._external_mask = mask.detach()
        self._backing_epoch += 1


@dataclass(frozen=True)
class SharedCanvas:
    """World-shaped presentation of world and backing from one snapshot."""

    _component_reference: ClassVar[str] = "arti/shared-canvas@3"

    values: Tensor
    mask: Tensor
    world_values: Tensor
    world_mask: Tensor
    backing_values: Tensor
    backing_mask: Tensor
    source_plane: Tensor
    source_index: Tensor
    backing_epoch: int
    step_index: int


class SharedCanvasFold(nn.Module):
    """Fold declared tensor coordinates onto world positions without concatenation."""

    _component_reference: ClassVar[str] = "arti/shared-canvas-fold@3"

    def __init__(self, spec: PortSpec) -> None:
        super().__init__()
        if not isinstance(spec, PortSpec):
            raise TypeError("spec must be PortSpec")
        self.spec = spec
        self.register_buffer(
            "_canvas_index",
            torch.tensor(spec.tensor_to_canvas, dtype=torch.int64),
            persistent=False,
        )
        self.register_buffer(
            "_backing_index",
            torch.tensor(spec.folded_tensor_offsets, dtype=torch.int64),
            persistent=False,
        )

    def forward(
        self,
        world: Tensor,
        snapshot: PortSnapshot,
        *,
        world_mask: Tensor | None = None,
    ) -> SharedCanvas:
        if not isinstance(world, Tensor) or world.ndim != 3:
            raise TypeError("world must be a Tensor with shape [B, N, D]")
        batch = world.shape[0]
        if tuple(world.shape[1:]) != self.spec.canvas_shape:
            raise ValueError(
                f"world must have trailing shape {self.spec.canvas_shape}, "
                f"got {tuple(world.shape[1:])}"
            )
        if world.dtype != self.spec.dtype:
            raise TypeError(f"world dtype must be {self.spec.dtype}")
        if not world.is_contiguous():
            raise ValueError("world must be contiguous")
        self.spec.validate_backing(snapshot.value, snapshot.mask, batch_size=batch)
        if world.device != snapshot.value.device:
            raise ValueError("world and backing must use the same device")
        if world_mask is None:
            world_mask = torch.ones(
                (batch, self.spec.canvas_tokens),
                dtype=torch.bool,
                device=world.device,
            )
        if (
            not isinstance(world_mask, Tensor)
            or tuple(world_mask.shape) != (batch, self.spec.canvas_tokens)
            or world_mask.dtype != torch.bool
        ):
            raise TypeError("world_mask must be boolean with shape [B, N]")
        if world_mask.device != world.device:
            raise ValueError("world and world_mask must use the same device")

        empty = torch.full_like(world, self.spec.empty_value)
        canvas = torch.where(world_mask.unsqueeze(-1), world, empty)
        canvas_mask = world_mask.clone()
        source_plane = torch.full(
            (batch, self.spec.canvas_tokens),
            int(CanvasSource.WORLD),
            dtype=torch.int8,
            device=world.device,
        )
        source_index = torch.arange(
            self.spec.canvas_tokens,
            dtype=torch.int64,
            device=world.device,
        ).expand(batch, -1).clone()
        source_plane = torch.where(
            world_mask,
            source_plane,
            torch.full_like(source_plane, int(CanvasSource.EMPTY)),
        )
        source_index = torch.where(world_mask, source_index, torch.full_like(source_index, -1))

        canvas_index = self._canvas_index
        backing_index = self._backing_index
        if canvas_index.device != world.device:
            # SharedCanvasFold is otherwise stateless, so a standalone call should
            # remain device-native like the original implementation. Composite
            # modules still move this buffer once with ``module.to(device)``.
            canvas_index = canvas_index.to(device=world.device, non_blocking=True)
            backing_index = backing_index.to(device=world.device, non_blocking=True)
        flat_backing = self.spec.flatten_value(snapshot.value)
        flat_backing_mask = self.spec.flatten_mask(snapshot.mask)
        exposed_value = flat_backing.index_select(1, backing_index)
        exposed_mask = flat_backing_mask.index_select(1, backing_index)
        previous = canvas.index_select(1, canvas_index)
        canvas[:, canvas_index, :] = torch.where(
            exposed_mask.unsqueeze(-1),
            exposed_value,
            previous,
        )
        previous_mask = canvas_mask.index_select(1, canvas_index)
        canvas_mask[:, canvas_index] = exposed_mask | previous_mask
        backing_plane = torch.full(
            exposed_mask.shape,
            int(CanvasSource.BACKING),
            dtype=torch.int8,
            device=world.device,
        )
        selected_plane = source_plane.index_select(1, canvas_index)
        source_plane[:, canvas_index] = torch.where(
            exposed_mask,
            backing_plane,
            selected_plane,
        )
        exposed_index = backing_index.expand(batch, -1)
        selected_index = source_index.index_select(1, canvas_index)
        source_index[:, canvas_index] = torch.where(
            exposed_mask,
            exposed_index,
            selected_index,
        )
        return SharedCanvas(
            values=canvas,
            mask=canvas_mask,
            world_values=world,
            world_mask=world_mask,
            backing_values=snapshot.value,
            backing_mask=snapshot.mask,
            source_plane=source_plane,
            source_index=source_index,
            backing_epoch=snapshot.backing_epoch,
            step_index=snapshot.step_index,
        )


def _tensor_hash(value: Tensor) -> str:
    payload = value.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return hashlib.sha256(payload.numpy().tobytes()).hexdigest()


class TensorOperationQuery(nn.Module):
    """Fixed projection over complete world and backing snapshots."""

    _component_reference: ClassVar[str] = "arti/tensor-operation-query@4"

    def __init__(self, spec: PortSpec, key_dim: int, *, seed: int = 0) -> None:
        super().__init__()
        if not isinstance(spec, PortSpec):
            raise TypeError("spec must be PortSpec")
        if isinstance(key_dim, bool) or not isinstance(key_dim, int) or key_dim <= 0:
            raise ValueError("key_dim must be a positive integer")
        self.spec = spec
        self.key_dim = key_dim
        self.seed = int(seed)
        generator = torch.Generator(device="cpu").manual_seed(self.seed)
        scale = ((spec.canvas_tokens + spec.element_count) * spec.dim) ** -0.5
        basis = torch.randn(
            key_dim,
            spec.canvas_tokens,
            spec.dim,
            generator=generator,
            dtype=spec.dtype,
        ) * scale
        backing_basis = torch.randn(
            key_dim,
            spec.element_count,
            spec.dim,
            generator=generator,
            dtype=spec.dtype,
        ) * scale
        mask_basis = torch.randn(
            key_dim,
            spec.canvas_tokens,
            generator=generator,
            dtype=spec.dtype,
        ) * (spec.canvas_tokens + spec.element_count) ** -0.5
        backing_mask_basis = torch.randn(
            key_dim,
            spec.element_count,
            generator=generator,
            dtype=spec.dtype,
        ) * (spec.canvas_tokens + spec.element_count) ** -0.5
        self.register_buffer("basis", basis, persistent=True)
        self.register_buffer("backing_basis", backing_basis, persistent=True)
        self.register_buffer("mask_basis", mask_basis, persistent=True)
        self.register_buffer("backing_mask_basis", backing_mask_basis, persistent=True)

    def forward(self, canvas: SharedCanvas) -> Tensor:
        if not isinstance(canvas, SharedCanvas):
            raise TypeError("canvas must be SharedCanvas")
        if tuple(canvas.values.shape[1:]) != self.spec.canvas_shape:
            raise ValueError("canvas shape does not match the TensorOperation Query PortSpec")
        if canvas.values.dtype != self.spec.dtype:
            raise TypeError("canvas dtype does not match the TensorOperation Query PortSpec")
        world_visible = canvas.world_mask.detach()
        backing_visible = self.spec.flatten_mask(canvas.backing_mask.detach())
        world = torch.where(
            world_visible.unsqueeze(-1),
            canvas.world_values.detach(),
            torch.full_like(canvas.world_values, self.spec.empty_value),
        )
        backing = torch.where(
            backing_visible.unsqueeze(-1),
            self.spec.flatten_value(canvas.backing_values.detach()),
            torch.full_like(
                self.spec.flatten_value(canvas.backing_values), self.spec.empty_value
            ),
        )
        world_mask = world_visible.to(dtype=world.dtype)
        backing_mask = backing_visible.to(dtype=backing.dtype)
        return (
            torch.einsum("bnd,knd->bk", world, self.basis.to(world))
            + torch.einsum("bn,kn->bk", world_mask, self.mask_basis.to(world))
            + torch.einsum("bsd,ksd->bk", backing, self.backing_basis.to(backing))
            + torch.einsum(
                "bs,ks->bk", backing_mask, self.backing_mask_basis.to(backing)
            )
        )

    def operation_query_contract(self) -> dict[str, object]:
        return {
            "ref": canonical_contract_reference(self._component_reference),
            "key_dim": self.key_dim,
            "seed": self.seed,
            "basis_hash": _tensor_hash(self.basis),
            "backing_basis_hash": _tensor_hash(self.backing_basis),
            "mask_basis_hash": _tensor_hash(self.mask_basis),
            "backing_mask_basis_hash": _tensor_hash(self.backing_mask_basis),
            "input_view": "complete_world_and_backing",
            "fixed": True,
            "deterministic": True,
            "stateful": False,
        }


@dataclass(frozen=True)
class TensorOperationRouteSelection:
    """Hard member identity plus concat-local routing diagnostics."""

    route: Tensor
    hard_indices: Tensor
    logits: Tensor
    entropy: Tensor
    estimator: Literal["hard", "straight-through"]
    bank_indices: Tensor
    local_indices: Tensor
    local_probabilities: Tensor
    bank_probabilities: Tensor
    bank_ids: tuple[str, ...]


class TensorOperationBank(FormulaOperandBank):
    """Trainable candidates whose values are complete bounded operation fields."""

    _component_reference: ClassVar[str] = "arti/tensor-operation-bank@3"

    def __init__(
        self,
        spec: PortSpec,
        *,
        candidate_count: int,
        key_dim: int,
        support_size: int | None = None,
        bank_id: str | None = None,
        member_ids: Sequence[str] | None = None,
        seed: int = 0,
        init_scale: float = 0.02,
    ) -> None:
        if not isinstance(spec, PortSpec):
            raise TypeError("spec must be PortSpec")
        for value, name in ((candidate_count, "candidate_count"), (key_dim, "key_dim")):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(init_scale, bool)
            or not isinstance(init_scale, (int, float))
            or not math.isfinite(float(init_scale))
            or float(init_scale) <= 0
        ):
            raise ValueError("init_scale must be a finite positive number")
        size = spec.element_count if support_size is None else support_size
        field_spec = TensorOperationFieldSpec(spec, size)
        resolved_bank_id = f"tensor-operation-{int(seed)}" if bank_id is None else bank_id
        if not isinstance(resolved_bank_id, str) or not resolved_bank_id:
            raise ValueError("bank_id must be a non-empty string")
        resolved_members = (
            tuple(f"{resolved_bank_id}-member-{index:03d}" for index in range(candidate_count))
            if member_ids is None
            else tuple(member_ids)
        )
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        keys = torch.randn(
            candidate_count,
            key_dim,
            generator=generator,
            dtype=spec.dtype,
        ) * key_dim**-0.5

        def logits(*shape: int) -> Tensor:
            return torch.randn(
                candidate_count,
                *shape,
                generator=generator,
                dtype=spec.dtype,
            ) * float(init_scale)

        operands = {
            "active": logits(field_spec.support_size) - float(init_scale),
            "operation": logits(field_spec.support_size, len(EditOperation)),
            "source": logits(field_spec.support_size, field_spec.source_capacity),
            "destination": logits(field_spec.support_size, spec.element_count),
        }
        operands["operation"][:, :, int(EditOperation.KEEP)] += float(init_scale)
        super().__init__(
            keys=keys,
            operands=operands,
            source_ref=self._component_reference,
            bundle_id=resolved_bank_id,
            member_ids=resolved_members,
        )
        self.spec = spec
        self.field_spec = field_spec
        self.seed = int(seed)
        self.init_scale = float(init_scale)
        self.bank_ids = (resolved_bank_id,)
        self.group_slices = ((0, candidate_count),)
        self.parent_fingerprints: tuple[str, ...] = ()
        self.composition_kind = "native"
        self.register_buffer(
            "_group_influences",
            torch.ones(1, dtype=keys.dtype, device=keys.device),
            persistent=True,
        )
        self.register_buffer(
            "_member_group",
            torch.zeros(candidate_count, dtype=torch.int64, device=keys.device),
            persistent=False,
        )
        self.register_buffer(
            "_member_local",
            torch.arange(candidate_count, dtype=torch.int64, device=keys.device),
            persistent=False,
        )

    @classmethod
    def concat(
        cls,
        banks: Sequence[TensorOperationBank],
        *,
        name: str,
        influences: Sequence[float] | None = None,
    ) -> TensorOperationBank:
        """Materialize a typed candidate-axis union without remixing member values."""

        resolved = tuple(banks)
        if not resolved or any(not isinstance(bank, cls) for bank in resolved):
            raise TypeError("banks must be a non-empty sequence of TensorOperationBank")
        if not isinstance(name, str) or not name:
            raise ValueError("name must be a non-empty string")
        if influences is None:
            parent_influences = (1.0,) * len(resolved)
        else:
            parent_influences = tuple(float(value) for value in influences)
            if len(parent_influences) != len(resolved):
                raise ValueError("influences must contain one value per Bank")
        if any(not math.isfinite(value) or value < 0 for value in parent_influences):
            raise ValueError("influences must be finite and non-negative")
        if not any(value > 0 for value in parent_influences):
            raise ValueError("at least one Bank influence must be positive")

        first = resolved[0]
        operand_names = tuple(first.operands)
        for bank in resolved[1:]:
            if bank.spec != first.spec or bank.field_spec != first.field_spec:
                raise ValueError("all Banks must share the exact operation field contract")
            if bank.key_dim != first.key_dim or tuple(bank.operands) != operand_names:
                raise ValueError("all Banks must share key and operand schemas")
            if any(
                bank.operands[key].shape[1:] != first.operands[key].shape[1:]
                for key in operand_names
            ):
                raise ValueError("all Bank operand trailing shapes must match")
            if bank.keys.dtype != first.keys.dtype or bank.keys.device != first.keys.device:
                raise ValueError("all Banks must share key dtype and device")

        member_ids = tuple(member for bank in resolved for member in bank.member_ids)
        if len(set(member_ids)) != len(member_ids):
            raise ValueError("concat requires globally unique member_ids")
        group_ids = tuple(group for bank in resolved for group in bank.bank_ids)
        if len(set(group_ids)) != len(group_ids):
            raise ValueError("concat requires globally unique bank_ids")
        keys = torch.cat(tuple(bank.keys.detach() for bank in resolved), dim=0)
        operands = {
            key: torch.cat(tuple(bank.operands[key].detach() for bank in resolved), dim=0)
            for key in operand_names
        }
        result = cls.__new__(cls)
        FormulaOperandBank.__init__(
            result,
            keys=keys,
            operands=operands,
            source_ref=cls._component_reference,
            bundle_id=name,
            member_ids=member_ids,
        )
        result.spec = first.spec
        result.field_spec = first.field_spec
        result.seed = 0
        result.init_scale = first.init_scale
        result.bank_ids = group_ids
        result.composition_kind = "concat"
        result.parent_fingerprints = tuple(bank.operation_bank_fingerprint() for bank in resolved)

        group_slices: list[tuple[int, int]] = []
        group_weights: list[Tensor] = []
        member_groups: list[Tensor] = []
        member_locals: list[Tensor] = []
        candidate_offset = 0
        group_offset = 0
        for bank, parent_weight in zip(resolved, parent_influences, strict=True):
            for start, end in bank.group_slices:
                group_slices.append((candidate_offset + start, candidate_offset + end))
            group_weights.append(bank._group_influences.detach() * parent_weight)
            member_groups.append(bank._member_group.detach() + group_offset)
            member_locals.append(bank._member_local.detach())
            candidate_offset += bank.candidate_count
            group_offset += len(bank.group_slices)
        result.group_slices = tuple(group_slices)
        result.register_buffer(
            "_group_influences",
            torch.cat(group_weights).to(device=keys.device, dtype=keys.dtype),
            persistent=True,
        )
        result.register_buffer(
            "_member_group",
            torch.cat(member_groups).to(device=keys.device),
            persistent=False,
        )
        result.register_buffer(
            "_member_local",
            torch.cat(member_locals).to(device=keys.device),
            persistent=False,
        )
        return result

    def route(
        self,
        query: Tensor,
        *,
        estimator: Literal["hard", "straight-through"] = "straight-through",
        temperature: float = 1.0,
    ) -> TensorOperationRouteSelection:
        if not isinstance(query, Tensor) or not query.is_floating_point():
            raise TypeError("query must be a floating Tensor")
        if query.ndim != 2 or query.shape[0] <= 0 or query.shape[1] != self.key_dim:
            raise ValueError(f"query must have shape [B, {self.key_dim}]")
        if query.device != self.keys.device or query.dtype != self.keys.dtype:
            raise ValueError("query and TensorOperationBank keys must share device and dtype")
        if estimator not in {"hard", "straight-through"}:
            raise ValueError("estimator must be 'hard' or 'straight-through'")
        if not math.isfinite(float(temperature)) or float(temperature) <= 0:
            raise ValueError("temperature must be a finite positive number")
        _require_tensor(torch.isfinite(query).all(), "query must contain only finite values")

        logits = torch.nn.functional.normalize(query, dim=-1) @ torch.nn.functional.normalize(
            self.keys, dim=-1
        ).transpose(0, 1)
        weights = self._group_influences.to(logits)
        normalized_weights = weights / weights.sum()
        member_weights = normalized_weights.index_select(0, self._member_group)
        disabled_bias = torch.full_like(member_weights, torch.finfo(logits.dtype).min)
        member_bias = torch.where(member_weights > 0, member_weights.log(), disabled_bias)
        effective_logits = logits + member_bias.unsqueeze(0)
        hard = hard_formula_route(
            effective_logits,
            estimator="hard",
            temperature=float(temperature),
            member_ids=self.member_ids,
            member_priority=self._member_priority,
        )
        local_parts = [
            torch.softmax(logits[:, start:end] / float(temperature), dim=-1)
            for start, end in self.group_slices
        ]
        local_probabilities = torch.cat(local_parts, dim=-1)
        soft_route = torch.cat(
            [
                probabilities * normalized_weights[index]
                for index, probabilities in enumerate(local_parts)
            ],
            dim=-1,
        )
        route = hard.route if estimator == "hard" else hard.route + soft_route - soft_route.detach()
        entropy = -(
            soft_route
            * soft_route.clamp_min(torch.finfo(soft_route.dtype).tiny).log()
        ).sum(dim=-1)
        bank_indices = self._member_group.index_select(0, hard.hard_indices)
        local_indices = self._member_local.index_select(0, hard.hard_indices)
        return TensorOperationRouteSelection(
            route=route,
            hard_indices=hard.hard_indices,
            logits=effective_logits,
            entropy=entropy,
            estimator=estimator,
            bank_indices=bank_indices,
            local_indices=local_indices,
            local_probabilities=local_probabilities,
            bank_probabilities=normalized_weights.unsqueeze(0).expand(query.shape[0], -1),
            bank_ids=self.bank_ids,
        )

    def operation_bank_contract(self) -> dict[str, object]:
        operand_schema = {
            name: {
                "shape": tuple(value.shape),
                "dtype": str(value.dtype),
            }
            for name, value in sorted(self.operands.items())
        }
        return {
            "ref": canonical_contract_reference(self._component_reference),
            "schema_version": 3,
            "candidate_count": self.candidate_count,
            "key_dim": self.key_dim,
            "field": self.field_spec.contract(),
            "operand_schema": operand_schema,
            "member_ids": self.member_ids,
            "bank_ids": self.bank_ids,
            "group_slices": self.group_slices,
            "group_influences": tuple(float(value) for value in self._group_influences.tolist()),
            "route_normalizer": "per_bank_local",
            "composition_kind": self.composition_kind,
            "parent_fingerprints": self.parent_fingerprints,
        }

    def operation_bank_fingerprint(self) -> str:
        payload = json.dumps(
            self.operation_bank_contract(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class TensorOperationDecision:
    """One selected complete operation field plus differentiable diagnostics."""

    _component_reference: ClassVar[str] = "arti/tensor-operation-decision@3"

    instruction: TensorEditInstruction
    route: TensorOperationRouteSelection
    active_logits: Tensor
    operation_logits: Tensor
    source_logits: Tensor
    destination_logits: Tensor


class TensorOperationSelector(nn.Module):
    """Select and decode one complete field from a fixed Query and trainable Bank."""

    _component_reference: ClassVar[str] = "arti/tensor-operation-selector@3"

    def __init__(
        self,
        spec: PortSpec,
        bank: TensorOperationBank,
        *,
        query: TensorOperationQuery | None = None,
        estimator: Literal["hard", "straight-through"] = "straight-through",
        temperature: float = 1.0,
        query_seed: int = 0,
    ) -> None:
        super().__init__()
        if not isinstance(spec, PortSpec):
            raise TypeError("spec must be PortSpec")
        if not isinstance(bank, TensorOperationBank) or bank.spec != spec:
            raise ValueError("bank must be a TensorOperationBank for the selector PortSpec")
        if estimator not in {"hard", "straight-through"}:
            raise ValueError("estimator must be 'hard' or 'straight-through'")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or float(temperature) <= 0
        ):
            raise ValueError("temperature must be a finite positive number")
        resolved_query = (
            TensorOperationQuery(spec, bank.key_dim, seed=query_seed)
            if query is None
            else query
        )
        if not isinstance(resolved_query, TensorOperationQuery):
            raise TypeError("query must be TensorOperationQuery or None")
        if resolved_query.spec != spec or resolved_query.key_dim != bank.key_dim:
            raise ValueError("query shape contract does not match the TensorOperation Bank")
        if any(parameter.requires_grad for parameter in resolved_query.parameters()):
            raise ValueError("TensorOperation Query must remain fixed")
        contract = resolved_query.operation_query_contract()
        if (
            contract.get("fixed") is not True
            or contract.get("deterministic") is not True
            or contract.get("stateful") is not False
        ):
            raise ValueError("TensorOperation Query contract must be fixed and deterministic")
        self.spec = spec
        self.bank = bank
        self.query = resolved_query
        self.estimator = estimator
        self.temperature = float(temperature)

    def forward(self, canvas: SharedCanvas) -> TensorOperationDecision:
        route = self.bank.route(
            self.query(canvas),
            estimator=self.estimator,
            temperature=self.temperature,
        )

        def select(name: str) -> Tensor:
            return torch.einsum("bk,k...->b...", route.route, self.bank.operands[name])

        active_logits = select("active")
        operation_logits = select("operation")
        source_logits = select("source")
        destination_logits = select("destination")
        unified_source = source_logits.argmax(dim=-1)
        from_world = unified_source < self.spec.canvas_tokens
        instruction = TensorEditInstruction(
            operation=operation_logits.argmax(dim=-1).to(torch.int64),
            source_plane=torch.where(
                from_world,
                torch.full_like(unified_source, int(CanvasSource.WORLD)),
                torch.full_like(unified_source, int(CanvasSource.BACKING)),
            ).to(torch.int64),
            source_offset=torch.where(
                from_world,
                unified_source,
                unified_source - self.spec.canvas_tokens,
            ).to(torch.int64),
            destination_offset=destination_logits.argmax(dim=-1).to(torch.int64),
            active=active_logits >= 0,
        )
        return TensorOperationDecision(
            instruction=instruction,
            route=route,
            active_logits=active_logits,
            operation_logits=operation_logits,
            source_logits=source_logits,
            destination_logits=destination_logits,
        )


@dataclass(frozen=True)
class TensorEditInstruction:
    """One bounded parallel index-map operation over a logical tensor."""

    _component_reference: ClassVar[str] = "arti/tensor-edit-instruction@3"

    operation: Tensor
    source_plane: Tensor
    source_offset: Tensor
    destination_offset: Tensor
    active: Tensor


@dataclass(frozen=True)
class TensorEditResult:
    """Functional next backing returned by one synchronous operation field."""

    _component_reference: ClassVar[str] = "arti/tensor-edit-result@3"

    value: Tensor
    mask: Tensor
    instruction: TensorEditInstruction
    changed_support: Tensor


class TensorEditFormula(nn.Module):
    """Apply a complete KEEP/COPY/ERASE field from one immutable snapshot."""

    _component_reference: ClassVar[str] = "arti/tensor-edit-formula@3"

    def __init__(self, spec: PortSpec) -> None:
        super().__init__()
        if not isinstance(spec, PortSpec):
            raise TypeError("spec must be PortSpec")
        self.spec = spec

    def forward(
        self,
        canvas: SharedCanvas,
        snapshot: PortSnapshot,
        instruction: TensorEditInstruction,
    ) -> TensorEditResult:
        batch = snapshot.value.shape[0]
        self.spec.validate_backing(snapshot.value, snapshot.mask, batch_size=batch)
        if canvas.backing_epoch != snapshot.backing_epoch or canvas.step_index != snapshot.step_index:
            raise ValueError("canvas and backing snapshot do not describe the same call")
        if canvas.values.device != snapshot.value.device:
            raise ValueError("canvas and backing must use the same device")
        self._validate_instruction(instruction, batch=batch, device=snapshot.value.device)

        support_size = instruction.operation.shape[1]
        feature_index = (-1, -1, self.spec.dim)
        world_source = instruction.source_offset.clamp(0, self.spec.canvas_tokens - 1)
        backing_source = instruction.source_offset.clamp(0, self.spec.element_count - 1)
        flat_value = self.spec.flatten_value(snapshot.value)
        flat_mask = self.spec.flatten_mask(snapshot.mask)
        world_value = torch.gather(
            canvas.world_values,
            1,
            world_source.unsqueeze(-1).expand(*feature_index),
        )
        backing_value = torch.gather(
            flat_value,
            1,
            backing_source.unsqueeze(-1).expand(*feature_index),
        )
        world_visible = torch.gather(canvas.world_mask, 1, world_source)
        backing_visible = torch.gather(flat_mask, 1, backing_source)
        from_world = instruction.source_plane == int(CanvasSource.WORLD)
        copied_value = torch.where(from_world.unsqueeze(-1), world_value, backing_value)
        copied_mask = torch.where(from_world, world_visible, backing_visible)

        writes = instruction.active & (instruction.operation != int(EditOperation.KEEP))
        destination = instruction.destination_offset.clamp(0, self.spec.element_count - 1)
        elements = torch.arange(self.spec.element_count, device=snapshot.value.device)
        lanes = torch.arange(1, support_size + 1, device=snapshot.value.device)
        destination_match = destination.unsqueeze(-1) == elements
        priority = torch.where(
            writes.unsqueeze(-1) & destination_match,
            lanes.view(1, support_size, 1),
            torch.zeros((), dtype=lanes.dtype, device=lanes.device),
        )
        winning_priority, winner = priority.max(dim=1)
        has_write = winning_priority > 0
        gather_lane = winner.unsqueeze(-1)
        selected_operation = torch.gather(instruction.operation, 1, winner)
        selected_value = torch.gather(
            copied_value,
            1,
            gather_lane.expand(-1, -1, self.spec.dim),
        )
        selected_mask = torch.gather(copied_mask, 1, winner)
        copy = has_write & (selected_operation == int(EditOperation.COPY))
        erase = has_write & (selected_operation == int(EditOperation.ERASE))
        next_flat_value = torch.where(copy.unsqueeze(-1), selected_value, flat_value)
        next_flat_value = torch.where(
            erase.unsqueeze(-1),
            torch.full_like(next_flat_value, self.spec.empty_value),
            next_flat_value,
        )
        next_flat_mask = torch.where(copy, selected_mask, flat_mask)
        next_flat_mask = torch.where(erase, torch.zeros_like(next_flat_mask), next_flat_mask)
        winner_for_support = torch.gather(winner, 1, destination)
        support_index = torch.arange(support_size, device=snapshot.value.device).view(
            1, support_size
        )
        changed_support = writes & (winner_for_support == support_index)
        return TensorEditResult(
            self.spec.restore_value(next_flat_value).contiguous(),
            self.spec.restore_mask(next_flat_mask).contiguous(),
            instruction,
            changed_support,
        )

    def _validate_instruction(
        self,
        instruction: TensorEditInstruction,
        *,
        batch: int,
        device: torch.device,
    ) -> None:
        if not isinstance(instruction, TensorEditInstruction):
            raise TypeError("instruction must be TensorEditInstruction")
        fields = (
            instruction.operation,
            instruction.source_plane,
            instruction.source_offset,
            instruction.destination_offset,
            instruction.active,
        )
        if any(not isinstance(value, Tensor) or value.ndim != 2 for value in fields):
            raise ValueError("every instruction field must have shape [B, M]")
        shape = instruction.operation.shape
        if shape[0] != batch or shape[1] <= 0 or any(value.shape != shape for value in fields):
            raise ValueError("every instruction field must share shape [B, M]")
        if any(value.device != device for value in fields):
            raise ValueError("instruction fields must use the backing device")
        if instruction.active.dtype != torch.bool:
            raise TypeError("instruction active field must be boolean")
        for value in fields[:-1]:
            if value.dtype != torch.int64:
                raise TypeError("instruction index and enum fields must use int64")

        active = instruction.active
        valid_operation = (
            (instruction.operation == int(EditOperation.KEEP))
            | (instruction.operation == int(EditOperation.COPY))
            | (instruction.operation == int(EditOperation.ERASE))
        )
        _require_tensor(
            ~(active & ~valid_operation).any(),
            "instruction contains an unknown edit operation",
        )
        writes = active & (instruction.operation != int(EditOperation.KEEP))
        _require_tensor(
            ~(
                writes
                & (
                    (instruction.destination_offset < 0)
                    | (instruction.destination_offset >= self.spec.element_count)
                )
            ).any(),
            "instruction contains an invalid destination index",
        )
        copy = active & (instruction.operation == int(EditOperation.COPY))
        valid_plane = (
            (instruction.source_plane == int(CanvasSource.WORLD))
            | (instruction.source_plane == int(CanvasSource.BACKING))
        )
        _require_tensor(~(copy & ~valid_plane).any(), "COPY requires a valid source plane")
        _require_tensor(
            ~(
                copy
                & (instruction.source_plane == int(CanvasSource.WORLD))
                & (
                    (instruction.source_offset < 0)
                    | (instruction.source_offset >= self.spec.canvas_tokens)
                )
            ).any(),
            "COPY contains an invalid world source index",
        )
        _require_tensor(
            ~(
                copy
                & (instruction.source_plane == int(CanvasSource.BACKING))
                & (
                    (instruction.source_offset < 0)
                    | (instruction.source_offset >= self.spec.element_count)
                )
            ).any(),
            "COPY contains an invalid backing source index",
        )


class TensorEditSurrogate(nn.Module):
    """Keep the hard field forward while relaxing its route for gradients."""

    _component_reference: ClassVar[str] = "arti/tensor-edit-surrogate@3"

    def __init__(self, spec: PortSpec, *, temperature: float = 1.0) -> None:
        super().__init__()
        if not isinstance(spec, PortSpec):
            raise TypeError("spec must be PortSpec")
        if (
            isinstance(temperature, bool)
            or not isinstance(temperature, (int, float))
            or not math.isfinite(float(temperature))
            or float(temperature) <= 0
        ):
            raise ValueError("temperature must be a finite positive number")
        self.spec = spec
        self.temperature = float(temperature)

    def forward(
        self,
        canvas: SharedCanvas,
        snapshot: PortSnapshot,
        decision: TensorOperationDecision,
        hard_edit: TensorEditResult,
        *,
        active: Tensor | None = None,
    ) -> TensorEditResult:
        if not isinstance(decision, TensorOperationDecision):
            raise TypeError("decision must be TensorOperationDecision")
        if not isinstance(hard_edit, TensorEditResult):
            raise TypeError("hard_edit must be TensorEditResult")
        batch = snapshot.value.shape[0]
        self.spec.validate_backing(snapshot.value, snapshot.mask, batch_size=batch)
        if active is None:
            row_active = torch.ones((batch,), dtype=torch.bool, device=snapshot.value.device)
        else:
            if (
                not isinstance(active, Tensor)
                or active.shape != (batch,)
                or active.dtype != torch.bool
                or active.device != snapshot.value.device
            ):
                raise TypeError(
                    "active must be boolean with shape [B] on the snapshot device"
                )
            row_active = active
        temperature = self.temperature
        active_probability = torch.sigmoid(decision.active_logits / temperature)
        active_probability = active_probability * row_active.unsqueeze(-1)
        operation = torch.softmax(decision.operation_logits / temperature, dim=-1)
        source = torch.softmax(decision.source_logits / temperature, dim=-1)
        destination = torch.softmax(decision.destination_logits / temperature, dim=-1)
        flat_value = self.spec.flatten_value(snapshot.value)
        source_values = torch.cat((canvas.world_values, flat_value), dim=1)
        copied_value = torch.einsum("bmv,bvd->bmd", source, source_values)
        copy_probability = operation[..., int(EditOperation.COPY)]
        erase_probability = operation[..., int(EditOperation.ERASE)]
        write_probability = (copy_probability + erase_probability).clamp_min(
            torch.finfo(snapshot.value.dtype).tiny
        )
        conditional_value = (
            copy_probability.unsqueeze(-1) * copied_value
            + erase_probability.unsqueeze(-1) * self.spec.empty_value
        ) / write_probability.unsqueeze(-1)
        assignment = (
            active_probability.unsqueeze(-1)
            * write_probability.unsqueeze(-1)
            * destination
        )
        total = assignment.sum(dim=1)
        normalized = assignment / total.unsqueeze(1).clamp_min(
            torch.finfo(snapshot.value.dtype).tiny
        )
        replacement = torch.einsum("bms,bmd->bsd", normalized, conditional_value)
        occupancy = 1 - torch.prod((1 - assignment).clamp(0, 1), dim=1)
        relaxed_flat_value = (
            flat_value * (1 - occupancy).unsqueeze(-1)
            + replacement * occupancy.unsqueeze(-1)
        )
        relaxed_value = self.spec.restore_value(relaxed_flat_value)
        value = _HardForwardRelaxedBackward.apply(hard_edit.value, relaxed_value)
        return TensorEditResult(
            value,
            hard_edit.mask,
            hard_edit.instruction,
            hard_edit.changed_support,
        )


@dataclass(frozen=True)
class TensorOperationStepResult:
    """One fresh Query/Bank/Formula transition over a local backing."""

    _component_reference: ClassVar[str] = "arti/tensor-operation-step-result@3"

    canvas: SharedCanvas
    decision: TensorOperationDecision
    edit: TensorEditResult


class TensorOperation(nn.Module):
    """Apply one Bank-selected typed transition without mutating a live port."""

    _component_reference: ClassVar[str] = "arti/tensor-operation@3"

    def __init__(
        self,
        spec: PortSpec,
        selector: TensorOperationSelector,
        *,
        fold: SharedCanvasFold | None = None,
        formula: TensorEditFormula | None = None,
        surrogate: TensorEditSurrogate | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(spec, PortSpec):
            raise TypeError("spec must be PortSpec")
        if not isinstance(selector, TensorOperationSelector) or selector.spec != spec:
            raise ValueError("selector must be a TensorOperationSelector for the PortSpec")
        self.spec = spec
        self.selector = selector
        self.fold = SharedCanvasFold(spec) if fold is None else fold
        self.formula = TensorEditFormula(spec) if formula is None else formula
        self.surrogate = surrogate
        if not isinstance(self.fold, SharedCanvasFold) or self.fold.spec != spec:
            raise ValueError("fold must be SharedCanvasFold for the PortSpec")
        if not isinstance(self.formula, TensorEditFormula) or self.formula.spec != spec:
            raise ValueError("formula must be TensorEditFormula for the PortSpec")
        if surrogate is not None and (
            not isinstance(surrogate, TensorEditSurrogate) or surrogate.spec != spec
        ):
            raise ValueError("surrogate must be TensorEditSurrogate for the PortSpec or None")

    def forward(
        self,
        world: Tensor,
        snapshot: PortSnapshot,
        *,
        world_mask: Tensor | None = None,
        active: Tensor | None = None,
    ) -> TensorOperationStepResult:
        canvas = self.fold(world, snapshot, world_mask=world_mask)
        decision = self.selector(canvas)
        if active is not None:
            if (
                not isinstance(active, Tensor)
                or active.shape != (world.shape[0],)
                or active.dtype != torch.bool
                or active.device != world.device
            ):
                raise TypeError("active must be boolean with shape [B] on the world device")
            decision = replace(
                decision,
                instruction=replace(
                    decision.instruction,
                    active=decision.instruction.active & active.unsqueeze(-1),
                ),
            )
        hard_edit = self.formula(canvas, snapshot, decision.instruction)
        edit = (
            hard_edit
            if self.surrogate is None
            else self.surrogate(
                canvas,
                snapshot,
                decision,
                hard_edit,
                active=active,
            )
        )
        _require_tensor(
            torch.isfinite(edit.value).all(),
            "TensorOperation produced a non-finite backing",
        )
        return TensorOperationStepResult(canvas, decision, edit)


class TensorOperationStopReason(IntEnum):
    """Logical completion reason for each batch row."""

    ZERO_STEPS = 0
    MAX_STEPS = 1
    STABLE = 2


@dataclass(frozen=True)
class TensorOperationStopPolicy:
    """Bounded post-transition stopping for the operation axis only."""

    _component_reference: ClassVar[str] = "arti/tensor-operation-stop@1"

    min_operation_steps: int = 0
    stop_on_stable: bool = False

    def __post_init__(self) -> None:
        if (
            isinstance(self.min_operation_steps, bool)
            or not isinstance(self.min_operation_steps, int)
            or self.min_operation_steps < 0
        ):
            raise ValueError("min_operation_steps must be a non-negative integer")
        if not isinstance(self.stop_on_stable, bool):
            raise TypeError("stop_on_stable must be boolean")


@dataclass(frozen=True)
class TensorOperationSchedule:
    """Requested operation depth, independent from Reader iteration depth."""

    _component_reference: ClassVar[str] = "arti/tensor-operation-schedule@1"

    operation_steps: int | Tensor = 1
    max_steps: int | None = None

    def __post_init__(self) -> None:
        if self.max_steps is not None and (
            isinstance(self.max_steps, bool)
            or not isinstance(self.max_steps, int)
            or self.max_steps < 0
        ):
            raise ValueError("max_steps must be a non-negative integer or None")
        if isinstance(self.operation_steps, int) and self.max_steps is not None:
            if self.operation_steps > self.max_steps:
                raise ValueError("operation_steps cannot exceed max_steps")

    def resolve(self, *, batch: int, device: torch.device) -> Tensor:
        if isinstance(self.operation_steps, bool):
            raise TypeError("operation_steps must be an integer or int64 Tensor")
        if isinstance(self.operation_steps, int):
            if self.operation_steps < 0:
                raise ValueError("operation_steps must be non-negative")
            return torch.full(
                (batch,), self.operation_steps, dtype=torch.int64, device=device
            )
        steps = self.operation_steps
        if (
            not isinstance(steps, Tensor)
            or steps.shape != (batch,)
            or steps.dtype != torch.int64
            or steps.device != device
        ):
            raise TypeError("operation_steps Tensor must be int64 [B] on the input device")
        _require_tensor((steps >= 0).all(), "operation_steps must be non-negative")
        if self.max_steps is not None:
            _require_tensor(
                (steps <= self.max_steps).all(),
                "operation_steps cannot exceed max_steps",
            )
        return steps

    def capacity(self, requested: Tensor) -> int:
        if isinstance(self.operation_steps, int):
            return self.operation_steps
        if self.max_steps is not None:
            return self.max_steps
        if torch.compiler.is_compiling() or requested.device.type != "cpu":
            raise ValueError(
                "Tensor operation_steps require max_steps for compiled or accelerator execution"
            )
        return int(requested.max().item()) if requested.numel() else 0


@dataclass(frozen=True)
class TensorOperationTrace:
    """Bounded operation-axis diagnostics; it never owns persistent state."""

    _component_reference: ClassVar[str] = "arti/tensor-operation-trace@3"

    requested_steps: Tensor
    completed_steps: Tensor
    stop_reason: Tensor
    attempted: Tensor
    changed: Tensor
    route_index: Tensor
    operation: Tensor


@dataclass(frozen=True)
class TensorOperationResult:
    """Final local backing proposal plus an operation-only trace."""

    _component_reference: ClassVar[str] = "arti/tensor-operation-result@3"

    value: Tensor
    mask: Tensor
    trace: TensorOperationTrace


class TensorOperationLoop(nn.Module):
    """Repeatedly re-query a private shadow backing and return one proposal."""

    _component_reference: ClassVar[str] = "arti/tensor-operation-loop@3"

    def __init__(
        self,
        operation: TensorOperation,
        *,
        stop: TensorOperationStopPolicy | None = None,
        executor: TensorOperationExecutor = "static_masked",
    ) -> None:
        super().__init__()
        if not isinstance(operation, TensorOperation):
            raise TypeError("operation must be TensorOperation")
        if stop is not None and not isinstance(stop, TensorOperationStopPolicy):
            raise TypeError("stop must be TensorOperationStopPolicy or None")
        if executor not in {"static_masked", "early_break"}:
            raise ValueError("executor must be 'static_masked' or 'early_break'")
        self.operation = operation
        self.spec = operation.spec
        self.stop = stop or TensorOperationStopPolicy()
        self.executor = executor

    def forward(
        self,
        world: Tensor,
        snapshot: PortSnapshot,
        *,
        schedule: TensorOperationSchedule | None = None,
        world_mask: Tensor | None = None,
    ) -> TensorOperationResult:
        if schedule is None:
            schedule = TensorOperationSchedule()
        if not isinstance(schedule, TensorOperationSchedule):
            raise TypeError("schedule must be TensorOperationSchedule or None")
        if self.executor == "early_break" and (
            world.device.type != "cpu" or torch.compiler.is_compiling()
        ):
            raise ValueError("early_break is only supported by the eager CPU executor")
        batch = world.shape[0]
        requested = schedule.resolve(batch=batch, device=world.device)
        _require_tensor(
            (requested >= self.stop.min_operation_steps).all(),
            "operation_steps cannot be smaller than min_operation_steps",
        )

        shadow_value = snapshot.value.clone()
        shadow_mask = snapshot.mask.clone()
        stopped = torch.zeros((batch,), dtype=torch.bool, device=world.device)
        completed = torch.zeros((batch,), dtype=torch.int64, device=world.device)
        attempted_rows: list[Tensor] = []
        changed_rows: list[Tensor] = []
        route_rows: list[Tensor] = []
        operation_rows: list[Tensor] = []
        capacity = schedule.capacity(requested)

        for step_index in range(capacity):
            active = (requested > step_index) & ~stopped
            if self.executor == "early_break" and not bool(active.any()):
                break
            shadow = PortSnapshot(
                shadow_value,
                shadow_mask,
                snapshot.source,
                snapshot.backing_epoch,
                snapshot.step_index,
            )
            step = self.operation(
                world,
                shadow,
                world_mask=world_mask,
                active=active,
            )
            value_changed = (step.edit.value != shadow_value).flatten(1).any(dim=1)
            mask_changed = (step.edit.mask != shadow_mask).flatten(1).any(dim=1)
            changed = active & (value_changed | mask_changed)
            shadow_value = step.edit.value
            shadow_mask = step.edit.mask
            completed = completed + active.to(torch.int64)
            attempted_rows.append(active)
            changed_rows.append(changed)
            route_rows.append(
                torch.where(
                    active,
                    step.decision.route.hard_indices,
                    torch.full_like(step.decision.route.hard_indices, -1),
                )
            )
            operation_rows.append(
                torch.where(
                    active.unsqueeze(-1),
                    step.decision.instruction.operation,
                    torch.full_like(step.decision.instruction.operation, -1),
                )
            )
            if self.stop.stop_on_stable and step_index + 1 >= self.stop.min_operation_steps:
                stopped = stopped | (active & ~changed)

        empty_routes = torch.empty((0, batch), dtype=torch.int64, device=world.device)
        empty_operations = torch.empty(
            (0, batch, self.operation.selector.bank.field_spec.support_size),
            dtype=torch.int64,
            device=world.device,
        )
        attempted = (
            torch.stack(attempted_rows)
            if attempted_rows
            else torch.empty((0, batch), dtype=torch.bool, device=world.device)
        )
        changed = (
            torch.stack(changed_rows)
            if changed_rows
            else torch.empty((0, batch), dtype=torch.bool, device=world.device)
        )
        route_index = torch.stack(route_rows) if route_rows else empty_routes
        operations = torch.stack(operation_rows) if operation_rows else empty_operations
        reason = torch.full(
            (batch,), int(TensorOperationStopReason.MAX_STEPS), dtype=torch.int64, device=world.device
        )
        reason = torch.where(
            requested == 0,
            torch.full_like(reason, int(TensorOperationStopReason.ZERO_STEPS)),
            reason,
        )
        reason = torch.where(
            stopped,
            torch.full_like(reason, int(TensorOperationStopReason.STABLE)),
            reason,
        )
        trace = TensorOperationTrace(
            requested,
            completed,
            reason,
            attempted,
            changed,
            route_index,
            operations,
        )
        return TensorOperationResult(shadow_value, shadow_mask, trace)


@dataclass(frozen=True)
class ReaderIterationSchedule:
    """Reader depth kept deliberately separate from operation depth."""

    _component_reference: ClassVar[str] = "arti/reader-iteration-schedule@1"

    reader_steps: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.reader_steps, bool)
            or not isinstance(self.reader_steps, int)
            or self.reader_steps < 0
        ):
            raise ValueError("reader_steps must be a non-negative integer")


@dataclass(frozen=True)
class TensorInvocationResult:
    """Independent Reader result and next-call operation proposal."""

    _component_reference: ClassVar[str] = "arti/tensor-invocation-result@2"

    output: Tensor
    canvas: SharedCanvas
    operation: TensorOperationResult
    reader_steps: int


class TensorInvocation(nn.Module):
    """Coordinate independent Reader and tensor-operation axes for one call."""

    _component_reference: ClassVar[str] = "arti/tensor-invocation@2"

    def __init__(
        self,
        spec: PortSpec,
        reader: nn.Module,
        operation: TensorOperationLoop,
        *,
        fold: SharedCanvasFold | None = None,
    ) -> None:
        super().__init__()
        if not isinstance(spec, PortSpec):
            raise TypeError("spec must be PortSpec")
        if not isinstance(reader, nn.Module):
            raise TypeError("reader must be a torch.nn.Module")
        if not isinstance(operation, TensorOperationLoop) or operation.spec != spec:
            raise ValueError("operation must be a TensorOperationLoop for the PortSpec")
        self.spec = spec
        self.reader = reader
        self.operation = operation
        self.fold = SharedCanvasFold(spec) if fold is None else fold
        if not isinstance(self.fold, SharedCanvasFold) or self.fold.spec != spec:
            raise ValueError("fold must be SharedCanvasFold for the PortSpec")

    def forward(
        self,
        world: Tensor,
        snapshot: PortSnapshot,
        *,
        reader_schedule: ReaderIterationSchedule | None = None,
        operation_schedule: TensorOperationSchedule | None = None,
        world_mask: Tensor | None = None,
    ) -> TensorInvocationResult:
        reader_schedule = reader_schedule or ReaderIterationSchedule()
        if not isinstance(reader_schedule, ReaderIterationSchedule):
            raise TypeError("reader_schedule must be ReaderIterationSchedule or None")
        canvas = self.fold(world, snapshot, world_mask=world_mask)
        output = canvas.values
        for _ in range(reader_schedule.reader_steps):
            output = self.reader(output, mask=canvas.mask)
            if not isinstance(output, Tensor) or output.shape != canvas.values.shape:
                raise TypeError("reader must return a Tensor with the shared-canvas shape")
        operation = self.operation(
            world,
            snapshot,
            schedule=operation_schedule,
            world_mask=world_mask,
        )
        return TensorInvocationResult(
            output,
            canvas,
            operation,
            reader_schedule.reader_steps,
        )


__all__ = [
    "BackingSource",
    "CanvasSource",
    "EditOperation",
    "OperableTensorPort",
    "PortSnapshot",
    "PortSpec",
    "ReaderIterationSchedule",
    "SharedCanvas",
    "SharedCanvasFold",
    "TensorEditInstruction",
    "TensorEditFormula",
    "TensorEditResult",
    "TensorEditSurrogate",
    "TensorInvocation",
    "TensorInvocationResult",
    "TensorOperation",
    "TensorOperationBank",
    "TensorOperationDecision",
    "TensorOperationExecutor",
    "TensorOperationFieldSpec",
    "TensorOperationLoop",
    "TensorOperationQuery",
    "TensorOperationResult",
    "TensorOperationRouteSelection",
    "TensorOperationSchedule",
    "TensorOperationSelector",
    "TensorOperationStepResult",
    "TensorOperationStopPolicy",
    "TensorOperationStopReason",
    "TensorOperationTrace",
]
