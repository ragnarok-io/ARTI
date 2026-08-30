"""Stable mounted tensor port and exact shared-canvas edit semantics."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, replace
from enum import IntEnum
from typing import ClassVar, Literal

import torch
from torch import Tensor, nn

from .formula_learning import FormulaOperandBank, FormulaRouteSelection


BackingSource = Literal["default", "external"]
TensorOperationExecutor = Literal["static_masked", "early_break"]


def _require_tensor(condition: Tensor, message: str) -> None:
    if torch.compiler.is_compiling() or condition.device.type != "cpu":
        torch._assert_async(condition, message)
    elif not bool(condition):
        raise ValueError(message)


class CanvasSource(IntEnum):
    """Logical source recorded for each shared-canvas position."""

    EMPTY = -1
    WORLD = 0
    BACKING = 1


class EditOperation(IntEnum):
    """Hard v1 edit operations over one backing snapshot."""

    KEEP = 0
    COPY = 1
    CLEAR = 2


@dataclass(frozen=True)
class PortSpec:
    """Immutable shape, mapping, and empty-value contract for one tensor port."""

    _component_reference: ClassVar[str] = "arti/operable-tensor-port-spec@1"

    canvas_tokens: int
    port_slots: int
    dim: int
    port_to_canvas: tuple[int, ...]
    dtype: torch.dtype = torch.float32
    empty_value: float = 0.0
    default_visible: bool = False
    coordinate_frame: str = "flat"

    def __post_init__(self) -> None:
        for name in ("canvas_tokens", "port_slots", "dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if len(self.port_to_canvas) != self.port_slots:
            raise ValueError("port_to_canvas must contain one index per port slot")
        if len(set(self.port_to_canvas)) != self.port_slots:
            raise ValueError("port_to_canvas must be injective")
        if any(index < 0 or index >= self.canvas_tokens for index in self.port_to_canvas):
            raise ValueError("port_to_canvas contains an out-of-range canvas index")
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
    def port_shape(self) -> tuple[int, int]:
        return self.port_slots, self.dim

    def validate_backing(self, value: Tensor, mask: Tensor, *, batch_size: int) -> None:
        if not isinstance(value, Tensor) or tuple(value.shape) != (
            batch_size,
            self.port_slots,
            self.dim,
        ):
            raise ValueError(
                "backing must have shape "
                f"[{batch_size}, {self.port_slots}, {self.dim}]"
            )
        if value.dtype != self.dtype:
            raise TypeError(f"backing dtype must be {self.dtype}")
        if not value.is_contiguous():
            raise ValueError("backing must be contiguous")
        if not isinstance(mask, Tensor) or tuple(mask.shape) != (
            batch_size,
            self.port_slots,
        ):
            raise ValueError(
                f"backing mask must have shape [{batch_size}, {self.port_slots}]"
            )
        if mask.dtype != torch.bool:
            raise TypeError("backing mask must be boolean")
        if value.device != mask.device:
            raise ValueError("backing and mask must use the same device")


@dataclass(frozen=True)
class PortSnapshot:
    """Resolved pre-step backing selected at a call boundary."""

    _component_reference: ClassVar[str] = "arti/operable-tensor-snapshot@1"

    value: Tensor
    mask: Tensor
    source: BackingSource
    backing_epoch: int
    step_index: int


class OperableTensorPort:
    """Runtime-owned stable port with an always-present default backing."""

    _component_reference: ClassVar[str] = "arti/operable-tensor-port@1"

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
                (batch_size, spec.port_slots, spec.dim),
                spec.empty_value,
                dtype=spec.dtype,
                device=resolved_device,
            )
        if default_mask is None:
            default_mask = torch.full(
                (batch_size, spec.port_slots),
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

    _component_reference: ClassVar[str] = "arti/shared-canvas@1"

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
    """Overlay a mounted backing onto world coordinates without concatenation."""

    _component_reference: ClassVar[str] = "arti/shared-canvas-fold@1"

    def __init__(self, spec: PortSpec) -> None:
        super().__init__()
        if not isinstance(spec, PortSpec):
            raise TypeError("spec must be PortSpec")
        self.spec = spec
        self.register_buffer(
            "_canvas_index",
            torch.tensor(spec.port_to_canvas, dtype=torch.int64),
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
        if canvas_index.device != world.device:
            # SharedCanvasFold is otherwise stateless, so a standalone call should
            # remain device-native like the original implementation. Composite
            # modules still move this buffer once with ``module.to(device)``.
            canvas_index = canvas_index.to(device=world.device, non_blocking=True)
        previous = canvas.index_select(1, canvas_index)
        canvas[:, canvas_index, :] = torch.where(
            snapshot.mask.unsqueeze(-1),
            snapshot.value,
            previous,
        )
        previous_mask = canvas_mask.index_select(1, canvas_index)
        canvas_mask[:, canvas_index] = snapshot.mask | previous_mask
        backing_plane = torch.full(
            snapshot.mask.shape,
            int(CanvasSource.BACKING),
            dtype=torch.int8,
            device=world.device,
        )
        selected_plane = source_plane.index_select(1, canvas_index)
        source_plane[:, canvas_index] = torch.where(
            snapshot.mask,
            backing_plane,
            selected_plane,
        )
        backing_index = torch.arange(
            self.spec.port_slots,
            dtype=torch.int64,
            device=world.device,
        ).expand(batch, -1)
        selected_index = source_index.index_select(1, canvas_index)
        source_index[:, canvas_index] = torch.where(
            snapshot.mask,
            backing_index,
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
    """Deterministic, non-trainable projection of one shared canvas."""

    _component_reference: ClassVar[str] = "arti/tensor-operation-query@1"

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
        scale = (spec.canvas_tokens * spec.dim) ** -0.5
        basis = torch.randn(
            key_dim,
            spec.canvas_tokens,
            spec.dim,
            generator=generator,
            dtype=spec.dtype,
        ) * scale
        mask_basis = torch.randn(
            key_dim,
            spec.canvas_tokens,
            generator=generator,
            dtype=spec.dtype,
        ) * spec.canvas_tokens**-0.5
        self.register_buffer("basis", basis, persistent=True)
        self.register_buffer("mask_basis", mask_basis, persistent=True)

    def forward(self, canvas: SharedCanvas) -> Tensor:
        if not isinstance(canvas, SharedCanvas):
            raise TypeError("canvas must be SharedCanvas")
        if tuple(canvas.values.shape[1:]) != self.spec.canvas_shape:
            raise ValueError("canvas shape does not match the TensorOperation Query PortSpec")
        if canvas.values.dtype != self.spec.dtype:
            raise TypeError("canvas dtype does not match the TensorOperation Query PortSpec")
        value = canvas.values.detach()
        mask = canvas.mask.detach().to(dtype=value.dtype)
        return torch.einsum("bnd,knd->bk", value, self.basis.to(value)) + torch.einsum(
            "bn,kn->bk", mask, self.mask_basis.to(value)
        )

    def operation_query_contract(self) -> dict[str, object]:
        return {
            "ref": self._component_reference,
            "key_dim": self.key_dim,
            "seed": self.seed,
            "basis_hash": _tensor_hash(self.basis),
            "mask_basis_hash": _tensor_hash(self.mask_basis),
            "fixed": True,
            "deterministic": True,
            "stateful": False,
        }


class TensorOperationBank(FormulaOperandBank):
    """Trainable candidate Bank for one bounded hard edit instruction."""

    _component_reference: ClassVar[str] = "arti/tensor-operation-bank@1"

    def __init__(
        self,
        spec: PortSpec,
        *,
        candidate_count: int,
        key_dim: int,
        seed: int = 0,
        init_scale: float = 0.02,
    ) -> None:
        if not isinstance(spec, PortSpec):
            raise TypeError("spec must be PortSpec")
        for value, name in (
            (candidate_count, "candidate_count"),
            (key_dim, "key_dim"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            isinstance(init_scale, bool)
            or not isinstance(init_scale, (int, float))
            or not math.isfinite(float(init_scale))
            or float(init_scale) <= 0
        ):
            raise ValueError("init_scale must be a finite positive number")
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        keys = torch.randn(
            candidate_count,
            key_dim,
            generator=generator,
            dtype=spec.dtype,
        ) * key_dim**-0.5

        def logits(width: int) -> Tensor:
            return (
                torch.randn(
                    candidate_count,
                    width,
                    generator=generator,
                    dtype=spec.dtype,
                )
                * float(init_scale)
            )

        operands = {
            "operation": logits(len(EditOperation)),
            "source_plane": logits(2),
            "world_source": logits(spec.canvas_tokens),
            "backing_source": logits(spec.port_slots),
            "destination": logits(spec.port_slots),
        }
        operands["operation"][:, int(EditOperation.KEEP)] += float(init_scale)
        super().__init__(
            keys=keys,
            operands=operands,
            source_ref=self._component_reference,
            bundle_id="tensor-operation",
            member_ids=tuple(
                f"tensor-operation-{index:03d}" for index in range(candidate_count)
            ),
        )
        self.spec = spec
        self.seed = int(seed)
        self.init_scale = float(init_scale)

    def operation_bank_contract(self) -> dict[str, object]:
        return {
            "ref": self._component_reference,
            "candidate_count": self.candidate_count,
            "key_dim": self.key_dim,
            "canvas_tokens": self.spec.canvas_tokens,
            "port_slots": self.spec.port_slots,
            "dim": self.spec.dim,
            "seed": self.seed,
        }


@dataclass(frozen=True)
class TensorOperationDecision:
    """Hard instruction plus differentiable Bank-routing diagnostics."""

    _component_reference: ClassVar[str] = "arti/tensor-operation-decision@1"

    instruction: TensorEditInstruction
    route: FormulaRouteSelection
    operation_logits: Tensor
    source_plane_logits: Tensor
    world_source_logits: Tensor
    backing_source_logits: Tensor
    destination_logits: Tensor


class TensorOperationSelector(nn.Module):
    """Select one hard edit from a fixed Query and trainable operand Bank."""

    _component_reference: ClassVar[str] = "arti/tensor-operation-selector@1"

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
        query = self.query(canvas)
        route = self.bank.route(
            query,
            estimator=self.estimator,
            temperature=self.temperature,
        )

        def select(name: str) -> Tensor:
            return torch.einsum("bk,k...->b...", route.route, self.bank.operands[name])

        operation_logits = select("operation")
        source_plane_logits = select("source_plane")
        world_source_logits = select("world_source")
        backing_source_logits = select("backing_source")
        destination_logits = select("destination")
        operation = operation_logits.argmax(dim=-1)
        source_plane = source_plane_logits.argmax(dim=-1)
        world_source = world_source_logits.argmax(dim=-1)
        backing_source = backing_source_logits.argmax(dim=-1)
        source_index = torch.where(
            source_plane == int(CanvasSource.WORLD),
            world_source,
            backing_source,
        )
        instruction = TensorEditInstruction(
            operation=operation.to(torch.int64),
            source_plane=source_plane.to(torch.int64),
            source_index=source_index.to(torch.int64),
            destination_index=destination_logits.argmax(dim=-1).to(torch.int64),
            active=torch.ones(
                operation.shape,
                dtype=torch.bool,
                device=operation.device,
            ),
        )
        return TensorOperationDecision(
            instruction=instruction,
            route=route,
            operation_logits=operation_logits,
            source_plane_logits=source_plane_logits,
            world_source_logits=world_source_logits,
            backing_source_logits=backing_source_logits,
            destination_logits=destination_logits,
        )


@dataclass(frozen=True)
class TensorEditInstruction:
    """One hard edit instruction per batch row."""

    _component_reference: ClassVar[str] = "arti/tensor-edit-instruction@1"

    operation: Tensor
    source_plane: Tensor
    source_index: Tensor
    destination_index: Tensor
    active: Tensor


@dataclass(frozen=True)
class TensorEditResult:
    """Functional next backing and mask returned by a tensor edit."""

    _component_reference: ClassVar[str] = "arti/tensor-edit-result@1"

    value: Tensor
    mask: Tensor
    instruction: TensorEditInstruction


class TensorEditFormula(nn.Module):
    """Apply exact KEEP, COPY, or CLEAR semantics to a backing snapshot."""

    _component_reference: ClassVar[str] = "arti/tensor-edit-formula@1"

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

        next_value = snapshot.value.clone()
        next_mask = snapshot.mask.clone()
        row = torch.arange(batch, device=snapshot.value.device)
        active = instruction.active
        copy = active & (instruction.operation == int(EditOperation.COPY))
        clear = active & (instruction.operation == int(EditOperation.CLEAR))

        destination = instruction.destination_index.clamp(0, self.spec.port_slots - 1)
        world_source = instruction.source_index.clamp(0, self.spec.canvas_tokens - 1)
        backing_source = instruction.source_index.clamp(0, self.spec.port_slots - 1)
        from_world = instruction.source_plane == int(CanvasSource.WORLD)
        copied_value = torch.where(
            from_world.unsqueeze(-1),
            canvas.world_values[row, world_source],
            snapshot.value[row, backing_source],
        )
        copied_mask = torch.where(
            from_world,
            canvas.world_mask[row, world_source],
            snapshot.mask[row, backing_source],
        )
        destination_value = next_value[row, destination]
        destination_mask = next_mask[row, destination]
        destination_value = torch.where(copy.unsqueeze(-1), copied_value, destination_value)
        destination_mask = torch.where(copy, copied_mask, destination_mask)
        destination_value = torch.where(
            clear.unsqueeze(-1),
            torch.full_like(destination_value, self.spec.empty_value),
            destination_value,
        )
        destination_mask = torch.where(clear, torch.zeros_like(destination_mask), destination_mask)
        next_value[row, destination] = destination_value
        next_mask[row, destination] = destination_mask
        return TensorEditResult(next_value, next_mask, instruction)

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
            instruction.source_index,
            instruction.destination_index,
            instruction.active,
        )
        if any(not isinstance(value, Tensor) or value.shape != (batch,) for value in fields):
            raise ValueError("every instruction field must have shape [B]")
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
            | (instruction.operation == int(EditOperation.CLEAR))
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
                    (instruction.destination_index < 0)
                    | (instruction.destination_index >= self.spec.port_slots)
                )
            ).any(),
            "instruction contains an invalid destination index",
        )
        copy = active & (instruction.operation == int(EditOperation.COPY))
        valid_plane = (
            (instruction.source_plane == int(CanvasSource.WORLD))
            | (instruction.source_plane == int(CanvasSource.BACKING))
        )
        _require_tensor(
            ~(copy & ~valid_plane).any(),
            "COPY requires WORLD or BACKING as its source plane",
        )
        _require_tensor(
            ~(
                copy
                & (instruction.source_plane == int(CanvasSource.WORLD))
                & (
                    (instruction.source_index < 0)
                    | (instruction.source_index >= self.spec.canvas_tokens)
                )
            ).any(),
            "COPY contains an invalid world source index",
        )
        _require_tensor(
            ~(
                copy
                & (instruction.source_plane == int(CanvasSource.BACKING))
                & (
                    (instruction.source_index < 0)
                    | (instruction.source_index >= self.spec.port_slots)
                )
            ).any(),
            "COPY contains an invalid backing source index",
        )


class TensorEditSurrogate(nn.Module):
    """Attach field-level gradients while preserving the exact hard edit forward."""

    _component_reference: ClassVar[str] = "arti/tensor-edit-surrogate@1"

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
    ) -> TensorEditResult:
        """Return hard values with the gradient of a continuous edit relaxation."""

        if not isinstance(canvas, SharedCanvas):
            raise TypeError("canvas must be SharedCanvas")
        if not isinstance(snapshot, PortSnapshot):
            raise TypeError("snapshot must be PortSnapshot")
        if not isinstance(decision, TensorOperationDecision):
            raise TypeError("decision must be TensorOperationDecision")
        if not isinstance(hard_edit, TensorEditResult):
            raise TypeError("hard_edit must be TensorEditResult")
        batch = snapshot.value.shape[0]
        self.spec.validate_backing(snapshot.value, snapshot.mask, batch_size=batch)
        if tuple(canvas.world_values.shape) != (
            batch,
            self.spec.canvas_tokens,
            self.spec.dim,
        ):
            raise ValueError("canvas world values do not match the surrogate PortSpec")
        if hard_edit.value.shape != snapshot.value.shape:
            raise ValueError("hard edit does not match the backing shape")

        temperature = self.temperature
        operation = torch.softmax(decision.operation_logits / temperature, dim=-1)
        source_plane = torch.softmax(decision.source_plane_logits / temperature, dim=-1)
        world_source = torch.softmax(decision.world_source_logits / temperature, dim=-1)
        backing_source = torch.softmax(
            decision.backing_source_logits / temperature,
            dim=-1,
        )
        destination = torch.softmax(decision.destination_logits / temperature, dim=-1)

        world_value = torch.einsum("bn,bnd->bd", world_source, canvas.world_values)
        backing_value = torch.einsum("bs,bsd->bd", backing_source, snapshot.value)
        copied_value = (
            source_plane[:, int(CanvasSource.WORLD)].unsqueeze(-1) * world_value
            + source_plane[:, int(CanvasSource.BACKING)].unsqueeze(-1) * backing_value
        )
        copy_probability = operation[:, int(EditOperation.COPY)]
        clear_probability = operation[:, int(EditOperation.CLEAR)]
        active = decision.instruction.active.to(dtype=snapshot.value.dtype)
        destination_probability = destination * active.unsqueeze(-1)
        replacement = (
            copy_probability.unsqueeze(-1) * copied_value
            + clear_probability.unsqueeze(-1) * self.spec.empty_value
        )
        write_probability = copy_probability + clear_probability
        delta = replacement.unsqueeze(1) - (
            write_probability[:, None, None] * snapshot.value
        )
        relaxed_value = snapshot.value + destination_probability.unsqueeze(-1) * delta
        value = hard_edit.value + (relaxed_value - relaxed_value.detach())
        return TensorEditResult(value, hard_edit.mask, hard_edit.instruction)


@dataclass(frozen=True)
class TensorOperationStepResult:
    """One fresh Query/Bank/Formula transition over a local backing."""

    _component_reference: ClassVar[str] = "arti/tensor-operation-step-result@1"

    canvas: SharedCanvas
    decision: TensorOperationDecision
    edit: TensorEditResult


class TensorOperation(nn.Module):
    """Apply one Bank-selected typed transition without mutating a live port."""

    _component_reference: ClassVar[str] = "arti/tensor-operation@1"

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
                or active.shape != decision.instruction.active.shape
                or active.dtype != torch.bool
                or active.device != world.device
            ):
                raise TypeError("active must be boolean with shape [B] on the world device")
            decision = replace(
                decision,
                instruction=replace(decision.instruction, active=active),
            )
        hard_edit = self.formula(canvas, snapshot, decision.instruction)
        edit = (
            hard_edit
            if self.surrogate is None
            else self.surrogate(canvas, snapshot, decision, hard_edit)
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
    """Requested operation depth, independent from Reader Refine depth."""

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

    _component_reference: ClassVar[str] = "arti/tensor-operation-trace@1"

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

    _component_reference: ClassVar[str] = "arti/tensor-operation-result@1"

    value: Tensor
    mask: Tensor
    trace: TensorOperationTrace


class TensorOperationLoop(nn.Module):
    """Repeatedly re-query a private shadow backing and return one proposal."""

    _component_reference: ClassVar[str] = "arti/tensor-operation-loop@1"

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
            mask_changed = (step.edit.mask != shadow_mask).any(dim=1)
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
                    active,
                    step.decision.instruction.operation,
                    torch.full_like(step.decision.instruction.operation, -1),
                )
            )
            if self.stop.stop_on_stable and step_index + 1 >= self.stop.min_operation_steps:
                stopped = stopped | (active & ~changed)

        empty = torch.empty((0, batch), dtype=torch.int64, device=world.device)
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
        route_index = torch.stack(route_rows) if route_rows else empty
        operations = torch.stack(operation_rows) if operation_rows else empty
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
class ReaderRefineSchedule:
    """Reader depth kept deliberately separate from operation depth."""

    _component_reference: ClassVar[str] = "arti/reader-refine-schedule@1"

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

    _component_reference: ClassVar[str] = "arti/tensor-invocation-result@1"

    output: Tensor
    canvas: SharedCanvas
    operation: TensorOperationResult
    reader_steps: int


class TensorInvocation(nn.Module):
    """Coordinate independent Reader and tensor-operation axes for one call."""

    _component_reference: ClassVar[str] = "arti/tensor-invocation@1"

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
        reader_schedule: ReaderRefineSchedule | None = None,
        operation_schedule: TensorOperationSchedule | None = None,
        world_mask: Tensor | None = None,
    ) -> TensorInvocationResult:
        reader_schedule = reader_schedule or ReaderRefineSchedule()
        if not isinstance(reader_schedule, ReaderRefineSchedule):
            raise TypeError("reader_schedule must be ReaderRefineSchedule or None")
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
    "ReaderRefineSchedule",
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
    "TensorOperationLoop",
    "TensorOperationQuery",
    "TensorOperationResult",
    "TensorOperationSchedule",
    "TensorOperationSelector",
    "TensorOperationStepResult",
    "TensorOperationStopPolicy",
    "TensorOperationStopReason",
    "TensorOperationTrace",
]
