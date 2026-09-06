"""Fixed-capacity device execution wave for prepared Formula programs.

The module joins grouped numerical dispatch, append-only value allocation and
the existing frame transition. It owns no Formula mathematics and performs no
host inspection of routes, counts or validity while a wave is running.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor, nn
from ._formula_device_sources import FormulaDeviceSources

from ._formula_device_dispatch import (
    FormulaDeviceDispatchPacket,
    FormulaDeviceNumericResult,
    FormulaDeviceNumericalDispatch,
)
from ._formula_device_frames import (
    FormulaDeviceFrameEvent,
    FormulaDeviceFrameKernel,
    FormulaDeviceFrameState,
)


class FormulaDevicePoolReceipt(NamedTuple):
    """Opaque handles and allocator state produced by one device wave."""

    output_handles: Tensor
    bank_handles: Tensor
    data_cursor: Tensor
    bank_cursor: Tensor
    overflow: Tensor


class FormulaDeviceExecutionResult(NamedTuple):
    """A complete numerical and frame-transition result."""

    state: FormulaDeviceFrameState
    event: FormulaDeviceFrameEvent
    pool: FormulaDevicePoolReceipt
    numeric: FormulaDeviceNumericResult


class FormulaDeviceExecutionWave(nn.Module):
    """Execute every packed lane and advance its forked device frame.

    ``data_pool`` and ``bank_pool`` are fixed-address append-only stores with a
    final sink row. Their public capacities exclude that sink. A failed global
    allocation leaves both cursors unchanged and rejects the complete packet so
    the caller can use an explicit larger bucket or native fallback.
    Typed layouts use tuples of real-shaped pools and per-pool cursor vectors;
    ordinary pools retain their scalar-cursor calling convention.
    """

    def __init__(
        self,
        dispatch: FormulaDeviceNumericalDispatch,
        frame_kernel: FormulaDeviceFrameKernel,
        *,
        data_capacity: int | tuple[int, ...],
        bank_capacity: int | tuple[int, ...],
    ) -> None:
        super().__init__()
        if dispatch.frame_kernel is not frame_kernel:
            raise ValueError("dispatch and frame kernel must share one prepared spec")
        for name, value in (
            ("data_capacity", data_capacity),
            ("bank_capacity", bank_capacity),
        ):
            capacities = (value,) if isinstance(value, int) else tuple(value)
            if not capacities or any(type(c) is not int or c < 1 for c in capacities):
                raise ValueError(f"{name} must contain positive integer capacities")
        self.dispatch = dispatch
        self.frame_kernel = frame_kernel
        self.data_capacity = data_capacity
        self.bank_capacity = bank_capacity
        if dispatch.data_layout is not None:
            if (data_capacity != dispatch.data_layout.capacities
                    or bank_capacity != dispatch.bank_layout.capacities):
                raise ValueError("typed capacities must match the prepared pool layouts")

    @staticmethod
    def _fork_state(
        state: FormulaDeviceFrameState,
        source_rows: Tensor,
    ) -> FormulaDeviceFrameState:
        return FormulaDeviceFrameState(*(
            tensor.index_select(0, source_rows) for tensor in state
        ))

    @staticmethod
    def _reserve(
        active: Tensor,
        cursor: Tensor,
        capacity: int,
    ) -> tuple[Tensor, Tensor, Tensor]:
        count = active.to(torch.int64).sum()
        rank = active.to(torch.int64).cumsum(0) - 1
        handles = cursor + rank
        cursor_valid = (cursor >= 0) & (cursor <= capacity)
        overflow = ~cursor_valid | (cursor + count > capacity)
        committed = active & ~overflow
        handles = torch.where(committed, handles, handles.new_full((), -1))
        next_cursor = torch.where(overflow, cursor, cursor + count)
        return handles, next_cursor, overflow

    @staticmethod
    def _write_pool(
        pool: Tensor,
        values: Tensor,
        handles: Tensor,
        active: Tensor,
        capacity: int,
    ) -> None:
        sink = handles.new_full((), capacity)
        indices = torch.where(active, handles, sink)
        mask = active.reshape((active.shape[0],) + (1,) * (values.ndim - 1))
        pool[indices] = torch.where(mask, values, torch.zeros_like(values))

    @torch.no_grad()
    def forward(
        self,
        state: FormulaDeviceFrameState,
        packet: FormulaDeviceDispatchPacket,
        data_pool: Tensor | tuple[Tensor, ...],
        bank_pool: Tensor | tuple[Tensor, ...],
        data_cursor: Tensor,
        bank_cursor: Tensor,
        effect_tape: tuple[tuple[Tensor, ...], ...] = (),
        effect_tape_position: Tensor | None = None,
        input_sources=None,
    ) -> FormulaDeviceExecutionResult:
        state = FormulaDeviceFrameState(*state)
        packet = FormulaDeviceDispatchPacket(*packet)
        typed = self.dispatch.data_layout is not None
        data_pools = tuple(data_pool) if typed else (data_pool,)
        bank_pools = tuple(bank_pool) if typed else (bank_pool,)
        data_capacities = self.data_capacity if typed else (self.data_capacity,)
        bank_capacities = self.bank_capacity if typed else (self.bank_capacity,)
        data_offsets = self.dispatch.data_layout.offsets if typed else (0,)
        bank_offsets = self.dispatch.bank_layout.offsets if typed else (0,)
        for pools, capacities, cursor, layout in (
            (data_pools, data_capacities, data_cursor, self.dispatch.data_layout),
            (bank_pools, bank_capacities, bank_cursor, self.dispatch.bank_layout),
        ):
            for index, (pool, capacity) in enumerate(zip(pools, capacities, strict=True)):
                if pool.shape[0] != capacity + 1:
                    raise ValueError("pool must include its prepared capacity and one sink row")
                if typed and (tuple(pool.shape[1:]) != layout.shapes[index] or pool.dtype != layout.dtypes[index]):
                    raise ValueError("pool shape/dtype differs from its prepared layout")
            expected = (len(pools),) if typed else ()
            if tuple(cursor.shape) != expected or cursor.dtype is not torch.int64:
                raise TypeError("cursor must be int64 with one entry per typed pool (scalar for homogeneous)")
        if any(
            tensor.device != data_pools[0].device
            for tensor in (*state, *packet, *data_pools, *bank_pools, data_cursor, bank_cursor)
        ):
            raise ValueError("frame, packet, pools and cursors must share one device")

        if input_sources is not None:
            input_sources = FormulaDeviceSources(*input_sources)
            allocated = torch.zeros_like(input_sources.ready)
            for offset, cursor in zip(data_offsets, data_cursor.unbind() if typed else (data_cursor,), strict=True):
                allocated |= (input_sources.handles >= offset) & (input_sources.handles < offset + cursor)
            required = self.frame_kernel.candidate_input_slots.index_select(
                0, packet.candidate_ids.clamp(0, self.frame_kernel.candidate_count - 1),
            ) >= 0
            input_sources = input_sources._replace(
                ready=input_sources.ready & (~required | allocated),
            )
            port = self.frame_kernel.candidate_effect_port.index_select(
                0, packet.candidate_ids.clamp(0, self.frame_kernel.candidate_count - 1),
            )
            effect_bank = input_sources.bank.gather(1, port.clamp_min(0)[:, None]).squeeze(1)
            bank_count = state.bank_value_handles.shape[1]
            valid_target = torch.zeros_like(packet.valid)
            if bank_count:
                row = packet.source_rows.clamp(0, state.active.shape[0] - 1)
                bank = effect_bank.clamp(0, bank_count - 1)
                current_handle = state.bank_value_handles[row, bank]
                current_revision = state.bank_revisions[row, bank]
                handle = input_sources.bank_handle.gather(1, port.clamp_min(0)[:, None]).squeeze(1)
                revision = input_sources.revision.gather(1, port.clamp_min(0)[:, None]).squeeze(1)
                valid_target = (effect_bank >= 0) & (effect_bank < bank_count)
                valid_target &= (handle == current_handle) & (revision == current_revision)
            packet = packet._replace(valid=packet.valid & ((port < 0) | valid_target))
        numeric = FormulaDeviceNumericResult(*self.dispatch(
            tuple(state), tuple(packet), data_pool, bank_pool,
            effect_tape, effect_tape_position,
            input_sources,
        ))
        packet_valid = packet.well_formed & ~numeric.overflow
        row_valid = numeric.numeric_valid & packet_valid
        output_values = numeric.output_values if typed else (numeric.output_values,)
        output_present = numeric.output_present if typed else (numeric.output_present,)
        bank_values = numeric.bank_successors if typed else (numeric.bank_successors,)
        bank_present = numeric.bank_successor_present if typed else (numeric.bank_successor_present,)
        data_cursors = data_cursor.unbind() if typed else (data_cursor,)
        bank_cursors = bank_cursor.unbind() if typed else (bank_cursor,)
        aliases = numeric.output_alias_handles.reshape(-1)
        valid_heads = row_valid.repeat_interleave(self.frame_kernel.spec.max_outputs)
        overflow = ~packet_valid
        data_reservations, bank_reservations = [], []
        encoded_outputs = torch.zeros_like(aliases)
        encoded_banks = torch.zeros_like(row_valid, dtype=torch.int64)
        for presence, cursor, capacity, offset in zip(
            output_present, data_cursors, data_capacities, data_offsets, strict=True,
        ):
            present = presence.reshape(-1) & valid_heads
            alias_present = present & (aliases >= 0)
            alias_valid = ~alias_present | ((aliases >= offset) & (aliases < offset + cursor))
            needs_data = present & ~alias_present
            handles, next_cursor, full = self._reserve(needs_data, cursor, capacity)
            overflow = overflow | full | ~alias_valid.all()
            encoded_outputs = encoded_outputs + torch.where(
                alias_present, aliases + 1, torch.where(needs_data & ~full, handles + offset + 1, 0),
            )
            data_reservations.append((handles, next_cursor, needs_data))
        for presence, cursor, capacity, offset in zip(
            bank_present, bank_cursors, bank_capacities, bank_offsets, strict=True,
        ):
            needed = presence & row_valid
            handles, next_cursor, full = self._reserve(needed, cursor, capacity)
            overflow = overflow | full
            encoded_banks = encoded_banks + torch.where(needed & ~full, handles + offset + 1, 0)
            bank_reservations.append((handles, next_cursor, needed))
        commit = ~overflow
        output_handles = torch.where(commit, encoded_outputs, 0).reshape(numeric.output_alias_handles.shape) - 1
        committed_bank_handles = torch.where(commit, encoded_banks, 0) - 1
        next_cursors = []
        for pools, values, capacities, cursors, reservations in (
            (data_pools, output_values, data_capacities, data_cursors, data_reservations),
            (bank_pools, bank_values, bank_capacities, bank_cursors, bank_reservations),
        ):
            updated = []
            for pool, value, capacity, cursor, (handles, next_cursor, needed) in zip(
                pools, values, capacities, cursors, reservations, strict=True,
            ):
                self._write_pool(pool, value.reshape(-1, *pool.shape[1:]), handles, needed & commit, capacity)
                updated.append(torch.where(commit, next_cursor, cursor))
            next_cursors.append(torch.stack(updated) if typed else updated[0])
        next_data_cursor, next_bank_cursor = next_cursors

        source_rows = packet.source_rows.clamp(0, state.active.shape[0] - 1)
        forked = self._fork_state(state, source_rows)
        next_state, event = self.frame_kernel(
            tuple(forked),
            torch.where(commit, packet.candidate_ids, torch.full_like(packet.candidate_ids, -1)),
            output_handles,
            row_valid & commit,
            committed_bank_handles,
            input_sources,
        )
        return FormulaDeviceExecutionResult(
            FormulaDeviceFrameState(*next_state),
            FormulaDeviceFrameEvent(*event),
            FormulaDevicePoolReceipt(
                output_handles,
                committed_bank_handles,
                next_data_cursor,
                next_bank_cursor,
                overflow,
            ),
            numeric,
        )


class FormulaDeviceCapturedExecutionWave:
    """Fixed-address CUDA Graph replay handle for one complete device wave."""

    def __init__(
        self,
        wave: FormulaDeviceExecutionWave,
        state: FormulaDeviceFrameState,
        packet: FormulaDeviceDispatchPacket,
        data_pool: Tensor | tuple[Tensor, ...],
        bank_pool: Tensor | tuple[Tensor, ...],
        data_cursor: Tensor,
        bank_cursor: Tensor,
        graph: torch.cuda.CUDAGraph,
        result: FormulaDeviceExecutionResult,
        input_sources=None,
    ) -> None:
        self.wave = wave
        self.state = state
        self.packet = packet
        self.data_pool = data_pool
        self.bank_pool = bank_pool
        self.data_cursor = data_cursor
        self.bank_cursor = bank_cursor
        self.graph: torch.cuda.CUDAGraph | None = graph
        self.result = result
        self.input_sources = input_sources

    @staticmethod
    def _pool_fields(pool):
        return (pool,) if isinstance(pool, Tensor) else tuple(pool)

    @classmethod
    def _clone_pool(cls, pool):
        copied = tuple(value.detach().clone() for value in cls._pool_fields(pool))
        return copied[0] if isinstance(pool, Tensor) else copied

    @classmethod
    def _copy_pool(cls, destination, source, *, name):
        cls._copy_fields(cls._pool_fields(destination), cls._pool_fields(source), name=name)

    @staticmethod
    def _copy_tensor(destination: Tensor, source: Tensor, *, name: str) -> None:
        if (
            destination.shape != source.shape
            or destination.dtype != source.dtype
            or destination.device != source.device
        ):
            raise ValueError(f"{name} differs from the captured tensor contract")
        destination.copy_(source.detach(), non_blocking=True)

    @classmethod
    def _copy_fields(
        cls,
        destination: tuple[Tensor, ...],
        source: tuple[Tensor, ...],
        *,
        name: str,
    ) -> None:
        if len(destination) != len(source):
            raise ValueError(f"{name} field count differs from the captured contract")
        for index, (destination_tensor, source_tensor) in enumerate(
            zip(destination, source, strict=True)
        ):
            cls._copy_tensor(
                destination_tensor,
                source_tensor,
                name=f"{name}[{index}]",
            )

    @classmethod
    def capture(
        cls,
        wave: FormulaDeviceExecutionWave,
        state: FormulaDeviceFrameState,
        packet: FormulaDeviceDispatchPacket,
        data_pool: Tensor | tuple[Tensor, ...],
        bank_pool: Tensor | tuple[Tensor, ...],
        data_cursor: Tensor,
        bank_cursor: Tensor,
        *,
        warmup_steps: int = 3,
        input_sources=None,
    ) -> "FormulaDeviceCapturedExecutionWave":
        if type(warmup_steps) is not int or warmup_steps <= 0:
            raise ValueError("warmup_steps must be a positive integer")
        state = FormulaDeviceFrameState(*state)
        packet = FormulaDeviceDispatchPacket(*packet)
        tensors = (*state, *packet, *cls._pool_fields(data_pool), *cls._pool_fields(bank_pool),
                   data_cursor, bank_cursor, *(input_sources or ()))
        devices = {tensor.device for tensor in tensors}
        device = cls._pool_fields(data_pool)[0].device
        if len(devices) != 1 or device.type != "cuda":
            raise ValueError("captured execution wave requires one CUDA device")
        static_state = FormulaDeviceFrameState(*(tensor.detach().clone() for tensor in state))
        static_packet = FormulaDeviceDispatchPacket(
            *(tensor.detach().clone() for tensor in packet)
        )
        static_data_pool = cls._clone_pool(data_pool)
        static_bank_pool = cls._clone_pool(bank_pool)
        static_data_cursor = data_cursor.detach().clone()
        static_bank_cursor = bank_cursor.detach().clone()
        static_sources = (None if input_sources is None else
                          FormulaDeviceSources(*(field.detach().clone() for field in input_sources)))
        source_kwargs = {} if static_sources is None else {"input_sources": static_sources}
        pristine_data = cls._clone_pool(static_data_pool)
        pristine_bank = cls._clone_pool(static_bank_pool)
        wave.dispatch.refresh_operands_()

        with torch.no_grad(), torch.cuda.device(device):
            current_stream = torch.cuda.current_stream(device)
            warmup_stream = torch.cuda.Stream(device=device)
            warmup_stream.wait_stream(current_stream)
            with torch.cuda.stream(warmup_stream):
                for _ in range(warmup_steps):
                    wave(
                        tuple(static_state),
                        tuple(static_packet),
                        static_data_pool,
                        static_bank_pool,
                        static_data_cursor,
                        static_bank_cursor,
                        **source_kwargs,
                    )
            current_stream.wait_stream(warmup_stream)
            torch.cuda.synchronize(device)
            cls._copy_pool(static_data_pool, pristine_data, name="data_pool")
            cls._copy_pool(static_bank_pool, pristine_bank, name="bank_pool")
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = wave(
                    tuple(static_state),
                    tuple(static_packet),
                    static_data_pool,
                    static_bank_pool,
                    static_data_cursor,
                    static_bank_cursor,
                    **source_kwargs,
                )
            torch.cuda.synchronize(device)
            cls._copy_pool(static_data_pool, pristine_data, name="data_pool")
            cls._copy_pool(static_bank_pool, pristine_bank, name="bank_pool")
        return cls(
            wave,
            static_state,
            static_packet,
            static_data_pool,
            static_bank_pool,
            static_data_cursor,
            static_bank_cursor,
            graph,
            FormulaDeviceExecutionResult(*result),
            static_sources,
        )

    @property
    def closed(self) -> bool:
        return self.graph is None

    @torch.no_grad()
    def copy_inputs_(
        self,
        *,
        state: FormulaDeviceFrameState | None = None,
        packet: FormulaDeviceDispatchPacket | None = None,
        data_pool: Tensor | tuple[Tensor, ...] | None = None,
        bank_pool: Tensor | tuple[Tensor, ...] | None = None,
        data_cursor: Tensor | None = None,
        bank_cursor: Tensor | None = None,
        input_sources=None,
    ) -> None:
        """Refresh a prepared bucket; connected device graphs can write it directly."""
        if self.closed:
            raise RuntimeError("captured execution wave is closed")
        if state is not None:
            self._copy_fields(tuple(self.state), tuple(FormulaDeviceFrameState(*state)), name="state")
        if packet is not None:
            self._copy_fields(
                tuple(self.packet),
                tuple(FormulaDeviceDispatchPacket(*packet)),
                name="packet",
            )
        if data_pool is not None:
            self._copy_pool(self.data_pool, data_pool, name="data_pool")
        if bank_pool is not None:
            self._copy_pool(self.bank_pool, bank_pool, name="bank_pool")
        if data_cursor is not None:
            self._copy_tensor(self.data_cursor, data_cursor, name="data_cursor")
        if bank_cursor is not None:
            self._copy_tensor(self.bank_cursor, bank_cursor, name="bank_cursor")
        if input_sources is not None:
            if self.input_sources is None:
                raise ValueError("input sources were not part of this capture")
            self._copy_fields(self.input_sources, input_sources, name="input_sources")

    @torch.no_grad()
    def replay(self) -> FormulaDeviceExecutionResult:
        graph = self.graph
        if graph is None:
            raise RuntimeError("captured execution wave is closed")
        graph.replay()
        return self.result

    def close(self) -> bool:
        if self.graph is None:
            return False
        torch.cuda.synchronize(self._pool_fields(self.data_pool)[0].device)
        self.graph = None
        return True


__all__ = [
    "FormulaDeviceCapturedExecutionWave",
    "FormulaDeviceExecutionResult",
    "FormulaDeviceExecutionWave",
    "FormulaDevicePoolReceipt",
]
