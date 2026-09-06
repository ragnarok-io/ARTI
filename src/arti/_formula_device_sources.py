"""Port-level input views over immutable device pool products.

Views do not install values or donor Bank state into receiver frames. The same
resolved view is used for admission, packed execution and lineage checks.
"""

from typing import NamedTuple

import torch
from torch import Tensor, nn


class FormulaDeviceSources(NamedTuple):
    handles: Tensor
    ready: Tensor
    finite: Tensor
    producer: Tensor
    bank: Tensor
    revision: Tensor
    bank_handle: Tensor
    occurrence: Tensor
    port: Tensor


def select_sources(sources, rows, candidates):
    """Select [row, action, port] views in an execution packet's exact order."""
    return FormulaDeviceSources(*(field[rows, candidates] for field in sources))


def append_completed_sources(directory, cursor, products):
    """Append valid metadata in occurrence order; retain the oldest capacity.

    Pool payloads are never copied. Directory capacity is independent of answer
    pruning. Dropped publications do not roll back already executed work.
    """
    directory, products = FormulaDeviceSources(*directory), FormulaDeviceSources(*products)
    capacity, count = directory.handles.numel(), products.handles.numel()
    valid = products.ready & products.finite
    incoming = valid.to(torch.int64).cumsum(0)
    total = valid.sum()
    if capacity == 0 or count == 0:
        return directory, cursor, total
    positions = torch.arange(capacity, device=cursor.device) - cursor
    index = torch.searchsorted(incoming, (positions + 1).clamp_min(1)).clamp_max(count - 1)
    write = (positions >= 0) & (positions < total)
    result = FormulaDeviceSources(*(torch.where(write, new[index], old)
                                   for old, new in zip(directory, products, strict=True)))
    next_cursor = (cursor + total).clamp_max(capacity)
    return result, next_cursor, total - (next_cursor - cursor)


def completed_sources(frame_kernel, result, packet, occurrences):
    """Publish successful numeric ports, retaining lineage after frame pruning.

    ``occurrences`` identifies actual executions, not candidate names or Bank
    revision numbers. Returned directory columns own metadata only; payloads
    remain in their immutable pool rows.
    """
    state = result.state
    slots = frame_kernel.candidate_output_slots.index_select(
        0, packet.candidate_ids.clamp(0, frame_kernel.candidate_count - 1),
    )
    depth = state.depth.clamp(0, frame_kernel.spec.max_depth - 1)

    def lineage(field):
        return frame_kernel._frame(field, depth).gather(1, slots.clamp_min(0)).reshape(-1)

    handles = result.pool.output_handles
    ready = result.event.accepted[:, None] & (slots >= 0) & (handles >= 0)
    return FormulaDeviceSources(
        handles.reshape(-1), ready.reshape(-1), ready.reshape(-1),
        lineage(state.producer_candidate), lineage(state.producer_bank),
        lineage(state.producer_revision), lineage(state.producer_bank_handle),
        occurrences[:, None].expand_as(handles).reshape(-1),
        torch.arange(handles.shape[1], device=handles.device)[None].expand_as(handles).reshape(-1),
    )


class FormulaDeviceSourceBindings(nn.Module):
    """Resolve finite action bindings without inspecting device values on host.

    Directory fields have shape [capacity]. Product references, expected
    occurrence and output port have shape [rows, actions, input_ports]. A -1
    product reference selects the prepared local input; other invalid references
    remain unavailable rather than falling back to the local tensor.
    """

    def __init__(self, frame_kernel):
        super().__init__()
        self.max_depth = frame_kernel.spec.max_depth
        self.register_buffer("slots", frame_kernel.candidate_input_slots, persistent=False)

    def forward(self, state, local_finite, product_refs, occurrences, ports, directory):
        from ._formula_device_frames import FormulaDeviceFrameState

        state = FormulaDeviceFrameState(*state)
        directory = FormulaDeviceSources(*directory)
        depth = state.depth.clamp(0, self.max_depth - 1)
        required = self.slots >= 0
        safe_slots = self.slots.clamp_min(0).reshape(-1)

        def local(field):
            current = field.gather(1, depth[:, None, None].expand(-1, 1, field.shape[-1])).squeeze(1)
            return current.index_select(1, safe_slots).reshape(product_refs.shape)

        handles = local(state.value_handles)
        finite = local_finite.index_select(1, safe_slots).reshape(product_refs.shape)
        negative = torch.full_like(handles, -1)
        local_view = FormulaDeviceSources(
            handles, handles >= 0, finite, local(state.producer_candidate),
            local(state.producer_bank), local(state.producer_revision),
            local(state.producer_bank_handle), negative, negative,
        )
        capacity = directory.handles.shape[0]
        in_range = (product_refs >= 0) & (product_refs < capacity)
        safe = torch.where(in_range, product_refs, capacity).reshape(-1)
        fetched = FormulaDeviceSources(*(
            torch.cat((field, field.new_full((1,), False if field.dtype == torch.bool else -1)))
            .index_select(0, safe).reshape(product_refs.shape)
            for field in directory
        ))
        match = in_range & (fetched.occurrence == occurrences) & (fetched.port == ports)
        use_product = product_refs != -1
        resolved = FormulaDeviceSources(*(
            torch.where(use_product, remote, current)
            for remote, current in zip(fetched, local_view, strict=True)
        ))
        ready = resolved.ready & (resolved.handles >= 0) & (~use_product | match)
        return resolved._replace(
            ready=~required[None] | ready,
            finite=~required[None] | (ready & resolved.finite),
        )
