"""Device-side packing for complete K-wide Formula decision waves.

This module owns no Formula mathematics.  It turns the fixed-width result of
``FormulaDeviceDecisionWave`` into a stable, dropless group order that later
numeric dispatch regions can consume without reading candidate ids on the
host.  The group table is prepared from the mounted Query graph; runtime rows,
scores and routes remain tensors.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from dataclasses import dataclass
from itertools import product
from typing import NamedTuple

import torch
from torch import Tensor, nn
from ._formula_device_sources import FormulaDeviceSources
from ._formula_grouped_training import automatic_cuda_compilation

from ._formula_device_decision import (
    FormulaDeviceDecisionResult,
    FormulaDeviceDecisionWave,
)
from ._formula_device_frames import FormulaDeviceFrameKernel, FormulaDeviceFrameState
from ._formula_device_pools import FormulaDevicePoolLayout
from ._formula_candidate_batch import _checked_plan, _program
from ._formula_effect_execution import _TensorEffectPlan
from .formula_program_call import FormulaProgramCallCandidateV1
from .formula_program_query_v4 import (
    FormulaProgramEffectCandidateV3,
    FormulaProgramTensorCandidateV3,
)
from .formula_program_query_v5 import (
    FormulaProgramQueryV5,
    FormulaProgramTensorCandidateV4,
)
from .formula_v2 import (
    BankBinding, FormulaBindingError, FormulaExecutionPlanV2, InputBinding, PreparedFormulaBindings,
    _bind_runtime_values, _resolved_type_shape, _validate_tensor_metadata_against_type,
)
from .formula_v3 import (
    NEURAL_PLASTICITY_OUTER_V2_ATOM_REF,
    NEURAL_PLASTICITY_POLYNOMIAL_ATOM_REF,
    NEURAL_PLASTICITY_TRANSPORT_ATOM_REF,
)


class FormulaDeviceDispatchPacket(NamedTuple):
    """A complete K-wide packet ordered by a prepared execution group.

    ``source_rows`` and ``source_lanes`` identify the original ``[R, K]``
    position.  Invalid entries are retained at the tail with ``valid=False``;
    no selected row disappears during packing.  ``inverse_order`` restores the
    original flattened order.
    """

    candidate_ids: Tensor
    source_rows: Tensor
    source_lanes: Tensor
    group_ids: Tensor
    valid: Tensor
    group_counts: Tensor
    group_offsets: Tensor
    inverse_order: Tensor
    well_formed: Tensor


class FormulaDeviceRoutedDecisionResult(NamedTuple):
    """Decision tensors and their device-resident grouped dispatch packet."""

    eligible: Tensor
    scores: Tensor
    selected_candidates: Tensor
    input_finite: Tensor
    score_finite: Tensor
    coverage_satisfied: Tensor
    masked_logits: Tensor
    score_is_double: Tensor
    logits_finite: Tensor
    candidate_ids: Tensor
    source_rows: Tensor
    source_lanes: Tensor
    group_ids: Tensor
    valid: Tensor
    group_counts: Tensor
    group_offsets: Tensor
    inverse_order: Tensor
    well_formed: Tensor


class FormulaDeviceNumericResult(NamedTuple):
    """Packed numerical results before pool allocation and frame advance."""

    output_values: Tensor | tuple[Tensor, ...]
    output_present: Tensor | tuple[Tensor, ...]
    output_alias_handles: Tensor
    bank_successors: Tensor | tuple[Tensor, ...]
    bank_successor_present: Tensor | tuple[Tensor, ...]
    numeric_valid: Tensor
    overflow: Tensor


class FormulaDeviceDispatchLayout(nn.Module):
    """Pack every selected action by a static execution-group table.

    The implementation performs one stable device sort over the complete
    ``R * K`` packet.  It never calls ``nonzero`` or converts dynamic counts to
    Python.  Group capacities and spill policy belong to the following numeric
    execution region; this layer only provides exact counts and offsets.
    """

    def __init__(self, candidate_group_ids: Tensor | Sequence[int]) -> None:
        super().__init__()
        groups = torch.as_tensor(candidate_group_ids, dtype=torch.int64, device="cpu")
        if groups.ndim != 1 or groups.numel() == 0:
            raise ValueError("candidate_group_ids must be a non-empty vector")
        if bool((groups < 0).any()):
            raise ValueError("candidate_group_ids must be non-negative")
        unique = torch.unique(groups, sorted=True)
        expected = torch.arange(unique.numel(), dtype=torch.int64, device="cpu")
        if not torch.equal(unique, expected):
            raise ValueError("candidate_group_ids must use contiguous group ids")
        self.register_buffer("candidate_group_ids", groups, persistent=False)
        self.group_count = int(unique.numel())

    @property
    def candidate_count(self) -> int:
        return int(self.candidate_group_ids.numel())

    def forward(self, selected_candidates: Tensor) -> FormulaDeviceDispatchPacket:
        if selected_candidates.ndim != 2 or selected_candidates.dtype is not torch.int64:
            raise TypeError("selected_candidates must be an int64 [rows, K] tensor")
        rows, width = selected_candidates.shape
        flat = selected_candidates.reshape(-1)
        positions = torch.arange(flat.numel(), device=flat.device)
        in_range = (flat >= 0) & (flat < self.candidate_count)
        well_formed = ((flat == -1) | in_range).all()
        safe = flat.clamp(0, self.candidate_count - 1)
        groups = self.candidate_group_ids.index_select(0, safe)
        invalid_group = groups.new_full((), self.group_count)
        dispatch_group = torch.where(in_range, groups, invalid_group)

        # The position suffix makes the key unique and preserves source order
        # within each group without relying on backend-specific unstable ties.
        key = dispatch_group * (flat.numel() + 1) + positions
        order = torch.argsort(key, stable=True)
        ordered_positions = positions.index_select(0, order)
        ordered_valid = in_range.index_select(0, order)
        ordered_groups = dispatch_group.index_select(0, order)

        counts = torch.zeros(
            self.group_count + 1, dtype=torch.int64, device=flat.device,
        ).scatter_add(0, dispatch_group, torch.ones_like(dispatch_group))
        group_counts = counts[: self.group_count]
        group_offsets = torch.cat((group_counts.new_zeros(1), group_counts.cumsum(0)))
        inverse = torch.empty_like(order).scatter(0, order, positions)
        return FormulaDeviceDispatchPacket(
            flat.index_select(0, order),
            ordered_positions // width,
            ordered_positions % width,
            ordered_groups,
            ordered_valid,
            group_counts,
            group_offsets,
            inverse,
            well_formed,
        )


def _query_graph(query: FormulaProgramQueryV5) -> tuple[FormulaProgramQueryV5, ...]:
    queries: list[FormulaProgramQueryV5] = []
    visited: set[int] = set()

    def visit(current: FormulaProgramQueryV5) -> None:
        if id(current) in visited:
            return
        visited.add(id(current))
        queries.append(current)
        for candidate in current.candidates:
            if isinstance(candidate, FormulaProgramCallCandidateV1):
                visit(candidate.child)

    visit(query)
    return tuple(queries)


def _candidate_group_key(candidate: object) -> Hashable:
    if isinstance(candidate, FormulaProgramEffectCandidateV3):
        count = candidate.execution_count_tensor()
        return (
            "effect",
            candidate.effect_program.program.fingerprint,
            candidate.atom_ref,
            candidate.max_executions,
            count is not None,
            tuple(sorted(candidate.batch_broadcast_operands)),
        )
    if isinstance(candidate, (FormulaProgramTensorCandidateV3, FormulaProgramTensorCandidateV4)):
        return (
            "ordinary",
            candidate.candidate.program.fingerprint,
            len(candidate.output_slot_ids),
            candidate.plastic_bank_slot,
            tuple(sorted(candidate.candidate.batch_broadcast_operands)),
        )
    if isinstance(candidate, FormulaProgramCallCandidateV1):
        return "call"
    raise TypeError(f"unsupported Formula candidate {type(candidate).__name__}")


def formula_device_dispatch_groups(
    query: FormulaProgramQueryV5,
) -> tuple[tuple[int, ...], tuple[Hashable, ...]]:
    """Prepare contiguous execution groups in frame-candidate order."""
    keys: list[Hashable] = []
    ids: list[int] = []
    group_by_key: dict[Hashable, int] = {}
    for current in _query_graph(query):
        for candidate in current.candidates:
            key = _candidate_group_key(candidate)
            if key not in group_by_key:
                group_by_key[key] = len(keys)
                keys.append(key)
            ids.append(group_by_key[key])
        key = "stop"
        if key not in group_by_key:
            group_by_key[key] = len(keys)
            keys.append(key)
        ids.append(group_by_key[key])
    return tuple(ids), tuple(keys)


def _candidate_records(
    query: FormulaProgramQueryV5,
) -> tuple[tuple[int, FormulaProgramQueryV5, object], ...]:
    records: list[tuple[int, FormulaProgramQueryV5, object]] = []
    candidate_id = 0
    for current in _query_graph(query):
        for candidate in current.candidates:
            records.append((candidate_id, current, candidate))
            candidate_id += 1
        candidate_id += 1  # STOP is the final action of every Query.
    return tuple(records)


def _operand_store(candidate: object):
    if isinstance(candidate, FormulaProgramEffectCandidateV3):
        return candidate.operand_store
    if isinstance(candidate, (FormulaProgramTensorCandidateV3, FormulaProgramTensorCandidateV4)):
        return candidate.candidate.operand_store
    raise TypeError("control candidates have no Formula operands")


def _operand(candidate: object, name: str) -> Tensor:
    return _operand_store(candidate).tensor(name)


@dataclass(frozen=True)
class _TypedProgramShape:
    bindings: tuple[int, ...]
    outputs: tuple[int, ...]
    target: int
    batch: int


@dataclass(frozen=True)
class _EffectTapeSpec:
    group_id: int
    counted: bool
    capacity: int


class _FormulaDeviceProgramGroup(nn.Module):
    """One statically compatible numerical/effect Formula group."""

    def __init__(
        self,
        group_id: int,
        candidate_count: int,
        records: Sequence[tuple[int, FormulaProgramQueryV5, object]],
        frame_kernel: FormulaDeviceFrameKernel,
        *,
        bank_positions: dict,
        capacity: int | None = None,
    ) -> None:
        super().__init__()
        if not records:
            raise ValueError("a numerical dispatch group cannot be empty")
        if capacity is not None and (type(capacity) is not int or capacity <= 0):
            raise ValueError("group capacity must be positive or None")
        first = records[0][2]
        if not isinstance(first, (
            FormulaProgramTensorCandidateV3,
            FormulaProgramTensorCandidateV4,
            FormulaProgramEffectCandidateV3,
        )):
            raise TypeError("a numerical group requires Formula candidates")
        checked = _checked_plan(_program(first))
        if checked is None:
            raise ValueError("candidate group cannot use the checked Formula executor")
        if any(
            _checked_plan(_program(candidate)) is None
            or _checked_plan(_program(candidate)).program_fingerprint != checked.program_fingerprint
            for _global_id, _query, candidate in records
        ):
            raise ValueError("all candidates in a group must share one checked Formula plan")

        self.group_id = group_id
        self.capacity = capacity
        self.is_effect = isinstance(first, FormulaProgramEffectCandidateV3)
        if any(isinstance(candidate, FormulaProgramEffectCandidateV3) != self.is_effect
               for _global_id, _query, candidate in records):
            raise ValueError("ordinary and effect candidates require separate groups")
        self.candidates = nn.ModuleList(tuple(record[2] for record in records))
        self.plan = FormulaExecutionPlanV2(checked.program)
        self.output_count = 1 if self.is_effect else len(first.output_slot_ids)

        global_ids = torch.tensor(
            [record[0] for record in records], dtype=torch.int64, device="cpu",
        )
        global_to_local = torch.zeros(candidate_count, dtype=torch.int64, device="cpu")
        global_to_local[global_ids] = torch.arange(
            len(records), dtype=torch.int64, device="cpu",
        )
        self.register_buffer("global_to_local", global_to_local, persistent=False)
        self.register_buffer("global_ids", global_ids, persistent=False)

        binding_kinds: list[str] = []
        binding_sources: list[Tensor] = []
        broadcast: list[bool] = []
        for binding in self.plan.program.bindings:
            kinds: list[str] = []
            sources: list[int] = []
            broadcasts: list[bool] = []
            for global_id, owner_query, candidate in records:
                if isinstance(binding, InputBinding):
                    kinds.append("input")
                    sources.append(owner_query.slot_ids.index(candidate.input_slots[binding.name]))
                    broadcasts.append(False)
                elif isinstance(binding, BankBinding):
                    dynamic = (
                        not self.is_effect
                        and candidate.plastic_bank_slot == binding.name
                    )
                    if dynamic:
                        kinds.append("bank")
                        bank_index = bank_positions.get(candidate.bank_slot_ref, -1)
                        if bank_index < 0:
                            raise ValueError("plastic Formula binding has no prepared Bank owner")
                        sources.append(bank_index)
                    else:
                        kinds.append("operand")
                        sources.append(-1)
                    broadcasts.append(binding.name in (
                        candidate.batch_broadcast_operands
                        if self.is_effect else candidate.candidate.batch_broadcast_operands
                    ))
                else:
                    raise TypeError("unsupported Formula binding kind")
            if len(set(kinds)) != 1 or len(set(broadcasts)) != 1:
                raise ValueError("one dispatch group must share binding-source semantics")
            binding_kinds.append(kinds[0])
            binding_sources.append(torch.tensor(sources, dtype=torch.int64, device="cpu"))
            broadcast.append(broadcasts[0])
        self.binding_kinds = tuple(binding_kinds)
        for index, binding in enumerate(self.plan.program.bindings):
            self.register_buffer(f"binding_port_{index:04d}", torch.tensor([
                tuple(candidate.input_slots).index(binding.name)
                if isinstance(binding, InputBinding) else -1
                for candidate in self.candidates
            ], dtype=torch.int64, device="cpu"), persistent=False)
        self.binding_broadcast = tuple(broadcast)
        for index, sources in enumerate(binding_sources):
            self.register_buffer(f"binding_source_{index:04d}", sources, persistent=False)
        operand_representatives = []
        for index, (name, kind) in enumerate(zip(self.plan.binding_names, self.binding_kinds, strict=True)):
            representatives, rows, owners = [], [], {}
            if kind == "operand":
                # Expanded wiring shares a store; equal independent Banks do not.
                for candidate_index, candidate in enumerate(self.candidates):
                    owner = id(_operand_store(candidate))
                    if owner not in owners:
                        owners[owner] = len(representatives)
                        representatives.append(candidate_index)
                    rows.append(owners[owner])
                if 1 < len(representatives) < len(self.candidates):
                    sample = _operand(self.candidates[0], name)
                    saved_bytes = (len(self.candidates) - len(representatives)) * sample.numel() * sample.element_size()
                    if saved_bytes <= 8 * len(self.candidates):
                        representatives = list(range(len(self.candidates)))
                self.register_buffer(
                    f"operand_table_{index:04d}",
                    torch.stack(tuple(_operand(self.candidates[i], name).detach() for i in representatives)),
                    persistent=False,
                )
                if 1 < len(representatives) < len(self.candidates):
                    self.register_buffer(f"operand_row_{index:04d}",
                                         torch.tensor(rows, dtype=torch.int64, device="cpu"),
                                         persistent=False)
            operand_representatives.append(tuple(representatives))
        self.operand_representatives = tuple(operand_representatives)

        self.effect_plan: _TensorEffectPlan | None = None
        self.effect_counted = False
        self.effect_input_index = 0
        if self.is_effect:
            assert isinstance(first, FormulaProgramEffectCandidateV3)
            self.effect_input_index = self.plan.binding_names.index(first.effect_program.data_input_name)
            instruction = first.effect_program.effect_instruction
            attributes = dict(instruction.attributes)
            axis = 0
            if first.atom_ref in {
                NEURAL_PLASTICITY_TRANSPORT_ATOM_REF,
                NEURAL_PLASTICITY_POLYNOMIAL_ATOM_REF,
            }:
                axis = first.effect_program.state_type.axis_names.index(attributes["state_axis"])
            maximum = (
                attributes["max_executions"]
                if first.atom_ref == NEURAL_PLASTICITY_OUTER_V2_ATOM_REF else 1
            )
            counted = first.execution_count_tensor() is not None and (
                first.atom_ref != NEURAL_PLASTICITY_OUTER_V2_ATOM_REF
            )
            if any(
                (candidate.execution_count_tensor() is not None
                 and candidate.atom_ref != NEURAL_PLASTICITY_OUTER_V2_ATOM_REF) != counted
                for _global_id, _query, candidate in records
            ):
                raise ValueError("effect group must share execution-count semantics")
            self.effect_counted = counted
            self.effect_plan = _TensorEffectPlan(
                first.atom_ref, axis, int(maximum), first.max_executions, counted,
            )
            if counted:
                self.register_buffer("execution_counts", torch.stack(tuple(
                    candidate.execution_count_tensor().detach() for candidate in self.candidates
                )), persistent=False)

    @torch.no_grad()
    def refresh_operands_(self) -> None:
        for index, (name, kind) in enumerate(zip(self.plan.binding_names, self.binding_kinds, strict=True)):
            if kind == "operand":
                table = getattr(self, f"operand_table_{index:04d}")
                values = tuple(_operand(self.candidates[i], name) for i in self.operand_representatives[index])
                if any(value.shape != table.shape[1:] or value.dtype != table.dtype
                       or value.device != table.device for value in values):
                    raise ValueError("operand metadata changed; rebuild the prepared dispatch before search")
                torch.stack(
                    values, out=table,
                )
        if self.effect_counted:
            torch.stack(tuple(candidate.execution_count_tensor() for candidate in self.candidates),
                        out=self.execution_counts)

    def _operand_lanes(self, binding_index: int, table: Tensor, local: Tensor) -> Tensor:
        owner_count = len(self.operand_representatives[binding_index])
        if owner_count == 1:
            return table.expand(local.shape[0], *table.shape[1:])
        rows = (local if owner_count == len(self.candidates) else
                getattr(self, f"operand_row_{binding_index:04d}").index_select(0, local))
        return table.index_select(0, rows)

    def operand_finite(self) -> Tensor:
        """Candidate flags from the current prepared snapshot, not unique rows."""
        local = torch.arange(len(self.candidates), device=self.global_ids.device)
        valid = torch.ones_like(local, dtype=torch.bool)
        for index, kind in enumerate(self.binding_kinds):
            if kind == "operand":
                table = getattr(self, f"operand_table_{index:04d}")
                finite = torch.isfinite(table.reshape(table.shape[0], -1)).all(1)
                valid = valid & self._operand_lanes(index, finite, local)
        return valid

    def preflight_shape(self, data_sample: Tensor, bank_sample: Tensor) -> tuple[tuple[int, ...], ...]:
        """Admit a homogeneous bucket through the native Formula metadata path."""
        candidate = self.candidates[0]
        program = _program(candidate)
        inputs, banks = {}, {}
        for index, (binding, kind, broadcast) in enumerate(zip(
            program.bindings, self.binding_kinds, self.binding_broadcast, strict=True,
        )):
            if kind == "input":
                value = data_sample
            elif kind == "bank":
                value = bank_sample
            else:
                value = getattr(self, f"operand_table_{index:04d}")[0]
                if broadcast:
                    value = value.expand(data_sample.shape[0], *value.shape[1:])
            if isinstance(binding, InputBinding):
                inputs[binding.name] = value
            else:
                banks[binding.name] = binding.bind(value)
        _, axes = _bind_runtime_values(program, inputs=inputs, banks=banks, defer_finite=True)
        if self.is_effect:
            _validate_tensor_metadata_against_type(
                bank_sample, candidate.effect_program.state_type, name="predecessor Bank",
            )
        return tuple(_resolved_type_shape(program.slot_types[name], axes, name=name)
                     for name in program.outputs)

    @staticmethod
    def _gather_pool(pool: Tensor, handles: Tensor) -> Tensor:
        dummy = torch.zeros_like(pool[:1])
        padded = torch.cat((pool, dummy), dim=0)
        safe = torch.where(handles >= 0, handles, handles.new_full((), pool.shape[0]))
        return padded.index_select(0, safe.clamp(0, pool.shape[0]))

    def typed_shapes(self, data_layout, bank_layout):
        """Resolve finite shape variants with the canonical binding/ATen plan."""
        choices = tuple(
            range(len(data_layout.shapes)) if kind == "input" else
            range(len(bank_layout.shapes)) if kind == "bank" else (-1,)
            for kind in self.binding_kinds
        )
        targets = range(len(bank_layout.shapes)) if self.is_effect else (-1,)
        variants = []
        for buckets, target in product(product(*choices), targets):
            values = []
            first_input = next((i for i, kind in enumerate(self.binding_kinds) if kind == "input"), None)
            input_shape = () if first_input is None else data_layout.shapes[buckets[first_input]]
            batch = input_shape[0] if input_shape else 1
            for index, (kind, bucket, broadcast) in enumerate(zip(
                self.binding_kinds, buckets, self.binding_broadcast, strict=True,
            )):
                if kind == "operand":
                    value = getattr(self, f"operand_table_{index:04d}")[0].to("meta")
                    if broadcast:
                        value = value.expand(batch, *value.shape[1:])
                else:
                    layout = data_layout if kind == "input" else bank_layout
                    value = torch.empty(layout.shapes[bucket], dtype=layout.dtypes[bucket], device="meta")
                values.append(value)
            inputs, banks = {}, {}
            program = _program(self.candidates[0])
            for binding, value in zip(program.bindings, values, strict=True):
                if isinstance(binding, InputBinding):
                    inputs[binding.name] = value
                else:
                    banks[binding.name] = binding.bind(value)
            try:
                _bind_runtime_values(program, inputs=inputs, banks=banks, defer_finite=True)
                if self.is_effect:
                    previous = torch.empty(bank_layout.shapes[target], dtype=bank_layout.dtypes[target], device="meta")
                    _validate_tensor_metadata_against_type(
                        previous, self.candidates[0].effect_program.state_type, name="predecessor Bank",
                    )
                outputs, _ = self.plan.forward_checked(PreparedFormulaBindings(
                    self.plan.program_fingerprint, self.plan.binding_names, tuple(values),
                ))
            except FormulaBindingError:
                continue
            if self.is_effect:
                output_buckets = (buckets[self.effect_input_index],)
            else:
                try:
                    output_buckets = tuple(data_layout.index(value) for value in outputs)
                except ValueError as error:
                    raise FormulaBindingError(
                        "FF_DEVICE_MISSING_OUTPUT_POOL",
                        "a legal Formula output has no prepared shape/dtype pool",
                    ) from error
            variants.append(_TypedProgramShape(tuple(buckets), output_buckets, target, batch))
        return tuple(variants)

    @staticmethod
    def _sanitize(value: Tensor, active: Tensor) -> Tensor:
        mask = active.reshape((active.shape[0],) + (1,) * (value.ndim - 1))
        return torch.where(mask, value, torch.zeros_like(value))

    def effect_operand_samples(self, data_pool, bank_pool, shape, data_layout, bank_layout):
        """Resolve tape metadata through the existing checked Formula plan."""
        values = []
        for index, (kind, broadcast) in enumerate(zip(self.binding_kinds, self.binding_broadcast, strict=True)):
            if kind == "operand":
                value = getattr(self, f"operand_table_{index:04d}")[0].to("meta")
                if broadcast:
                    batch = data_pool.shape[1] if shape is None else shape.batch
                    value = value.expand(batch, *value.shape[1:])
            elif shape is None:
                pool = data_pool if kind == "input" else bank_pool
                value = torch.empty(pool.shape[1:], dtype=pool.dtype, device="meta")
            else:
                layout = data_layout if kind == "input" else bank_layout
                bucket = shape.bindings[index]
                value = torch.empty(layout.shapes[bucket], dtype=layout.dtypes[bucket], device="meta")
            values.append(value)
        outputs, _ = self.plan.forward_checked(PreparedFormulaBindings(
            self.plan.program_fingerprint, self.plan.binding_names, tuple(values),
        ))
        available = dict(zip(self.plan.binding_names, values, strict=True))
        available.update(zip(self.plan.program.outputs, outputs, strict=True))
        return tuple(available[name] for name in self.candidates[0].effect_program.effect_instruction.input_slots[1:])

    def forward(
        self,
        packet_candidate_ids: Tensor,
        packet_source_rows: Tensor,
        packet_group_ids: Tensor,
        packet_valid: Tensor,
        group_offsets: Tensor,
        group_counts: Tensor,
        current_value_handles: Tensor,
        current_producer_bank_handles: Tensor,
        branch_bank_handles: Tensor,
        data_pool: Tensor | tuple[Tensor, ...],
        bank_pool: Tensor | tuple[Tensor, ...],
        shape: _TypedProgramShape | None = None,
        data_layout: FormulaDevicePoolLayout | None = None,
        bank_layout: FormulaDevicePoolLayout | None = None,
        effect_tape: tuple[Tensor, ...] | None = None,
        effect_tape_position: Tensor | None = None,
        input_sources=None,
    ) -> tuple[Tensor, Tensor, tuple[Tensor, ...], Tensor, Tensor, Tensor]:
        packet_size = packet_candidate_ids.shape[0]
        capacity = packet_size if self.capacity is None else self.capacity
        lane = torch.arange(capacity, device=packet_candidate_ids.device)
        count = group_counts[self.group_id]
        positions = group_offsets[self.group_id] + lane
        safe_positions = positions.clamp(0, packet_size - 1)
        candidate_ids = packet_candidate_ids.index_select(0, safe_positions)
        source_rows = packet_source_rows.index_select(0, safe_positions)
        active = (
            (lane < count)
            & packet_valid.index_select(0, safe_positions)
            & packet_group_ids.index_select(0, safe_positions).eq(self.group_id)
        )
        safe_candidate = candidate_ids.clamp(0, self.global_to_local.shape[0] - 1)
        local = self.global_to_local.index_select(0, safe_candidate)
        safe_rows = source_rows.clamp(0, current_value_handles.shape[0] - 1)
        row_values = current_value_handles.index_select(0, safe_rows)
        row_producer_banks = current_producer_bank_handles.index_select(0, safe_rows)
        row_banks = branch_bank_handles.index_select(0, safe_rows)
        port_values = None
        port_banks = None
        if input_sources is not None:
            port_values = input_sources.handles.index_select(0, safe_positions)
            port_banks = input_sources.bank_handle.index_select(0, safe_positions)
            active = active & (input_sources.ready & input_sources.finite).all(-1).index_select(0, safe_positions)

        def input_handles(index, slots):
            if port_values is not None:
                ports = getattr(self, f"binding_port_{index:04d}").index_select(0, local)
                return port_values.gather(1, ports[:, None]).squeeze(1)
            return row_values.gather(1, slots[:, None]).squeeze(1)

        def predecessor_handles(slots):
            if port_banks is not None:
                ports = getattr(self, f"binding_port_{self.effect_input_index:04d}").index_select(0, local)
                return port_banks.gather(1, ports[:, None]).squeeze(1)
            return row_producer_banks.gather(1, slots[:, None]).squeeze(1)

        if shape is not None:
            for index, (kind, bucket) in enumerate(zip(self.binding_kinds, shape.bindings, strict=True)):
                sources = getattr(self, f"binding_source_{index:04d}").index_select(0, local)
                if kind == "input":
                    handles = input_handles(index, sources)
                    active = active & data_layout.contains(handles, bucket)
                elif kind == "bank":
                    handles = row_banks.gather(1, sources[:, None]).squeeze(1)
                    active = active & bank_layout.contains(handles, bucket)
            if self.is_effect:
                sources = getattr(self, f"binding_source_{self.effect_input_index:04d}").index_select(0, local)
                targets = predecessor_handles(sources)
                active = active & bank_layout.contains(targets, shape.target)

        bindings: list[Tensor] = []
        for binding_index, (name, kind, do_broadcast) in enumerate(zip(
            self.plan.binding_names,
            self.binding_kinds,
            self.binding_broadcast,
            strict=True,
        )):
            sources = getattr(self, f"binding_source_{binding_index:04d}").index_select(0, local)
            if kind == "input":
                handles = input_handles(binding_index, sources)
                value = (self._gather_pool(data_pool, handles) if shape is None else
                         data_layout.gather(data_pool, handles, shape.bindings[binding_index]))
            elif kind == "bank":
                handles = row_banks.gather(1, sources[:, None]).squeeze(1)
                value = (self._gather_pool(bank_pool, handles) if shape is None else
                         bank_layout.gather(bank_pool, handles, shape.bindings[binding_index]))
            else:
                table = getattr(self, f"operand_table_{binding_index:04d}")
                value = self._operand_lanes(binding_index, table, local)
                if do_broadcast:
                    batch = data_pool.shape[1] if shape is None else shape.batch
                    value = value.expand(value.shape[0], batch, *value.shape[2:])
            bindings.append(self._sanitize(value, active))

        def run(*values: Tensor) -> tuple[tuple[Tensor, ...], Tensor]:
            return self.plan.forward_checked(PreparedFormulaBindings(
                self.plan.program_fingerprint,
                self.plan.binding_names,
                values,
            ))

        outputs, finite = torch.vmap(run, randomness="error")(*bindings)
        finite = finite & active
        alias = torch.full(
            (capacity,), -1, dtype=torch.int64, device=packet_candidate_ids.device,
        )
        target_pool = bank_pool if shape is None else bank_pool[max(shape.target, 0)]
        successor = target_pool.new_zeros((capacity, *target_pool.shape[1:]))
        if self.is_effect:
            originals = dict(zip(self.plan.binding_names, bindings, strict=True))
            numeric = dict(zip(self.plan.program.outputs, outputs, strict=True))
            instruction = self.candidates[0].effect_program.effect_instruction
            effect_operands = tuple(
                originals[name] if name in originals else numeric[name]
                for name in instruction.input_slots[1:]
            )
            effect_slots = getattr(self, f"binding_source_{self.effect_input_index:04d}").index_select(0, local)
            effect_handles = input_handles(self.effect_input_index, effect_slots)
            alias = torch.where(active, effect_handles, alias)
            target_handles = predecessor_handles(effect_slots)
            previous = (self._gather_pool(bank_pool, target_handles) if shape is None else
                        bank_layout.gather(bank_pool, target_handles, shape.target))
            previous = self._sanitize(previous, active)
            if self.effect_counted:
                counts = self.execution_counts.index_select(0, local)
            else:
                counts = previous.new_ones(capacity)
            assert self.effect_plan is not None
            successor, effect_finite = self.effect_plan(previous, counts, *effect_operands)
            finite = finite & effect_finite
            successor = self._sanitize(successor, finite)
            if effect_tape is not None:
                recorded = (
                    torch.where(active & finite, source_rows, -1),
                    *effect_operands,
                    *((counts,) if self.effect_counted else ()),
                )
                for target, value in zip(effect_tape, recorded, strict=True):
                    target[effect_tape_position] = value.unsqueeze(0)
        return positions, active, outputs, finite, alias, successor


class FormulaDeviceNumericalDispatch(nn.Module):
    """Execute packed Formula groups using homogeneous or typed value pools.

    This first connected implementation reserves the complete packet capacity
    for every static group.  It is dropless and establishes semantic parity;
    per-group capacities and spill buckets are a subsequent physical-efficiency
    optimization measured against this exact path.

    Immutable Formula operands share packed rows within each group when this
    reduces storage, with device-side candidate-to-row indices when needed. After changing
    model weights, call ``refresh_operands_`` at the search boundary, not in a
    wave. Mutable predecessor Bank values always come from the live Bank pool.
    ``prepare_typed_pools_`` prepares heterogeneous numerical execution; capture
    it with ``FormulaDeviceCapturedExecutionWave``. The older standalone numeric
    capture helper is homogeneous; typed search preparation must supply every
    reachable output shape/dtype pool.
    """

    def __init__(
        self,
        query: FormulaProgramQueryV5,
        frame_kernel: FormulaDeviceFrameKernel,
        *,
        group_capacities: Sequence[int | None] | None = None,
        execution_backend: str = "auto",
    ) -> None:
        super().__init__()
        if execution_backend not in {"auto", "native"}:
            raise ValueError("execution_backend must be auto or native")
        self.execution_backend = execution_backend
        self._preparing_compilation = False
        self._automatic_ready = False
        group_ids, keys = formula_device_dispatch_groups(query)
        records = _candidate_records(query)
        if len(group_ids) != frame_kernel.candidate_count:
            raise ValueError("query and frame candidate orders do not match")
        if group_capacities is None:
            capacities = (None,) * len(keys)
        else:
            capacities = tuple(group_capacities)
            if len(capacities) != len(keys):
                raise ValueError("group_capacities must cover every dispatch group")
        grouped: dict[int, list[tuple[int, FormulaProgramQueryV5, object]]] = {}
        for record in records:
            grouped.setdefault(group_ids[record[0]], []).append(record)
        modules = []
        numeric_group_ids = []
        bank_positions = {ref: index for index, ref in enumerate(query.initial_bank_state().slot_refs)}
        for group_id, rows in grouped.items():
            if keys[group_id] in {"call", "stop"}:
                continue
            modules.append(_FormulaDeviceProgramGroup(
                group_id,
                frame_kernel.candidate_count,
                rows,
                frame_kernel,
                bank_positions=bank_positions,
                capacity=capacities[group_id],
            ))
            numeric_group_ids.append(group_id)
        self.groups = nn.ModuleList(modules)
        self.frame_kernel = frame_kernel
        self.max_outputs = frame_kernel.spec.max_outputs
        self.group_count = len(keys)
        self.numeric_group_ids = tuple(numeric_group_ids)
        self.shape_excluded_groups: frozenset[int] = frozenset()
        self.data_layout: FormulaDevicePoolLayout | None = None
        self.bank_layout: FormulaDevicePoolLayout | None = None
        self.typed_variants: tuple[tuple[_TypedProgramShape, ...], ...] = ()
        self.effect_tape_specs: tuple[_EffectTapeSpec, ...] = ()
        self._compiled_groups = {}
        self._compiled_dispatch = False

    def _apply(self, fn, recurse=True):
        if self._automatic_ready:
            self._compiled_groups = {}
            self._compiled_dispatch = False
            self._automatic_ready = False
        if self._compiled_groups:
            self._compiled_groups = None
        if self._compiled_dispatch:
            self._compiled_dispatch = None
        return super()._apply(fn, recurse=recurse)

    def prepare_compiled_dispatch_(self, sample_run, *, backend="inductor"):
        """Prepare the ordinary typed dispatch as one graph, before capture."""
        from ._formula_device_group_compile import prepare_dispatch
        self._automatic_ready = False
        previous = self._preparing_compilation
        self._preparing_compilation = True
        try:
            return prepare_dispatch(self, sample_run, backend)
        finally:
            self._preparing_compilation = previous

    def prepare_compiled_groups_(self, sample_run, *, group_ids, backend="inductor"):
        """Compile ordinary typed groups before capture using a scratch run.

        The callback must own its pools/frames and exercise every required call
        shape. Values and mutable predecessor Banks remain runtime inputs.
        Reprepare after changing metadata/layout; refresh operand values in
        place before reusing the compiled groups and any outer CUDA Graph.
        """
        from ._formula_device_group_compile import prepare_groups
        if self.data_layout is None:
            raise ValueError("compiled groups require prepared typed pools")
        self._automatic_ready = False
        previous = self._preparing_compilation
        self._preparing_compilation = True
        try:
            return prepare_groups(self, sample_run, group_ids, backend)
        finally:
            self._preparing_compilation = previous

    def allocate_effect_tape(self, steps, packet_size, data_pool, bank_pool):
        """Optional event-owned operand tape; never a module-global scratch state."""
        device = data_pool.device if self.data_layout is None else data_pool[0].device
        tape, specs = [], []
        for index, group in enumerate(self.groups):
            if not group.is_effect or group.group_id in self.shape_excluded_groups:
                continue
            variants = (None,) if self.data_layout is None else self.typed_variants[index]
            capacity = packet_size if group.capacity is None else group.capacity
            for shape in variants:
                samples = group.effect_operand_samples(
                    data_pool, bank_pool, shape, self.data_layout, self.bank_layout,
                )
                if group.effect_counted:
                    samples = (*samples, group.execution_counts[0])
                tape.append((
                    torch.full((steps, capacity), -1, dtype=torch.int64, device=device),
                    *(torch.empty((steps, capacity, *v.shape), dtype=v.dtype, device=device) for v in samples),
                ))
                specs.append(_EffectTapeSpec(group.group_id, group.effect_counted, capacity))
        self.effect_tape_specs = tuple(specs)
        return tuple(tape)

    def prepare_typed_pools_(self, data_layout, bank_layout) -> None:
        """Prepare shape variants once; values keep their actual physical shapes.

        A missing legal output pool is an explicit preparation failure, not a
        reason to remove that candidate. Unsupported structured Formula atoms
        still require native execution through the existing checked-plan gate.
        """
        variants = tuple(group.typed_shapes(data_layout, bank_layout) for group in self.groups)
        self.data_layout, self.bank_layout = data_layout, bank_layout
        self.typed_variants = variants
        self._compiled_groups = {}
        self._compiled_dispatch = False

    def typed_admission(self, values, producer_bank_handles, bank_handles, input_sources=None):
        """Match runtime handles to the same shape variants used by execution."""
        allowed = torch.ones((values.shape[0], self.frame_kernel.candidate_count),
                             dtype=torch.bool, device=values.device)
        for group, variants in zip(self.groups, self.typed_variants, strict=True):
            matches = torch.zeros((values.shape[0], group.global_ids.shape[0]),
                                  dtype=torch.bool, device=values.device)
            for shape in variants:
                current = torch.ones_like(matches)
                for index, (kind, bucket) in enumerate(zip(group.binding_kinds, shape.bindings, strict=True)):
                    sources = getattr(group, f"binding_source_{index:04d}")
                    if kind == "input":
                        if input_sources is None:
                            handles = values.index_select(1, sources)
                        else:
                            ports = getattr(group, f"binding_port_{index:04d}")
                            handles = input_sources.handles.index_select(1, group.global_ids).gather(
                                2, ports[None, :, None].expand(values.shape[0], -1, 1),
                            ).squeeze(-1)
                        current = current & self.data_layout.contains(handles, bucket)
                    elif kind == "bank":
                        current = current & self.bank_layout.contains(bank_handles.index_select(1, sources), bucket)
                if group.is_effect:
                    sources = getattr(group, f"binding_source_{group.effect_input_index:04d}")
                    if input_sources is None:
                        handles = producer_bank_handles.index_select(1, sources)
                    else:
                        ports = getattr(group, f"binding_port_{group.effect_input_index:04d}")
                        handles = input_sources.bank_handle.index_select(1, group.global_ids).gather(
                            2, ports[None, :, None].expand(values.shape[0], -1, 1),
                        ).squeeze(-1)
                    current = current & self.bank_layout.contains(handles, shape.target)
                matches = matches | current
            allowed = allowed.index_copy(1, group.global_ids, matches)
        return allowed

    def prepare_shape_bucket_(self, data_sample: Tensor, bank_sample: Tensor) -> Tensor:
        """Freeze metadata admission for one shape bucket, before graph capture.

        Inapplicable groups stay in the global action table but are never
        executed. A legal shape-changing output requires a different executor;
        it must not be silently removed from the search space.
        """
        allowed = torch.ones(self.frame_kernel.candidate_count, dtype=torch.bool, device=data_sample.device)
        excluded = set()
        for group in self.groups:
            try:
                shapes = group.preflight_shape(data_sample, bank_sample)
            except FormulaBindingError:
                excluded.add(group.group_id)
                allowed[group.global_ids] = False
                continue
            if not group.is_effect and any(shape != tuple(data_sample.shape) for shape in shapes):
                raise FormulaBindingError(
                    "FF_DEVICE_HETEROGENEOUS_OUTPUT",
                    "a legal Formula output requires another value shape bucket",
                )
        self.shape_excluded_groups = frozenset(excluded)
        return allowed

    @classmethod
    def from_query(
        cls,
        query: FormulaProgramQueryV5,
        *,
        frame_kernel: FormulaDeviceFrameKernel | None = None,
        group_capacities: Sequence[int | None] | None = None,
        execution_backend: str = "auto",
    ) -> "FormulaDeviceNumericalDispatch":
        kernel = FormulaDeviceFrameKernel.from_query(query) if frame_kernel is None else frame_kernel
        return cls(query, kernel, group_capacities=group_capacities, execution_backend=execution_backend)

    @staticmethod
    def _frame(tensor: Tensor, depth: Tensor) -> Tensor:
        index = depth[:, None, None].expand(-1, 1, tensor.shape[-1])
        return tensor.gather(1, index).squeeze(1)

    @torch.no_grad()
    def refresh_operands_(self) -> None:
        """Refresh the no-grad search snapshot without changing graph addresses.

        Canonical parameters and their optimizer remain on the original Query.
        Refresh on the search stream after an optimizer/load step and before
        launching the next search; never interleave it with an active search.
        Changed operand metadata or store sharing requires a rebuilt dispatch;
        device/dtype migration also invalidates any previously captured graph.
        """
        for group in self.groups:
            group.refresh_operands_()

    @staticmethod
    def _normalize_call(state, packet, data_pool, bank_pool, effect_tape=(), effect_tape_position=None, input_sources=None):
        return (FormulaDeviceFrameState(*state), FormulaDeviceDispatchPacket(*packet), data_pool, bank_pool,
                effect_tape, effect_tape_position,
                None if input_sources is None else FormulaDeviceSources(*input_sources))

    @torch.no_grad()
    def forward(
        self,
        state: FormulaDeviceFrameState,
        packet: FormulaDeviceDispatchPacket,
        data_pool: Tensor | tuple[Tensor, ...],
        bank_pool: Tensor | tuple[Tensor, ...],
        effect_tape: tuple[tuple[Tensor, ...], ...] = (),
        effect_tape_position: Tensor | None = None,
        input_sources=None,
    ) -> FormulaDeviceNumericResult:
        args = self._normalize_call(state, packet, data_pool, bank_pool, effect_tape, effect_tape_position, input_sources)
        state, packet, data_pool, bank_pool, effect_tape, effect_tape_position, input_sources = args
        if (self.execution_backend == "auto" and not self._preparing_compilation
                and automatic_cuda_compilation(packet.candidate_ids.device)
                and not torch.compiler.is_compiling() and self.data_layout is not None
                and self._compiled_dispatch is False and not self._compiled_groups
                and not any(group.is_effect for group in self.groups)):
            if automatic_cuda_compilation(packet.candidate_ids.device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("warm up the Formula dispatch outside CUDA capture")
                self._preparing_compilation = True
                try:
                    self.prepare_compiled_dispatch_(lambda: self(*args))
                    self._automatic_ready = True
                finally:
                    self._preparing_compilation = False
        if self._automatic_ready and self._compiled_dispatch is not False and not torch.compiler.is_compiling():
            from ._formula_device_group_compile import _signature
            if _signature(args)[0] not in self._compiled_dispatch.calls:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("warm up the changed Formula signature outside CUDA capture")
                compiled = self._compiled_dispatch
                self._compiled_dispatch = False
                self._preparing_compilation = True
                try:
                    compiled.prepare_variant(args)
                finally:
                    self._compiled_dispatch = compiled
                    self._preparing_compilation = False
        if self._compiled_dispatch is None:
            raise RuntimeError("dispatch moved after compilation; prepare compiled dispatch and recapture")
        if self._compiled_dispatch is not False:
            return self._compiled_dispatch(*args)
        if self.data_layout is not None:
            return self._typed_forward(state, packet, data_pool, bank_pool, effect_tape, effect_tape_position, input_sources)
        if data_pool.ndim < 2 or bank_pool.ndim < 2:
            raise ValueError("data_pool and bank_pool require a capacity axis")
        packet_size = packet.candidate_ids.shape[0]
        depth = state.depth.clamp(0, self.frame_kernel.spec.max_depth - 1)
        current_values = self._frame(state.value_handles, depth)
        current_producer_banks = self._frame(state.producer_bank_handle, depth)

        output_values = data_pool.new_zeros(
            (packet_size, self.max_outputs, *data_pool.shape[1:])
        )
        output_counts = data_pool.new_zeros(
            (packet_size, self.max_outputs), dtype=torch.int64,
        )
        alias_counts = torch.zeros_like(output_counts)
        bank_successors = bank_pool.new_zeros((packet_size, *bank_pool.shape[1:]))
        bank_counts = torch.zeros(packet_size, dtype=torch.int64, device=bank_pool.device)
        invalid_counts = data_pool.new_zeros(packet_size, dtype=torch.int64)
        overflow = torch.zeros((), dtype=torch.bool, device=data_pool.device)

        tapes = iter(effect_tape)
        for group in self.groups:
            if group.group_id in self.shape_excluded_groups:
                # Preparation-time shape decision, never a CUDA scalar branch.
                invalid_counts += (packet.valid & packet.group_ids.eq(group.group_id)).to(torch.int64)
                continue
            positions, active, outputs, finite, alias, successor = group(
                packet.candidate_ids,
                packet.source_rows,
                packet.group_ids,
                packet.valid,
                packet.group_offsets,
                packet.group_counts,
                current_values,
                current_producer_banks,
                state.bank_value_handles,
                data_pool,
                bank_pool,
                effect_tape=next(tapes) if effect_tape and group.is_effect else None,
                effect_tape_position=effect_tape_position,
                input_sources=input_sources,
            )
            capacity = positions.shape[0]
            count = packet.group_counts[group.group_id]
            overflow = overflow | (count > capacity)
            admitted = active & finite
            safe_positions = torch.where(
                active,
                positions,
                positions.new_full((), packet_size),
            )

            output_extended = torch.cat((
                output_values,
                output_values.new_zeros((1, *output_values.shape[1:])),
            ))
            count_extended = torch.cat((
                output_counts,
                output_counts.new_zeros((1, self.max_outputs)),
            ))
            alias_extended = torch.cat((
                alias_counts,
                alias_counts.new_zeros((1, self.max_outputs)),
            ))
            if group.is_effect:
                effect_values = output_values.new_zeros(
                    (capacity, self.max_outputs, *data_pool.shape[1:])
                )
                effect_counts = output_counts.new_zeros((capacity, self.max_outputs))
                effect_alias = alias_counts.new_zeros((capacity, self.max_outputs))
                effect_counts[:, 0] = admitted.to(torch.int64)
                effect_alias[:, 0] = torch.where(admitted, alias + 1, torch.zeros_like(alias))
                output_values = output_extended.index_add(0, safe_positions, effect_values)[:-1]
                output_counts = count_extended.index_add(0, safe_positions, effect_counts)[:-1]
                alias_counts = alias_extended.index_add(0, safe_positions, effect_alias)[:-1]
                bank_extended = torch.cat((bank_successors, bank_successors.new_zeros((1, *bank_successors.shape[1:]))))
                bank_count_extended = torch.cat((bank_counts, bank_counts.new_zeros(1)))
                bank_successors = bank_extended.index_add(
                    0, safe_positions, self.__mask(successor, admitted),
                )[:-1]
                bank_counts = bank_count_extended.index_add(
                    0, safe_positions, admitted.to(torch.int64),
                )[:-1]
            else:
                group_values = output_values.new_zeros(
                    (capacity, self.max_outputs, *data_pool.shape[1:])
                )
                group_counts = output_counts.new_zeros((capacity, self.max_outputs))
                for head, value in enumerate(outputs):
                    group_values[:, head] = self.__mask(value, admitted)
                    group_counts[:, head] = admitted.to(torch.int64)
                output_values = output_extended.index_add(0, safe_positions, group_values)[:-1]
                output_counts = count_extended.index_add(0, safe_positions, group_counts)[:-1]
                alias_counts = alias_extended[:-1]

            invalid_extended = torch.cat((invalid_counts, invalid_counts.new_zeros(1)))
            invalid_counts = invalid_extended.index_add(
                0, safe_positions, (active & ~finite).to(torch.int64),
            )[:-1]

        numeric_valid = packet.valid & invalid_counts.eq(0) & ~overflow
        if input_sources is not None:
            numeric_valid &= (input_sources.ready & input_sources.finite).all(-1)
        return FormulaDeviceNumericResult(
            output_values,
            output_counts > 0,
            alias_counts - 1,
            bank_successors,
            bank_counts > 0,
            numeric_valid,
            overflow,
        )

    @staticmethod
    def __mask(value: Tensor, active: Tensor) -> Tensor:
        mask = active.reshape((active.shape[0],) + (1,) * (value.ndim - 1))
        return torch.where(mask, value, torch.zeros_like(value))

    def _typed_forward(self, state, packet, data_pools, bank_pools, effect_tape=(), effect_tape_position=None, input_sources=None):
        size = packet.candidate_ids.shape[0]
        depth = state.depth.clamp(0, self.frame_kernel.spec.max_depth - 1)
        current = self._frame(state.value_handles, depth)
        producers = self._frame(state.producer_bank_handle, depth)
        values = [pool.new_zeros((size + 1, self.max_outputs, *pool.shape[1:])) for pool in data_pools]
        counts = [torch.zeros((size + 1, self.max_outputs), dtype=torch.int64, device=current.device)
                  for _ in data_pools]
        successors = [pool.new_zeros((size + 1, *pool.shape[1:])) for pool in bank_pools]
        bank_counts = [torch.zeros(size + 1, dtype=torch.int64, device=current.device) for _ in bank_pools]
        aliases = torch.zeros_like(counts[0])
        matches = torch.zeros(size + 1, dtype=torch.int64, device=current.device)
        numeric_row = torch.zeros_like(packet.valid)
        overflow = torch.zeros((), dtype=torch.bool, device=current.device)
        tapes = iter(effect_tape)
        for group, variants in zip(self.groups, self.typed_variants, strict=True):
            numeric_row = numeric_row | packet.group_ids.eq(group.group_id)
            capacity = size if group.capacity is None else group.capacity
            overflow = overflow | (packet.group_counts[group.group_id] > capacity)
            for shape in variants:
                if self._compiled_groups is None:
                    raise RuntimeError("dispatch moved after compilation; prepare compiled groups and recapture")
                run = self._compiled_groups.get(group.group_id, group)
                group_args = (
                    packet.candidate_ids, packet.source_rows, packet.group_ids, packet.valid,
                    packet.group_offsets, packet.group_counts, current, producers,
                    state.bank_value_handles, data_pools, bank_pools,
                    shape, self.data_layout, self.bank_layout,
                    next(tapes) if effect_tape and group.is_effect else None, effect_tape_position,
                    input_sources,
                )
                if (not group.is_effect and self.execution_backend == "auto"
                        and not self._preparing_compilation
                        and automatic_cuda_compilation(packet.candidate_ids.device)
                        and not torch.compiler.is_compiling()):
                    if automatic_cuda_compilation(packet.candidate_ids.device):
                        from ._formula_device_group_compile import _CompiledGroup, _signature
                        if torch.cuda.is_current_stream_capturing():
                            if run is group or _signature(group_args)[0] not in run.calls:
                                raise RuntimeError("warm up the Formula group outside CUDA capture")
                        elif run is group:
                            run = _CompiledGroup(group, (group_args,), "inductor")
                            self._compiled_groups[group.group_id] = run
                            self._automatic_ready = True
                        else:
                            run.prepare_variant(group_args)
                positions, active, outputs, finite, alias, successor = run(*group_args)
                admitted = active & finite
                safe = torch.where(active, positions, positions.new_full((), size))
                matches = matches.index_add(0, safe, admitted.to(torch.int64))
                for head, bucket in enumerate(shape.outputs):
                    head_index = torch.full_like(safe, head)
                    counts[bucket] = counts[bucket].index_put(
                        (safe, head_index), admitted.to(torch.int64), accumulate=True,
                    )
                    if group.is_effect:
                        aliases = aliases.index_put(
                            (safe, head_index), torch.where(admitted, alias + 1, torch.zeros_like(alias)),
                            accumulate=True,
                        )
                    else:
                        values[bucket] = values[bucket].index_put(
                            (safe, head_index), self.__mask(outputs[head], admitted), accumulate=True,
                        )
                if group.is_effect:
                    bucket = shape.target
                    successors[bucket] = successors[bucket].index_add(
                        0, safe, self.__mask(successor, admitted),
                    )
                    bank_counts[bucket] = bank_counts[bucket].index_add(0, safe, admitted.to(torch.int64))
        valid = packet.valid & (~numeric_row | matches[:-1].eq(1)) & ~overflow
        if input_sources is not None:
            valid &= (input_sources.ready & input_sources.finite).all(-1)
        return FormulaDeviceNumericResult(
            tuple(value[:-1] for value in values), tuple(count[:-1] > 0 for count in counts),
            aliases[:-1] - 1, tuple(value[:-1] for value in successors),
            tuple(count[:-1] > 0 for count in bank_counts),
            valid, overflow,
        )


class FormulaDeviceCapturedNumericalDispatch:
    """Replay one fixed-shape numerical dispatch without per-group host launches.

    The mounted Formula graph and tensor addresses are fixed for the lifetime of
    the capture. Runtime routes and values remain mutable tensor contents. This
    is the no-grad search executor; retained training paths use the autograd
    replay backend rather than treating inactive gradients as numerical zeros.
    """

    def __init__(
        self,
        dispatch: FormulaDeviceNumericalDispatch,
        state: FormulaDeviceFrameState,
        packet: FormulaDeviceDispatchPacket,
        data_pool: Tensor,
        bank_pool: Tensor,
        graph: torch.cuda.CUDAGraph,
        result: FormulaDeviceNumericResult,
    ) -> None:
        self.dispatch = dispatch
        self.state = state
        self.packet = packet
        self.data_pool = data_pool
        self.bank_pool = bank_pool
        self.graph: torch.cuda.CUDAGraph | None = graph
        self.result = result

    @classmethod
    def capture(
        cls,
        dispatch: FormulaDeviceNumericalDispatch,
        state: FormulaDeviceFrameState,
        packet: FormulaDeviceDispatchPacket,
        data_pool: Tensor,
        bank_pool: Tensor,
        *,
        warmup_steps: int = 3,
    ) -> "FormulaDeviceCapturedNumericalDispatch":
        if type(warmup_steps) is not int or warmup_steps <= 0:
            raise ValueError("warmup_steps must be a positive integer")
        state = FormulaDeviceFrameState(*state)
        packet = FormulaDeviceDispatchPacket(*packet)
        tensors = (*state, *packet, data_pool, bank_pool)
        devices = {tensor.device for tensor in tensors}
        if len(devices) != 1 or data_pool.device.type != "cuda":
            raise ValueError("captured numerical dispatch requires one CUDA device")
        device = data_pool.device
        static_state = FormulaDeviceFrameState(*(tensor.detach().clone() for tensor in state))
        static_packet = FormulaDeviceDispatchPacket(
            *(tensor.detach().clone() for tensor in packet)
        )
        static_data_pool = data_pool.detach().clone()
        static_bank_pool = bank_pool.detach().clone()
        dispatch.refresh_operands_()

        with torch.no_grad(), torch.cuda.device(device):
            current_stream = torch.cuda.current_stream(device)
            warmup_stream = torch.cuda.Stream(device=device)
            warmup_stream.wait_stream(current_stream)
            with torch.cuda.stream(warmup_stream):
                for _ in range(warmup_steps):
                    dispatch(
                        tuple(static_state),
                        tuple(static_packet),
                        static_data_pool,
                        static_bank_pool,
                    )
            current_stream.wait_stream(warmup_stream)
            torch.cuda.synchronize(device)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                result = dispatch(
                    tuple(static_state),
                    tuple(static_packet),
                    static_data_pool,
                    static_bank_pool,
                )
        return cls(
            dispatch,
            static_state,
            static_packet,
            static_data_pool,
            static_bank_pool,
            graph,
            FormulaDeviceNumericResult(*result),
        )

    @property
    def closed(self) -> bool:
        return self.graph is None

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
        destination: Sequence[Tensor],
        source: Sequence[Tensor],
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

    @torch.no_grad()
    def copy_inputs_(
        self,
        *,
        state: FormulaDeviceFrameState | None = None,
        packet: FormulaDeviceDispatchPacket | None = None,
        data_pool: Tensor | None = None,
        bank_pool: Tensor | None = None,
    ) -> None:
        """Refresh fixed-address inputs outside the captured execution segment."""
        if self.closed:
            raise RuntimeError("captured numerical dispatch is closed")
        if state is not None:
            self._copy_fields(self.state, FormulaDeviceFrameState(*state), name="state")
        if packet is not None:
            self._copy_fields(
                self.packet,
                FormulaDeviceDispatchPacket(*packet),
                name="packet",
            )
        if data_pool is not None:
            self._copy_tensor(self.data_pool, data_pool, name="data_pool")
        if bank_pool is not None:
            self._copy_tensor(self.bank_pool, bank_pool, name="bank_pool")

    @torch.no_grad()
    def replay(self) -> FormulaDeviceNumericResult:
        """Enqueue the complete captured dispatch and return its stable output views."""
        graph = self.graph
        if graph is None:
            raise RuntimeError("captured numerical dispatch is closed")
        graph.replay()
        return self.result

    def close(self) -> bool:
        if self.graph is None:
            return False
        torch.cuda.synchronize(self.data_pool.device)
        self.graph = None
        return True


class FormulaDeviceRoutedDecisionWave(nn.Module):
    """Join device decision and dropless execution-group packing."""

    def __init__(
        self,
        decision: FormulaDeviceDecisionWave,
        dispatch: FormulaDeviceDispatchLayout,
    ) -> None:
        super().__init__()
        if decision.frame_kernel.candidate_count != dispatch.candidate_count:
            raise ValueError("decision and dispatch candidate tables must match")
        self.decision = decision
        self.dispatch = dispatch

    @classmethod
    def from_query(
        cls,
        query: FormulaProgramQueryV5,
        *,
        candidate_family_ids: Tensor | Sequence[int],
        width: int,
        frame_kernel: FormulaDeviceFrameKernel | None = None,
    ) -> "FormulaDeviceRoutedDecisionWave":
        kernel = FormulaDeviceFrameKernel.from_query(query) if frame_kernel is None else frame_kernel
        decision = FormulaDeviceDecisionWave.from_query(
            query,
            candidate_family_ids=candidate_family_ids,
            width=width,
            frame_kernel=kernel,
        )
        group_ids, _keys = formula_device_dispatch_groups(query)
        return cls(decision, FormulaDeviceDispatchLayout(group_ids))

    @torch.no_grad()
    def forward(
        self,
        state: FormulaDeviceFrameState,
        data_pool: Tensor,
        bank_pool: Tensor,
        operand_finite: Tensor,
        parent_scores: Tensor,
        input_sources=None,
    ) -> FormulaDeviceRoutedDecisionResult:
        decision = FormulaDeviceDecisionResult(*self.decision(
            state, data_pool, bank_pool, operand_finite, parent_scores,
            input_sources=input_sources,
        ))
        packet = FormulaDeviceDispatchPacket(*self.dispatch(decision.selected_candidates))
        return FormulaDeviceRoutedDecisionResult(*decision, *packet)


__all__ = [
    "FormulaDeviceCapturedNumericalDispatch",
    "FormulaDeviceDispatchLayout",
    "FormulaDeviceDispatchPacket",
    "FormulaDeviceNumericResult",
    "FormulaDeviceNumericalDispatch",
    "FormulaDeviceRoutedDecisionResult",
    "FormulaDeviceRoutedDecisionWave",
    "formula_device_dispatch_groups",
]
