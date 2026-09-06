"""Opt-in indexed admission for native Query candidate graphs.

Architecture and tensor metadata are prepared separately from current numerical
admission. Only immutable binding indices and metadata verdicts are cached.
"""

from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar

import torch
from torch import nn

from ._formula_finite_rows import _FiniteTensorRows
from .formula_program_query_v4 import FormulaProgramEffectCandidateV3
from .formula_program_query_v5 import FormulaProgramQueryV5
from .formula_v2 import InputBinding, _binding_tensor_metadata, _capture_finite_validation


_ADMISSION_TABLES = ContextVar("arti_indexed_candidate_admission", default=None)


@contextmanager
def indexed_candidate_admission():
    """Prepare one immutable architecture for repeated search/gradient replay.

    Keep candidate programs, bindings, wiring, query budgets and device fixed
    inside this scope. Exit and enter a new scope after editing those fields.
    Bank/operand numerical values and branch state may change on every call;
    none of their numerical admission results are cached. Nested scopes rebuild
    independently and restore their parent's tables on exit.
    """
    token = _ADMISSION_TABLES.set({})
    try:
        yield
    finally:
        _ADMISSION_TABLES.reset(token)


class _AdmissionKernel(nn.Module):
    def __init__(self, query, include_stop):
        super().__init__()
        self.maximum = query.max_steps
        self.minimum = query.min_steps
        self.minimum_tensor = query.min_tensor_steps
        self.maximum_tensor = query.max_tensor_steps
        self.maximum_effect = query.max_effect_steps
        self.include_stop = include_stop
        self.version5 = isinstance(query, FormulaProgramQueryV5)
        self.terminal_closes = False if self.version5 else query._terminal_closes_tensor
        self.terminal_requires = False if self.version5 else query._terminal_requires_tensor

    def forward(self, flags, sources, metadata, occupied, required, empty, kinds, terminal_writes,
                terminal_indices, counts, step):
        ready = (~required[None] | occupied[:, None, :]).all(-1)
        ready = ready & ~(empty[None] & occupied[:, None, :]).any(-1)
        allowed = metadata[None] & flags[:, sources].all(-1) & ready & (step < self.maximum)
        effects = kinds == 1
        tensor_count, effect_count = counts[:, :1], counts[:, 1:2]
        if self.maximum_effect is not None:
            allowed = allowed & (~effects[None] | (effect_count < self.maximum_effect))
        terminal_ready = occupied[:, terminal_indices].all(-1, keepdim=True)
        if self.maximum_tensor is not None:
            remaining = self.maximum_tensor - tensor_count
            allowed = allowed & (effects[None] | (remaining > 0))
            if self.terminal_requires:
                allowed = allowed & (effects[None] | (remaining != 1) | terminal_ready | terminal_writes[None])
        if self.terminal_closes:
            allowed = allowed & (~terminal_writes[None] | (tensor_count + (kinds == 0)[None] >= self.minimum_tensor))
        if self.include_stop:
            stop = terminal_ready & (step >= self.minimum) & (tensor_count >= self.minimum_tensor)
            allowed = torch.cat((allowed, stop), dim=1)
        return allowed


class _AdmissionTable:
    def __init__(self, query, candidates, include_stop):
        self.candidates = tuple(candidates)
        self.query = query
        self.include_stop = include_stop
        self.sources = []
        self.requirements = []
        self.layouts = OrderedDict()
        self.source_groups = {}
        self.kernel = _AdmissionKernel(query, include_stop)
        lookup = {}
        slots = {name: index for index, name in enumerate(query.slot_ids)}
        required, empty, kinds, terminal = [], [], [], []
        for candidate in candidates:
            effect = isinstance(candidate, FormulaProgramEffectCandidateV3)
            implementation = candidate if effect else candidate.candidate
            program = candidate.effect_program.program if effect else implementation.program
            row = []
            for binding in program.bindings:
                if type(binding) is InputBinding:
                    source = ("input", slots[candidate.input_slots[binding.name]])
                elif not effect and binding.name == candidate.plastic_bank_slot:
                    source = ("bank", candidate.bank_slot_ref)
                else:
                    source = ("operand", implementation.operand_store, binding.name)
                if source not in lookup:
                    lookup[source] = len(self.sources)
                    self.sources.append(source)
                row.append(lookup[source])
            if effect:
                source = ("target", slots[candidate.input_slot])
                if source not in lookup:
                    lookup[source] = len(self.sources)
                    self.sources.append(source)
                row.append(lookup[source])
            self.requirements.append(row)
            required.append([slot in candidate.input_slots.values() for slot in query.slot_ids])
            empty.append([slot in (*candidate.output_slot_ids, *candidate.requires_empty_slots) for slot in query.slot_ids])
            kinds.append(int(effect))
            terminal.append(False if isinstance(query, FormulaProgramQueryV5) else candidate.output_slot == query.terminal_slot)
        device = query._action_priority.device
        self.required = torch.tensor(required, dtype=torch.bool, device=device).reshape(len(candidates), len(slots))
        self.empty = torch.tensor(empty, dtype=torch.bool, device=device).reshape(len(candidates), len(slots))
        self.kinds = torch.tensor(kinds, dtype=torch.int64, device=device)
        self.terminal_writes = torch.tensor(terminal, dtype=torch.bool, device=device)
        terminals = tuple(query.terminal_slots.values()) if isinstance(query, FormulaProgramQueryV5) else (query.terminal_slot,)
        self.terminal_indices = torch.tensor([slots[name] for name in terminals], device=device)

    def current_sources(self, arena, shared, needed):
        current = dict(zip(arena.bank_state.slot_refs, zip(arena.bank_state.values, arena.bank_state.revisions, strict=True), strict=True))
        for proposal in arena.proposals:
            current[proposal.target] = proposal.successor, proposal.successor_revision
        values = []
        for source_index in needed:
            source = self.sources[source_index]
            kind, key, *name = source
            if kind == "input":
                value = arena.values.values[key]
            elif kind == "bank":
                value = current.get(key, (None, None))[0]
            elif kind == "target":
                lineage = arena.producers[key]
                value, revision = (None, None) if lineage is None else current.get(lineage.plastic_slot, (None, None))
                if lineage is None or lineage.plastic_value is not value or lineage.plastic_revision != revision:
                    value = None
            else:
                if source not in shared:
                    shared[source] = key.tensor(name[0])
                value = shared[source]
            values.append(value)
        return values

    def prepare_layout(self, arena, key, ready, needed):
        if key not in self.layouts:
            # Native accepts performs shape/dtype/symbol/lineage admission;
            # numerical predicates are captured, not inspected or cached.
            allowed = [False] * len(self.candidates)
            with _capture_finite_validation():
                for index in ready:
                    allowed[index] = self.candidates[index].accepts(arena)
            used = tuple(sorted({source for row, valid in zip(self.requirements, allowed, strict=True) if valid for source in row}))
            positions = {source: index + 1 for index, source in enumerate(used)}
            arity = max(1, max(map(len, self.requirements), default=0))
            rows = [([positions[source] for source in row] if valid else [])
                    for row, valid in zip(self.requirements, allowed, strict=True)]
            indices = torch.tensor([row + [0] * (arity - len(row)) for row in rows],
                                   dtype=torch.int64, device=arena.device).reshape(len(rows), arity)
            source_positions = {source: position for position, source in enumerate(needed)}
            self.layouts[key] = tuple(source_positions[source] for source in used), indices, torch.tensor(allowed, device=arena.device, dtype=torch.bool)
            if len(self.layouts) > 128:
                self.layouts.popitem(last=False)
        self.layouts.move_to_end(key)
        return self.layouts[key]

    def evaluate(self, arenas, *, steps):
        shared, metadata = {}, {}
        groups = {}
        structural = self.query._structural_candidates(arenas, self.candidates)
        for index, (arena, ready) in enumerate(zip(arenas, structural, strict=True)):
            if arena.values.slot_ids != self.query.slot_ids:
                raise ValueError("arena layout does not match ProgramQuery")
            if ready not in self.source_groups:
                self.source_groups[ready] = tuple(sorted({source for column in ready for source in self.requirements[column]}))
            needed = self.source_groups[ready]
            values = self.current_sources(arena, shared, needed)
            signatures = []
            for value in values:
                if value is None:
                    signatures.append(None)
                    continue
                identity = id(value)
                if identity not in metadata:
                    signature = _binding_tensor_metadata(value)
                    if signature is None:
                        return None
                    metadata[identity] = value, signature
                signatures.append(metadata[identity][1])
            occupied = tuple(value is not None for value in arena.values.values)
            key = (tuple(signatures), occupied, arena.batch_size, ready)
            groups.setdefault(key, []).append((index, arena, values))
        output = torch.zeros((len(arenas), len(self.candidates) + self.include_stop), dtype=torch.bool, device=arenas[0].device)
        step = torch.tensor(steps, dtype=torch.int64, device=output.device)
        for key, group in groups.items():
            ready = key[3]
            used, sources, allowed = self.prepare_layout(group[0][1], key, ready, self.source_groups[ready])
            checks = _FiniteTensorRows()
            for _index, _arena, values in group:
                for source in used:
                    checks.add((values[source],))
            flags = checks.evaluate(device=output.device).reshape(len(group), len(used))
            flags = torch.cat((torch.ones((len(group), 1), dtype=torch.bool, device=output.device), flags), dim=1)
            occupied = torch.tensor(key[1], dtype=torch.bool, device=output.device).expand(len(group), -1)
            counts = torch.tensor([(arena.tensor_steps, arena.effect_steps) for _, arena, _ in group], device=output.device)
            selected = self.kernel(flags, sources, allowed, occupied, self.required, self.empty, self.kinds,
                                   self.terminal_writes, self.terminal_indices, counts, step)
            output[torch.tensor([index for index, _, _ in group], device=output.device)] = selected
        return output


def indexed_candidate_mask(query, arenas, candidates, *, steps, include_stop):
    cache = _ADMISSION_TABLES.get()
    if cache is None or not arenas:
        return None
    from ._formula_candidate_admission import _prepare_native_candidate, _NATIVE_STOP
    if not candidates or "_stop_eligible" in query.__dict__ or type(query)._stop_eligible is not _NATIVE_STOP.get(type(query)):
        return None
    key = (query, tuple(candidates), include_stop)
    if key not in cache:
        if not all(_prepare_native_candidate(query, candidate)[0] for candidate in candidates):
            cache[key] = None
        else:
            try:
                cache[key] = _AdmissionTable(query, candidates, include_stop)
            except (KeyError, ValueError, TypeError):
                cache[key] = None
    plan = cache[key]
    return None if plan is None else plan.evaluate(arenas, steps=steps)
