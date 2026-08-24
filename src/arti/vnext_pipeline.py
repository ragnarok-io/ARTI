"""Executable composition boundary for the ARTI vNext pipeline."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar
from types import MappingProxyType

import torch
from torch import nn

from .component_registry import component_ref, component_spec
from .reversible_topology import FoldedTensor, UnfoldedTensor
from .vnext_contracts import (
    EnvelopeRef,
    FoldedPulseSupports,
    PulseStageGraph,
    PulseSupports,
    StageMode,
    StageRole,
    TensorEnvelope,
    apply_intervention,
    fold_pulse_supports,
    unfold_pulse_supports,
)


_EXECUTABLE_ROLES = frozenset({StageRole.HALF, StageRole.FOLD, StageRole.UNFOLD})


class PulseExecutor(nn.Module):
    """Bind a Pulse stage manifest to exact component instances and execute it."""

    _component_reference: ClassVar[str] = "arti/pulse-executor@1"

    def __init__(
        self,
        manifest: PulseStageGraph,
        modules: Mapping[str, nn.Module],
    ) -> None:
        super().__init__()
        if not isinstance(manifest, PulseStageGraph):
            raise TypeError("manifest must be a PulseStageGraph")
        enabled = tuple(stage for stage in manifest.stages if stage.mode is StageMode.ENABLED)
        enabled_ids = {stage.stage_id for stage in enabled}
        if set(modules) != enabled_ids:
            raise ValueError("modules must match enabled stage IDs exactly")
        for stage in enabled:
            if stage.role not in _EXECUTABLE_ROLES:
                raise ValueError(
                    f"enabled role {stage.role.value!r} is not executable in PulseExecutor@1"
                )
            module = modules[stage.stage_id]
            if not isinstance(module, nn.Module):
                raise TypeError("Pulse stage implementations must be nn.Module instances")
            if component_ref(module) != stage.component_ref:
                raise ValueError("Pulse stage module identity does not match its manifest")
            if component_spec(module).config_fingerprint != stage.config_fingerprint:
                raise ValueError("Pulse stage module config does not match its manifest")
        self.manifest = manifest
        self._stage_keys = MappingProxyType(
            {stage.stage_id: f"stage_{index}" for index, stage in enumerate(enabled)}
        )
        self.stages = nn.ModuleDict(
            {
                self._stage_keys[stage.stage_id]: modules[stage.stage_id]
                for stage in enabled
            }
        )

    @property
    def enabled_stage_ids(self) -> tuple[str, ...]:
        return tuple(self._stage_keys)

    def forward(self, world: TensorEnvelope, supports: PulseSupports) -> TensorEnvelope:
        if not isinstance(world, TensorEnvelope) or world.ref is not EnvelopeRef.WORLD:
            raise TypeError("PulseExecutor expects a WORLD TensorEnvelope")
        if not isinstance(supports, PulseSupports):
            raise TypeError("PulseExecutor requires PulseSupports")
        if supports.observed.domain != world.domain:
            raise ValueError("Pulse supports must belong to the WORLD envelope domain")
        if not world._mask_for_execution().equal(supports._validity):
            raise ValueError("Pulse support validity must match the WORLD envelope mask")
        payload: TensorEnvelope | FoldedTensor = world
        current_ref = world.ref
        domain_stack = []
        support_stack: list[FoldedPulseSupports] = []

        for stage in self.manifest.stages:
            if current_ref is not stage.input_schema:
                raise ValueError("runtime envelope does not match Pulse stage input")
            if stage.mode is StageMode.OFF:
                current_ref = stage.output_schema
                continue

            module = self.stages[self._stage_keys[stage.stage_id]]
            if stage.role is StageRole.HALF:
                if not isinstance(payload, TensorEnvelope):
                    raise TypeError("Half stage requires a TensorEnvelope")
                exposed_input = torch.where(
                    supports.exposed._mask.unsqueeze(-1),
                    payload.value,
                    torch.zeros_like(payload.value),
                )
                candidate = TensorEnvelope(
                    payload.ref,
                    module(exposed_input),
                    payload._mask_for_execution(),
                    payload.domain,
                )
                payload = apply_intervention(payload, candidate, supports)
            elif stage.role is StageRole.FOLD:
                if not isinstance(payload, TensorEnvelope):
                    raise TypeError("Fold stage requires a TensorEnvelope")
                observed_fold = getattr(module, "_observed", None)
                if not callable(observed_fold):
                    raise TypeError("Pulse Fold stage requires observation-aware transport")
                folded = observed_fold(
                    payload.value, payload.mask, supports.observed._mask
                )
                if not isinstance(folded, FoldedTensor):
                    raise TypeError("Fold stage must return FoldedTensor")
                stage.topology_binding.validate_record(folded.record)
                domain_stack.append(payload.domain)
                transported = fold_pulse_supports(supports, folded.record)
                support_stack.append(transported)
                supports = transported.active
                payload = folded
            elif stage.role is StageRole.UNFOLD:
                if not isinstance(payload, FoldedTensor) or not domain_stack:
                    raise TypeError("UnFold stage requires a paired FoldedTensor")
                stage.topology_binding.validate_record(payload.record)
                if not support_stack or support_stack[-1].record is not payload.record:
                    raise ValueError("UnFold support transport does not match FoldRecord")
                unfolded = module(payload)
                if not isinstance(unfolded, UnfoldedTensor):
                    raise TypeError("UnFold stage must return UnfoldedTensor")
                domain = domain_stack.pop()
                payload = TensorEnvelope(
                    stage.output_schema,
                    unfolded.value,
                    unfolded.mask,
                    domain,
                )
                supports = unfold_pulse_supports(support_stack.pop())
            current_ref = stage.output_schema

        if domain_stack or support_stack or not isinstance(payload, TensorEnvelope):
            raise ValueError("Pulse execution ended with unclosed topology state")
        return payload


__all__ = ["PulseExecutor"]
