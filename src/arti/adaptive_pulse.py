"""Composable ARTI Pulse@2 execution over versioned tensor stages."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import ClassVar, Mapping

import torch
from torch import Tensor, nn

from .aggregate import ReunionAggregate
from .component_registry import ComponentRef, component_ref, component_spec
from .formula_attention import ActiveWorkspace, FormulaAttention, SelectiveCompute
from .observation import AdaptiveObservation
from .reversible_topology import FoldRecord, TopologyFold, TopologyUnFold
from .runtime_contracts import (
    EnvelopeRef,
    OffSemantics,
    OperandContract,
    OperandKind,
    PulseStageGraph,
    PulseStageSpec,
    PulseSupports,
    StageMode,
    StageRole,
    SupportDomain,
    SupportKind,
    SupportMask,
    TensorEnvelope,
    TopologyBinding,
    _gather_active_pulse_supports,
    _restore_active_pulse_supports,
    lift_observation_supports,
)


_OFF = {
    StageRole.OBSERVATION: OffSemantics.IDENTITY_OBSERVATION,
    StageRole.HALF: OffSemantics.IDENTITY_VALUES,
    StageRole.FOLD: OffSemantics.ALL_OBSERVED_EXPOSED,
    StageRole.INTERVENTION: OffSemantics.PRESERVE_INTERVENTION_SUPPORT,
    StageRole.SELECTIVE_COMPUTE: OffSemantics.NO_COMPUTE,
    StageRole.UNFOLD: OffSemantics.REUNION_BYPASS,
    StageRole.AGGREGATE: OffSemantics.NO_AGGREGATION,
    StageRole.BANK_UPDATE: OffSemantics.NO_BANK_UPDATE,
}


@dataclass(frozen=True)
class BankState:
    """Caller-owned Bank values bound to one stable asset identity."""

    value: Tensor
    mask: Tensor
    source_ref: str
    partition_id: str
    asset_fingerprint: str
    _runtime_contract_ref: ClassVar[str] = "arti/bank-state@1"

    def __post_init__(self) -> None:
        if not isinstance(self.value, Tensor) or not self.value.is_floating_point():
            raise TypeError("BankState value must be a floating-point Tensor")
        if self.value.ndim != 3 or self.value.shape[-2] <= 0 or self.value.shape[-1] <= 0:
            raise ValueError("BankState value must have shape [B, S, D]")
        if (
            not isinstance(self.mask, Tensor)
            or self.mask.dtype != torch.bool
            or self.mask.shape != self.value.shape[:-1]
            or self.mask.device != self.value.device
        ):
            raise ValueError("BankState mask must be boolean [B, S] on the value device")
        ComponentRef.parse(self.source_ref)
        if re.fullmatch(r"[a-z0-9][a-z0-9._-]*", self.partition_id) is None:
            raise ValueError("BankState partition_id is invalid")
        if re.fullmatch(r"[0-9a-f]{64}", self.asset_fingerprint) is None:
            raise ValueError("BankState asset_fingerprint must be SHA-256 hex")
        object.__setattr__(self, "mask", self.mask.detach().clone())


class BankUpdateStatus(str, Enum):
    OFF = "off"
    NO_WRITE = "no_write"
    UPDATED = "updated"


@dataclass(frozen=True)
class BankUpdateOutput:
    """Explicit result of the optional functional Bank transition."""

    status: BankUpdateStatus
    state: BankState | None
    contract: OperandContract | None = None
    write_mask: Tensor | None = None
    updater_info: Mapping[str, Tensor] | None = None

    @property
    def updated(self) -> bool:
        return self.status is BankUpdateStatus.UPDATED


@dataclass(frozen=True)
class PulseOutput:
    """Tensor output, pre-aggregation supports, and an optional next Bank state."""

    envelope: TensorEnvelope
    source_supports: PulseSupports
    manifest_fingerprint: str
    bank: BankUpdateOutput
    value_identity: bool
    diagnostics: PulseDiagnostics

    @property
    def value(self) -> Tensor:
        return self.envelope.value

    @property
    def mask(self) -> Tensor:
        return self.envelope.mask

    @property
    def ref(self) -> EnvelopeRef:
        return self.envelope.ref

    @property
    def domain(self) -> SupportDomain:
        return self.envelope.domain

    @property
    def next_bank(self) -> BankState | None:
        return self.bank.state


@dataclass(frozen=True)
class PulseDiagnostics:
    """Runtime-owned records from the topology and compute stages."""

    topology_record: FoldRecord | None = None
    compute: object | None = None


class AdaptivePulse(nn.Module):
    """Execute the canonical Pulse@2 stage graph with independently replaceable parts."""

    _component_reference: ClassVar[str] = "arti/pulse@2"

    def __init__(
        self,
        *,
        observation: AdaptiveObservation | None = None,
        half: nn.Module | None = None,
        fold: TopologyFold | None = None,
        intervention: FormulaAttention | None = None,
        selective_compute: SelectiveCompute | None = None,
        unfold: TopologyUnFold | None = None,
        aggregate: ReunionAggregate | None = None,
        bank_update: nn.Module | None = None,
    ) -> None:
        super().__init__()
        for module, expected, name in (
            (observation, AdaptiveObservation, "observation"),
            (fold, TopologyFold, "fold"),
            (intervention, FormulaAttention, "intervention"),
            (unfold, TopologyUnFold, "unfold"),
            (aggregate, ReunionAggregate, "aggregate"),
        ):
            if module is not None and not isinstance(module, expected):
                raise TypeError(f"{name} must be {expected.__name__} or None")
        if half is not None and not isinstance(half, nn.Module):
            raise TypeError("half must be an nn.Module or None")
        if selective_compute is not None:
            if not isinstance(selective_compute, nn.Module):
                raise TypeError("selective_compute must be an nn.Module or None")
            registration = component_spec(selective_compute)
            if "pulse.stage.selective-compute" not in registration.capabilities:
                raise ValueError(
                    "selective_compute must declare pulse.stage.selective-compute capability"
                )
        if bank_update is not None:
            if not isinstance(bank_update, nn.Module):
                raise TypeError("bank_update must be an nn.Module or None")
            registration = component_spec(bank_update)
            if "pulse.stage.bank-update" not in registration.capabilities:
                raise ValueError("bank_update must declare pulse.stage.bank-update capability")
        if (fold is None) != (unfold is None):
            raise ValueError("Pulse@2 requires Fold and UnFold to be enabled together")
        if (intervention is not None or selective_compute is not None) and fold is None:
            raise ValueError("intervention and selective compute require reversible topology")
        if intervention is not None and selective_compute is None:
            raise ValueError("FormulaAttention requires SelectiveCompute")
        if bank_update is not None and aggregate is None:
            raise ValueError("BankUpdate requires an enabled Aggregate stage")

        binding = None
        if fold is not None:
            assert unfold is not None
            topology = fold.topology
            if unfold.inverse_contract.contract_fingerprint != topology.contract_fingerprint:
                raise ValueError("Fold and UnFold do not share one topology contract")
            binding = TopologyBinding(
                topology_ref=topology._component_reference,
                topology_config_fingerprint=topology.contract_fingerprint,
                producer_provenance_fingerprint=topology.producer_provenance_fingerprint,
            )
            compute_active_count = getattr(selective_compute, "active_count", None)
            if (
                compute_active_count is not None
                and compute_active_count != topology.active_count
            ):
                raise ValueError(
                    "selective compute active_count must match reversible topology"
                )

        self.observation = observation
        self.half_stage = half
        self.fold = fold
        self.intervention = intervention
        self.selective_compute = selective_compute
        self.unfold = unfold
        self.aggregate = aggregate
        self.bank_update = bank_update
        self._bank_update_ref = (
            None if bank_update is None else component_ref(bank_update)
        )
        self.manifest = self._build_manifest(binding)
        self._validate_manifest_binding()
        self._bound_stage_modules = (
            self.observation,
            self.half_stage,
            self.fold,
            self.intervention,
            self.selective_compute,
            self.unfold,
            self.aggregate,
            self.bank_update,
        )
        self.register_forward_pre_hook(self._manifest_identity_preflight)

    @torch.compiler.disable
    def _manifest_identity_preflight(
        self,
        _module: nn.Module,
        _args: tuple[object, ...],
    ) -> None:
        """Reject stage replacement without registry access or tensor sync."""

        current = (
            self.observation,
            self.half_stage,
            self.fold,
            self.intervention,
            self.selective_compute,
            self.unfold,
            self.aggregate,
            self.bank_update,
        )
        roles = (
            "observation",
            "half",
            "fold",
            "intervention",
            "selective_compute",
            "unfold",
            "aggregate",
            "bank_update",
        )
        for role, expected, actual in zip(
            roles, self._bound_stage_modules, current, strict=True
        ):
            if actual is expected:
                continue
            if expected is None:
                raise ValueError(
                    f"Pulse@2 stage {role!r} is enabled after manifest binding"
                )
            if actual is None:
                raise ValueError(
                    f"Pulse@2 stage {role!r} is missing after manifest binding"
                )
            raise ValueError(
                f"Pulse@2 stage {role!r} was replaced after manifest binding"
            )

    def validate_manifest(self) -> None:
        """Validate the frozen stage graph after explicit structural changes."""

        self._validate_manifest_binding()

    def _validate_manifest_binding(self) -> None:
        modules = {
            StageRole.OBSERVATION: self.observation,
            StageRole.HALF: self.half_stage,
            StageRole.FOLD: self.fold,
            StageRole.INTERVENTION: self.intervention,
            StageRole.SELECTIVE_COMPUTE: self.selective_compute,
            StageRole.UNFOLD: self.unfold,
            StageRole.AGGREGATE: self.aggregate,
            StageRole.BANK_UPDATE: self.bank_update,
        }
        for stage in self.manifest.stages:
            module = modules[stage.role]
            if stage.mode is StageMode.OFF:
                if module is not None:
                    raise ValueError(
                        f"Pulse@2 stage {stage.stage_id!r} is enabled after manifest binding"
                    )
                continue
            if module is None:
                raise ValueError(
                    f"Pulse@2 stage {stage.stage_id!r} is missing after manifest binding"
                )
            spec = component_spec(module)
            if (
                spec.reference != stage.component_ref
                or spec.config_fingerprint != stage.config_fingerprint
            ):
                raise ValueError(
                    f"Pulse@2 stage {stage.stage_id!r} drifted from its manifest"
                )

    @staticmethod
    def _stage(
        role: StageRole,
        module: nn.Module | None,
        input_ref: EnvelopeRef,
        output_ref: EnvelopeRef,
        *,
        pair_id: str | None = None,
        binding: TopologyBinding | None = None,
    ) -> PulseStageSpec:
        if module is None:
            return PulseStageSpec(
                stage_id=role.value.replace("_", "-"),
                role=role,
                mode=StageMode.OFF,
                component_ref=None,
                input_schema=input_ref,
                output_schema=input_ref,
                off_semantics=_OFF[role],
            )
        spec = component_spec(module)
        return PulseStageSpec(
            stage_id=role.value.replace("_", "-"),
            role=role,
            mode=StageMode.ENABLED,
            component_ref=spec.reference,
            input_schema=input_ref,
            output_schema=output_ref,
            config=spec.config,
            pair_id=pair_id,
            topology_binding=binding,
        )

    def _build_manifest(self, binding: TopologyBinding | None) -> PulseStageGraph:
        current = EnvelopeRef.WORLD
        stages: list[PulseStageSpec] = []

        output = EnvelopeRef.OBSERVATION if self.observation is not None else current
        stages.append(self._stage(StageRole.OBSERVATION, self.observation, current, output))
        current = output
        stages.append(self._stage(StageRole.HALF, self.half_stage, current, current))
        output = EnvelopeRef.FOLDED if self.fold is not None else current
        stages.append(
            self._stage(
                StageRole.FOLD,
                self.fold,
                current,
                output,
                pair_id="pulse-topology" if self.fold is not None else None,
                binding=binding,
            )
        )
        current = output
        stages.append(
            self._stage(StageRole.INTERVENTION, self.intervention, current, current)
        )
        stages.append(
            self._stage(
                StageRole.SELECTIVE_COMPUTE,
                self.selective_compute,
                current,
                current,
            )
        )
        output = EnvelopeRef.REUNITED if self.unfold is not None else current
        stages.append(
            self._stage(
                StageRole.UNFOLD,
                self.unfold,
                current,
                output,
                pair_id="pulse-topology" if self.unfold is not None else None,
                binding=binding,
            )
        )
        current = output
        output = EnvelopeRef.PULSE if self.aggregate is not None else current
        stages.append(self._stage(StageRole.AGGREGATE, self.aggregate, current, output))
        current = output
        stages.append(
            self._stage(StageRole.BANK_UPDATE, self.bank_update, current, current)
        )
        return PulseStageGraph(tuple(stages))

    @property
    def enabled_components(self) -> tuple[str, ...]:
        return self.manifest.enabled_dependencies

    def forward(
        self,
        world: TensorEnvelope,
        supports: PulseSupports,
        *,
        intervention_factors: Tensor | None = None,
        compute_factors: Tensor | None = None,
        visibility: Tensor | None = None,
        aggregate_q: Tensor | None = None,
        bank_state: BankState | None = None,
        write_exposure: float | Tensor = 1.0,
        write_policy: object | None = None,
        formula_route: object | None = None,
        objective_query: Tensor | None = None,
        topology_source: nn.Module | None = None,
        topology_source_inputs: tuple[Tensor, ...] | None = None,
    ) -> PulseOutput:
        if not isinstance(world, TensorEnvelope) or world.ref is not EnvelopeRef.WORLD:
            raise TypeError("Pulse@2 expects a WORLD TensorEnvelope")
        if not isinstance(supports, PulseSupports):
            raise TypeError("Pulse@2 requires PulseSupports")
        if supports.observed.domain != world.domain:
            raise ValueError("Pulse supports must belong to the WORLD envelope domain")
        if formula_route is not None and self.selective_compute is None:
            raise ValueError("formula_route requires an enabled selective compute stage")
        objective_contract = (
            "forbidden"
            if self.selective_compute is None
            else getattr(
                self.selective_compute, "objective_query_contract", "forbidden"
            )
        )
        if objective_query is not None and objective_contract != "required":
            raise ValueError(
                "objective_query requires an Objective-controlled compute stage"
            )
        if objective_query is None and objective_contract == "required":
            raise ValueError(
                "Objective-controlled compute stage requires objective_query"
            )
        if (topology_source is None) != (topology_source_inputs is None):
            raise ValueError(
                "topology_source and topology_source_inputs must be supplied together"
            )
        if topology_source is not None and self.fold is None:
            raise ValueError("topology_source requires an enabled Fold stage")
        validity_matches = torch.eq(
            supports._validity, world._mask_for_execution()
        ).all()
        if torch.compiler.is_compiling():
            torch._assert_async(
                validity_matches,
                "Pulse support validity must match the WORLD envelope",
            )
        elif not bool(validity_matches):
            raise ValueError("Pulse support validity must match the WORLD envelope")

        payload = world
        current_supports = supports
        topology_record: FoldRecord | None = None
        compute_info: object | None = None
        if self.observation is not None:
            payload = self.observation(payload)
            current_supports = lift_observation_supports(current_supports, payload)

        if self.half_stage is not None:
            exposed = current_supports.exposed._mask
            redacted = torch.where(
                exposed.unsqueeze(-1), payload.value, torch.zeros_like(payload.value)
            )
            candidate = self.half_stage(redacted)
            if not isinstance(candidate, Tensor) or candidate.shape != payload.value.shape:
                raise ValueError("Half stage must preserve the envelope value shape")
            payload = payload.replace(
                torch.where(exposed.unsqueeze(-1), candidate, payload.value)
            )

        if self.fold is not None:
            source_supports = current_supports
            sourced_fold = None
            overlay = None
            if topology_source is None:
                overlay = self.fold._overlay(
                    payload.value,
                    payload._mask_for_execution(),
                    observed=current_supports.observed._mask,
                )
                active_value = overlay.active
                active_mask = overlay.active_mask
                topology_record = overlay.record
            else:
                assert topology_source_inputs is not None
                sourced_fold = self.fold.from_source(
                    payload.value,
                    source=topology_source,
                    source_inputs=topology_source_inputs,
                    mask=payload._mask_for_execution(),
                    observed=current_supports.observed._mask,
                    require_source_axes=True,
                )
                active_value = sourced_fold.active
                active_mask = sourced_fold.active_mask
                topology_record = sourced_fold.record
            active_supports = _gather_active_pulse_supports(
                current_supports, topology_record
            )
            active = ActiveWorkspace(
                active_value,
                active_mask,
                active_supports.exposed._mask,
                active_supports.intervened._mask,
            )

            if self.intervention is not None:
                active = self.intervention(active, intervention_factors)
            if self.selective_compute is not None:
                compute_kwargs = {
                    "visibility": visibility,
                    "formula_route": formula_route,
                    "return_info": True,
                }
                if objective_contract == "required":
                    compute_kwargs["objective_query"] = objective_query
                active, compute_info = self.selective_compute(
                    active,
                    compute_factors,
                    **compute_kwargs,
                )
            elif formula_route is not None:
                raise ValueError("formula_route requires an enabled selective compute stage")

            for actual, expected, name in (
                (active.validity, active_supports._validity, "validity"),
                (active.exposed, active_supports.exposed._mask, "exposed"),
            ):
                unchanged = torch.eq(actual, expected).all()
                if torch.compiler.is_compiling():
                    torch._assert_async(
                        unchanged, f"selective stage changed {name} support"
                    )
                elif not bool(unchanged):
                    raise ValueError(f"selective stage changed {name} support")

            assert self.unfold is not None
            if sourced_fold is None:
                assert overlay is not None
                restored = self.unfold._overlay(overlay.replace_active(active.value))
            else:
                restored = self.unfold(sourced_fold.replace(active=active.value))
            active_supports = active_supports.replace_intervened(active.intervened)
            current_supports = _restore_active_pulse_supports(
                source_supports, active_supports, topology_record
            )
            payload = TensorEnvelope(
                EnvelopeRef.REUNITED,
                restored.value,
                restored.mask,
                current_supports.observed.domain,
            )

        if self.aggregate is not None:
            payload = self.aggregate(payload, q=aggregate_q)

        bank_output = BankUpdateOutput(BankUpdateStatus.OFF, bank_state)
        if self.bank_update is not None:
            if not isinstance(bank_state, BankState):
                raise ValueError("enabled BankUpdate requires bank_state")
            write = current_supports.write
            contract = current_supports.write_contract
            if write is None or contract is None:
                raise ValueError("enabled BankUpdate requires typed write authority")
            if contract.kind is not OperandKind.WRITE:
                raise ValueError("BankUpdate requires a WRITE operand contract")
            if contract.consumer_ref != self._bank_update_ref:
                raise ValueError("write authority consumer does not match BankUpdate")
            if write.domain != contract.domain:
                raise ValueError("write authority domain does not match its contract")
            if contract.source_ref != bank_state.source_ref:
                raise ValueError("write authority source does not match BankState")
            if contract.partition_id != bank_state.partition_id:
                raise ValueError("write authority partition does not match BankState")
            if contract.source_asset_fingerprint != bank_state.asset_fingerprint:
                raise ValueError("write authority asset does not match BankState")
            if contract.factor_dim != bank_state.value.shape[-1]:
                raise ValueError("write authority factor_dim does not match BankState")
            if bank_state.value.shape[:-1] != write._mask.shape:
                raise ValueError("bank_state does not match write support shape")
            if bank_state.value.device != write._mask.device:
                raise ValueError("bank_state and write support must share device and use floating values")
            invalid_write = (write._mask & ~bank_state.mask).any()
            if torch.compiler.is_compiling() or invalid_write.device.type != "cpu":
                torch._assert_async(
                    ~invalid_write,
                    "write authority cannot address invalid Bank slots",
                )
            elif bool(invalid_write):
                raise ValueError("write authority cannot address invalid Bank slots")
            if torch.compiler.is_compiling():
                torch._assert_async(
                    write._mask.any(),
                    "compiled BankUpdate requires at least one authorized write slot",
                )
                execute_update = True
            else:
                execute_update = bool(write._mask.any())
            if execute_update:
                private_input = bank_state.value.clone()
                candidate, updater_info = self.bank_update(
                    payload.value,
                    private_input,
                    trace_mask=payload._mask_for_execution(),
                    target_mask=bank_state.mask,
                    write_mask=write._mask,
                    exposure=write_exposure,
                    policy=write_policy,
                    return_info=True,
                )
                if (
                    not isinstance(candidate, Tensor)
                    or candidate.shape != bank_state.value.shape
                    or candidate.dtype != bank_state.value.dtype
                    or candidate.device != bank_state.value.device
                ):
                    raise ValueError("BankUpdate must propose a Bank tensor matching bank_state")
                next_value = torch.where(
                    write._mask.unsqueeze(-1), candidate, bank_state.value
                )
                next_state = BankState(
                    next_value,
                    bank_state.mask,
                    bank_state.source_ref,
                    bank_state.partition_id,
                    bank_state.asset_fingerprint,
                )
                bank_output = BankUpdateOutput(
                    BankUpdateStatus.UPDATED,
                    next_state,
                    contract,
                    write.mask,
                    updater_info,
                )
            else:
                bank_output = BankUpdateOutput(
                    BankUpdateStatus.NO_WRITE,
                    bank_state,
                    contract,
                    write.mask,
                )

        return PulseOutput(
            envelope=payload,
            source_supports=current_supports,
            manifest_fingerprint=self.manifest.fingerprint,
            bank=bank_output,
            value_identity=payload.value is world.value,
            diagnostics=PulseDiagnostics(topology_record, compute_info),
        )

    def run_tensor(
        self,
        value: Tensor,
        *,
        mask: Tensor | None = None,
        observed: Tensor | None = None,
        exposed: Tensor | None = None,
        intervened: Tensor | None = None,
        domain_id: str = "pulse-world",
        partition_id: str = "world",
        transition_id: str = "tensor-call",
        **forward_kwargs: object,
    ) -> PulseOutput:
        """Run non-writing Pulse stages from ordinary tensors and masks."""

        if self.bank_update is not None:
            raise ValueError(
                "run_tensor cannot infer typed Bank write authority; use forward"
            )
        write_arguments = {"bank_state", "write_exposure", "write_policy"}
        supplied_write_arguments = write_arguments.intersection(forward_kwargs)
        if supplied_write_arguments:
            names = ", ".join(sorted(supplied_write_arguments))
            raise ValueError(f"run_tensor does not accept Bank write arguments: {names}")
        if not isinstance(value, Tensor) or not (
            value.is_floating_point() or value.is_complex()
        ):
            raise TypeError("value must be a floating or complex Tensor")
        if value.ndim < 2:
            raise ValueError("value must have shape [..., N, D]")
        if mask is None:
            mask = torch.ones(value.shape[:-1], dtype=torch.bool, device=value.device)
        if observed is None:
            observed = mask
        if exposed is None:
            exposed = observed
        if intervened is None:
            intervened = torch.zeros_like(mask)
        domain = SupportDomain.for_tensor(
            mask,
            domain_id=domain_id,
            owner_ref=self._component_reference,
            partition_id=partition_id,
            transition_id=transition_id,
        )
        world = TensorEnvelope(EnvelopeRef.WORLD, value, mask, domain)
        supports = PulseSupports(
            SupportMask(SupportKind.OBSERVED, observed, domain),
            SupportMask(SupportKind.EXPOSED, exposed, domain),
            SupportMask(SupportKind.INTERVENED, intervened, domain),
            validity=mask,
        )
        return self(world, supports, **forward_kwargs)


__all__ = [
    "AdaptivePulse",
    "BankState",
    "BankUpdateOutput",
    "BankUpdateStatus",
    "PulseOutput",
    "PulseDiagnostics",
]
