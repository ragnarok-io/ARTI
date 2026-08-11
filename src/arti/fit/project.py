"""Gradle-like ARTI project and fit API."""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping
from dataclasses import replace
from time import perf_counter
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

from ..config import STATE_RECALL_COMPOSITION_FACTOR
from ..layers import ARTILatentRecallField
from ..recall_experts import _concatenate_bank_values
from ..recall_formula import RecallFormulaContract
from .artifacts import (
    ARTIFitReport,
    ARTIFitResult,
    BuildTaskSpec,
    FitTaskRecord,
    ForwardProfile,
    MechanismSummary,
    ParameterSummary,
    report_adapter_state_dict,
    validate_artifact,
)
from .config import (
    FitProjectConfig,
    MechanismOverrides,
    apply_mechanism_overrides,
    load_fit_config,
    validate_fit_config,
)
from .insertion import (
    ARTIAdapterWrapper,
    AdapterInsertionPlan,
    InsertionSpec,
    adapters_enabled,
    insert_adapters,
    iter_adapter_wrappers,
    mark_recall_state_banks_calibrated,
    reset_recall_queries,
    plan_adapters,
)
from .objectives import infer_objectives, resolve_objectives
from .plugins import default_strategy_for, get_plugin, plugin_report
from .profiles import AdapterProfile, resolve_profile
from .runtime import RuntimeFieldConfig
from .scales import AdapterScale, resolve_scale
from .scanner import InsertionCandidate, ScanReport, run_model, scan_model
from .strategies import resolve_where


class ARTIProject:
    """A Gradle-like adaptation project for a PyTorch model."""

    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.plugins: list[str] = ["torch"]
        self.profile_config: AdapterProfile = resolve_profile(None)
        self.scale_config: AdapterScale = resolve_scale(None)
        self.mechanism_overrides = MechanismOverrides()
        self.scale_name = "small"
        self.scan_report: ScanReport | None = None
        self.scan_positions: tuple[str, ...] = ("output",)
        self.insertion_spec = InsertionSpec()
        self.default_max_extra_params: int | str | None = None
        self.inserted = ()
        self.insert_attempted = False
        self.insertion_plan: AdapterInsertionPlan | None = None
        self.fit_steps = 0
        self.runtime_causal = False
        self.runtime_fields = RuntimeFieldConfig()
        self.objective_plan: tuple[str, ...] = ()
        self.loss_history: list[float] = []
        self.calibration_history: list[float] = []
        self.calibration_objective: str | None = None
        self.validation_history: list[dict[str, float]] = []
        self.task_history: list[FitTaskRecord] = []
        self.forward_profiles: list[ForwardProfile] = []
        self.fit_config: FitProjectConfig | None = None
        self.applied_artifact: dict[str, Any] | None = None

    def plugin(self, name: str) -> "ARTIProject":
        get_plugin(name)
        if name not in self.plugins:
            self.plugins.append(name)
        return self

    def profile(self, name: str | AdapterProfile, *, phases: int | None = None) -> "ARTIProject":
        self.profile_config = resolve_profile(name, phases=phases)
        return self

    def scale(self, name: str | AdapterScale) -> "ARTIProject":
        self.scale_config = resolve_scale(name)
        self.scale_name = name if isinstance(name, str) else "custom"
        return self

    def runtime(
        self,
        *,
        causal: bool = False,
        mask_key: str | None = None,
        coord_key: str | None = None,
        observer_coord_key: str | None = None,
        frame_operators_key: str | None = None,
    ) -> "ARTIProject":
        self.runtime_causal = causal
        if any(
            value is not None
            for value in (mask_key, coord_key, observer_coord_key, frame_operators_key)
        ):
            self.runtime_fields = RuntimeFieldConfig(
                mask_key=self.runtime_fields.mask_key if mask_key is None else mask_key,
                coord_key=self.runtime_fields.coord_key if coord_key is None else coord_key,
                observer_coord_key=self.runtime_fields.observer_coord_key
                if observer_coord_key is None
                else observer_coord_key,
                frame_operators_key=self.runtime_fields.frame_operators_key
                if frame_operators_key is None
                else frame_operators_key,
            )
        return self

    def objectives(self, objective: str | Iterable[str] | None) -> "ARTIProject":
        self.objective_plan = resolve_objectives(objective)
        return self

    def at(
        self,
        where: str | Iterable[str],
        *,
        exclude: str | Iterable[str] | None = None,
        positions: str | tuple[str, ...] | None = None,
        scale_pattern: Mapping[str, str] | None = None,
        every: int = 1,
    ) -> "ARTIProject":
        """Select insertion paths without scanning or mutating the model."""

        patterns = (where,) if isinstance(where, str) else tuple(where)
        if not patterns:
            raise ValueError("where must select at least one module pattern")
        if every <= 0:
            raise ValueError("every must be positive")
        resolved_positions = (
            self.insertion_spec.positions
            if positions is None
            else (positions,)
            if isinstance(positions, str)
            else tuple(positions)
        )
        if not resolved_positions or set(resolved_positions) - {"input", "output"}:
            raise ValueError("positions must contain 'input' and/or 'output'")
        self.insertion_spec = InsertionSpec(
            where=patterns,
            exclude=self.insertion_spec.exclude
            if exclude is None
            else (exclude,)
            if isinstance(exclude, str)
            else tuple(exclude),
            positions=resolved_positions,
            scale_pattern=self.insertion_spec.scale_pattern
            if scale_pattern is None
            else _scale_pattern_items(scale_pattern),
            every=every,
            freeze_base=self.insertion_spec.freeze_base,
            max_adapters=self.insertion_spec.max_adapters,
            max_extra_params=self.insertion_spec.max_extra_params,
            identity_gate=self.insertion_spec.identity_gate,
            zero_init_output=self.insertion_spec.zero_init_output,
            bridge_mode=self.insertion_spec.bridge_mode,
            boundary_mask_key=self.insertion_spec.boundary_mask_key,
            require_runtime_context=self.insertion_spec.require_runtime_context,
        )
        return self

    def freeze(self, base: bool = True) -> "ARTIProject":
        """Declare the base-model freezing policy without applying it yet."""

        self.insertion_spec = InsertionSpec(
            where=self.insertion_spec.where,
            exclude=self.insertion_spec.exclude,
            positions=self.insertion_spec.positions,
            scale_pattern=self.insertion_spec.scale_pattern,
            every=self.insertion_spec.every,
            freeze_base=base,
            max_adapters=self.insertion_spec.max_adapters,
            max_extra_params=self.insertion_spec.max_extra_params,
            identity_gate=self.insertion_spec.identity_gate,
            zero_init_output=self.insertion_spec.zero_init_output,
            bridge_mode=self.insertion_spec.bridge_mode,
            boundary_mask_key=self.insertion_spec.boundary_mask_key,
            require_runtime_context=self.insertion_spec.require_runtime_context,
        )
        return self

    def budget(
        self,
        *,
        max_adapters: int | None = None,
        max_extra_params: int | str | None = None,
    ) -> "ARTIProject":
        """Declare adapter-count and parameter budgets without mutating the model."""

        if max_adapters is not None and max_adapters < 0:
            raise ValueError("max_adapters must be non-negative")
        self.insertion_spec = InsertionSpec(
            where=self.insertion_spec.where,
            exclude=self.insertion_spec.exclude,
            positions=self.insertion_spec.positions,
            scale_pattern=self.insertion_spec.scale_pattern,
            every=self.insertion_spec.every,
            freeze_base=self.insertion_spec.freeze_base,
            max_adapters=max_adapters,
            max_extra_params=self.insertion_spec.max_extra_params,
            identity_gate=self.insertion_spec.identity_gate,
            zero_init_output=self.insertion_spec.zero_init_output,
            bridge_mode=self.insertion_spec.bridge_mode,
            boundary_mask_key=self.insertion_spec.boundary_mask_key,
            require_runtime_context=self.insertion_spec.require_runtime_context,
        )
        self.default_max_extra_params = max_extra_params
        return self

    def preview(self, sample_batch: Any | None = None) -> ARTIFitReport:
        """Scan and plan insertion without wrapping modules or freezing parameters."""

        if sample_batch is not None or self.scan_report is None:
            self.scan(sample_batch)
        self.plan_insert()
        return self.report()

    def configure(self, config: FitProjectConfig | dict[str, Any]) -> "ARTIProject":
        """Apply a declarative ARTI fit config to this project."""

        resolved = validate_fit_config(config)
        for plugin_name in resolved.plugins:
            self.plugin(plugin_name)
        self.profile(resolved.profile, phases=resolved.phases)
        self.runtime(causal=resolved.causal)
        self.runtime_fields = resolved.runtime_fields
        self.scale(resolved.scale)
        self.mechanism_overrides = resolved.mechanism
        self._apply_mechanism_overrides(resolved.mechanism)
        self.objectives(resolved.objectives)
        self.insertion_spec = InsertionSpec(
            where=("*",) if resolved.where is None else resolved.where,
            exclude=resolved.exclude,
            positions=resolved.positions,
            scale_pattern=resolved.scale_pattern,
            every=resolved.every,
            freeze_base=resolved.freeze_base,
            max_adapters=resolved.max_adapters,
            max_extra_params=None,
            identity_gate=resolved.identity_gate,
            zero_init_output=resolved.zero_init_output,
            bridge_mode=resolved.bridge_mode,
            boundary_mask_key=resolved.boundary_mask_key,
            require_runtime_context=resolved.require_runtime_context,
        )
        self.default_max_extra_params = resolved.max_extra_params
        self.fit_config = resolved
        return self

    def mechanism(
        self,
        *,
        coord_dim: int | None = None,
        coord_frame_mode: str | None = None,
        observer_phase: bool | None = None,
        virtual_recall: bool | None = None,
        operator_count: int | None = None,
        interface_slots: int | None = None,
        recall_slots: int | None = None,
        recall_steps: int | None = None,
        recall_min_steps: int | None = None,
        recall_tolerance: float | None = None,
        recall_activation: str | None = None,
        recall_recognition_mode: str | None = None,
        recall_bank_fraction: float | None = None,
        recall_routing: str | None = None,
        recall_key_dim: int | None = None,
        recall_group_size: int | None = None,
        recall_group_topk: int | None = None,
        recall_value_composition: str | None = None,
        recall_formula: str | None = None,
        hidden_multiplier: float | None = None,
    ) -> "ARTIProject":
        """Override resolved ARTI mechanism dimensions for this project."""

        overrides = MechanismOverrides(
            coord_dim=coord_dim,
            coord_frame_mode=coord_frame_mode,
            observer_phase=observer_phase,
            virtual_recall=virtual_recall,
            operator_count=operator_count,
            interface_slots=interface_slots,
            recall_slots=recall_slots,
            recall_steps=recall_steps,
            recall_min_steps=recall_min_steps,
            recall_tolerance=recall_tolerance,
            recall_activation=recall_activation,
            recall_recognition_mode=recall_recognition_mode,
            recall_bank_fraction=recall_bank_fraction,
            recall_routing=recall_routing,
            recall_key_dim=recall_key_dim,
            recall_group_size=recall_group_size,
            recall_group_topk=recall_group_topk,
            recall_value_composition=recall_value_composition,
            recall_formula=recall_formula,
            hidden_multiplier=hidden_multiplier,
        ).validate()
        self.mechanism_overrides = overrides
        self._apply_mechanism_overrides(overrides)
        return self

    def _apply_mechanism_overrides(self, overrides: MechanismOverrides) -> None:
        if not overrides.has_values():
            return
        self.profile_config, self.scale_config = apply_mechanism_overrides(
            self.profile_config, self.scale_config, overrides
        )

    def build_plan(self, objective: str | Iterable[str] | None = None) -> tuple[BuildTaskSpec, ...]:
        objectives = resolve_objectives(objective) if objective is not None else self.objective_plan
        tasks = [
            BuildTaskSpec(name="scan", kind="discovery"),
            BuildTaskSpec(name="insert", kind="mutation", depends_on=("scan",)),
        ]
        previous = "insert"
        for task in objectives:
            kind = (
                "calibration"
                if task == "preserve-output"
                else "training"
                if task == "task-fit"
                else "validation"
            )
            tasks.append(BuildTaskSpec(name=task, kind=kind, depends_on=(previous,)))
            previous = task
        return tuple(tasks)

    def scan(
        self,
        sample_batch: Any | None = None,
        *,
        batch_axis: int | Mapping[str, int] | None = None,
        feature_axis: int | Mapping[str, int] | None = None,
        positions: str | tuple[str, ...] | None = None,
    ) -> "ARTIProject":
        if positions is None:
            positions = self.insertion_spec.positions
        self.scan_positions = (positions,) if isinstance(positions, str) else tuple(positions)
        self.scan_report = scan_model(
            self.model,
            sample_batch,
            causal=self.runtime_causal,
            runtime_fields=self.runtime_fields,
            batch_axis=batch_axis,
            feature_axis=feature_axis,
            positions=self.scan_positions,
        )
        return self

    def insert(
        self,
        where: str | list[str] | tuple[str, ...] | None = None,
        *,
        every: int = 1,
        exclude: str | list[str] | tuple[str, ...] | None = None,
        positions: str | tuple[str, ...] | None = None,
        scale_pattern: Mapping[str, str] | None = None,
        freeze_base: bool = True,
        max_adapters: int | None = None,
        max_extra_params: int | str | None = None,
        identity_gate: bool = False,
        zero_init_output: bool = False,
        bridge_mode: str | None = None,
        boundary_mask_key: str | None = None,
        require_runtime_context: bool = False,
    ) -> "ARTIProject":
        if self.scan_report is None:
            self.scan()
        default_where = default_strategy_for(self.plugins)
        if where is None and self.insertion_spec.where != ("*",):
            where = list(self.insertion_spec.where)
        if every == 1:
            every = self.insertion_spec.every
        if freeze_base is True:
            freeze_base = self.insertion_spec.freeze_base
        if max_adapters is None:
            max_adapters = self.insertion_spec.max_adapters
        if max_extra_params is None:
            max_extra_params = self.default_max_extra_params
        if exclude is None:
            exclude = self.insertion_spec.exclude
        if positions is None:
            positions = self.scan_positions
        resolved_positions = (positions,) if isinstance(positions, str) else tuple(positions)
        unavailable = set(resolved_positions) - set(self.scan_positions)
        if unavailable:
            raise ValueError(
                f"positions {sorted(unavailable)} were not scanned; call scan(..., positions=...) first"
            )
        if scale_pattern is None:
            scale_pattern = dict(self.insertion_spec.scale_pattern)
        if identity_gate is False:
            identity_gate = self.insertion_spec.identity_gate
        if zero_init_output is False:
            zero_init_output = self.insertion_spec.zero_init_output
        if bridge_mode is None:
            bridge_mode = self.insertion_spec.bridge_mode
        if bridge_mode not in {"radial", "dense"}:
            raise ValueError("bridge_mode must be 'radial' or 'dense'")
        if boundary_mask_key is None:
            boundary_mask_key = self.insertion_spec.boundary_mask_key
        if identity_gate and zero_init_output:
            raise ValueError("identity_gate and zero_init_output are mutually exclusive")
        if require_runtime_context is False:
            require_runtime_context = self.insertion_spec.require_runtime_context
        patterns = resolve_where(where or default_where)
        assert self.scan_report is not None
        resolved_max_extra_params = resolve_param_budget(
            max_extra_params, total_parameters=self.scan_report.total_parameters
        )
        self.insertion_spec = InsertionSpec(
            where=patterns,
            exclude=_as_patterns(exclude),
            positions=resolved_positions,
            scale_pattern=_scale_pattern_items(scale_pattern),
            every=every,
            freeze_base=freeze_base,
            max_adapters=max_adapters,
            max_extra_params=resolved_max_extra_params,
            identity_gate=identity_gate,
            zero_init_output=zero_init_output,
            bridge_mode=bridge_mode,
            boundary_mask_key=boundary_mask_key,
            require_runtime_context=require_runtime_context,
        )
        if freeze_base:
            for param in self.model.parameters():
                param.requires_grad = False
        self.inserted = insert_adapters(
            self.model,
            self.scan_report.candidates,
            self.insertion_spec,
            self.profile_config,
            self.scale_config,
            scale_name=self.scale_name,
        )
        self.insertion_plan = None
        self.insert_attempted = True
        return self

    def plan_insert(
        self,
        where: str | list[str] | tuple[str, ...] | None = None,
        *,
        every: int = 1,
        exclude: str | list[str] | tuple[str, ...] | None = None,
        positions: str | tuple[str, ...] | None = None,
        scale_pattern: Mapping[str, str] | None = None,
        freeze_base: bool = True,
        max_adapters: int | None = None,
        max_extra_params: int | str | None = None,
        identity_gate: bool = False,
        zero_init_output: bool = False,
        bridge_mode: str | None = None,
        boundary_mask_key: str | None = None,
        require_runtime_context: bool = False,
    ) -> AdapterInsertionPlan:
        if self.scan_report is None:
            self.scan()
        default_where = default_strategy_for(self.plugins)
        if where is None and self.insertion_spec.where != ("*",):
            where = list(self.insertion_spec.where)
        if every == 1:
            every = self.insertion_spec.every
        if freeze_base is True:
            freeze_base = self.insertion_spec.freeze_base
        if max_adapters is None:
            max_adapters = self.insertion_spec.max_adapters
        if max_extra_params is None:
            max_extra_params = self.default_max_extra_params
        if exclude is None:
            exclude = self.insertion_spec.exclude
        if positions is None:
            positions = self.scan_positions
        resolved_positions = (positions,) if isinstance(positions, str) else tuple(positions)
        unavailable = set(resolved_positions) - set(self.scan_positions)
        if unavailable:
            raise ValueError(
                f"positions {sorted(unavailable)} were not scanned; call scan(..., positions=...) first"
            )
        if scale_pattern is None:
            scale_pattern = dict(self.insertion_spec.scale_pattern)
        if identity_gate is False:
            identity_gate = self.insertion_spec.identity_gate
        if zero_init_output is False:
            zero_init_output = self.insertion_spec.zero_init_output
        if bridge_mode is None:
            bridge_mode = self.insertion_spec.bridge_mode
        if bridge_mode not in {"radial", "dense"}:
            raise ValueError("bridge_mode must be 'radial' or 'dense'")
        if boundary_mask_key is None:
            boundary_mask_key = self.insertion_spec.boundary_mask_key
        if identity_gate and zero_init_output:
            raise ValueError("identity_gate and zero_init_output are mutually exclusive")
        if require_runtime_context is False:
            require_runtime_context = self.insertion_spec.require_runtime_context
        patterns = resolve_where(where or default_where)
        assert self.scan_report is not None
        spec = InsertionSpec(
            where=patterns,
            exclude=_as_patterns(exclude),
            positions=resolved_positions,
            scale_pattern=_scale_pattern_items(scale_pattern),
            every=every,
            freeze_base=freeze_base,
            max_adapters=max_adapters,
            max_extra_params=resolve_param_budget(
                max_extra_params, total_parameters=self.scan_report.total_parameters
            ),
            identity_gate=identity_gate,
            zero_init_output=zero_init_output,
            bridge_mode=bridge_mode,
            boundary_mask_key=boundary_mask_key,
            require_runtime_context=require_runtime_context,
        )
        self.insertion_spec = spec
        self.insertion_plan = plan_adapters(
            self.scan_report.candidates,
            spec,
            self.profile_config,
            self.scale_config,
            scale_name=self.scale_name,
        )
        return self.insertion_plan

    def write_plan(
        self,
        path: str | Path,
        *,
        where: str | list[str] | tuple[str, ...] | None = None,
        every: int = 1,
        exclude: str | list[str] | tuple[str, ...] | None = None,
        positions: str | tuple[str, ...] | None = None,
        scale_pattern: Mapping[str, str] | None = None,
        freeze_base: bool = True,
        max_adapters: int | None = None,
        max_extra_params: int | str | None = None,
        objective: str | Iterable[str] | None = None,
    ) -> Path:
        """Write an auditable dry-run build plan without mutating the model."""

        if objective is not None:
            self.objectives(objective)
        if not self.insert_attempted:
            if where is None and self.insertion_spec.where != ("*",):
                where = list(self.insertion_spec.where)
            if every == 1:
                every = self.insertion_spec.every
            if freeze_base is True:
                freeze_base = self.insertion_spec.freeze_base
            if max_adapters is None:
                max_adapters = self.insertion_spec.max_adapters
            if max_extra_params is None:
                max_extra_params = self.default_max_extra_params
            self.plan_insert(
                where=where,
                every=every,
                exclude=exclude,
                positions=positions,
                scale_pattern=scale_pattern,
                freeze_base=freeze_base,
                max_adapters=max_adapters,
                max_extra_params=max_extra_params,
            )
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        report = self.report()
        if output.suffix.lower() in {".md", ".markdown"}:
            output.write_text(report.to_markdown(), encoding="utf-8")
            return output
        payload = {
            "format_version": 1,
            "package_name": "arti",
            "kind": "fit-plan",
            "report": report.to_dict(),
        }
        output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return output

    def fit(
        self,
        train_loader: Iterable[Any] | None = None,
        *,
        steps: int = 0,
        lr: float = 3e-4,
        loss_fn: Callable[[Any, Any], torch.Tensor] | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> ARTIFitResult:
        if not self.insert_attempted:
            self.insert()
        if train_loader is not None and steps > 0:
            self._train(train_loader, steps=steps, lr=lr, loss_fn=loss_fn, optimizer=optimizer)
        return ARTIFitResult(model=self.model, report=self.report())

    def calibrate(
        self,
        calibration_loader: Iterable[Any],
        *,
        steps: int = 100,
        lr: float = 3e-4,
        objective: str = "preserve-output",
        optimizer: torch.optim.Optimizer | None = None,
    ) -> "ARTIProject":
        if objective != "preserve-output":
            raise ValueError("calibrate currently supports objective='preserve-output'")
        self.calibration_objective = objective
        if not self.insert_attempted:
            self.insert()
        params = [param for param in self.model.parameters() if param.requires_grad]
        if not params:
            raise ValueError("no trainable parameters are available for calibration")
        opt = optimizer or torch.optim.AdamW(params, lr=lr)
        iterator = iter(calibration_loader)
        self.model.train()
        for _ in range(steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(calibration_loader)
                batch = next(iterator)
            inputs = batch_inputs(batch)
            with torch.no_grad(), adapters_enabled(self.model, False):
                target = tensor_output(
                    run_model(
                        self.model,
                        inputs,
                        causal=self.runtime_causal,
                        runtime_fields=self.runtime_fields,
                    )
                ).detach()
            output = tensor_output(
                run_model(
                    self.model,
                    inputs,
                    causal=self.runtime_causal,
                    runtime_fields=self.runtime_fields,
                )
            )
            loss = F.mse_loss(output, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            self.calibration_history.append(float(loss.detach().cpu()))
        self._record_task(
            "preserve-output",
            steps=steps,
            metric_name="mse",
            metric_value=self.calibration_history[-1] if self.calibration_history else None,
        )
        return self

    def validate(
        self,
        val_loader: Iterable[Any],
        *,
        steps: int | None = None,
        metric_fn: Callable[[Any, Any], torch.Tensor | float] | None = None,
    ) -> dict[str, float]:
        self.model.eval()
        values = []
        with torch.no_grad():
            for index, batch in enumerate(val_loader):
                if steps is not None and index >= steps:
                    break
                inputs, target = split_batch(batch)
                output = run_model(
                    self.model,
                    inputs,
                    causal=self.runtime_causal,
                    runtime_fields=self.runtime_fields,
                )
                value = (
                    default_metric(output, target)
                    if metric_fn is None
                    else metric_fn(output, target)
                )
                values.append(
                    float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
                )
        result = {"mean_metric": sum(values) / max(1, len(values)), "batches": float(len(values))}
        self.validation_history.append(result)
        self._record_task(
            "validate",
            steps=len(values),
            metric_name="mean_metric",
            metric_value=result["mean_metric"],
        )
        return result

    def profile_forward(
        self, sample_batch: Any, *, warmup: int = 1, repeats: int = 5
    ) -> ForwardProfile:
        if repeats <= 0:
            raise ValueError("profile_forward requires repeats > 0")
        self.model.eval()
        with torch.no_grad():
            for _ in range(max(0, warmup)):
                output = run_model(
                    self.model,
                    sample_batch,
                    causal=self.runtime_causal,
                    runtime_fields=self.runtime_fields,
                )
                synchronize_output(output)
            timings = []
            last_output = None
            for _ in range(repeats):
                start = perf_counter()
                last_output = run_model(
                    self.model,
                    sample_batch,
                    causal=self.runtime_causal,
                    runtime_fields=self.runtime_fields,
                )
                synchronize_output(last_output)
                timings.append((perf_counter() - start) * 1000.0)
        tensor = tensor_output(last_output)
        profile = ForwardProfile(
            repeats=repeats,
            warmup=max(0, warmup),
            mean_ms=sum(timings) / len(timings),
            min_ms=min(timings),
            max_ms=max(timings),
            output_shape=tuple(int(dim) for dim in tensor.shape),
            output_dtype=str(tensor.dtype),
            output_device=str(tensor.device),
        )
        self.forward_profiles.append(profile)
        self._record_task(
            "profile-forward",
            steps=repeats,
            metric_name="mean_ms",
            metric_value=profile.mean_ms,
        )
        return profile

    def report(self) -> ARTIFitReport:
        if self.scan_report is None:
            self.scan()
        assert self.scan_report is not None
        mechanism = MechanismSummary.from_config(
            self.profile_config,
            self.scale_config,
            scale_name=self.scale_name,
        )
        formula_contract = _recall_formula_contract(
            self.model,
            module_paths=(item.module_path for item in self.inserted),
        )
        if formula_contract is not None:
            mechanism = replace(
                mechanism,
                recall_formula_contract=formula_contract.to_dict(),
                recall_formula_contract_fingerprint=formula_contract.fingerprint,
            )
        return ARTIFitReport(
            profile=self.profile_config.name,
            scale=self.scale_name,
            plugins=tuple(self.plugins),
            plugin_details=plugin_report(self.plugins),
            scanned=self.scan_report,
            inserted=self.inserted,
            frozen_base=self.insertion_spec.freeze_base,
            mechanism=mechanism,
            insertion=self.insertion_spec,
            runtime_causal=self.runtime_causal,
            steps=self.fit_steps,
            objective_plan=self.objective_plan,
            calibration_objective=self.calibration_objective,
            loss_history=tuple(self.loss_history),
            calibration_history=tuple(self.calibration_history),
            validation_history=tuple(self.validation_history),
            task_history=tuple(self.task_history),
            build_plan=self.build_plan(),
            parameters=self.parameter_summary(),
            forward_profiles=tuple(self.forward_profiles),
            insertion_plan=self.insertion_plan,
            fit_config=self.effective_config().to_dict(),
            config_fingerprint=self.effective_config().fingerprint,
            applied_artifact=self.applied_artifact,
        )

    def effective_config(self) -> FitProjectConfig:
        insertion_where = None if self.insertion_spec.where == ("*",) else self.insertion_spec.where
        phases = self.profile_config.coord_dim if self.profile_config.observer_phase else None
        return FitProjectConfig(
            plugins=tuple(self.plugins),
            profile=self.profile_config.name,
            phases=phases,
            scale=self.scale_name,
            mechanism=self.mechanism_overrides,
            causal=self.runtime_causal,
            runtime_fields=self.runtime_fields,
            objectives=self.objective_plan,
            where=insertion_where,
            exclude=self.insertion_spec.exclude,
            positions=self.insertion_spec.positions,
            scale_pattern=self.insertion_spec.scale_pattern,
            every=self.insertion_spec.every,
            freeze_base=self.insertion_spec.freeze_base,
            identity_gate=self.insertion_spec.identity_gate,
            zero_init_output=self.insertion_spec.zero_init_output,
            bridge_mode=self.insertion_spec.bridge_mode,
            boundary_mask_key=self.insertion_spec.boundary_mask_key,
            require_runtime_context=self.insertion_spec.require_runtime_context,
            max_adapters=self.insertion_spec.max_adapters,
            max_extra_params=self.default_max_extra_params
            if self.default_max_extra_params is not None
            else self.insertion_spec.max_extra_params,
        )

    def parameter_summary(self) -> ParameterSummary:
        params = list(self.model.parameters())
        total = sum(param.numel() for param in params)
        trainable = sum(param.numel() for param in params if param.requires_grad)
        adapter_params = 0
        trainable_adapter_params = 0
        base_params = 0
        trainable_base_params = 0
        for wrapper in iter_adapter_wrappers(self.model):
            adapter_params += sum(param.numel() for param in wrapper.adapter.parameters())
            trainable_adapter_params += sum(
                param.numel() for param in wrapper.adapter.parameters() if param.requires_grad
            )
            if wrapper.output_gate is not None:
                adapter_params += wrapper.output_gate.numel()
                if wrapper.output_gate.requires_grad:
                    trainable_adapter_params += wrapper.output_gate.numel()
            base_params += sum(param.numel() for param in wrapper.base.parameters())
            trainable_base_params += sum(
                param.numel() for param in wrapper.base.parameters() if param.requires_grad
            )
        return ParameterSummary(
            total_parameters=total,
            trainable_parameters=trainable,
            adapter_parameters=adapter_params,
            trainable_adapter_parameters=trainable_adapter_params,
            base_parameters=base_params,
            trainable_base_parameters=trainable_base_params,
            frozen_base=self.insertion_spec.freeze_base,
        )

    def _record_task(
        self,
        name: str,
        *,
        steps: int,
        status: str = "success",
        metric_name: str | None = None,
        metric_value: float | None = None,
    ) -> None:
        self.task_history.append(
            FitTaskRecord(
                name=name,
                status=status,
                steps=steps,
                metric_name=metric_name,
                metric_value=metric_value,
            )
        )

    def _train(
        self,
        train_loader: Iterable[Any],
        *,
        steps: int,
        lr: float,
        loss_fn: Callable[[Any, Any], torch.Tensor] | None,
        optimizer: torch.optim.Optimizer | None,
    ) -> None:
        params = [param for param in self.model.parameters() if param.requires_grad]
        if not params:
            raise ValueError(
                "no trainable parameters are available; relax the insertion budget or set freeze_base=False"
            )
        opt = optimizer or torch.optim.AdamW(params, lr=lr)
        iterator = iter(train_loader)
        self.model.train()
        for _ in range(steps):
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(train_loader)
                batch = next(iterator)
            inputs, target = split_batch(batch)
            output = run_model(
                self.model, inputs, causal=self.runtime_causal, runtime_fields=self.runtime_fields
            )
            loss = default_loss(output, target) if loss_fn is None else loss_fn(output, target)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            self.fit_steps += 1
            self.loss_history.append(float(loss.detach().cpu()))
        self._record_task(
            "task-fit",
            steps=steps,
            metric_name="loss",
            metric_value=self.loss_history[-1] if self.loss_history else None,
        )


def split_batch(batch: Any) -> tuple[Any, Any]:
    if isinstance(batch, dict):
        if "labels" in batch:
            inputs = {key: value for key, value in batch.items() if key != "labels"}
            return inputs, batch["labels"]
        if "y" in batch:
            inputs = {key: value for key, value in batch.items() if key != "y"}
            return inputs, batch["y"]
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0] if len(batch) == 2 else tuple(batch[:-1]), batch[-1]
    raise ValueError(
        "batch must be (inputs, target), (*inputs, target), or a dict with 'labels'/'y'"
    )


def batch_inputs(batch: Any) -> Any:
    if isinstance(batch, dict):
        return {key: value for key, value in batch.items() if key not in {"labels", "label", "y"}}
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0] if len(batch) == 2 else tuple(batch[:-1])
    return batch


def tensor_output(output: Any) -> torch.Tensor:
    if torch.is_tensor(output):
        return output
    if isinstance(output, dict):
        for key in ("logits", "last_hidden_state", "output"):
            if key in output and torch.is_tensor(output[key]):
                return output[key]
    if isinstance(output, (tuple, list)) and output and torch.is_tensor(output[0]):
        return output[0]
    raise ValueError(
        "model output must be a Tensor, tuple/list with Tensor first, or dict with logits/last_hidden_state/output"
    )


def synchronize_output(output: Any) -> None:
    try:
        tensor = tensor_output(output)
    except ValueError:
        return
    if tensor.device.type == "cuda":
        torch.cuda.synchronize(tensor.device)


def default_loss(output: Any, target: Any) -> torch.Tensor:
    pred = tensor_output(output)
    if (
        torch.is_tensor(target)
        and target.dtype == torch.long
        and pred.ndim >= 2
        and pred.shape[-1] > 1
    ):
        return F.cross_entropy(pred.reshape(-1, pred.shape[-1]), target.reshape(-1))
    return F.mse_loss(pred, target)


def default_metric(output: Any, target: Any) -> torch.Tensor:
    pred = tensor_output(output)
    if (
        torch.is_tensor(target)
        and target.dtype == torch.long
        and pred.ndim >= 2
        and pred.shape[-1] > 1
    ):
        return (pred.argmax(dim=-1).reshape(-1) == target.reshape(-1)).float().mean()
    return F.mse_loss(pred, target)


def project(model: nn.Module) -> ARTIProject:
    return ARTIProject(model)


def resolve_param_budget(value: int | str | None, *, total_parameters: int) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    stripped = value.strip()
    if stripped.endswith("%"):
        percent = float(stripped[:-1]) / 100.0
        return max(0, int(total_parameters * percent))
    return int(stripped.replace("_", ""))


def fit(
    model: nn.Module,
    train_loader: Iterable[Any] | None = None,
    *,
    config: FitProjectConfig | dict[str, Any] | str | Path | None = None,
    sample_batch: Any | None = None,
    target_modules: str | list[str] | None = None,
    exclude_modules: str | list[str] | tuple[str, ...] | None = None,
    positions: str | tuple[str, ...] | None = None,
    scale_pattern: Mapping[str, str] | None = None,
    profile: str | AdapterProfile = "latent-adapt",
    phases: int | None = None,
    scale: str | AdapterScale = "small",
    mechanism: MechanismOverrides | dict[str, Any] | None = None,
    freeze_base: bool = True,
    max_adapters: int | None = None,
    max_extra_params: int | str | None = None,
    causal: bool = False,
    mask_key: str | None = None,
    coord_key: str | None = None,
    observer_coord_key: str | None = None,
    frame_operators_key: str | None = None,
    batch_axis: int | Mapping[str, int] | None = None,
    feature_axis: int | Mapping[str, int] | None = None,
    calibration_loader: Iterable[Any] | None = None,
    calibration_steps: int = 0,
    calibration_lr: float | None = None,
    calibration_objective: str = "preserve-output",
    objective: str | Iterable[str] | None = None,
    val_loader: Iterable[Any] | None = None,
    validation_steps: int | None = None,
    metric_fn: Callable[[Any, Any], torch.Tensor | float] | None = None,
    dry_run: bool = False,
    steps: int = 0,
    lr: float = 3e-4,
) -> ARTIFitResult:
    builder = project(model)
    config_obj = resolve_fit_config(config)
    config_phases: int | None = None
    mechanism_overrides = (
        MechanismOverrides.from_mapping(mechanism) if isinstance(mechanism, dict) else mechanism
    )
    if config_obj is not None:
        builder.configure(config_obj)
        config_phases = config_obj.phases
        if mechanism_overrides is None:
            mechanism_overrides = config_obj.mechanism
        if target_modules is None and config_obj.where is not None:
            target_modules = list(config_obj.where)
        if profile == "latent-adapt":
            profile = config_obj.profile
        if scale == "small":
            scale = config_obj.scale
        if freeze_base is True:
            freeze_base = config_obj.freeze_base
        if max_adapters is None:
            max_adapters = config_obj.max_adapters
        if max_extra_params is None:
            max_extra_params = config_obj.max_extra_params
        if exclude_modules is None:
            exclude_modules = config_obj.exclude
        if positions is None:
            positions = config_obj.positions
        if scale_pattern is None:
            scale_pattern = dict(config_obj.scale_pattern)
        if causal is False:
            causal = config_obj.causal
        if objective is None and config_obj.objectives:
            objective = config_obj.objectives
    if isinstance(profile, str) and profile in {"transformer", "transformers"}:
        builder.plugin("transformers")
        profile = "latent-adapt"
    if isinstance(profile, str) and profile in {"timm", "vit", "vision-transformer"}:
        builder.plugin("timm")
        profile = "latent-adapt"
    if isinstance(profile, str) and profile in {"cnn", "vision-cnn", "convnet"}:
        builder.plugin("vision-cnn")
        profile = "latent-adapt"
    if isinstance(profile, str) and profile in {"rnn", "lstm", "gru", "recurrent"}:
        builder.plugin("recurrent")
        profile = "latent-adapt"
    resolved_phases = phases
    if (
        resolved_phases is None
        and isinstance(profile, str)
        and config_obj is not None
        and profile == config_obj.profile
    ):
        resolved_phases = config_phases
    builder.profile(profile, phases=resolved_phases).runtime(
        causal=causal,
        mask_key=mask_key,
        coord_key=coord_key,
        observer_coord_key=observer_coord_key,
        frame_operators_key=frame_operators_key,
    ).scale(scale)
    if mechanism_overrides is not None:
        builder.mechanism(**mechanism_overrides.to_dict())
    builder.scan(
        sample_batch, batch_axis=batch_axis, feature_axis=feature_axis, positions=positions
    )
    objective_plan = infer_objectives(
        objective=objective,
        has_calibration=calibration_loader is not None and calibration_steps > 0,
        has_training=train_loader is not None and steps > 0,
        has_validation=val_loader is not None,
    )
    builder.objective_plan = objective_plan
    if dry_run:
        builder.plan_insert(
            where=target_modules,
            exclude=exclude_modules,
            positions=positions,
            scale_pattern=scale_pattern,
            freeze_base=freeze_base,
            max_adapters=max_adapters,
            max_extra_params=max_extra_params,
        )
        return ARTIFitResult(model=builder.model, report=builder.report())
    builder.insert(
        where=target_modules,
        exclude=exclude_modules,
        positions=positions,
        scale_pattern=scale_pattern,
        freeze_base=freeze_base,
        max_adapters=max_adapters,
        max_extra_params=max_extra_params,
    )
    for task in objective_plan:
        if task == "preserve-output":
            if calibration_loader is None or calibration_steps <= 0:
                raise ValueError(
                    "objective 'preserve-output' requires calibration_loader and calibration_steps > 0"
                )
            builder.calibrate(
                calibration_loader,
                steps=calibration_steps,
                lr=lr if calibration_lr is None else calibration_lr,
                objective=calibration_objective,
            )
        elif task == "task-fit":
            if train_loader is None or steps <= 0:
                raise ValueError("objective 'task-fit' requires train_loader and steps > 0")
            builder.fit(train_loader, steps=steps, lr=lr)
        elif task == "validate":
            if val_loader is None:
                raise ValueError("objective 'validate' requires val_loader")
            builder.validate(val_loader, steps=validation_steps, metric_fn=metric_fn)
    if not objective_plan and calibration_loader is not None and calibration_steps > 0:
        builder.calibrate(
            calibration_loader,
            steps=calibration_steps,
            lr=lr if calibration_lr is None else calibration_lr,
            objective=calibration_objective,
        )
    if not objective_plan and train_loader is not None and steps > 0:
        builder.fit(train_loader, steps=steps, lr=lr)
    return ARTIFitResult(model=builder.model, report=builder.report())


def resolve_fit_config(
    config: FitProjectConfig | dict[str, Any] | str | Path | None,
) -> FitProjectConfig | None:
    if config is None:
        return None
    if isinstance(config, FitProjectConfig):
        return validate_fit_config(config)
    if isinstance(config, dict):
        return validate_fit_config(config)
    return load_fit_config(config)


def apply_adapter(
    model: nn.Module,
    artifact: str | Path,
    *,
    sample_batch: Any | None = None,
    freeze_base: bool | None = None,
    identity_gate: bool | None = None,
    zero_init_output: bool | None = None,
    bridge_mode: str | None = None,
    boundary_mask_key: str | None = None,
    mechanism_overrides: MechanismOverrides | Mapping[str, Any] | None = None,
    map_location: str | torch.device | None = None,
    trust_artifact_contract: bool = False,
    reset_recall_query: bool = False,
) -> ARTIFitResult:
    payload = validate_artifact(artifact, map_location=map_location)
    report = payload["report"]
    insertion = report.get("insertion") or {}
    inserted = report.get("inserted") or []
    retired_recall_bridges = {"direct-recall", "recall-influence"}
    if any(
        isinstance(row, dict) and row.get("bridge_mode") in retired_recall_bridges
        for row in inserted
    ):
        raise ValueError(
            "this artifact uses a retired direct-Recall contract; "
            "train a fresh recall-write artifact"
        )
    where = [row["name"] for row in inserted] or insertion.get("where", "*")
    fit_config = report.get("fit_config")
    has_fit_config = isinstance(fit_config, dict)
    artifact_bridge_mode = insertion.get("bridge_mode")
    if artifact_bridge_mode is None and isinstance(fit_config, dict):
        fit_insertion = fit_config.get("insertion")
        if isinstance(fit_insertion, dict):
            artifact_bridge_mode = fit_insertion.get("bridge_mode")
    if artifact_bridge_mode is None:
        artifact_bridge_mode = "dense"
    resolved_bridge_mode = str(artifact_bridge_mode) if bridge_mode is None else bridge_mode
    if resolved_bridge_mode not in {"radial", "dense"}:
        raise ValueError("bridge_mode must be 'radial' or 'dense'")
    requested_mechanism = _resolve_mechanism_override(mechanism_overrides)
    artifact_mechanism = report.get("mechanism") or {}
    artifact_formula = (
        artifact_mechanism.get("recall_formula") if isinstance(artifact_mechanism, dict) else None
    )
    if (
        requested_mechanism is not None
        and requested_mechanism.recall_formula is not None
        and requested_mechanism.recall_formula != artifact_formula
    ):
        raise ValueError(
            "recall_formula cannot be replaced while loading an artifact; "
            "use an explicit artifact migration"
        )
    if has_fit_config:
        resolved_fit_config = validate_fit_config(fit_config)
        if identity_gate is not None:
            resolved_fit_config = replace(resolved_fit_config, identity_gate=identity_gate)
        if zero_init_output is not None:
            resolved_fit_config = replace(
                resolved_fit_config,
                zero_init_output=zero_init_output,
            )
        resolved_fit_config = replace(resolved_fit_config, bridge_mode=resolved_bridge_mode)
        if boundary_mask_key is not None:
            resolved_fit_config = replace(
                resolved_fit_config,
                boundary_mask_key=boundary_mask_key,
            )
        if requested_mechanism is not None:
            resolved_fit_config = replace(
                resolved_fit_config,
                mechanism=_merge_mechanism_overrides(
                    resolved_fit_config.mechanism,
                    requested_mechanism,
                ),
            )
        project_builder = project(model).configure(replace(resolved_fit_config, every=1))
    else:
        project_builder = (
            project(model)
            .profile(report.get("profile", "latent-adapt"))
            .runtime(causal=bool(report.get("runtime_causal", False)))
            .scale(report.get("scale", "small"))
        )
        if requested_mechanism is not None:
            project_builder.mechanism(
                **{
                    key: value
                    for key, value in requested_mechanism.to_dict().items()
                    if value is not None
                }
            )
    candidates = report.get("scanned", {}).get("candidates", [])
    batch_axes = {
        row["name"]: int(row["batch_axis"])
        for row in candidates
        if isinstance(row, dict) and row.get("batch_axis") is not None
    }
    feature_axes = {
        row["name"]: int(row["feature_axis"])
        for row in candidates
        if isinstance(row, dict) and row.get("feature_axis") is not None
    }
    positions = tuple(dict.fromkeys(row.get("position", "output") for row in inserted)) or (
        "output",
    )
    if sample_batch is None:
        if not trust_artifact_contract:
            raise ValueError(
                "sample_batch is required unless trust_artifact_contract=True; "
                "contract-only loading is intended for lazy deployment runtimes"
            )
        project_builder.scan_report = _artifact_scan_report(model, report, inserted)
        project_builder.scan_positions = positions
    else:
        project_builder.scan(
            sample_batch,
            batch_axis=batch_axes or None,
            feature_axis=feature_axes or None,
            positions=positions,
        )
    _validate_boundary_contract(project_builder.scan_report, inserted)
    scale_pattern = (
        None
        if has_fit_config
        else {
            row["name"]: row["scale"]
            for row in inserted
            if isinstance(row, dict) and isinstance(row.get("scale"), str)
        }
    )
    project_builder.insert(
        where=where,
        positions=positions,
        scale_pattern=scale_pattern,
        every=1,
        freeze_base=report.get("frozen_base", True) if freeze_base is None else freeze_base,
        max_adapters=len(inserted) if inserted else insertion.get("max_adapters"),
        identity_gate=(
            bool(insertion.get("identity_gate", False)) if identity_gate is None else identity_gate
        ),
        zero_init_output=(
            bool(insertion.get("zero_init_output", False))
            if zero_init_output is None
            else zero_init_output
        ),
        bridge_mode=resolved_bridge_mode,
        boundary_mask_key=(
            insertion.get("boundary_mask_key") if boundary_mask_key is None else boundary_mask_key
        ),
        require_runtime_context=bool(insertion.get("require_runtime_context", False)),
    )
    expected_adapter_keys = set(
        report_adapter_state_dict(model, project_builder.report())
    )
    _validate_recall_formula_contract(model, report)
    target_state = model.state_dict()
    adapter_state = dict(payload["adapter_state_dict"])
    migrated_retention_keys = tuple(
        key
        for key in target_state
        if key.endswith("._state_input_retention") and key not in adapter_state
    )
    for key in migrated_retention_keys:
        adapter_state[key] = torch.zeros_like(target_state[key])
    adapter_state, migrated_legacy_keys = _migrate_legacy_recall_strength_gate(
        adapter_state,
        target_keys=set(target_state),
    )
    adapter_state, migrated_dense_bridge_keys = _migrate_legacy_dense_bridge(
        adapter_state,
        target_keys=set(model.state_dict()),
    )
    adapter_state, disabled_identity_gate_keys = _migrate_disabled_identity_gates(
        adapter_state,
        target_keys=set(model.state_dict()),
        enabled=identity_gate is False,
    )
    uses_state_recall = any(
        wrapper.adapter.layer.config.recall_value_composition == "state"
        for wrapper in iter_adapter_wrappers(model)
    )
    if uses_state_recall:
        (
            adapter_state,
            migrated_state_factor_keys,
            migrated_state_source_factors,
        ) = _migrate_legacy_state_factor_banks(adapter_state, target_state=model.state_dict())
    else:
        migrated_state_factor_keys = ()
        migrated_state_source_factors = ()
    try:
        missing, unexpected = model.load_state_dict(adapter_state, strict=False)
    except RuntimeError as exc:
        raise ValueError(
            adapter_mismatch_message(
                report, where, missing_adapter=[], unexpected_adapter=[], detail=str(exc)
            )
        ) from exc
    unexpected_adapter = [
        key
        for key in unexpected
        if ".adapter." in key or key.startswith("adapter.") or key.endswith(".output_gate")
    ]
    missing_adapter = [
        key
        for key in missing
        if key in expected_adapter_keys
    ]
    if unexpected_adapter or missing_adapter:
        raise ValueError(
            adapter_mismatch_message(
                report,
                where,
                missing_adapter=missing_adapter,
                unexpected_adapter=unexpected_adapter,
            )
        )
    reset_query_fields = reset_recall_queries(model) if reset_recall_query else 0
    mark_recall_state_banks_calibrated(model)
    manifest = payload["manifest"]
    project_builder.applied_artifact = {
        "path": str(artifact),
        "format_version": manifest.get("format_version"),
        "backend": manifest.get("backend"),
        "profile": manifest.get("profile"),
        "scale": manifest.get("scale"),
        "adapter_key_count": manifest.get("adapter_key_count"),
        "adapter_parameters": manifest.get("adapter_parameters"),
        "adapter_state_sha256": manifest.get("adapter_state_sha256"),
        "report_sha256": manifest.get("report_sha256"),
        "config_fingerprint": manifest.get("config_fingerprint"),
        "migrations": [
            *(
                []
                if not reset_query_fields
                else [
                    {
                        "name": "reset-fixed-recall-query",
                        "field_count": reset_query_fields,
                    }
                ]
            ),
            *(
                []
                if not migrated_retention_keys
                else [
                    {
                        "name": "default-state-input-retention",
                        "inserted_key_count": len(migrated_retention_keys),
                    }
                ]
            ),
            *(
                []
                if not migrated_legacy_keys
                else [
                    {
                        "name": "drop-recall-strength-gate",
                        "dropped_key_count": len(migrated_legacy_keys),
                    }
                ]
            ),
            *(
                []
                if not disabled_identity_gate_keys
                else [
                    {
                        "name": "disable-identity-gate",
                        "dropped_key_count": len(disabled_identity_gate_keys),
                    }
                ]
            ),
            *(
                []
                if not migrated_dense_bridge_keys
                else [
                    {
                        "name": "dense-host-bridge-keys",
                        "migrated_key_count": len(migrated_dense_bridge_keys),
                    }
                ]
            ),
            *(
                []
                if not migrated_state_factor_keys
                else [
                    {
                        "name": "legacy-to-split-content-state-bank",
                        "migrated_key_count": len(migrated_state_factor_keys),
                        "source_factor_counts": list(migrated_state_source_factors),
                    }
                ]
            ),
        ],
    }
    return ARTIFitResult(model=model, report=project_builder.report())


def concatenate_adapter_banks(
    model: nn.Module,
    artifacts: Iterable[str | Path],
    *,
    map_location: str | torch.device = "cpu",
    bank_names: Iterable[str] | None = None,
    weights: Mapping[str, float] | Iterable[float] | None = None,
    influences: Mapping[str, float] | Iterable[float] | None = None,
) -> dict[str, Any]:
    """Materialize compatible Recall banks with routing and signed influence controls."""

    paths = tuple(Path(path).resolve() for path in artifacts)
    if len(paths) < 2:
        raise ValueError("concatenate_adapter_banks requires at least two artifacts")
    resolved_names = (
        tuple(str(name) for name in bank_names)
        if bank_names is not None
        else tuple(f"bank_{index}" for index in range(len(paths)))
    )
    if len(resolved_names) != len(paths):
        raise ValueError("bank_names must match the number of artifacts")
    if any(not name.strip() for name in resolved_names):
        raise ValueError("bank_names must not contain empty names")
    if len(set(resolved_names)) != len(resolved_names):
        raise ValueError("bank_names must be unique")
    resolved_weights = _resolve_adapter_bank_weights(
        resolved_names,
        weights,
    )
    resolved_influences = _resolve_adapter_bank_influences(
        resolved_names,
        influences,
    )
    payloads = tuple(validate_artifact(path, map_location=map_location) for path in paths)
    states = tuple(dict(payload["adapter_state_dict"]) for payload in payloads)
    keys = set(states[0])
    if any(set(state) != keys for state in states[1:]):
        raise ValueError("Recall adapter artifacts do not have matching state structure")

    fields = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, ARTILatentRecallField)
    }
    bank_bindings: dict[str, tuple[ARTILatentRecallField, str]] = {}
    for module_name, module in fields.items():
        for leaf in ("bank", "key_bank", "group_bank"):
            key = f"{module_name}.{leaf}" if module_name else leaf
            if key in keys:
                bank_bindings[key] = (module, leaf)
    artifact_bank_keys = {
        key
        for key in keys
        if key.endswith(".bank") or key.endswith(".key_bank") or key.endswith(".group_bank")
    }
    if not artifact_bank_keys:
        raise ValueError("Recall adapter artifacts expose no concatenable banks")
    if set(bank_bindings) != artifact_bank_keys:
        missing = sorted(artifact_bank_keys - set(bank_bindings))
        raise ValueError(
            "active model does not expose every Recall artifact bank: " + ", ".join(missing[:3])
        )

    active_state = model.state_dict()
    for key in sorted(keys - artifact_bank_keys):
        reference = states[0][key]
        if any(not torch.equal(reference, state[key].to(reference)) for state in states[1:]):
            raise ValueError(f"Recall adapter artifacts differ at shared state {key!r}")
        if key not in active_state or not torch.equal(
            reference,
            active_state[key].to(reference),
        ):
            raise ValueError(
                f"active model does not use the artifacts' shared Recall reader at {key!r}"
            )

    slots_by_field: dict[str, int] = {}
    route_ranges_by_field: dict[str, tuple[tuple[int, int], ...]] = {}
    for key in sorted(artifact_bank_keys):
        module, leaf = bank_bindings[key]
        current = getattr(module, leaf)
        if not isinstance(current, nn.Parameter):
            raise ValueError(f"active Recall bank {key!r} is not a Parameter")
        values = [state[key].to(current) for state in states]
        joined, physical_ranges = _concatenate_bank_values(
            module,
            leaf,
            values,
            concat_dim=0,
        )
        setattr(
            module,
            leaf,
            nn.Parameter(joined, requires_grad=current.requires_grad),
        )
        if leaf == "bank":
            module.slots = int(joined.shape[0])
            if module.slots % module.composition_factor:
                raise ValueError(f"materialized Recall slots at {key!r} are not factor-aligned")
            slots_per_factor = module.slots // module.composition_factor
            module.factor_slices = tuple(
                (
                    factor_index * slots_per_factor,
                    (factor_index + 1) * slots_per_factor,
                )
                for factor_index in range(module.composition_factor)
            )
            slots_by_field[key.removesuffix(".bank")] = module.slots
        route_leaf = "group_bank" if module.routing == "grouped" else "bank"
        if leaf == route_leaf:
            field_name = key.removesuffix(f".{leaf}")
            route_ranges_by_field[field_name] = _relative_factor_ranges(
                physical_ranges,
                total_rows=int(joined.shape[0]),
                factor_count=module.composition_factor,
            )

    if set(route_ranges_by_field) != set(slots_by_field):
        raise RuntimeError("Recall concat did not resolve every field's routing axis")
    for field_name, ranges in route_ranges_by_field.items():
        field = fields[field_name]
        field.configure_expert_routes(resolved_names, ranges)
        field.set_expert_weights(resolved_weights)
        field.set_expert_influences(resolved_influences)

    return {
        "artifacts": tuple(str(path) for path in paths),
        "bank_names": resolved_names,
        "weights": dict(zip(resolved_names, resolved_weights, strict=True)),
        "influences": dict(zip(resolved_names, resolved_influences, strict=True)),
        "source_count": len(paths),
        "bank_tensor_count": len(artifact_bank_keys),
        "field_count": len(slots_by_field),
        "slots_by_field": slots_by_field,
    }


def set_adapter_bank_weights(
    model: nn.Module,
    weights: Mapping[str, float] | Iterable[float],
) -> dict[str, Any]:
    """Update routing priors for a previously concatenated Recall assembly."""

    fields = tuple(
        module
        for module in model.modules()
        if isinstance(module, ARTILatentRecallField) and module.expert_names
    )
    if not fields:
        raise ValueError("model has no concatenated Recall bank assembly")
    names = fields[0].expert_names
    if any(field.expert_names != names for field in fields[1:]):
        raise RuntimeError("model Recall fields use inconsistent bank assemblies")
    resolved = _resolve_adapter_bank_weights(names, weights)
    for field in fields:
        field.set_expert_weights(resolved)
    return {
        "bank_names": names,
        "weights": dict(zip(names, resolved, strict=True)),
        "field_count": len(fields),
    }


def set_adapter_bank_influences(
    model: nn.Module,
    influences: Mapping[str, float] | Iterable[float],
) -> dict[str, Any]:
    """Set signed Recall write influence for a concatenated bank assembly."""

    fields = tuple(
        module
        for module in model.modules()
        if isinstance(module, ARTILatentRecallField) and module.expert_names
    )
    if not fields:
        raise ValueError("model has no concatenated Recall bank assembly")
    names = fields[0].expert_names
    if any(field.expert_names != names for field in fields[1:]):
        raise RuntimeError("model Recall fields use inconsistent bank assemblies")
    resolved = _resolve_adapter_bank_influences(names, influences)
    for field in fields:
        field.set_expert_influences(resolved)
    return {
        "bank_names": names,
        "influences": dict(zip(names, resolved, strict=True)),
        "field_count": len(fields),
    }


def _resolve_adapter_bank_weights(
    names: tuple[str, ...],
    weights: Mapping[str, float] | Iterable[float] | None,
) -> tuple[float, ...]:
    if weights is None:
        return (1.0,) * len(names)
    if isinstance(weights, Mapping):
        unknown = set(weights) - set(names)
        missing = set(names) - set(weights)
        if unknown or missing:
            raise ValueError("bank weight mapping must exactly match bank_names")
        return tuple(float(weights[name]) for name in names)
    resolved = tuple(float(weight) for weight in weights)
    if len(resolved) != len(names):
        raise ValueError(f"expected {len(names)} bank weights, got {len(resolved)}")
    return resolved


def _resolve_adapter_bank_influences(
    names: tuple[str, ...],
    influences: Mapping[str, float] | Iterable[float] | None,
) -> tuple[float, ...]:
    if influences is None:
        return (1.0,) * len(names)
    if isinstance(influences, Mapping):
        unknown = set(influences) - set(names)
        missing = set(names) - set(influences)
        if unknown or missing:
            raise ValueError("bank influence mapping must exactly match bank_names")
        resolved = tuple(float(influences[name]) for name in names)
    else:
        resolved = tuple(float(influence) for influence in influences)
        if len(resolved) != len(names):
            raise ValueError(f"expected {len(names)} bank influences, got {len(resolved)}")
    if any(not math.isfinite(influence) for influence in resolved):
        raise ValueError("bank influences must be finite")
    return resolved


def _relative_factor_ranges(
    physical_ranges: tuple[tuple[tuple[int, int], ...], ...],
    *,
    total_rows: int,
    factor_count: int,
) -> tuple[tuple[int, int], ...]:
    if total_rows % factor_count:
        raise ValueError("Recall routing rows must be factor-aligned")
    rows_per_factor = total_rows // factor_count
    relative: list[tuple[int, int]] = []
    for expert_ranges in physical_ranges:
        if len(expert_ranges) != factor_count:
            raise ValueError("Recall expert route ranges do not match factor count")
        normalized = tuple(
            (
                start - factor_index * rows_per_factor,
                stop - factor_index * rows_per_factor,
            )
            for factor_index, (start, stop) in enumerate(expert_ranges)
        )
        if len(set(normalized)) != 1:
            raise ValueError("Recall expert route ranges differ across factors")
        relative.append(normalized[0])
    return tuple(relative)


def _migrate_legacy_recall_strength_gate(
    state: Mapping[str, torch.Tensor],
    *,
    target_keys: set[str],
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    """Drop only obsolete internal Recall strength-controller tensors."""

    dropped = tuple(
        sorted(
            key
            for key in state
            if key not in target_keys and ".adapter.layer.state.recall.gate." in key
        )
    )
    if not dropped:
        return dict(state), ()
    ignored = set(dropped)
    return {key: value for key, value in state.items() if key not in ignored}, dropped


def _migrate_legacy_dense_bridge(
    state: Mapping[str, torch.Tensor],
    *,
    target_keys: set[str],
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    """Map historical ``adapter.out.*`` tensors into the versioned bridge."""

    migrated = dict(state)
    changed = []
    for key in tuple(state):
        if key in target_keys:
            continue
        suffix = next(
            (value for value in ("weight", "bias") if key.endswith(f".adapter.out.{value}")), None
        )
        if suffix is None:
            continue
        target = key.removesuffix(f".adapter.out.{suffix}") + f".adapter.out.linear.{suffix}"
        if target not in target_keys or target in migrated:
            continue
        migrated[target] = migrated.pop(key)
        changed.append(key)
    return migrated, tuple(sorted(changed))


def _migrate_disabled_identity_gates(
    state: Mapping[str, torch.Tensor],
    *,
    target_keys: set[str],
    enabled: bool,
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    """Drop artifact identity gates only for an explicit gate-free continuation."""

    if not enabled:
        return dict(state), ()
    dropped = tuple(
        sorted(
            key
            for key in state
            if key not in target_keys and (key == "output_gate" or key.endswith(".output_gate"))
        )
    )
    if not dropped:
        return dict(state), ()
    ignored = set(dropped)
    return {key: value for key, value in state.items() if key not in ignored}, dropped


def _migrate_legacy_state_factor_banks(
    state: Mapping[str, torch.Tensor],
    *,
    target_state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], tuple[str, ...], tuple[int, ...]]:
    """Expand legacy state banks into the current constrained polynomial bank."""

    migrated = dict(state)
    changed: list[str] = []
    source_factor_counts: set[int] = set()
    for bank_key, source_bank in state.items():
        if not bank_key.endswith(".recall.bank"):
            continue
        target_bank = target_state.get(bank_key)
        if (
            target_bank is None
            or source_bank.ndim != 2
            or target_bank.ndim != 2
            or source_bank.shape[1] != target_bank.shape[1]
        ):
            continue

        group_key = bank_key.removesuffix(".bank") + ".group_bank"
        key_key = bank_key.removesuffix(".bank") + ".key_bank"
        source_group_bank = state.get(group_key)
        target_group_bank = target_state.get(group_key)
        if source_group_bank is None and target_group_bank is None:
            source_group_count = source_bank.shape[0]
            target_group_count = target_bank.shape[0]
            source_group_size = 1
            target_group_size = 1
        elif (
            source_group_bank is not None
            and target_group_bank is not None
            and source_group_bank.ndim == 2
            and target_group_bank.ndim == 2
            and source_group_bank.shape[1] == target_group_bank.shape[1]
            and source_bank.shape[0] % source_group_bank.shape[0] == 0
            and target_bank.shape[0] % target_group_bank.shape[0] == 0
        ):
            source_group_count = source_group_bank.shape[0]
            target_group_count = target_group_bank.shape[0]
            source_group_size = source_bank.shape[0] // source_group_count
            target_group_size = target_bank.shape[0] // target_group_count
        else:
            continue

        if (
            source_bank.shape == target_bank.shape
            and source_group_count == target_group_count
            and source_group_size == target_group_size
        ):
            continue
        if target_group_count % STATE_RECALL_COMPOSITION_FACTOR:
            continue
        target_groups_per_factor = target_group_count // STATE_RECALL_COMPOSITION_FACTOR
        source_factor_candidates = tuple(
            factor_count for factor_count in (4, 8, 16) if source_group_count % factor_count == 0
        )
        if not source_factor_candidates:
            continue
        source_factor_count = min(
            source_factor_candidates,
            key=lambda factor_count: (
                abs(source_group_count // factor_count - target_groups_per_factor),
                -factor_count,
            ),
        )
        source_groups_per_factor = source_group_count // source_factor_count
        if source_groups_per_factor <= 0 or target_groups_per_factor <= 0:
            continue
        group_indices = (
            torch.linspace(
                0,
                source_groups_per_factor - 1,
                target_groups_per_factor,
                device=source_bank.device,
            )
            .round()
            .to(torch.long)
        )
        row_indices = (
            torch.linspace(
                0,
                source_group_size - 1,
                target_group_size,
                device=source_bank.device,
            )
            .round()
            .to(torch.long)
        )

        source_values = source_bank.reshape(
            source_factor_count,
            source_groups_per_factor,
            source_group_size,
            source_bank.shape[1],
        )
        selected_values = source_values.index_select(1, group_indices).index_select(
            2,
            row_indices,
        )
        source_modulation_count = source_factor_count - 3
        target_modulation_count = STATE_RECALL_COMPOSITION_FACTOR - 4
        repeats = tuple(
            target_modulation_count // source_modulation_count
            + (index < target_modulation_count % source_modulation_count)
            for index in range(source_modulation_count)
        )
        expanded_modulations = []
        for source_index, repeat_count in enumerate(repeats, start=1):
            old_modulation = selected_values[source_index]
            modulation_root = (
                (1.0 + torch.tanh(old_modulation.float()) / source_modulation_count)
                .clamp_min(torch.finfo(torch.float32).eps)
                .pow(1.0 / repeat_count)
            )
            expanded_modulation = torch.atanh(
                (target_modulation_count * (modulation_root - 1.0)).clamp(-0.999, 0.999)
            ).to(old_modulation)
            expanded_modulations.extend([expanded_modulation] * repeat_count)
        target_values = torch.stack(
            [
                selected_values[0],
                torch.zeros_like(selected_values[0]),
                *expanded_modulations,
                selected_values[-2],
                selected_values[-1],
            ],
            dim=0,
        ).reshape_as(target_bank)
        migrated[bank_key] = target_values
        changed.append(bank_key)
        source_factor_counts.add(source_factor_count)

        source_key_bank = state.get(key_key)
        target_key_bank = target_state.get(key_key)
        if source_key_bank is not None and target_key_bank is not None:
            source_keys = source_key_bank.reshape(
                source_factor_count,
                source_groups_per_factor,
                source_group_size,
                source_key_bank.shape[1],
            )
            source_keys = source_keys.index_select(
                1,
                group_indices.to(source_key_bank.device),
            ).index_select(
                2,
                row_indices.to(source_key_bank.device),
            )
            expanded_keys = [
                source_keys[source_index]
                for source_index, repeat_count in enumerate(
                    repeats,
                    start=1,
                )
                for _repeat in range(repeat_count)
            ]
            migrated[key_key] = torch.stack(
                [
                    source_keys[0],
                    source_keys[0],
                    *expanded_keys,
                    source_keys[-2],
                    source_keys[-1],
                ],
                dim=0,
            ).reshape_as(target_key_bank)
            changed.append(key_key)

        if source_group_bank is not None and target_group_bank is not None:
            source_groups = source_group_bank.reshape(
                source_factor_count,
                source_groups_per_factor,
                source_group_bank.shape[1],
            ).index_select(1, group_indices.to(source_group_bank.device))
            expanded_groups = [
                source_groups[source_index]
                for source_index, repeat_count in enumerate(
                    repeats,
                    start=1,
                )
                for _repeat in range(repeat_count)
            ]
            migrated[group_key] = torch.stack(
                [
                    source_groups[0],
                    source_groups[0],
                    *expanded_groups,
                    source_groups[-2],
                    source_groups[-1],
                ],
                dim=0,
            ).reshape_as(target_group_bank)
            changed.append(group_key)

    return (
        migrated,
        tuple(sorted(changed)),
        tuple(sorted(source_factor_counts)),
    )


def _resolve_mechanism_override(
    value: MechanismOverrides | Mapping[str, Any] | None,
) -> MechanismOverrides | None:
    if value is None:
        return None
    resolved = MechanismOverrides.from_mapping(dict(value)) if isinstance(value, Mapping) else value
    if not isinstance(resolved, MechanismOverrides):
        raise TypeError("mechanism_overrides must be MechanismOverrides or a mapping")
    resolved.validate()
    return resolved if resolved.has_values() else None


def _merge_mechanism_overrides(
    base: MechanismOverrides,
    override: MechanismOverrides,
) -> MechanismOverrides:
    payload = base.to_dict()
    payload.update({key: value for key, value in override.to_dict().items() if value is not None})
    return MechanismOverrides.from_mapping(payload).validate()


def _recall_formula_contract(
    model: nn.Module,
    *,
    module_paths: Iterable[str] | None = None,
) -> RecallFormulaContract | None:
    contracts: dict[str, RecallFormulaContract] = {}
    if module_paths is None:
        wrappers: Iterable[nn.Module] = iter_adapter_wrappers(model)
    else:
        resolved_wrappers: list[nn.Module] = []
        for module_path in module_paths:
            try:
                wrapper = model.get_submodule(module_path) if module_path else model
            except AttributeError as exc:
                raise ValueError(
                    f"reported adapter path {module_path!r} is not present in the model"
                ) from exc
            if not isinstance(wrapper, ARTIAdapterWrapper):
                raise ValueError(
                    f"reported adapter path {module_path!r} is not an ARTI adapter wrapper"
                )
            resolved_wrappers.append(wrapper)
        wrappers = resolved_wrappers
    for wrapper in wrappers:
        layer = getattr(wrapper.adapter, "layer", None)
        state = getattr(layer, "state", None)
        recall = getattr(state, "recall", None)
        contract = getattr(recall, "formula_contract", None)
        if contract is not None:
            contracts[contract.fingerprint] = contract
    if not contracts:
        return None
    if len(contracts) != 1:
        raise ValueError("one ARTI fit artifact cannot mix different Recall formula contracts")
    return next(iter(contracts.values()))


def _validate_recall_formula_contract(
    model: nn.Module,
    report: dict[str, Any],
) -> None:
    mechanism = report.get("mechanism") or {}
    if not isinstance(mechanism, dict):
        return
    expected = mechanism.get("recall_formula_contract_fingerprint")
    if expected is None:
        return
    inserted = report.get("inserted") or []
    module_paths = tuple(
        str(row.get("module_path", row.get("name", "")))
        for row in inserted
        if isinstance(row, dict)
    )
    contract = _recall_formula_contract(
        model,
        module_paths=module_paths,
    )
    actual = None if contract is None else contract.fingerprint
    if actual != expected:
        raise ValueError(
            "Recall formula contract fingerprint changed: "
            f"artifact={expected!r}, registered={actual!r}"
        )


def _artifact_scan_report(
    model: nn.Module,
    report: dict[str, Any],
    inserted: list[dict[str, Any]],
) -> ScanReport:
    """Build a deployment scan from a validated artifact's exact boundaries."""

    scanned_rows = {
        (str(row.get("name", "")), str(row.get("position", "output"))): row
        for row in report.get("scanned", {}).get("candidates", [])
        if isinstance(row, dict)
    }
    candidates: list[InsertionCandidate] = []
    for row in inserted:
        if not isinstance(row, dict):
            raise ValueError("ARTI adapter report.inserted rows must be dictionaries")
        name = str(row.get("name", ""))
        module_path = str(row.get("module_path", name.removesuffix("::input")))
        position = str(row.get("position", "output"))
        metadata = scanned_rows.get((name, position), {})
        try:
            module = model.get_submodule(module_path)
        except (AttributeError, KeyError) as exc:
            raise ValueError(f"ARTI adapter module path {module_path!r} was not found") from exc
        if isinstance(module, ARTIAdapterWrapper):
            raise ValueError(f"ARTI adapter module path {module_path!r} is already adapted")
        dim = row.get("dim")
        if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
            raise ValueError(f"ARTI adapter boundary {name!r} has an invalid feature dim")
        declared_dim = _declared_module_dim(module)
        if declared_dim is not None and declared_dim != dim:
            raise ValueError(
                f"ARTI adapter boundary {name!r} changed dim: "
                f"artifact={dim!r}, target={declared_dim!r}"
            )
        reference = next(module.parameters(), None)
        tensor_path = tuple(row.get("tensor_path", metadata.get("tensor_path", ())))
        candidates.append(
            InsertionCandidate(
                name=name,
                module_path=module_path,
                position=position,
                module_type=module.__class__.__name__,
                output_shape=tuple(metadata.get("output_shape", ())),
                dim=dim,
                parameters=sum(parameter.numel() for parameter in module.parameters()),
                source="artifact-contract",
                tensor_rank=metadata.get("tensor_rank"),
                path_depth=module_path.count(".") + 1,
                output_path=tensor_path if position == "output" else (),
                tensor_path=tensor_path,
                batch_axis=metadata.get("batch_axis"),
                feature_axis=metadata.get("feature_axis"),
                device=None if reference is None else str(reference.device),
                dtype=None if reference is None else str(reference.dtype),
                is_leaf=not any(module.children()),
            )
        )
    parameters = tuple(model.parameters())
    reference = next(iter(parameters), None)
    scanned = report.get("scanned", {})
    return ScanReport(
        candidates=tuple(candidates),
        total_parameters=sum(parameter.numel() for parameter in parameters),
        trainable_parameters=sum(
            parameter.numel() for parameter in parameters if parameter.requires_grad
        ),
        device=str(reference.device) if reference is not None else "cpu",
        dtype=str(reference.dtype) if reference is not None else "unknown",
        batch_schema=None,
        scanned_modules=int(scanned.get("scanned_modules", len(candidates))),
        candidate_events=len(candidates),
        duplicate_events=0,
    )


def _declared_module_dim(module: nn.Module) -> int | None:
    if isinstance(module, nn.Linear):
        return int(module.out_features)
    for name in ("dim", "hidden_size", "embed_dim"):
        value = getattr(module, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value > 0:
            return value
    return None


def _as_patterns(value: str | list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    if value is None:
        return ()
    return (value,) if isinstance(value, str) else tuple(value)


def _scale_pattern_items(value: Mapping[str, str] | None) -> tuple[tuple[str, str], ...]:
    if value is None:
        return ()
    items = tuple((str(pattern), str(scale)) for pattern, scale in value.items())
    for _, scale_name in items:
        resolve_scale(scale_name)
    return items


def _validate_boundary_contract(
    scan_report: ScanReport | None, inserted: list[dict[str, Any]]
) -> None:
    if scan_report is None:
        raise ValueError("ARTI adapter application requires a completed boundary scan")
    candidates = {
        (candidate.name, candidate.position): candidate for candidate in scan_report.candidates
    }
    for row in inserted:
        if not isinstance(row, dict):
            continue
        key = (str(row.get("name", "")), str(row.get("position", "output")))
        candidate = candidates.get(key)
        if candidate is None:
            raise ValueError(f"ARTI adapter boundary {key!r} was not found in the target model")
        checks = {
            "module_path": candidate.module_path,
            "position": candidate.position,
            "tensor_path": candidate.tensor_path,
            "dim": candidate.dim,
        }
        for field, actual in checks.items():
            if field not in row:
                continue
            expected = tuple(row[field]) if field == "tensor_path" else row[field]
            if expected != actual:
                raise ValueError(
                    "ARTI adapter is incompatible with the target model structure: "
                    "artifact target_modules resolved to a changed boundary before "
                    "missing_adapter_keys validation; "
                    f"boundary {row.get('name')!r} changed {field}: "
                    f"artifact={expected!r}, target={actual!r}"
                )


def adapter_mismatch_message(
    report: dict[str, Any],
    where: list[str] | tuple[str, ...] | str,
    *,
    missing_adapter: list[str],
    unexpected_adapter: list[str],
    detail: str | None = None,
) -> str:
    message = (
        "ARTI adapter artifact is incompatible with the target model structure. "
        f"profile={report.get('profile')!r}, scale={report.get('scale')!r}, target_modules={where!r}, "
        f"missing_adapter_keys={len(missing_adapter)}, unexpected_adapter_keys={len(unexpected_adapter)}. "
        f"missing={missing_adapter}, unexpected={unexpected_adapter}. "
        "Apply the artifact to a model with the same module paths and adapter plan, or regenerate the plan/artifact for this model."
    )
    if detail:
        message += f" Loader detail: {detail}"
    return message
