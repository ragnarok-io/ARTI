"""Gradle-like ARTI project and fit API."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from time import perf_counter
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from pathlib import Path

from .artifacts import ARTIFitReport, ARTIFitResult, BuildTaskSpec, FitTaskRecord, ForwardProfile, MechanismSummary, ParameterSummary, validate_artifact
from .config import FitProjectConfig, MechanismOverrides, apply_mechanism_overrides, load_fit_config, validate_fit_config
from .insertion import AdapterInsertionPlan, InsertionSpec, adapters_enabled, insert_adapters, iter_adapter_wrappers, plan_adapters
from .objectives import infer_objectives, resolve_objectives
from .plugins import default_strategy_for, get_plugin, plugin_report
from .profiles import AdapterProfile, resolve_profile
from .runtime import RuntimeFieldConfig
from .scales import AdapterScale, resolve_scale
from .scanner import ScanReport, run_model, scan_model
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
        if any(value is not None for value in (mask_key, coord_key, observer_coord_key, frame_operators_key)):
            self.runtime_fields = RuntimeFieldConfig(
                mask_key=self.runtime_fields.mask_key if mask_key is None else mask_key,
                coord_key=self.runtime_fields.coord_key if coord_key is None else coord_key,
                observer_coord_key=self.runtime_fields.observer_coord_key if observer_coord_key is None else observer_coord_key,
                frame_operators_key=self.runtime_fields.frame_operators_key if frame_operators_key is None else frame_operators_key,
            )
        return self

    def objectives(self, objective: str | Iterable[str] | None) -> "ARTIProject":
        self.objective_plan = resolve_objectives(objective)
        return self

    def at(self, where: str | Iterable[str], *, every: int = 1) -> "ARTIProject":
        """Select insertion paths without scanning or mutating the model."""

        patterns = (where,) if isinstance(where, str) else tuple(where)
        if not patterns:
            raise ValueError("where must select at least one module pattern")
        if every <= 0:
            raise ValueError("every must be positive")
        self.insertion_spec = InsertionSpec(
            where=patterns,
            every=every,
            freeze_base=self.insertion_spec.freeze_base,
            max_adapters=self.insertion_spec.max_adapters,
            max_extra_params=self.insertion_spec.max_extra_params,
            identity_gate=self.insertion_spec.identity_gate,
            require_runtime_context=self.insertion_spec.require_runtime_context,
        )
        return self

    def freeze(self, base: bool = True) -> "ARTIProject":
        """Declare the base-model freezing policy without applying it yet."""

        self.insertion_spec = InsertionSpec(
            where=self.insertion_spec.where,
            every=self.insertion_spec.every,
            freeze_base=base,
            max_adapters=self.insertion_spec.max_adapters,
            max_extra_params=self.insertion_spec.max_extra_params,
            identity_gate=self.insertion_spec.identity_gate,
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
            every=self.insertion_spec.every,
            freeze_base=self.insertion_spec.freeze_base,
            max_adapters=max_adapters,
            max_extra_params=self.insertion_spec.max_extra_params,
            identity_gate=self.insertion_spec.identity_gate,
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
            every=resolved.every,
            freeze_base=resolved.freeze_base,
            max_adapters=resolved.max_adapters,
            max_extra_params=None,
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
        recall_activation: str | None = None,
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
            recall_activation=recall_activation,
            hidden_multiplier=hidden_multiplier,
        ).validate()
        self.mechanism_overrides = overrides
        self._apply_mechanism_overrides(overrides)
        return self

    def _apply_mechanism_overrides(self, overrides: MechanismOverrides) -> None:
        if not overrides.has_values():
            return
        self.profile_config, self.scale_config = apply_mechanism_overrides(self.profile_config, self.scale_config, overrides)

    def build_plan(self, objective: str | Iterable[str] | None = None) -> tuple[BuildTaskSpec, ...]:
        objectives = resolve_objectives(objective) if objective is not None else self.objective_plan
        tasks = [
            BuildTaskSpec(name="scan", kind="discovery"),
            BuildTaskSpec(name="insert", kind="mutation", depends_on=("scan",)),
        ]
        previous = "insert"
        for task in objectives:
            kind = "calibration" if task == "preserve-output" else "training" if task == "task-fit" else "validation"
            tasks.append(BuildTaskSpec(name=task, kind=kind, depends_on=(previous,)))
            previous = task
        return tuple(tasks)

    def scan(self, sample_batch: Any | None = None) -> "ARTIProject":
        self.scan_report = scan_model(self.model, sample_batch, causal=self.runtime_causal, runtime_fields=self.runtime_fields)
        return self

    def insert(
        self,
        where: str | list[str] | tuple[str, ...] | None = None,
        *,
        every: int = 1,
        freeze_base: bool = True,
        max_adapters: int | None = None,
        max_extra_params: int | str | None = None,
        identity_gate: bool = False,
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
        patterns = resolve_where(where or default_where)
        assert self.scan_report is not None
        resolved_max_extra_params = resolve_param_budget(max_extra_params, total_parameters=self.scan_report.total_parameters)
        self.insertion_spec = InsertionSpec(
            where=patterns,
            every=every,
            freeze_base=freeze_base,
            max_adapters=max_adapters,
            max_extra_params=resolved_max_extra_params,
            identity_gate=identity_gate,
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
        freeze_base: bool = True,
        max_adapters: int | None = None,
        max_extra_params: int | str | None = None,
        identity_gate: bool = False,
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
        patterns = resolve_where(where or default_where)
        assert self.scan_report is not None
        spec = InsertionSpec(
            where=patterns,
            every=every,
            freeze_base=freeze_base,
            max_adapters=max_adapters,
            max_extra_params=resolve_param_budget(max_extra_params, total_parameters=self.scan_report.total_parameters),
            identity_gate=identity_gate,
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
                target = tensor_output(run_model(self.model, inputs, causal=self.runtime_causal, runtime_fields=self.runtime_fields)).detach()
            output = tensor_output(run_model(self.model, inputs, causal=self.runtime_causal, runtime_fields=self.runtime_fields))
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
                output = run_model(self.model, inputs, causal=self.runtime_causal, runtime_fields=self.runtime_fields)
                value = default_metric(output, target) if metric_fn is None else metric_fn(output, target)
                values.append(float(value.detach().cpu()) if torch.is_tensor(value) else float(value))
        result = {"mean_metric": sum(values) / max(1, len(values)), "batches": float(len(values))}
        self.validation_history.append(result)
        self._record_task(
            "validate",
            steps=len(values),
            metric_name="mean_metric",
            metric_value=result["mean_metric"],
        )
        return result

    def profile_forward(self, sample_batch: Any, *, warmup: int = 1, repeats: int = 5) -> ForwardProfile:
        if repeats <= 0:
            raise ValueError("profile_forward requires repeats > 0")
        self.model.eval()
        with torch.no_grad():
            for _ in range(max(0, warmup)):
                output = run_model(self.model, sample_batch, causal=self.runtime_causal, runtime_fields=self.runtime_fields)
                synchronize_output(output)
            timings = []
            last_output = None
            for _ in range(repeats):
                start = perf_counter()
                last_output = run_model(self.model, sample_batch, causal=self.runtime_causal, runtime_fields=self.runtime_fields)
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
        return ARTIFitReport(
            profile=self.profile_config.name,
            scale=self.scale_name,
            plugins=tuple(self.plugins),
            plugin_details=plugin_report(self.plugins),
            scanned=self.scan_report,
            inserted=self.inserted,
            frozen_base=self.insertion_spec.freeze_base,
            mechanism=MechanismSummary.from_config(self.profile_config, self.scale_config, scale_name=self.scale_name),
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
            every=self.insertion_spec.every,
            freeze_base=self.insertion_spec.freeze_base,
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
            trainable_adapter_params += sum(param.numel() for param in wrapper.adapter.parameters() if param.requires_grad)
            if wrapper.output_gate is not None:
                adapter_params += wrapper.output_gate.numel()
                if wrapper.output_gate.requires_grad:
                    trainable_adapter_params += wrapper.output_gate.numel()
            base_params += sum(param.numel() for param in wrapper.base.parameters())
            trainable_base_params += sum(param.numel() for param in wrapper.base.parameters() if param.requires_grad)
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
            raise ValueError("no trainable parameters are available; relax the insertion budget or set freeze_base=False")
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
            output = run_model(self.model, inputs, causal=self.runtime_causal, runtime_fields=self.runtime_fields)
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
    raise ValueError("batch must be (inputs, target), (*inputs, target), or a dict with 'labels'/'y'")


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
    raise ValueError("model output must be a Tensor, tuple/list with Tensor first, or dict with logits/last_hidden_state/output")


def synchronize_output(output: Any) -> None:
    try:
        tensor = tensor_output(output)
    except ValueError:
        return
    if tensor.device.type == "cuda":
        torch.cuda.synchronize(tensor.device)


def default_loss(output: Any, target: Any) -> torch.Tensor:
    pred = tensor_output(output)
    if torch.is_tensor(target) and target.dtype == torch.long and pred.ndim >= 2 and pred.shape[-1] > 1:
        return F.cross_entropy(pred.reshape(-1, pred.shape[-1]), target.reshape(-1))
    return F.mse_loss(pred, target)


def default_metric(output: Any, target: Any) -> torch.Tensor:
    pred = tensor_output(output)
    if torch.is_tensor(target) and target.dtype == torch.long and pred.ndim >= 2 and pred.shape[-1] > 1:
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
    mechanism_overrides = MechanismOverrides.from_mapping(mechanism) if isinstance(mechanism, dict) else mechanism
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
    if resolved_phases is None and isinstance(profile, str) and config_obj is not None and profile == config_obj.profile:
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
    builder.scan(sample_batch)
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
            freeze_base=freeze_base,
            max_adapters=max_adapters,
            max_extra_params=max_extra_params,
        )
        return ARTIFitResult(model=builder.model, report=builder.report())
    builder.insert(
        where=target_modules,
        freeze_base=freeze_base,
        max_adapters=max_adapters,
        max_extra_params=max_extra_params,
    )
    for task in objective_plan:
        if task == "preserve-output":
            if calibration_loader is None or calibration_steps <= 0:
                raise ValueError("objective 'preserve-output' requires calibration_loader and calibration_steps > 0")
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


def resolve_fit_config(config: FitProjectConfig | dict[str, Any] | str | Path | None) -> FitProjectConfig | None:
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
    map_location: str | torch.device | None = None,
) -> ARTIFitResult:
    payload = validate_artifact(artifact, map_location=map_location)
    report = payload["report"]
    insertion = report.get("insertion") or {}
    inserted = report.get("inserted") or []
    where = [row["name"] for row in inserted] or insertion.get("where", "*")
    project_builder = project(model).profile(report.get("profile", "latent-adapt")).runtime(causal=bool(report.get("runtime_causal", False))).scale(report.get("scale", "small"))
    fit_config = report.get("fit_config")
    if isinstance(fit_config, dict):
        runtime_config = fit_config.get("runtime")
        if isinstance(runtime_config, dict):
            project_builder.runtime_fields = RuntimeFieldConfig.from_mapping(runtime_config)
    project_builder.scan(sample_batch)
    project_builder.insert(
        where=where,
        every=1,
        freeze_base=report.get("frozen_base", True) if freeze_base is None else freeze_base,
        max_adapters=len(inserted) if inserted else insertion.get("max_adapters"),
    )
    try:
        missing, unexpected = model.load_state_dict(payload["adapter_state_dict"], strict=False)
    except RuntimeError as exc:
        raise ValueError(adapter_mismatch_message(report, where, missing_adapter=[], unexpected_adapter=[], detail=str(exc))) from exc
    unexpected_adapter = [key for key in unexpected if ".adapter." in key or key.startswith("adapter.") or key.endswith(".output_gate")]
    missing_adapter = [key for key in missing if ".adapter." in key or key.startswith("adapter.") or key.endswith(".output_gate")]
    if unexpected_adapter or missing_adapter:
        raise ValueError(adapter_mismatch_message(report, where, missing_adapter=missing_adapter, unexpected_adapter=unexpected_adapter))
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
    }
    return ARTIFitResult(model=model, report=project_builder.report())


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
