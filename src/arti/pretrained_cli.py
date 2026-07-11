"""CLI helpers for declarative pretrained workflows."""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any, Mapping

from ._toml import loads as load_toml
from .pretrained import ARTIPlan, pretrained, validate_pretrained_lock
from .providers import provider_report


def load_pretrained_config(path: str | Path) -> dict[str, Any]:
    target = Path(path)
    if target.suffix.lower() == ".json":
        payload = json.loads(target.read_text(encoding="utf-8"))
    elif target.suffix.lower() == ".toml":
        payload = load_toml(target.read_text(encoding="utf-8"))
    else:
        raise ValueError("pretrained config must be .json or .toml")
    if not isinstance(payload, dict):
        raise ValueError("pretrained config must contain an object")
    model = _mapping(payload, "model")
    if not isinstance(model.get("id"), str) and not isinstance(model.get("factory"), str):
        raise ValueError("pretrained config model requires either id or factory")
    if model.get("id") is not None and model.get("factory") is not None:
        raise ValueError("pretrained config model.id and model.factory are mutually exclusive")
    return payload


def pretrained_cli_report(action: str, config_path: Path | None, **options: Any) -> dict[str, Any]:
    if action == "doctor":
        rows = list(provider_report())
        requested = options.get("provider")
        if requested is not None:
            rows = [row for row in rows if row["name"] == requested]
            if not rows:
                raise ValueError(f"unknown pretrained provider {requested!r}")
        return {"ok": all(row["available"] for row in rows), "kind": "pretrained-doctor", "providers": rows}
    if action == "validate-lock":
        lock_path = options.get("lock")
        if lock_path is None:
            raise ValueError("validate-lock requires --lock")
        payload = validate_pretrained_lock(lock_path)
        return {"ok": True, "kind": "pretrained-lock", "path": str(lock_path), "payload": payload}
    if config_path is None:
        raise ValueError(f"pretrained {action} requires a config path")
    config = load_pretrained_config(config_path)
    workflow, sample, component_samples = _workflow_from_config(config)
    workflow.scan(sample, component_samples=component_samples)
    if action == "scan":
        report = workflow.doctor()
        report.update(
            {
                "ok": True,
                "kind": "pretrained-scan",
                "candidates": {name: len(builder.scan_report.candidates) for name, builder in workflow.projects.items()},
                "structure_fingerprints": {
                    name: _structure(builder.model) for name, builder in workflow.projects.items()
                },
            }
        )
        _write_json(options.get("output"), report)
        return report
    plan_path = options.get("plan")
    if plan_path is not None and Path(plan_path).exists():
        plan = ARTIPlan.read(plan_path)
        workflow.plan_value = plan
    else:
        plan = _compile_plan(workflow, config)
    if action == "plan":
        output = options.get("output")
        if output is None:
            raise ValueError("pretrained plan requires --output")
        plan.write(output)
        return {"ok": True, "kind": "pretrained-plan", "output": str(output), "fingerprint": plan.fingerprint}
    workflow.apply(plan)
    if action == "apply":
        weights = options.get("weights")
        if weights is None:
            raise ValueError("pretrained apply requires --weights so the applied state is durable")
        exported = workflow.export(weights, include_base=bool(options.get("include_base", False)))
        return _export_summary("pretrained-apply", exported, workflow)
    if action == "export":
        weights = options.get("weights")
        if weights is None:
            raise ValueError("pretrained export requires --weights")
        exported = workflow.export(weights, include_base=bool(options.get("include_base", False)))
        return _export_summary("pretrained-export", exported, workflow)
    if action == "fit":
        data_factory = _mapping(config, "data").get("factory")
        if not isinstance(data_factory, str):
            raise ValueError("pretrained fit requires data.factory in the config")
        train_data = _invoke_factory(data_factory, _mapping(config, "data").get("kwargs", {}))
        result = workflow.fit(train_data)
        weights = options.get("weights")
        if weights is None:
            raise ValueError("pretrained fit requires --weights")
        exported = result.export(weights, include_base=bool(options.get("include_base", False)))
        summary = _export_summary("pretrained-fit", exported, workflow)
        summary.update({"engine": result.engine, "steps": result.steps, "final_loss": result.loss_history[-1] if result.loss_history else None})
        return summary
    raise ValueError(f"unsupported pretrained action {action!r}")


def _workflow_from_config(config: Mapping[str, Any]):
    model = _mapping(config, "model")
    factory = model.get("factory")
    source = _invoke_factory(factory, model.get("factory_kwargs", {})) if isinstance(factory, str) else model["id"]
    components = model.get("components")
    workflow = pretrained(
        source,
        provider=str(model.get("provider", "auto")),
        task=model.get("task"),
        revision=model.get("revision"),
        loader_kwargs=_mapping(config, "loader"),
        components=components,
    )
    sample_section = _mapping(config, "sample")
    sample_factory = sample_section.get("factory")
    sample = _invoke_factory(sample_factory, sample_section.get("kwargs", {})) if isinstance(sample_factory, str) else None
    component_samples = sample if sample_section.get("mode") == "components" else None
    return workflow, None if component_samples is not None else sample, component_samples


def _compile_plan(workflow, config: Mapping[str, Any]) -> ARTIPlan:
    insertion = _mapping(config, "insertion")
    return workflow.plan(
        features=_mapping(config, "features"),
        where=insertion.get("where"),
        scale=str(insertion.get("scale", "small")),
        freeze_base=bool(insertion.get("freeze_base", True)),
        max_adapters=insertion.get("max_adapters"),
        max_extra_params=insertion.get("max_extra_params"),
        allow_empty=bool(insertion.get("allow_empty", False)),
        training=_mapping(config, "training"),
    )


def _invoke_factory(path: str, kwargs: Any) -> Any:
    if ":" not in path:
        raise ValueError(f"factory {path!r} must use package.module:callable syntax")
    module_name, attribute = path.split(":", 1)
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise ValueError(f"factory {path!r} is not callable")
    return factory(**dict(kwargs or {}))


def _mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"pretrained config {key} must be an object/table")
    return dict(value)


def _write_json(path: Path | None, payload: Mapping[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _export_summary(kind: str, exported, workflow) -> dict[str, Any]:
    return {
        "ok": True,
        "kind": kind,
        "provider": workflow.provider.name,
        "weights": str(exported.saved.weights_path),
        "plan": str(exported.plan_path),
        "lock": str(exported.lock_path),
        "plan_fingerprint": workflow.plan_value.fingerprint,
        "weights_sha256": exported.saved.weights_sha256,
    }


def _structure(model) -> str:
    from .pretrained import model_structure_fingerprint

    return model_structure_fingerprint(model)


__all__ = ["load_pretrained_config", "pretrained_cli_report"]
