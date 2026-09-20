from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("verify_downstream_protocol", ROOT / "benchmarks" / "verify_downstream_protocol.py")
assert SPEC is not None
verify_downstream_protocol = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_downstream_protocol)


def protocol() -> dict:
    return {
        "status": "planned_not_executed",
        "benchmark_registry": "benchmarks/public_benchmark_registry.json",
        "hyperparameter_budget": "benchmarks/hyperparameter_budget.json",
        "minimum_locked_public_benchmarks": 2,
        "required_model_families": ["arti_full", "arti_ablation", "mlp", "transformer"],
        "selection_metric": "validation_accuracy",
        "minimum_reporting": sorted(verify_downstream_protocol.REQUIRED_REPORTING),
        "claim_rules": {
            "no_claim_from_smoke": "Generated smoke CSV results are excluded.",
            "baseline_superiority": "ARTI must beat MLP and Transformer.",
        },
    }


def registry(count: int = 2) -> dict:
    return {"benchmarks": [{"id": str(index)} for index in range(count)]}


def budget(selection_metric: str = "validation_accuracy") -> dict:
    return {
        "selection_metric": selection_metric,
        "model_families": {
            "arti_full": {},
            "arti_ablation": {},
            "mlp": {},
            "transformer": {},
        },
    }


def test_valid_downstream_protocol_passes() -> None:
    assert verify_downstream_protocol.verify(protocol(), registry(), budget()) == []


def test_downstream_protocol_requires_enough_registered_candidates() -> None:
    failures = verify_downstream_protocol.verify(protocol(), registry(count=1), budget())

    assert any("fewer candidates" in failure for failure in failures)


def test_downstream_protocol_requires_reporting_items() -> None:
    payload = protocol()
    payload["minimum_reporting"] = []

    failures = verify_downstream_protocol.verify(payload, registry(), budget())

    assert any("missing reporting" in failure for failure in failures)


def test_downstream_protocol_requires_selection_metric_alignment() -> None:
    failures = verify_downstream_protocol.verify(protocol(), registry(), budget(selection_metric="loss"))

    assert any("selection_metric" in failure for failure in failures)
