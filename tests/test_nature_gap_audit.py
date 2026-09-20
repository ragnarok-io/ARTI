from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("audit_nature_gaps", ROOT / "benchmarks" / "audit_nature_gaps.py")
assert SPEC is not None
audit_nature_gaps = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(audit_nature_gaps)


def test_gap_audit_marks_current_like_evidence_incomplete() -> None:
    summary = {
        "high_power": {
            "available": True,
            "mechanism_gates": {
                "coordinate_routing": {"claim_verdict": "supported"},
                "visibility_reasoning": {"claim_verdict": "supported"},
            },
        },
        "tabular_adapter": {"dataset_lock": {"adapter_status": "smoke_generated"}},
    }
    scaling = {"rows": [{"cuda_peak_allocated_bytes": None}]}
    bundle_comparison = {"passed": True}
    environment = {"platform": "test", "torch": {"version": "x", "cuda_available": False}}

    budget = {
        "status": "planned_not_executed",
        "max_trials_per_model": 6,
        "model_families": {"arti_full": {}, "arti_ablation": {}, "mlp": {}, "transformer": {}},
    }

    registry = {"status": "registered_not_executed", "benchmarks": [{"id": "a"}, {"id": "b"}]}

    cuda_protocol = {"status": "planned_not_executed", "required_device_prefix": "cuda"}
    downstream_protocol = {
        "status": "planned_not_executed",
        "minimum_locked_public_benchmarks": 2,
        "required_model_families": ["arti_full", "arti_ablation", "mlp", "transformer"],
    }
    independent_protocol = {
        "status": "planned_not_executed",
        "required_return_artifacts": ["benchmarks/results/evidence_bundle_lock.json"],
        "comparison_commands": ["uv run --extra dev python benchmarks/compare_evidence_bundle_locks.py"],
    }

    payload = audit_nature_gaps.audit(
        summary,
        scaling,
        bundle_comparison,
        environment,
        budget,
        registry,
        cuda_protocol,
        downstream_protocol,
        independent_protocol,
    )
    statuses = {item["name"]: item["status"] for item in payload["items"]}

    assert payload["overall_status"] == "incomplete"
    assert statuses["local_high_power_mechanism_evidence"] == "met"
    assert statuses["artifact_lock_and_self_comparison"] == "met"
    assert statuses["cuda_peak_memory_profile"] == "partial"
    assert statuses["pinned_external_public_benchmark"] == "partial"
    assert statuses["baseline_hyperparameter_budget"] == "met"
    assert statuses["downstream_task_improvement"] == "partial"
    assert statuses["independent_second_machine_reproduction"] == "partial"


def test_gap_audit_marks_cuda_and_external_benchmark_met_when_present() -> None:
    summary = {
        "high_power": {"available": False},
        "tabular_adapter": {"dataset_lock": {"adapter_status": "smoke_generated"}},
        "public_tabular_adapter": {
            "available": True,
            "benchmark_count": 1,
            "dataset_lock": {"adapter_status": "external_public_locked"},
        },
        "downstream_results": {
            "claim_verdicts": {"mechanism_utility": "not_supported", "baseline_superiority": "not_supported"}
        },
    }
    scaling = {"rows": [{"cuda_peak_allocated_bytes": 1234}]}

    payload = audit_nature_gaps.audit(summary, scaling, {"passed": False}, {})
    statuses = {item["name"]: item["status"] for item in payload["items"]}

    assert statuses["cuda_peak_memory_profile"] == "met"
    assert statuses["pinned_external_public_benchmark"] == "met"
    assert statuses["local_high_power_mechanism_evidence"] == "unmet"
    assert statuses["baseline_hyperparameter_budget"] == "partial"
