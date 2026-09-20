from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VERIFY_SPEC = importlib.util.spec_from_file_location(
    "verify_public_benchmark_registry", ROOT / "benchmarks" / "verify_public_benchmark_registry.py"
)
assert VERIFY_SPEC is not None
verify_public_benchmark_registry = importlib.util.module_from_spec(VERIFY_SPEC)
assert VERIFY_SPEC.loader is not None
sys.modules["verify_public_benchmark_registry"] = verify_public_benchmark_registry
VERIFY_SPEC.loader.exec_module(verify_public_benchmark_registry)

TABULAR_SPEC = importlib.util.spec_from_file_location("run_tabular_adapter", ROOT / "benchmarks" / "run_tabular_adapter.py")
assert TABULAR_SPEC is not None
run_tabular_adapter = importlib.util.module_from_spec(TABULAR_SPEC)
assert TABULAR_SPEC.loader is not None
sys.modules["run_tabular_adapter"] = run_tabular_adapter
TABULAR_SPEC.loader.exec_module(run_tabular_adapter)


def valid_registry() -> dict:
    return {
        "status": "registered_not_executed",
        "adapter": "benchmarks/run_tabular_adapter.py",
        "benchmarks": [
            {
                "id": "dataset_a",
                "task_type": "binary_tabular_classification",
                "download_url": "https://example.com/a.csv",
                "target": "target",
                "expected_feature_count": 4,
                "required_lock_fields": sorted(verify_public_benchmark_registry.REQUIRED_LOCK_FIELDS),
                "required_adapter_status": "external_public_locked",
            },
            {
                "id": "dataset_b",
                "task_type": "binary_tabular_classification",
                "download_url": "https://example.com/b.csv",
                "target": "label",
                "expected_feature_count": 8,
                "required_lock_fields": sorted(verify_public_benchmark_registry.REQUIRED_LOCK_FIELDS),
                "required_adapter_status": "external_public_locked",
            },
        ],
    }


def test_valid_public_registry_passes() -> None:
    assert verify_public_benchmark_registry.verify(valid_registry()) == []


def test_registry_requires_two_candidates() -> None:
    registry = valid_registry()
    registry["benchmarks"] = registry["benchmarks"][:1]

    failures = verify_public_benchmark_registry.verify(registry)

    assert any("at least two" in failure for failure in failures)


def test_registry_requires_external_public_lock_status() -> None:
    registry = valid_registry()
    registry["benchmarks"][0]["required_adapter_status"] = "external_csv"

    failures = verify_public_benchmark_registry.verify(registry)

    assert any("external_public_locked" in failure for failure in failures)


def test_dataset_lock_can_record_external_public_status(tmp_path: Path) -> None:
    lock_path = tmp_path / "lock.json"
    payload = {
        "audit": {
            "input_csv": "public.csv",
            "sha256": "a" * 64,
            "target": "target",
            "rows": 10,
            "feature_count": 4,
            "feature_names": ["a", "b", "c", "d"],
            "positive_rate": 0.5,
            "adapter_status": "external_public_locked",
        }
    }

    run_tabular_adapter.write_dataset_lock(lock_path, payload)

    assert json.loads(lock_path.read_text(encoding="utf-8"))["adapter_status"] == "external_public_locked"
