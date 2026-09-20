from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("analyze_statistics", ROOT / "benchmarks" / "analyze_statistics.py")
assert SPEC is not None
analyze_statistics = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(analyze_statistics)


def test_sign_test_is_exact_two_sided() -> None:
    assert analyze_statistics.sign_test_p_value([1.0, 1.0, 1.0]) == 0.25
    assert analyze_statistics.sign_test_p_value([1.0, -1.0, 1.0, -1.0]) == 1.0
    assert analyze_statistics.sign_test_p_value([0.0, 0.0]) is None


def test_paired_values_use_intersecting_seeds() -> None:
    grouped = {
        ("task", "left"): {
            0: {"accuracy": 0.8},
            1: {"accuracy": 0.7},
        },
        ("task", "right"): {
            1: {"accuracy": 0.4},
            2: {"accuracy": 0.9},
        },
    }

    assert analyze_statistics.paired_values(grouped, "task", "left", "right") == [0.29999999999999993]
