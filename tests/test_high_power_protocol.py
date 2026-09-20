from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_high_power_protocol", ROOT / "benchmarks" / "verify_high_power_protocol.py"
)
assert SPEC is not None
verify_high_power_protocol = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_high_power_protocol)


def test_protocol_passes_when_planned_seeds_cover_power_plan() -> None:
    protocol = {
        "status": "generated_local_not_independent",
        "planned_seed_count": 6,
        "required_records": ["metric"],
        "commands": [
            "uv run --extra dev python benchmarks/run_synthetic.py --steps 100 --seeds 6 --output out.json",
            "uv run --extra dev python benchmarks/analyze_statistics.py --output benchmarks/results/high_power_statistical_audit.json",
        ],
    }
    power_plan = {
        "records": [
            {"name": "metric", "min_n_for_all_positive_sign_test_p_lt_alpha": 6},
        ]
    }

    assert verify_high_power_protocol.verify(protocol, power_plan) == []


def test_protocol_fails_when_seed_count_is_too_low() -> None:
    protocol = {
        "status": "generated_local_not_independent",
        "planned_seed_count": 5,
        "required_records": ["metric"],
        "commands": [
            "uv run --extra dev python benchmarks/run_synthetic.py --steps 100 --seeds 5 --output out.json",
            "uv run --extra dev python benchmarks/analyze_statistics.py --output benchmarks/results/high_power_statistical_audit.json",
        ],
    }
    power_plan = {
        "records": [
            {"name": "metric", "min_n_for_all_positive_sign_test_p_lt_alpha": 6},
        ]
    }

    failures = verify_high_power_protocol.verify(protocol, power_plan)

    assert any("below sign-test requirement" in failure for failure in failures)


def test_protocol_fails_when_command_seed_count_drifts() -> None:
    protocol = {
        "status": "generated_local_not_independent",
        "planned_seed_count": 6,
        "required_records": ["metric"],
        "commands": [
            "uv run --extra dev python benchmarks/run_synthetic.py --steps 100 --seeds 5 --output out.json",
            "uv run --extra dev python benchmarks/analyze_statistics.py --output benchmarks/results/high_power_statistical_audit.json",
        ],
    }
    power_plan = {
        "records": [
            {"name": "metric", "min_n_for_all_positive_sign_test_p_lt_alpha": 6},
        ]
    }

    failures = verify_high_power_protocol.verify(protocol, power_plan)

    assert any("command seed count" in failure for failure in failures)
