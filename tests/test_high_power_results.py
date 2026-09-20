from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_high_power_results", ROOT / "benchmarks" / "verify_high_power_results.py"
)
assert SPEC is not None
verify_high_power_results = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_high_power_results)


def minimal_nature() -> dict:
    tasks = [
        "coordinate_routing",
        "visibility_reasoning",
        "masked_denoising",
        "recall_operator_routing",
    ]
    return {
        "provenance": {"seed_values": [0, 1, 2, 3, 4, 5]},
        "summary": [{"task": task, "model": "arti_full", "mean_accuracy": 1.0} for task in tasks],
        "claim_verdicts": [{"task": task, "claim_verdict": "supported"} for task in tasks],
    }


def minimal_proxy() -> dict:
    return {"provenance": {"seed_values": [0, 1, 2, 3, 4, 5]}}


def statistical_audit(p_value: float = 0.03125) -> dict:
    return {
        "files": [
            {
                "records": [
                    {
                        "name": name,
                        "sign_test_p_two_sided": p_value,
                        "positive_fraction": 1.0,
                        "bootstrap_ci95": [0.1, 0.2],
                    }
                    for name in verify_high_power_results.REQUIRED_STATISTICAL_RECORDS
                ]
            }
        ]
    }


def test_high_power_results_pass_minimal_supported_payload() -> None:
    assert (
        verify_high_power_results.verify(
            nature_results=minimal_nature(),
            proxy_results=minimal_proxy(),
            visibility_results=minimal_proxy(),
            statistical_audit=statistical_audit(),
            min_seeds=6,
        )
        == []
    )


def test_high_power_results_fail_on_low_seed_count() -> None:
    proxy = {"provenance": {"seed_values": [0, 1, 2]}}

    failures = verify_high_power_results.verify(
        nature_results=minimal_nature(),
        proxy_results=proxy,
        visibility_results=minimal_proxy(),
        statistical_audit=statistical_audit(),
        min_seeds=6,
    )

    assert any("seed count below" in failure for failure in failures)


def test_high_power_results_fail_on_non_significant_record() -> None:
    failures = verify_high_power_results.verify(
        nature_results=minimal_nature(),
        proxy_results=minimal_proxy(),
        visibility_results=minimal_proxy(),
        statistical_audit=statistical_audit(p_value=0.25),
        min_seeds=6,
    )

    assert any("sign-test p" in failure for failure in failures)
