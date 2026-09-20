from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "verify_half_recall_trace_survival",
    ROOT / "benchmarks" / "verify_half_recall_trace_survival.py",
)
verify_half_recall_trace_survival = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verify_half_recall_trace_survival
assert SPEC.loader is not None
SPEC.loader.exec_module(verify_half_recall_trace_survival)


def payload() -> dict:
    base_rows = [
        {
            "variant": "baseline_residual",
            "state_mse": 0.07,
            "noise_leakage": 1.4,
            "weak_trace_survival": 1.0,
            "delta_norm": 4.0,
            "signal_retention": 1.0,
            "positive_signal_retention": 1.0,
            "negative_signal_retention": 1.0,
            "selectivity": 0.7,
            "probe_accuracy": 1.0,
        },
        {
            "variant": "recall_half",
            "state_mse": 0.04,
            "noise_leakage": 0.9,
            "weak_trace_survival": 0.67,
            "delta_norm": 3.9,
            "signal_retention": 0.99,
            "positive_signal_retention": 0.99,
            "negative_signal_retention": 0.99,
            "selectivity": 1.1,
            "probe_accuracy": 1.0,
        },
        {
            "variant": "recall_fixed_shrink",
            "state_mse": 0.08,
            "noise_leakage": 0.91,
            "weak_trace_survival": 0.675,
            "delta_norm": 2.8,
            "signal_retention": 0.675,
            "positive_signal_retention": 0.675,
            "negative_signal_retention": 0.675,
            "selectivity": 0.74,
            "probe_accuracy": 1.0,
        },
        {
            "variant": "recall_softshrink",
            "state_mse": 0.045,
            "noise_leakage": 0.32,
            "weak_trace_survival": 0.23,
            "delta_norm": 2.2,
            "signal_retention": 0.74,
            "positive_signal_retention": 0.74,
            "negative_signal_retention": 0.74,
            "selectivity": 2.3,
            "probe_accuracy": 1.0,
        },
        {
            "variant": "stacked_residual",
            "state_mse": 0.2,
            "noise_leakage": 2.4,
            "weak_trace_survival": 0.58,
            "delta_norm": 7.0,
            "signal_retention": 1.0,
            "positive_signal_retention": 1.0,
            "negative_signal_retention": 1.0,
            "selectivity": 0.4,
            "probe_accuracy": 1.0,
        },
        {
            "variant": "stacked_half",
            "state_mse": 0.1,
            "noise_leakage": 1.6,
            "weak_trace_survival": 0.39,
            "delta_norm": 5.9,
            "signal_retention": 0.99,
            "positive_signal_retention": 0.99,
            "negative_signal_retention": 0.99,
            "selectivity": 0.64,
            "probe_accuracy": 1.0,
        },
        {
            "variant": "stacked_fixed_shrink",
            "state_mse": 0.145,
            "noise_leakage": 1.61,
            "weak_trace_survival": 0.39,
            "delta_norm": 4.8,
            "signal_retention": 0.675,
            "positive_signal_retention": 0.675,
            "negative_signal_retention": 0.675,
            "selectivity": 0.43,
            "probe_accuracy": 1.0,
        },
        {
            "variant": "stacked_softshrink",
            "state_mse": 0.055,
            "noise_leakage": 0.59,
            "weak_trace_survival": 0.14,
            "delta_norm": 3.0,
            "signal_retention": 0.74,
            "positive_signal_retention": 0.74,
            "negative_signal_retention": 0.74,
            "selectivity": 1.25,
            "probe_accuracy": 1.0,
        },
        {
            "variant": "recall_dropout",
            "state_mse": 0.3,
            "noise_leakage": 1.7,
            "weak_trace_survival": 1.23,
            "delta_norm": 5.0,
            "signal_retention": 1.0,
            "positive_signal_retention": 1.0,
            "negative_signal_retention": 1.0,
            "selectivity": 0.6,
            "probe_accuracy": 1.0,
        },
        {
            "variant": "recall_gelu",
            "state_mse": 0.2,
            "noise_leakage": 0.7,
            "weak_trace_survival": 0.52,
            "delta_norm": 2.2,
            "signal_retention": 0.5,
            "positive_signal_retention": 0.9,
            "negative_signal_retention": 0.1,
            "selectivity": 0.7,
            "probe_accuracy": 1.0,
        },
        {
            "variant": "recall_relu",
            "state_mse": 0.2,
            "noise_leakage": 1.0,
            "weak_trace_survival": 0.7,
            "delta_norm": 2.5,
            "signal_retention": 0.5,
            "positive_signal_retention": 1.0,
            "negative_signal_retention": 0.0,
            "selectivity": 0.5,
            "probe_accuracy": 1.0,
        },
        {
            "variant": "recall_half_stochastic",
            "state_mse": 0.05,
            "noise_leakage": 1.1,
            "weak_trace_survival": 0.81,
            "delta_norm": 4.0,
            "signal_retention": 1.0,
            "positive_signal_retention": 1.0,
            "negative_signal_retention": 1.0,
            "selectivity": 0.9,
            "probe_accuracy": 0.95,
        },
    ]
    rows = []
    for scenario in ("separable_trace", "low_separation_trace"):
        for row in base_rows:
            copied = dict(row)
            copied["scenario"] = scenario
            copied["primary_claim_environment"] = scenario == "separable_trace"
            copied["seed_metrics"] = [
                {
                    "replicate": index,
                    "state_mse": copied["state_mse"],
                    "noise_leakage": copied["noise_leakage"],
                    "weak_trace_survival": copied["weak_trace_survival"],
                    "delta_norm": copied["delta_norm"],
                    "signal_retention": copied["signal_retention"],
                    "positive_signal_retention": copied["positive_signal_retention"],
                    "negative_signal_retention": copied["negative_signal_retention"],
                    "selectivity": copied["selectivity"],
                }
                for index in range(3)
            ]
            if scenario == "low_separation_trace" and copied["variant"] == "recall_half":
                copied["signal_retention"] = 0.82
                copied["positive_signal_retention"] = 0.82
                copied["negative_signal_retention"] = 0.82
                for seed_row in copied["seed_metrics"]:
                    seed_row["signal_retention"] = 0.82
                    seed_row["positive_signal_retention"] = 0.82
                    seed_row["negative_signal_retention"] = 0.82
            rows.append(copied)
    return {
        "scope": "paired multi-seed synthetic recall-trace survival activation benchmark",
        "claim_boundary": (
            "controlled latent delta task; Half is expected to help when strong recall traces are salience-separable "
            "from weak ambiguous traces; low-separation scenarios are boundary checks; not a stateful recall mechanism"
        ),
        "protocol": {
            "comparison_unit": "same sampled recall proposal tensors are replayed across variants",
            "controlled_variable": "activation applied to the Recall delta before residual addition",
            "paired": True,
            "replicates": 3,
        },
        "scenarios": [
            {"name": "separable_trace", "primary_claim": True},
            {"name": "low_separation_trace", "primary_claim": False},
        ],
        "task": {
            "metrics": [
                "state_mse",
                "noise_leakage",
                "weak_trace_survival",
                "signal_retention",
                "positive_signal_retention",
                "negative_signal_retention",
                "selectivity",
                "probe_accuracy",
                "delta_norm",
            ],
            "anti_shrink_controls": {"purpose": "show Half is not merely winning by globally shrinking all recall deltas"},
        },
        "variants": rows,
        "comparisons": [
            {
                "scenario": "separable_trace",
                "comparison": "recall_half_vs_baseline",
                "state_mse_delta": -0.03,
                "noise_leakage_ratio": 0.64,
                "weak_trace_survival_ratio": 0.67,
                "signal_retention_delta": -0.01,
                "negative_signal_retention_delta": -0.01,
                "selectivity_ratio": 1.57,
                "positive_seed_fraction": 1.0,
                "verdict": "primary_test",
            },
            {
                "scenario": "separable_trace",
                "comparison": "recall_half_vs_fixed_shrink",
                "state_mse_delta": -0.04,
                "noise_leakage_ratio": 0.99,
                "weak_trace_survival_ratio": 0.99,
                "signal_retention_delta": 0.315,
                "negative_signal_retention_delta": 0.315,
                "selectivity_ratio": 1.49,
                "positive_seed_fraction": 1.0,
                "verdict": "primary_test",
            },
            {
                "scenario": "separable_trace",
                "comparison": "recall_half_vs_softshrink",
                "state_mse_delta": -0.005,
                "noise_leakage_ratio": 2.8,
                "weak_trace_survival_ratio": 2.9,
                "signal_retention_delta": 0.25,
                "negative_signal_retention_delta": 0.25,
                "selectivity_ratio": 0.48,
                "positive_seed_fraction": 0.0,
                "verdict": "primary_test",
            },
            {
                "scenario": "separable_trace",
                "comparison": "stacked_half_vs_stacked_residual",
                "state_mse_delta": -0.10,
                "noise_leakage_ratio": 0.67,
                "weak_trace_survival_ratio": 0.67,
                "signal_retention_delta": -0.01,
                "negative_signal_retention_delta": -0.01,
                "selectivity_ratio": 1.60,
                "positive_seed_fraction": 1.0,
                "verdict": "primary_test",
            },
            {
                "scenario": "separable_trace",
                "comparison": "stacked_half_vs_stacked_fixed_shrink",
                "state_mse_delta": -0.045,
                "noise_leakage_ratio": 0.99,
                "weak_trace_survival_ratio": 1.0,
                "signal_retention_delta": 0.315,
                "negative_signal_retention_delta": 0.315,
                "selectivity_ratio": 1.49,
                "positive_seed_fraction": 1.0,
                "verdict": "primary_test",
            },
            {
                "scenario": "separable_trace",
                "comparison": "stacked_half_vs_stacked_softshrink",
                "state_mse_delta": 0.045,
                "noise_leakage_ratio": 2.7,
                "weak_trace_survival_ratio": 2.8,
                "signal_retention_delta": 0.25,
                "negative_signal_retention_delta": 0.25,
                "selectivity_ratio": 0.51,
                "positive_seed_fraction": 0.0,
                "verdict": "primary_test",
            },
            {
                "scenario": "low_separation_trace",
                "comparison": "recall_half_vs_baseline",
                "state_mse_delta": 0.01,
                "noise_leakage_ratio": 0.75,
                "weak_trace_survival_ratio": 0.75,
                "signal_retention_delta": -0.18,
                "negative_signal_retention_delta": -0.18,
                "selectivity_ratio": 1.10,
                "positive_seed_fraction": 1.0,
                "verdict": "boundary_only",
            },
        ],
    }


def test_valid_half_recall_trace_survival_passes() -> None:
    assert verify_half_recall_trace_survival.verify(payload()) == []


def test_half_recall_trace_survival_rejects_noise_leakage() -> None:
    data = payload()
    rows = {(row["scenario"], row["variant"]): row for row in data["variants"]}
    rows[("separable_trace", "recall_half")]["noise_leakage"] = 1.3
    failures = verify_half_recall_trace_survival.verify(data)
    assert any("reduce weak recall noise" in failure for failure in failures)


def test_half_recall_trace_survival_rejects_stacked_pollution() -> None:
    data = payload()
    rows = {(row["scenario"], row["variant"]): row for row in data["variants"]}
    rows[("separable_trace", "stacked_half")]["selectivity"] = 0.41
    failures = verify_half_recall_trace_survival.verify(data)
    assert any("improve trace selectivity" in failure for failure in failures)


def test_half_recall_trace_survival_rejects_global_shrink_only_result() -> None:
    data = payload()
    rows = {(row["scenario"], row["variant"]): row for row in data["variants"]}
    rows[("separable_trace", "recall_half")]["signal_retention"] = 0.70
    rows[("separable_trace", "recall_half")]["negative_signal_retention"] = 0.70
    failures = verify_half_recall_trace_survival.verify(data)
    assert any("fixed shrink" in failure for failure in failures)


def test_half_recall_trace_survival_rejects_missing_boundary() -> None:
    data = payload()
    comparisons = {(row["scenario"], row["comparison"]): row for row in data["comparisons"]}
    comparisons[("low_separation_trace", "recall_half_vs_baseline")]["verdict"] = "primary_test"
    failures = verify_half_recall_trace_survival.verify(data)
    assert any("boundary_only" in failure for failure in failures)
