import argparse
import importlib.util
import json
from pathlib import Path

import pytest


def load_collector():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "collect_evidence_summary.py"
    spec = importlib.util.spec_from_file_location("collect_evidence_summary", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


collector = load_collector()


def write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def minimal_args(tmp_path):
    return argparse.Namespace(
        nature=tmp_path / "nature.json",
        scaling=tmp_path / "scaling.json",
        proxy=tmp_path / "proxy.json",
        visibility=tmp_path / "visibility.json",
        tabular=tmp_path / "tabular.json",
        tabular_lock=tmp_path / "tabular_lock.json",
        public_tabular=tmp_path / "public_tabular.json",
        public_tabular_lock=tmp_path / "public_tabular_lock.json",
        public_openml_phishing=tmp_path / "public_openml_phishing.json",
        public_openml_phishing_lock=tmp_path / "public_openml_phishing_lock.json",
        downstream_results=tmp_path / "downstream_results.json",
        hyperparameter_trial_plan=tmp_path / "hyperparameter_trial_plan.json",
        hyperparameter_trial_smoke=tmp_path / "hyperparameter_trial_smoke.json",
        high_power_nature=tmp_path / "high_power_nature.json",
        high_power_proxy=tmp_path / "high_power_proxy.json",
        high_power_visibility=tmp_path / "high_power_visibility.json",
        high_power_statistics=tmp_path / "high_power_statistics.json",
        bundle_lock=tmp_path / "bundle_lock.json",
    )


def write_minimal_inputs(tmp_path, omit_private_margin=False):
    tasks = ("coordinate_routing", "visibility_reasoning", "masked_denoising", "recall_operator_routing")
    nature = {
        "summary": [{"task": task, "model": "arti_full", "mean_accuracy": 0.9, "std_accuracy": 0.01} for task in tasks],
        "claim_verdicts": [
            {
                "task": task,
                "claim_verdict": "supported",
                "best_model": "arti_full",
                "arti_full_margin_vs_transformer": 0.1,
                "arti_full_margin_vs_matched_transformer": 0.2,
                "arti_full_margin_vs_mlp": 0.3,
                "arti_full_margin_vs_matched_mlp": 0.4,
                "arti_full_margin_vs_no_recall": 0.5,
            }
            for task in tasks
        ],
    }
    scaling = {
        "rows": [
            {"model": "arti_interface_only", "tokens": 16, "mean_ms": 1.0, "estimated_activation_bytes": 10, "cuda_peak_allocated_bytes": None},
            {"model": "arti_interface_only", "tokens": 32, "mean_ms": 2.0, "estimated_activation_bytes": 20, "cuda_peak_allocated_bytes": None},
        ],
        "slopes": {"arti_interface_only": 0.5},
    }
    proxy_margins = [] if omit_private_margin else [{"comparison": "arti_full - arti_no_recall", "mean_margin": 0.1, "positive_fraction": 1.0}]
    proxy = {
        "summary": [{"model": "arti_full"}],
        "paired_margins": proxy_margins + [{"comparison": "arti_full - transformer_recall", "mean_margin": 0.0, "positive_fraction": 0.0}],
    }
    visibility = {
        "summary": [{"model": "arti_full"}],
        "paired_margins": [
            {"comparison": "arti_full - arti_no_visibility", "mean_margin": 0.2, "positive_fraction": 1.0},
            {"comparison": "arti_full - transformer_query", "mean_margin": 0.0, "positive_fraction": 0.0},
        ],
    }
    tabular = {"summary": [{"model": "arti", "mean_accuracy": 0.7, "std_accuracy": 0.1}]}
    write_json(tmp_path / "nature.json", nature)
    write_json(tmp_path / "scaling.json", scaling)
    write_json(tmp_path / "proxy.json", proxy)
    write_json(tmp_path / "visibility.json", visibility)
    write_json(tmp_path / "tabular.json", tabular)
    write_json(tmp_path / "tabular_lock.json", {"sha256": "abc"})


def test_collect_uses_max_token_scaling_row(tmp_path):
    write_minimal_inputs(tmp_path)

    payload = collector.collect(minimal_args(tmp_path))

    assert payload["scaling"]["max_tokens"] == 32
    assert payload["scaling"]["at_max_tokens"]["arti_interface_only"]["mean_ms"] == 2.0
    assert payload["mechanism_gates"]["coordinate_routing"]["margins"]["vs_matched_transformer"] == 0.2
    assert payload["high_power"]["available"] is False


def test_collect_fails_when_required_proxy_margin_missing(tmp_path):
    write_minimal_inputs(tmp_path, omit_private_margin=True)

    with pytest.raises(KeyError):
        collector.collect(minimal_args(tmp_path))


def test_collect_includes_public_tabular_when_artifacts_exist(tmp_path):
    write_minimal_inputs(tmp_path)
    write_json(
        tmp_path / "public_tabular.json",
        {"summary": [{"model": "arti", "mean_accuracy": 0.88, "std_accuracy": 0.02}]},
    )
    write_json(
        tmp_path / "public_tabular_lock.json",
        {"adapter_status": "external_public_locked", "target": "diagnosis"},
    )

    payload = collector.collect(minimal_args(tmp_path))

    assert payload["public_tabular_adapter"]["available"] is True
    assert payload["public_tabular_adapter"]["benchmark_count"] == 1
    assert payload["public_tabular_adapter"]["dataset_lock"]["adapter_status"] == "external_public_locked"


def test_collect_counts_two_public_tabular_benchmarks(tmp_path):
    write_minimal_inputs(tmp_path)
    for stem in ("public_tabular", "public_openml_phishing"):
        write_json(
            tmp_path / f"{stem}.json",
            {"summary": [{"model": "arti", "mean_accuracy": 0.88, "std_accuracy": 0.02}]},
        )
        write_json(
            tmp_path / f"{stem}_lock.json",
            {"adapter_status": "external_public_locked", "target": "diagnosis", "rows": 10},
        )

    payload = collector.collect(minimal_args(tmp_path))

    assert payload["public_tabular_adapter"]["benchmark_count"] == 2


def test_collect_includes_hyperparameter_trial_plan(tmp_path):
    write_minimal_inputs(tmp_path)
    write_json(
        tmp_path / "hyperparameter_trial_plan.json",
        {"status": "planned_not_executed", "trial_count": 48, "selection_metric": "validation_accuracy"},
    )

    payload = collector.collect(minimal_args(tmp_path))

    assert payload["hyperparameter_trial_plan"]["available"] is True
    assert payload["hyperparameter_trial_plan"]["trial_count"] == 48


def test_collect_includes_hyperparameter_trial_smoke(tmp_path):
    write_minimal_inputs(tmp_path)
    write_json(
        tmp_path / "hyperparameter_trial_plan.json",
        {"status": "planned_not_executed", "trial_count": 48, "selection_metric": "validation_accuracy"},
    )
    write_json(
        tmp_path / "hyperparameter_trial_smoke.json",
        {"status": "partial_smoke", "trial_results": [{"trial_id": "a"}, {"trial_id": "b"}]},
    )

    payload = collector.collect(minimal_args(tmp_path))

    assert payload["hyperparameter_trial_plan"]["smoke_execution_status"] == "partial_smoke"
    assert payload["hyperparameter_trial_plan"]["smoke_trial_count"] == 2


def test_collect_includes_high_power_when_artifacts_exist(tmp_path):
    write_minimal_inputs(tmp_path)
    tasks = ("coordinate_routing", "visibility_reasoning", "masked_denoising", "recall_operator_routing")
    write_json(
        tmp_path / "high_power_nature.json",
        {
            "summary": [{"task": task, "model": "arti_full", "mean_accuracy": 0.95} for task in tasks],
            "claim_verdicts": [{"task": task, "claim_verdict": "supported"} for task in tasks],
        },
    )
    write_json(
        tmp_path / "high_power_proxy.json",
        {"paired_margins": [{"comparison": "arti_full - arti_no_recall", "mean_margin": 0.05, "positive_fraction": 1.0}]},
    )
    write_json(
        tmp_path / "high_power_visibility.json",
        {"paired_margins": [{"comparison": "arti_full - arti_no_visibility", "mean_margin": 0.5, "positive_fraction": 1.0}]},
    )
    write_json(
        tmp_path / "high_power_statistics.json",
        {
            "files": [
                {
                    "records": [
                        {
                            "name": name,
                            "n": 6,
                            "mean": 0.1,
                            "positive_fraction": 1.0,
                            "sign_test_p_two_sided": 0.03125,
                            "bootstrap_ci95": [0.05, 0.15],
                        }
                        for name in (
                            "length_shift_private_rule: paired margin arti_full - arti_no_recall",
                            "length_shift_visibility_readout: paired margin arti_full - arti_no_visibility",
                            "recall_operator_routing: ablation drop accuracy - zero_recall_accuracy",
                        )
                    ]
                }
            ]
        },
    )
    write_json(tmp_path / "bundle_lock.json", {"artifacts": [{"path": "a"}, {"path": "b"}]})

    payload = collector.collect(minimal_args(tmp_path))

    assert payload["high_power"]["available"] is True
    assert payload["high_power"]["proxy"]["private_rule_arti_full_vs_no_recall"]["mean_margin"] == 0.05
    assert payload["high_power"]["bundle_lock_artifact_count"] == 2
