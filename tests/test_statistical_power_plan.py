from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("plan_statistical_power", ROOT / "benchmarks" / "plan_statistical_power.py")
assert SPEC is not None
plan_statistical_power = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(plan_statistical_power)


def test_sign_test_min_n_requires_six_all_positive_seeds() -> None:
    assert plan_statistical_power.sign_test_min_n(0.05) == 6


def test_load_records_filters_target_records(tmp_path: Path) -> None:
    audit = {
        "files": [
            {
                "path": "results.json",
                "records": [
                    {
                        "name": "masked_denoising: ablation drop accuracy - no_mask_accuracy",
                        "n": 3,
                        "mean": 0.2,
                        "std": 0.01,
                        "positive_fraction": 1.0,
                        "sign_test_p_two_sided": 0.25,
                        "values": [0.18, 0.2, 0.22],
                    },
                    {
                        "name": "untracked metric",
                        "n": 3,
                        "mean": 0.0,
                        "std": 0.0,
                        "positive_fraction": 0.0,
                        "sign_test_p_two_sided": 1.0,
                        "values": [0.0, 0.0, 0.0],
                    },
                ],
            }
        ]
    }
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(audit), encoding="utf-8")

    records = plan_statistical_power.load_records(path)

    assert [record["name"] for record in records] == ["masked_denoising: ablation drop accuracy - no_mask_accuracy"]
    assert records[0]["source"] == "results.json"


def test_plan_record_reports_seed_targets() -> None:
    args = argparse.Namespace(
        alpha=0.05,
        target_power=0.8,
        max_n=8,
        outer_samples=10,
        inner_bootstrap_samples=10,
        seed=1,
    )
    record = {
        "name": "masked_denoising: ablation drop accuracy - no_mask_accuracy",
        "source": "results.json",
        "n": 3,
        "mean": 0.2,
        "std": 0.01,
        "positive_fraction": 1.0,
        "sign_test_p_two_sided": 0.25,
        "values": [0.18, 0.2, 0.22],
    }

    plan = plan_statistical_power.plan_record(record, args)

    assert plan["min_n_for_all_positive_sign_test_p_lt_alpha"] == 6
    assert plan["estimated_min_n_for_bootstrap_ci_lower_gt_zero"] is not None
