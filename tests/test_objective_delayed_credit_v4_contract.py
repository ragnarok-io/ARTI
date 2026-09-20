from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(name, None)
    return module


def test_v4_preregistration_matches_runner_contract() -> None:
    prereg = json.loads(
        (BENCHMARKS / "objective_delayed_credit_v4_prereg.json").read_text(
            encoding="utf-8"
        )
    )
    runner = load_module(
        "objective_v4_runner_test",
        BENCHMARKS / "train_objective_delayed_credit_v4.py",
    )
    preflight = load_module(
        "objective_v4_preflight_test",
        BENCHMARKS / "objective_delayed_credit_v4_preflight.py",
    )
    assert preflight.validate_source_contract(prereg, runner.EXPERIMENT_CONTRACT) == []


def test_v4_objective_and_matched_gate_have_equal_parameter_count() -> None:
    runner = load_module(
        "objective_v4_runner_parameters_test",
        BENCHMARKS / "train_objective_delayed_credit_v4.py",
    )
    objective = runner.ObjectiveExposureBank(
        runner.OBJECTIVE_SLOTS,
        runner.QUERY_DIM,
        key_seed=7,
    )
    matched = runner.MatchedNonlinearGate()
    assert sum(parameter.numel() for parameter in objective.parameters()) == 16
    assert sum(parameter.numel() for parameter in matched.parameters()) == 16


def test_v4_required_controls_are_unique_and_formula_is_disabled() -> None:
    runner = load_module(
        "objective_v4_runner_controls_test",
        BENCHMARKS / "train_objective_delayed_credit_v4.py",
    )
    contract = runner.EXPERIMENT_CONTRACT
    assert len(contract["required_arms"]) == len(set(contract["required_arms"]))
    assert contract["formula_fabric_enabled"] is False
    assert contract["future_visibility"] == "frozen_scorer_only"
    assert contract["true_reset"] == "replace_and_score_candidate"


def test_v4_preflight_binds_every_formal_executable() -> None:
    preflight = load_module(
        "objective_v4_preflight_sources_test",
        BENCHMARKS / "objective_delayed_credit_v4_preflight.py",
    )
    assert {path.name for path in preflight.SOURCE_PATHS} == {
        "objective_delayed_credit_v4_prereg.json",
        "train_objective_delayed_credit_v4.py",
        "verify_objective_delayed_credit_v4.py",
        "objective_delayed_credit_v4_preflight.py",
        "objective_delayed_credit_v4_cost_target.py",
        "profile_objective_delayed_credit_v4_cost.py",
        "run_objective_delayed_credit_v4_gate.py",
    }


def test_v4_cost_gate_fails_closed_on_missing_or_excess_ratio() -> None:
    prereg = json.loads(
        (BENCHMARKS / "objective_delayed_credit_v4_prereg.json").read_text(
            encoding="utf-8"
        )
    )
    gate = load_module(
        "objective_v4_gate_cost_test",
        BENCHMARKS / "run_objective_delayed_credit_v4_gate.py",
    )
    required = (
        prereg["cost_gate"]["required_physical_counters"]
        + prereg["cost_gate"]["required_attributed_counts"]
    )
    passing = {name: 1.0 for name in required}
    assert gate.validate_cost_ratios(
        prereg, {"objective_to_matched_ratios": passing}
    ) == []
    missing = dict(passing)
    missing.pop(required[0])
    assert "missing pre-run cost ratio" in gate.validate_cost_ratios(
        prereg, {"objective_to_matched_ratios": missing}
    )[0]
    excess = dict(passing)
    excess[required[-1]] = 1.051
    assert "exceeded" in gate.validate_cost_ratios(
        prereg, {"objective_to_matched_ratios": excess}
    )[0]


def test_v4_timeout_output_hashing_accepts_text_and_bytes() -> None:
    gate = load_module(
        "objective_v4_gate_output_test",
        BENCHMARKS / "run_objective_delayed_credit_v4_gate.py",
    )
    assert gate.output_bytes(None) == b""
    assert gate.output_bytes("receipt") == b"receipt"
    assert gate.output_bytes(b"receipt") == b"receipt"


def test_v4_hardware_manifest_is_rebuilt_from_raw_receipts(tmp_path: Path) -> None:
    prereg = json.loads(
        (BENCHMARKS / "objective_delayed_credit_v4_prereg.json").read_text(
            encoding="utf-8"
        )
    )
    gate = load_module(
        "objective_v4_gate_raw_cost_test",
        BENCHMARKS / "run_objective_delayed_credit_v4_gate.py",
    )
    collector = load_module(
        "objective_v4_cost_collector_test",
        BENCHMARKS / "profile_objective_delayed_credit_v4_cost.py",
    )
    raw_receipts: dict[str, str] = {}
    for arm in collector.ARMS:
        for phase in collector.PHASES:
            attributed = tmp_path / f"{arm}-{phase}-attributed.json"
            attributed.write_text(
                json.dumps(
                    {
                        "format": "arti.objective-delayed-credit-attributed-cost.v4",
                        "arm": arm,
                        "phase": phase,
                        "batch_size": prereg["training"]["batch_size"],
                        "attributed_flops": 100.0,
                        "optimizer_parameter_elements": 16.0,
                    }
                ),
                encoding="utf-8",
            )
            raw_receipts[attributed.name] = gate.sha256_file(attributed)
        ncu = tmp_path / f"{arm}-train-step-ncu.csv"
        ncu.write_text(
            '"ID","Process ID","Kernel Name","Metric Name","Metric Value"\n'
            '"1","1","kernel","dram__bytes_read.sum","100"\n'
            '"1","1","kernel","dram__bytes_write.sum","50"\n',
            encoding="utf-8",
        )
        raw_receipts[ncu.name] = gate.sha256_file(ncu)
    measurements = collector.reconstruct_raw_measurements(
        tmp_path,
        raw_receipts,
        batch_size=prereg["training"]["batch_size"],
    )
    totals, ratios = collector.aggregate_measurements(measurements)
    hardware = {
        "format": "arti.objective-delayed-credit-hardware-cost.v4",
        "physical_counter_status": "AVAILABLE",
        "finalized": False,
        "artifact_sha256": {},
        "batch_size": prereg["training"]["batch_size"],
        "profiling_wall_seconds": 1.0,
        "raw_receipts": raw_receipts,
        "measurements": measurements,
        "totals": totals,
        "objective_to_matched_ratios": ratios,
    }
    manifest = tmp_path / "hardware-cost-base.json"
    assert gate.validate_hardware_cost_base(prereg, hardware, manifest) == []
    hardware["objective_to_matched_ratios"]["forward_flops"] = 0.5
    assert "differ from raw receipts" in gate.validate_hardware_cost_base(
        prereg, hardware, manifest
    )[0]


def test_v4_ncu_parser_requires_both_dram_metrics(tmp_path: Path) -> None:
    collector = load_module(
        "objective_v4_cost_parser_test",
        BENCHMARKS / "profile_objective_delayed_credit_v4_cost.py",
    )
    receipt = tmp_path / "incomplete.csv"
    receipt.write_text(
        '"ID","Process ID","Kernel Name","Metric Name","Metric Value"\n'
        '"1","1","kernel","dram__bytes_read.sum","100"\n',
        encoding="utf-8",
    )
    try:
        collector.parse_ncu_csv(receipt)
    except RuntimeError as exc:
        assert "missing required DRAM metrics" in str(exc)
    else:
        raise AssertionError("incomplete Nsight receipt was accepted")


def test_v4_ncu_parser_supports_blackwell_metric_names(tmp_path: Path) -> None:
    collector = load_module(
        "objective_v4_cost_parser_blackwell_test",
        BENCHMARKS / "profile_objective_delayed_credit_v4_cost.py",
    )
    receipt = tmp_path / "blackwell.csv"
    receipt.write_text(
        '==PROF== Connected\n'
        '"ID","Process ID","Kernel Name","Metric Name","Metric Value"\n'
        '"0","1","summary","dram__bytes_op_read.sum","n/a"\n'
        '"1","1","kernel","dram__bytes_op_read.sum","1,024"\n'
        '"1","1","kernel","dram__bytes_op_write.sum","256"\n',
        encoding="utf-8",
    )
    parsed = collector.parse_ncu_csv(receipt)
    assert parsed["dram_bytes_read"] == 1024.0
    assert parsed["dram_bytes_written"] == 256.0
    assert parsed["kernel_launches"] == 1.0
    assert parsed["metric_names"]["dram_bytes_read"] == "dram__bytes_op_read.sum"


def test_v4_ncu_parser_rejects_n_a_only_receipt(tmp_path: Path) -> None:
    collector = load_module(
        "objective_v4_cost_parser_na_test",
        BENCHMARKS / "profile_objective_delayed_credit_v4_cost.py",
    )
    receipt = tmp_path / "na-only.csv"
    receipt.write_text(
        '"ID","Process ID","Kernel Name","Metric Name","Metric Value"\n'
        '"1","1","kernel","dram__bytes_op_read.sum","n/a"\n'
        '"1","1","kernel","dram__bytes_op_write.sum","n/a"\n',
        encoding="utf-8",
    )
    try:
        collector.parse_ncu_csv(receipt)
    except RuntimeError as exc:
        assert "missing required DRAM metrics" in str(exc)
    else:
        raise AssertionError("n/a-only Nsight receipt was accepted")


def test_v4_cost_ratio_treats_zero_over_zero_as_equal_cost() -> None:
    collector = load_module(
        "objective_v4_cost_ratio_zero_test",
        BENCHMARKS / "profile_objective_delayed_credit_v4_cost.py",
    )
    assert collector.ratio(0.0, 0.0) == 1.0
    assert collector.ratio(1.0, 0.0) == float("inf")
