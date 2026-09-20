from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"


def load_calibration():
    sys.path.insert(0, str(BENCHMARKS))
    try:
        spec = importlib.util.spec_from_file_location(
            "formula_topology_v4_metric_calibration_test",
            BENCHMARKS / "calibrate_formula_topology_v4_metrics.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(BENCHMARKS))


def load_supervisor():
    sys.path.insert(0, str(BENCHMARKS))
    try:
        spec = importlib.util.spec_from_file_location(
            "formula_topology_v4_metric_supervisor_test",
            BENCHMARKS / "run_formula_topology_v4_metric_calibration.py",
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        sys.path.remove(str(BENCHMARKS))


def observation(
    *,
    dram_write: float,
    l2_read: float,
    l2_write: float,
    dram_read: float = 0.0,
):
    return {
        "metrics": {
            "dram_read": {"sum": dram_read},
            "dram_write": {"sum": dram_write},
            "l2_read": {"sum": l2_read},
            "l2_write": {"sum": l2_write},
        }
    }


CALIBRATION_THRESHOLDS = {
    "baseline_dram_absolute_difference_max": 10.0,
    "baseline_l2_absolute_difference_max": 10.0,
    "baseline_repeat_relative_difference_max": 0.5,
    "dram_absolute_excess_min": 10.0,
    "large_to_small_min": 1.5,
    "l2_absolute_excess_min": 10.0,
    "positive_to_baseline_ratio_min": 2.0,
    "repeat_relative_difference_max": 0.1,
}


def valid_observations(*, dram_read: bool, dram_write: bool):
    baseline = observation(dram_write=0, l2_read=1, l2_write=1, dram_read=0)
    return {
        "sleep-a": baseline,
        "sleep-b": observation(dram_write=0, l2_read=1, l2_write=1, dram_read=0),
        "fill-small": observation(
            dram_write=100 if dram_write else 0,
            l2_read=1,
            l2_write=100,
            dram_read=0,
        ),
        "readwrite-small": observation(
            dram_write=100 if dram_write else 0,
            l2_read=100,
            l2_write=100,
            dram_read=100 if dram_read else 0,
        ),
        "readwrite-large-a": observation(
            dram_write=200 if dram_write else 0,
            l2_read=200,
            l2_write=200,
            dram_read=200 if dram_read else 0,
        ),
        "readwrite-large-b": observation(
            dram_write=202 if dram_write else 0,
            l2_read=202,
            l2_write=202,
            dram_read=202 if dram_read else 0,
        ),
    }


def write_raw_csv(
    path: Path,
    *,
    metrics: dict[str, str],
    kernel_count: int = 4,
    wrong_l2_unit: bool = False,
    blank_metric: tuple[int, str] | None = None,
    process_id: int = 73,
    range_name: str = "ARTI_RANGE",
    metric_override: tuple[int, str, str] | None = None,
) -> None:
    range_column = (
        "thread Domain:Push/Pop_Range:PL_Type:PL_Value:"
        "CLR_Type:Color:Msg_Type:Msg"
    )
    fieldnames = ["ID", "Process ID", "Kernel Name", range_column, *metrics.values()]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                metrics["dram_read"]: "byte",
                metrics["dram_write"]: "byte",
                metrics["l2_read"]: "byte" if wrong_l2_unit else "sector",
                metrics["l2_write"]: "sector",
            }
        )
        for kernel_id in range(kernel_count):
            row = {
                "ID": str(kernel_id),
                "Process ID": str(process_id),
                "Kernel Name": "kernel",
                range_column: (
                    f'13416  "<default domain>:{range_name}:'
                    'none:none:none:none:none:none" '
                ),
                **{name: "1024" for name in metrics.values()},
            }
            if blank_metric is not None and blank_metric[0] == kernel_id:
                row[metrics[blank_metric[1]]] = ""
            if metric_override is not None and metric_override[0] == kernel_id:
                row[metrics[metric_override[1]]] = metric_override[2]
            writer.writerow(row)


def test_l2_fallback_is_explicit_when_dram_write_is_unidentifiable() -> None:
    calibration = load_calibration()
    observations = valid_observations(dram_read=False, dram_write=False)
    decision, primary, errors = calibration.evaluate(observations, CALIBRATION_THRESHOLDS)
    assert decision == "CALIBRATED_L2_FALLBACK"
    assert primary == {"read": "l2_read", "write": "l2_write"}
    assert errors == []


def test_partial_zero_dram_write_and_bad_l2_controls_fail_closed() -> None:
    calibration = load_calibration()
    observations = valid_observations(dram_read=False, dram_write=False)
    observations["readwrite-large-a"]["metrics"]["l2_write"]["sum"] = 110
    observations["readwrite-large-b"]["metrics"]["l2_write"]["sum"] = 110
    decision, primary, errors = calibration.evaluate(observations, CALIBRATION_THRESHOLDS)
    assert decision == "INVALID_METRIC_IDENTIFIABILITY"
    assert primary is None
    assert errors


def test_baseline_combined_tolerance_is_fail_closed() -> None:
    calibration = load_calibration()

    absolute_pass = valid_observations(dram_read=True, dram_write=True)
    absolute_pass["sleep-a"]["metrics"]["dram_read"]["sum"] = 1
    absolute_pass["sleep-b"]["metrics"]["dram_read"]["sum"] = 8
    _, _, errors = calibration.evaluate(absolute_pass, CALIBRATION_THRESHOLDS)
    assert "dram_read sleep baseline was unstable" not in errors

    relative_pass = valid_observations(dram_read=True, dram_write=True)
    relative_pass["sleep-a"]["metrics"]["dram_read"]["sum"] = 100
    relative_pass["sleep-b"]["metrics"]["dram_read"]["sum"] = 140
    relative_pass["readwrite-small"]["metrics"]["dram_read"]["sum"] = 1000
    relative_pass["readwrite-large-a"]["metrics"]["dram_read"]["sum"] = 2000
    relative_pass["readwrite-large-b"]["metrics"]["dram_read"]["sum"] = 2020
    _, _, errors = calibration.evaluate(relative_pass, CALIBRATION_THRESHOLDS)
    assert "dram_read sleep baseline was unstable" not in errors

    both_fail = valid_observations(dram_read=True, dram_write=True)
    both_fail["sleep-a"]["metrics"]["dram_read"]["sum"] = 0
    both_fail["sleep-b"]["metrics"]["dram_read"]["sum"] = 20
    decision, primary, errors = calibration.evaluate(
        both_fail, CALIBRATION_THRESHOLDS
    )
    assert decision == "INVALID_METRIC_IDENTIFIABILITY"
    assert primary is None
    assert "dram_read sleep baseline was unstable" in errors


def test_csv_parser_requires_single_pid_and_all_metric_rows(tmp_path: Path) -> None:
    calibration = load_calibration()
    metrics = {
        "dram_read": "dram-read",
        "dram_write": "dram-write",
        "l2_read": "l2-read",
        "l2_write": "l2-write",
    }
    path = tmp_path / "metrics.csv"
    write_raw_csv(path, metrics=metrics)
    parsed = calibration.parse_csv(
        path,
        process_id=73,
        metrics=metrics,
        expected_range="ARTI_RANGE",
        expected_kernel_count=4,
    )
    assert all(item["sum"] == 4096 for item in parsed["metrics"].values())
    assert len(parsed["kernels"]) == 4


def test_csv_parser_rejects_metric_unit_drift(tmp_path: Path) -> None:
    calibration = load_calibration()
    metrics = {
        "dram_read": "dram-read",
        "dram_write": "dram-write",
        "l2_read": "l2-read",
        "l2_write": "l2-write",
    }
    path = tmp_path / "metrics.csv"
    write_raw_csv(path, metrics=metrics, kernel_count=1, wrong_l2_unit=True)
    try:
        calibration.parse_csv(
            path,
            process_id=73,
            metrics=metrics,
            expected_range="ARTI_RANGE",
            expected_kernel_count=1,
        )
    except RuntimeError as exc:
        assert "unit drift" in str(exc)
    else:
        raise AssertionError("metric unit drift must fail closed")


def test_csv_parser_rejects_incomplete_kernel_metric_product(tmp_path: Path) -> None:
    calibration = load_calibration()
    metrics = {
        "dram_read": "dram-read",
        "dram_write": "dram-write",
        "l2_read": "l2-read",
        "l2_write": "l2-write",
    }
    path = tmp_path / "metrics.csv"
    write_raw_csv(path, metrics=metrics, blank_metric=(3, "l2_write"))
    try:
        calibration.parse_csv(
            path,
            process_id=73,
            metrics=metrics,
            expected_range="ARTI_RANGE",
            expected_kernel_count=4,
        )
    except RuntimeError as exc:
        assert "invalid calibration metric value" in str(exc)
    else:
        raise AssertionError("incomplete kernel x metric coverage must fail closed")


def test_csv_parser_rejects_noncontiguous_kernel_ids(tmp_path: Path) -> None:
    calibration = load_calibration()
    metrics = {
        "dram_read": "dram-read",
        "dram_write": "dram-write",
        "l2_read": "l2-read",
        "l2_write": "l2-write",
    }
    path = tmp_path / "metrics.csv"
    write_raw_csv(path, metrics=metrics)
    text = path.read_text(encoding="utf-8")
    path.write_text(text.replace("3,73,kernel", "7,73,kernel"), encoding="utf-8")
    try:
        calibration.parse_csv(
            path,
            process_id=73,
            metrics=metrics,
            expected_range="ARTI_RANGE",
            expected_kernel_count=4,
        )
    except RuntimeError as exc:
        assert "coverage mismatch" in str(exc)
    else:
        raise AssertionError("noncontiguous kernel IDs must fail closed")


def test_csv_parser_rejects_pid_range_and_nonfinite_drift(tmp_path: Path) -> None:
    calibration = load_calibration()
    metrics = {
        "dram_read": "dram-read",
        "dram_write": "dram-write",
        "l2_read": "l2-read",
        "l2_write": "l2-write",
    }
    cases = [
        ({"process_id": 74}, "coverage mismatch"),
        ({"range_name": "WRONG_RANGE"}, "identity mismatch"),
        ({"metric_override": (0, "dram_read", "NaN")}, "invalid calibration"),
        ({"metric_override": (0, "dram_read", "Inf")}, "invalid calibration"),
        ({"metric_override": (0, "dram_read", "-1")}, "invalid calibration"),
    ]
    for index, (kwargs, message) in enumerate(cases):
        path = tmp_path / f"invalid-{index}.csv"
        write_raw_csv(path, metrics=metrics, **kwargs)
        try:
            calibration.parse_csv(
                path,
                process_id=73,
                metrics=metrics,
                expected_range="ARTI_RANGE",
                expected_kernel_count=4,
            )
        except RuntimeError as exc:
            assert message in str(exc)
        else:
            raise AssertionError(f"invalid raw-view case {index} must fail closed")


def test_csv_parser_rejects_duplicate_required_column(tmp_path: Path) -> None:
    calibration = load_calibration()
    metrics = {
        "dram_read": "dram-read",
        "dram_write": "dram-write",
        "l2_read": "l2-read",
        "l2_write": "l2-write",
    }
    path = tmp_path / "duplicate.csv"
    write_raw_csv(path, metrics=metrics)
    rows = list(csv.reader(path.open(encoding="utf-8")))
    rows[0].append(metrics["dram_read"])
    rows[1].append("byte")
    for row in rows[2:]:
        row.append("1024")
    with path.open("w", encoding="utf-8", newline="") as handle:
        csv.writer(handle).writerows(rows)
    try:
        calibration.parse_csv(
            path,
            process_id=73,
            metrics=metrics,
            expected_range="ARTI_RANGE",
            expected_kernel_count=4,
        )
    except RuntimeError as exc:
        assert "schema mismatch" in str(exc)
    else:
        raise AssertionError("duplicate required metric column must fail closed")


def test_csv_parser_rejects_nonempty_unit_identity_and_extra_rows(tmp_path: Path) -> None:
    calibration = load_calibration()
    metrics = {
        "dram_read": "dram-read",
        "dram_write": "dram-write",
        "l2_read": "l2-read",
        "l2_write": "l2-write",
    }
    for index, mutation in enumerate(("unit", "extra")):
        path = tmp_path / f"row-{index}.csv"
        write_raw_csv(path, metrics=metrics)
        rows = list(csv.reader(path.open(encoding="utf-8")))
        if mutation == "unit":
            rows[1][0] = "99"
            expected = "unit row identity mismatch"
        else:
            extra = [""] * len(rows[0])
            extra[2] = "unexpected"
            rows.append(extra)
            expected = "unexpected nonempty"
        with path.open("w", encoding="utf-8", newline="") as handle:
            csv.writer(handle).writerows(rows)
        try:
            calibration.parse_csv(
                path,
                process_id=73,
                metrics=metrics,
                expected_range="ARTI_RANGE",
                expected_kernel_count=4,
            )
        except RuntimeError as exc:
            assert expected in str(exc)
        else:
            raise AssertionError(f"invalid row case {mutation} must fail closed")


def test_ncu_launcher_resolves_and_binds_real_executable(tmp_path: Path) -> None:
    calibration = load_calibration()
    launcher = tmp_path / "ncu.BAT"
    executable = tmp_path / "target" / "windows-desktop-win7-x64" / "ncu.exe"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"ncu executable")
    launcher.write_text(
        '@echo off\n"%~dp0\\target\\windows-desktop-win7-x64\\ncu.exe" %*\n',
        encoding="utf-8",
    )
    assert calibration.resolve_ncu_executable(launcher) == executable.resolve()
    launcher.write_text("@echo off\necho wrong\n", encoding="utf-8")
    try:
        calibration.resolve_ncu_executable(launcher)
    except RuntimeError as exc:
        assert "unrecognized" in str(exc)
    else:
        raise AssertionError("unknown launcher indirection must fail closed")


def test_target_receipt_is_bound_to_exact_workload() -> None:
    calibration = load_calibration()
    receipt = {
        "checksum": 524800.0,
        "device": "GPU",
        "dtype": "torch.float32",
        "format": "arti.formula-topology-metric-calibration-target.v1",
        "mode": "readwrite",
        "nvtx_range": "ARTI_V4_CALIBRATION_READWRITE",
        "operator": "torch.add(source, 1.0, out=destination)",
        "process_id": 73,
        "repeats": 4,
        "size_mib": 256,
    }
    parsed = calibration.target_receipt(
        json.dumps(receipt),
        expected={"mode": "readwrite", "repeats": 4, "size_mib": 256},
        expected_device="GPU",
    )
    assert parsed == receipt
    receipt["repeats"] = 5
    try:
        calibration.target_receipt(
            json.dumps(receipt),
            expected={"mode": "readwrite", "repeats": 4, "size_mib": 256},
            expected_device="GPU",
        )
    except RuntimeError as exc:
        assert "invalid calibration target receipt" in str(exc)
    else:
        raise AssertionError("workload drift must fail closed")


def test_capture_log_binds_pid_and_report_path(tmp_path: Path) -> None:
    calibration = load_calibration()
    report = tmp_path / "profile.ncu-rep"
    report.write_bytes(b"report")
    capture = tmp_path / "capture.log"
    capture.write_text(
        "==PROF== Connected to process 73 (python.exe)\n"
        "==PROF== Disconnected from process 73\n"
        f"==PROF== Report: {report}\n",
        encoding="utf-8",
    )
    calibration.validate_capture_log(capture, process_id=73, report=report)
    capture.write_text(capture.read_text(encoding="utf-8").replace("73", "74"), encoding="utf-8")
    try:
        calibration.validate_capture_log(capture, process_id=73, report=report)
    except RuntimeError as exc:
        assert "capture lifecycle mismatch" in str(exc)
    else:
        raise AssertionError("capture PID drift must fail closed")


def test_hybrid_metric_selection_requires_identifiable_dram_read() -> None:
    calibration = load_calibration()
    observations = valid_observations(dram_read=True, dram_write=False)
    decision, primary, errors = calibration.evaluate(observations, CALIBRATION_THRESHOLDS)
    assert decision == "CALIBRATED_HYBRID_DRAM_READ_L2_WRITE"
    assert primary == {"read": "dram_read", "write": "l2_write"}
    assert errors == []


def test_nonzero_unidentifiable_dram_read_fails_closed() -> None:
    calibration = load_calibration()
    observations = valid_observations(dram_read=True, dram_write=False)
    observations["readwrite-small"]["metrics"]["dram_read"]["sum"] = 0
    decision, primary, errors = calibration.evaluate(observations, CALIBRATION_THRESHOLDS)
    assert decision == "INVALID_METRIC_IDENTIFIABILITY"
    assert primary is None
    assert errors == [
        "dram_read was active but failed baseline separation, scaling, or repeatability"
    ]


def test_prereg_and_ncu_command_are_frozen(tmp_path: Path) -> None:
    calibration = load_calibration()
    prereg = calibration.load_prereg()
    budgets = prereg["budgets_seconds"]
    assert prereg["experiment_id"].endswith("-005")
    assert (
        budgets["metric_query"]
        + len(prereg["workloads"])
        * (
            budgets["profile_per_workload"]
            + budgets["report_import_per_workload"]
        )
        + budgets["finalization"]
        + budgets["unallocated"]
        == budgets["worker_global"]
    )
    workload = prereg["workloads"][2]
    command = calibration.ncu_command(
        Path("ncu.BAT"), tmp_path, workload, prereg["candidate_metrics"]
    )
    assert calibration.metric_catalog_command(Path("ncu.BAT")) == (
        "ncu.BAT",
        "--query-metrics-mode",
        "all",
    )
    assert "--export" in command
    assert str(tmp_path / "profile") in command
    assert "ARTI_V4_CALIBRATION_FILL/" in command
    assert command[-6:] == (
        "--mode",
        "fill",
        "--size-mib",
        "256",
        "--repeats",
        "4",
    )
    import_command = calibration.report_import_command(
        Path("ncu.BAT"), tmp_path / "profile.ncu-rep", tmp_path / "derived.csv"
    )
    assert import_command == (
        "ncu.BAT",
        "--import",
        str(tmp_path / "profile.ncu-rep"),
        "--csv",
        "--page",
        "raw",
        "--print-units",
        "base",
        "--log-file",
        str(tmp_path / "derived.csv"),
    )


def test_supervisor_requires_hash_bound_non_authoritative_worker(
    tmp_path: Path,
) -> None:
    supervisor = load_supervisor()
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    manifest = {
        "format": "arti.formula-topology-metric-calibration.v1",
        "classification": "CALIBRATED_L2_FALLBACK",
        "scientific_score": False,
        "formal_seeds_consumed": False,
        "formal_authorization": False,
        "reusable_as_preflight": False,
    }
    manifest_path = artifact / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    for name in supervisor.worker_success_allowlist() - {"manifest.json"}:
        path = artifact / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture")
    hashes = {
        name: supervisor.sha256_file(artifact / name)
        for name in supervisor.worker_success_allowlist()
    }
    (artifact / "SHA256SUMS.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())),
        encoding="ascii",
    )
    assert supervisor.load_worker_artifact(
        artifact, deadline=float("inf")
    ) == manifest
    manifest["classification"] = "INVALID_CALIBRATION_EVIDENCE"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    hashes["manifest.json"] = supervisor.sha256_file(manifest_path)
    (artifact / "SHA256SUMS.txt").write_text(
        "".join(f"{digest}  {name}\n" for name, digest in sorted(hashes.items())),
        encoding="ascii",
    )
    try:
        supervisor.load_worker_artifact(artifact, deadline=float("inf"))
    except RuntimeError as exc:
        assert "authority/schema" in str(exc)
    else:
        raise AssertionError("invalid worker classification must fail closed")
    manifest["classification"] = "CALIBRATED_L2_FALLBACK"
    manifest["formal_authorization"] = True
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    try:
        supervisor.load_worker_artifact(artifact, deadline=float("inf"))
    except RuntimeError as exc:
        assert "authority/schema" in str(exc)
    else:
        raise AssertionError("worker authority drift must fail closed")
