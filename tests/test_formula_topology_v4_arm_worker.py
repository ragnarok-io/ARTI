from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"


def load_worker():
    if str(BENCHMARKS) not in sys.path:
        sys.path.insert(0, str(BENCHMARKS))
    spec = importlib.util.spec_from_file_location(
        "formula_topology_v4_arm_worker_test",
        BENCHMARKS / "characterize_formula_topology_v4_arm.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


METRICS = {
    "dram_read": "dram__bytes_op_read.sum",
    "dram_write": "dram__bytes_op_write.sum",
    "l2_read": "lts__t_sectors_op_read.sum",
    "l2_write": "lts__t_sectors_op_write.sum",
}


def report_text(*, pid: int = 73, duplicate: bool = False, unit: str = "byte") -> str:
    header = (
        'Process ID,ID,Kernel Name,Push/Pop_Range:Name,'
        + ",".join(METRICS.values())
        + "\n"
    )
    units = f",,,,{unit},byte,sector,sector\n"
    rows = []
    for index, phase in enumerate(
        ("FORWARD", "BACKWARD", "GRADIENT_POSTPROCESS", "OPTIMIZER")
    ):
        rows.append(
            f'{pid},{index},kernel_{index},"1 ""<default domain>:'
            f'ARTI_FORMULA_TOPOLOGY_V3_COST:{phase}:none:none:none:none:none""",'
            "2.5,5,7.5,10\n"
        )
    return header + units + "".join(rows) + (rows[0] if duplicate else "")


def test_report_parser_reconstructs_unique_kernel_and_metrics(tmp_path: Path) -> None:
    worker = load_worker()
    path = tmp_path / "report-derived.csv"
    path.write_text(report_text(), encoding="utf-8")
    launches, totals = worker._validate_report_csv(path, process_id=73, metrics=METRICS)
    assert launches == 4
    assert totals == {
        "dram_read": 10.0,
        "dram_write": 20.0,
        "l2_read": 30.0,
        "l2_write": 40.0,
    }


@pytest.mark.parametrize(
    ("content", "process_id", "message"),
    [
        (report_text(pid=74), 73, "PID"),
        (report_text(duplicate=True), 73, "duplicate"),
        (report_text(unit="sector"), 73, "unit"),
    ],
)
def test_report_parser_fails_closed(
    tmp_path: Path, content: str, process_id: int, message: str
) -> None:
    worker = load_worker()
    path = tmp_path / "report-derived.csv"
    path.write_text(content, encoding="utf-8")
    with pytest.raises(RuntimeError, match=message):
        worker._validate_report_csv(path, process_id=process_id, metrics=METRICS)


def test_ncu_command_is_one_kernel_replay_without_target_processes(tmp_path: Path) -> None:
    worker = load_worker()
    command = worker._ncu_command(
        Path("ncu.exe"),
        tmp_path,
        ("python.exe", "target.py"),
        list(METRICS.values()),
    )
    assert command.count("--replay-mode") == 1
    assert command[command.index("--replay-mode") + 1] == "kernel"
    assert command.count("--nvtx-include") == 1
    assert command[command.index("--nvtx-include") + 1] == (
        "ARTI_FORMULA_TOPOLOGY_V3_COST/"
    )
    assert "--target-processes" not in command
    assert command[command.index("--metrics") + 1] == ",".join(METRICS.values())


def test_hash_manifest_requires_every_declared_file(tmp_path: Path) -> None:
    worker = load_worker()
    (tmp_path / "a.txt").write_text("a", encoding="ascii")
    with pytest.raises(RuntimeError, match="missing"):
        worker._write_hash_manifest(
            tmp_path,
            ["a.txt", "missing.txt"],
            deadline=float("inf"),
        )
