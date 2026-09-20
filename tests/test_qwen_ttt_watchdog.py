from __future__ import annotations

from pathlib import Path

from benchmarks.run_qwen_ttt_gate1_watchdog import _option_value, build_command


def test_watchdog_owns_training_budget_and_removes_duplicate_option() -> None:
    command = build_command(
        Path("train.py"),
        ["--protocol", "p.json", "--max-seconds", "999", "--response-mode", "none"],
        train_max_seconds=240.0,
    )
    assert command[-2:] == ["--max-seconds", "240.0"]
    assert command.count("--max-seconds") == 1
    assert "--response-mode" in command


def test_watchdog_accepts_separator_before_runner_arguments() -> None:
    command = build_command(
        Path("train.py"),
        ["--", "--max-seconds=999", "--plan-only"],
        train_max_seconds=12.5,
    )
    assert command[1].endswith("train.py")
    assert command[-2:] == ["--max-seconds", "12.5"]
    assert "--max-seconds=999" not in command


def test_watchdog_reads_artifact_directory_for_run_scoped_logs() -> None:
    assert _option_value(["--output-dir", ".tmp/run-a"], "--output-dir") == ".tmp/run-a"
    assert _option_value(["--output-dir=.tmp/run-b"], "--output-dir") == ".tmp/run-b"
    assert _option_value(["--plan-only"], "--output-dir") is None
