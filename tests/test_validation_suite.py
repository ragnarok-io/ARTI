import importlib.util
from pathlib import Path


def load_runner():
    path = Path(__file__).resolve().parents[1] / "benchmarks" / "run_validation_suite.py"
    spec = importlib.util.spec_from_file_location("run_validation_suite", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


runner = load_runner()


def test_run_suite_fail_fast_stops_after_first_failure(monkeypatch):
    calls = []

    def fake_run(command):
        calls.append(command)
        return {"command": command, "returncode": 1 if command == "bad" else 0, "seconds": 0.0, "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr(runner, "run_command", fake_run)

    runs = runner.run_suite(["ok", "bad", "later"], fail_fast=True)

    assert calls == ["ok", "bad"]
    assert [row["returncode"] for row in runs] == [0, 1]


def test_run_suite_without_fail_fast_runs_all(monkeypatch):
    calls = []

    def fake_run(command):
        calls.append(command)
        return {"command": command, "returncode": 1 if command == "bad" else 0, "seconds": 0.0, "stdout_tail": "", "stderr_tail": ""}

    monkeypatch.setattr(runner, "run_command", fake_run)

    runs = runner.run_suite(["ok", "bad", "later"], fail_fast=False)

    assert calls == ["ok", "bad", "later"]
    assert [row["command"] for row in runs] == ["ok", "bad", "later"]
