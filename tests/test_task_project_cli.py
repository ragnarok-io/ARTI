from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import arti
from arti.cli import main


def test_cli_creates_moves_and_checks_without_importing_application(tmp_path, capsys, monkeypatch):
    source = tmp_path / "source"
    assert main(["task", "init", str(source), "--name", "generated-project"]) == 0
    initialized = json.loads(capsys.readouterr().out)
    assert initialized["project"] == str(source / "project.json")
    assert not initialized["application_executed"] and not initialized["trained"]
    moved = tmp_path / "moved"
    shutil.move(source, moved)
    monkeypatch.chdir(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("inspection must not resolve application code")

    monkeypatch.setattr(arti.PythonFactoryConfig, "resolve", forbidden)
    report = tmp_path / "check.json"
    assert main(["task", "check", str(moved / "project.json"), "--output", str(report)]) == 0
    checked = json.loads(capsys.readouterr().out)
    assert checked == json.loads(report.read_text(encoding="utf-8"))
    assert checked["ok"] and not checked["execution"]["user_code_imported"]
    assert main(["task", "init", str(moved)]) == 1
    assert "fresh" in capsys.readouterr().err


def test_cli_explains_fresh_task_and_emits_same_structured_report(tmp_path, capsys):
    project = Path(arti.create_task_project(tmp_path / "project")["project"])
    assert main(["task", "status", str(project)]) == 0
    status = json.loads(capsys.readouterr().out)
    assert not status["summary"]["lifecycle_complete"]
    assert status["artifacts"]["training_run"]["status"] == "not_created"
    output = tmp_path / "explanation.json"
    assert main(["task", "explain", str(project), "--output", str(output)]) == 0
    text = capsys.readouterr().out
    assert "Task project:" in text and "incomplete" in text
    assert json.loads(output.read_text(encoding="utf-8"))["next_actions"]
    assert main(["task", "explain", str(project), "--format", "json"]) == 0
    assert not json.loads(capsys.readouterr().out)["executes_actions"]


def test_cli_returns_field_located_missing_input_error(tmp_path, capsys):
    project = Path(arti.create_task_project(tmp_path / "project")["project"])
    value = json.loads(project.read_text(encoding="utf-8"))
    value["inputs"]["requests"] = "data/absent.jsonl"
    project.write_text(json.dumps(value), encoding="utf-8")
    assert main(["task", "check", str(project)]) == 1
    failed = json.loads(capsys.readouterr().out)
    assert not failed["ok"]
    assert any(issue["field"] == "project.inputs.requests" for issue in failed["issues"])
    assert main(["task", "check", str(project), "--format", "text"]) == 1
    assert "project.inputs.requests" in capsys.readouterr().out


def test_cli_generates_checks_and_rejects_stale_workflow_schema(tmp_path, capsys):
    path = tmp_path / "recipe.schema.json"
    args = ["schema", "task-workflow", "generate", "--kind", "recipe", "--output", str(path)]
    assert main(args) == 0
    assert json.loads(capsys.readouterr().out)["schema"] == "recipe"
    assert json.loads(path.read_text(encoding="utf-8")) == arti.project_schema("recipe")
    args[2] = "check"
    assert main(args) == 0
    capsys.readouterr()
    path.write_text("{}", encoding="utf-8")
    assert main(args) == 1
    assert "stale" in capsys.readouterr().err


def test_new_public_api_keeps_backend_aliases():
    if os.environ.get("ARTI_WHEEL_ROOT"):
        assert Path(arti.__file__).resolve().parent == (Path(os.environ["ARTI_WHEEL_ROOT"]) / "arti").resolve()
    for name in ("TaskProjectConfig", "read_task_project", "check_task_project",
                 "project_schema", "packaged_project_schema", "create_task_project",
                 "task_project_status", "explain_task_project", "render_task_project_status"):
        assert getattr(arti, name) is getattr(arti.torch, name)


def test_cli_report_cannot_replace_project_or_prediction_authority(tmp_path, capsys):
    project = Path(arti.create_task_project(tmp_path / "project")["project"])
    before = project.read_bytes()
    assert main(["task", "status", str(project), "--output", str(project)]) == 1
    assert "outside" in capsys.readouterr().err
    assert project.read_bytes() == before
    prediction = project.parent / "artifacts/prediction/run.json"
    assert main(["task", "explain", str(project), "--output", str(prediction)]) == 1
    assert "outside" in capsys.readouterr().err
    assert not prediction.exists()
    future_parent = project.parent / "runs"
    assert main(["task", "check", str(project), "--output", str(future_parent)]) == 1
    assert "outside" in capsys.readouterr().err
    assert not future_parent.exists()
