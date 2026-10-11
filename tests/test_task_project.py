from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys

import pytest

from arti.python_factory import PythonFactoryConfig
from arti.recipe_workflow import write_json, write_jsonl
from arti.task_project import (
    ARTIFACT_ROLES,
    SCHEMA_KINDS,
    TaskProjectConfig,
    check_task_project,
    check_task_project_schema,
    project_schema,
    read_task_project,
    write_task_project_schema,
)


def project_fixture(directory):
    directory.mkdir(parents=True)
    configs = directory / "configs"
    configs.mkdir()
    data = directory / "data"
    data.mkdir()
    write_jsonl(data / "records.jsonl", [{"id": "row-a", "split": "train", "values": [0.0, 1.0]}, {"id": "row-b", "split": "validation", "values": [1.0, 0.0]}])
    (data / "input.bin").write_bytes(bytes(range(16)))
    (directory / "untouched_caller.py").write_text("raise AssertionError('caller must not be imported by project checks')\n", encoding="utf-8")
    values = {
        "recipe": {"task": "generated-native-task", "factory": "untouched_caller:make_task", "options": {"application_path": "keep/me", "native_shape": [2, 3]}, "resources": {"caller_value": {"application": "owned"}}, "training": {"device": "cpu", "steps": 3}, "run": {"output_dir": "../runs/train"}},
        "snapshot": {"records": "../data/records.jsonl", "assets": {"input": "../data/input.bin", "reference": None}, "bindings": {"row-a": [{"asset_id": "input"}, {"asset_id": "reference", "role": "reference", "required": False}]}, "source_revision": "generated-v1"},
        "transform": {"factory": "untouched_caller:transform", "revision": "generated-v1", "options": {"native_dtype": "float32"}},
        "prediction": {"format": "arti.prediction-run-config", "output_dir": "../runs/prediction", "batch_size": 2, "id_field": "id"},
        "files": {"format": "arti.prediction-files-config", "field": "files", "validator": "untouched_caller:validate_files", "revision": "generated-v1"},
        "suite": {"views": [{"name": "native-reference", "scoring": {"scorer": "untouched_caller:score", "scorer_id": "generated-v1"}}]},
        "runtime": {"entrypoint": "untouched_caller:run", "revision": "generated-v1", "inputs": {"records": "file", "dataset": "directory"}, "outputs": {"prediction": "directory"}, "network": "disabled", "limits": {"max_seconds": 12}, "backend": {"factory": "untouched_caller:host", "revision": "generated-host-v1"}},
        "export": {"writer": "untouched_caller:write", "revision": "generated-v1", "archive": "zip", "instructions": "Read the local result receipts."},
    }
    for key, value in values.items():
        write_json(configs / f"{key}.json", value)
    declaration = {
        "format": "arti.task-project", "format_version": 1, "name": "neutral-project",
        "configs": {key: f"configs/{key}.json" for key in values},
        "resources": {"source": "data/input.bin"}, "inputs": {"records": "data/records.jsonl"},
        "artifacts": {key: f"runs/{key}" for key in ARTIFACT_ROLES},
    }
    declaration["artifacts"].update(training_run="runs/train", prediction_run="runs/prediction")
    target = directory / "project.json"
    write_json(target, declaration)
    return target, values


def issues(report, code):
    return [item for item in report["issues"] if item["code"] == code]


def test_full_static_project_keeps_future_artifacts_and_never_imports_code(tmp_path, monkeypatch):
    path, _ = project_fixture(tmp_path / "project")
    monkeypatch.syspath_prepend(str(path.parent))
    monkeypatch.setattr(PythonFactoryConfig, "resolve", lambda *_: pytest.fail("static checker resolved caller code"))
    before = {target: target.read_bytes() for target in path.parent.rglob("*") if target.is_file()}
    report = check_task_project(path)
    assert report["ok"], report["issues"]
    assert report["execution"] == {"user_code_imported": False, "model_constructed": False, "host_probed": False}
    assert "untouched_caller" not in sys.modules
    assert all(item["status"] == "valid" for item in report["configs"].values())
    assert all(item["status"] == "not_created" for item in report["artifacts"].values())
    assert before == {target: target.read_bytes() for target in path.parent.rglob("*") if target.is_file()}
    assert not (path.parent / "runs").exists()


def test_project_relocation_resolves_config_paths_without_chdir_or_option_rewriting(tmp_path, monkeypatch):
    path, values = project_fixture(tmp_path / "original")
    moved = tmp_path / "moved"
    shutil.move(str(path.parent), moved)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    loaded = read_task_project(moved / "project.json")
    recipe = loaded.resolved_recipe()
    assert recipe.options == values["recipe"]["options"]
    assert recipe.resources["caller_value"] == values["recipe"]["resources"]["caller_value"]
    assert recipe.resources["source"] == str((moved / "data/input.bin").resolve())
    assert Path(recipe.run.output_dir) == (moved / "runs/train").resolve()
    assert loaded.project_root == moved.resolve()
    assert check_task_project(moved / "project.json")["ok"]
    assert Path.cwd() == elsewhere
    assert loaded.to_dict()["resources"]["source"] == "data/input.bin"
    with pytest.raises(TypeError):
        loaded.configs["recipe"] = "other.json"


@pytest.mark.parametrize(("change", "field"), [
    ({"configs": {}}, "project.configs.recipe"),
    ({"configs": {"recipe": 2}}, "project.configs.recipe"),
    ({"format_version": True}, "project.format_version"),
    ({"configs": {"recipe": "config.json", "scheduler": "no.json"}}, "project.configs.scheduler"),
    ({"artifacts": {"typo": "no"}}, "project.artifacts.typo"),
    ({"inputs": {"data": "  "}}, "project.inputs.data"),
])
def test_malformed_project_errors_have_field_locations(tmp_path, change, field):
    path, _ = project_fixture(tmp_path / "project")
    value = json.loads(path.read_text())
    value.update(change)
    write_json(path, value)
    report = check_task_project(path)
    assert not report["ok"]
    assert any(item["field"] == field for item in report["issues"])
    assert report["execution"]["user_code_imported"] is False


def test_invalid_nested_recipe_and_missing_binding_are_both_reported(tmp_path):
    path, values = project_fixture(tmp_path / "project")
    values["recipe"]["training"]["learning_rate"] = 0
    write_json(path.parent / "configs/recipe.json", values["recipe"])
    (path.parent / "data/input.bin").unlink()
    report = check_task_project(path)
    assert not report["ok"]
    assert any(item["field"] == "configs.recipe.training.learning_rate" for item in report["issues"])
    assert issues(report, "missing_binding")[0]["field"] == "project.resources.source"


def test_invalid_factory_reference_is_field_located_without_import(tmp_path):
    path, values = project_fixture(tmp_path / "project")
    values["recipe"]["factory"] = {"reference": "untouched_caller..invalid:make_task"}
    values["suite"]["views"][0]["scoring"]["scorer"] = "untouched_caller..invalid:score"
    write_json(path.parent / "configs/recipe.json", values["recipe"])
    write_json(path.parent / "configs/suite.json", values["suite"])
    report = check_task_project(path)
    assert not report["ok"]
    assert {item["field"] for item in issues(report, "invalid_factory")} == {"configs.recipe.factory.reference", "configs.suite.views[0].scoring.scorer"}


def test_runtime_input_binding_ambiguity_is_not_hidden(tmp_path):
    path, _ = project_fixture(tmp_path / "project")
    declaration = json.loads(path.read_text())
    declaration["resources"]["records"] = "data/input.bin"
    write_json(path, declaration)
    report = check_task_project(path)
    assert not report["ok"]
    assert issues(report, "conflicting_binding")[0]["field"] == "project.resources.records"


def test_original_resource_and_future_snapshot_can_share_an_unused_runtime_name(tmp_path):
    path, values = project_fixture(tmp_path / "project")
    declaration = json.loads(path.read_text())
    declaration["resources"]["dataset"] = "data"
    write_json(path, declaration)
    values["runtime"]["inputs"].pop("dataset")
    runtime_config = path.parent / "configs/runtime.json"
    write_json(runtime_config, values["runtime"])
    report = check_task_project(path)
    assert report["ok"], report["issues"]
    assert report["bindings"]["resources"]["dataset"]["status"] == "present"
    assert report["artifacts"]["dataset"]["status"] == "not_created"
    assert read_task_project(path).resolved_recipe().resources["dataset"] == str((path.parent / "data").resolve())
    # Once this name is consumed by a runtime port, its differing bindings are ambiguous.
    values["runtime"]["inputs"]["dataset"] = "directory"
    write_json(runtime_config, values["runtime"])
    report = check_task_project(path)
    assert not report["ok"]
    assert issues(report, "conflicting_binding")[0]["field"] == "project.artifacts.dataset"


def test_unknown_required_asset_reference_is_not_silently_optional(tmp_path):
    path, values = project_fixture(tmp_path / "project")
    values["snapshot"]["bindings"]["row-a"].append({"asset_id": "unbound", "required": True})
    write_json(path.parent / "configs/snapshot.json", values["snapshot"])
    report = check_task_project(path)
    assert not report["ok"]
    assert issues(report, "unknown_asset")[0]["field"] == "configs.snapshot.bindings.row-a"


def episode_config():
    return {"adapter": "untouched_caller:episode", "policy_id": "generated-policy-v1", "environment_id": "generated-environment-v1", "evaluator_id": "generated-evaluator-v1", "initial_state_field": "initial", "seed_field": "seed", "effects": "local", "recovery": "snapshot"}


def test_separate_episode_and_file_configs_cannot_be_combined(tmp_path):
    path, _ = project_fixture(tmp_path / "project")
    project = json.loads(path.read_text())
    project["configs"]["episodes"] = "configs/episodes.json"
    write_json(path, project)
    write_json(path.parent / "configs/episodes.json", episode_config())
    report = check_task_project(path)
    assert not report["ok"]
    assert issues(report, "invalid_config")[0]["field"] == "configs.prediction"


def test_episode_only_configuration_stays_static_and_external_replay_rejected(tmp_path):
    path, _ = project_fixture(tmp_path / "project")
    project = json.loads(path.read_text())
    project["configs"].pop("files")
    project["configs"]["episodes"] = "configs/episodes.json"
    write_json(path, project)
    config = episode_config()
    write_json(path.parent / "configs/episodes.json", config)
    assert check_task_project(path)["ok"]
    config.update(effects="external", recovery="replay")
    write_json(path.parent / "configs/episodes.json", config)
    report = check_task_project(path)
    assert not report["ok"]
    assert any(item["field"] == "configs.episodes.recovery" for item in report["issues"])


def test_different_inline_file_declaration_is_reported(tmp_path):
    path, values = project_fixture(tmp_path / "project")
    values["prediction"]["files"] = {"field": "other_files"}
    write_json(path.parent / "configs/prediction.json", values["prediction"])
    report = check_task_project(path)
    assert not report["ok"]
    assert issues(report, "conflicting_config")[0]["field"] == "configs.prediction.files"


def test_changed_prediction_journal_path_is_a_conflict_but_selected_training_run_is_allowed(tmp_path):
    path, values = project_fixture(tmp_path / "project")
    project = json.loads(path.read_text())
    project["artifacts"]["training_run"] = "runs/recovered"
    write_json(path, project)
    report = check_task_project(path)
    assert report["ok"]
    assert issues(report, "selected_run")[0]["severity"] == "info"
    values["prediction"]["output_dir"] = "../runs/other_prediction"
    write_json(path.parent / "configs/prediction.json", values["prediction"])
    report = check_task_project(path)
    assert not report["ok"]
    assert issues(report, "conflicting_path")[0]["field"] == "configs.prediction.output_dir"


def test_runtime_named_inputs_require_explicit_binding_and_correct_kind(tmp_path):
    path, values = project_fixture(tmp_path / "project")
    values["runtime"]["inputs"].update(records="directory", other="file")
    write_json(path.parent / "configs/runtime.json", values["runtime"])
    report = check_task_project(path)
    assert not report["ok"]
    assert issues(report, "runtime_input_kind")[0]["field"] == "configs.runtime.inputs.records"
    assert issues(report, "unbound_runtime_input")[0]["field"] == "configs.runtime.inputs.other"


def test_schemas_are_editor_usable_valid_and_generated_deterministically(tmp_path):
    from jsonschema import Draft202012Validator
    path, values = project_fixture(tmp_path / "project")
    payloads = {"project": json.loads(path.read_text()), **values, "episodes": episode_config()}
    for kind in SCHEMA_KINDS:
        schema = project_schema(kind)
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(payloads[kind])
        target = tmp_path / f"task-{kind}.schema.json"
        write_task_project_schema(target, kind)
        assert check_task_project_schema(target, kind)
        target.write_text("{}", encoding="utf-8")
        assert not check_task_project_schema(target, kind)
        schema["title"] = "caller mutation"
        assert project_schema(kind)["title"] != "caller mutation"


@pytest.mark.parametrize(("kind", "value"), [
    ("files", {"validator": "untouched_caller:validate"}),
    ("episodes", {**episode_config(), "effects": "external", "recovery": "replay"}),
    ("prediction", {"output_dir": "results", "episodes": episode_config(), "files": {"field": "files"}}),
    ("suite", {"views": [{"name": "external", "requires": "external", "external_protocol": "generated-v1", "scoring": {"scorer": "untouched_caller:score", "scorer_id": "v1"}}]}),
    ("runtime", {"entrypoint": "untouched_caller:run", "revision": "v1", "limits": {"max_seconds": -2}}),
    ("export", {"writer": "untouched_caller:write", "revision": "v1", "validation_id": "v1"}),
])
def test_schema_rejects_lifecycle_conflicts(kind, value):
    from jsonschema import Draft202012Validator
    assert not Draft202012Validator(project_schema(kind)).is_valid(value)


def test_project_read_supports_toml_and_unbound_config_rejects_implicit_cwd(tmp_path):
    directory = tmp_path / "project"
    directory.mkdir()
    path = directory / "project.toml"
    path.write_text('format = "arti.task-project"\nformat_version = 1\nname = "neutral"\n[configs]\nrecipe = "recipe.json"\n', encoding="utf-8")
    assert read_task_project(path).resolved_paths("configs")["recipe"] == directory / "recipe.json"
    with pytest.raises(ValueError, match="read a saved project"):
        TaskProjectConfig("neutral", {"recipe": "recipe.json"}).resolve_path("data.jsonl")


def test_unreadable_project_produces_machine_readable_failure(tmp_path):
    report = check_task_project(tmp_path / "absent.json")
    assert report["ok"] is False
    assert report["issues"][0]["code"] == "project_unreadable"
    json.dumps(report, allow_nan=False)
