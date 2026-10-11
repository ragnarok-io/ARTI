"""Fresh application-owned task projects over the existing lifecycle services."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
import struct
import tempfile

from .recipe_assets import DatasetSnapshotConfig, create_dataset_snapshot
from .recipe_workflow import write_json, write_jsonl
from .serialization import _file_sha256, _replace_atomic_file


def create_task_project(destination, *, name="task", template="files"):
    """Publish a runnable neutral CPU project without importing application code.

    The template owns its example model, native objective, binary codec and
    callbacks. Generation prepares a local immutable snapshot only; it does not
    train, install dependencies, run a provider or submit results. Publication
    follows the existing fresh-directory, local single-writer contract.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("project name must be a nonempty string")
    if template != "files":
        raise ValueError("supported task template: files")
    if not isinstance(destination, (str, Path)) or not str(destination).strip():
        raise ValueError("project destination is required")
    requested = Path(destination)
    if requested.exists() or requested.is_symlink():
        raise FileExistsError("task project already exists; choose a fresh directory")
    target = requested.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{target.name}-", dir=target.parent) as temporary:
        staging = Path(temporary) / "project"
        staging.mkdir()
        resources = files("arti").joinpath("templates", "task")
        for source, relative in (
            ("application.py", "code/task_app.py"),
            ("workflow.py", "workflow.py"),
            ("README.md", "README.md"),
            ("TUTORIAL.md", "TUTORIAL.md"),
        ):
            path = staging / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(resources.joinpath(source).read_bytes())
        (staging / ".gitignore").write_text(
            "__pycache__/\ndata/cache/\nruns/\nartifacts/\n", encoding="utf-8"
        )
        source = staging / "data/source"
        source.mkdir(parents=True)
        rows, assets, bindings = [], {}, {}
        # Small deterministic generated vectors; repeated references exercise
        # content deduplication without any dataset or model download.
        for index in range(12):
            asset = f"vector-{index % 6}"
            features = [((index % 6 + 1) * (axis + 2) % 11 - 5) / 4 for axis in range(4)]
            path = source / f"{asset}.bin"
            path.write_bytes(struct.pack("<4f", *features))
            identifier = f"record-{index}"
            rows.append(
                {
                    "id": identifier,
                    "target": sum(features) * 0.8,
                    "split": "train" if index < 6 else "validation" if index < 8 else "test",
                    "group": f"group-{index}",
                }
            )
            assets[asset] = f"../data/source/{asset}.bin"
            bindings[identifier] = [{"asset_id": asset, "role": "input", "required": True}]
        write_jsonl(source / "records.jsonl", rows)
        write_jsonl(
            staging / "data/requests.jsonl",
            [{"id": row["id"]} for row in rows if row["split"] == "test"],
        )
        write_jsonl(
            staging / "data/references.jsonl", [row for row in rows if row["split"] == "test"]
        )
        configs = staging / "configs"
        write_json(
            configs / "snapshot.json",
            {
                "format": "arti.dataset-snapshot-config",
                "records": "../data/source/records.jsonl",
                "assets": assets,
                "bindings": bindings,
                "asset_metadata": {asset: {"shape": [4]} for asset in assets},
                "group_field": "group",
                "source_revision": "generated-v1",
            },
        )
        create_dataset_snapshot(
            DatasetSnapshotConfig.read(configs / "snapshot.json"), staging / "data/snapshot"
        )
        (staging / "data/cache").mkdir()
        write_json(
            configs / "transform.json",
            {
                "factory": "task_app:decode_vectors",
                "revision": "vectors-v1",
                "options": {"scale": 0.5},
            },
        )
        write_json(
            configs / "recipe.json",
            {
                "format": "arti.task-recipe-config",
                "task": name,
                "factory": "task_app:build_task",
                "options": {
                    "seed": 29,
                    "input_shape": [4],
                    "batch_size": 3,
                    "transform": {"scale": 0.5},
                },
                "resources": {},
                "training": {
                    "device": "cpu",
                    "steps": 4,
                    "learning_rate": 0.02,
                    "gradient_accumulation_steps": 2,
                },
                "run": {
                    "output_dir": "../runs/training",
                    "checkpoint_steps": 1,
                    "evaluation_steps": 1,
                    "validation": {"monitor": "loss", "selection": "best"},
                },
            },
        )
        files_config = {
            "field": "files",
            "validator": "task_app:validate_files",
            "revision": "scalar-file-v1",
        }
        write_json(
            configs / "files.json", {"format": "arti.prediction-files-config", **files_config}
        )
        write_json(
            configs / "prediction.json",
            {
                "format": "arti.prediction-run-config",
                "output_dir": "../artifacts/prediction",
                "batch_size": 2,
            },
        )
        write_json(
            configs / "suite.json",
            {
                "format": "arti.evaluation-suite-config",
                "views": [
                    {
                        "name": "reference",
                        "scoring": {
                            "scorer": "task_app:reference_scores",
                            "scorer_id": "scalar-error-v1",
                        },
                        "reference_field": "target",
                    },
                    {
                        "name": "files",
                        "scoring": {
                            "scorer": "task_app:file_scores",
                            "scorer_id": "scalar-file-v1",
                        },
                        "requires": "output",
                        "files": True,
                    },
                ],
            },
        )
        write_json(
            configs / "runtime.json",
            {
                "format": "arti.task-runtime-profile",
                "entrypoint": "task_app:runtime",
                "revision": "file-task-v1",
                "requirements": ["torch>=2.2"],
                "network": "disabled",
                "inputs": {"requests": "file", "references": "file", "suite": "file"},
                "outputs": {"results": "directory", "evaluation": "file"},
                "options": {
                    "task_factory": "task_app:build_task",
                    "batch_size": 2,
                    "files": files_config,
                },
            },
        )
        write_json(
            configs / "export.json",
            {
                "format": "arti.result-export-config",
                "writer": "task_app:write_results",
                "revision": "records-v1",
                "validator": "task_app:validate_results",
                "validation_id": "records-v1",
                "archive": "zip",
                "instructions": "Inspect before use. Extract the archive; the selected deployment and attached original inputs/journal support an offline consumer with ARTI and PyTorch already installed.",
                "unresolved_checks": [
                    "Task quality is unevaluated against an application acceptance policy.",
                    "Strong network, memory and process-tree limits are not enforced by the local worker.",
                ],
            },
        )
        write_json(
            staging / "project.json",
            {
                "format": "arti.task-project",
                "format_version": 1,
                "name": name,
                "configs": {
                    role: f"configs/{role}.json"
                    for role in (
                        "recipe",
                        "snapshot",
                        "transform",
                        "prediction",
                        "files",
                        "suite",
                        "runtime",
                        "export",
                    )
                },
                "resources": {"snapshot": "data/snapshot", "cache": "data/cache"},
                "inputs": {
                    "requests": "data/requests.jsonl",
                    "references": "data/references.jsonl",
                    "suite": "configs/suite.json",
                },
                "artifacts": {
                    "training_run": "runs/training",
                    "model": "runs/training/model",
                    "dataset": "data/snapshot",
                    "cache": "data/cache",
                "preparation_report": "artifacts/preparation.json",
                    "prediction_run": "artifacts/prediction",
                    "evaluation": "artifacts/evaluation.json",
                    "deployment": "artifacts/deployment",
                    "result_export": "artifacts/results.zip",
                    "runtime_receipt": "artifacts/runtime.json",
                },
            },
        )
        inventory = {
            path.relative_to(staging).as_posix(): {
                "sha256": _file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for path in sorted(staging.rglob("*"))
            if path.is_file()
        }
        if target.exists() or target.is_symlink():
            raise FileExistsError("task project already exists; choose a fresh directory")
        _replace_atomic_file(staging, target)
    return {
        "ok": True,
        "directory": str(target),
        "project": str(target / "project.json"),
        "template": template,
        "files": inventory,
        "application_executed": False,
        "trained": False,
        "installs_dependencies": False,
    }


__all__ = ["create_task_project"]
