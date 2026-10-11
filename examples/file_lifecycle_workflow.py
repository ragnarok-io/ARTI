"""Neutral CPU file lifecycle using the shared task, journal and delivery services.

Run from the checkout: python -m examples.file_lifecycle_workflow --output outputs/files
The output directory must be fresh. No download, installation or submission occurs.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys

import torch

import arti
from arti.recipe_data import read_jsonl
from arti.recipe_workflow import write_json, write_jsonl
from examples.asset_task_recipe import FACTORY, TRANSFORM


def _equal(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _equal(a, b)
    else:
        assert left == right


def _files(directory):
    return {
        path.relative_to(directory).as_posix(): (path.stat().st_mtime_ns, path.read_bytes())
        for path in directory.rglob("*")
        if path.is_file()
    }


def _remove(directory, root):
    # Validate the absolute deletion target within this example's fresh workspace.
    target, root = directory.resolve(), root.resolve()
    assert target != root and target.is_relative_to(root)
    shutil.rmtree(target)


def _inputs(directory):
    values = torch.randn(8, 3, generator=torch.Generator().manual_seed(23))
    records, assets, bindings, metadata = [], {}, {}, {}
    for index in range(14):
        identifier, asset = f"sample-{index}", f"input-{index}"
        path = directory / f"{index % len(values)}.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        value = values[index % len(values)]
        path.write_bytes(struct.pack("<3f", *value.tolist()))
        assets[asset], metadata[asset] = path, {"shape": [3]}
        bindings[identifier] = [arti.AssetReference(asset)]
        records.append(
            {
                "id": identifier,
                "target": float(value.sum() * 1.5),
                "split": "train" if index < 8 else "validation" if index < 10 else "test",
                "group": f"group-{index}",
            }
        )
    assets["optional-reference"] = None
    bindings[records[-1]["id"]].append(
        arti.AssetReference("optional-reference", "reference", required=False)
    )
    path = directory / "records.jsonl"
    write_jsonl(path, records)
    return arti.DatasetSnapshotConfig(
        path, assets, bindings, metadata, group_field="group", source_revision="generated-v1"
    ), records, values


def _suite():
    return arti.EvaluationSuiteConfig(
        (
            arti.EvaluationViewConfig(
                "reference",
                arti.PredictionScoringConfig(
                    "examples.prediction_scorers:regression", "original-units-v1"
                ),
                reference_field="target",
            ),
            arti.EvaluationViewConfig(
                "files",
                arti.PredictionScoringConfig(
                    "examples.evaluation_suite_scorers:file_validity", "array-binary-v1"
                ),
                requires="output",
                files=True,
            ),
        )
    )


def _cold_runtime(package, inputs, job, evaluation, receipt, directory):
    directory.mkdir()
    program = (
        "import json,sys;from pathlib import Path;import arti;"
        "assert Path(arti.__file__).resolve()==Path(sys.argv[6]).resolve();"
        "root=Path(sys.argv[2]);"
        "value=arti.run_task_runtime(sys.argv[1],"
        "inputs={name:root/(name+'.jsonl' if name!='suite' else 'suite.json') "
        "for name in ('requests','references','suite')},"
        "outputs={'results':sys.argv[3],'evaluation':sys.argv[4]},"
        "options={'resume':True},output=sys.argv[5]);"
        "print(json.dumps(value));assert value['status']=='completed'"
    )
    process = subprocess.run(
        [
            sys.executable, "-c", program, str(package), str(inputs), str(job),
            str(evaluation), str(receipt), str(Path(arti.__file__).resolve()),
        ],
        cwd=directory,
        env={
            **os.environ,
            "PYTHONPATH": str(Path(arti.__file__).resolve().parent.parent),
            "PYTHONDONTWRITEBYTECODE": "1",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
        },
        text=True,
        capture_output=True,
        check=True,
    )
    result = json.loads(process.stdout)
    assert result["runtime_identity"]["framework_root"] == str(
        Path(arti.__file__).resolve().parent.parent
    )
    assert result["worker"]["result"]["prediction"]["newly_predicted_records"] == 0
    return result


def run_workflow(output):
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    original, local = root / "original", root / "local"
    cfg, rows, values = _inputs(original / "source")
    snapshot = arti.create_dataset_snapshot(cfg, original / "snapshot")
    transform = arti.AssetTransformConfig(TRANSFORM, "float-array-v1", {"scale": 0.75})
    prepared = arti.prepare_dataset_assets(snapshot, original / "cache", transform, batch_size=3)
    cache_before = _files(original / "cache")
    reused = arti.prepare_dataset_assets(snapshot, original / "cache", transform, batch_size=2)
    assert reused.report["produced"] == 0 and reused.report["reused"] == 8
    assert _files(original / "cache") == cache_before
    resources = {"snapshot": str(snapshot.directory), "cache": str(original / "cache")}
    config = arti.TaskRecipeConfig(
        "generated-array-regression", FACTORY,
        options={
            "input_shape": [3], "batch_size": 4, "dropout": 0.2,
            "transform": {"scale": 0.75}, "file_formats": ["array", "binary"],
        },
        resources=resources,
        training=arti.ARTITrainingConfig(
            device="cpu", steps=4, learning_rate=0.02, gradient_accumulation_steps=2
        ),
    )
    full, interrupted = arti.TaskRecipe.from_config(config), arti.TaskRecipe.from_config(config)
    initial = {name: parameter.detach().clone() for name, parameter in full.model.named_parameters()}
    policy = arti.ARTITrainingRunConfig(
        output_dir=original / "full", checkpoint_steps=1, evaluation_steps=1,
        validation=arti.ARTIValidationConfig(monitor="loss", selection="best"),
    )
    rng = torch.get_rng_state().clone()
    full_report = full.fit(run=policy)
    torch.set_rng_state(rng)

    def stop_training(session, _path):
        if session.global_step == 2:
            raise KeyboardInterrupt

    try:
        interrupted.fit(run=replace(policy, output_dir=original / "partial"), on_checkpoint=stop_training)
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("training did not reach the requested interruption")
    moved, cache = original / "moved-snapshot", original / "moved-cache"
    shutil.move(snapshot.directory, moved)
    shutil.move(original / "cache", cache)
    _remove(original / "source", root)
    resources = {"snapshot": str(moved), "cache": str(cache)}
    recovered = arti.TaskRecipe.from_config(replace(config, resources=resources))
    training = recovered.resume_run(original / "partial", output_dir=original / "recovered")
    _equal(full.model.state_dict(), recovered.model.state_dict())
    _equal(full.session.optimizer.state_dict(), recovered.session.optimizer.state_dict())
    _equal(full.session.loss_history, recovered.session.loss_history)
    assert full_report["selection"] == training["selection"]
    assert recovered.task_fingerprint == full.task_fingerprint
    assert recovered.task_contract["objective"] == "scaled-mse"
    assert _files(cache) == cache_before
    _equal(recovered.model.feature_mean, (values * 0.75).mean(0))
    _equal(recovered.model.feature_scale, (values * 0.75).std(0, unbiased=False))
    changed = [
        name for name, parameter in recovered.model.named_parameters()
        if not torch.equal(initial[name], parameter)
    ]
    assert any(name.startswith("program.") for name in changed)
    assert any(name.startswith("output.") for name in changed)

    files = arti.PredictionFilesConfig(
        "files", "examples.asset_task_recipe:validate_files", "native-files-v1"
    )
    code = original / "application/examples"
    code.mkdir(parents=True)
    (code / "__init__.py").write_text("", encoding="utf-8")
    for name in (
        "runtime_task.py", "asset_task_recipe.py", "custom_task_recipe.py",
        "evaluation_suite_scorers.py", "prediction_scorers.py",
    ):
        shutil.copyfile(Path(__file__).with_name(name), code / name)
    package, inputs = local / "package", local / "inputs"
    runtime = arti.TaskRuntimeProfile(
        "examples.runtime_task:validate", "file-lifecycle-v1", network="disabled",
        requirements=("torch>=2",),
        inputs={"requests": "file", "references": "file", "suite": "file"},
        outputs={"results": "directory", "evaluation": "file"},
        options={"task_factory": FACTORY, "batch_size": 3, "files": files.to_dict()},
    )
    arti.package_recipe_deployment(
        training["model"], package, resources=resources, runtime=runtime,
        code={"application": code.parent},
    )
    test_rows = [row for row in rows if row["split"] == "test"]
    requests = [{"id": row["id"]} for row in test_rows]
    expected = [row["id"] for row in requests]
    write_jsonl(inputs / "requests.jsonl", requests)
    write_jsonl(inputs / "references.jsonl", test_rows)
    suite = _suite()
    write_json(inputs / "suite.json", suite.to_dict())
    write_json(inputs / "training.json", training)
    write_json(inputs / "preparation.json", prepared.report)
    with recovered.session.checkpoint_context(full_report["selected_checkpoint"]):
        expected_values = recovered.predictor().predict_records(requests, batch_size=3)
    job = local / "prediction"
    run = arti.PredictionRunConfig(job, batch_size=3, files=files)
    try:
        arti.run_prediction(
            package, inputs / "requests.jsonl", run, task_factory=FACTORY,
            on_batch=lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()),
        )
    except KeyboardInterrupt:
        pass
    else:
        raise AssertionError("prediction did not reach the requested interruption")
    committed = _files(job / "parts")
    predictor = arti.TaskPredictor.from_pretrained(package, task_factory=FACTORY)
    producer, generated = predictor._predict_fn, []

    def track(model, records, **kwargs):
        generated.extend(row["id"] for row in records)
        return producer(model, records, **kwargs)

    predictor._predict_fn = track
    prediction = arti.run_prediction(package, None, run, resume=True, predictor=predictor)
    assert generated == expected[3:]
    assert all(_files(job / "parts")[name] == value for name, value in committed.items())
    actual = list(read_jsonl(job / "predictions.jsonl"))
    assert [{key: row[key] for key in ("id", "prediction")} for row in actual] == expected_values
    evaluation = arti.evaluate_prediction_suite(
        job, suite, references=inputs / "references.jsonl", output=local / "suite.json"
    )
    assert evaluation["complete_evidence"]
    phases = [row["execution"]["phase"] for row in evaluation["views"]["files"]["outcomes"]]
    assert phases == ["initial"] * 3 + ["recovered"]
    parts_before = _files(job / "parts")
    _remove(original, root)
    cold = _cold_runtime(
        package, inputs, job, local / "validation.json", local / "runtime.json", root / "consumer"
    )
    assert _files(job / "parts") == parts_before
    validation = cold["worker"]["result"]
    archive = root / "results.zip"
    exported = arti.export_prediction_results(
        job,
        arti.ResultExportConfig(
            "examples.result_writers:write_results", "record-layout-v1", archive="zip",
            validator="examples.result_writers:validate_results", validation_id="record-validator-v1",
            instructions="Extract; run the selected entrypoint with the attached inputs and journal.",
            unresolved_checks=("task quality and external host resource enforcement are unqualified",),
            examples=(expected[0],),
        ),
        archive, expected_ids=expected, deployment=package,
        evidence={
            "suite": local / "suite.json", "runtime": local / "runtime.json",
            "validation": local / "validation.json", "cold_suite": validation["scores"]["file"],
        },
        attachments={"inputs": inputs, "journal": job},
    )
    assert _files(job / "parts") == parts_before and generated == expected[3:]
    _remove(local, root)
    inspected = arti.inspect_result_export(archive)
    assert inspected["sha256"] == exported["sha256"] and inspected["records"] == len(expected)
    report = {
        "ok": True, "framework_file": str(Path(arti.__file__).resolve()),
        "snapshot": snapshot.manifest(), "preparation": prepared.report, "cache_reuse": reused.report,
        "cache_files_unchanged": True, "native_state_equal": True, "changed_parameters": changed,
        "objective": recovered.task_contract["objective"], "selection": training["selection"],
        "optimizer_updates_recovered": training["optimizer_updates_this_attempt"],
        "prediction": {key: prediction[key] for key in (
            "committed_records", "reused_records", "newly_predicted_records", "output_files"
        )},
        "generated_on_recovery": generated, "execution_phases": phases,
        "suite": {"complete_evidence": True, "views": evaluation["views"]},
        "cold_runtime": {"status": cold["status"], "framework_root": cold["runtime_identity"]["framework_root"],
                         "newly_predicted_records": validation["prediction"]["newly_predicted_records"],
                         "network_enforced": cold["network_enforced"], "enforced": cold["enforced"],
                         "installs_dependencies": cold["installs_dependencies"]},
        "originals_deleted": not original.exists() and not local.exists(),
        "archive": str(archive), "export": exported, "inspection": inspected,
    }
    write_json(root / "receipt.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = run_workflow(args.output)
    print(json.dumps({"ok": report["ok"], "archive": report["archive"], "receipt": str(args.output / "receipt.json")}))


if __name__ == "__main__":
    main()
