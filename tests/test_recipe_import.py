from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil

import pytest
import torch

import arti
from arti.cli import main
from arti.recipe_data import json_fingerprint, read_jsonl

from test_language_recipe import base_model, config, records, tokenizer, write
from test_recipe_recovery import equal, interrupt_after, policy


def sources(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    declarations = []
    for split, prefix in (("train", "old"), ("validation", "new"), ("test", "please")):
        rows = []
        for number, row in enumerate(record for record in records() if record["split"] == split):
            row["messages"][-2]["content"] = (
                f"{prefix} " + "color " * (number + 1) + row["messages"][-1]["content"]
            )
            rows.append({"key": row["id"], "dialog": row["messages"], "reviewer": "accepted"})
        declarations.append(
            arti.CausalLMDataSource(
                write(tmp_path / f"{split}.jsonl", rows),
                name=split,
                split=split,
                fields=arti.CausalLMImportFields(id="key", messages="dialog"),
            )
        )
    return arti.CausalLMDataImportConfig(tuple(declarations))


def imported(tmp_path):
    declared = sources(tmp_path / "raw")
    report = arti.import_causal_lm_data(declared, tmp_path / "snapshot")
    cfg = config(tmp_path)
    cfg = replace(
        cfg,
        data_path=report["data"]["path"],
        source_manifest=report["data"]["source_manifest"],
        cache_dir=tmp_path / "tokens",
    )
    return declared, report, cfg


def test_messages_import_is_lossless_split_explicit_and_portable(tmp_path):
    declared, result, cfg = imported(tmp_path)
    output = list(read_jsonl(cfg.data_path))
    manifest = result["manifest"]
    assert result["ok"] and not result["weights_loaded"] and not result["optimizer_created"]
    assert len(output) == 10 and manifest["records_dropped"] == 0
    assert {key: value["records"] for key, value in manifest["splits"].items()} == {
        "train": 5,
        "validation": 3,
        "test": 2,
    }
    assert str(tmp_path) not in json.dumps(manifest)
    assert manifest["output"]["sha256"] == hashlib.sha256(cfg.data_path.read_bytes()).hexdigest()
    for source in declared.sources:
        before = list(read_jsonl(source.path))
        after = [row for row in output if row["split"] == source.split]
        assert [row["messages"] for row in after] == [row["dialog"] for row in before]
        assert [row["id"] for row in after] == [f"{source.name}:{row['key']}" for row in before]
        assert (
            next(item for item in manifest["sources"] if item["name"] == source.name)["sha256"]
            == hashlib.sha256(source.path.read_bytes()).hexdigest()
        )
    assert arti.torch.import_causal_lm_data is arti.import_causal_lm_data


@pytest.mark.parametrize("conversational", [False, True])
@pytest.mark.parametrize("file_format", ["jsonl", "json"])
def test_prompt_completion_preserves_history_whitespace_and_answer(
    tmp_path, conversational, file_format
):
    train_messages = [
        {"role": "system", "content": "Answer only"},
        {"role": "user", "content": "old color blue"},
        {"role": "assistant", "content": "blue"},
        {"role": "user", "content": "new color red"},
    ]
    answer = "  red\n"
    source = tmp_path / "input.json"
    rows = [
        {
            "question": train_messages if conversational else "new color red",
            "answer": [{"role": "assistant", "content": answer}] if conversational else answer,
        }
    ]
    if file_format == "json":
        source.write_text(json.dumps(rows), encoding="utf-8")
    else:
        write(source, rows)
    validation = write(
        tmp_path / "validation.jsonl",
        [
            {
                "messages": [
                    {"role": "user", "content": "color green"},
                    {"role": "assistant", "content": "green"},
                ]
            }
        ],
    )
    cfg = arti.CausalLMDataImportConfig(
        (
            arti.CausalLMDataSource(
                source,
                "answers",
                "train",
                "prompt_completion",
                file_format,
                arti.CausalLMImportFields(prompt="question", completion="answer"),
            ),
            arti.CausalLMDataSource(validation, "heldout", "validation"),
        )
    )
    result = arti.import_causal_lm_data(cfg, tmp_path / "snapshot")
    row = list(read_jsonl(result["data"]["path"]))[0]
    assert row["id"] == "answers:1" and row["messages"][-1]["content"] == answer
    assert row["messages"][:-1] == (
        train_messages if conversational else [{"role": "user", "content": "new color red"}]
    )
    assert result["manifest"]["splits"]["test"]["records"] == 0


def test_cli_relative_config_does_not_load_tokenizer_host_or_optimizer(
    tmp_path, monkeypatch, capsys
):
    declared = sources(tmp_path / "raw")
    payload = {
        "task": "causal-lm",
        "sources": [
            {
                "path": str(item.path.relative_to(tmp_path)),
                "name": item.name,
                "split": item.split,
                "fields": {"id": "key", "messages": "dialog"},
            }
            for item in declared.sources
        ],
    }
    path = tmp_path / "import.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    import arti.language_recipe as language

    def fail(*_args, **_kwargs):
        raise AssertionError("data import must not load a tokenizer/model/optimizer")

    monkeypatch.setattr(language, "_load_model", fail)
    monkeypatch.setattr(language, "_load_tokenizer", fail)
    monkeypatch.setattr(torch.optim, "AdamW", fail)
    assert main(["recipe", "import-data", str(path), "--output", str(tmp_path / "snapshot")]) == 0
    report = json.loads(capsys.readouterr().out)
    assert len(list(read_jsonl(report["data"]["path"]))) == 10
    assert arti.CausalLMDataImportConfig.read(path).sources == declared.sources


def test_toml_import_config_and_source_manifest_paths_are_relative(tmp_path):
    declared = sources(tmp_path / "raw")
    path = tmp_path / "import.toml"
    path.write_text(
        'task = "causal-lm"\n'
        + "".join(
            f'[[sources]]\npath = "raw/{source.split}.jsonl"\nname = "{source.name}"\nsplit = "{source.split}"\n[sources.fields]\nid = "key"\nmessages = "dialog"\n'
            for source in declared.sources
        ),
        encoding="utf-8",
    )
    assert arti.CausalLMDataImportConfig.read(path) == declared
    result = arti.import_causal_lm_data(path, tmp_path / "snapshot")
    payload = config(tmp_path).to_dict()
    payload["data"].update(path="snapshot/dataset.jsonl", source_manifest="snapshot/manifest.json")
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(payload), encoding="utf-8")
    cfg = arti.CausalLMRecipeConfig.read(recipe_path)
    assert str(cfg.source_manifest) == result["data"]["source_manifest"]
    assert arti.CausalLMRecipeConfig.read(recipe_path).to_dict()["data"]["source_manifest"] == str(
        cfg.source_manifest
    )


@pytest.mark.parametrize("engine", ["torch", "accelerate"])
def test_import_prepare_interrupt_resume_evaluate_and_native_deploy(tmp_path, engine):
    pytest.importorskip("transformers")
    if engine == "accelerate":
        pytest.importorskip("accelerate")
    declared, report, cfg = imported(tmp_path)
    cfg = replace(cfg, training=replace(cfg.training, engine=engine))
    prepared = arti.prepare_causal_lm_data(cfg, tokenizer=tokenizer())
    assert prepared["dataset"]["source_manifest"] == report["manifest"]
    assert prepared["dataset"]["sha256"] == report["manifest"]["output"]["dataset_sha256"]
    raw_paths = [source.path for source in declared.sources]
    for path in raw_paths:
        path.unlink()
    assert arti.prepare_causal_lm_data(cfg, tokenizer=tokenizer())["data_preparation"]["cache_hit"]
    full = arti.CausalLMRecipe(cfg, model=base_model(), tokenizer=tokenizer())
    full.fit(run=policy(tmp_path / "full"))
    current = arti.CausalLMRecipe(cfg, model=base_model(), tokenizer=tokenizer())
    with pytest.raises(KeyboardInterrupt):
        current.fit(run=policy(tmp_path / "first"), on_checkpoint=interrupt_after(1))
    plan = arti.plan_recipe_resume(tmp_path / "first")
    assert plan["remaining_optimizer_steps"] == 3
    assert (
        json.loads((tmp_path / "first" / "data_source_manifest.json").read_text())
        == report["manifest"]
    )
    restored = arti.CausalLMRecipe(cfg, model=base_model(), tokenizer=tokenizer())
    result = restored.resume_run(tmp_path / "first", run=policy(tmp_path / "restored"))
    equal(full.model.state_dict(), restored.model.state_dict())
    equal(full.session.optimizer.state_dict(), restored.session.optimizer.state_dict())
    equal(full.session.scheduler.state_dict(), restored.session.scheduler.state_dict())
    assert full.session.loss_history == restored.session.loss_history
    assert full.train_data.state_dict() == restored.train_data.state_dict()
    assert result["validation"]["source_manifest_sha256"] == report["manifest"]["sha256"]
    assert (
        arti.serialization.load(result["checkpoint"]).manifest["architecture"]["config"][
            "task_recipe"
        ]["dataset"]["source_manifest"]
        == report["manifest"]
    )
    native = tmp_path / "native-base"
    base_model().save_pretrained(native)
    packaged = arti.package_causal_lm_deployment(
        Path(result["model"]).parent, tmp_path / "package", include_base=True, base_model=native
    )
    moved = tmp_path / "moved-package"
    shutil.copytree(packaged["directory"], moved)
    # Deployment keeps provenance as metadata, not a dependency on raw/source files.
    cfg.data_path.unlink()
    cfg.source_manifest.unlink()
    predictor = arti.CausalLMPredictor.from_pretrained(moved, local_files_only=True)
    assert predictor.contract["dataset"]["source_manifest"] == report["manifest"]
    batch = restored.data.splits["test"].batches(2)[0]
    with torch.no_grad():
        equal(predictor.model(**batch).logits, restored.model(**batch).logits)
    assert (
        len(
            predictor.predict_records(
                [{"id": "new", "messages": [{"role": "user", "content": "color blue"}]}]
            )
        )
        == 1
    )


def test_active_field_mapping_and_integer_source_ids(tmp_path):
    declared = sources(tmp_path / "raw")
    source = declared.sources[0]
    row = list(read_jsonl(source.path))[0]
    write(source.path, [{"row": 7, "messages": row["dialog"][:-1], "answer": row["dialog"][-1:]}])
    source = replace(
        source,
        format="prompt_completion",
        fields=arti.CausalLMImportFields(id="row", prompt="messages", completion="answer"),
    )
    result = arti.import_causal_lm_data(
        replace(declared, sources=(source, *declared.sources[1:])), tmp_path / "snapshot"
    )
    assert list(read_jsonl(result["data"]["path"]))[0]["id"] == "train:7"
    with pytest.raises(ValueError, match="active source fields"):
        replace(source, fields=arti.CausalLMImportFields(prompt="answer", completion="answer"))


def test_imported_tokens_labels_and_masks_equal_original_chat_contract(tmp_path):
    pytest.importorskip("transformers")
    declared, _report, cfg = imported(tmp_path)
    source = declared.sources[0]
    data = arti.CausalLMRecipe(cfg, model=base_model(), tokenizer=tokenizer()).data
    from arti.recipe_chat import encode_final_answer

    for raw, item in zip(read_jsonl(source.path), data.splits["train"].examples, strict=True):
        original = encode_final_answer(
            item.id, raw["dialog"], tokenizer(), template_kwargs=dict(cfg.template_kwargs)
        )
        equal(item.input_ids, original.input_ids)
        equal(item.labels, original.labels)
        assert item.labels[: item.prompt_length].eq(-100).all()
        assert item.labels[-1] == tokenizer().eos_token_id
    for split in data.splits.values():
        for batch in split.batches(2):
            assert batch["labels"][batch["attention_mask"].eq(0)].eq(-100).all()


def test_same_split_repeats_are_retained_and_reported(tmp_path):
    declared = sources(tmp_path / "raw")
    source = declared.sources[0]
    rows = list(read_jsonl(source.path))
    rows.append({**rows[0], "key": "repeat"})
    write(source.path, rows)
    report = arti.import_causal_lm_data(declared, tmp_path / "snapshot")
    assert report["manifest"]["splits"]["train"]["records"] == 6
    assert report["manifest"]["splits"]["train"]["repeated_conversations"] == 1


@pytest.mark.parametrize(
    "change",
    ["duplicate-id", "cross-split-repeat", "missing-answer", "tool-message", "malformed", "empty"],
)
def test_invalid_sources_identify_location_without_publishing_partial_data(tmp_path, change):
    declared = sources(tmp_path / "raw")
    source = declared.sources[0]
    rows = list(read_jsonl(source.path))
    if change == "duplicate-id":
        rows.append(rows[0])
    elif change == "cross-split-repeat":
        rows[0]["dialog"] = list(read_jsonl(declared.sources[1].path))[0]["dialog"]
    elif change == "missing-answer":
        rows[0]["dialog"] = rows[0]["dialog"][:-1]
    elif change == "tool-message":
        rows[0]["dialog"][0]["tool_calls"] = []
    elif change == "empty":
        rows = []
    write(source.path, rows)
    if change == "malformed":
        source.path.write_text("{broken}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="source"):
        arti.import_causal_lm_data(declared, tmp_path / "snapshot")
    assert not (tmp_path / "snapshot").exists()


def test_snapshot_drift_and_manifest_drift_are_checked_before_host(tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    _declared, _report, cfg = imported(tmp_path)
    import arti.language_recipe as language

    monkeypatch.setattr(
        language,
        "_load_model",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("host")),
    )
    rows = list(read_jsonl(cfg.data_path))
    rows[0]["messages"][-1]["content"] = "blue"
    write(cfg.data_path, rows)
    with pytest.raises(ValueError, match="does not describe"):
        arti.CausalLMRecipe(cfg, tokenizer=tokenizer())
    manifest = json.loads(cfg.source_manifest.read_text())
    manifest["records_dropped"] = 1
    cfg.source_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="fingerprint"):
        arti.prepare_causal_lm_data(cfg, tokenizer=tokenizer())


def test_snapshot_relocation_preserves_identity_and_token_cache(tmp_path):
    pytest.importorskip("transformers")
    _declared, report, cfg = imported(tmp_path)
    before = arti.prepare_causal_lm_data(cfg, tokenizer=tokenizer())
    shutil.copytree(tmp_path / "snapshot", tmp_path / "moved")
    moved = replace(
        cfg,
        data_path=tmp_path / "moved" / "dataset.jsonl",
        source_manifest=tmp_path / "moved" / "manifest.json",
    )
    after = arti.prepare_causal_lm_data(moved, tokenizer=tokenizer())
    assert before["dataset"] == after["dataset"]
    assert after["data_preparation"]["cache_hit"]
    left = arti.CausalLMRecipe(cfg, model=base_model(), tokenizer=tokenizer())
    right = arti.CausalLMRecipe(moved, model=base_model(), tokenizer=tokenizer())
    assert left.task_fingerprint == right.task_fingerprint
    assert str(tmp_path) not in json.dumps(left.task_contract)
    assert report["manifest"]["sha256"] == json_fingerprint(
        {key: value for key, value in report["manifest"].items() if key != "sha256"}
    )


def test_reimport_from_relocated_sources_preserves_snapshot_bytes_and_lineage(tmp_path):
    declared, report, cfg = imported(tmp_path)
    copied = tmp_path / "relocated-exports"
    copied.mkdir()
    sources = []
    for source in declared.sources:
        target = copied / source.path.name
        shutil.copyfile(source.path, target)
        sources.append(replace(source, path=target))
    next_report = arti.import_causal_lm_data(
        replace(declared, sources=tuple(sources)), tmp_path / "snapshot-again"
    )
    assert next_report["manifest"] == report["manifest"]
    assert Path(next_report["data"]["path"]).read_bytes() == cfg.data_path.read_bytes()


def test_import_config_serialization_roundtrip(tmp_path):
    declared = sources(tmp_path / "raw")
    path = tmp_path / "import-roundtrip.json"
    path.write_text(json.dumps(declared.to_dict()), encoding="utf-8")
    assert arti.CausalLMDataImportConfig.read(path) == declared


@pytest.mark.parametrize(
    "completion",
    [
        None,
        [],
        [{"role": "assistant", "content": "red"}, {"role": "assistant", "content": "blue"}],
        [{"role": "user", "content": "red"}],
    ],
)
def test_import_does_not_guess_partial_or_multiple_completions(tmp_path, completion):
    declared = sources(tmp_path / "raw")
    source = declared.sources[0]
    write(source.path, [{"prompt": "old color red", "completion": completion}])
    source = replace(source, format="prompt_completion", fields=arti.CausalLMImportFields())
    with pytest.raises(ValueError, match="source 'train', record 1"):
        arti.import_causal_lm_data(
            replace(declared, sources=(source, *declared.sources[1:])), tmp_path / "snapshot"
        )
    assert not (tmp_path / "snapshot").exists()


def test_publication_failure_and_existing_snapshot_do_not_overwrite(tmp_path, monkeypatch):
    import arti.recipe_import as module

    declared = sources(tmp_path / "raw")
    monkeypatch.setattr(
        module, "_replace_atomic_file", lambda *_args: (_ for _ in ()).throw(OSError("publish"))
    )
    with pytest.raises(OSError, match="publish"):
        arti.import_causal_lm_data(declared, tmp_path / "snapshot")
    assert not (tmp_path / "snapshot").exists()
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        arti.import_causal_lm_data(declared, existing)


@pytest.mark.parametrize("value", [True, 4, ""])
def test_invalid_source_manifest_config(tmp_path, value):
    with pytest.raises(ValueError, match="source_manifest"):
        replace(config(tmp_path), source_manifest=value)
