from __future__ import annotations

from dataclasses import replace
import json
import shutil
from pathlib import Path as PathLike

import pytest
import torch

import arti
from arti.cli import main
from arti.language_recipe import (
    CausalLMRecipe,
    CausalLMRecipeConfig,
    CausalLMModelConfig,
    CausalLMGenerationConfig,
    CausalLMOptimizerConfig,
    CausalLMPredictor,
)
from arti.recipe_chat import (
    ChatDataset,
    ChatFields,
    collate_chat,
    encode_final_answer,
    target_token_weight,
    tokenizer_fingerprint,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def tokenizer():
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast

    vocab = {
        word: index
        for index, word in enumerate(
            [
                "[UNK]",
                "[PAD]",
                "<|im_start|>",
                "<|im_end|>",
                "system",
                "user",
                "assistant",
                "color",
                "red",
                "blue",
                "green",
                "Answer",
                "only",
                "please",
                "old",
                "new",
                "42",
                "7",
            ]
        )
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = WhitespaceSplit()
    value = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="<|im_end|>",
        additional_special_tokens=["<|im_start|>"],
    )
    value.chat_template = "{% for m in messages %}{{ '<|im_start|>' + m['role'] + '\n' + m['content'] + '<|im_end|>\n' }}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    return value


def base_model():
    from transformers import Qwen3Config, Qwen3ForCausalLM

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(99)
        config = Qwen3Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            max_position_embeddings=128,
            pad_token_id=1,
            eos_token_id=3,
            attention_dropout=0.0,
        )
        return Qwen3ForCausalLM(config)


def records():
    return [
        {
            "id": f"{split}-{index}",
            "split": split,
            "messages": [
                {"role": "system", "content": "Answer only"},
                {"role": "user", "content": f"color {color}"},
                {"role": "assistant", "content": color},
            ],
        }
        for split, colors in (
            ("train", ["red", "blue", "green", "red", "blue"]),
            ("validation", ["red", "green", "blue"]),
            ("test", ["blue", "red"]),
        )
        for index, color in enumerate(colors)
    ]


def write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


def config(tmp_path, **kwargs):
    return CausalLMRecipeConfig(
        write(tmp_path / "chat.jsonl", records()),
        model=CausalLMModelConfig("example/tiny-qwen", "cpu-fixture", "float32", True),
        adapter=arti.FormulaAdapterConfig(rank=4, last_layers=2),
        training=arti.ARTITrainingConfig(device="cpu", steps=4, learning_rate=0.01),
        optimizer=CausalLMOptimizerConfig(weight_decay=0),
        generation=CausalLMGenerationConfig(max_new_tokens=4, batch_size=2),
        batch_size=2,
        max_length=64,
        **kwargs,
    )


def recipe(tmp_path, **kwargs):
    return CausalLMRecipe(config(tmp_path, **kwargs), model=base_model(), tokenizer=tokenizer())


def test_final_answer_mask_keeps_full_history_and_native_eos():
    tok = tokenizer()
    messages = [
        {"role": "user", "content": "old color blue"},
        {"role": "assistant", "content": "blue"},
        {"role": "user", "content": "new color red"},
        {"role": "assistant", "content": "red"},
    ]
    example = encode_final_answer("history", messages, tok, template_kwargs={})
    assert len(example.messages) == 4
    assert "blue" in tok.decode(example.input_ids[: example.prompt_length])
    assert example.labels[: example.prompt_length].eq(-100).all()
    target = example.labels[example.labels.ne(-100)].tolist()
    assert target == tok("red", add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    assert example.target_tokens == len(target)


def test_padding_does_not_mask_real_eos_when_pad_equals_eos():
    tok = tokenizer()
    tok.pad_token = tok.eos_token
    examples = [
        encode_final_answer(str(index), row["messages"], tok, template_kwargs={})
        for index, row in enumerate(records()[:2])
    ]
    batch = collate_chat(examples, pad_token_id=tok.pad_token_id, pad_to_multiple_of=8)
    assert batch["labels"][batch["attention_mask"].eq(0)].eq(-100).all()
    assert int(batch["labels"].eq(tok.eos_token_id).sum()) == 2
    assert target_token_weight(batch) == sum(example.target_tokens for example in examples)


def test_overlength_is_reported_not_silently_truncated(tmp_path):
    tok = tokenizer()
    rows = records()
    rows[0]["messages"][1]["content"] = "red " * 200
    path = write(tmp_path / "data.jsonl", rows)
    with pytest.raises(ValueError, match="silently truncated"):
        ChatDataset(
            path,
            tok,
            fields=ChatFields(),
            template_kwargs={},
            max_length=64,
            overlength="error",
            pad_to_multiple_of=8,
        )
    data = ChatDataset(
        path,
        tok,
        fields=ChatFields(),
        template_kwargs={},
        max_length=64,
        overlength="drop",
        pad_to_multiple_of=8,
    )
    assert data.manifest(include_ids=True)["splits"]["train"]["dropped_ids"] == ["train-0"]
    assert len(data.splits["train"]) == 4


def test_qwen_native_objective_artifact_and_frozen_base(tmp_path):
    current = recipe(tmp_path)
    assert current.check()["ok"]
    assert current.plan()["backbone_frozen"]
    assert current.plan()["task_recipe"]["supervision"] == "final-assistant-through-native-eos"
    owned = {id(parameter) for parameter in current.model.arti.parameters()}
    assert all(
        not parameter.requires_grad
        for parameter in current.model.parameters()
        if id(parameter) not in owned
    )
    before = {
        name: parameter.detach().clone() for name, parameter in current.model.named_parameters()
    }
    result = current.fit(
        run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "run", checkpoint_steps=2)
    )
    assert result["optimizer_steps"] == 4
    assert result["status"] == "completed"
    assert result["data_cursor"]["samples_seen"] == 7
    assert result["validation"]["supervised_tokens"] > 0
    assert (tmp_path / "run" / "tokenizer" / "tokenizer.json").exists()
    assert (tmp_path / "run" / "model" / "arti-hub.json").exists()
    assert any(
        not torch.equal(parameter, before[name])
        for name, parameter in current.model.named_parameters()
        if id(parameter) in owned
    )
    assert all(
        torch.equal(parameter, before[name])
        for name, parameter in current.model.named_parameters()
        if id(parameter) not in owned
    )
    inspected = arti.load(result["model"], load_checkpoint=False)
    assert "task_recipe" in inspected.manifest["architecture"]["config"]
    assert not any("embed_tokens" in name or "lm_head" in name for name in inspected.state_dict)
    assert str(tmp_path) not in json.dumps(inspected.manifest["architecture"]["config"])


@pytest.mark.parametrize("grouped", [True, False])
def test_resume_matches_uninterrupted_weights_and_data_cursor(tmp_path, grouped):
    cfg = config(tmp_path, group_by_length=grouped)
    cfg = replace(cfg, training=replace(cfg.training, gradient_accumulation_steps=2))
    complete = CausalLMRecipe(cfg, model=base_model(), tokenizer=tokenizer())
    complete.fit(steps=5, run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "complete"))
    first = CausalLMRecipe(cfg, model=base_model(), tokenizer=tokenizer())
    saved = first.fit(steps=2, run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "first"))
    resumed = CausalLMRecipe(cfg, model=base_model(), tokenizer=tokenizer())
    result = resumed.fit(
        steps=3,
        resume_from_checkpoint=saved["checkpoint"],
        run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "resume"),
    )
    assert result["optimizer_steps"] == 5 and result["microbatches"] == 10
    for expected, actual in zip(
        complete.model.arti.parameters(), resumed.model.arti.parameters(), strict=True
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert complete.train_data.state_dict() == resumed.train_data.state_dict()
    assert complete.session.loss_history == resumed.session.loss_history


def test_portable_qwen_bundle_reloads_tokenizer_and_generation(tmp_path):
    current = recipe(tmp_path)
    result = current.fit(run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "run"))
    asset = PathLike(result["model"]).parent
    predictor = CausalLMPredictor.from_pretrained(asset, model=base_model())
    test = current.data.splits["test"]
    batch = test.batches(2)[0]
    with torch.no_grad():
        torch.testing.assert_close(
            predictor.model(**batch).logits, current.model(**batch).logits, atol=0, rtol=0
        )
    assert tokenizer_fingerprint(predictor.tokenizer) == tokenizer_fingerprint(current.tokenizer)
    requests = [{"id": item.id, "messages": list(item.messages[:-1])} for item in test.examples]
    expected = predictor.predict_records(requests)
    current.config.data_path.unlink()
    shutil.copytree(asset, tmp_path / "moved")
    moved = CausalLMPredictor.from_pretrained(tmp_path / "moved", model=base_model())
    assert moved.predict_records(requests) == expected
    assert moved.predict_records([]) == []
    assert all(result["finish_reason"] in {"stop", "length"} for result in expected)
    assert moved.evaluate_records(records()[-2:])["samples"] == 2


def test_custom_factory_is_explicit_and_reload_does_not_guess_a_default(tmp_path, monkeypatch):
    import arti.language_recipe as module

    factory = arti.formula_adapter_factory(arti.FormulaAdapterConfig(rank=3, activation="gelu"))
    current = CausalLMRecipe(
        config(tmp_path), model=base_model(), tokenizer=tokenizer(), layer_factory=factory
    )
    assert current.task_contract["adapter_factory"] == "caller-supplied"
    assert current.plan()["deployment"]["state_scope"] == "caller-managed"
    artifact = current.export(tmp_path / "custom")
    declaration = json.loads((artifact.parent / "deployment.json").read_text())
    assert declaration["adapter_factory"] == "caller-supplied"
    assert declaration["state_scope"] == "caller-managed"

    def unexpected_host_load(*args, **kwargs):
        raise AssertionError("missing factory must be reported before loading host weights")

    monkeypatch.setattr(module, "_load_model", unexpected_host_load)
    with pytest.raises(ValueError, match="caller-supplied layer_factory"):
        CausalLMPredictor.from_pretrained(artifact.parent)
    restored = CausalLMPredictor.from_pretrained(
        artifact.parent, model=base_model(), layer_factory=factory
    )
    batch = current.data.splits["test"].batches(2)[0]
    with torch.no_grad():
        torch.testing.assert_close(
            restored.model(**batch).logits, current.model(**batch).logits, atol=0, rtol=0
        )


def test_full_task_eval_is_token_weighted_and_generation_uses_no_gold_answer(tmp_path, monkeypatch):
    current = recipe(tmp_path)
    split = current.data.splits["validation"]
    native = current.model(
        **collate_chat(
            split.examples, pad_token_id=current.tokenizer.pad_token_id, pad_to_multiple_of=8
        )
    ).loss
    observed = []
    import arti.language_recipe as module

    generation = module._generate_chat

    def capture(model, tok, conversations, kwargs, config):
        observed.extend(conversations)
        return generation(model, tok, conversations, kwargs, config)

    monkeypatch.setattr(module, "_generate_chat", capture)
    result = current.evaluate("validation", generate_samples=2, output=tmp_path / "eval.json")
    assert result["loss"] == pytest.approx(float(native.detach()), rel=1e-6)
    assert all(messages[-1]["role"] == "user" for messages in observed)
    assert len(observed) == 2
    assert current.model.training
    assert len((tmp_path / "eval.predictions.jsonl").read_text().splitlines()) == 3


def test_template_or_data_change_cannot_silently_resume(tmp_path):
    current = recipe(tmp_path)
    result = current.fit(steps=1, run=arti.ARTITrainingRunConfig(output_dir=tmp_path / "run"))
    changed = CausalLMRecipe(
        replace(current.config, template_kwargs={"enable_thinking": True}),
        model=base_model(),
        tokenizer=tokenizer(),
    )
    with pytest.raises(ValueError, match="contract"):
        changed.session.load_checkpoint(result["checkpoint"])
    changed = CausalLMRecipe(
        replace(current.config, batch_size=1), model=base_model(), tokenizer=tokenizer()
    )
    with pytest.raises(ValueError, match="data/order"):
        changed.session.load_checkpoint(result["checkpoint"])


def test_compiled_formula_profile_preserves_gradients(tmp_path):
    current = recipe(tmp_path)
    layer = next(iter(current.model.arti.layers.values()))
    values = torch.randn(2, 3, 16)
    compiled = torch.compile(layer, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(values), layer(values))
    compiled(values).square().mean().backward()
    assert all(parameter.grad is not None for parameter in layer.parameters())


def test_cli_uses_the_same_sft_train_resume_eval_export_and_predict(tmp_path, monkeypatch, capsys):
    import arti.language_recipe as module

    native_tokenizer_loader = module._load_tokenizer

    def token_loader(source, **kwargs):
        return (
            native_tokenizer_loader(source, **kwargs) if PathLike(source).exists() else tokenizer()
        )

    monkeypatch.setattr(module, "_load_tokenizer", token_loader)
    monkeypatch.setattr(module, "_load_model", lambda *args, **kwargs: base_model())
    from transformers import AutoConfig

    monkeypatch.setattr(AutoConfig, "from_pretrained", lambda *args, **kwargs: base_model().config)
    cfg = config(tmp_path)
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(cfg.to_dict()))
    assert main(["recipe", "plan", str(path)]) == 0
    planned = json.loads(capsys.readouterr().out)
    assert planned["task_recipe"]["task"] == "causal-lm"
    assert planned["model_allocation"] == "meta" and not planned["weights_loaded"]
    assert (
        main(["recipe", "train", str(path), "--run-dir", str(tmp_path / "first"), "--steps", "2"])
        == 0
    )
    first = json.loads(capsys.readouterr().out)
    assert (
        main(
            [
                "recipe",
                "train",
                str(path),
                "--run-dir",
                str(tmp_path / "resumed"),
                "--steps",
                "1",
                "--resume-from-checkpoint",
                first["checkpoint"],
            ]
        )
        == 0
    )
    second = json.loads(capsys.readouterr().out)
    assert second["optimizer_steps"] == 3
    assert (
        main(
            [
                "recipe",
                "evaluate",
                str(path),
                "--model",
                second["model"],
                "--split",
                "test",
                "--generate-samples",
                "1",
                "--output",
                str(tmp_path / "eval.json"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["generated_samples"] == 1
    assert (
        main(
            [
                "recipe",
                "export",
                str(path),
                "--checkpoint",
                second["checkpoint"],
                "--output",
                str(tmp_path / "deployed"),
            ]
        )
        == 0
    )
    capsys.readouterr()
    requests = write(
        tmp_path / "requests.jsonl",
        [{"id": "query", "messages": [{"role": "user", "content": "color red"}]}],
    )
    assert (
        main(
            [
                "recipe",
                "predict",
                str(tmp_path / "deployed"),
                "--data",
                str(requests),
                "--output",
                str(tmp_path / "predictions.jsonl"),
                "--max-new-tokens",
                "2",
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["samples"] == 1
    assert (
        main(
            [
                "recipe",
                "evaluate",
                str(tmp_path / "deployed"),
                "--data",
                str(cfg.data_path),
                "--output",
                str(tmp_path / "new-data.json"),
            ]
        )
        == 0
    )
    assert json.loads(capsys.readouterr().out)["supervised_tokens"] > 0
