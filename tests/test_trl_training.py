from __future__ import annotations

import copy

import pytest
import torch

import arti

trl = pytest.importorskip("trl")
datasets = pytest.importorskip("datasets")


def fixture():
    from test_language_recipe import base_model, tokenizer
    from test_attachment_training_engines import trainable_layer

    model = arti.ARTI.attach(base_model(), trainable_layer(), layers=("model.layers.1.mlp",))
    session = model.arti.trainer(engine="transformers", device="cpu", learning_rate=0.02)
    return session, tokenizer()


def arguments(path, **kwargs):
    return trl.SFTConfig(
        output_dir=str(path),
        use_cpu=True,
        bf16=False,
        fp16=False,
        gradient_checkpointing=False,
        max_steps=2,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        logging_steps=1,
        save_steps=1,
        max_length=64,
        packing=False,
        completion_only_loss=True,
        report_to=[],
        disable_tqdm=True,
        dataloader_pin_memory=False,
        **kwargs,
    )


def examples():
    return datasets.Dataset.from_list(
        [
            {
                "prompt": [
                    {"role": "user", "content": "color red"},
                    {"role": "assistant", "content": "old"},
                    {"role": "user", "content": "color blue"},
                ],
                "completion": [{"role": "assistant", "content": "blue"}],
            },
            {
                "prompt": [{"role": "user", "content": "color green"}],
                "completion": [{"role": "assistant", "content": "green"}],
            },
        ]
    )


def test_native_sft_preprocessing_loss_checkpoint_and_resume(tmp_path):
    from test_transformers_training import assert_tree_equal

    session, tokenizer = fixture()
    frozen = [(p, p.detach().clone()) for p in session.model.parameters() if not p.requires_grad]
    initial = copy.deepcopy(session.binding.state_module().state_dict())
    trainer = session.transformers_trainer(
        examples(),
        trainer_class=trl.SFTTrainer,
        trainer_kwargs={
            "args": arguments(tmp_path / "sft"),
            "processing_class": tokenizer,
            "eval_dataset": examples(),
        },
    )
    assert isinstance(trainer, trl.SFTTrainer)
    assert trainer.args.completion_only_loss is True
    encoded = trainer.train_dataset[0]
    assert tokenizer.convert_tokens_to_ids("old") in encoded["input_ids"]
    batch = trainer.data_collator([encoded])
    supervised = batch["input_ids"][batch["labels"] != -100].tolist()
    assert supervised == [tokenizer.convert_tokens_to_ids("blue"), tokenizer.eos_token_id]
    trainer.train()
    assert session.global_step == 2 and session.micro_batches_seen == 4
    assert any(
        not torch.equal(initial[k], v)
        for k, v in session.binding.state_module().state_dict().items()
    )
    for parameter, before in frozen:
        assert torch.equal(parameter, before)
    evaluation = trainer.evaluate()
    assert torch.isfinite(torch.tensor(evaluation["eval_loss"]))
    checkpoint = tmp_path / "sft" / "checkpoint-2"
    assert (checkpoint / "tokenizer.json").exists()
    assert not (checkpoint / "model.safetensors").exists()
    resumed, _ = fixture()
    resumed.resume_from_checkpoint = checkpoint
    outcome = resumed.fit(
        examples(),
        steps=1,
        trainer_kwargs={
            "trainer_class": trl.SFTTrainer,
            "args": arguments(tmp_path / "resumed"),
            "processing_class": tokenizer,
        },
    )
    assert outcome.steps == 3 and resumed.micro_batches_seen == 6
    trainer.save_model(str(tmp_path / "export"))
    reloaded, _ = fixture()
    reloaded.binding.load(tmp_path / "export" / "model.arti.st", load_checkpoint=False)
    assert_tree_equal(
        reloaded.binding.state_module().state_dict(), session.binding.state_module().state_dict()
    )


def test_sft_pretokenized_labels_are_not_replaced_by_language_modeling_targets(tmp_path):
    session, tokenizer = fixture()
    data = datasets.Dataset.from_list(
        [
            {
                "input_ids": [4, 7, 8, 6, 8, 3],
                "attention_mask": [1] * 6,
                "labels": [-100, -100, -100, -100, 8, 3],
            },
        ]
    )
    trainer = session.transformers_trainer(
        data,
        trainer_class=trl.SFTTrainer,
        trainer_kwargs={
            "args": arguments(tmp_path / "labels"),
            "processing_class": tokenizer,
        },
    )
    batch = trainer.data_collator([trainer.train_dataset[0]])
    assert batch["labels"].tolist() == [[-100, -100, -100, -100, 8, 3]]
    trainer.train()
    assert session.global_step == 2
