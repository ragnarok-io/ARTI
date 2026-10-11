"""Offline CPU Trainer/SFTTrainer continuation and ARTI asset example."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import Qwen3Config, Qwen3ForCausalLM, TrainerCallback, TrainingArguments

import arti


class StopAtTwo(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == 2:
            control.should_training_stop = True
            control.should_save = True
        return control


def make_session(resume=None):
    torch.manual_seed(91)
    config = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=64,
        pad_token_id=0,
        eos_token_id=2,
    )
    model = Qwen3ForCausalLM(config)
    layer = arti.ARTILayer(
        arti.mechanisms.AdaptivePulse(half=arti.Half(stochastic=False, learnable=True))
    )
    arti.ARTI.attach(model, layer, layers=("model.layers.0.mlp",))
    return model.arti.trainer(
        engine="transformers", device="cpu", learning_rate=0.02, resume_from_checkpoint=resume
    )


def trainer_options(path, kind):
    values = dict(
        output_dir=str(path),
        max_steps=6,
        use_cpu=True,
        bf16=False,
        learning_rate=0.02,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=2,
        save_steps=1,
        logging_steps=1,
        dataloader_pin_memory=False,
        disable_tqdm=True,
        report_to=[],
    )
    if kind == "transformers":
        return {"args": TrainingArguments(**values)}
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import WhitespaceSplit
    from transformers import PreTrainedTokenizerFast
    from trl import SFTConfig, SFTTrainer

    backend = Tokenizer(WordLevel({f"w{i}": i for i in range(32)}, unk_token="w3"))
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, pad_token="w0", eos_token="w2", unk_token="w3"
    )
    return {
        "trainer_class": SFTTrainer,
        "processing_class": tokenizer,
        "args": SFTConfig(
            **values,
            gradient_checkpointing=False,
            max_length=64,
            completion_only_loss=True,
            packing=False,
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".tmp/transformers-workflow"))
    parser.add_argument("--trainer", choices=("transformers", "trl"), default="transformers")
    args = parser.parse_args()
    torch.set_num_threads(1)
    rows = [
        {
            "input_ids": [1, i + 4, i + 5, 2],
            "attention_mask": [1] * 4,
            "labels": [-100, -100, i + 5, 2],
        }
        for i in range(12)
    ]
    if args.trainer == "trl":
        from datasets import Dataset

        data = Dataset.from_list(rows)
    else:
        data = rows
    first = make_session()
    initial = [p.detach().clone() for p in first.binding.parameters()]
    frozen = [(p, p.detach().clone()) for p in first.model.parameters() if not p.requires_grad]
    options = trainer_options(args.output / "first", args.trainer)
    options["callbacks"] = [StopAtTwo()]
    first.fit(data, steps=6, trainer_kwargs=options)
    checkpoint = args.output / "first" / "checkpoint-2"
    session = make_session(checkpoint)
    final = session.fit(
        data,
        steps=4,
        checkpoint_path=args.output / "training.arti.st",
        trainer_kwargs=trainer_options(args.output / "continued", args.trainer),
    )
    session.engine_object.save_model(str(args.output / "inference"))
    reloaded = make_session()
    reloaded.binding.load(args.output / "inference" / "model.arti.st", load_checkpoint=False)
    inputs = {"input_ids": torch.tensor([[1, 4, 5, 2]])}
    session.model.eval()
    reloaded.model.eval()
    with torch.no_grad():
        error = float(
            (session.model(**inputs).logits - reloaded.model(**inputs).logits).abs().max()
        )
    changed = any(
        not torch.equal(before, after)
        for before, after in zip(initial, session.binding.parameters())
    )
    backbone_unchanged = all(torch.equal(parameter, before) for parameter, before in frozen)
    assert final.steps == 6 and session.micro_batches_seen == 12
    assert changed and backbone_unchanged and error == 0
    assert not (checkpoint / "model.safetensors").exists()
    receipt = {
        "trainer": args.trainer,
        "profile": "offline_cpu_fixture",
        "global_step": final.steps,
        "micro_batches_seen": session.micro_batches_seen,
        "arti_parameters_changed": changed,
        "frozen_backbone_unchanged": backbone_unchanged,
        "reload_max_error": error,
        "native_resume": "first/checkpoint-2",
        "portable_checkpoint": "training.arti.st",
        "inference_asset": "inference/model.arti.st",
    }
    (args.output / "result.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
