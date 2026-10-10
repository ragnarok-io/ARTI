"""Pinned GSM8K final-answer SFT with Qwen, resume and portable deployment.

Preparation fetches public, human-annotated data; it performs no training.
Run preparation and the bounded GPU workflow as separate explicit commands.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from decimal import Decimal
import gc
import hashlib
import json
from pathlib import Path
import random
import time
from urllib.request import Request, urlopen

import torch

import arti
from arti.recipe_data import read_jsonl
from arti.recipe_workflow import write_json, write_jsonl


DATA_REVISION = "3101c7d5072418e28b9008a6636bde82a006892c"
MODEL_REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
DATA_ROOT = f"https://raw.githubusercontent.com/openai/grade-school-math/{DATA_REVISION}/grade_school_math/data"


def _download(path, url):
    if not path.exists():
        request = Request(url, headers={"User-Agent": "ARTI-task-recipe"})
        with urlopen(request, timeout=30) as response:
            contents = response.read()
        path.write_bytes(contents)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def final_answer(answer):
    prefix, separator, value = answer.rpartition("####")
    if not prefix or not separator:
        raise ValueError("GSM8K record has no human-annotated final-answer delimiter")
    number = Decimal(value.strip().replace(",", ""))
    if not number.is_finite():
        raise ValueError("GSM8K final answer must be finite")
    return format(number.normalize(), "f")


def prepare(output, *, train_samples=256, validation_samples=32, test_samples=32, seed=17):
    output = Path(output)
    source = output / "source"
    source.mkdir(parents=True, exist_ok=True)
    digests = {}
    for split in ("train", "test"):
        digests[split] = _download(source / f"{split}.jsonl", f"{DATA_ROOT}/{split}.jsonl")
    original_train = list(read_jsonl(source / "train.jsonl"))
    original_test = list(read_jsonl(source / "test.jsonl"))
    order = list(range(len(original_train)))
    random.Random(seed).shuffle(order)
    if train_samples + validation_samples > len(order) or test_samples > len(original_test):
        raise ValueError("requested source split sizes exceed GSM8K")
    groups = [
        ("train", "train", order[:train_samples], original_train),
        (
            "validation",
            "train",
            order[train_samples : train_samples + validation_samples],
            original_train,
        ),
        (
            "test",
            "test",
            random.Random(seed + 1).sample(range(len(original_test)), test_samples),
            original_test,
        ),
    ]
    records = []
    for role, original_split, indices, rows in groups:
        for index in indices:
            records.append(
                {
                    "id": f"gsm8k:{original_split}:{index:05d}",
                    "split": role,
                    "messages": [
                        {
                            "role": "system",
                            "content": "Solve the problem. Return only the final number, without explanation.",
                        },
                        {"role": "user", "content": rows[index]["question"]},
                        {"role": "assistant", "content": final_answer(rows[index]["answer"])},
                    ],
                }
            )
    write_jsonl(output / "gsm8k.jsonl", records)
    write_jsonl(
        output / "requests.jsonl",
        (
            {"id": row["id"], "messages": row["messages"][:-1]}
            for row in records
            if row["split"] == "test"
        ),
    )
    write_json(
        output / "source.json",
        {
            "dataset": "openai/grade-school-math",
            "revision": DATA_REVISION,
            "source_sha256": digests,
            "license": "MIT",
            "seed": seed,
            "targets": "human final numeric answer after ####; CoT/calculation annotations excluded",
            "validation": "held out from original train rows",
            "test": "original test rows",
            "source_records": {"train": len(original_train), "test": len(original_test)},
        },
    )
    config = {
        "task": "causal-lm",
        "seed": seed,
        "model": {
            "id": "Qwen/Qwen3-0.6B",
            "revision": MODEL_REVISION,
            "dtype": "bfloat16",
            "local_files_only": True,
            "attn_implementation": "sdpa",
        },
        "data": {
            "path": "gsm8k.jsonl",
            "batch_size": 16,
            "max_length": 256,
            "overlength": "drop",
            "pad_to_multiple_of": 8,
            "group_by_length": True,
            "template_kwargs": {"enable_thinking": False},
        },
        "adapter": {"rank": 32, "activation": "silu", "last_layers": 2},
        "training": {
            "device": "cuda",
            "steps": 200,
            "learning_rate": 0.001,
            "mixed_precision": "bf16",
            "gradient_accumulation_steps": 1,
        },
        "optimizer": {"weight_decay": 0.01, "scheduler": "constant", "implementation": "auto"},
        "generation": {"max_new_tokens": 32, "batch_size": 8},
        "run": {
            "output_dir": "train",
            "checkpoint_steps": 25,
            "checkpoint_seconds": 30,
            "evaluation_steps": 50,
            "log_steps": 10,
        },
    }
    write_json(output / "recipe.json", config)
    return output / "recipe.json"


def run(output, *, max_seconds=300):
    output = Path(output)
    started = time.monotonic()
    deadline = started + max_seconds
    torch.cuda.reset_peak_memory_stats()
    current = arti.CausalLMRecipe.from_config(output / "recipe.json")
    write_json(output / "plan.json", current.plan())
    preflight = current.check()
    write_json(output / "preflight.json", preflight)
    if not preflight["ok"]:
        raise RuntimeError("real Qwen objective is not connected to trainable ARTI parameters")
    before = current.evaluate("validation", output=output / "before-validation.json")
    available = deadline - time.monotonic() - 90
    if available <= 0:
        raise RuntimeError("model/preflight consumed the bounded job; no new training started")
    first = current.fit(
        run=replace(
            current.config.run, output_dir=output / "train", max_seconds=min(150, available)
        )
    )
    reference = {
        key: value.detach().cpu().clone()
        for key, value in current.model.arti._bundle().state_dict().items()
    }
    first_steps = first["optimizer_steps"]
    del current
    gc.collect()
    torch.cuda.empty_cache()
    continued = arti.CausalLMRecipe.from_config(output / "recipe.json")
    available = deadline - time.monotonic() - 45
    if available <= 0:
        raise RuntimeError(
            "no time remains for a new resumed update; completed first-stage assets are retained"
        )
    second = continued.fit(
        steps=40,
        resume_from_checkpoint=first["checkpoint"],
        run=replace(continued.config.run, output_dir=output / "resume", max_seconds=available),
    )
    continued.model.arti.disable()
    try:
        before_generation = continued.evaluate(
            "test", generate_samples=16, output=output / "base-test.json"
        )
    finally:
        continued.model.arti.enable()
    test = continued.evaluate("test", generate_samples=16, output=output / "test.json")
    probe = continued.data.splits["test"].batches(1)[0]
    with torch.inference_mode():
        inputs = {key: value.to("cuda") for key, value in probe.items()}
        expected = continued.model(**inputs, use_cache=False).logits.detach().cpu()
    state_changed_after_resume = any(
        not torch.equal(value.detach().cpu(), reference[key])
        for key, value in continued.model.arti._bundle().state_dict().items()
    )
    del continued
    gc.collect()
    torch.cuda.empty_cache()
    predictor = arti.CausalLMPredictor.from_pretrained(
        second["model"], device="cuda", local_files_only=True
    )
    with torch.inference_mode():
        actual = predictor.model(**inputs, use_cache=False).logits.cpu()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    requests = list(read_jsonl(output / "requests.jsonl"))[:8]
    predictions = predictor.predict_records(requests)
    write_jsonl(output / "deployment-predictions.jsonl", predictions)
    report = {
        "before_validation": before,
        "before_test": before_generation,
        "validation": second["validation"],
        "test": test,
        "optimizer_steps": second["optimizer_steps"],
        "first_stage_steps": first_steps,
        "resume_steps": second["optimizer_steps"] - first_steps,
        "resume_changed_parameters": state_changed_after_resume,
        "model": second["model"],
        "checkpoint": second["checkpoint"],
        "reload_logits_max_abs_error": float((actual - expected).abs().max()),
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "max_seconds": max_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "scope": "bounded real supervised task workflow, not a full GSM8K benchmark or architecture-superiority claim",
    }
    write_json(output / "workflow_result.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(".tmp/qwen-sft-workflow"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--train-samples", type=int, default=256)
    parser.add_argument("--validation-samples", type=int, default=32)
    parser.add_argument("--test-samples", type=int, default=32)
    parser.add_argument("--max-seconds", type=float, default=300)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(
        json.dumps(
            {
                "config": str(
                    prepare(
                        args.output,
                        train_samples=args.train_samples,
                        validation_samples=args.validation_samples,
                        test_samples=args.test_samples,
                    )
                )
            }
            if args.prepare_only
            else run(args.output, max_seconds=args.max_seconds),
            indent=2,
        )
    )
