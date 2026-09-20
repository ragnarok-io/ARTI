from __future__ import annotations

import json
from pathlib import Path

import torch

from benchmarks.generate_qwen_tensor_reasoning_teacher import (
    FrozenReasoningExample,
    ReasoningTeacherCorpus,
    build_reasoning_prompts,
    load_generation_prefix,
    load_reasoning_prompts,
    save_corpus,
)
from benchmarks.run_qwen_tensor_workspace_reasoning import (
    build_minibatch_schedule,
    build_reasoning_answer_batch,
    build_reasoning_prompt_batch,
)


class FakeTokenizer:
    eos_token_id = 99

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking,
        return_tensors,
        return_dict,
    ):
        assert tokenize and add_generation_prompt and return_tensors == "pt"
        assert return_dict and enable_thinking is False
        assert messages == [{"role": "user", "content": "same complete question"}]
        return {"input_ids": torch.tensor([[10, 11, 12]])}


def _example() -> FrozenReasoningExample:
    return FrozenReasoningExample(
        "reasoning-1",
        "same complete question",
        "final answer",
        torch.tensor([20, 21, 99]),
        torch.tensor([[20, 30], [21, 30], [99, 30]]),
        torch.tensor([[4.0, 1.0], [5.0, 1.0], [6.0, 1.0]]),
    )


def test_reasoning_teacher_artifact_persists_answer_but_not_cot(tmp_path: Path) -> None:
    root = tmp_path / "teacher"
    save_corpus(
        root,
        [_example()],
        model_id="Qwen/Qwen3-0.6B",
        model_revision="test",
        topk=2,
    )

    corpus = ReasoningTeacherCorpus(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    rows = (root / "examples.jsonl").read_text(encoding="utf-8")

    assert manifest["teacher_thinking_enabled"] is True
    assert manifest["student_exposed_cot"] is False
    assert manifest["cot_payload_persisted"] is False
    assert manifest["teacher_generation_limit"] == "model_context_window_only"
    assert manifest["generation_complete"] is True
    assert "<think>" not in rows and "</think>" not in rows
    assert corpus.example(0).answer == "final answer"
    assert corpus.example(0).answer_ids.tolist() == [20, 21, 99]


def test_reasoning_teacher_generation_resumes_only_matching_prefix(
    tmp_path: Path,
) -> None:
    root = tmp_path / "teacher"
    save_corpus(
        root,
        [_example()],
        model_id="Qwen/Qwen3-0.6B",
        model_revision="test",
        topk=2,
        requested_examples=2,
    )

    rows = load_generation_prefix(
        root,
        (("reasoning-1", "same complete question"), ("next", "Next question")),
        model_id="Qwen/Qwen3-0.6B",
        topk=2,
    )
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))

    assert len(rows) == 1
    assert manifest["generation_complete"] is False
    assert manifest["requested_examples"] == 2


def test_reasoning_teacher_generation_rejects_mismatched_resume(
    tmp_path: Path,
) -> None:
    root = tmp_path / "teacher"
    save_corpus(
        root,
        [_example()],
        model_id="Qwen/Qwen3-0.6B",
        model_revision="test",
        topk=2,
    )

    try:
        load_generation_prefix(
            root,
            (("different", "same complete question"),),
            model_id="Qwen/Qwen3-0.6B",
            topk=2,
        )
    except ValueError as error:
        assert "not a prefix" in str(error)
    else:
        raise AssertionError("mismatched prompt suite must fail closed")


def test_reasoning_prompts_do_not_constrain_teacher_cot_length() -> None:
    prompts = "\n".join(prompt for _, prompt in build_reasoning_prompts()).lower()

    assert "reasoning step" not in prompts
    assert "think in" not in prompts


def test_extended_reasoning_prompt_suite_is_answer_free_and_unique() -> None:
    prompts = load_reasoning_prompts(
        Path("benchmarks/data/qwen_reasoning_prompts_v2.jsonl")
    )

    assert len(prompts) == 32
    assert len({example_id for example_id, _ in prompts}) == len(prompts)


def test_reasoning_prompt_file_is_strict_and_preserves_order(tmp_path: Path) -> None:
    path = tmp_path / "prompts.jsonl"
    path.write_text(
        '{"example_id":"first","prompt":"First question?"}\n'
        '{"example_id":"second","prompt":"Second question?"}\n',
        encoding="utf-8",
    )

    assert load_reasoning_prompts(path) == (
        ("first", "First question?"),
        ("second", "Second question?"),
    )


def test_reasoning_prompt_file_rejects_duplicates_and_extra_fields(
    tmp_path: Path,
) -> None:
    duplicate = tmp_path / "duplicate.jsonl"
    duplicate.write_text(
        '{"example_id":"same","prompt":"One"}\n'
        '{"example_id":"same","prompt":"Two"}\n',
        encoding="utf-8",
    )
    extra = tmp_path / "extra.jsonl"
    extra.write_text(
        '{"example_id":"one","prompt":"Question","answer":"leak"}\n',
        encoding="utf-8",
    )

    try:
        load_reasoning_prompts(duplicate)
    except ValueError as error:
        assert "duplicate" in str(error)
    else:
        raise AssertionError("duplicate prompt ids must fail")
    try:
        load_reasoning_prompts(extra)
    except ValueError as error:
        assert "only example_id and prompt" in str(error)
    else:
        raise AssertionError("authored answer fields must fail")


def test_reasoning_minibatches_are_deterministic_diverse_and_cover_examples() -> None:
    first = build_minibatch_schedule(11, 4, 12, seed=17)
    second = build_minibatch_schedule(11, 4, 12, seed=17)

    assert first == second
    assert all(len(batch) == len(set(batch)) == 4 for batch in first)
    assert set(index for batch in first[:3] for index in batch) == set(range(11))
    assert len(set(first)) > 3


def test_student_prompt_is_identical_across_reasoning_and_answer_passes() -> None:
    tokenizer = FakeTokenizer()
    example = _example()
    prompt = build_reasoning_prompt_batch(
        tokenizer,
        [example],
        canvas_tokens=8,
        placeholder_token_id=0,
    )
    answer = build_reasoning_answer_batch(
        tokenizer,
        [example],
        canvas_tokens=8,
        placeholder_token_id=0,
    )

    assert prompt.input_ids[0, :3].tolist() == [10, 11, 12]
    assert answer.input_ids[0, :3].tolist() == [10, 11, 12]
    assert answer.input_ids[0, 3:5].tolist() == [20, 21]
    assert answer.target_ids is not None
    assert answer.target_ids[answer.target_mask].tolist() == [20, 21, 99]
