from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from benchmarks._qwen_self_teacher_data import SelfTeacherRolloutCorpus
from benchmarks.generate_qwen_self_teacher_rollouts import (
    FIXED_RECORD_QUERIES,
    FORMAT,
    build_dialogue_plans,
    build_fixed_record_query_plans,
    encode_hard_target_turn,
    fixed_record_assistant_target,
    generate_assistant_turn,
    load_dialogue_plans,
    write_plan_only,
)
from benchmarks.run_qwen_self_teacher_affine_ttt import (
    AdaptiveControllerScale,
    RolloutTurnStream,
    collate_teacher_turns,
    resolve_batch_size,
    select_recommended_multiplier,
    teacher_distribution_loss,
)
from benchmarks.run_qwen_dialogue_recall_transition import dialogue_source_metadata


def test_builtin_plans_are_deterministic_diverse_and_multiturn() -> None:
    first = build_dialogue_plans(16, turns=4, seed=77)
    second = build_dialogue_plans(16, turns=4, seed=77)

    assert first == second
    assert len({plan.scenario for plan in first}) == 8
    assert all(len(plan.user_turns) == 4 for plan in first)
    assert all(plan.user_turns[-1] != plan.user_turns[0] for plan in first)
    assert any(any(ord(char) > 127 for char in turn) for plan in first for turn in plan.user_turns)


def test_fixed_record_query_plans_hold_query_form_constant_and_resample_values() -> None:
    first = build_fixed_record_query_plans(32, turns=4, seed=77)
    second = build_fixed_record_query_plans(32, turns=4, seed=77)

    assert first == second
    assert all(plan.scenario == "fixed-record-query" for plan in first)
    assert all(plan.user_turns[1:] == FIXED_RECORD_QUERIES for plan in first)
    writes = [plan.user_turns[0] for plan in first]
    assert len(set(writes)) == len(writes)
    for field in ("CODE", "OWNER", "VERSION"):
        values = [write.split(f"{field}=", 1)[1].split(";", 1)[0].rstrip(".") for write in writes]
        assert len(set(values)) == len(values)


def test_fixed_record_query_turn_limit_preserves_write_then_fixed_queries() -> None:
    plans = build_fixed_record_query_plans(3, turns=2, seed=19)

    assert all(len(plan.user_turns) == 2 for plan in plans)
    assert all(plan.user_turns[1] == FIXED_RECORD_QUERIES[0] for plan in plans)
    with pytest.raises(ValueError, match="turns"):
        build_fixed_record_query_plans(3, turns=5, seed=19)


def test_fixed_record_targets_are_derived_from_each_record_not_an_answer_table() -> None:
    first, second = build_fixed_record_query_plans(2, turns=4, seed=91)

    assert fixed_record_assistant_target(first, 0) == "Stored."
    first_targets = tuple(fixed_record_assistant_target(first, index) for index in range(1, 4))
    second_targets = tuple(fixed_record_assistant_target(second, index) for index in range(1, 4))
    assert first_targets != second_targets
    assert all(target in first.user_turns[0] for target in first_targets)
    assert all(target in second.user_turns[0] for target in second_targets)


def test_hard_target_encoding_needs_no_teacher_logits() -> None:
    class Tokenizer:
        def apply_chat_template(self, *args: object, **kwargs: object) -> torch.Tensor:
            if kwargs["add_generation_prompt"]:
                return torch.tensor([[3, 4, 5]])
            return torch.tensor([[3, 4, 5, 6, 7]])

    text, prompt, response, topk_ids, topk_logits = encode_hard_target_turn(
        Tokenizer(),
        [{"role": "user", "content": "query"}],
        "answer",
    )

    assert text == "answer"
    assert prompt.tolist() == [3, 4, 5]
    assert response.tolist() == [6, 7]
    assert topk_ids.tolist() == [[6], [7]]
    assert torch.equal(topk_logits, torch.zeros(2, 1, dtype=torch.float16))


def test_external_plans_are_validated_without_answer_data(tmp_path: Path) -> None:
    path = tmp_path / "plans.jsonl"
    path.write_text(
        json.dumps(
            {
                "plan_id": "custom-1",
                "scenario": "custom",
                "user_turns": ["First natural question", "Follow it up"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    plans = load_dialogue_plans(path)
    assert plans[0].user_turns == ("First natural question", "Follow it up")

    path.write_text('{"user_turns": ["only one"]}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="at least two"):
        load_dialogue_plans(path)


def test_plan_only_artifact_contains_no_assistant_style_targets(tmp_path: Path) -> None:
    plans = build_dialogue_plans(3, turns=3, seed=19)
    path = tmp_path / "plans.jsonl"
    write_plan_only(path, plans)
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    assert len(rows) == 3
    assert all("user_turns" in row for row in rows)
    assert all("messages" not in row and "assistant" not in row for row in rows)


def test_self_teacher_dialogue_metadata_replaces_ultrachat_label(tmp_path: Path) -> None:
    dialogues = tmp_path / "dialogues.jsonl"
    dialogues.write_text("", encoding="utf-8")
    dialogues.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "dataset": "frozen Qwen self-teacher rollouts",
                "license": "generated research artifact",
                "source": "local frozen base model",
                "format": FORMAT,
            }
        ),
        encoding="utf-8",
    )
    metadata = dialogue_source_metadata(dialogues)
    assert metadata["dataset"] == "frozen Qwen self-teacher rollouts"
    assert metadata["source"] == "local frozen base model"


def test_rollout_source_has_no_ultrachat_dependency() -> None:
    source = Path("benchmarks/generate_qwen_self_teacher_rollouts.py").read_text(
        encoding="utf-8"
    )
    assert "ultrachat" not in source.casefold()
    assert "teacher_topk_logits" in source
    assert "assistant_responses" in source


def test_self_teacher_training_disallows_same_answer_train_eval_shortcut() -> None:
    source = Path("benchmarks/run_qwen_self_teacher_affine_ttt.py").read_text(
        encoding="utf-8"
    )

    assert "overfit-indices" not in source
    assert "unseen_record_values_fixed_queries" in source
    assert "teacher_forced_sequence_accuracy" in source


class _FakeTokenizer:
    eos_token_id = 2

    def __init__(self, *, mapping: bool) -> None:
        self.mapping = mapping

    def apply_chat_template(self, *args: object, **kwargs: object) -> object:
        input_ids = torch.tensor([[3, 4, 5]])
        if self.mapping:
            return {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
            }
        return input_ids

    def decode(self, ids: torch.Tensor, **kwargs: object) -> str:
        return "answer"


class _FakeModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def generate(self, *, input_ids: torch.Tensor, **kwargs: object) -> object:
        suffix = torch.tensor([[6, 7]], device=input_ids.device)
        return SimpleNamespace(sequences=torch.cat((input_ids, suffix), dim=1))

    def forward(self, *, input_ids: torch.Tensor, **kwargs: object) -> object:
        positions = input_ids.shape[1]
        logits = torch.arange(positions * 16, dtype=torch.float32).reshape(
            1, positions, 16
        )
        return SimpleNamespace(logits=logits)


@pytest.mark.parametrize("mapping", [False, True])
def test_generate_assistant_turn_accepts_tensor_or_mapping_template_output(
    mapping: bool,
) -> None:
    text, prompt, ids, topk_ids, topk_logits = generate_assistant_turn(
        _FakeModel(),
        _FakeTokenizer(mapping=mapping),
        [{"role": "user", "content": "question"}],
        max_new_tokens=2,
        topk=4,
        do_sample=False,
        temperature=0.7,
        top_p=0.9,
    )

    assert text == "answer"
    assert prompt.tolist() == [3, 4, 5]
    assert ids.tolist() == [6, 7]
    assert topk_ids.shape == topk_logits.shape == (2, 4)


def test_self_teacher_corpus_validates_and_slices_exact_turns(tmp_path: Path) -> None:
    dialogues = tmp_path / "dialogues.jsonl"
    dialogues.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "first question"},
                    {"role": "assistant", "content": "first answer"},
                    {"role": "user", "content": "second question"},
                    {"role": "assistant", "content": "second answer"},
                ],
                "assistant_responses": [
                    {"token_count": 2, "token_ids": [6, 7]},
                    {"token_count": 1, "token_ids": [8]},
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    traces = tmp_path / "teacher-token-traces.safetensors"
    save_file(
        {
            "prompt_token_ids": torch.tensor([1, 2, 3, 4, 5]),
            "prompt_offsets": torch.tensor([0, 2, 5]),
            "response_token_ids": torch.tensor([6, 7, 8]),
            "response_offsets": torch.tensor([0, 2, 3]),
            "dialogue_response_offsets": torch.tensor([0, 2]),
            "teacher_topk_ids": torch.tensor([[6, 9], [7, 9], [8, 9]]),
            "teacher_topk_logits": torch.tensor([[2.0, 1.0], [3.0, 1.0], [4.0, 1.0]]),
        },
        str(traces),
    )

    def digest(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "format": FORMAT,
                "vocabulary_size": 16,
                "turns": 2,
                "teacher_tokens": 3,
                "dialogues_sha256": digest(dialogues),
                "traces_sha256": digest(traces),
            }
        ),
        encoding="utf-8",
    )
    corpus = SelfTeacherRolloutCorpus(tmp_path)

    assert len(corpus) == 2
    assert corpus.turn(0).prompt_ids.tolist() == [1, 2]
    assert corpus.turn(0).response_ids.tolist() == [6, 7]
    assert corpus.turn(1).prompt_ids.tolist() == [3, 4, 5]
    assert corpus.turn(1).teacher_topk_logits.shape == (1, 2)
    context = corpus.dialogue_context(1)
    assert [message.role for message in context.history_messages] == ["user", "assistant"]
    assert context.current_user_message.content == "second question"

    batch = collate_teacher_turns(
        corpus,
        [0, 1],
        tokenizer=_FakeTokenizer(mapping=True),
        max_context_tokens=8,
        max_response_tokens=8,
        pad_token_id=0,
    )
    assert len(batch.history_rounds) == 1
    assert batch.history_rounds[0].row_indices.tolist() == [1]
    assert batch.current.input_ids[0].tolist() == [3, 4, 5, 6]
    assert batch.target_mask[0].tolist() == [False, False, True, True]
    assert batch.current.target_ids[0, 2:].tolist() == [6, 7]


def test_teacher_distribution_loss_uses_generated_token_and_teacher_shape() -> None:
    student = torch.zeros(2, 16, requires_grad=True)
    targets = torch.tensor([6, 7])
    topk_ids = torch.tensor([[6, 9], [7, 9]])
    topk_logits = torch.tensor([[3.0, 1.0], [4.0, 1.0]])
    loss, metrics = teacher_distribution_loss(
        student,
        targets,
        topk_ids,
        topk_logits,
        temperature=1.5,
        hard_weight=0.5,
    )

    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(student.grad).all()
    assert set(metrics) == {
        "hard_ce",
        "worst_token_ce",
        "teacher_cross_entropy",
        "token_accuracy",
    }


def test_rollout_stream_preserves_explicit_training_split_across_resume() -> None:
    stream = RolloutTurnStream((2, 4, 8), seed=17)
    first = stream.next(2)
    state = stream.state_dict()
    expected = stream.next(5)
    restored = RolloutTurnStream((2, 4, 8), seed=999)
    restored.load_state_dict(state)

    assert set(first).issubset({2, 4, 8})
    assert restored.next(5) == expected
    with pytest.raises(ValueError, match="split changed"):
        RolloutTurnStream((2, 4, 9), seed=17).load_state_dict(state)


def test_auto_batch_uses_parallelism_without_formal_repetition() -> None:
    assert resolve_batch_size(0, training_turns=24, allow_repeated_turns=False) == 24
    assert resolve_batch_size(0, training_turns=100, allow_repeated_turns=False) == 64
    assert resolve_batch_size(64, training_turns=24, allow_repeated_turns=True) == 64

    with pytest.raises(ValueError, match="cannot repeat"):
        resolve_batch_size(64, training_turns=24, allow_repeated_turns=False)
    with pytest.raises(ValueError, match="training-turns"):
        resolve_batch_size(0, training_turns=0, allow_repeated_turns=False)


def test_recommended_multiplier_uses_heldout_loss_not_last_optimizer_scale() -> None:
    selected = select_recommended_multiplier(
        reference_multiplier=0.015,
        operational_multiplier=0.005,
        reference_metrics={"loss": 16.7},
        operational_metrics={"loss": 18.0},
    )
    assert selected == pytest.approx(0.015)

    selected = select_recommended_multiplier(
        reference_multiplier=0.015,
        operational_multiplier=0.01,
        reference_metrics={"loss": 16.7},
        operational_metrics={"loss": 16.5},
    )
    assert selected == pytest.approx(0.01)


def test_adaptive_controller_scale_retries_and_restores() -> None:
    scale = AdaptiveControllerScale()
    scale.reject()
    scale.reject()
    state = scale.state_dict()
    scale.recover()
    restored = AdaptiveControllerScale()
    restored.load_state_dict(state)

    assert restored.value == pytest.approx(0.25)
    assert restored.rejections == 2
    assert scale.value > restored.value
