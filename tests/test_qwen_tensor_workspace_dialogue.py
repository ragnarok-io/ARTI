from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file
from torch import nn

from arti import mechanisms
from benchmarks._tensor_workspace_dialogue import WorkspaceTurnSchedule, empty_workspace_snapshot
from benchmarks.run_qwen_tensor_workspace_dialogue import (
    FrozenTeacherTarget,
    QwenWorkspaceBridge,
    _initialize_operation_library,
    _pad_sequences,
    load_natural_workspace_rollouts,
    make_workspace_loop,
)


class FakeLayer(nn.Module):
    def forward(self, value: torch.Tensor) -> tuple[torch.Tensor]:
        return (value,)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_workspace_dialogues_use_frozen_natural_teacher_outputs(tmp_path: Path) -> None:
    root = tmp_path / "rollouts"
    root.mkdir()
    dialogues_path = root / "dialogues.jsonl"
    dialogues_path.write_text(
        json.dumps(
            {
                "format": "arti.qwen-self-teacher-rollouts.v1",
                "source": "frozen-qwen-self-teacher",
                "plan_id": "natural-1",
                "scenario": "state-tracking",
                "messages": [
                    {"role": "user", "content": "A box contains three blue tokens."},
                    {"role": "assistant", "content": "The box starts with three blue tokens."},
                    {"role": "user", "content": "Add two. What is the new total?"},
                    {"role": "assistant", "content": "The new total is five."},
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
    traces_path = root / "teacher-token-traces.safetensors"
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
        str(traces_path),
    )
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "format": "arti.qwen-self-teacher-rollouts.v1",
                "model": "Qwen/Qwen3-0.6B",
                "dialogues": 1,
                "turns": 2,
                "teacher_tokens": 3,
                "vocabulary_size": 32,
                "dialogues_sha256": _sha256(dialogues_path),
                "traces_sha256": _sha256(traces_path),
            }
        ),
        encoding="utf-8",
    )

    dialogues, teacher, manifest = load_natural_workspace_rollouts(root, count=1)

    assert manifest["model"] == "Qwen/Qwen3-0.6B"
    assert dialogues[0].dialogue_id == "natural-1"
    assert dialogues[0].turns[-1].assistant == "The new total is five."
    assert teacher[0][-1].response_ids.tolist() == [8]


def test_student_batch_keeps_the_original_message_canvas() -> None:
    target = FrozenTeacherTarget(
        response_ids=torch.tensor([5, 6]),
        topk_ids=torch.tensor([[5, 7], [6, 8]]),
        topk_logits=torch.tensor([[3.0, 1.0], [4.0, 0.5]]),
    )
    batch = _pad_sequences(
        [torch.tensor([10, 11, 12])],
        canvas_tokens=8,
        placeholder_token_id=0,
        target_starts=[1],
        targets=[target],
    )
    assert batch.input_ids[0, :3].tolist() == [10, 11, 12]
    assert batch.attention_mask[0, :3].tolist() == [True] * 3
    assert batch.world_mask[0, :3].tolist() == [True] * 3
    assert batch.target_mask is not None
    assert batch.target_mask[0].nonzero().flatten().tolist() == [1, 2]


def test_operation_library_freezes_formula_operands_but_not_bank_keys() -> None:
    spec = mechanisms.PortSpec(6, (2,), 4, (0, 1))
    candidate_count = 2 * (6 - 2)
    bank = mechanisms.TensorOperationBank(
        spec, candidate_count=candidate_count, key_dim=5, seed=11
    )
    _initialize_operation_library(bank, world_source_stop=6)
    assert bank.keys.requires_grad
    assert not any(parameter.requires_grad for parameter in bank.operands.values())
    operations = bank.operands["operation"].argmax(dim=-1)
    assert int((operations == int(mechanisms.EditOperation.COPY)).sum()) == candidate_count
    assert (bank.operands["source"].argmax(dim=-1)[:, 0] < spec.canvas_tokens).all()


def test_workspace_loop_concatenates_complete_copy_width_banks() -> None:
    spec = mechanisms.PortSpec(8, (3,), 4, (0, 1, 2))
    loop = make_workspace_loop(
        spec,
        key_dim=5,
        seed=13,
        route_temperature=0.7,
        edit_temperature=1.0,
        world_source_stop=8,
        copy_widths=(1, 2, 3),
    )
    bank = loop.operation.selector.bank

    assert bank.bank_ids == ("copy-width-1", "copy-width-2", "copy-width-3")
    assert bank.group_slices == ((0, 15), (15, 23), (23, 26))
    assert bank.candidate_count == 26
    assert not any(parameter.requires_grad for parameter in bank.operands.values())
    operations = bank.operands["operation"].argmax(dim=-1)
    active = bank.operands["active"] >= 0
    for (start, end), width in zip(bank.group_slices, (1, 2, 3), strict=True):
        assert active[start:end].sum(dim=-1).unique().tolist() == [width]
        assert (
            (operations[start:end] == int(mechanisms.EditOperation.COPY)).sum(dim=-1).unique().tolist()
            == [width]
        )


def test_workspace_loop_can_apply_bank_owned_formula_after_range_copy() -> None:
    spec = mechanisms.PortSpec(8, (3,), 4, (0, 1, 2))
    loop = make_workspace_loop(
        spec,
        key_dim=5,
        seed=17,
        route_temperature=0.7,
        edit_temperature=1.0,
        world_source_stop=8,
        copy_widths=(1, 3),
        formula_rank=2,
    )

    assert loop.workspace_formula is not None
    assert loop.workspace_formula.rank == 2
    assert (
        loop.workspace_formula.operand_bank.member_ids
        == loop.operation.selector.bank.member_ids
    )
    assert not loop.workspace_formula.operand_bank.keys.requires_grad
    assert all(
        parameter.requires_grad
        for parameter in loop.workspace_formula.operand_bank.operands.values()
    )
    trainable = {
        name for name, parameter in loop.named_parameters() if parameter.requires_grad
    }
    assert "operation.selector.bank.keys" in trainable
    assert "workspace_formula.operand_bank.operands.A" in trainable
    assert "workspace_formula.operand_bank.operands.B" in trainable
    assert "workspace_formula.operand_bank.operands.gain" in trainable


def test_workspace_formula_only_transforms_selected_copy_destinations() -> None:
    spec = mechanisms.PortSpec(8, (3,), 4, (0, 1, 2))
    loop = make_workspace_loop(
        spec,
        key_dim=5,
        seed=18,
        route_temperature=0.7,
        edit_temperature=1.0,
        world_source_stop=8,
        copy_widths=(1,),
        formula_rank=2,
    )
    assert loop.workspace_formula is not None
    with torch.no_grad():
        loop.workspace_formula.operand_bank.operands["B"].fill_(0.25)
    snapshot = empty_workspace_snapshot(spec, 2, device="cpu")
    world = torch.randn(2, spec.canvas_tokens, spec.dim)
    step = loop.operation(world, snapshot)
    transformed = loop.workspace_formula(
        step.edit.value,
        step.decision.route.route,
        step.decision.instruction,
        active=torch.ones(2, dtype=torch.bool),
    )
    destination = step.decision.instruction.destination_offset[:, 0]
    support = torch.nn.functional.one_hot(
        destination,
        num_classes=spec.element_count,
    ).bool()

    torch.testing.assert_close(
        transformed.masked_select(~support.unsqueeze(-1)),
        step.edit.value.masked_select(~support.unsqueeze(-1)),
    )
    assert not torch.equal(
        transformed.masked_select(support.unsqueeze(-1)),
        step.edit.value.masked_select(support.unsqueeze(-1)),
    )


def test_qwen_bridge_replaces_layer_output_and_preserves_next_workspace() -> None:
    spec = mechanisms.PortSpec(4, (1,), 3, (0,))
    loop = make_workspace_loop(
        spec,
        key_dim=4,
        seed=19,
        route_temperature=0.7,
        edit_temperature=1.3,
        world_source_stop=4,
    )
    assert loop.operation.selector.temperature == 0.7
    assert loop.operation.surrogate.temperature == 1.3
    layer = FakeLayer()
    bridge = QwenWorkspaceBridge(layer, loop)
    snapshot = empty_workspace_snapshot(spec, 1, device="cpu")
    world = torch.randn((1, 4, 3))
    world_mask = torch.tensor([[False, True, True, True]])
    with bridge.use(snapshot, world_mask, WorkspaceTurnSchedule(0, 1)):
        output = layer(world)[0]
        result = bridge.take_result()
    assert output.shape == world.shape
    assert result.next_snapshot.step_index == 1
    assert result.next_snapshot.value.shape == (1, 1, 3)
    assert result.next_snapshot.mask.shape == (1, 1)
    assert torch.isfinite(result.next_snapshot.value).all()
    bridge.close()


def test_model_batch_to_preserves_optional_fields() -> None:
    target = FrozenTeacherTarget(
        response_ids=torch.tensor([2]),
        topk_ids=torch.tensor([[2]]),
        topk_logits=torch.tensor([[1.0]]),
    )
    built = _pad_sequences(
        [torch.tensor([3])],
        canvas_tokens=3,
        placeholder_token_id=0,
        target_starts=[0],
        targets=[target],
    ).to(torch.device("cpu"))
    assert built.target_ids is not None
    assert built.teacher_topk_logits is not None
