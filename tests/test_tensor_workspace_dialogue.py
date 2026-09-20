from __future__ import annotations

import torch
from torch import nn

from arti import mechanisms
from benchmarks._tensor_workspace_dialogue import (
    DialogueWorkspaceLoop,
    WorkspaceTurnSchedule,
    detach_workspace_snapshot,
    empty_workspace_snapshot,
)


class AddOneReader(nn.Module):
    def forward(self, value: torch.Tensor, *, mask: torch.Tensor) -> torch.Tensor:
        return value + mask.unsqueeze(-1).to(value.dtype)


def _spec() -> mechanisms.PortSpec:
    return mechanisms.PortSpec(
        canvas_tokens=4,
        tensor_shape=(1,),
        dim=3,
        tensor_to_canvas=(3,),
    )


def _copy_operation(spec: mechanisms.PortSpec) -> mechanisms.TensorOperation:
    bank = mechanisms.TensorOperationBank(spec, candidate_count=1, key_dim=4, seed=3)
    with torch.no_grad():
        bank.operands["active"].fill_(4)
        bank.operands["operation"].fill_(-4)
        bank.operands["operation"][:, :, int(mechanisms.EditOperation.COPY)] = 4
        bank.operands["source"].fill_(-4)
        bank.operands["source"][:, :, 0] = 4
        bank.operands["destination"].zero_()
    selector = mechanisms.TensorOperationSelector(spec, bank, query_seed=5)
    return mechanisms.TensorOperation(
        spec,
        selector,
        surrogate=mechanisms.TensorEditSurrogate(spec),
    )


def test_new_dialogues_start_from_independent_empty_workspaces() -> None:
    spec = _spec()
    first = empty_workspace_snapshot(spec, 2, device="cpu")
    second = empty_workspace_snapshot(spec, 2, device="cpu")
    assert not first.mask.any()
    assert not second.mask.any()
    first.value[0, 0, 0] = 7
    assert second.value[0, 0, 0] == 0


def test_terminal_fold_makes_internal_edit_visible_in_current_call() -> None:
    spec = _spec()
    loop = DialogueWorkspaceLoop(spec, AddOneReader(), _copy_operation(spec))
    world = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    snapshot = empty_workspace_snapshot(spec, 2, device="cpu")
    result = loop(
        world,
        snapshot,
        schedule=WorkspaceTurnSchedule(reader_steps=0, operation_steps=1),
    )
    torch.testing.assert_close(result.next_snapshot.value[:, 0], world[:, 0])
    torch.testing.assert_close(result.output[:, 3], world[:, 0])
    assert result.next_snapshot.mask.all()
    assert result.trace.operation_attempted.shape == (1, 2)


def test_reader_and_operation_have_independent_depths_and_same_snapshot_lag() -> None:
    spec = _spec()
    loop = DialogueWorkspaceLoop(spec, AddOneReader(), _copy_operation(spec))
    world = torch.zeros((1, 4, 3))
    world[:, 0] = 2
    result = loop(
        world,
        empty_workspace_snapshot(spec, 1, device="cpu"),
        schedule=WorkspaceTurnSchedule(reader_steps=2, operation_steps=1),
    )
    assert result.trace.reader_attempted[:, 0].tolist() == [True, True]
    assert result.trace.operation_attempted[:, 0].tolist() == [True, False]
    # The ordinary world slots receive both Reader steps. The terminal Fold
    # preserves the copied workspace value at its mapped canvas position.
    torch.testing.assert_close(result.output[:, 1], torch.full((1, 3), 2.0))
    torch.testing.assert_close(result.output[:, 3], torch.full((1, 3), 2.0))


def test_later_turn_loss_reaches_the_earlier_operation_bank() -> None:
    spec = _spec()
    operation = _copy_operation(spec)
    loop = DialogueWorkspaceLoop(spec, AddOneReader(), operation)
    world = torch.randn((3, 4, 3))
    first = loop(
        world,
        empty_workspace_snapshot(spec, 3, device="cpu"),
        schedule=WorkspaceTurnSchedule(reader_steps=0, operation_steps=1),
    )
    second_world = torch.zeros_like(world)
    second = loop(
        second_world,
        first.next_snapshot,
        schedule=WorkspaceTurnSchedule(reader_steps=1, operation_steps=0),
    )
    second.output.square().mean().backward()
    gradients = [
        parameter.grad
        for parameter in operation.selector.bank.parameters()
        if parameter.requires_grad
    ]
    assert any(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)


def test_detach_is_explicit_and_preserves_values() -> None:
    value = torch.randn((1, 1, 3), requires_grad=True)
    snapshot = mechanisms.PortSnapshot(
        value,
        torch.ones((1, 1), dtype=torch.bool),
        "default",
        2,
        9,
    )
    detached = detach_workspace_snapshot(snapshot)
    torch.testing.assert_close(detached.value, value)
    assert not detached.value.requires_grad
    assert detached.backing_epoch == 2
    assert detached.step_index == 9
