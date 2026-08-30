from __future__ import annotations

import pytest
import torch

import arti
from arti import alpha


def _spec() -> alpha.PortSpec:
    return alpha.PortSpec(
        canvas_tokens=4,
        port_slots=2,
        dim=3,
        port_to_canvas=(1, 3),
    )


def _instruction(
    operation: list[int],
    source_plane: list[int],
    source_index: list[int],
    destination_index: list[int],
    *,
    device: torch.device | str = "cpu",
) -> alpha.TensorEditInstruction:
    return alpha.TensorEditInstruction(
        operation=torch.tensor(operation, dtype=torch.int64, device=device),
        source_plane=torch.tensor(source_plane, dtype=torch.int64, device=device),
        source_index=torch.tensor(source_index, dtype=torch.int64, device=device),
        destination_index=torch.tensor(destination_index, dtype=torch.int64, device=device),
        active=torch.ones(len(operation), dtype=torch.bool, device=device),
    )


def test_port_spec_rejects_ambiguous_maps() -> None:
    with pytest.raises(ValueError, match="injective"):
        alpha.PortSpec(4, 2, 3, (1, 1))
    with pytest.raises(ValueError, match="out-of-range"):
        alpha.PortSpec(4, 2, 3, (1, 4))


def test_operable_port_hotplug_retains_default_state() -> None:
    spec = _spec()
    port = alpha.OperableTensorPort(spec, batch_size=1)
    default = port.resolve()
    assert default.source == "default"
    assert default.value.shape == (1, 2, 3)
    assert not default.mask.any()

    evolved_default = torch.full_like(default.value, 2.0)
    evolved_mask = torch.ones_like(default.mask)
    port.advance(evolved_default, evolved_mask)

    external_a = torch.full_like(default.value, 5.0)
    port.mount(external_a, evolved_mask)
    assert port.resolve().source == "external"
    assert torch.equal(port.resolve().value, external_a)

    external_b = torch.full_like(default.value, 7.0)
    port.replace(external_b, evolved_mask)
    assert torch.equal(port.resolve().value, external_b)

    port.detach_to_default()
    restored = port.resolve()
    assert restored.source == "default"
    assert torch.equal(restored.value, evolved_default)
    assert torch.equal(restored.mask, evolved_mask)
    assert port.backing_epoch == 3
    assert port.step_index == 1


def test_shared_canvas_is_world_shaped_masked_overlay() -> None:
    spec = _spec()
    world = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    backing = torch.tensor(
        [
            [[100.0, 101.0, 102.0], [200.0, 201.0, 202.0]],
            [[300.0, 301.0, 302.0], [400.0, 401.0, 402.0]],
        ]
    )
    mask = torch.tensor([[True, False], [False, True]])
    snapshot = alpha.PortSnapshot(backing, mask, "external", 4, 9)
    world_before = world.clone()
    backing_before = backing.clone()

    canvas = alpha.SharedCanvasFold(spec)(world, snapshot)

    assert canvas.values.shape == world.shape
    torch.testing.assert_close(canvas.values[0, 1], backing[0, 0])
    torch.testing.assert_close(canvas.values[0, 3], world[0, 3])
    torch.testing.assert_close(canvas.values[1, 1], world[1, 1])
    torch.testing.assert_close(canvas.values[1, 3], backing[1, 1])
    assert canvas.source_plane[0, 1].item() == int(alpha.CanvasSource.BACKING)
    assert canvas.source_index[0, 1].item() == 0
    assert canvas.source_plane[0, 3].item() == int(alpha.CanvasSource.WORLD)
    assert canvas.backing_epoch == 4
    assert canvas.step_index == 9
    torch.testing.assert_close(world, world_before)
    torch.testing.assert_close(backing, backing_before)


def test_shared_canvas_uses_explicit_empty_value_when_both_sources_are_hidden() -> None:
    spec = alpha.PortSpec(4, 2, 3, (1, 3), empty_value=-2.0)
    world = torch.full((1, 4, 3), 9.0)
    world_mask = torch.tensor([[True, False, True, False]])
    backing = torch.full((1, 2, 3), 7.0)
    backing_mask = torch.zeros((1, 2), dtype=torch.bool)
    snapshot = alpha.PortSnapshot(backing, backing_mask, "default", 0, 0)

    canvas = alpha.SharedCanvasFold(spec)(world, snapshot, world_mask=world_mask)

    torch.testing.assert_close(canvas.values[:, 1], torch.full((1, 3), -2.0))
    torch.testing.assert_close(canvas.values[:, 3], torch.full((1, 3), -2.0))
    assert not canvas.mask[:, [1, 3]].any()
    assert (canvas.source_plane[:, [1, 3]] == int(alpha.CanvasSource.EMPTY)).all()


def test_tensor_edit_formula_applies_keep_copy_and_clear_from_snapshot() -> None:
    spec = _spec()
    world = torch.arange(36, dtype=torch.float32).reshape(3, 4, 3)
    backing = torch.tensor(
        [
            [[10.0, 11.0, 12.0], [20.0, 21.0, 22.0]],
            [[30.0, 31.0, 32.0], [40.0, 41.0, 42.0]],
            [[50.0, 51.0, 52.0], [60.0, 61.0, 62.0]],
        ]
    )
    mask = torch.ones((3, 2), dtype=torch.bool)
    snapshot = alpha.PortSnapshot(backing, mask, "default", 0, 0)
    canvas = alpha.SharedCanvasFold(spec)(world, snapshot)
    instruction = _instruction(
        [int(alpha.EditOperation.KEEP), int(alpha.EditOperation.COPY), int(alpha.EditOperation.CLEAR)],
        [-1, int(alpha.CanvasSource.BACKING), -1],
        [-1, 0, -1],
        [-1, 1, 0],
    )
    world_before = world.clone()
    backing_before = backing.clone()

    result = alpha.TensorEditFormula(spec)(canvas, snapshot, instruction)

    torch.testing.assert_close(result.value[0], backing[0])
    torch.testing.assert_close(result.value[1, 1], backing[1, 0])
    torch.testing.assert_close(result.value[2, 0], torch.zeros(3))
    assert result.mask[2, 0].item() is False
    torch.testing.assert_close(world, world_before)
    torch.testing.assert_close(backing, backing_before)
    assert result.value.data_ptr() != backing.data_ptr()


def test_copy_world_becomes_visible_only_after_next_port_advance() -> None:
    spec = _spec()
    port = alpha.OperableTensorPort(spec, batch_size=1)
    fold = alpha.SharedCanvasFold(spec)
    formula = alpha.TensorEditFormula(spec)
    world = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    before = fold(world, port.resolve())
    instruction = _instruction(
        [int(alpha.EditOperation.COPY)],
        [int(alpha.CanvasSource.WORLD)],
        [2],
        [0],
    )

    edited = formula(before, port.resolve(), instruction)

    torch.testing.assert_close(before.values[:, 1], world[:, 1])
    port.advance(edited.value, edited.mask)
    after = fold(world, port.resolve())
    torch.testing.assert_close(after.values[:, 1], world[:, 2])


def test_invalid_edit_fails_before_modifying_backing() -> None:
    spec = _spec()
    world = torch.zeros((1, 4, 3))
    backing = torch.ones((1, 2, 3))
    mask = torch.ones((1, 2), dtype=torch.bool)
    snapshot = alpha.PortSnapshot(backing, mask, "default", 0, 0)
    canvas = alpha.SharedCanvasFold(spec)(world, snapshot)
    invalid = _instruction(
        [int(alpha.EditOperation.COPY)],
        [int(alpha.CanvasSource.WORLD)],
        [9],
        [0],
    )
    before = backing.clone()

    with pytest.raises(ValueError, match="world source"):
        alpha.TensorEditFormula(spec)(canvas, snapshot, invalid)

    torch.testing.assert_close(backing, before)


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_tensor_edit_copy_is_differentiable_and_device_native(device: str) -> None:
    spec = _spec()
    world = torch.randn((2, 4, 3), device=device, requires_grad=True)
    backing = torch.randn((2, 2, 3), device=device, requires_grad=True)
    mask = torch.ones((2, 2), dtype=torch.bool, device=device)
    snapshot = alpha.PortSnapshot(backing, mask, "external", 1, 0)
    canvas = alpha.SharedCanvasFold(spec)(world, snapshot)
    instruction = _instruction(
        [int(alpha.EditOperation.COPY), int(alpha.EditOperation.COPY)],
        [int(alpha.CanvasSource.WORLD), int(alpha.CanvasSource.BACKING)],
        [2, 0],
        [0, 1],
        device=device,
    )
    result = alpha.TensorEditFormula(spec)(canvas, snapshot, instruction)

    result.value.sum().backward()

    assert world.grad is not None and torch.isfinite(world.grad).all()
    assert backing.grad is not None and torch.isfinite(backing.grad).all()
    assert result.value.device.type == device


def test_tensor_plane_component_identities_are_versioned() -> None:
    spec = _spec()
    port = alpha.OperableTensorPort(spec, batch_size=1)
    snapshot = port.resolve()
    world = torch.zeros((1, 4, 3))
    canvas = alpha.SharedCanvasFold(spec)(world, snapshot)
    instruction = _instruction([0], [-1], [-1], [-1])
    result = alpha.TensorEditFormula(spec)(canvas, snapshot, instruction)

    assert arti.component_ref(spec) == "arti/operable-tensor-port-spec@1"
    assert arti.component_ref(port) == "arti/operable-tensor-port@1"
    assert arti.component_ref(snapshot) == "arti/operable-tensor-snapshot@1"
    assert arti.component_ref(canvas) == "arti/shared-canvas@1"
    assert arti.component_ref(alpha.SharedCanvasFold(spec)) == "arti/shared-canvas-fold@1"
    assert arti.component_ref(instruction) == "arti/tensor-edit-instruction@1"
    assert arti.component_ref(result) == "arti/tensor-edit-result@1"


def _configured_copy_selector(spec: alpha.PortSpec) -> alpha.TensorOperationSelector:
    bank = alpha.TensorOperationBank(spec, candidate_count=1, key_dim=5, seed=3)
    with torch.no_grad():
        bank.operands["operation"].fill_(-5.0)
        bank.operands["operation"][:, int(alpha.EditOperation.COPY)] = 5.0
        bank.operands["source_plane"].fill_(-5.0)
        bank.operands["source_plane"][:, int(alpha.CanvasSource.WORLD)] = 5.0
        bank.operands["world_source"].fill_(-5.0)
        bank.operands["world_source"][:, 2] = 5.0
        bank.operands["backing_source"].zero_()
        bank.operands["destination"].fill_(-5.0)
        bank.operands["destination"][:, 0] = 5.0
    return alpha.TensorOperationSelector(spec, bank, estimator="hard", query_seed=7)


def test_tensor_operation_selector_uses_fixed_query_and_bank_owned_parameters() -> None:
    spec = _spec()
    bank = alpha.TensorOperationBank(spec, candidate_count=4, key_dim=6, seed=11)
    selector = alpha.TensorOperationSelector(spec, bank, query_seed=13)

    assert list(selector.query.parameters()) == []
    assert selector.query.operation_query_contract()["fixed"] is True
    assert all(name.startswith("bank.") for name, _ in selector.named_parameters())
    assert {name for name, _ in selector.named_parameters()} == {
        "bank.keys",
        "bank.operands.backing_source",
        "bank.operands.destination",
        "bank.operands.operation",
        "bank.operands.source_plane",
        "bank.operands.world_source",
    }


def test_tensor_operation_selector_decodes_bank_selected_hard_instruction() -> None:
    spec = _spec()
    port = alpha.OperableTensorPort(spec, batch_size=2)
    world = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    canvas = alpha.SharedCanvasFold(spec)(world, port.resolve())
    selector = _configured_copy_selector(spec)

    control = selector(canvas)

    assert (control.instruction.operation == int(alpha.EditOperation.COPY)).all()
    assert (control.instruction.source_plane == int(alpha.CanvasSource.WORLD)).all()
    assert (control.instruction.source_index == 2).all()
    assert (control.instruction.destination_index == 0).all()
    assert control.route.hard_indices.shape == (2,)


def test_tensor_operation_selector_training_logits_update_only_bank_parameters() -> None:
    spec = _spec()
    port = alpha.OperableTensorPort(spec, batch_size=5)
    world = torch.randn((5, 4, 3), requires_grad=True)
    canvas = alpha.SharedCanvasFold(spec)(world, port.resolve())
    bank = alpha.TensorOperationBank(spec, candidate_count=4, key_dim=7, seed=17)
    selector = alpha.TensorOperationSelector(spec, bank, query_seed=19)
    control = selector(canvas)
    targets = torch.arange(5) % 2
    loss = (
        torch.nn.functional.cross_entropy(
            control.operation_logits, torch.arange(5) % len(alpha.EditOperation)
        )
        + torch.nn.functional.cross_entropy(control.source_plane_logits, targets)
        + torch.nn.functional.cross_entropy(
            control.world_source_logits, torch.arange(5) % spec.canvas_tokens
        )
        + torch.nn.functional.cross_entropy(
            control.backing_source_logits, targets % spec.port_slots
        )
        + torch.nn.functional.cross_entropy(
            control.destination_logits, (targets + 1) % spec.port_slots
        )
    )

    loss.backward()

    assert world.grad is None
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in bank.parameters()
    )


def test_tensor_edit_surrogate_is_exact_forward_and_routes_final_loss_to_bank() -> None:
    torch.manual_seed(41)
    spec = _spec()
    batch = 6
    world = torch.randn((batch, spec.canvas_tokens, spec.dim))
    backing = torch.randn((batch, spec.port_slots, spec.dim))
    mask = torch.ones((batch, spec.port_slots), dtype=torch.bool)
    snapshot = alpha.PortSnapshot(backing, mask, "default", 0, 0)
    bank = alpha.TensorOperationBank(spec, candidate_count=5, key_dim=7, seed=43)
    selector = alpha.TensorOperationSelector(spec, bank, query_seed=47)
    query_basis = selector.query.basis.clone()
    operation = alpha.TensorOperation(
        spec,
        selector,
        surrogate=alpha.TensorEditSurrogate(spec, temperature=0.8),
    )

    step = operation(world, snapshot)
    hard = alpha.TensorEditFormula(spec)(
        step.canvas,
        snapshot,
        step.decision.instruction,
    )

    assert torch.equal(step.edit.value, hard.value)
    assert torch.equal(step.edit.mask, hard.mask)
    next_snapshot = alpha.PortSnapshot(
        step.edit.value,
        step.edit.mask,
        snapshot.source,
        snapshot.backing_epoch,
        snapshot.step_index + 1,
    )
    next_world = torch.zeros_like(world)
    next_canvas = alpha.SharedCanvasFold(spec)(next_world, next_snapshot)
    target = torch.randn_like(next_canvas.values)
    torch.nn.functional.mse_loss(next_canvas.values, target).backward()

    assert list(selector.query.parameters()) == []
    torch.testing.assert_close(selector.query.basis, query_basis)
    assert bank.keys.grad is not None
    assert torch.isfinite(bank.keys.grad).all()
    assert bank.keys.grad.abs().sum() > 0
    assert all(
        parameter.grad is not None
        and torch.isfinite(parameter.grad).all()
        and parameter.grad.abs().sum() > 0
        for parameter in bank.operands.parameters()
    )


def test_multi_step_surrogate_reaches_bank_from_next_call_consumer_only() -> None:
    torch.manual_seed(53)
    spec = _spec()
    batch = 4
    world = torch.randn((batch, spec.canvas_tokens, spec.dim))
    backing = torch.randn((batch, spec.port_slots, spec.dim))
    snapshot = alpha.PortSnapshot(
        backing,
        torch.ones((batch, spec.port_slots), dtype=torch.bool),
        "default",
        0,
        0,
    )
    bank = alpha.TensorOperationBank(spec, candidate_count=6, key_dim=8, seed=59)
    selector = alpha.TensorOperationSelector(spec, bank, query_seed=61)
    loop = alpha.TensorOperationLoop(
        alpha.TensorOperation(
            spec,
            selector,
            surrogate=alpha.TensorEditSurrogate(spec),
        )
    )

    result = loop(
        world,
        snapshot,
        schedule=alpha.TensorOperationSchedule(2),
    )
    next_snapshot = alpha.PortSnapshot(
        result.value,
        result.mask,
        snapshot.source,
        snapshot.backing_epoch,
        snapshot.step_index + 1,
    )
    consumer_world = torch.zeros_like(world)
    consumer = alpha.SharedCanvasFold(spec)(consumer_world, next_snapshot)
    target = torch.randn_like(consumer.values)
    torch.nn.functional.mse_loss(consumer.values, target).backward()

    assert result.trace.completed_steps.tolist() == [2] * batch
    assert result.trace.route_index.shape == (2, batch)
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in bank.parameters()
    )


def test_tensor_operation_surrogate_arti_st_round_trip(tmp_path) -> None:
    spec = _spec()

    def build() -> alpha.TensorOperationLoop:
        bank = alpha.TensorOperationBank(spec, candidate_count=5, key_dim=7, seed=71)
        selector = alpha.TensorOperationSelector(
            spec,
            bank,
            query_seed=73,
            temperature=0.6,
        )
        return alpha.TensorOperationLoop(
            alpha.TensorOperation(
                spec,
                selector,
                surrogate=alpha.TensorEditSurrogate(spec, temperature=0.6),
            ),
            stop=alpha.TensorOperationStopPolicy(
                min_operation_steps=1,
                stop_on_stable=True,
            ),
        )

    source = build()
    target = build()
    with torch.no_grad():
        source.operation.selector.bank.keys.add_(0.25)
    world = torch.randn(4, 4, 3)
    snapshot = alpha.OperableTensorPort(spec, batch_size=4).resolve()
    expected = source(
        world,
        snapshot,
        schedule=alpha.TensorOperationSchedule(4),
    )

    saved = arti.save(source, tmp_path / "tensor-operation.arti.st")
    loaded = arti.load(saved.weights_path, model=target)
    actual = target(
        world,
        snapshot,
        schedule=alpha.TensorOperationSchedule(4),
    )

    assert loaded.missing_keys == ()
    assert loaded.unexpected_keys == ()
    torch.testing.assert_close(actual.value, expected.value, rtol=0, atol=0)
    assert torch.equal(actual.mask, expected.mask)
    assert torch.equal(actual.trace.route_index, expected.trace.route_index)


def test_tensor_invocation_keeps_operation_out_of_current_reader() -> None:
    class RecordingRefine(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.seen: torch.Tensor | None = None

        def forward(self, value: torch.Tensor, *, mask: torch.Tensor) -> torch.Tensor:
            self.seen = value.detach().clone()
            return torch.where(mask.unsqueeze(-1), value + 1, value)

    spec = _spec()
    port = alpha.OperableTensorPort(spec, batch_size=1)
    refine = RecordingRefine()
    operation = alpha.TensorOperation(spec, _configured_copy_selector(spec))
    invocation = alpha.TensorInvocation(
        spec,
        refine,
        alpha.TensorOperationLoop(operation),
    )
    world = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)

    result = invocation(
        world,
        port.resolve(),
        operation_schedule=alpha.TensorOperationSchedule(1),
    )

    assert refine.seen is not None
    torch.testing.assert_close(refine.seen[:, 1], world[:, 1])
    torch.testing.assert_close(result.output[:, 1], world[:, 1] + 1)
    torch.testing.assert_close(result.operation.value[:, 0], world[:, 2])
    torch.testing.assert_close(world, torch.arange(12, dtype=torch.float32).reshape(1, 4, 3))
    port.advance(result.operation.value, result.operation.mask)
    later = invocation.fold(world, port.resolve())
    torch.testing.assert_close(later.values[:, 1], world[:, 2])


def test_tensor_operation_components_have_canonical_alpha_identities() -> None:
    spec = _spec()
    selector = _configured_copy_selector(spec)
    port = alpha.OperableTensorPort(spec, batch_size=1)
    canvas = alpha.SharedCanvasFold(spec)(torch.zeros((1, 4, 3)), port.resolve())
    decision = selector(canvas)
    operation = alpha.TensorOperation(spec, selector)
    loop = alpha.TensorOperationLoop(operation)
    invocation = alpha.TensorInvocation(spec, torch.nn.Identity(), loop)

    assert arti.component_ref(selector.query) == "arti/tensor-operation-query@1"
    assert arti.component_ref(selector.bank) == "arti/tensor-operation-bank@1"
    assert arti.component_ref(selector) == "arti/tensor-operation-selector@1"
    assert arti.component_ref(decision) == "arti/tensor-operation-decision@1"
    assert arti.component_ref(operation) == "arti/tensor-operation@1"
    assert arti.component_ref(loop) == "arti/tensor-operation-loop@1"
    assert arti.component_ref(invocation) == "arti/tensor-invocation@1"


def test_tensor_operation_loop_refolds_and_requeries_latest_shadow() -> None:
    spec = _spec()
    base_selector = _configured_copy_selector(spec)

    class RecordingSelector(alpha.TensorOperationSelector):
        def __init__(self) -> None:
            super().__init__(spec, base_selector.bank, estimator="hard", query=base_selector.query)
            self.seen_backings: list[torch.Tensor] = []

        def forward(self, canvas: alpha.SharedCanvas) -> alpha.TensorOperationDecision:
            self.seen_backings.append(canvas.backing_values.detach().clone())
            return super().forward(canvas)

    selector = RecordingSelector()
    loop = alpha.TensorOperationLoop(alpha.TensorOperation(spec, selector))
    port = alpha.OperableTensorPort(spec, batch_size=1)
    root = port.resolve()
    world = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)

    result = loop(
        world,
        root,
        schedule=alpha.TensorOperationSchedule(operation_steps=2),
    )

    assert len(selector.seen_backings) == 2
    torch.testing.assert_close(selector.seen_backings[0], root.value)
    torch.testing.assert_close(selector.seen_backings[1][:, 0], world[:, 2])
    torch.testing.assert_close(result.value[:, 0], world[:, 2])
    assert result.trace.completed_steps.tolist() == [2]
    assert result.trace.attempted.shape == (2, 1)
    torch.testing.assert_close(port.resolve().value, root.value)
    assert port.step_index == root.step_index


def test_tensor_operation_schedule_supports_independent_per_row_depth() -> None:
    spec = _spec()
    batch = 3
    port = alpha.OperableTensorPort(spec, batch_size=batch)
    world = torch.arange(batch * 12, dtype=torch.float32).reshape(batch, 4, 3)
    loop = alpha.TensorOperationLoop(
        alpha.TensorOperation(spec, _configured_copy_selector(spec))
    )
    steps = torch.tensor([0, 1, 3], dtype=torch.int64)

    result = loop(
        world,
        port.resolve(),
        schedule=alpha.TensorOperationSchedule(steps),
    )

    assert result.trace.completed_steps.tolist() == [0, 1, 3]
    assert result.trace.attempted.shape == (3, batch)
    assert not result.trace.attempted[:, 0].any()
    assert result.trace.attempted[:, 1].tolist() == [True, False, False]
    assert result.trace.attempted[:, 2].all()
    assert result.trace.stop_reason[0].item() == int(
        alpha.TensorOperationStopReason.ZERO_STEPS
    )


def test_tensor_operation_stable_stop_is_post_transition_and_bounded() -> None:
    spec = _spec()
    selector = _configured_copy_selector(spec)
    with torch.no_grad():
        selector.bank.operands["operation"].fill_(-5.0)
        selector.bank.operands["operation"][:, int(alpha.EditOperation.KEEP)] = 5.0
    loop = alpha.TensorOperationLoop(
        alpha.TensorOperation(spec, selector),
        stop=alpha.TensorOperationStopPolicy(
            min_operation_steps=2,
            stop_on_stable=True,
        ),
    )
    port = alpha.OperableTensorPort(spec, batch_size=1)
    world = torch.zeros((1, 4, 3))

    result = loop(
        world,
        port.resolve(),
        schedule=alpha.TensorOperationSchedule(8),
    )

    assert result.trace.completed_steps.tolist() == [2]
    assert result.trace.stop_reason.tolist() == [int(alpha.TensorOperationStopReason.STABLE)]


def test_tensor_operation_static_masked_keeps_capacity_after_logical_stop() -> None:
    class CountingSelector(alpha.TensorOperationSelector):
        def __init__(self, spec: alpha.PortSpec) -> None:
            super().__init__(spec, _configured_copy_selector(spec).bank, query_seed=7)
            self.calls = 0

        def forward(self, canvas: alpha.SharedCanvas) -> alpha.TensorOperationDecision:
            self.calls += 1
            return super().forward(canvas)

    spec = _spec()
    selector = CountingSelector(spec)
    with torch.no_grad():
        selector.bank.operands["operation"].fill_(-5.0)
        selector.bank.operands["operation"][:, int(alpha.EditOperation.KEEP)] = 5.0
    loop = alpha.TensorOperationLoop(
        alpha.TensorOperation(spec, selector),
        stop=alpha.TensorOperationStopPolicy(
            min_operation_steps=1,
            stop_on_stable=True,
        ),
        executor="static_masked",
    )
    world = torch.zeros((1, 4, 3))
    snapshot = alpha.OperableTensorPort(spec, batch_size=1).resolve()

    result = loop(
        world,
        snapshot,
        schedule=alpha.TensorOperationSchedule(8),
    )

    assert selector.calls == 8
    assert result.trace.attempted.shape == (8, 1)
    assert result.trace.completed_steps.tolist() == [1]
    assert result.trace.attempted[:, 0].tolist() == [True] + [False] * 7


def test_tensor_operation_early_break_short_circuits_after_logical_stop() -> None:
    class CountingSelector(alpha.TensorOperationSelector):
        def __init__(self, spec: alpha.PortSpec) -> None:
            super().__init__(spec, _configured_copy_selector(spec).bank, query_seed=7)
            self.calls = 0

        def forward(self, canvas: alpha.SharedCanvas) -> alpha.TensorOperationDecision:
            self.calls += 1
            return super().forward(canvas)

    spec = _spec()
    selector = CountingSelector(spec)
    with torch.no_grad():
        selector.bank.operands["operation"].fill_(-5.0)
        selector.bank.operands["operation"][:, int(alpha.EditOperation.KEEP)] = 5.0
    loop = alpha.TensorOperationLoop(
        alpha.TensorOperation(spec, selector),
        stop=alpha.TensorOperationStopPolicy(
            min_operation_steps=1,
            stop_on_stable=True,
        ),
        executor="early_break",
    )
    world = torch.zeros((1, 4, 3))
    snapshot = alpha.OperableTensorPort(spec, batch_size=1).resolve()

    result = loop(
        world,
        snapshot,
        schedule=alpha.TensorOperationSchedule(8),
    )

    assert selector.calls == 1
    assert result.trace.attempted.shape == (1, 1)
    assert result.trace.completed_steps.tolist() == [1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_tensor_operation_early_break_rejects_cuda_host_sync() -> None:
    spec = _spec()
    loop = alpha.TensorOperationLoop(
        alpha.TensorOperation(
            spec,
            alpha.TensorOperationSelector(
                spec,
                alpha.TensorOperationBank(spec, candidate_count=2, key_dim=7, seed=71),
                query_seed=73,
            ),
        ),
        executor="early_break",
    ).cuda()
    world = torch.randn(2, 4, 3, device="cuda")
    snapshot = alpha.PortSnapshot(
        torch.zeros(2, 2, 3, device="cuda"),
        torch.ones(2, 2, dtype=torch.bool, device="cuda"),
        "default",
        0,
        0,
    )

    with pytest.raises(ValueError, match="eager CPU"):
        loop(world, snapshot)


def test_tensor_operation_tensor_schedule_uses_explicit_static_capacity() -> None:
    spec = _spec()
    loop = alpha.TensorOperationLoop(
        alpha.TensorOperation(spec, _configured_copy_selector(spec)),
        executor="static_masked",
    )
    world = torch.zeros((2, 4, 3))
    snapshot = alpha.OperableTensorPort(spec, batch_size=2).resolve()
    schedule = alpha.TensorOperationSchedule(
        torch.tensor([1, 3], dtype=torch.int64),
        max_steps=4,
    )

    result = loop(world, snapshot, schedule=schedule)

    assert result.trace.attempted.shape == (4, 2)
    assert result.trace.completed_steps.tolist() == [1, 3]
    assert result.trace.attempted[:, 0].tolist() == [True, False, False, False]
    assert result.trace.attempted[:, 1].tolist() == [True, True, True, False]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_tensor_operation_static_masked_cuda_has_no_host_sync() -> None:
    spec = _spec()
    loop = alpha.TensorOperationLoop(
        alpha.TensorOperation(
            spec,
            alpha.TensorOperationSelector(
                spec,
                alpha.TensorOperationBank(
                    spec,
                    candidate_count=4,
                    key_dim=7,
                    seed=79,
                ),
                query_seed=83,
            ),
            surrogate=alpha.TensorEditSurrogate(spec),
        ),
        executor="static_masked",
    ).cuda()
    world = torch.randn(3, 4, 3, device="cuda")
    value = torch.zeros(3, 2, 3, device="cuda")
    mask = torch.ones(3, 2, dtype=torch.bool, device="cuda")
    snapshot = alpha.PortSnapshot(value, mask, "default", 0, 0)
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()

    try:
        torch.cuda.set_sync_debug_mode("error")
        result = loop(
            world,
            snapshot,
            schedule=alpha.TensorOperationSchedule(4),
        )
    finally:
        torch.cuda.set_sync_debug_mode(previous)

    torch.cuda.synchronize()
    assert result.value.is_cuda
    assert result.trace.attempted.shape == (4, 3)


def test_reader_and_operation_depths_are_orthogonal() -> None:
    class AddOne(torch.nn.Module):
        def forward(self, value: torch.Tensor, *, mask: torch.Tensor) -> torch.Tensor:
            return torch.where(mask.unsqueeze(-1), value + 1, value)

    spec = _spec()
    port = alpha.OperableTensorPort(spec, batch_size=1)
    world = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    operation = alpha.TensorOperationLoop(
        alpha.TensorOperation(spec, _configured_copy_selector(spec))
    )
    invocation = alpha.TensorInvocation(spec, AddOne(), operation)

    r1_h1 = invocation(
        world,
        port.resolve(),
        reader_schedule=alpha.ReaderRefineSchedule(1),
        operation_schedule=alpha.TensorOperationSchedule(1),
    )
    r3_h1 = invocation(
        world,
        port.resolve(),
        reader_schedule=alpha.ReaderRefineSchedule(3),
        operation_schedule=alpha.TensorOperationSchedule(1),
    )
    r1_h3 = invocation(
        world,
        port.resolve(),
        reader_schedule=alpha.ReaderRefineSchedule(1),
        operation_schedule=alpha.TensorOperationSchedule(3),
    )

    torch.testing.assert_close(r1_h1.operation.value, r3_h1.operation.value)
    torch.testing.assert_close(r1_h1.output, r1_h3.output)
    assert not torch.equal(r1_h1.output, r3_h1.output)
    assert r1_h1.operation.trace.completed_steps.tolist() == [1]
    assert r1_h3.operation.trace.completed_steps.tolist() == [3]
