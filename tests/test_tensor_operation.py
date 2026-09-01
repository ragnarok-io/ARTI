from __future__ import annotations

import pytest
import torch

import arti
from arti import mechanisms


def _spec() -> mechanisms.PortSpec:
    return mechanisms.PortSpec(
        canvas_tokens=4,
        tensor_shape=(2,),
        dim=3,
        tensor_to_canvas=(1, 3),
    )


def _matrix_spec() -> mechanisms.PortSpec:
    return mechanisms.PortSpec(
        canvas_tokens=4,
        tensor_shape=(1, 2),
        dim=3,
        tensor_to_canvas=(1, 3),
        folded_tensor_coordinates=((0, 0), (0, 1)),
    )


def _instruction(
    operation: list[int],
    source_plane: list[int],
    source_offset: list[int],
    destination_offset: list[int],
    *,
    device: torch.device | str = "cpu",
) -> mechanisms.TensorEditInstruction:
    def field(values: list[int]) -> torch.Tensor:
        return torch.tensor(values, dtype=torch.int64, device=device).unsqueeze(1)

    return mechanisms.TensorEditInstruction(
        operation=field(operation),
        source_plane=field(source_plane),
        source_offset=field(source_offset),
        destination_offset=field(destination_offset),
        active=torch.ones((len(operation), 1), dtype=torch.bool, device=device),
    )


def test_port_spec_rejects_ambiguous_maps() -> None:
    with pytest.raises(ValueError, match="injective"):
        mechanisms.PortSpec(4, (2,), 3, (1, 1))
    with pytest.raises(ValueError, match="out-of-range"):
        mechanisms.PortSpec(4, (2,), 3, (1, 4))


def test_port_spec_preserves_logical_tensor_shape_and_compiles_regions() -> None:
    spec = mechanisms.PortSpec(
        canvas_tokens=4,
        tensor_shape=(2, 3),
        dim=5,
        tensor_to_canvas=(0, 3),
        folded_tensor_coordinates=((0, 0), (1, 2)),
    )
    port = mechanisms.OperableTensorPort(spec, batch_size=2)

    assert spec.element_count == 6
    assert spec.region_offsets(slice(None), slice(1, 3)) == (1, 2, 4, 5)
    assert port.resolve().value.shape == (2, 2, 3, 5)
    assert port.resolve().mask.shape == (2, 2, 3)


def test_one_bank_member_copies_a_tensor_region_as_one_index_map() -> None:
    spec = mechanisms.PortSpec(
        canvas_tokens=4,
        tensor_shape=(2, 3),
        dim=2,
        tensor_to_canvas=(0,),
        folded_tensor_coordinates=((0, 0),),
    )
    source_offsets = spec.region_offsets(slice(None), slice(0, 2))
    destination_offsets = spec.region_offsets(slice(None), slice(1, 3))
    bank = mechanisms.TensorOperationBank(
        spec,
        candidate_count=1,
        key_dim=4,
        support_size=len(source_offsets),
        seed=31,
    )
    with torch.no_grad():
        bank.operands["active"].fill_(10)
        bank.operands["operation"].fill_(-10)
        bank.operands["operation"][..., int(mechanisms.EditOperation.COPY)] = 10
        bank.operands["source"].fill_(-10)
        bank.operands["destination"].fill_(-10)
        for lane, (source, destination) in enumerate(
            zip(source_offsets, destination_offsets, strict=True)
        ):
            bank.operands["source"][0, lane, spec.canvas_tokens + source] = 10
            bank.operands["destination"][0, lane, destination] = 10

    value = torch.arange(12, dtype=torch.float32).reshape(1, 2, 3, 2)
    snapshot = mechanisms.PortSnapshot(
        value,
        torch.ones((1, 2, 3), dtype=torch.bool),
        "default",
        0,
        0,
    )
    operation = mechanisms.TensorOperation(
        spec,
        mechanisms.TensorOperationSelector(spec, bank, estimator="hard", query_seed=37),
    )
    result = operation(torch.zeros((1, 4, 2)), snapshot)
    expected = value.clone()
    expected_flat = expected.reshape(1, 6, 2)
    source_flat = value.reshape(1, 6, 2)
    expected_flat[:, list(destination_offsets)] = source_flat[:, list(source_offsets)]

    torch.testing.assert_close(result.edit.value, expected, rtol=0, atol=0)
    assert result.edit.changed_support.all()


def test_operable_port_hotplug_retains_default_state() -> None:
    spec = _spec()
    port = mechanisms.OperableTensorPort(spec, batch_size=1)
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
    snapshot = mechanisms.PortSnapshot(backing, mask, "external", 4, 9)
    world_before = world.clone()
    backing_before = backing.clone()

    canvas = mechanisms.SharedCanvasFold(spec)(world, snapshot)

    assert canvas.values.shape == world.shape
    torch.testing.assert_close(canvas.values[0, 1], backing[0, 0])
    torch.testing.assert_close(canvas.values[0, 3], world[0, 3])
    torch.testing.assert_close(canvas.values[1, 1], world[1, 1])
    torch.testing.assert_close(canvas.values[1, 3], backing[1, 1])
    assert canvas.source_plane[0, 1].item() == int(mechanisms.CanvasSource.BACKING)
    assert canvas.source_index[0, 1].item() == 0
    assert canvas.source_plane[0, 3].item() == int(mechanisms.CanvasSource.WORLD)
    assert canvas.backing_epoch == 4
    assert canvas.step_index == 9
    torch.testing.assert_close(world, world_before)
    torch.testing.assert_close(backing, backing_before)


def test_reader_fold_is_partial_while_operation_query_sees_complete_backing() -> None:
    spec = mechanisms.PortSpec(
        canvas_tokens=4,
        tensor_shape=(2, 3),
        dim=3,
        tensor_to_canvas=(1, 3),
        folded_tensor_coordinates=((0, 2), (1, 2)),
    )
    world = torch.zeros((1, 4, 3))
    mask = torch.ones((1, 2, 3), dtype=torch.bool)
    first_value = torch.zeros((1, 2, 3, 3))
    second_value = first_value.clone()
    second_value.reshape(1, 6, 3)[:, 4, 0] = 7.0
    first = mechanisms.PortSnapshot(first_value, mask, "default", 0, 0)
    second = mechanisms.PortSnapshot(second_value, mask, "default", 0, 0)
    fold = mechanisms.SharedCanvasFold(spec)
    first_canvas = fold(world, first)
    second_canvas = fold(world, second)

    torch.testing.assert_close(first_canvas.values, second_canvas.values, rtol=0, atol=0)
    assert first_canvas.source_index[:, 1].item() == 2
    assert first_canvas.source_index[:, 3].item() == 5
    torch.testing.assert_close(first_canvas.backing_values, first_value)
    torch.testing.assert_close(second_canvas.backing_values, second_value)

    query = mechanisms.TensorOperationQuery(spec, key_dim=2, seed=29)
    with torch.no_grad():
        query.basis.zero_()
        query.mask_basis.zero_()
        query.backing_basis.zero_()
        query.backing_mask_basis.zero_()
        query.backing_basis[0, 4, 0] = 1.0
    first_query = query(first_canvas)
    second_query = query(second_canvas)

    assert first_query.tolist() == [[0.0, 0.0]]
    assert second_query.tolist() == [[7.0, 0.0]]


def test_shared_canvas_uses_explicit_empty_value_when_both_sources_are_hidden() -> None:
    spec = mechanisms.PortSpec(4, (2,), 3, (1, 3), empty_value=-2.0)
    world = torch.full((1, 4, 3), 9.0)
    world_mask = torch.tensor([[True, False, True, False]])
    backing = torch.full((1, 2, 3), 7.0)
    backing_mask = torch.zeros((1, 2), dtype=torch.bool)
    snapshot = mechanisms.PortSnapshot(backing, backing_mask, "default", 0, 0)

    canvas = mechanisms.SharedCanvasFold(spec)(world, snapshot, world_mask=world_mask)

    torch.testing.assert_close(canvas.values[:, 1], torch.full((1, 3), -2.0))
    torch.testing.assert_close(canvas.values[:, 3], torch.full((1, 3), -2.0))
    assert not canvas.mask[:, [1, 3]].any()
    assert (canvas.source_plane[:, [1, 3]] == int(mechanisms.CanvasSource.EMPTY)).all()


def test_tensor_edit_formula_applies_keep_copy_and_erase_from_snapshot() -> None:
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
    snapshot = mechanisms.PortSnapshot(backing, mask, "default", 0, 0)
    canvas = mechanisms.SharedCanvasFold(spec)(world, snapshot)
    instruction = _instruction(
        [int(mechanisms.EditOperation.KEEP), int(mechanisms.EditOperation.COPY), int(mechanisms.EditOperation.ERASE)],
        [-1, int(mechanisms.CanvasSource.BACKING), -1],
        [-1, 0, -1],
        [-1, 1, 0],
    )
    world_before = world.clone()
    backing_before = backing.clone()

    result = mechanisms.TensorEditFormula(spec)(canvas, snapshot, instruction)

    torch.testing.assert_close(result.value[0], backing[0])
    torch.testing.assert_close(result.value[1, 1], backing[1, 0])
    torch.testing.assert_close(result.value[2, 0], torch.zeros(3))
    assert result.mask[2, 0].item() is False
    torch.testing.assert_close(world, world_before)
    torch.testing.assert_close(backing, backing_before)
    assert result.value.data_ptr() != backing.data_ptr()


def test_copy_world_becomes_visible_only_after_next_port_advance() -> None:
    spec = _spec()
    port = mechanisms.OperableTensorPort(spec, batch_size=1)
    fold = mechanisms.SharedCanvasFold(spec)
    formula = mechanisms.TensorEditFormula(spec)
    world = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    before = fold(world, port.resolve())
    instruction = _instruction(
        [int(mechanisms.EditOperation.COPY)],
        [int(mechanisms.CanvasSource.WORLD)],
        [2],
        [0],
    )

    edited = formula(before, port.resolve(), instruction)

    torch.testing.assert_close(before.values[:, 1], world[:, 1])
    port.advance(edited.value, edited.mask)
    after = fold(world, port.resolve())
    torch.testing.assert_close(after.values[:, 1], world[:, 2])


def test_operation_field_copies_multiple_noncontiguous_sources_in_one_selection() -> None:
    spec = _spec()
    world = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    snapshot = mechanisms.PortSnapshot(
        torch.full((1, 2, 3), -1.0),
        torch.ones((1, 2), dtype=torch.bool),
        "default",
        0,
        0,
    )
    canvas = mechanisms.SharedCanvasFold(spec)(world, snapshot)
    instruction = mechanisms.TensorEditInstruction(
        operation=torch.tensor(
            [[int(mechanisms.EditOperation.COPY), int(mechanisms.EditOperation.COPY)]],
            dtype=torch.int64,
        ),
        source_plane=torch.tensor(
            [[int(mechanisms.CanvasSource.WORLD), int(mechanisms.CanvasSource.WORLD)]],
            dtype=torch.int64,
        ),
        source_offset=torch.tensor([[0, 3]], dtype=torch.int64),
        destination_offset=torch.tensor([[1, 0]], dtype=torch.int64),
        active=torch.tensor([[True, True]]),
    )

    result = mechanisms.TensorEditFormula(spec)(canvas, snapshot, instruction)

    torch.testing.assert_close(result.value[:, 0], world[:, 3])
    torch.testing.assert_close(result.value[:, 1], world[:, 0])
    assert result.changed_support.tolist() == [[True, True]]


def test_operation_field_swaps_backing_from_one_pre_state_snapshot() -> None:
    spec = _spec()
    backing = torch.tensor([[[1.0, 2.0, 3.0], [7.0, 8.0, 9.0]]])
    snapshot = mechanisms.PortSnapshot(
        backing,
        torch.ones((1, 2), dtype=torch.bool),
        "external",
        1,
        0,
    )
    canvas = mechanisms.SharedCanvasFold(spec)(torch.zeros((1, 4, 3)), snapshot)
    instruction = mechanisms.TensorEditInstruction(
        operation=torch.full((1, 2), int(mechanisms.EditOperation.COPY), dtype=torch.int64),
        source_plane=torch.full(
            (1, 2), int(mechanisms.CanvasSource.BACKING), dtype=torch.int64
        ),
        source_offset=torch.tensor([[1, 0]], dtype=torch.int64),
        destination_offset=torch.tensor([[0, 1]], dtype=torch.int64),
        active=torch.ones((1, 2), dtype=torch.bool),
    )

    result = mechanisms.TensorEditFormula(spec)(canvas, snapshot, instruction)

    torch.testing.assert_close(result.value[:, 0], backing[:, 1])
    torch.testing.assert_close(result.value[:, 1], backing[:, 0])
    torch.testing.assert_close(snapshot.value, backing)


def _configure_bank_field(
    bank: mechanisms.TensorOperationBank,
    *,
    source: int,
    destination: int,
) -> None:
    with torch.no_grad():
        bank.operands["active"].fill_(-10)
        bank.operands["active"][:, 0] = 10
        bank.operands["operation"].fill_(-10)
        bank.operands["operation"][:, :, int(mechanisms.EditOperation.KEEP)] = 10
        bank.operands["operation"][:, 0, int(mechanisms.EditOperation.COPY)] = 20
        bank.operands["source"].fill_(-10)
        bank.operands["source"][:, 0, source] = 10
        bank.operands["destination"].fill_(-10)
        bank.operands["destination"][:, 0, destination] = 10


def test_tensor_operation_bank_concat_preserves_members_and_isolates_influence() -> None:
    spec = _spec()
    bank_a = mechanisms.TensorOperationBank(
        spec,
        candidate_count=1,
        key_dim=5,
        bank_id="bank-a",
        seed=101,
    )
    bank_b = mechanisms.TensorOperationBank(
        spec,
        candidate_count=1,
        key_dim=5,
        bank_id="bank-b",
        seed=103,
    )
    _configure_bank_field(bank_a, source=0, destination=0)
    _configure_bank_field(bank_b, source=3, destination=1)
    concat = mechanisms.TensorOperationBank.concat(
        (bank_a, bank_b),
        name="bank-a-plus-bank-b",
        influences=(1.0, 0.0),
    )

    assert concat.member_ids == bank_a.member_ids + bank_b.member_ids
    assert concat.bank_ids == ("bank-a", "bank-b")
    assert concat.group_slices == ((0, 1), (1, 2))
    torch.testing.assert_close(concat.keys, torch.cat((bank_a.keys, bank_b.keys)))
    for name in concat.operands:
        torch.testing.assert_close(
            concat.operands[name],
            torch.cat((bank_a.operands[name], bank_b.operands[name])),
        )

    selector = mechanisms.TensorOperationSelector(spec, concat, query_seed=107)
    world = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    snapshot = mechanisms.OperableTensorPort(spec, batch_size=1).resolve()
    step = mechanisms.TensorOperation(spec, selector)(world, snapshot)

    assert step.decision.route.bank_indices.tolist() == [0]
    assert step.decision.route.local_indices.tolist() == [0]
    assert step.decision.route.bank_probabilities.tolist() == [[1.0, 0.0]]
    torch.testing.assert_close(step.edit.value[:, 0], world[:, 0])
    torch.testing.assert_close(step.edit.value[:, 1], snapshot.value[:, 1])


def test_tensor_operation_bank_concat_routes_gradient_only_to_enabled_bank() -> None:
    spec = _spec()
    bank_a = mechanisms.TensorOperationBank(
        spec, candidate_count=2, key_dim=5, bank_id="grad-a", seed=109
    )
    bank_b = mechanisms.TensorOperationBank(
        spec, candidate_count=2, key_dim=5, bank_id="grad-b", seed=113
    )
    concat = mechanisms.TensorOperationBank.concat(
        (bank_a, bank_b),
        name="gradient-union",
        influences=(1.0, 0.0),
    )
    selector = mechanisms.TensorOperationSelector(spec, concat, query_seed=127)
    canvas = mechanisms.SharedCanvasFold(spec)(
        torch.randn((4, 4, 3)),
        mechanisms.OperableTensorPort(spec, batch_size=4).resolve(),
    )

    decision = selector(canvas)
    loss = (
        decision.active_logits.square().mean()
        + decision.operation_logits.square().mean()
        + decision.source_logits.square().mean()
        + decision.destination_logits.square().mean()
    )
    loss.backward()

    split = bank_a.candidate_count
    assert concat.keys.grad is not None
    assert concat.keys.grad[:split].abs().sum() > 0
    assert concat.keys.grad[split:].abs().sum() == 0
    for operand in concat.operands.values():
        assert operand.grad is not None
        assert operand.grad[:split].abs().sum() > 0
        assert operand.grad[split:].abs().sum() == 0


def test_concatenated_operation_bank_arti_st_round_trip(tmp_path) -> None:
    spec = _matrix_spec()

    def build() -> mechanisms.TensorOperationLoop:
        first = mechanisms.TensorOperationBank(
            spec, candidate_count=2, key_dim=5, bank_id="saved-a", seed=131
        )
        second = mechanisms.TensorOperationBank(
            spec, candidate_count=3, key_dim=5, bank_id="saved-b", seed=137
        )
        bank = mechanisms.TensorOperationBank.concat(
            (first, second),
            name="saved-union",
            influences=(0.75, 1.25),
        )
        selector = mechanisms.TensorOperationSelector(spec, bank, query_seed=139)
        return mechanisms.TensorOperationLoop(
            mechanisms.TensorOperation(spec, selector, surrogate=mechanisms.TensorEditSurrogate(spec))
        )

    source = build()
    target = build()
    with torch.no_grad():
        source.operation.selector.bank.operands["active"].add_(0.5)
    world = torch.randn((3, 4, 3))
    snapshot = mechanisms.OperableTensorPort(spec, batch_size=3).resolve()
    expected = source(world, snapshot, schedule=mechanisms.TensorOperationSchedule(2))

    saved = arti.save(source, tmp_path / "concat-operation.arti.st")
    loaded = arti.load(saved.weights_path, model=target)
    actual = target(world, snapshot, schedule=mechanisms.TensorOperationSchedule(2))

    assert loaded.missing_keys == ()
    assert loaded.unexpected_keys == ()
    torch.testing.assert_close(actual.value, expected.value, rtol=0, atol=0)
    assert torch.equal(actual.trace.route_index, expected.trace.route_index)
    assert torch.equal(actual.trace.operation, expected.trace.operation)


@pytest.mark.skipif(not hasattr(torch, "compile"), reason="torch.compile is unavailable")
def test_complete_operation_field_supports_fullgraph_compile_and_backward() -> None:
    spec = _matrix_spec()
    bank = mechanisms.TensorOperationBank(spec, candidate_count=3, key_dim=5, seed=149)
    operation = mechanisms.TensorOperation(
        spec,
        mechanisms.TensorOperationSelector(spec, bank, query_seed=151),
        surrogate=mechanisms.TensorEditSurrogate(spec),
    )
    compiled = torch.compile(operation, backend="eager", fullgraph=True)
    world = torch.randn((2, 4, 3))
    snapshot = mechanisms.OperableTensorPort(spec, batch_size=2).resolve()

    eager = operation(world, snapshot)
    actual = compiled(world, snapshot)
    torch.testing.assert_close(actual.edit.value, eager.edit.value, rtol=0, atol=0)
    actual.edit.value.square().mean().backward()

    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in bank.parameters()
    )


def test_invalid_edit_fails_before_modifying_backing() -> None:
    spec = _spec()
    world = torch.zeros((1, 4, 3))
    backing = torch.ones((1, 2, 3))
    mask = torch.ones((1, 2), dtype=torch.bool)
    snapshot = mechanisms.PortSnapshot(backing, mask, "default", 0, 0)
    canvas = mechanisms.SharedCanvasFold(spec)(world, snapshot)
    invalid = _instruction(
        [int(mechanisms.EditOperation.COPY)],
        [int(mechanisms.CanvasSource.WORLD)],
        [9],
        [0],
    )
    before = backing.clone()

    with pytest.raises(ValueError, match="world source"):
        mechanisms.TensorEditFormula(spec)(canvas, snapshot, invalid)

    torch.testing.assert_close(backing, before)


@pytest.mark.parametrize(
    "device",
    ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))],
)
def test_tensor_edit_copy_is_differentiable_and_device_native(device: str) -> None:
    spec = _matrix_spec()
    world = torch.randn((2, 4, 3), device=device, requires_grad=True)
    backing = torch.randn((2, *spec.backing_shape), device=device, requires_grad=True)
    mask = torch.ones((2, *spec.backing_mask_shape), dtype=torch.bool, device=device)
    snapshot = mechanisms.PortSnapshot(backing, mask, "external", 1, 0)
    canvas = mechanisms.SharedCanvasFold(spec)(world, snapshot)
    instruction = _instruction(
        [int(mechanisms.EditOperation.COPY), int(mechanisms.EditOperation.COPY)],
        [int(mechanisms.CanvasSource.WORLD), int(mechanisms.CanvasSource.BACKING)],
        [2, 0],
        [0, 1],
        device=device,
    )
    result = mechanisms.TensorEditFormula(spec)(canvas, snapshot, instruction)

    result.value.sum().backward()

    assert world.grad is not None and torch.isfinite(world.grad).all()
    assert backing.grad is not None and torch.isfinite(backing.grad).all()
    assert result.value.device.type == device


def test_tensor_plane_component_identities_are_versioned() -> None:
    spec = _spec()
    port = mechanisms.OperableTensorPort(spec, batch_size=1)
    snapshot = port.resolve()
    world = torch.zeros((1, 4, 3))
    canvas = mechanisms.SharedCanvasFold(spec)(world, snapshot)
    instruction = _instruction([0], [-1], [-1], [-1])
    result = mechanisms.TensorEditFormula(spec)(canvas, snapshot, instruction)

    assert arti.component_ref(spec) == "arti/operable-tensor-port-spec@3"
    assert arti.component_ref(port) == "arti/operable-tensor-port@2"
    assert arti.component_ref(snapshot) == "arti/operable-tensor-snapshot@2"
    assert arti.component_ref(canvas) == "arti/shared-canvas@3"
    assert arti.component_ref(mechanisms.SharedCanvasFold(spec)) == "arti/shared-canvas-fold@3"
    assert arti.component_ref(instruction) == "arti/tensor-edit-instruction@3"
    assert arti.component_ref(result) == "arti/tensor-edit-result@3"


def _configured_copy_selector(spec: mechanisms.PortSpec) -> mechanisms.TensorOperationSelector:
    bank = mechanisms.TensorOperationBank(spec, candidate_count=1, key_dim=5, seed=3)
    with torch.no_grad():
        bank.operands["active"].fill_(-5.0)
        bank.operands["active"][:, 0] = 5.0
        bank.operands["operation"].fill_(-5.0)
        bank.operands["operation"][:, :, int(mechanisms.EditOperation.KEEP)] = 5.0
        bank.operands["operation"][:, 0, int(mechanisms.EditOperation.COPY)] = 10.0
        bank.operands["source"].fill_(-5.0)
        bank.operands["source"][:, 0, 2] = 5.0
        bank.operands["destination"].fill_(-5.0)
        bank.operands["destination"][:, 0, 0] = 5.0
    return mechanisms.TensorOperationSelector(spec, bank, estimator="hard", query_seed=7)


def test_tensor_operation_selector_uses_fixed_query_and_bank_owned_parameters() -> None:
    spec = _spec()
    bank = mechanisms.TensorOperationBank(spec, candidate_count=4, key_dim=6, seed=11)
    selector = mechanisms.TensorOperationSelector(spec, bank, query_seed=13)

    assert list(selector.query.parameters()) == []
    assert selector.query.operation_query_contract()["fixed"] is True
    assert all(name.startswith("bank.") for name, _ in selector.named_parameters())
    assert {name for name, _ in selector.named_parameters()} == {
        "bank.keys",
        "bank.operands.active",
        "bank.operands.destination",
        "bank.operands.operation",
        "bank.operands.source",
    }


def test_tensor_operation_selector_decodes_bank_selected_hard_instruction() -> None:
    spec = _spec()
    port = mechanisms.OperableTensorPort(spec, batch_size=2)
    world = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    canvas = mechanisms.SharedCanvasFold(spec)(world, port.resolve())
    selector = _configured_copy_selector(spec)

    control = selector(canvas)

    assert (control.instruction.operation[:, 0] == int(mechanisms.EditOperation.COPY)).all()
    assert (control.instruction.source_plane[:, 0] == int(mechanisms.CanvasSource.WORLD)).all()
    assert (control.instruction.source_offset[:, 0] == 2).all()
    assert (control.instruction.destination_offset[:, 0] == 0).all()
    assert control.instruction.active[:, 0].all()
    assert not control.instruction.active[:, 1:].any()
    assert control.route.hard_indices.shape == (2,)


def test_tensor_operation_selector_training_logits_update_only_bank_parameters() -> None:
    spec = _spec()
    port = mechanisms.OperableTensorPort(spec, batch_size=5)
    world = torch.randn((5, 4, 3), requires_grad=True)
    canvas = mechanisms.SharedCanvasFold(spec)(world, port.resolve())
    bank = mechanisms.TensorOperationBank(spec, candidate_count=4, key_dim=7, seed=17)
    selector = mechanisms.TensorOperationSelector(spec, bank, query_seed=19)
    control = selector(canvas)
    width = bank.field_spec.support_size
    operation_target = torch.arange(5 * width) % len(mechanisms.EditOperation)
    source_target = torch.arange(5 * width) % bank.field_spec.source_capacity
    destination_target = torch.arange(5 * width) % spec.element_count
    loss = (
        torch.nn.functional.cross_entropy(
            control.operation_logits.reshape(-1, len(mechanisms.EditOperation)),
            operation_target,
        )
        + torch.nn.functional.cross_entropy(
            control.source_logits.reshape(-1, bank.field_spec.source_capacity),
            source_target,
        )
        + torch.nn.functional.cross_entropy(
            control.destination_logits.reshape(-1, spec.element_count),
            destination_target,
        )
        + torch.nn.functional.binary_cross_entropy_with_logits(
            control.active_logits,
            torch.ones_like(control.active_logits),
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
    backing = torch.randn((batch, *spec.backing_shape))
    mask = torch.ones((batch, *spec.backing_mask_shape), dtype=torch.bool)
    snapshot = mechanisms.PortSnapshot(backing, mask, "default", 0, 0)
    bank = mechanisms.TensorOperationBank(spec, candidate_count=5, key_dim=7, seed=43)
    selector = mechanisms.TensorOperationSelector(spec, bank, query_seed=47)
    query_basis = selector.query.basis.clone()
    operation = mechanisms.TensorOperation(
        spec,
        selector,
        surrogate=mechanisms.TensorEditSurrogate(spec, temperature=0.8),
    )

    step = operation(world, snapshot)
    hard = mechanisms.TensorEditFormula(spec)(
        step.canvas,
        snapshot,
        step.decision.instruction,
    )

    assert torch.equal(step.edit.value, hard.value)
    assert torch.equal(step.edit.mask, hard.mask)
    next_snapshot = mechanisms.PortSnapshot(
        step.edit.value,
        step.edit.mask,
        snapshot.source,
        snapshot.backing_epoch,
        snapshot.step_index + 1,
    )
    next_world = torch.zeros_like(world)
    next_canvas = mechanisms.SharedCanvasFold(spec)(next_world, next_snapshot)
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


def test_tensor_edit_surrogate_inactive_rows_have_zero_bank_gradient() -> None:
    torch.manual_seed(49)
    spec = _spec()
    batch = 3
    world = torch.randn((batch, spec.canvas_tokens, spec.dim))
    snapshot = mechanisms.PortSnapshot(
        torch.randn((batch, *spec.backing_shape)),
        torch.ones((batch, *spec.backing_mask_shape), dtype=torch.bool),
        "default",
        0,
        0,
    )
    bank = mechanisms.TensorOperationBank(spec, candidate_count=5, key_dim=7, seed=51)
    operation = mechanisms.TensorOperation(
        spec,
        mechanisms.TensorOperationSelector(spec, bank, query_seed=53),
        surrogate=mechanisms.TensorEditSurrogate(spec),
    )

    result = operation(
        world,
        snapshot,
        active=torch.zeros(batch, dtype=torch.bool),
    )
    result.edit.value.square().sum().backward()

    for name, parameter in bank.named_parameters():
        assert parameter.grad is not None, name
        torch.testing.assert_close(
            parameter.grad,
            torch.zeros_like(parameter.grad),
            rtol=0,
            atol=0,
        )


@pytest.mark.parametrize(
    ("requested_steps", "stop_on_stable"),
    ((1, False), (4, True)),
)
def test_static_masked_inactive_capacity_does_not_change_bank_gradient(
    requested_steps: int, stop_on_stable: bool
) -> None:
    def run(*, steps: int, max_steps: int) -> dict[str, torch.Tensor]:
        torch.manual_seed(57)
        spec = _spec()
        bank = mechanisms.TensorOperationBank(spec, candidate_count=3, key_dim=5, seed=3)
        selector = mechanisms.TensorOperationSelector(spec, bank, query_seed=7)
        with torch.no_grad():
            selector.bank.operands["active"].fill_(5.0)
            selector.bank.operands["operation"].fill_(-5.0)
            selector.bank.operands["operation"][:, :, int(mechanisms.EditOperation.KEEP)] = 5.0
        loop = mechanisms.TensorOperationLoop(
            mechanisms.TensorOperation(
                spec,
                selector,
                surrogate=mechanisms.TensorEditSurrogate(spec),
            ),
            stop=mechanisms.TensorOperationStopPolicy(
                min_operation_steps=1,
                stop_on_stable=stop_on_stable,
            ),
            executor="static_masked",
        )
        world = torch.randn((2, spec.canvas_tokens, spec.dim))
        snapshot = mechanisms.PortSnapshot(
            torch.randn((2, *spec.backing_shape)),
            torch.ones((2, *spec.backing_mask_shape), dtype=torch.bool),
            "default",
            0,
            0,
        )
        target = torch.randn_like(snapshot.value)
        result = loop(
            world,
            snapshot,
            schedule=mechanisms.TensorOperationSchedule(
                steps,
                max_steps=max_steps,
            ),
        )
        torch.nn.functional.mse_loss(result.value, target, reduction="sum").backward()
        return {
            name: parameter.grad.detach().clone()
            for name, parameter in selector.bank.named_parameters()
        }

    actual = run(steps=requested_steps, max_steps=4)
    expected = run(steps=1, max_steps=1)

    assert actual.keys() == expected.keys()
    for name in actual:
        assert expected[name].abs().sum() > 0, name
        torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)


def test_multi_step_surrogate_reaches_bank_from_next_call_consumer_only() -> None:
    torch.manual_seed(53)
    spec = _spec()
    batch = 4
    world = torch.randn((batch, spec.canvas_tokens, spec.dim))
    backing = torch.randn((batch, *spec.backing_shape))
    snapshot = mechanisms.PortSnapshot(
        backing,
        torch.ones((batch, *spec.backing_mask_shape), dtype=torch.bool),
        "default",
        0,
        0,
    )
    bank = mechanisms.TensorOperationBank(spec, candidate_count=6, key_dim=8, seed=59)
    selector = mechanisms.TensorOperationSelector(spec, bank, query_seed=61)
    loop = mechanisms.TensorOperationLoop(
        mechanisms.TensorOperation(
            spec,
            selector,
            surrogate=mechanisms.TensorEditSurrogate(spec),
        )
    )

    result = loop(
        world,
        snapshot,
        schedule=mechanisms.TensorOperationSchedule(2),
    )
    next_snapshot = mechanisms.PortSnapshot(
        result.value,
        result.mask,
        snapshot.source,
        snapshot.backing_epoch,
        snapshot.step_index + 1,
    )
    consumer_world = torch.zeros_like(world)
    consumer = mechanisms.SharedCanvasFold(spec)(consumer_world, next_snapshot)
    target = torch.randn_like(consumer.values)
    torch.nn.functional.mse_loss(consumer.values, target).backward()

    assert result.trace.completed_steps.tolist() == [2] * batch
    assert result.trace.route_index.shape == (2, batch)
    assert any(
        parameter.grad is not None and parameter.grad.abs().sum() > 0
        for parameter in bank.parameters()
    )


def test_tensor_operation_surrogate_arti_st_round_trip(tmp_path) -> None:
    spec = _matrix_spec()

    def build() -> mechanisms.TensorOperationLoop:
        bank = mechanisms.TensorOperationBank(spec, candidate_count=5, key_dim=7, seed=71)
        selector = mechanisms.TensorOperationSelector(
            spec,
            bank,
            query_seed=73,
            temperature=0.6,
        )
        return mechanisms.TensorOperationLoop(
            mechanisms.TensorOperation(
                spec,
                selector,
                surrogate=mechanisms.TensorEditSurrogate(spec, temperature=0.6),
            ),
            stop=mechanisms.TensorOperationStopPolicy(
                min_operation_steps=1,
                stop_on_stable=True,
            ),
        )

    source = build()
    target = build()
    with torch.no_grad():
        source.operation.selector.bank.keys.add_(0.25)
    world = torch.randn(4, 4, 3)
    snapshot = mechanisms.OperableTensorPort(spec, batch_size=4).resolve()
    expected = source(
        world,
        snapshot,
        schedule=mechanisms.TensorOperationSchedule(4),
    )

    saved = arti.save(source, tmp_path / "tensor-operation.arti.st")
    loaded = arti.load(saved.weights_path, model=target)
    actual = target(
        world,
        snapshot,
        schedule=mechanisms.TensorOperationSchedule(4),
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
    port = mechanisms.OperableTensorPort(spec, batch_size=1)
    refine = RecordingRefine()
    operation = mechanisms.TensorOperation(spec, _configured_copy_selector(spec))
    invocation = mechanisms.TensorInvocation(
        spec,
        refine,
        mechanisms.TensorOperationLoop(operation),
    )
    world = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)

    result = invocation(
        world,
        port.resolve(),
        operation_schedule=mechanisms.TensorOperationSchedule(1),
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
    port = mechanisms.OperableTensorPort(spec, batch_size=1)
    canvas = mechanisms.SharedCanvasFold(spec)(torch.zeros((1, 4, 3)), port.resolve())
    decision = selector(canvas)
    operation = mechanisms.TensorOperation(spec, selector)
    loop = mechanisms.TensorOperationLoop(operation)
    invocation = mechanisms.TensorInvocation(spec, torch.nn.Identity(), loop)

    assert arti.component_ref(selector.query) == "arti/tensor-operation-query@4"
    assert arti.component_ref(selector.bank) == "arti/tensor-operation-bank@3"
    assert arti.component_ref(selector) == "arti/tensor-operation-selector@3"
    assert arti.component_ref(decision) == "arti/tensor-operation-decision@3"
    assert arti.component_ref(operation) == "arti/tensor-operation@3"
    assert arti.component_ref(loop) == "arti/tensor-operation-loop@3"
    assert arti.component_ref(invocation) == "arti/tensor-invocation@2"


def test_tensor_operation_loop_refolds_and_requeries_latest_shadow() -> None:
    spec = _spec()
    base_selector = _configured_copy_selector(spec)

    class RecordingSelector(mechanisms.TensorOperationSelector):
        def __init__(self) -> None:
            super().__init__(spec, base_selector.bank, estimator="hard", query=base_selector.query)
            self.seen_backings: list[torch.Tensor] = []

        def forward(self, canvas: mechanisms.SharedCanvas) -> mechanisms.TensorOperationDecision:
            self.seen_backings.append(canvas.backing_values.detach().clone())
            return super().forward(canvas)

    selector = RecordingSelector()
    loop = mechanisms.TensorOperationLoop(mechanisms.TensorOperation(spec, selector))
    port = mechanisms.OperableTensorPort(spec, batch_size=1)
    root = port.resolve()
    world = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)

    result = loop(
        world,
        root,
        schedule=mechanisms.TensorOperationSchedule(operation_steps=2),
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
    port = mechanisms.OperableTensorPort(spec, batch_size=batch)
    world = torch.arange(batch * 12, dtype=torch.float32).reshape(batch, 4, 3)
    loop = mechanisms.TensorOperationLoop(
        mechanisms.TensorOperation(spec, _configured_copy_selector(spec))
    )
    steps = torch.tensor([0, 1, 3], dtype=torch.int64)

    result = loop(
        world,
        port.resolve(),
        schedule=mechanisms.TensorOperationSchedule(steps),
    )

    assert result.trace.completed_steps.tolist() == [0, 1, 3]
    assert result.trace.attempted.shape == (3, batch)
    assert not result.trace.attempted[:, 0].any()
    assert result.trace.attempted[:, 1].tolist() == [True, False, False]
    assert result.trace.attempted[:, 2].all()
    assert result.trace.stop_reason[0].item() == int(
        mechanisms.TensorOperationStopReason.ZERO_STEPS
    )


def test_tensor_operation_stable_stop_is_post_transition_and_bounded() -> None:
    spec = _spec()
    selector = _configured_copy_selector(spec)
    with torch.no_grad():
        selector.bank.operands["operation"].fill_(-5.0)
        selector.bank.operands["operation"][:, :, int(mechanisms.EditOperation.KEEP)] = 5.0
    loop = mechanisms.TensorOperationLoop(
        mechanisms.TensorOperation(spec, selector),
        stop=mechanisms.TensorOperationStopPolicy(
            min_operation_steps=2,
            stop_on_stable=True,
        ),
    )
    port = mechanisms.OperableTensorPort(spec, batch_size=1)
    world = torch.zeros((1, 4, 3))

    result = loop(
        world,
        port.resolve(),
        schedule=mechanisms.TensorOperationSchedule(8),
    )

    assert result.trace.completed_steps.tolist() == [2]
    assert result.trace.stop_reason.tolist() == [int(mechanisms.TensorOperationStopReason.STABLE)]


def test_tensor_operation_static_masked_keeps_capacity_after_logical_stop() -> None:
    class CountingSelector(mechanisms.TensorOperationSelector):
        def __init__(self, spec: mechanisms.PortSpec) -> None:
            super().__init__(spec, _configured_copy_selector(spec).bank, query_seed=7)
            self.calls = 0

        def forward(self, canvas: mechanisms.SharedCanvas) -> mechanisms.TensorOperationDecision:
            self.calls += 1
            return super().forward(canvas)

    spec = _spec()
    selector = CountingSelector(spec)
    with torch.no_grad():
        selector.bank.operands["operation"].fill_(-5.0)
        selector.bank.operands["operation"][:, :, int(mechanisms.EditOperation.KEEP)] = 5.0
    loop = mechanisms.TensorOperationLoop(
        mechanisms.TensorOperation(spec, selector),
        stop=mechanisms.TensorOperationStopPolicy(
            min_operation_steps=1,
            stop_on_stable=True,
        ),
        executor="static_masked",
    )
    world = torch.zeros((1, 4, 3))
    snapshot = mechanisms.OperableTensorPort(spec, batch_size=1).resolve()

    result = loop(
        world,
        snapshot,
        schedule=mechanisms.TensorOperationSchedule(8),
    )

    assert selector.calls == 8
    assert result.trace.attempted.shape == (8, 1)
    assert result.trace.completed_steps.tolist() == [1]
    assert result.trace.attempted[:, 0].tolist() == [True] + [False] * 7


def test_tensor_operation_early_break_short_circuits_after_logical_stop() -> None:
    class CountingSelector(mechanisms.TensorOperationSelector):
        def __init__(self, spec: mechanisms.PortSpec) -> None:
            super().__init__(spec, _configured_copy_selector(spec).bank, query_seed=7)
            self.calls = 0

        def forward(self, canvas: mechanisms.SharedCanvas) -> mechanisms.TensorOperationDecision:
            self.calls += 1
            return super().forward(canvas)

    spec = _spec()
    selector = CountingSelector(spec)
    with torch.no_grad():
        selector.bank.operands["operation"].fill_(-5.0)
        selector.bank.operands["operation"][:, :, int(mechanisms.EditOperation.KEEP)] = 5.0
    loop = mechanisms.TensorOperationLoop(
        mechanisms.TensorOperation(spec, selector),
        stop=mechanisms.TensorOperationStopPolicy(
            min_operation_steps=1,
            stop_on_stable=True,
        ),
        executor="early_break",
    )
    world = torch.zeros((1, 4, 3))
    snapshot = mechanisms.OperableTensorPort(spec, batch_size=1).resolve()

    result = loop(
        world,
        snapshot,
        schedule=mechanisms.TensorOperationSchedule(8),
    )

    assert selector.calls == 1
    assert result.trace.attempted.shape == (1, 1)
    assert result.trace.completed_steps.tolist() == [1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_tensor_operation_early_break_rejects_cuda_host_sync() -> None:
    spec = _spec()
    loop = mechanisms.TensorOperationLoop(
        mechanisms.TensorOperation(
            spec,
            mechanisms.TensorOperationSelector(
                spec,
                mechanisms.TensorOperationBank(spec, candidate_count=2, key_dim=7, seed=71),
                query_seed=73,
            ),
        ),
        executor="early_break",
    ).cuda()
    world = torch.randn(2, 4, 3, device="cuda")
    snapshot = mechanisms.PortSnapshot(
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
    loop = mechanisms.TensorOperationLoop(
        mechanisms.TensorOperation(spec, _configured_copy_selector(spec)),
        executor="static_masked",
    )
    world = torch.zeros((2, 4, 3))
    snapshot = mechanisms.OperableTensorPort(spec, batch_size=2).resolve()
    schedule = mechanisms.TensorOperationSchedule(
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
    loop = mechanisms.TensorOperationLoop(
        mechanisms.TensorOperation(
            spec,
            mechanisms.TensorOperationSelector(
                spec,
                mechanisms.TensorOperationBank(
                    spec,
                    candidate_count=4,
                    key_dim=7,
                    seed=79,
                ),
                query_seed=83,
            ),
            surrogate=mechanisms.TensorEditSurrogate(spec),
        ),
        executor="static_masked",
    ).cuda()
    world = torch.randn(3, 4, 3, device="cuda")
    value = torch.zeros(3, 2, 3, device="cuda")
    mask = torch.ones(3, 2, dtype=torch.bool, device="cuda")
    snapshot = mechanisms.PortSnapshot(value, mask, "default", 0, 0)
    torch.cuda.synchronize()
    previous = torch.cuda.get_sync_debug_mode()

    try:
        torch.cuda.set_sync_debug_mode("error")
        result = loop(
            world,
            snapshot,
            schedule=mechanisms.TensorOperationSchedule(4),
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
    port = mechanisms.OperableTensorPort(spec, batch_size=1)
    world = torch.arange(12, dtype=torch.float32).reshape(1, 4, 3)
    operation = mechanisms.TensorOperationLoop(
        mechanisms.TensorOperation(spec, _configured_copy_selector(spec))
    )
    invocation = mechanisms.TensorInvocation(spec, AddOne(), operation)

    r1_h1 = invocation(
        world,
        port.resolve(),
        reader_schedule=mechanisms.ReaderRefineSchedule(1),
        operation_schedule=mechanisms.TensorOperationSchedule(1),
    )
    r3_h1 = invocation(
        world,
        port.resolve(),
        reader_schedule=mechanisms.ReaderRefineSchedule(3),
        operation_schedule=mechanisms.TensorOperationSchedule(1),
    )
    r1_h3 = invocation(
        world,
        port.resolve(),
        reader_schedule=mechanisms.ReaderRefineSchedule(1),
        operation_schedule=mechanisms.TensorOperationSchedule(3),
    )

    torch.testing.assert_close(r1_h1.operation.value, r3_h1.operation.value)
    torch.testing.assert_close(r1_h1.output, r1_h3.output)
    assert not torch.equal(r1_h1.output, r3_h1.output)
    assert r1_h1.operation.trace.completed_steps.tolist() == [1]
    assert r1_h3.operation.trace.completed_steps.tolist() == [3]
