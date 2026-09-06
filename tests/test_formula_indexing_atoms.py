from __future__ import annotations

import json

import pytest
import torch

import arti
from arti import mechanisms as m
from arti._formula_candidate_batch import _checked_plan
from arti._formula_grouped_training import execute_grouped_training, grouped_formula_training
from arti.formula_indexing import INDEX_ATOM_CLASSES


def _fabric(*outputs, **limits):
    program = m.FormulaProgram.build(outputs=outputs, limits=m.FormulaLimits(**limits))
    restored = m.FormulaProgram.from_dict(json.loads(json.dumps(program.to_dict())))
    assert restored.fingerprint == program.fingerprint
    assert arti.validate_component_provenance(arti.component_provenance(m.FormulaFabricV2(restored)))
    for instruction in restored.instructions:
        if instruction.atom_ref in INDEX_ATOM_CLASSES:
            atom = INDEX_ATOM_CLASSES[instruction.atom_ref](
                tuple(restored.slot_types[name] for name in instruction.input_slots), **dict(instruction.attributes),
            )
            assert arti.validate_component_provenance(arti.component_provenance(atom))
    return m.FormulaFabricV2(restored)


@pytest.mark.parametrize("count", (1, 7))
def test_axis_index_is_runtime_integer_data_and_gather_can_reorder_it(count):
    x = m.InputBinding("x", m.TensorType(("B", "N", "D"), ("B", "N", 2)))
    ids = m.InputBinding("ids", m.TensorType(("K",), ("K",), dtype="int64"))
    positions = m.axis_index(x, axis="N")
    fabric = _fabric(positions, m.gather_v2(positions, ids, axis="N", index_axis="K"))
    value = torch.randn(2, count, 2, requires_grad=True)
    index = torch.tensor([count - 1, 0, count - 1])
    actual = fabric(inputs={"x": value, "ids": index}, banks={}).values
    assert torch.equal(actual[0], torch.arange(count))
    assert torch.equal(actual[1], index)
    assert all(v.dtype == torch.int64 and not v.requires_grad for v in actual)
    integer = m.InputBinding("integer", m.TensorType(("N",), (3,), dtype="int64"))
    huge = torch.tensor([2**60, 2**60 + 1, 2**60 + 2])
    output = _fabric(m.gather_v2(integer, ids, axis="N", index_axis="K"))(
        inputs={"integer": huge, "ids": torch.tensor([1, 2, 1])}, banks={},
    ).values[0]
    assert torch.equal(output, huge[[1, 2, 1]])
    with pytest.raises(m.FormulaTypeError, match="int64"):
        m.gather(integer, ids, axis="N", index_axis="K")


@pytest.mark.parametrize("mode", ("eq", "ne", "lt", "le", "gt", "ge"))
def test_compare_and_boolean_atoms_preserve_integer_precision(mode):
    t = m.TensorType(("D",), (3,), dtype="int64")
    a, b = m.InputBinding("a", t), m.InputBinding("b", t)
    comparison = m.compare(a, b, mode=mode)
    complement = m.boolean_not(comparison)
    fabric = _fabric(comparison, complement, *(m.boolean_binary(comparison, complement, mode=op) for op in ("and", "or", "xor")))
    values = {"a": torch.tensor([2**60, 2**60 + 1, 2**60 + 2]), "b": torch.tensor([2**60 + 1] * 3)}
    output = fabric(inputs=values, banks={}).values
    expected = getattr(torch, mode)(values["a"], values["b"])
    assert torch.equal(output[0], expected) and torch.equal(output[1], ~expected)
    assert not output[2].any() and output[3].all() and output[4].all()


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64))
def test_scatter_add_repeated_indices_values_and_independent_update_gradients(dtype):
    t = m.TensorType(("B", "N", "D"), (2, 4, 2), dtype=str(dtype).removeprefix("torch."))
    it = m.TensorType(("B", "K"), (2, 3), dtype="int64")
    ut = m.TensorType(("B", "K", "D"), (2, 3, 2), dtype=t.dtype)
    base, ids, updates = m.InputBinding("base", t), m.InputBinding("ids", it), m.InputBinding("updates", ut)
    fabric = _fabric(m.scatter_add(base, ids, updates, axis="N", index_axis="K"))
    b = torch.randn(2, 4, 2, dtype=dtype, requires_grad=True)
    original = b.detach().clone()
    u = torch.randn(2, 3, 2, dtype=dtype, requires_grad=True)
    i = torch.tensor([[0, 0, 3], [2, 1, 2]])
    output = fabric(inputs={"base": b, "ids": i, "updates": u}, banks={}).values[0]
    compute = torch.float64 if dtype == torch.float64 else torch.float32
    expected = b.to(compute).scatter_add(1, i[..., None].expand_as(u), u.to(compute)).to(dtype)
    torch.testing.assert_close(output, expected)
    cotangent = torch.arange(16).reshape(2, 4, 2).to(dtype)
    gb, gu = torch.autograd.grad((output * cotangent).sum(), (b, u))
    torch.testing.assert_close(gb, cotangent)
    torch.testing.assert_close(gu, cotangent.gather(1, i[..., None].expand_as(u)))
    assert torch.equal(b.detach(), original)
    with pytest.raises(m.FormulaBindingError, match="unique"):
        _fabric(m.scatter(base, ids, updates, axis="N", index_axis="K"))(
            inputs={"base": b, "ids": i, "updates": u}, banks={},
        )


def _segment_program(dtype, mode):
    t = m.TensorType(("B", "N", "D"), ("B", "N", 2), dtype=str(dtype).removeprefix("torch."))
    x = m.InputBinding("x", t)
    ids = m.InputBinding("ids", m.TensorType(("B", "N"), ("B", "N"), dtype="int64"))
    mask = m.InputBinding("mask", m.TensorType(t.axis_names, t.sizes, dtype="boolean"))
    return _fabric(m.segment(x, ids, mask, axis="N", segment_axis="G", num_segments=4, mode=mode))


def _segment_values(dtype):
    x = torch.tensor([[[-2, -3], [-2, -4], [-5, 1], [1000, -1000], [3, 2]]], dtype=dtype, requires_grad=True)
    ids = torch.tensor([[1, 1, 0, -99, 0]])
    mask = torch.tensor([[[True, True], [True, False], [True, True], [False, False], [False, True]]])
    return {"x": x, "ids": ids, "mask": mask}


def _segment_reference(inputs, mode):
    value = inputs["x"]
    compute = value.double() if value.dtype == torch.float64 else value.float()
    ids, mask = inputs["ids"], inputs["mask"]
    batches = []
    for b in range(value.shape[0]):
        features = []
        for d in range(value.shape[-1]):
            outputs = []
            soft = compute[b, :, d] * 0
            for group in range(4):
                selected = (ids[b] == group) & mask[b, :, d]
                indices = torch.where(selected)[0]
                members = compute[b, :, d][selected]
                if mode == "softmax":
                    soft = soft.index_copy(0, indices, torch.softmax(members, 0))
                elif members.numel():
                    outputs.append(getattr(torch, mode)(members))
                else:
                    outputs.append(compute[b, :, d].sum() * 0)
            features.append(soft if mode == "softmax" else torch.stack(outputs))
        batches.append(torch.stack(features, -1))
    return torch.stack(batches).to(value.dtype)


@pytest.mark.parametrize("mode", ("sum", "mean", "amax", "softmax"))
@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64))
def test_segment_mask_negative_groups_ties_empty_groups_and_gradients(mode, dtype):
    inputs = _segment_values(dtype)
    actual = _segment_program(dtype, mode)(inputs=inputs, banks={}).values[0]
    expected = _segment_reference(inputs, mode)
    torch.testing.assert_close(actual, expected)
    cotangent = torch.arange(actual.numel(), dtype=actual.dtype).reshape_as(actual) + 1
    ga, = torch.autograd.grad((actual * cotangent).sum(), inputs["x"], retain_graph=True)
    ge, = torch.autograd.grad((expected * cotangent).sum(), inputs["x"])
    assert torch.isfinite(ga).all()
    torch.testing.assert_close(ga, ge)
    assert torch.equal(ga[~inputs["mask"]], torch.zeros_like(ga[~inputs["mask"]]))


@pytest.mark.parametrize("kind", ("gather", "scatter", "segment"))
def test_invalid_active_indices_fail_checked_rows_without_out_of_bounds_access(kind):
    x = m.InputBinding("x", m.TensorType(("N",), (3,), dtype="float32"))
    ids = m.InputBinding("ids", m.TensorType(("K",), (2,), dtype="int64"))
    inputs = {"x": torch.ones(3), "ids": torch.tensor([-1, 20])}
    if kind == "gather":
        output = m.gather_v2(x, ids, axis="N", index_axis="K")
    elif kind == "scatter":
        u = m.InputBinding("u", m.TensorType(("K",), (2,), dtype="float32"))
        inputs["u"] = torch.ones(2)
        output = m.scatter_add(x, ids, u, axis="N", index_axis="K")
    else:
        ids = m.InputBinding("ids", m.TensorType(("N",), (3,), dtype="int64"))
        mask = m.InputBinding("mask", m.TensorType(("N",), (3,), dtype="boolean"))
        inputs.update(ids=torch.tensor([0, 2, -1]), mask=torch.tensor([True, True, False]))
        output = m.segment(x, ids, mask, axis="N", segment_axis="G", num_segments=2)
    fabric = _fabric(output)
    prepared = fabric.bind_tensors(inputs=inputs, banks={})
    result, valid = fabric.execution_plan().forward_checked(prepared)
    assert not valid and torch.isfinite(result[0]).all()
    with pytest.raises(m.FormulaBindingError, match="active indices"):
        fabric(inputs=inputs, banks={})


@pytest.mark.parametrize("backend", ("eager", "aot_eager"))
@pytest.mark.parametrize("mode", ("sum", "mean", "amax", "softmax"))
def test_grouped_segments_share_values_gradients_and_invalid_row_validity(backend, mode):
    fabric = _segment_program(torch.float64, mode)
    plan = _checked_plan(fabric.program)
    assert plan is not None
    inputs = [_segment_values(torch.float64) for _ in range(2)]
    inputs[1]["ids"] = torch.tensor([[9, 1, 0, -99, 0]])
    rows = tuple(fabric.bind_tensors(inputs=v, banks={}) for v in inputs)
    expected = _segment_reference(inputs[0], mode)
    with grouped_formula_training(backend=backend):
        outputs, valid = execute_grouped_training(plan, rows)
        assert valid.tolist() == [True, False]
        torch.testing.assert_close(outputs[0][0], expected)
        actual = torch.autograd.grad(outputs[0][0].square().sum(), tuple(v["x"] for v in inputs), allow_unused=True)
    expected_grad, = torch.autograd.grad(expected.square().sum(), inputs[0]["x"])
    torch.testing.assert_close(actual[0], expected_grad)
    assert actual[1] is None


@pytest.mark.parametrize("backend", ("eager", "aot_eager"))
def test_computed_mask_does_not_invent_threshold_gradient_or_wrong_select_dtype(backend):
    t = m.TensorType(("D",), (4,), dtype="floating")
    x, threshold, zero = (m.InputBinding(name, t) for name in ("x", "threshold", "zero"))
    mask = m.compare(x, threshold, mode="gt")
    output = m.select(mask, x, zero)
    fabric = _fabric(m.scale(output, output), mask, m.axis_index(threshold, axis="D"))
    values = {"x": torch.tensor([-2., -1., 1., 2.], dtype=torch.float64, requires_grad=True),
              "threshold": torch.zeros(4, dtype=torch.float64, requires_grad=True),
              "zero": torch.zeros(4, dtype=torch.float64, requires_grad=True)}
    row = fabric.bind_tensors(inputs=values, banks={})
    with grouped_formula_training(backend=backend):
        outputs, valid = execute_grouped_training(_checked_plan(fabric.program), (row,))
        assert valid.all() and outputs[0][0].dtype == torch.float64
        assert not outputs[0][1].requires_grad and not outputs[0][2].requires_grad
        gx, gt, gz = torch.autograd.grad(outputs[0][0].sum(), tuple(values.values()), allow_unused=True)
    assert gt is None
    torch.testing.assert_close(gx, torch.tensor([0., 0., 2., 4.], dtype=torch.float64))
    torch.testing.assert_close(gz, torch.zeros(4, dtype=torch.float64))


@pytest.mark.parametrize("mode", ("sum", "mean", "amax", "softmax"))
def test_all_masked_segments_are_zero_with_zero_value_gradient(mode):
    inputs = _segment_values(torch.float32)
    inputs["mask"].zero_()
    inputs["ids"].fill_(-10)
    output = _segment_program(torch.float32, mode)(inputs=inputs, banks={}).values[0]
    assert torch.equal(output, torch.zeros_like(output))
    gradient, = torch.autograd.grad(output.sum(), inputs["x"])
    assert torch.equal(gradient, torch.zeros_like(gradient))


def test_index_output_and_select_storage_bytes_are_admitted_as_actual_dtypes():
    x = m.InputBinding("x", m.TensorType(("D",), (4,), dtype="float16"))
    fabric = _fabric(m.axis_index(x, axis="D"), max_tensor_bytes=16)
    with pytest.raises(m.FormulaBindingError, match="byte"):
        fabric.bind_tensors(inputs={"x": torch.ones(4, dtype=torch.float16)}, banks={})
    t = m.TensorType(("D",), (4,), dtype="floating")
    a, b = m.InputBinding("a", t), m.InputBinding("b", t)
    mask = m.InputBinding("mask", m.TensorType(("D",), (4,), dtype="boolean"))
    fabric = _fabric(m.select(mask, a, b), max_working_bytes=180)
    with pytest.raises(m.FormulaBindingError, match="working"):
        fabric.bind_tensors(inputs={"a": torch.ones(4, dtype=torch.float64),
                                    "b": torch.ones(4, dtype=torch.float64), "mask": torch.ones(4, dtype=torch.bool)}, banks={})


def test_grouped_attention_example_has_real_bank_gradients_and_reload():
    from examples.formula_segment_attention import build_program
    program, weight = build_program(dim=2, groups=4, dtype="float64")
    fabric = m.FormulaFabricV2(m.FormulaProgram.from_dict(program.to_dict()))
    inputs = _segment_values(torch.float64)
    inputs["mask"] = inputs["mask"].all(-1)
    p = torch.tensor([0.2, -0.3], dtype=torch.float64, requires_grad=True)
    output = fabric(inputs=inputs, banks={"score_weight": weight.bind(p)}).values
    reference = torch.stack([
        (torch.softmax(inputs["x"][0, (inputs["ids"][0] == g) & inputs["mask"][0]] @ p, 0)[:, None]
         * inputs["x"][0, (inputs["ids"][0] == g) & inputs["mask"][0]]).sum(0)
        for g in range(4)
    ])[None]
    torch.testing.assert_close(output[-1], reference)
    ga = torch.autograd.grad(output[-1].square().sum(), (inputs["x"], p), retain_graph=True)
    ge = torch.autograd.grad(reference.square().sum(), (inputs["x"], p))
    for a, e in zip(ga, ge, strict=True):
        torch.testing.assert_close(a, e)
        assert torch.isfinite(a).all()


def test_integer_boolean_and_select_outputs_reach_real_typed_pools_on_cpu():
    from arti._formula_device_dispatch import FormulaDeviceDispatchLayout, FormulaDeviceNumericalDispatch, formula_device_dispatch_groups
    from arti._formula_device_frames import FormulaDeviceFrameKernel
    from arti._formula_device_pools import FormulaDevicePoolLayout

    t = m.TensorType(("B", "N"), ("B", 3), dtype="float32")
    x = m.InputBinding("x", t)
    gain = m.BankBinding("gain", "arti/indexing-test@1", "gain", t)
    positions = m.broadcast(m.axis_index(x, axis="N"), output_axes=t.axis_names, output_sizes=t.sizes)
    mask = m.compare(x, gain, mode="ge")
    program = m.FormulaProgram.build(outputs=(positions, mask, m.select(mask, x, gain)))
    candidate = m.FormulaProgramTensorCandidateV4(m.FormulaProgramCandidateV3(
        "numeric", program, input_slots={"x": "x"},
        output_slots=dict(zip(program.outputs, ("positions", "mask", "selected"), strict=True)),
        operands={"gain": torch.zeros(1, 3)},
    ), plastic_bank_slot="gain", bank_owner_id="numeric")
    query = m.FormulaProgramQueryV5(slot_ids=("x", "positions", "mask", "selected"), candidates=(candidate,),
                                  terminal_slots={"answer": "selected"}, max_steps=1)
    kernel = FormulaDeviceFrameKernel.from_query(query)
    layout = FormulaDevicePoolLayout.from_samples(tuple(torch.zeros(1, 3, dtype=d) for d in (torch.float32, torch.int64, torch.bool)), 4)
    banks_layout = FormulaDevicePoolLayout.from_samples((torch.zeros(1, 3),), 4)
    dispatch = FormulaDeviceNumericalDispatch(query, kernel)
    dispatch.prepare_typed_pools_(layout, banks_layout)
    state = kernel.initial_state(1, torch.tensor([[0, -1, -1, -1]]), bank_value_handles=torch.tensor([0]))
    data, banks = layout.allocate("cpu"), banks_layout.allocate("cpu")
    data[0][0] = torch.tensor([[-1., 0., 2.]])
    packet = FormulaDeviceDispatchLayout(formula_device_dispatch_groups(query)[0])(torch.tensor([[0]]))
    result = dispatch(state, packet, data, banks)
    assert result.numeric_valid.all() and not result.overflow
    expected = (torch.tensor([[0, 1, 2]]), torch.tensor([[False, True, True]]), torch.tensor([[0., 0., 2.]]))
    for head, value in enumerate(expected):
        bucket = layout.index(value)
        assert result.output_present[bucket][0, head]
        assert torch.equal(result.output_values[bucket][0, head], value)


def test_plain_prepared_plan_executes_valid_indices_and_rejects_invalid_ones():
    x = m.InputBinding("x", m.TensorType(("N",), (3,), dtype="floating"))
    ids = m.InputBinding("ids", m.TensorType(("K",), (2,), dtype="int64"))
    fabric = _fabric(m.axis_index(x, axis="N"), m.gather_v2(x, ids, axis="N", index_axis="K"))
    plan = fabric.execution_plan()
    inputs = {"x": torch.tensor([1., 2., 3.]), "ids": torch.tensor([2, 0])}
    prepared = fabric.bind_tensors(inputs=inputs, banks={})
    output = plan(prepared)
    assert torch.equal(output[0], torch.arange(3))
    torch.testing.assert_close(output[1], torch.tensor([3., 1.]))
    invalid = fabric.bind_tensors(inputs={**inputs, "ids": torch.tensor([3, -1])}, banks={})
    with pytest.raises(m.FormulaBindingError, match="active indices"):
        plan(invalid)
