from __future__ import annotations

import json

import pytest
import torch
import torch.nn.functional as F

import arti
from arti import mechanisms as m
from arti._formula_candidate_batch import _checked_plan
from arti._formula_grouped_training import execute_grouped_training, grouped_formula_training


def _fabric(*outputs, **limits):
    program = m.FormulaProgram.build(outputs=outputs, limits=m.FormulaLimits(**limits))
    restored = m.FormulaProgram.from_dict(json.loads(json.dumps(program.to_dict())))
    assert restored.fingerprint == program.fingerprint
    fabric = m.FormulaFabricV2(restored)
    assert arti.validate_component_provenance(arti.component_provenance(fabric))
    return fabric


def _window(x, **kwargs):
    return m.window(x, axis="N", output_axis="P", window_axis="K", kernel_size=3, **kwargs)


def _reference(x, dim, kernel, stride, dilation, padding):
    count = (x.shape[dim] + sum(padding) - dilation * (kernel - 1) - 1) // stride + 1
    zero = x.select(dim, 0) * 0
    return torch.stack([torch.stack([
        x.select(dim, p * stride - padding[0] + k * dilation)
        if 0 <= p * stride - padding[0] + k * dilation < x.shape[dim] else zero
        for k in range(kernel)
    ], dim=dim) for p in range(count)], dim=dim)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64, torch.int64, torch.bool))
@pytest.mark.parametrize("axis", (0, 1, 2))
def test_window_named_axes_noncontiguous_values_overlap_gradients_and_atom_reload(dtype, axis):
    x = torch.arange(2 * 3 * 7).reshape(2, 3, 7).movedim(2, axis).to(dtype)
    if dtype.is_floating_point:
        x.requires_grad_()
    axes = ["B", "C"]
    axes.insert(axis, "N")
    t = m.TensorType(tuple(axes), tuple(x.shape), dtype="boolean" if dtype == torch.bool else str(dtype)[6:])
    attrs = dict(axis="N", output_axis="P", window_axis="K", kernel_size=3, stride=2, dilation=2, padding=(2, 3))
    atom = m.WindowAtom(t, **attrs)
    assert arti.validate_component_provenance(arti.component_provenance(atom))
    fabric = _fabric(m.window(m.InputBinding("x", t), **attrs))
    prepared = fabric.bind_tensors(inputs={"x": x}, banks={})
    expected = _reference(x, axis, 3, 2, 2, (2, 3))
    for output in (atom(x), fabric(inputs={"x": x}, banks={}).values[0],
                   fabric.execution_plan()(prepared)[0], fabric.execution_plan().forward_checked(prepared)[0][0]):
        assert output.dtype == x.dtype and torch.equal(output, expected)
    if dtype.is_floating_point:
        cotangent = torch.linspace(-1, 1, expected.numel()).reshape(expected.shape).to(dtype)
        actual = fabric.execution_plan()(prepared)[0]
        ga, = torch.autograd.grad((actual * cotangent).sum(), x, retain_graph=True)
        ge, = torch.autograd.grad((expected * cotangent).sum(), x)
        torch.testing.assert_close(ga, ge)


def test_no_padding_is_a_read_only_overlapping_view():
    x = torch.arange(7., requires_grad=True)
    atom = m.WindowAtom(m.TensorType(("N",), (7,)), axis="N", output_axis="P", window_axis="K", kernel_size=3)
    result = atom(x)
    assert result.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
    assert result.shape == (5, 3)
    result.sum().backward()
    torch.testing.assert_close(x.grad, torch.tensor([1., 2., 3., 3., 3., 2., 1.]))


@pytest.mark.parametrize("count", (1, 5, 11))
def test_derived_symbol_reaches_downstream_broadcast_and_prepared_execution(count):
    x = m.InputBinding("x", m.TensorType(("B", "N", "C"), ("B", "N", 2)))
    patch = _window(x, padding=(2, 2), output_size="P")
    positions = m.axis_index(patch, axis="P")
    fabric = _fabric(patch, m.broadcast(positions, output_axes=("B", "P"), output_sizes=("B", "P")))
    value = torch.randn(2, count, 2)
    prepared = fabric.bind_tensors(inputs={"x": value}, banks={})
    actual = fabric.execution_plan()(prepared)
    assert actual[0].shape == (2, count + 2, 3, 2)
    assert torch.equal(actual[1], torch.arange(count + 2).expand(2, -1))
    expected = _reference(value, 1, 3, 1, 1, (2, 2))
    torch.testing.assert_close(actual[0], expected)


@pytest.mark.parametrize("kwargs", ({"stride": 0}, {"dilation": True}, {"padding": (-1, 0)},
                                    {"padding": (1,)}, {"output_size": 17}))
def test_invalid_static_window_contract(kwargs):
    x = m.InputBinding("x", m.TensorType(("N",), (7,)))
    with pytest.raises(m.FormulaTypeError):
        _window(x, **kwargs)


def test_dynamic_geometry_and_symbol_conflicts_fail_before_execution(monkeypatch):
    import arti.formula_window as module
    x = m.InputBinding("x", m.TensorType(("N",), ("N",)))
    with pytest.raises(m.FormulaTypeError, match="explicit output_size"):
        _window(x)
    patch = _window(x, output_size="P")
    reference = m.InputBinding("reference", m.TensorType(("P",), ("P",)))
    fabric = _fabric(patch, m.axis_index(reference, axis="P"))
    monkeypatch.setattr(module, "execute_window", lambda *args: pytest.fail("allocated before invalid shape rejection"))
    with pytest.raises(m.FormulaBindingError, match="at least one"):
        fabric.bind_tensors(inputs={"x": torch.ones(1), "reference": torch.ones(1)}, banks={})
    with pytest.raises(m.FormulaBindingError, match="conflicts"):
        fabric(inputs={"x": torch.ones(7), "reference": torch.ones(9)}, banks={})
    atom = m.WindowAtom(x.value_type, **dict(patch.attributes))
    with pytest.raises(m.FormulaBindingError, match="at least one"):
        atom(torch.ones(1))


def test_logical_window_and_padded_base_are_budgeted_before_native_allocation(monkeypatch):
    import arti.formula_window as module
    x = m.InputBinding("x", m.TensorType(("N",), ("N",)))
    fabric = _fabric(_window(x, output_size="P"), max_tensor_elements=10)
    monkeypatch.setattr(module, "execute_window", lambda *args: pytest.fail("unexpected allocation"))
    with pytest.raises(m.FormulaBindingError, match="element limit"):
        fabric(inputs={"x": torch.ones(7)}, banks={})
    # 4 input + 4 working input + 8 output budgets + 804 padded base bytes.
    x = m.InputBinding("x", m.TensorType(("N",), (1,)))
    patch = m.window(x, axis="N", output_axis="P", window_axis="K", kernel_size=1,
                     stride=201, padding=(100, 100))
    fabric = _fabric(patch, max_working_bytes=800)
    with pytest.raises(m.FormulaBindingError, match="working byte"):
        fabric.bind_tensors(inputs={"x": torch.ones(1)}, banks={})
    second = m.window(x, axis="N", output_axis="Q", window_axis="L", kernel_size=1,
                      stride=201, padding=(99, 101))
    fabric = _fabric(patch, second, max_working_bytes=900)
    with pytest.raises(m.FormulaBindingError, match="working byte"):
        fabric.bind_tensors(inputs={"x": torch.ones(1)}, banks={})
    # The one-element view keeps its 201-element padded base alive downstream.
    expanded = m.broadcast(patch, output_axes=("P", "K", "D"), output_sizes=(1, 1, 300))
    fabric = _fabric(expanded, max_working_bytes=3000)
    with pytest.raises(m.FormulaBindingError, match="working byte"):
        fabric.bind_tensors(inputs={"x": torch.ones(1)}, banks={})


@pytest.mark.parametrize("mode", ("depthwise", "shared", "local"))
@pytest.mark.parametrize("backend", ("eager", "aot_eager"))
def test_window_contract_local_program_value_and_bank_gradients(mode, backend):
    from examples.formula_local_computation import build_program
    args = dict(channels=2, kernel_size=3, stride=2, dilation=2, padding=(2, 1), dtype="float64")
    if mode != "depthwise":
        args["out_channels"] = 3
    if mode == "local":
        args["position_size"] = 3
    program, bindings = build_program(**args)
    fabric = m.FormulaFabricV2(m.FormulaProgram.from_dict(json.loads(json.dumps(program.to_dict()))))
    assert arti.validate_component_provenance(arti.component_provenance(fabric))
    x = torch.randn(2, 7, 2, dtype=torch.float64, requires_grad=True)
    weights = {k: torch.randn(*binding.value_type.sizes, dtype=torch.float64).requires_grad_()
               for k, binding in bindings.items() if k != "negative_unit"}
    weights["negative_unit"] = torch.tensor(-1., dtype=torch.float64)
    prepared = fabric.bind_tensors(inputs={"x": x}, banks={k: bindings[k].bind(v) for k, v in weights.items()})
    with grouped_formula_training(backend=backend):
        results, valid = execute_grouped_training(_checked_plan(program), (prepared, prepared))
        assert valid.tolist() == [True, True]
        actual = results[0][0]
        gradient = torch.autograd.grad(actual.square().sum(), (x, weights["kernel"], weights["slope"]), retain_graph=True)
    w = weights["kernel"]
    if mode == "depthwise":
        pre = F.conv1d(F.pad(x.transpose(1, 2), (2, 1)), w.T[:, None], stride=2, dilation=2, groups=2).transpose(1, 2)
    elif mode == "shared":
        pre = F.conv1d(F.pad(x.transpose(1, 2), (2, 1)), w.permute(2, 1, 0), stride=2, dilation=2).transpose(1, 2)
    else:
        patch = _reference(x, 1, 3, 2, 2, (2, 1))
        pre = torch.einsum("bpkc,pkco->bpo", patch, w)
    expected = pre.relu() + weights["slope"] * (pre - pre.relu())
    torch.testing.assert_close(actual, expected)
    reference = torch.autograd.grad(expected.square().sum(), (x, w, weights["slope"]))
    for ga, ge in zip(gradient, reference, strict=True):
        torch.testing.assert_close(ga, ge)
        assert ga.isfinite().all()


def test_two_windows_form_a_two_dimensional_depthwise_neighborhood():
    t = m.TensorType(("B", "H", "W", "C"), (2, "H", "W", 2), dtype="float64")
    x = m.InputBinding("x", t)
    kh = m.window(x, axis="H", output_axis="PH", window_axis="KH", kernel_size=2, output_size="PH")
    patch = m.window(kh, axis="W", output_axis="PW", window_axis="KW", kernel_size=3, output_size="PW", padding=(1, 0))
    kernel = m.BankBinding("kernel", "arti/window-test@1", "kernel", m.TensorType(("KH", "KW", "C"), (2, 3, 2), dtype="float64"))
    output = m.reduce_tensor(m.contract(patch, kernel, reduce_axes=(("KH", "KH"),)), axis="KW", mode="sum")
    fabric = _fabric(output)
    value = torch.randn(2, 4, 5, 2, dtype=torch.float64, requires_grad=True)
    weight = torch.randn(2, 3, 2, dtype=torch.float64, requires_grad=True)
    actual = fabric(inputs={"x": value}, banks={"kernel": kernel.bind(weight)}).values[0]
    expected = F.conv2d(F.pad(value.permute(0, 3, 1, 2), (1, 0, 0, 0)), weight.permute(2, 0, 1)[:, None], groups=2).permute(0, 2, 3, 1)
    torch.testing.assert_close(actual, expected)
    ga = torch.autograd.grad(actual.square().sum(), (value, weight), retain_graph=True)
    ge = torch.autograd.grad(expected.square().sum(), (value, weight))
    for a, e in zip(ga, ge, strict=True):
        torch.testing.assert_close(a, e)


def test_window_outputs_reach_real_heterogeneous_device_pools_on_cpu():
    from arti._formula_device_dispatch import FormulaDeviceDispatchLayout, FormulaDeviceNumericalDispatch, formula_device_dispatch_groups
    from arti._formula_device_frames import FormulaDeviceFrameKernel
    from arti._formula_device_pools import FormulaDevicePoolLayout

    t = m.TensorType(("B", "N", "C"), ("B", "N", 2), dtype="float32")
    x = m.InputBinding("x", t)
    gain = m.BankBinding("gain", "arti/window-test@1", "gain", m.TensorType(("C",), (2,), dtype="float32"))
    patch = _window(x, output_size="P", padding=(1, 0))
    program = m.FormulaProgram.build(outputs=(m.scale(patch, gain),))
    candidate = m.FormulaProgramTensorCandidateV4(m.FormulaProgramCandidateV3(
        "window", program, input_slots={"x": "x"}, output_slots={program.outputs[0]: "patch"},
        operands={"gain": torch.tensor([2., -1.])},
    ), plastic_bank_slot="gain", bank_owner_id="local")
    query = m.FormulaProgramQueryV5(slot_ids=("x", "patch"), candidates=(candidate,), terminal_slots={"answer": "patch"}, max_steps=1)
    kernel = FormulaDeviceFrameKernel.from_query(query)
    layout = FormulaDevicePoolLayout.from_samples((torch.zeros(1, 5, 2), torch.zeros(1, 4, 3, 2)), 4)
    banks_layout = FormulaDevicePoolLayout.from_samples((torch.zeros(2),), 4)
    dispatch = FormulaDeviceNumericalDispatch(query, kernel)
    dispatch.prepare_typed_pools_(layout, banks_layout)
    state = kernel.initial_state(1, torch.tensor([[0, -1]]), bank_value_handles=torch.tensor([0]))
    data, banks = layout.allocate("cpu"), banks_layout.allocate("cpu")
    value = torch.arange(10.).reshape(1, 5, 2)
    data[layout.index(value)][0] = value
    banks[0][0] = torch.tensor([2., -1.])
    packet = FormulaDeviceDispatchLayout(formula_device_dispatch_groups(query)[0])(torch.tensor([[0]]))
    result = dispatch(state, packet, data, banks)
    assert result.numeric_valid.all() and not result.overflow
    expected = _reference(value, 1, 3, 1, 1, (1, 0)) * banks[0][0]
    bucket = layout.index(expected)
    assert result.output_present[bucket][0, 0]
    torch.testing.assert_close(result.output_values[bucket][0, 0], expected)


def test_local_bank_parameters_safetensors_reload(tmp_path):
    from safetensors.torch import load_file, save_file
    from examples.formula_local_computation import build_program

    program, bindings = build_program(channels=2, padding=(1, 1))
    weights = {"kernel": torch.randn(3, 2), "slope": torch.tensor([0.2, 0.4]), "negative_unit": torch.tensor(-1.)}
    value = torch.randn(1, 7, 2)
    fabric = m.FormulaFabricV2(program)
    expected = fabric(inputs={"x": value}, banks={k: bindings[k].bind(v) for k, v in weights.items()}).values[0]
    save_file(weights, tmp_path / "local.safetensors")
    restored = load_file(tmp_path / "local.safetensors")
    assert sum(t.numel() for t in restored.values()) == 3 * 2 + 2 + 1
    fresh = m.FormulaFabricV2(m.FormulaProgram.from_dict(json.loads(json.dumps(program.to_dict()))))
    actual = fresh(inputs={"x": value}, banks={k: bindings[k].bind(v) for k, v in restored.items()}).values[0]
    assert torch.equal(actual, expected)
