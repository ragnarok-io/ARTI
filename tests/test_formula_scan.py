from __future__ import annotations

import json
from dataclasses import replace

import pytest
import torch

import arti
from arti import mechanisms as m
from arti._formula_candidate_batch import _checked_plan
from arti._formula_grouped_training import execute_grouped_training, grouped_formula_training
from examples.formula_recurrence import gated_scan


LIMITS = m.FormulaLimits(max_instructions=2048, max_slots=4096, max_steps=1024)


def _values(length=4, dtype=torch.float64):
    inputs = {"x": torch.randn(2, length, 2, dtype=dtype, requires_grad=True),
              "h": torch.randn(2, 2, dtype=dtype, requires_grad=True),
              "total": torch.randn(2, 2, dtype=dtype, requires_grad=True)}
    weights = {"gain": torch.randn(2, dtype=dtype, requires_grad=True),
               "retention": torch.randn(2, dtype=dtype, requires_grad=True),
               "unit": torch.ones(2, dtype=dtype), "negative_unit": -torch.ones(2, dtype=dtype),
               "zero": torch.zeros(2, dtype=dtype, requires_grad=True)}
    return inputs, weights


def _reference(inputs, weights):
    h, total = inputs["h"], inputs["total"]
    states = []
    for x in inputs["x"].unbind(1):
        g = (x + weights["retention"] * h).sigmoid()
        u = (x * weights["gain"] + h).tanh()
        h = g * h + (1 - g) * u
        total = total + h
        states.append(h)
    stacked = torch.stack(states)
    return h, total, stacked, stacked > 0


def _bound(bindings, values):
    return {n: bindings[n].bind(v) for n, v in values.items()}


@pytest.mark.parametrize("length", (1, 4))
@pytest.mark.parametrize("backend", ("reference", "prepared", "eager", "aot_eager"))
def test_shared_scan_state_emissions_and_bank_gradients(length, backend):
    scan, bindings = gated_scan(2, dtype="float64")
    restored = m.FormulaScan.from_dict(json.loads(json.dumps(scan.to_dict())))
    assert restored.to_dict() == scan.to_dict()
    assert restored.lower(length, limits=LIMITS).fingerprint == scan.lower(length, limits=LIMITS).fingerprint
    assert arti.validate_component_provenance(arti.component_provenance(restored))
    inputs, weights = _values(length)
    before = {n: v.detach().clone() for n, v in {**inputs, **weights}.items()}
    plan, prepared = restored.prepare(inputs=inputs, banks=_bound(bindings, weights), limits=LIMITS)
    leaves = tuple(v for v in (*inputs.values(), *weights.values()) if v.requires_grad)
    if backend == "reference":
        actual = m.FormulaFabricV2(plan.program)(inputs=inputs, banks=_bound(bindings, weights)).values
    elif backend == "prepared":
        actual = plan.forward_checked(prepared)[0]
    else:
        with grouped_formula_training(backend=backend):
            output, valid = execute_grouped_training(_checked_plan(plan.program), (prepared, prepared))
            assert valid.tolist() == [True, True]
            actual = output[0]
            gradients = torch.autograd.grad(sum(v.square().sum() for v in actual[:3]), leaves, retain_graph=True, allow_unused=True)
    if backend in {"reference", "prepared"}:
        gradients = torch.autograd.grad(sum(v.square().sum() for v in actual[:3]), leaves, retain_graph=True, allow_unused=True)
    expected = _reference(inputs, weights)
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e)
    targets = torch.autograd.grad(sum(v.square().sum() for v in expected[:3]), leaves, allow_unused=True)
    for a, e in zip(gradients, targets, strict=True):
        if e is None:
            assert a is None
        else:
            torch.testing.assert_close(a, e)
            assert a.isfinite().all()
    for n, v in {**inputs, **weights}.items():
        assert torch.equal(v.detach(), before[n])
    assert len(plan.program.bank_names) == len(bindings)


def test_second_derivative_and_upstream_bank_update_are_not_detached():
    scan, bindings = gated_scan(2, dtype="float64")
    inputs, weights = _values(3)
    base = weights["gain"]
    weights["gain"] = base + 0.1 * base.square()
    actual = scan(inputs=inputs, banks=_bound(bindings, weights), limits=LIMITS)["carry.h"]
    expected = _reference(inputs, weights)[0]
    ga, = torch.autograd.grad(actual.square().sum(), base, create_graph=True, retain_graph=True)
    ge, = torch.autograd.grad(expected.square().sum(), base, create_graph=True, retain_graph=True)
    torch.testing.assert_close(ga, ge)
    gga, = torch.autograd.grad(ga.sum(), base, retain_graph=True)
    gge, = torch.autograd.grad(ge.sum(), base)
    torch.testing.assert_close(gga, gge)


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
def test_scan_storage_dtype_and_finite_gradient(dtype):
    scan, bindings = gated_scan(2, dtype=str(dtype)[6:])
    inputs, weights = _values(3, dtype)
    result = scan(inputs=inputs, banks=_bound(bindings, weights), limits=LIMITS)
    assert result["carry.h"].dtype == dtype and result["emit.positive"].dtype == torch.bool
    grad, = torch.autograd.grad(result["emit.hidden"].float().square().sum(), weights["gain"])
    assert grad.isfinite().all()


def test_dynamic_length_reprepare_and_exact_sequence_shape_admission():
    scan, bindings = gated_scan(2, dtype="float64")
    inputs, weights = _values(1)
    plan1, _ = scan.prepare(inputs=inputs, banks=_bound(bindings, weights), limits=LIMITS)
    inputs, weights = _values(5)
    plan5, prepared = scan.prepare(inputs=inputs, banks=_bound(bindings, weights), limits=LIMITS)
    assert plan1.program_fingerprint != plan5.program_fingerprint
    assert scan.lower(5, limits=LIMITS) is plan5.program
    with pytest.raises(m.FormulaBindingError, match="different program"):
        plan1(prepared)
    with pytest.raises(m.FormulaBindingError):
        m.FormulaFabricV2(plan1.program).bind_tensors(inputs=inputs, banks=_bound(bindings, weights))
    with pytest.raises(m.FormulaTypeError, match="horizon"):
        scan.lower(257, limits=LIMITS)
    with pytest.raises(m.FormulaProgramError, match="instruction/slot"):
        scan.lower(64)
    with pytest.raises(m.FormulaProgramError, match="dependency depth"):
        scan.lower(3, limits=m.FormulaLimits(max_instructions=2048, max_slots=4096, max_steps=2))
    with pytest.raises(m.FormulaBindingError, match="working"):
        scan.prepare(inputs=inputs, banks=_bound(bindings, weights),
                     limits=replace(LIMITS, max_working_bytes=1))


def test_multiple_sequences_preserve_shared_symbol_equations():
    t = m.TensorType(("D",), ("T",), dtype="float64")
    x, y, carry = (m.InputBinding(n, t) for n in ("x", "y", "h"))
    result = m.add(carry, m.add(x, y))
    body = m.FormulaProgram.build(outputs=(result,))
    seq = m.TensorType(("TIME", "D"), ("T", "T"), dtype="float64")
    scan = m.FormulaScan(body, axis="TIME", sequence_types={"x": seq, "y": seq},
                         carry_outputs={"h": body.outputs[0]}, emissions={"y": body.outputs[0]}, max_length=5)
    good = {"x": torch.ones(3, 3, dtype=torch.float64), "y": torch.ones(3, 3, dtype=torch.float64), "h": torch.zeros(3, dtype=torch.float64)}
    assert torch.equal(scan(inputs=good, banks={})["carry.h"], torch.full((3,), 6., dtype=torch.float64))
    with pytest.raises(m.FormulaBindingError, match="lengths must match"):
        scan.prepare(inputs={**good, "y": torch.ones(2, 3, dtype=torch.float64)}, banks={})
    with pytest.raises(m.FormulaBindingError):
        scan.prepare(inputs={**good, "h": torch.zeros(2, dtype=torch.float64)}, banks={})


def test_temporal_order_and_simultaneous_carry_advance():
    t = m.TensorType((), (), dtype="float64")
    a, b, x = (m.InputBinding(n, t) for n in ("a", "b", "x"))
    first, second = m.add(b, x), m.scale(a, x)
    body = m.FormulaProgram.build(outputs=(first, second))
    scan = m.FormulaScan(body, axis="T", sequence_types={"x": m.TensorType(("T",), ("T",), dtype="float64")},
                         carry_outputs={"a": body.outputs[0], "b": body.outputs[1]}, emissions={"y": body.outputs[0]}, max_length=8)
    inputs = {"x": torch.tensor([2., 3., 4.], dtype=torch.float64), "a": torch.tensor(1., dtype=torch.float64), "b": torch.tensor(2., dtype=torch.float64)}
    result = scan(inputs=inputs, banks={})
    assert result["carry.a"] == 16 and result["carry.b"] == 20
    assert torch.equal(result["emit.y"], torch.tensor([4., 5., 16.], dtype=torch.float64))
    reverse = scan(inputs={**inputs, "x": inputs["x"].flip(0)}, banks={})
    assert reverse["carry.a"] != result["carry.a"]


def test_invalid_scan_ports_carry_type_and_effects():
    scan, _ = gated_scan(2)
    config = scan.to_dict()
    config["carry_outputs"] = {"h": scan.body.outputs[-1]}
    with pytest.raises(m.FormulaTypeError, match="carry output"):
        m.FormulaScan.from_dict(config)
    t = m.TensorType(("D",), (2,))
    x, h = m.InputBinding("x", t), m.InputBinding("h", t)
    effect = m.neural_plasticity(x, h, h)
    body = m.FormulaProgram.build(outputs=(effect,))
    with pytest.raises(m.FormulaTypeError, match="pure"):
        m.FormulaScan(body, axis="T", sequence_types={"x": m.TensorType(("T", "D"), (3, 2))},
                      carry_outputs={"h": body.outputs[0]}, emissions={"y": body.outputs[0]}, max_length=3)


def test_scan_bank_safetensors_and_config_roundtrip(tmp_path):
    from safetensors.torch import load_file, save_file

    scan, bindings = gated_scan(2, dtype="float64")
    inputs, weights = _values(4)
    expected = scan(inputs=inputs, banks=_bound(bindings, weights), limits=LIMITS)
    save_file({n: w.detach() for n, w in weights.items()}, tmp_path / "scan.safetensors")
    restored = m.FormulaScan.from_dict(json.loads(json.dumps(scan.to_dict())))
    actual = restored(inputs=inputs, banks=_bound(bindings, load_file(tmp_path / "scan.safetensors")), limits=LIMITS)
    assert actual.keys() == expected.keys()
    for n in actual:
        assert torch.equal(actual[n], expected[n])


def test_lowered_scan_reaches_existing_candidate_and_real_typed_pools():
    from arti._formula_device_dispatch import FormulaDeviceDispatchLayout, FormulaDeviceNumericalDispatch, formula_device_dispatch_groups
    from arti._formula_device_frames import FormulaDeviceFrameKernel
    from arti._formula_device_pools import FormulaDevicePoolLayout

    t = m.TensorType(("B", "D"), ("B", 2), dtype="float32")
    x, h = m.InputBinding("x", t), m.InputBinding("h", t)
    gain = m.BankBinding("gain", "arti/scan-test@1", "gain", m.TensorType(("D",), (2,), dtype="float32"))
    y = m.add(h, m.scale(x, gain))
    body = m.FormulaProgram.build(outputs=(y,))
    scan = m.FormulaScan(body, axis="T", sequence_types={"x": m.TensorType(("B", "T", "D"), ("B", "T", 2), dtype="float32")},
                         carry_outputs={"h": body.outputs[0]}, emissions={"y": body.outputs[0]}, max_length=8)
    program = scan.lower(3)
    candidate = m.FormulaProgramTensorCandidateV4(m.FormulaProgramCandidateV3(
        "scan", program, input_slots={"x": "x", "h": "h"},
        output_slots=dict(zip(program.outputs, ("last", "sequence"), strict=True)), operands={"gain": torch.tensor([2., -1.])},
    ), plastic_bank_slot="gain", bank_owner_id="scan")
    query = m.FormulaProgramQueryV5(slot_ids=("x", "h", "last", "sequence"), candidates=(candidate,),
                                   terminal_slots={"last": "last", "sequence": "sequence"}, max_steps=1)
    kernel = FormulaDeviceFrameKernel.from_query(query)
    layout = FormulaDevicePoolLayout.from_samples((torch.zeros(1, 3, 2), torch.zeros(1, 2), torch.zeros(3, 1, 2)), 4)
    bank_layout = FormulaDevicePoolLayout.from_samples((torch.zeros(2),), 4)
    dispatch = FormulaDeviceNumericalDispatch(query, kernel)
    dispatch.prepare_typed_pools_(layout, bank_layout)
    state = kernel.initial_state(1, torch.tensor([[layout.offsets[0], layout.offsets[1], -1, -1]]),
                                 bank_value_handles=torch.tensor([0]))
    data, banks = layout.allocate("cpu"), bank_layout.allocate("cpu")
    value = torch.arange(6.).reshape(1, 3, 2)
    initial = torch.tensor([[0.1, 0.3]])
    data[layout.index(value)][0], data[layout.index(initial)][0] = value, initial
    banks[0][0] = torch.tensor([2., -1.])
    packet = FormulaDeviceDispatchLayout(formula_device_dispatch_groups(query)[0])(torch.tensor([[0]]))
    result = dispatch(state, packet, data, banks)
    assert result.numeric_valid.all() and not result.overflow
    expected = initial + (value * banks[0][0]).cumsum(1)
    for head, tensor in enumerate((expected[:, -1], expected.transpose(0, 1))):
        bucket = layout.index(tensor)
        assert result.output_present[bucket][0, head]
        torch.testing.assert_close(result.output_values[bucket][0, head], tensor)
