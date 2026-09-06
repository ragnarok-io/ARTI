from __future__ import annotations

import json

import pytest
import torch

import arti
from arti import mechanisms as m
from arti._formula_candidate_batch import _checked_plan
from arti._formula_grouped_training import execute_grouped_training, grouped_formula_training
from examples.formula_bank_nonlinearity import build_program, build_temperature_program


SCALAR_FUNCTIONS = {
    "gelu": torch.nn.functional.gelu, "relu": torch.relu, "rsqrt": torch.rsqrt,
    "sigmoid": torch.sigmoid, "silu": torch.nn.functional.silu, "tanh": torch.tanh,
    "abs": torch.abs, "exp": torch.exp, "expm1": torch.expm1, "log": torch.log,
    "log1p": torch.log1p, "softplus": torch.nn.functional.softplus,
    "reciprocal": torch.reciprocal, "sin": torch.sin, "cos": torch.cos,
}


@pytest.mark.parametrize("mode", SCALAR_FUNCTIONS)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_scalar_v2_values_gradients_and_identity(mode, dtype):
    value_type = m.TensorType(("B", "D"), ("B", 4), dtype=str(dtype).removeprefix("torch."))
    atom = m.ScalarMapAtomV2(value_type, mode=mode)
    x = torch.tensor([[0.01, 0.3, 0.8, 1.2]], dtype=dtype, requires_grad=True)
    actual = atom(x)
    expected = SCALAR_FUNCTIONS[mode](x)
    torch.testing.assert_close(actual, expected)
    left, = torch.autograd.grad(actual.sum(), x, retain_graph=True)
    right, = torch.autograd.grad(expected.sum(), x)
    torch.testing.assert_close(left, right)
    assert arti.component_ref(atom) == "arti/formula-atom-scalar-map@2"
    assert arti.validate_component_provenance(arti.component_provenance(atom))
    assert not tuple(atom.parameters())


@pytest.mark.parametrize("mode", ("sum", "mean", "amax", "logsumexp"))
@pytest.mark.parametrize("count", (1, 7))
def test_native_reduce_dynamic_shapes_gradients_and_round_trip(mode, count):
    value_type = m.TensorType(("B", "N", "D"), ("B", "N", 3), dtype="float64")
    x_binding = m.InputBinding("x", value_type)
    program = m.FormulaProgram.build(outputs=(m.reduce_tensor(x_binding, axis="N", mode=mode),))
    restored = m.FormulaProgram.from_dict(json.loads(json.dumps(program.to_dict())))
    assert restored.fingerprint == program.fingerprint
    x = torch.arange(2 * count * 3, dtype=torch.float64).reshape(2, count, 3).requires_grad_()
    actual = m.FormulaFabricV2(restored)(inputs={"x": x}, banks={}).values[0]
    expected = getattr(torch, mode)(x, dim=1)
    assert actual.shape == (2, 3)
    torch.testing.assert_close(actual, expected)
    left, = torch.autograd.grad(actual.sum(), x, retain_graph=True)
    right, = torch.autograd.grad(expected.sum(), x)
    torch.testing.assert_close(left, right)
    atom = m.ReduceAtomV2(value_type, axis="N", mode=mode)
    torch.testing.assert_close(atom(x), actual)
    assert arti.component_ref(atom) == "arti/formula-atom-reduce@2"
    assert arti.validate_component_provenance(arti.component_provenance(atom))


def test_reduction_and_scalar_v1_contracts_are_unchanged():
    t = m.TensorType(("D",), (3,), dtype="float32")
    x = m.InputBinding("x", t)
    old = m.FormulaProgram.build(outputs=(m.reduce_sum(x, axis="D"), m.scalar_map(x, mode="relu")))
    assert {instruction.atom_ref for instruction in old.instructions} == {
        "arti/formula-atom-reduce@1", "arti/formula-atom-scalar-map@1",
    }
    value = torch.tensor([1e20, -1e20, 3.0])
    assert m.ReduceAtom(t, axis="D")(value).item() == 3.0
    with pytest.raises(m.FormulaTypeError, match="unsupported ScalarMap"):
        m.scalar_map(x, mode="expm1")
    payload = old.to_dict()
    payload["instructions"][0]["attributes"]["mode"] = "mean"
    with pytest.raises(m.FormulaProgramError, match="only supports"):
        m.FormulaProgram.from_dict(payload)


def test_native_functions_are_stable_and_ties_have_torch_gradients():
    t = m.TensorType(("D",), (3,), dtype="float64")
    tiny = torch.tensor([-1e-14, 0.0, 1e-14], dtype=torch.float64)
    for mode in ("expm1", "log1p"):
        actual = m.ScalarMapAtomV2(t, mode=mode)(tiny)
        torch.testing.assert_close(actual, SCALAR_FUNCTIONS[mode](tiny), atol=0, rtol=0)
        assert actual[0] != 0 and actual[-1] != 0
    large = torch.tensor([1000.0, 1001.0, 999.0], dtype=torch.float64)
    assert torch.isfinite(m.ScalarMapAtomV2(t, mode="softplus")(large)).all()
    torch.testing.assert_close(m.ReduceAtomV2(t, axis="D", mode="logsumexp")(large), torch.logsumexp(large, 0))
    tied = torch.tensor([2.0, 2.0, 0.0], dtype=torch.float64, requires_grad=True)
    m.ReduceAtomV2(t, axis="D", mode="amax")(tied).backward()
    torch.testing.assert_close(tied.grad, torch.tensor([0.5, 0.5, 0.0], dtype=torch.float64))


@pytest.mark.parametrize("mode,value", (("log", -1.0), ("log1p", -1.0), ("reciprocal", 0.0)))
def test_invalid_domains_are_not_silently_repaired(mode, value):
    t = m.TensorType((), (), dtype="float32")
    x = m.InputBinding("x", t)
    fabric = m.FormulaFabricV2(m.FormulaProgram.build(outputs=(m.scalar_map_v2(x, mode=mode),)))
    prepared = fabric.bind_tensors(inputs={"x": torch.tensor(value)}, banks={})
    _outputs, finite = fabric.execution_plan().forward_checked(prepared)
    assert not finite
    with pytest.raises(m.FormulaBindingError):
        fabric(inputs={"x": torch.tensor(value)}, banks={})


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64))
def test_masked_softmax_excludes_hidden_overflow_before_exp(dtype):
    t = m.TensorType(("B", "D"), (2, 2), dtype=str(dtype).removeprefix("torch."))
    mask_type = m.TensorType(("B", "D"), (2, 2), dtype="boolean")
    x = torch.tensor([[0.0, 1000.0], [1000.0, -1000.0]], dtype=dtype, requires_grad=True)
    mask = torch.tensor([[True, False], [False, False]])
    for policy in ("float32", "activation"):
        actual = m.MaskedSoftmaxAtom(t, mask_type, axis="D", accumulation_dtype=policy)(x, mask)
        torch.testing.assert_close(actual, torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=dtype))
        grad, = torch.autograd.grad(actual.sum(), x)
        assert torch.equal(grad, torch.zeros_like(x))


def _composition_data(dtype, *, batch=2, count=3):
    generator = torch.Generator().manual_seed(415)
    inputs = {
        "x": torch.randn(batch, count, 4, generator=generator, dtype=dtype).requires_grad_(),
        "negative_one": torch.tensor(-1.0, dtype=dtype), "one": torch.tensor(1.0, dtype=dtype),
        "epsilon": torch.tensor(1e-5, dtype=dtype),
    }
    parameters = {
        name: torch.randn(4, generator=generator, dtype=dtype).requires_grad_()
        for name in ("slope", "beta", "offset", "radius", "gamma", "bias")
    }
    return inputs, parameters


def _composition_reference(inputs, p):
    dtype = inputs["x"].dtype
    compute = torch.float64 if dtype == torch.float64 else torch.float32
    x = inputs["x"].to(compute)
    p = {name: value.to(compute) for name, value in p.items()}
    eps = inputs["epsilon"].to(compute)
    prelu = torch.where(x > 0, x, x * p["slope"])
    swish = x * torch.sigmoid(p["beta"] * x)
    radius = torch.nn.functional.softplus(p["radius"]) + eps
    saturated = radius * torch.tanh((x + p["offset"]) / radius)
    centered = x - x.mean(dim=-1, keepdim=True)
    layer = centered * torch.rsqrt(centered.square().mean(-1, keepdim=True) + eps) * p["gamma"] + p["bias"]
    rms = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * p["gamma"]
    return tuple(value.to(dtype) for value in (prelu, swish, saturated, layer, rms))


@pytest.mark.parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32, torch.float64))
def test_bank_composition_has_explicit_precision_and_real_parameter_gradients(dtype):
    program, bindings = build_program(dtype=str(dtype).removeprefix("torch."))
    restored = m.FormulaProgram.from_dict(json.loads(json.dumps(program.to_dict())))
    assert program.fingerprint == restored.fingerprint
    fabric = m.FormulaFabricV2(restored)
    inputs, parameters = _composition_data(dtype)
    actual = fabric(inputs=inputs, banks={name: binding.bind(parameters[name]) for name, binding in bindings.items()}).values
    expected = _composition_reference(inputs, parameters)
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left, right)
        assert left.dtype == dtype
    leaves = (inputs["x"], *parameters.values())
    left = torch.autograd.grad(sum(v.float().square().mean() for v in actual), leaves, retain_graph=True)
    right = torch.autograd.grad(sum(v.float().square().mean() for v in expected), leaves)
    for a, b in zip(left, right, strict=True):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b, atol=0.01 if dtype == torch.bfloat16 else 0.002, rtol=0.02)
    assert arti.validate_component_provenance(arti.component_provenance(fabric))


def test_half_normalization_keeps_square_in_fp32_and_cast_preflight_tracks_output():
    program, bindings = build_program(dtype="float16")
    fabric = m.FormulaFabricV2(program)
    inputs, p = _composition_data(torch.float16)
    inputs["x"] = torch.full((2, 3, 4), 300.0, dtype=torch.float16, requires_grad=True)
    outputs = fabric(inputs=inputs, banks={name: binding.bind(p[name]) for name, binding in bindings.items()}).values
    assert all(torch.isfinite(v).all() for v in outputs)
    torch.testing.assert_close(outputs[-1], p["gamma"].expand_as(outputs[-1]))
    atom = m.CastAtom(m.TensorType(("D",), (4,), dtype="float16"), dtype="float64")
    assert arti.component_ref(atom) == "arti/formula-atom-cast@1"
    assert arti.validate_component_provenance(arti.component_provenance(atom))
    assert atom(torch.ones(4, dtype=torch.float16)).dtype == torch.float64
    x = m.InputBinding("x", m.TensorType(("D",), ("D",), dtype="float16"))
    limited = m.FormulaFabricV2(m.FormulaProgram.build(
        outputs=(m.cast(x, dtype="float64"),), limits=m.FormulaLimits(max_tensor_bytes=16)
    ))
    with pytest.raises(m.FormulaBindingError):
        limited.bind_tensors(inputs={"x": torch.ones(4, dtype=torch.float16)}, banks={})


@pytest.mark.parametrize("backend", ("eager", "aot_eager"))
@pytest.mark.parametrize("dtype", (torch.float16, torch.float64))
def test_grouped_composition_and_checked_plan_use_same_values_and_vjp(backend, dtype):
    program, bindings = build_program(dtype=str(dtype).removeprefix("torch."))
    fabric = m.FormulaFabricV2(program)
    plan = _checked_plan(program)
    assert plan is not None
    rows = []
    all_inputs = []
    references = []
    for count in (2, 2):
        inputs, p = _composition_data(dtype, batch=1, count=count)
        all_inputs.extend((inputs["x"], *p.values()))
        rows.append(fabric.bind_tensors(inputs=inputs, banks={name: binding.bind(p[name]) for name, binding in bindings.items()}))
        references.append(_composition_reference(inputs, p))
    with grouped_formula_training(backend=backend):
        outputs, finite = execute_grouped_training(plan, tuple(rows))
        assert finite.all()
        for row, reference in zip(outputs, references, strict=True):
            for a, b in zip(row, reference, strict=True):
                torch.testing.assert_close(a, b)
        # Only one row contributes; unused row Bank operands must stay unused.
        actual = torch.autograd.grad(sum(v.square().mean() for v in outputs[0]), all_inputs, allow_unused=True)
    expected = torch.autograd.grad(sum(v.square().mean() for v in references[0]), all_inputs, allow_unused=True)
    for a, b in zip(actual, expected, strict=True):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b)


@pytest.mark.parametrize("backend", ("eager", "aot_eager"))
@pytest.mark.parametrize("sink_enabled", (True, False))
def test_temperature_bias_and_sink_bank_parameters_keep_unnormalized_remaining_mass(backend, sink_enabled):
    program, bindings = build_temperature_program(dtype="float64")
    assert _checked_plan(program) is not None
    values = {
        "temperature": torch.tensor(-0.5, dtype=torch.float64, requires_grad=True),
        "bias": torch.tensor([0.3, -0.4, 0.2], dtype=torch.float64, requires_grad=True),
        "sink": torch.tensor([2.0], dtype=torch.float64, requires_grad=True),
    }
    logits = torch.tensor([[0.1, 0.5, -0.2], [0.2, 0.3, 0.4]], dtype=torch.float64, requires_grad=True)
    mask = torch.tensor([[True, False, True], [False, False, False]])
    inputs = {"logits": logits, "mask": mask, "sink_mask": torch.full((2, 1), sink_enabled, dtype=torch.bool),
              "temperature_floor": torch.tensor(0.01, dtype=torch.float64)}
    fabric = m.FormulaFabricV2(program)
    bound = {name: binding.bind(values[name]) for name, binding in bindings.items()}
    actual = fabric(inputs=inputs, banks=bound).values[0]
    tau = torch.nn.functional.softplus(values["temperature"]) + inputs["temperature_floor"]
    scaled = torch.cat((logits + values["bias"], values["sink"].expand(2, 1)), dim=-1) / tau
    visible = torch.cat((mask, inputs["sink_mask"]), dim=-1)
    masked = scaled.masked_fill(~visible, -torch.inf)
    safe = torch.where(visible.any(-1, keepdim=True), masked, torch.zeros_like(masked))
    expected = torch.softmax(safe, dim=-1).masked_fill(~visible, 0)[:, :3]
    torch.testing.assert_close(actual, expected)
    if sink_enabled:
        assert 0 < actual[0].sum() < 1
    else:
        torch.testing.assert_close(actual[0].sum(), torch.tensor(1., dtype=torch.float64))
    assert torch.equal(actual[1], torch.zeros_like(actual[1]))
    leaves = (logits, *values.values())
    left = torch.autograd.grad(actual.square().sum(), leaves, retain_graph=True)
    right = torch.autograd.grad(expected.square().sum(), leaves, retain_graph=True)
    for a, b in zip(left, right, strict=True):
        assert torch.isfinite(a).all()
        torch.testing.assert_close(a, b)
    assert left[2][1] == 0  # A masked position cannot train its visible-logit bias.
    assert left[2][0] != 0
    with grouped_formula_training(backend=backend):
        rows, finite = execute_grouped_training(_checked_plan(program), (fabric.bind_tensors(inputs=inputs, banks=bound),))
        assert finite.all()
        torch.testing.assert_close(rows[0][0], expected)
        grouped_grad = torch.autograd.grad(rows[0][0].square().sum(), leaves)
    for a, b in zip(grouped_grad, right, strict=True):
        torch.testing.assert_close(a, b)


def test_temperature_bias_sink_program_and_bank_roundtrip(tmp_path):
    from safetensors.torch import load_file, save_file

    program, _ = build_temperature_program(dtype="float64")
    values = {"temperature": torch.tensor(0.4, dtype=torch.float64),
              "bias": torch.tensor([0.2, -0.1, 0.5], dtype=torch.float64),
              "sink": torch.tensor([0.3], dtype=torch.float64)}
    inputs = {"logits": torch.tensor([[0.7, 0.1, -0.2]], dtype=torch.float64),
              "mask": torch.ones(1, 3, dtype=torch.bool), "sink_mask": torch.ones(1, 1, dtype=torch.bool),
              "temperature_floor": torch.tensor(0.01, dtype=torch.float64)}

    def candidate(body):
        return m.FormulaProgramCandidateV3(
            "temperature-bias-sink", body, input_slots={name: name for name in inputs},
            output_slots={body.outputs[0]: "out"}, operands=values,
            trainable_operands=tuple(values),
        )

    original = candidate(program)
    restored = candidate(m.FormulaProgram.from_dict(program.to_dict()))
    with torch.no_grad():
        original.operand_store.tensor("bias")[0].add_(0.25)
    path = tmp_path / "normalization.safetensors"
    save_file(original.state_dict(), str(path))
    restored.load_state_dict(load_file(str(path)))

    def run(item):
        return item.fabric(inputs=inputs, banks={binding.name: binding.bind(item.operand_store.tensor(binding.name))
            for binding in item.fabric.program.bindings if isinstance(binding, m.BankBinding)}).values[0]

    torch.testing.assert_close(run(original), run(restored), atol=0, rtol=0)
    assert sum(value.numel() for value in original.parameters()) == 5


def test_bank_owned_nonlinear_operands_reload_from_safetensors(tmp_path):
    from safetensors.torch import load_file, save_file

    program, bindings = build_program(dtype="float64")
    inputs, values = _composition_data(torch.float64)

    def candidate():
        return m.FormulaProgramCandidateV3(
            "nonlinear", program, input_slots={name: name for name in inputs},
            output_slots={name: f"head-{index}" for index, name in enumerate(program.outputs)},
            operands=values, trainable_operands=tuple(values),
        )

    original, restored = candidate(), candidate()
    with torch.no_grad():
        original.operand_store.tensor("beta").add_(0.25)
    path = tmp_path / "nonlinear.safetensors"
    save_file(original.state_dict(), str(path))
    restored.load_state_dict(load_file(str(path)))

    def run(item):
        return item.fabric(inputs=inputs, banks={name: binding.bind(item.operand_store.tensor(name)) for name, binding in bindings.items()}).values

    for a, b in zip(run(original), run(restored), strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert sum(value.numel() for value in original.parameters()) == 6 * 4


def test_logsumexp_preflight_counts_full_size_native_scratch():
    x = m.InputBinding("x", m.TensorType(("D",), (64,), dtype="float16"))
    program = m.FormulaProgram.build(
        outputs=(m.reduce_tensor(x, axis="D", mode="logsumexp"),),
        limits=m.FormulaLimits(max_working_bytes=512),
    )
    with pytest.raises(m.FormulaBindingError, match="working"):
        m.FormulaFabricV2(program).bind_tensors(inputs={"x": torch.zeros(64, dtype=torch.float16)}, banks={})


def test_cast_and_reduce_reach_typed_device_output_bucket_on_cpu():
    from arti._formula_device_dispatch import (
        FormulaDeviceDispatchLayout, FormulaDeviceNumericalDispatch, formula_device_dispatch_groups,
    )
    from arti._formula_device_frames import FormulaDeviceFrameKernel
    from arti._formula_device_pools import FormulaDevicePoolLayout

    t = m.TensorType(("B", "D"), ("B", 3), dtype="float16")
    x = m.InputBinding("x", t)
    gain = m.BankBinding("gain", "arti/numeric-dispatch-test@1", "gain", t)
    value = m.scalar_map_v2(m.scale(m.cast(x, dtype="float64"), m.cast(gain, dtype="float64")), mode="softplus")
    program = m.FormulaProgram.build(outputs=(m.reduce_tensor(value, axis="D", mode="mean"),))
    candidate = m.FormulaProgramTensorCandidateV3(m.FormulaProgramCandidateV2(
        "numeric", program, input_slots={"x": "x"}, output_slot="out",
        operands={"gain": torch.full((1, 3), 0.5, dtype=torch.float16)},
    ), plastic_bank_slot="gain", bank_owner_id="numeric")
    # This test exercises numeric dispatch, not a new selection network.
    query = m.FormulaProgramQueryV5(slot_ids=("x", "out"), candidates=(candidate,),
                                  terminal_slots={"answer": "out"}, max_steps=1)
    kernel = FormulaDeviceFrameKernel.from_query(query)
    dl = FormulaDevicePoolLayout.from_samples((torch.zeros(1, 3, dtype=torch.float16), torch.zeros(1, dtype=torch.float64)), 4)
    bl = FormulaDevicePoolLayout.from_samples((torch.zeros(1, 3, dtype=torch.float16),), 4)
    dispatch = FormulaDeviceNumericalDispatch(query, kernel)
    dispatch.prepare_typed_pools_(dl, bl)
    sample = torch.tensor([[1.0, 2.0, 3.0]], dtype=torch.float16)
    input_bucket = dl.index(sample)
    output_bucket = dl.index(torch.zeros(1, dtype=torch.float64))
    state = kernel.initial_state(1, torch.tensor([[dl.offsets[input_bucket], -1]]),
                                 bank_value_handles=torch.tensor([bl.offsets[0]]))
    data, banks = dl.allocate("cpu"), bl.allocate("cpu")
    data[input_bucket][0] = sample
    banks[0][0] = 0.5
    packet = FormulaDeviceDispatchLayout(formula_device_dispatch_groups(query)[0])(torch.tensor([[0]]))
    result = dispatch(state, packet, data, banks)
    assert result.numeric_valid.all() and not result.overflow
    assert result.output_present[output_bucket][0, 0]
    actual = result.output_values[output_bucket][0, 0]
    expected = torch.nn.functional.softplus(sample.double() * 0.5).mean(dim=-1)
    assert actual.dtype == torch.float64
    torch.testing.assert_close(actual, expected)
