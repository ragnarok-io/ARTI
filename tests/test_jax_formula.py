from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

from arti import mechanisms as m
import arti.jax as aj


@pytest.fixture(autouse=True)
def exact_types():
    previous = jax.config.x64_enabled
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", previous)


def parity(expression, inputs, banks=None, *, gradients=True, tolerance=2e-5):
    program = m.FormulaProgram.build(outputs=(expression,))
    torch_inputs = {
        name: torch.tensor(
            value, requires_grad=gradients and np.issubdtype(np.asarray(value).dtype, np.floating)
        )
        for name, value in inputs.items()
    }
    torch_banks = {
        binding.name: binding.bind(torch.tensor(value, requires_grad=gradients))
        for binding, value in (banks or {}).items()
    }
    expected = m.FormulaFabricV2(program)(inputs=torch_inputs, banks=torch_banks).values[0]
    native = aj.bind_formula(program)
    params = native.init(
        jax.random.key(0),
        bank_initializers={binding.name: value for binding, value in (banks or {}).items()},
    )
    values = {name: jnp.asarray(value) for name, value in inputs.items()}
    actual = jax.jit(native.apply_checked)(params, values)
    assert bool(actual.valid)
    np.testing.assert_allclose(
        actual.outputs[program.outputs[0]],
        expected.detach().numpy(),
        rtol=tolerance,
        atol=tolerance,
    )
    if gradients:
        expected.square().sum().backward()
        float_values = {
            name: value
            for name, value in values.items()
            if jnp.issubdtype(value.dtype, jnp.floating)
        }
        constants = {name: value for name, value in values.items() if name not in float_values}

        def loss(p, x):
            result = native.apply(p, {**constants, **x})[program.outputs[0]]
            return jnp.square(result).sum()

        dp, dx = jax.jit(jax.grad(loss, argnums=(0, 1)))(params, float_values)
        for name, value in dx.items():
            reference = torch_inputs[name].grad
            reference = np.zeros_like(inputs[name]) if reference is None else reference.numpy()
            np.testing.assert_allclose(value, reference, rtol=tolerance, atol=tolerance)
        for name, value in dp.items():
            np.testing.assert_allclose(
                value, torch_banks[name].value.grad.numpy(), rtol=tolerance, atol=tolerance
            )
    return native, params, values


@pytest.mark.parametrize(
    "mode",
    (
        "gelu",
        "relu",
        "sigmoid",
        "silu",
        "tanh",
        "abs",
        "exp",
        "expm1",
        "log",
        "log1p",
        "softplus",
        "reciprocal",
        "sin",
        "cos",
        "rsqrt",
    ),
)
def test_scalar_map_exact_forward_and_derivative(mode):
    value = m.InputBinding("x", m.TensorType(("B", "D"), ("B", 4), dtype="float32"))
    inputs = np.array([[0.2, 0.4, 0.8, 1.2], [0.3, 0.5, 0.9, 1.3]], dtype=np.float32)
    parity(m.scalar_map_v2(value, mode=mode), {"x": inputs})


def test_abs_zero_derivative_matches_torch():
    value = m.InputBinding("x", m.TensorType(("N",), (3,), dtype="float32"))
    native = aj.bind_formula(m.FormulaProgram.build(outputs=(m.scalar_map_v2(value, mode="abs"),)))
    gradient = jax.grad(lambda x: next(iter(native.apply({}, {"x": x}).values())).sum())(
        jnp.array([-1, 0, 1], dtype=jnp.float32)
    )
    np.testing.assert_array_equal(gradient, [-1, 0, 1])


def test_lora_ir_named_contraction_bank_gradients_and_jit():
    x = m.InputBinding("x", m.TensorType(("B", "N", "D"), ("B", 3, 4), dtype="float32"))
    a = m.BankBinding(
        "a", "arti/jax-test@1", "a", m.TensorType(("D", "R"), (4, 2), dtype="float32")
    )
    b = m.BankBinding(
        "b", "arti/jax-test@1", "b", m.TensorType(("R", "D"), (2, 4), dtype="float32")
    )
    hidden = m.contract(x, a, reduce_axes=(("D", "D"),), output_axes=("B", "N", "R"))
    result = m.add(x, m.contract(hidden, b, reduce_axes=(("R", "R"),), output_axes=("B", "N", "D")))
    rng = np.random.default_rng(11)
    native, params, inputs = parity(
        result,
        {"x": rng.normal(size=(2, 3, 4)).astype(np.float32)},
        {
            a: rng.normal(size=(4, 2)).astype(np.float32),
            b: rng.normal(size=(2, 4)).astype(np.float32),
        },
    )
    assert native.fingerprint == native.program.fingerprint
    lowered = jax.jit(native.apply).lower(params, inputs).as_text()
    assert "callback" not in lowered.lower()


def test_all_masked_softmax_output_and_gradient_are_zero():
    x = m.InputBinding("x", m.TensorType(("B", "N", "K"), ("B", 2, 3), dtype="float32"))
    mask = m.InputBinding("mask", m.TensorType(("B", "K"), ("B", 3), dtype="boolean"))
    native, _, inputs = parity(
        m.masked_softmax(x, mask, axis="K"),
        {
            "x": np.arange(12, dtype=np.float32).reshape(2, 2, 3),
            "mask": np.array([[True, False, True], [False, False, False]]),
        },
    )
    result = next(iter(native.apply({}, inputs).values()))
    np.testing.assert_array_equal(result[1], 0)


@pytest.mark.parametrize("additive", (False, True))
def test_index_fold_restore_and_duplicate_scatter_add(additive):
    x = m.InputBinding("x", m.TensorType(("B", "N", "D"), ("B", 4, 2), dtype="float32"))
    ids = m.InputBinding("ids", m.TensorType(("B", "K"), ("B", 2), dtype="int64"))
    folded = m.gather(x, ids, axis="N", index_axis="K")
    output = (
        m.scatter_add(x, ids, folded, axis="N", index_axis="K")
        if additive
        else m.scatter(x, ids, folded, axis="N", index_axis="K")
    )
    parity(
        output,
        {
            "x": np.arange(16, dtype=np.float32).reshape(2, 4, 2),
            "ids": np.array([[1, 1], [2, 2]] if additive else [[3, 1], [0, 2]], dtype=np.int64),
        },
    )


def test_index_failures_are_device_validity_receipts():
    x = m.InputBinding("x", m.TensorType(("B", "N", "D"), ("B", 4, 2), dtype="float32"))
    ids = m.InputBinding("ids", m.TensorType(("B", "K"), ("B", 2), dtype="int64"))
    active = m.gather(x, ids, axis="N", index_axis="K")
    native = aj.bind_formula(
        m.FormulaProgram.build(outputs=(m.scatter(x, ids, active, axis="N", index_axis="K"),))
    )
    for indices in ([[1, 1]], [[1, 5]]):
        result = jax.jit(native.apply_checked)(
            {},
            {
                "x": jnp.ones((1, 4, 2), dtype=jnp.float32),
                "ids": jnp.array(indices, dtype=jnp.int64),
            },
        )
        assert not bool(result.valid)


@pytest.mark.parametrize("mode", ("sum", "mean", "amax", "softmax"))
def test_segment_empty_groups_and_gradients(mode):
    x = m.InputBinding("x", m.TensorType(("B", "N", "D"), ("B", 4, 2), dtype="float32"))
    ids = m.InputBinding("ids", m.TensorType(("B", "N"), ("B", 4), dtype="int64"))
    mask = m.InputBinding("mask", m.TensorType(("B", "N"), ("B", 4), dtype="boolean"))
    parity(
        m.segment(x, ids, mask, axis="N", segment_axis="G", num_segments=3, mode=mode),
        {
            "x": np.array([[[1, 2], [3, 4], [5, 6], [2, 1]]], dtype=np.float32),
            "ids": np.array([[0, 0, 1, -1]], dtype=np.int64),
            "mask": np.array([[True, True, True, False]]),
        },
    )


def test_window_and_ordered_reduction_match_shared_contract():
    x = m.InputBinding("x", m.TensorType(("B", "N", "D"), ("B", 5, 2), dtype="float32"))
    windows = m.window(x, axis="N", output_axis="P", window_axis="W", kernel_size=3, padding=(1, 1))
    parity(m.reduce_sum(windows, axis="W"), {"x": np.arange(20, dtype=np.float32).reshape(2, 5, 2)})


def test_position_atoms_match_torch_and_differentiation():
    x = m.InputBinding("x", m.TensorType(("B", "N", "D"), ("B", 3, 4), dtype="float32"))
    position = m.InputBinding("positions", m.TensorType(("B", "N"), ("B", 3), dtype="int64"))
    parity(
        m.rotary_position(x, position, feature_axis="D"),
        {
            "x": np.arange(24, dtype=np.float32).reshape(2, 3, 4) / 10,
            "positions": np.array([[0, 1, 2], [2, 1, 0]], dtype=np.int64),
        },
    )
    field = m.sinusoidal_position(
        m.axis_index(x, axis="N"), feature_axis="D", feature_size=4, dtype="float32"
    )
    parity(
        m.add(x, m.broadcast(field, output_axes=("B", "N", "D"), output_sizes=("B", 3, 4))),
        {"x": np.zeros((2, 3, 4), dtype=np.float32)},
    )


@pytest.mark.parametrize("mode", ("identity", "affine", "cartesian", "polar"))
def test_observation_native_compiled_forward_and_backward(mode):
    state_dim = 3 if mode == "polar" else 2
    bindings = (
        m.InputBinding("x", m.TensorType(("B", "N", "D"), ("B", 6, 2), dtype="float32")),
        m.InputBinding(
            "states", m.TensorType(("B", "T", "S"), ("B", 2, state_dim), dtype="float32")
        ),
        m.InputBinding("mask", m.TensorType(("B", "N"), ("B", 6), dtype="boolean")),
        m.InputBinding("activity", m.TensorType(("B", "T"), ("B", 2), dtype="float32")),
    )
    rng = np.random.default_rng(23)
    inputs = {
        "x": rng.normal(size=(2, 6, 2)).astype(np.float32),
        "states": rng.normal(size=(2, 2, state_dim)).astype(np.float32),
        "mask": np.array([[True, False, True, True, True, True], [False] * 6]),
        "activity": np.array([[1, 0.5], [1, 1]], dtype=np.float32),
    }
    banks = {}
    if mode == "identity":
        expression = m.observe_identity(*bindings)
    elif mode == "affine":
        weight = m.BankBinding(
            "weight",
            "arti/jax-test@1",
            "observation",
            m.TensorType(("P", "S"), (4, state_dim), dtype="float32"),
        )
        bias = m.BankBinding(
            "bias", "arti/jax-test@1", "observation", m.TensorType(("P",), (4,), dtype="float32")
        )
        expression = m.observe_affine(*bindings, weight, bias)
        banks = {
            weight: rng.normal(size=(4, state_dim)).astype(np.float32),
            bias: rng.normal(size=(4,)).astype(np.float32),
        }
    else:
        expression = m.observe_fourier(*bindings, spatial_shape=(2, 3), state_mode=mode)
    parity(expression, inputs, banks, tolerance=8e-5)


def test_formula_scan_native_carry_and_gradient():
    carry = m.InputBinding("carry", m.TensorType(("B", "D"), ("B", 2), dtype="float32"))
    frame = m.InputBinding("frame", carry.value_type)
    body = m.FormulaProgram.build(outputs=(m.add(carry, frame),))
    scan = m.FormulaScan(
        body,
        axis="T",
        sequence_types={"frame": m.TensorType(("B", "T", "D"), ("B", 3, 2), dtype="float32")},
        carry_outputs={"carry": body.outputs[0]},
        emissions={"value": body.outputs[0]},
        max_length=8,
    )
    native = aj.bind_formula_scan(scan)
    inputs = {
        "carry": jnp.zeros((2, 2), dtype=jnp.float32),
        "frame": jnp.arange(12, dtype=jnp.float32).reshape(2, 3, 2),
    }
    result = jax.jit(native.apply)({}, inputs)
    expected = scan(
        inputs={name: torch.tensor(np.asarray(value)) for name, value in inputs.items()}, banks={}
    )
    for name in expected:
        np.testing.assert_allclose(result[name], expected[name].numpy())
    gradient = jax.jit(
        jax.grad(lambda frames: native.apply({}, {**inputs, "frame": frames})["carry.carry"].sum())
    )(inputs["frame"])
    np.testing.assert_array_equal(gradient, np.ones((2, 3, 2)))


@pytest.mark.parametrize("compiled", (False, True))
def test_instruction_dtype_relations_reuse_shared_contract(compiled):
    value_type = m.TensorType(("B", "D"), (1, 2))
    left, right = m.InputBinding("left", value_type), m.InputBinding("right", value_type)
    native = aj.bind_formula(m.FormulaProgram.build(outputs=(m.add(left, right),)))
    apply = jax.jit(native.apply) if compiled else native.apply
    with pytest.raises(m.FormulaBindingError) as error:
        apply(
            {},
            {
                "left": jnp.ones((1, 2), dtype=jnp.float16),
                "right": jnp.ones((1, 2), dtype=jnp.float32),
            },
        )
    assert error.value.code == "FF2_RUNTIME_DTYPE_MISMATCH"


def test_default_sinusoidal_accepts_integer_coordinates_as_float32():
    coordinates = m.InputBinding("coordinates", m.TensorType(("N",), (3,), dtype="int64"))
    expression = m.sinusoidal_position(coordinates, feature_axis="D", feature_size=4)
    native, _, inputs = parity(
        expression, {"coordinates": np.array([0, 1, 2], dtype=np.int64)}, gradients=False
    )
    assert next(iter(native.apply({}, inputs).values())).dtype == jnp.float32


def test_polar_observation_has_finite_zero_direction_vjp():
    bindings = (
        m.InputBinding("x", m.TensorType(("B", "N", "D"), (1, 6, 1), dtype="float32")),
        m.InputBinding("states", m.TensorType(("B", "T", "S"), (1, 1, 3), dtype="float32")),
        m.InputBinding("mask", m.TensorType(("B", "N"), (1, 6), dtype="boolean")),
        m.InputBinding("activity", m.TensorType(("B", "T"), (1, 1), dtype="float32")),
    )
    expression = m.observe_fourier(*bindings, spatial_shape=(2, 3), state_mode="polar")
    program = m.FormulaProgram.build(outputs=(expression,))
    native = aj.bind_formula(program)
    inputs = {
        "x": jnp.arange(6, dtype=jnp.float32).reshape(1, 6, 1),
        "states": jnp.array([[[1, 0, 0]]], dtype=jnp.float32),
        "mask": jnp.ones((1, 6), dtype=bool),
        "activity": jnp.ones((1, 1), dtype=jnp.float32),
    }
    gradient = jax.jit(
        jax.grad(
            lambda states: native.apply({}, {**inputs, "states": states})[program.outputs[0]][
                0, 0, 1, 0
            ]
        )
    )(inputs["states"])
    assert np.isfinite(gradient).all()
    np.testing.assert_array_equal(gradient, 0)
