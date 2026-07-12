from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import arti.jax as arti_jax


def _params(coord_dim: int = 2):
    return arti_jax.init_layer(jax.random.PRNGKey(7), input_dim=4, hidden_dim=3, coord_dim=coord_dim)


@pytest.mark.parametrize("sequence", [False, True])
@pytest.mark.parametrize("with_context", [False, True])
def test_apply_layer_eager_and_jit_match(sequence: bool, with_context: bool) -> None:
    params = _params(2 if with_context else 0)
    x = jnp.arange(24 if sequence else 8, dtype=jnp.float32).reshape((2, 3, 4) if sequence else (2, 4)) / 10
    coord = None
    mask = None
    if with_context:
        coord = jnp.ones((2, 3, 2) if sequence else (2, 2), dtype=jnp.float32)
        mask = jnp.array([[True, False, True], [True, True, False]]) if sequence else jnp.array([True, False])
    eager = arti_jax.apply_layer(params, x, coord=coord, mask=mask)
    compiled = jax.jit(lambda p, values, phase, valid: arti_jax.apply_layer(p, values, coord=phase, mask=valid))(params, x, coord, mask)
    np.testing.assert_allclose(compiled["y"], eager["y"], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(compiled["pooled"], eager["pooled"], rtol=1e-5, atol=1e-6)
    assert bool(jnp.all(jnp.isfinite(compiled["y"])))


def test_whole_parameter_tree_supports_jitted_value_and_grad() -> None:
    params = _params()
    x = jnp.ones((2, 3, 4), dtype=jnp.float32)
    coord = jnp.ones((2, 3, 2), dtype=jnp.float32)
    mask = jnp.array([[True, True, False], [True, False, False]])

    def loss(p):
        return jnp.square(arti_jax.apply_layer(p, x, coord=coord, mask=mask)["pooled"]).mean()

    value, gradients = jax.jit(jax.value_and_grad(loss))(params)
    assert bool(jnp.isfinite(value))
    assert jax.tree_util.tree_structure(gradients) == jax.tree_util.tree_structure(params)
    for parameter, gradient in zip(jax.tree_util.tree_leaves(params), jax.tree_util.tree_leaves(gradients)):
        assert gradient.shape == parameter.shape
        assert bool(jnp.all(jnp.isfinite(gradient)))


def test_vmap_single_matches_direct_batch_and_composition_orders() -> None:
    params = _params()
    x = jnp.arange(24, dtype=jnp.float32).reshape(2, 3, 4) / 10
    coord = jnp.arange(12, dtype=jnp.float32).reshape(2, 3, 2) / 10
    mask = jnp.array([[True, False, True], [True, True, False]])
    direct = arti_jax.apply_layer(params, x, coord=coord, mask=mask)
    single = lambda values, phase, valid: arti_jax.apply_layer_single(params, values, coord=phase, mask=valid)
    mapped = jax.vmap(single)(x, coord, mask)
    jit_vmap = jax.jit(jax.vmap(single))(x, coord, mask)
    vmap_jit = jax.vmap(jax.jit(single))(x, coord, mask)
    for candidate in (mapped, jit_vmap, vmap_jit):
        np.testing.assert_allclose(candidate["y"], direct["y"], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(candidate["pooled"], direct["pooled"], rtol=1e-5, atol=1e-6)
        np.testing.assert_allclose(
            candidate["diagnostics"]["mask_coverage"],
            direct["diagnostics"]["mask_coverage"],
            rtol=1e-5,
            atol=1e-6,
        )


def test_jax_contract_rejects_invalid_shapes() -> None:
    params = _params()
    with pytest.raises(ValueError, match="x must have shape"):
        arti_jax.apply_layer(params, jnp.ones((4,)))
    with pytest.raises(ValueError, match="coord must have shape"):
        arti_jax.apply_layer(params, jnp.ones((2, 3, 4)), coord=jnp.ones((2, 2)))
    with pytest.raises(ValueError, match="mask must have shape"):
        arti_jax.apply_layer(params, jnp.ones((2, 3, 4)), coord=jnp.ones((2, 3, 2)), mask=jnp.ones((2, 1)))
    with pytest.raises(ValueError, match="visibility must have shape"):
        arti_jax.ensure_visibility(jnp.ones((2, 3, 1)), jnp.ones((2, 3)))
    with pytest.raises(ValueError, match="even latent dimension"):
        arti_jax.apply_coord_frame_inverse(jnp.ones((1, 2, 3)), jnp.ones((1, 2, 2)), mode="paired_rotation")
    with pytest.raises(ValueError, match="frame_operators must have shape"):
        arti_jax.apply_coord_frame_inverse(
            jnp.ones((1, 2, 4)),
            jnp.ones((1, 2, 2)),
            mode="operator_bank",
            frame_operators=jnp.ones((2, 3, 3)),
        )


def test_parameter_tree_rejects_unknown_and_non_differentiable_leaves() -> None:
    params = _params(0)
    with pytest.raises(ValueError, match="unsupported entries"):
        arti_jax.apply_layer({**params, "config": "not-an-array"}, jnp.ones((1, 4)))
    with pytest.raises(ValueError, match="floating-point or complex"):
        arti_jax.apply_layer({**params, "bias": jnp.ones((3,), dtype=jnp.int32)}, jnp.ones((1, 4)))


def test_none_coord_inverse_does_not_require_coord_contract() -> None:
    x = jnp.ones((1, 2, 4))
    output = arti_jax.apply_coord_frame_inverse(x, None, mode="none")
    np.testing.assert_array_equal(output, x)


def test_all_masked_contract_is_finite_and_zero() -> None:
    params = _params()
    output = arti_jax.apply_layer(
        params,
        jnp.ones((2, 3, 4)),
        coord=jnp.ones((2, 3, 2)),
        mask=jnp.zeros((2, 3), dtype=bool),
    )
    assert bool(jnp.all(output["y"] == 0))
    assert bool(jnp.all(output["pooled"] == 0))
    assert bool(jnp.all(output["diagnostics"]["mask_coverage"] == 0))


def test_smoke_report_covers_every_transformation() -> None:
    report = arti_jax.smoke_report()
    assert report["smoke_status"] == "passed"
    assert report["forward_ok"] is True
    assert report["jit_ok"] is True
    assert report["grad_ok"] is True
    assert report["vmap_ok"] is True
