from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import torch

import arti.jax as arti_jax
from arti import functional as torch_functional
from arti.fit.batch_schema import attention_mask_to_visibility as torch_attention_mask_to_visibility


def test_shared_functional_contract_matches_torch() -> None:
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    mask = torch.tensor([[True, False, True]])
    logits = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    visibility = torch.tensor([[[True, False, True], [True, True, True], [True, False, True]]])
    np.testing.assert_allclose(arti_jax.masked_mean(jnp.asarray(x.numpy()), jnp.asarray(mask.numpy()), axis=1), torch_functional.masked_mean(x, mask, dim=1).numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(arti_jax.masked_softmax(jnp.asarray(logits.numpy()), jnp.asarray(visibility.numpy()), axis=-1), torch_functional.masked_softmax(logits, visibility, dim=-1).numpy(), rtol=1e-5, atol=1e-6)
    assert np.array_equal(arti_jax.ensure_visibility(jnp.asarray(visibility.numpy()), jnp.asarray(mask.numpy())), torch_functional.ensure_visibility(visibility, mask).numpy())
    assert np.array_equal(arti_jax.attention_mask_to_visibility(jnp.asarray(mask.numpy()), causal=True), torch_attention_mask_to_visibility(mask, causal=True).numpy())


def test_minimal_layer_matches_explicit_torch_formula() -> None:
    params = {
        "input_kernel": jnp.asarray([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.6]]),
        "coord_kernel": jnp.asarray([[0.2, -0.1], [0.5, 0.3]]),
        "bias": jnp.asarray([0.05, -0.02]),
    }
    x = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    coord = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    mask = torch.tensor([[True, False]])
    kernel = torch.tensor(np.asarray(params["input_kernel"]))
    coord_kernel = torch.tensor(np.asarray(params["coord_kernel"]))
    bias = torch.tensor(np.asarray(params["bias"]))
    expected_y = torch.tanh(torch.einsum("bnd,dh->bnh", x, kernel) + torch.einsum("bnc,ch->bnh", coord, coord_kernel) + bias)
    expected_y = expected_y * mask.unsqueeze(-1)
    expected_pooled = torch_functional.masked_mean(expected_y, mask, dim=1)
    actual = arti_jax.apply_layer(params, jnp.asarray(x.numpy()), coord=jnp.asarray(coord.numpy()), mask=jnp.asarray(mask.numpy()))
    np.testing.assert_allclose(actual["y"], expected_y.numpy(), rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(actual["pooled"], expected_pooled.numpy(), rtol=1e-5, atol=1e-6)


def test_observer_frame_modes_match_torch() -> None:
    x = torch.arange(16, dtype=torch.float32).reshape(1, 2, 8) / 10
    paired = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])
    for mode in ("none", "paired_rotation"):
        torch_value = torch_functional.apply_coord_frame_inverse(x, paired, mode=mode)
        jax_value = arti_jax.apply_coord_frame_inverse(jnp.asarray(x.numpy()), jnp.asarray(paired.numpy()), mode=mode)
        np.testing.assert_allclose(jax_value, torch_value.numpy(), rtol=1e-5, atol=1e-6)

    operators = torch.stack([torch.eye(8), torch.flip(torch.eye(8), dims=(0,))])
    coord = torch.nn.functional.one_hot(torch.tensor([[0, 1]]), num_classes=2).float()
    observer = torch.tensor([[0.0, 1.0]])
    torch_value = torch_functional.apply_coord_frame_inverse(x, coord, "operator_bank", operators, observer_coord=observer)
    jax_value = arti_jax.apply_coord_frame_inverse(
        jnp.asarray(x.numpy()),
        jnp.asarray(coord.numpy()),
        "operator_bank",
        jnp.asarray(operators.numpy()),
        observer_coord=jnp.asarray(observer.numpy()),
    )
    np.testing.assert_allclose(jax_value, torch_value.numpy(), rtol=1e-5, atol=1e-6)
