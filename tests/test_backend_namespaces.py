import pytest
import torch

import arti
import arti.backend as backend
import arti.jax as arti_jax
import arti.torch as arti_torch
import arti.torch.blocks as torch_blocks
import arti.torch.functional as torch_functional
import arti.torch.layers as torch_layers
import arti.torch.training as torch_training


def test_torch_backend_namespace_matches_root_api():
    assert arti_torch.ARTILayer is arti.ARTILayer
    assert arti_torch.ARTIResidualBlock is arti.ARTIResidualBlock
    assert arti_torch.virtual_recall_alignment_loss is arti.virtual_recall_alignment_loss
    assert arti_torch.experiential_recall_alignment_loss is arti.experiential_recall_alignment_loss
    assert torch_layers.ARTILayer is arti.ARTILayer
    assert torch_blocks.ARTIResidualBlock is arti.ARTIResidualBlock
    assert torch_training.virtual_recall_alignment_loss is arti.virtual_recall_alignment_loss
    assert torch_functional.apply_coord_frame_inverse is arti_torch.apply_coord_frame_inverse
    assert arti_torch.cuda_runtime_available is arti.cuda_runtime_available
    assert arti_torch.cuda_smoke_report is arti.cuda_smoke_report


def test_backend_discovery_reports_current_and_planned_backends():
    assert "torch" in arti.available_backends()
    if arti_jax.backend_status() == "available":
        assert "jax" in arti.available_backends()
        assert "jax" not in arti.planned_backends()
    else:
        assert "jax" in arti.planned_backends()


def test_jax_backend_status_distinguishes_unavailable(monkeypatch) -> None:
    backend.jax_backend_status.cache_clear()
    monkeypatch.setattr(backend, "find_spec", lambda name: None)
    assert backend.jax_backend_status() == "unavailable"
    assert "jax" not in backend.available_backends()
    assert "jax" in backend.planned_backends()
    backend.jax_backend_status.cache_clear()


def test_jax_backend_status_distinguishes_broken(monkeypatch) -> None:
    import builtins

    original_import = builtins.__import__
    backend.jax_backend_status.cache_clear()
    monkeypatch.setattr(backend, "find_spec", lambda name: object())

    def broken_import(name, *args, **kwargs):
        if name == "jax":
            raise OSError("broken jaxlib")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_import)
    assert backend.jax_backend_status() == "broken"
    assert "jax" not in backend.available_backends()
    assert "jax" in backend.planned_backends()
    backend.jax_backend_status.cache_clear()


def test_jax_namespace_reports_optional_backend_status():
    assert arti_jax.backend_status() in {"available", "broken", "unavailable"}
    smoke = arti_jax.smoke_report()
    assert smoke["backend_status"] == arti_jax.backend_status()
    if arti_jax.backend_status() == "unavailable":
        assert smoke["smoke_status"] == "skipped"
        with pytest.raises(arti_jax.ARTIJAXBackendUnavailableError):
            arti_jax.require_jax_backend()
    elif arti_jax.backend_status() == "available":
        assert smoke["smoke_status"] == "passed"
        assert smoke["forward_ok"] is True
        assert smoke["jit_ok"] is True
        assert smoke["grad_ok"] is True
        arti_jax.require_jax_backend()
    else:
        assert smoke["smoke_status"] == "failed"
        with pytest.raises(arti_jax.ARTIJAXBackendUnavailableError):
            arti_jax.require_jax_backend()


def test_jax_functional_smoke_when_available():
    if arti_jax.backend_status() == "unavailable":
        pytest.skip("JAX optional dependency is not installed")
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    key = jax.random.PRNGKey(0)
    params = arti_jax.init_layer(key, input_dim=4, hidden_dim=6, coord_dim=2)
    x = jnp.ones((2, 3, 4))
    coord = jnp.ones((2, 3, 2))
    mask = jnp.array([[True, True, False], [True, False, False]])
    out = arti_jax.apply_layer(params, x, coord=coord, mask=mask)

    assert out["y"].shape == (2, 3, 6)
    assert out["pooled"].shape == (2, 6)
    assert out["diagnostics"]["mask_coverage"].shape == (2,)

    pooled = jax.jit(lambda a, b, c: arti_jax.apply_layer(params, a, coord=b, mask=c)["pooled"])(x, coord, mask)
    assert pooled.shape == (2, 6)

    def objective(kernel):
        updated = dict(params)
        updated["input_kernel"] = kernel
        return jnp.sum(arti_jax.apply_layer(updated, x, coord=coord, mask=mask)["pooled"])

    grad = jax.grad(objective)(params["input_kernel"])
    assert grad.shape == params["input_kernel"].shape


def test_jax_mask_helpers_when_available():
    if arti_jax.backend_status() == "unavailable":
        pytest.skip("JAX optional dependency is not installed")
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    x = jnp.array([[[1.0, 2.0], [3.0, 4.0], [100.0, 100.0]], [[2.0, 4.0], [8.0, 10.0], [0.0, 0.0]]])
    mask = jnp.array([[True, True, False], [True, False, False]])
    mean = arti_jax.masked_mean(x, mask, axis=1)
    assert jnp.allclose(mean, jnp.array([[2.0, 3.0], [2.0, 4.0]]))
    assert jnp.allclose(arti_jax.mask_coverage(mask), jnp.array([2.0 / 3.0, 1.0 / 3.0]))

    logits = jnp.array([[1.0, 2.0, 9.0], [3.0, 0.0, -1.0]])
    weights = arti_jax.masked_softmax(logits, mask, axis=-1)
    assert jnp.allclose(weights[:, -1], jnp.array([0.0, 0.0]))
    assert jnp.allclose(jnp.sum(weights, axis=-1), jnp.array([1.0, 1.0]))

    jitted = jax.jit(lambda values, valid: arti_jax.masked_mean(values, valid, axis=1))(x, mask)
    assert jnp.allclose(jitted, mean)


def test_jax_visibility_helpers_when_available():
    if arti_jax.backend_status() == "unavailable":
        pytest.skip("JAX optional dependency is not installed")
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    mask = jnp.array([[True, True, False]])
    visibility = arti_jax.ensure_visibility(None, mask)
    assert visibility.shape == (1, 3, 3)
    assert bool(visibility[0, 0, 0])
    assert bool(visibility[0, 1, 0])
    assert not bool(visibility[0, 0, 2])
    assert not bool(visibility[0, 2, 0])

    explicit = jnp.array([[[True, False, True], [True, True, True], [True, True, True]]])
    gated = arti_jax.ensure_visibility(explicit, mask)
    assert not bool(gated[0, 0, 1])
    assert not bool(gated[0, 0, 2])
    assert bool(gated[0, 1, 0])

    causal = arti_jax.attention_mask_to_visibility(mask, causal=True)
    assert bool(causal[0, 1, 0])
    assert not bool(causal[0, 0, 1])
    assert not bool(causal[0, 2, 0])

    jitted = jax.jit(lambda valid: arti_jax.attention_mask_to_visibility(valid, causal=True))(mask)
    assert jnp.array_equal(jitted, causal)


def test_jax_coord_frame_inverse_when_available():
    if arti_jax.backend_status() == "unavailable":
        pytest.skip("JAX optional dependency is not installed")
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    canonical = jnp.array([[[1.0, 0.0, 0.0, 1.0]]])
    sin_t = jnp.array([[1.0]])
    cos_t = jnp.array([[0.0]])
    observed = jnp.stack(
        (
            cos_t[..., None] * canonical[..., 0::2] - sin_t[..., None] * canonical[..., 1::2],
            sin_t[..., None] * canonical[..., 0::2] + cos_t[..., None] * canonical[..., 1::2],
        ),
        axis=-1,
    ).reshape(canonical.shape)
    coord = jnp.array([[[1.0, 0.0]]])

    recovered = arti_jax.apply_coord_frame_inverse(observed, coord, mode="paired_rotation")
    assert jnp.allclose(recovered, canonical)
    jitted = jax.jit(lambda values, phase: arti_jax.apply_coord_frame_inverse(values, phase, mode="paired_rotation"))(observed, coord)
    assert jnp.allclose(jitted, canonical)


def test_jax_operator_bank_observer_frame_when_available():
    if arti_jax.backend_status() == "unavailable":
        pytest.skip("JAX optional dependency is not installed")
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    identity = jnp.eye(2)
    rotate_ccw = jnp.array([[0.0, -1.0], [1.0, 0.0]])
    rotate_cw = jnp.array([[0.0, 1.0], [-1.0, 0.0]])
    inverse = jnp.stack([identity, rotate_cw])
    observed = jnp.array([[[1.0, 0.0], [0.0, 1.0]]])
    coord = jnp.array([[[1.0, 0.0], [0.0, 1.0]]])

    own_frame = arti_jax.apply_coord_frame_inverse(observed, coord, mode="operator_bank", frame_operators=inverse)
    assert jnp.allclose(own_frame, jnp.array([[[1.0, 0.0], [1.0, 0.0]]]))

    observer_coord = jnp.array([[0.0, 1.0]])
    observer_view = arti_jax.apply_coord_frame_inverse(
        observed,
        coord,
        mode="operator_bank",
        frame_operators=inverse,
        observer_coord=observer_coord,
    )
    assert jnp.allclose(observer_view, jnp.array([[[0.0, -1.0], [1.0, 0.0]]]))

    jitted = jax.jit(
        lambda values, phase, observer: arti_jax.apply_coord_frame_inverse(
            values,
            phase,
            mode="operator_bank",
            frame_operators=inverse,
            observer_coord=observer,
        )
    )(observed, coord, observer_coord)
    assert jnp.allclose(jitted, observer_view)


def test_jax_functional_helpers_match_torch_when_available():
    if arti_jax.backend_status() == "unavailable":
        pytest.skip("JAX optional dependency is not installed")
    jnp = pytest.importorskip("jax.numpy")

    x_torch = torch.tensor(
        [[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [100.0, 100.0, 100.0, 100.0]]],
        dtype=torch.float32,
    )
    mask_torch = torch.tensor([[True, True, False]])
    logits_torch = torch.tensor([[[1.0, 2.0, 9.0], [3.0, 0.0, -1.0], [4.0, 4.0, 4.0]]], dtype=torch.float32)
    visibility_torch = torch.tensor([[[True, False, True], [True, True, True], [True, True, True]]])
    coord_torch = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]], dtype=torch.float32)
    frame_operators_torch = torch.stack(
        [
            torch.eye(4),
            torch.tensor(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                    [0.0, 0.0, -1.0, 0.0],
                ],
                dtype=torch.float32,
            ),
        ]
    )
    observer_coord_torch = torch.tensor([[0.0, 1.0]], dtype=torch.float32)

    x_jax = jnp.asarray(x_torch.numpy())
    mask_jax = jnp.asarray(mask_torch.numpy())
    logits_jax = jnp.asarray(logits_torch.numpy())
    visibility_jax = jnp.asarray(visibility_torch.numpy())
    coord_jax = jnp.asarray(coord_torch.numpy())
    frame_operators_jax = jnp.asarray(frame_operators_torch.numpy())
    observer_coord_jax = jnp.asarray(observer_coord_torch.numpy())

    torch_mean = torch_functional.masked_mean(x_torch, mask_torch, dim=1)
    jax_mean = arti_jax.masked_mean(x_jax, mask_jax, axis=1)
    assert torch.allclose(torch.tensor(jax_mean.tolist()), torch_mean)

    torch_visibility = torch_functional.ensure_visibility(visibility_torch, mask_torch)
    jax_visibility = arti_jax.ensure_visibility(visibility_jax, mask_jax)
    assert torch.equal(torch.tensor(jax_visibility.tolist()), torch_visibility)

    torch_softmax = torch_functional.masked_softmax(logits_torch, torch_visibility, dim=-1)
    jax_softmax = arti_jax.masked_softmax(logits_jax, jax_visibility, axis=-1)
    assert torch.allclose(torch.tensor(jax_softmax.tolist()), torch_softmax, atol=1e-6)

    torch_frame = torch_functional.apply_coord_frame_inverse(
        x_torch,
        coord_torch,
        mode="operator_bank",
        frame_operators=frame_operators_torch,
        observer_coord=observer_coord_torch,
    )
    jax_frame = arti_jax.apply_coord_frame_inverse(
        x_jax,
        coord_jax,
        mode="operator_bank",
        frame_operators=frame_operators_jax,
        observer_coord=observer_coord_jax,
    )
    assert torch.allclose(torch.tensor(jax_frame.tolist()), torch_frame, atol=1e-6)


def test_torch_cuda_helpers_reflect_runtime():
    report = arti_torch.cuda_device_report()
    assert report["cuda_available"] == torch.cuda.is_available()
    assert report["device_count"] == torch.cuda.device_count()
    smoke = arti_torch.cuda_smoke_report(size=2)
    assert smoke["cuda_available"] == torch.cuda.is_available()
    if torch.cuda.is_available():
        assert arti_torch.require_cuda().type == "cuda"
        assert smoke["smoke_status"] == "passed"
        assert smoke["allocation_ok"] is True
        assert smoke["compute_ok"] is True
        assert smoke["smoke_tensor_shape"] == [2, 2]
    else:
        assert smoke["smoke_status"] == "skipped"
        assert smoke["allocation_ok"] is False
        with pytest.raises(RuntimeError):
            arti_torch.require_cuda()
    with pytest.raises(ValueError):
        arti_torch.cuda_smoke_report(size=0)
