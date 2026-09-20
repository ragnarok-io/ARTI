import pytest
import torch

from arti import FrameContext, TensorContext
from arti.legacy import ARTILayer
from arti.functional import apply_coord_frame_inverse


def _paired_coord(batch: int, tokens: int) -> torch.Tensor:
    coord = torch.zeros(batch, tokens, 2)
    coord[..., 1] = 1.0
    return coord


def _minimal_layer(**kwargs) -> ARTILayer:
    options = dict(
        input_dim=4,
        hidden_dim=4,
        operator_count=1,
        interface_slots=1,
        recall_slots=1,
        use_recall=False,
        use_virtual_recall=False,
        use_virtual_interface=False,
        use_pairwise_context=False,
        use_phase_mixer=False,
        use_layer_norm=False,
    )
    options.update(kwargs)
    return ARTILayer(**options).eval()


def test_tensor_context_intersects_visibility_with_valid_mask() -> None:
    valid_mask = torch.tensor([[True, True, False]])
    visibility = torch.ones(1, 3, 3, dtype=torch.bool)

    context = TensorContext(valid_mask=valid_mask, visibility=visibility)
    mask, effective = context.validate(
        batch=1,
        tokens=3,
        hidden_dim=4,
        coord_dim=0,
        configured_mode="none",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    assert torch.equal(mask, valid_mask)
    assert effective is not None
    assert not bool(effective[0, :, 2].any())
    assert not bool(effective[0, 2, :].any())


def test_tensor_context_rejects_implicit_mask_and_visibility_coercion() -> None:
    with pytest.raises(TypeError, match="valid_mask.*torch.bool"):
        TensorContext(valid_mask=torch.ones(1, 2)).validate(
            batch=1,
            tokens=2,
            hidden_dim=4,
            coord_dim=0,
            configured_mode="none",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    with pytest.raises(TypeError, match="visibility.*torch.bool"):
        TensorContext(
            valid_mask=torch.ones(1, 2, dtype=torch.bool),
            visibility=torch.ones(1, 2, 2),
        ).validate(
            batch=1,
            tokens=2,
            hidden_dim=4,
            coord_dim=0,
            configured_mode="none",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_paired_frame_requires_unit_coordinates_and_exact_dtype() -> None:
    coord = _paired_coord(1, 2)
    context = TensorContext(frame=FrameContext(coord=coord))
    context.validate(
        batch=1,
        tokens=2,
        hidden_dim=4,
        coord_dim=2,
        configured_mode="paired_rotation",
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    bad_coord = coord.clone()
    bad_coord[..., 0] = 0.5
    with pytest.raises(ValueError, match="unit rotation"):
        TensorContext(frame=FrameContext(coord=bad_coord)).validate(
            batch=1,
            tokens=2,
            hidden_dim=4,
            coord_dim=2,
            configured_mode="paired_rotation",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

    with pytest.raises(ValueError, match="dtype torch.float32"):
        TensorContext(frame=FrameContext(coord=coord.double())).validate(
            batch=1,
            tokens=2,
            hidden_dim=4,
            coord_dim=2,
            configured_mode="paired_rotation",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )


def test_disabled_frame_rejects_observer_coordinate_in_strict_context() -> None:
    with pytest.raises(ValueError, match="enabled coordinate-frame inverse"):
        TensorContext(
            frame=FrameContext(
                coord=torch.zeros(1, 2, 2),
                observer_coord=torch.zeros(1, 2),
            )
        ).validate(
            batch=1,
            tokens=2,
            hidden_dim=4,
            coord_dim=2,
            configured_mode="none",
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

def test_context_path_matches_legacy_explicit_path() -> None:
    torch.manual_seed(7)
    layer = ARTILayer(
        input_dim=4,
        hidden_dim=4,
        coord_dim=2,
        coord_frame_mode="paired_rotation",
        require_coord=True,
        require_visibility=True,
        operator_count=1,
        interface_slots=1,
        recall_slots=1,
        use_recall=False,
        use_virtual_recall=False,
        use_virtual_interface=False,
        use_pairwise_context=True,
        use_phase_mixer=False,
        use_layer_norm=False,
    ).eval()
    x = torch.randn(2, 3, 4)
    coord = _paired_coord(2, 3)
    mask = torch.tensor([[True, True, False], [True, False, True]])
    visibility = torch.ones(2, 3, 3, dtype=torch.bool)

    legacy = layer(x, coord=coord, mask=mask, visibility=visibility).y
    context = TensorContext(
        valid_mask=mask,
        visibility=visibility,
        frame=FrameContext(coord=coord),
    )
    strict = layer(x, context=context).y

    assert strict.shape == x.shape
    assert torch.equal(strict, legacy)


def test_context_rejects_mixed_legacy_arguments() -> None:
    layer = _minimal_layer()
    x = torch.randn(1, 2, 4)
    context = TensorContext()

    with pytest.raises(ValueError, match="cannot be combined"):
        layer(x, context=context, mask=torch.ones(1, 2, dtype=torch.bool))


def test_disabled_context_is_a_noop_contract_for_legacy_and_strict_paths() -> None:
    torch.manual_seed(11)
    layer = _minimal_layer(coord_dim=0)
    x = torch.randn(2, 3, 4)

    legacy = layer(x).y
    strict = layer(x, context=TensorContext()).y

    assert torch.equal(strict, legacy)


def test_visibility_blocks_an_invisible_source_from_pairwise_layer_output() -> None:
    torch.manual_seed(23)
    layer = _minimal_layer(use_pairwise_context=True)
    x = torch.randn(1, 3, 4)
    valid_mask = torch.ones(1, 3, dtype=torch.bool)
    visibility = torch.tensor(
        [
            [True, False, False],
            [True, True, True],
            [False, False, True],
        ],
        dtype=torch.bool,
    ).unsqueeze(0)
    context = TensorContext(valid_mask=valid_mask, visibility=visibility)

    baseline = layer(x, context=context).y
    changed_source = x.clone()
    changed_source[:, 1] += 100.0
    changed = layer(changed_source, context=context).y

    assert torch.equal(changed[:, 0], baseline[:, 0])
    assert not torch.equal(changed[:, 1], baseline[:, 1])


def test_strict_context_applies_nontrivial_paired_rotation_inverse() -> None:
    torch.manual_seed(29)
    layer = ARTILayer(
        input_dim=4,
        hidden_dim=4,
        coord_dim=2,
        coord_frame_mode="paired_rotation",
        require_coord=True,
        operator_count=1,
        interface_slots=1,
        recall_slots=1,
        use_recall=False,
        use_virtual_recall=False,
        use_virtual_interface=False,
        use_pairwise_context=False,
        use_phase_mixer=False,
        use_layer_norm=False,
    ).eval()
    reference = ARTILayer(
        input_dim=4,
        hidden_dim=4,
        coord_dim=2,
        coord_frame_mode="paired_rotation",
        require_coord=True,
        operator_count=1,
        interface_slots=1,
        recall_slots=1,
        use_recall=False,
        use_virtual_recall=False,
        use_virtual_interface=False,
        use_pairwise_context=False,
        use_phase_mixer=False,
        use_layer_norm=False,
    ).eval()
    reference.load_state_dict(layer.state_dict())

    x = torch.randn(1, 2, 4)
    coord = torch.zeros(1, 2, 2)
    coord[..., 0] = 1.0
    coord[..., 1] = 0.0
    identity_coord = _paired_coord(1, 2)
    rotated_context = TensorContext(frame=FrameContext(coord=coord))
    identity_context = TensorContext(frame=FrameContext(coord=identity_coord))

    strict_output = layer(x, context=rotated_context).y
    canonical_input = apply_coord_frame_inverse(x, coord, "paired_rotation")
    reference_output = reference(canonical_input, context=identity_context).y

    assert torch.allclose(strict_output, reference_output, atol=1e-6, rtol=1e-6)
    assert not torch.allclose(strict_output, reference(x, context=identity_context).y)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_strict_context_preserves_cuda_device_and_dtype() -> None:
    device = torch.device("cuda")
    dtype = torch.float16
    layer = _minimal_layer().to(device=device, dtype=dtype)
    x = torch.randn(2, 3, 4, device=device, dtype=dtype)
    context = TensorContext(
        valid_mask=torch.ones(2, 3, device=device, dtype=torch.bool),
        visibility=torch.ones(2, 3, 3, device=device, dtype=torch.bool),
    )

    output = layer(x, context=context)

    assert output.y.device.type == device.type
    assert output.y.device.index == torch.cuda.current_device()
    assert output.y.dtype == dtype
    assert output.pooled.device.type == device.type
    assert output.pooled.device.index == torch.cuda.current_device()
    assert output.pooled.dtype == dtype
