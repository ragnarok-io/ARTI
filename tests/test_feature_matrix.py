from __future__ import annotations

import io

import pytest
import torch

import arti


CASES = (
    ("minimal", arti.features()),
    ("phase", arti.features(phase=True, coord_dim=2)),
    ("interface", arti.features(virtual_interface=True)),
    ("pairwise", arti.features(pairwise_context=True)),
    ("recall", arti.features(recall=True)),
    ("virtual_recall", arti.features(virtual_recall=True)),
    ("phase_recall", arti.features(phase=True, coord_dim=2, recall=True)),
    ("interface_recall", arti.features(virtual_interface=True, recall=True)),
    ("phase_pairwise", arti.features(phase=True, coord_dim=2, pairwise_context=True)),
    ("visible_interface", arti.features(visibility=True, virtual_interface=True)),
)


@pytest.mark.parametrize(("name", "selected"), CASES, ids=[case[0] for case in CASES])
def test_feature_combinations_forward_mask_gradient_and_parameter_presence(name: str, selected: arti.FeatureConfig) -> None:
    layer = arti.nn.Layer(8, features=selected)
    x = torch.randn(2, 5, 8, requires_grad=True)
    mask = torch.tensor([[True, True, True, False, False], [True, True, True, True, True]])
    kwargs = {"mask": mask}
    if selected.coord_dim:
        kwargs["coord"] = torch.randn(2, 5, selected.coord_dim)
    if selected.visibility:
        kwargs["visibility"] = mask.unsqueeze(1) & mask.unsqueeze(2)

    output = layer(x, **kwargs)
    output.y.square().mean().backward()
    names = tuple(parameter_name for parameter_name, _ in layer.named_parameters())

    assert output.y.shape == x.shape
    assert torch.count_nonzero(output.y[~mask]) == 0
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert any("state.phase" in parameter_name for parameter_name in names) is selected.phase
    assert any("state.interface" in parameter_name for parameter_name in names) is selected.virtual_interface
    assert any("state.recall" in parameter_name for parameter_name in names) is selected.recall
    assert any("virtual_recall_proj" in parameter_name for parameter_name in names) is selected.virtual_recall
    assert (output.virtual_y is not None) is selected.virtual_recall


@pytest.mark.parametrize(("name", "selected"), CASES, ids=[case[0] for case in CASES])
def test_feature_combinations_state_dict_roundtrip(name: str, selected: arti.FeatureConfig, paired_rng) -> None:
    source = arti.nn.Layer(8, features=selected).eval()
    restored = arti.nn.Layer(8, features=arti.FeatureConfig.from_dict(selected.to_dict())).eval()
    buffer = io.BytesIO()
    torch.save(source.state_dict(), buffer)
    buffer.seek(0)
    restored.load_state_dict(torch.load(buffer, weights_only=True))

    x = torch.randn(2, 4, 8)
    kwargs = {}
    if selected.coord_dim:
        kwargs["coord"] = torch.randn(2, 4, selected.coord_dim)
    if selected.visibility:
        kwargs["visibility"] = torch.ones(2, 4, 4, dtype=torch.bool)
    expected, actual = paired_rng(
        lambda: source(x, **kwargs).y,
        lambda: restored(x, **kwargs).y,
    )
    assert torch.allclose(expected, actual)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("profile_name", ("minimal", "recall", "multisource"))
def test_profiles_cuda_amp(profile_name: str) -> None:
    overrides = {"coord_dim": 2} if profile_name == "multisource" else {}
    layer = arti.nn.Layer(16, profile=profile_name, **overrides).cuda()
    x = torch.randn(4, 6, 16, device="cuda")
    kwargs = {}
    if profile_name == "multisource":
        kwargs["coord"] = torch.nn.functional.one_hot(torch.zeros(4, 6, dtype=torch.long, device="cuda"), num_classes=2).float()
        kwargs["visibility"] = torch.ones(4, 6, 6, dtype=torch.bool, device="cuda")
        kwargs["frame_operators"] = torch.eye(16, device="cuda").repeat(2, 1, 1)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        output = layer(x, **kwargs)
    assert output.y.is_cuda
    assert output.y.dtype in {torch.bfloat16, torch.float32}
    assert torch.isfinite(output.y).all()
