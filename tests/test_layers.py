import torch

from arti import ARTIConfig
from arti import ARTILayer
from arti.functional import apply_coord_frame_inverse
from arti.layers import ARTILatentRecallField, ARTILatentTensorLayer
from arti.training import experiential_recall_alignment_loss, experiential_recall_selectivity_loss, virtual_recall_alignment_loss


def test_layer_sequence_shapes_and_diagnostics():
    layer = ARTILayer(input_dim=32, coord_dim=8, hidden_dim=64)
    x = torch.randn(4, 16, 32)
    coord = torch.randn(4, 16, 8)
    mask = torch.ones(4, 16, dtype=torch.bool)

    out = layer(x, coord=coord, mask=mask)

    assert out.y.shape == (4, 16, 64)
    assert out.virtual_y is not None
    assert out.virtual_y.shape == (4, 16, 64)
    assert out.recall_trace is not None
    assert out.recall_prediction is not None
    assert out.recall_trace.shape == out.virtual_y.shape
    assert out.recall_prediction.shape == out.virtual_y.shape
    assert out.pooled.shape == (4, 64)
    assert "operator_weights" in out.diagnostics
    assert "mask_coverage" in out.diagnostics
    assert "experiential_recall_familiarity" in out.diagnostics
    assert "experiential_recall_trace_norm" in out.diagnostics
    assert "recall_recognition" in out.diagnostics


def test_recall_recognition_blocks_unseen_trace_before_half():
    field = ARTILatentRecallField(2, 2, recognition_mode="explicit", recognition_threshold=0.5, recognition_temperature=0.05)
    with torch.no_grad():
        field.bank.copy_(torch.eye(2))
        field.query.weight.copy_(torch.eye(2))
        field.gate.weight.zero_()
        field.gate.bias.fill_(10.0)
    z = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]]])
    context, _, _, recognition = field(z, torch.ones(1, 3, dtype=torch.bool))
    assert recognition[0, 0] > 0.99
    assert recognition[0, 1] > 0.99
    assert recognition[0, 2] < 0.01
    assert context[0, 2].norm() < 1e-3


def test_alignment_recognition_learns_seen_and_unseen_without_fixed_threshold():
    field = ARTILatentRecallField(2, 2, recognition_mode="alignment")
    with torch.no_grad():
        field.bank.copy_(torch.eye(2))
        field.query.weight.copy_(torch.eye(2))
        field.gate.weight.zero_()
        field.gate.bias.fill_(10.0)
    for parameter in field.parameters():
        parameter.requires_grad_(parameter is field.alignment_recognizer.weight or parameter is field.alignment_recognizer.bias)
    seen = torch.tensor([[[1.0, 0.0]]])
    unseen = torch.tensor([[[-1.0, 0.0]]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    target = torch.tensor([[[0.6697615, 0.3302385]]])
    optimizer = torch.optim.Adam(field.alignment_recognizer.parameters(), lr=0.1)
    for _ in range(80):
        seen_context, _, _, _ = field(seen, mask)
        unseen_context, _, _, _ = field(unseen, mask)
        loss = torch.nn.functional.mse_loss(seen_context, target) + unseen_context.square().mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    _, _, _, seen_recognition = field(seen, mask)
    unseen_context, _, _, unseen_recognition = field(unseen, mask)
    assert seen_recognition.item() > 0.9
    assert unseen_recognition.item() < 0.1
    assert unseen_context.norm() < 0.1


def test_recall_recognition_mode_validation():
    try:
        ARTIConfig(input_dim=4, recall_recognition_mode="unknown")
    except ValueError as exc:
        assert "recall_recognition_mode" in str(exc)
    else:
        raise AssertionError("invalid recall_recognition_mode should fail")


def test_layer_vector_input_round_trips_rank():
    layer = ARTILayer(input_dim=32, hidden_dim=32)
    x = torch.randn(4, 32)

    out = layer(x)

    assert out.y.shape == (4, 32)
    assert out.pooled.shape == (4, 32)


def test_external_recall_can_influence_output():
    torch.manual_seed(7)
    layer = ARTILayer(input_dim=8, hidden_dim=8, recall_steps=1)
    x = torch.randn(2, 5, 8)
    recall_a = torch.zeros(2, 1, 8)
    recall_b = torch.zeros(2, 1, 8)
    recall_b[:, :, 0] = 4.0

    out_a = layer(x, recall=recall_a).pooled
    out_b = layer(x, recall=recall_b).pooled

    assert not torch.allclose(out_a, out_b)
    assert out_b.shape == out_a.shape


def test_recall_activation_defaults_to_half_and_can_be_disabled():
    torch.manual_seed(13)
    default_layer = ARTILayer(input_dim=8, hidden_dim=8, recall_steps=1, use_pairwise_context=False)
    raw_layer = ARTILayer(input_dim=8, hidden_dim=8, recall_steps=1, recall_activation="none", use_pairwise_context=False)
    x = torch.randn(2, 5, 8)
    recall = torch.randn(2, 2, 8)

    default_out = default_layer(x, recall=recall)
    raw_out = raw_layer(x, recall=recall)

    assert default_layer.config.recall_activation == "half"
    assert torch.all(default_out.diagnostics["recall_activation_half"] == 1.0)
    assert torch.all(raw_out.diagnostics["recall_activation_half"] == 0.0)
    assert torch.isfinite(default_out.diagnostics["recall_activation_survival_ratio"]).all()
    assert torch.isfinite(raw_out.diagnostics["recall_activation_survival_ratio"]).all()


def test_recall_activation_config_validation():
    try:
        ARTIConfig(input_dim=4, recall_activation="gelu")
    except ValueError as exc:
        assert "recall_activation" in str(exc)
    else:
        raise AssertionError("invalid recall_activation should fail")


def test_pairwise_context_can_be_disabled_for_interface_only_path():
    layer = ARTILayer(input_dim=8, hidden_dim=8, recall_steps=0, use_pairwise_context=False)
    x = torch.randn(2, 6, 8)

    out = layer(x)

    assert out.y.shape == (2, 6, 8)
    assert out.diagnostics["visibility_weights"].shape == (2, 6, 0)


def test_phase_mixer_can_be_disabled_independently():
    layer = ARTILayer(input_dim=8, coord_dim=3, hidden_dim=8, recall_steps=0, use_phase_mixer=False)
    x = torch.randn(2, 5, 8)
    coord = torch.randn(2, 5, 3)

    out = layer(x, coord=coord)

    assert out.y.shape == (2, 5, 8)
    assert out.diagnostics["operator_weights"].shape == (2, 5, 0)
    assert out.diagnostics["phase_receptor_gain"].shape == (2, 5, 0, 8)


def test_virtual_interface_can_be_disabled_independently():
    layer = ARTILayer(input_dim=8, hidden_dim=8, recall_steps=0, use_virtual_interface=False)
    x = torch.randn(2, 5, 8)

    out = layer(x)

    assert out.y.shape == (2, 5, 8)
    assert out.diagnostics["interface_read_weights"].shape == (2, 5, 0)
    assert out.diagnostics["interface_write_weights"].shape == (2, 5, 0)


def test_all_optional_mechanisms_can_be_disabled_for_plain_residual_transform():
    layer = ARTILayer(
        input_dim=8,
        hidden_dim=8,
        recall_steps=0,
        use_phase_mixer=False,
        use_virtual_interface=False,
        use_pairwise_context=False,
        use_recall=False,
        use_virtual_recall=False,
        fallback_context="none",
    )
    x = torch.randn(2, 5, 8)

    out = layer(x)

    assert out.y.shape == (2, 5, 8)
    assert out.diagnostics["operator_weights"].numel() == 0
    assert out.diagnostics["interface_read_weights"].numel() == 0
    assert out.diagnostics["visibility_weights"].numel() == 0
    assert out.diagnostics["recall_bank_weights"].numel() == 0
    assert out.virtual_y is None
    assert out.recall_trace is None
    assert out.recall_prediction is None
    parameter_names = tuple(name for name, _ in layer.named_parameters())
    assert not any("state.phase" in name for name in parameter_names)
    assert not any("state.interface" in name for name in parameter_names)
    assert not any("state.recall" in name for name in parameter_names)
    assert not any("virtual_recall_proj" in name for name in parameter_names)


def test_no_phase_configuration_requires_no_coord_or_fallback_identity():
    layer = ARTILayer(
        input_dim=8,
        hidden_dim=8,
        coord_dim=0,
        use_phase_mixer=False,
        coord_frame_mode="none",
        fallback_context="none",
        use_recall=False,
        use_virtual_recall=False,
    )
    x = torch.randn(2, 5, 8)

    out = layer(x)

    assert out.y.shape == x.shape
    assert torch.count_nonzero(out.diagnostics["coord_frame_delta"]) == 0
    assert torch.count_nonzero(out.diagnostics["observer_frame_active"]) == 0


def test_disabled_mechanisms_accept_zero_capacity_without_allocating_parameters():
    config = ARTIConfig(
        input_dim=8,
        operator_count=0,
        interface_slots=0,
        recall_slots=0,
        use_phase_mixer=False,
        use_virtual_interface=False,
        use_pairwise_context=False,
        use_recall=False,
        use_virtual_recall=False,
    )
    layer = ARTILatentTensorLayer(config)

    assert layer(torch.randn(1, 3, 8)).y.shape == (1, 3, 8)


def test_coord_frame_inverse_recovers_paired_rotation():
    theta = torch.tensor([[0.25, -0.75]])
    coord = torch.stack([torch.sin(theta), torch.cos(theta)], dim=-1)
    z = torch.randn(1, 2, 8)
    even = z[..., 0::2]
    odd = z[..., 1::2]
    observed = torch.empty_like(z)
    observed[..., 0::2] = torch.cos(theta).unsqueeze(-1) * even - torch.sin(theta).unsqueeze(-1) * odd
    observed[..., 1::2] = torch.sin(theta).unsqueeze(-1) * even + torch.cos(theta).unsqueeze(-1) * odd

    recovered = apply_coord_frame_inverse(observed, coord, "paired_rotation")

    assert torch.allclose(recovered, z, atol=1e-6)


def test_layer_reports_coord_frame_delta_when_enabled():
    layer = ARTILayer(input_dim=8, coord_dim=2, hidden_dim=8, recall_steps=0, coord_frame_mode="paired_rotation")
    x = torch.randn(2, 4, 8)
    coord = torch.zeros(2, 4, 2)
    coord[..., 1] = 1.0

    out = layer(x, coord=coord)

    assert out.y.shape == (2, 4, 8)
    assert "coord_frame_delta" in out.diagnostics


def test_random_coord_fallback_is_stable_when_coord_is_omitted():
    torch.manual_seed(31)
    layer = ARTILayer(input_dim=8, coord_dim=3, hidden_dim=8, recall_steps=0, fallback_context="random_coord", fallback_slots=4)
    x = torch.randn(2, 6, 8)

    coord_a = layer._resolve_coord(None, 2, 6, x.device, x.dtype)
    coord_b = layer._resolve_coord(None, 2, 6, x.device, x.dtype)
    out = layer(x)

    assert torch.allclose(coord_a, coord_b)
    assert coord_a.shape == (2, 6, 3)
    assert not torch.allclose(coord_a, torch.zeros_like(coord_a))
    assert out.y.shape == (2, 6, 8)


def test_random_context_fallback_supplies_visibility_when_omitted():
    layer = ARTILayer(input_dim=4, coord_dim=2, hidden_dim=4, recall_steps=0, fallback_context="random_context", fallback_slots=3)
    x = torch.randn(1, 4, 4)
    mask = torch.tensor([[True, True, False, True]])

    visibility = layer._resolve_visibility(None, mask)
    out = layer(x, mask=mask)

    assert visibility is not None
    assert visibility.shape == (1, 4, 4)
    assert visibility[0, 0].tolist() == [True, True, False, True]
    assert out.y.shape == (1, 4, 4)


def test_operator_bank_frame_inverse_recovers_context_observer():
    operators = torch.stack(
        [
            torch.eye(4),
            torch.tensor(
                [
                    [0.0, 1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, -1.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            ),
        ]
    )
    inverse = torch.linalg.inv(operators)
    frame_id = torch.tensor([[0, 1, 0]])
    coord = torch.nn.functional.one_hot(frame_id, num_classes=2).float()
    canonical = torch.randn(1, 3, 4)
    observed = torch.einsum("bnk,kde,bne->bnd", coord, operators, canonical)

    recovered = apply_coord_frame_inverse(observed, coord, "operator_bank", inverse)

    assert torch.allclose(recovered, canonical, atol=1e-6)


def test_operator_bank_observer_frame_keeps_relative_phase_difference():
    operators = torch.stack(
        [
            torch.eye(4),
            torch.tensor(
                [
                    [0.0, -1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0, -1.0],
                    [0.0, 0.0, 1.0, 0.0],
                ]
            ),
        ]
    )
    inverse = torch.linalg.inv(operators)
    frame_id = torch.tensor([[0, 1, 0]])
    coord = torch.nn.functional.one_hot(frame_id, num_classes=2).float()
    observer_coord = torch.nn.functional.one_hot(torch.tensor([1]), num_classes=2).float()
    canonical = torch.randn(1, 3, 4)
    observed = torch.einsum("bnk,kde,bne->bnd", coord, operators, canonical)

    observer_view = apply_coord_frame_inverse(observed, coord, "operator_bank", inverse, observer_coord=observer_coord)
    query_inverse = inverse[1]
    expected = torch.einsum("de,bne->bnd", query_inverse, observed)

    assert torch.allclose(observer_view, expected, atol=1e-6)
    assert torch.allclose(observer_view[:, 1], canonical[:, 1], atol=1e-6)
    assert not torch.allclose(observer_view[:, 0], canonical[:, 0])


def test_layer_accepts_autoregressive_observer_coord():
    layer = ARTILayer(input_dim=4, coord_dim=2, hidden_dim=4, recall_steps=0, coord_frame_mode="operator_bank")
    x = torch.randn(2, 3, 4)
    coord = torch.zeros(2, 3, 2)
    coord[:, :, 0] = 1.0
    observer_coord = torch.zeros(2, 2)
    observer_coord[:, 1] = 1.0
    inverse = torch.eye(4).repeat(2, 1, 1)

    out = layer(x, coord=coord, observer_coord=observer_coord, frame_operators=inverse)

    assert out.y.shape == (2, 3, 4)
    assert torch.all(out.diagnostics["observer_frame_active"] == 1.0)


def test_virtual_recall_alignment_loss_has_warmup_and_alignment_targets():
    torch.manual_seed(11)
    layer = ARTILayer(input_dim=8, hidden_dim=8, recall_steps=1, use_pairwise_context=False)
    clean = torch.randn(3, 5, 8)
    corrupt = clean * (torch.rand_like(clean) > 0.35).float()
    mask = torch.ones(3, 5, dtype=torch.bool)

    warmup_loss, _, warmup_out = virtual_recall_alignment_loss(layer, clean, corrupt, mask=mask, epoch=1, align_start_epoch=2)
    align_loss, clean_out, align_out = virtual_recall_alignment_loss(layer, clean, corrupt, mask=mask, epoch=2, align_start_epoch=2)

    assert warmup_out.virtual_y is not None
    assert align_out.virtual_y is not None
    assert warmup_loss.requires_grad
    assert align_loss.requires_grad
    assert clean_out.y.shape == align_out.virtual_y.shape

    align_loss.backward()
    assert layer.virtual_recall_proj.weight.grad is not None


def test_experiential_recall_alignment_reduces_corruption_trace_error():
    torch.manual_seed(23)
    layer = ARTILayer(input_dim=6, hidden_dim=6, recall_steps=0, use_pairwise_context=False, use_layer_norm=False)
    clean = torch.randn(4, 5, 6)
    corrupt = clean.clone()
    corrupt[:, 2:, :] = 0.0
    mask = torch.ones(4, 5, dtype=torch.bool)

    for name, param in layer.named_parameters():
        param.requires_grad = name.startswith("virtual_recall_proj")

    with torch.no_grad():
        _, clean_out, before_out = experiential_recall_alignment_loss(
            layer,
            clean,
            corrupt,
            mask=mask,
            epoch=2,
            align_start_epoch=2,
        )
        before = torch.nn.functional.mse_loss(before_out.recall_prediction, clean_out.y).item()

    optimizer = torch.optim.Adam(layer.virtual_recall_proj.parameters(), lr=0.05)
    for _ in range(40):
        optimizer.zero_grad()
        loss, _, _ = experiential_recall_alignment_loss(
            layer,
            clean,
            corrupt,
            mask=mask,
            epoch=2,
            align_start_epoch=2,
        )
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        _, clean_out, after_out = experiential_recall_alignment_loss(
            layer,
            clean,
            corrupt,
            mask=mask,
            epoch=2,
            align_start_epoch=2,
        )
        after = torch.nn.functional.mse_loss(after_out.recall_prediction, clean_out.y).item()

    assert after < before * 0.75


def test_experiential_recall_selectivity_loss_trains_alignment_recognizer():
    torch.manual_seed(41)
    layer = ARTILayer(
        input_dim=6,
        hidden_dim=6,
        recall_steps=1,
        recall_activation="none",
        recall_recognition_mode="alignment",
        use_pairwise_context=False,
    )
    clean = torch.randn(3, 4, 6)
    corrupt = clean * (torch.rand_like(clean) > 0.4)
    unseen = torch.randn(3, 4, 6) + 5.0
    mask = torch.ones(3, 4, dtype=torch.bool)
    loss, _, corrupt_out, unseen_out = experiential_recall_selectivity_loss(
        layer,
        clean,
        corrupt,
        unseen,
        mask=mask,
        unseen_mask=mask,
        epoch=2,
    )
    assert corrupt_out.recall_influence is not None
    assert unseen_out.recall_influence is not None
    loss.backward()
    assert layer.state.recall.alignment_recognizer.weight.grad is not None
