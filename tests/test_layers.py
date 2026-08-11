import copy

import pytest
import torch

from arti import ARTIConfig, ARTILayer, ARTIOutput
from arti.config import STATE_RECALL_COMPOSITION_FACTOR
from arti.functional import apply_coord_frame_inverse, masked_mean
from arti.init import init_arti_module
from arti.layers import ARTILatentRecallField, ARTILatentTensorLayer
from arti.training import (
    experiential_recall_alignment_loss,
    experiential_recall_selectivity_loss,
    recall_route_exterior_penalty,
    virtual_recall_alignment_loss,
)


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
    assert "experiential_recall_self_input_norm" in out.diagnostics
    assert "recall_recognition" in out.diagnostics


def test_virtual_recall_trace_is_consumed_as_private_recall_input():
    torch.manual_seed(7)
    layer = ARTILayer(
        input_dim=6,
        hidden_dim=6,
        recall_steps=2,
        use_pairwise_context=False,
    )
    x = torch.randn(2, 5, 6)
    mask = torch.tensor([[True, True, False, False, False], [True, True, True, True, False]])
    captured = []
    assert layer.state.recall is not None
    handle = layer.state.recall.register_forward_pre_hook(
        lambda _module, args: captured.append(args[2].detach().clone())
    )
    try:
        output = layer(x, mask=mask)
    finally:
        handle.remove()

    assert output.virtual_y is not None
    expected = masked_mean(output.virtual_y, mask, dim=1).unsqueeze(1)
    assert len(captured) == 2
    assert all(value.shape == (2, 1, 6) for value in captured)
    assert all(torch.allclose(value, expected) for value in captured)


def test_virtual_recall_self_input_precedes_optional_external_slots():
    torch.manual_seed(8)
    layer = ARTILayer(
        input_dim=4,
        hidden_dim=4,
        recall_steps=1,
        use_pairwise_context=False,
    )
    x = torch.randn(2, 3, 4)
    external = torch.randn(2, 2, 4)
    captured = []
    assert layer.state.recall is not None
    handle = layer.state.recall.register_forward_pre_hook(
        lambda _module, args: captured.append(args[2].detach().clone())
    )
    try:
        output = layer(x, recall=external)
    finally:
        handle.remove()

    assert output.virtual_y is not None
    assert len(captured) == 1
    assert captured[0].shape == (2, 3, 4)
    assert torch.allclose(
        captured[0][:, :1],
        output.virtual_y.mean(dim=1, keepdim=True),
    )
    assert torch.equal(captured[0][:, 1:], external)


def test_recall_recognition_blocks_unseen_trace_before_half():
    field = ARTILatentRecallField(
        2, 2, recognition_mode="explicit", recognition_threshold=0.5, recognition_temperature=0.05
    )
    with torch.no_grad():
        field.bank.copy_(torch.eye(2))
        field.query.weight.copy_(torch.eye(2))
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
    for parameter in field.parameters():
        parameter.requires_grad_(
            parameter is field.alignment_recognizer.weight
            or parameter is field.alignment_recognizer.bias
        )
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


def test_recall_defaults_open_without_a_strength_controller():
    field = ARTILatentRecallField(4, 3)

    assert field.recognition_mode == "none"
    assert not hasattr(field, "gate")
    assert not any(name.startswith("gate.") for name, _ in field.named_parameters())


def test_fixed_recall_query_is_deterministic_frozen_and_reinitialization_safe():
    torch.manual_seed(1)
    first = ARTILatentRecallField(
        8,
        32,
        routing="grouped",
        key_dim=4,
        group_size=8,
        group_topk=2,
        query_seed=17,
    )
    torch.manual_seed(999)
    second = ARTILatentRecallField(
        8,
        32,
        routing="grouped",
        key_dim=4,
        group_size=8,
        group_topk=2,
        query_seed=17,
    )

    assert torch.equal(first.query.weight, second.query.weight)
    assert not first.query.weight.requires_grad
    before = first.query.weight.detach().clone()
    init_arti_module(first)
    assert torch.equal(first.query.weight, before)
    with torch.no_grad():
        first.query.weight.zero_()
    first.reset_query()
    assert torch.equal(first.query.weight, before)
    assert first.query_contract == {
        "mode": "fixed",
        "algorithm": "rademacher-shake256-v1",
        "seed": 17,
        "hidden_dim": 8,
        "key_dim": 4,
    }


def test_fixed_recall_query_uses_identity_without_compression():
    field = ARTILatentRecallField(
        4,
        8,
        routing="grouped",
        key_dim=4,
        group_size=2,
        group_topk=1,
    )

    assert torch.equal(field.query.weight, torch.eye(4))


def test_legacy_learned_recall_query_remains_an_explicit_compatibility_mode():
    field = ARTILatentRecallField(4, 3, query_mode="legacy_learned")

    assert field.query.weight.requires_grad
    assert field.query_contract["mode"] == "legacy_learned"


def test_open_recall_projects_query_only_once(monkeypatch) -> None:
    field = ARTILatentRecallField(
        8,
        32,
        routing="grouped",
        key_dim=4,
        group_size=8,
        group_topk=2,
        recognition_mode="none",
    )
    calls = 0
    project_query = field._project_query

    def counted_project_query(z):
        nonlocal calls
        calls += 1
        return project_query(z)

    monkeypatch.setattr(field, "_project_query", counted_project_query)
    field(
        torch.randn(2, 5, 8),
        torch.ones(2, 5, dtype=torch.bool),
    )

    assert calls == 1


def test_open_recall_parameters_receive_first_step_gradients():
    torch.manual_seed(3)
    field = ARTILatentRecallField(4, 3)
    hidden = torch.randn(2, 5, 4)
    external = torch.randn(2, 2, 4)
    context, _, influence, recognition = field(
        hidden,
        torch.ones(2, 5, dtype=torch.bool),
        external,
    )

    context.square().mean().backward()

    assert torch.all(recognition == 1)
    assert torch.all(influence == 1)
    assert field.query.weight.grad is None
    assert not field.query.weight.requires_grad
    assert all(
        parameter.grad is not None
        for parameter in field.parameters()
        if parameter.requires_grad
    )
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in field.parameters()
        if parameter.grad is not None
    )


def test_grouped_recall_reads_sparse_independent_bank_and_backpropagates():
    torch.manual_seed(31)
    field = ARTILatentRecallField(
        8,
        32,
        routing="grouped",
        key_dim=4,
        group_size=8,
        group_topk=2,
    )
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    read = field(hidden, torch.ones(2, 5, dtype=torch.bool))
    context, weights, _, _ = read

    assert context.shape == hidden.shape
    assert weights.shape == (2, 5, 2, 8)
    assert read.indices.shape == (2, 5, 2, 8)
    assert read.route.shape == (2, 5, 4)
    assert torch.allclose(read.route.sum(dim=-1), torch.ones(2, 5))
    assert field.bank.numel() == 32 * 8
    assert field.key_bank is not None and field.key_bank.numel() == 32 * 4
    assert field.group_bank is not None and field.group_bank.numel() == 4 * 4

    context.square().mean().backward()
    assert field.bank.grad is not None
    assert field.key_bank.grad is not None
    assert field.group_bank.grad is not None
    assert hidden.grad is not None


def test_grouped_recall_weighted_value_read_matches_materialized_gather():
    torch.manual_seed(32)
    indices = torch.randint(0, 17, (2, 5, 2, 4))
    weights = torch.softmax(torch.randn(2, 5, 2, 4), dim=-1)
    efficient_bank = torch.randn(17, 8, requires_grad=True)
    reference_bank = efficient_bank.detach().clone().requires_grad_(True)
    efficient_weights = weights.detach().clone().requires_grad_(True)
    reference_weights = weights.detach().clone().requires_grad_(True)

    efficient = ARTILatentRecallField._weighted_value_read(
        indices,
        efficient_weights,
        efficient_bank,
        output_dtype=torch.float32,
    )
    reference = torch.einsum(
        "bntm,bntmh->bnh",
        reference_weights,
        reference_bank[indices],
    )
    assert torch.allclose(efficient, reference, rtol=1e-6, atol=1e-6)

    efficient.square().mean().backward()
    reference.square().mean().backward()
    assert torch.allclose(efficient_bank.grad, reference_bank.grad, rtol=1e-6, atol=1e-6)
    assert torch.allclose(
        efficient_weights.grad,
        reference_weights.grad,
        rtol=1e-6,
        atol=1e-6,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_grouped_recall_sparse_backward_matches_dense_cuda(dtype: torch.dtype) -> None:
    if not hasattr(torch.nn.functional, "grouped_mm"):
        pytest.skip("PyTorch native grouped_mm is unavailable")
    torch.manual_seed(35)
    selected_groups = torch.tensor(
        [[[[1], [1]], [[3], [0]], [[1], [3]]]],
        device="cuda",
    )
    offsets = torch.arange(4, device="cuda")
    indices = selected_groups * 4 + offsets
    sparse_bank = torch.randn(16, 15, device="cuda", dtype=dtype, requires_grad=True)
    dense_bank = sparse_bank.detach().clone().requires_grad_(True)
    sparse_weights = torch.randn(
        1,
        3,
        2,
        4,
        device="cuda",
        dtype=dtype,
        requires_grad=True,
    )
    dense_weights = sparse_weights.detach().clone().requires_grad_(True)

    sparse = ARTILatentRecallField._weighted_value_read(
        indices,
        sparse_weights,
        sparse_bank,
        output_dtype=dtype,
        sparse=True,
    )
    dense = torch.einsum(
        "bntm,bntmh->bnh",
        dense_weights,
        dense_bank[indices],
    )
    tolerance = 2e-2 if dtype == torch.bfloat16 else 1e-5
    torch.testing.assert_close(sparse, dense, rtol=tolerance, atol=tolerance)

    gradient = torch.randn_like(sparse)
    sparse.backward(gradient)
    dense.backward(gradient)
    torch.testing.assert_close(
        sparse_bank.grad.to_dense(),
        dense_bank.grad,
        rtol=tolerance,
        atol=tolerance,
    )
    torch.testing.assert_close(
        sparse_weights.grad,
        dense_weights.grad,
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_grouped_recall_native_grouped_mm_matches_sparse_value_read() -> None:
    if not hasattr(torch.nn.functional, "grouped_mm"):
        pytest.skip("PyTorch native grouped_mm is unavailable")
    torch.manual_seed(37)
    selected_groups = torch.randint(0, 8, (2, 17, 2), device="cuda")
    offsets = torch.arange(16, device="cuda")
    indices = selected_groups.unsqueeze(-1) * 16 + offsets
    native_logits = torch.randn(
        2,
        17,
        2,
        16,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    reference_logits = native_logits.detach().clone().requires_grad_(True)
    native_weights = torch.softmax(native_logits, dim=-1)
    reference_weights = torch.softmax(reference_logits, dim=-1)
    native_bank = torch.randn(128, 64, device="cuda", requires_grad=True)
    reference_bank = native_bank.detach().clone().requires_grad_(True)

    native = ARTILatentRecallField._grouped_mm_value_read(
        selected_groups,
        native_weights,
        native_bank,
        output_dtype=torch.bfloat16,
    )
    reference = ARTILatentRecallField._weighted_value_read(
        indices,
        reference_weights,
        reference_bank,
        output_dtype=torch.bfloat16,
    )
    assert (
        torch.nn.functional.cosine_similarity(
            native.float().flatten(),
            reference.float().flatten(),
            dim=0,
        )
        >= 0.999
    )

    native.float().square().mean().backward()
    reference.float().square().mean().backward()
    assert (
        torch.nn.functional.cosine_similarity(
            native_bank.grad.flatten(),
            reference_bank.grad.flatten(),
            dim=0,
        )
        >= 0.999
    )
    assert (
        torch.nn.functional.cosine_similarity(
            native_logits.grad.float().flatten(),
            reference_logits.grad.float().flatten(),
            dim=0,
        )
        >= 0.999
    )


def test_grouped_recall_route_exploration_is_reproducible_and_expands_coverage():
    torch.manual_seed(41)
    field = ARTILatentRecallField(
        8,
        128,
        routing="grouped",
        key_dim=8,
        group_size=4,
        group_topk=2,
        route_exploration=1.0,
    )
    with torch.no_grad():
        field.group_bank.zero_()
    hidden = torch.randn(2, 12, 8)
    mask = torch.ones(2, 12, dtype=torch.bool)

    torch.manual_seed(101)
    first = field(hidden, mask)
    torch.manual_seed(101)
    repeated = field(hidden, mask)
    assert torch.equal(first.indices, repeated.indices)
    assert torch.equal(first.context, repeated.context)
    first_groups = first.indices[..., 0] // field.group_size
    assert torch.equal(first_groups, first_groups[:, :1].expand_as(first_groups))
    first.context.square().mean().backward()
    assert field.group_bank is not None and field.group_bank.grad is not None
    assert torch.isfinite(field.group_bank.grad).all()
    assert field.key_bank is not None and field.key_bank.grad is not None
    assert torch.isfinite(field.key_bank.grad).all()
    assert field.bank.grad is not None and torch.isfinite(field.bank.grad).all()

    explored = set()
    for seed in range(8):
        torch.manual_seed(seed)
        indices = field(hidden, mask).indices
        explored.update((indices[..., 0] // field.group_size).flatten().tolist())
    assert len(explored) > 16


def test_disabled_route_exploration_preserves_training_rng_and_legacy_selection():
    torch.manual_seed(42)
    field = ARTILatentRecallField(
        8,
        32,
        routing="grouped",
        key_dim=8,
        group_size=4,
        group_topk=2,
    )
    hidden = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5, dtype=torch.bool)
    query = field.query(hidden)
    assert field.group_bank is not None
    expected_groups = torch.topk(
        torch.einsum("bnd,gd->bng", query, field.group_bank) * field.scale,
        field.group_topk,
        dim=-1,
    ).indices

    state_before = torch.random.get_rng_state()
    read = field(hidden, mask)
    state_after = torch.random.get_rng_state()

    assert torch.equal(state_before, state_after)
    assert torch.equal(read.indices[..., 0] // field.group_size, expected_groups)


def test_grouped_recall_route_exploration_keeps_eval_deterministic_and_sparse():
    torch.manual_seed(43)
    field = ARTILatentRecallField(
        8,
        64,
        routing="grouped",
        key_dim=8,
        group_size=4,
        group_topk=3,
        route_exploration=1.0,
    ).eval()
    hidden = torch.randn(2, 5, 8)
    mask = torch.ones(2, 5, dtype=torch.bool)

    state_before = torch.random.get_rng_state()
    first = field(hidden, mask)
    state_after = torch.random.get_rng_state()
    second = field(hidden, mask)

    assert torch.equal(state_before, state_after)
    assert torch.equal(first.indices, second.indices)
    assert torch.equal(first.context, second.context)
    assert first.indices.shape[-2:] == (3, 4)


def test_grouped_recall_training_partitions_limit_routing_without_disabling_layer():
    field = ARTILatentRecallField(
        8,
        240,
        routing="grouped",
        key_dim=4,
        group_size=4,
        group_topk=2,
    )
    field.train()
    field.set_training_group_partitions(30, (2, 7, 13, 29))

    result = field(
        torch.randn(2, 3, 8),
        torch.ones(2, 3, dtype=torch.bool),
    )

    selected_groups = result.indices[..., 0, 0] // field.group_size
    assert set(selected_groups.remainder(30).unique().tolist()) <= {2, 7, 13, 29}
    assert field._bank_gradient_enabled
    result.context.square().mean().backward()
    assert field.bank.grad is not None and field.bank.grad.is_sparse
    assert field.key_bank is not None
    assert field.key_bank.grad is not None and field.key_bank.grad.is_sparse
    assert field.group_bank is not None
    assert field.group_bank.grad is not None and field.group_bank.grad.is_sparse
    assert field.query.weight.grad is None

    field.eval()
    unrestricted = field(
        torch.randn(2, 3, 8),
        torch.ones(2, 3, dtype=torch.bool),
    )
    assert torch.isfinite(unrestricted.context).all()


def test_grouped_recall_all_partitions_active_keeps_dense_optimizer_gradients():
    field = ARTILatentRecallField(
        8,
        16,
        routing="grouped",
        key_dim=4,
        group_size=2,
        group_topk=1,
    )
    field.train()
    field.set_training_group_partitions(1, (0,))

    result = field(
        torch.randn(2, 3, 8),
        torch.ones(2, 3, dtype=torch.bool),
    )
    result.context.square().mean().backward()

    assert field.bank.grad is not None and not field.bank.grad.is_sparse
    assert field.key_bank is not None
    assert field.key_bank.grad is not None and not field.key_bank.grad.is_sparse
    assert field.group_bank is not None
    assert field.group_bank.grad is not None and not field.group_bank.grad.is_sparse


def test_grouped_recall_hard_top1_preserves_soft_route_gradients():
    field = ARTILatentRecallField(
        4,
        3,
        routing="grouped",
        key_dim=2,
        group_size=1,
        group_topk=1,
    )
    hidden = torch.randn(2, 3, 4)
    mask = torch.ones(2, 3, dtype=torch.bool)

    result = field(hidden, mask)
    result.context.square().mean().backward()

    torch.testing.assert_close(result.weights, torch.ones_like(result.weights))
    assert field.group_bank is not None
    assert field.group_bank.grad is not None
    assert torch.count_nonzero(field.group_bank.grad)
    assert field.query.weight.grad is None
    assert field.key_bank is not None and field.key_bank.grad is None


def test_product_recall_training_partitions_apply_to_both_value_banks():
    field = ARTILatentRecallField(
        8,
        240,
        routing="grouped",
        key_dim=4,
        group_size=4,
        group_topk=2,
        value_composition="product",
    )
    field.train()
    field.set_training_group_partitions(30, (1, 5, 9, 17))

    result = field(
        torch.randn(2, 3, 8),
        torch.ones(2, 3, dtype=torch.bool),
    )

    factor_groups = (field.slots // field.group_size) // 2
    selected_groups = result.indices[..., 0] // field.group_size
    local_groups = selected_groups.remainder(factor_groups)
    assert set(local_groups.remainder(30).unique().tolist()) <= {1, 5, 9, 17}
    assert torch.isfinite(result.context).all()
    result.context.square().mean().backward()
    assert field.bank.grad is not None and field.bank.grad.is_sparse
    assert field.key_bank is not None
    assert field.key_bank.grad is not None and field.key_bank.grad.is_sparse
    assert field.group_bank is not None
    assert field.group_bank.grad is not None and field.group_bank.grad.is_sparse


def test_grouped_recall_supports_fp32_master_banks_with_bfloat16_compute():
    torch.manual_seed(44)
    field = ARTILatentRecallField(
        8,
        32,
        routing="grouped",
        key_dim=8,
        group_size=4,
        group_topk=2,
        route_exploration=1.0,
    ).to(dtype=torch.bfloat16)
    for parameter in (field.bank, field.key_bank, field.group_bank):
        assert parameter is not None
        parameter.data = parameter.data.float()
    field.query.weight.data = field.query.weight.data.float()
    hidden = torch.randn(2, 5, 8, dtype=torch.bfloat16)
    mask = torch.ones(2, 5, dtype=torch.bool)

    read = field(hidden, mask)
    read.context.float().square().mean().backward()

    assert read.context.dtype == torch.bfloat16
    for parameter in (field.bank, field.key_bank, field.group_bank):
        assert parameter is not None and parameter.dtype == torch.float32
        assert parameter.grad is not None and parameter.grad.dtype == torch.float32
        assert torch.isfinite(parameter.grad).all()
    assert field.query.weight.dtype == torch.float32
    assert field.query.weight.grad is None


def test_grouped_recall_can_detach_one_bank_shard_without_changing_forward():
    torch.manual_seed(45)
    field = ARTILatentRecallField(
        8,
        32,
        routing="grouped",
        key_dim=8,
        group_size=4,
        group_topk=2,
        route_exploration=1.0,
    )
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    mask = torch.ones(2, 5, dtype=torch.bool)
    field.set_bank_gradient_enabled(False)

    state_before = torch.random.get_rng_state()
    detached = field(hidden, mask)
    state_after = torch.random.get_rng_state()
    detached.context.square().mean().backward()

    assert torch.equal(state_before, state_after)
    assert field.bank.grad is None
    assert field.key_bank is not None and field.key_bank.grad is None
    assert field.group_bank is not None and field.group_bank.grad is None
    assert field.query.weight.grad is None
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()

    field.zero_grad(set_to_none=True)
    field.set_bank_gradient_enabled(True)
    torch.manual_seed(46)
    enabled = field(hidden.detach(), mask)
    enabled.context.square().mean().backward()
    assert field.bank.grad is not None
    assert field.key_bank.grad is not None
    assert field.group_bank.grad is not None
    assert field.query.weight.grad is None


def test_grouped_recall_reuses_owner_groups_across_refinement_steps():
    torch.manual_seed(47)
    layer = ARTILayer(
        input_dim=8,
        hidden_dim=8,
        recall_slots=32,
        recall_steps=3,
        recall_routing="grouped",
        recall_key_dim=8,
        recall_group_size=4,
        recall_group_topk=2,
        recall_route_exploration=1.0,
        use_pairwise_context=False,
    )
    assert layer.state.recall is not None
    selected_groups = []

    def capture_groups(_module, _args, output):
        selected_groups.append((output.indices[..., 0] // layer.state.recall.group_size).detach())

    handle = layer.state.recall.register_forward_hook(capture_groups)
    try:
        layer(torch.randn(2, 5, 8))
    finally:
        handle.remove()

    assert len(selected_groups) == 3
    assert all(torch.equal(selected_groups[0], groups) for groups in selected_groups[1:])


def test_product_recall_composes_two_independent_bank_reads() -> None:
    field = ARTILatentRecallField(
        4,
        8,
        routing="grouped",
        key_dim=4,
        group_size=2,
        group_topk=1,
        value_composition="product",
    )
    with torch.no_grad():
        field.bank[:4].fill_(2.0)
        field.bank[4:].fill_(3.0)

    hidden = torch.full((2, 3, 4), 4.0)
    read = field(hidden, torch.ones(2, 3, dtype=torch.bool))

    expected = (1.0 + torch.tanh(torch.tensor(2.0))) * 7.0 - 4.0
    torch.testing.assert_close(read.context, torch.full_like(read.context, expected))
    left_groups = read.indices[..., 0, 0] // field.group_size
    right_groups = read.indices[..., 1, 0] // field.group_size
    assert torch.all(left_groups < 2)
    assert torch.all(right_groups >= 2)


def test_packed_product_recall_matches_two_independent_reads_and_gradients() -> None:
    torch.manual_seed(43)
    packed = ARTILatentRecallField(
        4,
        16,
        routing="grouped",
        key_dim=4,
        group_size=2,
        group_topk=2,
        value_composition="product",
    )
    reference = copy.deepcopy(packed)
    packed_hidden = torch.randn(2, 3, 4, requires_grad=True)
    reference_hidden = packed_hidden.detach().clone().requires_grad_(True)
    mask = torch.ones(2, 3, dtype=torch.bool)

    actual = packed(packed_hidden, mask)
    factor_groups = reference.slots // reference.group_size // 2
    left = reference._grouped_read(
        reference_hidden,
        return_route=True,
        group_offset=0,
        group_count=factor_groups,
    )
    right = reference._grouped_read(
        reference_hidden,
        return_route=True,
        group_offset=factor_groups,
        group_count=factor_groups,
    )
    expected_context = reference._compose_product_write(
        reference_hidden,
        left[0],
        right[0],
    )
    expected_weights = torch.cat((left[1], right[1]), dim=-2)
    expected_indices = torch.cat((left[2], right[2]), dim=-2)
    expected_route = torch.cat((left[3], right[3]), dim=-1)

    torch.testing.assert_close(actual.context, expected_context)
    torch.testing.assert_close(actual.weights, expected_weights)
    torch.testing.assert_close(actual.indices, expected_indices)
    torch.testing.assert_close(actual.route, expected_route)

    actual.context.square().mean().backward()
    expected_context.square().mean().backward()
    torch.testing.assert_close(packed_hidden.grad, reference_hidden.grad)
    for (packed_name, packed_parameter), (
        reference_name,
        reference_parameter,
    ) in zip(
        packed.named_parameters(),
        reference.named_parameters(),
        strict=True,
    ):
        assert packed_name == reference_name
        assert (packed_parameter.grad is None) == (reference_parameter.grad is None)
        if packed_parameter.grad is not None:
            torch.testing.assert_close(
                packed_parameter.grad,
                reference_parameter.grad,
            )


def test_product_recall_zero_init_is_identity_and_trains_both_banks() -> None:
    field = ARTILatentRecallField(
        4,
        8,
        routing="grouped",
        key_dim=4,
        group_size=2,
        group_topk=1,
        value_composition="product",
    )
    torch.nn.init.zeros_(field.bank)
    hidden = torch.randn(2, 3, 4)
    context = field.read_context(hidden)

    torch.testing.assert_close(context, torch.zeros_like(context))
    context.sum().backward()
    assert field.bank.grad is not None
    assert torch.count_nonzero(field.bank.grad[:4]).item() > 0
    assert torch.count_nonzero(field.bank.grad[4:]).item() > 0


def test_product_recall_bounds_the_multiplicative_gain() -> None:
    z = torch.tensor([[[-3.0, -1.0, 1.0, 3.0]]])
    shift = torch.tensor([[[0.25, 0.25, 0.25, 0.25]]])
    positive = ARTILatentRecallField._compose_product_write(
        z,
        torch.full_like(z, 100.0),
        shift,
    )
    negative = ARTILatentRecallField._compose_product_write(
        z,
        torch.full_like(z, -100.0),
        shift,
    )

    torch.testing.assert_close(z + positive, 2.0 * (z + shift))
    torch.testing.assert_close(z + negative, torch.zeros_like(z))


def test_product_recall_does_not_increase_parameter_count() -> None:
    single = ARTILatentRecallField(
        4,
        8,
        routing="grouped",
        key_dim=4,
        group_size=2,
        group_topk=1,
    )
    product = ARTILatentRecallField(
        4,
        8,
        routing="grouped",
        key_dim=4,
        group_size=2,
        group_topk=1,
        value_composition="product",
    )

    assert sum(parameter.numel() for parameter in product.parameters()) == sum(
        parameter.numel() for parameter in single.parameters()
    )


def test_state_recall_uses_recalled_factors_to_control_host_transparency() -> None:
    field = ARTILatentRecallField(
        2,
        17,
        routing="dense",
        value_composition="state",
        factor_activation="none",
    )
    with torch.no_grad():
        field.bank.zero_()
        field.bank[0].copy_(torch.tensor([2.0, -3.0]))
        field.bank[1].copy_(torch.tensor([0.1, -0.2]))
        field.bank[2:15].copy_(torch.tensor([0.2, -0.1]).expand(13, -1))
        field.bank[15].copy_(torch.tensor([0.4, -0.2]))
        field.bank[16].copy_(torch.tensor([0.3, -0.5]))

    first = field(
        torch.tensor([[[100.0, -100.0]]]),
        torch.ones(1, 1, dtype=torch.bool),
    )
    second = field(
        torch.tensor([[[-7.0, 12.0]]]),
        torch.ones(1, 1, dtype=torch.bool),
    )

    content = field.bank[0] + field.bank[0].abs().clamp_min(1.0) * torch.tanh(field.bank[1])
    modulation = field.bank[2:15]
    write_direction = torch.tanh(field.bank[15])
    memory_opacity = torch.tanh(field.bank[16]).square()
    recalled = (
        memory_opacity
        * write_direction
        * content
        * (1.0 + torch.tanh(modulation) / 13.0).prod(dim=0)
    )
    torch.testing.assert_close(
        first.context,
        (1.0 - memory_opacity) * torch.tensor([[[100.0, -100.0]]]) + recalled,
    )
    torch.testing.assert_close(
        second.context,
        (1.0 - memory_opacity) * torch.tensor([[[-7.0, 12.0]]]) + recalled,
    )
    assert not torch.equal(first.context, second.context)


def test_state_recall_constrains_host_and_memory_coefficients() -> None:
    factors = torch.randn(3, 5, STATE_RECALL_COMPOSITION_FACTOR, 7)
    memory_opacity = torch.tanh(factors[..., -1, :]).square()
    host_weight = 1.0 - memory_opacity
    memory_weight = memory_opacity * torch.tanh(factors[..., -2, :])

    assert torch.all(host_weight >= 0)
    assert torch.all(host_weight + memory_weight.abs() <= 1.0 + 1e-6)


def test_state_recall_fraction_scales_gradient_without_coupling_coarse_limb() -> None:
    coarse = torch.tensor([[-4.0, 0.25, 3.0]], requires_grad=True)
    fine = torch.zeros_like(coarse, requires_grad=True)

    content = ARTILatentRecallField._compose_state_content(coarse, fine)
    content.sum().backward()

    torch.testing.assert_close(content, coarse)
    torch.testing.assert_close(coarse.grad, torch.ones_like(coarse))
    torch.testing.assert_close(fine.grad, torch.tensor([[4.0, 1.0, 3.0]]))


def test_grouped_state_recall_keeps_parameter_count_and_trains_all_factors() -> None:
    single = ARTILatentRecallField(
        4,
        34,
        routing="grouped",
        key_dim=4,
        group_size=2,
        group_topk=1,
    )
    state = ARTILatentRecallField(
        4,
        34,
        routing="grouped",
        key_dim=4,
        group_size=2,
        group_topk=1,
        value_composition="state",
        factor_activation="none",
    )
    assert sum(parameter.numel() for parameter in state.parameters()) == sum(
        parameter.numel() for parameter in single.parameters()
    )
    with torch.no_grad():
        (
            coarse_content,
            fine_content,
            *modulations,
            write_direction,
            memory_opacity,
        ) = state.bank.chunk(
            STATE_RECALL_COMPOSITION_FACTOR,
            dim=0,
        )
        coarse_content.fill_(0.5)
        fine_content.fill_(0.1)
        for modulation in modulations:
            modulation.fill_(0.2)
        write_direction.fill_(0.3)
        memory_opacity.fill_(-0.4)

    hidden = torch.randn(2, 3, 4, requires_grad=True)
    read = state(hidden, torch.ones(2, 3, dtype=torch.bool))
    read.context.square().mean().backward()

    assert torch.isfinite(read.context).all()
    assert state.bank.grad is not None
    for factor in state.bank.grad.chunk(STATE_RECALL_COMPOSITION_FACTOR, dim=0):
        assert torch.count_nonzero(factor).item() > 0


def test_post_bank_recall_prediction_supervises_bank_and_query():
    torch.manual_seed(48)
    layer = ARTILayer(
        input_dim=8,
        hidden_dim=8,
        recall_slots=32,
        recall_steps=2,
        recall_routing="grouped",
        recall_key_dim=8,
        recall_group_size=4,
        recall_group_topk=2,
        use_pairwise_context=False,
    )
    output = layer(torch.randn(2, 5, 8))
    assert output.recall_prediction is not None
    target = torch.randn_like(output.recall_prediction)

    torch.nn.functional.mse_loss(output.recall_prediction, target).backward()

    recall = layer.state.recall
    assert recall is not None
    for parameter in (recall.bank, recall.key_bank, recall.group_bank):
        assert parameter is not None
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert recall.query.weight.grad is None


def test_recall_prediction_accumulates_all_refinement_corrections():
    torch.manual_seed(49)
    single = ARTILayer(
        input_dim=8,
        hidden_dim=8,
        recall_slots=16,
        recall_steps=1,
        recall_activation="none",
        use_pairwise_context=False,
        dropout=0.0,
    )
    stacked = ARTILayer(
        input_dim=8,
        hidden_dim=8,
        recall_slots=16,
        recall_steps=3,
        recall_activation="none",
        use_pairwise_context=False,
        dropout=0.0,
    )
    stacked.load_state_dict(single.state_dict(), strict=False)
    hidden = torch.randn(2, 5, 8)

    one = single(hidden)
    three = stacked(hidden)

    assert one.recall_prediction is not None
    assert three.recall_prediction is not None
    assert "recall_effect_norm" in three.diagnostics
    assert not torch.allclose(one.recall_prediction, three.recall_prediction)


@pytest.mark.parametrize("value", [-0.1, float("inf"), float("nan")])
def test_recall_route_exploration_rejects_invalid_values(value: float):
    with pytest.raises(ValueError, match="route_exploration"):
        ARTILatentRecallField(
            4,
            8,
            routing="grouped",
            group_size=4,
            route_exploration=value,
        )
    with pytest.raises(ValueError, match="recall_route_exploration"):
        ARTIConfig(input_dim=4, recall_route_exploration=value)


def test_recall_route_exterior_penalty_has_dead_zone_and_detached_anchor():
    torch.manual_seed(37)
    layer = ARTILayer(
        input_dim=8,
        hidden_dim=8,
        recall_slots=16,
        recall_steps=1,
        recall_routing="grouped",
        recall_key_dim=4,
        recall_group_size=4,
        recall_group_topk=2,
        use_pairwise_context=False,
    )
    anchor = layer(torch.randn(2, 3, 8))
    similar = layer(torch.randn(2, 3, 8))
    assert anchor.recall_route is not None
    assert similar.recall_route is not None
    anchor.recall_route.retain_grad()
    similar.recall_route.retain_grad()

    zero = recall_route_exterior_penalty(anchor, similar, tolerance=1.0)
    assert zero.item() == 0.0
    penalty = recall_route_exterior_penalty(anchor, similar, tolerance=0.0)
    penalty.backward()
    assert anchor.recall_route.grad is None
    assert similar.recall_route.grad is not None


def test_arti_output_keeps_legacy_diagnostics_positional_slot():
    diagnostics = {"coverage": torch.ones(1)}
    output = ARTIOutput(
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        None,
        None,
        None,
        None,
        diagnostics,
    )

    assert output.diagnostics is diagnostics
    assert output.recall_context is None
    assert output.recall_route is None


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
    raw_layer = ARTILayer(
        input_dim=8,
        hidden_dim=8,
        recall_steps=1,
        recall_activation="none",
        use_pairwise_context=False,
    )
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


def test_recall_microcycle_stops_after_configured_minimum_steps() -> None:
    layer = ARTILayer(
        input_dim=8,
        hidden_dim=8,
        recall_steps=4,
        recall_min_steps=2,
        recall_tolerance=0.0,
        use_pairwise_context=False,
        use_virtual_recall=False,
    )
    assert layer.state.recall is not None
    with torch.no_grad():
        layer.state.recall.bank.zero_()

    output = layer(torch.randn(3, 5, 8))

    assert torch.equal(output.diagnostics["recall_steps_executed"], torch.full((3,), 2.0))
    assert torch.equal(
        output.diagnostics["recall_step_active"],
        torch.tensor([[1.0, 1.0, 0.0, 0.0]]).expand(3, -1),
    )
    assert torch.count_nonzero(output.diagnostics["recall_step_update_ratio"]) == 0


def test_adaptive_recall_microcycle_preserves_gradients() -> None:
    layer = ARTILayer(
        input_dim=6,
        hidden_dim=6,
        recall_steps=3,
        recall_min_steps=1,
        recall_tolerance=1e-6,
        use_pairwise_context=False,
    )
    output = layer(
        torch.randn(2, 4, 6, requires_grad=True),
        recall=torch.randn(2, 2, 6),
    )

    output.y.square().mean().backward()

    recall = layer.state.recall
    assert recall is not None
    assert recall.query.weight.grad is None
    assert all(
        parameter.grad is not None
        for parameter in recall.parameters()
        if parameter.requires_grad
    )
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in recall.parameters()
        if parameter.grad is not None
    )
    assert output.diagnostics["recall_step_active"].shape == (2, 3)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"recall_steps": 3, "recall_min_steps": 0},
        {"recall_steps": 3, "recall_min_steps": 4},
        {"recall_tolerance": -1.0},
        {"recall_tolerance": float("inf")},
    ],
)
def test_adaptive_recall_config_validation(kwargs) -> None:
    with pytest.raises(ValueError):
        ARTIConfig(input_dim=4, **kwargs)


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
    observed[..., 0::2] = (
        torch.cos(theta).unsqueeze(-1) * even - torch.sin(theta).unsqueeze(-1) * odd
    )
    observed[..., 1::2] = (
        torch.sin(theta).unsqueeze(-1) * even + torch.cos(theta).unsqueeze(-1) * odd
    )

    recovered = apply_coord_frame_inverse(observed, coord, "paired_rotation")

    assert torch.allclose(recovered, z, atol=1e-6)


def test_layer_reports_coord_frame_delta_when_enabled():
    layer = ARTILayer(
        input_dim=8, coord_dim=2, hidden_dim=8, recall_steps=0, coord_frame_mode="paired_rotation"
    )
    x = torch.randn(2, 4, 8)
    coord = torch.zeros(2, 4, 2)
    coord[..., 1] = 1.0

    out = layer(x, coord=coord)

    assert out.y.shape == (2, 4, 8)
    assert "coord_frame_delta" in out.diagnostics


def test_random_coord_fallback_is_stable_when_coord_is_omitted():
    torch.manual_seed(31)
    layer = ARTILayer(
        input_dim=8,
        coord_dim=3,
        hidden_dim=8,
        recall_steps=0,
        fallback_context="random_coord",
        fallback_slots=4,
    )
    x = torch.randn(2, 6, 8)

    coord_a = layer._resolve_coord(None, 2, 6, x.device, x.dtype)
    coord_b = layer._resolve_coord(None, 2, 6, x.device, x.dtype)
    out = layer(x)

    assert torch.allclose(coord_a, coord_b)
    assert coord_a.shape == (2, 6, 3)
    assert not torch.allclose(coord_a, torch.zeros_like(coord_a))
    assert out.y.shape == (2, 6, 8)


def test_random_context_fallback_supplies_visibility_when_omitted():
    layer = ARTILayer(
        input_dim=4,
        coord_dim=2,
        hidden_dim=4,
        recall_steps=0,
        fallback_context="random_context",
        fallback_slots=3,
    )
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

    observer_view = apply_coord_frame_inverse(
        observed, coord, "operator_bank", inverse, observer_coord=observer_coord
    )
    query_inverse = inverse[1]
    expected = torch.einsum("de,bne->bnd", query_inverse, observed)

    assert torch.allclose(observer_view, expected, atol=1e-6)
    assert torch.allclose(observer_view[:, 1], canonical[:, 1], atol=1e-6)
    assert not torch.allclose(observer_view[:, 0], canonical[:, 0])


def test_layer_accepts_autoregressive_observer_coord():
    layer = ARTILayer(
        input_dim=4, coord_dim=2, hidden_dim=4, recall_steps=0, coord_frame_mode="operator_bank"
    )
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

    warmup_loss, _, warmup_out = virtual_recall_alignment_loss(
        layer, clean, corrupt, mask=mask, epoch=1, align_start_epoch=2
    )
    align_loss, clean_out, align_out = virtual_recall_alignment_loss(
        layer, clean, corrupt, mask=mask, epoch=2, align_start_epoch=2
    )

    assert warmup_out.virtual_y is not None
    assert align_out.virtual_y is not None
    assert warmup_loss.requires_grad
    assert align_loss.requires_grad
    assert clean_out.y.shape == align_out.virtual_y.shape

    align_loss.backward()
    assert layer.virtual_recall_proj.weight.grad is not None


def test_experiential_recall_alignment_reduces_corruption_trace_error():
    torch.manual_seed(23)
    layer = ARTILayer(
        input_dim=6, hidden_dim=6, recall_steps=0, use_pairwise_context=False, use_layer_norm=False
    )
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
