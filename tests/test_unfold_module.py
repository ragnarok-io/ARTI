from __future__ import annotations

from pathlib import Path

import pytest
import torch

import arti
from arti.nn import UnFold


def _original_counts(source: torch.Tensor, length: int) -> torch.Tensor:
    return torch.stack([(source == index).sum(-1) for index in range(length)], dim=-1)


def test_unfold_expands_and_preserves_every_original_instance_exactly() -> None:
    torch.manual_seed(3)
    layer = UnFold(dim=2, exposed=3)
    x = torch.tensor([[[100.0, 101.0], [200.0, 201.0], [300.0, 301.0]]])
    y, exposed_mask, source = layer(x, return_exposed_mask=True, return_source_index=True)

    assert y.shape == (1, 6, 2)
    assert exposed_mask.sum().item() == 3
    assert torch.equal(_original_counts(source, x.shape[1]), torch.ones(1, 3, dtype=torch.long))
    for index in range(x.shape[1]):
        assert torch.equal(y[source == index], x[:, index])


def test_unfold_runtime_target_length_reuses_one_capacity_at_different_sizes() -> None:
    torch.manual_seed(4)
    layer = UnFold(dim=3, exposed=4)
    single = torch.randn(2, 2, 3)
    concatenated = torch.randn(2, 8, 3)

    single_y, single_exposed, single_source = layer(
        single,
        target_length=3,
        return_exposed_mask=True,
        return_source_index=True,
    )
    concat_y, concat_exposed, concat_source = layer(
        concatenated,
        target_length=12,
        return_exposed_mask=True,
        return_source_index=True,
    )

    assert single_y.shape == (2, 3, 3)
    assert concat_y.shape == (2, 12, 3)
    assert torch.equal(single_exposed.sum(1), torch.ones(2, dtype=torch.long))
    assert torch.equal(concat_exposed.sum(1), torch.full((2,), 4, dtype=torch.long))
    assert torch.equal(
        _original_counts(single_source, single.shape[1]),
        torch.ones(2, single.shape[1], dtype=torch.long),
    )
    assert torch.equal(
        _original_counts(concat_source, concatenated.shape[1]),
        torch.ones(2, concatenated.shape[1], dtype=torch.long),
    )


def test_unfold_runtime_target_length_activates_only_required_parameter_slots() -> None:
    torch.manual_seed(6)
    layer = UnFold(dim=3, exposed=4)
    x = torch.randn(2, 3, 3, requires_grad=True)
    layer(x, target_length=4).square().mean().backward()

    assert layer.exposed_queries.grad is not None
    assert layer.exposed_value_mix.grad is not None
    assert layer.exposed_queries.grad[0].abs().sum() > 0
    assert layer.exposed_value_mix.grad[0].abs().sum() > 0
    assert torch.count_nonzero(layer.exposed_queries.grad[1:]) == 0
    assert torch.count_nonzero(layer.exposed_value_mix.grad[1:]) == 0


def test_unfold_runtime_target_length_shapes_masks_and_exposed_guides() -> None:
    layer = UnFold(
        dim=2,
        exposed=4,
        guide_dim=1,
        layout_mode="canonical",
    )
    x = torch.randn(2, 3, 2)
    mask = torch.tensor([[True, True, False], [True, True, True]])
    guide = torch.randn(2, 3, 1)
    exposed_guide = torch.randn(2, 2, 1)
    exposed_mask = torch.tensor([[True, False], [True, True]])

    y, output_mask, output_exposed = layer(
        x,
        mask,
        target_length=5,
        guide=guide,
        exposed_guide=exposed_guide,
        exposed_mask=exposed_mask,
        return_exposed_mask=True,
    )

    assert y.shape == (2, 5, 2)
    assert output_mask.shape == (2, 5)
    assert output_exposed.shape == (2, 5)
    assert torch.equal(output_mask.sum(1), torch.tensor([3, 5]))
    assert torch.equal(output_exposed.sum(1), torch.tensor([2, 2]))


@pytest.mark.parametrize("target_length", [3, 8, 3.5, True])
def test_unfold_rejects_invalid_runtime_target_length(target_length: object) -> None:
    layer = UnFold(dim=2, exposed=3)
    x = torch.randn(1, 3, 2)
    with pytest.raises(ValueError, match="target_length"):
        layer(x, target_length=target_length)


def test_unfold_layout_is_a_valid_sample_conditioned_rank() -> None:
    torch.manual_seed(5)
    layer = UnFold(dim=4, exposed=2)
    x = torch.randn(2, 5, 4)
    x[1] = -2 * x[0]
    _, source, soft_layout = layer(x, return_source_index=True, return_soft_layout=True)
    assert torch.equal(source.sort(1).values, torch.arange(7).expand(2, -1))
    assert not torch.equal(soft_layout[0], soft_layout[1])


def test_unfold_exposed_values_are_queried_from_input() -> None:
    torch.manual_seed(7)
    layer = UnFold(dim=3, exposed=2).eval()
    x = torch.randn(1, 4, 3)
    y1, exposed1 = layer(x, return_exposed_mask=True)
    y2, exposed2 = layer(x + 2.0, return_exposed_mask=True)
    assert not torch.equal(y1[exposed1], y2[exposed2])


def test_unfold_value_parameter_growth_is_operator_bank_plus_slot_modulation() -> None:
    dim, exposed, operators = 7, 11, 3
    layer = UnFold(dim=dim, exposed=exposed, value_operators=operators)
    value_parameters = (
        layer.exposed_value_operators.numel()
        + layer.exposed_value_mix.numel()
        + layer.exposed_scale.numel()
        + layer.exposed_bias.numel()
    )
    assert value_parameters == operators * dim * dim + exposed * operators + 2 * exposed * dim


def test_unfold_requires_positive_value_operator_count() -> None:
    with pytest.raises(ValueError, match="value_operators"):
        UnFold(dim=3, value_operators=0)


def test_unfold_requires_positive_value_rank() -> None:
    with pytest.raises(ValueError, match="value_rank"):
        UnFold(dim=3, value_rank=0)


def test_unfold_factorized_operator_parameter_formula_and_gradients() -> None:
    dim, exposed, operators, rank = 9, 7, 3, 4
    layer = UnFold(
        dim=dim,
        exposed=exposed,
        value_operators=operators,
        value_rank=rank,
    )
    assert layer.exposed_value_operators is None
    assert layer.exposed_value_left is not None
    assert layer.exposed_value_right is not None
    operator_parameters = (
        layer.exposed_value_left.numel()
        + layer.exposed_value_right.numel()
        + layer.exposed_value_mix.numel()
    )
    assert operator_parameters == 2 * operators * dim * rank + exposed * operators
    x = torch.randn(2, 5, dim, requires_grad=True)
    layer(x).square().mean().backward()
    assert layer.exposed_value_left.grad is not None
    assert layer.exposed_value_right.grad is not None


@pytest.mark.parametrize("field", ["query_chunk_size", "operator_chunk_size"])
def test_unfold_requires_positive_chunk_sizes(field: str) -> None:
    with pytest.raises(ValueError, match=field):
        UnFold(dim=3, **{field: 0})


def test_unfold_chunked_execution_matches_default_and_has_gradients() -> None:
    torch.manual_seed(31)
    default = UnFold(dim=6, exposed=5, value_operators=4)
    chunked = UnFold(
        dim=6,
        exposed=5,
        value_operators=4,
        query_chunk_size=2,
        operator_chunk_size=2,
    )
    chunked.load_state_dict(default.state_dict())
    x_default = torch.randn(3, 7, 6, requires_grad=True)
    x_chunked = x_default.detach().clone().requires_grad_(True)
    expected = default(x_default)
    actual = chunked(x_chunked)
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-6)
    expected.square().mean().backward()
    actual.square().mean().backward()
    assert torch.allclose(x_chunked.grad, x_default.grad, atol=1e-6, rtol=1e-6)


def test_unfold_exact_operator_schedules_match_forward_and_full_gradients() -> None:
    torch.manual_seed(37)
    reference = UnFold(
        dim=5,
        exposed=4,
        value_operators=3,
        query_chunk_size=2,
    ).double()
    x = torch.randn(3, 6, 5, dtype=torch.float64)
    mask = torch.tensor(
        [
            [True, True, True, True, True, True],
            [True, False, True, False, True, False],
            [False, False, False, False, False, False],
        ]
    )
    exposed_weight = torch.randn(3, 4, 5, dtype=torch.float64)
    attended_weight = torch.randn(3, 4, 5, dtype=torch.float64)
    results: dict[str, tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor | None, ...]]] = {}

    for schedule in ("slot", "palette", "effective"):
        layer = UnFold(
            dim=5,
            exposed=4,
            value_operators=3,
            query_chunk_size=2,
        ).double()
        layer.load_state_dict(reference.state_dict())
        layer._operator_schedule = schedule
        schedule_x = x.clone().requires_grad_(True)
        exposed, attended = layer._query_exposed(schedule_x, mask)
        loss = (exposed * exposed_weight).sum() + (attended * attended_weight).sum()
        gradients = torch.autograd.grad(loss, (schedule_x, *layer.parameters()), allow_unused=True)
        results[schedule] = (exposed, attended, gradients)

    expected_exposed, expected_attended, expected_gradients = results["slot"]
    for schedule in ("palette", "effective"):
        actual_exposed, actual_attended, actual_gradients = results[schedule]
        assert torch.allclose(actual_exposed, expected_exposed, atol=1e-10, rtol=1e-10)
        assert torch.allclose(actual_attended, expected_attended, atol=1e-12, rtol=1e-12)
        for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
            if expected is None:
                assert actual is None
            else:
                assert actual is not None
                assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_unfold_auto_operator_schedule_uses_conservative_shape_costs() -> None:
    layer = UnFold(dim=16, exposed=64, value_operators=4)
    assert layer._select_operator_schedule(torch.randn(8, 4, 16)) == "slot"
    assert layer._estimate_operator_schedule(batch=1, length=4) == "palette"
    assert layer._estimate_operator_schedule(batch=1, length=64) == "slot"
    assert layer._estimate_operator_schedule(batch=8, length=64) == "effective"

    factorized = UnFold(dim=16, exposed=64, value_operators=4, value_rank=4)
    assert factorized._estimate_operator_schedule(batch=8, length=4) == "slot"


def test_unfold_rejects_invalid_private_operator_schedule() -> None:
    layer = UnFold(dim=3, exposed=2)
    layer._operator_schedule = "unknown"
    with pytest.raises(RuntimeError, match="operator schedule"):
        layer(torch.randn(1, 4, 3))


def test_unfold_auto_schedule_keeps_training_and_cpu_on_slot() -> None:
    layer = UnFold(dim=128, exposed=128, value_operators=4).eval()
    cpu_x = torch.randn(2, 32, 128)
    assert layer._select_operator_schedule(cpu_x) == "slot"

    layer.train()
    assert layer._select_operator_schedule(cpu_x) == "slot"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_unfold_auto_triton_inference_matches_slot(dtype: torch.dtype) -> None:
    torch.manual_seed(41)
    automatic = UnFold(dim=128, exposed=128, value_operators=4).cuda().to(dtype).eval()
    slot = UnFold(dim=128, exposed=128, value_operators=4).cuda().to(dtype).eval()
    slot.load_state_dict(automatic.state_dict())
    slot._operator_schedule = "slot"
    x = torch.randn(2, 32, 128, device="cuda", dtype=dtype)
    mask = torch.tensor(
        [
            [index % 3 != 1 for index in range(32)],
            [False] * 32,
        ],
        device="cuda",
    )

    assert automatic._select_operator_schedule(x) == "slot"
    with torch.inference_mode():
        assert automatic._select_operator_schedule(x) == "fused_triton"
        actual_exposed, actual_attended = automatic._query_exposed(x, mask)
        expected_exposed, expected_attended = slot._query_exposed(x, mask)

    atol = 1e-6 if dtype == torch.float32 else 4e-3
    rtol = 1e-5 if dtype == torch.float32 else 2e-2
    assert torch.allclose(actual_exposed, expected_exposed, atol=atol, rtol=rtol)
    assert torch.allclose(actual_attended, expected_attended, atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_unfold_auto_triton_gate_rejects_unmeasured_shapes() -> None:
    with torch.inference_mode():
        low_dim = UnFold(dim=64, exposed=128).cuda().to(torch.bfloat16).eval()
        low_exposed = UnFold(dim=128, exposed=64).cuda().to(torch.bfloat16).eval()
        long_input = UnFold(dim=128, exposed=128).cuda().to(torch.bfloat16).eval()
        assert low_dim._select_operator_schedule(
            torch.randn(2, 32, 64, device="cuda", dtype=torch.bfloat16)
        ) == "slot"
        assert low_exposed._select_operator_schedule(
            torch.randn(2, 32, 128, device="cuda", dtype=torch.bfloat16)
        ) == "slot"
        assert long_input._select_operator_schedule(
            torch.randn(2, 129, 128, device="cuda", dtype=torch.bfloat16)
        ) == "slot"

        fp32_shared_memory_fallback = (
            UnFold(dim=256, exposed=128).cuda().eval()
        )
        fp32_input = torch.randn(2, 128, 256, device="cuda")
        assert fp32_shared_memory_fallback._select_operator_schedule(
            fp32_input
        ) == "palette_triton"
        fp32_shared_memory_fallback._operator_schedule = "fused_triton"
        with pytest.raises(RuntimeError, match="fused Triton"):
            fp32_shared_memory_fallback._select_operator_schedule(fp32_input)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_unfold_auto_missing_triton_falls_back_to_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = UnFold(dim=128, exposed=128).cuda().to(torch.bfloat16).eval()
    x = torch.randn(2, 32, 128, device="cuda", dtype=torch.bfloat16)
    monkeypatch.setattr("arti.nn._triton_palette_is_available", lambda: False)
    with torch.inference_mode():
        assert layer._select_operator_schedule(x) == "slot"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_unfold_auto_compile_uses_standard_path() -> None:
    class QueryStage(torch.nn.Module):
        def __init__(self, layer: UnFold) -> None:
            super().__init__()
            self.layer = layer

        def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return self.layer._query_exposed(x, mask)[0]

    automatic = UnFold(dim=128, exposed=128, value_operators=4).cuda().eval()
    expected_layer = UnFold(dim=128, exposed=128, value_operators=4).cuda().eval()
    expected_layer.load_state_dict(automatic.state_dict())
    expected_layer._operator_schedule = "slot"
    x = torch.randn(2, 32, 128, device="cuda")
    mask = torch.ones(2, 32, device="cuda", dtype=torch.bool)
    compiled = torch.compile(QueryStage(automatic), fullgraph=True)
    with torch.inference_mode():
        actual = compiled(x, mask)
        expected = expected_layer._query_exposed(x, mask)[0]
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)


def test_unfold_guide_and_condition_are_optional_and_independently_disabled() -> None:
    x = torch.randn(2, 4, 3)
    guide = torch.randn(2, 4, 2)
    condition = torch.randn(2, 5)
    plain = UnFold(dim=3, exposed=2)
    with pytest.raises(ValueError, match="guide_dim is disabled"):
        plain(x, guide=guide)
    with pytest.raises(ValueError, match="condition_dim is disabled"):
        plain(x, condition=condition)

    guided = UnFold(dim=3, exposed=2, guide_dim=2, condition_dim=5)
    assert guided(x).shape == (2, 6, 3)
    assert guided(x, guide=guide).shape == (2, 6, 3)
    assert guided(x, condition=condition).shape == (2, 6, 3)
    assert guided(x, guide=guide, condition=condition).shape == (2, 6, 3)


def test_unfold_guide_normalization_is_positive_affine_invariant() -> None:
    guide = torch.randn(3, 7, 4)
    mask = torch.tensor(
        [[True] * 7, [True, True, True, False, False, False, False], [False] * 7]
    )
    first = UnFold._normalize_guide(guide, mask)
    second = UnFold._normalize_guide(3.5 * guide + 17.0, mask)
    assert torch.allclose(first, second, atol=2e-5, rtol=2e-5)
    assert torch.equal(first[~mask], torch.zeros_like(first[~mask]))
    assert torch.isfinite(first).all()


def test_unfold_masked_guide_values_do_not_affect_valid_layout() -> None:
    torch.manual_seed(9)
    layer = UnFold(dim=3, exposed=2, guide_dim=2).eval()
    x = torch.randn(1, 5, 3)
    mask = torch.tensor([[True, True, True, False, False]])
    guide = torch.randn(1, 5, 2)
    changed = guide.clone()
    changed[:, 3:] = 1e20
    _, output_mask1, source1 = layer(x, mask, guide=guide, return_source_index=True)
    _, output_mask2, source2 = layer(x, mask, guide=changed, return_source_index=True)
    assert torch.equal(source1, source2)
    assert torch.equal(output_mask1, output_mask2)


def test_unfold_guide_and_condition_gradients_are_finite() -> None:
    layer = UnFold(dim=4, exposed=2, guide_dim=3, condition_dim=2)
    x = torch.randn(2, 5, 4, requires_grad=True)
    guide = torch.randn(2, 5, 3, requires_grad=True)
    condition = torch.randn(2, 2, requires_grad=True)
    layer(x, guide=guide, condition=condition).square().mean().backward()
    assert guide.grad is not None and torch.isfinite(guide.grad).all()
    assert condition.grad is not None and torch.isfinite(condition.grad).all()


@pytest.mark.parametrize("field", ["guide", "condition"])
def test_unfold_rejects_nonfinite_layout_inputs(field: str) -> None:
    layer = UnFold(
        dim=3,
        guide_dim=2,
        condition_dim=2,
        validate_values=True,
    )
    x = torch.randn(1, 4, 3)
    value = torch.randn(1, 4, 2) if field == "guide" else torch.randn(1, 2)
    value.reshape(-1)[0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        layer(x, **{field: value})


def test_unfold_value_validation_is_an_explicit_debug_option() -> None:
    layer = UnFold(dim=3, guide_dim=1)
    x = torch.randn(1, 4, 3)
    guide = torch.full((1, 4, 1), torch.nan)
    assert layer(x, guide=guide).shape == (1, 5, 3)
    assert "validate_values=True" not in repr(layer)
    assert "validate_values=True" in repr(
        UnFold(dim=3, guide_dim=1, validate_values=True)
    )


def test_unfold_sort_eval_can_return_soft_layout() -> None:
    layer = UnFold(dim=3, exposed=2).eval()
    y, soft = layer(torch.randn(2, 4, 3), return_soft_layout=True)
    assert y.shape == (2, 6, 3)
    assert soft.shape == (2, 6, 6)
    assert torch.isfinite(soft).all()


def test_unfold_mask_moves_with_values_and_invalid_values_sort_last() -> None:
    layer = UnFold(dim=2, exposed=2)
    x = torch.randn(2, 4, 2)
    mask = torch.tensor([[True, False, True, False], [False, False, False, False]])
    y, output_mask, exposed_mask, source = layer(
        x, mask, return_exposed_mask=True, return_source_index=True
    )
    assert y.shape == (2, 6, 2)
    assert output_mask.shape == (2, 6)
    assert torch.equal(output_mask[0], torch.tensor([True, True, True, True, False, False]))
    assert not output_mask[1].any()
    assert torch.equal(exposed_mask, source >= x.shape[1])


def test_unfold_batch_like_leading_dimensions() -> None:
    layer = UnFold(dim=3, exposed=2)
    x = torch.randn(2, 3, 5, 3)
    y, source = layer(x, return_source_index=True)
    assert y.shape == (2, 3, 7, 3)
    assert source.shape == (2, 3, 7)


def test_unfold_enforces_alpha_backend_length_limit() -> None:
    layer = UnFold(dim=3, exposed=2, max_length=8, hard_backend="greedy")
    with pytest.raises(ValueError, match="exceeds max_length=8"):
        layer(torch.randn(1, 7, 3))


def test_unfold_sort_backend_is_not_bound_by_dense_backend_limit() -> None:
    layer = UnFold(dim=3, exposed=2, max_length=8)
    assert layer(torch.randn(1, 17, 3)).shape == (1, 19, 3)


def test_unfold_sort_backend_allows_exposed_to_exceed_max_length() -> None:
    layer = UnFold(dim=3, exposed=9, max_length=4)
    assert layer(torch.randn(1, 2, 3)).shape == (1, 11, 3)


def test_unfold_layout_is_permutation_equivariant_as_a_value_set() -> None:
    torch.manual_seed(19)
    layer = UnFold(dim=3, exposed=2, guide_dim=1).eval()
    x = torch.randn(4, 7, 3)
    guide = torch.randn(4, 7, 1)
    first = layer(x, guide=guide)
    permutation = torch.rand(4, 7).argsort(1)
    permuted_x = x.gather(1, permutation[..., None].expand_as(x))
    permuted_guide = guide.gather(1, permutation[..., None].expand_as(guide))
    second = layer(permuted_x, guide=permuted_guide)
    assert torch.allclose(first, second, atol=1e-6, rtol=1e-6)


def test_unfold_valid_output_is_padding_invariant() -> None:
    torch.manual_seed(23)
    layer = UnFold(dim=3, exposed=2, guide_dim=1).eval()
    x = torch.randn(3, 5, 3)
    guide = torch.randn(3, 5, 1)
    mask = torch.tensor([[True] * 5, [True, True, True, False, False], [True, False, True, False, False]])
    short, short_mask = layer(x, mask, guide=guide)
    padded_x = torch.cat((x, torch.zeros(3, 4, 3)), dim=1)
    padded_guide = torch.cat((guide, torch.zeros(3, 4, 1)), dim=1)
    padded_mask = torch.cat((mask, torch.zeros(3, 4, dtype=torch.bool)), dim=1)
    long, long_mask = layer(padded_x, padded_mask, guide=padded_guide)
    for short_row, short_valid, long_row, long_valid in zip(
        short, short_mask, long, long_mask, strict=True
    ):
        assert torch.allclose(short_row[short_valid], long_row[long_valid], atol=1e-6, rtol=1e-6)


def test_unfold_rejects_unknown_hard_backend() -> None:
    with pytest.raises(ValueError, match="hard_backend"):
        UnFold(dim=3, hard_backend="unknown")


@pytest.mark.parametrize("temperature", [float("nan"), float("inf")])
def test_unfold_rejects_nonfinite_temperature(temperature: float) -> None:
    with pytest.raises(ValueError, match="temperature"):
        UnFold(dim=3, temperature=temperature)


def test_unfold_rejects_invalid_layout_configuration() -> None:
    with pytest.raises(ValueError, match="layout_mode"):
        UnFold(dim=3, layout_mode="unknown")
    with pytest.raises(ValueError, match="guide_dim=1"):
        UnFold(dim=3, guide_dim=2, layout_mode="canonical")
    with pytest.raises(ValueError, match="hard_backend='sort'"):
        UnFold(
            dim=3,
            guide_dim=1,
            layout_mode="canonical",
            hard_backend="greedy",
        )


def test_unfold_canonical_guide_layout_is_length_stable() -> None:
    torch.manual_seed(29)
    layer = UnFold(
        dim=3,
        exposed=2,
        guide_dim=1,
        layout_mode="canonical",
    ).eval()
    x = torch.randn(2, 6, 3)
    guide = torch.randn(2, 6, 1)
    short = layer(x, guide=guide)
    permutation = torch.rand(2, 6).argsort(1)
    permuted = layer(
        x.gather(1, permutation[..., None].expand_as(x)),
        guide=guide.gather(1, permutation[..., None].expand_as(guide)),
    )
    assert torch.allclose(short, permuted, atol=1e-6, rtol=1e-6)


def test_unfold_canonical_layout_requires_runtime_guide() -> None:
    layer = UnFold(dim=3, guide_dim=1, layout_mode="canonical")
    with pytest.raises(ValueError, match="requires guide"):
        layer(torch.randn(1, 4, 3))


def test_unfold_external_exposed_guide_controls_canonical_insertion() -> None:
    layer = UnFold(dim=2, exposed=2, guide_dim=1, layout_mode="canonical").eval()
    x = torch.tensor([[[1.0, 10.0], [2.0, 20.0], [3.0, 30.0]]])
    guide = torch.tensor([[[-1.0], [0.0], [1.0]]])
    exposed_guide = torch.tensor([[[-0.5], [0.5]]])
    _, source = layer(
        x,
        guide=guide,
        exposed_guide=exposed_guide,
        return_source_index=True,
    )
    assert torch.equal(source, torch.tensor([[0, 3, 1, 4, 2]]))


def test_unfold_exposed_mask_controls_dynamic_valid_expansion() -> None:
    layer = UnFold(dim=3, exposed=3).eval()
    x = torch.randn(2, 4, 3)
    mask = torch.ones(2, 4, dtype=torch.bool)
    requested = torch.tensor([[True, False, True], [False, False, True]])
    _, output_mask, exposed_origin = layer(
        x,
        mask,
        exposed_mask=requested,
        return_exposed_mask=True,
    )
    assert torch.equal(output_mask.sum(1), torch.tensor([6, 5]))
    assert torch.equal((output_mask & exposed_origin).sum(1), torch.tensor([2, 1]))


def test_unfold_validates_exposed_layout_inputs() -> None:
    canonical = UnFold(dim=3, exposed=2, guide_dim=1, layout_mode="canonical")
    x = torch.randn(2, 4, 3)
    guide = torch.randn(2, 4, 1)
    with pytest.raises(ValueError, match="exposed_guide must have shape"):
        canonical(x, guide=guide, exposed_guide=torch.randn(2, 3, 1))
    with pytest.raises(ValueError, match="exposed_mask must be a boolean"):
        canonical(x, guide=guide, exposed_mask=torch.ones(2, 2))
    learned = UnFold(dim=3, exposed=2, guide_dim=1)
    with pytest.raises(ValueError, match="only supported by canonical"):
        learned(x, guide=guide, exposed_guide=torch.randn(2, 2, 1))


def test_unfold_auction_backend_preserves_a_bijection() -> None:
    layer = UnFold(dim=3, exposed=1, hard_backend="auction").eval()
    x = torch.randn(2, 4, 3)
    y, source = layer(x, return_source_index=True)
    assert y.shape == (2, 5, 3)
    assert torch.equal(source.sort(1).values, torch.arange(5).expand(2, -1))


def test_unfold_hard_forward_and_soft_layout_gradient() -> None:
    torch.manual_seed(11)
    layer = UnFold(dim=4, exposed=2)
    x = torch.randn(3, 5, 4, requires_grad=True)
    y, source, soft_layout = layer(x, return_source_index=True, return_soft_layout=True)
    y.square().mean().backward()
    assert torch.equal(_original_counts(source, 5), torch.ones(3, 5, dtype=torch.long))
    assert torch.allclose(soft_layout.sum(-1), torch.ones_like(soft_layout.sum(-1)), atol=1e-2)
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert layer.layout_score[0].weight.grad is not None
    assert torch.count_nonzero(layer.layout_score[0].weight.grad) > 0
    assert layer.exposed_queries.grad is not None


@pytest.mark.parametrize(
    ("x", "mask", "message"),
    [
        (torch.randn(4), None, "shape [..., N, D]"),
        (torch.randn(2, 4, 2), None, "feature dimension"),
        (torch.randn(2, 4, 3), torch.ones(2, 3, dtype=torch.bool), "mask must have shape"),
        (torch.randn(2, 4, 3), torch.ones(2, 4), "boolean"),
    ],
)
def test_unfold_rejects_invalid_inputs(x: torch.Tensor, mask: torch.Tensor | None, message: str) -> None:
    layer = UnFold(dim=3, exposed=2)
    with pytest.raises(ValueError, match=message.replace("[", r"\[").replace("]", r"\]")):
        layer(x, mask)


def test_unfold_state_dict_and_arti_st_round_trip(tmp_path: Path) -> None:
    torch.manual_seed(13)
    source = UnFold(dim=4, exposed=2).eval()
    x = torch.randn(2, 5, 4)
    expected = source(x)
    target = UnFold(dim=4, exposed=2).eval()
    target.load_state_dict(source.state_dict())
    assert torch.equal(target(x), expected)
    saved = arti.save(source, tmp_path / "unfold.st")
    safe_target = UnFold(dim=4, exposed=2).eval()
    arti.load(saved.weights_path, model=safe_target)
    assert torch.equal(safe_target(x), expected)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_unfold_cpu_dtype_and_autograd(dtype: torch.dtype) -> None:
    layer = UnFold(dim=4, exposed=2).to(dtype=dtype)
    x = torch.randn(2, 4, 4, dtype=dtype, requires_grad=True)
    y = layer(x)
    y.float().square().mean().backward()
    assert y.dtype == dtype
    assert x.grad is not None and torch.isfinite(x.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_unfold_cuda_dtype_and_autograd(dtype: torch.dtype) -> None:
    layer = UnFold(dim=8, exposed=3).cuda().to(dtype=dtype)
    x = torch.randn(2, 9, 8, device="cuda", dtype=dtype, requires_grad=True)
    y = layer(x)
    y.float().square().mean().backward()
    assert y.device.type == "cuda"
    assert y.dtype == dtype
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_public_unfold_exports_are_consistent() -> None:
    assert arti.UnFold is UnFold
    assert arti.torch.UnFold is UnFold
