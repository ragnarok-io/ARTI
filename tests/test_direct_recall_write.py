from __future__ import annotations

import copy

import pytest
import torch
from torch import nn

from arti.blocks import ARTIResidualBlock
from arti.config import ARTIConfig, STATE_RECALL_COMPOSITION_FACTOR
from arti.layers import ARTILatentRecallField, ARTIRecallWriteLayer
from arti.recall_formula import FactorSpec, FormulaIdentity, RecallFormulaContract


class _IdentityNextStateFormula(nn.Module):
    recall_formula_contract = RecallFormulaContract(
        identity=FormulaIdentity("tests/identity-next-state", 1),
        factors=(FactorSpec("unused", init="zero"),),
        identity_preserving=True,
    )

    def forward(self, state: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
        del factors
        return state


class _SharedRouteFormula(nn.Module):
    recall_formula_contract = RecallFormulaContract(
        identity=FormulaIdentity("tests/shared-route", 1),
        factors=(
            FactorSpec("left", route="pair", init="zero"),
            FactorSpec("right", route="pair", init="zero"),
        ),
        identity_preserving=True,
    )

    def forward(self, state: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
        return state + factors.sum(dim=-2)


class _IndependentRouteFormula(nn.Module):
    recall_formula_contract = RecallFormulaContract(
        identity=FormulaIdentity("tests/independent-route", 1),
        factors=(
            FactorSpec("left", route="left", init="normal"),
            FactorSpec("right", route="right", init="normal"),
        ),
        identity_preserving=True,
    )

    def forward(self, state: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
        return state + factors.sum(dim=-2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_formula_route_buffers_follow_runtime_tensor_device() -> None:
    block = ARTIResidualBlock(
        dim=4,
        recall_slots=4,
        recall_steps=1,
        recall_activation="none",
        recall_routing="dense",
        recall_formula=_SharedRouteFormula(),
        use_phase_mixer=False,
        use_virtual_interface=False,
        use_virtual_recall=False,
        direct_recall=True,
    )
    field = block.layer.state.recall
    logits = torch.randn(2, 3, 2, 4, device="cuda")

    shared = field._share_factor_route_logits(logits)

    assert shared.device.type == "cuda"
    assert shared.shape == logits.shape


def _direct_block(
    *,
    dim: int = 4,
    steps: int = 1,
    tolerance: float | None = None,
    use_virtual_recall: bool = False,
    zero_init_output: bool = False,
    recall_slots: int = 1,
    value_composition: str = "single",
) -> ARTIResidualBlock:
    return ARTIResidualBlock(
        dim=dim,
        hidden_dim=2,
        recall_slots=recall_slots,
        recall_steps=steps,
        recall_tolerance=tolerance,
        recall_activation="none",
        recall_routing="dense",
        recall_value_composition=value_composition,
        use_phase_mixer=False,
        use_virtual_interface=False,
        use_virtual_recall=use_virtual_recall,
        zero_init_output=zero_init_output,
        direct_recall=True,
    )


def test_direct_recall_bank_values_are_host_dimensional_writes() -> None:
    block = _direct_block(steps=2)
    write = torch.tensor([0.5, -1.0, 1.5, -2.0])
    with torch.no_grad():
        block.layer.state.recall.bank.copy_(write.unsqueeze(0))

    x = torch.randn(2, 3, 4)
    output = block(x)

    assert block.layer.state.recall.bank.shape == (1, 4)
    assert not hasattr(block.layer, "out_proj")
    assert not hasattr(block, "out")
    torch.testing.assert_close(
        output,
        x + 2.0 * write.view(1, 1, -1),
    )


def test_formula_route_assignment_preserves_forward_and_isolates_bank_gradients() -> None:
    block = ARTIResidualBlock(
        dim=4,
        recall_slots=4,
        recall_steps=1,
        recall_activation="none",
        recall_routing="dense",
        recall_formula=_IndependentRouteFormula(),
        use_phase_mixer=False,
        use_virtual_interface=False,
        use_virtual_recall=False,
        direct_recall=True,
    )
    x = torch.randn(2, 3, 4)
    baseline = block(x)
    assignment = torch.tensor([[1.0, 0.0], [1.0, 0.0]])

    routed = block(x, route_assignment=assignment)

    torch.testing.assert_close(routed, baseline, rtol=0.0, atol=0.0)
    routed.square().mean().backward()
    gradient = block.layer.state.recall.bank.grad
    assert gradient is not None
    left, right = gradient.chunk(2, dim=0)
    assert torch.count_nonzero(left).item() > 0
    assert torch.count_nonzero(right).item() == 0


def test_formula_route_assignment_validates_shape_and_values() -> None:
    block = ARTIResidualBlock(
        dim=4,
        recall_slots=4,
        recall_steps=1,
        recall_activation="none",
        recall_routing="dense",
        recall_formula=_IndependentRouteFormula(),
        use_phase_mixer=False,
        use_virtual_interface=False,
        use_virtual_recall=False,
        direct_recall=True,
    )
    x = torch.randn(2, 3, 4)

    for invalid in (
        torch.ones(2),
        torch.ones(2, 3),
        torch.ones(2, 2, dtype=torch.long),
    ):
        with pytest.raises((TypeError, ValueError), match="route_assignment"):
            block(x, route_assignment=invalid)


def test_route_assignment_rejects_legacy_composition_without_formula_routes() -> None:
    block = _direct_block(value_composition="product", recall_slots=4)

    with pytest.raises(ValueError, match="Formula"):
        block(
            torch.randn(2, 3, 4),
            route_assignment=torch.ones(2, 2),
        )


def test_direct_recall_write_can_create_features_absent_from_input() -> None:
    block = _direct_block()
    write = torch.tensor([1.0, -2.0, 3.0, -4.0])
    with torch.no_grad():
        block.layer.state.recall.bank.copy_(write.unsqueeze(0))

    output = block(torch.zeros(1, 2, 4))

    torch.testing.assert_close(output, write.view(1, 1, -1).expand(1, 2, -1))


def test_direct_recall_mask_blocks_write_without_changing_invalid_tokens() -> None:
    block = _direct_block()
    write = torch.tensor([1.0, -2.0, 3.0, -4.0])
    with torch.no_grad():
        block.layer.state.recall.bank.copy_(write.unsqueeze(0))
    x = torch.randn(1, 3, 4)
    mask = torch.tensor([[True, False, True]])

    output = block(x, mask=mask)

    torch.testing.assert_close(output[:, 0], x[:, 0] + write)
    torch.testing.assert_close(output[:, 1], x[:, 1])
    torch.testing.assert_close(output[:, 2], x[:, 2] + write)


def test_identity_initialized_recall_preserves_input_and_bank_receives_gradient() -> None:
    block = _direct_block(zero_init_output=True)
    x = torch.randn(2, 3, 4)
    target = torch.randn_like(x)

    output = block(x)
    torch.testing.assert_close(output, x, rtol=0.0, atol=0.0)

    torch.nn.functional.mse_loss(output, target).backward()
    bank = block.layer.state.recall.bank
    assert bank.grad is not None
    assert torch.count_nonzero(bank.grad).item() > 0


def test_identity_initialized_product_recall_trains_both_value_banks() -> None:
    block = _direct_block(
        zero_init_output=True,
        recall_slots=4,
        value_composition="product",
    )
    x = torch.randn(2, 3, 4)
    target = torch.randn_like(x)

    output = block(x)
    torch.testing.assert_close(output, x, rtol=0.0, atol=0.0)

    torch.nn.functional.mse_loss(output, target).backward()
    bank = block.layer.state.recall.bank
    assert bank.grad is not None
    assert torch.count_nonzero(bank.grad[:2]).item() > 0
    assert torch.count_nonzero(bank.grad[2:]).item() > 0


def test_state_recall_composes_input_and_trains_all_factor_banks() -> None:
    block = _direct_block(
        recall_slots=17,
        value_composition="state",
    )
    with torch.no_grad():
        block.layer.state.recall.bank[0].fill_(0.5)
        block.layer.state.recall.bank[1].fill_(0.1)
        block.layer.state.recall.bank[2:15].fill_(0.2)
        block.layer.state.recall.bank[15].fill_(0.3)
        block.layer.state.recall.bank[16].fill_(-0.4)
    x = torch.randn(2, 3, 4)
    target = torch.randn_like(x)

    output = block(x)
    opacity = torch.tanh(torch.tensor(-0.4)).square()
    polynomial = (1.0 + torch.tanh(torch.tensor(0.2)) / 13.0) ** 13
    content = 0.5 + torch.tanh(torch.tensor(0.1))
    expected = (1.0 - opacity) * x + (
        opacity * torch.tanh(torch.tensor(0.3)) * content * polynomial
    )
    torch.testing.assert_close(output, expected)

    torch.nn.functional.mse_loss(output, target).backward()
    bank = block.layer.state.recall.bank
    assert bank.grad is not None
    for factor in bank.grad.chunk(STATE_RECALL_COMPOSITION_FACTOR, dim=0):
        assert torch.count_nonzero(factor).item() > 0


def test_state_recall_input_retention_interpolates_without_persisting() -> None:
    block = _direct_block(
        zero_init_output=True,
        recall_slots=17,
        value_composition="state",
    )
    recalled = torch.tensor([2.0, -3.0, 4.0, -5.0])
    with torch.no_grad():
        block.layer.state.recall.bank[0].copy_(recalled)
        block.layer.state.recall.bank[1:15].zero_()
        block.layer.state.recall.bank[15].fill_(20.0)
        block.layer.state.recall.bank[16].fill_(20.0)
    block.layer.state.mark_state_bank_calibrated()
    x = torch.randn(1, 3, 4)

    assert block.layer.state.state_input_retention == 1.0
    torch.testing.assert_close(block(x), x, rtol=0.0, atol=0.0)

    block.layer.state.set_state_input_retention(0.5)
    torch.testing.assert_close(
        block(x),
        0.5 * x + 0.5 * recalled.view(1, 1, -1),
    )

    block.layer.state.set_state_input_retention(0.0)
    torch.testing.assert_close(
        block(x),
        recalled.view(1, 1, -1).expand_as(x),
    )
    assert any("state_input_retention" in key for key in block.state_dict())


def test_state_recall_calibrates_random_content_to_host_feature_scale() -> None:
    block = _direct_block(
        zero_init_output=True,
        recall_slots=68,
        value_composition="state",
    )
    x = 4.0 + 3.0 * torch.randn(8, 16, 4)

    assert block.layer.state.state_bank_calibrated is False
    torch.testing.assert_close(block(x), x, rtol=0.0, atol=0.0)

    factors = block.layer.state.recall.bank.chunk(
        STATE_RECALL_COMPOSITION_FACTOR,
        dim=0,
    )
    content = factors[0]
    fine_content = factors[1]
    modulations = factors[2:15]
    write_direction = factors[15]
    memory_opacity = factors[16]
    assert block.layer.state.state_bank_calibrated is True
    torch.testing.assert_close(
        content.float().mean(dim=0),
        x.float().reshape(-1, x.shape[-1]).mean(dim=0),
    )
    torch.testing.assert_close(
        content.float().std(dim=0, unbiased=False),
        x.float().reshape(-1, x.shape[-1]).std(dim=0, unbiased=False),
    )
    assert torch.count_nonzero(content).item() == content.numel()
    assert torch.count_nonzero(fine_content).item() == 0
    for modulation in modulations:
        assert torch.count_nonzero(modulation).item() == modulation.numel()
    assert torch.count_nonzero(write_direction).item() == write_direction.numel()
    assert torch.count_nonzero(memory_opacity).item() == memory_opacity.numel()


def test_loaded_state_recall_bank_is_not_recalibrated() -> None:
    torch.manual_seed(59)
    source = _direct_block(
        zero_init_output=True,
        recall_slots=68,
        value_composition="state",
    )
    source(6.0 + 2.5 * torch.randn(3, 5, 4))
    expected_bank = source.layer.state.recall.bank.detach().clone()

    restored = _direct_block(
        zero_init_output=True,
        recall_slots=68,
        value_composition="state",
    )
    assert restored.layer.state.state_bank_calibrated is False
    restored.load_state_dict(source.state_dict())
    assert restored.layer.state.state_bank_calibrated is True

    restored(-11.0 + 0.1 * torch.randn(2, 7, 4))

    torch.testing.assert_close(restored.layer.state.recall.bank, expected_bank)


def test_multifactor_state_initialization_stays_stable_across_depth() -> None:
    torch.manual_seed(7)
    layers = torch.nn.ModuleList(
        [
            _direct_block(
                dim=16,
                zero_init_output=True,
                recall_slots=68,
                value_composition="state",
            )
            for _index in range(30)
        ]
    )
    x = torch.randn(2, 8, 16)
    output = x
    for layer in layers:
        layer.layer.state.set_state_input_retention(0.0)
        output = layer(output)

    relative_drift = (output - x).norm() / x.norm()
    norm_ratio = output.norm() / x.norm()
    assert torch.isfinite(output).all()
    assert relative_drift < 0.2
    assert 0.8 < norm_ratio < 1.2


def test_state_recall_retention_below_one_preserves_bank_gradients() -> None:
    block = _direct_block(
        zero_init_output=True,
        recall_slots=17,
        value_composition="state",
    )
    block.layer.state.set_state_input_retention(0.9)
    x = torch.randn(2, 3, 4)
    target = torch.randn_like(x)

    torch.nn.functional.mse_loss(block(x), target).backward()

    bank = block.layer.state.recall.bank
    assert bank.grad is not None
    for factor in bank.grad.chunk(STATE_RECALL_COMPOSITION_FACTOR, dim=0):
        assert torch.count_nonzero(factor).item() > 0


def test_state_recall_retention_is_anchored_to_original_input_across_steps() -> None:
    block = _direct_block(
        steps=3,
        recall_slots=17,
        value_composition="state",
    )
    recalled = torch.tensor([2.0, -3.0, 4.0, -5.0])
    with torch.no_grad():
        block.layer.state.recall.bank[0].copy_(recalled)
        block.layer.state.recall.bank[1:15].zero_()
        block.layer.state.recall.bank[15].fill_(20.0)
        block.layer.state.recall.bank[16].fill_(20.0)
    block.layer.state.set_state_input_retention(0.5)
    x = torch.randn(1, 3, 4)

    output = block(x)

    torch.testing.assert_close(
        output,
        0.5 * x + 0.5 * recalled.view(1, 1, -1),
    )


def test_state_recall_preserves_only_masked_out_input_tokens() -> None:
    block = _direct_block(
        recall_slots=17,
        value_composition="state",
    )
    recalled = torch.tensor([2.0, -3.0, 4.0, -5.0])
    with torch.no_grad():
        block.layer.state.recall.bank[0].copy_(recalled)
        block.layer.state.recall.bank[1:15].zero_()
        block.layer.state.recall.bank[15].fill_(20.0)
        block.layer.state.recall.bank[16].fill_(20.0)
    x = torch.randn(1, 3, 4)
    mask = torch.tensor([[True, False, True]])

    output = block(x, mask=mask)

    torch.testing.assert_close(output[:, 0], recalled.unsqueeze(0))
    torch.testing.assert_close(output[:, 1], x[:, 1])
    torch.testing.assert_close(output[:, 2], recalled.unsqueeze(0))


def test_custom_formula_next_state_is_not_added_to_previous_state() -> None:
    layer = ARTIRecallWriteLayer(
        ARTIConfig(
            input_dim=2,
            hidden_dim=2,
            recall_slots=2,
            recall_steps=1,
            recall_activation="none",
            recall_routing="grouped",
            recall_key_dim=2,
            recall_group_size=1,
            recall_group_topk=1,
            use_phase_mixer=False,
            use_virtual_interface=False,
            use_pairwise_context=False,
        ),
        formula=_IdentityNextStateFormula(),
    )
    inputs = torch.tensor([[[1.5, -2.0], [0.25, 3.0]]])

    output = layer.forward_write(inputs)

    torch.testing.assert_close(output, inputs, rtol=0.0, atol=0.0)


def test_state_recall_can_route_independent_factors_and_replace_input() -> None:
    layer = ARTIRecallWriteLayer(
        ARTIConfig(
            input_dim=2,
            hidden_dim=2,
            recall_slots=34,
            recall_steps=1,
            recall_activation="none",
            recall_routing="grouped",
            recall_key_dim=2,
            recall_group_size=1,
            recall_group_topk=1,
            recall_value_composition="state",
            use_phase_mixer=False,
            use_virtual_interface=False,
            use_pairwise_context=False,
        )
    )
    field = layer.state.recall
    assert field.group_bank is not None
    assert field.key_bank is not None
    with torch.no_grad():
        field.query.weight.copy_(torch.eye(2))
        routes = torch.tensor([[1.0, 0.0], [0.0, 1.0]]).repeat(
            STATE_RECALL_COMPOSITION_FACTOR,
            1,
        )
        field.group_bank.copy_(routes)
        field.key_bank.copy_(routes)
        field.bank.zero_()
        field.bank[0].fill_(3.0)
        field.bank[1].fill_(-2.0)
        field.bank[-4:].fill_(20.0)

    inputs = torch.tensor([[[10.0, 0.0]], [[0.0, 10.0]]])
    output = layer.forward_write(inputs, mask=None)

    torch.testing.assert_close(
        output,
        torch.tensor([[[3.0, 3.0]], [[-2.0, -2.0]]]),
    )


def test_custom_formula_can_share_one_grouped_route_across_factor_pair() -> None:
    field = ARTILatentRecallField(
        hidden_dim=2,
        slots=4,
        recognition_mode="none",
        routing="grouped",
        key_dim=2,
        group_size=1,
        group_topk=1,
        formula=_SharedRouteFormula(),
    )
    assert field.group_bank is not None
    assert field.key_bank is not None
    with torch.no_grad():
        field.query.weight.copy_(torch.eye(2))
        field.group_bank.copy_(
            torch.tensor(
                [
                    [10.0, 0.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                    [1.0, 0.0],
                ]
            )
        )
        field.key_bank.copy_(field.group_bank)
        field.bank.copy_(
            torch.tensor(
                [
                    [2.0, 0.0],
                    [20.0, 0.0],
                    [3.0, 0.0],
                    [30.0, 0.0],
                ]
            )
        )

    result = field(
        torch.tensor([[[1.0, 0.0]]]),
        torch.ones(1, 1, dtype=torch.bool),
    )

    assert result.indices.squeeze().tolist() == [0, 2]
    torch.testing.assert_close(result.context, torch.tensor([[[5.0, 0.0]]]))
    result.context.square().mean().backward()
    assert field.bank.grad is not None
    assert field.group_bank.grad is not None
    assert torch.count_nonzero(field.group_bank.grad)
    assert field.query.weight.grad is None
    assert not field.query.weight.requires_grad
    assert field.key_bank.grad is None


def test_direct_recall_state_dict_round_trip_preserves_output() -> None:
    source = _direct_block(steps=2)
    with torch.no_grad():
        source.layer.state.recall.bank.copy_(torch.tensor([[0.25, -0.5, 0.75, -1.0]]))
    restored = copy.deepcopy(source)
    restored.load_state_dict(source.state_dict())
    x = torch.randn(2, 4)

    torch.testing.assert_close(restored(x), source(x))
    assert all(".out." not in name for name in source.state_dict())
    assert all("out_proj" not in name for name in source.state_dict())


def test_direct_recall_write_fast_path_matches_full_output_and_gradients() -> None:
    torch.manual_seed(29)
    full = _direct_block(steps=3, tolerance=0.01)
    fast = copy.deepcopy(full)
    x_full = torch.randn(2, 5, 4, requires_grad=True)
    x_fast = x_full.detach().clone().requires_grad_(True)
    mask = torch.tensor(
        [
            [True, True, True, False, False],
            [True, True, True, True, True],
        ]
    )

    full_output = full.layer(x_full, mask=mask).y
    fast_output = fast.layer.forward_write(x_fast, mask=mask)
    torch.testing.assert_close(fast_output, full_output)

    full_output.square().mean().backward()
    fast_output.square().mean().backward()
    torch.testing.assert_close(x_fast.grad, x_full.grad)
    for full_parameter, fast_parameter in zip(
        full.parameters(),
        fast.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(fast_parameter.grad, full_parameter.grad)


def test_direct_recall_write_unmasked_fast_path_matches_full_output_and_gradients() -> None:
    torch.manual_seed(31)
    full = _direct_block(steps=3)
    fast = copy.deepcopy(full)
    x_full = torch.randn(2, 5, 4, requires_grad=True)
    x_fast = x_full.detach().clone().requires_grad_(True)

    full_output = full.layer(x_full).y
    fast_output = fast.layer.forward_write(x_fast)
    torch.testing.assert_close(fast_output, full_output)

    full_output.square().mean().backward()
    fast_output.square().mean().backward()
    torch.testing.assert_close(x_fast.grad, x_full.grad)
    for full_parameter, fast_parameter in zip(
        full.parameters(),
        fast.parameters(),
        strict=True,
    ):
        torch.testing.assert_close(fast_parameter.grad, full_parameter.grad)


def test_direct_recall_write_hook_observes_fast_path_without_changing_output() -> None:
    block = _direct_block(steps=2)
    x = torch.randn(2, 3, 4)
    captured: list[torch.Tensor] = []
    handle = block.layer._register_write_hook(lambda _layer, written: captured.append(written))

    observed = block.layer.forward_write(x)
    handle.remove()
    uncaptured = block.layer.forward_write(x)

    assert len(captured) == 1
    assert captured[0] is observed
    torch.testing.assert_close(observed, uncaptured)


def test_direct_block_uses_full_output_while_layer_hook_is_active() -> None:
    block = _direct_block(steps=2, use_virtual_recall=True)
    captured = []
    handle = block.layer.register_forward_hook(
        lambda _module, _inputs, output: captured.append(output)
    )
    try:
        output = block(torch.randn(2, 3, 4))
    finally:
        handle.remove()

    assert output.shape == (2, 3, 4)
    assert len(captured) == 1
    assert captured[0].recall_prediction is not None


def test_direct_recall_write_fast_path_does_not_poll_device_state(
    monkeypatch,
) -> None:
    block = _direct_block(steps=3, tolerance=0.01)

    def fail_any(*_args, **_kwargs):
        raise AssertionError("forward_write must not call torch.any")

    monkeypatch.setattr(torch, "any", fail_any)
    written = block.layer.forward_write(
        torch.randn(2, 3, 4),
        static_steps=True,
    )

    assert written.shape == (2, 3, 4)


def test_direct_recall_training_uses_compiler_friendly_static_steps(
    monkeypatch,
) -> None:
    block = _direct_block(steps=3, tolerance=0.01)
    block.train()

    def fail_any(*_args, **_kwargs):
        raise AssertionError("training must not poll Recall convergence on the host")

    monkeypatch.setattr(torch, "any", fail_any)

    output = block(torch.randn(2, 3, 4))

    assert output.shape == (2, 3, 4)


def _compiled_product_layer() -> ARTIRecallWriteLayer:
    return ARTIRecallWriteLayer(
        ARTIConfig(
            input_dim=8,
            hidden_dim=8,
            coord_dim=0,
            recall_slots=16,
            recall_steps=2,
            recall_activation="half",
            recall_routing="grouped",
            recall_key_dim=4,
            recall_group_size=2,
            recall_group_topk=1,
            recall_value_composition="product",
            use_pairwise_context=False,
            use_phase_mixer=False,
            use_virtual_interface=False,
        )
    )


def test_product_recall_layers_share_runtime_compiled_tail() -> None:
    first = _compiled_product_layer()
    second = _compiled_product_layer()

    first.compile_write_hotpath(fullgraph=True)
    second.compile_write_hotpath(fullgraph=True)

    assert first.write_hotpath_compiled
    assert second.write_hotpath_compiled
    assert first.state._compiled_product_tail is second.state._compiled_product_tail
    assert first._compiled_state_write is None
    assert second._compiled_state_write is None


def test_compiled_product_tail_is_runtime_only_and_enable_is_idempotent() -> None:
    layer = _compiled_product_layer()
    keys_before = tuple(layer.state_dict())

    layer.compile_write_hotpath(fullgraph=True)
    compiled = layer.state._compiled_product_tail
    layer.compile_write_hotpath(fullgraph=True)

    assert layer.state._compiled_product_tail is compiled
    assert tuple(layer.state_dict()) == keys_before
