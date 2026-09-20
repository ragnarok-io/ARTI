from __future__ import annotations

import copy

import torch

import arti
from arti._recall_state import (
    AffineRecallTransition,
    AffineRecallValueUpdater,
    FixedFeatureValueBank,
    MatrixAffineRecallTransition,
    NormalizedDeltaRecallValueUpdater,
    RecallStateExpertAssembly,
    RecallStateController,
    RecallValueUpdater,
    StackedRecallValueUpdater,
    adapt_recall_capacity_state,
    reduce_affine_recall_transitions,
    reduce_matrix_affine_recall_transitions,
)
from benchmarks.run_qwen_dialogue_recall_transition import (
    DialogueBatch,
    RecallQualityFlowSchedule,
    dialogue_loss,
    evaluate_quality_depths,
    quality_refine_auxiliary,
    select_positions,
)


def make_updater() -> RecallValueUpdater:
    return RecallValueUpdater(
        hidden_dim=4,
        slots=3,
        workspace_dim=8,
        depth=1,
        interface_slots=2,
    )


def test_updater_captures_and_reuses_full_block_route_stack() -> None:
    updater = RecallValueUpdater(
        hidden_dim=4,
        slots=3,
        workspace_dim=8,
        depth=2,
        interface_slots=2,
        recall_slots=8,
        recall_group_topk=2,
        recall_steps=3,
    )
    trace = torch.randn(2, 5, 4)
    previous = torch.randn(2, 3, 4)
    mask = torch.ones(2, 5, dtype=torch.bool)

    captured, info = updater(
        trace,
        previous,
        mask=mask,
        recall_steps=1,
        return_route_stack=True,
    )
    route_stack = info["route_stack"]
    assert isinstance(route_stack, arti.RecallRouteStack)
    assert route_stack.axis == "block"
    assert len(route_stack.items) == updater.depth
    replayed = updater(
        trace,
        previous,
        mask=mask,
        recall_steps=3,
        route_stack=route_stack,
    )

    assert isinstance(captured, torch.Tensor)
    assert isinstance(replayed, torch.Tensor)
    assert captured.shape == replayed.shape == previous.shape
    assert torch.isfinite(replayed).all()


def test_affine_recall_transition_is_associative_and_order_sensitive() -> None:
    first = AffineRecallTransition(torch.tensor([[[0.5]]]), torch.tensor([[[1.0]]]))
    second = AffineRecallTransition(torch.tensor([[[0.25]]]), torch.tensor([[[2.0]]]))
    third = AffineRecallTransition(torch.tensor([[[0.75]]]), torch.tensor([[[-1.0]]]))
    value = torch.tensor([[[4.0]]])

    left = first.then(second).then(third)
    right = first.then(second.then(third))

    torch.testing.assert_close(left.retention, right.retention, rtol=0, atol=0)
    torch.testing.assert_close(left.write, right.write, rtol=0, atol=0)
    torch.testing.assert_close(left.apply(value), right.apply(value), rtol=0, atol=0)
    assert not torch.equal(first.then(second).write, second.then(first).write)


def test_fixed_feature_value_bank_is_exactly_linear_in_values() -> None:
    bank = FixedFeatureValueBank(5, 7, 3, seed=19)
    x = torch.randn(2, 4, 5)
    values = torch.randn(2, 7, 3)
    increment = torch.randn(2, 7, 3)

    before = bank(x, values)
    after = bank(x, values + increment)

    torch.testing.assert_close(after - before, bank(x, increment))
    assert not tuple(bank.named_parameters())
    assert bank.state_semantics == "values_only"
    assert bank.feature_semantics == "fixed"


def test_recall_state_controller_is_exact_noop_for_zero_bank() -> None:
    controller = RecallStateController(8, control_dim=8, heads=2)
    hidden = torch.randn(2, 5, 8)
    zero_bank = torch.zeros(2, 3, 8)
    with torch.no_grad():
        controller.control_scale.fill_(1.0)

    result = controller(hidden, zero_bank)

    torch.testing.assert_close(result, torch.zeros_like(result), rtol=0, atol=0)
    assert all(module.bias is None for module in (
        controller.query,
        controller.key,
        controller.value,
        controller.output,
    ))


def test_recall_state_controller_reads_values_and_backpropagates() -> None:
    controller = RecallStateController(8, control_dim=8, heads=2)
    with torch.no_grad():
        controller.control_scale.fill_(0.5)
    hidden = torch.randn(2, 5, 8, requires_grad=True)
    bank = torch.randn(2, 3, 8, requires_grad=True)
    mask = torch.tensor([[True, True, False, False, False], [True] * 5])

    result = controller(hidden, bank, mask=mask)
    result.square().mean().backward()

    assert torch.count_nonzero(result[:, :2]) > 0
    assert torch.count_nonzero(result[0, 2:]) == 0
    assert hidden.grad is not None and torch.isfinite(hidden.grad).all()
    assert bank.grad is not None and torch.isfinite(bank.grad).all()
    assert all(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in controller.parameters()
    )


def test_recall_expert_assembly_preserves_forced_branch_exactly() -> None:
    old = RecallStateController(8, control_dim=8, heads=2)
    new = RecallStateController(8, control_dim=8, heads=2)
    assembly = RecallStateExpertAssembly((old, new), topk=1)
    hidden = torch.randn(2, 5, 8)
    banks = torch.randn(2, 2, 4, 8)
    route = torch.zeros(2, 5, 2)
    route[..., 0] = 1.0

    expected = old(hidden, banks[:, 0])
    actual = assembly(hidden, banks, route_weights=route)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    changed_new_bank = banks.clone()
    changed_new_bank[:, 1].mul_(1000.0)
    torch.testing.assert_close(
        assembly(hidden, changed_new_bank, route_weights=route),
        expected,
        rtol=0,
        atol=0,
    )


def test_recall_expert_assembly_only_backpropagates_selected_value_branch() -> None:
    assembly = RecallStateExpertAssembly(
        (
            RecallStateController(8, control_dim=8, heads=2),
            RecallStateController(8, control_dim=8, heads=2),
        ),
        topk=1,
    )
    hidden = torch.randn(2, 3, 8, requires_grad=True)
    banks = torch.randn(2, 2, 4, 8, requires_grad=True)
    route = torch.zeros(2, 3, 2)
    route[..., 1] = 1.0

    assembly(hidden, banks, route_weights=route).square().mean().backward()

    assert banks.grad is not None
    assert torch.count_nonzero(banks.grad[:, 0]) == 0
    assert torch.count_nonzero(banks.grad[:, 1]) > 0
    assert all(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) == 0
        for parameter in assembly.experts[0].parameters()
    )
    assert all(
        parameter.grad is not None and torch.count_nonzero(parameter.grad) > 0
        for parameter in assembly.experts[1].parameters()
    )


def test_recall_expert_assembly_auto_route_is_deterministic_after_reload() -> None:
    assembly = RecallStateExpertAssembly(
        (
            RecallStateController(8, control_dim=8, heads=2),
            RecallStateController(8, control_dim=8, heads=2),
        ),
        topk=1,
    ).eval()
    clone = RecallStateExpertAssembly(
        (
            RecallStateController(8, control_dim=8, heads=2),
            RecallStateController(8, control_dim=8, heads=2),
        ),
        topk=1,
    ).eval()
    clone.load_state_dict(copy.deepcopy(assembly.state_dict()))
    hidden = torch.randn(2, 3, 8)
    banks = torch.randn(2, 2, 4, 8)

    first = assembly(hidden, banks, return_info=True)
    second = clone(hidden, banks, return_info=True)

    torch.testing.assert_close(first.delta, second.delta, rtol=0, atol=0)
    torch.testing.assert_close(first.route_weights, second.route_weights, rtol=0, atol=0)
    assert torch.equal(first.selected_experts, second.selected_experts)


def test_recall_expert_assembly_supports_three_independent_branches() -> None:
    experts = tuple(
        RecallStateController(8, control_dim=8, heads=2) for _ in range(3)
    )
    assembly = RecallStateExpertAssembly(experts, topk=1)
    hidden = torch.randn(2, 4, 8)
    banks = torch.randn(2, 3, 5, 8)
    route = torch.zeros(2, 4, 3)
    route[..., 2] = 1.0

    expected = experts[2](hidden, banks[:, 2])
    actual = assembly(hidden, banks, route_weights=route)

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert tuple(assembly.route_weight.shape) == (3, 3)
    torch.testing.assert_close(
        assembly.route_weight,
        torch.eye(3),
        rtol=0,
        atol=0,
    )


def test_affine_recall_value_updater_matches_token_and_chunk_composition() -> None:
    updater = AffineRecallValueUpdater(4, 3, workspace_dim=8).eval()
    with torch.no_grad():
        updater.retention_weight.normal_(std=0.1)
        updater.value_weight.normal_(std=0.1)
    trace = torch.randn(2, 6, 4)
    initial = torch.randn(2, 3, 4)
    token_transitions = updater.transitions(trace)
    whole = reduce_affine_recall_transitions(
        token_transitions.retention,
        token_transitions.write,
    )
    left = reduce_affine_recall_transitions(
        token_transitions.retention[:, :2],
        token_transitions.write[:, :2],
    )
    right = reduce_affine_recall_transitions(
        token_transitions.retention[:, 2:],
        token_transitions.write[:, 2:],
    )
    recurrent = initial
    for position in range(trace.shape[1]):
        recurrent = updater(trace[:, position : position + 1], recurrent)

    torch.testing.assert_close(whole.retention, left.then(right).retention)
    torch.testing.assert_close(whole.write, left.then(right).write)
    torch.testing.assert_close(updater(trace, initial), whole.apply(initial))
    torch.testing.assert_close(recurrent, whole.apply(initial))


def test_affine_recall_value_updater_masks_identity_and_backpropagates() -> None:
    updater = AffineRecallValueUpdater(4, 2, workspace_dim=6)
    with torch.no_grad():
        updater.value_weight.normal_(std=0.1)
    trace = torch.randn(2, 3, 4)
    initial = torch.randn(2, 2, 4, requires_grad=True)
    mask = torch.tensor([[False, False, False], [True, False, True]])

    result = updater(trace, initial, mask=mask)
    result.square().mean().backward()

    torch.testing.assert_close(result[0], initial.detach()[0], rtol=0, atol=0)
    assert initial.grad is not None and torch.isfinite(initial.grad).all()
    assert updater.value_weight.grad is not None
    assert torch.isfinite(updater.value_weight.grad).all()
    assert updater.state_semantics == "values_only"
    assert updater.transition_semantics == "affine_monoid"


def test_matrix_affine_transition_composes_slot_mixing_exactly() -> None:
    first = MatrixAffineRecallTransition(
        torch.tensor([[[1.0, 0.0], [0.5, 0.5]]]),
        torch.tensor([[[0.1, 0.2], [0.3, 0.4]]]),
    )
    second = MatrixAffineRecallTransition(
        torch.tensor([[[0.75, 0.25], [0.0, 1.0]]]),
        torch.tensor([[[0.5, -0.5], [0.2, 0.1]]]),
    )
    value = torch.randn(1, 2, 2)

    expected = second.apply(first.apply(value))
    actual = first.then(second).apply(value)

    torch.testing.assert_close(actual, expected)


def test_normalized_delta_updater_matches_token_and_chunk_composition() -> None:
    updater = NormalizedDeltaRecallValueUpdater(4, 3, workspace_dim=8).eval()
    with torch.no_grad():
        updater.value_weight.normal_(std=0.1)
        updater.rate_weight.normal_(std=0.1)
    trace = torch.randn(2, 6, 4)
    initial = torch.randn(2, 3, 4)
    token_transitions = updater.transitions(trace)
    whole = reduce_matrix_affine_recall_transitions(
        token_transitions.matrix,
        token_transitions.write,
    )
    left = reduce_matrix_affine_recall_transitions(
        token_transitions.matrix[:, :2],
        token_transitions.write[:, :2],
    )
    right = reduce_matrix_affine_recall_transitions(
        token_transitions.matrix[:, 2:],
        token_transitions.write[:, 2:],
    )
    recurrent = initial
    for position in range(trace.shape[1]):
        recurrent = updater(trace[:, position : position + 1], recurrent)

    torch.testing.assert_close(whole.matrix, left.then(right).matrix)
    torch.testing.assert_close(whole.write, left.then(right).write)
    torch.testing.assert_close(updater(trace, initial), whole.apply(initial))
    torch.testing.assert_close(recurrent, whole.apply(initial))


def test_normalized_delta_updater_masks_identity_and_backpropagates() -> None:
    updater = NormalizedDeltaRecallValueUpdater(4, 2, workspace_dim=6)
    with torch.no_grad():
        updater.value_weight.normal_(std=0.1)
    trace = torch.randn(2, 3, 4)
    initial = torch.randn(2, 2, 4, requires_grad=True)
    mask = torch.tensor([[False, False, False], [True, False, True]])

    result = updater(trace, initial, mask=mask)
    result.square().mean().backward()

    torch.testing.assert_close(result[0], initial.detach()[0], rtol=0, atol=0)
    assert initial.grad is not None and torch.isfinite(initial.grad).all()
    assert updater.value_weight.grad is not None
    assert torch.isfinite(updater.value_weight.grad).all()
    assert updater.transition_semantics == "matrix_affine_monoid"


def test_factored_normalized_delta_updater_matches_explicit_transition() -> None:
    updater = NormalizedDeltaRecallValueUpdater(
        4, 3, workspace_dim=8, factors=2
    ).eval()
    with torch.no_grad():
        updater.value_weight.normal_(std=0.1)
        updater.rate_weight.normal_(std=0.1)
    trace = torch.randn(2, 5, 4)
    initial = torch.randn(2, 3, 4, requires_grad=True)
    mask = torch.tensor([[True, False, True, True, False], [True] * 5])

    fast = updater(trace, initial, mask=mask)
    explicit, transition = updater(
        trace, initial, mask=mask, return_transition=True
    )

    torch.testing.assert_close(fast, explicit, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(explicit, transition.apply(initial))
    fast.square().mean().backward()
    assert updater.value_weight.grad is not None
    assert torch.isfinite(updater.value_weight.grad).all()
    assert updater.factors == 2


def test_legacy_writer_hypothesis_is_not_public() -> None:
    assert not hasattr(arti, "RecallWriter")
    assert not hasattr(arti, "RecallWriteProgram")


def test_updater_starts_as_complete_identity_transition() -> None:
    updater = make_updater()
    trace = torch.randn(2, 5, 4)
    previous = torch.randn(2, 3, 4)

    next_value, info = updater(trace, previous, return_info=True)

    assert torch.equal(next_value, previous)
    assert torch.equal(info["update"], torch.zeros_like(previous))
    assert updater.transition_parameter_count() == 3 * 8 * 4 + 3 * 4


def test_updater_can_enable_zero_write_internal_recall_refinement() -> None:
    updater = RecallValueUpdater(
        hidden_dim=4,
        slots=3,
        workspace_dim=8,
        depth=1,
        interface_slots=2,
        recall_slots=4,
        recall_steps=2,
    )
    recall = updater.workspace[0].layer.state.recall
    assert recall is not None
    assert updater.workspace[0].layer.config.recall_steps == 2
    assert torch.count_nonzero(recall.bank) == 0
    assert recall.key_bank is not None
    assert not torch.equal(recall.key_bank[0], recall.key_bank[1])
    assert not recall.query.weight.requires_grad

    trace = torch.randn(2, 5, 4)
    previous = torch.randn(2, 3, 4)
    assert torch.equal(updater(trace, previous), previous)

    with torch.no_grad():
        updater.value_weight.normal_(std=0.05)
    updater(trace, previous).square().mean().backward()
    assert recall.bank.grad is not None
    assert torch.isfinite(recall.bank.grad).all()
    assert torch.count_nonzero(recall.bank.grad) > 0


def test_internal_recall_upgrade_preserves_existing_updater_output() -> None:
    torch.manual_seed(17)
    legacy = make_updater().eval()
    with torch.no_grad():
        legacy.value_weight.normal_(std=0.05)
        legacy.value_bias.normal_(std=0.01)
    refined = RecallValueUpdater(
        hidden_dim=4,
        slots=3,
        workspace_dim=8,
        depth=1,
        interface_slots=2,
        recall_slots=4,
        recall_steps=2,
    ).eval()
    result = refined.load_state_dict(legacy.state_dict(), strict=False)
    assert result.unexpected_keys == []
    assert result.missing_keys
    assert all(".layer.state.recall." in name for name in result.missing_keys)

    trace = torch.randn(2, 5, 4)
    previous = torch.randn(2, 3, 4)
    assert torch.equal(refined(trace, previous), legacy(trace, previous))


def test_internal_recall_depth_can_be_selected_per_forward() -> None:
    torch.manual_seed(19)
    updater = RecallValueUpdater(
        hidden_dim=4,
        slots=3,
        workspace_dim=8,
        depth=1,
        interface_slots=2,
        recall_slots=4,
        recall_steps=3,
    ).eval()
    with torch.no_grad():
        updater.value_weight.normal_(std=0.05)
        recall = updater.workspace[0].layer.state.recall
        assert recall is not None
        recall.bank.normal_(std=0.1)
    trace = torch.randn(2, 5, 4)
    previous = torch.randn(2, 3, 4)

    depth_zero = updater(trace, previous, recall_steps=0)
    depth_one = updater(trace, previous, recall_steps=1)
    depth_three = updater(trace, previous, recall_steps=3)

    assert depth_zero.shape == previous.shape
    assert not torch.equal(depth_zero, depth_one)
    assert not torch.equal(depth_one, depth_three)
    assert updater.recall_steps == 3
    assert updater.workspace[0].layer.config.recall_steps == 3


def test_internal_recall_candidate_groups_restrict_search_with_live_gradients() -> None:
    torch.manual_seed(29)
    updater = RecallValueUpdater(
        hidden_dim=4,
        slots=3,
        workspace_dim=8,
        depth=2,
        interface_slots=2,
        recall_slots=4,
        recall_group_topk=2,
        recall_route_exploration=1.0,
        recall_steps=2,
    ).train()
    with torch.no_grad():
        updater.value_weight.normal_(std=0.05)
        for block in updater.workspace:
            recall = block.layer.state.recall
            assert recall is not None
            recall.bank.normal_(std=0.1)
    trace = torch.randn(2, 5, 4)
    previous = torch.randn(2, 3, 4)

    candidate, info = updater(trace, previous, return_info=True)
    candidate_groups = info["candidate_groups"]
    replay, replay_info = updater(
        trace,
        previous,
        candidate_groups=candidate_groups,
        return_info=True,
    )
    replay.square().mean().backward()

    assert candidate_groups.shape == (2, 2, 8, 2)
    torch.testing.assert_close(
        replay_info["candidate_groups"], candidate_groups, rtol=0, atol=0
    )
    assert torch.isfinite(candidate).all() and torch.isfinite(replay).all()
    for block in updater.workspace:
        recall = block.layer.state.recall
        assert recall is not None
        assert recall.group_bank is not None
        assert recall.group_bank.grad is not None
        assert torch.isfinite(recall.group_bank.grad).all()


def test_stacked_updater_candidate_groups_restrict_each_site() -> None:
    torch.manual_seed(31)
    modules = [
        RecallValueUpdater(
            hidden_dim=4,
            slots=3,
            workspace_dim=8,
            depth=2,
            interface_slots=2,
            recall_slots=4,
            recall_group_topk=2,
            recall_route_exploration=1.0,
            recall_steps=2,
        ).train()
        for _ in range(2)
    ]
    with torch.no_grad():
        for module in modules:
            module.value_weight.normal_(std=0.05)
            for block in module.workspace:
                recall = block.layer.state.recall
                assert recall is not None
                recall.bank.normal_(std=0.1)
    stacked = StackedRecallValueUpdater(modules).train()
    trace = torch.randn(2, 2, 5, 4)
    previous = torch.randn(2, 2, 3, 4)
    mask = torch.ones(2, 2, 5, dtype=torch.bool)

    candidate, candidate_groups = stacked(
        trace,
        previous,
        mask=mask,
        recall_steps=2,
        return_candidate_groups=True,
    )
    replay = stacked(
        trace,
        previous,
        mask=mask,
        recall_steps=2,
        candidate_groups=candidate_groups,
    )
    replay.square().mean().backward()

    assert candidate_groups.shape == (2, 2, 2, 8, 2)
    assert torch.isfinite(candidate).all() and torch.isfinite(replay).all()
    live_gradients = [
        parameter.grad
        for parameter in stacked.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert live_gradients
    assert all(torch.isfinite(gradient).all() for gradient in live_gradients)


def test_internal_recall_capacity_expansion_preserves_inherited_read() -> None:
    torch.manual_seed(23)
    source = RecallValueUpdater(
        hidden_dim=4,
        slots=3,
        workspace_dim=8,
        depth=1,
        interface_slots=2,
        recall_slots=4,
        recall_group_topk=2,
        recall_steps=2,
    ).eval()
    with torch.no_grad():
        source.value_weight.normal_(std=0.05)
        source.value_bias.normal_(std=0.01)
        recall = source.workspace[0].layer.state.recall
        assert recall is not None
        recall.bank.normal_(std=0.1)
    expanded = RecallValueUpdater(
        hidden_dim=4,
        slots=3,
        workspace_dim=8,
        depth=1,
        interface_slots=2,
        recall_slots=8,
        recall_group_topk=4,
        recall_steps=2,
    ).eval()
    state, migration = adapt_recall_capacity_state(expanded, source.state_dict())
    expanded.load_state_dict(state, strict=True)

    trace = torch.randn(2, 5, 4)
    previous = torch.randn(2, 3, 4)
    torch.manual_seed(47)
    expanded_output = expanded(trace, previous)
    torch.manual_seed(47)
    source_output = source(trace, previous)
    torch.testing.assert_close(
        expanded_output,
        source_output,
        rtol=2e-5,
        atol=2e-6,
    )
    assert migration["factor"] == 2
    assert migration["source_slots"] == 4
    assert migration["target_slots"] == 8


def test_internal_recall_capacity_expansion_can_break_route_symmetry() -> None:
    source = RecallValueUpdater(
        hidden_dim=4,
        slots=3,
        workspace_dim=8,
        depth=1,
        interface_slots=2,
        recall_slots=4,
        recall_group_topk=2,
        recall_steps=1,
    )
    expanded = RecallValueUpdater(
        hidden_dim=4,
        slots=3,
        workspace_dim=8,
        depth=1,
        interface_slots=2,
        recall_slots=8,
        recall_group_topk=4,
        recall_steps=1,
    )
    state, _migration = adapt_recall_capacity_state(
        expanded,
        source.state_dict(),
        symmetry_break=1e-4,
    )
    group_name = "workspace.0.layer.state.recall.group_bank"
    bank_name = "workspace.0.layer.state.recall.bank"
    group_bank = state[group_name].reshape(4, 2, -1)
    bank = state[bank_name].reshape(4, 2, -1)

    assert torch.equal(bank[:, 0], bank[:, 1])
    assert not torch.equal(group_bank[:, 0], group_bank[:, 1])
    torch.testing.assert_close(group_bank.mean(dim=1), source.state_dict()[group_name])


def test_runtime_recall_depth_can_exceed_construction_default() -> None:
    updater = RecallValueUpdater(
        hidden_dim=4,
        slots=3,
        workspace_dim=8,
        depth=1,
        interface_slots=2,
        recall_slots=4,
        recall_steps=2,
    )
    trace = torch.randn(2, 5, 4)
    previous = torch.randn(2, 3, 4)

    result = updater(trace, previous, recall_steps=3)

    assert result.shape == previous.shape
    assert torch.isfinite(result).all()
    assert updater.recall_steps == 2


def test_quality_refine_schedule_reuses_sparse_z_image_pairing() -> None:
    schedule = RecallQualityFlowSchedule((1, 3, 6), pair_interval=4, seed=7)
    plans = [schedule.plan(step) for step in range(12)]

    assert all(len(plans[step]) == 2 for step in (0, 4, 8))
    assert all(len(plans[step]) == 1 for step in set(range(12)) - {0, 4, 8})
    assert set(schedule.summary()["pair_histogram"]).issubset({"0:1", "1:3", "3:6"})


def test_quality_refine_schedule_accepts_disabled_empty_depths() -> None:
    schedule = RecallQualityFlowSchedule((), pair_interval=4, seed=7)

    assert not schedule.enabled
    assert schedule.plan(0) == ()
    assert schedule.summary()["enabled"] is False


def test_quality_refine_auxiliary_stops_shallow_gradient() -> None:
    current = torch.zeros(1, 2, 2)
    teacher = torch.tensor([[[1.0, 0.0]]])
    shallow = torch.zeros(1, 2, 2, requires_grad=True)
    deep = torch.zeros(1, 2, 2, requires_grad=True)
    deep.data[0, 0, 0] = 0.25
    batch = DialogueBatch(
        history=torch.zeros(1, 1, 1, 2),
        history_mask=torch.ones(1, 1, 1, dtype=torch.bool),
        current_local=current,
        current_mask=torch.ones(1, 2, dtype=torch.bool),
        prediction_positions=torch.zeros(1, 1, dtype=torch.long),
        prediction_mask=torch.ones(1, 1, dtype=torch.bool),
        teacher_hidden=teacher,
        target_ids=torch.zeros(1, 1, dtype=torch.long),
    )
    primary = (deep[0, 0, 0] - 1.0).square()
    auxiliary, parts = quality_refine_auxiliary(
        primary,
        torch.tensor(0.5),
        deep,
        shallow,
        batch,
        rank_weight=0.05,
        direction_weight=0.02,
        margin=0.01,
    )

    auxiliary.backward()

    assert shallow.grad is None
    assert deep.grad is not None
    assert torch.isfinite(deep.grad).all()
    assert set(parts) == {
        "quality_rank",
        "quality_direction",
        "quality_relative_improvement",
    }


def test_quality_depth_evaluation_pairs_adjacent_depths() -> None:
    updater = RecallValueUpdater(
        hidden_dim=2,
        slots=3,
        workspace_dim=4,
        depth=1,
        interface_slots=2,
        recall_slots=2,
        recall_steps=1,
    ).eval()
    field = arti.ARTILatentRecallField(
        2,
        3,
        routing="grouped",
        key_dim=2,
        query_mode="fixed",
        group_size=1,
        group_topk=1,
        project_external=False,
    ).eval()
    batch = DialogueBatch(
        history=torch.randn(2, 1, 2, 2),
        history_mask=torch.ones(2, 1, 2, dtype=torch.bool),
        current_local=torch.randn(2, 2, 2),
        current_mask=torch.ones(2, 2, dtype=torch.bool),
        prediction_positions=torch.zeros(2, 1, dtype=torch.long),
        prediction_mask=torch.ones(2, 1, dtype=torch.bool),
        teacher_hidden=torch.randn(2, 1, 2),
        target_ids=torch.zeros(2, 1, dtype=torch.long),
    )

    result = evaluate_quality_depths(
        updater,
        field,
        batch,
        (0, 1),
        final_norm=torch.nn.LayerNorm(2),
        lm_head=torch.nn.Linear(2, 3),
    )

    assert result["depths"] == [0, 1]
    assert set(result["adjacent_monotonic_rate"]) == {"0:1"}
    assert set(result["adjacent_direction_cosine"]) == {"0:1"}


def test_all_token_supervision_selects_every_causal_target() -> None:
    assert torch.equal(select_positions(3, 8, None), torch.arange(3, 8))


def test_target_token_objective_is_an_explicit_opt_in() -> None:
    batch = DialogueBatch(
        history=torch.zeros(1, 1, 1, 2),
        history_mask=torch.ones(1, 1, 1, dtype=torch.bool),
        current_local=torch.zeros(1, 2, 2),
        current_mask=torch.ones(1, 2, dtype=torch.bool),
        prediction_positions=torch.zeros(1, 1, dtype=torch.long),
        prediction_mask=torch.ones(1, 1, dtype=torch.bool),
        teacher_hidden=torch.tensor([[[1.0, -1.0]]]),
        target_ids=torch.ones(1, 1, dtype=torch.long),
    )
    student = torch.tensor([[[0.5, -0.5], [0.0, 0.0]]])
    norm = torch.nn.LayerNorm(2)
    head = torch.nn.Linear(2, 3)

    plain, plain_parts = dialogue_loss(
        student,
        batch,
        final_norm=norm,
        lm_head=head,
        temperature=1.0,
    )
    supervised, supervised_parts = dialogue_loss(
        student,
        batch,
        final_norm=norm,
        lm_head=head,
        temperature=1.0,
        target_token_weight=0.1,
    )

    assert "target_token" not in plain_parts
    assert "target_token" in supervised_parts
    assert supervised > plain


def test_chunked_logit_alignment_matches_dense_loss_and_hidden_gradient() -> None:
    torch.manual_seed(41)
    batch = DialogueBatch(
        history=torch.zeros(2, 1, 1, 4),
        history_mask=torch.ones(2, 1, 1, dtype=torch.bool),
        current_local=torch.zeros(2, 3, 4),
        current_mask=torch.ones(2, 3, dtype=torch.bool),
        prediction_positions=torch.tensor([[0, 1], [1, 2]]),
        prediction_mask=torch.ones(2, 2, dtype=torch.bool),
        teacher_hidden=torch.randn(2, 2, 4),
        target_ids=torch.tensor([[1, 3], [2, 4]]),
    )
    dense_student = torch.randn(2, 3, 4, requires_grad=True)
    chunked_student = dense_student.detach().clone().requires_grad_(True)
    dense_norm = torch.nn.LayerNorm(4)
    chunked_norm = copy.deepcopy(dense_norm)
    dense_head = torch.nn.Linear(4, 7, bias=False)
    chunked_head = copy.deepcopy(dense_head)

    dense_loss, dense_parts = dialogue_loss(
        dense_student,
        batch,
        final_norm=dense_norm,
        lm_head=dense_head,
        temperature=1.5,
        target_token_weight=0.1,
    )
    chunked_loss, chunked_parts = dialogue_loss(
        chunked_student,
        batch,
        final_norm=chunked_norm,
        lm_head=chunked_head,
        temperature=1.5,
        target_token_weight=0.1,
        logit_token_chunk_size=2,
    )
    dense_loss.backward()
    chunked_loss.backward()

    torch.testing.assert_close(chunked_loss, dense_loss, rtol=1e-5, atol=1e-6)
    for name in dense_parts:
        torch.testing.assert_close(chunked_parts[name], dense_parts[name], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        chunked_student.grad,
        dense_student.grad,
        rtol=2e-5,
        atol=2e-6,
    )


def test_masked_trace_values_do_not_affect_transition() -> None:
    torch.manual_seed(18)
    updater = make_updater().eval()
    with torch.no_grad():
        updater.value_weight.normal_(std=0.1)
    previous = torch.randn(2, 3, 4)
    left = torch.randn(2, 5, 4)
    right = left.clone()
    right[:, 3:] = 10_000.0
    mask = torch.tensor([[True, True, True, False, False]]).expand(2, -1)

    left_value = updater(left, previous, mask=mask)
    right_value = updater(right, previous, mask=mask)

    assert torch.equal(left_value, right_value)


def test_empty_trace_mask_is_an_exact_no_op() -> None:
    updater = make_updater()
    with torch.no_grad():
        updater.value_weight.normal_()
        updater.value_bias.normal_()
    trace = torch.randn(2, 4, 4)
    previous = torch.randn(2, 3, 4)
    mask = torch.zeros(2, 4, dtype=torch.bool)

    assert torch.equal(updater(trace, previous, mask=mask), previous)


def test_slot_output_maps_receive_independent_gradients() -> None:
    torch.manual_seed(22)
    updater = make_updater()
    trace = torch.randn(2, 5, 4)
    previous = torch.zeros(2, 3, 4)
    target = torch.stack(
        (
            torch.full((2, 4), -1.0),
            torch.full((2, 4), 0.5),
            torch.full((2, 4), 2.0),
        ),
        dim=1,
    )

    torch.nn.functional.mse_loss(updater(trace, previous), target).backward()

    gradient = updater.value_weight.grad
    assert gradient is not None
    assert torch.isfinite(gradient).all()
    assert not torch.equal(gradient[0], gradient[1])
    assert not torch.equal(gradient[1], gradient[2])


def test_unrolled_state_transition_is_differentiable_across_turns() -> None:
    torch.manual_seed(26)
    updater = make_updater()
    with torch.no_grad():
        updater.value_weight.normal_(std=0.05)
    first_trace = torch.randn(2, 3, 4, requires_grad=True)
    second_trace = torch.randn(2, 2, 4, requires_grad=True)
    value = torch.zeros(2, 3, 4)

    value = updater(first_trace.detach(), value)
    value = updater(second_trace.detach(), value)
    value.square().mean().backward()

    assert first_trace.grad is None
    assert second_trace.grad is None
    assert updater.value_weight.grad is not None
    assert updater.state_projection.weight.grad is not None
    assert torch.isfinite(updater.state_projection.weight.grad).all()


def test_state_dict_roundtrip_is_exact_in_eval_mode() -> None:
    torch.manual_seed(31)
    updater = make_updater().eval()
    with torch.no_grad():
        updater.value_weight.normal_(std=0.03)
        updater.value_bias.normal_(std=0.01)
    trace = torch.randn(2, 5, 4)
    previous = torch.randn(2, 3, 4)
    expected = updater(trace, previous)
    restored = make_updater().eval()

    restored.load_state_dict(copy.deepcopy(updater.state_dict()))

    assert torch.equal(restored(trace, previous), expected)
    restored.freeze()
    assert not restored.training
    assert all(not parameter.requires_grad for parameter in restored.parameters())


def test_stacked_updater_route_exploration_is_reproducible_per_candidate() -> None:
    torch.manual_seed(37)
    updater = RecallValueUpdater(
        hidden_dim=4,
        slots=3,
        workspace_dim=8,
        depth=1,
        interface_slots=2,
        recall_slots=8,
        recall_group_topk=1,
        recall_route_exploration=10.0,
        recall_steps=1,
    )
    with torch.no_grad():
        updater.value_weight.normal_(std=0.2)
        recall = updater.workspace[0].layer.state.recall
        assert recall is not None
        recall.bank.normal_(std=0.4)
        assert recall.group_bank is not None
        recall.group_bank.normal_(std=0.4)
    stacked = StackedRecallValueUpdater((updater, copy.deepcopy(updater))).train()
    trace = torch.randn(3, 2, 5, 4)
    previous = torch.randn(3, 2, 3, 4)
    mask = torch.ones(3, 2, 5, dtype=torch.bool)

    torch.manual_seed(1234)
    first = stacked(trace, previous, mask=mask, recall_steps=1)
    torch.manual_seed(1234)
    replay = stacked(trace, previous, mask=mask, recall_steps=1)

    torch.testing.assert_close(replay, first, rtol=0, atol=0)
    assert not torch.equal(first[:, 0], first[:, 1])


def test_validation_rejects_mismatched_state_contracts() -> None:
    updater = make_updater()
    trace = torch.randn(2, 5, 4)
    previous = torch.randn(2, 3, 4)

    try:
        updater(trace, previous, mask=torch.ones(2, 4, dtype=torch.bool))
    except ValueError as exc:
        assert "mask must have shape" in str(exc)
    else:
        raise AssertionError("invalid mask shape was accepted")

    try:
        updater(trace, torch.randn(2, 4, 4))
    except ValueError as exc:
        assert "previous_value trailing shape" in str(exc)
    else:
        raise AssertionError("invalid Bank slot count was accepted")


def test_explicit_batched_value_state_drives_dense_recall_without_mutation() -> None:
    field = arti.ARTILatentRecallField(
        4,
        4,
        routing="dense",
        value_composition="single",
        project_external=False,
    )
    parameter_before = field.bank.detach().clone()
    value = torch.randn(2, 4, 4, requires_grad=True)
    hidden = torch.randn(2, 3, 4)
    mask = torch.ones(2, 3, dtype=torch.bool)

    context = field.read_context(hidden, mask, memory=value)
    context.square().mean().backward()

    assert value.grad is not None
    assert torch.isfinite(value.grad).all()
    assert torch.equal(field.bank, parameter_before)


def test_explicit_batched_value_state_supports_fixed_grouped_query() -> None:
    field = arti.ARTILatentRecallField(
        4,
        4,
        routing="grouped",
        key_dim=3,
        query_mode="fixed",
        group_size=1,
        group_topk=1,
        value_composition="single",
        project_external=False,
    )
    value = torch.randn(2, 4, 4, requires_grad=True)
    hidden = torch.randn(2, 3, 4)
    mask = torch.ones(2, 3, dtype=torch.bool)

    context = field.read_context(hidden, mask, memory=value)
    context.square().mean().backward()

    assert context.shape == hidden.shape
    assert value.grad is not None


def test_stacked_recall_value_updater_rejects_behavior_mismatch() -> None:
    left = RecallValueUpdater(8, 3, workspace_dim=8, depth=1, recall_slots=4)
    right = RecallValueUpdater(
        8,
        3,
        workspace_dim=8,
        depth=1,
        recall_slots=4,
        dropout=0.2,
    )

    try:
        StackedRecallValueUpdater((left, right))
    except ValueError as exc:
        assert "behavior signature" in str(exc)
    else:
        raise AssertionError("behaviorally different Updaters were stacked")
