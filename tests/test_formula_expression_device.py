"""CPU correctness of the device executor over expanded Fabric grammars."""

import pytest
import torch

from arti import mechanisms as m
from arti._formula_device_dispatch import (
    FormulaDeviceDispatchLayout, FormulaDeviceNumericalDispatch, formula_device_dispatch_groups,
)
from arti._formula_device_frames import FormulaDeviceFrameKernel
from arti._formula_device_pools import FormulaDevicePoolLayout
from benchmarks._formula_expression_runtime import expression_runtime


def snapshot(mode, features=16):
    t = m.TensorType(("B", "D"), ("B", features), dtype="float32")
    gain_type = m.TensorType(("D",), (features,), dtype="float32")
    x = m.InputBinding("value", t)
    gain = m.BankBinding("gain", "arti/expression-device-test@1", "gain", gain_type)
    program = m.FormulaProgram.build(outputs=(m.scale(x, gain),))

    def prototype(name):
        return m.FormulaProgramTensorCandidateV4(m.FormulaProgramCandidateV3(
            name, program, input_slots={"value": "x"}, output_slots={program.outputs[0]: "answer"},
            operands={"gain": torch.ones(features)}, trainable_operands=("gain",),
        ))

    first, independent = prototype("scale"), prototype("independent")
    nodes = list(m.expand_candidate_bindings(first,
        slot_types={s: t for s in ("x", "z", "answer")}, input_choices={"value": ("x", "z")},
        output_choices={program.outputs[0]: ("answer",)}))
    if mode == "partial":
        nodes.append(independent)
    elif mode == "independent":
        nodes[1] = independent.with_bindings("independent", input_slots={"value": "z"},
                                            output_slots={program.outputs[0]: "answer"})
    query = m.FormulaProgramQueryV7(slot_ids=("x", "z", "answer"), candidates=nodes,
        terminal_slots={"answer": "answer"}, entry_candidates=tuple(n.candidate_id for n in nodes),
        continuations={}, max_steps=1)
    kernel = FormulaDeviceFrameKernel.from_query(query)
    dl = FormulaDevicePoolLayout.from_samples((torch.zeros(1, features),), 8)
    bl = FormulaDevicePoolLayout.from_samples((torch.zeros(features),), 1)
    dispatch = FormulaDeviceNumericalDispatch(query, kernel)
    dispatch.prepare_typed_pools_(dl, bl)
    state = kernel.initial_state(1, torch.tensor([[0, 1, -1]]))
    data, banks = dl.allocate("cpu"), bl.allocate("cpu")
    data[0][0], data[0][1] = torch.arange(features), -torch.arange(features)
    packet = FormulaDeviceDispatchLayout(formula_device_dispatch_groups(query)[0])(
        torch.arange(len(nodes)).unsqueeze(0))
    return query, dispatch, (state, packet, data, banks)


@pytest.mark.parametrize("mode,owners", [("shared", 1), ("partial", 2), ("independent", 2)])
def test_device_snapshot_deduplicates_only_shared_stores(mode, owners):
    query, dispatch, args = snapshot(mode)
    group, = dispatch.groups
    index = group.plan.binding_names.index("gain")
    table = getattr(group, f"operand_table_{index:04d}")
    assert table.shape == (owners, 16)
    assert hasattr(group, f"operand_row_{index:04d}") == (mode == "partial")
    prepared_bytes = sum(v.numel() * v.element_size() for name, v in group.named_buffers()
                         if name.startswith(("operand_table_", "operand_row_")))
    old_bytes = len(query.candidates) * 16 * 4
    assert prepared_bytes < old_bytes if mode != "independent" else prepared_bytes == old_bytes
    assert not any("operand_table_" in k or "operand_row_" in k for k in dispatch.state_dict())
    actual = dispatch(*args)
    expected = [args[2][0][0], args[2][0][1]]
    if mode == "partial":
        expected.append(args[2][0][0])
    assert actual.numeric_valid.all() and not actual.overflow
    torch.testing.assert_close(actual.output_values[0][:, 0], torch.stack(expected))


def test_shared_snapshot_refresh_after_parameter_replacement_and_fullgraph_compile(monkeypatch):
    import arti._formula_device_dispatch as implementation

    query, dispatch, args = snapshot("partial")
    args = (tuple(args[0]), tuple(args[1]), *args[2:])
    exported = torch.export.export(dispatch, args, strict=True).module()
    assert not any("_local_scalar_dense" in str(node.target) for node in exported.graph.nodes)
    compiled = torch.compile(exported, backend="aot_eager", fullgraph=True)
    expected = dispatch(*args)
    torch.testing.assert_close(compiled(*args).output_values[0], expected.output_values[0])
    group, = dispatch.groups
    pointers = {n: b.data_ptr() for n, b in group.named_buffers() if n.startswith("operand_")}
    store = query.candidates[0].candidate.operand_store
    before = store.tensor("gain")
    store.load_state_dict({name: torch.full_like(value, 3) for name, value in store.state_dict().items()}, assign=True)
    assert store.tensor("gain") is not before
    assert query.candidates[1].candidate.operand_store is store
    assert torch.equal(query.candidates[2].candidate.operand_store.tensor("gain"), torch.ones(16))
    torch.testing.assert_close(compiled(*args).output_values[0], expected.output_values[0])
    dispatch.refresh_operands_()
    assert pointers == {n: b.data_ptr() for n, b in group.named_buffers() if n.startswith("operand_")}

    def no_lookup(*args):
        raise AssertionError("dispatch must use the prepared operand snapshot")

    monkeypatch.setattr(implementation, "_operand", no_lookup)
    actual = compiled(*args)
    expected.output_values[0][:2].mul_(3)
    torch.testing.assert_close(actual.output_values[0], expected.output_values[0])
    assert actual.numeric_valid.all()


def test_shared_rows_preserve_reordered_duplicate_requests_and_nonfinite_validity():
    query, dispatch, args = snapshot("partial")
    shared = query.candidates[0].candidate.operand_store.tensor("gain")
    separate = query.candidates[2].candidate.operand_store.tensor("gain")
    with torch.no_grad():
        shared.fill_(2)
        separate.fill_(3)
    dispatch.refresh_operands_()
    packet = FormulaDeviceDispatchLayout(formula_device_dispatch_groups(query)[0])(
        torch.tensor([[2, 0, 2, 1]]))
    result = dispatch(args[0], packet, *args[2:])
    x, z = args[2][0][:2]
    torch.testing.assert_close(result.output_values[0][:, 0], torch.stack((3*x, 2*x, 3*x, 2*z)))
    assert result.numeric_valid.all()
    with torch.no_grad():
        shared.fill_(float("nan"))
    dispatch.refresh_operands_()
    result = dispatch(args[0], packet, *args[2:])
    assert result.numeric_valid.tolist() == [True, False, True, False]
    assert result.output_present[0][:, 0].tolist() == [True, False, True, False]


@pytest.mark.parametrize("mode,rows", [("shared", 1), ("partial", 3)])
def test_scalar_snapshot_does_not_spend_more_bytes_on_row_indices(mode, rows):
    _, dispatch, args = snapshot(mode, features=1)
    group, = dispatch.groups
    index = group.plan.binding_names.index("gain")
    assert getattr(group, f"operand_table_{index:04d}").shape == (rows, 1)
    assert not hasattr(group, f"operand_row_{index:04d}")
    assert dispatch(*args).numeric_valid.all()


def test_prepared_search_refresh_maps_unique_operand_flags_to_every_candidate():
    from benchmarks._federated_captured_search import _PreparedSearch

    original, _, args = snapshot("partial")
    query = m.FormulaProgramQueryV6(slot_ids=original.slot_ids, candidates=tuple(original.candidates),
        terminal_slots=original.terminal_slots, entry_candidates=original.entry_candidates,
        continuations={}, max_steps=1)
    sample = args[2][0][0]
    session = _PreparedSearch(query, sample, frame_kernel=FormulaDeviceFrameKernel.from_query(query),
        width=1, local_width=3, coverage=False, horizon=2, explore=False,
        value_samples=(sample,), bank_samples=(sample,))
    assert session.finite.tolist() == [True, True, True, True]
    with torch.no_grad():
        query.candidates[0].candidate.operand_store.tensor("gain").fill_(float("nan"))
    session.refresh()
    assert session.finite.tolist() == [False, False, True, True]


@pytest.mark.parametrize("change", ["shape", "dtype"])
def test_snapshot_refresh_cannot_resize_or_cast_a_prepared_table(change):
    query, dispatch, _ = snapshot("shared")
    group, = dispatch.groups
    index = group.plan.binding_names.index("gain")
    table = getattr(group, f"operand_table_{index:04d}")
    original, pointer = table.clone(), table.data_ptr()
    store = query.candidates[0].candidate.operand_store
    replacement = {name: (value[:8] if change == "shape" else value.double())
                   for name, value in store.state_dict().items()}
    if change == "shape":
        name, = dict(store.named_parameters())
        setattr(store, name, torch.nn.Parameter(replacement[name]))
    else:
        store.load_state_dict(replacement, assign=True)
    with pytest.raises(ValueError, match="rebuild the prepared dispatch"):
        dispatch.refresh_operands_()
    assert pointer == table.data_ptr()
    torch.testing.assert_close(table, original)


@pytest.mark.parametrize("width", [1, 3])
def test_expanded_atomic_grammar_completes_typed_device_search_and_matches_reference(width):
    from benchmarks.train_formula_expression_search import build_graph, data, reference
    from benchmarks._federated_recursive_search import search_cooperative_graphs, start_recursive_search

    query, (values, _) = build_graph(), data(29, positions=3)
    heads = 3
    native = search_cooperative_graphs((start_recursive_search(query, values),), product_slots=(),
                                      publish_slots=(), width=heads, beam_width=width)
    wave, args = expression_runtime(query, values, width, heads)
    dispatch = wave.wave.execution.dispatch
    snapshot_bytes = sum(v.numel() * v.element_size() for group in dispatch.groups
                         for name, v in group.named_buffers() if name.startswith(("operand_table_", "operand_row_")))
    old_bytes = sum(c.candidate.operand_store.tensor(binding.name).numel() * 4
                    for c in query.candidates for binding in c.candidate.program.bindings
                    if isinstance(binding, m.BankBinding))
    assert snapshot_bytes == sum(p.numel() * p.element_size() for p in query.parameters()) == 452
    assert old_bytes == 808
    with torch.no_grad():
        records = wave.forward_steps(*args, steps=6)
        for result in records:
            assert not result.requires_fallback and not result.dropped_products
    state = records[-1].state
    rows = state.frames.completed.nonzero().flatten().tolist()
    assert len(rows) == len(native.branches) > 0
    for row, branch in zip(rows, native.branches, strict=True):
        names = tuple(sorted(query.action_ids)[i] for i in state.routes[row, :state.lengths[row]].tolist())
        assert names == tuple(item["candidate_id"] for item in branch.route)
        torch.testing.assert_close(state.scores[row], branch.log_probability)
        handle = state.frames.value_handles[row, 0, query.slot_ids.index("answer")]
        output = wave.wave.execution.dispatch.data_layout.gather(args[4], handle.reshape(1), 2)[0]
        torch.testing.assert_close(output, branch.execution.outputs["answer"])
        torch.testing.assert_close(output, reference(query, values, branch))


def test_complete_expression_search_segment_is_fullgraph_and_keeps_round_records():
    from torch.utils._pytree import tree_leaves, tree_map
    from torch.fx.experimental.proxy_tensor import make_fx
    from benchmarks.train_formula_expression_search import build_graph, data

    query, (values, _) = build_graph(), data(29, positions=3)
    wave, args = expression_runtime(query, values, width=1, heads=2)
    def clone(tree):
        return tree_map(lambda value: value.detach().clone(), tree)
    expected_args, actual_args = clone(args), clone(args)
    expected = wave.forward_steps(*expected_args, steps=5)
    graph = make_fx(lambda *current: wave.forward_steps(*current, steps=5))(*clone(args))
    assert not any("_local_scalar_dense" in str(node.target) for node in graph.graph.nodes)
    compiled = torch.compile(graph, backend="aot_eager", fullgraph=True)
    actual = compiled(*actual_args)
    assert len(actual) == 5 and actual[-1].state.frames.completed.any()
    for left, right in zip(tree_leaves(actual), tree_leaves(expected), strict=True):
        torch.testing.assert_close(left, right)
    for left, right in zip(actual_args[4], expected_args[4], strict=True):
        torch.testing.assert_close(left, right)
    # The traced graph must read new inputs and refreshed Banks, not tracing values.
    changed = clone(args)
    changed[4][0][:2].mul_(-0.7)
    with torch.no_grad():
        query.candidates[0].candidate.operand_store.tensor("gain").mul_(2)
    wave.wave.execution.dispatch.refresh_operands_()
    fresh_args, reused_args = clone(changed), clone(changed)
    fresh = wave.forward_steps(*fresh_args, steps=5)
    reused = compiled(*reused_args)
    for left, right in zip(tree_leaves(reused), tree_leaves(fresh), strict=True):
        torch.testing.assert_close(left, right)
    for left, right in zip(tree_leaves(reused_args[4:6]), tree_leaves(fresh_args[4:6]), strict=True):
        torch.testing.assert_close(left, right)
    assert not torch.equal(actual_args[4][0], reused_args[4][0])
    # An unfinished horizon is still live, never fabricated into a STOP.
    unfinished = wave.forward_steps(*clone(args), steps=1)[0]
    assert unfinished.state.frames.active.any() and not unfinished.state.frames.completed.any()
    assert actual[-1].numeric_attempts == 0


def test_device_expression_records_replay_full_endpoint_risk_and_bank_gradients():
    from benchmarks.train_formula_expression_search import build_graph, data
    from benchmarks._federated_device_product_replay import decode_cooperative_device_products
    from benchmarks._federated_product_replay import replay_cooperative_dependencies_many
    from benchmarks._federated_recursive_search import search_cooperative_graphs, start_recursive_search

    query, (inputs, target) = build_graph(), data(29, positions=3)
    values = {name: value.requires_grad_() for name, value in inputs.items()}
    with torch.no_grad():
        native = search_cooperative_graphs((start_recursive_search(query, values),), product_slots=(),
                                          publish_slots=(), width=3, beam_width=3)
        wave, args = expression_runtime(query, values, width=3, heads=3)
        records = wave.forward_steps(*args, steps=6)
    tape, endpoints = decode_cooperative_device_products(query, records,
        initial_states=(query.initial_bank_state(),), input_slots=("x", "z"), include_decisions=True)
    # Reuse the numerical pool with different inputs before replaying the decoded tape.
    args[4][0][:2].neg_()
    wave.forward_steps(*args, steps=6)
    actual = replay_cooperative_dependencies_many(query, values, tape=tape, endpoints=endpoints,
                                                 score_decisions=True)
    expected = replay_cooperative_dependencies_many(query, values, tape=native.dependency_tape,
        endpoints=tuple(b.dependency_endpoint for b in native.branches), score_decisions=True)
    assert len(actual) == len(expected) == 3
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left.outputs["answer"], right.outputs["answer"])
        torch.testing.assert_close(left.decision_energy, right.decision_energy)

    def risk(runs):
        energy = torch.stack([r.decision_energy for r in runs])
        loss = torch.stack([(r.outputs["answer"] - target).square().mean() for r in runs])
        return (energy.softmax(0) * loss).sum()

    parameters = (*values.values(), *query.parameters())
    wanted = torch.autograd.grad(risk(expected), parameters, allow_unused=True)
    gradients = torch.autograd.grad(risk(actual), parameters, allow_unused=True)
    assert any(g is not None and g.abs().sum() > 0 for g in gradients[2:])
    for left, right in zip(gradients, wanted, strict=True):
        if right is None:
            assert left is None
        else:
            torch.testing.assert_close(left, right)


def test_search_segment_preserves_independent_events_and_stopped_tail():
    from torch.utils._pytree import tree_leaves, tree_map
    from benchmarks.train_formula_expression_search import build_graph, data

    query, (values, _) = build_graph(), data(29, positions=3)
    wave, original = expression_runtime(query, values, width=1, heads=2)
    left = tree_map(lambda value: value.detach().clone(), original)
    right = tree_map(lambda value: value.detach().clone(), original)
    right[4][0][0].mul_(-0.5)
    right[4][0][1].add_(1)
    packed = (*tree_map(lambda a, b: torch.stack((a, b)), left[:8], right[:8]), original[8])
    bias = torch.zeros(7, 2, wave.width, wave.kernel.candidate_count)
    actual = wave.forward_steps(*packed, steps=7, batched=True, selection_bias=bias)
    expected = [wave.forward_steps(*args, steps=7, selection_bias=bias[:, i]) for i, args in enumerate((left, right))]
    for i, records in enumerate(expected):
        for a, b in zip(tree_leaves(actual), tree_leaves(records), strict=True):
            torch.testing.assert_close(a[i], b)
    assert actual[-1].state.frames.completed.any(1).all()
    assert not actual[-1].state.frames.active.any()
    for name in ("occurrence_cursor", "data_cursor", "bank_cursor"):
        torch.testing.assert_close(getattr(actual[-1], name), getattr(actual[-2], name))
    assert not actual[-1].numeric_attempts.any()
