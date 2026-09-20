from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from arti import mechanisms as m
from benchmarks._federated_episode_beam import event_routes, replay_episode_path, replay_episode_paths, search_episode_beam
from benchmarks._federated_recursive_search import (
    final_graph_loss, replay_recursive_graph, replay_recursive_query_scores,
    search_recursive_graphs, start_recursive_search,
)


def _type():
    return m.TensorType(("B", "D"), ("B", 2), dtype="floating", domain="activation")


def _producer(name, gain=1.0, *, owner=None):
    x = m.InputBinding("x", _type())
    w = m.BankBinding("weight", "arti/recursive-search-test@1", "weight", _type())
    g = m.BankBinding("gain", "arti/recursive-search-test@1", "gain", _type())
    program = m.FormulaProgram.build(outputs=(m.scale(m.scale(x, w), g), m.add(x, x)))
    return m.FormulaProgramTensorCandidateV4(m.FormulaProgramCandidateV3(
        name, program, input_slots={"x": "x"},
        output_slots=dict(zip(program.outputs, ("owned", "data"), strict=True)),
        operands={"weight": torch.ones(1, 2), "gain": torch.full((1, 2), gain)},
        trainable_operands=("gain",),
    ), plastic_bank_slot="weight", bank_owner_id=name if owner is None else owner)


def _zero_policy(query):
    with torch.no_grad():
        query.network[-1].weight.zero_()
        query.network[-1].bias.zero_()
    return query


def _choice_child(prefix, gains):
    return _zero_policy(m.FormulaProgramQueryV5(
        slot_ids=("x", "owned", "data"),
        candidates=tuple(_producer(f"{prefix}-{i}", gain) for i, gain in enumerate(gains)),
        terminal_slots={"answer": "owned", "data": "data"}, max_steps=1, hidden_dim=8,
    ))


def _parent(children):
    calls = tuple(m.FormulaProgramCallCandidateV1(
        f"call-{i}", child, input_slots={"x": "x"}, output_slots={"answer": "answer", "data": "data"},
    ) for i, child in enumerate(children))
    return _zero_policy(m.FormulaProgramQueryV5(
        slot_ids=("x", "answer", "data"), candidates=calls,
        terminal_slots={"answer": "answer", "data": "data"}, max_steps=1, hidden_dim=8,
    ))


def _model():
    return _parent((_choice_child("a", (1.0, 2.0)), _choice_child("b", (3.0, 4.0))))


def test_global_beam_searches_children_and_returns_all_four_whole_graphs():
    query = _model()
    result = search_recursive_graphs((start_recursive_search(query, {"x": torch.ones(1, 2)}),),
                                     width=4, beam_width=16, preserve_effect_coverage=False)
    assert len(result.branches) == 4
    assert sorted(branch.execution.outputs["answer"][0, 0].item() for branch in result.branches) == [1, 2, 3, 4]
    torch.testing.assert_close(torch.stack([branch.log_probability.exp() for branch in result.branches]), torch.full((4,), 0.25))
    for branch in result.branches:
        assert branch.execution.trace.stopped and branch.frames == ()
        assert set(branch.execution.outputs) == {"answer", "data"}
        assert [row["kind"] for row in branch.route] == ["call", "ordinary", "stop", "stop"]
        assert branch.route[2]["query_path"] != () and branch.route[3]["query_path"] == ()
        assert branch.execution.trace.total_dispatches == 2
    assert result.entered_calls == 2 and result.executed_expansions == 4
    assert result.returned_calls == result.root_stops == 4
    assert result.maximum_active <= 16 and result.maximum_completed <= 16 and result.maximum_call_depth == 1


def test_exploration_is_repeatable_changes_candidates_and_preserves_original_scores():
    query = _parent((_choice_child("choice", tuple(float(i) for i in range(8))),))
    values = {"x": torch.ones(1, 2)}
    initial = query.initial_bank_state()
    def run(seed):
        return search_recursive_graphs(
            (start_recursive_search(query, values),), width=2, beam_width=2,
            preserve_effect_coverage=False, record_query_choices=True,
            exploration_generator=None if seed is None else torch.Generator().manual_seed(seed),
        )
    baseline = run(None)
    routes = set()
    global_rng = torch.random.get_rng_state().clone()
    for seed in range(5):
        result, repeated = run(seed), run(seed)
        assert [b.route for b in result.branches] == [b.route for b in repeated.branches]
        routes.add(tuple(tuple(row["candidate_id"] for row in b.route) for b in result.branches))
        assert result.maximum_active <= 2 and result.maximum_completed <= 2
        for branch in result.branches:
            replay = replay_recursive_graph(query, values, branch.route, use_recorded_choices=True)
            torch.testing.assert_close(branch.log_probability, replay.log_probability)
            assert branch.selection_priority is not None
            parameters = tuple(p for p in query.parameters() if p.requires_grad)
            def loss(b):
                return b.execution.outputs["answer"].square().mean() + b.log_probability
            actual = torch.autograd.grad(loss(branch), parameters, allow_unused=True, retain_graph=True)
            expected = torch.autograd.grad(loss(replay), parameters, allow_unused=True)
            for a, b in zip(actual, expected, strict=True):
                if a is None or b is None:
                    # Batched execution retains zero-gradient sibling operands;
                    # serial path replay never instantiates those siblings.
                    other = b if a is None else a
                    assert other is None or not bool(other.any())
                else:
                    torch.testing.assert_close(a, b)
        assert result.winner.log_probability == max(b.log_probability for b in result.branches)
    assert len(routes) > 1
    assert torch.equal(global_rng, torch.random.get_rng_state())
    assert [b.route for b in baseline.branches] == [b.route for b in run(None).branches]
    for before, after in zip(initial.values, query.initial_bank_state().values, strict=True):
        torch.testing.assert_close(before, after)


def test_exploration_refill_reuses_one_priority_without_redrawing():
    from benchmarks._federated_recursive_search import _rank_expansions, _selection_prune
    from benchmarks.train_federated_branch_visible_federation import _prune

    query = _choice_child("a", (1., 2., 3., 4.))
    generator = torch.Generator().manual_seed(42)
    expansions = _rank_expansions(
        (start_recursive_search(query, {"x": torch.ones(1, 2)}),), width=4,
        preserve_effect_coverage=False, rejections=[], exploration_generator=generator,
    )
    rng_after_draw = generator.get_state().clone()
    ordered = _selection_prune(expansions, beam_width=4, prune=_prune)
    # Simulate a numerical rejection and refill from the original pool.
    survivors = tuple(item for item in expansions if item is not ordered[0])
    refilled = _selection_prune(survivors, beam_width=2, prune=_prune)
    assert all(a is b for a, b in zip(refilled, ordered[1:3], strict=True))
    assert torch.equal(rng_after_draw, generator.get_state())


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("parent_dtype", (torch.float32, torch.float64))
def test_grouped_exploration_preserves_ragged_draws_scores_and_gradients(monkeypatch, device, dtype, parent_dtype):
    import benchmarks._federated_recursive_search as search
    from torch.utils._python_dispatch import TorchDispatchMode

    query = _choice_child("choice", (1., 2., 3., 4.)).to(device)
    parents = torch.tensor([-0.1, -3.0, -1e7, -0.3], device=device, dtype=parent_dtype, requires_grad=True)
    scores = torch.tensor([-0.3, -0.3, -2.1, -0.7, -0.2, -1.5, -1.2],
                          device=device, dtype=dtype, requires_grad=True)
    branches = tuple(replace(start_recursive_search(query, {"x": torch.ones(1, 2, device=device)}),
                             log_probability=parent) for parent in parents)
    ranked_rows = (
        tuple(zip(query.candidates[:3], scores[:3], strict=True)), (),
        ((None, scores[3]), (query.candidates[0], scores[4])),
        tuple(zip(query.candidates[1:3], scores[5:], strict=True)),
    )
    seen_widths = []
    def ranked_many(*args, **kwargs):
        seen_widths.append(kwargs["width"])
        return ranked_rows
    monkeypatch.setattr(search, "_ranked_candidates_many", ranked_many)

    reference_rng = torch.Generator().manual_seed(937)
    expected = []
    for branch, ranked in zip(branches, ranked_rows, strict=True):
        if not ranked:
            continue
        uniforms = torch.rand(len(ranked), generator=reference_rng, dtype=torch.float64)
        noise = -torch.log(-torch.log(uniforms.clamp_min(torch.finfo(torch.float64).tiny)))
        priorities = (torch.stack([s for _, s in ranked]) + branch.log_probability).detach().cpu() + noise
        order = sorted(range(len(ranked)), key=lambda i: float(priorities[i]), reverse=True)[:2]
        expected.extend((branch, ranked[i][0], branch.log_probability + ranked[i][1], priorities[i]) for i in order)

    copies, additions = [], []
    class HostCopies(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if (func is torch.ops.aten._to_copy.default and args[0].device.type == "cuda"
                    and kwargs.get("device") == torch.device("cpu")):
                copies.append(args[0].numel())
            if (func is torch.ops.aten.add.Tensor and args[0].device.type == device
                    and len(args) > 1 and isinstance(args[1], torch.Tensor) and args[1].ndim == 0):
                additions.append(args[0].numel())
            return func(*args, **kwargs)

    generator = torch.Generator().manual_seed(937)
    with HostCopies():
        actual = search._rank_expansions(branches, width=2, preserve_effect_coverage=False,
                                         rejections=[], exploration_generator=generator)
    assert copies == ([7] if device == "cuda" else [])
    assert additions == [3, 2, 2] + ([1] * 7 if parent_dtype != dtype else [])
    assert seen_widths == [len(query.action_ids)]
    assert torch.equal(generator.get_state(), reference_rng.get_state())
    for expansion, (parent, candidate, probability, priority) in zip(actual, expected, strict=True):
        assert expansion.parent is parent and expansion.candidate is candidate
        torch.testing.assert_close(expansion.log_probability, probability, rtol=0, atol=0)
        torch.testing.assert_close(expansion.selection_priority, priority, rtol=0, atol=0)
    actual_grads = torch.autograd.grad(sum(x.log_probability for x in actual), (parents, scores), retain_graph=True)
    expected_grads = torch.autograd.grad(sum(row[2] for row in expected), (parents, scores))
    for a, b in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_selection_prune_preserves_mixed_priorities_dtype_and_stable_ties(device):
    from benchmarks._federated_recursive_search import _host_score_rows, _selection_prune, _SelectionView
    from benchmarks.train_federated_branch_visible_federation import _prune

    rows = tuple(torch.tensor([1., 3., 5., 7.], device=device, dtype=dtype, requires_grad=True)[::2]
                 for dtype in (torch.float32, torch.float64, torch.float32))
    host = _host_score_rows((*rows, torch.tensor([])))
    assert _host_score_rows(()) == ()
    for before, after in zip((*rows, torch.tensor([])), host, strict=True):
        assert not after.requires_grad and after.device.type == "cpu"
        torch.testing.assert_close(after, before.detach().cpu(), rtol=0, atol=0)
    branches = tuple(SimpleNamespace(
        log_probability=row[0], selection_priority=torch.tensor(1., dtype=torch.float64) if i == 1 else None,
        route=({"candidate_id": name},),
    ) for i, (row, name) in enumerate(zip(rows, ("zeta", "alpha", "mu"), strict=True)))
    views = tuple(_SelectionView(b, b.log_probability.detach().cpu() if b.selection_priority is None
                                 else b.selection_priority) for b in branches)
    expected = tuple(view.branch for view in _prune(views, beam_width=2))
    actual = _selection_prune(branches, beam_width=2, prune=_prune)
    assert all(a is b for a, b in zip(actual, expected, strict=True))
    assert _selection_prune((), beam_width=2, prune=_prune) == ()


def test_final_loss_credit_matches_exact_parent_child_enumeration():
    query = _model()
    reference = copy.deepcopy(query)
    x = torch.tensor([[1.0, -0.5]])
    target = torch.full((1, 2), 3.0)
    result = search_recursive_graphs((start_recursive_search(query, {"x": x}),), width=4, beam_width=16,
                                     preserve_effect_coverage=False)
    losses = torch.stack([(branch.execution.outputs["answer"] - target).square().mean() for branch in result.branches])
    loss = final_graph_loss(result, losses)
    loss.backward()
    expected = x.new_zeros(())
    root_weights = reference.network[-1].bias[:2].softmax(0)
    for i, call in enumerate(reference.candidates):
        child_weights = call.child.network[-1].bias[:2].softmax(0)
        for j, producer in enumerate(call.child.candidates):
            value = x * producer.candidate.operand_store.tensor("gain")
            expected = expected + root_weights[i] * child_weights[j] * (value - target).square().mean()
    expected.backward()
    torch.testing.assert_close(loss, expected)
    torch.testing.assert_close(query.network[-1].bias.grad, reference.network[-1].bias.grad)
    assert query.network[-1].bias.grad.abs().sum() > 0
    for actual_call, reference_call in zip(query.candidates, reference.candidates, strict=True):
        torch.testing.assert_close(actual_call.child.network[-1].bias.grad, reference_call.child.network[-1].bias.grad)
        assert actual_call.child.network[-1].bias.grad.abs().sum() > 0
        for actual, ref in zip(actual_call.child.candidates, reference_call.child.candidates, strict=True):
            torch.testing.assert_close(actual.candidate.operand_store.tensor("gain").grad,
                                       ref.candidate.operand_store.tensor("gain").grad)
            assert not actual.bank_owner.value.requires_grad


def test_query_only_replay_matches_routing_gradients_without_operand_graph():
    query = _parent((_writer_child(),))
    x = torch.ones(1, 2, requires_grad=True)
    state = query.initial_bank_state()
    result = search_recursive_graphs((start_recursive_search(query, {"x": x}, bank_state=state),),
                                     width=4, beam_width=16, preserve_effect_coverage=False)
    route = result.winner.route
    full = replay_recursive_graph(query, {"x": x}, route, bank_state=state)
    scores = replay_recursive_query_scores(query, {"x": x}, route, bank_state=state)
    parameters = tuple(p for module in query.modules() if isinstance(module, m.FormulaProgramQueryV5)
                       for p in module.network.parameters())
    expected = torch.autograd.grad(full.log_probability, parameters)
    actual = torch.autograd.grad(scores.log_probability, parameters)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)
    torch.testing.assert_close(scores.log_probability, full.log_probability)
    for name in full.execution.outputs:
        torch.testing.assert_close(scores.execution.outputs[name], full.execution.outputs[name])
        assert not scores.execution.outputs[name].requires_grad
    assert all(not value.requires_grad for value in scores.execution.bank_state.values)


@pytest.mark.parametrize("logit_offset", (0.0, 1e7))
def test_same_round_recorded_choices_preserve_loss_and_all_parameter_gradients(logit_offset):
    query = _parent((_writer_child(),))
    with torch.no_grad():
        for module in query.modules():
            if isinstance(module, m.FormulaProgramQueryV5):
                module.network[-1].bias.add_(logit_offset)
    x = torch.ones(1, 2)
    state = query.initial_bank_state()
    with torch.no_grad():
        result = search_recursive_graphs((start_recursive_search(query, {"x": x}, bank_state=state),),
                                         width=4, beam_width=16, preserve_effect_coverage=False,
                                         record_query_choices=True)
    parameters = tuple(p for p in query.parameters() if p.requires_grad)
    for branch in result.branches:
        full = replay_recursive_graph(query, {"x": x}, branch.route, bank_state=state)
        fast = replay_recursive_graph(query, {"x": x}, branch.route, bank_state=state,
                                       use_recorded_choices=True)
        def loss(row):
            return row.execution.outputs["answer"].square().mean() + row.log_probability
        torch.testing.assert_close(loss(full), loss(fast))
        expected = torch.autograd.grad(loss(full), parameters, allow_unused=True)
        actual = torch.autograd.grad(loss(fast), parameters, allow_unused=True)
        for a, b in zip(actual, expected):
            if a is None or b is None:
                assert a is b
            else:
                torch.testing.assert_close(a, b)
        assert full.execution.bank_state.revisions == fast.execution.bank_state.revisions


def test_recorded_choice_mode_is_explicit_and_requires_records():
    query = _model()
    values = {"x": torch.ones(1, 2)}
    result = search_recursive_graphs((start_recursive_search(query, values),), width=4, beam_width=16)
    with pytest.raises(ValueError, match="recorded eligibility"):
        replay_recursive_graph(query, values, result.winner.route, use_recorded_choices=True)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_width_one_matches_full_native_child_execution(device):
    query = _model().to(device)
    with torch.no_grad():
        query.network[-1].bias[1] = 1.0
        query.candidates[1].child.network[-1].bias[1] = 1.0
    x = torch.ones(1, 2, device=device)
    with torch.no_grad():
        expected = query({"x": x})
        search = search_recursive_graphs((start_recursive_search(query, {"x": x}),), width=1, beam_width=1,
                                        preserve_effect_coverage=False)
    actual = search.winner.execution
    assert actual.trace == expected.trace
    for head in actual.outputs:
        torch.testing.assert_close(actual.outputs[head], expected.outputs[head])
    assert actual.bank_state.revisions == expected.bank_state.revisions


def _writer_child():
    x = m.InputBinding("x", _type())
    rate = m.BankBinding("rate", "arti/recursive-search-test@1", "rate", _type())
    zero = m.BankBinding("zero", "arti/recursive-search-test@1", "zero", _type())
    program = m.FormulaProgram.build(outputs=(m.neural_plasticity(x, m.scale(x, rate), m.scale(x, zero)),))
    effects = tuple(m.FormulaProgramEffectCandidateV3(
        f"write-{i}", m.FormulaEffectProgramV2(program, data_input_name="x", state_type=_type()),
        input_slot="owned", output_slot="tail",
        operands={"rate": torch.full((1, 2), value), "zero": torch.zeros(1, 2)},
        trainable_operands=("rate",), execution_count=torch.tensor(2.0),
        trainable_execution_count=True, max_executions=4,
    ) for i, value in enumerate((0.1, 0.3)))
    return _zero_policy(m.FormulaProgramQueryV5(
        slot_ids=("x", "owned", "data", "tail"), candidates=(_producer("memory"), *effects),
        terminal_slots={"answer": "tail", "data": "data"}, max_steps=2, hidden_dim=8,
    ))


def test_future_task_loss_trains_support_writers_through_all_nested_decisions():
    query = _parent((_writer_child(),))
    x = torch.ones(1, 2)
    support = search_recursive_graphs((start_recursive_search(query, {"x": x}),), width=4, beam_width=16,
                                      preserve_effect_coverage=False, record_effect_metrics=True)
    assert len(support.branches) == 2
    futures = tuple(start_recursive_search(
        query, {"x": x}, bank_state=branch.execution.bank_state,
        log_probability=branch.log_probability, route=branch.route,
    ) for branch in support.branches)
    result = search_recursive_graphs(futures, width=4, beam_width=16, preserve_effect_coverage=False)
    losses = torch.stack([(branch.execution.outputs["answer"] - 1.6).square().mean() for branch in result.branches])
    assert len(result.branches) == 4
    final_graph_loss(result, losses).backward()
    child = query.candidates[0].child
    assert child.network[-1].bias.grad.abs().sum() > 0
    assert child.candidates[1].operand_store.tensor("rate").grad.abs().sum() > 0
    count_gradient = child.candidates[1].execution_count.grad
    assert count_gradient is not None and torch.isfinite(count_gradient).all() and count_gradient.abs().sum() > 0
    assert query.initial_bank_state().revisions == (0,)
    for branch in support.branches:
        effect = next(row for row in branch.route if row["kind"] == "effect")
        assert effect["data_identity"] is True and effect["execution_count"] == 2
        assert effect["predecessor_execution_id"] == "call-0/memory"
    query.requires_grad_(False)
    with torch.no_grad():
        frozen = search_recursive_graphs((start_recursive_search(query, {"x": x}),), width=4, beam_width=16,
                                        preserve_effect_coverage=False)
    query.commit_(frozen.winner.execution)
    assert query.initial_bank_state().revisions == (1,)
    assert not query.owner_states[0].value.requires_grad


def test_episode_beam_requeries_each_own_bank_and_replays_complete_events():
    parent = _parent((_writer_child(),))
    query = _zero_policy(m.FormulaProgramQueryV5(
        slot_ids=parent.slot_ids, candidates=tuple(parent.candidates),
        terminal_slots={"output": "answer"}, max_steps=1, hidden_dim=8,
    ))
    inputs = ({"x": torch.ones(1, 2)}, {"x": torch.full((1, 2), 0.5)})
    beam = search_episode_beam(query, inputs, width=16)
    assert [row["retained"] for row in beam.event_searches] == [2, 4]
    assert all(row["maximum_frontier"] <= 32 for row in beam.event_searches)
    assert query.initial_bank_state().revisions == (0,)
    paths = []
    for branch in beam.result.branches:
        chunks = event_routes(branch.route)
        assert len(chunks) == 2
        assert all(sum(row["kind"] == "stop" for row in chunk) == 2 for chunk in chunks)
        replay = replay_episode_path(query, inputs, branch.route)
        torch.testing.assert_close(replay.log_probability, branch.log_probability)
        torch.testing.assert_close(replay.output, branch.execution.outputs["output"])
        torch.testing.assert_close(replay.events[-1].execution.bank_state.values, branch.execution.bank_state.values)
        assert replay.events[0].execution.bank_state.revisions == (1,)
        assert replay.events[1].execution.bank_state.revisions == (2,)
        paths.append(replay)
    combined, event_count = replay_episode_paths(query, inputs, [branch.route for branch in beam.result.branches])
    assert event_count == 6  # Two first-event choices, each with two continuations.
    assert len({id(row.events[0]) for row in combined}) == 2
    parameters = tuple(p for p in query.parameters() if p.requires_grad)
    def objective(rows):
        logits = torch.stack([row.log_probability for row in rows])
        losses = torch.stack([(row.output - 0.8).square().mean() for row in rows])
        return (logits.softmax(0) * losses).sum()
    expected = torch.autograd.grad(objective(paths), parameters, retain_graph=True, allow_unused=True)
    actual = torch.autograd.grad(objective(combined), parameters, allow_unused=True)
    for left, right in zip(expected, actual):
        if left is None or right is None:
            assert left is right
        else:
            torch.testing.assert_close(left, right)
    # Final data depends on the preceding event's write, not the last identity effect.
    assert len({tuple(row.output.detach().flatten().tolist()) for row in paths}) == 2
    losses = torch.stack([(row.output - 0.8).square().mean() for row in paths])
    scores = torch.stack([row.log_probability for row in paths])
    (scores.softmax(0) * losses).sum().backward()
    child = query.candidates[0].child
    for effect in child.candidates[1:]:
        assert effect.operand_store.tensor("rate").grad.abs().sum() > 0
        assert effect.execution_count.grad.abs().sum() > 0
    assert child.network[-1].bias.grad.abs().sum() > 0
    assert query.initial_bank_state().revisions == (0,)


def test_episode_replay_requires_one_root_stop_per_event():
    with pytest.raises(ValueError, match="root STOP"):
        event_routes(({"kind": "stop", "query_path": ("child",)},))
    with pytest.raises(ValueError, match="one complete root trace"):
        replay_episode_path(_model(), ({"x": torch.ones(1, 2)},), ())
    with pytest.raises(ValueError, match="at least one event"):
        search_episode_beam(_model(), ())


def test_write_causality_distinguishes_future_read_from_last_identity_effect():
    from benchmarks.probe_federated_interaction_causality import later_owner_reads, write_gradients

    parent = _parent((_writer_child(),))
    query = _zero_policy(m.FormulaProgramQueryV5(
        slot_ids=parent.slot_ids, candidates=tuple(parent.candidates),
        terminal_slots={"output": "answer"}, max_steps=1, hidden_dim=8,
    ))
    inputs = ({"x": torch.ones(1, 2)}, {"x": torch.ones(1, 2)})
    beam = search_episode_beam(query, inputs, width=16)
    branch = beam.result.branches[0]
    replay = replay_episode_path(query, inputs, branch.route)
    reads = later_owner_reads(branch.route)
    assert len(reads) == 2
    assert reads[0]["later_owner_reads"] and not reads[1]["later_owner_reads"]
    gradients = write_gradients((replay.output - 2).square().mean(), replay.events)
    assert [row["gradient"] for row in gradients] == ["nonzero", "absent"]
    zero = write_gradients(replay.output.sum() * 0, replay.events)
    assert [row["gradient"] for row in zero] == ["zero", "absent"]


def test_nested_depth_does_not_multiply_the_global_frontier():
    query = _parent((_model(), _parent((_choice_child("c", (5.0, 6.0)),))))
    result = search_recursive_graphs((start_recursive_search(query, {"x": torch.ones(1, 2)}),),
                                     width=4, beam_width=3, preserve_effect_coverage=False)
    assert result.maximum_active <= 3 and result.maximum_completed <= 3
    assert result.maximum_call_depth == 2
    assert all(branch.execution is not None for branch in result.branches)


def test_frontier_counts_simultaneously_live_and_completed_graphs():
    child = _writer_child()
    query = _zero_policy(m.FormulaProgramQueryV5(
        slot_ids=child.slot_ids, candidates=child.candidates,
        terminal_slots={"answer": "owned", "data": "data"},
        min_steps=1, max_steps=2, hidden_dim=8,
    ))
    result = search_recursive_graphs((start_recursive_search(query, {"x": torch.ones(1, 2)}),),
                                     width=3, beam_width=2, preserve_effect_coverage=False)
    assert result.maximum_active == result.maximum_completed == 2
    assert result.maximum_frontier == 3


def test_terminal_readout_rejection_refills_without_reexecuting_successful_candidates():
    query = _model()
    result = search_recursive_graphs((start_recursive_search(query, {"x": torch.ones(1, 2)}),),
                                     width=4, beam_width=4, preserve_effect_coverage=False,
                                     terminal_admission=lambda branch: branch.execution.outputs["answer"][0, 0].item() >= 3)
    assert len(result.branches) == 2
    assert result.executed_expansions == 4
    assert len(result.numerical_rejections) == 2
    assert sorted(branch.execution.outputs["answer"][0, 0].item() for branch in result.branches) == [3, 4]


@pytest.mark.parametrize("seed", (None, 17))
def test_numeric_rejection_refills_without_reexecuting_successful_candidates(seed):
    query = _parent((_choice_child("a", (1.0, 3e38, 3.0)),))
    with torch.no_grad():
        query.candidates[0].child.network[-1].bias[1] = 100
    result = search_recursive_graphs((start_recursive_search(query, {"x": torch.full((1, 2), 2.0)}),),
                                     width=3, beam_width=2, preserve_effect_coverage=False,
                                     exploration_generator=None if seed is None else torch.Generator().manual_seed(seed))
    assert sorted(branch.execution.outputs["answer"][0, 0].item() for branch in result.branches) == [2, 6]
    assert result.executed_expansions == 3
    assert len(result.numerical_rejections) == 1
    assert result.numerical_rejections[0]["code"] == "FF2_NONFINITE"


def test_resumed_search_counts_existing_stack_depth():
    from benchmarks._federated_recursive_search import _advance, _Expansion

    query = _parent((_parent((_choice_child("a", (2.0,)),)),))
    branch = start_recursive_search(query, {"x": torch.ones(1, 2)})
    for _ in range(3):
        frame = branch.frames[-1]
        candidate = frame.query.candidates[0]
        numeric = None if isinstance(candidate, m.FormulaProgramCallCandidateV1) else candidate(frame.arena)
        branch = _advance(_Expansion(branch, candidate, branch.log_probability, branch.route), numeric,
                          record_effect_metrics=False)
    assert len(branch.frames) == 3
    result = search_recursive_graphs((branch,), width=1, beam_width=1, preserve_effect_coverage=False)
    assert result.maximum_call_depth == 2
    assert result.executed_expansions == 0 and result.returned_calls == 2


def test_search_and_replay_reject_wrapper_hooks_instead_of_silently_skipping_them():
    query = _parent((_choice_child("a", (2.0,)),))
    x = torch.ones(1, 2)
    original = search_recursive_graphs((start_recursive_search(query, {"x": x}),), width=1, beam_width=1,
                                       preserve_effect_coverage=False)
    child = query.candidates[0].child
    handle = child.register_forward_pre_hook(lambda module, args: ({"x": args[0]["x"] * 2},))
    try:
        torch.testing.assert_close(query({"x": x}).outputs["answer"], torch.full((1, 2), 4.0))
        with pytest.raises(NotImplementedError, match="module hooks"):
            search_recursive_graphs((start_recursive_search(query, {"x": x}),), width=1, beam_width=1,
                                    preserve_effect_coverage=False)
        with pytest.raises(NotImplementedError, match="module hooks"):
            replay_recursive_graph(query, {"x": x}, original.winner.route)
    finally:
        handle.remove()


def test_recorded_child_routes_replay_exact_outputs_states_scores_and_gradients():
    query = _parent((_writer_child(),))
    x = torch.tensor([[1.0, -0.5]], requires_grad=True)
    result = search_recursive_graphs((start_recursive_search(query, {"x": x}),), width=4, beam_width=16,
                                     preserve_effect_coverage=False)
    rate = query.candidates[0].child.candidates[1].operand_store.tensor("rate")
    for original in result.branches:
        replay = replay_recursive_graph(query, {"x": x}, original.route)
        assert replay.execution.trace == original.execution.trace
        torch.testing.assert_close(replay.log_probability, original.log_probability)
        torch.testing.assert_close(replay.execution.bank_state.values, original.execution.bank_state.values)
        for name in original.execution.outputs:
            torch.testing.assert_close(replay.execution.outputs[name], original.execution.outputs[name])
        original_loss = original.execution.bank_state.values[0].square().sum()
        replay_loss = replay.execution.bank_state.values[0].square().sum()
        expected = torch.autograd.grad(original_loss, (x, rate), retain_graph=True, allow_unused=True)
        actual = torch.autograd.grad(replay_loss, (x, rate), retain_graph=True, allow_unused=True)
        for left, right in zip(actual, expected, strict=True):
            if left is None or right is None:
                assert left is right
            else:
                torch.testing.assert_close(left, right)


def test_qwen_credit_parameter_collection_includes_shared_children_once():
    from benchmarks.train_qwen_federated_autonomous_federation import (
        _execution_count_parameters, _routing_parameters,
    )

    child = _writer_child()
    root = _parent((child, child))
    wrapper = SimpleNamespace(query=root)
    routing = _routing_parameters(wrapper)
    assert len(routing) == len({id(parameter) for parameter in routing})
    expected = {id(parameter) for query in (root, child) for parameter in query.network.parameters()}
    assert {id(parameter) for parameter in routing} == expected
    counts = _execution_count_parameters(wrapper)
    assert len(counts) == 2
    assert {id(count) for count in counts} == {id(candidate.execution_count) for candidate in child.candidates[1:]}


def test_existing_large_federation_mount_preserves_parameters_counts_and_adam_state():
    from benchmarks._federated_search_space_migration import named_query_view
    from benchmarks._federated_v4_federation import build_autonomous_effect_federation

    federation = build_autonomous_effect_federation(
        hidden_dim=8, rank=4, seed=220904, device=torch.device("cpu"), plastic_branches=16,
        min_operations=1, max_operations=2, max_effect_operations=2,
    )
    original = federation.query
    optimizer = torch.optim.Adam(original.parameters(), lr=1e-3)
    parameter = next(original.network.parameters())
    optimizer.state[parameter] = {"step": torch.tensor(7.0), "exp_avg": torch.ones_like(parameter),
                                  "exp_avg_sq": torch.full_like(parameter, 2.0)}
    state = optimizer.state[parameter]
    rng = torch.random.get_rng_state().clone()
    child = named_query_view(original)
    assert torch.equal(rng, torch.random.get_rng_state())
    assert {id(p) for p in original.parameters()} == {id(p) for p in child.parameters()}
    assert child.network is original.network and optimizer.state[parameter] is state
    assert child.owner_states[0] is original.owner_states[0]
    assert len({candidate.atom_ref for candidate in federation.effects}) == 6
    assert all(candidate.hard_execution_count() == 2 for candidate in federation.effects)
    before, after = original.state_dict(), child.state_dict()
    assert before.keys() == after.keys()
    for name in before:
        assert torch.equal(before[name], after[name])
    call = m.FormulaProgramCallCandidateV1("federation", child, input_slots={"x": "x"}, output_slots={"output": "out"})
    root = m.FormulaProgramQueryV5(slot_ids=("x", "out"), candidates=(call,), terminal_slots={"answer": "out"}, max_steps=1)
    with torch.no_grad():
        result = search_recursive_graphs((start_recursive_search(root, {"x": torch.randn(1, 2, 8)}),),
                                         width=16, beam_width=16, preserve_effect_coverage=True)
    assert result.maximum_active <= 16 and result.maximum_completed <= 16
    assert result.maximum_call_depth == 1
    assert result.winner.execution.outputs["answer"].shape == (1, 2, 8)
    assert all(owner.revision.item() == 0 for owner in original.owner_states)
