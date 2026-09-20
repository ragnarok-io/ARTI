from contextlib import nullcontext
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from benchmarks._federated_device_frontier import (
    DeviceFrontier, _FrontierKernel, device_prune, tensor_frontier_execution,
)
from benchmarks._federated_recursive_search import (
    _selection_prune, final_graph_loss, search_recursive_graphs, start_recursive_search,
)
from benchmarks.train_federated_autonomous_federation import _coverage_prune, _effect_coverage_labels
from benchmarks.train_federated_branch_visible_federation import _prune


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _items(device, dtype, count=39):
    generator = torch.Generator().manual_seed(808)
    scores = torch.randint(-3, 3, (count,), generator=generator).to(device=device, dtype=dtype)
    scores[-2:] = -torch.inf
    items = []
    for index, score in enumerate(scores):
        route = tuple({
            "candidate_id": f"branch-{(index + offset) % 9}",
            "kind": "effect" if (index + offset) % 3 else "ordinary",
            "atom_ref": f"atom-{(index + offset) % 11}",
            "structure_family": ("low-rank", "attention", "gelu")[offset % 3],
        } for offset in range(index % 4))
        items.append(SimpleNamespace(log_probability=score, route=route, selection_priority=None))
    # Coverage deduplicates object identity; plain top-k intentionally does not.
    return (*items, items[2], items[7])


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("coverage", [False, True])
@pytest.mark.parametrize("width", [1, 3, 16, 99])
def test_tensor_prune_matches_full_route_coverage_and_ties(device, dtype, coverage, width):
    items = _items(device, dtype)
    reference = _coverage_prune if coverage else _prune
    expected = reference(items, beam_width=width)
    with tensor_frontier_execution(backend="eager"):
        actual = device_prune(items, width=width, coverage=coverage,
                              labels=tuple(_effect_coverage_labels(item) for item in items))
    assert tuple(map(id, actual)) == tuple(map(id, expected))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("coverage", [False, True])
def test_refill_recomputes_representatives_and_does_not_reexecute(device, coverage):
    items = _items(device, torch.float32)[:25]
    generator = torch.Generator().manual_seed(515)
    for item in items:
        item.selection_priority = item.log_probability.detach().cpu().double() + torch.rand((), generator=generator)
    reference = _coverage_prune if coverage else _prune
    remaining = tuple(items)
    ready = {id(items[8]), id(items[10])}
    worklist = DeviceFrontier(items, width=7, coverage=coverage,
                              labels=tuple(_effect_coverage_labels(item) for item in items),
                              is_expansion=lambda item: id(item) not in ready, exploration=True)
    rng_state = generator.get_state().clone()
    while remaining:
        selected, pending = worklist.select()
        expected = _selection_prune(remaining, beam_width=7, prune=reference)
        assert tuple(map(id, selected)) == tuple(map(id, expected))
        assert tuple(map(id, pending)) == tuple(id(item) for item in expected if id(item) not in ready)
        rejected = {id(item) for item in selected[::2]}
        accepted = {id(item) for item in pending} - rejected
        worklist.record(accepted=accepted, rejected=rejected)
        ready.update(accepted)
        remaining = tuple(item for item in remaining if id(item) not in rejected)
    assert worklist.select() == ((), ())
    assert torch.equal(generator.get_state(), rng_state)


@pytest.mark.parametrize("coverage", [False, True])
def test_selection_is_one_graph_with_dynamic_masks_and_no_host_scalars(coverage):
    graphs = []
    def capture(graph, _inputs):
        graphs.append(graph)
        return graph.forward
    torch.compiler.reset()
    kernel = _FrontierKernel(4, coverage)
    compiled = torch.compile(kernel, backend=capture, fullgraph=True)
    scores = torch.tensor([0., 2., 1., 2., -torch.inf, 0., 1., 3.])
    alive = torch.ones(8, dtype=torch.bool)
    ready = torch.zeros(8, dtype=torch.bool)
    members = torch.eye(4, dtype=torch.bool).repeat(2, 1)
    ties = torch.arange(7, -1, -1)
    with torch.no_grad():
        for index in range(8):
            alive[index] = False
            ready[(index + 1) % 8] = True
            scores = scores.roll(1)
            members = members.roll(1, dims=1)
            expected = kernel(scores, alive, ready, members, ties)
            actual = compiled(scores, alive, ready, members, ties)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert len(graphs) == 1
    assert not any("_local_scalar_dense" in str(node.target) or str(node.target).endswith(".item")
                   for node in graphs[0].graph.nodes)


def _compare_results(expected, actual):
    for name in ("maximum_active", "maximum_completed", "maximum_frontier", "maximum_call_depth",
                 "scored_expansions", "executed_expansions", "entered_calls", "returned_calls",
                 "root_stops", "numerical_rejections", "checked_stop_readouts"):
        assert getattr(actual, name) == getattr(expected, name), name
    assert [item.route for item in actual.branches] == [item.route for item in expected.branches]
    for left, right in zip(expected.branches, actual.branches, strict=True):
        torch.testing.assert_close(left.log_probability, right.log_probability)
        for name in left.execution.outputs:
            torch.testing.assert_close(left.execution.outputs[name], right.execution.outputs[name])
        assert left.execution.bank_state.revisions == right.execution.bank_state.revisions
        for a, b in zip(left.execution.bank_state.values, right.execution.bank_state.values, strict=True):
            torch.testing.assert_close(a, b)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("coverage", [False, True])
@pytest.mark.parametrize("explore", [False, True])
def test_nested_search_terminal_refill_and_full_retained_gradient(device, coverage, explore):
    from test_federated_recursive_search import _parent, _choice_child

    query = _parent((_choice_child("a", (1., 2., 3., 4.)), _choice_child("b", (5., 6., 7.)))).to(device)
    parameters = tuple(p for p in query.parameters() if p.requires_grad)
    def run(backend):
        with nullcontext() if backend is None else tensor_frontier_execution(backend=backend):
            x = torch.ones(1, 2, device=device, requires_grad=True)
            branches = tuple(replace(start_recursive_search(query, {"x": x}), log_probability=x.new_tensor(-0.1 * i))
                             for i in range(2))
            result = search_recursive_graphs(branches, width=4, beam_width=5,
                preserve_effect_coverage=coverage, record_query_choices=True,
                terminal_admission=lambda item: bool(item.execution.outputs["answer"].max() < 6),
                exploration_generator=torch.Generator().manual_seed(53) if explore else None)
            losses = torch.stack([b.execution.outputs["answer"].square().mean() for b in result.branches])
            gradient = torch.autograd.grad(final_graph_loss(result, losses), (x, *parameters), allow_unused=True)
            return result, gradient
    expected, left_grads = run(None)
    actual, right_grads = run("eager")
    _compare_results(expected, actual)
    for left, right in zip(left_grads, right_grads, strict=True):
        assert (left is None) == (right is None)
        if left is not None:
            torch.testing.assert_close(left, right)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("count", [39, 245])
def test_inductor_frontier_reuses_current_scores_masks_and_retained_flags(count):
    items = _items("cuda", torch.float32, count=count)
    with tensor_frontier_execution(backend="inductor"):
        worklist = DeviceFrontier(items, width=8, coverage=True,
                                  labels=tuple(_effect_coverage_labels(item) for item in items),
                                  is_expansion=lambda _item: True)
        remaining = items
        for _ in range(3):
            selected, pending = worklist.select()
            expected = _coverage_prune(remaining, beam_width=8)
            assert tuple(map(id, selected)) == tuple(map(id, expected))
            rejected = {id(selected[0])}
            worklist.record(accepted=(id(item) for item in pending[1:]), rejected=rejected)
            remaining = tuple(item for item in remaining if id(item) not in rejected)
