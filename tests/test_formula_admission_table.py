from dataclasses import replace
from contextlib import nullcontext

import pytest
import torch

from arti import mechanisms as m
from arti._formula_admission_table import _ADMISSION_TABLES, _AdmissionKernel, indexed_candidate_admission
from arti._formula_candidate_admission import candidate_mask


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("version", [4, 5])
@pytest.mark.parametrize("device", DEVICES)
def test_indexed_mask_matches_native_wiring_shapes_budgets_and_finite(version, device):
    from test_formula_candidate_admission import _query

    query = _query(version).to(device)
    root = query._arena({"x": torch.ones(1, 3, device=device)})
    first = query.candidates[0](root)
    final = query.candidates[1](first)
    roots = (root, first, final, query._arena({"x": torch.ones(2, 3, device=device)}),
             query._arena({"x": torch.ones(1, 4, device=device)}),
             query._arena({"x": torch.full((1, 3), torch.nan, device=device)}))
    expected = [candidate_mask(query, roots, query.candidates, steps=step, include_stop=True) for step in range(5)]
    with indexed_candidate_admission():
        for step, native in enumerate(expected):
            torch.testing.assert_close(candidate_mask(query, roots, query.candidates, steps=step, include_stop=True), native)
        plan = next(iter(_ADMISSION_TABLES.get().values()))
        assert plan is not None
        layouts = len(plan.layouts)
        root.values.get("x").fill_(torch.nan)
        actual = candidate_mask(query, roots, query.candidates, steps=0, include_stop=True)
        assert not actual[0].any()
        assert len(plan.layouts) == layouts


@pytest.mark.parametrize("device", DEVICES)
def test_indexed_effects_reread_current_owner_and_stale_lineage(device):
    from benchmarks._federated_v4_federation import build_autonomous_effect_federation

    model = build_autonomous_effect_federation(hidden_dim=4, rank=4, seed=421, device=torch.device(device),
                                              plastic_branches=8, min_operations=1, max_operations=2)
    query = model.query
    root = query._arena({"x": torch.ones(1, 2, 4, device=device)})
    producer = next(c for c in query.candidates if isinstance(c, m.FormulaProgramTensorCandidateV3)
                    and c.bank_slot_ref is not None and c.accepts(root))
    written = producer(root)
    effects = tuple(c for c in query.candidates if isinstance(c, m.FormulaProgramEffectCandidateV3) and c.accepts(written))
    changed = tuple(effect(written) for effect in effects)
    stale = replace(written, bank_state=changed[0].committed_state(), proposals=())
    roots = (root, written, *changed, stale)
    expected = candidate_mask(query, roots, query.candidates, steps=1, include_stop=True)
    with indexed_candidate_admission():
        actual = candidate_mask(query, roots, query.candidates, steps=1, include_stop=True)
        torch.testing.assert_close(actual, expected)
        assert next(iter(_ADMISSION_TABLES.get().values())) is not None
        # Same metadata, different current values: the cache must not remember
        # an owner's former finite value or a prototype branch's result.
        written.bank_state.value(producer.bank_slot_ref).fill_(torch.inf)
        indexed = candidate_mask(query, roots, query.candidates, steps=1, include_stop=True)
    native = candidate_mask(query, roots, query.candidates, steps=1, include_stop=True)
    torch.testing.assert_close(indexed, native)


def test_extensions_and_scope_changes_keep_native_semantics(monkeypatch):
    from test_formula_candidate_admission import _query

    query = _query()
    root = query._arena({"x": torch.ones(1, 3)})
    monkeypatch.setattr(query.candidates[1], "accepts", lambda arena: True)
    native = query.eligible(root, steps=0)
    with indexed_candidate_admission():
        torch.testing.assert_close(query.eligible(root, steps=0), native)
        assert next(iter(_ADMISSION_TABLES.get().values())) is None
    monkeypatch.undo()
    query = _query()
    root = query._arena({"x": torch.ones(1, 3)})
    with indexed_candidate_admission():
        query.eligible(root, steps=0)
        outer = _ADMISSION_TABLES.get()
        assert next(iter(outer.values())) is not None
        with pytest.raises(RuntimeError), indexed_candidate_admission():
            assert _ADMISSION_TABLES.get() is not outer
            raise RuntimeError("cancel")
        assert _ADMISSION_TABLES.get() is outer
    assert _ADMISSION_TABLES.get() is None


def test_admission_kernel_is_fullgraph_with_device_budget_counts():
    from test_formula_candidate_admission import _query

    query = _query()
    kernel = _AdmissionKernel(query, True)
    graphs = []
    def capture(graph, _inputs):
        graphs.append(graph)
        return graph.forward
    compiled = torch.compile(kernel, fullgraph=True, backend=capture)
    flags = torch.tensor([[True, True, False], [True, False, True]])
    sources = torch.tensor([[1], [2]])
    metadata = torch.ones(2, dtype=torch.bool)
    occupied = torch.tensor([[True, False, False], [True, True, False]])
    required = torch.tensor([[True, False, False], [False, True, False]])
    empty = torch.tensor([[False, True, True], [False, False, True]])
    kinds = torch.zeros(2, dtype=torch.int64)
    terminal_writes = torch.tensor([False, True])
    terminal_indices = torch.tensor([2])
    counts = torch.tensor([[0, 0], [1, 1]])
    for depth in range(5):
        inputs = (flags, sources, metadata, occupied, required, empty, kinds, terminal_writes,
                  terminal_indices, counts, torch.tensor(depth))
        torch.testing.assert_close(compiled(*inputs), kernel(*inputs))
    assert len(graphs) == 1
    assert not any("_local_scalar_dense" in str(node.target) for node in graphs[0].graph.nodes)


@pytest.mark.parametrize("version", [4, 5])
def test_new_architecture_scope_rebuilds_changed_budgets_and_wiring(version):
    from test_formula_candidate_admission import _query

    query = _query(version)
    root = query._arena({"x": torch.ones(1, 3)})
    with indexed_candidate_admission():
        assert query.eligible(root, steps=0).tolist() == [True, False, False]
        original = next(iter(_ADMISSION_TABLES.get().values()))
    query.max_steps = 1
    query.candidates[1].candidate.input_slots["x"] = "x"
    expected = [query.eligible(root, steps=step) for step in (0, 1)]
    assert expected[0].tolist() == [True, True, False]
    assert expected[1].tolist() == [False, False, False]
    with indexed_candidate_admission():
        for step, native in enumerate(expected):
            torch.testing.assert_close(query.eligible(root, steps=step), native)
        assert next(iter(_ADMISSION_TABLES.get().values())) is not original


@pytest.mark.parametrize("device", DEVICES)
def test_indexed_nested_search_preserves_retained_loss_and_unused_gradients(device):
    from test_federated_recursive_search import _parent, _choice_child
    from test_federated_device_frontier import _compare_results
    from benchmarks._federated_recursive_search import final_graph_loss, search_recursive_graphs, start_recursive_search

    query = _parent((_choice_child("a", (1., 2., 3., 4.)), _choice_child("b", (5., 6., 7.)))).to(device)
    parameters = tuple(p for p in query.parameters() if p.requires_grad)

    def run(indexed):
        with indexed_candidate_admission() if indexed else nullcontext():
            x = torch.ones(1, 2, device=device, requires_grad=True)
            result = search_recursive_graphs((start_recursive_search(query, {"x": x}),), width=3, beam_width=4,
                record_query_choices=True, exploration_generator=torch.Generator().manual_seed(53),
                terminal_admission=lambda item: bool(item.execution.outputs["answer"].max() < 6))
            losses = torch.stack([branch.execution.outputs["answer"].square().mean() for branch in result.branches])
            gradients = torch.autograd.grad(final_graph_loss(result, losses), (x, *parameters), allow_unused=True)
            if indexed:
                tables = tuple(_ADMISSION_TABLES.get().values())
                assert any(table is not None for table in tables)
                assert any(table is None for table in tables)  # CALL stays native.
            return result, gradients

    expected, left = run(False)
    actual, right = run(True)
    _compare_results(expected, actual)
    assert any(gradient is None for gradient in left)
    for a, b in zip(left, right, strict=True):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b, rtol=0, atol=0)
