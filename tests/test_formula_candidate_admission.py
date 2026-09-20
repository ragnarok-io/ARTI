from dataclasses import replace

import pytest
import torch

from arti import mechanisms as m


@pytest.fixture(autouse=True)
def _isolated_rng():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(9701)
        yield


def _producer(name, source, destination, *, guard=()):
    kind = m.TensorType(("B", "D"), ("B", 3), dtype="floating")
    x = m.InputBinding("x", kind)
    return m.FormulaProgramTensorCandidateV3(m.FormulaProgramCandidate(
        name, m.FormulaProgram.build(outputs=(m.add(x, x),)),
        input_slots={"x": source}, output_slot=destination, requires_empty_slots=guard,
    ))


def _query(version=5):
    candidates = (_producer("a", "x", "a", guard=("b",)), _producer("b", "a", "b"))
    options = dict(slot_ids=("x", "a", "b"), candidates=candidates, max_steps=3, max_tensor_steps=2)
    if version == 5:
        return m.FormulaProgramQueryV5(**options, terminal_slots={"output": "b"})
    return m.FormulaProgramQueryV4(**options, terminal_slot="b")


@pytest.mark.parametrize("version", [4, 5])
def test_full_candidate_mask_matches_wiring_prefilter_for_each_arena_and_depth(version):
    query = _query(version)
    initial = query._arena({"x": torch.ones(1, 3)})
    first = query.candidates[0](initial)
    final = query.candidates[1](first)
    bad_shape = query._arena({"x": torch.ones(1, 4)})
    bad_value = query._arena({"x": torch.full((1, 3), torch.nan)})
    for arena in (initial, first, final, bad_shape, bad_value):
        for steps in range(5):
            expected = [query._candidate_eligible(candidate, arena, steps=steps) for candidate in query.candidates]
            expected.append(query._stop_eligible(arena, steps=steps))
            assert query.eligible(arena, steps=steps).tolist() == expected
            assert query._has_eligible(arena, steps=steps) == any(expected)


def test_wiring_filter_does_not_cache_current_bank_or_lineage():
    from benchmarks._federated_v4_federation import build_autonomous_effect_federation

    federation = build_autonomous_effect_federation(
        hidden_dim=4, rank=4, seed=421, device=torch.device("cpu"),
        plastic_branches=8, min_operations=1, max_operations=2,
    )
    query = federation.query
    entry = query._arena({"x": torch.ones(1, 2, 4)})
    producer = next(c for c in query.candidates if isinstance(c, m.FormulaProgramTensorCandidateV3)
                    and c.bank_slot_ref is not None and c.accepts(entry))
    written = producer(entry)
    effect = next(c for c in query.candidates if isinstance(c, m.FormulaProgramEffectCandidateV3) and c.accepts(written))
    updated = effect(written)
    # Keep the same SSA tensors/shape but use the newer Bank revision. The old
    # producer lineage can no longer authorize another effect on that value.
    stale = replace(written, bank_state=updated.committed_state(), proposals=())
    assert effect.accepts(written) and not effect.accepts(stale)
    for arena in (entry, written, updated, stale):
        expected = [query._candidate_eligible(c, arena, steps=1) for c in query.candidates]
        expected.append(query._stop_eligible(arena, steps=1))
        assert query.eligible(arena, steps=1).tolist() == expected


def test_custom_admission_and_current_wiring_are_not_hidden_by_prefilter(monkeypatch):
    query = _query()
    entry = query._arena({"x": torch.ones(1, 3)})
    assert query._structural_candidates((entry,)) == ((0,),)
    monkeypatch.setattr(query.candidates[1], "accepts", lambda arena: True)
    assert query.eligible(entry, steps=0).tolist() == [True, True, False]
    monkeypatch.undo()
    query.candidates[1].candidate.input_slots["x"] = "x"
    assert query.eligible(entry, steps=0).tolist() == [True, True, False]
    query.candidates[1].candidate.input_slots["x"] = "unknown"
    assert query.eligible(entry, steps=0).tolist() == [True, False, False]


def test_search_structure_scope_reuses_wiring_but_not_numerical_admission(monkeypatch):
    from arti.formula_program_query_v4 import _candidate_structure_scope

    query = _query()
    value = torch.ones(1, 3)
    entry = query._arena({"x": value})
    compiled = []
    original = query._compile_candidate_wiring

    def record(candidates):
        compiled.append(tuple(map(id, candidates)))
        return original(candidates)

    monkeypatch.setattr(query, "_compile_candidate_wiring", record)
    with _candidate_structure_scope():
        assert query.eligible(entry, steps=0).tolist() == [True, False, False]
        assert query.eligible(entry, steps=0).tolist() == [True, False, False]
        value.fill_(torch.nan)
        assert query.eligible(entry, steps=0).tolist() == [False, False, False]
        assert len(compiled) == 1
    value.fill_(1)
    query.candidates[1].candidate.input_slots["x"] = "x"
    with _candidate_structure_scope():
        assert query.eligible(entry, steps=0).tolist() == [True, True, False]
        assert len(compiled) == 2
        reversed_candidates = tuple(reversed(query.candidates))
        assert query._structural_candidates((entry,), reversed_candidates) == ((0, 1),)
        assert len(compiled) == 3


def test_search_structure_scope_restores_nested_and_failed_invocations(monkeypatch):
    from arti.formula_program_query_v4 import _candidate_structure_scope, _CANDIDATE_STRUCTURE_PLANS

    query = _query()
    entry = query._arena({"x": torch.ones(1, 3)})
    assert _CANDIDATE_STRUCTURE_PLANS.get() is None
    with _candidate_structure_scope():
        query.eligible(entry, steps=0)
        outer = _CANDIDATE_STRUCTURE_PLANS.get()
        assert len(outer) == 1
        with pytest.raises(RuntimeError), _candidate_structure_scope():
            query.eligible(entry, steps=0)
            assert _CANDIDATE_STRUCTURE_PLANS.get() is not outer
            raise RuntimeError("cancel invocation")
        assert _CANDIDATE_STRUCTURE_PLANS.get() is outer
    assert _CANDIDATE_STRUCTURE_PLANS.get() is None


def test_child_existence_check_short_circuits_but_parent_query_scores_all(monkeypatch):
    child = m.FormulaProgramQueryV5(
        slot_ids=("x", "a", "b"), candidates=(_producer("a", "x", "a"), _producer("b", "x", "b")),
        terminal_slots={"output": "a"},
    )
    call = m.FormulaProgramCallCandidateV1("child", child, input_slots={"x": "x"}, output_slots={"output": "y"})
    parent = m.FormulaProgramQueryV5(slot_ids=("x", "y"), candidates=(call,), terminal_slots={"output": "y"})
    entry = parent._arena({"x": torch.ones(1, 3)})
    checks = []
    original = child._candidate_eligible

    def counted(candidate, arena, *, steps):
        checks.append(candidate.candidate_id)
        return original(candidate, arena, steps=steps)

    monkeypatch.setattr(m.FormulaProgramQueryV5, "_candidate_eligible", lambda self, candidate, arena, steps: counted(candidate, arena, steps=steps))
    assert call.accepts(entry)
    assert checks == ["a"]
    checks.clear()
    assert child.eligible(call._entry(entry), steps=0).tolist() == [True, True, False]
    assert checks == ["a", "b"]
    monkeypatch.setattr(child, "eligible", lambda arena, steps: torch.tensor([False]))
    assert not call.accepts(entry)


def test_child_custom_admission_exception_keeps_original_call_result(monkeypatch):
    child = _query()
    call = m.FormulaProgramCallCandidateV1("child", child, input_slots={"x": "x"}, output_slots={"output": "y"})
    parent = m.FormulaProgramQueryV5(slot_ids=("x", "y"), candidates=(call,), terminal_slots={"output": "y"})

    def invalid(_arena):
        raise ValueError("custom admission")

    monkeypatch.setattr(child.candidates[1], "accepts", invalid)
    assert not call.accepts(parent._arena({"x": torch.ones(1, 3)}))


def test_custom_child_entry_is_not_prefiltered_by_original_input_slots(monkeypatch):
    child = _query()
    call = m.FormulaProgramCallCandidateV1("child", child, input_slots={"x": "empty"}, output_slots={"output": "y"})
    parent = m.FormulaProgramQueryV5(slot_ids=("x", "empty", "y"), candidates=(call,), terminal_slots={"output": "y"})
    monkeypatch.setattr(call, "_entry", lambda arena: child._arena({"x": arena.values.get("x")}))
    assert parent.eligible(parent._arena({"x": torch.ones(1, 3)}), steps=0).tolist() == [True, False]


def test_custom_fabric_binding_retains_native_validation_context(monkeypatch):
    from arti.formula_v2 import _FINITE_VALIDATION_CAPTURE

    query = _query()
    fabric = query.candidates[0].candidate.fabric
    original = fabric.bind_tensors

    def custom(**kwargs):
        assert _FINITE_VALIDATION_CAPTURE.get() is None
        return original(**kwargs)

    monkeypatch.setattr(fabric, "bind_tensors", custom)
    assert query.eligible(query._arena({"x": torch.ones(1, 3)}), steps=0).tolist() == [True, False, False]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_device_admission_has_no_local_scalar_and_matches_bad_row_reference(device):
    from torch.utils._python_dispatch import TorchDispatchMode
    from arti._formula_candidate_admission import candidate_mask

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    query = _query().to(device)
    arenas = (query._arena({"x": torch.ones(1, 3, device=device)}),
              query._arena({"x": torch.full((1, 3), torch.nan, device=device)}))
    reads = []

    class NoScalarRead(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            if func is torch.ops.aten._local_scalar_dense.default:
                reads.append(func)
            return func(*args, **(kwargs or {}))

    with NoScalarRead():
        actual = candidate_mask(query, arenas, query.candidates, steps=0, include_stop=True)
    assert reads == []
    assert actual.tolist() == [[True, False, False], [False, False, False]]


@pytest.mark.parametrize("encoder", [False, True])
def test_dead_nan_row_is_excluded_before_scoring_and_does_not_change_neighbor_vjp(encoder):
    from types import SimpleNamespace
    from benchmarks.train_federated_branch_visible_federation import _ranked_candidates_many, _SearchBranch

    query = _query()
    if encoder:
        query = m.FormulaProgramQueryV5(
            slot_ids=query.slot_ids, candidates=tuple(query.candidates), terminal_slots=query.terminal_slots,
            tensor_encoder=m.FormulaProgramQueryTensorEncoderV1(3, 4),
        )
    good = torch.ones(1, 3, requires_grad=True)
    bad = torch.full((1, 3), torch.nan, requires_grad=True)
    branches = tuple(_SearchBranch(query._arena({"x": x}), x.new_zeros(()), ()) for x in (good, bad))
    before = _ranked_candidates_many(SimpleNamespace(query=query), branches[:1], tuple(query.candidates),
                                     steps=0, width=2, include_stop=True)
    rejections = []
    after = _ranked_candidates_many(SimpleNamespace(query=query), branches, tuple(query.candidates),
                                    steps=0, width=2, include_stop=True, numerical_rejections=rejections)
    assert after[1] == () and rejections == []
    assert [c.candidate_id for c, _ in before[0]] == [c.candidate_id for c, _ in after[0]]
    args = (good, bad, *query.network.parameters())
    grads = [torch.autograd.grad(sum(s for _, s in rows[0]), args, allow_unused=True) for rows in (before, after)]
    for left, right in zip(*grads, strict=True):
        assert (left is None) == (right is None)
        if left is not None:
            torch.testing.assert_close(left, right, rtol=0, atol=0)


@pytest.mark.parametrize("device", ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",))
@pytest.mark.parametrize("width", (1, 2, 9))
def test_device_ranking_matches_full_legal_softmax_and_name_ties(device, width):
    from types import SimpleNamespace
    from benchmarks.train_federated_branch_visible_federation import _ranked_candidates_many, _SearchBranch

    candidates = (_producer("zeta", "x", "z"), _producer("alpha", "x", "a"),
                  _producer("mu", "x", "m"), _producer("blocked", "z", "later"))
    query = m.FormulaProgramQueryV5(
        slot_ids=("x", "z", "a", "m", "later"), candidates=candidates,
        terminal_slots={"output": "later"}, max_steps=4,
    ).to(device)
    with torch.no_grad():
        for parameter in query.network.parameters():
            parameter.zero_()
        query.network[-1].bias.copy_(torch.tensor([0.2, 0.2, -0.5, 100., 100.], device=device))
    x = torch.ones(1, 3, device=device, requires_grad=True)
    arena = query._arena({"x": x})
    branch = _SearchBranch(arena, x.new_zeros(()), ())
    records = {}
    from torch.utils._python_dispatch import TorchDispatchMode

    transfers = []

    class RecordHostTransfers(TorchDispatchMode):
        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if func is torch.ops.aten._to_copy.default and args[0].device.type == "cuda":
                destination = kwargs.get("device")
                if destination is not None and torch.device(destination).type == "cpu":
                    transfers.append(args[0].dtype)
            return func(*args, **kwargs)

    with RecordHostTransfers():
        actual = _ranked_candidates_many(SimpleNamespace(query=query), (branch,), candidates,
                                        steps=0, width=width, include_stop=True, eligibility_records=records)[0]
    if device == "cuda":
        assert transfers and all(dtype in (torch.bool, torch.int32, torch.int64) for dtype in transfers)
    mask = torch.tensor([True, True, True, False, False], device=device)
    expected = query.network(query._summarize_values(arena.values))[0].masked_fill(~mask, -torch.inf).log_softmax(0)
    indices = [1, 0, 2][:width]
    assert [candidate.candidate_id for candidate, _ in actual] == [candidates[i].candidate_id for i in indices]
    assert records == {0: (0, 1, 2)}
    for (_, score), index in zip(actual, indices, strict=True):
        torch.testing.assert_close(score, expected[index], rtol=0, atol=0)
    parameters = (x, *query.network.parameters())
    actual_grads = torch.autograd.grad(sum(score for _, score in actual), parameters, allow_unused=True)
    expected_grads = torch.autograd.grad(expected[indices].sum(), parameters, allow_unused=True)
    for left, right in zip(actual_grads, expected_grads, strict=True):
        assert (left is None) == (right is None)
        if left is not None:
            torch.testing.assert_close(left, right, rtol=0, atol=0)
