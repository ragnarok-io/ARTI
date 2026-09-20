from types import SimpleNamespace

import pytest
import torch

from arti import mechanisms as m
from arti.formula_program_query_v4 import _CANDIDATE_STRUCTURE_PLANS, _candidate_structure_scope
from benchmarks.train_federated_branch_visible_federation import (
    _SearchBranch, _query_action_width, _ranked_candidates_many, _search_action_plan,
)


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _query(device):
    kind = m.TensorType(("B", "D"), ("B", 3), dtype="float32")
    x = m.InputBinding("x", kind)
    program = m.FormulaProgram.build(outputs=(m.add(x, x),))
    candidates = tuple(m.FormulaProgramTensorCandidateV3(m.FormulaProgramCandidate(
        name, program, input_slots={"x": "x"}, output_slot="terminal", operands={},
    )) for name in ("z", "a", "m"))
    query = m.FormulaProgramQueryV4(
        candidates=candidates, slot_ids=("x", "terminal"), terminal_slot="terminal",
        max_steps=3, hidden_dim=4,
    ).to(device)
    with torch.no_grad():
        for parameter in query.network.parameters():
            parameter.zero_()
    return query


@pytest.mark.parametrize("device", DEVICES)
def test_action_plan_is_scoped_ordered_and_indices_can_cross_inference_mode(device):
    query = _query(device)
    candidates = tuple(query.candidates)
    with _candidate_structure_scope():
        with torch.inference_mode():
            first = _search_action_plan(query, candidates, include_stop=True)
            columns, ties = first.indices(torch.device(device))
        assert not columns.is_inference() and not ties.is_inference()
        assert first is _search_action_plan(query, candidates, include_stop=True)
        assert first.indices(torch.device(device))[0] is columns
        assert columns.tolist() == [0, 1, 2, 3]
        assert ties.tolist() == [1, 2, 3, 0]
        logits = torch.randn(2, 4, device=device, requires_grad=True)
        logits.index_select(1, columns).sum().backward()
        torch.testing.assert_close(logits.grad, torch.ones_like(logits))
        reverse = _search_action_plan(query, candidates[::-1], include_stop=False)
        assert reverse.columns == (2, 1, 0) and reverse.actions == candidates[::-1]
        with pytest.raises(RuntimeError), _candidate_structure_scope():
            assert _search_action_plan(query, candidates, include_stop=True) is not first
            raise RuntimeError("scope exit")
        assert _search_action_plan(query, candidates, include_stop=True) is first
    assert _CANDIDATE_STRUCTURE_PLANS.get() is None
    candidates[0].candidate.candidate_id = "b"
    with _candidate_structure_scope():
        after = _search_action_plan(query, candidates, include_stop=True)
        assert after is not first and after.tie_order == (1, 0, 2, 3)
    assert _search_action_plan(query, candidates, include_stop=True) is not _search_action_plan(query, candidates, include_stop=True)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("width", [1, 3, 7])
def test_scoped_rank_preserves_ties_eligibility_full_softmax_and_gradients(device, width):
    query = _query(device)
    candidates = tuple(reversed(query.candidates))
    values = tuple(torch.full((1, 3), v, device=device, requires_grad=True) for v in (1., 2.))
    branches = tuple(_SearchBranch(query._arena({"x": value}), value.new_zeros(()), ()) for value in values)
    federation = SimpleNamespace(query=query)
    expected_records, actual_records = {}, {}
    expected = _ranked_candidates_many(
        federation, branches, candidates, steps=0, width=width,
        include_stop=True, eligibility_records=expected_records,
    )
    with _candidate_structure_scope():
        actual = _ranked_candidates_many(
            federation, branches, candidates, steps=0, width=width,
            include_stop=True, eligibility_records=actual_records,
        )
        with torch.no_grad():
            query.network[-1].bias[0].add_(1)
        changed = _ranked_candidates_many(federation, branches, candidates, steps=0, width=width, include_stop=True)
        assert changed[0][0][0].candidate_id == "z"
    assert actual_records == expected_records == {0: (2, 1, 0), 1: (2, 1, 0)}
    for reference, row in zip(expected, actual, strict=True):
        assert [item.candidate_id for item, _ in row] == ["a", "m", "z"][:width]
        torch.testing.assert_close(torch.stack([score for _, score in row]), torch.stack([score for _, score in reference]), rtol=0, atol=0)
    parameters = (*query.network.parameters(), *values)
    gradients = [torch.autograd.grad(-rows[0][0][1], parameters, allow_unused=True, retain_graph=True) for rows in (expected, actual)]
    for reference, gradient in zip(*gradients, strict=True):
        assert (reference is None) == (gradient is None)
        if reference is not None:
            torch.testing.assert_close(gradient, reference, rtol=0, atol=0)


def test_custom_candidate_names_and_action_protocol_are_not_cached(monkeypatch):
    query = _query("cpu")
    candidates = tuple(query.candidates)
    candidate_type = type(candidates[0])
    original_name = candidate_type.candidate_id
    original_actions = type(query).action_ids
    with _candidate_structure_scope():
        native = _search_action_plan(query, candidates, include_stop=True)
        names = {"z": "b", "a": "c", "m": "a"}
        monkeypatch.setattr(candidate_type, "candidate_id", property(lambda item: names[original_name.fget(item)]))
        changed = _search_action_plan(query, candidates, include_stop=True)
        assert changed is not native and changed.tie_order == (2, 0, 1, 3)
        assert _search_action_plan(query, candidates, include_stop=True) is not changed
        monkeypatch.undo()
        monkeypatch.setattr(type(query), "action_ids", property(lambda item: (*original_actions.fget(item), "custom")))
        assert _query_action_width(query) == 5
        assert _search_action_plan(query, candidates, include_stop=True) is not native


def test_global_column_mapping_tracks_declared_candidate_order():
    query = _query("cpu")
    candidates = tuple(query.candidates)
    with _candidate_structure_scope():
        first = _search_action_plan(query, candidates, include_stop=True)
        query.candidates = torch.nn.ModuleList(reversed(candidates))
        second = _search_action_plan(query, candidates, include_stop=True)
        assert second is not first
        assert second.columns == (2, 1, 0, 3)


def test_wrapped_candidate_dynamic_name_disables_action_reuse():
    class DynamicCandidate(m.FormulaProgramCandidate):
        @property
        def candidate_id(self):
            return self.name_value

        @candidate_id.setter
        def candidate_id(self, value):
            self.name_value = value

    query = _query("cpu")
    wrapper = query.candidates[0]
    wrapper.candidate = DynamicCandidate(
        "z", wrapper.candidate.program, input_slots={"x": "x"},
        output_slot="terminal", operands={},
    )
    with _candidate_structure_scope():
        first = _search_action_plan(query, tuple(query.candidates), include_stop=True)
        wrapper.candidate.name_value = "b"
        second = _search_action_plan(query, tuple(query.candidates), include_stop=True)
        assert second is not first and second.tie_order == (1, 0, 2, 3)
