from types import SimpleNamespace

import pytest
import torch

from benchmarks._federated_recursive_search import (
    replay_recursive_graph, replay_recursive_query_scores, search_recursive_graphs, start_recursive_search,
)
from benchmarks.train_qwen_federated_autonomous_federation import (
    _clip_training_gradients, _clipping_routing_parameters, _query_credit_objective, _routing_parameters,
)
from test_formula_program_query_v6 import federation


def search(query, values, state):
    with torch.no_grad():
        return search_recursive_graphs(
            (start_recursive_search(query, values, bank_state=state),),
            width=2, beam_width=4, preserve_effect_coverage=False, record_query_choices=True,
        )


def test_k_wide_uses_federation_responses_and_recorded_replay_once():
    query = federation(write=True)
    values = {"x": torch.ones(1, 1)}
    state = query.initial_bank_state()
    result = search(query, values, state)
    assert len(result.branches) == 2
    assert {branch.route[-2]["local_candidate_id"] for branch in result.branches} == {"left", "right"}
    probabilities = torch.stack(tuple(branch.log_probability.exp() for branch in result.branches))
    torch.testing.assert_close(probabilities.sum(), torch.tensor(1.0))
    for branch in result.branches:
        replay = replay_recursive_graph(query, values, branch.route, bank_state=state, use_recorded_choices=True)
        torch.testing.assert_close(replay.log_probability, branch.log_probability)
        torch.testing.assert_close(replay.execution.outputs["result"], branch.execution.outputs["result"])
        assert len(replay.execution.proposals) == 1


def test_later_route_credit_reaches_earlier_real_write_without_extra_effects():
    query = federation(write=True)
    values = {"x": torch.ones(1, 1)}
    initial = query.initial_bank_state()
    first_search = search(query, values, initial)
    first = replay_recursive_graph(query, values, first_search.branches[0].route, bank_state=initial,
                                   use_recorded_choices=True)
    state = first.execution.bank_state
    second_search = search(query, values, state)
    second_route = second_search.branches[0].route
    replay = replay_recursive_query_scores(query, values, second_route, bank_state=state,
                                          use_recorded_choices=True)
    rate = query.candidates[1].operand_store.tensor("rate")
    actual = torch.autograd.grad(replay.log_probability, rate, retain_graph=True)[0]
    v = state.value(query.candidates[0].bank_slot_ref).reshape(())
    choice = second_route[-2]["local_candidate_id"]
    manual = torch.stack((v, -v)).log_softmax(0)[0 if choice == "left" else 1]
    expected = torch.autograd.grad(manual, rate, retain_graph=True)[0]
    assert expected.abs().sum() > 0
    torch.testing.assert_close(actual, expected)
    assert len(first.execution.proposals) == len(replay.execution.proposals) == 1
    slow = _routing_parameters(SimpleNamespace(query=query))
    assert any(p is rate for p in slow)
    assert not any(p is owner.value for owner in query.owner_states for p in slow)
    task = sum(value.square().mean() for value in replay.execution.outputs.values())
    objective = _query_credit_objective(task, replay.log_probability, (*slow, *slow))
    combined = torch.autograd.grad(objective, slow, allow_unused=True, retain_graph=True)
    direct = torch.autograd.grad(task + replay.log_probability, slow, allow_unused=True)
    for a, b in zip(combined, direct, strict=True):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b)


def test_credit_recipients_do_not_collapse_clipping_roles():
    query = federation(write=True)
    model = SimpleNamespace(query=query)
    assert _clipping_routing_parameters(model) == ()
    writer = query.candidates[1].operand_store.tensor("rate")
    count = torch.nn.Parameter(torch.tensor(2.0))
    writer.grad = torch.full_like(writer, 3.0 * 8)
    count.grad = torch.tensor(4.0 * 8)
    norms = _clip_training_gradients(
        (writer, count), _clipping_routing_parameters(model), norm=1.0,
        mode="routing-count-task", execution_counts=(count,), gradient_scale=8.0,
    )
    assert norms == {"routing": 0.0, "task": 3.0, "count": 4.0}
    torch.testing.assert_close(writer.grad, torch.ones_like(writer))
    torch.testing.assert_close(count.grad, torch.ones_like(count))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("shared", [False, True])
def test_full_cuda_capture_reuses_graph_but_not_old_bank_responses(dtype, shared):
    from benchmarks._federated_captured_search import captured_search_execution
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    query = federation(write=True, shared=shared).to(device="cuda", dtype=dtype)
    values = {"x": torch.ones(1, 1, device="cuda", dtype=dtype)}
    state = query.initial_bank_state()
    with captured_search_execution(horizon=8, record_effect_operands=True) as backend:
        for _ in range(2):
            actual = search(query, values, state)
            assert not backend.fallbacks
            for branch in actual.branches:
                replay = replay_recursive_graph(query, values, branch.route, bank_state=state,
                                                 use_recorded_choices=True)
                torch.testing.assert_close(branch.log_probability, replay.log_probability)
                torch.testing.assert_close(branch.execution.outputs["result"], replay.execution.outputs["result"])
                assert len(branch.execution.proposals) == 1
                torch.testing.assert_close(branch.execution.bank_state.values, replay.execution.bank_state.values)
            state = actual.branches[0].execution.bank_state
        assert backend.completed == 2 and len(backend.sessions) == 1


def test_mixed_float64_responses_choose_native_before_capture():
    from arti import mechanisms as m
    from benchmarks._federated_captured_search import captured_search_execution
    from test_formula_program_query_v6 import member
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    producer = member("a", "double_input", "a").to(device="cuda", dtype=torch.float64)
    finish = member("finish", "x", "y").cuda()
    query = m.FormulaProgramQueryV6(
        slot_ids=("x", "double_input", "a", "a_negative", "y", "y_negative"),
        candidates=(producer, finish), terminal_slots={"answer": "y"},
        entry_candidates=("a",), continuations={"a": {"finish": "a"}}, max_steps=2,
    ).cuda()
    values = {"x": torch.ones(1, 1, device="cuda"),
              "double_input": torch.ones(1, 1, device="cuda", dtype=torch.float64)}
    initial = query.initial_bank_state()
    expected = search(query, values, initial)
    samples = (values["x"].double(), torch.zeros(1, device="cuda", dtype=torch.float64))
    with captured_search_execution(horizon=8, value_samples=samples) as backend:
        actual = search(query, values, initial)
        assert backend.completed == 0 and not backend.sessions
        assert backend.fallbacks == ["response graph mixed float64 scoring requires native execution"]
    for a, b in zip(actual.branches, expected.branches, strict=True):
        assert a.route == b.route
        torch.testing.assert_close(a.log_probability, b.log_probability, rtol=0, atol=0)
        torch.testing.assert_close(a.execution.outputs["answer"], b.execution.outputs["answer"], rtol=0, atol=0)
