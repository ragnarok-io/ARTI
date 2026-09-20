import pytest
import torch

from arti import mechanisms as m
from benchmarks._federated_endpoint_risk import endpoint_risk_coefficients, backward_cooperative_endpoint_risk
from benchmarks._federated_product_replay import replay_cooperative_dependencies_many
from benchmarks._federated_recursive_search import search_cooperative_graphs, start_recursive_search
from test_formula_completed_products import private_write_graph, sharing_graph
from test_formula_program_query_v6 import federation, member
from test_formula_program_query_v7 import cooperative, wrap_child


def search(model, x, **kwargs):
    with torch.no_grad():
        return search_cooperative_graphs((start_recursive_search(model, {"x": x}),),
                                         product_slots=kwargs.pop("product_slots", ()), width=4, beam_width=4, **kwargs)


def rebuild(model, x, result, **kwargs):
    return replay_cooperative_dependencies_many(model, {"x": x}, tape=result.dependency_tape,
        endpoints=tuple(b.dependency_endpoint for b in result.branches), score_decisions=True, **kwargs)


@pytest.mark.parametrize("temperature", [0.25, 1.0, 3.0])
def test_fixed_panel_grouped_gradient_matches_direct_risk(temperature):
    parameter = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    scores = torch.stack((parameter, -2 * parameter, parameter.square()))
    losses = torch.stack(((parameter - 1).square(), (parameter + 2).square(), parameter.exp()))
    direct = (torch.softmax(scores / temperature, 0) * losses).sum()
    coefficients = endpoint_risk_coefficients(scores, losses, temperature=temperature)
    surrogate = coefficients.surrogate(losses[:2], scores[:2], indices=[0, 1])
    surrogate = surrogate + coefficients.surrogate(losses[2:], scores[2:], indices=[2])
    expected, = torch.autograd.grad(direct, parameter, retain_graph=True)
    actual, = torch.autograd.grad(surrogate, parameter)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(coefficients.risk, direct)
    assert not coefficients.numerical.requires_grad and not coefficients.choice.requires_grad


def test_shared_energy_shift_and_singleton_have_zero_choice_credit():
    scores = torch.tensor([1., 2., 3.], requires_grad=True)
    losses = torch.tensor([2., 0.5, 4.], requires_grad=True)
    common = torch.tensor(7., requires_grad=True)
    coefficients = endpoint_risk_coefficients(scores + common, losses)
    gradient, = torch.autograd.grad(coefficients.surrogate(losses, scores + common), common)
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-6, rtol=0)
    one = endpoint_risk_coefficients(scores[:1], losses[:1])
    assert one.choice.item() == 0
    equal = endpoint_risk_coefficients(scores, torch.ones_like(scores))
    torch.testing.assert_close(equal.choice, torch.zeros_like(equal.choice), atol=1e-15, rtol=0)


@pytest.mark.parametrize("child", [False, True])
@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_complete_native_energy_loss_and_bank_meta_gradient(child, device):
    model = cooperative(federation(write=True, shared=True)).to(device)
    rate = model.candidates[1].operand_store.tensor("rate")
    if child:
        model = wrap_child(model)
    x = torch.ones(1, 1, device=device, requires_grad=True)
    result = search(model, x)
    replayed = rebuild(model, x, result)
    natives = [model.replay({"x": x}, b.execution.frontiers,
                           bank_state=result.dependency_tape.initial_states[0]) for b in result.branches]
    energies = torch.stack([r.decision_energy for r in replayed])
    expected_scores = torch.stack([n.decision_log_score for n in natives])
    torch.testing.assert_close(energies, expected_scores)

    def loss(run):
        return (run.outputs["result"] - 0.7).square().mean() + 0.4 * (run.bank_state.values[0] - 0.2).square().mean()

    losses, native_losses = torch.stack([loss(r) for r in replayed]), torch.stack([loss(n) for n in natives])
    coefficients = endpoint_risk_coefficients(energies, losses, temperature=0.7)
    proxy = coefficients.surrogate(losses, energies)
    direct = (torch.softmax(expected_scores.double() / 0.7, 0) * native_losses.double()).sum()
    actual = torch.autograd.grad(proxy, (rate, x))
    expected = torch.autograd.grad(direct, (rate, x))
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b)
    assert actual[0].abs().sum() > 0
    fast = result.dependency_tape.initial_states[0].values
    assert all(not any(v is p for p in model.parameters()) for v in fast)


def test_donor_response_dependencies_share_replay_but_do_not_commit_donor():
    model, write = private_write_graph()
    x = torch.ones(1, 1, requires_grad=True)
    result = search(model, x, product_slots=("shared",), publish_slots=("u",))
    receiver = next(b for b in result.branches if not b.dependency_endpoint.writes)
    calls = []
    handles = [c.register_forward_hook(lambda c, args, out: calls.append(c.candidate_id)) for c in model.candidates]
    replayed, = replay_cooperative_dependencies_many(model, {"x": x}, tape=result.dependency_tape,
        endpoints=(receiver.dependency_endpoint,), score_decisions=True)
    for handle in handles:
        handle.remove()
    assert replayed.proposals == ()
    assert replayed.bank_state.values[0] is result.dependency_tape.initial_states[0].values[0]
    assert len(replayed.scored_decisions) == len(set(replayed.scored_decisions))
    assert calls.count("write") == 1
    assert torch.autograd.grad(replayed.outputs["y"].sum(), write.operand_store.tensor("rate"),
                               retain_graph=True)[0].abs().sum() > 0
    assert replayed.decision_energy.requires_grad


def test_denominator_keeps_response_producer_absent_from_answer_ancestry():
    original = sharing_graph()
    model = m.FormulaProgramQueryV7(
        slot_ids=original.slot_ids, candidates=tuple(original.candidates), terminal_slots=original.terminal_slots,
        entry_candidates=original.entry_candidates,
        continuations={"a_receiver": {"c": "h", "d": "h_negative", "final": "h"}},
        max_steps=original.max_steps, cooperation_width=original.cooperation_width)
    x = torch.ones(1, 1, requires_grad=True)
    result = search(model, x, product_slots=("shared",), publish_slots=("p",))
    runs = rebuild(model, x, result)
    weight = model.candidates[0].candidate.operand_store.tensor("weight")
    for run in runs:
        assert torch.autograd.grad(run.outputs["y"].sum(), weight, retain_graph=True, allow_unused=True)[0] is None
        gradient, = torch.autograd.grad(run.decision_energy, weight, retain_graph=True)
        torch.testing.assert_close(gradient, 2 * x * torch.sigmoid(-2 * weight * x))
        assert gradient.abs().sum() > 0


def test_numeric_only_device_tape_cannot_silently_supply_choice_credit():
    from test_formula_device_product_replay import collect
    model = cooperative(federation())
    tape, endpoints, _, _ = collect(model, product_slots=(), publish_slots=())
    with pytest.raises(ValueError, match="full decision dependencies"):
        replay_cooperative_dependencies_many(model, {"x": torch.ones(1, 1)}, tape=tape,
                                             endpoints=endpoints, score_decisions=True)


def test_completed_call_response_visibility_matches_native_denominator():
    child = cooperative(federation(write=True, shared=True))
    child = m.FormulaProgramQueryV7(
        slot_ids=child.slot_ids, candidates=tuple(child.candidates),
        terminal_slots={"result": "y", "negative": "y_negative"},
        entry_candidates=child.entry_candidates, continuations=child.continuations,
        max_steps=child.max_steps, cooperation_width=child.cooperation_width)
    call = wrap_child(child).candidates[0]
    returned = next(iter(call.output_slots.values()))
    left = member("left", returned, "answer", weight=1.3)
    right = member("right", returned, "answer", weight=-0.7)
    model = m.FormulaProgramQueryV7(
        slot_ids=tuple(dict.fromkeys(("x", *call.output_slot_ids, "answer", "answer_negative"))),
        candidates=(call, left, right), terminal_slots={"y": "answer"}, entry_candidates=(call.candidate_id,),
        continuations={call.candidate_id: {"left": returned, "right": call.output_slots["negative"]}},
        max_steps=2, cooperation_width=2)
    x = torch.ones(1, 1, requires_grad=True)
    result = search(model, x)
    runs = rebuild(model, x, result)
    rate = child.candidates[1].operand_store.tensor("rate")
    for branch, run in zip(result.branches, runs, strict=True):
        native = model.replay({"x": x}, branch.execution.frontiers,
                              bank_state=result.dependency_tape.initial_states[0])
        torch.testing.assert_close(run.decision_energy, native.decision_log_score)
        a = torch.autograd.grad(run.decision_energy, (x, rate), retain_graph=True)
        b = torch.autograd.grad(native.decision_log_score, (x, rate), retain_graph=True)
        for actual, expected in zip(a, b, strict=True):
            torch.testing.assert_close(actual, expected)
        assert a[0].abs().sum() > 0


def test_serial_child_has_no_silent_complete_choice_credit():
    model = wrap_child(federation(write=True, shared=True))
    x = torch.ones(1, 1)
    result = search(model, x)
    with pytest.raises(NotImplementedError, match="V7 child decisions"):
        rebuild(model, x, result)


@pytest.mark.parametrize("group_size", [1, 2, 8])
def test_two_pass_training_accumulates_all_slow_gradients_without_install(group_size):
    model = cooperative(federation(write=True, shared=True)).double()
    x = torch.ones(1, 1, dtype=torch.float64)
    result = search(model, x)
    initial = result.dependency_tape.initial_states[0]
    owner_before = tuple(v.clone() for v in initial.values)
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    def answer(outputs):
        return (outputs["result"] - 0.4).square().mean()

    def reuse(state):
        return (state.values[0] - 0.8).square().mean()
    runs = rebuild(model, x, result)
    scores = torch.stack([r.decision_energy for r in runs]).double()
    losses = torch.stack([answer(r.outputs) + 0.6 * reuse(r.bank_state) for r in runs])
    direct = (torch.softmax(scores / 0.9, 0) * losses).sum()
    expected = torch.autograd.grad(direct, parameters, allow_unused=True)
    report = backward_cooperative_endpoint_risk(model, {"x": x}, tape=result.dependency_tape,
        endpoints=tuple(b.dependency_endpoint for b in result.branches), answer_loss=answer,
        readonly_reuse_loss=reuse, reuse_weight=0.6, temperature=0.9, replay_group_size=group_size)
    torch.testing.assert_close(report.risk, direct)
    for parameter, gradient in zip(parameters, expected, strict=True):
        if gradient is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, gradient)
    for old, value in zip(owner_before, model.initial_bank_state().values, strict=True):
        torch.testing.assert_close(value, old, rtol=0, atol=0)
    assert report.replayed_occurrences > 0 and not report.risk.requires_grad


def test_grouped_replay_preserves_incoming_cross_event_meta_gradient():
    from dataclasses import replace

    model = cooperative(federation(write=True, shared=True)).double()
    previous_law = torch.tensor(0.2, dtype=torch.float64, requires_grad=True)
    x = torch.ones(1, 1, dtype=torch.float64)
    state = model.initial_bank_state()
    state = replace(state, values=tuple(v + previous_law.square() for v in state.values))
    with torch.no_grad():
        result = search_cooperative_graphs((start_recursive_search(model, {"x": x}, bank_state=state),),
                                           product_slots=(), width=4, beam_width=4)
    assert len(result.branches) > 1
    runs = rebuild(model, x, result)
    scores = torch.stack([r.decision_energy for r in runs]).double()
    losses = torch.stack([(r.outputs["result"] - 0.4).square().mean() + r.bank_state.values[0].square().mean()
                          for r in runs])
    expected, = torch.autograd.grad((scores.softmax(0) * losses).sum(), previous_law, retain_graph=True)
    backward_cooperative_endpoint_risk(model, {"x": x}, tape=result.dependency_tape,
        endpoints=tuple(b.dependency_endpoint for b in result.branches),
        answer_loss=lambda outputs: (outputs["result"] - 0.4).square().mean(),
        readonly_reuse_loss=lambda state: state.values[0].square().mean(),
        reuse_weight=1.0, replay_group_size=1)
    torch.testing.assert_close(previous_law.grad, expected)
