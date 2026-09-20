import pytest
import torch

from arti import mechanisms as m
from arti._formula_device_cooperative import cooperative_sequence_score
from arti._formula_device_response import FormulaDeviceResponse
from benchmarks._federated_device_choice_replay import DeviceChoiceView
from benchmarks._federated_device_product_replay import decode_cooperative_device_products
from benchmarks._federated_endpoint_risk import backward_cooperative_endpoint_risk
from benchmarks._federated_product_replay import ProductReference, replay_cooperative_dependencies_many
from test_formula_completed_products import sharing_graph, private_write_graph
from test_formula_device_cooperative_search import runtime, advance
from test_formula_device_product_replay import collect
from test_formula_program_query_v6 import federation, member
from test_formula_program_query_v7 import cooperative, wrap_child


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    with torch.device(request.param):
        yield request.param


def replay(query, tape, endpoints, x):
    return replay_cooperative_dependencies_many(query, {"x": x}, tape=tape, endpoints=endpoints, score_decisions=True)


@pytest.mark.parametrize("graph", ["effect", "child", "sharing", "private"])
def test_omitted_port_diagnostics_preserve_same_event_replay_and_gradients(device, graph):
    if graph == "sharing":
        query, options = sharing_graph(), {}
    elif graph == "private":
        query, _ = private_write_graph()
        options = {"width": 2, "publish_slots": ("u",)}
    else:
        query = cooperative(federation(write=True, shared=True))
        if graph == "child":
            query = wrap_child(query)
        options = {"width": 3, "product_slots": (), "publish_slots": ()}
    query = query.to(device)
    wave, state, directory, cursor, occurrence, data, bank, dc, bc = runtime(query, **options)
    rounds = []
    for _ in range(8):
        row = advance(wave, state, directory, cursor, occurrence, data, bank, dc, bc)
        rounds.append(row)
        state, directory, cursor, occurrence = row.state, row.directory, row.directory_cursor, row.occurrence_cursor
        dc, bc = row.data_cursor, row.bank_cursor
    initial = (query.initial_bank_state(),)
    full, endpoints = decode_cooperative_device_products(query, rounds,
        initial_states=initial, input_slots=("x",), include_decisions=True)
    lean, lean_endpoints = decode_cooperative_device_products(query, rounds,
        initial_states=initial, input_slots=("x",), include_decisions=True, include_port_inputs=False)
    assert endpoints == lean_endpoints
    assert len(full.occurrences) == len(lean.occurrences)
    assert any(d.device_view.port_inputs for d in full.decisions)
    for left, right in zip(full.decisions, lean.decisions, strict=True):
        assert right.device_view.port_inputs == ()
        assert left.device_view.sources == right.device_view.sources
        assert left.device_view.reference == right.device_view.reference
        assert left.actions == right.actions and left.occurrences == right.occurrences
        for name in ("remaining", "candidates", "score_is_double"):
            assert torch.equal(getattr(left.device_view, name), getattr(right.device_view, name))
    x = torch.ones(1, 1, requires_grad=True)
    parameters = (x, *(p for p in query.parameters() if p.requires_grad))
    expected = replay(query, full, endpoints, x)
    actual = replay(query, lean, lean_endpoints, x)
    for left, right in zip(expected, actual, strict=True):
        assert left.executed_occurrences == right.executed_occurrences
        torch.testing.assert_close(left.decision_energy, right.decision_energy, rtol=0, atol=0)
        for name in left.outputs:
            torch.testing.assert_close(left.outputs[name], right.outputs[name], rtol=0, atol=0)
        for a, b in zip(left.bank_state.values, right.bank_state.values, strict=True):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
    def loss(runs):
        return sum(r.decision_energy + sum(v.square().mean() for v in r.outputs.values())
                   + sum(v.square().mean() for v in r.bank_state.values) for r in runs)
    wanted = torch.autograd.grad(loss(expected), parameters, allow_unused=True)
    gradients = torch.autograd.grad(loss(actual), parameters, allow_unused=True)
    for a, b in zip(wanted, gradients, strict=True):
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_mixed_precision_unused_reduction_cannot_poison_double_gradient(device):
    scores = torch.tensor([[1e100, 0., -1e100], [1., 2., 3.]], dtype=torch.float64, requires_grad=True)
    candidates = torch.tensor([[[0]], [[1]]])
    remaining = torch.ones((2, 1, 1, 3), dtype=torch.bool)
    actual = cooperative_sequence_score(scores, candidates, remaining, torch.tensor([True, False]))[:, 0]
    expected = torch.stack((scores[0, 0] - scores[0].logsumexp(0),
                            (scores[1, 1].float() - scores[1].float().logsumexp(0)).double()))
    torch.testing.assert_close(actual, expected)
    grad, = torch.autograd.grad(actual.sum(), scores, retain_graph=True)
    wanted, = torch.autograd.grad(expected.sum(), scores)
    assert torch.isfinite(grad).all()
    torch.testing.assert_close(grad, wanted)


def test_device_choice_large_double_response_has_finite_source_gradient(device):
    response = FormulaDeviceResponse(action_ids=("emit", "next", "stop"), entry_candidates=("emit",),
                                    continuations={"emit": {"next": "score"}}, width=1).to(device)
    ref = ProductReference(None, "score")
    view = DeviceChoiceView(response=response, sources=(ref,), reference=ref, port_inputs=(),
        remaining=torch.ones((1, 1, 1, 3), dtype=torch.bool), candidates=torch.tensor([[[1]]]),
        score_is_double=torch.tensor([True]), storage_dtype=torch.float64)
    source = torch.tensor([[1e100]], dtype=torch.float64, requires_grad=True)
    energy = view.score(lambda _: (source, None))
    expected = source.sum() - torch.stack((source.new_zeros(()), source.sum(), source.new_zeros(()))).logsumexp(0)
    torch.testing.assert_close(energy, expected)
    grad, = torch.autograd.grad(energy, source, retain_graph=True)
    wanted, = torch.autograd.grad(expected, source)
    assert torch.isfinite(grad).all()
    torch.testing.assert_close(grad, wanted)


@pytest.mark.parametrize("mixed", [False, True])
def test_fixed_presence_compression_preserves_scores_gradients_and_denominator(device, mixed):
    actions = tuple(f"p{i}" for i in range(8)) + ("answer", "stop")
    response = FormulaDeviceResponse(action_ids=actions, entry_candidates=("p0",), width=1,
        continuations={f"p{i}": {"answer": f"score{i}", "stop": f"score{i}"} for i in range(8)}).to(device)
    refs = tuple(ProductReference(None, f"s{i}") if i in (0, 3, 6) else None for i in range(8))
    witness = ProductReference(None, "x0")
    values = [torch.tensor([[number]], dtype=torch.float64 if mixed and i == 1 else torch.float32,
                           requires_grad=True) for i, number in enumerate((1e8, 1., -1e8))]
    products = dict(zip((ref for ref in refs if ref is not None), values, strict=True))
    products[witness] = torch.zeros((1, 1))
    dtype = torch.float64 if mixed else torch.float32
    view = DeviceChoiceView(response, refs, witness, (), torch.ones((1, 1, 1, len(actions)), dtype=torch.bool),
        torch.tensor([[[8]]]), torch.tensor([mixed]), dtype)
    full = torch.stack([torch.zeros(1, dtype=dtype) if ref is None else products[ref].reshape(1).to(dtype)
                        for ref in refs], -1)
    present = torch.tensor([[ref is not None for ref in refs]])
    flags = torch.tensor([[ref is not None and products[ref].dtype == torch.float64 for ref in refs]])
    original, _ = response.response_logits(full, present, torch.zeros(1, dtype=dtype), torch.tensor([False]), flags)
    expected = cooperative_sequence_score(original, view.candidates, view.remaining, view.score_is_double)[0, 0]
    actual = view.score(lambda ref: (products[ref], None))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    gradients = torch.autograd.grad(actual, values, retain_graph=True)
    wanted = torch.autograd.grad(expected, values)
    for gradient, reference in zip(gradients, wanted, strict=True):
        torch.testing.assert_close(gradient, reference, rtol=0, atol=0)
    active, order = view._active_response
    assert len(active) == 3 and order.shape == (3, len(actions))
    assert response.ordered_sources.shape[0] == 8
    assert view._active_response[1] is order
    # Cached metadata must not cache the values produced on a later replay.
    with torch.no_grad():
        values[0].zero_()
    assert view.score(lambda ref: (products[ref], None)) != actual


def native_frontiers(tape, endpoint):
    records = {r.node.occurrence_id: r.node for r in tape.occurrences}
    result = []
    for identity in endpoint.decisions:
        decision = tape.decisions[identity]
        nodes = tuple(records[i] for i in decision.occurrences)
        if not nodes:
            nodes = (m.FormulaProgramGraphNodeV1("stop", (), (), ()),)
        result.append(m.FormulaProgramGraphFrontierV1(nodes, torch.tensor(0.)))
    return tuple(result)


@pytest.mark.parametrize("child", [False, True])
def test_device_decisions_rebuild_native_complete_energy_and_meta_gradient(device, child):
    model = cooperative(federation(write=True, shared=True)).to(device)
    rate = model.candidates[1].operand_store.tensor("rate")
    if child:
        model = wrap_child(model)
    tape, endpoints, _, _ = collect(model, include_decisions=True, product_slots=(), publish_slots=(), width=3)
    x = torch.ones(1, 1, requires_grad=True)
    runs = replay(model, tape, endpoints, x)
    for run, endpoint in zip(runs, endpoints, strict=True):
        native = model.replay({"x": x}, native_frontiers(tape, endpoint), bank_state=tape.initial_states[endpoint.root])
        torch.testing.assert_close(run.decision_energy, native.decision_log_score)
        torch.testing.assert_close(run.outputs["result"], native.outputs["result"])
        loss = run.decision_energy + run.outputs["result"].square().mean() + run.bank_state.values[0].square().mean()
        expected = native.decision_log_score + native.outputs["result"].square().mean() + native.bank_state.values[0].square().mean()
        actual = torch.autograd.grad(loss, (rate, x), retain_graph=True, allow_unused=True)
        wanted = torch.autograd.grad(expected, (rate, x), retain_graph=True, allow_unused=True)
        for a, b in zip(actual, wanted, strict=True):
            if b is None:
                assert a is None
            else:
                torch.testing.assert_close(a, b)


def test_pruned_donor_and_denominator_only_response_keep_separate_credit(device):
    old = sharing_graph()
    query = m.FormulaProgramQueryV7(slot_ids=old.slot_ids, candidates=tuple(old.candidates),
        terminal_slots=old.terminal_slots, entry_candidates=old.entry_candidates,
        continuations={"a_receiver": {"c": "h", "d": "h_negative", "final": "h"}},
        max_steps=old.max_steps, cooperation_width=2).to(device)
    tape, endpoints, old_data, _ = collect(query, include_decisions=True, count=4)
    old_data.fill_(torch.nan)
    x = torch.ones(1, 1, requires_grad=True)
    run, = replay(query, tape, endpoints, x)
    w = query.candidates[0].candidate.operand_store.tensor("weight")
    assert torch.autograd.grad(run.outputs["y"].sum(), w, allow_unused=True, retain_graph=True)[0] is None
    dw, = torch.autograd.grad(run.decision_energy, w, retain_graph=True)
    torch.testing.assert_close(dw, 2 * x * torch.sigmoid(-2 * w * x))
    assert dw.abs().sum() > 0 and run.proposals == ()
    assert len(run.executed_occurrences) == len(set(run.executed_occurrences))


def test_device_choice_bank_reuse_training_matches_direct_fixed_panel(device):
    query = cooperative(federation(write=True, shared=True)).to(device)
    tape, endpoints, _, _ = collect(query, include_decisions=True, product_slots=(), publish_slots=(), width=4)
    x = torch.ones(1, 1)
    runs = replay(query, tape, endpoints, x)
    parameters = tuple(p for p in query.parameters() if p.requires_grad)
    scores = torch.stack([r.decision_energy for r in runs]).double()
    losses = torch.stack([(r.outputs["result"] - .3).square().mean() + .6 * r.bank_state.values[0].square().mean()
                          for r in runs])
    expected = torch.autograd.grad((scores.softmax(0) * losses).sum(), parameters, allow_unused=True)
    report = backward_cooperative_endpoint_risk(query, {"x": x}, tape=tape, endpoints=endpoints,
        answer_loss=lambda outputs: (outputs["result"] - .3).square().mean(),
        readonly_reuse_loss=lambda state: state.values[0].square().mean(), reuse_weight=.6, replay_group_size=1)
    for parameter, value in zip(parameters, expected, strict=True):
        if value is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, value)
    assert torch.isfinite(report.risk)


def test_private_donor_write_is_not_installed_by_choice_replay(device):
    query, write = private_write_graph()
    query = query.to(device)
    tape, endpoints, _, _ = collect(query, include_decisions=True, width=2, publish_slots=("u",))
    endpoint = next(e for e in endpoints if not e.writes)
    x = torch.ones(1, 1, requires_grad=True)
    run, = replay(query, tape, (endpoint,), x)
    assert run.proposals == () and run.bank_state.values[0] is tape.initial_states[0].values[0]
    rate = write.operand_store.tensor("rate")
    gradient, = torch.autograd.grad(run.outputs["y"].sum(), rate)
    torch.testing.assert_close(gradient, 10 * x.square())


def test_alias_port_views_keep_exact_sources_in_device_choice_tape(device):
    left = member("left", "x", "p", weight=2.)
    right = member("right", "x", "q", weight=5.)
    kind = m.TensorType(("B", "D"), ("B", 1), dtype="floating", domain="activation")
    a, b = m.InputBinding("a", kind), m.InputBinding("b", kind)
    coefficient = m.BankBinding("c", "arti/device-choice@1", "c", kind)
    program = m.FormulaProgram.build(outputs=(m.add(a, m.scale(b, coefficient)),))
    base = m.FormulaProgramTensorCandidateV4(m.FormulaProgramCandidateV3(
        "subtract", program, input_slots={"a": "shared", "b": "shared"},
        output_slots={program.outputs[0]: "y"}, operands={"c": torch.tensor([[-3.]])}))
    query = m.FormulaProgramQueryV7(
        slot_ids=("x", "shared", "d0", "d1", "p", "p_negative", "q", "q_negative", "y"),
        candidates=(left, right, base), terminal_slots={"y": "y"}, entry_candidates=("left", "right"),
        continuations={"left": {"subtract": "p"}}, max_steps=3, cooperation_width=2).to(device)
    names = tuple(base.input_slots)
    tape, endpoints, _, _ = collect(query, include_decisions=True, count=4,
        product_slots=("d0", "d1"), publish_slots=("p", "q"),
        product_bindings={("subtract", names[0]): "d0", ("subtract", names[1]): "d1"})
    x = torch.ones(1, 1, requires_grad=True)
    run, = replay(query, tape, endpoints, x)
    torch.testing.assert_close(run.outputs["y"], -13 * x)
    decision = next(d for d in tape.decisions if d.actions == ("subtract",))
    ports = {name: ref for candidate, name, ref in decision.device_view.port_inputs if candidate == "subtract"}
    assert ports[names[0]] != ports[names[1]]
    assert "shared" not in dict(decision.products)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_capture_choice_metadata_owns_masks_and_has_no_old_payload_dependency():
    with torch.device("cuda"):
        child = cooperative(federation(write=True, shared=True))
        query = wrap_child(child).cuda()
        wave, initial, empty, zero, first, data, bank, initial_dc, initial_bc = runtime(
            query, product_slots=(), publish_slots=())

        def run():
            state, directory, cursor, occurrence = initial, empty, zero, first
            dc, bc = initial_dc, initial_bc
            records = []
            for _ in range(3):
                result = advance(wave, state, directory, cursor, occurrence, data, bank, dc, bc)
                records.append(result)
                state, directory, cursor, occurrence = (result.state, result.directory,
                                                        result.directory_cursor, result.occurrence_cursor)
                dc, bc = result.data_cursor, result.bank_cursor
            return records

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        capture = torch.cuda.CUDAGraph()
        with torch.cuda.graph(capture):
            records = run()
        capture.replay()
        tape, endpoints = decode_cooperative_device_products(query, records,
            initial_states=(query.initial_bank_state(),), input_slots=("x",), include_decisions=True)
        data.fill_(torch.nan)
        bank.fill_(torch.nan)
        for record in records:
            record.remaining.zero_()
            record.candidates.fill_(-1)
        x = torch.tensor([[2.]], requires_grad=True)
        rebuilt, = replay(query, tape, endpoints, x)
        expected = query.replay({"x": x}, native_frontiers(tape, endpoints[0]), bank_state=tape.initial_states[0])
        torch.testing.assert_close(rebuilt.decision_energy, expected.decision_log_score)
        rate = child.candidates[1].operand_store.tensor("rate")
        left = rebuilt.decision_energy + rebuilt.outputs["result"].square().sum() + rebuilt.bank_state.values[0].square().sum()
        right = expected.decision_log_score + expected.outputs["result"].square().sum() + expected.bank_state.values[0].square().sum()
        a = torch.autograd.grad(left, (x, rate))
        b = torch.autograd.grad(right, (x, rate))
        for actual, wanted in zip(a, b, strict=True):
            torch.testing.assert_close(actual, wanted)
