import pytest
import torch

from arti import mechanisms as m
from benchmarks._federated_device_product_replay import decode_cooperative_device_products
from benchmarks._federated_product_replay import replay_cooperative_dependencies
from test_formula_completed_products import private_write_graph, sharing_graph
from test_formula_device_cooperative_search import runtime, advance
from test_formula_program_query_v6 import federation, member


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    with torch.device(request.param):
        yield request.param


def collect(query, *, count=8, include_decisions=False, **kwargs):
    wave, state, directory, cursor, occurrence, data, bank, dc, bc = runtime(query, **kwargs)
    rounds = []
    for _ in range(count):
        result = advance(wave, state, directory, cursor, occurrence, data, bank, dc, bc)
        assert not result.requires_fallback
        rounds.append(result)
        state, directory, cursor, occurrence = (result.state, result.directory,
                                                result.directory_cursor, result.occurrence_cursor)
        dc, bc = result.data_cursor, result.bank_cursor
    tape, endpoints = decode_cooperative_device_products(query, rounds,
        initial_states=(query.initial_bank_state(),), input_slots=("x",), include_decisions=include_decisions)
    return tape, endpoints, data, state


def test_device_tape_replays_pruned_donor_once_and_sums_consumer_gradients(device):
    query = sharing_graph().to(device)
    tape, endpoints, data, state = collect(query, count=4)
    x = torch.tensor([[3.]], requires_grad=True)
    calls = []
    handles = [c.register_forward_hook(lambda c, args, out: calls.append(c.candidate_id)) for c in query.candidates]
    try:
        result = replay_cooperative_dependencies(query, {"x": x}, tape=tape, endpoint=endpoints[0])
    finally:
        for h in handles:
            h.remove()
    assert calls == ["z_donor", "c", "d", "final"]
    assert len(tape.occurrences) == 5 and len(result.executed_occurrences) == 4
    torch.testing.assert_close(result.outputs["y"], 20 * x)
    dw, dx = torch.autograd.grad(result.outputs["y"].sum(),
        (query.candidates[1].candidate.operand_store.tensor("weight"), x))
    torch.testing.assert_close(dw, 5 * x)
    torch.testing.assert_close(dx, torch.full_like(x, 20))
    assert result.proposals == ()
    assert data[state.frames.value_handles[1, 0, query.slot_ids.index("y")]] == 40


def test_device_tape_private_effect_is_numeric_ancestor_not_endpoint_write(device):
    query, write = private_write_graph()
    query = query.to(device)
    tape, endpoints, _, _ = collect(query, width=2, publish_slots=("u",))
    x = torch.tensor([[.75]], requires_grad=True)
    result = replay_cooperative_dependencies(query, {"x": x}, tape=tape, endpoint=endpoints[0])
    rate = write.operand_store.tensor("rate")
    expected = 5 * x * (1 + 2 * x * rate)
    torch.testing.assert_close(result.outputs["y"], expected)
    actual = torch.autograd.grad(result.outputs["y"].sum(), (rate, x), create_graph=True)
    wanted = torch.autograd.grad(expected.sum(), (rate, x), create_graph=True)
    for a, b in zip(actual, wanted, strict=True):
        torch.testing.assert_close(a, b)
    mixed, = torch.autograd.grad(actual[0].sum(), (x,))
    torch.testing.assert_close(mixed, 20 * x)
    assert endpoints[0].writes == () and result.proposals == ()
    assert result.bank_state.values[0] is tape.initial_states[0].values[0]
    assert any(record.bank_writes for record in tape.occurrences)


def test_same_local_slot_distinct_named_sources_survive_device_replay(device):
    left, right = member("left", "x", "l", weight=2.), member("right", "x", "r", weight=5.)
    kind = m.TensorType(("B", "D"), ("B", 1), dtype="floating", domain="activation")
    a, b = m.InputBinding("a", kind), m.InputBinding("b", kind)
    coefficient = m.BankBinding("c", "arti/device-source-replay@1", "c", kind)
    program = m.FormulaProgram.build(outputs=(m.add(a, m.scale(b, coefficient)),))
    join = m.FormulaProgramTensorCandidateV4(m.FormulaProgramCandidateV3(
        "join", program, input_slots={"b": "shared", "a": "shared"},
        output_slots={program.outputs[0]: "y"}, operands={"c": torch.tensor([[-3.]])}))
    query = m.FormulaProgramQueryV7(
        slot_ids=("x", "l", "l_negative", "r", "r_negative", "shared", "shared_right", "y"),
        candidates=(left, right, join), terminal_slots={"y": "y"}, entry_candidates=("left", "right"),
        continuations={"left": {"join": "l"}}, max_steps=3, cooperation_width=2).to(device)
    tape, endpoints, data, state = collect(query, count=3,
        product_slots=("shared", "shared_right"), publish_slots=("l", "r"),
        product_bindings={("join", "b"): "shared_right", ("join", "a"): "shared"})
    assert data[state.frames.value_handles[1, 0, query.slot_ids.index("y")]] == -26
    record = next(r for r in tape.occurrences if r.node.candidate_id == "join")
    assert tuple(name for name, _ in record.inputs) == ("b", "a")
    assert record.inputs[0][1].occurrence != record.inputs[1][1].occurrence
    x = torch.tensor([[1.5]], requires_grad=True)
    result = replay_cooperative_dependencies(query, {"x": x}, tape=tape, endpoint=endpoints[0])
    torch.testing.assert_close(result.outputs["y"], -13 * x)
    dx, dl, dr = torch.autograd.grad(result.outputs["y"].sum(),
        (x, left.candidate.operand_store.tensor("weight"), right.candidate.operand_store.tensor("weight")))
    torch.testing.assert_close(dx, torch.full_like(x, -13))
    torch.testing.assert_close(dl, x)
    torch.testing.assert_close(dr, -3 * x)


def test_device_tape_retains_real_writer_and_its_meta_gradient(device):
    old = federation(write=True, shared=True)
    query = m.FormulaProgramQueryV7(slot_ids=old.slot_ids, candidates=tuple(old.candidates),
        terminal_slots=old.terminal_slots, entry_candidates=old.entry_candidates,
        continuations=old.continuations, cooperation_width=2, max_steps=4).to(device)
    tape, endpoints, _, _ = collect(query, product_slots=(), publish_slots=(), width=2)
    endpoint = next(e for e in endpoints if e.writes)
    x = torch.tensor([[1.25]], requires_grad=True)
    result = replay_cooperative_dependencies(query, {"x": x}, tape=tape, endpoint=endpoint)
    assert len(result.proposals) == 1
    assert result.bank_state.revisions[0] == tape.initial_states[0].revisions[0] + 1
    assert result.bank_state.values[0] is not tape.initial_states[0].values[0]
    write = next(c for c in query.candidates if isinstance(c, m.FormulaProgramEffectCandidateV3))
    rate = write.operand_store.tensor("rate")
    expected = tape.initial_states[0].values[0] + 2 * x * rate
    torch.testing.assert_close(result.bank_state.values[0], expected)
    # Endpoint reuse credit must reach the writing law, not an optimizer-owned fast Bank.
    actual = torch.autograd.grad(result.bank_state.values[0].sum(), (rate, x), create_graph=True)
    wanted = torch.autograd.grad(expected.sum(), (rate, x), create_graph=True)
    for a, b in zip(actual, wanted, strict=True):
        torch.testing.assert_close(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_captured_search_metadata_rebuilds_fresh_differentiable_values():
    with torch.device("cuda"):
        query = sharing_graph().cuda()
        wave, initial, empty, zero, first, data, bank, initial_dc, initial_bc = runtime(query)

        def run():
            state, directory, cursor, occurrence = initial, empty, zero, first
            dc, bc = initial_dc, initial_bc
            records = []
            for _ in range(4):
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
            rounds = run()
        data[0].fill_(4.)
        capture.replay()
        tape, endpoints = decode_cooperative_device_products(query, rounds,
            initial_states=(query.initial_bank_state(),), input_slots=("x",))
        data.fill_(float("nan"))  # The recorded device payload is not a replay input.
        x = torch.tensor([[.5]], requires_grad=True)
        result = replay_cooperative_dependencies(query, {"x": x}, tape=tape, endpoint=endpoints[0])
        torch.testing.assert_close(result.outputs["y"], 20 * x)
        dx, = torch.autograd.grad(result.outputs["y"].sum(), (x,))
        torch.testing.assert_close(dx, torch.full_like(x, 20))


def test_device_tape_equal_revision_roots_keep_their_own_values(device):
    from dataclasses import replace
    from benchmarks._federated_product_replay import replay_cooperative_dependencies_many

    producer = member("producer", "x", "p", owner="memory")
    left = member("left", "shared0", "y")
    right = left.with_bindings("right", input_slots={"x": "shared1"}, output_slots=left.candidate.output_slots)
    query = m.FormulaProgramQueryV7(
        slot_ids=("x", "p", "p_negative", "shared0", "shared1", "y", "y_negative"),
        candidates=(producer, left, right), terminal_slots={"y": "y"}, entry_candidates=("producer",),
        continuations={"producer": {"left": "p", "right": "p_negative"}}, max_steps=2,
        cooperation_width=2).to(device)
    wave, state, directory, cursor, occurrence, data, bank, dc, bc = runtime(
        query, width=2, heads=1, product_slots=("shared0", "shared1"), publish_slots=("p",))
    initial = query.initial_bank_state()
    initial_states = (initial, replace(initial, values=(-initial.values[0],)))
    state.frames.active[1] = True
    state.frames.bank_value_handles[1, 0] = 1
    bank[1] = -bank[0]
    bc.fill_(2)
    rounds = []
    for _ in range(3):
        result = advance(wave, state, directory, cursor, occurrence, data, bank, dc, bc)
        rounds.append(result)
        state, directory, cursor, occurrence = (result.state, result.directory,
                                                result.directory_cursor, result.occurrence_cursor)
        dc, bc = result.data_cursor, result.bank_cursor
    tape, endpoints = decode_cooperative_device_products(query, rounds,
        initial_states=initial_states, input_slots=("x",))
    fast = (torch.tensor([[5.]], requires_grad=True), torch.tensor([[-7.]], requires_grad=True))
    fresh = tuple(replace(s, values=(v,)) for s, v in zip(initial_states, fast, strict=True))
    x = torch.tensor([[3.]], requires_grad=True)
    results = replay_cooperative_dependencies_many(query, {"x": x}, tape=tape,
        endpoints=endpoints, initial_states=fresh)
    assert {endpoint.root for endpoint in endpoints} == {0, 1}
    for result, endpoint in zip(results, endpoints, strict=True):
        torch.testing.assert_close(result.outputs["y"], fast[endpoint.root] * x)
        assert result.bank_state.values[0] is fast[endpoint.root]
    loss = sum((e.root + 1) * r.outputs["y"].sum() for e, r in zip(endpoints, results, strict=True))
    gradients = torch.autograd.grad(loss, fast)
    torch.testing.assert_close(gradients[0], x)
    torch.testing.assert_close(gradients[1], 2 * x)
