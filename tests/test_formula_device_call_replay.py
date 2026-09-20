import pytest
import torch

from arti import mechanisms as m
from benchmarks._federated_device_product_replay import decode_cooperative_device_products
from benchmarks._federated_product_replay import replay_cooperative_dependencies
from test_formula_device_product_replay import collect
from test_formula_device_cooperative_search import runtime, advance
from test_formula_program_query_v6 import federation, member, effect
from test_formula_program_query_v7 import cooperative, graph, wrap_child, join


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    with torch.device(request.param):
        yield request.param


@pytest.mark.parametrize("middle_kind", [m.FormulaProgramQueryV5, m.FormulaProgramQueryV6, m.FormulaProgramQueryV7])
def test_device_nested_trace_matches_native_and_replays_current_values(device, middle_kind):
    leaf = graph()
    query = wrap_child(wrap_child(leaf, middle_kind)).to(device)
    tape, endpoints, data, _state = collect(query, count=3, product_slots=(), publish_slots=())
    expected = query({"x": data[0]})
    record, = tape.occurrences
    assert record.node.child_trace == expected.trace.steps[0].child_trace
    assert record.node.child_trace.invocation_path == ("child",)
    assert record.node.child_trace.steps[0].child_trace.invocation_path == ("child", "child")
    x = torch.tensor([[3.]], requires_grad=True)
    calls = []
    hook = leaf.register_forward_hook(lambda *_: calls.append(1))
    try:
        replay = replay_cooperative_dependencies(query, {"x": x}, tape=tape, endpoint=endpoints[0])
    finally:
        hook.remove()
    assert calls == [1] and replay.executed_occurrences == (record.node.occurrence_id,)
    native = query({"x": x})
    torch.testing.assert_close(replay.outputs["y"], native.outputs["y"])
    parameters = (x, *leaf.parameters())
    actual = torch.autograd.grad(replay.outputs["y"].sum(), parameters, allow_unused=True)
    wanted = torch.autograd.grad(native.outputs["y"].sum(), parameters, allow_unused=True)
    for a, b in zip(actual, wanted, strict=True):
        if b is None:
            assert a is None
        else:
            torch.testing.assert_close(a, b)


def test_device_call_replay_does_not_query_a_new_route(device):
    leaf = cooperative(federation())
    query = wrap_child(leaf).to(device)
    tape, endpoints, data, _state = collect(query, count=3, product_slots=(), publish_slots=())
    data.fill_(torch.nan)
    x = torch.tensor([[-1.]], requires_grad=True)
    replay = replay_cooperative_dependencies(query, {"x": x}, tape=tape, endpoint=endpoints[0])
    fresh = query({"x": x})
    assert tape.occurrences[0].node.child_trace.steps[-2].candidate_id == "left"
    assert fresh.trace.steps[0].child_trace.steps[-2].candidate_id == "right"
    torch.testing.assert_close(replay.outputs["result"], 2 * x)
    torch.testing.assert_close(fresh.outputs["result"], -2 * x)
    weight = next(c for c in leaf.candidates if c.candidate_id == "left").candidate.operand_store.tensor("weight")
    dx, dw = torch.autograd.grad(replay.outputs["result"].sum(), (x, weight))
    torch.testing.assert_close(dx, torch.full_like(x, 2.))
    torch.testing.assert_close(dw, x)


def test_device_shared_child_calls_replay_ordered_bank_updates_and_meta_gradients(device):
    child = cooperative(federation(write=True, shared=True))
    calls = tuple(m.FormulaProgramCallCandidateV1(name, child, input_slots={"x": "x"},
        output_slots={"result": name + "_out"}) for name in ("first", "second"))
    query = m.FormulaProgramQueryV7(slot_ids=("x", "first_out", "second_out"), candidates=calls,
        terminal_slots={"y": "second_out"}, entry_candidates=("first",),
        continuations={"first": {"second": "first_out"}}, max_steps=2, cooperation_width=4).to(device)
    tape, endpoints, data, _state = collect(query, count=4, product_slots=(), publish_slots=())
    native = query({"x": data[0]})
    assert [r.node.child_trace for r in tape.occurrences] == [s.child_trace for s in native.trace.steps[:-1]]
    assert len(endpoints[0].writes) == 2
    x = torch.tensor([[.75]], requires_grad=True)
    replay = replay_cooperative_dependencies(query, {"x": x}, tape=tape, endpoint=endpoints[0])
    expected = query.replay({"x": x}, native.frontiers)
    assert len(replay.proposals) == 2
    assert replay.proposals[1].previous is replay.proposals[0].successor
    assert replay.bank_state.revisions[0] == 2
    torch.testing.assert_close(replay.outputs["y"], expected.outputs["y"])
    torch.testing.assert_close(replay.bank_state.values[0], expected.bank_state.values[0])
    rate = next(c for c in child.candidates if c.candidate_id == "write").operand_store.tensor("rate")
    actual = torch.autograd.grad((replay.outputs["y"] + replay.bank_state.values[0]).sum(), (rate, x), create_graph=True)
    wanted = torch.autograd.grad((expected.outputs["y"] + expected.bank_state.values[0]).sum(), (rate, x), create_graph=True)
    for a, b in zip(actual, wanted, strict=True):
        torch.testing.assert_close(a, b)
    a, = torch.autograd.grad(actual[0].sum(), x)
    b, = torch.autograd.grad(wanted[0].sum(), x)
    torch.testing.assert_close(a, b)


def test_pruned_multihead_call_replayed_once_without_retaining_donor_writes(device):
    original = federation(write=True, shared=True)
    a, write, read = original.candidates[:3]
    child = m.FormulaProgramQueryV7(slot_ids=("x", "h", "h_negative", "tail", "u", "u_negative"),
        candidates=(a, write, read), terminal_slots={"positive": "u", "negative": "u_negative"},
        entry_candidates=("a",), continuations={"a": {"write": "h", "b": "h"}},
        max_steps=3, cooperation_width=2)
    receiver = member("a_receiver", "x", "h")
    receiver.candidate.requires_empty_slots = ("p",)
    donor = m.FormulaProgramCallCandidateV1("z_donor", child, input_slots={"x": "x"},
        output_slots={"positive": "p", "negative": "q"}, requires_empty_slots=("h",))
    c, d = member("c", "shared", "u", weight=2.), member("d", "shared_negative", "v", weight=3.)
    candidates = (receiver, donor, c, d, join("final", "u", "v", "y"))
    query = m.FormulaProgramQueryV7(
        slot_ids=("x", "shared", "shared_negative", *(s for c in candidates for s in c.output_slot_ids)),
        candidates=candidates, terminal_slots={"y": "y"}, entry_candidates=("a_receiver", "z_donor"),
        continuations={"a_receiver": {"c": "h", "d": "h", "final": "h"}},
        max_steps=4, cooperation_width=2).to(device)
    tape, endpoints, data, _state = collect(query, count=4,
        product_slots=("shared", "shared_negative"), publish_slots=("p", "q"))
    donor_record = next(r for r in tape.occurrences if r.node.candidate_id == "z_donor")
    assert donor_record.bank_writes and not endpoints[0].writes
    data.fill_(torch.nan)
    x = torch.tensor([[.75]], requires_grad=True)
    calls = []
    hook = child.register_forward_hook(lambda *_: calls.append(1))
    try:
        replay = replay_cooperative_dependencies(query, {"x": x}, tape=tape, endpoint=endpoints[0])
    finally:
        hook.remove()
    assert calls == [1]
    assert replay.proposals == () and replay.bank_state.values[0] is tape.initial_states[0].values[0]
    rate = write.operand_store.tensor("rate")
    expected = -x * (1 + 2 * x * rate)
    torch.testing.assert_close(replay.outputs["y"], expected)
    actual = torch.autograd.grad(replay.outputs["y"].sum(), (rate, x), create_graph=True)
    wanted = torch.autograd.grad(expected.sum(), (rate, x), create_graph=True)
    for a, b in zip(actual, wanted, strict=True):
        torch.testing.assert_close(a, b)
    mixed, = torch.autograd.grad(actual[0].sum(), x)
    torch.testing.assert_close(mixed, -4 * x)


def test_call_named_ports_sharing_one_local_slot_keep_distinct_values_and_lineage(device):
    kind = m.TensorType(("B", "D"), ("B", 1), dtype="floating", domain="activation")
    a, b = m.InputBinding("a", kind), m.InputBinding("b", kind)
    coefficient = m.BankBinding("c", "arti/device-call-replay@1", "c", kind)
    program = m.FormulaProgram.build(outputs=(m.add(a, m.scale(b, coefficient)),))
    operation = m.FormulaProgramTensorCandidateV4(m.FormulaProgramCandidateV3(
        "join", program, input_slots={"b": "right", "a": "left"},
        output_slots={program.outputs[0]: "y"}, operands={"c": torch.tensor([[-3.]])}))
    child = m.FormulaProgramQueryV7(slot_ids=("left", "right", "y"), candidates=(operation,),
        terminal_slots={"result": "y"}, entry_candidates=("join",), continuations={}, max_steps=1)
    left, right = member("left", "x", "l", weight=2.), member("right", "x", "r", weight=5.)
    call = m.FormulaProgramCallCandidateV1("call", child,
        input_slots={"right": "shared", "left": "shared"}, output_slots={"result": "y"})
    query = m.FormulaProgramQueryV7(
        slot_ids=("x", "l", "l_negative", "r", "r_negative", "shared", "shared_right", "y"),
        candidates=(left, right, call), terminal_slots={"y": "y"}, entry_candidates=("left", "right"),
        continuations={"left": {"call": "l"}}, max_steps=3, cooperation_width=2).to(device)
    tape, endpoints, data, state = collect(query, count=3,
        product_slots=("shared", "shared_right"), publish_slots=("l", "r"),
        product_bindings={("call", "right"): "shared_right", ("call", "left"): "shared"})
    assert data[state.frames.value_handles[1, 0, query.slot_ids.index("y")]] == -26
    record = next(r for r in tape.occurrences if r.node.candidate_id == "call")
    assert record.node.child_trace.frontiers[0].nodes[0].parents == ("right", "left")
    x = torch.tensor([[1.5]], requires_grad=True)
    replay = replay_cooperative_dependencies(query, {"x": x}, tape=tape, endpoint=endpoints[0])
    torch.testing.assert_close(replay.outputs["y"], -13 * x)
    dx, dl, dr = torch.autograd.grad(replay.outputs["y"].sum(),
        (x, left.candidate.operand_store.tensor("weight"), right.candidate.operand_store.tensor("weight")))
    torch.testing.assert_close(dx, torch.full_like(x, -13))
    torch.testing.assert_close(dl, x)
    torch.testing.assert_close(dr, -3 * x)


def test_returned_child_producer_is_real_target_of_later_child_effect(device):
    producer = member("producer", "x", "h", owner="memory")
    first_child = m.FormulaProgramQueryV7(slot_ids=("x", "h", "h_negative"), candidates=(producer,),
        terminal_slots={"result": "h"}, entry_candidates=("producer",), continuations={}, max_steps=1)
    writer = effect()
    second_child = m.FormulaProgramQueryV5(slot_ids=("h", "tail"), candidates=(writer,),
        terminal_slots={"result": "tail"}, max_steps=1, hidden_dim=8)
    first = m.FormulaProgramCallCandidateV1("first", first_child,
        input_slots={"x": "x"}, output_slots={"result": "h"})
    second = m.FormulaProgramCallCandidateV1("second", second_child,
        input_slots={"h": "h"}, output_slots={"result": "y"})
    query = m.FormulaProgramQueryV7(slot_ids=("x", "h", "y"), candidates=(first, second),
        terminal_slots={"y": "y"}, entry_candidates=("first",), continuations={"first": {"second": "h"}},
        max_steps=2).to(device)
    tape, endpoints, data, _state = collect(query, count=3, product_slots=(), publish_slots=())
    native = query({"x": data[0]})
    assert tape.occurrences[1].node.child_trace == native.trace.steps[1].child_trace
    assert tape.occurrences[1].node.parents == ("first/producer",)
    x = torch.tensor([[.25]], requires_grad=True)
    replay = replay_cooperative_dependencies(query, {"x": x}, tape=tape, endpoint=endpoints[0])
    assert len(replay.proposals) == 1
    assert replay.proposals[0].predecessor_execution_id == "first/producer"
    rate = writer.operand_store.tensor("rate")
    torch.testing.assert_close(replay.outputs["y"], x)
    torch.testing.assert_close(replay.bank_state.values[0], 1 + 2 * x * rate)
    dx, dr = torch.autograd.grad((replay.outputs["y"] + replay.bank_state.values[0]).sum(), (x, rate))
    torch.testing.assert_close(dx, 1 + 2 * rate)
    torch.testing.assert_close(dr, 2 * x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_captured_call_records_replay_without_old_numeric_pools():
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
        data[0].fill_(4.)
        capture.replay()
        tape, endpoints = decode_cooperative_device_products(query, records,
            initial_states=(query.initial_bank_state(),), input_slots=("x",))
        data.fill_(torch.nan)
        bank.fill_(torch.nan)
        x = torch.tensor([[.75]], requires_grad=True)
        replay = replay_cooperative_dependencies(query, {"x": x}, tape=tape, endpoint=endpoints[0])
        expected = query({"x": x})
        torch.testing.assert_close(replay.outputs["result"], expected.outputs["result"])
        torch.testing.assert_close(replay.bank_state.values[0], expected.bank_state.values[0])
        rate = next(c for c in child.candidates if c.candidate_id == "write").operand_store.tensor("rate")
        actual = torch.autograd.grad((replay.outputs["result"] + replay.bank_state.values[0]).sum(), (rate, x))
        wanted = torch.autograd.grad((expected.outputs["result"] + expected.bank_state.values[0]).sum(), (rate, x))
        for a, b in zip(actual, wanted, strict=True):
            torch.testing.assert_close(a, b)
