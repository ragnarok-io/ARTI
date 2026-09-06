from contextlib import nullcontext

import pytest
import torch

from arti import mechanisms as m
from arti._formula_candidate_batch import _checked_plan, _prepare, execute_many
from arti._formula_grouped_training import (
    _GROUPED_TRAINING, execute_grouped_training, grouped_formula_training,
)
from arti.formula_v2 import PreparedFormulaBindings


DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.mark.parametrize("backend", ["eager", "aot_eager", "captured", "inductor"])
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("zero_used", [False, True])
def test_grouped_training_keeps_partial_heads_unused_rows_and_shared_parameters(backend, device, zero_used):
    from test_formula_program_query_v5 import _multi

    candidates = (_multi("a", owner_id="a"), _multi("b", owner_id="b"))
    query = m.FormulaProgramQueryV5(slot_ids=("x", "plain", "owned"), candidates=candidates,
        terminal_slots={"plain": "plain", "owned": "owned"}, max_steps=1).to(device)
    parameters = tuple(query.parameters())

    def run(grouped):
        with grouped_formula_training(backend=backend if grouped else "native"):
            inputs = tuple(torch.full((1, 3), float(i + 1), device=device, requires_grad=True) for i in range(3))
            requests = tuple((candidates[i % 2], query._arena({"x": x})) for i, x in enumerate(inputs))
            results = query.execute_many(requests)
            loss = results[0].values.get("plain").square().mean() + results[2].values.get("owned").square().mean()
            if zero_used:
                loss = loss + results[1].values.get("owned").sum() * 0
            gradients = torch.autograd.grad(loss, (*inputs, *parameters), allow_unused=True)
            return results, gradients

    expected, left = run(False)
    actual, right = run(True)
    assert (right[1] is None) == (not zero_used)
    for a, b in zip(expected, actual, strict=True):
        for name in ("plain", "owned"):
            torch.testing.assert_close(a.values.get(name), b.values.get(name))
        assert a.bank_state.revisions == b.bank_state.revisions
    for a, b in zip(left, right, strict=True):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a, b)


@pytest.mark.parametrize("backend", ["eager", "aot_eager", "captured", "inductor"])
@pytest.mark.parametrize("device", DEVICES)
def test_invalid_unused_row_cannot_poison_other_gradients(backend, device):
    from test_formula_candidate_batch import _ordinary, _query

    candidate = _ordinary("op").to(device)
    query = _query((candidate,))
    inputs = (torch.full((1, 3), torch.inf, device=device, requires_grad=True),
              torch.ones(1, 3, device=device, requires_grad=True))
    with grouped_formula_training(backend=backend):
        results = execute_many(tuple((candidate, query._arena({"x": x})) for x in inputs), reject_nonfinite=True)
        assert results[0] is None
        gradients = torch.autograd.grad(results[1].values.get("terminal").sum(),
                                        (*inputs, candidate.candidate.operand_store.tensor("weight")), allow_unused=True)
    assert gradients[0] is None
    assert all(torch.isfinite(g).all() for g in gradients[1:])


@pytest.mark.parametrize("backend", ["eager", "aot_eager", "captured"])
@pytest.mark.parametrize("device", DEVICES)
def test_repeated_groups_keep_first_and_second_derivatives(backend, device):
    from test_formula_candidate_batch import _ordinary, _query

    candidate = _ordinary("op", dtype="float64", saturate=True).double().to(device)
    query = _query((candidate,)).double()
    prepared = _prepare(candidate, query._arena({"x": torch.ones(1, 3, dtype=torch.float64, device=device)}))
    plan = _checked_plan(candidate.candidate.program)
    values = tuple(value.detach().requires_grad_(True) for value in prepared.values)

    def run(*values):
        row = PreparedFormulaBindings(prepared.program_fingerprint, prepared.binding_names, values)
        outputs, _finite = execute_grouped_training(plan, (row, row))
        return outputs[0][0] + outputs[1][0]

    with grouped_formula_training(backend=backend):
        assert torch.autograd.gradcheck(run, values)
        assert torch.autograd.gradgradcheck(run, values)


@pytest.mark.parametrize("backend", ["eager", "aot_eager", "captured"])
@pytest.mark.parametrize("device", DEVICES)
def test_effect_operands_still_write_real_predecessor_bank(backend, device):
    from test_formula_candidate_batch import _ordinary, _effect, _query

    first = _ordinary("first", "x", "made", owner="owner")
    effects = tuple(_effect(f"effect-{i}", count=True) for i in range(3))
    reread = _ordinary("reread", "changed", "terminal", owner="owner")
    query = _query((first, *effects, reread), slots=("x", "made", "changed", "terminal")).to(device)
    parameters = tuple(query.parameters())

    def run(grouped):
        with grouped_formula_training(backend=backend if grouped else "native"):
            roots = tuple(query._arena({"x": torch.full((1, 3), i + 0.5, device=device)}) for i in range(3))
            produced = query.execute_many(tuple((first, root) for root in roots))
            written = query.execute_many(tuple(zip(effects, produced, strict=True)))
            results = query.execute_many(tuple((reread, row) for row in written))
            for before, after in zip(produced, written, strict=True):
                assert after.values.get("changed") is before.values.get("made")
            gradients = torch.autograd.grad(results[0].values.get("terminal").square().mean(), parameters, allow_unused=True)
            return results, gradients

    left, a = run(False)
    right, b = run(True)
    for expected, actual in zip(left, right, strict=True):
        torch.testing.assert_close(actual.values.get("terminal"), expected.values.get("terminal"))
        assert actual.proposals[-1].target == first.bank_slot_ref
        assert actual.proposals[-1].successor_revision == expected.proposals[-1].successor_revision
    for expected, actual in zip(a, b, strict=True):
        assert (expected is None) == (actual is None)
        if expected is not None:
            torch.testing.assert_close(actual, expected)
    assert first.initial_revision() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("backend", ["inductor", "captured"])
def test_compiled_groups_keep_saved_outputs_across_graph_replay(backend):
    from test_formula_candidate_batch import _ordinary, _query

    candidate = _ordinary("op", saturate=True).cuda()
    query = _query((candidate,))
    with grouped_formula_training(backend=backend):
        retained = []
        for i in range(3):
            x = torch.full((1, 3), float(i + 1), device="cuda", requires_grad=True)
            result = query.execute_many(tuple((candidate, query._arena({"x": x})) for _ in range(4)))
            retained.append((x, result[0].values.get("terminal")))
        for x, value in retained:
            expected = torch.tanh(x * candidate.candidate.operand_store.tensor("weight"))
            torch.testing.assert_close(value, expected)
            (actual,) = torch.autograd.grad(value.sum(), x)
            (reference,) = torch.autograd.grad(expected.sum(), x)
            torch.testing.assert_close(actual, reference)
        runtime = next(iter(_GROUPED_TRAINING.get()[1].values()))
        assert len(runtime.buckets) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_captured_vjp_replays_forward_autocast_context(dtype):
    x = m.InputBinding("x", m.TensorType(("B", "D"), (2, 8), dtype="floating"))
    bank = m.BankBinding("bank", "arti/amp-test@1", "w", m.TensorType(("R", "D"), (4, 8), dtype="floating"))
    program = m.FormulaProgram.build(outputs=(m.contract(x, bank, reduce_axes=(("D", "D"),)),))
    plan = _checked_plan(program)
    original = {"x": torch.randn(2, 8, device="cuda"), "bank": torch.randn(4, 8, device="cuda")}

    def run(grouped, enabled):
        values = tuple(original[name].detach().clone().requires_grad_(True) for name in plan.binding_names)
        row = PreparedFormulaBindings(plan.program_fingerprint, plan.binding_names, values)
        with torch.autocast("cuda", dtype=dtype, enabled=enabled):
            output = execute_grouped_training(plan, (row,))[0][0][0] if grouped else plan.forward_checked(row)[0][0]
            loss = output.float().square().mean()
        # Backward is outside autocast; recomputation must retain forward mode.
        return output, torch.autograd.grad(loss, values)

    with grouped_formula_training(backend="captured"):
        for enabled in (True, False, True):
            torch.testing.assert_close(run(True, enabled), run(False, enabled), rtol=1e-5, atol=1e-6)
        runtime = next(iter(_GROUPED_TRAINING.get()[1].values()))
        assert len(runtime.buckets) == 4
        assert runtime.cache_builds == 4
        assert runtime.cache_hits == 2
        assert runtime.cache_evictions == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_captured_groups_do_not_share_mutable_input_buffers_between_streams():
    from test_formula_candidate_batch import _ordinary, _query

    candidate = _ordinary("op", saturate=True).cuda()
    query = _query((candidate,))
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    with grouped_formula_training(backend="captured"):
        outputs = []
        for repeat in range(2):
            for i, stream in enumerate(streams):
                with torch.cuda.stream(stream):
                    x = torch.full((1, 3), 0.5 + repeat + i, device="cuda", requires_grad=True)
                    rows = query.execute_many(((candidate, query._arena({"x": x})),))
                    y = rows[0].values.get("terminal")
                    outputs.append((stream, x, y, torch.autograd.grad(y.sum(), x)[0]))
        for stream in streams:
            torch.cuda.current_stream().wait_stream(stream)
        for stream, x, y, gradient in outputs:
            with torch.cuda.stream(stream):
                expected = torch.tanh(x * candidate.candidate.operand_store.tensor("weight"))
                expected_gradient = torch.autograd.grad(expected.sum(), x)[0]
            torch.cuda.current_stream().wait_stream(stream)
            torch.testing.assert_close(y, expected)
            torch.testing.assert_close(gradient, expected_gradient)
        runtime = next(iter(_GROUPED_TRAINING.get()[1].values()))
        assert len(runtime.buckets) == 4


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_captured_bucket_eviction_rebuilds_with_current_inputs():
    from test_formula_candidate_batch import _ordinary, _query

    candidate = _ordinary("op", saturate=True).cuda()
    query = _query((candidate,))
    plan = _checked_plan(candidate.candidate.program)
    with grouped_formula_training(backend="captured"):
        for rows in (*range(1, 34), 33, 1):
            x = torch.full((1, 3), rows / 10, device="cuda", requires_grad=True)
            prepared = _prepare(candidate, query._arena({"x": x}))
            outputs, finite = execute_grouped_training(plan, (prepared,) * rows)
            expected, expected_finite = plan.forward_checked(prepared)
            torch.testing.assert_close(outputs[0], expected)
            assert bool(finite.all()) == bool(expected_finite)
        runtime = next(iter(_GROUPED_TRAINING.get()[1].values()))
        assert len(runtime.buckets) == 32
        assert runtime.cache_builds == 34
        assert runtime.cache_hits == 1
        assert runtime.cache_evictions == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("enabled", [True, False])
@pytest.mark.parametrize("fail", [True, False])
def test_capture_restores_callers_gc_mode(monkeypatch, enabled, fail):
    import gc
    from test_formula_candidate_batch import _ordinary, _query

    candidate = _ordinary("op", saturate=True).cuda()
    query = _query((candidate,))
    capture = torch.cuda.make_graphed_callables

    def checked_capture(*args, **kwargs):
        assert not gc.isenabled()
        if fail:
            raise RuntimeError("test capture construction failure")
        return capture(*args, **kwargs)

    monkeypatch.setattr(torch.cuda, "make_graphed_callables", checked_capture)
    original = gc.isenabled()
    (gc.enable if enabled else gc.disable)()
    try:
        with grouped_formula_training(backend="captured"):
            x = torch.ones(1, 3, device="cuda", requires_grad=True)
            expected = pytest.raises(RuntimeError, match="test capture construction failure") if fail else nullcontext()
            with expected:
                query.execute_many(((candidate, query._arena({"x": x})),))
        assert gc.isenabled() == enabled
    finally:
        (gc.enable if original else gc.disable)()
