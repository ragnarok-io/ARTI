from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from arti import mechanisms as m
from arti import _formula_candidate_batch as batch


def _type(width=3, *, dtype="float32"):
    return m.TensorType(("B", "D"), ("B", width), dtype=dtype, domain="activation")


def _ordinary(
    name, input_slot="x", output_slot="terminal", *, owner=None,
    width=3, dtype="float32", weight=None, saturate=False, scalar_bank=False,
):
    value_type = _type(width, dtype=dtype)
    bank_type = (
        m.TensorType((), (), dtype=dtype, domain="activation")
        if scalar_bank else value_type
    )
    value = m.InputBinding("value", value_type)
    binding = m.BankBinding("weight", "arti/test-candidate-batch@1", "weight", bank_type)
    output = m.scale(value, binding)
    if saturate:
        output = m.scalar_map(output, mode="tanh")
    if weight is None:
        weight = torch.full(() if scalar_bank else (1, width), 2.0)
    candidate_type = m.FormulaProgramCandidateV2 if saturate else m.FormulaProgramCandidate
    candidate = candidate_type(
        name, m.FormulaProgram.build(outputs=(output,)),
        input_slots={"value": input_slot}, output_slot=output_slot,
        operands={"weight": weight},
        trainable_operands=() if owner else ("weight",),
    )
    return m.FormulaProgramTensorCandidateV3(
        candidate, plastic_bank_slot="weight" if owner else None, bank_owner_id=owner,
    )


def _effect(name, input_slot="made", output_slot="changed", *, binding_only=False, count=False):
    value = m.InputBinding("value", _type())
    gain = m.BankBinding("gain", "arti/test-candidate-batch@1", "gain", _type())
    operands = {"gain": torch.zeros(1, 3)}
    if binding_only:
        update = value
    else:
        writer = m.BankBinding("writer", "arti/test-candidate-batch@1", "writer", _type())
        update = m.scale(value, writer)
        operands["writer"] = torch.full((1, 3), 0.25)
    expression = m.neural_plasticity(value, update, gain)
    return m.FormulaProgramEffectCandidateV3(
        name,
        m.FormulaEffectProgramV2(
            m.FormulaProgram.build(outputs=(expression,)),
            data_input_name="value", state_type=_type(),
        ),
        input_slot=input_slot, output_slot=output_slot, operands=operands,
        trainable_operands=() if binding_only else ("writer",),
        execution_count=torch.tensor(1.0) if count else None,
        trainable_execution_count=count, max_executions=3,
    )


def _query(candidates, *, slots=("x", "terminal")):
    return m.FormulaProgramQueryV4(
        slot_ids=slots, candidates=candidates, terminal_slot="terminal",
        min_steps=1, max_steps=5, hidden_dim=4,
    )


def _record_groups(monkeypatch):
    sizes = []
    original = batch._run_group

    def record(requests, rows, plan):
        sizes.append(len(rows))
        return original(requests, rows, plan)

    monkeypatch.setattr(batch, "_run_group", record)
    return sizes


@pytest.mark.parametrize("chunk_size,expected_groups", [(None, [5]), (2, [2, 2]), (1, [])])
def test_inference_batches_existing_plan_and_chunks_before_stack(
    monkeypatch, chunk_size, expected_groups,
):
    candidate = _ordinary("ordinary")
    query = _query((candidate,))
    requests = tuple(
        (candidate, query._arena({"x": torch.full((1, 3), float(index + 1))}))
        for index in range(5)
    )
    groups = _record_groups(monkeypatch)
    keys = tuple(query.state_dict())
    with torch.inference_mode():
        expected = query.execute_many(requests, serial=True)
        actual = query.execute_many(requests, chunk_size=chunk_size)
    assert groups == expected_groups
    assert tuple(query.state_dict()) == keys
    for reference, result in zip(expected, actual, strict=True):
        torch.testing.assert_close(result.values.get("terminal"), reference.values.get("terminal"))
        assert result.producer("terminal") == reference.producer("terminal")


def test_heterogeneous_groups_preserve_request_order(monkeypatch):
    wide = _ordinary("wide", width=3)
    narrow = _ordinary("narrow", width=2)
    query = _query((wide, narrow))
    requests = tuple(
        (candidate, query._arena({"x": torch.full((1, width), value)}))
        for candidate, width, value in (
            (wide, 3, 1.0), (narrow, 2, 2.0), (wide, 3, 3.0), (narrow, 2, 4.0),
        )
    )
    groups = _record_groups(monkeypatch)
    with torch.no_grad():
        actual = query.execute_many(requests)
        expected = query.execute_many(requests, serial=True)
    assert groups == [2, 2]
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result.values.get("terminal"), reference.values.get("terminal"))


@pytest.mark.parametrize("chunk_size", [None, 2])
def test_effect_identity_branch_revisions_and_shared_owner_reread(monkeypatch, chunk_size):
    first = _ordinary("first", "x", "made", owner="owner")
    effect = _effect("effect")
    again = _effect("again", "changed", "twice")
    reread = _ordinary("reread", "twice", "terminal", owner="owner")
    query = _query((first, effect, again, reread), slots=("x", "made", "changed", "twice", "terminal"))
    roots = tuple(query._arena({"x": torch.full((1, 3), value)}) for value in (0.5, 1.0, 1.5))
    groups = _record_groups(monkeypatch)
    with torch.no_grad():
        actual, expected = roots, roots
        for candidate in (first, effect, again, reread):
            previous = actual
            actual = query.execute_many(tuple((candidate, arena) for arena in actual), chunk_size=chunk_size)
            expected = query.execute_many(tuple((candidate, arena) for arena in expected), serial=True)
            if isinstance(candidate, m.FormulaProgramEffectCandidateV3):
                for before, after in zip(previous, actual, strict=True):
                    assert after.values.get(candidate.output_slot) is before.values.get(candidate.input_slot)
    assert groups == ([3] * 4 if chunk_size is None else [2] * 4)
    assert first.bank_owner is reread.bank_owner
    assert first.initial_revision() == 0
    torch.testing.assert_close(first.initial_bank_value(), torch.full((1, 3), 2.0))
    successors = []
    for result, reference in zip(actual, expected, strict=True):
        torch.testing.assert_close(result.values.get("terminal"), reference.values.get("terminal"))
        assert len(result.proposals) == 2
        for proposal, original in zip(result.proposals, reference.proposals, strict=True):
            assert proposal.target == original.target == first.bank_slot_ref
            assert proposal.previous_revision == original.previous_revision
            assert proposal.successor_revision == original.successor_revision
            torch.testing.assert_close(proposal.successor, original.successor)
        assert result.proposals[1].previous is result.proposals[0].successor
        assert result.bank_state.revision(first.bank_slot_ref) == 0
        successors.append(result.proposals[-1].successor)
    assert len({id(value) for value in successors}) == len(roots)


def test_intermediate_nonfinite_is_rejected_even_after_tanh(monkeypatch):
    candidate = _ordinary("overflow", weight=torch.full((1, 3), 1e20), saturate=True)
    query = _query((candidate,))
    requests = tuple((candidate, query._arena({"x": torch.full((1, 3), 1e20)})) for _ in range(2))
    groups = _record_groups(monkeypatch)
    with torch.no_grad():
        for serial in (True, False):
            with pytest.raises(m.FormulaBindingError) as error:
                query.execute_many(requests, serial=serial)
            assert error.value.code == "FF2_NONFINITE"
    assert groups == [2]
    assert all(arena.values.get("terminal") is None for _, arena in requests)


@pytest.mark.parametrize("serial", [False, True])
def test_runtime_instruction_dtype_checks_are_not_skipped(serial):
    candidate = _ordinary("dtype", dtype="floating", weight=torch.ones(1, 3, dtype=torch.float64))
    query = _query((candidate,))
    requests = tuple((candidate, query._arena({"x": torch.ones(1, 3)})) for _ in range(2))
    with torch.no_grad(), pytest.raises(m.FormulaBindingError) as error:
        query.execute_many(requests, serial=serial)
    assert error.value.code == "FF2_RUNTIME_DTYPE_MISMATCH"


@pytest.mark.parametrize("bad_index", [False, True])
def test_index_programs_retain_native_checks_and_fallback(monkeypatch, bad_index):
    value = m.InputBinding("value", _type())
    index_type = m.TensorType(("J",), (2,), dtype="int64", domain="index")
    indices = m.BankBinding("indices", "arti/test-candidate-batch@1", "indices", index_type)
    expression = m.gather(value, indices, axis="D", index_axis="J")
    candidate = m.FormulaProgramTensorCandidateV3(m.FormulaProgramCandidate(
        "gather", m.FormulaProgram.build(outputs=(expression,)),
        input_slots={"value": "x"}, output_slot="terminal",
        operands={"indices": torch.tensor([0, 9 if bad_index else 2])},
    ))
    query = _query((candidate,))
    requests = tuple((candidate, query._arena({"x": torch.tensor([[1., 2., 3.]])})) for _ in range(2))
    groups = _record_groups(monkeypatch)
    with torch.no_grad():
        if bad_index:
            with pytest.raises(m.FormulaBindingError) as expected:
                query.execute_many(requests, serial=True)
            with pytest.raises(m.FormulaBindingError) as actual:
                query.execute_many(requests)
            assert actual.value.code == expected.value.code
        else:
            actual = query.execute_many(requests)
            for result in actual:
                torch.testing.assert_close(result.values.get("terminal"), torch.tensor([[1., 3.]]))
    assert groups == []


def test_binding_only_effect_uses_native_without_empty_program(monkeypatch):
    first = _ordinary("first", "x", "made", owner="owner")
    effect = _effect("direct", "made", "terminal", binding_only=True)
    query = _query((first, effect), slots=("x", "made", "terminal"))
    with torch.no_grad():
        roots = tuple(first(query._arena({"x": torch.ones(1, 3)})) for _ in range(2))
        groups = _record_groups(monkeypatch)
        results = query.execute_many(tuple((effect, arena) for arena in roots))
    assert groups == []
    for root, result in zip(roots, results, strict=True):
        assert result.values.get("terminal") is root.values.get("made")
        torch.testing.assert_close(result.proposals[0].successor, torch.full((1, 3), 4.0))
    assert first.initial_revision() == 0


def test_same_data_shape_does_not_make_predecessor_bank_types_compatible(monkeypatch):
    vector = _ordinary("vector", "x", "made", owner="vector-owner")
    scalar = _ordinary("scalar", "x", "made", owner="scalar-owner", scalar_bank=True)
    effect = _effect("effect", "made", "terminal")
    query = _query((vector, scalar, effect), slots=("x", "made", "terminal"))
    groups = _record_groups(monkeypatch)
    with torch.no_grad():
        root = query._arena({"x": torch.ones(1, 3)})
        vector_root, scalar_root = vector(root), scalar(root)
        torch.testing.assert_close(vector_root.values.get("made"), scalar_root.values.get("made"))
        with pytest.raises(m.FormulaBindingError) as error:
            query.execute_many(((effect, vector_root), (effect, scalar_root)))
        assert error.value.code == "FF2_BINDING_RANK"
    assert groups == []
    assert vector.initial_revision() == scalar.initial_revision() == 0


def test_grad_enabled_partial_consumption_preserves_none_and_optimizer_semantics(monkeypatch):
    def build():
        return _query((_ordinary("selected"), _ordinary("unused")))

    reference, actual = build(), build()
    actual.load_state_dict(reference.state_dict())
    groups = _record_groups(monkeypatch)
    for query, serial in ((reference, True), (actual, False)):
        value = torch.ones(1, 3, requires_grad=True)
        arena = query._arena({"x": value})
        results = query.execute_many(tuple((candidate, arena) for candidate in query.candidates), serial=serial)
        results[0].values.get("terminal").square().mean().backward()
        selected, unused = query.candidates
        assert selected.candidate.operand_store.tensor("weight").grad is not None
        assert unused.candidate.operand_store.tensor("weight").grad is None
        assert value.grad is not None
        torch.optim.AdamW(query.parameters(), lr=0.01, weight_decay=0.1).step()
    assert groups == []
    for (name, expected), (actual_name, value) in zip(reference.named_parameters(), actual.named_parameters(), strict=True):
        assert name == actual_name
        torch.testing.assert_close(value, expected)
        assert (value.grad is None) == (expected.grad is None)
        if value.grad is not None:
            torch.testing.assert_close(value.grad, expected.grad)


def test_grad_enabled_effect_writer_and_count_remain_independent(monkeypatch):
    first = _ordinary("first", "x", "made", owner="owner")
    selected = _effect("selected", count=True)
    unused = _effect("unused", count=True)
    reread = _ordinary("reread", "changed", "terminal", owner="owner")
    query = _query((first, selected, unused, reread), slots=("x", "made", "changed", "terminal"))
    value = torch.ones(1, 3, requires_grad=True)
    root = first(query._arena({"x": value}))
    groups = _record_groups(monkeypatch)
    results = query.execute_many(((selected, root), (unused, root)))
    reread(results[0]).values.get("terminal").sum().backward()
    assert groups == []
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert selected.operand_store.tensor("writer").grad is not None
    assert selected.execution_count.grad is not None and torch.isfinite(selected.execution_count.grad)
    assert unused.operand_store.tensor("writer").grad is None
    assert unused.execution_count.grad is None
    assert first.initial_revision() == 0


@pytest.mark.parametrize("chunk_size", [1, 2, 8])
@pytest.mark.parametrize("consumed,zero_loss", [((0,), False), ((0, 3), False), ((0,), True)])
@pytest.mark.parametrize("history", [False, True])
def test_independent_graphs_preserve_sparse_and_zero_gradients_with_adamw(
    monkeypatch, chunk_size, consumed, zero_loss, history,
):
    def build():
        value = m.InputBinding("value", _type())
        shared = m.BankBinding("shared", "arti/test-candidate-batch@1", "shared", _type())
        unique = m.BankBinding("unique", "arti/test-candidate-batch@1", "unique", _type())
        program = m.FormulaProgram.build(outputs=(m.add(m.scale(value, shared), m.scale(value, unique)),))
        candidates = tuple(m.FormulaProgramTensorCandidateV3(m.FormulaProgramCandidateV2(
            f"candidate-{index}", program, input_slots={"value": "x"}, output_slot="terminal",
            operands={"shared": torch.full((1, 3), 0.5), "unique": torch.full((1, 3), 0.25)},
            trainable_operands=("shared", "unique"),
        )) for index in range(5))
        shared_parameter = candidates[0].candidate.operand_store.tensor("shared")
        for candidate in candidates[1:]:
            operand_store = candidate.candidate.operand_store
            setattr(operand_store, operand_store._attributes["shared"], shared_parameter)
        return _query(candidates)

    reference, actual = build(), build()
    actual.load_state_dict(reference.state_dict())
    optimizers = tuple(torch.optim.AdamW(query.parameters(), lr=0.01, weight_decay=0.1) for query in (reference, actual))
    groups = _record_groups(monkeypatch)
    singles = []
    run_single = batch._run_single

    def record_single(prepared, plan):
        singles.append(True)
        return run_single(prepared, plan)

    monkeypatch.setattr(batch, "_run_single", record_single)
    input_gradients = []
    for query, optimizer, serial in ((reference, optimizers[0], True), (actual, optimizers[1], False)):
        if history:
            arena = query._arena({"x": torch.ones(1, 3)})
            warm = query.execute_many(tuple((candidate, arena) for candidate in query.candidates), serial=True)
            sum(result.values.get("terminal").square().mean() for result in warm).backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        inputs = tuple(torch.full((1, 3), float(index + 1), requires_grad=True) for index in range(5))
        requests = tuple((candidate, query._arena({"x": value})) for candidate, value in zip(query.candidates, inputs, strict=True))
        results = query.execute_many(requests, chunk_size=chunk_size, serial=serial)
        loss = sum(results[index].values.get("terminal").square().mean() for index in consumed)
        (loss * 0.0 if zero_loss else loss).backward()
        for index, candidate in enumerate(query.candidates):
            gradient = candidate.candidate.operand_store.tensor("unique").grad
            assert (gradient is None) == (index not in consumed)
            if index in consumed and zero_loss:
                assert torch.count_nonzero(gradient) == 0
        input_gradients.append(tuple(None if value.grad is None else value.grad.clone() for value in inputs))
        optimizer.step()
    assert groups == []
    assert len(singles) == 5
    for expected, value in zip(input_gradients[0], input_gradients[1], strict=True):
        assert (expected is None) == (value is None)
        if value is not None:
            torch.testing.assert_close(value, expected)
    for (name, expected), (other_name, value) in zip(reference.named_parameters(), actual.named_parameters(), strict=True):
        assert name == other_name
        torch.testing.assert_close(value, expected)
        assert (value.grad is None) == (expected.grad is None)
        if value.grad is not None:
            torch.testing.assert_close(value.grad, expected.grad)
        assert (expected in optimizers[0].state) == (value in optimizers[1].state)
        expected_state = optimizers[0].state.get(expected, {})
        actual_state = optimizers[1].state.get(value, {})
        assert actual_state.keys() == expected_state.keys()
        for field, state in actual_state.items():
            torch.testing.assert_close(state, expected_state[field])


def test_custom_forward_and_hooks_keep_native_calls(monkeypatch):
    candidate = _ordinary("ordinary")
    query = _query((candidate,))
    arena = query._arena({"x": torch.ones(1, 3)})
    original = candidate.forward
    calls, hooks = [], []

    def custom(value):
        calls.append(value)
        return original(value)

    monkeypatch.setattr(candidate, "forward", custom)
    handle = candidate.register_forward_hook(lambda *args: hooks.append(True))
    groups = _record_groups(monkeypatch)
    try:
        with torch.no_grad():
            query.execute_many(((candidate, arena), (candidate, arena)))
    finally:
        handle.remove()
    assert len(calls) == len(hooks) == 2
    assert groups == []


def test_global_forward_hook_that_changes_values_uses_native(monkeypatch):
    candidate = _ordinary("ordinary")
    query = _query((candidate,))
    arena = query._arena({"x": torch.ones(1, 3)})
    groups = _record_groups(monkeypatch)

    def scale_fabric_output(module, inputs, output):
        if isinstance(module, m.FormulaFabricV2):
            return replace(output, values=tuple(value * 2 for value in output.values))
        return None

    handle = torch.nn.modules.module.register_module_forward_hook(scale_fabric_output)
    try:
        with torch.no_grad():
            results = query.execute_many(((candidate, arena), (candidate, arena)))
    finally:
        handle.remove()
    assert groups == []
    for result in results:
        torch.testing.assert_close(result.values.get("terminal"), torch.full((1, 3), 4.0))


@pytest.mark.parametrize("chunk_size", [0, -1, True, 1.5])
def test_invalid_chunk_size_is_rejected(chunk_size):
    query = _query((_ordinary("ordinary"),))
    with pytest.raises(ValueError, match="chunk_size"):
        query.execute_many((), chunk_size=chunk_size)


def test_empty_and_foreign_requests():
    candidate = _ordinary("ordinary")
    query = _query((candidate,))
    assert query.execute_many(()) == ()
    with pytest.raises(ValueError, match="belong"):
        query.execute_many(((_ordinary("foreign"), query._arena({"x": torch.ones(1, 3)})),))
