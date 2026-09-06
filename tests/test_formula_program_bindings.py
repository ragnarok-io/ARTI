import pytest
import torch

import arti
from arti import mechanisms as m
from examples.formula_binding_search import build_graph


def template(*, plastic=False):
    t = m.TensorType(("B", "D"), ("B", 2), dtype="float32")
    x = m.InputBinding("value", t)
    bank = m.BankBinding("gain", "arti/binding-test@1", "gain", m.TensorType(("D",), (2,), dtype="float32"))
    program = m.FormulaProgram.build(outputs=(m.scale(x, bank),))
    candidate = m.FormulaProgramTensorCandidateV4(m.FormulaProgramCandidateV3(
        "scale", program, input_slots={"value": "x"}, output_slots={program.outputs[0]: "a"},
        operands={"gain": torch.tensor([1.5, -0.5])}, trainable_operands=() if plastic else ("gain",),
    ), plastic_bank_slot="gain" if plastic else None, bank_owner_id="shared" if plastic else None)
    return candidate, {name: t for name in ("x", "z", "a", "b")}


def expand(candidate, types, *, inputs=("x", "z"), outputs=("a", "b"), **kwargs):
    return m.expand_candidate_bindings(candidate, slot_types=types, input_choices={"value": inputs},
        output_choices={candidate.candidate.program.outputs[0]: outputs}, **kwargs)


@pytest.mark.parametrize("plastic", (False, True))
def test_choices_share_real_bank_and_fabric_not_copied_weights(plastic):
    prototype, types = template(plastic=plastic)
    nodes = expand(prototype, types)
    assert len(nodes) == 4
    for node in nodes:
        assert node.candidate.fabric is prototype.candidate.fabric
        assert node.candidate.operand_store is prototype.candidate.operand_store
        if plastic:
            assert node.bank_owner is prototype.bank_owner
    forward = [(n.candidate_id, n.input_slots, n.candidate.output_slots) for n in nodes]
    reversed_choices = expand(prototype, dict(reversed(tuple(types.items()))), inputs=("z", "x"), outputs=("b", "a"))
    assert forward == [(n.candidate_id, n.input_slots, n.candidate.output_slots) for n in reversed_choices]
    mounted = m.FormulaProgramQueryV7(slot_ids=tuple(types), candidates=nodes,
        terminal_slots={"answer": "a"}, entry_candidates=tuple(n.candidate_id for n in nodes),
        continuations={}, max_steps=1).double()
    assert all(n.candidate.operand_store.tensor("gain").dtype == torch.float64 for n in mounted.candidates)
    if not plastic:
        assert len(tuple(mounted.parameters())) == 1
    else:
        assert len(mounted.owner_states) == 1


def test_shared_bank_gradient_sums_over_two_distinct_input_connections():
    prototype, types = template()
    nodes = expand(prototype, types, outputs=("a",))
    x, z = torch.randn(1, 2, requires_grad=True), torch.randn(1, 2, requires_grad=True)
    arena = m.FormulaProgramArena.from_mapping(tuple(types), {"x": x, "z": z})
    results = [node.candidate(arena).get("a") for node in nodes]
    gain = prototype.candidate.operand_store.tensor("gain")
    actual = torch.autograd.grad(sum(v.square().sum() for v in results), (x, z, gain), retain_graph=True)
    expected = torch.autograd.grad((x * gain).square().sum() + (z * gain).square().sum(), (x, z, gain))
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e)


def test_exact_types_and_ssa_read_write_exclusions():
    prototype, types = template()
    types["integer"] = m.TensorType(("B", "D"), ("B", 2), dtype="int64")
    nodes = expand(prototype, types, inputs=("integer", "x", "a"), outputs=("a",))
    assert len(nodes) == 1 and nodes[0].input_slots == {"value": "x"}
    with pytest.raises(ValueError, match="exact-type"):
        expand(prototype, types, inputs=("integer",))
    with pytest.raises(ValueError, match="SSA wiring"):
        expand(prototype, types, inputs=("a",), outputs=("a",))
    with pytest.raises(ValueError, match="SSA wiring"):
        expand(prototype, types, inputs=("x",), requires_empty_slots=("x",))
    with pytest.raises(ValueError, match="before expansion"):
        expand(prototype, types, max_candidates=3)
    with pytest.raises(ValueError, match="unique declared"):
        expand(prototype, types, inputs=("missing",))


def test_actual_atomic_alternatives_and_cross_depth_graph_replay(tmp_path):
    model = build_graph()
    assert not hasattr(model, "network")
    activations = [c for c in model.candidates if c.candidate_id.startswith(("silu.", "tanh."))]
    assert len(activations) == 6
    assert {c.input_slots["value"] for c in activations} == {"x", "z", "seed"}
    joins = [c for c in model.candidates if c.candidate_id.startswith("join.")]
    assert len(joins) == 4
    values = {"x": torch.tensor([[1., -0.5]], requires_grad=True), "z": torch.tensor([[-0.2, 0.7]], requires_grad=True)}
    result = model(values)
    replay = model.replay(values, result.frontiers)
    assert result.trace.stopped and replay.frontiers == result.frontiers
    torch.testing.assert_close(result.outputs["answer"], replay.outputs["answer"])
    leaves = (*values.values(), *model.parameters())
    first = torch.autograd.grad(result.outputs["answer"].square().sum(), leaves, allow_unused=True)
    second = torch.autograd.grad(replay.outputs["answer"].square().sum(), leaves, allow_unused=True)
    for a, b in zip(first, second, strict=True):
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a, b)
    saved = arti.save(model, tmp_path / "bindings.arti.st")
    restored = build_graph()
    arti.load(saved.weights_path, model=restored, strict=True, verify_architecture=True)
    fresh = restored(values)
    torch.testing.assert_close(result.outputs["answer"], fresh.outputs["answer"])
    assert [[n.inputs for n in f.nodes] for f in result.frontiers] == [[n.inputs for n in f.nodes] for f in fresh.frontiers]


def test_template_guards_are_inherited_unless_explicitly_replaced():
    prototype, types = template()
    guarded = prototype.with_bindings("guarded", input_slots=prototype.input_slots,
        output_slots=prototype.candidate.output_slots, requires_empty_slots=("b",))
    nodes = expand(guarded, types, inputs=("x",), outputs=("a",))
    arena = m.FormulaProgramArena.from_mapping(tuple(types), {"x": torch.ones(1, 2), "b": torch.ones(1, 2)})
    assert nodes[0].requires_empty_slots == ("b",)
    assert not guarded.candidate.accepts(arena) and not nodes[0].candidate.accepts(arena)
    assert expand(guarded, types, inputs=("x",), outputs=("a",), requires_empty_slots=())[0].candidate.accepts(arena)
    with pytest.raises(ValueError, match="empty slots"):
        expand(guarded, {k: t for k, t in types.items() if k != "b"}, inputs=("x",), outputs=("a",))


def test_expanded_parameter_sharing_survives_reconstructed_asset_roundtrip(tmp_path):
    def model():
        prototype, types = template()
        nodes = expand(prototype, types, outputs=("a",))
        return m.FormulaProgramQueryV7(slot_ids=tuple(types), candidates=nodes,
            terminal_slots={"answer": "a"}, entry_candidates=tuple(n.candidate_id for n in nodes),
            continuations={}, max_steps=1)

    original = model()
    with torch.no_grad():
        original.candidates[0].candidate.operand_store.tensor("gain").add_(0.7)
    saved = arti.save(original, tmp_path / "shared.arti.st")
    restored = model()
    arti.load(saved.weights_path, model=restored, strict=True, verify_architecture=True)
    assert len(tuple(restored.parameters())) == 1
    assert restored.candidates[0].candidate.operand_store is restored.candidates[1].candidate.operand_store
    values = {"x": torch.tensor([[2., 1.]]), "z": torch.tensor([[-3., 4.]])}
    torch.testing.assert_close(original(values).outputs["answer"], restored(values).outputs["answer"])


@pytest.mark.parametrize("backend", ("eager", "aot_eager"))
def test_expanded_candidates_keep_grouped_training_shared_parameter_gradients(backend):
    from arti._formula_candidate_batch import _checked_plan
    from arti._formula_grouped_training import execute_grouped_training, grouped_formula_training

    prototype, types = template()
    nodes = expand(prototype, types, outputs=("a",))
    inputs = {"x": torch.randn(1, 2, requires_grad=True), "z": torch.randn(1, 2, requires_grad=True)}
    arena = m.FormulaProgramArena.from_mapping(tuple(types), inputs)
    prepared = []
    for node in nodes:
        values, banks = node.candidate._bindings(arena)
        prepared.append(node.candidate.fabric.bind_tensors(inputs=values, banks=banks))
    gain = prototype.candidate.operand_store.tensor("gain")
    with grouped_formula_training(backend=backend):
        outputs, valid = execute_grouped_training(_checked_plan(prototype.candidate.program), tuple(prepared))
        assert valid.all()
        actual = torch.autograd.grad(sum(row[0].square().sum() for row in outputs), (*inputs.values(), gain))
    expected = torch.autograd.grad(sum((x * gain).square().sum() for x in inputs.values()), (*inputs.values(), gain))
    for a, e in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, e)


def test_two_generated_input_bindings_reach_real_device_numeric_group_on_cpu():
    from arti._formula_device_dispatch import FormulaDeviceDispatchLayout, FormulaDeviceNumericalDispatch, formula_device_dispatch_groups
    from arti._formula_device_frames import FormulaDeviceFrameKernel
    from arti._formula_device_pools import FormulaDevicePoolLayout

    prototype, types = template()
    nodes = expand(prototype, types, outputs=("a",))
    query = m.FormulaProgramQueryV7(slot_ids=tuple(types), candidates=nodes, terminal_slots={"y": "a"},
        entry_candidates=tuple(n.candidate_id for n in nodes), continuations={}, max_steps=1)
    kernel = FormulaDeviceFrameKernel.from_query(query)
    layout = FormulaDevicePoolLayout.from_samples((torch.zeros(1, 2),), 4)
    bank_layout = FormulaDevicePoolLayout.from_samples((torch.zeros(2),), 1)
    dispatch = FormulaDeviceNumericalDispatch(query, kernel)
    dispatch.prepare_typed_pools_(layout, bank_layout)
    state = kernel.initial_state(1, torch.tensor([[0, 1, -1, -1]]))
    data, banks = layout.allocate("cpu"), bank_layout.allocate("cpu")
    data[0][0], data[0][1] = torch.tensor([[2., 3.]]), torch.tensor([[-4., 1.]])
    packet = FormulaDeviceDispatchLayout(formula_device_dispatch_groups(query)[0])(torch.tensor([[0, 1]]))
    result = dispatch(state, packet, data, banks)
    assert result.numeric_valid.all() and not result.overflow
    assert result.output_present[0][:, 0].all()
    torch.testing.assert_close(result.output_values[0][:, 0], data[0][:2] * prototype.candidate.operand_store.tensor("gain"))
