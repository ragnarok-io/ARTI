import pytest
import torch

from arti import mechanisms as m
from arti.observation import fourier_observation
from benchmarks._federated_spatial_events import SpatialEventSpec, make_spatial_events
from benchmarks._federated_spatial_programs import (
    build_spatial_templates, SpatialProgramSpec, ROUTES, build_spatial_search_graph, SpatialReadOnlyHead,
)


@pytest.fixture(params=["cpu", "cuda"])
def templates(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    return build_spatial_templates(SpatialEventSpec(height=3, width=4, markers=3, reuse_questions=3),
        SpatialProgramSpec(latent_positions=2, hidden_dim=8, rank=4, banks=2), seed=17, device=request.param)


def numeric(candidate, **inputs):
    tensors = candidate.candidate.operand_store.tensors()
    banks = {binding.name: binding.bind(tensors[binding.name]) for binding in candidate.candidate.program.bindings
             if isinstance(binding, m.BankBinding)}
    result = candidate.candidate.fabric(inputs=inputs, banks=banks)
    return dict(zip(candidate.output_slot_ids, result.values, strict=True))


def test_spatial_encoding_and_multisource_relations_have_real_gradients(templates):
    candidate = templates.candidates["direct"]
    device = next(candidate.parameters()).device
    event = make_spatial_events(templates.event_spec, split="train", seed=23, device=device)
    x = event.x0.requires_grad_()
    first = numeric(candidate, image=x, question=event.question)
    assert first["hidden"].shape == (1, 2, 8)
    assert all(first[f"response.{route}"].shape == (1,) for route in ROUTES)
    second = first["hidden"].square()
    for family in ("ordered-gelu", "bilinear"):
        relation = templates.candidates[f"relation.{family}"]
        result = numeric(relation, left=first["hidden"], right=second)["hidden"]
        prediction = numeric(templates.candidates["answer"], hidden=result)["answer"]
        assert prediction.shape == (1, 4)
        grad, = torch.autograd.grad((prediction - event.answer).square().mean(), x, retain_graph=True)
        assert torch.isfinite(grad).all() and grad.abs().sum() > 0
        reverse = numeric(relation, left=second, right=first["hidden"])["hidden"]
        assert not torch.allclose(result, reverse)


def test_observation_reuses_original_pixels_with_differentiable_displacement(templates):
    device = next(templates.candidates["direct"].parameters()).device
    event = make_spatial_events(templates.event_spec, split="train", seed=29, device=device)
    initial = numeric(templates.candidates["direct"], image=event.x0, question=event.question)
    shift = numeric(templates.candidates["shift"], hidden=initial["hidden"])["displacement"]
    first = numeric(templates.candidates["observation"], image=event.x0, question=event.question, displacement=shift)
    next_shift = numeric(templates.candidates["shift"], hidden=first["hidden"])["displacement"]
    second = numeric(templates.candidates["observation"], image=event.x0, question=event.question, displacement=next_shift)
    expected = fourier_observation(event.x0, next_shift, spatial_shape=(3, 4), state_mode="cartesian",
                                   direction_epsilon=1e-6, compile_policy="safe_training").squeeze(1)
    torch.testing.assert_close(second["view"], expected)
    gradient, = torch.autograd.grad(second["hidden"].square().mean(), shift)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert templates.candidates["observation"].input_slots["image"] == "x0"


def test_all_six_effects_target_actual_matrix_bank_and_keep_data_identity(templates):
    base = templates.candidates["gelu.0"]
    producer = base.with_bindings("read", input_slots={"hidden": "seed"}, output_slots=base.candidate.output_slots)
    slots = ("seed", *producer.output_slot_ids, "tail")
    model = m.FormulaProgramQueryV7(slot_ids=slots, candidates=(producer, *templates.effects),
        terminal_slots={"result": "tail"}, entry_candidates=("read",), max_steps=2,
        continuations={"read": {effect.candidate_id: "response." + effect.candidate_id.removeprefix("effect.")
                                 for effect in templates.effects}})
    assert len(templates.effects) == 6
    assert model.initial_bank_state().values[0].shape == (8, 4)
    assert not model.initial_bank_state().values[0].requires_grad
    device = next(producer.parameters()).device
    x = torch.randn((1, 2, 8), device=device, requires_grad=True)
    arena = producer(model._arena({"seed": x}))
    original = arena.values.get("hidden")
    for effect in templates.effects:
        assert effect.accepts(arena)
        result = effect(arena)
        torch.testing.assert_close(result.values.get("tail"), original, rtol=0, atol=0)
        state = result.committed_state()
        assert state.revisions[0] > arena.committed_state().revisions[0]
        assert result.proposals[-1].target == producer.bank_slot_ref
        gradient, = torch.autograd.grad(state.values[0].square().mean(), x, retain_graph=True)
        assert torch.isfinite(gradient).all()


def test_binding_variants_reuse_slow_parameters_and_real_owner(templates):
    base = templates.candidates["gated.0"]
    variant = base.with_bindings("other", input_slots={"hidden": "other_input"},
        output_slots={key: "other." + value for key, value in base.candidate.output_slots.items()})
    assert variant.bank_owner is base.bank_owner
    assert variant.candidate.operand_store is base.candidate.operand_store
    assert variant.candidate.fabric is base.candidate.fabric
    assert len([c for c in templates.candidates.values() if c.bank_slot_ref is not None]) == 4


def test_graph_keeps_answer_ready_distinct_from_return_and_readonly_meta_gradient(templates):
    graph = build_spatial_search_graph(templates, latent_slots=2, views=1, max_steps=16)
    query = graph.query
    device = next(query.parameters()).device
    event = make_spatial_events(templates.event_spec, split="train", seed=29, device=device)
    by_id = {node.candidate_id: node for node in query.candidates}
    arena = query._arena(event.writing_inputs())
    for name in ("direct", "gelu.0.to0.from0", "answer.0"):
        arena = by_id[name](arena)
    assert arena.values.get("answer") is not None
    effect = by_id["effect.outer.at0"]
    assert effect.accepts(arena)
    assert query.eligible(arena, steps=3)[query.action_ids.index(effect.candidate_id)]
    written = effect(arena)
    reader = SpatialReadOnlyHead(graph, seed=59)
    predicted = reader(written.committed_state(), event.readonly_questions[0])
    assert predicted.shape == event.readonly_answers[0].shape
    loss = (predicted - event.readonly_answers[0]).square().mean()
    variables = tuple(p for p in effect.parameters() if p.requires_grad)
    gradients = torch.autograd.grad(loss, variables, allow_unused=True)
    assert any(g is not None and g.abs().sum() > 0 for g in gradients)
    assert all(g is None or torch.isfinite(g).all() for g in gradients)
    assert len(query.initial_bank_state().values) == 4
    assert graph.product_bindings
    assert set(graph.publish_slots) >= {"seed.hidden", "observed.0", "latent.0", "tail.0"}
    # Same template at another occurrence, no proliferation of learned state.
    other = by_id["gelu.0.to1.from0"]
    assert other.bank_owner is by_id["gelu.0.to0.from0"].bank_owner
    assert by_id["effect.outer.at1"].operand_store is effect.operand_store
    assert set(reader.candidate.operand_store.trainable_names) == {"question.weight", "answer.weight"}


def test_graph_admits_old_products_and_fresh_observations(templates):
    graph = build_spatial_search_graph(templates, latent_slots=2, views=2)
    pairs = {(node.input_slots.get("left"), node.input_slots.get("right"))
             for node in graph.query.candidates if node.candidate_id.startswith("relation.")}
    for old in ("latent.0", "tail.0", "import.0", "import.1"):
        for new in ("observed.0", "observed.1"):
            assert (old, new) in pairs
            assert (new, old) in pairs
    assert ("import.0", "import.1") in pairs
    assert ("import.1", "import.0") in pairs


def test_readonly_preserves_owner_coordinates_and_all_bank_gradients(templates):
    graph = build_spatial_search_graph(templates, latent_slots=1, views=1)
    reader = SpatialReadOnlyHead(graph, seed=59)
    state = graph.query.initial_bank_state()
    values = tuple(torch.full_like(value, .2, requires_grad=True) for value in state.values)
    state = type(state)(state.slot_refs, values, state.revisions)
    questions = torch.eye(2 * templates.event_spec.markers, device=values[0].device)
    result = reader(state, questions)
    operands = reader.candidate.operand_store.tensors()
    keys = torch.einsum("bq,qal->bal", questions, operands["question.weight"])
    keys = keys.reshape(len(questions), len(values), templates.program_spec.latent_positions, -1)
    hidden = torch.nn.functional.gelu(torch.einsum("basr,adr->basd", keys, torch.stack(values)))
    expected = torch.einsum("basd,day->bay", hidden, operands["answer.weight"]).sum(1)
    torch.testing.assert_close(result, expected)
    gradients = torch.autograd.grad(result.square().sum(), values)
    assert all(torch.isfinite(g).all() and g.abs().sum() > 0 for g in gradients)
    assert not torch.allclose(gradients[0], gradients[1])
    assert len(reader.candidate.program.bindings) == 4
