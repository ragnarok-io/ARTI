from dataclasses import replace

import pytest
import torch
from torch.utils._pytree import tree_flatten

from benchmarks._federated_endpoint_risk import backward_cooperative_endpoint_risk
from benchmarks._federated_product_replay import replay_cooperative_dependencies_many
from benchmarks._federated_spatial_events import SpatialEventSpec, make_spatial_events
from benchmarks._federated_spatial_programs import (
    SpatialProgramSpec, SpatialReadOnlyHead, build_spatial_templates, build_spatial_search_graph,
)
from benchmarks.probe_federated_spatial_event import SpatialDeviceBatch


@pytest.fixture(params=["cpu", "cuda"])
def setup(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    spec = SpatialEventSpec(height=3, width=4, markers=3, reuse_questions=3)
    templates = build_spatial_templates(spec, SpatialProgramSpec(latent_positions=2, hidden_dim=8, rank=4, banks=2),
                                       seed=17, device=request.param)
    graph = build_spatial_search_graph(templates, latent_slots=2, views=1, cooperation_width=2, max_steps=12)
    events = make_spatial_events(spec, split="train", seed=29, count=2, device=request.param)
    batch = SpatialDeviceBatch.prepare(graph, events, width=2, heads=3, steps=12)
    return graph, events, batch


def event_slice(events, index):
    return replace(events, **{name: getattr(events, name)[index:index + 1]
                              for name in ("x0", "question", "answer", "readonly_questions", "readonly_answers")})


def test_parallel_events_match_serial_and_reload_resident_buffers(setup):
    graph, events, batch = setup
    records = batch.search()
    for index in range(2):
        local = SpatialDeviceBatch.prepare(graph, event_slice(events, index), width=2, heads=3, steps=12)
        single = local.search()
        for actual, expected in zip(tree_flatten(records[-1].state)[0], tree_flatten(single[-1].state)[0], strict=True):
            torch.testing.assert_close(actual[index], expected[0])
        tape, endpoints = batch.decode(records, index)
        assert endpoints and tape.initial_states[0] is batch.initial_states[index]
        assert all(endpoint.root == 0 for endpoint in endpoints)
    addresses = tuple(t.data_ptr() for t in (*batch.data, *batch.bank))
    other = make_spatial_events(events.spec, split="train", seed=47, count=2, device=events.x0.device)
    batch.load(other)
    changed = batch.search()
    fresh = SpatialDeviceBatch.prepare(graph, other, width=2, heads=3, steps=12).search()
    for actual, expected in zip(tree_flatten(changed[-1].state)[0], tree_flatten(fresh[-1].state)[0], strict=True):
        torch.testing.assert_close(actual, expected)
    assert tuple(t.data_ptr() for t in (*batch.data, *batch.bank)) == addresses


def test_event_average_gradient_matches_joint_objective(setup):
    graph, events, batch = setup
    reader = SpatialReadOnlyHead(graph, seed=59)
    # Exercise reader gradients independently of an untrained route choosing a write.
    states = tuple(type(state)(state.slot_refs, tuple(torch.full_like(value, .1 * (index + 1))
                   for value in state.values), tuple(index + 1 for _ in state.revisions))
                   for index, state in enumerate(batch.initial_states))
    batch.load(events, states)
    records = batch.search()
    decoded = [batch.decode(records, index) for index in range(2)]
    parameters = tuple(dict.fromkeys((*graph.query.parameters(), *reader.parameters())))
    parameters = tuple(p for p in parameters if p.requires_grad)
    risks = []
    for index, (tape, endpoints) in enumerate(decoded):
        runs = replay_cooperative_dependencies_many(graph.query, events.writing_inputs(index), tape=tape,
            endpoints=endpoints, initial_states=(batch.initial_states[index],), score_decisions=True)
        losses = torch.stack([(run.outputs["answer"] - events.answer[index:index+1]).square().mean() +
                              (reader(run.bank_state, events.readonly_questions[index]) -
                               events.readonly_answers[index]).square().mean() for run in runs])
        energies = torch.stack([run.decision_energy for run in runs])
        risks.append((energies.double().softmax(0) * losses.double()).sum())
    objective = torch.stack(risks).mean()
    expected = torch.autograd.grad(objective, parameters, allow_unused=True)
    reports = []
    for index, (tape, endpoints) in enumerate(decoded):
        reports.append(backward_cooperative_endpoint_risk(graph.query, events.writing_inputs(index),
            tape=tape, endpoints=endpoints, initial_states=(batch.initial_states[index],),
            answer_loss=lambda outputs, i=index: (outputs["answer"] - events.answer[i:i+1]).square().mean(),
            readonly_reuse_loss=lambda state, i=index: (reader(state, events.readonly_questions[i]) -
                                                       events.readonly_answers[i]).square().mean(),
            reuse_weight=1., replay_group_size=2, gradient_scale=.5))
    torch.testing.assert_close(torch.stack([report.risk for report in reports]).mean(), objective)
    for parameter, gradient in zip(parameters, expected, strict=True):
        if gradient is None:
            assert parameter.grad is None
        else:
            torch.testing.assert_close(parameter.grad, gradient, rtol=2e-5, atol=2e-6)
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in reader.parameters())
    fast_ids = {id(value) for state in batch.initial_states for value in state.values}
    assert not fast_ids.intersection(map(id, parameters))
    assert all(not value.requires_grad for state in batch.initial_states for value in state.values)


def test_refresh_updates_search_operand_snapshot_without_reallocation(setup):
    graph, events, batch = setup
    dispatch = batch.wave.wave.execution.dispatch
    addresses = tuple(value.data_ptr() for name, value in dispatch.named_buffers() if "operand_table_" in name)
    before = tuple(value.clone() for name, value in dispatch.named_buffers() if "operand_table_" in name)
    with torch.no_grad():
        graph.templates.candidates["direct"].candidate.operand_store.tensor("encode.spatial").add_(.1)
    batch.load(events)
    after = tuple(value for name, value in dispatch.named_buffers() if "operand_table_" in name)
    assert tuple(value.data_ptr() for value in after) == addresses
    assert any(not torch.equal(a, b) for a, b in zip(before, after, strict=True))


def test_captured_batch_reuses_inputs_and_owned_decode(setup):
    graph, events, batch = setup
    if events.x0.device.type != "cuda":
        with pytest.raises(ValueError, match="CUDA"):
            batch.capture_search()
        return
    batch.capture_search()
    records = batch.execute()
    tape, endpoints = batch.decode(records, 0)
    old = replay_cooperative_dependencies_many(graph.query, events.writing_inputs(0), tape=tape,
        endpoints=endpoints, initial_states=(batch.initial_states[0],), score_decisions=True)
    expected = tuple(run.outputs["answer"].detach().clone() for run in old)
    other = make_spatial_events(events.spec, split="test", seed=47, count=2, device=events.x0.device)
    batch.load(other)
    changed = batch.execute()
    actual = tuple(value.clone() for value in tree_flatten(changed[-1].state)[0])
    direct = batch.search()
    for value, reference in zip(actual, tree_flatten(direct[-1].state)[0], strict=True):
        torch.testing.assert_close(value, reference)
    replayed = replay_cooperative_dependencies_many(graph.query, events.writing_inputs(0), tape=tape,
        endpoints=endpoints, initial_states=(batch.initial_states[0],), score_decisions=True)
    for run, reference in zip(replayed, expected, strict=True):
        torch.testing.assert_close(run.outputs["answer"], reference)


def test_exploration_is_replayable_and_preserves_raw_score_replay(setup):
    graph, events, batch = setup
    baseline = tuple(value.clone() for value in tree_flatten(batch.search()[-1].state)[0])
    batch.exploration(scale=0., seed=101)
    for actual, expected in zip(tree_flatten(batch.search()[-1].state)[0], baseline, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    batch.exploration(scale=.7, seed=101)
    noise = batch.selection_bias.clone()
    address = batch.selection_bias.data_ptr()
    if events.x0.device.type == "cuda":
        batch.capture_search()
    records = batch.execute()
    state = tuple(value.clone() for value in tree_flatten(records[-1].state)[0])
    completed = records[-1].state.frames.completed[0].nonzero().flatten()
    tape, endpoints = batch.decode(records, 0)
    scores = records[-1].state.scores[0, completed].clone()
    outputs = replay_cooperative_dependencies_many(graph.query, events.writing_inputs(0), tape=tape,
        endpoints=endpoints, initial_states=(batch.initial_states[0],), score_decisions=True)
    torch.testing.assert_close(torch.stack([r.decision_energy for r in outputs]), scores, rtol=2e-5, atol=2e-6)
    batch.exploration(scale=.7, seed=101)
    torch.testing.assert_close(batch.selection_bias, noise, rtol=0, atol=0)
    for actual, expected in zip(tree_flatten(batch.execute()[-1].state)[0], state, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    batch.exploration(scale=.7, seed=102)
    assert batch.selection_bias.data_ptr() == address
    assert not torch.equal(batch.selection_bias, noise)
    captured = tuple(value.clone() for value in tree_flatten(batch.execute()[-1].state)[0])
    for actual, expected in zip(tree_flatten(batch.search()[-1].state)[0], captured, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
