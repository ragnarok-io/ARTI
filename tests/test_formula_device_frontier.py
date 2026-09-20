from types import SimpleNamespace

import pytest
import torch

from arti._formula_device_frontier import TensorRouteFrontier, TensorFrontierRecord
from benchmarks.train_federated_autonomous_federation import _coverage_prune, _effect_coverage_labels
from benchmarks.train_federated_branch_visible_federation import _prune


def _pool(device):
    names = ((), ("a",), ("a", "child/stop"), ("a", "child/stop", "stop"),
             ("a", "child/step-10"), ("a", "child/step-2"), ("a", "child/step-2"), ("b",), ())
    vocabulary = sorted({name for route in names for name in route})
    routes = torch.tensor([[vocabulary.index(name) for name in row] + [-1] * (4 - len(row)) for row in names], device=device)
    scores = torch.tensor([0., 0., 0., 0., 0., 0., 0., -torch.inf, 0.], device=device)
    items = [SimpleNamespace(log_probability=score, route=tuple({
        "candidate_id": name, "kind": "effect", "atom_ref": "atom-" + str(i % 3),
    } for name in route)) for i, (score, route) in enumerate(zip(scores, names, strict=True))]
    items[8] = items[0]
    ids = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 0], device=device)
    labels = tuple(_effect_coverage_labels(item) for item in items)
    families = sorted({label for row in labels for label in row})
    membership = torch.tensor([[family in row for family in families] for row in labels], device=device)
    return items, scores, routes, ids, membership


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("coverage", [False, True])
@pytest.mark.parametrize("width", [1, 5, 12])
def test_device_lexicographic_routes_and_refill(device, coverage, width):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    items, scores, routes, ids, membership = _pool(device)
    alive = torch.ones(len(items), dtype=torch.bool, device=device)
    ready = torch.zeros_like(alive)
    select, record = TensorRouteFrontier(width, coverage), TensorFrontierRecord()
    for _ in range(4):
        packet = select(scores, alive, ready, membership, routes, ids)
        remaining = [item for item, valid in zip(items, alive.tolist(), strict=True) if valid]
        expected = (_coverage_prune if coverage else _prune)(remaining, beam_width=width)
        actual = [items[i] for i in packet[:, 0].tolist() if i >= 0]
        assert [id(item) for item in actual] == [id(item) for item in expected]
        accepted = torch.ones(packet.shape[0], dtype=torch.bool, device=device)
        accepted[::2] = False
        old_alive, old_ready = alive.clone(), ready.clone()
        alive, ready = record(alive, ready, ids, packet, accepted)
        for i, identity in enumerate(ids.tolist()):
            changes = [ok for (j, pending), ok in zip(packet.tolist(), accepted.tolist(), strict=True)
                       if j >= 0 and pending and int(ids[j]) == identity]
            assert bool(alive[i]) == (bool(old_alive[i]) and all(changes))
            assert bool(ready[i]) == (bool(old_ready[i]) or any(changes))


def test_route_selection_and_record_form_one_dynamic_graph():
    _, scores, routes, ids, membership = _pool("cpu")
    alive, ready = torch.ones_like(scores, dtype=torch.bool), torch.zeros_like(scores, dtype=torch.bool)
    select, record = TensorRouteFrontier(5, True), TensorFrontierRecord()
    graphs = []
    def backend(graph, _inputs):
        graphs.append(graph)
        return graph.forward
    def wave(scores, alive, ready, membership, routes, ids, accepted):
        packet = select(scores, alive, ready, membership, routes, ids)
        return packet, *record(alive, ready, ids, packet, accepted)
    compiled = torch.compile(wave, backend=backend, fullgraph=True)
    for index in range(4):
        accepted = torch.arange(5) % 2 == index % 2
        torch.testing.assert_close(compiled(scores, alive, ready, membership, routes, ids, accepted),
                                   wave(scores, alive, ready, membership, routes, ids, accepted))
        routes = routes.roll(1, dims=0)
        scores = scores.roll(1)
        alive[index] = False
    assert len(graphs) == 1
    assert not any("_local_scalar_dense" in str(node.target) for node in graphs[0].graph.nodes)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_inductor_route_order_and_refill_stay_on_device():
    _, scores, routes, ids, membership = _pool("cuda")
    alive, ready = torch.ones_like(scores, dtype=torch.bool), torch.zeros_like(scores, dtype=torch.bool)
    class Wave(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.select, self.record = TensorRouteFrontier(5, True), TensorFrontierRecord()

        def forward(self, scores, alive, ready, membership, routes, ids, accepted):
            packet = self.select(scores, alive, ready, membership, routes, ids)
            alive, ready = self.record(alive, ready, ids, packet, accepted)
            return self.select(scores, alive, ready, membership, routes, ids), alive, ready
    module = Wave()
    accepted = torch.arange(5, device="cuda") % 2 == 0
    inputs = (scores, alive, ready, membership, routes, ids, accepted)
    compiled = torch.compile(torch.export.export(module, inputs).module(), backend="inductor", fullgraph=True)
    with torch.no_grad():
        for _ in range(3):
            expected = module(*inputs)
            actual = compiled(*inputs)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            inputs = (scores.roll(1), actual[1].clone(), actual[2].clone(), membership,
                      routes.roll(1, dims=0), ids, ~accepted)
