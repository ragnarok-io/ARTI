from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from arti import mechanisms as m
from benchmarks._federated_device_query import device_query_execution
from benchmarks._federated_recursive_search import search_recursive_graphs, start_recursive_search
from benchmarks.train_federated_branch_visible_federation import _SearchBranch, _ranked_candidates_many
from test_federated_recursive_search import _choice_child, _model
from test_federated_device_frontier import _compare_results


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_real_nested_query_wave_matches_reference(device, backend):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    query = _model().to(device)
    def run(use_device):
        with torch.no_grad(), device_query_execution(backend=backend) if use_device else nullcontext():
            return search_recursive_graphs((start_recursive_search(query, {"x": torch.ones(1, 2, device=device)}),),
                width=4, beam_width=8, preserve_effect_coverage=False, record_query_choices=True)
    _compare_results(run(False), run(True))


def _rank(query, arenas):
    rows = tuple(_SearchBranch(arena, next(query.network.parameters()).new_zeros(()), ()) for arena in arenas)
    return _ranked_candidates_many(SimpleNamespace(query=query), rows, tuple(query.candidates),
                                   steps=0, width=len(query.candidates) + 1, include_stop=True)


@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_device_query_uses_current_weights_and_variable_occupancy(backend):
    query = _choice_child("native", (1., 2., 3.))
    arenas = tuple(query._arena({"x": torch.full((1, 2), float(i))}) for i in range(3))
    with torch.no_grad(), device_query_execution(backend=backend):
        first = _rank(query, arenas)
        query.network[-1].bias[0].add_(1.25)
        second = _rank(query, arenas)
    assert not torch.equal(first[0][0][1], second[0][0][1])
    with torch.no_grad():
        expected = _rank(query, arenas)
    for actual_row, expected_row in zip(second, expected, strict=True):
        assert [id(c) for c, _ in actual_row] == [id(c) for c, _ in expected_row]
        torch.testing.assert_close(torch.stack([s for _, s in actual_row]), torch.stack([s for _, s in expected_row]))


def test_device_query_scope_leaves_gradient_replay_native():
    query = _choice_child("native", (1., 2.))
    with device_query_execution(backend="eager"):
        scores = _rank(query, (query._arena({"x": torch.ones(1, 2)}),))
    scores[0][0][1].backward()
    assert query.network[-1].weight.grad is not None


def test_tensor_encoder_rejects_integer_input_instead_of_coercing_it():
    from arti._formula_device_query import FormulaDeviceQuery
    from benchmarks._federated_device_query import device_ranked_candidates

    query = _choice_child("native", (1., 2.))
    encoder = m.FormulaProgramQueryTensorEncoderV1(2, 2)
    query.tensor_encoder = encoder
    query.network = torch.nn.Linear(len(query.slot_ids) * encoder.output_width, len(query.candidates) + 1)
    arena = query._arena({"x": torch.ones(1, 2, dtype=torch.int64)})
    branches = (_SearchBranch(arena, torch.zeros(()), ()),)
    with torch.no_grad(), device_query_execution(backend="eager"):
        assert device_ranked_candidates(query, branches, tuple(query.candidates), steps=0,
            width=2, include_stop=True, numerical_rejections=None, eligibility_records=None) is None
    with pytest.raises(ValueError, match="expects floating"):
        query._summarize_values(arena.values)
    kernel = FormulaDeviceQuery(query.network, slot_count=len(query.slot_ids),
        candidate_family_ids=[0] * (len(query.candidates) + 1), width=2, tensor_encoder=encoder)
    with pytest.raises(ValueError, match="expects floating"):
        kernel(tuple(torch.ones(1, 2, dtype=torch.int64) for _ in query.slot_ids),
               torch.ones(1, len(query.slot_ids), dtype=torch.bool),
               torch.ones(1, len(query.candidates) + 1, dtype=torch.bool), torch.zeros(1))


@pytest.mark.parametrize("backend", ["eager", "aot_eager"])
def test_native_tensor_encoder_and_six_effects_keep_search(backend):
    from benchmarks._federated_v4_federation import build_autonomous_effect_federation
    from benchmarks._federated_search_space_migration import named_query_view
    local = build_autonomous_effect_federation(hidden_dim=4, rank=4, seed=997, device=torch.device("cpu"),
        plastic_branches=2, min_operations=1, max_operations=2, max_effect_operations=2,
        ordinary_families=("gelu", "gated"))
    query = named_query_view(local.query)
    assert query.tensor_encoder is not None
    assert len({candidate.atom_ref for candidate in query.candidates
                if isinstance(candidate, m.FormulaProgramEffectCandidateV3)}) == 6
    x = torch.randn(1, 2, 4, generator=torch.Generator().manual_seed(55))
    def run(use_device):
        with torch.no_grad(), device_query_execution(backend=backend) if use_device else nullcontext():
            return search_recursive_graphs((start_recursive_search(query, {"x": x}),), width=16, beam_width=16,
                                           record_query_choices=True)
    _compare_results(run(False), run(True))
