from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

import arti
from arti import mechanisms as m
from arti._formula_device_pools import FormulaDevicePoolLayout
from benchmarks._federated_episode_risk import replay_cooperative_episode_panel
from benchmarks._federated_product_replay import replay_cooperative_dependencies_many
from benchmarks._federated_spatial_episodes import SpatialEpisodeSpec, make_calibrated_spatial_episodes
from benchmarks._federated_spatial_episode_search import collect_spatial_episode_panel, completed_bank_snapshots
from benchmarks._federated_spatial_events import SpatialEventSpec
from benchmarks._federated_spatial_programs import SpatialProgramSpec, build_spatial_templates, build_spatial_search_graph
from benchmarks.probe_federated_spatial_event import SpatialDeviceBatch


def test_snapshots_follow_typed_handles_event_rows_and_real_revisions():
    source_ref = arti.canonical_contract_reference("arti/spatial-episode-source@1")
    refs = tuple(m.BankSlotRef(f"p{i}", "0" * 64, "value", source_ref, "bank") for i in range(2))
    initial = m.FormulaProgramBankState(refs, (torch.zeros(1, 2), torch.zeros(2, 1, dtype=torch.float64)), (0, 0))
    layout = FormulaDevicePoolLayout.from_samples(initial.values, (4, 4))
    pools = tuple(torch.arange(2 * 5 * 2, dtype=dtype).reshape(2, 5, *shape)
                  for dtype, shape in zip(layout.dtypes, layout.shapes, strict=True))
    frames = SimpleNamespace(completed=torch.tensor([[True, False, True], [False, True, False]]),
        bank_value_handles=torch.tensor([[[2, 8], [0, 5], [1, 7]], [[0, 5], [3, 6], [0, 5]]]),
        bank_revisions=torch.tensor([[[8, 9], [0, 0], [10, 11]], [[0, 0], [12, 13], [0, 0]]]))
    records = (SimpleNamespace(state=SimpleNamespace(frames=frames, scores=torch.tensor([[3., 0., 2.], [0., 7., 0.]]))),)
    batch = SimpleNamespace(initial_states=(initial, initial), bank=pools, bank_layout=layout)
    left, scores = completed_bank_snapshots(batch, records, 0)
    right, other = completed_bank_snapshots(batch, records, 1)
    assert scores == (3., 2.) and other == (7.,)
    assert left[0].revisions == (8, 9) and right[0].revisions == (12, 13)
    torch.testing.assert_close(left[0].values, (pools[0][0, 2], pools[1][0, 3]))
    torch.testing.assert_close(left[1].values, (pools[0][0, 1], pools[1][0, 2]))
    torch.testing.assert_close(right[0].values, (pools[0][1, 3], pools[1][1, 1]))
    frames.bank_value_handles[0, 0, 1] = 0
    with pytest.raises(ValueError, match="typed pool"):
        completed_bank_snapshots(batch, records, 0)


def test_parent_padding_and_incomplete_branch_are_not_retained(monkeypatch):
    from benchmarks import _federated_spatial_episode_search as collector
    empty = m.FormulaProgramBankState.empty()
    counts = ((2, 1), (0, 1), (1, 1))
    decoded = []
    class Batch:
        event_count = 2
        selection_bias = None
        graph = SimpleNamespace(query=SimpleNamespace(initial_bank_state=lambda: empty, candidates=()))
        calls = 0
        def load(self, event, states):
            self.initial_states = states
        def execute(self):
            self.calls += 1
            return self.calls - 1
        def decode(self, records, row, **kwargs):
            decoded.append((records, row))
            return SimpleNamespace(occurrences=(1,), initial_states=(empty,)), tuple(range(counts[records][row]))
    def snapshots(batch, records, row):
        n = counts[records][row]
        return (empty,) * n, tuple(10. - i for i in range(n))
    monkeypatch.setattr(collector, "completed_bank_snapshots", snapshots)
    batch = Batch()
    episode = SimpleNamespace(events=(SimpleNamespace(x0=torch.zeros(2, 1)),) * 2)
    panel = collect_spatial_episode_panel(batch, episode, panel_width=2)
    assert panel.batch_executions == 3 and panel.searched_events == 5 and panel.padded_events == 1
    assert decoded == [(0, 0), (0, 1), (1, 1), (2, 0)]
    assert panel.paths[0][0].energy == 19. and panel.paths[1][0].energy == 20.


@pytest.fixture
def setup():
    torch.set_num_threads(1)
    spec = SpatialEpisodeSpec(SpatialEventSpec(3, 4, 3, reuse_questions=3), length=2)
    episode = make_calibrated_spatial_episodes(spec, split="train", family_seed=7, scene_seed=29, count=2)
    templates = build_spatial_templates(spec.event, SpatialProgramSpec(latent_positions=2, hidden_dim=8, rank=4, banks=2),
                                       seed=17, device="cpu")
    graph = build_spatial_search_graph(templates, latent_slots=2, views=1, cooperation_width=2, max_steps=12)
    batch = SpatialDeviceBatch.prepare(graph, episode.events[0], width=2, heads=3, steps=12)
    return graph, episode, batch


def assert_bank(actual, expected):
    assert actual.slot_refs == expected.slot_refs and actual.revisions == expected.revisions
    torch.testing.assert_close(actual.values, expected.values, rtol=2e-5, atol=2e-6)


def test_completed_pool_snapshots_match_native_and_survive_reuse(setup):
    graph, episode, batch = setup
    initial = tuple(replace(state, values=tuple(torch.full_like(v, .1 * (row + 1)) for v in state.values),
                            revisions=tuple(row + 3 for _ in state.revisions))
                    for row, state in enumerate(batch.initial_states))
    batch.load(episode.events[0], initial)
    records = batch.execute()
    saved = []
    for row in range(2):
        tape, endpoints = batch.decode(records, row, include_decisions=True, include_port_inputs=False)
        banks, scores = completed_bank_snapshots(batch, records, row)
        with torch.no_grad():
            runs = replay_cooperative_dependencies_many(graph.query, episode.events[0].writing_inputs(row),
                tape=tape, endpoints=endpoints, initial_states=(initial[row],), score_decisions=True)
        for bank, score, run in zip(banks, scores, runs, strict=True):
            assert_bank(bank, run.bank_state)
            assert score == pytest.approx(float(run.decision_energy), rel=2e-5, abs=2e-6)
            assert all(not value.requires_grad for value in bank.values)
        saved.extend((bank, tuple(v.clone() for v in bank.values)) for bank in banks)
    for pool in batch.bank:
        pool.fill_(99.)
    batch.load(episode.events[1], initial)
    batch.execute()
    for bank, values in saved:
        torch.testing.assert_close(bank.values, values, rtol=0, atol=0)


def test_episode_panel_matches_full_live_replay_and_does_not_install_state(setup):
    graph, episode, batch = setup
    initial = graph.query.initial_bank_state()
    before = {name: value.clone() for name, value in graph.query.state_dict().items()}
    panel = collect_spatial_episode_panel(batch, episode, panel_width=2)
    assert panel.batch_executions <= 3
    assert panel.searched_events + panel.padded_events == panel.batch_executions * 2
    assert panel.decoded_occurrences > 0
    # Replay after overwriting device scratch: only owned tapes/Bank snapshots survive.
    for pool in (*batch.data, *batch.bank):
        pool.zero_()
    for row, paths in enumerate(panel.paths):
        assert 0 < len(paths) <= 2 and all(len(path.steps) == 2 for path in paths)
        assert paths[0].energy >= paths[-1].energy
        runs = replay_cooperative_episode_panel(graph.query,
            tuple(event.writing_inputs(row) for event in episode.events), tuple(path.steps for path in paths))
        torch.testing.assert_close(runs.energies, torch.tensor([path.energy for path in paths]), rtol=2e-5, atol=2e-6)
        for path, run in zip(paths, runs.terminals, strict=True):
            assert_bank(path.return_state, run.bank_state)
            assert all(v.untyped_storage().nbytes() == v.numel() * v.element_size() for v in path.return_state.values)
            assert path.steps[0].tape.initial_states[0] is not path.steps[-1].tape.initial_states[0]
    assert_bank(graph.query.initial_bank_state(), initial)
    torch.testing.assert_close(graph.query.state_dict(), before, rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in graph.query.parameters())


def test_panel_one_matches_explicit_carry_and_never_uses_labels(setup):
    graph, episode, batch = setup
    panel = collect_spatial_episode_panel(batch, episode, panel_width=1)
    states = (graph.query.initial_bank_state(),) * 2
    energies = [0., 0.]
    for event in episode.events:
        batch.load(event, states)
        records = batch.execute()
        next_states = []
        for row in range(2):
            banks, scores = completed_bank_snapshots(batch, records, row)
            best = max(range(len(scores)), key=scores.__getitem__)
            next_states.append(banks[best])
            energies[row] += scores[best]
        states = tuple(next_states)
    for row in range(2):
        assert panel.paths[row][0].energy == pytest.approx(energies[row])
        assert_bank(panel.paths[row][0].return_state, states[row])
    changed = replace(episode, events=tuple(replace(event, answer=event.answer + 100.,
        readonly_questions=event.readonly_questions + 100., readonly_answers=event.readonly_answers + 100.)
        for event in episode.events))
    again = collect_spatial_episode_panel(batch, changed, panel_width=1)
    for a, b in zip(panel.paths, again.paths, strict=True):
        assert a[0].energy == b[0].energy
        assert_bank(a[0].return_state, b[0].return_state)
