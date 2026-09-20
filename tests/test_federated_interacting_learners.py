from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
import benchmarks.train_federated_interacting_learners as harness

from arti import mechanisms as m
from benchmarks._federated_peer_composition import compose_peer_query
from benchmarks.train_federated_interacting_learners import (
    Learner, build_learners, interact, learner_objective, load_round,
    make_episode, save_round, train_round,
)


def _tiny_learners():
    value_type = m.TensorType(("B", "N", "D"), ("B", "N", 3), dtype="floating", domain="activation")
    factor_type = m.TensorType(("D",), (3,), dtype="floating", domain="activation")
    learners = []
    for index in range(2):
        candidates = []
        for source in ("x", "message"):
            for choice, gain in enumerate((0.6, 1.2)):
                x = m.InputBinding("x", value_type)
                weight = m.BankBinding("weight", "arti/interacting-test@1", "weight", factor_type)
                scale = m.BankBinding("gain", "arti/interacting-test@1", "gain", factor_type)
                program = m.FormulaProgram.build(outputs=(m.scale(m.scale(x, weight), scale),))
                name = f"{source}-{choice}"
                candidates.append(m.FormulaProgramTensorCandidateV4(m.FormulaProgramCandidateV3(
                    name, program, input_slots={"x": source}, output_slots={program.outputs[0]: "out"},
                    operands={"weight": torch.ones(3), "gain": torch.full((3,), gain + 0.1 * index)},
                    trainable_operands=("gain",),
                ), plastic_bank_slot="weight", bank_owner_id=f"owner-{name}"))
        query = m.FormulaProgramQueryV5(slot_ids=("x", "message", "out"), candidates=candidates,
                                      terminal_slots={"output": "out"}, max_steps=1, hidden_dim=8)
        with torch.no_grad():
            query.network[-1].weight.zero_()
            query.network[-1].bias.copy_(torch.tensor([1., 0., 3., 2., 0.]))
        learners.append(Learner(f"learner-{index}", SimpleNamespace(query=query),
                                torch.optim.AdamW([p for p in query.parameters() if p.requires_grad], lr=1e-3)))
    return tuple(learners)


def _episode(seed=5):
    return make_episode(seed=seed, hidden_dim=3, device=torch.device("cpu"))


def test_both_learners_receive_complete_supports_but_distinct_queries():
    episode = _episode()
    assert len(episode.supports) == 3
    assert not torch.equal(*episode.queries)
    for index, (a, b) in enumerate(((0, 1), (1, 2))):
        expected = episode.queries[index] + (episode.supports[a][:, 1:] + episode.supports[b][:, 1:]) / 2**0.5
        torch.testing.assert_close(episode.targets[index], expected)


def test_real_build_preserves_six_families_counts_and_separate_storage():
    learners = build_learners(seed=4, hidden_dim=3, branches=2, device=torch.device("cpu"),
                              max_operations=1, max_effect_operations=1)
    params = [{p.data_ptr() for p in learner.query.parameters()} for learner in learners]
    assert not params[0] & params[1]
    banks = [{owner.value.data_ptr() for owner in learner.query.owner_states} for learner in learners]
    assert not banks[0] & banks[1]
    for learner in learners:
        effects = [module for module in learner.query.modules() if isinstance(module, m.FormulaProgramEffectCandidateV3)]
        assert len({effect.atom_ref for effect in effects}) == 6
        assert all(effect.execution_count_tensor().item() == 2 for effect in effects)
        optimized = {id(p) for group in learner.optimizer.param_groups for p in group["params"]}
        assert not optimized & {id(owner.value) for owner in learner.query.owner_states}


def test_external_tensor_input_does_not_mount_peer_or_invent_lineage():
    source = _tiny_learners()[0].query
    local = m.FormulaProgramQueryV5(slot_ids=("x", "out"), candidates=tuple(source.candidates[:2]),
                                   terminal_slots={"output": "out"}, max_steps=1, hidden_dim=8)
    root = compose_peer_query({"local": local}, hidden_dim=3, external_inputs=("x", "message"))
    message = torch.randn(1, 2, 3)
    arena = root._arena({"x": message * 2, "message": message})
    assert arena.producer("message") is None
    candidate = next(c for c in root.candidates if c.candidate_id == "return-external-message")
    output = candidate(arena)
    torch.testing.assert_close(output.values.get("terminal"), message)
    assert not output.proposals
    with pytest.raises(ValueError, match="alias"):
        compose_peer_query({"local": local}, hidden_dim=3, external_inputs=("x", "terminal"))


def test_interaction_is_same_round_detached_and_target_independent():
    learners, episode = _tiny_learners(), _episode()
    result = interact(learners, episode)
    for index in range(2):
        final = result.events[index][-1]
        service = result.events[1 - index][result.service_index]
        torch.testing.assert_close(final.inputs["message"], service.output)
        assert not final.inputs["message"].requires_grad
        assert final.inputs["message"].data_ptr() != service.output.data_ptr()
        for event, support in zip(result.events[index], episode.supports):
            assert event.inputs["x"] is support
    different_targets = replace(episode, targets=tuple(t + 100 for t in episode.targets))
    other = interact(learners, different_targets)
    assert [[e.route for e in h] for h in result.events] == [[e.route for e in h] for h in other.events]
    for left, right in zip(result.events, other.events):
        torch.testing.assert_close(left[-1].output, right[-1].output)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_independent_interaction_batch_preserves_event_order_and_peer_messages(device):
    from benchmarks._federated_captured_search import captured_search_execution
    from benchmarks.probe_federated_episode_batch import _compare

    learners = _tiny_learners()
    for learner in learners:
        learner.query.to(device)
    episodes = tuple(make_episode(seed=seed, hidden_dim=3, device=torch.device(device)) for seed in (3, 9, 27))
    expected = tuple(interact(learners, e, width=3) for e in episodes)
    with captured_search_execution(horizon=8) as backend:
        actual = harness.interact_many(learners, episodes, width=3)
        assert _compare(expected, actual)["output_max_abs_error"] < 1e-6
        if device == "cuda":
            assert backend.batched_searches == 30
            assert not backend.fallbacks
    for interaction in actual:
        for index in range(2):
            torch.testing.assert_close(interaction.events[index][-1].inputs["message"],
                                       interaction.events[1 - index][interaction.service_index].output)
    assert all(p.grad is None for learner in learners for p in learner.query.parameters())


def test_frozen_evaluation_recomputes_messages_after_reset():
    from benchmarks.evaluate_federated_interacting_learners import evaluate_episode, reset_service_outputs

    learners, episode = _tiny_learners(), _episode()
    for learner in learners:
        learner.query.requires_grad_(False)
    row = evaluate_episode(learners, episode, width=3, reset_control=True)
    # This fixture has no write primitive, so a reset must change nothing.
    assert all(r["pre_service_revision_sum"] == 0 and r["support_effect_nodes"] == 0 for r in row["learners"])
    assert all(r["reset_output_max_abs_difference"] == 0 and r["mse"] == r["reset_mse"] for r in row["learners"])
    first = reset_service_outputs(learners, episode, width=3)
    second = reset_service_outputs(learners, replace(episode, targets=tuple(t + 100 for t in episode.targets)), width=3)
    for a, b in zip(first, second, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert all(p.grad is None for learner in learners for p in learner.query.parameters())


def test_local_objective_has_no_peer_vjp_and_external_credit_trains_sender_query():
    learners, episode = _tiny_learners(), _episode()
    interaction = interact(learners, episode)
    objective, _ = learner_objective(learners, episode, interaction, 0, external_credit=0.5)
    peer_parameters = tuple(p for p in learners[1].query.parameters() if p.requires_grad)
    assert all(g is None for g in torch.autograd.grad(objective, peer_parameters, allow_unused=True, retain_graph=True))
    parameter = learners[0].query.network[-1].bias
    with_credit = torch.autograd.grad(objective, parameter)[0]
    objective, _ = learner_objective(learners, episode, interaction, 0, external_credit=0)
    without_credit = torch.autograd.grad(objective, parameter)[0]
    assert (with_credit - without_credit).abs().max() > 1e-7


def test_input_credit_trains_sender_contents_without_connecting_peer_parameters():
    learners, episode = _tiny_learners(), _episode()
    interaction = interact(learners, episode)
    sender_gain = learners[0].query.candidates[0].candidate.operand_store.tensor("gain")
    scalar, _ = learner_objective(learners, episode, interaction, 0, message_credit="scalar-route")
    baseline = torch.autograd.grad(scalar, sender_gain, allow_unused=True)[0]
    objective, row = learner_objective(learners, episode, interaction, 0, message_credit="input-vjp")
    peer = tuple(p for p in learners[1].query.parameters() if p.requires_grad)
    assert all(g is None for g in torch.autograd.grad(objective, peer, allow_unused=True, retain_graph=True))
    improved = torch.autograd.grad(objective, sender_gain, allow_unused=True)[0]
    assert row["message_input_gradient_norm"] > 0
    assert row["service_replay_max_abs_error"] == 0
    assert improved is not None and torch.isfinite(improved).all()
    assert (improved if baseline is None else improved - baseline).abs().max() > 1e-7


@pytest.mark.parametrize("recipe", ("episode-beam", "retained-vjp"))
def test_trajectory_input_credit_reaches_an_unselected_sender_service(recipe):
    learners, episode = _tiny_learners(), _episode()
    interaction = interact(learners, episode,
                           plasticity_participation="retained" if recipe == "retained-vjp" else "winner")
    gain = learners[0].query.candidates[1].candidate.operand_store.tensor("gain")
    assert all(row["candidate_id"] != "x-1" for event in interaction.events[0] for row in event.route)
    baseline, _ = learner_objective(learners, episode, interaction, 0,
                                   message_credit="input-vjp", route_credit=recipe)
    before = torch.autograd.grad(baseline, gain, allow_unused=True)[0]
    objective, row = learner_objective(learners, episode, interaction, 0,
                                      message_credit="trajectory-input-vjp", route_credit=recipe)
    peer_parameters = tuple(p for p in learners[1].query.parameters() if p.requires_grad)
    assert all(g is None for g in torch.autograd.grad(objective, peer_parameters, allow_unused=True, retain_graph=True))
    after = torch.autograd.grad(objective, gain)[0]
    assert row["candidate_message_gradient_norm"] > 0
    assert (after if before is None else after - before).abs().max() > 1e-7


def test_fixed_suffix_cannot_silently_accept_trajectory_input_recipe():
    with pytest.raises(ValueError, match="requires episode-beam"):
        learner_objective(None, None, None, 0, route_credit="fixed-suffix", message_credit="trajectory-input-vjp")


@pytest.mark.parametrize("option", (("--candidate-exploration", "gumbel"),
                                   ("--message-credit", "trajectory-input-vjp")))
def test_cli_rejects_incompatible_recipe_before_allocating_or_loading(option, monkeypatch, tmp_path):
    def unexpected(**kwargs):
        raise AssertionError("invalid configuration must not allocate models")
    monkeypatch.setattr(harness, "build_learners", unexpected)
    output = tmp_path / "unused"
    monkeypatch.setattr("sys.argv", ["train", "--output-dir", str(output),
                                     "--route-credit", "fixed-suffix", *option])
    with pytest.raises(SystemExit) as error:
        harness.main()
    assert error.value.code == 2
    assert not output.exists()


def test_random_training_candidates_leave_actual_interaction_unchanged():
    learners, episode = _tiny_learners(), _episode()
    actual = interact(learners, episode, width=3)
    before = [[e.route for e in h] for h in actual.events]
    objective, row = learner_objective(learners, episode, actual, 0, width=3,
                                      message_credit="trajectory-input-vjp", exploration_seed=123)
    objective.backward()
    assert row["candidate_exploration_seed"] == 123
    assert row["service_replay_max_abs_error"] == 0
    assert all(p.grad is None for p in learners[1].query.parameters())
    after = interact(learners, replace(episode, targets=tuple(t + 100 for t in episode.targets)), width=3)
    assert before == [[e.route for e in h] for h in after.events]
    for a, b in zip(actual.events, after.events, strict=True):
        torch.testing.assert_close(a[-1].output, b[-1].output)


def test_each_optimizer_waits_for_both_backwards_and_resume_is_exact(tmp_path, monkeypatch):
    learners, episode = _tiny_learners(), _episode()
    before = [[p.detach().clone() for p in learner.query.parameters()] for learner in learners]
    for learner in learners:
        original = learner.optimizer.step

        def checked_step(*args, _original=original, **kwargs):
            assert all(any(p.grad is not None for p in item.query.parameters()) for item in learners)
            return _original(*args, **kwargs)

        monkeypatch.setattr(learner.optimizer, "step", checked_step)
    interaction, metrics = train_round(learners, episode)
    for index, learner in enumerate(learners):
        assert any(not torch.equal(a, b) for a, b in zip(before[index], learner.query.parameters()))
        assert learner.optimizer.state
    save_round(tmp_path / "round-1", learners, step=1, config={"fixture": True},
               metrics=metrics, interaction=interaction)
    restored = _tiny_learners()
    assert load_round(tmp_path / "round-1", restored)["step"] == 1
    for a, b in zip(learners, restored):
        for p, q in zip(a.query.parameters(), b.query.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
    _, left = train_round(learners, _episode(8))
    _, right = train_round(restored, _episode(8))
    assert [{k: v for k, v in row.items() if not k.endswith("_seconds")} for row in left["learners"]] == [
        {k: v for k, v in row.items() if not k.endswith("_seconds")} for row in right["learners"]
    ]
    for a, b in zip(learners, restored):
        for p, q in zip(a.query.parameters(), b.query.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_training_credit_width_and_recipe_follow_actual_configuration(monkeypatch):
    original = learner_objective
    received = []

    def record(*args, **kwargs):
        received.append((kwargs["width"], kwargs["route_credit"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(harness, "learner_objective", record)
    train_round(_tiny_learners(), _episode(), width=4, route_credit="fixed-suffix")
    assert received == [(4, "fixed-suffix"), (4, "fixed-suffix")]


def test_peer_credit_does_not_count_duplicate_final_continuations(monkeypatch):
    learners, episode = _tiny_learners(), _episode()
    interaction = interact(learners, episode)
    parameter = learners[0].query.network[-1].bias

    def external_gradient():
        base, _ = learner_objective(learners, episode, interaction, 0, external_credit=0)
        no_credit = torch.autograd.grad(base, parameter)[0]
        objective, row = learner_objective(learners, episode, interaction, 0, external_credit=0.5)
        return torch.autograd.grad(objective, parameter)[0] - no_credit, row

    expected, original_row = external_gradient()
    search = harness.search_episode_beam

    def duplicate(*args, **kwargs):
        result = search(*args, **kwargs)
        return replace(result, result=replace(result.result, branches=(
            *result.result.branches, result.result.branches[0], result.result.branches[0],
        )))

    monkeypatch.setattr(harness, "search_episode_beam", duplicate)
    actual, duplicate_row = external_gradient()
    torch.testing.assert_close(actual, expected, atol=1e-7, rtol=1e-5)
    assert original_row["unique_service_prefixes"] == duplicate_row["unique_service_prefixes"]
    assert original_row["external_final_task_risk"] == duplicate_row["external_final_task_risk"]


def test_frozen_deployment_keeps_parameters_and_allows_message_disable():
    learners = _tiny_learners()
    for learner in learners:
        learner.query.requires_grad_(False)
    before = [{name: tensor.detach().clone() for name, tensor in learner.query.state_dict().items()}
              for learner in learners]
    with_messages = interact(learners, _episode())
    without_messages = interact(learners, _episode(), communicate=False)
    assert any(not torch.equal(a[-1].output, b[-1].output)
               for a, b in zip(with_messages.events, without_messages.events))
    for original, learner in zip(before, learners):
        for name, tensor in learner.query.state_dict().items():
            torch.testing.assert_close(tensor, original[name], rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_real_cuda_adam_resume_keeps_step_on_cpu_and_moments_on_device(tmp_path):
    def build():
        return build_learners(seed=17, hidden_dim=3, branches=2, device=torch.device("cuda"),
                              max_operations=1, max_effect_operations=1)

    learners = build()
    for learner in learners:
        for parameter in learner.query.network.parameters():
            parameter.grad = torch.full_like(parameter, 0.1)
        learner.optimizer.step()
    save_round(tmp_path / "cuda-round", learners, step=1, config={}, metrics={})
    restored = build()
    load_round(tmp_path / "cuda-round", restored)
    for learner in restored:
        for state in learner.optimizer.state.values():
            assert state["step"].device.type == "cpu"
            assert state["exp_avg"].device.type == "cuda"
    for a, b in zip(learners, restored):
        for p, q in zip(a.query.parameters(), b.query.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
