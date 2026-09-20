from __future__ import annotations

from dataclasses import replace
from functools import partial
import json
import math
from types import SimpleNamespace
import weakref

import pytest
import torch
from safetensors.torch import load_file

from arti import mechanisms
from benchmarks._federated_v4_federation import build_autonomous_effect_federation
from benchmarks.generate_qwen_federated_autonomous import generate_batch, prepare_readouts
from benchmarks.probe_qwen_federated_online_meta import Episode
from benchmarks.train_qwen_federated_autonomous_federation import (
    _branch_readout,
    _causal_checkpoint_eligible,
    _model_hidden_size,
    _should_replace_checkpoint,
    _validate_search_budget,
    evaluate_frozen_qwen_adaptation,
    expected_qwen_batch_loss,
    search_cached_episode,
)
from benchmarks.train_qwen_federated_online_meta import CachedEpisode


def _federation() -> object:
    with torch.random.fork_rng():
        torch.manual_seed(20261041)
        return build_autonomous_effect_federation(
            hidden_dim=4,
            rank=3,
            seed=20261041,
            device=torch.device("cpu"),
            plastic_branches=4,
            min_operations=1,
            max_operations=2,
            effect_families=("outer", "blend"),
            event_write_layout="learned-key-value-gates",
        )


def _episode(answer: int, *, event_value: float) -> CachedEpisode:
    generator = torch.Generator().manual_seed(100 + answer)
    answer_ids = torch.tensor([[answer, (answer + 1) % 4]], dtype=torch.int64)
    return CachedEpisode(
        Episode("pair", f"event-{answer}", "latest", str(answer)),
        torch.full((1, 5, 4), event_value),
        torch.randn(1, 4, 4, generator=generator),
        torch.randn(2, 4, generator=generator),
        torch.randn(2, 4, generator=generator),
        torch.ones(1, 2, dtype=torch.int64),
        answer_ids,
        2,
    )


def _fake_suffix(
    _model: object,
    hidden_rows: object,
    *,
    prompt_lengths: object,
    answer_lengths: object,
    layer_index: int,
) -> tuple[torch.Tensor, ...]:
    del layer_index
    rows = tuple(hidden_rows)
    lengths = tuple(answer_lengths)
    prompts = tuple(prompt_lengths)
    return tuple(
        row[0, prompt - 1 : prompt + length - 1, :].float()
        for row, prompt, length in zip(rows, prompts, lengths, strict=True)
    )


def _force_effect_route(federation: object) -> None:
    query = federation.query  # type: ignore[attr-defined]
    ordinary = next(
        candidate
        for candidate in federation.transition_layers[0]  # type: ignore[attr-defined]
        if isinstance(candidate, mechanisms.FormulaProgramTensorCandidateV3)
    )
    effect = next(
        candidate
        for candidate in federation.transition_layers[1]  # type: ignore[attr-defined]
        if isinstance(candidate, mechanisms.FormulaProgramEffectCandidateV3)
    )
    terminal = federation.terminal_layers[1][0]  # type: ignore[attr-defined]
    with torch.no_grad():
        for parameter in query.network.parameters():
            parameter.zero_()
        bias = query.network[-1].bias
        bias[query.candidate_ids.index(ordinary.candidate_id)] = 10.0
        bias[query.candidate_ids.index(effect.candidate_id)] = 20.0
        bias[query.candidate_ids.index(terminal.candidate_id)] = 10.0


def _peer_fixture():
    from benchmarks._federated_peer_composition import mount_peer_federation

    source = _federation()
    _force_effect_route(source)
    federation, _ = mount_peer_federation(source, hidden_dim=4, peer_slots=2, seed=942)
    with torch.no_grad():
        for parameter in federation.query.network.parameters():
            parameter.zero_()
        for index, name in enumerate(federation.query.candidate_ids):
            federation.query.network[-1].bias[index] = (
                8.0 if "x-to-peer" in name else 12.0 if name.startswith("join-scale") else -12.0
            )
    return federation


def test_recursive_qwen_search_and_readout_keep_answers_out_of_adaptation(monkeypatch):
    from benchmarks.train_qwen_federated_autonomous_federation import _replay_route

    federation = _peer_fixture()
    item = _episode(0, event_value=1.0)
    altered = replace(item, student_hidden=torch.cat((item.student_hidden[:, :2], torch.full((1, 6, 4), 100.0)), dim=1),
                      teacher_logits=torch.full((6, 4), -100.0), answer_ids=torch.full((1, 6), 3, dtype=torch.long))
    searches = tuple(search_cached_episode(federation, sample, width=16, beam_width=16,
                                           preserve_effect_coverage=False) for sample in (item, altered))
    for left, right in zip(searches[0].branches, searches[1].branches, strict=True):
        assert left.route == right.route
        torch.testing.assert_close(left.log_probability, right.log_probability)
        torch.testing.assert_close(left.arena.committed_state().values, right.arena.committed_state().values)
    event = searches[0].prompt
    assert sum(row["kind"] == "call" for row in event.winner.route) == 2
    assert len(event.proposals) >= 2
    replay = _replay_route(federation, _prompt_hidden_for_test(item), searches[0].support.bank_state, event.route)
    torch.testing.assert_close(replay.bank_state.values, event.bank_state.values)
    discarded = _replay_route(federation, _prompt_hidden_for_test(item), searches[0].support.bank_state,
                              event.route, apply_effects=False)
    assert discarded.proposal_count == len(event.proposals)
    assert discarded.bank_state.revisions == searches[0].support.bank_state.revisions

    def forbidden(*args, **kwargs):
        raise AssertionError("answer readout queried or wrote Bank")

    for module in federation.query.modules():
        if isinstance(module, mechanisms.FormulaProgramQueryV5):
            monkeypatch.setattr(module, "query", forbidden)
        elif isinstance(module, mechanisms.FormulaProgramEffectCandidateV3):
            monkeypatch.setattr(module, "forward", forbidden)
    branch = searches[0].branches[searches[0].deployed_index]
    hidden = item.student_hidden.clone().requires_grad_()
    result = _branch_readout(federation, replace(item, student_hidden=hidden), branch)
    other = _branch_readout(federation, altered, branch)
    torch.testing.assert_close(result[:, :2], other[:, :2])
    gradient = torch.autograd.grad(result[:, :2].sum(), hidden)[0]
    assert torch.count_nonzero(gradient[:, 2:]) == 0
    assert all(owner.revision.item() == 0 for owner in federation.query.owner_states)


def _prompt_hidden_for_test(item):
    return item.student_hidden[:, :item.student_prompt_length].float()


def test_recursive_qwen_final_loss_reaches_parent_child_and_join(monkeypatch):
    monkeypatch.setattr("benchmarks.train_qwen_federated_autonomous_federation.qwen_suffix_answer_logits_batched", _fake_suffix)
    federation = _peer_fixture()
    with torch.no_grad():
        federation.query.network[-1].bias.zero_()
        for index, name in enumerate(federation.query.candidate_ids):
            if name.startswith("return"):
                federation.query.network[-1].bias[index] = -1.0
    item = _episode(0, event_value=1.0)
    loss, diagnostics = expected_qwen_batch_loss(
        federation, SimpleNamespace(), (item,), (0,), layer_index=0, model_dtype=torch.float32,
        width=16, beam_width=16, ce_weight=1.0, teacher_kl_weight=0.1, hard_alignment_weight=0.1,
        exploration_weight=0.05, preserve_effect_coverage=False, route_objective="task-risk", route_credit_scope="query",
    )
    loss.backward()
    assert diagnostics["deployed_effect_nodes"] >= 2
    child = federation.query.candidates[0].child
    for query in (federation.query, child):
        gradient = query.network[-1].bias.grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    join = next(candidate for candidate in federation.query.candidates if candidate.candidate_id.startswith("join-scale"))
    for parameter in join.parameters():
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0
    assert all(not owner.value.requires_grad and owner.value.grad is None for owner in federation.query.owner_states)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_peer_mount_preserves_adam_and_exposes_chain_and_cooperation(device):
    from benchmarks._federated_peer_composition import mount_peer_federation

    source = _federation()
    source.query.to(device)
    _force_effect_route(source)
    old = source.query
    optimizer = torch.optim.AdamW(old.parameters(), lr=0.002)
    parameter = next(old.network.parameters())
    optimizer.state[parameter] = {"step": torch.tensor(178.0), "exp_avg": torch.ones_like(parameter),
                                  "exp_avg_sq": torch.full_like(parameter, 2.0)}
    saved = optimizer.state[parameter]
    original_counts = tuple(effect.execution_count_tensor().clone() for effect in source.effects)
    federation, receipt = mount_peer_federation(source, hidden_dim=4, optimizer=optimizer)
    assert optimizer.state[parameter] is saved
    assert optimizer.param_groups[-1]["lr"] == 0.002
    assert {id(p) for p in federation.query.parameters() if p.requires_grad} == {
        id(p) for group in optimizer.param_groups for p in group["params"] if p.requires_grad
    }
    names = dict(federation.query.named_parameters())
    for name, param in old.named_parameters():
        assert names[receipt["parameter_name_map"][name]] is param
    assert {"call-inherited-x-to-peer-0", "call-inherited-peer-0-to-peer-1",
            "call-inherited-peer-1-to-peer-0", "join-add-peer-0-peer-1",
            "join-scale-peer-0-peer-1"}.issubset(federation.query.candidate_ids)
    assert all(owner is previous for owner, previous in zip(federation.query.owner_states, old.owner_states, strict=True))
    torch.testing.assert_close(tuple(effect.execution_count_tensor() for effect in federation.effects), original_counts)
    assert federation.query._action_priority.device == parameter.device
    with torch.no_grad():
        execution = federation.query({"x": torch.ones(1, 3, 4, device=device)})
    assert execution.outputs["output"].device == parameter.device
    assert execution.trace.stopped


def test_peer_training_fork_and_fresh_resume_preserve_asset_and_unique_storage(tmp_path, monkeypatch):
    from benchmarks import train_qwen_federated_autonomous_federation as experiment
    from benchmarks._federated_peer_composition import mount_peer_federation
    from benchmarks.train_qwen_federated_online_meta import save_checkpoint, load_checkpoint

    def build(*args, peer_slots=0, **kwargs):
        source = _federation()
        return mount_peer_federation(source, hidden_dim=4, peer_slots=peer_slots, seed=21)[0] if peer_slots else source

    monkeypatch.setattr(experiment, "_build", build)
    monkeypatch.setattr(experiment, "qwen_suffix_answer_logits_batched", _fake_suffix)
    protocol = {"seed": 21, "attachment": {"layer_index": 0},
                "training": {"weight_decay": 0.0, "gradient_clip_norm": 1.0, "ce_weight": 1.0, "teacher_kl_weight": 0.0}}
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol), encoding="utf-8")
    source = build()
    parameter = next(source.query.network.parameters())
    optimizer = torch.optim.AdamW(source.query.parameters(), lr=0.001)
    optimizer.state[parameter] = {"step": torch.tensor(178.0), "exp_avg": torch.zeros_like(parameter),
                                  "exp_avg_sq": torch.ones_like(parameter)}
    save_checkpoint(tmp_path / "source", source.query, optimizer, step=178, protocol_path=path,
                    qwen_digest="fixture", metrics={})
    model = torch.nn.Identity()
    model.config = SimpleNamespace(hidden_size=4)
    kwargs = dict(protocol_path=path, model=model, model_dtype=torch.float32,
                  cached=(_episode(0, event_value=1.0),), evaluation_cached=(), device=torch.device("cpu"),
                  plastic_branches=4, min_operations=1, max_operations=2, search_width=16, beam_width=16,
                  maximum_steps=1, batch_size=1, evaluate_every=1, learning_rate=0.001,
                  hard_alignment_weight=0.1, exploration_weight=0.05, initialize_from=None,
                  route_objective="task-risk", route_credit_scope="query", gradient_clipping="routing-count-task",
                  defer_validation=True)
    fork = experiment.run_training(protocol, output_dir=tmp_path / "fork", expand_peer_slots=2,
                                    resume_from=tmp_path / "source/checkpoints/latest", **kwargs)
    assert fork["completed_steps"] == 179 and fork["peer_slots"] == 2
    checkpoint = tmp_path / "fork/checkpoints/latest"
    assert json.loads((checkpoint / "manifest.json").read_text())["metrics"]["peer_slots"] == 2
    step = fork["training_step_timings"][0]
    assert 1 <= step["deployed_prompt_calls"] <= 2
    assert 0 <= step["retained_prompt_multi_call_fraction"] <= 1
    assert step["maximum_frontier"] >= 1
    restored = build(peer_slots=2)
    restored_optimizer = torch.optim.AdamW(restored.query.parameters(), lr=0.001)
    assert load_checkpoint(checkpoint, restored.query, restored_optimizer, protocol_path=path,
                            device=torch.device("cpu")) == 179
    mapping = fork["peer_expansion"]["parameter_name_map"]
    inherited = dict(restored.query.named_parameters())[mapping["network.0.weight"]]
    assert restored_optimizer.state[inherited]["step"].item() == 179
    stored = load_file(checkpoint / "weights.safetensors")
    nominal = sum(tensor.numel() * tensor.element_size() for tensor in restored.query.state_dict().values())
    actual = sum(tensor.numel() * tensor.element_size() for tensor in stored.values())
    assert actual < nominal / 2
    assert all(owner.revision.item() == 0 for owner in restored.query.owner_states)
    resumed = experiment.run_training(protocol, output_dir=tmp_path / "resumed", peer_slots=2,
                                      resume_from=checkpoint, **kwargs)
    assert resumed["completed_steps"] == 180 and resumed["peer_expansion"] is None
    assert resumed["optimizer_resumed"] and resumed["peer_slots"] == 2


@pytest.mark.parametrize("limit", (None, 4))
def test_generation_cli_records_explicit_prefix_and_preserves_all_arms(tmp_path, monkeypatch, limit):
    import sys
    from benchmarks import _qwen_numerics
    from benchmarks import generate_qwen_federated_autonomous as generation

    path = tmp_path / "input.json"
    path.write_text("{}", encoding="utf-8")
    output = tmp_path / "generated"
    args = ["generate", "--protocol", str(path), "--evaluation-protocol", str(path),
            "--checkpoint", str(path), "--output-dir", str(output), "--batch-size", "4"]
    if limit is not None:
        args += ["--maximum-episodes", str(limit)]
    monkeypatch.setattr(sys, "argv", args)
    items = tuple(_episode(index % 4, event_value=float(index)) for index in range(6))
    model = torch.nn.Identity()
    model.config = SimpleNamespace(hidden_size=4)
    monkeypatch.setattr(generation, "load_protocol", lambda _path: {"attachment": {"layer_index": 0}})
    monkeypatch.setattr(generation, "_load_qwen", lambda *a, **k: (object(), model, torch.float32))
    monkeypatch.setattr(_qwen_numerics, "enable_scaled_qwen_rmsnorm", lambda _model: None)
    monkeypatch.setattr(generation, "load_cache", lambda *a, **k: items)
    monkeypatch.setattr(generation, "_build", lambda *a, **k: _federation())
    monkeypatch.setattr(generation, "_load_query_weights", lambda *a, **k: None)
    selected = []
    def prepare(_federation, cached, **kwargs):
        selected.extend(cached)
        return {arm: (None,) * len(cached) for arm in ("correct", "no_effect", "reset")}, []
    monkeypatch.setattr(generation, "prepare_readouts", prepare)
    monkeypatch.setattr(generation, "generate_batch", lambda _f, _m, _t, cached, _readouts, **kw:
                        ([{"exact_match": False} for _ in cached], {"layer_calls": 1}))
    generation.main()
    result = json.loads((output / "generation.json").read_text())
    count = 6 if limit is None else limit
    assert selected == list(items[:count])
    assert result["episode_count"] == count and result["maximum_episodes"] == limit
    assert result["episode_selection"] == "cached-split-prefix"
    assert set(result["arms"]) == {"base", "correct", "no_effect", "reset"}
    assert all(len(rows) == count for rows in result["arms"].values())
    assert result["duration_seconds"] >= result["generation_host_seconds"] >= 0
    assert result["adaptation_host_seconds"] >= 0


def test_cached_qwen_search_keeps_task_banks_functional() -> None:
    federation = _federation()
    before = tuple(owner.value.clone() for owner in federation.query.owner_states)

    result = search_cached_episode(
        federation,
        _episode(0, event_value=1.0),
        width=4,
        beam_width=4,
        preserve_effect_coverage=True,
    )

    assert result.branches
    assert result.maximum_frontier >= len(result.branches)
    assert all(
        branch.arena.values.get(federation.query.terminal_slot) is not None
        for branch in result.branches
    )
    assert any(
        any(row["kind"] == "effect" for row in branch.route)
        for branch in result.branches
    )
    assert all("update_norm" not in row for branch in result.branches for row in branch.route)
    assert all(
        torch.equal(value, owner.value)
        for value, owner in zip(before, federation.query.owner_states, strict=True)
    )


def test_answers_cannot_change_prompt_routes_or_bank() -> None:
    federation = _federation()
    item = _episode(0, event_value=1.0)
    changed = replace(
        item,
        student_hidden=torch.cat(
            (item.student_hidden[:, :2], torch.full((1, 7, 4), 100.0)), dim=1
        ),
        teacher_logits=torch.full((7, 4), -100.0),
        answer_ids=torch.full((1, 7), 3, dtype=torch.long),
    )
    searches = tuple(
        search_cached_episode(
            federation,
            candidate,
            width=4,
            beam_width=4,
            preserve_effect_coverage=True,
        )
        for candidate in (item, changed)
    )
    assert len(searches[0].branches) == len(searches[1].branches)
    assert searches[0].support.checked_stop_readouts == 0
    assert searches[0].prompt.checked_stop_readouts > 0
    assert searches[0].checked_stop_readouts == searches[1].checked_stop_readouts
    for left, right in zip(searches[0].branches, searches[1].branches, strict=True):
        assert left.route == right.route
        assert torch.equal(left.log_probability, right.log_probability)
        left_state = left.arena.committed_state()
        right_state = right.arena.committed_state()
        assert left_state.revisions == right_state.revisions
        assert all(
            torch.equal(a, b)
            for a, b in zip(left_state.values, right_state.values, strict=True)
        )
        left_output = _branch_readout(federation, item, left)
        right_output = _branch_readout(federation, changed, right)
        assert torch.equal(left_output[:, :2], right_output[:, :2])


def test_answer_readout_skips_effects_and_has_no_future_gradient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    federation = _federation()
    _force_effect_route(federation)
    original = _episode(0, event_value=1.0)
    hidden = original.student_hidden.clone().requires_grad_(True)
    item = replace(original, student_hidden=hidden)
    searched = search_cached_episode(
        federation, item, width=4, beam_width=4, preserve_effect_coverage=True
    )
    branch = next(
        branch
        for branch in searched.branches
        if any(row["kind"] == "effect" and row.get("phase") != "support" for row in branch.route)
    )

    def forbidden_effect(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("answer readout must not execute self-modification")

    for effect in federation.effects:
        monkeypatch.setattr(effect, "forward", forbidden_effect)
    output = _branch_readout(federation, item, branch)
    output[:, item.student_prompt_length - 1, :].sum().backward()
    assert hidden.grad is not None
    assert torch.count_nonzero(hidden.grad[:, item.student_prompt_length :]) == 0


def test_batched_generation_uses_only_prompts_and_frozen_readouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class TinyGenerator(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList((torch.nn.Identity(),))
            self.generation_config = SimpleNamespace(eos_token_id=3)

        def generate(self, *, input_ids: torch.Tensor, **_kwargs: object) -> torch.Tensor:
            sequence = input_ids.clone()
            hidden = torch.nn.functional.one_hot(input_ids % 4, 4).float()
            for _ in range(3):
                adapted = self.model.layers[0](hidden)
                token = adapted[:, -1].argmax(dim=-1, keepdim=True)
                sequence = torch.cat((sequence, token), dim=-1)
                hidden = torch.nn.functional.one_hot(token, 4).float()
            return sequence

    class TinyTokenizer:
        pad_token_id = 0
        eos_token_id = 3

        def decode(self, values: list[int], **_kwargs: object) -> str:
            return " ".join(str(value) for value in values)

    federation = _federation()
    _force_effect_route(federation)
    cached = (_episode(0, event_value=1.0), _episode(2, event_value=-1.0))
    readouts, _receipts = prepare_readouts(federation, cached)
    assert set(readouts) == {"correct", "reset", "no_effect"}
    snapshots = tuple(tuple(value.clone() for value in item.state.values) for item in readouts["correct"])

    def forbidden(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("answer generation cannot route or update Bank")

    monkeypatch.setattr(federation.query, "forward", forbidden)
    for effect in federation.effects:
        monkeypatch.setattr(effect, "forward", forbidden)
    model = TinyGenerator()
    first, receipt = generate_batch(
        federation, model, TinyTokenizer(), cached, readouts["correct"],
        layer_index=0, device=torch.device("cpu"), max_new_tokens=3,
    )
    changed = tuple(
        replace(item, episode=replace(item.episode, answer="different target"),
                answer_ids=torch.full((1, 9), 3), teacher_logits=torch.zeros(9, 4))
        for item in cached
    )
    second, _ = generate_batch(
        federation, model, TinyTokenizer(), changed, readouts["correct"],
        layer_index=0, device=torch.device("cpu"), max_new_tokens=3,
    )
    assert [row["text"] for row in first] == [row["text"] for row in second]
    assert receipt == {"layer_calls": 3, "token_rows": 8}
    assert not model.model.layers[0]._forward_hooks
    assert all(
        torch.equal(before, after)
        for snapshot, item in zip(snapshots, readouts["correct"], strict=True)
        for before, after in zip(snapshot, item.state.values, strict=True)
    )


@pytest.mark.parametrize("route_objective", ("argmin", "task-risk"))
@pytest.mark.parametrize("route_credit_scope", ("all", "query"))
def test_qwen_objective_trains_slow_law_without_mutating_task_banks(
    monkeypatch: pytest.MonkeyPatch,
    route_objective: str,
    route_credit_scope: str,
) -> None:
    from benchmarks import train_qwen_federated_autonomous_federation as experiment

    monkeypatch.setattr(experiment, "qwen_suffix_answer_logits_batched", _fake_suffix)
    federation = _federation()
    cached = (_episode(0, event_value=1.0), _episode(2, event_value=-1.0))
    before = tuple(owner.value.clone() for owner in federation.query.owner_states)

    loss, diagnostics = expected_qwen_batch_loss(
        federation,
        torch.nn.Identity(),
        cached,
        (0, 1),
        layer_index=0,
        model_dtype=torch.float32,
        width=4,
        beam_width=4,
        ce_weight=1.0,
        teacher_kl_weight=0.25,
        hard_alignment_weight=0.1,
        exploration_weight=0.05,
        preserve_effect_coverage=True,
        route_objective=route_objective,
        route_credit_scope=route_credit_scope,
    )
    loss.backward()

    assert bool(torch.isfinite(loss))
    assert diagnostics["effect_path_fraction"] > 0.0
    assert diagnostics["scored_expansions"] >= diagnostics["executed_expansions"] > 0
    assert any(parameter.grad is not None for parameter in federation.query.parameters())
    assert any(
        parameter.grad is not None and bool(parameter.grad.abs().sum() > 0)
        for parameter in federation.query.network.parameters()
    )
    assert all(
        torch.equal(value, owner.value)
        for value, owner in zip(before, federation.query.owner_states, strict=True)
    )


@pytest.mark.parametrize("route_objective", ("argmin", "task-risk"))
@pytest.mark.parametrize("route_credit_scope", ("all", "query"))
def test_qwen_loss_and_optimizer_match_before_and_after_execution_pruning(
    monkeypatch: pytest.MonkeyPatch,
    route_objective: str,
    route_credit_scope: str,
) -> None:
    from benchmarks import train_qwen_federated_autonomous_federation as experiment

    monkeypatch.setattr(experiment, "qwen_suffix_answer_logits_batched", _fake_suffix)
    original_search = experiment.search_to_terminal
    cached = (_episode(0, event_value=1.0), _episode(2, event_value=-1.0))
    results = []
    for lazy in (False, True):
        federation = _federation()
        before = tuple(owner.value.clone() for owner in federation.query.owner_states)
        monkeypatch.setattr(
            experiment, "search_to_terminal", partial(original_search, prune_before_execute=lazy),
        )
        optimizer = torch.optim.AdamW(federation.query.parameters(), lr=0.001)
        loss, diagnostics = expected_qwen_batch_loss(
            federation, torch.nn.Identity(), cached, (0, 1), layer_index=0,
            model_dtype=torch.float32, width=4, beam_width=4, ce_weight=1.0,
            teacher_kl_weight=0.25, hard_alignment_weight=0.1, exploration_weight=0.05,
            preserve_effect_coverage=True, route_objective=route_objective,
            route_credit_scope=route_credit_scope,
        )
        loss.backward()
        gradients = {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in federation.query.named_parameters()
        }
        optimizer.step()
        results.append((loss.detach(), diagnostics, gradients, federation.query.state_dict()))
        assert all(
            torch.equal(value, owner.value)
            for value, owner in zip(before, federation.query.owner_states, strict=True)
        )

    eager, lazy = results
    torch.testing.assert_close(eager[0], lazy[0], rtol=0, atol=0)
    for key, value in eager[1].items():
        if key.endswith("_host_seconds") or key in ("executed_expansions", "checked_stop_readouts"):
            continue
        assert lazy[1][key] == value
    assert lazy[1]["executed_expansions"] < eager[1]["executed_expansions"]
    assert eager[2].keys() == lazy[2].keys()
    for name, gradient in eager[2].items():
        other = lazy[2][name]
        assert (gradient is None) == (other is None), name
        if gradient is not None:
            torch.testing.assert_close(gradient, other, rtol=0, atol=0)
    for name, value in eager[3].items():
        torch.testing.assert_close(value, lazy[3][name], rtol=0, atol=0)


@pytest.mark.parametrize("route_objective", ("argmin", "task-risk"))
@pytest.mark.parametrize("route_credit_scope", ("all", "query"))
def test_microbatch_accumulation_matches_full_batch_gradients_and_adam(
    monkeypatch, route_objective, route_credit_scope,
):
    from benchmarks import train_qwen_federated_autonomous_federation as experiment

    monkeypatch.setattr(experiment, "qwen_suffix_answer_logits_batched", _fake_suffix)
    cached = tuple(_episode(index, event_value=1.0 - index * 0.7) for index in range(4))
    results = []
    for microbatch_size in (4, 1, 3):
        federation = _federation()
        initial = {name: p.detach().clone() for name, p in federation.query.named_parameters()}
        parameters = tuple(federation.query.parameters())
        optimizer = torch.optim.AdamW(parameters, lr=0.001)
        optimizer.zero_grad(set_to_none=True)
        rows, total_loss = [], 0.0
        for offset in range(0, 4, microbatch_size):
            indices = tuple(range(offset, min(offset + microbatch_size, 4)))
            loss, diagnostics = expected_qwen_batch_loss(
                federation, torch.nn.Identity(), cached, indices,
                layer_index=0, model_dtype=torch.float32, width=4, beam_width=4,
                ce_weight=1.0, teacher_kl_weight=0.25, hard_alignment_weight=0.1,
                exploration_weight=0.05, preserve_effect_coverage=True,
                route_objective=route_objective, route_credit_scope=route_credit_scope,
                objective_scale=len(indices) / 4,
            )
            total_loss += float(loss.detach())
            loss.backward()
            del loss
            rows.append((len(indices), diagnostics))
        gradients = {name: None if p.grad is None else p.grad.clone()
                     for name, p in federation.query.named_parameters()}
        experiment._clip_training_gradients(
            parameters, experiment._routing_parameters(federation), norm=1.0, mode="routing-task",
        )
        clipped = {name: None if p.grad is None else p.grad.clone()
                   for name, p in federation.query.named_parameters()}
        optimizer.step()
        # Adam amplifies rounding near zero. Check its exact one-step law for
        # each accumulated gradient, rather than claiming bitwise-identical weights.
        for name, parameter in federation.query.named_parameters():
            gradient = clipped[name]
            expected = initial[name] if gradient is None else (
                initial[name] * (1 - 0.001 * 0.01)
                - 0.001 * gradient / (gradient.abs() + 1e-8)
            )
            torch.testing.assert_close(parameter, expected, rtol=1e-6, atol=1e-7, msg=name)
            if gradient is not None:
                assert float(optimizer.state[parameter]["step"]) == 1.0
        results.append((total_loss, gradients, federation.query.state_dict(),
                        experiment._merge_microbatch_diagnostics(rows)))
    reference = results[0]
    for actual in results[1:]:
        assert actual[0] == pytest.approx(reference[0], abs=1e-6, rel=1e-6)
        for name, expected in reference[1].items():
            assert (expected is None) == (actual[1][name] is None), name
            if expected is not None:
                torch.testing.assert_close(actual[1][name], expected, rtol=2e-5, atol=2e-6, msg=name)
        for key, expected in reference[3].items():
            if key.endswith("_host_seconds"):
                continue
            if isinstance(expected, (int, float)):
                assert actual[3][key] == pytest.approx(expected, abs=1e-6, rel=1e-6), key
            else:
                assert actual[3][key] == expected, key


def test_objective_scale_applies_before_query_credit_autograd(monkeypatch):
    from benchmarks import train_qwen_federated_autonomous_federation as experiment

    monkeypatch.setattr(experiment, "qwen_suffix_answer_logits_batched", _fake_suffix)
    original = experiment._query_credit_objective
    recorded = []

    def record(task, credit, routing):
        recorded.append((task.detach(), torch.autograd.grad(credit, routing, retain_graph=True)))
        return original(task, credit, routing)

    monkeypatch.setattr(experiment, "_query_credit_objective", record)
    for scale in (1.0, 0.25):
        federation = _federation()
        expected_qwen_batch_loss(
            federation, torch.nn.Identity(), (_episode(0, event_value=1.0),), (0,),
            layer_index=0, model_dtype=torch.float32, width=4, beam_width=4,
            ce_weight=1.0, teacher_kl_weight=0.25, hard_alignment_weight=0.1,
            exploration_weight=0.05, preserve_effect_coverage=True,
            route_objective="task-risk", route_credit_scope="query", objective_scale=scale,
        )
    torch.testing.assert_close(recorded[1][0], recorded[0][0] / 4)
    for whole, scaled in zip(recorded[0][1], recorded[1][1], strict=True):
        torch.testing.assert_close(scaled, whole / 4)


def test_event_winner_is_committed_before_later_prompt_and_matches_generation(monkeypatch) -> None:
    federation = _federation()
    cached = (_episode(0, event_value=1.0), _episode(2, event_value=-1.0))

    def forbid_greedy(*_args, **_kwargs):
        raise AssertionError("per-event execution must use K-wide search")

    monkeypatch.setattr(federation.query, "forward", forbid_greedy)
    searches = tuple(search_cached_episode(
        federation, item, width=4, beam_width=4, preserve_effect_coverage=True,
    ) for item in cached)
    changed = replace(cached[0], student_hidden=cached[0].student_hidden * -3.0)
    alternative = search_cached_episode(
        federation, changed, width=4, beam_width=4, preserve_effect_coverage=True,
    )
    assert searches[0].support.route == alternative.support.route
    assert searches[0].support.bank_state.revisions == alternative.support.bank_state.revisions
    for left, right in zip(searches[0].support.bank_state.values,
                           alternative.support.bank_state.values, strict=True):
        torch.testing.assert_close(left, right, rtol=0, atol=0)

    readouts, _ = prepare_readouts(federation, cached, search_width=4, beam_width=4)
    for search, readout, item in zip(searches, readouts["correct"], cached, strict=True):
        assert search.prompt.route == readout.route
        assert search.prompt.bank_state.revisions == readout.state.revisions
        for left, right in zip(search.prompt.bank_state.values, readout.state.values, strict=True):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        primary = search.branches[search.deployed_index]
        deployed_output = _branch_readout(federation, item, primary)
        from benchmarks.train_qwen_federated_autonomous_federation import _replay_data_route
        generated_output = _replay_data_route(
            federation, item.student_hidden.float(), readout.state, readout.route,
        )
        torch.testing.assert_close(deployed_output, generated_output, rtol=0, atol=0)
        assert float(search.support.winner.log_probability.detach()) == max(
            float(branch.log_probability.detach()) for branch in search.support.branches
        )


@pytest.mark.parametrize("route_objective", ("argmin", "task-risk"))
def test_query_credit_matches_explicit_task_and_combined_gradient_oracle(monkeypatch, route_objective):
    from benchmarks import train_qwen_federated_autonomous_federation as experiment

    monkeypatch.setattr(experiment, "qwen_suffix_answer_logits_batched", _fake_suffix)
    cached = (_episode(0, event_value=1.0), _episode(2, event_value=-1.0))
    results = []
    for proxy in (False, True):
        federation = _federation()
        parameters = tuple(p for p in federation.query.parameters() if p.requires_grad)
        routing = experiment._routing_parameters(federation)
        routing_ids = {id(p) for p in routing}
        kwargs = dict(layer_index=0, model_dtype=torch.float32, width=4, beam_width=4,
                      ce_weight=1.0, teacher_kl_weight=0.25, exploration_weight=0.05,
                      preserve_effect_coverage=True, route_objective=route_objective)
        if not proxy:
            task, _ = expected_qwen_batch_loss(
                federation, torch.nn.Identity(), cached, (0, 1), hard_alignment_weight=0.0, **kwargs,
            )
            main = torch.autograd.grad(task, parameters, allow_unused=True)
        total, _ = expected_qwen_batch_loss(
            federation, torch.nn.Identity(), cached, (0, 1), hard_alignment_weight=0.1,
            route_credit_scope="query" if proxy else "all", **kwargs,
        )
        combined = torch.autograd.grad(total, parameters, allow_unused=True)
        for index, (parameter, gradient) in enumerate(zip(parameters, combined, strict=True)):
            parameter.grad = gradient if proxy or id(parameter) in routing_ids else main[index]
        gradients = tuple(None if p.grad is None else p.grad.clone() for p in parameters)
        experiment._clip_training_gradients(parameters, routing, norm=1.0, mode="routing-task")
        torch.optim.AdamW(parameters, lr=0.001, weight_decay=0.01).step()
        results.append((total.detach(), gradients, federation.query.state_dict()))
    oracle, proxy = results
    torch.testing.assert_close(oracle[0], proxy[0])
    for left, right in zip(oracle[1], proxy[1], strict=True):
        if left is None:
            assert right is None
        else:
            assert right is not None
            torch.testing.assert_close(left, right)
    for key in oracle[2]:
        torch.testing.assert_close(oracle[2][key], proxy[2][key])


def test_nonfinite_gradient_cannot_reach_optimizer_step():
    from benchmarks import train_qwen_federated_autonomous_federation as experiment

    parameter = torch.nn.Parameter(torch.ones(3))
    parameter.grad = torch.full_like(parameter, float("nan"))
    optimizer = torch.optim.AdamW((parameter,), lr=0.001)
    with pytest.raises(RuntimeError, match="non-finite"):
        experiment._clip_training_gradients((parameter,), (), norm=1.0, mode="global")
        optimizer.step()
    torch.testing.assert_close(parameter, torch.ones(3), rtol=0, atol=0)
    assert not optimizer.state


@pytest.mark.parametrize("magnitude", (1.0, 1e20, 1e37))
def test_route_risk_credit_matches_fp64_without_intermediate_overflow(magnitude):
    from benchmarks import train_qwen_federated_autonomous_federation as experiment

    scores = torch.tensor([-magnitude, *([0.0] * 15)], requires_grad=True)
    losses = torch.tensor([magnitude, *([0.0] * 15)], requires_grad=True)
    credit, risk, temperature = experiment._retained_task_risk_credit(scores, losses)
    gradient, target_gradient = torch.autograd.grad(credit, (scores, losses), allow_unused=True)

    reference = scores.detach().double().requires_grad_()
    scale = reference.detach().std(correction=0).clamp_min(1.0)
    probabilities = ((reference - reference.detach().max()) / scale).softmax(0)
    expected_risk = (probabilities * losses.detach().double()).sum()
    expected_gradient = torch.autograd.grad(scale * expected_risk, reference)[0]
    assert float(credit.detach()) == 0.0
    assert credit.dtype == scores.dtype and risk.dtype == losses.dtype
    assert temperature.dtype == scores.dtype
    assert target_gradient is None
    assert torch.isfinite(gradient).all() and torch.isfinite(risk) and torch.isfinite(temperature)
    torch.testing.assert_close(gradient, expected_gradient.float())
    torch.testing.assert_close(risk, expected_risk.float())
    torch.testing.assert_close(temperature, scale.float())


def test_query_row_rejections_are_not_counted_as_executed_expansions(monkeypatch):
    from benchmarks import train_qwen_federated_autonomous_federation as experiment

    original_search = experiment.search_cached_episode

    def search(*args, **kwargs):
        result = original_search(*args, **kwargs)
        return replace(result, numerical_rejections=(
            {"kind": "query", "code": "FF2_NONFINITE", "step": 0, "prefix": []},
            {"kind": "effect", "code": "FF2_NONFINITE", "step": 1, "prefix": []},
        ))

    monkeypatch.setattr(experiment, "search_cached_episode", search)
    monkeypatch.setattr(experiment, "qwen_suffix_answer_logits_batched", _fake_suffix)
    _, diagnostics = experiment.expected_qwen_batch_loss(
        _federation(), torch.nn.Identity(), (_episode(0, event_value=1.0),), (0,),
        layer_index=0, model_dtype=torch.float32, width=4, beam_width=4, ce_weight=1.0,
        teacher_kl_weight=0.25, hard_alignment_weight=0.1, exploration_weight=0.05,
        preserve_effect_coverage=True,
    )
    assert diagnostics["nonfinite_query_rows"] == 1
    assert diagnostics["nonfinite_expansions"] == 1


def test_query_credit_reports_bad_gradient_before_zero_times_infinity():
    from benchmarks import train_qwen_federated_autonomous_federation as experiment

    parameter = torch.nn.Parameter(torch.zeros(()))
    credit = parameter.sqrt()
    task = parameter * 0 + 3.0
    assert torch.isfinite(task) and torch.isfinite(credit)
    with pytest.raises(FloatingPointError, match="non-finite Query-credit gradients"):
        experiment._query_credit_objective(task, credit, (parameter,))
    assert parameter.grad is None


def test_readout_rejection_preserves_preselected_winner_and_original_scores(monkeypatch):
    from benchmarks import train_qwen_federated_autonomous_federation as experiment

    branches = tuple(SimpleNamespace(
        route=({"candidate_id": f"route-{index}"},),
        log_probability=torch.tensor(float(-index), requires_grad=True),
    ) for index in range(3))
    search = experiment._EpisodeSearch(branches, 3, 1, None, None)
    calls = []

    def readout(_federation, _cached, branch):
        index = next(index for index, item in enumerate(branches) if item is branch)
        calls.append(index)
        if index == 0:
            raise mechanisms.FormulaBindingError("FF2_NONFINITE", "counterfactual overflow")
        return torch.full((1, 2, 4), float(index), requires_grad=True)

    monkeypatch.setattr(experiment, "_branch_readout", readout)
    retained, values, rejected = experiment._training_readouts(
        None, (None,), (0,), (search,), model_dtype=torch.float32,
    )
    assert calls == [1, 0, 2]
    assert search.deployed_index == 1 and len(search.branches) == 3
    assert retained[0].deployed_index == 0
    assert retained[0].branches[0] is branches[1]
    assert retained[0].branches[1] is branches[2]
    assert len(rejected) == 1 and rejected[0]["route"] == ["route-0"]
    assert len(values) == 2 and values[0].requires_grad

    calls.clear()
    deployed_invalid = replace(search, deployed_index=0)
    with pytest.raises(RuntimeError, match="no teacher-assisted replacement"):
        experiment._training_readouts(
            None, (None,), (0,), (deployed_invalid,), model_dtype=torch.float32,
        )
    assert calls == [0]


def test_readout_rejection_keeps_other_errors_and_checks_cast(monkeypatch):
    from benchmarks import train_qwen_federated_autonomous_federation as experiment

    branch = SimpleNamespace(route=({"candidate_id": "only"},))
    search = experiment._EpisodeSearch((branch,), 1, 0, None, None)
    monkeypatch.setattr(experiment, "_branch_readout", lambda *args: torch.full((1, 2, 4), 1e20))
    with pytest.raises(RuntimeError, match="deployed Qwen readout is nonfinite"):
        experiment._training_readouts(None, (None,), (0,), (search,), model_dtype=torch.float16)

    def wrong_type(*args):
        raise mechanisms.FormulaBindingError("FF2_RUNTIME_DTYPE_MISMATCH", "wrong dtype")

    monkeypatch.setattr(experiment, "_branch_readout", wrong_type)
    with pytest.raises(mechanisms.FormulaBindingError) as error:
        experiment._training_readouts(None, (None,), (0,), (search,), model_dtype=torch.float32)
    assert error.value.code == "FF2_RUNTIME_DTYPE_MISMATCH"


def test_prompt_stop_check_uses_final_bank_and_prompt_route_without_grad(monkeypatch):
    from benchmarks import train_qwen_federated_autonomous_federation as experiment

    prompt = torch.ones(1, 3, 4, requires_grad=True)
    bank = object()
    branch = SimpleNamespace(
        arena=SimpleNamespace(committed_state=lambda: bank),
        route=({"candidate_id": "prior", "phase": "support"},
               {"candidate_id": "stop", "phase": "support"},
               {"candidate_id": "current"}, {"candidate_id": "stop"}),
    )
    calls = []

    def replay(_federation, value, final_bank, route):
        assert value is prompt and final_bank is bank
        assert route == ("current", "stop")
        assert not torch.is_grad_enabled()
        calls.append(route)
        return value * 1e10

    monkeypatch.setattr(experiment, "_replay_data_route", replay)
    assert experiment._prompt_stop_admission(None, prompt, torch.bfloat16)(branch)
    assert not experiment._prompt_stop_admission(None, prompt, torch.float16)(branch)
    assert len(calls) == 2
    assert prompt.grad is None and prompt.requires_grad

    def reject(*_args):
        raise mechanisms.FormulaBindingError("FF2_NONFINITE", "final-bank readout overflow")

    monkeypatch.setattr(experiment, "_replay_data_route", reject)
    assert not experiment._prompt_stop_admission(None, prompt, torch.float32)(branch)

    def wrong_contract(*_args):
        raise mechanisms.FormulaBindingError("FF2_RUNTIME_DTYPE_MISMATCH", "wrong dtype")

    monkeypatch.setattr(experiment, "_replay_data_route", wrong_contract)
    with pytest.raises(mechanisms.FormulaBindingError, match="wrong dtype"):
        experiment._prompt_stop_admission(None, prompt, torch.float32)(branch)


def test_main_task_loss_is_actual_causal_hard_winner_not_offline_best(monkeypatch) -> None:
    from benchmarks import train_qwen_federated_autonomous_federation as experiment

    monkeypatch.setattr(experiment, "qwen_suffix_answer_logits_batched", _fake_suffix)
    federation = _federation()
    item = _episode(0, event_value=1.0)
    search = search_cached_episode(
        federation, item, width=4, beam_width=4, preserve_effect_coverage=True,
    )
    value = _branch_readout(federation, item, search.branches[search.deployed_index])
    logits = _fake_suffix(
        None, (value,), prompt_lengths=(item.student_prompt_length,),
        answer_lengths=(item.answer_ids.numel(),), layer_index=0,
    )[0]
    expected = torch.nn.functional.cross_entropy(logits, item.answer_ids.reshape(-1))
    loss, diagnostics = expected_qwen_batch_loss(
        federation, torch.nn.Identity(), (item,), (0,), layer_index=0,
        model_dtype=torch.float32, width=4, beam_width=4,
        ce_weight=1.0, teacher_kl_weight=0.0, hard_alignment_weight=0.0,
        exploration_weight=0.0, preserve_effect_coverage=True,
    )
    torch.testing.assert_close(loss, expected)
    assert diagnostics["deployed_task_loss"] == pytest.approx(float(expected.detach()))
    assert diagnostics["best_retained_task_loss"] <= diagnostics["deployed_task_loss"]


def test_frozen_qwen_evaluation_physically_executes_matched_no_effect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from benchmarks import train_qwen_federated_autonomous_federation as experiment

    monkeypatch.setattr(experiment, "qwen_suffix_answer_logits_batched", _fake_suffix)
    federation = _federation()
    _force_effect_route(federation)
    federation.query.requires_grad_(False)

    result = evaluate_frozen_qwen_adaptation(
        federation,
        torch.nn.Identity(),
        (_episode(0, event_value=1.0), _episode(2, event_value=-1.0)),
        layer_index=0,
        model_dtype=torch.float32,
    )

    assert result["replay_exact_fraction"] == 1.0
    assert result["matched_no_effect_selected_path_fraction"] == 1.0
    assert result["replay_close_fraction"] == 1.0
    assert result["event_routing"] == "per-event-k-wide-prompt-readout-admission-v2"
    assert result["slow_state_digest_unchanged"] is True
    assert "swap" not in result["scores"]
    assert "correct_vs_swap_ce_margin" not in result
    receipts = result["sample_receipts"]
    assert any(row["support"]["effects"] for row in receipts)  # type: ignore[index]


def test_model_shape_and_search_budget_validation() -> None:
    assert _model_hidden_size(SimpleNamespace(config=SimpleNamespace(hidden_size=4))) == 4
    with pytest.raises(ValueError, match="hidden_size"):
        _model_hidden_size(SimpleNamespace(config=SimpleNamespace(hidden_size=0)))
    with pytest.raises(ValueError, match="positive"):
        _validate_search_budget(
            search_width=0,
            beam_width=4,
            maximum_steps=1,
            batch_size=1,
            evaluate_every=1,
            hard_alignment_weight=0.1,
            exploration_weight=0.05,
        )


def test_checkpoint_selection_requires_causal_history_margins() -> None:
    validation = {
        "correct_vs_base_ce_margin": 0.4,
        "correct_vs_no_effect_ce_margin": 0.4,
        "correct_vs_reset_ce_margin": 0.4,
    }
    assert _causal_checkpoint_eligible(validation)
    assert _causal_checkpoint_eligible(
        {**validation, "correct_vs_swap_ce_margin": -0.01}
    )
    assert not _causal_checkpoint_eligible({**validation, "correct_vs_reset_ce_margin": -0.01})

    assert _should_replace_checkpoint(
        candidate_ce=9.0,
        candidate_causal=False,
        incumbent_ce=None,
        incumbent_causal=False,
    )
    assert not _should_replace_checkpoint(
        candidate_ce=8.0,
        candidate_causal=False,
        incumbent_ce=9.0,
        incumbent_causal=True,
    )
    assert _should_replace_checkpoint(
        candidate_ce=10.0,
        candidate_causal=True,
        incumbent_ce=9.0,
        incumbent_causal=False,
    )
    assert _should_replace_checkpoint(
        candidate_ce=8.5,
        candidate_causal=True,
        incumbent_ce=9.0,
        incumbent_causal=True,
    )


def test_full_training_resume_keeps_adam_moments_and_step(tmp_path, monkeypatch) -> None:
    from benchmarks import train_qwen_federated_autonomous_federation as experiment

    monkeypatch.setattr(experiment, "qwen_suffix_answer_logits_batched", _fake_suffix)
    monkeypatch.setattr(experiment, "_build", lambda *_args, **_kwargs: _federation())
    protocol = {
        "seed": 21,
        "attachment": {"layer_index": 0},
        "training": {
            "weight_decay": 0.0,
            "gradient_clip_norm": 1.0,
            "ce_weight": 1.0,
            "teacher_kl_weight": 0.0,
        },
    }
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    model = torch.nn.Identity()
    model.config = SimpleNamespace(hidden_size=4)
    cached = (_episode(0, event_value=1.0), _episode(2, event_value=-1.0))
    kwargs = {
        "protocol_path": protocol_path,
        "model": model,
        "model_dtype": torch.float32,
        "cached": cached,
        "evaluation_cached": cached,
        "device": torch.device("cpu"),
        "plastic_branches": 4,
        "min_operations": 1,
        "max_operations": 2,
        "search_width": 4,
        "beam_width": 4,
        "maximum_steps": 1,
        "batch_size": 2,
        "evaluate_every": 1,
        "learning_rate": 0.001,
        "hard_alignment_weight": 0.1,
        "exploration_weight": 0.05,
        "initialize_from": None,
    }
    first = experiment.run_training(protocol, output_dir=tmp_path / "first", **kwargs)
    resumed = experiment.run_training(
        protocol,
        output_dir=tmp_path / "resumed",
        resume_from=tmp_path / "first" / "checkpoints" / "latest",
        paired_batches=True,
        route_credit_scope="query",
        gradient_clipping="routing-task",
        **kwargs,
    )
    assert first["optimizer_resumed"] is False
    assert resumed["optimizer_resumed"] is True
    assert resumed["step_offset"] == 1
    assert resumed["completed_steps"] == 2
    assert resumed["resolved_training"]["paired_batches"] is True
    assert resumed["resolved_training"]["route_credit_scope"] == "query"
    assert resumed["resolved_training"]["gradient_clipping"] == "routing-task"
    assert set(resumed["training_step_timings"][0]["gradient_group_norms_before_clip"]) == {"routing", "task"}
    assert resumed["resolved_training"]["candidate_execution"] == "prune-before-execute-v1"
    assert resumed["training_step_timings"][0]["scored_expansions"] > 0
    assert resumed["training_step_timings"][0]["executed_expansions"] > 0
    assert resumed["training_step_timings"][0]["training_indices"] == [0, 1]
    assert first["training_step_timings"][0]["training_step"] == 1
    assert resumed["training_step_timings"][0]["training_step"] == 2
    assert resumed["training_step_timings"][0]["search_host_seconds"] >= 0
    assert resumed["timing_kind"] == "host-wall-no-extra-device-barriers"
    stored = load_file(tmp_path / "resumed" / "checkpoints" / "latest" / "optimizer.safetensors")
    assert max(float(value) for key, value in stored.items() if key.endswith(".step")) == 2.0
    assert any(bool(value.abs().sum() > 0) for key, value in stored.items() if key.endswith(".exp_avg"))

    def failed_validation(*_args, **_kwargs):
        raise RuntimeError("validation interrupted")

    monkeypatch.setattr(experiment, "evaluate_frozen_qwen_adaptation", failed_validation)
    with pytest.raises(RuntimeError, match="validation interrupted"):
        experiment.run_training(protocol, output_dir=tmp_path / "interrupted", **kwargs)
    saved = tmp_path / "interrupted" / "checkpoints" / "latest"
    manifest = json.loads((saved / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["step"] == 1
    assert manifest["metrics"]["validation_pending"] is True
    assert manifest["metrics"]["candidate_execution"] == "prune-before-execute-v1"
    optimizer_values = load_file(saved / "optimizer.safetensors")
    assert max(float(value) for key, value in optimizer_values.items() if key.endswith(".step")) == 1.0

    # A short continuation can save between validations without reevaluating its source.
    with pytest.raises(RuntimeError, match="validation interrupted"):
        experiment.run_training(
            protocol, output_dir=tmp_path / "sparse-validation",
            resume_from=tmp_path / "first" / "checkpoints" / "latest",
            **{**kwargs, "maximum_steps": 2, "evaluate_every": 2,
               "checkpoint_every": 1, "skip_initial_validation": True,
               "route_objective": "task-risk"},
        )
    saved = tmp_path / "sparse-validation" / "checkpoints" / "latest"
    manifest = json.loads((saved / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["step"] == 3
    assert manifest["metrics"]["route_objective"] == "task-risk"
    progress = json.loads((tmp_path / "sparse-validation" / "progress.json").read_text())
    assert progress["completed_steps"] == 2
    assert progress["validation_pending"] is True

    # Explicit checkpoint-only segments never turn a skipped validation into a pass.
    original_objective = experiment.expected_qwen_batch_loss
    losses = []

    def tracked_objective(*args, **kwargs):
        assert all(ref() is None for ref in losses)
        result = original_objective(*args, **kwargs)
        losses.append(weakref.ref(result[0]))
        return result

    monkeypatch.setattr(experiment, "expected_qwen_batch_loss", tracked_objective)
    deferred_dir = tmp_path / "deferred-validation"
    deferred = experiment.run_training(
        protocol, output_dir=deferred_dir,
        resume_from=tmp_path / "first" / "checkpoints" / "latest",
        **{**kwargs, "evaluation_cached": (), "defer_validation": True,
           "checkpoint_every": 10, "evaluate_every": 10, "microbatch_size": 1},
    )
    assert deferred["completed_steps"] == 2
    assert deferred["validation_pending"] is True
    assert deferred["validation"] is None
    assert deferred["best_checkpoint_step"] is None
    assert not (deferred_dir / "deployment.safetensors").exists()
    assert not (deferred_dir / "best.safetensors").exists()
    saved = deferred_dir / "checkpoints" / "latest"
    manifest = json.loads((saved / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["step"] == 2
    assert manifest["metrics"]["validation_pending"] is True
    assert (saved / "optimizer.safetensors").is_file()
    assert manifest["metrics"]["event_routing"] == experiment.EVENT_ROUTING
    assert manifest["metrics"]["microbatch_size"] == 1
    progress = json.loads((deferred_dir / "progress.json").read_text())
    row = progress["training_step_timings"][0]
    assert row["checked_stop_readouts"] > 0
    assert row["nonfinite_stop_readouts"] == 0
    assert row["nonfinite_query_rows"] == 0
    assert row == deferred["training_step_timings"][0]
    assert row["microbatch_count"] == 2
    assert deferred["resolved_training"]["batch_size"] == 2
    assert deferred["resolved_training"]["microbatch_size"] == 1
    assert len(losses) == 2 and all(ref() is None for ref in losses)

    def failed_objective(*_args, **_kwargs):
        return torch.tensor(float("nan"), requires_grad=True), {"deployed_task_loss": float("nan")}

    monkeypatch.setattr(experiment, "expected_qwen_batch_loss", failed_objective)
    failed_dir = tmp_path / "failed-objective"
    with pytest.raises(RuntimeError, match="task loss became non-finite"):
        experiment.run_training(
            protocol, output_dir=failed_dir, defer_validation=True,
            resume_from=tmp_path / "first" / "checkpoints" / "latest", **kwargs,
        )
    failure = json.loads((failed_dir / "failure.json").read_text())
    assert failure["attempted_step"] == 2
    assert failure["completed_steps"] == 1
    assert failure["optimizer_step_applied"] is False
    assert failure["diagnostics"]["deployed_task_loss"] == "nan"
    assert failure["completed_diagnostics"] == {}
    assert failure["failure_stage"] == "loss_validation"
    assert not (failed_dir / "checkpoints").exists()

    calls = []
    updates = []

    def partially_failed_objective(*args, **kwargs):
        calls.append(tuple(args[3]))
        if len(calls) == 2:
            return failed_objective()
        return original_objective(*args, **kwargs)

    def forbidden_update(*_args, **_kwargs):
        updates.append(True)
        raise AssertionError("an incomplete effective batch must not update Adam")

    monkeypatch.setattr(experiment, "expected_qwen_batch_loss", partially_failed_objective)
    monkeypatch.setattr(torch.optim.AdamW, "step", forbidden_update)
    partial_dir = tmp_path / "partial-failure"
    with pytest.raises(RuntimeError, match="task loss became non-finite"):
        experiment.run_training(
            protocol, output_dir=partial_dir, defer_validation=True, microbatch_size=1,
            resume_from=tmp_path / "first" / "checkpoints" / "latest", **kwargs,
        )
    failure = json.loads((partial_dir / "failure.json").read_text())
    assert len(calls) == 2 and updates == []
    assert failure["completed_microbatches"] == 1
    assert failure["failed_microbatch_indices"] == list(calls[-1])
    assert failure["failure_stage"] == "loss_validation"
    assert failure["diagnostics"]["deployed_task_loss"] == "nan"
    assert math.isfinite(failure["completed_diagnostics"]["deployed_task_loss"])
    assert failure["optimizer_step_applied"] is False
    assert failure["completed_steps"] == 1
    assert not (partial_dir / "checkpoints").exists()

    calls.clear()

    def raised_objective(*args, **kwargs):
        calls.append(tuple(args[3]))
        if len(calls) == 2:
            raise RuntimeError("second microbatch objective failed")
        return original_objective(*args, **kwargs)

    monkeypatch.setattr(experiment, "expected_qwen_batch_loss", raised_objective)
    raised_dir = tmp_path / "raised-objective"
    with pytest.raises(RuntimeError, match="second microbatch objective failed"):
        experiment.run_training(
            protocol, output_dir=raised_dir, defer_validation=True, microbatch_size=1,
            resume_from=tmp_path / "first" / "checkpoints" / "latest", **kwargs,
        )
    failure = json.loads((raised_dir / "failure.json").read_text())
    assert len(calls) == 2 and updates == []
    assert failure["completed_microbatches"] == 1
    assert failure["failed_microbatch_indices"] == list(calls[-1])
    assert failure["failure_stage"] == "objective"
    assert failure["diagnostics"] == {}
    assert math.isfinite(failure["completed_diagnostics"]["deployed_task_loss"])
    assert failure["optimizer_step_applied"] is False
    assert failure["completed_steps"] == 1
    assert not (raised_dir / "checkpoints").exists()


def test_complete_event_extension_reuses_answer_cache_and_batches_capture(tmp_path, monkeypatch) -> None:
    from benchmarks import train_qwen_federated_online_meta as cache

    items = tuple(_episode(index, event_value=float(index)) for index in range(3))
    episodes = tuple(replace(item.episode, event_1="e" * (3 + index // 2)) for index, item in enumerate(items))
    monkeypatch.setattr(cache, "protocol_episodes", lambda _protocol: episodes)
    monkeypatch.setattr(cache, "_module_parameter_digest", lambda _model: "frozen-model")
    monkeypatch.setattr(
        cache, "_chat_ids", lambda _tokenizer, messages: torch.arange(len(messages[-1]["content"])).unsqueeze(0)
    )
    captures = []

    def capture(_model, ids, *, layer_index, device):
        captures.append(tuple(ids.shape))
        assert layer_index == 6 and device == torch.device("cpu")
        return ids.float().unsqueeze(-1).expand(-1, -1, 4).clone(), torch.empty(0)

    monkeypatch.setattr(cache, "capture_layer_output", capture)
    protocol_path = tmp_path / "protocol.json"
    protocol_path.write_text("{}", encoding="utf-8")
    source, output = tmp_path / "old", tmp_path / "full"
    tensors = {}
    rows = []
    for index, item in enumerate(items):
        for field in ("event_hidden", "student_hidden", "teacher_logits", "base_logits", "student_prompt_ids", "answer_ids"):
            value = getattr(item, field)
            tensors[f"episode.{index:03d}.{field}"] = value[:, -1:] if field == "event_hidden" else value
        rows.append({"index": index, "student_prompt_length": item.student_prompt_length})
    tensor_path, manifest_path = cache._cache_paths(source)
    cache._atomic_safetensors(tensor_path, tensors)
    cache._atomic_json(manifest_path, {
        "format": cache.CACHE_FORMAT,
        "protocol_sha256": cache._file_digest(protocol_path),
        "model": {"parameter_digest": "frozen-model"},
        "layer_index": 6,
        "episodes": rows,
    })
    source_hash = cache._file_digest(tensor_path)
    with pytest.raises(ValueError, match="complete-event"):
        cache.load_cache({}, protocol_path=protocol_path, output_dir=source, device=torch.device("cpu"), require_full_event=True)
    manifest = cache.extend_event_cache(
        {"system_prompt": "system"},
        protocol_path=protocol_path,
        tokenizer=object(),
        model=torch.nn.Identity(),
        device=torch.device("cpu"),
        source_dir=source,
        output_dir=output,
        batch_size=8,
    )
    restored = cache.load_cache({}, protocol_path=protocol_path, output_dir=output, device=torch.device("cpu"), require_full_event=True)
    assert captures == [(2, 3), (1, 4)]
    assert [item.event_hidden.shape[1] for item in restored] == [3, 3, 4]
    assert manifest["answer_cache_reused"] is True
    assert cache._file_digest(tensor_path) == source_hash
    extended = load_file(cache._cache_paths(output)[0])
    for name, value in tensors.items():
        if not name.endswith(".event_hidden"):
            assert torch.equal(extended[name], value)
