from __future__ import annotations

import copy
from pathlib import Path

import pytest
import torch

from arti import mechanisms
from benchmarks.probe_qwen_federated_online_meta import (
    build_program_query,
    independent_bank_roots,
)
from benchmarks.evaluate_qwen_federated_forward_only_memory import (
    build_forward_only_query,
    select_configuration,
)
from benchmarks.evaluate_qwen_federated_forward_writer import select_rate
from benchmarks.evaluate_qwen_federated_forward_sequence_memory import (
    execute_sequence_writes,
    select_configuration as select_sequence_configuration,
    sequence_pairs,
)
from benchmarks.evaluate_qwen_federated_forward_multisite_memory import (
    counterpart_indices,
    select_configuration as select_multisite_configuration,
)
from benchmarks.search_qwen_federated_forward_writer import (
    candidate_spec,
    select_finalist,
)
from benchmarks.train_qwen_federated_forward_writer import (
    TRAINABLE_WRITER_OPERANDS,
    build_forward_writer,
)
from benchmarks.evaluate_qwen_federated_forward_transition_memory import (
    adjacent_transition_pairs,
    transition_pairs,
)
from benchmarks.train_qwen_federated_online_meta import (
    DEFAULT_PROTOCOL,
    CachedEpisode,
    _batch_indices,
    build_optimizer,
    causal_control_hidden_rows,
    conditional_path_loss,
    load_checkpoint,
    load_protocol,
    protocol_episodes,
    save_checkpoint,
)
from benchmarks.train_qwen_federated_effect_federation import (
    EFFECT_FAMILIES,
    build_effect_federation,
    enumerate_effect_paths,
)
from benchmarks.train_federated_self_modifying_federation import (
    BankTrajectory,
    _hard_supports,
    evaluate_frozen_forward_adaptation,
    expand_support_trajectories,
    expected_episode_loss,
    make_association_episodes,
)
from benchmarks.probe_qwen_federated_online_meta import Episode


def test_overfit_protocol_expands_to_counterfactual_episode_pairs() -> None:
    protocol = load_protocol(DEFAULT_PROTOCOL)
    episodes = protocol_episodes(protocol)
    grouped: dict[str, list[object]] = {}
    for episode in episodes:
        grouped.setdefault(episode.pair_id, []).append(episode)

    assert len(episodes) == 32
    assert len(grouped) == 16
    assert all(len(rows) == 2 for rows in grouped.values())
    for rows in grouped.values():
        left, right = rows
        assert left.event_2 == right.event_2
        assert left.event_1 != right.event_1
        assert left.answer != right.answer


def test_generalization_protocols_have_disjoint_keys_values_and_wording() -> None:
    root = Path(__file__).resolve().parents[1] / "benchmarks"
    protocols = tuple(
        load_protocol(root / f"qwen_federated_generalization_{split}.json")
        for split in ("train", "validation", "test")
    )
    key_sets: list[set[str]] = []
    value_sets: list[set[str]] = []
    templates: list[tuple[str, str]] = []
    for protocol in protocols:
        bindings = protocol["bindings"]
        template = protocol["binding_template"]
        assert isinstance(bindings, list) and isinstance(template, dict)
        key_sets.append({str(binding["key"]) for binding in bindings})
        value_sets.append(
            {
                str(value)
                for binding in bindings
                for value in binding["values"]
            }
        )
        templates.append((str(template["event_1"]), str(template["event_2"])))

    for left in range(len(protocols)):
        for right in range(left + 1, len(protocols)):
            assert key_sets[left].isdisjoint(key_sets[right])
            assert value_sets[left].isdisjoint(value_sets[right])
    assert len(set(templates)) == len(protocols)


def test_composition_diagnostic_holds_out_pairs_but_reuses_training_vocabulary() -> None:
    root = Path(__file__).resolve().parents[1] / "benchmarks"
    training = load_protocol(root / "qwen_federated_generalization_train.json")
    diagnostic = load_protocol(
        root / "qwen_federated_generalization_composition.json"
    )
    train_bindings = training["bindings"]
    diagnostic_bindings = diagnostic["bindings"]
    assert isinstance(train_bindings, list) and isinstance(diagnostic_bindings, list)
    train_values = {
        str(value) for binding in train_bindings for value in binding["values"]
    }
    train_words = {word for value in train_values for word in value.split()}
    diagnostic_values = {
        str(value) for binding in diagnostic_bindings for value in binding["values"]
    }
    diagnostic_words = {word for value in diagnostic_values for word in value.split()}

    assert train_values.isdisjoint(diagnostic_values)
    assert diagnostic_words <= train_words
    assert training["binding_template"] == diagnostic["binding_template"]


def test_search_query_exposes_four_plastic_predecessors_and_one_dead_end() -> None:
    query, _producer, effect = build_program_query(
        hidden_dim=8,
        rank=3,
        seed=19,
        device=torch.device("cpu"),
        plastic_branches=4,
        include_dead_end=True,
        event_write_layout="key-value-pair",
    )
    root = independent_bank_roots(query, 1)[0]
    arena = query._arena({"x": torch.randn(1, 2, 8)}, bank_state=root)
    eligible = query.eligible(arena, steps=0)

    assert len(query.plastic_candidates) == 4
    assert query.candidate_ids == (
        "plastic-lora-0",
        "plastic-lora-1",
        "plastic-lora-2",
        "plastic-lora-3",
        "dead-end",
        "outer-write",
    )
    assert eligible.tolist() == [True, True, True, True, True, False, False]
    for producer in query.plastic_candidates:
        assert effect.accepts(producer(arena))
    dead_end = next(
        candidate for candidate in query.candidates if candidate.candidate_id == "dead-end"
    )
    assert isinstance(dead_end, mechanisms.FormulaProgramTensorCandidateV2)
    assert not effect.accepts(dead_end(arena))


def test_effect_federation_enumerates_every_predecessor_bank_successor() -> None:
    query, producers, effects = build_effect_federation(
        hidden_dim=8,
        rank=3,
        seed=23,
        device=torch.device("cpu"),
        plastic_branches=4,
    )
    episode = Episode("pair", "event", "latest", "answer")
    cached = (
        CachedEpisode(
            episode,
            torch.randn(1, 2, 8),
            torch.randn(1, 4, 8),
            torch.randn(1, 2, 11),
            torch.randn(1, 2, 11),
            torch.ones(1, 2, dtype=torch.int64),
            torch.ones(1, 2, dtype=torch.int64),
            2,
        ),
    )

    paths = enumerate_effect_paths(
        query,
        producers,
        effects,
        cached,
        (0,),
        model_dtype=torch.float32,
        search_width=4,
    )

    assert len(paths) == len(producers) * len(EFFECT_FAMILIES)
    assert {
        (path.producer_id, path.effect_id.removesuffix("-write")) for path in paths
    } == {
        (producer.candidate_id, family)
        for producer in producers
        for family in EFFECT_FAMILIES
    }
    assert {path.bank_state.slot_refs for path in paths} == {
        query.initial_bank_state().slot_refs
    }
    assert {
        next(
            ref
            for ref, revision in zip(
                path.bank_state.slot_refs,
                path.bank_state.revisions,
                strict=True,
            )
            if revision == 1
        ).producer_id
        for path in paths
    } == {producer.candidate_id for producer in producers}
    assert all(sum(path.bank_state.revisions) == 1 for path in paths)
    assert tuple(path.effect_id.removesuffix("-write") for path in paths[:6]) == (
        EFFECT_FAMILIES
    )
    torch.testing.assert_close(
        torch.stack(tuple(path.route_probability for path in paths)).sum(),
        torch.tensor(1.0),
    )
    assert all(path.adapted_hidden.shape == cached[0].student_hidden.shape for path in paths)

    narrowed = enumerate_effect_paths(
        query,
        producers,
        effects,
        cached,
        (0,),
        model_dtype=torch.float32,
        search_width=2,
    )
    assert len(narrowed) == 2 * len(EFFECT_FAMILIES)
    torch.testing.assert_close(
        torch.stack(tuple(path.route_probability for path in narrowed)).sum(),
        torch.tensor(1.0),
    )


def test_effect_federation_task_banks_are_not_optimizer_parameters() -> None:
    query, producers, effects = build_effect_federation(
        hidden_dim=8,
        rank=3,
        seed=29,
        device=torch.device("cpu"),
        plastic_branches=8,
    )
    assert len(producers) == 8
    assert all(
        not isinstance(
            producer.candidate.operand_store.tensor(producer.plastic_bank_slot),
            torch.nn.Parameter,
        )
        for producer in producers
    )
    assert all(
        producer.plastic_bank_slot
        not in producer.candidate.operand_store.trainable_names
        for producer in producers
    )
    assert any(parameter.requires_grad for effect in effects for parameter in effect.parameters())


def test_episodic_association_splits_are_disjoint_and_answerable() -> None:
    train = make_association_episodes(
        split="train",
        seed=101,
        count=3,
        hidden_dim=8,
        support_count=4,
        device=torch.device("cpu"),
    )
    validation = make_association_episodes(
        split="validation",
        seed=202,
        count=3,
        hidden_dim=8,
        support_count=4,
        device=torch.device("cpu"),
    )

    assert {item.episode_id for item in train}.isdisjoint(
        item.episode_id for item in validation
    )
    assert all(len(item.supports) == 4 for item in train + validation)
    assert all(
        item.query.shape == item.desired.shape == item.target.shape == (1, 1, 8)
        for item in train
    )
    assert all(support.shape == (1, 2, 8) for item in train for support in item.supports)
    assert all(not torch.equal(item.query, item.target) for item in train)


def test_episodic_beam_keeps_independent_hard_bank_successors() -> None:
    query, producers, effects = build_effect_federation(
        hidden_dim=8,
        rank=8,
        seed=303,
        device=torch.device("cpu"),
        plastic_branches=4,
        learned_content_projection=False,
        event_write_layout="key-value-pair",
    )
    episode = make_association_episodes(
        split="train",
        seed=404,
        count=1,
        hidden_dim=8,
        support_count=2,
        device=torch.device("cpu"),
    )[0]
    root = query.initial_bank_state()
    trajectories = expand_support_trajectories(
        query,
        producers,
        effects,
        (BankTrajectory(root, torch.tensor(0.0), ()),),
        episode.supports[0],
        support_index=0,
        search_width=2,
        beam_width=8,
    )

    assert len(trajectories) == 8
    assert all(sum(item.bank_state.revisions) == 1 for item in trajectories)
    assert all(item.steps[0].effect_data_identity for item in trajectories)
    assert all(item.steps[0].target in root.slot_refs for item in trajectories)
    assert len(
        {
            tuple(value.detach().numpy().tobytes() for value in item.bank_state.values)
            for item in trajectories
        }
    ) > 1


def test_episodic_loss_backpropagates_only_to_slow_state() -> None:
    query, producers, effects = build_effect_federation(
        hidden_dim=6,
        rank=6,
        seed=505,
        device=torch.device("cpu"),
        plastic_branches=4,
        learned_content_projection=False,
        event_write_layout="key-value-pair",
    )
    episode = make_association_episodes(
        split="train",
        seed=606,
        count=1,
        hidden_dim=6,
        support_count=2,
        device=torch.device("cpu"),
    )[0]

    loss, diagnostics = expected_episode_loss(
        query,
        producers,
        effects,
        episode,
        search_width=2,
        beam_width=4,
    )
    loss.backward()

    bank_values = query.initial_bank_state().values
    assert all(not isinstance(value, torch.nn.Parameter) for value in bank_values)
    assert all(value.grad is None for value in bank_values)
    assert any(parameter.grad is not None for parameter in query.network.parameters())
    assert any(
        parameter.grad is not None
        for effect in effects
        for parameter in effect.parameters()
        if parameter.requires_grad
    )
    assert diagnostics["retained_trajectories"] == 4.0


def test_frozen_episodic_forward_replays_and_carries_every_support() -> None:
    query, producers, _effects = build_effect_federation(
        hidden_dim=8,
        rank=8,
        seed=707,
        device=torch.device("cpu"),
        plastic_branches=4,
        learned_content_projection=False,
        event_write_layout="key-value-pair",
    )
    episodes = make_association_episodes(
        split="test",
        seed=808,
        count=2,
        hidden_dim=8,
        support_count=3,
        device=torch.device("cpu"),
    )

    state, receipts = _hard_supports(query, episodes[0])
    result = evaluate_frozen_forward_adaptation(query, producers, episodes)

    assert sum(state.revisions) == len(episodes[0].supports)
    assert len(receipts) == len(episodes[0].supports)
    assert all(receipt["effect_data_identity"] for receipt in receipts)
    assert all(receipt["effect_data_max_abs_change"] == 0.0 for receipt in receipts)
    assert result["fresh_replay_exact_fraction"] == 1.0
    assert result["slow_state_digest_unchanged"] is True
    assert result["mse"]["proposal_discard"] == result["mse"]["reset"]
    assert result["retrieval_residual_mse"]["correct"] == pytest.approx(
        result["mse"]["correct"],
        abs=1e-8,
    )
    assert sum(result["write_route_counts"].values()) == 6


def test_forward_only_memory_shares_fixed_read_write_coordinates() -> None:
    query, effect = build_forward_only_query(
        {
            "seed": 37,
            "attachment": {"rank": 3, "plastic_branches": 4},
        },
        hidden_dim=8,
        device=torch.device("cpu"),
        rate_multiplier=1.0,
    )
    writer = effect.operand_store.tensor("writer")
    assert not any(parameter.requires_grad for parameter in query.parameters())
    assert set(effect.operand_store.names) == {"rate", "writer"}
    for producer in query.plastic_candidates:
        torch.testing.assert_close(
            producer.candidate.operand_store.tensor("lora.a"),
            writer,
        )

    event = torch.randn(1, 2, 8)
    root = independent_bank_roots(query, 1)[0]
    execution = query({"x": event}, bank_state=root)
    torch.testing.assert_close(execution.value, event)
    assert len(execution.proposals) == 1
    proposal = execution.proposals[0]
    assert torch.count_nonzero(proposal.successor) > 0
    rate = effect.operand_store.tensor("rate")
    expected = (
        rate
        * event[0, 1].unsqueeze(-1)
        * (writer @ event[0, 0]).unsqueeze(0)
    )
    torch.testing.assert_close(proposal.successor, expected)

    latest = torch.randn(1, 2, 8)
    before = query.reexecute(
        proposal.predecessor_id,
        {"x": latest},
        bank_state=root,
    )
    after = query.reexecute(
        proposal.predecessor_id,
        {"x": latest},
        bank_state=execution.bank_state,
    )
    torch.testing.assert_close(before, latest)
    assert not torch.equal(after, before)


def test_forward_only_selection_prefers_task_loss_after_causal_gate() -> None:
    selected = select_configuration(
        (
            {
                "correct_ce": 2.0,
                "correct_vs_swapped_ce_margin": 3.0,
                "correct_vs_no_effect_ce_margin": 1.0,
            },
            {
                "correct_ce": 1.0,
                "correct_vs_swapped_ce_margin": 0.5,
                "correct_vs_no_effect_ce_margin": 0.25,
            },
            {
                "correct_ce": 0.5,
                "correct_vs_swapped_ce_margin": -0.1,
                "correct_vs_no_effect_ce_margin": 2.0,
            },
        )
    )

    assert selected["correct_ce"] == 1.0


def test_learned_event_gates_receive_only_downstream_output_gradient() -> None:
    query, _producer, effect = build_program_query(
        hidden_dim=8,
        rank=3,
        seed=41,
        device=torch.device("cpu"),
        plastic_branches=2,
        shared_write_read_projection=True,
        event_write_layout="learned-key-value-gates",
    )
    query.requires_grad_(False)
    selector = effect.operand_store.tensor("event.selector")
    selector.requires_grad_(True)

    execution = query({"x": torch.randn(1, 5, 8)})
    assert len(execution.proposals) == 1
    adapted = query.reexecute(
        execution.proposals[0].predecessor_id,
        {"x": torch.randn(1, 3, 8)},
        bank_state=execution.bank_state,
    )
    adapted.square().mean().backward()

    assert selector.grad is not None
    assert torch.count_nonzero(selector.grad) > 0
    assert all(
        parameter is selector or parameter.grad is None
        for parameter in query.parameters()
    )


def test_forward_writer_rate_selection_preserves_causality_then_token_quality() -> None:
    selected = select_rate(
        (
            {
                "rate_multiplier": 1.0,
                "correct_token_accuracy": 0.5,
                "correct_ce": 2.0,
                "correct_vs_swapped_ce_margin": 1.0,
                "correct_vs_no_effect_ce_margin": 1.0,
            },
            {
                "rate_multiplier": 2.0,
                "correct_token_accuracy": 0.6,
                "correct_ce": 3.0,
                "correct_vs_swapped_ce_margin": 0.5,
                "correct_vs_no_effect_ce_margin": 0.5,
            },
            {
                "rate_multiplier": 4.0,
                "correct_token_accuracy": 0.9,
                "correct_ce": 1.0,
                "correct_vs_swapped_ce_margin": -0.1,
                "correct_vs_no_effect_ce_margin": 2.0,
            },
        )
    )

    assert selected["rate_multiplier"] == 2.0


def test_sequence_write_observation_is_split_into_ordered_pairs() -> None:
    value = torch.arange(48, dtype=torch.float32).reshape(1, 6, 8)

    pairs = sequence_pairs(value)

    assert len(pairs) == 3
    torch.testing.assert_close(pairs[0], value[:, :2])
    torch.testing.assert_close(pairs[2], value[:, 4:6])


def test_repeated_forward_writes_accumulate_in_one_predecessor_bank() -> None:
    query, producer, _effect = build_program_query(
        hidden_dim=8,
        rank=3,
        seed=97,
        device=torch.device("cpu"),
        plastic_branches=1,
        include_dead_end=False,
        shared_write_read_projection=True,
        event_write_layout="key-value-pair",
    )
    query.requires_grad_(False)
    item = CachedEpisode(
        Episode("pair", "event", "question", "value"),
        torch.randn(1, 6, 8),
        torch.randn(1, 4, 8),
        torch.randn(2, 16),
        torch.randn(2, 16),
        torch.ones(1, 2, dtype=torch.int64),
        torch.tensor([[1, 2]], dtype=torch.int64),
        2,
    )

    (execution,) = execute_sequence_writes(query, (item,))

    assert execution.producer_id == producer.candidate_id
    assert producer.bank_slot_ref is not None
    assert execution.execution.bank_state.revision(producer.bank_slot_ref) == 3
    assert torch.count_nonzero(
        execution.execution.bank_state.value(producer.bank_slot_ref)
    ) > 0
    assert not any(parameter.requires_grad for parameter in query.parameters())


def test_sequence_configuration_selection_requires_causal_memory() -> None:
    selected = select_sequence_configuration(
        (
            {
                "rate_multiplier": 1.0,
                "correct_ce": 3.0,
                "correct_token_accuracy": 0.5,
                "correct_vs_swapped_ce_margin": 1.0,
                "correct_vs_no_effect_ce_margin": 1.0,
            },
            {
                "rate_multiplier": 2.0,
                "correct_ce": 1.0,
                "correct_token_accuracy": 0.9,
                "correct_vs_swapped_ce_margin": -0.1,
                "correct_vs_no_effect_ce_margin": 2.0,
            },
        )
    )

    assert selected["rate_multiplier"] == 1.0


def test_multisite_counterpart_indices_swap_only_within_pairs() -> None:
    cached = tuple(
        CachedEpisode(
            Episode(pair_id, "event", "question", answer),
            torch.randn(1, 2, 8),
            torch.randn(1, 4, 8),
            torch.randn(2, 16),
            torch.randn(2, 16),
            torch.ones(1, 2, dtype=torch.int64),
            torch.tensor([[1, 2]], dtype=torch.int64),
            2,
        )
        for pair_id, answer in (("a", "a0"), ("a", "a1"), ("b", "b0"), ("b", "b1"))
    )

    assert counterpart_indices(cached) == (1, 0, 3, 2)


def test_multisite_selection_keeps_causal_gate_before_lower_ce() -> None:
    selected = select_multisite_configuration(
        (
            {
                "layer_indices": (6,),
                "correct_ce": 2.0,
                "correct_token_accuracy": 0.4,
                "correct_vs_swapped_ce_margin": 0.4,
                "correct_vs_no_effect_ce_margin": 0.5,
            },
            {
                "layer_indices": (1, 3, 4, 6),
                "correct_ce": 1.0,
                "correct_token_accuracy": 0.8,
                "correct_vs_swapped_ce_margin": -0.1,
                "correct_vs_no_effect_ce_margin": 1.0,
            },
        )
    )

    assert selected["layer_indices"] == (6,)


def test_forward_search_candidate_schedule_is_deterministic_and_rate_balanced() -> None:
    specs = tuple(
        candidate_spec(index, seed=11, rate_multipliers=(1.0, 2.0, 4.0))
        for index in range(6)
    )

    assert tuple(rate for _seed, rate in specs) == (1.0, 2.0, 4.0, 1.0, 2.0, 4.0)
    assert len({seed for seed, _rate in specs}) == len(specs)
    assert specs == tuple(
        candidate_spec(index, seed=11, rate_multipliers=(1.0, 2.0, 4.0))
        for index in range(6)
    )


def test_forward_search_selection_rejects_noncausal_low_ce_candidate() -> None:
    selected = select_finalist(
        (
            {
                "candidate_index": 0,
                "correct_ce": 2.0,
                "correct_token_accuracy": 0.4,
                "correct_vs_swapped_ce_margin": 0.3,
                "correct_vs_no_effect_ce_margin": 0.5,
            },
            {
                "candidate_index": 1,
                "correct_ce": 1.0,
                "correct_token_accuracy": 0.9,
                "correct_vs_swapped_ce_margin": -0.1,
                "correct_vs_no_effect_ce_margin": 1.0,
            },
        )
    )

    assert selected["candidate_index"] == 0


def test_adjacent_transition_pairs_preserve_order_and_pair_boundaries() -> None:
    value = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1)

    pairs = adjacent_transition_pairs(value)

    assert pairs.shape == (1, 6, 1)
    torch.testing.assert_close(
        pairs,
        torch.tensor([[[0.0], [1.0], [1.0], [2.0], [2.0], [3.0]]]),
    )


def test_multiscale_transition_pairs_add_forward_skip_relations() -> None:
    value = torch.arange(5, dtype=torch.float32).reshape(1, 5, 1)

    pairs = transition_pairs(value, offsets=(1, 2, 4))

    assert pairs.shape == (1, 16, 1)
    torch.testing.assert_close(
        pairs[:, -2:],
        torch.tensor([[[0.0], [4.0]]]),
    )


def test_forward_writer_trains_only_generic_self_modification_law() -> None:
    protocol = load_protocol(DEFAULT_PROTOCOL)
    query, trainable = build_forward_writer(
        protocol,
        hidden_dim=8,
        device=torch.device("cpu"),
        rank=3,
        rate_multiplier=1.0,
        seed=41,
    )

    assert tuple(trainable) == TRAINABLE_WRITER_OPERANDS
    assert {id(parameter) for parameter in query.parameters() if parameter.requires_grad} == {
        id(parameter) for parameter in trainable.values()
    }
    execution = query({"x": torch.randn(1, 5, 8)})
    adapted = query.reexecute(
        execution.proposals[0].predecessor_id,
        {"x": torch.randn(1, 3, 8)},
        bank_state=execution.bank_state,
    )
    adapted.square().mean().backward()
    assert all(parameter.grad is not None for parameter in trainable.values())


def test_normalized_forward_association_is_finite_and_gradient_free() -> None:
    protocol = load_protocol(DEFAULT_PROTOCOL)
    query, _effect = build_forward_only_query(
        protocol,
        hidden_dim=8,
        device=torch.device("cpu"),
        rate_multiplier=0.1,
        rank=4,
        normalize_association=True,
    )
    event = torch.randn(1, 2, 8)

    execution = query({"x": event})
    output = query.reexecute(
        execution.proposals[0].predecessor_id,
        {"x": torch.randn(1, 3, 8)},
        bank_state=execution.bank_state,
    )

    assert torch.isfinite(output).all()
    assert not any(parameter.requires_grad for parameter in query.parameters())


def test_learned_effect_content_projection_receives_final_output_gradient() -> None:
    query, producer, effect = build_program_query(
        hidden_dim=8,
        rank=3,
        seed=29,
        device=torch.device("cpu"),
        learned_content_projection=True,
    )
    assert producer.bank_slot_ref is not None
    execution = query({"x": torch.randn(1, 1, 8)})
    output = query.reexecute(
        producer.candidate_id,
        {"x": torch.randn(1, 3, 8)},
        bank_state=execution.bank_state,
    )
    output.square().mean().backward()

    content_writer = effect.operand_store.tensor("content.writer")
    assert content_writer.grad is not None
    assert torch.count_nonzero(content_writer.grad) > 0


def test_causal_controls_keep_event_roots_independent_and_swap_only_the_bank() -> None:
    query, _producer, _effect = build_program_query(
        hidden_dim=8,
        rank=3,
        seed=31,
        device=torch.device("cpu"),
        plastic_branches=2,
        include_dead_end=True,
        learned_content_projection=True,
    )
    with torch.no_grad():
        for parameter in query.network.parameters():
            parameter.zero_()
        query.network[-1].bias[0] = 10.0
    cached = tuple(
        CachedEpisode(
            Episode("pair", f"event {value}", "question", value),
            torch.full((1, 1, 8), fill_value),
            torch.randn(1, 4, 8),
            torch.randn(2, 16),
            torch.randn(2, 16),
            torch.ones(1, 2, dtype=torch.int64),
            torch.tensor([[1, 2]], dtype=torch.int64),
            2,
        )
        for value, fill_value in (("left value", 1.0), ("right value", 2.0))
    )

    arms, executions = causal_control_hidden_rows(
        query,
        cached,
        model_dtype=torch.float32,
    )

    assert set(arms) == {
        "correct",
        "no-effect",
        "proposal-discard",
        "state-scrub",
        "wrong-bank-slot",
        "wrong-predecessor",
        "swapped-event-bank",
        "no-predecessor-reexecution",
    }
    assert all(len(rows) == 2 for rows in arms.values())
    assert all(row.producer_id == "plastic-lora-0" for row in executions)
    assert not torch.equal(arms["correct"][0], arms["correct"][1])
    torch.testing.assert_close(arms["no-effect"][0], cached[0].student_hidden)
    torch.testing.assert_close(arms["proposal-discard"][0], cached[0].student_hidden)
    torch.testing.assert_close(arms["state-scrub"][0], cached[0].student_hidden)
    torch.testing.assert_close(arms["wrong-bank-slot"][0], cached[0].student_hidden)
    torch.testing.assert_close(arms["wrong-predecessor"][0], cached[0].student_hidden)
    torch.testing.assert_close(
        arms["no-predecessor-reexecution"][0],
        cached[0].student_hidden,
    )
    assert not torch.equal(arms["swapped-event-bank"][0], arms["correct"][0])


def test_deterministic_epoch_batches_cover_every_episode() -> None:
    batches = [
        _batch_indices(step=step, count=32, batch_size=8, seed=73)
        for step in range(1, 5)
    ]
    assert sorted(index for batch in batches for index in batch) == list(range(32))
    assert batches == [
        _batch_indices(step=step, count=32, batch_size=8, seed=73)
        for step in range(1, 5)
    ]


def test_invalid_route_mass_cannot_escape_the_conditional_task_loss() -> None:
    success = torch.tensor(0.1, requires_grad=True)
    loss = conditional_path_loss(
        success * 10.0,
        success_probability=success,
        invalid_path_weight=2.0,
    )

    torch.testing.assert_close(loss, torch.tensor(11.8))
    loss.backward()
    assert success.grad is not None
    assert float(success.grad) < 0


def test_checkpoint_restores_formula_query_and_optimizer(tmp_path: Path) -> None:
    protocol = load_protocol(DEFAULT_PROTOCOL)
    training = copy.deepcopy(protocol["training"])
    assert isinstance(training, dict)
    query, _producer, effect = build_program_query(
        hidden_dim=8,
        rank=3,
        seed=23,
        device=torch.device("cpu"),
        plastic_branches=2,
        include_dead_end=True,
    )
    optimizer = build_optimizer(query, effect, training)
    optimizer.zero_grad(set_to_none=True)
    loss = sum(parameter.square().sum() for parameter in query.parameters())
    loss.backward()
    optimizer.step()
    expected = {name: value.detach().clone() for name, value in query.state_dict().items()}
    output = tmp_path / "artifact"
    save_checkpoint(
        output,
        query,
        optimizer,
        step=7,
        protocol_path=DEFAULT_PROTOCOL,
        qwen_digest="frozen-qwen-test-digest",
        metrics={"loss": 1.0},
    )

    with torch.no_grad():
        for parameter in query.parameters():
            parameter.add_(10)
    restored_step = load_checkpoint(
        output / "checkpoints" / "latest",
        query,
        optimizer,
        protocol_path=DEFAULT_PROTOCOL,
        device=torch.device("cpu"),
    )

    assert restored_step == 7
    for name, value in query.state_dict().items():
        torch.testing.assert_close(value, expected[name])
    assert optimizer.state
