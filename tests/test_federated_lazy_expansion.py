from __future__ import annotations

import pytest
import torch

from arti import mechanisms as m
from benchmarks import train_federated_autonomous_federation as training
from benchmarks._federated_v4_federation import (
    AutonomousEffectFederation,
    build_autonomous_effect_federation,
)
from benchmarks.train_federated_branch_visible_federation import _SearchBranch


def _federation(*, minimum=1, maximum=3, ties=False, structures=()):
    with torch.random.fork_rng():
        torch.manual_seed(942)
        federation = build_autonomous_effect_federation(
            hidden_dim=4, rank=4, seed=942, device=torch.device("cpu"),
            plastic_branches=4, min_operations=minimum, max_operations=maximum,
            ordinary_families=structures,
        )
    with torch.no_grad():
        for effect in federation.effects:
            effect.execution_count_tensor().fill_(2.0)
        if ties:
            federation.query.network[-1].weight.zero_()
            federation.query.network[-1].bias.zero_()
    return federation


def _initial_branches(federation, *, count=2):
    branches, leaves = [], []
    original = federation.query.initial_bank_state()
    for index in range(count):
        x = (
            torch.linspace(-0.4, 0.6, 8).reshape(1, 2, 4) + index * 0.1
        ).requires_grad_()
        values = tuple(
            (value.detach().clone() + index * 0.02).requires_grad_()
            for value in original.values
        )
        root = m.FormulaProgramBankState(
            original.slot_refs, values, tuple(index + 2 for _ in values),
        )
        prior_score = torch.tensor(-0.05 * index, requires_grad=True)
        # Prior support history participates in the same path/family ordering.
        route = ({
            "step": -1, "candidate_id": f"history-{index}",
            "atom_ref": "arti/formula-program@2", "kind": "ordinary",
            "structure_family": "gelu" if index else "low-rank",
        },)
        branches.append(_SearchBranch(
            federation.query._arena({"x": x}, bank_state=root), prior_score, route,
        ))
        leaves.extend((x, *values, prior_score))
    return tuple(branches), tuple(leaves)


def _objective(result):
    scores = torch.stack(tuple(branch.log_probability for branch in result.branches))
    losses = torch.stack(tuple(
        branch.arena.values.get("terminal").square().mean()
        + 0.03 * sum(value.square().mean() for value in branch.arena.committed_state().values)
        for branch in result.branches
    ))
    return (scores.softmax(0) * losses).sum() - 0.1 * scores.mean()


def _assert_state_equal(left, right):
    assert left.slot_refs == right.slot_refs
    assert left.revisions == right.revisions
    for left_value, right_value in zip(left.values, right.values, strict=True):
        torch.testing.assert_close(left_value, right_value, atol=1e-6, rtol=1e-5)


def _assert_result_equal(federation, expected, actual):
    assert actual.maximum_frontier == expected.maximum_frontier
    assert actual.scored_expansions == expected.scored_expansions
    assert len(actual.branches) == len(expected.branches)
    candidates = {candidate.candidate_id: candidate for candidate in federation.query.candidates}
    for left, right in zip(expected.branches, actual.branches, strict=True):
        torch.testing.assert_close(left.log_probability, right.log_probability)
        assert len(left.route) == len(right.route)
        for left_row, right_row in zip(left.route, right.route, strict=True):
            assert left_row.keys() == right_row.keys()
            for key in left_row:
                if key == "update_norm":
                    assert left_row[key] == pytest.approx(right_row[key], abs=1e-6, rel=1e-5)
                else:
                    assert left_row[key] == right_row[key]
        assert left.arena.values.slot_ids == right.arena.values.slot_ids
        for left_value, right_value in zip(
            left.arena.values.values, right.arena.values.values, strict=True,
        ):
            assert (left_value is None) == (right_value is None)
            if left_value is not None:
                torch.testing.assert_close(left_value, right_value, atol=1e-6, rtol=1e-5)
        _assert_state_equal(left.arena.bank_state, right.arena.bank_state)
        _assert_state_equal(left.arena.committed_state(), right.arena.committed_state())
        for left_lineage, right_lineage in zip(
            left.arena.producers, right.arena.producers, strict=True,
        ):
            assert (left_lineage is None) == (right_lineage is None)
            if left_lineage is None:
                continue
            for name in (
                "execution_id", "owner_id", "output_slot", "plastic_slot", "plastic_revision",
            ):
                assert getattr(left_lineage, name) == getattr(right_lineage, name)
            if left_lineage.plastic_value is not None:
                torch.testing.assert_close(left_lineage.plastic_value, right_lineage.plastic_value)
        assert len(left.arena.proposals) == len(right.arena.proposals)
        for left_proposal, right_proposal in zip(
            left.arena.proposals, right.arena.proposals, strict=True,
        ):
            for name in (
                "target", "predecessor_execution_id", "predecessor_owner_id",
                "effect_candidate_id", "effect_instruction_id", "effect_atom_ref",
                "previous_revision", "successor_revision",
            ):
                assert getattr(left_proposal, name) == getattr(right_proposal, name)
            torch.testing.assert_close(left_proposal.previous, right_proposal.previous)
            torch.testing.assert_close(left_proposal.successor, right_proposal.successor)
        # Identity, observed predecessor Bank, and count are actual execution receipts.
        for branch in (left, right):
            for row in branch.route:
                if row["kind"] != "effect":
                    continue
                candidate = candidates[row["candidate_id"]]
                assert row["data_identity"] is True
                if "execution_count" in row:
                    assert row["execution_count"] == candidate.hard_execution_count() == 2
                assert branch.arena.values.get(candidate.output_slot) is branch.arena.values.get(
                    candidate.input_slot
                )
            current = dict(zip(branch.arena.bank_state.slot_refs, branch.arena.bank_state.values))
            for proposal in branch.arena.proposals:
                assert proposal.previous is current[proposal.target]
                current[proposal.target] = proposal.successor


def _assert_gradients_equal(federation, expected, actual, expected_leaves, actual_leaves):
    named_parameters = tuple(federation.query.named_parameters())
    parameters = tuple(value for _, value in named_parameters)
    expected_grads = torch.autograd.grad(
        _objective(expected), (*expected_leaves, *parameters), allow_unused=True,
    )
    actual_grads = torch.autograd.grad(
        _objective(actual), (*actual_leaves, *parameters), allow_unused=True,
    )
    names = (
        *(f"input-or-bank-{index}" for index in range(len(expected_leaves))),
        *(name for name, _ in named_parameters),
    )
    assert any(value is not None for value in expected_grads[:len(expected_leaves)])
    assert any(value is None for value in expected_grads[len(expected_leaves):])
    for name, left, right in zip(names, expected_grads, actual_grads, strict=True):
        assert (left is None) == (right is None), name
        if left is not None:
            torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5, msg=name)


@pytest.mark.parametrize(
    "coverage,minimum,maximum,width,beam,ties,structures",
    [
        (False, 1, 1, 3, 1, False, ()),
        (False, 1, 2, 8, 2, False, ()),
        (False, 2, 3, 6, 4, True, ()),
        (True, 1, 2, 8, 8, True, ()),
        (True, 1, 3, 8, 8, False, ()),
        (True, 2, 3, 8, 3, True, ()),
        (True, 1, 2, 10, 9, False, ("gelu",)),
    ],
)
def test_lazy_matches_oracle_paths_banks_all_gradients_and_work_receipts(
    monkeypatch, coverage, minimum, maximum, width, beam, ties, structures,
):
    federation = _federation(
        minimum=minimum, maximum=maximum, ties=ties, structures=structures,
    )
    original_state = {
        name: value.detach().clone() for name, value in federation.query.state_dict().items()
    }
    original_execute = training.execute_candidates_many
    requests = []

    def counted(rows):
        requests.append(len(rows))
        return original_execute(rows)

    monkeypatch.setattr(training, "execute_candidates_many", counted)
    kwargs = dict(width=width, beam_width=beam, preserve_effect_coverage=coverage)
    expected_branches, expected_leaves = _initial_branches(federation)
    expected = training.search_to_terminal(
        federation, expected_branches, prune_before_execute=False, **kwargs,
    )
    assert expected.executed_expansions == sum(requests)
    assert expected.executed_expansions == expected.scored_expansions - expected.scored_stop_expansions
    requests.clear()
    actual_branches, actual_leaves = _initial_branches(federation)
    # The default path is deliberately exercised, not only the explicit True switch.
    actual = training.search_to_terminal(federation, actual_branches, **kwargs)
    assert actual.executed_expansions == sum(requests)
    assert actual.executed_expansions < expected.executed_expansions
    _assert_result_equal(federation, expected, actual)
    _assert_gradients_equal(federation, expected, actual, expected_leaves, actual_leaves)
    for name, value in federation.query.state_dict().items():
        torch.testing.assert_close(value, original_state[name], atol=0.0, rtol=0.0)


@pytest.mark.parametrize("coverage", [False, True])
def test_ties_existing_completed_and_all_six_effects(monkeypatch, coverage):
    federation = _federation(minimum=1, maximum=3, ties=True)
    original_execute = training.execute_candidates_many
    terminal_results = []

    def record(rows):
        results = original_execute(rows)
        for (candidate, _), arena in zip(rows, results, strict=True):
            if candidate.output_slot == "terminal":
                terminal_results.append((candidate.input_slots["value"], arena))
        return results

    monkeypatch.setattr(training, "execute_candidates_many", record)
    initial, _ = _initial_branches(federation, count=1)
    kwargs = dict(width=8, beam_width=8, preserve_effect_coverage=coverage)
    expected = training.search_to_terminal(
        federation, initial, prune_before_execute=False, **kwargs,
    )
    terminal_results.clear()
    actual = training.search_to_terminal(federation, initial, **kwargs)
    _assert_result_equal(federation, expected, actual)
    assert actual.executed_expansions < expected.executed_expansions
    early = tuple(arena for slot, arena in terminal_results if slot == "operation-0")
    assert early
    # Completed paths from a prior step stay as the original arena objects, without replay.
    assert any(any(branch.arena is arena for arena in early) for branch in actual.branches)
    assert len({id(arena) for _, arena in terminal_results}) == len(terminal_results)
    if coverage:
        observed = {
            row["atom_ref"] for branch in actual.branches for row in branch.route
            if row["kind"] == "effect"
        }
        assert observed == {effect.atom_ref for effect in federation.effects}
        assert len(observed) == 6
        assert len({len(branch.route) for branch in actual.branches}) > 1


@pytest.mark.parametrize("coverage", [False, True])
def test_inference_grouping_preserves_lazy_results(coverage):
    federation = _federation(minimum=1, maximum=2, ties=True)
    branches, _ = _initial_branches(federation)
    kwargs = dict(
        width=8, beam_width=8, preserve_effect_coverage=coverage, record_effect_metrics=False,
    )
    with torch.inference_mode():
        expected = training.search_to_terminal(
            federation, branches, prune_before_execute=False, **kwargs,
        )
        actual = training.search_to_terminal(federation, branches, **kwargs)
    _assert_result_equal(federation, expected, actual)
    assert actual.executed_expansions < expected.executed_expansions


def _scale_candidate(name, input_slot, output_slot, *, weight=1.0):
    value_type = m.TensorType(("B", "D"), ("B", 3), dtype="float32", domain="activation")
    value = m.InputBinding("value", value_type)
    bank = m.BankBinding("weight", "arti/test-lazy-expansion@1", "weight", value_type)
    candidate = m.FormulaProgramCandidateV2(
        name, m.FormulaProgram.build(outputs=(m.scale(value, bank),)),
        input_slots={"value": input_slot}, output_slot=output_slot,
        operands={"weight": torch.full((1, 3), weight)}, trainable_operands=("weight",),
    )
    return m.FormulaProgramTensorCandidateV3(candidate)


def _small_federation(*, bad_weight=1.0):
    good = _scale_candidate("a-good", "x", "operation-0")
    bad = _scale_candidate("b-discarded", "x", "operation-0", weight=bad_weight)
    terminal = _scale_candidate("terminal", "operation-0", "terminal")
    with torch.random.fork_rng():
        torch.manual_seed(715)
        query = m.FormulaProgramQueryV4(
            slot_ids=("x", "operation-0", "terminal"), candidates=(good, bad, terminal),
            terminal_slot="terminal", min_steps=2, max_steps=2, hidden_dim=4,
        )
    with torch.no_grad():
        query.network[-1].bias[0] = 2.0
        query.network[-1].bias[1] = -2.0
    return AutonomousEffectFederation(query, ((good, bad),), ((terminal,),))


def test_full_legal_softmax_keeps_discarded_score_and_query_input_gradients():
    federation = _small_federation()
    query = federation.query
    leaves, results, gradients = [], [], []
    parameters = tuple(query.parameters())
    for lazy in (False, True):
        x = torch.tensor([[0.2, 0.4, 0.7]], requires_grad=True)
        branch = _SearchBranch(query._arena({"x": x}), x.new_zeros(()), ())
        result = training.search_to_terminal(
            federation, (branch,), width=2, beam_width=1,
            preserve_effect_coverage=False, prune_before_execute=lazy,
        )
        leaves.append(x)
        results.append(result)
        # Score-only loss cannot obtain x's gradient through the terminal task value.
        gradients.append(torch.autograd.grad(
            -result.branches[0].log_probability, (x, *parameters), allow_unused=True,
        ))
    for left, right in zip(*gradients, strict=True):
        assert (left is None) == (right is None)
        if left is not None:
            torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)
    assert gradients[1][0] is not None and gradients[1][0].abs().sum() > 0
    bias_index = next(index for index, value in enumerate(parameters) if value is query.network[-1].bias)
    assert gradients[1][bias_index + 1][1].abs() > 0
    discarded = federation.transition_layers[0][1].candidate.operand_store.tensor("weight")
    discarded_index = next(index for index, value in enumerate(parameters) if value is discarded)
    assert gradients[0][discarded_index + 1] is gradients[1][discarded_index + 1] is None
    assert results[0].scored_expansions == results[1].scored_expansions == 4
    assert results[0].scored_stop_expansions == results[1].scored_stop_expansions == 1
    assert results[0].executed_expansions == 3
    assert results[1].executed_expansions == 2


@pytest.mark.parametrize("retain_bad", [False, True])
def test_numerical_rejection_refills_beam_without_reweighting_or_replay(monkeypatch, retain_bad):
    federation = _small_federation(bad_weight=1e38)
    query = federation.query
    with torch.no_grad():
        query.network[-1].weight.zero_()
        query.network[-1].bias[1] = 4.0 if retain_bad else -2.0
    branch = _SearchBranch(query._arena({"x": torch.full((1, 3), 4.0)}), torch.zeros(()), ())
    kwargs = dict(width=2, beam_width=1, preserve_effect_coverage=False)
    assert all(candidate.accepts(branch.arena) for candidate in federation.transition_layers[0])
    expected = training.search_to_terminal(
        federation, (branch,), prune_before_execute=False, **kwargs,
    )
    calls = []
    original = training.execute_candidates_many

    def record(rows):
        calls.extend((candidate.candidate_id, id(arena)) for candidate, arena in rows)
        return original(rows)

    monkeypatch.setattr(training, "execute_candidates_many", record)
    result = training.search_to_terminal(federation, (branch,), **kwargs)
    _assert_result_equal(federation, expected, result)
    assert len(calls) == len(set(calls)) == result.executed_expansions
    assert len(expected.numerical_rejections) == 1
    assert len(result.numerical_rejections) == int(retain_bad)
    assert result.executed_expansions == 2 + int(retain_bad)
    assert result.scored_expansions == 4
    assert result.scored_stop_expansions == 1
    if retain_bad:
        assert result.numerical_rejections[0]["candidate_id"] == "b-discarded"
        assert result.numerical_rejections[0]["code"] == "FF2_NONFINITE"
    assert branch.arena.values.get("terminal") is None
    loss = result.branches[0].arena.values.get("terminal").sum()
    loss.backward()
    rejected_weight = federation.transition_layers[0][1].candidate.operand_store.tensor("weight")
    assert rejected_weight.grad is None


def test_all_numerical_candidates_invalid_reports_failure_not_identity():
    federation = _small_federation(bad_weight=1e38)
    with torch.no_grad():
        federation.transition_layers[0][0].candidate.operand_store.tensor("weight").fill_(1e38)
    branch = _SearchBranch(
        federation.query._arena({"x": torch.full((1, 3), 4.0)}), torch.zeros(()), (),
    )
    with pytest.raises(RuntimeError, match="2 numerical expansions rejected"):
        training.search_to_terminal(
            federation, (branch,), width=2, beam_width=1, preserve_effect_coverage=False,
        )


@pytest.mark.parametrize("coverage", [False, True])
def test_stop_admission_refills_completed_and_preserves_effect_continuations(coverage):
    federation = build_autonomous_effect_federation(
        hidden_dim=4, rank=4, seed=942, device=torch.device("cpu"),
        plastic_branches=4, min_operations=1, max_operations=2, max_effect_operations=2,
    )
    with torch.no_grad():
        federation.query.network[-1].weight.zero_()
        federation.query.network[-1].bias.zero_()
    calls = []

    def admit(branch):
        key = tuple(row["candidate_id"] for row in branch.route)
        assert key not in calls
        calls.append(key)
        # A rejected early STOP must not prevent further post-output effects.
        return any("-stage-tail-" in name for name in key)

    initial, _ = _initial_branches(federation, count=1)
    kwargs = dict(width=8, beam_width=4, preserve_effect_coverage=coverage,
                  terminal_admission=admit)
    expected = training.search_to_terminal(
        federation, initial, prune_before_execute=False, **kwargs,
    )
    assert expected.checked_stop_readouts == len(calls)
    calls.clear()
    result = training.search_to_terminal(federation, initial, **kwargs)
    _assert_result_equal(federation, expected, result)
    assert result.checked_stop_readouts == len(calls)
    assert result.checked_stop_readouts <= expected.checked_stop_readouts
    assert any(row["kind"] == "stop_readout" for row in result.numerical_rejections)
    assert all(any("-stage-tail-" in row["candidate_id"] for row in branch.route)
               for branch in result.branches)


@pytest.mark.parametrize("lazy", [False, True])
def test_all_rejected_stops_cannot_escape_as_unchecked_previews(lazy):
    federation = _small_federation()
    branch = _SearchBranch(
        federation.query._arena({"x": torch.ones(1, 3)}), torch.zeros(()), (),
    )
    with pytest.raises(RuntimeError, match="did not reach a terminal path"):
        training.search_to_terminal(
            federation, (branch,), width=2, beam_width=2,
            preserve_effect_coverage=False, prune_before_execute=lazy,
            terminal_admission=lambda _branch: False,
        )
