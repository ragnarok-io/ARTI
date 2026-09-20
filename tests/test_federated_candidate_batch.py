from __future__ import annotations

import pytest
import torch

from arti import mechanisms
from benchmarks._federated_candidate_batch import execute_candidates_many
from benchmarks._federated_v4_federation import build_autonomous_effect_federation


def _fixture(device="cpu"):
    with torch.random.fork_rng():
        torch.manual_seed(834)
        return build_autonomous_effect_federation(
            hidden_dim=4,
            rank=4,
            seed=834,
            device=torch.device(device),
            plastic_branches=4,
            min_operations=1,
            max_operations=3,
        )


@pytest.mark.parametrize(
    "family", ("ordinary", "affine", "blend", "outer", "transport", "polynomial", "proximal")
)
def test_batch_matches_native_outputs_bank_lineage_and_gradients(family):
    federation = _fixture()
    producers = tuple(
        item
        for item in federation.transition_layers[0]
        if isinstance(item, mechanisms.FormulaProgramTensorCandidateV3)
    )
    requests = []
    leaves = []
    for index, producer in enumerate(producers[:3]):
        x = torch.randn(1, 2, 4, requires_grad=True)
        bank_value = torch.randn_like(producer.bank_owner.value, requires_grad=True)
        root = federation.query.initial_bank_state().replace(
            producer.bank_slot_ref, bank_value, revision=index + 1
        )
        arena = federation.query._arena({"x": x}, bank_state=root)
        leaves.extend((x, bank_value))
        if family == "ordinary":
            requests.append((producer, arena))
        else:
            effect = next(
                item
                for item in federation.transition_layers[1]
                if item.candidate_id == f"{family}-write-stage-1"
            )
            requests.append((effect, producer(arena)))
    expected = tuple(candidate(arena) for candidate, arena in requests)
    actual = execute_candidates_many(requests)

    def objective(rows):
        result = torch.zeros(())
        for index, ((candidate, _), row) in enumerate(zip(requests, rows, strict=True)):
            value = row.values.get(candidate.output_slot)
            result = result + (index + 1) * value.square().mean()
            if family != "ordinary":
                result = result + (index + 2) * row.proposals[-1].successor.square().mean()
        return result

    for (candidate, source), serial, batched in zip(requests, expected, actual, strict=True):
        torch.testing.assert_close(
            batched.values.get(candidate.output_slot),
            serial.values.get(candidate.output_slot),
            atol=1e-6,
            rtol=1e-5,
        )
        assert batched.bank_state is source.bank_state
        if family != "ordinary":
            assert batched.values.get(candidate.output_slot) is source.values.get(
                candidate.input_slot
            )
            left, right = serial.proposals[-1], batched.proposals[-1]
            assert left.target == right.target
            assert left.predecessor_execution_id == right.predecessor_execution_id
            assert left.previous_revision == right.previous_revision
            assert left.successor_revision == right.successor_revision
            assert left.previous is right.previous
            torch.testing.assert_close(left.successor, right.successor, atol=1e-6, rtol=1e-5)
            assert batched.producer(candidate.output_slot).plastic_value is right.successor
    parameters = tuple(federation.query.parameters())
    expected_grad = torch.autograd.grad(
        objective(expected), (*leaves, *parameters), allow_unused=True, retain_graph=True
    )
    actual_grad = torch.autograd.grad(objective(actual), (*leaves, *parameters), allow_unused=True)
    for left, right in zip(expected_grad, actual_grad, strict=True):
        assert (left is None) == (right is None)
        if left is not None:
            torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)


def test_batch_groups_different_input_shapes_and_preserves_request_order():
    federation = _fixture()
    producer = federation.producers[0]
    requests = tuple(
        (producer, federation.query._arena({"x": torch.randn(1, length, 4)}))
        for length in (2, 5, 2, 3)
    )
    actual = execute_candidates_many(requests)
    for (candidate, arena), row in zip(requests, actual, strict=True):
        torch.testing.assert_close(
            row.values.get(candidate.output_slot),
            candidate(arena).values.get(candidate.output_slot),
        )
    assert execute_candidates_many(()) == ()


@pytest.mark.parametrize("grad_enabled", (False, True))
@pytest.mark.parametrize("device", ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",))
def test_sliced_effects_use_checked_plan_without_native_dispatch(monkeypatch, grad_enabled, device):
    from arti._formula_candidate_batch import _checked_plan

    federation = _fixture(device)
    requests = []
    for candidate in federation.transition_layers[1]:
        if not isinstance(candidate, mechanisms.FormulaProgramEffectCandidateV3):
            continue
        program = candidate.effect_program.program
        assert any(item.atom_ref == "arti/formula-atom-slice@1" for item in program.instructions)
        assert _checked_plan(program) is not None
        for producer in federation.producers[:2]:
            arena = federation.query._arena({"x": torch.randn(1, 2, 4, device=device)})
            requests.append((candidate, producer(arena)))
    assert requests

    with torch.set_grad_enabled(grad_enabled):
        expected = tuple(candidate(arena) for candidate, arena in requests)

        def reject_native(*args, **kwargs):
            raise AssertionError("sliced effect unexpectedly used native dispatch")

        monkeypatch.setattr(mechanisms.FormulaProgramEffectCandidateV3, "forward", reject_native)
        actual = execute_candidates_many(requests)
    for serial, checked in zip(expected, actual, strict=True):
        torch.testing.assert_close(checked.proposals[-1].successor, serial.proposals[-1].successor)
    if grad_enabled:
        parameters = tuple(federation.query.parameters())
        native_loss = sum(row.proposals[-1].successor.square().mean() for row in expected)
        checked_loss = sum(row.proposals[-1].successor.square().mean() for row in actual)
        native_grads = torch.autograd.grad(native_loss, parameters, allow_unused=True, retain_graph=True)
        checked_grads = torch.autograd.grad(checked_loss, parameters, allow_unused=True)
        for native_grad, checked_grad in zip(native_grads, checked_grads, strict=True):
            assert (native_grad is None) == (checked_grad is None)
            if native_grad is not None:
                torch.testing.assert_close(native_grad, checked_grad, rtol=1e-5, atol=1e-7)


def test_search_paths_probabilities_and_task_gradients_match_native(monkeypatch):
    from benchmarks import train_federated_autonomous_federation as training
    from benchmarks.train_federated_self_modifying_federation import make_association_episodes

    federation = _fixture()
    episode = make_association_episodes(
        split="train",
        seed=921,
        count=1,
        hidden_dim=4,
        support_count=2,
        device=torch.device("cpu"),
    )[0]
    kwargs = dict(width=8, beam_width=8, hard_alignment_weight=0.1)
    batch_loss, batch_info = training.expected_episode_loss(federation, episode, **kwargs)
    parameters = tuple(federation.query.parameters())
    batch_grads = torch.autograd.grad(batch_loss, parameters, allow_unused=True)
    monkeypatch.setattr(
        training,
        "execute_candidates_many",
        lambda rows: tuple(candidate(arena) for candidate, arena in rows),
    )
    native_loss, native_info = training.expected_episode_loss(federation, episode, **kwargs)
    native_grads = torch.autograd.grad(native_loss, parameters, allow_unused=True)
    torch.testing.assert_close(batch_loss, native_loss)
    for key in (
        "retained_paths",
        "maximum_frontier",
        "effect_path_fraction",
        "best_branch_uses_effect",
    ):
        assert batch_info[key] == native_info[key]
    for left, right in zip(batch_grads, native_grads, strict=True):
        assert (left is None) == (right is None)
        if left is not None:
            torch.testing.assert_close(left, right, atol=2e-6, rtol=2e-5)
