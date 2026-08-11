from __future__ import annotations

import pytest
import torch

from arti._recall_state import (
    NormalizedDeltaRecallValueUpdater,
    compose_matrix_affine_recall_transitions,
    reduce_matrix_affine_recall_transitions,
)


def test_composed_batch_transition_matches_ordered_serial_application() -> None:
    generator = torch.Generator().manual_seed(7123)
    batch_size, slots, hidden_dim = 4, 6, 9
    matrix = torch.randn(batch_size, slots, slots, generator=generator) * 0.05
    matrix = matrix + torch.eye(slots).unsqueeze(0)
    write = torch.randn(batch_size, slots, hidden_dim, generator=generator) * 0.1
    value = torch.randn(slots, hidden_dim, generator=generator)

    transition = compose_matrix_affine_recall_transitions(matrix, write)
    expected = value
    for index in range(batch_size):
        expected = matrix[index] @ expected + write[index]

    actual = transition.apply(value)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_composed_batch_transition_honors_prompt_order() -> None:
    matrix = torch.stack(
        [
            torch.eye(2) * 0.50,
            torch.eye(2) * 0.25,
            torch.eye(2) * 0.75,
            torch.eye(2) * 0.90,
        ]
    )
    write = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 0.0]],
            [[2.0, 0.0], [0.0, 0.0]],
            [[4.0, 0.0], [0.0, 0.0]],
            [[8.0, 0.0], [0.0, 0.0]],
        ]
    )
    value = torch.zeros(2, 2)

    forward = compose_matrix_affine_recall_transitions(matrix, write).apply(value)
    reverse = compose_matrix_affine_recall_transitions(
        matrix, write, order=(3, 2, 1, 0)
    ).apply(value)

    assert not torch.allclose(forward, reverse)
    expected = value
    for index in range(4):
        expected = matrix[index] @ expected + write[index]
    torch.testing.assert_close(forward, expected)


def test_composed_batch_transition_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="at least one transition"):
        compose_matrix_affine_recall_transitions(
            torch.empty(0, 2, 2),
            torch.empty(0, 2, 2),
        )


@pytest.mark.parametrize("factors", [1, 2, 4])
def test_batched_updater_transition_matches_four_serial_writes(factors: int) -> None:
    generator = torch.Generator().manual_seed(8841 + factors)
    batch_size, token_count, hidden_dim, slots = 4, 7, 12, 5
    updater = NormalizedDeltaRecallValueUpdater(
        hidden_dim,
        slots,
        workspace_dim=8,
        factors=factors,
    )
    trace = torch.randn(batch_size, token_count, hidden_dim, generator=generator)
    mask = torch.rand(batch_size, token_count, generator=generator) > 0.2
    previous = torch.randn(slots, hidden_dim, generator=generator)

    token_transitions = updater.transitions(trace, mask=mask)
    per_prompt = reduce_matrix_affine_recall_transitions(
        token_transitions.matrix,
        token_transitions.write,
        mask=mask,
    )
    composed = compose_matrix_affine_recall_transitions(
        per_prompt.matrix,
        per_prompt.write,
    )
    actual = composed.apply(previous)

    expected = previous
    for index in range(batch_size):
        expected = updater(trace[index], expected, mask=mask[index])

    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
