from __future__ import annotations

from benchmarks.train_predecessor_bank_associative_lookup import summarize_trials


def _trial(*, seed: int, correct: float, control: float) -> dict[str, object]:
    receipt = {
        "predecessor_id": "primary-producer",
        "proposal_count": 1,
        "previous_revision": 0,
        "successor_revision": 1,
        "state_change_norm": 0.5,
        "valid": True,
    }
    return {
        "seed": seed,
        "evaluation": {
            "mse": {
                "correct": correct,
                "no_effect": control,
                "fixed_untrained_effect": control,
                "state_scrub": control,
                "wrong_bank_slot": control,
                "wrong_predecessor": control,
                "without_predecessor_reexecution": control,
            },
            "path_accuracy": 1.0,
            "sample_mechanism_receipts": [receipt],
            "receipt_count": 4,
            "valid_receipt_count": 4,
            "all_receipts_valid": True,
        },
        "save_fresh_reload": {"passed": True},
    }


def test_acceptance_requires_every_seed_to_beat_every_control() -> None:
    accepted = summarize_trials(
        [_trial(seed=17, correct=0.1, control=1.0), _trial(seed=29, correct=0.2, control=1.1)]
    )
    claims = accepted["claims"]
    assert claims["implementation_evidence"] is True
    assert claims["controlled_mechanism_evidence"] is True
    assert claims["real_task_evidence"] is False

    rejected_trials = [
        _trial(seed=17, correct=0.1, control=1.0),
        _trial(seed=29, correct=1.2, control=1.1),
    ]
    rejected = summarize_trials(rejected_trials)
    assert rejected["claims"]["controlled_mechanism_evidence"] is False


def test_acceptance_rejects_partial_receipts_and_missing_reload() -> None:
    partial = _trial(seed=17, correct=0.1, control=1.0)
    partial["evaluation"]["valid_receipt_count"] = 3
    assert summarize_trials([partial])["claims"]["controlled_mechanism_evidence"] is False

    missing_reload = _trial(seed=29, correct=0.1, control=1.0)
    missing_reload["save_fresh_reload"] = None
    claims = summarize_trials([missing_reload])["claims"]
    assert claims["implementation_evidence"] is False
    assert claims["controlled_mechanism_evidence"] is False
