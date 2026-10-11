from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from torch import nn

import arti
from arti.training_validation import ARTIValidationConfig, _ValidationTracker


def policy(path, **values):
    return arti.ARTITrainingRunConfig(
        output_dir=path, evaluation_steps=1, validation=ARTIValidationConfig(**values)
    )


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.register_buffer("observed", torch.zeros(()))

    def forward(self, inputs, labels):
        self.observed.add_(1)
        return {"loss": (inputs * self.weight - labels).square().mean()}


def batches():
    return [{"inputs": torch.ones(2), "labels": torch.zeros(2)}]


def trainer(model=None, **kwargs):
    return arti.trainer(
        model or Model(), config=arti.ARTITrainingConfig(device="cpu", learning_rate=0.1), **kwargs
    )


def test_config_round_trip_and_backend_identity():
    assert arti.ARTIValidationConfig is arti.torch.ARTIValidationConfig
    config = policy("outputs", monitor="accuracy", mode="max", patience=3)
    restored = arti.ARTITrainingRunConfig(**json.loads(json.dumps(config.to_dict())))
    assert restored == config


@pytest.mark.parametrize(
    "options",
    [
        {"mode": "auto"},
        {"selection": "test"},
        {"monitor": ""},
        {"patience": 0},
        {"patience": True},
        {"start_step": -1},
        {"min_evaluations": 0},
        {"min_delta": float("nan")},
        {"min_delta": True},
    ],
)
def test_invalid_selection_policy(options):
    with pytest.raises(ValueError):
        ARTIValidationConfig(**options)


def test_policy_requires_explicit_cadence():
    with pytest.raises(ValueError, match="evaluation_steps"):
        arti.ARTITrainingRunConfig(validation=ARTIValidationConfig())


def test_true_best_and_significant_improvement_are_separate():
    tracker = _ValidationTracker(ARTIValidationConfig(min_delta=0.1, patience=2))
    assert tracker.observe({"loss": 1.0}, 1) == (True, False)
    assert tracker.observe({"loss": 0.99}, 2) == (True, False)
    assert tracker.observe({"loss": 0.98}, 3) == (True, True)
    assert tracker.best_value == 0.98 and tracker.reference_value == 1.0


def test_warmup_and_minimum_validation_count():
    tracker = _ValidationTracker(ARTIValidationConfig(start_step=3, min_evaluations=4, patience=1))
    assert tracker.observe({"loss": 0.1}, 1) == (False, False)
    for step in (3, 4, 5):
        assert not tracker.observe({"loss": 1.0}, step)[1]
    assert tracker.observe({"loss": 1.0}, 6)[1]
    assert tracker.best_step == 3


@pytest.mark.parametrize(
    "mode,scores,best", [("min", [1.0, 2.0, 3.0], 1), ("max", [1.0, 2.0, 1.0], 2)]
)
def test_best_and_last_are_distinct_and_context_does_not_rewind_training(
    tmp_path, mode, scores, best
):
    session = trainer()
    weights = {}

    def validation(current):
        weights[current.global_step] = current.model.weight.detach().clone()
        current.model.observed.add_(100)
        return {"loss": scores[current.global_step - 1], "split": "validation"}

    result = session.fit(
        batches(), steps=3, run=policy(tmp_path, mode=mode), validation_fn=validation
    )
    assert result.selection["selected_step"] == best
    assert result.selected_checkpoint_path == result.best_checkpoint_path != result.checkpoint_path
    assert session.global_step == 3 and session.model.observed == 3
    last = session.model.weight.detach().clone()
    optimizer = copy.deepcopy(session.optimizer.state_dict())
    with session.checkpoint_context(result.selected_checkpoint_path):
        torch.testing.assert_close(session.model.weight, weights[best], rtol=0, atol=0)
    torch.testing.assert_close(session.model.weight, last, rtol=0, atol=0)
    for key, state in session.optimizer.state_dict()["state"].items():
        for name, value in state.items():
            torch.testing.assert_close(value, optimizer["state"][key][name], rtol=0, atol=0)
    assert session.global_step == 3
    report = json.loads((tmp_path / "run.json").read_text())
    assert report["best_checkpoint"] != report["latest_checkpoint"]


def test_patience_stops_at_validation_boundary_and_saves_final_state(tmp_path):
    session = trainer()
    result = session.fit(
        batches(), steps=20, run=policy(tmp_path, patience=2), validation_fn=lambda _: {"loss": 1.0}
    )
    assert result.steps == 3 and result.stop_reason == "early_stopping"
    restored = trainer(resume_from_checkpoint=result.checkpoint_path)
    assert restored.global_step == 3 and restored._validation.bad_evaluations == 2


def test_default_policy_does_not_stop_on_bad_validation(tmp_path):
    session = trainer()
    result = session.fit(
        batches(),
        steps=5,
        run=policy(tmp_path),
        validation_fn=lambda current: {"loss": float(current.global_step)},
    )
    assert result.steps == 5 and result.stop_reason == "steps_completed"


def test_resume_keeps_best_bundle_and_patience(tmp_path):
    full = trainer()
    full_result = full.fit(
        batches(),
        steps=10,
        run=policy(tmp_path / "full", patience=3),
        validation_fn=lambda _: {"loss": 1.0},
    )
    first = trainer()
    first_result = first.fit(
        batches(),
        steps=2,
        run=policy(tmp_path / "first", patience=3),
        validation_fn=lambda _: {"loss": 1.0},
    )
    continued = trainer(resume_from_checkpoint=first_result.checkpoint_path)
    result = continued.fit(
        batches(),
        steps=8,
        run=policy(tmp_path / "continued", patience=3),
        validation_fn=lambda _: {"loss": 1.0},
    )
    assert result.steps == full_result.steps == 4
    assert result.selection == full_result.selection
    assert result.best_checkpoint_path.parent == tmp_path / "continued" / "best"
    torch.testing.assert_close(continued.model.weight, full.model.weight, rtol=0, atol=0)
    snapshot = arti.load(result.checkpoint_path).training_state
    assert not Path(snapshot["validation"]["best_checkpoint"]).is_absolute()


def test_off_cadence_final_check_does_not_reset_patience(tmp_path):
    config = ARTIValidationConfig(patience=2)
    full, first = trainer(), trainer()
    full_result = full.fit(
        batches(),
        steps=10,
        run=arti.ARTITrainingRunConfig(
            output_dir=tmp_path / "full", evaluation_steps=2, validation=config
        ),
        validation_fn=lambda _: {"loss": 1.0},
    )
    partial = first.fit(
        batches(),
        steps=3,
        run=arti.ARTITrainingRunConfig(
            output_dir=tmp_path / "first", evaluation_steps=2, validation=config
        ),
        validation_fn=lambda _: {"loss": 1.0},
    )
    continued = trainer(resume_from_checkpoint=partial.checkpoint_path)
    actual = continued.fit(
        batches(),
        steps=7,
        run=arti.ARTITrainingRunConfig(
            output_dir=tmp_path / "continued", evaluation_steps=2, validation=config
        ),
        validation_fn=lambda _: {"loss": 1.0},
    )
    assert actual.steps == full_result.steps == 6
    assert actual.selection["evaluations"] == 3
    torch.testing.assert_close(continued.model.weight, full.model.weight, rtol=0, atol=0)


@pytest.mark.parametrize(
    "metrics", [{"split": "test", "loss": 1.0}, {"accuracy": 0.5}, {"loss": float("nan")}]
)
def test_invalid_monitor_preserves_first_completed_update(tmp_path, metrics):
    session = trainer()
    with pytest.raises(ValueError):
        session.fit(batches(), steps=2, run=policy(tmp_path), validation_fn=lambda _: metrics)
    payload = json.loads((tmp_path / "run.json").read_text())
    assert payload["status"] == "failed"
    assert (tmp_path / payload["latest_checkpoint"]).exists()


def test_no_eligible_check_records_last_fallback(tmp_path):
    result = trainer().fit(
        batches(),
        steps=2,
        run=policy(tmp_path, start_step=100),
        validation_fn=lambda _: {"loss": 1.0},
    )
    assert result.best_checkpoint_path is None
    assert result.selection["reason"] == "no_eligible_validation"
    assert result.selected_checkpoint_path == result.checkpoint_path


def test_zero_step_resume_preserves_loader_cursor_and_best(tmp_path):
    session = trainer()
    first = session.fit(batches(), steps=2, run=policy(tmp_path / "first"), eval_data=batches())
    continued = trainer(resume_from_checkpoint=first.checkpoint_path)
    result = continued.fit(
        batches(), steps=0, run=policy(tmp_path / "continued"), eval_data=batches()
    )
    assert result.steps == 2 and result.best_checkpoint_path.exists()
    assert continued.micro_batches_seen == 2
    assert continued._validation.config == ARTIValidationConfig()


def test_resume_rejects_implicit_policy_change_before_updates(tmp_path):
    first = trainer().fit(
        batches(), steps=1, run=policy(tmp_path / "first", patience=3), eval_data=batches()
    )
    current = trainer(resume_from_checkpoint=first.checkpoint_path)
    with pytest.raises(ValueError, match="resume validation policy"):
        current.fit(
            batches(), steps=3, run=policy(tmp_path / "changed", patience=2), eval_data=batches()
        )
    assert current.global_step == 1 and not (tmp_path / "changed").exists()


def test_resume_rejects_changed_validation_cadence(tmp_path):
    first = trainer().fit(batches(), steps=1, run=policy(tmp_path / "first"), eval_data=batches())
    current = trainer(resume_from_checkpoint=first.checkpoint_path)
    run = arti.ARTITrainingRunConfig(
        output_dir=tmp_path / "changed", evaluation_steps=2, validation=ARTIValidationConfig()
    )
    with pytest.raises(ValueError, match="cadence"):
        current.fit(batches(), steps=1, run=run, eval_data=batches())


def test_portable_training_export_carries_best_bundle(tmp_path):
    session = trainer()
    session.fit(
        batches(),
        steps=3,
        run=policy(tmp_path / "run"),
        validation_fn=lambda current: {"loss": float(current.global_step)},
    )
    exported = session.save_checkpoint(tmp_path / "export" / "model.st")
    state = arti.load(exported).training_state
    assert state["validation"]["best_checkpoint"] == str(Path("validation-best") / "model.arti.st")
    restored = trainer(resume_from_checkpoint=exported)
    result = restored.fit(
        batches(),
        steps=1,
        run=policy(tmp_path / "resumed"),
        validation_fn=lambda current: {"loss": float(current.global_step)},
    )
    assert result.selection["selected_step"] == 1


def test_cached_session_inherits_prior_best_and_uses_same_patience(tmp_path):
    session = trainer()
    first = session.fit(
        batches(),
        steps=2,
        run=policy(tmp_path / "first", patience=3),
        validation_fn=lambda _: {"loss": 1.0},
    )
    second = session.fit(
        batches(),
        steps=10,
        run=policy(tmp_path / "second", patience=3),
        validation_fn=lambda _: {"loss": 1.0},
    )
    assert first.steps == 2 and second.steps == 4
    assert second.selection["best_step"] == 1


def test_zero_updates_preserve_pending_stateful_loader_cursor(tmp_path):
    class Loader:
        def __init__(self):
            self.cursor = 0

        def __iter__(self):
            while self.cursor < 100:
                self.cursor += 1
                yield batches()[0]

        def state_dict(self):
            return {"cursor": self.cursor}

        def load_state_dict(self, state):
            self.cursor = state["cursor"]

    first = trainer().fit(Loader(), steps=3, run=policy(tmp_path / "first"), eval_data=batches())
    resumed = trainer(resume_from_checkpoint=first.checkpoint_path)
    saved = resumed.fit(Loader(), steps=0, run=policy(tmp_path / "zero"), eval_data=batches())
    assert arti.load(saved.checkpoint_path).training_state["data_state"] == {"cursor": 3}
    continued = trainer(resume_from_checkpoint=saved.checkpoint_path)
    actual = continued.fit(Loader(), steps=1, run=policy(tmp_path / "next"), eval_data=batches())
    assert arti.load(actual.checkpoint_path).training_state["data_state"] == {"cursor": 4}


def test_model_only_context_cannot_mix_best_weights_with_last_optimizer(tmp_path):
    session = trainer()
    result = session.fit(batches(), steps=2, run=policy(tmp_path / "run"), eval_data=batches())
    with session.checkpoint_context(result.selected_checkpoint_path):
        with pytest.raises(ValueError, match="model-only checkpoint context"):
            session.save_checkpoint(tmp_path / "bad.st")
        with pytest.raises(ValueError, match="model-only checkpoint context"):
            session.fit(batches(), steps=1)
    assert not (tmp_path / "bad.st").exists()
