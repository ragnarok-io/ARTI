from __future__ import annotations

import json

import pytest

import arti
from test_training_run import attached, data, policy


def test_completed_checkpoint_hook_receives_coherent_assets(tmp_path):
    session = attached().arti.trainer()
    received = []

    def completed(current, path):
        assert current is session and path.exists()
        assert len(current.loss_history) == current.global_step
        assert current._run.latest == path
        received.append(current.global_step)

    result = session.fit(
        data(), steps=4, run=policy(tmp_path, checkpoint_steps=2), on_checkpoint=completed
    )
    assert received == [1, 2, 4]
    assert result.steps == 4 and session._on_checkpoint is None


def test_critical_handoff_failure_is_not_a_disabled_reporter(tmp_path):
    session = attached().arti.trainer()

    def fail(_current, _path):
        raise OSError("checkpoint destination unavailable")

    with pytest.raises(OSError, match="destination unavailable"):
        session.fit(data(), steps=3, run=policy(tmp_path), on_checkpoint=fail)
    assert session.global_step == 1
    assert session._on_checkpoint is None
    saved = session._run.latest
    assert saved.exists()
    state = json.loads((tmp_path / "run.json").read_text())
    assert state["status"] == "failed" and state["error_type"] == "OSError"
    assert "reporter_error" not in state


def test_callback_cleared_for_the_next_fit(tmp_path):
    session = attached().arti.trainer()
    received = []
    session.fit(
        data(),
        steps=1,
        run=policy(tmp_path / "first"),
        on_checkpoint=lambda current, _: received.append(current.global_step),
    )
    session.fit(data(), steps=1, run=policy(tmp_path / "second"))
    assert received == [1]


def test_portable_best_closure_even_when_export_stays_inside_run(tmp_path):
    session = attached().arti.trainer(learning_rate=0.02)
    validation_data = [{"inputs": row["inputs"], "labels": row["labels"] * 0} for row in data()]
    run = policy(tmp_path / "run", evaluation_steps=1, validation=arti.ARTIValidationConfig())
    session.fit(data(), steps=3, run=run, eval_data=validation_data)
    assert session.selection_receipt()["best_step"] == 1
    saved = session.save_checkpoint(tmp_path / "run" / "portable" / "model.arti.st", portable=True)
    restored = attached().arti.trainer()
    restored.load_checkpoint(saved)
    best = restored._validation.best_checkpoint
    assert best.is_relative_to(saved.parent) and best.exists()
    assert (saved.parent / "validation-best").is_dir()


def test_portable_copy_cannot_recurse_into_its_source(tmp_path):
    session = attached().arti.trainer()
    validation_data = [{"inputs": row["inputs"], "labels": row["labels"] * 0} for row in data()]
    run = policy(tmp_path, evaluation_steps=1, validation=arti.ARTIValidationConfig())
    session.fit(data(), steps=1, run=run, eval_data=validation_data)
    best = session._validation.best_checkpoint
    with pytest.raises(ValueError, match="inside its best checkpoint bundle"):
        session.save_checkpoint(best.parent / "nested" / "model.arti.st", portable=True)


def test_pretrained_hook_uses_shared_session(tmp_path):
    from test_shared_training import paired_data, workflow

    current = workflow()
    seen = []
    result = current.fit(
        paired_data(),
        steps=2,
        run=policy(tmp_path),
        on_checkpoint=lambda session, _: seen.append(session.global_step),
    )
    assert seen == [1, 2] and result.steps == 2


def test_classification_and_language_recipes_forward_hook(tmp_path):
    pytest.importorskip("transformers")
    from test_task_recipes import _config
    from test_language_recipe import recipe

    classification_dir = tmp_path / "classification"
    classification_dir.mkdir()
    classifier = arti.ClassificationRecipe(_config(classification_dir))
    class_seen = []
    classifier.fit(
        steps=2,
        run=policy(classification_dir / "run"),
        on_checkpoint=lambda session, _: class_seen.append(session.global_step),
    )
    assert class_seen == [1, 2]
    language_dir = tmp_path / "language"
    language_dir.mkdir()
    language = recipe(language_dir)
    language_seen = []
    language.fit(
        steps=2,
        run=policy(language_dir / "run"),
        on_checkpoint=lambda session, _: language_seen.append(session.global_step),
    )
    assert language_seen == [1, 2]


def test_nonlocal_engine_uses_native_checkpoint_callbacks():
    session = attached().arti.trainer(engine="transformers", device="cpu")
    with pytest.raises(ValueError, match="other engines own their run lifecycle"):
        session.fit(data(), steps=1, on_checkpoint=lambda *_: None)
