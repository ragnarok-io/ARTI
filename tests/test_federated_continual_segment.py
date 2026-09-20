from types import SimpleNamespace

from benchmarks import train_federated_spatial_continual as runner


def harness(monkeypatch, *, step, pending):
    clock, logs, saves = [0.], [], []
    monkeypatch.setattr(runner.time, "perf_counter", lambda: clock[0])
    progress = SimpleNamespace(pending_validation_step=pending, next_batch_index=0,
        training_seconds=0., evaluation_seconds=0.)
    def checkpoint(*, reason):
        saves.append(reason)
        clock[0] += 1
        return "checkpoint"
    run = SimpleNamespace(progress=SimpleNamespace(step=step), checkpoint=checkpoint, maybe_checkpoint=lambda: None)
    def train():
        assert progress.pending_validation_step is None
        run.progress.step += 1
        progress.next_batch_index += 1
        progress.training_seconds += 4
        clock[0] += 4
        return dict(kind="train", seconds=4., updated=True)
    def validate():
        assert progress.pending_validation_step == run.progress.step
        progress.pending_validation_step = None
        progress.evaluation_seconds += 2
        clock[0] += 2
        return dict(kind="validate", seconds=2.)
    return run, progress, clock, logs, saves, train, validate


def test_segment_validates_before_training_and_after_final_update(monkeypatch):
    run, progress, clock, logs, saves, train, validate = harness(monkeypatch, step=5, pending=5)
    runner.run_segment(run, progress, target_step=7, segment_seconds=100, started=0,
        update_estimate=4, validation_estimate=2, evaluate_every=6, train=train, validate=validate, log=logs.append)
    assert [row["kind"] for row in logs] == ["validate", "train", "validate", "train", "validate", "continuous_segment_end"]
    assert run.progress.step == 7 and progress.pending_validation_step is None
    assert saves == ["validation-complete"] * 3 + ["segment-end"]
    assert logs[-1]["phase_complete"] and logs[-1]["segment_wall_seconds"] == clock[0]


def test_setup_time_reduces_segment_budget_without_starting_a_batch(monkeypatch):
    run, progress, clock, logs, saves, train, validate = harness(monkeypatch, step=5, pending=None)
    clock[0] = 45
    runner.run_segment(run, progress, target_step=7, segment_seconds=50, started=0,
        update_estimate=10, validation_estimate=2, evaluate_every=6, train=train, validate=validate, log=logs.append)
    assert run.progress.step == 5 and progress.next_batch_index == 0
    assert saves == ["segment-end"] and not logs[-1]["phase_complete"]


def test_pending_partial_validation_survives_segment_boundary(monkeypatch):
    run, progress, clock, logs, saves, train, _ = harness(monkeypatch, step=7, pending=7)
    parts = []
    def validate():
        parts.append(len(parts))
        clock[0] += 4
        if len(parts) == 2:
            progress.pending_validation_step = None
        return dict(kind="validate", seconds=4.)
    kwargs = dict(target_step=7, update_estimate=4, validation_estimate=4,
                  evaluate_every=6, train=train, validate=validate, log=logs.append)
    runner.run_segment(run, progress, segment_seconds=18, started=0, **kwargs)
    assert parts == [0] and progress.pending_validation_step == 7
    assert logs[-1]["target_reached"] and not logs[-1]["phase_complete"]
    runner.run_segment(run, progress, segment_seconds=30, started=clock[0], **kwargs)
    assert parts == [0, 1] and progress.pending_validation_step is None
    assert run.progress.step == 7 and progress.next_batch_index == 0
    assert logs[-1]["phase_complete"]
