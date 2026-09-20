from __future__ import annotations

import json

from benchmarks._local_metrics import LocalMetricWriter


def test_local_metric_writer_emits_jsonl_and_tensorboard_without_network(tmp_path) -> None:
    run_dir = tmp_path / "run"
    with LocalMetricWriter(run_dir, config={"condition": "one", "path": tmp_path}) as writer:
        writer.log(0, {"train/loss": 1.0})
        writer.log(1, {"train/loss": 0.5, "system/steps_per_second": 2.0})

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (run_dir / "metrics.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    events = list((run_dir / "tensorboard").glob("events.out.tfevents.*"))

    assert config["condition"] == "one"
    assert [record["step"] for record in records] == [0, 1]
    assert records[1]["metrics"]["train/loss"] == 0.5
    assert len(events) == 1
