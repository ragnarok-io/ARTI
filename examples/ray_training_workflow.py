"""Offline Ray Train V2 checkpoint handoff across independent CPU processes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import uuid

os.environ.setdefault("RAY_TRAIN_V2_ENABLED", "1")
os.environ.setdefault("RAY_USAGE_STATS_ENABLED", "0")

import torch  # noqa: E402
import ray  # noqa: E402
from ray import train  # noqa: E402
from ray.train.torch import TorchTrainer  # noqa: E402

import arti  # noqa: E402


class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        pattern = arti.mechanisms.TensorViewPattern(
            min_rank=2, max_rank=2, allowed_axis_roles=("batch", "feature")
        )
        resources = tuple(
            arti.mechanisms.TensorResource(
                arti.mechanisms.TensorResourceSpec(name, pattern),
                arti.mechanisms.TensorView.from_tensor(
                    torch.zeros(1, 4),
                    axis_names=("batch", "feature"),
                    axis_roles=("batch", "feature"),
                ),
            )
            for name in ("inputs", "outputs")
        )
        graph = arti.mechanisms.ProgramGraph(
            resources,
            (
                arti.mechanisms.Connection(
                    "adapt",
                    arti.mechanisms.ResourcePort("inputs"),
                    arti.mechanisms.ResourcePort("outputs"),
                    transfer=arti.mechanisms.LearnableAffineTransfer(gain=1.0),
                ),
            ),
            programs={"task": ("adapt",)},
        )
        self.layer = arti.ARTILayer(
            graph=graph,
            graph_program_id="task",
            graph_input_resource_id="inputs",
            graph_output_resource_id="outputs",
            axis_names=("batch", "feature"),
            axis_roles=("batch", "feature"),
        )

    def forward(self, inputs, labels):
        value = self.layer(inputs)
        return {"prediction": value, "loss": (value - labels).square().mean()}


def objective(model, batch):
    labels = batch["labels"] + torch.rand_like(batch["labels"]) * random.random()
    return model(batch["inputs"], labels)["loss"]


def session():
    current = arti.trainer(
        Model(), learning_rate=0.02, gradient_accumulation_steps=2, objective=objective
    )
    current.scheduler = torch.optim.lr_scheduler.StepLR(current.optimizer, step_size=2, gamma=0.8)
    torch.manual_seed(57)
    random.seed(57)
    return current


def data():
    return [
        {"inputs": torch.full((3, 4), value), "labels": torch.full((3, 4), value * 2)}
        for value in (1.0, 2.0, 3.0)
    ]


def run_stage(root, stage):
    current = session()

    def train_fn(_config):
        initial = 0
        if stage == "resume":
            first = json.loads((root / "first.json").read_text())
            checkpoint = train.Checkpoint.from_directory(str(root / first["checkpoint"]))
            arti.restore_ray_checkpoint(
                current, directory=root / "materialized", checkpoint=checkpoint
            )
            initial = current.global_step
        result = current.fit(
            data(),
            steps=2 if stage == "first" else 6 - initial,
            run=arti.ARTITrainingRunConfig(
                output_dir=root / f"{stage}-{uuid.uuid4().hex}", checkpoint_steps=2
            ),
            on_checkpoint=arti.RayTrainCheckpointHandler(checkpoint_dir=root / "handoff" / stage),
        )
        return {
            "initial_step": initial,
            "global_step": result.steps,
            "micro_batches": current.micro_batches_seen,
        }

    outcome = TorchTrainer(
        train_fn,
        scaling_config=train.ScalingConfig(num_workers=0),
        run_config=train.RunConfig(name=f"arti-{stage}"),
    ).fit()
    assert not ray.is_initialized()
    receipt = {
        **outcome.return_value,
        "pid": os.getpid(),
        "checkpoint": Path(outcome.checkpoint.path).resolve().relative_to(root).as_posix(),
        "ray_cluster_started": False,
    }
    (root / f"{stage}.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def compare(root):
    receipts = {
        name: json.loads((root / f"{name}.json").read_text())
        for name in ("continuous", "first", "resume")
    }
    full, resumed = session(), session()
    for name, current in (("continuous", full), ("resume", resumed)):
        checkpoint = train.Checkpoint.from_directory(str(root / receipts[name]["checkpoint"]))
        arti.restore_ray_checkpoint(
            current, directory=root / f"verify-{uuid.uuid4().hex}", checkpoint=checkpoint
        )

    def equal(left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            assert left.keys() == right.keys()
            for key in left:
                equal(left[key], right[key])
        elif isinstance(left, (tuple, list)):
            assert len(left) == len(right)
            for a, b in zip(left, right):
                equal(a, b)
        else:
            assert left == right

    equal(full.binding.state_module().state_dict(), resumed.binding.state_module().state_dict())
    equal(full.optimizer.state_dict(), resumed.optimizer.state_dict())
    equal(full.scheduler.state_dict(), resumed.scheduler.state_dict())
    assert full.loss_history == resumed.loss_history
    assert full.global_step == resumed.global_step == 6
    assert full.micro_batches_seen == resumed.micro_batches_seen == 12
    assert receipts["first"]["pid"] != receipts["resume"]["pid"]
    result = {
        "profile": "ray_train_v2_local_cpu",
        "global_step": 6,
        "micro_batches": 12,
        "fresh_process_restore": True,
        "state_optimizer_scheduler_rng_data_parity": True,
        "ray_cluster_started": False,
        "automatic_worker_retry_verified": False,
    }
    (root / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(".tmp/ray-workflow"))
    parser.add_argument(
        "--stage", choices=("all", "continuous", "first", "resume", "check"), default="all"
    )
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if args.stage == "all":
        for stage in ("continuous", "first", "resume"):
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--output",
                    str(root),
                    "--stage",
                    stage,
                ],
                check=True,
            )
        receipt = compare(root)
    else:
        receipt = compare(root) if args.stage == "check" else run_stage(root, args.stage)
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
