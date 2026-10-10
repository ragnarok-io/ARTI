"""CPU workflow: train a real resource graph, resume, evaluate and reload.

Run: python examples/attachment_training_workflow.py --output .tmp/workflow-example
This validates SDK lifecycle, not large-model quality or CUDA speed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

import arti
from arti import mechanisms as m


class TaskModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Linear(4, 4, bias=False)
        with torch.no_grad():
            self.features.weight.copy_(torch.eye(4))

    def forward(self, inputs, labels=None):
        prediction = self.features(inputs)
        loss = None if labels is None else (prediction - labels).square().mean()
        return {"prediction": prediction, "loss": loss}


def layer():
    pattern = m.TensorViewPattern(min_rank=2, max_rank=2, allowed_axis_roles=("batch", "feature"))
    resources = tuple(
        m.TensorResource(
            m.TensorResourceSpec(name, pattern),
            m.TensorView.from_tensor(torch.zeros(1, 4), axis_names=("batch", "feature"), axis_roles=("batch", "feature")),
        )
        for name in ("inputs", "outputs")
    )
    graph = m.ProgramGraph(
        resources,
        (m.Connection("adapt", m.ResourcePort("inputs"), m.ResourcePort("outputs"), transfer=m.LearnableAffineTransfer()),),
        programs={"task": ("adapt",)},
    )
    return arti.ARTILayer(graph=graph, graph_program_id="task", graph_input_resource_id="inputs", graph_output_resource_id="outputs", axis_names=("batch", "feature"), axis_roles=("batch", "feature"))


def main(output: Path):
    torch.manual_seed(17)
    generator = torch.Generator().manual_seed(18)
    train = [{"inputs": x, "labels": 2 * x} for x in (torch.randn(8, 4, generator=generator) for _ in range(4))]
    validation = [{"inputs": x, "labels": 2 * x} for x in (torch.randn(5, 4, generator=generator) for _ in range(2))]
    model = arti.ARTI.attach(TaskModel(), layer(), layers="features")
    session = model.arti.trainer(learning_rate=0.05)
    assert session.check(train[0])["ok"]
    before = session.evaluate(validation).loss
    result = session.fit(train, steps=32, run=arti.ARTITrainingRunConfig(output_dir=output / "train", checkpoint_steps=16, evaluation_steps=16, metadata={"task": "tensor-regression", "dataset": "workflow-example-v1"}), eval_data=validation)
    restored = arti.ARTI.load(TaskModel(), result.checkpoint_path, layer=layer())
    resumed = restored.arti.trainer(learning_rate=0.05, resume_from_checkpoint=result.checkpoint_path)
    continuation = resumed.fit(train, steps=8, run=arti.ARTITrainingRunConfig(output_dir=output / "resume"))
    after = resumed.evaluate(validation).loss
    assert after < before
    asset = restored.arti.save_pretrained(output / "model", base_model="example/fixed-identity-base", training_session=resumed)
    reloaded = arti.ARTI.from_pretrained(asset.directory, model=TaskModel(), layer=layer())
    torch.testing.assert_close(reloaded(validation[0]["inputs"])["prediction"], restored(validation[0]["inputs"])["prediction"])
    print(json.dumps({"initial_validation_loss": before, "final_validation_loss": after, "optimizer_steps": continuation.steps, "artifact": str(asset.artifact_path), "reloaded": True}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(".arti-runs/workflow-example"))
    main(parser.parse_args().output)
