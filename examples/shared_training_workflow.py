"""Three developer entrances using one training session, entirely on CPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn

import arti
from attachment_training_workflow import TaskModel, layer


class NativeProgram(nn.Module):
    def __init__(self):
        super().__init__()
        self.program = layer()

    def forward(self, inputs, labels):
        return {"loss": (self.program(inputs) - labels).square().mean()}


def main(output: Path):
    torch.manual_seed(8)
    inputs = torch.randn(4, 4)
    batches = [{"inputs": inputs, "labels": 2 * inputs}]
    native = arti.trainer(NativeProgram(), learning_rate=0.02, gradient_accumulation_steps=2)
    attached_model = arti.ARTI.attach(TaskModel(), layer(), layers="features")
    attached = attached_model.arti.trainer(learning_rate=0.02, gradient_accumulation_steps=2)
    provider = arti.from_pretrained(TaskModel(), provider="torch", sample_batch={"inputs": inputs}, where="features", max_adapters=1, training={"steps": 4, "gradient_accumulation_steps": 2})
    rows = []
    for name, session in (("native", native), ("attachment", attached), ("pretrained", provider.trainer())):
        assert session.check(batches[0])["ok"]
        result = session.fit(batches, steps=4, run=arti.ARTITrainingRunConfig(output_dir=output / name))
        rows.append({"entrance": name, "session_class": type(session).__name__, "optimizer_steps": result.steps, "microbatches": session.micro_batches_seen, "checkpoint": str(result.checkpoint_path)})
        assert result.checkpoint_path.exists()
        assert result.steps == 4 and session.micro_batches_seen == 8
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path(".arti-runs/shared-workflow"))
    main(parser.parse_args().output)
