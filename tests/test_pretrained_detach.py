from __future__ import annotations

import torch
import torch.nn as nn

import arti


def test_pretrained_workflow_detach_restores_host_structure_and_trainability() -> None:
    torch.manual_seed(23)
    model = nn.Sequential(nn.Linear(4, 4), nn.Tanh())
    sample = torch.randn(2, 4)
    original_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }

    workflow = arti.pretrained(model, provider="torch")
    workflow.scan(sample).plan(
        where="0",
        scale="tiny",
        max_adapters=1,
        training={"engine": "torch", "steps": 1},
    )
    workflow.apply()
    assert workflow.doctor()["applied"] is True
    assert model[0].__class__.__name__ == "ARTIAdapterWrapper"

    assert workflow.detach() is model
    assert isinstance(model[0], nn.Linear)
    assert workflow.doctor()["applied"] is False
    assert all(parameter.requires_grad for parameter in model.parameters())
    for name, value in model.state_dict().items():
        assert torch.equal(value, original_state[name])
