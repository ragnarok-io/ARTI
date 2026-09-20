from __future__ import annotations

import torch

from benchmarks.federal_associative_fitting import (
    ExperimentConfig,
    FederatedARTIFitter,
    TransformerFitter,
    make_dataset,
    trainable_parameters,
)


def tiny_config() -> ExperimentConfig:
    return ExperimentConfig(
        groups=2,
        slots_per_bank=3,
        sequence=4,
        dim=8,
        root_dim=2,
        leaf_dim=3,
        transformer_width=16,
        transformer_layers=1,
        transformer_heads=2,
        transformer_ffn=32,
        steps=10,
        batch_size=6,
        eval_size=6,
        eval_interval=5,
        max_seconds=10.0,
    )


def test_federal_associative_bank_routes_and_reads_formula_value() -> None:
    config = tiny_config()
    dataset = make_dataset(config, torch.device("cpu"))
    model = FederatedARTIFitter(config, dataset)

    group, slot, flat = model.address(dataset.x)
    assert torch.equal(group, dataset.group_index)
    assert torch.equal(slot, dataset.slot_index)
    assert torch.equal(
        flat,
        dataset.group_index * config.slots_per_bank + dataset.slot_index,
    )

    with torch.no_grad():
        model.value.copy_(dataset.target)
    output, trace = model(dataset.x, return_trace=True)
    torch.testing.assert_close(output, dataset.target)
    assert trace.atom_refs == ("arti/formula-atom-gather@1",)


def test_federal_associative_value_bank_learns_independent_targets() -> None:
    config = tiny_config()
    dataset = make_dataset(config, torch.device("cpu"))
    model = FederatedARTIFitter(config, dataset)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.2, weight_decay=0.0)

    initial = torch.nn.functional.mse_loss(model(dataset.x), dataset.target).item()
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(dataset.x), dataset.target)
        loss.backward()
        optimizer.step()
    final = torch.nn.functional.mse_loss(model(dataset.x), dataset.target).item()

    assert final < initial * 0.1
    assert model.root_keys.grad is None
    assert model.leaf_keys.grad is None


def test_transformer_control_matches_tensor_shape_and_has_trainable_capacity() -> None:
    config = tiny_config()
    dataset = make_dataset(config, torch.device("cpu"))
    model = TransformerFitter(config)

    assert model(dataset.x).shape == dataset.target.shape
    torch.testing.assert_close(model(dataset.x), torch.zeros_like(dataset.target))
    assert trainable_parameters(model) > 0

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    initial = torch.nn.functional.mse_loss(model(dataset.x), dataset.target).item()
    for _ in range(100):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(dataset.x), dataset.target)
        loss.backward()
        optimizer.step()
    final = torch.nn.functional.mse_loss(model(dataset.x), dataset.target).item()
    assert final < initial * 0.1
