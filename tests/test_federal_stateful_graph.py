import torch
from torch import nn

from arti.alpha import (
    FederalRaggedShapeCompiler,
    FederalStatefulGraphCompiler,
)


class _SelfModifyingTransition(nn.Module):
    """Keep the data lane simple while changing the explicit Bank/effect lanes."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.5))

    def forward(self, value, bank_state, effect_state):
        next_value = value * self.scale
        next_bank = bank_state + value
        next_effect = effect_state + 1.0
        stop = next_value.abs().squeeze(-1) < 0.2
        return next_value, next_bank, next_effect, stop


class _DataDependentLength(nn.Module):
    def forward(self, values, lengths):
        grows = (values[:, 0, 0] >= 0).to(lengths.dtype)
        next_lengths = lengths + grows
        return values + 1.0, next_lengths


def test_stateful_graph_exports_mutable_bank_effect_and_open_refine(tmp_path):
    transition = _SelfModifyingTransition()
    value = torch.tensor([[2.0], [0.5]], requires_grad=True)
    bank = torch.zeros_like(value)
    effect = torch.zeros_like(value)
    graph = FederalStatefulGraphCompiler.compile(
        transition,
        example_value=value.detach(),
        example_bank_state=bank,
        example_effect_state=effect,
        source_ref="arti/federal-federation@1",
        source_snapshot_fingerprint="stateful-graph-test",
        max_steps=8,
    )

    original_bank = bank.clone()
    actual = graph(value, bank, effect, torch.tensor(8))
    output, next_bank, next_effect, done, steps = actual
    assert output.shape == value.shape
    assert done.tolist() == [True, True]
    assert steps.tolist() == [4, 2]
    torch.testing.assert_close(next_bank, torch.tensor([[3.75], [0.75]]))
    torch.testing.assert_close(next_effect, torch.tensor([[4.0], [2.0]]))
    torch.testing.assert_close(bank, original_bank)

    loss = output.square().sum() + next_bank.sum() + next_effect.sum()
    loss.backward()
    assert graph.transition.scale.grad is not None
    assert value.grad is not None

    exported = FederalStatefulGraphCompiler.export(
        graph,
        value.detach(),
        bank,
        effect,
        torch.tensor(8),
    )
    assert "while_loop" in str(exported.graph_module.graph)
    exported_result = exported.module()(value.detach(), bank, effect, torch.tensor(8))
    for actual_item, exported_item in zip(actual, exported_result, strict=True):
        torch.testing.assert_close(exported_item, actual_item.detach())
    artifact = tmp_path / "stateful-graph.pt2"
    torch.export.save(exported, artifact)
    restored = torch.export.load(artifact)
    restored_result = restored.module()(value.detach(), bank, effect, torch.tensor(8))
    for actual_item, restored_item in zip(actual, restored_result, strict=True):
        torch.testing.assert_close(restored_item, actual_item.detach())

    compiled = torch.compile(graph, backend="eager", fullgraph=True)
    compiled_result = compiled(value.detach(), bank, effect, torch.tensor(8))
    for actual_item, compiled_item in zip(actual, compiled_result, strict=True):
        torch.testing.assert_close(compiled_item, actual_item.detach())


def test_stateful_graph_manifest_exposes_host_bound_and_state_lanes():
    value = torch.zeros(2, 1)
    graph = FederalStatefulGraphCompiler.compile(
        _SelfModifyingTransition(),
        example_value=value,
        example_bank_state=value.clone(),
        example_effect_state=value.clone(),
        source_ref="arti/federal-federation@1",
        source_snapshot_fingerprint="manifest-test",
        max_steps=16,
    )
    manifest = graph.manifest.to_dict()
    assert manifest["max_steps"] == 16
    assert manifest["state_semantics"] == "explicit-bank-effect-input-output"
    assert manifest["refine_semantics"] == "while-loop-stop-or-host-bound"


def test_ragged_shape_graph_exports_data_dependent_logical_length():
    values = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            [[-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0], [-1.0, 0.0]],
        ]
    )
    lengths = torch.tensor([2, 2], dtype=torch.int64)
    graph = FederalRaggedShapeCompiler.compile(
        _DataDependentLength(),
        example_values=values,
        example_lengths=lengths,
        source_ref="arti/federal-federation@1",
        source_snapshot_fingerprint="ragged-shape-test",
    )

    output_values, output_lengths = graph(values, lengths)
    assert output_lengths.tolist() == [3, 2]
    torch.testing.assert_close(output_values[0, 3], torch.zeros(2))
    torch.testing.assert_close(output_values[0, 0], torch.tensor([2.0, 1.0]))
    torch.testing.assert_close(output_values[1, 2], torch.zeros(2))

    exported = FederalRaggedShapeCompiler.export(graph, values, lengths)
    exported_values, exported_lengths = exported.module()(values, lengths)
    torch.testing.assert_close(exported_values, output_values)
    torch.testing.assert_close(exported_lengths, output_lengths)

    compiled = torch.compile(graph, backend="eager", fullgraph=True)
    compiled_values, compiled_lengths = compiled(values, lengths)
    torch.testing.assert_close(compiled_values, output_values)
    torch.testing.assert_close(compiled_lengths, output_lengths)
