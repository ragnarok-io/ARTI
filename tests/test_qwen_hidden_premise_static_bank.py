from __future__ import annotations

import torch
from safetensors import safe_open
from safetensors.torch import load_file
from torch import nn

from benchmarks.run_qwen_hidden_premise_static_bank import StaticBankLowRankLinear
from benchmarks.run_qwen_dialogue_recall_recurrence_curriculum import (
    LowRankBankRuntime,
    ValuesOnlyLowRankBankRuntime,
    load_affine_controller_runtime_state,
    load_values_only_affine_runtime_state,
    make_affine_controller_bank_runtime,
    overlay_affine_controller_runtime_state,
)
from benchmarks.run_qwen_layered_dialogue_recall_transition import save_runtime
from arti._recall_state import AffineRecallValueUpdater, RecallValueUpdater


def test_one_static_bank_is_exact_low_rank_adapter() -> None:
    torch.manual_seed(7)
    base = nn.Linear(5, 3, bias=False)
    layer = StaticBankLowRankLinear(base, rank=2)
    x = torch.randn(4, 5)

    torch.testing.assert_close(layer(x), base(x), rtol=0.0, atol=0.0)
    with torch.no_grad():
        layer.value_up.normal_()
    expected = base(x) + torch.nn.functional.linear(
        torch.nn.functional.linear(x, layer.value_down[0]),
        layer.value_up[0],
    )

    torch.testing.assert_close(layer(x), expected)


def test_multiple_static_banks_only_train_values() -> None:
    layer = StaticBankLowRankLinear(
        nn.Linear(6, 4),
        rank=3,
        bank_count=4,
        query_seed=19,
    )
    trainable = {
        name
        for name, parameter in layer.named_parameters()
        if parameter.requires_grad
    }

    assert trainable == {"value_down", "value_up"}
    assert "fixed_query" in dict(layer.named_buffers())
    output = layer(torch.randn(2, 5, 6))
    assert output.shape == (2, 5, 4)
    output.square().mean().backward()
    assert layer.value_down.grad is not None
    assert layer.value_up.grad is not None


def test_dynamic_low_rank_runtime_starts_as_noop_and_reads_bank_values() -> None:
    class Attention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(4, 8, bias=False)
            self.v_proj = nn.Linear(4, 4, bias=False)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = Attention()

    layer = Layer()
    updater = RecallValueUpdater(
        4,
        10,
        workspace_dim=8,
        depth=1,
        interface_slots=2,
        recall_slots=2,
        recall_steps=0,
    )
    runtime = LowRankBankRuntime(
        (0,),
        (updater,),
        rank=2,
        hidden_dim=4,
        q_output_dim=8,
        v_output_dim=4,
        seed=31,
    )
    original_q = layer.self_attn.q_proj
    x = torch.randn(3, 5, 4)
    expected = original_q(x)
    runtime.install((layer,))
    values = runtime.initial_values(3, x)
    mask = torch.ones(3, 5, dtype=torch.bool)
    with runtime.use(values, mask):
        torch.testing.assert_close(layer.self_attn.q_proj(x), expected)

    changed = values.clone()
    changed[:, :, 2:6].fill_(0.25)
    with runtime.use(changed, mask):
        assert not torch.equal(layer.self_attn.q_proj(x), expected)
    runtime.close()
    assert layer.self_attn.q_proj is original_q


def test_values_only_runtime_keeps_features_fixed_and_state_zero() -> None:
    class Attention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = nn.Linear(4, 8, bias=False)
            self.v_proj = nn.Linear(4, 4, bias=False)

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = Attention()

    layer = Layer()
    updater = AffineRecallValueUpdater(4, 6, workspace_dim=8)
    runtime = ValuesOnlyLowRankBankRuntime(
        (0,),
        (updater,),
        rank=2,
        hidden_dim=4,
        q_output_dim=8,
        v_output_dim=4,
        seed=31,
    )
    original_q = layer.self_attn.q_proj
    x = torch.randn(3, 5, 4)
    expected = original_q(x)
    runtime.install((layer,))
    values = runtime.initial_values(3, x)
    mask = torch.ones(3, 5, dtype=torch.bool)

    assert torch.count_nonzero(values) == 0
    with runtime.use(values, mask):
        torch.testing.assert_close(layer.self_attn.q_proj(x), expected)
    changed = values.clone()
    changed[:, :, :4].fill_(0.25)
    with runtime.use(changed, mask):
        assert not torch.equal(layer.self_attn.q_proj(x), expected)
    assert not runtime.q_feature_basis.requires_grad
    assert not runtime.v_feature_basis.requires_grad
    assert runtime.state_semantics == "values_only"
    runtime.close()
    assert layer.self_attn.q_proj is original_q


def test_values_only_runtime_updates_match_random_chunking() -> None:
    updater = AffineRecallValueUpdater(4, 6, workspace_dim=8)
    with torch.no_grad():
        updater.retention_weight.normal_(std=0.1)
        updater.value_weight.normal_(std=0.1)
    runtime = ValuesOnlyLowRankBankRuntime(
        (0,),
        (updater,),
        rank=2,
        hidden_dim=4,
        q_output_dim=8,
        v_output_dim=4,
        seed=43,
    )
    traces = torch.randn(2, 1, 7, 4)
    initial = runtime.initial_values(2, traces)

    whole = runtime.update_values(traces, initial)
    first = runtime.update_values(traces[:, :, :3], initial)
    chunked = runtime.update_values(traces[:, :, 3:], first)
    reversed_state = runtime.update_values(traces.flip(2), initial)

    torch.testing.assert_close(chunked, whole)
    assert not torch.equal(reversed_state, whole)


def test_values_only_runtime_artifact_round_trip_is_semantically_strict(tmp_path) -> None:
    source_updater = AffineRecallValueUpdater(4, 6, workspace_dim=8)
    source = ValuesOnlyLowRankBankRuntime(
        (0,),
        (source_updater,),
        rank=2,
        hidden_dim=4,
        q_output_dim=8,
        v_output_dim=4,
        seed=43,
    )
    with torch.no_grad():
        source_updater.value_weight.normal_(std=0.1)
    path = tmp_path / "values-only.recall.arti.st"
    save_runtime(
        path,
        source,
        source.initial_values(1, source.q_feature_basis),
        model_id="tiny",
        formula_reference=source.formula_reference,
    )
    restored = ValuesOnlyLowRankBankRuntime(
        (0,),
        (AffineRecallValueUpdater(4, 6, workspace_dim=8),),
        rank=2,
        hidden_dim=4,
        q_output_dim=8,
        v_output_dim=4,
        seed=99,
    )

    load_values_only_affine_runtime_state(restored, load_file(str(path)))

    torch.testing.assert_close(restored.q_feature_basis, source.q_feature_basis)
    torch.testing.assert_close(restored.v_feature_basis, source.v_feature_basis)
    for actual, expected in zip(
        restored.updaters[0].state_dict().values(),
        source.updaters[0].state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)
    with safe_open(path, framework="pt") as handle:
        metadata = handle.metadata()
    assert metadata["state_semantics"] == "values_only"
    assert metadata["transition_semantics"] == "affine_monoid"
    assert metadata["fixed_feature_basis"] == "true"


def test_affine_controller_runtime_is_zero_then_reads_written_values() -> None:
    class Layer(nn.Module):
        def forward(self, x):
            return x * 2

    layer = Layer()
    runtime = make_affine_controller_bank_runtime(
        layer_indices=(0,),
        hidden_dim=4,
        slots=3,
        workspace_dim=8,
        control_dim=4,
        heads=2,
        seed=53,
        device=torch.device("cpu"),
    )
    x = torch.randn(2, 5, 4)
    mask = torch.ones(2, 5, dtype=torch.bool)
    zero = runtime.initial_values(2, x)
    runtime.install((layer,))
    with runtime.use(zero, mask):
        torch.testing.assert_close(layer(x), x * 2, rtol=0, atol=0)
    with torch.no_grad():
        runtime.controllers[0].control_scale.fill_(1.0)
    written = torch.randn_like(zero)
    with runtime.use(written, mask):
        assert not torch.equal(layer(x), x * 2)
    runtime.close()


def test_affine_controller_runtime_artifact_round_trip(tmp_path) -> None:
    source = make_affine_controller_bank_runtime(
        layer_indices=(0,),
        hidden_dim=4,
        slots=3,
        workspace_dim=8,
        control_dim=4,
        heads=2,
        seed=59,
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        source.controllers[0].control_scale.fill_(0.75)
    path = tmp_path / "controller.recall.arti.st"
    save_runtime(
        path,
        source,
        source.initial_values(1, source.fields[0].bank),
        model_id="tiny",
        formula_reference=source.formula_reference,
    )
    restored = make_affine_controller_bank_runtime(
        layer_indices=(0,),
        hidden_dim=4,
        slots=3,
        workspace_dim=8,
        control_dim=4,
        heads=2,
        seed=61,
        device=torch.device("cpu"),
    )

    load_affine_controller_runtime_state(restored, load_file(str(path)))

    for actual, expected in zip(
        restored.state_dict().values(),
        source.state_dict().values(),
        strict=True,
    ):
        torch.testing.assert_close(actual, expected)
    with safe_open(path, framework="pt") as handle:
        metadata = handle.metadata()
    assert metadata["control_backend"] == "decoupled_cross_attention"
    assert metadata["fixed_feature_basis"] == "false"


def test_affine_controller_overlay_preserves_matching_layer_only() -> None:
    source = make_affine_controller_bank_runtime(
        layer_indices=(14,),
        hidden_dim=4,
        slots=3,
        workspace_dim=8,
        control_dim=4,
        heads=2,
        seed=73,
        device=torch.device("cpu"),
    )
    tensors = {
        f"sites.0.updater.{name}": tensor.clone()
        for name, tensor in source.updaters[0].state_dict().items()
    }
    tensors.update(
        {
            f"sites.0.controller.{name}": tensor.clone()
            for name, tensor in source.controllers[0].state_dict().items()
        }
    )
    destination = make_affine_controller_bank_runtime(
        layer_indices=(4, 14, 25),
        hidden_dim=4,
        slots=3,
        workspace_dim=8,
        control_dim=4,
        heads=2,
        seed=79,
        device=torch.device("cpu"),
    )

    mapping = overlay_affine_controller_runtime_state(
        destination,
        tensors,
        source_layers=(14,),
    )

    assert mapping == {0: 1}
    torch.testing.assert_close(
        destination.controllers[1].control_scale,
        source.controllers[0].control_scale,
    )
    assert destination.controllers[0].control_scale == 0
    assert destination.controllers[2].control_scale == 0
