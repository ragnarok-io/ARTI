from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

import arti


def backbone() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(8, 8),
        nn.GELU(),
        nn.Linear(8, 8),
        nn.GELU(),
        nn.Linear(8, 8),
    )


def test_attach_freezes_backbone_and_keeps_only_layer_recalls_trainable() -> None:
    model = arti.LayeredRecallModel.attach(backbone(), ("0", "2", "4"), rank=3, slots=4)
    output = model(torch.randn(2, 5, 8))

    assert output.shape == (2, 5, 8)
    assert set(model.wrappers) == {"0", "2", "4"}
    assert all(not parameter.requires_grad for wrapper in model.wrappers.values() for parameter in wrapper.base.parameters())
    assert all(parameter.requires_grad for parameter in model.recall_parameters())


def test_local_trajectory_loss_uses_clean_hidden_targets_and_backpropagates_only_recall() -> None:
    model = arti.LayeredRecallModel.attach(backbone(), ("0", "2", "4"), rank=3, slots=4)
    clean = torch.randn(3, 6, 8)
    corrupt = clean.clone()
    corrupt[:, 2:4] = 0
    unseen = torch.randn_like(clean) + 8.0
    mask = torch.ones(3, 6, dtype=torch.bool)

    result = arti.layered_recall_trajectory_loss(model, clean, corrupt, mask=mask, unseen_inputs=unseen)
    result.loss.backward()

    assert set(result.per_layer_mse) == {"0", "2", "4"}
    assert set(result.per_layer_raw_delta_norm) == {"0", "2", "4"}
    assert set(result.per_layer_survival) == {"0", "2", "4"}
    assert result.repair_loss.item() >= 0
    assert result.unseen_loss.item() >= 0
    assert all(parameter.grad is not None for parameter in model.recall_parameters())
    assert all(parameter.grad is None for wrapper in model.wrappers.values() for parameter in wrapper.base.parameters())


def test_local_trajectory_loss_accepts_per_layer_baseline_scales() -> None:
    model = arti.LayeredRecallModel.attach(backbone(), ("0", "2"), rank=2, slots=3)
    clean = torch.randn(2, 4, 8)
    corrupt = clean.clone()
    corrupt[:, 1] = 0
    raw = arti.layered_recall_trajectory_loss(model, clean, corrupt)
    scales = {path: value.detach() for path, value in raw.per_layer_mse.items()}
    normalized = arti.layered_recall_trajectory_loss(model, clean, corrupt, layer_scales=scales)

    assert torch.allclose(normalized.repair_loss, torch.ones_like(normalized.repair_loss), atol=1e-5)


def test_layer_recall_strictly_applies_half_to_candidate_delta() -> None:
    with_half = arti.LayerRecall(8, rank=3, slots=4, use_half=True, recognition_mode="none")
    without_half = arti.LayerRecall(8, rank=3, slots=4, use_half=False, recognition_mode="none")
    without_half.load_state_dict(with_half.state_dict(), strict=False)
    x = torch.randn(2, 5, 8)
    half_delta = with_half(x)
    raw_delta = without_half(x)

    assert half_delta.shape == raw_delta.shape
    assert torch.all(half_delta.abs() <= raw_delta.abs() + 1e-7)


def test_tuple_output_transformer_layer_contract_is_preserved() -> None:
    class TupleBlock(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(8, 8)

        def forward(self, x):
            return self.proj(x), torch.ones((), device=x.device)

    class Host(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = TupleBlock()

        def forward(self, x):
            return self.block(x)

    model = arti.LayeredRecallModel.attach(Host(), ("block",), sample_batch=torch.randn(2, 4, 8), rank=2, slots=3)
    output = model(torch.randn(2, 4, 8))

    assert isinstance(output, tuple)
    assert output[0].shape == (2, 4, 8)
    assert output[1].shape == ()


def test_layer_artifacts_are_path_bound_loadable_and_concatenable(tmp_path) -> None:
    model = arti.LayeredRecallModel.attach(backbone(), ("0", "2", "4"), rank=2, slots=3)
    first = tmp_path / "early-a.recall.arti.st"
    second = tmp_path / "early-b.recall.arti.st"
    model.export_layer("0", first)
    with torch.no_grad():
        model.wrappers["0"].recall.bank.add_(0.25)
    model.export_layer("0", second)

    restored = arti.LayeredRecallModel.attach(backbone(), ("0", "2", "4"), rank=2, slots=3)
    # Structure fingerprints include the same layer paths and branch shapes.
    restored.load_layer("0", first)
    merged = restored.concat_layer("0", {"a": first, "b": second})

    assert merged.bank.shape == (6, 2)
    assert merged.slots == 6
    with pytest.raises(ValueError, match="belongs to layer"):
        restored.load_layer("2", first)


def test_complete_recall_lines_append_without_overwriting_existing_line(tmp_path) -> None:
    torch.manual_seed(4)
    first_model = arti.LayeredRecallModel.attach(backbone(), ("0",), rank=2, slots=3)
    second_model = copy.deepcopy(first_model)
    with torch.no_grad():
        second_model.wrappers["0"].recall.query.weight.add_(0.4)
        second_model.wrappers["0"].recall.bank.mul_(1.3)
    artifact = tmp_path / "second.recall.arti.st"
    second_model.export_layer("0", artifact)
    original = copy.deepcopy(first_model.wrappers["0"].recall.state_dict())

    stack = first_model.append_layer_artifacts("0", {"second": artifact})

    assert isinstance(stack, arti.LayerRecallStack)
    assert len(stack.branches) == 2
    for name, tensor in original.items():
        assert torch.equal(stack.branches[0].state_dict()[name], tensor)
    assert not torch.equal(stack.branches[0].query.weight, stack.branches[1].query.weight)
    with stack.enabled_lines((0,)):
        old_only = stack(torch.randn(2, 4, 8))
    assert old_only.shape == (2, 4, 8)


def test_attach_can_infer_dimensions_from_runtime_scan() -> None:
    host = backbone()
    model = arti.LayeredRecallModel.attach(host, ("0", "2", "4"), sample_batch=torch.randn(2, 8), rank=2, slots=3)
    assert [wrapper.recall.dim for wrapper in model.wrappers.values()] == [8, 8, 8]


def test_public_torch_namespace_matches_root() -> None:
    assert arti.torch.LayerRecall is arti.LayerRecall
    assert arti.torch.LayeredRecallModel is arti.LayeredRecallModel
    assert arti.torch.LayeredRecallConfig is arti.LayeredRecallConfig
    assert arti.torch.calibrate_layered_recall is arti.calibrate_layered_recall


def test_stable_config_calibration_and_normalized_loss() -> None:
    config = arti.LayeredRecallConfig(layer_paths=("0", "2", "4"), rank=2, slots=3)
    model = arti.LayeredRecallModel.from_config(backbone(), config)
    clean = torch.randn(3, 5, 8)
    corrupt = clean.clone()
    corrupt[:, 1:3] = 0
    calibration = model.calibrate(clean, corrupt)
    result = arti.layered_recall_trajectory_loss(model, clean, corrupt, calibration=calibration)

    assert set(calibration.scales) == set(config.layer_paths)
    assert all(value.item() > 0 for value in calibration.scales.values())
    assert result.loss.isfinite()


def test_open_ended_layer_specs_allow_independent_sizes_and_features() -> None:
    config = arti.LayeredRecallConfig(
        layers=(
            arti.LayerRecallSpec("0", dim=8, rank=1, slots=2, use_half=False, recognition_mode="none"),
            arti.LayerRecallSpec("2", dim=8, rank=3, slots=5, use_half=True, recognition_mode="explicit"),
            arti.LayerRecallSpec("4", dim=8, rank=2, slots=7, recognition_mode="alignment"),
        )
    )
    model = arti.LayeredRecallModel.from_config(backbone(), config)

    assert config.paths == ("0", "2", "4")
    assert [model.wrappers[path].recall.rank for path in config.paths] == [1, 3, 2]
    assert [model.wrappers[path].recall.slots for path in config.paths] == [2, 5, 7]
    assert [model.wrappers[path].recall.use_half for path in config.paths] == [False, True, True]
    assert [model.wrappers[path].recall.recognition_mode for path in config.paths] == ["none", "explicit", "alignment"]


def test_repeated_lines_can_share_one_physical_layer() -> None:
    config = arti.LayeredRecallConfig(
        layers=(arti.LayerRecallSpec("2", dim=8, rank=2, slots=3, copies=3, combine="mean"),)
    )
    model = arti.LayeredRecallModel.from_config(backbone(), config)
    stack = model.wrappers["2"].recall
    output = model(torch.randn(2, 4, 8))

    assert isinstance(stack, arti.LayerRecallStack)
    assert len(stack.branches) == 3
    assert stack.combine == "mean"
    assert output.shape == (2, 4, 8)
    with stack.enabled_lines((0,)):
        assert stack(torch.randn(2, 4, 8)).shape == (2, 4, 8)
    with stack.enabled_lines(()) as disabled:
        assert torch.count_nonzero(disabled(torch.randn(2, 4, 8))) == 0
    assert stack._line_enabled == [True, True, True]
    with pytest.raises(ValueError, match="single Recall line"):
        model.concat_layer("2", {})


def test_layer_enable_contexts_restore_state_and_validate_paths() -> None:
    model = arti.LayeredRecallModel.attach(backbone(), ("0", "2", "4"), rank=2, slots=3)
    model.set_enabled(False, paths=("2",))
    before = {path: wrapper.enabled for path, wrapper in model.wrappers.items()}
    with model.enabled_layers(("4",)):
        assert {path: wrapper.enabled for path, wrapper in model.wrappers.items()} == {"0": False, "2": False, "4": True}
    assert {path: wrapper.enabled for path, wrapper in model.wrappers.items()} == before
    with model.disabled():
        assert not any(wrapper.enabled for wrapper in model.wrappers.values())
    with pytest.raises(ValueError, match="unknown Recall layer paths"):
        model.set_enabled(True, paths=("missing",))


def test_layer_diagnostics_exposes_latest_trace_components() -> None:
    model = arti.LayeredRecallModel.attach(backbone(), ("0", "2"), rank=2, slots=3)
    for wrapper in model.wrappers.values():
        wrapper.capture = True
    model(torch.randn(2, 4, 8))
    report = model.diagnostics()

    assert set(report) == {"0", "2"}
    assert set(report["0"]) == {"raw_delta", "delta", "recognition", "survival"}


def test_layered_recall_delegates_generate() -> None:
    class Generative(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = nn.Linear(8, 8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)

        def generate(self, x: torch.Tensor, *, scale: float = 1.0) -> torch.Tensor:
            return self.forward(x) * scale

    wrapped = arti.LayeredRecallModel.attach(Generative(), ("block",), dims={"block": 8}, rank=2, slots=2)
    assert wrapped.generate(torch.randn(2, 8), scale=2.0).shape == (2, 8)


def test_qwen_style_nested_layer_paths_can_be_attached() -> None:
    class Decoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(6)])

        def forward(self, x):
            for layer in self.layers:
                x = torch.tanh(layer(x))
            return x

    class QwenLike(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = Decoder()

        def forward(self, x):
            return self.model(x)

    layered = arti.LayeredRecallModel.attach(
        QwenLike(),
        ("model.layers.1", "model.layers.3", "model.layers.5"),
        rank=2,
        slots=3,
    )

    assert layered(torch.randn(2, 4, 8)).shape == (2, 4, 8)
    assert all(not parameter.requires_grad for wrapper in layered.wrappers.values() for parameter in wrapper.base.parameters())
