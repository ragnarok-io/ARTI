from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from arti import experimental


def backbone() -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(8, 8),
        nn.GELU(),
        nn.Linear(8, 8),
        nn.GELU(),
        nn.Linear(8, 8),
    )


def test_attach_freezes_backbone_and_keeps_only_layer_recalls_trainable() -> None:
    model = experimental.LayeredRecallModel.attach(backbone(), ("0", "2", "4"), rank=3, slots=4)
    output = model(torch.randn(2, 5, 8))

    assert output.shape == (2, 5, 8)
    assert set(model.wrappers) == {"0", "2", "4"}
    assert all(not parameter.requires_grad for wrapper in model.wrappers.values() for parameter in wrapper.base.parameters())
    assert all(parameter.requires_grad for parameter in model.recall_parameters())


def test_local_trajectory_loss_uses_clean_hidden_targets_and_backpropagates_only_recall() -> None:
    model = experimental.LayeredRecallModel.attach(backbone(), ("0", "2", "4"), rank=3, slots=4)
    clean = torch.randn(3, 6, 8)
    corrupt = clean.clone()
    corrupt[:, 2:4] = 0
    unseen = torch.randn_like(clean) + 8.0
    mask = torch.ones(3, 6, dtype=torch.bool)

    result = experimental.layered_recall_trajectory_loss(model, clean, corrupt, mask=mask, unseen_inputs=unseen)
    result.loss.backward()

    assert set(result.per_layer_mse) == {"0", "2", "4"}
    assert set(result.per_layer_raw_delta_norm) == {"0", "2", "4"}
    assert set(result.per_layer_survival) == {"0", "2", "4"}
    assert result.repair_loss.item() >= 0
    assert result.unseen_loss.item() >= 0
    assert all(parameter.grad is not None for parameter in model.recall_parameters())
    assert all(parameter.grad is None for wrapper in model.wrappers.values() for parameter in wrapper.base.parameters())


def test_local_trajectory_loss_accepts_per_layer_baseline_scales() -> None:
    model = experimental.LayeredRecallModel.attach(backbone(), ("0", "2"), rank=2, slots=3)
    for wrapper in model.wrappers.values():
        wrapper.recall.survival.stochastic = False
    clean = torch.randn(2, 4, 8)
    corrupt = clean.clone()
    corrupt[:, 1] = 0
    raw = experimental.layered_recall_trajectory_loss(model, clean, corrupt)
    scales = {path: value.detach() for path, value in raw.per_layer_mse.items()}
    normalized = experimental.layered_recall_trajectory_loss(model, clean, corrupt, layer_scales=scales)

    assert torch.allclose(normalized.repair_loss, torch.ones_like(normalized.repair_loss), atol=1e-5)


def test_layer_recall_strictly_applies_half_to_candidate_delta() -> None:
    with_half = experimental.LayerRecall(8, rank=3, slots=4, use_half=True, recognition_mode="none")
    with_half.survival.stochastic = False
    without_half = experimental.LayerRecall(8, rank=3, slots=4, use_half=False, recognition_mode="none")
    without_half.load_state_dict(with_half.state_dict(), strict=False)
    x = torch.randn(2, 5, 8)
    half_delta = with_half(x)
    raw_delta = without_half(x)

    assert half_delta.shape == raw_delta.shape
    assert torch.all(half_delta.abs() <= raw_delta.abs() + 1e-7)


def test_layer_recall_defaults_to_half_without_strength_or_recognition_gates() -> None:
    recall = experimental.LayerRecall(8, rank=3, slots=4)

    assert recall.use_half
    assert recall.recognition_mode == "none"
    assert not hasattr(recall, "gate")
    assert recall.recognizer is None
    assert set(dict(recall.named_parameters())) == {"bank", "query.weight", "emit.weight"}


def test_layer_recall_first_step_reaches_every_parameter() -> None:
    torch.manual_seed(5)
    recall = experimental.LayerRecall(8, rank=3, slots=4)
    loss = recall(torch.randn(2, 5, 8)).square().mean()

    loss.backward()

    assert all(parameter.grad is not None for parameter in recall.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in recall.parameters())


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

    model = experimental.LayeredRecallModel.attach(Host(), ("block",), sample_batch=torch.randn(2, 4, 8), rank=2, slots=3)
    output = model(torch.randn(2, 4, 8))

    assert isinstance(output, tuple)
    assert output[0].shape == (2, 4, 8)
    assert output[1].shape == ()


def test_layered_recall_attaches_to_arbitrary_spatial_tensor_boundary() -> None:
    class SpatialWarp(nn.Module):
        def forward(self, hidden: torch.Tensor):
            return {"aux": torch.tensor(1), "sample": hidden.tanh()}

    class Host(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.warp = SpatialWarp()

        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            return self.warp(hidden)["sample"]

    sample = torch.randn(2, 3, 4, 5, 6)
    model = experimental.LayeredRecallModel.attach(
        Host(), ("warp",), sample_batch=sample, rank=2, slots=3
    )
    output = model(sample)
    output.square().mean().backward()

    assert output.shape == sample.shape
    assert model.wrappers["warp"].recall.dim == 6
    assert any(parameter.grad is not None for parameter in model.recall_parameters())


def test_layered_recall_accepts_explicit_feature_axis_for_channel_first_boundary() -> None:
    class ChannelFirstWarp(nn.Module):
        def forward(self, hidden: torch.Tensor) -> torch.Tensor:
            return hidden.sin()

    host = nn.Sequential(ChannelFirstWarp())
    sample = torch.randn(2, 3, 4, 5)
    model = experimental.LayeredRecallModel.attach(
        host,
        ("0",),
        sample_batch=sample,
        feature_axis={"0": 1},
        rank=2,
        slots=3,
    )

    assert model(sample).shape == sample.shape
    assert model.wrappers["0"].recall.dim == 3


def test_attach_can_infer_dimensions_from_runtime_scan() -> None:
    host = backbone()
    model = experimental.LayeredRecallModel.attach(host, ("0", "2", "4"), sample_batch=torch.randn(2, 8), rank=2, slots=3)
    assert [wrapper.recall.dim for wrapper in model.wrappers.values()] == [8, 8, 8]


def test_public_torch_namespace_matches_root() -> None:
    assert experimental.LayerRecall is not None
    assert experimental.LayeredRecallModel is not None
    assert experimental.LayeredRecallConfig is not None
    assert experimental.calibrate_layered_recall is not None


def test_stable_config_calibration_and_normalized_loss() -> None:
    config = experimental.LayeredRecallConfig(layer_paths=("0", "2", "4"), rank=2, slots=3)
    model = experimental.LayeredRecallModel.from_config(backbone(), config)
    clean = torch.randn(3, 5, 8)
    corrupt = clean.clone()
    corrupt[:, 1:3] = 0
    calibration = model.calibrate(clean, corrupt)
    result = experimental.layered_recall_trajectory_loss(model, clean, corrupt, calibration=calibration)

    assert set(calibration.scales) == set(config.layer_paths)
    assert all(value.item() > 0 for value in calibration.scales.values())
    assert result.loss.isfinite()


def test_open_ended_layer_specs_allow_independent_sizes_and_features() -> None:
    config = experimental.LayeredRecallConfig(
        layers=(
            experimental.LayerRecallSpec("0", dim=8, rank=1, slots=2, use_half=False, recognition_mode="none"),
            experimental.LayerRecallSpec("2", dim=8, rank=3, slots=5, use_half=True, recognition_mode="explicit"),
            experimental.LayerRecallSpec("4", dim=8, rank=2, slots=7, recognition_mode="alignment"),
        )
    )
    model = experimental.LayeredRecallModel.from_config(backbone(), config)

    assert config.paths == ("0", "2", "4")
    assert [model.wrappers[path].recall.rank for path in config.paths] == [1, 3, 2]
    assert [model.wrappers[path].recall.slots for path in config.paths] == [2, 5, 7]
    assert [model.wrappers[path].recall.use_half for path in config.paths] == [False, True, True]
    assert [model.wrappers[path].recall.recognition_mode for path in config.paths] == ["none", "explicit", "alignment"]


def test_repeated_lines_can_share_one_physical_layer() -> None:
    config = experimental.LayeredRecallConfig(
        layers=(experimental.LayerRecallSpec("2", dim=8, rank=2, slots=3, copies=3, combine="mean"),)
    )
    model = experimental.LayeredRecallModel.from_config(backbone(), config)
    stack = model.wrappers["2"].recall
    output = model(torch.randn(2, 4, 8))

    assert isinstance(stack, experimental.LayerRecallStack)
    assert len(stack.branches) == 3
    assert stack.combine == "mean"
    assert output.shape == (2, 4, 8)
    with stack.enabled_lines((0,)):
        assert stack(torch.randn(2, 4, 8)).shape == (2, 4, 8)
    with stack.enabled_lines(()) as disabled:
        assert torch.count_nonzero(disabled(torch.randn(2, 4, 8))) == 0
    assert stack._line_enabled == [True, True, True]


def test_layer_enable_contexts_restore_state_and_validate_paths() -> None:
    model = experimental.LayeredRecallModel.attach(backbone(), ("0", "2", "4"), rank=2, slots=3)
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
    model = experimental.LayeredRecallModel.attach(backbone(), ("0", "2"), rank=2, slots=3)
    for wrapper in model.wrappers.values():
        wrapper.capture = True
    model(torch.randn(2, 4, 8))
    report = model.diagnostics()

    assert set(report) == {"0", "2"}
    assert set(report["0"]) == {"raw_delta", "delta", "recognition", "survival"}


def test_layer_survival_diagnostic_is_finite_for_zero_fp16_delta() -> None:
    model = experimental.LayeredRecallModel.attach(
        nn.Sequential(nn.Identity()),
        ("0",),
        dims={"0": 8},
        rank=2,
        slots=3,
    ).half()
    wrapper = model.wrappers["0"]
    wrapper.capture = True
    with torch.no_grad():
        wrapper.recall.emit.weight.zero_()

    model(torch.randn(2, 4, 8, dtype=torch.float16))

    assert wrapper.last_survival is not None
    assert torch.isfinite(wrapper.last_survival).all()
    assert torch.count_nonzero(wrapper.last_survival) == 0


def test_layered_recall_delegates_generate() -> None:
    class Generative(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = nn.Linear(8, 8)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.block(x)

        def generate(self, x: torch.Tensor, *, scale: float = 1.0) -> torch.Tensor:
            return self.forward(x) * scale

    wrapped = experimental.LayeredRecallModel.attach(Generative(), ("block",), dims={"block": 8}, rank=2, slots=2)
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

    layered = experimental.LayeredRecallModel.attach(
        QwenLike(),
        ("model.layers.1", "model.layers.3", "model.layers.5"),
        rank=2,
        slots=3,
    )

    assert layered(torch.randn(2, 4, 8)).shape == (2, 4, 8)
    assert all(not parameter.requires_grad for wrapper in layered.wrappers.values() for parameter in wrapper.base.parameters())
