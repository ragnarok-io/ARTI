import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import arti
from arti.fit import ARTIProject
from arti.fit.artifacts import hash_tensor_state_dict, stable_json_sha256
from arti.fit.insertion import ARTIAdapterWrapper
from arti.fit.scanner import run_model


def tiny_model() -> nn.Module:
    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))


class TinyTransformerBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attn = nn.Module()
        self.attn.out_proj = nn.Linear(4, 4)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(4, 8)
        self.mlp.fc2 = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn.out_proj(x)
        return self.mlp.fc2(torch.relu(self.mlp.fc1(x)))


def tiny_transformer_like_model() -> nn.Module:
    return nn.Sequential(TinyTransformerBlock(), nn.Linear(4, 2))


class TinyTimmBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(4)
        self.attn = nn.Module()
        self.attn.proj = nn.Linear(4, 4)
        self.norm2 = nn.LayerNorm(4)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(4, 8)
        self.mlp.fc2 = nn.Linear(8, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.attn.proj(self.norm1(x))
        return self.mlp.fc2(torch.relu(self.mlp.fc1(self.norm2(x))))


class TinyTimmViT(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([TinyTimmBlock()])
        self.norm = nn.LayerNorm(4)
        self.head = nn.Linear(4, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))


def test_project_scan_finds_linear_latent_candidates():
    model = tiny_model()
    sample = torch.randn(3, 4)

    report = arti.project(model).scan(sample).report()

    names = [candidate.name for candidate in report.scanned.candidates]
    assert "0" in names
    assert "2" in names
    assert report.scanned.total_parameters > 0


def test_scan_deduplicates_reused_module_candidates():
    class ReuseLike(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.shared = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.shared(self.shared(x))

    report = arti.project(ReuseLike()).scan(torch.randn(2, 4)).report().scanned
    names = [candidate.name for candidate in report.candidates]

    assert names.count("shared") == 1
    assert report.scanned_modules == 1
    assert report.candidate_events == 2
    assert report.duplicate_events == 1
    assert report.to_dict()["candidate_count"] == 1
    markdown = arti.project(ReuseLike()).scan(torch.randn(2, 4)).report().to_markdown()
    assert "Candidate events: `2`" in markdown
    assert "Duplicate events: `1`" in markdown


def test_scan_records_pretrained_style_batch_schema():
    class DictModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(16, 4)
            self.proj = nn.Linear(4, 2)

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
            return self.proj(self.embed(input_ids).float())

    batch = {
        "input_ids": torch.randint(0, 16, (2, 5)),
        "attention_mask": torch.tensor([[1, 1, 1, 0, 0], [1, 1, 0, 0, 0]]),
        "labels": torch.randint(0, 2, (2, 5)),
    }

    report = arti.project(DictModel()).plugin("transformers").scan(batch).report().to_dict()
    schema = report["scanned"]["batch_schema"]

    assert schema["kind"] == "dict"
    assert schema["token_key"] == "input_ids"
    assert schema["mask_key"] == "attention_mask"
    assert schema["label_key"] == "labels"


def test_scan_records_embedding_layernorm_and_candidate_metadata():
    class EncoderLike(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_tokens = nn.Embedding(16, 4)
            self.layer_norm = nn.LayerNorm(4)
            self.proj = nn.Linear(4, 2)

        def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None):
            hidden = self.layer_norm(self.embed_tokens(input_ids))
            return self.proj(hidden)

    batch = {
        "input_ids": torch.randint(0, 16, (2, 5)),
        "attention_mask": torch.ones(2, 5, dtype=torch.long),
    }

    report = arti.project(EncoderLike()).plugin("transformers").scan(batch).report().scanned
    by_name = {candidate.name: candidate for candidate in report.candidates}

    assert by_name["embed_tokens"].module_type == "Embedding"
    assert by_name["embed_tokens"].tensor_rank == 3
    assert by_name["embed_tokens"].source == "forward"
    assert by_name["layer_norm"].module_type == "LayerNorm"
    assert by_name["layer_norm"].dim == 4


def test_scan_handles_multihead_attention_tuple_output():
    class AttentionLike(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = nn.MultiheadAttention(4, 2, batch_first=True)
            self.layer_norm = nn.LayerNorm(4)

        def forward(self, x: torch.Tensor):
            hidden, _ = self.self_attn(x, x, x, need_weights=False)
            return self.layer_norm(hidden)

    sample = torch.randn(2, 3, 4)

    report = arti.project(AttentionLike()).plugin("transformers").scan(sample).insert(where="attention").report()
    names = [candidate.name for candidate in report.scanned.candidates]

    assert "self_attn" in names
    assert "layer_norm" in names
    assert report.inserted[0].name == "self_attn"
    assert report.scanned.candidates[names.index("self_attn")].module_type == "MultiheadAttention"


def test_scan_and_insert_structured_mapping_output():
    class DictLinear(nn.Linear):
        def forward(self, x: torch.Tensor):
            hidden = super().forward(x)
            return {"last_hidden_state": hidden, "side": hidden.mean(dim=-1)}

    class AddOneAdapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, x: torch.Tensor, **kwargs):
            self.calls += 1
            return x + 1.0

    class MappingOutputModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = DictLinear(4, 4)
            self.head = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor):
            output = self.block(x)
            return self.head(output["last_hidden_state"])

    model = MappingOutputModel()
    sample = torch.randn(2, 3, 4)
    project = arti.project(model).scan(sample).insert(where="block")
    recorder = AddOneAdapter()
    model.block.adapter = recorder

    output = model.block(sample)
    expected = model.block.base(sample)["last_hidden_state"] + 1.0

    assert project.report().scanned.candidates[0].name == "block"
    assert project.report().scanned.candidates[0].output_shape == (2, 3, 4)
    assert isinstance(output, dict)
    assert torch.allclose(output["last_hidden_state"], expected)
    assert output["side"].shape == (2, 3)
    assert recorder.calls == 1
    assert model(sample).shape == (2, 3, 2)


def test_scan_and_insert_lstm_tuple_output_batch_first():
    class LSTMLike(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(3, 5, batch_first=True)
            self.head = nn.Linear(5, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            output, _ = self.lstm(x)
            return self.head(output[:, -1])

    model = LSTMLike()
    sample = torch.randn(2, 4, 3)
    project = arti.project(model).plugin("recurrent").scan(sample)
    by_name = {candidate.name: candidate for candidate in project.report().scanned.candidates}

    assert by_name["lstm"].module_type == "LSTM"
    assert by_name["lstm"].tensor_rank == 3
    assert by_name["lstm"].dim == 5
    project.insert()
    output = model(sample)
    output.square().mean().backward()

    assert isinstance(model.lstm, ARTIAdapterWrapper)
    assert output.shape == (2, 2)
    assert any(param.grad is not None for param in model.lstm.adapter.parameters())


def test_fit_recurrent_profile_handles_time_first_gru():
    class GRULike(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.gru = nn.GRU(3, 4)
            self.head = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            output, _ = self.gru(x)
            return self.head(output[-1])

    result = arti.fit(GRULike(), sample_batch=torch.randn(5, 2, 3), profile="recurrent", max_adapters=1)

    assert result.report.plugins == ("torch", "recurrent")
    assert result.report.inserted[0].name == "gru"
    assert result.model(torch.randn(5, 2, 3)).shape == (2, 2)


def test_static_scan_includes_common_latent_modules():
    class StaticLike(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Embedding(8, 4)
            self.norm = nn.LayerNorm(4)
            self.proj = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor):
            return self.proj(self.norm(self.embed(x)))

    report = arti.project(StaticLike()).scan().report().scanned
    by_name = {candidate.name: candidate for candidate in report.candidates}

    assert by_name["embed"].source == "static"
    assert by_name["embed"].dim == 4
    assert by_name["norm"].dim == 4
    assert by_name["proj"].dim == 2
    assert report.scanned_modules == 3
    assert report.candidate_events == 3
    assert report.duplicate_events == 0


def test_scan_and_insert_conv2d_spatial_latent_candidate():
    class ConvLike(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 5, kernel_size=3, padding=1)
            self.head = nn.Linear(5, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            hidden = self.conv(x).mean(dim=(-1, -2))
            return self.head(hidden)

    model = ConvLike()
    sample = torch.randn(2, 3, 8, 8)
    project = arti.project(model).scan(sample)
    by_name = {candidate.name: candidate for candidate in project.report().scanned.candidates}

    assert by_name["conv"].module_type == "Conv2d"
    assert by_name["conv"].tensor_rank == 4
    assert by_name["conv"].dim == 5
    project.insert(where="conv")

    output = model(sample)
    loss = output.square().mean()
    loss.backward()

    assert output.shape == (2, 2)
    assert isinstance(model.conv, ARTIAdapterWrapper)
    assert any(param.grad is not None for param in model.conv.adapter.parameters())


def test_vision_cnn_plugin_uses_conv_strategy():
    class ConvLike(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(3, 4, kernel_size=3, padding=1)
            self.features = nn.Sequential(nn.Conv2d(4, 5, kernel_size=3, padding=1), nn.ReLU())
            self.head = nn.Linear(5, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            hidden = self.features(self.conv1(x)).mean(dim=(-1, -2))
            return self.head(hidden)

    model = ConvLike()
    project = arti.project(model).plugin("vision-cnn").scan(torch.randn(2, 3, 8, 8)).insert(max_adapters=2)

    assert [adapter.name for adapter in project.report().inserted] == ["conv1", "features.0"]
    assert project.report().plugins == ("torch", "vision-cnn")
    assert project.report().to_dict()["plugin_details"][-1]["default_strategy"] == "vision-cnn"


def test_scan_and_insert_cnn_normalization_layers():
    class NormConvLike(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
            self.bn1 = nn.BatchNorm2d(4)
            self.group_norm = nn.GroupNorm(2, 4)
            self.head = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            hidden = self.group_norm(self.bn1(self.conv(x))).mean(dim=(-1, -2))
            return self.head(hidden)

    model = NormConvLike()
    sample = torch.randn(2, 3, 8, 8)
    project = arti.project(model).plugin("vision-cnn").scan(sample)
    by_name = {candidate.name: candidate for candidate in project.report().scanned.candidates}

    assert by_name["bn1"].module_type == "BatchNorm2d"
    assert by_name["bn1"].tensor_rank == 4
    assert by_name["bn1"].dim == 4
    assert by_name["group_norm"].module_type == "GroupNorm"
    assert by_name["group_norm"].dim == 4

    project.insert(where="normalization", max_adapters=2)
    output = model(sample)
    output.mean().backward()

    assert isinstance(model.bn1, ARTIAdapterWrapper)
    assert isinstance(model.group_norm, ARTIAdapterWrapper)
    assert output.shape == (2, 2)
    assert any(param.grad is not None for param in model.bn1.adapter.parameters())


def test_fit_cnn_profile_uses_vision_cnn_strategy():
    class ConvLike(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.conv = nn.Conv2d(3, 4, kernel_size=3, padding=1)
            self.head = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.head(self.conv(x).mean(dim=(-1, -2)))

    result = arti.fit(ConvLike(), sample_batch=torch.randn(2, 3, 8, 8), profile="cnn", max_adapters=1)

    assert result.report.plugins == ("torch", "vision-cnn")
    assert result.report.inserted[0].name == "conv"
    assert result.model(torch.randn(2, 3, 8, 8)).shape == (2, 2)


def test_attention_mask_to_visibility_supports_causal_bridge():
    mask = torch.tensor([[1, 1, 0]])

    visibility = arti.attention_mask_to_visibility(mask, causal=True)

    assert visibility.shape == (1, 3, 3)
    assert visibility[0, 0, 0]
    assert visibility[0, 1, 0]
    assert visibility[0, 1, 1]
    assert not visibility[0, 0, 1]
    assert not visibility[0, 2, 0]


def test_inserted_adapter_receives_attention_mask_from_dict_batch():
    class DictModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None):
            return self.proj(x)

    class RecordingAdapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.kwargs = None

        def forward(self, x: torch.Tensor, **kwargs):
            self.kwargs = kwargs
            return x

    model = DictModel()
    batch = {
        "x": torch.randn(2, 3, 4),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
    }
    arti.project(model).scan(batch).insert(where="proj")
    recorder = RecordingAdapter()
    model.proj.adapter = recorder

    run_model(model, batch)

    assert recorder.kwargs is not None
    assert torch.equal(recorder.kwargs["mask"].bool(), batch["attention_mask"].bool())
    assert recorder.kwargs["visibility"].shape == (2, 3, 3)


def test_project_runtime_causal_controls_adapter_visibility():
    class DictModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None):
            return self.proj(x)

    class RecordingAdapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.kwargs = None

        def forward(self, x: torch.Tensor, **kwargs):
            self.kwargs = kwargs
            return x

    model = DictModel()
    batch = {
        "x": torch.randn(2, 3, 4),
        "attention_mask": torch.ones(2, 3, dtype=torch.long),
        "labels": torch.zeros(2, 3, 4),
    }
    project = arti.project(model).runtime(causal=True).scan(batch).insert(where="proj")
    recorder = RecordingAdapter()
    model.proj.adapter = recorder

    project.validate([batch])

    assert recorder.kwargs is not None
    assert not recorder.kwargs["visibility"][0, 0, 1]
    assert recorder.kwargs["visibility"][0, 1, 0]
    assert project.report().runtime_causal is True


def test_configured_runtime_field_names_feed_adapter_context_without_leaking_to_model():
    class DictModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(4, 4)
            self.forward_kwargs = None

        def forward(self, x: torch.Tensor, **kwargs):
            self.forward_kwargs = kwargs
            return self.proj(x)

    class RecordingAdapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.kwargs = None

        def forward(self, x: torch.Tensor, **kwargs):
            self.kwargs = kwargs
            return x

    model = DictModel()
    batch = {
        "x": torch.randn(2, 3, 4),
        "token_mask": torch.tensor([[1, 1, 0], [1, 0, 0]]),
        "phase_coord": torch.randn(2, 3, 2),
        "observer_phase": torch.randn(2, 2),
        "inverse_ops": torch.eye(4).repeat(2, 1, 1),
    }
    config = {
        "fit": {"profile": "observer-phase", "phases": 2, "scale": "tiny"},
        "runtime": {
            "mask_key": "token_mask",
            "coord_key": "phase_coord",
            "observer_coord_key": "observer_phase",
            "frame_operators_key": "inverse_ops",
        },
        "insertion": {"where": "proj"},
    }

    project = arti.project(model).configure(config).scan(batch).insert()
    original_adapter = model.proj.adapter
    recorder = RecordingAdapter()
    recorder.layer = original_adapter.layer
    model.proj.adapter = recorder

    project.profile_forward(batch, warmup=0, repeats=1)

    assert model.forward_kwargs == {}
    assert recorder.kwargs is not None
    assert torch.equal(recorder.kwargs["mask"].bool(), batch["token_mask"].bool())
    assert torch.equal(recorder.kwargs["coord"], batch["phase_coord"])
    assert torch.equal(recorder.kwargs["observer_coord"], batch["observer_phase"])
    assert torch.equal(recorder.kwargs["frame_operators"], batch["inverse_ops"])
    assert recorder.kwargs["visibility"].shape == (2, 3, 3)
    assert project.report().fit_config["runtime"]["mask_key"] == "token_mask"


def test_observer_phase_adapter_consumes_runtime_frame_context():
    class DictModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor):
            return self.proj(x)

    model = DictModel()
    coord = torch.zeros(2, 3, 2)
    coord[:, :, 0] = 1.0
    observer_coord = torch.zeros(2, 2)
    observer_coord[:, 0] = 1.0
    batch = {
        "x": torch.randn(2, 3, 4),
        "coord": coord,
        "observer_coord": observer_coord,
        "frame_operators": torch.eye(4).repeat(2, 1, 1),
    }

    project = arti.project(model).profile("observer-phase", phases=2).scan(batch).insert(where="proj")

    out = run_model(model, batch)

    assert out.shape == (2, 3, 4)
    assert project.report().inserted[0].profile == "observer-phase"
    assert project.report().mechanism is not None
    assert project.report().mechanism.coord_dim == 2
    assert project.report().mechanism.coord_frame_mode == "operator_bank"
    assert model.proj.adapter.layer.config.coord_frame_mode == "operator_bank"


def test_observer_phase_adapter_requires_frame_operators():
    class DictModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor):
            return self.proj(x)

    model = DictModel()
    coord = torch.zeros(1, 2, 2)
    coord[:, :, 0] = 1.0
    batch = {"x": torch.randn(1, 2, 4), "coord": coord}

    arti.project(model).profile("observer-phase", phases=2).scan(batch).insert(where="proj")

    try:
        run_model(model, batch)
    except ValueError as exc:
        assert "frame_operators" in str(exc)
    else:
        raise AssertionError("observer-phase adapter should require frame_operators")


def test_project_insert_wraps_selected_modules_and_freezes_base():
    model = tiny_model()
    project = arti.project(model).scan(torch.randn(2, 4)).insert(where="0", freeze_base=True)

    assert isinstance(model[0], ARTIAdapterWrapper)
    assert not any(param.requires_grad for param in model[0].base.parameters())
    assert all(param.requires_grad for param in model[0].adapter.parameters())
    assert project.report().parameters is not None
    assert project.report().parameters.trainable_base_parameters == 0
    assert project.report().parameters.trainable_adapter_parameters == project.report().parameters.adapter_parameters
    assert project.report().inserted[0].name == "0"
    assert model(torch.randn(2, 4)).shape == (2, 2)


def test_project_insert_can_keep_base_trainable():
    model = tiny_model()
    project = arti.project(model).scan(torch.randn(2, 4)).insert(where="0", freeze_base=False)

    assert project.report().parameters is not None
    assert project.report().parameters.trainable_base_parameters > 0
    assert project.report().parameters.trainable_adapter_parameters == project.report().parameters.adapter_parameters
    assert project.report().to_dict()["parameters"]["frozen_base"] is False


def test_project_plan_insert_dry_run_does_not_mutate_model():
    model = tiny_model()
    project = arti.project(model).scan(torch.randn(2, 4))

    plan = project.plan_insert(where=["0", "2"], max_adapters=1)

    assert plan.selected[0].name == "0"
    assert plan.adapter_parameters == plan.selected[0].parameters
    assert isinstance(model[0], nn.Linear)
    project.insert(where=["0", "2"], max_adapters=1)
    assert [adapter.name for adapter in project.report().inserted] == [adapter.name for adapter in plan.selected]
    assert isinstance(model[0], ARTIAdapterWrapper)


def test_project_plan_insert_records_budget_skips():
    model = tiny_model()
    project = arti.project(model).scan(torch.randn(2, 4))

    plan = project.plan_insert(where=["0", "2"], max_extra_params=1)

    assert not plan.selected
    assert [adapter.name for adapter in plan.skipped_budget] == ["0", "2"]
    assert plan.to_dict()["skipped_budget"]
    assert project.report().insertion_plan is plan


def test_project_progressive_preview_is_non_mutating_and_auditable():
    model = tiny_model()
    original_types = tuple(type(module) for module in model)
    original_trainable = tuple(parameter.requires_grad for parameter in model.parameters())

    report = (
        arti.project(model)
        .at(["0", "2"])
        .freeze(True)
        .budget(max_adapters=1, max_extra_params="10000%")
        .preview(torch.randn(2, 4))
    )

    assert tuple(type(module) for module in model) == original_types
    assert tuple(parameter.requires_grad for parameter in model.parameters()) == original_trainable
    assert report.inserted == ()
    assert report.insertion_plan is not None
    assert [row.name for row in report.insertion_plan.selected] == ["0"]
    assert report.insertion_plan.spec.freeze_base is True
    assert report.fit_config["insertion"]["where"] == ["0", "2"]
    assert report.fit_config["insertion"]["max_extra_params"] == "10000%"


def test_fit_dry_run_plans_without_mutating_or_training():
    model = tiny_model()
    before = {name: param.detach().clone() for name, param in model.named_parameters()}
    x = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    result = arti.fit(
        model,
        train_loader=loader,
        sample_batch=x[:2],
        target_modules=["0", "2"],
        max_adapters=1,
        steps=2,
        dry_run=True,
    )

    assert isinstance(model[0], nn.Linear)
    assert result.adapter_count == 0
    assert result.report.inserted == ()
    assert result.report.steps == 0
    assert result.report.loss_history == ()
    assert result.report.task_history == ()
    assert result.report.objective_plan == ("task-fit",)
    assert result.report.insertion_plan is not None
    assert [adapter.name for adapter in result.report.insertion_plan.selected] == ["0"]
    assert result.report.to_dict()["insertion_plan"]["selected"][0]["name"] == "0"
    assert "## Insertion Plan" in result.report.to_markdown()
    for name, param in model.named_parameters():
        assert torch.equal(param, before[name])


def test_project_write_plan_exports_json_and_markdown_without_mutating(tmp_path: Path):
    model = tiny_model()
    project = arti.project(model).scan(torch.randn(2, 4)).objectives(["task-fit", "validate"])

    json_path = project.write_plan(tmp_path / "arti-plan.json", where=["0", "2"], max_adapters=1)
    md_path = project.write_plan(tmp_path / "arti-plan.md")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    markdown = md_path.read_text(encoding="utf-8")

    assert payload["format_version"] == 1
    assert payload["package_name"] == "arti"
    assert payload["kind"] == "fit-plan"
    assert payload["report"]["objective_plan"] == ["task-fit", "validate"]
    assert payload["report"]["insertion_plan"]["selected"][0]["name"] == "0"
    assert payload["report"]["inserted"] == []
    assert isinstance(model[0], nn.Linear)
    assert "## Insertion Plan" in markdown
    assert "`0`" in markdown

    validated = arti.validate_plan(json_path)
    assert validated["report"]["insertion_plan"]["adapter_parameters"] == payload["report"]["insertion_plan"]["adapter_parameters"]


def test_validate_plan_rejects_budget_gate_mismatch(tmp_path: Path):
    plan_path = arti.project(tiny_model()).scan(torch.randn(2, 4)).write_plan(
        tmp_path / "budget-plan.json",
        where="0",
        max_adapters=1,
        max_extra_params="10000%",
    )
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["report"]["insertion_plan"]["spec"]["max_extra_params"] = 1
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        arti.validate_plan(plan_path)
    except ValueError as exc:
        assert "max_extra_params" in str(exc)
    else:
        raise AssertionError("plan exceeding parameter budget should fail validation")

    payload["report"]["insertion_plan"]["spec"]["max_extra_params"] = 1000000
    payload["report"]["insertion_plan"]["spec"]["max_adapters"] = 0
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        arti.validate_plan(plan_path)
    except ValueError as exc:
        assert "max_adapters" in str(exc)
    else:
        raise AssertionError("plan exceeding adapter count budget should fail validation")


def test_fit_project_config_drives_plan_and_insert(tmp_path: Path):
    config_path = tmp_path / "arti.json"
    config_path.write_text(
        json.dumps(
            {
                "fit": {
                    "plugins": ["torch"],
                    "profile": "observer-phase",
                    "phases": 6,
                    "scale": "tiny",
                    "objectives": ["task-fit"],
                },
                "runtime": {"causal": True},
                "insertion": {"where": ["0", "2"], "max_adapters": 1, "max_extra_params": "10000%"},
            }
        ),
        encoding="utf-8",
    )
    config = arti.load_fit_config(config_path)
    model = tiny_model()
    project = arti.project(model).configure(config).scan(torch.randn(2, 4))

    plan = project.plan_insert()
    assert config.profile == "observer-phase"
    assert config.phases == 6
    assert project.report().runtime_causal is True
    assert project.report().objective_plan == ("task-fit",)
    assert plan.selected[0].name == "0"
    assert project.report().mechanism.coord_dim == 6
    assert project.report().fit_config["profile"] == "observer-phase"
    assert project.report().fit_config["phases"] == 6
    assert project.report().config_fingerprint == config.fingerprint

    project.insert()
    assert [adapter.name for adapter in project.report().inserted] == ["0"]


def test_fit_project_config_overrides_mechanism_dimensions(tmp_path: Path):
    config_path = tmp_path / "arti.json"
    config_path.write_text(
        json.dumps(
            {
                "fit": {"profile": "observer-phase", "phases": 6, "scale": "small"},
                "mechanism": {
                    "coord_dim": 3,
                    "operator_count": 5,
                    "interface_slots": 6,
                    "recall_slots": 2,
                    "recall_steps": 1,
                    "hidden_multiplier": 1.5,
                },
                "insertion": {"where": "0", "max_adapters": 1},
            }
        ),
        encoding="utf-8",
    )

    config = arti.load_fit_config(config_path)
    model = tiny_model()
    project = arti.project(model).configure(config).scan(torch.randn(2, 4)).insert()
    mechanism = project.report().mechanism
    wrapper = model[0]

    assert mechanism.coord_dim == 3
    assert mechanism.operator_count == 5
    assert mechanism.interface_slots == 6
    assert mechanism.recall_slots == 2
    assert mechanism.recall_steps == 1
    assert mechanism.hidden_multiplier == 1.5
    assert isinstance(wrapper, ARTIAdapterWrapper)
    assert wrapper.adapter.layer.config.coord_dim == 3
    assert wrapper.adapter.layer.config.hidden_dim == 12
    assert wrapper.adapter.layer.config.operator_count == 5
    assert project.report().fit_config["mechanism"]["operator_count"] == 5


def test_fit_config_mechanism_overrides_survive_convenience_api(tmp_path: Path):
    config_path = tmp_path / "arti.json"
    config_path.write_text(
        json.dumps(
            {
                "fit": {"profile": "observer-phase", "scale": "small"},
                "mechanism": {"coord_dim": 5, "operator_count": 6, "interface_slots": 7, "recall_slots": 2},
                "insertion": {"where": "0", "max_adapters": 1},
            }
        ),
        encoding="utf-8",
    )

    result = arti.fit(tiny_model(), config=config_path, sample_batch=torch.randn(2, 4), dry_run=True)

    assert result.report.mechanism.coord_dim == 5
    assert result.report.mechanism.operator_count == 6
    assert result.report.mechanism.interface_slots == 7
    assert result.report.mechanism.recall_slots == 2
    assert result.report.fit_config["mechanism"]["coord_dim"] == 5


def test_project_mechanism_fluent_api_overrides_profile_and_scale():
    model = tiny_model()
    project = (
        arti.project(model)
        .profile("latent-adapt")
        .scale("tiny")
        .mechanism(observer_phase=True, coord_dim=4, coord_frame_mode="operator_bank", operator_count=3, interface_slots=5, recall_slots=2)
        .scan(torch.randn(2, 4))
        .insert(where="0")
    )
    mechanism = project.report().mechanism

    assert mechanism.observer_phase is True
    assert mechanism.coord_dim == 4
    assert mechanism.coord_frame_mode == "operator_bank"
    assert mechanism.operator_count == 3
    assert mechanism.interface_slots == 5
    assert mechanism.recall_slots == 2
    assert isinstance(model[0], ARTIAdapterWrapper)
    assert model[0].adapter.layer.config.coord_frame_mode == "operator_bank"


def test_cli_validate_config_outputs_normalized_config(tmp_path: Path, capsys):
    from arti.cli import main

    config_path = tmp_path / "arti.json"
    config_path.write_text(
        json.dumps({"fit": {"profile": "virtual-recall", "scale": "base"}, "insertion": {"where": "0"}}),
        encoding="utf-8",
    )

    assert main(["validate", "config", str(config_path)]) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["kind"] == "fit-config"
    assert output["config"]["profile"] == "virtual-recall"
    assert output["config"]["scale"] == "base"
    assert output["config"]["insertion"]["where"] == ["0"]
    assert output["mechanism"]["profile"] == "virtual-recall"
    assert output["mechanism"]["scale"] == "base"
    assert output["mechanism"]["recall_slots"] == 8
    assert output["config_fingerprint"] == arti.load_fit_config(config_path).fingerprint


def test_cli_validate_config_can_require_profile_scale_and_mechanism(tmp_path: Path, capsys):
    from arti.cli import main

    config_path = tmp_path / "arti.json"
    config_path.write_text(
        json.dumps(
            {
                "fit": {"profile": "observer-phase", "phases": 6, "scale": "small"},
                "mechanism": {"coord_dim": 4, "operator_count": 5},
                "runtime": {"mask_key": "token_mask", "coord_key": "phase_coord"},
            }
        ),
        encoding="utf-8",
    )

    profile, scale = arti.resolve_fit_config_mechanism(arti.load_fit_config(config_path))
    assert profile.coord_dim == 4
    assert scale.operator_count == 5

    assert main(
        [
            "validate",
            "config",
            str(config_path),
            "--expect-profile",
            "observer-phase",
            "--expect-scale",
            "small",
            "--expect-mechanism",
            "coord_dim=4",
            "--expect-mechanism",
            "operator_count=5",
            "--expect-runtime-field",
            "mask_key=token_mask",
            "--expect-runtime-field",
            "coord_key=phase_coord",
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["expected_mechanism"]["coord_dim"] == 4
    assert output["mechanism"]["operator_count"] == 5
    assert output["expected_runtime_fields"]["mask_key"] == "token_mask"
    assert output["runtime"]["coord_key"] == "phase_coord"

    assert main(["validate", "config", str(config_path), "--expect-profile", "latent-adapt"]) == 1
    assert "profile" in capsys.readouterr().err
    assert main(["validate", "config", str(config_path), "--expect-scale", "base"]) == 1
    assert "scale" in capsys.readouterr().err
    assert main(["validate", "config", str(config_path), "--expect-mechanism", "operator_count=4"]) == 1
    assert "mechanism.operator_count" in capsys.readouterr().err
    assert main(["validate", "config", str(config_path), "--expect-runtime-field", "mask_key=attention_mask"]) == 1
    assert "runtime.mask_key" in capsys.readouterr().err


def test_write_fit_config_template_round_trips_json_and_toml(tmp_path: Path):
    json_path = arti.write_fit_config_template(tmp_path / "arti.json")
    toml_path = arti.write_fit_config_template(tmp_path / "arti.toml", profile="virtual-recall", scale="base")

    json_config = arti.load_fit_config(json_path)
    toml_config = arti.load_fit_config(toml_path)
    json_payload = json.loads(json_path.read_text(encoding="utf-8"))

    assert json_payload["$schema"] == "docs/reference/fit-config.schema.json"
    assert json_config.profile == "latent-adapt"
    assert json_config.scale == "small"
    assert json_config.where == ("*",)
    assert json_config.fingerprint == arti.template_fit_config().fingerprint
    assert toml_config.profile == "virtual-recall"
    assert toml_config.scale == "base"


def test_cli_init_config_writes_template_and_respects_force(tmp_path: Path, capsys):
    from arti.cli import main

    config_path = tmp_path / "arti.json"

    assert main(["init-config", str(config_path), "--profile", "virtual-recall"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["kind"] == "fit-config-template"
    assert output["config"]["profile"] == "virtual-recall"
    assert json.loads(config_path.read_text(encoding="utf-8"))["$schema"] == "docs/reference/fit-config.schema.json"
    assert arti.load_fit_config(config_path).profile == "virtual-recall"

    assert main(["init-config", str(config_path)]) == 1
    assert "already exists" in capsys.readouterr().err
    assert main(["init-config", str(config_path), "--force"]) == 0
    assert arti.load_fit_config(config_path).profile == "latent-adapt"


def test_validate_fit_config_rejects_unknown_registry_values(tmp_path: Path):
    config_path = tmp_path / "bad-config.json"
    config_path.write_text(json.dumps({"fit": {"profile": "missing-profile"}}), encoding="utf-8")

    try:
        arti.load_fit_config(config_path)
    except ValueError as exc:
        assert "unknown ARTI profile" in str(exc)
    else:
        raise AssertionError("unknown profile should fail config validation")

    try:
        arti.validate_fit_config({"fit": {"scale": "huge"}})
    except ValueError as exc:
        assert "unknown ARTI scale" in str(exc)
    else:
        raise AssertionError("unknown scale should fail config validation")

    try:
        arti.validate_fit_config({"fit": {"plugins": ["missing-plugin"]}})
    except ValueError as exc:
        assert "unknown ARTI fit plugin" in str(exc)
    else:
        raise AssertionError("unknown plugin should fail config validation")


def test_cli_validate_config_rejects_invalid_config(tmp_path: Path, capsys):
    from arti.cli import main

    config_path = tmp_path / "bad-config.json"
    config_path.write_text(json.dumps({"fit": {"objective": "unknown-task"}}), encoding="utf-8")

    assert main(["validate", "config", str(config_path)]) == 1
    output = capsys.readouterr()
    assert "unknown ARTI fit objective" in output.err


def test_fit_convenience_accepts_declarative_config(tmp_path: Path):
    config_path = tmp_path / "arti.json"
    config_path.write_text(
        json.dumps(
            {
                "fit": {"profile": "observer-phase", "phases": 5, "scale": "tiny", "objectives": ["task-fit"]},
                "runtime": {"causal": True},
                "insertion": {"where": "0", "max_adapters": 1, "max_extra_params": "10000%"},
            }
        ),
        encoding="utf-8",
    )

    result = arti.fit(tiny_model(), config=config_path, sample_batch=torch.randn(2, 4), dry_run=True)

    assert result.report.profile == "observer-phase"
    assert result.report.scale == "tiny"
    assert result.report.runtime_causal is True
    assert result.report.objective_plan == ("task-fit",)
    assert result.report.mechanism.coord_dim == 5
    assert [adapter.name for adapter in result.report.insertion_plan.selected] == ["0"]
    assert result.report.inserted == ()
    assert result.report.fit_config["profile"] == "observer-phase"
    assert result.report.config_fingerprint == arti.load_fit_config(config_path).fingerprint


def test_fit_convenience_trains_adapter_and_exports(tmp_path: Path):
    torch.manual_seed(1)
    model = tiny_model()
    x = torch.randn(16, 4)
    y = torch.randint(0, 2, (16,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    result = arti.fit(model, train_loader=loader, sample_batch=x[:2], target_modules="0", freeze_base=True, steps=2)
    artifact = result.export(tmp_path / "adapter.pt")
    report_path = result.write_report(tmp_path / "report.md")
    payload = torch.load(artifact, weights_only=True)

    assert result.adapter_count == 1
    assert result.report.steps == 2
    assert len(result.report.loss_history) == 2
    assert "adapter_state_dict" in payload
    assert payload["manifest"]["format_version"] == 1
    assert payload["manifest"]["package_name"] == "arti"
    assert payload["manifest"]["backend"] == "torch"
    assert payload["manifest"]["include_base"] is False
    assert payload["manifest"]["adapter_key_count"] == len(payload["adapter_state_dict"])
    assert payload["manifest"]["config_fingerprint"] == payload["report"]["config_fingerprint"]
    assert len(payload["manifest"]["adapter_state_sha256"]) == 64
    assert payload["manifest"]["report_sha256"] == stable_json_sha256(payload["report"])
    assert payload["report"]["fit_config"]["profile"] == "latent-adapt"
    assert payload["report"]["loss_history"]
    assert payload["report"]["summary"]["inserted_count"] == 1
    assert payload["report"]["summary"]["last_loss"] == result.report.loss_history[-1]
    assert payload["report"]["parameters"]["trainable_adapter_parameters"] == result.report.parameters.trainable_adapter_parameters
    assert "state_dict" not in payload
    assert payload["adapter_state_dict"]
    assert artifact.exists()
    assert report_path.read_text(encoding="utf-8").startswith("# ARTI Fit Report")


def test_fit_supports_multiple_patterns_and_budget():
    model = tiny_model()

    result = arti.fit(
        model,
        sample_batch=torch.randn(2, 4),
        target_modules=["0", "2"],
        max_adapters=1,
        freeze_base=True,
    )

    assert result.adapter_count == 1
    assert result.report.inserted[0].name == "0"
    assert result.report.to_dict()["insertion"]["where"] == ["0", "2"]
    assert result.report.to_dict()["mechanism"]["operator_count"] == 4
    assert result.report.to_dict()["mechanism"]["interface_slots"] == 8


def test_project_validate_records_validation_history():
    model = tiny_model()
    x = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)
    project = arti.project(model).scan(x[:2]).insert(where="0")

    metrics = project.validate(loader)
    report = project.report()

    assert metrics["batches"] == 2.0
    assert len(report.validation_history) == 1
    assert report.to_dict()["validation_history"][0]["batches"] == 2.0


def test_project_profile_forward_records_runtime_diagnostics(tmp_path: Path):
    model = tiny_model()
    project = arti.project(model).scan(torch.randn(2, 4)).insert(where="0")

    profile = project.profile_forward(torch.randn(3, 4), warmup=0, repeats=2)
    result = project.fit()
    artifact = result.export(tmp_path / "profiled.pt")
    payload = torch.load(artifact, weights_only=True)

    assert profile.repeats == 2
    assert profile.mean_ms >= 0.0
    assert profile.output_shape == (3, 2)
    assert result.report.forward_profiles[0].output_shape == (3, 2)
    assert payload["report"]["forward_profiles"][0]["output_shape"] == [3, 2]
    assert result.report.task_history[-1].name == "profile-forward"
    assert "## Forward Profiles" in result.report.to_markdown()


def test_project_calibrate_records_preserve_output_history(tmp_path: Path):
    torch.manual_seed(4)
    model = tiny_model()
    x = torch.randn(12, 4)
    loader = DataLoader(TensorDataset(x, torch.zeros(12, dtype=torch.long)), batch_size=4)
    project = arti.project(model).scan(x[:2]).insert(where="0")

    project.calibrate(loader, steps=2)
    result = project.fit()
    artifact = result.export(tmp_path / "calibrated.pt")
    payload = torch.load(artifact, weights_only=True)

    assert len(result.report.calibration_history) == 2
    assert result.report.calibration_objective == "preserve-output"
    assert payload["report"]["calibration_history"]


def test_fit_convenience_can_calibrate_before_training():
    torch.manual_seed(5)
    model = tiny_model()
    x = torch.randn(12, 4)
    y = torch.randint(0, 2, (12,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    result = arti.fit(
        model,
        train_loader=loader,
        calibration_loader=loader,
        calibration_steps=2,
        sample_batch=x[:2],
        target_modules="0",
        steps=1,
    )

    assert len(result.report.calibration_history) == 2
    assert len(result.report.loss_history) == 1
    assert result.report.calibration_objective == "preserve-output"


def test_fit_objective_plan_runs_calibrate_train_validate():
    torch.manual_seed(6)
    model = tiny_model()
    x = torch.randn(12, 4)
    y = torch.randint(0, 2, (12,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    result = arti.fit(
        model,
        train_loader=loader,
        calibration_loader=loader,
        val_loader=loader,
        sample_batch=x[:2],
        target_modules="0",
        objective=["preserve-output", "task-fit", "validate"],
        calibration_steps=1,
        steps=1,
        validation_steps=1,
    )

    assert result.report.objective_plan == ("preserve-output", "task-fit", "validate")
    assert len(result.report.calibration_history) == 1
    assert len(result.report.loss_history) == 1
    assert result.report.validation_history[0]["batches"] == 1.0
    assert result.report.to_dict()["objective_plan"] == ["preserve-output", "task-fit", "validate"]
    assert [task.name for task in result.report.task_history] == ["preserve-output", "task-fit", "validate"]
    assert [task.name for task in result.report.build_plan] == ["scan", "insert", "preserve-output", "task-fit", "validate"]
    assert result.report.build_plan[3].depends_on == ("preserve-output",)
    assert result.report.to_dict()["task_history"][-1]["metric_name"] == "mean_metric"
    assert result.report.to_dict()["build_plan"][1]["depends_on"] == ["scan"]
    assert "## Task History" in result.report.to_markdown()
    assert "## Build Plan" in result.report.to_markdown()


def test_fit_explicit_objective_requires_matching_loader():
    model = tiny_model()

    try:
        arti.fit(model, sample_batch=torch.randn(2, 4), target_modules="0", objective="validate")
    except ValueError as exc:
        assert "val_loader" in str(exc)
    else:
        raise AssertionError("validate objective should require val_loader")


def test_transformers_plugin_uses_named_insertion_strategy():
    model = tiny_transformer_like_model()

    project = arti.project(model).plugin("transformers").scan(torch.randn(2, 4)).insert(max_adapters=2)
    names = [adapter.name for adapter in project.report().inserted]

    assert "0.attn.out_proj" in names
    assert "0.mlp.fc2" in names
    assert "transformers" in project.report().plugins
    details = project.report().to_dict()["plugin_details"]
    assert any(detail["name"] == "transformers" and "attention-output-strategy" in detail["capabilities"] for detail in details)


def test_fit_transformer_profile_uses_transformer_strategy():
    model = tiny_transformer_like_model()

    result = arti.fit(model, sample_batch=torch.randn(2, 4), profile="transformer", max_adapters=1)

    assert result.adapter_count == 1
    assert result.report.inserted[0].name == "0.attn.out_proj"


def test_timm_plugin_uses_vision_transformer_strategy_and_artifact_rehydrates(tmp_path: Path):
    model = TinyTimmViT()
    sample = torch.randn(2, 3, 4)

    result = arti.fit(model, sample_batch=sample, profile="timm", max_adapters=2, freeze_base=True)
    artifact = result.export(tmp_path / "timm-adapter.pt")
    fresh = TinyTimmViT()
    applied = arti.apply_adapter(fresh, artifact, sample_batch=sample)

    names = [adapter.name for adapter in result.report.inserted]
    assert result.report.plugins == ("torch", "timm")
    assert result.report.to_dict()["plugin_details"][-1]["default_strategy"] == "vision-transformer"
    assert names == ["blocks.0.attn.proj", "blocks.0.mlp.fc2"]
    assert applied.adapter_count == 2
    assert fresh(sample).shape == (2, 3, 2)


def test_fit_respects_extra_parameter_budget():
    model = tiny_model()

    result = arti.fit(
        model,
        sample_batch=torch.randn(2, 4),
        target_modules=["0", "2"],
        max_extra_params=1,
        freeze_base=True,
    )

    assert result.adapter_count == 0
    assert result.report.adapter_parameters == 0
    assert result.report.summary.budget_exhausted is False
    assert result.report.summary.budget_limit == 1
    assert result.report.summary.inserted_count == 0


def test_fit_accepts_percent_parameter_budget():
    model = tiny_model()

    result = arti.fit(
        model,
        sample_batch=torch.randn(2, 4),
        target_modules=["0", "2"],
        max_extra_params="1%",
        freeze_base=True,
    )

    assert result.report.to_dict()["insertion"]["max_extra_params"] == 0
    assert result.adapter_count == 0


def test_apply_adapter_rehydrates_adapter_only_artifact(tmp_path: Path):
    torch.manual_seed(2)
    model = tiny_model()
    source = arti.fit(model, sample_batch=torch.randn(2, 4), target_modules="0", freeze_base=True)
    artifact = source.export(tmp_path / "adapter.pt")
    fresh = tiny_model()

    applied = arti.apply_adapter(fresh, artifact, sample_batch=torch.randn(2, 4))

    assert applied.adapter_count == 1
    assert isinstance(fresh[0], ARTIAdapterWrapper)
    payload = torch.load(artifact, weights_only=True)
    assert "adapter_state_dict" in payload
    assert applied.report.applied_artifact["path"] == str(artifact)
    assert applied.report.applied_artifact["adapter_state_sha256"] == payload["manifest"]["adapter_state_sha256"]
    assert applied.report.to_dict()["applied_artifact"]["adapter_key_count"] == payload["manifest"]["adapter_key_count"]
    assert "## Applied Artifact" in applied.report.to_markdown()
    assert fresh(torch.randn(2, 4)).shape == (2, 2)


def test_apply_adapter_reports_structure_mismatch(tmp_path: Path):
    source = arti.fit(tiny_model(), sample_batch=torch.randn(2, 4), target_modules="0", freeze_base=True)
    artifact = source.export(tmp_path / "adapter.pt")
    incompatible = nn.Sequential(nn.Linear(4, 2))

    try:
        arti.apply_adapter(incompatible, artifact, sample_batch=torch.randn(2, 4))
    except ValueError as exc:
        message = str(exc)
        assert "incompatible with the target model structure" in message
        assert "target_modules" in message
        assert "missing_adapter_keys" in message
    else:
        raise AssertionError("incompatible target model should fail adapter application")


def test_cli_apply_adapter_writes_application_report(tmp_path: Path, capsys, monkeypatch):
    from arti.cli import main

    module_path = tmp_path / "apply_fixture_model.py"
    module_path.write_text(
        "\n".join(
            [
                "import torch.nn as nn",
                "",
                "def make_model():",
                "    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    sample = torch.randn(2, 4)
    artifact = arti.fit(
        tiny_model(),
        sample_batch=sample,
        target_modules="0",
        freeze_base=True,
        mask_key="token_mask",
    ).export(tmp_path / "adapter.pt")
    lock_path = arti.create_build_lock(tmp_path / "arti.lock.json", artifact=artifact)
    report_path = tmp_path / "applied.json"
    state_path = tmp_path / "patched-state.pt"
    deployment_path = tmp_path / "deployment.json"
    apply_task_graph_path = tmp_path / "apply-task-graph.json"

    digest = torch.load(artifact, weights_only=True)["manifest"]["adapter_state_sha256"]
    assert main(
        [
            "apply",
            "apply_fixture_model:make_model",
            str(artifact),
            str(report_path),
            "--sample-shape",
            "2,4",
            "--expect-adapter-state-sha256",
            digest,
            "--lock",
            str(lock_path),
            "--max-adapters",
            "1",
            "--save-state-dict",
            str(state_path),
            "--deployment-output",
            str(deployment_path),
            "--task-graph-output",
            str(apply_task_graph_path),
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    payload = torch.load(artifact, weights_only=True)
    saved_state = torch.load(state_path, weights_only=True)

    assert output["kind"] == "applied-adapter-report"
    assert output["saved_state_dict"] == str(state_path)
    assert output["saved_state_dict_sha256"] == hash_tensor_state_dict(saved_state)
    assert output["deployment"] == str(deployment_path)
    assert output["deployment_summary"]["state_dict_sha256"] == output["saved_state_dict_sha256"]
    assert output["task_graph_output"] == str(apply_task_graph_path)
    assert apply_task_graph_path.exists()
    assert output["task_graph"]["artifacts"]["apply_report"] == str(report_path)
    assert output["task_graph"]["artifacts"]["state_dict"] == str(state_path)
    assert output["task_graph"]["artifacts"]["deployment"] == str(deployment_path)
    assert [task["name"] for task in output["task_graph"]["tasks"]] == [
        "apply-adapter",
        "write-apply-report",
        "write-state-dict",
        "write-deployment-manifest",
    ]
    assert output["task_graph"]["tasks"][-1]["depends_on"] == ["write-apply-report", "write-state-dict"]
    assert output["adapter_count"] == 1
    assert output["cli_max_adapters"] == 1
    assert output["lock"] == str(lock_path)
    assert output["lock_report_sha256"] == payload["manifest"]["report_sha256"]
    assert output["expected_adapter_state_sha256"] == digest
    assert output["adapter_state_sha256"] == payload["manifest"]["adapter_state_sha256"]
    assert report["applied_artifact"]["path"] == str(artifact)
    assert report["applied_artifact"]["adapter_state_sha256"] == payload["manifest"]["adapter_state_sha256"]
    assert state_path.exists()
    assert deployment_path.exists()
    assert any(".adapter." in key for key in saved_state)

    assert main(["validate", "artifact", str(artifact), "--expect-runtime-field", "mask_key=token_mask"]) == 0
    validated_artifact = json.loads(capsys.readouterr().out)
    assert validated_artifact["runtime"]["mask_key"] == "token_mask"
    assert validated_artifact["expected_runtime_fields"]["mask_key"] == "token_mask"

    assert main(["validate", "artifact", str(artifact), "--expect-runtime-field", "mask_key=attention_mask"]) == 1
    assert "runtime.mask_key" in capsys.readouterr().err

    assert main(["validate", "lock", str(lock_path), "--expect-runtime-field", "mask_key=token_mask"]) == 0
    validated_lock = json.loads(capsys.readouterr().out)
    assert validated_lock["runtime"]["mask_key"] == "token_mask"

    assert main(["validate", "state-dict", str(state_path), "--expect-state-dict-sha256", output["saved_state_dict_sha256"]]) == 0
    validate_state_output = json.loads(capsys.readouterr().out)
    assert validate_state_output["state_dict_sha256"] == output["saved_state_dict_sha256"]
    assert validate_state_output["expected_state_dict_sha256"] == output["saved_state_dict_sha256"]

    assert main(["validate", "task-graph", str(apply_task_graph_path), "--expect-kind", "apply", "--expect-artifact", f"deployment={deployment_path}", "--require-existing-artifacts"]) == 0
    validate_task_graph_output = json.loads(capsys.readouterr().out)
    assert validate_task_graph_output["command_kind"] == "apply"
    assert validate_task_graph_output["artifacts"]["deployment"] == str(deployment_path)
    assert validate_task_graph_output["missing_artifacts"] == []

    assert main(["validate", "state-dict", str(state_path), "--expect-state-dict-sha256", "wrong"]) == 1
    assert "state_dict_sha256" in capsys.readouterr().err

    assert main(
        [
            "validate",
            "deployment",
            str(deployment_path),
            "--expect-adapter-state-sha256",
            payload["manifest"]["adapter_state_sha256"],
            "--expect-state-dict-sha256",
            output["saved_state_dict_sha256"],
            "--expect-profile",
            "latent-adapt",
            "--expect-scale",
            "small",
            "--expect-mechanism",
            "recall_slots=4",
            "--expect-mechanism",
            "operator_count=4",
            "--expect-runtime-field",
            "mask_key=token_mask",
            "--max-adapters",
            "1",
            "--max-extra-params",
            "100000",
        ]
    ) == 0
    validated_deployment = json.loads(capsys.readouterr().out)
    assert validated_deployment["state_dict_sha256"] == output["saved_state_dict_sha256"]
    assert validated_deployment["expected_adapter_state_sha256"] == payload["manifest"]["adapter_state_sha256"]
    assert validated_deployment["expected_state_dict_sha256"] == output["saved_state_dict_sha256"]
    assert validated_deployment["expected_profile"] == "latent-adapt"
    assert validated_deployment["expected_scale"] == "small"
    assert validated_deployment["expected_mechanism"]["recall_slots"] == 4
    assert validated_deployment["mechanism"]["operator_count"] == 4
    assert validated_deployment["runtime"]["mask_key"] == "token_mask"
    assert validated_deployment["expected_runtime_fields"]["mask_key"] == "token_mask"
    assert validated_deployment["inserted_count"] == 1
    assert validated_deployment["cli_max_adapters"] == 1
    assert validated_deployment["cli_max_extra_params"] == 100000

    assert main(["validate", "deployment", str(deployment_path), "--expect-adapter-state-sha256", "wrong"]) == 1
    assert "adapter_state_sha256" in capsys.readouterr().err

    assert main(["validate", "deployment", str(deployment_path), "--expect-state-dict-sha256", "wrong"]) == 1
    assert "state_dict_sha256" in capsys.readouterr().err

    assert main(["validate", "deployment", str(deployment_path), "--expect-profile", "observer-phase"]) == 1
    assert "profile" in capsys.readouterr().err

    assert main(["validate", "deployment", str(deployment_path), "--expect-scale", "base"]) == 1
    assert "scale" in capsys.readouterr().err

    assert main(["validate", "deployment", str(deployment_path), "--expect-mechanism", "operator_count=8"]) == 1
    assert "mechanism.operator_count" in capsys.readouterr().err

    assert main(["validate", "deployment", str(deployment_path), "--expect-runtime-field", "mask_key=attention_mask"]) == 1
    assert "runtime.mask_key" in capsys.readouterr().err

    assert main(["validate", "deployment", str(deployment_path), "--max-adapters", "0"]) == 1
    assert "max_adapters" in capsys.readouterr().err

    assert main(["validate", "deployment", str(deployment_path), "--max-extra-params", "1"]) == 1
    assert "max_extra_params" in capsys.readouterr().err

    failed_state_path = tmp_path / "failed-state.pt"
    assert main(["apply", "apply_fixture_model:make_model", str(artifact), str(report_path), "--sample-shape", "2,4", "--max-adapters", "0", "--save-state-dict", str(failed_state_path)]) == 1
    assert not failed_state_path.exists()
    assert "max_adapters" in capsys.readouterr().err

    assert main(["apply", "apply_fixture_model:make_model", str(artifact), str(report_path), "--sample-shape", "2,4", "--save-state-dict", str(state_path), "--deployment-output", str(tmp_path / "missing-lock-deployment.json")]) == 1
    assert "requires --lock" in capsys.readouterr().err

    assert main(["apply", "apply_fixture_model:make_model", str(artifact), str(report_path), "--sample-shape", "2,4", "--lock", str(lock_path), "--deployment-output", str(tmp_path / "missing-state-deployment.json")]) == 1
    assert "requires --save-state-dict" in capsys.readouterr().err

    assert main(["apply", "apply_fixture_model:make_model", str(artifact), str(report_path), "--sample-shape", "2,4", "--max-adapters", "0"]) == 1
    assert "max_adapters" in capsys.readouterr().err

    assert main(["apply", "apply_fixture_model:make_model", str(artifact), str(report_path), "--sample-shape", "2,4", "--expect-adapter-state-sha256", "wrong"]) == 1
    assert "adapter_state_sha256" in capsys.readouterr().err

    other_artifact = arti.fit(tiny_model(), sample_batch=sample, target_modules="2", freeze_base=True).export(tmp_path / "other-adapter.pt")
    assert main(["apply", "apply_fixture_model:make_model", str(other_artifact), str(report_path), "--sample-shape", "2,4", "--lock", str(lock_path)]) == 1
    assert "build lock artifact path" in capsys.readouterr().err


def test_cli_apply_adapter_can_require_matching_config(tmp_path: Path, capsys, monkeypatch):
    from arti.cli import main

    module_path = tmp_path / "apply_config_fixture_model.py"
    module_path.write_text(
        "\n".join(
            [
                "import torch.nn as nn",
                "",
                "def make_model():",
                "    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    config_path = tmp_path / "arti.json"
    config_path.write_text(json.dumps({"insertion": {"where": "0"}}), encoding="utf-8")
    other_path = tmp_path / "other.json"
    other_path.write_text(json.dumps({"insertion": {"where": "2"}}), encoding="utf-8")
    artifact = arti.fit(tiny_model(), config=config_path, sample_batch=torch.randn(2, 4)).export(tmp_path / "adapter.pt")
    report_path = tmp_path / "applied.json"
    config = arti.load_fit_config(config_path)

    assert main(["apply", "apply_config_fixture_model:make_model", str(artifact), str(report_path), "--sample-shape", "2,4", "--expect-config", str(config_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["config_fingerprint"] == config.fingerprint
    assert output["expected_config_fingerprint"] == config.fingerprint

    assert main(["apply", "apply_config_fixture_model:make_model", str(artifact), str(report_path), "--sample-shape", "2,4", "--expect-config", str(other_path)]) == 1
    assert "expected config" in capsys.readouterr().err


def test_export_includes_task_history(tmp_path: Path):
    torch.manual_seed(7)
    model = tiny_model()
    x = torch.randn(8, 4)
    y = torch.randint(0, 2, (8,))
    loader = DataLoader(TensorDataset(x, y), batch_size=4)

    result = arti.fit(
        model,
        train_loader=loader,
        sample_batch=x[:2],
        target_modules="0",
        objective="task-fit",
        steps=1,
    )
    artifact = result.export(tmp_path / "task-artifact.pt")
    payload = torch.load(artifact, weights_only=True)

    assert payload["report"]["task_history"][0]["name"] == "task-fit"
    assert payload["report"]["task_history"][0]["status"] == "success"
    assert payload["report"]["mechanism"]["scale"] == "small"
    assert payload["report"]["mechanism"]["recall_slots"] == 4
    assert [task["name"] for task in payload["report"]["build_plan"]] == ["scan", "insert", "task-fit"]


def test_export_manifest_records_include_base(tmp_path: Path):
    model = tiny_model()
    result = arti.fit(model, sample_batch=torch.randn(2, 4), target_modules="0")

    artifact = result.export(tmp_path / "with-base.pt", include_base=True)
    payload = torch.load(artifact, weights_only=True)

    assert payload["manifest"]["include_base"] is True
    assert payload["manifest"]["adapter_parameters"] == result.report.adapter_parameters
    assert payload["manifest"]["profile"] == result.report.profile
    assert "state_dict" in payload


def test_validate_artifact_accepts_exported_payload(tmp_path: Path):
    model = tiny_model()
    result = arti.fit(model, sample_batch=torch.randn(2, 4), target_modules="0")
    artifact = result.export(tmp_path / "valid.pt")

    payload = arti.validate_artifact(artifact)

    assert payload["manifest"]["backend"] == "torch"
    assert payload["report"]["adapter_parameters"] == result.report.adapter_parameters
    assert payload["report"]["summary"]["candidate_count"] >= 1


def test_report_summary_is_ci_friendly():
    result = arti.fit(tiny_model(), sample_batch=torch.randn(2, 4), target_modules="0")

    summary = result.report.summary.to_dict()

    assert summary["candidate_count"] >= 1
    assert summary["inserted_count"] == 1
    assert summary["adapter_parameters"] == result.report.adapter_parameters
    assert summary["adapter_parameter_ratio"] > 0
    assert "## Summary" in result.report.to_markdown()


def test_validate_artifact_rejects_manifest_mismatch(tmp_path: Path):
    model = tiny_model()
    result = arti.fit(model, sample_batch=torch.randn(2, 4), target_modules="0")
    artifact = result.export(tmp_path / "invalid.pt")
    payload = torch.load(artifact, weights_only=True)
    payload["manifest"]["adapter_key_count"] += 1
    torch.save(payload, artifact)

    try:
        arti.validate_artifact(artifact)
    except ValueError as exc:
        assert "adapter_key_count" in str(exc)
    else:
        raise AssertionError("invalid artifact manifest should fail validation")


def test_validate_artifact_rejects_invalid_build_metadata(tmp_path: Path):
    model = tiny_model()
    result = arti.fit(model, sample_batch=torch.randn(2, 4), target_modules="0")
    artifact = result.export(tmp_path / "invalid-build.pt", build_metadata={"expected_plan": "plan.json"})
    payload = torch.load(artifact, weights_only=True)
    payload["build"] = "not-a-dict"
    torch.save(payload, artifact)

    try:
        arti.validate_artifact(artifact)
    except ValueError as exc:
        assert "build metadata" in str(exc)
    else:
        raise AssertionError("invalid artifact build metadata should fail validation")


def test_validate_artifact_rejects_missing_package_version(tmp_path: Path):
    model = tiny_model()
    result = arti.fit(model, sample_batch=torch.randn(2, 4), target_modules="0")
    artifact = result.export(tmp_path / "invalid-version.pt")
    payload = torch.load(artifact, weights_only=True)
    payload["manifest"]["package_version"] = ""
    torch.save(payload, artifact)

    with pytest.raises(ValueError, match="package_version"):
        arti.validate_artifact(artifact)


def test_validate_artifact_rejects_invalid_manifest_hash_shape(tmp_path: Path):
    model = tiny_model()
    result = arti.fit(model, sample_batch=torch.randn(2, 4), target_modules="0")
    artifact = result.export(tmp_path / "invalid-hash-shape.pt")
    payload = torch.load(artifact, weights_only=True)
    payload["manifest"]["adapter_state_sha256"] = "not-a-sha256"
    torch.save(payload, artifact)

    with pytest.raises(ValueError, match="64-character lowercase sha256"):
        arti.validate_artifact(artifact)


def test_validate_artifact_rejects_report_summary_mismatch(tmp_path: Path):
    model = tiny_model()
    result = arti.fit(model, sample_batch=torch.randn(2, 4), target_modules="0")
    artifact = result.export(tmp_path / "invalid-summary.pt")
    payload = torch.load(artifact, weights_only=True)
    payload["report"]["summary"]["inserted_count"] += 1
    payload["manifest"]["report_sha256"] = stable_json_sha256(payload["report"])
    torch.save(payload, artifact)

    with pytest.raises(ValueError, match="inserted_count"):
        arti.validate_artifact(artifact)


def test_task_graph_public_api_writes_and_validates(tmp_path: Path):
    import arti.torch as arti_torch

    graph = {
        "tasks": [
            {"name": "scan", "kind": "scan", "depends_on": [], "enabled": True},
            {"name": "insert", "kind": "insert", "depends_on": ["scan"], "enabled": True},
        ],
        "artifacts": {"plan": "plan.json"},
    }

    payload = arti.create_task_graph_payload(command_kind="build", task_graph=graph)
    path = arti.write_task_graph_artifact(tmp_path / "task-graph.json", command_kind="build", task_graph=graph)
    loaded = arti.validate_task_graph(path)

    assert payload["kind"] == "task-graph"
    assert loaded["command_kind"] == "build"
    assert loaded["task_graph"]["artifacts"]["plan"] == "plan.json"
    assert arti_torch.validate_task_graph(path)["task_graph"]["tasks"][1]["depends_on"] == ["scan"]
    bad_kind = dict(payload)
    bad_kind["command_kind"] = "deploy"
    with pytest.raises(ValueError, match="command_kind"):
        arti.validate_task_graph_payload(bad_kind)
    bad_artifact = arti.create_task_graph_payload(command_kind="build", task_graph=graph)
    bad_artifact["task_graph"]["artifacts"]["plan"] = 123
    with pytest.raises(ValueError, match="artifact values"):
        arti.validate_task_graph_payload(bad_artifact)


def test_validate_artifact_rejects_config_fingerprint_mismatch(tmp_path: Path):
    model = tiny_model()
    result = arti.fit(model, sample_batch=torch.randn(2, 4), target_modules="0")
    artifact = result.export(tmp_path / "invalid-config.pt")
    payload = torch.load(artifact, weights_only=True)
    payload["manifest"]["config_fingerprint"] = "wrong"
    torch.save(payload, artifact)

    try:
        arti.validate_artifact(artifact)
    except ValueError as exc:
        assert "config_fingerprint" in str(exc)
    else:
        raise AssertionError("invalid config fingerprint should fail validation")


def test_validate_artifact_rejects_adapter_state_hash_mismatch(tmp_path: Path):
    model = tiny_model()
    result = arti.fit(model, sample_batch=torch.randn(2, 4), target_modules="0")
    artifact = result.export(tmp_path / "invalid-hash.pt")
    payload = torch.load(artifact, weights_only=True)
    key = next(iter(payload["adapter_state_dict"]))
    payload["adapter_state_dict"][key] = payload["adapter_state_dict"][key] + 1
    torch.save(payload, artifact)

    try:
        arti.validate_artifact(artifact)
    except ValueError as exc:
        assert "adapter_state_sha256" in str(exc)
    else:
        raise AssertionError("tampered adapter state should fail validation")


def test_validate_artifact_rejects_report_hash_mismatch(tmp_path: Path):
    model = tiny_model()
    result = arti.fit(model, sample_batch=torch.randn(2, 4), target_modules="0")
    artifact = result.export(tmp_path / "invalid-report.pt")
    payload = torch.load(artifact, weights_only=True)
    payload["report"]["scale"] = "tampered"
    torch.save(payload, artifact)

    try:
        arti.validate_artifact(artifact)
    except ValueError as exc:
        assert "report_sha256" in str(exc)
    else:
        raise AssertionError("tampered report should fail validation")


def test_build_lock_validates_artifact_plan_and_config(tmp_path: Path):
    config_path = tmp_path / "arti.json"
    config_path.write_text(json.dumps({"insertion": {"where": "0"}}), encoding="utf-8")
    config = arti.load_fit_config(config_path)
    sample = torch.randn(2, 4)
    plan_path = arti.project(tiny_model()).configure(config).scan(sample).write_plan(tmp_path / "plan.json")
    artifact = arti.fit(tiny_model(), config=config, sample_batch=sample).export(tmp_path / "adapter.pt")
    lock_path = arti.create_build_lock(tmp_path / "arti.lock.json", artifact=artifact, plan=plan_path, config=config_path)

    payload = arti.validate_build_lock(lock_path)

    assert payload["kind"] == "build-lock"
    assert payload["artifact"]["adapter_state_sha256"] == torch.load(artifact, weights_only=True)["manifest"]["adapter_state_sha256"]
    assert payload["artifact"]["report_sha256"] == torch.load(artifact, weights_only=True)["manifest"]["report_sha256"]
    assert payload["plan"]["config_fingerprint"] == config.fingerprint
    assert payload["config"]["config_fingerprint"] == config.fingerprint


def test_build_lock_carries_artifact_build_metadata(tmp_path: Path):
    sample = torch.randn(2, 4)
    plan_path = arti.project(tiny_model()).scan(sample).write_plan(tmp_path / "plan.json", where="0")
    build_metadata = {
        "expected_plan": str(plan_path),
        "expected_plan_selected": ["0"],
    }
    artifact = arti.fit(tiny_model(), sample_batch=sample, target_modules="0").export(
        tmp_path / "adapter.pt",
        build_metadata=build_metadata,
    )
    lock_path = arti.create_build_lock(tmp_path / "arti.lock.json", artifact=artifact, plan=plan_path)

    payload = arti.validate_build_lock(lock_path)

    assert payload["artifact"]["build"] == build_metadata


def test_build_lock_stores_paths_relative_to_lockfile(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    config_path = artifacts_dir / "arti.json"
    config_path.write_text(json.dumps({"insertion": {"where": "0"}}), encoding="utf-8")
    sample = torch.randn(2, 4)
    plan_path = arti.project(tiny_model()).configure(arti.load_fit_config(config_path)).scan(sample).write_plan(artifacts_dir / "plan.json")
    artifact = arti.fit(tiny_model(), config=config_path, sample_batch=sample).export(artifacts_dir / "adapter.pt")

    lock_path = arti.create_build_lock(
        Path("artifacts") / "arti.lock.json",
        artifact=Path("artifacts") / "adapter.pt",
        plan=Path("artifacts") / "plan.json",
        config=Path("artifacts") / "arti.json",
    )
    payload = json.loads(lock_path.read_text(encoding="utf-8"))

    assert payload["artifact"]["path"] == "adapter.pt"
    assert payload["plan"]["path"] == "plan.json"
    assert payload["config"]["path"] == "arti.json"
    assert arti.validate_build_lock(lock_path)["artifact"]["adapter_state_sha256"] == torch.load(artifact, weights_only=True)["manifest"]["adapter_state_sha256"]


def test_validate_build_lock_rejects_changed_artifact(tmp_path: Path):
    sample = torch.randn(2, 4)
    artifact = arti.fit(tiny_model(), sample_batch=sample, target_modules="0").export(tmp_path / "adapter.pt")
    lock_path = arti.create_build_lock(tmp_path / "arti.lock.json", artifact=artifact)
    payload = torch.load(artifact, weights_only=True)
    key = next(iter(payload["adapter_state_dict"]))
    payload["adapter_state_dict"][key] = payload["adapter_state_dict"][key] + 1
    payload["manifest"]["adapter_state_sha256"] = stable_json_sha256({"fake": "hash"})
    torch.save(payload, artifact)

    try:
        arti.validate_build_lock(lock_path)
    except ValueError as exc:
        assert "adapter_state_sha256" in str(exc)
    else:
        raise AssertionError("changed artifact should fail build lock validation")


def test_validate_build_lock_rejects_changed_artifact_build_metadata(tmp_path: Path):
    sample = torch.randn(2, 4)
    artifact = arti.fit(tiny_model(), sample_batch=sample, target_modules="0").export(
        tmp_path / "adapter.pt",
        build_metadata={"expected_plan": "plan.json", "expected_plan_selected": ["0"]},
    )
    lock_path = arti.create_build_lock(tmp_path / "arti.lock.json", artifact=artifact)
    payload = torch.load(artifact, weights_only=True)
    payload["build"]["expected_plan_selected"] = ["2"]
    torch.save(payload, artifact)

    try:
        arti.validate_build_lock(lock_path)
    except ValueError as exc:
        assert "artifact.build" in str(exc)
    else:
        raise AssertionError("changed artifact build metadata should fail build lock validation")


def test_validate_build_lock_rejects_changed_inserted_count(tmp_path: Path):
    sample = torch.randn(2, 4)
    artifact = arti.fit(tiny_model(), sample_batch=sample, target_modules="0").export(tmp_path / "adapter.pt")
    lock_path = arti.create_build_lock(tmp_path / "arti.lock.json", artifact=artifact)
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    payload["artifact"]["inserted_count"] += 1
    lock_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="inserted_count"):
        arti.validate_build_lock(lock_path)


def test_deployment_manifest_carries_artifact_build_metadata(tmp_path: Path):
    sample = torch.randn(2, 4)
    artifact = arti.fit(tiny_model(), sample_batch=sample, target_modules="0").export(
        tmp_path / "adapter.pt",
        build_metadata={"expected_plan": "plan.json", "expected_plan_selected": ["0"]},
    )
    lock_path = arti.create_build_lock(tmp_path / "arti.lock.json", artifact=artifact)
    applied_report = tmp_path / "applied.json"
    state_path = tmp_path / "patched-state.pt"
    applied_report.write_text(json.dumps({"applied_artifact": {"path": str(artifact)}}), encoding="utf-8")
    torch.save(tiny_model().state_dict(), state_path)

    manifest_path = arti.create_deployment_manifest(
        tmp_path / "deployment.json",
        lock=lock_path,
        artifact=artifact,
        applied_report=applied_report,
        state_dict=state_path,
    )
    payload = arti.validate_deployment_manifest(manifest_path)

    assert payload["artifact"]["build"] == {"expected_plan": "plan.json", "expected_plan_selected": ["0"]}
    assert payload["artifact"]["adapter_key_count"] == torch.load(artifact, weights_only=True)["manifest"]["adapter_key_count"]


def test_deployment_manifest_rejects_changed_adapter_key_count(tmp_path: Path):
    sample = torch.randn(2, 4)
    artifact = arti.fit(tiny_model(), sample_batch=sample, target_modules="0").export(tmp_path / "adapter.pt")
    lock_path = arti.create_build_lock(tmp_path / "arti.lock.json", artifact=artifact)
    applied_report = tmp_path / "applied.json"
    state_path = tmp_path / "patched-state.pt"
    applied_report.write_text(json.dumps({"applied_artifact": {"path": str(artifact)}}), encoding="utf-8")
    torch.save(tiny_model().state_dict(), state_path)
    manifest_path = arti.create_deployment_manifest(
        tmp_path / "deployment.json",
        lock=lock_path,
        artifact=artifact,
        applied_report=applied_report,
        state_dict=state_path,
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["artifact"]["adapter_key_count"] += 1
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="adapter_key_count"):
        arti.validate_deployment_manifest(manifest_path)


def test_deployment_manifest_rejects_changed_state_dict(tmp_path: Path):
    sample = torch.randn(2, 4)
    artifact = arti.fit(tiny_model(), sample_batch=sample, target_modules="0").export(tmp_path / "adapter.pt")
    lock_path = arti.create_build_lock(tmp_path / "arti.lock.json", artifact=artifact)
    applied_report = tmp_path / "applied.json"
    state_path = tmp_path / "patched-state.pt"
    state = tiny_model().state_dict()
    applied_report.write_text(json.dumps({"applied_artifact": {"path": str(artifact)}}), encoding="utf-8")
    torch.save(state, state_path)
    manifest_path = arti.create_deployment_manifest(
        tmp_path / "deployment.json",
        lock=lock_path,
        artifact=artifact,
        applied_report=applied_report,
        state_dict=state_path,
    )
    state[next(iter(state))] = state[next(iter(state))] + 1
    torch.save(state, state_path)

    try:
        arti.validate_deployment_manifest(manifest_path)
    except ValueError as exc:
        assert "state_dict_sha256" in str(exc)
    else:
        raise AssertionError("changed deployment state_dict should fail manifest validation")


def test_deployment_manifest_rejects_artifact_not_approved_by_lock(tmp_path: Path):
    sample = torch.randn(2, 4)
    artifact = arti.fit(tiny_model(), sample_batch=sample, target_modules="0").export(tmp_path / "adapter.pt")
    other_artifact = arti.fit(tiny_model(), sample_batch=sample, target_modules="2").export(tmp_path / "other-adapter.pt")
    lock_path = arti.create_build_lock(tmp_path / "arti.lock.json", artifact=artifact)
    applied_report = tmp_path / "applied.json"
    state_path = tmp_path / "patched-state.pt"
    applied_report.write_text(json.dumps({"applied_artifact": {"path": str(other_artifact)}}), encoding="utf-8")
    torch.save(tiny_model().state_dict(), state_path)
    manifest_path = arti.create_deployment_manifest(
        tmp_path / "deployment.json",
        lock=lock_path,
        artifact=artifact,
        applied_report=applied_report,
        state_dict=state_path,
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    other_payload = torch.load(other_artifact, weights_only=True)
    payload["artifact"]["path"] = "other-adapter.pt"
    payload["artifact"]["adapter_state_sha256"] = other_payload["manifest"]["adapter_state_sha256"]
    payload["artifact"]["report_sha256"] = other_payload["manifest"]["report_sha256"]
    payload["artifact"]["config_fingerprint"] = other_payload["manifest"]["config_fingerprint"]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        arti.validate_deployment_manifest(manifest_path)
    except ValueError as exc:
        assert "does not match lock" in str(exc)
    else:
        raise AssertionError("deployment artifact not approved by lock should fail validation")


def test_deployment_manifest_rejects_applied_report_artifact_mismatch(tmp_path: Path):
    sample = torch.randn(2, 4)
    artifact = arti.fit(tiny_model(), sample_batch=sample, target_modules="0").export(tmp_path / "adapter.pt")
    lock_path = arti.create_build_lock(tmp_path / "arti.lock.json", artifact=artifact)
    applied_report = tmp_path / "applied.json"
    state_path = tmp_path / "patched-state.pt"
    applied_report.write_text(json.dumps({"applied_artifact": {"path": str(artifact), "adapter_state_sha256": "wrong"}}), encoding="utf-8")
    torch.save(tiny_model().state_dict(), state_path)
    manifest_path = arti.create_deployment_manifest(
        tmp_path / "deployment.json",
        lock=lock_path,
        artifact=artifact,
        applied_report=applied_report,
        state_dict=state_path,
    )

    try:
        arti.validate_deployment_manifest(manifest_path)
    except ValueError as exc:
        assert "applied_report adapter_state_sha256" in str(exc)
    else:
        raise AssertionError("applied report artifact mismatch should fail deployment validation")


def test_deployment_manifest_rejects_changed_artifact_build_metadata(tmp_path: Path):
    sample = torch.randn(2, 4)
    artifact = arti.fit(tiny_model(), sample_batch=sample, target_modules="0").export(
        tmp_path / "adapter.pt",
        build_metadata={"expected_plan": "plan.json", "expected_plan_selected": ["0"]},
    )
    lock_path = arti.create_build_lock(tmp_path / "arti.lock.json", artifact=artifact)
    applied_report = tmp_path / "applied.json"
    state_path = tmp_path / "patched-state.pt"
    applied_report.write_text(json.dumps({"applied_artifact": {"path": str(artifact)}}), encoding="utf-8")
    torch.save(tiny_model().state_dict(), state_path)
    manifest_path = arti.create_deployment_manifest(
        tmp_path / "deployment.json",
        lock=lock_path,
        artifact=artifact,
        applied_report=applied_report,
        state_dict=state_path,
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["artifact"]["build"]["expected_plan_selected"] = ["2"]
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        arti.validate_deployment_manifest(manifest_path)
    except ValueError as exc:
        assert "artifact.build" in str(exc)
    else:
        raise AssertionError("changed deployment build metadata should fail deployment validation")


def test_validate_plan_rejects_inconsistent_adapter_parameters(tmp_path: Path):
    model = tiny_model()
    plan_path = arti.project(model).scan(torch.randn(2, 4)).write_plan(tmp_path / "plan.json", where="0")
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["report"]["insertion_plan"]["adapter_parameters"] += 1
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        arti.validate_plan(plan_path)
    except ValueError as exc:
        assert "adapter_parameters" in str(exc)
    else:
        raise AssertionError("invalid fit plan should fail validation")


def test_validate_plan_rejects_config_fingerprint_mismatch(tmp_path: Path):
    plan_path = arti.project(tiny_model()).scan(torch.randn(2, 4)).write_plan(tmp_path / "bad-config-plan.json", where="0")
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["report"]["config_fingerprint"] = "wrong"
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        arti.validate_plan(plan_path)
    except ValueError as exc:
        assert "config_fingerprint" in str(exc)
    else:
        raise AssertionError("invalid fit plan fingerprint should fail validation")


def test_cli_validates_plan_and_artifact(tmp_path: Path, capsys):
    from arti.cli import main

    model = tiny_model()
    sample = torch.randn(2, 4)
    plan_path = arti.project(model).scan(sample).write_plan(tmp_path / "plan.json", where="0")
    artifact_path = arti.fit(tiny_model(), sample_batch=sample, target_modules="0").export(tmp_path / "adapter.pt")

    assert main(["validate", "plan", str(plan_path)]) == 0
    plan_output = capsys.readouterr()
    assert main(["validate", "artifact", str(artifact_path)]) == 0
    artifact_output = capsys.readouterr()

    assert json.loads(plan_output.out)["kind"] == "fit-plan"
    assert json.loads(plan_output.out)["planned_count"] == 1
    assert json.loads(plan_output.out)["budget_limit"] is None
    assert json.loads(plan_output.out)["skipped_budget_count"] == 0
    assert json.loads(artifact_output.out)["kind"] == "adapter-artifact"
    assert json.loads(artifact_output.out)["inserted_count"] == 1


def test_cli_plan_creates_dry_run_plan_from_importable_factory(tmp_path: Path, capsys, monkeypatch):
    from arti.cli import main

    module_path = tmp_path / "fixture_model.py"
    module_path.write_text(
        "\n".join(
            [
                "import torch.nn as nn",
                "",
                "def make_model():",
                "    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    plan_path = tmp_path / "plan.json"

    assert main(
        [
            "plan",
            "fixture_model:make_model",
            str(plan_path),
            "--sample-shape",
            "2,4",
            "--target-modules",
            "0",
            "--max-adapters",
            "1",
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))

    assert output["kind"] == "fit-plan"
    assert output["planned_count"] == 1
    assert output["output"] == str(plan_path)
    assert output["provenance"]["model"] == "fixture_model:make_model"
    assert output["provenance"]["sample_shape"] == [2, 4]
    assert output["provenance_fingerprint"] == arti.plan_provenance_fingerprint(output["provenance"])
    assert payload["kind"] == "fit-plan"
    assert payload["provenance"]["target_modules"] == ["0"]
    assert payload["provenance_fingerprint"] == output["provenance_fingerprint"]
    assert payload["report"]["insertion_plan"]["selected"][0]["name"] == "0"
    assert payload["report"]["summary"]["inserted_count"] == 0


def test_cli_build_exports_adapter_artifact_from_importable_factory(tmp_path: Path, capsys, monkeypatch):
    from arti.cli import main

    module_path = tmp_path / "build_fixture_model.py"
    module_path.write_text(
        "\n".join(
            [
                "import torch.nn as nn",
                "",
                "def make_model():",
                "    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    plan_path = tmp_path / "plan.json"
    artifact_path = tmp_path / "adapter.pt"
    report_path = tmp_path / "report.json"
    lock_path = tmp_path / "arti.lock.json"
    build_task_graph_path = tmp_path / "build-task-graph.json"

    assert main(
        [
            "plan",
            "build_fixture_model:make_model",
            str(plan_path),
            "--sample-shape",
            "2,4",
            "--target-modules",
            "0",
            "--max-adapters",
            "1",
        ]
    ) == 0
    capsys.readouterr()

    assert main(
        [
            "build",
            "build_fixture_model:make_model",
            str(artifact_path),
            "--sample-shape",
            "2,4",
            "--target-modules",
            "0",
            "--max-adapters",
            "1",
            "--report",
            str(report_path),
            "--lock-output",
            str(lock_path),
            "--task-graph-output",
            str(build_task_graph_path),
            "--expect-plan",
            str(plan_path),
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    payload = arti.validate_artifact(artifact_path)
    plan_payload = json.loads(plan_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert output["kind"] == "adapter-artifact"
    assert output["artifact"] == str(artifact_path)
    assert output["report"] == str(report_path)
    assert output["lock"] == str(lock_path)
    assert output["lock_summary"]["build"]["expected_plan_selected"] == ["0"]
    assert output["task_graph_output"] == str(build_task_graph_path)
    assert build_task_graph_path.exists()
    assert output["task_graph"]["artifacts"]["adapter"] == str(artifact_path)
    assert output["task_graph"]["artifacts"]["report"] == str(report_path)
    assert output["task_graph"]["artifacts"]["lock"] == str(lock_path)
    assert [task["name"] for task in output["task_graph"]["tasks"]][-3:] == ["export-artifact", "write-report", "write-lock"]
    assert output["task_graph"]["tasks"][-1]["depends_on"] == ["export-artifact"]
    assert output["adapter_count"] == 1
    assert output["adapter_state_sha256"] == payload["manifest"]["adapter_state_sha256"]
    assert output["report_sha256"] == payload["manifest"]["report_sha256"]
    assert output["expected_plan"] == str(plan_path)
    assert output["expected_plan_provenance_fingerprint"] == plan_payload["provenance_fingerprint"]
    assert output["build"]["expected_plan_selected"] == ["0"]
    assert payload["build"]["expected_plan"] == str(plan_path)
    assert payload["build"]["expected_plan_selected"] == ["0"]
    assert payload["build"]["expected_plan_provenance_fingerprint"] == plan_payload["provenance_fingerprint"]
    assert payload["build"]["expected_plan_config_fingerprint"] == plan_payload["report"]["config_fingerprint"]
    assert report["summary"]["inserted_count"] == 1

    assert main(["validate", "task-graph", str(build_task_graph_path), "--expect-kind", "build", "--expect-artifact", f"adapter={artifact_path}", "--require-existing-artifacts"]) == 0
    validated_task_graph = json.loads(capsys.readouterr().out)
    assert validated_task_graph["command_kind"] == "build"
    assert validated_task_graph["artifacts"]["adapter"] == str(artifact_path)
    assert validated_task_graph["missing_artifacts"] == []
    task_graph_payload = json.loads(build_task_graph_path.read_text(encoding="utf-8"))
    task_graph_payload["task_graph"]["artifacts"]["adapter"] = str(tmp_path / "missing-adapter.pt")
    build_task_graph_path.write_text(json.dumps(task_graph_payload), encoding="utf-8")
    assert main(["validate", "task-graph", str(build_task_graph_path), "--require-existing-artifacts"]) == 1
    assert "artifacts are missing" in capsys.readouterr().err

    assert main(["validate", "artifact", str(artifact_path), "--max-adapters", "1", "--expect-plan", str(plan_path)]) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["inserted_count"] == 1
    assert validated["expected_plan"] == str(plan_path)
    assert validated["build"]["expected_plan_selected"] == ["0"]

    assert main(["validate", "lock", str(lock_path), "--expect-plan", str(plan_path)]) == 0
    validated_lock = json.loads(capsys.readouterr().out)
    assert validated_lock["expected_plan"] == str(plan_path)
    assert validated_lock["build"]["expected_plan_selected"] == ["0"]

    applied_report_path = tmp_path / "applied.json"
    state_path = tmp_path / "patched-state.pt"
    applied_report_path.write_text(json.dumps({"applied_artifact": {"path": str(artifact_path), "adapter_state_sha256": payload["manifest"]["adapter_state_sha256"]}}), encoding="utf-8")
    torch.save(tiny_model().state_dict(), state_path)
    deployment_path = tmp_path / "deployment.json"
    assert main(
        [
            "deployment-manifest",
            str(deployment_path),
            "--lock",
            str(lock_path),
            "--artifact",
            str(artifact_path),
            "--applied-report",
            str(applied_report_path),
            "--state-dict",
            str(state_path),
        ]
    ) == 0
    capsys.readouterr()
    assert main(["validate", "deployment", str(deployment_path), "--expect-plan", str(plan_path)]) == 0
    validated_deployment = json.loads(capsys.readouterr().out)
    assert validated_deployment["expected_plan"] == str(plan_path)
    assert validated_deployment["build"]["expected_plan_selected"] == ["0"]

    mismatched_plan_path = tmp_path / "mismatched-plan.json"
    assert main(
        [
            "plan",
            "build_fixture_model:make_model",
            str(mismatched_plan_path),
            "--sample-shape",
            "2,4",
            "--target-modules",
            "2",
            "--max-adapters",
            "1",
        ]
    ) == 0
    capsys.readouterr()
    assert main(
        [
            "build",
            "build_fixture_model:make_model",
            str(tmp_path / "mismatched.pt"),
            "--sample-shape",
            "2,4",
            "--target-modules",
            "0",
            "--max-adapters",
            "1",
            "--expect-plan",
            str(mismatched_plan_path),
        ]
    ) == 1
    assert "expected plan" in capsys.readouterr().err
    assert main(["validate", "artifact", str(artifact_path), "--expect-plan", str(mismatched_plan_path)]) == 1
    assert "expected plan" in capsys.readouterr().err
    assert main(["validate", "lock", str(lock_path), "--expect-plan", str(mismatched_plan_path)]) == 1
    assert "expected plan" in capsys.readouterr().err
    assert main(["validate", "deployment", str(deployment_path), "--expect-plan", str(mismatched_plan_path)]) == 1
    assert "expected plan" in capsys.readouterr().err


def test_cli_plan_accepts_mechanism_overrides(tmp_path: Path, capsys, monkeypatch):
    from arti.cli import main

    module_path = tmp_path / "mechanism_fixture_model.py"
    module_path.write_text(
        "\n".join(
            [
                "import torch.nn as nn",
                "",
                "def make_model():",
                "    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    plan_path = tmp_path / "mechanism-plan.json"

    assert main(
        [
            "plan",
            "mechanism_fixture_model:make_model",
            str(plan_path),
            "--sample-shape",
            "2,4",
            "--target-modules",
            "0",
            "--profile",
            "observer-phase",
            "--phases",
            "6",
            "--mechanism-coord-dim",
            "4",
            "--mechanism-coord-frame-mode",
            "operator_bank",
            "--mechanism-observer-phase",
            "--mechanism-virtual-recall",
            "--mechanism-operator-count",
            "5",
            "--mechanism-interface-slots",
            "6",
            "--mechanism-recall-slots",
            "2",
            "--mechanism-recall-steps",
            "1",
            "--mechanism-hidden-multiplier",
            "1.5",
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    mechanism = payload["report"]["mechanism"]

    assert mechanism["coord_dim"] == 4
    assert mechanism["operator_count"] == 5
    assert mechanism["interface_slots"] == 6
    assert mechanism["recall_slots"] == 2
    assert mechanism["recall_steps"] == 1
    assert mechanism["hidden_multiplier"] == 1.5
    assert payload["provenance"]["phases"] == 6
    assert payload["provenance"]["mechanism"]["operator_count"] == 5
    assert output["provenance_fingerprint"] == arti.plan_provenance_fingerprint(output["provenance"])


def test_cli_plan_accepts_runtime_field_overrides(tmp_path: Path, capsys, monkeypatch):
    from arti.cli import main

    module_path = tmp_path / "runtime_field_fixture_model.py"
    module_path.write_text(
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class DictModel(nn.Module):",
                "    def __init__(self):",
                "        super().__init__()",
                "        self.proj = nn.Linear(4, 4)",
                "",
                "    def forward(self, x: torch.Tensor):",
                "        return self.proj(x)",
                "",
                "def make_model():",
                "    return DictModel()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    sample_path = tmp_path / "sample.json"
    sample_path.write_text(
        json.dumps(
            {
                "fields": {
                    "x": {"shape": [2, 3, 4], "dtype": "float32", "kind": "randn"},
                    "token_mask": {"shape": [2, 3], "dtype": "long", "kind": "ones"},
                    "phase_coord": {"shape": [2, 3, 2], "dtype": "float32", "kind": "randn"},
                    "observer_phase": {"shape": [2, 2], "dtype": "float32", "kind": "randn"},
                    "inverse_ops": {"shape": [2, 4, 4], "dtype": "float32", "kind": "randn"},
                }
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    plan_path = tmp_path / "runtime-field-plan.json"

    assert main(
        [
            "plan",
            "runtime_field_fixture_model:make_model",
            str(plan_path),
            "--sample-json",
            str(sample_path),
            "--target-modules",
            "proj",
            "--profile",
            "observer-phase",
            "--phases",
            "2",
            "--mask-key",
            "token_mask",
            "--coord-key",
            "phase_coord",
            "--observer-coord-key",
            "observer_phase",
            "--frame-operators-key",
            "inverse_ops",
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))

    assert payload["report"]["fit_config"]["runtime"]["mask_key"] == "token_mask"
    assert payload["report"]["fit_config"]["runtime"]["coord_key"] == "phase_coord"
    assert payload["report"]["fit_config"]["runtime"]["observer_coord_key"] == "observer_phase"
    assert payload["report"]["fit_config"]["runtime"]["frame_operators_key"] == "inverse_ops"
    assert payload["provenance"]["runtime_fields"]["mask_key"] == "token_mask"
    assert output["provenance"]["runtime_fields"]["coord_key"] == "phase_coord"
    assert output["provenance_fingerprint"] == arti.plan_provenance_fingerprint(output["provenance"])


def test_cli_plan_passes_model_kwargs_json_to_factory(tmp_path: Path, capsys, monkeypatch):
    from arti.cli import main

    module_path = tmp_path / "kwarg_fixture_model.py"
    module_path.write_text(
        "\n".join(
            [
                "import torch.nn as nn",
                "",
                "def make_model(input_dim, hidden_dim, output_dim):",
                "    return nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, output_dim))",
                "",
            ]
        ),
        encoding="utf-8",
    )
    kwargs_path = tmp_path / "model-kwargs.json"
    kwargs_path.write_text(json.dumps({"input_dim": 4, "hidden_dim": 6, "output_dim": 2}), encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    plan_path = tmp_path / "kwarg-plan.json"

    assert main(
        [
            "plan",
            "kwarg_fixture_model:make_model",
            str(plan_path),
            "--model-kwargs-json",
            str(kwargs_path),
            "--sample-shape",
            "2,4",
            "--target-modules",
            "0",
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))

    assert output["planned_count"] == 1
    assert output["provenance"]["model_kwargs_json"] == str(kwargs_path)
    assert payload["report"]["scanned"]["candidates"][0]["dim"] == 6


def test_cli_plan_markdown_includes_provenance(tmp_path: Path, capsys, monkeypatch):
    from arti.cli import main

    module_path = tmp_path / "markdown_fixture_model.py"
    module_path.write_text(
        "\n".join(
            [
                "import torch.nn as nn",
                "",
                "def make_model():",
                "    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    plan_path = tmp_path / "plan.md"

    assert main(["plan", "markdown_fixture_model:make_model", str(plan_path), "--sample-shape", "2,4", "--target-modules", "0"]) == 0
    output = json.loads(capsys.readouterr().out)
    markdown = plan_path.read_text(encoding="utf-8")

    assert "## Plan Provenance" in markdown
    assert f"Provenance fingerprint: `{output['provenance_fingerprint']}`" in markdown
    assert "| `model` | `markdown_fixture_model:make_model` |" in markdown


def test_cli_plan_accepts_json_dict_sample_schema(tmp_path: Path, capsys, monkeypatch):
    from arti.cli import main

    module_path = tmp_path / "dict_fixture_model.py"
    module_path.write_text(
        "\n".join(
            [
                "import torch",
                "import torch.nn as nn",
                "",
                "class DictModel(nn.Module):",
                "    def __init__(self):",
                "        super().__init__()",
                "        self.embed = nn.Embedding(16, 4)",
                "        self.proj = nn.Linear(4, 2)",
                "",
                "    def forward(self, input_ids, attention_mask=None):",
                "        hidden = self.embed(input_ids).float()",
                "        return self.proj(hidden)",
                "",
                "def make_model():",
                "    return DictModel()",
                "",
            ]
        ),
        encoding="utf-8",
    )
    sample_path = tmp_path / "sample.json"
    sample_path.write_text(
        json.dumps(
            {
                "input_ids": {"shape": [2, 5], "dtype": "long", "kind": "randint", "low": 0, "high": 16},
                "attention_mask": {"shape": [2, 5], "dtype": "long", "kind": "ones"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    plan_path = tmp_path / "dict-plan.json"

    assert main(["plan", "dict_fixture_model:make_model", str(plan_path), "--sample-json", str(sample_path), "--profile", "transformer", "--target-modules", "proj", "--max-adapters", "1"]) == 0
    output = json.loads(capsys.readouterr().out)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    schema = payload["report"]["scanned"]["batch_schema"]

    assert output["planned_count"] == 1
    assert output["provenance"]["sample_json"] == str(sample_path)
    assert schema["kind"] == "dict"
    assert schema["token_key"] == "input_ids"
    assert schema["mask_key"] == "attention_mask"
    assert payload["report"]["plugins"] == ["torch", "transformers"]


def test_cli_validate_plan_can_require_matching_config(tmp_path: Path, capsys):
    from arti.cli import main

    config_path = tmp_path / "arti.json"
    config_path.write_text(json.dumps({"insertion": {"where": "0"}}), encoding="utf-8")
    config = arti.load_fit_config(config_path)
    plan_path = arti.project(tiny_model()).configure(config).scan(torch.randn(2, 4)).write_plan(tmp_path / "plan.json")

    assert main(["validate", "plan", str(plan_path), "--expect-config", str(config_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["config_fingerprint"] == config.fingerprint
    assert output["expected_config_fingerprint"] == config.fingerprint


def test_cli_validate_plan_can_require_profile_and_scale(tmp_path: Path, capsys):
    from arti.cli import main

    plan_path = arti.project(tiny_model()).profile("observer-phase").scale("base").scan(torch.randn(2, 4)).write_plan(tmp_path / "plan.json", where="0")

    assert main(["validate", "plan", str(plan_path), "--expect-profile", "observer-phase", "--expect-scale", "base"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["expected_profile"] == "observer-phase"
    assert output["expected_scale"] == "base"

    assert main(["validate", "plan", str(plan_path), "--expect-profile", "latent-adapt"]) == 1
    assert "profile" in capsys.readouterr().err
    assert main(["validate", "plan", str(plan_path), "--expect-scale", "small"]) == 1
    assert "scale" in capsys.readouterr().err


def test_cli_validate_plan_can_require_mechanism_fields(tmp_path: Path, capsys):
    from arti.cli import main

    plan_path = (
        arti.project(tiny_model())
        .profile("observer-phase", phases=5)
        .runtime(mask_key="token_mask")
        .mechanism(operator_count=6, interface_slots=7, recall_slots=2)
        .scan(torch.randn(2, 4))
        .write_plan(tmp_path / "plan.json", where="0")
    )

    assert main(
        [
            "validate",
            "plan",
            str(plan_path),
            "--expect-mechanism",
            "coord_dim=5",
            "--expect-mechanism",
            "operator_count=6",
            "--expect-mechanism",
            "observer_phase=true",
            "--expect-runtime-field",
            "mask_key=token_mask",
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["expected_mechanism"]["operator_count"] == 6
    assert output["mechanism"]["interface_slots"] == 7
    assert output["expected_runtime_fields"]["mask_key"] == "token_mask"
    assert output["runtime"]["mask_key"] == "token_mask"

    assert main(["validate", "plan", str(plan_path), "--expect-mechanism", "operator_count=4"]) == 1
    assert "mechanism.operator_count" in capsys.readouterr().err
    assert main(["validate", "plan", str(plan_path), "--expect-mechanism", "operator-count=6"]) == 1
    assert "unknown ARTI mechanism field" in capsys.readouterr().err
    assert main(["validate", "plan", str(plan_path), "--expect-mechanism", "operator_count"]) == 1
    assert "KEY=VALUE" in capsys.readouterr().err
    assert main(["validate", "plan", str(plan_path), "--expect-runtime-field", "mask_key=attention_mask"]) == 1
    assert "runtime.mask_key" in capsys.readouterr().err
    assert main(["validate", "plan", str(plan_path), "--expect-runtime-field", "operator_count=6"]) == 1
    assert "unknown ARTI runtime field" in capsys.readouterr().err


def test_validate_plan_rejects_invalid_provenance(tmp_path: Path):
    plan_path = arti.project(tiny_model()).scan(torch.randn(2, 4)).write_plan(tmp_path / "plan.json", where="0")
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["provenance"] = {"model": 123, "sample_shape": [2, 4]}
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    try:
        arti.validate_plan(plan_path)
    except ValueError as exc:
        assert "provenance.model" in str(exc)
    else:
        raise AssertionError("invalid plan provenance should fail validation")


def test_validate_plan_rejects_provenance_fingerprint_mismatch(tmp_path: Path):
    from arti.cli import main

    module_path = tmp_path / "fingerprint_fixture_model.py"
    module_path.write_text(
        "\n".join(
            [
                "import torch.nn as nn",
                "",
                "def make_model():",
                "    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))",
                "",
            ]
        ),
        encoding="utf-8",
    )
    import sys

    sys.path.insert(0, str(tmp_path))
    try:
        plan_path = tmp_path / "plan.json"
        assert main(["plan", "fingerprint_fixture_model:make_model", str(plan_path), "--sample-shape", "2,4", "--target-modules", "0"]) == 0
        payload = json.loads(plan_path.read_text(encoding="utf-8"))
        payload["provenance"]["model"] = "other:factory"
        plan_path.write_text(json.dumps(payload), encoding="utf-8")

        try:
            arti.validate_plan(plan_path)
        except ValueError as exc:
            assert "provenance_fingerprint" in str(exc)
        else:
            raise AssertionError("tampered provenance should fail validation")
    finally:
        sys.path.remove(str(tmp_path))


def test_cli_validate_plan_rejects_unexpected_config(tmp_path: Path, capsys):
    from arti.cli import main

    config_path = tmp_path / "arti.json"
    config_path.write_text(json.dumps({"insertion": {"where": "0"}}), encoding="utf-8")
    other_path = tmp_path / "other.json"
    other_path.write_text(json.dumps({"insertion": {"where": "2"}}), encoding="utf-8")
    plan_path = arti.project(tiny_model()).configure(arti.load_fit_config(config_path)).scan(torch.randn(2, 4)).write_plan(tmp_path / "plan.json")

    assert main(["validate", "plan", str(plan_path), "--expect-config", str(other_path)]) == 1
    assert "expected config" in capsys.readouterr().err


def test_cli_validate_plan_can_require_matching_provenance_fingerprint(tmp_path: Path, capsys, monkeypatch):
    from arti.cli import main

    module_path = tmp_path / "gate_fixture_model.py"
    module_path.write_text(
        "\n".join(
            [
                "import torch.nn as nn",
                "",
                "def make_model():",
                "    return nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 2))",
                "",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    plan_path = tmp_path / "plan.json"

    assert main(["plan", "gate_fixture_model:make_model", str(plan_path), "--sample-shape", "2,4", "--target-modules", "0"]) == 0
    plan_output = json.loads(capsys.readouterr().out)
    fingerprint = plan_output["provenance_fingerprint"]

    assert main(["validate", "plan", str(plan_path), "--expect-provenance-fingerprint", fingerprint]) == 0
    validate_output = json.loads(capsys.readouterr().out)
    assert validate_output["expected_provenance_fingerprint"] == fingerprint

    assert main(["validate", "plan", str(plan_path), "--expect-provenance-fingerprint", "wrong"]) == 1
    assert "provenance_fingerprint" in capsys.readouterr().err


def test_cli_validate_artifact_can_require_matching_config(tmp_path: Path, capsys):
    from arti.cli import main

    config_path = tmp_path / "arti.json"
    config_path.write_text(json.dumps({"insertion": {"where": "0"}}), encoding="utf-8")
    artifact = arti.fit(
        tiny_model(),
        config=config_path,
        sample_batch=torch.randn(2, 4),
        dry_run=False,
    ).export(tmp_path / "adapter.pt")
    config = arti.load_fit_config(config_path)

    assert main(["validate", "artifact", str(artifact), "--expect-config", str(config_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["config_fingerprint"] == config.fingerprint
    assert output["expected_config_fingerprint"] == config.fingerprint


def test_cli_validate_artifact_rejects_unexpected_config(tmp_path: Path, capsys):
    from arti.cli import main

    config_path = tmp_path / "arti.json"
    config_path.write_text(json.dumps({"insertion": {"where": "0"}}), encoding="utf-8")
    other_path = tmp_path / "other.json"
    other_path.write_text(json.dumps({"insertion": {"where": "2"}}), encoding="utf-8")
    artifact = arti.fit(
        tiny_model(),
        config=config_path,
        sample_batch=torch.randn(2, 4),
    ).export(tmp_path / "adapter.pt")

    assert main(["validate", "artifact", str(artifact), "--expect-config", str(other_path)]) == 1
    assert "expected config" in capsys.readouterr().err


def test_cli_validate_artifact_can_require_matching_adapter_state_sha256(tmp_path: Path, capsys):
    from arti.cli import main

    artifact = arti.fit(tiny_model(), sample_batch=torch.randn(2, 4), target_modules="0").export(tmp_path / "adapter.pt")
    payload = torch.load(artifact, weights_only=True)
    digest = payload["manifest"]["adapter_state_sha256"]

    assert main(["validate", "artifact", str(artifact), "--expect-adapter-state-sha256", digest]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["adapter_state_sha256"] == digest
    assert output["expected_adapter_state_sha256"] == digest

    assert main(["validate", "artifact", str(artifact), "--expect-adapter-state-sha256", "wrong"]) == 1
    assert "adapter_state_sha256" in capsys.readouterr().err


def test_cli_validate_artifact_can_require_profile_and_scale(tmp_path: Path, capsys):
    from arti.cli import main

    artifact = arti.fit(tiny_model(), sample_batch=torch.randn(2, 4), target_modules="0", profile="observer-phase", scale="base").export(tmp_path / "adapter.pt")

    assert main(["validate", "artifact", str(artifact), "--expect-profile", "observer-phase", "--expect-scale", "base"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["expected_profile"] == "observer-phase"
    assert output["expected_scale"] == "base"

    assert main(["validate", "artifact", str(artifact), "--expect-profile", "latent-adapt"]) == 1
    assert "profile" in capsys.readouterr().err
    assert main(["validate", "artifact", str(artifact), "--expect-scale", "small"]) == 1
    assert "scale" in capsys.readouterr().err


def test_cli_validate_artifact_can_require_mechanism_fields(tmp_path: Path, capsys):
    from arti.cli import main

    artifact = arti.fit(
        tiny_model(),
        sample_batch=torch.randn(2, 4),
        target_modules="0",
        profile="observer-phase",
        phases=5,
        mechanism={"operator_count": 6, "interface_slots": 7, "recall_slots": 2},
    ).export(tmp_path / "adapter.pt")

    assert main(
        [
            "validate",
            "artifact",
            str(artifact),
            "--expect-mechanism",
            "coord_dim=5",
            "--expect-mechanism",
            "operator_count=6",
            "--expect-mechanism",
            "observer_phase=true",
        ]
    ) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["expected_mechanism"]["coord_dim"] == 5
    assert output["mechanism"]["interface_slots"] == 7

    assert main(["validate", "artifact", str(artifact), "--expect-mechanism", "recall_slots=4"]) == 1
    assert "mechanism.recall_slots" in capsys.readouterr().err


def test_cli_lock_creates_and_validates_build_lock(tmp_path: Path, capsys):
    from arti.cli import main

    sample = torch.randn(2, 4)
    config_path = tmp_path / "arti.json"
    config_path.write_text(json.dumps({"fit": {"profile": "observer-phase", "phases": 16, "scale": "base"}, "insertion": {"where": "0"}}), encoding="utf-8")
    config = arti.load_fit_config(config_path)
    plan_path = arti.project(tiny_model()).configure(arti.load_fit_config(config_path)).scan(sample).write_plan(tmp_path / "plan.json")
    artifact = arti.fit(tiny_model(), sample_batch=sample, config=config_path).export(tmp_path / "adapter.pt")
    lock_path = tmp_path / "arti.lock.json"

    assert main(["lock", str(lock_path), "--artifact", str(artifact), "--plan", str(plan_path), "--config", str(config_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["kind"] == "build-lock"
    assert output["artifact"] == "adapter.pt"
    assert output["plan"] == "plan.json"
    assert len(output["adapter_state_sha256"]) == 64
    assert len(output["report_sha256"]) == 64

    assert main(
        [
            "validate",
            "lock",
            str(lock_path),
            "--expect-config",
            str(config_path),
            "--expect-provenance-fingerprint",
            output["provenance_fingerprint"],
            "--expect-adapter-state-sha256",
            output["adapter_state_sha256"],
            "--expect-report-sha256",
            output["report_sha256"],
            "--expect-profile",
            "observer-phase",
            "--expect-scale",
            "base",
            "--expect-mechanism",
            "recall_slots=8",
            "--expect-mechanism",
            "operator_count=4",
            "--max-adapters",
            "1",
            "--max-extra-params",
            "100000",
        ]
    ) == 0
    validated = json.loads(capsys.readouterr().out)
    assert validated["artifact"] == "adapter.pt"
    assert validated["report_sha256"] == output["report_sha256"]
    assert validated["expected_adapter_state_sha256"] == output["adapter_state_sha256"]
    assert validated["expected_report_sha256"] == output["report_sha256"]
    assert validated["expected_config_fingerprint"] == config.fingerprint
    assert validated["expected_provenance_fingerprint"] == output["provenance_fingerprint"]
    assert validated["expected_profile"] == "observer-phase"
    assert validated["expected_scale"] == "base"
    assert validated["expected_mechanism"]["recall_slots"] == 8
    assert validated["mechanism"]["operator_count"] == 4
    assert validated["inserted_count"] == 1
    assert validated["cli_max_adapters"] == 1
    assert validated["cli_max_extra_params"] == 100000

    other_config_path = tmp_path / "other.json"
    other_config_path.write_text(json.dumps({"insertion": {"where": "2"}}), encoding="utf-8")
    assert main(["validate", "lock", str(lock_path), "--expect-config", str(other_config_path)]) == 1
    assert "expected config" in capsys.readouterr().err
    assert main(["validate", "lock", str(lock_path), "--expect-provenance-fingerprint", "wrong"]) == 1
    assert "provenance_fingerprint" in capsys.readouterr().err
    assert main(["validate", "lock", str(lock_path), "--expect-adapter-state-sha256", "wrong"]) == 1
    assert "adapter_state_sha256" in capsys.readouterr().err
    assert main(["validate", "lock", str(lock_path), "--expect-report-sha256", "wrong"]) == 1
    assert "report_sha256" in capsys.readouterr().err
    assert main(["validate", "lock", str(lock_path), "--expect-profile", "latent-adapt"]) == 1
    assert "profile" in capsys.readouterr().err
    assert main(["validate", "lock", str(lock_path), "--expect-scale", "small"]) == 1
    assert "scale" in capsys.readouterr().err
    assert main(["validate", "lock", str(lock_path), "--expect-mechanism", "recall_slots=4"]) == 1
    assert "mechanism.recall_slots" in capsys.readouterr().err
    assert main(["validate", "lock", str(lock_path), "--max-adapters", "0"]) == 1
    assert "max_adapters" in capsys.readouterr().err
    assert main(["validate", "lock", str(lock_path), "--max-extra-params", "1"]) == 1
    assert "max_extra_params" in capsys.readouterr().err


def test_cli_validate_plan_accepts_external_budget_limits(tmp_path: Path, capsys):
    from arti.cli import main

    plan_path = arti.project(tiny_model()).scan(torch.randn(2, 4)).write_plan(tmp_path / "plan.json", where="0")

    assert main(["validate", "plan", str(plan_path), "--max-adapters", "1", "--max-extra-params", "100000"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["cli_max_adapters"] == 1
    assert output["cli_max_extra_params"] == 100000


def test_cli_validate_plan_rejects_external_budget_limits(tmp_path: Path, capsys):
    from arti.cli import main

    plan_path = arti.project(tiny_model()).scan(torch.randn(2, 4)).write_plan(tmp_path / "plan.json", where="0")

    assert main(["validate", "plan", str(plan_path), "--max-adapters", "0"]) == 1
    assert "max_adapters" in capsys.readouterr().err
    assert main(["validate", "plan", str(plan_path), "--max-extra-params", "1"]) == 1
    assert "max_extra_params" in capsys.readouterr().err


def test_cli_validate_artifact_rejects_external_budget_limits(tmp_path: Path, capsys):
    from arti.cli import main

    artifact = arti.fit(tiny_model(), sample_batch=torch.randn(2, 4), target_modules="0").export(tmp_path / "adapter.pt")

    assert main(["validate", "artifact", str(artifact), "--max-adapters", "0"]) == 1
    assert "max_adapters" in capsys.readouterr().err
    assert main(["validate", "artifact", str(artifact), "--max-extra-params", "1"]) == 1
    assert "max_extra_params" in capsys.readouterr().err


def test_cli_rejects_invalid_plan(tmp_path: Path, capsys):
    from arti.cli import main

    plan_path = arti.project(tiny_model()).scan(torch.randn(2, 4)).write_plan(tmp_path / "bad-plan.json", where="0")
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["kind"] = "wrong"
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    result = main(["validate", "plan", str(plan_path)])
    output = capsys.readouterr()

    assert result == 1
    assert "fit plan kind" in output.err


def test_torch_namespace_reexports_fit_api():
    import arti.torch as arti_torch

    assert arti_torch.project is arti.project
    assert arti_torch.AdapterArtifactManifest is arti.AdapterArtifactManifest
    assert arti_torch.fit is arti.fit
    assert arti_torch.apply_adapter is arti.apply_adapter
    assert arti_torch.validate_artifact is arti.validate_artifact
    assert arti_torch.validate_artifact_payload is arti.validate_artifact_payload
    assert arti_torch.validate_plan is arti.validate_plan
    assert arti_torch.validate_plan_payload is arti.validate_plan_payload
    assert arti_torch.get_plugin is arti.get_plugin
    assert arti_torch.infer_batch_schema is arti.infer_batch_schema
    assert arti_torch.attention_mask_to_visibility is arti.attention_mask_to_visibility
    assert arti_torch.resolve_objectives is arti.resolve_objectives
    assert arti_torch.BuildTaskSpec is arti.BuildTaskSpec
    assert arti_torch.FitReportSummary is arti.FitReportSummary
    assert arti_torch.FitProjectConfig is arti.FitProjectConfig
    assert arti_torch.load_fit_config is arti.load_fit_config
    assert arti_torch.AdapterInsertionPlan is arti.AdapterInsertionPlan
    assert arti_torch.FitTaskRecord is arti.FitTaskRecord
    assert arti_torch.ForwardProfile is arti.ForwardProfile
    assert arti_torch.MechanismSummary is arti.MechanismSummary
    assert arti_torch.ParameterSummary is arti.ParameterSummary
    assert ARTIProject is arti.ARTIProject


def test_fit_plugin_registry_reports_optional_dependency_status():
    plugin = arti.get_plugin("transformers")

    assert plugin.default_strategy == "transformer"
    assert plugin.optional_dependency == "transformers"
    assert isinstance(plugin.available, bool)


def test_capabilities_report_profiles_scales_and_plugins():
    report = arti.capabilities(phases=12)

    assert report["kind"] == "capabilities"
    assert report["profiles"]["observer-phase"]["coord_dim"] == 12
    assert report["profiles"]["virtual-recall"]["virtual_recall"] is True
    assert report["scales"]["small"]["interface_slots"] == 8
    assert report["plugins"]["torch"]["available"] is True
    assert arti.list_profiles(phases=6)["observer-phase"]["coord_dim"] == 6
    assert "large" in arti.list_scales()
    assert "transformers" in arti.list_plugins()
    assert arti.list_plugins()["vision-cnn"]["default_strategy"] == "vision-cnn"
    assert arti.list_plugins()["recurrent"]["default_strategy"] == "recurrent"


def test_cli_inspect_reports_capabilities(capsys):
    from arti.cli import main

    assert main(["inspect", "--phases", "10"]) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["ok"] is True
    assert output["kind"] == "capabilities"
    assert output["profiles"]["observer-phase"]["coord_dim"] == 10
    assert "plugins" in output

    assert main(["inspect", "plugins"]) == 0
    plugin_output = json.loads(capsys.readouterr().out)
    assert plugin_output["kind"] == "plugins"
    assert "torch" in plugin_output["plugins"]


def test_unknown_fit_plugin_fails_fast():
    model = tiny_model()

    try:
        arti.project(model).plugin("unknown")
    except ValueError as exc:
        assert "unknown ARTI fit plugin" in str(exc)
    else:
        raise AssertionError("unknown plugin should fail")
