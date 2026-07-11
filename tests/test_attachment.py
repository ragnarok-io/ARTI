from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

import arti


def tiny_model() -> nn.Sequential:
    return nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 8))


def test_attach_returns_original_model_and_infers_sequential_layers() -> None:
    model = tiny_model()
    original_id = id(model)
    preview = arti.ARTI.preview(model, {"rank": 2, "slots": 3})
    attached = arti.ARTI.attach(model, {"rank": 2, "slots": 3})

    assert id(attached) == original_id
    assert isinstance(attached, nn.Sequential)
    assert attached.arti.paths == ("0", "2")
    assert preview.trainable_parameters == attached.arti.summary().trainable_parameters
    assert preview.trainable_parameters == sum(parameter.numel() for parameter in attached.arti.parameters())
    assert preview.trainable_parameters > 0
    assert attached(torch.randn(2, 4, 8)).shape == (2, 4, 8)
    assert all(parameter.requires_grad for parameter in attached.arti.parameters())


def test_glob_discovery_feature_switches_and_reversible_detach() -> None:
    class Decoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(8, 8) for _ in range(4)])

        def forward(self, x):
            for layer in self.layers:
                x = layer(x)
            return x

    model = Decoder()
    before = copy.deepcopy(model.state_dict())
    discovered = arti.ARTI.discover(model, "layers.*")
    arti.ARTI.attach(model, {"layers": "layers.*", "rank": 2, "slots": 2})

    assert tuple(item.path for item in discovered) == ("layers.0", "layers.1", "layers.2", "layers.3")
    model.arti.disable("half", paths=("layers.0",))
    assert isinstance(model.layers[0].recall.survival, nn.Identity)
    model.arti.enable("half", paths=("layers.0",))
    assert isinstance(model.layers[0].recall.survival, arti.Half)
    model.arti.disable("recognition")
    assert all(wrapper.recall.recognition_mode == "none" for wrapper in model.arti._layered.wrappers.values())
    model.arti.disable()
    assert all(not wrapper.enabled for wrapper in model.arti._layered.wrappers.values())

    restored = model.arti.detach()
    assert not hasattr(restored, "arti")
    assert all(not isinstance(layer, arti.LayerRecallWrapper) for layer in restored.layers)
    for name, tensor in before.items():
        assert torch.equal(restored.state_dict()[name], tensor)


def test_train_save_reload_and_forward_consistency(tmp_path) -> None:
    torch.manual_seed(7)
    base = tiny_model()
    initial = copy.deepcopy(base.state_dict())
    model = arti.ARTI.attach(base, {"layers": ("0", "2"), "rank": 2, "slots": 3})
    optimizer = torch.optim.AdamW(model.arti.parameters(), lr=1e-2)
    x = torch.randn(3, 5, 8)
    target = torch.randn_like(x)
    loss = torch.nn.functional.mse_loss(model(x), target)
    loss.backward()
    optimizer.step()
    model.eval()
    expected = model(x).detach()
    artifact = tmp_path / "tiny.recall.arti.st"
    model.arti.save(artifact)

    restored = tiny_model()
    restored.load_state_dict(initial)
    arti.ARTI.load(restored, artifact)
    restored.eval()

    assert torch.equal(expected, restored(x))
    assert restored.arti.summary().trainable_parameters == model.arti.summary().trainable_parameters
    with pytest.raises(ValueError, match="topology"):
        wrong = arti.ARTI.attach(tiny_model(), {"layers": ("0",), "rank": 2, "slots": 3})
        wrong.arti.load(artifact)


def test_transformers_style_model_keeps_class_config_generate_and_tuple_contract(tmp_path) -> None:
    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(8, 8)

        def forward(self, hidden, **_kwargs):
            return self.proj(hidden), torch.ones((), device=hidden.device)

    class FakeCausalLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = {"model_type": "fake"}
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([Block(), Block()])

        def forward(self, hidden, **kwargs):
            aux = None
            for layer in self.model.layers:
                hidden, aux = layer(hidden, **kwargs)
            return hidden, aux

        def generate(self, hidden, *, scale=1.0):
            return self.forward(hidden)[0] * scale

    model = FakeCausalLM()
    original_type = type(model)
    attached = arti.ARTI.attach(model, {"rank": 2, "slots": 2})
    x = torch.randn(2, 3, 8)

    assert type(attached) is original_type
    assert attached.config == {"model_type": "fake"}
    assert attached.arti.paths == ("model.layers.0", "model.layers.1")
    assert attached(x)[0].shape == x.shape
    assert attached.generate(x, scale=2).shape == x.shape
    attached.arti.save(tmp_path / "fake.recall.arti.st")


def test_bad_patterns_and_artifact_suffix_fail_clearly(tmp_path) -> None:
    with pytest.raises(ValueError, match="matched no modules"):
        arti.ARTI.attach(tiny_model(), {"layers": "missing.*"})
    model = arti.ARTI.attach(tiny_model(), {"layers": "0"})
    with pytest.raises(ValueError, match="recall.arti.st"):
        model.arti.save(tmp_path / "weights.arti.st")
    with pytest.raises(ValueError, match="already"):
        arti.ARTI.attach(model)


def test_failed_attach_and_load_are_transactional(tmp_path) -> None:
    host = tiny_model()
    trainability = {name: value.requires_grad for name, value in host.named_parameters()}
    config = arti.LayeredRecallConfig(
        layers=(arti.LayerRecallSpec("0", dim=8, rank=2, slots=2), arti.LayerRecallSpec("missing", dim=8))
    )
    with pytest.raises(AttributeError):
        arti.ARTI.attach(host, config)
    assert isinstance(host[0], nn.Linear)
    assert {name: value.requires_grad for name, value in host.named_parameters()} == trainability

    source = arti.ARTI.attach(tiny_model(), {"layers": ("0", "2"), "rank": 2, "slots": 2})
    artifact = tmp_path / "source.recall.arti.st"
    source.arti.save(artifact)
    incompatible = nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 8), nn.GELU())
    with pytest.raises(ValueError, match="host structure"):
        arti.ARTI.load(incompatible, artifact)
    assert not hasattr(incompatible, "arti")
    assert isinstance(incompatible[0], nn.Linear)


def test_torch_namespace_exports_unified_attachment() -> None:
    assert arti.torch.ARTI is arti.ARTI
    assert arti.torch.ARTIAttachment is arti.ARTIAttachment
