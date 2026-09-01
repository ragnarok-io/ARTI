from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

import arti


def tiny_model() -> nn.Sequential:
    return nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 8))


def trainable_layer() -> arti.ARTILayer:
    return arti.ARTILayer(
        arti.mechanisms.AdaptivePulse(
            half=arti.Half(stochastic=False, learnable=True),
        )
    )


def test_default_artilayer_is_adaptive_pulse_identity() -> None:
    layer = arti.ARTILayer()
    x = torch.randn(2, 4, 8)
    value, result = layer(x, return_info=True)

    assert arti.component_ref(layer) == "arti/layer@2"
    assert isinstance(layer.pulse, arti.mechanisms.AdaptivePulse)
    assert torch.equal(value, x)
    assert result.value_identity
    assert arti.nn.Layer is arti.Layer
    assert arti.nn.Layer is not arti.ARTILayer
    assert arti.torch.ARTILayer is arti.ARTILayer


def test_attach_returns_original_model_and_infers_sequential_layers() -> None:
    model = tiny_model()
    original_id = id(model)
    layer = trainable_layer()
    preview = arti.ARTI.preview(model, layer, layers=("0", "2"))
    attached = arti.ARTI.attach(model, layer, layers=("0", "2"))

    assert id(attached) == original_id
    assert attached.arti.paths == ("0", "2")
    assert preview.trainable_parameters == attached.arti.summary().trainable_parameters
    assert preview.trainable_parameters == sum(
        parameter.numel() for parameter in attached.arti.parameters()
    )
    assert preview.trainable_parameters > 0
    assert attached(torch.randn(2, 4, 8)).shape == (2, 4, 8)


def test_glob_discovery_enable_disable_and_reversible_detach() -> None:
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
    arti.ARTI.attach(model, trainable_layer(), layers="layers.*")

    assert tuple(item.path for item in discovered) == (
        "layers.0",
        "layers.1",
        "layers.2",
        "layers.3",
    )
    model.arti.disable(paths=("layers.0",))
    assert not model.layers[0].enabled
    model.arti.enable(paths=("layers.0",))
    assert model.layers[0].enabled
    model.arti.disable()
    assert all(not wrapper.enabled for wrapper in model.arti._layers.wrappers.values())

    restored = model.arti.detach()
    assert not hasattr(restored, "arti")
    assert all(isinstance(layer, nn.Linear) for layer in restored.layers)
    for name, tensor in before.items():
        assert torch.equal(restored.state_dict()[name], tensor)


def test_train_save_reload_and_forward_consistency(tmp_path, paired_rng) -> None:
    torch.manual_seed(7)
    base = tiny_model()
    initial = copy.deepcopy(base.state_dict())
    model = arti.ARTI.attach(base, trainable_layer(), layers=("0", "2"))
    optimizer = torch.optim.AdamW(model.arti.parameters(), lr=1e-2)
    x = torch.randn(3, 5, 8)
    target = torch.randn_like(x)
    torch.nn.functional.mse_loss(model(x), target).backward()
    optimizer.step()
    model.eval()
    artifact = tmp_path / "tiny.arti.st"
    model.arti.save(artifact)

    restored = tiny_model()
    restored.load_state_dict(initial)
    arti.ARTI.load(restored, artifact, layer=trainable_layer())
    restored.eval()
    expected, actual = paired_rng(
        lambda: model(x).detach(),
        lambda: restored(x).detach(),
    )

    assert torch.equal(expected, actual)
    assert restored.arti.summary().trainable_parameters == model.arti.summary().trainable_parameters
    with pytest.raises(ValueError, match="topology"):
        wrong = arti.ARTI.attach(tiny_model(), trainable_layer(), layers=("0",))
        wrong.arti.load(artifact)


def test_transformers_style_output_tree_and_generate_contract(tmp_path) -> None:
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
    attached = arti.ARTI.attach(
        model,
        trainable_layer(),
        layers="model.layers.*",
    )
    x = torch.randn(2, 3, 8)

    assert type(attached) is original_type
    assert attached.config == {"model_type": "fake"}
    assert attached.arti.paths == ("model.layers.0", "model.layers.1")
    assert attached(x)[0].shape == x.shape
    assert attached.generate(x, scale=2).shape == x.shape
    attached.arti.save(tmp_path / "fake.arti.st")


def test_bad_patterns_suffix_and_transactional_failure(tmp_path) -> None:
    with pytest.raises(ValueError, match="matched no modules"):
        arti.ARTI.attach(tiny_model(), layers="missing.*")
    model = arti.ARTI.attach(tiny_model(), trainable_layer(), layers="0")
    with pytest.raises(ValueError, match=".arti.st"):
        model.arti.save(tmp_path / "weights.st")
    with pytest.raises(ValueError, match="already"):
        arti.ARTI.attach(model)

    host = tiny_model()
    trainability = {name: value.requires_grad for name, value in host.named_parameters()}

    def factory(spec):
        if spec.path == "2":
            raise RuntimeError("factory failed")
        return trainable_layer()

    with pytest.raises(RuntimeError, match="factory failed"):
        arti.ARTI.attach(host, factory, layers=("0", "2"))
    assert isinstance(host[0], nn.Linear)
    assert isinstance(host[2], nn.Linear)
    assert {name: value.requires_grad for name, value in host.named_parameters()} == trainability


def test_torch_namespace_exports_unified_attachment() -> None:
    assert arti.torch.ARTI is arti.ARTI
    assert arti.torch.ARTIAttachment is arti.ARTIAttachment
