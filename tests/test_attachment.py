from __future__ import annotations

import copy

import pytest
import torch
import torch.nn as nn

import arti
from arti.component_registry import canonical_contract_reference
from arti.serialization import load as load_arti
from benchmarks.federal_refine_architecture_search import ArchitectureTaskBank, ExperimentConfig


def tiny_model() -> nn.Sequential:
    return nn.Sequential(nn.Linear(8, 8), nn.GELU(), nn.Linear(8, 8))


def federal_layer() -> arti.ARTILayer:
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1017)
        config = ExperimentConfig(
            tasks=2,
            depths=(1, 2),
            steps=2,
            batch_size=2,
            candidates=2,
            eval_size_per_task=2,
            max_seconds=30.0,
        )
        bank = ArchitectureTaskBank(config, task_id=0, seed=17)
        program = bank.sealed_program(refine_steps=2)
        federation = arti.mechanisms.FederalRecallV3(
            {program.bank_id: program},
            terminal_abi=bank.terminal_abi(),
            root_bank_ids=(program.bank_id,),
            max_levels=1,
            max_k=16,
        )
    return arti.ARTILayer(
        federation,
        axis_names=("batch", "feature"),
        axis_roles=("batch", "feature"),
    )


def test_default_artilayer_is_explicit_federal_identity_shell() -> None:
    layer = arti.ARTILayer()
    x = torch.randn(2, 4, 8)
    value, result = layer(x, return_info=True)

    assert arti.component_ref(layer) == canonical_contract_reference("arti/layer@3")
    assert layer.federation is None
    assert torch.equal(value, x)
    assert result.value_identity
    assert arti.nn.Layer is arti.Layer
    assert arti.nn.Layer is not arti.ARTILayer
    assert arti.torch.ARTILayer is arti.ARTILayer
    provenance = layer.runtime_provenance()
    assert provenance["surface"] == "federal-bank-runtime"
    assert provenance["layer_ref"] == canonical_contract_reference("arti/layer@3")
    assert provenance["federation_ref"] is None
    assert provenance["identity_when_unconfigured"] is True


def test_configured_artilayer_executes_a_real_federal_v3_runtime() -> None:
    layer = federal_layer()
    x = torch.randn(1, 16)

    value, result = layer(x, return_info=True, return_trace=True)

    assert layer.federation is not None
    assert value.shape == x.shape
    assert result.outputs is not None
    assert result.trace is not None
    assert result.trace.winner_paths
    assert result.view is not None
    spec = arti.component_spec(layer)
    assert spec.reference == canonical_contract_reference("arti/layer@3")
    assert spec.config["federation_ref"] == canonical_contract_reference(
        "arti/federal-recall@3"
    )
    assert spec.dependencies == (canonical_contract_reference("arti/federal-recall@3"),)


def test_configured_artilayer_backpropagates_through_real_federal_operands() -> None:
    torch.manual_seed(19)
    layer = federal_layer()
    x = torch.randn(1, 16)
    optimizer = torch.optim.SGD(
        [parameter for parameter in layer.parameters() if parameter.requires_grad],
        lr=0.05,
    )
    losses: list[float] = []
    for _ in range(6):
        optimizer.zero_grad()
        loss = layer(x).square().mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    assert losses[-1] < losses[0]


def test_attach_returns_original_model_and_infers_sequential_layers() -> None:
    model = tiny_model()
    original_id = id(model)
    layer = arti.ARTILayer()
    preview = arti.ARTI.preview(model, layer, layers=("0", "2"))
    attached = arti.ARTI.attach(model, layer, layers=("0", "2"))

    assert id(attached) == original_id
    assert attached.arti.paths == ("0", "2")
    assert preview.trainable_parameters == attached.arti.summary().trainable_parameters
    assert preview.trainable_parameters == sum(
        parameter.numel() for parameter in attached.arti.parameters()
    )
    assert preview.trainable_parameters == 0
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
    arti.ARTI.attach(model, arti.ARTILayer(), layers="layers.*")

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


def test_federal_save_reload_and_forward_consistency(tmp_path, paired_rng) -> None:
    torch.manual_seed(7)
    base = nn.Sequential(nn.Linear(16, 16))
    initial = copy.deepcopy(base.state_dict())
    model = arti.ARTI.attach(base, federal_layer(), layers=("0",))
    x = torch.randn(1, 16)
    model.eval()
    artifact = tmp_path / "tiny.arti.st"
    model.arti.save(artifact)
    saved_manifest = load_arti(
        artifact,
        load_resources=False,
        load_checkpoint=False,
    ).manifest
    surface = saved_manifest["architecture"]["config"]["unified_attachment"]["execution_surface"]
    assert surface["kind"] == "federal-bank-attachment"
    assert set(surface["layers"]) == {"0"}
    assert surface["federal_compiler_ref"] == canonical_contract_reference(
        "arti/federal-static-compiler@1"
    )
    assert surface["compiled_artifact_is_separate"] is True

    restored = nn.Sequential(nn.Linear(16, 16))
    restored.load_state_dict(initial)
    arti.ARTI.load(restored, artifact, layer=federal_layer())
    restored.eval()
    expected, actual = paired_rng(
        lambda: model(x).detach(),
        lambda: restored(x).detach(),
    )

    assert torch.equal(expected, actual)
    assert restored.arti.summary().trainable_parameters == model.arti.summary().trainable_parameters
    with pytest.raises(ValueError, match="topology"):
        wrong = arti.ARTI.attach(
            nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16)),
            federal_layer(),
            layers=("0", "1"),
        )
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
        arti.ARTILayer(),
        layers="model.layers.*",
    )
    x = torch.randn(2, 3, 8)

    assert type(attached) is original_type
    assert attached.config == {"model_type": "fake"}
    assert attached.arti.paths == ("model.layers.0", "model.layers.1")
    assert attached(x)[0].shape == x.shape
    assert attached.generate(x, scale=2).shape == x.shape


def test_bad_patterns_suffix_and_transactional_failure(tmp_path) -> None:
    with pytest.raises(ValueError, match="matched no modules"):
        arti.ARTI.attach(tiny_model(), layers="missing.*")
    model = arti.ARTI.attach(tiny_model(), arti.ARTILayer(), layers="0")
    with pytest.raises(ValueError, match=".arti.st"):
        model.arti.save(tmp_path / "weights.st")
    with pytest.raises(ValueError, match="already"):
        arti.ARTI.attach(model)

    host = tiny_model()
    trainability = {name: value.requires_grad for name, value in host.named_parameters()}

    def factory(spec):
        if spec.path == "2":
            raise RuntimeError("factory failed")
        return arti.ARTILayer()

    with pytest.raises(RuntimeError, match="factory failed"):
        arti.ARTI.attach(host, factory, layers=("0", "2"))
    assert isinstance(host[0], nn.Linear)
    assert isinstance(host[2], nn.Linear)
    assert {name: value.requires_grad for name, value in host.named_parameters()} == trainability


def test_torch_namespace_exports_unified_attachment() -> None:
    assert arti.torch.ARTI is arti.ARTI
    assert arti.torch.ARTIAttachment is arti.ARTIAttachment
