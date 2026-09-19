import torch

from arti.alpha import (
    ContextInput,
    FourierContextPlacementEncoder,
    LearnedTokenPositionEncoder,
    LanguageHead,
    LexicalEmbedding,
    LinearContextPlacementEncoder,
    RecursiveContextCompiler,
    SinusoidalTokenPositionEncoder,
)
from arti.component_registry import component_spec, resolve_component


def channel(value, position=0.0, valid=None):
    if valid is None:
        valid = torch.ones(value.shape[:2], dtype=torch.bool)
    return ContextInput(
        value,
        valid,
        torch.tensor([[position, 0.5]], dtype=torch.float32).expand(value.shape[0], -1),
    )


def test_context_is_cross_layer_and_downstream_is_normal_token_state():
    torch.manual_seed(5)
    fusion = RecursiveContextCompiler(8, context_width=4, heads=2, layers=3, readout_layers=(1, 3))
    inputs = [channel(torch.randn(2, 3, 8)), channel(torch.randn(2, 2, 8), 4.0)]
    context = fusion(inputs, output="context")
    downstream = fusion(inputs, output="downstream")
    assert context.shape == (2, 4, 8)
    assert downstream.shape == (2, 8)
    (context.square().mean() + downstream.square().mean()).backward()
    assert fusion.blocks[0].qkv.weight.grad is not None
    assert fusion.blocks[2].qkv.weight.grad is not None
    assert fusion.context_seed.grad is not None


def test_native_context_width_and_same_level_compilation(monkeypatch):
    torch.manual_seed(19)
    fusion = RecursiveContextCompiler(8, context_width=4, heads=2, layers=2)
    source = torch.randn(1, 5, 8, requires_grad=True)
    calls = []
    original = fusion._network

    def record(channels, width):
        calls.append(tuple(channel.value.shape[1] for channel in channels))
        return original(channels, width)

    monkeypatch.setattr(fusion, "_network", record)
    result = fusion.compile([channel(source)], depth=3, context_width=2)
    assert result.shape == (1, 2, 8)
    assert calls == [(5,), (2,), (2,)]
    result.square().mean().backward()
    assert source.grad is not None and torch.isfinite(source.grad).all()
    assert fusion.context_seed.grad is not None
    assert fusion.output_schema.dimensions == ("B", "M", 8)


def test_native_context_waist_is_before_lexical_source(monkeypatch):
    torch.manual_seed(21)
    fusion = RecursiveContextCompiler(8, context_width=2, heads=2, layers=2)
    source = torch.randn(1, 3, 8)
    observed = {}

    def capture(value, valid):
        observed["value"] = value.detach().clone()
        observed["valid"] = valid.detach().clone()
        return fusion.blocks[0].__class__.forward(fusion.blocks[0], value, valid)

    monkeypatch.setattr(fusion.blocks[0], "forward", capture)
    fusion([channel(source)], context_width=2)
    assert observed["value"].shape == (1, 5, 8)
    torch.testing.assert_close(observed["value"][:, :2], fusion.context_seed.detach()[None])
    assert bool(observed["valid"][:, :2].all())
    assert bool(observed["valid"][:, 2:].all())


def test_context_only_replacement_and_invalid_width():
    fusion = RecursiveContextCompiler(8, context_width=4, heads=2, layers=2)
    source = channel(torch.randn(1, 5, 8))
    for width in (1, 2, 4):
        context = fusion.compile([source], depth=2, context_width=width)
        assert context.shape == (1, width, 8)
        replaced = fusion([channel(context)], output="downstream")
        assert replaced.shape == (1, 8)
    for width in (0, 5):
        try:
            fusion([source], context_width=width)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid context width accepted")


def test_compiled_context_is_an_input_channel_not_a_control():
    torch.manual_seed(7)
    fusion = RecursiveContextCompiler(8, context_width=3, heads=2, layers=2)
    lexical = channel(torch.randn(1, 2, 8), 8.0)
    child = fusion([channel(torch.randn(1, 5, 8))], output="context")
    child.retain_grad()
    logits = LanguageHead(8, 19)(fusion([channel(child, 0.0), lexical], output="downstream"))
    logits.sum().backward()
    assert child.grad is not None and child.grad.abs().sum() > 0


def test_late_placement_does_not_mutate_context():
    torch.manual_seed(11)
    fusion = RecursiveContextCompiler(8, context_width=3, heads=2, layers=2).eval()
    context = fusion([channel(torch.randn(1, 4, 8))], output="context").detach()
    original = context.clone()
    left = fusion([channel(context, 1.0)], output="downstream")
    right = fusion([channel(context, 100.0)], output="downstream")
    torch.testing.assert_close(context, original, rtol=0, atol=0)
    assert not torch.allclose(left, right)


def test_registry_has_only_current_rcc_components():
    fusion = RecursiveContextCompiler(8, context_width=3, heads=2, layers=2, readout_layers=(2,))
    assert component_spec(fusion).reference == "arti/recursive-context-compiler@1"
    restored = resolve_component(component_spec(fusion).reference, **fusion.config())
    assert isinstance(restored, RecursiveContextCompiler)


def test_placement_encoder_is_explicit_and_serialized():
    default = RecursiveContextCompiler(8, context_width=3, heads=2, layers=2)
    assert isinstance(default.placement_encoder, LinearContextPlacementEncoder)

    compiler = RecursiveContextCompiler(
        8,
        context_width=3,
        heads=2,
        layers=2,
        placement_encoder=FourierContextPlacementEncoder(2, 8, bands=4, base=32.0),
    )
    spec = component_spec(compiler)
    placement = spec.config["placement_encoder"]
    assert placement["reference"] == "arti/context-placement-fourier@1"
    restored = resolve_component(spec.reference, **spec.config)
    assert isinstance(restored.placement_encoder, FourierContextPlacementEncoder)
    assert restored.placement_encoder.bands == 4
    assert restored.placement_encoder.base == 32.0


def test_lexical_embedding_is_fixed_by_default_and_can_be_explicitly_trainable():
    fixed = LexicalEmbedding(
        11,
        8,
        position_encoder=SinusoidalTokenPositionEncoder(8, base=10000.0),
    )
    assert fixed.trainable is False
    assert fixed.embedding.weight.requires_grad is False
    trainable = LexicalEmbedding(
        11,
        8,
        trainable=True,
        position_encoder=SinusoidalTokenPositionEncoder(8, base=10000.0),
    )
    assert trainable.trainable is True
    assert trainable.embedding.weight.requires_grad is True


def test_lexical_position_encoder_is_explicit_and_serialized():
    try:
        LexicalEmbedding(11, 8)
    except (TypeError, ValueError) as error:
        assert "position_encoder" in str(error)
    else:
        raise AssertionError("implicit lexical position encoder accepted")

    standard = LexicalEmbedding(
        11,
        8,
        position_encoder=SinusoidalTokenPositionEncoder(8, base=10000.0),
    )
    assert isinstance(standard.position_encoder, SinusoidalTokenPositionEncoder)

    learned = LexicalEmbedding(
        11,
        8,
        position_encoder=LearnedTokenPositionEncoder(8, max_length=32),
    )
    spec = component_spec(learned)
    position = spec.config["position_encoder"]
    assert position["reference"] == "arti/token-position-learned@1"
    restored = resolve_component(spec.reference, **spec.config)
    assert isinstance(restored.position_encoder, LearnedTokenPositionEncoder)
    assert restored.position_encoder.max_length == 32
