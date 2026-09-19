import json

import torch

from arti.alpha import (
    ContextEvaluator,
    LanguageHead,
    LexicalEmbedding,
    RecursiveContextCompiler,
    RecursiveLanguageModel,
    SinusoidalTokenPositionEncoder,
)
from arti.recursive_language import prefix_plan


def model():
    return RecursiveLanguageModel(
        ContextEvaluator(
            RecursiveContextCompiler(8, context_width=3, heads=2, layers=2),
            LexicalEmbedding(
                23,
                8,
                position_encoder=SinusoidalTokenPositionEncoder(8, base=10000.0),
            ),
        ),
        LanguageHead(8, 23),
    )


def test_language_loss_trains_recursive_readout_and_normal_head():
    torch.manual_seed(17)
    network = model()
    loss = network.token_loss([1, 2, 3, 4, 5, 6, 7], chunk_size=2, depth=1, fanout=2)
    loss.backward()
    assert network.evaluator.front.context_seed.grad is not None


def test_language_loss_with_variable_context_and_repeated_compilation():
    torch.manual_seed(23)
    network = model()
    plan = prefix_plan(
        [1, 2, 3, 4, 5, 6],
        [6],
        chunk_size=2,
        depth=1,
        fanout=2,
        context_width=2,
        compile_depth=3,
    )
    assert any(node.compile_depth == 3 for node in plan.nodes.values())
    result = network.evaluator(plan, root_output="downstream")
    assert any(
        value.shape == (1, 2, 8) for name, value in result.values.items() if name not in plan.roots
    )
    loss = network.token_loss(
        [1, 2, 3, 4, 5, 6, 7],
        chunk_size=2,
        depth=1,
        fanout=2,
        context_width=2,
        compile_depth=3,
    )
    loss.backward()
    assert network.evaluator.front.context_seed.grad is not None
    assert torch.isfinite(network.evaluator.front.context_seed.grad).all()
    assert network.language_head.lm_head.weight.grad is not None


def test_save_load_uses_only_current_schema(tmp_path):
    network = model().eval()
    plan = prefix_plan([1, 2, 3, 4], [4], chunk_size=2, depth=1, fanout=2)
    expected = network(plan)
    network.save(tmp_path)
    payload = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert payload["schema"] == "arti/recursive-language@1"
    assert "language_head" in payload["components"]
    assert "continuation" not in payload["components"]
    restored = RecursiveLanguageModel.load(tmp_path)
    torch.testing.assert_close(restored(plan), expected)
