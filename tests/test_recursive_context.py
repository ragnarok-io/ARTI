import torch

from arti.alpha import (
    ContextEvaluator,
    ContextMemo,
    LexicalEmbedding,
    RecursiveContextCompiler,
    SinusoidalTokenPositionEncoder,
)
from arti.recursive_context import ContextNode, ContextPlan
from arti.recursive_language import prefix_plan


def evaluator():
    return ContextEvaluator(
        RecursiveContextCompiler(8, context_width=3, heads=2, layers=2),
        LexicalEmbedding(
            32,
            8,
            position_encoder=SinusoidalTokenPositionEncoder(8, base=10000.0),
        ),
    )


def test_recursive_children_are_context_but_roots_are_downstream_states():
    plan = prefix_plan(range(12), [12], chunk_size=3, depth=1, fanout=2)
    result = evaluator()(plan, root_output="downstream")
    root = plan.roots[0]
    assert result.values[root].shape == (1, 8)
    assert any(value.shape == (1, 3, 8) for name, value in result.values.items() if name != root)
    assert result.batches[-1].output_positions == 1


def test_batched_and_serial_recursive_evaluation_match():
    torch.manual_seed(13)
    model = evaluator().eval()
    plan = prefix_plan(range(15), [9, 12, 15], chunk_size=3, depth=1, fanout=2)
    batched = model(plan, root_output="downstream")
    serial = model(plan, serial=True, root_output="downstream")
    for left, right in zip(batched.roots, serial.roots):
        torch.testing.assert_close(left, right)


def test_variable_width_context_memo_round_trip(tmp_path):
    torch.manual_seed(29)
    model = evaluator().eval()
    plan = prefix_plan(
        range(12), [12], chunk_size=3, depth=1, fanout=2,
        context_width=2, compile_depth=2,
    )
    with torch.no_grad():
        memo = ContextMemo()
        first = model(plan, memo=memo, root_output="downstream")
        assert len(memo) > 0
        assert any(value.shape == (1, 2, 8) for value in memo._values.values())
        path = tmp_path / "memo.safetensors"
        memo.save(path, model)
        restored = ContextMemo.load(path, model)
        second = model(plan, memo=restored, root_output="downstream")
    assert second.cache_hits > 0
    torch.testing.assert_close(first.roots[0], second.roots[0])


def test_downstream_root_cannot_be_reused_as_recursive_context():
    plan = ContextPlan(
        {"tokens": tuple(range(6))},
        (
            ContextNode("child", "tokens", 0, 3, placements=((0.0, 0.0),)),
            ContextNode(
                "parent",
                "tokens",
                3,
                6,
                children=("child",),
                placements=((0.0, 0.0), (3.0, 0.0)),
            ),
        ),
        ("child", "parent"),
    )
    # A root that is also a child would require one value to be both C and Y.
    try:
        evaluator()(plan, root_output="downstream")
    except ValueError as error:
        assert "downstream root" in str(error)
    else:
        raise AssertionError("mixed recursive/downstream root must fail closed")
