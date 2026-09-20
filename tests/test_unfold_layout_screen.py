from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import torch


SCRIPT = Path(__file__).parents[1] / "benchmarks" / "unfold_layout_screen.py"
SPEC = importlib.util.spec_from_file_location("unfold_layout_screen", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_hard_layout_is_exactly_value_preserving_and_bijective() -> None:
    config = MODULE.Config(batch_size=8, eval_size=8, steps=1)
    model = MODULE.LayoutModel("hard_ste", config)
    x = MODULE.make_batch(8, config, torch.device("cpu"))
    candidates = MODULE.candidates_from_input(x)
    output, source, _ = model(x)
    expected = MODULE.gather_candidates(candidates, source)
    assert torch.equal(output, expected)
    assert torch.equal(source.sort(dim=1).values, torch.arange(config.output_length).expand(8, -1))


def test_soft_layout_is_doubly_stochastic_but_not_atomic() -> None:
    config = MODULE.Config(batch_size=8, eval_size=8, steps=1)
    model = MODULE.LayoutModel("soft_sinkhorn", config)
    x = MODULE.make_batch(8, config, torch.device("cpu"))
    output, _, layout = model(x)
    candidates = MODULE.candidates_from_input(x)
    assert torch.allclose(layout.sum(-1), torch.ones_like(layout.sum(-1)), atol=1e-5)
    assert torch.allclose(layout.sum(-2), torch.ones_like(layout.sum(-2)), atol=1e-5)
    exact_candidate = (output[:, :, None, :] == candidates[:, None, :, :]).all(dim=-1).any(dim=-1)
    assert not exact_candidate.any()


def test_hard_ste_routes_layout_gradient() -> None:
    config = MODULE.Config(batch_size=8, eval_size=8, steps=1)
    model = MODULE.LayoutModel("hard_ste", config)
    x = MODULE.make_batch(8, config, torch.device("cpu"))
    target = MODULE.gather_candidates(MODULE.candidates_from_input(x), MODULE.target_source_index(x))
    output, _, _ = model(x)
    (output - target).square().mean().backward()
    assert model.base_logits.grad is not None
    assert model.control_logits.grad is not None
    assert torch.isfinite(model.base_logits.grad).all()
