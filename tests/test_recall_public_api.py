from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

import arti
import arti.nn as arti_nn


class AdditiveFormula(nn.Module):
    factor_names = ("content",)

    def forward(self, state: Tensor, factors: Tensor) -> Tensor:
        return state + factors[..., 0, :]


@pytest.mark.parametrize(
    "formula,slots",
    [("arti/delta@1", 4), ("arti/affine@1", 4), ("arti/state@1", 17)],
)
def test_recall_builtin_formulas_preserve_shape(formula: str, slots: int) -> None:
    layer = arti_nn.Recall(8, slots, formula=formula, activation="none")
    x = torch.randn(2, 3, 8)

    y = layer(x)

    assert y.shape == x.shape
    assert y.dtype == x.dtype
    assert layer.formula_id == formula


def test_recall_rejects_legacy_formula_aliases() -> None:
    with pytest.raises(ValueError, match="canonical|reference"):
        arti_nn.Recall(4, 4, formula="single")


def test_custom_formula_uses_bank_factors_and_propagates_gradients() -> None:
    layer = arti_nn.Recall(
        4,
        4,
        formula=AdditiveFormula(),
        activation="none",
    )
    with torch.no_grad():
        layer.state.recall.bank.normal_()
    x = torch.randn(2, 3, 4, requires_grad=True)

    y = layer(x)
    y.square().mean().backward()

    assert layer.formula is not None
    assert layer.factor_names == ("content",)
    assert layer.state.recall.bank.grad is not None
    assert x.grad is not None
    assert torch.isfinite(layer.state.recall.bank.grad).all()


def test_custom_formula_defaults_to_independent_nonzero_factor_initialization() -> None:
    torch.manual_seed(7)
    layer = arti_nn.Recall(
        4,
        8,
        formula=AdditiveFormula(),
        activation="none",
    )
    x = torch.randn(2, 3, 4, requires_grad=True)

    layer(x).square().mean().backward()

    assert torch.count_nonzero(layer.state.recall.bank).item() == 32
    query_grad = layer.state.recall.query.weight.grad
    assert query_grad is None
    assert not layer.state.recall.query.weight.requires_grad


def test_custom_formula_requires_slots_divisible_by_factor_count() -> None:
    class TwoFactor(nn.Module):
        factor_names = ("left", "right")

        def forward(self, state: Tensor, factors: Tensor) -> Tensor:
            return state + factors.sum(dim=-2)

    with pytest.raises(ValueError, match="divisible"):
        arti_nn.Recall(4, 3, formula=TwoFactor())


def test_recall_mask_preserves_invalid_tokens_and_vector_rank() -> None:
    layer = arti_nn.Recall(4, 4, activation="none")
    sequence = torch.randn(2, 3, 4)
    mask = torch.tensor([[True, False, True], [False, True, True]])

    output = layer(sequence, mask=mask)
    vector_output = layer(sequence[:, 0], mask=mask[:, 0])

    torch.testing.assert_close(output[~mask], sequence[~mask])
    assert vector_output.shape == (2, 4)


def test_custom_formula_isolates_tokens_and_batch_items() -> None:
    class ReducingFormula(nn.Module):
        factor_names = ("content",)

        def forward(self, state: Tensor, factors: Tensor) -> Tensor:
            return state + state.mean() + factors[..., 0, :]

    layer = arti_nn.Recall(
        4,
        8,
        formula=ReducingFormula(),
        activation="none",
    ).eval()
    x = torch.randn(2, 3, 4)
    mask = torch.tensor([[True, False, True], [True, True, True]])
    changed = x.clone()
    changed[0, 1] = 1000
    changed[1] = -1000

    baseline = layer(x, mask=mask)
    perturbed = layer(changed, mask=mask)

    torch.testing.assert_close(perturbed[0, 0], baseline[0, 0])
    torch.testing.assert_close(perturbed[0, 2], baseline[0, 2])


def test_custom_formula_is_checked_at_the_runtime_vector_rank() -> None:
    class RankSensitiveFormula(nn.Module):
        factor_names = ("content",)

        def forward(self, state: Tensor, factors: Tensor) -> Tensor:
            if state.ndim != 1 or factors.ndim != 2:
                raise ValueError("expected runtime vector rank")
            return state + factors[0]

    layer = arti_nn.Recall(4, 4, formula=RankSensitiveFormula(), activation="none")

    assert layer(torch.randn(2, 3, 4)).shape == (2, 3, 4)


def test_recall_return_info_reports_existing_diagnostics() -> None:
    layer = arti_nn.Recall(4, 4)

    output, info = layer(torch.randn(2, 3, 4), return_info=True)

    assert output.shape == (2, 3, 4)
    assert "recall_effect_norm" in info
    assert "recall_steps_executed" in info


def test_recall_exposes_passive_formula_manifest() -> None:
    layer = arti_nn.Recall(4, 4, formula="arti/delta@1")

    manifest = layer.formula_manifest()

    assert manifest.id == "arti/delta"
    assert manifest.version == "1"
    assert manifest.factor_names == ("content",)
    assert manifest.origin == "builtin"
    assert manifest.portable


def test_explicit_registered_formula_can_be_resolved_without_discovery() -> None:
    reference = "tests/additive@1"
    arti.register_formula(reference, factory=AdditiveFormula)

    layer = arti_nn.Recall(4, 4, formula=reference)

    assert isinstance(layer.formula, AdditiveFormula)
    assert layer.formula_id == reference


def test_recall_is_exported_from_root_and_nn() -> None:
    assert arti.Recall is arti_nn.Recall
    assert arti.torch.Recall is arti_nn.Recall
    assert callable(arti.formula_dtype_supported)
    assert callable(arti.validate_formula)
    assert callable(arti.register_formula)
    assert callable(arti.list_formulas)


def test_recall_strictly_loads_public_one_x_state() -> None:
    source = arti_nn.Recall(4, 4, activation="none")
    with torch.no_grad():
        source.state.recall.bank.normal_()
    legacy_state = {
        key: value.detach().clone()
        for key, value in source.state_dict().items()
        if key != "state._state_input_retention"
    }
    legacy_state["state.recall.key_bank"] = torch.randn(4, 32)

    restored = arti_nn.Recall(4, 4, activation="none")
    expected_retention = restored.state._state_input_retention.detach().clone()
    restored.load_state_dict(legacy_state, strict=True)

    torch.testing.assert_close(
        restored.state.recall.bank,
        legacy_state["state.recall.bank"],
    )
    torch.testing.assert_close(
        restored.state.recall.query.weight,
        legacy_state["state.recall.query.weight"],
    )
    torch.testing.assert_close(restored.state._state_input_retention, expected_retention)
    assert not restored.state.recall.query.weight.requires_grad
