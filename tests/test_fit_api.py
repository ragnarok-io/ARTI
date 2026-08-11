import copy
import json
from pathlib import Path

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import arti
from arti.config import STATE_RECALL_COMPOSITION_FACTOR
from arti.fit import ARTIProject
from arti.fit.artifacts import hash_tensor_state_dict, stable_json_sha256
from arti.fit.insertion import ARTIAdapterWrapper
from arti.fit.project import (
    _migrate_legacy_state_factor_banks,
    _migrate_legacy_recall_strength_gate,
)
from arti.fit.runtime import adapter_context
from arti.fit.scanner import run_model
from arti.recall_registry import UnknownRecallFormulaError, resolve_formula


class FitRecallFormula(nn.Module):
    recall_formula_contract = arti.RecallFormulaContract(
        identity=arti.FormulaIdentity("tests/fit-recall", 1),
        factors=(
            arti.FactorSpec("content", init="normal", init_scale=0.02),
            arti.FactorSpec("gate", init="normal", init_scale=0.02),
        ),
        identity_preserving=True,
    )

    def forward(
        self,
        state: torch.Tensor,
        factors: torch.Tensor,
    ) -> torch.Tensor:
        content, gate = factors.unbind(dim=-2)
        return state + torch.tanh(gate) * content


class SecondFitRecallFormula(nn.Module):
    recall_formula_contract = arti.RecallFormulaContract(
        identity=arti.FormulaIdentity("tests/fit-recall-second", 1),
        factors=(
            arti.FactorSpec("gain", init="zero"),
            arti.FactorSpec("shift", init="zero"),
        ),
        identity_preserving=True,
    )

    def forward(
        self,
        state: torch.Tensor,
        factors: torch.Tensor,
    ) -> torch.Tensor:
        gain, shift = factors.unbind(dim=-2)
        return state * torch.exp(torch.tanh(gain)) + torch.tanh(shift)


def register_fit_recall_formula() -> str:
    reference = "tests/fit-recall@1"
    try:
        resolve_formula(reference)
    except UnknownRecallFormulaError:
        arti.register_formula(reference, factory=FitRecallFormula)
    return reference


def register_second_fit_recall_formula() -> str:
    reference = "tests/fit-recall-second@1"
    try:
        resolve_formula(reference)
    except UnknownRecallFormulaError:
        arti.register_formula(reference, factory=SecondFitRecallFormula)
    return reference


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


def test_configure_preserves_declarative_wrapper_controls():
    config = arti.FitProjectConfig(
        where=("0",),
        identity_gate=True,
        boundary_mask_key="attention_mask",
        require_runtime_context=True,
    )
    project = arti.project(tiny_model()).configure(config).scan(torch.randn(2, 4))
    plan = project.plan_insert()

    assert plan.spec.identity_gate is True
    assert plan.spec.boundary_mask_key == "attention_mask"
    assert plan.spec.require_runtime_context is True
    assert config.to_dict()["insertion"]["identity_gate"] is True
    assert config.to_dict()["insertion"]["boundary_mask_key"] == "attention_mask"
    assert config.to_dict()["insertion"]["require_runtime_context"] is True


def test_boundary_mask_key_resolves_positional_host_argument_and_preserves_padding():
    class MaskedBlock(nn.Module):
        def forward(
            self, hidden_states: torch.Tensor, attention_mask: torch.Tensor
        ) -> torch.Tensor:
            return hidden_states * 2.0

    class RecordingAdapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mask = None

        def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
            self.mask = kwargs.get("mask")
            return x + 1.0

    adapter = RecordingAdapter()
    wrapper = ARTIAdapterWrapper(
        MaskedBlock(),
        adapter,
        freeze_base=False,
        boundary_mask_key="attention_mask",
    )
    x = torch.randn(2, 3, 4)
    mask = torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool)

    output = wrapper(x, mask)
    base = x * 2.0

    assert torch.equal(adapter.mask, mask)
    assert torch.equal(output[~mask], base[~mask])
    assert torch.equal(output[mask], (base + 1.0)[mask])


def test_runtime_adapter_scale_controls_residual_without_changing_state_dict():
    class AddOne(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + 1.0

    wrapper = ARTIAdapterWrapper(
        nn.Identity(),
        AddOne(),
        freeze_base=False,
    )
    x = torch.randn(2, 3, 4)
    state_keys = tuple(wrapper.state_dict())

    assert arti.set_adapter_scale(wrapper, 0.0) == 1
    assert torch.equal(wrapper(x), x)
    assert arti.set_adapter_scale(wrapper, 0.5) == 1
    assert torch.allclose(wrapper(x), x + 0.5)
    assert arti.set_adapter_scale(wrapper, 1.0) == 1
    assert torch.allclose(wrapper(x), x + 1.0)
    assert arti.set_adapter_scale(wrapper, 2.0) == 1
    assert torch.allclose(wrapper(x), x + 2.0)
    assert tuple(wrapper.state_dict()) == state_keys

    for invalid in (-0.1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite and non-negative"):
            arti.set_adapter_scale(wrapper, invalid)


def test_runtime_recall_refine_steps_support_fixed_adaptive_and_bypass_modes():
    adapter = arti.ARTIResidualBlock(
        dim=4,
        recall_slots=4,
        recall_steps=3,
        recall_min_steps=2,
        recall_tolerance=0.01,
        use_phase_mixer=False,
        use_virtual_interface=False,
        zero_init_output=True,
        direct_recall=True,
    )
    wrapper = ARTIAdapterWrapper(
        nn.Identity(),
        adapter,
        freeze_base=False,
    )
    state = {name: value.clone() for name, value in wrapper.state_dict().items()}

    assert arti.set_recall_refine_steps(wrapper, 7) == 1
    assert adapter.layer.config.recall_steps == 7
    assert adapter.layer.state.config.recall_min_steps == 7
    assert adapter.layer.state.config.recall_tolerance is None

    assert arti.set_recall_refine_steps(
        wrapper,
        10,
        min_steps=2,
        tolerance=0.003,
    ) == 1
    assert adapter.layer.config.recall_steps == 10
    assert adapter.layer.state.config.recall_min_steps == 2
    assert adapter.layer.state.config.recall_tolerance == pytest.approx(0.003)

    assert arti.set_recall_refine_steps(wrapper, 0) == 1
    assert adapter.layer.config.recall_steps == 0
    assert adapter.layer.state.config.recall_min_steps == 0
    x = torch.randn(2, 3, 4)
    torch.testing.assert_close(wrapper(x), x, rtol=0.0, atol=0.0)
    for name, value in wrapper.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0.0, atol=0.0)

    for invalid in (-1, 1.5, True):
        with pytest.raises(ValueError, match="non-negative integer"):
            arti.set_recall_refine_steps(wrapper, invalid)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="zero Recall"):
        arti.set_recall_refine_steps(wrapper, 0, tolerance=0.1)
    with pytest.raises(ValueError, match=r"\[1, steps\]"):
        arti.set_recall_refine_steps(wrapper, 2, min_steps=3, tolerance=0.1)


def test_runtime_recall_refine_schedule_supports_ordered_and_named_depths():
    def make_wrapper() -> ARTIAdapterWrapper:
        return ARTIAdapterWrapper(
            nn.Identity(),
            arti.ARTIResidualBlock(
                dim=4,
                recall_slots=2,
                recall_steps=1,
                use_phase_mixer=False,
                use_virtual_interface=False,
                direct_recall=True,
            ),
            freeze_base=False,
        )

    model = nn.Sequential(make_wrapper(), make_wrapper(), make_wrapper())

    assert arti.set_recall_refine_schedule(model, (0, 2, 4)) == 3
    for wrapper, depth in zip(model, (0, 2, 4), strict=True):
        assert wrapper.adapter.layer.config.recall_steps == depth
        assert wrapper.adapter.layer.state.config.recall_min_steps == depth

    assert arti.set_recall_refine_schedule(model, {"0": 5, "1": 3, "2": 1}) == 3
    for wrapper, depth in zip(model, (5, 3, 1), strict=True):
        assert wrapper.adapter.layer.config.recall_steps == depth
        assert wrapper.adapter.layer.state.config.recall_min_steps == depth

    with pytest.raises(ValueError, match="length must match"):
        arti.set_recall_refine_schedule(model, (1, 2))
    with pytest.raises(ValueError, match="paths must match"):
        arti.set_recall_refine_schedule(model, {"0": 1, "1": 2, "missing": 3})
    before = tuple(
        wrapper.adapter.layer.config.recall_steps for wrapper in model
    )
    with pytest.raises(ValueError, match="non-negative integer"):
        arti.set_recall_refine_schedule(model, (1, -1, 2))
    assert tuple(
        wrapper.adapter.layer.config.recall_steps for wrapper in model
    ) == before


def test_runtime_recall_refine_schedule_updates_nested_wrappers_once():
    def make_wrapper(base: nn.Module) -> ARTIAdapterWrapper:
        return ARTIAdapterWrapper(
            base,
            arti.ARTIResidualBlock(
                dim=4,
                recall_slots=2,
                recall_steps=1,
                use_phase_mixer=False,
                use_virtual_interface=False,
                direct_recall=True,
            ),
            freeze_base=False,
        )

    inner = make_wrapper(nn.Identity())
    outer = make_wrapper(inner)

    assert arti.set_recall_refine_schedule(outer, {"": 2, "base": 4}) == 2
    assert outer.adapter.layer.config.recall_steps == 2
    assert inner.adapter.layer.config.recall_steps == 4
    assert arti.set_recall_refine_schedule(nn.Identity(), ()) == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_adapter_wrapper_supports_fp32_master_weights_with_bfloat16_host():
    wrapper = ARTIAdapterWrapper(
        nn.Identity(),
        nn.Linear(4, 4, bias=False, dtype=torch.float32),
        freeze_base=False,
    ).cuda()
    value = torch.randn(2, 3, 4, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        output = wrapper(value)

    assert output.dtype == torch.bfloat16
    assert output.shape == value.shape
    assert torch.isfinite(output).all()


def test_direct_recall_wrapper_preserves_host_dtype():
    class FloatDirectRecall(nn.Module):
        direct_recall = True

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x.float() + 0.25

    wrapper = ARTIAdapterWrapper(
        nn.Identity(),
        FloatDirectRecall(),
        freeze_base=False,
    )
    value = torch.randn(2, 3, 4, dtype=torch.bfloat16)

    output = wrapper(value)

    assert output.dtype == torch.bfloat16
    torch.testing.assert_close(output, (value.float() + 0.25).to(torch.bfloat16))

    wrapper.influence_scale = 0.5
    assert wrapper(value).dtype == torch.bfloat16

    gated = ARTIAdapterWrapper(
        nn.Identity(),
        FloatDirectRecall(),
        freeze_base=False,
        identity_gate=True,
    )
    with torch.no_grad():
        assert gated.output_gate is not None
        gated.output_gate.fill_(0.5)
    assert gated(value).dtype == torch.bfloat16


def test_adapter_context_forwards_recall_route_assignment_without_changing_values():
    class RecordingDirectRecall(nn.Module):
        direct_recall = True

        def __init__(self) -> None:
            super().__init__()
            self.route_assignment = None

        def forward(
            self,
            x: torch.Tensor,
            *,
            route_assignment: torch.Tensor | None = None,
        ) -> torch.Tensor:
            self.route_assignment = route_assignment
            return x

    adapter = RecordingDirectRecall()
    wrapper = ARTIAdapterWrapper(nn.Identity(), adapter, freeze_base=False)
    value = torch.randn(2, 3, 4)
    assignment = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    with adapter_context(route_assignment=assignment):
        output = wrapper(value)

    assert torch.equal(output, value)
    assert adapter.route_assignment is not None
    assert torch.equal(adapter.route_assignment, assignment)


def test_adapter_route_override_survives_without_runtime_context():
    class RecordingDirectRecall(nn.Module):
        direct_recall = True

        def __init__(self) -> None:
            super().__init__()
            self.route_assignment = None

        def forward(
            self,
            x: torch.Tensor,
            *,
            route_assignment: torch.Tensor | None = None,
        ) -> torch.Tensor:
            self.route_assignment = route_assignment
            return x

    adapter = RecordingDirectRecall()
    wrapper = ARTIAdapterWrapper(nn.Identity(), adapter, freeze_base=False)
    assignment = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    wrapper._route_assignment_override = assignment

    output = wrapper(torch.randn(2, 3, 4))

    assert output.shape == (2, 3, 4)
    assert adapter.route_assignment is not None
    assert torch.equal(adapter.route_assignment, assignment)


def test_recall_route_context_is_not_forwarded_to_recall_free_adapter():
    model = nn.Sequential(nn.Linear(4, 4))
    sample = torch.randn(2, 3, 4)
    arti.fit(
        model,
        sample_batch=sample,
        config=arti.FitProjectConfig(
            where=("0",),
            mechanism=arti.MechanismOverrides(recall_steps=0),
        ),
    )
    assignment = torch.tensor([[1.0, 0.0], [0.0, 1.0]])

    with adapter_context(route_assignment=assignment):
        output = model(sample)

    assert output.shape == sample.shape


def test_configure_supports_zero_initialized_identity_output():
    config = arti.FitProjectConfig(
        where=("0",),
        zero_init_output=True,
    )
    project = arti.project(tiny_model()).configure(config).scan(torch.randn(2, 4))
    plan = project.plan_insert()

    assert plan.spec.identity_gate is False
    assert plan.spec.zero_init_output is True
    assert config.to_dict()["insertion"]["zero_init_output"] is True

    result = arti.fit(tiny_model(), config=config, sample_batch=torch.randn(2, 4))
    wrapper = result.model[0]
    assert isinstance(wrapper, ARTIAdapterWrapper)
    assert wrapper.output_gate is None
    assert wrapper.adapter.zero_init_output is True
    assert wrapper.adapter.out.mode == "radial"


def test_radial_bridge_budget_is_shared_across_selected_path():
    sample = torch.randn(2, 4)
    builder = arti.project(tiny_model()).scan(sample)
    plan = builder.plan_insert(
        where=("0", "2"),
        zero_init_output=True,
        bridge_mode="radial",
    )

    assert len(plan.selected) == 2
    assert sum(item.residual_budget**2 for item in plan.selected) == pytest.approx(1.0)
    assert all(item.bridge_mode == "radial" for item in plan.selected)

    builder.insert(
        where=("0", "2"),
        zero_init_output=True,
        bridge_mode="radial",
    )
    wrappers = [
        module for module in builder.model.modules() if isinstance(module, ARTIAdapterWrapper)
    ]
    assert [wrapper.adapter.out.residual_budget for wrapper in wrappers] == pytest.approx(
        [2**-0.5, 2**-0.5]
    )


def test_legacy_dense_bridge_artifact_keys_load_without_semantic_change(tmp_path: Path):
    torch.manual_seed(19)
    sample = torch.randn(2, 4)
    base = nn.Sequential(nn.Linear(4, 4))
    base_state = {key: value.detach().clone() for key, value in base.state_dict().items()}
    source = arti.fit(
        base,
        config=arti.FitProjectConfig(
            where=("0",),
            zero_init_output=True,
            bridge_mode="dense",
        ),
        sample_batch=sample,
    )
    source_wrapper = source.model[0]
    with torch.no_grad():
        source_wrapper.adapter.out.linear.weight.normal_(std=0.1)
        source_wrapper.adapter.out.linear.bias.normal_(std=0.1)
    expected = source.model(sample)
    artifact = source.export(tmp_path / "dense.pt")
    payload = torch.load(artifact, weights_only=False)

    legacy_state = {}
    for key, value in payload["adapter_state_dict"].items():
        legacy_key = key.replace(".adapter.out.linear.", ".adapter.out.")
        legacy_state[legacy_key] = value
    payload["adapter_state_dict"] = legacy_state
    payload["manifest"]["adapter_state_sha256"] = hash_tensor_state_dict(legacy_state)
    payload["report"]["insertion"].pop("bridge_mode", None)
    payload["report"]["fit_config"]["insertion"].pop("bridge_mode", None)
    payload["manifest"]["report_sha256"] = stable_json_sha256(payload["report"])
    torch.save(payload, artifact)

    fresh = nn.Sequential(nn.Linear(4, 4))
    fresh.load_state_dict(base_state)
    applied = arti.apply_adapter(fresh, artifact, sample_batch=sample)
    fresh_wrapper = applied.model[0]

    assert fresh_wrapper.adapter.out.mode == "dense"
    assert torch.allclose(applied.model(sample), expected)


def test_fit_config_rejects_two_identity_initializers():
    with pytest.raises(ValueError, match="mutually exclusive"):
        arti.FitProjectConfig(
            identity_gate=True,
            zero_init_output=True,
        ).validate()


def test_repeated_composite_strategy_is_model_name_independent():
    class Composite(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.tanh(self.proj(x))

    model = nn.Sequential(Composite(), Composite(), nn.Linear(4, 2))
    project = arti.project(model).scan(torch.randn(2, 3, 4))
    plan = project.plan_insert(
        where="repeated-composites",
        max_adapters=2,
        max_extra_params="10000%",
    )

    assert [item.module_path for item in plan.selected] == ["0", "1"]
    assert all(item.dim == 4 for item in plan.selected)


def test_repeated_stages_prefers_shallow_dominant_stack():
    class Inner(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.tanh(self.proj(x))

    class Stage(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = Inner()
            self.second = Inner()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.second(self.first(x))

    class Stack(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.stages = nn.ModuleList(Stage() for _ in range(4))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for stage in self.stages:
                x = stage(x)
            return x

    project = arti.project(Stack()).scan(torch.randn(2, 3, 4))
    plan = project.plan_insert(
        where="repeated-stages",
        max_adapters=4,
        max_extra_params="10000%",
    )

    assert [item.module_path for item in plan.selected] == [
        "stages.0",
        "stages.1",
        "stages.2",
        "stages.3",
    ]


def test_repeated_stages_does_not_mix_sibling_stacks_of_the_same_type():
    class Stage(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.proj(x)

    class Stack(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.auxiliary = nn.ModuleList(Stage() for _ in range(2))
            self.layers = nn.ModuleList(Stage() for _ in range(6))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            for stage in self.auxiliary:
                x = stage(x)
            for stage in self.layers:
                x = stage(x)
            return x

    plan = (
        arti.project(Stack())
        .scan(torch.randn(2, 3, 4))
        .plan_insert(
            where="repeated-stages",
            max_adapters=6,
            max_extra_params="10000%",
        )
    )

    assert [item.module_path for item in plan.selected] == [f"layers.{index}" for index in range(6)]


def test_freeze_base_freezes_modules_outside_selected_boundaries():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 4))

    result = arti.fit(
        model,
        sample_batch=torch.randn(2, 4),
        target_modules="0",
        scale="tiny",
        freeze_base=True,
    )

    assert result.adapter_count == 1
    assert not any(parameter.requires_grad for parameter in model[2].parameters())
    assert any(parameter.requires_grad for parameter in model[0].adapter.parameters())


def test_artifact_rehydrates_mechanism_overrides_and_identity_gate(tmp_path: Path):
    sample = torch.randn(2, 3, 4)
    config = arti.FitProjectConfig(
        where=("0",),
        scale="tiny",
        mechanism=arti.MechanismOverrides(
            hidden_multiplier=0.5,
            operator_count=1,
            interface_slots=2,
            recall_slots=1,
            recall_steps=0,
            recall_recognition_mode="alignment",
        ),
        identity_gate=True,
    )
    source = arti.fit(
        nn.Sequential(nn.Linear(4, 4)),
        config=config,
        sample_batch=sample,
    )
    artifact = source.export(tmp_path / "custom-mechanism.pt")
    fresh = nn.Sequential(nn.Linear(4, 4))

    arti.apply_adapter(fresh, artifact, sample_batch=sample)

    assert isinstance(fresh[0], ARTIAdapterWrapper)
    assert fresh[0].output_gate is not None
    assert fresh[0].adapter.layer.config.hidden_dim == 2
    assert fresh[0].adapter.layer.config.operator_count == 1
    assert fresh[0].adapter.layer.config.recall_recognition_mode == "alignment"


def test_fit_recall_recognition_defaults_off() -> None:
    result = arti.fit(
        nn.Sequential(nn.Linear(4, 4)),
        sample_batch=torch.randn(2, 3, 4),
        target_modules="0",
        scale="base",
    )
    wrapper = result.model[0]

    assert isinstance(wrapper, ARTIAdapterWrapper)
    assert wrapper.adapter.layer.config.recall_recognition_mode == "none"


def test_fit_bank_first_budget_makes_recall_bank_the_largest_parameter_surface() -> None:
    model = nn.Sequential(nn.Linear(16, 16))
    config = arti.FitProjectConfig(
        scale="base",
        mechanism=arti.MechanismOverrides(
            recall_steps=1,
            recall_bank_fraction=0.6,
            recall_routing="grouped",
            recall_key_dim=8,
            recall_group_size=16,
            recall_group_topk=2,
        ),
        where=("0",),
        max_adapters=1,
        max_extra_params=200_000,
    )

    project = arti.project(model).configure(config).scan(torch.randn(2, 3, 16)).insert()
    inserted = project.report().inserted[0]
    wrapper = model[0]
    assert isinstance(wrapper, ARTIAdapterWrapper)
    bank_parameters = sum(
        parameter.numel()
        for name, parameter in wrapper.adapter.named_parameters()
        if name.rsplit(".", 1)[-1] == "bank" or name.rsplit(".", 1)[-1].endswith("_bank")
    )

    assert inserted.parameters <= 200_000
    assert inserted.recall_routing == "grouped"
    assert inserted.recall_slots % inserted.recall_group_size == 0
    assert inserted.recall_bank_parameters == bank_parameters
    assert inserted.recall_bank_fraction > 0.5
    assert inserted.recall_bank_fraction == pytest.approx(bank_parameters / inserted.parameters)
    summary = project.report().summary
    assert summary.recall_bank_parameters == bank_parameters
    assert summary.recall_bank_parameter_ratio == pytest.approx(inserted.recall_bank_fraction)

    target = torch.randn(2, 3, 16)
    loss = torch.nn.functional.mse_loss(model(torch.randn(2, 3, 16)), target)
    loss.backward()
    bank_gradients = [
        parameter.grad
        for name, parameter in wrapper.adapter.named_parameters()
        if name.rsplit(".", 1)[-1] == "bank" or name.rsplit(".", 1)[-1].endswith("_bank")
    ]
    assert bank_gradients
    assert all(gradient is not None for gradient in bank_gradients)
    assert any(torch.count_nonzero(gradient).item() > 0 for gradient in bank_gradients)


def test_fit_bank_first_budget_supports_state_recall_without_extra_parameters() -> None:
    model = nn.Sequential(nn.Linear(16, 16))
    config = arti.FitProjectConfig(
        scale="base",
        mechanism=arti.MechanismOverrides(
            recall_steps=1,
            recall_bank_fraction=0.6,
            recall_routing="grouped",
            recall_key_dim=8,
            recall_group_size=16,
            recall_group_topk=1,
            recall_value_composition="state",
        ),
        where=("0",),
        max_adapters=1,
        max_extra_params=200_000,
        zero_init_output=False,
    )
    sample = torch.randn(2, 3, 16)
    result = arti.fit(model, config=config, sample_batch=sample)
    wrapper = result.model[0]

    assert isinstance(wrapper, ARTIAdapterWrapper)
    assert wrapper.adapter.layer.config.recall_value_composition == "state"
    groups = (
        wrapper.adapter.layer.config.recall_slots // wrapper.adapter.layer.config.recall_group_size
    )
    assert groups % STATE_RECALL_COMPOSITION_FACTOR == 0
    assert result.report.inserted[0].parameters <= 200_000
    assert torch.isfinite(model(sample)).all()


def test_fit_recall_attachment_uses_host_write_bank() -> None:
    model = nn.Sequential(nn.Linear(8, 8))
    sample = torch.randn(2, 3, 8)
    expected = model(sample).detach()
    config = arti.FitProjectConfig(
        scale="base",
        mechanism=arti.MechanismOverrides(
            recall_steps=2,
            recall_bank_fraction=0.6,
            recall_routing="grouped",
            recall_key_dim=4,
            recall_group_size=4,
            recall_group_topk=1,
        ),
        where=("0",),
        max_adapters=1,
        max_extra_params=20_000,
        zero_init_output=True,
    )

    result = arti.fit(model, config=config, sample_batch=sample)
    wrapper = result.model[0]

    assert isinstance(wrapper, ARTIAdapterWrapper)
    assert wrapper.adapter.direct_recall is True
    assert not hasattr(wrapper.adapter, "out")
    assert not hasattr(wrapper.adapter.layer, "out_proj")
    assert wrapper.adapter.layer.config.hidden_dim == 8
    assert wrapper.adapter.layer.state.recall.bank.shape[1] == 8
    assert result.report.inserted[0].bridge_mode == "recall-write"
    torch.testing.assert_close(model(sample), expected, rtol=0.0, atol=0.0)


def test_fit_uses_registered_formula_and_records_artifact_contract(
    tmp_path: Path,
) -> None:
    torch.manual_seed(43)
    reference = register_fit_recall_formula()
    source_model = nn.Sequential(nn.Linear(8, 8))
    base_state = copy.deepcopy(source_model.state_dict())
    sample = torch.randn(2, 3, 8)
    expected_base = source_model(sample).detach()
    config = arti.FitProjectConfig(
        scale="base",
        mechanism=arti.MechanismOverrides(
            recall_steps=1,
            recall_bank_fraction=0.6,
            recall_routing="grouped",
            recall_key_dim=4,
            recall_group_size=4,
            recall_group_topk=1,
            recall_formula=reference,
        ),
        where=("0",),
        max_adapters=1,
        max_extra_params=20_000,
        zero_init_output=True,
    )

    source = arti.fit(source_model, config=config, sample_batch=sample)
    source_wrapper = source_model[0]
    assert isinstance(source_wrapper, ARTIAdapterWrapper)
    field = source_wrapper.adapter.layer.state.recall
    assert isinstance(field.formula, FitRecallFormula)
    assert field.factor_names == ("content", "gate")
    assert source.report.mechanism is not None
    assert source.report.mechanism.recall_formula == reference
    assert (
        source.report.mechanism.recall_formula_contract
        == FitRecallFormula.recall_formula_contract.to_dict()
    )
    assert (
        source.report.mechanism.recall_formula_contract_fingerprint
        == FitRecallFormula.recall_formula_contract.fingerprint
    )
    assert source.report.fit_config is not None
    assert source.report.fit_config["mechanism"]["recall_formula"] == reference
    torch.testing.assert_close(source_model(sample), expected_base, rtol=0.0, atol=0.0)

    source_wrapper.adapter.layer.state.set_state_input_retention(0.375)
    expected = source_model(sample)
    assert not torch.equal(expected, expected_base)
    artifact = source.export(tmp_path / "formula-recall.pt")

    fresh = nn.Sequential(nn.Linear(8, 8))
    fresh.load_state_dict(base_state)
    applied = arti.apply_adapter(fresh, artifact, sample_batch=sample)
    fresh_wrapper = applied.model[0]

    assert isinstance(fresh_wrapper, ARTIAdapterWrapper)
    assert isinstance(fresh_wrapper.adapter.layer.state.recall.formula, FitRecallFormula)
    assert fresh_wrapper.adapter.layer.state.state_input_retention == 0.375
    torch.testing.assert_close(fresh(sample), expected)

    incompatible = nn.Sequential(nn.Linear(8, 8))
    incompatible.load_state_dict(base_state)
    with pytest.raises(ValueError, match="cannot be replaced"):
        arti.apply_adapter(
            incompatible,
            artifact,
            sample_batch=sample,
            mechanism_overrides={"recall_formula": "arti/delta@1"},
        )


def test_stacked_fit_reports_only_its_own_recall_formula_contract() -> None:
    first_reference = register_fit_recall_formula()
    second_reference = register_second_fit_recall_formula()
    model = nn.Sequential(nn.Linear(8, 8))
    sample = torch.randn(2, 3, 8)

    def config(reference: str) -> arti.FitProjectConfig:
        return arti.FitProjectConfig(
            scale="base",
            mechanism=arti.MechanismOverrides(
                recall_steps=1,
                recall_bank_fraction=0.6,
                recall_routing="grouped",
                recall_key_dim=4,
                recall_group_size=4,
                recall_group_topk=1,
                recall_formula=reference,
            ),
            where=("0",),
            max_adapters=1,
            max_extra_params=20_000,
            zero_init_output=True,
        )

    first = arti.fit(model, config=config(first_reference), sample_batch=sample)
    second = arti.fit(model, config=config(second_reference), sample_batch=sample)

    assert (
        first.report.mechanism.recall_formula_contract_fingerprint
        == FitRecallFormula.recall_formula_contract.fingerprint
    )
    assert (
        second.report.mechanism.recall_formula_contract_fingerprint
        == SecondFitRecallFormula.recall_formula_contract.fingerprint
    )


def test_direct_recall_artifact_rehydrates_without_output_bridge(
    tmp_path: Path,
) -> None:
    torch.manual_seed(47)
    source_model = nn.Sequential(nn.Linear(8, 8))
    base_state = copy.deepcopy(source_model.state_dict())
    sample = torch.randn(2, 3, 8)
    config = arti.FitProjectConfig(
        scale="base",
        mechanism=arti.MechanismOverrides(
            recall_steps=1,
            recall_bank_fraction=0.6,
            recall_routing="grouped",
            recall_key_dim=4,
            recall_group_size=4,
            recall_group_topk=1,
        ),
        where=("0",),
        max_adapters=1,
        max_extra_params=20_000,
        zero_init_output=True,
    )
    source = arti.fit(source_model, config=config, sample_batch=sample)
    source_wrapper = source_model[0]
    assert isinstance(source_wrapper, ARTIAdapterWrapper)
    with torch.no_grad():
        source_wrapper.adapter.layer.state.recall.bank.normal_(std=0.02)
    expected = source_model(sample)
    artifact = source.export(tmp_path / "direct-recall.pt")

    fresh = nn.Sequential(nn.Linear(8, 8))
    fresh.load_state_dict(base_state)
    applied = arti.apply_adapter(fresh, artifact, sample_batch=sample)
    fresh_wrapper = applied.model[0]

    assert isinstance(fresh_wrapper, ARTIAdapterWrapper)
    assert fresh_wrapper.adapter.direct_recall is True
    assert not hasattr(fresh_wrapper.adapter, "out")
    torch.testing.assert_close(fresh(sample), expected)


def test_state_recall_artifact_loads_with_training_residual_disabled(
    tmp_path: Path,
) -> None:
    torch.manual_seed(53)
    source_model = nn.Sequential(nn.Linear(8, 8))
    base_state = copy.deepcopy(source_model.state_dict())
    sample = torch.randn(2, 3, 8)
    config = arti.FitProjectConfig(
        scale="base",
        mechanism=arti.MechanismOverrides(
            recall_steps=1,
            recall_bank_fraction=0.6,
            recall_routing="grouped",
            recall_key_dim=4,
            recall_group_size=4,
            recall_group_topk=1,
            recall_value_composition="state",
        ),
        where=("0",),
        max_adapters=1,
        max_extra_params=20_000,
        zero_init_output=True,
    )
    source = arti.fit(source_model, config=config, sample_batch=sample)
    source_wrapper = source_model[0]
    assert isinstance(source_wrapper, ARTIAdapterWrapper)
    with torch.no_grad():
        source_wrapper.adapter.layer.state.recall.bank.normal_(std=0.02)
    source_wrapper.adapter.layer.state.mark_state_bank_calibrated()
    source_wrapper.adapter.layer.state.set_state_input_retention(0.0)
    expected = source_model(sample)
    artifact = source.export(tmp_path / "state-recall.pt")

    fresh = nn.Sequential(nn.Linear(8, 8))
    fresh.load_state_dict(base_state)
    applied = arti.apply_adapter(fresh, artifact, sample_batch=sample)
    fresh_wrapper = applied.model[0]

    assert fresh_wrapper.adapter.layer.state.state_input_retention == 0.0
    assert fresh_wrapper.adapter.layer.state.state_bank_calibrated is True
    torch.testing.assert_close(fresh(sample), expected)


def test_compile_adapter_hotpaths_compiles_only_attached_adapters(monkeypatch) -> None:
    model = nn.Sequential(nn.Linear(8, 8))
    sample = torch.randn(2, 3, 8)
    config = arti.FitProjectConfig(
        scale="base",
        mechanism=arti.MechanismOverrides(
            recall_steps=1,
            recall_bank_fraction=0.6,
            recall_routing="grouped",
            recall_key_dim=4,
            recall_group_size=4,
            recall_group_topk=1,
        ),
        where=("0",),
        max_adapters=1,
        max_extra_params=20_000,
        zero_init_output=True,
    )
    arti.fit(model, config=config, sample_batch=sample)
    wrapper = model[0]
    assert isinstance(wrapper, ARTIAdapterWrapper)
    calls = []

    def record_compile(*_args, **kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(wrapper.adapter.layer, "compile_write_hotpath", record_compile)

    count = arti.compile_adapter_hotpaths(
        model,
        mode="reduce-overhead",
        dynamic=False,
        fullgraph=False,
    )

    assert count == 1
    assert calls == [
        {
            "mode": "reduce-overhead",
            "dynamic": False,
            "fullgraph": False,
        }
    ]
    assert wrapper.adapter._compiled_static_recall_steps is True
    assert arti.compile_adapter_hotpaths(nn.Linear(8, 8)) == 0


def test_bank_fraction_and_explicit_recall_slots_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        arti.MechanismOverrides(
            recall_bank_fraction=0.6,
            recall_slots=128,
        ).validate()


def test_bank_first_dry_run_does_not_advance_torch_rng() -> None:
    model = nn.Sequential(nn.Linear(16, 16), nn.Linear(16, 16))
    project = (
        arti.project(model).profile("virtual-recall").scale("base").scan(torch.randn(2, 3, 16))
    )
    torch.manual_seed(101)
    before = torch.random.get_rng_state().clone()

    plan = project.plan_insert(where=("0", "1"), max_extra_params=400_000)

    assert len(plan.selected) == 2
    assert torch.equal(torch.random.get_rng_state(), before)


def test_scan_distinguishes_exact_input_and_output_boundaries():
    model = tiny_model()
    project = arti.project(model).scan(torch.randn(2, 4), positions=("input", "output"))
    candidates = {candidate.name: candidate for candidate in project.report().scanned.candidates}

    assert candidates["0::input"].module_path == "0"
    assert candidates["0::input"].position == "input"
    assert candidates["0::input"].tensor_path == ("args", 0)
    assert candidates["0::input"].dim == 4
    assert candidates["0"].module_path == "0"
    assert candidates["0"].position == "output"
    assert candidates["0"].tensor_path == ()
    assert candidates["0"].dim == 8


def test_input_boundary_insertion_uses_the_scanned_tensor_path():
    class KeywordLayer(nn.Module):
        def forward(self, *, hidden: torch.Tensor) -> torch.Tensor:
            return hidden.square()

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = KeywordLayer()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.layer(hidden=x)

    model = Model()
    sample = torch.randn(2, 3, 4)
    project = arti.project(model).scan(sample, positions="input")
    candidate = project.report().scanned.candidates[0]

    assert candidate.name == "layer::input"
    assert candidate.tensor_path == ("kwargs", "hidden")
    project.insert(where="layer", positions="input")
    output = model(sample)
    output.mean().backward()

    assert isinstance(model.layer, ARTIAdapterWrapper)
    assert model.layer.position == "input"
    assert model.layer.tensor_path == ("kwargs", "hidden")
    assert output.shape == sample.shape
    assert any(parameter.grad is not None for parameter in model.layer.adapter.parameters())


def test_input_boundary_requires_a_runtime_sample():
    with pytest.raises(ValueError, match="requires sample_batch"):
        arti.project(tiny_model()).scan(positions="input")


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


def test_failed_runtime_scan_restores_mode_and_removes_all_hooks():
    class Explodes(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = nn.Linear(4, 4)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.proj(x)
            raise RuntimeError("expected scan failure")

    model = Explodes().train()
    with pytest.raises(RuntimeError, match="expected scan failure"):
        arti.project(model).scan(torch.randn(2, 4), positions=("input", "output"))

    assert model.training
    assert all(not module._forward_hooks for module in model.modules())
    assert all(not module._forward_pre_hooks for module in model.modules())


def test_scan_and_insert_accepts_arbitrary_parameterless_tensor_layer():
    class TensorWarp(nn.Module):
        def forward(self, hidden: torch.Tensor):
            return {"metadata": torch.arange(hidden.shape[0]), "payload": [{"state": hidden.sin()}]}

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.input = nn.Linear(4, 4)
            self.warp = TensorWarp()
            self.output = nn.Linear(4, 2)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            hidden = self.warp(self.input(x))["payload"][0]["state"]
            return self.output(hidden)

    model = Model().double()
    sample = torch.randn(3, 4, dtype=torch.float64)
    project = arti.project(model).scan(sample)
    candidate = {item.name: item for item in project.report().scanned.candidates}["warp"]

    assert candidate.module_type == "TensorWarp"
    assert candidate.output_path == ("payload", 0, "state")
    assert candidate.dim == 4
    assert candidate.dtype == "torch.float64"
    project.insert(where="warp")
    result = model(sample)
    result.square().mean().backward()

    assert isinstance(model.warp, ARTIAdapterWrapper)
    assert result.shape == (3, 2)
    assert all(parameter.dtype == torch.float64 for parameter in model.warp.adapter.parameters())
    assert any(parameter.grad is not None for parameter in model.warp.adapter.parameters())


def test_arbitrary_rank_last_feature_boundary_is_shape_stable():
    class VolumeTokens(nn.Module):
        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            return tensor.cos()

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.volume = VolumeTokens()

        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            return self.volume(tensor)

    model = Model()
    sample = torch.randn(2, 3, 4, 5, 6)
    project = arti.project(model).scan(sample)
    candidate = {item.name: item for item in project.report().scanned.candidates}["volume"]

    assert candidate.tensor_rank == 5
    assert candidate.feature_axis == 4
    assert candidate.dim == 6
    project.insert(where="volume")
    output = model(sample)
    output.mean().backward()

    assert output.shape == sample.shape
    assert any(parameter.grad is not None for parameter in model.volume.adapter.parameters())


def test_explicit_feature_axis_handles_ambiguous_parameterless_channel_first_layer():
    class ChannelFirstWarp(nn.Module):
        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            return tensor.sin()

    model = nn.Sequential(ChannelFirstWarp())
    sample = torch.randn(2, 3, 4, 5)
    project = arti.project(model).scan(sample, feature_axis={"0": 1})
    candidate = project.report().scanned.candidates[0]

    assert candidate.feature_axis == 1
    assert candidate.dim == 3
    project.insert(where="0")
    output = model(sample)
    output.mean().backward()

    assert output.shape == sample.shape
    assert model[0].adapter.layer.config.input_dim == 3


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

    report = (
        arti.project(AttentionLike())
        .plugin("transformers")
        .scan(sample)
        .insert(where="attention")
        .report()
    )
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

    result = arti.fit(
        GRULike(), sample_batch=torch.randn(5, 2, 3), profile="recurrent", max_adapters=1
    )

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
    project = (
        arti.project(model)
        .plugin("vision-cnn")
        .scan(torch.randn(2, 3, 8, 8))
        .insert(max_adapters=2)
    )

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

    result = arti.fit(
        ConvLike(), sample_batch=torch.randn(2, 3, 8, 8), profile="cnn", max_adapters=1
    )

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

    project = (
        arti.project(model).profile("observer-phase", phases=2).scan(batch).insert(where="proj")
    )

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
    assert (
        project.report().parameters.trainable_adapter_parameters
        == project.report().parameters.adapter_parameters
    )
    assert project.report().inserted[0].name == "0"
    assert model(torch.randn(2, 4)).shape == (2, 2)


def test_project_insert_can_keep_base_trainable():
    model = tiny_model()
    project = arti.project(model).scan(torch.randn(2, 4)).insert(where="0", freeze_base=False)

    assert project.report().parameters is not None
    assert project.report().parameters.trainable_base_parameters > 0
    assert (
        project.report().parameters.trainable_adapter_parameters
        == project.report().parameters.adapter_parameters
    )
    assert project.report().to_dict()["parameters"]["frozen_base"] is False


def test_project_plan_insert_dry_run_does_not_mutate_model():
    model = tiny_model()
    project = arti.project(model).scan(torch.randn(2, 4))

    plan = project.plan_insert(where=["0", "2"], max_adapters=1)

    assert plan.selected[0].name == "0"
    assert plan.adapter_parameters == plan.selected[0].parameters
    assert isinstance(model[0], nn.Linear)
    project.insert(where=["0", "2"], max_adapters=1)
    assert [adapter.name for adapter in project.report().inserted] == [
        adapter.name for adapter in plan.selected
    ]
    assert isinstance(model[0], ARTIAdapterWrapper)


def test_project_plan_insert_records_budget_skips():
    model = tiny_model()
    project = arti.project(model).scan(torch.randn(2, 4))

    plan = project.plan_insert(where=["0", "2"], max_extra_params=1)

    assert not plan.selected
    assert [adapter.name for adapter in plan.skipped_budget] == ["0", "2"]
    assert plan.to_dict()["skipped_budget"]
    assert project.report().insertion_plan is plan


def test_plan_supports_explicit_exclusions_and_per_boundary_scales():
    model = tiny_model()
    project = arti.project(model).scan(torch.randn(2, 4))

    plan = project.plan_insert(
        where="*",
        exclude="1",
        scale_pattern={"0": "tiny", "2": "large"},
        max_extra_params="10000%",
    )
    by_name = {adapter.name: adapter for adapter in plan.selected}

    assert plan.excluded == ("1",)
    assert by_name["0"].scale == "tiny"
    assert by_name["2"].scale == "large"
    assert by_name["2"].parameters > by_name["0"].parameters
    assert by_name["0"].position == "output"
    assert by_name["0"].module_path == "0"
    assert plan.to_dict()["spec"]["scale_pattern"] == {"0": "tiny", "2": "large"}


def test_selection_rejects_two_boundary_sides_for_one_module():
    project = arti.project(nn.Sequential(nn.Linear(4, 4))).scan(
        torch.randn(2, 4),
        positions=("input", "output"),
    )

    with pytest.raises(ValueError, match="one tensor boundary per module"):
        project.plan_insert(where="0", positions=("input", "output"))


def test_fluent_at_declares_the_complete_boundary_policy():
    report = (
        arti.project(tiny_model())
        .at(
            ["0", "2"],
            exclude="2",
            positions="output",
            scale_pattern={"0": "tiny"},
        )
        .preview(torch.randn(2, 4))
    )

    assert report.insertion_plan is not None
    assert [adapter.name for adapter in report.insertion_plan.selected] == ["0"]
    assert report.insertion_plan.selected[0].scale == "tiny"
    assert report.insertion_plan.excluded == ("2",)


def test_input_boundary_and_scale_rehydrate_from_adapter_artifact(tmp_path: Path):
    sample = torch.randn(2, 4)
    source_model = nn.Sequential(nn.Linear(4, 4))
    source = arti.fit(
        source_model,
        sample_batch=sample,
        target_modules="0",
        positions="input",
        scale_pattern={"0": "tiny"},
    )
    artifact = source.export(tmp_path / "input-boundary.pt")
    fresh = nn.Sequential(nn.Linear(4, 4))

    applied = arti.apply_adapter(fresh, artifact, sample_batch=sample)

    assert isinstance(fresh[0], ARTIAdapterWrapper)
    assert fresh[0].position == "input"
    assert fresh[0].tensor_path == ("args", 0)
    assert applied.report.inserted[0].name == "0::input"
    assert applied.report.inserted[0].scale == "tiny"
    assert fresh(sample).shape == sample.shape


def test_apply_adapter_rejects_changed_tensor_path_before_mutation(tmp_path: Path):
    class SourceLayer(nn.Module):
        def forward(self, x: torch.Tensor):
            return {"hidden_state": x.sin()}

    class TargetLayer(nn.Module):
        def forward(self, x: torch.Tensor):
            return {"payload": x.sin()}

    class Model(nn.Module):
        def __init__(self, layer: nn.Module, key: str) -> None:
            super().__init__()
            self.layer = layer
            self.key = key

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.layer(x)[self.key]

    sample = torch.randn(2, 4)
    source = arti.fit(
        Model(SourceLayer(), "hidden_state"),
        sample_batch=sample,
        target_modules="layer",
    )
    artifact = source.export(tmp_path / "structured-boundary.pt")
    target = Model(TargetLayer(), "payload")

    with pytest.raises(ValueError, match="changed tensor_path"):
        arti.apply_adapter(target, artifact, sample_batch=sample)

    assert isinstance(target.layer, TargetLayer)


def test_fit_config_parses_precise_insertion_control():
    config = arti.validate_fit_config(
        {
            "insertion": {
                "where": ["blocks.*"],
                "exclude": ["*.norm"],
                "positions": ["input", "output"],
                "scale_pattern": {"blocks.0*": "tiny", "blocks.8*": "base"},
            }
        }
    )

    assert config.exclude == ("*.norm",)
    assert config.positions == ("input", "output")
    assert dict(config.scale_pattern) == {"blocks.0*": "tiny", "blocks.8*": "base"}
    assert config.to_dict()["insertion"]["positions"] == ["input", "output"]


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
    assert (
        validated["report"]["insertion_plan"]["adapter_parameters"]
        == payload["report"]["insertion_plan"]["adapter_parameters"]
    )


def test_validate_plan_rejects_budget_gate_mismatch(tmp_path: Path):
    plan_path = (
        arti.project(tiny_model())
        .scan(torch.randn(2, 4))
        .write_plan(
            tmp_path / "budget-plan.json",
            where="0",
            max_adapters=1,
            max_extra_params="10000%",
        )
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
                    "recall_steps": 3,
                    "recall_min_steps": 1,
                    "recall_tolerance": 0.01,
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
    assert mechanism.recall_steps == 3
    assert mechanism.recall_min_steps == 1
    assert mechanism.recall_tolerance == pytest.approx(0.01)
    assert mechanism.hidden_multiplier == 1.5
    assert isinstance(wrapper, ARTIAdapterWrapper)
    assert wrapper.adapter.layer.config.coord_dim == 3
    assert wrapper.adapter.layer.config.hidden_dim == 8
    assert wrapper.adapter.layer.state.recall.bank.shape[1] == 8
    assert wrapper.adapter.layer.config.operator_count == 5
    assert wrapper.adapter.layer.config.recall_steps == 3
    assert wrapper.adapter.layer.config.recall_min_steps == 1
    assert wrapper.adapter.layer.config.recall_tolerance == pytest.approx(0.01)
    assert project.report().fit_config["mechanism"]["operator_count"] == 5


def test_fit_config_mechanism_overrides_survive_convenience_api(tmp_path: Path):
    config_path = tmp_path / "arti.json"
    config_path.write_text(
        json.dumps(
            {
                "fit": {"profile": "observer-phase", "scale": "small"},
                "mechanism": {
                    "coord_dim": 5,
                    "operator_count": 6,
                    "interface_slots": 7,
                    "recall_slots": 2,
                },
                "insertion": {"where": "0", "max_adapters": 1},
            }
        ),
        encoding="utf-8",
    )

    result = arti.fit(
        tiny_model(), config=config_path, sample_batch=torch.randn(2, 4), dry_run=True
    )

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
        .mechanism(
            observer_phase=True,
            coord_dim=4,
            coord_frame_mode="operator_bank",
            operator_count=3,
            interface_slots=5,
            recall_slots=2,
        )
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
        json.dumps(
            {"fit": {"profile": "virtual-recall", "scale": "base"}, "insertion": {"where": "0"}}
        ),
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

    assert (
        main(
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
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["expected_mechanism"]["coord_dim"] == 4
    assert output["mechanism"]["operator_count"] == 5
    assert output["expected_runtime_fields"]["mask_key"] == "token_mask"
    assert output["runtime"]["coord_key"] == "phase_coord"

    assert main(["validate", "config", str(config_path), "--expect-profile", "latent-adapt"]) == 1
    assert "profile" in capsys.readouterr().err
    assert main(["validate", "config", str(config_path), "--expect-scale", "base"]) == 1
    assert "scale" in capsys.readouterr().err
    assert (
        main(["validate", "config", str(config_path), "--expect-mechanism", "operator_count=4"])
        == 1
    )
    assert "mechanism.operator_count" in capsys.readouterr().err
    assert (
        main(
            [
                "validate",
                "config",
                str(config_path),
                "--expect-runtime-field",
                "mask_key=attention_mask",
            ]
        )
        == 1
    )
    assert "runtime.mask_key" in capsys.readouterr().err


def test_write_fit_config_template_round_trips_json_and_toml(tmp_path: Path):
    json_path = arti.write_fit_config_template(tmp_path / "arti.json")
    toml_path = arti.write_fit_config_template(
        tmp_path / "arti.toml", profile="virtual-recall", scale="base"
    )

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
    assert (
        json.loads(config_path.read_text(encoding="utf-8"))["$schema"]
        == "docs/reference/fit-config.schema.json"
    )
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
                "fit": {
                    "profile": "observer-phase",
                    "phases": 5,
                    "scale": "tiny",
                    "objectives": ["task-fit"],
                },
                "runtime": {"causal": True},
                "insertion": {"where": "0", "max_adapters": 1, "max_extra_params": "10000%"},
            }
        ),
        encoding="utf-8",
    )

    result = arti.fit(
        tiny_model(), config=config_path, sample_batch=torch.randn(2, 4), dry_run=True
    )

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

    result = arti.fit(
        model,
        train_loader=loader,
        sample_batch=x[:2],
        target_modules="0",
        freeze_base=True,
        steps=2,
    )
    artifact = result.export(tmp_path / "adapter.pt")
    report_path = result.write_report(tmp_path / "report.md")
    payload = torch.load(artifact, weights_only=False)

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
    assert (
        payload["report"]["parameters"]["trainable_adapter_parameters"]
        == result.report.parameters.trainable_adapter_parameters
    )
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
    payload = torch.load(artifact, weights_only=False)

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
    payload = torch.load(artifact, weights_only=False)

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
    assert [task.name for task in result.report.task_history] == [
        "preserve-output",
        "task-fit",
        "validate",
    ]
    assert [task.name for task in result.report.build_plan] == [
        "scan",
        "insert",
        "preserve-output",
        "task-fit",
        "validate",
    ]
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

    project = (
        arti.project(model).plugin("transformers").scan(torch.randn(2, 4)).insert(max_adapters=2)
    )
    names = [adapter.name for adapter in project.report().inserted]

    assert "0.attn.out_proj" in names
    assert "0.mlp.fc2" in names
    assert "transformers" in project.report().plugins
    details = project.report().to_dict()["plugin_details"]
    assert any(
        detail["name"] == "transformers" and "attention-output-strategy" in detail["capabilities"]
        for detail in details
    )


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
    payload = torch.load(artifact, weights_only=False)
    assert "adapter_state_dict" in payload
    assert applied.report.applied_artifact["path"] == str(artifact)
    assert (
        applied.report.applied_artifact["adapter_state_sha256"]
        == payload["manifest"]["adapter_state_sha256"]
    )
    assert (
        applied.report.to_dict()["applied_artifact"]["adapter_key_count"]
        == payload["manifest"]["adapter_key_count"]
    )
    assert "## Applied Artifact" in applied.report.to_markdown()
    assert fresh(torch.randn(2, 4)).shape == (2, 2)


def test_apply_adapter_can_reset_legacy_query_without_replacing_bank(
    tmp_path: Path,
) -> None:
    sample = torch.randn(2, 4)
    source_model = tiny_model()
    source = arti.fit(
        source_model,
        sample_batch=sample,
        config=arti.FitProjectConfig(
            where=("0",),
            mechanism=arti.MechanismOverrides(recall_steps=1),
        ),
    )
    source_wrapper = source_model[0]
    assert isinstance(source_wrapper, ARTIAdapterWrapper)
    source_field = source_wrapper.adapter.layer.state.recall
    with torch.no_grad():
        source_field.bank.normal_(mean=0.25, std=0.05)
        source_field.query.weight.fill_(7.0)
    expected_bank = source_field.bank.detach().clone()
    artifact = source.export(tmp_path / "legacy-query.pt")

    target = tiny_model()
    applied = arti.apply_adapter(
        target,
        artifact,
        sample_batch=sample,
        reset_recall_query=True,
    )

    target_wrapper = target[0]
    assert isinstance(target_wrapper, ARTIAdapterWrapper)
    target_field = target_wrapper.adapter.layer.state.recall
    torch.testing.assert_close(target_field.bank, expected_bank)
    assert not torch.all(target_field.query.weight == 7.0)
    assert not target_field.query.weight.requires_grad
    assert target_field.query_contract["mode"] == "fixed"
    assert applied.report.applied_artifact is not None
    assert applied.report.applied_artifact["migrations"] == [
        {"name": "reset-fixed-recall-query", "field_count": 1}
    ]


def test_apply_adapter_can_disable_identity_gate_without_losing_adapter_state(
    tmp_path: Path,
) -> None:
    torch.manual_seed(23)
    sample = torch.randn(2, 4)
    source_model = tiny_model()
    source = arti.fit(
        source_model,
        sample_batch=sample,
        config=arti.FitProjectConfig(
            where=("0",),
            mechanism=arti.MechanismOverrides(recall_steps=1),
            identity_gate=True,
        ),
    )
    source_wrapper = source_model[0]
    assert isinstance(source_wrapper, ARTIAdapterWrapper)
    with torch.no_grad():
        for parameter in source_wrapper.adapter.parameters():
            parameter.add_(torch.randn_like(parameter) * 0.01)
        assert source_wrapper.output_gate is not None
        source_wrapper.output_gate.fill_(-3.0)
    expected_adapter = {
        key: value.detach().clone() for key, value in source_wrapper.adapter.state_dict().items()
    }
    artifact = source.export(tmp_path / "gated-adapter.pt")

    target = tiny_model()
    applied = arti.apply_adapter(
        target,
        artifact,
        sample_batch=sample,
        identity_gate=False,
        mechanism_overrides=arti.MechanismOverrides(
            recall_steps=3,
            recall_min_steps=1,
            recall_tolerance=0.01,
        ),
    )

    target_wrapper = target[0]
    assert isinstance(target_wrapper, ARTIAdapterWrapper)
    assert target_wrapper.output_gate is None
    assert target_wrapper.adapter.layer.config.recall_steps == 3
    assert target_wrapper.adapter.layer.config.recall_min_steps == 1
    assert target_wrapper.adapter.layer.config.recall_tolerance == pytest.approx(0.01)
    assert applied.report.insertion is not None
    assert applied.report.insertion.identity_gate is False
    assert applied.report.applied_artifact is not None
    assert applied.report.applied_artifact["migrations"] == [
        {"name": "disable-identity-gate", "dropped_key_count": 1}
    ]
    for key, expected in expected_adapter.items():
        assert torch.equal(target_wrapper.adapter.state_dict()[key], expected)


def test_legacy_fit_recall_strength_gate_migration_is_narrow() -> None:
    keep = torch.tensor([1.0])
    legacy_weight = torch.ones(2, 4)
    legacy_bias = torch.ones(2)
    state = {
        "layers.0.adapter.layer.state.recall.bank": keep,
        "layers.0.adapter.layer.state.recall.gate.weight": legacy_weight,
        "layers.0.adapter.layer.state.recall.gate.bias": legacy_bias,
        "layers.0.adapter.layer.state.update_gate.weight": keep,
    }
    migrated, dropped = _migrate_legacy_recall_strength_gate(
        state,
        target_keys={
            "layers.0.adapter.layer.state.recall.bank",
            "layers.0.adapter.layer.state.update_gate.weight",
        },
    )

    assert dropped == (
        "layers.0.adapter.layer.state.recall.gate.bias",
        "layers.0.adapter.layer.state.recall.gate.weight",
    )
    assert set(migrated) == {
        "layers.0.adapter.layer.state.recall.bank",
        "layers.0.adapter.layer.state.update_gate.weight",
    }


@pytest.mark.parametrize("source_factor_count", [4, 8, 16])
def test_legacy_state_bank_migration_preserves_routes_and_modulation(
    source_factor_count: int,
) -> None:
    prefix = "layers.0.adapter.layer.state.recall"
    groups_per_factor = 3
    source_group_size = 2
    source_group_count = source_factor_count * groups_per_factor
    target_factor_count = 17
    target_group_count = target_factor_count * groups_per_factor
    source_rows = source_group_count * source_group_size
    target_rows = target_group_count
    source_bank = (
        torch.arange(source_rows * 3, dtype=torch.float32).reshape(
            source_rows,
            3,
        )
        / 500.0
    )
    source_keys = torch.arange(
        source_rows * 2,
        dtype=torch.float32,
    ).reshape(source_rows, 2)
    source_groups = torch.arange(
        source_group_count * 2,
        dtype=torch.float32,
    ).reshape(source_group_count, 2)
    state = {
        f"{prefix}.bank": source_bank,
        f"{prefix}.key_bank": source_keys,
        f"{prefix}.group_bank": source_groups,
    }
    target_state = {
        f"{prefix}.bank": torch.empty(target_rows, 3),
        f"{prefix}.key_bank": torch.empty(target_rows, 2),
        f"{prefix}.group_bank": torch.empty(target_group_count, 2),
    }

    migrated, changed, source_factors = _migrate_legacy_state_factor_banks(
        state,
        target_state=target_state,
    )

    assert changed == (
        f"{prefix}.bank",
        f"{prefix}.group_bank",
        f"{prefix}.key_bank",
    )
    assert source_factors == (source_factor_count,)
    assert migrated[f"{prefix}.bank"].shape == (target_rows, 3)
    assert migrated[f"{prefix}.key_bank"].shape == (target_rows, 2)
    assert migrated[f"{prefix}.group_bank"].shape == (
        target_group_count,
        2,
    )

    old_values = source_bank.reshape(
        source_factor_count,
        groups_per_factor,
        source_group_size,
        3,
    )[:, :, :1]
    new_values = migrated[f"{prefix}.bank"].reshape(
        target_factor_count,
        groups_per_factor,
        1,
        3,
    )
    torch.testing.assert_close(new_values[0], old_values[0])
    torch.testing.assert_close(new_values[1], torch.zeros_like(new_values[1]))
    torch.testing.assert_close(new_values[-2], old_values[-2])
    torch.testing.assert_close(new_values[-1], old_values[-1])
    expanded_modulation = (1.0 + torch.tanh(new_values[2:-2]) / 13.0).prod(dim=0)
    old_modulation_count = source_factor_count - 3
    original_modulation = (1.0 + torch.tanh(old_values[1:-2]) / old_modulation_count).prod(dim=0)
    torch.testing.assert_close(
        expanded_modulation,
        original_modulation,
        rtol=1e-5,
        atol=1e-6,
    )

    old_group_routes = source_groups.reshape(
        source_factor_count,
        groups_per_factor,
        2,
    )
    new_group_routes = migrated[f"{prefix}.group_bank"].reshape(
        target_factor_count,
        groups_per_factor,
        2,
    )
    torch.testing.assert_close(new_group_routes[0], old_group_routes[0])
    torch.testing.assert_close(new_group_routes[1], old_group_routes[0])
    torch.testing.assert_close(new_group_routes[-2], old_group_routes[-2])
    torch.testing.assert_close(new_group_routes[-1], old_group_routes[-1])
    repeats = tuple(
        13 // old_modulation_count + (index < 13 % old_modulation_count)
        for index in range(old_modulation_count)
    )
    cursor = 2
    for old_index, repeat_count in enumerate(repeats, start=1):
        for _repeat in range(repeat_count):
            torch.testing.assert_close(
                new_group_routes[cursor],
                old_group_routes[old_index],
            )
            cursor += 1


def test_apply_adapter_reports_structure_mismatch(tmp_path: Path):
    source = arti.fit(
        tiny_model(), sample_batch=torch.randn(2, 4), target_modules="0", freeze_base=True
    )
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

    digest = torch.load(artifact, weights_only=False)["manifest"]["adapter_state_sha256"]
    assert (
        main(
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
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    payload = torch.load(artifact, weights_only=False)
    saved_state = torch.load(state_path, weights_only=False)

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
    assert output["task_graph"]["tasks"][-1]["depends_on"] == [
        "write-apply-report",
        "write-state-dict",
    ]
    assert output["adapter_count"] == 1
    assert output["cli_max_adapters"] == 1
    assert output["lock"] == str(lock_path)
    assert output["lock_report_sha256"] == payload["manifest"]["report_sha256"]
    assert output["expected_adapter_state_sha256"] == digest
    assert output["adapter_state_sha256"] == payload["manifest"]["adapter_state_sha256"]
    assert report["applied_artifact"]["path"] == str(artifact)
    assert (
        report["applied_artifact"]["adapter_state_sha256"]
        == payload["manifest"]["adapter_state_sha256"]
    )
    assert state_path.exists()
    assert deployment_path.exists()
    assert any(".adapter." in key for key in saved_state)

    assert (
        main(
            ["validate", "artifact", str(artifact), "--expect-runtime-field", "mask_key=token_mask"]
        )
        == 0
    )
    validated_artifact = json.loads(capsys.readouterr().out)
    assert validated_artifact["runtime"]["mask_key"] == "token_mask"
    assert validated_artifact["expected_runtime_fields"]["mask_key"] == "token_mask"

    assert (
        main(
            [
                "validate",
                "artifact",
                str(artifact),
                "--expect-runtime-field",
                "mask_key=attention_mask",
            ]
        )
        == 1
    )
    assert "runtime.mask_key" in capsys.readouterr().err

    assert (
        main(["validate", "lock", str(lock_path), "--expect-runtime-field", "mask_key=token_mask"])
        == 0
    )
    validated_lock = json.loads(capsys.readouterr().out)
    assert validated_lock["runtime"]["mask_key"] == "token_mask"

    assert (
        main(
            [
                "validate",
                "state-dict",
                str(state_path),
                "--expect-state-dict-sha256",
                output["saved_state_dict_sha256"],
            ]
        )
        == 0
    )
    validate_state_output = json.loads(capsys.readouterr().out)
    assert validate_state_output["state_dict_sha256"] == output["saved_state_dict_sha256"]
    assert validate_state_output["expected_state_dict_sha256"] == output["saved_state_dict_sha256"]

    assert (
        main(
            [
                "validate",
                "task-graph",
                str(apply_task_graph_path),
                "--expect-kind",
                "apply",
                "--expect-artifact",
                f"deployment={deployment_path}",
                "--require-existing-artifacts",
            ]
        )
        == 0
    )
    validate_task_graph_output = json.loads(capsys.readouterr().out)
    assert validate_task_graph_output["command_kind"] == "apply"
    assert validate_task_graph_output["artifacts"]["deployment"] == str(deployment_path)
    assert validate_task_graph_output["missing_artifacts"] == []

    assert (
        main(["validate", "state-dict", str(state_path), "--expect-state-dict-sha256", "wrong"])
        == 1
    )
    assert "state_dict_sha256" in capsys.readouterr().err

    assert (
        main(
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
        )
        == 0
    )
    validated_deployment = json.loads(capsys.readouterr().out)
    assert validated_deployment["state_dict_sha256"] == output["saved_state_dict_sha256"]
    assert (
        validated_deployment["expected_adapter_state_sha256"]
        == payload["manifest"]["adapter_state_sha256"]
    )
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

    assert (
        main(
            [
                "validate",
                "deployment",
                str(deployment_path),
                "--expect-adapter-state-sha256",
                "wrong",
            ]
        )
        == 1
    )
    assert "adapter_state_sha256" in capsys.readouterr().err

    assert (
        main(
            ["validate", "deployment", str(deployment_path), "--expect-state-dict-sha256", "wrong"]
        )
        == 1
    )
    assert "state_dict_sha256" in capsys.readouterr().err

    assert (
        main(["validate", "deployment", str(deployment_path), "--expect-profile", "observer-phase"])
        == 1
    )
    assert "profile" in capsys.readouterr().err

    assert main(["validate", "deployment", str(deployment_path), "--expect-scale", "base"]) == 1
    assert "scale" in capsys.readouterr().err

    assert (
        main(
            [
                "validate",
                "deployment",
                str(deployment_path),
                "--expect-mechanism",
                "operator_count=8",
            ]
        )
        == 1
    )
    assert "mechanism.operator_count" in capsys.readouterr().err

    assert (
        main(
            [
                "validate",
                "deployment",
                str(deployment_path),
                "--expect-runtime-field",
                "mask_key=attention_mask",
            ]
        )
        == 1
    )
    assert "runtime.mask_key" in capsys.readouterr().err

    assert main(["validate", "deployment", str(deployment_path), "--max-adapters", "0"]) == 1
    assert "max_adapters" in capsys.readouterr().err

    assert main(["validate", "deployment", str(deployment_path), "--max-extra-params", "1"]) == 1
    assert "max_extra_params" in capsys.readouterr().err

    failed_state_path = tmp_path / "failed-state.pt"
    assert (
        main(
            [
                "apply",
                "apply_fixture_model:make_model",
                str(artifact),
                str(report_path),
                "--sample-shape",
                "2,4",
                "--max-adapters",
                "0",
                "--save-state-dict",
                str(failed_state_path),
            ]
        )
        == 1
    )
    assert not failed_state_path.exists()
    assert "max_adapters" in capsys.readouterr().err

    assert (
        main(
            [
                "apply",
                "apply_fixture_model:make_model",
                str(artifact),
                str(report_path),
                "--sample-shape",
                "2,4",
                "--save-state-dict",
                str(state_path),
                "--deployment-output",
                str(tmp_path / "missing-lock-deployment.json"),
            ]
        )
        == 1
    )
    assert "requires --lock" in capsys.readouterr().err

    assert (
        main(
            [
                "apply",
                "apply_fixture_model:make_model",
                str(artifact),
                str(report_path),
                "--sample-shape",
                "2,4",
                "--lock",
                str(lock_path),
                "--deployment-output",
                str(tmp_path / "missing-state-deployment.json"),
            ]
        )
        == 1
    )
    assert "requires --save-state-dict" in capsys.readouterr().err

    assert (
        main(
            [
                "apply",
                "apply_fixture_model:make_model",
                str(artifact),
                str(report_path),
                "--sample-shape",
                "2,4",
                "--max-adapters",
                "0",
            ]
        )
        == 1
    )
    assert "max_adapters" in capsys.readouterr().err

    assert (
        main(
            [
                "apply",
                "apply_fixture_model:make_model",
                str(artifact),
                str(report_path),
                "--sample-shape",
                "2,4",
                "--expect-adapter-state-sha256",
                "wrong",
            ]
        )
        == 1
    )
    assert "adapter_state_sha256" in capsys.readouterr().err

    other_artifact = arti.fit(
        tiny_model(), sample_batch=sample, target_modules="2", freeze_base=True
    ).export(tmp_path / "other-adapter.pt")
    assert (
        main(
            [
                "apply",
                "apply_fixture_model:make_model",
                str(other_artifact),
                str(report_path),
                "--sample-shape",
                "2,4",
                "--lock",
                str(lock_path),
            ]
        )
        == 1
    )
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
    artifact = arti.fit(tiny_model(), config=config_path, sample_batch=torch.randn(2, 4)).export(
        tmp_path / "adapter.pt"
    )
    report_path = tmp_path / "applied.json"
    config = arti.load_fit_config(config_path)

    assert (
        main(
            [
                "apply",
                "apply_config_fixture_model:make_model",
                str(artifact),
                str(report_path),
                "--sample-shape",
                "2,4",
                "--expect-config",
                str(config_path),
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["config_fingerprint"] == config.fingerprint
    assert output["expected_config_fingerprint"] == config.fingerprint

    assert (
        main(
            [
                "apply",
                "apply_config_fixture_model:make_model",
                str(artifact),
                str(report_path),
                "--sample-shape",
                "2,4",
                "--expect-config",
                str(other_path),
            ]
        )
        == 1
    )
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
    payload = torch.load(artifact, weights_only=False)

    assert payload["report"]["task_history"][0]["name"] == "task-fit"
    assert payload["report"]["task_history"][0]["status"] == "success"
    assert payload["report"]["mechanism"]["scale"] == "small"
    assert payload["report"]["mechanism"]["recall_slots"] == 4
    assert [task["name"] for task in payload["report"]["build_plan"]] == [
        "scan",
        "insert",
        "task-fit",
    ]


def test_export_manifest_records_include_base(tmp_path: Path):
    model = tiny_model()
    result = arti.fit(model, sample_batch=torch.randn(2, 4), target_modules="0")

    artifact = result.export(tmp_path / "with-base.pt", include_base=True)
    payload = torch.load(artifact, weights_only=False)

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
    payload = torch.load(artifact, weights_only=False)
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
    artifact = result.export(
        tmp_path / "invalid-build.pt", build_metadata={"expected_plan": "plan.json"}
    )
    payload = torch.load(artifact, weights_only=False)
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
    payload = torch.load(artifact, weights_only=False)
    payload["manifest"]["package_version"] = ""
    torch.save(payload, artifact)

    with pytest.raises(ValueError, match="package_version"):
        arti.validate_artifact(artifact)


def test_validate_artifact_rejects_invalid_manifest_hash_shape(tmp_path: Path):
    model = tiny_model()
    result = arti.fit(model, sample_batch=torch.randn(2, 4), target_modules="0")
    artifact = result.export(tmp_path / "invalid-hash-shape.pt")
    payload = torch.load(artifact, weights_only=False)
    payload["manifest"]["adapter_state_sha256"] = "not-a-sha256"
    torch.save(payload, artifact)

    with pytest.raises(ValueError, match="64-character lowercase sha256"):
        arti.validate_artifact(artifact)


def test_validate_artifact_rejects_report_summary_mismatch(tmp_path: Path):
    model = tiny_model()
    result = arti.fit(model, sample_batch=torch.randn(2, 4), target_modules="0")
    artifact = result.export(tmp_path / "invalid-summary.pt")
    payload = torch.load(artifact, weights_only=False)
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
    path = arti.write_task_graph_artifact(
        tmp_path / "task-graph.json", command_kind="build", task_graph=graph
    )
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
    payload = torch.load(artifact, weights_only=False)
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
    payload = torch.load(artifact, weights_only=False)
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
    payload = torch.load(artifact, weights_only=False)
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
    plan_path = (
        arti.project(tiny_model()).configure(config).scan(sample).write_plan(tmp_path / "plan.json")
    )
    artifact = arti.fit(tiny_model(), config=config, sample_batch=sample).export(
        tmp_path / "adapter.pt"
    )
    lock_path = arti.create_build_lock(
        tmp_path / "arti.lock.json", artifact=artifact, plan=plan_path, config=config_path
    )

    payload = arti.validate_build_lock(lock_path)

    assert payload["kind"] == "build-lock"
    assert (
        payload["artifact"]["adapter_state_sha256"]
        == torch.load(artifact, weights_only=False)["manifest"]["adapter_state_sha256"]
    )
    assert (
        payload["artifact"]["report_sha256"]
        == torch.load(artifact, weights_only=False)["manifest"]["report_sha256"]
    )
    assert payload["plan"]["config_fingerprint"] == config.fingerprint
    assert payload["config"]["config_fingerprint"] == config.fingerprint


def test_build_lock_carries_artifact_build_metadata(tmp_path: Path):
    sample = torch.randn(2, 4)
    plan_path = (
        arti.project(tiny_model()).scan(sample).write_plan(tmp_path / "plan.json", where="0")
    )
    build_metadata = {
        "expected_plan": str(plan_path),
        "expected_plan_selected": ["0"],
    }
    artifact = arti.fit(tiny_model(), sample_batch=sample, target_modules="0").export(
        tmp_path / "adapter.pt",
        build_metadata=build_metadata,
    )
    lock_path = arti.create_build_lock(
        tmp_path / "arti.lock.json", artifact=artifact, plan=plan_path
    )

    payload = arti.validate_build_lock(lock_path)

    assert payload["artifact"]["build"] == build_metadata


def test_build_lock_stores_paths_relative_to_lockfile(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    config_path = artifacts_dir / "arti.json"
    config_path.write_text(json.dumps({"insertion": {"where": "0"}}), encoding="utf-8")
    sample = torch.randn(2, 4)
    arti.project(tiny_model()).configure(arti.load_fit_config(config_path)).scan(sample).write_plan(
        artifacts_dir / "plan.json"
    )
    artifact = arti.fit(tiny_model(), config=config_path, sample_batch=sample).export(
        artifacts_dir / "adapter.pt"
    )

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
    assert (
        arti.validate_build_lock(lock_path)["artifact"]["adapter_state_sha256"]
        == torch.load(artifact, weights_only=False)["manifest"]["adapter_state_sha256"]
    )


def test_validate_build_lock_rejects_changed_artifact(tmp_path: Path):
    sample = torch.randn(2, 4)
    artifact = arti.fit(tiny_model(), sample_batch=sample, target_modules="0").export(
        tmp_path / "adapter.pt"
    )
    lock_path = arti.create_build_lock(tmp_path / "arti.lock.json", artifact=artifact)
    payload = torch.load(artifact, weights_only=False)
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
    payload = torch.load(artifact, weights_only=False)
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
    artifact = arti.fit(tiny_model(), sample_batch=sample, target_modules="0").export(
        tmp_path / "adapter.pt"
    )
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
    applied_report.write_text(
        json.dumps({"applied_artifact": {"path": str(artifact)}}), encoding="utf-8"
    )
    torch.save(tiny_model().state_dict(), state_path)

    manifest_path = arti.create_deployment_manifest(
        tmp_path / "deployment.json",
        lock=lock_path,
        artifact=artifact,
        applied_report=applied_report,
        state_dict=state_path,
    )
    payload = arti.validate_deployment_manifest(manifest_path)

    assert payload["artifact"]["build"] == {
        "expected_plan": "plan.json",
        "expected_plan_selected": ["0"],
    }
    assert (
        payload["artifact"]["adapter_key_count"]
        == torch.load(artifact, weights_only=False)["manifest"]["adapter_key_count"]
    )


def test_deployment_manifest_rejects_changed_adapter_key_count(tmp_path: Path):
    sample = torch.randn(2, 4)
    artifact = arti.fit(tiny_model(), sample_batch=sample, target_modules="0").export(
        tmp_path / "adapter.pt"
    )
    lock_path = arti.create_build_lock(tmp_path / "arti.lock.json", artifact=artifact)
    applied_report = tmp_path / "applied.json"
    state_path = tmp_path / "patched-state.pt"
    applied_report.write_text(
        json.dumps({"applied_artifact": {"path": str(artifact)}}), encoding="utf-8"
    )
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
    artifact = arti.fit(tiny_model(), sample_batch=sample, target_modules="0").export(
        tmp_path / "adapter.pt"
    )
    lock_path = arti.create_build_lock(tmp_path / "arti.lock.json", artifact=artifact)
    applied_report = tmp_path / "applied.json"
    state_path = tmp_path / "patched-state.pt"
    state = tiny_model().state_dict()
    applied_report.write_text(
        json.dumps({"applied_artifact": {"path": str(artifact)}}), encoding="utf-8"
    )
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
    artifact = arti.fit(tiny_model(), sample_batch=sample, target_modules="0").export(
        tmp_path / "adapter.pt"
    )
    other_artifact = arti.fit(tiny_model(), sample_batch=sample, target_modules="2").export(
        tmp_path / "other-adapter.pt"
    )
    lock_path = arti.create_build_lock(tmp_path / "arti.lock.json", artifact=artifact)
    applied_report = tmp_path / "applied.json"
    state_path = tmp_path / "patched-state.pt"
    applied_report.write_text(
        json.dumps({"applied_artifact": {"path": str(other_artifact)}}), encoding="utf-8"
    )
    torch.save(tiny_model().state_dict(), state_path)
    manifest_path = arti.create_deployment_manifest(
        tmp_path / "deployment.json",
        lock=lock_path,
        artifact=artifact,
        applied_report=applied_report,
        state_dict=state_path,
    )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    other_payload = torch.load(other_artifact, weights_only=False)
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
    artifact = arti.fit(tiny_model(), sample_batch=sample, target_modules="0").export(
        tmp_path / "adapter.pt"
    )
    lock_path = arti.create_build_lock(tmp_path / "arti.lock.json", artifact=artifact)
    applied_report = tmp_path / "applied.json"
    state_path = tmp_path / "patched-state.pt"
    applied_report.write_text(
        json.dumps({"applied_artifact": {"path": str(artifact), "adapter_state_sha256": "wrong"}}),
        encoding="utf-8",
    )
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
    applied_report.write_text(
        json.dumps({"applied_artifact": {"path": str(artifact)}}), encoding="utf-8"
    )
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
    plan_path = (
        arti.project(model).scan(torch.randn(2, 4)).write_plan(tmp_path / "plan.json", where="0")
    )
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
    plan_path = (
        arti.project(tiny_model())
        .scan(torch.randn(2, 4))
        .write_plan(tmp_path / "bad-config-plan.json", where="0")
    )
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
    artifact_path = arti.fit(tiny_model(), sample_batch=sample, target_modules="0").export(
        tmp_path / "adapter.pt"
    )

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

    assert (
        main(
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
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))

    assert output["kind"] == "fit-plan"
    assert output["planned_count"] == 1
    assert output["output"] == str(plan_path)
    assert output["provenance"]["model"] == "fixture_model:make_model"
    assert output["provenance"]["sample_shape"] == [2, 4]
    assert output["provenance_fingerprint"] == arti.plan_provenance_fingerprint(
        output["provenance"]
    )
    assert payload["kind"] == "fit-plan"
    assert payload["provenance"]["target_modules"] == ["0"]
    assert payload["provenance_fingerprint"] == output["provenance_fingerprint"]
    assert payload["report"]["insertion_plan"]["selected"][0]["name"] == "0"
    assert payload["report"]["summary"]["inserted_count"] == 0


def test_cli_build_exports_adapter_artifact_from_importable_factory(
    tmp_path: Path, capsys, monkeypatch
):
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

    assert (
        main(
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
        )
        == 0
    )
    capsys.readouterr()

    assert (
        main(
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
        )
        == 0
    )
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
    assert [task["name"] for task in output["task_graph"]["tasks"]][-3:] == [
        "export-artifact",
        "write-report",
        "write-lock",
    ]
    assert output["task_graph"]["tasks"][-1]["depends_on"] == ["export-artifact"]
    assert output["adapter_count"] == 1
    assert output["adapter_state_sha256"] == payload["manifest"]["adapter_state_sha256"]
    assert output["report_sha256"] == payload["manifest"]["report_sha256"]
    assert output["expected_plan"] == str(plan_path)
    assert output["expected_plan_provenance_fingerprint"] == plan_payload["provenance_fingerprint"]
    assert output["build"]["expected_plan_selected"] == ["0"]
    assert payload["build"]["expected_plan"] == str(plan_path)
    assert payload["build"]["expected_plan_selected"] == ["0"]
    assert (
        payload["build"]["expected_plan_provenance_fingerprint"]
        == plan_payload["provenance_fingerprint"]
    )
    assert (
        payload["build"]["expected_plan_config_fingerprint"]
        == plan_payload["report"]["config_fingerprint"]
    )
    assert report["summary"]["inserted_count"] == 1

    assert (
        main(
            [
                "validate",
                "task-graph",
                str(build_task_graph_path),
                "--expect-kind",
                "build",
                "--expect-artifact",
                f"adapter={artifact_path}",
                "--require-existing-artifacts",
            ]
        )
        == 0
    )
    validated_task_graph = json.loads(capsys.readouterr().out)
    assert validated_task_graph["command_kind"] == "build"
    assert validated_task_graph["artifacts"]["adapter"] == str(artifact_path)
    assert validated_task_graph["missing_artifacts"] == []
    task_graph_payload = json.loads(build_task_graph_path.read_text(encoding="utf-8"))
    task_graph_payload["task_graph"]["artifacts"]["adapter"] = str(tmp_path / "missing-adapter.pt")
    build_task_graph_path.write_text(json.dumps(task_graph_payload), encoding="utf-8")
    assert (
        main(["validate", "task-graph", str(build_task_graph_path), "--require-existing-artifacts"])
        == 1
    )
    assert "artifacts are missing" in capsys.readouterr().err

    assert (
        main(
            [
                "validate",
                "artifact",
                str(artifact_path),
                "--max-adapters",
                "1",
                "--expect-plan",
                str(plan_path),
            ]
        )
        == 0
    )
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
    applied_report_path.write_text(
        json.dumps(
            {
                "applied_artifact": {
                    "path": str(artifact_path),
                    "adapter_state_sha256": payload["manifest"]["adapter_state_sha256"],
                }
            }
        ),
        encoding="utf-8",
    )
    torch.save(tiny_model().state_dict(), state_path)
    deployment_path = tmp_path / "deployment.json"
    assert (
        main(
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
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(["validate", "deployment", str(deployment_path), "--expect-plan", str(plan_path)]) == 0
    )
    validated_deployment = json.loads(capsys.readouterr().out)
    assert validated_deployment["expected_plan"] == str(plan_path)
    assert validated_deployment["build"]["expected_plan_selected"] == ["0"]

    mismatched_plan_path = tmp_path / "mismatched-plan.json"
    assert (
        main(
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
        )
        == 0
    )
    capsys.readouterr()
    assert (
        main(
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
        )
        == 1
    )
    assert "expected plan" in capsys.readouterr().err
    assert (
        main(
            ["validate", "artifact", str(artifact_path), "--expect-plan", str(mismatched_plan_path)]
        )
        == 1
    )
    assert "expected plan" in capsys.readouterr().err
    assert (
        main(["validate", "lock", str(lock_path), "--expect-plan", str(mismatched_plan_path)]) == 1
    )
    assert "expected plan" in capsys.readouterr().err
    assert (
        main(
            [
                "validate",
                "deployment",
                str(deployment_path),
                "--expect-plan",
                str(mismatched_plan_path),
            ]
        )
        == 1
    )
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

    assert (
        main(
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
                "3",
                "--mechanism-recall-min-steps",
                "1",
                "--mechanism-recall-tolerance",
                "0.01",
                "--mechanism-hidden-multiplier",
                "1.5",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    mechanism = payload["report"]["mechanism"]

    assert mechanism["coord_dim"] == 4
    assert mechanism["operator_count"] == 5
    assert mechanism["interface_slots"] == 6
    assert mechanism["recall_slots"] == 2
    assert mechanism["recall_steps"] == 3
    assert mechanism["recall_min_steps"] == 1
    assert mechanism["recall_tolerance"] == pytest.approx(0.01)
    assert mechanism["hidden_multiplier"] == 1.5
    assert payload["provenance"]["phases"] == 6
    assert payload["provenance"]["mechanism"]["operator_count"] == 5
    assert output["provenance_fingerprint"] == arti.plan_provenance_fingerprint(
        output["provenance"]
    )


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

    assert (
        main(
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
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    payload = json.loads(plan_path.read_text(encoding="utf-8"))

    assert payload["report"]["fit_config"]["runtime"]["mask_key"] == "token_mask"
    assert payload["report"]["fit_config"]["runtime"]["coord_key"] == "phase_coord"
    assert payload["report"]["fit_config"]["runtime"]["observer_coord_key"] == "observer_phase"
    assert payload["report"]["fit_config"]["runtime"]["frame_operators_key"] == "inverse_ops"
    assert payload["provenance"]["runtime_fields"]["mask_key"] == "token_mask"
    assert output["provenance"]["runtime_fields"]["coord_key"] == "phase_coord"
    assert output["provenance_fingerprint"] == arti.plan_provenance_fingerprint(
        output["provenance"]
    )


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
    kwargs_path.write_text(
        json.dumps({"input_dim": 4, "hidden_dim": 6, "output_dim": 2}), encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    plan_path = tmp_path / "kwarg-plan.json"

    assert (
        main(
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
        )
        == 0
    )
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

    assert (
        main(
            [
                "plan",
                "markdown_fixture_model:make_model",
                str(plan_path),
                "--sample-shape",
                "2,4",
                "--target-modules",
                "0",
            ]
        )
        == 0
    )
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
                "input_ids": {
                    "shape": [2, 5],
                    "dtype": "long",
                    "kind": "randint",
                    "low": 0,
                    "high": 16,
                },
                "attention_mask": {"shape": [2, 5], "dtype": "long", "kind": "ones"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    plan_path = tmp_path / "dict-plan.json"

    assert (
        main(
            [
                "plan",
                "dict_fixture_model:make_model",
                str(plan_path),
                "--sample-json",
                str(sample_path),
                "--profile",
                "transformer",
                "--target-modules",
                "proj",
                "--max-adapters",
                "1",
            ]
        )
        == 0
    )
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
    plan_path = (
        arti.project(tiny_model())
        .configure(config)
        .scan(torch.randn(2, 4))
        .write_plan(tmp_path / "plan.json")
    )

    assert main(["validate", "plan", str(plan_path), "--expect-config", str(config_path)]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["config_fingerprint"] == config.fingerprint
    assert output["expected_config_fingerprint"] == config.fingerprint


def test_cli_validate_plan_can_require_profile_and_scale(tmp_path: Path, capsys):
    from arti.cli import main

    plan_path = (
        arti.project(tiny_model())
        .profile("observer-phase")
        .scale("base")
        .scan(torch.randn(2, 4))
        .write_plan(tmp_path / "plan.json", where="0")
    )

    assert (
        main(
            [
                "validate",
                "plan",
                str(plan_path),
                "--expect-profile",
                "observer-phase",
                "--expect-scale",
                "base",
            ]
        )
        == 0
    )
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

    assert (
        main(
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
        )
        == 0
    )
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
    assert (
        main(
            [
                "validate",
                "plan",
                str(plan_path),
                "--expect-runtime-field",
                "mask_key=attention_mask",
            ]
        )
        == 1
    )
    assert "runtime.mask_key" in capsys.readouterr().err
    assert (
        main(["validate", "plan", str(plan_path), "--expect-runtime-field", "operator_count=6"])
        == 1
    )
    assert "unknown ARTI runtime field" in capsys.readouterr().err


def test_validate_plan_rejects_invalid_provenance(tmp_path: Path):
    plan_path = (
        arti.project(tiny_model())
        .scan(torch.randn(2, 4))
        .write_plan(tmp_path / "plan.json", where="0")
    )
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
        assert (
            main(
                [
                    "plan",
                    "fingerprint_fixture_model:make_model",
                    str(plan_path),
                    "--sample-shape",
                    "2,4",
                    "--target-modules",
                    "0",
                ]
            )
            == 0
        )
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
    plan_path = (
        arti.project(tiny_model())
        .configure(arti.load_fit_config(config_path))
        .scan(torch.randn(2, 4))
        .write_plan(tmp_path / "plan.json")
    )

    assert main(["validate", "plan", str(plan_path), "--expect-config", str(other_path)]) == 1
    assert "expected config" in capsys.readouterr().err


def test_cli_validate_plan_can_require_matching_provenance_fingerprint(
    tmp_path: Path, capsys, monkeypatch
):
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

    assert (
        main(
            [
                "plan",
                "gate_fixture_model:make_model",
                str(plan_path),
                "--sample-shape",
                "2,4",
                "--target-modules",
                "0",
            ]
        )
        == 0
    )
    plan_output = json.loads(capsys.readouterr().out)
    fingerprint = plan_output["provenance_fingerprint"]

    assert (
        main(["validate", "plan", str(plan_path), "--expect-provenance-fingerprint", fingerprint])
        == 0
    )
    validate_output = json.loads(capsys.readouterr().out)
    assert validate_output["expected_provenance_fingerprint"] == fingerprint

    assert (
        main(["validate", "plan", str(plan_path), "--expect-provenance-fingerprint", "wrong"]) == 1
    )
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

    artifact = arti.fit(tiny_model(), sample_batch=torch.randn(2, 4), target_modules="0").export(
        tmp_path / "adapter.pt"
    )
    payload = torch.load(artifact, weights_only=False)
    digest = payload["manifest"]["adapter_state_sha256"]

    assert (
        main(["validate", "artifact", str(artifact), "--expect-adapter-state-sha256", digest]) == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["adapter_state_sha256"] == digest
    assert output["expected_adapter_state_sha256"] == digest

    assert (
        main(["validate", "artifact", str(artifact), "--expect-adapter-state-sha256", "wrong"]) == 1
    )
    assert "adapter_state_sha256" in capsys.readouterr().err


def test_cli_validate_artifact_can_require_profile_and_scale(tmp_path: Path, capsys):
    from arti.cli import main

    artifact = arti.fit(
        tiny_model(),
        sample_batch=torch.randn(2, 4),
        target_modules="0",
        profile="observer-phase",
        scale="base",
    ).export(tmp_path / "adapter.pt")

    assert (
        main(
            [
                "validate",
                "artifact",
                str(artifact),
                "--expect-profile",
                "observer-phase",
                "--expect-scale",
                "base",
            ]
        )
        == 0
    )
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

    assert (
        main(
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
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["expected_mechanism"]["coord_dim"] == 5
    assert output["mechanism"]["interface_slots"] == 7

    assert (
        main(["validate", "artifact", str(artifact), "--expect-mechanism", "recall_slots=4"]) == 1
    )
    assert "mechanism.recall_slots" in capsys.readouterr().err


def test_cli_lock_creates_and_validates_build_lock(tmp_path: Path, capsys):
    from arti.cli import main

    sample = torch.randn(2, 4)
    config_path = tmp_path / "arti.json"
    config_path.write_text(
        json.dumps(
            {
                "fit": {"profile": "observer-phase", "phases": 16, "scale": "base"},
                "insertion": {"where": "0"},
            }
        ),
        encoding="utf-8",
    )
    config = arti.load_fit_config(config_path)
    plan_path = (
        arti.project(tiny_model())
        .configure(arti.load_fit_config(config_path))
        .scan(sample)
        .write_plan(tmp_path / "plan.json")
    )
    artifact = arti.fit(tiny_model(), sample_batch=sample, config=config_path).export(
        tmp_path / "adapter.pt"
    )
    lock_path = tmp_path / "arti.lock.json"

    assert (
        main(
            [
                "lock",
                str(lock_path),
                "--artifact",
                str(artifact),
                "--plan",
                str(plan_path),
                "--config",
                str(config_path),
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["kind"] == "build-lock"
    assert output["artifact"] == "adapter.pt"
    assert output["plan"] == "plan.json"
    assert len(output["adapter_state_sha256"]) == 64
    assert len(output["report_sha256"]) == 64

    assert (
        main(
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
        )
        == 0
    )
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
    assert (
        main(["validate", "lock", str(lock_path), "--expect-config", str(other_config_path)]) == 1
    )
    assert "expected config" in capsys.readouterr().err
    assert (
        main(["validate", "lock", str(lock_path), "--expect-provenance-fingerprint", "wrong"]) == 1
    )
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

    plan_path = (
        arti.project(tiny_model())
        .scan(torch.randn(2, 4))
        .write_plan(tmp_path / "plan.json", where="0")
    )

    assert (
        main(
            [
                "validate",
                "plan",
                str(plan_path),
                "--max-adapters",
                "1",
                "--max-extra-params",
                "100000",
            ]
        )
        == 0
    )
    output = json.loads(capsys.readouterr().out)
    assert output["cli_max_adapters"] == 1
    assert output["cli_max_extra_params"] == 100000


def test_cli_validate_plan_rejects_external_budget_limits(tmp_path: Path, capsys):
    from arti.cli import main

    plan_path = (
        arti.project(tiny_model())
        .scan(torch.randn(2, 4))
        .write_plan(tmp_path / "plan.json", where="0")
    )

    assert main(["validate", "plan", str(plan_path), "--max-adapters", "0"]) == 1
    assert "max_adapters" in capsys.readouterr().err
    assert main(["validate", "plan", str(plan_path), "--max-extra-params", "1"]) == 1
    assert "max_extra_params" in capsys.readouterr().err


def test_cli_validate_artifact_rejects_external_budget_limits(tmp_path: Path, capsys):
    from arti.cli import main

    artifact = arti.fit(tiny_model(), sample_batch=torch.randn(2, 4), target_modules="0").export(
        tmp_path / "adapter.pt"
    )

    assert main(["validate", "artifact", str(artifact), "--max-adapters", "0"]) == 1
    assert "max_adapters" in capsys.readouterr().err
    assert main(["validate", "artifact", str(artifact), "--max-extra-params", "1"]) == 1
    assert "max_extra_params" in capsys.readouterr().err


def test_cli_rejects_invalid_plan(tmp_path: Path, capsys):
    from arti.cli import main

    plan_path = (
        arti.project(tiny_model())
        .scan(torch.randn(2, 4))
        .write_plan(tmp_path / "bad-plan.json", where="0")
    )
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
