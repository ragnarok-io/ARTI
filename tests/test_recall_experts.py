from __future__ import annotations

import copy
import importlib
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from arti.layered_recall import LayerRecall
from arti.layers import ARTILatentRecallField
from arti.fit import (
    concatenate_adapter_banks,
    set_adapter_bank_influences,
    set_adapter_bank_weights,
)
from arti.recall_formula import FactorSpec, FormulaIdentity, RecallFormulaContract
from arti.recall_experts import (
    RecallExpertAssembly,
    canonical_tensor_state_sha256,
    create_recall_expert_contract,
    export_recall_expert_bank,
    freeze_for_recall_expert,
    inspect_recall_expert_bank,
    load_recall_expert_bank,
    validate_recall_expert_contract,
)


class _FourFactorFormula(nn.Module):
    recall_formula_contract = RecallFormulaContract(
        identity=FormulaIdentity("tests/four-factor-expert", 1),
        factors=tuple(
            FactorSpec(f"factor_{index}", route=f"route_{index}", init="zero") for index in range(4)
        ),
        identity_preserving=True,
    )

    def forward(self, state: torch.Tensor, factors: torch.Tensor) -> torch.Tensor:
        return state + factors.sum(dim=-2)


class _Host(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(6, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(x)


def _template() -> LayerRecall:
    torch.manual_seed(7)
    return LayerRecall(6, rank=4, slots=2, recognition_mode="alignment")


def _export(
    tmp_path: Path,
    host: nn.Module,
    template: LayerRecall,
    contract,
    expert_id: str,
    offset: float,
) -> Path:
    expert = copy.deepcopy(template)
    freeze_for_recall_expert(host, expert)
    with torch.no_grad():
        expert.bank.add_(offset)
    path = tmp_path / f"{expert_id}.recall.arti.st"
    export_recall_expert_bank(expert, path, host=host, expert_id=expert_id, contract=contract)
    return path


def test_exact_tensor_hash_distinguishes_positive_and_negative_zero() -> None:
    positive = torch.tensor([0.0])
    negative = torch.tensor([-0.0])
    assert torch.equal(positive, negative)
    assert canonical_tensor_state_sha256({"x": positive}) != canonical_tensor_state_sha256(
        {"x": negative}
    )


def test_contract_rejects_host_shared_state_and_behavior_drift() -> None:
    host = _Host()
    expert = _template()
    contract = create_recall_expert_contract(
        host, expert, preset_id="toy.fixed.v1", revision="abc123"
    )

    changed_host = copy.deepcopy(host)
    with torch.no_grad():
        changed_host.proj.weight.add_(1.0)
    with pytest.raises(ValueError, match="host_state_sha256"):
        validate_recall_expert_contract(contract, changed_host, expert)

    changed_reader = copy.deepcopy(expert)
    with torch.no_grad():
        changed_reader.query.weight[0, 0] += 1.0
    with pytest.raises(ValueError, match="shared_state_sha256"):
        validate_recall_expert_contract(contract, host, changed_reader)

    changed_behavior = copy.deepcopy(expert)
    changed_behavior.recognition_mode = "none"
    with pytest.raises(ValueError, match="shared_config_sha256"):
        validate_recall_expert_contract(contract, host, changed_behavior)


def test_bank_only_freeze_training_export_and_load(tmp_path: Path) -> None:
    host = _Host()
    expert = _template()
    contract = create_recall_expert_contract(host, expert, preset_id="toy.fixed.v1")
    before_host = copy.deepcopy(host.state_dict())
    before_shared = {
        name: value.detach().clone()
        for name, value in expert.state_dict().items()
        if name != "bank"
    }
    selected = freeze_for_recall_expert(host, expert)
    assert [name for name, _ in selected] == ["bank"]
    assert [name for name, parameter in expert.named_parameters() if parameter.requires_grad] == [
        "bank"
    ]
    assert not any(parameter.requires_grad for parameter in host.parameters())

    optimizer = torch.optim.SGD([parameter for _, parameter in selected], lr=0.2)
    hidden = torch.randn(3, 5, 6)
    loss = (expert(hidden) - torch.ones_like(hidden)).square().mean()
    loss.backward()
    optimizer.step()

    assert all(torch.equal(before_host[name], value) for name, value in host.state_dict().items())
    assert all(
        torch.equal(before_shared[name], value)
        for name, value in expert.state_dict().items()
        if name != "bank"
    )

    path = tmp_path / "trained.recall.arti.st"
    export_recall_expert_bank(
        expert,
        path,
        host=host,
        expert_id="trained",
        contract=contract,
        training_metadata={"steps": 1},
    )
    asset = inspect_recall_expert_bank(path)
    assert set(asset.state_dict) == {"bank"}
    assert asset.training_metadata == {"steps": 1}
    exposed = asset.state_dict["bank"]
    exposed.zero_()
    assert not torch.equal(exposed, asset.state_dict["bank"])

    restored = _template()
    original_shared = {
        name: value.detach().clone()
        for name, value in restored.state_dict().items()
        if name != "bank"
    }
    load_recall_expert_bank(path, restored, contract=contract)
    assert torch.equal(restored.bank, expert.bank)
    assert all(
        torch.equal(original_shared[name], value)
        for name, value in restored.state_dict().items()
        if name != "bank"
    )


def test_private_extension_is_hashed_separately_and_ignored_by_bank_assembly(
    tmp_path: Path,
) -> None:
    host = _Host()
    expert = _template()
    contract = create_recall_expert_contract(host, expert, preset_id="toy.fixed.v1")
    freeze_for_recall_expert(host, expert)
    gate = nn.Sequential(nn.Linear(6, 3, bias=False), nn.Linear(3, 1))
    path = tmp_path / "private.recall.arti.st"

    export_recall_expert_bank(
        expert,
        path,
        host=host,
        expert_id="private",
        contract=contract,
        private_module=gate,
        private_metadata={"kind": "test.private", "version": 1},
    )
    asset = inspect_recall_expert_bank(path)

    assert asset.artifact_version == 2
    assert set(asset.state_dict) == {"bank"}
    assert set(asset.private_state or {}) == {"0.weight", "1.weight", "1.bias"}
    assert asset.private_state_sha256 == canonical_tensor_state_sha256(asset.private_state or {})
    assert asset.private_metadata == {"kind": "test.private", "version": 1}
    assembly = RecallExpertAssembly(_template(), contract)
    assembly.add(path)
    merged, layout = assembly.materialize()
    assert merged.bank.shape == expert.bank.shape
    assert layout.expert_ids == ("private",)


def test_private_extension_requires_module_and_metadata_together(tmp_path: Path) -> None:
    host = _Host()
    expert = _template()
    contract = create_recall_expert_contract(host, expert, preset_id="toy.fixed.v1")
    freeze_for_recall_expert(host, expert)

    with pytest.raises(ValueError, match="provided together"):
        export_recall_expert_bank(
            expert,
            tmp_path / "invalid.recall.arti.st",
            host=host,
            expert_id="invalid",
            contract=contract,
            private_module=nn.Linear(6, 1),
        )


def test_assembly_is_canonical_and_remove_rebuilds_fresh_result(tmp_path: Path) -> None:
    host = _Host()
    template = _template()
    contract = create_recall_expert_contract(host, template, preset_id="toy.fixed.v1")
    paths = {
        name: _export(tmp_path, host, template, contract, name, offset)
        for name, offset in (("gamma", 3.0), ("alpha", 1.0), ("beta", 2.0))
    }

    assembly = RecallExpertAssembly(template, contract)
    for name in ("gamma", "alpha", "beta"):
        assembly.add(paths[name])
    merged_abc, layout_abc = assembly.materialize()
    assert layout_abc.expert_ids == ("alpha", "beta", "gamma")
    assert merged_abc.bank.shape == (6, 4)
    assert merged_abc.slots == 6
    assert layout_abc.ranges["beta"]["bank"] == (2, 4)

    assembly.remove("beta")
    rebuilt_ac, layout_ac = assembly.materialize()
    fresh = RecallExpertAssembly(template, contract)
    fresh.add(paths["gamma"])
    fresh.add(paths["alpha"])
    fresh_ac, fresh_layout = fresh.materialize()
    assert layout_ac == fresh_layout
    assert all(
        torch.equal(value, fresh_ac.state_dict()[name])
        for name, value in rebuilt_ac.state_dict().items()
    )

    probe = torch.randn(2, 3, 6)
    assert torch.equal(rebuilt_ac(probe), fresh_ac(probe))


def test_grouped_recall_assembly_concatenates_values_keys_and_groups(tmp_path: Path) -> None:
    host = _Host()
    template = ARTILatentRecallField(
        6,
        8,
        routing="grouped",
        key_dim=4,
        group_size=4,
        group_topk=1,
    )
    contract = create_recall_expert_contract(
        host,
        template,
        preset_id="toy.grouped.v1",
    )
    paths = []
    for expert_id, offset in (("alpha", 1.0), ("beta", 2.0)):
        expert = copy.deepcopy(template)
        freeze_for_recall_expert(host, expert)
        with torch.no_grad():
            expert.bank.add_(offset)
            assert expert.key_bank is not None
            assert expert.group_bank is not None
            expert.key_bank.add_(offset)
            expert.group_bank.add_(offset)
        path = tmp_path / f"{expert_id}.recall.arti.st"
        export_recall_expert_bank(
            expert,
            path,
            host=host,
            expert_id=expert_id,
            contract=contract,
        )
        paths.append(path)

    assembly = RecallExpertAssembly(template, contract)
    assembly.replace(paths)
    merged, layout = assembly.materialize()

    assert merged.bank.shape == (16, 6)
    assert merged.key_bank is not None and merged.key_bank.shape == (16, 4)
    assert merged.group_bank is not None and merged.group_bank.shape == (4, 4)
    assert merged.slots == 16
    assert layout.ranges["alpha"]["bank"] == (0, 8)
    assert layout.ranges["beta"]["group_bank"] == (2, 4)
    read = merged(torch.randn(2, 3, 6), torch.ones(2, 3, dtype=torch.bool))
    assert read.context.shape == (2, 3, 6)


def test_formula_recall_assembly_preserves_factor_contiguous_layout(tmp_path: Path) -> None:
    host = _Host()
    template = ARTILatentRecallField(
        6,
        8,
        routing="grouped",
        key_dim=4,
        group_size=2,
        group_topk=1,
        formula=_FourFactorFormula(),
    )
    contract = create_recall_expert_contract(
        host,
        template,
        preset_id="toy.formula.v1",
    )
    paths = []
    for expert_id, expert_offset in (("alpha", 100.0), ("beta", 200.0)):
        expert = copy.deepcopy(template)
        freeze_for_recall_expert(host, expert)
        assert expert.key_bank is not None
        assert expert.group_bank is not None
        with torch.no_grad():
            for factor_index, (start, stop) in enumerate(expert.factor_slices):
                marker = expert_offset + factor_index * 10.0
                expert.bank[start:stop].fill_(marker)
                expert.key_bank[start:stop].fill_(marker + 1.0)
                expert.group_bank[factor_index : factor_index + 1].fill_(marker + 2.0)
        path = tmp_path / f"{expert_id}.recall.arti.st"
        export_recall_expert_bank(
            expert,
            path,
            host=host,
            expert_id=expert_id,
            contract=contract,
        )
        paths.append(path)

    assembly = RecallExpertAssembly(template, contract)
    assembly.replace(paths)
    merged, layout = assembly.materialize()

    expected_bank_markers = torch.tensor(
        [
            100.0,
            100.0,
            200.0,
            200.0,
            110.0,
            110.0,
            210.0,
            210.0,
            120.0,
            120.0,
            220.0,
            220.0,
            130.0,
            130.0,
            230.0,
            230.0,
        ]
    )
    assert torch.equal(merged.bank[:, 0], expected_bank_markers)
    assert merged.key_bank is not None
    assert torch.equal(merged.key_bank[:, 0], expected_bank_markers + 1.0)
    assert merged.group_bank is not None
    assert torch.equal(
        merged.group_bank[:, 0],
        torch.tensor([102.0, 202.0, 112.0, 212.0, 122.0, 222.0, 132.0, 232.0]),
    )
    assert merged.slots == 16
    assert merged.factor_slices == ((0, 4), (4, 8), (8, 12), (12, 16))
    assert layout.ranges["alpha"]["bank"] == (0, 8)
    assert layout.physical_ranges["alpha"]["bank"] == (
        (0, 2),
        (4, 6),
        (8, 10),
        (12, 14),
    )
    assert layout.physical_ranges["beta"]["group_bank"] == (
        (1, 2),
        (3, 4),
        (5, 6),
        (7, 8),
    )
    read = merged(torch.randn(2, 3, 6), torch.ones(2, 3, dtype=torch.bool))
    assert read.context.shape == (2, 3, 6)


def test_fit_adapter_bank_concat_uses_active_shared_reader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    host = nn.Module()
    host.adapter = nn.Module()
    host.adapter.recall = ARTILatentRecallField(
        6,
        8,
        routing="grouped",
        key_dim=4,
        group_size=2,
        group_topk=1,
        formula=_FourFactorFormula(),
    )
    field = host.adapter.recall
    prefix = "adapter.recall"
    first = {
        f"{prefix}.bank": field.bank.detach().clone(),
        f"{prefix}.key_bank": field.key_bank.detach().clone(),
        f"{prefix}.group_bank": field.group_bank.detach().clone(),
        f"{prefix}.query.weight": field.query.weight.detach().clone(),
    }
    second = {name: value.clone() for name, value in first.items()}
    for name in (f"{prefix}.bank", f"{prefix}.key_bank", f"{prefix}.group_bank"):
        second[name].add_(10.0)
    paths = (tmp_path / "first.arti.st", tmp_path / "second.arti.st")
    payloads = {
        paths[0].resolve(): {"adapter_state_dict": first},
        paths[1].resolve(): {"adapter_state_dict": second},
    }
    fit_project = importlib.import_module("arti.fit.project")
    monkeypatch.setattr(
        fit_project,
        "validate_artifact",
        lambda path, map_location: payloads[Path(path).resolve()],
    )

    summary = concatenate_adapter_banks(
        host,
        paths,
        bank_names=("alpha", "beta"),
        weights={"alpha": 1.0, "beta": 0.0},
    )

    assert summary["source_count"] == 2
    assert summary["field_count"] == 1
    assert summary["bank_tensor_count"] == 3
    assert summary["bank_names"] == ("alpha", "beta")
    assert summary["weights"] == {"alpha": 1.0, "beta": 0.0}
    assert summary["influences"] == {"alpha": 1.0, "beta": 1.0}
    assert field.slots == 16
    assert field.factor_slices == ((0, 4), (4, 8), (8, 12), (12, 16))
    expected = torch.cat(
        [
            torch.cat((first[f"{prefix}.bank"][start:stop], second[f"{prefix}.bank"][start:stop]))
            for start, stop in ((0, 2), (2, 4), (4, 6), (6, 8))
        ]
    )
    assert torch.equal(field.bank, expected)
    assert torch.equal(field.query.weight, first[f"{prefix}.query.weight"])
    probe = torch.randn(2, 3, 6)
    mask = torch.ones(2, 3, dtype=torch.bool)
    alpha_read = field(probe, mask)
    alpha_route = alpha_read.route.reshape(2, 3, 4, 2)
    assert torch.equal(alpha_route[..., 0], torch.ones_like(alpha_route[..., 0]))
    assert torch.equal(alpha_route[..., 1], torch.zeros_like(alpha_route[..., 1]))

    updated = set_adapter_bank_weights(host, {"alpha": 0.0, "beta": 2.0})
    assert updated == {
        "bank_names": ("alpha", "beta"),
        "weights": {"alpha": 0.0, "beta": 2.0},
        "field_count": 1,
    }
    beta_route = field(probe, mask).route.reshape(2, 3, 4, 2)
    assert torch.equal(beta_route[..., 0], torch.zeros_like(beta_route[..., 0]))
    assert torch.equal(beta_route[..., 1], torch.ones_like(beta_route[..., 1]))
    assert "_route_log_prior" not in field.state_dict()
    assert "_route_influence" not in field.state_dict()
    with pytest.raises(ValueError, match="at least 1"):
        set_adapter_bank_weights(host, (0.0, 0.0))


def test_dense_recall_expert_weights_bias_routing_without_scaling_values() -> None:
    field = ARTILatentRecallField(2, 2, routing="dense")
    with torch.no_grad():
        field.bank.copy_(torch.eye(2))
        field.query.weight.zero_()
    field.configure_expert_routes(
        ("first", "second"),
        ((0, 1), (1, 2)),
    )
    probe = torch.zeros(1, 1, 2)
    mask = torch.ones(1, 1, dtype=torch.bool)

    field.set_expert_weights({"first": 1.0, "second": 0.0})
    first = field(probe, mask)
    assert torch.equal(first.context, torch.tensor([[[1.0, 0.0]]]))
    field.set_expert_weights({"first": 0.0, "second": 3.0})
    second = field(probe, mask)
    assert torch.equal(second.context, torch.tensor([[[0.0, 1.0]]]))


def test_signed_expert_influence_reverses_write_without_changing_route() -> None:
    field = ARTILatentRecallField(2, 2, routing="dense")
    with torch.no_grad():
        field.bank.copy_(torch.eye(2))
        field.query.weight.zero_()
    field.configure_expert_routes(
        ("first", "second"),
        ((0, 1), (1, 2)),
    )
    field.set_expert_weights({"first": 1.0, "second": 0.0})
    probe = torch.zeros(1, 1, 2)
    mask = torch.ones(1, 1, dtype=torch.bool)

    positive = field(probe, mask)
    updated = set_adapter_bank_influences(
        nn.Sequential(field),
        {"first": -1.0, "second": 1.0},
    )
    negative = field(probe, mask)

    assert updated == {
        "bank_names": ("first", "second"),
        "influences": {"first": -1.0, "second": 1.0},
        "field_count": 1,
    }
    assert torch.equal(negative.route, positive.route)
    assert torch.equal(negative.context, -positive.context)
    assert torch.equal(negative.influence, -positive.influence)
    assert field.expert_weights == (1.0, 0.0)
    assert field.expert_influences == (-1.0, 1.0)
    assert "_route_influence" not in field.state_dict()


def test_signed_expert_influence_uses_route_mass_and_accepts_zero() -> None:
    field = ARTILatentRecallField(2, 2, routing="dense")
    with torch.no_grad():
        field.bank.copy_(torch.eye(2))
        field.query.weight.zero_()
    field.configure_expert_routes(("first", "second"), ((0, 1), (1, 2)))
    probe = torch.zeros(1, 1, 2)
    mask = torch.ones(1, 1, dtype=torch.bool)

    field.set_expert_influences((-1.0, 1.0))
    cancelled = field(probe, mask)
    assert torch.equal(cancelled.context, torch.zeros_like(cancelled.context))
    assert torch.equal(cancelled.route, torch.full_like(cancelled.route, 0.5))

    field.set_expert_influences((0.0, 0.0))
    disabled = field(probe, mask)
    assert torch.equal(disabled.context, torch.zeros_like(disabled.context))
    with pytest.raises(ValueError, match="finite"):
        field.set_expert_influences((float("nan"), 1.0))


def test_asset_rejects_different_shared_contract(tmp_path: Path) -> None:
    host = _Host()
    template = _template()
    contract = create_recall_expert_contract(host, template, preset_id="toy.fixed.v1")
    path = _export(tmp_path, host, template, contract, "one", 1.0)
    other = create_recall_expert_contract(host, template, preset_id="toy.other.v1")
    assembly = RecallExpertAssembly(template, other)
    with pytest.raises(ValueError, match="incompatible"):
        assembly.add(path)


def test_export_rejects_forged_shared_reader_contract(tmp_path: Path) -> None:
    host = _Host()
    template = _template()
    contract = create_recall_expert_contract(host, template, preset_id="toy.fixed.v1")
    forged = copy.deepcopy(template)
    freeze_for_recall_expert(host, forged)
    with torch.no_grad():
        forged.query.weight.add_(1.0)
    with pytest.raises(ValueError, match="shared_state_sha256"):
        export_recall_expert_bank(
            forged,
            tmp_path / "forged.recall.arti.st",
            host=host,
            expert_id="forged",
            contract=contract,
        )
