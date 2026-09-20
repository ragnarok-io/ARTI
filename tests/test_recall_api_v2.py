from __future__ import annotations

from importlib.util import find_spec

import pytest
import torch
from torch import Tensor, nn

import arti
from arti.recall_registry import InvalidRecallFormulaIdError


class TinyBank(nn.Module):
    def __init__(self, values: Tensor) -> None:
        super().__init__()
        self.bank = nn.Parameter(values.clone())

    def forward(self, x: Tensor) -> Tensor:
        return x + self.bank.mean(dim=0)


class NextStateIdentity(nn.Module):
    output_semantics = "next_state"

    def forward(self, x: Tensor) -> Tensor:
        return x


class DeltaProducer(nn.Module):
    output_semantics = "delta"

    def forward(self, x: Tensor) -> Tensor:
        return torch.zeros_like(x)


def _bank_asset(tmp_path, name: str, values: Tensor):
    torch.manual_seed(1234)
    host = nn.Linear(values.shape[-1], values.shape[-1], bias=False)
    bank = TinyBank(values)
    contract = arti.create_recall_bank_contract(host, bank, bank_id=name)
    arti.freeze_for_recall_bank(host, bank)
    path = tmp_path / f"{name}.recall.arti.st"
    arti.save_recall_bank(bank, path, host=host, bank_id=name, contract=contract)
    return host, bank, contract, path


def test_public_surface_is_hard_cut_and_updater_is_explicit_alpha() -> None:
    assert arti.mechanisms.RecallValueUpdater is not None
    assert not hasattr(arti, "StatefulRecall")
    assert not hasattr(arti, "LayerRecall")
    assert not hasattr(arti, "FormulaIdentity")
    assert not hasattr(arti, "migrate_pt")


def test_retired_module_paths_are_not_available() -> None:
    assert find_spec("arti.layered_recall") is None
    assert find_spec("arti.stateful_recall") is None
    assert find_spec("arti.web") is None
    from arti.experimental.web import export

    assert callable(export)


def test_builtin_formula_ids_are_canonical() -> None:
    layer = arti.Recall(4, 2, formula="arti/delta@1")
    assert layer.formula_id.startswith("arti/delta@sha256:")
    assert {"arti/delta", "arti/affine", "arti/state"} <= {
        item.reference.split("@", maxsplit=1)[0] for item in arti.list_formulas()
    }
    with pytest.raises(InvalidRecallFormulaIdError):
        arti.Recall(4, 2, formula="delta-v1")


def test_refiner_delegates_one_explicit_policy_to_recall() -> None:
    recall = arti.Recall(4, 4, activation="none")
    refiner = arti.RecallRefiner(recall)
    value = torch.randn(2, 3, 4)
    refined, info = refiner(
        value,
        policy=arti.RefinePolicy.fixed(2, trace_level="summary"),
        return_info=True,
    )
    assert refined.shape == value.shape
    assert info["recall_step_attempted"].shape == (2, 2)
    with pytest.raises(TypeError, match="arti.nn.Recall"):
        arti.RecallRefiner(NextStateIdentity())


def test_forward_updater_is_preserved_as_alpha_api() -> None:
    updater = arti.mechanisms.RecallValueUpdater(
        hidden_dim=4,
        slots=3,
        workspace_dim=8,
        depth=1,
    )
    trace = torch.randn(2, 5, 4)
    previous = torch.zeros(2, 3, 4)
    next_value = updater(trace, previous, mask=torch.ones(2, 5, dtype=torch.bool))
    assert next_value.shape == previous.shape
    assert torch.isfinite(next_value).all()


def test_single_bank_round_trip(tmp_path) -> None:
    _host, bank, contract, path = _bank_asset(
        tmp_path,
        "alpha",
        torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
    asset = arti.inspect_recall_bank(path)
    assert asset.bank_id == "alpha"
    assert asset.artifact_version == arti.RECALL_BANK_ARTIFACT_VERSION
    fresh = TinyBank(torch.zeros(2, 2))
    loaded = arti.load_recall_bank(path, fresh, contract=contract)
    assert loaded.bank_id == "alpha"
    assert torch.equal(fresh.bank, bank.bank)


def test_two_bank_composition_uses_shared_contract_fingerprint(tmp_path) -> None:
    _, bank_a, contract_a, path_a = _bank_asset(
        tmp_path,
        "first",
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
    )
    _, bank_b, contract_b, path_b = _bank_asset(
        tmp_path,
        "second",
        torch.tensor([[2.0, 0.0], [0.0, 2.0]]),
    )
    assert contract_a.fingerprint == contract_b.fingerprint
    assembly = arti.RecallBankAssembly(TinyBank(torch.zeros(2, 2)), contract_a)
    assembly.add(path_a)
    assembly.add(path_b)
    merged, layout = assembly.materialize()
    assert tuple(merged.bank.shape) == (4, 2)
    assert layout.bank_ids == ("first", "second")
    assert torch.equal(merged.bank[:2], bank_a.bank)
    assert torch.equal(merged.bank[2:], bank_b.bank)


def test_wrong_bank_kind_and_version_are_rejected(tmp_path) -> None:
    wrong_kind = tmp_path / "wrong-kind.recall.arti.st"
    module = nn.Linear(2, 2)
    arti.save(
        module,
        wrong_kind,
        config={
            "artifact_kind": "other",
            "artifact_version": arti.RECALL_BANK_ARTIFACT_VERSION,
        },
    )
    with pytest.raises(arti.RecallBankError) as kind_error:
        arti.inspect_recall_bank(wrong_kind)
    assert kind_error.value.code == "wrong_kind"

    wrong_version = tmp_path / "wrong-version.recall.arti.st"
    arti.save(
        module,
        wrong_version,
        config={
            "artifact_kind": arti.RECALL_BANK_ARTIFACT_KIND,
            "artifact_version": 2,
        },
    )
    with pytest.raises(arti.RecallBankError) as version_error:
        arti.inspect_recall_bank(wrong_version)
    assert version_error.value.code == "unsupported_version"
