from __future__ import annotations

import copy

import pytest
import torch
from torch import Tensor, nn

import arti


class TinyBank(nn.Module):
    def __init__(self, values: Tensor) -> None:
        super().__init__()
        self.bank = nn.Parameter(values.clone())

    def forward(self, x: Tensor) -> Tensor:
        return x + self.bank.mean(dim=0)


def _save_tiny_bank(
    tmp_path,
    name: str,
    values: Tensor,
    *,
    formula: str | None = None,
    updater: nn.Module | None = None,
    shared_config=None,
):
    torch.manual_seed(1200)
    host = nn.Linear(values.shape[-1], values.shape[-1], bias=False)
    bank = TinyBank(values)
    contract = arti.create_recall_bank_contract(
        host,
        bank,
        bank_id=name,
        formula=formula,
        updater=updater,
        shared_config=shared_config,
    )
    arti.freeze_for_recall_bank(host, bank)
    path = tmp_path / f"{name}.recall.arti.st"
    arti.save_recall_bank(
        bank,
        path,
        host=host,
        bank_id=name,
        contract=contract,
        formula=formula,
        updater=updater,
        shared_config=shared_config,
    )
    return host, bank, contract, path


def test_provenance_records_reader_formula_updater_and_layout() -> None:
    host = nn.Linear(4, 4, bias=False)
    recall = arti.Recall(4, slots=3, formula="arti/delta@1")
    updater = arti.alpha.RecallValueUpdater(
        hidden_dim=4,
        slots=3,
        workspace_dim=8,
        depth=1,
    )
    contract = arti.create_recall_bank_contract(
        host,
        recall,
        bank_id="portrait",
        updater=updater,
    )

    provenance = contract.provenance
    reader_refs = {item["ref"] for item in provenance.reader["component_provenance"]["components"]}
    updater_refs = {
        item["ref"] for item in provenance.updater["component_provenance"]["components"]
    }
    assert "arti/recall@4" in reader_refs
    assert provenance.formula["reference"] == "arti/delta@1"
    assert "arti/updater@1" in updater_refs
    assert provenance.bank_layout["dimensions"]["slots"] == 3
    assert provenance.bank_layout["dimensions"]["hidden_dim"] == 4
    assert provenance.schema_fingerprint
    assert provenance.fingerprint


def test_provenance_round_trip_and_payload_copy(tmp_path) -> None:
    updater = nn.Linear(4, 4, bias=False)
    _host, _bank, contract, path = _save_tiny_bank(
        tmp_path,
        "formula-bank",
        torch.ones(3, 4),
        formula="acme/formula@1",
        updater=updater,
    )
    asset = arti.inspect_recall_bank(path)
    assert asset.contract.provenance.to_dict() == contract.provenance.to_dict()
    loaded = arti.load(path, load_resources=False, load_checkpoint=False)
    payload = loaded.manifest["architecture"]["config"]["recall_bank"]
    assert payload["provenance"] == contract.provenance.to_dict()

    fresh = TinyBank(torch.zeros(3, 4))
    restored = arti.load_recall_bank(
        path,
        fresh,
        contract=contract,
        formula="acme/formula@1",
        updater=updater,
    )
    assert restored.bank_id == "formula-bank"
    assert torch.equal(fresh.bank, torch.ones(3, 4))


def test_concat_preserves_one_contract_and_explicit_roles(tmp_path) -> None:
    updater = nn.Linear(4, 4, bias=False)
    _, bank_a, contract_a, path_a = _save_tiny_bank(
        tmp_path,
        "first",
        torch.eye(3, 4),
        formula="acme/formula@1",
        updater=updater,
    )
    _, bank_b, contract_b, path_b = _save_tiny_bank(
        tmp_path,
        "second",
        torch.full((3, 4), 2.0),
        formula="acme/formula@1",
        updater=updater,
    )
    assert contract_a.fingerprint == contract_b.fingerprint
    assembly = arti.RecallBankAssembly(
        TinyBank(torch.zeros(3, 4)),
        contract_a,
        formula="acme/formula@1",
        updater=updater,
    )
    assembly.add(path_a)
    assembly.add(path_b)
    merged, layout = assembly.materialize()
    assert tuple(merged.bank.shape) == (6, 4)
    assert layout.bank_ids == ("first", "second")
    assert torch.equal(merged.bank[:3], bank_a.bank)
    assert torch.equal(merged.bank[3:], bank_b.bank)


def test_fresh_load_rejects_shape_formula_and_updater_drift(tmp_path) -> None:
    updater = nn.Linear(4, 4, bias=False)
    _host, _bank, contract, path = _save_tiny_bank(
        tmp_path,
        "drift",
        torch.ones(3, 4),
        formula="acme/formula@1",
        updater=updater,
    )
    with pytest.raises(ValueError, match="shape or dtype"):
        arti.load_recall_bank(
            path,
            TinyBank(torch.zeros(4, 4)),
            contract=contract,
            formula="acme/formula@1",
            updater=updater,
        )
    with pytest.raises(ValueError, match="provenance"):
        arti.load_recall_bank(
            path,
            TinyBank(torch.zeros(3, 4)),
            contract=contract,
            formula="acme/other-formula@1",
            updater=updater,
        )
    with pytest.raises(ValueError, match="provenance"):
        arti.load_recall_bank(
            path,
            TinyBank(torch.zeros(3, 4)),
            contract=contract,
            formula="acme/formula@1",
            updater=nn.Linear(5, 5, bias=False),
        )


def test_tampered_provenance_is_rejected() -> None:
    host = nn.Linear(2, 2, bias=False)
    bank = TinyBank(torch.ones(2, 2))
    contract = arti.create_recall_bank_contract(host, bank, bank_id="tamper")
    payload = contract.to_dict()
    tampered = copy.deepcopy(payload)
    tampered["provenance"]["schema_fingerprint"] = "0" * 64
    with pytest.raises(ValueError, match="schema fingerprint"):
        arti.RecallBankContract.from_dict(tampered)


def test_migration_is_explicit_and_records_source(tmp_path) -> None:
    _source_host, _source_bank, source_contract, source_path = _save_tiny_bank(
        tmp_path,
        "source",
        torch.ones(2, 3),
        shared_config={"revision": "v1"},
    )
    torch.manual_seed(9900)
    target_host = nn.Linear(3, 3, bias=False)
    target_bank = TinyBank(torch.zeros(2, 3))
    target_contract = arti.create_recall_bank_contract(
        target_host,
        target_bank,
        bank_id="target",
        shared_config={"revision": "v2"},
    )
    target_path = tmp_path / "target.recall.arti.st"
    result = arti.migrate_recall_bank(
        source_path,
        target_path,
        target_expert=target_bank,
        target_host=target_host,
        target_contract=target_contract,
        state_transform=lambda state: {name: value + 1.0 for name, value in state.items()},
        shared_config={"revision": "v2"},
    )
    assert result.weights_path == target_path
    asset = arti.inspect_recall_bank(target_path)
    assert asset.training_metadata["migration"]["explicit"] is True
    assert (
        asset.training_metadata["migration"]["source_contract_fingerprint"]
        == source_contract.fingerprint
    )
    assert torch.equal(asset.state_dict["bank"], torch.full((2, 3), 2.0))

    with pytest.raises(ValueError, match="identical"):
        arti.migrate_recall_bank(
            source_path,
            tmp_path / "same.recall.arti.st",
            target_expert=TinyBank(torch.zeros(2, 3)),
            target_host=copy.deepcopy(_source_host),
            target_contract=source_contract,
        )
