from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import arti
from arti.layers import ARTILatentRecallField
from arti.recall_ttt import RecallArtifactSpec, RecallCapacityPlan, RecallExpertPool, RecallExpertRegistry, RecallTTTSession, export_recall_artifact, load_recall_artifact, module_structure_fingerprint


class TinyRecallHost(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Linear(3, 3, bias=False)
        self.expert = nn.Linear(3, 3, bias=False)
        with torch.no_grad():
            self.backbone.weight.copy_(torch.eye(3))
            self.expert.weight.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x) + self.expert(x)


class TinyFieldHost(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.recall = ARTILatentRecallField(4, 2, recognition_mode="none")


def consistency_loss(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    return F.mse_loss(model(batch["support"]), batch["reference"])


def spec(host: TinyRecallHost) -> RecallArtifactSpec:
    return RecallArtifactSpec(
        capability="toy-correction",
        base_model_fingerprint=module_structure_fingerprint(host),
        injection_fingerprint=module_structure_fingerprint(host.expert),
        training_metadata={"objective": "consistency"},
    )


def test_recall_ttt_updates_only_expert_and_rolls_back() -> None:
    host = TinyRecallHost()
    initial = host.backbone.weight.detach().clone()
    session = RecallTTTSession(host, host.expert, consistency_loss)
    record = session.adapt(
        [{"support": torch.ones(4, 3), "reference": torch.zeros(4, 3)}],
        steps=3,
        learning_rate=0.1,
    )
    assert record.initial_weights_sha256 != record.final_weights_sha256
    assert record.backbone_sha256_before == record.backbone_sha256_after
    assert torch.equal(host.backbone.weight, initial)
    assert not torch.equal(host.expert.weight, torch.zeros_like(host.expert.weight))
    rolled = session.rollback()
    assert rolled is not None and rolled.rolled_back
    assert torch.equal(host.expert.weight, torch.zeros_like(host.expert.weight))


def test_recall_ttt_rejects_label_bearing_support() -> None:
    host = TinyRecallHost()
    session = RecallTTTSession(host, host.expert, consistency_loss)
    with pytest.raises(ValueError, match="label-bearing"):
        session.adapt(
            [{"support": torch.ones(2, 3), "labels": torch.zeros(2, dtype=torch.long)}],
            steps=1,
            learning_rate=0.1,
        )


def test_recall_ttt_can_train_bank_only_for_concat_safe_artifacts() -> None:
    host = TinyFieldHost()
    query_before = host.recall.query.weight.detach().clone()
    bank_before = host.recall.bank.detach().clone()

    def bank_loss(model: nn.Module, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        return model.recall.bank.square().mean()

    session = RecallTTTSession(
        host,
        host.recall,
        bank_loss,
        trainable_parameters=("bank",),
    )
    record = session.adapt([{"support": torch.ones(1, 4)}], steps=2, learning_rate=0.05)
    assert record.trainable_parameters == ("bank",)
    assert not torch.equal(host.recall.bank, bank_before)
    assert torch.equal(host.recall.query.weight, query_before)
    assert host.recall.bank.requires_grad
    assert not host.recall.query.weight.requires_grad


def test_recall_artifact_requires_name_and_host_compatibility(tmp_path: Path) -> None:
    host = TinyRecallHost()
    with pytest.raises(ValueError, match=".recall.arti.st"):
        export_recall_artifact(host.expert, tmp_path / "expert.arti.st", spec(host))

    path = tmp_path / "coder.recall.arti.st"
    export_recall_artifact(host.expert, path, spec(host))
    with torch.no_grad():
        host.expert.weight.fill_(2.0)
    load_recall_artifact(path, host.expert, base_model=host)
    assert torch.equal(host.expert.weight, torch.zeros_like(host.expert.weight))

    wrong_host = TinyRecallHost()
    wrong_host.backbone = nn.Linear(3, 4, bias=False)
    with pytest.raises(ValueError, match="base model fingerprint"):
        load_recall_artifact(path, TinyRecallHost().expert, base_model=wrong_host)


def test_registry_activation_failure_restores_previous_expert(tmp_path: Path) -> None:
    host = TinyRecallHost()
    path = tmp_path / "profile.recall.arti.st"
    export_recall_artifact(host.expert, path, spec(host))
    registry = RecallExpertRegistry(host.expert, base_model=host)
    with torch.no_grad():
        host.expert.weight.fill_(1.0)
    before = host.expert.weight.detach().clone()
    with pytest.raises(FileNotFoundError):
        registry.activate(tmp_path / "wrong.recall.arti.st")
    assert torch.equal(host.expert.weight, before)


def test_recall_ttt_is_exposed_through_root_and_torch_namespaces() -> None:
    assert arti.RecallTTTSession is RecallTTTSession
    assert arti.torch.RecallTTTSession is RecallTTTSession
    assert arti.torch.export_recall_artifact is export_recall_artifact
    assert arti.RecallCapacityPlan is RecallCapacityPlan
    assert arti.torch.RecallCapacityPlan is RecallCapacityPlan
    assert arti.RecallExpertPool is RecallExpertPool
    assert arti.torch.RecallExpertPool is RecallExpertPool


def test_capacity_plan_routes_and_protects_overflow() -> None:
    plan = RecallCapacityPlan(slots_per_expert=4, experts=2, overflow_policy="shard")
    decision = plan.decide(11)
    assert decision.overflowed is True
    assert decision.accepted_items == 8
    assert decision.dropped_items == 3
    assert decision.expert_item_counts == (4, 4)
    assert decision.protected is True
    abstain = RecallCapacityPlan(slots_per_expert=4, experts=2, overflow_policy="abstain").decide(11)
    assert abstain.accepted_items == 0
    assert abstain.dropped_items == 11


def test_artifact_roundtrip_preserves_capacity_plan(tmp_path: Path) -> None:
    host = TinyRecallHost()
    path = tmp_path / "expert.recall.arti.st"
    configured = RecallArtifactSpec(
        capability="toy-correction",
        base_model_fingerprint=module_structure_fingerprint(host),
        injection_fingerprint=module_structure_fingerprint(host.expert),
        capacity_plan=RecallCapacityPlan(slots_per_expert=3, experts=2, overflow_policy="shard"),
    )
    export_recall_artifact(host.expert, path, configured)
    loaded = load_recall_artifact(path, host.expert, base_model=host)
    assert loaded.manifest["architecture"]["config"]["recall_expert"]["capacity_plan"] == {
        "slots_per_expert": 3,
        "experts": 2,
        "overflow_policy": "shard",
    }


def test_recall_expert_pool_keeps_routes_resident_and_supports_mixtures(tmp_path: Path) -> None:
    host = TinyRecallHost()
    first = tmp_path / "first.recall.arti.st"
    second = tmp_path / "second.recall.arti.st"
    with torch.no_grad():
        host.expert.weight.copy_(torch.eye(3))
    export_recall_artifact(host.expert, first, spec(host))
    with torch.no_grad():
        host.expert.weight.copy_(3.0 * torch.eye(3))
    export_recall_artifact(host.expert, second, spec(host))

    pool = RecallExpertPool(host.expert, base_model=host)
    pool.load_expert("first", first)
    pool.load_expert("second", second)
    x = torch.ones(2, 3)

    assert pool.loaded_experts == ("first", "second")
    assert torch.allclose(pool(x, expert="first"), torch.ones_like(x))
    assert torch.allclose(pool(x, expert="second"), torch.full_like(x, 3.0))
    with pytest.raises(ValueError, match="pool.concatenate"):
        pool(x)
    weights = torch.tensor([[1.0, 0.0], [0.25, 0.75]])
    mixed = pool(x, mixture_weights=weights)
    assert torch.allclose(mixed[0], torch.ones(3))
    assert torch.allclose(mixed[1], torch.full((3,), 2.5))
    assert "experts.first.weight" in pool.state_dict()
    assert "experts.second.weight" in pool.state_dict()


def test_recall_expert_pool_rejects_invalid_mixture_and_duplicate_name(tmp_path: Path) -> None:
    host = TinyRecallHost()
    path = tmp_path / "one.recall.arti.st"
    export_recall_artifact(host.expert, path, spec(host))
    pool = RecallExpertPool(host.expert, base_model=host)
    pool.load_expert("one", path)
    with pytest.raises(ValueError, match="already loaded"):
        pool.load_expert("one", path)
    with pytest.raises(ValueError, match="positive mass"):
        pool(torch.ones(1, 3), mixture_weights=torch.zeros(1))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_recall_expert_pool_can_keep_multiple_artifacts_on_cuda(tmp_path: Path) -> None:
    host = TinyRecallHost().cuda()
    first = tmp_path / "cuda-first.recall.arti.st"
    second = tmp_path / "cuda-second.recall.arti.st"
    export_recall_artifact(host.expert, first, spec(host))
    with torch.no_grad():
        host.expert.weight.copy_(torch.eye(3, device="cuda"))
    export_recall_artifact(host.expert, second, spec(host))
    pool = RecallExpertPool(host.expert, base_model=host)
    pool.load_expert("first", first, map_location="cuda")
    pool.load_expert("second", second, map_location="cuda")
    output = pool(torch.ones(2, 3, device="cuda"), mixture_weights=torch.tensor([0.4, 0.6], device="cuda"))
    assert output.is_cuda
    assert all(next(expert.parameters()).is_cuda for expert in pool.experts.values())


def test_recall_expert_pool_concatenates_native_banks_without_output_remix(tmp_path: Path) -> None:
    torch.manual_seed(19)
    host = TinyFieldHost()
    first = tmp_path / "field-first.recall.arti.st"
    second = tmp_path / "field-second.recall.arti.st"
    artifact_spec = RecallArtifactSpec(
        capability="field-concat",
        base_model_fingerprint=module_structure_fingerprint(host),
        injection_fingerprint=module_structure_fingerprint(host.recall),
    )
    first_bank = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]])
    second_bank = torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    with torch.no_grad():
        host.recall.bank.copy_(first_bank)
    export_recall_artifact(host.recall, first, artifact_spec)
    with torch.no_grad():
        host.recall.bank.copy_(second_bank)
    export_recall_artifact(host.recall, second, artifact_spec)

    pool = RecallExpertPool(host.recall, base_model=host)
    pool.load_expert("first", first)
    pool.load_expert("second", second)
    merged = pool.concatenate()

    assert isinstance(merged, ARTILatentRecallField)
    assert merged.bank.shape == (4, 4)
    assert torch.equal(merged.bank, torch.cat([first_bank, second_bank], dim=0))
    context, weights, _, _ = merged(torch.ones(1, 1, 4), torch.ones(1, 1, dtype=torch.bool))
    assert context.shape == (1, 1, 4)
    assert weights.shape == (1, 1, 4)


def test_recall_concat_rejects_divergent_shared_operators(tmp_path: Path) -> None:
    host = TinyFieldHost()
    first = tmp_path / "shared-first.recall.arti.st"
    second = tmp_path / "shared-second.recall.arti.st"
    artifact_spec = RecallArtifactSpec(
        capability="field-concat",
        base_model_fingerprint=module_structure_fingerprint(host),
        injection_fingerprint=module_structure_fingerprint(host.recall),
    )
    export_recall_artifact(host.recall, first, artifact_spec)
    with torch.no_grad():
        host.recall.query.weight.add_(1.0)
    export_recall_artifact(host.recall, second, artifact_spec)
    pool = RecallExpertPool(host.recall, base_model=host)
    pool.load_expert("first", first)
    pool.load_expert("second", second)
    with pytest.raises(ValueError, match="shared state"):
        pool.concatenate()
