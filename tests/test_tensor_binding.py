from __future__ import annotations

import pytest
import torch

import arti
from arti import alpha
from arti.tensor_transaction import TensorOwnershipError, TensorTransactionContractError


HASH = "0" * 64
CONFIG = "1" * 64
STATE = "2" * 64
ABI = "3" * 64


def runtime(initial: dict[str, torch.Tensor]) -> alpha.VolatileTensorRuntime:
    return alpha.VolatileTensorRuntime(
        initial,
        world_id="binding-world",
        store_instance_id="binding-store",
        abi_fingerprint=ABI,
        provenance_fingerprint=HASH,
    )


def bind(
    store: alpha.VolatileTensorRuntime,
    snapshot: alpha.TensorSnapshot,
    key: str,
    *,
    component_ref: str,
    schema_ref: str,
    role: str,
    authority: alpha.TensorAuthority = alpha.TensorAuthority.READ_WRITE,
) -> alpha.BoundTensorRead:
    return alpha.bind_external_tensor(
        store,
        snapshot,
        key,
        address_namespace="session",
        partition_id="main",
        logical_id=key,
        role=role,
        authority=authority,
        component_ref=component_ref,
        component_config_fingerprint=CONFIG,
        state_schema_ref=schema_ref,
        producer_state_fingerprint=STATE,
        provenance_fingerprint=HASH,
    )


def proposal(
    bound: alpha.BoundTensorRead,
    value: torch.Tensor,
    *,
    component_ref: str,
    semantics: alpha.ProposalSemantics = alpha.ProposalSemantics.COMPLETE_NEXT_STATE,
) -> alpha.ExternalTensorProposal:
    return alpha.ExternalTensorProposal(
        bound.binding,
        value,
        producer_ref=component_ref,
        producer_config_fingerprint=CONFIG,
        producer_state_fingerprint=STATE,
        semantics=semantics,
    )


def test_formula_arena_publishes_once_while_ssa_versions_remain_local() -> None:
    value = torch.tensor([[[2.0], [0.0]]])
    mask = torch.tensor([[True, False]])
    ssa_version = torch.zeros(1, 2, dtype=torch.int64)
    store = runtime({"arena-value": value, "arena-mask": mask, "arena-ssa": ssa_version})
    snapshot = store.snapshot()
    component_ref = "arti/formula-fabric@1"
    value_b = bind(
        store,
        snapshot,
        "arena-value",
        component_ref=component_ref,
        schema_ref="arti/formula-arena-value@1",
        role="formula-value",
    )
    mask_b = bind(
        store,
        snapshot,
        "arena-mask",
        component_ref=component_ref,
        schema_ref="arti/formula-arena-mask@1",
        role="formula-mask",
    )
    ssa_b = bind(
        store,
        snapshot,
        "arena-ssa",
        component_ref=component_ref,
        schema_ref="arti/formula-arena-ssa@1",
        role="formula-ssa",
    )
    program = alpha.FormulaFabricProgram(
        arena_capacity=2,
        feature_dim=1,
        steps=((alpha.FormulaInvocation(alpha.FormulaPrimitive.IDENTITY, 1),),),
    )
    fabric = alpha.FormulaFabric(program)
    state = alpha.FormulaArenaState(
        value_b.read.value,
        mask_b.read.value,
        ssa_b.read.value,
    )
    weights = torch.zeros(1, 1, 1, 1, 2)
    weights[..., 0] = 1
    enabled = torch.ones(1, 1, 1, dtype=torch.bool)
    result = fabric(state, alpha.FormulaRoutePlan(weights, enabled, enabled, enabled))
    assert result.state.version[0, 1].item() == 1

    tx = store.begin(snapshot, transaction_id="formula", branch_id="main")
    for bound, candidate in (
        (value_b, result.state.value),
        (mask_b, result.state.mask),
        (ssa_b, result.state.version),
    ):
        alpha.stage_external_proposal(
            tx,
            proposal(bound, candidate, component_ref=component_ref),
        )
    receipt = tx.commit(idempotency_key="formula-commit")
    assert receipt.new_epoch == snapshot.epoch + 1
    after = store.snapshot()
    assert all(ref.version == 2 for ref in after.page_refs)
    assert store.read(after, "arena-ssa").value[0, 1].item() == 1


def test_target_bank_updater_returns_candidate_and_host_alone_commits() -> None:
    store = runtime({"target": torch.zeros(3, 2)})
    snapshot = store.snapshot()
    component_ref = "arti/target-bank-updater@2"
    bound = bind(
        store,
        snapshot,
        "target",
        component_ref=component_ref,
        schema_ref="arti/target-bank-state@1",
        role="target-bank",
    )
    updater = alpha.TargetBankUpdater(
        hidden_dim=2,
        slots=3,
        target_coupling="required_after_bootstrap",
        policy=alpha.WriteRefinePolicy.fixed(3),
    )
    candidate = updater(torch.ones(2, 2), bound.read.value)
    assert store.snapshot().epoch == snapshot.epoch
    with pytest.raises(TensorOwnershipError, match="must not require gradients"):
        proposal(bound, candidate, component_ref=component_ref)

    tx = store.begin(snapshot, transaction_id="updater", branch_id="main")
    alpha.stage_external_proposal(
        tx,
        proposal(bound, candidate.detach(), component_ref=component_ref),
    )
    tx.commit(idempotency_key="updater-commit")
    assert store.read(store.snapshot(), "target").ref.version == 2


def test_fold_address_binding_keeps_logical_ids_separate_from_permutation() -> None:
    source = torch.arange(6, dtype=torch.float32).reshape(1, 3, 2)
    store = runtime({"source": source})
    snapshot = store.snapshot()
    bound = bind(
        store,
        snapshot,
        "source",
        component_ref="arti/fold@2",
        schema_ref="arti/fold-source@1",
        role="fold-source",
    )
    record = alpha.FoldRecord(
        permutation=torch.tensor([[2, 0, 1]]),
        original_mask=torch.ones(1, 3, dtype=torch.bool),
        original_shape=source.shape,
        axis=-2,
        active_count=1,
        topology_config_fingerprint=CONFIG,
    )
    binding = alpha.FoldAddressBinding.from_record(
        bound.binding,
        record,
        logical_ids=("item-0", "item-1", "item-2"),
        write_logical_ids=("item-0", "item-1", "item-2"),
    )
    assert binding.logical_ids == ("item-0", "item-1", "item-2")
    assert record.permutation.tolist() == [[2, 0, 1]]
    assert binding.transported_logical_ids == (("item-2", "item-0", "item-1"),)
    binding.validate_record(record)

    other = alpha.FoldRecord(
        permutation=torch.tensor([[1, 2, 0]]),
        original_mask=torch.ones(1, 3, dtype=torch.bool),
        original_shape=source.shape,
        axis=-2,
        active_count=1,
        topology_config_fingerprint=CONFIG,
    )
    with pytest.raises(TensorTransactionContractError, match="does not match"):
        binding.validate_record(other)


def test_delta_stale_binding_and_foreign_component_fail_closed() -> None:
    store = runtime({"target": torch.zeros(1, 2)})
    snapshot = store.snapshot()
    bound = bind(
        store,
        snapshot,
        "target",
        component_ref="arti/target-bank-updater@2",
        schema_ref="arti/target-bank-state@1",
        role="target-bank",
    )
    delta = proposal(
        bound,
        torch.ones(1, 2),
        component_ref="arti/target-bank-updater@2",
        semantics=alpha.ProposalSemantics.DELTA,
    )
    tx = store.begin(snapshot, transaction_id="delta", branch_id="main")
    with pytest.raises(TensorTransactionContractError, match="complete_next_state"):
        alpha.stage_external_proposal(tx, delta)

    with pytest.raises(TensorTransactionContractError, match="bound component"):
        proposal(bound, torch.ones(1, 2), component_ref="arti/formula-fabric@1")
    with pytest.raises(TensorTransactionContractError, match="state fingerprint"):
        alpha.ExternalTensorProposal(
            bound.binding,
            torch.ones(1, 2),
            producer_ref="arti/target-bank-updater@2",
            producer_config_fingerprint=CONFIG,
            producer_state_fingerprint="9" * 64,
        )

    first = store.begin(snapshot, transaction_id="first", branch_id="main")
    alpha.stage_external_proposal(
        first,
        proposal(bound, torch.ones(1, 2), component_ref="arti/target-bank-updater@2"),
    )
    first.commit(idempotency_key="first")
    current = store.snapshot()
    stale_tx = store.begin(current, transaction_id="stale", branch_id="main")
    with pytest.raises(TensorTransactionContractError, match="transaction snapshot"):
        alpha.stage_external_proposal(
            stale_tx,
            proposal(bound, torch.ones(1, 2), component_ref="arti/target-bank-updater@2"),
        )
    assert store.snapshot().root_id == current.root_id


def test_proposal_owns_value_and_is_immutable() -> None:
    store = runtime({"target": torch.zeros(1, 2)})
    bound = bind(
        store,
        store.snapshot(),
        "target",
        component_ref="arti/target-bank-updater@2",
        schema_ref="arti/target-bank-state@1",
        role="target-bank",
    )
    candidate = torch.ones(1, 2)
    item = proposal(bound, candidate, component_ref="arti/target-bank-updater@2")
    candidate.zero_()
    torch.testing.assert_close(item.value, torch.ones(1, 2))
    with pytest.raises(AttributeError):
        item.binding = bound.binding

    tx = store.begin(store.snapshot(), transaction_id="mutated", branch_id="main")
    item._value.add_(1)
    with pytest.raises(TensorTransactionContractError, match="candidate was mutated"):
        alpha.stage_external_proposal(tx, item)


def test_external_proposal_rejects_nonfinite_candidate() -> None:
    store = runtime({"target": torch.zeros(1, 2)})
    snapshot = store.snapshot()
    bound = bind(
        store,
        snapshot,
        "target",
        component_ref="arti/target-bank-updater@2",
        schema_ref="arti/target-bank-value@1",
        role="target-bank",
    )
    with pytest.raises(TensorOwnershipError, match="finite"):
        proposal(
            bound,
            torch.tensor([[float("nan"), 1.0]]),
            component_ref="arti/target-bank-updater@2",
        )


def test_read_only_binding_cannot_stage_a_proposal() -> None:
    store = runtime({"target": torch.zeros(1, 2)})
    snapshot = store.snapshot()
    bound = bind(
        store,
        snapshot,
        "target",
        component_ref="arti/target-bank-updater@2",
        schema_ref="arti/target-bank-state@1",
        role="target-bank",
        authority=alpha.TensorAuthority.READ_ONLY,
    )
    tx = store.begin(snapshot, transaction_id="read-only", branch_id="main")
    with pytest.raises(TensorTransactionContractError, match="write authority"):
        alpha.stage_external_proposal(
            tx,
            proposal(bound, torch.ones(1, 2), component_ref="arti/target-bank-updater@2"),
        )


def test_binding_surface_is_alpha_only() -> None:
    assert not hasattr(arti, "ExternalTensorBinding")
    assert not hasattr(arti.nn, "ExternalTensorBinding")
    assert alpha.ExternalTensorBinding._runtime_contract_ref == "arti/external-binding@1"
    assert alpha.FoldAddressBinding._runtime_contract_ref == "arti/fold-address-binding@1"
