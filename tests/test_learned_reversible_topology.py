from __future__ import annotations

import pytest
import torch

import arti
from arti import alpha


def _markers(batch: int = 2, length: int = 8, dim: int = 4) -> torch.Tensor:
    return torch.arange(batch * length * dim, dtype=torch.float32).reshape(
        batch, length, dim
    )


def _permutation(
    policy: torch.nn.Module, x: torch.Tensor, mask: torch.Tensor, active_count: int
) -> torch.Tensor:
    return alpha.ReversibleTopology(active_count, policy=policy).fold(
        x, mask
    ).record.permutation


def test_stable_priority_partition_is_valid_first_and_stable() -> None:
    operator = alpha.StablePriorityPartition()
    action = alpha.TopologyAction(torch.tensor([[1.0, 3.0, 3.0, 100.0]]))
    mask = torch.tensor([[True, True, True, False]])

    permutation = operator(action, mask)

    assert torch.equal(permutation, torch.tensor([[1, 2, 0, 3]]))


def test_surrogate_gradient_promotes_more_useful_instances() -> None:
    scores = torch.zeros(1, 4, requires_grad=True)
    values = torch.tensor([[[4.0], [3.0], [2.0], [1.0]]])
    action = alpha.TopologyAction(scores)
    assignment = alpha.SoftTopKTopologySurrogate()(
        action, torch.ones(1, 4, dtype=torch.bool), 2
    )
    soft_active = torch.einsum("bkn,bnd->bkd", assignment, values)

    (-soft_active.mean()).backward()

    assert scores.grad is not None
    # Gradient descent must raise the relative priority of the most useful
    # instance and lower that of the least useful one.
    assert scores.grad[0, 0] < scores.grad[0, -1]


def test_surrogate_swap_direction_agrees_with_hard_top_k() -> None:
    generator = torch.Generator().manual_seed(9107)
    scores = torch.randn(2048, 16, generator=generator, requires_grad=True)
    utility = torch.randn(2048, 16, generator=generator)
    mask = torch.ones_like(scores, dtype=torch.bool)
    assignment = alpha.SoftTopKTopologySurrogate()(
        alpha.TopologyAction(scores), mask, 4
    )
    objective = torch.einsum("bkn,bn->bk", assignment, utility).sum(-1)
    gradient = torch.autograd.grad(-objective.mean(), scores)[0]

    selected = torch.topk(scores.detach(), 4, dim=-1).indices
    target = torch.topk(utility, 4, dim=-1).indices
    selected_mask = torch.zeros_like(mask).scatter_(1, selected, True)
    target_mask = torch.zeros_like(mask).scatter_(1, target, True)
    missing = target_mask & ~selected_mask
    extra = selected_mask & ~target_mask
    compared_pairs = missing.unsqueeze(-1) & extra.unsqueeze(-2)
    agrees = gradient.unsqueeze(-1) < gradient.unsqueeze(-2)

    assert float(agrees[compared_pairs].float().mean()) >= 0.70


def test_surrogate_zeroes_ranks_beyond_the_valid_instance_count() -> None:
    action = alpha.TopologyAction(torch.tensor([[4.0, 3.0, 2.0, 1.0]]))
    mask = torch.tensor([[True, True, False, False]])

    assignment = alpha.SoftTopKTopologySurrogate()(action, mask, 4)

    torch.testing.assert_close(assignment[:, :2].sum(-1), torch.ones(1, 2))
    assert torch.equal(assignment[:, 2:], torch.zeros(1, 2, 4))


def test_surrogate_masked_instances_have_zero_assignment_and_gradient() -> None:
    scores = torch.tensor([[1.0, 100.0, 2.0, -100.0]], requires_grad=True)
    mask = torch.tensor([[True, False, True, False]])
    utility = torch.tensor([[1.0, 1000.0, 2.0, -1000.0]])
    assignment = alpha.SoftTopKTopologySurrogate()(
        alpha.TopologyAction(scores), mask, 2
    )

    torch.einsum("bkn,bn->", assignment, utility).backward()

    assert torch.equal(assignment[..., ~mask[0]], torch.zeros(1, 2, 2))
    assert torch.equal(scores.grad[..., ~mask[0]], torch.zeros(1, 2))


def test_nonfinite_folded_payload_cannot_pollute_hard_active_forward() -> None:
    policy = alpha.LearnedTopologyPolicy(dim=2)
    with torch.no_grad():
        for parameter in policy.scorer.parameters():
            parameter.zero_()
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [float("nan"), float("inf")]]])
    # Keep the non-finite lineage invalid so the scorer/operator never uses it,
    # while the backward-only carrier still sees the complete detached payload.
    mask = torch.tensor([[True, True, False]])
    state = alpha.ReversibleTopology(2, policy=policy).fold(x, mask)

    assert torch.equal(state.active, x[:, :2])


def test_no_grad_skips_the_training_surrogate_without_changing_permutation() -> None:
    torch.manual_seed(5)
    policy = alpha.LearnedTopologyPolicy(dim=3)
    x = torch.randn(2, 12, 3)
    mask = torch.ones(2, 12, dtype=torch.bool)
    topology = alpha.ReversibleTopology(4, policy=policy)
    with torch.enable_grad():
        with_surrogate = topology.fold(x, mask)
    with torch.no_grad():
        without_surrogate = topology.fold(x, mask)

    assert with_surrogate.active.requires_grad
    assert not without_surrogate.active.requires_grad
    assert torch.equal(
        with_surrogate.record.permutation, without_surrogate.record.permutation
    )


def test_eval_skips_the_training_surrogate_with_grad_enabled() -> None:
    topology = alpha.ReversibleTopology(
        4, policy=alpha.LearnedTopologyPolicy(dim=3)
    ).eval()
    x = torch.randn(2, 12, 3)
    mask = torch.ones(2, 12, dtype=torch.bool)

    state = topology.fold(x, mask)

    assert not state.active.requires_grad


def test_learned_policy_preserves_exact_value_transport() -> None:
    torch.manual_seed(7)
    x = _markers()
    topology = alpha.ReversibleTopology(
        active_count=3,
        policy=alpha.LearnedTopologyPolicy(dim=4),
    )

    state = topology.fold(x)
    restored = topology.unfold(state).value

    assert torch.equal(restored, x)
    packed = torch.cat((state.active, state.folded), dim=-2)
    expected = torch.gather(
        x,
        -2,
        state.record.permutation.unsqueeze(-1).expand_as(x),
    )
    assert torch.equal(packed, expected)


def test_surrogate_trains_policy_without_soft_value_gradient_leakage() -> None:
    torch.manual_seed(11)
    x = _markers(batch=1, length=6, dim=3).requires_grad_()
    policy = alpha.LearnedTopologyPolicy(dim=3, hidden_dim=12)
    topology = alpha.ReversibleTopology(active_count=2, policy=policy)

    state = topology.fold(x)
    state.active.square().sum().backward()

    assert x.grad is not None
    selected = state.record.active_index
    expected = torch.zeros_like(x)
    expected.scatter_(
        -2,
        selected.unsqueeze(-1).expand(*selected.shape, x.shape[-1]),
        2 * torch.gather(
            x.detach(),
            -2,
            selected.unsqueeze(-1).expand(*selected.shape, x.shape[-1]),
        ),
    )
    torch.testing.assert_close(x.grad, expected, rtol=0, atol=0)
    gradients = [parameter.grad for parameter in policy.scorer.parameters()]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert sum(float(gradient.abs().sum()) for gradient in gradients if gradient is not None) > 0


def test_core_detaches_policy_input_even_for_a_nonconforming_custom_policy() -> None:
    class NonDetachingPolicy(torch.nn.Module):
        _component_reference = "example/non-detaching-policy@1"

        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))

        def forward(
            self, x: torch.Tensor, _mask: torch.Tensor
        ) -> alpha.TopologyProposal:
            return alpha.TopologyProposal(alpha.TopologyAction(x[..., 0] * self.scale))

        def topology_contract(self) -> dict[str, object]:
            return {"ref": self._component_reference}

    x = torch.randn(1, 6, 3, requires_grad=True)
    policy = NonDetachingPolicy()
    topology = alpha.ReversibleTopology(2, policy=policy)

    state = topology.fold(x)
    state.active.square().sum().backward()

    assert x.grad is not None
    selected = state.record.active_index
    expected = torch.zeros_like(x)
    selected_values = torch.gather(
        x.detach(), -2, selected.unsqueeze(-1).expand(-1, -1, x.shape[-1])
    )
    expected.scatter_(-2, selected.unsqueeze(-1).expand_as(selected_values), 2 * selected_values)
    torch.testing.assert_close(x.grad, expected, rtol=0, atol=0)
    assert policy.scale.grad is not None
    assert torch.isfinite(policy.scale.grad)


def test_learned_policy_is_permutation_equivariant_without_coordinates() -> None:
    torch.manual_seed(19)
    x = torch.randn(3, 9, 5)
    mask = torch.ones(3, 9, dtype=torch.bool)
    policy = alpha.LearnedTopologyPolicy(dim=5).eval()
    permutation = torch.tensor([5, 1, 8, 0, 3, 7, 2, 6, 4])
    inverse = torch.argsort(permutation)

    original = _permutation(policy, x, mask, 4)[..., :4]
    permuted = _permutation(policy, x[:, permutation], mask[:, permutation], 4)[..., :4]

    # Indices selected in the permuted host map back to the same original lineage.
    mapped = permutation[permuted]
    assert torch.equal(torch.sort(original, dim=-1).values, torch.sort(mapped, dim=-1).values)
    assert torch.equal(inverse[permutation], torch.arange(9))


def test_learned_policy_train_and_eval_execute_the_same_hard_topology() -> None:
    torch.manual_seed(23)
    x = torch.randn(2, 7, 4)
    policy = alpha.LearnedTopologyPolicy(dim=4)

    policy.train()
    train_permutation = _permutation(policy, x, torch.ones(2, 7, dtype=torch.bool), 3)
    policy.eval()
    eval_permutation = _permutation(policy, x, torch.ones(2, 7, dtype=torch.bool), 3)

    assert torch.equal(train_permutation, eval_permutation)


def _manual_bank(values: list[float], bank_id: str) -> alpha.TopologyOperandBank:
    bank = alpha.TopologyOperandBank(
        slots=2, key_dim=2, factor_dim=1, bank_id=bank_id
    )
    with torch.no_grad():
        bank.keys.copy_(torch.eye(2))
        bank.values.copy_(torch.tensor(values).reshape(2, 1))
    return bank


def _manual_bank_policy(
    banks: list[alpha.TopologyOperandBank],
    weights: list[float] | None = None,
    *,
    diagnostics: str = "none",
) -> alpha.BankFormulaTopologyPolicy:
    policy = alpha.BankFormulaTopologyPolicy(
        dim=2,
        key_dim=2,
        banks=banks,
        bank_weights=weights,
        diagnostics=diagnostics,
    )
    with torch.no_grad():
        policy.query.basis.copy_(torch.eye(2))
    return policy


def test_bank_switch_has_a_causal_effect_on_hard_topology() -> None:
    x = torch.tensor([[[4.0, 0.0], [0.0, 4.0], [3.0, 0.0], [0.0, 3.0]]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    bank_a = _manual_bank([2.0, -2.0], "a")
    bank_b = _manual_bank([-2.0, 2.0], "b")

    selected_a = _permutation(_manual_bank_policy([bank_a]), x, mask, 2)[..., :2]
    selected_b = _permutation(_manual_bank_policy([bank_b]), x, mask, 2)[..., :2]

    assert torch.equal(selected_a, torch.tensor([[0, 2]]))
    assert torch.equal(selected_b, torch.tensor([[1, 3]]))


def test_concat_keeps_each_bank_addressable_by_explicit_weight() -> None:
    x = torch.tensor([[[4.0, 0.0], [0.0, 4.0], [3.0, 0.0], [0.0, 3.0]]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    bank_a = _manual_bank([2.0, -2.0], "a")
    bank_b = _manual_bank([-2.0, 2.0], "b")
    combined_a = _manual_bank_policy(
        [bank_a, bank_b], [1.0, 0.0], diagnostics="summary"
    )
    combined_b = _manual_bank_policy([bank_a, bank_b], [0.0, 1.0])

    selected_a = _permutation(combined_a, x, mask, 2)[..., :2]
    selected_b = _permutation(combined_b, x, mask, 2)[..., :2]

    assert torch.equal(selected_a, torch.tensor([[0, 2]]))
    assert torch.equal(selected_b, torch.tensor([[1, 3]]))
    assert len(combined_a.last_route_summary) == 2
    assert all(routes.shape == (2,) for routes in combined_a.last_route_summary)
    assert all(routes.device.type == "cpu" for routes in combined_a.last_route_summary)
    combined_a.clear_diagnostics()
    assert combined_a.last_route_summary == ()
    assert combined_a.last_confidence_summary == ()


def test_bank_diagnostics_are_opt_in() -> None:
    bank = _manual_bank([2.0, -2.0], "diagnostics")
    policy = _manual_bank_policy([bank])
    x = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])

    policy(x, torch.ones(1, 2, dtype=torch.bool))

    assert policy.last_route_summary == ()
    assert policy.last_confidence_summary == ()


def test_bank_diagnostic_summary_has_a_hard_slot_budget() -> None:
    bank = alpha.TopologyOperandBank(5, 2, bank_id="too-large-diagnostic")

    with pytest.raises(ValueError, match="diagnostic_slot_limit"):
        alpha.BankFormulaTopologyPolicy(
            dim=2,
            key_dim=2,
            banks=[bank],
            diagnostics="summary",
            diagnostic_slot_limit=4,
        )


def test_priority_ties_use_the_declared_stable_host_index_rule() -> None:
    action = alpha.TopologyAction(torch.ones(1, 5))
    mask = torch.tensor([[True, True, False, True, False]])

    permutation = alpha.StablePriorityPartition()(action, mask)

    assert torch.equal(permutation, torch.tensor([[0, 1, 3, 2, 4]]))


def test_bfloat16_priority_ties_follow_the_same_declared_rule() -> None:
    action = alpha.TopologyAction(
        torch.tensor([[1.0, 1.0, 0.5, 0.5]], dtype=torch.bfloat16)
    )
    mask = torch.ones(1, 4, dtype=torch.bool)

    permutation = alpha.StablePriorityPartition()(action, mask)

    assert torch.equal(permutation, torch.tensor([[0, 1, 2, 3]]))


def test_reset_and_shuffle_bank_change_the_priority_field() -> None:
    x = torch.tensor([[[4.0, 0.0], [0.0, 4.0], [3.0, 0.0], [0.0, 3.0]]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    bank = _manual_bank([2.0, -2.0], "a")
    policy = _manual_bank_policy([bank])
    correct = policy(x, mask).action.priority.detach().clone()

    with torch.no_grad():
        bank.values.zero_()
    reset = policy(x, mask).action.priority.detach().clone()
    with torch.no_grad():
        bank.values.copy_(torch.tensor([[-2.0], [2.0]]))
    shuffled = policy(x, mask).action.priority.detach().clone()

    assert not torch.equal(correct, reset)
    assert not torch.equal(correct, shuffled)
    assert torch.equal(_permutation(policy, x, mask, 2)[..., :2], torch.tensor([[1, 3]]))


def test_bank_values_receive_surrogate_gradients_but_query_is_fixed() -> None:
    torch.manual_seed(29)
    x = torch.randn(2, 8, 3, requires_grad=True)
    bank = alpha.TopologyOperandBank(slots=6, key_dim=4, factor_dim=1)
    policy = alpha.BankFormulaTopologyPolicy(dim=3, key_dim=4, banks=[bank])
    topology = alpha.ReversibleTopology(active_count=3, policy=policy)

    state = topology.fold(x)
    state.active.square().mean().backward()

    assert bank.values.grad is not None
    assert torch.isfinite(bank.values.grad).all()
    assert float(bank.values.grad.abs().sum()) > 0
    assert "query_basis" not in dict(policy.named_parameters())
    assert "banks.0.keys" not in dict(policy.named_parameters())


def test_duplicate_topology_bank_ids_are_rejected() -> None:
    first = alpha.TopologyOperandBank(4, 3, bank_id="duplicate")
    second = alpha.TopologyOperandBank(4, 3, bank_id="duplicate")
    with pytest.raises(ValueError, match="IDs must be unique"):
        alpha.BankFormulaTopologyPolicy(dim=3, key_dim=3, banks=[first, second])


def test_bank_policy_accepts_a_versioned_custom_formula_contract() -> None:
    class ScaledFormula(torch.nn.Module):
        _component_reference = "example/scaled-topology-formula@1"

        def __init__(self) -> None:
            super().__init__()
            self.contract = alpha.TopologyFormulaContract(factor_dim=1)
            self.register_buffer("scale", torch.tensor([2.0]))

        def evaluate(self, operands: torch.Tensor) -> alpha.TopologyFormulaOutput:
            priority = operands[..., 0] * self.scale.to(operands)
            confidence = torch.ones_like(priority[..., :1])
            return alpha.TopologyFormulaOutput(priority, confidence)

        @property
        def formula_lock(self) -> alpha.TopologyFormulaLock:
            return alpha.TopologyFormulaLock(
                formula_ref=self._component_reference,
                contract_fingerprint=self.contract.fingerprint,
                factor_dim=1,
                weight_hash="custom-fixed-scale",
            )

    bank = alpha.TopologyOperandBank(4, 3, factor_dim=1, bank_id="custom")
    policy = alpha.BankFormulaTopologyPolicy(
        dim=3,
        key_dim=3,
        banks=[bank],
        formula=ScaledFormula(),
    )
    x = torch.randn(2, 6, 3)
    mask = torch.ones(2, 6, dtype=torch.bool)

    proposal = policy(x, mask)

    assert proposal.action.priority.shape == (2, 6)
    assert policy.topology_contract()["formula"]["ref"] == (
        "example/scaled-topology-formula@1"
    )


def test_bank_policy_rejects_a_trainable_query() -> None:
    class TrainableQuery(torch.nn.Module):
        _component_reference = "example/trainable-query@1"

        def __init__(self) -> None:
            super().__init__()
            self.dim = 3
            self.key_dim = 3
            self.weight = torch.nn.Parameter(torch.eye(3))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x @ self.weight

        def topology_contract(self) -> dict[str, object]:
            return {
                "ref": self._component_reference,
                "dim": 3,
                "key_dim": 3,
                "fixed": True,
                "deterministic": True,
                "stateful": False,
            }

    bank = alpha.TopologyOperandBank(4, 3, bank_id="fixed-query-required")

    with pytest.raises(ValueError, match="requires a fixed Query"):
        alpha.BankFormulaTopologyPolicy(
            dim=3,
            key_dim=3,
            banks=[bank],
            query=TrainableQuery(),
        )


def test_bank_policy_rejects_a_query_without_a_fixed_deterministic_contract() -> None:
    class RandomQuery(torch.nn.Module):
        _component_reference = "example/random-query@1"
        dim = 3
        key_dim = 3

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.randn(*x.shape[:-1], 3, device=x.device, dtype=x.dtype)

        def topology_contract(self) -> dict[str, object]:
            return {
                "ref": self._component_reference,
                "dim": 3,
                "key_dim": 3,
                "fixed": False,
                "deterministic": False,
                "stateful": False,
            }

    bank = alpha.TopologyOperandBank(4, 3, bank_id="random-query-rejected")
    with pytest.raises(ValueError, match="fixed deterministic Query contract"):
        alpha.BankFormulaTopologyPolicy(
            dim=3,
            key_dim=3,
            banks=[bank],
            query=RandomQuery(),
        )


def test_bank_policy_rejects_a_formula_lock_from_another_contract() -> None:
    class StaleLockFormula(alpha.TopologyPriorityFormula):
        @property
        def formula_lock(self) -> alpha.TopologyFormulaLock:
            lock = super().formula_lock
            return alpha.TopologyFormulaLock(
                formula_ref=lock.formula_ref,
                contract_fingerprint="stale-contract",
                factor_dim=lock.factor_dim,
                weight_hash=lock.weight_hash,
            )

    bank = alpha.TopologyOperandBank(4, 3, bank_id="stale-lock-rejected")
    with pytest.raises(ValueError, match="lock does not match"):
        alpha.BankFormulaTopologyPolicy(
            dim=3,
            key_dim=3,
            banks=[bank],
            formula=StaleLockFormula(),
        )


def _serializable_bank_fold(*, reverse: bool = False) -> alpha.Fold:
    banks = [
        alpha.TopologyOperandBank(6, 4, seed=31, bank_id="first"),
        alpha.TopologyOperandBank(6, 4, seed=32, bank_id="second"),
    ]
    if reverse:
        banks.reverse()
    policy = alpha.BankFormulaTopologyPolicy(
        dim=3,
        key_dim=4,
        query_seed=77,
        banks=banks,
        bank_weights=[0.75, 0.25] if not reverse else [0.25, 0.75],
    )
    return alpha.Fold(active_count=2, policy=policy)


def test_bank_formula_arti_st_round_trip_preserves_hard_topology_and_layout(
    tmp_path,
) -> None:
    torch.manual_seed(41)
    source = _serializable_bank_fold().eval()
    x = torch.randn(3, 9, 3)
    before = source(x).record.permutation
    saved = arti.save(source, tmp_path / "topology-bank.arti.st")
    target = _serializable_bank_fold().eval()

    loaded = arti.load(saved.weights_path, model=target)
    after = loaded.model(x).record.permutation
    nodes = loaded.manifest["architecture"]["component_graph"]["nodes"]
    policy_node = next(
        node for node in nodes if node["ref"] == "arti/bank-formula-topology-policy@1"
    )

    assert torch.equal(before, after)
    assert [
        bank["bank_id"] for bank in policy_node["config"]["ordered_banks"]
    ] == ["first", "second"]
    assert policy_node["config"]["bank_weights"] == [0.75, 0.25]
    assert policy_node["config"]["formula"]["contract_fingerprint"]
    assert policy_node["config"]["query"]["basis_hash"]


def test_trained_bank_values_load_into_a_fresh_structural_match(tmp_path) -> None:
    source = _serializable_bank_fold().eval()
    with torch.no_grad():
        source.topology.policy.banks[0].values.add_(0.75)
        source.topology.policy.banks[1].values.sub_(0.25)
    x = torch.randn(2, 9, 3)
    expected = source(x).record.permutation
    saved = arti.save(source, tmp_path / "trained-topology-bank.arti.st")
    target = _serializable_bank_fold().eval()

    loaded = arti.load(saved.weights_path, model=target)

    assert torch.equal(loaded.model(x).record.permutation, expected)
    for source_bank, target_bank in zip(
        source.topology.policy.banks,
        loaded.model.topology.policy.banks,
        strict=True,
    ):
        torch.testing.assert_close(target_bank.values, source_bank.values)


def test_record_producer_fingerprint_binds_structure_not_mutable_bank_values() -> None:
    fold = _serializable_bank_fold()
    before = fold.topology.producer_provenance_fingerprint
    with torch.no_grad():
        fold.topology.policy.banks[0].values.add_(1.0)
    state = fold(torch.randn(1, 7, 3))

    assert fold.topology.producer_provenance_fingerprint == before
    assert state.record.producer_provenance_fingerprint == before


def test_reversed_bank_layout_has_distinct_component_config() -> None:
    source = _serializable_bank_fold()
    reversed_layout = _serializable_bank_fold(reverse=True)

    assert (
        arti.component_spec(source.topology.policy).config_fingerprint
        != arti.component_spec(reversed_layout.topology.policy).config_fingerprint
    )


def test_fixed_formula_weight_changes_producer_provenance() -> None:
    bank_a = alpha.TopologyOperandBank(6, 4, factor_dim=2, seed=31, bank_id="same")
    bank_b = alpha.TopologyOperandBank(6, 4, factor_dim=2, seed=31, bank_id="same")
    first = alpha.BankFormulaTopologyPolicy(
        dim=3,
        key_dim=4,
        banks=[bank_a],
        formula=alpha.TopologyPriorityFormula(2, weight=[1.0, 0.0]),
    )
    second = alpha.BankFormulaTopologyPolicy(
        dim=3,
        key_dim=4,
        banks=[bank_b],
        formula=alpha.TopologyPriorityFormula(2, weight=[0.0, 1.0]),
    )

    assert (
        alpha.ReversibleTopology(2, policy=first).producer_provenance_fingerprint
        != alpha.ReversibleTopology(2, policy=second).producer_provenance_fingerprint
    )


def test_bfloat16_bank_component_config_is_hashable() -> None:
    fold = _serializable_bank_fold().to(dtype=torch.bfloat16)

    assert arti.component_spec(fold.topology.policy).config_fingerprint


@torch.no_grad()
def test_unfold_operation_does_not_retain_the_topology_learner() -> None:
    bank = alpha.TopologyOperandBank(8, 4, bank_id="detached-inverse")
    policy = alpha.BankFormulaTopologyPolicy(dim=3, key_dim=4, banks=[bank])
    topology = alpha.ReversibleTopology(2, policy=policy)
    fold, unfold = topology.operations()

    assert isinstance(fold.topology.policy, alpha.BankFormulaTopologyPolicy)
    assert isinstance(unfold.inverse_contract, alpha.InverseTopologyContract)
    assert unfold.state_dict() == {}


def test_bank_formula_component_graph_declares_every_executable_role() -> None:
    fold = _serializable_bank_fold()

    graph = arti.component_graph(fold)
    refs = {node["ref"] for node in graph["nodes"]}

    assert refs >= {
        "arti/fold@2",
        "arti/reversible-topology@1",
        "arti/bank-formula-topology-policy@1",
        "arti/fixed-topology-query@1",
        "arti/topology-operand-bank@1",
        "arti/topology-priority-formula@1",
        "arti/stable-priority-partition@1",
        "arti/topology-surrogate@1",
    }


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("kind", ["learned", "bank"])
def test_cuda_fullgraph_matches_eager_value_and_gradient(kind: str) -> None:
    class CompiledLearnedTopology(torch.nn.Module):
        def __init__(self, selected_kind: str) -> None:
            super().__init__()
            if selected_kind == "learned":
                policy: torch.nn.Module = alpha.LearnedTopologyPolicy(dim=4)
            else:
                bank = alpha.TopologyOperandBank(8, 4, bank_id="compiled")
                policy = alpha.BankFormulaTopologyPolicy(
                    dim=4, key_dim=4, banks=[bank]
                )
            self.topology = alpha.ReversibleTopology(3, policy=policy)

        def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
            return self.topology.fold(x, mask).active

    torch.manual_seed(101)
    eager = CompiledLearnedTopology(kind).cuda()
    compiled_source = CompiledLearnedTopology(kind).cuda()
    compiled_source.load_state_dict(eager.state_dict())
    compiled = torch.compile(compiled_source, fullgraph=True)
    eager_x = torch.randn(2, 12, 4, device="cuda", requires_grad=True)
    compiled_x = eager_x.detach().clone().requires_grad_()
    mask = torch.rand(2, 12, device="cuda") > 0.25
    mask[:, :3] = True

    eager_output = eager(eager_x, mask)
    compiled_output = compiled(compiled_x, mask)
    torch.testing.assert_close(compiled_output, eager_output, rtol=0, atol=0)

    eager_output.square().mean().backward()
    compiled_output.square().mean().backward()

    torch.testing.assert_close(compiled_x.grad, eager_x.grad, rtol=1e-5, atol=1e-6)
    eager_gradients = {
        name: parameter.grad
        for name, parameter in eager.topology.policy.named_parameters()
        if parameter.requires_grad
    }
    compiled_gradients = {
        name: parameter.grad
        for name, parameter in compiled_source.topology.policy.named_parameters()
        if parameter.requires_grad
    }

    assert eager_gradients.keys() == compiled_gradients.keys()
    for name in eager_gradients:
        assert eager_gradients[name] is not None
        assert compiled_gradients[name] is not None
        torch.testing.assert_close(
            compiled_gradients[name], eager_gradients[name], rtol=2e-4, atol=2e-5
        )
