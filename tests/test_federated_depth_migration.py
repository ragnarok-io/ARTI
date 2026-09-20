from pathlib import Path

import pytest
import torch

from arti import mechanisms

from benchmarks._federated_search_space_migration import (
    depth_candidate_donors, expand_search_space, reset_execution_counts,
)
from benchmarks._federated_v4_federation import build_autonomous_effect_federation
from benchmarks.train_qwen_federated_online_meta import load_checkpoint, save_checkpoint
from benchmarks.train_federated_autonomous_federation import _replay_route


def _fixture(depth):
    return build_autonomous_effect_federation(
        hidden_dim=4, rank=4, seed=711, device=torch.device("cpu"),
        plastic_branches=4, min_operations=1, max_operations=3,
        max_effect_operations=depth, event_write_layout="learned-key-value-gates",
        ordinary_families=("gelu", "gated", "causal-attention"),
    )


def test_depth_migration_preserves_paths_logits_adam_and_new_occurrence_donors(tmp_path: Path):
    source, target = _fixture(None), _fixture(3)
    old_optimizer = torch.optim.AdamW(source.query.parameters(), lr=0.001)
    new_optimizer = torch.optim.AdamW(target.query.parameters(), lr=0.001)
    for parameter in source.query.parameters():
        parameter.grad = torch.randn_like(parameter)
    old_optimizer.step()
    with torch.no_grad():
        for owner in source.query.owner_states:
            owner.value.copy_(torch.randn_like(owner.value))
    before = {name: value.clone() for name, value in source.query.state_dict().items()}
    donors = depth_candidate_donors(source.query, target.query)
    receipt = expand_search_space(
        source.query, target.query, source_optimizer=old_optimizer, target_optimizer=new_optimizer,
        candidate_donors=donors, split_routing_prior=False,
    )
    assert len(source.query.owner_states) == len(target.query.owner_states)
    assert set(donors) == set(target.query.candidate_ids) - set(source.query.candidate_ids)
    assert all(torch.equal(value, source.query.state_dict()[name]) for name, value in before.items())
    columns = receipt["summary_column_targets"]
    new_columns = sorted(set(range(target.query.network[0].in_features)) - set(columns))
    old_rows = [target.query.action_ids.index(action) for action in source.query.action_ids]
    new_state = target.query.state_dict()
    for name, value in before.items():
        actual = new_state[receipt["parameter_name_map"][name]]
        if name == "network.0.weight":
            assert torch.count_nonzero(actual[:, new_columns]) == 0
            actual = actual[:, columns]
        elif name.startswith("network.2."):
            actual = actual[old_rows]
        assert torch.equal(value, actual)
    for destination, origin in receipt["donor_parameter_name_map"].items():
        assert torch.equal(new_state[destination], before[origin])

    old_names = dict(source.query.named_parameters())
    new_names = dict(target.query.named_parameters())
    mappings = {target_name: name for name, target_name in receipt["parameter_name_map"].items()}
    mappings.update(receipt["donor_parameter_name_map"])
    for target_name, parameter in new_names.items():
        origin = mappings[target_name]
        fields = old_optimizer.state[old_names[origin]]
        for field, value in fields.items():
            actual = new_optimizer.state[parameter][field]
            if field != "step" and origin == "network.0.weight":
                assert torch.count_nonzero(actual[:, new_columns]) == 0
                actual = actual[:, columns]
            elif field != "step" and origin.startswith("network.2."):
                actual = actual[old_rows]
            assert torch.equal(actual, value)

    values = {slot: torch.randn(1, 3, 4) for slot in source.query.slot_ids}
    old_logits = source.query.network(source.query._summarize(source.query._arena(values)))
    new_logits = target.query.network(target.query._summarize(target.query._arena(values)))
    torch.testing.assert_close(new_logits[:, old_rows], old_logits, rtol=1e-5, atol=1e-6)
    for target_id, source_id in donors.items():
        torch.testing.assert_close(
            new_logits[:, target.query.action_ids.index(target_id)],
            old_logits[:, source.query.action_ids.index(source_id)], rtol=1e-5, atol=1e-6,
        )
    effect = next(item for item in source.transition_layers[1]
                  if isinstance(item, mechanisms.FormulaProgramEffectCandidateV3))
    route = (source.producers[0].candidate_id, effect.candidate_id,
             source.terminal_layers[1][0].candidate_id, "stop")
    old_result = _replay_route(source, values["x"], source.query.initial_bank_state(), route)
    new_result = _replay_route(target, values["x"], target.query.initial_bank_state(), route)
    torch.testing.assert_close(new_result.value, old_result.value, rtol=0, atol=0)
    for left, right in zip(old_result.bank_state.values, new_result.bank_state.values, strict=True):
        assert torch.equal(left, right)

    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}", encoding="utf-8")
    save_checkpoint(tmp_path, target.query, new_optimizer, step=160,
                    protocol_path=protocol, qwen_digest="unit-test", metrics={"max_effect_operations": 3})
    restored = _fixture(3)
    restored_optimizer = torch.optim.AdamW(restored.query.parameters(), lr=0.001)
    assert load_checkpoint(tmp_path / "checkpoints/latest", restored.query, restored_optimizer,
                           protocol_path=protocol, device=torch.device("cpu")) == 160
    assert all(torch.equal(value, restored.query.state_dict()[name]) for name, value in new_state.items())
    for name, parameter in restored.query.named_parameters():
        for field, value in restored_optimizer.state[parameter].items():
            assert torch.equal(value, new_optimizer.state[new_names[name]][field])
    for parameter in restored.query.parameters():
        parameter.grad = torch.zeros_like(parameter)
    restored_optimizer.step()


@pytest.mark.parametrize("change", ("broadcast", "count-trainability"))
def test_depth_donor_preserves_operand_execution_contract(change):
    source, target = _fixture(None), _fixture(1)
    tail = target.tail_layers[0][0]
    if change == "broadcast":
        tail.batch_broadcast_operands = (
            frozenset() if tail.batch_broadcast_operands
            else frozenset((tail.operand_store.names[0],))
        )
    else:
        assert tail.execution_count is not None
        tail.execution_count.requires_grad_(False)
    with pytest.raises(ValueError, match="operand contract differs"):
        expand_search_space(
            source.query, target.query,
            candidate_donors=depth_candidate_donors(source.query, target.query),
            split_routing_prior=False,
        )


def test_later_depth_expansion_inherits_existing_tail_before_operation_fallback():
    source, target = _fixture(2), _fixture(3)
    donors = depth_candidate_donors(source.query, target.query)
    for new, old in zip(target.tail_layers[2], source.tail_layers[1], strict=True):
        assert donors[new.candidate_id] == old.candidate_id


def test_count_fork_changes_only_counts_and_their_adam_history(tmp_path):
    federation = _fixture(2)
    query = federation.query
    optimizer = torch.optim.AdamW(query.parameters(), lr=0.001, amsgrad=True)
    for parameter in query.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()
    before = {name: tensor.clone() for name, tensor in query.state_dict().items()}
    before_optimizer = {
        name: {field: value.clone() for field, value in optimizer.state[parameter].items()}
        for name, parameter in query.named_parameters()
    }
    receipt = reset_execution_counts(query, optimizer, 2.0)
    changed = {row["parameter"] for row in receipt["changes"]}
    assert len(changed) == len(federation.effects) == receipt["changed_parameter_count"]
    assert any(name.endswith("execution_count") for name in changed)
    assert any(not name.endswith("execution_count") for name in changed)
    assert all("max_exp_avg_sq" in row["removed_optimizer_fields"] for row in receipt["changes"])
    for name, value in query.state_dict().items():
        if name in changed:
            assert float(value) == 2.0
        else:
            assert torch.equal(value, before[name])
    for name, parameter in query.named_parameters():
        if name in changed:
            assert parameter not in optimizer.state
        else:
            for field, value in optimizer.state[parameter].items():
                assert torch.equal(value, before_optimizer[name][field])
    protocol = tmp_path / "protocol.json"
    protocol.write_text("{}", encoding="utf-8")
    save_checkpoint(tmp_path, query, optimizer, step=160, protocol_path=protocol,
                    qwen_digest="unit-test", metrics={"count_fork": receipt})
    restored = _fixture(2).query
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.001, amsgrad=True)
    assert load_checkpoint(tmp_path / "checkpoints/latest", restored, restored_optimizer,
                           protocol_path=protocol, device=torch.device("cpu")) == 160
    assert all(torch.equal(value, restored.state_dict()[name]) for name, value in query.state_dict().items())
    for name, parameter in restored.named_parameters():
        if name in changed:
            assert parameter not in restored_optimizer.state
        parameter.grad = torch.zeros_like(parameter)
    restored_optimizer.step()
    assert all(torch.isfinite(parameter).all() for parameter in restored.parameters())


@pytest.mark.parametrize("value", (-1.0, float("nan"), float("inf"), 5.0))
def test_count_fork_rejects_invalid_count_before_changes(value):
    query = _fixture(1).query
    optimizer = torch.optim.AdamW(query.parameters(), lr=0.001)
    before = {name: tensor.clone() for name, tensor in query.state_dict().items()}
    with pytest.raises(ValueError, match="execution-count initialization"):
        reset_execution_counts(query, optimizer, value)
    assert all(torch.equal(value, before[name]) for name, value in query.state_dict().items())
