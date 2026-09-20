from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from arti import mechanisms
from benchmarks._federated_candidate_batch import execute_candidates_many
from benchmarks._federated_ordinary_programs import (
    CausalAttentionCandidate,
    ORDINARY_FAMILIES,
    replay_ordinary_route,
)
from benchmarks._federated_v4_federation import build_autonomous_effect_federation
from benchmarks._federated_search_space_migration import expand_search_space
from benchmarks.generate_qwen_federated_autonomous import FrozenReadout, _mounted_readouts


def _fixture(device="cpu", families=ORDINARY_FAMILIES):
    with torch.random.fork_rng():
        torch.manual_seed(203)
        return build_autonomous_effect_federation(
            hidden_dim=4,
            rank=3,
            seed=203,
            device=torch.device(device),
            plastic_branches=2,
            min_operations=1,
            max_operations=2,
            event_write_layout="learned-key-value-gates",
            ordinary_families=families,
        )


def _state(federation):
    state = federation.query.initial_bank_state()
    for owner in federation.query.owner_states:
        state = state.replace(owner.slot_ref, torch.randn_like(owner.value), revision=1)
    return state


def _candidate(federation, family, layer=0, terminal=False):
    candidates = (
        federation.terminal_layers[layer] if terminal else federation.transition_layers[layer]
    )
    return next(
        item for item in candidates if item.candidate_id.startswith(f"plastic-{family}-v1-0-")
    )


def _attention_route(federation):
    return (
        _candidate(federation, "causal-attention").candidate_id,
        _candidate(federation, "causal-attention", terminal=True).candidate_id,
        "stop",
    )


def test_expansion_preserves_legacy_programs_and_exposes_different_structures():
    old = _fixture(families=())
    new = _fixture()
    lookup = {item.candidate_id: item for item in new.query.candidates}
    for candidate in old.producers:
        successor = lookup[candidate.candidate_id]
        assert candidate.bank_slot_ref == successor.bank_slot_ref
        assert candidate.candidate.program.fingerprint == successor.candidate.program.fingerprint
        for name, value in candidate.candidate.operand_store.tensors().items():
            if name != candidate.plastic_bank_slot:
                assert torch.equal(value, successor.candidate.operand_store.tensor(name))
    assert len(new.query.owner_states) == 8
    programs = {
        candidate.candidate.program.fingerprint
        for candidate in new.transition_layers[0]
        if isinstance(candidate, mechanisms.FormulaProgramTensorCandidateV3)
    }
    assert len(programs) == 4
    assert len(new.effects) == 12
    parameters = {id(value) for value in new.query.parameters()}
    assert all(id(owner.value) not in parameters for owner in new.query.owner_states)


@pytest.mark.parametrize("family", ORDINARY_FAMILIES)
@pytest.mark.parametrize(
    "effect_family", ("affine", "blend", "outer", "transport", "polynomial", "proximal")
)
def test_family_executes_through_fabric_with_gradients_and_predecessor_write(family, effect_family):
    federation = _fixture()
    candidate = _candidate(federation, family)
    value = torch.randn(1, 5, 4, requires_grad=True)
    root = _state(federation)
    arena = federation.query._arena({"x": value}, bank_state=root)
    assert candidate.accepts(arena)
    result = candidate(arena)
    output = result.values.get(candidate.output_slot)
    assert output.shape == value.shape
    assert torch.isfinite(output).all()
    assert not torch.allclose(output, value)
    effect = next(
        item
        for item in federation.transition_layers[1]
        if item.candidate_id == f"{effect_family}-write-stage-1"
    )
    updated = effect(result)
    proposal = updated.proposals[-1]
    assert proposal.target == candidate.bank_slot_ref
    assert proposal.predecessor_owner_id == candidate.bank_owner_id
    assert updated.values.get(effect.output_slot) is output
    output.square().mean().backward()
    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert candidate.candidate.operand_store.tensor("lora.a").grad is not None
    assert candidate.bank_owner.value.grad is None


@pytest.mark.parametrize("family", ORDINARY_FAMILIES)
def test_zero_input_and_tiny_values_have_finite_gradients(family):
    federation = _fixture()
    candidate = _candidate(federation, family)
    value = torch.zeros(1, 4, 4, requires_grad=True)
    output = candidate(
        federation.query._arena({"x": value}, bank_state=_state(federation))
    ).values.get(candidate.output_slot)
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert torch.isfinite(value.grad).all()


def test_attention_prefix_is_causal_and_does_not_enter_query_as_mask():
    federation = _fixture()
    candidate = _candidate(federation, "causal-attention")
    value = torch.randn(1, 7, 4, requires_grad=True)
    state = _state(federation)
    arena = federation.query._arena({"x": value}, bank_state=state)
    inputs, _ = candidate._bindings(arena)
    assert inputs["causal_mask"].dtype == torch.bool
    assert federation.query.slot_ids == ("x", "operation-0", "operation-1", "terminal")
    output = candidate(arena).values.get(candidate.output_slot)
    gradient = torch.autograd.grad(output[:, :3].sum(), value)[0]
    assert torch.count_nonzero(gradient[:, 3:]) == 0
    changed = torch.cat((value[:, :3], 50 * torch.randn_like(value[:, 3:])), dim=1)
    other = candidate(federation.query._arena({"x": changed}, bank_state=state))
    torch.testing.assert_close(output[:, :3], other.values.get(candidate.output_slot)[:, :3])


def test_attention_incremental_readout_matches_full_causal_fabric():
    federation = _fixture()
    state = _state(federation)
    value = torch.randn(1, 9, 4)
    route = _attention_route(federation)
    full = replay_ordinary_route(federation, value, state, route)
    contexts = {}
    incremental = torch.cat(
        tuple(
            replay_ordinary_route(federation, chunk, state, route, contexts=contexts)
            for chunk in value.split((3, 1, 2, 3), dim=1)
        ),
        dim=1,
    )
    torch.testing.assert_close(incremental, full, atol=2e-6, rtol=1e-5)
    assert len(contexts) == 2
    assert all(item.shape == value.shape for item in contexts.values())
    assert all(owner.revision == 0 for owner in federation.query.owner_states)


@pytest.mark.parametrize("device", ["cpu"] + (["cuda"] if torch.cuda.is_available() else []))
def test_shared_child_attention_readout_keeps_context_per_full_call_path(device):
    from benchmarks._federated_peer_composition import mount_peer_federation
    from benchmarks.train_qwen_federated_autonomous_federation import _replay_data_route

    source = _fixture(device, families=("causal-attention",))
    state = _state(source)
    local_route = _attention_route(source)
    federation, _ = mount_peer_federation(source, hidden_dim=4, peer_slots=2)
    first = "call-inherited-x-to-peer-0"
    second = "call-inherited-peer-0-to-peer-1"
    route = (first, *(f"{first}/{item}" for item in local_route),
             second, *(f"{second}/{item}" for item in local_route), "join-add-peer-0-peer-1", "stop")
    value = torch.randn(1, 9, 4, device=device, requires_grad=True)
    full = _replay_data_route(federation, value, state, route)
    contexts = {}
    incremental = torch.cat(tuple(
        _replay_data_route(federation, chunk, state, route, contexts=contexts)
        for chunk in value.split((3, 1, 2, 3), dim=1)
    ), dim=1)
    torch.testing.assert_close(incremental, full, atol=2e-5, rtol=1e-5)
    assert len(contexts) == 4
    assert all(item.shape == value.shape for item in contexts.values())
    assert any(not torch.equal(contexts[f"{first}/{item}"], contexts[f"{second}/{item}"])
               for item in local_route[:-1])
    gradient = torch.autograd.grad(full[:, :3].sum(), value)[0]
    assert torch.count_nonzero(gradient[:, 3:]) == 0
    assert all(owner.revision.item() == 0 for owner in federation.query.owner_states)


def test_grouped_attention_candidates_match_native_gradients():
    federation = _fixture()
    state = _state(federation)
    candidates = tuple(
        item
        for item in federation.transition_layers[0]
        if isinstance(item, CausalAttentionCandidate)
    )
    leaves = tuple(torch.randn(1, 5, 4, requires_grad=True) for _ in candidates)
    requests = tuple(
        (item, federation.query._arena({"x": x}, bank_state=state))
        for item, x in zip(candidates, leaves)
    )
    native = tuple(item(arena).values.get(item.output_slot) for item, arena in requests)
    grouped = tuple(
        row.values.get(item.output_slot)
        for row, (item, _) in zip(execute_candidates_many(requests), requests)
    )
    for left, right in zip(native, grouped):
        torch.testing.assert_close(left, right)
    inputs = (*leaves, *(item.candidate.operand_store.tensor("lora.a") for item in candidates))
    grads = [
        torch.autograd.grad(sum(x.square().mean() for x in outputs), inputs, retain_graph=True)
        for outputs in (native, grouped)
    ]
    for left, right in zip(*grads):
        torch.testing.assert_close(left, right, atol=2e-6, rtol=1e-5)


def test_generation_hook_strips_padding_and_keeps_per_site_context():
    federation = _fixture()
    state = _state(federation)
    route = _attention_route(federation)
    layer = torch.nn.Identity()
    model = SimpleNamespace(model=SimpleNamespace(layers=[layer]))
    first = torch.randn(1, 5, 4)
    second = torch.randn(1, 3, 4)
    next_values = torch.randn(2, 1, 4)
    prompts = torch.cat((first, torch.cat((torch.full((1, 2, 4), 999.0), second), dim=1)), dim=0)
    with _mounted_readouts(
        federation,
        model,
        [FrozenReadout(state, route)] * 2,
        layer_index=0,
        prompt_lengths=(5, 3),
    ) as receipt:
        prompt_outputs = layer(prompts)
        continuation = layer(next_values)
    assert not layer._forward_hooks
    assert receipt["layer_calls"] == 2
    for row, prefix in enumerate((first, second)):
        complete = torch.cat((prefix, next_values[row : row + 1]), dim=1)
        expected = replay_ordinary_route(federation, complete, state, route)
        torch.testing.assert_close(continuation[row : row + 1], expected[:, -1:])
        torch.testing.assert_close(
            prompt_outputs[row : row + 1, -prefix.shape[1] :], expected[:, :-1]
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_causal_attention_incremental_parity():
    federation = _fixture("cuda")
    state = _state(federation)
    value = torch.randn(1, 9, 4, device="cuda")
    route = _attention_route(federation)
    contexts = {}
    full = replay_ordinary_route(federation, value, state, route)
    incremental = torch.cat(
        tuple(
            replay_ordinary_route(federation, chunk, state, route, contexts=contexts)
            for chunk in value.split((5, 1, 1, 2), dim=1)
        ),
        dim=1,
    )
    torch.testing.assert_close(incremental, full, atol=2e-5, rtol=1e-5)


def test_identity_based_migration_preserves_old_paths_adam_and_aggregate_prior():
    source = _fixture(families=())
    target = _fixture()
    old_optimizer = torch.optim.AdamW(source.query.parameters(), lr=0.001)
    new_optimizer = torch.optim.AdamW(target.query.parameters(), lr=0.001)
    for parameter in source.query.parameters():
        parameter.grad = torch.randn_like(parameter)
    old_optimizer.step()
    with torch.no_grad():
        for owner in source.query.owner_states:
            owner.value.copy_(torch.randn_like(owner.value))
    source_before = {name: value.clone() for name, value in source.query.state_dict().items()}
    receipt = expand_search_space(
        source.query,
        target.query,
        source_optimizer=old_optimizer,
        target_optimizer=new_optimizer,
    )
    assert receipt["migrated_adam_parameters"] == len(old_optimizer.state)
    assert all(
        torch.equal(value, source.query.state_dict()[name]) for name, value in source_before.items()
    )
    source_parameters = dict(source.query.named_parameters())
    target_parameters = dict(target.query.named_parameters())
    for old_name, parameter in source_parameters.items():
        new_parameter = target_parameters[receipt["parameter_name_map"][old_name]]
        for field in ("step", "exp_avg", "exp_avg_sq"):
            before, after = (
                old_optimizer.state[parameter][field],
                new_optimizer.state[new_parameter][field],
            )
            if old_name.startswith("network.2.") and field != "step":
                for i, action in enumerate(source.query.action_ids):
                    j = target.query.action_ids.index(action)
                    assert torch.equal(before[i], after[j])
            else:
                assert torch.equal(before, after)
    x = torch.randn(1, 5, 4)
    old_arena = source.query._arena({"x": x})
    new_arena = target.query._arena({"x": x})
    old_logits = source.query.network(source.query._summarize(old_arena))[0]
    new_logits = target.query.network(target.query._summarize(new_arena))[0]
    old_probabilities, new_probabilities = old_logits.softmax(-1), new_logits.softmax(-1)
    for i, action in enumerate(source.query.action_ids):
        related = [target.query.action_ids.index(action)]
        if action.startswith("plastic-lora-"):
            suffix = action.removeprefix("plastic-lora-")
            related += [
                target.query.action_ids.index(f"plastic-{family}-v1-{suffix}")
                for family in ORDINARY_FAMILIES
            ]
        torch.testing.assert_close(new_probabilities[related].sum(), old_probabilities[i])
    route = (source.producers[0].candidate_id, source.terminal_layers[0][0].candidate_id, "stop")
    old_output = replay_ordinary_route(source, x, source.query.initial_bank_state(), route)
    new_output = replay_ordinary_route(target, x, target.query.initial_bank_state(), route)
    torch.testing.assert_close(new_output, old_output, rtol=0, atol=0)
    old_refs = {owner.slot_ref for owner in source.query.owner_states}
    assert all(
        torch.count_nonzero(owner.value) == 0
        for owner in target.query.owner_states
        if owner.slot_ref not in old_refs
    )
    for parameter in target.query.parameters():
        parameter.grad = torch.zeros_like(parameter)
    new_optimizer.step()


def test_search_coverage_can_reach_each_ordinary_family():
    from benchmarks.train_federated_autonomous_federation import (
        _ranked_candidates_with_coverage_many,
    )
    from benchmarks.train_federated_branch_visible_federation import _SearchBranch
    from benchmarks._federated_ordinary_programs import ordinary_family

    federation = _fixture()
    x = torch.randn(1, 4, 4)
    arena = federation.query._arena({"x": x})
    branch = _SearchBranch(arena, x.new_zeros(()), ())
    ranked = _ranked_candidates_with_coverage_many(
        federation,
        (branch,),
        federation.transition_layers[0],
        steps=0,
        width=16,
    )[0]
    families = {ordinary_family(candidate.candidate_id) for candidate, _ in ranked}
    assert families == {"low-rank", *ORDINARY_FAMILIES}


def test_richer_training_cli_selects_sufficient_width(monkeypatch):
    from benchmarks.train_qwen_federated_autonomous_federation import parse_args

    monkeypatch.setattr("sys.argv", ["train", "--ordinary-families", *ORDINARY_FAMILIES])
    config = parse_args()
    assert config.search_width == config.beam_width == 16
    monkeypatch.setattr("sys.argv", ["train"])
    assert parse_args().search_width == 8
