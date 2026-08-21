from __future__ import annotations

import pytest
import torch

import arti
from arti import alpha
from arti.reversible_topology import FoldRecord, FoldedTensor


def _markers(batch: int = 2, length: int = 4, dim: int = 3) -> torch.Tensor:
    return torch.arange(batch * length * dim, dtype=torch.float32).reshape(batch, length, dim)


def test_fixed_topology_fold_and_unfold_are_exact() -> None:
    x = _markers()
    topology = alpha.ReversibleTopology(
        active_count=2,
        policy=alpha.FixedTopologyPolicy(order=[2, 0, 3, 1]),
    )
    fold, unfold = topology.operations()

    state = fold(x)

    torch.testing.assert_close(state.active, x[:, [2, 0]], rtol=0, atol=0)
    torch.testing.assert_close(state.folded, x[:, [3, 1]], rtol=0, atol=0)
    result = unfold(state)
    torch.testing.assert_close(result.value, x, rtol=0, atol=0)
    assert torch.equal(result.mask, torch.ones(2, 4, dtype=torch.bool))


def test_masked_instances_do_not_displace_valid_active_instances() -> None:
    x = _markers(batch=1)
    mask = torch.tensor([[False, True, False, True]])
    fold = alpha.Fold(active_count=2)

    state = fold(x, mask)

    assert torch.equal(state.record.permutation, torch.tensor([[1, 3, 0, 2]]))
    assert torch.equal(state.active_mask, torch.tensor([[True, True]]))
    assert torch.equal(state.folded_mask, torch.tensor([[False, False]]))
    restored = fold.topology.unfold(state)
    torch.testing.assert_close(restored.value, x, rtol=0, atol=0)
    assert torch.equal(restored.mask, mask)


def test_ragged_and_all_masked_batches_have_valid_records() -> None:
    x = _markers(batch=3, length=5, dim=2)
    mask = torch.tensor(
        [
            [True, False, True, False, False],
            [False, False, False, False, False],
            [True, True, True, True, True],
        ]
    )
    topology = alpha.ReversibleTopology(active_count=3)

    state = topology.fold(x, mask)
    restored = topology.unfold(state)

    torch.testing.assert_close(restored.value, x, rtol=0, atol=0)
    assert torch.equal(restored.mask, mask)
    assert torch.equal(torch.sort(state.record.permutation, dim=-1).values, torch.arange(5).expand(3, 5))


def test_fold_payload_does_not_alias_the_original_input_or_public_record_views() -> None:
    original = _markers(batch=1)
    x = original.clone()
    topology = alpha.ReversibleTopology(
        active_count=2,
        policy=alpha.FixedTopologyPolicy(order=[3, 1, 0, 2]),
    )
    state = topology.fold(x)

    x.fill_(-999)
    public_permutation = state.record.permutation
    public_permutation.zero_()

    restored = topology.unfold(state)
    torch.testing.assert_close(restored.value, original, rtol=0, atol=0)
    assert torch.equal(state.record.permutation, torch.tensor([[3, 1, 0, 2]]))


def test_active_and_folded_mutations_return_to_their_host_slots_only() -> None:
    x = _markers(batch=1, length=4, dim=2)
    topology = alpha.ReversibleTopology(
        active_count=2,
        policy=alpha.FixedTopologyPolicy(order=[2, 0, 3, 1]),
    )
    state = topology.fold(x)

    active = state.active.clone()
    active[:, 0] += 100
    active_result = topology.unfold(state.replace(active=active)).value
    expected_active = x.clone()
    expected_active[:, 2] += 100
    torch.testing.assert_close(active_result, expected_active, rtol=0, atol=0)

    folded = state.folded.clone()
    folded[:, 1] -= 50
    folded_result = topology.unfold(state.replace(folded=folded)).value
    expected_folded = x.clone()
    expected_folded[:, 1] -= 50
    torch.testing.assert_close(folded_result, expected_folded, rtol=0, atol=0)


def test_nested_topologies_unfold_in_lifo_order() -> None:
    x = _markers(batch=1, length=6, dim=2)
    outer = alpha.ReversibleTopology(
        active_count=4,
        policy=alpha.FixedTopologyPolicy(order=[5, 1, 3, 0, 4, 2]),
    )
    inner = alpha.ReversibleTopology(
        active_count=2,
        policy=alpha.FixedTopologyPolicy(order=[2, 0, 3, 1]),
    )

    outer_state = outer.fold(x)
    inner_state = inner.fold(outer_state.active)
    restored_outer_active = inner.unfold(inner_state).value
    restored = outer.unfold(outer_state.replace(active=restored_outer_active)).value

    torch.testing.assert_close(restored, x, rtol=0, atol=0)


def test_value_gradient_follows_only_the_selected_transport_path() -> None:
    x = _markers(batch=1, length=4, dim=1).requires_grad_()
    topology = alpha.ReversibleTopology(
        active_count=2,
        policy=alpha.FixedTopologyPolicy(order=[2, 0, 3, 1]),
    )
    state = topology.fold(x)
    y = topology.unfold(state.replace(active=state.active * 2)).value

    y.sum().backward()

    assert x.grad is not None
    expected = torch.tensor([[[2.0], [1.0], [2.0], [1.0]]])
    torch.testing.assert_close(x.grad, expected, rtol=0, atol=0)


def test_invalid_permutation_fails_and_inverse_needs_only_transport_contract() -> None:
    x = _markers(batch=1)
    mask = torch.ones(1, 4, dtype=torch.bool)
    with pytest.raises(ValueError, match="complete per-sample bijection"):
        FoldRecord(
            permutation=torch.tensor([[0, 0, 2, 3]]),
            original_mask=mask,
            original_shape=x.shape,
            axis=-2,
            active_count=2,
            topology_config_fingerprint="test",
        )

    source = alpha.ReversibleTopology(active_count=2)
    state = source.fold(x)
    different = alpha.ReversibleTopology(
        active_count=2,
        policy=alpha.FixedTopologyPolicy(order=[1, 0, 2, 3]),
    )
    restored = different.unfold(state)
    assert torch.equal(restored.value, x)
    assert source.producer_provenance_fingerprint != different.producer_provenance_fingerprint


def test_mask_lineage_tampering_is_rejected() -> None:
    x = _markers(batch=1)
    topology = alpha.ReversibleTopology(active_count=2)
    state = topology.fold(x, torch.tensor([[True, True, False, False]]))
    tampered = FoldedTensor(
        active=state.active,
        folded=state.folded,
        active_mask=~state.active_mask,
        folded_mask=state.folded_mask,
        record=state.record,
    )

    with pytest.raises(ValueError, match="mask lineage"):
        topology.unfold(tampered)


def test_fold_record_metadata_is_immutable_after_construction() -> None:
    record = alpha.ReversibleTopology(2).fold(_markers(batch=1)).record

    with pytest.raises(AttributeError, match="immutable"):
        record.active_count = 3

    exposed = record.permutation
    exposed.zero_()
    assert not torch.equal(exposed, record.permutation)


def test_component_versions_and_dependencies_are_explicit() -> None:
    old_fold = arti.resolve_component("arti/fold@1", k=2, dim=3)
    new_fold = arti.resolve_component("arti/fold@2", active_count=2)
    new_unfold = arti.resolve_component("arti/unfold@2", active_count=2)

    assert arti.component_ref(old_fold) == "arti/fold@1"
    assert arti.component_ref(new_fold) == "arti/fold@2"
    assert arti.component_ref(new_unfold) == "arti/unfold@2"
    assert arti.component_ref(new_fold.topology) == "arti/reversible-topology@1"
    assert arti.component_ref(new_fold.topology.policy) == "arti/fixed-topology-policy@1"

    assert arti.component_spec(new_fold).dependencies == ("arti/reversible-topology@1",)
    assert arti.component_spec(new_unfold).dependencies == (
        "arti/inverse-topology-contract@1",
    )
    assert new_unfold.state_dict() == {}


def test_state_dict_round_trip_preserves_the_fixed_topology() -> None:
    source = alpha.Fold(
        active_count=2,
        policy=alpha.FixedTopologyPolicy(order=[3, 1, 0, 2]),
    )
    target = alpha.Fold(
        active_count=2,
        policy=alpha.FixedTopologyPolicy(order=[0, 1, 2, 3]),
    )
    target.load_state_dict(source.state_dict())
    x = _markers(batch=1)

    assert torch.equal(source(x).record.permutation, target(x).record.permutation)
    assert source.topology.contract_fingerprint == target.topology.contract_fingerprint


def test_arti_st_round_trip_preserves_component_versions_and_order(tmp_path) -> None:
    source = alpha.Fold(
        active_count=2,
        policy=alpha.FixedTopologyPolicy(order=[2, 0, 3, 1]),
    ).eval()
    saved = arti.save(source, tmp_path / "topology.arti.st")
    target = alpha.Fold(
        active_count=2,
        policy=alpha.FixedTopologyPolicy(order=[2, 0, 3, 1]),
    ).eval()

    loaded = arti.load(saved.weights_path, model=target)
    refs = {node["ref"] for node in loaded.manifest["architecture"]["component_graph"]["nodes"]}

    assert loaded.model is target
    assert refs >= {
        "arti/fixed-topology-policy@1",
        "arti/fold@2",
        "arti/reversible-topology@1",
    }
    assert torch.equal(
        source(_markers(batch=1)).record.permutation,
        target(_markers(batch=1)).record.permutation,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cpu_dtype_round_trip(dtype: torch.dtype) -> None:
    x = _markers(batch=1).to(dtype)
    topology = alpha.ReversibleTopology(active_count=2)
    restored = topology.unfold(topology.fold(x)).value
    assert restored.dtype == dtype
    assert torch.equal(restored, x)


def test_single_instance_has_an_empty_folded_payload() -> None:
    x = _markers(batch=2, length=1, dim=3)
    topology = alpha.ReversibleTopology(active_count=1)
    state = topology.fold(x)

    assert state.active.shape == (2, 1, 3)
    assert state.folded.shape == (2, 0, 3)
    assert torch.equal(topology.unfold(state).value, x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_cuda_round_trip_and_gradient(dtype: torch.dtype) -> None:
    x = _markers(batch=2, length=8, dim=4).cuda().to(dtype).requires_grad_()
    topology = alpha.ReversibleTopology(
        active_count=3,
        policy=alpha.FixedTopologyPolicy(order=[7, 1, 4, 0, 6, 3, 5, 2]),
    ).cuda()
    state = topology.fold(x)
    result = topology.unfold(state.replace(active=state.active + 1)).value

    assert result.device.type == "cuda"
    assert result.dtype == dtype
    result.float().sum().backward()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cuda_fullgraph_compile_preserves_the_topology_contract(
    dtype: torch.dtype,
) -> None:
    class CompiledTopology(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            topology = alpha.ReversibleTopology(active_count=2)
            self.fold, self.unfold = topology.operations()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            state = self.fold(x)
            return self.unfold(state.replace(active=state.active * 2)).value

    eager = CompiledTopology().cuda().to(dtype=dtype)
    model = torch.compile(CompiledTopology().cuda().to(dtype=dtype), fullgraph=True)
    x = torch.randn(2, 4, 3, device="cuda", dtype=dtype, requires_grad=True)
    compiled_x = x.detach().clone().requires_grad_()
    expected = x.clone()
    expected[:, :2] *= 2

    eager_output = eager(x)
    compiled_output = model(compiled_x)
    torch.testing.assert_close(eager_output, expected, rtol=0, atol=0)
    torch.testing.assert_close(compiled_output, expected, rtol=0, atol=0)
    eager_output.float().sum().backward()
    compiled_output.float().sum().backward()
    torch.testing.assert_close(compiled_x.grad, x.grad, rtol=0, atol=0)
