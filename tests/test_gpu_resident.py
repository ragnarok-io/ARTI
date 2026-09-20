from __future__ import annotations

import pytest
import torch

from arti import Recall, RefinePolicy, alpha, get_component_registry
from arti.component_registry import canonical_contract_reference


def _program() -> alpha.FormulaFabricProgram:
    return alpha.FormulaFabricProgram(
        arena_capacity=3,
        feature_dim=1,
        steps=((alpha.FormulaInvocation(alpha.FormulaPrimitive.ADD, 2),),),
    )


def _route(device: torch.device) -> alpha.FormulaRoutePlan:
    weights = torch.zeros(1, 1, 1, 2, 3, device=device)
    weights[..., 0, 0] = 1
    weights[..., 1, 1] = 1
    enabled = torch.ones(1, 1, 1, dtype=torch.bool, device=device)
    return alpha.FormulaRoutePlan(weights, enabled, enabled, enabled)


def _operation(device: torch.device) -> alpha.FormulaResidentOperation:
    compute = alpha.FormulaFabricCompute(
        alpha.FormulaFabric(_program()).to(device),
        active_count=3,
    )
    return alpha.FormulaResidentOperation(compute, _route(device))


def _topology_operation(device: torch.device) -> alpha.TopologyFormulaResidentOperation:
    topology = alpha.ReversibleTopology(
        active_count=3,
        policy=alpha.FixedTopologyPolicy(order=[2, 0, 1]),
    ).to(device)
    fold, unfold = topology.operations()
    compute = alpha.FormulaFabricCompute(
        alpha.FormulaFabric(_program()).to(device),
        active_count=3,
    )
    return alpha.TopologyFormulaResidentOperation(
        fold,
        unfold,
        compute,
        _route(device),
    )


def _refs(*, duplicate: bool = False, stale: bool = False) -> alpha.FixedPageRefs:
    offsets = torch.tensor([[0, 1, 1 if duplicate else 2]], dtype=torch.int64)
    support = torch.ones(1, 3, dtype=torch.bool)
    return alpha.FixedPageRefs(
        logical_slot=torch.tensor([[10, 11, 12]], dtype=torch.int64),
        page_id=torch.zeros(1, 3, dtype=torch.int64),
        offset=offsets,
        expected_generation=torch.full(
            (1, 3),
            1 if stale else 0,
            dtype=torch.int64,
        ),
        read_mask=support,
        write_mask=support,
        commit_mask=support,
    )


def _bound(device: torch.device):
    pool = alpha.HotPagePool(
        torch.tensor([[[2.0], [3.0], [0.0], [9.0]]], device=device)
    )
    bucket = alpha.FixedResidentBucket(
        batch_size=1,
        workset_slots=3,
        feature_dim=1,
        dtype=torch.float32,
        device=device,
    )
    return alpha.bind_hot_page_pool(pool, bucket, _refs())


def _resident_branch_fixture(
    device: torch.device,
    *,
    batch_size: int = 1,
    active_k: int | torch.Tensor = 3,
    branch_order: torch.Tensor | None = None,
):
    torch.manual_seed(1279)
    value = torch.randn(batch_size, 3, 4, device=device)
    recall = Recall(
        dim=4,
        slots=12,
        formula="arti/delta@1",
        activation="none",
        routing="grouped",
        group_size=2,
        group_topk=3,
        key_dim=4,
    ).to(device)
    with torch.no_grad():
        recall.state.recall.bank.normal_(std=0.5)
        candidates = alpha.query_recall_branches(
            recall,
            value,
            active_k=active_k,
        )
        if branch_order is not None:
            candidates = candidates.permute_branches(branch_order)
        result = alpha.run_batched_refine(
            recall,
            value,
            candidates=candidates,
            refine_policy=RefinePolicy.fixed(2, trace_level="routes"),
        )
    canonical = _canonical_branch_values(result)
    future = canonical[:, 1].detach().cpu().contiguous()
    runtime = alpha.VolatileTensorRuntime(
        {"state": value.detach().cpu().contiguous()},
        world_id="resident-wide-search-world",
        store_instance_id="resident-wide-search-store",
        abi_fingerprint="3" * 64,
        provenance_fingerprint="0" * 64,
    )
    snapshot = runtime.snapshot()
    executor = alpha.BatchedRefineExecutor.from_resident_result(result)
    spec = alpha.BranchBatchSpecV2.from_batched_refine(
        snapshot,
        executor,
        run_id="resident-wide-search-1",
        branch_ids=("route-0", "route-1", "route-2"),
        input_fingerprint="6" * 64,
        future_tape_fingerprint=alpha.tensor_content_fingerprint(
            future
        ),
        scorer_ref=alpha.FROZEN_MSE_SCORER_REF,
        scorer_config_fingerprint=alpha.frozen_mse_scorer_config_fingerprint(),
        allowed_write_keys=("state",),
        budgets=tuple(alpha.BranchBudget(1, 4) for _ in range(3)),
    )
    page_id = torch.arange(batch_size, dtype=torch.int64).unsqueeze(1).expand(-1, 3)
    offset = torch.arange(3, dtype=torch.int64).unsqueeze(0).expand(batch_size, -1)
    support = torch.ones(batch_size, 3, dtype=torch.bool)
    refs = alpha.FixedPageRefs(
        logical_slot=offset.contiguous(),
        page_id=page_id.contiguous(),
        offset=offset.contiguous(),
        expected_generation=torch.zeros(batch_size, 3, dtype=torch.int64),
        read_mask=support,
        write_mask=support,
        commit_mask=support,
    )
    pool = alpha.HotPagePool(value.detach().clone())
    bucket = alpha.FixedResidentBucket(
        batch_size=batch_size,
        workset_slots=3,
        feature_dim=4,
        dtype=torch.float32,
        device=device,
    )
    bound = alpha.bind_hot_page_pool(pool, bucket, refs)
    return value, result, executor, spec, future, bound


def _canonical_branch_values(result: alpha.BatchedRefineResult) -> torch.Tensor:
    order = torch.argsort(result.candidates.branch_origin_index, dim=1, stable=True)
    index = order[:, :, None, None].expand_as(result.value)
    return result.value.gather(1, index)


def test_physical_counter_never_encodes_unknown_as_zero() -> None:
    unknown = alpha.PhysicalCounter(
        None,
        False,
        "not-collected",
        "counter unavailable",
    )
    assert unknown.value is None and not unknown.available
    with pytest.raises(alpha.GPUResidentContractError, match="require a value"):
        alpha.PhysicalCounter(0, False, "not-collected")
    with pytest.raises(alpha.GPUResidentContractError, match="require None"):
        alpha.PhysicalCounter(None, True, "profiler")


def test_resident_branch_contracts_are_runtime_only_and_versioned() -> None:
    registry = get_component_registry()
    references = {
        "arti/resident-branch-run@1": "host_bound",
        "arti/resident-branch-score@1": "runtime_only",
        "arti/resident-branch-decision@1": "runtime_only",
        "arti/resident-branch-commit@1": "runtime_only",
    }
    for reference, policy in references.items():
        registration = registry.resolve_registration(reference)
        assert registration.reference == canonical_contract_reference(reference)
        assert not registration.constructible
        assert registration.artifact_policy == policy
    assert alpha.ResidentBranchRun._runtime_contract_ref in references


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_resident_k_way_score_and_winner_commit_keep_values_on_gpu() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    _value, result, executor, spec, future, bound = _resident_branch_fixture(device)
    run = alpha.bind_resident_branch_run(result, executor, spec, future, bound)

    score = run.score()
    assert len(score.scores) == 3
    assert score.scores[1] == pytest.approx(0.0, abs=1e-8)
    decision = run.decide(score, idempotency_key="resident-winner-1")
    assert decision.kind == "winner" and decision.winner_origin == 1

    receipt = run.commit(decision)
    assert receipt.status == "committed"
    torch.testing.assert_close(bound.pool.value, _canonical_branch_values(result)[:, 1])
    assert receipt.versions_before == (0, 0, 0)
    assert receipt.versions_after == (1, 1, 1)
    assert run.commit(decision) is receipt
    assert result.value.device.type == "cuda"
    assert result.delta.device.type == "cuda"

    cpu_authority_executor = alpha.BatchedRefineExecutor.from_result(result)
    with pytest.raises(
        alpha.TensorTransactionContractError,
        match="from_resident_result",
    ):
        alpha.bind_resident_branch_run(
            result,
            cpu_authority_executor,
            spec,
            future,
            bound,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_resident_mixture_is_fp32_ordered_and_discard_is_non_mutating() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    _value, result, executor, spec, future, bound = _resident_branch_fixture(device)
    run = alpha.bind_resident_branch_run(result, executor, spec, future, bound)
    weights = (0.2, 0.3, 0.5)
    expected = (
        _canonical_branch_values(result).float()
        * torch.tensor(weights, device=device).view(1, 3, 1, 1)
    ).sum(dim=1)
    decision = run.mix(weights, idempotency_key="resident-mixture-1")
    receipt = run.commit(decision)
    assert receipt.status == "committed"
    torch.testing.assert_close(bound.pool.value, expected)

    value2, result2, executor2, spec2, future2, bound2 = _resident_branch_fixture(
        device
    )
    run2 = alpha.bind_resident_branch_run(
        result2, executor2, spec2, future2, bound2
    )
    before = bound2.pool.value.clone()
    decision2 = run2.discard(idempotency_key="resident-discard-1")
    receipt2 = run2.commit(decision2)
    assert receipt2.status == "discarded"
    torch.testing.assert_close(bound2.pool.value, before)
    assert torch.equal(bound2.pool.version, torch.zeros_like(bound2.pool.version))
    assert value2.device.type == "cuda"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_resident_authority_is_equivariant_to_physical_branch_permutation() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    base = _resident_branch_fixture(device)
    order = torch.tensor([2, 0, 1], dtype=torch.long, device=device)
    permuted = _resident_branch_fixture(device, branch_order=order)

    run_base = alpha.bind_resident_branch_run(
        base[1], base[2], base[3], base[4], base[5]
    )
    run_permuted = alpha.bind_resident_branch_run(
        permuted[1], permuted[2], permuted[3], permuted[4], permuted[5]
    )
    score_base = run_base.score()
    score_permuted = run_permuted.score()
    assert score_permuted.scores == pytest.approx(score_base.scores)

    decision_base = run_base.decide(
        score_base,
        idempotency_key="resident-permutation-base",
    )
    decision_permuted = run_permuted.decide(
        score_permuted,
        idempotency_key="resident-permutation-permuted",
    )
    assert decision_base.winner_origin == decision_permuted.winner_origin == 1
    run_base.commit(decision_base)
    run_permuted.commit(decision_permuted)
    torch.testing.assert_close(base[5].pool.value, permuted[5].pool.value)

    base_mix = _resident_branch_fixture(device)
    permuted_mix = _resident_branch_fixture(device, branch_order=order)
    run_base_mix = alpha.bind_resident_branch_run(
        base_mix[1], base_mix[2], base_mix[3], base_mix[4], base_mix[5]
    )
    run_permuted_mix = alpha.bind_resident_branch_run(
        permuted_mix[1],
        permuted_mix[2],
        permuted_mix[3],
        permuted_mix[4],
        permuted_mix[5],
    )
    weights = (0.1, 0.2, 0.7)
    mix_base = run_base_mix.mix(
        weights,
        idempotency_key="resident-permutation-mix-base",
    )
    mix_permuted = run_permuted_mix.mix(
        weights,
        idempotency_key="resident-permutation-mix-permuted",
    )
    run_base_mix.commit(mix_base)
    run_permuted_mix.commit(mix_permuted)
    torch.testing.assert_close(
        base_mix[5].pool.value,
        permuted_mix[5].pool.value,
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_resident_run_fails_closed_on_stale_pool_and_ragged_k() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    _value, result, executor, spec, future, bound = _resident_branch_fixture(device)
    run = alpha.bind_resident_branch_run(result, executor, spec, future, bound)
    bound.pool.version.add_(1)
    with pytest.raises(
        alpha.TensorTransactionContractError,
        match="pool version changed",
    ):
        run.score()

    active_k = torch.tensor([3, 2], dtype=torch.long, device=device)
    _value, result, executor, spec, future, bound = _resident_branch_fixture(
        device,
        batch_size=2,
        active_k=active_k,
    )
    with pytest.raises(
        alpha.TensorTransactionContractError,
        match="shared active branch set",
    ):
        alpha.bind_resident_branch_run(result, executor, spec, future, bound)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_resident_runs_sharing_one_pool_reject_the_stale_second_commit() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    _value, result, executor, spec, future, bound = _resident_branch_fixture(device)
    second_binding = alpha.bind_hot_page_pool(
        bound.pool,
        bound.bucket,
        bound.refs,
    )
    first = alpha.bind_resident_branch_run(result, executor, spec, future, bound)
    second = alpha.bind_resident_branch_run(
        result,
        executor,
        spec,
        future,
        second_binding,
    )
    first_score = first.score()
    second_score = second.score()
    assert first_score.run_instance_token != second_score.run_instance_token
    with pytest.raises(
        alpha.TensorTransactionContractError,
        match="does not bind this run",
    ):
        second.decide(first_score, idempotency_key="resident-foreign-score")
    first_decision = first.decide(
        first_score,
        idempotency_key="resident-shared-first",
    )
    second_decision = second.decide(
        second_score,
        idempotency_key="resident-shared-second",
    )

    first.commit(first_decision)
    with pytest.raises(
        alpha.TensorTransactionContractError,
        match="pool version changed",
    ):
        second.commit(second_decision)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_hot_pool_validates_generation_and_duplicate_writes() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    pool = alpha.HotPagePool(torch.zeros(1, 4, 1, device=device))
    bucket = alpha.FixedResidentBucket(1, 3, 1, torch.float32, device)
    with pytest.raises(alpha.GPUResidentContractError, match="bind_hot_page_pool"):
        alpha.BoundHotPagePool(pool, bucket, _refs())
    with pytest.raises(alpha.GPUResidentContractError, match="duplicate"):
        alpha.bind_hot_page_pool(pool, bucket, _refs(duplicate=True))
    with pytest.raises(alpha.GPUResidentContractError, match="stale"):
        alpha.bind_hot_page_pool(pool, bucket, _refs(stale=True))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_bound_pool_owns_an_immutable_snapshot_of_page_refs() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    pool = alpha.HotPagePool(
        torch.tensor([[[2.0], [3.0], [0.0], [9.0]]], device=device)
    )
    bucket = alpha.FixedResidentBucket(1, 3, 1, torch.float32, device)
    refs = _refs()
    bound = alpha.bind_hot_page_pool(pool, bucket, refs)

    refs.offset.fill_(3)
    refs.read_mask.zero_()

    assert torch.equal(bound.refs.offset, torch.tensor([[0, 1, 2]]))
    assert bool(bound.refs.read_mask.all())
    output = bound.eager_step(_operation(device), commit=False, copy_output=True)
    torch.testing.assert_close(output[:, 2], torch.tensor([[5.0]], device=device))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_eager_hot_path_matches_existing_formula_and_commit_semantics() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    bound = _bound(device)
    operation = _operation(device)
    before = bound.pool.value.clone()
    output = bound.eager_step(operation, commit=False)
    torch.testing.assert_close(output[:, 2], torch.tensor([[5.0]], device=device))
    torch.testing.assert_close(bound.pool.value, before)
    assert torch.equal(bound.pool.version, torch.zeros_like(bound.pool.version))

    committed = bound.eager_step(operation, commit=True)
    torch.testing.assert_close(bound.pool.value[:, :3], committed)
    assert torch.equal(
        bound.pool.version[:, :3],
        torch.ones_like(bound.pool.version[:, :3]),
    )
    assert bound.pointer_layout_receipt().shape == (1, 3, 1)

    wrong_depth = alpha.FormulaResidentOperation(
        operation.compute,
        operation.route,
        refine_steps=2,
    )
    with pytest.raises(alpha.GPUResidentContractError, match="refine_steps differ"):
        bound.eager_step(wrong_depth, commit=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_bound_close_is_idempotent_and_does_not_own_shared_pool() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    bound = _bound(device)
    pool = bound.pool
    operation = _operation(device)

    assert bound.lifecycle_state == "open"
    assert bound.close() is True
    assert bound.close() is False
    assert bound.closed
    assert pool.lifecycle_state == "open"
    assert pool.value.numel() > 0
    assert bound.workset_input.numel() == 0
    with pytest.raises(alpha.GPUResidentContractError, match="closed"):
        bound.eager_step(operation, commit=False)
    with pytest.raises(alpha.GPUResidentContractError, match="closed"):
        bound.pointer_layout_receipt()

    assert pool.close() is True
    assert pool.close() is False
    assert pool.value.numel() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_close_failure_stays_fail_closed_and_can_be_retried(monkeypatch) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    bound = _bound(device)
    original_quiesce = bound.pool._quiesce
    attempts = 0

    def fail_once() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("injected quiesce failure")
        original_quiesce()

    monkeypatch.setattr(bound.pool, "_quiesce", fail_once)
    with pytest.raises(RuntimeError, match="injected quiesce failure"):
        bound.close()

    assert bound.lifecycle_state == "closing"
    with pytest.raises(alpha.GPUResidentContractError, match="closing"):
        bound.eager_step(_operation(device), commit=False)
    assert bound.close() is True
    assert bound.closed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_pool_close_failure_stays_fail_closed_and_can_be_retried(monkeypatch) -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    bound = _bound(device)
    pool = bound.pool
    original_quiesce = pool._quiesce
    attempts = 0

    def fail_once() -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("injected pool quiesce failure")
        original_quiesce()

    monkeypatch.setattr(pool, "_quiesce", fail_once)
    with pytest.raises(RuntimeError, match="injected pool quiesce failure"):
        pool.close()

    assert pool.lifecycle_state == "closing"
    assert bound.lifecycle_state == "open"
    with pytest.raises(alpha.GPUResidentContractError, match="closing"):
        bound.eager_step(_operation(device), commit=False)
    assert pool.close() is True
    assert pool.closed and bound.closed


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_pool_close_cascades_to_bindings_and_captured_steps() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    first = _bound(device)
    second = alpha.bind_hot_page_pool(first.pool, first.bucket, first.refs)
    operation = torch.compile(_operation(device), fullgraph=True, dynamic=False)
    for _ in range(2):
        first.eager_step(operation, commit=False)
    torch.cuda.synchronize(device)
    captured = first.capture(operation, commit=False)
    captured.replay()

    assert first.pool.close() is True
    assert first.closed and second.closed and captured.closed
    assert first.workset_output.numel() == 0
    assert second.workset_output.numel() == 0
    with pytest.raises(alpha.GPUResidentContractError, match="closed"):
        captured.replay()
    with pytest.raises(alpha.GPUResidentContractError, match="closed"):
        second.eager_step(operation, commit=False)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_topology_formula_hot_path_uses_existing_fold_and_unfold() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    bound = _bound(device)
    operation = _topology_operation(device)
    before = bound.pool.value.clone()

    output = bound.eager_step(operation, commit=False)

    torch.testing.assert_close(output[:, 0], before[:, 0])
    torch.testing.assert_close(output[:, 1], torch.tensor([[2.0]], device=device))
    torch.testing.assert_close(output[:, 2], before[:, 2])
    torch.testing.assert_close(bound.pool.value, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_compiled_and_cuda_graph_hot_paths_preserve_fixed_buffers() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    bound = _bound(device)
    operation = _operation(device)
    compiled = torch.compile(operation, fullgraph=True, dynamic=False)

    warmup_stream = torch.cuda.Stream(device=device)
    warmup_stream.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(warmup_stream):
        for _ in range(3):
            bound.eager_step(compiled, commit=False)
    torch.cuda.current_stream(device).wait_stream(warmup_stream)
    torch.cuda.synchronize(device)

    eager = bound.eager_step(compiled, commit=False, copy_output=True)
    before = bound.pointer_layout_receipt()
    captured = bound.capture(compiled, commit=False)
    actual = captured.replay(copy_output=True)
    torch.cuda.synchronize(device)
    torch.testing.assert_close(actual, eager)
    assert bound.pointer_layout_receipt() == before

    receipt = alpha.measure_captured_replays(captured, warmups=2, samples=5)
    assert receipt.pointer_stable
    assert receipt.page_miss_count.value == 0
    assert receipt.page_miss_count.source == "hot-only-contract"
    assert receipt.eviction_count.value == 0
    assert receipt.prefetch_count.value == 0
    assert not receipt.allocation_count.available


    assert receipt.p50_ms >= 0
    assert receipt.h2d_bytes.value is None and not receipt.h2d_bytes.available
    assert receipt.hbm_read_bytes.value is None and not receipt.hbm_read_bytes.available

    activity = alpha.profile_captured_cuda_activity(captured, replays=3)
    assert activity.trace_export_complete
    assert not activity.dropped_record_count.available
    assert activity.dropped_record_count.value is None
    assert activity.kernel_count > 0
    assert activity.h2d_bytes == 0
    assert activity.d2h_bytes == 0

    committed_bound = _bound(device)
    committed_operation = torch.compile(
        _operation(device),
        fullgraph=True,
        dynamic=False,
    )
    with pytest.raises(
        alpha.GPUResidentContractError,
        match="commit-capable CUDA Graph",
    ):
        committed_bound.capture(committed_operation, commit=True)

    version_before = committed_bound.pool.version.clone()
    committed_bound.eager_step(committed_operation, commit=True)
    expected_version = version_before.clone()
    expected_version[:, :3] += committed_bound.commit_mask.to(torch.int64)
    assert torch.equal(committed_bound.pool.version, expected_version)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_cross_stream_replays_are_fenced_and_owned_results_survive_reuse() -> None:
    device = torch.device("cuda", torch.cuda.current_device())
    bound = _bound(device)
    compiled = torch.compile(_operation(device), fullgraph=True, dynamic=False)
    bound.eager_step(compiled, commit=False)
    torch.cuda.synchronize(device)
    captured = bound.capture(compiled, commit=False)
    first_stream = torch.cuda.Stream(device=device)
    second_stream = torch.cuda.Stream(device=device)

    with torch.cuda.stream(first_stream):
        first = captured.replay(copy_output=True)
    with torch.cuda.stream(second_stream):
        second = captured.replay(copy_output=True)
    bound.synchronize()

    torch.testing.assert_close(first, second)
    assert first.data_ptr() != bound.workset_output.data_ptr()
    assert second.data_ptr() != bound.workset_output.data_ptr()
    assert first.data_ptr() != second.data_ptr()
