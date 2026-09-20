import pytest
import torch
from torch import nn
import arti.nn as arti_nn

from arti.alpha import (
    FederalCompileError,
    FederalFormulaBlock,
    FederalParallel,
    FederalPath,
    FederalPathCompiler,
    FederalIteration,
    FederalResidual,
    FederalStaticFold,
    FederalStaticUnFold,
    FederalTensorQueryAdapter,
    FederalTensorQueryCompiler,
    FederalTensorBank,
    FederalTensorFederationCompiler,
    FederalTopologyBlock,
)
from arti import mechanisms
from arti.component_registry import canonical_contract_reference
from arti.formula_fabric import (
    FormulaFabric,
    FormulaFabricProgram,
    FormulaInvocation,
    FormulaPrimitive,
    FormulaRoutePlan,
)


class _SignChangingQuery(nn.Module):
    """Route once, change the state, then route to the tensorized exit."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        score = value[:, :1]
        return torch.cat((-score, score, 2.0 * score), dim=-1)


class _AddThree(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + 3.0


class _Double(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value * 2.0


class _RootFederationQuery(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        score = value[:, :1]
        return torch.cat((-score, 2.0 * score), dim=-1)


class _ChildFederationQuery(nn.Module):
    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value.new_ones((value.shape[0], 1))


def _path(*, route_mode="frozen", **flags):
    torch.manual_seed(41)
    parallel = FederalParallel(
        (nn.Linear(4, 2), nn.Linear(4, 2)),
        merge="concat",
    )
    return FederalPath(
        (parallel, nn.Tanh(), nn.Linear(4, 4)),
        source_ref="arti/federal-recall@2",
        source_snapshot_fingerprint="snapshot-41",
        path_ids=("root", "root/transform", "root/terminal"),
        terminal_abi_ref="arti/terminal-output-abi@1",
        refine_steps=2,
        input_shape=(None, 4),
        output_shape=(None, 4),
        route_mode=route_mode,
        operation_refs=(
            "arti/federal-parallel@1",
            "torch/tanh@1",
            "torch/linear@1",
        ),
        **flags,
    )


def test_frozen_federal_path_compiles_to_independent_layerwise_network(tmp_path):
    path = _path()
    compiled = FederalPathCompiler.compile(path)
    value = torch.randn(3, 4, requires_grad=True)
    expected = path(value)
    actual = compiled(value)
    torch.testing.assert_close(actual, expected)
    expected_final, expected_nodes = path.forward_with_intermediates(value)
    actual_final, actual_nodes = compiled.forward_with_intermediates(value)
    torch.testing.assert_close(actual_final, expected_final)
    assert len(actual_nodes) == len(expected_nodes)
    for actual_node, expected_node in zip(actual_nodes, expected_nodes, strict=True):
        torch.testing.assert_close(actual_node, expected_node)
    actual.square().mean().backward()
    assert value.grad is not None
    assert compiled.manifest.source_ref.startswith("arti/federal-recall@sha256:")
    assert compiled.manifest.refine_steps == 2
    assert compiled.manifest.input_shape == (None, 4)
    assert compiled.manifest.output_shape == (None, 4)
    assert compiled.manifest.action_sha256.startswith("sha256:")
    assert compiled.manifest.action_sha256 != compiled.manifest.fingerprint
    assert not any(module is path for module in compiled.modules())

    with torch.no_grad():
        path.operations[2].weight.zero_()
    torch.testing.assert_close(compiled(value.detach()), actual.detach())

    compiled.save(tmp_path)
    restored = compiled.load(tmp_path, _path())
    torch.testing.assert_close(restored(value.detach()), actual.detach())


def test_compile_action_identity_tracks_the_complete_compile_request():
    first = FederalPathCompiler.compile(_path()).manifest
    changed = FederalPath(
        _path().operations,
        source_ref="arti/federal-recall@2",
        source_snapshot_fingerprint="other-snapshot",
        path_ids=("root",),
        terminal_abi_ref="arti/terminal-output-abi@1",
        input_shape=(None, 4),
        output_shape=(None, 4),
    )
    second = FederalPathCompiler.compile(changed).manifest

    assert first.action_sha256 != second.action_sha256


@pytest.mark.parametrize(
    ("kwargs", "classification"),
    [
        ({"route_mode": "gated"}, "gated-static"),
        ({"route_mode": "dynamic"}, "uncompilable"),
        ({"data_dependent_shape": True}, "uncompilable"),
        ({"mutable_state": True}, "uncompilable"),
        ({"runtime_callback": True}, "uncompilable"),
        ({"unbounded_refine": True}, "uncompilable"),
    ],
)
def test_federal_assessment_keeps_dynamic_boundaries_explicit(kwargs, classification):
    path = _path(**kwargs)
    assessment = FederalPathCompiler.assess(path)
    assert assessment.classification == classification
    if classification != "exact-static":
        with pytest.raises(FederalCompileError, match="not exact-static compilable"):
            FederalPathCompiler.compile(path)


def test_non_static_operation_is_rejected():
    operation = nn.Identity()
    operation._static_compile_compatible = False
    path = FederalPath(
        (operation,),
        source_ref="arti/federal-recall@2",
        source_snapshot_fingerprint="snapshot",
        path_ids=("root",),
        terminal_abi_ref="arti/terminal-output-abi@1",
    )
    assessment = FederalPathCompiler.assess(path)
    assert assessment.classification == "uncompilable"
    assert "not static-compatible" in assessment.reasons[0]


def test_dynamic_public_fold_unfold_is_not_silently_compiled():
    path = FederalPath(
        (nn.Sequential(nn.Identity(), arti_nn.UnFold(dim=4, exposed=1)),),
        source_ref="arti/federal-recall@3",
        source_snapshot_fingerprint="snapshot-dynamic-layout",
        path_ids=("root", "root/unfold"),
        terminal_abi_ref="arti/terminal-output-abi@1",
        input_shape=(None, 3, 4),
        output_shape=(None, 4, 4),
    )

    assessment = FederalPathCompiler.assess(path)
    assert assessment.classification == "uncompilable"
    assert any("dynamic arti.nn.UnFold" in reason for reason in assessment.reasons)


def test_fixed_topology_refine_and_residual_compile_as_ordinary_modules():
    topology = mechanisms.ReversibleTopology(
        active_count=2,
        policy=mechanisms.FixedTopologyPolicy(order=[2, 0, 3, 1]),
    )
    fold, unfold = topology.operations()
    path = FederalPath(
        (
            FederalTopologyBlock(fold, nn.Linear(4, 4), unfold),
            FederalIteration(nn.Identity(), steps=3),
            FederalResidual(nn.Identity()),
        ),
        source_ref="arti/federal-recall@3",
        source_snapshot_fingerprint="snapshot-topology-1",
        path_ids=("root", "root/fold", "root/refine", "root/residual"),
        terminal_abi_ref="arti/terminal-output-abi@1",
        refine_steps=3,
        input_shape=(None, 4, 4),
        output_shape=(None, 4, 4),
    )

    compiled = FederalPathCompiler.compile(path)
    value = torch.randn(2, 4, 4)
    torch.testing.assert_close(compiled(value), path(value))
    assert tuple(
        reference.split("@", maxsplit=1)[0]
        for reference in compiled.manifest.dependency_refs
    ) == ("arti/fold", "arti/unfold")


def test_dynamic_topology_is_not_compiled_as_a_fixed_path():
    topology = mechanisms.ReversibleTopology(
        active_count=2,
        policy=mechanisms.LearnedTopologyPolicy(dim=4),
    )
    fold, unfold = topology.operations()
    path = FederalPath(
        (FederalTopologyBlock(fold, nn.Identity(), unfold),),
        source_ref="arti/federal-recall@3",
        source_snapshot_fingerprint="snapshot-topology-dynamic",
        path_ids=("root", "root/fold"),
        terminal_abi_ref="arti/terminal-output-abi@1",
        input_shape=(None, 4, 4),
        output_shape=(None, 4, 4),
    )

    assessment = FederalPathCompiler.assess(path)
    assert assessment.classification == "uncompilable"
    assert any("static-compatible" in reason for reason in assessment.reasons)


def test_fixed_shape_changing_fold_unfold_compile_as_tensor_transport():
    path = FederalPath(
        (
            FederalStaticFold(input_length=5, active_indices=(4, 1, 3)),
            nn.Linear(4, 4),
            FederalStaticUnFold(
                input_length=3,
                source_indices=(0, -1, 1, 2, -1, 0),
                insert_values=torch.full((6, 4), 0.25),
            ),
        ),
        source_ref="arti/federal-recall@3",
        source_snapshot_fingerprint="snapshot-shape-changing-1",
        path_ids=("root", "root/fold", "root/head", "root/unfold"),
        terminal_abi_ref="arti/terminal-output-abi@1",
        input_shape=(None, 5, 4),
        output_shape=(None, 6, 4),
    )

    compiled = FederalPathCompiler.compile(path)
    value = torch.randn(2, 5, 4)
    expected = path(value)
    actual = compiled(value)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2, 6, 4)
    assert tuple(
        reference.split("@", maxsplit=1)[0]
        for reference in compiled.manifest.dependency_refs
    ) == ("arti/fold", "arti/unfold")


def test_path_can_be_built_from_federal_trace_without_serializing_runtime_state():
    trace = mechanisms.FederalTrace(
        max_k=2,
        max_levels=2,
        steps=(),
        winner_paths=("root/child/terminal",),
    )
    path = FederalPath.from_trace(
        trace,
        (nn.Identity(),),
        source_ref="arti/federal-recall@3",
        source_snapshot_fingerprint="snapshot-trace-1",
        terminal_abi_ref="arti/terminal-output-abi@1",
        input_shape=(None, 4),
        output_shape=(None, 4),
    )

    assert path.path_ids == ("root", "child", "terminal")
    assert path.source_snapshot_fingerprint == "snapshot-trace-1"
    assert path.source_ref == canonical_contract_reference("arti/federal-recall@3")
    assert path.terminal_abi_ref == canonical_contract_reference(
        "arti/terminal-output-abi@1"
    )


def test_fixed_formula_fabric_route_compiles_to_tensor_module():
    fabric = FormulaFabric(
        FormulaFabricProgram(
            arena_capacity=4,
            feature_dim=4,
            steps=((FormulaInvocation(FormulaPrimitive.IDENTITY, 1),),),
        )
    )
    weights = torch.zeros(1, 1, 1, 1, 4)
    weights[..., 0] = 1
    route = FormulaRoutePlan(
        weights=weights,
        valid_mask=torch.ones(1, 1, 1, dtype=torch.bool),
        fire_mask=torch.ones(1, 1, 1, dtype=torch.bool),
        commit_mask=torch.ones(1, 1, 1, dtype=torch.bool),
    )
    path = FederalPath(
        (FederalFormulaBlock(fabric, route, output_slots=(1, 2)),),
        source_ref="arti/federal-recall@3",
        source_snapshot_fingerprint="snapshot-formula-1",
        path_ids=("root", "root/formula"),
        terminal_abi_ref="arti/terminal-output-abi@1",
        input_shape=(None, 4, 4),
        output_shape=(None, 2, 4),
    )
    compiled = FederalPathCompiler.compile(path)
    value = torch.randn(3, 4, 4)

    torch.testing.assert_close(compiled(value), path(value))
    assert tuple(
        reference.split("@", maxsplit=1)[0]
        for reference in compiled.manifest.dependency_refs
    ) == ("arti/formula-fabric",)


def test_tensorized_query_requeries_latest_state_without_python_dispatch():
    value = torch.tensor([[-2.0], [2.0]])
    graph = FederalTensorQueryCompiler.compile(
        _SignChangingQuery(),
        (_AddThree(), _Double()),
        example_input=value,
        source_ref="arti/federal-recall@3",
        source_snapshot_fingerprint="snapshot-tensor-query",
        max_steps=4,
        route_mode="hard",
        input_shape=(None, 1),
        output_shape=(None, 1),
    )

    actual = graph(value)
    torch.testing.assert_close(actual, torch.tensor([[1.0], [2.0]]))
    assert graph.manifest is not None
    assert graph.manifest.route_mode == "hard"
    assert graph.manifest.max_steps == 4

    exported = FederalTensorQueryCompiler.export(graph, value)
    restored = exported.module()
    torch.testing.assert_close(restored(value), actual)

    compiled = torch.compile(graph, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(value), actual)


def test_tensorized_query_straight_through_route_trains_query():
    torch.manual_seed(73)
    value = torch.randn(6, 2)
    query = nn.Linear(2, 3)
    graph = FederalTensorQueryCompiler.compile(
        query,
        (nn.Linear(2, 2), nn.Linear(2, 2)),
        example_input=value,
        source_ref="arti/federal-recall@3",
        source_snapshot_fingerprint="snapshot-tensor-query-grad",
        max_steps=3,
        route_mode="straight_through",
    )

    loss = graph(value).square().mean()
    loss.backward()
    assert query.weight.grad is None
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in graph.query.parameters()
    )


def test_sealed_bank_query_result_can_be_unwrapped_before_export():
    input_schema = mechanisms.TensorSchema(
        dtype="float32",
        device_class="any",
        dimensions=("B", 1),
        semantic_axes=("batch", "feature"),
        mask_semantics="none",
    )
    output_schema = mechanisms.TensorSchema(
        dtype="float32",
        device_class="any",
        dimensions=("B", 3),
        semantic_axes=("batch", "action"),
        mask_semantics="none",
    )
    query = mechanisms.LinearBankQuery(
        input_schema,
        output_schema,
        input_dim=1,
        query_dim=3,
        bias=False,
    )
    adapted = FederalTensorQueryAdapter(query)
    value = torch.randn(2, 1)
    graph = FederalTensorQueryCompiler.compile(
        adapted,
        (nn.Identity(), nn.Identity()),
        example_input=value,
        source_ref="arti/federal-recall@2",
        source_snapshot_fingerprint="snapshot-bank-query-adapter",
        max_steps=2,
    )
    exported = FederalTensorQueryCompiler.export(graph, value)
    torch.testing.assert_close(exported.module()(value), graph(value))


def test_tensorized_federation_compiles_dynamic_bank_dispatch_and_k_wide_routes():
    value = torch.tensor([[-2.0], [2.0]])
    federation = FederalTensorFederationCompiler.compile(
        (
            FederalTensorBank(
                "root",
                _RootFederationQuery(),
                (_AddThree(), _Double()),
                ("child", None),
            ),
            FederalTensorBank(
                "child",
                _ChildFederationQuery(),
                (nn.Identity(),),
                (None,),
            ),
        ),
        root_bank_id="root",
        example_input=value,
        source_ref="arti/federal-recall@3",
        source_snapshot_fingerprint="snapshot-tensor-federation",
        max_levels=2,
        beam_width=2,
        input_shape=(None, 1),
        output_shape=(None, 1),
    )

    actual = federation(value)
    torch.testing.assert_close(actual, torch.tensor([[1.0], [4.0]]))
    exported = FederalTensorFederationCompiler.export(federation, value)
    torch.testing.assert_close(exported.module()(value), actual)
    compiled = torch.compile(federation, backend="eager", fullgraph=True)
    torch.testing.assert_close(compiled(value), actual)


@pytest.mark.parametrize("boundary", ("mutable_state", "data_dependent_shape", "runtime_callback"))
def test_tensorized_federation_rejects_runtime_only_boundaries(boundary):
    value = torch.randn(2, 1)
    bank = FederalTensorBank(
        "root",
        _RootFederationQuery(),
        (_AddThree(), _Double()),
        (None, None),
        **{boundary: True},
    )
    with pytest.raises(FederalCompileError, match="runtime-only|not tensorized"):
        FederalTensorFederationCompiler.compile(
            (bank,),
            root_bank_id="root",
            example_input=value,
            source_ref="arti/federal-recall@3",
            source_snapshot_fingerprint="snapshot-boundary",
            max_levels=2,
        )


@pytest.mark.parametrize("boundary", ("mutable_state", "data_dependent_shape", "runtime_callback"))
def test_tensorized_query_rejects_runtime_only_boundaries(boundary):
    value = torch.randn(2, 1)
    with pytest.raises(FederalCompileError, match="not tensorized"):
        FederalTensorQueryCompiler.compile(
            _SignChangingQuery(),
            (_AddThree(), _Double()),
            example_input=value,
            source_ref="arti/federal-recall@3",
            source_snapshot_fingerprint="snapshot-query-boundary",
            max_steps=2,
            **{boundary: True},
        )
