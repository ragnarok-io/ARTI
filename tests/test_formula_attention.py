from __future__ import annotations

import pytest
import torch
from torch import Tensor, nn

import arti
from arti.alpha import (
    ActiveWorkspace,
    FactorInterventionPolicy,
    FoldedWorkspace,
    FormulaAttention,
    MagnitudeInterventionPolicy,
    PulseSupports,
    ScaleShiftFormula,
    SelectiveCompute,
    StableTopKIntervention,
    SupportDomain,
    SupportKind,
    SupportMask,
    fold_pulse_supports,
)
from arti.component_registry import component_spec
from arti.recall_refine import RefinePolicy


def active_workspace(
    value: Tensor,
    *,
    exposed: Tensor,
    intervened: Tensor | None = None,
) -> ActiveWorkspace:
    validity = torch.ones(value.shape[:-1], dtype=torch.bool, device=value.device)
    return ActiveWorkspace(
        value,
        validity,
        exposed,
        torch.zeros_like(exposed) if intervened is None else intervened,
    )


def folded_fixture(
    x: Tensor,
    *,
    exposed: Tensor,
    intervened: Tensor,
    active_count: int = 4,
) -> tuple[arti.alpha.ReversibleTopology, FoldedWorkspace]:
    validity = torch.ones(x.shape[:-1], dtype=torch.bool, device=x.device)
    domain = SupportDomain.for_tensor(
        validity,
        domain_id="selective-world",
        owner_ref="arti/fold@2",
        partition_id="world",
        transition_id="selective-transition",
    )
    supports = PulseSupports(
        SupportMask(SupportKind.OBSERVED, validity, domain),
        SupportMask(SupportKind.EXPOSED, exposed, domain),
        SupportMask(SupportKind.INTERVENED, intervened, domain),
        validity=validity,
    )
    topology = arti.alpha.ReversibleTopology(
        active_count=active_count,
        policy=arti.alpha.FixedTopologyPolicy(order=list(range(x.shape[-2]))),
    )
    state = topology.fold(x, validity)
    return topology, FoldedWorkspace(
        state,
        fold_pulse_supports(supports, state.record),
    )


def test_formula_attention_selects_support_without_changing_values() -> None:
    value = torch.tensor([[[1.0, 0.0], [5.0, 0.0], [3.0, 0.0], [9.0, 0.0]]])
    exposed = torch.tensor([[True, True, True, False]])
    workspace = active_workspace(value, exposed=exposed)
    attention = FormulaAttention(
        MagnitudeInterventionPolicy(),
        StableTopKIntervention(max_interventions=2),
    )

    result = attention(workspace)

    assert result.value is workspace.value
    assert torch.equal(result.intervened, torch.tensor([[False, True, True, False]]))
    assert torch.equal(result.exposed, workspace.exposed)


def test_unexposed_values_cannot_influence_intervention_support() -> None:
    exposed = torch.tensor([[True, True, True, False]])
    base = torch.tensor([[[1.0], [5.0], [3.0], [7.0]]])
    changed = base.clone()
    changed[..., 3, :] = 1.0e20
    attention = FormulaAttention(
        MagnitudeInterventionPolicy(),
        StableTopKIntervention(max_interventions=2),
    )

    first = attention(active_workspace(base, exposed=exposed))
    second = attention(active_workspace(changed, exposed=exposed))

    assert torch.equal(first.intervened, second.intervened)
    assert not first.intervened[..., 3].any()


def test_factor_policy_selects_expected_exposed_instances_stably() -> None:
    value = torch.zeros(1, 5, 2)
    exposed = torch.tensor([[True, True, False, True, True]])
    factors = torch.tensor([[[0.1], [0.9], [100.0], [0.9], [0.2]]])
    attention = FormulaAttention(
        FactorInterventionPolicy(factor_index=0),
        StableTopKIntervention(max_interventions=2),
    )

    result = attention(active_workspace(value, exposed=exposed), factors)

    assert torch.equal(result.intervened, torch.tensor([[False, True, False, True, False]]))


def test_unexposed_factors_cannot_influence_intervention_support() -> None:
    value = torch.zeros(1, 4, 2)
    exposed = torch.tensor([[True, True, True, False]])
    first = torch.tensor([[[0.1], [0.9], [0.2], [0.0]]])
    second = first.clone()
    second[..., 3, :] = 1.0e20
    attention = FormulaAttention(
        FactorInterventionPolicy(0),
        StableTopKIntervention(1),
    )

    a = attention(active_workspace(value, exposed=exposed), first)
    b = attention(active_workspace(value, exposed=exposed), second)

    assert torch.equal(a.intervened, b.intervened)


def test_nonfinite_exposed_priority_fails_closed() -> None:
    value = torch.zeros(1, 3, 2)
    exposed = torch.ones(1, 3, dtype=torch.bool)
    factors = torch.tensor([[[0.1], [float("nan")], [0.3]]])
    attention = FormulaAttention(
        FactorInterventionPolicy(0),
        StableTopKIntervention(1),
    )

    with pytest.raises(ValueError, match="must be finite"):
        attention(active_workspace(value, exposed=exposed), factors)


def test_formula_attention_rejects_external_intervention_support() -> None:
    value = torch.randn(1, 4, 3)
    exposed = torch.ones(1, 4, dtype=torch.bool)
    intervened = torch.tensor([[True, False, False, False]])
    attention = FormulaAttention(
        MagnitudeInterventionPolicy(),
        StableTopKIntervention(max_interventions=1),
    )

    with pytest.raises(ValueError, match="empty incoming intervention"):
        attention(active_workspace(value, exposed=exposed, intervened=intervened))


def test_folded_workspace_rejects_external_intervention_outside_active() -> None:
    x = torch.randn(1, 6, 2)
    exposed = torch.ones(1, 6, dtype=torch.bool)
    intervened = torch.tensor([[False, False, False, False, False, True]])

    with pytest.raises(ValueError, match="intervened value in active"):
        folded_fixture(x, exposed=exposed, intervened=intervened, active_count=4)


def test_folded_workspace_commit_transports_selected_support_to_unfold() -> None:
    x = torch.randn(1, 6, 2)
    exposed = torch.tensor([[True, True, True, True, False, False]])
    topology, folded = folded_fixture(
        x,
        exposed=exposed,
        intervened=torch.zeros_like(exposed),
        active_count=4,
    )
    attention = FormulaAttention(
        MagnitudeInterventionPolicy(),
        StableTopKIntervention(max_interventions=2),
    )

    committed = folded.commit(attention(folded.active_workspace()))
    reunited = arti.alpha.unfold_pulse_supports(committed.supports)

    assert int(reunited.intervened.mask.sum()) == 2
    assert not reunited.intervened.mask[..., 4:].any()
    torch.testing.assert_close(topology.unfold(committed.state).value, x)


def test_selective_compute_changes_only_intervened_values_and_gradients() -> None:
    value = torch.randn(2, 5, 3, requires_grad=True)
    exposed = torch.ones(2, 5, dtype=torch.bool)
    intervened = torch.tensor([[True, False, True, False, False]]).expand(2, -1)
    factors = torch.zeros(2, 5, 6)
    factors[..., :3] = 0.5
    factors[..., 3:] = 2.0
    compute = SelectiveCompute(
        ScaleShiftFormula(dim=3),
        max_queries=2,
        max_sources=5,
    )

    result = compute(
        active_workspace(value, exposed=exposed, intervened=intervened),
        factors,
    )
    expected = torch.where(intervened.unsqueeze(-1), value * 1.5 + 2.0, value)
    torch.testing.assert_close(result.value, expected)
    result.value.sum().backward()
    expected_grad = torch.where(
        intervened.unsqueeze(-1),
        torch.full_like(value, 1.5),
        torch.ones_like(value),
    )
    torch.testing.assert_close(value.grad, expected_grad)


def test_support_overflow_is_rejected_before_kernel_call() -> None:
    kernel = ScaleShiftFormula(2)
    called: list[bool] = []
    hook = kernel.register_forward_pre_hook(lambda _module, _args: called.append(True))
    compute = SelectiveCompute(kernel, max_queries=1, max_sources=3)
    value = torch.randn(1, 4, 2)
    exposed = torch.tensor([[True, True, True, False]])
    intervened = torch.tensor([[True, True, False, False]])

    with pytest.raises(ValueError, match="query support exceeds"):
        compute(active_workspace(value, exposed=exposed, intervened=intervened))
    hook.remove()
    assert not called


def test_empty_intervention_support_is_exact_identity_and_zero_recall_gradient() -> None:
    recall = arti.nn.Recall(dim=2, slots=4, activation="none")
    compute = SelectiveCompute(
        arti.alpha.SelectiveRecallKernel(recall),
        max_queries=2,
        max_sources=2,
    )
    value = torch.randn(1, 4, 2, requires_grad=True)
    exposed = torch.tensor([[True, True, False, False]])
    intervened = torch.zeros_like(exposed)

    called: list[bool] = []
    hook = recall.register_forward_pre_hook(lambda _module, _args: called.append(True))
    result = compute(active_workspace(value, exposed=exposed, intervened=intervened))
    hook.remove()
    assert torch.equal(result.value, value)
    assert not called
    result.value.sum().backward()
    assert torch.equal(value.grad, torch.ones_like(value))
    for parameter in recall.parameters():
        if parameter.grad is not None:
            assert torch.equal(parameter.grad, torch.zeros_like(parameter.grad))


def test_selective_recall_matches_direct_packed_recall() -> None:
    recall = arti.nn.Recall(dim=3, slots=8, activation="none", identity_init=True)
    policy = RefinePolicy.fixed(3)
    kernel = arti.alpha.SelectiveRecallKernel(recall, refine_policy=policy)
    compute = SelectiveCompute(kernel, max_queries=2, max_sources=3)
    value = torch.randn(2, 4, 3)
    exposed = torch.tensor([[True, True, True, False]]).expand(2, -1)
    intervened = torch.tensor([[True, False, True, False]]).expand(2, -1)
    workspace = active_workspace(value, exposed=exposed, intervened=intervened)

    result, info = compute(workspace, return_info=True)
    direct_query = torch.gather(
        value,
        -2,
        info.query_index.unsqueeze(-1).expand(*info.query_index.shape, 3),
    )
    direct = recall(direct_query, mask=info.query_mask, refine_policy=policy)
    expected = value.scatter(
        -2,
        info.query_index.unsqueeze(-1).expand(*info.query_index.shape, 3),
        direct,
    )
    torch.testing.assert_close(result.value, expected)


def test_component_graph_separates_intervention_from_compute() -> None:
    attention = FormulaAttention(
        FactorInterventionPolicy(1),
        StableTopKIntervention(2),
    )
    attention_spec = component_spec(attention)
    formula_spec = component_spec(ScaleShiftFormula(4))

    assert attention_spec.reference == "arti/formula-attention@1"
    assert attention_spec.capabilities == ("pulse.stage.intervention",)
    assert set(attention_spec.dependencies) == {
        "arti/factor-intervention-policy@1",
        "arti/stable-topk-intervention@1",
    }
    assert formula_spec.capabilities == ("selective.compute.kernel",)


def test_selective_compute_rejects_kernel_without_capability() -> None:
    with pytest.raises(ValueError, match="selective.compute.kernel"):
        SelectiveCompute(nn.Identity(), max_queries=1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_selective_compute_cuda_inductor_fullgraph_forward_and_gradient() -> None:
    value = torch.randn(2, 4, 3, device="cuda", requires_grad=True)
    exposed = torch.ones(2, 4, dtype=torch.bool, device="cuda")
    intervened = torch.tensor([[True, False, True, False]], device="cuda").expand(2, -1)
    factors = torch.randn(2, 4, 6, device="cuda", requires_grad=True)
    compute = SelectiveCompute(
        ScaleShiftFormula(3),
        max_queries=2,
        max_sources=4,
    ).cuda()
    compiled = torch.compile(compute, backend="inductor", fullgraph=True)
    workspace = active_workspace(value, exposed=exposed, intervened=intervened)

    actual = compiled(workspace, factors)
    eager = compute(workspace, factors)
    torch.testing.assert_close(actual.value, eager.value)
    actual.value.square().mean().backward()

    assert value.grad is not None and torch.isfinite(value.grad).all()
    assert factors.grad is not None and torch.isfinite(factors.grad).all()


def test_selective_recall_component_binds_recall_and_refine_policy() -> None:
    kernel = arti.alpha.SelectiveRecallKernel(
        arti.nn.Recall(dim=3, slots=4, activation="none"),
        refine_policy=RefinePolicy.fixed(2),
    )
    spec = component_spec(kernel)

    assert spec.reference == "arti/selective-recall-kernel@1"
    assert spec.capabilities == ("selective.compute.kernel",)
    assert set(spec.dependencies) == {"arti/recall@2", "arti/refine-policy@1"}
