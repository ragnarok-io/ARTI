from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.func import functional_call

from arti import CreditBoundary, CreditBoundaryError, CreditBoundaryMode, CreditStructureChoice, StructureWindowPolicy, bernoulli_expected_query_loss, bernoulli_score_function_objective, credit_gradient_field, functional_sgd, paired_credit_update
from arti.resource_graph import Connection, DifferentiableFabricNode, MultiPortProgramNode, MultiPortProgramNodeInvocation, ResourcePort
from arti.tensor_view import TensorView


def test_credit_boundary_is_forward_identity_and_scales_only_value_gradient() -> None:
    boundary = CreditBoundary(initial_logit=0.0)
    value = torch.tensor([2.0, -3.0], requires_grad=True)

    output = boundary(value)
    assert torch.equal(output, value)

    output.sum().backward()
    assert torch.allclose(value.grad, torch.full_like(value, 0.5))
    assert boundary.alpha.grad is None


@pytest.mark.parametrize(
    ("mode", "expected"),
    ((CreditBoundaryMode.OPEN, 1.0), (CreditBoundaryMode.CLOSED, 0.0), (CreditBoundaryMode.MEAN, 0.5)),
)
def test_fixed_credit_modes_have_identity_forward_and_declared_vjp(mode: CreditBoundaryMode, expected: float) -> None:
    boundary = CreditBoundary(initial_logit=0.0, mode=mode)
    value = torch.tensor([1.0, 2.0], requires_grad=True)
    assert torch.equal(boundary(value), value)
    boundary(value).sum().backward()
    assert torch.allclose(value.grad, torch.full_like(value, expected))


def test_bernoulli_credit_mask_replays_exact_vjp_and_only_affects_its_branch() -> None:
    theta = torch.zeros(2, requires_grad=True)
    boundary = CreditBoundary(mode=CreditBoundaryMode.BERNOULLI)
    mask = torch.tensor([True, False])
    left = 0.5 * (theta - torch.tensor([-1.0, 0.0])).square().sum()
    right = 0.5 * (boundary(theta, credit_mask=mask) - torch.tensor([1.0, -1.0])).square().sum()
    right_reference = 0.5 * (theta - torch.tensor([1.0, -1.0])).square().sum()
    expected = credit_gradient_field(
        {"left": left, "right": right_reference},
        {"left": torch.tensor(1.0), "right": mask},
        {"theta": theta},
        create_graph=False,
    )
    (left + right).backward()
    assert torch.equal(theta.grad, expected.as_dict()["theta"])


def test_bernoulli_credit_mask_rejects_non_boolean_or_non_broadcastable_masks() -> None:
    boundary = CreditBoundary(mode="bernoulli")
    value = torch.zeros(2, 3)
    with pytest.raises(TypeError, match="boolean"):
        boundary(value, credit_mask=torch.ones(2, 3))
    with pytest.raises(CreditBoundaryError, match="broadcast"):
        boundary(value, credit_mask=torch.ones(4, dtype=torch.bool))
    with pytest.raises(CreditBoundaryError, match="only valid"):
        CreditBoundary(mode=CreditBoundaryMode.OPEN)(
            value,
            credit_mask=torch.ones((), dtype=torch.bool),
        )


def _outer_credit_loss(
    theta: torch.Tensor,
    permeability: torch.Tensor,
    second_target: torch.Tensor,
    query_target: torch.Tensor,
) -> torch.Tensor:
    eta = 0.1
    first = 0.5 * (theta - torch.tensor([-1.0, 0.0])).square().sum()
    second = 0.5 * (theta - second_target).square().sum()
    field = credit_gradient_field(
        {"first": first, "second": second},
        {"first": torch.ones_like(permeability), "second": permeability},
        {"theta": theta},
    )
    candidate = functional_sgd({"theta": theta}, field.as_dict(), learning_rate=eta)["theta"]
    return 0.5 * (candidate - query_target).square().sum()


def test_conflicting_branch_credit_learns_to_close_with_analytic_outer_gradient() -> None:
    theta = torch.zeros(2, requires_grad=True)
    boundary = CreditBoundary(initial_logit=0.0)
    outer_loss = _outer_credit_loss(
        theta,
        boundary.permeability,
        torch.tensor([1.0, -1.0]),
        torch.tensor([-0.1, 0.0]),
    )
    outer_loss.backward()

    assert boundary.alpha.grad is not None
    expected = 2.0 * 0.1**2 * 0.5**2 * 0.5
    assert boundary.alpha.grad.item() == pytest.approx(expected)


def test_cooperative_branch_credit_learns_to_open_with_analytic_outer_gradient() -> None:
    theta = torch.zeros(2, requires_grad=True)
    boundary = CreditBoundary(initial_logit=0.0)
    outer_loss = _outer_credit_loss(
        theta,
        boundary.permeability,
        torch.tensor([0.0, -1.0]),
        torch.tensor([-0.1, -0.1]),
    )
    outer_loss.backward()

    assert boundary.alpha.grad is not None
    expected = -(0.1**2) * 0.5 * 0.5**2
    assert boundary.alpha.grad.item() == pytest.approx(expected)


def test_credit_outer_gradient_matches_centered_difference() -> None:
    alpha = 0.0
    epsilon = 1e-4

    def evaluate(logit: float) -> float:
        theta = torch.zeros(2, requires_grad=True)
        permeability = torch.sigmoid(torch.tensor(logit))
        return _outer_credit_loss(
            theta,
            permeability,
            torch.tensor([1.0, -1.0]),
            torch.tensor([-0.1, 0.0]),
        ).item()

    finite_difference = (evaluate(alpha + epsilon) - evaluate(alpha - epsilon)) / (2.0 * epsilon)
    theta = torch.zeros(2, requires_grad=True)
    learned_alpha = torch.tensor(alpha, requires_grad=True)
    _outer_credit_loss(
        theta,
        torch.sigmoid(learned_alpha),
        torch.tensor([1.0, -1.0]),
        torch.tensor([-0.1, 0.0]),
    ).backward()
    assert learned_alpha.grad is not None
    assert learned_alpha.grad.item() == pytest.approx(finite_difference, abs=2e-5)


def test_nonlinear_credit_outer_gradient_matches_centered_difference_without_double_gate() -> None:
    def evaluate(alpha: torch.Tensor) -> torch.Tensor:
        theta = torch.tensor(0.3, requires_grad=True)
        support = {
            "stable": 0.5 * (torch.tanh(theta) + 0.7).square(),
            "conflicting": 0.5 * (torch.tanh(theta) - 0.9).square(),
        }
        field = credit_gradient_field(
            support,
            {"stable": torch.ones(()), "conflicting": torch.sigmoid(alpha)},
            {"theta": theta},
        )
        candidate = functional_sgd({"theta": theta}, field.as_dict(), learning_rate=0.15)["theta"]
        return 0.5 * (torch.tanh(candidate) + 0.45).square()

    alpha = torch.tensor(-0.2, requires_grad=True)
    evaluate(alpha).backward()
    assert alpha.grad is not None
    epsilon = 1e-3
    finite_difference = (
        evaluate(torch.tensor(alpha.item() + epsilon)).item()
        - evaluate(torch.tensor(alpha.item() - epsilon)).item()
    ) / (2.0 * epsilon)
    assert alpha.grad.item() == pytest.approx(finite_difference, abs=2e-5)


def test_credit_field_matches_analytic_branch_vjp_sum() -> None:
    theta = torch.tensor([1.0, -2.0], requires_grad=True)
    left_scale = torch.tensor(0.25, requires_grad=True)
    right_scale = torch.tensor(0.75, requires_grad=True)
    left = theta.square().sum()
    right = (theta + 2.0).square().sum()

    field = credit_gradient_field(
        {"left": left, "right": right},
        {"left": left_scale, "right": right_scale},
        {"theta": theta},
    )
    expected = left_scale * 2.0 * theta + right_scale * 2.0 * (theta + 2.0)
    assert torch.allclose(field.as_dict()["theta"], expected)


def test_functional_sgd_does_not_mutate_live_parameter() -> None:
    parameter = torch.tensor(2.0, requires_grad=True)
    updated = functional_sgd({"weight": parameter}, {"weight": torch.tensor(3.0)}, learning_rate=0.1)
    assert parameter.item() == pytest.approx(2.0)
    assert updated["weight"].item() == pytest.approx(1.7)


def test_connection_boundary_is_forward_identity_and_only_closes_its_declared_port() -> None:
    source = torch.tensor([[2.0, -3.0]], requires_grad=True)
    view = TensorView.from_tensor(
        source,
        axis_names=("batch", "feature"),
        axis_roles=("batch", "feature"),
    )
    closed = Connection(
        "closed_port",
        ResourcePort("source"),
        ResourcePort("closed"),
        credit_boundary=CreditBoundary(mode=CreditBoundaryMode.CLOSED),
    )
    open_connection = Connection(
        "open_port",
        ResourcePort("source"),
        ResourcePort("open"),
    )

    closed_value = closed(view).value
    open_value = open_connection(view).value
    assert torch.equal(closed_value, source)
    (closed_value.square().sum() + open_value.sum()).backward()

    assert torch.equal(source.grad, torch.ones_like(source))
    assert closed.contract_config()["credit_boundary"] == {
        "mode": "closed",
        "learnable": True,
        "alpha_logit": 0.0,
    }


def test_structure_choice_uses_paired_candidate_update_losses_for_beta_gradient() -> None:
    choice = CreditStructureChoice(initial_logit=0.0)
    direct = torch.tensor(2.0)
    boundary = torch.tensor(1.0)

    objective = choice.expected_query_loss(direct, boundary)
    objective.backward()

    assert objective.item() == pytest.approx(1.5)
    assert choice.beta.grad is not None
    assert choice.beta.grad.item() == pytest.approx(-0.25)


def test_structure_choice_rejects_non_scalar_candidate_losses() -> None:
    choice = CreditStructureChoice()
    with pytest.raises(CreditBoundaryError, match="scalar"):
        choice.expected_query_loss(torch.ones(2), torch.ones(()))


class _TwoHeadTask(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.theta = nn.Parameter(torch.zeros(2))

    def forward(self, batch_size: int) -> torch.Tensor:
        return self.theta.expand(batch_size, -1)


def test_paired_credit_update_uses_same_snapshot_and_real_functional_query_task() -> None:
    model = _TwoHeadTask()
    parameters = dict(model.named_parameters())
    support_output = model(batch_size=1).squeeze(0)
    support = {
        "stable": 0.5 * (support_output - torch.tensor([-1.0, 0.0])).square().sum(),
        "conflicting": 0.5 * (support_output - torch.tensor([1.0, -1.0])).square().sum(),
    }
    boundary = CreditBoundary(initial_logit=0.0)
    choice = CreditStructureChoice(initial_logit=0.0)

    def query_loss(candidate_parameters: dict[str, torch.Tensor]) -> torch.Tensor:
        output = functional_call(model, candidate_parameters, (1,)).squeeze(0)
        return 0.5 * (output - torch.tensor([-0.1, 0.0])).square().sum()

    paired = paired_credit_update(
        support,
        direct_scales={name: torch.ones(()) for name in support},
        boundary_scales={
            "stable": torch.ones(()),
            "conflicting": boundary.permeability,
        },
        parameters=parameters,
        query_loss=query_loss,
        learning_rate=0.1,
    )

    # Neither candidate is an in-place support update of the live model.
    assert torch.equal(model.theta, torch.zeros(2))
    assert torch.allclose(paired.direct_parameters["theta"], torch.tensor([0.0, -0.1]))
    assert torch.allclose(paired.boundary_parameters["theta"], torch.tensor([-0.05, -0.05]))
    assert paired.boundary_query_loss < paired.direct_query_loss

    paired.structure_objective(choice).backward()
    assert boundary.alpha.grad is not None
    assert choice.beta.grad is not None
    assert boundary.alpha.grad.item() > 0.0
    assert choice.beta.grad.item() < 0.0


def test_credit_boundary_paired_update_uses_the_common_hard_fate_consequence_interface() -> None:
    class _Candidate(MultiPortProgramNode):
        def __init__(self, node_id: str) -> None:
            super().__init__(
                node_id,
                input_ports={"value": ResourcePort("source")},
                output_ports={"value": ResourcePort("output")},
            )

        def invoke_ports(self, inputs: dict[str, TensorView]) -> MultiPortProgramNodeInvocation:
            return MultiPortProgramNodeInvocation({"value": inputs["value"]})

    model = _TwoHeadTask()
    boundary = CreditBoundary(initial_logit=0.0)
    choice = CreditStructureChoice(initial_logit=0.0)
    support = model(batch_size=1).squeeze(0)
    paired = paired_credit_update(
        {
            "stable": 0.5 * (support - torch.tensor([-1.0, 0.0])).square().sum(),
            "conflicting": 0.5 * (support - torch.tensor([1.0, -1.0])).square().sum(),
        },
        direct_scales={"stable": torch.ones(()), "conflicting": torch.ones(())},
        boundary_scales={"stable": torch.ones(()), "conflicting": boundary.permeability},
        parameters=dict(model.named_parameters()),
        query_loss=lambda trial: 0.5
        * (functional_call(model, trial, (1,)).squeeze(0) - torch.tensor([-0.1, 0.0])).square().sum(),
        learning_rate=0.1,
    )
    node = DifferentiableFabricNode(
        "credit_choice", {"direct": _Candidate("direct"), "membrane": _Candidate("membrane")}
    )
    objective = node.paired_credit_structure_objective(
        {"direct": paired.direct_query_loss, "membrane": paired.boundary_query_loss},
        choice=choice,
        direct_candidate_id="direct",
        boundary_candidate_id="membrane",
    )
    objective.backward()
    assert boundary.alpha.grad is not None
    assert choice.beta.grad is not None
    assert choice.beta.grad.item() < 0.0


def test_paired_credit_update_rejects_non_scalar_query_loss() -> None:
    theta = torch.tensor(0.0, requires_grad=True)
    support = {"branch": theta.square()}
    with pytest.raises(CreditBoundaryError, match="scalar"):
        paired_credit_update(
            support,
            direct_scales={"branch": torch.ones(())},
            boundary_scales={"branch": torch.ones(())},
            parameters={"theta": theta},
            query_loss=lambda _: torch.ones(2),
            learning_rate=0.1,
        )


def test_structure_specialization_uses_paired_query_outcomes_not_beta_threshold() -> None:
    choice = CreditStructureChoice(initial_logit=-20.0)
    decision = choice.specialize(torch.tensor(3.0), torch.tensor(2.0))

    # A strongly direct-biased training logit cannot override the actual
    # post-update evaluation at the explicit structure window boundary.
    assert decision.use_boundary
    assert not decision.retained_declared_boundary


def test_structure_specialization_defaults_ties_to_direct_unless_declared() -> None:
    choice = CreditStructureChoice()
    direct = torch.tensor(1.0)
    boundary = torch.tensor(1.0 + 5e-7)

    default = choice.specialize(direct, boundary, tolerance=1e-6)
    retained = choice.specialize(
        direct,
        boundary,
        tolerance=1e-6,
        retain_declared_boundary_on_tie=True,
    )

    assert not default.use_boundary
    assert retained.use_boundary
    assert retained.retained_declared_boundary


def test_paired_credit_rule_learns_credit_and_structure_from_query_after_support_update() -> None:
    model = _TwoHeadTask()
    boundary = CreditBoundary(initial_logit=0.0)
    choice = CreditStructureChoice(initial_logit=0.0)
    optimizer = torch.optim.SGD((boundary.alpha, choice.beta), lr=2.0)

    initial_loss: float | None = None
    final_loss: float | None = None
    for _ in range(12):
        optimizer.zero_grad()
        support_output = model(batch_size=1).squeeze(0)
        paired = paired_credit_update(
            {
                "stable": 0.5 * (support_output - torch.tensor([-1.0, 0.0])).square().sum(),
                "conflicting": 0.5 * (support_output - torch.tensor([1.0, -1.0])).square().sum(),
            },
            direct_scales={"stable": torch.ones(()), "conflicting": torch.ones(())},
            boundary_scales={
                "stable": torch.ones(()),
                "conflicting": boundary.permeability,
            },
            parameters=dict(model.named_parameters()),
            query_loss=lambda trial: 0.5
            * (
                functional_call(model, trial, (1,)).squeeze(0)
                - torch.tensor([-0.1, 0.0])
            )
            .square()
            .sum(),
            learning_rate=0.1,
        )
        initial_loss = (
            paired.boundary_query_loss.item() if initial_loss is None else initial_loss
        )
        outer_loss = paired.structure_objective(choice)
        outer_loss.backward()
        optimizer.step()
        final_loss = paired.boundary_query_loss.item()

    assert initial_loss is not None and final_loss is not None
    assert final_loss < initial_loss
    assert boundary.permeability.item() < 0.5
    assert choice.boundary_probability.item() > 0.5


def test_bernoulli_expected_query_loss_enumerates_paired_credit_trials() -> None:
    model = _TwoHeadTask()
    boundary = CreditBoundary(initial_logit=0.0, mode=CreditBoundaryMode.BERNOULLI)
    support_output = model(batch_size=1).squeeze(0)
    support = {
        "stable": 0.5 * (support_output - torch.tensor([-1.0, 0.0])).square().sum(),
        "conflicting": 0.5 * (support_output - torch.tensor([1.0, -1.0])).square().sum(),
    }

    def query_loss(candidate_parameters: dict[str, torch.Tensor]) -> torch.Tensor:
        output = functional_call(model, candidate_parameters, (1,)).squeeze(0)
        return 0.5 * (output - torch.tensor([-0.1, 0.0])).square().sum()

    closed = paired_credit_update(
        support,
        direct_scales={name: torch.ones(()) for name in support},
        boundary_scales={"stable": torch.ones(()), "conflicting": torch.zeros(())},
        parameters=dict(model.named_parameters()),
        query_loss=query_loss,
        learning_rate=0.1,
    ).boundary_query_loss
    open_ = paired_credit_update(
        support,
        direct_scales={name: torch.ones(()) for name in support},
        boundary_scales={"stable": torch.ones(()), "conflicting": torch.ones(())},
        parameters=dict(model.named_parameters()),
        query_loss=query_loss,
        learning_rate=0.1,
    ).boundary_query_loss
    objective = bernoulli_expected_query_loss(closed, open_, boundary.permeability)

    objective.backward()

    assert closed.item() == pytest.approx(0.0)
    assert open_.item() == pytest.approx(0.01)
    assert objective.item() == pytest.approx(0.005)
    assert boundary.alpha.grad is not None
    assert boundary.alpha.grad.item() == pytest.approx(0.0025)


def test_bernoulli_score_function_estimates_distribution_gradient_without_query_model_gradient() -> None:
    alpha = torch.tensor(0.0, requires_grad=True)
    permeability = torch.sigmoid(alpha)
    closed_loss = torch.tensor(0.0, requires_grad=True)
    open_loss = torch.tensor(0.01, requires_grad=True)
    closed_objective = bernoulli_score_function_objective(
        closed_loss,
        torch.tensor(False),
        permeability,
    )
    open_objective = bernoulli_score_function_objective(
        open_loss,
        torch.tensor(True),
        permeability,
    )

    # At p=0.5, averaging the two possible sampled estimators recovers the
    # exact one-gate gradient p(1-p)(J_open - J_closed).
    (0.5 * (closed_objective + open_objective)).backward()

    assert alpha.grad is not None
    assert alpha.grad.item() == pytest.approx(0.0025)
    assert closed_loss.grad is None
    assert open_loss.grad is None


def test_structure_window_policy_learns_only_from_a_pre_query_sampled_action() -> None:
    policy = StructureWindowPolicy(2, min_steps=2)
    with torch.no_grad():
        policy.controller.weight.zero_()
        policy.controller.bias.zero_()
    proposal = policy.propose(
        torch.tensor([3.0, -2.0]),
        steps_since_window=2,
        sample=torch.tensor(True),
    )
    query_cost = torch.tensor(2.0, requires_grad=True)
    objective = policy.score_function_objective(query_cost, proposal)
    objective.backward()

    assert proposal.eligible and not proposal.forced
    assert proposal.sampled_open.item() is True
    assert policy.controller.bias.grad is not None
    assert policy.controller.bias.grad.item() == pytest.approx(1.0)
    assert query_cost.grad is None


def test_structure_window_policy_does_not_train_for_ineligible_or_forced_actions() -> None:
    policy = StructureWindowPolicy(1, min_steps=2, max_steps=4)
    ineligible = policy.propose(torch.ones(1), steps_since_window=1, sample=torch.tensor(True))
    forced = policy.propose(torch.ones(1), steps_since_window=4, sample=torch.tensor(False))

    assert not ineligible.eligible and not ineligible.sampled_open.item()
    assert forced.eligible and forced.forced and forced.sampled_open.item()
    (policy.score_function_objective(torch.tensor(1.0), ineligible) + policy.score_function_objective(
        torch.tensor(1.0), forced
    )).backward()
    assert policy.controller.weight.grad is not None
    torch.testing.assert_close(policy.controller.weight.grad, torch.zeros_like(policy.controller.weight.grad))


def test_credit_field_rejects_incomplete_branch_contract() -> None:
    theta = torch.tensor(1.0, requires_grad=True)
    with pytest.raises(CreditBoundaryError, match="identical keys"):
        credit_gradient_field({"a": theta.square()}, {"b": torch.tensor(1.0)}, {"theta": theta})
