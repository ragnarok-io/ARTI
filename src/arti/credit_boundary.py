"""Forward-identity boundaries with explicit credit-gradient semantics.

This module deliberately does not reuse :mod:`arti.membrane`: that module
routes token visibility, while a credit boundary leaves data values unchanged
and only controls which downstream credit reaches an upstream computation.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

import torch
from torch import Tensor, nn


class CreditBoundaryError(ValueError):
    """Raised when an explicit credit-gradient contract is invalid."""


class CreditBoundaryMode(str, Enum):
    """Declared credit rule; ``train()`` and ``eval()`` never change it."""

    OPEN = "open"
    CLOSED = "closed"
    MEAN = "mean"
    BERNOULLI = "bernoulli"


@dataclass(frozen=True)
class CreditStructureDecision:
    """A discrete direct-versus-boundary result for one structure window."""

    use_boundary: bool
    direct_query_loss: float
    boundary_query_loss: float
    tolerance: float
    retained_declared_boundary: bool


class CreditStructureChoice(nn.Module):
    """Learn direct versus boundary structure from candidate update outcomes.

    Call :meth:`expected_query_loss` only with losses evaluated after paired
    candidate updates from the same parameter and optimizer snapshot.  The
    choice intentionally has no ordinary forward-data behavior and does not
    commit a boundary mode by thresholding its probability.
    """

    def __init__(self, initial_logit: float = 0.0, *, learnable: bool = True) -> None:
        super().__init__()
        initial = torch.tensor(float(initial_logit))
        if learnable:
            self.beta = nn.Parameter(initial)
        else:
            self.register_buffer("beta", initial)

    @property
    def boundary_probability(self) -> Tensor:
        """Continuous probability of using the boundary candidate."""

        return torch.sigmoid(self.beta)

    def expected_query_loss(self, direct: Tensor, boundary: Tensor) -> Tensor:
        """Return the paired-candidate structure objective.

        ``direct`` and ``boundary`` are scalar query losses from distinct
        candidate updates.  They cannot be two evaluations of an unchanged
        forward tensor, because that would leave this choice unidentifiable.
        """

        if not isinstance(direct, Tensor) or direct.numel() != 1:
            raise CreditBoundaryError("direct candidate loss must be a scalar Tensor")
        if not isinstance(boundary, Tensor) or boundary.numel() != 1:
            raise CreditBoundaryError("boundary candidate loss must be a scalar Tensor")
        if direct.device != boundary.device:
            raise CreditBoundaryError("candidate losses must share a device")
        probability = self.boundary_probability.to(device=direct.device, dtype=direct.dtype)
        return (1.0 - probability) * direct + probability * boundary

    def specialize(
        self,
        direct: Tensor,
        boundary: Tensor,
        *,
        tolerance: float = 1e-6,
        retain_declared_boundary_on_tie: bool = False,
    ) -> CreditStructureDecision:
        """Resolve a completed structure window from paired query outcomes.

        This is deliberately a window-boundary operation, not a forward route.
        It neither reads ``beta`` nor mutates a graph: callers use the returned
        decision when constructing their next committed ProgramGraph.  A tie
        falls back to direct connectivity unless the caller explicitly keeps a
        boundary for its declared region semantics.
        """

        if tolerance < 0.0:
            raise CreditBoundaryError("tolerance must be non-negative")
        for candidate_name, candidate_loss in (("direct", direct), ("boundary", boundary)):
            if not isinstance(candidate_loss, Tensor) or candidate_loss.numel() != 1:
                raise CreditBoundaryError(f"{candidate_name} candidate loss must be a scalar Tensor")
            if not torch.isfinite(candidate_loss.detach()).item():
                raise CreditBoundaryError(f"{candidate_name} candidate loss must be finite")
        if direct.device != boundary.device:
            raise CreditBoundaryError("candidate losses must share a device")
        direct_value = float(direct.detach().cpu())
        boundary_value = float(boundary.detach().cpu())
        tied = abs(boundary_value - direct_value) <= tolerance
        use_boundary = boundary_value < direct_value and not tied
        if tied and retain_declared_boundary_on_tie:
            use_boundary = True
        return CreditStructureDecision(
            use_boundary=use_boundary,
            direct_query_loss=direct_value,
            boundary_query_loss=boundary_value,
            tolerance=tolerance,
            retained_declared_boundary=tied and retain_declared_boundary_on_tie,
        )


@dataclass(frozen=True)
class StructureWindowProposal:
    """One pre-query proposal to spend a structure-evaluation window."""

    probability: Tensor
    sampled_open: Tensor
    eligible: bool
    forced: bool
    steps_since_window: int


class StructureWindowPolicy(nn.Module):
    """Learn when to open a future structure-evaluation window.

    ``evidence`` must be available before the candidate query targets are
    read.  The sampled action is intentionally a control-plane event: callers
    may convert it to a Python bool only between training windows, then use
    the observed post-window objective to train this policy with
    :meth:`score_function_objective`.
    """

    def __init__(
        self,
        evidence_dim: int,
        *,
        min_steps: int = 1,
        max_steps: int | None = None,
    ) -> None:
        super().__init__()
        if type(evidence_dim) is not int or evidence_dim <= 0:
            raise ValueError("evidence_dim must be a positive integer")
        if type(min_steps) is not int or min_steps < 0:
            raise ValueError("min_steps must be a non-negative integer")
        if max_steps is not None and (type(max_steps) is not int or max_steps < min_steps):
            raise ValueError("max_steps must be None or an integer no smaller than min_steps")
        self.evidence_dim = evidence_dim
        self.min_steps = min_steps
        self.max_steps = max_steps
        self.controller = nn.Linear(evidence_dim, 1)

    def propose(
        self,
        evidence: Tensor,
        *,
        steps_since_window: int,
        sample: Tensor | None = None,
    ) -> StructureWindowProposal:
        """Sample a next-window action from pre-query support evidence."""

        if not isinstance(evidence, Tensor) or evidence.ndim != 1 or evidence.shape[0] != self.evidence_dim:
            raise CreditBoundaryError("structure-window evidence must have shape [evidence_dim]")
        if type(steps_since_window) is not int or steps_since_window < 0:
            raise CreditBoundaryError("steps_since_window must be a non-negative integer")
        logit = self.controller(evidence.unsqueeze(0)).reshape(())
        probability = torch.sigmoid(logit)
        eligible = steps_since_window >= self.min_steps
        forced = self.max_steps is not None and steps_since_window >= self.max_steps
        if sample is None:
            sampled_open = torch.bernoulli(probability).to(dtype=torch.bool)
        else:
            if not isinstance(sample, Tensor) or sample.numel() != 1 or sample.dtype is not torch.bool:
                raise CreditBoundaryError("structure-window sample must be one boolean scalar Tensor")
            sampled_open = sample.to(device=probability.device).reshape(())
        if not eligible:
            sampled_open = torch.zeros((), dtype=torch.bool, device=probability.device)
        elif forced:
            sampled_open = torch.ones((), dtype=torch.bool, device=probability.device)
        return StructureWindowProposal(
            probability=probability,
            sampled_open=sampled_open,
            eligible=eligible,
            forced=forced,
            steps_since_window=steps_since_window,
        )

    def score_function_objective(
        self,
        observed_cost: Tensor,
        proposal: StructureWindowProposal,
        *,
        baseline: Tensor | float = 0.0,
    ) -> Tensor:
        """Return the REINFORCE objective for a sampled evaluation decision.

        ``observed_cost`` is normally the post-window query loss plus a
        declared evaluation cost.  It is detached so query targets train the
        window policy, not ordinary model parameters through this path.
        """

        if not isinstance(observed_cost, Tensor) or observed_cost.numel() != 1:
            raise CreditBoundaryError("structure-window observed_cost must be a scalar Tensor")
        if isinstance(baseline, Tensor):
            if baseline.numel() != 1:
                raise CreditBoundaryError("structure-window baseline must be scalar")
            baseline_value = baseline.to(device=observed_cost.device, dtype=observed_cost.dtype)
        else:
            baseline_value = torch.as_tensor(
                float(baseline), device=observed_cost.device, dtype=observed_cost.dtype
            )
        probability = proposal.probability.to(device=observed_cost.device, dtype=observed_cost.dtype)
        if not proposal.eligible or proposal.forced:
            # Ineligible and host-forced decisions are not policy samples.
            return probability * 0.0
        sample = proposal.sampled_open.to(device=observed_cost.device, dtype=observed_cost.dtype)
        log_probability = sample * torch.log(probability) + (1.0 - sample) * torch.log1p(-probability)
        return (observed_cost.detach() - baseline_value.detach()) * log_probability


class _CreditIdentity(torch.autograd.Function):
    """Identity in the data lane, scaled cotangent in the credit lane."""

    @staticmethod
    def forward(value: Tensor, credit_scale: Tensor) -> Tensor:
        return value

    @staticmethod
    def setup_context(
        ctx: object,
        inputs: tuple[Tensor, Tensor],
        output: Tensor,
    ) -> None:
        del output
        _, credit_scale = inputs
        ctx.save_for_backward(credit_scale)

    @staticmethod
    def backward(ctx: object, grad_output: Tensor) -> tuple[Tensor, None]:
        (credit_scale,) = ctx.saved_tensors
        # A closed credit edge must not turn an unrelated NaN cotangent into a
        # NaN through ``0 * NaN``.
        return torch.where(credit_scale == 0, torch.zeros_like(grad_output), grad_output * credit_scale), None


class CreditBoundary(nn.Module):
    """Preserve forward values while applying an explicit backward scale.

    ``alpha`` is intentionally not differentiated by the ordinary forward
    loss. Learn it through :func:`credit_gradient_field` and an outer
    update-after-loss objective, where the boundary's credit choice has an
    observable learning consequence.
    """

    def __init__(
        self,
        initial_logit: float = 0.0,
        *,
        learnable: bool = True,
        mode: CreditBoundaryMode | str = CreditBoundaryMode.MEAN,
    ) -> None:
        super().__init__()
        try:
            self.mode = CreditBoundaryMode(mode)
        except ValueError as error:
            raise CreditBoundaryError(f"unknown credit boundary mode: {mode!r}") from error
        initial = torch.tensor(float(initial_logit))
        if learnable:
            self.alpha = nn.Parameter(initial)
        else:
            self.register_buffer("alpha", initial)

    @property
    def permeability(self) -> Tensor:
        """Continuous mean credit permeability in ``(0, 1)``."""

        return torch.sigmoid(self.alpha)

    def contract_config(self) -> dict[str, object]:
        """Describe the declared credit rule without treating alpha as data flow."""

        return {
            "mode": self.mode.value,
            "learnable": isinstance(self.alpha, nn.Parameter),
            "alpha_logit": float(self.alpha.detach().cpu()),
        }

    def resolve_credit_scale(
        self,
        value: Tensor,
        *,
        credit_mask: Tensor | None = None,
        credit_scale: Tensor | None = None,
    ) -> Tensor:
        """Return this invocation's declared VJP scale without applying it."""

        if not isinstance(value, Tensor):
            raise TypeError("value must be a Tensor")
        if credit_scale is not None and credit_mask is not None:
            raise CreditBoundaryError("credit_scale and credit_mask are mutually exclusive")
        if credit_scale is not None:
            scale = credit_scale
        elif credit_mask is not None and self.mode is not CreditBoundaryMode.BERNOULLI:
            raise CreditBoundaryError("credit_mask is only valid for bernoulli mode")
        elif self.mode is CreditBoundaryMode.OPEN:
            scale = torch.ones((), device=value.device, dtype=value.dtype)
        elif self.mode is CreditBoundaryMode.CLOSED:
            scale = torch.zeros((), device=value.device, dtype=value.dtype)
        elif self.mode is CreditBoundaryMode.MEAN:
            scale = self.permeability
        else:
            mask = credit_mask
            if mask is None:
                mask = torch.rand((), device=value.device) < self.permeability.detach()
            if not isinstance(mask, Tensor):
                raise TypeError("credit_mask must be a Tensor")
            if mask.dtype is not torch.bool:
                raise TypeError("credit_mask must be boolean")
            try:
                torch.broadcast_shapes(tuple(value.shape), tuple(mask.shape))
            except RuntimeError as error:
                raise CreditBoundaryError("credit_mask must broadcast to value") from error
            scale = mask
        if not isinstance(scale, Tensor):
            raise TypeError("credit_scale must be a Tensor")
        if not (scale.is_floating_point() or scale.dtype is torch.bool):
            raise TypeError("credit_scale must be floating point")
        return scale.to(device=value.device, dtype=value.dtype)

    def forward(
        self,
        value: Tensor,
        *,
        credit_mask: Tensor | None = None,
        credit_scale: Tensor | None = None,
    ) -> Tensor:
        """Return ``value`` exactly while applying this call's credit rule.

        ``credit_mask`` is only meaningful for ``BERNOULLI`` and lets callers
        replay a boundary sample through checkpoint recomputation.  Supplying
        it explicitly is also how callers choose a batch/channel grouping;
        the built-in fallback samples one scalar for the whole invocation.
        """

        scale = self.resolve_credit_scale(
            value,
            credit_mask=credit_mask,
            credit_scale=credit_scale,
        )
        return _CreditIdentity.apply(value, scale)


@dataclass(frozen=True)
class CreditGradientField:
    """Explicit branch-wise VJP field used for credit-rule learning."""

    parameter_ids: tuple[str, ...]
    gradients: tuple[Tensor | None, ...]
    branch_scales: tuple[tuple[str, Tensor], ...]

    def as_dict(self) -> dict[str, Tensor | None]:
        return dict(zip(self.parameter_ids, self.gradients, strict=True))


@dataclass(frozen=True)
class PairedCreditUpdate:
    """Two functional support updates evaluated from one parameter snapshot.

    The direct and boundary candidates deliberately share neither a mutable
    parameter update nor an optimizer-state side effect.  A caller supplies a
    query evaluator which may use :func:`torch.func.functional_call` for a
    regular ``nn.Module``.  This keeps support/query separation explicit while
    avoiding a second graph execution runtime.
    """

    direct_gradients: CreditGradientField
    boundary_gradients: CreditGradientField
    direct_parameters: Mapping[str, Tensor]
    boundary_parameters: Mapping[str, Tensor]
    direct_query_loss: Tensor
    boundary_query_loss: Tensor

    def structure_objective(self, choice: CreditStructureChoice) -> Tensor:
        """Return the exact two-candidate expected query loss for ``choice``."""

        if not isinstance(choice, CreditStructureChoice):
            raise TypeError("choice must be CreditStructureChoice")
        return choice.expected_query_loss(self.direct_query_loss, self.boundary_query_loss)

    def structure_decision(
        self,
        choice: CreditStructureChoice,
        *,
        tolerance: float = 1e-6,
        retain_declared_boundary_on_tie: bool = False,
    ) -> CreditStructureDecision:
        """Resolve this paired trial at an explicit structure-window boundary."""

        if not isinstance(choice, CreditStructureChoice):
            raise TypeError("choice must be CreditStructureChoice")
        return choice.specialize(
            self.direct_query_loss,
            self.boundary_query_loss,
            tolerance=tolerance,
            retain_declared_boundary_on_tie=retain_declared_boundary_on_tie,
        )


def credit_gradient_field(
    branch_objectives: Mapping[str, Tensor],
    branch_scales: Mapping[str, Tensor],
    parameters: Mapping[str, Tensor] | Sequence[tuple[str, Tensor]],
    *,
    create_graph: bool = True,
) -> CreditGradientField:
    """Build ``sum_b scale_b * d objective_b / d theta`` explicitly.

    Each branch must have an individual objective. This is the small reference
    form of the credit-gradient operator; graph lowering can later derive the
    same branch decomposition from port-level VJPs.
    """

    if not branch_objectives:
        raise CreditBoundaryError("at least one branch objective is required")
    objective_ids = tuple(branch_objectives)
    if set(objective_ids) != set(branch_scales):
        raise CreditBoundaryError("branch objectives and branch scales must have identical keys")
    parameter_items = tuple(parameters.items()) if isinstance(parameters, Mapping) else tuple(parameters)
    if not parameter_items:
        raise CreditBoundaryError("at least one parameter is required")
    parameter_ids = tuple(parameter_id for parameter_id, _ in parameter_items)
    if len(set(parameter_ids)) != len(parameter_ids):
        raise CreditBoundaryError("parameter ids must be unique")
    parameter_values = tuple(parameter for _, parameter in parameter_items)
    if any(not isinstance(parameter, Tensor) for parameter in parameter_values):
        raise TypeError("parameters must be Tensors")

    totals: list[Tensor | None] = [None] * len(parameter_values)
    recorded_scales: list[tuple[str, Tensor]] = []
    for branch_id in objective_ids:
        objective = branch_objectives[branch_id]
        scale = branch_scales[branch_id]
        if not isinstance(objective, Tensor) or objective.numel() != 1:
            raise CreditBoundaryError(f"objective for branch {branch_id!r} must be scalar")
        if not isinstance(scale, Tensor):
            raise CreditBoundaryError(f"scale for branch {branch_id!r} must be a Tensor")
        gradients = torch.autograd.grad(
            objective,
            parameter_values,
            retain_graph=True,
            create_graph=create_graph,
            allow_unused=True,
        )
        recorded_scales.append((branch_id, scale))
        for index, gradient in enumerate(gradients):
            if gradient is None:
                continue
            scaled = scale.to(device=gradient.device, dtype=gradient.dtype)
            try:
                contribution = scaled * gradient
            except RuntimeError as error:
                raise CreditBoundaryError(
                    f"scale for branch {branch_id!r} must broadcast to gradient for parameter {parameter_ids[index]!r}"
                ) from error
            totals[index] = contribution if totals[index] is None else totals[index] + contribution
    return CreditGradientField(parameter_ids, tuple(totals), tuple(recorded_scales))


def functional_sgd(
    parameters: Mapping[str, Tensor],
    gradients: Mapping[str, Tensor | None],
    *,
    learning_rate: float | Tensor,
) -> dict[str, Tensor]:
    """Return a same-snapshot SGD candidate without mutating live parameters."""

    if set(parameters) != set(gradients):
        raise CreditBoundaryError("parameters and gradients must have identical keys")
    updated: dict[str, Tensor] = {}
    for parameter_id, parameter in parameters.items():
        gradient = gradients[parameter_id]
        if gradient is None:
            updated[parameter_id] = parameter
            continue
        updated[parameter_id] = parameter - learning_rate * gradient
    return updated


def paired_credit_update(
    support_objectives: Mapping[str, Tensor],
    *,
    direct_scales: Mapping[str, Tensor],
    boundary_scales: Mapping[str, Tensor],
    parameters: Mapping[str, Tensor],
    query_loss: Callable[[Mapping[str, Tensor]], Tensor],
    learning_rate: float | Tensor,
    create_graph: bool = True,
) -> PairedCreditUpdate:
    """Evaluate direct and boundary support updates against one query task.

    Both candidate gradients are derived from the same live parameter snapshot
    and the same support objectives.  ``query_loss`` receives a *functional*
    parameter mapping for each trial and must return one scalar query loss.
    It is intentionally the caller's responsibility to keep query data out of
    the support objectives.

    The function returns both trial mappings rather than committing either one.
    Ordinary model training can therefore retain its chosen support update,
    while :class:`CreditStructureChoice` learns from their paired query losses.
    """

    if not callable(query_loss):
        raise TypeError("query_loss must be callable")
    direct_gradients = credit_gradient_field(
        support_objectives,
        direct_scales,
        parameters,
        create_graph=create_graph,
    )
    boundary_gradients = credit_gradient_field(
        support_objectives,
        boundary_scales,
        parameters,
        create_graph=create_graph,
    )
    direct_parameters = functional_sgd(
        parameters,
        direct_gradients.as_dict(),
        learning_rate=learning_rate,
    )
    boundary_parameters = functional_sgd(
        parameters,
        boundary_gradients.as_dict(),
        learning_rate=learning_rate,
    )
    direct_query_loss = query_loss(direct_parameters)
    boundary_query_loss = query_loss(boundary_parameters)
    for candidate_name, candidate_loss in (
        ("direct", direct_query_loss),
        ("boundary", boundary_query_loss),
    ):
        if not isinstance(candidate_loss, Tensor) or candidate_loss.numel() != 1:
            raise CreditBoundaryError(f"{candidate_name} query loss must be a scalar Tensor")
    if direct_query_loss.device != boundary_query_loss.device:
        raise CreditBoundaryError("paired query losses must share a device")
    return PairedCreditUpdate(
        direct_gradients=direct_gradients,
        boundary_gradients=boundary_gradients,
        direct_parameters=direct_parameters,
        boundary_parameters=boundary_parameters,
        direct_query_loss=direct_query_loss,
        boundary_query_loss=boundary_query_loss,
    )


def bernoulli_expected_query_loss(
    closed_query_loss: Tensor,
    open_query_loss: Tensor,
    permeability: Tensor,
) -> Tensor:
    """Return the exact one-gate Bernoulli expected post-update query loss.

    ``closed_query_loss`` and ``open_query_loss`` must come from paired trials
    that differ only in one replayed Bernoulli credit mask.  This is deliberately
    distinct from evaluating a mean-VJP candidate at ``permeability``: in
    general ``E[J(z)] != J(E[z])``.
    """

    for name, value in (("closed", closed_query_loss), ("open", open_query_loss)):
        if not isinstance(value, Tensor) or value.numel() != 1:
            raise CreditBoundaryError(f"{name} query loss must be a scalar Tensor")
    if not isinstance(permeability, Tensor) or permeability.numel() != 1:
        raise CreditBoundaryError("permeability must be a scalar Tensor")
    if closed_query_loss.device != open_query_loss.device:
        raise CreditBoundaryError("paired Bernoulli query losses must share a device")
    probability = permeability.to(device=closed_query_loss.device, dtype=closed_query_loss.dtype)
    return (1.0 - probability) * closed_query_loss + probability * open_query_loss


def bernoulli_score_function_objective(
    sampled_query_loss: Tensor,
    sample: Tensor,
    permeability: Tensor,
    *,
    baseline: Tensor | float = 0.0,
) -> Tensor:
    """Return a REINFORCE objective for a replayed Bernoulli mask sample.

    The returned scalar carries a gradient only to the mask distribution.  The
    sampled query loss is detached deliberately: query data evaluates the
    learning rule but does not become an ordinary training gradient for model
    parameters through this helper.  ``baseline`` is likewise treated as a
    control variate, not as a trainable prediction target.
    """

    if not isinstance(sampled_query_loss, Tensor) or sampled_query_loss.numel() != 1:
        raise CreditBoundaryError("sampled query loss must be a scalar Tensor")
    if not isinstance(sample, Tensor) or sample.dtype is not torch.bool:
        raise TypeError("sample must be a boolean Tensor")
    if not isinstance(permeability, Tensor) or not permeability.is_floating_point():
        raise TypeError("permeability must be a floating Tensor")
    if isinstance(baseline, Tensor):
        if baseline.numel() != 1:
            raise CreditBoundaryError("baseline must be a scalar Tensor or float")
        baseline_value = baseline.to(device=sampled_query_loss.device, dtype=sampled_query_loss.dtype)
    else:
        baseline_value = torch.as_tensor(
            baseline,
            device=sampled_query_loss.device,
            dtype=sampled_query_loss.dtype,
        )
    probability = permeability.to(device=sample.device, dtype=sampled_query_loss.dtype)
    try:
        probability = probability.expand_as(sample)
    except RuntimeError as error:
        raise CreditBoundaryError("permeability must broadcast to sample") from error
    sample_value = sample.to(dtype=probability.dtype)
    log_probability = (
        sample_value * probability.log() + (1.0 - sample_value) * (1.0 - probability).log()
    ).sum()
    advantage = (sampled_query_loss - baseline_value).detach()
    return advantage * log_probability
