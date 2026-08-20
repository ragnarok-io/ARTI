"""Target-Bank-addressable forward memory updates."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, ClassVar, Literal, overload

import torch
from torch import Tensor, nn

from .functional import masked_mean, masked_softmax
from .recall_formula import RecallFormulaContract, validate_formula
from .recall_refine import RefineBudget, RefineStop


@dataclass(frozen=True)
class WriteRefinePolicy:
    """Bounded integration policy for one target-Bank write exposure."""

    _component_reference: ClassVar[str] = "arti/write-refine-policy@1"

    budget: RefineBudget = field(
        default_factory=lambda: RefineBudget(max_steps=1, min_steps=1)
    )
    stop: RefineStop | None = None
    exposure_schedule: str = "normalized"

    def __post_init__(self) -> None:
        if not isinstance(self.budget, RefineBudget):
            raise TypeError("budget must be a RefineBudget")
        if self.budget.max_steps <= 0 or self.budget.min_steps <= 0:
            raise ValueError("write-refine budgets must contain at least one step")
        if self.stop is not None and not isinstance(self.stop, RefineStop):
            raise TypeError("stop must be a RefineStop or None")
        if self.stop is not None and self.stop.scope != "sample":
            raise ValueError("write refinement currently supports sample-scoped stopping")
        if self.exposure_schedule != "normalized":
            raise ValueError("exposure_schedule must be 'normalized'")

    @classmethod
    def fixed(cls, steps: int) -> "WriteRefinePolicy":
        return cls(budget=RefineBudget(max_steps=steps, min_steps=steps))

    @classmethod
    def adaptive(
        cls,
        *,
        max_steps: int,
        min_steps: int = 1,
        absolute_tolerance: float = 0.0,
        relative_tolerance: float = 1e-4,
        patience: int = 1,
    ) -> "WriteRefinePolicy":
        return cls(
            budget=RefineBudget(max_steps=max_steps, min_steps=min_steps),
            stop=RefineStop(
                scope="sample",
                absolute_tolerance=absolute_tolerance,
                relative_tolerance=relative_tolerance,
                patience=patience,
            ),
        )

    def replace(self, **changes: Any) -> "WriteRefinePolicy":
        return replace(self, **changes)


def _fixed_query_weight(dim: int, seed: int) -> Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    weight = torch.randn(dim, dim, generator=generator)
    q, r = torch.linalg.qr(weight)
    signs = torch.sign(torch.diagonal(r)).clamp_min(0).mul(2).sub(1)
    return q * signs.unsqueeze(0)


class TargetBankUpdater(nn.Module):
    """Refine and update a Bank that is itself addressable Recall memory.

    Every step independently reads a private memory partition, when configured,
    and the current target Bank. The partitions normalize internally and are
    combined by an explicit gate. Only the target Bank is updated; the private
    memory remains an ordinary module parameter.
    """

    def __init__(
        self,
        hidden_dim: int,
        slots: int,
        *,
        workspace_dim: int | None = None,
        private_slots: int = 0,
        formula: nn.Module | None = None,
        policy: WriteRefinePolicy | None = None,
        query_seed: int = 0,
        target_coupling: Literal["optional", "required_after_bootstrap"] = "optional",
        epsilon: float = 1e-8,
    ) -> None:
        super().__init__()
        if hidden_dim <= 0 or slots <= 0:
            raise ValueError("hidden_dim and slots must be positive")
        resolved_workspace = hidden_dim if workspace_dim is None else workspace_dim
        if resolved_workspace <= 0:
            raise ValueError("workspace_dim must be positive")
        if private_slots < 0:
            raise ValueError("private_slots must be non-negative")
        if isinstance(query_seed, bool) or not isinstance(query_seed, int):
            raise TypeError("query_seed must be an integer")
        if not 0 <= query_seed < 2**63:
            raise ValueError("query_seed must be in [0, 2**63)")
        if target_coupling not in {"optional", "required_after_bootstrap"}:
            raise ValueError(
                "target_coupling must be 'optional' or 'required_after_bootstrap'"
            )
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        if policy is not None and not isinstance(policy, WriteRefinePolicy):
            raise TypeError("policy must be a WriteRefinePolicy or None")

        self.hidden_dim = int(hidden_dim)
        self.slots = int(slots)
        self.workspace_dim = int(resolved_workspace)
        self.private_slots = int(private_slots)
        self.query_seed = int(query_seed)
        self.target_coupling = target_coupling
        self.policy = policy or WriteRefinePolicy()
        self.epsilon = float(epsilon)

        self.trace_projection = nn.Linear(hidden_dim, self.workspace_dim)
        self.target_projection = nn.Linear(hidden_dim, self.workspace_dim, bias=False)
        self.slot_identity = nn.Parameter(torch.empty(slots, self.workspace_dim))
        if private_slots:
            self.private_bank = nn.Parameter(
                torch.empty(private_slots, self.workspace_dim)
            )
            self.partition_gate = nn.Linear(self.workspace_dim * 3, 2)
        else:
            self.register_parameter("private_bank", None)
            self.partition_gate = None
        self.register_buffer(
            "query_weight",
            _fixed_query_weight(self.workspace_dim, query_seed),
        )
        self.state_update = nn.Sequential(
            nn.Linear(self.workspace_dim * 3, self.workspace_dim),
            nn.SiLU(),
            nn.Linear(self.workspace_dim, self.workspace_dim),
        )
        self.state_norm = nn.LayerNorm(self.workspace_dim)
        if target_coupling == "required_after_bootstrap":
            self.coupling_gate = nn.Linear(self.workspace_dim * 2, self.workspace_dim)
        else:
            self.coupling_gate = None

        self.formula = formula
        self.formula_contract: RecallFormulaContract | None = None
        if formula is None:
            self.gain_head = nn.Linear(self.workspace_dim, hidden_dim)
            self.shift_head = nn.Linear(self.workspace_dim, hidden_dim)
            self.factor_head = None
        else:
            probe = torch.zeros(hidden_dim)
            self.formula_contract = validate_formula(formula, probe)
            factor_count = self.formula_contract.factor_count
            self.factor_head = nn.Linear(
                self.workspace_dim,
                factor_count * hidden_dim,
            )
            self.gain_head = None
            self.shift_head = None
        self.reset_parameters()

    @property
    def _component_reference(self) -> str:
        if self.target_coupling == "required_after_bootstrap":
            return "arti/target-bank-updater@2"
        return "arti/target-bank-updater@1"

    @property
    def transition_semantics(self) -> str:
        if self.target_coupling == "required_after_bootstrap":
            return "target_coupled_locally_affine_after_bootstrap"
        return "target_conditioned_locally_affine"

    def reset_parameters(self) -> None:
        nn.init.normal_(self.slot_identity, std=self.workspace_dim**-0.5)
        if self.private_bank is not None:
            nn.init.normal_(self.private_bank, std=self.workspace_dim**-0.5)
        if self.gain_head is not None and self.shift_head is not None:
            nn.init.zeros_(self.gain_head.weight)
            nn.init.zeros_(self.gain_head.bias)
            nn.init.zeros_(self.shift_head.weight)
            nn.init.zeros_(self.shift_head.bias)
        if self.factor_head is not None:
            nn.init.zeros_(self.factor_head.weight)
            assert self.formula_contract is not None
            identities = torch.tensor(
                [factor.identity for factor in self.formula_contract.factors]
            ).repeat_interleave(self.hidden_dim)
            with torch.no_grad():
                self.factor_head.bias.copy_(identities)

    @overload
    def forward(
        self,
        trace: Tensor,
        target_bank: Tensor,
        *,
        trace_mask: Tensor | None = None,
        target_mask: Tensor | None = None,
        write_mask: Tensor | None = None,
        exposure: float | Tensor = 1.0,
        policy: WriteRefinePolicy | None = None,
        return_info: Literal[False] = False,
    ) -> Tensor: ...

    @overload
    def forward(
        self,
        trace: Tensor,
        target_bank: Tensor,
        *,
        trace_mask: Tensor | None = None,
        target_mask: Tensor | None = None,
        write_mask: Tensor | None = None,
        exposure: float | Tensor = 1.0,
        policy: WriteRefinePolicy | None = None,
        return_info: Literal[True],
    ) -> tuple[Tensor, dict[str, Tensor]]: ...

    def forward(
        self,
        trace: Tensor,
        target_bank: Tensor,
        *,
        trace_mask: Tensor | None = None,
        target_mask: Tensor | None = None,
        write_mask: Tensor | None = None,
        exposure: float | Tensor = 1.0,
        policy: WriteRefinePolicy | None = None,
        return_info: bool = False,
        _addressable_target: Tensor | None = None,
        _reuse_first_target_route: bool = False,
    ) -> Tensor | tuple[Tensor, dict[str, Tensor]]:
        selected = self.policy if policy is None else policy
        if not isinstance(selected, WriteRefinePolicy):
            raise TypeError("policy must be a WriteRefinePolicy")
        trace_b, target_b, trace_mask_b, target_mask_b, squeeze = self._normalize_inputs(
            trace,
            target_bank,
            trace_mask,
            target_mask,
        )
        write_mask_b = self._mask(
            write_mask,
            target_bank.shape[:-1],
            target_bank.device,
            squeeze,
            "write_mask",
        )
        if write_mask_b is None:
            write_mask_b = target_mask_b
        if _addressable_target is not None:
            addressable_b = _addressable_target.unsqueeze(0) if squeeze else _addressable_target
            if addressable_b.shape != target_b.shape:
                raise ValueError("_addressable_target must match target_bank shape")
            if addressable_b.device != target_b.device or addressable_b.dtype != target_b.dtype:
                raise ValueError("_addressable_target must match target_bank device and dtype")
        else:
            addressable_b = None

        exposure_b = self._normalize_exposure(
            exposure,
            batch_size=target_b.shape[0],
            reference=target_b,
        )
        event = masked_mean(trace_b, trace_mask_b, dim=1)
        event_state = self.trace_projection(event)
        current = target_b
        state = self.state_norm(
            event_state.unsqueeze(1) + self.slot_identity.to(current).unsqueeze(0)
        )
        active = exposure_b > 0
        stable_count = torch.zeros_like(exposure_b, dtype=torch.long)
        executed = torch.zeros_like(stable_count)
        applied = torch.zeros_like(exposure_b)
        max_steps = selected.budget.max_steps
        min_steps = selected.budget.min_steps
        step_exposure = exposure_b / max_steps

        active_history: list[Tensor] = []
        target_weight_history: list[Tensor] = []
        target_index_history: list[Tensor] = []
        private_weight_history: list[Tensor] = []
        partition_gate_history: list[Tensor] = []
        target_read_history: list[Tensor] = []
        change_history: list[Tensor] = []
        state_history: list[Tensor] = []
        first_target_weights: Tensor | None = None

        for step in range(max_steps):
            active_before = active
            if not bool(active_before.any()):
                break
            memory_target = current if addressable_b is None else addressable_b
            query = torch.nn.functional.linear(state, self.query_weight.to(state))
            target_values = self.target_projection(memory_target)
            target_keys = target_values + self.slot_identity.to(target_values).unsqueeze(0)
            target_logits = torch.einsum("bsw,btw->bst", query, target_keys)
            target_logits = target_logits * self.workspace_dim**-0.5
            route_mask = target_mask_b.unsqueeze(1).expand_as(target_logits)
            if _reuse_first_target_route and first_target_weights is not None:
                target_weights = first_target_weights
            else:
                target_weights = masked_softmax(target_logits, route_mask, dim=-1)
                if _reuse_first_target_route:
                    first_target_weights = target_weights.detach()
            target_context = torch.einsum(
                "bst,btw->bsw", target_weights, target_values
            )
            if self.private_bank is None:
                private_weights = state.new_empty(
                    state.shape[0], state.shape[1], 0
                )
                gates = torch.stack(
                    (torch.zeros_like(target_weights[..., 0]), torch.ones_like(target_weights[..., 0])),
                    dim=-1,
                )
                context = target_context
            else:
                private = self.private_bank.to(state)
                private_logits = torch.einsum("bsw,pw->bsp", query, private)
                private_logits = private_logits * self.workspace_dim**-0.5
                private_weights = torch.softmax(private_logits, dim=-1)
                private_context = torch.einsum(
                    "bsp,pw->bsw", private_weights, private
                )
                assert self.partition_gate is not None
                gate_input = torch.cat(
                    (state, event_state.unsqueeze(1).expand_as(state), target_context),
                    dim=-1,
                )
                gates = torch.softmax(self.partition_gate(gate_input), dim=-1)
                context = (
                    gates[..., :1] * private_context
                    + gates[..., 1:] * target_context
                )

            update_input = torch.cat(
                (state, event_state.unsqueeze(1).expand_as(state), context),
                dim=-1,
            )
            proposed_state = self.state_norm(state + self.state_update(update_input))
            if self.target_coupling == "required_after_bootstrap" and step > 0:
                assert self.coupling_gate is not None
                event_slots = event_state.unsqueeze(1).expand_as(proposed_state)
                coupling = target_context * torch.sigmoid(
                    self.coupling_gate(torch.cat((event_slots, proposed_state), dim=-1))
                )
                proposed_target = self._apply_formula(
                    current,
                    coupling,
                    identity_at_zero=True,
                )
            else:
                proposed_target = self._apply_formula(current, proposed_state)
            raw_change = proposed_target - current
            scaled_change = raw_change * step_exposure.reshape(-1, 1, 1)
            scaled_change = scaled_change * write_mask_b.unsqueeze(-1).to(
                scaled_change.dtype
            )
            active_view = active_before.reshape(-1, 1, 1)
            scaled_change = scaled_change * active_view.to(scaled_change.dtype)
            next_target = torch.where(active_view, current + scaled_change, current)
            next_state = torch.where(active_view, proposed_state, state)

            change_norm = torch.linalg.vector_norm(
                scaled_change.float().reshape(target_b.shape[0], -1), dim=-1
            )
            target_norm = torch.linalg.vector_norm(
                next_target.float().reshape(target_b.shape[0], -1), dim=-1
            )
            relative = change_norm / target_norm.clamp_min(self.epsilon)
            executed = executed + active_before.to(executed.dtype)
            applied = applied + active_before.to(applied.dtype) * step_exposure
            active_history.append(active_before)
            target_weight_history.append(target_weights)
            target_index_history.append(target_weights.argmax(dim=-1))
            private_weight_history.append(private_weights)
            partition_gate_history.append(gates)
            target_read_history.append(target_context)
            change_history.append(change_norm)
            state_history.append(target_norm)
            current = next_target
            state = next_state

            eligible = active_before & (step + 1 >= min_steps)
            if selected.stop is None:
                stable = torch.zeros_like(eligible)
            else:
                stable = eligible
                if selected.stop.absolute_tolerance > 0:
                    stable = stable & (change_norm <= selected.stop.absolute_tolerance)
                if selected.stop.relative_tolerance > 0:
                    stable = stable & (relative <= selected.stop.relative_tolerance)
            stable_count = torch.where(
                stable, stable_count + 1, torch.zeros_like(stable_count)
            )
            patience = 1 if selected.stop is None else selected.stop.patience
            active = active_before & (stable_count < patience)

        output = current.squeeze(0) if squeeze else current
        if not return_info:
            return output
        batch = target_b.shape[0]
        width = len(active_history)
        empty_float = target_b.new_empty((batch, 0), dtype=torch.float32)
        empty_bool = torch.empty(batch, 0, dtype=torch.bool, device=target_b.device)
        empty_target_weights = target_b.new_empty((batch, 0, self.slots, self.slots))
        empty_private_weights = target_b.new_empty(
            (batch, 0, self.slots, self.private_slots)
        )
        empty_indices = torch.empty(
            batch, 0, self.slots, dtype=torch.long, device=target_b.device
        )
        empty_gate = target_b.new_empty((batch, 0, self.slots, 2))
        empty_context = target_b.new_empty(
            (batch, 0, self.slots, self.workspace_dim)
        )
        info = {
            "requested_exposure": exposure_b.detach(),
            "applied_exposure": applied.detach(),
            "executed_write_steps": executed.detach(),
            "active_steps": torch.stack(active_history, dim=1).detach() if width else empty_bool,
            "private_route_weights": (
                torch.stack(private_weight_history, dim=1).detach()
                if width
                else empty_private_weights
            ),
            "target_route_weights": (
                torch.stack(target_weight_history, dim=1).detach()
                if width
                else empty_target_weights
            ),
            "target_route_indices": (
                torch.stack(target_index_history, dim=1).detach()
                if width
                else empty_indices
            ),
            "partition_gate": (
                torch.stack(partition_gate_history, dim=1).detach()
                if width
                else empty_gate
            ),
            "target_read_context": (
                torch.stack(target_read_history, dim=1).detach()
                if width
                else empty_context
            ),
            "target_change_norm": (
                torch.stack(change_history, dim=1).detach() if width else empty_float
            ),
            "target_state_norm": (
                torch.stack(state_history, dim=1).detach() if width else empty_float
            ),
            "stopped": (~active).detach(),
        }
        return output, info

    def _apply_formula(
        self,
        target: Tensor,
        state: Tensor,
        *,
        identity_at_zero: bool = False,
    ) -> Tensor:
        if self.formula is None:
            assert self.gain_head is not None and self.shift_head is not None
            gain_raw = self.gain_head(state)
            shift = self.shift_head(state)
            if identity_at_zero:
                gain_raw = gain_raw - self.gain_head.bias
                shift = shift - self.shift_head.bias
            gain = torch.tanh(gain_raw)
            return (1.0 + gain) * target + shift
        assert self.factor_head is not None and self.formula_contract is not None
        factor_values = self.factor_head(state)
        if identity_at_zero:
            identities = state.new_tensor(
                [factor.identity for factor in self.formula_contract.factors]
            ).repeat_interleave(self.hidden_dim)
            factor_values = factor_values - self.factor_head.bias + identities
        factors = factor_values.reshape(
            *state.shape[:-1], self.formula_contract.factor_count, self.hidden_dim
        )
        flat_target = target.reshape(-1, self.hidden_dim)
        flat_factors = factors.reshape(
            -1, self.formula_contract.factor_count, self.hidden_dim
        )
        if self.formula_contract.execution.vectorization == "batched":
            result = self.formula(flat_target, flat_factors)
        else:
            result = torch.vmap(self.formula)(flat_target, flat_factors)
        if not isinstance(result, Tensor) or result.shape != flat_target.shape:
            raise ValueError("formula must return a complete next target state")
        return result.reshape_as(target)

    def _normalize_inputs(
        self,
        trace: Tensor,
        target_bank: Tensor,
        trace_mask: Tensor | None,
        target_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, bool]:
        if not isinstance(trace, Tensor) or not trace.is_floating_point():
            raise TypeError("trace must be a floating-point Tensor")
        if not isinstance(target_bank, Tensor) or not target_bank.is_floating_point():
            raise TypeError("target_bank must be a floating-point Tensor")
        if trace.ndim not in {2, 3} or target_bank.ndim != trace.ndim:
            raise ValueError("trace and target_bank must both be [T/S,H] or [B,T/S,H]")
        if trace.shape[-1] != self.hidden_dim:
            raise ValueError(f"trace feature dimension must be {self.hidden_dim}")
        if target_bank.shape[-2:] != (self.slots, self.hidden_dim):
            raise ValueError(
                f"target_bank trailing shape must be ({self.slots}, {self.hidden_dim})"
            )
        if trace.device != target_bank.device or trace.dtype != target_bank.dtype:
            raise ValueError("trace and target_bank must share device and dtype")
        squeeze = trace.ndim == 2
        trace_b = trace.unsqueeze(0) if squeeze else trace
        target_b = target_bank.unsqueeze(0) if squeeze else target_bank
        if trace_b.shape[0] != target_b.shape[0]:
            raise ValueError("trace and target_bank must share batch size")
        trace_mask_b = self._mask(
            trace_mask, trace.shape[:-1], trace.device, squeeze, "trace_mask"
        )
        target_mask_b = self._mask(
            target_mask, target_bank.shape[:-1], target_bank.device, squeeze, "target_mask"
        )
        if trace_mask_b is None:
            trace_mask_b = torch.ones(trace_b.shape[:2], dtype=torch.bool, device=trace.device)
        if target_mask_b is None:
            target_mask_b = torch.ones(
                target_b.shape[:2], dtype=torch.bool, device=target_bank.device
            )
        return trace_b, target_b, trace_mask_b, target_mask_b, squeeze

    @staticmethod
    def _mask(
        mask: Tensor | None,
        expected: torch.Size,
        device: torch.device,
        squeeze: bool,
        name: str,
    ) -> Tensor | None:
        if mask is None:
            return None
        if mask.dtype != torch.bool or mask.shape != expected or mask.device != device:
            raise ValueError(f"{name} must be boolean with shape {tuple(expected)}")
        return mask.unsqueeze(0) if squeeze else mask

    @staticmethod
    def _normalize_exposure(
        exposure: float | Tensor,
        *,
        batch_size: int,
        reference: Tensor,
    ) -> Tensor:
        value = torch.as_tensor(exposure, device=reference.device, dtype=reference.dtype)
        if value.ndim == 0:
            value = value.expand(batch_size)
        if value.ndim != 1 or value.shape[0] != batch_size:
            raise ValueError("exposure must be a scalar or have shape [B]")
        if not bool(torch.isfinite(value).all()) or bool((value < 0).any()):
            raise ValueError("exposure must be finite and non-negative")
        return value

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, slots={self.slots}, "
            f"workspace_dim={self.workspace_dim}, private_slots={self.private_slots}, "
            f"max_steps={self.policy.budget.max_steps}"
        )


__all__ = ["TargetBankUpdater", "WriteRefinePolicy"]
