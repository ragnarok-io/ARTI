"""Run a caller-routed Formula Fabric inside Pulse@2."""

import torch

from arti import alpha


value = torch.tensor(
    [[[1.0, 2.0], [3.0, 4.0], [10.0, 12.0], [0.5, 0.5]]]
)
mask = torch.ones(1, 4, dtype=torch.bool)

topology = alpha.ReversibleTopology(active_count=3)
fold, unfold = topology.operations()
program = alpha.FormulaFabricProgram(
    arena_capacity=4,
    feature_dim=2,
    steps=((alpha.FormulaInvocation(alpha.FormulaPrimitive.ADD, 2),),),
    domain="pulse-active",
)
pulse = alpha.AdaptivePulse(
    fold=fold,
    intervention=alpha.FormulaAttention(
        alpha.MagnitudeInterventionPolicy(), alpha.StableTopKIntervention(1)
    ),
    selective_compute=alpha.FormulaFabricCompute(
        alpha.FormulaFabric(program), active_count=3
    ),
    unfold=unfold,
)

weights = value.new_zeros((1, 1, 1, 2, 4))
weights[:, 0, 0, 0, 0] = 1
weights[:, 0, 0, 1, 1] = 1
enabled = torch.ones(1, 1, 1, dtype=torch.bool)
route = alpha.FormulaRoutePlan(weights, enabled, enabled, enabled)

result = pulse.run_tensor(value, mask=mask, formula_route=route)
print(result.value)
print(result.diagnostics.compute.program_fingerprint)
