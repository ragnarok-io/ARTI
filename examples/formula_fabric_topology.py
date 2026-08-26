"""Run Formula Fabric over a reversibly folded active tensor workspace."""

import torch

from arti import alpha


x = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]]])
mask = torch.ones(1, 5, dtype=torch.bool)
topology = alpha.ReversibleTopology(active_count=3)
folded = topology.fold(x, mask)

program = alpha.FormulaFabricProgram(
    arena_capacity=3,
    feature_dim=1,
    steps=((alpha.FormulaInvocation(alpha.FormulaPrimitive.ADD, 0),),),
    domain="example-active",
)
arena = alpha.FormulaArenaState(
    folded.active,
    folded.active_mask,
    torch.zeros_like(folded.active_mask, dtype=torch.int64),
    program.domain,
)
weights = torch.zeros(1, 1, 1, 2, 3)
weights[:, 0, 0, 0, 1] = 1
weights[:, 0, 0, 1, 2] = 1
enabled = torch.ones(1, 1, 1, dtype=torch.bool)
route = alpha.FormulaRoutePlan(weights, enabled, enabled, enabled)

computed = alpha.FormulaFabric(program).to(x.device)(arena, route)
y = topology.unfold(folded.replace(active=computed.state.value)).value

print(y)
