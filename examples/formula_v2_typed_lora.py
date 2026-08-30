"""Run one hard-routed FormulaFabric@2 LoRA-shaped Bank recipe."""

import torch

from arti import alpha


torch.manual_seed(1)
batch, sequence, input_dim, output_dim, rank = 2, 4, 8, 6, 2
candidates, query_dim = 3, 5

bank = alpha.FormulaOperandBank(
    keys=torch.randn(candidates, query_dim),
    operands={
        "A": torch.randn(candidates, rank, input_dim),
        "B": torch.randn(candidates, output_dim, rank),
        "gain": torch.ones(candidates),
    },
    source_ref="example/formula-operand-bank@1",
    bundle_id="adapter",
)
program = alpha.build_routed_lora_program(
    input_dim=input_dim,
    output_dim=output_dim,
    rank=rank,
    candidate_count=candidates,
    source_ref=bank.source_ref,
    bundle_id=bank.bundle_id,
    member_ids=bank.member_ids,
)
fabric = alpha.FormulaFabricV2(program)

query = torch.randn(batch, query_dim)
selection = bank.route(query, estimator="hard")
x = torch.randn(batch, sequence, input_dim)
base = torch.randn(batch, sequence, output_dim)
result = fabric(
    inputs={"x": x, "base": base, "formula.route": selection.route},
    banks=bank.bind(program),
    return_trace=True,
)

print(result.values[0].shape)
print(selection.hard_indices)
print(result.trace.program_fingerprint if result.trace is not None else None)
