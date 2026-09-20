"""Keep differently shaped Formula outputs in one Query-selected graph.

Run from the repository: python examples/formula_named_outputs.py
This is a multi-output building block, not a recursive ARTI call.
"""

from __future__ import annotations

import torch

from arti import mechanisms as m


def build_query() -> m.FormulaProgramQueryV5:
    tensor_type = m.TensorType(("B", "D"), ("B", 3), dtype="floating", domain="activation")
    x = m.InputBinding("x", tensor_type)
    weight = m.BankBinding("weight", "arti/named-output-example@1", "weight", tensor_type)
    program = m.FormulaProgram.build(outputs=(m.concat(x, x, axis="D"), m.scale(x, weight)))
    candidate = m.FormulaProgramCandidateV3(
        "heads", program,
        input_slots={"x": "input"},
        output_slots=dict(zip(program.outputs, ("wide", "weighted"), strict=True)),
        operands={"weight": torch.full((1, 3), 2.0)},
    )
    producer = m.FormulaProgramTensorCandidateV4(
        candidate, plastic_bank_slot="weight", bank_owner_id="memory",
    )
    return m.FormulaProgramQueryV5(
        slot_ids=("input", "wide", "weighted"), candidates=(producer,),
        terminal_slots={"wide-head": "wide", "weighted-head": "weighted"}, max_steps=1,
    )


def main() -> None:
    query = build_query()
    x = torch.tensor([[1.0, -2.0, 0.5]], requires_grad=True)
    result = query({"input": x})
    for name, value in result.outputs.items():
        print(f"{name}: shape={tuple(value.shape)}, values={value.detach().tolist()}")
    assert result.output_producers["wide-head"].plastic_slot is None
    assert result.output_producers["weighted-head"].plastic_slot is not None
    sum(value.sum() for value in result.outputs.values()).backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 4.0))
    print("Both heads survive; only the Bank-dependent head carries a write target.")
    print("Shared-input gradient verified. No automatic Bank commit.")


if __name__ == "__main__":
    main()
