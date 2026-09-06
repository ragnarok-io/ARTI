"""Automatic CUDA differentiation with an explicit native reference."""

from contextlib import nullcontext

import torch

from arti import mechanisms as m
from arti._formula_grouped_training import grouped_formula_training


def build_query():
    t = m.TensorType(("B", "D"), ("B", 3), dtype="floating", domain="activation")
    x = m.InputBinding("x", t)
    weight = m.BankBinding("weight", "arti/example-weight@1", "weight", t)
    program = m.FormulaProgram.build(outputs=(m.scale(x, weight),))
    candidate = m.FormulaProgramTensorCandidateV3(
        m.FormulaProgramCandidateV2("scale", program, input_slots={"x": "x"},
            output_slot="output", operands={"weight": torch.ones(1, 3)}),
        plastic_bank_slot="weight", bank_owner_id="shared",
    )
    return m.FormulaProgramQueryV4(slot_ids=("x", "output"), candidates=(candidate,),
        terminal_slot="output", max_steps=1)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    query = build_query().to(device)
    def run(grouped):
        inputs = tuple(torch.full((1, 3), float(i + 1), device=device, requires_grad=True) for i in range(4))
        with nullcontext() if grouped else grouped_formula_training(backend="native"):
            rows = query.execute_many(tuple((query.candidates[0], query._arena({"x": x})) for x in inputs))
            outputs = tuple(row.values.get("output") for row in rows)
            loss = sum(value.square().mean() for value in outputs)
            gradients = torch.autograd.grad(loss, (*inputs, *query.parameters()), allow_unused=True)
        return outputs, gradients
    native, expected = run(False)
    grouped, actual = run(True)
    torch.testing.assert_close(grouped, native)
    torch.testing.assert_close(actual, expected)
    print("Grouped outputs and shared-parameter gradients match native execution.")


if __name__ == "__main__":
    main()
