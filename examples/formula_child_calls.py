"""Call one complete child ARTI twice, retaining every output and shared Bank.

Run from the repository: python examples/formula_child_calls.py
This demonstrates returning subgraphs, not learned topology or parallel speedup.
"""

from __future__ import annotations

import torch

from arti import mechanisms as m
from formula_named_outputs import build_query


def main() -> None:
    child = build_query()
    calls = tuple(m.FormulaProgramCallCandidateV1(
        name, child, input_slots={"input": "x"},
        output_slots={"wide-head": f"{name}-wide", "weighted-head": f"{name}-weighted"},
    ) for name in ("left", "right"))
    outputs = {slot: slot for call in calls for slot in call.output_slot_ids}
    parent = m.FormulaProgramQueryV5(
        slot_ids=("x", *outputs), candidates=calls,
        terminal_slots=outputs, max_steps=2,
    )
    x = torch.tensor([[1.0, -2.0, 0.5]], requires_grad=True)
    result = parent({"x": x})
    for name, value in result.outputs.items():
        print(f"{name}: {tuple(value.shape)}")
    assert calls[0].child is calls[1].child
    assert parent.owner_states[0] is child.owner_states[0]
    assert len(parent.owner_states) == 1
    sum(value.sum() for value in result.outputs.values()).backward()
    torch.testing.assert_close(x.grad, torch.full_like(x, 8.0))
    print(f"Parent calls: 2; total nested dispatches: {result.trace.total_dispatches}")
    print("One shared child and Bank owner, four retained heads, both gradient paths verified.")


if __name__ == "__main__":
    main()
