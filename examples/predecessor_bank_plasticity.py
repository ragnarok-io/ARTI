"""A minimal producer/effect/re-execution graph; no task training required."""

from __future__ import annotations

import copy

import torch

from arti import mechanisms as m


def build_query() -> m.FormulaProgramQueryV4:
    tensor_type = m.TensorType(
        ("B", "D"), ("B", 3), dtype="float32", domain="activation"
    )
    source = "example/predecessor-bank@1"

    def producer(name: str, input_slot: str, output_slot: str):
        value = m.InputBinding("value", tensor_type)
        weight = m.BankBinding("weight", source, "weight", tensor_type)
        candidate = m.FormulaProgramCandidate(
            name,
            m.FormulaProgram.build(outputs=(m.scale(value, weight),)),
            input_slots={"value": input_slot},
            output_slot=output_slot,
            operands={"weight": torch.full((1, 3), 2.0)},
        )
        return m.FormulaProgramTensorCandidateV3(
            candidate, plastic_bank_slot="weight", bank_owner_id="producer"
        )

    value = m.InputBinding("value", tensor_type)
    writer = m.BankBinding("writer", source, "writer", tensor_type)
    gain = m.BankBinding("gain", source, "gain", tensor_type)
    effect = m.neural_plasticity(
        value, m.scale(value, writer), m.scale(value, gain)
    )
    update = m.FormulaProgramEffectCandidateV3(
        "adapt",
        m.FormulaEffectProgramV2(
            m.FormulaProgram.build(outputs=(effect,)),
            data_input_name="value",
            state_type=tensor_type,
        ),
        input_slot="produced",
        output_slot="effected",
        operands={"writer": torch.full((1, 3), 0.1), "gain": torch.zeros(1, 3)},
        trainable_operands=("writer",),
        execution_count=torch.tensor(2.0),
        max_executions=4,
        trainable_execution_count=True,
    )
    return m.FormulaProgramQueryV4(
        slot_ids=("x", "produced", "effected", "output"),
        candidates=(
            producer("first", "x", "produced"),
            update,
            producer("second", "effected", "output"),
        ),
        terminal_slot="output",
        min_steps=3,
        max_steps=3,
    )


def main() -> None:
    query = build_query()
    x = torch.tensor([[3.0, -1.0, 0.5]])
    initial = copy.deepcopy(query.state_dict())
    execution = query({"x": x})
    assert execution.trace.stopped
    assert len(execution.proposals) == 1
    assert all(torch.equal(value, query.state_dict()[name]) for name, value in initial.items())
    query.commit_(execution)

    saved = copy.deepcopy(query.state_dict())
    restored = build_query()
    restored.load_state_dict(saved)
    torch.testing.assert_close(query({"x": x}).value, restored({"x": x}).value)
    print("First output:", execution.value.detach())
    print("Committed Bank:", query.initial_bank_state().values)
    print("Fresh reload: matched")


if __name__ == "__main__":
    main()
