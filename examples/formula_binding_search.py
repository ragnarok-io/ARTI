"""Finite atomic choices and cross-depth connections in the existing Query@7."""

import torch

from arti import mechanisms as m


def build_graph():
    tensor = m.TensorType(("B", "D"), ("B", 2), dtype="float32")
    scalar = m.TensorType(("B",), ("B",), dtype="float32")
    x = m.InputBinding("value", tensor)
    gain = m.BankBinding("gain", "arti/binding-example@1", "gain", m.TensorType(("D",), (2,), dtype="float32"))
    weight = torch.tensor([1.5, 0.5])
    seed = m.scale(x, gain)
    score = m.reduce_tensor(seed, axis="D", mode="mean")
    body = m.FormulaProgram.build(outputs=(seed, score))
    entry = m.FormulaProgramTensorCandidateV4(m.FormulaProgramCandidateV3(
        "entry", body, input_slots={"value": "x"},
        output_slots=dict(zip(body.outputs, ("seed", "seed.score"), strict=True)),
        operands={"gain": weight}, trainable_operands=("gain",),
    ))
    types = {name: tensor for name in ("x", "z", "seed", "hidden", "sum")}
    types.update({name: scalar for name in ("seed.score", "hidden.score", "sum.score", "answer")})
    nodes = [entry]
    for mode in ("silu", "tanh"):
        activated = m.scalar_map(x, mode=mode)
        response = m.reduce_tensor(activated, axis="D", mode="mean")
        program = m.FormulaProgram.build(outputs=(activated, response))
        template = m.FormulaProgramTensorCandidateV4(m.FormulaProgramCandidateV3(
            mode, program, input_slots={"value": "seed"},
            output_slots=dict(zip(program.outputs, ("hidden", "hidden.score"), strict=True)),
        ))
        nodes.extend(m.expand_candidate_bindings(
            template, slot_types=types, input_choices={"value": ("x", "z", "seed")},
            output_choices={program.outputs[0]: ("hidden",), program.outputs[1]: ("hidden.score",)},
        ))
    left, right = m.InputBinding("left", tensor), m.InputBinding("right", tensor)
    total = m.add(left, right)
    program = m.FormulaProgram.build(outputs=(total, m.reduce_tensor(total, axis="D", mode="mean")))
    template = m.FormulaProgramTensorCandidateV4(m.FormulaProgramCandidateV3(
        "join", program, input_slots={"left": "hidden", "right": "z"},
        output_slots=dict(zip(program.outputs, ("sum", "sum.score"), strict=True)),
    ))
    nodes.extend(m.expand_candidate_bindings(
        template, slot_types=types, input_choices={"left": ("hidden", "seed"), "right": ("x", "z")},
        output_choices={program.outputs[0]: ("sum",), program.outputs[1]: ("sum.score",)},
    ))
    program = m.FormulaProgram.build(outputs=(m.reduce_tensor(x, axis="D", mode="mean"),))
    template = m.FormulaProgramTensorCandidateV4(m.FormulaProgramCandidateV3(
        "read", program, input_slots={"value": "sum"}, output_slots={program.outputs[0]: "answer"},
    ))
    nodes.extend(m.expand_candidate_bindings(
        template, slot_types=types, input_choices={"value": ("seed", "hidden", "sum")},
        output_choices={program.outputs[0]: ("answer",)},
    ))
    # No task labels or new Query network: responses are already executed SSA.
    # The example leaves response calibration to a subsequent final-loss task.
    edges = {}
    for producer in nodes:
        response = next((s for s in producer.output_slot_ids if s.endswith(".score")), None)
        if response is not None:
            edges[producer.candidate_id] = {
                other.candidate_id: response for other in nodes if other is not entry
            }
    return m.FormulaProgramQueryV7(
        slot_ids=tuple(types), candidates=tuple(nodes), terminal_slots={"answer": "answer"},
        entry_candidates=("entry",), continuations=edges, max_steps=4, cooperation_width=2,
    )


if __name__ == "__main__":
    model = build_graph()
    values = {"x": torch.tensor([[1., -0.5]]), "z": torch.tensor([[-0.2, 0.7]])}
    result = model(values)
    print(result.outputs)
    for frontier in result.frontiers:
        print([(node.candidate_id, node.inputs) for node in frontier.nodes])
