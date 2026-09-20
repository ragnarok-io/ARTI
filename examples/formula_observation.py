"""Run an existing learned Observation trajectory inside Formula Fabric."""

import torch

from arti import mechanisms as m


def main():
    policy = m.LearnedObservationPolicy(8, 2, max_observations=4)
    operator = m.StateAffineObservationOperator(8, 2)
    x = torch.randn(2, 6, 8)
    mask = torch.ones(2, 6, dtype=torch.bool)
    plan = policy(x, mask)

    value = m.InputBinding("x", m.TensorType(("B", "N", "D"), ("B", "N", 8)))
    states = m.InputBinding("states", m.TensorType(("B", "T", "S"), ("B", "T", 2)))
    source_mask = m.InputBinding("mask", m.TensorType(("B", "N"), ("B", "N"), dtype="boolean"))
    activity = m.InputBinding("activity", m.TensorType(("B", "T"), ("B", "T")))
    trajectory_mask = m.InputBinding(
        "trajectory_mask", m.TensorType(("B", "T"), ("B", "T"), dtype="boolean")
    )
    weight = m.BankBinding(
        "weight", "example/observation-bank@1", "projection", m.TensorType(("P", "S"), (16, 2))
    )
    bias = m.BankBinding(
        "bias", "example/observation-bank@1", "projection", m.TensorType(("P",), (16,))
    )

    observed = m.observe_affine(value, states, source_mask, activity, weight, bias)
    visible = m.observation_mask(source_mask, trajectory_mask)
    fabric = m.FormulaFabricV2(m.FormulaProgram.build(outputs=(observed, visible)))
    result = fabric(
        inputs={
            "x": x,
            "states": plan.states,
            "mask": mask,
            "activity": plan.activity_weights(),
            "trajectory_mask": plan.mask,
        },
        banks={
            "weight": weight.bind(operator.projection.weight),
            "bias": bias.bind(operator.projection.bias),
        },
    )
    print("values:", tuple(result.values[0].shape), "mask:", tuple(result.values[1].shape))


if __name__ == "__main__":
    main()
