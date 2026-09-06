"""Grouped attention pooling as ordinary typed Fabric and Bank operands."""

import torch

from arti import mechanisms as m


def build_program(dim=4, groups=3, dtype="float32"):
    x = m.InputBinding("x", m.TensorType(("B", "N", "D"), ("B", "N", dim), dtype=dtype))
    ids = m.InputBinding("ids", m.TensorType(("B", "N"), ("B", "N"), dtype="int64"))
    mask = m.InputBinding("mask", m.TensorType(("B", "N"), ("B", "N"), dtype="boolean"))
    weight = m.BankBinding("score_weight", "arti/segmented-example@1", "score_weight",
                           m.TensorType(("D",), (dim,), dtype=dtype))
    compute = "float64" if dtype == "float64" else "float32"
    values = m.cast(x, dtype=compute)
    scores = m.contract(values, m.cast(weight, dtype=compute), reduce_axes=(("D", "D"),))
    attention = m.segment(scores, ids, mask, axis="N", segment_axis="G", num_segments=groups, mode="softmax")
    pooled = m.segment(m.scale(values, attention), ids, mask, axis="N", segment_axis="G", num_segments=groups)
    program = m.FormulaProgram.build(outputs=(m.axis_index(x, axis="N"), attention, m.cast(pooled, dtype=dtype)))
    return program, weight


def main():
    program, binding = build_program()
    score_weight = torch.nn.Parameter(torch.randn(4))
    inputs = {"x": torch.randn(2, 8, 4), "ids": torch.arange(8).remainder(3).expand(2, 8),
              "mask": torch.ones(2, 8, dtype=torch.bool)}
    result = m.FormulaFabricV2(program)(inputs=inputs, banks={"score_weight": binding.bind(score_weight)})
    result.values[-1].square().mean().backward()
    print("positions, attention, pooled:", [tuple(t.shape) for t in result.values])
    print("Bank gradient finite:", bool(torch.isfinite(score_weight.grad).all()))


if __name__ == "__main__":
    main()
