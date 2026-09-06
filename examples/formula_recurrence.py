"""Shared Bank-gated recurrence expressed as a pure typed Fabric Scan."""

import torch

from arti import mechanisms as m


def gated_scan(dim=3, dtype="float32"):
    t = m.TensorType(("B", "D"), ("B", dim), dtype=dtype)
    x, h, total = (m.InputBinding(n, t) for n in ("x", "h", "total"))
    banks = {n: m.BankBinding(n, "arti/recurrent-example@1", n, m.TensorType(("D",), (dim,), dtype=dtype))
             for n in ("gain", "retention", "unit", "negative_unit", "zero")}
    gate = m.scalar_map(m.add(x, m.scale(h, banks["retention"])), mode="sigmoid")
    target = m.scalar_map(m.add(m.scale(x, banks["gain"]), h), mode="tanh")
    unit = m.broadcast(banks["unit"], output_axes=t.axis_names, output_sizes=t.sizes)
    complement = m.add(unit, m.scale(gate, banks["negative_unit"]))
    next_h = m.add(m.scale(gate, h), m.scale(complement, target))
    next_total = m.add(total, next_h)
    zero = m.broadcast(banks["zero"], output_axes=t.axis_names, output_sizes=t.sizes)
    positive = m.compare(next_h, zero, mode="gt")
    body = m.FormulaProgram.build(outputs=(next_h, next_total, positive))
    scan = m.FormulaScan(body, axis="T", sequence_types={
        "x": m.TensorType(("B", "T", "D"), ("B", "T", dim), dtype=dtype),
    }, carry_outputs={"h": body.outputs[0], "total": body.outputs[1]},
        emissions={"hidden": body.outputs[0], "positive": body.outputs[2]}, max_length=256)
    return scan, banks


def main():
    scan, bindings = gated_scan()
    weights = {"gain": torch.nn.Parameter(torch.randn(3)), "retention": torch.nn.Parameter(torch.randn(3)),
               "unit": torch.ones(3), "negative_unit": -torch.ones(3), "zero": torch.zeros(3)}
    inputs = {"x": torch.randn(2, 5, 3), "h": torch.zeros(2, 3), "total": torch.zeros(2, 3)}
    limits = m.FormulaLimits(max_instructions=2048, max_slots=4096, max_steps=1024)
    result = scan(inputs=inputs, banks={n: bindings[n].bind(w) for n, w in weights.items()}, limits=limits)
    result["emit.hidden"].square().mean().backward()
    print({n: tuple(t.shape) for n, t in result.items()})
    print("Shared Bank gradient finite:", bool(weights["gain"].grad.isfinite().all()))


if __name__ == "__main__":
    main()
