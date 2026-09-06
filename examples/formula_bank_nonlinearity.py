"""Bank-parameterized nonlinearities and normalization as ordinary Fabric SSA.

The functions below construct expressions, not opaque neural operators. The
same operands can be shared by other programs or supplied by a routed Bank.
"""

import torch

from arti import mechanisms as m


def _like(value, target):
    return m.broadcast(
        value, output_axes=target.value_type.axis_names, output_sizes=target.value_type.sizes
    )


def build_program(*, dim=4, dtype="float32"):
    value_type = m.TensorType(("B", "N", "D"), ("B", "N", dim), dtype=dtype)
    channel_type = m.TensorType(("D",), (dim,), dtype=dtype)
    scalar_type = m.TensorType((), (), dtype=dtype)
    x = m.InputBinding("x", value_type)
    negative_one = m.InputBinding("negative_one", scalar_type)
    one = m.InputBinding("one", scalar_type)
    epsilon = m.InputBinding("epsilon", scalar_type)
    bindings = {
        name: m.BankBinding(name, "example/nonlinear-bank@1", name, channel_type)
        for name in ("slope", "beta", "offset", "radius", "gamma", "bias")
    }

    # Explicitly retain intermediate precision through the whole composition.
    compute_dtype = "float64" if dtype == "float64" else "float32"
    work = m.cast(x, dtype=compute_dtype)
    minus = m.cast(negative_one, dtype=compute_dtype)
    unit = m.cast(one, dtype=compute_dtype)
    eps = m.cast(epsilon, dtype=compute_dtype)
    parameters = {name: m.cast(binding, dtype=compute_dtype) for name, binding in bindings.items()}

    relu = m.scalar_map_v2(work, mode="relu")
    one_minus_slope = m.add(_like(unit, parameters["slope"]), m.scale(parameters["slope"], minus))
    # At x=0, this composition has derivative slope, matching PyTorch PReLU.
    prelu = m.add(m.scale(work, parameters["slope"]), m.scale(relu, one_minus_slope))
    swish = m.scale(work, m.scalar_map_v2(m.scale(work, parameters["beta"]), mode="sigmoid"))
    radius = m.add(m.scalar_map_v2(parameters["radius"], mode="softplus"), _like(eps, parameters["radius"]))
    shifted = m.add(work, _like(parameters["offset"], work))
    saturated = m.scale(
        m.scalar_map_v2(m.scale(shifted, m.scalar_map_v2(radius, mode="reciprocal")), mode="tanh"),
        radius,
    )

    mean = m.reduce_tensor(work, axis="D", mode="mean")
    centered = m.add(work, _like(m.scale(mean, minus), work))

    def normalized(value):
        variance = m.reduce_tensor(m.scale(value, value), axis="D", mode="mean")
        denominator = m.add(variance, _like(eps, variance))
        inverse = m.scalar_map_v2(denominator, mode="rsqrt")
        return m.scale(m.scale(value, _like(inverse, value)), parameters["gamma"])

    layer_norm = m.add(normalized(centered), _like(parameters["bias"], work))
    rms_norm = normalized(work)
    outputs = tuple(m.cast(value, dtype=dtype) for value in (prelu, swish, saturated, layer_norm, rms_norm))
    return m.FormulaProgram.build(outputs=outputs), bindings


def build_temperature_program(*, count=3, dtype="float32"):
    logits_type = m.TensorType(("B", "K"), ("B", count), dtype=dtype)
    scalar_type = m.TensorType((), (), dtype=dtype)
    sink_type = m.TensorType(("K",), (1,), dtype=dtype)
    logits = m.InputBinding("logits", logits_type)
    mask = m.InputBinding("mask", m.TensorType(("B", "K"), ("B", count), dtype="boolean"))
    sink_mask = m.InputBinding("sink_mask", m.TensorType(("B", "K"), ("B", 1), dtype="boolean"))
    floor = m.InputBinding("temperature_floor", scalar_type)
    bindings = {
        "temperature": m.BankBinding("temperature", "example/nonlinear-bank@1", "temperature", scalar_type),
        "bias": m.BankBinding("bias", "example/nonlinear-bank@1", "bias", m.TensorType(("K",), (count,), dtype=dtype)),
        "sink": m.BankBinding("sink", "example/nonlinear-bank@1", "sink", sink_type),
    }
    compute = "float64" if dtype == "float64" else "float32"
    tau = m.add(m.scalar_map_v2(m.cast(bindings["temperature"], dtype=compute), mode="softplus"), m.cast(floor, dtype=compute))
    sink = m.broadcast(m.cast(bindings["sink"], dtype=compute), output_axes=("B", "K"), output_sizes=("B", 1))
    work = m.cast(logits, dtype=compute)
    biased = m.add(work, _like(m.cast(bindings["bias"], dtype=compute), work))
    # Only visible logits receive bias; visible and sink logits share temperature.
    extended = m.concat(biased, sink, axis="K")
    scaled = m.scale(extended, m.scalar_map_v2(tau, mode="reciprocal"))
    probabilities = m.masked_softmax(scaled, m.concat(mask, sink_mask, axis="K"), axis="K")
    # Sink probability is intentionally not redistributed to the visible data.
    result = m.cast(m.slice_tensor(probabilities, axis="K", stop=count), dtype=dtype)
    return m.FormulaProgram.build(outputs=(result,)), bindings


def main():
    program, bindings = build_program()
    fabric = m.FormulaFabricV2(program)
    parameters = {
        name: torch.nn.Parameter(torch.full((4,), 0.5 if name == "slope" else 1.0))
        for name in bindings
    }
    result = fabric(
        inputs={
            "x": torch.randn(2, 3, 4), "negative_one": torch.tensor(-1.0),
            "one": torch.tensor(1.0), "epsilon": torch.tensor(1e-5),
        },
        banks={name: binding.bind(parameters[name]) for name, binding in bindings.items()},
    )
    sum(value.square().mean() for value in result.values).backward()
    print({"instructions": len(program.instructions), "outputs": [tuple(v.shape) for v in result.values]})
    print({name: parameter.grad.norm().item() for name, parameter in parameters.items()})


if __name__ == "__main__":
    main()
