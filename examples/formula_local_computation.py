"""Depthwise/local layers assembled from Window, Contract and Bank operands."""

import torch

from arti import mechanisms as m


def build_program(channels=3, kernel_size=3, *, out_channels=None, position_size=None,
                  stride=1, dilation=1, padding=(0, 0), dtype="float32"):
    x = m.InputBinding("x", m.TensorType(("B", "N", "C"), ("B", "N", channels), dtype=dtype))
    axes, sizes = ("K", "C"), (kernel_size, channels)
    if out_channels is not None:
        axes, sizes = axes + ("O",), sizes + (out_channels,)
    if position_size is not None:
        axes, sizes = ("P",) + axes, (position_size,) + sizes
    kernel = m.BankBinding("kernel", "arti/local-example@1", "kernel", m.TensorType(axes, sizes, dtype=dtype))
    compute = "float64" if dtype == "float64" else "float32"
    patch = m.window(m.cast(x, dtype=compute), axis="N", output_axis="P", window_axis="K",
                     kernel_size=kernel_size, stride=stride, dilation=dilation, padding=padding,
                     output_size="P" if position_size is None else position_size)
    result = m.contract(patch, m.cast(kernel, dtype=compute), reduce_axes=(("K", "K"),))
    if out_channels is not None:
        result = m.reduce_tensor(result, axis="C", mode="sum")
    axis, dim = ("C", channels) if out_channels is None else ("O", out_channels)
    slope = m.BankBinding("slope", "arti/local-example@1", "slope", m.TensorType((axis,), (dim,), dtype=dtype))
    negative = m.scale(result, m.cast(slope, dtype=compute))
    positive = m.scalar_map(result, mode="relu")
    # PReLU = relu(z) + a*z - a*relu(z), including the chosen derivative at zero.
    negative_unit = m.BankBinding("negative_unit", "arti/local-example@1", "negative_unit", m.TensorType((), (), dtype=dtype))
    positive_correction = m.scale(m.scale(positive, m.cast(slope, dtype=compute)), m.cast(negative_unit, dtype=compute))
    output = m.add(positive, m.add(negative, positive_correction))
    return m.FormulaProgram.build(outputs=(m.cast(output, dtype=dtype),)), {b.name: b for b in (kernel, slope, negative_unit)}


def main():
    program, bindings = build_program(padding=(1, 1))
    values = {"kernel": torch.nn.Parameter(torch.randn(3, 3)),
              "slope": torch.nn.Parameter(torch.full((3,), 0.2)), "negative_unit": torch.tensor(-1.)}
    result = m.FormulaFabricV2(program)(inputs={"x": torch.randn(2, 8, 3)},
                                      banks={k: bindings[k].bind(v) for k, v in values.items()}).values[0]
    result.square().mean().backward()
    print("Output:", tuple(result.shape), "kernel gradient finite:", bool(values["kernel"].grad.isfinite().all()))


if __name__ == "__main__":
    main()
