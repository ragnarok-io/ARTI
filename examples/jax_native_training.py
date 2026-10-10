"""Jitted native model training with a persistent ARTI resource graph."""

from __future__ import annotations

import json

from flax import linen as nn
import jax
import jax.numpy as jnp
import optax

from arti import mechanisms as m
import arti.jax as aj


def main():
    @aj.fabric_layer(inputs=("x", "memory"), outputs=("out", "memory"))
    def operate(params, state, x, memory, *, rng=None):
        next_memory = 0.9 * memory + 0.1 * x
        return {"out": x * params["gain"] + 0.05 * next_memory, "memory": next_memory}, state

    module = aj.NativeModule(
        lambda key, x, memory: ({"gain": jnp.ones((), dtype=x.dtype)}, {}),
        operate,
        implementation_ref="example/native-memory@1",
    )
    node = aj.as_fabric_node(
        "memory_layer",
        module,
        input_ports={"x": m.ResourcePort("x"), "memory": m.ResourcePort("memory")},
        output_ports={"out": m.ResourcePort("out"), "memory": m.ResourcePort("memory")},
    )

    def resource(name, lifetime):
        spec = m.TensorResourceSpec(name, m.TensorViewPattern(), lifetime)
        return aj.tensor_resource(spec, jnp.zeros((1, 8), dtype=jnp.float32), axis_names=("B", "D"))

    graph = m.ProgramGraph(
        (
            resource("x", m.ResourceLifetime.CALL),
            resource("out", m.ResourceLifetime.CALL),
            resource("memory", m.ResourceLifetime.PERSISTENT),
        ),
        (),
        nodes=(node,),
        programs={"forward": ("memory_layer",)},
    )
    layer = aj.ARTILayer(
        aj.bind_resource_graph(graph, "forward"), input_resource_id="x", output_resource_id="out"
    )

    class Model(nn.Module):
        @nn.compact
        def __call__(self, value):
            hidden = nn.Dense(8, name="encoder")(value)
            return nn.Dense(1, name="head")(jax.nn.gelu(hidden))

    model = aj.attach(Model(), layer, targets="encoder")
    key = jax.random.key(31)
    x = jax.random.normal(key, (32, 3), dtype=jnp.float32)
    target = (jnp.sin(x[:, 0]) + 0.3 * x[:, 1] - x[:, 2])[:, None]
    variables = model.init(key, x)
    optimizer = optax.adam(0.02)
    optimizer_state = optimizer.init(variables["params"])

    def step(carry, _):
        params, state, optimizer_state = carry

        def objective(candidate):
            output, next_state = model.apply(
                {"params": candidate, **state}, x, mutable=("arti_state",)
            )
            return jnp.square(output - target).mean(), next_state

        (loss, state), gradients = jax.value_and_grad(objective, has_aux=True)(params)
        updates, optimizer_state = optimizer.update(gradients, optimizer_state, params)
        params = optax.apply_updates(params, updates)
        return (params, state, optimizer_state), loss

    state = {name: value for name, value in variables.items() if name != "params"}
    train = jax.jit(lambda carry: jax.lax.scan(step, carry, None, length=80))
    (_, final_state, _), losses = train((variables["params"], state, optimizer_state))
    losses.block_until_ready()
    runtime = final_state["arti_state"]["arti_encoder"]["program"]
    assert int(runtime.steps["memory"][0]) == len(losses)
    assert float(losses[-1]) < float(losses[0])
    print(
        json.dumps(
            {
                "first_loss": float(losses[0]),
                "last_loss": float(losses[-1]),
                "steps": len(losses),
                "memory_writes": int(runtime.steps["memory"][0]),
                "devices": [str(device) for device in jax.devices()],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
