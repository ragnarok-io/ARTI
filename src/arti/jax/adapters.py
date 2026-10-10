"""Optional Flax integration using native JAX init/apply contracts."""

from __future__ import annotations

from fnmatch import fnmatchcase

import jax

from .program import ARTILayer, NativeModule


def from_flax(
    module,
    *,
    inputs=("value",),
    outputs=("value",),
    mutable=(),
    rng_collections=(),
    init_kwargs=None,
    apply_kwargs=None,
    state_batch_axis=None,
):
    """Bind a Linen-style module without depending on Flax in the core backend.

    Parameters live in the parameter tree. Other variable collections remain
    explicit module state; the selected mutable collections are returned by
    Flax and installed in the next state.
    """
    if not callable(getattr(module, "init", None)) or not callable(getattr(module, "apply", None)):
        raise TypeError("from_flax requires a module exposing init and apply")
    mutable = tuple(mutable)
    rng_collections = tuple(rng_collections)
    init_kwargs, apply_kwargs = dict(init_kwargs or {}), dict(apply_kwargs or {})

    def rngs(key, *, initializing):
        if key is None and (initializing or rng_collections):
            raise ValueError("Flax initialization and stochastic collections require an RNG key")
        names = (("params",) if initializing else ()) + rng_collections
        return {name: jax.random.fold_in(key, index) for index, name in enumerate(names)}

    def initialize(key, *values):
        variables = module.init(rngs(key, initializing=True), *values, **init_kwargs)
        return dict(variables.get("params", {})), {
            name: value for name, value in variables.items() if name != "params"
        }

    def apply(params, state, *values, rng=None):
        variables = {**state, "params": params}
        raw = module.apply(
            variables,
            *values,
            rngs=rngs(rng, initializing=False),
            mutable=mutable if mutable else False,
            **apply_kwargs,
        )
        if mutable:
            outputs, updates = raw
            return outputs, {**state, **dict(updates)}
        return raw, state

    return NativeModule(
        initialize,
        apply,
        tuple(inputs),
        tuple(outputs),
        bool(rng_collections),
        bool(mutable),
        state_batch_axis,
    )


def as_flax_module(layer: ARTILayer, *, state_collection="arti_state"):
    """Expose a native ARTI host as an ordinary Flax Linen module."""
    if not isinstance(layer, ARTILayer):
        raise TypeError("as_flax_module requires arti.jax.ARTILayer")
    try:
        from flax import linen as nn
    except ImportError as error:
        raise ImportError("as_flax_module requires the optional flax dependency") from error

    class ARTIProgramModule(nn.Module):
        @nn.compact
        def __call__(self, x):
            initial = None
            if self.is_initializing():
                initial = layer.init(self.make_rng("params"), x)
            params = self.param(
                "program", lambda key: initial[0] if initial is not None else layer.init(key, x)[0]
            )
            state = self.variable(state_collection, "program", lambda: initial[1])
            stochastic = layer.program is not None and layer.program.requires_rng
            rng = (
                self.make_rng("arti")
                if self.has_rng("arti") or (stochastic and self.is_initializing())
                else None
            )
            value, next_state = layer.apply(params, state.value, x, rng=rng)
            if self.is_mutable_collection(state_collection) and not self.is_initializing():
                state.value = next_state
            return value

    return ARTIProgramModule()


def attach(model, layer: ARTILayer, *, targets, state_collection="arti_state"):
    """Return a Linen model with native ARTI hosts at named module outputs.

    Targets are dot-separated Linen scope paths relative to the host model.
    Glob patterns select static call sites during tracing; compiled calls then
    execute the inserted JAX graph directly.
    """
    try:
        from flax import linen as nn
    except ImportError as error:
        raise ImportError(
            "native model attachment requires the optional flax dependency"
        ) from error
    if not isinstance(model, nn.Module):
        raise TypeError("attach requires a Flax Linen module")
    patterns = (targets,) if isinstance(targets, str) else tuple(targets)
    if not patterns:
        raise ValueError("attach requires at least one target module path")

    class AttachedModel(nn.Module):
        @nn.compact
        def __call__(self, *args, **kwargs):
            adapters = {}

            def intercept(next_function, values, options, context):
                path = context.module.scope.path
                result = next_function(*values, **options)
                if context.method_name != "__call__" or not path or path[0] != "model":
                    return result
                relative = ".".join(path[1:])
                if not any(fnmatchcase(relative, pattern) for pattern in patterns):
                    return result
                if not hasattr(result, "shape") or not hasattr(result, "dtype"):
                    raise TypeError("an attached ARTI boundary requires a tensor output")
                if relative not in adapters:
                    name = (
                        "arti_" + relative.replace("_", "__").replace(".", "_d")
                        if relative
                        else "arti"
                    )
                    adapters[relative] = as_flax_module(
                        layer, state_collection=state_collection
                    ).clone(parent=self, name=name)
                return adapters[relative](result)

            host = model.clone(parent=self, name="model")
            with nn.intercept_methods(intercept):
                return host(*args, **kwargs)

    return AttachedModel()


__all__ = ["as_flax_module", "attach", "from_flax"]
