"""Optional backend binding identities in the shared component registry."""

from __future__ import annotations

from ..component_registry import register_component


def register_binding(component_type, reference, *, config_builder):
    return register_component(
        reference,
        component_type=component_type,
        lifecycle="alpha",
        variant="native-jax-functional-binding",
        constructible=False,
        artifact_policy="host_bound",
        config_builder=config_builder,
        capabilities=("explicit-parameter-state", "jax-init-apply", "jax-transformable"),
    )
