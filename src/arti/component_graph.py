"""Explicit component graphs for nested ARTI modules.

The regular component provenance list remains the compact compatibility
contract.  This module adds a richer, optional graph representation for
nested modules, mounts, typed bindings, and shared parameters without making
ordinary PyTorch composition depend on a new base class.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

from torch import nn

from .component_registry import (
    ComponentCompatibilityError,
    ComponentRef,
    component_spec,
    get_component_registry,
    state_dict_schema,
)


COMPONENT_GRAPH_FORMAT = "arti.component.graph"
COMPONENT_GRAPH_VERSION = 1


class ComponentGraphError(ComponentCompatibilityError):
    """Raised when an explicit component graph is invalid."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _path(raw_path: str) -> str:
    return "$" if raw_path == "" else f"$.{raw_path}"


def _api_name(module: nn.Module) -> str:
    cls = type(module)
    return f"{cls.__module__}.{cls.__qualname__}"


def _module_occurrences(model: nn.Module) -> list[tuple[str, nn.Module]]:
    try:
        return list(model.named_modules(remove_duplicate=False))
    except TypeError:  # pragma: no cover - compatibility with old PyTorch
        return list(model.named_modules())


def _parameter_groups(model: nn.Module) -> list[dict[str, Any]]:
    groups: dict[tuple[str, int], list[str]] = {}
    try:
        parameters = model.named_parameters(remove_duplicate=False)
        buffers = model.named_buffers(remove_duplicate=False)
    except TypeError:  # pragma: no cover - compatibility with old PyTorch
        parameters = model.named_parameters()
        buffers = model.named_buffers()

    for name, value in parameters:
        groups.setdefault(("same_parameter", id(value)), []).append(name)
    for name, value in buffers:
        groups.setdefault(("same_buffer", id(value)), []).append(name)

    result: list[dict[str, Any]] = []
    counters: dict[str, int] = {"same_parameter": 0, "same_buffer": 0}
    for (kind, _identity), members in sorted(groups.items(), key=lambda item: (item[0][0], sorted(item[1]))):
        unique_members = sorted(set(members))
        if len(unique_members) < 2:
            continue
        index = counters[kind]
        counters[kind] += 1
        result.append(
            {
                "id": f"{kind}-{index:04d}",
                "kind": kind,
                "members": unique_members,
                "canonical": unique_members[0],
            }
        )
    return result


def _known_reference(reference: str) -> bool:
    registry = get_component_registry()
    try:
        registry.registration_for_reference(reference)
        return True
    except Exception:
        pass
    try:
        from .recall_registry import describe_formula

        describe_formula(reference)
        return True
    except Exception:
        pass
    try:
        from .survival import describe_survival

        describe_survival(reference)
        return True
    except Exception:
        return False


def _normalise_bindings(model: nn.Module, bindings: Sequence[Any] | None) -> list[dict[str, Any]]:
    source = bindings
    if source is None:
        candidate = getattr(model, "component_bindings", None)
        if callable(candidate):
            source = candidate()
        elif candidate is not None:
            source = candidate
    if source is None:
        return []
    if isinstance(source, (str, bytes)) or not isinstance(source, Sequence):
        raise TypeError("component bindings must be a sequence")

    result: list[dict[str, Any]] = []
    for item in source:
        if isinstance(item, Mapping):
            kind = item.get("kind", "data")
            from_port = item.get("from", item.get("from_port"))
            to_port = item.get("to", item.get("to_port"))
            record = dict(item)
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            kind = "data"
            from_port, to_port = item
            record = {}
        else:
            raise TypeError("each component binding must be a mapping or a (from, to) pair")
        if not isinstance(kind, str) or not kind:
            raise ValueError("component binding kind must be a non-empty string")
        if not isinstance(from_port, str) or not from_port:
            raise ValueError("component binding 'from' must be a non-empty string")
        if not isinstance(to_port, str) or not to_port:
            raise ValueError("component binding 'to' must be a non-empty string")
        result.append({**record, "kind": kind, "from": from_port, "to": to_port})
    return sorted(result, key=_canonical_json)


def _contains_edges(
    mounts: Sequence[Mapping[str, Any]],
    path_to_node: Mapping[str, str],
) -> list[dict[str, Any]]:
    edges: set[tuple[str, str, str]] = set()
    for mount in mounts:
        mount_path = str(mount["path"])
        child = str(mount["node"])
        if mount_path == "$":
            continue
        parts = mount_path[2:].split(".")
        for index in range(len(parts) - 1, -1, -1):
            parent_path = "$" if index == 0 else "$." + ".".join(parts[:index])
            parent = path_to_node.get(parent_path)
            if parent is None or parent == child:
                continue
            edges.add((parent, child, mount_path))
            break
    return [
        {"kind": "contains", "from": parent, "to": child, "mount": mount_path}
        for parent, child, mount_path in sorted(edges)
    ]


def component_graph(model: nn.Module, *, bindings: Sequence[Any] | None = None) -> dict[str, Any]:
    """Build a deterministic graph for a PyTorch module tree.

    Registered ARTI modules are represented as ``component`` nodes.  Other
    ``nn.Module`` objects remain ``module`` nodes so a composite model can be
    inspected without requiring every PyTorch primitive to be registered.
    Repeated mounts of the same module instance share one node identity.
    """

    if not isinstance(model, nn.Module):
        raise TypeError("component_graph expects a torch.nn.Module")

    registry = get_component_registry()
    occurrences = _module_occurrences(model)
    groups: dict[int, dict[str, Any]] = {}
    for raw_path, module in occurrences:
        path = _path(raw_path)
        group = groups.setdefault(
            id(module),
            {"module": module, "paths": [], "registration": registry.registration_for(module)},
        )
        group["paths"].append(path)

    ordered_groups = sorted(groups.values(), key=lambda item: min(item["paths"]))
    path_to_node: dict[str, str] = {}
    nodes: list[dict[str, Any]] = []
    mounts: list[dict[str, Any]] = []
    for index, group in enumerate(ordered_groups):
        node_id = f"node-{index:04d}"
        module = group["module"]
        registration = group["registration"]
        first_path = min(group["paths"])
        if registration is not None:
            spec = component_spec(module, path=first_path).to_dict()
            node = {
                "id": node_id,
                "kind": "component",
                "ref": spec["ref"],
                "api": spec["api"],
                "variant": spec["variant"],
                "lifecycle": spec["lifecycle"],
                "config_schema_version": spec["config_schema_version"],
                "state_schema_version": spec["state_schema_version"],
                "config": spec["config"],
                "config_fingerprint": spec["config_fingerprint"],
                "dependencies": spec["dependencies"],
            }
        else:
            config: dict[str, Any] = {}
            node = {
                "id": node_id,
                "kind": "module",
                "ref": None,
                "api": _api_name(module),
                "variant": "opaque",
                "lifecycle": "opaque",
                "config_schema_version": 1,
                "state_schema_version": 1,
                "config": config,
                "config_fingerprint": _sha256_json(config),
                "dependencies": [],
            }
        node["mounts"] = sorted(group["paths"])
        nodes.append(node)
        for mount_path in group["paths"]:
            path_to_node[mount_path] = node_id
            mounts.append({"path": mount_path, "node": node_id})

    mounts.sort(key=lambda item: item["path"])
    edges = _contains_edges(mounts, path_to_node)
    for node in nodes:
        for dependency in node["dependencies"]:
            edges.append({"kind": "requires", "from": node["id"], "to_ref": dependency})
    edges.sort(key=_canonical_json)
    graph = {
        "format": COMPONENT_GRAPH_FORMAT,
        "schema_version": COMPONENT_GRAPH_VERSION,
        "root": path_to_node.get("$"),
        "nodes": sorted(nodes, key=lambda item: item["id"]),
        "mounts": mounts,
        "edges": edges,
        "bindings": _normalise_bindings(model, bindings),
        "parameter_groups": _parameter_groups(model),
        "state_schema_fingerprint": state_dict_schema(model.state_dict())["fingerprint"],
    }
    return {**graph, "closure_fingerprint": component_closure_fingerprint(graph)}


def component_closure_fingerprint(graph: Mapping[str, Any]) -> str:
    """Hash graph structure, configuration, state schema, and typed edges."""

    content = {key: value for key, value in graph.items() if key != "closure_fingerprint"}
    return _sha256_json(content)


def _validate_no_contains_cycle(graph: Mapping[str, Any]) -> None:
    adjacency: dict[str, list[str]] = {}
    for edge in graph["edges"]:
        if edge.get("kind") == "contains":
            adjacency.setdefault(edge["from"], []).append(edge["to"])
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ComponentGraphError("component graph contains a cycle")
        if node in visited:
            return
        visiting.add(node)
        for child in adjacency.get(node, ()):
            visit(child)
        visiting.remove(node)
        visited.add(node)

    for node in adjacency:
        visit(node)


def validate_component_graph(
    value: Mapping[str, Any],
    *,
    model: nn.Module | None = None,
) -> dict[str, Any]:
    """Validate a serialized component graph and optionally a target model."""

    if not isinstance(value, Mapping):
        raise ComponentGraphError("component graph must be a mapping")
    required = {
        "format",
        "schema_version",
        "root",
        "nodes",
        "mounts",
        "edges",
        "bindings",
        "parameter_groups",
        "state_schema_fingerprint",
        "closure_fingerprint",
    }
    if set(value) != required:
        raise ComponentGraphError("component graph has missing or unknown fields")
    if value["format"] != COMPONENT_GRAPH_FORMAT or value["schema_version"] != COMPONENT_GRAPH_VERSION:
        raise ComponentGraphError("unsupported component graph format or version")
    nodes = value["nodes"]
    mounts = value["mounts"]
    edges = value["edges"]
    if not isinstance(nodes, list) or any(not isinstance(item, Mapping) for item in nodes):
        raise ComponentGraphError("component graph nodes must be a list of mappings")
    if not isinstance(mounts, list) or any(not isinstance(item, Mapping) for item in mounts):
        raise ComponentGraphError("component graph mounts must be a list of mappings")
    if not isinstance(edges, list) or any(not isinstance(item, Mapping) for item in edges):
        raise ComponentGraphError("component graph edges must be a list of mappings")
    node_values = [item.get("id") for item in nodes]
    if any(not isinstance(node_id, str) or not node_id for node_id in node_values):
        raise ComponentGraphError("component graph node ids must be unique non-empty strings")
    node_ids = set(node_values)
    if len(node_ids) != len(nodes):
        raise ComponentGraphError("component graph node ids must be unique non-empty strings")
    mount_paths: set[str] = set()
    mount_nodes: dict[str, str] = {}
    for mount in mounts:
        if set(mount) != {"path", "node"} or not isinstance(mount["path"], str) or mount["path"] in mount_paths:
            raise ComponentGraphError("component graph mount paths must be unique")
        if mount["node"] not in node_ids:
            raise ComponentGraphError("component graph mount references an unknown node")
        mount_paths.add(mount["path"])
        mount_nodes[mount["path"]] = mount["node"]
    if value["root"] not in node_ids:
        raise ComponentGraphError("component graph root references an unknown node")

    registry = get_component_registry()
    for node in nodes:
        required_node = {
            "id",
            "kind",
            "ref",
            "api",
            "variant",
            "lifecycle",
            "config_schema_version",
            "state_schema_version",
            "config",
            "config_fingerprint",
            "dependencies",
            "mounts",
        }
        if set(node) != required_node:
            raise ComponentGraphError("component graph node has missing or unknown fields")
        if node["config_fingerprint"] != _sha256_json(node["config"]):
            raise ComponentGraphError(f"component graph config fingerprint is invalid at {node['id']!r}")
        expected_mounts = sorted(mount["path"] for mount in mounts if mount["node"] == node["id"])
        if not isinstance(node["mounts"], list) or node["mounts"] != expected_mounts:
            raise ComponentGraphError(f"component graph mounts disagree at {node['id']!r}")
        if node["ref"] is None:
            if node["kind"] != "module" or node["lifecycle"] != "opaque":
                raise ComponentGraphError("unregistered component nodes must be opaque modules")
        else:
            if node["kind"] != "component":
                raise ComponentGraphError("registered component nodes must have kind='component'")
            identity = ComponentRef.parse(node["ref"])
            try:
                registration = registry.registration_for_reference(identity.reference)
            except Exception as error:
                raise ComponentGraphError(f"unknown component reference: {identity.reference!r}") from error
            if node["variant"] != registration.variant or node["lifecycle"] != registration.lifecycle:
                raise ComponentGraphError(f"component registration drift at {node['id']!r}")
            if node["config_schema_version"] != registration.config_schema_version or node["state_schema_version"] != registration.state_schema_version:
                raise ComponentGraphError(f"component schema drift at {node['id']!r}")
        if not isinstance(node["dependencies"], list) or any(not isinstance(dep, str) for dep in node["dependencies"]):
            raise ComponentGraphError(f"component dependencies are invalid at {node['id']!r}")

    for edge in edges:
        kind = edge.get("kind")
        if kind == "contains":
            if set(edge) != {"kind", "from", "to", "mount"}:
                raise ComponentGraphError("contains edge is invalid")
            if (
                edge["from"] not in node_ids
                or edge["to"] not in node_ids
                or edge["mount"] not in mount_paths
                or mount_nodes[edge["mount"]] != edge["to"]
            ):
                raise ComponentGraphError("contains edge references an unknown node or mount")
        elif kind == "requires":
            if set(edge) != {"kind", "from", "to_ref"} or edge["from"] not in node_ids:
                raise ComponentGraphError("requires edge is invalid")
            ComponentRef.parse(edge["to_ref"])
            if not _known_reference(edge["to_ref"]):
                raise ComponentGraphError(f"requires edge references an unknown component: {edge['to_ref']!r}")
        elif kind == "data":
            if set(edge) != {"kind", "from", "to"} or not isinstance(edge["from"], str) or not isinstance(edge["to"], str):
                raise ComponentGraphError("data edge is invalid")
        else:
            raise ComponentGraphError(f"unsupported component graph edge kind: {kind!r}")
    _validate_no_contains_cycle(value)

    bindings = value["bindings"]
    if not isinstance(bindings, list) or any(not isinstance(item, Mapping) for item in bindings):
        raise ComponentGraphError("component graph bindings must be a list of mappings")
    if any(not isinstance(item.get("kind"), str) or not isinstance(item.get("from"), str) or not isinstance(item.get("to"), str) for item in bindings):
        raise ComponentGraphError("component graph bindings must have string kind, from, and to")
    groups = value["parameter_groups"]
    if not isinstance(groups, list) or any(not isinstance(item, Mapping) for item in groups):
        raise ComponentGraphError("component graph parameter_groups must be a list of mappings")
    group_ids: set[str] = set()
    for group in groups:
        if set(group) != {"id", "kind", "members", "canonical"} or group["kind"] not in {"same_parameter", "same_buffer"}:
            raise ComponentGraphError("component graph parameter group is invalid")
        if not isinstance(group["id"], str) or not group["id"] or group["id"] in group_ids:
            raise ComponentGraphError("component graph parameter group ids must be unique")
        group_ids.add(group["id"])
        if not isinstance(group["members"], list) or len(group["members"]) < 2 or group["canonical"] not in group["members"]:
            raise ComponentGraphError("component graph parameter group members are invalid")
    if not isinstance(value["state_schema_fingerprint"], str) or len(value["state_schema_fingerprint"]) != 64:
        raise ComponentGraphError("component graph state schema fingerprint is invalid")
    expected_closure = component_closure_fingerprint(value)
    if value["closure_fingerprint"] != expected_closure:
        raise ComponentGraphError("component graph closure fingerprint is invalid")
    if model is not None and component_graph(model) != dict(value):
        raise ComponentGraphError("component graph does not match target model")
    return dict(value)


def verify_component_graph(model: nn.Module, expected: Mapping[str, Any]) -> None:
    """Verify a target module has exactly the serialized component graph."""

    validate_component_graph(expected)
    actual = component_graph(model)
    if actual != dict(expected):
        raise ComponentGraphError(
            "arti.st component graph does not match target model; "
            "construct the same nested components and parameter sharing topology"
        )


__all__ = [
    "COMPONENT_GRAPH_FORMAT",
    "COMPONENT_GRAPH_VERSION",
    "ComponentGraphError",
    "component_closure_fingerprint",
    "component_graph",
    "validate_component_graph",
    "verify_component_graph",
]

