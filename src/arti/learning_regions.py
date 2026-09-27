"""Logical learning regions with a batched AdamW execution plan.

The module deliberately separates a region's learning identity from the
physical optimizer invocation.  A plan may contain many logical domains while
executing compatible parameter groups through a small number of foreach AdamW
calls.  It is a training-side companion to :mod:`arti.resource_graph`; it does
not add a second graph executor or alter forward tensor values.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from torch import Tensor, nn

from .credit_boundary import (
    CreditBoundary,
    CreditBoundaryMode,
    CreditStructureChoice,
    CreditStructureDecision,
    PairedCreditUpdate,
)


LEARNING_REGION_SCHEMA_VERSION = 1


class LearningRegionError(ValueError):
    """Raised when a learning-region or optimizer-domain contract is invalid."""


class OptimizerDomainStatus(str, Enum):
    """Whether a logical optimizer domain may update parameters."""

    ACTIVE = "active"
    PAUSED = "paused"
    DORMANT = "dormant"


def _require_identifier(value: str, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise LearningRegionError(f"{field} must be a non-empty string")
    return value


@dataclass(frozen=True)
class LearningRegion:
    """A committed connected component of the credit dependency graph.

    ``learning_sources`` records why the region may receive a learning signal.
    It is deliberately declarative: an empty collection makes an automatically
    created domain dormant instead of manufacturing a local objective.
    """

    region_id: str
    node_ids: tuple[str, ...]
    learning_sources: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_identifier(self.region_id, field="region_id")
        nodes = tuple(self.node_ids)
        if not nodes or len(set(nodes)) != len(nodes):
            raise LearningRegionError("LearningRegion node_ids must be unique and non-empty")
        for node_id in nodes:
            _require_identifier(node_id, field="LearningRegion node id")
        sources = tuple(self.learning_sources)
        if len(set(sources)) != len(sources):
            raise LearningRegionError("LearningRegion learning_sources must be unique")
        for source in sources:
            _require_identifier(source, field="LearningRegion learning source")
        object.__setattr__(self, "node_ids", tuple(sorted(nodes)))
        object.__setattr__(self, "learning_sources", tuple(sorted(sources)))

    @property
    def is_learnable(self) -> bool:
        return bool(self.learning_sources)


@dataclass(frozen=True)
class CreditEdge:
    """A declared credit dependency; sealed edges do not connect regions."""

    edge_id: str
    source_node_id: str
    target_node_id: str
    sealed: bool = False

    def __post_init__(self) -> None:
        _require_identifier(self.edge_id, field="edge_id")
        _require_identifier(self.source_node_id, field="source_node_id")
        _require_identifier(self.target_node_id, field="target_node_id")
        if self.source_node_id == self.target_node_id:
            raise LearningRegionError("CreditEdge endpoints must be distinct")


def derive_learning_regions(
    node_ids: Iterable[str],
    edges: Iterable[CreditEdge],
    *,
    learning_sources: Mapping[str, Iterable[str]] | None = None,
    region_prefix: str = "region",
) -> tuple[LearningRegion, ...]:
    """Derive weak components after removing committed sealed credit edges.

    This intentionally preserves bypasses.  A sealed ``A -> B`` edge does not
    claim to split ``A`` and ``B`` when another unsealed path joins them.
    """

    _require_identifier(region_prefix, field="region_prefix")
    nodes = tuple(sorted(set(node_ids)))
    if not nodes:
        raise LearningRegionError("derive_learning_regions requires at least one node")
    for node_id in nodes:
        _require_identifier(node_id, field="node_id")
    known = set(nodes)
    adjacency: dict[str, set[str]] = {node_id: set() for node_id in nodes}
    edge_ids: set[str] = set()
    for edge in edges:
        if not isinstance(edge, CreditEdge):
            raise TypeError("edges must contain CreditEdge values")
        if edge.edge_id in edge_ids:
            raise LearningRegionError("CreditEdge edge_ids must be unique")
        edge_ids.add(edge.edge_id)
        if edge.source_node_id not in known or edge.target_node_id not in known:
            raise LearningRegionError("CreditEdge endpoint is not a known node")
        if not edge.sealed:
            adjacency[edge.source_node_id].add(edge.target_node_id)
            adjacency[edge.target_node_id].add(edge.source_node_id)
    source_map = {} if learning_sources is None else {
        region_node: tuple(sources) for region_node, sources in learning_sources.items()
    }
    if set(source_map).difference(known):
        raise LearningRegionError("learning_sources references an unknown node")

    result: list[LearningRegion] = []
    unseen = set(nodes)
    while unseen:
        root = min(unseen)
        stack = [root]
        component: set[str] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            unseen.discard(node)
            stack.extend(sorted(adjacency[node].difference(component), reverse=True))
        component_nodes = tuple(sorted(component))
        component_sources = tuple(
            sorted({source for node in component_nodes for source in source_map.get(node, ())})
        )
        result.append(
            LearningRegion(
                region_id=f"{region_prefix}-{len(result):04d}",
                node_ids=component_nodes,
                learning_sources=component_sources,
            )
        )
    return tuple(result)


def program_graph_credit_edges(graph: object) -> tuple[CreditEdge, ...]:
    """Project a declared ``ProgramGraph`` into credit dependency edges.

    The adapter intentionally uses only graph declarations already exposed by
    ``ProgramGraph``: connection resource reads/writes, node ports and explicit
    connection dependencies.  It does not inspect a runtime trace, create a
    scheduler, or infer an unobserved edge from tensor values.
    """

    connections = getattr(graph, "connections", None)
    nodes = getattr(graph, "nodes", None)
    if not isinstance(connections, nn.ModuleDict) or not isinstance(nodes, nn.ModuleDict):
        raise TypeError("program_graph_credit_edges expects an arti ProgramGraph")

    inputs: dict[str, set[str]] = {}
    outputs: dict[str, set[str]] = {}
    dependencies: list[tuple[str, str]] = []
    for connection_id, connection in connections.items():
        source = getattr(getattr(connection, "source", None), "resource_id", None)
        destination = getattr(getattr(connection, "destination", None), "resource_id", None)
        if not isinstance(source, str) or not isinstance(destination, str):
            raise LearningRegionError("ProgramGraph connection ports must have resource ids")
        connection_inputs = {source}
        operand_views = getattr(connection, "operand_views", {})
        if not isinstance(operand_views, Mapping):
            raise LearningRegionError("ProgramGraph connection operand_views must be a mapping")
        for view in operand_views.values():
            resource_id = getattr(view, "resource_id", None)
            if not isinstance(resource_id, str):
                raise LearningRegionError("ProgramGraph operand view must have a resource id")
            connection_inputs.add(resource_id)
        inputs[connection_id] = connection_inputs
        outputs[connection_id] = {destination}
        for dependency in getattr(connection, "depends_on", ()):
            if not isinstance(dependency, str):
                raise LearningRegionError("ProgramGraph connection dependency must be an id")
            dependencies.append((dependency, connection_id))
    for node_id, node in nodes.items():
        if hasattr(node, "input_resource_id") and hasattr(node, "output_resource_id"):
            node_inputs = {node.input_resource_id}
            node_outputs = {node.output_resource_id}
        else:
            input_ports = getattr(node, "input_ports", None)
            output_ports = getattr(node, "output_ports", None)
            if not isinstance(input_ports, Mapping) or not isinstance(output_ports, Mapping):
                raise LearningRegionError("ProgramGraph node does not expose declared ports")
            node_inputs = {port.resource_id for port in input_ports.values()}
            node_outputs = {port.resource_id for port in output_ports.values()}
        if not all(isinstance(resource_id, str) for resource_id in (*node_inputs, *node_outputs)):
            raise LearningRegionError("ProgramGraph node ports must have resource ids")
        inputs[node_id] = node_inputs
        outputs[node_id] = node_outputs

    producers: dict[str, set[str]] = defaultdict(set)
    for step_id, resource_ids in outputs.items():
        for resource_id in resource_ids:
            producers[resource_id].add(step_id)
    edge_pairs: set[tuple[str, str, str]] = set()
    for consumer_id, resource_ids in inputs.items():
        for resource_id in resource_ids:
            for producer_id in producers.get(resource_id, ()):
                if producer_id != consumer_id:
                    edge_pairs.add((producer_id, consumer_id, resource_id))
    for producer_id, consumer_id in dependencies:
        if producer_id not in inputs or consumer_id not in inputs:
            raise LearningRegionError("ProgramGraph connection dependency references an unknown step")
        edge_pairs.add((producer_id, consumer_id, "dependency"))
    def closed_boundary(connection: object) -> bool:
        mode = getattr(getattr(connection, "credit_boundary", None), "mode", None)
        return getattr(mode, "value", mode) == "closed"

    sealed_consumers = {
        connection_id for connection_id, connection in connections.items() if closed_boundary(connection)
    }
    return tuple(
        CreditEdge(
            edge_id=f"credit:{producer_id}->{consumer_id}:{resource_id}",
            source_node_id=producer_id,
            target_node_id=consumer_id,
            sealed=consumer_id in sealed_consumers,
        )
        for producer_id, consumer_id, resource_id in sorted(edge_pairs)
    )


def derive_learning_regions_from_program_graph(
    graph: object,
    *,
    sealed_credit_edge_ids: Iterable[str] = (),
    learning_sources: Mapping[str, Iterable[str]] | None = None,
    region_prefix: str = "region",
) -> tuple[LearningRegion, ...]:
    """Derive committed regions directly from a declared ``ProgramGraph``.

    ``sealed_credit_edge_ids`` uses the stable ids returned by
    :func:`program_graph_credit_edges`.  The graph itself remains untouched;
    sealing affects learning-region formation only.
    """

    sealed = set(sealed_credit_edge_ids)
    if any(not isinstance(edge_id, str) or not edge_id for edge_id in sealed):
        raise LearningRegionError("sealed_credit_edge_ids must contain non-empty strings")
    edges = program_graph_credit_edges(graph)
    known_edge_ids = {edge.edge_id for edge in edges}
    unknown = sealed.difference(known_edge_ids)
    if unknown:
        raise LearningRegionError(f"sealed_credit_edge_ids are not graph edges: {sorted(unknown)!r}")
    connections = getattr(graph, "connections", {})
    nodes = getattr(graph, "nodes", {})
    node_ids = tuple(sorted(set(connections).union(nodes)))
    return derive_learning_regions(
        node_ids,
        tuple(
            CreditEdge(
                edge.edge_id,
                edge.source_node_id,
                edge.target_node_id,
                sealed=edge.sealed or edge.edge_id in sealed,
            )
            for edge in edges
        ),
        learning_sources=learning_sources,
        region_prefix=region_prefix,
    )


@dataclass(frozen=True)
class ParameterUse:
    """One stable parameter identity used by one learning region."""

    parameter_id: str
    region_id: str
    parameter: nn.Parameter

    def __post_init__(self) -> None:
        _require_identifier(self.parameter_id, field="parameter_id")
        _require_identifier(self.region_id, field="region_id")
        if not isinstance(self.parameter, nn.Parameter):
            raise TypeError("ParameterUse parameter must be torch.nn.Parameter")


@dataclass(frozen=True)
class ProgramParameterBinding:
    """Declare that a stable parameter belongs to one ProgramGraph step.

    A shared parameter has more than one binding with the same
    ``parameter_id``.  The ownership table later assigns it one shared domain
    instead of updating it once for every use site.
    """

    parameter_id: str
    node_id: str
    parameter: nn.Parameter

    def __post_init__(self) -> None:
        _require_identifier(self.parameter_id, field="parameter_id")
        _require_identifier(self.node_id, field="node_id")
        if not isinstance(self.parameter, nn.Parameter):
            raise TypeError("ProgramParameterBinding parameter must be torch.nn.Parameter")


@dataclass(frozen=True)
class ParameterOwnership:
    """The single optimizer domain authorized to update one parameter."""

    parameter_id: str
    owner_domain_id: str
    use_region_ids: tuple[str, ...]
    shared: bool

    def __post_init__(self) -> None:
        _require_identifier(self.parameter_id, field="parameter_id")
        _require_identifier(self.owner_domain_id, field="owner_domain_id")
        regions = tuple(sorted(set(self.use_region_ids)))
        if not regions:
            raise LearningRegionError("ParameterOwnership use_region_ids must be non-empty")
        for region_id in regions:
            _require_identifier(region_id, field="ParameterOwnership use region")
        if self.shared != (len(regions) > 1):
            raise LearningRegionError("ParameterOwnership shared must match the number of use regions")
        object.__setattr__(self, "use_region_ids", regions)


@dataclass(frozen=True)
class ParameterOwnershipTable:
    """Stable parameter ownership independent of Python optimizer instances."""

    entries: tuple[ParameterOwnership, ...]
    shared_domain_id: str = "shared"

    def __post_init__(self) -> None:
        _require_identifier(self.shared_domain_id, field="shared_domain_id")
        entries = tuple(self.entries)
        parameter_ids = [entry.parameter_id for entry in entries]
        if len(set(parameter_ids)) != len(parameter_ids):
            raise LearningRegionError("ParameterOwnershipTable parameter ids must be unique")
        if any(not isinstance(entry, ParameterOwnership) for entry in entries):
            raise TypeError("entries must contain ParameterOwnership values")
        object.__setattr__(self, "entries", tuple(sorted(entries, key=lambda entry: entry.parameter_id)))

    @classmethod
    def from_uses(
        cls,
        uses: Iterable[ParameterUse],
        *,
        shared_domain_id: str = "shared",
    ) -> "ParameterOwnershipTable":
        grouped: dict[str, list[ParameterUse]] = defaultdict(list)
        parameter_objects: dict[str, nn.Parameter] = {}
        for use in uses:
            if not isinstance(use, ParameterUse):
                raise TypeError("uses must contain ParameterUse values")
            existing = parameter_objects.setdefault(use.parameter_id, use.parameter)
            if existing is not use.parameter:
                raise LearningRegionError(
                    f"parameter_id {use.parameter_id!r} refers to more than one parameter object"
                )
            grouped[use.parameter_id].append(use)
        if not grouped:
            raise LearningRegionError("ParameterOwnershipTable requires at least one ParameterUse")
        entries: list[ParameterOwnership] = []
        for parameter_id, parameter_uses in grouped.items():
            region_ids = tuple(sorted({use.region_id for use in parameter_uses}))
            shared = len(region_ids) > 1
            entries.append(
                ParameterOwnership(
                    parameter_id=parameter_id,
                    owner_domain_id=shared_domain_id if shared else region_ids[0],
                    use_region_ids=region_ids,
                    shared=shared,
                )
            )
        return cls(tuple(entries), shared_domain_id=shared_domain_id)

    def owner_for(self, parameter_id: str) -> ParameterOwnership:
        for entry in self.entries:
            if entry.parameter_id == parameter_id:
                return entry
        raise LearningRegionError(f"unknown parameter_id: {parameter_id!r}")

    def as_dict(self) -> dict[str, ParameterOwnership]:
        return {entry.parameter_id: entry for entry in self.entries}


def parameter_uses_from_program_graph(
    graph: object,
    bindings: Iterable[ProgramParameterBinding],
    *,
    sealed_credit_edge_ids: Iterable[str] = (),
    learning_sources: Mapping[str, Iterable[str]] | None = None,
    region_prefix: str = "region",
) -> tuple[tuple[LearningRegion, ...], tuple[ParameterUse, ...]]:
    """Project declared graph ownership into committed learning regions.

    This reads only declared graph edges and discrete committed boundary modes,
    never a batch-local forward trace.
    """

    regions = derive_learning_regions_from_program_graph(
        graph,
        sealed_credit_edge_ids=sealed_credit_edge_ids,
        learning_sources=learning_sources,
        region_prefix=region_prefix,
    )
    region_by_node = {
        node_id: region.region_id for region in regions for node_id in region.node_ids
    }
    declared = tuple(bindings)
    if not declared:
        raise LearningRegionError("at least one ProgramParameterBinding is required")
    uses: list[ParameterUse] = []
    for binding in declared:
        if not isinstance(binding, ProgramParameterBinding):
            raise TypeError("bindings must contain ProgramParameterBinding values")
        try:
            region_id = region_by_node[binding.node_id]
        except KeyError as error:
            raise LearningRegionError(
                f"parameter binding node {binding.node_id!r} is not a ProgramGraph step"
            ) from error
        uses.append(ParameterUse(binding.parameter_id, region_id, binding.parameter))
    return regions, tuple(uses)


def optimizer_domains_from_learning_regions(
    regions: Iterable[LearningRegion],
    ownership: ParameterOwnershipTable,
    *,
    learning_rate: float,
    beta1: float = 0.9,
    beta2: float = 0.999,
    epsilon: float = 1e-8,
    weight_decay: float = 0.0,
    cadence: int = 1,
    domain_overrides: Mapping[str, "OptimizerDomain"] | None = None,
) -> tuple[OptimizerDomain, ...]:
    """Create logical domains while preserving physical foreach grouping.

    ``domain_overrides`` is a control-plane mapping for a freshly committed
    topology.  It lets new regions receive distinct AdamW configurations
    without constructing Python optimizer objects in the training hot path.
    Every expected region (and ``shared`` when present) must be represented;
    regions without a declared learning source must remain ``DORMANT``.
    """

    region_values = tuple(regions)
    if not region_values or any(not isinstance(region, LearningRegion) for region in region_values):
        raise TypeError("regions must be a non-empty sequence of LearningRegion values")
    by_id = {region.region_id: region for region in region_values}
    if len(by_id) != len(region_values):
        raise LearningRegionError("learning region ids must be unique")

    def domain(domain_id: str, *, active: bool) -> OptimizerDomain:
        return OptimizerDomain(
            domain_id,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            weight_decay=weight_decay,
            cadence=cadence,
            status=OptimizerDomainStatus.ACTIVE if active else OptimizerDomainStatus.DORMANT,
        )

    region_activity = {
        region.region_id: region.is_learnable
        for region in sorted(region_values, key=lambda item: item.region_id)
    }
    shared_entries = [entry for entry in ownership.entries if entry.shared]
    shared_active = False
    if shared_entries:
        shared_region_ids = {region_id for entry in shared_entries for region_id in entry.use_region_ids}
        try:
            shared_active = any(by_id[region_id].is_learnable for region_id in shared_region_ids)
        except KeyError as error:
            raise LearningRegionError("ownership refers to an unknown learning region") from error
    expected_activity = {
        **region_activity,
        **({ownership.shared_domain_id: shared_active} if shared_entries else {}),
    }
    if domain_overrides is None:
        return tuple(
            domain(domain_id, active=active)
            for domain_id, active in sorted(expected_activity.items())
        )
    overrides = dict(domain_overrides)
    if set(overrides) != set(expected_activity):
        missing = sorted(set(expected_activity).difference(overrides))
        unknown = sorted(set(overrides).difference(expected_activity))
        raise LearningRegionError(
            f"domain_overrides must cover exactly the committed domains; missing={missing!r}, unknown={unknown!r}"
        )
    for domain_id, active in expected_activity.items():
        override = overrides[domain_id]
        if not isinstance(override, OptimizerDomain):
            raise TypeError("domain_overrides must map ids to OptimizerDomain values")
        if override.domain_id != domain_id:
            raise LearningRegionError("domain_overrides keys must match OptimizerDomain ids")
        if not active and override.status is OptimizerDomainStatus.ACTIVE:
            raise LearningRegionError(
                f"domain {domain_id!r} has no learning source and must be dormant or paused"
            )
    return tuple(overrides[domain_id] for domain_id in sorted(expected_activity))


@dataclass(frozen=True)
class OptimizerDomain:
    """Logical AdamW configuration for one learning or shared domain."""

    domain_id: str
    learning_rate: float
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    weight_decay: float = 0.0
    cadence: int = 1
    status: OptimizerDomainStatus = OptimizerDomainStatus.ACTIVE

    def __post_init__(self) -> None:
        _require_identifier(self.domain_id, field="domain_id")
        if self.learning_rate <= 0:
            raise LearningRegionError("learning_rate must be positive")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise LearningRegionError("AdamW betas must be in [0, 1)")
        if self.epsilon <= 0:
            raise LearningRegionError("epsilon must be positive")
        if self.weight_decay < 0:
            raise LearningRegionError("weight_decay must be non-negative")
        if type(self.cadence) is not int or self.cadence <= 0:
            raise LearningRegionError("cadence must be a positive integer")
        if not isinstance(self.status, OptimizerDomainStatus):
            raise TypeError("status must be OptimizerDomainStatus")

    @property
    def is_active(self) -> bool:
        return self.status is OptimizerDomainStatus.ACTIVE


@dataclass
class OptimizerStateEntry:
    """Per-parameter AdamW moments stored independently of a domain object."""

    exp_avg: Tensor
    exp_avg_sq: Tensor
    parameter_step: Tensor


@dataclass(frozen=True)
class FunctionalOptimizerTrial:
    """A same-snapshot AdamW proposal with no mutation of the live plan."""

    parameters: Mapping[str, Tensor]
    state_entries: Mapping[str, OptimizerStateEntry]
    domain_steps: Mapping[str, Tensor]
    updated_parameter_ids: tuple[str, ...]
    skipped_parameter_ids: tuple[str, ...]
    advanced_domain_ids: tuple[str, ...]


class OptimizerStateArena:
    """Lazily allocated, stable-identity optimizer state for AdamW domains."""

    def __init__(self, *, state_dtype: torch.dtype | None = None) -> None:
        self.state_dtype = state_dtype
        self._entries: dict[str, OptimizerStateEntry] = {}
        self._domain_steps: dict[str, Tensor] = {}

    def entry_for(self, parameter_id: str, parameter: nn.Parameter) -> OptimizerStateEntry:
        entry = self._entries.get(parameter_id)
        if entry is not None:
            if entry.exp_avg.shape != parameter.shape:
                raise LearningRegionError(f"optimizer state shape changed for {parameter_id!r}")
            if entry.exp_avg.device != parameter.device:
                raise LearningRegionError(f"optimizer state device changed for {parameter_id!r}")
            return entry
        state_dtype = parameter.dtype if self.state_dtype is None else self.state_dtype
        if not torch.empty((), dtype=state_dtype).is_floating_point():
            raise LearningRegionError("OptimizerStateArena state_dtype must be floating point")
        entry = OptimizerStateEntry(
            exp_avg=torch.zeros_like(parameter, memory_format=torch.preserve_format, dtype=state_dtype),
            exp_avg_sq=torch.zeros_like(parameter, memory_format=torch.preserve_format, dtype=state_dtype),
            parameter_step=torch.zeros((), dtype=torch.float32, device=parameter.device),
        )
        self._entries[parameter_id] = entry
        return entry

    def domain_step_for(self, domain_id: str, *, device: torch.device) -> Tensor:
        _require_identifier(domain_id, field="domain_id")
        step = self._domain_steps.get(domain_id)
        if step is not None:
            if step.device != device:
                raise LearningRegionError(f"domain state device changed for {domain_id!r}")
            return step
        step = torch.zeros((), dtype=torch.int64, device=device)
        self._domain_steps[domain_id] = step
        return step

    def select_for(
        self, parameter_ids: Iterable[str], domain_ids: Iterable[str],
    ) -> "OptimizerStateArena":
        """Make a new topology's state index without moving retained GPU tensors."""

        selected = OptimizerStateArena(state_dtype=self.state_dtype)
        keep_parameters = set(parameter_ids)
        keep_domains = set(domain_ids)
        selected._entries = {
            parameter_id: entry for parameter_id, entry in self._entries.items()
            if parameter_id in keep_parameters
        }
        selected._domain_steps = {
            domain_id: step for domain_id, step in self._domain_steps.items()
            if domain_id in keep_domains
        }
        return selected

    def state_dict(self) -> dict[str, Any]:
        """Return a tensor-preserving checkpoint payload for the optimizer state."""

        return {
            "schema_version": LEARNING_REGION_SCHEMA_VERSION,
            "state_dtype": None if self.state_dtype is None else str(self.state_dtype),
            "entries": {
                parameter_id: {
                    "exp_avg": entry.exp_avg.detach().clone(),
                    "exp_avg_sq": entry.exp_avg_sq.detach().clone(),
                    "parameter_step": entry.parameter_step.detach().clone(),
                }
                for parameter_id, entry in self._entries.items()
            },
            "domain_steps": {
                domain_id: step.detach().clone() for domain_id, step in self._domain_steps.items()
            },
        }

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        parameters: Mapping[str, nn.Parameter],
    ) -> None:
        if state_dict.get("schema_version") != LEARNING_REGION_SCHEMA_VERSION:
            raise LearningRegionError("unsupported OptimizerStateArena schema version")
        entries = state_dict.get("entries")
        domain_steps = state_dict.get("domain_steps")
        if not isinstance(entries, Mapping) or not isinstance(domain_steps, Mapping):
            raise LearningRegionError("invalid OptimizerStateArena state payload")
        unknown = set(entries).difference(parameters)
        if unknown:
            raise LearningRegionError(f"optimizer state has unknown parameter ids: {sorted(unknown)!r}")
        restored: dict[str, OptimizerStateEntry] = {}
        for parameter_id, raw_entry in entries.items():
            if not isinstance(raw_entry, Mapping):
                raise LearningRegionError("optimizer state entry must be a mapping")
            parameter = parameters[parameter_id]
            try:
                exp_avg = raw_entry["exp_avg"].to(device=parameter.device)
                exp_avg_sq = raw_entry["exp_avg_sq"].to(device=parameter.device)
                parameter_step = raw_entry["parameter_step"].to(device=parameter.device)
            except (AttributeError, KeyError) as error:
                raise LearningRegionError("optimizer state entry is incomplete") from error
            if exp_avg.shape != parameter.shape or exp_avg_sq.shape != parameter.shape:
                raise LearningRegionError(f"optimizer state shape mismatch for {parameter_id!r}")
            if parameter_step.shape != torch.Size([]):
                raise LearningRegionError("parameter_step must be scalar")
            restored[parameter_id] = OptimizerStateEntry(exp_avg, exp_avg_sq, parameter_step)
        self._entries = restored
        self._domain_steps = {
            str(domain_id): step.detach().clone()
            for domain_id, step in domain_steps.items()
            if isinstance(step, Tensor) and step.ndim == 0
        }
        if len(self._domain_steps) != len(domain_steps):
            raise LearningRegionError("domain_steps must contain scalar tensors")


@dataclass(frozen=True)
class OptimizerBucketReceipt:
    """One physical grouped AdamW invocation."""

    domain_ids: tuple[str, ...]
    parameter_ids: tuple[str, ...]
    device: str
    parameter_dtype: str


@dataclass(frozen=True)
class OptimizerStepReceipt:
    """A synchronized optimizer commit, including skipped parameters/domains."""

    transaction_index: int
    updated_parameter_ids: tuple[str, ...]
    skipped_parameter_ids: tuple[str, ...]
    advanced_domain_ids: tuple[str, ...]
    buckets: tuple[OptimizerBucketReceipt, ...]


@dataclass(frozen=True)
class StructureCommitReceipt:
    """Metadata-only evidence for a new logical optimizer topology."""

    previous_generation: int
    generation: int
    retained_parameter_ids: tuple[str, ...]
    added_parameter_ids: tuple[str, ...]
    removed_parameter_ids: tuple[str, ...]
    previous_owner_domain_ids: tuple[str, ...]
    owner_domain_ids: tuple[str, ...]


@dataclass(frozen=True)
class CreditStructureCandidate:
    """One direct-versus-boundary choice resolved at a structure window.

    ``paired_update`` must come from two support updates evaluated from the
    same live parameter snapshot.  This object does not run those trials: it
    only binds their finished query outcomes to one declared graph connection.
    """

    connection_id: str
    boundary: CreditBoundary
    choice: CreditStructureChoice
    paired_update: PairedCreditUpdate
    boundary_mode: CreditBoundaryMode = CreditBoundaryMode.MEAN
    seal_on_boundary: bool = False
    tolerance: float = 1e-6
    retain_declared_boundary_on_tie: bool = False

    def __post_init__(self) -> None:
        _require_identifier(self.connection_id, field="CreditStructureCandidate connection_id")
        if not isinstance(self.boundary, CreditBoundary):
            raise TypeError("CreditStructureCandidate boundary must be CreditBoundary")
        if not isinstance(self.choice, CreditStructureChoice):
            raise TypeError("CreditStructureCandidate choice must be CreditStructureChoice")
        if not isinstance(self.paired_update, PairedCreditUpdate):
            raise TypeError("CreditStructureCandidate paired_update must be PairedCreditUpdate")
        if not isinstance(self.boundary_mode, CreditBoundaryMode):
            raise TypeError("CreditStructureCandidate boundary_mode must be CreditBoundaryMode")
        if self.boundary_mode is CreditBoundaryMode.OPEN:
            raise LearningRegionError("boundary candidate mode must not be open")
        if self.seal_on_boundary and self.boundary_mode is not CreditBoundaryMode.CLOSED:
            raise LearningRegionError("sealing a structure candidate requires closed boundary mode")
        if self.tolerance < 0.0:
            raise LearningRegionError("CreditStructureCandidate tolerance must be non-negative")

    def decide(self) -> CreditStructureDecision:
        """Resolve the candidate only from its paired post-update outcomes."""

        return self.paired_update.structure_decision(
            self.choice,
            tolerance=self.tolerance,
            retain_declared_boundary_on_tie=self.retain_declared_boundary_on_tie,
        )


@dataclass(frozen=True)
class CreditStructureWindowReceipt:
    """Evidence for one completed, atomically committed structure window."""

    structure_commit: StructureCommitReceipt
    decisions: tuple[tuple[str, CreditStructureDecision], ...]
    selected_boundary_connection_ids: tuple[str, ...]
    sealed_connection_ids: tuple[str, ...]


class PreparedOptimizerStep(nn.Module):
    """Fixed-layout, tensor-controlled AdamW commit for one optimizer plan.

    Construction is control-plane work: it binds a single-device plan's
    parameters and optimizer-state arena into a stable argument layout.  Each
    later call accepts only GPU/CPU tensors for the domain update mask,
    gradient-presence mask, and gradients.  It contains no per-domain Python
    dispatch, so callers may lower it with :func:`torch.compile` after the
    learning topology has been committed.

    The prepared step borrows the plan's parameters and state tensors.  Call
    :meth:`synchronize_control_state` at a control-plane boundary before using
    the source plan for a structural rebuild or an eager ``step`` again.
    """

    def __init__(self, plan: "OptimizerExecutionPlan") -> None:
        super().__init__()
        parameter_items = tuple(plan.parameters.items())
        devices = {parameter.device for _, parameter in parameter_items}
        if len(devices) != 1:
            raise LearningRegionError("prepared optimizer step requires parameters on one device")
        self._source_plan = plan
        self.parameter_ids = tuple(parameter_id for parameter_id, _ in parameter_items)
        self.domain_ids = tuple(sorted(plan.domains))
        self.values = nn.ParameterList([parameter for _, parameter in parameter_items])
        self._state_names: list[tuple[str, str, str]] = []
        ownership = plan.ownership.as_dict()
        domain_index = {domain_id: index for index, domain_id in enumerate(self.domain_ids)}
        parameter_domains: list[int] = []
        statuses: list[bool] = []
        cadences: list[int] = []
        learning_rates: list[float] = []
        beta1s: list[float] = []
        beta2s: list[float] = []
        epsilons: list[float] = []
        weight_decays: list[float] = []
        for index, (parameter_id, parameter) in enumerate(parameter_items):
            state = plan.state_arena.entry_for(parameter_id, parameter)
            exp_avg_name = f"_exp_avg_{index}"
            exp_avg_sq_name = f"_exp_avg_sq_{index}"
            parameter_step_name = f"_parameter_step_{index}"
            self.register_buffer(exp_avg_name, state.exp_avg)
            self.register_buffer(exp_avg_sq_name, state.exp_avg_sq)
            self.register_buffer(parameter_step_name, state.parameter_step)
            self._state_names.append((exp_avg_name, exp_avg_sq_name, parameter_step_name))
            domain = plan.domains[ownership[parameter_id].owner_domain_id]
            parameter_domains.append(domain_index[domain.domain_id])
            statuses.append(domain.is_active)
            cadences.append(domain.cadence)
            learning_rates.append(domain.learning_rate)
            beta1s.append(domain.beta1)
            beta2s.append(domain.beta2)
            epsilons.append(domain.epsilon)
            weight_decays.append(domain.weight_decay)
        device = next(iter(devices))
        self.register_buffer("_parameter_domains", torch.tensor(parameter_domains, device=device, dtype=torch.long))
        self._parameter_domain_indices = tuple(parameter_domains)
        self.register_buffer("_active_domains", torch.tensor(statuses, device=device, dtype=torch.bool))
        self.register_buffer("_cadences", torch.tensor(cadences, device=device, dtype=torch.int64))
        self.register_buffer("_learning_rates", torch.tensor(learning_rates, device=device))
        self.register_buffer("_beta1s", torch.tensor(beta1s, device=device))
        self.register_buffer("_beta2s", torch.tensor(beta2s, device=device))
        self.register_buffer("_epsilons", torch.tensor(epsilons, device=device))
        self.register_buffer("_weight_decays", torch.tensor(weight_decays, device=device))
        self._domain_step_names: list[str] = []
        for index, domain_id in enumerate(self.domain_ids):
            name = f"_domain_step_{index}"
            self.register_buffer(name, plan.state_arena.domain_step_for(domain_id, device=device))
            self._domain_step_names.append(name)
        self.register_buffer(
            "_transaction_index",
            torch.tensor(plan.transaction_index, device=device, dtype=torch.int64),
        )

    def forward(
        self,
        update_mask: Tensor,
        gradient_present: Tensor,
        *gradients: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Commit one fixed-layout transaction and return parameter/domain masks.

        ``update_mask`` is boolean ``[domain_count]`` and
        ``gradient_present`` is boolean ``[parameter_count]``.  A false mask
        preserves values, moments, parameter clocks, domain clocks, and AdamW
        weight decay exactly; a numerical zero gradient with a true presence
        mask remains a real update.
        """

        if update_mask.dtype is not torch.bool or update_mask.ndim != 1:
            raise TypeError("update_mask must be a rank-1 boolean Tensor")
        if gradient_present.dtype is not torch.bool or gradient_present.ndim != 1:
            raise TypeError("gradient_present must be a rank-1 boolean Tensor")
        if update_mask.shape[0] != len(self.domain_ids):
            raise LearningRegionError("update_mask does not match prepared optimizer domains")
        if gradient_present.shape[0] != len(self.parameter_ids):
            raise LearningRegionError("gradient_present does not match prepared optimizer parameters")
        if len(gradients) != len(self.parameter_ids):
            raise LearningRegionError("gradients do not match prepared optimizer parameters")
        next_transaction = self._transaction_index + 1
        parameter_updates: list[Tensor] = []
        with torch.no_grad():
            for index, (parameter, gradient) in enumerate(zip(self.values, gradients, strict=True)):
                if gradient.shape != parameter.shape or gradient.device != parameter.device:
                    raise LearningRegionError(
                        f"prepared gradient for {self.parameter_ids[index]!r} does not match parameter shape or device"
                    )
                if gradient.dtype != parameter.dtype or gradient.is_sparse:
                    raise LearningRegionError(
                        f"prepared gradient for {self.parameter_ids[index]!r} must be dense and match parameter dtype"
                    )
                domain_index = self._parameter_domain_indices[index]
                cadence_due = torch.remainder(next_transaction - 1, self._cadences[index]) == 0
                apply_update = (
                    update_mask[domain_index]
                    & gradient_present[index]
                    & self._active_domains[index]
                    & cadence_due
                )
                exp_avg_name, exp_avg_sq_name, parameter_step_name = self._state_names[index]
                exp_avg = getattr(self, exp_avg_name)
                exp_avg_sq = getattr(self, exp_avg_sq_name)
                parameter_step = getattr(self, parameter_step_name)
                gradient_state = gradient.to(dtype=exp_avg.dtype)
                beta1 = self._beta1s[index].to(dtype=exp_avg.dtype)
                beta2 = self._beta2s[index].to(dtype=exp_avg.dtype)
                next_exp_avg = exp_avg * beta1 + gradient_state * (1.0 - beta1)
                next_exp_avg_sq = exp_avg_sq * beta2 + gradient_state.square() * (1.0 - beta2)
                next_parameter_step = parameter_step + 1
                bias_correction1 = 1.0 - beta1.pow(next_parameter_step)
                bias_correction2 = 1.0 - beta2.pow(next_parameter_step)
                denominator = (next_exp_avg_sq / bias_correction2).sqrt().add(self._epsilons[index])
                update = (next_exp_avg / bias_correction1) / denominator
                candidate = parameter * (1.0 - self._learning_rates[index] * self._weight_decays[index])
                candidate = candidate - self._learning_rates[index] * update.to(dtype=parameter.dtype)
                parameter.copy_(torch.where(apply_update, candidate, parameter))
                exp_avg.copy_(torch.where(apply_update, next_exp_avg, exp_avg))
                exp_avg_sq.copy_(torch.where(apply_update, next_exp_avg_sq, exp_avg_sq))
                parameter_step.copy_(torch.where(apply_update, next_parameter_step, parameter_step))
                parameter_updates.append(apply_update)
            update_tensor = torch.stack(parameter_updates)
            domain_counts = torch.zeros(
                len(self.domain_ids),
                dtype=torch.int64,
                device=update_tensor.device,
            ).scatter_add_(0, self._parameter_domains, update_tensor.to(dtype=torch.int64))
            domain_updates = domain_counts > 0
            for index, name in enumerate(self._domain_step_names):
                domain_step = getattr(self, name)
                domain_step.copy_(domain_step + domain_updates[index].to(dtype=domain_step.dtype))
            self._transaction_index.copy_(next_transaction)
        return update_tensor, domain_updates

    def synchronize_control_state(self) -> None:
        """Publish the prepared transaction count at a non-hot control boundary."""

        self._source_plan._transaction_index = int(self._transaction_index.detach().cpu())


class OptimizerExecutionPlan:
    """Synchronous multi-domain AdamW plan using grouped functional updates.

    Logical domains may differ in configuration.  Parameters sharing a device,
    dtype and AdamW configuration are sent through one functional foreach call.
    The implementation does not create one ``torch.optim.Optimizer`` per
    region; its Python grouping is plan construction work, while the numerical
    update is grouped by physical compatibility.
    """

    def __init__(
        self,
        parameters: Mapping[str, nn.Parameter],
        ownership: ParameterOwnershipTable,
        domains: Sequence[OptimizerDomain],
        *,
        state_arena: OptimizerStateArena | None = None,
        topology_generation: int = 0,
    ) -> None:
        if type(topology_generation) is not int or topology_generation < 0:
            raise LearningRegionError("topology_generation must be a non-negative integer")
        self.parameters = dict(parameters)
        if not self.parameters:
            raise LearningRegionError("OptimizerExecutionPlan requires parameters")
        for parameter_id, parameter in self.parameters.items():
            _require_identifier(parameter_id, field="parameter_id")
            if not isinstance(parameter, nn.Parameter):
                raise TypeError("parameters must map ids to torch.nn.Parameter")
        self.ownership = ownership
        owner_ids = {entry.parameter_id for entry in ownership.entries}
        if set(self.parameters) != owner_ids:
            raise LearningRegionError("parameters and ownership ids must match exactly")
        domain_values = tuple(domains)
        if not domain_values or any(not isinstance(domain, OptimizerDomain) for domain in domain_values):
            raise TypeError("domains must be a non-empty sequence of OptimizerDomain values")
        if len({domain.domain_id for domain in domain_values}) != len(domain_values):
            raise LearningRegionError("optimizer domain ids must be unique")
        self.domains = {domain.domain_id: domain for domain in domain_values}
        missing_domains = {entry.owner_domain_id for entry in ownership.entries}.difference(self.domains)
        if missing_domains:
            raise LearningRegionError(f"ownership refers to missing domains: {sorted(missing_domains)!r}")
        self.state_arena = OptimizerStateArena() if state_arena is None else state_arena
        self.topology_generation = topology_generation
        self._transaction_index = 0

    @property
    def transaction_index(self) -> int:
        return self._transaction_index

    def prepare_static_step(self) -> PreparedOptimizerStep:
        """Bind this committed single-device plan to a compilable step module.

        Topology changes, parameter replacement, or a return to eager
        :meth:`step` require a control-plane boundary and a fresh prepared
        module.  This method performs no optimizer update itself.
        """

        return PreparedOptimizerStep(self)

    @classmethod
    def from_program_graph(
        cls,
        graph: object,
        bindings: Iterable[ProgramParameterBinding],
        *,
        learning_sources: Mapping[str, Iterable[str]] | None = None,
        sealed_credit_edge_ids: Iterable[str] = (),
        region_prefix: str = "region",
        learning_rate: float,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
        weight_decay: float = 0.0,
        cadence: int = 1,
        domain_overrides: Mapping[str, OptimizerDomain] | None = None,
        state_arena: OptimizerStateArena | None = None,
        topology_generation: int = 0,
    ) -> "OptimizerExecutionPlan":
        """Create logical optimizer domains from a committed ProgramGraph view."""

        regions, uses = parameter_uses_from_program_graph(
            graph,
            bindings,
            sealed_credit_edge_ids=sealed_credit_edge_ids,
            learning_sources=learning_sources,
            region_prefix=region_prefix,
        )
        ownership = ParameterOwnershipTable.from_uses(uses)
        domains = optimizer_domains_from_learning_regions(
            regions,
            ownership,
            learning_rate=learning_rate,
            beta1=beta1,
            beta2=beta2,
            epsilon=epsilon,
            weight_decay=weight_decay,
            cadence=cadence,
            domain_overrides=domain_overrides,
        )
        parameters: dict[str, nn.Parameter] = {}
        for use in uses:
            existing = parameters.setdefault(use.parameter_id, use.parameter)
            if existing is not use.parameter:
                raise LearningRegionError(
                    f"parameter_id {use.parameter_id!r} refers to more than one parameter object"
                )
        return cls(
            parameters,
            ownership,
            domains,
            state_arena=state_arena,
            topology_generation=topology_generation,
        )

    def state_dict(self) -> dict[str, Any]:
        arena = self.state_arena.state_dict()
        arena["entries"] = {
            parameter_id: entry for parameter_id, entry in arena["entries"].items()
            if parameter_id in self.parameters
        }
        arena["domain_steps"] = {
            domain_id: step for domain_id, step in arena["domain_steps"].items()
            if domain_id in self.domains
        }
        return {
            "schema_version": LEARNING_REGION_SCHEMA_VERSION,
            "topology_generation": self.topology_generation,
            "transaction_index": self._transaction_index,
            "ownership": [
                {
                    "parameter_id": entry.parameter_id,
                    "owner_domain_id": entry.owner_domain_id,
                    "use_region_ids": list(entry.use_region_ids),
                    "shared": entry.shared,
                }
                for entry in self.ownership.entries
            ],
            "domains": [
                {
                    "domain_id": domain.domain_id,
                    "learning_rate": domain.learning_rate,
                    "beta1": domain.beta1,
                    "beta2": domain.beta2,
                    "epsilon": domain.epsilon,
                    "weight_decay": domain.weight_decay,
                    "cadence": domain.cadence,
                    "status": domain.status.value,
                }
                for domain in self.domains.values()
            ],
            "state_arena": arena,
        }

    def load_state_dict(self, state_dict: Mapping[str, Any]) -> None:
        if state_dict.get("schema_version") != LEARNING_REGION_SCHEMA_VERSION:
            raise LearningRegionError("unsupported OptimizerExecutionPlan schema version")
        if state_dict.get("topology_generation") != self.topology_generation:
            raise LearningRegionError("optimizer checkpoint topology generation does not match this plan")
        saved_ownership = state_dict.get("ownership")
        current_ownership = [
            {
                "parameter_id": entry.parameter_id,
                "owner_domain_id": entry.owner_domain_id,
                "use_region_ids": list(entry.use_region_ids),
                "shared": entry.shared,
            }
            for entry in self.ownership.entries
        ]
        if saved_ownership != current_ownership:
            raise LearningRegionError("optimizer checkpoint ownership does not match this plan")
        if state_dict.get("domains") != [
            {
                "domain_id": domain.domain_id,
                "learning_rate": domain.learning_rate,
                "beta1": domain.beta1,
                "beta2": domain.beta2,
                "epsilon": domain.epsilon,
                "weight_decay": domain.weight_decay,
                "cadence": domain.cadence,
                "status": domain.status.value,
            }
            for domain in self.domains.values()
        ]:
            raise LearningRegionError("optimizer checkpoint domain descriptors do not match this plan")
        transaction_index = state_dict.get("transaction_index")
        if type(transaction_index) is not int or transaction_index < 0:
            raise LearningRegionError("optimizer checkpoint transaction_index is invalid")
        self.state_arena.load_state_dict(state_dict["state_arena"], self.parameters)
        self._transaction_index = transaction_index

    @staticmethod
    def _bucket_key(domain: OptimizerDomain, parameter: nn.Parameter) -> tuple[object, ...]:
        return (
            str(parameter.device),
            parameter.dtype,
            domain.learning_rate,
            domain.beta1,
            domain.beta2,
            domain.epsilon,
            domain.weight_decay,
        )

    @staticmethod
    def _normalise_update_mask(
        domains: Mapping[str, OptimizerDomain], update_mask: Mapping[str, bool] | None
    ) -> dict[str, bool]:
        if update_mask is None:
            return {domain_id: True for domain_id in domains}
        unknown = set(update_mask).difference(domains)
        if unknown:
            raise LearningRegionError(f"update_mask has unknown domains: {sorted(unknown)!r}")
        result = {domain_id: bool(update_mask.get(domain_id, True)) for domain_id in domains}
        return result

    def functional_adamw_trial(
        self,
        *,
        gradients: Mapping[str, Tensor | None],
        update_mask: Mapping[str, bool] | None = None,
    ) -> FunctionalOptimizerTrial:
        """Return an AdamW proposal without changing parameters or arena state.

        This is the candidate-side counterpart to :meth:`step`.  It retains
        normal tensor autograd through supplied gradients, while the live arena
        remains the authoritative state until a caller elects one proposal for
        an ordinary committed update.
        """

        if set(gradients) != set(self.parameters):
            raise LearningRegionError("explicit gradients must match parameter ids exactly")
        masks = self._normalise_update_mask(self.domains, update_mask)
        next_transaction = self._transaction_index + 1
        ownership_by_id = self.ownership.as_dict()
        parameters: dict[str, Tensor] = {}
        states = {
            parameter_id: OptimizerStateEntry(
                entry.exp_avg.clone(),
                entry.exp_avg_sq.clone(),
                entry.parameter_step.clone(),
            )
            for parameter_id, entry in self.state_arena._entries.items()
        }
        updated: list[str] = []
        skipped: list[str] = []
        advanced_domains: set[str] = set()
        for parameter_id, parameter in self.parameters.items():
            owner = ownership_by_id[parameter_id]
            domain = self.domains[owner.owner_domain_id]
            gradient = gradients[parameter_id]
            cadence_due = (next_transaction - 1) % domain.cadence == 0
            if (
                not domain.is_active
                or not masks[domain.domain_id]
                or not cadence_due
                or gradient is None
            ):
                parameters[parameter_id] = parameter
                skipped.append(parameter_id)
                continue
            if not isinstance(gradient, Tensor):
                raise TypeError("explicit gradients must contain Tensor or None values")
            if gradient.is_sparse:
                raise LearningRegionError("functional AdamW trial does not support sparse gradients")
            if gradient.shape != parameter.shape or gradient.device != parameter.device:
                raise LearningRegionError(
                    f"explicit gradient for {parameter_id!r} does not match parameter shape or device"
                )
            if gradient.dtype != parameter.dtype:
                raise LearningRegionError(
                    f"explicit gradient for {parameter_id!r} does not match parameter dtype"
                )
            existing = states.get(parameter_id)
            if existing is None:
                state_dtype = parameter.dtype if self.state_arena.state_dtype is None else self.state_arena.state_dtype
                exp_avg = torch.zeros_like(parameter, memory_format=torch.preserve_format, dtype=state_dtype)
                exp_avg_sq = torch.zeros_like(parameter, memory_format=torch.preserve_format, dtype=state_dtype)
                parameter_step = torch.zeros((), dtype=torch.float32, device=parameter.device)
            else:
                exp_avg = existing.exp_avg
                exp_avg_sq = existing.exp_avg_sq
                parameter_step = existing.parameter_step
            gradient_state = gradient.to(dtype=exp_avg.dtype)
            next_exp_avg = exp_avg * domain.beta1 + gradient_state * (1.0 - domain.beta1)
            next_exp_avg_sq = exp_avg_sq * domain.beta2 + gradient_state.square() * (1.0 - domain.beta2)
            next_parameter_step = parameter_step + 1
            bias_correction1 = 1.0 - domain.beta1**next_parameter_step
            bias_correction2 = 1.0 - domain.beta2**next_parameter_step
            denominator = (next_exp_avg_sq / bias_correction2).sqrt().add(domain.epsilon)
            update = (next_exp_avg / bias_correction1) / denominator
            next_parameter = parameter * (1.0 - domain.learning_rate * domain.weight_decay)
            parameters[parameter_id] = next_parameter - domain.learning_rate * update.to(dtype=parameter.dtype)
            states[parameter_id] = OptimizerStateEntry(next_exp_avg, next_exp_avg_sq, next_parameter_step)
            updated.append(parameter_id)
            advanced_domains.add(domain.domain_id)
        domain_steps = {
            domain_id: step.clone() for domain_id, step in self.state_arena._domain_steps.items()
        }
        for domain_id in advanced_domains:
            current = domain_steps.get(domain_id)
            if current is None:
                device = next(
                    parameter.device
                    for parameter_id, parameter in self.parameters.items()
                    if ownership_by_id[parameter_id].owner_domain_id == domain_id
                )
                current = torch.zeros((), dtype=torch.int64, device=device)
            domain_steps[domain_id] = current + 1
        return FunctionalOptimizerTrial(
            parameters=parameters,
            state_entries=states,
            domain_steps=domain_steps,
            updated_parameter_ids=tuple(sorted(updated)),
            skipped_parameter_ids=tuple(sorted(skipped)),
            advanced_domain_ids=tuple(sorted(advanced_domains)),
        )

    def gradients_from_losses(
        self,
        task_loss: Tensor | None = None,
        *,
        local_objectives: Mapping[str, Tensor] | None = None,
        create_graph: bool = False,
        retain_graph: bool = False,
        vjp_batch_size: int = 8,
    ) -> dict[str, Tensor | None]:
        """Credit each owner from its region loss and an optional common task loss.

        The batched VJP evaluates all objectives against one parameter snapshot.
        A shared owner receives the sum of objectives from its use regions;
        an omitted local objective does not create an optimizer update.
        """

        objectives = {} if local_objectives is None else dict(local_objectives)
        known_regions = {
            region_id for owner in self.ownership.entries for region_id in owner.use_region_ids
        }
        unknown = set(objectives).difference(known_regions)
        if unknown:
            raise LearningRegionError(f"local_objectives name unknown regions: {sorted(unknown)!r}")
        if task_loss is None and not objectives:
            raise LearningRegionError("task_loss or local_objectives is required")
        if type(vjp_batch_size) is not int or vjp_batch_size <= 0:
            raise LearningRegionError("vjp_batch_size must be a positive integer")
        losses = ([task_loss] if task_loss is not None else []) + [
            objectives[region_id] for region_id in sorted(objectives)
        ]
        if any(not isinstance(loss, Tensor) or loss.ndim != 0 or not loss.is_floating_point() for loss in losses):
            raise TypeError("each objective must be a scalar floating Tensor")
        if len({loss.device for loss in losses}) != 1:
            raise LearningRegionError("objectives must share a device")
        if any(not loss.requires_grad for loss in losses):
            raise LearningRegionError("each objective must have a gradient path")

        loss_indices = {region_id: index + (task_loss is not None)
                        for index, region_id in enumerate(sorted(objectives))}
        owners = self.ownership.as_dict()
        gradients: dict[str, Tensor | None] = dict.fromkeys(self.parameters)
        selected: list[tuple[str, nn.Parameter, tuple[int, ...]]] = []
        for parameter_id, parameter in self.parameters.items():
            owner = owners[parameter_id]
            if not parameter.requires_grad or not self.domains[owner.owner_domain_id].is_active:
                continue
            indices = ((0,) if task_loss is not None else ()) + tuple(
                loss_indices[region_id] for region_id in owner.use_region_ids
                if region_id in loss_indices
            )
            if indices:
                selected.append((parameter_id, parameter, indices))
        if not selected:
            return gradients

        for start in range(0, len(losses), vjp_batch_size):
            end = min(start + vjp_batch_size, len(losses))
            chunk = [
                (parameter_id, parameter, tuple(index - start for index in indices if start <= index < end))
                for parameter_id, parameter, indices in selected
            ]
            chunk = [entry for entry in chunk if entry[2]]
            if not chunk:
                continue
            chunk_losses = losses[start:end]
            if len(chunk_losses) == 1:
                computed = torch.autograd.grad(
                    chunk_losses[0], tuple(parameter for _, parameter, _ in chunk),
                    allow_unused=True, create_graph=create_graph,
                    retain_graph=retain_graph or create_graph or end < len(losses),
                )
            else:
                stacked = torch.stack(chunk_losses)
                computed = torch.autograd.grad(
                    stacked, tuple(parameter for _, parameter, _ in chunk),
                    grad_outputs=torch.eye(len(chunk_losses), device=stacked.device, dtype=stacked.dtype),
                    is_grads_batched=True, allow_unused=True, create_graph=create_graph,
                    retain_graph=retain_graph or create_graph or end < len(losses),
                )
            for (parameter_id, _, indices), gradient in zip(chunk, computed, strict=True):
                if gradient is None:
                    continue
                contribution = gradient if len(chunk_losses) == 1 else gradient[list(indices)].sum(dim=0)
                previous = gradients[parameter_id]
                gradients[parameter_id] = contribution if previous is None else previous + contribution
        return gradients

    def step_with_losses(
        self,
        task_loss: Tensor | None = None,
        *,
        local_objectives: Mapping[str, Tensor] | None = None,
        update_mask: Mapping[str, bool] | None = None,
        vjp_batch_size: int = 8,
    ) -> OptimizerStepReceipt:
        """Compute independent region credit and commit one grouped optimizer step."""

        return self.step(
            gradients=self.gradients_from_losses(
                task_loss, local_objectives=local_objectives, vjp_batch_size=vjp_batch_size,
            ),
            update_mask=update_mask,
        )

    def step(
        self,
        *,
        update_mask: Mapping[str, bool] | None = None,
        gradients: Mapping[str, Tensor | None] | None = None,
    ) -> OptimizerStepReceipt:
        """Apply one same-snapshot AdamW transaction.

        By default gradients are read from parameters. ``gradients`` lets a
        compiled credit rule supply the same stable parameter-id field without
        mutating ``parameter.grad``. ``None`` skips a parameter entirely; a
        numerical zero remains a valid AdamW update.
        """

        masks = self._normalise_update_mask(self.domains, update_mask)
        if gradients is not None and set(gradients) != set(self.parameters):
            raise LearningRegionError("explicit gradients must match parameter ids exactly")
        next_transaction = self._transaction_index + 1
        eligible: dict[tuple[object, ...], list[tuple[str, nn.Parameter, OptimizerDomain, OptimizerStateEntry]]] = defaultdict(list)
        eligible_gradients: dict[tuple[object, ...], list[Tensor]] = defaultdict(list)
        skipped: list[str] = []
        ownership_by_id = self.ownership.as_dict()
        for parameter_id, parameter in self.parameters.items():
            owner = ownership_by_id[parameter_id]
            domain = self.domains[owner.owner_domain_id]
            cadence_due = (next_transaction - 1) % domain.cadence == 0
            gradient = parameter.grad if gradients is None else gradients[parameter_id]
            if (
                not domain.is_active
                or not masks[domain.domain_id]
                or not cadence_due
                or gradient is None
            ):
                skipped.append(parameter_id)
                continue
            if not isinstance(gradient, Tensor):
                raise TypeError("explicit gradients must contain Tensor or None values")
            if gradient.is_sparse:
                raise LearningRegionError("OptimizerExecutionPlan does not support sparse AdamW gradients")
            if gradient.shape != parameter.shape or gradient.device != parameter.device:
                raise LearningRegionError(
                    f"explicit gradient for {parameter_id!r} does not match parameter shape or device"
                )
            if gradient.dtype != parameter.dtype:
                raise LearningRegionError(
                    f"explicit gradient for {parameter_id!r} does not match parameter dtype"
                )
            state = self.state_arena.entry_for(parameter_id, parameter)
            bucket = self._bucket_key(domain, parameter)
            eligible[bucket].append((parameter_id, parameter, domain, state))
            eligible_gradients[bucket].append(gradient)

        buckets: list[OptimizerBucketReceipt] = []
        advanced_domains: set[str] = set()
        updated: list[str] = []
        with torch.no_grad():
            for key, values in eligible.items():
                _, parameter_dtype, lr, beta1, beta2, epsilon, weight_decay = key
                parameters = [value[1] for value in values]
                states = [value[3] for value in values]
                _functional_adamw(
                    parameters,
                    eligible_gradients[key],
                    [state.exp_avg for state in states],
                    [state.exp_avg_sq for state in states],
                    [state.parameter_step for state in states],
                    lr=lr,
                    beta1=beta1,
                    beta2=beta2,
                    epsilon=epsilon,
                    weight_decay=weight_decay,
                )
                parameter_ids = tuple(value[0] for value in values)
                domain_ids = tuple(sorted({value[2].domain_id for value in values}))
                updated.extend(parameter_ids)
                advanced_domains.update(domain_ids)
                buckets.append(
                    OptimizerBucketReceipt(
                        domain_ids=domain_ids,
                        parameter_ids=parameter_ids,
                        device=key[0],
                        parameter_dtype=str(parameter_dtype),
                    )
                )
            for domain_id in sorted(advanced_domains):
                device = next(
                    parameter.device
                    for parameter_id, parameter in self.parameters.items()
                    if ownership_by_id[parameter_id].owner_domain_id == domain_id
                )
                self.state_arena.domain_step_for(domain_id, device=device).add_(1)
        self._transaction_index += 1
        return OptimizerStepReceipt(
            transaction_index=self._transaction_index,
            updated_parameter_ids=tuple(sorted(updated)),
            skipped_parameter_ids=tuple(sorted(skipped)),
            advanced_domain_ids=tuple(sorted(advanced_domains)),
            buckets=tuple(buckets),
        )

    def commit_structure(
        self,
        parameters: Mapping[str, nn.Parameter],
        ownership: ParameterOwnershipTable,
        domains: Sequence[OptimizerDomain],
    ) -> tuple["OptimizerExecutionPlan", StructureCommitReceipt]:
        """Build the next immutable topology plan while retaining compatible state.

        Call this only at an optimizer-window boundary.  Parameter moments
        follow unchanged parameter objects.  A domain clock survives only when
        its members and their parameter objects are unchanged.  This method
        never performs an optimizer update itself.
        """

        next_parameters = dict(parameters)
        old_ids = set(self.parameters)
        next_ids = set(next_parameters)
        retained_ids = {
            parameter_id for parameter_id in old_ids.intersection(next_ids)
            if self.parameters[parameter_id] is next_parameters[parameter_id]
        }
        old_members = {
            domain_id: frozenset(
                entry.parameter_id for entry in self.ownership.entries
                if entry.owner_domain_id == domain_id
            )
            for domain_id in self.domains
        }
        next_members = {
            domain.domain_id: frozenset(
                entry.parameter_id for entry in ownership.entries
                if entry.owner_domain_id == domain.domain_id
            )
            for domain in domains
        }
        retained_domains = {
            domain_id for domain_id, members in next_members.items()
            if members and old_members.get(domain_id) == members and members <= retained_ids
        }
        previous_owners = tuple(sorted({entry.owner_domain_id for entry in self.ownership.entries}))
        next_owners = tuple(sorted({entry.owner_domain_id for entry in ownership.entries}))
        next_plan = OptimizerExecutionPlan(
            next_parameters,
            ownership,
            domains,
            state_arena=self.state_arena.select_for(
                retained_ids, retained_domains,
            ),
            topology_generation=self.topology_generation + 1,
        )
        next_plan._transaction_index = self._transaction_index
        return next_plan, StructureCommitReceipt(
            previous_generation=self.topology_generation,
            generation=next_plan.topology_generation,
            retained_parameter_ids=tuple(sorted(retained_ids)),
            added_parameter_ids=tuple(sorted(next_ids.difference(retained_ids))),
            removed_parameter_ids=tuple(sorted(old_ids.difference(retained_ids))),
            previous_owner_domain_ids=previous_owners,
            owner_domain_ids=next_owners,
        )

    def commit_credit_structure_window(
        self,
        graph: object,
        bindings: Iterable[ProgramParameterBinding],
        candidates: Iterable[CreditStructureCandidate],
        *,
        learning_sources: Mapping[str, Iterable[str]] | None = None,
        sealed_credit_edge_ids: Iterable[str] = (),
        region_prefix: str = "region",
        domain_overrides: Mapping[str, OptimizerDomain] | None = None,
    ) -> tuple["OptimizerExecutionPlan", CreditStructureWindowReceipt]:
        """Commit paired credit choices and rebuild logical optimizer domains.

        This is intentionally a training-window control-plane operation.  It
        validates every candidate and resolves every paired query outcome
        before changing a boundary mode, then delegates domain reconstruction
        to :meth:`commit_program_graph_structure`.  It is not a batch-local
        route and it does not rerun support or query computation.
        """

        candidate_values = tuple(candidates)
        if not candidate_values:
            raise LearningRegionError("commit_credit_structure_window requires at least one candidate")
        if any(not isinstance(candidate, CreditStructureCandidate) for candidate in candidate_values):
            raise TypeError("candidates must contain CreditStructureCandidate values")
        candidate_ids = [candidate.connection_id for candidate in candidate_values]
        if len(set(candidate_ids)) != len(candidate_ids):
            raise LearningRegionError("credit structure candidate connection ids must be unique")
        boundaries = [candidate.boundary for candidate in candidate_values]
        if len({id(boundary) for boundary in boundaries}) != len(boundaries):
            raise LearningRegionError("each credit structure candidate requires its own boundary")

        connections = getattr(graph, "connections", None)
        if not isinstance(connections, nn.ModuleDict):
            raise TypeError("commit_credit_structure_window expects an arti ProgramGraph")
        decisions: list[tuple[CreditStructureCandidate, CreditStructureDecision]] = []
        for candidate in candidate_values:
            try:
                connection = connections[candidate.connection_id]
            except KeyError as error:
                raise LearningRegionError(
                    f"credit structure candidate references unknown connection {candidate.connection_id!r}"
                ) from error
            if getattr(connection, "credit_boundary", None) is not candidate.boundary:
                raise LearningRegionError(
                    f"credit structure candidate boundary does not belong to connection {candidate.connection_id!r}"
                )
            decisions.append((candidate, candidate.decide()))

        previous_modes = {candidate.boundary: candidate.boundary.mode for candidate, _ in decisions}
        try:
            for candidate, decision in decisions:
                candidate.boundary.mode = (
                    candidate.boundary_mode if decision.use_boundary else CreditBoundaryMode.OPEN
                )
            next_plan, structure_receipt = self.commit_program_graph_structure(
                graph,
                bindings,
                learning_sources=learning_sources,
                sealed_credit_edge_ids=sealed_credit_edge_ids,
                region_prefix=region_prefix,
                domain_overrides=domain_overrides,
            )
        except Exception:
            for boundary, previous_mode in previous_modes.items():
                boundary.mode = previous_mode
            raise

        return next_plan, CreditStructureWindowReceipt(
            structure_commit=structure_receipt,
            decisions=tuple(
                (candidate.connection_id, decision)
                for candidate, decision in decisions
            ),
            selected_boundary_connection_ids=tuple(
                candidate.connection_id
                for candidate, decision in decisions
                if decision.use_boundary
            ),
            sealed_connection_ids=tuple(
                candidate.connection_id
                for candidate, decision in decisions
                if decision.use_boundary and candidate.seal_on_boundary
            ),
        )

    def commit_program_graph_structure(
        self,
        graph: object,
        bindings: Iterable[ProgramParameterBinding],
        *,
        learning_sources: Mapping[str, Iterable[str]] | None = None,
        sealed_credit_edge_ids: Iterable[str] = (),
        region_prefix: str = "region",
        domain_overrides: Mapping[str, OptimizerDomain] | None = None,
    ) -> tuple["OptimizerExecutionPlan", StructureCommitReceipt]:
        """Commit a graph-derived domain topology at an optimizer-window boundary."""

        current_domains = tuple(self.domains.values())
        if not current_domains:
            raise LearningRegionError("OptimizerExecutionPlan has no domains")
        optimizer_signatures = {
            (
                domain.learning_rate,
                domain.beta1,
                domain.beta2,
                domain.epsilon,
                domain.weight_decay,
                domain.cadence,
            )
            for domain in current_domains
        }
        if domain_overrides is None and len(optimizer_signatures) != 1:
            raise LearningRegionError(
                "commit_program_graph_structure requires domain_overrides when existing domains differ"
            )
        template = current_domains[0]
        candidate = OptimizerExecutionPlan.from_program_graph(
            graph,
            bindings,
            learning_sources=learning_sources,
            sealed_credit_edge_ids=sealed_credit_edge_ids,
            region_prefix=region_prefix,
            learning_rate=template.learning_rate,
            beta1=template.beta1,
            beta2=template.beta2,
            epsilon=template.epsilon,
            weight_decay=template.weight_decay,
            cadence=template.cadence,
            domain_overrides=domain_overrides,
            state_arena=self.state_arena,
            topology_generation=self.topology_generation + 1,
        )
        return self.commit_structure(
            candidate.parameters,
            candidate.ownership,
            tuple(candidate.domains.values()),
        )


def _functional_adamw(
    parameters: list[nn.Parameter],
    grads: list[Tensor],
    exp_avgs: list[Tensor],
    exp_avg_sqs: list[Tensor],
    state_steps: list[Tensor],
    *,
    lr: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    weight_decay: float,
) -> None:
    """Invoke PyTorch's grouped AdamW functional API with a scalar fallback.

    ``torch.optim.adamw.adamw`` is the public functional implementation behind
    ``AdamW.step``.  Older supported Torch versions have a slightly narrower
    signature, hence the small compatibility fallback.  Both paths operate on
    grouped lists, never one optimizer object per learning domain.
    """

    try:
        from torch.optim.adamw import adamw

        adamw(
            parameters,
            grads,
            exp_avgs,
            exp_avg_sqs,
            [],
            state_steps,
            foreach=True,
            capturable=False,
            differentiable=False,
            fused=None,
            grad_scale=None,
            found_inf=None,
            has_complex=False,
            amsgrad=False,
            beta1=beta1,
            beta2=beta2,
            lr=lr,
            weight_decay=weight_decay,
            eps=epsilon,
            maximize=False,
        )
    except (ImportError, TypeError):  # pragma: no cover - old supported Torch fallback
        from torch.optim import _functional

        _functional.adamw(
            parameters,
            grads,
            exp_avgs,
            exp_avg_sqs,
            [],
            state_steps,
            foreach=True,
            capturable=False,
            differentiable=False,
            fused=None,
            grad_scale=None,
            found_inf=None,
            has_complex=False,
            amsgrad=False,
            beta1=beta1,
            beta2=beta2,
            lr=lr,
            weight_decay=weight_decay,
            eps=epsilon,
            maximize=False,
        )


__all__ = [
    "LEARNING_REGION_SCHEMA_VERSION",
    "CreditStructureCandidate",
    "CreditStructureWindowReceipt",
    "CreditEdge",
    "FunctionalOptimizerTrial",
    "LearningRegion",
    "LearningRegionError",
    "OptimizerBucketReceipt",
    "OptimizerDomain",
    "OptimizerDomainStatus",
    "OptimizerExecutionPlan",
    "PreparedOptimizerStep",
    "OptimizerStateArena",
    "OptimizerStateEntry",
    "OptimizerStepReceipt",
    "ProgramParameterBinding",
    "ParameterOwnership",
    "ParameterOwnershipTable",
    "ParameterUse",
    "StructureCommitReceipt",
    "derive_learning_regions",
    "derive_learning_regions_from_program_graph",
    "optimizer_domains_from_learning_regions",
    "parameter_uses_from_program_graph",
    "program_graph_credit_edges",
]
