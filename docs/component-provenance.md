# Component Provenance

ARTI components use canonical references such as `arti/fold@2` and
`arti/pulse@2`. The reference identifies a mechanism contract; the Python
class name and registry aliases are convenience surfaces and are not stored as
substitutes for that identity.

## Provenance Record

Component provenance records:

- the canonical component reference and lifecycle;
- deterministic configuration and its fingerprint;
- parameter/state schema fingerprints;
- declared capabilities;
- ordered component dependencies.

The record describes executable structure, not learned tensor values. Weight
files remain responsible for tensor integrity.

Version 2 provenance normalizes valid version 1 records before validation. It
does not silently reinterpret unknown component references, missing enabled
dependencies, or incompatible configuration fingerprints.

## Capabilities

Capabilities describe where a component may be placed, for example a Pulse
stage or a selective-compute kernel. They do not grant application authority
and do not attach task semantics to masks or tensors.

An enabled stage must declare a registered component with the required
capability. A disabled stage is an identity operation and must not add a
component dependency merely because a default implementation exists in
Python.

## Composition

`component_spec()` returns one component and its direct dependencies.
`component_provenance()` records the reachable, ordered component graph.
Validation fails closed for unknown references, inconsistent fingerprints,
undeclared enabled dependencies, and cycles.

This boundary is especially important for `AdaptivePulse`: the stage manifest
is the execution contract, while Python module composition supplies the
implementations bound to that manifest.

## Recall Identities

`arti/recall@2` identifies globally normalized mixed-route Recall.
`arti/recall@3` adds per-Bank normalization and explicit member asset identity.
`arti/recall@4` identifies independent K-wide candidate refinement and records
`breadth`, `breadth_mode`, and `breadth_aggregation`.

The default `Recall@4` aggregation is `winner`: one complete candidate
trajectory is forwarded. `route_weighted` is an explicit optional mode. Loading
or resolving an older Recall identity never silently enables K-wide execution.
